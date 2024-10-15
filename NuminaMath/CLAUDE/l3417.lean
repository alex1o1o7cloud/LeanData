import Mathlib

namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l3417_341716

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then -x^2 + 2*a*x - 2*a else a*x + 1

theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a ∈ Set.Icc (-2 : ℝ) 0 ∧ a ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l3417_341716


namespace NUMINAMATH_CALUDE_rain_gear_needed_l3417_341796

structure WeatherForecast where
  rain_probability : ℝ
  rain_probability_valid : 0 ≤ rain_probability ∧ rain_probability ≤ 1

def high_possibility (p : ℝ) : Prop := p > 0.5

theorem rain_gear_needed (forecast : WeatherForecast) 
  (h : forecast.rain_probability = 0.95) : 
  high_possibility forecast.rain_probability :=
by
  sorry

#check rain_gear_needed

end NUMINAMATH_CALUDE_rain_gear_needed_l3417_341796


namespace NUMINAMATH_CALUDE_perimeter_ratio_of_similar_squares_l3417_341734

theorem perimeter_ratio_of_similar_squares (s : ℝ) (h : s > 0) : 
  let s1 := s * ((Real.sqrt 5 + 1) / 2)
  let p1 := 4 * s1
  let p2 := 4 * s
  let diagonal_first := Real.sqrt (2 * s1 ^ 2)
  diagonal_first = s → p1 / p2 = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_of_similar_squares_l3417_341734


namespace NUMINAMATH_CALUDE_tourist_group_size_l3417_341740

theorem tourist_group_size (k : ℕ) : 
  (∃ n : ℕ, n > 0 ∧ 2 * k ≡ 1 [MOD n] ∧ 3 * k ≡ 13 [MOD n]) → 
  (∃ n : ℕ, n = 23 ∧ 2 * k ≡ 1 [MOD n] ∧ 3 * k ≡ 13 [MOD n]) :=
by sorry

end NUMINAMATH_CALUDE_tourist_group_size_l3417_341740


namespace NUMINAMATH_CALUDE_jimmy_action_figures_sale_earnings_l3417_341737

theorem jimmy_action_figures_sale_earnings :
  let regular_figure_count : ℕ := 4
  let special_figure_count : ℕ := 1
  let regular_figure_value : ℕ := 15
  let special_figure_value : ℕ := 20
  let discount : ℕ := 5

  let regular_sale_price : ℕ := regular_figure_value - discount
  let special_sale_price : ℕ := special_figure_value - discount

  let total_earnings : ℕ := regular_figure_count * regular_sale_price + special_figure_count * special_sale_price

  total_earnings = 55 := by sorry

end NUMINAMATH_CALUDE_jimmy_action_figures_sale_earnings_l3417_341737


namespace NUMINAMATH_CALUDE_gardener_work_days_l3417_341791

/-- Calculates the number of days a gardener works on a rose bush replanting project. -/
theorem gardener_work_days
  (num_rose_bushes : ℕ)
  (cost_per_rose_bush : ℚ)
  (gardener_hourly_wage : ℚ)
  (gardener_hours_per_day : ℕ)
  (soil_cubic_feet : ℕ)
  (soil_cost_per_cubic_foot : ℚ)
  (total_project_cost : ℚ)
  (h1 : num_rose_bushes = 20)
  (h2 : cost_per_rose_bush = 150)
  (h3 : gardener_hourly_wage = 30)
  (h4 : gardener_hours_per_day = 5)
  (h5 : soil_cubic_feet = 100)
  (h6 : soil_cost_per_cubic_foot = 5)
  (h7 : total_project_cost = 4100) :
  (total_project_cost - (num_rose_bushes * cost_per_rose_bush + soil_cubic_feet * soil_cost_per_cubic_foot)) / (gardener_hourly_wage * gardener_hours_per_day) = 4 := by
  sorry


end NUMINAMATH_CALUDE_gardener_work_days_l3417_341791


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3417_341739

/-- Proves that the repeating decimal 0.53207207207... is equal to 5316750/999900 -/
theorem repeating_decimal_to_fraction : 
  ∃ (x : ℚ), x = 0.53207207207 ∧ x = 5316750 / 999900 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3417_341739


namespace NUMINAMATH_CALUDE_age_sum_problem_l3417_341741

theorem age_sum_problem (a b c : ℕ+) : 
  b = c → b < a → a * b * c = 144 → a + b + c = 22 := by sorry

end NUMINAMATH_CALUDE_age_sum_problem_l3417_341741


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3417_341793

theorem cubic_equation_roots : 
  {x : ℝ | x^9 + (9/8)*x^6 + (27/64)*x^3 - x + 219/512 = 0} = 
  {1/2, (-1 - Real.sqrt 13)/4, (-1 + Real.sqrt 13)/4} := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3417_341793


namespace NUMINAMATH_CALUDE_negative_two_is_square_root_of_four_l3417_341745

-- Define square root
def is_square_root (x y : ℝ) : Prop := y^2 = x

-- Theorem statement
theorem negative_two_is_square_root_of_four :
  is_square_root 4 (-2) :=
sorry

end NUMINAMATH_CALUDE_negative_two_is_square_root_of_four_l3417_341745


namespace NUMINAMATH_CALUDE_mary_marbles_l3417_341762

/-- Given that Joan has 3 yellow marbles and the total number of yellow marbles between Mary and Joan is 12, prove that Mary has 9 yellow marbles. -/
theorem mary_marbles (joan_marbles : ℕ) (total_marbles : ℕ) (h1 : joan_marbles = 3) (h2 : total_marbles = 12) :
  total_marbles - joan_marbles = 9 := by
  sorry

end NUMINAMATH_CALUDE_mary_marbles_l3417_341762


namespace NUMINAMATH_CALUDE_coin_value_proof_l3417_341785

theorem coin_value_proof (total_coins : ℕ) (penny_value : ℕ) (nickel_value : ℕ) :
  total_coins = 16 ∧ 
  penny_value = 1 ∧ 
  nickel_value = 5 →
  ∃ (pennies nickels : ℕ),
    pennies + nickels = total_coins ∧
    nickels = pennies + 2 ∧
    pennies * penny_value + nickels * nickel_value = 52 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_proof_l3417_341785


namespace NUMINAMATH_CALUDE_green_bean_to_corn_ratio_l3417_341708

/-- Represents the number of servings produced by each type of plant. -/
structure PlantServings where
  carrot : ℕ
  corn : ℕ
  greenBean : ℕ

/-- Represents the number of plants in each plot. -/
def plantsPerPlot : ℕ := 9

/-- Represents the total number of servings produced. -/
def totalServings : ℕ := 306

/-- The theorem stating the ratio of green bean to corn servings. -/
theorem green_bean_to_corn_ratio (s : PlantServings) :
  s.carrot = 4 →
  s.corn = 5 * s.carrot →
  s.greenBean * plantsPerPlot + s.carrot * plantsPerPlot + s.corn * plantsPerPlot = totalServings →
  s.greenBean * 2 = s.corn := by
  sorry

#check green_bean_to_corn_ratio

end NUMINAMATH_CALUDE_green_bean_to_corn_ratio_l3417_341708


namespace NUMINAMATH_CALUDE_largest_number_with_seven_front_l3417_341790

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ (n / 100 = 7) ∧ (n % 100 / 10 ≠ 7) ∧ (n % 10 ≠ 7)

theorem largest_number_with_seven_front :
  ∀ n : ℕ, is_valid_number n → n ≤ 743 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_seven_front_l3417_341790


namespace NUMINAMATH_CALUDE_nested_series_sum_l3417_341704

def nested_series : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * (1 + nested_series n)

theorem nested_series_sum : nested_series 7 = 510 := by
  sorry

end NUMINAMATH_CALUDE_nested_series_sum_l3417_341704


namespace NUMINAMATH_CALUDE_equivalence_condition_l3417_341766

theorem equivalence_condition (a b : ℝ) (h : a * b > 0) :
  a > b ↔ 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_equivalence_condition_l3417_341766


namespace NUMINAMATH_CALUDE_root_sum_fraction_l3417_341761

theorem root_sum_fraction (a b c : ℝ) : 
  a^3 - 8*a^2 + 7*a - 3 = 0 → 
  b^3 - 8*b^2 + 7*b - 3 = 0 → 
  c^3 - 8*c^2 + 7*c - 3 = 0 → 
  a / (b*c + 1) + b / (a*c + 1) + c / (a*b + 1) = 17/2 := by
sorry

end NUMINAMATH_CALUDE_root_sum_fraction_l3417_341761


namespace NUMINAMATH_CALUDE_paint_time_theorem_l3417_341753

/-- The time required to paint a square wall using a cylindrical paint roller -/
theorem paint_time_theorem (roller_length roller_diameter wall_side_length roller_speed : ℝ) :
  roller_length = 20 →
  roller_diameter = 15 →
  wall_side_length = 300 →
  roller_speed = 2 →
  (wall_side_length ^ 2) / (2 * π * (roller_diameter / 2) * roller_length * roller_speed) = 90000 / (600 * π) :=
by sorry

end NUMINAMATH_CALUDE_paint_time_theorem_l3417_341753


namespace NUMINAMATH_CALUDE_digit_subtraction_problem_l3417_341782

def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

theorem digit_subtraction_problem :
  ∃ (F G D E H I : ℕ),
    is_digit F ∧ is_digit G ∧ is_digit D ∧ is_digit E ∧ is_digit H ∧ is_digit I ∧
    F ≠ G ∧ F ≠ D ∧ F ≠ E ∧ F ≠ H ∧ F ≠ I ∧
    G ≠ D ∧ G ≠ E ∧ G ≠ H ∧ G ≠ I ∧
    D ≠ E ∧ D ≠ H ∧ D ≠ I ∧
    E ≠ H ∧ E ≠ I ∧
    H ≠ I ∧
    F * 10 + G = 93 ∧
    D * 10 + E = 68 ∧
    H * 10 + I = 25 ∧
    (F * 10 + G) - (D * 10 + E) = H * 10 + I :=
by
  sorry

end NUMINAMATH_CALUDE_digit_subtraction_problem_l3417_341782


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3417_341738

theorem unique_quadratic_solution (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b = 2*x ↔ x = 2) → 
  (a = -2 ∧ b = 4) := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3417_341738


namespace NUMINAMATH_CALUDE_janet_lives_count_janet_final_lives_l3417_341786

theorem janet_lives_count (initial_lives lost_lives gained_lives : ℕ) : 
  initial_lives - lost_lives + gained_lives = (initial_lives - lost_lives) + gained_lives :=
by sorry

theorem janet_final_lives : 
  38 - 16 + 32 = 54 :=
by sorry

end NUMINAMATH_CALUDE_janet_lives_count_janet_final_lives_l3417_341786


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l3417_341702

/-- The vertex of the parabola y = x^2 - 4x + a lies on the line y = -4x - 1 -/
def vertex_on_line (a : ℝ) : Prop :=
  ∃ x y : ℝ, y = x^2 - 4*x + a ∧ y = -4*x - 1

/-- The coordinates of the vertex of the parabola y = x^2 - 4x + a -/
def vertex_coordinates (a : ℝ) : ℝ × ℝ := (2, -9)

theorem parabola_vertex_coordinates (a : ℝ) :
  vertex_on_line a → vertex_coordinates a = (2, -9) := by
  sorry


end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l3417_341702


namespace NUMINAMATH_CALUDE_expected_score_is_80_l3417_341728

/-- A math test with multiple-choice questions -/
structure MathTest where
  num_questions : ℕ
  points_per_correct : ℕ
  prob_correct : ℝ

/-- Expected score for a math test -/
def expected_score (test : MathTest) : ℝ :=
  test.num_questions * test.points_per_correct * test.prob_correct

/-- Theorem: The expected score for the given test is 80 points -/
theorem expected_score_is_80 (test : MathTest) 
    (h1 : test.num_questions = 25)
    (h2 : test.points_per_correct = 4)
    (h3 : test.prob_correct = 0.8) : 
  expected_score test = 80 := by
  sorry

end NUMINAMATH_CALUDE_expected_score_is_80_l3417_341728


namespace NUMINAMATH_CALUDE_mistaken_operation_l3417_341769

/-- Given an operation O on real numbers that results in a 99% error
    compared to multiplying by 10, prove that O(x) = 0.1 * x for all x. -/
theorem mistaken_operation (O : ℝ → ℝ) (h : ∀ x : ℝ, O x = 0.01 * (10 * x)) :
  ∀ x : ℝ, O x = 0.1 * x := by
sorry

end NUMINAMATH_CALUDE_mistaken_operation_l3417_341769


namespace NUMINAMATH_CALUDE_alice_has_ball_after_two_turns_l3417_341717

/-- Represents the probability of Alice tossing the ball to Bob -/
def alice_toss_prob : ℚ := 5/8

/-- Represents the probability of Alice keeping the ball -/
def alice_keep_prob : ℚ := 3/8

/-- Represents the probability of Bob tossing the ball to Alice -/
def bob_toss_prob : ℚ := 1/4

/-- Represents the probability of Bob keeping the ball -/
def bob_keep_prob : ℚ := 3/4

/-- The theorem stating the probability of Alice having the ball after two turns -/
theorem alice_has_ball_after_two_turns : 
  alice_toss_prob * bob_toss_prob + alice_keep_prob * alice_keep_prob = 19/64 := by
  sorry

#check alice_has_ball_after_two_turns

end NUMINAMATH_CALUDE_alice_has_ball_after_two_turns_l3417_341717


namespace NUMINAMATH_CALUDE_max_area_rectangle_in_circle_max_area_is_8_l3417_341778

theorem max_area_rectangle_in_circle (x y : ℝ) : 
  x > 0 → y > 0 → x^2 + y^2 = 16 → x * y ≤ 8 := by
  sorry

theorem max_area_is_8 : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 + y^2 = 16 ∧ x * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_in_circle_max_area_is_8_l3417_341778


namespace NUMINAMATH_CALUDE_intersection_two_elements_l3417_341775

/-- The set M represents lines passing through (1,1) with slope k -/
def M (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 - 1) + 1}

/-- The set N represents a circle with center (0,1) and radius 1 -/
def N : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.2 = 0}

/-- The intersection of M and N contains exactly two elements -/
theorem intersection_two_elements (k : ℝ) : ∃ (p q : ℝ × ℝ), p ≠ q ∧
  M k ∩ N = {p, q} :=
sorry

end NUMINAMATH_CALUDE_intersection_two_elements_l3417_341775


namespace NUMINAMATH_CALUDE_bag_empty_probability_l3417_341788

/-- Probability of forming a pair when drawing 3 cards from n pairs -/
def prob_pair (n : ℕ) : ℚ :=
  (3 : ℚ) / (2 * n - 1)

/-- Probability of emptying the bag with n pairs of cards -/
def P (n : ℕ) : ℚ :=
  if n ≤ 2 then 1
  else (prob_pair n) * (P (n - 1))

theorem bag_empty_probability :
  P 6 = 9 / 385 :=
sorry

#eval (9 : ℕ) + 385

end NUMINAMATH_CALUDE_bag_empty_probability_l3417_341788


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l3417_341773

/-- Represents the number of tickets sold for a theater performance. -/
structure TicketSales where
  orchestra : ℕ
  balcony : ℕ

/-- Calculates the total number of tickets sold. -/
def TicketSales.total (s : TicketSales) : ℕ :=
  s.orchestra + s.balcony

/-- Calculates the total revenue from ticket sales. -/
def TicketSales.revenue (s : TicketSales) : ℕ :=
  12 * s.orchestra + 8 * s.balcony

theorem theater_ticket_difference (s : TicketSales) :
  s.total = 355 → s.revenue = 3320 → s.balcony - s.orchestra = 115 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l3417_341773


namespace NUMINAMATH_CALUDE_chef_used_one_apple_l3417_341750

/-- The number of apples used by a chef when making pies -/
def apples_used (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that the chef used 1 apple -/
theorem chef_used_one_apple :
  apples_used 40 39 = 1 := by
  sorry

end NUMINAMATH_CALUDE_chef_used_one_apple_l3417_341750


namespace NUMINAMATH_CALUDE_factorization_identity_l3417_341743

theorem factorization_identity (x y a : ℝ) : x * (a - y) - y * (y - a) = (x + y) * (a - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identity_l3417_341743


namespace NUMINAMATH_CALUDE_rock_max_height_l3417_341799

/-- The height function of the rock -/
def h (t : ℝ) : ℝ := 150 * t - 15 * t^2

/-- The maximum height reached by the rock -/
theorem rock_max_height : ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 375 := by
  sorry

end NUMINAMATH_CALUDE_rock_max_height_l3417_341799


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l3417_341707

theorem smallest_multiplier_for_perfect_square : 
  ∃ (n : ℕ), n > 0 ∧ ∃ (m : ℕ), 1008 * n = m^2 ∧ ∀ (k : ℕ), k > 0 → k < n → ¬∃ (l : ℕ), 1008 * k = l^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l3417_341707


namespace NUMINAMATH_CALUDE_cost_per_bushel_approx_12_l3417_341715

-- Define the given constants
def apple_price : ℚ := 0.40
def apples_per_bushel : ℕ := 48
def profit : ℚ := 15
def apples_sold : ℕ := 100

-- Define the function to calculate the cost per bushel
def cost_per_bushel : ℚ :=
  let revenue := apple_price * apples_sold
  let cost := revenue - profit
  let bushels_sold := apples_sold / apples_per_bushel
  cost / bushels_sold

-- Theorem statement
theorem cost_per_bushel_approx_12 : 
  ∃ ε > 0, |cost_per_bushel - 12| < ε :=
sorry

end NUMINAMATH_CALUDE_cost_per_bushel_approx_12_l3417_341715


namespace NUMINAMATH_CALUDE_four_intersection_points_iff_c_gt_one_l3417_341705

-- Define the ellipse equation
def ellipse (x y c : ℝ) : Prop :=
  x^2 + y^2/4 = c^2

-- Define the parabola equation
def parabola (x y c : ℝ) : Prop :=
  y = x^2 - 2*c

-- Define the intersection points
def intersection_points (c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ellipse p.1 p.2 c ∧ parabola p.1 p.2 c}

-- Theorem statement
theorem four_intersection_points_iff_c_gt_one (c : ℝ) :
  (∃ (p₁ p₂ p₃ p₄ : ℝ × ℝ), p₁ ∈ intersection_points c ∧
                            p₂ ∈ intersection_points c ∧
                            p₃ ∈ intersection_points c ∧
                            p₄ ∈ intersection_points c ∧
                            p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧
                            p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧
                            p₃ ≠ p₄) ↔
  c > 1 := by
  sorry

end NUMINAMATH_CALUDE_four_intersection_points_iff_c_gt_one_l3417_341705


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3417_341709

theorem solve_exponential_equation :
  ∃ y : ℕ, (8 : ℕ)^4 = 2^y ∧ y = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3417_341709


namespace NUMINAMATH_CALUDE_prime_equation_value_l3417_341798

theorem prime_equation_value (p q : ℕ) : 
  Prime p → Prime q → (∃ x : ℤ, p * x + 5 * q = 97) → (40 * p + 101 * q + 4 = 2003) := by
  sorry

end NUMINAMATH_CALUDE_prime_equation_value_l3417_341798


namespace NUMINAMATH_CALUDE_malcolm_red_lights_l3417_341779

def malcolm_lights (red : ℕ) (blue : ℕ) (green : ℕ) (left_to_buy : ℕ) (total_white : ℕ) : Prop :=
  blue = 3 * red ∧
  green = 6 ∧
  left_to_buy = 5 ∧
  total_white = 59 ∧
  red + blue + green + left_to_buy = total_white

theorem malcolm_red_lights :
  ∃ (red : ℕ), malcolm_lights red (3 * red) 6 5 59 ∧ red = 12 := by
  sorry

end NUMINAMATH_CALUDE_malcolm_red_lights_l3417_341779


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l3417_341733

theorem smallest_solution_of_equation :
  let f : ℝ → ℝ := λ x => (3*x)/(x-3) + (3*x^2 - 27*x)/x
  ∃ x : ℝ, f x = 14 ∧ x = (-41 - Real.sqrt 4633) / 12 ∧
  ∀ y : ℝ, f y = 14 → y ≥ (-41 - Real.sqrt 4633) / 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l3417_341733


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3417_341774

def M : Set ℤ := {x | x < 3}
def N : Set ℤ := {x | 0 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3417_341774


namespace NUMINAMATH_CALUDE_sum_of_squares_l3417_341748

theorem sum_of_squares (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ = 1)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ = 14)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ = 135) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ = 832 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3417_341748


namespace NUMINAMATH_CALUDE_min_value_theorem_l3417_341797

/-- Given a function f(x) = x(x-a)(x-b) where f'(0) = 4, 
    the minimum value of a^2 + 2b^2 is 8√2 -/
theorem min_value_theorem (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x * (x - a) * (x - b)
  let f' : ℝ → ℝ := λ x ↦ (3 * x^2) - 2 * (a + b) * x + a * b
  (f' 0 = 4) → (∀ a b : ℝ, a^2 + 2*b^2 ≥ 8 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3417_341797


namespace NUMINAMATH_CALUDE_equation_solution_l3417_341794

theorem equation_solution : ∃ x : ℝ, x + 1 - 2 * (x - 1) = 1 - 3 * x ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3417_341794


namespace NUMINAMATH_CALUDE_slope_of_line_with_60_degree_inclination_l3417_341721

theorem slope_of_line_with_60_degree_inclination :
  let angle_of_inclination : ℝ := 60 * π / 180
  let slope : ℝ := Real.tan angle_of_inclination
  slope = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_with_60_degree_inclination_l3417_341721


namespace NUMINAMATH_CALUDE_prob_four_green_out_of_seven_l3417_341736

/-- The probability of drawing exactly 4 green marbles out of 7 draws, with replacement,
    from a bag containing 10 green marbles and 5 purple marbles. -/
theorem prob_four_green_out_of_seven (total_marbles : ℕ) (green_marbles : ℕ) (purple_marbles : ℕ)
  (h1 : total_marbles = green_marbles + purple_marbles)
  (h2 : green_marbles = 10)
  (h3 : purple_marbles = 5)
  (h4 : total_marbles > 0) :
  (Nat.choose 7 4 : ℚ) * (green_marbles / total_marbles : ℚ)^4 * (purple_marbles / total_marbles : ℚ)^3 =
  35 * (2/3 : ℚ)^4 * (1/3 : ℚ)^3 :=
by sorry

end NUMINAMATH_CALUDE_prob_four_green_out_of_seven_l3417_341736


namespace NUMINAMATH_CALUDE_second_number_value_l3417_341749

theorem second_number_value (a b : ℝ) 
  (eq1 : a * (a - 6) = 7)
  (eq2 : b * (b - 6) = 7)
  (neq : a ≠ b)
  (sum : a + b = 6) :
  b = 7 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l3417_341749


namespace NUMINAMATH_CALUDE_asterisk_replacement_l3417_341760

theorem asterisk_replacement : ∃ x : ℝ, (x / 20) * (x / 180) = 1 ∧ x = 60 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l3417_341760


namespace NUMINAMATH_CALUDE_sqrt_necessary_not_sufficient_l3417_341710

-- Define the necessary condition
def necessary_condition (x y : ℝ) : Prop :=
  (∀ x y, (Real.log x > Real.log y) → (Real.sqrt x > Real.sqrt y))

-- Define the sufficient condition
def sufficient_condition (x y : ℝ) : Prop :=
  (∀ x y, (Real.sqrt x > Real.sqrt y) → (Real.log x > Real.log y))

-- Theorem stating that the condition is necessary but not sufficient
theorem sqrt_necessary_not_sufficient :
  (∃ x y, necessary_condition x y) ∧ (¬∃ x y, sufficient_condition x y) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_necessary_not_sufficient_l3417_341710


namespace NUMINAMATH_CALUDE_range_of_m_plus_n_l3417_341732

/-- The function f(x) -/
noncomputable def f (m n x : ℝ) : ℝ := m * Real.exp x + x^2 + n * x

/-- The set of roots of f(x) -/
def roots (m n : ℝ) : Set ℝ := {x | f m n x = 0}

/-- The set of roots of f(f(x)) -/
def double_roots (m n : ℝ) : Set ℝ := {x | f m n (f m n x) = 0}

/-- Main theorem: Given f(x) = me^x + x^2 + nx, where the roots of f and f(f) are the same and non-empty,
    the range of m+n is [0, 4) -/
theorem range_of_m_plus_n (m n : ℝ) 
    (h1 : roots m n = double_roots m n) 
    (h2 : roots m n ≠ ∅) : 
    0 ≤ m + n ∧ m + n < 4 := by sorry

end NUMINAMATH_CALUDE_range_of_m_plus_n_l3417_341732


namespace NUMINAMATH_CALUDE_rent_distribution_l3417_341727

/-- Represents an individual renting the pasture -/
structure Renter where
  name : String
  oxen : ℕ
  months : ℕ

/-- Calculates the share of rent for a renter -/
def calculateShare (r : Renter) (totalRent : ℚ) (totalOxMonths : ℕ) : ℚ :=
  (r.oxen * r.months : ℚ) * totalRent / totalOxMonths

/-- The main theorem stating the properties of rent distribution -/
theorem rent_distribution
  (renters : List Renter)
  (totalRent : ℚ)
  (h_positive_rent : totalRent > 0)
  (h_renters : renters = [
    ⟨"A", 10, 7⟩,
    ⟨"B", 12, 5⟩,
    ⟨"C", 15, 3⟩,
    ⟨"D", 8, 6⟩,
    ⟨"E", 20, 2⟩
  ])
  (h_total_rent : totalRent = 385) :
  let totalOxMonths := (renters.map (fun r => r.oxen * r.months)).sum
  let shares := renters.map (fun r => calculateShare r totalRent totalOxMonths)
  (∀ (r : Renter), r ∈ renters → 
    calculateShare r totalRent totalOxMonths = 
    (r.oxen * r.months : ℚ) * totalRent / totalOxMonths) ∧
  shares.sum = totalRent :=
sorry

end NUMINAMATH_CALUDE_rent_distribution_l3417_341727


namespace NUMINAMATH_CALUDE_no_solution_exists_l3417_341770

/-- Sum of digits of a natural number in decimal notation -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There does not exist a natural number n such that n * s(n) = 20222022 -/
theorem no_solution_exists : ¬ ∃ n : ℕ, n * sum_of_digits n = 20222022 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3417_341770


namespace NUMINAMATH_CALUDE_stone_transport_impossible_l3417_341771

/-- The number of stone blocks -/
def n : ℕ := 50

/-- The weight of the first stone block in kg -/
def first_weight : ℕ := 370

/-- The weight increase for each subsequent block in kg -/
def weight_increase : ℕ := 2

/-- The number of available trucks -/
def num_trucks : ℕ := 7

/-- The capacity of each truck in kg -/
def truck_capacity : ℕ := 3000

/-- The total weight of n stone blocks -/
def total_weight (n : ℕ) : ℕ :=
  n * first_weight + (n * (n - 1) / 2) * weight_increase

/-- The total capacity of all trucks -/
def total_capacity : ℕ := num_trucks * truck_capacity

theorem stone_transport_impossible : total_weight n > total_capacity := by
  sorry

end NUMINAMATH_CALUDE_stone_transport_impossible_l3417_341771


namespace NUMINAMATH_CALUDE_tile_difference_l3417_341747

/-- The side length of the nth square in the sequence -/
def side_length (n : ℕ) : ℕ := 1 + 2 * (n - 1)

/-- The number of tiles in the nth square -/
def tiles (n : ℕ) : ℕ := (side_length n) ^ 2

/-- The difference in tiles between the 11th and 10th squares -/
theorem tile_difference : tiles 11 - tiles 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_tile_difference_l3417_341747


namespace NUMINAMATH_CALUDE_smallest_bdf_value_l3417_341722

theorem smallest_bdf_value (a b c d e f : ℕ) : 
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) →
  (((a + 1) / b * c / d * e / f) - (a / b * c / d * e / f) = 3) →
  ((a / b * (c + 1) / d * e / f) - (a / b * c / d * e / f) = 4) →
  ((a / b * c / d * (e + 1) / f) - (a / b * c / d * e / f) = 5) →
  60 ≤ b * d * f ∧ ∃ (b' d' f' : ℕ), b' * d' * f' = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_bdf_value_l3417_341722


namespace NUMINAMATH_CALUDE_total_oil_volume_l3417_341768

-- Define the volume of each bottle in mL
def bottle_volume : ℕ := 200

-- Define the number of bottles
def num_bottles : ℕ := 20

-- Define the conversion factor from mL to L
def ml_per_liter : ℕ := 1000

-- Theorem to prove
theorem total_oil_volume (bottle_volume : ℕ) (num_bottles : ℕ) (ml_per_liter : ℕ) :
  bottle_volume = 200 → num_bottles = 20 → ml_per_liter = 1000 →
  (bottle_volume * num_bottles) / ml_per_liter = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_oil_volume_l3417_341768


namespace NUMINAMATH_CALUDE_sandy_book_purchase_l3417_341712

/-- The number of books Sandy bought from the second shop -/
def books_from_second_shop : ℕ := 55

/-- The number of books Sandy bought from the first shop -/
def books_from_first_shop : ℕ := 65

/-- The amount Sandy spent at the first shop in cents -/
def cost_first_shop : ℕ := 148000

/-- The amount Sandy spent at the second shop in cents -/
def cost_second_shop : ℕ := 92000

/-- The average price per book in cents -/
def average_price : ℕ := 2000

theorem sandy_book_purchase :
  books_from_second_shop = 55 :=
sorry

end NUMINAMATH_CALUDE_sandy_book_purchase_l3417_341712


namespace NUMINAMATH_CALUDE_ellipse_equation_l3417_341751

/-- Given an ellipse with equation x²/a² + y²/b² = 1, where a > b > 0,
    if its right focus is at (3, 0) and the point (0, -3) is on the ellipse,
    then a² = 18 and b² = 9. -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (c : ℝ), c^2 = a^2 - b^2 ∧ c = 3) →
  (0^2 / a^2 + (-3)^2 / b^2 = 1) →
  a^2 = 18 ∧ b^2 = 9 := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3417_341751


namespace NUMINAMATH_CALUDE_days_from_friday_l3417_341742

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def addDays (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | Nat.succ m => nextDay (addDays start m)

theorem days_from_friday :
  addDays DayOfWeek.Friday 72 = DayOfWeek.Sunday :=
by
  sorry

end NUMINAMATH_CALUDE_days_from_friday_l3417_341742


namespace NUMINAMATH_CALUDE_equation_solution_l3417_341713

theorem equation_solution : 
  {x : ℝ | x + 60 / (x - 5) = -12} = {0, -7} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3417_341713


namespace NUMINAMATH_CALUDE_apple_vendor_waste_percentage_l3417_341763

/-- Calculates the percentage of apples thrown away given the selling and discarding percentages -/
theorem apple_vendor_waste_percentage
  (initial_apples : ℝ)
  (day1_sell_percentage : ℝ)
  (day1_discard_percentage : ℝ)
  (day2_sell_percentage : ℝ)
  (h1 : initial_apples > 0)
  (h2 : day1_sell_percentage = 0.5)
  (h3 : day1_discard_percentage = 0.2)
  (h4 : day2_sell_percentage = 0.5)
  : (day1_discard_percentage * (1 - day1_sell_percentage) +
     (1 - day2_sell_percentage) * (1 - day1_sell_percentage) * (1 - day1_discard_percentage)) = 0.3 := by
  sorry

#check apple_vendor_waste_percentage

end NUMINAMATH_CALUDE_apple_vendor_waste_percentage_l3417_341763


namespace NUMINAMATH_CALUDE_mother_escape_time_max_mother_time_l3417_341725

/-- Represents a family member with their tunnel traversal time -/
structure FamilyMember where
  name : String
  time : Nat

/-- Represents the cave escape scenario -/
structure CaveEscape where
  father : FamilyMember
  mother : FamilyMember
  son : FamilyMember
  daughter : FamilyMember
  timeLimit : Nat

/-- The main theorem to prove -/
theorem mother_escape_time (scenario : CaveEscape) :
  scenario.father.time = 1 ∧
  scenario.son.time = 4 ∧
  scenario.daughter.time = 5 ∧
  scenario.timeLimit = 12 →
  scenario.mother.time = 2 := by
  sorry

/-- Helper function to calculate the minimum time for two people to cross -/
def crossTime (a b : FamilyMember) : Nat :=
  max a.time b.time

/-- Helper function to check if a given escape plan is valid -/
def isValidEscapePlan (scenario : CaveEscape) (motherTime : Nat) : Prop :=
  let totalTime := crossTime scenario.father scenario.daughter +
                   scenario.father.time +
                   crossTime scenario.father scenario.son +
                   motherTime
  totalTime ≤ scenario.timeLimit

/-- Theorem stating that 2 minutes is the maximum possible time for the mother -/
theorem max_mother_time (scenario : CaveEscape) :
  scenario.father.time = 1 ∧
  scenario.son.time = 4 ∧
  scenario.daughter.time = 5 ∧
  scenario.timeLimit = 12 →
  ∀ t : Nat, t > 2 → ¬(isValidEscapePlan scenario t) := by
  sorry

end NUMINAMATH_CALUDE_mother_escape_time_max_mother_time_l3417_341725


namespace NUMINAMATH_CALUDE_race_time_proof_l3417_341706

/-- Represents the time taken by the first five runners to finish the race -/
def first_five_time : ℝ → ℝ := λ t => 5 * t

/-- Represents the time taken by the last three runners to finish the race -/
def last_three_time : ℝ → ℝ := λ t => 3 * (t + 2)

/-- Represents the total time taken by all runners to finish the race -/
def total_time : ℝ → ℝ := λ t => first_five_time t + last_three_time t

theorem race_time_proof :
  ∃ t : ℝ, total_time t = 70 ∧ first_five_time t = 40 :=
sorry

end NUMINAMATH_CALUDE_race_time_proof_l3417_341706


namespace NUMINAMATH_CALUDE_fair_draw_l3417_341720

/-- Represents the number of players in the game -/
def num_players : ℕ := 10

/-- Represents the number of red balls in the hat -/
def red_balls : ℕ := 1

/-- Represents the number of white balls in the hat -/
def white_balls (h : ℕ) : ℕ := 10 * h - 1

/-- The probability of the host drawing a red ball -/
def host_probability (k n : ℕ) : ℚ := k / (k + n)

/-- The probability of the next player drawing a red ball -/
def next_player_probability (k n : ℕ) : ℚ := (n / (k + n)) * (k / (k + n - 1))

/-- Theorem stating the condition for a fair draw -/
theorem fair_draw (h : ℕ) :
  host_probability red_balls (white_balls h) = next_player_probability red_balls (white_balls h) :=
sorry

end NUMINAMATH_CALUDE_fair_draw_l3417_341720


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3417_341746

-- Define the sets A and B
def A : Set ℝ := {x | x * (x - 2) < 0}
def B : Set ℝ := {x | x - 1 > 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3417_341746


namespace NUMINAMATH_CALUDE_total_results_l3417_341752

theorem total_results (avg : ℚ) (first_12_avg : ℚ) (last_12_avg : ℚ) (result_13 : ℚ) :
  avg = 50 →
  first_12_avg = 14 →
  last_12_avg = 17 →
  result_13 = 878 →
  ∃ N : ℕ, (N : ℚ) * avg = 12 * first_12_avg + 12 * last_12_avg + result_13 ∧ N = 25 :=
by sorry

end NUMINAMATH_CALUDE_total_results_l3417_341752


namespace NUMINAMATH_CALUDE_nathaniel_ticket_distribution_l3417_341701

/-- The number of tickets Nathaniel gives to each of his best friends -/
def tickets_per_friend (initial_tickets : ℕ) (remaining_tickets : ℕ) (num_friends : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) / num_friends

/-- Proof that Nathaniel gave 2 tickets to each of his best friends -/
theorem nathaniel_ticket_distribution :
  tickets_per_friend 11 3 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_nathaniel_ticket_distribution_l3417_341701


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l3417_341777

theorem quadratic_integer_roots (p : ℕ) : 
  Prime p ∧ 
  (∃ x y : ℤ, x ≠ y ∧ x^2 + p*x - 512*p = 0 ∧ y^2 + p*y - 512*p = 0) ↔ 
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l3417_341777


namespace NUMINAMATH_CALUDE_initial_boarders_l3417_341718

theorem initial_boarders (initial_ratio_boarders initial_ratio_day_scholars : ℕ)
  (new_ratio_boarders new_ratio_day_scholars : ℕ)
  (new_boarders : ℕ) :
  initial_ratio_boarders = 7 →
  initial_ratio_day_scholars = 16 →
  new_ratio_boarders = 1 →
  new_ratio_day_scholars = 2 →
  new_boarders = 80 →
  ∃ (x : ℕ),
    x * initial_ratio_boarders + new_boarders = x * initial_ratio_day_scholars * new_ratio_boarders / new_ratio_day_scholars →
    x * initial_ratio_boarders = 560 :=
by sorry

end NUMINAMATH_CALUDE_initial_boarders_l3417_341718


namespace NUMINAMATH_CALUDE_disk_with_hole_moment_of_inertia_l3417_341756

/-- The moment of inertia of a disk with a hole -/
theorem disk_with_hole_moment_of_inertia
  (R M : ℝ)
  (h_R : R > 0)
  (h_M : M > 0) :
  let I₀ : ℝ := (1 / 2) * M * R^2
  let m_hole : ℝ := M / 4
  let R_hole : ℝ := R / 2
  let I_center_hole : ℝ := (1 / 2) * m_hole * R_hole^2
  let d : ℝ := R / 2
  let I_hole : ℝ := I_center_hole + m_hole * d^2
  I₀ - I_hole = (13 / 32) * M * R^2 :=
sorry

end NUMINAMATH_CALUDE_disk_with_hole_moment_of_inertia_l3417_341756


namespace NUMINAMATH_CALUDE_expression_value_l3417_341757

theorem expression_value (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 1)
  (h2 : a / x + b / y + c / z = 0)
  : x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3417_341757


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l3417_341711

-- Define the plane
variable (Plane : Type)

-- Define points
variable (O A B P P1 P2 A' B' : Plane)

-- Define the angle
variable (angle : Plane → Plane → Plane → Prop)

-- Define the property of being inside an angle
variable (insideAngle : Plane → Plane → Plane → Plane → Prop)

-- Define the property of being on a line
variable (onLine : Plane → Plane → Plane → Prop)

-- Define symmetry with respect to a line
variable (symmetricToLine : Plane → Plane → Plane → Plane → Prop)

-- Define intersection of two lines
variable (intersect : Plane → Plane → Plane → Plane → Plane → Prop)

-- Define perimeter of a triangle
variable (perimeter : Plane → Plane → Plane → ℝ)

-- Define the theorem
theorem min_perimeter_triangle
  (h1 : angle O A B)
  (h2 : insideAngle O A B P)
  (h3 : onLine O A A)
  (h4 : onLine O B B)
  (h5 : symmetricToLine P P1 O A)
  (h6 : symmetricToLine P P2 O B)
  (h7 : intersect P1 P2 O A A')
  (h8 : intersect P1 P2 O B B') :
  ∀ X Y, onLine O A X → onLine O B Y →
    perimeter P X Y ≥ perimeter P A' B' :=
  sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l3417_341711


namespace NUMINAMATH_CALUDE_george_boxes_count_l3417_341700

/-- The number of blocks each box can hold -/
def blocks_per_box : ℕ := 6

/-- The total number of blocks George has -/
def total_blocks : ℕ := 12

/-- The number of boxes George has -/
def number_of_boxes : ℕ := total_blocks / blocks_per_box

theorem george_boxes_count : number_of_boxes = 2 := by
  sorry

end NUMINAMATH_CALUDE_george_boxes_count_l3417_341700


namespace NUMINAMATH_CALUDE_siblings_combined_weight_l3417_341789

/-- Given Antonio's weight and the difference between his and his sister's weight,
    calculate their combined weight. -/
theorem siblings_combined_weight (antonio_weight sister_weight_diff : ℕ) :
  antonio_weight = 50 →
  sister_weight_diff = 12 →
  antonio_weight + (antonio_weight - sister_weight_diff) = 88 := by
  sorry

#check siblings_combined_weight

end NUMINAMATH_CALUDE_siblings_combined_weight_l3417_341789


namespace NUMINAMATH_CALUDE_rectangle_area_l3417_341784

theorem rectangle_area (length width : ℝ) (h1 : length = 2 * Real.sqrt 6) (h2 : width = 2 * Real.sqrt 3) :
  length * width = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3417_341784


namespace NUMINAMATH_CALUDE_hamburger_price_is_5_l3417_341730

-- Define the variables
def num_hamburgers : ℕ := 2
def num_cola : ℕ := 3
def cola_price : ℚ := 2
def discount : ℚ := 4
def total_paid : ℚ := 12

-- Define the theorem
theorem hamburger_price_is_5 :
  ∃ (hamburger_price : ℚ),
    hamburger_price * num_hamburgers + cola_price * num_cola - discount = total_paid ∧
    hamburger_price = 5 :=
by sorry

end NUMINAMATH_CALUDE_hamburger_price_is_5_l3417_341730


namespace NUMINAMATH_CALUDE_unique_grid_with_star_one_l3417_341703

/-- Represents a 5x5 grid of integers -/
def Grid := Fin 5 → Fin 5 → Fin 5

/-- Checks if a given row in the grid contains 1, 2, 3, 4, 5 without repetition -/
def valid_row (g : Grid) (row : Fin 5) : Prop :=
  ∀ n : Fin 5, ∃! col : Fin 5, g row col = n

/-- Checks if a given column in the grid contains 1, 2, 3, 4, 5 without repetition -/
def valid_column (g : Grid) (col : Fin 5) : Prop :=
  ∀ n : Fin 5, ∃! row : Fin 5, g row col = n

/-- Checks if a given 3x3 box in the grid contains 1, 2, 3, 4, 5 without repetition -/
def valid_box (g : Grid) (box_row box_col : Fin 2) : Prop :=
  ∀ n : Fin 5, ∃! (row col : Fin 3), g (3 * box_row + row) (3 * box_col + col) = n

/-- Checks if the entire grid is valid according to the problem constraints -/
def valid_grid (g : Grid) : Prop :=
  (∀ row : Fin 5, valid_row g row) ∧
  (∀ col : Fin 5, valid_column g col) ∧
  (∀ box_row box_col : Fin 2, valid_box g box_row box_col)

/-- The position of the cell marked with a star -/
def star_position : Fin 5 × Fin 5 := ⟨2, 4⟩

/-- The main theorem: There exists a unique valid grid where the star cell contains 1 -/
theorem unique_grid_with_star_one :
  ∃! g : Grid, valid_grid g ∧ g star_position.1 star_position.2 = 1 := by sorry

end NUMINAMATH_CALUDE_unique_grid_with_star_one_l3417_341703


namespace NUMINAMATH_CALUDE_line_of_symmetry_l3417_341795

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the symmetry property
axiom symmetry_property : ∀ x, g x = g (3 - x)

-- Define what it means for a line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- State the theorem
theorem line_of_symmetry :
  (∀ x, g x = g (3 - x)) → is_axis_of_symmetry g 1.5 :=
by sorry

end NUMINAMATH_CALUDE_line_of_symmetry_l3417_341795


namespace NUMINAMATH_CALUDE_fraction_simplification_l3417_341724

theorem fraction_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  (a^3 - a^2*b) / (a^2*b) - (a^2*b - b^3) / (a*b - b^2) - (a*b) / (a^2 - b^2) = -3*a / (a^2 - b^2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3417_341724


namespace NUMINAMATH_CALUDE_population_theorem_l3417_341781

/-- The combined population of Pirajussaraí and Tucupira three years ago -/
def combined_population_three_years_ago (pirajussarai_population : ℕ) (tucupira_population : ℕ) : ℕ :=
  pirajussarai_population + tucupira_population

/-- The current combined population of Pirajussaraí and Tucupira -/
def current_combined_population (pirajussarai_population : ℕ) (tucupira_population : ℕ) : ℕ :=
  pirajussarai_population + tucupira_population * 3 / 2

theorem population_theorem (pirajussarai_population : ℕ) (tucupira_population : ℕ) :
  current_combined_population pirajussarai_population tucupira_population = 9000 →
  combined_population_three_years_ago pirajussarai_population tucupira_population = 7200 :=
by
  sorry

#check population_theorem

end NUMINAMATH_CALUDE_population_theorem_l3417_341781


namespace NUMINAMATH_CALUDE_intersection_theorem_subset_theorem_l3417_341767

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | -x^2 + 2*m*x + 4 - m^2 ≥ 0}
def B : Set ℝ := {x | 2*x^2 - 5*x - 7 < 0}

-- Define the intersection of A and B
def A_intersect_B (m : ℝ) : Set ℝ := A m ∩ B

-- Define the complement of A in ℝ
def complement_A (m : ℝ) : Set ℝ := {x | x ∉ A m}

-- Theorem for part (1)
theorem intersection_theorem (m : ℝ) :
  A_intersect_B m = {x | 0 ≤ x ∧ x < 7/2} ↔ m = 2 :=
sorry

-- Theorem for part (2)
theorem subset_theorem (m : ℝ) :
  B ⊆ complement_A m ↔ m ≤ -3 ∨ m ≥ 11/2 :=
sorry

end NUMINAMATH_CALUDE_intersection_theorem_subset_theorem_l3417_341767


namespace NUMINAMATH_CALUDE_equation_solutions_l3417_341764

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => 1/((x-2)*(x-3)) + 1/((x-3)*(x-4)) + 1/((x-4)*(x-5))
  ∀ x : ℝ, f x = 1/12 ↔ x = (7 + Real.sqrt 153)/2 ∨ x = (7 - Real.sqrt 153)/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3417_341764


namespace NUMINAMATH_CALUDE_discounted_price_theorem_l3417_341755

/-- Given a bag marked at $200 with a 40% discount, prove that the discounted price is $120. -/
theorem discounted_price_theorem (marked_price : ℝ) (discount_percentage : ℝ) 
  (h1 : marked_price = 200)
  (h2 : discount_percentage = 40) :
  marked_price * (1 - discount_percentage / 100) = 120 := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_theorem_l3417_341755


namespace NUMINAMATH_CALUDE_prime_sum_equality_l3417_341780

theorem prime_sum_equality (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p + q = r → p < q → q < r → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_equality_l3417_341780


namespace NUMINAMATH_CALUDE_law_of_sines_extended_l3417_341765

theorem law_of_sines_extended 
  {a b c α β γ : ℝ} 
  (law_of_sines : a / Real.sin α = b / Real.sin β ∧ 
                  b / Real.sin β = c / Real.sin γ)
  (angle_sum : α + β + γ = Real.pi) :
  a = b * Real.cos γ + c * Real.cos β ∧
  b = c * Real.cos α + a * Real.cos γ ∧
  c = a * Real.cos β + b * Real.cos α := by
sorry

end NUMINAMATH_CALUDE_law_of_sines_extended_l3417_341765


namespace NUMINAMATH_CALUDE_math_olympiad_scores_l3417_341735

theorem math_olympiad_scores (n : ℕ) (scores : Fin n → ℕ) : 
  n = 20 →
  (∀ i j : Fin n, i ≠ j → scores i ≠ scores j) →
  (∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k → scores i < scores j + scores k) →
  ∀ i : Fin n, scores i > 18 := by
  sorry

end NUMINAMATH_CALUDE_math_olympiad_scores_l3417_341735


namespace NUMINAMATH_CALUDE_geometric_progression_special_ratio_l3417_341729

/-- A geometric progression with positive terms where any term is equal to the sum of the next two following terms has a common ratio of (√5 - 1)/2. -/
theorem geometric_progression_special_ratio (a : ℝ) (r : ℝ) :
  a > 0 →  -- First term is positive
  r > 0 →  -- Common ratio is positive
  (∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2)) →  -- Any term is sum of next two
  r = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_special_ratio_l3417_341729


namespace NUMINAMATH_CALUDE_arc_length_from_sector_area_l3417_341719

/-- Given a circle with radius 5 cm and a sector with area 10 cm², 
    prove that the length of the arc forming the sector is 4 cm. -/
theorem arc_length_from_sector_area (r : ℝ) (area : ℝ) (arc_length : ℝ) : 
  r = 5 → 
  area = 10 → 
  area = (arc_length / (2 * r)) * r^2 → 
  arc_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_from_sector_area_l3417_341719


namespace NUMINAMATH_CALUDE_f_symmetry_l3417_341731

-- Define a convex polygon as a list of vectors
def ConvexPolygon := List (ℝ × ℝ)

-- Define the projection function
def projection (v : ℝ × ℝ) (line : ℝ × ℝ) : ℝ := sorry

-- Define the function f
def f (P Q : ConvexPolygon) : ℝ :=
  List.sum (List.map (λ p => 
    (norm p) * (List.sum (List.map (λ q => abs (projection q p)) Q))
  ) P)

-- State the theorem
theorem f_symmetry (P Q : ConvexPolygon) : f P Q = f Q P := by sorry

end NUMINAMATH_CALUDE_f_symmetry_l3417_341731


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l3417_341723

theorem unique_solution_for_exponential_equation :
  ∀ x y : ℕ, x ≥ 1 → y ≥ 1 → (7^x = 3^y + 4) → (x = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l3417_341723


namespace NUMINAMATH_CALUDE_arithmetic_mean_fractions_l3417_341792

theorem arithmetic_mean_fractions (b c x : ℝ) (hbc : b ≠ c) (hx : x ≠ 0) :
  ((x + b) / x + (x - c) / x) / 2 = 1 + (b - c) / (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_fractions_l3417_341792


namespace NUMINAMATH_CALUDE_f_properties_l3417_341776

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x - 2

theorem f_properties :
  (∀ x y, x < y ∧ ((x < -1 ∧ y < -1) ∨ (x > 3 ∧ y > 3)) → f x > f y) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≥ -7) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x = -7) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3417_341776


namespace NUMINAMATH_CALUDE_savings_percentage_l3417_341754

theorem savings_percentage (salary : ℝ) (savings_after_increase : ℝ) 
  (expense_increase_rate : ℝ) :
  salary = 5500 →
  savings_after_increase = 220 →
  expense_increase_rate = 0.2 →
  ∃ (original_savings_percentage : ℝ),
    original_savings_percentage = 20 ∧
    savings_after_increase = salary - (1 + expense_increase_rate) * 
      (salary - (original_savings_percentage / 100) * salary) :=
by sorry

end NUMINAMATH_CALUDE_savings_percentage_l3417_341754


namespace NUMINAMATH_CALUDE_square_side_increase_l3417_341759

theorem square_side_increase (a : ℝ) (h : a > 0) : 
  let b := 2 * a
  let c := b * (1 + 80 / 100)
  c^2 = (a^2 + b^2) * (1 + 159.20000000000002 / 100) := by
  sorry

end NUMINAMATH_CALUDE_square_side_increase_l3417_341759


namespace NUMINAMATH_CALUDE_claire_pets_l3417_341744

theorem claire_pets (total_pets : ℕ) (total_males : ℕ) 
  (h_total : total_pets = 92)
  (h_males : total_males = 25) :
  ∃ (gerbils hamsters : ℕ),
    gerbils + hamsters = total_pets ∧
    (gerbils / 4 : ℚ) + (hamsters / 3 : ℚ) = total_males ∧
    gerbils = 68 := by
  sorry


end NUMINAMATH_CALUDE_claire_pets_l3417_341744


namespace NUMINAMATH_CALUDE_gold_coins_count_l3417_341758

theorem gold_coins_count (gold_value : ℕ) (silver_value : ℕ) (silver_count : ℕ) (cash : ℕ) (total : ℕ) :
  gold_value = 50 →
  silver_value = 25 →
  silver_count = 5 →
  cash = 30 →
  total = 305 →
  ∃ (gold_count : ℕ), gold_count * gold_value + silver_count * silver_value + cash = total ∧ gold_count = 3 :=
by sorry

end NUMINAMATH_CALUDE_gold_coins_count_l3417_341758


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l3417_341772

theorem no_positive_integer_solution :
  ¬ ∃ (x y : ℕ+), x^2006 - 4*y^2006 - 2006 = 4*y^2007 + 2007*y := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l3417_341772


namespace NUMINAMATH_CALUDE_problem_statement_l3417_341714

theorem problem_statement (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (heq : a + b = 1/a + 1/b) : 
  (a + b ≥ 2) ∧ ¬(a^2 + a < 2 ∧ b^2 + b < 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3417_341714


namespace NUMINAMATH_CALUDE_product_set_sum_l3417_341783

theorem product_set_sum (a₁ a₂ a₃ a₄ : ℚ) : 
  ({a₁ * a₂, a₁ * a₃, a₁ * a₄, a₂ * a₃, a₂ * a₄, a₃ * a₄} : Set ℚ) = {-24, -2, -3/2, -1/8, 1, 3} →
  a₁ + a₂ + a₃ + a₄ = 9/4 ∨ a₁ + a₂ + a₃ + a₄ = -9/4 := by
sorry

end NUMINAMATH_CALUDE_product_set_sum_l3417_341783


namespace NUMINAMATH_CALUDE_dot_product_not_sufficient_nor_necessary_for_parallel_l3417_341787

-- Define the type for plane vectors
def PlaneVector := ℝ × ℝ

-- Define dot product for plane vectors
def dot_product (a b : PlaneVector) : ℝ :=
  (a.1 * b.1) + (a.2 * b.2)

-- Define parallelism for plane vectors
def parallel (a b : PlaneVector) : Prop :=
  ∃ (k : ℝ), a = (k * b.1, k * b.2)

-- Theorem statement
theorem dot_product_not_sufficient_nor_necessary_for_parallel :
  ∃ (a b : PlaneVector),
    (dot_product a b > 0 ∧ ¬parallel a b) ∧
    (parallel a b ∧ ¬(dot_product a b > 0)) :=
sorry

end NUMINAMATH_CALUDE_dot_product_not_sufficient_nor_necessary_for_parallel_l3417_341787


namespace NUMINAMATH_CALUDE_hydrochloric_acid_moles_l3417_341726

/-- Represents the chemical reaction between Sodium bicarbonate and Hydrochloric acid -/
structure ChemicalReaction where
  sodium_bicarbonate : ℝ  -- moles of Sodium bicarbonate
  hydrochloric_acid : ℝ   -- moles of Hydrochloric acid
  sodium_chloride : ℝ     -- moles of Sodium chloride produced

/-- Theorem stating that when 1 mole of Sodium bicarbonate reacts to produce 1 mole of Sodium chloride,
    the amount of Hydrochloric acid used is also 1 mole -/
theorem hydrochloric_acid_moles (reaction : ChemicalReaction)
  (h1 : reaction.sodium_bicarbonate = 1)
  (h2 : reaction.sodium_chloride = 1) :
  reaction.hydrochloric_acid = 1 := by
  sorry


end NUMINAMATH_CALUDE_hydrochloric_acid_moles_l3417_341726
