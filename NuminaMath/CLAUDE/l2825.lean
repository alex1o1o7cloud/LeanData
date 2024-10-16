import Mathlib

namespace NUMINAMATH_CALUDE_complex_number_existence_l2825_282596

theorem complex_number_existence : ∃ z : ℂ, (z^2).re = 5 ∧ z.im ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_existence_l2825_282596


namespace NUMINAMATH_CALUDE_ram_independent_time_l2825_282593

/-- The number of days Gohul takes to complete the job independently -/
def gohul_days : ℝ := 15

/-- The number of days Ram and Gohul take to complete the job together -/
def combined_days : ℝ := 6

/-- The number of days Ram takes to complete the job independently -/
def ram_days : ℝ := 10

/-- Theorem stating that given Gohul's time and the combined time, Ram's independent time is 10 days -/
theorem ram_independent_time : 
  (1 / ram_days + 1 / gohul_days = 1 / combined_days) → ram_days = 10 := by
  sorry

end NUMINAMATH_CALUDE_ram_independent_time_l2825_282593


namespace NUMINAMATH_CALUDE_domain_and_rule_determine_function_exists_non_increasing_power_function_exists_function_without_zero_l2825_282529

-- Define a function type
def Function (α β : Type) := α → β

-- Statement 1
theorem domain_and_rule_determine_function (α β : Type) :
  ∀ (D : Set α) (f : Function α β), ∃! (F : Function α β), ∀ x ∈ D, F x = f x :=
sorry

-- Statement 2
theorem exists_non_increasing_power_function :
  ∃ (n : ℝ), ¬ (∀ x y : ℝ, 0 < x ∧ x < y → x^n < y^n) :=
sorry

-- Statement 3
theorem exists_function_without_zero :
  ∃ (f : ℝ → ℝ) (a b : ℝ), a ≠ b ∧ f a > 0 ∧ f b < 0 ∧ ¬ (∃ c ∈ Set.Ioo a b, f c = 0) :=
sorry

end NUMINAMATH_CALUDE_domain_and_rule_determine_function_exists_non_increasing_power_function_exists_function_without_zero_l2825_282529


namespace NUMINAMATH_CALUDE_courtyard_breadth_l2825_282513

/-- Calculates the breadth of a rectangular courtyard given its length, the number of bricks used, and the dimensions of each brick. -/
theorem courtyard_breadth
  (length : ℝ)
  (num_bricks : ℕ)
  (brick_length brick_width : ℝ)
  (h1 : length = 20)
  (h2 : num_bricks = 16000)
  (h3 : brick_length = 0.2)
  (h4 : brick_width = 0.1) :
  length * (num_bricks : ℝ) * brick_length * brick_width / length = 16 :=
by sorry

end NUMINAMATH_CALUDE_courtyard_breadth_l2825_282513


namespace NUMINAMATH_CALUDE_remainder_of_1731_base12_div_9_l2825_282563

/-- Converts a base-12 number to decimal --/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

/-- The base-12 representation of the number --/
def base12Number : List Nat := [1, 7, 3, 1]

theorem remainder_of_1731_base12_div_9 :
  (base12ToDecimal base12Number) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1731_base12_div_9_l2825_282563


namespace NUMINAMATH_CALUDE_x_fifth_plus_72x_l2825_282576

theorem x_fifth_plus_72x (x : ℝ) (h : x^2 + 6*x = 12) : x^5 + 72*x = 2808*x - 4320 := by
  sorry

end NUMINAMATH_CALUDE_x_fifth_plus_72x_l2825_282576


namespace NUMINAMATH_CALUDE_range_of_m_l2825_282545

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 1/x + 4/y = 1) (h_solvable : ∃ m : ℝ, x + y/4 < m^2 - 3*m) :
  ∀ m : ℝ, (x + y/4 < m^2 - 3*m) → (m < -1 ∨ m > 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2825_282545


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2825_282507

theorem arithmetic_mean_of_fractions (x a : ℝ) (hx : x ≠ 0) :
  (1 / 2) * ((2 * x + a) / x + (2 * x - a) / x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2825_282507


namespace NUMINAMATH_CALUDE_exponent_subtraction_l2825_282567

theorem exponent_subtraction (m : ℝ) : m^2020 / m^2019 = m :=
by sorry

end NUMINAMATH_CALUDE_exponent_subtraction_l2825_282567


namespace NUMINAMATH_CALUDE_eighth_fibonacci_term_l2825_282534

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem eighth_fibonacci_term : fibonacci 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_eighth_fibonacci_term_l2825_282534


namespace NUMINAMATH_CALUDE_a_5_value_l2825_282514

def S (n : ℕ) : ℤ := n^2 - 10*n

theorem a_5_value : 
  (S 5 : ℤ) - (S 4 : ℤ) = -1 :=
by sorry

end NUMINAMATH_CALUDE_a_5_value_l2825_282514


namespace NUMINAMATH_CALUDE_sum_squares_50_rings_l2825_282598

/-- The number of squares in the nth ring of a square array -/
def squares_in_ring (n : ℕ) : ℕ := 8 * n

/-- The sum of squares from the 1st to the nth ring -/
def sum_squares (n : ℕ) : ℕ := 
  (List.range n).map squares_in_ring |>.sum

/-- Theorem stating that the sum of squares in the first 50 rings is 10200 -/
theorem sum_squares_50_rings : sum_squares 50 = 10200 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_50_rings_l2825_282598


namespace NUMINAMATH_CALUDE_product_sum_relation_l2825_282509

theorem product_sum_relation (a b m : ℝ) : 
  a * b = m * (a + b) + 12 → b = 10 → b - a = 6 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l2825_282509


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l2825_282521

theorem quadratic_two_roots (a b c : ℝ) (h1 : b > a + c) (h2 : a + c > 0) :
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l2825_282521


namespace NUMINAMATH_CALUDE_park_trees_l2825_282557

theorem park_trees (blackbirds_per_tree : ℕ) (magpies : ℕ) (total_birds : ℕ) :
  blackbirds_per_tree = 3 →
  magpies = 13 →
  total_birds = 34 →
  ∃ trees : ℕ, trees * blackbirds_per_tree + magpies = total_birds ∧ trees = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_park_trees_l2825_282557


namespace NUMINAMATH_CALUDE_polygon_sides_l2825_282530

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →
  ((n - 2) * 180 = 3 * 360) →
  n = 8 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l2825_282530


namespace NUMINAMATH_CALUDE_percentage_problem_l2825_282540

theorem percentage_problem (p : ℝ) : p = 80 :=
  by
  -- Define the number as 15
  let number : ℝ := 15
  
  -- Define the condition: 40% of 15 is greater than p% of 5 by 2
  have h : 0.4 * number = p / 100 * 5 + 2 := by sorry
  
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2825_282540


namespace NUMINAMATH_CALUDE_calculation_proof_l2825_282558

theorem calculation_proof : (4 + 6 + 10) / 3 - 2 / 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2825_282558


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2825_282579

theorem at_least_one_not_less_than_two (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2825_282579


namespace NUMINAMATH_CALUDE_mixture_division_l2825_282500

/-- Converts pounds to ounces -/
def pounds_to_ounces (pounds : ℚ) : ℚ := pounds * 16

/-- Calculates the amount of mixture in each container -/
def mixture_per_container (total_weight : ℚ) (num_containers : ℕ) : ℚ :=
  (pounds_to_ounces total_weight) / num_containers

theorem mixture_division (total_weight : ℚ) (num_containers : ℕ) 
  (h1 : total_weight = 57 + 3/8) 
  (h2 : num_containers = 7) :
  ∃ (ε : ℚ), abs (mixture_per_container total_weight num_containers - 131.14) < ε ∧ ε > 0 :=
by sorry

end NUMINAMATH_CALUDE_mixture_division_l2825_282500


namespace NUMINAMATH_CALUDE_fertilizer_on_half_field_l2825_282587

/-- Theorem: Amount of fertilizer on half a football field -/
theorem fertilizer_on_half_field (total_area : ℝ) (total_fertilizer : ℝ) 
  (h1 : total_area = 7200)
  (h2 : total_fertilizer = 1200) :
  (total_fertilizer / total_area) * (total_area / 2) = 600 := by
  sorry

end NUMINAMATH_CALUDE_fertilizer_on_half_field_l2825_282587


namespace NUMINAMATH_CALUDE_roots_sum_squares_l2825_282523

theorem roots_sum_squares (r s : ℝ) : 
  r^2 - 5*r + 6 = 0 → s^2 - 5*s + 6 = 0 → r^2 + s^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_squares_l2825_282523


namespace NUMINAMATH_CALUDE_final_total_cost_is_correct_l2825_282564

def spiral_notebook_price : ℝ := 15
def personal_planner_price : ℝ := 10
def spiral_notebook_discount_threshold : ℕ := 5
def personal_planner_discount_threshold : ℕ := 10
def spiral_notebook_discount_rate : ℝ := 0.2
def personal_planner_discount_rate : ℝ := 0.15
def sales_tax_rate : ℝ := 0.07
def spiral_notebooks_bought : ℕ := 6
def personal_planners_bought : ℕ := 12

def calculate_discounted_price (price : ℝ) (quantity : ℕ) (discount_rate : ℝ) : ℝ :=
  price * quantity * (1 - discount_rate)

def calculate_total_cost : ℝ :=
  let spiral_notebook_cost := 
    calculate_discounted_price spiral_notebook_price spiral_notebooks_bought spiral_notebook_discount_rate
  let personal_planner_cost := 
    calculate_discounted_price personal_planner_price personal_planners_bought personal_planner_discount_rate
  let subtotal := spiral_notebook_cost + personal_planner_cost
  subtotal * (1 + sales_tax_rate)

theorem final_total_cost_is_correct : calculate_total_cost = 186.18 := by sorry

end NUMINAMATH_CALUDE_final_total_cost_is_correct_l2825_282564


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2825_282584

theorem cubic_equation_solution (a : ℝ) (h : 2 * a^3 + a^2 - 275 = 0) : a = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2825_282584


namespace NUMINAMATH_CALUDE_diamond_symmetry_lines_l2825_282536

/-- Definition of the diamond operation -/
def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

/-- The set of points (x, y) where x ◊ y = y ◊ x forms four lines -/
theorem diamond_symmetry_lines :
  {p : ℝ × ℝ | diamond p.1 p.2 = diamond p.2 p.1} =
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2} :=
by sorry

end NUMINAMATH_CALUDE_diamond_symmetry_lines_l2825_282536


namespace NUMINAMATH_CALUDE_perfect_game_score_l2825_282577

/-- Given that a perfect score is 21 points, prove that the total points
    after 3 perfect games is equal to 63. -/
theorem perfect_game_score (perfect_score : ℕ) (num_games : ℕ) :
  perfect_score = 21 → num_games = 3 → perfect_score * num_games = 63 := by
  sorry

end NUMINAMATH_CALUDE_perfect_game_score_l2825_282577


namespace NUMINAMATH_CALUDE_existence_of_special_point_l2825_282578

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is acute-angled -/
def isAcute (t : Triangle) : Prop := sorry

/-- Checks if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop := sorry

/-- The feet of perpendiculars from a point to the sides of a triangle -/
def feetOfPerpendiculars (p : Point) (t : Triangle) : Triangle := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- The main theorem -/
theorem existence_of_special_point (t : Triangle) (h : isAcute t) : 
  ∃ Q : Point, isInside Q t ∧ isEquilateral (feetOfPerpendiculars Q t) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_point_l2825_282578


namespace NUMINAMATH_CALUDE_pond_water_after_45_days_l2825_282597

def water_amount (initial_amount : ℕ) (days : ℕ) : ℕ :=
  initial_amount - days + 2 * (days / 3)

theorem pond_water_after_45_days :
  water_amount 300 45 = 285 := by
  sorry

end NUMINAMATH_CALUDE_pond_water_after_45_days_l2825_282597


namespace NUMINAMATH_CALUDE_evaluate_expression_l2825_282580

theorem evaluate_expression (b : ℝ) : 
  let x := b + 9
  (x - b + 4) = 13 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2825_282580


namespace NUMINAMATH_CALUDE_candy_bar_cost_is_131_l2825_282581

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of quarters John uses -/
def num_quarters : ℕ := 4

/-- The number of dimes John uses -/
def num_dimes : ℕ := 3

/-- The number of nickels John uses -/
def num_nickels : ℕ := 1

/-- The amount of change John receives in cents -/
def change_received : ℕ := 4

/-- The cost of the candy bar in cents -/
def candy_bar_cost : ℕ := 
  num_quarters * quarter_value + 
  num_dimes * dime_value + 
  num_nickels * nickel_value - 
  change_received

theorem candy_bar_cost_is_131 : candy_bar_cost = 131 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_is_131_l2825_282581


namespace NUMINAMATH_CALUDE_P_50_is_identity_l2825_282533

def P : Matrix (Fin 2) (Fin 2) ℤ := !![3, 2; -4, -3]

theorem P_50_is_identity : P^50 = 1 := by sorry

end NUMINAMATH_CALUDE_P_50_is_identity_l2825_282533


namespace NUMINAMATH_CALUDE_minimum_m_value_l2825_282515

theorem minimum_m_value (a b c m : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : m > 0) 
  (h4 : (1 / (a - b)) + (m / (b - c)) ≥ (9 / (a - c))) : 
  m ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_minimum_m_value_l2825_282515


namespace NUMINAMATH_CALUDE_min_pencils_divisible_by_3_and_4_l2825_282590

theorem min_pencils_divisible_by_3_and_4 : 
  ∃ n : ℕ, n > 0 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ ∀ m : ℕ, m > 0 → m % 3 = 0 → m % 4 = 0 → n ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_min_pencils_divisible_by_3_and_4_l2825_282590


namespace NUMINAMATH_CALUDE_log_and_inverse_properties_l2825_282516

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the inverse function of log_a(x)
noncomputable def log_inverse (a : ℝ) (x : ℝ) : ℝ := a ^ x

-- Theorem statement
theorem log_and_inverse_properties (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  -- 1. Same monotonicity
  (∀ x y, x < y → log a x < log a y ↔ log_inverse a x < log_inverse a y) ∧
  -- 2. No intersection when a > 1
  (a > 1 → ∀ x, log a x ≠ log_inverse a x) ∧
  -- 3. Intersection point on y = x
  (∀ x, log a x = log_inverse a x → log a x = x) :=
by sorry

end NUMINAMATH_CALUDE_log_and_inverse_properties_l2825_282516


namespace NUMINAMATH_CALUDE_root_sum_theorem_l2825_282556

theorem root_sum_theorem (a b : ℝ) 
  (h1 : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1^2 + a*r1 + b = 0) ∧ (r2^2 + a*r2 + b = 0))
  (h2 : ∃ s1 s2 : ℝ, s1 ≠ s2 ∧ (s1^2 + b*s1 + a = 0) ∧ (s2^2 + b*s2 + a = 0))
  (h3 : ∃ t1 t2 t3 : ℝ, t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧ 
    ((t1^2 + a*t1 + b) * (t1^2 + b*t1 + a) = 0) ∧
    ((t2^2 + a*t2 + b) * (t2^2 + b*t2 + a) = 0) ∧
    ((t3^2 + a*t3 + b) * (t3^2 + b*t3 + a) = 0)) :
  t1 + t2 + t3 = -2 := by sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l2825_282556


namespace NUMINAMATH_CALUDE_sum_difference_implies_sum_l2825_282551

theorem sum_difference_implies_sum (a b : ℕ+) : 
  (a.val * b.val * (a.val * b.val + 1)) / 2 - 
  (a.val * (a.val + 1) * b.val * (b.val + 1)) / 4 = 1200 →
  a.val + b.val = 21 := by sorry

end NUMINAMATH_CALUDE_sum_difference_implies_sum_l2825_282551


namespace NUMINAMATH_CALUDE_product_of_primes_sum_74_l2825_282541

theorem product_of_primes_sum_74 (p q : ℕ) : 
  Prime p → Prime q → p + q = 74 → p * q = 1369 := by sorry

end NUMINAMATH_CALUDE_product_of_primes_sum_74_l2825_282541


namespace NUMINAMATH_CALUDE_remainder_65_pow_65_plus_65_mod_97_l2825_282543

theorem remainder_65_pow_65_plus_65_mod_97 (h1 : Prime 97) (h2 : 65 < 97) : 
  (65^65 + 65) % 97 = 33 := by
  sorry

end NUMINAMATH_CALUDE_remainder_65_pow_65_plus_65_mod_97_l2825_282543


namespace NUMINAMATH_CALUDE_distribute_five_among_three_l2825_282504

/-- The number of ways to distribute n distinguishable objects among k distinct categories -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to distribute 5 distinguishable objects among 3 distinct categories -/
theorem distribute_five_among_three : distribute 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_among_three_l2825_282504


namespace NUMINAMATH_CALUDE_picture_frame_area_l2825_282548

theorem picture_frame_area (x y : ℤ) 
  (x_gt_one : x > 1) 
  (y_gt_one : y > 1) 
  (frame_area : (2*x + 4)*(y + 2) - x*y = 45) : 
  x*y = 15 := by
sorry

end NUMINAMATH_CALUDE_picture_frame_area_l2825_282548


namespace NUMINAMATH_CALUDE_all_six_lines_tangent_l2825_282561

/-- A line in a plane -/
structure Line :=
  (id : ℕ)

/-- A circle in a plane -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Predicate to check if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- A set of six lines in a plane -/
def six_lines : Finset Line :=
  sorry

/-- Condition: For any three lines, there exists a fourth line such that all four are tangent to some circle -/
def four_line_tangent_condition (lines : Finset Line) : Prop :=
  ∀ (l1 l2 l3 : Line), l1 ∈ lines → l2 ∈ lines → l3 ∈ lines →
    ∃ (l4 : Line) (c : Circle), l4 ∈ lines ∧
      is_tangent l1 c ∧ is_tangent l2 c ∧ is_tangent l3 c ∧ is_tangent l4 c

/-- Theorem: If the four_line_tangent_condition holds for six lines, then all six lines are tangent to the same circle -/
theorem all_six_lines_tangent (h : four_line_tangent_condition six_lines) :
  ∃ (c : Circle), ∀ (l : Line), l ∈ six_lines → is_tangent l c :=
sorry

end NUMINAMATH_CALUDE_all_six_lines_tangent_l2825_282561


namespace NUMINAMATH_CALUDE_complex_number_simplification_l2825_282527

/-- Given a complex number z = (-1-2i) / (1+i)^2, prove that z = -1 + (1/2)i -/
theorem complex_number_simplification :
  let z : ℂ := (-1 - 2*I) / (1 + I)^2
  z = -1 + (1/2)*I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l2825_282527


namespace NUMINAMATH_CALUDE_ball_bounce_height_l2825_282501

theorem ball_bounce_height (h₀ : ℝ) (r : ℝ) (h_target : ℝ) (k : ℕ) 
  (h_initial : h₀ = 800)
  (h_rebound : r = 1 / 2)
  (h_target_def : h_target = 2) :
  (∀ n : ℕ, n < k → h₀ * r ^ n ≥ h_target) ∧
  (h₀ * r ^ k < h_target) →
  k = 9 := by
sorry

end NUMINAMATH_CALUDE_ball_bounce_height_l2825_282501


namespace NUMINAMATH_CALUDE_car_sales_second_day_l2825_282537

theorem car_sales_second_day 
  (total_sales : ℕ) 
  (first_day_sales : ℕ) 
  (third_day_sales : ℕ) 
  (h1 : total_sales = 57)
  (h2 : first_day_sales = 14)
  (h3 : third_day_sales = 27) :
  total_sales - first_day_sales - third_day_sales = 16 := by
  sorry

end NUMINAMATH_CALUDE_car_sales_second_day_l2825_282537


namespace NUMINAMATH_CALUDE_no_integer_roots_l2825_282599

theorem no_integer_roots (a b c : ℤ) (h_a : a ≠ 0) 
  (h_f0 : Odd (c)) 
  (h_f1 : Odd (a + b + c)) : 
  ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_integer_roots_l2825_282599


namespace NUMINAMATH_CALUDE_calculate_expression_l2825_282568

theorem calculate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : -a - b^3 + a*b^2 = 59 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2825_282568


namespace NUMINAMATH_CALUDE_area_of_triangle_AKF_l2825_282553

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -1

-- Define points A, B, K, and F
variable (A B K : ℝ × ℝ)
def F : ℝ × ℝ := focus

-- State that A is on the parabola
axiom A_on_parabola : parabola A.1 A.2

-- State that B is on the directrix
axiom B_on_directrix : directrix B.1

-- State that K is on the directrix
axiom K_on_directrix : directrix K.1

-- State that A, F, and B are collinear
axiom A_F_B_collinear : ∃ (t : ℝ), A = F + t • (B - F) ∨ B = F + t • (A - F)

-- State that AK is perpendicular to the directrix
axiom AK_perp_directrix : (A.1 - K.1) * 0 + (A.2 - K.2) * 1 = 0

-- State that |AF| = |BF|
axiom AF_eq_BF : (A.1 - F.1)^2 + (A.2 - F.2)^2 = (B.1 - F.1)^2 + (B.2 - F.2)^2

-- Theorem to prove
theorem area_of_triangle_AKF : 
  (1/2) * abs ((A.1 - F.1) * (K.2 - F.2) - (K.1 - F.1) * (A.2 - F.2)) = 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_AKF_l2825_282553


namespace NUMINAMATH_CALUDE_affordable_housing_theorem_l2825_282585

/-- Represents the affordable housing investment and construction scenario -/
structure AffordableHousing where
  investment_2011 : ℝ
  area_2011 : ℝ
  total_investment : ℝ
  growth_rate : ℝ

/-- The affordable housing scenario satisfies the given conditions -/
def valid_scenario (ah : AffordableHousing) : Prop :=
  ah.investment_2011 = 200 ∧
  ah.area_2011 = 0.08 ∧
  ah.total_investment = 950 ∧
  ah.investment_2011 * (1 + ah.growth_rate + (1 + ah.growth_rate)^2) = ah.total_investment

/-- The growth rate is 50% and the total area built is 38 million square meters -/
theorem affordable_housing_theorem (ah : AffordableHousing) 
  (h : valid_scenario ah) : 
  ah.growth_rate = 0.5 ∧ 
  ah.total_investment / (ah.investment_2011 / ah.area_2011) = 38 := by
  sorry


end NUMINAMATH_CALUDE_affordable_housing_theorem_l2825_282585


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l2825_282508

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 10000)
  (h2 : rate = 0.04)
  (h3 : time = 1) :
  principal * rate * time = 400 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l2825_282508


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2825_282532

theorem inequality_solution_set (x : ℝ) : 
  x ≠ -2 → ((x - 2) / (x + 2) ≤ 0 ↔ -2 < x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2825_282532


namespace NUMINAMATH_CALUDE_farm_corn_cobs_l2825_282591

/-- Calculates the total number of corn cobs grown on a farm with two fields -/
def total_corn_cobs (field1_rows : ℕ) (field2_rows : ℕ) (cobs_per_row : ℕ) : ℕ :=
  (field1_rows * cobs_per_row) + (field2_rows * cobs_per_row)

/-- Theorem stating that the total number of corn cobs on the farm is 116 -/
theorem farm_corn_cobs : total_corn_cobs 13 16 4 = 116 := by
  sorry

end NUMINAMATH_CALUDE_farm_corn_cobs_l2825_282591


namespace NUMINAMATH_CALUDE_initial_budget_calculation_l2825_282573

def lyras_budget (chicken_cost beef_cost_per_pound beef_pounds remaining_budget : ℕ) : ℕ :=
  chicken_cost + beef_cost_per_pound * beef_pounds + remaining_budget

theorem initial_budget_calculation :
  lyras_budget 12 3 5 53 = 80 := by
  sorry

end NUMINAMATH_CALUDE_initial_budget_calculation_l2825_282573


namespace NUMINAMATH_CALUDE_star_calculation_l2825_282594

-- Define the * operation
def star (a b : ℤ) : ℤ := a * (a - b)

-- State the theorem
theorem star_calculation : star 2 3 + star (6 - 2) 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l2825_282594


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l2825_282588

-- Define the functions
def f (a b x : ℝ) : ℝ := -|x - a| + b
def g (c d x : ℝ) : ℝ := |x - c| + d

-- State the theorem
theorem intersection_implies_sum (a b c d : ℝ) :
  (f a b 3 = 6 ∧ f a b 9 = 2) ∧
  (g c d 3 = 6 ∧ g c d 9 = 2) →
  a + c = 12 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l2825_282588


namespace NUMINAMATH_CALUDE_lunch_to_novel_ratio_l2825_282511

theorem lunch_to_novel_ratio (initial_amount : ℕ) (novel_cost : ℕ) (remaining_amount : ℕ)
  (h1 : initial_amount = 50)
  (h2 : novel_cost = 7)
  (h3 : remaining_amount = 29) :
  (initial_amount - novel_cost - remaining_amount) / novel_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_lunch_to_novel_ratio_l2825_282511


namespace NUMINAMATH_CALUDE_square_diagonal_point_l2825_282589

-- Define the square EFGH
def Square (E F G H : ℝ × ℝ) : Prop :=
  let side := dist E F
  dist E F = side ∧ dist F G = side ∧ dist G H = side ∧ dist H E = side ∧
  (E.1 - G.1) * (F.1 - H.1) + (E.2 - G.2) * (F.2 - H.2) = 0

-- Define point Q on diagonal AC
def OnDiagonal (Q E G : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (t * E.1 + (1 - t) * G.1, t * E.2 + (1 - t) * G.2)

-- Define circumcenter
def Circumcenter (R E F Q : ℝ × ℝ) : Prop :=
  dist R E = dist R F ∧ dist R F = dist R Q

-- Main theorem
theorem square_diagonal_point (E F G H Q R₁ R₂ : ℝ × ℝ) :
  Square E F G H →
  dist E F = 8 →
  OnDiagonal Q E G →
  dist E Q > dist G Q →
  Circumcenter R₁ E F Q →
  Circumcenter R₂ G H Q →
  (R₁.1 - Q.1) * (R₂.1 - Q.1) + (R₁.2 - Q.2) * (R₂.2 - Q.2) = 0 →
  dist E Q = 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_point_l2825_282589


namespace NUMINAMATH_CALUDE_square_root_equality_l2825_282552

theorem square_root_equality (n : ℕ) :
  (((n * 2021^2) / n : ℝ).sqrt = 2021^2) → n = 2021^2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equality_l2825_282552


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2825_282595

theorem trigonometric_identity (α : Real) : 
  4.10 * (Real.cos (π/4 - α))^2 - (Real.cos (π/3 + α))^2 - 
  Real.cos (5*π/12) * Real.sin (5*π/12 - 2*α) = Real.sin (2*α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2825_282595


namespace NUMINAMATH_CALUDE_game_points_theorem_l2825_282562

/-- The total points of four players in a game -/
def total_points (eric_points mark_points samanta_points daisy_points : ℕ) : ℕ :=
  eric_points + mark_points + samanta_points + daisy_points

/-- Theorem stating the total points of the four players given the conditions -/
theorem game_points_theorem (eric_points mark_points samanta_points daisy_points : ℕ) :
  eric_points = 6 →
  mark_points = eric_points + eric_points / 2 →
  samanta_points = mark_points + 8 →
  daisy_points = (samanta_points + mark_points + eric_points) - (samanta_points + mark_points + eric_points) / 4 →
  total_points eric_points mark_points samanta_points daisy_points = 56 :=
by
  sorry

#check game_points_theorem

end NUMINAMATH_CALUDE_game_points_theorem_l2825_282562


namespace NUMINAMATH_CALUDE_lucia_hip_hop_classes_l2825_282546

/-- Represents the number of hip-hop classes Lucia takes in a week -/
def hip_hop_classes : ℕ := sorry

/-- Represents the cost of one hip-hop class -/
def hip_hop_cost : ℕ := 10

/-- Represents the number of ballet classes Lucia takes in a week -/
def ballet_classes : ℕ := 2

/-- Represents the cost of one ballet class -/
def ballet_cost : ℕ := 12

/-- Represents the number of jazz classes Lucia takes in a week -/
def jazz_classes : ℕ := 1

/-- Represents the cost of one jazz class -/
def jazz_cost : ℕ := 8

/-- Represents the total cost of Lucia's dance classes in one week -/
def total_cost : ℕ := 52

/-- Theorem stating that Lucia takes 2 hip-hop classes in a week -/
theorem lucia_hip_hop_classes : 
  hip_hop_classes = 2 :=
by sorry

end NUMINAMATH_CALUDE_lucia_hip_hop_classes_l2825_282546


namespace NUMINAMATH_CALUDE_arithmetic_progression_proof_l2825_282570

theorem arithmetic_progression_proof (a₁ d : ℕ) : 
  (a₁ * (a₁ + d) * (a₁ + 2*d) = 6) ∧ 
  (a₁ * (a₁ + d) * (a₁ + 2*d) * (a₁ + 3*d) = 24) → 
  (a₁ = 1 ∧ d = 1) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_proof_l2825_282570


namespace NUMINAMATH_CALUDE_bargain_bin_books_l2825_282538

/-- The number of books initially in the bargain bin -/
def initial_books : ℝ := 41.0

/-- The number of books added in the first addition -/
def first_addition : ℝ := 33.0

/-- The number of books added in the second addition -/
def second_addition : ℝ := 2.0

/-- The total number of books after both additions -/
def total_books : ℝ := 76.0

/-- Theorem stating that the initial number of books plus the two additions equals the total -/
theorem bargain_bin_books : 
  initial_books + first_addition + second_addition = total_books := by
  sorry

end NUMINAMATH_CALUDE_bargain_bin_books_l2825_282538


namespace NUMINAMATH_CALUDE_tree_height_difference_l2825_282566

/-- Given three trees with specific height relationships, prove the difference between half the height of the tallest tree and the height of the middle-sized tree. -/
theorem tree_height_difference (tallest middle smallest : ℝ) : 
  tallest = 108 →
  smallest = 12 →
  smallest = (1/4) * middle →
  (tallest / 2) - middle = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_height_difference_l2825_282566


namespace NUMINAMATH_CALUDE_range_of_a_l2825_282583

/-- An odd function f(x) = ax³ + bx² + cx + d satisfying certain conditions -/
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- Theorem stating the range of 'a' given the conditions -/
theorem range_of_a (a b c d : ℝ) :
  (∀ x, f a b c d x = -f a b c d (-x)) →  -- f is odd
  (f a b c d 1 = 1) →  -- f(1) = 1
  (∀ x ∈ Set.Icc (-1) 1, |f a b c d x| ≤ 1) →  -- |f(x)| ≤ 1 for x ∈ [-1, 1]
  a ∈ Set.Icc (-1/2) 4 :=  -- a ∈ [-1/2, 4]
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2825_282583


namespace NUMINAMATH_CALUDE_negative_reciprocal_of_negative_three_l2825_282565

theorem negative_reciprocal_of_negative_three :
  -(1 / -3) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_negative_reciprocal_of_negative_three_l2825_282565


namespace NUMINAMATH_CALUDE_point_distance_from_two_l2825_282571

theorem point_distance_from_two : ∀ x : ℝ, |x - 2| = 3 → x = -1 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_distance_from_two_l2825_282571


namespace NUMINAMATH_CALUDE_exponential_equation_implication_l2825_282503

theorem exponential_equation_implication (x : ℝ) : 
  4 * (3 : ℝ)^x = 2187 → (x + 2) * (x - 2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_implication_l2825_282503


namespace NUMINAMATH_CALUDE_equation_solution_l2825_282520

theorem equation_solution : ∃! x : ℝ, -2 * x^2 = (4*x + 2) / (x + 4) :=
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2825_282520


namespace NUMINAMATH_CALUDE_average_income_proof_l2825_282592

def daily_incomes : List ℝ := [200, 150, 750, 400, 500]

theorem average_income_proof : 
  (daily_incomes.sum / daily_incomes.length : ℝ) = 400 := by
  sorry

end NUMINAMATH_CALUDE_average_income_proof_l2825_282592


namespace NUMINAMATH_CALUDE_point_in_bottom_right_region_of_line_l2825_282531

/-- A point (x, y) is in the bottom-right region of the line ax + by + c = 0 (including the boundary) if ax + by + c ≥ 0 -/
def in_bottom_right_region (a b c x y : ℝ) : Prop := a * x + b * y + c ≥ 0

theorem point_in_bottom_right_region_of_line (t : ℝ) :
  in_bottom_right_region 1 (-2) 4 2 t → t ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_point_in_bottom_right_region_of_line_l2825_282531


namespace NUMINAMATH_CALUDE_magnitude_of_AB_l2825_282528

def vector_AB : ℝ × ℝ := (1, 1)

theorem magnitude_of_AB : Real.sqrt ((vector_AB.1)^2 + (vector_AB.2)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_AB_l2825_282528


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2825_282555

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given two parallel vectors (1,2) and (m,1), prove that m = 1/2 -/
theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (m, 1)
  are_parallel a b → m = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2825_282555


namespace NUMINAMATH_CALUDE_horner_V₃_eq_71_l2825_282524

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℚ) (x : ℚ) : ℚ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- Coefficients of the polynomial f(x) = 2x⁶ + 5x⁵ + 6x⁴ + 23x³ - 8x² + 10x - 3 -/
def f_coeffs : List ℚ := [2, 5, 6, 23, -8, 10, -3]

/-- The value of x -/
def x : ℚ := 2

/-- V₃ in Horner's method -/
def V₃ : ℚ := 
  let v₀ : ℚ := f_coeffs[0]!
  let v₁ : ℚ := v₀ * x + f_coeffs[1]!
  let v₂ : ℚ := v₁ * x + f_coeffs[2]!
  v₂ * x + f_coeffs[3]!

theorem horner_V₃_eq_71 : V₃ = 71 := by
  sorry

#eval V₃

end NUMINAMATH_CALUDE_horner_V₃_eq_71_l2825_282524


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l2825_282547

/-- Given a line L1 defined by 3x - 2y = 6, prove that a line L2 perpendicular to L1
    with y-intercept 2 has x-intercept 3. -/
theorem perpendicular_line_x_intercept :
  ∀ (L1 L2 : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ L1 ↔ 3 * x - 2 * y = 6) →
  (∃ (m : ℝ), ∀ (x y : ℝ), (x, y) ∈ L2 ↔ y = m * x + 2) →
  (∀ (x1 y1 x2 y2 : ℝ), (x1, y1) ∈ L1 → (x2, y2) ∈ L2 → 
    (x2 - x1) * (3 * (y2 - y1) + 2 * (x2 - x1)) = 0) →
  (3, 0) ∈ L2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l2825_282547


namespace NUMINAMATH_CALUDE_square_of_T_number_is_T_number_l2825_282519

/-- Definition of a T number -/
def is_T_number (x : ℤ) : Prop := ∃ (a b : ℤ), x = a^2 + a*b + b^2

/-- Theorem: The square of a T number is still a T number -/
theorem square_of_T_number_is_T_number (x : ℤ) (h : is_T_number x) : is_T_number (x^2) := by
  sorry

end NUMINAMATH_CALUDE_square_of_T_number_is_T_number_l2825_282519


namespace NUMINAMATH_CALUDE_trash_can_purchase_l2825_282502

/-- Represents the unit price of trash can type A -/
def price_A : ℕ := 500

/-- Represents the unit price of trash can type B -/
def price_B : ℕ := 550

/-- Represents the total number of trash cans to be purchased -/
def total_cans : ℕ := 6

/-- Represents the maximum total cost allowed -/
def max_cost : ℕ := 3100

/-- Theorem stating the correct unit prices and purchase options -/
theorem trash_can_purchase :
  (price_B = price_A + 50) ∧
  (2000 / price_A = 2200 / price_B) ∧
  (∀ a b : ℕ, 
    a + b = total_cans ∧ 
    price_A * a + price_B * b ≤ max_cost ∧
    a ≥ 0 ∧ b ≥ 0 →
    (a = 4 ∧ b = 2) ∨ (a = 5 ∧ b = 1) ∨ (a = 6 ∧ b = 0)) := by
  sorry

end NUMINAMATH_CALUDE_trash_can_purchase_l2825_282502


namespace NUMINAMATH_CALUDE_rice_price_fall_l2825_282574

theorem rice_price_fall (original_price : ℝ) (original_quantity : ℝ) : 
  original_price > 0 →
  original_quantity > 0 →
  let new_price := 0.8 * original_price
  let new_quantity := 50
  original_price * original_quantity = new_price * new_quantity →
  original_quantity = 40 := by
sorry

end NUMINAMATH_CALUDE_rice_price_fall_l2825_282574


namespace NUMINAMATH_CALUDE_max_halls_visited_l2825_282525

structure Museum :=
  (total_halls : ℕ)
  (painting_halls : ℕ)
  (sculpture_halls : ℕ)
  (is_even : total_halls % 2 = 0)
  (half_paintings : painting_halls = total_halls / 2)
  (half_sculptures : sculpture_halls = total_halls / 2)

def alternating_tour (m : Museum) (start_painting : Bool) (end_painting : Bool) : ℕ → Prop
  | 0 => start_painting
  | 1 => ¬start_painting
  | (n+2) => alternating_tour m start_painting end_painting n

theorem max_halls_visited 
  (m : Museum) 
  (h : m.total_halls = 16) 
  (start_painting : Bool) 
  (end_painting : Bool) 
  (h_start_end : start_painting = end_painting) :
  ∃ (n : ℕ), n ≤ m.total_halls - 1 ∧ 
    alternating_tour m start_painting end_painting n ∧ 
    ∀ (k : ℕ), alternating_tour m start_painting end_painting k → k ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_halls_visited_l2825_282525


namespace NUMINAMATH_CALUDE_lamp_arrangement_theorem_l2825_282572

/-- The probability of a specific arrangement of lamps -/
def lamp_arrangement_probability (total_lamps green_lamps on_lamps : ℕ) : ℚ :=
  let favorable_arrangements := (Nat.choose 6 3) * (Nat.choose 7 3)
  let total_arrangements := (Nat.choose total_lamps green_lamps) * (Nat.choose total_lamps on_lamps)
  (favorable_arrangements : ℚ) / total_arrangements

/-- The specific lamp arrangement probability for 8 lamps, 4 green, 4 on -/
def specific_lamp_probability : ℚ := lamp_arrangement_probability 8 4 4

theorem lamp_arrangement_theorem : specific_lamp_probability = 10 / 49 := by
  sorry

end NUMINAMATH_CALUDE_lamp_arrangement_theorem_l2825_282572


namespace NUMINAMATH_CALUDE_recurring_decimal_multiplication_l2825_282505

theorem recurring_decimal_multiplication : 
  (37 / 999) * (7 / 9) = 259 / 8991 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_multiplication_l2825_282505


namespace NUMINAMATH_CALUDE_smallest_number_with_17_proper_factors_l2825_282535

def number_of_factors (n : ℕ) : ℕ := (Nat.divisors n).card

def number_of_proper_factors (n : ℕ) : ℕ := (number_of_factors n) - 2

theorem smallest_number_with_17_proper_factors :
  ∃ (n : ℕ), n > 0 ∧ 
    number_of_factors n = 19 ∧ 
    number_of_proper_factors n = 17 ∧
    ∀ (m : ℕ), m > 0 → number_of_factors m = 19 → m ≥ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_17_proper_factors_l2825_282535


namespace NUMINAMATH_CALUDE_angle_measure_in_special_quadrilateral_l2825_282506

theorem angle_measure_in_special_quadrilateral :
  ∀ (P Q R S : ℝ),
  P = 3 * Q →
  P = 4 * R →
  P = 6 * S →
  P + Q + R + S = 360 →
  P = 206 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_quadrilateral_l2825_282506


namespace NUMINAMATH_CALUDE_jasons_shopping_expenses_l2825_282526

theorem jasons_shopping_expenses (total_spent jacket_price : ℚ) 
  (h1 : total_spent = 14.28)
  (h2 : jacket_price = 4.74) :
  total_spent - jacket_price = 9.54 := by
  sorry

end NUMINAMATH_CALUDE_jasons_shopping_expenses_l2825_282526


namespace NUMINAMATH_CALUDE_initial_oranges_l2825_282539

/-- Theorem: Initial number of oranges in the bin -/
theorem initial_oranges (thrown_away removed : ℕ) (added new_count : ℕ) :
  removed = 25 →
  added = 21 →
  new_count = 36 →
  ∃ initial : ℕ, initial - removed + added = new_count ∧ initial = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_oranges_l2825_282539


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l2825_282559

/-- Proves that given the conditions, the ratio of father's age to son's age is 19:7 -/
theorem father_son_age_ratio :
  ∀ (son_age father_age : ℕ),
    (father_age - 6 = 3 * (son_age - 6)) →
    (son_age + father_age = 156) →
    (father_age : ℚ) / son_age = 19 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_father_son_age_ratio_l2825_282559


namespace NUMINAMATH_CALUDE_inequalities_proof_l2825_282544

theorem inequalities_proof (a b : ℝ) (h1 : a < 0) (h2 : 0 < b) (h3 : a + b > 0) :
  a / b > -1 ∧ |a| < |b| := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2825_282544


namespace NUMINAMATH_CALUDE_hall_breadth_calculation_l2825_282549

/-- Proves that given a hall of length 36 meters, paved with 1350 stones each measuring 8 dm by 5 dm, the breadth of the hall is 15 meters. -/
theorem hall_breadth_calculation (hall_length : ℝ) (stone_length : ℝ) (stone_width : ℝ) (num_stones : ℕ) :
  hall_length = 36 →
  stone_length = 0.8 →
  stone_width = 0.5 →
  num_stones = 1350 →
  (num_stones * stone_length * stone_width) / hall_length = 15 :=
by sorry

end NUMINAMATH_CALUDE_hall_breadth_calculation_l2825_282549


namespace NUMINAMATH_CALUDE_functions_and_tangent_line_l2825_282575

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x
def g (b c : ℝ) (x : ℝ) : ℝ := b * x^2 + c

-- Define the point P
def P : ℝ × ℝ := (2, 0)

-- State the theorem
theorem functions_and_tangent_line 
  (a b c : ℝ) 
  (h1 : f a P.1 = P.2) 
  (h2 : g b c P.1 = P.2) 
  (h3 : (deriv (f a)) P.1 = (deriv (g b c)) P.1) :
  (∃ (k : ℝ), 
    (∀ x, f a x = 2 * x^3 - 8 * x) ∧ 
    (∀ x, g b c x = 4 * x^2 - 16) ∧
    (∀ x y, k * x - y - k * P.1 + P.2 = 0 ↔ y = (deriv (f a)) P.1 * (x - P.1) + P.2)) :=
sorry

end

end NUMINAMATH_CALUDE_functions_and_tangent_line_l2825_282575


namespace NUMINAMATH_CALUDE_right_triangle_from_leg_and_projection_l2825_282522

/-- Right triangle determined by one leg and projection of other leg onto hypotenuse -/
theorem right_triangle_from_leg_and_projection
  (a c₂ : ℝ) (ha : a > 0) (hc₂ : c₂ > 0) :
  ∃! (b c : ℝ), 
    b > 0 ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    c₂ * c = b^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_from_leg_and_projection_l2825_282522


namespace NUMINAMATH_CALUDE_power_of_product_l2825_282554

theorem power_of_product (a : ℝ) : (2 * a^2)^3 = 8 * a^6 := by sorry

end NUMINAMATH_CALUDE_power_of_product_l2825_282554


namespace NUMINAMATH_CALUDE_binomial_9_choose_3_l2825_282542

theorem binomial_9_choose_3 : Nat.choose 9 3 = 84 := by sorry

end NUMINAMATH_CALUDE_binomial_9_choose_3_l2825_282542


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l2825_282550

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 49)
  (h2 : x * y = 12) : 
  x^2 + y^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l2825_282550


namespace NUMINAMATH_CALUDE_probability_multiple_of_five_l2825_282582

theorem probability_multiple_of_five (total_pages : ℕ) (h : total_pages = 300) :
  (Finset.filter (fun n => n % 5 = 0) (Finset.range total_pages)).card / total_pages = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_of_five_l2825_282582


namespace NUMINAMATH_CALUDE_range_of_m_l2825_282512

-- Define the conditions
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(q x m) → ¬(p x)) →
  (∃ x, p x ∧ ¬(q x m)) →
  m ≥ 9 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2825_282512


namespace NUMINAMATH_CALUDE_bacterial_growth_l2825_282560

/-- The time interval between bacterial divisions in minutes -/
def division_interval : ℕ := 20

/-- The total duration of the culturing process in minutes -/
def total_time : ℕ := 3 * 60

/-- The number of divisions that occur during the culturing process -/
def num_divisions : ℕ := total_time / division_interval

/-- The final number of bacteria after the culturing process -/
def final_bacteria_count : ℕ := 2^num_divisions

theorem bacterial_growth :
  final_bacteria_count = 512 :=
sorry

end NUMINAMATH_CALUDE_bacterial_growth_l2825_282560


namespace NUMINAMATH_CALUDE_lower_limit_of_x_l2825_282510

theorem lower_limit_of_x (n x y : ℤ) : 
  x > n → 
  x < 8 → 
  y > 8 → 
  y < 13 → 
  (∀ a b : ℤ, a > n ∧ a < 8 ∧ b > 8 ∧ b < 13 → b - a ≤ 7) → 
  (∃ a b : ℤ, a > n ∧ a < 8 ∧ b > 8 ∧ b < 13 ∧ b - a = 7) → 
  n = 2 := by
sorry

end NUMINAMATH_CALUDE_lower_limit_of_x_l2825_282510


namespace NUMINAMATH_CALUDE_rounds_played_l2825_282569

def total_points : ℕ := 154
def points_per_round : ℕ := 11

theorem rounds_played (total : ℕ) (per_round : ℕ) (h1 : total = total_points) (h2 : per_round = points_per_round) :
  total / per_round = 14 := by
  sorry

end NUMINAMATH_CALUDE_rounds_played_l2825_282569


namespace NUMINAMATH_CALUDE_function_max_min_sum_l2825_282517

/-- Given a function f and a positive real number t, 
    this theorem states that if the sum of the maximum and minimum values of f is 4, 
    then t must equal 2. -/
theorem function_max_min_sum (t : ℝ) (h1 : t > 0) : 
  let f : ℝ → ℝ := λ x ↦ (t*x^2 + 2*x + t^2 + Real.sin x) / (x^2 + t)
  ∃ (M N : ℝ), (∀ x, f x ≤ M) ∧ (∀ x, f x ≥ N) ∧ (M + N = 4) → t = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_max_min_sum_l2825_282517


namespace NUMINAMATH_CALUDE_problem_solution_l2825_282518

def p (x : ℝ) : Prop := x^2 - 7*x + 10 < 0

def q (x m : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0

theorem problem_solution (m : ℝ) (h : m > 0) :
  (∀ x, m = 4 → (p x ∧ q x m) → (4 < x ∧ x < 5)) ∧
  ((∀ x, ¬(q x m) → ¬(p x)) ∧ ¬(∀ x, ¬(p x) → ¬(q x m)) → (5/3 ≤ m ∧ m ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2825_282518


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l2825_282586

/-- Given a geometric sequence where the fifth term is 48 and the sixth term is 72,
    prove that the second term of the sequence is 1152/81. -/
theorem geometric_sequence_second_term
  (a : ℚ) -- First term of the sequence
  (r : ℚ) -- Common ratio of the sequence
  (h1 : a * r^4 = 48) -- Fifth term is 48
  (h2 : a * r^5 = 72) -- Sixth term is 72
  : a * r = 1152 / 81 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l2825_282586
