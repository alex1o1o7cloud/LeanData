import Mathlib

namespace NUMINAMATH_CALUDE_two_numbers_sum_and_difference_l1057_105782

theorem two_numbers_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 30)
  (diff_eq : x - y = 6) : 
  x = 18 ∧ y = 12 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_difference_l1057_105782


namespace NUMINAMATH_CALUDE_cookie_jar_solution_l1057_105743

def cookie_jar_problem (initial_amount : ℝ) : Prop :=
  let doris_spent : ℝ := 6
  let martha_spent : ℝ := doris_spent / 2
  let remaining_after_doris_martha : ℝ := initial_amount - doris_spent - martha_spent
  let john_spent_percentage : ℝ := 0.2
  let john_spent : ℝ := john_spent_percentage * remaining_after_doris_martha
  let final_amount : ℝ := remaining_after_doris_martha - john_spent
  final_amount = 15

theorem cookie_jar_solution :
  ∃ (initial_amount : ℝ), cookie_jar_problem initial_amount ∧ initial_amount = 27.75 := by
  sorry

end NUMINAMATH_CALUDE_cookie_jar_solution_l1057_105743


namespace NUMINAMATH_CALUDE_quadratic_radical_equality_l1057_105781

theorem quadratic_radical_equality (a : ℝ) : 
  (∃ k : ℝ, k > 0 ∧ a + 2 = k * (3 * a)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_equality_l1057_105781


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l1057_105771

/-- The perimeter of a hexagon ABCDEF where five sides are of length 1 and the sixth side is √5 -/
theorem hexagon_perimeter (AB BC CD DE EF : ℝ) (AF : ℝ) 
  (h1 : AB = 1) (h2 : BC = 1) (h3 : CD = 1) (h4 : DE = 1) (h5 : EF = 1)
  (h6 : AF = Real.sqrt 5) : AB + BC + CD + DE + EF + AF = 5 + Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l1057_105771


namespace NUMINAMATH_CALUDE_green_ball_probability_l1057_105725

-- Define the number of balls of each color
def green_balls : ℕ := 2
def black_balls : ℕ := 3
def red_balls : ℕ := 6

-- Define the total number of balls
def total_balls : ℕ := green_balls + black_balls + red_balls

-- Define the probability of drawing a green ball
def prob_green_ball : ℚ := green_balls / total_balls

-- Theorem stating the probability of drawing a green ball
theorem green_ball_probability : prob_green_ball = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l1057_105725


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l1057_105734

/-- Given two circles in the plane, this theorem proves that the equation of the line
    passing through their intersection points has a specific form. -/
theorem intersection_line_of_circles (x y : ℝ) : 
  (x^2 + y^2 = 10) ∧ ((x-1)^2 + (y-3)^2 = 10) → x + 3*y - 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l1057_105734


namespace NUMINAMATH_CALUDE_wax_calculation_l1057_105756

/-- The amount of wax required for the feathers -/
def required_wax : ℕ := 166

/-- The additional amount of wax needed -/
def additional_wax : ℕ := 146

/-- The current amount of wax -/
def current_wax : ℕ := required_wax - additional_wax

theorem wax_calculation : current_wax = 20 := by
  sorry

end NUMINAMATH_CALUDE_wax_calculation_l1057_105756


namespace NUMINAMATH_CALUDE_earth_inhabitable_surface_l1057_105773

theorem earth_inhabitable_surface (exposed_land : ℚ) (inhabitable_land : ℚ) 
  (h1 : exposed_land = 3 / 8)
  (h2 : inhabitable_land = 2 / 3) :
  exposed_land * inhabitable_land = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_earth_inhabitable_surface_l1057_105773


namespace NUMINAMATH_CALUDE_smallest_difference_fractions_l1057_105740

theorem smallest_difference_fractions :
  ∃ (x y a b : ℤ),
    (0 < x) ∧ (x < 8) ∧ (0 < y) ∧ (y < 13) ∧
    (0 < a) ∧ (a < 8) ∧ (0 < b) ∧ (b < 13) ∧
    (x / 8 ≠ y / 13) ∧ (a / 8 ≠ b / 13) ∧
    (|x / 8 - y / 13| = |13 * x - 8 * y| / 104) ∧
    (|a / 8 - b / 13| = |13 * a - 8 * b| / 104) ∧
    (|13 * x - 8 * y| = 1) ∧ (|13 * a - 8 * b| = 1) ∧
    ∀ (p q : ℤ), (0 < p) → (p < 8) → (0 < q) → (q < 13) → (p / 8 ≠ q / 13) →
      |p / 8 - q / 13| ≥ |x / 8 - y / 13| ∧
      |p / 8 - q / 13| ≥ |a / 8 - b / 13| :=
by
  sorry

#check smallest_difference_fractions

end NUMINAMATH_CALUDE_smallest_difference_fractions_l1057_105740


namespace NUMINAMATH_CALUDE_correct_operation_l1057_105759

theorem correct_operation (x : ℝ) : x - 2*x = -x := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1057_105759


namespace NUMINAMATH_CALUDE_marshmallow_challenge_l1057_105706

/-- The marshmallow challenge problem -/
theorem marshmallow_challenge 
  (haley michael brandon : ℕ) 
  (haley_marshmallows : haley = 8)
  (brandon_half_michael : brandon = michael / 2)
  (total_marshmallows : haley + michael + brandon = 44) :
  michael / haley = 3 := by
  sorry

end NUMINAMATH_CALUDE_marshmallow_challenge_l1057_105706


namespace NUMINAMATH_CALUDE_log_43_between_consecutive_integers_l1057_105799

theorem log_43_between_consecutive_integers : 
  ∃ (c d : ℤ), c + 1 = d ∧ (c : ℝ) < Real.log 43 / Real.log 10 ∧ Real.log 43 / Real.log 10 < d ∧ c + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_43_between_consecutive_integers_l1057_105799


namespace NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l1057_105748

/-- The solution set of the inequality (ax-1)(x+1) < 0 with respect to x -/
def SolutionSet (a : ℝ) : Set ℝ := {x | -1 < x ∧ x < 1}

/-- The theorem stating that if the solution set of (ax-1)(x+1) < 0 is {x | -1 < x < 1}, then a = 1 -/
theorem solution_set_implies_a_equals_one (a : ℝ) 
  (h : ∀ x, x ∈ SolutionSet a ↔ (a*x - 1)*(x + 1) < 0) : 
  a = 1 := by
  sorry

#check solution_set_implies_a_equals_one

end NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l1057_105748


namespace NUMINAMATH_CALUDE_petya_max_spend_l1057_105766

/-- Represents the cost of a book in rubles -/
def BookCost := ℕ

/-- Represents Petya's purchasing behavior -/
structure PetyaPurchase where
  initialMoney : ℕ  -- Initial amount of money Petya had
  expensiveBookThreshold : ℕ  -- Threshold for expensive books (100 rubles)
  spentHalf : Bool  -- Whether Petya spent exactly half of his money

/-- Theorem stating that Petya couldn't have spent 5000 rubles or more on books -/
theorem petya_max_spend (purchase : PetyaPurchase) : 
  purchase.spentHalf → purchase.expensiveBookThreshold = 100 →
  ∃ (maxSpend : ℕ), maxSpend < 5000 ∧ 
  ∀ (actualSpend : ℕ), actualSpend ≤ maxSpend :=
sorry

end NUMINAMATH_CALUDE_petya_max_spend_l1057_105766


namespace NUMINAMATH_CALUDE_tim_zoo_cost_l1057_105744

/-- The total cost of animals Tim bought for his zoo -/
def total_cost (goat_price : ℝ) (goat_count : ℕ) (llama_price_ratio : ℝ) : ℝ :=
  let llama_count := 2 * goat_count
  let llama_price := goat_price * llama_price_ratio
  goat_price * goat_count + llama_price * llama_count

/-- Theorem stating the total cost of animals for Tim's zoo -/
theorem tim_zoo_cost : total_cost 400 3 1.5 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_tim_zoo_cost_l1057_105744


namespace NUMINAMATH_CALUDE_point_transformation_l1057_105729

def rotate_180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

def reflect_y_eq_neg_x (x y : ℝ) : ℝ × ℝ :=
  (-y, -x)

theorem point_transformation (a b : ℝ) :
  let Q := (a, b)
  let rotated := rotate_180 a b 2 3
  let final := reflect_y_eq_neg_x rotated.1 rotated.2
  final = (5, -1) → b - a = 6 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l1057_105729


namespace NUMINAMATH_CALUDE_third_grade_sample_size_l1057_105767

theorem third_grade_sample_size
  (total_students : ℕ)
  (first_grade_students : ℕ)
  (sample_size : ℕ)
  (second_grade_sample_ratio : ℚ)
  (h1 : total_students = 2800)
  (h2 : first_grade_students = 910)
  (h3 : sample_size = 40)
  (h4 : second_grade_sample_ratio = 3 / 10)
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_third_grade_sample_size_l1057_105767


namespace NUMINAMATH_CALUDE_calculate_expression_l1057_105752

theorem calculate_expression : 12 - (-18) + (-7) = 23 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1057_105752


namespace NUMINAMATH_CALUDE_klinked_from_connectivity_and_edges_l1057_105722

/-- A graph is k-linked if for any k pairs of vertices (s₁, t₁), ..., (sₖ, tₖ),
    there exist k vertex-disjoint paths P₁, ..., Pₖ such that Pᵢ connects sᵢ to tᵢ. -/
def IsKLinked (G : SimpleGraph α) (k : ℕ) : Prop := sorry

/-- A graph is k-connected if it remains connected after removing any k-1 vertices. -/
def IsKConnected (G : SimpleGraph α) (k : ℕ) : Prop := sorry

/-- The number of edges in a graph. -/
def NumEdges (G : SimpleGraph α) : ℕ := sorry

theorem klinked_from_connectivity_and_edges
  {α : Type*} (G : SimpleGraph α) (k : ℕ) :
  IsKConnected G (2 * k) →
  NumEdges G ≥ 8 * k →
  IsKLinked G k :=
sorry

end NUMINAMATH_CALUDE_klinked_from_connectivity_and_edges_l1057_105722


namespace NUMINAMATH_CALUDE_range_of_a_l1057_105751

-- Define the quadratic function
def f (a x : ℝ) : ℝ := (a - 1) * x^2 + 2 * (a - 1) * x - 4

-- Define the solution set of the inequality
def solution_set (a : ℝ) : Set ℝ := {x | f a x ≥ 0}

-- State the theorem
theorem range_of_a : 
  (∀ a : ℝ, solution_set a = ∅) ↔ (∀ a : ℝ, -3 < a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1057_105751


namespace NUMINAMATH_CALUDE_existence_of_even_floor_l1057_105720

theorem existence_of_even_floor (n : ℕ) : ∃ k ∈ Finset.range (n + 1), Even (⌊(2 ^ (n + k) : ℝ) * Real.sqrt 2⌋) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_even_floor_l1057_105720


namespace NUMINAMATH_CALUDE_frank_pizza_slices_l1057_105774

/-- Proves that Frank ate 3 slices of Hawaiian pizza given the conditions of the problem -/
theorem frank_pizza_slices (total_slices dean_slices sammy_slices leftover_slices : ℕ) :
  total_slices = 2 * 12 →
  dean_slices = 12 / 2 →
  sammy_slices = 12 / 3 →
  leftover_slices = 11 →
  total_slices - leftover_slices - dean_slices - sammy_slices = 3 := by
  sorry

#check frank_pizza_slices

end NUMINAMATH_CALUDE_frank_pizza_slices_l1057_105774


namespace NUMINAMATH_CALUDE_set_star_A_B_l1057_105769

-- Define the sets A and B
def A : Set ℝ := {y | y ≥ 0}
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define the set difference operation
def set_difference (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∧ x ∉ Y}

-- Define the * operation
def set_star (X Y : Set ℝ) : Set ℝ := (set_difference X Y) ∪ (set_difference Y X)

-- State the theorem
theorem set_star_A_B :
  set_star A B = {x | -3 < x ∧ x < 0} ∪ {x | x > 3} := by
  sorry

end NUMINAMATH_CALUDE_set_star_A_B_l1057_105769


namespace NUMINAMATH_CALUDE_books_read_theorem_l1057_105784

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Define the number of books read in a week
def books_read_in_week : ℕ :=
  2 + (fib 0) + (fib 1) + (fib 2) + (fib 3) + (fib 4) + (fib 5)

-- Theorem statement
theorem books_read_theorem : books_read_in_week = 22 := by
  sorry

end NUMINAMATH_CALUDE_books_read_theorem_l1057_105784


namespace NUMINAMATH_CALUDE_right_triangle_sets_l1057_105715

theorem right_triangle_sets : ∃! (a b c : ℝ), (a = 2 ∧ b = 3 ∧ c = 4) ∧
  ¬(a^2 + b^2 = c^2) ∧
  (3^2 + 4^2 = 5^2) ∧
  (6^2 + 8^2 = 10^2) ∧
  (1^2 + (Real.sqrt 2)^2 = (Real.sqrt 3)^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l1057_105715


namespace NUMINAMATH_CALUDE_divisibility_implication_l1057_105733

theorem divisibility_implication (k n : ℤ) :
  (13 ∣ (k + 4*n)) → (13 ∣ (10*k + n)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l1057_105733


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_square_leq_zero_l1057_105785

theorem negation_of_forall_positive_square_leq_zero :
  (¬ ∀ x : ℝ, x > 0 → x^2 ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_square_leq_zero_l1057_105785


namespace NUMINAMATH_CALUDE_shirt_price_calculation_l1057_105716

/-- The original price of the shirt -/
def shirt_price : ℝ := 156.52

/-- The original price of the coat -/
def coat_price : ℝ := 3 * shirt_price

/-- The original price of the pants -/
def pants_price : ℝ := 2 * shirt_price

/-- The total cost after discounts -/
def total_cost : ℝ := 900

theorem shirt_price_calculation :
  (shirt_price * 0.9 + coat_price * 0.95 + pants_price) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_shirt_price_calculation_l1057_105716


namespace NUMINAMATH_CALUDE_whitney_book_cost_l1057_105739

/-- Proves that given the conditions of Whitney's purchase, each book costs $11. -/
theorem whitney_book_cost (num_books num_magazines : ℕ) (magazine_cost total_cost book_cost : ℚ) : 
  num_books = 16 →
  num_magazines = 3 →
  magazine_cost = 1 →
  total_cost = 179 →
  total_cost = num_books * book_cost + num_magazines * magazine_cost →
  book_cost = 11 := by
sorry

end NUMINAMATH_CALUDE_whitney_book_cost_l1057_105739


namespace NUMINAMATH_CALUDE_shopping_time_calculation_l1057_105721

/-- Calculates the time spent shopping, performing tasks, and traveling between sections --/
theorem shopping_time_calculation (total_trip_time waiting_times break_time browsing_times walking_time_per_trip num_sections : ℕ) :
  total_trip_time = 165 ∧
  waiting_times = 5 + 10 + 8 + 15 + 20 ∧
  break_time = 10 ∧
  browsing_times = 12 + 7 + 10 ∧
  walking_time_per_trip = 2 ∧ -- Rounded up from 1.5
  num_sections = 8 →
  total_trip_time - (waiting_times + break_time + browsing_times + walking_time_per_trip * (num_sections - 1)) = 86 :=
by sorry

end NUMINAMATH_CALUDE_shopping_time_calculation_l1057_105721


namespace NUMINAMATH_CALUDE_initial_red_marbles_l1057_105702

/-- Represents the number of marbles in a bag -/
structure MarbleBag where
  red : ℚ
  green : ℚ

/-- The initial ratio of red to green marbles is 7:3 -/
def initial_ratio (bag : MarbleBag) : Prop :=
  bag.red / bag.green = 7 / 3

/-- After removing 14 red marbles and adding 30 green marbles, the new ratio is 1:4 -/
def new_ratio (bag : MarbleBag) : Prop :=
  (bag.red - 14) / (bag.green + 30) = 1 / 4

/-- Theorem stating that the initial number of red marbles is 24 -/
theorem initial_red_marbles (bag : MarbleBag) :
  initial_ratio bag → new_ratio bag → bag.red = 24 := by
  sorry

end NUMINAMATH_CALUDE_initial_red_marbles_l1057_105702


namespace NUMINAMATH_CALUDE_max_value_equation_l1057_105741

theorem max_value_equation (p : ℕ) (x y : ℕ) (h_prime : Nat.Prime p) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 9 * x * y = p * (p + 3 * x + 6 * y)) :
  p^2 + x^2 + y^2 ≤ 29 ∧ ∃ (p' x' y' : ℕ), 
    Nat.Prime p' ∧ x' > 0 ∧ y' > 0 ∧ 
    9 * x' * y' = p' * (p' + 3 * x' + 6 * y') ∧
    p'^2 + x'^2 + y'^2 = 29 :=
by sorry


end NUMINAMATH_CALUDE_max_value_equation_l1057_105741


namespace NUMINAMATH_CALUDE_median_mean_equality_l1057_105780

theorem median_mean_equality (n : ℝ) : 
  let s := {n, n + 2, n + 7, n + 10, n + 16}
  n + 7 = 10 → (Finset.sum s id) / 5 = 10 := by
sorry

end NUMINAMATH_CALUDE_median_mean_equality_l1057_105780


namespace NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_250_l1057_105710

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_prime_factors (n : ℕ) : Set ℕ :=
  {p : ℕ | is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q → q ∣ n → p ≤ q}

theorem sum_two_smallest_prime_factors_250 :
  ∃ (a b : ℕ), a ∈ smallest_prime_factors 250 ∧ 
               b ∈ smallest_prime_factors 250 ∧ 
               a ≠ b ∧
               a + b = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_250_l1057_105710


namespace NUMINAMATH_CALUDE_video_game_players_l1057_105708

/-- The number of players who quit the game -/
def players_quit : ℕ := 5

/-- The number of lives each remaining player has -/
def lives_per_player : ℕ := 5

/-- The total number of lives of all remaining players -/
def total_lives : ℕ := 30

/-- The initial number of players in the game -/
def initial_players : ℕ := 11

theorem video_game_players :
  initial_players = (total_lives / lives_per_player) + players_quit :=
by sorry

end NUMINAMATH_CALUDE_video_game_players_l1057_105708


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l1057_105768

/-- Given a line segment with one endpoint at (3, 4) and midpoint at (5, -8),
    the sum of the coordinates of the other endpoint is -13. -/
theorem endpoint_coordinate_sum :
  let a : ℝ × ℝ := (3, 4)  -- First endpoint
  let m : ℝ × ℝ := (5, -8) -- Midpoint
  let b : ℝ × ℝ := (2 * m.1 - a.1, 2 * m.2 - a.2) -- Other endpoint
  b.1 + b.2 = -13 := by sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l1057_105768


namespace NUMINAMATH_CALUDE_marcos_strawberries_weight_l1057_105713

theorem marcos_strawberries_weight 
  (total_weight : ℝ) 
  (dads_weight : ℝ) 
  (h1 : total_weight = 20)
  (h2 : dads_weight = 17) : 
  total_weight - dads_weight = 3 := by
sorry

end NUMINAMATH_CALUDE_marcos_strawberries_weight_l1057_105713


namespace NUMINAMATH_CALUDE_gross_profit_percentage_l1057_105758

/-- Given a sales price and gross profit, calculate the percentage of gross profit relative to the cost -/
theorem gross_profit_percentage 
  (sales_price : ℝ) 
  (gross_profit : ℝ) 
  (h1 : sales_price = 81)
  (h2 : gross_profit = 51) :
  (gross_profit / (sales_price - gross_profit)) * 100 = 170 := by
sorry


end NUMINAMATH_CALUDE_gross_profit_percentage_l1057_105758


namespace NUMINAMATH_CALUDE_tree_planting_problem_l1057_105794

theorem tree_planting_problem (total_trees : ℕ) 
  (h1 : 205000 ≤ total_trees ∧ total_trees ≤ 205300) 
  (h2 : ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 7 * (x - 1) = total_trees ∧ 13 * (y - 1) = total_trees) : 
  ∃ (students : ℕ), students = 62 ∧ 
    ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y = students ∧ 
      7 * (x - 1) = total_trees ∧ 13 * (y - 1) = total_trees :=
by sorry

end NUMINAMATH_CALUDE_tree_planting_problem_l1057_105794


namespace NUMINAMATH_CALUDE_shower_water_usage_l1057_105738

theorem shower_water_usage (roman remy riley ronan : ℝ) : 
  remy = 3 * roman + 1 →
  riley = roman + remy - 2 →
  ronan = riley / 2 →
  roman + remy + riley + ronan = 60 →
  remy = 18.85 := by
sorry

end NUMINAMATH_CALUDE_shower_water_usage_l1057_105738


namespace NUMINAMATH_CALUDE_midpoint_chain_l1057_105754

/-- Given a line segment AB with multiple midpoints, prove its length --/
theorem midpoint_chain (A B C D E F G : ℝ) : 
  (C = (A + B) / 2) →  -- C is midpoint of AB
  (D = (A + C) / 2) →  -- D is midpoint of AC
  (E = (A + D) / 2) →  -- E is midpoint of AD
  (F = (A + E) / 2) →  -- F is midpoint of AE
  (G = (A + F) / 2) →  -- G is midpoint of AF
  (G - A = 5) →        -- AG = 5
  (B - A = 160) :=     -- AB = 160
by sorry

end NUMINAMATH_CALUDE_midpoint_chain_l1057_105754


namespace NUMINAMATH_CALUDE_ratio_problem_l1057_105798

theorem ratio_problem (a b : ℝ) (h1 : a / b = 5) (h2 : a = 40) : b = 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1057_105798


namespace NUMINAMATH_CALUDE_min_difference_theorem_l1057_105718

noncomputable def f (x : ℝ) : ℝ := Real.exp x

noncomputable def g (x : ℝ) : ℝ := Real.log (x / 2) + 1 / 2

theorem min_difference_theorem (m : ℝ) (hm : m > 0) :
  ∃ (a b : ℝ), f a = m ∧ f b = m ∧
  ∀ (a' b' : ℝ), f a' = m → f b' = m → b - a ≤ b' - a' ∧ b - a = 2 + Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_min_difference_theorem_l1057_105718


namespace NUMINAMATH_CALUDE_problem_solution_l1057_105709

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) : 
  (x - 1)^2 + 16/((x - 1)^2) = 23/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1057_105709


namespace NUMINAMATH_CALUDE_divisor_proof_l1057_105707

theorem divisor_proof (dividend quotient remainder divisor : ℤ) : 
  dividend = 474232 →
  quotient = 594 →
  remainder = -968 →
  dividend = divisor * quotient + remainder →
  divisor = 800 := by
sorry

end NUMINAMATH_CALUDE_divisor_proof_l1057_105707


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l1057_105737

theorem range_of_a_minus_b (a b : ℝ) (θ : ℝ) 
  (h1 : |a - Real.sin θ ^ 2| ≤ 1) 
  (h2 : |b + Real.cos θ ^ 2| ≤ 1) : 
  -1 ≤ a - b ∧ a - b ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l1057_105737


namespace NUMINAMATH_CALUDE_smallRectLengthIsFourTimesWidth_l1057_105779

/-- Represents the arrangement of squares and a rectangle -/
structure SquareArrangement where
  s : ℝ
  largeTotalWidth : ℝ
  largeLength : ℝ
  smallRectWidth : ℝ
  smallRectLength : ℝ

/-- The conditions of the problem -/
def validArrangement (a : SquareArrangement) : Prop :=
  a.largeTotalWidth = 3 * a.s ∧
  a.largeLength = 2 * a.largeTotalWidth ∧
  a.smallRectWidth = a.s

/-- The theorem to prove -/
theorem smallRectLengthIsFourTimesWidth (a : SquareArrangement) 
  (h : validArrangement a) : a.smallRectLength = 4 * a.smallRectWidth :=
sorry

end NUMINAMATH_CALUDE_smallRectLengthIsFourTimesWidth_l1057_105779


namespace NUMINAMATH_CALUDE_room_number_ratio_bounds_l1057_105760

def remove_zero (n : ℕ) : ℕ := sorry

theorem room_number_ratio_bounds :
  ∀ (N : ℕ), N > 0 →
  let M := remove_zero N
  M > 0 ∧ M < N ∧
  (1 : ℚ) / 10 ≤ M / N ∧ M / N < (2 : ℚ) / 11 ∧
  ∀ (a b c d : ℕ), a > 0 → b > 0 → c > 0 → d > 0 →
  Nat.gcd a b = 1 → Nat.gcd c d = 1 →
  (∀ (K L : ℕ), K > 0 → L = remove_zero K → (a : ℚ) / b ≤ L / K ∧ L / K < (c : ℚ) / d) →
  (1 : ℚ) / 10 ≤ (a : ℚ) / b ∧ (c : ℚ) / d ≤ (2 : ℚ) / 11 :=
by sorry

end NUMINAMATH_CALUDE_room_number_ratio_bounds_l1057_105760


namespace NUMINAMATH_CALUDE_P_root_nature_l1057_105745

/-- The polynomial P(x) = x^6 - 4x^5 - 9x^3 + 2x + 9 -/
def P (x : ℝ) : ℝ := x^6 - 4*x^5 - 9*x^3 + 2*x + 9

theorem P_root_nature :
  (∀ x < 0, P x > 0) ∧ (∃ x > 0, P x = 0) := by sorry


end NUMINAMATH_CALUDE_P_root_nature_l1057_105745


namespace NUMINAMATH_CALUDE_remainder_problem_l1057_105700

theorem remainder_problem (d : ℕ) (h1 : d = 170) (h2 : d ∣ 690) (h3 : d ∣ 875) 
  (h4 : 875 % d = 25) (h5 : ∀ k : ℕ, k > d → ¬(k ∣ 690 ∧ k ∣ 875)) : 
  690 % d = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1057_105700


namespace NUMINAMATH_CALUDE_mean_median_difference_l1057_105723

def frequency_distribution : List (ℕ × ℕ) := [
  (0, 2), (1, 3), (2, 4), (3, 5), (4, 3), (5, 1)
]

def total_students : ℕ := 18

def median (fd : List (ℕ × ℕ)) (total : ℕ) : ℚ :=
  sorry

def mean (fd : List (ℕ × ℕ)) (total : ℕ) : ℚ :=
  sorry

theorem mean_median_difference :
  let m := mean frequency_distribution total_students
  let med := median frequency_distribution total_students
  |m - med| = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l1057_105723


namespace NUMINAMATH_CALUDE_smaller_sphere_radius_l1057_105753

-- Define the type for a sphere
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define the function to check if two spheres are externally tangent
def are_externally_tangent (s1 s2 : Sphere) : Prop :=
  let (x1, y1, z1) := s1.center
  let (x2, y2, z2) := s2.center
  (x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2 = (s1.radius + s2.radius)^2

-- Define the theorem
theorem smaller_sphere_radius 
  (s1 s2 s3 s4 : Sphere)
  (h1 : s1.radius = 2)
  (h2 : s2.radius = 2)
  (h3 : s3.radius = 3)
  (h4 : s4.radius = 3)
  (h5 : are_externally_tangent s1 s2)
  (h6 : are_externally_tangent s1 s3)
  (h7 : are_externally_tangent s1 s4)
  (h8 : are_externally_tangent s2 s3)
  (h9 : are_externally_tangent s2 s4)
  (h10 : are_externally_tangent s3 s4)
  (s5 : Sphere)
  (h11 : are_externally_tangent s1 s5)
  (h12 : are_externally_tangent s2 s5)
  (h13 : are_externally_tangent s3 s5)
  (h14 : are_externally_tangent s4 s5) :
  s5.radius = 6/11 :=
by sorry

end NUMINAMATH_CALUDE_smaller_sphere_radius_l1057_105753


namespace NUMINAMATH_CALUDE_mechanic_parts_cost_l1057_105778

theorem mechanic_parts_cost (hourly_rate : ℕ) (daily_hours : ℕ) (work_days : ℕ) (total_cost : ℕ) : 
  hourly_rate = 60 →
  daily_hours = 8 →
  work_days = 14 →
  total_cost = 9220 →
  total_cost - (hourly_rate * daily_hours * work_days) = 2500 := by
sorry

end NUMINAMATH_CALUDE_mechanic_parts_cost_l1057_105778


namespace NUMINAMATH_CALUDE_fraction_power_equality_l1057_105731

theorem fraction_power_equality : (81000 ^ 5) / (9000 ^ 5) = 59049 := by sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l1057_105731


namespace NUMINAMATH_CALUDE_ampersand_example_l1057_105762

-- Define the & operation
def ampersand (a b : ℚ) : ℚ := (a + 1) / b

-- State the theorem
theorem ampersand_example : ampersand 2 (ampersand 3 4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ampersand_example_l1057_105762


namespace NUMINAMATH_CALUDE_race_speed_ratio_race_speed_ratio_is_four_l1057_105761

/-- The ratio of A's speed to B's speed in a race where:
  * A's speed is k times B's speed
  * A runs 80 meters while B runs 20 meters
  * A and B finish at the same time
-/
theorem race_speed_ratio : ℝ → Prop :=
  fun k =>
    k > 0 ∧
    80 / k = 20 →
    k = 4

/-- The proof that k = 4 satisfies the race conditions -/
theorem race_speed_ratio_is_four : race_speed_ratio 4 := by
  sorry

end NUMINAMATH_CALUDE_race_speed_ratio_race_speed_ratio_is_four_l1057_105761


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1057_105730

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (3 - 2 * i) / (1 + 4 * i) = (1 : ℂ) / 3 + (14 : ℂ) / 15 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1057_105730


namespace NUMINAMATH_CALUDE_prove_weekly_savings_l1057_105796

def employee1_rate : ℝ := 20
def employee2_rate : ℝ := 22
def subsidy_rate : ℝ := 6
def hours_per_week : ℝ := 40

def weekly_savings : ℝ := (employee1_rate * hours_per_week) - ((employee2_rate - subsidy_rate) * hours_per_week)

theorem prove_weekly_savings : weekly_savings = 160 := by
  sorry

end NUMINAMATH_CALUDE_prove_weekly_savings_l1057_105796


namespace NUMINAMATH_CALUDE_fair_coin_four_tosses_l1057_105701

/-- A fair coin is a coin with equal probability of landing on either side -/
def fairCoin (p : ℝ) : Prop := p = 1/2

/-- The probability of n consecutive tosses landing on the same side -/
def consecutiveSameSide (p : ℝ) (n : ℕ) : ℝ := p^(n-1)

/-- Theorem: The probability of a fair coin landing on the same side 4 times in a row is 1/8 -/
theorem fair_coin_four_tosses (p : ℝ) (h : fairCoin p) : consecutiveSameSide p 4 = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_fair_coin_four_tosses_l1057_105701


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_squared_is_zero_l1057_105764

theorem imaginary_part_of_i_squared_is_zero :
  Complex.im (Complex.I ^ 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_squared_is_zero_l1057_105764


namespace NUMINAMATH_CALUDE_notebook_cost_l1057_105711

theorem notebook_cost (total_students : Nat) (buyers : Nat) (total_cost : ℚ) 
  (h1 : total_students = 36)
  (h2 : buyers > total_students / 2)
  (h3 : ∃ (notebooks_per_student : Nat) (cost_per_notebook : ℚ),
    notebooks_per_student > 0 ∧
    cost_per_notebook > notebooks_per_student ∧
    buyers * notebooks_per_student * cost_per_notebook = total_cost)
  (h4 : total_cost = 2664 / 100) :
  ∃ (notebooks_per_student : Nat) (cost_per_notebook : ℚ),
    notebooks_per_student > 0 ∧
    cost_per_notebook > notebooks_per_student ∧
    buyers * notebooks_per_student * cost_per_notebook = total_cost ∧
    cost_per_notebook = 37 / 100 :=
by sorry

end NUMINAMATH_CALUDE_notebook_cost_l1057_105711


namespace NUMINAMATH_CALUDE_blue_tile_probability_l1057_105783

/-- A function that determines if a number is congruent to 3 mod 7 -/
def isBlue (n : ℕ) : Bool :=
  n % 7 = 3

/-- The total number of tiles in the box -/
def totalTiles : ℕ := 70

/-- The number of blue tiles in the box -/
def blueTiles : ℕ := (List.range totalTiles).filter isBlue |>.length

/-- The probability of selecting a blue tile -/
def probabilityBlue : ℚ := blueTiles / totalTiles

theorem blue_tile_probability :
  probabilityBlue = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_blue_tile_probability_l1057_105783


namespace NUMINAMATH_CALUDE_g_of_5_eq_50_l1057_105746

/-- The polynomial g(x) = 3x^4 - 20x^3 + 40x^2 - 50x - 75 -/
def g (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 40*x^2 - 50*x - 75

/-- Theorem: g(5) = 50 -/
theorem g_of_5_eq_50 : g 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_eq_50_l1057_105746


namespace NUMINAMATH_CALUDE_parabola_point_range_l1057_105789

/-- The range of y-coordinates for point C on a parabola, given specific conditions -/
theorem parabola_point_range (b c : ℝ) : 
  let A : ℝ × ℝ := (0, 2)
  let B : ℝ × ℝ := (b^2 - 4, b)
  let C : ℝ × ℝ := (c^2 - 4, c)
  (∀ x y, y^2 = x + 4 → (x = b^2 - 4 ∧ y = b) ∨ (x = c^2 - 4 ∧ y = c)) →  -- B and C are on the parabola
  ((0 - (b^2 - 4)) * ((c^2 - 4) - (b^2 - 4)) + (2 - b) * (c - b) = 0) →  -- AB ⟂ BC
  c ≤ 0 ∨ c ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_point_range_l1057_105789


namespace NUMINAMATH_CALUDE_function_equality_l1057_105792

theorem function_equality (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x ≤ x) 
  (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) : 
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_function_equality_l1057_105792


namespace NUMINAMATH_CALUDE_unique_n_solution_l1057_105790

theorem unique_n_solution : ∃! (n : ℕ), 
  n > 0 ∧ (Nat.factorial (n + 1) + Nat.factorial (n + 2) = Nat.factorial n * 440) ∧ n = 19 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_solution_l1057_105790


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l1057_105765

/-- A right triangle with sides 12, 16, and 20 -/
structure RightTriangle where
  de : ℝ
  ef : ℝ
  df : ℝ
  is_right : de^2 + ef^2 = df^2
  de_eq : de = 12
  ef_eq : ef = 16
  df_eq : df = 20

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) where
  side_length : ℝ
  on_hypotenuse : side_length ≤ t.df
  on_other_sides : side_length ≤ t.de ∧ side_length ≤ t.ef

/-- The side length of the inscribed square is 80/9 -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 80 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l1057_105765


namespace NUMINAMATH_CALUDE_soccer_tournament_theorem_l1057_105787

/-- Represents a soccer tournament -/
structure SoccerTournament where
  n : ℕ  -- Total number of teams
  m : ℕ  -- Number of teams placed last
  h1 : n > m
  h2 : m ≥ 1

/-- Checks if the given n and m satisfy the tournament conditions -/
def validTournament (t : SoccerTournament) : Prop :=
  ∃ k : ℕ, k ≥ 1 ∧ t.n = (k + 1)^2 ∧ t.m = k * (k + 1) / 2

/-- The main theorem stating that only specific values of n and m are possible -/
theorem soccer_tournament_theorem (t : SoccerTournament) : validTournament t :=
  sorry

end NUMINAMATH_CALUDE_soccer_tournament_theorem_l1057_105787


namespace NUMINAMATH_CALUDE_solutions_based_on_discriminant_l1057_105795

/-- Represents the system of equations -/
def SystemOfEquations (a b c : ℝ) (n : ℕ) (x : ℕ → ℝ) : Prop :=
  (a ≠ 0) ∧ 
  (∀ i ∈ Finset.range n, a * (x i)^2 + b * (x i) + c = x ((i + 1) % n))

/-- Theorem stating the number of solutions based on the discriminant -/
theorem solutions_based_on_discriminant (a b c : ℝ) (n : ℕ) :
  (a ≠ 0) ∧ (n > 0) →
  (((b - 1)^2 - 4*a*c < 0 → ¬∃ x, SystemOfEquations a b c n x) ∧
   ((b - 1)^2 - 4*a*c = 0 → ∃! x, SystemOfEquations a b c n x) ∧
   ((b - 1)^2 - 4*a*c > 0 → ∃ x y, x ≠ y ∧ SystemOfEquations a b c n x ∧ SystemOfEquations a b c n y)) :=
by sorry

end NUMINAMATH_CALUDE_solutions_based_on_discriminant_l1057_105795


namespace NUMINAMATH_CALUDE_robotics_club_enrollment_l1057_105749

theorem robotics_club_enrollment (total : ℕ) (cs : ℕ) (elec : ℕ) (both : ℕ) 
  (h1 : total = 80)
  (h2 : cs = 50)
  (h3 : elec = 40)
  (h4 : both = 25) :
  total - (cs + elec - both) = 15 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_enrollment_l1057_105749


namespace NUMINAMATH_CALUDE_cookie_batch_size_l1057_105717

theorem cookie_batch_size 
  (num_batches : ℕ) 
  (num_people : ℕ) 
  (cookies_per_person : ℕ) 
  (h1 : num_batches = 4)
  (h2 : num_people = 16)
  (h3 : cookies_per_person = 6) :
  (num_people * cookies_per_person) / num_batches / 12 = 2 := by
sorry

end NUMINAMATH_CALUDE_cookie_batch_size_l1057_105717


namespace NUMINAMATH_CALUDE_pencil_price_theorem_l1057_105736

/-- Calculates the final price of a pencil after applying discounts and taxes -/
def final_price (initial_cost christmas_discount seasonal_discount final_discount tax_rate : ℚ) : ℚ :=
  let price_after_christmas := initial_cost - christmas_discount
  let price_after_seasonal := price_after_christmas * (1 - seasonal_discount)
  let price_after_final := price_after_seasonal * (1 - final_discount)
  price_after_final * (1 + tax_rate)

/-- The final price of the pencil is approximately $3.17 -/
theorem pencil_price_theorem :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ 
  |final_price 4 0.63 0.07 0.05 0.065 - 3.17| < ε :=
sorry

end NUMINAMATH_CALUDE_pencil_price_theorem_l1057_105736


namespace NUMINAMATH_CALUDE_inequality_proofs_l1057_105726

theorem inequality_proofs 
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < 1) 
  (x : ℝ) (hx : x ≥ 0) 
  (a b p q : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) (hq : q > 0)
  (hpq : 1 / p + 1 / q = 1) : 
  (x^α - α*x ≤ 1 - α) ∧ (a * b ≤ (1/p) * a^p + (1/q) * b^q) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proofs_l1057_105726


namespace NUMINAMATH_CALUDE_glove_probability_l1057_105747

/-- The probability of picking one left-handed glove and one right-handed glove -/
theorem glove_probability (left_gloves right_gloves : ℕ) 
  (h1 : left_gloves = 12) 
  (h2 : right_gloves = 10) : 
  (left_gloves * right_gloves : ℚ) / (Nat.choose (left_gloves + right_gloves) 2) = 120 / 231 := by
  sorry

end NUMINAMATH_CALUDE_glove_probability_l1057_105747


namespace NUMINAMATH_CALUDE_concert_ticket_cost_l1057_105786

theorem concert_ticket_cost (num_tickets : ℕ) (discount_rate : ℚ) (discount_threshold : ℕ) (total_paid : ℚ) :
  num_tickets = 12 →
  discount_rate = 5 / 100 →
  discount_threshold = 10 →
  total_paid = 476 →
  ∃ (original_cost : ℚ), 
    original_cost * (num_tickets - discount_rate * (num_tickets - discount_threshold)) = total_paid ∧
    original_cost = 40 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_cost_l1057_105786


namespace NUMINAMATH_CALUDE_rebecca_eggs_count_l1057_105724

theorem rebecca_eggs_count (num_groups : ℕ) (eggs_per_group : ℕ) 
  (h1 : num_groups = 11) (h2 : eggs_per_group = 2) : 
  num_groups * eggs_per_group = 22 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_eggs_count_l1057_105724


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1057_105714

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  x + 1 > 0 ∧ x + 3 ≤ 4

-- Define the solution set
def solution_set : Set ℝ :=
  {x : ℝ | -1 < x ∧ x ≤ 1}

-- Theorem statement
theorem inequality_system_solution_set :
  {x : ℝ | inequality_system x} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1057_105714


namespace NUMINAMATH_CALUDE_line_intersection_canonical_equations_l1057_105776

/-- The canonical equations of the line of intersection of two planes -/
theorem line_intersection_canonical_equations 
  (x y z : ℝ) : 
  (6*x - 5*y + 3*z + 8 = 0) ∧ (6*x + 5*y - 4*z + 4 = 0) →
  ∃ (t : ℝ), x = 5*t - 1 ∧ y = 42*t + 2/5 ∧ z = 60*t :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_canonical_equations_l1057_105776


namespace NUMINAMATH_CALUDE_sin_1320_degrees_l1057_105732

theorem sin_1320_degrees : Real.sin (1320 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_1320_degrees_l1057_105732


namespace NUMINAMATH_CALUDE_halloween_candy_percentage_l1057_105788

theorem halloween_candy_percentage (maggie_candy : ℕ) (neil_percentage : ℚ) (neil_candy : ℕ) :
  maggie_candy = 50 →
  neil_percentage = 40 / 100 →
  neil_candy = 91 →
  ∃ harper_percentage : ℚ,
    harper_percentage = 30 / 100 ∧
    neil_candy = (1 + neil_percentage) * (maggie_candy + harper_percentage * maggie_candy) :=
by sorry

end NUMINAMATH_CALUDE_halloween_candy_percentage_l1057_105788


namespace NUMINAMATH_CALUDE_scavenger_hunt_theorem_l1057_105775

/-- Represents the number of choices for each day of the scavenger hunt --/
def scavenger_hunt_choices : List Nat := [1, 2, 4, 3, 1]

/-- The total number of combinations for the scavenger hunt --/
def total_combinations : Nat := scavenger_hunt_choices.prod

theorem scavenger_hunt_theorem :
  total_combinations = 24 := by
  sorry

end NUMINAMATH_CALUDE_scavenger_hunt_theorem_l1057_105775


namespace NUMINAMATH_CALUDE_candy_left_is_49_l1057_105763

/-- The number of pieces of candy Brent has left after trick-or-treating and giving some to his sister -/
def candy_left : ℕ :=
  let kit_kats := 5
  let hershey_kisses := 3 * kit_kats
  let nerds := 8
  let lollipops := 11
  let baby_ruths := 10
  let reese_cups := baby_ruths / 2
  let total_candy := kit_kats + hershey_kisses + nerds + lollipops + baby_ruths + reese_cups
  let given_away := 5
  total_candy - given_away

theorem candy_left_is_49 : candy_left = 49 := by
  sorry

end NUMINAMATH_CALUDE_candy_left_is_49_l1057_105763


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1057_105772

/-- The distance between the foci of an ellipse with center (3, 2), semi-major axis 7, and semi-minor axis 3 is 4√10. -/
theorem ellipse_foci_distance :
  ∀ (center : ℝ × ℝ) (semi_major semi_minor : ℝ),
    center = (3, 2) →
    semi_major = 7 →
    semi_minor = 3 →
    let c := Real.sqrt (semi_major^2 - semi_minor^2)
    2 * c = 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1057_105772


namespace NUMINAMATH_CALUDE_min_rods_eq_2n_minus_2_l1057_105755

/-- A puzzle is an n × n grid with n cells removed, no two in the same row or column -/
structure Puzzle (n : ℕ) where
  (n_ge_two : n ≥ 2)

/-- A rod is a 1 × k or k × 1 subgrid where k is a positive integer -/
inductive Rod
  | horizontal : ℕ+ → Rod
  | vertical : ℕ+ → Rod

/-- m(A) is the minimum number of rods needed to partition puzzle A -/
def min_rods (n : ℕ) (A : Puzzle n) : ℕ := sorry

/-- The main theorem: For any n × n puzzle A, m(A) = 2n - 2 -/
theorem min_rods_eq_2n_minus_2 (n : ℕ) (A : Puzzle n) : 
  min_rods n A = 2 * n - 2 := by sorry

end NUMINAMATH_CALUDE_min_rods_eq_2n_minus_2_l1057_105755


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_ab_product_l1057_105791

-- Define the ellipse and hyperbola equations
def ellipse_equation (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def hyperbola_equation (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the theorem
theorem ellipse_hyperbola_ab_product 
  (a b : ℝ) 
  (h_ellipse : ∃ (x y : ℝ), ellipse_equation x y a b ∧ (x = 0 ∧ y = 5 ∨ x = 0 ∧ y = -5))
  (h_hyperbola : ∃ (x y : ℝ), hyperbola_equation x y a b ∧ (x = 7 ∧ y = 0 ∨ x = -7 ∧ y = 0)) :
  |a * b| = 2 * Real.sqrt 111 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_ab_product_l1057_105791


namespace NUMINAMATH_CALUDE_maciek_purchase_cost_l1057_105742

/-- The cost of Maciek's purchases -/
def total_cost (pretzel_cost : ℝ) (chip_cost_percentage : ℝ) : ℝ :=
  let chip_cost := pretzel_cost * (1 + chip_cost_percentage)
  2 * pretzel_cost + 2 * chip_cost

/-- Proof that Maciek's purchases cost $22 -/
theorem maciek_purchase_cost :
  total_cost 4 0.75 = 22 := by
  sorry

end NUMINAMATH_CALUDE_maciek_purchase_cost_l1057_105742


namespace NUMINAMATH_CALUDE_units_digit_of_six_to_fifth_l1057_105735

theorem units_digit_of_six_to_fifth (n : ℕ) : n = 6^5 → n % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_six_to_fifth_l1057_105735


namespace NUMINAMATH_CALUDE_change_percentage_l1057_105777

-- Define the prices of the items
def price1 : ℚ := 15.50
def price2 : ℚ := 3.25
def price3 : ℚ := 6.75

-- Define the amount paid
def amountPaid : ℚ := 50.00

-- Define the total price of items
def totalPrice : ℚ := price1 + price2 + price3

-- Define the change received
def change : ℚ := amountPaid - totalPrice

-- Define the percentage of change
def percentageChange : ℚ := (change / amountPaid) * 100

-- Theorem statement
theorem change_percentage : percentageChange = 49 := by
  sorry

end NUMINAMATH_CALUDE_change_percentage_l1057_105777


namespace NUMINAMATH_CALUDE_equal_negative_exponents_l1057_105728

theorem equal_negative_exponents : -2^3 = (-2)^3 ∧ 
  -3^2 ≠ -2^3 ∧ 
  (-3 * 2)^2 ≠ -3 * 2^2 ∧ 
  -3^2 ≠ (-3)^2 :=
by sorry

end NUMINAMATH_CALUDE_equal_negative_exponents_l1057_105728


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1057_105705

theorem right_triangle_side_length 
  (Q R S : ℝ × ℝ) 
  (right_angle_Q : (R.1 - Q.1) * (S.1 - Q.1) + (R.2 - Q.2) * (S.2 - Q.2) = 0) 
  (cos_R : (R.1 - Q.1) / Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = 4/9)
  (RS_length : Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 9) :
  Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) = Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1057_105705


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1057_105793

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + d * (n - 1)

theorem arithmetic_sequence_formula :
  ∀ n : ℕ, arithmetic_sequence (-1) 4 n = 4 * n - 5 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1057_105793


namespace NUMINAMATH_CALUDE_equation_solution_l1057_105797

theorem equation_solution : ∃ x : ℝ, (3 / x - 2 / (x + 1) = 0) ∧ (x = -3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1057_105797


namespace NUMINAMATH_CALUDE_total_wheels_count_l1057_105703

/-- Represents the total number of wheels in Jordan's driveway -/
def total_wheels : ℕ :=
  let cars := 2
  let car_wheels := 4
  let bikes := 3
  let bike_wheels := 2
  let trash_can_wheels := 2
  let tricycle_wheels := 3
  let roller_skate_wheels := 4
  let wheelchair_wheels := 6
  let wagon_wheels := 4
  
  cars * car_wheels +
  (bikes - 1) * bike_wheels + 1 +
  trash_can_wheels +
  tricycle_wheels +
  (roller_skate_wheels - 1) +
  wheelchair_wheels +
  wagon_wheels

theorem total_wheels_count : total_wheels = 31 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_count_l1057_105703


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l1057_105704

/-- The area of an isosceles right triangle with hypotenuse 6√2 is 18 square units -/
theorem isosceles_right_triangle_area (h : ℝ) (a : ℝ) 
  (hyp_length : h = 6 * Real.sqrt 2)
  (isosceles_right : a = h / Real.sqrt 2) : a * a / 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l1057_105704


namespace NUMINAMATH_CALUDE_system_solution_l1057_105757

theorem system_solution : ∃ (x y : ℝ), (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ∧ (x = 5) ∧ (y = 5) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1057_105757


namespace NUMINAMATH_CALUDE_transaction_result_l1057_105770

def initial_x : ℝ := 15000
def initial_y : ℝ := 18000
def painting_value : ℝ := 15000
def first_sale_price : ℝ := 20000
def second_sale_price : ℝ := 14000
def commission_rate : ℝ := 0.05

def first_transaction_x (initial : ℝ) (sale_price : ℝ) (commission : ℝ) : ℝ :=
  initial + sale_price * (1 - commission)

def first_transaction_y (initial : ℝ) (purchase_price : ℝ) : ℝ :=
  initial - purchase_price

def second_transaction_x (cash : ℝ) (purchase_price : ℝ) : ℝ :=
  cash - purchase_price

def second_transaction_y (cash : ℝ) (sale_price : ℝ) (commission : ℝ) : ℝ :=
  cash + sale_price * (1 - commission)

theorem transaction_result :
  let x_final := second_transaction_x (first_transaction_x initial_x first_sale_price commission_rate) second_sale_price
  let y_final := second_transaction_y (first_transaction_y initial_y first_sale_price) second_sale_price commission_rate
  (x_final - initial_x = 5000) ∧ (y_final - initial_y = -6700) :=
by sorry

end NUMINAMATH_CALUDE_transaction_result_l1057_105770


namespace NUMINAMATH_CALUDE_julie_income_calculation_l1057_105727

/-- Calculates Julie's net monthly income based on given conditions --/
def julies_net_monthly_income (
  starting_pay : ℝ)
  (experience_bonus : ℝ)
  (years_experience : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (biweekly_bonus : ℝ)
  (tax_rate : ℝ)
  (insurance_premium : ℝ)
  (missed_days : ℕ) : ℝ :=
  sorry

/-- Theorem stating that Julie's net monthly income is $963.20 --/
theorem julie_income_calculation :
  julies_net_monthly_income 5 0.5 3 8 6 50 0.12 40 1 = 963.20 :=
by sorry

end NUMINAMATH_CALUDE_julie_income_calculation_l1057_105727


namespace NUMINAMATH_CALUDE_equal_ratios_imply_k_value_l1057_105719

theorem equal_ratios_imply_k_value (x y z k : ℝ) 
  (h1 : 12 / (x + z) = k / (z - y))
  (h2 : k / (z - y) = 5 / (y - x))
  (h3 : y = 0) : k = 17 := by
  sorry

end NUMINAMATH_CALUDE_equal_ratios_imply_k_value_l1057_105719


namespace NUMINAMATH_CALUDE_z_axis_symmetry_of_M_l1057_105750

/-- A point in 3D Cartesian space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The z-axis symmetry operation on a 3D point -/
def zAxisSymmetry (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := p.z }

/-- The original point M -/
def M : Point3D :=
  { x := 3, y := -4, z := 5 }

/-- The expected symmetric point -/
def SymmetricPoint : Point3D :=
  { x := -3, y := 4, z := 5 }

theorem z_axis_symmetry_of_M :
  zAxisSymmetry M = SymmetricPoint := by sorry

end NUMINAMATH_CALUDE_z_axis_symmetry_of_M_l1057_105750


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_x_l1057_105712

theorem gcd_polynomial_and_x (x : ℤ) (h : ∃ k : ℤ, x = 23478 * k) :
  Int.gcd ((2*x+3)*(7*x+2)*(13*x+7)*(x+13)) x = 546 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_x_l1057_105712
