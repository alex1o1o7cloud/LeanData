import Mathlib

namespace NUMINAMATH_CALUDE_evaluate_logarithmic_expression_l2657_265720

theorem evaluate_logarithmic_expression :
  Real.sqrt (Real.log 8 / Real.log 3 - Real.log 8 / Real.log 2 + Real.log 8 / Real.log 4) =
  Real.sqrt (3 * (2 * Real.log 2 - Real.log 3)) / Real.sqrt (2 * Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_evaluate_logarithmic_expression_l2657_265720


namespace NUMINAMATH_CALUDE_four_books_weigh_one_kilogram_l2657_265795

-- Define the weight of one book in grams
def book_weight : ℕ := 250

-- Define the number of books
def num_books : ℕ := 4

-- Define the weight of a kilogram in grams
def kilogram_in_grams : ℕ := 1000

-- Theorem to prove
theorem four_books_weigh_one_kilogram :
  num_books * book_weight = kilogram_in_grams := by sorry

end NUMINAMATH_CALUDE_four_books_weigh_one_kilogram_l2657_265795


namespace NUMINAMATH_CALUDE_second_meeting_time_is_six_minutes_l2657_265700

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the pool and the swimming scenario --/
structure Pool where
  length : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  firstMeetingTime : ℝ
  firstMeetingPosition : ℝ

/-- Calculates the time of the second meeting --/
def secondMeetingTime (pool : Pool) : ℝ :=
  sorry

/-- Theorem stating the conditions and the result to be proved --/
theorem second_meeting_time_is_six_minutes 
  (pool : Pool)
  (h1 : pool.length = 120)
  (h2 : pool.swimmer1.startPosition = 0)
  (h3 : pool.swimmer2.startPosition = 120)
  (h4 : pool.firstMeetingPosition = 40)
  (h5 : pool.firstMeetingTime = 2) :
  secondMeetingTime pool = 6 := by
  sorry

end NUMINAMATH_CALUDE_second_meeting_time_is_six_minutes_l2657_265700


namespace NUMINAMATH_CALUDE_monotonic_functional_equation_solution_l2657_265703

-- Define a monotonic function f from real numbers to real numbers
def monotonic_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y ∨ ∀ x y, x ≤ y → f x ≥ f y

-- Define the functional equation
def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x * f y

-- Theorem statement
theorem monotonic_functional_equation_solution :
  ∀ f : ℝ → ℝ, monotonic_function f → functional_equation f →
  ∃ a : ℝ, (a > 1 ∨ 0 < a ∧ a < 1) ∧ ∀ x, f x = a^x :=
sorry

end NUMINAMATH_CALUDE_monotonic_functional_equation_solution_l2657_265703


namespace NUMINAMATH_CALUDE_problem_1_l2657_265789

theorem problem_1 : (1/2)⁻¹ - Real.tan (π/4) + |Real.sqrt 2 - 1| = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2657_265789


namespace NUMINAMATH_CALUDE_harry_says_1111_l2657_265776

/-- Represents a student in the counting game -/
inductive Student
| Adam
| Beth
| Claire
| Debby
| Eva
| Frank
| Gina
| Harry

/-- Defines the rules for each student's counting pattern -/
def countingRule (s : Student) : ℕ → Prop :=
  match s with
  | Student.Adam => λ n => n % 4 ≠ 0
  | Student.Beth => λ n => (n % 4 = 0) ∧ (n % 3 ≠ 2)
  | Student.Claire => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 ≠ 0)
  | Student.Debby => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 = 0) ∧ (n % 2 ≠ 0)
  | Student.Eva => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 ≠ 0)
  | Student.Frank => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 2 ≠ 0)
  | Student.Gina => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 ≠ 0)
  | Student.Harry => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 = 0)

/-- The theorem stating that Harry says the number 1111 -/
theorem harry_says_1111 : countingRule Student.Harry 1111 := by
  sorry

end NUMINAMATH_CALUDE_harry_says_1111_l2657_265776


namespace NUMINAMATH_CALUDE_book_cost_is_300_divided_by_num_books_l2657_265723

/-- Represents the cost of lawn mowing and video games -/
structure Costs where
  lawn_price : ℕ
  video_game_price : ℕ

/-- Represents Kenny's lawn mowing and purchasing activities -/
structure KennyActivities where
  costs : Costs
  lawns_mowed : ℕ
  video_games_bought : ℕ

/-- Calculates the cost of each book based on Kenny's activities -/
def book_cost (activities : KennyActivities) (num_books : ℕ) : ℚ :=
  let total_earned := activities.costs.lawn_price * activities.lawns_mowed
  let spent_on_games := activities.costs.video_game_price * activities.video_games_bought
  let remaining_for_books := total_earned - spent_on_games
  (remaining_for_books : ℚ) / num_books

/-- Theorem stating that the cost of each book is $300 divided by the number of books -/
theorem book_cost_is_300_divided_by_num_books 
  (activities : KennyActivities) 
  (num_books : ℕ) 
  (h1 : activities.costs.lawn_price = 15)
  (h2 : activities.costs.video_game_price = 45)
  (h3 : activities.lawns_mowed = 35)
  (h4 : activities.video_games_bought = 5)
  (h5 : num_books > 0) :
  book_cost activities num_books = 300 / num_books :=
by
  sorry

#check book_cost_is_300_divided_by_num_books

end NUMINAMATH_CALUDE_book_cost_is_300_divided_by_num_books_l2657_265723


namespace NUMINAMATH_CALUDE_smallest_common_factor_l2657_265714

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m < 57 → Nat.gcd (15*m - 9) (11*m + 10) = 1) ∧ 
  Nat.gcd (15*57 - 9) (11*57 + 10) > 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l2657_265714


namespace NUMINAMATH_CALUDE_minnie_horses_per_day_l2657_265727

theorem minnie_horses_per_day (mickey_weekly : ℕ) (days_per_week : ℕ) 
  (h1 : mickey_weekly = 98)
  (h2 : days_per_week = 7) :
  ∃ (minnie_daily : ℕ),
    (2 * minnie_daily - 6) * days_per_week = mickey_weekly ∧
    minnie_daily > days_per_week ∧
    minnie_daily - days_per_week = 3 := by
  sorry

end NUMINAMATH_CALUDE_minnie_horses_per_day_l2657_265727


namespace NUMINAMATH_CALUDE_treasure_day_l2657_265738

/-- Pongpong's starting amount -/
def pongpong_start : ℕ := 8000

/-- Longlong's starting amount -/
def longlong_start : ℕ := 5000

/-- Pongpong's daily increase -/
def pongpong_daily : ℕ := 300

/-- Longlong's daily increase -/
def longlong_daily : ℕ := 500

/-- The number of days until Pongpong and Longlong have the same amount -/
def days_until_equal : ℕ := 15

theorem treasure_day :
  pongpong_start + pongpong_daily * days_until_equal =
  longlong_start + longlong_daily * days_until_equal :=
by sorry

end NUMINAMATH_CALUDE_treasure_day_l2657_265738


namespace NUMINAMATH_CALUDE_prob_three_even_in_five_rolls_l2657_265765

/-- A fair 10-sided die -/
def TenSidedDie : Type := Fin 10

/-- The probability of rolling an even number on a 10-sided die -/
def probEven : ℚ := 1 / 2

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The number of dice we want to show even numbers -/
def numEven : ℕ := 3

/-- The probability of rolling exactly three even numbers when five fair 10-sided dice are rolled -/
theorem prob_three_even_in_five_rolls : 
  (numDice.choose numEven : ℚ) * probEven ^ numEven * (1 - probEven) ^ (numDice - numEven) = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_even_in_five_rolls_l2657_265765


namespace NUMINAMATH_CALUDE_cube_root_of_five_cubed_times_two_to_sixth_l2657_265726

theorem cube_root_of_five_cubed_times_two_to_sixth (x : ℝ) : x^3 = 5^3 * 2^6 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_five_cubed_times_two_to_sixth_l2657_265726


namespace NUMINAMATH_CALUDE_max_value_of_function_l2657_265750

theorem max_value_of_function (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x = x * (1 - x^2)) →
  (∃ x₀ ∈ Set.Icc 0 1, ∀ x ∈ Set.Icc 0 1, f x ≤ f x₀) →
  (∃ x₀ ∈ Set.Icc 0 1, f x₀ = 2 * Real.sqrt 3 / 9) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2657_265750


namespace NUMINAMATH_CALUDE_system_solution_l2657_265751

theorem system_solution (x y b : ℚ) : 
  (4 * x + 3 * y = b) →
  (3 * x + 4 * y = 3 * b) →
  (x = 3) →
  (b = -21 / 5) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2657_265751


namespace NUMINAMATH_CALUDE_wire_division_l2657_265779

/-- Given a wire of length 49 cm divided into 7 equal parts, prove that each part is 7 cm long -/
theorem wire_division (wire_length : ℝ) (num_parts : ℕ) (part_length : ℝ) 
  (h1 : wire_length = 49)
  (h2 : num_parts = 7)
  (h3 : part_length * num_parts = wire_length) :
  part_length = 7 := by
  sorry

end NUMINAMATH_CALUDE_wire_division_l2657_265779


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2657_265711

theorem fraction_evaluation : (5 * 7) / 10 = 3.5 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2657_265711


namespace NUMINAMATH_CALUDE_same_quotient_remainder_divisible_by_seven_l2657_265728

theorem same_quotient_remainder_divisible_by_seven :
  {n : ℕ | ∃ r : ℕ, 1 ≤ r ∧ r ≤ 6 ∧ n = 8 * r} = {8, 16, 24, 32, 40, 48} := by
sorry

end NUMINAMATH_CALUDE_same_quotient_remainder_divisible_by_seven_l2657_265728


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2657_265794

theorem simplify_trig_expression :
  1 / Real.sin (15 * π / 180) - 1 / Real.cos (15 * π / 180) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2657_265794


namespace NUMINAMATH_CALUDE_geometric_sum_of_powers_of_five_l2657_265774

theorem geometric_sum_of_powers_of_five : 
  (Finset.range 6).sum (fun i => 5^(i+1)) = 19530 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_of_powers_of_five_l2657_265774


namespace NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l2657_265758

theorem sin_40_tan_10_minus_sqrt_3 : 
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l2657_265758


namespace NUMINAMATH_CALUDE_work_project_solution_l2657_265782

/-- Represents a work project with a number of workers and days to complete. -/
structure WorkProject where
  workers : ℕ
  days : ℕ

/-- The condition when 2 workers are removed. -/
def condition1 (wp : WorkProject) : Prop :=
  (wp.workers - 2) * (wp.days + 4) = wp.workers * wp.days

/-- The condition when 3 workers are added. -/
def condition2 (wp : WorkProject) : Prop :=
  (wp.workers + 3) * (wp.days - 2) > wp.workers * wp.days

/-- The condition when 4 workers are added. -/
def condition3 (wp : WorkProject) : Prop :=
  (wp.workers + 4) * (wp.days - 3) > wp.workers * wp.days

/-- The main theorem stating the solution to the work project problem. -/
theorem work_project_solution :
  ∃ (wp : WorkProject),
    condition1 wp ∧
    condition2 wp ∧
    condition3 wp ∧
    wp.workers = 6 ∧
    wp.days = 8 := by
  sorry


end NUMINAMATH_CALUDE_work_project_solution_l2657_265782


namespace NUMINAMATH_CALUDE_permutations_of_eight_distinct_objects_l2657_265787

theorem permutations_of_eight_distinct_objects : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_eight_distinct_objects_l2657_265787


namespace NUMINAMATH_CALUDE_correct_pizza_dough_amounts_l2657_265763

/-- Calculates the required amounts of milk and water for a given amount of flour in Luca's pizza dough recipe. -/
def pizzaDoughCalculation (flourAmount : ℚ) : ℚ × ℚ :=
  let milkToFlourRatio : ℚ := 80 / 400
  let waterToMilkRatio : ℚ := 1 / 2
  let milkAmount : ℚ := flourAmount * milkToFlourRatio
  let waterAmount : ℚ := milkAmount * waterToMilkRatio
  (milkAmount, waterAmount)

/-- Theorem stating the correct amounts of milk and water for 1200 mL of flour. -/
theorem correct_pizza_dough_amounts :
  pizzaDoughCalculation 1200 = (240, 120) := by
  sorry

#eval pizzaDoughCalculation 1200

end NUMINAMATH_CALUDE_correct_pizza_dough_amounts_l2657_265763


namespace NUMINAMATH_CALUDE_find_twelfth_number_l2657_265725

/-- Given a set of 12 numbers where the sum of the first 11 is known and the arithmetic mean of all 12 is known, find the 12th number. -/
theorem find_twelfth_number (sum_first_eleven : ℕ) (arithmetic_mean : ℚ) (h1 : sum_first_eleven = 137) (h2 : arithmetic_mean = 12) :
  ∃ x : ℕ, (sum_first_eleven + x : ℚ) / 12 = arithmetic_mean ∧ x = 7 :=
by sorry

end NUMINAMATH_CALUDE_find_twelfth_number_l2657_265725


namespace NUMINAMATH_CALUDE_initial_bushes_count_l2657_265724

/-- The number of new bushes that grow between each pair of neighboring bushes every hour. -/
def new_bushes_per_hour : ℕ := 2

/-- The total number of hours of growth. -/
def total_hours : ℕ := 3

/-- The total number of bushes after the growth period. -/
def final_bush_count : ℕ := 190

/-- Calculate the number of bushes after one hour of growth. -/
def bushes_after_one_hour (initial_bushes : ℕ) : ℕ :=
  initial_bushes + new_bushes_per_hour * (initial_bushes - 1)

/-- Calculate the number of bushes after the total growth period. -/
def bushes_after_growth (initial_bushes : ℕ) : ℕ :=
  (bushes_after_one_hour^[total_hours]) initial_bushes

/-- The theorem stating that 8 is the correct initial number of bushes. -/
theorem initial_bushes_count : 
  ∃ (n : ℕ), n > 0 ∧ bushes_after_growth n = final_bush_count ∧ 
  ∀ (m : ℕ), m ≠ n → bushes_after_growth m ≠ final_bush_count :=
sorry

end NUMINAMATH_CALUDE_initial_bushes_count_l2657_265724


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2657_265766

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℕ+, (x ≤ 4 → a * x.val + 4 ≥ 0) ∧ (x > 4 → a * x.val + 4 < 0)) → 
  -1 ≤ a ∧ a < -4/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2657_265766


namespace NUMINAMATH_CALUDE_complex_power_six_l2657_265752

theorem complex_power_six : (2 + 3*I : ℂ)^6 = -845 + 2028*I := by sorry

end NUMINAMATH_CALUDE_complex_power_six_l2657_265752


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2657_265721

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - 3*x + 5 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^2 - 3*x₀ + 5 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2657_265721


namespace NUMINAMATH_CALUDE_polyhedron_problem_l2657_265796

/-- Represents a convex polyhedron with hexagonal and quadrilateral faces. -/
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  hexagons : ℕ
  quadrilaterals : ℕ
  H : ℕ
  Q : ℕ

/-- Euler's formula for convex polyhedra -/
def euler_formula (p : Polyhedron) : Prop :=
  p.vertices - p.edges + p.faces = 2

/-- The number of edges in terms of hexagons and quadrilaterals -/
def edge_count (p : Polyhedron) : Prop :=
  p.edges = 2 * p.quadrilaterals + 3 * p.hexagons

/-- Theorem about the specific polyhedron described in the problem -/
theorem polyhedron_problem :
  ∀ p : Polyhedron,
    p.faces = 44 →
    p.hexagons = 12 →
    p.quadrilaterals = 32 →
    p.H = 2 →
    p.Q = 2 →
    euler_formula p →
    edge_count p →
    100 * p.H + 10 * p.Q + p.vertices = 278 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_problem_l2657_265796


namespace NUMINAMATH_CALUDE_area_of_curve_l2657_265734

/-- The curve defined by the equation x^2 + y^2 = 2(|x| + |y|) -/
def curve (x y : ℝ) : Prop := x^2 + y^2 = 2 * (abs x + abs y)

/-- The region enclosed by the curve -/
def enclosed_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ curve x y}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

theorem area_of_curve : area enclosed_region = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_of_curve_l2657_265734


namespace NUMINAMATH_CALUDE_angle_measure_theorem_l2657_265739

theorem angle_measure_theorem (x : ℝ) : 
  (90 - x) = (180 - x) - 4 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_theorem_l2657_265739


namespace NUMINAMATH_CALUDE_short_trees_planted_l2657_265757

theorem short_trees_planted (initial_short : ℕ) (final_short : ℕ) :
  initial_short = 31 →
  final_short = 95 →
  final_short - initial_short = 64 := by
sorry

end NUMINAMATH_CALUDE_short_trees_planted_l2657_265757


namespace NUMINAMATH_CALUDE_intersection_points_l2657_265793

-- Define the equations
def eq1 (x y : ℝ) : Prop := 4 + (x + 2) * y = x^2
def eq2 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 16

-- Theorem stating the intersection points
theorem intersection_points :
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    (x1 = -2 ∧ y1 = -4) ∧
    (x2 = -2 ∧ y2 = 4) ∧
    (x3 = 2 ∧ y3 = 0) ∧
    eq1 x1 y1 ∧ eq2 x1 y1 ∧
    eq1 x2 y2 ∧ eq2 x2 y2 ∧
    eq1 x3 y3 ∧ eq2 x3 y3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_l2657_265793


namespace NUMINAMATH_CALUDE_system_solution_l2657_265702

theorem system_solution (x y : ℝ) : 
  (x / y + y / x = 173 / 26 ∧ 1 / x + 1 / y = 15 / 26) → 
  ((x = 13 ∧ y = 2) ∨ (x = 2 ∧ y = 13)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2657_265702


namespace NUMINAMATH_CALUDE_second_month_sale_l2657_265701

/-- Given the sales of a grocer for four months, prove that the sale in the second month is 4000 --/
theorem second_month_sale
  (sale1 : ℕ)
  (sale3 : ℕ)
  (sale4 : ℕ)
  (average : ℕ)
  (h1 : sale1 = 2500)
  (h2 : sale3 = 3540)
  (h3 : sale4 = 1520)
  (h4 : average = 2890)
  (h5 : (sale1 + sale3 + sale4 + (4 * average - sale1 - sale3 - sale4)) / 4 = average) :
  4 * average - sale1 - sale3 - sale4 = 4000 := by
sorry

end NUMINAMATH_CALUDE_second_month_sale_l2657_265701


namespace NUMINAMATH_CALUDE_incenter_distance_l2657_265735

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  let pq := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let qr := Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2)
  let rp := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  pq = 17 ∧ qr = 15 ∧ rp = 8

-- Define the incenter
def Incenter (P Q R J : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
  (J.1 - P.1)^2 + (J.2 - P.2)^2 = r^2 ∧
  (J.1 - Q.1)^2 + (J.2 - Q.2)^2 = r^2 ∧
  (J.1 - R.1)^2 + (J.2 - R.2)^2 = r^2

-- Theorem statement
theorem incenter_distance (P Q R J : ℝ × ℝ) :
  Triangle P Q R → Incenter P Q R J →
  (J.1 - P.1)^2 + (J.2 - P.2)^2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_incenter_distance_l2657_265735


namespace NUMINAMATH_CALUDE_log_inequality_l2657_265778

theorem log_inequality (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (Real.log 3 / Real.log m < Real.log 3 / Real.log n) ∧ (Real.log 3 / Real.log n < 0) →
  1 > m ∧ m > n ∧ n > 0 := by
sorry

end NUMINAMATH_CALUDE_log_inequality_l2657_265778


namespace NUMINAMATH_CALUDE_problem_statement_l2657_265737

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → a * b ≤ m) → m ≥ 1/4) ∧
  (∀ x : ℝ, (1/a + 1/b ≥ |2*x - 1| - |x + 1|) ↔ -2 ≤ x ∧ x ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2657_265737


namespace NUMINAMATH_CALUDE_range_of_a_l2657_265718

/-- The proposition p: The equation ax^2 + ax - 2 = 0 has a solution on the interval [-1, 1] -/
def p (a : ℝ) : Prop := ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a * x^2 + a * x - 2 = 0

/-- The proposition q: There is exactly one real number x such that x^2 + 2ax + 2a ≤ 0 -/
def q (a : ℝ) : Prop := ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

/-- The function f(x) = ax^2 + ax - 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x - 2

theorem range_of_a (a : ℝ) :
  (¬(p a ∨ q a)) ∧ (f a (-1) = -2) ∧ (f a 0 = -2) →
  (-8 < a ∧ a < 0) ∨ (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2657_265718


namespace NUMINAMATH_CALUDE_new_average_after_22_innings_l2657_265715

def calculate_new_average (initial_innings : ℕ) (score_17th : ℕ) (average_increase : ℕ) 
  (additional_scores : List ℕ) : ℕ :=
  let total_innings := initial_innings + additional_scores.length
  let initial_average := (initial_innings - 1) * (average_increase + 1) / initial_innings
  let total_runs_17 := initial_innings * (initial_average + average_increase)
  let total_runs_22 := total_runs_17 + additional_scores.sum
  total_runs_22 / total_innings

theorem new_average_after_22_innings : 
  calculate_new_average 17 85 3 [100, 120, 45, 75, 65] = 47 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_22_innings_l2657_265715


namespace NUMINAMATH_CALUDE_range_of_k_for_quadratic_inequality_l2657_265729

theorem range_of_k_for_quadratic_inequality :
  {k : ℝ | ∀ x : ℝ, k * x^2 - k * x - 1 < 0} = {k : ℝ | -4 < k ∧ k ≤ 0} :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_for_quadratic_inequality_l2657_265729


namespace NUMINAMATH_CALUDE_triangle_area_l2657_265772

theorem triangle_area (a c B : ℝ) (h1 : a = 1) (h2 : c = 2) (h3 : B = Real.pi / 3) :
  (1/2) * a * c * Real.sin B = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2657_265772


namespace NUMINAMATH_CALUDE_max_area_rectangular_enclosure_l2657_265712

/-- Given a rectangular area with perimeter P (excluding one side) and length L twice the width W,
    the maximum area A is (P/4)^2 square units. -/
theorem max_area_rectangular_enclosure (P : ℝ) (h : P > 0) :
  let W := P / 4
  let L := 2 * W
  let A := L * W
  A = (P / 4) ^ 2 := by
  sorry

#check max_area_rectangular_enclosure

end NUMINAMATH_CALUDE_max_area_rectangular_enclosure_l2657_265712


namespace NUMINAMATH_CALUDE_parallelogram_diagonals_bisect_is_universal_quantifier_parallelogram_diagonals_theorem_l2657_265731

-- Define what a parallelogram is
structure Parallelogram where
  points : Fin 4 → ℝ × ℝ
  is_parallelogram : sorry

-- Define what a diagonal is
def diagonal (p : Parallelogram) (i j : Fin 4) : ℝ × ℝ := sorry

-- Define what it means for two line segments to bisect each other
def bisect (seg1 seg2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

-- The theorem to be proved
theorem parallelogram_diagonals_bisect (p : Parallelogram) :
  bisect (diagonal p 0 2, diagonal p 1 3) (diagonal p 1 3, diagonal p 0 2) :=
sorry

-- The statement is a universal quantifier
theorem is_universal_quantifier : 
  ∀ (p : Parallelogram), 
    bisect (diagonal p 0 2, diagonal p 1 3) (diagonal p 1 3, diagonal p 0 2) :=
sorry

-- The combined theorem
theorem parallelogram_diagonals_theorem :
  (∀ (p : Parallelogram), bisect (diagonal p 0 2, diagonal p 1 3) (diagonal p 1 3, diagonal p 0 2)) ∧
  (∃ (p : Parallelogram), bisect (diagonal p 0 2, diagonal p 1 3) (diagonal p 1 3, diagonal p 0 2)) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_diagonals_bisect_is_universal_quantifier_parallelogram_diagonals_theorem_l2657_265731


namespace NUMINAMATH_CALUDE_product_seventeen_reciprocal_squares_sum_l2657_265773

theorem product_seventeen_reciprocal_squares_sum (x y : ℕ) :
  x * y = 17 → (1 : ℚ) / x^2 + 1 / y^2 = 290 / 289 := by
  sorry

end NUMINAMATH_CALUDE_product_seventeen_reciprocal_squares_sum_l2657_265773


namespace NUMINAMATH_CALUDE_kangaroo_exhibition_arrangements_l2657_265716

/-- The number of ways to arrange n uniquely tall kangaroos in a row -/
def kangaroo_arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n uniquely tall kangaroos in a row,
    with the two tallest at the ends -/
def kangaroo_arrangements_with_tallest_at_ends (n : ℕ) : ℕ :=
  2 * kangaroo_arrangements (n - 2)

theorem kangaroo_exhibition_arrangements :
  kangaroo_arrangements_with_tallest_at_ends 8 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_exhibition_arrangements_l2657_265716


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2657_265736

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 4 + a 6 = 3 →               -- given condition
  a 4^2 + 2*a 4*a 6 + a 5*a 7 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2657_265736


namespace NUMINAMATH_CALUDE_inscribed_square_area_equals_rectangle_area_l2657_265706

/-- Given a right triangle with legs a and b, and a square with side length x
    inscribed such that one angle coincides with the right angle of the triangle
    and one vertex lies on the hypotenuse, the area of the square is equal to
    the area of the rectangle formed by the remaining segments of the legs. -/
theorem inscribed_square_area_equals_rectangle_area 
  (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : 0 < x ∧ x < min a b) : 
  x^2 = (a - x) * (b - x) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_equals_rectangle_area_l2657_265706


namespace NUMINAMATH_CALUDE_distribute_five_to_three_l2657_265707

def distribute_objects (n : ℕ) (k : ℕ) : ℕ :=
  let c311 := (n.choose 3) * ((n - 3).choose 1) * ((n - 4).choose 1) / 2
  let c221 := (n.choose 2) * ((n - 2).choose 2) * ((n - 4).choose 1) / 2
  (c311 + c221) * 6

theorem distribute_five_to_three :
  distribute_objects 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_to_three_l2657_265707


namespace NUMINAMATH_CALUDE_volume_maximized_at_one_l2657_265764

/-- The volume function of the lidless square box -/
def V (x : ℝ) : ℝ := x * (6 - 2*x)^2

/-- The derivative of the volume function -/
def V' (x : ℝ) : ℝ := 12*x^2 - 48*x + 36

theorem volume_maximized_at_one :
  ∀ x ∈ Set.Ioo 0 3, V x ≤ V 1 :=
sorry

end NUMINAMATH_CALUDE_volume_maximized_at_one_l2657_265764


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l2657_265754

theorem least_integer_absolute_value (x : ℤ) : (∀ y : ℤ, |3*y + 4| ≤ 18 → y ≥ -7) ∧ |3*(-7) + 4| ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l2657_265754


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_l2657_265719

-- Define the function f
def f (a x : ℝ) : ℝ := |2*x - a| + |2*x - 1|

-- Part I
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x ≤ 6} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 5/2} :=
sorry

-- Part II
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ a^2 - a - 13) ↔ 
  (a ≥ -Real.sqrt 14 ∧ a ≤ 1 + Real.sqrt 13) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_l2657_265719


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l2657_265705

theorem max_value_expression (x y : ℝ) :
  (x + 2*y + 3) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 14 :=
sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (x + 2*y + 3) / Real.sqrt (x^2 + y^2 + 1) = Real.sqrt 14 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l2657_265705


namespace NUMINAMATH_CALUDE_job_completion_time_l2657_265730

/-- Represents the job completion scenario with changing number of workers -/
structure JobCompletion where
  initial_workers : ℕ
  initial_days : ℕ
  work_days_before_change : ℕ
  additional_workers : ℕ
  total_days : ℚ

/-- Theorem stating that under the given conditions, the job will be completed in 3.5 days -/
theorem job_completion_time (job : JobCompletion) :
  job.initial_workers = 6 ∧
  job.initial_days = 8 ∧
  job.work_days_before_change = 3 ∧
  job.additional_workers = 4 →
  job.total_days = 3.5 := by
  sorry

#check job_completion_time

end NUMINAMATH_CALUDE_job_completion_time_l2657_265730


namespace NUMINAMATH_CALUDE_divisibility_implies_gcd_greater_than_one_l2657_265740

theorem divisibility_implies_gcd_greater_than_one
  (a b c d : ℕ+)
  (h : (a.val * c.val + b.val * d.val) % (a.val^2 + b.val^2) = 0) :
  Nat.gcd (c.val^2 + d.val^2) (a.val^2 + b.val^2) > 1 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implies_gcd_greater_than_one_l2657_265740


namespace NUMINAMATH_CALUDE_paint_usage_fraction_l2657_265771

theorem paint_usage_fraction (total_paint : ℝ) (paint_used_total : ℝ) :
  total_paint = 360 →
  paint_used_total = 168 →
  let paint_used_first_week := total_paint / 3
  let paint_remaining := total_paint - paint_used_first_week
  let paint_used_second_week := paint_used_total - paint_used_first_week
  paint_used_second_week / paint_remaining = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_paint_usage_fraction_l2657_265771


namespace NUMINAMATH_CALUDE_polynomial_coefficient_bound_l2657_265710

theorem polynomial_coefficient_bound (a b c d : ℝ) : 
  (∀ x : ℝ, |x| < 1 → |a * x^3 + b * x^2 + c * x + d| ≤ 1) → 
  |a| + |b| + |c| + |d| ≤ 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_bound_l2657_265710


namespace NUMINAMATH_CALUDE_common_root_of_polynomials_l2657_265786

theorem common_root_of_polynomials :
  let p₁ (x : ℚ) := 3*x^4 + 13*x^3 + 20*x^2 + 17*x + 7
  let p₂ (x : ℚ) := 3*x^4 + x^3 - 8*x^2 + 11*x - 7
  p₁ (-7/3) = 0 ∧ p₂ (-7/3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_common_root_of_polynomials_l2657_265786


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2657_265775

theorem simplify_sqrt_expression (x : ℝ) (h : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 1) / (2*x^3))^2) = (x^3 / 2) + (1 / (2*x^3)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2657_265775


namespace NUMINAMATH_CALUDE_mary_fruits_left_l2657_265742

/-- The number of apples Mary bought -/
def apples : Nat := 14

/-- The number of oranges Mary bought -/
def oranges : Nat := 9

/-- The number of blueberries Mary bought -/
def blueberries : Nat := 6

/-- The number of each type of fruit Mary ate -/
def eaten : Nat := 1

/-- The total number of fruits Mary has left -/
def fruits_left : Nat := (apples - eaten) + (oranges - eaten) + (blueberries - eaten)

theorem mary_fruits_left : fruits_left = 26 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruits_left_l2657_265742


namespace NUMINAMATH_CALUDE_non_monotonic_range_l2657_265756

/-- A function f is not monotonic on an interval if there exists a point in the interval where f' is zero --/
def NotMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x ∈ Set.Ioo a b, deriv f x = 0

/-- The main theorem --/
theorem non_monotonic_range (k : ℝ) :
  NotMonotonic (fun x => x^3 - 12*x) (k - 1) (k + 1) →
  k ∈ Set.union (Set.Ioo (-3) (-1)) (Set.Ioo 1 3) := by
  sorry

end NUMINAMATH_CALUDE_non_monotonic_range_l2657_265756


namespace NUMINAMATH_CALUDE_no_good_number_l2657_265745

def sum_of_digits (n : ℕ) : ℕ := sorry

def is_divisible_by_sum_of_digits (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem no_good_number :
  ¬ ∃ n : ℕ, 
    is_divisible_by_sum_of_digits n ∧
    is_divisible_by_sum_of_digits (n + 1) ∧
    is_divisible_by_sum_of_digits (n + 2) ∧
    is_divisible_by_sum_of_digits (n + 3) :=
sorry

end NUMINAMATH_CALUDE_no_good_number_l2657_265745


namespace NUMINAMATH_CALUDE_sum_of_third_and_fifth_layers_l2657_265746

-- Define the number of balls in the n-th layer of the pyramid
def balls_in_layer (n : ℕ) : ℕ := n^2

-- State the theorem
theorem sum_of_third_and_fifth_layers :
  balls_in_layer 3 + balls_in_layer 5 = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_third_and_fifth_layers_l2657_265746


namespace NUMINAMATH_CALUDE_translated_point_coordinates_l2657_265768

-- Define the points in the 2D plane
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, 3)
def A' : ℝ × ℝ := (2, 1)

-- Define the translation vector
def translation_vector : ℝ × ℝ := (A'.1 - A.1, A'.2 - A.2)

-- Define the translated point B'
def B' : ℝ × ℝ := (B.1 + translation_vector.1, B.2 + translation_vector.2)

-- Theorem statement
theorem translated_point_coordinates :
  B' = (4, 4) := by sorry

end NUMINAMATH_CALUDE_translated_point_coordinates_l2657_265768


namespace NUMINAMATH_CALUDE_caitlin_age_l2657_265783

/-- Proves that Caitlin's age is 13 years given the ages of Aunt Anna and Brianna -/
theorem caitlin_age (anna_age : ℕ) (brianna_age : ℕ) (caitlin_age : ℕ)
  (h1 : anna_age = 60)
  (h2 : brianna_age = anna_age / 3)
  (h3 : caitlin_age = brianna_age - 7) :
  caitlin_age = 13 := by
sorry

end NUMINAMATH_CALUDE_caitlin_age_l2657_265783


namespace NUMINAMATH_CALUDE_polynomial_value_l2657_265743

theorem polynomial_value (x y : ℝ) (h : x - y = 1) :
  x^4 - x*y^3 - x^3*y - 3*x^2*y + 3*x*y^2 + y^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l2657_265743


namespace NUMINAMATH_CALUDE_parabola_vertex_l2657_265769

/-- The vertex of the parabola y = -3x^2 + 6x + 4 is (1, 7) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -3 * x^2 + 6 * x + 4 → (1, 7) = (x, y) ∧ ∀ (x' : ℝ), y ≥ -3 * x'^2 + 6 * x' + 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2657_265769


namespace NUMINAMATH_CALUDE_simplify_polynomial_expression_l2657_265753

theorem simplify_polynomial_expression (x : ℝ) : 
  3 * ((5 * x^2 - 4 * x + 8) - (3 * x^2 - 2 * x + 6)) = 6 * x^2 - 6 * x + 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_expression_l2657_265753


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2657_265749

theorem quadratic_equation_solution :
  ∀ x : ℝ, x > 0 → (7 * x^2 - 8 * x - 6 = 0) → (x = 6/7 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2657_265749


namespace NUMINAMATH_CALUDE_quadratic_sum_l2657_265791

/-- A quadratic function f(x) = px^2 + qx + r with vertex (-3, 4) passing through (0, 1) -/
def QuadraticFunction (p q r : ℝ) : ℝ → ℝ := fun x ↦ p * x^2 + q * x + r

/-- The vertex of the quadratic function -/
def vertex (p q r : ℝ) : ℝ × ℝ := (-3, 4)

/-- The function passes through the point (0, 1) -/
def passes_through_origin (p q r : ℝ) : Prop :=
  QuadraticFunction p q r 0 = 1

theorem quadratic_sum (p q r : ℝ) :
  vertex p q r = (-3, 4) →
  passes_through_origin p q r →
  p + q + r = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2657_265791


namespace NUMINAMATH_CALUDE_billys_coin_piles_l2657_265709

theorem billys_coin_piles (x : ℕ) : 
  (x + 3) * 4 = 20 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_billys_coin_piles_l2657_265709


namespace NUMINAMATH_CALUDE_arithmetic_sequence_log_theorem_l2657_265788

-- Define the logarithm base 2
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

-- State the theorem
theorem arithmetic_sequence_log_theorem (x : ℝ) :
  is_arithmetic_sequence (lg 2) (lg (2^x - 1)) (lg (2^x + 3)) →
  x = Real.log 5 / Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_log_theorem_l2657_265788


namespace NUMINAMATH_CALUDE_quadratic_sum_l2657_265767

/-- Given a quadratic expression 4x^2 - 8x + 1, when expressed in the form a(x-h)^2 + k,
    the sum of a, h, and k equals 2. -/
theorem quadratic_sum (a h k : ℝ) : 
  (∀ x, 4 * x^2 - 8 * x + 1 = a * (x - h)^2 + k) → a + h + k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2657_265767


namespace NUMINAMATH_CALUDE_adam_lawn_mowing_earnings_l2657_265798

/-- Adam's lawn mowing earnings problem -/
theorem adam_lawn_mowing_earnings 
  (dollars_per_lawn : ℕ) 
  (total_lawns : ℕ) 
  (forgotten_lawns : ℕ) 
  (h1 : dollars_per_lawn = 9)
  (h2 : total_lawns = 12)
  (h3 : forgotten_lawns = 8)
  : (total_lawns - forgotten_lawns) * dollars_per_lawn = 36 := by
  sorry

end NUMINAMATH_CALUDE_adam_lawn_mowing_earnings_l2657_265798


namespace NUMINAMATH_CALUDE_negation_of_exists_is_forall_l2657_265792

theorem negation_of_exists_is_forall :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_is_forall_l2657_265792


namespace NUMINAMATH_CALUDE_expected_defective_60000_l2657_265781

/-- Represents a shipment of computer chips -/
structure Shipment where
  defective : ℕ
  total : ℕ

/-- Calculates the expected number of defective chips in a future shipment -/
def expectedDefective (shipments : List Shipment) (futureTotal : ℕ) : ℕ :=
  let totalDefective := shipments.map (·.defective) |>.sum
  let totalChips := shipments.map (·.total) |>.sum
  (totalDefective * futureTotal) / totalChips

/-- Theorem stating the expected number of defective chips in a shipment of 60,000 -/
theorem expected_defective_60000 (shipments : List Shipment) 
    (h1 : shipments = [
      ⟨2, 5000⟩, 
      ⟨4, 12000⟩, 
      ⟨2, 15000⟩, 
      ⟨4, 16000⟩
    ]) : 
    expectedDefective shipments 60000 = 15 := by
  sorry

end NUMINAMATH_CALUDE_expected_defective_60000_l2657_265781


namespace NUMINAMATH_CALUDE_consecutive_sum_theorem_l2657_265755

theorem consecutive_sum_theorem (n : ℕ) (h : n ≥ 6) :
  ∃ (k a : ℕ), k ≥ 3 ∧ n = k * a + k * (k - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_sum_theorem_l2657_265755


namespace NUMINAMATH_CALUDE_range_of_m_l2657_265785

theorem range_of_m (x : ℝ) :
  (∀ x, (1/3 < x ∧ x < 1/2) → (m - 1 < x ∧ x < m + 1)) →
  (-1/2 ≤ m ∧ m ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2657_265785


namespace NUMINAMATH_CALUDE_carol_invitation_packs_l2657_265759

theorem carol_invitation_packs (invitations_per_pack : ℕ) (total_invitations : ℕ) (h1 : invitations_per_pack = 9) (h2 : total_invitations = 45) :
  total_invitations / invitations_per_pack = 5 :=
by sorry

end NUMINAMATH_CALUDE_carol_invitation_packs_l2657_265759


namespace NUMINAMATH_CALUDE_equal_reading_time_l2657_265770

/-- The total number of pages in the novel -/
def total_pages : ℕ := 760

/-- Bob's reading speed in seconds per page -/
def bob_speed : ℕ := 45

/-- Chandra's reading speed in seconds per page -/
def chandra_speed : ℕ := 30

/-- The number of pages Chandra reads -/
def chandra_pages : ℕ := 456

/-- The number of pages Bob reads -/
def bob_pages : ℕ := total_pages - chandra_pages

theorem equal_reading_time : chandra_speed * chandra_pages = bob_speed * bob_pages := by
  sorry

end NUMINAMATH_CALUDE_equal_reading_time_l2657_265770


namespace NUMINAMATH_CALUDE_number_of_students_in_section_B_l2657_265777

theorem number_of_students_in_section_B (students_A : ℕ) (avg_weight_A : ℚ) (avg_weight_B : ℚ) (avg_weight_total : ℚ) :
  students_A = 26 →
  avg_weight_A = 50 →
  avg_weight_B = 30 →
  avg_weight_total = 38.67 →
  ∃ (students_B : ℕ), 
    (students_A * avg_weight_A + students_B * avg_weight_B : ℚ) / (students_A + students_B : ℚ) = avg_weight_total ∧
    students_B = 34 :=
by sorry

end NUMINAMATH_CALUDE_number_of_students_in_section_B_l2657_265777


namespace NUMINAMATH_CALUDE_hexagon_not_to_quadrilateral_other_polygons_to_quadrilateral_l2657_265741

-- Define a polygon type
inductive Polygon
| triangle : Polygon
| quadrilateral : Polygon
| pentagon : Polygon
| hexagon : Polygon

-- Define a function that represents cutting off one angle
def cutOffAngle (p : Polygon) : Polygon :=
  match p with
  | Polygon.triangle => Polygon.triangle  -- Assuming it remains a triangle
  | Polygon.quadrilateral => Polygon.triangle
  | Polygon.pentagon => Polygon.quadrilateral
  | Polygon.hexagon => Polygon.pentagon

-- Theorem stating that a hexagon cannot become a quadrilateral by cutting off one angle
theorem hexagon_not_to_quadrilateral :
  ∀ (p : Polygon), p = Polygon.hexagon → cutOffAngle p ≠ Polygon.quadrilateral :=
by sorry

-- Theorem stating that other polygons can potentially become a quadrilateral
theorem other_polygons_to_quadrilateral :
  ∃ (p : Polygon), p ≠ Polygon.hexagon ∧ (cutOffAngle p = Polygon.quadrilateral ∨ p = Polygon.quadrilateral) :=
by sorry

end NUMINAMATH_CALUDE_hexagon_not_to_quadrilateral_other_polygons_to_quadrilateral_l2657_265741


namespace NUMINAMATH_CALUDE_counting_units_theorem_l2657_265732

/-- The progression rate between adjacent counting units -/
def progression_rate : ℕ := 10

/-- The number of ten millions in one hundred million -/
def ten_millions_in_hundred_million : ℕ := progression_rate

/-- The number of hundred thousands in one million -/
def hundred_thousands_in_million : ℕ := progression_rate

theorem counting_units_theorem :
  ten_millions_in_hundred_million = 10 ∧
  hundred_thousands_in_million = 10 := by
  sorry

end NUMINAMATH_CALUDE_counting_units_theorem_l2657_265732


namespace NUMINAMATH_CALUDE_walking_speed_problem_l2657_265717

/-- Proves that the speed at which a person would have walked is 10 km/hr,
    given the conditions of the problem. -/
theorem walking_speed_problem (actual_distance : ℝ) (additional_distance : ℝ) 
  (actual_speed : ℝ) :
  actual_distance = 20 →
  additional_distance = 20 →
  actual_speed = 5 →
  ∃ (speed : ℝ),
    speed = actual_speed + 5 ∧
    actual_distance / actual_speed = (actual_distance + additional_distance) / speed ∧
    speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l2657_265717


namespace NUMINAMATH_CALUDE_xyz_sum_lower_bound_l2657_265784

theorem xyz_sum_lower_bound (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (h : x * y * z + x * y + y * z + z * x = 4) : 
  x + y + z ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_lower_bound_l2657_265784


namespace NUMINAMATH_CALUDE_range_of_a_l2657_265722

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc (-π/3) a, f x = Real.sin (x + π/6)) →
  Set.range f = Set.Icc (-1/2) 1 →
  a ∈ Set.Icc (π/3) π :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2657_265722


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2657_265790

/-- The number of games in a chess tournament -/
def tournament_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  (n * (n - 1) / 2) * games_per_pair

/-- Theorem: In a chess tournament with 30 players, where each player plays
    four times with each opponent, the total number of games is 1740 -/
theorem chess_tournament_games :
  tournament_games 30 4 = 1740 := by
  sorry

#eval tournament_games 30 4

end NUMINAMATH_CALUDE_chess_tournament_games_l2657_265790


namespace NUMINAMATH_CALUDE_total_sewing_time_is_40_hours_l2657_265761

/-- Represents the time (in hours) to sew a given number of items -/
def sewing_time (time_per_item : ℝ) (num_items : ℕ) : ℝ :=
  time_per_item * (num_items : ℝ)

/-- Proves that the total sewing time for skirts and coats is 40 hours -/
theorem total_sewing_time_is_40_hours 
  (skirt_time : ℝ) 
  (coat_time : ℝ) 
  (num_skirts : ℕ) 
  (num_coats : ℕ) 
  (h1 : skirt_time = 2)
  (h2 : coat_time = 7)
  (h3 : num_skirts = 6)
  (h4 : num_coats = 4) : 
  sewing_time skirt_time num_skirts + sewing_time coat_time num_coats = 40 := by
  sorry

#check total_sewing_time_is_40_hours

end NUMINAMATH_CALUDE_total_sewing_time_is_40_hours_l2657_265761


namespace NUMINAMATH_CALUDE_merry_go_round_time_l2657_265733

theorem merry_go_round_time (dave chuck erica : ℝ) : 
  chuck = 5 * dave →
  erica = chuck + 0.3 * chuck →
  erica = 65 →
  dave = 10 :=
by sorry

end NUMINAMATH_CALUDE_merry_go_round_time_l2657_265733


namespace NUMINAMATH_CALUDE_fifty_cows_fifty_bags_l2657_265713

/-- The number of bags of husk eaten by a group of cows over a fixed period -/
def bagsEaten (numCows : ℕ) (daysPerBag : ℕ) (totalDays : ℕ) : ℕ :=
  numCows * (totalDays / daysPerBag)

/-- Theorem: 50 cows eat 50 bags of husk in 50 days -/
theorem fifty_cows_fifty_bags :
  bagsEaten 50 50 50 = 50 := by
  sorry

end NUMINAMATH_CALUDE_fifty_cows_fifty_bags_l2657_265713


namespace NUMINAMATH_CALUDE_sigma_phi_inequality_l2657_265708

open Nat

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- A natural number is prime if it has exactly two divisors -/
def isPrime (n : ℕ) : Prop := sorry

theorem sigma_phi_inequality (n : ℕ) (h : n > 1) :
  sigma n * phi n ≤ n^2 - 1 ∧ (sigma n * phi n = n^2 - 1 ↔ isPrime n) := by sorry

end NUMINAMATH_CALUDE_sigma_phi_inequality_l2657_265708


namespace NUMINAMATH_CALUDE_identity_function_unique_l2657_265744

def PositiveNat := {n : ℕ // n > 0}

theorem identity_function_unique 
  (f : PositiveNat → PositiveNat) 
  (h : ∀ (m n : PositiveNat), ∃ (k : ℕ), k * (m.val^2 + (f n).val) = m.val * (f m).val + n.val) : 
  ∀ (n : PositiveNat), f n = n :=
sorry

end NUMINAMATH_CALUDE_identity_function_unique_l2657_265744


namespace NUMINAMATH_CALUDE_probability_one_of_each_l2657_265747

/-- The number of t-shirts in the wardrobe -/
def num_tshirts : ℕ := 3

/-- The number of pairs of jeans in the wardrobe -/
def num_jeans : ℕ := 7

/-- The number of hats in the wardrobe -/
def num_hats : ℕ := 4

/-- The total number of clothing items in the wardrobe -/
def total_items : ℕ := num_tshirts + num_jeans + num_hats

/-- The probability of selecting one t-shirt, one pair of jeans, and one hat -/
theorem probability_one_of_each : 
  (num_tshirts * num_jeans * num_hats : ℚ) / (total_items.choose 3) = 21 / 91 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_of_each_l2657_265747


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2657_265760

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (-3, 1/2) and (7, 9) is equal to 6.75 -/
theorem midpoint_coordinate_sum : 
  let x₁ : ℝ := -3
  let y₁ : ℝ := 1/2
  let x₂ : ℝ := 7
  let y₂ : ℝ := 9
  let mx : ℝ := (x₁ + x₂) / 2
  let my : ℝ := (y₁ + y₂) / 2
  mx + my = 6.75 := by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2657_265760


namespace NUMINAMATH_CALUDE_absolute_sum_diff_equal_implies_product_zero_l2657_265704

theorem absolute_sum_diff_equal_implies_product_zero (a b : ℝ) :
  |a + b| = |a - b| → a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_sum_diff_equal_implies_product_zero_l2657_265704


namespace NUMINAMATH_CALUDE_surface_area_of_seven_solid_arrangement_l2657_265799

/-- Represents a 1 × 1 × 2 solid -/
structure Solid :=
  (length : ℝ := 1)
  (width : ℝ := 1)
  (height : ℝ := 2)

/-- Represents the arrangement of solids as shown in the diagram -/
def Arrangement := List Solid

/-- Calculates the surface area of the arrangement -/
def surfaceArea (arr : Arrangement) : ℝ :=
  sorry

/-- The specific arrangement of seven solids as shown in the diagram -/
def sevenSolidArrangement : Arrangement :=
  List.replicate 7 { length := 1, width := 1, height := 2 }

theorem surface_area_of_seven_solid_arrangement :
  surfaceArea sevenSolidArrangement = 42 :=
by sorry

end NUMINAMATH_CALUDE_surface_area_of_seven_solid_arrangement_l2657_265799


namespace NUMINAMATH_CALUDE_stock_value_decrease_l2657_265797

theorem stock_value_decrease (n : ℕ) (n_pos : 0 < n) : (0.99 : ℝ) ^ n < 1 := by
  sorry

#check stock_value_decrease

end NUMINAMATH_CALUDE_stock_value_decrease_l2657_265797


namespace NUMINAMATH_CALUDE_sum_fourth_fifth_terms_l2657_265762

def geometric_sequence (a₀ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₀ * r ^ n

theorem sum_fourth_fifth_terms (a₀ r : ℝ) :
  a₀ > 0 →
  r > 0 →
  r < 1 →
  geometric_sequence a₀ r 0 = 4096 →
  geometric_sequence a₀ r 1 = 1024 →
  geometric_sequence a₀ r 2 = 256 →
  geometric_sequence a₀ r 5 = 4 →
  geometric_sequence a₀ r 6 = 1 →
  geometric_sequence a₀ r 7 = 1/4 →
  geometric_sequence a₀ r 3 + geometric_sequence a₀ r 4 = 80 := by
sorry

end NUMINAMATH_CALUDE_sum_fourth_fifth_terms_l2657_265762


namespace NUMINAMATH_CALUDE_rectangle_area_from_perimeter_and_diagonal_l2657_265780

/-- The area of a rectangle given its perimeter and diagonal -/
theorem rectangle_area_from_perimeter_and_diagonal (p d : ℝ) (hp : p > 0) (hd : d > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2 * (x + y) = p ∧ x^2 + y^2 = d^2 ∧
  x * y = (p^2 - 4 * d^2) / 8 := by
  sorry

#check rectangle_area_from_perimeter_and_diagonal

end NUMINAMATH_CALUDE_rectangle_area_from_perimeter_and_diagonal_l2657_265780


namespace NUMINAMATH_CALUDE_thirteen_factorial_divisible_by_eleven_l2657_265748

/-- Definition of factorial for positive integers -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem: 13! is divisible by 11 -/
theorem thirteen_factorial_divisible_by_eleven :
  13 % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_factorial_divisible_by_eleven_l2657_265748
