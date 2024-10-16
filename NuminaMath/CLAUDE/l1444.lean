import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_degree_reduction_l1444_144423

theorem quadratic_degree_reduction (x : ℝ) (h1 : x^2 - x - 1 = 0) (h2 : x > 0) :
  x^4 - 2*x^3 + 3*x = 1 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_degree_reduction_l1444_144423


namespace NUMINAMATH_CALUDE_binomial_9_choose_5_l1444_144410

theorem binomial_9_choose_5 : Nat.choose 9 5 = 126 := by
  sorry

end NUMINAMATH_CALUDE_binomial_9_choose_5_l1444_144410


namespace NUMINAMATH_CALUDE_square_areas_and_perimeters_l1444_144403

theorem square_areas_and_perimeters (x : ℝ) : 
  (x^2 + 8*x + 16 = (x + 4)^2) ∧ 
  (4*x^2 - 12*x + 9 = (2*x - 3)^2) ∧ 
  (4*(x + 4) + 4*(2*x - 3) = 32) → 
  x = 7/3 := by
sorry

end NUMINAMATH_CALUDE_square_areas_and_perimeters_l1444_144403


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1444_144481

/-- Given a > 0, if in the expansion of (1+a√x)^n, the coefficient of x^2 is 9 times
    the coefficient of x, and the third term is 135x, then a = 3 -/
theorem binomial_expansion_coefficient (a n : ℝ) (ha : a > 0) : 
  (∃ k₁ k₂ : ℝ, k₁ ≠ 0 ∧ k₂ ≠ 0 ∧ 
    k₁ * a^4 = 9 * k₂ * a^2 ∧
    k₂ * a^2 = 135) →
  a = 3 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1444_144481


namespace NUMINAMATH_CALUDE_combined_salaries_combined_salaries_proof_l1444_144496

/-- The combined salaries of A, B, C, and E, given D's salary and the average salary of all five. -/
theorem combined_salaries (salary_D : ℕ) (average_salary : ℕ) : ℕ :=
  let total_salary := average_salary * 5
  total_salary - salary_D

/-- Proof that the combined salaries of A, B, C, and E is 38000, given the conditions. -/
theorem combined_salaries_proof (salary_D : ℕ) (average_salary : ℕ)
    (h1 : salary_D = 7000)
    (h2 : average_salary = 9000) :
    combined_salaries salary_D average_salary = 38000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_combined_salaries_proof_l1444_144496


namespace NUMINAMATH_CALUDE_max_cells_crossed_cells_crossed_achievable_l1444_144456

/-- Represents a circle on a grid --/
structure GridCircle where
  radius : ℝ
  center : ℝ × ℝ

/-- Represents a cell on a grid --/
structure GridCell where
  x : ℤ
  y : ℤ

/-- Function to count the number of cells crossed by a circle --/
def countCrossedCells (c : GridCircle) : ℕ :=
  sorry

/-- Theorem stating the maximum number of cells crossed by a circle with radius 10 --/
theorem max_cells_crossed (c : GridCircle) (h : c.radius = 10) :
  countCrossedCells c ≤ 80 :=
sorry

/-- Theorem stating that 80 cells can be crossed --/
theorem cells_crossed_achievable :
  ∃ (c : GridCircle), c.radius = 10 ∧ countCrossedCells c = 80 :=
sorry

end NUMINAMATH_CALUDE_max_cells_crossed_cells_crossed_achievable_l1444_144456


namespace NUMINAMATH_CALUDE_awards_assignment_count_l1444_144417

/-- The number of different types of awards -/
def num_awards : ℕ := 4

/-- The number of students -/
def num_students : ℕ := 8

/-- The total number of ways to assign awards -/
def total_assignments : ℕ := num_awards ^ num_students

/-- Theorem stating that the total number of assignments is 65536 -/
theorem awards_assignment_count :
  total_assignments = 65536 := by
  sorry

end NUMINAMATH_CALUDE_awards_assignment_count_l1444_144417


namespace NUMINAMATH_CALUDE_company_fund_problem_l1444_144484

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  (50 * n = initial_fund + 5) →
  (45 * n + 95 = initial_fund) →
  initial_fund = 995 := by
  sorry

end NUMINAMATH_CALUDE_company_fund_problem_l1444_144484


namespace NUMINAMATH_CALUDE_small_bottles_sold_percentage_l1444_144434

theorem small_bottles_sold_percentage
  (initial_small : ℕ)
  (initial_big : ℕ)
  (big_sold_percentage : ℚ)
  (total_remaining : ℕ)
  (h1 : initial_small = 6000)
  (h2 : initial_big = 15000)
  (h3 : big_sold_percentage = 12 / 100)
  (h4 : total_remaining = 18540)
  (h5 : ∃ (x : ℚ), 
    initial_small - (x * initial_small) + 
    initial_big - (big_sold_percentage * initial_big) = total_remaining) :
  ∃ (x : ℚ), x = 11 / 100 ∧ 
    initial_small - (x * initial_small) + 
    initial_big - (big_sold_percentage * initial_big) = total_remaining :=
by sorry

end NUMINAMATH_CALUDE_small_bottles_sold_percentage_l1444_144434


namespace NUMINAMATH_CALUDE_container_initial_percentage_l1444_144422

theorem container_initial_percentage (capacity : ℝ) (added_water : ℝ) (final_fraction : ℝ) :
  capacity = 60 →
  added_water = 27 →
  final_fraction = 3/4 →
  (capacity * final_fraction - added_water) / capacity * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_container_initial_percentage_l1444_144422


namespace NUMINAMATH_CALUDE_class_average_weight_l1444_144439

theorem class_average_weight (students_A : ℕ) (students_B : ℕ) (avg_weight_A : ℝ) (avg_weight_B : ℝ)
  (h1 : students_A = 24)
  (h2 : students_B = 16)
  (h3 : avg_weight_A = 40)
  (h4 : avg_weight_B = 35) :
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B : ℝ) = 38 := by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l1444_144439


namespace NUMINAMATH_CALUDE_greendale_points_greendale_points_proof_l1444_144415

/-- Calculates the total points for Greendale High School in a basketball tournament -/
theorem greendale_points (roosevelt_first_game : ℕ) (bonus : ℕ) (difference : ℕ) : ℕ :=
  let roosevelt_second_game := roosevelt_first_game / 2
  let roosevelt_third_game := roosevelt_second_game * 3
  let roosevelt_total := roosevelt_first_game + roosevelt_second_game + roosevelt_third_game + bonus
  roosevelt_total - difference

/-- Proves that Greendale High School's total points equal 130 -/
theorem greendale_points_proof :
  greendale_points 30 50 10 = 130 := by
  sorry

end NUMINAMATH_CALUDE_greendale_points_greendale_points_proof_l1444_144415


namespace NUMINAMATH_CALUDE_line_slope_range_l1444_144490

/-- A line passing through (1,1) with y-intercept in (0,2) has slope in (-1,1) -/
theorem line_slope_range (l : Set (ℝ × ℝ)) (y_intercept : ℝ) :
  (∀ p ∈ l, ∃ k : ℝ, p.2 - 1 = k * (p.1 - 1)) →  -- l is a line
  (1, 1) ∈ l →  -- l passes through (1,1)
  0 < y_intercept ∧ y_intercept < 2 →  -- y-intercept is in (0,2)
  (∃ b : ℝ, ∀ x y : ℝ, (x, y) ∈ l ↔ y = y_intercept + (y_intercept - 1) * (x - 1)) →
  ∃ k : ℝ, -1 < k ∧ k < 1 ∧ ∀ x y : ℝ, (x, y) ∈ l ↔ y - 1 = k * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_range_l1444_144490


namespace NUMINAMATH_CALUDE_white_roses_needed_l1444_144433

/-- Calculates the total number of white roses needed for wedding arrangements -/
theorem white_roses_needed (num_bouquets num_table_decorations roses_per_bouquet roses_per_table_decoration : ℕ) : 
  num_bouquets = 5 → 
  num_table_decorations = 7 → 
  roses_per_bouquet = 5 → 
  roses_per_table_decoration = 12 → 
  num_bouquets * roses_per_bouquet + num_table_decorations * roses_per_table_decoration = 109 := by
  sorry

#check white_roses_needed

end NUMINAMATH_CALUDE_white_roses_needed_l1444_144433


namespace NUMINAMATH_CALUDE_function_equation_solution_l1444_144472

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x ^ 2 + f y) = x * f x + y) →
  (∀ x : ℝ, f x = x ∨ f x = -x) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l1444_144472


namespace NUMINAMATH_CALUDE_quadrilateral_reconstruction_l1444_144499

-- Define the quadrilateral and its points
variable (E F G H E' F' G' H' : ℝ × ℝ)

-- Define the conditions
variable (h1 : E' - F = E - F)
variable (h2 : F' - G = F - G)
variable (h3 : G' - H = G - H)
variable (h4 : H' - E = H - E)

-- Define the theorem
theorem quadrilateral_reconstruction :
  ∃ (x y z w : ℝ),
    E = x • E' + y • F' + z • G' + w • H' ∧
    x = 1/15 ∧ y = 2/15 ∧ z = 4/15 ∧ w = 8/15 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_reconstruction_l1444_144499


namespace NUMINAMATH_CALUDE_sum_product_inequality_l1444_144488

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 1) : a * b + b * c + c * a ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l1444_144488


namespace NUMINAMATH_CALUDE_system_solution_and_sum_l1444_144441

theorem system_solution_and_sum :
  ∃ (x y : ℚ),
    (4 * x - 6 * y = -3) ∧
    (8 * x + 3 * y = 6) ∧
    (x = 9/20) ∧
    (y = 4/5) ∧
    (x + y = 5/4) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_and_sum_l1444_144441


namespace NUMINAMATH_CALUDE_race_participants_l1444_144458

theorem race_participants (first_year : ℕ) (second_year : ℕ) : 
  first_year = 8 →
  second_year = 5 * first_year →
  first_year + second_year = 48 := by
  sorry

end NUMINAMATH_CALUDE_race_participants_l1444_144458


namespace NUMINAMATH_CALUDE_tuesday_grading_percentage_l1444_144430

theorem tuesday_grading_percentage 
  (total_exams : ℕ) 
  (monday_percentage : ℚ) 
  (wednesday_exams : ℕ) : 
  total_exams = 120 → 
  monday_percentage = 60 / 100 → 
  wednesday_exams = 12 → 
  (((total_exams : ℚ) - (monday_percentage * total_exams) - wednesday_exams) / 
   ((total_exams : ℚ) - (monday_percentage * total_exams))) = 75 / 100 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_grading_percentage_l1444_144430


namespace NUMINAMATH_CALUDE_triangle_problem_l1444_144463

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  A = π / 6 →
  b = (4 + 2 * Real.sqrt 3) * a * Real.cos B →
  b = 1 →
  B = 5 * π / 12 ∧ 
  (1 / 2) * b * c * Real.sin A = 1 / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_problem_l1444_144463


namespace NUMINAMATH_CALUDE_hacker_guarantee_l1444_144428

/-- A computer network with the given properties -/
structure ComputerNetwork where
  num_computers : ℕ
  is_connected : Bool
  no_shared_cycle_vertices : Bool

/-- The game state -/
structure GameState where
  network : ComputerNetwork
  hacked_computers : ℕ
  protected_computers : ℕ

/-- The game rules -/
def game_rules (state : GameState) : Bool :=
  state.hacked_computers + state.protected_computers ≤ state.network.num_computers

/-- The theorem statement -/
theorem hacker_guarantee (network : ComputerNetwork) 
  (h1 : network.num_computers = 2008)
  (h2 : network.is_connected = true)
  (h3 : network.no_shared_cycle_vertices = true) :
  ∃ (final_state : GameState), 
    final_state.network = network ∧ 
    game_rules final_state ∧ 
    final_state.hacked_computers ≥ 671 :=
sorry

end NUMINAMATH_CALUDE_hacker_guarantee_l1444_144428


namespace NUMINAMATH_CALUDE_sally_orange_balloons_l1444_144493

/-- The number of orange balloons Sally has now, given her initial count and the number she lost. -/
def remaining_balloons (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem stating that Sally now has 7 orange balloons -/
theorem sally_orange_balloons : remaining_balloons 9 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sally_orange_balloons_l1444_144493


namespace NUMINAMATH_CALUDE_point_returns_after_seven_steps_l1444_144400

/-- Represents a point in a triangle -/
structure TrianglePoint where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : TrianglePoint
  B : TrianglePoint
  C : TrianglePoint

/-- Represents the movement of a point parallel to a side of the triangle -/
def moveParallel (start : TrianglePoint) (side : Triangle → TrianglePoint × TrianglePoint) : TrianglePoint := sorry

/-- Represents the sequence of movements in the triangle -/
def moveSequence (start : TrianglePoint) (triangle : Triangle) : TrianglePoint := 
  let step1 := moveParallel start (λ t => (t.B, t.C))
  let step2 := moveParallel step1 (λ t => (t.A, t.B))
  let step3 := moveParallel step2 (λ t => (t.C, t.A))
  let step4 := moveParallel step3 (λ t => (t.B, t.C))
  let step5 := moveParallel step4 (λ t => (t.A, t.B))
  let step6 := moveParallel step5 (λ t => (t.C, t.A))
  moveParallel step6 (λ t => (t.B, t.C))

/-- The main theorem stating that the point returns to its original position after 7 steps -/
theorem point_returns_after_seven_steps (triangle : Triangle) (start : TrianglePoint) :
  moveSequence start triangle = start := sorry

end NUMINAMATH_CALUDE_point_returns_after_seven_steps_l1444_144400


namespace NUMINAMATH_CALUDE_right_triangle_tan_l1444_144483

theorem right_triangle_tan (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) :
  b = 12 ∧ a / b = 5 / 12 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_tan_l1444_144483


namespace NUMINAMATH_CALUDE_earnings_difference_l1444_144476

/-- Proves the difference in earnings between Evan and Markese -/
theorem earnings_difference : 
  ∀ (E : ℕ), 
  E > 16 →  -- Evan earned more than Markese
  E + 16 = 37 →  -- Their combined earnings
  E - 16 = 5  -- The difference in earnings
  := by sorry

end NUMINAMATH_CALUDE_earnings_difference_l1444_144476


namespace NUMINAMATH_CALUDE_gcd_228_2008_l1444_144487

theorem gcd_228_2008 : Nat.gcd 228 2008 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_2008_l1444_144487


namespace NUMINAMATH_CALUDE_x_equals_one_l1444_144438

theorem x_equals_one (x y : ℕ+) 
  (h : ∀ n : ℕ+, (n * y)^2 + 1 ∣ x^(Nat.totient n) - 1) : 
  x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_one_l1444_144438


namespace NUMINAMATH_CALUDE_nabla_equation_solution_l1444_144459

/-- The nabla operation defined for real numbers -/
def nabla (a b : ℝ) : ℝ := (a + 1) * (b - 2)

/-- Theorem: If 5 ∇ x = 30, then x = 7 -/
theorem nabla_equation_solution :
  ∀ x : ℝ, nabla 5 x = 30 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_nabla_equation_solution_l1444_144459


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1444_144416

/-- Given that 15 is the arithmetic mean of the set {7, 12, 19, 8, 10, y}, prove that y = 34 -/
theorem arithmetic_mean_problem (y : ℝ) : 
  (7 + 12 + 19 + 8 + 10 + y) / 6 = 15 → y = 34 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1444_144416


namespace NUMINAMATH_CALUDE_our_circle_center_and_radius_l1444_144411

/-- A circle in the xy-plane --/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The center of a circle --/
def center (c : Circle) : ℝ × ℝ := sorry

/-- The radius of a circle --/
def radius (c : Circle) : ℝ := sorry

/-- Our specific circle --/
def our_circle : Circle :=
  { equation := λ x y => x^2 + y^2 - 6*x = 0 }

theorem our_circle_center_and_radius :
  center our_circle = (3, 0) ∧ radius our_circle = 3 := by sorry

end NUMINAMATH_CALUDE_our_circle_center_and_radius_l1444_144411


namespace NUMINAMATH_CALUDE_dispatch_plans_count_l1444_144447

/-- The number of vehicles in the fleet -/
def total_vehicles : ℕ := 7

/-- The number of vehicles to be dispatched -/
def dispatched_vehicles : ℕ := 4

/-- The number of ways to arrange vehicles A and B with A before B -/
def arrange_A_B : ℕ := 6

/-- The number of remaining vehicles after A and B are selected -/
def remaining_vehicles : ℕ := total_vehicles - 2

/-- The number of additional vehicles to be selected after A and B -/
def additional_vehicles : ℕ := dispatched_vehicles - 2

/-- Calculate the number of permutations of n elements taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - r)

theorem dispatch_plans_count :
  arrange_A_B * permutations remaining_vehicles additional_vehicles = 120 :=
sorry

end NUMINAMATH_CALUDE_dispatch_plans_count_l1444_144447


namespace NUMINAMATH_CALUDE_milburg_grownups_l1444_144429

/-- The number of grown-ups in Milburg -/
def num_grownups (total_population children : ℕ) : ℕ :=
  total_population - children

/-- Theorem: The number of grown-ups in Milburg is 5256 -/
theorem milburg_grownups :
  num_grownups 8243 2987 = 5256 := by
  sorry

end NUMINAMATH_CALUDE_milburg_grownups_l1444_144429


namespace NUMINAMATH_CALUDE_choose_four_from_eight_l1444_144477

theorem choose_four_from_eight : Nat.choose 8 4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_eight_l1444_144477


namespace NUMINAMATH_CALUDE_total_entertainment_cost_l1444_144469

def computer_game_cost : ℕ := 66
def movie_ticket_cost : ℕ := 12
def number_of_tickets : ℕ := 3

theorem total_entertainment_cost : 
  computer_game_cost + number_of_tickets * movie_ticket_cost = 102 := by
  sorry

end NUMINAMATH_CALUDE_total_entertainment_cost_l1444_144469


namespace NUMINAMATH_CALUDE_initial_crayons_count_l1444_144442

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := sorry

/-- The number of crayons Benny added to the drawer -/
def added_crayons : ℕ := 3

/-- The total number of crayons in the drawer after Benny's addition -/
def total_crayons : ℕ := 12

/-- Theorem stating that the initial number of crayons is 9 -/
theorem initial_crayons_count : initial_crayons = 9 := by sorry

end NUMINAMATH_CALUDE_initial_crayons_count_l1444_144442


namespace NUMINAMATH_CALUDE_correct_sum_exists_l1444_144494

def num1 : ℕ := 3742586
def num2 : ℕ := 4829430
def given_sum : ℕ := 72120116

def replace_digit (n : ℕ) (d e : ℕ) : ℕ :=
  -- Function to replace all occurrences of d with e in n
  sorry

theorem correct_sum_exists : ∃ (d e : ℕ), d ≠ e ∧ 
  d < 10 ∧ e < 10 ∧ 
  replace_digit num1 d e + replace_digit num2 d e = given_sum ∧
  d + e = 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_sum_exists_l1444_144494


namespace NUMINAMATH_CALUDE_bicycle_owners_without_cars_proof_l1444_144440

/-- Represents the number of adults who own bicycles but not cars in a population where every adult owns either a bicycle, a car, or both. -/
def bicycle_owners_without_cars (total_adults bicycle_owners car_owners : ℕ) : ℕ :=
  bicycle_owners - (bicycle_owners + car_owners - total_adults)

/-- Theorem stating that in a population of 500 adults where each adult owns either a bicycle, a car, or both, 
    given that 450 adults own bicycles and 120 adults own cars, the number of bicycle owners who do not own a car is 380. -/
theorem bicycle_owners_without_cars_proof :
  bicycle_owners_without_cars 500 450 120 = 380 := by
  sorry

#eval bicycle_owners_without_cars 500 450 120

end NUMINAMATH_CALUDE_bicycle_owners_without_cars_proof_l1444_144440


namespace NUMINAMATH_CALUDE_no_prime_solution_l1444_144448

/-- Convert a number from base p to decimal --/
def baseP_to_decimal (digits : List Nat) (p : Nat) : Nat :=
  digits.foldr (fun d acc => d + p * acc) 0

/-- The equation that needs to be satisfied --/
def equation (p : Nat) : Prop :=
  baseP_to_decimal [1, 0, 1, 3] p +
  baseP_to_decimal [2, 0, 7] p +
  baseP_to_decimal [2, 1, 4] p +
  baseP_to_decimal [1, 0, 0] p +
  baseP_to_decimal [1, 0] p =
  baseP_to_decimal [3, 2, 1] p +
  baseP_to_decimal [4, 0, 3] p +
  baseP_to_decimal [2, 1, 0] p

theorem no_prime_solution :
  ∀ p, Nat.Prime p → ¬(equation p) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_l1444_144448


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1444_144473

theorem modulus_of_complex_fraction : Complex.abs (2 / (1 + Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1444_144473


namespace NUMINAMATH_CALUDE_yellow_chip_value_l1444_144465

theorem yellow_chip_value :
  ∀ (y : ℕ) (b : ℕ),
    y > 0 →
    b > 0 →
    y^4 * (4 * b)^b * (5 * b)^b = 16000 →
    y = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_chip_value_l1444_144465


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_system_two_equations_solution_system_three_equations_solution_l1444_144424

-- Problem 1
theorem equation_one_solution (x : ℝ) : 3 * x - 2 = 10 - 2 * (x + 1) → x = 2 := by
  sorry

-- Problem 2
theorem equation_two_solution (x : ℝ) : (2 * x + 1) / 3 - (5 * x - 1) / 6 = 1 → x = -3 := by
  sorry

-- Problem 3
theorem system_two_equations_solution (x y : ℝ) : 
  x + 2 * y = 5 ∧ 3 * x - 2 * y = -1 → x = 1 ∧ y = 2 := by
  sorry

-- Problem 4
theorem system_three_equations_solution (x y z : ℝ) :
  2 * x + y + z = 15 ∧ x + 2 * y + z = 16 ∧ x + y + 2 * z = 17 → 
  x = 3 ∧ y = 4 ∧ z = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_system_two_equations_solution_system_three_equations_solution_l1444_144424


namespace NUMINAMATH_CALUDE_zoo_count_l1444_144474

theorem zoo_count (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 200) 
  (h2 : total_legs = 522) : 
  ∃ (birds mammals : ℕ), 
    birds + mammals = total_animals ∧ 
    2 * birds + 4 * mammals = total_legs ∧ 
    birds = 139 := by
  sorry

end NUMINAMATH_CALUDE_zoo_count_l1444_144474


namespace NUMINAMATH_CALUDE_true_discount_calculation_l1444_144453

/-- Given the present worth and banker's gain, calculate the true discount -/
theorem true_discount_calculation (present_worth banker_gain : ℕ) 
  (h1 : present_worth = 576) 
  (h2 : banker_gain = 16) : 
  present_worth + banker_gain = 592 := by
  sorry

#check true_discount_calculation

end NUMINAMATH_CALUDE_true_discount_calculation_l1444_144453


namespace NUMINAMATH_CALUDE_reflect_A_x_axis_l1444_144418

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point A -/
def A : ℝ × ℝ := (-3, 2)

theorem reflect_A_x_axis : reflect_x A = (-3, -2) := by
  sorry

end NUMINAMATH_CALUDE_reflect_A_x_axis_l1444_144418


namespace NUMINAMATH_CALUDE_zacks_friends_l1444_144479

def zacks_marbles : ℕ := 65
def marbles_kept : ℕ := 5
def marbles_per_friend : ℕ := 20

theorem zacks_friends :
  (zacks_marbles - marbles_kept) / marbles_per_friend = 3 :=
by sorry

end NUMINAMATH_CALUDE_zacks_friends_l1444_144479


namespace NUMINAMATH_CALUDE_sandwich_apple_cost_l1444_144444

/-- The cost of items given fixed prices per item -/
theorem sandwich_apple_cost 
  (sandwich_price apple_price : ℚ) 
  (h1 : sandwich_price + 4 * apple_price = 3.6)
  (h2 : 3 * sandwich_price + 2 * apple_price = 4.8) :
  2 * sandwich_price + 5 * apple_price = 5.4 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_apple_cost_l1444_144444


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1444_144406

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, 5 * x^2 + k * x - 10 = 0 ∧ x = -5) →
  (∃ x : ℝ, 5 * x^2 + k * x - 10 = 0 ∧ x = 2/5) ∧ k = 23 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1444_144406


namespace NUMINAMATH_CALUDE_count_pairs_eq_three_l1444_144491

/-- The number of distinct ordered pairs of positive integers (m,n) satisfying 1/m + 1/n = 1/3 -/
def count_pairs : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    p.1 > 0 ∧ p.2 > 0 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = (1 : ℚ) / 3)
    (Finset.product (Finset.range 100) (Finset.range 100))).card

theorem count_pairs_eq_three : count_pairs = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_three_l1444_144491


namespace NUMINAMATH_CALUDE_corn_harvest_difference_l1444_144407

theorem corn_harvest_difference (greg_harvest sharon_harvest : ℝ) 
  (h1 : greg_harvest = 0.4)
  (h2 : sharon_harvest = 0.1) :
  greg_harvest - sharon_harvest = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_corn_harvest_difference_l1444_144407


namespace NUMINAMATH_CALUDE_square_difference_equals_1290_l1444_144405

theorem square_difference_equals_1290 : (43 + 15)^2 - (43^2 + 15^2) = 1290 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_1290_l1444_144405


namespace NUMINAMATH_CALUDE_smallest_integer_inequality_l1444_144471

theorem smallest_integer_inequality (x y z : ℝ) :
  ∃ (n : ℕ), (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4) ∧
  ∀ (m : ℕ), m < n → ∃ (a b c : ℝ), (a^2 + b^2 + c^2)^2 > m * (a^4 + b^4 + c^4) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_inequality_l1444_144471


namespace NUMINAMATH_CALUDE_quadratic_vertex_value_and_range_l1444_144413

/-- The quadratic function y = ax^2 + 2ax + a -/
def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + a

/-- The x-coordinate of the vertex of the quadratic function -/
def vertex_x (a : ℝ) : ℝ := -1

theorem quadratic_vertex_value_and_range (a : ℝ) :
  f a (vertex_x a) = 0 ∧ f a (vertex_x a) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_value_and_range_l1444_144413


namespace NUMINAMATH_CALUDE_expression_value_l1444_144402

theorem expression_value (a b : ℝ) (h : a^2 + 2*a*b + b^2 = 0) :
  a*(a + 4*b) - (a + 2*b)*(a - 2*b) = 0 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1444_144402


namespace NUMINAMATH_CALUDE_modulus_of_one_plus_three_i_l1444_144498

theorem modulus_of_one_plus_three_i : Complex.abs (1 + 3 * Complex.I) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_one_plus_three_i_l1444_144498


namespace NUMINAMATH_CALUDE_prob_all_red_before_both_green_is_one_third_l1444_144480

/-- The number of red chips in the hat -/
def num_red : ℕ := 4

/-- The number of green chips in the hat -/
def num_green : ℕ := 2

/-- The total number of chips in the hat -/
def total_chips : ℕ := num_red + num_green

/-- The probability of drawing all red chips before both green chips -/
def prob_all_red_before_both_green : ℚ :=
  (total_chips - 1).choose num_green / total_chips.choose num_green

theorem prob_all_red_before_both_green_is_one_third :
  prob_all_red_before_both_green = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_all_red_before_both_green_is_one_third_l1444_144480


namespace NUMINAMATH_CALUDE_minimum_fourth_quarter_score_l1444_144460

def required_average : ℝ := 85
def num_quarters : ℕ := 4
def first_quarter_score : ℝ := 84
def second_quarter_score : ℝ := 82
def third_quarter_score : ℝ := 80

theorem minimum_fourth_quarter_score :
  let total_required := required_average * num_quarters
  let current_total := first_quarter_score + second_quarter_score + third_quarter_score
  let minimum_fourth_score := total_required - current_total
  minimum_fourth_score = 94 ∧
  (first_quarter_score + second_quarter_score + third_quarter_score + minimum_fourth_score) / num_quarters ≥ required_average :=
by sorry

end NUMINAMATH_CALUDE_minimum_fourth_quarter_score_l1444_144460


namespace NUMINAMATH_CALUDE_smallest_perfect_square_multiple_l1444_144435

def n : ℕ := 2023

-- Define 2023 as 7 * 17^2
axiom n_factorization : n = 7 * 17^2

-- Define the function to check if a number is a perfect square
def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y^2

-- Define the function to check if a number is a multiple of 2023
def is_multiple_of_2023 (x : ℕ) : Prop := ∃ k : ℕ, x = k * n

-- Theorem statement
theorem smallest_perfect_square_multiple :
  (7 * n = (7 * 17)^2) ∧
  is_perfect_square (7 * n) ∧
  is_multiple_of_2023 (7 * n) ∧
  (∀ m : ℕ, m < 7 * n → ¬(is_perfect_square m ∧ is_multiple_of_2023 m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_multiple_l1444_144435


namespace NUMINAMATH_CALUDE_marys_next_birthday_age_l1444_144425

theorem marys_next_birthday_age 
  (mary sally danielle : ℝ) 
  (h1 : mary = 1.3 * sally) 
  (h2 : sally = 0.5 * danielle) 
  (h3 : mary + sally + danielle = 45) : 
  ⌊mary⌋ + 1 = 14 := by
  sorry

end NUMINAMATH_CALUDE_marys_next_birthday_age_l1444_144425


namespace NUMINAMATH_CALUDE_meaningful_range_for_sqrt_fraction_l1444_144467

/-- The range of x for which the expression sqrt(x-1)/(x-3) is meaningful in the real number system. -/
theorem meaningful_range_for_sqrt_fraction (x : ℝ) :
  (∃ y : ℝ, y^2 = x - 1 ∧ x - 3 ≠ 0) ↔ x ≥ 1 ∧ x ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_range_for_sqrt_fraction_l1444_144467


namespace NUMINAMATH_CALUDE_trees_to_plant_l1444_144451

theorem trees_to_plant (current_trees final_trees : ℕ) : 
  current_trees = 25 → final_trees = 98 → final_trees - current_trees = 73 := by
  sorry

end NUMINAMATH_CALUDE_trees_to_plant_l1444_144451


namespace NUMINAMATH_CALUDE_fractional_equation_solution_condition_l1444_144470

theorem fractional_equation_solution_condition (m : ℝ) : 
  (∃ x : ℝ, x ≠ 2 ∧ (m + x) / (2 - x) - 3 = 0) ↔ m ≠ -2 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_condition_l1444_144470


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l1444_144482

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_proof (a b c : ℝ) :
  (∀ x, f a b c x ≥ 0) ∧  -- Minimum value is 0
  (∀ x, f a b c x = f a b c (-2 - x)) ∧  -- Symmetric about x = -1
  (∀ x ∈ Set.Ioo 0 5, x ≤ f a b c x ∧ f a b c x ≤ 2 * |x - 1| + 1) →
  ∀ x, f a b c x = (1/4) * (x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l1444_144482


namespace NUMINAMATH_CALUDE_march_temperature_data_inconsistent_l1444_144404

/-- Represents the statistical data for March temperatures --/
structure MarchTemperatureData where
  mean : ℝ
  median : ℝ
  variance : ℝ
  mean_eq_zero : mean = 0
  median_eq_four : median = 4
  variance_eq : variance = 15.917

/-- Theorem stating that the given data is inconsistent --/
theorem march_temperature_data_inconsistent (data : MarchTemperatureData) :
  (data.mean - data.median)^2 > data.variance := by
  sorry

#check march_temperature_data_inconsistent

end NUMINAMATH_CALUDE_march_temperature_data_inconsistent_l1444_144404


namespace NUMINAMATH_CALUDE_bat_costs_60_l1444_144492

/-- The cost of a ball in pounds -/
def ball_cost : ℝ := sorry

/-- The cost of a bat in pounds -/
def bat_cost : ℝ := sorry

/-- The sum of the cost of a ball and a bat is £90 -/
axiom sum_ball_bat : ball_cost + bat_cost = 90

/-- The sum of the cost of three balls and two bats is £210 -/
axiom sum_three_balls_two_bats : 3 * ball_cost + 2 * bat_cost = 210

/-- The cost of a bat is £60 -/
theorem bat_costs_60 : bat_cost = 60 := by sorry

end NUMINAMATH_CALUDE_bat_costs_60_l1444_144492


namespace NUMINAMATH_CALUDE_nail_fractions_l1444_144449

theorem nail_fractions (fraction_4d fraction_total : ℝ) 
  (h1 : fraction_4d = 0.5)
  (h2 : fraction_total = 0.75) : 
  fraction_total - fraction_4d = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_nail_fractions_l1444_144449


namespace NUMINAMATH_CALUDE_quadrilateral_is_parallelogram_l1444_144412

-- Define the points
variable (A B C D M N P : ℝ × ℝ)

-- Define the conditions
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def is_midpoint (M B C : ℝ × ℝ) : Prop := sorry

def lines_intersect (A M B N P : ℝ × ℝ) : Prop := sorry

def ratio_equals (P M A : ℝ × ℝ) (r : ℚ) : Prop := sorry

def is_parallelogram (A B C D : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem quadrilateral_is_parallelogram 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_midpoint M B C)
  (h3 : is_midpoint N C D)
  (h4 : lines_intersect A M B N P)
  (h5 : ratio_equals P M A (1/5))
  (h6 : ratio_equals B P N (2/5))
  : is_parallelogram A B C D := by sorry

end NUMINAMATH_CALUDE_quadrilateral_is_parallelogram_l1444_144412


namespace NUMINAMATH_CALUDE_museum_visit_l1444_144462

theorem museum_visit (num_students : ℕ) (ticket_price : ℕ) :
  (∃ k : ℕ, num_students = 5 * k) →
  (num_students + 1) * (ticket_price / 2) = 1599 →
  ticket_price % 2 = 0 →
  num_students = 40 ∧ ticket_price = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_museum_visit_l1444_144462


namespace NUMINAMATH_CALUDE_product_place_value_l1444_144431

theorem product_place_value : 
  (216 * 5 ≥ 1000 ∧ 216 * 5 < 10000) ∧ 
  (126 * 5 ≥ 100 ∧ 126 * 5 < 1000) := by
  sorry

end NUMINAMATH_CALUDE_product_place_value_l1444_144431


namespace NUMINAMATH_CALUDE_parabolas_intersection_l1444_144443

/-- First parabola equation -/
def f (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 5

/-- Second parabola equation -/
def g (x : ℝ) : ℝ := x^2 - 6 * x + 10

/-- Intersection points -/
def intersection_points : Set (ℝ × ℝ) :=
  {((3 - Real.sqrt 19) / 2, 12), ((3 + Real.sqrt 19) / 2, 12)}

theorem parabolas_intersection :
  ∀ p : ℝ × ℝ, f p.1 = g p.1 ↔ p ∈ intersection_points :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l1444_144443


namespace NUMINAMATH_CALUDE_square_field_division_l1444_144409

/-- Represents a square field with side length and division properties -/
structure SquareField where
  side_length : ℝ
  division_fence_length : ℝ

/-- Theorem: A square field of side 33m can be divided into three equal areas with at most 54m of fencing -/
theorem square_field_division (field : SquareField) 
  (h1 : field.side_length = 33) 
  (h2 : field.division_fence_length ≤ 54) : 
  ∃ (area_1 area_2 area_3 : ℝ), 
    area_1 = area_2 ∧ 
    area_2 = area_3 ∧ 
    area_1 + area_2 + area_3 = field.side_length * field.side_length := by
  sorry

end NUMINAMATH_CALUDE_square_field_division_l1444_144409


namespace NUMINAMATH_CALUDE_haley_balls_count_l1444_144457

theorem haley_balls_count (balls_per_bag : ℕ) (num_bags : ℕ) (h1 : balls_per_bag = 4) (h2 : num_bags = 9) :
  balls_per_bag * num_bags = 36 := by
  sorry

end NUMINAMATH_CALUDE_haley_balls_count_l1444_144457


namespace NUMINAMATH_CALUDE_problem_solution_l1444_144437

theorem problem_solution (x y : ℝ) (h1 : x - y = 1) (h2 : x^3 - y^3 = 2) :
  x^4 + y^4 = 23/9 ∧ x^5 - y^5 = 29/9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1444_144437


namespace NUMINAMATH_CALUDE_journey_speeds_correct_l1444_144485

/-- Represents the journey details and speeds -/
structure Journey where
  uphill_distance : ℝ
  downhill_distance : ℝ
  flat_distance : ℝ
  time_ab : ℝ
  time_ba : ℝ
  flat_speed : ℝ
  uphill_speed : ℝ
  downhill_speed : ℝ

/-- Checks if the given speeds satisfy the journey conditions -/
def satisfies_conditions (j : Journey) : Prop :=
  let flat_time := j.flat_distance / j.flat_speed
  let hill_time_ab := j.time_ab - flat_time
  let hill_time_ba := j.time_ba - flat_time
  (j.uphill_distance / j.uphill_speed + j.downhill_distance / j.downhill_speed = hill_time_ab) ∧
  (j.uphill_distance / j.downhill_speed + j.downhill_distance / j.uphill_speed = hill_time_ba)

/-- Theorem stating that the given speeds satisfy the journey conditions -/
theorem journey_speeds_correct (j : Journey) 
  (h1 : j.uphill_distance = 3)
  (h2 : j.downhill_distance = 6)
  (h3 : j.flat_distance = 12)
  (h4 : j.time_ab = 67/60)
  (h5 : j.time_ba = 76/60)
  (h6 : j.flat_speed = 18)
  (h7 : j.uphill_speed = 12)
  (h8 : j.downhill_speed = 30) :
  satisfies_conditions j := by
  sorry

end NUMINAMATH_CALUDE_journey_speeds_correct_l1444_144485


namespace NUMINAMATH_CALUDE_jorge_ticket_cost_l1444_144427

def number_of_tickets : ℕ := 24
def price_per_ticket : ℚ := 7
def discount_percentage : ℚ := 50 / 100

def total_cost_with_discount : ℚ :=
  number_of_tickets * price_per_ticket * (1 - discount_percentage)

theorem jorge_ticket_cost : total_cost_with_discount = 84 := by
  sorry

end NUMINAMATH_CALUDE_jorge_ticket_cost_l1444_144427


namespace NUMINAMATH_CALUDE_diamonds_10th_pattern_l1444_144450

/-- The number of diamonds in the n-th pattern of the sequence -/
def diamonds (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 4
  else diamonds (n - 1) + 4 * (2 * n - 1)

/-- The theorem stating that the 10th pattern has 400 diamonds -/
theorem diamonds_10th_pattern : diamonds 10 = 400 := by
  sorry

end NUMINAMATH_CALUDE_diamonds_10th_pattern_l1444_144450


namespace NUMINAMATH_CALUDE_expression_value_l1444_144436

theorem expression_value : (36 + 9)^2 - (9^2 + 36^2) = -1894224 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1444_144436


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_1500_l1444_144452

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The difference between the sum of even and odd numbers -/
def even_odd_sum_difference (n : ℕ) : ℤ :=
  arithmetic_sum 0 2 n - arithmetic_sum 1 2 n

theorem even_odd_sum_difference_1500 :
  even_odd_sum_difference 1500 = -1500 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_1500_l1444_144452


namespace NUMINAMATH_CALUDE_train_speed_proof_l1444_144455

/-- Proves that a train crossing a 280-meter platform in 30 seconds and passing a stationary man in 16 seconds has a speed of 72 km/h -/
theorem train_speed_proof (platform_length : Real) (platform_crossing_time : Real) 
  (man_passing_time : Real) (speed_kmh : Real) : 
  platform_length = 280 ∧ 
  platform_crossing_time = 30 ∧ 
  man_passing_time = 16 ∧
  speed_kmh = (platform_length / (platform_crossing_time - man_passing_time)) * 3.6 →
  speed_kmh = 72 := by
sorry

end NUMINAMATH_CALUDE_train_speed_proof_l1444_144455


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1444_144421

-- Define the set P
def P : Set ℝ := {x | x^2 - x - 2 ≥ 0}

-- Define the set Q
def Q : Set ℝ := {y | ∃ x ∈ P, y = 1/2 * x^2 - 1}

-- Theorem statement
theorem intersection_of_P_and_Q : P ∩ Q = Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1444_144421


namespace NUMINAMATH_CALUDE_translation_theorem_l1444_144466

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point left by a given amount -/
def translateLeft (p : Point) (amount : ℝ) : Point :=
  { x := p.x - amount, y := p.y }

/-- Translates a point down by a given amount -/
def translateDown (p : Point) (amount : ℝ) : Point :=
  { x := p.x, y := p.y - amount }

theorem translation_theorem :
  let p := Point.mk (-4) 3
  let p' := translateDown (translateLeft p 2) 2
  p'.x = -6 ∧ p'.y = 1 := by sorry

end NUMINAMATH_CALUDE_translation_theorem_l1444_144466


namespace NUMINAMATH_CALUDE_product_equality_l1444_144445

theorem product_equality (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1444_144445


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1444_144495

-- Define the sets A and B
def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 6}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1444_144495


namespace NUMINAMATH_CALUDE_inverse_functions_l1444_144408

/-- A function type representing the described graphs --/
inductive FunctionGraph
  | Parabola
  | StraightLine
  | HorizontalLine
  | Semicircle
  | CubicFunction

/-- Predicate to determine if a function graph has an inverse --/
def has_inverse (f : FunctionGraph) : Prop :=
  match f with
  | FunctionGraph.StraightLine => true
  | FunctionGraph.Semicircle => true
  | _ => false

/-- Theorem stating which function graphs have inverses --/
theorem inverse_functions (f : FunctionGraph) :
  has_inverse f ↔ (f = FunctionGraph.StraightLine ∨ f = FunctionGraph.Semicircle) :=
sorry

end NUMINAMATH_CALUDE_inverse_functions_l1444_144408


namespace NUMINAMATH_CALUDE_zero_in_M_l1444_144489

def M : Set Int := {-1, 0, 1}

theorem zero_in_M : 0 ∈ M := by sorry

end NUMINAMATH_CALUDE_zero_in_M_l1444_144489


namespace NUMINAMATH_CALUDE_isosceles_triangle_figure_triangle_count_l1444_144478

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Represents the figure described in the problem -/
structure IsoscelesTriangleFigure where
  base : ℝ
  apex : Point
  baseLeft : Point
  baseRight : Point
  midpointLeft : Point
  midpointRight : Point

/-- Returns the number of triangles in the figure -/
def countTriangles (figure : IsoscelesTriangleFigure) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem isosceles_triangle_figure_triangle_count 
  (figure : IsoscelesTriangleFigure) 
  (h1 : figure.base = 2)
  (h2 : figure.baseLeft.y = figure.baseRight.y)
  (h3 : (figure.baseRight.x - figure.baseLeft.x) = figure.base)
  (h4 : figure.midpointLeft.x = (figure.baseLeft.x + figure.apex.x) / 2)
  (h5 : figure.midpointLeft.y = (figure.baseLeft.y + figure.apex.y) / 2)
  (h6 : figure.midpointRight.x = (figure.baseRight.x + figure.apex.x) / 2)
  (h7 : figure.midpointRight.y = (figure.baseRight.y + figure.apex.y) / 2)
  (h8 : figure.midpointLeft.y = figure.midpointRight.y) :
  countTriangles figure = 5 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_figure_triangle_count_l1444_144478


namespace NUMINAMATH_CALUDE_nuts_in_tree_l1444_144432

theorem nuts_in_tree (squirrels : ℕ) (difference : ℕ) (nuts : ℕ) : 
  squirrels = 4 → 
  squirrels = nuts + difference → 
  difference = 2 → 
  nuts = 2 := by sorry

end NUMINAMATH_CALUDE_nuts_in_tree_l1444_144432


namespace NUMINAMATH_CALUDE_max_sum_consecutive_triples_l1444_144486

/-- Represents a permutation of the digits 1 to 9 -/
def Permutation := Fin 9 → Fin 9

/-- Calculates the sum of seven consecutive three-digit numbers formed from a permutation -/
def sumConsecutiveTriples (p : Permutation) : ℕ :=
  (100 * p 0 + 110 * p 1 + 111 * p 2 + 111 * p 3 + 111 * p 4 + 111 * p 5 + 111 * p 6 + 11 * p 7 + p 8).val

/-- The maximum possible sum of consecutive triples -/
def maxSum : ℕ := 4648

/-- Theorem stating that the maximum sum of consecutive triples is 4648 -/
theorem max_sum_consecutive_triples :
  ∀ p : Permutation, sumConsecutiveTriples p ≤ maxSum :=
sorry

end NUMINAMATH_CALUDE_max_sum_consecutive_triples_l1444_144486


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1444_144419

theorem p_necessary_not_sufficient (p q : Prop) : 
  (∀ (h : p ∧ q), p) ∧ 
  (∃ (h : p), ¬(p ∧ q)) := by
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1444_144419


namespace NUMINAMATH_CALUDE_charles_initial_skittles_l1444_144475

theorem charles_initial_skittles (current_skittles taken_skittles : ℕ) 
  (h1 : current_skittles = 18)
  (h2 : taken_skittles = 7) :
  current_skittles + taken_skittles = 25 := by
  sorry

end NUMINAMATH_CALUDE_charles_initial_skittles_l1444_144475


namespace NUMINAMATH_CALUDE_min_distance_parabola_circle_l1444_144426

/-- The minimum distance between a point on the parabola y^2 = x and a point on the circle (x-3)^2 + y^2 = 1 is 1/2 (√11 - 2). -/
theorem min_distance_parabola_circle :
  let parabola := {p : ℝ × ℝ | p.2^2 = p.1}
  let circle := {q : ℝ × ℝ | (q.1 - 3)^2 + q.2^2 = 1}
  ∃ (d : ℝ), d = (Real.sqrt 11 - 2) / 2 ∧
    ∀ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ parabola → q ∈ circle →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_parabola_circle_l1444_144426


namespace NUMINAMATH_CALUDE_negative_two_times_inequality_l1444_144497

theorem negative_two_times_inequality (m n : ℝ) (h : m > n) : -2 * m < -2 * n := by
  sorry

end NUMINAMATH_CALUDE_negative_two_times_inequality_l1444_144497


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l1444_144420

/-- The surface area of a cube after removing smaller cubes from its corners --/
def surface_area_after_removal (cube_size : ℝ) (corner_size : ℝ) : ℝ :=
  6 * cube_size^2

/-- Theorem: The surface area remains unchanged after corner removal --/
theorem surface_area_unchanged
  (cube_size : ℝ)
  (corner_size : ℝ)
  (h1 : cube_size = 4)
  (h2 : corner_size = 1.5)
  : surface_area_after_removal cube_size corner_size = 96 := by
  sorry

#check surface_area_unchanged

end NUMINAMATH_CALUDE_surface_area_unchanged_l1444_144420


namespace NUMINAMATH_CALUDE_hyperbola_foci_l1444_144468

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

-- Define the foci of the hyperbola
def foci : Set (ℝ × ℝ) :=
  {(5, 0), (-5, 0)}

-- Theorem statement
theorem hyperbola_foci :
  ∀ (x y : ℝ), hyperbola_equation x y → (x, y) ∈ foci :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l1444_144468


namespace NUMINAMATH_CALUDE_roots_position_l1444_144461

theorem roots_position (a b : ℝ) :
  ∃ (x₁ x₂ : ℝ), (x₁ - a) * (x₁ - a - b) = 1 ∧
                  (x₂ - a) * (x₂ - a - b) = 1 ∧
                  x₁ < a ∧ a < x₂ := by
  sorry

end NUMINAMATH_CALUDE_roots_position_l1444_144461


namespace NUMINAMATH_CALUDE_probability_of_valid_pair_l1444_144454

/-- Represents a ball with a color and a label -/
structure Ball where
  color : Bool  -- True for red, False for blue
  label : Nat

/-- The bag of balls -/
def bag : Finset Ball := sorry

/-- The condition for a pair of balls to meet our criteria -/
def validPair (b1 b2 : Ball) : Prop :=
  b1.color ≠ b2.color ∧ b1.label + b2.label ≥ 4

/-- The number of ways to choose 2 balls from the bag -/
def totalChoices : Nat := sorry

/-- The number of valid pairs of balls -/
def validChoices : Nat := sorry

theorem probability_of_valid_pair :
  (validChoices : ℚ) / totalChoices = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_probability_of_valid_pair_l1444_144454


namespace NUMINAMATH_CALUDE_great_pyramid_tallest_duration_l1444_144414

/-- Represents the dimensions and historical facts about the Great Pyramid of Giza -/
structure GreatPyramid where
  height : ℕ
  width : ℕ
  year_built : Int
  year_surpassed : Int
  height_above_500 : height = 500 + 20
  width_relation : width = height + 234
  sum_height_width : height + width = 1274
  built_BC : year_built < 0
  surpassed_AD : year_surpassed > 0

/-- Theorem stating the duration for which the Great Pyramid was the tallest structure -/
theorem great_pyramid_tallest_duration (p : GreatPyramid) : 
  p.year_surpassed - p.year_built = 3871 :=
sorry

end NUMINAMATH_CALUDE_great_pyramid_tallest_duration_l1444_144414


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_of_zeros_l1444_144401

/-- Piecewise function f(x) as defined in the problem -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then k * x^2 + 2 * x - 1
  else if x > 1 then k * x + 1
  else 0  -- Define f(x) as 0 for x ≤ 0 to make it total

/-- Theorem stating the maximum value of 1/x₁ + 1/x₂ -/
theorem max_reciprocal_sum_of_zeros (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0 ∧ 1/x₁ + 1/x₂ ≤ 9/4) ∧
  (∃ k₀ : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k₀ x₁ = 0 ∧ f k₀ x₂ = 0 ∧ 1/x₁ + 1/x₂ = 9/4) :=
by sorry


end NUMINAMATH_CALUDE_max_reciprocal_sum_of_zeros_l1444_144401


namespace NUMINAMATH_CALUDE_range_of_a_l1444_144446

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a + 7)*x + 5 ≤ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : A ⊆ B a → -4 ≤ a ∧ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1444_144446


namespace NUMINAMATH_CALUDE_jogger_ahead_of_train_l1444_144464

/-- Calculates the distance a jogger is ahead of a train given their speeds and the time for the train to pass the jogger. -/
def jogger_distance_ahead (jogger_speed : Real) (train_speed : Real) (train_length : Real) (passing_time : Real) : Real :=
  (train_speed - jogger_speed) * passing_time - train_length

/-- Theorem stating the distance a jogger is ahead of a train under specific conditions. -/
theorem jogger_ahead_of_train (jogger_speed : Real) (train_speed : Real) (train_length : Real) (passing_time : Real)
  (h1 : jogger_speed = 9 * 1000 / 3600)
  (h2 : train_speed = 45 * 1000 / 3600)
  (h3 : train_length = 120)
  (h4 : passing_time = 40.00000000000001) :
  ∃ ε > 0, |jogger_distance_ahead jogger_speed train_speed train_length passing_time - 280| < ε :=
by sorry

end NUMINAMATH_CALUDE_jogger_ahead_of_train_l1444_144464
