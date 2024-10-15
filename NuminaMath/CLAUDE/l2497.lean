import Mathlib

namespace NUMINAMATH_CALUDE_seventh_observation_value_l2497_249799

/-- Given 6 initial observations with an average of 15, prove that adding a 7th observation
    that decreases the overall average by 1 results in the 7th observation having a value of 8. -/
theorem seventh_observation_value (n : ℕ) (initial_average new_average : ℚ) :
  n = 6 →
  initial_average = 15 →
  new_average = initial_average - 1 →
  ∃ x : ℚ, x = 8 ∧ (n : ℚ) * initial_average + x = (n + 1 : ℚ) * new_average :=
by sorry

end NUMINAMATH_CALUDE_seventh_observation_value_l2497_249799


namespace NUMINAMATH_CALUDE_number_to_add_for_divisibility_l2497_249704

theorem number_to_add_for_divisibility (a b : ℕ) (h : b > 0) : 
  ∃ n : ℕ, (a + n) % b = 0 ∧ n = if a % b = 0 then 0 else b - a % b :=
sorry

end NUMINAMATH_CALUDE_number_to_add_for_divisibility_l2497_249704


namespace NUMINAMATH_CALUDE_value_range_of_function_l2497_249735

theorem value_range_of_function : 
  ∀ (y : ℝ), (∃ (x : ℝ), y = (x^2 - 1) / (x^2 + 1)) ↔ -1 ≤ y ∧ y < 1 := by
  sorry

end NUMINAMATH_CALUDE_value_range_of_function_l2497_249735


namespace NUMINAMATH_CALUDE_mistaken_subtraction_l2497_249775

theorem mistaken_subtraction (x : ℤ) : x - 64 = 122 → x - 46 = 140 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_subtraction_l2497_249775


namespace NUMINAMATH_CALUDE_jack_heavier_than_sam_l2497_249747

theorem jack_heavier_than_sam (total_weight jack_weight : ℕ) 
  (h1 : total_weight = 96)
  (h2 : jack_weight = 52) :
  jack_weight - (total_weight - jack_weight) = 8 :=
by sorry

end NUMINAMATH_CALUDE_jack_heavier_than_sam_l2497_249747


namespace NUMINAMATH_CALUDE_max_watching_count_is_five_l2497_249789

/-- Represents a direction a guard can look -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a position on the board -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents a guard on the board -/
structure Guard :=
  (pos : Position)
  (dir : Direction)

/-- The type of a board configuration -/
def Board := Fin 8 → Fin 8 → Guard

/-- Count of guards watching a specific position -/
def watchingCount (b : Board) (p : Position) : Nat :=
  sorry

/-- The maximum k for which every guard is watched by at least k other guards -/
def maxWatchingCount (b : Board) : Nat :=
  sorry

theorem max_watching_count_is_five :
  ∀ b : Board, maxWatchingCount b ≤ 5 ∧ ∃ b' : Board, maxWatchingCount b' = 5 :=
sorry

end NUMINAMATH_CALUDE_max_watching_count_is_five_l2497_249789


namespace NUMINAMATH_CALUDE_seedling_difference_seedling_difference_proof_l2497_249791

theorem seedling_difference : ℕ → ℕ → ℕ → Prop :=
  fun pine_seedlings poplar_multiplier difference =>
    pine_seedlings = 180 →
    poplar_multiplier = 4 →
    difference = poplar_multiplier * pine_seedlings - pine_seedlings →
    difference = 540

-- Proof
theorem seedling_difference_proof : seedling_difference 180 4 540 := by
  sorry

end NUMINAMATH_CALUDE_seedling_difference_seedling_difference_proof_l2497_249791


namespace NUMINAMATH_CALUDE_shoes_sold_main_theorem_l2497_249720

/-- Represents the inventory of a shoe shop -/
structure ShoeInventory where
  large_boots : Nat
  medium_sandals : Nat
  small_sneakers : Nat
  large_sandals : Nat
  medium_boots : Nat
  small_boots : Nat

/-- Calculates the total number of shoes in the inventory -/
def total_shoes (inventory : ShoeInventory) : Nat :=
  inventory.large_boots + inventory.medium_sandals + inventory.small_sneakers +
  inventory.large_sandals + inventory.medium_boots + inventory.small_boots

/-- Theorem: The shop sold 106 pairs of shoes -/
theorem shoes_sold (initial_inventory : ShoeInventory) (pairs_left : Nat) : Nat :=
  let initial_total := total_shoes initial_inventory
  initial_total - pairs_left

/-- Main theorem: The shop sold 106 pairs of shoes -/
theorem main_theorem : shoes_sold
  { large_boots := 22
    medium_sandals := 32
    small_sneakers := 24
    large_sandals := 45
    medium_boots := 35
    small_boots := 26 }
  78 = 106 := by
  sorry

end NUMINAMATH_CALUDE_shoes_sold_main_theorem_l2497_249720


namespace NUMINAMATH_CALUDE_crayon_difference_l2497_249795

theorem crayon_difference (red : ℕ) (yellow : ℕ) (blue : ℕ) : 
  red = 14 → 
  yellow = 32 → 
  yellow = 2 * blue - 6 → 
  blue - red = 5 := by
sorry

end NUMINAMATH_CALUDE_crayon_difference_l2497_249795


namespace NUMINAMATH_CALUDE_special_ellipse_d_value_l2497_249764

/-- An ellipse in the first quadrant tangent to both axes with foci at (5,10) and (d,10) --/
structure Ellipse where
  d : ℝ
  tangent_x : Bool
  tangent_y : Bool
  first_quadrant : Bool
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ

/-- The d value for the special ellipse described in the problem --/
def special_ellipse_d : ℝ := 20

/-- Theorem stating that the d value for the special ellipse is 20 --/
theorem special_ellipse_d_value (e : Ellipse) 
  (h_tangent_x : e.tangent_x = true)
  (h_tangent_y : e.tangent_y = true)
  (h_first_quadrant : e.first_quadrant = true)
  (h_focus1 : e.focus1 = (5, 10))
  (h_focus2 : e.focus2 = (e.d, 10)) :
  e.d = special_ellipse_d := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_d_value_l2497_249764


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2497_249742

theorem arithmetic_sequence_middle_term 
  (a : Fin 5 → ℝ)  -- a is a function from Fin 5 to ℝ, representing the 5 terms
  (h1 : a 0 = -8)  -- first term is -8
  (h2 : a 4 = 10)  -- last term is 10
  (h3 : ∀ i : Fin 4, a (i + 1) - a i = a 1 - a 0)  -- arithmetic sequence condition
  : a 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2497_249742


namespace NUMINAMATH_CALUDE_square_properties_l2497_249706

/-- Given a square with perimeter 48 feet, prove its side length and area. -/
theorem square_properties (perimeter : ℝ) (h : perimeter = 48) :
  ∃ (side_length area : ℝ),
    side_length = 12 ∧
    area = 144 ∧
    perimeter = 4 * side_length ∧
    area = side_length * side_length := by
  sorry

end NUMINAMATH_CALUDE_square_properties_l2497_249706


namespace NUMINAMATH_CALUDE_sandys_change_l2497_249717

/-- Calculates Sandy's change after shopping for toys -/
theorem sandys_change 
  (football_price : ℝ)
  (baseball_price : ℝ)
  (basketball_price : ℝ)
  (football_count : ℕ)
  (baseball_count : ℕ)
  (basketball_count : ℕ)
  (pounds_paid : ℝ)
  (euros_paid : ℝ)
  (h1 : football_price = 9.14)
  (h2 : baseball_price = 6.81)
  (h3 : basketball_price = 7.95)
  (h4 : football_count = 3)
  (h5 : baseball_count = 2)
  (h6 : basketball_count = 4)
  (h7 : pounds_paid = 50)
  (h8 : euros_paid = 20) :
  let pounds_spent := football_price * football_count + baseball_price * baseball_count
  let euros_spent := basketball_price * basketball_count
  let pounds_change := pounds_paid - pounds_spent
  let euros_change := max (euros_paid - euros_spent) 0
  (pounds_change = 8.96 ∧ euros_change = 0) :=
by sorry

end NUMINAMATH_CALUDE_sandys_change_l2497_249717


namespace NUMINAMATH_CALUDE_flea_jump_angle_rational_l2497_249730

/-- A flea jumping between two intersecting lines --/
structure FleaJump where
  α : ℝ  -- Angle between the lines in radians
  jumpLength : ℝ  -- Length of each jump
  returnsToStart : Prop  -- Flea eventually returns to starting point
  noPreviousPosition : Prop  -- Flea never returns to previous position

/-- Theorem stating that if a flea jumps as described, the angle is rational --/
theorem flea_jump_angle_rational (jump : FleaJump) 
  (h1 : jump.jumpLength = 1)
  (h2 : jump.returnsToStart)
  (h3 : jump.noPreviousPosition) :
  ∃ (p q : ℤ), jump.α = (p / q) * (π / 180) :=
sorry

end NUMINAMATH_CALUDE_flea_jump_angle_rational_l2497_249730


namespace NUMINAMATH_CALUDE_population_growth_l2497_249793

theorem population_growth (initial_population : ℝ) : 
  (initial_population * (1 + 0.1)^2 = 16940) → initial_population = 14000 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_l2497_249793


namespace NUMINAMATH_CALUDE_triangle_intersection_ratio_l2497_249703

/-- Given a triangle XYZ, this theorem proves that if point P is on XY with XP:PY = 4:1,
    point Q is on YZ with YQ:QZ = 4:1, and lines PQ and XZ intersect at R,
    then PQ:QR = 4:1. -/
theorem triangle_intersection_ratio (X Y Z P Q R : ℝ × ℝ) : 
  (∃ t : ℝ, P = (1 - t) • X + t • Y ∧ t = 1/5) →  -- P is on XY with XP:PY = 4:1
  (∃ s : ℝ, Q = (1 - s) • Y + s • Z ∧ s = 4/5) →  -- Q is on YZ with YQ:QZ = 4:1
  (∃ u v : ℝ, R = (1 - u) • X + u • Z ∧ R = (1 - v) • P + v • Q) →  -- R is intersection of XZ and PQ
  ∃ k : ℝ, k • (Q - P) = R - Q ∧ k = 1/4 :=  -- PQ:QR = 4:1
by sorry

end NUMINAMATH_CALUDE_triangle_intersection_ratio_l2497_249703


namespace NUMINAMATH_CALUDE_equation_solution_l2497_249776

theorem equation_solution :
  ∃ x : ℝ, (4 * x + 6 * x = 360 - 9 * (x - 4)) ∧ (x = 396 / 19) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2497_249776


namespace NUMINAMATH_CALUDE_number_puzzle_l2497_249736

theorem number_puzzle : 
  ∃ x : ℝ, (x / 5 + 4 = x / 4 - 4) ∧ (x = 160) := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2497_249736


namespace NUMINAMATH_CALUDE_max_single_player_salary_l2497_249726

/-- Represents the number of players in a team -/
def num_players : ℕ := 18

/-- Represents the minimum salary for each player in dollars -/
def min_salary : ℕ := 20000

/-- Represents the maximum total salary for the team in dollars -/
def max_total_salary : ℕ := 800000

/-- Theorem stating the maximum possible salary for a single player -/
theorem max_single_player_salary :
  ∃ (max_salary : ℕ),
    max_salary = 460000 ∧
    max_salary + (num_players - 1) * min_salary = max_total_salary ∧
    ∀ (salary : ℕ),
      salary + (num_players - 1) * min_salary ≤ max_total_salary →
      salary ≤ max_salary :=
by sorry

end NUMINAMATH_CALUDE_max_single_player_salary_l2497_249726


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l2497_249714

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem min_value_geometric_sequence (a : ℕ → ℝ) 
    (h_geo : is_geometric_sequence a)
    (h_pos : ∀ n, a n > 0)
    (h_2018 : a 2018 = Real.sqrt 2 / 2) :
    (1 / a 2017 + 2 / a 2019) ≥ 4 ∧ 
    ∃ a, is_geometric_sequence a ∧ (∀ n, a n > 0) ∧ 
         a 2018 = Real.sqrt 2 / 2 ∧ 1 / a 2017 + 2 / a 2019 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l2497_249714


namespace NUMINAMATH_CALUDE_cos_150_degrees_l2497_249725

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l2497_249725


namespace NUMINAMATH_CALUDE_sqrt_simplification_complex_expression_simplification_square_difference_simplification_l2497_249796

-- Problem 1
theorem sqrt_simplification :
  Real.sqrt 27 + Real.sqrt 3 - Real.sqrt 12 = 2 * Real.sqrt 3 := by sorry

-- Problem 2
theorem complex_expression_simplification :
  1 / Real.sqrt 24 + |Real.sqrt 6 - 3| + (1 / 2)⁻¹ - 2016^0 = 4 - 13 * Real.sqrt 6 / 12 := by sorry

-- Problem 3
theorem square_difference_simplification :
  (Real.sqrt 3 + Real.sqrt 2)^2 - (Real.sqrt 3 - Real.sqrt 2)^2 = 4 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_simplification_complex_expression_simplification_square_difference_simplification_l2497_249796


namespace NUMINAMATH_CALUDE_total_raised_l2497_249773

/-- The amount of money raised by a local business for charity -/
def charity_fundraiser (num_tickets : ℕ) (ticket_price : ℚ) (donation1 : ℚ) (num_donation1 : ℕ) (donation2 : ℚ) : ℚ :=
  num_tickets * ticket_price + num_donation1 * donation1 + donation2

/-- Theorem stating the total amount raised for charity -/
theorem total_raised :
  charity_fundraiser 25 2 15 2 20 = 100 :=
by sorry

end NUMINAMATH_CALUDE_total_raised_l2497_249773


namespace NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_l2497_249734

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_from_perpendicular_lines 
  (m n : Line) (α β : Plane) :
  parallel m n → 
  perpendicular_line_plane m α → 
  perpendicular_line_plane n β → 
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_l2497_249734


namespace NUMINAMATH_CALUDE_ranch_cows_l2497_249752

/-- The number of cows owned by We the People -/
def wtp_cows : ℕ := 17

/-- The number of cows owned by Happy Good Healthy Family -/
def hghf_cows : ℕ := 3 * wtp_cows + 2

/-- The total number of cows in the ranch -/
def total_cows : ℕ := wtp_cows + hghf_cows

theorem ranch_cows : total_cows = 70 := by
  sorry

end NUMINAMATH_CALUDE_ranch_cows_l2497_249752


namespace NUMINAMATH_CALUDE_supermarket_spending_l2497_249733

theorem supermarket_spending (total : ℚ) 
  (h1 : total = 120) 
  (h2 : ∃ (fruits meat bakery candy : ℚ), 
    fruits + meat + bakery + candy = total ∧
    fruits = (1/2) * total ∧
    meat = (1/3) * total ∧
    bakery = (1/10) * total) : 
  ∃ (candy : ℚ), candy = 8 := by
sorry

end NUMINAMATH_CALUDE_supermarket_spending_l2497_249733


namespace NUMINAMATH_CALUDE_solve_slurpee_problem_l2497_249797

def slurpee_problem (initial_amount : ℕ) (slurpee_cost : ℕ) (change : ℕ) : Prop :=
  let amount_spent : ℕ := initial_amount - change
  let num_slurpees : ℕ := amount_spent / slurpee_cost
  num_slurpees = 6

theorem solve_slurpee_problem :
  slurpee_problem 20 2 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_slurpee_problem_l2497_249797


namespace NUMINAMATH_CALUDE_stratified_sampling_sophomores_l2497_249780

theorem stratified_sampling_sophomores 
  (total_students : ℕ) 
  (sophomores : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 1000)
  (h2 : sophomores = 320)
  (h3 : sample_size = 50) : 
  (sophomores * sample_size) / total_students = 16 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_sophomores_l2497_249780


namespace NUMINAMATH_CALUDE_new_line_properties_new_line_equation_correct_l2497_249748

/-- Given two lines in the plane -/
def line1 (x y : ℝ) : Prop := 3 * x + 2 * y - 5 = 0
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0

/-- The intersection point of the two lines -/
def intersection : ℝ × ℝ := (1, 1)

/-- The new line passing through the intersection point and having y-intercept -5 -/
def new_line (x y : ℝ) : Prop := 6 * x - y - 5 = 0

/-- Theorem stating that the new line passes through the intersection point and has y-intercept -5 -/
theorem new_line_properties :
  (line1 intersection.1 intersection.2) ∧
  (line2 intersection.1 intersection.2) ∧
  (new_line intersection.1 intersection.2) ∧
  (new_line 0 (-5)) :=
sorry

/-- Main theorem proving that the new line equation is correct -/
theorem new_line_equation_correct (x y : ℝ) :
  (line1 x y ∧ line2 x y) →
  (∃ t : ℝ, new_line (x + t * (intersection.1 - x)) (y + t * (intersection.2 - y))) :=
sorry

end NUMINAMATH_CALUDE_new_line_properties_new_line_equation_correct_l2497_249748


namespace NUMINAMATH_CALUDE_power_sum_equality_l2497_249718

theorem power_sum_equality : (-1 : ℤ) ^ 47 + 2 ^ (3^3 + 4^2 - 6^2) = 127 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2497_249718


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2497_249708

/-- The speed of a boat in still water, given its speeds with and against a stream -/
theorem boat_speed_in_still_water (along_stream : ℝ) (against_stream : ℝ) 
    (h1 : along_stream = 21) 
    (h2 : against_stream = 9) : 
    (along_stream + against_stream) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2497_249708


namespace NUMINAMATH_CALUDE_cloth_sale_gain_percentage_l2497_249732

/-- Calculates the gain percentage given the profit amount and total amount sold -/
def gainPercentage (profitAmount : ℕ) (totalAmount : ℕ) : ℚ :=
  (profitAmount : ℚ) / (totalAmount : ℚ) * 100

/-- Theorem: The gain percentage is 40% when the profit is 10 and the total amount sold is 25 -/
theorem cloth_sale_gain_percentage :
  gainPercentage 10 25 = 40 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_gain_percentage_l2497_249732


namespace NUMINAMATH_CALUDE_cats_to_dogs_ratio_l2497_249755

theorem cats_to_dogs_ratio (cats : ℕ) (dogs : ℕ) : 
  cats = 16 → dogs = 8 → (cats : ℚ) / dogs = 2 := by
  sorry

end NUMINAMATH_CALUDE_cats_to_dogs_ratio_l2497_249755


namespace NUMINAMATH_CALUDE_sum_interior_angles_formula_l2497_249744

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

/-- Theorem: For a polygon with n sides (where n ≥ 3), 
    the sum of interior angles is (n-2) * 180° -/
theorem sum_interior_angles_formula (n : ℕ) (h : n ≥ 3) : 
  sum_interior_angles n = (n - 2) * 180 := by
  sorry

#check sum_interior_angles_formula

end NUMINAMATH_CALUDE_sum_interior_angles_formula_l2497_249744


namespace NUMINAMATH_CALUDE_find_number_l2497_249768

theorem find_number : ∃ x : ℤ, x - 29 + 64 = 76 ∧ x = 41 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2497_249768


namespace NUMINAMATH_CALUDE_simplify_expression_l2497_249741

theorem simplify_expression (b : ℝ) : (1:ℝ)*(2*b)*(3*b^2)*(4*b^3)*(5*b^4)*(6*b^5) = 720 * b^15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2497_249741


namespace NUMINAMATH_CALUDE_green_balls_count_l2497_249715

theorem green_balls_count (total : ℕ) (white yellow red purple : ℕ) (prob : ℚ) :
  total = 60 ∧
  white = 22 ∧
  yellow = 17 ∧
  red = 3 ∧
  purple = 1 ∧
  prob = 95 / 100 ∧
  (total - red - purple : ℚ) / total = prob →
  total - white - yellow - red - purple = 17 := by
  sorry

end NUMINAMATH_CALUDE_green_balls_count_l2497_249715


namespace NUMINAMATH_CALUDE_merchant_markup_percentage_l2497_249781

/-- Proves that if a merchant marks up goods by x%, then offers a 10% discount,
    and makes a 57.5% profit, the value of x is 75%. -/
theorem merchant_markup_percentage
  (cost_price : ℝ)
  (markup_percentage : ℝ)
  (discount_percentage : ℝ)
  (profit_percentage : ℝ)
  (h1 : discount_percentage = 10)
  (h2 : profit_percentage = 57.5)
  (h3 : cost_price > 0)
  : markup_percentage = 75 :=
by sorry

end NUMINAMATH_CALUDE_merchant_markup_percentage_l2497_249781


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l2497_249774

theorem mean_equality_implies_z_value :
  let mean1 := (5 + 8 + 17) / 3
  let mean2 := (15 + z) / 2
  mean1 = mean2 → z = 5 := by
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l2497_249774


namespace NUMINAMATH_CALUDE_bus_rows_count_l2497_249758

theorem bus_rows_count (total_capacity : ℕ) (row_capacity : ℕ) (h1 : total_capacity = 80) (h2 : row_capacity = 4) :
  total_capacity / row_capacity = 20 :=
by sorry

end NUMINAMATH_CALUDE_bus_rows_count_l2497_249758


namespace NUMINAMATH_CALUDE_constant_pace_run_time_l2497_249731

/-- Represents the time taken to run a certain distance at a constant pace -/
structure RunTime where
  distance : ℝ
  time : ℝ

/-- Calculates the time taken to run a given distance at a constant pace -/
def calculateTime (pace : ℝ) (distance : ℝ) : ℝ :=
  pace * distance

theorem constant_pace_run_time 
  (store_run : RunTime) 
  (friend_house_distance : ℝ) 
  (h1 : store_run.distance = 3) 
  (h2 : store_run.time = 24) 
  (h3 : friend_house_distance = 1.5) :
  calculateTime (store_run.time / store_run.distance) friend_house_distance = 12 := by
  sorry

#check constant_pace_run_time

end NUMINAMATH_CALUDE_constant_pace_run_time_l2497_249731


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_all_zero_l2497_249724

theorem square_sum_zero_implies_all_zero (a b c : ℝ) : 
  a^2 + b^2 + c^2 = 0 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_all_zero_l2497_249724


namespace NUMINAMATH_CALUDE_largest_base6_4digit_in_base10_l2497_249701

def largest_base6_4digit : ℕ := 5 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0

theorem largest_base6_4digit_in_base10 : 
  largest_base6_4digit = 1295 := by sorry

end NUMINAMATH_CALUDE_largest_base6_4digit_in_base10_l2497_249701


namespace NUMINAMATH_CALUDE_goose_price_after_increases_l2497_249757

-- Define the initial prices
def initial_goose_price : ℝ := 0.8
def initial_wine_price : ℝ := 0.4

-- Define the price increase factor
def price_increase_factor : ℝ := 1.2

theorem goose_price_after_increases (goose_price : ℝ) (wine_price : ℝ) :
  goose_price = initial_goose_price ∧ 
  wine_price = initial_wine_price ∧
  goose_price + wine_price = 1 ∧
  goose_price + 0.5 * wine_price = 1 →
  goose_price * price_increase_factor * price_increase_factor < 1 := by
  sorry

#check goose_price_after_increases

end NUMINAMATH_CALUDE_goose_price_after_increases_l2497_249757


namespace NUMINAMATH_CALUDE_naomi_saw_58_wheels_l2497_249711

/-- The number of regular bikes at the park -/
def regular_bikes : ℕ := 7

/-- The number of children's bikes at the park -/
def children_bikes : ℕ := 11

/-- The number of wheels on a regular bike -/
def regular_bike_wheels : ℕ := 2

/-- The number of wheels on a children's bike -/
def children_bike_wheels : ℕ := 4

/-- The total number of wheels Naomi saw at the park -/
def total_wheels : ℕ := regular_bikes * regular_bike_wheels + children_bikes * children_bike_wheels

theorem naomi_saw_58_wheels : total_wheels = 58 := by
  sorry

end NUMINAMATH_CALUDE_naomi_saw_58_wheels_l2497_249711


namespace NUMINAMATH_CALUDE_tournament_probability_l2497_249746

/-- The number of teams in the tournament -/
def num_teams : ℕ := 35

/-- The number of games each team plays -/
def games_per_team : ℕ := num_teams - 1

/-- The total number of games in the tournament -/
def total_games : ℕ := (num_teams * games_per_team) / 2

/-- The probability of a team winning a single game -/
def win_probability : ℚ := 1 / 2

/-- The number of possible outcomes in the tournament -/
def total_outcomes : ℕ := 2^total_games

/-- The number of ways to assign unique victory counts to all teams -/
def unique_victory_assignments : ℕ := num_teams.factorial

theorem tournament_probability : 
  (unique_victory_assignments : ℚ) / total_outcomes = (num_teams.factorial : ℚ) / 2^595 :=
sorry

end NUMINAMATH_CALUDE_tournament_probability_l2497_249746


namespace NUMINAMATH_CALUDE_cistern_fill_time_l2497_249751

/-- Time to fill a cistern with two pipes -/
theorem cistern_fill_time 
  (fill_time_A : ℝ) 
  (empty_time_B : ℝ) 
  (h1 : fill_time_A = 16) 
  (h2 : empty_time_B = 20) : 
  (fill_time_A * empty_time_B) / (empty_time_B - fill_time_A) = 80 :=
by sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l2497_249751


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2497_249709

def M : Set ℝ := {x | x^2 < x}
def N : Set ℝ := {x | x^2 + 2*x - 3 < 0}

theorem union_of_M_and_N : M ∪ N = Set.Ioo (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2497_249709


namespace NUMINAMATH_CALUDE_roots_geometric_sequence_range_l2497_249722

theorem roots_geometric_sequence_range (a b : ℝ) (q : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    (∀ x : ℝ, (x^2 - a*x + 1)*(x^2 - b*x + 1) = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) ∧ 
    (∃ r : ℝ, x₂ = x₁ * r ∧ x₃ = x₂ * r ∧ x₄ = x₃ * r) ∧
    q = r ∧ 
    1/3 ≤ q ∧ q ≤ 2) →
  4 ≤ a*b ∧ a*b ≤ 112/9 := by
sorry

end NUMINAMATH_CALUDE_roots_geometric_sequence_range_l2497_249722


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_l2497_249761

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : Point) : Point :=
  { x := -p.x, y := p.y }

theorem symmetric_point_y_axis :
  let P : Point := { x := 2, y := 1 }
  let P' : Point := reflect_y_axis P
  P'.x = -2 ∧ P'.y = 1 := by sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_l2497_249761


namespace NUMINAMATH_CALUDE_replacement_paint_intensity_l2497_249729

/-- Proves that the intensity of replacement paint is 25% given the original paint intensity,
    new paint intensity after mixing, and the fraction of original paint replaced. -/
theorem replacement_paint_intensity
  (original_intensity : ℝ)
  (new_intensity : ℝ)
  (replaced_fraction : ℝ)
  (h_original : original_intensity = 50)
  (h_new : new_intensity = 40)
  (h_replaced : replaced_fraction = 0.4)
  : (1 - replaced_fraction) * original_intensity + replaced_fraction * 25 = new_intensity :=
by sorry


end NUMINAMATH_CALUDE_replacement_paint_intensity_l2497_249729


namespace NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l2497_249753

theorem largest_n_for_trig_inequality : 
  (∃ (n : ℕ), n > 0 ∧ ∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / (n^2 : ℝ)) ∧ 
  (∀ (n : ℕ), n > 10 → ∃ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n < 1 / (n^2 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l2497_249753


namespace NUMINAMATH_CALUDE_solution_set_implies_ab_value_l2497_249792

theorem solution_set_implies_ab_value (a b : ℝ) : 
  (∀ x, x^2 + 2*a*x - 4*b ≤ 0 ↔ -2 ≤ x ∧ x ≤ 6) → 
  a^b = -8 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_ab_value_l2497_249792


namespace NUMINAMATH_CALUDE_parabola_focus_hyperbola_equation_l2497_249794

-- Part 1: Parabola
theorem parabola_focus (p : ℝ) (h1 : p > 0) :
  (∃ x y : ℝ, y^2 = 2*p*x ∧ 2*x - y - 4 = 0 ∧ x = p/2 ∧ y = 0) →
  p = 4 := by sorry

-- Part 2: Hyperbola
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (b/a = 3/4 ∧ a^2/(a^2 + b^2)^(1/2) = 16/5) →
  (∀ x y : ℝ, x^2/16 - y^2/9 = 1 ↔ x^2/a^2 - y^2/b^2 = 1) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_hyperbola_equation_l2497_249794


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2497_249756

/-- Given two real numbers p and q with arithmetic mean 10, and a third real number r
    such that r - p = 30, prove that the arithmetic mean of q and r is 25. -/
theorem arithmetic_mean_problem (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : r - p = 30) : 
  (q + r) / 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2497_249756


namespace NUMINAMATH_CALUDE_octagon_area_in_circle_l2497_249798

theorem octagon_area_in_circle (r : ℝ) (h : r = 2.5) : 
  let octagon_area := 8 * (r^2 * Real.sin (π/8) * Real.cos (π/8))
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |octagon_area - 17.672| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_octagon_area_in_circle_l2497_249798


namespace NUMINAMATH_CALUDE_german_shepherd_vs_golden_retriever_pups_l2497_249749

/-- The number of pups each breed has -/
structure DogBreedPups where
  husky : Nat
  golden_retriever : Nat
  pitbull : Nat
  german_shepherd : Nat

/-- The number of dogs James has for each breed -/
structure DogCounts where
  huskies : Nat
  golden_retrievers : Nat
  pitbulls : Nat
  german_shepherds : Nat

/-- Calculate the difference in total pups between German shepherds and golden retrievers -/
def pup_difference (breed_pups : DogBreedPups) (counts : DogCounts) : Int :=
  (breed_pups.german_shepherd * counts.german_shepherds) - 
  (breed_pups.golden_retriever * counts.golden_retrievers)

theorem german_shepherd_vs_golden_retriever_pups : 
  ∀ (breed_pups : DogBreedPups) (counts : DogCounts),
  counts.huskies = 5 →
  counts.pitbulls = 2 →
  counts.golden_retrievers = 4 →
  counts.german_shepherds = 3 →
  breed_pups.husky = 4 →
  breed_pups.golden_retriever = breed_pups.husky + 2 →
  breed_pups.pitbull = 3 →
  breed_pups.german_shepherd = breed_pups.pitbull + 3 →
  pup_difference breed_pups counts = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_german_shepherd_vs_golden_retriever_pups_l2497_249749


namespace NUMINAMATH_CALUDE_rational_cube_equality_l2497_249777

theorem rational_cube_equality (a b c : ℚ) 
  (eq1 : (a^2 + 1)^3 = b + 1)
  (eq2 : (b^2 + 1)^3 = c + 1)
  (eq3 : (c^2 + 1)^3 = a + 1) :
  a = 0 ∧ b = 0 ∧ c = 0 := by
sorry

end NUMINAMATH_CALUDE_rational_cube_equality_l2497_249777


namespace NUMINAMATH_CALUDE_locus_is_circle_l2497_249721

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

/-- Calculates the sum of squares of distances from a point to the vertices of an isosceles triangle -/
def sumOfSquaredDistances (p : Point) (t : IsoscelesTriangle) : ℝ :=
  3 * p.x^2 + 4 * p.y^2 - 2 * t.height * p.y + t.height^2 + t.base^2

/-- Theorem: The locus of points with constant sum of squared distances to the vertices of an isosceles triangle is a circle iff the sum exceeds h^2 + b^2 -/
theorem locus_is_circle (t : IsoscelesTriangle) (a : ℝ) :
  (∃ (center : Point) (radius : ℝ), ∀ (p : Point), 
    sumOfSquaredDistances p t = a ↔ (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2) ↔ 
  a > t.height^2 + t.base^2 := by
  sorry

end NUMINAMATH_CALUDE_locus_is_circle_l2497_249721


namespace NUMINAMATH_CALUDE_greenleaf_academy_history_class_l2497_249712

/-- The number of students in the history class at Greenleaf Academy -/
def history_class_size : ℕ := by sorry

theorem greenleaf_academy_history_class :
  let total_students : ℕ := 70
  let both_subjects : ℕ := 10
  let geography_only : ℕ := 16  -- Derived from the solution, but mathematically necessary
  let history_only : ℕ := total_students - both_subjects - geography_only
  let history_class_size : ℕ := history_only + both_subjects
  let geography_class_size : ℕ := geography_only + both_subjects
  (total_students = geography_only + history_only + both_subjects) ∧
  (history_class_size = 2 * geography_class_size) →
  history_class_size = 52 :=
by sorry

end NUMINAMATH_CALUDE_greenleaf_academy_history_class_l2497_249712


namespace NUMINAMATH_CALUDE_aunts_gift_amount_l2497_249738

def shirts_cost : ℕ := 5
def shirts_price : ℕ := 5
def pants_price : ℕ := 26
def remaining_money : ℕ := 20

theorem aunts_gift_amount : 
  shirts_cost * shirts_price + pants_price + remaining_money = 71 := by
  sorry

end NUMINAMATH_CALUDE_aunts_gift_amount_l2497_249738


namespace NUMINAMATH_CALUDE_eldoria_license_plates_l2497_249710

/-- The number of possible uppercase letters in a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate. -/
def num_digits : ℕ := 10

/-- The number of letter positions in a license plate. -/
def letter_positions : ℕ := 3

/-- The number of digit positions in a license plate. -/
def digit_positions : ℕ := 4

/-- The total number of possible license plates in Eldoria. -/
def total_license_plates : ℕ := num_letters ^ letter_positions * num_digits ^ digit_positions

theorem eldoria_license_plates :
  total_license_plates = 175760000 := by
  sorry

end NUMINAMATH_CALUDE_eldoria_license_plates_l2497_249710


namespace NUMINAMATH_CALUDE_problem_statement_l2497_249772

theorem problem_statement (a b : ℝ) : 
  |a - 2| + (b + 1/2)^2 = 0 → a^2022 * b^2023 = -1/2 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2497_249772


namespace NUMINAMATH_CALUDE_bigger_part_is_38_l2497_249770

theorem bigger_part_is_38 (x y : ℕ) (h1 : x + y = 56) (h2 : 10 * x + 22 * y = 780) :
  max x y = 38 := by
sorry

end NUMINAMATH_CALUDE_bigger_part_is_38_l2497_249770


namespace NUMINAMATH_CALUDE_rachel_initial_lives_l2497_249766

/-- Rachel's initial number of lives -/
def initial_lives : ℕ := 10

/-- The number of lives Rachel lost -/
def lives_lost : ℕ := 4

/-- The number of lives Rachel gained -/
def lives_gained : ℕ := 26

/-- The final number of lives Rachel had -/
def final_lives : ℕ := 32

/-- Theorem stating that Rachel's initial number of lives was 10 -/
theorem rachel_initial_lives :
  initial_lives = 10 ∧
  final_lives = initial_lives - lives_lost + lives_gained :=
sorry

end NUMINAMATH_CALUDE_rachel_initial_lives_l2497_249766


namespace NUMINAMATH_CALUDE_exponential_inequality_l2497_249713

theorem exponential_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (3 : ℝ)^b < (3 : ℝ)^a ∧ (3 : ℝ)^a < (4 : ℝ)^a :=
by sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2497_249713


namespace NUMINAMATH_CALUDE_top_is_nine_l2497_249786

/-- Represents a valid labeling of the figure -/
structure Labeling where
  labels : Fin 9 → Fin 9
  bijective : Function.Bijective labels
  equal_sums : ∃ (s : ℕ), 
    (labels 0 + labels 1 + labels 3 + labels 4 = s) ∧
    (labels 1 + labels 2 + labels 4 + labels 5 = s) ∧
    (labels 0 + labels 3 + labels 6 = s) ∧
    (labels 1 + labels 4 + labels 7 = s) ∧
    (labels 2 + labels 5 + labels 8 = s) ∧
    (labels 3 + labels 4 + labels 5 = s)

/-- The theorem stating that the top number is always 9 in a valid labeling -/
theorem top_is_nine (l : Labeling) : l.labels 0 = 9 := by
  sorry

end NUMINAMATH_CALUDE_top_is_nine_l2497_249786


namespace NUMINAMATH_CALUDE_extra_workers_for_clay_soil_l2497_249700

/-- Represents the digging problem with different soil types and worker requirements -/
structure DiggingProblem where
  sandy_workers : ℕ
  sandy_hours : ℕ
  clay_time_factor : ℕ
  new_hours : ℕ

/-- Calculates the number of extra workers needed for the clay soil digging task -/
def extra_workers_needed (p : DiggingProblem) : ℕ :=
  let sandy_man_hours := p.sandy_workers * p.sandy_hours
  let clay_man_hours := sandy_man_hours * p.clay_time_factor
  let total_workers_needed := clay_man_hours / p.new_hours
  total_workers_needed - p.sandy_workers

/-- Theorem stating that given the problem conditions, 75 extra workers are needed -/
theorem extra_workers_for_clay_soil : 
  let p : DiggingProblem := {
    sandy_workers := 45,
    sandy_hours := 8,
    clay_time_factor := 2,
    new_hours := 6
  }
  extra_workers_needed p = 75 := by sorry

end NUMINAMATH_CALUDE_extra_workers_for_clay_soil_l2497_249700


namespace NUMINAMATH_CALUDE_total_distance_driven_l2497_249719

/-- The total distance driven by Renaldo and Ernesto -/
def total_distance (renaldo_distance : ℝ) (ernesto_distance : ℝ) : ℝ :=
  renaldo_distance + ernesto_distance

/-- Theorem stating the total distance driven by Renaldo and Ernesto -/
theorem total_distance_driven :
  let renaldo_distance : ℝ := 15
  let ernesto_distance : ℝ := (1/3 * renaldo_distance) + 7
  total_distance renaldo_distance ernesto_distance = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_driven_l2497_249719


namespace NUMINAMATH_CALUDE_function_properties_l2497_249767

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x + m

theorem function_properties (m : ℝ) :
  (∀ x > 0, f m x ≤ 0) →
  m = 1 ∧ ∀ a b, 0 < a → a < b → (f m b - f m a) / (b - a) < 1 / (a * (a + 1)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2497_249767


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l2497_249716

-- Define a function to convert binary to decimal
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

-- Define the binary numbers
def b1101 : List Bool := [true, false, true, true]
def b1111 : List Bool := [true, true, true, true]
def b1001 : List Bool := [true, false, false, true]
def b10 : List Bool := [false, true]
def b1010 : List Bool := [false, true, false, true]

-- State the theorem
theorem binary_arithmetic_equality :
  (binary_to_decimal b1101 + binary_to_decimal b1111) -
  (binary_to_decimal b1001 * binary_to_decimal b10) =
  binary_to_decimal b1010 := by
  sorry


end NUMINAMATH_CALUDE_binary_arithmetic_equality_l2497_249716


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2497_249788

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (h_sum : a * b + b * c + a * c = 1) :
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) ≥ (9 + 3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2497_249788


namespace NUMINAMATH_CALUDE_octal_addition_sum_l2497_249785

/-- Given an octal addition 3XY₈ + 52₈ = 4X3₈, prove that X + Y = 1 in base 10 -/
theorem octal_addition_sum (X Y : ℕ) : 
  (3 * 8^2 + X * 8 + Y) + (5 * 8 + 2) = 4 * 8^2 + X * 8 + 3 → X + Y = 1 := by
  sorry

end NUMINAMATH_CALUDE_octal_addition_sum_l2497_249785


namespace NUMINAMATH_CALUDE_line_arrangement_result_l2497_249707

/-- The number of ways to arrange 3 boys and 3 girls in a line with two girls together -/
def line_arrangement (num_boys num_girls : ℕ) : ℕ :=
  -- Define the function here
  sorry

/-- Theorem stating that the number of arrangements is 432 -/
theorem line_arrangement_result : line_arrangement 3 3 = 432 := by
  sorry

end NUMINAMATH_CALUDE_line_arrangement_result_l2497_249707


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2497_249750

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x + 2 > 0) ↔ (0 ≤ m ∧ m < 8) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2497_249750


namespace NUMINAMATH_CALUDE_pencil_count_original_pencils_count_l2497_249778

/-- The number of pencils originally in the drawer -/
def original_pencils : ℕ := sorry

/-- The number of pencils Tim added to the drawer -/
def added_pencils : ℕ := 3

/-- The total number of pencils in the drawer after Tim added some -/
def total_pencils : ℕ := 5

/-- Theorem stating that the original number of pencils plus the added pencils equals the total pencils -/
theorem pencil_count : original_pencils + added_pencils = total_pencils := by sorry

/-- Theorem proving that the original number of pencils in the drawer was 2 -/
theorem original_pencils_count : original_pencils = 2 := by sorry

end NUMINAMATH_CALUDE_pencil_count_original_pencils_count_l2497_249778


namespace NUMINAMATH_CALUDE_sin_150_minus_alpha_l2497_249784

theorem sin_150_minus_alpha (α : Real) (h : α = 240 * Real.pi / 180) :
  Real.sin (150 * Real.pi / 180 - α) = -1 := by sorry

end NUMINAMATH_CALUDE_sin_150_minus_alpha_l2497_249784


namespace NUMINAMATH_CALUDE_min_time_to_target_l2497_249769

/-- Represents the number of steps to the right per minute -/
def right_steps : ℕ := 47

/-- Represents the number of steps to the left per minute -/
def left_steps : ℕ := 37

/-- Represents the target position (one step to the right) -/
def target : ℤ := 1

/-- Theorem stating the minimum time to reach the target position -/
theorem min_time_to_target :
  ∃ (x y : ℕ), 
    right_steps * x - left_steps * y = target ∧
    (∀ (a b : ℕ), right_steps * a - left_steps * b = target → x + y ≤ a + b) ∧
    x + y = 59 := by
  sorry

end NUMINAMATH_CALUDE_min_time_to_target_l2497_249769


namespace NUMINAMATH_CALUDE_outfits_count_l2497_249723

/-- The number of outfits that can be made with given numbers of shirts, pants, and hats -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (hats : ℕ) : ℕ :=
  shirts * pants * hats

/-- Theorem stating that the number of outfits with 4 shirts, 5 pants, and 3 hats is 60 -/
theorem outfits_count :
  number_of_outfits 4 5 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l2497_249723


namespace NUMINAMATH_CALUDE_age_of_other_man_l2497_249705

/-- Proves that the age of the other replaced man is 20 years old given the problem conditions -/
theorem age_of_other_man (n : ℕ) (avg_increase : ℝ) (age_one_man : ℕ) (avg_age_women : ℝ) : 
  n = 8 ∧ 
  avg_increase = 2 ∧ 
  age_one_man = 22 ∧ 
  avg_age_women = 29 → 
  ∃ (age_other_man : ℕ), 
    age_other_man = 20 ∧ 
    2 * avg_age_women - (age_one_man + age_other_man) = n * avg_increase :=
by sorry

end NUMINAMATH_CALUDE_age_of_other_man_l2497_249705


namespace NUMINAMATH_CALUDE_total_spent_on_toys_l2497_249728

-- Define the cost of toy cars
def toy_cars_cost : ℚ := 14.88

-- Define the cost of toy trucks
def toy_trucks_cost : ℚ := 5.86

-- Define the total cost of toys
def total_toys_cost : ℚ := toy_cars_cost + toy_trucks_cost

-- Theorem to prove
theorem total_spent_on_toys :
  total_toys_cost = 20.74 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_on_toys_l2497_249728


namespace NUMINAMATH_CALUDE_opinion_change_percentage_l2497_249782

theorem opinion_change_percentage
  (physics_initial_enjoy : ℝ)
  (physics_initial_dislike : ℝ)
  (physics_final_enjoy : ℝ)
  (physics_final_dislike : ℝ)
  (chem_initial_enjoy : ℝ)
  (chem_initial_dislike : ℝ)
  (chem_final_enjoy : ℝ)
  (chem_final_dislike : ℝ)
  (h1 : physics_initial_enjoy = 40)
  (h2 : physics_initial_dislike = 60)
  (h3 : physics_final_enjoy = 75)
  (h4 : physics_final_dislike = 25)
  (h5 : chem_initial_enjoy = 30)
  (h6 : chem_initial_dislike = 70)
  (h7 : chem_final_enjoy = 65)
  (h8 : chem_final_dislike = 35)
  (h9 : physics_initial_enjoy + physics_initial_dislike = 100)
  (h10 : physics_final_enjoy + physics_final_dislike = 100)
  (h11 : chem_initial_enjoy + chem_initial_dislike = 100)
  (h12 : chem_final_enjoy + chem_final_dislike = 100) :
  ∃ (min_change max_change : ℝ),
    min_change = 70 ∧
    max_change = 70 ∧
    (∀ (actual_change : ℝ),
      actual_change ≥ min_change ∧
      actual_change ≤ max_change) :=
by sorry

end NUMINAMATH_CALUDE_opinion_change_percentage_l2497_249782


namespace NUMINAMATH_CALUDE_one_sixth_percent_of_180_l2497_249759

theorem one_sixth_percent_of_180 : (1 / 6 : ℚ) / 100 * 180 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_one_sixth_percent_of_180_l2497_249759


namespace NUMINAMATH_CALUDE_jakes_desired_rate_l2497_249763

/-- Jake's hourly rate for planting flowers -/
def jakes_hourly_rate (total_charge : ℚ) (hours_worked : ℚ) : ℚ :=
  total_charge / hours_worked

/-- Theorem: Jake's hourly rate for planting flowers is $22.50 -/
theorem jakes_desired_rate :
  jakes_hourly_rate 45 2 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_jakes_desired_rate_l2497_249763


namespace NUMINAMATH_CALUDE_diver_B_depth_l2497_249771

/-- The depth of diver A in meters -/
def depth_A : ℝ := -55

/-- The vertical distance between diver B and diver A in meters -/
def distance_B_above_A : ℝ := 5

/-- The depth of diver B in meters -/
def depth_B : ℝ := depth_A + distance_B_above_A

theorem diver_B_depth : depth_B = -50 := by
  sorry

end NUMINAMATH_CALUDE_diver_B_depth_l2497_249771


namespace NUMINAMATH_CALUDE_final_cell_count_l2497_249745

/-- Calculates the number of cells after a given number of days, 
    given an initial population and a tripling period. -/
def cell_population (initial_cells : ℕ) (tripling_period : ℕ) (total_days : ℕ) : ℕ :=
  initial_cells * (3 ^ (total_days / tripling_period))

/-- Theorem stating that given the specific conditions of the problem, 
    the final cell population after 9 days is 45. -/
theorem final_cell_count : cell_population 5 3 9 = 45 := by
  sorry

end NUMINAMATH_CALUDE_final_cell_count_l2497_249745


namespace NUMINAMATH_CALUDE_perpendicular_lines_l2497_249779

theorem perpendicular_lines (x y : ℝ) : 
  let angle1 : ℝ := 50 + x - y
  let angle2 : ℝ := angle1 - (10 + 2*x - 2*y)
  angle1 + angle2 = 90 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l2497_249779


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_l2497_249760

theorem purely_imaginary_complex (m : ℝ) : 
  let z : ℂ := (m + Complex.I) / (1 + Complex.I)
  (∃ (y : ℝ), z = Complex.I * y) ↔ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_l2497_249760


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2497_249737

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 6*p^2 + 11*p - 6 = 0 →
  q^3 - 6*q^2 + 11*q - 6 = 0 →
  r^3 - 6*r^2 + 11*r - 6 = 0 →
  p * q / r + p * r / q + q * r / p = 49 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2497_249737


namespace NUMINAMATH_CALUDE_clock_90_degree_times_l2497_249740

/-- The angle between the hour hand and minute hand at time t minutes after 12:00 -/
def angle_between (t : ℝ) : ℝ :=
  |6 * t - 0.5 * t|

/-- The times when the hour hand and minute hand form a 90° angle after 12:00 -/
theorem clock_90_degree_times :
  ∃ (t₁ t₂ : ℝ), t₁ < t₂ ∧
  angle_between t₁ = 90 ∧
  angle_between t₂ = 90 ∧
  t₁ = 180 / 11 ∧
  t₂ = 540 / 11 :=
sorry

end NUMINAMATH_CALUDE_clock_90_degree_times_l2497_249740


namespace NUMINAMATH_CALUDE_probability_at_least_one_female_l2497_249790

def total_students : ℕ := 5
def male_students : ℕ := 3
def female_students : ℕ := 2
def students_to_select : ℕ := 2

theorem probability_at_least_one_female :
  (1 : ℚ) - (Nat.choose male_students students_to_select : ℚ) / (Nat.choose total_students students_to_select : ℚ) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_female_l2497_249790


namespace NUMINAMATH_CALUDE_complex_square_one_plus_i_l2497_249762

theorem complex_square_one_plus_i : (1 + Complex.I) ^ 2 = 2 * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_square_one_plus_i_l2497_249762


namespace NUMINAMATH_CALUDE_skee_ball_tickets_value_l2497_249787

/-- The number of tickets Luke won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 2

/-- The cost of one candy in tickets -/
def candy_cost : ℕ := 3

/-- The number of candies Luke could buy -/
def candies_bought : ℕ := 5

/-- The number of tickets Luke won playing 'skee ball' -/
def skee_ball_tickets : ℕ := candies_bought * candy_cost - whack_a_mole_tickets

theorem skee_ball_tickets_value : skee_ball_tickets = 13 := by
  sorry

end NUMINAMATH_CALUDE_skee_ball_tickets_value_l2497_249787


namespace NUMINAMATH_CALUDE_scarves_per_yarn_l2497_249783

/-- Given the total number of scarves and yarns, calculate the number of scarves per yarn -/
theorem scarves_per_yarn (total_scarves total_yarns : ℕ) 
  (h1 : total_scarves = 36)
  (h2 : total_yarns = 12) :
  total_scarves / total_yarns = 3 := by
  sorry

#eval 36 / 12  -- This should output 3

end NUMINAMATH_CALUDE_scarves_per_yarn_l2497_249783


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2497_249702

def A : Set ℕ := {1, 2}
def B : Set ℕ := {x | 2^x = 8}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2497_249702


namespace NUMINAMATH_CALUDE_sin_cos_sum_fifteen_seventyfive_degrees_l2497_249727

theorem sin_cos_sum_fifteen_seventyfive_degrees : 
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (75 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_fifteen_seventyfive_degrees_l2497_249727


namespace NUMINAMATH_CALUDE_potato_difference_l2497_249739

/-- The number of potato wedges -/
def x : ℕ := 8 * 13

/-- The number of potatoes used for french fries or potato chips -/
def k : ℕ := (67 - 13) / 2

/-- The number of potato chips -/
def z : ℕ := 20 * k

/-- The difference between the number of potato chips and potato wedges -/
def d : ℤ := z - x

theorem potato_difference : d = 436 := by
  sorry

end NUMINAMATH_CALUDE_potato_difference_l2497_249739


namespace NUMINAMATH_CALUDE_wilson_family_ages_l2497_249754

theorem wilson_family_ages : ∃ (w e j h t d : ℕ),
  (w > 0) ∧ (e > 0) ∧ (j > 0) ∧ (h > 0) ∧ (t > 0) ∧ (d > 0) ∧
  (w / 2 = e + j + h) ∧
  (w + 5 = (e + 5) + (j + 5) + (h + 5) + 0) ∧
  (e + j + h + t + d = 2 * w) ∧
  (w = e + j) ∧
  (e = t + d) := by
  sorry

end NUMINAMATH_CALUDE_wilson_family_ages_l2497_249754


namespace NUMINAMATH_CALUDE_trapezoid_circles_problem_l2497_249765

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Check if four points are concyclic -/
def are_concyclic (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Check if a circle is tangent to a line -/
def is_tangent (c : Circle) (l : Line) : Prop := sorry

/-- Check if two line segments are parallel -/
def are_parallel (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculate the ratio of two line segments -/
def segment_ratio (p1 p2 p3 : Point) : ℝ := sorry

theorem trapezoid_circles_problem 
  (A B C D E : Point) 
  (circle1 circle2 : Circle) 
  (line_CD : Line) :
  are_parallel A D B C →
  E.x > B.x ∧ E.x < C.x →
  are_concyclic A C D E →
  circle2.center = circle1.center →
  is_tangent circle2 line_CD →
  distance A B = 12 →
  segment_ratio B E C = 4/5 →
  distance B C = 36 ∧ 
  2/3 < circle1.radius / circle2.radius ∧ 
  circle1.radius / circle2.radius < 4/3 := by sorry

end NUMINAMATH_CALUDE_trapezoid_circles_problem_l2497_249765


namespace NUMINAMATH_CALUDE_arithmetic_square_root_is_function_l2497_249743

theorem arithmetic_square_root_is_function : 
  ∀ (x : ℝ), x > 0 → ∃! (y : ℝ), y > 0 ∧ y^2 = x :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_is_function_l2497_249743
