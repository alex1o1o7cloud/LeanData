import Mathlib

namespace NUMINAMATH_CALUDE_original_price_calculation_l2908_290851

theorem original_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 220)
  (h2 : profit_percentage = 0.1) : 
  ∃ (original_price : ℝ), 
    selling_price = original_price * (1 + profit_percentage) ∧ 
    original_price = 200 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2908_290851


namespace NUMINAMATH_CALUDE_suit_pants_cost_l2908_290831

theorem suit_pants_cost (budget initial_budget remaining : ℕ) 
  (shirt_cost coat_cost socks_cost belt_cost shoes_cost : ℕ) :
  initial_budget = 200 →
  shirt_cost = 30 →
  coat_cost = 38 →
  socks_cost = 11 →
  belt_cost = 18 →
  shoes_cost = 41 →
  remaining = 16 →
  ∃ (pants_cost : ℕ),
    pants_cost = initial_budget - (shirt_cost + coat_cost + socks_cost + belt_cost + shoes_cost + remaining) ∧
    pants_cost = 46 :=
by sorry

end NUMINAMATH_CALUDE_suit_pants_cost_l2908_290831


namespace NUMINAMATH_CALUDE_target_score_proof_l2908_290861

theorem target_score_proof (a b c : ℕ) 
  (h1 : 2 * b + c = 29) 
  (h2 : 2 * a + c = 43) : 
  a + b + c = 36 := by
sorry

end NUMINAMATH_CALUDE_target_score_proof_l2908_290861


namespace NUMINAMATH_CALUDE_salary_proof_l2908_290888

/-- Represents the man's salary in dollars -/
def salary : ℝ := 190000

/-- Theorem stating that given the spending conditions, the salary is $190000 -/
theorem salary_proof :
  let food_expense := (1 / 5 : ℝ) * salary
  let rent_expense := (1 / 10 : ℝ) * salary
  let clothes_expense := (3 / 5 : ℝ) * salary
  let remaining := salary - (food_expense + rent_expense + clothes_expense)
  remaining = 19000 := by sorry

end NUMINAMATH_CALUDE_salary_proof_l2908_290888


namespace NUMINAMATH_CALUDE_factory_weekly_production_l2908_290858

/-- Represents a toy production line with a daily production rate -/
structure ProductionLine where
  dailyRate : ℕ

/-- Represents a factory with multiple production lines -/
structure Factory where
  lines : List ProductionLine
  daysPerWeek : ℕ

/-- Calculates the total weekly production of a factory -/
def weeklyProduction (factory : Factory) : ℕ :=
  (factory.lines.map (λ line => line.dailyRate * factory.daysPerWeek)).sum

/-- The theorem stating the total weekly production of the given factory -/
theorem factory_weekly_production :
  let lineA : ProductionLine := ⟨1500⟩
  let lineB : ProductionLine := ⟨1800⟩
  let lineC : ProductionLine := ⟨2200⟩
  let factory : Factory := ⟨[lineA, lineB, lineC], 5⟩
  weeklyProduction factory = 27500 := by
  sorry


end NUMINAMATH_CALUDE_factory_weekly_production_l2908_290858


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l2908_290866

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, |x^2 + 3*a*x + 4*a| ≤ 3) ↔ (a = 8 + 2*Real.sqrt 13 ∨ a = 8 - 2*Real.sqrt 13) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l2908_290866


namespace NUMINAMATH_CALUDE_raft_cannot_turn_l2908_290885

/-- A raft is a shape with a measurable area -/
class Raft :=
  (area : ℝ)

/-- A canal is a path with a width and ability to turn -/
class Canal :=
  (width : ℝ)
  (turn_angle : ℝ)

/-- Determines if a raft can turn in a given canal -/
def can_turn (r : Raft) (c : Canal) : Prop :=
  sorry

/-- Theorem: A raft with area ≥ 2√2 cannot turn in a canal of width 1 with a 90° turn -/
theorem raft_cannot_turn (r : Raft) (c : Canal) :
  r.area ≥ 2 * Real.sqrt 2 →
  c.width = 1 →
  c.turn_angle = Real.pi / 2 →
  ¬(can_turn r c) :=
sorry

end NUMINAMATH_CALUDE_raft_cannot_turn_l2908_290885


namespace NUMINAMATH_CALUDE_sum_interior_angles_convex_polygon_l2908_290833

/-- The sum of interior angles of a convex polygon with n sides, in degrees -/
def sumInteriorAngles (n : ℕ) : ℝ :=
  180 * (n - 2)

/-- Theorem: The sum of interior angles of a convex n-gon is 180 * (n - 2) degrees -/
theorem sum_interior_angles_convex_polygon (n : ℕ) (h : n ≥ 3) :
  sumInteriorAngles n = 180 * (n - 2) := by
  sorry

#check sum_interior_angles_convex_polygon

end NUMINAMATH_CALUDE_sum_interior_angles_convex_polygon_l2908_290833


namespace NUMINAMATH_CALUDE_bert_stamp_cost_l2908_290897

/-- The total cost of stamps Bert purchased -/
def total_cost (type_a_count type_b_count type_c_count : ℕ) 
               (type_a_price type_b_price type_c_price : ℕ) : ℕ :=
  type_a_count * type_a_price + 
  type_b_count * type_b_price + 
  type_c_count * type_c_price

/-- Theorem stating the total cost of Bert's stamp purchase -/
theorem bert_stamp_cost : 
  total_cost 150 90 60 2 3 5 = 870 := by
  sorry

end NUMINAMATH_CALUDE_bert_stamp_cost_l2908_290897


namespace NUMINAMATH_CALUDE_jill_peaches_l2908_290832

theorem jill_peaches (steven_peaches : ℕ) (jake_fewer : ℕ) (jake_more : ℕ)
  (h1 : steven_peaches = 14)
  (h2 : jake_fewer = 6)
  (h3 : jake_more = 3)
  : steven_peaches - jake_fewer - jake_more = 5 := by
  sorry

end NUMINAMATH_CALUDE_jill_peaches_l2908_290832


namespace NUMINAMATH_CALUDE_distance_to_origin_l2908_290896

/-- The distance from point P(3,4) to the origin in the Cartesian coordinate system is 5. -/
theorem distance_to_origin : Real.sqrt (3^2 + 4^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l2908_290896


namespace NUMINAMATH_CALUDE_smaller_factor_of_4536_l2908_290874

theorem smaller_factor_of_4536 (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4536 → 
  min a b = 63 := by
sorry

end NUMINAMATH_CALUDE_smaller_factor_of_4536_l2908_290874


namespace NUMINAMATH_CALUDE_complex_minimum_value_l2908_290824

theorem complex_minimum_value (z : ℂ) (h : Complex.abs (z - (5 + I)) = 5) :
  Complex.abs (z - (1 - 2*I))^2 + Complex.abs (z - (9 + 4*I))^2 = 100 :=
by sorry

end NUMINAMATH_CALUDE_complex_minimum_value_l2908_290824


namespace NUMINAMATH_CALUDE_statement_2_statement_4_l2908_290812

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Statement ②
theorem statement_2 
  (l m : Line) (α β : Plane) 
  (h1 : subset l α) 
  (h2 : parallel_line_plane l β) 
  (h3 : intersect α β m) : 
  parallel l m :=
sorry

-- Statement ④
theorem statement_4 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicular l α) 
  (h2 : parallel l m) 
  (h3 : parallel_plane α β) : 
  perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_statement_2_statement_4_l2908_290812


namespace NUMINAMATH_CALUDE_bobby_shoe_cost_l2908_290889

/-- Calculates the total cost of Bobby's handmade shoes -/
def calculate_total_cost (mold_cost : ℝ) (material_cost : ℝ) (material_discount : ℝ) 
  (hourly_rate : ℝ) (rate_increase : ℝ) (work_hours : ℝ) (work_discount : ℝ) (tax_rate : ℝ) : ℝ :=
  let discounted_material := material_cost * (1 - material_discount)
  let new_hourly_rate := hourly_rate + rate_increase
  let work_cost := work_hours * new_hourly_rate * work_discount
  let subtotal := mold_cost + discounted_material + work_cost
  let total := subtotal * (1 + tax_rate)
  total

/-- Theorem stating that Bobby's total cost is $1005.40 -/
theorem bobby_shoe_cost : 
  calculate_total_cost 250 150 0.2 75 10 8 0.8 0.1 = 1005.40 := by
  sorry

end NUMINAMATH_CALUDE_bobby_shoe_cost_l2908_290889


namespace NUMINAMATH_CALUDE_fraction_of_week_worked_l2908_290880

/-- Proves that given a usual work week of 40 hours, an hourly rate of $15, and a weekly salary of $480, the fraction of the usual week worked is 4/5. -/
theorem fraction_of_week_worked 
  (usual_hours : ℕ) 
  (hourly_rate : ℚ) 
  (weekly_salary : ℚ) 
  (h1 : usual_hours = 40)
  (h2 : hourly_rate = 15)
  (h3 : weekly_salary = 480) :
  (weekly_salary / hourly_rate) / usual_hours = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_week_worked_l2908_290880


namespace NUMINAMATH_CALUDE_no_solution_condition_l2908_290887

theorem no_solution_condition (b : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - b) ≠ 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l2908_290887


namespace NUMINAMATH_CALUDE_class_fraction_proof_l2908_290876

theorem class_fraction_proof (G : ℚ) (B : ℚ) (T : ℚ) (h1 : B / G = 3 / 2) (h2 : T = B + G) :
  (G / 2) / T = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_class_fraction_proof_l2908_290876


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_sixth_l2908_290853

theorem cos_alpha_plus_pi_sixth (α : Real) :
  (∃ (x y : Real), x = 1 ∧ y = 2 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos (α + π/6) = (Real.sqrt 15 - 2 * Real.sqrt 5) / 10 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_sixth_l2908_290853


namespace NUMINAMATH_CALUDE_largest_non_60multiple_composite_sum_l2908_290862

/-- A positive integer is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop := ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- A function that represents the sum of a positive integral multiple of 60 and a positive composite integer -/
def SumOf60MultipleAndComposite (k m : ℕ) : ℕ := 60 * (k + 1) + m

theorem largest_non_60multiple_composite_sum :
  ∀ n : ℕ, n > 5 →
    ∃ k m : ℕ, IsComposite m ∧ n = SumOf60MultipleAndComposite k m :=
by sorry

end NUMINAMATH_CALUDE_largest_non_60multiple_composite_sum_l2908_290862


namespace NUMINAMATH_CALUDE_car_trip_duration_l2908_290837

/-- Represents the duration of a car trip with varying speeds -/
def car_trip (initial_speed initial_duration additional_speed average_speed : ℝ) : Prop :=
  ∃ (total_time additional_time : ℝ),
    total_time > 0 ∧
    additional_time ≥ 0 ∧
    total_time = initial_duration + additional_time ∧
    (initial_speed * initial_duration + additional_speed * additional_time) / total_time = average_speed

/-- The car trip lasts 12 hours given the specified conditions -/
theorem car_trip_duration :
  car_trip 45 4 75 65 → ∃ (total_time : ℝ), total_time = 12 :=
by sorry

end NUMINAMATH_CALUDE_car_trip_duration_l2908_290837


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l2908_290873

-- Define sets A and B
def A : Set ℝ := {x | Real.sqrt (x - 1) < Real.sqrt 2}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem A_intersect_B_eq_open_interval : A ∩ B = Set.Ioo 2 3 := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l2908_290873


namespace NUMINAMATH_CALUDE_romanian_sequence_swaps_l2908_290860

/-- Represents a Romanian sequence -/
def RomanianSequence (n : ℕ) := { s : List Char // s.length = 3*n ∧ s.count 'I' = n ∧ s.count 'M' = n ∧ s.count 'O' = n }

/-- The minimum number of swaps required to transform one sequence into another -/
def minSwaps (s1 s2 : List Char) : ℕ := sorry

theorem romanian_sequence_swaps (n : ℕ) :
  ∀ (X : RomanianSequence n), ∃ (Y : RomanianSequence n), minSwaps X.val Y.val ≥ (3 * n^2) / 2 := by sorry

end NUMINAMATH_CALUDE_romanian_sequence_swaps_l2908_290860


namespace NUMINAMATH_CALUDE_cube_difference_152_l2908_290806

theorem cube_difference_152 : ∃! n : ℤ, 
  (∃ a : ℤ, a > 0 ∧ n - 76 = a^3) ∧ 
  (∃ b : ℤ, b > 0 ∧ n + 76 = b^3) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_difference_152_l2908_290806


namespace NUMINAMATH_CALUDE_inverse_proportion_example_l2908_290875

/-- Represents an inverse proportional relationship between two variables -/
def InverseProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function f(x) = 3/x is inversely proportional -/
theorem inverse_proportion_example : InverseProportion (fun x => 3 / x) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_example_l2908_290875


namespace NUMINAMATH_CALUDE_sound_speed_model_fits_data_sound_speed_model_unique_l2908_290822

/-- Represents the relationship between temperature and sound speed -/
def sound_speed_model (x : ℝ) : ℝ := 330 + 0.6 * x

/-- The set of data points for temperature and sound speed -/
def data_points : List (ℝ × ℝ) := [
  (-20, 318), (-10, 324), (0, 330), (10, 336), (20, 342), (30, 348)
]

/-- Theorem stating that the sound_speed_model fits the given data points -/
theorem sound_speed_model_fits_data : 
  ∀ (point : ℝ × ℝ), point ∈ data_points → 
    sound_speed_model point.1 = point.2 := by
  sorry

/-- Theorem stating that the sound_speed_model is the unique linear model fitting the data -/
theorem sound_speed_model_unique : 
  ∀ (a b : ℝ), (∀ (point : ℝ × ℝ), point ∈ data_points → 
    a + b * point.1 = point.2) → a = 330 ∧ b = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_sound_speed_model_fits_data_sound_speed_model_unique_l2908_290822


namespace NUMINAMATH_CALUDE_circle_circumference_with_inscribed_rectangle_l2908_290805

theorem circle_circumference_with_inscribed_rectangle :
  let rectangle_width : ℝ := 9
  let rectangle_height : ℝ := 12
  let diagonal : ℝ := (rectangle_width^2 + rectangle_height^2).sqrt
  let diameter : ℝ := diagonal
  let circumference : ℝ := π * diameter
  circumference = 15 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_circumference_with_inscribed_rectangle_l2908_290805


namespace NUMINAMATH_CALUDE_train_speed_time_reduction_l2908_290855

theorem train_speed_time_reduction :
  ∀ (v S : ℝ),
  v > 0 → S > 0 →
  let original_time := S / v
  let new_speed := 1.25 * v
  let new_time := S / new_speed
  (original_time - new_time) / original_time = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_time_reduction_l2908_290855


namespace NUMINAMATH_CALUDE_gcf_of_48_180_120_l2908_290820

theorem gcf_of_48_180_120 : Nat.gcd 48 (Nat.gcd 180 120) = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_48_180_120_l2908_290820


namespace NUMINAMATH_CALUDE_solve_for_a_l2908_290810

theorem solve_for_a : ∃ a : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = -3 → a * x - y = 1) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2908_290810


namespace NUMINAMATH_CALUDE_student_ticket_cost_l2908_290870

/-- Proves that the cost of each student ticket is $6 given the conditions of the problem -/
theorem student_ticket_cost (adult_ticket_cost : ℕ) (num_students : ℕ) (num_adults : ℕ) (total_revenue : ℕ) :
  adult_ticket_cost = 8 →
  num_students = 20 →
  num_adults = 12 →
  total_revenue = 216 →
  ∃ (student_ticket_cost : ℕ), 
    student_ticket_cost * num_students + adult_ticket_cost * num_adults = total_revenue ∧
    student_ticket_cost = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_student_ticket_cost_l2908_290870


namespace NUMINAMATH_CALUDE_odd_digits_365_base5_l2908_290843

/-- Counts the number of odd digits in the base-5 representation of a natural number -/
def countOddDigitsBase5 (n : ℕ) : ℕ :=
  sorry

theorem odd_digits_365_base5 : countOddDigitsBase5 365 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_365_base5_l2908_290843


namespace NUMINAMATH_CALUDE_unpainted_squares_count_l2908_290879

/-- Calculates the number of unpainted squares in a grid strip with a repeating pattern -/
def unpainted_squares (width : ℕ) (length : ℕ) (pattern_width : ℕ) 
  (unpainted_per_pattern : ℕ) (unpainted_remainder : ℕ) : ℕ :=
  let complete_patterns := length / pattern_width
  let remainder_columns := length % pattern_width
  complete_patterns * unpainted_per_pattern + unpainted_remainder

/-- The number of unpainted squares in a 5x250 grid with the given pattern is 812 -/
theorem unpainted_squares_count :
  unpainted_squares 5 250 4 13 6 = 812 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_squares_count_l2908_290879


namespace NUMINAMATH_CALUDE_waiter_customers_l2908_290842

theorem waiter_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) 
  (h1 : num_tables = 7)
  (h2 : women_per_table = 7)
  (h3 : men_per_table = 2) :
  num_tables * (women_per_table + men_per_table) = 63 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l2908_290842


namespace NUMINAMATH_CALUDE_sum_of_all_expressions_l2908_290877

/-- Represents an expression formed by replacing * with + or - in 1 * 2 * 3 * 4 * 5 * 6 -/
def Expression := List (Bool × ℕ)

/-- Generates all possible expressions -/
def generateExpressions : List Expression :=
  sorry

/-- Evaluates a single expression -/
def evaluateExpression (expr : Expression) : ℤ :=
  sorry

/-- Sums the results of all expressions -/
def sumAllExpressions : ℤ :=
  (generateExpressions.map evaluateExpression).sum

theorem sum_of_all_expressions :
  sumAllExpressions = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_all_expressions_l2908_290877


namespace NUMINAMATH_CALUDE_road_width_calculation_l2908_290845

/-- Represents the width of the roads in meters -/
def road_width : ℝ := 10

/-- The length of the lawn in meters -/
def lawn_length : ℝ := 80

/-- The breadth of the lawn in meters -/
def lawn_breadth : ℝ := 60

/-- The cost per square meter in Rupees -/
def cost_per_sq_m : ℝ := 5

/-- The total cost of traveling the two roads in Rupees -/
def total_cost : ℝ := 6500

theorem road_width_calculation :
  (lawn_length * road_width + lawn_breadth * road_width - road_width^2) * cost_per_sq_m = total_cost :=
sorry

end NUMINAMATH_CALUDE_road_width_calculation_l2908_290845


namespace NUMINAMATH_CALUDE_horner_v1_value_l2908_290825

def horner_polynomial (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

def horner_v1 (x : ℝ) : ℝ := x * 1 - 5

theorem horner_v1_value :
  horner_v1 (-2) = -7 :=
by sorry

end NUMINAMATH_CALUDE_horner_v1_value_l2908_290825


namespace NUMINAMATH_CALUDE_largest_valid_number_l2908_290836

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∃ (a b : ℕ), a ≠ b ∧ a < 10 ∧ b < 10 ∧
    n = 7000 + 100 * a + 20 + b ∧
    n % 30 = 0

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 7920 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l2908_290836


namespace NUMINAMATH_CALUDE_uncle_fyodor_wins_l2908_290823

/-- Represents the state of a sandwich (with or without sausage) -/
inductive SandwichState
  | WithSausage
  | WithoutSausage

/-- Represents a player in the game -/
inductive Player
  | UncleFyodor
  | Matroskin

/-- The game state -/
structure GameState where
  sandwiches : List SandwichState
  currentPlayer : Player
  fyodorMoves : Nat
  matroskinMoves : Nat

/-- A move in the game -/
inductive Move
  | EatSandwich : Move  -- For Uncle Fyodor
  | RemoveSausage : Nat → Move  -- For Matroskin, with sandwich index

/-- Function to apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Function to check if the game is over -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Function to determine the winner -/
def getWinner (state : GameState) : Option Player :=
  sorry

/-- Theorem stating that Uncle Fyodor can always win for N = 2^100 - 1 -/
theorem uncle_fyodor_wins :
  ∀ (initialState : GameState),
    initialState.sandwiches.length = 100 * (2^100 - 1) →
    initialState.currentPlayer = Player.UncleFyodor →
    initialState.fyodorMoves = 0 →
    initialState.matroskinMoves = 0 →
    ∀ (matroskinStrategy : GameState → Move),
      ∃ (fyodorStrategy : GameState → Move),
        let finalState := sorry  -- Play out the game using the strategies
        getWinner finalState = some Player.UncleFyodor :=
  sorry


end NUMINAMATH_CALUDE_uncle_fyodor_wins_l2908_290823


namespace NUMINAMATH_CALUDE_sector_area_l2908_290869

/-- Given an arc length of 4 cm corresponding to a central angle of 2 radians,
    the area of the sector enclosed by this central angle is 4 cm². -/
theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (h1 : arc_length = 4) (h2 : central_angle = 2) :
  let radius := arc_length / central_angle
  let sector_area := (1 / 2) * radius^2 * central_angle
  sector_area = 4 := by
sorry

end NUMINAMATH_CALUDE_sector_area_l2908_290869


namespace NUMINAMATH_CALUDE_lemonade_sold_l2908_290803

/-- Represents the number of cups of lemonade sold -/
def lemonade : ℕ := sorry

/-- Represents the number of cups of hot chocolate sold -/
def hotChocolate : ℕ := sorry

/-- The total number of cups sold -/
def totalCups : ℕ := 400

/-- The total money earned in yuan -/
def totalMoney : ℕ := 546

/-- The price of a cup of lemonade in yuan -/
def lemonadePrice : ℕ := 1

/-- The price of a cup of hot chocolate in yuan -/
def hotChocolatePrice : ℕ := 2

theorem lemonade_sold : 
  lemonade = 254 ∧ 
  lemonade + hotChocolate = totalCups ∧ 
  lemonade * lemonadePrice + hotChocolate * hotChocolatePrice = totalMoney :=
sorry

end NUMINAMATH_CALUDE_lemonade_sold_l2908_290803


namespace NUMINAMATH_CALUDE_total_pupils_l2908_290892

def number_of_girls : ℕ := 542
def number_of_boys : ℕ := 387

theorem total_pupils : number_of_girls + number_of_boys = 929 := by
  sorry

end NUMINAMATH_CALUDE_total_pupils_l2908_290892


namespace NUMINAMATH_CALUDE_first_tree_growth_rate_l2908_290886

/-- The daily growth rate of the first tree -/
def first_tree_growth : ℝ := 1

/-- The daily growth rate of the second tree -/
def second_tree_growth : ℝ := 2 * first_tree_growth

/-- The daily growth rate of the third tree -/
def third_tree_growth : ℝ := 2

/-- The daily growth rate of the fourth tree -/
def fourth_tree_growth : ℝ := 3

/-- The number of days the trees grew -/
def days : ℕ := 4

/-- The total growth of all trees -/
def total_growth : ℝ := 32

theorem first_tree_growth_rate :
  first_tree_growth * days +
  second_tree_growth * days +
  third_tree_growth * days +
  fourth_tree_growth * days = total_growth :=
by sorry

end NUMINAMATH_CALUDE_first_tree_growth_rate_l2908_290886


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l2908_290840

theorem simplify_sqrt_difference : (Real.sqrt 300 / Real.sqrt 75) - (Real.sqrt 128 / Real.sqrt 32) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l2908_290840


namespace NUMINAMATH_CALUDE_log_relation_l2908_290839

theorem log_relation (a b : ℝ) : 
  a = Real.log 343 / Real.log 6 → 
  b = Real.log 18 / Real.log 7 → 
  a = 6 / (b + 2 * Real.log 2 / Real.log 7) := by
sorry

end NUMINAMATH_CALUDE_log_relation_l2908_290839


namespace NUMINAMATH_CALUDE_condition_relationship_l2908_290865

theorem condition_relationship (x₁ x₂ : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 4 ∧ x₂ > 4 → x₁ + x₂ > 8 ∧ x₁ * x₂ > 16) ∧
  (∃ x₁ x₂ : ℝ, x₁ + x₂ > 8 ∧ x₁ * x₂ > 16 ∧ ¬(x₁ > 4 ∧ x₂ > 4)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l2908_290865


namespace NUMINAMATH_CALUDE_cubic_zeros_sum_less_than_two_l2908_290884

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

noncomputable def F (a b c x : ℝ) : ℝ := f a b c x - x * Real.exp (-x)

theorem cubic_zeros_sum_less_than_two (a b c : ℝ) (ha : a ≠ 0) 
    (h1 : 6 * a + b = 0) (h2 : f a b c 1 = 4 * a) :
    ∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ 
    0 ≤ x₁ ∧ x₃ ≤ 3 ∧
    F a b c x₁ = 0 ∧ F a b c x₂ = 0 ∧ F a b c x₃ = 0 ∧
    x₁ + x₂ + x₃ < 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_zeros_sum_less_than_two_l2908_290884


namespace NUMINAMATH_CALUDE_turner_amusement_park_tickets_l2908_290813

/-- Calculates the total number of tickets needed for a multi-day amusement park visit -/
def total_tickets (days : ℕ) 
                  (rollercoaster_rides_per_day : ℕ) 
                  (catapult_rides_per_day : ℕ) 
                  (ferris_wheel_rides_per_day : ℕ) 
                  (rollercoaster_tickets_per_ride : ℕ) 
                  (catapult_tickets_per_ride : ℕ) 
                  (ferris_wheel_tickets_per_ride : ℕ) : ℕ :=
  days * (rollercoaster_rides_per_day * rollercoaster_tickets_per_ride +
          catapult_rides_per_day * catapult_tickets_per_ride +
          ferris_wheel_rides_per_day * ferris_wheel_tickets_per_ride)

theorem turner_amusement_park_tickets : 
  total_tickets 3 3 2 1 4 4 1 = 63 := by
  sorry

end NUMINAMATH_CALUDE_turner_amusement_park_tickets_l2908_290813


namespace NUMINAMATH_CALUDE_cost_per_meat_type_l2908_290818

/-- Calculates the cost per type of sliced meat in a 4-pack with rush delivery --/
theorem cost_per_meat_type (base_cost : ℝ) (rush_delivery_rate : ℝ) (num_types : ℕ) :
  base_cost = 40 →
  rush_delivery_rate = 0.3 →
  num_types = 4 →
  (base_cost + base_cost * rush_delivery_rate) / num_types = 13 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_meat_type_l2908_290818


namespace NUMINAMATH_CALUDE_sum_products_sides_projections_equality_l2908_290827

/-- Represents a convex polygon in a 2D plane -/
structure ConvexPolygon where
  -- Add necessary fields here
  -- This is a placeholder structure

/-- Calculates the sum of products of side lengths and projected widths -/
def sumProductsSidesProjections (P Q : ConvexPolygon) : ℝ :=
  -- Placeholder definition
  0

/-- Theorem stating the equality of sumProductsSidesProjections for two polygons -/
theorem sum_products_sides_projections_equality (P Q : ConvexPolygon) :
  sumProductsSidesProjections P Q = sumProductsSidesProjections Q P :=
by
  sorry

#check sum_products_sides_projections_equality

end NUMINAMATH_CALUDE_sum_products_sides_projections_equality_l2908_290827


namespace NUMINAMATH_CALUDE_franks_initial_money_l2908_290808

/-- Frank's lamp purchase problem -/
theorem franks_initial_money (cheapest_lamp : ℕ) (expensive_multiplier : ℕ) (remaining_money : ℕ) : 
  cheapest_lamp = 20 →
  expensive_multiplier = 3 →
  remaining_money = 30 →
  cheapest_lamp * expensive_multiplier + remaining_money = 90 := by
  sorry

end NUMINAMATH_CALUDE_franks_initial_money_l2908_290808


namespace NUMINAMATH_CALUDE_bob_cleaning_time_l2908_290850

/-- Given that Alice takes 40 minutes to clean her room and Bob spends 3/8 of Alice's time,
    prove that Bob's cleaning time is 15 minutes. -/
theorem bob_cleaning_time (alice_time : ℕ) (bob_fraction : ℚ) :
  alice_time = 40 →
  bob_fraction = 3 / 8 →
  (bob_fraction * alice_time : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_bob_cleaning_time_l2908_290850


namespace NUMINAMATH_CALUDE_distance_from_displacements_l2908_290899

/-- The distance between two points given their net displacements -/
theorem distance_from_displacements (south west : ℝ) :
  south = 20 →
  west = 50 →
  Real.sqrt (south^2 + west^2) = 50 * Real.sqrt 2.9 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_displacements_l2908_290899


namespace NUMINAMATH_CALUDE_abc_product_l2908_290801

theorem abc_product (a b c : ℕ+) 
  (h1 : a * b = 13)
  (h2 : b * c = 52)
  (h3 : c * a = 4) :
  a * b * c = 52 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l2908_290801


namespace NUMINAMATH_CALUDE_subsets_with_sum_2008_l2908_290878

def set_63 : Finset ℕ := Finset.range 64 \ {0}

theorem subsets_with_sum_2008 : 
  (Finset.filter (fun S => S.sum id = 2008) (Finset.powerset set_63)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_subsets_with_sum_2008_l2908_290878


namespace NUMINAMATH_CALUDE_company_employees_l2908_290815

theorem company_employees (total : ℕ) 
  (h1 : (60 : ℚ) / 100 * total = (total : ℚ) - (40 : ℚ) / 100 * total)
  (h2 : (20 : ℚ) / 100 * total = (40 : ℚ) / 100 * total / 2)
  (h3 : (20 : ℚ) / 100 * total = 20) :
  total = 100 := by
sorry

end NUMINAMATH_CALUDE_company_employees_l2908_290815


namespace NUMINAMATH_CALUDE_pencil_cost_l2908_290891

/-- Given Mrs. Hilt's initial amount and the amount left after buying a pencil,
    prove that the cost of the pencil is the difference between these two amounts. -/
theorem pencil_cost (initial_amount amount_left : ℕ) 
    (h1 : initial_amount = 15)
    (h2 : amount_left = 4) :
    initial_amount - amount_left = 11 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l2908_290891


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_l2908_290846

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x

theorem tangent_line_at_zero : 
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    y = m * x + b ∧ 
    (∃ (h : ℝ), h ≠ 0 ∧ (f (0 + h) - f 0) / h = m) ∧
    f 0 = b ∧
    m = 1 ∧ b = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_l2908_290846


namespace NUMINAMATH_CALUDE_base7_135_equals_base10_75_l2908_290894

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- Theorem stating that 135 in base 7 is equal to 75 in base 10 --/
theorem base7_135_equals_base10_75 : base7ToBase10 1 3 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_base7_135_equals_base10_75_l2908_290894


namespace NUMINAMATH_CALUDE_reciprocal_of_lcm_l2908_290895

def a : ℕ := 24
def b : ℕ := 195

theorem reciprocal_of_lcm (a b : ℕ) : (1 : ℚ) / (Nat.lcm a b) = 1 / 1560 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_lcm_l2908_290895


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2908_290834

/-- Calculate the number of games in a round-robin tournament stage -/
def gamesInRoundRobin (n : ℕ) : ℕ := n * (n - 1)

/-- Calculate the number of games in a knockout tournament stage -/
def gamesInKnockout (n : ℕ) : ℕ := n - 1

/-- The total number of games in the chess tournament -/
def totalGames : ℕ :=
  gamesInRoundRobin 20 + gamesInRoundRobin 10 + gamesInKnockout 4

theorem chess_tournament_games :
  totalGames = 474 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2908_290834


namespace NUMINAMATH_CALUDE_simplify_expression_l2908_290802

theorem simplify_expression (s r : ℝ) : 
  (2 * s^2 + 4 * r - 5) - (s^2 + 6 * r - 8) = s^2 - 2 * r + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2908_290802


namespace NUMINAMATH_CALUDE_emily_sleep_duration_l2908_290854

/-- Calculates the time Emily slept during her flight -/
def time_emily_slept (flight_duration : ℕ) (num_episodes : ℕ) (episode_duration : ℕ) 
  (num_movies : ℕ) (movie_duration : ℕ) (remaining_time : ℕ) : ℚ :=
  let total_flight_minutes := flight_duration * 60
  let total_tv_minutes := num_episodes * episode_duration
  let total_movie_minutes := num_movies * movie_duration
  let sleep_minutes := total_flight_minutes - total_tv_minutes - total_movie_minutes - remaining_time
  (sleep_minutes : ℚ) / 60

/-- Theorem stating that Emily slept for 4.5 hours -/
theorem emily_sleep_duration :
  time_emily_slept 10 3 25 2 105 45 = 4.5 := by sorry

end NUMINAMATH_CALUDE_emily_sleep_duration_l2908_290854


namespace NUMINAMATH_CALUDE_smallest_next_divisor_l2908_290816

theorem smallest_next_divisor (m : ℕ) : 
  m % 2 = 0 ∧ 
  1000 ≤ m ∧ m < 10000 ∧ 
  m % 391 = 0 → 
  (∃ (d : ℕ), d ∣ m ∧ d > 391 ∧ d ≤ 782 ∧ ∀ (x : ℕ), x ∣ m ∧ x > 391 → x ≥ d) ∧
  782 ∣ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_l2908_290816


namespace NUMINAMATH_CALUDE_inequality_solution_l2908_290857

theorem inequality_solution (x : ℝ) : 
  x^3 - 3*x^2 - 4*x - 12 ≤ 0 ∧ 2*x + 6 > 0 → x ∈ Set.Icc (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2908_290857


namespace NUMINAMATH_CALUDE_problem_solution_l2908_290800

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -12)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 8) :
  b / (a + b) + c / (b + c) + a / (c + a) = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2908_290800


namespace NUMINAMATH_CALUDE_inequality_preserved_by_exponential_l2908_290872

theorem inequality_preserved_by_exponential (a b : ℝ) (h : a > b) :
  ∀ x : ℝ, a * (2 : ℝ)^x > b * (2 : ℝ)^x :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_preserved_by_exponential_l2908_290872


namespace NUMINAMATH_CALUDE_triangle_inequality_for_powers_l2908_290863

theorem triangle_inequality_for_powers (a b c : ℝ) :
  (∀ n : ℕ, a^n + b^n > c^n ∧ a^n + c^n > b^n ∧ b^n + c^n > a^n) ↔ 
  ((a = b ∧ a > c) ∨ (a = b ∧ b = c)) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_for_powers_l2908_290863


namespace NUMINAMATH_CALUDE_natural_number_puzzle_l2908_290871

def first_digit (n : ℕ) : ℕ := n.div (10 ^ (n.log 10))

def last_digit (n : ℕ) : ℕ := n % 10

def swap_first_last (n : ℕ) : ℕ :=
  let d := n.log 10
  last_digit n * 10^d + (n - first_digit n * 10^d - last_digit n) + first_digit n

theorem natural_number_puzzle (x : ℕ) :
  first_digit x = 2 →
  last_digit x = 5 →
  swap_first_last x = 2 * x + 2 →
  x ≤ 10000 →
  x = 25 ∨ x = 295 ∨ x = 2995 := by
  sorry

end NUMINAMATH_CALUDE_natural_number_puzzle_l2908_290871


namespace NUMINAMATH_CALUDE_solution_inequality_minimum_value_l2908_290807

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| - |x - 4|

-- Theorem for the solution of f(x) > 3
theorem solution_inequality (x : ℝ) : f x > 3 ↔ x > 2 := by sorry

-- Theorem for the minimum value of f(x)
theorem minimum_value : ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ m = 0 := by sorry

end NUMINAMATH_CALUDE_solution_inequality_minimum_value_l2908_290807


namespace NUMINAMATH_CALUDE_problem_statement_l2908_290847

/-- Given real numbers a and b satisfying the conditions, 
    prove the minimum value of m and the inequality for x, y, z -/
theorem problem_statement 
  (a b : ℝ) 
  (h1 : a * b > 0) 
  (h2 : a^2 * b = 2) 
  (m : ℝ := a * b + a^2) : 
  (∃ (t : ℝ), t = 3 ∧ ∀ m', m' = a * b + a^2 → m' ≥ t) ∧ 
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → |x + 2*y + 2*z| ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2908_290847


namespace NUMINAMATH_CALUDE_prob_event_A_is_three_eighths_l2908_290819

/-- Represents the faces of the tetrahedron -/
inductive Face : Type
  | zero : Face
  | one : Face
  | two : Face
  | three : Face

/-- Converts a Face to its numerical value -/
def faceValue : Face → ℕ
  | Face.zero => 0
  | Face.one => 1
  | Face.two => 2
  | Face.three => 3

/-- Defines the event A: m^2 + n^2 ≤ 4 -/
def eventA (m n : Face) : Prop :=
  (faceValue m)^2 + (faceValue n)^2 ≤ 4

/-- The probability of event A occurring -/
def probEventA : ℚ := 3/8

/-- Theorem stating that the probability of event A is 3/8 -/
theorem prob_event_A_is_three_eighths :
  probEventA = 3/8 := by sorry

end NUMINAMATH_CALUDE_prob_event_A_is_three_eighths_l2908_290819


namespace NUMINAMATH_CALUDE_box_surface_area_l2908_290867

/-- The surface area of a rectangular parallelepiped with dimensions a, b, c -/
def surfaceArea (a b c : ℕ) : ℕ := 2 * (a * b + b * c + c * a)

theorem box_surface_area :
  ∀ a b c : ℕ,
    0 < a ∧ a < 10 →
    0 < b ∧ b < 10 →
    0 < c ∧ c < 10 →
    a * b * c = 280 →
    surfaceArea a b c = 262 := by
  sorry

end NUMINAMATH_CALUDE_box_surface_area_l2908_290867


namespace NUMINAMATH_CALUDE_negation_of_all_is_some_not_l2908_290859

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (M : U → Prop)  -- M x means "x is a member of the math club"
variable (E : U → Prop)  -- E x means "x enjoys puzzles"

-- State the theorem
theorem negation_of_all_is_some_not :
  (¬ ∀ x, M x → E x) ↔ (∃ x, M x ∧ ¬ E x) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_is_some_not_l2908_290859


namespace NUMINAMATH_CALUDE_least_possible_y_l2908_290838

/-- Given that x is an even integer, y and z are odd integers,
    y - x > 5, and the least possible value of z - x is 9,
    prove that the least possible value of y is 7. -/
theorem least_possible_y (x y z : ℤ) 
  (h_x_even : Even x)
  (h_y_odd : Odd y)
  (h_z_odd : Odd z)
  (h_y_minus_x : y - x > 5)
  (h_z_minus_x_min : ∀ w, z - x ≤ w - x → w - x ≥ 9) :
  y ≥ 7 ∧ ∀ w, (Odd w ∧ w - x > 5) → y ≤ w := by
  sorry

end NUMINAMATH_CALUDE_least_possible_y_l2908_290838


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2908_290890

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 6) :
  Real.sqrt (x + 3) + Real.sqrt (y + 3) + Real.sqrt (z + 3) ≤ 3 * Real.sqrt 5 ∧
  ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 6 ∧
    Real.sqrt (a + 3) + Real.sqrt (b + 3) + Real.sqrt (c + 3) = 3 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2908_290890


namespace NUMINAMATH_CALUDE_volleyball_team_math_players_l2908_290835

/-- The number of players taking mathematics in a volleyball team -/
def players_taking_mathematics (total_players : ℕ) (physics_players : ℕ) (both_subjects : ℕ) : ℕ :=
  total_players - (physics_players - both_subjects)

/-- Theorem stating the number of players taking mathematics -/
theorem volleyball_team_math_players :
  let total_players : ℕ := 30
  let physics_players : ℕ := 15
  let both_subjects : ℕ := 6
  players_taking_mathematics total_players physics_players both_subjects = 21 := by
  sorry

#check volleyball_team_math_players

end NUMINAMATH_CALUDE_volleyball_team_math_players_l2908_290835


namespace NUMINAMATH_CALUDE_reciprocal_of_point_three_l2908_290883

theorem reciprocal_of_point_three (h : (0.3 : ℚ) = 3/10) : 
  (0.3 : ℚ)⁻¹ = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_point_three_l2908_290883


namespace NUMINAMATH_CALUDE_fraction_denominator_expression_l2908_290856

theorem fraction_denominator_expression 
  (x y a b : ℝ) 
  (h1 : x / y = 3) 
  (h2 : (2 * a - x) / (3 * b - y) = 3) 
  (h3 : a / b = 4.5) : 
  ∃ (E : ℝ), (2 * a - x) / E = 3 ∧ E = 3 * b - y := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_expression_l2908_290856


namespace NUMINAMATH_CALUDE_system_condition_l2908_290852

theorem system_condition : 
  (∀ x y : ℝ, x > 2 ∧ y > 3 → x + y > 5 ∧ x * y > 6) ∧ 
  (∃ x y : ℝ, x + y > 5 ∧ x * y > 6 ∧ ¬(x > 2 ∧ y > 3)) := by
sorry

end NUMINAMATH_CALUDE_system_condition_l2908_290852


namespace NUMINAMATH_CALUDE_common_point_l2908_290844

/-- A function of the form f(x) = x^2 + ax + b where a + b = 2021 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + (2021 - a)

/-- Theorem: All functions f(x) = x^2 + ax + b where a + b = 2021 have a common point at (1, 2022) -/
theorem common_point : ∀ a : ℝ, f a 1 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_common_point_l2908_290844


namespace NUMINAMATH_CALUDE_hexagon_sixth_angle_l2908_290882

/-- The sum of angles in a hexagon -/
def hexagon_angle_sum : ℝ := 720

/-- The given angles in the hexagon -/
def given_angles : List ℝ := [150, 110, 120, 130, 100]

/-- Theorem: In a hexagon where five angles are 150°, 110°, 120°, 130°, and 100°, 
    the measure of the sixth angle is 110°. -/
theorem hexagon_sixth_angle : 
  hexagon_angle_sum - (given_angles.sum) = 110 := by sorry

end NUMINAMATH_CALUDE_hexagon_sixth_angle_l2908_290882


namespace NUMINAMATH_CALUDE_veronica_brown_balls_l2908_290849

/-- Given that Veronica carried 27 yellow balls and 45% of the total balls were yellow,
    prove that she carried 33 brown balls. -/
theorem veronica_brown_balls :
  ∀ (total_balls : ℕ) (yellow_balls : ℕ) (brown_balls : ℕ),
    yellow_balls = 27 →
    (yellow_balls : ℚ) / (total_balls : ℚ) = 45 / 100 →
    total_balls = yellow_balls + brown_balls →
    brown_balls = 33 := by
  sorry

end NUMINAMATH_CALUDE_veronica_brown_balls_l2908_290849


namespace NUMINAMATH_CALUDE_consecutive_squares_equivalence_l2908_290898

theorem consecutive_squares_equivalence (n : ℤ) : 
  (∃ a : ℤ, n = a^2 + (a + 1)^2) ↔ (∃ b : ℤ, 2*n - 1 = b^2) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_squares_equivalence_l2908_290898


namespace NUMINAMATH_CALUDE_gumball_ratio_l2908_290841

/-- The number of red gumballs in the machine -/
def red_gumballs : ℕ := 16

/-- The total number of gumballs in the machine -/
def total_gumballs : ℕ := 56

/-- The number of blue gumballs in the machine -/
def blue_gumballs : ℕ := red_gumballs / 2

/-- The number of green gumballs in the machine -/
def green_gumballs : ℕ := total_gumballs - red_gumballs - blue_gumballs

/-- The ratio of green gumballs to blue gumballs is 4:1 -/
theorem gumball_ratio : 
  green_gumballs / blue_gumballs = 4 := by sorry

end NUMINAMATH_CALUDE_gumball_ratio_l2908_290841


namespace NUMINAMATH_CALUDE_smallest_b_for_quadratic_inequality_l2908_290864

theorem smallest_b_for_quadratic_inequality :
  ∀ b : ℝ, b^2 - 16*b + 55 ≥ 0 → b ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_quadratic_inequality_l2908_290864


namespace NUMINAMATH_CALUDE_line_x_eq_1_properties_l2908_290828

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-axis in the 2D plane -/
def x_axis : Line := { a := 0, b := 1, c := 0 }

/-- Check if a line passes through a point -/
def Line.passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The main theorem -/
theorem line_x_eq_1_properties :
  ∃ (l : Line),
    (∀ (x y : ℝ), l.passes_through (x, y) ↔ x = 1) ∧
    l.passes_through (1, 2) ∧
    l.perpendicular x_axis := by
  sorry

end NUMINAMATH_CALUDE_line_x_eq_1_properties_l2908_290828


namespace NUMINAMATH_CALUDE_cargo_weight_calculation_l2908_290830

/-- Calculates the total cargo weight after loading and unloading activities -/
def total_cargo_weight (initial_cargo : Real) (additional_cargo : Real) (unloaded_cargo : Real) 
  (short_ton_to_kg : Real) (pound_to_kg : Real) : Real :=
  (initial_cargo * short_ton_to_kg) + (additional_cargo * short_ton_to_kg) - (unloaded_cargo * pound_to_kg)

/-- Theorem stating the total cargo weight after loading and unloading activities -/
theorem cargo_weight_calculation :
  let initial_cargo : Real := 5973.42
  let additional_cargo : Real := 8723.18
  let unloaded_cargo : Real := 2256719.55
  let short_ton_to_kg : Real := 907.18474
  let pound_to_kg : Real := 0.45359237
  total_cargo_weight initial_cargo additional_cargo unloaded_cargo short_ton_to_kg pound_to_kg = 12302024.7688159 := by
  sorry


end NUMINAMATH_CALUDE_cargo_weight_calculation_l2908_290830


namespace NUMINAMATH_CALUDE_solution_set_f_geq_1_minus_x_sq_range_of_a_for_nonempty_solution_l2908_290821

-- Define the function f(x) = |x-1|
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part (1)
theorem solution_set_f_geq_1_minus_x_sq :
  {x : ℝ | f x ≥ 1 - x^2} = {x : ℝ | x ≤ 0 ∨ x ≥ 1} := by sorry

-- Theorem for part (2)
theorem range_of_a_for_nonempty_solution (a : ℝ) :
  (∃ x : ℝ, f x < a - x^2 + |x + 1|) ↔ a > -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_1_minus_x_sq_range_of_a_for_nonempty_solution_l2908_290821


namespace NUMINAMATH_CALUDE_triangle_max_side_length_l2908_290893

theorem triangle_max_side_length (D E F : Real) (side1 side2 : Real) :
  -- Triangle DEF exists
  0 < D ∧ 0 < E ∧ 0 < F ∧
  D + E + F = Real.pi ∧
  -- Given condition
  Real.cos (2 * D) + Real.cos (2 * E) + Real.cos (2 * F) = 1 ∧
  -- Two sides have lengths 8 and 15
  side1 = 8 ∧ side2 = 15 →
  -- The maximum length of the third side is 17
  ∃ side3 : Real, side3 ≤ 17 ∧
    ∀ x : Real, (∃ D' E' F' : Real,
      0 < D' ∧ 0 < E' ∧ 0 < F' ∧
      D' + E' + F' = Real.pi ∧
      Real.cos (2 * D') + Real.cos (2 * E') + Real.cos (2 * F') = 1 ∧
      x = ((side1^2 + side2^2 - 2 * side1 * side2 * Real.cos F')^(1/2))) →
    x ≤ 17 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_side_length_l2908_290893


namespace NUMINAMATH_CALUDE_regression_lines_intersect_l2908_290817

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point where a regression line passes through -/
def passingPoint (l : RegressionLine) (x : ℝ) : ℝ × ℝ :=
  (x, l.slope * x + l.intercept)

/-- Theorem: Two regression lines with the same average x and y values intersect at (a, b) -/
theorem regression_lines_intersect (l₁ l₂ : RegressionLine) (a b : ℝ) 
  (h₁ : passingPoint l₁ a = (a, b))
  (h₂ : passingPoint l₂ a = (a, b)) :
  ∃ (x y : ℝ), passingPoint l₁ x = (x, y) ∧ passingPoint l₂ x = (x, y) ∧ x = a ∧ y = b :=
sorry

end NUMINAMATH_CALUDE_regression_lines_intersect_l2908_290817


namespace NUMINAMATH_CALUDE_bowl_water_problem_l2908_290804

theorem bowl_water_problem (C : ℝ) (h1 : C > 0) : 
  C / 2 + 4 = 0.7 * C → 0.7 * C = 14 := by
  sorry

end NUMINAMATH_CALUDE_bowl_water_problem_l2908_290804


namespace NUMINAMATH_CALUDE_absent_students_percentage_l2908_290829

theorem absent_students_percentage
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (boys_absent_fraction : ℚ)
  (girls_absent_fraction : ℚ)
  (h1 : total_students = 240)
  (h2 : boys = 150)
  (h3 : girls = 90)
  (h4 : boys_absent_fraction = 1 / 5)
  (h5 : girls_absent_fraction = 1 / 2)
  (h6 : total_students = boys + girls) :
  (boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students = 5 / 16 :=
by sorry

end NUMINAMATH_CALUDE_absent_students_percentage_l2908_290829


namespace NUMINAMATH_CALUDE_complex_number_simplification_l2908_290848

theorem complex_number_simplification (i : ℂ) (h : i^2 = -1) :
  (2 + i^3) / (1 - i) = (3 + i) / 2 := by sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l2908_290848


namespace NUMINAMATH_CALUDE_smallest_multiple_thirty_two_satisfies_smallest_satisfying_integer_l2908_290809

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 900 * x % 640 = 0 → x ≥ 32 := by
  sorry

theorem thirty_two_satisfies : 900 * 32 % 640 = 0 := by
  sorry

theorem smallest_satisfying_integer : ∃! x : ℕ, x > 0 ∧ 900 * x % 640 = 0 ∧ ∀ y : ℕ, (y > 0 ∧ 900 * y % 640 = 0 → y ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_thirty_two_satisfies_smallest_satisfying_integer_l2908_290809


namespace NUMINAMATH_CALUDE_function_composition_property_l2908_290881

theorem function_composition_property (n : ℕ) :
  (∃ (f g : Fin n → Fin n), ∀ i : Fin n, 
    (f (g i) = i ∧ g (f i) ≠ i) ∨ (g (f i) = i ∧ f (g i) ≠ i)) ↔ 
  Even n :=
by sorry

end NUMINAMATH_CALUDE_function_composition_property_l2908_290881


namespace NUMINAMATH_CALUDE_rectangle_area_l2908_290826

/-- The area of a rectangle with width 7 meters and length 2 meters longer than the width is 63 square meters. -/
theorem rectangle_area : ℝ → ℝ → ℝ → Prop :=
  fun width length area =>
    width = 7 ∧ length = width + 2 → area = width * length → area = 63

/-- Proof of the theorem -/
lemma rectangle_area_proof : rectangle_area 7 9 63 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2908_290826


namespace NUMINAMATH_CALUDE_B_equals_C_equals_A_union_complement_B_l2908_290811

-- Define the sets A, B, C, and U
def A : Set ℝ := {x | x^2 ≥ 9}
def B : Set ℝ := {x | (x - 7) / (x + 1) ≤ 0}
def C : Set ℝ := {x | |x - 2| < 4}
def U : Set ℝ := Set.univ

-- Theorem statements
theorem B_equals : B = {x | -1 < x ∧ x ≤ 7} := by sorry

theorem C_equals : C = {x | -2 < x ∧ x < 6} := by sorry

theorem A_union_complement_B :
  A ∪ (U \ B) = {x | x ≥ 3 ∨ x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_B_equals_C_equals_A_union_complement_B_l2908_290811


namespace NUMINAMATH_CALUDE_q_div_p_equals_168_l2908_290868

/-- The number of slips in the hat -/
def total_slips : ℕ := 60

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 15

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The number of slips with each number -/
def slips_per_number : ℕ := 4

/-- The probability that all drawn slips bear the same number -/
def p : ℚ := (distinct_numbers : ℚ) / Nat.choose total_slips drawn_slips

/-- The probability that three slips bear one number and two bear a different number -/
def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 3 * Nat.choose slips_per_number 2 : ℚ) / Nat.choose total_slips drawn_slips

/-- The main theorem stating the ratio of q to p -/
theorem q_div_p_equals_168 : q / p = 168 := by sorry

end NUMINAMATH_CALUDE_q_div_p_equals_168_l2908_290868


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_21_19_l2908_290814

theorem half_abs_diff_squares_21_19 : (1/2 : ℝ) * |21^2 - 19^2| = 40 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_21_19_l2908_290814
