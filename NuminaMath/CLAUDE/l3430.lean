import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l3430_343091

theorem problem_solution (x y : ℝ) (h : 3 * x - 4 * y = 5) :
  (y = (3 * x - 5) / 4) ∧
  (y ≤ x → x ≥ -5) ∧
  (∀ a : ℝ, x + 2 * y = a ∧ x > 2 * y → a < 10) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3430_343091


namespace NUMINAMATH_CALUDE_quotient_sum_and_difference_l3430_343077

theorem quotient_sum_and_difference (a b : ℝ) (h : a / b = -1) : 
  (a + b = 0) ∧ (|a - b| = 2 * |b|) := by
  sorry

end NUMINAMATH_CALUDE_quotient_sum_and_difference_l3430_343077


namespace NUMINAMATH_CALUDE_johny_east_south_difference_l3430_343059

/-- Represents Johny's travel distances in different directions -/
structure TravelDistances where
  south : ℝ
  east : ℝ
  north : ℝ

/-- Johny's travel conditions -/
def johny_travel : TravelDistances → Prop :=
  λ d => d.south = 40 ∧
         d.east > d.south ∧
         d.north = 2 * d.east ∧
         d.south + d.east + d.north = 220

/-- The theorem to prove -/
theorem johny_east_south_difference (d : TravelDistances) 
  (h : johny_travel d) : d.east - d.south = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_johny_east_south_difference_l3430_343059


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3430_343000

theorem quadratic_inequality_range (a b c : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 3 → -1 < a * x^2 + b * x + c ∧ a * x^2 + b * x + c < 1) ∧
  (∀ x, x ∉ Set.Ioo (-1 : ℝ) 3 → a * x^2 + b * x + c ≤ -1 ∨ a * x^2 + b * x + c ≥ 1) →
  -1/2 < a ∧ a < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3430_343000


namespace NUMINAMATH_CALUDE_f_lower_bound_l3430_343004

noncomputable section

def f (x t : ℝ) : ℝ := ((x + t) / (x - 1)) * Real.exp (x - 1)

theorem f_lower_bound (x t : ℝ) (hx : x > 1) (ht : t > -1) :
  f x t > Real.sqrt x * (1 + (1/2) * Real.log x) := by
  sorry

end NUMINAMATH_CALUDE_f_lower_bound_l3430_343004


namespace NUMINAMATH_CALUDE_sum_of_consecutive_odds_l3430_343021

def is_valid_sum (n : ℕ) : Prop :=
  ∃ (k : ℤ), 
    (4 * k + 12 = n) ∧ 
    (k % 2 = 1) ∧ 
    ((4 * k + 4) % 10 = 0)

theorem sum_of_consecutive_odds : 
  is_valid_sum 28 ∧ 
  is_valid_sum 52 ∧ 
  is_valid_sum 84 ∧ 
  is_valid_sum 220 ∧ 
  ¬(is_valid_sum 112) :=
sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_odds_l3430_343021


namespace NUMINAMATH_CALUDE_car_wash_earnings_ratio_l3430_343093

theorem car_wash_earnings_ratio (total : ℕ) (lisa tommy : ℕ) : 
  total = 60 →
  lisa = total / 2 →
  lisa = tommy + 15 →
  Nat.gcd tommy lisa = tommy →
  (tommy : ℚ) / lisa = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_car_wash_earnings_ratio_l3430_343093


namespace NUMINAMATH_CALUDE_intersection_range_l3430_343012

/-- The function f(x) = 2x³ - 3x² + 1 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

/-- The theorem stating that if 2x³ - 3x² + (1 + b) = 0 has three distinct real roots, then -1 < b < 0 -/
theorem intersection_range (b : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f x = -b ∧ f y = -b ∧ f z = -b) →
  -1 < b ∧ b < 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l3430_343012


namespace NUMINAMATH_CALUDE_division_with_remainder_l3430_343041

theorem division_with_remainder : ∃ (q r : ℤ), 1234567 = 145 * q + r ∧ 0 ≤ r ∧ r < 145 ∧ r = 67 := by
  sorry

end NUMINAMATH_CALUDE_division_with_remainder_l3430_343041


namespace NUMINAMATH_CALUDE_pattern_D_cannot_fold_into_cube_only_pattern_D_cannot_fold_into_cube_l3430_343005

-- Define a type for the patterns
inductive Pattern : Type
  | A : Pattern
  | B : Pattern
  | C : Pattern
  | D : Pattern

-- Define a predicate to check if a pattern can be folded into a cube
def can_fold_into_cube (p : Pattern) : Prop :=
  match p with
  | Pattern.A => true
  | Pattern.B => true
  | Pattern.C => true
  | Pattern.D => false

-- Theorem stating that Pattern D cannot be folded into a cube
theorem pattern_D_cannot_fold_into_cube :
  ¬(can_fold_into_cube Pattern.D) :=
by sorry

-- Theorem stating that Pattern D is the only pattern that cannot be folded into a cube
theorem only_pattern_D_cannot_fold_into_cube :
  ∀ (p : Pattern), ¬(can_fold_into_cube p) ↔ p = Pattern.D :=
by sorry

end NUMINAMATH_CALUDE_pattern_D_cannot_fold_into_cube_only_pattern_D_cannot_fold_into_cube_l3430_343005


namespace NUMINAMATH_CALUDE_second_markdown_percentage_l3430_343019

theorem second_markdown_percentage 
  (original_price : ℝ) 
  (first_markdown_percentage : ℝ) 
  (second_markdown_percentage : ℝ) 
  (h1 : first_markdown_percentage = 10)
  (h2 : (1 - first_markdown_percentage / 100) * (1 - second_markdown_percentage / 100) * original_price = 0.81 * original_price) :
  second_markdown_percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_second_markdown_percentage_l3430_343019


namespace NUMINAMATH_CALUDE_congruence_problem_l3430_343064

theorem congruence_problem (x : ℤ) : (3 * x + 7) % 16 = 2 → (2 * x + 11) % 16 = 13 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3430_343064


namespace NUMINAMATH_CALUDE_fox_initial_coins_l3430_343038

/-- The number of times Fox crosses the bridge -/
def num_crossings : ℕ := 3

/-- The toll Fox pays after each crossing -/
def toll : ℚ := 50

/-- The final amount Fox wants to have -/
def final_amount : ℚ := 50

/-- The factor by which Fox's money is multiplied each crossing -/
def multiplier : ℚ := 3

theorem fox_initial_coins (x : ℚ) :
  (((x * multiplier - toll) * multiplier - toll) * multiplier - toll = final_amount) →
  (x = 700 / 27) :=
by sorry

end NUMINAMATH_CALUDE_fox_initial_coins_l3430_343038


namespace NUMINAMATH_CALUDE_sunday_school_average_class_size_l3430_343086

/-- The average class size in a Sunday school with two classes -/
theorem sunday_school_average_class_size 
  (three_year_olds : ℕ) 
  (four_year_olds : ℕ) 
  (five_year_olds : ℕ) 
  (six_year_olds : ℕ) 
  (h1 : three_year_olds = 13)
  (h2 : four_year_olds = 20)
  (h3 : five_year_olds = 15)
  (h4 : six_year_olds = 22) :
  (three_year_olds + four_year_olds + five_year_olds + six_year_olds) / 2 = 35 := by
  sorry

#check sunday_school_average_class_size

end NUMINAMATH_CALUDE_sunday_school_average_class_size_l3430_343086


namespace NUMINAMATH_CALUDE_larger_number_problem_l3430_343027

theorem larger_number_problem (x y : ℝ) : 
  x - y = 5 → x + y = 37 → max x y = 21 := by sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3430_343027


namespace NUMINAMATH_CALUDE_shiny_igneous_fraction_l3430_343024

/-- Represents Cliff's rock collection -/
structure RockCollection where
  total : ℕ
  sedimentary : ℕ
  igneous : ℕ
  shinyIgneous : ℕ
  shinySedimentary : ℕ

/-- Properties of Cliff's rock collection -/
def isValidCollection (c : RockCollection) : Prop :=
  c.igneous = c.sedimentary / 2 ∧
  c.shinySedimentary = c.sedimentary / 5 ∧
  c.shinyIgneous = 30 ∧
  c.total = 270 ∧
  c.total = c.sedimentary + c.igneous

theorem shiny_igneous_fraction (c : RockCollection) 
  (h : isValidCollection c) : 
  (c.shinyIgneous : ℚ) / c.igneous = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shiny_igneous_fraction_l3430_343024


namespace NUMINAMATH_CALUDE_train_length_l3430_343063

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (cross_time : ℝ) (h1 : speed_kmh = 90) (h2 : cross_time = 9) :
  speed_kmh * (1000 / 3600) * cross_time = 225 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3430_343063


namespace NUMINAMATH_CALUDE_subtract_from_twenty_l3430_343066

theorem subtract_from_twenty (x : ℤ) (h : x + 40 = 52) : 20 - x = 8 := by
  sorry

end NUMINAMATH_CALUDE_subtract_from_twenty_l3430_343066


namespace NUMINAMATH_CALUDE_point_P_coordinates_l3430_343094

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2

-- Theorem statement
theorem point_P_coordinates :
  ∃ (P : ℝ × ℝ), (f' P.1 = 3) ∧ ((P = (-1, -1)) ∨ (P = (1, 1))) :=
sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l3430_343094


namespace NUMINAMATH_CALUDE_max_value_of_e_l3430_343003

theorem max_value_of_e (a b c d e : ℝ) 
  (sum_condition : a + b + c + d + e = 8)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 + e^2 = 16) :
  e ≤ 16/5 ∧ ∃ a b c d, a + b + c + d + 16/5 = 8 ∧ a^2 + b^2 + c^2 + d^2 + (16/5)^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_e_l3430_343003


namespace NUMINAMATH_CALUDE_water_depth_when_upright_l3430_343074

/-- Represents a right cylindrical water tank -/
structure WaterTank where
  height : ℝ
  baseDiameter : ℝ

/-- Calculates the volume of water in the tank when horizontal -/
def horizontalWaterVolume (tank : WaterTank) (depth : ℝ) : ℝ :=
  sorry

/-- Calculates the depth of water when the tank is upright -/
def uprightWaterDepth (tank : WaterTank) (horizontalDepth : ℝ) : ℝ :=
  sorry

theorem water_depth_when_upright 
  (tank : WaterTank) 
  (h1 : tank.height = 20)
  (h2 : tank.baseDiameter = 6)
  (h3 : horizontalWaterVolume tank 4 = π * (tank.baseDiameter / 2)^2 * tank.height) :
  uprightWaterDepth tank 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_when_upright_l3430_343074


namespace NUMINAMATH_CALUDE_new_books_count_l3430_343097

def adventure_books : ℕ := 13
def mystery_books : ℕ := 17
def used_books : ℕ := 15

def total_books : ℕ := adventure_books + mystery_books

theorem new_books_count : total_books - used_books = 15 := by
  sorry

end NUMINAMATH_CALUDE_new_books_count_l3430_343097


namespace NUMINAMATH_CALUDE_traffic_light_probability_l3430_343022

/-- Represents the duration of traffic light phases in seconds -/
structure TrafficLightCycle where
  greenDuration : ℕ
  redDuration : ℕ

/-- Calculates the probability of waiting at least a given time in a traffic light cycle -/
def waitingProbability (cycle : TrafficLightCycle) (minWaitTime : ℕ) : ℚ :=
  let totalDuration := cycle.greenDuration + cycle.redDuration
  let waitInterval := cycle.redDuration - minWaitTime
  waitInterval / totalDuration

theorem traffic_light_probability (cycle : TrafficLightCycle) 
    (h1 : cycle.greenDuration = 40)
    (h2 : cycle.redDuration = 50) :
    waitingProbability cycle 20 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_probability_l3430_343022


namespace NUMINAMATH_CALUDE_zoo_visitors_l3430_343084

theorem zoo_visitors (friday_visitors : ℕ) (saturday_multiplier : ℕ) : 
  friday_visitors = 3575 →
  saturday_multiplier = 5 →
  friday_visitors * saturday_multiplier = 17875 := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_l3430_343084


namespace NUMINAMATH_CALUDE_red_squares_per_row_is_six_l3430_343047

/-- Represents a colored grid -/
structure ColoredGrid where
  rows : Nat
  columns : Nat
  redRows : Nat
  blueRows : Nat
  greenSquares : Nat

/-- Calculates the number of squares in each red row -/
def redSquaresPerRow (grid : ColoredGrid) : Nat :=
  let totalSquares := grid.rows * grid.columns
  let blueSquares := grid.blueRows * grid.columns
  let redSquares := totalSquares - blueSquares - grid.greenSquares
  redSquares / grid.redRows

/-- Theorem: In the given grid, there are 6 red squares in each red row -/
theorem red_squares_per_row_is_six (grid : ColoredGrid) 
  (h1 : grid.rows = 10)
  (h2 : grid.columns = 15)
  (h3 : grid.redRows = 4)
  (h4 : grid.blueRows = 4)
  (h5 : grid.greenSquares = 66) :
  redSquaresPerRow grid = 6 := by
  sorry

#eval redSquaresPerRow { rows := 10, columns := 15, redRows := 4, blueRows := 4, greenSquares := 66 }

end NUMINAMATH_CALUDE_red_squares_per_row_is_six_l3430_343047


namespace NUMINAMATH_CALUDE_model_car_velocities_l3430_343076

/-- A model car on a closed circuit -/
structure ModelCar where
  circuit_length : ℕ
  uphill_length : ℕ
  flat_length : ℕ
  downhill_length : ℕ
  vs : ℕ  -- uphill velocity
  vp : ℕ  -- flat velocity
  vd : ℕ  -- downhill velocity

/-- The conditions of the problem -/
def satisfies_conditions (car : ModelCar) : Prop :=
  car.circuit_length = 600 ∧
  car.uphill_length = car.downhill_length ∧
  car.uphill_length + car.flat_length + car.downhill_length = car.circuit_length ∧
  car.vs < car.vp ∧ car.vp < car.vd ∧
  (car.uphill_length / car.vs + car.flat_length / car.vp + car.downhill_length / car.vd : ℚ) = 50

/-- The theorem to prove -/
theorem model_car_velocities (car : ModelCar) :
  satisfies_conditions car →
  ((car.vs = 7 ∧ car.vp = 12 ∧ car.vd = 42) ∨
   (car.vs = 8 ∧ car.vp = 12 ∧ car.vd = 24) ∨
   (car.vs = 9 ∧ car.vp = 12 ∧ car.vd = 18) ∨
   (car.vs = 10 ∧ car.vp = 12 ∧ car.vd = 15)) :=
by sorry

end NUMINAMATH_CALUDE_model_car_velocities_l3430_343076


namespace NUMINAMATH_CALUDE_a_range_proof_l3430_343045

def is_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

theorem a_range_proof (f : ℝ → ℝ) (a : ℝ) 
  (h1 : is_decreasing f (-1) 1)
  (h2 : f (2*a - 1) < f (1 - a)) :
  2/3 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_a_range_proof_l3430_343045


namespace NUMINAMATH_CALUDE_tan_half_product_l3430_343039

theorem tan_half_product (a b : Real) :
  7 * (Real.sin a + Real.sin b) + 6 * (Real.cos a * Real.cos b - 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2) = 1) ∨ (Real.tan (a / 2) * Real.tan (b / 2) = -1) :=
by sorry

end NUMINAMATH_CALUDE_tan_half_product_l3430_343039


namespace NUMINAMATH_CALUDE_positive_integer_solution_is_perfect_square_l3430_343009

theorem positive_integer_solution_is_perfect_square (t : ℤ) (n : ℕ+) 
  (h : n^2 + (4*t - 1)*n + 4*t^2 = 0) : 
  ∃ (k : ℕ), n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solution_is_perfect_square_l3430_343009


namespace NUMINAMATH_CALUDE_expression_equals_sixteen_times_twelve_to_1001_l3430_343078

theorem expression_equals_sixteen_times_twelve_to_1001 :
  (3^1001 + 4^1002)^2 - (3^1001 - 4^1002)^2 = 16 * 12^1001 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_sixteen_times_twelve_to_1001_l3430_343078


namespace NUMINAMATH_CALUDE_frog_jump_probability_l3430_343034

/-- Represents a jump in 3D space -/
structure Jump where
  direction : Real × Real × Real
  length : Real

/-- Calculates the final position after a series of jumps -/
def finalPosition (jumps : List Jump) : Real × Real × Real :=
  sorry

/-- Calculates the distance between two points in 3D space -/
def distance (p1 p2 : Real × Real × Real) : Real :=
  sorry

/-- Calculates the probability of an event given a sample space -/
def probability (event : α → Prop) (sampleSpace : Set α) : Real :=
  sorry

theorem frog_jump_probability :
  let jumps := [
    { direction := sorry, length := 1 },
    { direction := sorry, length := 2 },
    { direction := sorry, length := 3 }
  ]
  let start := (0, 0, 0)
  let final := finalPosition jumps
  probability (λ jumps => distance start final ≤ 2) (sorry : Set (List Jump)) = 1/5 :=
sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l3430_343034


namespace NUMINAMATH_CALUDE_ticket_order_solution_l3430_343062

/-- Represents the ticket order information -/
structure TicketOrder where
  childPrice : ℚ
  adultPrice : ℚ
  discountThreshold : ℕ
  discountRate : ℚ
  childrenExcess : ℕ
  totalBill : ℚ

/-- Calculates the number of adult and children tickets -/
def calculateTickets (order : TicketOrder) : ℕ × ℕ :=
  sorry

/-- Checks if the discount was applied -/
def wasDiscountApplied (order : TicketOrder) (adultTickets childTickets : ℕ) : Bool :=
  sorry

theorem ticket_order_solution (order : TicketOrder)
    (h1 : order.childPrice = 7.5)
    (h2 : order.adultPrice = 12)
    (h3 : order.discountThreshold = 20)
    (h4 : order.discountRate = 0.1)
    (h5 : order.childrenExcess = 8)
    (h6 : order.totalBill = 138) :
    let (adultTickets, childTickets) := calculateTickets order
    adultTickets = 4 ∧ childTickets = 12 ∧ ¬wasDiscountApplied order adultTickets childTickets :=
  sorry

end NUMINAMATH_CALUDE_ticket_order_solution_l3430_343062


namespace NUMINAMATH_CALUDE_gcf_of_75_and_90_l3430_343020

theorem gcf_of_75_and_90 : Nat.gcd 75 90 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_90_l3430_343020


namespace NUMINAMATH_CALUDE_second_year_compound_interest_l3430_343073

/-- Represents the compound interest for a given year -/
def CompoundInterest (principal : ℝ) (rate : ℝ) (year : ℕ) : ℝ :=
  principal * (1 + rate) ^ year - principal

/-- Theorem stating that given a 5% interest rate and a third-year compound interest of $1260,
    the second-year compound interest is $1200 -/
theorem second_year_compound_interest
  (principal : ℝ)
  (h1 : CompoundInterest principal 0.05 3 = 1260)
  (h2 : principal > 0) :
  CompoundInterest principal 0.05 2 = 1200 := by
  sorry


end NUMINAMATH_CALUDE_second_year_compound_interest_l3430_343073


namespace NUMINAMATH_CALUDE_three_consecutive_heads_sequences_l3430_343096

def coin_flip_sequence (n : ℕ) : ℕ := 2^n

def no_three_consecutive_heads : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => no_three_consecutive_heads (n + 2) + no_three_consecutive_heads (n + 1) + no_three_consecutive_heads n

theorem three_consecutive_heads_sequences (n : ℕ) (h : n = 10) :
  coin_flip_sequence n - no_three_consecutive_heads n = 520 := by
  sorry

end NUMINAMATH_CALUDE_three_consecutive_heads_sequences_l3430_343096


namespace NUMINAMATH_CALUDE_red_shirt_pairs_l3430_343035

theorem red_shirt_pairs (total_students : ℕ) (blue_students : ℕ) (red_students : ℕ)
  (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  total_students = 144 →
  blue_students = 63 →
  red_students = 81 →
  total_pairs = 72 →
  blue_blue_pairs = 21 →
  total_students = blue_students + red_students →
  ∃ (red_red_pairs : ℕ), red_red_pairs = 30 ∧
    red_red_pairs + blue_blue_pairs + (blue_students - 2 * blue_blue_pairs) = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_red_shirt_pairs_l3430_343035


namespace NUMINAMATH_CALUDE_comparison_theorem_l3430_343095

theorem comparison_theorem (x : ℝ) (n : ℕ) (h1 : x > -1) (h2 : n ≥ 2) :
  (1 + x)^n ≥ 1 + n*x := by
  sorry

end NUMINAMATH_CALUDE_comparison_theorem_l3430_343095


namespace NUMINAMATH_CALUDE_right_triangle_sin_q_l3430_343056

/-- In a right triangle PQR with angle R = 90° and 3sin Q = 4cos Q, sin Q = 4/5 -/
theorem right_triangle_sin_q (Q : Real) (h1 : 3 * Real.sin Q = 4 * Real.cos Q) : Real.sin Q = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_q_l3430_343056


namespace NUMINAMATH_CALUDE_A_intersect_B_is_singleton_one_l3430_343042

def A : Set ℝ := {0.1, 1, 10}

def B : Set ℝ := { y | ∃ x ∈ A, y = Real.log x / Real.log 10 }

theorem A_intersect_B_is_singleton_one : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_is_singleton_one_l3430_343042


namespace NUMINAMATH_CALUDE_A_intersect_B_empty_l3430_343083

-- Define set A
def A : Set ℝ := {0, 1, 2}

-- Define set B
def B : Set ℝ := {x : ℝ | (x + 1) * (x + 2) < 0}

-- Theorem statement
theorem A_intersect_B_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_empty_l3430_343083


namespace NUMINAMATH_CALUDE_equilateral_triangle_division_l3430_343013

/-- Represents a point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  center : Point2D
  sideLength : ℝ

/-- Represents a division of an equilateral triangle -/
def TriangleDivision (t : EquilateralTriangle) (n : ℕ) :=
  { subdivisions : List EquilateralTriangle // 
    subdivisions.length = n * n ∧
    ∀ sub ∈ subdivisions, sub.sideLength = t.sideLength / n }

/-- Theorem: An equilateral triangle can be divided into 9 smaller congruent equilateral triangles -/
theorem equilateral_triangle_division (t : EquilateralTriangle) : 
  ∃ (div : TriangleDivision t 3), True := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_division_l3430_343013


namespace NUMINAMATH_CALUDE_exam_total_boys_l3430_343092

theorem exam_total_boys (average_all : ℚ) (average_passed : ℚ) (average_failed : ℚ) 
  (passed_count : ℕ) : 
  average_all = 40 ∧ average_passed = 39 ∧ average_failed = 15 ∧ passed_count = 125 → 
  ∃ (total_count : ℕ), total_count = 120 ∧ 
    average_all * total_count = average_passed * passed_count + 
      average_failed * (total_count - passed_count) :=
by sorry

end NUMINAMATH_CALUDE_exam_total_boys_l3430_343092


namespace NUMINAMATH_CALUDE_max_value_constraint_l3430_343026

theorem max_value_constraint (a b c : ℝ) (h : 9*a^2 + 4*b^2 + 25*c^2 = 1) : 
  ∃ (M : ℝ), M = 3.2 ∧ ∀ (x y z : ℝ), 9*x^2 + 4*y^2 + 25*z^2 = 1 → 6*x + 3*y + 10*z ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3430_343026


namespace NUMINAMATH_CALUDE_barn_painting_area_l3430_343061

theorem barn_painting_area (width length height : ℝ) 
  (h_width : width = 10)
  (h_length : length = 13)
  (h_height : height = 5) :
  2 * (width * height + length * height) + width * length = 590 :=
by sorry

end NUMINAMATH_CALUDE_barn_painting_area_l3430_343061


namespace NUMINAMATH_CALUDE_smallest_greater_than_1_1_l3430_343051

def given_set : Set ℚ := {1.4, 9/10, 1.2, 0.5, 13/10}

theorem smallest_greater_than_1_1 :
  ∃ x ∈ given_set, x > 1.1 ∧ ∀ y ∈ given_set, y > 1.1 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_greater_than_1_1_l3430_343051


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_72_l3430_343085

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

theorem five_digit_divisible_by_72 (a b : ℕ) : 
  a < 10 → b < 10 → 
  is_divisible_by (a * 10000 + 6790 + b) 72 → 
  a = 3 ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_72_l3430_343085


namespace NUMINAMATH_CALUDE_set_equality_l3430_343037

-- Define the set A
def A : Set ℝ := {x : ℝ | 2 * x^2 + x - 3 = 0}

-- Define the set B
def B : Set ℝ := {i : ℝ | i^2 ≥ 4}

-- Define the complement of set C in real numbers
def compl_C : Set ℝ := {-1, 1, 3/2}

-- Theorem statement
theorem set_equality : A ∩ B ∪ compl_C = {-1, 1, 3/2} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l3430_343037


namespace NUMINAMATH_CALUDE_no_x_axis_intersection_l3430_343036

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x + 2)^2 - 1

-- Theorem stating that the function does not intersect the x-axis
theorem no_x_axis_intersection :
  ∀ x : ℝ, f x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_x_axis_intersection_l3430_343036


namespace NUMINAMATH_CALUDE_crayon_cost_theorem_l3430_343068

/-- The number of crayons in a half dozen -/
def half_dozen : ℕ := 6

/-- The number of half dozens bought -/
def num_half_dozens : ℕ := 4

/-- The cost of one crayon in dollars -/
def cost_per_crayon : ℕ := 2

/-- The total number of crayons bought -/
def total_crayons : ℕ := num_half_dozens * half_dozen

/-- The total cost of the crayons in dollars -/
def total_cost : ℕ := total_crayons * cost_per_crayon

theorem crayon_cost_theorem : total_cost = 48 := by
  sorry

end NUMINAMATH_CALUDE_crayon_cost_theorem_l3430_343068


namespace NUMINAMATH_CALUDE_seven_minus_sqrt_five_floor_l3430_343090

-- Define the integer part function
noncomputable def integerPart (x : ℝ) : ℤ :=
  ⌊x⌋

-- State the theorem
theorem seven_minus_sqrt_five_floor : integerPart (7 - Real.sqrt 5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_seven_minus_sqrt_five_floor_l3430_343090


namespace NUMINAMATH_CALUDE_cubic_equation_value_l3430_343016

theorem cubic_equation_value (x : ℝ) (h : 3 * x^2 - x = 1) : 
  6 * x^3 + 7 * x^2 - 5 * x + 2008 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_value_l3430_343016


namespace NUMINAMATH_CALUDE_original_number_proof_l3430_343087

theorem original_number_proof (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 5 * (1 / x)) :
  x = Real.sqrt 2 / 20 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3430_343087


namespace NUMINAMATH_CALUDE_g_injective_on_restricted_domain_c_is_smallest_l3430_343071

/-- The function g(x) = (x+3)^2 - 6 -/
def g (x : ℝ) : ℝ := (x + 3)^2 - 6

/-- c is the lower bound of the restricted domain -/
def c : ℝ := -3

theorem g_injective_on_restricted_domain :
  ∀ x y, x ≥ c → y ≥ c → g x = g y → x = y :=
sorry

theorem c_is_smallest :
  ∀ c' < c, ∃ x y, x ≥ c' ∧ y ≥ c' ∧ x ≠ y ∧ g x = g y :=
sorry

end NUMINAMATH_CALUDE_g_injective_on_restricted_domain_c_is_smallest_l3430_343071


namespace NUMINAMATH_CALUDE_f_negative_one_equals_negative_twelve_l3430_343014

def f (x : ℝ) : ℝ := sorry

theorem f_negative_one_equals_negative_twelve
  (h_odd : ∀ x, f x = -f (-x))
  (h_nonneg : ∀ x ≥ 0, ∃ a : ℝ, f x = a^(x+1) - 4) :
  f (-1) = -12 := by sorry

end NUMINAMATH_CALUDE_f_negative_one_equals_negative_twelve_l3430_343014


namespace NUMINAMATH_CALUDE_non_zero_coeffs_bound_l3430_343079

/-- A polynomial is non-zero if it has at least one non-zero coefficient -/
def NonZeroPoly (p : Polynomial ℝ) : Prop :=
  ∃ (i : ℕ), p.coeff i ≠ 0

/-- The number of non-zero coefficients in a polynomial -/
def NumNonZeroCoeffs (p : Polynomial ℝ) : ℕ :=
  (p.support).card

/-- The statement to be proved -/
theorem non_zero_coeffs_bound (Q : Polynomial ℝ) (n : ℕ) 
  (hQ : NonZeroPoly Q) (hn : n > 0) : 
  NumNonZeroCoeffs ((X - 1)^n * Q) ≥ n + 1 :=
sorry

end NUMINAMATH_CALUDE_non_zero_coeffs_bound_l3430_343079


namespace NUMINAMATH_CALUDE_cannot_tile_8x9_with_6x1_l3430_343015

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a rectangular tile -/
structure Tile :=
  (length : ℕ)
  (width : ℕ)

/-- Defines what it means for a board to be tileable by a given tile -/
def is_tileable (b : Board) (t : Tile) : Prop :=
  ∃ (n : ℕ), n * (t.length * t.width) = b.rows * b.cols ∧
  (t.length ∣ b.rows ∨ t.length ∣ b.cols) ∧
  (t.width ∣ b.rows ∨ t.width ∣ b.cols)

/-- The main theorem stating that an 8x9 board cannot be tiled with 6x1 tiles -/
theorem cannot_tile_8x9_with_6x1 :
  ¬ is_tileable (Board.mk 8 9) (Tile.mk 6 1) :=
sorry

end NUMINAMATH_CALUDE_cannot_tile_8x9_with_6x1_l3430_343015


namespace NUMINAMATH_CALUDE_book_selection_probability_book_selection_proof_l3430_343008

theorem book_selection_probability : ℕ → ℝ
  | 12 => 55 / 209
  | _ => 0

theorem book_selection_proof (n : ℕ) :
  n = 12 →
  (book_selection_probability n) = 
    (Nat.choose n 3 * Nat.choose (n - 3) 2 * Nat.choose (n - 5) 2 : ℝ) / 
    ((Nat.choose n 5 : ℝ) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_book_selection_probability_book_selection_proof_l3430_343008


namespace NUMINAMATH_CALUDE_function_f_theorem_l3430_343001

/-- A function f: ℝ → ℝ satisfying the given conditions -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∃ (S : Finset ℝ), ∀ x ≠ 0, ∃ c ∈ S, f x = c * x) ∧
  (∀ x, f (x - 1 - f x) = f x - 1 - x)

/-- The theorem stating that f(x) = x or f(x) = -x -/
theorem function_f_theorem (f : ℝ → ℝ) (h : FunctionF f) :
  (∀ x, f x = x) ∨ (∀ x, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_function_f_theorem_l3430_343001


namespace NUMINAMATH_CALUDE_sine_cosine_product_l3430_343025

theorem sine_cosine_product (α : Real) : 
  (∃ P : ℝ × ℝ, P.1 = Real.cos α ∧ P.2 = Real.sin α ∧ P.2 = -2 * P.1) →
  Real.sin α * Real.cos α = -2/5 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_product_l3430_343025


namespace NUMINAMATH_CALUDE_terminating_decimal_of_7_over_200_l3430_343070

theorem terminating_decimal_of_7_over_200 : 
  ∃ (n : ℕ) (d : ℕ+), (7 : ℚ) / 200 = (n : ℚ) / d ∧ (n : ℚ) / d = 0.028 := by
  sorry

end NUMINAMATH_CALUDE_terminating_decimal_of_7_over_200_l3430_343070


namespace NUMINAMATH_CALUDE_linear_function_decreasing_l3430_343082

/-- A linear function y = (m-3)x + 6 + 2m decreases as x increases if and only if m < 3 -/
theorem linear_function_decreasing (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((m - 3) * x₁ + 6 + 2 * m) > ((m - 3) * x₂ + 6 + 2 * m)) ↔ m < 3 :=
sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_l3430_343082


namespace NUMINAMATH_CALUDE_winter_fest_attendance_l3430_343044

theorem winter_fest_attendance (total_students : ℕ) (attending_students : ℕ) 
  (girls : ℕ) (boys : ℕ) (h1 : total_students = 1400) 
  (h2 : attending_students = 800) (h3 : girls + boys = total_students) 
  (h4 : 3 * girls / 4 + 3 * boys / 5 = attending_students) : 
  3 * girls / 4 = 600 := by
sorry

end NUMINAMATH_CALUDE_winter_fest_attendance_l3430_343044


namespace NUMINAMATH_CALUDE_solve_equation_l3430_343099

theorem solve_equation (x : ℝ) (h : 5*x - 8 = 15*x + 14) : 6*(x + 3) = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3430_343099


namespace NUMINAMATH_CALUDE_odd_red_faces_count_l3430_343075

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with odd number of red faces -/
def count_odd_red_faces (dims : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating the correct number of cubes with odd red faces -/
theorem odd_red_faces_count (block : BlockDimensions) 
  (h1 : block.length = 5)
  (h2 : block.width = 5)
  (h3 : block.height = 1) : 
  count_odd_red_faces block = 13 := by
  sorry

end NUMINAMATH_CALUDE_odd_red_faces_count_l3430_343075


namespace NUMINAMATH_CALUDE_michael_tom_flying_robots_ratio_l3430_343080

theorem michael_tom_flying_robots_ratio : 
  ∀ (michael_robots tom_robots : ℕ), 
    michael_robots = 12 → 
    tom_robots = 3 → 
    (michael_robots : ℚ) / (tom_robots : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_michael_tom_flying_robots_ratio_l3430_343080


namespace NUMINAMATH_CALUDE_bees_second_day_l3430_343017

def bees_first_day : ℕ := 144
def multiplier : ℕ := 3

theorem bees_second_day : bees_first_day * multiplier = 432 := by
  sorry

end NUMINAMATH_CALUDE_bees_second_day_l3430_343017


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l3430_343065

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (5 : ℚ) / 8 ∧ 
  (∀ p' q' : ℕ+, (3 : ℚ) / 5 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (5 : ℚ) / 8 → q' ≥ q) →
  p + q = 21 :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l3430_343065


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3430_343030

theorem polynomial_divisibility (k n : ℕ) (P : Polynomial ℤ) : 
  Even k → 
  (∀ i : ℕ, i < k → Odd (P.coeff i)) → 
  P.degree = k → 
  (∃ Q : Polynomial ℤ, (X + 1)^n - 1 = P * Q) → 
  (k + 1) ∣ n :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3430_343030


namespace NUMINAMATH_CALUDE_domain_of_g_l3430_343011

-- Define the function f with domain (0,1)
def f : {x : ℝ | 0 < x ∧ x < 1} → ℝ := sorry

-- Define the function g(x) = f(2x-1)
def g (x : ℝ) : ℝ := f ⟨2*x - 1, sorry⟩

-- Theorem stating that the domain of g is (1/2, 1)
theorem domain_of_g : 
  ∀ x : ℝ, (∃ y, g x = y) ↔ (1/2 < x ∧ x < 1) := by sorry

end NUMINAMATH_CALUDE_domain_of_g_l3430_343011


namespace NUMINAMATH_CALUDE_no_real_solutions_l3430_343098

theorem no_real_solutions : ¬ ∃ x : ℝ, (x + 8)^2 = -|x| - 4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3430_343098


namespace NUMINAMATH_CALUDE_count_squares_on_marked_grid_l3430_343029

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- A square grid with marked points -/
structure MarkedGrid where
  size : ℕ
  points : List GridPoint

/-- A square formed by four points -/
structure Square where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint
  p4 : GridPoint

/-- Check if four points form a valid square -/
def isValidSquare (s : Square) : Bool :=
  sorry

/-- Count the number of valid squares that can be formed from a list of points -/
def countValidSquares (points : List GridPoint) : ℕ :=
  sorry

/-- The main theorem -/
theorem count_squares_on_marked_grid :
  ∀ (g : MarkedGrid),
    g.size = 4 ∧ 
    g.points.length = 12 ∧ 
    (∀ p ∈ g.points, p.x < 4 ∧ p.y < 4) ∧
    (∀ x y, x = 0 ∨ x = 3 ∨ y = 0 ∨ y = 3 → ¬∃ p ∈ g.points, p.x = x ∧ p.y = y) →
    countValidSquares g.points = 11 :=
  sorry

end NUMINAMATH_CALUDE_count_squares_on_marked_grid_l3430_343029


namespace NUMINAMATH_CALUDE_average_rate_of_change_average_rate_of_change_on_interval_l3430_343057

def f (x : ℝ) : ℝ := 2 * x + 1

theorem average_rate_of_change (a b : ℝ) (h : a < b) :
  (f b - f a) / (b - a) = 2 :=
by sorry

theorem average_rate_of_change_on_interval :
  (f 2 - f 1) / (2 - 1) = 2 :=
by sorry

end NUMINAMATH_CALUDE_average_rate_of_change_average_rate_of_change_on_interval_l3430_343057


namespace NUMINAMATH_CALUDE_derivative_of_cubic_composition_l3430_343069

/-- The derivative of y = f(a - bx) where f(x) = x^3 and a, b are real numbers -/
theorem derivative_of_cubic_composition (a b : ℝ) :
  deriv (fun x => (a - b*x)^3) = fun x => -3*b*(a - b*x)^2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_cubic_composition_l3430_343069


namespace NUMINAMATH_CALUDE_chocolate_theorem_l3430_343067

/-- Represents a square chocolate bar -/
structure ChocolateBar where
  side_length : ℕ
  piece_size : ℕ

/-- Calculates the number of pieces eaten along the sides of a square chocolate bar -/
def pieces_eaten (bar : ChocolateBar) : ℕ :=
  4 * (bar.side_length * 2 - 4)

/-- Theorem stating that for a 100cm square chocolate bar with 1cm pieces,
    the number of pieces eaten along the sides is 784 -/
theorem chocolate_theorem :
  let bar : ChocolateBar := { side_length := 100, piece_size := 1 }
  pieces_eaten bar = 784 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_theorem_l3430_343067


namespace NUMINAMATH_CALUDE_four_lines_theorem_l3430_343089

-- Define the type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define the type for points
def Point : Type := ℝ × ℝ

-- Define a function to check if a point is on a line
def PointOnLine (p : Point) (l : Line) : Prop := l p.1 p.2

-- Define a function to check if a point is on a circle
def PointOnCircle (p : Point) (c : Point → Prop) : Prop := c p

-- Define a function to get the intersection point of two lines
def Intersection (l1 l2 : Line) : Point := sorry

-- Define a function to get the circle passing through three points
def CircleThrough (p1 p2 p3 : Point) : Point → Prop := sorry

-- Define a function to get the point corresponding to a triple of lines
def CorrespondingPoint (l1 l2 l3 : Line) : Point := sorry

-- State the theorem
theorem four_lines_theorem 
  (l1 l2 l3 l4 : Line) 
  (p1 p2 p3 p4 : Point) 
  (c : Point → Prop) 
  (h1 : PointOnLine p1 l1) 
  (h2 : PointOnLine p2 l2) 
  (h3 : PointOnLine p3 l3) 
  (h4 : PointOnLine p4 l4) 
  (hc1 : PointOnCircle p1 c) 
  (hc2 : PointOnCircle p2 c) 
  (hc3 : PointOnCircle p3 c) 
  (hc4 : PointOnCircle p4 c) :
  ∃ (c' : Point → Prop), 
    PointOnCircle (CorrespondingPoint l2 l3 l4) c' ∧ 
    PointOnCircle (CorrespondingPoint l1 l3 l4) c' ∧ 
    PointOnCircle (CorrespondingPoint l1 l2 l4) c' ∧ 
    PointOnCircle (CorrespondingPoint l1 l2 l3) c' :=
sorry

end NUMINAMATH_CALUDE_four_lines_theorem_l3430_343089


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3430_343032

def repeating_decimal_23 : ℚ := 23 / 99
def repeating_decimal_056 : ℚ := 56 / 999
def repeating_decimal_004 : ℚ := 4 / 999

theorem sum_of_repeating_decimals :
  repeating_decimal_23 + repeating_decimal_056 + repeating_decimal_004 = 28917 / 98901 ∧
  (∀ n : ℕ, n > 1 → ¬(n ∣ 28917 ∧ n ∣ 98901)) := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3430_343032


namespace NUMINAMATH_CALUDE_largest_base7_to_base3_l3430_343007

/-- Converts a number from base 7 to base 10 -/
def base7ToDecimal (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7 + (n % 10)

/-- Converts a number from base 10 to base 3 -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
  aux n []

/-- The largest three-digit number in base 7 -/
def largestBase7 : Nat := 666

theorem largest_base7_to_base3 :
  decimalToBase3 (base7ToDecimal largestBase7) = [1, 1, 0, 2, 0, 0] := by
  sorry

#eval decimalToBase3 (base7ToDecimal largestBase7)

end NUMINAMATH_CALUDE_largest_base7_to_base3_l3430_343007


namespace NUMINAMATH_CALUDE_tom_last_year_games_l3430_343058

/-- Represents the number of hockey games Tom attended in various scenarios -/
structure HockeyGames where
  this_year : ℕ
  missed_this_year : ℕ
  total_two_years : ℕ

/-- Calculates the number of hockey games Tom attended last year -/
def games_last_year (g : HockeyGames) : ℕ :=
  g.total_two_years - g.this_year

/-- Theorem stating that Tom attended 9 hockey games last year -/
theorem tom_last_year_games (g : HockeyGames) 
  (h1 : g.this_year = 4)
  (h2 : g.missed_this_year = 7)
  (h3 : g.total_two_years = 13) :
  games_last_year g = 9 := by
  sorry


end NUMINAMATH_CALUDE_tom_last_year_games_l3430_343058


namespace NUMINAMATH_CALUDE_problem_solution_l3430_343054

theorem problem_solution : (42 / (9 - 3 * 2)) * 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3430_343054


namespace NUMINAMATH_CALUDE_largest_common_term_under_300_l3430_343006

-- Define the first arithmetic progression
def seq1 (n : ℕ) : ℕ := 3 * n + 1

-- Define the second arithmetic progression
def seq2 (n : ℕ) : ℕ := 10 * n + 2

-- Define a function to check if a number is in both sequences
def isCommonTerm (x : ℕ) : Prop :=
  ∃ n m : ℕ, seq1 n = x ∧ seq2 m = x

-- Theorem statement
theorem largest_common_term_under_300 :
  (∀ x : ℕ, x < 300 → isCommonTerm x → x ≤ 290) ∧
  isCommonTerm 290 := by sorry

end NUMINAMATH_CALUDE_largest_common_term_under_300_l3430_343006


namespace NUMINAMATH_CALUDE_solve_system_for_p_l3430_343028

theorem solve_system_for_p (p q : ℚ) 
  (eq1 : 3 * p + 4 * q = 15) 
  (eq2 : 4 * p + 3 * q = 18) : 
  p = 27 / 7 := by sorry

end NUMINAMATH_CALUDE_solve_system_for_p_l3430_343028


namespace NUMINAMATH_CALUDE_alien_energy_conversion_l3430_343049

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem alien_energy_conversion :
  base7ToBase10 [1, 2, 3] = 162 := by
  sorry

end NUMINAMATH_CALUDE_alien_energy_conversion_l3430_343049


namespace NUMINAMATH_CALUDE_campsite_tent_ratio_l3430_343052

/-- Represents the number of tents in different areas of a campsite -/
structure CampsiteTents where
  north : ℕ
  east : ℕ
  south : ℕ
  center : ℕ
  total : ℕ

/-- The ratio of center tents to north tents is 4:1 given the campsite conditions -/
theorem campsite_tent_ratio (c : CampsiteTents) 
  (h1 : c.total = 900)
  (h2 : c.north = 100)
  (h3 : c.east = 2 * c.north)
  (h4 : c.south = 200)
  (h5 : c.total = c.north + c.east + c.south + c.center) :
  c.center / c.north = 4 := by
  sorry

#check campsite_tent_ratio

end NUMINAMATH_CALUDE_campsite_tent_ratio_l3430_343052


namespace NUMINAMATH_CALUDE_area_of_polygon_ABHFGD_l3430_343072

-- Define the points
variable (A B C D E F G H : ℝ × ℝ)

-- Define the squares
def is_square (P Q R S : ℝ × ℝ) : Prop := sorry

-- Define the area of a polygon
def area (points : List (ℝ × ℝ)) : ℝ := sorry

-- Define the midpoint of a line segment
def is_midpoint (M P Q : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem area_of_polygon_ABHFGD :
  is_square A B C D →
  is_square E F G D →
  area [A, B, C, D] = 36 →
  area [E, F, G, D] = 36 →
  is_midpoint H B C →
  is_midpoint H E F →
  area [A, B, H, F, G, D] = 36 := sorry

end NUMINAMATH_CALUDE_area_of_polygon_ABHFGD_l3430_343072


namespace NUMINAMATH_CALUDE_seven_rings_four_fingers_l3430_343088

/-- The number of ways to arrange rings on fingers -/
def ring_arrangements (total_rings : ℕ) (fingers : ℕ) : ℕ :=
  Nat.choose total_rings fingers * 
  Nat.factorial fingers * 
  Nat.choose (total_rings - 1) (fingers - 1)

/-- Theorem stating the number of ring arrangements for 7 rings on 4 fingers -/
theorem seven_rings_four_fingers : 
  ring_arrangements 7 4 = 29400 := by
  sorry

end NUMINAMATH_CALUDE_seven_rings_four_fingers_l3430_343088


namespace NUMINAMATH_CALUDE_tan_plus_cot_l3430_343031

theorem tan_plus_cot (α : ℝ) (h : Real.sin (2 * α) = 3 / 4) :
  Real.tan α + (Real.tan α)⁻¹ = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_plus_cot_l3430_343031


namespace NUMINAMATH_CALUDE_ice_cream_cost_l3430_343046

/-- The cost of ice cream problem -/
theorem ice_cream_cost (ice_cream_quantity : ℕ) (yoghurt_quantity : ℕ) (yoghurt_cost : ℚ) (price_difference : ℚ) :
  ice_cream_quantity = 10 →
  yoghurt_quantity = 4 →
  yoghurt_cost = 1 →
  price_difference = 36 →
  ∃ (ice_cream_cost : ℚ),
    ice_cream_cost * ice_cream_quantity = yoghurt_cost * yoghurt_quantity + price_difference ∧
    ice_cream_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l3430_343046


namespace NUMINAMATH_CALUDE_task_completion_time_l3430_343050

/-- Proves that the total time to complete a task is 8 days given the specified conditions -/
theorem task_completion_time 
  (john_rate : ℚ) 
  (jane_rate : ℚ) 
  (jane_leave_before_end : ℕ) :
  john_rate = 1/16 →
  jane_rate = 1/12 →
  jane_leave_before_end = 5 →
  ∃ (total_days : ℕ), total_days = 8 ∧ 
    (john_rate + jane_rate) * (total_days - jane_leave_before_end : ℚ) + 
    john_rate * (jane_leave_before_end : ℚ) = 1 :=
by sorry

end NUMINAMATH_CALUDE_task_completion_time_l3430_343050


namespace NUMINAMATH_CALUDE_value_of_expression_l3430_343053

theorem value_of_expression : 6 * 2017 - 2017 * 4 = 4034 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3430_343053


namespace NUMINAMATH_CALUDE_sphere_volume_increase_on_doubling_radius_l3430_343043

theorem sphere_volume_increase_on_doubling_radius :
  ∀ (r : ℝ), r > 0 →
  (4 / 3 * Real.pi * (2 * r)^3) = 8 * (4 / 3 * Real.pi * r^3) :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_increase_on_doubling_radius_l3430_343043


namespace NUMINAMATH_CALUDE_apples_ratio_l3430_343033

def apples_problem (tuesday wednesday thursday : ℕ) : Prop :=
  tuesday = 4 ∧
  thursday = tuesday / 2 ∧
  tuesday + wednesday + thursday = 14

theorem apples_ratio : 
  ∀ tuesday wednesday thursday : ℕ,
  apples_problem tuesday wednesday thursday →
  wednesday = 2 * tuesday :=
by sorry

end NUMINAMATH_CALUDE_apples_ratio_l3430_343033


namespace NUMINAMATH_CALUDE_julia_parrot_weeks_l3430_343055

/-- Represents the problem of determining how long Julia has had her parrot -/
theorem julia_parrot_weeks : 
  ∀ (total_weekly_cost rabbit_weekly_cost total_spent rabbit_weeks : ℕ),
  total_weekly_cost = 30 →
  rabbit_weekly_cost = 12 →
  rabbit_weeks = 5 →
  total_spent = 114 →
  ∃ (parrot_weeks : ℕ),
    parrot_weeks * (total_weekly_cost - rabbit_weekly_cost) = 
      total_spent - (rabbit_weeks * rabbit_weekly_cost) ∧
    parrot_weeks = 3 :=
by sorry

end NUMINAMATH_CALUDE_julia_parrot_weeks_l3430_343055


namespace NUMINAMATH_CALUDE_friends_walking_problem_l3430_343081

/-- Two friends walking on a trail problem -/
theorem friends_walking_problem (v : ℝ) (h : v > 0) :
  let trail_length : ℝ := 22
  let speed_ratio : ℝ := 1.2
  let d : ℝ := trail_length / (1 + speed_ratio)
  trail_length - d = 12 := by sorry

end NUMINAMATH_CALUDE_friends_walking_problem_l3430_343081


namespace NUMINAMATH_CALUDE_food_bank_donation_ratio_l3430_343048

/-- Proves the ratio of food donations in the second week to the first week -/
theorem food_bank_donation_ratio :
  let first_week_donation : ℝ := 40
  let second_week_multiple : ℝ := x
  let total_donation : ℝ := first_week_donation + first_week_donation * second_week_multiple
  let remaining_percentage : ℝ := 0.3
  let remaining_food : ℝ := 36
  remaining_percentage * total_donation = remaining_food →
  second_week_multiple = 2 := by
  sorry

end NUMINAMATH_CALUDE_food_bank_donation_ratio_l3430_343048


namespace NUMINAMATH_CALUDE_sector_area_l3430_343002

theorem sector_area (θ : Real) (L : Real) (A : Real) : 
  θ = π / 6 → 
  L = 2 * π / 3 → 
  A = 4 * π / 3 → 
  ∃ (r : Real), 
    L = r * θ ∧ 
    A = 1 / 2 * r^2 * θ := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3430_343002


namespace NUMINAMATH_CALUDE_isabellaPaintableArea_l3430_343010

/-- Calculates the total paintable wall area in multiple bedrooms -/
def totalPaintableArea (
  numBedrooms : ℕ
  ) (length width height : ℝ
  ) (unpaintableArea : ℝ
  ) : ℝ := by
  sorry

/-- Theorem stating the total paintable wall area for the given conditions -/
theorem isabellaPaintableArea :
  totalPaintableArea 3 12 10 8 60 = 876 := by
  sorry

end NUMINAMATH_CALUDE_isabellaPaintableArea_l3430_343010


namespace NUMINAMATH_CALUDE_leahs_outfits_l3430_343018

/-- Calculate the number of possible outfits given the number of options for each clothing item -/
def number_of_outfits (trousers shirts jackets shoes : ℕ) : ℕ :=
  trousers * shirts * jackets * shoes

/-- Theorem: The number of outfits for Leah's wardrobe is 840 -/
theorem leahs_outfits :
  number_of_outfits 5 6 4 7 = 840 := by
  sorry

end NUMINAMATH_CALUDE_leahs_outfits_l3430_343018


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3430_343060

theorem largest_n_satisfying_inequality : 
  ∀ n : ℤ, (1/4 : ℚ) + (n : ℚ)/6 < 3/2 ↔ n ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3430_343060


namespace NUMINAMATH_CALUDE_largest_three_digit_congruence_l3430_343040

theorem largest_three_digit_congruence :
  ∃ (n : ℕ), n = 991 ∧ 
  n < 1000 ∧ 
  n > 99 ∧
  55 * n ≡ 165 [MOD 260] ∧
  ∀ (m : ℕ), m < 1000 ∧ m > 99 ∧ 55 * m ≡ 165 [MOD 260] → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_congruence_l3430_343040


namespace NUMINAMATH_CALUDE_fewer_blue_chairs_than_yellow_l3430_343023

/-- Represents the number of chairs of each color in Rodrigo's classroom -/
structure ClassroomChairs where
  red : ℕ
  yellow : ℕ
  blue : ℕ

def total_chairs (c : ClassroomChairs) : ℕ := c.red + c.yellow + c.blue

theorem fewer_blue_chairs_than_yellow (c : ClassroomChairs) 
  (h1 : c.red = 4)
  (h2 : c.yellow = 2 * c.red)
  (h3 : total_chairs c - 3 = 15) :
  c.yellow - c.blue = 2 := by
  sorry

end NUMINAMATH_CALUDE_fewer_blue_chairs_than_yellow_l3430_343023
