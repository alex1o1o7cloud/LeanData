import Mathlib

namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l1937_193761

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.7 : ℝ)⌉ = 31 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l1937_193761


namespace NUMINAMATH_CALUDE_four_digit_sum_l1937_193752

theorem four_digit_sum (a b c d : ℕ) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
  6 * (a + b + c + d) * 1111 = 73326 →
  ({a, b, c, d} : Finset ℕ) = {1, 2, 3, 5} :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_l1937_193752


namespace NUMINAMATH_CALUDE_machine_selling_price_l1937_193729

/-- Calculates the selling price of a machine given its costs and desired profit percentage -/
def selling_price (purchase_price repair_cost transport_cost profit_percent : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transport_cost
  let profit := total_cost * profit_percent / 100
  total_cost + profit

/-- Theorem stating that the selling price of the machine is 30000 Rs -/
theorem machine_selling_price :
  selling_price 14000 5000 1000 50 = 30000 := by
  sorry

end NUMINAMATH_CALUDE_machine_selling_price_l1937_193729


namespace NUMINAMATH_CALUDE_exists_number_not_divisible_by_both_l1937_193781

def numbers : List Nat := [3654, 3664, 3674, 3684, 3694]

def divisible_by_4 (n : Nat) : Prop := n % 4 = 0

def divisible_by_3 (n : Nat) : Prop := n % 3 = 0

def units_digit (n : Nat) : Nat := n % 10

def tens_digit (n : Nat) : Nat := (n / 10) % 10

theorem exists_number_not_divisible_by_both :
  ∃ n ∈ numbers, ¬(divisible_by_4 n ∧ divisible_by_3 n) ∧
  (units_digit n * tens_digit n = 28 ∨ units_digit n * tens_digit n = 36) :=
by sorry

end NUMINAMATH_CALUDE_exists_number_not_divisible_by_both_l1937_193781


namespace NUMINAMATH_CALUDE_shopping_cart_fruit_ratio_l1937_193758

theorem shopping_cart_fruit_ratio (apples oranges pears : ℕ) : 
  oranges = 3 * apples →
  pears = 4 * oranges →
  (apples : ℚ) / pears = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_shopping_cart_fruit_ratio_l1937_193758


namespace NUMINAMATH_CALUDE_range_of_s_l1937_193716

-- Define the set of composite positive integers
def CompositePositiveIntegers : Set ℕ := {n : ℕ | n > 1 ∧ ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b}

-- Define the function s
def s (n : ℕ) : ℕ := sorry

-- State the theorem
theorem range_of_s :
  (∀ n ∈ CompositePositiveIntegers, s n > 7) ∧
  (∀ k > 7, ∃ n ∈ CompositePositiveIntegers, s n = k) :=
sorry

end NUMINAMATH_CALUDE_range_of_s_l1937_193716


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l1937_193772

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a := by sorry

theorem sum_of_roots_specific_quadratic :
  let r₁ := (7 + Real.sqrt 1) / 2
  let r₂ := (7 - Real.sqrt 1) / 2
  r₁ + r₂ = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l1937_193772


namespace NUMINAMATH_CALUDE_subset_pair_existence_l1937_193759

theorem subset_pair_existence (n : ℕ) (A : Fin n → Set ℕ) :
  ∃ (X Y : ℕ), ∀ i : Fin n, (X ∈ A i ∧ Y ∈ A i) ∨ (X ∉ A i ∧ Y ∉ A i) := by
  sorry

end NUMINAMATH_CALUDE_subset_pair_existence_l1937_193759


namespace NUMINAMATH_CALUDE_fred_basketball_games_l1937_193720

/-- The number of basketball games Fred went to last year -/
def last_year_games : ℕ := 36

/-- The difference in games between last year and this year -/
def game_difference : ℕ := 11

/-- The number of basketball games Fred went to this year -/
def this_year_games : ℕ := last_year_games - game_difference

theorem fred_basketball_games : this_year_games = 25 := by
  sorry

end NUMINAMATH_CALUDE_fred_basketball_games_l1937_193720


namespace NUMINAMATH_CALUDE_job_completion_proof_l1937_193741

/-- The number of days initially planned for 6 workers to complete a job -/
def initial_days : ℕ := sorry

/-- The number of workers who started the job -/
def initial_workers : ℕ := 6

/-- The number of days worked before additional workers joined -/
def days_before_joining : ℕ := 3

/-- The number of additional workers who joined -/
def additional_workers : ℕ := 4

/-- The number of days worked after additional workers joined -/
def days_after_joining : ℕ := 3

/-- The total number of worker-days required to complete the job -/
def total_worker_days : ℕ := initial_workers * initial_days

theorem job_completion_proof :
  total_worker_days = 
    initial_workers * days_before_joining + 
    (initial_workers + additional_workers) * days_after_joining ∧
  initial_days = 8 := by sorry

end NUMINAMATH_CALUDE_job_completion_proof_l1937_193741


namespace NUMINAMATH_CALUDE_inequality_proof_l1937_193799

theorem inequality_proof (a b : ℝ) (n : ℤ) (ha : a > 0) (hb : b > 0) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1937_193799


namespace NUMINAMATH_CALUDE_area_two_sectors_l1937_193766

/-- The area of a figure composed of two 45° sectors of a circle with radius 10 -/
theorem area_two_sectors (r : ℝ) (h : r = 10) : 
  2 * (π * r^2 * (45 / 360)) = 25 * π := by
  sorry

end NUMINAMATH_CALUDE_area_two_sectors_l1937_193766


namespace NUMINAMATH_CALUDE_angelina_driving_equation_l1937_193701

/-- Represents the driving scenario of Angelina --/
structure DrivingScenario where
  initial_speed : ℝ
  rest_time : ℝ
  final_speed : ℝ
  total_distance : ℝ
  total_time : ℝ

/-- The equation for Angelina's driving time before rest --/
def driving_equation (s : DrivingScenario) (t : ℝ) : Prop :=
  s.initial_speed * t + s.final_speed * (s.total_time - s.rest_time / 60 - t) = s.total_distance

/-- Theorem stating that the given equation correctly represents Angelina's driving scenario --/
theorem angelina_driving_equation :
  ∃ (s : DrivingScenario),
    s.initial_speed = 60 ∧
    s.rest_time = 15 ∧
    s.final_speed = 90 ∧
    s.total_distance = 255 ∧
    s.total_time = 4 ∧
    ∀ (t : ℝ), driving_equation s t ↔ (60 * t + 90 * (15 / 4 - t) = 255) :=
  sorry

end NUMINAMATH_CALUDE_angelina_driving_equation_l1937_193701


namespace NUMINAMATH_CALUDE_work_completion_time_l1937_193789

theorem work_completion_time (a b : ℕ) (h1 : a = 20) (h2 : (4 : ℝ) * ((1 : ℝ) / a + (1 : ℝ) / b) = (1 : ℝ) / 3) : b = 30 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1937_193789


namespace NUMINAMATH_CALUDE_pipe_B_fill_time_l1937_193774

/-- Time for pipe A to fill the tank -/
def time_A : ℝ := 5

/-- Time for the tank to drain -/
def time_drain : ℝ := 20

/-- Time to fill the tank with both pipes on and drainage open -/
def time_combined : ℝ := 3.6363636363636362

/-- Time for pipe B to fill the tank -/
def time_B : ℝ := 1.0526315789473684

/-- Theorem stating the relationship between the given times -/
theorem pipe_B_fill_time :
  time_B = (time_A * time_drain * time_combined) / 
           (time_A * time_drain - time_A * time_combined - time_drain * time_combined) :=
by sorry

end NUMINAMATH_CALUDE_pipe_B_fill_time_l1937_193774


namespace NUMINAMATH_CALUDE_dan_total_limes_l1937_193739

/-- The number of limes Dan picked -/
def limes_picked : ℕ := 9

/-- The number of limes Sara gave to Dan -/
def limes_given : ℕ := 4

/-- The total number of limes Dan has now -/
def total_limes : ℕ := limes_picked + limes_given

theorem dan_total_limes : total_limes = 13 := by
  sorry

end NUMINAMATH_CALUDE_dan_total_limes_l1937_193739


namespace NUMINAMATH_CALUDE_exists_empty_selection_l1937_193700

/-- Represents a chessboard with pieces -/
structure Chessboard (n : ℕ) :=
  (board : Fin (2*n) → Fin (2*n) → Bool)
  (piece_count : Nat)
  (piece_count_eq : piece_count = 3*n)

/-- Represents a selection of rows and columns -/
structure Selection (n : ℕ) :=
  (rows : Fin n → Fin (2*n))
  (cols : Fin n → Fin (2*n))

/-- Checks if a selection results in an empty n × n chessboard -/
def is_empty_selection (cb : Chessboard n) (sel : Selection n) : Prop :=
  ∀ i j, ¬(cb.board (sel.rows i) (sel.cols j))

/-- Main theorem: There exists a selection that results in an empty n × n chessboard -/
theorem exists_empty_selection (n : ℕ) (cb : Chessboard n) :
  ∃ (sel : Selection n), is_empty_selection cb sel :=
sorry

end NUMINAMATH_CALUDE_exists_empty_selection_l1937_193700


namespace NUMINAMATH_CALUDE_no_triangle_tangent_and_inscribed_l1937_193753

/-- The problem statement as a theorem -/
theorem no_triangle_tangent_and_inscribed (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
  let C₂ : Set (ℝ × ℝ) := {p | p.1^2 / a^2 + p.2^2 / b^2 = 1}
  (1 : ℝ)^2 / a^2 + (1 : ℝ)^2 / b^2 = 1 →
  ¬ ∃ (A B C : ℝ × ℝ),
    (A ∈ C₂ ∧ B ∈ C₂ ∧ C ∈ C₂) ∧
    (∀ p : ℝ × ℝ, p ∈ C₁ → (dist p A ≥ dist A B ∧ dist p B ≥ dist A B ∧ dist p C ≥ dist A B)) :=
by
  sorry


end NUMINAMATH_CALUDE_no_triangle_tangent_and_inscribed_l1937_193753


namespace NUMINAMATH_CALUDE_faster_speed_calculation_l1937_193796

/-- Prove that given a person walks 50 km at 10 km/hr, if they walked at a faster speed,
    they would cover 20 km more in the same time, then the faster speed is 14 km/hr -/
theorem faster_speed_calculation (actual_distance : ℝ) (original_speed : ℝ) (additional_distance : ℝ)
    (h1 : actual_distance = 50)
    (h2 : original_speed = 10)
    (h3 : additional_distance = 20) :
  let total_distance := actual_distance + additional_distance
  let time := actual_distance / original_speed
  let faster_speed := total_distance / time
  faster_speed = 14 := by
sorry

end NUMINAMATH_CALUDE_faster_speed_calculation_l1937_193796


namespace NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l1937_193737

theorem average_marks_chemistry_mathematics 
  (P C M : ℕ) -- P: Physics marks, C: Chemistry marks, M: Mathematics marks
  (h : P + C + M = P + 130) -- Total marks condition
  : (C + M) / 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l1937_193737


namespace NUMINAMATH_CALUDE_tom_found_seven_seashells_l1937_193777

/-- The number of seashells Tom found yesterday -/
def seashells_yesterday : ℕ := sorry

/-- The number of seashells Tom found today -/
def seashells_today : ℕ := 4

/-- The total number of seashells Tom found -/
def total_seashells : ℕ := 11

/-- Theorem stating that Tom found 7 seashells yesterday -/
theorem tom_found_seven_seashells : seashells_yesterday = 7 := by
  sorry

end NUMINAMATH_CALUDE_tom_found_seven_seashells_l1937_193777


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1937_193745

/-- Two points are symmetric about the x-axis if their x-coordinates are the same
    and their y-coordinates are opposite numbers -/
def symmetric_about_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

theorem symmetric_points_sum (b a : ℝ) :
  symmetric_about_x_axis (-2, b) (a, -3) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1937_193745


namespace NUMINAMATH_CALUDE_order_of_numbers_l1937_193780

theorem order_of_numbers : 
  0 < 0.89 → 0.89 < 1 → 90.8 > 1 → Real.log 0.89 < 0 → 
  Real.log 0.89 < 0.89 ∧ 0.89 < 90.8 := by
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l1937_193780


namespace NUMINAMATH_CALUDE_peanuts_remaining_l1937_193790

theorem peanuts_remaining (initial : ℕ) (eaten_by_bonita : ℕ) : 
  initial = 148 → 
  eaten_by_bonita = 29 → 
  82 = initial - (initial / 4) - eaten_by_bonita := by
  sorry

end NUMINAMATH_CALUDE_peanuts_remaining_l1937_193790


namespace NUMINAMATH_CALUDE_cube_root_and_square_roots_l1937_193765

theorem cube_root_and_square_roots (a b m : ℝ) : 
  (3 * a - 5)^(1/3) = -2 ∧ 
  m^2 = b ∧ 
  (1 - 5*m)^2 = b →
  a = -1 ∧ b = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_and_square_roots_l1937_193765


namespace NUMINAMATH_CALUDE_guessing_game_difference_l1937_193733

theorem guessing_game_difference : (2 * 51) - (3 * 33) = 3 := by
  sorry

end NUMINAMATH_CALUDE_guessing_game_difference_l1937_193733


namespace NUMINAMATH_CALUDE_min_tiles_for_floor_l1937_193798

-- Define the length and breadth of the floor in centimeters
def floor_length : ℚ := 1625 / 100
def floor_width : ℚ := 1275 / 100

-- Define the function to calculate the number of tiles
def num_tiles (length width : ℚ) : ℕ :=
  let gcd := (Nat.gcd (Nat.floor (length * 100)) (Nat.floor (width * 100))) / 100
  let tile_area := gcd * gcd
  let floor_area := length * width
  Nat.ceil (floor_area / tile_area)

-- Theorem statement
theorem min_tiles_for_floor : num_tiles floor_length floor_width = 3315 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_for_floor_l1937_193798


namespace NUMINAMATH_CALUDE_triangle_side_length_l1937_193785

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) →
  (0 < a ∧ 0 < b ∧ 0 < c) →
  -- Given conditions
  (Real.cos A = Real.sqrt 5 / 5) →
  (Real.cos B = Real.sqrt 10 / 10) →
  (c = Real.sqrt 2) →
  -- Sine rule
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (c / Real.sin C = a / Real.sin A) →
  -- Prove
  a = 4 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1937_193785


namespace NUMINAMATH_CALUDE_quadratic_radicals_same_type_l1937_193771

theorem quadratic_radicals_same_type (a : ℝ) : 
  (∃ k : ℝ, k > 0 ∧ a - 3 = k * (12 - 2*a)) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radicals_same_type_l1937_193771


namespace NUMINAMATH_CALUDE_triangle_angle_B_l1937_193768

theorem triangle_angle_B (a b : ℝ) (A B : ℝ) : 
  a = 1 → b = Real.sqrt 2 → A = 30 * π / 180 → 
  (B = 45 * π / 180 ∨ B = 135 * π / 180) ↔ 
  (Real.sin B = b * Real.sin A / a) := by sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l1937_193768


namespace NUMINAMATH_CALUDE_abc_sqrt_problem_l1937_193784

theorem abc_sqrt_problem (a b c : ℝ) 
  (h1 : b + c = 17)
  (h2 : c + a = 18)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 72 := by
  sorry

end NUMINAMATH_CALUDE_abc_sqrt_problem_l1937_193784


namespace NUMINAMATH_CALUDE_john_boxes_l1937_193712

theorem john_boxes (stan_boxes : ℕ) (joseph_percent : ℚ) (jules_more : ℕ) (john_percent : ℚ)
  (h1 : stan_boxes = 100)
  (h2 : joseph_percent = 80/100)
  (h3 : jules_more = 5)
  (h4 : john_percent = 20/100) :
  let joseph_boxes := stan_boxes * (1 - joseph_percent)
  let jules_boxes := joseph_boxes + jules_more
  let john_boxes := jules_boxes * (1 + john_percent)
  john_boxes = 30 := by sorry

end NUMINAMATH_CALUDE_john_boxes_l1937_193712


namespace NUMINAMATH_CALUDE_gills_arrival_time_l1937_193763

/-- Represents the travel details of Gill's train journey --/
structure TravelDetails where
  departure_time : Nat  -- in minutes past midnight
  first_segment_distance : Nat  -- in km
  second_segment_distance : Nat  -- in km
  speed : Nat  -- in km/h
  stop_duration : Nat  -- in minutes

/-- Calculates the arrival time given the travel details --/
def calculate_arrival_time (details : TravelDetails) : Nat :=
  let first_segment_time := details.first_segment_distance * 60 / details.speed
  let second_segment_time := details.second_segment_distance * 60 / details.speed
  let total_travel_time := first_segment_time + details.stop_duration + second_segment_time
  details.departure_time + total_travel_time

/-- Gill's travel details --/
def gills_travel : TravelDetails :=
  { departure_time := 9 * 60  -- 09:00 in minutes
    first_segment_distance := 27
    second_segment_distance := 29
    speed := 96
    stop_duration := 3 }

theorem gills_arrival_time :
  calculate_arrival_time gills_travel = 9 * 60 + 38 := by
  sorry

end NUMINAMATH_CALUDE_gills_arrival_time_l1937_193763


namespace NUMINAMATH_CALUDE_min_minutes_for_cheaper_plan_b_l1937_193795

/-- Represents the cost of a phone plan in cents -/
def PlanCost := ℕ → ℕ

/-- Cost function for Plan A: 10 cents per minute -/
def planA : PlanCost := λ minutes => 10 * minutes

/-- Cost function for Plan B: $20 flat fee (2000 cents) plus 5 cents per minute -/
def planB : PlanCost := λ minutes => 2000 + 5 * minutes

/-- Theorem stating that 401 is the minimum number of minutes for Plan B to be cheaper -/
theorem min_minutes_for_cheaper_plan_b : 
  (∀ m : ℕ, m < 401 → planA m ≤ planB m) ∧ 
  (∀ m : ℕ, m ≥ 401 → planB m < planA m) := by
  sorry

end NUMINAMATH_CALUDE_min_minutes_for_cheaper_plan_b_l1937_193795


namespace NUMINAMATH_CALUDE_no_solution_iff_k_eq_ten_l1937_193748

theorem no_solution_iff_k_eq_ten (k : ℝ) : 
  (∀ x : ℝ, (3*x + 1 ≠ 0 ∧ 5*x + 4 ≠ 0) → ((2*x - 4)/(3*x + 1) ≠ (2*x - k)/(5*x + 4))) ↔ 
  k = 10 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_eq_ten_l1937_193748


namespace NUMINAMATH_CALUDE_gcd_lcm_product_90_135_l1937_193782

theorem gcd_lcm_product_90_135 : Nat.gcd 90 135 * Nat.lcm 90 135 = 12150 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_90_135_l1937_193782


namespace NUMINAMATH_CALUDE_moon_permutations_eq_12_l1937_193792

/-- The number of distinct permutations of the letters in "MOON" -/
def moon_permutations : ℕ :=
  Nat.factorial 4 / Nat.factorial 2

theorem moon_permutations_eq_12 : moon_permutations = 12 := by
  sorry

end NUMINAMATH_CALUDE_moon_permutations_eq_12_l1937_193792


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l1937_193717

theorem quadratic_equation_problem (k : ℝ) : 
  (∀ x, 4 * x^2 - 6 * x * Real.sqrt 3 + k = 0 → 
    (6 * Real.sqrt 3)^2 - 4 * 4 * k = 18) → 
  k = 45/8 ∧ ∃ x y, x ≠ y ∧ 4 * x^2 - 6 * x * Real.sqrt 3 + k = 0 ∧ 
                           4 * y^2 - 6 * y * Real.sqrt 3 + k = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l1937_193717


namespace NUMINAMATH_CALUDE_initial_money_calculation_l1937_193718

/-- Calculates the initial amount of money given spending habits and remaining balance --/
theorem initial_money_calculation 
  (spend_per_trip : ℕ)
  (trips_per_month : ℕ)
  (months : ℕ)
  (money_left : ℕ)
  (h1 : spend_per_trip = 2)
  (h2 : trips_per_month = 4)
  (h3 : months = 12)
  (h4 : money_left = 104) :
  spend_per_trip * trips_per_month * months + money_left = 200 := by
  sorry

#check initial_money_calculation

end NUMINAMATH_CALUDE_initial_money_calculation_l1937_193718


namespace NUMINAMATH_CALUDE_angle_relations_l1937_193719

theorem angle_relations (α β : ℝ) 
  (h_acute_α : 0 < α ∧ α < π / 2) 
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_tan_α : Real.tan α = 4 / 3)
  (h_sin_diff : Real.sin (α - β) = -(Real.sqrt 5) / 5) :
  Real.cos (2 * α) = -7 / 25 ∧ 
  Real.tan (α + β) = -41 / 38 := by
sorry

end NUMINAMATH_CALUDE_angle_relations_l1937_193719


namespace NUMINAMATH_CALUDE_calculation_proof_l1937_193769

theorem calculation_proof :
  (2 * Real.sqrt 18 - 3 * Real.sqrt 2 - Real.sqrt (1/2) = (5 * Real.sqrt 2) / 2) ∧
  ((Real.sqrt 3 - 1)^2 - (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 3 - Real.sqrt 2) = 3 - 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l1937_193769


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1937_193762

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) :
  z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1937_193762


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l1937_193770

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

/-- Condition for f to have both maximum and minimum -/
def has_max_and_min (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f' a x = 0 ∧ f' a y = 0

theorem cubic_function_extrema (a : ℝ) :
  has_max_and_min a → a < -3 ∨ a > 6 :=
sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l1937_193770


namespace NUMINAMATH_CALUDE_absolute_value_of_x_minus_five_l1937_193725

theorem absolute_value_of_x_minus_five (x : ℝ) (h : x = 4) : |x - 5| = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_x_minus_five_l1937_193725


namespace NUMINAMATH_CALUDE_rain_thunder_prob_is_correct_l1937_193711

/-- The probability of rain with thunder on both Monday and Tuesday -/
def rain_thunder_prob : ℝ :=
  let rain_monday_prob : ℝ := 0.40
  let rain_tuesday_prob : ℝ := 0.30
  let thunder_given_rain_prob : ℝ := 0.10
  let rain_both_days_prob : ℝ := rain_monday_prob * rain_tuesday_prob
  let thunder_both_days_given_rain_prob : ℝ := thunder_given_rain_prob * thunder_given_rain_prob
  rain_both_days_prob * thunder_both_days_given_rain_prob * 100

theorem rain_thunder_prob_is_correct : rain_thunder_prob = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_rain_thunder_prob_is_correct_l1937_193711


namespace NUMINAMATH_CALUDE_original_number_l1937_193728

theorem original_number (x : ℝ) : ((x - 8 + 7) / 5) * 4 = 16 → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l1937_193728


namespace NUMINAMATH_CALUDE_value_of_a_l1937_193713

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3*x + 1

theorem value_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1) 1, f a x ≥ 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1937_193713


namespace NUMINAMATH_CALUDE_gate_code_combinations_l1937_193742

theorem gate_code_combinations : Nat.factorial 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gate_code_combinations_l1937_193742


namespace NUMINAMATH_CALUDE_star_properties_l1937_193714

-- Define the star operation
def star (a b : ℝ) : ℝ := a + b + a * b

-- State the theorem
theorem star_properties :
  -- There exists an identity element E
  ∃ E : ℝ, (∀ a : ℝ, star a E = a) ∧ (star E E = E) ∧
  -- The operation is commutative
  (∀ a b : ℝ, star a b = star b a) ∧
  -- The operation is associative
  (∀ a b c : ℝ, star (star a b) c = star a (star b c)) :=
sorry

end NUMINAMATH_CALUDE_star_properties_l1937_193714


namespace NUMINAMATH_CALUDE_sum_of_z_values_l1937_193793

-- Define the function f
def f (x : ℝ) : ℝ := (2*x)^2 - 2*(2*x) + 2

-- State the theorem
theorem sum_of_z_values (z : ℝ) : 
  (∃ z₁ z₂, f z₁ = 4 ∧ f z₂ = 4 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = 1/2) := by sorry

end NUMINAMATH_CALUDE_sum_of_z_values_l1937_193793


namespace NUMINAMATH_CALUDE_unique_teammate_d_score_l1937_193760

-- Define the scoring system
def single_points : ℕ := 1
def double_points : ℕ := 2
def triple_points : ℕ := 3
def home_run_points : ℕ := 4

-- Define the total team score
def total_team_score : ℕ := 68

-- Define Faye's score
def faye_score : ℕ := 28

-- Define Teammate A's score components
def teammate_a_singles : ℕ := 1
def teammate_a_doubles : ℕ := 3
def teammate_a_home_runs : ℕ := 1

-- Define Teammate B's score components
def teammate_b_singles : ℕ := 4
def teammate_b_doubles : ℕ := 2
def teammate_b_triples : ℕ := 1

-- Define Teammate C's score components
def teammate_c_singles : ℕ := 2
def teammate_c_doubles : ℕ := 1
def teammate_c_triples : ℕ := 2
def teammate_c_home_runs : ℕ := 1

-- Theorem: There must be exactly one more player (Teammate D) who scored 4 points
theorem unique_teammate_d_score : 
  ∃! teammate_d_score : ℕ, 
    faye_score + 
    (teammate_a_singles * single_points + teammate_a_doubles * double_points + teammate_a_home_runs * home_run_points) +
    (teammate_b_singles * single_points + teammate_b_doubles * double_points + teammate_b_triples * triple_points) +
    (teammate_c_singles * single_points + teammate_c_doubles * double_points + teammate_c_triples * triple_points + teammate_c_home_runs * home_run_points) +
    teammate_d_score = total_team_score ∧ 
    teammate_d_score = 4 := by sorry

end NUMINAMATH_CALUDE_unique_teammate_d_score_l1937_193760


namespace NUMINAMATH_CALUDE_triangle_area_l1937_193775

/-- The area of the triangle bounded by y = x, y = -x, and y = 8 is 64 -/
theorem triangle_area : Real := by
  -- Define the lines
  let line1 : Real → Real := λ x ↦ x
  let line2 : Real → Real := λ x ↦ -x
  let line3 : Real → Real := λ _ ↦ 8

  -- Define the intersection points
  let A : (Real × Real) := (8, 8)
  let B : (Real × Real) := (-8, 8)
  let O : (Real × Real) := (0, 0)

  -- Calculate the base and height of the triangle
  let base : Real := A.1 - B.1
  let height : Real := line3 0 - O.2

  -- Calculate the area
  let area : Real := (1 / 2) * base * height

  -- Prove that the area is 64
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1937_193775


namespace NUMINAMATH_CALUDE_sue_dog_walking_charge_l1937_193726

/-- The amount Sue charged per dog for walking --/
def sue_charge_per_dog (perfume_cost christian_initial sue_initial christian_yards christian_yard_price sue_dogs additional_needed : ℚ) : ℚ :=
  let christian_total := christian_initial + christian_yards * christian_yard_price
  let initial_total := christian_total + sue_initial
  let needed := perfume_cost - initial_total
  let sue_earned := needed - additional_needed
  sue_earned / sue_dogs

theorem sue_dog_walking_charge 
  (perfume_cost : ℚ)
  (christian_initial : ℚ)
  (sue_initial : ℚ)
  (christian_yards : ℚ)
  (christian_yard_price : ℚ)
  (sue_dogs : ℚ)
  (additional_needed : ℚ)
  (h1 : perfume_cost = 50)
  (h2 : christian_initial = 5)
  (h3 : sue_initial = 7)
  (h4 : christian_yards = 4)
  (h5 : christian_yard_price = 5)
  (h6 : sue_dogs = 6)
  (h7 : additional_needed = 6) :
  sue_charge_per_dog perfume_cost christian_initial sue_initial christian_yards christian_yard_price sue_dogs additional_needed = 2 :=
by sorry

end NUMINAMATH_CALUDE_sue_dog_walking_charge_l1937_193726


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l1937_193721

theorem gcd_lcm_sum : Nat.gcd 28 63 + Nat.lcm 18 24 = 79 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l1937_193721


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l1937_193708

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the foci of the hyperbola
def left_focus : ℝ × ℝ := (-5, 0)
def right_focus : ℝ × ℝ := (5, 0)

-- Define point A
def point_A : ℝ × ℝ := (6, -2)

-- Define the line passing through right focus and point A
def line_through_right_focus (x y : ℝ) : Prop := 2*x + y - 10 = 0

-- Define the perpendicular line passing through left focus
def perpendicular_line (x y : ℝ) : Prop := x - 2*y + 5 = 0

-- The theorem to prove
theorem intersection_point_coordinates :
  ∃ (x y : ℝ), 
    hyperbola x y ∧
    line_through_right_focus x y ∧
    perpendicular_line x y ∧
    x = 3 ∧ y = 4 := by sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l1937_193708


namespace NUMINAMATH_CALUDE_staircase_perimeter_l1937_193756

/-- A staircase-shaped region with specific properties -/
structure StaircaseRegion where
  num_sides : ℕ
  side_length : ℝ
  total_area : ℝ

/-- The perimeter of a StaircaseRegion -/
def perimeter (r : StaircaseRegion) : ℝ := sorry

/-- Theorem stating the perimeter of a specific StaircaseRegion -/
theorem staircase_perimeter :
  ∀ (r : StaircaseRegion),
    r.num_sides = 12 ∧
    r.side_length = 1 ∧
    r.total_area = 120 →
    perimeter r = 36 := by sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l1937_193756


namespace NUMINAMATH_CALUDE_largest_fraction_add_to_one_seventh_l1937_193734

theorem largest_fraction_add_to_one_seventh :
  ∀ (a b : ℕ) (hb : 0 < b) (hb_lt_5 : b < 5),
    (1 : ℚ) / 7 + (a : ℚ) / b < 1 →
    (a : ℚ) / b ≤ 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_add_to_one_seventh_l1937_193734


namespace NUMINAMATH_CALUDE_prob_three_tails_in_eight_flips_l1937_193730

/-- The probability of flipping a tail -/
def p_tail : ℚ := 3/4

/-- The probability of flipping a head -/
def p_head : ℚ := 1/4

/-- The number of coin flips -/
def n_flips : ℕ := 8

/-- The number of tails we want to get -/
def n_tails : ℕ := 3

/-- The probability of getting exactly n_tails in n_flips of an unfair coin -/
def prob_exact_tails (n_flips n_tails : ℕ) (p_tail : ℚ) : ℚ :=
  (n_flips.choose n_tails) * (p_tail ^ n_tails) * ((1 - p_tail) ^ (n_flips - n_tails))

theorem prob_three_tails_in_eight_flips : 
  prob_exact_tails n_flips n_tails p_tail = 189/512 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_tails_in_eight_flips_l1937_193730


namespace NUMINAMATH_CALUDE_task_completion_time_l1937_193732

/-- Ram's efficiency is half of Krish's, and Ram takes 27 days to complete a task alone.
    This theorem proves that Ram and Krish working together will complete the task in 9 days. -/
theorem task_completion_time (ram_efficiency krish_efficiency : ℝ) 
  (h1 : ram_efficiency = (1 / 2) * krish_efficiency) 
  (h2 : ram_efficiency * 27 = 1) : 
  (ram_efficiency + krish_efficiency) * 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_task_completion_time_l1937_193732


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1937_193731

theorem pure_imaginary_fraction (a : ℝ) : 
  (((1 : ℂ) + 2 * Complex.I) / (a + Complex.I)).re = 0 ∧ 
  (((1 : ℂ) + 2 * Complex.I) / (a + Complex.I)).im ≠ 0 → 
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1937_193731


namespace NUMINAMATH_CALUDE_shortened_card_area_l1937_193757

/-- Represents a rectangular card with given dimensions -/
structure Card where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular card -/
def area (c : Card) : ℝ := c.length * c.width

/-- Represents the amount by which each side is shortened -/
structure Shortening where
  length_reduction : ℝ
  width_reduction : ℝ

/-- Applies a shortening to a card -/
def apply_shortening (c : Card) (s : Shortening) : Card :=
  { length := c.length - s.length_reduction,
    width := c.width - s.width_reduction }

theorem shortened_card_area 
  (original : Card)
  (shortening : Shortening)
  (h1 : original.length = 5)
  (h2 : original.width = 7)
  (h3 : shortening.length_reduction = 2)
  (h4 : shortening.width_reduction = 1) :
  area (apply_shortening original shortening) = 18 := by
  sorry

end NUMINAMATH_CALUDE_shortened_card_area_l1937_193757


namespace NUMINAMATH_CALUDE_ten_children_same_cards_l1937_193797

/-- Represents the number of children who can form a specific word -/
structure WordCount where
  mama : ℕ
  nyanya : ℕ
  manya : ℕ

/-- Calculates the number of children with all three cards the same -/
def childrenWithSameCards (wc : WordCount) : ℕ :=
  wc.mama + wc.nyanya - wc.manya

/-- Theorem stating that 10 children have all three cards the same -/
theorem ten_children_same_cards (wc : WordCount) 
  (h_mama : wc.mama = 20)
  (h_nyanya : wc.nyanya = 30)
  (h_manya : wc.manya = 40) : 
  childrenWithSameCards wc = 10 := by
sorry

#eval childrenWithSameCards ⟨20, 30, 40⟩

end NUMINAMATH_CALUDE_ten_children_same_cards_l1937_193797


namespace NUMINAMATH_CALUDE_new_person_weight_l1937_193710

/-- The weight of the new person given the conditions of the problem -/
def weight_new_person (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + n * avg_increase

/-- Theorem stating that the weight of the new person is 87.5 kg -/
theorem new_person_weight :
  weight_new_person 9 2.5 65 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1937_193710


namespace NUMINAMATH_CALUDE_smallest_positive_a_quartic_polynomial_l1937_193715

theorem smallest_positive_a_quartic_polynomial (a b : ℝ) : 
  (∀ x : ℝ, x^4 - a*x^3 + b*x^2 - a*x + a = 0 → x > 0) →
  (∀ c : ℝ, c > 0 → (∃ d : ℝ, ∀ x : ℝ, x^4 - c*x^3 + d*x^2 - c*x + c = 0 → x > 0) → c ≥ a) →
  b = 6 * (4^(1/3))^2 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_a_quartic_polynomial_l1937_193715


namespace NUMINAMATH_CALUDE_expression_evaluation_l1937_193794

theorem expression_evaluation : 
  let mixed_number : ℚ := 20 + 94 / 95
  let expression := (mixed_number * 1.65 - mixed_number + 7 / 20 * mixed_number) * 47.5 * 0.8 * 2.5
  expression = 1994 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1937_193794


namespace NUMINAMATH_CALUDE_divisibility_of_expression_l1937_193786

theorem divisibility_of_expression (x : ℤ) (h : Odd x) :
  ∃ k : ℤ, (8 * x + 6) * (8 * x + 10) * (4 * x + 4) = 384 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_expression_l1937_193786


namespace NUMINAMATH_CALUDE_initial_rulers_count_l1937_193727

/-- The number of rulers initially in the drawer -/
def initial_rulers : ℕ := sorry

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 34

/-- The number of rulers taken out of the drawer -/
def rulers_taken : ℕ := 11

/-- The number of rulers remaining in the drawer after removal -/
def rulers_remaining : ℕ := 3

theorem initial_rulers_count : initial_rulers = 14 := by sorry

end NUMINAMATH_CALUDE_initial_rulers_count_l1937_193727


namespace NUMINAMATH_CALUDE_stevens_falls_l1937_193709

theorem stevens_falls (steven_falls : ℕ) (stephanie_falls : ℕ) (sonya_falls : ℕ) 
  (h1 : stephanie_falls = steven_falls + 13)
  (h2 : sonya_falls = 6)
  (h3 : sonya_falls = stephanie_falls / 2 - 2) : 
  steven_falls = 3 := by
  sorry

end NUMINAMATH_CALUDE_stevens_falls_l1937_193709


namespace NUMINAMATH_CALUDE_cos_inv_third_over_pi_irrational_l1937_193740

theorem cos_inv_third_over_pi_irrational : Irrational ((Real.arccos (1/3)) / Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cos_inv_third_over_pi_irrational_l1937_193740


namespace NUMINAMATH_CALUDE_table_tennis_arrangements_l1937_193738

def total_players : ℕ := 10
def main_players : ℕ := 3
def match_players : ℕ := 5
def remaining_players : ℕ := total_players - main_players

theorem table_tennis_arrangements :
  (main_players.factorial) * (remaining_players.choose 2) = 252 :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_arrangements_l1937_193738


namespace NUMINAMATH_CALUDE_stratified_sample_bulbs_l1937_193754

/-- Represents the types of bulbs -/
inductive BulbType
  | W20
  | W40
  | W60

/-- Calculates the number of bulbs of a given type in a sample -/
def sampleSize (totalBulbs : ℕ) (sampleBulbs : ℕ) (ratio : ℕ) (totalRatio : ℕ) : ℕ :=
  (ratio * totalBulbs * sampleBulbs) / (totalRatio * totalBulbs)

theorem stratified_sample_bulbs :
  let totalBulbs : ℕ := 400
  let sampleBulbs : ℕ := 40
  let ratio20W : ℕ := 4
  let ratio40W : ℕ := 3
  let ratio60W : ℕ := 1
  let totalRatio : ℕ := ratio20W + ratio40W + ratio60W
  (sampleSize totalBulbs sampleBulbs ratio20W totalRatio = 20) ∧
  (sampleSize totalBulbs sampleBulbs ratio40W totalRatio = 15) ∧
  (sampleSize totalBulbs sampleBulbs ratio60W totalRatio = 5) :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_bulbs_l1937_193754


namespace NUMINAMATH_CALUDE_problem_statement_l1937_193707

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_eq : 1/a + 9/b = 1) 
  (h_ineq : ∀ x : ℝ, a + b ≥ -x^2 + 4*x + 18 - m) : 
  m ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1937_193707


namespace NUMINAMATH_CALUDE_expand_difference_of_squares_simplify_fraction_l1937_193755

-- Define a as a real number
variable (a : ℝ)

-- Theorem 1: (a+2)(a-2) = a^2 - 4
theorem expand_difference_of_squares : (a + 2) * (a - 2) = a^2 - 4 := by
  sorry

-- Theorem 2: (a^2-4)/(a+2) + 2 = a
theorem simplify_fraction : (a^2 - 4) / (a + 2) + 2 = a := by
  sorry

end NUMINAMATH_CALUDE_expand_difference_of_squares_simplify_fraction_l1937_193755


namespace NUMINAMATH_CALUDE_cross_number_puzzle_digit_l1937_193750

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def power_of_2 (m : ℕ) : ℕ := 2^m
def power_of_3 (n : ℕ) : ℕ := 3^n

def same_digit_position (a b : ℕ) (pos : ℕ) : Prop :=
  (a / 10^pos) % 10 = (b / 10^pos) % 10

theorem cross_number_puzzle_digit :
  ∃! d : ℕ, d < 10 ∧
    ∃ (m n pos : ℕ),
      is_three_digit (power_of_2 m) ∧
      is_three_digit (power_of_3 n) ∧
      same_digit_position (power_of_2 m) (power_of_3 n) pos ∧
      (power_of_2 m / 10^pos) % 10 = d :=
by
  sorry

end NUMINAMATH_CALUDE_cross_number_puzzle_digit_l1937_193750


namespace NUMINAMATH_CALUDE_sqrt_two_expansion_l1937_193779

theorem sqrt_two_expansion (a b : ℚ) : 
  (1 + Real.sqrt 2)^5 = a + Real.sqrt 2 * b → a - b = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_expansion_l1937_193779


namespace NUMINAMATH_CALUDE_two_aces_probability_l1937_193747

-- Define the total number of cards in a standard deck
def totalCards : ℕ := 52

-- Define the number of Aces in a standard deck
def numAces : ℕ := 4

-- Define the probability of drawing two Aces
def probTwoAces : ℚ := 1 / 221

-- Theorem statement
theorem two_aces_probability :
  (numAces / totalCards) * ((numAces - 1) / (totalCards - 1)) = probTwoAces := by
  sorry

end NUMINAMATH_CALUDE_two_aces_probability_l1937_193747


namespace NUMINAMATH_CALUDE_distance_philadelphia_los_angeles_l1937_193788

/-- The distance between two points on a complex plane, where one point is at (1950, 1950) and the other is at (0, 0), is equal to 1950√2. -/
theorem distance_philadelphia_los_angeles : 
  let philadelphia : ℂ := 1950 + 1950 * Complex.I
  let los_angeles : ℂ := 0
  Complex.abs (philadelphia - los_angeles) = 1950 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_philadelphia_los_angeles_l1937_193788


namespace NUMINAMATH_CALUDE_variance_of_X_l1937_193705

/-- A random variable X with two possible values -/
def X : Fin 2 → ℝ
  | 0 => 0
  | 1 => 1

/-- The probability mass function of X -/
def P : Fin 2 → ℝ
  | 0 => 0.4
  | 1 => 0.6

/-- The expected value of X -/
def E : ℝ := X 0 * P 0 + X 1 * P 1

/-- The variance of X -/
def D : ℝ := (X 0 - E)^2 * P 0 + (X 1 - E)^2 * P 1

/-- Theorem: The variance of X is 0.24 -/
theorem variance_of_X : D = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_X_l1937_193705


namespace NUMINAMATH_CALUDE_percent_equality_l1937_193702

theorem percent_equality (x : ℝ) : (35 / 100 * 400 = 20 / 100 * x) → x = 700 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l1937_193702


namespace NUMINAMATH_CALUDE_library_books_total_l1937_193724

theorem library_books_total (initial_books additional_books : ℕ) 
  (h1 : initial_books = 54)
  (h2 : additional_books = 23) :
  initial_books + additional_books = 77 := by
sorry

end NUMINAMATH_CALUDE_library_books_total_l1937_193724


namespace NUMINAMATH_CALUDE_sum_and_count_equals_851_l1937_193722

/-- Sum of integers from a to b, inclusive -/
def sumIntegers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

/-- Count of even integers from a to b, inclusive -/
def countEvenIntegers (a b : ℕ) : ℕ := (b - a) / 2 + 1

/-- The sum of integers from 30 to 50 (inclusive) plus the count of even integers
    in the same range equals 851 -/
theorem sum_and_count_equals_851 : sumIntegers 30 50 + countEvenIntegers 30 50 = 851 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_equals_851_l1937_193722


namespace NUMINAMATH_CALUDE_prism_tetrahedron_surface_area_ratio_l1937_193783

/-- The ratio of surface areas of a rectangular prism to a tetrahedron --/
theorem prism_tetrahedron_surface_area_ratio :
  let prism_dimensions : Fin 3 → ℝ := ![2, 3, 4]
  let prism_surface_area := 2 * (prism_dimensions 0 * prism_dimensions 1 + 
                                 prism_dimensions 1 * prism_dimensions 2 + 
                                 prism_dimensions 0 * prism_dimensions 2)
  let tetrahedron_edge_length := Real.sqrt 13
  let tetrahedron_surface_area := Real.sqrt 3 * tetrahedron_edge_length ^ 2
  prism_surface_area / tetrahedron_surface_area = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_prism_tetrahedron_surface_area_ratio_l1937_193783


namespace NUMINAMATH_CALUDE_cube_root_three_identity_l1937_193778

theorem cube_root_three_identity (t : ℝ) : 
  t = 1 / (1 - Real.rpow 3 (1/3)) → 
  t = -(1 + Real.rpow 3 (1/3) + Real.rpow 3 (2/3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_three_identity_l1937_193778


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1937_193773

theorem imaginary_part_of_z (z : ℂ) : z = (1 - Complex.I) / Complex.I → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1937_193773


namespace NUMINAMATH_CALUDE_island_width_is_five_l1937_193723

/-- Represents a rectangular island -/
structure Island where
  length : ℝ
  width : ℝ
  area : ℝ

/-- The area of a rectangular island is equal to its length multiplied by its width -/
axiom island_area (i : Island) : i.area = i.length * i.width

/-- Given an island with area 50 square miles and length 10 miles, prove its width is 5 miles -/
theorem island_width_is_five (i : Island) 
  (h_area : i.area = 50) 
  (h_length : i.length = 10) : 
  i.width = 5 := by
sorry

end NUMINAMATH_CALUDE_island_width_is_five_l1937_193723


namespace NUMINAMATH_CALUDE_even_difference_of_coefficients_l1937_193749

theorem even_difference_of_coefficients (a₁ a₂ b₁ b₂ m n : ℤ) : 
  a₁ ≠ a₂ →
  m ≠ n →
  (m^2 + a₁*m + b₁ = n^2 + a₂*n + b₂) →
  (m^2 + a₂*m + b₂ = n^2 + a₁*n + b₁) →
  ∃ k : ℤ, a₁ - a₂ = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_even_difference_of_coefficients_l1937_193749


namespace NUMINAMATH_CALUDE_ab_geq_one_implies_conditions_l1937_193767

theorem ab_geq_one_implies_conditions (a b : ℝ) (h : a * b ≥ 1) :
  a^2 ≥ 1 / b^2 ∧ a^2 + b^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_geq_one_implies_conditions_l1937_193767


namespace NUMINAMATH_CALUDE_interior_angles_sum_l1937_193704

/-- The sum of interior angles of a triangle in degrees -/
def triangle_angle_sum : ℝ := 180

/-- The number of triangles a quadrilateral can be divided into -/
def quadrilateral_triangles : ℕ := 2

/-- The number of triangles a pentagon can be divided into -/
def pentagon_triangles : ℕ := 3

/-- The number of triangles a convex n-gon can be divided into -/
def n_gon_triangles (n : ℕ) : ℕ := n - 2

/-- The sum of interior angles of a quadrilateral -/
def quadrilateral_angle_sum : ℝ := triangle_angle_sum * quadrilateral_triangles

/-- The sum of interior angles of a convex pentagon -/
def pentagon_angle_sum : ℝ := triangle_angle_sum * pentagon_triangles

/-- The sum of interior angles of a convex n-gon -/
def n_gon_angle_sum (n : ℕ) : ℝ := triangle_angle_sum * n_gon_triangles n

theorem interior_angles_sum :
  (quadrilateral_angle_sum = 360) ∧
  (pentagon_angle_sum = 540) ∧
  (∀ n : ℕ, n_gon_angle_sum n = 180 * (n - 2)) :=
sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l1937_193704


namespace NUMINAMATH_CALUDE_final_cost_calculation_l1937_193744

def washing_machine_cost : ℝ := 100
def dryer_cost : ℝ := washing_machine_cost - 30
def discount_rate : ℝ := 0.1

theorem final_cost_calculation :
  let total_cost : ℝ := washing_machine_cost + dryer_cost
  let discount_amount : ℝ := total_cost * discount_rate
  let final_cost : ℝ := total_cost - discount_amount
  final_cost = 153 := by sorry

end NUMINAMATH_CALUDE_final_cost_calculation_l1937_193744


namespace NUMINAMATH_CALUDE_holiday_savings_l1937_193751

theorem holiday_savings (victory_savings sam_savings : ℕ) : 
  victory_savings = sam_savings - 100 →
  victory_savings + sam_savings = 1900 →
  sam_savings = 1000 := by
sorry

end NUMINAMATH_CALUDE_holiday_savings_l1937_193751


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1937_193746

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (7 * x₁^2 - (a + 13) * x₁ + a^2 - a - 2 = 0) ∧ 
    (7 * x₂^2 - (a + 13) * x₂ + a^2 - a - 2 = 0) ∧ 
    (0 < x₁) ∧ (x₁ < 1) ∧ (1 < x₂) ∧ (x₂ < 2)) →
  ((-2 < a ∧ a < -1) ∨ (3 < a ∧ a < 4)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1937_193746


namespace NUMINAMATH_CALUDE_total_notes_count_l1937_193735

/-- Proves that given a total amount of Rs. 10350 in Rs. 50 and Rs. 500 notes,
    with 77 notes of Rs. 50 denomination, the total number of notes is 90. -/
theorem total_notes_count (total_amount : ℕ) (notes_50_count : ℕ) (notes_50_value : ℕ) (notes_500_value : ℕ) :
  total_amount = 10350 →
  notes_50_count = 77 →
  notes_50_value = 50 →
  notes_500_value = 500 →
  ∃ (notes_500_count : ℕ),
    total_amount = notes_50_count * notes_50_value + notes_500_count * notes_500_value ∧
    notes_50_count + notes_500_count = 90 :=
by sorry

end NUMINAMATH_CALUDE_total_notes_count_l1937_193735


namespace NUMINAMATH_CALUDE_total_pushups_is_53_l1937_193791

/-- The number of push-ups David did -/
def david_pushups : ℕ := 51

/-- The difference between David's and Zachary's push-ups -/
def pushup_difference : ℕ := 49

/-- Calculates the total number of push-ups done by David and Zachary -/
def total_pushups : ℕ := david_pushups + (david_pushups - pushup_difference)

theorem total_pushups_is_53 : total_pushups = 53 := by
  sorry

end NUMINAMATH_CALUDE_total_pushups_is_53_l1937_193791


namespace NUMINAMATH_CALUDE_class_reading_total_l1937_193764

/-- Calculates the total number of books read by a class per week given the following conditions:
  * There are 12 girls and 10 boys in the class.
  * 5/6 of the girls and 4/5 of the boys are reading.
  * Girls read at an average rate of 3 books per week.
  * Boys read at an average rate of 2 books per week.
  * 20% of reading girls read at a faster rate of 5 books per week.
  * 10% of reading boys read at a slower rate of 1 book per week.
-/
theorem class_reading_total (girls : ℕ) (boys : ℕ) 
  (girls_reading_ratio : ℚ) (boys_reading_ratio : ℚ)
  (girls_avg_rate : ℕ) (boys_avg_rate : ℕ)
  (girls_faster_ratio : ℚ) (boys_slower_ratio : ℚ)
  (girls_faster_rate : ℕ) (boys_slower_rate : ℕ) :
  girls = 12 →
  boys = 10 →
  girls_reading_ratio = 5/6 →
  boys_reading_ratio = 4/5 →
  girls_avg_rate = 3 →
  boys_avg_rate = 2 →
  girls_faster_ratio = 1/5 →
  boys_slower_ratio = 1/10 →
  girls_faster_rate = 5 →
  boys_slower_rate = 1 →
  (girls_reading_ratio * girls * girls_avg_rate +
   boys_reading_ratio * boys * boys_avg_rate +
   girls_reading_ratio * girls * girls_faster_ratio * (girls_faster_rate - girls_avg_rate) +
   boys_reading_ratio * boys * boys_slower_ratio * (boys_slower_rate - boys_avg_rate)) = 49 := by
  sorry


end NUMINAMATH_CALUDE_class_reading_total_l1937_193764


namespace NUMINAMATH_CALUDE_intersection_of_P_and_complement_of_M_l1937_193736

-- Define the sets
def U : Set ℝ := Set.univ
def P : Set ℝ := {x | x ≥ 3}
def M : Set ℝ := {x | x < 4}

-- State the theorem
theorem intersection_of_P_and_complement_of_M :
  P ∩ (Set.univ \ M) = {x : ℝ | x ≥ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_complement_of_M_l1937_193736


namespace NUMINAMATH_CALUDE_shoe_matching_probability_l1937_193743

/-- Represents the number of pairs of shoes for each color -/
structure ShoeInventory :=
  (black : ℕ)
  (brown : ℕ)
  (gray : ℕ)
  (red : ℕ)

/-- Calculates the probability of picking a matching pair of different feet -/
def matchingProbability (inventory : ShoeInventory) : ℚ :=
  let totalShoes := 2 * (inventory.black + inventory.brown + inventory.gray + inventory.red)
  let matchingPairs := 
    inventory.black * (inventory.black - 1) +
    inventory.brown * (inventory.brown - 1) +
    inventory.gray * (inventory.gray - 1) +
    inventory.red * (inventory.red - 1)
  ↑matchingPairs / (totalShoes * (totalShoes - 1))

theorem shoe_matching_probability (inventory : ShoeInventory) :
  inventory.black = 8 ∧ 
  inventory.brown = 4 ∧ 
  inventory.gray = 3 ∧ 
  inventory.red = 2 →
  matchingProbability inventory = 93 / 551 :=
by sorry

end NUMINAMATH_CALUDE_shoe_matching_probability_l1937_193743


namespace NUMINAMATH_CALUDE_xyz_sum_equals_zero_l1937_193787

theorem xyz_sum_equals_zero 
  (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_eq1 : x^2 + x*y + y^2 = 48)
  (h_eq2 : y^2 + y*z + z^2 = 25)
  (h_eq3 : z^2 + x*z + x^2 = 73) :
  x*y + y*z + x*z = 0 :=
sorry

end NUMINAMATH_CALUDE_xyz_sum_equals_zero_l1937_193787


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_l1937_193703

/-- Given two nonconstant geometric sequences with terms k, a₁, a₂ and k, b₁, b₂ respectively,
    with different common ratios p and r, if a₂-b₂=5(a₁-b₁), then p + r = 5. -/
theorem sum_of_common_ratios (k p r : ℝ) (h_p_neq_r : p ≠ r) (h_p_neq_1 : p ≠ 1) (h_r_neq_1 : r ≠ 1)
    (h_eq : k * p^2 - k * r^2 = 5 * (k * p - k * r)) :
  p + r = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_l1937_193703


namespace NUMINAMATH_CALUDE_percentage_difference_l1937_193776

theorem percentage_difference (third : ℝ) (first second : ℝ) 
  (h1 : first = 0.75 * third) 
  (h2 : second = first - 0.06 * first) : 
  (third - second) / third = 0.295 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1937_193776


namespace NUMINAMATH_CALUDE_wheat_profit_percentage_l1937_193706

/-- Calculates the profit percentage for wheat mixture sales --/
theorem wheat_profit_percentage
  (weight1 : ℝ) (price1 : ℝ) (weight2 : ℝ) (price2 : ℝ) (selling_price : ℝ)
  (h1 : weight1 = 30)
  (h2 : price1 = 11.5)
  (h3 : weight2 = 20)
  (h4 : price2 = 14.25)
  (h5 : selling_price = 17.01) :
  let total_cost := weight1 * price1 + weight2 * price2
  let total_weight := weight1 + weight2
  let cost_per_kg := total_cost / total_weight
  let total_selling_price := selling_price * total_weight
  let profit := total_selling_price - total_cost
  let profit_percentage := (profit / total_cost) * 100
  ∃ ε > 0, abs (profit_percentage - 35) < ε :=
by sorry

end NUMINAMATH_CALUDE_wheat_profit_percentage_l1937_193706
