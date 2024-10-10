import Mathlib

namespace initial_average_calculation_l3535_353593

theorem initial_average_calculation (n : ℕ) (correct_avg : ℚ) (error : ℚ) :
  n = 10 →
  correct_avg = 16 →
  error = 10 →
  (n * correct_avg - error) / n = 15 :=
by sorry

end initial_average_calculation_l3535_353593


namespace nearest_multiple_of_21_to_2304_l3535_353570

theorem nearest_multiple_of_21_to_2304 :
  ∀ n : ℤ, n ≠ 2304 → 21 ∣ n → |n - 2304| ≥ |2310 - 2304| :=
by sorry

end nearest_multiple_of_21_to_2304_l3535_353570


namespace least_positive_integer_mod_l3535_353549

theorem least_positive_integer_mod (n : ℕ) : 
  ∃ x : ℕ, x > 0 ∧ (x + 7237 : ℤ) ≡ 5017 [ZMOD 12] ∧ 
  ∀ y : ℕ, y > 0 ∧ (y + 7237 : ℤ) ≡ 5017 [ZMOD 12] → x ≤ y :=
by
  use 12
  sorry

end least_positive_integer_mod_l3535_353549


namespace product_equation_minimum_sum_l3535_353579

theorem product_equation_minimum_sum (x y z a : ℤ) : 
  (x - 10) * (y - a) * (z - 2) = 1000 →
  x + y + z ≥ 7 →
  (∀ x' y' z' : ℤ, (x' - 10) * (y' - a) * (z' - 2) = 1000 → x' + y' + z' ≥ x + y + z) →
  a = 1 := by
  sorry

end product_equation_minimum_sum_l3535_353579


namespace tan_product_l3535_353594

theorem tan_product (α β : Real) 
  (h1 : Real.cos (α + β) = 1/5)
  (h2 : Real.cos (α - β) = 3/5) : 
  Real.tan α * Real.tan β = 1/2 := by
  sorry

end tan_product_l3535_353594


namespace point_on_x_axis_l3535_353589

/-- A point M with coordinates (a-2, a+1) lies on the x-axis if and only if its coordinates are (-3, 0) -/
theorem point_on_x_axis (a : ℝ) : 
  (a + 1 = 0 ∧ (a - 2, a + 1) = (-3, 0)) ↔ (a - 2, a + 1) = (-3, 0) :=
by sorry

end point_on_x_axis_l3535_353589


namespace no_divisible_by_three_for_all_x_l3535_353556

theorem no_divisible_by_three_for_all_x : ¬∃ (p q : ℤ), ∀ (x : ℤ), 3 ∣ (x^2 + p*x + q) := by
  sorry

end no_divisible_by_three_for_all_x_l3535_353556


namespace fraction_simplification_l3535_353574

theorem fraction_simplification :
  (3 : ℝ) / (2 * Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = (3 * Real.sqrt 2) / 38 := by
  sorry

end fraction_simplification_l3535_353574


namespace range_of_m_l3535_353591

def P (x : ℝ) : Prop := x^2 - 4*x - 12 ≤ 0

def Q (x m : ℝ) : Prop := |x - m| ≤ m^2

theorem range_of_m : 
  ∀ m : ℝ, (∀ x : ℝ, P x → Q x m) ∧ 
            (∃ x : ℝ, Q x m ∧ ¬P x) ∧ 
            (∃ x : ℝ, P x) 
  ↔ m ≤ -3 ∨ m > 2 := by sorry

end range_of_m_l3535_353591


namespace white_washing_cost_l3535_353559

/-- Calculate the cost of white washing a room with given dimensions and openings. -/
theorem white_washing_cost
  (room_length room_width room_height : ℝ)
  (door_width door_height : ℝ)
  (window_width window_height : ℝ)
  (num_windows : ℕ)
  (cost_per_sqft : ℝ)
  (h_room_length : room_length = 25)
  (h_room_width : room_width = 15)
  (h_room_height : room_height = 12)
  (h_door_width : door_width = 6)
  (h_door_height : door_height = 3)
  (h_window_width : window_width = 4)
  (h_window_height : window_height = 3)
  (h_num_windows : num_windows = 3)
  (h_cost_per_sqft : cost_per_sqft = 6) :
  let total_wall_area := 2 * (room_length + room_width) * room_height
  let door_area := door_width * door_height
  let window_area := window_width * window_height
  let total_opening_area := door_area + num_windows * window_area
  let paintable_area := total_wall_area - total_opening_area
  let total_cost := paintable_area * cost_per_sqft
  total_cost = 5436 := by sorry


end white_washing_cost_l3535_353559


namespace intersection_implies_a_nonpositive_l3535_353576

def A : Set ℝ := {x | x ≤ 0}
def B (a : ℝ) : Set ℝ := {1, 3, a}

theorem intersection_implies_a_nonpositive (a : ℝ) :
  (A ∩ B a).Nonempty → a ≤ 0 := by
  sorry

end intersection_implies_a_nonpositive_l3535_353576


namespace athlete_arrangements_l3535_353521

/-- The number of athletes and tracks -/
def n : ℕ := 6

/-- Function to calculate the number of arrangements where A, B, and C are not adjacent -/
def arrangements_not_adjacent : ℕ := sorry

/-- Function to calculate the number of arrangements where there is one person between A and B -/
def arrangements_one_between : ℕ := sorry

/-- Function to calculate the number of arrangements where A is not on first or second track, and B is on fifth or sixth track -/
def arrangements_restricted : ℕ := sorry

/-- Theorem stating the correct number of arrangements for each scenario -/
theorem athlete_arrangements :
  arrangements_not_adjacent = 144 ∧
  arrangements_one_between = 192 ∧
  arrangements_restricted = 144 :=
sorry

end athlete_arrangements_l3535_353521


namespace fibonacci_fraction_bound_l3535_353557

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_fraction_bound (a b n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : n ≥ 2) :
  ((fib n / fib (n - 1) < a / b ∧ a / b < fib (n + 1) / fib n) ∨
   (fib (n + 1) / fib n < a / b ∧ a / b < fib n / fib (n - 1))) →
  b ≥ fib (n + 1) :=
by sorry

end fibonacci_fraction_bound_l3535_353557


namespace p_sufficient_not_necessary_for_q_l3535_353533

theorem p_sufficient_not_necessary_for_q :
  (∃ x : ℝ, x = 2 ∧ x^2 ≠ 4) ∨
  (∃ x : ℝ, x^2 = 4 ∧ x ≠ 2) ∨
  (∀ x : ℝ, x = 2 → x^2 = 4) :=
by sorry

end p_sufficient_not_necessary_for_q_l3535_353533


namespace chip_price_reduction_l3535_353597

theorem chip_price_reduction (a b : ℝ) : 
  (∃ (price_after_first_reduction : ℝ), 
    price_after_first_reduction = a * (1 - 0.1) ∧
    b = price_after_first_reduction * (1 - 0.2)) →
  b = a * (1 - 0.1) * (1 - 0.2) := by
sorry

end chip_price_reduction_l3535_353597


namespace sarahs_sweaters_sarahs_sweaters_proof_l3535_353587

theorem sarahs_sweaters (machine_capacity : ℕ) (num_shirts : ℕ) (num_loads : ℕ) : ℕ :=
  let total_pieces := machine_capacity * num_loads
  let num_sweaters := total_pieces - num_shirts
  num_sweaters

theorem sarahs_sweaters_proof 
  (h1 : sarahs_sweaters 5 43 9 = 2) : sarahs_sweaters 5 43 9 = 2 := by
  sorry

end sarahs_sweaters_sarahs_sweaters_proof_l3535_353587


namespace students_with_one_fruit_l3535_353522

theorem students_with_one_fruit (total_apples : Nat) (total_bananas : Nat) (both_fruits : Nat) 
  (h1 : total_apples = 12)
  (h2 : total_bananas = 8)
  (h3 : both_fruits = 5) :
  (total_apples - both_fruits) + (total_bananas - both_fruits) = 10 := by
  sorry

end students_with_one_fruit_l3535_353522


namespace representation_of_231_l3535_353580

theorem representation_of_231 : 
  ∃ (list : List ℕ), (list.sum = 231) ∧ (list.prod = 231) := by
  sorry

end representation_of_231_l3535_353580


namespace square_sum_given_sum_and_product_l3535_353515

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 5) (h2 : x * y = 2) : x^2 + y^2 = 21 := by
  sorry

end square_sum_given_sum_and_product_l3535_353515


namespace min_value_theorem_l3535_353503

theorem min_value_theorem (x : ℝ) (h : x > -1) : 
  x + 4 / (x + 1) ≥ 3 ∧ ∃ y > -1, y + 4 / (y + 1) = 3 :=
sorry

end min_value_theorem_l3535_353503


namespace inverse_f_zero_solution_l3535_353519

noncomputable section

variables (a b c : ℝ)
variable (f : ℝ → ℝ)

-- Define the function f
def f_def : f = λ x => 1 / (a * x^2 + b * x + c) := by sorry

-- Conditions: a, b, and c are nonzero
axiom a_nonzero : a ≠ 0
axiom b_nonzero : b ≠ 0
axiom c_nonzero : c ≠ 0

-- Theorem: The only solution to f^(-1)(x) = 0 is x = 1/c
theorem inverse_f_zero_solution :
  ∀ x : ℝ, (Function.invFun f) x = 0 ↔ x = 1 / c := by sorry

end

end inverse_f_zero_solution_l3535_353519


namespace parabola_vertex_l3535_353563

/-- The parabola defined by y = 2(x+9)^2 - 3 has vertex at (-9, -3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 2 * (x + 9)^2 - 3 → (∃ a b : ℝ, (a, b) = (-9, -3) ∧ ∀ x, y ≥ 2 * (x + 9)^2 - 3) :=
by
  sorry

end parabola_vertex_l3535_353563


namespace sin_2023pi_over_3_l3535_353510

theorem sin_2023pi_over_3 : Real.sin (2023 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_2023pi_over_3_l3535_353510


namespace fib_like_seq_a7_l3535_353534

/-- An increasing sequence of positive integers satisfying the Fibonacci-like recurrence -/
def FibLikeSeq (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, n ≥ 1 → a (n + 2) = a (n + 1) + a n)

theorem fib_like_seq_a7 (a : ℕ → ℕ) (h : FibLikeSeq a) (h6 : a 6 = 50) : 
  a 7 = 83 := by
sorry

end fib_like_seq_a7_l3535_353534


namespace school_journey_time_l3535_353573

/-- Calculates the remaining time to reach the classroom given the total time available,
    time to reach the school gate, and time to reach the school building from the gate. -/
def remaining_time (total_time gate_time building_time : ℕ) : ℕ :=
  total_time - (gate_time + building_time)

/-- Proves that given 30 minutes total time, 15 minutes to reach the gate,
    and 6 minutes to reach the building, there are 9 minutes left to reach the room. -/
theorem school_journey_time : remaining_time 30 15 6 = 9 := by
  sorry

end school_journey_time_l3535_353573


namespace min_sum_squares_l3535_353507

theorem min_sum_squares (a b : ℝ) (ha : a ≠ 0) :
  (∃ x ∈ Set.Icc 3 4, (a + 2) / x = a * x + 2 * b + 1) →
  (∀ c d : ℝ, (∃ x ∈ Set.Icc 3 4, (c + 2) / x = c * x + 2 * d + 1) → c^2 + d^2 ≥ 1/100) ∧
  (∃ c d : ℝ, (∃ x ∈ Set.Icc 3 4, (c + 2) / x = c * x + 2 * d + 1) ∧ c^2 + d^2 = 1/100) :=
by sorry

end min_sum_squares_l3535_353507


namespace max_voters_with_95_percent_support_l3535_353520

/-- Represents the election scenario with an initial poll and subsequent groups -/
structure ElectionPoll where
  initial_voters : ℕ
  initial_support : ℕ
  group_size : ℕ
  group_support : ℕ

/-- Calculates the total number of voters and supporters for a given number of additional groups -/
def totalVoters (poll : ElectionPoll) (additional_groups : ℕ) : ℕ × ℕ :=
  (poll.initial_voters + poll.group_size * additional_groups,
   poll.initial_support + poll.group_support * additional_groups)

/-- Checks if the support percentage is at least 95% -/
def isSupportAboveThreshold (total : ℕ) (support : ℕ) : Prop :=
  (support : ℚ) / (total : ℚ) ≥ 95 / 100

/-- Theorem stating the maximum number of voters while maintaining 95% support -/
theorem max_voters_with_95_percent_support :
  ∃ (poll : ElectionPoll) (max_groups : ℕ),
    poll.initial_voters = 100 ∧
    poll.initial_support = 98 ∧
    poll.group_size = 10 ∧
    poll.group_support = 9 ∧
    (let (total, support) := totalVoters poll max_groups
     isSupportAboveThreshold total support) ∧
    (∀ g > max_groups,
      let (total, support) := totalVoters poll g
      ¬(isSupportAboveThreshold total support)) ∧
    poll.initial_voters + poll.group_size * max_groups = 160 :=
  sorry

end max_voters_with_95_percent_support_l3535_353520


namespace system_solution_existence_l3535_353592

theorem system_solution_existence (a : ℝ) : 
  (∃ x y : ℝ, y = (x + |x|) / x ∧ (x - a)^2 = y + a) ↔ 
  (a > -1 ∧ a ≤ 0) ∨ (a > 0 ∧ a < 1) ∨ (a ≥ 1 ∧ a ≤ 2) ∨ (a > 2) :=
by sorry

end system_solution_existence_l3535_353592


namespace two_removable_cells_exist_l3535_353501

-- Define a 4x4 grid
def Grid := Fin 4 → Fin 4 → Bool

-- Define a cell position
structure CellPosition where
  row : Fin 4
  col : Fin 4

-- Define a function to remove a cell from the grid
def removeCell (g : Grid) (pos : CellPosition) : Grid :=
  fun r c => if r = pos.row ∧ c = pos.col then false else g r c

-- Define congruence between two parts of the grid
def isCongruent (part1 part2 : Set CellPosition) : Prop := sorry

-- Define a function to check if a grid can be divided into three congruent parts
def canDivideIntoThreeCongruentParts (g : Grid) : Prop := sorry

-- Theorem statement
theorem two_removable_cells_exist :
  ∃ (pos1 pos2 : CellPosition),
    pos1 ≠ pos2 ∧
    canDivideIntoThreeCongruentParts (removeCell (fun _ _ => true) pos1) ∧
    canDivideIntoThreeCongruentParts (removeCell (fun _ _ => true) pos2) := by
  sorry


end two_removable_cells_exist_l3535_353501


namespace divisibility_properties_l3535_353581

theorem divisibility_properties (a : ℤ) : 
  (2 ∣ (a^2 - a)) ∧ (3 ∣ (a^3 - a)) := by sorry

end divisibility_properties_l3535_353581


namespace evaluate_expression_l3535_353513

theorem evaluate_expression : (20 ^ 40) / (80 ^ 10) = 5 ^ 10 := by
  sorry

end evaluate_expression_l3535_353513


namespace fraction_equality_sum_l3535_353509

theorem fraction_equality_sum (P Q : ℚ) : 
  (4 : ℚ) / 7 = P / 49 ∧ (4 : ℚ) / 7 = 84 / Q → P + Q = 175 := by
  sorry

end fraction_equality_sum_l3535_353509


namespace steves_speed_back_l3535_353582

/-- Proves that Steve's speed on the way back from work is 14 km/h given the conditions --/
theorem steves_speed_back (distance : ℝ) (total_time : ℝ) (speed_ratio : ℝ) : 
  distance = 28 → 
  total_time = 6 → 
  speed_ratio = 2 → 
  (distance / (distance / (2 * (distance / (total_time - distance / (2 * (distance / total_time)))))) = 14) := by
  sorry

end steves_speed_back_l3535_353582


namespace charley_pencils_loss_l3535_353554

theorem charley_pencils_loss (initial_pencils : ℕ) (lost_moving : ℕ) (current_pencils : ℕ)
  (h1 : initial_pencils = 30)
  (h2 : lost_moving = 6)
  (h3 : current_pencils = 16) :
  (initial_pencils - lost_moving - current_pencils : ℚ) / (initial_pencils - lost_moving : ℚ) = 1 / 3 :=
by sorry

end charley_pencils_loss_l3535_353554


namespace warehouse_bins_count_l3535_353578

/-- Calculates the total number of bins in a warehouse given specific conditions. -/
def totalBins (totalCapacity : ℕ) (twentyTonBins : ℕ) (twentyTonCapacity : ℕ) (fifteenTonCapacity : ℕ) : ℕ :=
  twentyTonBins + (totalCapacity - twentyTonBins * twentyTonCapacity) / fifteenTonCapacity

/-- Theorem stating that under given conditions, the total number of bins is 30. -/
theorem warehouse_bins_count :
  totalBins 510 12 20 15 = 30 := by
  sorry

end warehouse_bins_count_l3535_353578


namespace sum_of_coefficients_l3535_353532

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 128 := by
sorry

end sum_of_coefficients_l3535_353532


namespace exists_point_product_nonnegative_l3535_353588

theorem exists_point_product_nonnegative 
  (f : ℝ → ℝ) 
  (hf : ContDiff ℝ 3 f) : 
  ∃ a : ℝ, f a * (deriv f a) * (deriv^[2] f a) * (deriv^[3] f a) ≥ 0 := by
  sorry

end exists_point_product_nonnegative_l3535_353588


namespace paula_candy_distribution_l3535_353565

theorem paula_candy_distribution (initial_candies : ℕ) (additional_candies : ℕ) (num_friends : ℕ) :
  initial_candies = 20 →
  additional_candies = 4 →
  num_friends = 6 →
  (initial_candies + additional_candies) / num_friends = 4 :=
by sorry

end paula_candy_distribution_l3535_353565


namespace average_licks_to_center_l3535_353550

def dan_licks : ℕ := 58
def michael_licks : ℕ := 63
def sam_licks : ℕ := 70
def david_licks : ℕ := 70
def lance_licks : ℕ := 39

def total_licks : ℕ := dan_licks + michael_licks + sam_licks + david_licks + lance_licks
def num_people : ℕ := 5

theorem average_licks_to_center (h : total_licks = dan_licks + michael_licks + sam_licks + david_licks + lance_licks) :
  (total_licks : ℚ) / num_people = 60 := by
  sorry

end average_licks_to_center_l3535_353550


namespace sine_monotonicity_implies_omega_range_l3535_353505

open Real

theorem sine_monotonicity_implies_omega_range 
  (f : ℝ → ℝ) (ω : ℝ) (h_pos : ω > 0) :
  (∀ x ∈ Set.Ioo (π/2) π, 
    ∀ y ∈ Set.Ioo (π/2) π, 
    x < y → f x < f y) →
  (∀ x, f x = 2 * sin (ω * x + π/6)) →
  0 < ω ∧ ω ≤ 1/3 := by
sorry

end sine_monotonicity_implies_omega_range_l3535_353505


namespace odometer_difference_l3535_353561

theorem odometer_difference (initial_reading final_reading : ℝ) 
  (h1 : initial_reading = 212.3)
  (h2 : final_reading = 584.3) :
  final_reading - initial_reading = 372 := by
  sorry

end odometer_difference_l3535_353561


namespace total_books_eq_sum_l3535_353506

/-- The number of different books in the 'crazy silly school' series -/
def total_books : ℕ := sorry

/-- The number of different movies in the 'crazy silly school' series -/
def total_movies : ℕ := 10

/-- The number of books you have read -/
def books_read : ℕ := 12

/-- The number of movies you have watched -/
def movies_watched : ℕ := 56

/-- The number of books you still have to read -/
def books_to_read : ℕ := 10

/-- Theorem: The total number of books in the series is equal to the sum of books read and books yet to read -/
theorem total_books_eq_sum : total_books = books_read + books_to_read := by sorry

end total_books_eq_sum_l3535_353506


namespace perpendicular_feet_circle_area_l3535_353567

/-- Given two points in the plane, calculate the area of the circle described by the perpendicular
    feet and find its floor. -/
theorem perpendicular_feet_circle_area (B C : ℝ × ℝ) (h_B : B = (20, 14)) (h_C : C = (18, 0)) :
  let midpoint := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let radius := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) / 2
  let area := π * radius^2
  area = 50 * π ∧ Int.floor area = 157 := by sorry

end perpendicular_feet_circle_area_l3535_353567


namespace stamp_problem_solution_l3535_353500

/-- The cost of a first-class postage stamp in pence -/
def first_class_cost : ℕ := 85

/-- The cost of a second-class postage stamp in pence -/
def second_class_cost : ℕ := 66

/-- The number of pence in a pound -/
def pence_per_pound : ℕ := 100

/-- The proposition that (r, s) is a valid solution to the stamp problem -/
def is_valid_solution (r s : ℕ) : Prop :=
  r ≥ 1 ∧ s ≥ 1 ∧ ∃ t : ℕ, t > 0 ∧ first_class_cost * r + second_class_cost * s = pence_per_pound * t

/-- The proposition that (r, s) is the optimal solution to the stamp problem -/
def is_optimal_solution (r s : ℕ) : Prop :=
  is_valid_solution r s ∧ ∀ r' s' : ℕ, is_valid_solution r' s' → r + s ≤ r' + s'

/-- The theorem stating the optimal solution to the stamp problem -/
theorem stamp_problem_solution :
  is_optimal_solution 2 5 ∧ 2 + 5 = 7 := by sorry

end stamp_problem_solution_l3535_353500


namespace ratio_of_sum_to_difference_l3535_353524

theorem ratio_of_sum_to_difference (a b : ℝ) : 
  0 < b ∧ b < a ∧ a + b = 7 * (a - b) → a / b = 4 / 3 := by
  sorry

end ratio_of_sum_to_difference_l3535_353524


namespace rotation_equivalence_l3535_353552

/-- Given two rotations about the same point Q:
    1. A 735-degree clockwise rotation of point P to point R
    2. A y-degree counterclockwise rotation of point P to the same point R
    where y < 360, prove that y = 345 degrees. -/
theorem rotation_equivalence (y : ℝ) (h1 : y < 360) : 
  (735 % 360 : ℝ) + y = 360 → y = 345 := by sorry

end rotation_equivalence_l3535_353552


namespace smallest_two_digit_with_product_12_l3535_353566

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → 26 ≤ n :=
sorry

end smallest_two_digit_with_product_12_l3535_353566


namespace cos_210_degrees_l3535_353538

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_degrees_l3535_353538


namespace correct_percentage_l3535_353571

theorem correct_percentage (x : ℕ) : 
  let total := 6 * x
  let missed := 2 * x
  let correct := total - missed
  (correct : ℚ) / total * 100 = 200 / 3 := by sorry

end correct_percentage_l3535_353571


namespace mixture_ratio_l3535_353516

def mixture (initial_water : ℝ) : Prop :=
  let initial_alcohol : ℝ := 10
  let added_water : ℝ := 10
  let new_ratio_alcohol : ℝ := 2
  let new_ratio_water : ℝ := 7
  (initial_alcohol / (initial_water + added_water) = new_ratio_alcohol / new_ratio_water) ∧
  (initial_alcohol / initial_water = 2 / 5)

theorem mixture_ratio : ∃ (initial_water : ℝ), mixture initial_water :=
  sorry

end mixture_ratio_l3535_353516


namespace no_valid_n_l3535_353528

theorem no_valid_n : ¬∃ (n : ℕ), n > 0 ∧ 
  (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ 
  (100 ≤ 4 * n ∧ 4 * n ≤ 999) := by
  sorry

end no_valid_n_l3535_353528


namespace max_reciprocal_sum_l3535_353523

theorem max_reciprocal_sum (t q u₁ u₂ : ℝ) : 
  (u₁ * u₂ = q) →
  (u₁ + u₂ = t) →
  (u₁ + u₂ = u₁^2 + u₂^2) →
  (u₁ + u₂ = u₁^4 + u₂^4) →
  (∃ (x : ℝ), x^2 - t*x + q = 0) →
  (∀ (v₁ v₂ : ℝ), v₁ * v₂ = q ∧ v₁ + v₂ = t ∧ v₁ + v₂ = v₁^2 + v₂^2 ∧ v₁ + v₂ = v₁^4 + v₂^4 →
    1/u₁^2009 + 1/u₂^2009 ≥ 1/v₁^2009 + 1/v₂^2009) →
  1/u₁^2009 + 1/u₂^2009 = 2 :=
by sorry

end max_reciprocal_sum_l3535_353523


namespace final_combined_price_theorem_l3535_353596

/-- Calculates the final price of an item after applying discount and tax --/
def finalPrice (initialPrice : ℝ) (discount : ℝ) (tax : ℝ) : ℝ :=
  initialPrice * (1 - discount) * (1 + tax)

/-- Calculates the price of an accessory after applying tax --/
def accessoryPrice (price : ℝ) (tax : ℝ) : ℝ :=
  price * (1 + tax)

/-- Theorem stating the final combined price of iPhone and accessories --/
theorem final_combined_price_theorem 
  (iPhoneInitialPrice : ℝ) 
  (iPhoneDiscount1 iPhoneDiscount2 : ℝ)
  (iPhoneTax1 iPhoneTax2 : ℝ)
  (screenProtectorPrice casePrice : ℝ)
  (accessoriesTax : ℝ)
  (h1 : iPhoneInitialPrice = 1000)
  (h2 : iPhoneDiscount1 = 0.1)
  (h3 : iPhoneDiscount2 = 0.2)
  (h4 : iPhoneTax1 = 0.08)
  (h5 : iPhoneTax2 = 0.06)
  (h6 : screenProtectorPrice = 30)
  (h7 : casePrice = 50)
  (h8 : accessoriesTax = 0.05) :
  let iPhoneFinalPrice := finalPrice (finalPrice iPhoneInitialPrice iPhoneDiscount1 iPhoneTax1) iPhoneDiscount2 iPhoneTax2
  let totalAccessoriesPrice := accessoryPrice screenProtectorPrice accessoriesTax + accessoryPrice casePrice accessoriesTax
  iPhoneFinalPrice + totalAccessoriesPrice = 908.256 := by
    sorry


end final_combined_price_theorem_l3535_353596


namespace number_problem_l3535_353586

theorem number_problem (x : ℝ) (h : x - 7 = 9) : 5 * x = 80 := by
  sorry

end number_problem_l3535_353586


namespace triangle_congruence_criteria_triangle_congruence_criteria_2_l3535_353547

-- Define the structure for a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define side lengths
def side_length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define angle measure
def angle_measure (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem triangle_congruence_criteria (ABC A'B'C' : Triangle) :
  (side_length ABC.A ABC.B = side_length A'B'C'.A A'B'C'.B ∧
   side_length ABC.B ABC.C = side_length A'B'C'.B A'B'C'.C ∧
   side_length ABC.A ABC.C = side_length A'B'C'.A A'B'C'.C) →
  congruent ABC A'B'C' :=
sorry

theorem triangle_congruence_criteria_2 (ABC A'B'C' : Triangle) :
  (side_length ABC.A ABC.B = side_length A'B'C'.A A'B'C'.B ∧
   angle_measure ABC.A ABC.B ABC.C = angle_measure A'B'C'.A A'B'C'.B A'B'C'.C ∧
   angle_measure ABC.B ABC.C ABC.A = angle_measure A'B'C'.B A'B'C'.C A'B'C'.A) →
  congruent ABC A'B'C' :=
sorry

end triangle_congruence_criteria_triangle_congruence_criteria_2_l3535_353547


namespace sum_is_six_digit_multiple_of_four_l3535_353564

def sum_of_numbers (A B : Nat) : Nat :=
  98765 + A * 1000 + 532 + B * 100 + 41 + 1021

theorem sum_is_six_digit_multiple_of_four (A B : Nat) 
  (h1 : 1 ≤ A ∧ A ≤ 9) (h2 : 1 ≤ B ∧ B ≤ 9) : 
  ∃ (n : Nat), sum_of_numbers A B = n ∧ 
  100000 ≤ n ∧ n < 1000000 ∧ 
  n % 4 = 0 :=
sorry

end sum_is_six_digit_multiple_of_four_l3535_353564


namespace least_largest_factor_l3535_353553

theorem least_largest_factor (a b c d e : ℕ+) : 
  a * b * c * d * e = 55 * 60 * 65 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e →
  (∀ x y z w v : ℕ+, 
    x * y * z * w * v = 55 * 60 * 65 ∧ 
    x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧ 
    y ≠ z ∧ y ≠ w ∧ y ≠ v ∧ 
    z ≠ w ∧ z ≠ v ∧ 
    w ≠ v →
    max a (max b (max c (max d e))) ≤ max x (max y (max z (max w v)))) →
  max a (max b (max c (max d e))) = 13 :=
by sorry

end least_largest_factor_l3535_353553


namespace divisor_between_l3535_353540

theorem divisor_between (n a b : ℕ) (hn : n > 8) (ha : 0 < a) (hb : 0 < b)
  (hab : a < b) (hdiv_a : a ∣ n) (hdiv_b : b ∣ n) (hneq : a ≠ b)
  (heq : n = a^2 + b) : 
  ∃ d : ℕ, d ∣ n ∧ a < d ∧ d < b := by
sorry

end divisor_between_l3535_353540


namespace george_money_left_l3535_353583

def monthly_income : ℕ := 240

def donation : ℕ := monthly_income / 2

def remaining_after_donation : ℕ := monthly_income - donation

def groceries_cost : ℕ := 20

def amount_left : ℕ := remaining_after_donation - groceries_cost

theorem george_money_left : amount_left = 100 := by
  sorry

end george_money_left_l3535_353583


namespace perpendicular_lines_plane_theorem_l3535_353517

/-- Represents a plane in 3D space -/
structure Plane :=
  (α : Type*)

/-- Represents a line in 3D space -/
structure Line :=
  (l : Type*)

/-- Indicates that a line is perpendicular to a plane -/
def perpendicular_to_plane (l : Line) (α : Plane) : Prop :=
  sorry

/-- Indicates that a line is perpendicular to another line -/
def perpendicular_to_line (l1 l2 : Line) : Prop :=
  sorry

/-- Indicates that a line is in a plane -/
def line_in_plane (l : Line) (α : Plane) : Prop :=
  sorry

/-- Indicates that a line is outside a plane -/
def line_outside_plane (l : Line) (α : Plane) : Prop :=
  sorry

theorem perpendicular_lines_plane_theorem 
  (α : Plane) (a b l : Line) 
  (h1 : a ≠ b)
  (h2 : line_in_plane a α)
  (h3 : line_in_plane b α)
  (h4 : line_outside_plane l α) :
  (∀ (α : Plane) (l : Line), perpendicular_to_plane l α → 
    perpendicular_to_line l a ∧ perpendicular_to_line l b) ∧
  (∃ (α : Plane) (a b l : Line), 
    perpendicular_to_line l a ∧ perpendicular_to_line l b ∧
    ¬perpendicular_to_plane l α) :=
sorry

end perpendicular_lines_plane_theorem_l3535_353517


namespace fixed_fee_calculation_l3535_353598

/-- Represents a cable service bill -/
structure CableBill where
  fixed_fee : ℝ
  hourly_rate : ℝ
  usage_hours : ℝ

/-- Calculates the total bill amount -/
def bill_amount (b : CableBill) : ℝ :=
  b.fixed_fee + b.hourly_rate * b.usage_hours

theorem fixed_fee_calculation 
  (feb : CableBill) (mar : CableBill) 
  (h_feb_amount : bill_amount feb = 20.72)
  (h_mar_amount : bill_amount mar = 35.28)
  (h_same_fee : feb.fixed_fee = mar.fixed_fee)
  (h_same_rate : feb.hourly_rate = mar.hourly_rate)
  (h_triple_usage : mar.usage_hours = 3 * feb.usage_hours) :
  feb.fixed_fee = 13.44 := by
sorry

end fixed_fee_calculation_l3535_353598


namespace intersection_locus_is_circle_l3535_353590

/-- The locus of points (x, y) satisfying both equations 2ux - 3y - 2u = 0 and x - 3uy + 2 = 0,
    where u is a real parameter, is a circle. -/
theorem intersection_locus_is_circle :
  ∀ (x y u : ℝ), (2 * u * x - 3 * y - 2 * u = 0) ∧ (x - 3 * u * y + 2 = 0) →
  ∃ (c : ℝ × ℝ) (r : ℝ), (x - c.1)^2 + (y - c.2)^2 = r^2 := by
  sorry

end intersection_locus_is_circle_l3535_353590


namespace total_students_is_150_l3535_353537

/-- Proves that the total number of students is 150 given the conditions -/
theorem total_students_is_150 
  (total : ℕ) 
  (boys : ℕ) 
  (girls : ℕ) 
  (h1 : total = boys + girls) 
  (h2 : boys = 60 → girls = (60 * total) / 100) : 
  total = 150 := by
  sorry

end total_students_is_150_l3535_353537


namespace triangle_problem_l3535_353536

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a + b = 5, c = √7, and 4sin²((A + B)/2) - cos(2C) = 7/2,
    then the measure of angle C is π/3 and the area of triangle ABC is 3√3/2 -/
theorem triangle_problem (a b c A B C : Real) : 
  a + b = 5 →
  c = Real.sqrt 7 →
  4 * Real.sin (A + B) ^ 2 / 4 - Real.cos (2 * C) = 7 / 2 →
  C = π / 3 ∧ 
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 := by
  sorry


end triangle_problem_l3535_353536


namespace geometric_figure_x_length_l3535_353558

theorem geometric_figure_x_length 
  (total_area : ℝ)
  (square1_side : ℝ → ℝ)
  (square2_side : ℝ → ℝ)
  (triangle_leg1 : ℝ → ℝ)
  (triangle_leg2 : ℝ → ℝ)
  (h1 : total_area = 1000)
  (h2 : ∀ x, square1_side x = 3 * x)
  (h3 : ∀ x, square2_side x = 4 * x)
  (h4 : ∀ x, triangle_leg1 x = 3 * x)
  (h5 : ∀ x, triangle_leg2 x = 4 * x)
  (h6 : ∀ x, (square1_side x)^2 + (square2_side x)^2 + 1/2 * (triangle_leg1 x) * (triangle_leg2 x) = total_area) :
  ∃ x : ℝ, x = 10 * Real.sqrt 31 / 31 := by
  sorry

end geometric_figure_x_length_l3535_353558


namespace complex_modulus_l3535_353502

theorem complex_modulus (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 26/5 := by
  sorry

end complex_modulus_l3535_353502


namespace negation_of_existence_cube_lt_pow_three_negation_l3535_353545

theorem negation_of_existence (p : ℕ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem cube_lt_pow_three_negation :
  (¬ ∃ x : ℕ, x^3 < 3^x) ↔ (∀ x : ℕ, x^3 ≥ 3^x) :=
by sorry

end negation_of_existence_cube_lt_pow_three_negation_l3535_353545


namespace no_solution_to_inequality_system_l3535_353542

theorem no_solution_to_inequality_system : 
  ¬ ∃ x : ℝ, (x - 3 ≥ 0) ∧ (2*x - 5 < 1) := by
sorry

end no_solution_to_inequality_system_l3535_353542


namespace phillips_jars_l3535_353525

-- Define the given quantities
def cucumbers : ℕ := 10
def initial_vinegar : ℕ := 100
def pickles_per_cucumber : ℕ := 6
def pickles_per_jar : ℕ := 12
def vinegar_per_jar : ℕ := 10
def remaining_vinegar : ℕ := 60

-- Define the function to calculate the number of jars
def number_of_jars : ℕ :=
  min
    (cucumbers * pickles_per_cucumber / pickles_per_jar)
    ((initial_vinegar - remaining_vinegar) / vinegar_per_jar)

-- Theorem statement
theorem phillips_jars :
  number_of_jars = 4 :=
sorry

end phillips_jars_l3535_353525


namespace abs_not_always_positive_l3535_353544

theorem abs_not_always_positive : ¬ (∀ x : ℝ, |x| > 0) := by
  sorry

end abs_not_always_positive_l3535_353544


namespace robotics_club_neither_l3535_353529

/-- The number of students in the robotics club who take neither computer science nor electronics -/
theorem robotics_club_neither (total : ℕ) (cs : ℕ) (elec : ℕ) (both : ℕ) 
  (h_total : total = 60)
  (h_cs : cs = 42)
  (h_elec : elec = 35)
  (h_both : both = 25) :
  total - (cs + elec - both) = 8 := by
  sorry

end robotics_club_neither_l3535_353529


namespace l_shaped_room_flooring_cost_l3535_353595

/-- Represents the dimensions of a rectangular room section -/
structure RoomSection where
  length : ℝ
  width : ℝ

/-- Calculates the total cost of replacing flooring in an L-shaped room -/
def total_flooring_cost (section1 section2 : RoomSection) (removal_cost per_sqft_cost : ℝ) : ℝ :=
  let total_area := section1.length * section1.width + section2.length * section2.width
  removal_cost + total_area * per_sqft_cost

/-- Theorem: The total cost to replace the floor in the given L-shaped room is $150 -/
theorem l_shaped_room_flooring_cost :
  let section1 : RoomSection := ⟨8, 7⟩
  let section2 : RoomSection := ⟨6, 4⟩
  let removal_cost : ℝ := 50
  let per_sqft_cost : ℝ := 1.25
  total_flooring_cost section1 section2 removal_cost per_sqft_cost = 150 := by
  sorry

end l_shaped_room_flooring_cost_l3535_353595


namespace max_area_is_12_l3535_353555

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the conditions of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let dist := λ p1 p2 : ℝ × ℝ => Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  dist q.A q.B = 5 ∧
  dist q.B q.C = 5 ∧
  dist q.C q.D = 5 ∧
  dist q.D q.A = 3

-- Define the deformation that maximizes ∠ABC
def max_angle_deformation (q : Quadrilateral) : Quadrilateral :=
  sorry

-- Define the area calculation function
def area (q : Quadrilateral) : ℝ :=
  sorry

-- Theorem statement
theorem max_area_is_12 (q : Quadrilateral) (h : is_valid_quadrilateral q) :
  area (max_angle_deformation q) = 12 :=
sorry

end max_area_is_12_l3535_353555


namespace value_of_3a_plus_6b_l3535_353575

theorem value_of_3a_plus_6b (a b : ℝ) (h : a + 2*b - 1 = 0) : 3*a + 6*b = 3 := by
  sorry

end value_of_3a_plus_6b_l3535_353575


namespace deposit_exceeds_target_on_saturday_l3535_353577

def initial_deposit : ℕ := 2
def multiplication_factor : ℕ := 3
def target_amount : ℕ := 500 * 100  -- Convert $500 to cents

def deposit_on_day (n : ℕ) : ℕ :=
  initial_deposit * multiplication_factor ^ n

def total_deposit (n : ℕ) : ℕ :=
  (List.range (n + 1)).map deposit_on_day |>.sum

def days_of_week := ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

theorem deposit_exceeds_target_on_saturday :
  (total_deposit 5 ≤ target_amount) ∧ 
  (total_deposit 6 > target_amount) ∧
  (days_of_week[(6 : ℕ) % 7] = "Saturday") := by
  sorry

end deposit_exceeds_target_on_saturday_l3535_353577


namespace solution_set_range_l3535_353508

theorem solution_set_range (t : ℝ) : 
  let A := {x : ℝ | x^2 - 4*x + t ≤ 0}
  (∃ x ∈ Set.Iic t, x ∈ A) → t ∈ Set.Icc 0 4 :=
by sorry

end solution_set_range_l3535_353508


namespace compound_molecular_weight_l3535_353531

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecularWeight (carbon_atoms : ℕ) (hydrogen_atoms : ℕ) (oxygen_atoms : ℕ) 
                    (carbon_weight : ℝ) (hydrogen_weight : ℝ) (oxygen_weight : ℝ) : ℝ :=
  (carbon_atoms : ℝ) * carbon_weight + (hydrogen_atoms : ℝ) * hydrogen_weight + (oxygen_atoms : ℝ) * oxygen_weight

/-- The molecular weight of a compound with 3 Carbon, 6 Hydrogen, and 1 Oxygen is approximately 58.078 g/mol -/
theorem compound_molecular_weight :
  let carbon_atoms : ℕ := 3
  let hydrogen_atoms : ℕ := 6
  let oxygen_atoms : ℕ := 1
  let carbon_weight : ℝ := 12.01
  let hydrogen_weight : ℝ := 1.008
  let oxygen_weight : ℝ := 16.00
  ∃ ε > 0, |molecularWeight carbon_atoms hydrogen_atoms oxygen_atoms 
                            carbon_weight hydrogen_weight oxygen_weight - 58.078| < ε :=
by
  sorry

end compound_molecular_weight_l3535_353531


namespace min_product_of_three_numbers_l3535_353568

theorem min_product_of_three_numbers (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1)
  (ordered : x ≤ y ∧ y ≤ z)
  (max_twice_min : z ≤ 2 * x) :
  x * y * z ≥ 1 / 32 ∧ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b + c = 1 ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 2 * a ∧ a * b * c = 1 / 32 := by
  sorry

end min_product_of_three_numbers_l3535_353568


namespace linear_regression_equation_l3535_353599

/-- Represents a linear regression model --/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Checks if two variables are positively correlated --/
def positively_correlated (x y : ℝ → ℝ) : Prop := sorry

/-- Calculates the sample mean of a variable --/
def sample_mean (x : ℝ → ℝ) : ℝ := sorry

/-- Checks if a point lies on the regression line --/
def point_on_line (model : LinearRegression) (x y : ℝ) : Prop :=
  y = model.slope * x + model.intercept

theorem linear_regression_equation 
  (x y : ℝ → ℝ) 
  (h_corr : positively_correlated x y)
  (h_mean_x : sample_mean x = 2)
  (h_mean_y : sample_mean y = 3)
  : ∃ (model : LinearRegression), 
    model.slope = 2 ∧ 
    model.intercept = -1 ∧ 
    point_on_line model 2 3 := by
  sorry

end linear_regression_equation_l3535_353599


namespace common_divisors_9240_10010_l3535_353548

theorem common_divisors_9240_10010 : 
  (Finset.filter (fun d => d ∣ 9240 ∧ d ∣ 10010) (Finset.range 10011)).card = 32 := by
  sorry

end common_divisors_9240_10010_l3535_353548


namespace max_value_x_cubed_over_y_fourth_l3535_353504

theorem max_value_x_cubed_over_y_fourth (x y : ℝ) 
  (h1 : 3 ≤ x * y^2) (h2 : x * y^2 ≤ 8) 
  (h3 : 4 ≤ x^2 / y) (h4 : x^2 / y ≤ 9) : 
  (∃ (a b : ℝ), 3 ≤ a * b^2 ∧ a * b^2 ≤ 8 ∧ 4 ≤ a^2 / b ∧ a^2 / b ≤ 9 ∧ a^3 / b^4 = 27) ∧ 
  (∀ (z w : ℝ), 3 ≤ z * w^2 → z * w^2 ≤ 8 → 4 ≤ z^2 / w → z^2 / w ≤ 9 → z^3 / w^4 ≤ 27) :=
sorry

end max_value_x_cubed_over_y_fourth_l3535_353504


namespace complex_simplification_l3535_353560

/-- The imaginary unit -/
axiom I : ℂ

/-- The property of the imaginary unit -/
axiom I_squared : I^2 = -1

/-- Theorem stating the equality of the complex expressions -/
theorem complex_simplification : 7 * (4 - 2*I) - 2*I * (7 - 3*I) = 22 - 28*I := by
  sorry

end complex_simplification_l3535_353560


namespace sum_of_consecutive_odd_primes_has_at_least_four_divisors_l3535_353585

/-- Two natural numbers are consecutive primes if they are both prime and there is no prime between them. -/
def ConsecutivePrimes (p q : ℕ) : Prop :=
  Prime p ∧ Prime q ∧ p < q ∧ ∀ k, p < k → k < q → ¬ Prime k

/-- The number of positive divisors of a natural number n. -/
def numPositiveDivisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range n.succ)).card

theorem sum_of_consecutive_odd_primes_has_at_least_four_divisors
  (p q : ℕ) (h : ConsecutivePrimes p q) :
  4 ≤ numPositiveDivisors (p + q) := by
  sorry

end sum_of_consecutive_odd_primes_has_at_least_four_divisors_l3535_353585


namespace repunit_existence_l3535_353569

theorem repunit_existence (p : Nat) (h_prime : Nat.Prime p) (h_p_gt_11 : p > 11) :
  ∃ k : Nat, ∃ n : Nat, p * k = (10^n - 1) / 9 := by
  sorry

end repunit_existence_l3535_353569


namespace smallest_sum_proof_l3535_353526

/-- Q(N, k) represents the probability that no blue ball is adjacent to the red ball -/
def Q (N k : ℕ) : ℚ := (N + 1 : ℚ) / (N + k + 1 : ℚ)

/-- The smallest sum of N and k satisfying the conditions -/
def smallest_sum : ℕ := 4

theorem smallest_sum_proof :
  ∀ N k : ℕ,
    (N + k) % 4 = 0 →
    Q N k < 7/9 →
    N + k ≥ smallest_sum :=
by sorry

end smallest_sum_proof_l3535_353526


namespace petr_speed_l3535_353530

theorem petr_speed (total_distance : ℝ) (ivan_speed : ℝ) (remaining_distance : ℝ) (time : ℝ) :
  total_distance = 153 →
  ivan_speed = 46 →
  remaining_distance = 24 →
  time = 1.5 →
  ∃ petr_speed : ℝ,
    petr_speed = 40 ∧
    total_distance - remaining_distance = (ivan_speed + petr_speed) * time :=
by sorry

end petr_speed_l3535_353530


namespace remaining_eggs_l3535_353551

theorem remaining_eggs (initial_eggs : ℕ) (morning_eaten : ℕ) (afternoon_eaten : ℕ) :
  initial_eggs = 20 → morning_eaten = 4 → afternoon_eaten = 3 →
  initial_eggs - (morning_eaten + afternoon_eaten) = 13 := by
  sorry

end remaining_eggs_l3535_353551


namespace simplify_expression_l3535_353527

theorem simplify_expression (m n : ℤ) (h : m * n = m + 3) :
  2 * m * n + 3 * m - 5 * m * n - 10 = -19 := by
  sorry

end simplify_expression_l3535_353527


namespace popcorn_distribution_l3535_353518

/-- Given the conditions of the popcorn problem, prove that each of Jared's friends can eat 60 pieces of popcorn. -/
theorem popcorn_distribution (pieces_per_serving : ℕ) (jared_pieces : ℕ) (num_friends : ℕ) (total_servings : ℕ)
  (h1 : pieces_per_serving = 30)
  (h2 : jared_pieces = 90)
  (h3 : num_friends = 3)
  (h4 : total_servings = 9) :
  (total_servings * pieces_per_serving - jared_pieces) / num_friends = 60 :=
by sorry

end popcorn_distribution_l3535_353518


namespace triangle_problem_l3535_353546

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  2 * a * Real.sin A = (2 * b + c) * Real.sin B + (2 * c + b) * Real.sin C →
  a = 7 →
  a * (15 * Real.sqrt 3 / 14) / 2 = b * c * Real.sin A →
  (A = 2 * π / 3 ∧ ((b = 3 ∧ c = 5) ∨ (b = 5 ∧ c = 3))) :=
by sorry


end triangle_problem_l3535_353546


namespace power_of_power_three_cubed_fourth_l3535_353535

theorem power_of_power_three_cubed_fourth : (3^3)^4 = 531441 := by
  sorry

end power_of_power_three_cubed_fourth_l3535_353535


namespace hotel_rooms_l3535_353562

theorem hotel_rooms (total_rooms : ℕ) (single_cost double_cost : ℚ) (total_revenue : ℚ) :
  total_rooms = 260 ∧
  single_cost = 35 ∧
  double_cost = 60 ∧
  total_revenue = 14000 →
  ∃ (single_rooms double_rooms : ℕ),
    single_rooms + double_rooms = total_rooms ∧
    single_cost * single_rooms + double_cost * double_rooms = total_revenue ∧
    single_rooms = 64 := by
  sorry

end hotel_rooms_l3535_353562


namespace angle_DAE_measure_l3535_353539

-- Define the points
variable (A B C D E F : Point)

-- Define the shapes
def is_equilateral_triangle (A B C : Point) : Prop := sorry

def is_regular_pentagon (B C D E F : Point) : Prop := sorry

-- Define the shared side
def share_side (A B C D E F : Point) : Prop := sorry

-- Define the angle measurement
def angle_measure (A D E : Point) : ℝ := sorry

-- Theorem statement
theorem angle_DAE_measure 
  (h1 : is_equilateral_triangle A B C) 
  (h2 : is_regular_pentagon B C D E F) 
  (h3 : share_side A B C D E F) : 
  angle_measure A D E = 108 := by sorry

end angle_DAE_measure_l3535_353539


namespace student_count_prove_student_count_l3535_353543

theorem student_count (weight_difference : ℝ) (average_decrease : ℝ) : ℝ :=
  weight_difference / average_decrease

theorem prove_student_count :
  let weight_difference : ℝ := 120 - 60
  let average_decrease : ℝ := 6
  student_count weight_difference average_decrease = 10 := by
    sorry

end student_count_prove_student_count_l3535_353543


namespace sandy_fish_count_l3535_353572

theorem sandy_fish_count (initial_fish : ℕ) (bought_fish : ℕ) : 
  initial_fish = 26 → bought_fish = 6 → initial_fish + bought_fish = 32 := by
  sorry

end sandy_fish_count_l3535_353572


namespace remainder_equality_l3535_353512

theorem remainder_equality (Q Q' E S S' s s' : ℕ) 
  (hQ : Q > Q') 
  (hS : S = Q % E) 
  (hS' : S' = Q' % E) 
  (hs : s = (Q^2 * Q') % E) 
  (hs' : s' = (S^2 * S') % E) : 
  s = s' := by
  sorry

end remainder_equality_l3535_353512


namespace min_abs_z_minus_one_l3535_353541

/-- For any complex number Z satisfying |Z-1| = |Z+1|, the minimum value of |Z-1| is 1. -/
theorem min_abs_z_minus_one (Z : ℂ) (h : Complex.abs (Z - 1) = Complex.abs (Z + 1)) :
  ∃ (min : ℝ), min = 1 ∧ ∀ (W : ℂ), Complex.abs (W - 1) = Complex.abs (W + 1) → Complex.abs (W - 1) ≥ min :=
by sorry

end min_abs_z_minus_one_l3535_353541


namespace luna_budget_theorem_l3535_353584

/-- Luna's monthly budget problem --/
theorem luna_budget_theorem 
  (house_rental : ℝ) 
  (food : ℝ) 
  (phone : ℝ) 
  (h1 : food = 0.6 * house_rental) 
  (h2 : house_rental + food = 240) 
  (h3 : house_rental + food + phone = 249) :
  phone = 0.1 * food := by
  sorry

end luna_budget_theorem_l3535_353584


namespace problem_1_l3535_353511

theorem problem_1 : Real.sqrt 3 ^ 2 + |-(Real.sqrt 3 / 3)| - (π - Real.sqrt 2) ^ 0 - Real.tan (π / 6) = 2 := by
  sorry

end problem_1_l3535_353511


namespace f_range_l3535_353514

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- State the theorem
theorem f_range :
  ∀ x ∈ Set.Icc (-2 : ℝ) 1,
    -1 ≤ f x ∧ f x ≤ 3 ∧
    (∃ x₁ ∈ Set.Icc (-2 : ℝ) 1, f x₁ = -1) ∧
    (∃ x₂ ∈ Set.Icc (-2 : ℝ) 1, f x₂ = 3) :=
by sorry

end f_range_l3535_353514
