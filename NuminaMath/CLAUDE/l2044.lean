import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2044_204420

theorem solution_set_equivalence : 
  {x : ℝ | (x + 3)^2 < 1} = {x : ℝ | -4 < x ∧ x < -2} := by
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2044_204420


namespace NUMINAMATH_CALUDE_green_hats_count_l2044_204430

theorem green_hats_count (total_hats : ℕ) (blue_cost green_cost total_cost : ℕ) 
  (h1 : total_hats = 85)
  (h2 : blue_cost = 6)
  (h3 : green_cost = 7)
  (h4 : total_cost = 550) :
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_cost * blue_hats + green_cost * green_hats = total_cost ∧
    green_hats = 40 :=
by sorry

end NUMINAMATH_CALUDE_green_hats_count_l2044_204430


namespace NUMINAMATH_CALUDE_binary_predecessor_and_successor_l2044_204449

def binary_number : ℕ := 84  -- 1010100₂ in decimal

theorem binary_predecessor_and_successor :
  (binary_number - 1 = 83) ∧ (binary_number + 1 = 85) := by
  sorry

-- Helper function to convert decimal to binary string (for reference)
def to_binary (n : ℕ) : String :=
  if n = 0 then "0"
  else
    let rec aux (m : ℕ) (acc : String) : String :=
      if m = 0 then acc
      else aux (m / 2) (toString (m % 2) ++ acc)
    aux n ""

-- These computations are to verify the binary representations
#eval to_binary binary_number        -- Should output "1010100"
#eval to_binary (binary_number - 1)  -- Should output "1010011"
#eval to_binary (binary_number + 1)  -- Should output "1010101"

end NUMINAMATH_CALUDE_binary_predecessor_and_successor_l2044_204449


namespace NUMINAMATH_CALUDE_gcd_6724_13104_l2044_204446

theorem gcd_6724_13104 : Nat.gcd 6724 13104 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_6724_13104_l2044_204446


namespace NUMINAMATH_CALUDE_complex_expression_equals_minus_half_minus_half_i_l2044_204448

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- The complex number (1+i)^2 / (1-i)^3 -/
noncomputable def complex_expression : ℂ := (1 + i)^2 / (1 - i)^3

/-- Theorem stating that the complex expression equals -1/2 - 1/2i -/
theorem complex_expression_equals_minus_half_minus_half_i :
  complex_expression = -1/2 - 1/2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equals_minus_half_minus_half_i_l2044_204448


namespace NUMINAMATH_CALUDE_discount_calculation_l2044_204467

/-- Calculates the total discount percentage given initial, member, and special promotion discounts -/
def total_discount (initial_discount : ℝ) (member_discount : ℝ) (special_discount : ℝ) : ℝ :=
  let remaining_after_initial := 1 - initial_discount
  let remaining_after_member := remaining_after_initial * (1 - member_discount)
  let final_remaining := remaining_after_member * (1 - special_discount)
  (1 - final_remaining) * 100

/-- Theorem stating that the total discount is 65.8% given the specific discounts -/
theorem discount_calculation :
  total_discount 0.6 0.1 0.05 = 65.8 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l2044_204467


namespace NUMINAMATH_CALUDE_games_for_512_players_l2044_204421

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  num_players : ℕ
  num_players_pos : 0 < num_players

/-- The number of games needed to determine the champion in a single-elimination tournament -/
def games_to_champion (t : SingleEliminationTournament) : ℕ :=
  t.num_players - 1

/-- Theorem: In a single-elimination tournament with 512 players, 511 games are needed to determine the champion -/
theorem games_for_512_players :
  let t : SingleEliminationTournament := ⟨512, by norm_num⟩
  games_to_champion t = 511 := by
  sorry

end NUMINAMATH_CALUDE_games_for_512_players_l2044_204421


namespace NUMINAMATH_CALUDE_girls_in_college_l2044_204465

theorem girls_in_college (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 312 →
  boys + girls = total →
  8 * girls = 5 * boys →
  girls = 120 := by
sorry

end NUMINAMATH_CALUDE_girls_in_college_l2044_204465


namespace NUMINAMATH_CALUDE_trig_inequalities_l2044_204484

theorem trig_inequalities :
  (Real.cos (3 * Real.pi / 5) > Real.cos (-4 * Real.pi / 5)) ∧
  (Real.sin (Real.pi / 10) < Real.cos (Real.pi / 10)) := by
  sorry

end NUMINAMATH_CALUDE_trig_inequalities_l2044_204484


namespace NUMINAMATH_CALUDE_distance_between_points_l2044_204493

theorem distance_between_points (x₁ x₂ y₁ y₂ : ℝ) :
  x₁^2 + y₁^2 = 29 →
  x₂^2 + y₂^2 = 29 →
  x₁ + y₁ = 11 →
  x₂ + y₂ = 11 →
  x₁ ≠ x₂ →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2044_204493


namespace NUMINAMATH_CALUDE_point_b_value_l2044_204474

/-- Given a point A representing 3 on the number line, moving 3 units from A to reach point B 
    results in B representing either 0 or 6. -/
theorem point_b_value (A B : ℝ) : 
  A = 3 → (B - A = 3 ∨ A - B = 3) → (B = 0 ∨ B = 6) := by
  sorry

end NUMINAMATH_CALUDE_point_b_value_l2044_204474


namespace NUMINAMATH_CALUDE_inequality_proof_l2044_204415

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  1 / Real.sqrt (x + y) + 1 / Real.sqrt (y + z) + 1 / Real.sqrt (z + x) 
  ≤ 1 / Real.sqrt (2 * x * y * z) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2044_204415


namespace NUMINAMATH_CALUDE_ordering_of_powers_l2044_204479

theorem ordering_of_powers : 3^15 < 4^12 ∧ 4^12 < 8^9 := by
  sorry

end NUMINAMATH_CALUDE_ordering_of_powers_l2044_204479


namespace NUMINAMATH_CALUDE_average_minutes_run_is_112_div_9_l2044_204478

/-- The average number of minutes run per day by all students in an elementary school -/
def average_minutes_run (third_grade_minutes fourth_grade_minutes fifth_grade_minutes : ℕ)
  (third_to_fourth_ratio fourth_to_fifth_ratio : ℕ) : ℚ :=
  let fifth_graders := 1
  let fourth_graders := fourth_to_fifth_ratio * fifth_graders
  let third_graders := third_to_fourth_ratio * fourth_graders
  let total_students := third_graders + fourth_graders + fifth_graders
  let total_minutes := third_grade_minutes * third_graders + 
                       fourth_grade_minutes * fourth_graders + 
                       fifth_grade_minutes * fifth_graders
  (total_minutes : ℚ) / total_students

theorem average_minutes_run_is_112_div_9 :
  average_minutes_run 10 18 16 3 2 = 112 / 9 := by
  sorry

end NUMINAMATH_CALUDE_average_minutes_run_is_112_div_9_l2044_204478


namespace NUMINAMATH_CALUDE_subtracted_value_l2044_204401

theorem subtracted_value (N V : ℝ) (h1 : N = 740) (h2 : N / 4 - V = 10) : V = 175 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l2044_204401


namespace NUMINAMATH_CALUDE_extra_digit_sum_l2044_204436

theorem extra_digit_sum (x y : ℕ) (a : Fin 10) :
  x + y = 23456 →
  (10 * x + a.val) + y = 55555 →
  a.val = 5 :=
by sorry

end NUMINAMATH_CALUDE_extra_digit_sum_l2044_204436


namespace NUMINAMATH_CALUDE_counseling_rooms_count_l2044_204473

theorem counseling_rooms_count :
  ∃ (x : ℕ) (total_students : ℕ),
    (total_students = 20 * x + 32) ∧
    (total_students = 24 * (x - 1)) ∧
    (x = 14) := by
  sorry

end NUMINAMATH_CALUDE_counseling_rooms_count_l2044_204473


namespace NUMINAMATH_CALUDE_final_alcohol_percentage_l2044_204408

/-- Calculates the final alcohol percentage after adding pure alcohol to a solution -/
theorem final_alcohol_percentage
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (added_alcohol : ℝ)
  (h_initial_volume : initial_volume = 6)
  (h_initial_percentage : initial_percentage = 0.25)
  (h_added_alcohol : added_alcohol = 3) :
  let initial_alcohol := initial_volume * initial_percentage
  let total_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol
  let final_percentage := total_alcohol / final_volume
  final_percentage = 0.5 := by sorry

end NUMINAMATH_CALUDE_final_alcohol_percentage_l2044_204408


namespace NUMINAMATH_CALUDE_candidates_calculation_l2044_204458

theorem candidates_calculation (total_candidates : ℕ) : 
  (total_candidates * 6 / 100 : ℚ) + 83 = (total_candidates * 7 / 100 : ℚ) → 
  total_candidates = 8300 := by
  sorry

end NUMINAMATH_CALUDE_candidates_calculation_l2044_204458


namespace NUMINAMATH_CALUDE_house_painting_time_l2044_204426

theorem house_painting_time (total_time joint_time john_time : ℝ) 
  (h1 : joint_time = 2.4)
  (h2 : john_time = 6)
  (h3 : 1 / total_time + 1 / john_time = 1 / joint_time) :
  total_time = 4 := by sorry

end NUMINAMATH_CALUDE_house_painting_time_l2044_204426


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2343_l2044_204457

theorem smallest_prime_factor_of_2343 : 
  Nat.minFac 2343 = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2343_l2044_204457


namespace NUMINAMATH_CALUDE_remaining_fuel_after_three_hours_l2044_204491

/-- Represents the remaining fuel in a car's tank after driving for a certain time -/
def remaining_fuel (initial_fuel : ℝ) (consumption_rate : ℝ) (hours : ℝ) : ℝ :=
  initial_fuel - consumption_rate * hours

/-- Theorem stating that the remaining fuel after 3 hours matches the expression a-3b -/
theorem remaining_fuel_after_three_hours (a b : ℝ) :
  remaining_fuel a b 3 = a - 3 * b := by
  sorry

end NUMINAMATH_CALUDE_remaining_fuel_after_three_hours_l2044_204491


namespace NUMINAMATH_CALUDE_fraction_comparison_and_inequality_l2044_204494

theorem fraction_comparison_and_inequality : 
  (37 : ℚ) / 29 < 41 / 31 ∧ 
  41 / 31 < 31 / 23 ∧ 
  37 / 29 ≠ 4 / 3 ∧ 
  41 / 31 ≠ 4 / 3 ∧ 
  31 / 23 ≠ 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_and_inequality_l2044_204494


namespace NUMINAMATH_CALUDE_max_min_xy_constraint_l2044_204453

theorem max_min_xy_constraint (x y : ℝ) : 
  x^2 + x*y + y^2 ≤ 1 → 
  (∃ (max min : ℝ), 
    (∀ z, x - y + 2*x*y ≤ z → z ≤ max) ∧ 
    (∀ w, min ≤ w → w ≤ x - y + 2*x*y) ∧
    max = 25/24 ∧ min = -4) := by
  sorry

end NUMINAMATH_CALUDE_max_min_xy_constraint_l2044_204453


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_opposite_l2044_204471

def Bag := Fin 4

def is_black : Bag → Prop :=
  fun b => b.val < 2

def Draw := Fin 2 → Bag

def exactly_one_black (draw : Draw) : Prop :=
  (is_black (draw 0) ∧ ¬is_black (draw 1)) ∨ (¬is_black (draw 0) ∧ is_black (draw 1))

def exactly_two_black (draw : Draw) : Prop :=
  is_black (draw 0) ∧ is_black (draw 1)

theorem mutually_exclusive_not_opposite :
  (∃ (draw : Draw), exactly_one_black draw) ∧
  (∃ (draw : Draw), exactly_two_black draw) ∧
  (¬∃ (draw : Draw), exactly_one_black draw ∧ exactly_two_black draw) ∧
  (∃ (draw : Draw), ¬exactly_one_black draw ∧ ¬exactly_two_black draw) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_opposite_l2044_204471


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l2044_204468

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of a parallel axis ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ :=
  sorry

/-- Theorem stating the distance between foci for the given ellipse -/
theorem ellipse_foci_distance :
  ∀ (e : ParallelAxisEllipse),
    e.x_tangent = (6, 0) →
    e.y_tangent = (0, 3) →
    foci_distance e = 6 * Real.sqrt 3 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l2044_204468


namespace NUMINAMATH_CALUDE_nine_times_two_sevenths_squared_l2044_204489

theorem nine_times_two_sevenths_squared :
  9 * (2 / 7)^2 = 36 / 49 := by sorry

end NUMINAMATH_CALUDE_nine_times_two_sevenths_squared_l2044_204489


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2044_204485

theorem simplify_and_evaluate (x : ℝ) (h : x = -3) :
  (x^2 - 1) / (x + 2) / (1 - 1 / (x + 2)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2044_204485


namespace NUMINAMATH_CALUDE_folded_rectangle_long_side_l2044_204483

/-- A rectangular sheet of paper with a special folding property -/
structure FoldedRectangle where
  short_side : ℝ
  long_side : ℝ
  is_folded_to_midpoint : Bool
  triangles_congruent : Bool

/-- The folded rectangle satisfies the problem conditions -/
def satisfies_conditions (r : FoldedRectangle) : Prop :=
  r.short_side = 8 ∧ r.is_folded_to_midpoint ∧ r.triangles_congruent

/-- The theorem stating that under the given conditions, the long side must be 12 units -/
theorem folded_rectangle_long_side
  (r : FoldedRectangle)
  (h : satisfies_conditions r) :
  r.long_side = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_folded_rectangle_long_side_l2044_204483


namespace NUMINAMATH_CALUDE_soft_drink_bottles_l2044_204416

theorem soft_drink_bottles (small_bottles : ℕ) : 
  (small_bottles : ℝ) * 0.89 + 15000 * 0.88 = 18540 → 
  small_bottles = 6000 := by
  sorry

end NUMINAMATH_CALUDE_soft_drink_bottles_l2044_204416


namespace NUMINAMATH_CALUDE_rain_period_end_time_l2044_204460

/-- Represents time in 24-hour format -/
structure Time where
  hour : ℕ
  minute : ℕ

/-- Adds hours to a given time -/
def addHours (t : Time) (h : ℕ) : Time :=
  { hour := (t.hour + h) % 24, minute := t.minute }

theorem rain_period_end_time 
  (start : Time)
  (rain_duration : ℕ)
  (no_rain_duration : ℕ)
  (h_start : start = { hour := 9, minute := 0 })
  (h_rain : rain_duration = 2)
  (h_no_rain : no_rain_duration = 6) :
  addHours start (rain_duration + no_rain_duration) = { hour := 17, minute := 0 } :=
sorry

end NUMINAMATH_CALUDE_rain_period_end_time_l2044_204460


namespace NUMINAMATH_CALUDE_unique_real_root_l2044_204486

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - 10

-- Theorem statement
theorem unique_real_root : ∃! x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_real_root_l2044_204486


namespace NUMINAMATH_CALUDE_problem_l2044_204400

def l₁ (x y : ℝ) : Prop := x - 2*y + 3 = 0

def l₂ (x y : ℝ) : Prop := 2*x + y + 3 = 0

def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c d : ℝ, (∀ x y, f x y ↔ a*x + b*y = c) ∧
                 (∀ x y, g x y ↔ d*x - a*y = 0)

def p : Prop := ¬(perpendicular l₁ l₂)

def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ + 2 > Real.exp x₀

theorem problem : (¬p) ∧ q := by sorry

end NUMINAMATH_CALUDE_problem_l2044_204400


namespace NUMINAMATH_CALUDE_noMoreThanOneHead_atLeastTwoHeads_mutually_exclusive_l2044_204441

/-- Represents the outcome of tossing 3 coins -/
inductive CoinToss
  | HHH
  | HHT
  | HTH
  | THH
  | HTT
  | THT
  | TTH
  | TTT

/-- The event of having no more than one head -/
def noMoreThanOneHead (outcome : CoinToss) : Prop :=
  match outcome with
  | CoinToss.HTT | CoinToss.THT | CoinToss.TTH | CoinToss.TTT => True
  | _ => False

/-- The event of having at least two heads -/
def atLeastTwoHeads (outcome : CoinToss) : Prop :=
  match outcome with
  | CoinToss.HHH | CoinToss.HHT | CoinToss.HTH | CoinToss.THH => True
  | _ => False

/-- Theorem stating that "No more than one head" and "At least two heads" are mutually exclusive -/
theorem noMoreThanOneHead_atLeastTwoHeads_mutually_exclusive :
  ∀ (outcome : CoinToss), ¬(noMoreThanOneHead outcome ∧ atLeastTwoHeads outcome) :=
by
  sorry

end NUMINAMATH_CALUDE_noMoreThanOneHead_atLeastTwoHeads_mutually_exclusive_l2044_204441


namespace NUMINAMATH_CALUDE_impossible_odd_black_cells_impossible_one_black_cell_l2044_204488

/-- Represents a chessboard --/
structure Chessboard where
  black_cells : ℕ

/-- Represents the operation of repainting a row or column --/
def repaint (board : Chessboard) : Chessboard :=
  { black_cells := board.black_cells + (8 - 2 * (board.black_cells % 8)) }

/-- Theorem stating that it's impossible to achieve an odd number of black cells --/
theorem impossible_odd_black_cells (initial_board : Chessboard) 
  (h : Even initial_board.black_cells) :
  ∀ (final_board : Chessboard), 
  (∃ (n : ℕ), final_board = (n.iterate repaint initial_board)) → 
  Even final_board.black_cells :=
sorry

/-- Corollary: It's impossible to achieve exactly one black cell --/
theorem impossible_one_black_cell (initial_board : Chessboard) 
  (h : Even initial_board.black_cells) :
  ¬∃ (final_board : Chessboard), 
  (∃ (n : ℕ), final_board = (n.iterate repaint initial_board)) ∧ 
  final_board.black_cells = 1 :=
sorry

end NUMINAMATH_CALUDE_impossible_odd_black_cells_impossible_one_black_cell_l2044_204488


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l2044_204455

theorem binary_addition_subtraction : 
  let a : ℕ := 0b1101
  let b : ℕ := 0b1010
  let c : ℕ := 0b1111
  let d : ℕ := 0b1001
  a + b - c + d = 0b11001 := by sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l2044_204455


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2044_204445

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 1 - a 9 + a 17 = 7) :
  a 3 + a 15 = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2044_204445


namespace NUMINAMATH_CALUDE_first_studio_students_l2044_204417

theorem first_studio_students (total : ℕ) (second : ℕ) (third : ℕ) 
  (h1 : total = 376)
  (h2 : second = 135)
  (h3 : third = 131) :
  total - (second + third) = 110 := by
  sorry

end NUMINAMATH_CALUDE_first_studio_students_l2044_204417


namespace NUMINAMATH_CALUDE_certain_number_proof_l2044_204492

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem certain_number_proof : 
  ∃! x : ℕ, x > 0 ∧ 
    is_divisible_by (3153 + x) 9 ∧
    is_divisible_by (3153 + x) 70 ∧
    is_divisible_by (3153 + x) 25 ∧
    is_divisible_by (3153 + x) 21 ∧
    ∀ y : ℕ, y > 0 → 
      (is_divisible_by (3153 + y) 9 ∧
       is_divisible_by (3153 + y) 70 ∧
       is_divisible_by (3153 + y) 25 ∧
       is_divisible_by (3153 + y) 21) → 
      x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2044_204492


namespace NUMINAMATH_CALUDE_fixed_points_of_moving_circle_l2044_204438

/-- The equation of the moving circle -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 4*m*y + 6*m - 2 = 0

/-- A point is a fixed point if it satisfies the circle equation for all m -/
def is_fixed_point (x y : ℝ) : Prop :=
  ∀ m : ℝ, circle_equation x y m

theorem fixed_points_of_moving_circle :
  (is_fixed_point 1 1 ∧ is_fixed_point (1/5) (7/5)) ∧
  ∀ x y : ℝ, is_fixed_point x y → (x = 1 ∧ y = 1) ∨ (x = 1/5 ∧ y = 7/5) :=
sorry

end NUMINAMATH_CALUDE_fixed_points_of_moving_circle_l2044_204438


namespace NUMINAMATH_CALUDE_triangle_trig_expression_l2044_204461

theorem triangle_trig_expression (D E F : Real) (DE DF EF : Real) : 
  DE = 8 → DF = 10 → EF = 6 → 
  (Real.cos ((D - E) / 2) / Real.sin (F / 2)) - (Real.sin ((D - E) / 2) / Real.cos (F / 2)) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trig_expression_l2044_204461


namespace NUMINAMATH_CALUDE_astronomy_club_committee_probability_l2044_204480

/-- The probability of selecting a committee with more boys than girls -/
theorem astronomy_club_committee_probability :
  let total_members : ℕ := 24
  let boys : ℕ := 14
  let girls : ℕ := 10
  let committee_size : ℕ := 5
  let total_committees : ℕ := Nat.choose total_members committee_size
  let committees_more_boys : ℕ := 
    Nat.choose boys 3 * Nat.choose girls 2 +
    Nat.choose boys 4 * Nat.choose girls 1 +
    Nat.choose boys 5
  (committees_more_boys : ℚ) / total_committees = 7098 / 10626 := by
sorry

end NUMINAMATH_CALUDE_astronomy_club_committee_probability_l2044_204480


namespace NUMINAMATH_CALUDE_toys_storage_time_l2044_204475

/-- The time required to put all toys in the box -/
def time_to_store_toys (total_toys : ℕ) (net_gain_per_interval : ℕ) (interval_seconds : ℕ) : ℚ :=
  (total_toys : ℚ) / (net_gain_per_interval : ℚ) * (interval_seconds : ℚ) / 60

/-- Theorem stating that it takes 15 minutes to store all toys -/
theorem toys_storage_time :
  time_to_store_toys 30 1 30 = 15 := by
  sorry

#eval time_to_store_toys 30 1 30

end NUMINAMATH_CALUDE_toys_storage_time_l2044_204475


namespace NUMINAMATH_CALUDE_mod_congruence_l2044_204435

theorem mod_congruence (m : ℕ) : 
  198 * 963 ≡ m [ZMOD 50] → 0 ≤ m → m < 50 → m = 24 := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_l2044_204435


namespace NUMINAMATH_CALUDE_points_on_line_l2044_204410

-- Define the points
def p1 : ℝ × ℝ := (4, 8)
def p2 : ℝ × ℝ := (2, 2)
def p3 : ℝ × ℝ := (3, 5)
def p4 : ℝ × ℝ := (0, -2)
def p5 : ℝ × ℝ := (1, 1)
def p6 : ℝ × ℝ := (5, 11)
def p7 : ℝ × ℝ := (6, 14)

-- Function to check if a point lies on the line
def lies_on_line (p : ℝ × ℝ) : Prop :=
  let m := (p1.2 - p2.2) / (p1.1 - p2.1)
  let b := p1.2 - m * p1.1
  p.2 = m * p.1 + b

-- Theorem stating which points lie on the line
theorem points_on_line :
  lies_on_line p3 ∧ lies_on_line p6 ∧ lies_on_line p7 ∧
  ¬lies_on_line p4 ∧ ¬lies_on_line p5 :=
sorry

end NUMINAMATH_CALUDE_points_on_line_l2044_204410


namespace NUMINAMATH_CALUDE_opposite_reciprocal_fraction_l2044_204432

theorem opposite_reciprocal_fraction (a b c d : ℝ) 
  (h1 : a + b = 0) -- a and b are opposite numbers
  (h2 : c * d = 1) -- c and d are reciprocals
  : (5*a + 5*b - 7*c*d) / ((-c*d)^3) = 7 := by sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_fraction_l2044_204432


namespace NUMINAMATH_CALUDE_karen_crayons_count_l2044_204456

/-- The number of crayons Cindy has -/
def cindy_crayons : ℕ := 504

/-- The number of additional crayons Karen has compared to Cindy -/
def karen_additional_crayons : ℕ := 135

/-- The number of crayons Karen has -/
def karen_crayons : ℕ := cindy_crayons + karen_additional_crayons

theorem karen_crayons_count : karen_crayons = 639 := by
  sorry

end NUMINAMATH_CALUDE_karen_crayons_count_l2044_204456


namespace NUMINAMATH_CALUDE_evaluate_expression_l2044_204423

theorem evaluate_expression (x y : ℝ) (hx : x = 4) (hy : y = 2) :
  y * (y - 2 * x)^2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2044_204423


namespace NUMINAMATH_CALUDE_polynomial_invariant_is_constant_l2044_204437

/-- A polynomial function from ℝ×ℝ to ℝ×ℝ -/
def PolynomialRR : Type := (ℝ × ℝ) → (ℝ × ℝ)

/-- The property that P(x,y) = P(x+y,x-y) for all x,y ∈ ℝ -/
def HasInvariantProperty (P : PolynomialRR) : Prop :=
  ∀ x y : ℝ, P (x, y) = P (x + y, x - y)

/-- The theorem stating that any polynomial with the invariant property is constant -/
theorem polynomial_invariant_is_constant (P : PolynomialRR) 
  (h : HasInvariantProperty P) : 
  ∃ a b : ℝ, ∀ x y : ℝ, P (x, y) = (a, b) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_invariant_is_constant_l2044_204437


namespace NUMINAMATH_CALUDE_divisibility_in_base_system_l2044_204427

theorem divisibility_in_base_system : ∃! (b : ℕ), b ≥ 8 ∧ (∃ (q : ℕ), 7 * b + 2 = q * (2 * b^2 + 7 * b + 5)) ∧ b = 8 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_in_base_system_l2044_204427


namespace NUMINAMATH_CALUDE_coefficient_x3y4_expansion_l2044_204498

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℚ := (Nat.choose n k : ℚ)

-- Define the expansion term
def expansionTerm (n k : ℕ) (x y : ℚ) : ℚ :=
  binomial n k * (x ^ k) * (y ^ (n - k))

-- Theorem statement
theorem coefficient_x3y4_expansion :
  let n : ℕ := 9
  let k : ℕ := 3
  let x : ℚ := 2/3
  let y : ℚ := -3/4
  expansionTerm n k x y = 441/992 := by
sorry

end NUMINAMATH_CALUDE_coefficient_x3y4_expansion_l2044_204498


namespace NUMINAMATH_CALUDE_vector_parallel_implies_x_value_l2044_204487

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is parallel to c, then the x-coordinate of a is -11/3 -/
theorem vector_parallel_implies_x_value 
  (a b c : ℝ × ℝ) 
  (hb : b = (1, 2)) 
  (hc : c = (-1, 3)) 
  (ha : a.2 = 1) 
  (h_parallel : ∃ (k : ℝ), (a + 2 • b) = k • c) : 
  a.1 = -11/3 := by sorry

end NUMINAMATH_CALUDE_vector_parallel_implies_x_value_l2044_204487


namespace NUMINAMATH_CALUDE_percentage_of_men_l2044_204418

/-- The percentage of employees who are men, given picnic attendance data. -/
theorem percentage_of_men (men_attendance : Real) (women_attendance : Real) (total_attendance : Real)
  (h1 : men_attendance = 0.2)
  (h2 : women_attendance = 0.4)
  (h3 : total_attendance = 0.29000000000000004) :
  ∃ (men_percentage : Real),
    men_percentage * men_attendance + (1 - men_percentage) * women_attendance = total_attendance ∧
    men_percentage = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_men_l2044_204418


namespace NUMINAMATH_CALUDE_min_omega_value_l2044_204490

/-- Given a function f(x) = 2 * sin(ω * x) where ω > 0, and f(x) has a minimum value of -2
    in the interval [-π/3, π/6], prove that the minimum value of ω is 3/2. -/
theorem min_omega_value (ω : ℝ) : 
  (ω > 0) →
  (∀ x ∈ Set.Icc (-π/3) (π/6), 2 * Real.sin (ω * x) ≥ -2) →
  (∃ x ∈ Set.Icc (-π/3) (π/6), 2 * Real.sin (ω * x) = -2) →
  ω ≥ 3/2 :=
sorry

end NUMINAMATH_CALUDE_min_omega_value_l2044_204490


namespace NUMINAMATH_CALUDE_saras_quarters_l2044_204433

/-- Sara's quarters problem -/
theorem saras_quarters (initial_quarters final_quarters dad_quarters : ℕ) 
  (h1 : initial_quarters = 21)
  (h2 : dad_quarters = 49)
  (h3 : final_quarters = initial_quarters + dad_quarters) :
  final_quarters = 70 := by
  sorry

end NUMINAMATH_CALUDE_saras_quarters_l2044_204433


namespace NUMINAMATH_CALUDE_basil_planter_problem_l2044_204411

theorem basil_planter_problem (total_seeds : Nat) (large_planters : Nat) (seeds_per_large : Nat) (seeds_per_small : Nat) :
  total_seeds = 200 →
  large_planters = 4 →
  seeds_per_large = 20 →
  seeds_per_small = 4 →
  (total_seeds - large_planters * seeds_per_large) / seeds_per_small = 30 := by
  sorry

end NUMINAMATH_CALUDE_basil_planter_problem_l2044_204411


namespace NUMINAMATH_CALUDE_cans_per_bag_l2044_204434

theorem cans_per_bag (total_bags : ℕ) (total_cans : ℕ) (h1 : total_bags = 8) (h2 : total_cans = 40) :
  total_cans / total_bags = 5 := by
  sorry

end NUMINAMATH_CALUDE_cans_per_bag_l2044_204434


namespace NUMINAMATH_CALUDE_determine_F_l2044_204451

def first_number (D E : ℕ) : ℕ := 9000000 + 600000 + 100000 * D + 10000 + 1000 * E + 800 + 2

def second_number (D E F : ℕ) : ℕ := 5000000 + 400000 + 100000 * E + 10000 * D + 2000 + 100 + 10 * F

theorem determine_F :
  ∀ D E F : ℕ,
  D < 10 → E < 10 → F < 10 →
  (first_number D E) % 3 = 0 →
  (second_number D E F) % 3 = 0 →
  F = 2 := by
sorry

end NUMINAMATH_CALUDE_determine_F_l2044_204451


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2044_204403

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 - 10 = 188 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2044_204403


namespace NUMINAMATH_CALUDE_andrey_solved_half_l2044_204469

theorem andrey_solved_half (N : ℕ) (x : ℕ) : 
  (N - x - (N - x) / 3 = N / 3) → 
  (x : ℚ) / N = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_andrey_solved_half_l2044_204469


namespace NUMINAMATH_CALUDE_sum_of_greatest_b_values_l2044_204444

theorem sum_of_greatest_b_values (b : ℝ) : 
  4 * b^4 - 41 * b^2 + 100 = 0 → 
  ∃ (b1 b2 : ℝ), b1 ≥ b2 ∧ b2 ≥ 0 ∧ 
    (4 * b1^4 - 41 * b1^2 + 100 = 0) ∧ 
    (4 * b2^4 - 41 * b2^2 + 100 = 0) ∧ 
    b1 + b2 = 4.5 ∧
    ∀ (x : ℝ), (4 * x^4 - 41 * x^2 + 100 = 0) → x ≤ b1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_greatest_b_values_l2044_204444


namespace NUMINAMATH_CALUDE_weight_lifting_duration_l2044_204443

-- Define the total practice time in minutes
def total_practice_time : ℕ := 120

-- Define the time spent on running and weight lifting combined
def run_lift_time : ℕ := total_practice_time / 2

-- Define the relationship between running and weight lifting time
def weight_lifting_time (x : ℕ) : Prop := 
  x + 2 * x = run_lift_time

-- Theorem statement
theorem weight_lifting_duration : 
  ∃ x : ℕ, weight_lifting_time x ∧ x = 20 := by sorry

end NUMINAMATH_CALUDE_weight_lifting_duration_l2044_204443


namespace NUMINAMATH_CALUDE_sector_arc_length_l2044_204464

/-- Given a sector with central angle π/3 and radius 3, its arc length is π. -/
theorem sector_arc_length (α : Real) (r : Real) (l : Real) : 
  α = π / 3 → r = 3 → l = r * α → l = π := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l2044_204464


namespace NUMINAMATH_CALUDE_prime_with_integer_roots_l2044_204470

theorem prime_with_integer_roots (p : ℕ) : 
  Prime p → 
  (∃ x y : ℤ, x^2 + p*x - 530*p = 0 ∧ y^2 + p*y - 530*p = 0) → 
  43 < p ∧ p ≤ 53 := by
sorry

end NUMINAMATH_CALUDE_prime_with_integer_roots_l2044_204470


namespace NUMINAMATH_CALUDE_science_club_enrollment_l2044_204450

theorem science_club_enrollment (total : ℕ) (biology : ℕ) (chemistry : ℕ) (both : ℕ) 
  (h1 : total = 60)
  (h2 : biology = 40)
  (h3 : chemistry = 35)
  (h4 : both = 25) :
  total - (biology + chemistry - both) = 10 := by
  sorry

end NUMINAMATH_CALUDE_science_club_enrollment_l2044_204450


namespace NUMINAMATH_CALUDE_x_plus_reciprocal_x_l2044_204472

theorem x_plus_reciprocal_x (x : ℝ) 
  (h1 : x^3 + 1/x^3 = 110) 
  (h2 : (x + 1/x)^2 - 2*x - 2/x = 38) : 
  x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_reciprocal_x_l2044_204472


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2044_204482

/-- A rectangle with given diagonal and area has a specific perimeter -/
theorem rectangle_perimeter (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  a^2 + b^2 = 25^2 → a * b = 168 → 2 * (a + b) = 62 := by
  sorry

#check rectangle_perimeter

end NUMINAMATH_CALUDE_rectangle_perimeter_l2044_204482


namespace NUMINAMATH_CALUDE_y_plus_two_over_y_l2044_204409

theorem y_plus_two_over_y (y : ℝ) (h : 5 = y^2 + 4/y^2) : 
  y + 2/y = 3 ∨ y + 2/y = -3 := by
sorry

end NUMINAMATH_CALUDE_y_plus_two_over_y_l2044_204409


namespace NUMINAMATH_CALUDE_missing_number_proof_l2044_204422

theorem missing_number_proof (x : ℝ) : 
  x * 54 = 75625 → 
  ⌊x + 0.5⌋ = 1400 := by
sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2044_204422


namespace NUMINAMATH_CALUDE_data_ratio_l2044_204495

theorem data_ratio (a b c : ℝ) 
  (h1 : a = b - c) 
  (h2 : a = 12) 
  (h3 : a + b + c = 96) : 
  b / a = 4 := by
sorry

end NUMINAMATH_CALUDE_data_ratio_l2044_204495


namespace NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_focus_l2044_204477

theorem parabola_equation_from_hyperbola_focus : ∃ (a b c : ℝ),
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 3 - y^2 / 6 = 1) →
  c^2 = a^2 + b^2 →
  (∀ x y : ℝ, y^2 = 4 * c * x ↔ y^2 = 12 * x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_focus_l2044_204477


namespace NUMINAMATH_CALUDE_pencils_given_l2044_204439

theorem pencils_given (initial : ℕ) (final : ℕ) (given : ℕ) : 
  initial = 51 → final = 57 → given = final - initial → given = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencils_given_l2044_204439


namespace NUMINAMATH_CALUDE_faulty_clock_correct_time_fraction_l2044_204463

/-- Represents a faulty digital clock that displays '5' instead of '2' over a 24-hour period -/
structure FaultyClock where
  /-- The number of hours in a day -/
  hours_per_day : ℕ
  /-- The number of minutes in an hour -/
  minutes_per_hour : ℕ
  /-- The number of hours affected by the fault -/
  faulty_hours : ℕ
  /-- The number of minutes per hour affected by the fault -/
  faulty_minutes : ℕ

/-- The fraction of the day a faulty clock displays the correct time -/
def correct_time_fraction (c : FaultyClock) : ℚ :=
  ((c.hours_per_day - c.faulty_hours) / c.hours_per_day) *
  ((c.minutes_per_hour - c.faulty_minutes) / c.minutes_per_hour)

/-- Theorem stating that the fraction of the day the faulty clock displays the correct time is 9/16 -/
theorem faulty_clock_correct_time_fraction :
  ∃ (c : FaultyClock), c.hours_per_day = 24 ∧ c.minutes_per_hour = 60 ∧
  c.faulty_hours = 6 ∧ c.faulty_minutes = 15 ∧
  correct_time_fraction c = 9 / 16 :=
by
  sorry

end NUMINAMATH_CALUDE_faulty_clock_correct_time_fraction_l2044_204463


namespace NUMINAMATH_CALUDE_log_difference_cubes_l2044_204402

theorem log_difference_cubes (x y : ℝ) (a : ℝ) (h : Real.log x - Real.log y = a) :
  Real.log ((x / 2) ^ 3) - Real.log ((y / 2) ^ 3) = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_log_difference_cubes_l2044_204402


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l2044_204419

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_t_value :
  let m : ℝ × ℝ := (2, 8)
  let n : ℝ → ℝ × ℝ := fun t ↦ (-4, t)
  ∀ t : ℝ, are_parallel m (n t) → t = -16 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l2044_204419


namespace NUMINAMATH_CALUDE_chip_price_is_two_l2044_204476

/-- The price of a packet of chips -/
def chip_price : ℝ := sorry

/-- The price of a packet of corn chips -/
def corn_chip_price : ℝ := 1.5

/-- The number of packets of chips John buys -/
def num_chips : ℕ := 15

/-- The number of packets of corn chips John buys -/
def num_corn_chips : ℕ := 10

/-- John's total budget -/
def total_budget : ℝ := 45

theorem chip_price_is_two :
  chip_price * num_chips + corn_chip_price * num_corn_chips = total_budget →
  chip_price = 2 := by sorry

end NUMINAMATH_CALUDE_chip_price_is_two_l2044_204476


namespace NUMINAMATH_CALUDE_f_at_negative_one_l2044_204431

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x^2 + 1

-- Theorem statement
theorem f_at_negative_one : f (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_at_negative_one_l2044_204431


namespace NUMINAMATH_CALUDE_solution_to_system_l2044_204466

theorem solution_to_system : ∃ (x y : ℝ), 
  x^2 * y - x * y^2 - 3*x + 3*y + 1 = 0 ∧
  x^3 * y - x * y^3 - 3*x^2 + 3*y^2 + 3 = 0 ∧
  x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_l2044_204466


namespace NUMINAMATH_CALUDE_complementary_angles_problem_l2044_204499

theorem complementary_angles_problem (C D : Real) : 
  C + D = 90 →  -- Angles C and D are complementary
  C = 3 * D →   -- The measure of angle C is 3 times angle D
  C = 67.5 :=   -- The measure of angle C is 67.5°
by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_problem_l2044_204499


namespace NUMINAMATH_CALUDE_garrett_granola_bars_l2044_204404

/-- Proves that Garrett bought 6 oatmeal raisin granola bars -/
theorem garrett_granola_bars :
  ∀ (total peanut oatmeal_raisin : ℕ),
    total = 14 →
    peanut = 8 →
    total = peanut + oatmeal_raisin →
    oatmeal_raisin = 6 := by
  sorry

end NUMINAMATH_CALUDE_garrett_granola_bars_l2044_204404


namespace NUMINAMATH_CALUDE_rectangle_perimeters_l2044_204496

def is_valid_perimeter (p : ℕ) : Prop :=
  ∃ (x y : ℕ), 
    (x > 0 ∧ y > 0) ∧
    (3 * (2 * (x + y)) = 10) ∧
    (p = 2 * (x + y) ∨ p = 2 * (3 * x) ∨ p = 2 * (3 * y))

theorem rectangle_perimeters : 
  {p : ℕ | is_valid_perimeter p} = {14, 16, 18, 22, 26} :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeters_l2044_204496


namespace NUMINAMATH_CALUDE_pythagorean_theorem_l2044_204452

-- Define a right-angled triangle
def RightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

-- Theorem statement
theorem pythagorean_theorem (a b c : ℝ) :
  RightTriangle a b c → a^2 + b^2 = c^2 :=
by
  sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_l2044_204452


namespace NUMINAMATH_CALUDE_max_cake_pieces_l2044_204414

/-- The size of the large cake in inches -/
def large_cake_size : ℕ := 20

/-- The size of the small piece in inches -/
def small_piece_size : ℕ := 2

/-- The area of the large cake in square inches -/
def large_cake_area : ℕ := large_cake_size * large_cake_size

/-- The area of a small piece in square inches -/
def small_piece_area : ℕ := small_piece_size * small_piece_size

/-- The maximum number of small pieces that can be cut from the large cake -/
def max_pieces : ℕ := large_cake_area / small_piece_area

theorem max_cake_pieces : max_pieces = 100 := by
  sorry

end NUMINAMATH_CALUDE_max_cake_pieces_l2044_204414


namespace NUMINAMATH_CALUDE_coconut_moving_theorem_l2044_204440

/-- The number of coconuts Barbie can carry in one trip -/
def barbie_capacity : ℕ := 4

/-- The number of coconuts Bruno can carry in one trip -/
def bruno_capacity : ℕ := 8

/-- The number of trips Barbie and Bruno make together -/
def num_trips : ℕ := 12

/-- The total number of coconuts Barbie and Bruno can move -/
def total_coconuts : ℕ := (barbie_capacity + bruno_capacity) * num_trips

theorem coconut_moving_theorem : total_coconuts = 144 := by
  sorry

end NUMINAMATH_CALUDE_coconut_moving_theorem_l2044_204440


namespace NUMINAMATH_CALUDE_f_property_f_1001_eq_1_f_1002_eq_1_l2044_204407

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def has_prime_divisor (n : ℕ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ p ∣ n

theorem f_property (f : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 1 →
    ∃ p : ℕ, is_prime p ∧ p ∣ n ∧ f n = f (n / p) - f p

theorem f_1001_eq_1 (f : ℕ → ℤ) : Prop := f 1001 = 1

theorem f_1002_eq_1 (f : ℕ → ℤ) : f_property f → f_1001_eq_1 f → f 1002 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_property_f_1001_eq_1_f_1002_eq_1_l2044_204407


namespace NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l2044_204481

/-- The area of a stripe wrapping around a cylindrical silo -/
theorem stripe_area_on_cylindrical_silo 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h_diameter : diameter = 40) 
  (h_stripe_width : stripe_width = 4) 
  (h_revolutions : revolutions = 3) : 
  stripe_width * revolutions * π * diameter = 480 * π := by
  sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l2044_204481


namespace NUMINAMATH_CALUDE_nut_problem_l2044_204413

theorem nut_problem (sue_nuts : ℕ) (harry_nuts : ℕ) (bill_nuts : ℕ) 
  (h1 : sue_nuts = 48)
  (h2 : harry_nuts = 2 * sue_nuts)
  (h3 : bill_nuts = 6 * harry_nuts) :
  bill_nuts + harry_nuts = 672 := by
sorry

end NUMINAMATH_CALUDE_nut_problem_l2044_204413


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l2044_204429

-- Define the ⋈ operation
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ h : ℝ, bowtie 5 h = 11 ∧ h = 30 :=
sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l2044_204429


namespace NUMINAMATH_CALUDE_shortest_side_right_triangle_l2044_204454

theorem shortest_side_right_triangle (a b c : ℝ) : 
  a = 7 → b = 10 → c^2 = a^2 + b^2 → min a (min b c) = 7 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_right_triangle_l2044_204454


namespace NUMINAMATH_CALUDE_min_moves_at_least_50_l2044_204406

/-- A 4x4 grid representing the puzzle state -/
def PuzzleState := Fin 4 → Fin 4 → Option (Fin 16)

/-- A move in the puzzle -/
inductive Move
| slide : Fin 4 → Fin 4 → Fin 4 → Fin 4 → Move
| jump  : Fin 4 → Fin 4 → Fin 4 → Fin 4 → Move

/-- Check if a PuzzleState is a valid magic square with sum 30 -/
def isMagicSquare (state : PuzzleState) : Prop := sorry

/-- Check if a move is valid for a given state -/
def isValidMove (state : PuzzleState) (move : Move) : Prop := sorry

/-- Apply a move to a state -/
def applyMove (state : PuzzleState) (move : Move) : PuzzleState := sorry

/-- The minimum number of moves required to solve the puzzle -/
def minMoves (initial : PuzzleState) : ℕ := sorry

/-- The theorem stating that the minimum number of moves is at least 50 -/
theorem min_moves_at_least_50 (initial : PuzzleState) : 
  minMoves initial ≥ 50 := by sorry

end NUMINAMATH_CALUDE_min_moves_at_least_50_l2044_204406


namespace NUMINAMATH_CALUDE_first_friend_cookies_l2044_204447

theorem first_friend_cookies (initial : ℕ) (eaten : ℕ) (brother : ℕ) (second : ℕ) (third : ℕ) (remaining : ℕ) : 
  initial = 22 → 
  eaten = 2 → 
  brother = 1 → 
  second = 5 → 
  third = 5 → 
  remaining = 6 → 
  initial - eaten - brother - second - third - remaining = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_friend_cookies_l2044_204447


namespace NUMINAMATH_CALUDE_function_property_l2044_204442

def is_solution_set (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x, x ∈ S ↔ (x > 0 ∧ f x + f (x - 8) ≤ 2)

theorem function_property (f : ℝ → ℝ) (h1 : ∀ x y, x > 0 → y > 0 → f (x * y) = f x + f y)
  (h2 : ∀ x y, x > 0 → y > 0 → x < y → f x < f y) (h3 : f 3 = 1) :
  is_solution_set f (Set.Ioo 8 9) :=
sorry

end NUMINAMATH_CALUDE_function_property_l2044_204442


namespace NUMINAMATH_CALUDE_abs_neg_one_sixth_gt_neg_one_seventh_l2044_204497

theorem abs_neg_one_sixth_gt_neg_one_seventh : |-(1/6)| > -(1/7) := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_sixth_gt_neg_one_seventh_l2044_204497


namespace NUMINAMATH_CALUDE_lcm_gcf_relation_l2044_204425

theorem lcm_gcf_relation (n : ℕ) (h1 : Nat.lcm n 14 = 56) (h2 : Nat.gcd n 14 = 10) : n = 40 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_relation_l2044_204425


namespace NUMINAMATH_CALUDE_basketball_campers_count_l2044_204424

theorem basketball_campers_count (total_campers soccer_campers football_campers : ℕ) 
  (h1 : total_campers = 88)
  (h2 : soccer_campers = 32)
  (h3 : football_campers = 32) :
  total_campers - soccer_campers - football_campers = 24 := by
  sorry

end NUMINAMATH_CALUDE_basketball_campers_count_l2044_204424


namespace NUMINAMATH_CALUDE_product_expansion_l2044_204405

theorem product_expansion (x : ℝ) : 
  (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l2044_204405


namespace NUMINAMATH_CALUDE_first_sequence_30th_term_l2044_204462

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- The 30th term of the first arithmetic sequence is 178 -/
theorem first_sequence_30th_term :
  arithmeticSequenceTerm 4 6 30 = 178 := by
  sorry

end NUMINAMATH_CALUDE_first_sequence_30th_term_l2044_204462


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_one_l2044_204428

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- Given two parallel lines y = ax - 2 and y = (2-a)x + 1, prove that a = 1 -/
theorem parallel_lines_a_equals_one :
  (∀ x y : ℝ, y = a * x - 2 ↔ y = (2 - a) * x + 1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_one_l2044_204428


namespace NUMINAMATH_CALUDE_sphere_dimensions_l2044_204459

-- Define the hole dimensions
def hole_diameter : ℝ := 12
def hole_depth : ℝ := 2

-- Define the sphere
def sphere_radius : ℝ := 10

-- Theorem statement
theorem sphere_dimensions (r : ℝ) (h : r = sphere_radius) :
  -- The radius satisfies the Pythagorean theorem for the right triangle formed
  (r - hole_depth) ^ 2 + (hole_diameter / 2) ^ 2 = r ^ 2 ∧
  -- The surface area of the sphere is 400π
  4 * Real.pi * r ^ 2 = 400 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_dimensions_l2044_204459


namespace NUMINAMATH_CALUDE_odd_power_sum_divisibility_l2044_204412

theorem odd_power_sum_divisibility (k : ℕ) (x y : ℤ) :
  (∃ q : ℤ, x^(2*k-1) + y^(2*k-1) = (x+y) * q) →
  (∃ r : ℤ, x^(2*k+1) + y^(2*k+1) = (x+y) * r) :=
by sorry

end NUMINAMATH_CALUDE_odd_power_sum_divisibility_l2044_204412
