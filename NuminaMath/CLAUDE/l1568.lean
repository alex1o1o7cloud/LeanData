import Mathlib

namespace arithmetic_sequence_a1_value_l1568_156893

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_a1_value
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a 1)
  (h_geom_mean : a 2 ^ 2 = a 1 * a 4) :
  a 1 = 1 := by
sorry

end arithmetic_sequence_a1_value_l1568_156893


namespace adam_red_balls_l1568_156842

/-- The number of red balls in Adam's collection --/
def red_balls (total blue pink orange : ℕ) : ℕ :=
  total - (blue + pink + orange)

/-- Theorem stating the number of red balls in Adam's collection --/
theorem adam_red_balls :
  ∀ (total blue pink orange : ℕ),
    total = 50 →
    blue = 10 →
    orange = 5 →
    pink = 3 * orange →
    red_balls total blue pink orange = 20 := by
  sorry

end adam_red_balls_l1568_156842


namespace parabola_locus_l1568_156850

/-- Parabola C: y² = 4x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of parabola C -/
def focus : ℝ × ℝ := (1, 0)

/-- Point P lies on parabola C -/
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola_C P.1 P.2

/-- Vector relation between P, Q, and F -/
def vector_relation (P Q : ℝ × ℝ) : Prop :=
  (P.1 - Q.1, P.2 - Q.2) = (2*(focus.1 - Q.1), 2*(focus.2 - Q.2))

/-- Curve E: 9y² = 12x - 8 -/
def curve_E (x y : ℝ) : Prop := 9*y^2 = 12*x - 8

theorem parabola_locus :
  ∀ Q : ℝ × ℝ,
  (∃ P : ℝ × ℝ, point_on_parabola P ∧ vector_relation P Q) →
  curve_E Q.1 Q.2 :=
by sorry

end parabola_locus_l1568_156850


namespace winner_third_difference_l1568_156888

/-- Represents the vote count for each candidate in the election. -/
structure ElectionResult where
  total_votes : Nat
  num_candidates : Nat
  winner_votes : Nat
  second_votes : Nat
  third_votes : Nat
  fourth_votes : Nat

/-- Theorem stating the difference between the winner's votes and the third opponent's votes. -/
theorem winner_third_difference (e : ElectionResult) 
  (h1 : e.total_votes = 963)
  (h2 : e.num_candidates = 4)
  (h3 : e.winner_votes = 195)
  (h4 : e.second_votes = 142)
  (h5 : e.third_votes = 116)
  (h6 : e.fourth_votes = 90)
  : e.winner_votes - e.third_votes = 79 := by
  sorry


end winner_third_difference_l1568_156888


namespace function_passes_through_point_l1568_156867

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f := fun x => a^(x - 1) + 3
  f 1 = 4 := by
  sorry

end function_passes_through_point_l1568_156867


namespace complex_number_subtraction_l1568_156865

theorem complex_number_subtraction (i : ℂ) (h : i * i = -1) :
  (7 - 3 * i) - 3 * (2 + 5 * i) = 1 - 18 * i :=
by sorry

end complex_number_subtraction_l1568_156865


namespace runner_stops_on_start_quarter_l1568_156877

/-- Represents the quarters of the circular track -/
inductive Quarter : Type
  | X : Quarter
  | Y : Quarter
  | Z : Quarter
  | W : Quarter

/-- The circular track -/
structure Track :=
  (circumference : ℝ)
  (quarters : Fin 4 → Quarter)

/-- Represents a runner on the track -/
structure Runner :=
  (start_quarter : Quarter)
  (distance_run : ℝ)

/-- Function to determine the quarter where a runner stops -/
def stop_quarter (track : Track) (runner : Runner) : Quarter :=
  runner.start_quarter

/-- Theorem stating that a runner stops on the same quarter they started on
    when running a multiple of the track's circumference -/
theorem runner_stops_on_start_quarter 
  (track : Track) 
  (runner : Runner) 
  (h1 : track.circumference = 200)
  (h2 : runner.distance_run = 3000) :
  stop_quarter track runner = runner.start_quarter :=
sorry

end runner_stops_on_start_quarter_l1568_156877


namespace equal_pay_implies_harry_worked_33_hours_l1568_156896

/-- Payment structure for an employee -/
structure PaymentStructure where
  base_rate : ℝ
  base_hours : ℕ
  overtime_multiplier : ℝ

/-- Calculate the total pay for an employee given their payment structure and hours worked -/
def calculate_pay (ps : PaymentStructure) (hours_worked : ℕ) : ℝ :=
  let base_pay := ps.base_rate * (min ps.base_hours hours_worked)
  let overtime_hours := max 0 (hours_worked - ps.base_hours)
  let overtime_pay := ps.base_rate * ps.overtime_multiplier * overtime_hours
  base_pay + overtime_pay

theorem equal_pay_implies_harry_worked_33_hours 
  (x : ℝ) 
  (harry_structure : PaymentStructure)
  (james_structure : PaymentStructure)
  (h_harry : harry_structure = { base_rate := x, base_hours := 15, overtime_multiplier := 1.5 })
  (h_james : james_structure = { base_rate := x, base_hours := 40, overtime_multiplier := 2 })
  (james_hours : ℕ)
  (h_james_hours : james_hours = 41)
  (harry_hours : ℕ)
  (h_equal_pay : calculate_pay harry_structure harry_hours = calculate_pay james_structure james_hours) :
  harry_hours = 33 := by
  sorry

end equal_pay_implies_harry_worked_33_hours_l1568_156896


namespace amber_max_ounces_l1568_156811

/-- Represents the amount of money Amber has to spend -/
def amberMoney : ℚ := 7

/-- Represents the cost of a bag of candy in dollars -/
def candyCost : ℚ := 1

/-- Represents the number of ounces in a bag of candy -/
def candyOunces : ℚ := 12

/-- Represents the cost of a bag of chips in dollars -/
def chipsCost : ℚ := 1.4

/-- Represents the number of ounces in a bag of chips -/
def chipsOunces : ℚ := 17

/-- Calculates the maximum number of ounces Amber can get -/
def maxOunces : ℚ := max (amberMoney / candyCost * candyOunces) (amberMoney / chipsCost * chipsOunces)

theorem amber_max_ounces : maxOunces = 85 := by sorry

end amber_max_ounces_l1568_156811


namespace test_scores_theorem_l1568_156894

def is_valid_sequence (s : List Nat) : Prop :=
  s.length > 0 ∧ 
  s.Nodup ∧ 
  s.sum = 119 ∧ 
  (s.take 3).sum = 23 ∧ 
  (s.reverse.take 3).sum = 49

theorem test_scores_theorem (s : List Nat) (h : is_valid_sequence s) : 
  s.length = 10 ∧ s.maximum? = some 18 := by
  sorry

end test_scores_theorem_l1568_156894


namespace salary_increase_percentage_l1568_156824

theorem salary_increase_percentage (initial_salary final_salary : ℝ) 
  (increase_percentage decrease_percentage : ℝ) :
  initial_salary = 5000 →
  final_salary = 5225 →
  decrease_percentage = 5 →
  final_salary = initial_salary * (1 + increase_percentage / 100) * (1 - decrease_percentage / 100) →
  increase_percentage = 10 := by
  sorry

end salary_increase_percentage_l1568_156824


namespace interest_rate_calculation_l1568_156852

theorem interest_rate_calculation (P r : ℝ) 
  (h1 : P * (1 + 3 * r) = 300)
  (h2 : P * (1 + 8 * r) = 400) :
  r = 1 / 12 := by
sorry

end interest_rate_calculation_l1568_156852


namespace clue_represents_8671_l1568_156837

/-- Represents a mapping from characters to digits -/
def CharToDigitMap := Char → Nat

/-- Creates a mapping from the string "BEST OF LUCK" to digits 0-9 in order -/
def createBestOfLuckMap : CharToDigitMap :=
  fun c => match c with
    | 'B' => 0
    | 'E' => 1
    | 'S' => 2
    | 'T' => 3
    | 'O' => 4
    | 'F' => 5
    | 'L' => 6
    | 'U' => 7
    | 'C' => 8
    | 'K' => 9
    | _ => 0  -- Default case, should not be reached for valid inputs

/-- Converts a string to a number using the given character-to-digit mapping -/
def stringToNumber (map : CharToDigitMap) (s : String) : Nat :=
  s.foldl (fun acc c => 10 * acc + map c) 0

/-- Theorem: The code word "CLUE" represents the number 8671 -/
theorem clue_represents_8671 :
  stringToNumber createBestOfLuckMap "CLUE" = 8671 := by
  sorry

#eval stringToNumber createBestOfLuckMap "CLUE"

end clue_represents_8671_l1568_156837


namespace cubes_fill_box_l1568_156898

/-- Proves that 2-inch cubes fill 100% of a 8×6×12 inch box -/
theorem cubes_fill_box (box_length box_width box_height cube_side: ℕ) 
  (h1: box_length = 8)
  (h2: box_width = 6)
  (h3: box_height = 12)
  (h4: cube_side = 2)
  (h5: box_length % cube_side = 0)
  (h6: box_width % cube_side = 0)
  (h7: box_height % cube_side = 0) :
  (((box_length / cube_side) * (box_width / cube_side) * (box_height / cube_side)) * cube_side^3) / (box_length * box_width * box_height) = 1 :=
by sorry

end cubes_fill_box_l1568_156898


namespace line_one_point_not_always_tangent_l1568_156823

-- Define a curve as a set of points in 2D space
def Curve := Set (ℝ × ℝ)

-- Define a line as a set of points in 2D space
def Line := Set (ℝ × ℝ)

-- Define what it means for a line to be tangent to a curve
def IsTangent (l : Line) (c : Curve) : Prop := sorry

-- Define what it means for a line to have only one common point with a curve
def HasOneCommonPoint (l : Line) (c : Curve) : Prop := sorry

-- Theorem statement
theorem line_one_point_not_always_tangent :
  ∃ (l : Line) (c : Curve), HasOneCommonPoint l c ∧ ¬IsTangent l c := by sorry

end line_one_point_not_always_tangent_l1568_156823


namespace find_number_l1568_156868

theorem find_number : ∃ x : ℝ, (0.38 * 80) - (0.12 * x) = 11.2 ∧ x = 160 := by
  sorry

end find_number_l1568_156868


namespace simplify_square_roots_l1568_156808

theorem simplify_square_roots : 
  (Real.sqrt 288 / Real.sqrt 32) - (Real.sqrt 242 / Real.sqrt 121) = 3 - Real.sqrt 2 := by
  sorry

end simplify_square_roots_l1568_156808


namespace monotonic_function_a_range_l1568_156890

/-- The function f(x) = x ln x - (a/2)x^2 - x is monotonic on (0, +∞) if and only if a ∈ [1/e, +∞) -/
theorem monotonic_function_a_range (a : ℝ) :
  (∀ x > 0, Monotone (fun x => x * Real.log x - a / 2 * x^2 - x)) ↔ a ≥ 1 / Real.exp 1 := by
  sorry

end monotonic_function_a_range_l1568_156890


namespace expression_factorization_l1568_156866

theorem expression_factorization (a : ℝ) :
  (8 * a^4 + 92 * a^3 - 15 * a^2 + 1) - (-2 * a^4 + 3 * a^3 - 5 * a^2 + 2) = 
  a^2 * (10 * a^2 + 89 * a - 10) - 1 := by
sorry

end expression_factorization_l1568_156866


namespace brad_age_l1568_156848

/-- Given the ages and relationships between Jaymee, Shara, and Brad, prove Brad's age -/
theorem brad_age (shara_age : ℕ) (jaymee_age : ℕ) (brad_age : ℕ) : 
  shara_age = 10 →
  jaymee_age = 2 * shara_age + 2 →
  brad_age = (shara_age + jaymee_age) / 2 - 3 →
  brad_age = 13 :=
by sorry

end brad_age_l1568_156848


namespace smallest_dual_base_representation_l1568_156836

/-- Represents a number in a given base -/
def representIn (n : ℕ) (base : ℕ) : List ℕ := sorry

/-- Converts a representation in a given base to a natural number -/
def fromBase (digits : List ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_dual_base_representation :
  ∃ (a b : ℕ), a > 2 ∧ b > 2 ∧
  representIn 8 a = [1, 1] ∧
  representIn 8 b = [2, 2] ∧
  (∀ (n : ℕ) (a' b' : ℕ), a' > 2 → b' > 2 →
    representIn n a' = [1, 1] →
    representIn n b' = [2, 2] →
    n ≥ 8) :=
by sorry

end smallest_dual_base_representation_l1568_156836


namespace equation_solution_l1568_156891

theorem equation_solution :
  ∃! x : ℚ, x ≠ -2 ∧ (5 * x^2 + 4 * x + 2) / (x + 2) = 5 * x - 3 := by
  sorry

end equation_solution_l1568_156891


namespace percentage_difference_l1568_156886

theorem percentage_difference (x : ℝ) : 
  (x / 100) * 170 - 0.35 * 300 = 31 → x = 80 := by
  sorry

end percentage_difference_l1568_156886


namespace x_eq_1_sufficient_not_necessary_for_quadratic_l1568_156899

theorem x_eq_1_sufficient_not_necessary_for_quadratic : 
  (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) ∧ 
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) :=
by sorry

end x_eq_1_sufficient_not_necessary_for_quadratic_l1568_156899


namespace unique_linear_equation_solution_l1568_156818

theorem unique_linear_equation_solution (m n : ℕ+) :
  ∃ (a b c : ℤ), ∀ (x y : ℕ+),
    (a * x.val + b * y.val = c) ↔ (x = m ∧ y = n) :=
sorry

end unique_linear_equation_solution_l1568_156818


namespace tank_filling_time_l1568_156812

/-- Given a tap that can fill a tank in 16 hours, and 3 additional similar taps opened after half the tank is filled, prove that the total time taken to fill the tank completely is 10 hours. -/
theorem tank_filling_time (fill_time : ℝ) (additional_taps : ℕ) : 
  fill_time = 16 → additional_taps = 3 → 
  (fill_time / 2) + (fill_time / (2 * (additional_taps + 1))) = 10 :=
by sorry

end tank_filling_time_l1568_156812


namespace smallest_value_for_x_greater_than_one_l1568_156810

theorem smallest_value_for_x_greater_than_one (x : ℝ) (hx : x > 1) :
  (1 / x < x) ∧ (1 / x < x^2) ∧ (1 / x < 2*x) ∧ (1 / x < Real.sqrt x) :=
by sorry

end smallest_value_for_x_greater_than_one_l1568_156810


namespace quadratic_equation_unique_solution_l1568_156859

theorem quadratic_equation_unique_solution (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b > 0 ∧ 
   ∃! x : ℝ, x^2 + (b + 1/b) * x + c = 0) ↔ 
  (c = (5 + Real.sqrt 21) / 2 ∨ c = (5 - Real.sqrt 21) / 2) :=
sorry

end quadratic_equation_unique_solution_l1568_156859


namespace f_of_5_equals_20_l1568_156861

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x

-- Theorem statement
theorem f_of_5_equals_20 : f 5 = 20 := by
  sorry

end f_of_5_equals_20_l1568_156861


namespace product_xyzw_l1568_156895

theorem product_xyzw (x y z w : ℝ) (h1 : x + 1/y = 1) (h2 : y + 1/z + w = 1) (h3 : w = 2) (h4 : y ≠ 0) :
  x * y * z * w = -2 * y^2 + 2 * y := by
  sorry

end product_xyzw_l1568_156895


namespace quadratic_solution_and_sum_l1568_156892

theorem quadratic_solution_and_sum (x : ℝ) : 
  x^2 + 14*x = 96 → 
  ∃ (a b : ℕ), 
    (x = Real.sqrt a - b) ∧ 
    (x > 0) ∧ 
    (a = 145) ∧ 
    (b = 7) ∧ 
    (a + b = 152) := by
  sorry

end quadratic_solution_and_sum_l1568_156892


namespace tracy_candies_l1568_156860

theorem tracy_candies : ∃ (x : ℕ), 
  (x % 4 = 0) ∧ 
  ((3 * x / 4) % 3 = 0) ∧ 
  (x / 2 - 29 = 10) ∧ 
  x = 78 := by
  sorry

end tracy_candies_l1568_156860


namespace probability_of_triangle_in_15_gon_l1568_156884

/-- Definition of a regular 15-gon -/
def regular_15_gon : Set (ℝ × ℝ) := sorry

/-- Function to check if three segments can form a triangle with positive area -/
def can_form_triangle (s1 s2 s3 : ℝ × ℝ × ℝ × ℝ) : Prop := sorry

/-- Total number of ways to choose 3 distinct segments from a 15-gon -/
def total_choices : ℕ := Nat.choose (Nat.choose 15 2) 3

/-- Number of ways to choose 3 distinct segments that form a triangle -/
def valid_choices : ℕ := sorry

theorem probability_of_triangle_in_15_gon :
  (valid_choices : ℚ) / total_choices = 163 / 455 := by sorry

end probability_of_triangle_in_15_gon_l1568_156884


namespace lottery_win_probability_l1568_156875

def megaBallCount : ℕ := 27
def winnerBallCount : ℕ := 44
def winnerBallPick : ℕ := 5

theorem lottery_win_probability :
  (1 : ℚ) / megaBallCount * (1 : ℚ) / Nat.choose winnerBallCount winnerBallPick = 1 / 29322216 :=
by sorry

end lottery_win_probability_l1568_156875


namespace max_cube_sum_on_sphere_l1568_156876

theorem max_cube_sum_on_sphere (x y z : ℝ) (h : x^2 + y^2 + z^2 = 9) :
  x^3 + y^3 + z^3 ≤ 27 ∧ ∃ a b c : ℝ, a^2 + b^2 + c^2 = 9 ∧ a^3 + b^3 + c^3 = 27 :=
by sorry

end max_cube_sum_on_sphere_l1568_156876


namespace absolute_value_inequality_solution_set_l1568_156817

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 1| < 3} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end absolute_value_inequality_solution_set_l1568_156817


namespace dave_won_ten_tickets_l1568_156800

/-- Calculates the number of tickets Dave won later at the arcade --/
def tickets_won_later (initial_tickets : ℕ) (spent_tickets : ℕ) (final_tickets : ℕ) : ℕ :=
  final_tickets - (initial_tickets - spent_tickets)

/-- Proves that Dave won 10 tickets later at the arcade --/
theorem dave_won_ten_tickets :
  tickets_won_later 11 5 16 = 10 := by
  sorry

end dave_won_ten_tickets_l1568_156800


namespace circle_chord_problem_l1568_156827

-- Define the circle C
def circle_C (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*a*y + a^2 - 24 = 0

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop :=
  2*x - y = 0

-- Define the line l
def line_l (x y m : ℝ) : Prop :=
  (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Theorem statement
theorem circle_chord_problem :
  ∃ (a : ℝ),
    (∀ x y, circle_C x y a → ∃ x₀ y₀, center_line x₀ y₀ ∧ (x - x₀)^2 + (y - y₀)^2 = 25) ∧
    a = 2 ∧
    (∀ m, ∃ chord_length,
      chord_length = Real.sqrt (4 * (25 - 5)) ∧
      (∀ x y, circle_C x y a ∧ line_l x y m →
        ∃ l, l ≤ chord_length ∧ l^2 = (x - 3)^2 + (y - 1)^2)) :=
by sorry

end circle_chord_problem_l1568_156827


namespace susan_decade_fraction_l1568_156845

/-- Represents the collection of quarters Susan has -/
structure QuarterCollection where
  total : ℕ
  decade_count : ℕ

/-- The fraction of quarters representing states that joined the union in a specific decade -/
def decade_fraction (c : QuarterCollection) : ℚ :=
  c.decade_count / c.total

/-- Susan's collection of quarters -/
def susan_collection : QuarterCollection :=
  { total := 22, decade_count := 7 }

theorem susan_decade_fraction :
  decade_fraction susan_collection = 7 / 22 := by
  sorry

end susan_decade_fraction_l1568_156845


namespace daily_tylenol_intake_l1568_156825

def tablets_per_dose : ℕ := 2
def mg_per_tablet : ℕ := 375
def hours_between_doses : ℕ := 6
def hours_per_day : ℕ := 24

theorem daily_tylenol_intake :
  tablets_per_dose * mg_per_tablet * (hours_per_day / hours_between_doses) = 3000 := by
  sorry

end daily_tylenol_intake_l1568_156825


namespace a_value_proof_l1568_156862

theorem a_value_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end a_value_proof_l1568_156862


namespace video_game_earnings_l1568_156834

def total_games : ℕ := 16
def non_working_games : ℕ := 8
def price_per_game : ℕ := 7

theorem video_game_earnings : 
  (total_games - non_working_games) * price_per_game = 56 := by
  sorry

end video_game_earnings_l1568_156834


namespace girls_attending_event_l1568_156858

theorem girls_attending_event (total_students : ℕ) (total_attendees : ℕ) 
  (girls : ℕ) (boys : ℕ) (h1 : total_students = 1500) 
  (h2 : total_attendees = 900) (h3 : girls + boys = total_students) 
  (h4 : (3 * girls) / 5 + (2 * boys) / 3 = total_attendees) : 
  (3 * girls) / 5 = 900 := by
  sorry

end girls_attending_event_l1568_156858


namespace rocky_first_round_knockouts_l1568_156832

def total_fights : ℕ := 190
def knockout_percentage : ℚ := 1/2
def first_round_knockout_percentage : ℚ := 1/5

theorem rocky_first_round_knockouts :
  (total_fights : ℚ) * knockout_percentage * first_round_knockout_percentage = 19 := by
  sorry

end rocky_first_round_knockouts_l1568_156832


namespace line_l_equation_tangent_circle_a_l1568_156854

-- Define the lines and circle
def l1 (x y : ℝ) : Prop := 2 * x - y = 1
def l2 (x y : ℝ) : Prop := x + 2 * y = 3
def l3 (x y : ℝ) : Prop := x - y + 1 = 0
def C (x y a : ℝ) : Prop := (x - a)^2 + y^2 = 8 ∧ a > 0

-- Define the intersection point P
def P : ℝ × ℝ := (1, 1)

-- Define line l
def l (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statements
theorem line_l_equation :
  (∀ x y : ℝ, l1 x y ∧ l2 x y → (x, y) = P) →
  (∀ x y : ℝ, l x y → l3 ((x + 2) / 2) ((2 - x) / 2)) →
  ∀ x y : ℝ, l x y ↔ x + y - 2 = 0 := by sorry

theorem tangent_circle_a :
  (∀ x y : ℝ, l x y → C x y 6) →
  (∀ x y a : ℝ, l x y → C x y a → a = 6) := by sorry

end line_l_equation_tangent_circle_a_l1568_156854


namespace polynomial_factorization_l1568_156839

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 6*x + 8) + (x^2 + 5*x - 7) = (x^2 + 5*x + 2) * (x^2 + 5*x + 9) := by
  sorry

end polynomial_factorization_l1568_156839


namespace triangle_cannot_have_two_right_angles_l1568_156822

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ

-- Define properties of a triangle
def Triangle.sumOfAngles (t : Triangle) : ℝ := t.angles 0 + t.angles 1 + t.angles 2

-- Define a right angle
def rightAngle : ℝ := 90

-- Theorem: A triangle cannot have two right angles
theorem triangle_cannot_have_two_right_angles (t : Triangle) :
  (t.angles 0 = rightAngle ∧ t.angles 1 = rightAngle) →
  t.sumOfAngles ≠ 180 :=
sorry

end triangle_cannot_have_two_right_angles_l1568_156822


namespace truck_fuel_efficiency_l1568_156856

theorem truck_fuel_efficiency 
  (distance : ℝ) 
  (current_gas : ℝ) 
  (additional_gas : ℝ) 
  (h1 : distance = 90) 
  (h2 : current_gas = 12) 
  (h3 : additional_gas = 18) : 
  distance / (current_gas + additional_gas) = 3 := by
sorry

end truck_fuel_efficiency_l1568_156856


namespace problem_statement_l1568_156813

theorem problem_statement : 
  (∃ x : ℝ, x - x + 1 ≥ 0) ∧ ¬(∀ a b : ℝ, a^2 < b^2 → a < b) := by
  sorry

end problem_statement_l1568_156813


namespace binomial_10_9_l1568_156874

theorem binomial_10_9 : (10 : ℕ).choose 9 = 10 := by sorry

end binomial_10_9_l1568_156874


namespace prime_plus_two_implies_divisible_by_six_l1568_156882

theorem prime_plus_two_implies_divisible_by_six (p : ℤ) : 
  Prime p → p > 3 → Prime (p + 2) → (6 : ℤ) ∣ (p + 1) := by
  sorry

end prime_plus_two_implies_divisible_by_six_l1568_156882


namespace subset_sum_modulo_l1568_156863

theorem subset_sum_modulo (N : ℕ) (A : Finset ℕ) :
  A.card = N →
  A ⊆ Finset.range (N^2) →
  ∃ (B : Finset ℕ), 
    B.card = N ∧ 
    B ⊆ Finset.range (N^2) ∧ 
    ((A.product B).image (λ (p : ℕ × ℕ) => (p.1 + p.2) % (N^2))).card ≥ N^2 / 2 :=
by sorry

end subset_sum_modulo_l1568_156863


namespace second_year_students_l1568_156826

/-- Represents the number of students in each year and the total number of students. -/
structure SchoolPopulation where
  firstYear : ℕ
  secondYear : ℕ
  thirdYear : ℕ
  total : ℕ

/-- Represents the sample size and the number of first-year students in the sample. -/
structure Sample where
  size : ℕ
  firstYearSample : ℕ

/-- 
Proves that given the conditions of the problem, the number of second-year students is 300.
-/
theorem second_year_students 
  (school : SchoolPopulation)
  (sample : Sample)
  (h1 : school.firstYear = 450)
  (h2 : school.thirdYear = 250)
  (h3 : sample.size = 60)
  (h4 : sample.firstYearSample = 27)
  (h5 : (school.firstYear : ℚ) / school.total = sample.firstYearSample / sample.size) :
  school.secondYear = 300 := by
  sorry

#check second_year_students

end second_year_students_l1568_156826


namespace intersection_chord_length_l1568_156853

/-- The line L: 3x - y - 6 = 0 -/
def line_L (x y : ℝ) : Prop := 3 * x - y - 6 = 0

/-- The circle C: x^2 + y^2 - 2x - 4y = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

/-- The length of the chord AB formed by the intersection of line L and circle C -/
noncomputable def chord_length : ℝ := Real.sqrt 10

theorem intersection_chord_length :
  chord_length = Real.sqrt 10 :=
by sorry

end intersection_chord_length_l1568_156853


namespace quadratic_expression_value_l1568_156870

theorem quadratic_expression_value (x : ℝ) (h : x^2 + 2*x - 2 = 0) : x*(x+2) + 3 = 5 := by
  sorry

end quadratic_expression_value_l1568_156870


namespace distance_from_negative_two_l1568_156878

-- Define the distance function on the real number line
def distance (x y : ℝ) : ℝ := |x - y|

-- Theorem statement
theorem distance_from_negative_two :
  ∀ x : ℝ, distance x (-2) = 3 ↔ x = -5 ∨ x = 1 := by
  sorry

end distance_from_negative_two_l1568_156878


namespace tourist_walking_speed_l1568_156815

/-- Represents the problem of calculating tourist walking speed -/
def TouristWalkingSpeedProblem (scheduled_arrival : ℝ) (actual_arrival : ℝ) (early_arrival : ℝ) (bus_speed : ℝ) : Prop :=
  ∃ (walking_speed : ℝ),
    walking_speed > 0 ∧
    scheduled_arrival > actual_arrival ∧
    early_arrival > 0 ∧
    bus_speed > 0 ∧
    let time_diff := scheduled_arrival - actual_arrival
    let encounter_time := time_diff - early_arrival
    let bus_travel_time := early_arrival / 2
    let distance := bus_speed * bus_travel_time
    walking_speed = distance / encounter_time ∧
    walking_speed = 5

/-- The main theorem stating the solution to the tourist walking speed problem -/
theorem tourist_walking_speed :
  TouristWalkingSpeedProblem 5 3.25 0.25 60 :=
by
  sorry


end tourist_walking_speed_l1568_156815


namespace first_last_checkpoint_distance_l1568_156879

-- Define the marathon parameters
def marathon_length : ℝ := 26
def num_checkpoints : ℕ := 4
def distance_between_checkpoints : ℝ := 6

-- Theorem statement
theorem first_last_checkpoint_distance :
  let total_checkpoint_distance := (num_checkpoints - 1 : ℝ) * distance_between_checkpoints
  let remaining_distance := marathon_length - total_checkpoint_distance
  let first_last_distance := remaining_distance / 2
  first_last_distance = 1 := by sorry

end first_last_checkpoint_distance_l1568_156879


namespace cube_difference_1234567_l1568_156828

theorem cube_difference_1234567 : ∃ a b : ℤ, a^3 - b^3 = 1234567 := by
  sorry

end cube_difference_1234567_l1568_156828


namespace equality_from_divisibility_l1568_156887

theorem equality_from_divisibility (a b : ℕ+) (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b := by
  sorry

end equality_from_divisibility_l1568_156887


namespace quadratic_equation_roots_l1568_156885

theorem quadratic_equation_roots (α β : ℝ) (h1 : α + β = 5) (h2 : α * β = 6) :
  (α ^ 2 - 5 * α + 6 = 0) ∧ (β ^ 2 - 5 * β + 6 = 0) := by
  sorry

end quadratic_equation_roots_l1568_156885


namespace ellipse_equation_l1568_156840

/-- Given an ellipse C₁ and a circle C₂, prove that C₁ has the equation x²/4 + y² = 1 -/
theorem ellipse_equation (a b : ℝ) (P : ℝ × ℝ) :
  a > b ∧ b > 0 ∧  -- a > b > 0
  P = (0, -1) ∧  -- P(0,-1) is a vertex of C₁
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 →  -- Equation of C₁
  2 * a = 4 ∧  -- Major axis of C₁ is diameter of C₂
  ∀ x y : ℝ, x^2 + y^2 = 4 →  -- Equation of C₂
  ∀ x y : ℝ, x^2 / 4 + y^2 = 1  -- Equation of C₁ we want to prove
  := by sorry

end ellipse_equation_l1568_156840


namespace complex_magnitude_problem_l1568_156802

theorem complex_magnitude_problem (x y : ℝ) (h : (x + y * Complex.I) * Complex.I = 1 + Complex.I) :
  Complex.abs (x + 2 * y * Complex.I) = Real.sqrt 5 := by
  sorry

end complex_magnitude_problem_l1568_156802


namespace expand_product_l1568_156881

theorem expand_product (x : ℝ) : -3 * (2 * x + 4) * (x - 7) = -6 * x^2 + 30 * x + 84 := by
  sorry

end expand_product_l1568_156881


namespace constant_speed_journey_time_l1568_156883

/-- Given a constant speed journey, prove the total travel time -/
theorem constant_speed_journey_time 
  (total_distance : ℝ) 
  (initial_distance : ℝ) 
  (initial_time : ℝ) 
  (h1 : total_distance = 400) 
  (h2 : initial_distance = 100) 
  (h3 : initial_time = 1) 
  (h4 : initial_distance / initial_time = (total_distance - initial_distance) / (total_time - initial_time)) : 
  total_time = 4 :=
by
  sorry

#check constant_speed_journey_time

end constant_speed_journey_time_l1568_156883


namespace equation_one_integral_root_l1568_156819

theorem equation_one_integral_root :
  ∃! (x : ℤ), x - 9 / (x - 5 : ℚ) = 4 - 9 / (x - 5 : ℚ) :=
by sorry

end equation_one_integral_root_l1568_156819


namespace system_solution_l1568_156816

theorem system_solution (x y : ℝ) : 
  (x + y = 1 ∧ 2*x + y = 5) → (x = 4 ∧ y = -3) := by
  sorry

end system_solution_l1568_156816


namespace factor_expression_l1568_156849

theorem factor_expression (b : ℝ) : 26 * b^2 + 78 * b = 26 * b * (b + 3) := by
  sorry

end factor_expression_l1568_156849


namespace triangle_proof_l1568_156864

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The median from vertex A to side BC -/
def median (t : Triangle) : ℝ := sorry

/-- The area of the triangle -/
def area (t : Triangle) : ℝ := sorry

theorem triangle_proof (t : Triangle) 
  (h1 : 2 * t.b * Real.cos t.A - Real.sqrt 3 * t.c * Real.cos t.A = Real.sqrt 3 * t.a * Real.cos t.C)
  (h2 : t.B = π / 6)
  (h3 : median t = Real.sqrt 7) :
  t.A = π / 6 ∧ area t = Real.sqrt 3 := by sorry

end triangle_proof_l1568_156864


namespace max_sum_of_goods_l1568_156814

theorem max_sum_of_goods (a b : ℕ+) : 
  7 * a + 19 * b = 213 →
  ∀ x y : ℕ+, 7 * x + 19 * y = 213 → a + b ≥ x + y →
  a + b = 27 := by
sorry

end max_sum_of_goods_l1568_156814


namespace reporters_covering_local_politics_l1568_156804

theorem reporters_covering_local_politics
  (total_reporters : ℕ)
  (h1 : total_reporters > 0)
  (politics_not_local : Real)
  (h2 : politics_not_local = 0.4)
  (not_politics : Real)
  (h3 : not_politics = 0.7) :
  (1 - politics_not_local) * (1 - not_politics) * 100 = 18 := by
  sorry

end reporters_covering_local_politics_l1568_156804


namespace range_of_m_l1568_156831

/-- A function f(x) = x^2 - 2x + m where x is a real number -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m

/-- The theorem stating the range of m given the conditions -/
theorem range_of_m (m : ℝ) : 
  (∃ x y, x ≠ y ∧ f m x = 0 ∧ f m y = 0) → 
  (∀ x, f m (1 - x) ≥ -1) → 
  m ∈ Set.Icc 0 1 := by
  sorry

end range_of_m_l1568_156831


namespace volume_to_surface_area_ratio_l1568_156843

/-- Represents a unit cube -/
structure UnitCube where
  volume : ℝ := 1
  surfaceArea : ℝ := 6

/-- Represents the custom shape described in the problem -/
structure CustomShape where
  baseCubes : Fin 5 → UnitCube
  topCube : UnitCube
  bottomCube : UnitCube

/-- Calculates the total volume of the CustomShape -/
def totalVolume (shape : CustomShape) : ℝ :=
  7  -- 5 base cubes + 1 top cube + 1 bottom cube

/-- Calculates the total surface area of the CustomShape -/
def totalSurfaceArea (shape : CustomShape) : ℝ :=
  28  -- As calculated in the problem

/-- The main theorem to be proved -/
theorem volume_to_surface_area_ratio (shape : CustomShape) :
  totalVolume shape / totalSurfaceArea shape = 1 / 4 := by
  sorry


end volume_to_surface_area_ratio_l1568_156843


namespace abcd_over_hife_value_l1568_156805

theorem abcd_over_hife_value (a b c d e f g h i : ℝ) 
  (hab : a / b = 1 / 3)
  (hbc : b / c = 2)
  (hcd : c / d = 1 / 2)
  (hde : d / e = 3)
  (hef : e / f = 1 / 10)
  (hfg : f / g = 3 / 4)
  (hgh : g / h = 1 / 5)
  (hhi : h / i = 5)
  (h_nonzero : b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧ i ≠ 0) :
  a * b * c * d / (h * i * f * e) = 432 / 25 := by
  sorry

end abcd_over_hife_value_l1568_156805


namespace least_with_twelve_factors_l1568_156820

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- n is the least positive integer with exactly k positive factors -/
def is_least_with_factors (n : ℕ+) (k : ℕ) : Prop :=
  num_factors n = k ∧ ∀ m : ℕ+, m < n → num_factors m ≠ k

theorem least_with_twelve_factors :
  is_least_with_factors 96 12 := by sorry

end least_with_twelve_factors_l1568_156820


namespace janes_drawing_paper_l1568_156830

/-- The number of old, brown sheets of drawing paper Jane has. -/
def brown_sheets : ℕ := 28

/-- The number of old, yellow sheets of drawing paper Jane has. -/
def yellow_sheets : ℕ := 27

/-- The total number of sheets of drawing paper Jane has. -/
def total_sheets : ℕ := brown_sheets + yellow_sheets

theorem janes_drawing_paper : total_sheets = 55 := by
  sorry

end janes_drawing_paper_l1568_156830


namespace ratio_expression_value_l1568_156844

theorem ratio_expression_value (A B C : ℚ) (h : A/B = 3/2 ∧ B/C = 2/6) :
  (4*A - 3*B) / (5*C + 2*A) = 1/4 := by sorry

end ratio_expression_value_l1568_156844


namespace book_pages_theorem_l1568_156871

theorem book_pages_theorem :
  ∀ (book1 book2 book3 : ℕ),
    (2 * book1) / 3 - (book1 / 3) = 20 →
    (3 * book2) / 5 - (2 * book2) / 5 = 15 →
    (3 * book3) / 4 - (book3 / 4) = 30 →
    book1 = 60 ∧ book2 = 75 ∧ book3 = 60 := by
  sorry

end book_pages_theorem_l1568_156871


namespace joes_money_from_mother_l1568_156873

def notebook_cost : ℕ := 4
def book_cost : ℕ := 7
def notebooks_bought : ℕ := 7
def books_bought : ℕ := 2
def money_left : ℕ := 14

theorem joes_money_from_mother : 
  notebook_cost * notebooks_bought + book_cost * books_bought + money_left = 56 := by
  sorry

end joes_money_from_mother_l1568_156873


namespace lemonade_proportion_l1568_156847

/-- Given that 24 lemons make 32 gallons of lemonade, proves that 3 lemons make 4 gallons -/
theorem lemonade_proportion :
  (24 : ℚ) / 32 = 3 / 4 := by sorry

end lemonade_proportion_l1568_156847


namespace johns_order_cost_l1568_156803

/-- Calculates the discounted price of an order given the store's discount policy and purchase details. -/
def discountedPrice (itemPrice : ℕ) (itemCount : ℕ) (discountThreshold : ℕ) (discountRate : ℚ) : ℚ :=
  let totalPrice := itemPrice * itemCount
  let discountableAmount := max (totalPrice - discountThreshold) 0
  let discount := (discountableAmount : ℚ) * discountRate
  (totalPrice : ℚ) - discount

/-- Theorem stating that John's order costs $1360 after the discount. -/
theorem johns_order_cost :
  discountedPrice 200 7 1000 (1 / 10) = 1360 := by
  sorry

end johns_order_cost_l1568_156803


namespace shoe_difference_l1568_156838

/-- Scott's number of shoe pairs -/
def scott_shoes : ℕ := 7

/-- Anthony's number of shoe pairs -/
def anthony_shoes : ℕ := 3 * scott_shoes

/-- Jim's number of shoe pairs -/
def jim_shoes : ℕ := anthony_shoes - 2

/-- The difference between Anthony's and Jim's shoe pairs -/
theorem shoe_difference : anthony_shoes - jim_shoes = 2 := by
  sorry

end shoe_difference_l1568_156838


namespace tetrahedron_stripe_probability_l1568_156829

/-- Represents the orientation of a stripe on a face of a tetrahedron -/
inductive StripeOrientation
  | First
  | Second
  | Third

/-- Represents the configuration of stripes on a tetrahedron -/
def TetrahedronStripes := Fin 4 → StripeOrientation

/-- Predicate to check if a given configuration of stripes forms a continuous stripe around the tetrahedron -/
def isContinuousStripe (config : TetrahedronStripes) : Prop := sorry

/-- The total number of possible stripe configurations -/
def totalConfigurations : ℕ := 3^4

/-- The number of configurations that form a continuous stripe -/
def favorableConfigurations : ℕ := 18

/-- Theorem stating the probability of a continuous stripe encircling the tetrahedron -/
theorem tetrahedron_stripe_probability :
  (favorableConfigurations : ℚ) / totalConfigurations = 2 / 9 := by sorry

end tetrahedron_stripe_probability_l1568_156829


namespace mirror_area_l1568_156869

/-- The area of a rectangular mirror with a frame -/
theorem mirror_area (overall_length overall_width frame_width : ℝ) 
  (h1 : overall_length = 100)
  (h2 : overall_width = 50)
  (h3 : frame_width = 8) : 
  (overall_length - 2 * frame_width) * (overall_width - 2 * frame_width) = 2856 := by
  sorry

end mirror_area_l1568_156869


namespace shifted_linear_to_proportional_l1568_156801

/-- A linear function y = ax + b -/
structure LinearFunction where
  a : ℝ
  b : ℝ

/-- Shift a linear function to the left by h units -/
def shift_left (f : LinearFunction) (h : ℝ) : LinearFunction :=
  { a := f.a, b := f.a * h + f.b }

/-- A function is directly proportional if it passes through the origin -/
def is_directly_proportional (f : LinearFunction) : Prop :=
  f.b = 0

/-- The main theorem -/
theorem shifted_linear_to_proportional (m : ℝ) : 
  let f : LinearFunction := { a := 2, b := m - 1 }
  let shifted_f := shift_left f 3
  is_directly_proportional shifted_f → m = -5 := by
sorry

end shifted_linear_to_proportional_l1568_156801


namespace triangle_side_length_l1568_156857

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = π/3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 :=
sorry

end triangle_side_length_l1568_156857


namespace circle_radius_is_zero_l1568_156806

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, -2)

/-- Theorem stating that the radius of the circle is 0 -/
theorem circle_radius_is_zero :
  ∀ x y : ℝ, circle_equation x y →
  (x - circle_center.1)^2 + (y - circle_center.2)^2 = 0 :=
sorry

end circle_radius_is_zero_l1568_156806


namespace table_formula_proof_l1568_156809

theorem table_formula_proof : 
  (∀ (x y : ℕ), (x = 1 ∧ y = 3) ∨ (x = 2 ∧ y = 7) ∨ (x = 3 ∧ y = 13) ∨ 
   (x = 4 ∧ y = 21) ∨ (x = 5 ∧ y = 31) → y = x^2 + x + 1) :=
by sorry

end table_formula_proof_l1568_156809


namespace exactly_one_false_l1568_156846

theorem exactly_one_false :
  (∀ a b : ℝ, a ≥ b ∧ b > -1 → a / (1 + a) ≥ b / (1 + b)) ∧
  (∀ m n : ℕ+, m ≤ n → Real.sqrt (m * (n - m)) ≤ n / 2) ∧
  ¬(∀ a b x₁ y₁ : ℝ, x₁^2 + y₁^2 = 9 ∧ (a - x₁)^2 + (b - y₁)^2 = 1 →
    ∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 9 ∧ (p.1 - a)^2 + (p.2 - b)^2 = 1) :=
by sorry

end exactly_one_false_l1568_156846


namespace bus_passengers_l1568_156835

theorem bus_passengers (initial : ℕ) (got_on : ℕ) (got_off : ℕ) (final : ℕ) : 
  got_on = 7 → got_off = 9 → final = 26 → initial + got_on - got_off = final → initial = 28 := by
sorry

end bus_passengers_l1568_156835


namespace min_formula_l1568_156851

theorem min_formula (a b : ℝ) : min a b = (a + b - Real.sqrt ((a - b)^2)) / 2 := by
  sorry

end min_formula_l1568_156851


namespace problem_solution_l1568_156880

noncomputable def equation (x a : ℝ) : Prop := Real.arctan (x / 2) + Real.arctan (2 - x) = a

theorem problem_solution :
  (∀ x : ℝ, equation x (π / 4) → Real.arccos (x / 2) = 2*π/3 ∨ Real.arccos (x / 2) = 0) ∧
  (∀ a : ℝ, (∃ x : ℝ, equation x a) → a ∈ Set.Icc (Real.arctan (1 / (-2 * Real.sqrt 10 - 6))) (Real.arctan (1 / (2 * Real.sqrt 10 - 6)))) ∧
  (∀ a : ℝ, (∃ α β : ℝ, α ≠ β ∧ α ∈ Set.Icc 5 15 ∧ β ∈ Set.Icc 5 15 ∧ equation α a ∧ equation β a) →
    (∀ γ δ : ℝ, γ ≠ δ ∧ γ ∈ Set.Icc 5 15 ∧ δ ∈ Set.Icc 5 15 ∧ equation γ a ∧ equation δ a → γ + δ ≤ 19)) :=
by sorry

end problem_solution_l1568_156880


namespace ab_positive_necessary_not_sufficient_l1568_156872

theorem ab_positive_necessary_not_sufficient (a b : ℝ) :
  (∀ a b, b / a + a / b > 2 → a * b > 0) ∧
  (∃ a b, a * b > 0 ∧ ¬(b / a + a / b > 2)) :=
sorry

end ab_positive_necessary_not_sufficient_l1568_156872


namespace train_speed_excluding_stoppages_l1568_156807

/-- The speed of a train excluding stoppages, given its speed including stoppages and stop duration. -/
theorem train_speed_excluding_stoppages 
  (speed_with_stops : ℝ) 
  (stop_duration : ℝ) 
  (h1 : speed_with_stops = 32) 
  (h2 : stop_duration = 20) : 
  speed_with_stops * 60 / (60 - stop_duration) = 48 := by
  sorry

end train_speed_excluding_stoppages_l1568_156807


namespace sqrt_problem_proportional_function_l1568_156841

-- Problem 1
theorem sqrt_problem : Real.sqrt 18 - Real.sqrt 24 / Real.sqrt 3 = Real.sqrt 2 := by sorry

-- Problem 2
theorem proportional_function (f : ℝ → ℝ) (h1 : ∀ x y, f (x + y) = f x + f y) (h2 : f 1 = 2) :
  ∀ x, f x = 2 * x := by sorry

end sqrt_problem_proportional_function_l1568_156841


namespace fraction_sum_equation_l1568_156833

theorem fraction_sum_equation (x : ℝ) : 
  (7 / (x - 2) + x / (2 - x) = 4) → x = 3 := by
  sorry

end fraction_sum_equation_l1568_156833


namespace fraction_comparison_l1568_156821

theorem fraction_comparison (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end fraction_comparison_l1568_156821


namespace convex_shape_volume_is_half_l1568_156889

/-- A cube with midlines of each face divided in a 1:3 ratio -/
structure DividedCube where
  /-- The volume of the original cube -/
  volume : ℝ
  /-- The ratio in which the midlines are divided -/
  divisionRatio : ℝ
  /-- Assumption that the division ratio is 1:3 -/
  ratio_is_one_three : divisionRatio = 1/3

/-- The volume of the convex shape formed by the points dividing the midlines -/
def convexShapeVolume (c : DividedCube) : ℝ := sorry

/-- Theorem stating that the volume of the convex shape is half the volume of the cube -/
theorem convex_shape_volume_is_half (c : DividedCube) : 
  convexShapeVolume c = c.volume / 2 := by sorry

end convex_shape_volume_is_half_l1568_156889


namespace polynomial_irreducibility_l1568_156855

theorem polynomial_irreducibility (a b c : ℤ) : 
  (0 < |c| ∧ |c| < |b| ∧ |b| < |a|) →
  (∀ x : ℤ, Irreducible (x * (x - a) * (x - b) * (x - c) + 1)) ↔
  (a ≠ 1 ∨ b ≠ 2 ∨ c ≠ 3) ∧ (a ≠ -1 ∨ b ≠ -2 ∨ c ≠ -3) :=
by sorry

end polynomial_irreducibility_l1568_156855


namespace sqrt_450_simplification_l1568_156897

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_simplification_l1568_156897
