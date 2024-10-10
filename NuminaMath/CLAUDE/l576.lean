import Mathlib

namespace girls_entered_classroom_l576_57662

theorem girls_entered_classroom (initial_boys initial_girls boys_left final_total : ℕ) :
  initial_boys = 5 →
  initial_girls = 4 →
  boys_left = 3 →
  final_total = 8 →
  ∃ girls_entered : ℕ, girls_entered = 2 ∧
    final_total = (initial_boys - boys_left) + (initial_girls + girls_entered) :=
by sorry

end girls_entered_classroom_l576_57662


namespace ball_bounce_distance_l576_57696

/-- A ball rolling on a half circular track and bouncing on the floor -/
theorem ball_bounce_distance 
  (R : ℝ) -- radius of the half circular track
  (v : ℝ) -- velocity of the ball when leaving the track
  (g : ℝ) -- acceleration due to gravity
  (h : R > 0) -- radius is positive
  (hv : v > 0) -- velocity is positive
  (hg : g > 0) -- gravity is positive
  : ∃ (d : ℝ), d = 2 * R - (2 * v / 3) * Real.sqrt (R / g) :=
by sorry

end ball_bounce_distance_l576_57696


namespace vertex_in_second_quadrant_l576_57673

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x + 1)^2 + 2

-- Define the vertex of the quadratic function
def vertex : ℝ × ℝ := (-1, 2)

-- Define what it means for a point to be in the second quadrant
def in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Theorem statement
theorem vertex_in_second_quadrant :
  in_second_quadrant vertex := by sorry

end vertex_in_second_quadrant_l576_57673


namespace set_operations_l576_57693

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {2, 4, 5}

-- Define set B
def B : Set Nat := {1, 2, 5}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {2, 5}) ∧ (A ∪ (U \ B) = {2, 3, 4, 5, 6}) := by sorry

end set_operations_l576_57693


namespace balloon_theorem_l576_57653

/-- Represents a person's balloon collection -/
structure BalloonCollection where
  count : ℕ
  cost : ℕ

/-- Calculates the total number of balloons from a list of balloon collections -/
def totalBalloons (collections : List BalloonCollection) : ℕ :=
  collections.map (·.count) |>.sum

/-- Calculates the total cost of balloons from a list of balloon collections -/
def totalCost (collections : List BalloonCollection) : ℕ :=
  collections.map (fun c => c.count * c.cost) |>.sum

theorem balloon_theorem (fred sam mary susan tom : BalloonCollection)
    (h1 : fred = ⟨5, 3⟩)
    (h2 : sam = ⟨6, 4⟩)
    (h3 : mary = ⟨7, 5⟩)
    (h4 : susan = ⟨4, 6⟩)
    (h5 : tom = ⟨10, 2⟩) :
    let collections := [fred, sam, mary, susan, tom]
    totalBalloons collections = 32 ∧ totalCost collections = 118 := by
  sorry

end balloon_theorem_l576_57653


namespace ceiling_lights_difference_l576_57615

theorem ceiling_lights_difference (medium large small : ℕ) : 
  medium = 12 →
  large = 2 * medium →
  small + 2 * medium + 3 * large = 118 →
  small - medium = 10 := by
sorry

end ceiling_lights_difference_l576_57615


namespace transform_f_to_g_l576_57648

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := (x + 1)^2 + 3

/-- The function after transformation -/
def g (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- Right shift transformation -/
def shift_right (h : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := fun x ↦ h (x - a)

/-- Down shift transformation -/
def shift_down (h : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := fun x ↦ h x - b

/-- Theorem stating that the transformation of f results in g -/
theorem transform_f_to_g : shift_down (shift_right f 2) 1 = g := by
  sorry

end transform_f_to_g_l576_57648


namespace walking_time_calculation_l576_57689

/-- Proves that given a distance that takes 40 minutes to cover at a speed of 16.5 kmph,
    it will take 165 minutes to cover the same distance at a speed of 4 kmph. -/
theorem walking_time_calculation (distance : ℝ) : 
  distance = 16.5 * (40 / 60) → distance / 4 * 60 = 165 := by
  sorry

end walking_time_calculation_l576_57689


namespace three_line_hexagon_angle_sum_l576_57671

/-- A hexagon formed by the intersection of three lines -/
structure ThreeLineHexagon where
  -- Define the six angles of the hexagon
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  angle5 : ℝ
  angle6 : ℝ

/-- The sum of angles in a hexagon formed by three intersecting lines is 360° -/
theorem three_line_hexagon_angle_sum (h : ThreeLineHexagon) : 
  h.angle1 + h.angle2 + h.angle3 + h.angle4 + h.angle5 + h.angle6 = 360 := by
  sorry

end three_line_hexagon_angle_sum_l576_57671


namespace money_sharing_l576_57680

theorem money_sharing (total : ℝ) (maggie_share : ℝ) : 
  maggie_share = 0.75 * total ∧ maggie_share = 4500 → total = 6000 :=
by sorry

end money_sharing_l576_57680


namespace power_product_equality_l576_57677

theorem power_product_equality : 2^4 * 3^2 * 5^2 * 11 = 39600 := by
  sorry

end power_product_equality_l576_57677


namespace solution_set_of_inequality_l576_57621

open Set

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State the theorem
theorem solution_set_of_inequality 
  (h1 : ∀ x, f' x < f x) 
  (h2 : f 1 = Real.exp 1) :
  {x : ℝ | f (Real.log x) > x} = Ioo 0 (Real.exp 1) := by sorry

end solution_set_of_inequality_l576_57621


namespace partial_fraction_sum_zero_l576_57668

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E : ℝ) :
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) =
  A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5) →
  A + B + C + D + E = 0 := by
sorry

end partial_fraction_sum_zero_l576_57668


namespace cubic_polynomial_coefficient_l576_57643

theorem cubic_polynomial_coefficient (a b c d : ℝ) : 
  let g := fun x => a * x^3 + b * x^2 + c * x + d
  (g (-2) = 0) → (g 0 = 0) → (g 2 = 0) → (g 1 = 3) → b = 0 := by
  sorry

end cubic_polynomial_coefficient_l576_57643


namespace min_c_for_unique_solution_l576_57632

/-- The system of equations -/
def system (x y c : ℝ) : Prop :=
  8 * (x + 7)^4 + (y - 4)^4 = c ∧ (x + 4)^4 + 8 * (y - 7)^4 = c

/-- The existence of a unique solution for the system -/
def has_unique_solution (c : ℝ) : Prop :=
  ∃! x y, system x y c

/-- The theorem stating the minimum value of c for a unique solution -/
theorem min_c_for_unique_solution :
  ∀ c, has_unique_solution c → c ≥ 24 ∧ has_unique_solution 24 :=
sorry

end min_c_for_unique_solution_l576_57632


namespace ten_thousand_scientific_notation_l576_57679

/-- Scientific notation representation of 10,000 -/
def scientific_notation_10000 : ℝ := 1 * (10 ^ 4)

/-- Theorem stating that 10,000 is equal to its scientific notation representation -/
theorem ten_thousand_scientific_notation : 
  (10000 : ℝ) = scientific_notation_10000 := by
  sorry

end ten_thousand_scientific_notation_l576_57679


namespace current_speed_l576_57644

/-- The speed of the current given a woman's swimming times and distances -/
theorem current_speed (downstream_distance upstream_distance : ℝ) 
  (time : ℝ) (h1 : downstream_distance = 125) (h2 : upstream_distance = 60) 
  (h3 : time = 10) : ∃ (v_w v_c : ℝ), 
  downstream_distance = (v_w + v_c) * time ∧ 
  upstream_distance = (v_w - v_c) * time ∧ 
  v_c = 3.25 :=
by sorry

end current_speed_l576_57644


namespace percentage_to_pass_l576_57655

/-- Given a test with maximum marks and a student's performance, 
    calculate the percentage needed to pass the test. -/
theorem percentage_to_pass (max_marks : ℕ) (student_marks : ℕ) (fail_margin : ℕ) :
  max_marks = 300 →
  student_marks = 80 →
  fail_margin = 10 →
  (((student_marks + fail_margin : ℝ) / max_marks) * 100 : ℝ) = 30 := by
  sorry

end percentage_to_pass_l576_57655


namespace arithmetic_sequence_fifth_term_l576_57678

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_d_nonzero : d ≠ 0)
  (h_condition : a 3 + a 9 = a 10 - a 8) :
  a 5 = 0 := by
sorry

end arithmetic_sequence_fifth_term_l576_57678


namespace no_double_composition_f_l576_57681

def q : ℕ+ → ℕ+ :=
  fun n => match n with
  | 1 => 3
  | 2 => 4
  | 3 => 2
  | 4 => 1
  | _ => n

theorem no_double_composition_f (f : ℕ+ → ℕ+) :
  ¬(∀ n : ℕ+, f (f n) = q n + 2) :=
sorry

end no_double_composition_f_l576_57681


namespace quotient_rational_l576_57694

-- Define the set A as a subset of positive reals
def A : Set ℝ := {x : ℝ | x > 0}

-- Define the property that A is non-empty
axiom A_nonempty : Set.Nonempty A

-- Define the condition that for all a, b, c in A, ab + bc + ca is rational
axiom sum_rational (a b c : ℝ) (ha : a ∈ A) (hb : b ∈ A) (hc : c ∈ A) :
  ∃ (q : ℚ), (a * b + b * c + c * a : ℝ) = q

-- State the theorem to be proved
theorem quotient_rational (a b : ℝ) (ha : a ∈ A) (hb : b ∈ A) :
  ∃ (q : ℚ), (a / b : ℝ) = q :=
sorry

end quotient_rational_l576_57694


namespace complex_fraction_simplification_l576_57629

theorem complex_fraction_simplification :
  (1 + 3*Complex.I) / (1 - Complex.I) = -1 + 2*Complex.I := by
  sorry

end complex_fraction_simplification_l576_57629


namespace library_books_l576_57672

theorem library_books (borrowed : ℕ) (left : ℕ) (initial : ℕ) : 
  borrowed = 18 → left = 57 → initial = borrowed + left → initial = 75 := by
  sorry

end library_books_l576_57672


namespace boat_stream_speed_l576_57657

/-- Proves that the speed of the stream is 6 kmph given the conditions of the boat problem -/
theorem boat_stream_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ)
  (h_boat_speed : boat_speed = 8)
  (h_distance : distance = 210)
  (h_total_time : total_time = 120)
  (h_equation : (distance / (boat_speed - stream_speed)) + (distance / (boat_speed + stream_speed)) = total_time)
  : stream_speed = 6 := by
  sorry

#check boat_stream_speed

end boat_stream_speed_l576_57657


namespace sin_arccos_three_fifths_l576_57684

theorem sin_arccos_three_fifths : Real.sin (Real.arccos (3/5)) = 4/5 := by
  sorry

end sin_arccos_three_fifths_l576_57684


namespace unique_positive_solution_l576_57613

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 4) / 9 = 4 / (x - 9) := by
  sorry

end unique_positive_solution_l576_57613


namespace triangle_perimeter_range_l576_57626

theorem triangle_perimeter_range (a b c : ℝ) : 
  a = 2 → b = 7 → (a + b > c ∧ b + c > a ∧ c + a > b) → 
  14 < a + b + c ∧ a + b + c < 18 := by sorry

end triangle_perimeter_range_l576_57626


namespace trig_expression_equality_l576_57638

theorem trig_expression_equality : 
  (Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + Real.cos (160 * π / 180) * Real.cos (110 * π / 180)) /
  (Real.sin (24 * π / 180) * Real.cos (6 * π / 180) + Real.cos (156 * π / 180) * Real.cos (96 * π / 180)) =
  (1 - Real.sin (40 * π / 180)) / (1 - Real.sin (48 * π / 180)) := by
  sorry

end trig_expression_equality_l576_57638


namespace exists_multiple_with_digit_sum_l576_57631

/-- Given a natural number, return the sum of its digits -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number that is a multiple of 2007 and whose sum of digits equals 2007 -/
theorem exists_multiple_with_digit_sum :
  ∃ n : ℕ, (∃ k : ℕ, n = k * 2007) ∧ sumOfDigits n = 2007 := by sorry

end exists_multiple_with_digit_sum_l576_57631


namespace addition_verification_l576_57649

theorem addition_verification (a b s : ℝ) (h : s = a + b) : 
  (s - a = b) ∧ (s - b = a) := by
sorry

end addition_verification_l576_57649


namespace min_value_ab_l576_57659

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 2 / a + 3 / b = Real.sqrt (a * b)) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 2 / x + 3 / y = Real.sqrt (x * y) → a * b ≤ x * y :=
by
  sorry

end min_value_ab_l576_57659


namespace sufficient_but_not_necessary_l576_57651

theorem sufficient_but_not_necessary (x : ℝ) :
  ((-1 < x ∧ x < 3) → (x^2 - 5*x - 6 < 0)) ∧
  ¬((x^2 - 5*x - 6 < 0) → (-1 < x ∧ x < 3)) :=
sorry

end sufficient_but_not_necessary_l576_57651


namespace arithmetic_sequence_sum_l576_57650

theorem arithmetic_sequence_sum : 
  let a₁ : ℤ := -5  -- First term
  let d : ℤ := 3    -- Common difference
  let n : ℕ := 20   -- Number of terms
  let S := n * (2 * a₁ + (n - 1) * d) / 2  -- Sum formula for arithmetic sequence
  S = 470
  := by sorry

end arithmetic_sequence_sum_l576_57650


namespace inequalities_not_necessarily_true_l576_57666

theorem inequalities_not_necessarily_true
  (x y z a b c : ℝ)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hxa : x < a) (hyb : y > b) (hzc : z < c) :
  ∃ (x' y' z' a' b' c' : ℝ),
    x' < a' ∧ y' > b' ∧ z' < c' ∧
    ¬(x'*z' + y' < a'*z' + b') ∧
    ¬(x'*y' < a'*b') ∧
    ¬((x' + y') / z' < (a' + b') / c') ∧
    ¬(x'*y'*z' < a'*b'*c') :=
by sorry

end inequalities_not_necessarily_true_l576_57666


namespace fraction_equality_l576_57610

theorem fraction_equality (a b : ℚ) (h : a / 5 = b / 3) : (a - b) / (3 * a) = 2 / 15 := by
  sorry

end fraction_equality_l576_57610


namespace rescue_team_distribution_l576_57687

/-- The number of ways to distribute rescue teams to disaster sites. -/
def distribute_teams (total_teams : ℕ) (num_sites : ℕ) : ℕ :=
  sorry

/-- Constraint that each site gets at least one team -/
def at_least_one_each (distribution : List ℕ) : Prop :=
  sorry

/-- Constraint that site A gets at least two teams -/
def site_A_at_least_two (distribution : List ℕ) : Prop :=
  sorry

theorem rescue_team_distribution :
  ∃ (distributions : List (List ℕ)),
    (∀ d ∈ distributions,
      d.length = 3 ∧
      d.sum = 6 ∧
      at_least_one_each d ∧
      site_A_at_least_two d) ∧
    distributions.length = 360 :=
  sorry

end rescue_team_distribution_l576_57687


namespace rectangle_diagonal_l576_57611

theorem rectangle_diagonal (a b d : ℝ) : 
  a = 13 →
  a * b = 142.40786495134319 →
  d^2 = a^2 + b^2 →
  d = 17 := by sorry

end rectangle_diagonal_l576_57611


namespace watch_cost_calculation_l576_57636

/-- The cost of a watch, given the amount saved and the additional amount needed. -/
def watch_cost (saved : ℕ) (additional_needed : ℕ) : ℕ :=
  saved + additional_needed

/-- Theorem: The cost of the watch is $55, given Connie saved $39 and needs $16 more. -/
theorem watch_cost_calculation : watch_cost 39 16 = 55 := by
  sorry

end watch_cost_calculation_l576_57636


namespace equal_intercept_line_equation_l576_57628

/-- A line with equal intercepts on both coordinate axes passing through (-3, -2) -/
structure EqualInterceptLine where
  -- The slope of the line
  k : ℝ
  -- The y-intercept of the line
  b : ℝ
  -- The line passes through (-3, -2)
  point_condition : -2 = k * (-3) + b
  -- The line has equal intercepts on both axes
  equal_intercepts : k * b + b = b

/-- The equation of an EqualInterceptLine is either 2x - 3y = 0 or x + y + 5 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (∀ x y, y = l.k * x + l.b → 2 * x - 3 * y = 0) ∨
  (∀ x y, y = l.k * x + l.b → x + y + 5 = 0) :=
sorry

end equal_intercept_line_equation_l576_57628


namespace unique_solution_l576_57619

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y : ℝ, f (2 * x - 2 * y) + x = f (3 * x) - f (2 * y) + k * y

/-- The theorem stating the unique solution to the functional equation -/
theorem unique_solution :
  ∃! f : ℝ → ℝ, ∃ k : ℝ, SatisfiesEquation f k ∧ f = id ∧ k = 0 :=
sorry

end unique_solution_l576_57619


namespace space_station_cost_share_l576_57606

/-- Calculates the individual share of a project cost -/
def calculate_share (total_cost : ℕ) (total_population : ℕ) : ℚ :=
  (total_cost : ℚ) / ((total_population : ℚ) / 2)

theorem space_station_cost_share :
  let total_cost : ℕ := 50000000000 -- $50 billion in dollars
  let total_population : ℕ := 400000000 -- 400 million people
  calculate_share total_cost total_population = 250 := by sorry

end space_station_cost_share_l576_57606


namespace floor_sqrt_80_l576_57642

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end floor_sqrt_80_l576_57642


namespace second_train_speed_l576_57618

/-- Calculates the speed of the second train given the parameters of two trains passing each other --/
theorem second_train_speed 
  (train1_length : ℝ)
  (train1_speed : ℝ)
  (train2_length : ℝ)
  (time_to_cross : ℝ)
  (h1 : train1_length = 420)
  (h2 : train1_speed = 72)
  (h3 : train2_length = 640)
  (h4 : time_to_cross = 105.99152067834574)
  : ∃ (train2_speed : ℝ), train2_speed = 36 := by
  sorry

end second_train_speed_l576_57618


namespace paige_to_remainder_ratio_l576_57692

/-- Represents the number of pieces in a chocolate bar -/
def total_pieces : ℕ := 60

/-- Represents the number of pieces Michael takes -/
def michael_pieces : ℕ := total_pieces / 2

/-- Represents the number of pieces Mandy gets -/
def mandy_pieces : ℕ := 15

/-- Represents the number of pieces Paige takes -/
def paige_pieces : ℕ := total_pieces - michael_pieces - mandy_pieces

/-- Theorem stating the ratio of Paige's pieces to pieces left after Michael's share -/
theorem paige_to_remainder_ratio :
  (paige_pieces : ℚ) / (total_pieces - michael_pieces : ℚ) = 1 / 2 := by
  sorry

end paige_to_remainder_ratio_l576_57692


namespace hexagon_circle_visibility_l576_57635

theorem hexagon_circle_visibility (s : ℝ) (r : ℝ) (h1 : s = 3) (h2 : r > 0) : 
  let a := s * Real.sqrt 3 / 2
  (2 * Real.pi * r / 3) / (2 * Real.pi * r) = 1 / 3 → r = 3 / 2 :=
by sorry

end hexagon_circle_visibility_l576_57635


namespace lowest_two_digit_product_12_l576_57658

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem lowest_two_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → 26 ≤ n :=
by sorry

end lowest_two_digit_product_12_l576_57658


namespace sufficient_not_necessary_condition_l576_57602

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (x ≥ 1 → |x + 1| + |x - 1| = 2 * |x|) ∧ 
  ¬(|x + 1| + |x - 1| = 2 * |x| → x ≥ 1) :=
sorry

end sufficient_not_necessary_condition_l576_57602


namespace problem_1_l576_57609

theorem problem_1 : Real.sqrt 9 + (-2)^3 - Real.cos (π / 3) = -11 / 2 := by
  sorry

end problem_1_l576_57609


namespace trigonometric_equation_solution_l576_57682

theorem trigonometric_equation_solution (x : ℝ) :
  2 * Real.cos (13 * x) + 3 * Real.cos (3 * x) + 3 * Real.cos (5 * x) - 8 * Real.cos x * (Real.cos (4 * x))^3 = 0 →
  ∃ k : ℤ, x = π * k / 12 := by
sorry

end trigonometric_equation_solution_l576_57682


namespace quadratic_equation_condition_l576_57688

theorem quadratic_equation_condition (m : ℝ) : 
  (∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, (m - 3) * x^2 + m * x + (-2 * m - 2) = a * x^2 + b * x + c) ↔ 
  m = -1 := by sorry

end quadratic_equation_condition_l576_57688


namespace polynomial_factorization_l576_57600

theorem polynomial_factorization :
  ∀ x : ℝ, x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by
  sorry

end polynomial_factorization_l576_57600


namespace students_per_bus_l576_57685

/-- Given a field trip scenario with buses and students, calculate the number of students per bus. -/
theorem students_per_bus (total_seats : ℕ) (num_buses : ℚ) : 
  total_seats = 28 → num_buses = 2 → (total_seats : ℚ) / num_buses = 14 := by
  sorry

end students_per_bus_l576_57685


namespace x_plus_y_equals_negative_one_l576_57604

theorem x_plus_y_equals_negative_one 
  (x y : ℝ) 
  (h1 : x + |x| + y = 5) 
  (h2 : x + |y| - y = 6) : 
  x + y = -1 := by
sorry

end x_plus_y_equals_negative_one_l576_57604


namespace quadratic_transform_h_value_l576_57674

/-- Given a quadratic equation ax^2 + bx + c that can be expressed as 3(x - 5)^2 + 15,
    prove that when 4ax^2 + 4bx + 4c is expressed as n(x - h)^2 + k, h equals 5. -/
theorem quadratic_transform_h_value
  (a b c : ℝ)
  (h : ∀ x, a * x^2 + b * x + c = 3 * (x - 5)^2 + 15) :
  ∃ (n k : ℝ), ∀ x, 4 * a * x^2 + 4 * b * x + 4 * c = n * (x - 5)^2 + k :=
by sorry

end quadratic_transform_h_value_l576_57674


namespace exponent_multiplication_l576_57633

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l576_57633


namespace max_y_coordinate_polar_graph_l576_57683

theorem max_y_coordinate_polar_graph :
  let r : ℝ → ℝ := λ θ ↦ 2 * Real.sin (2 * θ)
  let y : ℝ → ℝ := λ θ ↦ r θ * Real.sin θ
  (∀ θ, y θ ≤ (8 * Real.sqrt 3) / 9) ∧ 
  (∃ θ, y θ = (8 * Real.sqrt 3) / 9) := by
  sorry

end max_y_coordinate_polar_graph_l576_57683


namespace consecutive_integers_squares_sum_l576_57639

theorem consecutive_integers_squares_sum : ∃ a : ℕ,
  (a > 0) ∧
  ((a - 1) * a * (a + 1) = 8 * (3 * a)) ∧
  ((a - 1)^2 + a^2 + (a + 1)^2 = 77) := by
  sorry

end consecutive_integers_squares_sum_l576_57639


namespace banana_solution_l576_57641

/-- Represents the banana cutting scenario -/
def banana_problem (initial_bananas : ℕ) (cut_bananas : ℕ) (eaten_bananas : ℕ) : Prop :=
  initial_bananas ≥ cut_bananas ∧
  cut_bananas > eaten_bananas ∧
  cut_bananas - eaten_bananas = 2 * (initial_bananas - cut_bananas)

/-- Theorem stating the solution to the banana problem -/
theorem banana_solution :
  ∃ (cut_bananas : ℕ),
    banana_problem 310 cut_bananas 70 ∧
    310 - cut_bananas = 100 := by
  sorry

end banana_solution_l576_57641


namespace probability_one_defective_l576_57608

/-- The probability of selecting exactly one defective product from a batch -/
theorem probability_one_defective (total : ℕ) (defective : ℕ) : 
  total = 40 →
  defective = 12 →
  (Nat.choose (total - defective) 1 * Nat.choose defective 1) / Nat.choose total 2 = 28 / 65 := by
  sorry

end probability_one_defective_l576_57608


namespace garbage_classification_test_l576_57614

theorem garbage_classification_test (p_idea : ℝ) (p_no_idea : ℝ) (p_B : ℝ) :
  p_idea = 2/3 →
  p_no_idea = 1/4 →
  p_B = 0.6 →
  let E_A := (3/4 * p_idea + 1/4 * p_no_idea) * 2
  let E_B := p_B * 2
  E_B > E_A :=
by sorry

end garbage_classification_test_l576_57614


namespace marked_hexagon_properties_l576_57603

/-- A regular hexagon with diagonals marked -/
structure MarkedHexagon where
  /-- The area of the hexagon in square centimeters -/
  area : ℝ
  /-- The hexagon is regular -/
  regular : Bool
  /-- All diagonals are marked -/
  diagonals_marked : Bool

/-- The number of parts the hexagon is divided into by its diagonals -/
def num_parts (h : MarkedHexagon) : ℕ := sorry

/-- The area of the smaller hexagon formed by quadrilateral parts -/
def smaller_hexagon_area (h : MarkedHexagon) : ℝ := sorry

/-- Theorem about the properties of a marked regular hexagon -/
theorem marked_hexagon_properties (h : MarkedHexagon) 
  (h_area : h.area = 144)
  (h_regular : h.regular = true)
  (h_marked : h.diagonals_marked = true) :
  num_parts h = 24 ∧ smaller_hexagon_area h = 48 := by sorry

end marked_hexagon_properties_l576_57603


namespace tan_alpha_minus_pi_eighth_l576_57698

theorem tan_alpha_minus_pi_eighth (α : Real) 
  (h : 2 * Real.sin α = Real.sin (α - π/4)) : 
  Real.tan (α - π/8) = 3 - 3 * Real.sqrt 2 := by
  sorry

end tan_alpha_minus_pi_eighth_l576_57698


namespace triangle_inequality_reciprocal_l576_57620

theorem triangle_inequality_reciprocal (a b c : ℝ) 
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  1 / (a + c) + 1 / (b + c) > 1 / (a + b) ∧
  1 / (a + c) + 1 / (a + b) > 1 / (b + c) ∧
  1 / (b + c) + 1 / (a + b) > 1 / (a + c) := by
  sorry

end triangle_inequality_reciprocal_l576_57620


namespace soccer_team_average_age_l576_57697

def ages : List ℕ := [13, 14, 15, 16, 17, 18]
def players : List ℕ := [2, 6, 8, 3, 2, 1]

theorem soccer_team_average_age :
  (List.sum (List.zipWith (· * ·) ages players)) / (List.sum players) = 15 := by
  sorry

end soccer_team_average_age_l576_57697


namespace sum_has_48_divisors_l576_57640

def sum_of_numbers : ℕ := 9240 + 8820

theorem sum_has_48_divisors : Nat.card (Nat.divisors sum_of_numbers) = 48 := by
  sorry

end sum_has_48_divisors_l576_57640


namespace davids_crunches_l576_57637

/-- Given that David did 17 less crunches than Zachary, and Zachary did 62 crunches,
    prove that David did 45 crunches. -/
theorem davids_crunches (zachary_crunches : ℕ) (david_difference : ℤ) 
  (h1 : zachary_crunches = 62)
  (h2 : david_difference = -17) :
  zachary_crunches + david_difference = 45 :=
by sorry

end davids_crunches_l576_57637


namespace no_covering_compact_rationals_l576_57607

theorem no_covering_compact_rationals :
  ¬ (∃ (A : ℕ → Set ℝ),
    (∀ n, IsCompact (A n)) ∧
    (∀ n, A n ⊆ Set.range (Rat.cast : ℚ → ℝ)) ∧
    (∀ K : Set ℝ, IsCompact K → K ⊆ Set.range (Rat.cast : ℚ → ℝ) →
      ∃ m, K ⊆ A m)) :=
by sorry

end no_covering_compact_rationals_l576_57607


namespace first_interest_rate_is_eight_percent_l576_57665

/-- Proves that the first interest rate is 8% given the problem conditions -/
theorem first_interest_rate_is_eight_percent 
  (total_investment : ℝ) 
  (first_investment : ℝ) 
  (second_investment : ℝ) 
  (second_rate : ℝ) 
  (h1 : total_investment = 5400)
  (h2 : first_investment = 3000)
  (h3 : second_investment = total_investment - first_investment)
  (h4 : second_rate = 0.10)
  (h5 : first_investment * (first_rate : ℝ) = second_investment * second_rate) :
  first_rate = 0.08 := by
  sorry

end first_interest_rate_is_eight_percent_l576_57665


namespace possible_m_values_l576_57660

theorem possible_m_values (A B : Set ℝ) (m : ℝ) : 
  A = {-1, 1} →
  B = {x | m * x = 1} →
  A ∪ B = A →
  m = 0 ∨ m = 1 ∨ m = -1 := by
sorry

end possible_m_values_l576_57660


namespace problem_statement_l576_57630

theorem problem_statement (x y : ℚ) (hx : x = 5/6) (hy : y = 6/5) :
  (1/3) * x^8 * y^5 - x^2 * y = 5/6 := by
  sorry

end problem_statement_l576_57630


namespace sufficient_not_necessary_condition_l576_57656

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (((x > 1 ∧ y > 2) → x + y > 3) ∧
   ∃ x y, x + y > 3 ∧ ¬(x > 1 ∧ y > 2)) :=
by sorry

end sufficient_not_necessary_condition_l576_57656


namespace average_marks_chemistry_mathematics_l576_57645

theorem average_marks_chemistry_mathematics 
  (P C M : ℕ) 
  (h : P + C + M = P + 110) : 
  (C + M) / 2 = 55 := by
sorry

end average_marks_chemistry_mathematics_l576_57645


namespace greatest_lower_bound_system_l576_57612

theorem greatest_lower_bound_system (x y z u : ℕ+) 
  (h1 : x ≥ y) 
  (h2 : x + y = z + u) 
  (h3 : 2 * x * y = z * u) : 
  ∃ m : ℝ, m = 3 + 2 * Real.sqrt 2 ∧ 
  (∀ a b c d : ℕ+, a ≥ b → a + b = c + d → 2 * a * b = c * d → (a : ℝ) / b ≥ m) ∧
  (∀ ε > 0, ∃ a b c d : ℕ+, a ≥ b ∧ a + b = c + d ∧ 2 * a * b = c * d ∧ (a : ℝ) / b < m + ε) :=
sorry

end greatest_lower_bound_system_l576_57612


namespace no_consecutive_solution_l576_57634

theorem no_consecutive_solution : ¬ ∃ (a b c d e f : ℕ), 
  (b = a + 1) ∧ (c = a + 2) ∧ (d = a + 3) ∧ (e = a + 4) ∧ (f = a + 5) ∧
  (a * b^c * d + e^f * a * b = 2015) := by
  sorry

end no_consecutive_solution_l576_57634


namespace cube_expansion_coefficient_sum_l576_57669

theorem cube_expansion_coefficient_sum (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (Real.sqrt 3 * x - 1)^3 = a₀ + a₁*x + a₂*x^2 + a₃*x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -8 := by
  sorry

end cube_expansion_coefficient_sum_l576_57669


namespace vector_magnitude_proof_l576_57690

variable (a b : ℝ × ℝ)

theorem vector_magnitude_proof 
  (h1 : ‖a - 2 • b‖ = 1) 
  (h2 : a • b = 1) : 
  ‖a + 2 • b‖ = 3 := by sorry

end vector_magnitude_proof_l576_57690


namespace sum_of_x_and_y_l576_57675

theorem sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y + 20) :
  x + y = 12 + 2 * Real.sqrt 6 ∨ x + y = 12 - 2 * Real.sqrt 6 :=
by sorry

end sum_of_x_and_y_l576_57675


namespace quadratic_shift_theorem_l576_57699

/-- Represents a quadratic function of the form y = a(x-h)² + k -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a quadratic function vertically -/
def verticalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, h := f.h, k := f.k + shift }

/-- Shifts a quadratic function horizontally -/
def horizontalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, h := f.h - shift, k := f.k }

/-- The theorem stating that shifting y = 5(x-1)² + 1 down by 3 and left by 2 results in y = 5(x+1)² - 2 -/
theorem quadratic_shift_theorem :
  let f : QuadraticFunction := { a := 5, h := 1, k := 1 }
  let g := horizontalShift (verticalShift f (-3)) 2
  g = { a := 5, h := -1, k := -2 } := by sorry

end quadratic_shift_theorem_l576_57699


namespace probability_site_in_statistics_l576_57605

def letters_statistics : List Char := ['S', 'T', 'A', 'T', 'I', 'S', 'T', 'I', 'C', 'S']
def letters_site : List Char := ['S', 'I', 'T', 'E']

def count_in_statistics (c : Char) : Nat :=
  (letters_statistics.filter (· = c)).length

def is_in_site (c : Char) : Bool :=
  letters_site.contains c

def favorable_outcomes : Nat :=
  (letters_statistics.filter is_in_site).length

def total_outcomes : Nat :=
  letters_statistics.length

theorem probability_site_in_statistics :
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 3 := by
  sorry

end probability_site_in_statistics_l576_57605


namespace circular_permutation_divisibility_l576_57624

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def circular_permutation (n : ℕ) : Set ℕ :=
  {m : ℕ | ∃ k : ℕ, k < 5 ∧ m = (n * 10^k) % 100000 + n / (100000 / 10^k)}

theorem circular_permutation_divisibility (n : ℕ) (h1 : is_five_digit n) (h2 : n % 41 = 0) :
  ∀ m ∈ circular_permutation n, m % 41 = 0 := by
  sorry

end circular_permutation_divisibility_l576_57624


namespace carries_work_hours_l576_57601

/-- Proves that Carrie works 35 hours per week given the problem conditions -/
theorem carries_work_hours :
  let hourly_rate : ℕ := 8
  let weeks_worked : ℕ := 4
  let bike_cost : ℕ := 400
  let money_left : ℕ := 720
  let total_earned : ℕ := bike_cost + money_left
  ∃ (hours_per_week : ℕ),
    hours_per_week * hourly_rate * weeks_worked = total_earned ∧
    hours_per_week = 35
  := by sorry

end carries_work_hours_l576_57601


namespace fundamental_inequality_variant_l576_57663

theorem fundamental_inequality_variant (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  1 / (2 * a) + 1 / b ≥ 4 := by
sorry

end fundamental_inequality_variant_l576_57663


namespace geometric_sum_first_8_terms_l576_57652

/-- Sum of the first n terms of a geometric sequence -/
def geometric_sum (a₀ r : ℚ) (n : ℕ) : ℚ :=
  a₀ * (1 - r^n) / (1 - r)

/-- The sum of the first 8 terms of a geometric sequence
    with first term 1/3 and common ratio 1/3 is 6560/19683 -/
theorem geometric_sum_first_8_terms :
  geometric_sum (1/3) (1/3) 8 = 6560/19683 := by
  sorry

end geometric_sum_first_8_terms_l576_57652


namespace total_players_l576_57676

theorem total_players (kabadi : ℕ) (kho_kho_only : ℕ) (both : ℕ) : 
  kabadi = 10 → kho_kho_only = 25 → both = 5 → 
  kabadi + kho_kho_only - both = 30 := by
sorry

end total_players_l576_57676


namespace a_3_value_l576_57667

def sequence_a (n : ℕ+) : ℚ := 1 / (n.val * (n.val + 1))

theorem a_3_value : sequence_a 3 = 1 / 12 := by sorry

end a_3_value_l576_57667


namespace portfolio_growth_l576_57646

/-- Calculates the final portfolio value after two years of investment -/
theorem portfolio_growth (initial_investment : ℝ) (growth_rate_1 : ℝ) (additional_investment : ℝ) (growth_rate_2 : ℝ) 
  (h1 : initial_investment = 80)
  (h2 : growth_rate_1 = 0.15)
  (h3 : additional_investment = 28)
  (h4 : growth_rate_2 = 0.10) :
  let value_after_year_1 := initial_investment * (1 + growth_rate_1)
  let value_before_year_2 := value_after_year_1 + additional_investment
  let final_value := value_before_year_2 * (1 + growth_rate_2)
  final_value = 132 := by
  sorry

end portfolio_growth_l576_57646


namespace shaded_region_area_l576_57616

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- A line defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- The shaded region formed by two intersecting lines and a right triangle -/
structure ShadedRegion where
  line1 : Line
  line2 : Line

def area_of_shaded_region (region : ShadedRegion) : ℚ :=
  sorry

theorem shaded_region_area :
  let line1 := Line.mk (Point.mk 0 5) (Point.mk 10 2)
  let line2 := Line.mk (Point.mk 2 6) (Point.mk 9 0)
  let region := ShadedRegion.mk line1 line2
  area_of_shaded_region region = 151425 / 3136 := by
  sorry

end shaded_region_area_l576_57616


namespace disjoint_quadratic_sets_l576_57625

theorem disjoint_quadratic_sets (A B : ℤ) : 
  ∃ C : ℤ, (∀ x y : ℤ, x^2 + A*x + B ≠ 2*y^2 + 2*y + C) := by
  sorry

end disjoint_quadratic_sets_l576_57625


namespace divisible_by_prime_l576_57627

/-- Sequence of polynomials Q -/
def Q : ℕ → (ℤ → ℤ)
| 0 => λ x => 1
| 1 => λ x => x
| (n + 2) => λ x => x * Q (n + 1) x + (n + 1) * Q n x

/-- Theorem statement -/
theorem divisible_by_prime (p : ℕ) (hp : p.Prime) (hp2 : p > 2) :
  ∀ x : ℤ, (Q p x - x ^ p) % p = 0 := by sorry

end divisible_by_prime_l576_57627


namespace jack_classic_authors_l576_57622

/-- The number of books each classic author has -/
def books_per_author : ℕ := 33

/-- The total number of classic books in Jack's collection -/
def total_classic_books : ℕ := 198

/-- The number of classic authors in Jack's collection -/
def number_of_authors : ℕ := total_classic_books / books_per_author

theorem jack_classic_authors :
  number_of_authors = 6 :=
sorry

end jack_classic_authors_l576_57622


namespace altitude_intersection_property_l576_57617

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a line is an altitude of a triangle -/
def isAltitude (t : Triangle) (p1 p2 : Point) : Prop := sorry

/-- Checks if two lines intersect at a point -/
def intersectAt (p1 p2 p3 p4 p5 : Point) : Prop := sorry

theorem altitude_intersection_property (ABC : Triangle) (D E H : Point) :
  isAcute ABC →
  isAltitude ABC A D →
  isAltitude ABC B E →
  intersectAt A D B E H →
  distance H D = 3 →
  distance H E = 4 →
  ∃ (BD DC AE EC : ℝ),
    BD * DC - AE * EC = 3 * distance A D - 7 := by
  sorry

end altitude_intersection_property_l576_57617


namespace perpendicular_lines_a_values_l576_57686

theorem perpendicular_lines_a_values (a : ℝ) :
  (∀ x y : ℝ, 3 * a * x - y - 1 = 0 ∧ (a - 1) * x + y + 1 = 0 →
    (3 * a * ((a - 1) * x + y + 1) + (-1) * (3 * a * x - y - 1) = 0)) →
  a = -1 ∨ a = 1 :=
by sorry

end perpendicular_lines_a_values_l576_57686


namespace essay_time_calculation_l576_57647

/-- The time Rachel spent on her essay -/
def essay_time (
  page_writing_time : ℕ)  -- Time to write one page in seconds
  (research_time : ℕ)     -- Time spent researching in seconds
  (outline_time : ℕ)      -- Time spent on outline in minutes
  (brainstorm_time : ℕ)   -- Time spent brainstorming in seconds
  (total_pages : ℕ)       -- Total number of pages written
  (break_time : ℕ)        -- Break time after each page in seconds
  (editing_time : ℕ)      -- Time spent editing in seconds
  (proofreading_time : ℕ) -- Time spent proofreading in seconds
  : ℚ :=
  let total_seconds : ℕ := 
    research_time + 
    (outline_time * 60) + 
    brainstorm_time + 
    (total_pages * page_writing_time) + 
    (total_pages * break_time) + 
    editing_time + 
    proofreading_time
  (total_seconds : ℚ) / 3600

theorem essay_time_calculation : 
  essay_time 1800 2700 15 1200 6 600 4500 1800 = 25500 / 3600 := by
  sorry

#eval essay_time 1800 2700 15 1200 6 600 4500 1800

end essay_time_calculation_l576_57647


namespace julieta_total_spent_l576_57661

-- Define the original prices and price changes
def original_backpack_price : ℕ := 50
def original_binder_price : ℕ := 20
def backpack_price_increase : ℕ := 5
def binder_price_reduction : ℕ := 2
def number_of_binders : ℕ := 3

-- Define the theorem
theorem julieta_total_spent :
  let new_backpack_price := original_backpack_price + backpack_price_increase
  let new_binder_price := original_binder_price - binder_price_reduction
  let total_spent := new_backpack_price + number_of_binders * new_binder_price
  total_spent = 109 := by sorry

end julieta_total_spent_l576_57661


namespace determinant_calculation_l576_57691

variable (a₁ b₁ b₂ c₁ c₂ c₃ d₁ d₂ d₃ d₄ : ℝ)

def matrix : Matrix (Fin 4) (Fin 4) ℝ := λ i j =>
  match i, j with
  | 0, 0 => a₁
  | 0, 1 => b₁
  | 0, 2 => c₁
  | 0, 3 => d₁
  | 1, 0 => a₁
  | 1, 1 => b₂
  | 1, 2 => c₂
  | 1, 3 => d₂
  | 2, 0 => a₁
  | 2, 1 => b₂
  | 2, 2 => c₃
  | 2, 3 => d₃
  | 3, 0 => a₁
  | 3, 1 => b₂
  | 3, 2 => c₃
  | 3, 3 => d₄
  | _, _ => 0

theorem determinant_calculation :
  Matrix.det (matrix a₁ b₁ b₂ c₁ c₂ c₃ d₁ d₂ d₃ d₄) = a₁ * (b₂ - b₁) * (c₃ - c₂) * (d₄ - d₃) := by
  sorry

end determinant_calculation_l576_57691


namespace smallest_number_with_properties_l576_57623

def is_sum_of_five_fourth_powers (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
    n = a^4 + b^4 + c^4 + d^4 + e^4

def is_sum_of_six_consecutive_integers (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = k + (k+1) + (k+2) + (k+3) + (k+4) + (k+5)

theorem smallest_number_with_properties : 
  ∀ n : ℕ, n < 2019 → 
    ¬(is_sum_of_five_fourth_powers n ∧ is_sum_of_six_consecutive_integers n) :=
by
  sorry

#check smallest_number_with_properties

end smallest_number_with_properties_l576_57623


namespace product_equality_l576_57664

theorem product_equality (p r j : ℝ) : 
  (6 * p^2 - 4 * p + r) * (2 * p^2 + j * p - 7) = 12 * p^4 - 34 * p^3 - 19 * p^2 + 28 * p - 21 →
  r + j = 3 := by
sorry

end product_equality_l576_57664


namespace unique_positive_solution_l576_57695

theorem unique_positive_solution (x y z : ℝ) :
  x > 0 ∧ y > 0 ∧ z > 0 →
  x^3 + 2*y^2 + 1/(4*z) = 1 →
  y^3 + 2*z^2 + 1/(4*x) = 1 →
  z^3 + 2*x^2 + 1/(4*y) = 1 →
  x = (-1 + Real.sqrt 3) / 2 ∧
  y = (-1 + Real.sqrt 3) / 2 ∧
  z = (-1 + Real.sqrt 3) / 2 := by
sorry

end unique_positive_solution_l576_57695


namespace oliver_money_l576_57654

/-- 
Given that Oliver:
- Had x dollars in January
- Spent 4 dollars by March
- Received 32 dollars from his mom
- Then had 61 dollars

Prove that x must equal 33.
-/
theorem oliver_money (x : ℤ) 
  (spent : ℤ) 
  (received : ℤ) 
  (final_amount : ℤ) 
  (h1 : spent = 4)
  (h2 : received = 32)
  (h3 : final_amount = 61)
  (h4 : x - spent + received = final_amount) : 
  x = 33 := by
  sorry

end oliver_money_l576_57654


namespace green_to_yellow_area_ratio_l576_57670

-- Define the diameters of the circles
def small_diameter : ℝ := 2
def large_diameter : ℝ := 6

-- Define the theorem
theorem green_to_yellow_area_ratio :
  let small_radius := small_diameter / 2
  let large_radius := large_diameter / 2
  let yellow_area := π * small_radius^2
  let total_area := π * large_radius^2
  let green_area := total_area - yellow_area
  green_area / yellow_area = 8 := by
  sorry

end green_to_yellow_area_ratio_l576_57670
