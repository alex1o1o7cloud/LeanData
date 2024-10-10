import Mathlib

namespace incompatible_inequalities_l698_69882

theorem incompatible_inequalities (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ¬(a + b < c + d ∧ 
    (a + b) * (c + d) < a * b + c * d ∧ 
    (a + b) * c * d < (c + d) * a * b) :=
by sorry

end incompatible_inequalities_l698_69882


namespace bike_ride_distance_l698_69813

/-- Calculates the total distance traveled given a constant speed and time -/
def total_distance (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem bike_ride_distance :
  let rate := 1.5 / 10  -- miles per minute
  let time := 40        -- minutes
  total_distance rate time = 6 := by
  sorry

end bike_ride_distance_l698_69813


namespace triangle_area_proof_l698_69892

/-- The length of a li in meters -/
def li_to_meters : ℝ := 500

/-- The sides of the triangle in li -/
def side1 : ℝ := 5
def side2 : ℝ := 12
def side3 : ℝ := 13

/-- The area of the triangle in square kilometers -/
def triangle_area : ℝ := 7.5

theorem triangle_area_proof :
  let side1_m := side1 * li_to_meters
  let side2_m := side2 * li_to_meters
  let side3_m := side3 * li_to_meters
  side1_m ^ 2 + side2_m ^ 2 = side3_m ^ 2 →
  (1 / 2) * side1_m * side2_m / 1000000 = triangle_area := by
  sorry

end triangle_area_proof_l698_69892


namespace smallest_integer_satisfying_inequality_l698_69870

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3*x - 15 → x ≥ 8 ∧ 8 < 3*8 - 15 := by
  sorry

end smallest_integer_satisfying_inequality_l698_69870


namespace courtyard_paving_l698_69884

def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 16
def brick_length : ℝ := 0.2
def brick_width : ℝ := 0.1

theorem courtyard_paving :
  (courtyard_length * courtyard_width) / (brick_length * brick_width) = 20000 := by
  sorry

end courtyard_paving_l698_69884


namespace max_eggs_l698_69822

theorem max_eggs (x : ℕ) : 
  x < 200 ∧ 
  x % 3 = 2 ∧ 
  x % 4 = 3 ∧ 
  x % 5 = 4 ∧
  (∀ y : ℕ, y < 200 ∧ y % 3 = 2 ∧ y % 4 = 3 ∧ y % 5 = 4 → y ≤ x) →
  x = 179 :=
by sorry

end max_eggs_l698_69822


namespace kenneth_initial_money_l698_69872

/-- The amount of money Kenneth had initially -/
def initial_money : ℕ := 50

/-- The number of baguettes Kenneth bought -/
def num_baguettes : ℕ := 2

/-- The cost of each baguette in dollars -/
def cost_baguette : ℕ := 2

/-- The number of water bottles Kenneth bought -/
def num_water : ℕ := 2

/-- The cost of each water bottle in dollars -/
def cost_water : ℕ := 1

/-- The amount of money Kenneth has left after the purchase -/
def money_left : ℕ := 44

/-- Theorem stating that Kenneth's initial money equals $50 -/
theorem kenneth_initial_money :
  initial_money = 
    num_baguettes * cost_baguette + 
    num_water * cost_water + 
    money_left := by sorry

end kenneth_initial_money_l698_69872


namespace speed_calculation_l698_69856

-- Define the distance in meters
def distance_meters : ℝ := 375.03

-- Define the time in seconds
def time_seconds : ℝ := 25

-- Define the conversion factor from m/s to km/h
def mps_to_kmph : ℝ := 3.6

-- Theorem to prove
theorem speed_calculation :
  let speed_mps := distance_meters / time_seconds
  let speed_kmph := speed_mps * mps_to_kmph
  ∃ ε > 0, |speed_kmph - 54.009| < ε :=
by
  sorry

end speed_calculation_l698_69856


namespace pen_pencil_difference_l698_69814

theorem pen_pencil_difference (ratio_pens : ℕ) (ratio_pencils : ℕ) (total_pencils : ℕ) : 
  ratio_pens = 5 → ratio_pencils = 6 → total_pencils = 54 → 
  total_pencils - (total_pencils / ratio_pencils * ratio_pens) = 9 := by
sorry

end pen_pencil_difference_l698_69814


namespace distribute_eq_choose_l698_69809

/-- The number of ways to distribute n indistinguishable objects into k distinct boxes -/
def distribute (n k : ℕ+) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- Theorem stating that the number of ways to distribute n indistinguishable objects
    into k distinct boxes is equal to (n+k-1) choose (k-1) -/
theorem distribute_eq_choose (n k : ℕ+) :
  distribute n k = Nat.choose (n + k - 1) (k - 1) := by
  sorry

end distribute_eq_choose_l698_69809


namespace one_cut_divides_two_squares_equally_l698_69816

-- Define a square
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Function to check if a line passes through a point
def line_passes_through_point (l : Line) (p : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p = (l.point1.1 + t * (l.point2.1 - l.point1.1), 
               l.point1.2 + t * (l.point2.2 - l.point1.2))

-- Function to check if a line divides a square into two equal parts
def line_divides_square_equally (l : Line) (s : Square) : Prop :=
  line_passes_through_point l s.center

-- Theorem statement
theorem one_cut_divides_two_squares_equally 
  (s1 s2 : Square) (l : Line) : 
  line_passes_through_point l s1.center → 
  line_passes_through_point l s2.center → 
  line_divides_square_equally l s1 ∧ line_divides_square_equally l s2 :=
sorry

end one_cut_divides_two_squares_equally_l698_69816


namespace marathon_solution_l698_69828

def marathon_problem (dean_time : ℝ) : Prop :=
  let micah_time := dean_time * (3/2)
  let jake_time := micah_time * (4/3)
  let nia_time := micah_time * 2
  let eliza_time := dean_time * (4/5)
  let average_time := (dean_time + micah_time + jake_time + nia_time + eliza_time) / 5
  dean_time = 9 ∧ average_time = 15.14

theorem marathon_solution :
  ∃ (dean_time : ℝ), marathon_problem dean_time :=
by
  sorry

end marathon_solution_l698_69828


namespace olivers_money_l698_69808

/-- Oliver's money calculation -/
theorem olivers_money (initial_amount spent_amount received_amount : ℕ) :
  initial_amount = 33 →
  spent_amount = 4 →
  received_amount = 32 →
  initial_amount - spent_amount + received_amount = 61 :=
by
  sorry

end olivers_money_l698_69808


namespace circle_circumference_increase_l698_69897

theorem circle_circumference_increase (d : Real) : 
  let increase_in_diameter : Real := 2 * Real.pi
  let original_circumference : Real := Real.pi * d
  let new_circumference : Real := Real.pi * (d + increase_in_diameter)
  let Q : Real := new_circumference - original_circumference
  Q = 2 * Real.pi ^ 2 := by
  sorry

end circle_circumference_increase_l698_69897


namespace final_sum_theorem_l698_69877

theorem final_sum_theorem (x y R : ℝ) (h : x + y = R) :
  3 * (x + 5) + 3 * (y + 5) = 3 * R + 30 :=
by sorry

end final_sum_theorem_l698_69877


namespace log_equation_proof_l698_69832

-- Define the common logarithm (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation_proof :
  (log 5) ^ 2 + log 2 * log 50 = 1 := by
  sorry

end log_equation_proof_l698_69832


namespace hulk_jump_distance_l698_69826

def jump_distance (n : ℕ) : ℝ := 3 * (2 ^ (n - 1))

theorem hulk_jump_distance :
  (∀ k < 11, jump_distance k ≤ 3000) ∧ jump_distance 11 > 3000 := by
  sorry

end hulk_jump_distance_l698_69826


namespace ellipse_major_axis_length_l698_69833

/-- An ellipse with parametric equations x = 3cos(θ) and y = 2sin(θ) -/
structure Ellipse where
  x : ℝ → ℝ
  y : ℝ → ℝ
  h_x : ∀ θ, x θ = 3 * Real.cos θ
  h_y : ∀ θ, y θ = 2 * Real.sin θ

/-- The length of the major axis of an ellipse -/
def major_axis_length (e : Ellipse) : ℝ := 6

/-- Theorem: The length of the major axis of the given ellipse is 6 -/
theorem ellipse_major_axis_length (e : Ellipse) : 
  major_axis_length e = 6 := by sorry

end ellipse_major_axis_length_l698_69833


namespace initial_workers_l698_69889

theorem initial_workers (total : ℕ) (increase_percent : ℚ) : 
  total = 1065 → increase_percent = 25 / 100 → 
  ∃ initial : ℕ, initial * (1 + increase_percent) = total ∧ initial = 852 :=
by sorry

end initial_workers_l698_69889


namespace division_addition_problem_l698_69855

theorem division_addition_problem : (-144) / (-36) + 10 = 14 := by
  sorry

end division_addition_problem_l698_69855


namespace interview_bounds_l698_69885

theorem interview_bounds (students : ℕ) (junior_high : ℕ) (teachers : ℕ) (table_tennis : ℕ) (basketball : ℕ)
  (h1 : students = 6)
  (h2 : junior_high = 4)
  (h3 : teachers = 2)
  (h4 : table_tennis = 5)
  (h5 : basketball = 2)
  (h6 : junior_high ≤ students) :
  ∃ (min max : ℕ),
    (min = students + teachers) ∧
    (max = students - junior_high + teachers + table_tennis + basketball + junior_high) ∧
    (min = 8) ∧
    (max = 15) ∧
    (∀ n : ℕ, n ≥ min ∧ n ≤ max) := by
  sorry

end interview_bounds_l698_69885


namespace parallel_vectors_x_value_l698_69812

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (1 + x, 1 - 3*x)
  let b : ℝ × ℝ := (2, -1)
  are_parallel a b → x = 3/5 := by
sorry

end parallel_vectors_x_value_l698_69812


namespace scaled_triangle_not_valid_l698_69837

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given side lengths can form a valid triangle -/
def is_valid_triangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- The original triangle PQR -/
def original_triangle : Triangle :=
  { a := 15, b := 20, c := 25 }

/-- The scaled triangle PQR -/
def scaled_triangle : Triangle :=
  { a := 3 * original_triangle.a,
    b := 2 * original_triangle.b,
    c := original_triangle.c }

/-- Theorem stating that the scaled triangle is not valid -/
theorem scaled_triangle_not_valid :
  ¬(is_valid_triangle scaled_triangle) :=
sorry

end scaled_triangle_not_valid_l698_69837


namespace union_of_M_and_N_l698_69845

-- Define the sets M and N
def M : Set ℝ := {x | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x | x < -5 ∨ x > 5}

-- State the theorem
theorem union_of_M_and_N : 
  M ∪ N = {x : ℝ | x < -5 ∨ x > -3} := by
  sorry

end union_of_M_and_N_l698_69845


namespace x_squared_plus_5xy_plus_y_squared_l698_69896

theorem x_squared_plus_5xy_plus_y_squared (x y : ℝ) 
  (h1 : x * y = 4) 
  (h2 : x - y = 5) : 
  x^2 + 5*x*y + y^2 = 53 := by
sorry

end x_squared_plus_5xy_plus_y_squared_l698_69896


namespace complex_equation_roots_l698_69895

theorem complex_equation_roots : 
  let z₁ : ℂ := (1 + 2 * Real.sqrt 7 - Real.sqrt 7 * I) / 2
  let z₂ : ℂ := (1 - 2 * Real.sqrt 7 + Real.sqrt 7 * I) / 2
  (z₁^2 - z₁ = 3 - 7*I) ∧ (z₂^2 - z₂ = 3 - 7*I) := by
  sorry


end complex_equation_roots_l698_69895


namespace urn_theorem_l698_69871

/-- Represents the state of the two urns -/
structure UrnState where
  urn1 : ℕ
  urn2 : ℕ

/-- Represents the transfer rule between urns -/
def transfer (state : UrnState) : UrnState :=
  if state.urn1 % 2 = 0 then
    UrnState.mk (state.urn1 / 2) (state.urn2 + state.urn1 / 2)
  else if state.urn2 % 2 = 0 then
    UrnState.mk (state.urn1 + state.urn2 / 2) (state.urn2 / 2)
  else
    state

theorem urn_theorem (p k : ℕ) (h1 : Prime p) (h2 : Prime (2 * p + 1)) (h3 : k < 2 * p + 1) :
  ∃ (n : ℕ) (state : UrnState),
    state.urn1 + state.urn2 = 2 * p + 1 ∧
    (transfer^[n] state).urn1 = k ∨ (transfer^[n] state).urn2 = k :=
  sorry

end urn_theorem_l698_69871


namespace no_intersection_l698_69820

/-- The line equation 3x + 4y = 12 -/
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The circle equation x^2 + y^2 = 4 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The number of intersection points between the line and the circle -/
def intersection_count : ℕ := 0

theorem no_intersection :
  ∀ x y : ℝ, ¬(line_eq x y ∧ circle_eq x y) :=
by sorry

end no_intersection_l698_69820


namespace event_probability_l698_69858

theorem event_probability (P_A P_A_and_B P_A_or_B : ℝ) 
  (h1 : P_A = 0.4)
  (h2 : P_A_and_B = 0.25)
  (h3 : P_A_or_B = 0.8) :
  ∃ P_B : ℝ, P_B = 0.65 ∧ P_A_or_B = P_A + P_B - P_A_and_B :=
by
  sorry

end event_probability_l698_69858


namespace study_time_difference_l698_69879

-- Define the study times
def kwame_hours : ℝ := 2.5
def connor_hours : ℝ := 1.5
def lexia_minutes : ℝ := 97

-- Define the conversion factor from hours to minutes
def minutes_per_hour : ℝ := 60

-- Theorem to prove
theorem study_time_difference : 
  (kwame_hours * minutes_per_hour + connor_hours * minutes_per_hour) - lexia_minutes = 143 := by
  sorry

end study_time_difference_l698_69879


namespace john_unanswered_questions_l698_69830

/-- Represents the scoring systems and John's scores -/
structure ScoringSystem where
  new_correct : ℤ
  new_wrong : ℤ
  new_unanswered : ℤ
  old_start : ℤ
  old_correct : ℤ
  old_wrong : ℤ
  total_questions : ℕ
  new_score : ℤ
  old_score : ℤ

/-- Calculates the number of unanswered questions based on the scoring system -/
def unanswered_questions (s : ScoringSystem) : ℕ :=
  sorry

/-- Theorem stating that for the given scoring system, John left 2 questions unanswered -/
theorem john_unanswered_questions :
  let s : ScoringSystem := {
    new_correct := 6,
    new_wrong := -1,
    new_unanswered := 3,
    old_start := 25,
    old_correct := 5,
    old_wrong := -2,
    total_questions := 30,
    new_score := 105,
    old_score := 95
  }
  unanswered_questions s = 2 := by
  sorry

end john_unanswered_questions_l698_69830


namespace infinite_geometric_series_second_term_l698_69875

theorem infinite_geometric_series_second_term
  (r : ℝ) (S : ℝ) (h_r : r = 1/4) (h_S : S = 20) :
  let a := S * (1 - r)
  a * r = 15/4 := by
sorry

end infinite_geometric_series_second_term_l698_69875


namespace range_of_a_l698_69803

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

end range_of_a_l698_69803


namespace concert_ticket_price_l698_69806

theorem concert_ticket_price :
  ∀ (ticket_price : ℚ),
    (2 * ticket_price) +                    -- Cost of two tickets
    (0.15 * 2 * ticket_price) +             -- 15% processing fee
    10 +                                    -- Parking fee
    (2 * 5) =                               -- Entrance fee for two people
    135 →                                   -- Total cost
    ticket_price = 50 := by
  sorry

end concert_ticket_price_l698_69806


namespace largest_value_between_2_and_3_l698_69839

theorem largest_value_between_2_and_3 (x : ℝ) (h : 2 < x ∧ x < 3) :
  x^2 ≥ x ∧ x^2 ≥ 3*x ∧ x^2 ≥ Real.sqrt x ∧ x^2 ≥ 1/x :=
by sorry

end largest_value_between_2_and_3_l698_69839


namespace problem_solution_l698_69862

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 6) : 
  q = 3 + Real.sqrt 3 := by
sorry

end problem_solution_l698_69862


namespace nat_less_than_5_finite_int_solution_set_nonempty_l698_69880

-- Define the set of natural numbers less than 5
def nat_less_than_5 : Set ℕ := {n | n < 5}

-- Define the set of integers satisfying 2x + 1 > 7
def int_solution_set : Set ℤ := {x | 2 * x + 1 > 7}

-- Theorem 1: The set of natural numbers less than 5 is finite
theorem nat_less_than_5_finite : Finite nat_less_than_5 := by sorry

-- Theorem 2: The set of integers satisfying 2x + 1 > 7 is non-empty
theorem int_solution_set_nonempty : Set.Nonempty int_solution_set := by sorry

end nat_less_than_5_finite_int_solution_set_nonempty_l698_69880


namespace novel_reading_difference_novel_reading_difference_proof_l698_69801

theorem novel_reading_difference : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun jordan alexandre camille maxime =>
    jordan = 130 ∧
    alexandre = jordan / 10 ∧
    camille = 2 * alexandre ∧
    maxime = (jordan + alexandre + camille) / 2 - 5 →
    jordan - maxime = 51

-- Proof
theorem novel_reading_difference_proof :
  ∃ (jordan alexandre camille maxime : ℕ),
    novel_reading_difference jordan alexandre camille maxime :=
by
  sorry

end novel_reading_difference_novel_reading_difference_proof_l698_69801


namespace problem_statement_l698_69819

theorem problem_statement (a b c : ℝ) (h : a + 10 = b + 12 ∧ b + 12 = c + 15) :
  a^2 + b^2 + c^2 - a*b - b*c - a*c = 38 := by
sorry

end problem_statement_l698_69819


namespace john_unique_performance_l698_69825

/-- Represents the Australian Senior Mathematics Competition (ASMC) -/
structure ASMC where
  total_questions : ℕ
  score_formula : ℕ → ℕ → ℕ
  score_uniqueness : ℕ → Prop

/-- John's performance in the ASMC -/
structure JohnPerformance where
  asmc : ASMC
  correct : ℕ
  wrong : ℕ

/-- Theorem stating that John's performance is unique given his score -/
theorem john_unique_performance (asmc : ASMC) (h : asmc.total_questions = 25) 
    (h_formula : asmc.score_formula = fun c w => 25 + 5 * c - 2 * w)
    (h_uniqueness : asmc.score_uniqueness = fun s => 
      ∀ c₁ w₁ c₂ w₂, s = asmc.score_formula c₁ w₁ → s = asmc.score_formula c₂ w₂ → 
      c₁ + w₁ ≤ asmc.total_questions → c₂ + w₂ ≤ asmc.total_questions → c₁ = c₂ ∧ w₁ = w₂) :
  ∃! (jp : JohnPerformance), 
    jp.asmc = asmc ∧ 
    jp.correct + jp.wrong ≤ asmc.total_questions ∧
    asmc.score_formula jp.correct jp.wrong = 100 ∧
    jp.correct = 19 ∧ jp.wrong = 10 := by
  sorry


end john_unique_performance_l698_69825


namespace sample_size_calculation_l698_69838

theorem sample_size_calculation (total_parts : ℕ) (prob_sampled : ℚ) (n : ℕ) : 
  total_parts = 200 → prob_sampled = 1/4 → n = (total_parts : ℚ) * prob_sampled → n = 50 := by
sorry

end sample_size_calculation_l698_69838


namespace triangle_not_divisible_into_trapeziums_l698_69853

-- Define a shape as a type
inductive Shape
| Rectangle
| Square
| RegularHexagon
| Trapezium
| Triangle

-- Define a trapezium
def isTrapezium (s : Shape) : Prop :=
  ∃ (sides : ℕ), sides = 4 ∧ ∃ (parallel_sides : ℕ), parallel_sides ≥ 1

-- Define the property of being divisible into two trapeziums by a single straight line
def isDivisibleIntoTwoTrapeziums (s : Shape) : Prop :=
  ∃ (part1 part2 : Shape), isTrapezium part1 ∧ isTrapezium part2

-- State the theorem
theorem triangle_not_divisible_into_trapeziums :
  ¬(isDivisibleIntoTwoTrapeziums Shape.Triangle) :=
sorry

end triangle_not_divisible_into_trapeziums_l698_69853


namespace cos_minus_sin_seventeen_fourths_pi_l698_69886

theorem cos_minus_sin_seventeen_fourths_pi : 
  Real.cos (-17 * Real.pi / 4) - Real.sin (-17 * Real.pi / 4) = Real.sqrt 2 := by
  sorry

end cos_minus_sin_seventeen_fourths_pi_l698_69886


namespace three_zeros_implies_a_range_l698_69898

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

/-- The condition that f has three distinct zeros -/
def has_three_distinct_zeros (a : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0

/-- The main theorem stating that if f has three distinct zeros, then -2 < a < 2 -/
theorem three_zeros_implies_a_range (a : ℝ) :
  has_three_distinct_zeros a → -2 < a ∧ a < 2 :=
by sorry

end three_zeros_implies_a_range_l698_69898


namespace distance_between_foci_l698_69829

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 3)^2 + (y + 4)^2) + Real.sqrt ((x + 5)^2 + (y - 8)^2) = 20

-- Define the foci of the ellipse
def focus1 : ℝ × ℝ := (3, -4)
def focus2 : ℝ × ℝ := (-5, 8)

-- Theorem stating the distance between foci
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 4 * Real.sqrt 13 :=
by sorry

end distance_between_foci_l698_69829


namespace base_conversion_problem_l698_69811

theorem base_conversion_problem : 
  (∃ (S : Finset ℕ), 
    (∀ b ∈ S, b ≥ 2 ∧ b^3 ≤ 250 ∧ 250 < b^4) ∧ 
    (∀ b : ℕ, b ≥ 2 → b^3 ≤ 250 → 250 < b^4 → b ∈ S) ∧
    Finset.card S = 2) := by sorry

end base_conversion_problem_l698_69811


namespace solution_set_when_a_is_3_range_of_a_l698_69804

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

end solution_set_when_a_is_3_range_of_a_l698_69804


namespace smallest_y_for_perfect_cube_l698_69893

/-- Given x = 11 * 36 * 42, prove that the smallest positive integer y 
    such that xy is a perfect cube is 5929 -/
theorem smallest_y_for_perfect_cube (x : ℕ) (hx : x = 11 * 36 * 42) :
  ∃ y : ℕ, y > 0 ∧ 
    (∃ n : ℕ, x * y = n^3) ∧ 
    (∀ z : ℕ, z > 0 → z < y → ¬∃ m : ℕ, x * z = m^3) ∧
    y = 5929 := by
  sorry

end smallest_y_for_perfect_cube_l698_69893


namespace midpoint_quadrilateral_area_l698_69852

/-- The area of the quadrilateral formed by connecting the midpoints of a rectangle -/
theorem midpoint_quadrilateral_area (w l : ℝ) (hw : w = 10) (hl : l = 14) :
  let midpoint_quad_area := (w / 2) * (l / 2)
  midpoint_quad_area = 35 := by sorry

end midpoint_quadrilateral_area_l698_69852


namespace square_8x_minus_5_l698_69861

theorem square_8x_minus_5 (x : ℝ) (h : 8 * x^2 + 7 = 12 * x + 17) : (8 * x - 5)^2 = 465 := by
  sorry

end square_8x_minus_5_l698_69861


namespace two_white_balls_probability_l698_69883

/-- The probability of drawing two white balls without replacement from a box containing 8 white balls and 7 black balls is 4/15. -/
theorem two_white_balls_probability :
  let total_balls : ℕ := 8 + 7
  let white_balls : ℕ := 8
  let black_balls : ℕ := 7
  let prob_first_white : ℚ := white_balls / total_balls
  let prob_second_white : ℚ := (white_balls - 1) / (total_balls - 1)
  prob_first_white * prob_second_white = 4 / 15 := by
sorry

end two_white_balls_probability_l698_69883


namespace complex_modulus_problem_l698_69854

theorem complex_modulus_problem : Complex.abs ((Complex.I + 1) / Complex.I) = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l698_69854


namespace complex_number_properties_l698_69888

/-- Given two complex numbers z₁ and z₂ with unit magnitude, prove specific values for z₁ and z₂
    when their difference is given, and prove the value of their product when their sum is given. -/
theorem complex_number_properties (z₁ z₂ : ℂ) :
  (Complex.abs z₁ = 1 ∧ Complex.abs z₂ = 1) →
  (z₁ - z₂ = Complex.mk (Real.sqrt 6 / 3) (Real.sqrt 3 / 3) →
    z₁ = Complex.mk ((Real.sqrt 6 + 3) / 6) ((Real.sqrt 3 - 3 * Real.sqrt 2) / 6) ∧
    z₂ = Complex.mk ((-Real.sqrt 6 + 3) / 6) ((-Real.sqrt 3 - 3 * Real.sqrt 2) / 6)) ∧
  (z₁ + z₂ = Complex.mk (12/13) (-5/13) →
    z₁ * z₂ = Complex.mk (119/169) (-120/169)) :=
by sorry

end complex_number_properties_l698_69888


namespace machine_subtraction_l698_69836

theorem machine_subtraction (initial : ℕ) (added : ℕ) (subtracted : ℕ) (result : ℕ) :
  initial = 26 →
  added = 15 →
  result = 35 →
  initial + added - subtracted = result →
  subtracted = 6 := by
sorry

end machine_subtraction_l698_69836


namespace sum_divisible_by_twelve_l698_69868

theorem sum_divisible_by_twelve (b : ℤ) : 
  ∃ k : ℤ, 6 * b * (b + 1) = 12 * k := by sorry

end sum_divisible_by_twelve_l698_69868


namespace odd_periodic_function_property_l698_69815

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 5) 
  (h_f1 : f 1 = 1) 
  (h_f2 : f 2 = 2) : 
  f 3 - f 4 = -1 := by
  sorry

end odd_periodic_function_property_l698_69815


namespace remainder_divisibility_l698_69848

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 13 * k + 4) → (∃ m : ℤ, N = 39 * m + 4) :=
sorry

end remainder_divisibility_l698_69848


namespace composite_solid_volume_l698_69827

/-- The volume of a composite solid consisting of a rectangular prism and a cylinder -/
theorem composite_solid_volume :
  ∀ (prism_length prism_width prism_height cylinder_radius cylinder_height overlap_volume : ℝ),
  prism_length = 2 →
  prism_width = 2 →
  prism_height = 1 →
  cylinder_radius = 1 →
  cylinder_height = 3 →
  overlap_volume = π / 2 →
  prism_length * prism_width * prism_height + π * cylinder_radius^2 * cylinder_height - overlap_volume = 4 + 5 * π / 2 :=
by sorry

end composite_solid_volume_l698_69827


namespace max_flow_increase_l698_69864

/-- Represents a water purification system with two sections of pipes -/
structure WaterSystem :=
  (pipes_AB : ℕ)
  (pipes_BC : ℕ)
  (flow_increase : ℝ)

/-- The theorem stating the maximum flow rate increase -/
theorem max_flow_increase (system : WaterSystem) 
  (h1 : system.pipes_AB = 10)
  (h2 : system.pipes_BC = 10)
  (h3 : system.flow_increase = 40) : 
  ∃ (max_increase : ℝ), max_increase = 200 ∧ 
  ∀ (new_system : WaterSystem), 
    new_system.pipes_AB + new_system.pipes_BC = system.pipes_AB + system.pipes_BC →
    new_system.flow_increase ≤ max_increase :=
sorry

end max_flow_increase_l698_69864


namespace somu_age_proof_l698_69860

/-- Somu's present age -/
def somu_age : ℕ := 12

/-- Somu's father's present age -/
def father_age : ℕ := 3 * somu_age

theorem somu_age_proof :
  (somu_age = father_age / 3) ∧
  (somu_age - 6 = (father_age - 6) / 5) →
  somu_age = 12 := by
sorry

end somu_age_proof_l698_69860


namespace sport_gender_relationship_l698_69849

/-- The critical value of K² for P(K² ≥ k) = 0.05 -/
def critical_value : ℝ := 3.841

/-- The observed value of K² -/
def observed_value : ℝ := 4.892

/-- The significance level -/
def significance_level : ℝ := 0.05

/-- The sample size -/
def sample_size : ℕ := 200

/-- Theorem stating that the observed value exceeds the critical value,
    allowing us to conclude a relationship between liking the sport and gender
    with 1 - significance_level confidence -/
theorem sport_gender_relationship :
  observed_value > critical_value →
  ∃ (confidence_level : ℝ), confidence_level = 1 - significance_level ∧
    confidence_level > 0.95 ∧
    (∃ (relationship : Prop), relationship) :=
by
  sorry

end sport_gender_relationship_l698_69849


namespace tangent_line_at_point_one_three_l698_69859

def f (x : ℝ) : ℝ := x^3 - x + 3

theorem tangent_line_at_point_one_three :
  let p : ℝ × ℝ := (1, 3)
  let m : ℝ := (deriv f) p.1
  (λ (x y : ℝ) => 2*x - y + 1 = 0) = (λ (x y : ℝ) => y - p.2 = m * (x - p.1)) := by sorry

end tangent_line_at_point_one_three_l698_69859


namespace product_mod_25_l698_69805

theorem product_mod_25 (n : ℕ) : 
  (105 * 86 * 97 ≡ n [ZMOD 25]) → 
  (0 ≤ n ∧ n < 25) → 
  n = 10 := by
sorry

end product_mod_25_l698_69805


namespace min_value_expression_l698_69846

theorem min_value_expression (a x : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 15) (h3 : a ≤ x) (h4 : x ≤ 15) :
  ∃ (min_x : ℝ), min_x = 15 ∧
    ∀ y, a ≤ y ∧ y ≤ 15 →
      |y - a| + |y - 15| + |y - a - 15| ≥ |min_x - a| + |min_x - 15| + |min_x - a - 15| ∧
      |min_x - a| + |min_x - 15| + |min_x - a - 15| = 15 :=
by sorry

end min_value_expression_l698_69846


namespace no_integer_solutions_l698_69810

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), 14 * x^2 + 15 * y^2 = 7^2000 := by
  sorry

end no_integer_solutions_l698_69810


namespace marcus_earnings_theorem_l698_69834

/-- Calculates the after-tax earnings for Marcus over two weeks -/
def marcusEarnings (hoursWeek1 hoursWeek2 : ℕ) (extraEarnings : ℚ) (taxRate : ℚ) : ℚ :=
  let hourlyWage := extraEarnings / (hoursWeek2 - hoursWeek1)
  let totalHours := hoursWeek1 + hoursWeek2
  let totalEarnings := hourlyWage * totalHours
  totalEarnings * (1 - taxRate)

/-- Theorem stating that Marcus's earnings after tax for the two weeks is $293.40 -/
theorem marcus_earnings_theorem :
  marcusEarnings 20 30 65.20 0.1 = 293.40 := by
  sorry

end marcus_earnings_theorem_l698_69834


namespace power_equation_solution_l698_69887

theorem power_equation_solution : ∃ x : ℕ, 8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 = 2^x ∧ x = 39 := by
  sorry

end power_equation_solution_l698_69887


namespace ellipse_major_axis_l698_69890

/-- The equation of an ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

/-- The length of the major axis of the ellipse -/
def major_axis_length : ℝ := 6

/-- Theorem: The length of the major axis of the ellipse x^2 + 9y^2 = 9 is 6 -/
theorem ellipse_major_axis :
  ∀ x y : ℝ, ellipse_equation x y → major_axis_length = 6 :=
by sorry

end ellipse_major_axis_l698_69890


namespace not_magical_2099_l698_69878

/-- A year is magical if there exists a month and day such that their sum equals the last two digits of the year. -/
def isMagicalYear (year : ℕ) : Prop :=
  ∃ (month day : ℕ), 
    1 ≤ month ∧ month ≤ 12 ∧
    1 ≤ day ∧ day ≤ 31 ∧
    month + day = year % 100

/-- 2099 is not a magical year. -/
theorem not_magical_2099 : ¬ isMagicalYear 2099 := by
  sorry

#check not_magical_2099

end not_magical_2099_l698_69878


namespace min_turns_for_1000_pieces_l698_69899

/-- Represents the state of the game with black and white pieces on a circumference. -/
structure GameState where
  black : ℕ
  white : ℕ

/-- Represents a player's turn in the game. -/
inductive Turn
  | PlayerA
  | PlayerB

/-- Defines the rules for removing pieces based on the current player's turn. -/
def removePieces (state : GameState) (turn : Turn) : GameState :=
  match turn with
  | Turn.PlayerA => { black := state.black, white := state.white + 2 * state.black }
  | Turn.PlayerB => { black := state.black + 2 * state.white, white := state.white }

/-- Checks if the game has ended (only one color remains). -/
def isGameOver (state : GameState) : Bool :=
  state.black = 0 || state.white = 0

/-- Calculates the minimum number of turns required to end the game. -/
def minTurnsToEnd (initialState : GameState) : ℕ :=
  sorry

/-- Theorem stating that for 1000 initial pieces, the minimum number of turns to end the game is 8. -/
theorem min_turns_for_1000_pieces :
  ∃ (black white : ℕ), black + white = 1000 ∧ minTurnsToEnd { black := black, white := white } = 8 :=
  sorry

end min_turns_for_1000_pieces_l698_69899


namespace beatty_theorem_l698_69844

theorem beatty_theorem (α β : ℝ) (hα : Irrational α) (hβ : Irrational β) 
  (hpos_α : α > 0) (hpos_β : β > 0) (h_sum : 1/α + 1/β = 1) :
  (∀ k : ℕ+, ∃! n : ℕ+, k = ⌊n * α⌋ ∨ k = ⌊n * β⌋) ∧ 
  (∀ k : ℕ+, ¬(∃ m n : ℕ+, k = ⌊m * α⌋ ∧ k = ⌊n * β⌋)) := by
  sorry

end beatty_theorem_l698_69844


namespace kids_played_monday_tuesday_l698_69800

/-- The number of kids Julia played with on Monday, Tuesday, and Wednesday -/
structure KidsPlayedWith where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Theorem: The sum of kids Julia played with on Monday and Tuesday is 33 -/
theorem kids_played_monday_tuesday (k : KidsPlayedWith) 
  (h1 : k.monday = 15)
  (h2 : k.tuesday = 18)
  (h3 : k.wednesday = 97) : 
  k.monday + k.tuesday = 33 := by
  sorry


end kids_played_monday_tuesday_l698_69800


namespace batch_size_l698_69802

/-- The number of parts A can complete in one day -/
def a_rate : ℚ := 1 / 10

/-- The number of parts B can complete in one day -/
def b_rate : ℚ := 1 / 15

/-- The number of additional parts A completes compared to B in one day -/
def additional_parts : ℕ := 50

/-- The total number of parts in the batch -/
def total_parts : ℕ := 1500

theorem batch_size :
  (a_rate - b_rate) * total_parts = additional_parts := by sorry

end batch_size_l698_69802


namespace triangle_properties_l698_69841

/-- Properties of a triangle ABC with given circumradius, one side length, and ratio of other sides -/
theorem triangle_properties (R a t : ℝ) (h_R : R > 0) (h_a : a > 0) (h_t : t > 0) :
  ∃ (b c : ℝ) (A B C : ℝ),
    b = 2 * R * Real.sin B ∧
    c = b / t ∧
    A = Real.arcsin (a / (2 * R)) ∧
    B = Real.arctan ((t * Real.sin A) / (1 - t * Real.cos A)) ∧
    C = π - A - B :=
by sorry

end triangle_properties_l698_69841


namespace cleaning_hourly_rate_l698_69835

/-- Calculates the hourly rate for cleaning rooms in a building -/
theorem cleaning_hourly_rate
  (floors : ℕ)
  (rooms_per_floor : ℕ)
  (hours_per_room : ℕ)
  (total_earnings : ℕ)
  (h1 : floors = 4)
  (h2 : rooms_per_floor = 10)
  (h3 : hours_per_room = 6)
  (h4 : total_earnings = 3600) :
  total_earnings / (floors * rooms_per_floor * hours_per_room) = 15 := by
  sorry

#check cleaning_hourly_rate

end cleaning_hourly_rate_l698_69835


namespace sqrt_2_irrational_l698_69865

theorem sqrt_2_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a / b : ℚ)^2 = 2 := by
  sorry

end sqrt_2_irrational_l698_69865


namespace bent_strips_odd_l698_69863

/-- Represents a paper strip covering two unit squares on the cube's surface -/
structure Strip where
  isBent : Bool

/-- Represents a 9x9x9 cube covered with 2x1 paper strips -/
structure Cube where
  strips : List Strip

/-- The number of unit squares on the surface of a 9x9x9 cube -/
def surfaceSquares : Nat := 6 * 9 * 9

/-- Theorem: The number of bent strips covering a 9x9x9 cube is odd -/
theorem bent_strips_odd (cube : Cube) (h1 : cube.strips.length * 2 = surfaceSquares) : 
  Odd (cube.strips.filter Strip.isBent).length := by
  sorry


end bent_strips_odd_l698_69863


namespace meter_to_jumps_l698_69807

-- Define the conversion factors
variable (a p q r s t u v : ℚ)

-- Define the relationships between units
axiom hops_to_skips : a * 1 = p
axiom jumps_to_hops : q * 1 = r
axiom skips_to_leaps : s * 1 = t
axiom leaps_to_meters : u * 1 = v

-- The theorem to prove
theorem meter_to_jumps : 1 = (u * s * a * q) / (p * v * t * r) :=
sorry

end meter_to_jumps_l698_69807


namespace day_150_previous_year_is_friday_l698_69869

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : ℕ
  isLeapYear : Bool

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advanceDays (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

theorem day_150_previous_year_is_friday 
  (N : Year) 
  (h1 : N.isLeapYear = true) 
  (h2 : advanceDays DayOfWeek.Sunday 249 = DayOfWeek.Friday) : 
  advanceDays DayOfWeek.Sunday 149 = DayOfWeek.Friday :=
sorry

end day_150_previous_year_is_friday_l698_69869


namespace min_value_of_f_l698_69823

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + |x - a| + 1

-- State the theorem
theorem min_value_of_f (a : ℝ) : 
  ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), f a x ≥ m :=
sorry

end min_value_of_f_l698_69823


namespace smallest_clock_equivalent_hour_l698_69867

theorem smallest_clock_equivalent_hour : ∃ (n : ℕ), n > 10 ∧ n % 12 = (n^2) % 12 ∧ ∀ (m : ℕ), m > 10 ∧ m < n → m % 12 ≠ (m^2) % 12 :=
  sorry

end smallest_clock_equivalent_hour_l698_69867


namespace arithmetic_trapezoid_area_l698_69873

/-- Represents a trapezoid with bases and altitude in arithmetic progression -/
structure ArithmeticTrapezoid where
  b : ℝ  -- altitude
  d : ℝ  -- common difference

/-- The area of an arithmetic trapezoid is b^2 -/
theorem arithmetic_trapezoid_area (t : ArithmeticTrapezoid) : 
  (1 / 2 : ℝ) * ((t.b + t.d) + (t.b - t.d)) * t.b = t.b^2 := by
  sorry

#check arithmetic_trapezoid_area

end arithmetic_trapezoid_area_l698_69873


namespace binary_sum_equals_1100000_l698_69818

/-- Converts a list of bits to its decimal representation -/
def binaryToDecimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Represents a binary number as a list of booleans -/
def Binary := List Bool

theorem binary_sum_equals_1100000 :
  let a : Binary := [true, false, false, true, true]  -- 11001₂
  let b : Binary := [true, true, true]                -- 111₂
  let c : Binary := [false, false, true, false, true] -- 10100₂
  let d : Binary := [true, true, true, true]          -- 1111₂
  let e : Binary := [true, true, false, false, true, true] -- 110011₂
  let sum : Binary := [false, false, false, false, false, true, true] -- 1100000₂
  binaryToDecimal a + binaryToDecimal b + binaryToDecimal c +
  binaryToDecimal d + binaryToDecimal e = binaryToDecimal sum := by
  sorry


end binary_sum_equals_1100000_l698_69818


namespace work_completion_ratio_l698_69866

/-- Given that A can finish a work in 18 days and that A and B working together
    can finish 1/6 of the work in a day, prove that the ratio of the time taken
    by B to finish the work alone to the time taken by A is 1/2. -/
theorem work_completion_ratio (a b : ℝ) (ha : a = 18) 
    (hab : 1 / a + 1 / b = 1 / 6) : b / a = 1 / 2 := by
  sorry

end work_completion_ratio_l698_69866


namespace no_real_roots_condition_l698_69857

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : ℝ := x^2 + 2*x - m

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := 2^2 - 4*(1)*(-m)

-- Theorem statement
theorem no_real_roots_condition (m : ℝ) :
  (∀ x, quadratic_equation x m ≠ 0) ↔ m < -1 := by
  sorry

end no_real_roots_condition_l698_69857


namespace angle_triple_supplement_l698_69874

theorem angle_triple_supplement (x : ℝ) : x = 3 * (180 - x) → x = 135 := by
  sorry

end angle_triple_supplement_l698_69874


namespace no_solutions_in_interval_l698_69821

theorem no_solutions_in_interval (x : Real) : 
  x ∈ Set.Icc 0 Real.pi → 
  ¬(Real.sin (Real.pi * Real.cos x) = Real.cos (Real.pi * Real.sin x)) :=
by sorry

end no_solutions_in_interval_l698_69821


namespace larger_number_problem_l698_69881

theorem larger_number_problem (L S : ℕ) 
  (h1 : L - S = 2415)
  (h2 : L = 21 * S + 15) : 
  L = 2535 := by
sorry

end larger_number_problem_l698_69881


namespace product_of_difference_and_sum_of_squares_l698_69842

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 4) 
  (h2 : a^2 + b^2 = 30) : 
  a * b = 32 := by
sorry

end product_of_difference_and_sum_of_squares_l698_69842


namespace asphalt_work_hours_l698_69817

/-- The number of hours per day the first group worked -/
def hours_per_day : ℝ := 8

/-- The number of men in the first group -/
def men_group1 : ℕ := 30

/-- The number of days the first group worked -/
def days_group1 : ℕ := 12

/-- The length of road asphalted by the first group in km -/
def road_length_group1 : ℝ := 1

/-- The number of men in the second group -/
def men_group2 : ℕ := 20

/-- The number of hours per day the second group worked -/
def hours_per_day_group2 : ℕ := 15

/-- The number of days the second group worked -/
def days_group2 : ℝ := 19.2

/-- The length of road asphalted by the second group in km -/
def road_length_group2 : ℝ := 2

theorem asphalt_work_hours :
  hours_per_day * men_group1 * days_group1 * road_length_group2 =
  hours_per_day_group2 * men_group2 * days_group2 * road_length_group1 :=
by sorry

end asphalt_work_hours_l698_69817


namespace smallest_y_abs_eq_l698_69824

theorem smallest_y_abs_eq (y : ℝ) : (|2 * y + 6| = 18) → (∃ (z : ℝ), |2 * z + 6| = 18 ∧ z ≤ y) → y = -12 := by
  sorry

end smallest_y_abs_eq_l698_69824


namespace light_ray_equation_l698_69831

-- Define the circle M
def circle_M (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 4*y + 7 = 0

-- Define the point A
def point_A : ℝ × ℝ := (-3, 3)

-- Define the x-axis
def x_axis (x y : ℝ) : Prop := y = 0

-- Define the reflected ray equation
def reflected_ray (x y : ℝ) : Prop :=
  4*x - 3*y + 9 = 0

-- Theorem statement
theorem light_ray_equation :
  ∃ (x₀ y₀ : ℝ),
    -- The ray passes through point A
    reflected_ray x₀ y₀ ∧ (x₀, y₀) = point_A ∧
    -- The ray intersects the x-axis
    ∃ (x₁ : ℝ), reflected_ray x₁ 0 ∧
    -- The ray is tangent to circle M
    ∃ (x₂ y₂ : ℝ), circle_M x₂ y₂ ∧ reflected_ray x₂ y₂ ∧
      ∀ (x y : ℝ), circle_M x y → (x - x₂)^2 + (y - y₂)^2 ≥ 1 :=
by sorry

end light_ray_equation_l698_69831


namespace accurate_counting_requires_shaking_l698_69840

/-- Represents a yeast cell -/
structure YeastCell where
  id : ℕ

/-- Represents a culture fluid containing yeast cells -/
structure CultureFluid where
  cells : List YeastCell

/-- Represents a test tube containing culture fluid -/
structure TestTube where
  fluid : CultureFluid

/-- Represents a hemocytometer for counting cells -/
structure Hemocytometer where
  volume : ℝ
  count : CultureFluid → ℕ

/-- Yeast is a unicellular fungus -/
axiom yeast_is_unicellular : ∀ (y : YeastCell), true

/-- Yeast is a facultative anaerobe -/
axiom yeast_is_facultative_anaerobe : ∀ (y : YeastCell), true

/-- Yeast distribution in culture fluid is uneven -/
axiom yeast_distribution_uneven : ∀ (cf : CultureFluid), true

/-- A hemocytometer is used for counting yeast cells -/
axiom hemocytometer_used : ∃ (h : Hemocytometer), true

/-- Shaking the test tube before sampling leads to accurate counting -/
theorem accurate_counting_requires_shaking (tt : TestTube) (h : Hemocytometer) :
  (∀ (sample : CultureFluid), h.count sample = h.count tt.fluid) ↔ 
  (∃ (shaken_tt : TestTube), shaken_tt.fluid = tt.fluid ∧ 
    ∀ (sample : CultureFluid), h.count sample = h.count shaken_tt.fluid) :=
sorry

end accurate_counting_requires_shaking_l698_69840


namespace p_geq_q_l698_69850

theorem p_geq_q (a b : ℝ) (h : a > 2) : a + 1 / (a - 2) ≥ -b^2 - 2*b + 3 := by
  sorry

end p_geq_q_l698_69850


namespace temperature_conversion_l698_69876

theorem temperature_conversion (t k : ℚ) (f : ℚ) : 
  t = f * (k - 32) → t = 50 → k = 122 → f = 5/9 := by
  sorry

end temperature_conversion_l698_69876


namespace union_of_A_and_B_l698_69847

def A : Set ℕ := {2, 3}
def B : Set ℕ := {3, 4}

theorem union_of_A_and_B : A ∪ B = {2, 3, 4} := by sorry

end union_of_A_and_B_l698_69847


namespace total_notebooks_is_303_l698_69851

/-- The total number of notebooks in a classroom with specific distribution of notebooks among students. -/
def total_notebooks : ℕ :=
  let total_students : ℕ := 60
  let students_with_5 : ℕ := total_students / 4
  let students_with_3 : ℕ := total_students / 5
  let students_with_7 : ℕ := total_students / 3
  let students_with_4 : ℕ := total_students - (students_with_5 + students_with_3 + students_with_7)
  (students_with_5 * 5) + (students_with_3 * 3) + (students_with_7 * 7) + (students_with_4 * 4)

theorem total_notebooks_is_303 : total_notebooks = 303 := by
  sorry

end total_notebooks_is_303_l698_69851


namespace fourth_sample_seat_number_l698_69894

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  known_samples : Finset ℕ
  interval : ℕ

/-- The theorem to prove -/
theorem fourth_sample_seat_number
  (s : SystematicSampling)
  (h_total : s.total_students = 56)
  (h_size : s.sample_size = 4)
  (h_known : s.known_samples = {3, 17, 45})
  (h_interval : s.interval = s.total_students / s.sample_size) :
  ∃ (n : ℕ), n ∈ s.known_samples ∧ (n + s.interval) % s.total_students = 31 :=
sorry

end fourth_sample_seat_number_l698_69894


namespace quadratic_function_j_value_l698_69891

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := fun x ↦ (a * x^2 : ℝ) + (b * x : ℝ) + (c : ℝ)

theorem quadratic_function_j_value
  (a b c : ℤ)
  (h1 : QuadraticFunction a b c 1 = 0)
  (h2 : QuadraticFunction a b c (-1) = 0)
  (h3 : 70 < QuadraticFunction a b c 7 ∧ QuadraticFunction a b c 7 < 90)
  (h4 : 110 < QuadraticFunction a b c 8 ∧ QuadraticFunction a b c 8 < 140)
  (h5 : ∃ j : ℤ, 1000 * j < QuadraticFunction a b c 50 ∧ QuadraticFunction a b c 50 < 1000 * (j + 1)) :
  ∃ j : ℤ, j = 4 ∧ 1000 * j < QuadraticFunction a b c 50 ∧ QuadraticFunction a b c 50 < 1000 * (j + 1) := by
  sorry

end quadratic_function_j_value_l698_69891


namespace perfect_squares_between_50_and_200_l698_69843

theorem perfect_squares_between_50_and_200 : 
  (Finset.filter (fun n => 50 < n * n ∧ n * n < 200) (Finset.range 15)).card = 7 := by
  sorry

end perfect_squares_between_50_and_200_l698_69843
