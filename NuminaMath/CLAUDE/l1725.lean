import Mathlib

namespace NUMINAMATH_CALUDE_floor_sqrt_24_squared_l1725_172501

theorem floor_sqrt_24_squared : ⌊Real.sqrt 24⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_24_squared_l1725_172501


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1725_172513

theorem tan_alpha_value (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3)
  (h2 : Real.tan β = 2) : 
  Real.tan α = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1725_172513


namespace NUMINAMATH_CALUDE_no_infinite_set_with_perfect_square_property_l1725_172587

theorem no_infinite_set_with_perfect_square_property : 
  ¬ ∃ (S : Set ℕ), Set.Infinite S ∧ 
    (∀ a b c : ℕ, a ∈ S → b ∈ S → c ∈ S → ∃ k : ℕ, a * b * c + 1 = k * k) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_set_with_perfect_square_property_l1725_172587


namespace NUMINAMATH_CALUDE_squares_to_rectangles_ratio_l1725_172545

/-- The number of ways to choose 2 items from 10 --/
def choose_2_from_10 : ℕ := 45

/-- The number of rectangles on a 10x10 chessboard --/
def num_rectangles : ℕ := choose_2_from_10 * choose_2_from_10

/-- The sum of squares from 1^2 to 10^2 --/
def sum_squares : ℕ := (10 * 11 * 21) / 6

/-- The number of squares on a 10x10 chessboard --/
def num_squares : ℕ := sum_squares

/-- The ratio of squares to rectangles on a 10x10 chessboard is 7/37 --/
theorem squares_to_rectangles_ratio :
  (num_squares : ℚ) / (num_rectangles : ℚ) = 7 / 37 := by sorry

end NUMINAMATH_CALUDE_squares_to_rectangles_ratio_l1725_172545


namespace NUMINAMATH_CALUDE_average_weight_problem_l1725_172571

theorem average_weight_problem (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (B + C) / 2 = 43)
  (h3 : B = 31) :
  (A + B) / 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l1725_172571


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1725_172565

theorem inequality_solution_set (x : ℝ) : 
  (((2 * x - 1) / 3) > ((3 * x - 2) / 2 - 1)) ↔ (x < 2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1725_172565


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1725_172523

theorem cost_price_calculation (profit_difference : ℝ) 
  (h1 : profit_difference = 72) 
  (h2 : (0.18 - 0.09) * cost_price = profit_difference) : 
  cost_price = 800 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1725_172523


namespace NUMINAMATH_CALUDE_ice_cream_arrangement_l1725_172520

theorem ice_cream_arrangement (n : ℕ) (h : n = 6) : Nat.factorial n = 720 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_arrangement_l1725_172520


namespace NUMINAMATH_CALUDE_inequality_proof_l1725_172543

-- Define the functions f and g
def f (x : ℝ) : ℝ := |2*x + 1| - |2*x - 4|
def g (x : ℝ) : ℝ := 9 + 2*x - x^2

-- State the theorem
theorem inequality_proof (x : ℝ) : |8*x - 16| ≥ g x - 2 * f x := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1725_172543


namespace NUMINAMATH_CALUDE_halloween_candy_count_l1725_172574

/-- The number of candy pieces remaining after Halloween --/
def remaining_candy (debby_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) : ℕ :=
  debby_candy + sister_candy - eaten_candy

/-- Theorem stating the remaining candy count for the given scenario --/
theorem halloween_candy_count : remaining_candy 32 42 35 = 39 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_count_l1725_172574


namespace NUMINAMATH_CALUDE_election_votes_proof_l1725_172535

theorem election_votes_proof (total_votes : ℕ) 
  (winner_percent : ℚ) (second_percent : ℚ) (third_percent : ℚ)
  (winner_second_diff : ℕ) (winner_third_diff : ℕ) (winner_fourth_diff : ℕ) :
  winner_percent = 2/5 ∧ 
  second_percent = 7/25 ∧ 
  third_percent = 1/5 ∧
  winner_second_diff = 1536 ∧
  winner_third_diff = 3840 ∧
  winner_fourth_diff = 5632 →
  total_votes = 12800 ∧
  (winner_percent * total_votes).num = 5120 ∧
  (second_percent * total_votes).num = 3584 ∧
  (third_percent * total_votes).num = 2560 ∧
  total_votes - (winner_percent * total_votes).num - 
    (second_percent * total_votes).num - 
    (third_percent * total_votes).num = 1536 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_proof_l1725_172535


namespace NUMINAMATH_CALUDE_travel_distance_proof_l1725_172547

theorem travel_distance_proof (total_distance : ℝ) (bus_distance : ℝ) : 
  total_distance = 1800 →
  bus_distance = 720 →
  (1/3 : ℝ) * total_distance + (2/3 : ℝ) * bus_distance + bus_distance = total_distance :=
by
  sorry

end NUMINAMATH_CALUDE_travel_distance_proof_l1725_172547


namespace NUMINAMATH_CALUDE_exp_greater_than_linear_l1725_172556

theorem exp_greater_than_linear (x : ℝ) (h : x > 0) : Real.exp x ≥ Real.exp 1 * x := by
  sorry

end NUMINAMATH_CALUDE_exp_greater_than_linear_l1725_172556


namespace NUMINAMATH_CALUDE_sqrt_114_plus_44_sqrt_6_l1725_172560

theorem sqrt_114_plus_44_sqrt_6 :
  ∃ (x y z : ℤ), (x + y * Real.sqrt z : ℝ) = Real.sqrt (114 + 44 * Real.sqrt 6) ∧
  z > 0 ∧
  (∀ (w : ℤ), w ^ 2 ∣ z → w = 1 ∨ w = -1) ∧
  x = 5 ∧ y = 2 ∧ z = 6 :=
sorry

end NUMINAMATH_CALUDE_sqrt_114_plus_44_sqrt_6_l1725_172560


namespace NUMINAMATH_CALUDE_second_train_speed_l1725_172511

/-- Proves that the speed of the second train is 36 kmph given the conditions of the problem -/
theorem second_train_speed (first_train_speed : ℝ) (time_difference : ℝ) (meeting_distance : ℝ) :
  first_train_speed = 30 →
  time_difference = 5 →
  meeting_distance = 1050 →
  ∃ (second_train_speed : ℝ),
    second_train_speed * (meeting_distance / second_train_speed) =
    meeting_distance - first_train_speed * time_difference +
    first_train_speed * (meeting_distance / second_train_speed) ∧
    second_train_speed = 36 :=
by
  sorry


end NUMINAMATH_CALUDE_second_train_speed_l1725_172511


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1725_172537

theorem trigonometric_equation_solution (a : ℝ) : 
  (∀ x, Real.cos (3 * a) * Real.sin x + (Real.sin (3 * a) - Real.sin (7 * a)) * Real.cos x = 0) ∧
  Real.cos (3 * a) = 0 ∧
  Real.sin (3 * a) - Real.sin (7 * a) = 0 →
  ∃ t : ℤ, a = π * (2 * ↑t + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1725_172537


namespace NUMINAMATH_CALUDE_min_value_theorem_l1725_172554

theorem min_value_theorem (p q r s t u : ℝ) 
  (pos_p : p > 0) (pos_q : q > 0) (pos_r : r > 0) 
  (pos_s : s > 0) (pos_t : t > 0) (pos_u : u > 0)
  (sum_eq_10 : p + q + r + s + t + u = 10) :
  1/p + 9/q + 4/r + 16/s + 25/t + 36/u ≥ 44.1 ∧
  ∃ (p' q' r' s' t' u' : ℝ),
    p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ s' > 0 ∧ t' > 0 ∧ u' > 0 ∧
    p' + q' + r' + s' + t' + u' = 10 ∧
    1/p' + 9/q' + 4/r' + 16/s' + 25/t' + 36/u' = 44.1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1725_172554


namespace NUMINAMATH_CALUDE_range_of_a_for_intersection_l1725_172586

theorem range_of_a_for_intersection (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ 
   Real.cos (Real.pi * x₁) = 2^x₂ * a - 1/2) ↔ 
  a ∈ Set.Icc (-1/2) 0 ∪ Set.Ioc 0 (3/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_intersection_l1725_172586


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_3_to_2023_l1725_172529

theorem rightmost_three_digits_of_3_to_2023 : 3^2023 % 1000 = 787 := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_3_to_2023_l1725_172529


namespace NUMINAMATH_CALUDE_agent_007_encryption_possible_l1725_172558

theorem agent_007_encryption_possible : ∃ (m n : ℕ), (0.07 : ℝ) = 1 / m + 1 / n := by
  sorry

end NUMINAMATH_CALUDE_agent_007_encryption_possible_l1725_172558


namespace NUMINAMATH_CALUDE_calculate_required_hours_johns_work_schedule_l1725_172504

/-- Calculates the required weekly work hours for a target income given previous work data --/
theorem calculate_required_hours (winter_hours_per_week : ℕ) (winter_weeks : ℕ) (winter_earnings : ℕ) 
  (target_weeks : ℕ) (target_earnings : ℕ) : ℕ :=
  let hourly_rate := winter_earnings / (winter_hours_per_week * winter_weeks)
  let total_hours := target_earnings / hourly_rate
  total_hours / target_weeks

/-- John's work schedule problem --/
theorem johns_work_schedule : 
  calculate_required_hours 40 8 3200 24 4800 = 20 := by
  sorry

end NUMINAMATH_CALUDE_calculate_required_hours_johns_work_schedule_l1725_172504


namespace NUMINAMATH_CALUDE_max_area_rectangle_l1725_172514

/-- Represents a rectangle with integer dimensions and perimeter 40 -/
structure Rectangle where
  length : ℕ
  width : ℕ
  perimeter_constraint : length + width = 20

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Theorem: The maximum area of a rectangle with perimeter 40 and integer dimensions is 100 -/
theorem max_area_rectangle :
  ∀ r : Rectangle, area r ≤ 100 :=
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l1725_172514


namespace NUMINAMATH_CALUDE_first_protest_duration_l1725_172579

/-- 
Given a person who attends two protests where the second protest duration is 25% longer 
than the first, and the total time spent protesting is 9 days, prove that the duration 
of the first protest is 4 days.
-/
theorem first_protest_duration (first_duration : ℝ) 
  (h1 : first_duration > 0)
  (h2 : first_duration + (1.25 * first_duration) = 9) : 
  first_duration = 4 := by
sorry

end NUMINAMATH_CALUDE_first_protest_duration_l1725_172579


namespace NUMINAMATH_CALUDE_race_course_length_is_correct_l1725_172588

/-- The length of a race course where two runners finish at the same time -/
def race_course_length (v_B : ℝ) : ℝ :=
  let v_A : ℝ := 4 * v_B
  let head_start : ℝ := 75
  100

theorem race_course_length_is_correct (v_B : ℝ) (h : v_B > 0) :
  let v_A : ℝ := 4 * v_B
  let head_start : ℝ := 75
  let L : ℝ := race_course_length v_B
  L / v_A = (L - head_start) / v_B :=
by
  sorry

#check race_course_length_is_correct

end NUMINAMATH_CALUDE_race_course_length_is_correct_l1725_172588


namespace NUMINAMATH_CALUDE_crayon_selection_proof_l1725_172596

def choose (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

theorem crayon_selection_proof :
  choose 12 4 = 495 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_proof_l1725_172596


namespace NUMINAMATH_CALUDE_equation_solution_l1725_172522

theorem equation_solution :
  ∀ x : ℝ, (Real.sqrt (5 * x^3 - 1) + Real.sqrt (x^3 - 1) = 4) ↔ 
  (x = Real.rpow 10 (1/3) ∨ x = Real.rpow 2 (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1725_172522


namespace NUMINAMATH_CALUDE_d_range_l1725_172551

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- The condition that a₃a₄ + 1 = 0 for an arithmetic sequence -/
def sequence_condition (a₁ d : ℝ) : Prop :=
  (arithmetic_sequence a₁ d 3) * (arithmetic_sequence a₁ d 4) + 1 = 0

/-- The theorem stating the range of possible values for d -/
theorem d_range (a₁ d : ℝ) :
  sequence_condition a₁ d → d ≤ -2 ∨ d ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_d_range_l1725_172551


namespace NUMINAMATH_CALUDE_equal_quadratic_expressions_l1725_172509

theorem equal_quadratic_expressions (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 6) :
  a * (a - 6) = b * (b - 6) ∧ a * (a - 6) = -9 := by
  sorry

end NUMINAMATH_CALUDE_equal_quadratic_expressions_l1725_172509


namespace NUMINAMATH_CALUDE_sugar_consumption_reduction_l1725_172533

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (initial_price_positive : initial_price > 0)
  (new_price_positive : new_price > 0)
  (price_increase : new_price > initial_price) :
  let reduction_percentage := (1 - initial_price / new_price) * 100
  reduction_percentage = 60 :=
by sorry

end NUMINAMATH_CALUDE_sugar_consumption_reduction_l1725_172533


namespace NUMINAMATH_CALUDE_universal_set_intersection_l1725_172564

-- Define the universe
variable (U : Type)

-- Define sets A and B
variable (A B : Set U)

-- Define S as the universal set
variable (S : Set U)

-- Theorem statement
theorem universal_set_intersection (h1 : S = Set.univ) (h2 : A ∩ B = S) : A = S ∧ B = S := by
  sorry

end NUMINAMATH_CALUDE_universal_set_intersection_l1725_172564


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1725_172593

theorem algebraic_expression_equality (x : ℝ) (h : x^2 + 3*x + 5 = 7) : 
  3*x^2 + 9*x - 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1725_172593


namespace NUMINAMATH_CALUDE_ninth_grader_wins_l1725_172575

/-- Represents the grade of a student -/
inductive Grade
| Ninth
| Tenth

/-- Represents a chess tournament with ninth and tenth graders -/
structure ChessTournament where
  ninth_graders : ℕ
  tenth_graders : ℕ
  ninth_points : ℕ
  tenth_points : ℕ

/-- Chess tournament satisfying the given conditions -/
def valid_tournament (t : ChessTournament) : Prop :=
  t.tenth_graders = 9 * t.ninth_graders ∧
  t.tenth_points = 4 * t.ninth_points

/-- Maximum points a single player can score -/
def max_player_points (t : ChessTournament) (g : Grade) : ℕ :=
  match g with
  | Grade.Ninth => t.tenth_graders
  | Grade.Tenth => (t.tenth_graders - 1) / 2

/-- Theorem stating that a ninth grader wins the tournament with 9 points -/
theorem ninth_grader_wins (t : ChessTournament) 
  (h : valid_tournament t) (h_ninth : t.ninth_graders > 0) :
  ∃ (n : ℕ), n = 9 ∧ 
    n = max_player_points t Grade.Ninth ∧ 
    n > max_player_points t Grade.Tenth :=
  sorry

end NUMINAMATH_CALUDE_ninth_grader_wins_l1725_172575


namespace NUMINAMATH_CALUDE_probability_at_least_one_die_shows_one_or_ten_l1725_172581

/-- The number of sides on each die -/
def num_sides : ℕ := 10

/-- The number of outcomes where a die doesn't show 1 or 10 -/
def favorable_outcomes_per_die : ℕ := num_sides - 2

/-- The total number of outcomes when rolling two dice -/
def total_outcomes : ℕ := num_sides * num_sides

/-- The number of outcomes where neither die shows 1 or 10 -/
def unfavorable_outcomes : ℕ := favorable_outcomes_per_die * favorable_outcomes_per_die

/-- The number of favorable outcomes (at least one die shows 1 or 10) -/
def favorable_outcomes : ℕ := total_outcomes - unfavorable_outcomes

/-- The probability of at least one die showing 1 or 10 -/
theorem probability_at_least_one_die_shows_one_or_ten :
  (favorable_outcomes : ℚ) / total_outcomes = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_die_shows_one_or_ten_l1725_172581


namespace NUMINAMATH_CALUDE_final_S_equals_3_pow_10_l1725_172590

/-- Represents the state of the program at each iteration --/
structure ProgramState where
  S : ℕ
  i : ℕ

/-- The initial state of the program --/
def initial_state : ProgramState := { S := 1, i := 1 }

/-- The transition function for each iteration of the loop --/
def iterate (state : ProgramState) : ProgramState :=
  { S := state.S * 3, i := state.i + 1 }

/-- The final state after the loop completes --/
def final_state : ProgramState :=
  (iterate^[10]) initial_state

/-- The theorem stating that the final value of S is equal to 3^10 --/
theorem final_S_equals_3_pow_10 : final_state.S = 3^10 := by
  sorry

end NUMINAMATH_CALUDE_final_S_equals_3_pow_10_l1725_172590


namespace NUMINAMATH_CALUDE_price_restoration_l1725_172567

theorem price_restoration (original_price : ℝ) (original_price_positive : original_price > 0) : 
  let reduced_price := original_price * (1 - 0.15)
  let restoration_factor := original_price / reduced_price
  let percentage_increase := (restoration_factor - 1) * 100
  ∃ ε > 0, abs (percentage_increase - 17.65) < ε ∧ ε < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_price_restoration_l1725_172567


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l1725_172500

theorem circle_radius_from_area (A : ℝ) (h : A = 64 * Real.pi) :
  ∃ r : ℝ, r > 0 ∧ A = Real.pi * r^2 ∧ r = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l1725_172500


namespace NUMINAMATH_CALUDE_correct_height_l1725_172519

theorem correct_height (n : ℕ) (initial_avg actual_avg wrong_height : ℝ) :
  n = 35 →
  initial_avg = 180 →
  actual_avg = 178 →
  wrong_height = 156 →
  ∃ (correct_height : ℝ),
    correct_height = n * actual_avg - (n * initial_avg - wrong_height) := by
  sorry

end NUMINAMATH_CALUDE_correct_height_l1725_172519


namespace NUMINAMATH_CALUDE_school_children_count_l1725_172503

/-- The number of children in the school --/
def N : ℕ := sorry

/-- The number of bananas available --/
def B : ℕ := sorry

/-- The number of absent children --/
def absent : ℕ := 330

theorem school_children_count :
  (2 * N = B) ∧                 -- Initial distribution: 2 bananas per child
  (4 * (N - absent) = B) →      -- Actual distribution: 4 bananas per child after absences
  N = 660 := by sorry

end NUMINAMATH_CALUDE_school_children_count_l1725_172503


namespace NUMINAMATH_CALUDE_smallest_value_of_reciprocal_sum_l1725_172506

theorem smallest_value_of_reciprocal_sum (u q a₁ a₂ : ℝ) : 
  (a₁ * a₂ = q) →  -- Vieta's formula for product of roots
  (a₁ + a₂ = u) →  -- Vieta's formula for sum of roots
  (a₁ + a₂ = a₁^2 + a₂^2) →
  (a₁ + a₂ = a₁^3 + a₂^3) →
  (a₁ + a₂ = a₁^4 + a₂^4) →
  (∀ u' q' a₁' a₂' : ℝ, 
    (a₁' * a₂' = q') → 
    (a₁' + a₂' = u') → 
    (a₁' + a₂' = a₁'^2 + a₂'^2) → 
    (a₁' + a₂' = a₁'^3 + a₂'^3) → 
    (a₁' + a₂' = a₁'^4 + a₂'^4) → 
    (a₁' ≠ 0 ∧ a₂' ≠ 0) →
    (1 / a₁^10 + 1 / a₂^10 ≤ 1 / a₁'^10 + 1 / a₂'^10)) →
  1 / a₁^10 + 1 / a₂^10 = 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_reciprocal_sum_l1725_172506


namespace NUMINAMATH_CALUDE_gpa_ratio_is_one_third_l1725_172549

/-- Represents a class with two groups of students with different GPAs -/
structure ClassGPA where
  totalStudents : ℕ
  studentsGPA30 : ℕ
  gpa30 : ℝ := 30
  gpa33 : ℝ := 33
  overallGPA : ℝ := 32

/-- The ratio of students with GPA 30 to the total number of students is 1/3 -/
theorem gpa_ratio_is_one_third (c : ClassGPA) 
  (h1 : c.studentsGPA30 ≤ c.totalStudents)
  (h2 : c.totalStudents > 0)
  (h3 : c.gpa30 * c.studentsGPA30 + c.gpa33 * (c.totalStudents - c.studentsGPA30) = c.overallGPA * c.totalStudents) :
  c.studentsGPA30 / c.totalStudents = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_gpa_ratio_is_one_third_l1725_172549


namespace NUMINAMATH_CALUDE_xy_sum_problem_l1725_172518

theorem xy_sum_problem (x y : ℝ) 
  (h1 : 1/x + 1/y = 5)
  (h2 : x*y + x + y = 7) :
  x^2*y + x*y^2 = 245/36 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_problem_l1725_172518


namespace NUMINAMATH_CALUDE_equation_solution_l1725_172542

theorem equation_solution : 
  let f : ℝ → ℝ := fun y => y^2 - 3*y - 10 + (y + 2)*(y + 6)
  (f (-1/2) = 0 ∧ f (-2) = 0) ∧ 
  ∀ y : ℝ, f y = 0 → (y = -1/2 ∨ y = -2) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1725_172542


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1725_172580

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 3}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem intersection_complement_equality : B ∩ (U \ A) = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1725_172580


namespace NUMINAMATH_CALUDE_smallest_top_block_exists_l1725_172546

/-- Represents a block in the pyramid --/
structure Block where
  layer : Nat
  value : Nat

/-- Represents the pyramid structure --/
structure Pyramid where
  blocks : List Block
  layer1 : List Nat
  layer2 : List Nat
  layer3 : List Nat
  layer4 : Nat

/-- Check if a pyramid configuration is valid --/
def isValidPyramid (p : Pyramid) : Prop :=
  p.blocks.length = 54 ∧
  p.layer1.length = 30 ∧
  p.layer2.length = 15 ∧
  p.layer3.length = 8 ∧
  ∀ n ∈ p.layer1, 1 ≤ n ∧ n ≤ 30

/-- Calculate the value of a block in an upper layer --/
def calculateBlockValue (below : List Nat) : Nat :=
  below.sum

/-- The main theorem --/
theorem smallest_top_block_exists (p : Pyramid) :
  isValidPyramid p →
  ∃ (minTop : Nat), 
    p.layer4 = minTop ∧
    ∀ (p' : Pyramid), isValidPyramid p' → p'.layer4 ≥ minTop := by
  sorry


end NUMINAMATH_CALUDE_smallest_top_block_exists_l1725_172546


namespace NUMINAMATH_CALUDE_book_sale_profit_l1725_172539

theorem book_sale_profit (cost_price : ℝ) (discount_rate : ℝ) (no_discount_profit_rate : ℝ) :
  discount_rate = 0.05 →
  no_discount_profit_rate = 1.2 →
  let selling_price := cost_price * (1 + no_discount_profit_rate)
  let discounted_price := selling_price * (1 - discount_rate)
  let profit_with_discount := discounted_price - cost_price
  let profit_rate_with_discount := profit_with_discount / cost_price
  profit_rate_with_discount = 1.09 := by
sorry

end NUMINAMATH_CALUDE_book_sale_profit_l1725_172539


namespace NUMINAMATH_CALUDE_prime_divisibility_problem_l1725_172561

theorem prime_divisibility_problem (p n : ℕ) : 
  p.Prime → 
  n > 0 → 
  n ≤ 2 * p → 
  (n ^ (p - 1) ∣ (p - 1) ^ n + 1) → 
  ((p = 2 ∧ n = 2) ∨ (p = 3 ∧ n = 3) ∨ n = 1) := by
sorry

end NUMINAMATH_CALUDE_prime_divisibility_problem_l1725_172561


namespace NUMINAMATH_CALUDE_burning_time_3x5_rectangle_l1725_172578

/-- Represents a rectangular arrangement of toothpicks -/
structure ToothpickRectangle where
  rows : ℕ
  cols : ℕ
  total_toothpicks : ℕ

/-- Represents the burning properties of toothpicks -/
structure BurningProperties where
  single_toothpick_time : ℕ

/-- Calculates the maximum burning time for a toothpick rectangle -/
def max_burning_time (rect : ToothpickRectangle) (props : BurningProperties) : ℕ :=
  sorry

/-- Theorem: The maximum burning time for a 3x5 toothpick rectangle is 65 seconds -/
theorem burning_time_3x5_rectangle :
  let rect := ToothpickRectangle.mk 3 5 38
  let props := BurningProperties.mk 10
  max_burning_time rect props = 65 := by
  sorry

end NUMINAMATH_CALUDE_burning_time_3x5_rectangle_l1725_172578


namespace NUMINAMATH_CALUDE_student_score_l1725_172568

def max_marks : ℕ := 400
def pass_percentage : ℚ := 30 / 100
def fail_margin : ℕ := 40

theorem student_score : 
  ∀ (student_marks : ℕ),
    (student_marks = max_marks * pass_percentage - fail_margin) →
    student_marks = 80 := by
  sorry

end NUMINAMATH_CALUDE_student_score_l1725_172568


namespace NUMINAMATH_CALUDE_multiples_of_15_sequence_two_thousand_sixteen_position_l1725_172566

theorem multiples_of_15_sequence (n : ℕ) : ℕ → ℕ
  | 0 => 0
  | (k + 1) => 15 * (k + 1)

theorem two_thousand_sixteen_position :
  ∃ (n : ℕ), multiples_of_15_sequence n 134 < 2016 ∧ 
             2016 < multiples_of_15_sequence n 135 ∧ 
             multiples_of_15_sequence n 135 - 2016 = 9 := by
  sorry


end NUMINAMATH_CALUDE_multiples_of_15_sequence_two_thousand_sixteen_position_l1725_172566


namespace NUMINAMATH_CALUDE_tan_thirteen_pi_fourths_l1725_172557

theorem tan_thirteen_pi_fourths : Real.tan (13 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirteen_pi_fourths_l1725_172557


namespace NUMINAMATH_CALUDE_cistern_water_depth_l1725_172507

/-- Proves that for a rectangular cistern with given dimensions and wet surface area, the water depth is as calculated. -/
theorem cistern_water_depth
  (length : ℝ)
  (width : ℝ)
  (total_wet_surface_area : ℝ)
  (h : ℝ)
  (h_length : length = 4)
  (h_width : width = 8)
  (h_total_area : total_wet_surface_area = 62)
  (h_depth : h = (total_wet_surface_area - length * width) / (2 * (length + width))) :
  h = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_cistern_water_depth_l1725_172507


namespace NUMINAMATH_CALUDE_equal_perimeter_not_necessarily_congruent_l1725_172584

-- Define a triangle type
structure Triangle :=
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Define perimeter of a triangle
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Theorem statement
theorem equal_perimeter_not_necessarily_congruent :
  ∃ (t1 t2 : Triangle), perimeter t1 = perimeter t2 ∧ ¬congruent t1 t2 :=
sorry

end NUMINAMATH_CALUDE_equal_perimeter_not_necessarily_congruent_l1725_172584


namespace NUMINAMATH_CALUDE_goldbach_2024_l1725_172541

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem goldbach_2024 : ∃ p q : ℕ, 
  is_prime p ∧ 
  is_prime q ∧ 
  p ≠ q ∧ 
  p + q = 2024 :=
sorry

end NUMINAMATH_CALUDE_goldbach_2024_l1725_172541


namespace NUMINAMATH_CALUDE_soccer_games_per_month_l1725_172538

/-- Given a total of 27 soccer games equally divided over 3 months,
    the number of games played per month is 9. -/
theorem soccer_games_per_month :
  ∀ (total_games : ℕ) (num_months : ℕ) (games_per_month : ℕ),
    total_games = 27 →
    num_months = 3 →
    total_games = num_months * games_per_month →
    games_per_month = 9 := by
  sorry

end NUMINAMATH_CALUDE_soccer_games_per_month_l1725_172538


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l1725_172595

/-- Represents a hyperbola with foci on the y-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  focal_distance : ℝ
  focus_to_asymptote : ℝ
  h_positive : a > 0
  b_positive : b > 0
  h_focal_distance : focal_distance = 2 * Real.sqrt 3
  h_focus_to_asymptote : focus_to_asymptote = Real.sqrt 2
  h_c : c = Real.sqrt 3
  h_relation : c^2 = a^2 + b^2
  h_asymptote : b * c / Real.sqrt (a^2 + b^2) = focus_to_asymptote

/-- The standard equation of the hyperbola is y² - x²/2 = 1 -/
theorem hyperbola_standard_equation (h : Hyperbola) :
  h.a = 1 ∧ h.b = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l1725_172595


namespace NUMINAMATH_CALUDE_speed_ratio_eddy_freddy_l1725_172530

/-- Proves that the ratio of Eddy's average speed to Freddy's average speed is 38:15 -/
theorem speed_ratio_eddy_freddy : 
  ∀ (eddy_distance freddy_distance : ℝ) 
    (eddy_time freddy_time : ℝ),
  eddy_distance = 570 →
  freddy_distance = 300 →
  eddy_time = 3 →
  freddy_time = 4 →
  (eddy_distance / eddy_time) / (freddy_distance / freddy_time) = 38 / 15 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_eddy_freddy_l1725_172530


namespace NUMINAMATH_CALUDE_books_sold_total_l1725_172526

/-- The total number of books sold by three salespeople over three days -/
def total_books_sold (matias_monday olivia_monday luke_monday : ℕ) : ℕ :=
  let matias_tuesday := 2 * matias_monday
  let olivia_tuesday := 3 * olivia_monday
  let luke_tuesday := luke_monday / 2
  let matias_wednesday := 3 * matias_tuesday
  let olivia_wednesday := 4 * olivia_tuesday
  let luke_wednesday := luke_tuesday
  matias_monday + matias_tuesday + matias_wednesday +
  olivia_monday + olivia_tuesday + olivia_wednesday +
  luke_monday + luke_tuesday + luke_wednesday

/-- Theorem stating the total number of books sold by Matias, Olivia, and Luke over three days -/
theorem books_sold_total : total_books_sold 7 5 12 = 167 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_total_l1725_172526


namespace NUMINAMATH_CALUDE_triangle_side_a_triangle_angle_B_l1725_172515

-- Part I
theorem triangle_side_a (A B C : ℝ) (a b c : ℝ) : 
  b = Real.sqrt 3 → A = π / 4 → C = 5 * π / 12 → a = Real.sqrt 2 := by sorry

-- Part II
theorem triangle_angle_B (A B C : ℝ) (a b c : ℝ) :
  b^2 = a^2 + c^2 + Real.sqrt 2 * a * c → B = 3 * π / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_side_a_triangle_angle_B_l1725_172515


namespace NUMINAMATH_CALUDE_evenly_geometric_difference_l1725_172540

/-- A 3-digit number is evenly geometric if it comprises 3 distinct even digits
    which form a geometric sequence when read from left to right. -/
def EvenlyGeometric (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧
                 a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                 Even a ∧ Even b ∧ Even c ∧
                 ∃ (r : ℚ), b = a * r ∧ c = a * r^2

theorem evenly_geometric_difference :
  ∃ (max min : ℕ),
    (∀ n, EvenlyGeometric n → n ≤ max) ∧
    (∀ n, EvenlyGeometric n → min ≤ n) ∧
    (EvenlyGeometric max) ∧
    (EvenlyGeometric min) ∧
    max - min = 0 :=
sorry

end NUMINAMATH_CALUDE_evenly_geometric_difference_l1725_172540


namespace NUMINAMATH_CALUDE_max_income_at_11_l1725_172536

def bicycle_rental (x : ℕ) : ℝ :=
  if x ≤ 6 then 50 * x - 115
  else -3 * x^2 + 68 * x - 115

theorem max_income_at_11 :
  ∀ x : ℕ, 3 ≤ x → x ≤ 20 →
    bicycle_rental x ≤ bicycle_rental 11 := by
  sorry

end NUMINAMATH_CALUDE_max_income_at_11_l1725_172536


namespace NUMINAMATH_CALUDE_three_pump_fill_time_l1725_172576

/-- Represents the time taken (in hours) for three pumps to fill a tank when working together. -/
def combined_fill_time (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

/-- Theorem stating that three pumps with given rates will fill a tank in 6/29 hours. -/
theorem three_pump_fill_time :
  combined_fill_time (1/3) 4 (1/2) = 6/29 := by
  sorry

#eval combined_fill_time (1/3) 4 (1/2)

end NUMINAMATH_CALUDE_three_pump_fill_time_l1725_172576


namespace NUMINAMATH_CALUDE_dandelion_seed_production_l1725_172569

-- Define the number of seeds produced by a single dandelion plant
def seeds_per_plant : ℕ := 50

-- Define the germination rate (half of the seeds)
def germination_rate : ℚ := 1 / 2

-- Theorem statement
theorem dandelion_seed_production :
  let initial_seeds := seeds_per_plant
  let germinated_plants := (initial_seeds : ℚ) * germination_rate
  let total_seeds := (germinated_plants * seeds_per_plant : ℚ)
  total_seeds = 1250 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_seed_production_l1725_172569


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l1725_172591

/-- Convert a list of bits (0s and 1s) to a natural number -/
def binaryToNat (bits : List Nat) : Nat :=
  bits.foldl (fun acc b => 2 * acc + b) 0

/-- The theorem to prove -/
theorem binary_arithmetic_equality :
  let a := binaryToNat [1, 1, 0, 1]
  let b := binaryToNat [1, 1, 1, 0]
  let c := binaryToNat [1, 0, 1, 1]
  let d := binaryToNat [1, 0, 0, 1]
  let e := binaryToNat [1, 0, 1]
  a + b - c + d - e = binaryToNat [1, 0, 0, 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l1725_172591


namespace NUMINAMATH_CALUDE_function_machine_output_l1725_172589

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 15 then
    step1 + 10
  else
    step1 - 3

theorem function_machine_output : function_machine 12 = 33 := by
  sorry

end NUMINAMATH_CALUDE_function_machine_output_l1725_172589


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l1725_172510

/-- Given the cost of pencils and pens, calculate the cost of a specific combination -/
theorem pencil_pen_cost (pencil_cost pen_cost : ℝ) 
  (h1 : 4 * pencil_cost + pen_cost = 2.60)
  (h2 : pencil_cost + 3 * pen_cost = 2.15) :
  3 * pencil_cost + 2 * pen_cost = 2.63 := by
  sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l1725_172510


namespace NUMINAMATH_CALUDE_exponential_decrease_l1725_172521

theorem exponential_decrease (x y a : Real) 
  (h1 : x > y) (h2 : y > 1) (h3 : 0 < a) (h4 : a < 1) : 
  a^x < a^y := by
  sorry

end NUMINAMATH_CALUDE_exponential_decrease_l1725_172521


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l1725_172524

theorem geometric_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  (2 * a 3 = a 1 + a 2) →           -- arithmetic sequence condition
  (q = 1 ∨ q = -1/2) :=             -- conclusion
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l1725_172524


namespace NUMINAMATH_CALUDE_data_mode_and_mean_l1725_172577

def data : List ℕ := [5, 6, 8, 6, 8, 8, 8]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem data_mode_and_mean :
  mode data = 8 ∧ mean data = 7 := by
  sorry

end NUMINAMATH_CALUDE_data_mode_and_mean_l1725_172577


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1725_172508

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1725_172508


namespace NUMINAMATH_CALUDE_inequality_proof_l1725_172599

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  (a / (b + c + 1)) + (b / (c + a + 1)) + (c / (a + b + 1)) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1725_172599


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l1725_172544

theorem closest_integer_to_cube_root_250 : 
  ∀ n : ℤ, |n - (250 : ℝ)^(1/3)| ≥ |6 - (250 : ℝ)^(1/3)| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l1725_172544


namespace NUMINAMATH_CALUDE_exterior_angle_regular_pentagon_l1725_172559

theorem exterior_angle_regular_pentagon :
  ∀ (exterior_angle : ℝ),
  (exterior_angle = 180 - (540 / 5)) →
  exterior_angle = 72 := by
sorry

end NUMINAMATH_CALUDE_exterior_angle_regular_pentagon_l1725_172559


namespace NUMINAMATH_CALUDE_small_sphere_acceleration_l1725_172570

/-- The acceleration of a small charged sphere after material removal from a larger charged sphere -/
theorem small_sphere_acceleration
  (k : ℝ) -- Coulomb's constant
  (q Q : ℝ) -- Charges of small and large spheres
  (r R : ℝ) -- Radii of small and large spheres
  (m : ℝ) -- Mass of small sphere
  (L S : ℝ) -- Distances
  (g : ℝ) -- Acceleration due to gravity
  (h_r_small : r < R)
  (h_initial_balance : k * q * Q / (L + R)^2 = m * g)
  : ∃ (a : ℝ), a = (k * q * Q * r^3) / (m * R^3 * (L + 2*R - S)^2) :=
sorry

end NUMINAMATH_CALUDE_small_sphere_acceleration_l1725_172570


namespace NUMINAMATH_CALUDE_triangle_angle_sine_inequality_l1725_172555

theorem triangle_angle_sine_inequality (α β γ : Real) 
  (h_triangle : α + β + γ = π) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) :
  2 * (Real.sin α / α + Real.sin β / β + Real.sin γ / γ) ≤ 
    (1 / β + 1 / γ) * Real.sin α + 
    (1 / γ + 1 / α) * Real.sin β + 
    (1 / α + 1 / β) * Real.sin γ := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sine_inequality_l1725_172555


namespace NUMINAMATH_CALUDE_fish_birth_calculation_l1725_172563

theorem fish_birth_calculation (num_tanks : ℕ) (fish_per_tank : ℕ) (total_young : ℕ) :
  num_tanks = 3 →
  fish_per_tank = 4 →
  total_young = 240 →
  total_young / (num_tanks * fish_per_tank) = 20 :=
by sorry

end NUMINAMATH_CALUDE_fish_birth_calculation_l1725_172563


namespace NUMINAMATH_CALUDE_base9_sum_and_subtract_l1725_172594

/-- Converts a base 9 number represented as a list of digits to a natural number. -/
def base9ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 9 * acc + d) 0

/-- Converts a natural number to its base 9 representation as a list of digits. -/
def natToBase9 (n : Nat) : List Nat :=
  if n < 9 then [n]
  else (n % 9) :: natToBase9 (n / 9)

theorem base9_sum_and_subtract :
  let a := base9ToNat [1, 5, 3]  -- 351₉
  let b := base9ToNat [5, 6, 4]  -- 465₉
  let c := base9ToNat [2, 3, 1]  -- 132₉
  let d := base9ToNat [7, 4, 1]  -- 147₉
  natToBase9 (a + b + c - d) = [7, 4, 8] := by
  sorry

end NUMINAMATH_CALUDE_base9_sum_and_subtract_l1725_172594


namespace NUMINAMATH_CALUDE_B_power_101_l1725_172592

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  !![0, 1, 0;
     0, 0, 1;
     1, 0, 0]

theorem B_power_101 :
  B ^ 101 = !![0, 0, 1;
                1, 0, 0;
                0, 1, 0] := by sorry

end NUMINAMATH_CALUDE_B_power_101_l1725_172592


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1725_172531

/-- Given a polynomial function f(x) = ax^5 + bx^3 + cx + 8 where f(-2) = 10, prove that f(2) = 6 -/
theorem polynomial_evaluation (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^5 + b * x^3 + c * x + 8
  f (-2) = 10 → f 2 = 6 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1725_172531


namespace NUMINAMATH_CALUDE_unique_number_six_times_sum_of_digits_l1725_172505

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for numbers that are 6 times the sum of their digits -/
def is_six_times_sum_of_digits (n : ℕ) : Prop :=
  n = 6 * sum_of_digits n

theorem unique_number_six_times_sum_of_digits :
  ∃! n : ℕ, n < 1000 ∧ is_six_times_sum_of_digits n :=
sorry

end NUMINAMATH_CALUDE_unique_number_six_times_sum_of_digits_l1725_172505


namespace NUMINAMATH_CALUDE_max_m_value_l1725_172572

theorem max_m_value (m : ℝ) : 
  (∀ x ∈ Set.Icc (-π/4 : ℝ) (π/4 : ℝ), m ≤ Real.tan x + 1) → 
  (∃ M : ℝ, (∀ x ∈ Set.Icc (-π/4 : ℝ) (π/4 : ℝ), M ≤ Real.tan x + 1) ∧ M = 2) :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l1725_172572


namespace NUMINAMATH_CALUDE_windows_preference_l1725_172516

theorem windows_preference (total : ℕ) (mac : ℕ) (no_pref : ℕ) 
  (h1 : total = 210)
  (h2 : mac = 60)
  (h3 : no_pref = 90) :
  total - mac - (mac / 3) - no_pref = 40 := by
  sorry

end NUMINAMATH_CALUDE_windows_preference_l1725_172516


namespace NUMINAMATH_CALUDE_triangle_property_l1725_172562

open Real

theorem triangle_property (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a * cos B - b * cos A = c / 2 ∧
  B = π / 4 ∧
  b = sqrt 5 →
  tan A = 3 * tan B ∧
  (1 / 2) * a * b * sin C = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l1725_172562


namespace NUMINAMATH_CALUDE_rational_equation_equality_l1725_172573

theorem rational_equation_equality (x : ℝ) (h : x ≠ -1) : 
  (1 / (x + 1)) + (1 / (x + 1)^2) + ((-x - 1) / (x^2 + x + 1)) = 
  1 / ((x + 1)^2 * (x^2 + x + 1)) := by sorry

end NUMINAMATH_CALUDE_rational_equation_equality_l1725_172573


namespace NUMINAMATH_CALUDE_min_students_above_60_l1725_172502

/-- Represents a score distribution in a math competition. -/
structure ScoreDistribution where
  totalScore : ℕ
  topThreeScores : Fin 3 → ℕ
  lowestScore : ℕ
  maxSameScore : ℕ

/-- The minimum number of students who scored at least 60 points. -/
def minStudentsAbove60 (sd : ScoreDistribution) : ℕ := 61

/-- The given conditions of the math competition. -/
def mathCompetition : ScoreDistribution where
  totalScore := 8250
  topThreeScores := ![88, 85, 80]
  lowestScore := 30
  maxSameScore := 3

/-- Theorem stating that the minimum number of students who scored at least 60 points is 61. -/
theorem min_students_above_60 :
  minStudentsAbove60 mathCompetition = 61 := by
  sorry

#check min_students_above_60

end NUMINAMATH_CALUDE_min_students_above_60_l1725_172502


namespace NUMINAMATH_CALUDE_add_and_round_to_hundredth_l1725_172534

-- Define the two numbers to be added
def a : Float := 123.456
def b : Float := 78.9102

-- Define the sum of the two numbers
def sum : Float := a + b

-- Define a function to round to the nearest hundredth
def roundToHundredth (x : Float) : Float :=
  (x * 100).round / 100

-- Theorem statement
theorem add_and_round_to_hundredth :
  roundToHundredth sum = 202.37 := by
  sorry

end NUMINAMATH_CALUDE_add_and_round_to_hundredth_l1725_172534


namespace NUMINAMATH_CALUDE_set_operations_l1725_172583

def A : Set ℝ := {x | x ≤ 5}
def B : Set ℝ := {x | 3 < x ∧ x ≤ 7}

theorem set_operations :
  (A ∩ B = {x | 3 < x ∧ x ≤ 5}) ∧
  (A ∪ (Set.univ \ B) = {x | x ≤ 5 ∨ x > 7}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1725_172583


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l1725_172585

theorem smallest_integer_solution (x : ℝ) :
  (x - 3 * (x - 2) ≤ 4 ∧ (1 + 2 * x) / 3 < x - 1) →
  (∀ y : ℤ, y < 5 → ¬(y - 3 * (y - 2) ≤ 4 ∧ (1 + 2 * y) / 3 < y - 1)) ∧
  (5 - 3 * (5 - 2) ≤ 4 ∧ (1 + 2 * 5) / 3 < 5 - 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l1725_172585


namespace NUMINAMATH_CALUDE_arccos_cos_nine_l1725_172550

theorem arccos_cos_nine :
  Real.arccos (Real.cos 9) = 9 - 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_arccos_cos_nine_l1725_172550


namespace NUMINAMATH_CALUDE_job_selection_ways_l1725_172582

theorem job_selection_ways (method1_people : ℕ) (method2_people : ℕ) 
  (h1 : method1_people = 3) (h2 : method2_people = 5) : 
  method1_people + method2_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_job_selection_ways_l1725_172582


namespace NUMINAMATH_CALUDE_missing_number_problem_l1725_172552

theorem missing_number_problem (x n : ℕ) (h_pos : x > 0) :
  let numbers := [x, x + 2, x + n, x + 7, x + 17]
  let mean := (x + (x + 2) + (x + n) + (x + 7) + (x + 17)) / 5
  let median := x + n
  (mean = median + 2) → n = 4 := by
sorry

end NUMINAMATH_CALUDE_missing_number_problem_l1725_172552


namespace NUMINAMATH_CALUDE_decimal_representation_of_sqrt2_plus_sqrt3_power_1980_l1725_172517

theorem decimal_representation_of_sqrt2_plus_sqrt3_power_1980 :
  let x := (Real.sqrt 2 + Real.sqrt 3) ^ 1980
  ∃ y : ℝ, 0 ≤ y ∧ y < 1 ∧ x = 7 + y ∧ y > 0.9 := by
  sorry

end NUMINAMATH_CALUDE_decimal_representation_of_sqrt2_plus_sqrt3_power_1980_l1725_172517


namespace NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_3_mod_17_l1725_172597

theorem smallest_five_digit_congruent_to_3_mod_17 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 3 → n ≥ 10018 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_3_mod_17_l1725_172597


namespace NUMINAMATH_CALUDE_problem_statement_l1725_172527

theorem problem_statement (x y : ℝ) (h1 : x + y > 0) (h2 : x * y ≠ 0) :
  (x^3 + y^3 ≥ x^2*y + y^2*x) ∧
  (Set.Icc (-6 : ℝ) 2 = {m : ℝ | x / y^2 + y / x^2 ≥ m / 2 * (1 / x + 1 / y)}) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1725_172527


namespace NUMINAMATH_CALUDE_division_with_remainder_4032_98_l1725_172598

theorem division_with_remainder_4032_98 : ∃ (q r : ℤ), 4032 = 98 * q + r ∧ 0 ≤ r ∧ r < 98 ∧ r = 14 := by
  sorry

end NUMINAMATH_CALUDE_division_with_remainder_4032_98_l1725_172598


namespace NUMINAMATH_CALUDE_cab_cost_for_event_l1725_172512

/-- Calculates the total cost of cab rides for a one-week event -/
def total_cab_cost (event_duration : ℕ) (distance : ℝ) (fare_per_mile : ℝ) (rides_per_day : ℕ) : ℝ :=
  event_duration * distance * fare_per_mile * rides_per_day

/-- Proves that the total cost of cab rides for the given conditions is $7000 -/
theorem cab_cost_for_event : 
  total_cab_cost 7 200 2.5 2 = 7000 := by sorry

end NUMINAMATH_CALUDE_cab_cost_for_event_l1725_172512


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l1725_172528

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_formula
  (a : ℕ → ℝ)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_geometric : geometric_sequence a)
  (h_first : a 1 = 2)
  (h_relation : ∀ n : ℕ, (a (n + 2))^2 + 4*(a n)^2 = 4*(a (n + 1))^2) :
  ∀ n : ℕ, a n = 2^((n + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l1725_172528


namespace NUMINAMATH_CALUDE_sticker_distribution_l1725_172532

/-- The number of ways to distribute n identical objects into k distinct boxes --/
def stars_and_bars (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute stickers onto sheets --/
def distribute_stickers (total_stickers sheets min_per_sheet : ℕ) : ℕ :=
  stars_and_bars (total_stickers - sheets * min_per_sheet) sheets

theorem sticker_distribution :
  distribute_stickers 10 5 2 = 1 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l1725_172532


namespace NUMINAMATH_CALUDE_kates_hair_length_l1725_172548

/-- Given information about hair lengths of Kate, Emily, and Logan, prove Kate's hair length -/
theorem kates_hair_length (logan_length emily_length kate_length : ℝ) : 
  logan_length = 20 →
  emily_length = logan_length + 6 →
  kate_length = emily_length / 2 →
  kate_length = 13 := by
  sorry

end NUMINAMATH_CALUDE_kates_hair_length_l1725_172548


namespace NUMINAMATH_CALUDE_fertilizer_calculation_l1725_172525

theorem fertilizer_calculation (total_area : ℝ) (partial_area : ℝ) (partial_fertilizer : ℝ) 
  (h1 : total_area = 7200)
  (h2 : partial_area = 3600)
  (h3 : partial_fertilizer = 600) :
  (total_area / partial_area) * partial_fertilizer = 1200 := by
sorry

end NUMINAMATH_CALUDE_fertilizer_calculation_l1725_172525


namespace NUMINAMATH_CALUDE_range_of_a_l1725_172553

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x - 2|

-- Define the range M of y = 2f(x)
def M : Set ℝ := Set.range (λ x => 2 * f x)

-- Theorem statement
theorem range_of_a (a : ℝ) (h : Set.Icc a (2*a - 1) ⊆ M) : 1 ≤ a ∧ a ≤ 3/2 := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l1725_172553
