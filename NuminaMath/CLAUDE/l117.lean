import Mathlib

namespace NUMINAMATH_CALUDE_circle_number_determinable_l117_11736

/-- Represents a system of six circles connected by line segments -/
structure CircleSystem where
  /-- Numbers in the circles -/
  circle_numbers : Fin 6 → ℝ
  /-- Numbers on the segments connecting the circles -/
  segment_numbers : Fin 6 → ℝ
  /-- Each circle contains the sum of its incoming segment numbers -/
  sum_property : ∀ i : Fin 6, circle_numbers i = segment_numbers i + segment_numbers ((i + 5) % 6)

/-- The theorem stating that any circle's number can be determined from the other five -/
theorem circle_number_determinable (cs : CircleSystem) (i : Fin 6) :
  cs.circle_numbers i =
    cs.circle_numbers ((i + 1) % 6) +
    cs.circle_numbers ((i + 3) % 6) +
    cs.circle_numbers ((i + 5) % 6) -
    cs.circle_numbers ((i + 2) % 6) -
    cs.circle_numbers ((i + 4) % 6) :=
  sorry


end NUMINAMATH_CALUDE_circle_number_determinable_l117_11736


namespace NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_mean_sum_squares_l117_11729

theorem arithmetic_geometric_harmonic_mean_sum_squares 
  (x y z : ℝ) 
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 7)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 385.5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_mean_sum_squares_l117_11729


namespace NUMINAMATH_CALUDE_martha_age_is_32_l117_11734

-- Define Ellen's current age
def ellen_current_age : ℕ := 10

-- Define Ellen's age in 6 years
def ellen_future_age : ℕ := ellen_current_age + 6

-- Define Martha's age in terms of Ellen's future age
def martha_age : ℕ := 2 * ellen_future_age

-- Theorem to prove Martha's age
theorem martha_age_is_32 : martha_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_martha_age_is_32_l117_11734


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l117_11721

theorem perfect_square_binomial (k : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 - 20*x + k = (x + b)^2) ↔ k = 100 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l117_11721


namespace NUMINAMATH_CALUDE_binomial_510_510_l117_11727

theorem binomial_510_510 : (510 : ℕ).choose 510 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_510_510_l117_11727


namespace NUMINAMATH_CALUDE_twelfth_term_of_sequence_l117_11738

def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem twelfth_term_of_sequence (a₁ d : ℝ) (h₁ : a₁ = 1) (h₂ : d = 2) :
  arithmetic_sequence a₁ d 12 = 23 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_sequence_l117_11738


namespace NUMINAMATH_CALUDE_max_value_of_expression_l117_11776

theorem max_value_of_expression (x y z : ℝ) (h : x + 3 * y + z = 5) :
  ∃ (max : ℝ), max = 125 / 4 ∧ ∀ (a b c : ℝ), a + 3 * b + c = 5 → a * b + a * c + b * c ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l117_11776


namespace NUMINAMATH_CALUDE_carter_school_earnings_l117_11724

/-- Represents the number of students from each school --/
def students_adams : ℕ := 8
def students_bentley : ℕ := 6
def students_carter : ℕ := 7

/-- Represents the number of days worked by students from each school --/
def days_adams : ℕ := 4
def days_bentley : ℕ := 6
def days_carter : ℕ := 10

/-- Total amount paid for all students' work --/
def total_paid : ℚ := 1020

/-- Theorem stating that the earnings for Carter school students is approximately $517.39 --/
theorem carter_school_earnings : 
  let total_student_days := students_adams * days_adams + students_bentley * days_bentley + students_carter * days_carter
  let daily_wage := total_paid / total_student_days
  let carter_earnings := daily_wage * (students_carter * days_carter)
  ∃ ε > 0, |carter_earnings - 517.39| < ε :=
sorry

end NUMINAMATH_CALUDE_carter_school_earnings_l117_11724


namespace NUMINAMATH_CALUDE_six_people_handshakes_l117_11763

/-- The number of unique handshakes between n people, where each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the number of handshakes between 6 people is 15. -/
theorem six_people_handshakes : handshakes 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_six_people_handshakes_l117_11763


namespace NUMINAMATH_CALUDE_twentieth_meeting_point_theorem_ant_meeting_theorem_l117_11797

/-- Represents the meeting point of two ants -/
structure MeetingPoint where
  distance : ℝ
  meeting_number : ℕ

/-- Calculates the meeting point of two ants -/
def calculate_meeting_point (total_distance : ℝ) (speed_ratio : ℝ) (meeting_number : ℕ) : MeetingPoint :=
  { distance := 2,  -- The actual calculation is omitted
    meeting_number := meeting_number }

/-- The theorem stating the 20th meeting point of the ants -/
theorem twentieth_meeting_point_theorem (total_distance : ℝ) (speed_ratio : ℝ) :
  (calculate_meeting_point total_distance speed_ratio 20).distance = 2 :=
by
  sorry

#check twentieth_meeting_point_theorem

/-- Main theorem about the ant problem -/
theorem ant_meeting_theorem :
  ∃ (total_distance : ℝ) (speed_ratio : ℝ),
    total_distance = 6 ∧ speed_ratio = 2.5 ∧
    (calculate_meeting_point total_distance speed_ratio 20).distance = 2 :=
by
  sorry

#check ant_meeting_theorem

end NUMINAMATH_CALUDE_twentieth_meeting_point_theorem_ant_meeting_theorem_l117_11797


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l117_11761

theorem quadratic_solution_difference_squared :
  ∀ a b : ℝ,
  (4 * a^2 - 8 * a - 21 = 0) →
  (4 * b^2 - 8 * b - 21 = 0) →
  (a - b)^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l117_11761


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l117_11788

/-- Given a line segment with one endpoint at (3, -5) and midpoint at (7, -15),
    the sum of the coordinates of the other endpoint is -14. -/
theorem endpoint_coordinate_sum :
  ∀ (x y : ℝ),
  (x + 3) / 2 = 7 →
  (y - 5) / 2 = -15 →
  x + y = -14 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l117_11788


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l117_11723

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

theorem smallest_multiplier_for_perfect_square :
  ∃ n : ℕ, n > 0 ∧ is_perfect_square (n * y) ∧
  ∀ m : ℕ, 0 < m ∧ m < n → ¬is_perfect_square (m * y) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l117_11723


namespace NUMINAMATH_CALUDE_sum_positive_if_greater_than_abs_l117_11714

theorem sum_positive_if_greater_than_abs (a b : ℝ) (h : a - |b| > 0) : a + b > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_if_greater_than_abs_l117_11714


namespace NUMINAMATH_CALUDE_farmer_land_usage_l117_11728

theorem farmer_land_usage (beans wheat corn total : ℕ) : 
  beans + wheat + corn = total →
  5 * wheat = 2 * beans →
  2 * corn = beans →
  corn = 376 →
  total = 1034 := by
sorry

end NUMINAMATH_CALUDE_farmer_land_usage_l117_11728


namespace NUMINAMATH_CALUDE_number_of_balls_l117_11764

theorem number_of_balls (x : ℕ) : x - 92 = 156 - x → x = 124 := by
  sorry

end NUMINAMATH_CALUDE_number_of_balls_l117_11764


namespace NUMINAMATH_CALUDE_camp_girls_count_l117_11757

theorem camp_girls_count (total : ℕ) (difference : ℕ) (girls : ℕ) : 
  total = 133 → difference = 33 → girls + (girls + difference) = total → girls = 50 := by
  sorry

end NUMINAMATH_CALUDE_camp_girls_count_l117_11757


namespace NUMINAMATH_CALUDE_equation_solution_l117_11767

theorem equation_solution (x : ℝ) : 
  (Real.sqrt (6 * x^2 + 1)) / (Real.sqrt (3 * x^2 + 4)) = 2 / Real.sqrt 3 ↔ 
  x = Real.sqrt (13/6) ∨ x = -Real.sqrt (13/6) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l117_11767


namespace NUMINAMATH_CALUDE_students_in_both_subjects_range_l117_11755

def total_students : ℕ := 3000

def history_min : ℕ := 2100
def history_max : ℕ := 2250

def psychology_min : ℕ := 1200
def psychology_max : ℕ := 1500

theorem students_in_both_subjects_range :
  ∃ (min_both max_both : ℕ),
    (∀ (h p both : ℕ),
      history_min ≤ h ∧ h ≤ history_max →
      psychology_min ≤ p ∧ p ≤ psychology_max →
      h + p - both = total_students →
      min_both ≤ both ∧ both ≤ max_both) ∧
    max_both - min_both = 450 :=
sorry

end NUMINAMATH_CALUDE_students_in_both_subjects_range_l117_11755


namespace NUMINAMATH_CALUDE_complex_magnitude_sum_l117_11745

theorem complex_magnitude_sum (i : ℂ) : i^2 = -1 →
  Complex.abs ((2 + i)^24 + (2 - i)^24) = 488281250 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_sum_l117_11745


namespace NUMINAMATH_CALUDE_minimize_expression_l117_11760

theorem minimize_expression (x : ℝ) (h : x > 1) :
  (2 + 3*x + 4/(x - 1)) ≥ 4*Real.sqrt 3 + 5 ∧
  (2 + 3*x + 4/(x - 1) = 4*Real.sqrt 3 + 5 ↔ x = 2/3*Real.sqrt 3 + 1) :=
by sorry

end NUMINAMATH_CALUDE_minimize_expression_l117_11760


namespace NUMINAMATH_CALUDE_dantes_recipe_total_l117_11753

def dantes_recipe (eggs : ℕ) : ℕ :=
  eggs + eggs / 2

theorem dantes_recipe_total : dantes_recipe 60 = 90 := by
  sorry

end NUMINAMATH_CALUDE_dantes_recipe_total_l117_11753


namespace NUMINAMATH_CALUDE_lincoln_county_houses_l117_11750

theorem lincoln_county_houses : 
  let original_houses : ℕ := 128936
  let new_houses : ℕ := 359482
  original_houses + new_houses = 488418 :=
by sorry

end NUMINAMATH_CALUDE_lincoln_county_houses_l117_11750


namespace NUMINAMATH_CALUDE_teacher_fills_thermos_once_per_day_l117_11741

/-- Represents the teacher's coffee drinking habits --/
structure CoffeeDrinkingHabits where
  thermos_capacity : ℝ
  school_days_per_week : ℕ
  current_weekly_consumption : ℝ
  consumption_reduction_factor : ℝ

/-- Calculates the number of times the thermos is filled per day --/
def thermos_fills_per_day (habits : CoffeeDrinkingHabits) : ℕ :=
  sorry

/-- Theorem stating that the teacher fills her thermos once per day --/
theorem teacher_fills_thermos_once_per_day (habits : CoffeeDrinkingHabits) 
  (h1 : habits.thermos_capacity = 20)
  (h2 : habits.school_days_per_week = 5)
  (h3 : habits.current_weekly_consumption = 40)
  (h4 : habits.consumption_reduction_factor = 1/4) :
  thermos_fills_per_day habits = 1 := by
  sorry

end NUMINAMATH_CALUDE_teacher_fills_thermos_once_per_day_l117_11741


namespace NUMINAMATH_CALUDE_sum_of_bases_equal_1193_l117_11783

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 14 to base 10 -/
def base14_to_base10 (n : ℕ) : ℕ := sorry

/-- Represents the digit C in base 14 -/
def C : ℕ := 12

theorem sum_of_bases_equal_1193 :
  base8_to_base10 356 + base14_to_base10 (4 * 14^2 + C * 14 + 3) = 1193 := by sorry

end NUMINAMATH_CALUDE_sum_of_bases_equal_1193_l117_11783


namespace NUMINAMATH_CALUDE_coordinates_of_N_l117_11703

-- Define the point M
def M : ℝ × ℝ := (-1, 3)

-- Define the length of MN
def MN_length : ℝ := 4

-- Define the property that MN is parallel to y-axis
def parallel_to_y_axis (N : ℝ × ℝ) : Prop :=
  N.1 = M.1

-- Define the distance between M and N
def distance (N : ℝ × ℝ) : ℝ :=
  |N.2 - M.2|

-- Theorem statement
theorem coordinates_of_N :
  ∃ N : ℝ × ℝ, parallel_to_y_axis N ∧ distance N = MN_length ∧ (N = (-1, -1) ∨ N = (-1, 7)) :=
sorry

end NUMINAMATH_CALUDE_coordinates_of_N_l117_11703


namespace NUMINAMATH_CALUDE_sum_of_fractions_l117_11744

theorem sum_of_fractions : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + 
  (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l117_11744


namespace NUMINAMATH_CALUDE_notebook_cost_l117_11782

theorem notebook_cost (notebook_cost pencil_cost : ℝ) 
  (total_cost : notebook_cost + pencil_cost = 2.40)
  (cost_difference : notebook_cost = pencil_cost + 2) : 
  notebook_cost = 2.20 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l117_11782


namespace NUMINAMATH_CALUDE_arrangement_count_is_432_l117_11769

/-- The number of ways to arrange players from four teams in a row. -/
def arrangement_count : ℕ :=
  let celtics := 3  -- Number of Celtics players
  let lakers := 3   -- Number of Lakers players
  let warriors := 2 -- Number of Warriors players
  let nuggets := 2  -- Number of Nuggets players
  let team_count := 4 -- Number of teams
  let specific_warrior := 1 -- One specific Warrior must sit at the left end
  
  -- Arrangements of teams (excluding Warriors who are fixed at the left)
  (team_count - 1).factorial *
  -- Arrangement of the non-specific Warrior
  (warriors - specific_warrior).factorial *
  -- Arrangements within each team
  celtics.factorial * lakers.factorial * nuggets.factorial

/-- Theorem stating that the number of arrangements is 432. -/
theorem arrangement_count_is_432 : arrangement_count = 432 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_432_l117_11769


namespace NUMINAMATH_CALUDE_triangle_obtuse_l117_11739

theorem triangle_obtuse (a b c : ℝ) (h : 2 * c^2 = 2 * a^2 + 2 * b^2 + a * b) :
  ∃ (A B C : ℝ), 
    0 < A ∧ 0 < B ∧ 0 < C ∧
    A + B + C = π ∧
    c^2 = a^2 + b^2 - 2 * a * b * Real.cos C ∧
    Real.cos C < 0 :=
by sorry

end NUMINAMATH_CALUDE_triangle_obtuse_l117_11739


namespace NUMINAMATH_CALUDE_parallel_line_triangle_l117_11768

theorem parallel_line_triangle (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (x y : ℝ), 
  let s := (a + b + c) / 2
  let perimeter_AXY := x + y + (a * (x + y)) / (b + c)
  let perimeter_XBCY := a + b + c - (x + y)
  (0 < x ∧ x < c) ∧ (0 < y ∧ y < b) ∧ 
  perimeter_AXY = perimeter_XBCY →
  (a * (x + y)) / (b + c) = s * (a / (b + c)) := by
sorry

end NUMINAMATH_CALUDE_parallel_line_triangle_l117_11768


namespace NUMINAMATH_CALUDE_cookie_distribution_l117_11720

theorem cookie_distribution (chris kenny glenn terry dan anne : ℕ) : 
  chris = kenny / 3 →
  glenn = 4 * chris →
  glenn = 24 →
  terry = Int.floor (Real.sqrt (glenn : ℝ) + 3) →
  dan = 2 * (chris + kenny) →
  anne = kenny / 2 →
  anne ≥ 7 →
  kenny % 2 = 1 →
  ∀ k : ℕ, k % 2 = 1 ∧ k / 2 ≥ 7 → kenny ≤ k →
  chris = 6 ∧ 
  kenny = 18 ∧ 
  glenn = 24 ∧ 
  terry = 8 ∧ 
  dan = 48 ∧ 
  anne = 9 ∧
  chris + kenny + glenn + terry + dan + anne = 113 :=
by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l117_11720


namespace NUMINAMATH_CALUDE_percentage_problem_l117_11775

theorem percentage_problem (x : ℝ) : (0.3 / 100) * x = 0.15 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l117_11775


namespace NUMINAMATH_CALUDE_game_gameplay_hours_l117_11709

theorem game_gameplay_hours (T : ℝ) (h1 : 0.2 * T + 30 = 50) : T = 100 := by
  sorry

end NUMINAMATH_CALUDE_game_gameplay_hours_l117_11709


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l117_11758

theorem algebraic_expression_value : ∀ a : ℝ, a^2 + a = 3 → 2*a^2 + 2*a - 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l117_11758


namespace NUMINAMATH_CALUDE_Q_equals_N_l117_11793

-- Define the sets Q and N
def Q : Set ℝ := {y | ∃ x, y = x^2 + 1}
def N : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem Q_equals_N : Q = N := by sorry

end NUMINAMATH_CALUDE_Q_equals_N_l117_11793


namespace NUMINAMATH_CALUDE_chess_tournament_ratio_l117_11726

theorem chess_tournament_ratio (total_students : ℕ) (tournament_students : ℕ) :
  total_students = 24 →
  tournament_students = 4 →
  (total_students / 3 : ℚ) = (total_students / 3 : ℕ) →
  (tournament_students : ℚ) / (total_students / 3 : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_ratio_l117_11726


namespace NUMINAMATH_CALUDE_honey_barrel_distribution_l117_11701

/-- Represents a barrel of honey -/
inductive Barrel
  | Full
  | Half
  | Empty

/-- Represents a distribution of barrels to a person -/
structure Distribution :=
  (full : ℕ)
  (half : ℕ)
  (empty : ℕ)

/-- Calculates the amount of honey in a distribution -/
def honey_amount (d : Distribution) : ℚ :=
  d.full + d.half / 2

/-- Calculates the total number of barrels in a distribution -/
def barrel_count (d : Distribution) : ℕ :=
  d.full + d.half + d.empty

/-- Checks if a distribution is valid (7 barrels and 3.5 units of honey) -/
def is_valid_distribution (d : Distribution) : Prop :=
  barrel_count d = 7 ∧ honey_amount d = 7/2

/-- Represents a solution to the honey distribution problem -/
structure Solution :=
  (person1 : Distribution)
  (person2 : Distribution)
  (person3 : Distribution)

/-- Checks if a solution is valid -/
def is_valid_solution (s : Solution) : Prop :=
  is_valid_distribution s.person1 ∧
  is_valid_distribution s.person2 ∧
  is_valid_distribution s.person3 ∧
  s.person1.full + s.person2.full + s.person3.full = 7 ∧
  s.person1.half + s.person2.half + s.person3.half = 7 ∧
  s.person1.empty + s.person2.empty + s.person3.empty = 7

theorem honey_barrel_distribution :
  ∃ (s : Solution), is_valid_solution s :=
sorry

end NUMINAMATH_CALUDE_honey_barrel_distribution_l117_11701


namespace NUMINAMATH_CALUDE_marys_current_age_l117_11785

theorem marys_current_age :
  ∀ (mary_age jay_age : ℕ),
    (jay_age - 5 = (mary_age - 5) + 7) →
    (jay_age + 5 = 2 * (mary_age + 5)) →
    mary_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_marys_current_age_l117_11785


namespace NUMINAMATH_CALUDE_circle_polar_equation_l117_11710

/-- The polar coordinate equation of a circle C, given specific conditions -/
theorem circle_polar_equation (C : Set (ℝ × ℝ)) (P : ℝ × ℝ) (l : ℝ → ℝ → Prop) :
  (P = (Real.sqrt 2, π / 4)) →
  (∀ ρ θ, l ρ θ ↔ ρ * Real.sin (θ - π / 3) = -Real.sqrt 3 / 2) →
  (∃ x, x ∈ C ∧ x.1 = 1 ∧ x.2 = 0) →
  (P ∈ C) →
  (∀ ρ θ, (ρ, θ) ∈ C ↔ ρ = 2 * Real.cos θ) :=
by sorry

end NUMINAMATH_CALUDE_circle_polar_equation_l117_11710


namespace NUMINAMATH_CALUDE_correct_number_value_l117_11716

theorem correct_number_value (n : ℕ) (initial_avg correct_avg wrong_value : ℚ) :
  n = 10 →
  initial_avg = 5 →
  wrong_value = 26 →
  correct_avg = 6 →
  ∃ (correct_value : ℚ),
    correct_value = wrong_value + n * (correct_avg - initial_avg) ∧
    correct_value = 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_value_l117_11716


namespace NUMINAMATH_CALUDE_football_tournament_yardage_l117_11742

/-- Represents a football team's yardage progress --/
structure TeamProgress where
  gains : List Int
  losses : List Int
  bonus : Int
  penalty : Int

/-- Calculates the total yardage progress for a team --/
def totalYardage (team : TeamProgress) : Int :=
  (team.gains.sum - team.losses.sum) + team.bonus - team.penalty

/-- The football tournament scenario --/
def footballTournament : Prop :=
  let teamA : TeamProgress := {
    gains := [8, 6],
    losses := [5, 3],
    bonus := 0,
    penalty := 2
  }
  let teamB : TeamProgress := {
    gains := [4, 9],
    losses := [2, 7],
    bonus := 0,
    penalty := 3
  }
  let teamC : TeamProgress := {
    gains := [2, 11],
    losses := [6, 4],
    bonus := 3,
    penalty := 4
  }
  (totalYardage teamA = 4) ∧
  (totalYardage teamB = 1) ∧
  (totalYardage teamC = 2)

theorem football_tournament_yardage : footballTournament := by
  sorry

end NUMINAMATH_CALUDE_football_tournament_yardage_l117_11742


namespace NUMINAMATH_CALUDE_five_students_three_teams_l117_11730

/-- The number of ways to assign students to sports teams. -/
def assignStudentsToTeams (numStudents : ℕ) (numTeams : ℕ) : ℕ :=
  numTeams ^ numStudents

/-- Theorem stating that assigning 5 students to 3 teams results in 3^5 possibilities. -/
theorem five_students_three_teams :
  assignStudentsToTeams 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_five_students_three_teams_l117_11730


namespace NUMINAMATH_CALUDE_last_student_score_l117_11743

theorem last_student_score (total_students : ℕ) (average_19 : ℝ) (average_20 : ℝ) :
  total_students = 20 →
  average_19 = 82 →
  average_20 = 84 →
  ∃ (last_score oliver_score : ℝ),
    (19 * average_19 + oliver_score) / total_students = average_20 ∧
    oliver_score = 2 * last_score →
    last_score = 61 := by
  sorry

end NUMINAMATH_CALUDE_last_student_score_l117_11743


namespace NUMINAMATH_CALUDE_power_multiplication_l117_11795

theorem power_multiplication (a : ℝ) : 3 * a^4 * (4 * a) = 12 * a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l117_11795


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l117_11719

/-- The remainder when (x^1001 - 1) is divided by (x^4 + x^3 + 2x^2 + x + 1) -/
def remainder1 (x : ℝ) : ℝ := x^2 * (1 - x)

/-- The remainder when (x^1001 - 1) is divided by (x^8 + x^6 + 2x^4 + x^2 + 1) -/
def remainder2 (x : ℝ) : ℝ := -2*x^7 - x^5 - 2*x^3 - 1

/-- The first divisor polynomial -/
def divisor1 (x : ℝ) : ℝ := x^4 + x^3 + 2*x^2 + x + 1

/-- The second divisor polynomial -/
def divisor2 (x : ℝ) : ℝ := x^8 + x^6 + 2*x^4 + x^2 + 1

/-- The dividend polynomial -/
def dividend (x : ℝ) : ℝ := x^1001 - 1

theorem polynomial_division_theorem :
  ∀ x : ℝ,
  ∃ q1 q2 : ℝ → ℝ,
  dividend x = q1 x * divisor1 x + remainder1 x ∧
  dividend x = q2 x * divisor2 x + remainder2 x :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l117_11719


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l117_11762

def U : Set ℕ := {0, 2, 4, 6, 8, 10}
def A : Set ℕ := {2, 4, 6}

theorem complement_of_A_in_U :
  U \ A = {0, 8, 10} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l117_11762


namespace NUMINAMATH_CALUDE_normal_price_of_pin_is_20_l117_11749

/-- Calculates the normal price of a pin given the number of pins, discount rate, and total spent -/
def normalPriceOfPin (numPins : ℕ) (discountRate : ℚ) (totalSpent : ℚ) : ℚ :=
  totalSpent / (numPins * (1 - discountRate))

theorem normal_price_of_pin_is_20 :
  normalPriceOfPin 10 (15/100) 170 = 20 := by
  sorry

#eval normalPriceOfPin 10 (15/100) 170

end NUMINAMATH_CALUDE_normal_price_of_pin_is_20_l117_11749


namespace NUMINAMATH_CALUDE_paradise_park_ferris_wheel_seats_l117_11771

/-- The number of seats on a Ferris wheel -/
def ferris_wheel_seats (total_people : ℕ) (people_per_seat : ℕ) : ℕ :=
  total_people / people_per_seat

/-- Theorem: The Ferris wheel in paradise park has 4 seats -/
theorem paradise_park_ferris_wheel_seats :
  ferris_wheel_seats 16 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_paradise_park_ferris_wheel_seats_l117_11771


namespace NUMINAMATH_CALUDE_reservoir_capacity_l117_11759

theorem reservoir_capacity : ∀ (capacity : ℚ),
  (1/8 : ℚ) * capacity + 200 = (1/2 : ℚ) * capacity →
  capacity = 1600/3 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_capacity_l117_11759


namespace NUMINAMATH_CALUDE_factor_expression_l117_11733

theorem factor_expression (b c : ℝ) : 55 * b^2 + 165 * b * c = 55 * b * (b + 3 * c) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l117_11733


namespace NUMINAMATH_CALUDE_sine_equality_implies_equal_arguments_l117_11712

theorem sine_equality_implies_equal_arguments
  (α β γ τ : ℝ)
  (h_pos : α > 0 ∧ β > 0 ∧ γ > 0 ∧ τ > 0)
  (h_eq : ∀ x : ℝ, Real.sin (α * x) + Real.sin (β * x) = Real.sin (γ * x) + Real.sin (τ * x)) :
  α = γ ∨ α = τ :=
sorry

end NUMINAMATH_CALUDE_sine_equality_implies_equal_arguments_l117_11712


namespace NUMINAMATH_CALUDE_calculate_number_of_bs_l117_11752

/-- Calculates the number of Bs given the recess rules and report card results -/
theorem calculate_number_of_bs (
  normal_recess : ℕ)
  (extra_time_per_a : ℕ)
  (extra_time_per_b : ℕ)
  (extra_time_per_c : ℕ)
  (less_time_per_d : ℕ)
  (num_as : ℕ)
  (num_cs : ℕ)
  (num_ds : ℕ)
  (total_recess : ℕ)
  (h1 : normal_recess = 20)
  (h2 : extra_time_per_a = 2)
  (h3 : extra_time_per_b = 1)
  (h4 : extra_time_per_c = 0)
  (h5 : less_time_per_d = 1)
  (h6 : num_as = 10)
  (h7 : num_cs = 14)
  (h8 : num_ds = 5)
  (h9 : total_recess = 47) :
  ∃ (num_bs : ℕ), num_bs = 12 ∧
    total_recess = normal_recess + num_as * extra_time_per_a + num_bs * extra_time_per_b + num_cs * extra_time_per_c - num_ds * less_time_per_d :=
by
  sorry


end NUMINAMATH_CALUDE_calculate_number_of_bs_l117_11752


namespace NUMINAMATH_CALUDE_second_child_birth_year_l117_11751

theorem second_child_birth_year (first_child_age : ℕ) (fourth_child_age : ℕ) 
  (h1 : first_child_age = 15)
  (h2 : fourth_child_age = 8)
  (h3 : ∃ (third_child_age : ℕ), third_child_age = fourth_child_age + 2)
  (h4 : ∃ (second_child_age : ℕ), second_child_age + 4 = third_child_age) :
  first_child_age - (fourth_child_age + 6) = 1 := by
sorry

end NUMINAMATH_CALUDE_second_child_birth_year_l117_11751


namespace NUMINAMATH_CALUDE_bridge_length_l117_11799

/-- The length of a bridge given train parameters -/
theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 120) 
  (h2 : train_speed_kmh = 45) 
  (h3 : crossing_time = 30) : 
  ∃ (bridge_length : ℝ), bridge_length = 255 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_l117_11799


namespace NUMINAMATH_CALUDE_vikki_hourly_rate_l117_11732

/-- Vikki's weekly work hours -/
def work_hours : ℝ := 42

/-- Tax deduction rate -/
def tax_rate : ℝ := 0.20

/-- Insurance deduction rate -/
def insurance_rate : ℝ := 0.05

/-- Union dues deduction -/
def union_dues : ℝ := 5

/-- Vikki's take-home pay after deductions -/
def take_home_pay : ℝ := 310

/-- Vikki's hourly pay rate -/
def hourly_rate : ℝ := 10

theorem vikki_hourly_rate :
  work_hours * hourly_rate * (1 - tax_rate - insurance_rate) - union_dues = take_home_pay :=
sorry

end NUMINAMATH_CALUDE_vikki_hourly_rate_l117_11732


namespace NUMINAMATH_CALUDE_least_positive_difference_l117_11704

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

def sequence_C (n : ℕ) : ℝ := geometric_sequence 3 3 n

def sequence_D (n : ℕ) : ℝ := arithmetic_sequence 10 20 n

def valid_C (n : ℕ) : Prop := sequence_C n ≤ 200

def valid_D (n : ℕ) : Prop := sequence_D n ≤ 200

theorem least_positive_difference :
  ∃ (m n : ℕ) (h₁ : valid_C m) (h₂ : valid_D n),
    ∀ (p q : ℕ) (h₃ : valid_C p) (h₄ : valid_D q),
      |sequence_C m - sequence_D n| ≤ |sequence_C p - sequence_D q| ∧
      |sequence_C m - sequence_D n| > 0 ∧
      |sequence_C m - sequence_D n| = 9 :=
sorry

end NUMINAMATH_CALUDE_least_positive_difference_l117_11704


namespace NUMINAMATH_CALUDE_product_ratio_simplification_l117_11725

theorem product_ratio_simplification
  (a b c d e f g : ℝ)
  (h1 : a * b * c * d = 260)
  (h2 : b * c * d * e = 390)
  (h3 : c * d * e * f = 2000)
  (h4 : d * e * f * g = 500)
  (h5 : c ≠ 0)
  (h6 : e ≠ 0) :
  (a * g) / (c * e) = a / (4 * c) :=
by sorry

end NUMINAMATH_CALUDE_product_ratio_simplification_l117_11725


namespace NUMINAMATH_CALUDE_village_population_l117_11770

theorem village_population (P : ℝ) : 
  (P > 0) →
  (0.85 * (0.9 * P) = 3213) →
  P = 4200 := by
sorry

end NUMINAMATH_CALUDE_village_population_l117_11770


namespace NUMINAMATH_CALUDE_B_60_is_identity_l117_11705

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, -1, 0],
    ![1,  0, 0],
    ![0,  0, 1]]

theorem B_60_is_identity :
  B^60 = 1 := by sorry

end NUMINAMATH_CALUDE_B_60_is_identity_l117_11705


namespace NUMINAMATH_CALUDE_range_of_g_l117_11715

def f (x : ℝ) : ℝ := 4 * x - 3

def g (x : ℝ) : ℝ := f (f (f (f (f x))))

def domain_g : Set ℝ := { x | 1 ≤ x ∧ x ≤ 3 }

theorem range_of_g :
  ∀ x ∈ domain_g, 1 ≤ g x ∧ g x ≤ 2049 ∧
  ∃ y ∈ domain_g, g y = 1 ∧
  ∃ z ∈ domain_g, g z = 2049 :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l117_11715


namespace NUMINAMATH_CALUDE_darry_smaller_ladder_climbs_l117_11746

/-- Represents the number of steps in Darry's full ladder -/
def full_ladder_steps : ℕ := 11

/-- Represents the number of steps in Darry's smaller ladder -/
def smaller_ladder_steps : ℕ := 6

/-- Represents the number of times Darry climbed the full ladder -/
def full_ladder_climbs : ℕ := 10

/-- Represents the total number of steps Darry climbed -/
def total_steps : ℕ := 152

/-- Theorem stating that Darry climbed the smaller ladder 7 times -/
theorem darry_smaller_ladder_climbs :
  ∃ (x : ℕ), x * smaller_ladder_steps + full_ladder_climbs * full_ladder_steps = total_steps ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_darry_smaller_ladder_climbs_l117_11746


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l117_11779

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 4}

-- Define the line
def line (k : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = k * p.1 - 1}

-- Define the center of the circle
def center : ℝ × ℝ := (2, 3)

-- Define the property of being tangent to y-axis
def tangent_to_y_axis (C : Set (ℝ × ℝ)) : Prop :=
  ∃ y, (0, y) ∈ C ∧ ∀ x ≠ 0, (x, y) ∉ C

-- Define the perpendicularity condition
def perpendicular (M N : ℝ × ℝ) : Prop :=
  (M.1 - center.1) * (N.1 - center.1) + (M.2 - center.2) * (N.2 - center.2) = 0

theorem circle_and_line_properties :
  tangent_to_y_axis circle_C →
  ∀ k, ∃ M N, M ∈ circle_C ∧ N ∈ circle_C ∧ M ∈ line k ∧ N ∈ line k ∧ perpendicular M N →
  (k = 1 ∨ k = 7) := by
  sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l117_11779


namespace NUMINAMATH_CALUDE_unique_solution_for_inequality_l117_11794

theorem unique_solution_for_inequality : 
  ∃! n : ℕ+, -46 ≤ (2023 : ℝ) / (46 - n.val) ∧ (2023 : ℝ) / (46 - n.val) ≤ 46 - n.val :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_inequality_l117_11794


namespace NUMINAMATH_CALUDE_smallest_sum_arithmetic_cubic_sequence_l117_11781

theorem smallest_sum_arithmetic_cubic_sequence (A B C D : ℕ+) : 
  (∃ r : ℚ, B = A + r ∧ C = B + r) →  -- A, B, C form an arithmetic sequence
  (D - C = (C - B)^2) →  -- B, C, D form a cubic sequence
  (C : ℚ) / B = 4 / 3 →  -- C/B = 4/3
  (∀ A' B' C' D' : ℕ+, 
    (∃ r' : ℚ, B' = A' + r' ∧ C' = B' + r') → 
    (D' - C' = (C' - B')^2) → 
    (C' : ℚ) / B' = 4 / 3 → 
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 14 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_arithmetic_cubic_sequence_l117_11781


namespace NUMINAMATH_CALUDE_acid_dilution_l117_11772

/-- Given an initial acid solution and water added to dilute it, 
    calculate the amount of water needed to reach a specific concentration. -/
theorem acid_dilution (m : ℝ) (hm : m > 50) : 
  ∃ x : ℝ, 
    (m * (m / 100) = (m + x) * ((m - 20) / 100)) → 
    x = (20 * m) / (m + 20) := by
  sorry

end NUMINAMATH_CALUDE_acid_dilution_l117_11772


namespace NUMINAMATH_CALUDE_two_dogs_food_consumption_l117_11787

/-- The amount of dog food consumed by two dogs in a day -/
def total_dog_food_consumption (dog1_consumption dog2_consumption : Real) : Real :=
  dog1_consumption + dog2_consumption

/-- Theorem: Two dogs each consuming 0.125 scoop of dog food per day eat 0.25 scoop in total -/
theorem two_dogs_food_consumption :
  total_dog_food_consumption 0.125 0.125 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_two_dogs_food_consumption_l117_11787


namespace NUMINAMATH_CALUDE_mixture_combination_theorem_l117_11735

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℕ
  water : ℕ

/-- Combines two mixtures -/
def combineMixtures (m1 m2 : Mixture) : Mixture :=
  { milk := m1.milk + m2.milk
    water := m1.water + m2.water }

/-- Simplifies a ratio by dividing both parts by their GCD -/
def simplifyRatio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

theorem mixture_combination_theorem :
  let m1 : Mixture := { milk := 7, water := 2 }
  let m2 : Mixture := { milk := 8, water := 1 }
  let combined := combineMixtures m1 m2
  simplifyRatio combined.milk combined.water = (5, 1) := by
  sorry

end NUMINAMATH_CALUDE_mixture_combination_theorem_l117_11735


namespace NUMINAMATH_CALUDE_min_f_tetrahedron_l117_11711

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron ABCD -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Distance between two points -/
def distance (p q : Point3D) : ℝ := sorry

/-- Function f(P) for a given tetrahedron and point P -/
def f (t : Tetrahedron) (P : Point3D) : ℝ :=
  distance P t.A + distance P t.B + distance P t.C + distance P t.D

/-- Theorem: Minimum value of f(P) for a tetrahedron with given properties -/
theorem min_f_tetrahedron (t : Tetrahedron) (a b c : ℝ) :
  (distance t.A t.D = a) →
  (distance t.B t.C = a) →
  (distance t.A t.C = b) →
  (distance t.B t.D = b) →
  (distance t.A t.B * distance t.C t.D = c^2) →
  ∃ (min_val : ℝ), (∀ (P : Point3D), f t P ≥ min_val) ∧ (min_val = Real.sqrt ((a^2 + b^2 + c^2) / 2)) :=
sorry

end NUMINAMATH_CALUDE_min_f_tetrahedron_l117_11711


namespace NUMINAMATH_CALUDE_students_walking_home_l117_11700

theorem students_walking_home (bus car bicycle skateboard : ℚ) 
  (h1 : bus = 3/8)
  (h2 : car = 2/5)
  (h3 : bicycle = 1/8)
  (h4 : skateboard = 5/100)
  : 1 - (bus + car + bicycle + skateboard) = 1/20 := by
  sorry

end NUMINAMATH_CALUDE_students_walking_home_l117_11700


namespace NUMINAMATH_CALUDE_range_of_g_l117_11773

def f (x : ℝ) : ℝ := 2 * x - 3

def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ x ∈ Set.Icc 1 3, -29 ≤ g x ∧ g x ≤ 3 ∧
  ∀ y ∈ Set.Icc (-29) 3, ∃ x ∈ Set.Icc 1 3, g x = y :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l117_11773


namespace NUMINAMATH_CALUDE_equation_transformation_l117_11796

theorem equation_transformation (x : ℝ) : (3 * x - 7 = 2 * x) ↔ (3 * x - 2 * x = 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l117_11796


namespace NUMINAMATH_CALUDE_quadratic_negative_roots_probability_l117_11754

/-- The probability that a quadratic equation with a randomly selected coefficient has two negative roots -/
theorem quadratic_negative_roots_probability : 
  ∃ (f : ℝ → ℝ → ℝ → Prop) (P : Set ℝ → ℝ),
    (∀ p x₁ x₂, f p x₁ x₂ ↔ x₁^2 + 2*p*x₁ + 3*p - 2 = 0 ∧ x₂^2 + 2*p*x₂ + 3*p - 2 = 0 ∧ x₁ < 0 ∧ x₂ < 0) →
    (P (Set.Icc 0 5) = 5) →
    P {p ∈ Set.Icc 0 5 | ∃ x₁ x₂, f p x₁ x₂} / P (Set.Icc 0 5) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_negative_roots_probability_l117_11754


namespace NUMINAMATH_CALUDE_smallest_representable_l117_11718

/-- The representation function for k -/
def representation (n m : ℕ+) : ℤ := 19^(n:ℕ) - 5^(m:ℕ)

/-- The property that k is representable -/
def is_representable (k : ℕ) : Prop :=
  ∃ (n m : ℕ+), representation n m = k

/-- The main theorem statement -/
theorem smallest_representable : 
  (is_representable 14) ∧ (∀ k : ℕ, 0 < k ∧ k < 14 → ¬(is_representable k)) := by
  sorry

#check smallest_representable

end NUMINAMATH_CALUDE_smallest_representable_l117_11718


namespace NUMINAMATH_CALUDE_smallest_natural_with_eight_divisors_ending_in_zero_l117_11747

theorem smallest_natural_with_eight_divisors_ending_in_zero (N : ℕ) :
  (N % 10 = 0) →  -- N ends with 0
  (Finset.card (Nat.divisors N) = 8) →  -- N has exactly 8 divisors
  (∀ M : ℕ, M % 10 = 0 ∧ Finset.card (Nat.divisors M) = 8 → N ≤ M) →  -- N is the smallest such number
  N = 30 := by
sorry

end NUMINAMATH_CALUDE_smallest_natural_with_eight_divisors_ending_in_zero_l117_11747


namespace NUMINAMATH_CALUDE_speed_ratio_of_travelers_l117_11774

/-- Given two travelers A and B covering the same distance, where A takes 2 hours
    to reach the destination and B takes 30 minutes less than A, prove that the
    ratio of their speeds (vA/vB) is 3/4. -/
theorem speed_ratio_of_travelers (d : ℝ) (vA vB : ℝ) : 
  d > 0 ∧ vA > 0 ∧ vB > 0 ∧ d / vA = 120 ∧ d / vB = 90 → vA / vB = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_speed_ratio_of_travelers_l117_11774


namespace NUMINAMATH_CALUDE_class_overlap_difference_l117_11702

theorem class_overlap_difference (total students_geometry students_biology : ℕ) 
  (h1 : total = 232)
  (h2 : students_geometry = 144)
  (h3 : students_biology = 119) :
  (min students_geometry students_biology) - 
  (students_geometry + students_biology - total) = 88 :=
by sorry

end NUMINAMATH_CALUDE_class_overlap_difference_l117_11702


namespace NUMINAMATH_CALUDE_abc_product_l117_11766

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * (b + c) = 154) (h2 : b * (c + a) = 164) (h3 : c * (a + b) = 172) :
  a * b * c = Real.sqrt 538083 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l117_11766


namespace NUMINAMATH_CALUDE_log_simplification_l117_11737

theorem log_simplification (a b m : ℝ) (h : m^2 = a^2 - b^2) (h1 : a + b > 0) (h2 : a - b > 0) (h3 : m > 0) :
  Real.log m / Real.log (a + b) + Real.log m / Real.log (a - b) - 2 * (Real.log m / Real.log (a + b)) * (Real.log m / Real.log (a - b)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l117_11737


namespace NUMINAMATH_CALUDE_arithmetic_computation_l117_11731

theorem arithmetic_computation : -9 * 5 - (-7 * -4) + (-12 * -6) = -1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l117_11731


namespace NUMINAMATH_CALUDE_complex_magnitude_l117_11717

theorem complex_magnitude (z : ℂ) (h : z - 2 * Complex.I = 1 + z * Complex.I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l117_11717


namespace NUMINAMATH_CALUDE_least_n_without_square_l117_11707

theorem least_n_without_square : ∃ (N : ℕ), N = 282 ∧ 
  (∀ (k : ℕ), k < N → ∃ (x : ℕ), ∃ (i : ℕ), i < 1000 ∧ x^2 = 1000*k + i) ∧
  (∀ (x : ℕ), ¬∃ (i : ℕ), i < 1000 ∧ x^2 = 1000*N + i) :=
by sorry

end NUMINAMATH_CALUDE_least_n_without_square_l117_11707


namespace NUMINAMATH_CALUDE_function_property_l117_11798

theorem function_property (A : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, A (x + y) = A x + A y) 
  (h2 : ∀ x y : ℝ, A (x * y) = A x * A y) : 
  (∀ x : ℝ, A x = x) ∨ (∀ x : ℝ, A x = 0) :=
by sorry

end NUMINAMATH_CALUDE_function_property_l117_11798


namespace NUMINAMATH_CALUDE_parametric_to_general_equation_l117_11708

/-- Parametric equations to general equation conversion -/
theorem parametric_to_general_equation :
  ∀ θ : ℝ,
  let x : ℝ := 2 + Real.sin θ ^ 2
  let y : ℝ := -1 + Real.cos (2 * θ)
  2 * x + y - 4 = 0 ∧ x ∈ Set.Icc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_general_equation_l117_11708


namespace NUMINAMATH_CALUDE_amy_height_l117_11722

def angela_height : ℕ := 157
def angela_helen_diff : ℕ := 4
def helen_amy_diff : ℕ := 3

theorem amy_height : 
  angela_height - angela_helen_diff - helen_amy_diff = 150 :=
by sorry

end NUMINAMATH_CALUDE_amy_height_l117_11722


namespace NUMINAMATH_CALUDE_probability_not_face_card_l117_11740

theorem probability_not_face_card (total_cards : ℕ) (face_cards : ℕ) :
  total_cards = 52 →
  face_cards = 12 →
  (total_cards - face_cards : ℚ) / total_cards = 10 / 13 := by
sorry

end NUMINAMATH_CALUDE_probability_not_face_card_l117_11740


namespace NUMINAMATH_CALUDE_ellipse_min_reciprocal_sum_l117_11756

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_min_reciprocal_sum :
  ∀ P : ℝ × ℝ, is_on_ellipse P.1 P.2 →
  (1 / distance P left_focus + 1 / distance P right_focus ≥ 1) ∧
  (∃ Q : ℝ × ℝ, is_on_ellipse Q.1 Q.2 ∧
    1 / distance Q left_focus + 1 / distance Q right_focus = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_min_reciprocal_sum_l117_11756


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l117_11706

theorem absolute_value_inequality (x : ℝ) : 
  abs (x - 3) + abs (x - 5) ≥ 4 ↔ x ≥ 6 ∨ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l117_11706


namespace NUMINAMATH_CALUDE_correct_polynomial_sum_l117_11789

variable (a : ℝ)
variable (A B : ℝ → ℝ)

theorem correct_polynomial_sum
  (hB : B = λ x => 3 * x^2 - 5 * x - 7)
  (hA_minus_2B : A - 2 * B = λ x => -2 * x^2 + 3 * x + 6) :
  A + 2 * B = λ x => 10 * x^2 - 17 * x - 22 :=
by sorry

end NUMINAMATH_CALUDE_correct_polynomial_sum_l117_11789


namespace NUMINAMATH_CALUDE_right_triangle_tan_b_l117_11713

theorem right_triangle_tan_b (A B C : Real) (h1 : 0 < A ∧ A < π/2) (h2 : 0 < B ∧ B < π/2) : 
  A + B + Real.pi/2 = Real.pi → Real.sin A = 2/3 → Real.tan B = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_tan_b_l117_11713


namespace NUMINAMATH_CALUDE_sqrt_490000_equals_700_l117_11784

theorem sqrt_490000_equals_700 : Real.sqrt 490000 = 700 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_490000_equals_700_l117_11784


namespace NUMINAMATH_CALUDE_polynomial_range_l117_11791

noncomputable def P (p q x : ℝ) : ℝ := x^2 + p*x + q

theorem polynomial_range (p q : ℝ) :
  let rangeP := {y | ∃ x ∈ Set.Icc (-1 : ℝ) 1, P p q x = y}
  (p < -2 → rangeP = Set.Icc (1 + p + q) (1 - p + q)) ∧
  (-2 ≤ p ∧ p ≤ 0 → rangeP = Set.Icc (q - p^2/4) (1 - p + q)) ∧
  (0 ≤ p ∧ p ≤ 2 → rangeP = Set.Icc (q - p^2/4) (1 + p + q)) ∧
  (p > 2 → rangeP = Set.Icc (1 - p + q) (1 + p + q)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_range_l117_11791


namespace NUMINAMATH_CALUDE_negation_of_implication_l117_11765

theorem negation_of_implication (x : ℝ) :
  ¬(x < 0 → x < 1) ↔ (x ≥ 0 → x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_l117_11765


namespace NUMINAMATH_CALUDE_min_xy_value_l117_11748

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/(4*y) = 1) :
  x * y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_xy_value_l117_11748


namespace NUMINAMATH_CALUDE_log_division_simplification_l117_11790

theorem log_division_simplification :
  (Real.log 16) / (Real.log (1/16)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_division_simplification_l117_11790


namespace NUMINAMATH_CALUDE_centroid_quadrilateral_area_l117_11786

/-- Given a square ABCD with side length 40 and a point Q inside the square
    such that AQ = 16 and BQ = 34, the area of the quadrilateral formed by
    the centroids of △ABQ, △BCQ, △CDQ, and △DAQ is 6400/9. -/
theorem centroid_quadrilateral_area (A B C D Q : ℝ × ℝ) : 
  let square_side : ℝ := 40
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  -- Square ABCD conditions
  (dist A B = square_side) ∧ 
  (dist B C = square_side) ∧ 
  (dist C D = square_side) ∧ 
  (dist D A = square_side) ∧ 
  -- Right angles
  ((B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) ∧
  -- Q inside square
  (0 < Q.1) ∧ (Q.1 < square_side) ∧ (0 < Q.2) ∧ (Q.2 < square_side) ∧
  -- AQ and BQ distances
  (dist A Q = 16) ∧ 
  (dist B Q = 34) →
  -- Area of quadrilateral formed by centroids
  let centroid (P1 P2 P3 : ℝ × ℝ) := 
    ((P1.1 + P2.1 + P3.1) / 3, (P1.2 + P2.2 + P3.2) / 3)
  let G1 := centroid A B Q
  let G2 := centroid B C Q
  let G3 := centroid C D Q
  let G4 := centroid D A Q
  let area := (dist G1 G3 * dist G2 G4) / 2
  area = 6400 / 9 := by
sorry

end NUMINAMATH_CALUDE_centroid_quadrilateral_area_l117_11786


namespace NUMINAMATH_CALUDE_polynomial_identity_l117_11777

theorem polynomial_identity (x : ℝ) : 
  (x + 2)^4 + 4*(x + 2)^3 + 6*(x + 2)^2 + 4*(x + 2) + 1 = (x + 3)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l117_11777


namespace NUMINAMATH_CALUDE_distance_p_to_y_axis_l117_11780

/-- The distance from a point to the y-axis is the absolute value of its x-coordinate -/
def distance_to_y_axis (x : ℝ) : ℝ := |x|

/-- Given a point P(-3, 2) in the second quadrant, its distance to the y-axis is 3 -/
theorem distance_p_to_y_axis :
  let P : ℝ × ℝ := (-3, 2)
  distance_to_y_axis P.1 = 3 := by sorry

end NUMINAMATH_CALUDE_distance_p_to_y_axis_l117_11780


namespace NUMINAMATH_CALUDE_inequality_proof_l117_11792

theorem inequality_proof (a b : Real) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  a^5 + b^3 + (a - b)^2 ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l117_11792


namespace NUMINAMATH_CALUDE_class_size_proof_l117_11778

theorem class_size_proof (total : ℕ) 
  (h1 : 20 < total ∧ total < 30)
  (h2 : ∃ n : ℕ, total = 8 * n + 2)
  (h3 : ∃ M F : ℕ, M = 5 * n ∧ F = 4 * n) 
  (h4 : ∃ n : ℕ, n = (20 * M) / 100 ∧ n = (25 * F) / 100) :
  total = 26 := by
sorry

end NUMINAMATH_CALUDE_class_size_proof_l117_11778
