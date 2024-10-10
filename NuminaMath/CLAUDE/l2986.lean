import Mathlib

namespace hcf_problem_l2986_298684

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 62216) (h2 : Nat.lcm a b = 2828) :
  Nat.gcd a b = 22 := by
  sorry

end hcf_problem_l2986_298684


namespace union_of_M_and_N_l2986_298616

def M : Set ℤ := {0, 1}
def N : Set ℤ := {x | x^2 + x = 0}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1} := by sorry

end union_of_M_and_N_l2986_298616


namespace some_employees_not_in_management_team_l2986_298696

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Employee : U → Prop)
variable (ManagementTeam : U → Prop)
variable (CompletesTraining : U → Prop)

-- State the theorem
theorem some_employees_not_in_management_team
  (h1 : ∃ x, Employee x ∧ ¬CompletesTraining x)
  (h2 : ∀ x, ManagementTeam x → CompletesTraining x) :
  ∃ x, Employee x ∧ ¬ManagementTeam x :=
by sorry

end some_employees_not_in_management_team_l2986_298696


namespace line_through_points_with_45_degree_angle_l2986_298669

/-- A line passes through points A(m,2) and B(-m,2m-1) with an inclination angle of 45° -/
theorem line_through_points_with_45_degree_angle (m : ℝ) : 
  (∃ (line : Set (ℝ × ℝ)), 
    (m, 2) ∈ line ∧ 
    (-m, 2*m - 1) ∈ line ∧ 
    (∀ (x y : ℝ), (x, y) ∈ line → (y - 2) / (x - m) = 1)) → 
  m = 3/4 := by
sorry

end line_through_points_with_45_degree_angle_l2986_298669


namespace quadratic_root_property_l2986_298672

theorem quadratic_root_property (n : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁^2 - 3*x₁ + n = 0) ∧ (x₂^2 - 3*x₂ + n = 0) ∧ (x₁ + x₂ - 2 = x₁ * x₂)) → n = 1 := by
  sorry

end quadratic_root_property_l2986_298672


namespace jonah_running_time_l2986_298611

/-- Represents the problem of determining Jonah's running time. -/
theorem jonah_running_time (calories_per_hour : ℕ) (extra_time : ℕ) (extra_calories : ℕ) : 
  calories_per_hour = 30 →
  extra_time = 5 →
  extra_calories = 90 →
  ∃ (actual_time : ℕ), 
    actual_time * calories_per_hour = (actual_time + extra_time) * calories_per_hour - extra_calories ∧
    actual_time = 2 :=
by sorry

end jonah_running_time_l2986_298611


namespace sophies_spend_is_72_80_l2986_298618

/-- The total amount Sophie spends on her purchases -/
def sophies_total_spend : ℚ :=
  let cupcakes := 5 * 2
  let doughnuts := 6 * 1
  let apple_pie := 4 * 2
  let cookies := 15 * 0.6
  let chocolate_bars := 8 * 1.5
  let soda := 12 * 1.2
  let gum := 3 * 0.8
  let chips := 10 * 1.1
  cupcakes + doughnuts + apple_pie + cookies + chocolate_bars + soda + gum + chips

/-- Theorem stating that Sophie's total spend is $72.80 -/
theorem sophies_spend_is_72_80 : sophies_total_spend = 72.8 := by
  sorry

end sophies_spend_is_72_80_l2986_298618


namespace sunset_time_l2986_298685

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents the length of a time period in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

def addTime (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + d.hours * 60 + d.minutes
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

def sunrise : Time := { hours := 6, minutes := 12 }
def daylightLength : Duration := { hours := 12, minutes := 36 }

theorem sunset_time :
  addTime sunrise daylightLength = { hours := 18, minutes := 48 } := by
  sorry

end sunset_time_l2986_298685


namespace cyclist_speed_problem_l2986_298606

/-- The speed of cyclist C in mph -/
def speed_C : ℝ := 9

/-- The speed of cyclist D in mph -/
def speed_D : ℝ := speed_C + 6

/-- The distance between Newport and Kingston in miles -/
def distance : ℝ := 80

/-- The distance from Kingston where cyclists meet on D's return journey in miles -/
def meeting_distance : ℝ := 20

theorem cyclist_speed_problem :
  speed_C = 9 ∧
  speed_D = speed_C + 6 ∧
  distance / speed_C = (distance + meeting_distance) / speed_D :=
by sorry

end cyclist_speed_problem_l2986_298606


namespace positive_integer_division_l2986_298630

theorem positive_integer_division (a b : ℕ+) :
  (a * b^2 + b + 7) ∣ (a^2 * b + a + b) ↔
    ((a = 11 ∧ b = 1) ∨
     (a = 49 ∧ b = 1) ∨
     (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) := by
  sorry

end positive_integer_division_l2986_298630


namespace differential_savings_proof_l2986_298624

def calculate_differential_savings (income : ℝ) (old_rate : ℝ) (new_rate : ℝ) : ℝ :=
  income * (old_rate - new_rate)

theorem differential_savings_proof (income : ℝ) (old_rate : ℝ) (new_rate : ℝ) 
  (h1 : income = 48000)
  (h2 : old_rate = 0.45)
  (h3 : new_rate = 0.30) :
  calculate_differential_savings income old_rate new_rate = 7200 := by
  sorry

end differential_savings_proof_l2986_298624


namespace min_value_theorem_l2986_298636

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≤ b + c) (h2 : b + c ≤ 3 * a) (h3 : 3 * b^2 ≤ a * (a + c)) (h4 : a * (a + c) ≤ 5 * b^2) :
  -18/5 ≤ (b - 2*c) / a ∧ ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧
    a₀ ≤ b₀ + c₀ ∧ b₀ + c₀ ≤ 3 * a₀ ∧ 3 * b₀^2 ≤ a₀ * (a₀ + c₀) ∧ a₀ * (a₀ + c₀) ≤ 5 * b₀^2 ∧
    (b₀ - 2*c₀) / a₀ = -18/5 := by
  sorry

end min_value_theorem_l2986_298636


namespace sum_of_non_solutions_l2986_298691

/-- Given an equation with infinitely many solutions, prove the sum of non-solution x values -/
theorem sum_of_non_solutions (A B C : ℝ) : 
  (∀ x : ℝ, (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9)) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (x + B) * (A * x + 36) ≠ 3 * (x + C) * (x + 9) ↔ (x = x₁ ∨ x = x₂)) →
  x₁ + x₂ = -21 :=
sorry

end sum_of_non_solutions_l2986_298691


namespace tens_digit_of_6_pow_22_l2986_298665

/-- The tens digit of 6^n -/
def tens_digit_of_6_pow (n : ℕ) : ℕ :=
  match n % 5 with
  | 0 => 6
  | 1 => 3
  | 2 => 1
  | 3 => 9
  | 4 => 7
  | _ => 0  -- This case should never occur

theorem tens_digit_of_6_pow_22 : tens_digit_of_6_pow 22 = 3 := by
  sorry

end tens_digit_of_6_pow_22_l2986_298665


namespace triangle_area_l2986_298698

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where b = 1, c = √3, and ∠C = 2π/3, prove that its area is √3/4 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = 1 → 
  c = Real.sqrt 3 → 
  C = 2 * Real.pi / 3 → 
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 4 := by
sorry


end triangle_area_l2986_298698


namespace infinite_solutions_imply_specific_coefficients_l2986_298659

theorem infinite_solutions_imply_specific_coefficients :
  ∀ (a b : ℝ),
  (∀ x : ℝ, a * (x - 3) + b * (3 * x + 1) = 5 * (x + 1)) →
  (a = -1 ∧ b = 2) :=
by sorry

end infinite_solutions_imply_specific_coefficients_l2986_298659


namespace correct_score_is_even_l2986_298614

/-- Represents the scoring system for a math competition -/
structure ScoringSystem where
  correct : Int
  unanswered : Int
  incorrect : Int

/-- Represents the results of a class in the math competition -/
structure CompetitionResult where
  total_questions : Nat
  scoring : ScoringSystem
  first_calculation : Int
  second_calculation : Int

/-- Theorem stating that the correct total score must be even -/
theorem correct_score_is_even (result : CompetitionResult) 
  (h1 : result.scoring.correct = 3)
  (h2 : result.scoring.unanswered = 1)
  (h3 : result.scoring.incorrect = -1)
  (h4 : result.total_questions = 50)
  (h5 : result.first_calculation = 5734)
  (h6 : result.second_calculation = 5735)
  (h7 : result.first_calculation = 5734 ∨ result.second_calculation = 5734) :
  ∃ (n : Int), 2 * n = 5734 ∧ (result.first_calculation = 5734 ∨ result.second_calculation = 5734) :=
sorry

end correct_score_is_even_l2986_298614


namespace intersection_point_of_three_lines_l2986_298627

theorem intersection_point_of_three_lines (k b : ℝ) :
  (∀ x y : ℝ, (y = k * x + b) ∧ (y = 2 * k * x + 2 * b) ∧ (y = b * x + k)) →
  (k ≠ b) →
  (∃! p : ℝ × ℝ, 
    (p.2 = k * p.1 + b) ∧ 
    (p.2 = 2 * k * p.1 + 2 * b) ∧ 
    (p.2 = b * p.1 + k) ∧
    p = (1, 0)) :=
by sorry

end intersection_point_of_three_lines_l2986_298627


namespace circle_equation_proof_l2986_298695

/-- Given a circle with center (1, 1) intersecting the line x + y = 4 to form a chord of length 2√3,
    prove that the equation of the circle is (x-1)² + (y-1)² = 5. -/
theorem circle_equation_proof (x y : ℝ) : 
  let center : ℝ × ℝ := (1, 1)
  let line_equation := x + y = 4
  let chord_length : ℝ := 2 * Real.sqrt 3
  true → (x - 1)^2 + (y - 1)^2 = 5 := by
  sorry

end circle_equation_proof_l2986_298695


namespace distance_to_equidistant_line_in_unit_cube_l2986_298635

/-- Represents a 3D point -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a 3D line -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents a unit cube -/
structure UnitCube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  B1 : Point3D
  C1 : Point3D
  D1 : Point3D

/-- Distance between a point and a line in 3D space -/
def distancePointToLine (p : Point3D) (l : Line3D) : ℝ :=
  sorry

/-- Check if two lines are parallel -/
def areLinesParallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Check if a line is equidistant from three other lines -/
def isLineEquidistantFromThreeLines (l l1 l2 l3 : Line3D) : Prop :=
  sorry

/-- The main theorem -/
theorem distance_to_equidistant_line_in_unit_cube 
  (cube : UnitCube) 
  (l : Line3D) 
  (hParallel : areLinesParallel l (Line3D.mk cube.A cube.C1))
  (hEquidistant : isLineEquidistantFromThreeLines l 
    (Line3D.mk cube.B cube.D) 
    (Line3D.mk cube.A1 cube.D1) 
    (Line3D.mk cube.C cube.B1)) :
  distancePointToLine cube.B (Line3D.mk cube.B cube.D) = Real.sqrt 2 / 6 ∧
  distancePointToLine cube.A1 (Line3D.mk cube.A1 cube.D1) = Real.sqrt 2 / 6 ∧
  distancePointToLine cube.C (Line3D.mk cube.C cube.B1) = Real.sqrt 2 / 6 :=
sorry

end distance_to_equidistant_line_in_unit_cube_l2986_298635


namespace point_on_line_l2986_298654

/-- Given two points (m, n) and (m + p, n + 9) on the line x = y/3 - 2/5, prove that p = 3 -/
theorem point_on_line (m n p : ℝ) : 
  (m = n / 3 - 2 / 5) →
  (m + p = (n + 9) / 3 - 2 / 5) →
  p = 3 := by
  sorry

end point_on_line_l2986_298654


namespace two_solutions_iff_a_gt_neg_one_l2986_298662

/-- The equation has exactly two solutions if and only if a > -1 -/
theorem two_solutions_iff_a_gt_neg_one (a : ℝ) :
  (∃! x y, x ≠ y ∧ x^2 + 2*x + 2*|x+1| = a ∧ y^2 + 2*y + 2*|y+1| = a) ↔ a > -1 :=
sorry

end two_solutions_iff_a_gt_neg_one_l2986_298662


namespace remainder_of_1493824_div_4_l2986_298632

theorem remainder_of_1493824_div_4 : 1493824 % 4 = 0 := by
  sorry

end remainder_of_1493824_div_4_l2986_298632


namespace similar_polygons_ratio_l2986_298680

theorem similar_polygons_ratio (A₁ A₂ : ℝ) (s₁ s₂ : ℝ) :
  A₁ / A₂ = 9 / 4 →
  s₁ / s₂ = (A₁ / A₂).sqrt →
  s₁ / s₂ = 3 / 2 :=
by sorry

end similar_polygons_ratio_l2986_298680


namespace quadratic_integer_roots_l2986_298679

theorem quadratic_integer_roots (n : ℕ+) :
  (∃ x : ℤ, x^2 - 4*x + n.val = 0) ↔ (n.val = 3 ∨ n.val = 4) := by
  sorry

end quadratic_integer_roots_l2986_298679


namespace arrangements_a_middle_arrangements_a_b_not_adjacent_arrangements_a_b_not_ends_l2986_298692

-- Define the number of people
def n : ℕ := 5

-- Define the function for number of permutations
def permutations (n : ℕ) (r : ℕ) : ℕ := n.factorial / (n - r).factorial

-- Theorem 1: Person A in the middle
theorem arrangements_a_middle : permutations (n - 1) (n - 1) = 24 := by sorry

-- Theorem 2: Person A and B not adjacent
theorem arrangements_a_b_not_adjacent : 
  (permutations 3 3) * (permutations 4 2) = 72 := by sorry

-- Theorem 3: Person A and B not at ends
theorem arrangements_a_b_not_ends : 
  (permutations 3 2) * (permutations 3 3) = 36 := by sorry

end arrangements_a_middle_arrangements_a_b_not_adjacent_arrangements_a_b_not_ends_l2986_298692


namespace six_people_arrangement_l2986_298697

/-- The number of arrangements with A at the edge -/
def edge_arrangements : ℕ := 4 * 3 * 24

/-- The number of arrangements with A in the middle -/
def middle_arrangements : ℕ := 2 * 2 * 24

/-- The total number of valid arrangements -/
def total_arrangements : ℕ := edge_arrangements + middle_arrangements

theorem six_people_arrangement :
  total_arrangements = 384 :=
sorry

end six_people_arrangement_l2986_298697


namespace min_value_on_circle_l2986_298655

theorem min_value_on_circle (x y : ℝ) (h : x^2 + y^2 = 2) :
  (1 / (1 + x^2) + 1 / (1 + y^2)) ≥ 1 ∧
  (1 / (1 + x^2) + 1 / (1 + y^2) = 1 ↔ x^2 = 1 ∧ y^2 = 1) :=
by sorry

end min_value_on_circle_l2986_298655


namespace kennel_cats_dogs_difference_l2986_298615

theorem kennel_cats_dogs_difference (num_dogs : ℕ) (num_cats : ℕ) : 
  num_dogs = 32 →
  num_cats * 4 = num_dogs * 3 →
  num_dogs - num_cats = 8 :=
by
  sorry

end kennel_cats_dogs_difference_l2986_298615


namespace same_solution_implies_zero_power_l2986_298686

theorem same_solution_implies_zero_power (a b : ℝ) :
  (∃ x y : ℝ, 4*x + 3*y = 11 ∧ a*x + b*y = -2 ∧ 2*x - y = 3 ∧ b*x - a*y = 6) →
  (a + b)^2023 = 0 := by
sorry

end same_solution_implies_zero_power_l2986_298686


namespace candle_height_after_80000_seconds_l2986_298651

/-- Represents the burning pattern of a candle -/
structure BurningPattern where
  oddCentimeterTime : ℕ → ℕ  -- Time to burn odd-numbered centimeters
  evenCentimeterTime : ℕ → ℕ -- Time to burn even-numbered centimeters

/-- Calculates the remaining height of a candle after a given time -/
def remainingHeight (initialHeight : ℕ) (pattern : BurningPattern) (elapsedTime : ℕ) : ℕ :=
  sorry

/-- The specific burning pattern for this problem -/
def candlePattern : BurningPattern :=
  { oddCentimeterTime := λ k => 10 * k,
    evenCentimeterTime := λ k => 15 * k }

/-- Theorem stating the remaining height of the candle after 80,000 seconds -/
theorem candle_height_after_80000_seconds :
  remainingHeight 150 candlePattern 80000 = 70 :=
sorry

end candle_height_after_80000_seconds_l2986_298651


namespace heart_club_probability_l2986_298694

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size : cards.card = 52)

/-- Represents the suit of a card -/
inductive Suit
| Hearts | Clubs | Diamonds | Spades

/-- The probability of drawing a heart first and a club second from a standard deck -/
def prob_heart_then_club (d : Deck) : ℚ :=
  (13 : ℚ) / 204

/-- Theorem stating the probability of drawing a heart first and a club second -/
theorem heart_club_probability (d : Deck) :
  prob_heart_then_club d = 13 / 204 := by
  sorry

end heart_club_probability_l2986_298694


namespace negation_of_universal_statement_l2986_298690

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, Real.exp x > Real.log x) ↔ (∃ x₀ : ℝ, Real.exp x₀ ≤ Real.log x₀) :=
by sorry

end negation_of_universal_statement_l2986_298690


namespace parabola_directrix_l2986_298663

/-- Given a parabola with equation x = -2y^2, its directrix has equation x = 1/8 -/
theorem parabola_directrix (y : ℝ) : 
  (∃ x : ℝ, x = -2 * y^2) → 
  (∃ x : ℝ, x = 1/8 ∧ ∀ y : ℝ, (y, x) ∈ {p : ℝ × ℝ | p.1 = 1/8}) :=
by sorry

end parabola_directrix_l2986_298663


namespace solution_set_f_max_value_f_range_of_m_l2986_298631

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 3| - |x + 5|

-- Theorem for the solution set of f(x) ≥ 2
theorem solution_set_f (x : ℝ) : f x ≥ 2 ↔ x ≤ -2 := by sorry

-- Theorem for the maximum value of f(x)
theorem max_value_f : ∃ M : ℝ, M = 8 ∧ ∀ x : ℝ, f x ≤ M := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + m ≤ 8) ↔ m ≤ 9 := by sorry

end solution_set_f_max_value_f_range_of_m_l2986_298631


namespace journey_fraction_by_foot_l2986_298670

/-- Given a journey with a total distance of 24 km, where 1/4 of the distance
    is traveled by bus and 6 km is traveled by car, prove that the fraction
    of the distance traveled by foot is 1/2. -/
theorem journey_fraction_by_foot :
  ∀ (total_distance bus_fraction car_distance foot_distance : ℝ),
    total_distance = 24 →
    bus_fraction = 1/4 →
    car_distance = 6 →
    foot_distance = total_distance - (bus_fraction * total_distance + car_distance) →
    foot_distance / total_distance = 1/2 := by
  sorry

end journey_fraction_by_foot_l2986_298670


namespace today_is_wednesday_l2986_298640

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the number of days from Sunday -/
def daysFromSunday (d : DayOfWeek) : Nat :=
  match d with
  | .Sunday => 0
  | .Monday => 1
  | .Tuesday => 2
  | .Wednesday => 3
  | .Thursday => 4
  | .Friday => 5
  | .Saturday => 6

/-- Adds a number of days to a given day, wrapping around the week -/
def addDays (d : DayOfWeek) (n : Int) : DayOfWeek :=
  match (daysFromSunday d + n % 7 + 7) % 7 with
  | 0 => .Sunday
  | 1 => .Monday
  | 2 => .Tuesday
  | 3 => .Wednesday
  | 4 => .Thursday
  | 5 => .Friday
  | _ => .Saturday

/-- The condition given in the problem -/
def satisfiesCondition (today : DayOfWeek) : Prop :=
  let dayAfterTomorrow := addDays today 2
  let yesterday := addDays today (-1)
  let tomorrow := addDays today 1
  daysFromSunday (addDays dayAfterTomorrow 3) = daysFromSunday (addDays yesterday 2)

/-- The theorem to be proved -/
theorem today_is_wednesday : 
  ∃ (d : DayOfWeek), satisfiesCondition d ∧ d = DayOfWeek.Wednesday :=
sorry

end today_is_wednesday_l2986_298640


namespace student_d_score_l2986_298619

/-- Represents a student's answers and score -/
structure StudentAnswers :=
  (answers : List Bool)
  (score : Nat)

/-- The problem setup -/
def mathTestProblem :=
  let numQuestions : Nat := 8
  let pointsPerQuestion : Nat := 5
  let totalPossibleScore : Nat := 40
  let studentA : StudentAnswers := ⟨[false, true, false, true, false, false, true, false], 30⟩
  let studentB : StudentAnswers := ⟨[false, false, true, true, true, false, false, true], 25⟩
  let studentC : StudentAnswers := ⟨[true, false, false, false, true, true, true, false], 25⟩
  let studentD : StudentAnswers := ⟨[false, true, false, true, true, false, true, true], 0⟩ -- score unknown
  (numQuestions, pointsPerQuestion, totalPossibleScore, studentA, studentB, studentC, studentD)

/-- The theorem to prove -/
theorem student_d_score :
  let (numQuestions, pointsPerQuestion, totalPossibleScore, studentA, studentB, studentC, studentD) := mathTestProblem
  studentD.score = 30 := by
  sorry


end student_d_score_l2986_298619


namespace quotient_with_negative_remainder_l2986_298604

theorem quotient_with_negative_remainder
  (dividend : ℤ)
  (divisor : ℤ)
  (remainder : ℤ)
  (h1 : dividend = 474232)
  (h2 : divisor = 800)
  (h3 : remainder = -968)
  (h4 : dividend = divisor * (dividend / divisor) + remainder) :
  dividend / divisor = 594 := by
  sorry

end quotient_with_negative_remainder_l2986_298604


namespace remaining_money_l2986_298664

def gift_amount : ℕ := 200
def cassette_cost : ℕ := 15
def num_cassettes : ℕ := 3
def headphones_cost : ℕ := 55
def vinyl_cost : ℕ := 35
def poster_cost : ℕ := 45

def total_cost : ℕ := cassette_cost * num_cassettes + headphones_cost + vinyl_cost + poster_cost

theorem remaining_money :
  gift_amount - total_cost = 20 := by
  sorry

end remaining_money_l2986_298664


namespace inequality_solution_l2986_298620

theorem inequality_solution (y : ℝ) : 
  (y^2 + 2*y^3 - 3*y^4) / (y + 2*y^2 - 3*y^3) ≥ -1 ↔ 
  (y ∈ Set.Icc (-1) (-1/3) ∪ Set.Ioo (-1/3) 0 ∪ Set.Ioo 0 1 ∪ Set.Ioi 1) ∧ 
  (y ≠ -1/3) ∧ (y ≠ 0) ∧ (y ≠ 1) :=
by sorry

end inequality_solution_l2986_298620


namespace rome_trip_notes_l2986_298601

/-- Represents the number of notes carried by a person -/
structure Notes where
  euros : ℕ
  dollars : ℕ

/-- The total number of notes carried by both people -/
def total_notes (donald : Notes) (mona : Notes) : ℕ :=
  donald.euros + donald.dollars + mona.euros + mona.dollars

theorem rome_trip_notes :
  ∀ (donald : Notes),
    donald.euros + donald.dollars = 39 →
    donald.euros = donald.dollars →
    ∃ (mona : Notes),
      mona.euros = 3 * donald.euros ∧
      mona.dollars = donald.dollars ∧
      donald.euros + mona.euros = 2 * (donald.dollars + mona.dollars) ∧
      total_notes donald mona = 118 := by
  sorry


end rome_trip_notes_l2986_298601


namespace tv_weekly_cost_l2986_298689

/-- Calculate the weekly cost of running a TV -/
theorem tv_weekly_cost 
  (watt_per_hour : ℕ) 
  (hours_per_day : ℕ) 
  (cents_per_kwh : ℕ) 
  (h1 : watt_per_hour = 125)
  (h2 : hours_per_day = 4)
  (h3 : cents_per_kwh = 14) : 
  (watt_per_hour * hours_per_day * 7 * cents_per_kwh : ℚ) / 1000 = 49 := by
sorry

end tv_weekly_cost_l2986_298689


namespace negation_of_proposition_negation_of_greater_than_sin_l2986_298644

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_greater_than_sin :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) :=
by sorry

end negation_of_proposition_negation_of_greater_than_sin_l2986_298644


namespace tangerines_highest_frequency_l2986_298622

/-- Represents the number of boxes for each fruit type -/
def num_boxes_tangerines : ℕ := 5
def num_boxes_apples : ℕ := 3
def num_boxes_pears : ℕ := 4

/-- Represents the number of fruits per box for each fruit type -/
def fruits_per_box_tangerines : ℕ := 30
def fruits_per_box_apples : ℕ := 20
def fruits_per_box_pears : ℕ := 15

/-- Represents the weight of each fruit in grams -/
def weight_tangerine : ℕ := 200
def weight_apple : ℕ := 450
def weight_pear : ℕ := 800

/-- Calculates the total number of fruits for each type -/
def total_tangerines : ℕ := num_boxes_tangerines * fruits_per_box_tangerines
def total_apples : ℕ := num_boxes_apples * fruits_per_box_apples
def total_pears : ℕ := num_boxes_pears * fruits_per_box_pears

/-- Theorem: Tangerines have the highest frequency -/
theorem tangerines_highest_frequency :
  total_tangerines > total_apples ∧ total_tangerines > total_pears :=
sorry

end tangerines_highest_frequency_l2986_298622


namespace arithmetic_progression_middle_term_l2986_298617

/-- If 2, b, and 10 form an arithmetic progression, then b = 6 -/
theorem arithmetic_progression_middle_term : 
  ∀ b : ℝ, (2 - b = b - 10) → b = 6 := by
  sorry

end arithmetic_progression_middle_term_l2986_298617


namespace problem_solution_l2986_298677

theorem problem_solution (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 47)
  (eq2 : 8 * a + 2 * b = 50) :
  3 * a + 3 * b = 73 / 2 := by
  sorry

end problem_solution_l2986_298677


namespace expression_proof_l2986_298674

theorem expression_proof (x₁ x₂ : ℝ) (E : ℝ → ℝ) :
  (∀ x, (x + 3)^2 / (E x) = 2) →
  x₁ - x₂ = 14 →
  ∃ x, E x = (x + 3)^2 / 2 :=
by sorry

end expression_proof_l2986_298674


namespace sqrt_equation_solution_l2986_298678

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 - 5 * x) = 8 → x = -12 := by
  sorry

end sqrt_equation_solution_l2986_298678


namespace fraction_meaningful_l2986_298628

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (4 + x) / (4 - 2*x)) ↔ x ≠ 2 :=
sorry

end fraction_meaningful_l2986_298628


namespace angle_in_fourth_quadrant_l2986_298637

-- Define the hyperbola equation
def hyperbola_equation (x y α : ℝ) : Prop :=
  x^2 * Real.sin α + y^2 * Real.cos α = 1

-- Define the property of hyperbola with foci on y-axis
def foci_on_y_axis (α : ℝ) : Prop :=
  ∃ (x y : ℝ), hyperbola_equation x y α ∧ Real.cos α > 0 ∧ Real.sin α < 0

-- Theorem statement
theorem angle_in_fourth_quadrant (α : ℝ) (h : foci_on_y_axis α) :
  α > -π/2 ∧ α < 0 :=
sorry

end angle_in_fourth_quadrant_l2986_298637


namespace correct_divisor_l2986_298605

theorem correct_divisor (D : ℕ) (mistaken_divisor correct_quotient : ℕ) 
  (h1 : mistaken_divisor = 12)
  (h2 : D = mistaken_divisor * 35)
  (h3 : correct_quotient = 20)
  (h4 : D % (D / correct_quotient) = 0) :
  D / correct_quotient = 21 := by
sorry

end correct_divisor_l2986_298605


namespace age_problem_l2986_298693

theorem age_problem (p q : ℕ) 
  (h1 : p - 6 = (q - 6) / 2)  -- 6 years ago, p was half of q in age
  (h2 : p * 4 = q * 3)        -- The ratio of their present ages is 3:4
  : p + q = 21 := by           -- The total of their present ages is 21
sorry

end age_problem_l2986_298693


namespace jimmy_drinks_eight_times_per_day_l2986_298600

/-- The number of times Jimmy drinks water per day -/
def times_per_day : ℕ :=
  let ounces_per_drink : ℚ := 8
  let gallons_for_five_days : ℚ := 5/2
  let ounces_per_gallon : ℚ := 1 / 0.0078125
  let days : ℕ := 5
  let total_ounces : ℚ := gallons_for_five_days * ounces_per_gallon
  let ounces_per_day : ℚ := total_ounces / days
  (ounces_per_day / ounces_per_drink).num.toNat

theorem jimmy_drinks_eight_times_per_day : times_per_day = 8 := by
  sorry

end jimmy_drinks_eight_times_per_day_l2986_298600


namespace a_range_l2986_298667

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 > -a * x - 1 ∧ a ≠ 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, 
  x^2 + y^2 = a^2 → (x + 3)^2 + (y - 4)^2 > 4

-- Define the range of a
def range_a (a : ℝ) : Prop := (a > -3 ∧ a ≤ 0) ∨ (a ≥ 3 ∧ a < 4)

-- State the theorem
theorem a_range : 
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) → 
  (∀ a : ℝ, range_a a ↔ (p a ∨ q a) ∧ ¬(p a ∧ q a)) :=
sorry

end a_range_l2986_298667


namespace combined_selling_price_l2986_298610

/-- Calculate the combined selling price of three items with given costs, exchange rate, profits, discount, and tax. -/
theorem combined_selling_price (exchange_rate : ℝ) (cost_a cost_b cost_c : ℝ)
  (profit_a profit_b profit_c : ℝ) (discount_b tax : ℝ) :
  exchange_rate = 70 ∧
  cost_a = 10 ∧
  cost_b = 15 ∧
  cost_c = 20 ∧
  profit_a = 0.25 ∧
  profit_b = 0.30 ∧
  profit_c = 0.20 ∧
  discount_b = 0.10 ∧
  tax = 0.08 →
  let cost_rs_a := cost_a * exchange_rate
  let cost_rs_b := cost_b * exchange_rate * (1 - discount_b)
  let cost_rs_c := cost_c * exchange_rate
  let selling_price_a := cost_rs_a * (1 + profit_a) * (1 + tax)
  let selling_price_b := cost_rs_b * (1 + profit_b) * (1 + tax)
  let selling_price_c := cost_rs_c * (1 + profit_c) * (1 + tax)
  selling_price_a + selling_price_b + selling_price_c = 4086.18 := by
sorry


end combined_selling_price_l2986_298610


namespace max_min_values_l2986_298652

theorem max_min_values (x y : ℝ) (h : x^2 + 4*y^2 = 4) :
  (∃ a b : ℝ, a^2 + 4*b^2 = 4 ∧ x^2 + 2*x*y + 4*y^2 ≤ a^2 + 2*a*b + 4*b^2) ∧
  (∃ c d : ℝ, c^2 + 4*d^2 = 4 ∧ x^2 + 2*x*y + 4*y^2 ≥ c^2 + 2*c*d + 4*d^2) ∧
  (∃ e f : ℝ, e^2 + 4*f^2 = 4 ∧ e^2 + 2*e*f + 4*f^2 = 6) ∧
  (∃ g h : ℝ, g^2 + 4*h^2 = 4 ∧ g^2 + 2*g*h + 4*h^2 = 2) := by
  sorry

end max_min_values_l2986_298652


namespace infinite_series_sum_l2986_298682

/-- The sum of the infinite series Σ(n=1 to ∞) [2^(2n) / (1 + 2^n + 2^(2n) + 2^(3n) + 2^(3n+1))] is equal to 1/25. -/
theorem infinite_series_sum : 
  ∑' n : ℕ, (2^(2*n) : ℝ) / (1 + 2^n + 2^(2*n) + 2^(3*n) + 2^(3*n+1)) = 1/25 := by
  sorry

end infinite_series_sum_l2986_298682


namespace composite_with_large_smallest_prime_divisor_l2986_298638

theorem composite_with_large_smallest_prime_divisor 
  (N : ℕ) 
  (h_composite : ¬ Prime N) 
  (h_smallest_divisor : ∀ p : ℕ, Prime p → p ∣ N → p > N^(1/3)) : 
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ N = p * q :=
sorry

end composite_with_large_smallest_prime_divisor_l2986_298638


namespace double_factorial_sum_denominator_l2986_298612

/-- Double factorial for odd numbers -/
def odd_double_factorial (n : ℕ) : ℕ := sorry

/-- Double factorial for even numbers -/
def even_double_factorial (n : ℕ) : ℕ := sorry

/-- The sum of the ratios of double factorials -/
def double_factorial_sum : ℚ :=
  (Finset.range 2009).sum (fun i => (odd_double_factorial (2*i+1)) / (even_double_factorial (2*i+2)))

/-- The denominator of the sum when expressed in lowest terms -/
def denominator_of_sum : ℕ := sorry

/-- The power of 2 in the denominator -/
def a : ℕ := sorry

/-- The odd factor in the denominator -/
def b : ℕ := sorry

theorem double_factorial_sum_denominator :
  denominator_of_sum = 2^a * b ∧ Odd b ∧ a*b/10 = 401 := by sorry

end double_factorial_sum_denominator_l2986_298612


namespace limit_rational_function_l2986_298699

/-- The limit of (x^2 + 2x - 3) / (x^3 + 4x^2 + 3x) as x approaches -3 is -2/3 -/
theorem limit_rational_function : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x + 3| ∧ |x + 3| < δ → 
    |(x^2 + 2*x - 3) / (x^3 + 4*x^2 + 3*x) + 2/3| < ε :=
by sorry

end limit_rational_function_l2986_298699


namespace sum_of_combinations_l2986_298650

theorem sum_of_combinations : Nat.choose 10 3 + Nat.choose 10 4 = 330 := by
  sorry

end sum_of_combinations_l2986_298650


namespace sum_of_squares_of_coefficients_l2986_298653

def p (x : ℝ) : ℝ := 3 * (x^2 - 3*x + 3) - 5 * (x^3 - 2*x^2 + 4*x - 1)

theorem sum_of_squares_of_coefficients :
  ∃ (a b c d : ℝ),
    (∀ x, p x = a * x^3 + b * x^2 + c * x + d) ∧
    a^2 + b^2 + c^2 + d^2 = 1231 := by sorry

end sum_of_squares_of_coefficients_l2986_298653


namespace school_population_l2986_298656

theorem school_population (girls : ℕ) (boys : ℕ) (difference : ℕ) : 
  girls = 692 → difference = 458 → girls = boys + difference → girls + boys = 926 := by
  sorry

end school_population_l2986_298656


namespace michaels_initial_money_proof_l2986_298687

/-- Michael's initial amount of money -/
def michaels_initial_money : ℕ := 152

/-- Amount Michael's brother had initially -/
def brothers_initial_money : ℕ := 17

/-- Amount spent on candy -/
def candy_cost : ℕ := 3

/-- Amount Michael's brother has left after buying candy -/
def brothers_remaining_money : ℕ := 35

theorem michaels_initial_money_proof :
  michaels_initial_money = 
    2 * (brothers_remaining_money + candy_cost + brothers_initial_money - brothers_initial_money) :=
by sorry

end michaels_initial_money_proof_l2986_298687


namespace subset_condition_disjoint_condition_l2986_298673

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3*a) < 0}

-- Theorem for A ⊆ B
theorem subset_condition (a : ℝ) : 
  A ⊆ B a ↔ (4/3 ≤ a ∧ a ≤ 2 ∧ a ≠ 0) :=
sorry

-- Theorem for A ∩ B = ∅
theorem disjoint_condition (a : ℝ) :
  A ∩ B a = ∅ ↔ (a ≤ 2/3 ∨ a ≥ 4) :=
sorry

end subset_condition_disjoint_condition_l2986_298673


namespace max_revenue_l2986_298626

/-- Represents the production quantities of products A and B -/
structure Production where
  a : ℝ
  b : ℝ

/-- Calculates the revenue for a given production -/
def revenue (p : Production) : ℝ :=
  0.3 * p.a + 0.2 * p.b

/-- Checks if a production is feasible given the machine constraints -/
def is_feasible (p : Production) : Prop :=
  p.a ≥ 0 ∧ p.b ≥ 0 ∧
  1 * p.a + 2 * p.b ≤ 400 ∧
  2 * p.a + 1 * p.b ≤ 500

/-- Theorem stating the maximum monthly sales revenue -/
theorem max_revenue :
  ∃ (p : Production), is_feasible p ∧
    ∀ (q : Production), is_feasible q → revenue q ≤ revenue p ∧
    revenue p = 90 :=
sorry

end max_revenue_l2986_298626


namespace line_equation_proof_l2986_298634

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallelLines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_equation_proof (given_line : Line) (p : Point) :
  given_line.a = 1 ∧ given_line.b = -2 ∧ given_line.c = 3 →
  p.x = -1 ∧ p.y = 3 →
  ∃ (result_line : Line),
    result_line.a = 1 ∧ result_line.b = -2 ∧ result_line.c = 7 ∧
    pointOnLine p result_line ∧
    parallelLines given_line result_line := by
  sorry

end line_equation_proof_l2986_298634


namespace fifteen_percent_of_x_is_ninety_l2986_298681

theorem fifteen_percent_of_x_is_ninety (x : ℝ) : (15 / 100) * x = 90 → x = 600 := by
  sorry

end fifteen_percent_of_x_is_ninety_l2986_298681


namespace right_triangle_leg_square_l2986_298676

/-- In a right triangle, if the hypotenuse c is 2 more than one leg a,
    then the square of the other leg b is equal to 4a + 4 -/
theorem right_triangle_leg_square (a c : ℝ) (h1 : c = a + 2) :
  ∃ b : ℝ, a^2 + b^2 = c^2 ∧ b^2 = 4*a + 4 := by
  sorry

end right_triangle_leg_square_l2986_298676


namespace olympic_tournament_winners_l2986_298625

/-- Represents an Olympic system tournament -/
structure OlympicTournament where
  rounds : ℕ
  initialParticipants : ℕ
  winnersEachRound : List ℕ

/-- Checks if the tournament is valid -/
def isValidTournament (t : OlympicTournament) : Prop :=
  t.rounds > 0 ∧
  t.initialParticipants = 2^t.rounds ∧
  t.winnersEachRound.length = t.rounds ∧
  ∀ i, i ∈ t.winnersEachRound → i = t.initialParticipants / (2^(t.winnersEachRound.indexOf i + 1))

/-- Calculates the number of participants who won more games than they lost -/
def participantsWithMoreWins (t : OlympicTournament) : ℕ :=
  t.initialParticipants / 4

theorem olympic_tournament_winners (t : OlympicTournament) 
  (h1 : isValidTournament t) 
  (h2 : t.rounds = 6) : 
  participantsWithMoreWins t = 16 := by
  sorry

#check olympic_tournament_winners

end olympic_tournament_winners_l2986_298625


namespace repeating_decimal_sum_theorem_l2986_298657

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Converts a three-digit number represented by individual digits to an integer -/
def threeDigitToInt (a b c : Digit) : ℕ := 100 * a.val + 10 * b.val + c.val

/-- Converts a repeating decimal of the form 0.abab... to a rational number -/
def abRepeatingToRational (a b : Digit) : ℚ := (10 * a.val + b.val : ℚ) / 99

/-- Converts a repeating decimal of the form 0.abcabc... to a rational number -/
def abcRepeatingToRational (a b c : Digit) : ℚ := (100 * a.val + 10 * b.val + c.val : ℚ) / 999

theorem repeating_decimal_sum_theorem (a b c : Digit) :
  abRepeatingToRational a b + abcRepeatingToRational a b c = 17 / 37 →
  threeDigitToInt a b c = 270 := by
  sorry

end repeating_decimal_sum_theorem_l2986_298657


namespace ohms_law_application_l2986_298645

/-- Given a constant voltage U, current I inversely proportional to resistance R,
    prove that for I1 = 4A, R1 = 10Ω, and I2 = 5A, the value of R2 is 8Ω. -/
theorem ohms_law_application (U : ℝ) (I1 I2 R1 R2 : ℝ) : 
  U > 0 →  -- Voltage is positive
  I1 > 0 →  -- Current is positive
  I2 > 0 →  -- Current is positive
  R1 > 0 →  -- Resistance is positive
  R2 > 0 →  -- Resistance is positive
  (∀ I R, U = I * R) →  -- Ohm's law: U = IR
  I1 = 4 →
  R1 = 10 →
  I2 = 5 →
  R2 = 8 := by
sorry

end ohms_law_application_l2986_298645


namespace pie_crust_flour_calculation_l2986_298661

theorem pie_crust_flour_calculation (initial_crusts : ℕ) (initial_flour_per_crust : ℚ) 
  (new_crusts : ℕ) (h1 : initial_crusts = 40) (h2 : initial_flour_per_crust = 1/8) 
  (h3 : new_crusts = 25) :
  let total_flour := initial_crusts * initial_flour_per_crust
  let new_flour_per_crust := total_flour / new_crusts
  new_flour_per_crust = 1/5 := by
sorry

end pie_crust_flour_calculation_l2986_298661


namespace stamp_cost_correct_l2986_298658

/-- The cost of one stamp, given that three stamps cost $1.02 and the cost is constant -/
def stamp_cost : ℚ := 0.34

/-- The cost of three stamps -/
def three_stamps_cost : ℚ := 1.02

/-- Theorem stating that the cost of one stamp is correct -/
theorem stamp_cost_correct : 3 * stamp_cost = three_stamps_cost := by sorry

end stamp_cost_correct_l2986_298658


namespace hyperbola_eccentricity_l2986_298642

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 and asymptote y = 2x,
    its eccentricity e is either √5 or √5/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → y = 2*x) →
  ∃ e : ℝ, (e = Real.sqrt 5 ∨ e = Real.sqrt 5 / 2) ∧
    ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 →
      e = Real.sqrt ((a^2 + b^2) / a^2) :=
by sorry

end hyperbola_eccentricity_l2986_298642


namespace stream_speed_in_rowing_problem_l2986_298613

/-- Proves that the speed of the stream is 20 kmph given the conditions of the rowing problem. -/
theorem stream_speed_in_rowing_problem (boat_speed : ℝ) (stream_speed : ℝ) :
  boat_speed = 60 →
  (∀ d : ℝ, d > 0 → d / (boat_speed - stream_speed) = 2 * (d / (boat_speed + stream_speed))) →
  stream_speed = 20 :=
by sorry

end stream_speed_in_rowing_problem_l2986_298613


namespace boat_length_in_steps_l2986_298675

/-- Represents the scenario of Josie jogging alongside a moving boat --/
structure JosieAndBoat where
  josie_speed : ℝ
  boat_speed : ℝ
  boat_length : ℝ
  step_length : ℝ
  steps_forward : ℕ
  steps_backward : ℕ

/-- The conditions of the problem --/
def problem_conditions (scenario : JosieAndBoat) : Prop :=
  scenario.josie_speed > scenario.boat_speed ∧
  scenario.steps_forward = 130 ∧
  scenario.steps_backward = 70 ∧
  scenario.boat_length = scenario.step_length * 91

/-- The theorem to be proved --/
theorem boat_length_in_steps (scenario : JosieAndBoat) 
  (h : problem_conditions scenario) : 
  scenario.boat_length = scenario.step_length * 91 :=
sorry

end boat_length_in_steps_l2986_298675


namespace paint_calculation_l2986_298602

theorem paint_calculation (initial_paint : ℚ) : 
  (1 / 4 * initial_paint + 1 / 6 * (3 / 4 * initial_paint) = 135) → 
  ⌈initial_paint⌉ = 463 := by
  sorry

end paint_calculation_l2986_298602


namespace M_inter_complement_N_eq_l2986_298607

/-- The universal set U (real numbers) -/
def U : Set ℝ := Set.univ

/-- Set M defined as {x | -2 ≤ x ≤ 2} -/
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

/-- Set N defined as the domain of y = ln(x-1), which is {x | x > 1} -/
def N : Set ℝ := {x | x > 1}

/-- Theorem stating that the intersection of M and the complement of N in U
    is equal to the set {x | -2 ≤ x ≤ 1} -/
theorem M_inter_complement_N_eq :
  M ∩ (U \ N) = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end M_inter_complement_N_eq_l2986_298607


namespace product_of_roots_l2986_298609

theorem product_of_roots (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂)
  (h₁ : x₁^2 - 2006*x₁ = 1)
  (h₂ : x₂^2 - 2006*x₂ = 1) : 
  x₁ * x₂ = -1 := by sorry

end product_of_roots_l2986_298609


namespace binary_101_equals_5_l2986_298646

-- Define a binary number as a list of bits (0 or 1)
def BinaryNumber := List Nat

-- Define a function to convert a binary number to decimal
def binary_to_decimal (b : BinaryNumber) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + bit * 2^i) 0

-- State the theorem
theorem binary_101_equals_5 :
  binary_to_decimal [1, 0, 1] = 5 := by
  sorry

end binary_101_equals_5_l2986_298646


namespace regular_polygon_interior_angle_sum_l2986_298641

theorem regular_polygon_interior_angle_sum :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 2 →
    exterior_angle = 24 →
    n * exterior_angle = 360 →
    (n - 2) * 180 = 2340 :=
by sorry

end regular_polygon_interior_angle_sum_l2986_298641


namespace min_distance_circle_to_line_polar_to_cartesian_line_l2986_298666

/-- Given a line and a circle, find the minimum distance from a point on the circle to the line -/
theorem min_distance_circle_to_line :
  let line := {(x, y) : ℝ × ℝ | x + y = 1}
  let circle := {(x, y) : ℝ × ℝ | ∃ θ : ℝ, x = 2 * Real.cos θ ∧ y = -2 + 2 * Real.sin θ}
  ∃ d : ℝ, d = (3 * Real.sqrt 2) / 2 - 2 ∧
    ∀ p ∈ circle, ∀ q ∈ line, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by sorry

/-- The polar equation of the line can be converted to Cartesian form -/
theorem polar_to_cartesian_line :
  ∀ ρ θ : ℝ, ρ * Real.sin (θ + π/4) = Real.sqrt 2 / 2 →
  ∃ x y : ℝ, x + y = 1 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ :=
by sorry

end min_distance_circle_to_line_polar_to_cartesian_line_l2986_298666


namespace quadratic_equation_roots_l2986_298623

theorem quadratic_equation_roots (p q : ℝ) (a b : ℝ) : 
  (a^2 + p*a + q = 0) → 
  (b^2 + p*b + q = 0) → 
  ∃ y₁ y₂ : ℝ, 
    (y₁ = (a+b)^2 ∧ y₂ = (a-b)^2) ∧ 
    (y₁^2 - 2*(p^2 - 2*q)*y₁ + (p^4 - 4*q*p^2) = 0) ∧
    (y₂^2 - 2*(p^2 - 2*q)*y₂ + (p^4 - 4*q*p^2) = 0) :=
by sorry

end quadratic_equation_roots_l2986_298623


namespace cos_difference_value_l2986_298629

theorem cos_difference_value (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by sorry

end cos_difference_value_l2986_298629


namespace smallest_visible_sum_l2986_298671

/-- Represents a die in the cube --/
structure Die where
  sides : Fin 6 → ℕ
  sum_opposite : ∀ i : Fin 3, sides i + sides (i + 3) = 7

/-- Represents the 4x4x4 cube made of dice --/
def Cube := Fin 4 → Fin 4 → Fin 4 → Die

/-- Calculates the sum of visible faces on the large cube --/
def visible_sum (c : Cube) : ℕ := sorry

/-- Theorem stating the smallest possible sum of visible faces --/
theorem smallest_visible_sum (c : Cube) : 
  visible_sum c ≥ 136 ∧ ∃ c', visible_sum c' = 136 := by sorry

end smallest_visible_sum_l2986_298671


namespace max_candy_leftover_l2986_298660

theorem max_candy_leftover (x : ℕ+) : 
  ∃ (q r : ℕ), x = 12 * q + r ∧ r < 12 ∧ r ≤ 11 :=
by sorry

end max_candy_leftover_l2986_298660


namespace additional_candles_l2986_298603

/-- 
Given:
- initial_candles: The initial number of candles on Molly's birthday cake
- current_age: Molly's current age
Prove that the number of additional candles is equal to current_age - initial_candles
-/
theorem additional_candles (initial_candles current_age : ℕ) :
  initial_candles = 14 →
  current_age = 20 →
  current_age - initial_candles = 6 := by
  sorry

end additional_candles_l2986_298603


namespace first_part_value_l2986_298633

theorem first_part_value (x y : ℝ) 
  (sum_constraint : x + y = 36)
  (weighted_sum_constraint : 8 * x + 3 * y = 203) :
  x = 19 := by
sorry

end first_part_value_l2986_298633


namespace taxi_charge_theorem_l2986_298639

-- Define the parameters of the taxi service
def initial_fee : ℚ := 235 / 100
def charge_per_increment : ℚ := 35 / 100
def miles_per_increment : ℚ := 2 / 5
def trip_distance : ℚ := 36 / 10

-- Define the total charge function
def total_charge (initial_fee charge_per_increment miles_per_increment trip_distance : ℚ) : ℚ :=
  initial_fee + (trip_distance / miles_per_increment) * charge_per_increment

-- State the theorem
theorem taxi_charge_theorem :
  total_charge initial_fee charge_per_increment miles_per_increment trip_distance = 865 / 100 := by
  sorry

end taxi_charge_theorem_l2986_298639


namespace age_sum_l2986_298688

theorem age_sum (patrick michael monica : ℕ) 
  (h1 : 3 * michael = 5 * patrick)
  (h2 : 3 * monica = 5 * michael)
  (h3 : monica - patrick = 32) : 
  patrick + michael + monica = 98 := by
sorry

end age_sum_l2986_298688


namespace last_number_is_one_seventh_l2986_298647

/-- A sequence of 100 non-zero real numbers where each number (except the first and last) 
    is the product of its neighbors, and the first number is 7 -/
def SpecialSequence (a : Fin 100 → ℝ) : Prop :=
  a 0 = 7 ∧ 
  (∀ i : Fin 98, a (i + 1) = a i * a (i + 2)) ∧
  (∀ i : Fin 100, a i ≠ 0)

/-- The last number in the sequence is 1/7 -/
theorem last_number_is_one_seventh (a : Fin 100 → ℝ) (h : SpecialSequence a) : 
  a 99 = 1 / 7 := by
  sorry

end last_number_is_one_seventh_l2986_298647


namespace value_of_B_l2986_298648

theorem value_of_B : ∃ B : ℚ, (3/4 : ℚ) * (1/8 : ℚ) * B = (1/4 : ℚ) * (1/6 : ℚ) * 48 ∧ B = 64/3 :=
by sorry

end value_of_B_l2986_298648


namespace max_additional_plates_achievable_additional_plates_l2986_298668

/-- Represents the sets of symbols for car plates in Rivertown -/
structure CarPlateSymbols where
  firstLetters : Finset Char
  secondLetters : Finset Char
  digits : Finset Char

/-- Calculates the total number of possible car plates -/
def totalPlates (symbols : CarPlateSymbols) : ℕ :=
  symbols.firstLetters.card * symbols.secondLetters.card * symbols.digits.card

/-- The initial configuration of car plate symbols -/
def initialSymbols : CarPlateSymbols :=
  { firstLetters := {'A', 'B', 'G', 'H', 'T'},
    secondLetters := {'E', 'I', 'O', 'U'},
    digits := {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'} }

/-- Represents the addition of new symbols -/
structure NewSymbols where
  newLetters : ℕ
  newDigits : ℕ

/-- The new symbols to be added -/
def addedSymbols : NewSymbols :=
  { newLetters := 2,
    newDigits := 1 }

/-- Theorem: The maximum number of additional car plates after adding new symbols is 130 -/
theorem max_additional_plates :
  ∀ (newDistribution : CarPlateSymbols),
    (newDistribution.firstLetters.card + newDistribution.secondLetters.card = 
      initialSymbols.firstLetters.card + initialSymbols.secondLetters.card + addedSymbols.newLetters) →
    (newDistribution.digits.card = initialSymbols.digits.card + addedSymbols.newDigits) →
    totalPlates newDistribution - totalPlates initialSymbols ≤ 130 :=
by sorry

/-- Theorem: There exists a distribution that achieves 130 additional plates -/
theorem achievable_additional_plates :
  ∃ (newDistribution : CarPlateSymbols),
    (newDistribution.firstLetters.card + newDistribution.secondLetters.card = 
      initialSymbols.firstLetters.card + initialSymbols.secondLetters.card + addedSymbols.newLetters) ∧
    (newDistribution.digits.card = initialSymbols.digits.card + addedSymbols.newDigits) ∧
    totalPlates newDistribution - totalPlates initialSymbols = 130 :=
by sorry

end max_additional_plates_achievable_additional_plates_l2986_298668


namespace observation_count_l2986_298643

theorem observation_count (original_mean : ℝ) (incorrect_value : ℝ) (correct_value : ℝ) (new_mean : ℝ) : 
  original_mean = 36 →
  incorrect_value = 23 →
  correct_value = 30 →
  new_mean = 36.5 →
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) * new_mean = n * original_mean + (correct_value - incorrect_value) ∧ n = 14 :=
by
  sorry

end observation_count_l2986_298643


namespace maci_school_supplies_cost_l2986_298649

/-- The cost of Maci's school supplies -/
def school_supplies_cost (blue_pen_price : ℚ) : ℚ :=
  let red_pen_price := 2 * blue_pen_price
  let pencil_price := red_pen_price / 2
  let notebook_price := 10 * blue_pen_price
  10 * blue_pen_price +  -- 10 blue pens
  15 * red_pen_price +   -- 15 red pens
  5 * pencil_price +     -- 5 pencils
  3 * notebook_price     -- 3 notebooks

/-- Theorem stating that the cost of Maci's school supplies is $7.50 -/
theorem maci_school_supplies_cost :
  school_supplies_cost (10 / 100) = 75 / 10 := by
  sorry

end maci_school_supplies_cost_l2986_298649


namespace tournament_handshakes_l2986_298621

/-- Represents a women's doubles tennis tournament --/
structure Tournament where
  numTeams : Nat
  playersPerTeam : Nat
  handshakesPerPlayer : Nat

/-- Calculates the total number of handshakes in the tournament --/
def totalHandshakes (t : Tournament) : Nat :=
  (t.numTeams * t.playersPerTeam * t.handshakesPerPlayer) / 2

/-- Theorem stating that the specific tournament configuration results in 24 handshakes --/
theorem tournament_handshakes :
  ∃ (t : Tournament),
    t.numTeams = 4 ∧
    t.playersPerTeam = 2 ∧
    t.handshakesPerPlayer = 6 ∧
    totalHandshakes t = 24 := by
  sorry


end tournament_handshakes_l2986_298621


namespace fraction_reduction_l2986_298608

theorem fraction_reduction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^3 - b^3) / (a*b) - (a*b^2 - b^3) / (a*b - a^3) = (a^2 + a*b + b^2) / b :=
by sorry

end fraction_reduction_l2986_298608


namespace total_suit_cost_l2986_298683

/-- The cost of a suit given the following conditions:
  1. A jacket costs as much as trousers and a vest.
  2. A jacket and two pairs of trousers cost 175 dollars.
  3. Trousers and two vests cost 100 dollars. -/
def suit_cost (jacket trousers vest : ℝ) : Prop :=
  jacket = trousers + vest ∧
  jacket + 2 * trousers = 175 ∧
  trousers + 2 * vest = 100

/-- Theorem stating that the total cost of the suit is 150 dollars. -/
theorem total_suit_cost :
  ∀ (jacket trousers vest : ℝ),
    suit_cost jacket trousers vest →
    jacket + trousers + vest = 150 :=
by
  sorry

#check total_suit_cost

end total_suit_cost_l2986_298683
