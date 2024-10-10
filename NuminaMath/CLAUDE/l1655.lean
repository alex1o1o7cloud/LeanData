import Mathlib

namespace inequality_range_l1655_165542

theorem inequality_range (t : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → 
    (1/8 * (2*t - t^2) ≤ x^2 - 3*x + 2 ∧ x^2 - 3*x + 2 ≤ 3 - t^2)) ↔ 
  (t ∈ Set.Icc (-1) (1 - Real.sqrt 3)) :=
sorry

end inequality_range_l1655_165542


namespace max_sum_of_exponents_l1655_165589

theorem max_sum_of_exponents (x y : ℝ) (h : (2 : ℝ)^x + (2 : ℝ)^y = 1) :
  x + y ≤ -2 ∧ ∃ (a b : ℝ), (2 : ℝ)^a + (2 : ℝ)^b = 1 ∧ a + b = -2 :=
by sorry

end max_sum_of_exponents_l1655_165589


namespace test_score_problem_l1655_165579

/-- Proves that given a test with 25 problems, a scoring system of 4 points for each correct answer
    and -1 point for each wrong answer, and a total score of 85, the number of wrong answers is 3. -/
theorem test_score_problem (total_problems : Nat) (right_points : Int) (wrong_points : Int) (total_score : Int)
    (h1 : total_problems = 25)
    (h2 : right_points = 4)
    (h3 : wrong_points = -1)
    (h4 : total_score = 85) :
    ∃ (right : Nat) (wrong : Nat),
      right + wrong = total_problems ∧
      right_points * right + wrong_points * wrong = total_score ∧
      wrong = 3 :=
by sorry

end test_score_problem_l1655_165579


namespace ellipse_intersection_parallel_line_l1655_165511

-- Define the ellipse C
def ellipse_C (b : ℝ) (x y : ℝ) : Prop :=
  b > 0 ∧ x^2 / (5 * b^2) + y^2 / b^2 = 1

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line on the plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the focus of the ellipse
def focus : Point := ⟨2, 0⟩

-- Define point E
def point_E : Point := ⟨3, 0⟩

-- Define the line x = 5
def line_x_5 : Line := ⟨1, 0, -5⟩

-- Define the property of line l passing through (1,0) and not coinciding with x-axis
def line_l_property (l : Line) : Prop :=
  l.a * 1 + l.b * 0 + l.c = 0 ∧ l.b ≠ 0

-- Define the intersection of a line and the ellipse
def intersect_line_ellipse (l : Line) (b : ℝ) : Prop :=
  ∃ (M N : Point), 
    line_l_property l ∧
    ellipse_C b M.x M.y ∧ 
    ellipse_C b N.x N.y ∧
    l.a * M.x + l.b * M.y + l.c = 0 ∧
    l.a * N.x + l.b * N.y + l.c = 0

-- Define the intersection of two lines
def intersect_lines (l1 l2 : Line) (P : Point) : Prop :=
  l1.a * P.x + l1.b * P.y + l1.c = 0 ∧
  l2.a * P.x + l2.b * P.y + l2.c = 0

-- Define parallel lines
def parallel_lines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- The main theorem
theorem ellipse_intersection_parallel_line (b : ℝ) (l : Line) (M N F : Point) :
  ellipse_C b focus.x focus.y →
  intersect_line_ellipse l b →
  intersect_lines (Line.mk (M.y - point_E.y) (point_E.x - M.x) (M.x * point_E.y - M.y * point_E.x)) line_x_5 F →
  parallel_lines (Line.mk (F.y - N.y) (N.x - F.x) (F.x * N.y - F.y * N.x)) (Line.mk 0 1 0) :=
sorry

end ellipse_intersection_parallel_line_l1655_165511


namespace remi_seedlings_proof_l1655_165504

/-- The number of seedlings Remi planted on the first day -/
def first_day_seedlings : ℕ := 400

/-- The number of seedlings Remi planted on the second day -/
def second_day_seedlings : ℕ := 2 * first_day_seedlings

/-- The total number of seedlings transferred over the two days -/
def total_seedlings : ℕ := 1200

theorem remi_seedlings_proof :
  first_day_seedlings + second_day_seedlings = total_seedlings ∧
  first_day_seedlings = 400 := by
  sorry

end remi_seedlings_proof_l1655_165504


namespace intersection_sum_zero_l1655_165584

theorem intersection_sum_zero (x₁ x₂ : ℝ) :
  (x₁^2 + 9^2 = 169) →
  (x₂^2 + 9^2 = 169) →
  x₁ ≠ x₂ →
  x₁ + x₂ = 0 := by
sorry

end intersection_sum_zero_l1655_165584


namespace viewing_time_calculation_l1655_165544

/-- Represents the duration of a TV show episode in minutes -/
def episode_duration : ℕ := 30

/-- Represents the number of weekdays in a week -/
def weekdays : ℕ := 5

/-- Represents the number of episodes watched -/
def episodes_watched : ℕ := 4

/-- Calculates the total viewing time in hours -/
def total_viewing_time : ℚ :=
  (episode_duration * episodes_watched) / 60

theorem viewing_time_calculation :
  total_viewing_time = 2 :=
sorry

end viewing_time_calculation_l1655_165544


namespace investment_growth_l1655_165593

/-- Calculates the total amount after compound interest is applied --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem investment_growth :
  let initial_investment : ℝ := 300
  let monthly_rate : ℝ := 0.1
  let months : ℕ := 2
  compound_interest initial_investment monthly_rate months = 363 := by
sorry

end investment_growth_l1655_165593


namespace smallest_pencil_collection_l1655_165550

theorem smallest_pencil_collection (P : ℕ) : 
  P > 2 ∧ 
  P % 5 = 2 ∧ 
  P % 9 = 2 ∧ 
  P % 11 = 2 ∧ 
  (∀ Q : ℕ, Q > 2 ∧ Q % 5 = 2 ∧ Q % 9 = 2 ∧ Q % 11 = 2 → P ≤ Q) →
  P = 497 := by
sorry

end smallest_pencil_collection_l1655_165550


namespace student_number_problem_l1655_165546

theorem student_number_problem (x : ℝ) : 2 * x - 152 = 102 → x = 127 := by
  sorry

end student_number_problem_l1655_165546


namespace jens_height_l1655_165567

theorem jens_height (original_height : ℝ) (bakis_growth_rate : ℝ) (jens_growth_ratio : ℝ) (bakis_final_height : ℝ) :
  original_height > 0 ∧ 
  bakis_growth_rate = 0.25 ∧
  jens_growth_ratio = 2/3 ∧
  bakis_final_height = 75 ∧
  bakis_final_height = original_height * (1 + bakis_growth_rate) →
  original_height + jens_growth_ratio * (bakis_final_height - original_height) = 70 :=
by
  sorry

#check jens_height

end jens_height_l1655_165567


namespace cases_in_1975_l1655_165539

/-- Calculates the number of disease cases in a given year, assuming a linear decrease --/
def casesInYear (initialYear initialCases finalYear finalCases targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let yearlyDecrease := totalDecrease / totalYears
  let targetYearDiff := targetYear - initialYear
  initialCases - (yearlyDecrease * targetYearDiff)

/-- Theorem stating that given the conditions, the number of cases in 1975 would be 300,150 --/
theorem cases_in_1975 :
  casesInYear 1950 600000 2000 300 1975 = 300150 := by
  sorry

end cases_in_1975_l1655_165539


namespace solve_equation_l1655_165518

theorem solve_equation (x : ℚ) : (3 * x + 5) / 5 = 17 → x = 80 / 3 := by
  sorry

end solve_equation_l1655_165518


namespace line_intersection_b_range_l1655_165519

theorem line_intersection_b_range (b : ℝ) (h1 : b ≠ 0) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ 2 * x + b = 3) →
  -3 ≤ b ∧ b ≤ 3 ∧ b ≠ 0 := by
sorry

end line_intersection_b_range_l1655_165519


namespace opponent_total_score_l1655_165522

def team_scores : List ℕ := [1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def one_run_losses : ℕ := 3
def two_run_losses : ℕ := 2

def remaining_games (total_games : ℕ) : ℕ :=
  total_games - one_run_losses - two_run_losses

theorem opponent_total_score :
  ∃ (one_loss_scores two_loss_scores three_times_scores : List ℕ),
    (one_loss_scores.length = one_run_losses) ∧
    (two_loss_scores.length = two_run_losses) ∧
    (three_times_scores.length = remaining_games team_scores.length) ∧
    (∀ (s : ℕ), s ∈ one_loss_scores → s ∈ team_scores) ∧
    (∀ (s : ℕ), s ∈ two_loss_scores → s ∈ team_scores) ∧
    (∀ (s : ℕ), s ∈ three_times_scores → s ∈ team_scores) ∧
    (∀ (s : ℕ), s ∈ one_loss_scores → ∃ (o : ℕ), o = s + 1) ∧
    (∀ (s : ℕ), s ∈ two_loss_scores → ∃ (o : ℕ), o = s + 2) ∧
    (∀ (s : ℕ), s ∈ three_times_scores → ∃ (o : ℕ), o = s / 3) ∧
    (one_loss_scores.sum + two_loss_scores.sum + three_times_scores.sum = 42) :=
by
  sorry

end opponent_total_score_l1655_165522


namespace negation_of_universal_proposition_l1655_165598

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x - x - 1 ≥ 0) ↔ (∃ x : ℝ, Real.exp x - x - 1 < 0) := by
  sorry

end negation_of_universal_proposition_l1655_165598


namespace cubic_integer_values_l1655_165523

/-- A cubic polynomial function -/
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 - c * x - d

/-- Theorem: If a cubic polynomial takes integer values at -1, 0, 1, and 2, 
    then it takes integer values for all integer inputs -/
theorem cubic_integer_values 
  (a b c d : ℝ) 
  (h₁ : ∃ n₁ : ℤ, f a b c d (-1) = n₁) 
  (h₂ : ∃ n₂ : ℤ, f a b c d 0 = n₂)
  (h₃ : ∃ n₃ : ℤ, f a b c d 1 = n₃)
  (h₄ : ∃ n₄ : ℤ, f a b c d 2 = n₄) :
  ∀ x : ℤ, ∃ n : ℤ, f a b c d x = n :=
sorry

end cubic_integer_values_l1655_165523


namespace even_mono_decreasing_order_l1655_165549

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def isMonoDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

theorem even_mono_decreasing_order (f : ℝ → ℝ) 
  (h_even : isEven f) 
  (h_mono : isMonoDecreasing f 0 3) : 
  f (-1) > f 2 ∧ f 2 > f 3 := by
  sorry

end even_mono_decreasing_order_l1655_165549


namespace planes_perpendicular_to_line_are_parallel_l1655_165592

-- Define the basic geometric objects
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the geometric relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular_to_line_are_parallel
  (α β : Plane) (m : Line) (h_diff : α ≠ β) :
  perpendicular m α → perpendicular m β → parallel α β := by sorry

end planes_perpendicular_to_line_are_parallel_l1655_165592


namespace multiple_birth_statistics_l1655_165537

theorem multiple_birth_statistics (total_babies : ℕ) 
  (twins triplets quadruplets : ℕ) : 
  total_babies = 1000 →
  triplets = 4 * quadruplets →
  twins = 3 * triplets →
  2 * twins + 3 * triplets + 4 * quadruplets = total_babies →
  4 * quadruplets = 100 := by
  sorry

end multiple_birth_statistics_l1655_165537


namespace min_value_a_plus_2b_min_value_is_2root2_l1655_165521

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 1)⁻¹ + (b + 1)⁻¹ = 1) : 
  ∀ x y : ℝ, x > 0 → y > 0 → (x + 1)⁻¹ + (y + 1)⁻¹ = 1 → a + 2*b ≤ x + 2*y :=
by
  sorry

theorem min_value_is_2root2 (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 1)⁻¹ + (b + 1)⁻¹ = 1) : 
  a + 2*b = 2 * Real.sqrt 2 :=
by
  sorry

end min_value_a_plus_2b_min_value_is_2root2_l1655_165521


namespace point_in_third_quadrant_l1655_165591

theorem point_in_third_quadrant :
  let A : ℝ × ℝ := (Real.sin (2014 * π / 180), Real.cos (2014 * π / 180))
  A.1 < 0 ∧ A.2 < 0 :=
by sorry

end point_in_third_quadrant_l1655_165591


namespace ben_gross_income_l1655_165529

-- Define Ben's financial situation
def ben_finances (gross_income : ℝ) : Prop :=
  ∃ (after_tax_income : ℝ),
    -- 20% of after-tax income is spent on car
    0.2 * after_tax_income = 400 ∧
    -- 1/3 of gross income is paid in taxes
    gross_income - (1/3) * gross_income = after_tax_income

-- Theorem statement
theorem ben_gross_income :
  ∃ (gross_income : ℝ), ben_finances gross_income ∧ gross_income = 3000 :=
sorry

end ben_gross_income_l1655_165529


namespace endpoint_sum_thirteen_l1655_165570

/-- Given a line segment with one endpoint (6,1) and midpoint (3,7),
    the sum of the coordinates of the other endpoint is 13. -/
theorem endpoint_sum_thirteen (x y : ℝ) : 
  (6 + x) / 2 = 3 ∧ (1 + y) / 2 = 7 → x + y = 13 := by
  sorry

end endpoint_sum_thirteen_l1655_165570


namespace michael_crates_thursday_l1655_165588

/-- The number of crates Michael bought on Thursday -/
def crates_bought_thursday (initial_crates : ℕ) (crates_given : ℕ) (eggs_per_crate : ℕ) (final_eggs : ℕ) : ℕ :=
  (final_eggs - (initial_crates - crates_given) * eggs_per_crate) / eggs_per_crate

theorem michael_crates_thursday :
  crates_bought_thursday 6 2 30 270 = 5 := by
  sorry

end michael_crates_thursday_l1655_165588


namespace sum_of_trapezoid_areas_l1655_165545

/-- Represents a trapezoid with four side lengths --/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the area of a trapezoid given its side lengths --/
def trapezoidArea (t : Trapezoid) : ℝ := sorry

/-- Generates all valid trapezoid configurations with given side lengths --/
def validConfigurations (sides : List ℝ) : List Trapezoid := sorry

/-- Theorem: The sum of areas of all valid trapezoids with side lengths 4, 6, 8, and 10 
    is equal to a specific value --/
theorem sum_of_trapezoid_areas :
  let sides := [4, 6, 8, 10]
  let validTraps := validConfigurations sides
  let areas := validTraps.map trapezoidArea
  ∃ (total : ℝ), areas.sum = total := by sorry

end sum_of_trapezoid_areas_l1655_165545


namespace modular_congruence_solution_l1655_165580

theorem modular_congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -867 [ZMOD 13] ∧ n = 9 := by
  sorry

end modular_congruence_solution_l1655_165580


namespace chair_arrangement_l1655_165595

theorem chair_arrangement (total_chairs : ℕ) (h : total_chairs = 10000) :
  ∃ (n : ℕ), n * n = total_chairs :=
sorry

end chair_arrangement_l1655_165595


namespace midpoint_coordinates_l1655_165528

/-- The midpoint coordinates of the line segment cut by the parabola y^2 = 4x from the line y = x - 1 are (3, 2). -/
theorem midpoint_coordinates (x y : ℝ) : 
  y^2 = 4*x ∧ y = x - 1 → 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 
    (x1^2 - 1)^2 = 4*x1 ∧ 
    (x2^2 - 1)^2 = 4*x2 ∧
    ((x1 + x2) / 2 = 3 ∧ ((x1 - 1) + (x2 - 1)) / 2 = 2) :=
by sorry

end midpoint_coordinates_l1655_165528


namespace tetrahedron_acute_angle_vertex_l1655_165569

/-- A tetrahedron is represented by its four vertices in 3D space -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- The plane angle at a vertex of a tetrahedron -/
def planeAngle (t : Tetrahedron) (v : Fin 4) (e1 e2 : Fin 4) : ℝ :=
  sorry

/-- Theorem: In any tetrahedron, there exists at least one vertex where all plane angles are acute -/
theorem tetrahedron_acute_angle_vertex (t : Tetrahedron) : 
  ∃ v : Fin 4, ∀ e1 e2 : Fin 4, e1 ≠ e2 → e1 ≠ v → e2 ≠ v → planeAngle t v e1 e2 < π / 2 :=
sorry

end tetrahedron_acute_angle_vertex_l1655_165569


namespace inequality_solution_l1655_165531

theorem inequality_solution (b : ℝ) (h : ∀ x : ℝ, x ∈ Set.Icc 0 1 → ∃ a : ℝ, x * |x - a| + b < 0) :
  ((-1 ≤ b ∧ b < 2 * Real.sqrt 2 - 3 →
    ∃ a : ℝ, a ∈ Set.Ioo (1 + b) (2 * Real.sqrt (-b))) ∧
   (b < -1 →
    ∃ a : ℝ, a ∈ Set.Ioo (1 + b) (1 - b))) :=
by sorry

end inequality_solution_l1655_165531


namespace election_vote_ratio_l1655_165535

theorem election_vote_ratio (marcy_votes barry_votes joey_votes : ℕ) : 
  marcy_votes = 66 →
  marcy_votes = 3 * barry_votes →
  joey_votes = 8 →
  barry_votes ≠ 0 →
  barry_votes / (joey_votes + 3) = 2 / 1 := by
sorry

end election_vote_ratio_l1655_165535


namespace smallest_m_for_tax_price_l1655_165554

theorem smallest_m_for_tax_price : ∃ (x : ℕ), x > 0 ∧ x + (6 * x) / 100 = 2 * 53 * 100 ∧
  ∀ (m : ℕ) (y : ℕ), m > 0 ∧ m < 53 → y > 0 → y + (6 * y) / 100 ≠ 2 * m * 100 := by
  sorry

end smallest_m_for_tax_price_l1655_165554


namespace football_match_end_time_l1655_165558

-- Define a custom time type
structure Time where
  hour : Nat
  minute : Nat

-- Define a function to add minutes to a time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute + m
  { hour := totalMinutes / 60, minute := totalMinutes % 60 }

-- State the theorem
theorem football_match_end_time :
  let start_time : Time := { hour := 15, minute := 30 }
  let duration : Nat := 145
  let end_time : Time := addMinutes start_time duration
  end_time = { hour := 17, minute := 55 } := by
  sorry

end football_match_end_time_l1655_165558


namespace complex_modulus_problem_l1655_165543

theorem complex_modulus_problem (z : ℂ) (h : z * Complex.I = 2 - Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l1655_165543


namespace smallest_number_with_remainders_l1655_165525

theorem smallest_number_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧
  (n % 4 = 3) ∧
  (n % 5 = 4) ∧
  (n % 6 = 5) ∧
  (n % 7 = 6) ∧
  (∀ m : ℕ, m > 0 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 ∧ m % 7 = 6 → m ≥ n) ∧
  n = 419 :=
by sorry

end smallest_number_with_remainders_l1655_165525


namespace add_same_power_of_x_l1655_165502

theorem add_same_power_of_x (x : ℝ) : x^3 + x^3 = 2*x^3 := by
  sorry

end add_same_power_of_x_l1655_165502


namespace contractor_fine_calculation_l1655_165596

/-- Calculates the daily fine for a contractor given contract details -/
def calculate_daily_fine (contract_duration : ℕ) (daily_pay : ℕ) (total_payment : ℕ) (days_absent : ℕ) : ℚ :=
  let days_worked := contract_duration - days_absent
  let total_earned := days_worked * daily_pay
  ((total_earned - total_payment) : ℚ) / days_absent

theorem contractor_fine_calculation :
  let contract_duration : ℕ := 30
  let daily_pay : ℕ := 25
  let total_payment : ℕ := 425
  let days_absent : ℕ := 10
  calculate_daily_fine contract_duration daily_pay total_payment days_absent = 15/2 := by
  sorry

#eval calculate_daily_fine 30 25 425 10

end contractor_fine_calculation_l1655_165596


namespace circle_line_distance_l1655_165551

theorem circle_line_distance (a : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*x - 4*y = 0}
  let line := {(x, y) : ℝ × ℝ | x - y + a = 0}
  let center := (1, 2)
  let distance := |1 - 2 + a| / Real.sqrt 2
  (distance = Real.sqrt 2 / 2) → (a = 2 ∨ a = 0) := by
sorry

end circle_line_distance_l1655_165551


namespace arcade_spending_correct_l1655_165553

/-- Calculates the total money spent by John at the arcade --/
def arcade_spending (total_time : ℕ) (break_time : ℕ) (rate1 : ℚ) (interval1 : ℕ)
  (rate2 : ℚ) (interval2 : ℕ) (rate3 : ℚ) (interval3 : ℕ) (rate4 : ℚ) (interval4 : ℕ) : ℚ :=
  let play_time := total_time - break_time
  let hour1 := (60 / interval1) * rate1
  let hour2 := (60 / interval2) * rate2
  let hour3 := (60 / interval3) * rate3
  let hour4 := ((play_time - 180) / interval4) * rate4
  hour1 + hour2 + hour3 + hour4

theorem arcade_spending_correct :
  arcade_spending 275 50 0.5 4 0.75 5 1 3 1.25 7 = 42.75 := by
  sorry

end arcade_spending_correct_l1655_165553


namespace quadratic_binomial_square_l1655_165555

theorem quadratic_binomial_square (b r : ℚ) : 
  (∀ x, b * x^2 + 20 * x + 16 = (r * x - 4)^2) → b = 25/4 := by
  sorry

end quadratic_binomial_square_l1655_165555


namespace min_value_f_and_g_inequality_l1655_165520

noncomputable section

variable (t : ℝ)

def f (x : ℝ) := Real.exp (2 * x) - 2 * t * x

def g (x : ℝ) := -x^2 + 2 * t * Real.exp x - 2 * t^2 + 1/2

theorem min_value_f_and_g_inequality :
  (t ≤ 1 → ∀ x ≥ 0, f t x ≥ 1) ∧
  (t > 1 → ∀ x ≥ 0, f t x ≥ t - t * Real.log t) ∧
  (t = 1 → ∀ x ≥ 0, g t x ≥ 1/2) := by
  sorry

end

end min_value_f_and_g_inequality_l1655_165520


namespace increase_average_by_transfer_l1655_165538

def group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def removeElement (l : List ℕ) (x : ℕ) : List ℕ :=
  l.filter (· ≠ x)

theorem increase_average_by_transfer :
  ∃ g ∈ group1,
    average (removeElement group1 g) > average group1 ∧
    average (g :: group2) > average group2 := by
  sorry

end increase_average_by_transfer_l1655_165538


namespace books_per_week_after_second_l1655_165509

theorem books_per_week_after_second (total_books : ℕ) (first_week : ℕ) (second_week : ℕ) (total_weeks : ℕ) :
  total_books = 54 →
  first_week = 6 →
  second_week = 3 →
  total_weeks = 7 →
  (total_books - (first_week + second_week)) / (total_weeks - 2) = 9 :=
by sorry

end books_per_week_after_second_l1655_165509


namespace tangent_function_l1655_165533

/-- Given a function f(x) = ax / (x^2 + b), prove that if f(1) = 2 and f'(1) = 0, 
    then f(x) = 4x / (x^2 + 1) -/
theorem tangent_function (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * x / (x^2 + b)
  (f 1 = 2) → (deriv f 1 = 0) → ∀ x, f x = 4 * x / (x^2 + 1) := by
sorry

end tangent_function_l1655_165533


namespace f_decreasing_on_interval_l1655_165514

def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 1

theorem f_decreasing_on_interval : 
  ∀ x ∈ Set.Ioo (-2 : ℝ) 1, (deriv f) x < 0 := by sorry

end f_decreasing_on_interval_l1655_165514


namespace sequence_general_term_l1655_165547

theorem sequence_general_term (a : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 2 → a n / a (n - 1) = 2^(n - 1)) →
  a 1 = 1 →
  ∀ n : ℕ, n > 0 → a n = 2^(n * (n - 1) / 2) :=
by sorry

end sequence_general_term_l1655_165547


namespace baseball_league_games_played_l1655_165564

/-- Represents a baseball league with a given number of teams and games per pair of teams. -/
structure BaseballLeague where
  num_teams : ℕ
  games_per_pair : ℕ

/-- Calculates the total number of games played in a baseball league with one team forfeiting one game against each other team. -/
def games_actually_played (league : BaseballLeague) : ℕ :=
  let total_scheduled := (league.num_teams * (league.num_teams - 1) * league.games_per_pair) / 2
  let forfeited_games := league.num_teams - 1
  total_scheduled - forfeited_games

/-- Theorem stating that in a baseball league with 10 teams, 4 games per pair, and one team forfeiting one game against each other team, the total number of games actually played is 171. -/
theorem baseball_league_games_played :
  let league : BaseballLeague := { num_teams := 10, games_per_pair := 4 }
  games_actually_played league = 171 := by
  sorry

end baseball_league_games_played_l1655_165564


namespace special_function_property_l1655_165560

/-- A continuously differentiable function satisfying f'(t) > f(f(t)) for all t ∈ ℝ -/
structure SpecialFunction where
  f : ℝ → ℝ
  cont_diff : ContDiff ℝ 1 f
  property : ∀ t : ℝ, deriv f t > f (f t)

/-- The main theorem -/
theorem special_function_property (sf : SpecialFunction) :
  ∀ t : ℝ, t ≥ 0 → sf.f (sf.f (sf.f t)) ≤ 0 := by
  sorry

end special_function_property_l1655_165560


namespace probability_x_more_points_than_y_in_given_tournament_l1655_165571

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  num_teams : Nat
  num_games_per_team : Nat
  win_probability : ℚ
  x_beats_y_first : Bool

/-- The probability that team X finishes with more points than team Y -/
def probability_x_more_points_than_y (tournament : SoccerTournament) : ℚ :=
  sorry

/-- The specific tournament described in the problem -/
def given_tournament : SoccerTournament where
  num_teams := 8
  num_games_per_team := 7
  win_probability := 1/2
  x_beats_y_first := true

theorem probability_x_more_points_than_y_in_given_tournament :
  probability_x_more_points_than_y given_tournament = 610/1024 := by
  sorry

end probability_x_more_points_than_y_in_given_tournament_l1655_165571


namespace wanda_walking_distance_l1655_165526

/-- Represents the distance Wanda walks in miles -/
def distance_to_school : ℝ := 0.5

/-- Represents the number of round trips Wanda makes per day -/
def round_trips_per_day : ℕ := 2

/-- Represents the number of days Wanda walks to school per week -/
def school_days_per_week : ℕ := 5

/-- Represents the number of weeks we're considering -/
def weeks : ℕ := 4

/-- Theorem stating that Wanda walks 40 miles after 4 weeks -/
theorem wanda_walking_distance : 
  2 * distance_to_school * round_trips_per_day * school_days_per_week * weeks = 40 := by
  sorry


end wanda_walking_distance_l1655_165526


namespace cubic_from_quadratic_roots_l1655_165530

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots r and s,
    this theorem states the form of the cubic equation
    with roots r^2 + br + a and s^2 + bs + a. -/
theorem cubic_from_quadratic_roots (a b c r s : ℝ) : 
  (a * r^2 + b * r + c = 0) →
  (a * s^2 + b * s + c = 0) →
  (r ≠ s) →
  ∃ (p q : ℝ), 
    (x^3 + (b^2 - a*b^2 + 4*a^3 - 2*a*c)/a^2 * x^2 + p*x + q = 0) ↔ 
    (x = r^2 + b*r + a ∨ x = s^2 + b*s + a ∨ x = -(r^2 + b*r + a + s^2 + b*s + a)) :=
by sorry

end cubic_from_quadratic_roots_l1655_165530


namespace fraction_meaningful_l1655_165510

theorem fraction_meaningful (x : ℝ) : 
  (x - 2) / (2 * x - 3) ≠ 0 ↔ x ≠ 3/2 :=
by sorry

end fraction_meaningful_l1655_165510


namespace jorge_corn_yield_l1655_165585

/-- Calculates the total corn yield from Jorge's land -/
theorem jorge_corn_yield (total_land : ℝ) (good_soil_yield : ℝ) 
  (clay_soil_fraction : ℝ) (h1 : total_land = 60) 
  (h2 : good_soil_yield = 400) (h3 : clay_soil_fraction = 1/3) : 
  total_land * (clay_soil_fraction * (good_soil_yield / 2) + 
  (1 - clay_soil_fraction) * good_soil_yield) = 20000 := by
  sorry

#check jorge_corn_yield

end jorge_corn_yield_l1655_165585


namespace train_passing_time_l1655_165586

/-- The time taken for a slower train to pass the driver of a faster train -/
theorem train_passing_time (length : ℝ) (speed_fast speed_slow : ℝ) :
  length = 500 →
  speed_fast = 45 →
  speed_slow = 30 →
  let relative_speed := speed_fast + speed_slow
  let relative_speed_ms := relative_speed * 1000 / 3600
  let time := length / relative_speed_ms
  ∃ ε > 0, |time - 24| < ε :=
by sorry

end train_passing_time_l1655_165586


namespace function_inequality_l1655_165590

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x < (deriv^[2] f) x) : 
  f 1 > ℯ * f 0 ∧ f 2019 > ℯ^2019 * f 0 := by
  sorry

end function_inequality_l1655_165590


namespace regular_polygon_exterior_angle_l1655_165559

theorem regular_polygon_exterior_angle (n : ℕ) (n_pos : 0 < n) :
  (360 : ℝ) / n = 30 → n = 12 := by
  sorry

end regular_polygon_exterior_angle_l1655_165559


namespace no_real_solutions_l1655_165534

theorem no_real_solutions : ¬∃ x : ℝ, (2*x - 6)^2 + 4 = -(x - 3) := by
  sorry

end no_real_solutions_l1655_165534


namespace quadratic_equation_solution_l1655_165540

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x => x^2 - 2*x - 3
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = -1 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end quadratic_equation_solution_l1655_165540


namespace axis_of_symmetry_is_x_equals_one_l1655_165507

/-- Represents a parabola of the form y = a(x-h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The axis of symmetry of a parabola --/
def axisOfSymmetry (p : Parabola) : ℝ := p.h

/-- The given parabola y = -2(x-1)^2 + 3 --/
def givenParabola : Parabola := ⟨-2, 1, 3⟩

theorem axis_of_symmetry_is_x_equals_one :
  axisOfSymmetry givenParabola = 1 := by sorry

end axis_of_symmetry_is_x_equals_one_l1655_165507


namespace third_cube_edge_l1655_165583

theorem third_cube_edge (a b c x : ℝ) (ha : a = 3) (hb : b = 5) (hc : c = 6) :
  a^3 + b^3 + x^3 = c^3 → x = 4 := by sorry

end third_cube_edge_l1655_165583


namespace ramsey_r33_l1655_165548

/-- Represents the relationship between two people -/
inductive Relationship
| Acquaintance
| Stranger

/-- A group of people -/
def People := Fin 6

/-- The relationship between each pair of people -/
def RelationshipMap := People → People → Relationship

/-- Checks if three people are mutual acquaintances -/
def areMutualAcquaintances (rel : RelationshipMap) (a b c : People) : Prop :=
  rel a b = Relationship.Acquaintance ∧
  rel a c = Relationship.Acquaintance ∧
  rel b c = Relationship.Acquaintance

/-- Checks if three people are mutual strangers -/
def areMutualStrangers (rel : RelationshipMap) (a b c : People) : Prop :=
  rel a b = Relationship.Stranger ∧
  rel a c = Relationship.Stranger ∧
  rel b c = Relationship.Stranger

/-- Main theorem: In a group of 6 people, there are either 3 mutual acquaintances or 3 mutual strangers -/
theorem ramsey_r33 (rel : RelationshipMap) :
  (∃ a b c : People, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ areMutualAcquaintances rel a b c) ∨
  (∃ a b c : People, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ areMutualStrangers rel a b c) :=
sorry

end ramsey_r33_l1655_165548


namespace min_value_theorem_l1655_165566

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 4) :
  (9 / a) + (16 / b) + (25 / c) ≥ 36 := by
  sorry

end min_value_theorem_l1655_165566


namespace function_properties_and_triangle_l1655_165565

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.sin x + Real.cos x

theorem function_properties_and_triangle (m : ℝ) (A : ℝ) :
  f m (π / 2) = 1 →
  f m (π / 12) = Real.sqrt 2 * Real.sin A →
  0 < A ∧ A < π / 2 →
  (3 * Real.sqrt 3) / 2 = 1 / 2 * 2 * 3 * Real.sin A →
  m = 1 ∧
  (∀ x : ℝ, f m (x + 2 * π) = f m x) ∧
  (∀ x : ℝ, f m x ≤ Real.sqrt 2) ∧
  (∀ x : ℝ, f m x ≥ -Real.sqrt 2) ∧
  A = π / 3 ∧
  Real.sqrt 7 = Real.sqrt (3^2 + 2^2 - 2 * 2 * 3 * Real.cos A) :=
by sorry

end function_properties_and_triangle_l1655_165565


namespace shorter_leg_length_l1655_165572

/-- A right triangle that can be cut and rearranged into a square -/
structure CuttableRightTriangle where
  shorter_leg : ℝ
  longer_leg : ℝ
  is_right_triangle : shorter_leg > 0 ∧ longer_leg > 0
  can_form_square : shorter_leg * 2 = longer_leg

/-- Theorem: If a right triangle with longer leg 10 can be cut and rearranged 
    to form a square, then its shorter leg has length 5 -/
theorem shorter_leg_length (t : CuttableRightTriangle) 
    (h : t.longer_leg = 10) : t.shorter_leg = 5 := by
  sorry

end shorter_leg_length_l1655_165572


namespace union_equality_implies_range_l1655_165561

-- Define the sets P and M
def P : Set ℝ := {x | x^2 ≤ 1}
def M (a : ℝ) : Set ℝ := {a}

-- State the theorem
theorem union_equality_implies_range (a : ℝ) :
  P ∪ M a = P → a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end union_equality_implies_range_l1655_165561


namespace max_cells_hit_five_times_l1655_165573

/-- Represents a triangular cell in the grid -/
structure TriangularCell :=
  (id : ℕ)

/-- Represents the entire triangular grid -/
structure TriangularGrid :=
  (cells : List TriangularCell)

/-- Represents a shot fired by the marksman -/
structure Shot :=
  (target : TriangularCell)

/-- Function to determine if two cells are adjacent -/
def areAdjacent (c1 c2 : TriangularCell) : Bool :=
  sorry

/-- Function to determine where a shot lands -/
def shotLands (s : Shot) (g : TriangularGrid) : TriangularCell :=
  sorry

/-- Function to count the number of hits on a cell -/
def countHits (c : TriangularCell) (shots : List Shot) : ℕ :=
  sorry

/-- Theorem stating the maximum number of cells that can be hit exactly five times -/
theorem max_cells_hit_five_times (g : TriangularGrid) :
  (∃ (shots : List Shot), 
    (∀ c : TriangularCell, c ∈ g.cells → countHits c shots ≤ 5) ∧ 
    (∃ cells : List TriangularCell, 
      cells.length = 25 ∧ 
      (∀ c : TriangularCell, c ∈ cells → countHits c shots = 5))) ∧
  (∀ (shots : List Shot),
    ¬∃ cells : List TriangularCell, 
      cells.length > 25 ∧ 
      (∀ c : TriangularCell, c ∈ cells → countHits c shots = 5)) :=
  sorry

end max_cells_hit_five_times_l1655_165573


namespace similar_triangles_leg_l1655_165575

/-- Two similar right triangles with legs 12 and 9 in the first triangle,
    and y and 6 in the second triangle, have y = 8 -/
theorem similar_triangles_leg (y : ℝ) : 
  (12 : ℝ) / y = 9 / 6 → y = 8 := by
  sorry

end similar_triangles_leg_l1655_165575


namespace polynomial_roots_coefficient_sum_l1655_165576

theorem polynomial_roots_coefficient_sum (p q r : ℝ) : 
  (∃ a b c : ℝ, 0 < a ∧ a < 2 ∧ 0 < b ∧ b < 2 ∧ 0 < c ∧ c < 2 ∧
    ∀ x : ℝ, x^3 + p*x^2 + q*x + r = (x - a) * (x - b) * (x - c)) →
  -2 < p + q + r ∧ p + q + r < 0 :=
by sorry

end polynomial_roots_coefficient_sum_l1655_165576


namespace sin_cos_identity_l1655_165516

theorem sin_cos_identity (α c d : ℝ) (h : c > 0) (k : d > 0) 
  (eq : (Real.sin α)^6 / c + (Real.cos α)^6 / d = 1 / (c + d)) :
  (Real.sin α)^12 / c^5 + (Real.cos α)^12 / d^5 = 1 / (c + d)^5 := by
  sorry

end sin_cos_identity_l1655_165516


namespace maddy_graduation_time_l1655_165587

/-- The number of semesters Maddy needs to be in college -/
def semesters_needed (total_credits : ℕ) (credits_per_class : ℕ) (classes_per_semester : ℕ) : ℕ :=
  total_credits / (credits_per_class * classes_per_semester)

/-- Proof that Maddy needs 8 semesters to graduate -/
theorem maddy_graduation_time :
  semesters_needed 120 3 5 = 8 := by
  sorry

end maddy_graduation_time_l1655_165587


namespace min_value_reciprocal_sum_l1655_165503

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : Real.log (a + b) = 0) :
  (1 / a + 1 / b) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ Real.log (a₀ + b₀) = 0 ∧ 1 / a₀ + 1 / b₀ = 4 :=
by sorry

end min_value_reciprocal_sum_l1655_165503


namespace unique_prime_triple_l1655_165500

theorem unique_prime_triple : 
  ∃! (p q r : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    (∃ k : ℤ, (p^2 + 2*q : ℤ) = k * (q + r : ℤ)) ∧
    (∃ l : ℤ, (q^2 + 9*r : ℤ) = l * (r + p : ℤ)) ∧
    (∃ m : ℤ, (r^2 + 3*p : ℤ) = m * (p + q : ℤ)) ∧
    p = 2 ∧ q = 3 ∧ r = 7 :=
by sorry

end unique_prime_triple_l1655_165500


namespace score_difference_l1655_165556

/-- Represents the runs scored by each batsman -/
structure BatsmanScores where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- Theorem stating the difference between A's and E's scores -/
theorem score_difference (scores : BatsmanScores) : scores.a - scores.e = 8 :=
  by
  have h1 : scores.a + scores.b + scores.c + scores.d + scores.e = 180 :=
    sorry -- Average score is 36, so total is 5 * 36 = 180
  have h2 : scores.d = scores.e + 5 := sorry -- D scored 5 more than E
  have h3 : scores.e = 20 := sorry -- E scored 20 runs
  have h4 : scores.b = scores.d + scores.e := sorry -- B scored as many as D and E combined
  have h5 : scores.b + scores.c = 107 := sorry -- B and C scored 107 between them
  have h6 : scores.e < scores.a := sorry -- E scored fewer runs than A
  sorry -- Prove that scores.a - scores.e = 8

end score_difference_l1655_165556


namespace quiz_answer_key_l1655_165597

theorem quiz_answer_key (n : ℕ) : 
  (14 * n^2 = 224) → n = 4 :=
by
  sorry

#check quiz_answer_key

end quiz_answer_key_l1655_165597


namespace min_value_of_sqrt_sums_l1655_165501

theorem min_value_of_sqrt_sums (a b c : ℝ) : 
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c → 
  a * b + b * c + c * a = a + b + c → 
  0 < a + b + c → 
  2 ≤ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) :=
by sorry

end min_value_of_sqrt_sums_l1655_165501


namespace binomial_320_320_l1655_165578

theorem binomial_320_320 : Nat.choose 320 320 = 1 := by sorry

end binomial_320_320_l1655_165578


namespace alex_total_cost_l1655_165562

/-- Calculates the total cost of a cell phone plan given the usage and rates. -/
def calculate_total_cost (base_cost : ℚ) (included_hours : ℚ) (text_cost : ℚ) 
  (extra_minute_cost : ℚ) (texts_sent : ℕ) (hours_talked : ℚ) : ℚ :=
  let extra_hours := max (hours_talked - included_hours) 0
  let extra_minutes := extra_hours * 60
  base_cost + (text_cost * texts_sent) + (extra_minute_cost * extra_minutes)

/-- Proves that Alex's total cost is $109.00 given the specified plan and usage. -/
theorem alex_total_cost : 
  calculate_total_cost 25 25 0.08 0.15 150 33 = 109 := by
  sorry

end alex_total_cost_l1655_165562


namespace polynomial_simplification_l1655_165506

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^6 + x^5 + 3 * x^4 + 7 * x^2 + 2 * x + 25) - 
  (x^6 + 2 * x^5 + x^4 + x^3 + 8 * x^2 + 15) = 
  x^6 - x^5 + 2 * x^4 - x^3 - x^2 + 2 * x + 10 := by
sorry

end polynomial_simplification_l1655_165506


namespace student_arrangement_count_l1655_165577

/-- Represents the number of students -/
def n : ℕ := 5

/-- Represents the condition that two specific students must be together -/
def must_be_together : Prop := True

/-- Represents the condition that two specific students cannot be together -/
def cannot_be_together : Prop := True

/-- The number of ways to arrange the students -/
def arrangement_count : ℕ := 24

/-- Theorem stating that the number of arrangements satisfying the conditions is 24 -/
theorem student_arrangement_count :
  must_be_together → cannot_be_together → arrangement_count = 24 := by
  sorry

end student_arrangement_count_l1655_165577


namespace alices_number_l1655_165563

theorem alices_number (x : ℝ) : 3 * (3 * x - 6) = 141 → x = 17 := by
  sorry

end alices_number_l1655_165563


namespace max_notebooks_is_11_l1655_165512

/-- Represents the number of notebooks in a pack -/
inductive PackSize
  | Single
  | Pack4
  | Pack7

/-- Returns the number of notebooks for a given pack size -/
def notebooks (size : PackSize) : ℕ :=
  match size with
  | PackSize.Single => 1
  | PackSize.Pack4 => 4
  | PackSize.Pack7 => 7

/-- Returns the cost in dollars for a given pack size -/
def cost (size : PackSize) : ℕ :=
  match size with
  | PackSize.Single => 2
  | PackSize.Pack4 => 6
  | PackSize.Pack7 => 10

/-- Represents a purchase of notebook packs -/
structure Purchase where
  single : ℕ
  pack4 : ℕ
  pack7 : ℕ

/-- Calculates the total number of notebooks for a given purchase -/
def totalNotebooks (p : Purchase) : ℕ :=
  p.single * notebooks PackSize.Single +
  p.pack4 * notebooks PackSize.Pack4 +
  p.pack7 * notebooks PackSize.Pack7

/-- Calculates the total cost for a given purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.single * cost PackSize.Single +
  p.pack4 * cost PackSize.Pack4 +
  p.pack7 * cost PackSize.Pack7

/-- Represents the budget constraint -/
def budget : ℕ := 17

/-- Theorem: The maximum number of notebooks that can be purchased within the budget is 11 -/
theorem max_notebooks_is_11 :
  ∀ p : Purchase, totalCost p ≤ budget → totalNotebooks p ≤ 11 ∧
  ∃ p' : Purchase, totalCost p' ≤ budget ∧ totalNotebooks p' = 11 :=
sorry

end max_notebooks_is_11_l1655_165512


namespace jeanne_initial_tickets_l1655_165532

/-- The cost of all three attractions in tickets -/
def total_cost : ℕ := 13

/-- The number of additional tickets Jeanne needs to buy -/
def additional_tickets : ℕ := 8

/-- Jeanne's initial number of tickets -/
def initial_tickets : ℕ := total_cost - additional_tickets

theorem jeanne_initial_tickets : initial_tickets = 5 := by
  sorry

end jeanne_initial_tickets_l1655_165532


namespace duo_ball_playing_time_l1655_165582

theorem duo_ball_playing_time (num_children : ℕ) (total_time : ℕ) (players_per_game : ℕ) :
  num_children = 8 →
  total_time = 120 →
  players_per_game = 2 →
  (total_time * players_per_game) / num_children = 30 :=
by
  sorry

end duo_ball_playing_time_l1655_165582


namespace rhombus_area_l1655_165581

/-- The area of a rhombus with side length 4 cm and one angle of 45° is 8√2 square centimeters. -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = π/4) :
  let area := s * s * Real.sin θ
  area = 8 * Real.sqrt 2 := by sorry

end rhombus_area_l1655_165581


namespace quadratic_inequality_implies_upper_bound_l1655_165552

theorem quadratic_inequality_implies_upper_bound (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 3 → a ≤ x^2 - 4*x) →
  a ≤ -4 := by
sorry

end quadratic_inequality_implies_upper_bound_l1655_165552


namespace kevin_six_hops_l1655_165513

def kevin_hop (n : ℕ) : ℚ :=
  2 * (1 - (3/4)^n)

theorem kevin_six_hops :
  kevin_hop 6 = 3367 / 2048 := by
  sorry

end kevin_six_hops_l1655_165513


namespace divisible_by_eleven_l1655_165557

theorem divisible_by_eleven (n : ℕ) : ∃ k : ℤ, 5^(2*n) + 3^(n+2) + 3^n = 11 * k := by
  sorry

end divisible_by_eleven_l1655_165557


namespace students_not_in_sports_l1655_165574

/-- The number of students in the class -/
def total_students : ℕ := 50

/-- The number of students playing basketball -/
def basketball : ℕ := total_students / 2

/-- The number of students playing volleyball -/
def volleyball : ℕ := total_students / 3

/-- The number of students playing soccer -/
def soccer : ℕ := total_students / 5

/-- The number of students playing badminton -/
def badminton : ℕ := total_students / 8

/-- The number of students playing both basketball and volleyball -/
def basketball_and_volleyball : ℕ := total_students / 10

/-- The number of students playing both basketball and soccer -/
def basketball_and_soccer : ℕ := total_students / 12

/-- The number of students playing both basketball and badminton -/
def basketball_and_badminton : ℕ := total_students / 16

/-- The number of students playing both volleyball and soccer -/
def volleyball_and_soccer : ℕ := total_students / 8

/-- The number of students playing both volleyball and badminton -/
def volleyball_and_badminton : ℕ := total_students / 10

/-- The number of students playing both soccer and badminton -/
def soccer_and_badminton : ℕ := total_students / 20

/-- The number of students playing all four sports -/
def all_four_sports : ℕ := total_students / 25

/-- The theorem stating that 16 students do not engage in any of the four sports -/
theorem students_not_in_sports : 
  total_students - (basketball + volleyball + soccer + badminton 
  - basketball_and_volleyball - basketball_and_soccer - basketball_and_badminton 
  - volleyball_and_soccer - volleyball_and_badminton - soccer_and_badminton 
  + all_four_sports) = 16 := by
  sorry

end students_not_in_sports_l1655_165574


namespace binomial_square_constant_l1655_165541

theorem binomial_square_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 - 27*x + a = (3*x + b)^2) → a = 20.25 := by
  sorry

end binomial_square_constant_l1655_165541


namespace parabola_circle_tangent_l1655_165527

/-- The value of p for a parabola y^2 = 2px (p > 0) whose directrix is tangent to the circle (x-3)^2 + y^2 = 16 -/
theorem parabola_circle_tangent (p : ℝ) : 
  p > 0 ∧ 
  (∃ (x y : ℝ), y^2 = 2*p*x) ∧ 
  (∃ (x y : ℝ), (x - 3)^2 + y^2 = 16) ∧
  (∃ (x : ℝ), x = -p/2 ∧ (x - 3)^2 + (2*p*x) = 16) →
  p = 2 :=
sorry

end parabola_circle_tangent_l1655_165527


namespace smallest_x_absolute_value_l1655_165599

theorem smallest_x_absolute_value (x : ℝ) : 
  (|x - 10| = 15) → (x ≥ -5 ∧ (∃ y : ℝ, |y - 10| = 15 ∧ y = -5)) :=
by sorry

end smallest_x_absolute_value_l1655_165599


namespace ship_optimal_speed_and_cost_l1655_165515

/-- The optimal speed and cost for a ship's journey -/
theorem ship_optimal_speed_and_cost (distance : ℝ) (fuel_cost_coeff : ℝ) (fixed_cost : ℝ)
  (h_distance : distance = 100)
  (h_fuel_cost : fuel_cost_coeff = 0.005)
  (h_fixed_cost : fixed_cost = 80) :
  ∃ (optimal_speed : ℝ) (min_cost : ℝ),
    optimal_speed = 20 ∧
    min_cost = 600 ∧
    ∀ (v : ℝ), v > 0 →
      distance / v * (fuel_cost_coeff * v^3 + fixed_cost) ≥ min_cost :=
by sorry

end ship_optimal_speed_and_cost_l1655_165515


namespace pirate_costume_cost_l1655_165508

theorem pirate_costume_cost (num_friends : ℕ) (total_spent : ℕ) : 
  num_friends = 8 → total_spent = 40 → total_spent / num_friends = 5 := by
  sorry

end pirate_costume_cost_l1655_165508


namespace quadratic_inequality_range_l1655_165517

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) → a ∈ Set.Ioi 3 ∪ Set.Iio (-1) := by
  sorry

end quadratic_inequality_range_l1655_165517


namespace david_catches_cory_l1655_165568

/-- The length of the track in meters -/
def track_length : ℝ := 600

/-- Cory's initial lead in meters -/
def initial_lead : ℝ := 50

/-- David's speed relative to Cory's -/
def speed_ratio : ℝ := 1.5

/-- Number of laps David runs when he first catches up to Cory -/
def david_laps : ℝ := 2

theorem david_catches_cory :
  ∃ (cory_speed : ℝ), cory_speed > 0 →
  let david_speed := speed_ratio * cory_speed
  let catch_up_distance := david_laps * track_length
  catch_up_distance * (1 / david_speed - 1 / cory_speed) = initial_lead := by
  sorry

end david_catches_cory_l1655_165568


namespace min_height_box_l1655_165594

def box_height (side_length : ℝ) : ℝ := 2 * side_length

def surface_area (side_length : ℝ) : ℝ := 10 * side_length^2

theorem min_height_box (min_area : ℝ) (h_min_area : min_area = 120) :
  ∃ (h : ℝ), h = box_height (Real.sqrt (min_area / 10)) ∧
             h = 8 ∧
             ∀ (s : ℝ), surface_area s ≥ min_area → box_height s ≥ h :=
by sorry

end min_height_box_l1655_165594


namespace base_difference_theorem_l1655_165505

/-- Converts a number from base 5 to base 10 -/
def base5_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

theorem base_difference_theorem :
  let n1 := 543210
  let n2 := 43210
  (base5_to_base10 n1) - (base8_to_base10 n2) = 499 := by sorry

end base_difference_theorem_l1655_165505


namespace m_plus_e_equals_22_l1655_165524

def base_value (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

theorem m_plus_e_equals_22 (m e : Nat) :
  m > 0 →
  e < 10 →
  base_value [4, 1, e] m = 346 →
  base_value [4, 1, 6] m = base_value [1, 2, e, 1] 7 →
  m + e = 22 :=
by sorry

end m_plus_e_equals_22_l1655_165524


namespace lines_are_parallel_l1655_165536

/-- Two lines a₁x + b₁y + c₁ = 0 and a₂x + b₂y + c₂ = 0 are parallel if and only if a₁b₂ = a₂b₁ -/
def parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁ ∧ a₁ * c₂ ≠ a₂ * c₁

/-- The line x - 2y + 1 = 0 -/
def line1 : ℝ → ℝ → ℝ := λ x y => x - 2*y + 1

/-- The line 2x - 4y + 1 = 0 -/
def line2 : ℝ → ℝ → ℝ := λ x y => 2*x - 4*y + 1

theorem lines_are_parallel : parallel 1 (-2) 1 2 (-4) 1 :=
  sorry

end lines_are_parallel_l1655_165536
