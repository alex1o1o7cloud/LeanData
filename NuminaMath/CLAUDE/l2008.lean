import Mathlib

namespace workers_total_earning_l2008_200827

/-- Represents the daily wages and work days of three workers -/
structure Workers where
  a_days : ℕ
  b_days : ℕ
  c_days : ℕ
  c_wage : ℕ
  wage_ratio : Fin 3 → ℕ

/-- Calculates the total earnings of the workers -/
def total_earning (w : Workers) : ℕ :=
  let unit := w.c_wage / w.wage_ratio 2
  let a_wage := unit * w.wage_ratio 0
  let b_wage := unit * w.wage_ratio 1
  a_wage * w.a_days + b_wage * w.b_days + w.c_wage * w.c_days

/-- The main theorem stating the total earning of the workers -/
theorem workers_total_earning : ∃ (w : Workers), 
  w.a_days = 6 ∧ 
  w.b_days = 9 ∧ 
  w.c_days = 4 ∧ 
  w.c_wage = 105 ∧ 
  w.wage_ratio = ![3, 4, 5] ∧
  total_earning w = 1554 := by
  sorry

end workers_total_earning_l2008_200827


namespace smallest_prime_between_squares_l2008_200878

theorem smallest_prime_between_squares : ∃ (p : ℕ), 
  Prime p ∧ 
  (∃ (n : ℕ), p = n^2 + 6) ∧ 
  (∃ (m : ℕ), p = (m+1)^2 - 9) ∧
  (∀ (q : ℕ), q < p → 
    (Prime q → ¬(∃ (k : ℕ), q = k^2 + 6 ∧ q = (k+1)^2 - 9))) ∧
  p = 127 := by
sorry

end smallest_prime_between_squares_l2008_200878


namespace pechkin_calculation_error_l2008_200844

/-- Represents Pechkin's journey --/
structure PechkinJourney where
  totalDistance : ℝ
  walkingSpeed : ℝ
  cyclingSpeed : ℝ
  walkingDistance : ℝ
  cyclingTime : ℝ
  totalTime : ℝ

/-- Conditions of Pechkin's journey --/
def journeyConditions (j : PechkinJourney) : Prop :=
  j.walkingSpeed = 5 ∧
  j.cyclingSpeed = 12 ∧
  j.walkingDistance = j.totalDistance / 2 ∧
  j.cyclingTime = j.totalTime / 3

/-- Theorem stating that Pechkin's calculations are inconsistent --/
theorem pechkin_calculation_error (j : PechkinJourney) 
  (h : journeyConditions j) : 
  j.cyclingSpeed * j.cyclingTime ≠ j.totalDistance - j.walkingDistance :=
sorry

end pechkin_calculation_error_l2008_200844


namespace z_value_when_x_is_4_l2008_200897

/-- The constant k in the inverse relationship -/
def k : ℚ := 392

/-- The inverse relationship between z and x -/
def inverse_relation (z x : ℚ) : Prop :=
  7 * z = k / (x^3)

theorem z_value_when_x_is_4 :
  ∀ z : ℚ, inverse_relation 7 2 → inverse_relation z 4 → z = 7/8 :=
by sorry

end z_value_when_x_is_4_l2008_200897


namespace library_visitors_theorem_l2008_200861

/-- Represents the average number of visitors on a given day type -/
structure VisitorAverage where
  sunday : ℕ
  other : ℕ

/-- Represents a month with visitor data -/
structure Month where
  days : ℕ
  startsWithSunday : Bool
  avgVisitorsPerDay : ℕ
  visitorAvg : VisitorAverage

/-- Calculates the number of Sundays in a month -/
def numSundays (m : Month) : ℕ :=
  if m.startsWithSunday then
    (m.days + 6) / 7
  else
    m.days / 7

/-- Theorem: Given the conditions, prove that the average number of visitors on non-Sunday days is 80 -/
theorem library_visitors_theorem (m : Month) 
    (h1 : m.days = 30)
    (h2 : m.startsWithSunday = true)
    (h3 : m.visitorAvg.sunday = 140)
    (h4 : m.avgVisitorsPerDay = 90) :
    m.visitorAvg.other = 80 := by
  sorry


end library_visitors_theorem_l2008_200861


namespace table_tennis_tournament_l2008_200866

theorem table_tennis_tournament (n : ℕ) (total_matches : ℕ) (withdrawn_players : ℕ) 
  (matches_per_withdrawn : ℕ) (h1 : n = 13) (h2 : total_matches = 50) 
  (h3 : withdrawn_players = 3) (h4 : matches_per_withdrawn = 2) : 
  (n.choose 2) - ((n - withdrawn_players).choose 2) - 
  (withdrawn_players * matches_per_withdrawn) = 1 := by
sorry

end table_tennis_tournament_l2008_200866


namespace average_speed_calculation_l2008_200855

theorem average_speed_calculation (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) 
  (second_part_speed : ℝ) (h1 : total_distance = 400) (h2 : first_part_distance = 100) 
  (h3 : first_part_speed = 20) (h4 : second_part_speed = 15) : 
  (total_distance / (first_part_distance / first_part_speed + 
  (total_distance - first_part_distance) / second_part_speed)) = 16 := by
  sorry

end average_speed_calculation_l2008_200855


namespace imaginary_part_of_z_l2008_200826

theorem imaginary_part_of_z : Complex.im ((1 + 2 * Complex.I) / (3 - Complex.I)) = 7 / 10 := by
  sorry

end imaginary_part_of_z_l2008_200826


namespace cos_squared_minus_sin_squared_15_deg_l2008_200817

theorem cos_squared_minus_sin_squared_15_deg :
  Real.cos (15 * Real.pi / 180) ^ 2 - Real.sin (15 * Real.pi / 180) ^ 2 = Real.sqrt 3 / 2 := by
  sorry

end cos_squared_minus_sin_squared_15_deg_l2008_200817


namespace maggie_candy_count_l2008_200833

/-- Given the Halloween candy collection scenario, prove that Maggie collected 50 pieces of candy. -/
theorem maggie_candy_count :
  -- Harper collected 30% more candy than Maggie
  ∀ (maggie harper : ℕ), harper = (13 * maggie) / 10 →
  -- Neil collected 40% more candy than Harper
  ∀ (neil : ℕ), neil = (14 * harper) / 10 →
  -- Neil got 91 pieces of candy
  neil = 91 →
  -- Maggie collected 50 pieces of candy
  maggie = 50 := by
sorry

end maggie_candy_count_l2008_200833


namespace odd_sum_of_cubes_not_both_even_l2008_200889

theorem odd_sum_of_cubes_not_both_even (n m : ℤ) 
  (h : Odd (n^3 + m^3)) : ¬(Even n ∧ Even m) := by
  sorry

end odd_sum_of_cubes_not_both_even_l2008_200889


namespace equation_solution_l2008_200846

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 1 →
  ((3 * x + 6) / (x^2 + 6 * x - 7) = (3 - x) / (x - 1)) ↔ (x = -5 ∨ x = 3) :=
by sorry

end equation_solution_l2008_200846


namespace athletes_total_yards_l2008_200865

-- Define the athletes and their performances
def malik_yards_per_game : ℕ := 18
def malik_games : ℕ := 5

def josiah_yards_per_game : ℕ := 22
def josiah_games : ℕ := 7

def darnell_yards_per_game : ℕ := 11
def darnell_games : ℕ := 4

def kade_yards_per_game : ℕ := 15
def kade_games : ℕ := 6

-- Define the total yards function
def total_yards : ℕ := 
  malik_yards_per_game * malik_games + 
  josiah_yards_per_game * josiah_games + 
  darnell_yards_per_game * darnell_games + 
  kade_yards_per_game * kade_games

-- Theorem statement
theorem athletes_total_yards : total_yards = 378 := by
  sorry

end athletes_total_yards_l2008_200865


namespace annie_extracurricular_hours_l2008_200880

def hours_before_midterms (
  chess_hours_per_week : ℕ)
  (drama_hours_per_week : ℕ)
  (glee_hours_odd_week : ℕ)
  (robotics_hours_even_week : ℕ)
  (soccer_hours_odd_week : ℕ)
  (soccer_hours_even_week : ℕ)
  (weeks_in_semester : ℕ)
  (sick_weeks : ℕ)
  (midterm_week : ℕ)
  (drama_cancel_week : ℕ)
  (holiday_week : ℕ)
  (holiday_soccer_hours : ℕ) : ℕ :=
  -- Function body
  sorry

theorem annie_extracurricular_hours :
  hours_before_midterms 2 8 3 4 1 2 12 2 8 5 7 1 = 81 :=
  sorry

end annie_extracurricular_hours_l2008_200880


namespace rancher_problem_l2008_200815

theorem rancher_problem :
  ∃! (b h : ℕ), b > 0 ∧ h > 0 ∧ 30 * b + 32 * h = 1200 ∧ b > h := by
  sorry

end rancher_problem_l2008_200815


namespace book_ratio_is_four_to_one_l2008_200845

/-- The number of books Zig wrote -/
def zig_books : ℕ := 60

/-- The total number of books Zig and Flo wrote together -/
def total_books : ℕ := 75

/-- The number of books Flo wrote -/
def flo_books : ℕ := total_books - zig_books

/-- The ratio of books written by Zig to books written by Flo -/
def book_ratio : ℚ := zig_books / flo_books

theorem book_ratio_is_four_to_one :
  book_ratio = 4 / 1 := by sorry

end book_ratio_is_four_to_one_l2008_200845


namespace parallelogram_area_is_41_l2008_200856

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (v w : ℝ × ℝ) : ℝ :=
  |v.1 * w.2 - v.2 * w.1|

/-- Given vectors and their components -/
def v : ℝ × ℝ := (8, -5)
def w : ℝ × ℝ := (13, -3)

/-- Theorem: The area of the parallelogram formed by v and w is 41 -/
theorem parallelogram_area_is_41 : parallelogramArea v w = 41 := by
  sorry

end parallelogram_area_is_41_l2008_200856


namespace solve_system_l2008_200830

theorem solve_system (x y : ℤ) (h1 : x + y = 300) (h2 : x - y = 200) : y = 50 := by
  sorry

end solve_system_l2008_200830


namespace somu_age_problem_l2008_200819

/-- Represents the problem of finding when Somu was one-fifth of his father's age --/
theorem somu_age_problem (somu_age : ℕ) (father_age : ℕ) (years_ago : ℕ) : 
  somu_age = 20 →
  3 * somu_age = father_age →
  5 * (somu_age - years_ago) = father_age - years_ago →
  years_ago = 10 := by
  sorry

#check somu_age_problem

end somu_age_problem_l2008_200819


namespace car_speed_first_hour_l2008_200802

/-- Proves that given specific conditions, the speed of a car in the first hour is 10 km/h -/
theorem car_speed_first_hour 
  (total_time : ℝ) 
  (second_hour_speed : ℝ) 
  (average_speed : ℝ) 
  (h1 : total_time = 2)
  (h2 : second_hour_speed = 60)
  (h3 : average_speed = 35) : 
  ∃ (first_hour_speed : ℝ), first_hour_speed = 10 ∧ 
    average_speed = (first_hour_speed + second_hour_speed) / total_time :=
by
  sorry

end car_speed_first_hour_l2008_200802


namespace simple_interest_problem_l2008_200879

/-- Proves that given a simple interest of 4052.25, an annual interest rate of 9%,
    and a time period of 5 years, the principal sum is 9005. -/
theorem simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) 
    (h1 : interest = 4052.25)
    (h2 : rate = 9)
    (h3 : time = 5)
    (h4 : principal = interest / (rate * time / 100)) :
  principal = 9005 := by
  sorry

end simple_interest_problem_l2008_200879


namespace smallest_three_digit_congruence_l2008_200851

theorem smallest_three_digit_congruence :
  ∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    (75 * n) % 345 = 225 ∧
    (∀ m : ℕ, 100 ≤ m ∧ m < n → (75 * m) % 345 ≠ 225) ∧
    n = 118 :=
by sorry

end smallest_three_digit_congruence_l2008_200851


namespace coffee_savings_l2008_200825

/-- Calculates the savings in daily coffee expenditure after a price increase and consumption reduction -/
theorem coffee_savings (original_coffees : ℕ) (original_price : ℚ) (price_increase : ℚ) : 
  let new_price := original_price * (1 + price_increase)
  let new_coffees := original_coffees / 2
  let original_spending := original_coffees * original_price
  let new_spending := new_coffees * new_price
  original_spending - new_spending = 2 :=
by
  sorry

#check coffee_savings 4 2 (1/2)

end coffee_savings_l2008_200825


namespace closest_perfect_square_to_350_l2008_200863

theorem closest_perfect_square_to_350 :
  ∀ n : ℕ, n ≠ 19 → (n ^ 2 : ℝ) ≠ 361 → |350 - (19 ^ 2 : ℝ)| < |350 - (n ^ 2 : ℝ)| :=
by sorry

end closest_perfect_square_to_350_l2008_200863


namespace sqrt_x_plus_inv_sqrt_x_l2008_200887

theorem sqrt_x_plus_inv_sqrt_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52 := by
  sorry

end sqrt_x_plus_inv_sqrt_x_l2008_200887


namespace warship_path_safe_l2008_200801

/-- Represents the distance of the reefs from the island in nautical miles -/
def reef_distance : ℝ := 3.8

/-- Represents the distance the warship travels from A to C in nautical miles -/
def travel_distance : ℝ := 8

/-- Represents the angle at which the island is seen from point A (in degrees) -/
def angle_at_A : ℝ := 75

/-- Represents the angle at which the island is seen from point C (in degrees) -/
def angle_at_C : ℝ := 60

/-- Theorem stating that the warship's path is safe from the reefs -/
theorem warship_path_safe :
  ∃ (distance_to_island : ℝ),
    distance_to_island > reef_distance ∧
    distance_to_island = travel_distance * Real.sin ((angle_at_A - angle_at_C) / 2 * π / 180) :=
by sorry

end warship_path_safe_l2008_200801


namespace divisibility_inequality_l2008_200841

theorem divisibility_inequality (a b c d e f : ℕ) 
  (h_f_lt_a : f < a)
  (h_div_c : ∃ k : ℕ, a * b * d + 1 = k * c)
  (h_div_b : ∃ l : ℕ, a * c * e + 1 = l * b)
  (h_div_a : ∃ m : ℕ, b * c * f + 1 = m * a)
  (h_ineq : (d : ℚ) / c < 1 - (e : ℚ) / b) :
  (d : ℚ) / c < 1 - (f : ℚ) / a :=
by sorry

end divisibility_inequality_l2008_200841


namespace abs_f_properties_l2008_200834

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the absolute value function of f
def abs_f (x : ℝ) : ℝ := |f x|

-- Theorem stating the properties of |f(x)|
theorem abs_f_properties :
  (∀ x, abs_f f x ≥ 0) ∧ 
  (∀ x, f x ≥ 0 → abs_f f x = f x) ∧
  (∀ x, f x < 0 → abs_f f x = -f x) :=
by sorry

end abs_f_properties_l2008_200834


namespace four_weavers_four_days_l2008_200874

/-- The number of mats woven by a group of weavers over a period of days. -/
def mats_woven (weavers : ℕ) (days : ℕ) : ℚ :=
  (25 : ℚ) * weavers * days / (10 * 10)

/-- Theorem stating that 4 mat-weavers will weave 4 mats in 4 days given the rate
    at which 10 mat-weavers can weave 25 mats in 10 days. -/
theorem four_weavers_four_days :
  mats_woven 4 4 = 4 := by
  sorry

end four_weavers_four_days_l2008_200874


namespace staircase_steps_l2008_200842

theorem staircase_steps (x : ℤ) 
  (h1 : x % 2 = 1)
  (h2 : x % 3 = 2)
  (h3 : x % 4 = 3)
  (h4 : x % 5 = 4)
  (h5 : x % 6 = 5)
  (h6 : x % 7 = 0) :
  ∃ k : ℤ, x = 119 + 420 * k := by
sorry

end staircase_steps_l2008_200842


namespace nine_possible_values_for_E_l2008_200807

def is_digit (n : ℕ) : Prop := n < 10

theorem nine_possible_values_for_E :
  ∀ (A B C D E : ℕ),
    is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ is_digit E →
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E →
    A + B = E →
    (C + D = E ∨ C + D = E + 10) →
    ∃! (count : ℕ), count = 9 ∧ 
      ∃ (possible_E : Finset ℕ), 
        possible_E.card = count ∧
        (∀ e, e ∈ possible_E ↔ 
          ∃ (A' B' C' D' : ℕ),
            is_digit A' ∧ is_digit B' ∧ is_digit C' ∧ is_digit D' ∧ is_digit e ∧
            A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ A' ≠ e ∧
            B' ≠ C' ∧ B' ≠ D' ∧ B' ≠ e ∧
            C' ≠ D' ∧ C' ≠ e ∧
            D' ≠ e ∧
            A' + B' = e ∧
            (C' + D' = e ∨ C' + D' = e + 10)) :=
by
  sorry

end nine_possible_values_for_E_l2008_200807


namespace highest_score_can_be_less_than_15_l2008_200805

/-- Represents a team in the tournament -/
structure Team :=
  (score : ℕ)

/-- Represents the tournament -/
structure Tournament :=
  (teams : Finset Team)
  (num_teams : ℕ)
  (total_games : ℕ)
  (total_points : ℕ)

/-- The tournament satisfies the given conditions -/
def valid_tournament (t : Tournament) : Prop :=
  t.num_teams = 10 ∧
  t.total_games = t.num_teams * (t.num_teams - 1) / 2 ∧
  t.total_points = 3 * t.total_games ∧
  t.teams.card = t.num_teams

/-- The theorem to be proved -/
theorem highest_score_can_be_less_than_15 :
  ∃ (t : Tournament), valid_tournament t ∧
    (∀ team ∈ t.teams, team.score < 15) :=
  sorry

end highest_score_can_be_less_than_15_l2008_200805


namespace rationalize_sqrt_five_eighteenths_l2008_200888

theorem rationalize_sqrt_five_eighteenths : 
  Real.sqrt (5 / 18) = Real.sqrt 10 / 6 := by
  sorry

end rationalize_sqrt_five_eighteenths_l2008_200888


namespace min_value_of_expression_l2008_200875

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + 4*a^2 + a + 1) * (b^3 + 4*b^2 + b + 1) * (c^3 + 4*c^2 + c + 1) / (a * b * c) ≥ 512 ∧
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    (x^3 + 4*x^2 + x + 1) * (y^3 + 4*y^2 + y + 1) * (z^3 + 4*z^2 + z + 1) / (x * y * z) = 512) :=
by sorry

end min_value_of_expression_l2008_200875


namespace expense_difference_l2008_200816

theorem expense_difference (alice_paid bob_paid carol_paid : ℕ) 
  (h_alice : alice_paid = 120)
  (h_bob : bob_paid = 150)
  (h_carol : carol_paid = 210) : 
  let total := alice_paid + bob_paid + carol_paid
  let each_share := total / 3
  let alice_owes := each_share - alice_paid
  let bob_owes := each_share - bob_paid
  alice_owes - bob_owes = 30 := by
  sorry

end expense_difference_l2008_200816


namespace median_is_82_l2008_200808

/-- Represents the list where each integer n (1 ≤ n ≤ 100) appears 2n times -/
def special_list : List ℕ := sorry

/-- The total number of elements in the special list -/
def total_elements : ℕ := sorry

/-- The median of the special list -/
def median_of_special_list : ℚ := sorry

/-- Theorem stating that the median of the special list is 82 -/
theorem median_is_82 : median_of_special_list = 82 := by sorry

end median_is_82_l2008_200808


namespace two_and_one_third_of_eighteen_is_fortytwo_l2008_200858

theorem two_and_one_third_of_eighteen_is_fortytwo : 
  (7 : ℚ) / 3 * 18 = 42 := by sorry

end two_and_one_third_of_eighteen_is_fortytwo_l2008_200858


namespace decagon_angle_property_l2008_200886

theorem decagon_angle_property (n : ℕ) : 
  (n - 2) * 180 = 360 * 4 ↔ n = 10 := by sorry

end decagon_angle_property_l2008_200886


namespace x_power_ln_ln_minus_ln_x_power_ln_l2008_200869

theorem x_power_ln_ln_minus_ln_x_power_ln (x : ℝ) (h : x > 1) :
  x^(Real.log (Real.log x)) - (Real.log x)^(Real.log x) = 0 := by
  sorry

end x_power_ln_ln_minus_ln_x_power_ln_l2008_200869


namespace length_of_PC_l2008_200813

-- Define the triangle ABC and point P
structure Triangle :=
  (A B C P : ℝ × ℝ)

-- Define the conditions
def is_right_triangle_with_internal_right_angle (t : Triangle) : Prop :=
  -- Right angle at B
  (t.A.1 - t.B.1) * (t.C.1 - t.B.1) + (t.A.2 - t.B.2) * (t.C.2 - t.B.2) = 0 ∧
  -- ∠BPC = 90°
  (t.B.1 - t.P.1) * (t.C.1 - t.P.1) + (t.B.2 - t.P.2) * (t.C.2 - t.P.2) = 0

def satisfies_length_conditions (t : Triangle) : Prop :=
  -- PA = 12
  Real.sqrt ((t.A.1 - t.P.1)^2 + (t.A.2 - t.P.2)^2) = 12 ∧
  -- PB = 8
  Real.sqrt ((t.B.1 - t.P.1)^2 + (t.B.2 - t.P.2)^2) = 8

-- The theorem to prove
theorem length_of_PC (t : Triangle) 
  (h1 : is_right_triangle_with_internal_right_angle t)
  (h2 : satisfies_length_conditions t) :
  Real.sqrt ((t.C.1 - t.P.1)^2 + (t.C.2 - t.P.2)^2) = Real.sqrt 464 :=
sorry

end length_of_PC_l2008_200813


namespace complex_number_existence_l2008_200894

theorem complex_number_existence : ∃ (z : ℂ), 
  Complex.abs z = Real.sqrt 7 ∧ 
  z.re < 0 ∧ 
  z.im > 0 := by
  sorry

end complex_number_existence_l2008_200894


namespace x_positive_necessary_not_sufficient_l2008_200821

theorem x_positive_necessary_not_sufficient :
  (∀ x : ℝ, (x - 2) * (x - 4) < 0 → x > 0) ∧
  (∃ x : ℝ, x > 0 ∧ (x - 2) * (x - 4) ≥ 0) := by
  sorry

end x_positive_necessary_not_sufficient_l2008_200821


namespace sqrt_equation_solutions_l2008_200883

theorem sqrt_equation_solutions (x : ℝ) :
  Real.sqrt ((2 + Real.sqrt 5) ^ x) + Real.sqrt ((2 - Real.sqrt 5) ^ x) = 6 ↔ x = 2 ∨ x = -2 := by
  sorry

end sqrt_equation_solutions_l2008_200883


namespace xiaogangMathScore_l2008_200803

theorem xiaogangMathScore (chineseScore englishScore averageScore : ℕ) (mathScore : ℕ) :
  chineseScore = 88 →
  englishScore = 91 →
  averageScore = 90 →
  (chineseScore + mathScore + englishScore) / 3 = averageScore →
  mathScore = 91 := by
  sorry

end xiaogangMathScore_l2008_200803


namespace yellow_second_draw_probability_l2008_200899

/-- Represents the total number of ping-pong balls -/
def total_balls : ℕ := 10

/-- Represents the number of yellow balls -/
def yellow_balls : ℕ := 6

/-- Represents the number of white balls -/
def white_balls : ℕ := 4

/-- Represents the number of draws -/
def num_draws : ℕ := 2

/-- Calculates the probability of drawing a yellow ball on the second draw -/
def prob_yellow_second_draw : ℚ :=
  (white_balls : ℚ) / total_balls * yellow_balls / (total_balls - 1)

theorem yellow_second_draw_probability :
  prob_yellow_second_draw = 4 / 15 := by sorry

end yellow_second_draw_probability_l2008_200899


namespace salary_comparison_l2008_200806

theorem salary_comparison (a b : ℝ) (h : a = 0.8 * b) :
  b = 1.25 * a := by sorry

end salary_comparison_l2008_200806


namespace square_not_prime_plus_square_l2008_200840

theorem square_not_prime_plus_square (n : ℕ) (h1 : n ≥ 5) (h2 : n % 3 = 2) :
  ¬ ∃ (p k : ℕ), Prime p ∧ n^2 = p + k^2 := by
sorry

end square_not_prime_plus_square_l2008_200840


namespace quadratic_factorization_l2008_200836

theorem quadratic_factorization :
  ∀ x : ℝ, 2 * x^2 + 4 * x - 6 = 2 * (x - 1) * (x + 3) := by
  sorry

end quadratic_factorization_l2008_200836


namespace circle_condition_l2008_200882

/-- The equation of a potential circle -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + x + 2*m*y + m^2 + m - 1 = 0

/-- Theorem stating the condition for the equation to represent a circle -/
theorem circle_condition (m : ℝ) :
  (∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y m ↔ (x - h)^2 + (y - k)^2 = r^2) ↔
  m < 5/4 := by
  sorry

end circle_condition_l2008_200882


namespace special_geometric_sequence_q_values_l2008_200837

/-- A geometric sequence with special properties -/
structure SpecialGeometricSequence where
  a : ℕ+ → ℕ+
  q : ℕ+
  first_term : a 1 = 2^81
  geometric : ∀ n : ℕ+, a (n + 1) = a n * q
  product_closure : ∀ m n : ℕ+, ∃ p : ℕ+, a m * a n = a p

/-- The set of all possible values for the common ratio q -/
def possible_q_values : Set ℕ+ :=
  {2^81, 2^27, 2^9, 2^3, 2}

/-- Main theorem: The set of all possible values of q for a SpecialGeometricSequence -/
theorem special_geometric_sequence_q_values (seq : SpecialGeometricSequence) :
  seq.q ∈ possible_q_values := by
  sorry

end special_geometric_sequence_q_values_l2008_200837


namespace therapy_hours_calculation_l2008_200814

/-- Represents the pricing structure and patient charges for a psychologist's therapy sessions -/
structure TherapyPricing where
  first_hour : ℕ  -- Cost of the first hour
  additional_hour : ℕ  -- Cost of each additional hour
  first_patient_total : ℕ  -- Total charge for the first patient
  second_patient_total : ℕ  -- Total charge for the second patient (3 hours)

/-- Calculates the number of therapy hours for the first patient given the pricing structure -/
def calculate_hours (pricing : TherapyPricing) : ℕ :=
  sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem therapy_hours_calculation (pricing : TherapyPricing) 
  (h1 : pricing.first_hour = pricing.additional_hour + 20)
  (h2 : pricing.second_patient_total = pricing.first_hour + 2 * pricing.additional_hour)
  (h3 : pricing.second_patient_total = 188)
  (h4 : pricing.first_patient_total = 300) :
  calculate_hours pricing = 5 :=
sorry

end therapy_hours_calculation_l2008_200814


namespace one_face_colored_cubes_125_l2008_200818

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  edge_divisions : ℕ
  total_small_cubes : ℕ
  colored_faces : ℕ

/-- The number of small cubes with exactly one colored face -/
def one_face_colored_cubes (c : CutCube) : ℕ :=
  c.colored_faces * (c.edge_divisions - 2) ^ 2

/-- Theorem stating the number of cubes with one colored face for a specific case -/
theorem one_face_colored_cubes_125 :
  ∀ c : CutCube,
    c.edge_divisions = 5 →
    c.total_small_cubes = 125 →
    c.colored_faces = 6 →
    one_face_colored_cubes c = 54 := by
  sorry

end one_face_colored_cubes_125_l2008_200818


namespace max_xy_value_l2008_200881

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 5 * y = 200) : x * y ≤ 285 := by
  sorry

end max_xy_value_l2008_200881


namespace points_on_line_l2008_200829

theorem points_on_line (t : ℝ) :
  let x := Real.sin t ^ 2
  let y := Real.cos t ^ 2
  x + y = 1 := by
sorry

end points_on_line_l2008_200829


namespace levi_additional_baskets_l2008_200895

/-- Calculates the number of additional baskets Levi needs to score to beat his brother by at least the given margin. -/
def additional_baskets_needed (levi_initial : ℕ) (brother_initial : ℕ) (brother_additional : ℕ) (margin : ℕ) : ℕ :=
  (brother_initial + brother_additional + margin) - levi_initial

/-- Proves that Levi needs to score 12 more times to beat his brother by at least 5 baskets. -/
theorem levi_additional_baskets : 
  additional_baskets_needed 8 12 3 5 = 12 := by
  sorry

end levi_additional_baskets_l2008_200895


namespace line_and_circle_problem_l2008_200811

/-- Line l: x - y + m = 0 -/
def line_l (m : ℝ) (x y : ℝ) : Prop := x - y + m = 0

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point is on a line -/
def point_on_line (p : Point) (m : ℝ) : Prop := line_l m p.x p.y

/-- Rotate a line by 90 degrees counterclockwise around its x-axis intersection -/
def rotate_line (m : ℝ) (x y : ℝ) : Prop := y + x + m = 0

/-- Circle equation -/
def circle_equation (center : Point) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.x)^2 + (y - center.y)^2 = radius^2

theorem line_and_circle_problem (m : ℝ) :
  (∃ (x y : ℝ), rotate_line m x y ∧ x = 2 ∧ y = -3) →
  (∃ (center : Point) (radius : ℝ),
    point_on_line center m ∧
    circle_equation center radius 1 1 ∧
    circle_equation center radius 2 (-2)) →
  m = 1 ∧
  (∃ (center : Point),
    point_on_line center 1 ∧
    circle_equation center 5 1 1 ∧
    circle_equation center 5 2 (-2) ∧
    center.x = -3 ∧
    center.y = -2) := by
  sorry

end line_and_circle_problem_l2008_200811


namespace smallest_solution_congruence_l2008_200820

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 14 % 31 ∧
  ∀ (y : ℕ), y > 0 ∧ (5 * y) % 31 = 14 % 31 → x ≤ y :=
by sorry

end smallest_solution_congruence_l2008_200820


namespace angle_sum_around_point_l2008_200890

theorem angle_sum_around_point (x : ℝ) : 
  150 + 90 + x + 90 = 360 → x = 30 := by
  sorry

end angle_sum_around_point_l2008_200890


namespace sum_of_coefficients_zero_l2008_200831

theorem sum_of_coefficients_zero 
  (a b c d : ℝ) 
  (h : ∀ x : ℝ, (1 + x)^2 * (1 - x) = a + b*x + c*x^2 + d*x^3) : 
  a + b + c + d = 0 := by
sorry

end sum_of_coefficients_zero_l2008_200831


namespace lcm_of_20_45_36_l2008_200893

theorem lcm_of_20_45_36 : Nat.lcm (Nat.lcm 20 45) 36 = 180 := by sorry

end lcm_of_20_45_36_l2008_200893


namespace charity_event_probability_l2008_200870

/-- The probability of selecting a boy for Saturday and a girl for Sunday
    from a group of 2 boys and 2 girls for a two-day event. -/
theorem charity_event_probability :
  let total_people : ℕ := 2 + 2  -- 2 boys + 2 girls
  let total_combinations : ℕ := total_people * (total_people - 1)
  let favorable_outcomes : ℕ := 2 * 2  -- 2 boys for Saturday * 2 girls for Sunday
  (favorable_outcomes : ℚ) / total_combinations = 1 / 3 :=
by sorry

end charity_event_probability_l2008_200870


namespace correct_division_l2008_200854

theorem correct_division (dividend : ℕ) (incorrect_divisor correct_divisor incorrect_quotient : ℕ) 
  (h1 : incorrect_divisor = 72)
  (h2 : correct_divisor = 36)
  (h3 : incorrect_quotient = 24)
  (h4 : dividend = incorrect_divisor * incorrect_quotient) :
  dividend / correct_divisor = 48 := by
  sorry

end correct_division_l2008_200854


namespace garden_fence_length_l2008_200898

/-- The length of a fence surrounding a square garden -/
def fence_length (side_length : ℝ) : ℝ := 4 * side_length

/-- Theorem: The length of the fence surrounding a square garden with side length 28 meters is 112 meters -/
theorem garden_fence_length :
  fence_length 28 = 112 := by
  sorry

end garden_fence_length_l2008_200898


namespace digit2012_is_zero_l2008_200822

/-- The sequence of digits obtained by writing positive integers in order -/
def digitSequence : ℕ → ℕ :=
  sorry

/-- The 2012th digit in the sequence -/
def digit2012 : ℕ := digitSequence 2012

theorem digit2012_is_zero : digit2012 = 0 := by
  sorry

end digit2012_is_zero_l2008_200822


namespace last_digit_of_max_value_l2008_200867

/-- Represents the operation of replacing two numbers with their product plus one -/
def combine (a b : ℕ) : ℕ := a * b + 1

/-- The maximum value after performing the combine operation 127 times on 128 ones -/
def max_final_value : ℕ := sorry

/-- The problem statement -/
theorem last_digit_of_max_value :
  (max_final_value % 10) = 2 := by sorry

end last_digit_of_max_value_l2008_200867


namespace star_emilio_sum_difference_l2008_200853

def star_list : List Nat := List.range 50 |>.map (· + 1)

def emilio_transform (n : Nat) : Nat :=
  let s := toString n
  let s' := s.replace "2" "1" |>.replace "3" "2"
  s'.toNat!

def emilio_list : List Nat := star_list.map emilio_transform

theorem star_emilio_sum_difference : 
  star_list.sum - emilio_list.sum = 210 := by
  sorry

end star_emilio_sum_difference_l2008_200853


namespace square_sum_inequality_l2008_200828

theorem square_sum_inequality (a b c : ℝ) : 
  a^2 + b^2 + a*b + b*c + c*a < 0 → a^2 + b^2 < c^2 := by
  sorry

end square_sum_inequality_l2008_200828


namespace range_of_t_l2008_200809

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≥ -1}
def B (t : ℝ) : Set ℝ := {y : ℝ | y ≥ t}

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem range_of_t (t : ℝ) :
  (∀ x ∈ A, f x ∈ B t) → t ≤ 0 := by
  sorry

-- Define the final result
def result : Set ℝ := {t : ℝ | t ≤ 0}

end range_of_t_l2008_200809


namespace sin_cos_power_sum_l2008_200800

theorem sin_cos_power_sum (x : ℝ) (h : 3 * Real.sin x ^ 3 + Real.cos x ^ 3 = 3) :
  Real.sin x ^ 2018 + Real.cos x ^ 2018 = 1 := by
  sorry

end sin_cos_power_sum_l2008_200800


namespace expand_expression_l2008_200850

theorem expand_expression (a b : ℝ) : (a - 2) * (a - 2*b) = a^2 - 2*a + 4*b := by
  sorry

end expand_expression_l2008_200850


namespace max_x_value_l2008_200810

theorem max_x_value (x y z : ℝ) 
  (sum_eq : x + y + z = 7) 
  (sum_prod_eq : x * y + x * z + y * z = 11) : 
  x ≤ (7 + Real.sqrt 34) / 3 := by
  sorry

end max_x_value_l2008_200810


namespace max_value_sqrt_sum_l2008_200823

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_cond : x + y + z = 2)
  (x_bound : x ≥ -1/2)
  (y_bound : y ≥ -2)
  (z_bound : z ≥ -3)
  (xy_cond : 2*x + y = 1) :
  ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ + y₀ + z₀ = 2 ∧ 
    2*x₀ + y₀ = 1 ∧
    x₀ ≥ -1/2 ∧ 
    y₀ ≥ -2 ∧ 
    z₀ ≥ -3 ∧
    ∀ x y z, 
      x + y + z = 2 → 
      2*x + y = 1 → 
      x ≥ -1/2 → 
      y ≥ -2 → 
      z ≥ -3 →
      Real.sqrt (4*x + 2) + Real.sqrt (3*y + 6) + Real.sqrt (4*z + 12) ≤ 
      Real.sqrt (4*x₀ + 2) + Real.sqrt (3*y₀ + 6) + Real.sqrt (4*z₀ + 12) ∧
      Real.sqrt (4*x₀ + 2) + Real.sqrt (3*y₀ + 6) + Real.sqrt (4*z₀ + 12) = Real.sqrt 68 :=
by
  sorry

end max_value_sqrt_sum_l2008_200823


namespace distance_between_vertices_l2008_200848

/-- The distance between the vertices of two quadratic functions -/
theorem distance_between_vertices (a b c d e f : ℝ) : 
  let f1 := fun x : ℝ => x^2 + a*x + b
  let f2 := fun x : ℝ => x^2 + c*x + d
  let vertex1 := (-a/2, f1 (-a/2))
  let vertex2 := (-c/2, f2 (-c/2))
  (a = -4 ∧ b = 5 ∧ c = 6 ∧ d = 13) →
  Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = Real.sqrt 34 :=
by sorry

end distance_between_vertices_l2008_200848


namespace triangle_angle_sine_identity_l2008_200804

theorem triangle_angle_sine_identity 
  (A B C : ℝ) (n : ℤ) 
  (h_triangle : A + B + C = Real.pi) 
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0) :
  Real.sin (2 * n * A) + Real.sin (2 * n * B) + Real.sin (2 * n * C) = 
  (-1)^(n+1) * 4 * Real.sin (n * A) * Real.sin (n * B) * Real.sin (n * C) := by
  sorry

end triangle_angle_sine_identity_l2008_200804


namespace cricket_game_overs_l2008_200857

theorem cricket_game_overs (total_target : ℝ) (initial_rate : ℝ) (remaining_overs : ℝ) (required_rate : ℝ) 
  (h1 : total_target = 282)
  (h2 : initial_rate = 3.6)
  (h3 : remaining_overs = 40)
  (h4 : required_rate = 6.15) :
  ∃ (initial_overs : ℝ), 
    initial_overs * initial_rate + remaining_overs * required_rate = total_target ∧ 
    initial_overs = 10 := by
  sorry

end cricket_game_overs_l2008_200857


namespace percentage_of_three_digit_numbers_with_repeated_digit_l2008_200812

theorem percentage_of_three_digit_numbers_with_repeated_digit : 
  let total_three_digit_numbers : ℕ := 900
  let three_digit_numbers_without_repeat : ℕ := 9 * 9 * 8
  let three_digit_numbers_with_repeat : ℕ := total_three_digit_numbers - three_digit_numbers_without_repeat
  let percentage : ℚ := three_digit_numbers_with_repeat / total_three_digit_numbers
  ⌊percentage * 1000 + 5⌋ / 10 = 28 :=
by sorry

end percentage_of_three_digit_numbers_with_repeated_digit_l2008_200812


namespace sum_of_sequence_equals_63_over_19_l2008_200839

def A : ℕ → ℚ
  | 0 => 2
  | 1 => 3
  | (n + 2) => 2 * A (n + 1) + A n

theorem sum_of_sequence_equals_63_over_19 :
  ∑' n, A n / 5^n = 63 / 19 := by sorry

end sum_of_sequence_equals_63_over_19_l2008_200839


namespace inequality_solution_l2008_200868

theorem inequality_solution (x : ℝ) : 
  (x - 2) * (x - 3) * (x - 4) / ((x - 1) * (x - 5) * (x - 6)) > 0 ↔ 
  x < 1 ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ x > 6 :=
by sorry

end inequality_solution_l2008_200868


namespace rectangular_box_volume_l2008_200835

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 36)
  (area2 : w * h = 18)
  (area3 : l * h = 12) :
  l * w * h = 36 * Real.sqrt 6 := by
  sorry

end rectangular_box_volume_l2008_200835


namespace friend_game_l2008_200891

theorem friend_game (a b c d : ℕ) : 
  3^a * 7^b = 3 * 7 ∧ 3^c * 7^d = 3 * 7 → (a - 1) * (d - 1) = (b - 1) * (c - 1) :=
by sorry

end friend_game_l2008_200891


namespace number_of_children_l2008_200872

theorem number_of_children : ∃ n : ℕ, 
  (∃ b : ℕ, b = 3 * n + 4) ∧ 
  (∃ b : ℕ, b = 4 * n - 3) ∧ 
  n = 7 := by
  sorry

end number_of_children_l2008_200872


namespace cook_selection_ways_l2008_200843

theorem cook_selection_ways (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 2) :
  Nat.choose n k = 28 := by
  sorry

end cook_selection_ways_l2008_200843


namespace convex_polygon_20_sides_diagonals_l2008_200884

/-- A convex polygon is a polygon in which every interior angle is less than 180 degrees. -/
def ConvexPolygon (n : ℕ) : Prop := sorry

/-- A diagonal of a convex polygon is a line segment that connects two non-adjacent vertices. -/
def Diagonal (n : ℕ) : Prop := sorry

/-- The number of diagonals in a convex polygon with n sides. -/
def NumDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem convex_polygon_20_sides_diagonals :
  ∀ p : ConvexPolygon 20, NumDiagonals 20 = 170 := by sorry

end convex_polygon_20_sides_diagonals_l2008_200884


namespace tablet_charge_time_proof_l2008_200864

/-- Time in minutes to fully charge a smartphone -/
def smartphone_charge_time : ℕ := 26

/-- Time in minutes to fully charge a tablet -/
def tablet_charge_time : ℕ := 53

/-- The total time taken to charge half a smartphone and a full tablet -/
def total_charge_time : ℕ := 66

/-- Theorem stating that the time to fully charge a tablet is 53 minutes -/
theorem tablet_charge_time_proof :
  tablet_charge_time = total_charge_time - (smartphone_charge_time / 2) :=
by sorry

end tablet_charge_time_proof_l2008_200864


namespace video_game_points_l2008_200824

/-- The number of points earned in a video game level --/
def points_earned (total_enemies : ℕ) (enemies_left : ℕ) (points_per_enemy : ℕ) : ℕ :=
  (total_enemies - enemies_left) * points_per_enemy

/-- Theorem: In the given scenario, the player earns 40 points --/
theorem video_game_points : points_earned 7 2 8 = 40 := by
  sorry

end video_game_points_l2008_200824


namespace cuboid_volume_l2008_200885

/-- Given a cuboid with perimeters of opposite faces A, B, and C, prove its volume is 240 cubic centimeters -/
theorem cuboid_volume (A B C : ℝ) (hA : A = 20) (hB : B = 32) (hC : C = 28) :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  2 * (x + y) = A ∧
  2 * (y + z) = B ∧
  2 * (x + z) = C ∧
  x * y * z = 240 := by
  sorry

end cuboid_volume_l2008_200885


namespace sin_two_phi_l2008_200849

theorem sin_two_phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_two_phi_l2008_200849


namespace next_perfect_cube_l2008_200892

/-- Given a perfect cube x, the next larger perfect cube is x + 3(∛x)² + 3∛x + 1 -/
theorem next_perfect_cube (x : ℕ) (h : ∃ k : ℕ, x = k^3) :
  ∃ y : ℕ, y > x ∧ (∃ m : ℕ, y = m^3) ∧ y = x + 3 * (x^(1/3))^2 + 3 * x^(1/3) + 1 :=
sorry

end next_perfect_cube_l2008_200892


namespace water_depth_relation_l2008_200862

/-- Represents a cylindrical water tank -/
structure WaterTank where
  height : ℝ
  baseDiameter : ℝ

/-- Calculates the water depth when the tank is upright -/
def uprightDepth (tank : WaterTank) (horizontalDepth : ℝ) : ℝ :=
  sorry

/-- Theorem stating the relation between horizontal and upright water depths -/
theorem water_depth_relation (tank : WaterTank) (horizontalDepth : ℝ) :
  tank.height = 20 →
  tank.baseDiameter = 5 →
  horizontalDepth = 4 →
  abs (uprightDepth tank horizontalDepth - 8.1) < 0.1 := by
  sorry

end water_depth_relation_l2008_200862


namespace park_maple_trees_l2008_200877

/-- The number of maple trees in the park after planting -/
def total_maple_trees (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem: The park will have 11 maple trees after planting -/
theorem park_maple_trees :
  let initial_trees : ℕ := 2
  let trees_to_plant : ℕ := 9
  total_maple_trees initial_trees trees_to_plant = 11 := by
  sorry

end park_maple_trees_l2008_200877


namespace product_sign_l2008_200871

theorem product_sign (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x^4 - y^4 > x) (h2 : y^4 - x^4 > y) : x * y > 0 := by
  sorry

end product_sign_l2008_200871


namespace polynomial_value_at_three_l2008_200860

theorem polynomial_value_at_three : 
  let x : ℝ := 3
  (x^5 + 5*x^3 + 2*x) = 384 := by
  sorry

end polynomial_value_at_three_l2008_200860


namespace product_of_three_terms_l2008_200859

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem product_of_three_terms 
  (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : a 5 = 4) : 
  a 4 * a 5 * a 6 = 64 := by
sorry

end product_of_three_terms_l2008_200859


namespace restaurant_students_l2008_200847

theorem restaurant_students (burger_orders : ℕ) (hotdog_orders : ℕ) : 
  burger_orders = 30 →
  burger_orders = 2 * hotdog_orders →
  burger_orders + hotdog_orders = 45 := by
sorry

end restaurant_students_l2008_200847


namespace jacks_initial_dollars_l2008_200832

theorem jacks_initial_dollars (x : ℕ) : 
  x + 36 * 2 = 117 → x = 45 := by sorry

end jacks_initial_dollars_l2008_200832


namespace size_comparison_l2008_200896

-- Define a rectangular parallelepiped
structure RectParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0

-- Define the size of a rectangular parallelepiped
def size (p : RectParallelepiped) : ℝ :=
  p.length + p.width + p.height

-- Define the "fits inside" relation
def fits_inside (p' p : RectParallelepiped) : Prop :=
  p'.length ≤ p.length ∧ p'.width ≤ p.width ∧ p'.height ≤ p.height

-- Theorem statement
theorem size_comparison (p p' : RectParallelepiped) (h : fits_inside p' p) :
  size p' ≤ size p := by
  sorry

end size_comparison_l2008_200896


namespace remainder_17_63_mod_7_l2008_200852

theorem remainder_17_63_mod_7 : 17^63 % 7 = 6 := by
  sorry

end remainder_17_63_mod_7_l2008_200852


namespace gcf_three_digit_palindromes_l2008_200876

def three_digit_palindrome (a b : ℕ) : ℕ := 100 * a + 10 * b + a

theorem gcf_three_digit_palindromes :
  ∃ (gcf : ℕ), 
    gcf > 0 ∧
    (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → gcf ∣ three_digit_palindrome a b) ∧
    (∀ (d : ℕ), d > 0 ∧ (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → d ∣ three_digit_palindrome a b) → d ≤ gcf) ∧
    gcf = 1 :=
by sorry

end gcf_three_digit_palindromes_l2008_200876


namespace board_highest_point_l2008_200873

/-- Represents a rectangular board with length and height -/
structure Board where
  length : ℝ
  height : ℝ

/-- Calculates the distance from the ground to the highest point of an inclined board -/
def highestPoint (board : Board) (angle : ℝ) : ℝ :=
  sorry

theorem board_highest_point :
  let board := Board.mk 64 4
  let angle := 30 * π / 180
  ∃ (a b c : ℕ), 
    (highestPoint board angle = a + b * Real.sqrt c) ∧
    (a = 32) ∧ (b = 2) ∧ (c = 3) ∧
    (∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ c)) :=
  sorry

end board_highest_point_l2008_200873


namespace number_problem_l2008_200838

theorem number_problem : ∃ n : ℝ, 8 * n - 4 = 17 ∧ n = 2.625 := by
  sorry

end number_problem_l2008_200838
