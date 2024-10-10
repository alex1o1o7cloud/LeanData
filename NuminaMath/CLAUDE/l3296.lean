import Mathlib

namespace unique_real_solution_l3296_329682

theorem unique_real_solution (b : ℝ) :
  ∀ a : ℝ, (∃! x : ℝ, x^3 - a*x^2 - (2*a + b)*x + a^2 + b = 0) ↔ a < 3 + 4*b :=
by sorry

end unique_real_solution_l3296_329682


namespace remainder_theorem_l3296_329665

theorem remainder_theorem (x y u v : ℕ) (hx : x > 0) (hy : y > 0) 
  (hu : u = x / y) (hv : v = x % y) (hv_bound : v < y) : 
  (x + 3 * u * y + y) % y = v := by
sorry

end remainder_theorem_l3296_329665


namespace dividend_calculation_l3296_329684

/-- Given a division problem with the following parameters:
    - x: Real number equal to 0.25
    - quotient: Function of x defined as (3/2)x - 2,175.4
    - divisor: Function of x defined as 20,147x² - 785
    - remainder: Function of x defined as (-1/4)x³ + 1,112.7
    
    This theorem states that the dividend, calculated as (divisor * quotient) + remainder,
    is approximately equal to -1,031,103.16 (rounded to two decimal places). -/
theorem dividend_calculation (x : ℝ) 
    (hx : x = 0.25)
    (quotient : ℝ → ℝ)
    (hquotient : quotient = fun y => (3/2)*y - 2175.4)
    (divisor : ℝ → ℝ)
    (hdivisor : divisor = fun y => 20147*y^2 - 785)
    (remainder : ℝ → ℝ)
    (hremainder : remainder = fun y => (-1/4)*y^3 + 1112.7)
    : ∃ ε > 0, |((divisor x) * (quotient x) + (remainder x)) - (-1031103.16)| < ε :=
sorry

end dividend_calculation_l3296_329684


namespace circle_proof_l3296_329691

-- Define the points A and B
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (3, -1)

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 16

-- Theorem statement
theorem circle_proof :
  ∃ (center : ℝ × ℝ),
    center_line center.1 center.2 ∧
    circle_equation A.1 A.2 ∧
    circle_equation B.1 B.2 :=
by sorry

end circle_proof_l3296_329691


namespace insurance_cost_calculation_l3296_329614

/-- Calculates the total annual insurance cost given quarterly, monthly, annual, and semi-annual payments -/
def total_annual_insurance_cost (car_quarterly : ℕ) (home_monthly : ℕ) (health_annual : ℕ) (life_semiannual : ℕ) : ℕ :=
  car_quarterly * 4 + home_monthly * 12 + health_annual + life_semiannual * 2

/-- Theorem stating that given specific insurance costs, the total annual cost is 8757 -/
theorem insurance_cost_calculation :
  total_annual_insurance_cost 378 125 5045 850 = 8757 := by
  sorry

end insurance_cost_calculation_l3296_329614


namespace room_dimensions_l3296_329626

theorem room_dimensions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_not_equal : b ≠ c) :
  let x := (b^2 * (b^2 - a^2) / (b^2 - c^2))^(1/4)
  let y := a / x
  let z := b / x
  let u := c * x / b
  ∃ (room_I room_II room_III : ℝ × ℝ),
    room_I.1 * room_I.2 = a ∧
    room_II.1 * room_II.2 = b ∧
    room_III.1 * room_III.2 = c ∧
    room_I.1 = room_II.1 ∧
    room_II.2 = room_III.2 ∧
    room_I.1^2 + room_I.2^2 = room_III.1^2 + room_III.2^2 ∧
    room_I = (x, y) ∧
    room_II = (x, z) ∧
    room_III = (u, z) := by
  sorry

#check room_dimensions

end room_dimensions_l3296_329626


namespace parallelogram_height_l3296_329680

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area base height : ℝ) : 
  area = base * height → area = 960 → base = 60 → height = 16 := by
  sorry

end parallelogram_height_l3296_329680


namespace coefficient_of_x_cubed_l3296_329676

def expression (x : ℝ) : ℝ := 
  3 * (x^2 - x^3 + x) + 3 * (x + 2*x^3 - 3*x^2 + 3*x^5 + x^3) - 5 * (1 + x - 4*x^3 - x^2)

theorem coefficient_of_x_cubed : 
  ∃ (a b c d e : ℝ), expression x = a*x^5 + b*x^4 + 26*x^3 + c*x^2 + d*x + e :=
by sorry

end coefficient_of_x_cubed_l3296_329676


namespace sum_of_min_values_is_zero_l3296_329632

-- Define the polynomials P and Q
def P (a b x : ℝ) : ℝ := x^2 + a*x + b
def Q (c d x : ℝ) : ℝ := x^2 + c*x + d

-- Define the composition of P and Q
def PQ (a b c d x : ℝ) : ℝ := P a b (Q c d x)
def QP (a b c d x : ℝ) : ℝ := Q c d (P a b x)

-- State the theorem
theorem sum_of_min_values_is_zero 
  (a b c d : ℝ) 
  (h1 : PQ a b c d 1 = 0)
  (h2 : PQ a b c d 3 = 0)
  (h3 : PQ a b c d 5 = 0)
  (h4 : PQ a b c d 7 = 0)
  (h5 : QP a b c d 2 = 0)
  (h6 : QP a b c d 6 = 0)
  (h7 : QP a b c d 10 = 0)
  (h8 : QP a b c d 14 = 0) :
  ∃ (x y : ℝ), P a b x + Q c d y = 0 ∧ 
  (∀ z, P a b z ≥ P a b x) ∧ 
  (∀ w, Q c d w ≥ Q c d y) :=
sorry

end sum_of_min_values_is_zero_l3296_329632


namespace regular_octagon_interior_angle_l3296_329679

/-- The measure of one interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let sum_interior_angles : ℝ := 180 * (n - 2)  -- sum of interior angles formula
  let one_interior_angle : ℝ := sum_interior_angles / n  -- measure of one interior angle
  135

/-- Proof of the theorem -/
lemma prove_regular_octagon_interior_angle : regular_octagon_interior_angle = 135 := by
  sorry

end regular_octagon_interior_angle_l3296_329679


namespace tiffany_max_points_l3296_329688

/-- Represents the game setup and Tiffany's current state -/
structure GameState where
  initialMoney : ℕ
  costPerGame : ℕ
  ringsPerPlay : ℕ
  redBucketPoints : ℕ
  greenBucketPoints : ℕ
  gamesPlayed : ℕ
  redBucketsHit : ℕ
  greenBucketsHit : ℕ

/-- Calculates the maximum points achievable given a GameState -/
def maxPoints (state : GameState) : ℕ :=
  let pointsFromRed := state.redBucketsHit * state.redBucketPoints
  let pointsFromGreen := state.greenBucketsHit * state.greenBucketPoints
  let moneySpent := state.gamesPlayed * state.costPerGame
  let moneyLeft := state.initialMoney - moneySpent
  let gamesLeft := moneyLeft / state.costPerGame
  let maxPointsLastGame := state.ringsPerPlay * max state.redBucketPoints state.greenBucketPoints
  pointsFromRed + pointsFromGreen + gamesLeft * maxPointsLastGame

/-- Tiffany's game state -/
def tiffanyState : GameState where
  initialMoney := 3
  costPerGame := 1
  ringsPerPlay := 5
  redBucketPoints := 2
  greenBucketPoints := 3
  gamesPlayed := 2
  redBucketsHit := 4
  greenBucketsHit := 5

/-- Theorem stating that the maximum points Tiffany can achieve is 38 -/
theorem tiffany_max_points :
  maxPoints tiffanyState = 38 := by
  sorry

end tiffany_max_points_l3296_329688


namespace sin_330_degrees_l3296_329611

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end sin_330_degrees_l3296_329611


namespace four_people_seven_chairs_two_occupied_l3296_329652

/-- The number of ways to arrange n distinct objects in r positions --/
def permutation (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - r)

/-- The number of ways four people can sit in a row of seven chairs
    where two specific chairs are always occupied --/
theorem four_people_seven_chairs_two_occupied : 
  permutation 5 4 = 120 := by
  sorry

end four_people_seven_chairs_two_occupied_l3296_329652


namespace roots_equation_value_l3296_329633

theorem roots_equation_value (α β : ℝ) : 
  (α^2 + 2*α - 1 = 0) → 
  (β^2 + 2*β - 1 = 0) → 
  α^2 + 3*α + β = -1 := by
sorry

end roots_equation_value_l3296_329633


namespace percent_of_x_l3296_329629

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 50 + x / 25) / x * 100 = 6 := by
  sorry

end percent_of_x_l3296_329629


namespace division_in_ratio_l3296_329635

theorem division_in_ratio (total : ℕ) (x_ratio y_ratio : ℕ) (x_amount : ℕ) : 
  total = 5000 → 
  x_ratio = 2 → 
  y_ratio = 8 → 
  x_amount = total * x_ratio / (x_ratio + y_ratio) → 
  x_amount = 1000 := by
sorry

end division_in_ratio_l3296_329635


namespace min_sum_mn_l3296_329638

theorem min_sum_mn (m n : ℕ+) (h : m.val * n.val - 2 * m.val - 3 * n.val - 20 = 0) :
  ∃ (p q : ℕ+), p.val * q.val - 2 * p.val - 3 * q.val - 20 = 0 ∧ 
  p.val + q.val = 20 ∧ 
  ∀ (x y : ℕ+), x.val * y.val - 2 * x.val - 3 * y.val - 20 = 0 → x.val + y.val ≥ 20 :=
by sorry

end min_sum_mn_l3296_329638


namespace equation_solution_l3296_329668

theorem equation_solution : ∃ x : ℚ, (3 * x + 5 * x = 800 - (4 * x + 6 * x)) ∧ x = 400 / 9 := by
  sorry

end equation_solution_l3296_329668


namespace polynomial_no_real_roots_l3296_329690

theorem polynomial_no_real_roots (a b c : ℝ) (h : |a| + |b| + |c| ≤ Real.sqrt 2) :
  ∀ x : ℝ, x^4 + a*x^3 + b*x^2 + c*x + 1 > 0 := by
sorry

end polynomial_no_real_roots_l3296_329690


namespace exists_problem_solved_by_half_not_all_l3296_329667

/-- Represents a jury member -/
structure JuryMember where
  id : Nat
  solved_problems : Finset Nat

/-- Represents the contest setup -/
structure ContestSetup where
  jury_members : Finset JuryMember
  total_problems : Nat
  problems_per_member : Nat

/-- Main theorem: There exists a problem solved by at least half but not all jury members -/
theorem exists_problem_solved_by_half_not_all (setup : ContestSetup)
  (h1 : setup.jury_members.card = 40)
  (h2 : setup.total_problems = 30)
  (h3 : setup.problems_per_member = 26)
  (h4 : ∀ m1 m2 : JuryMember, m1 ∈ setup.jury_members → m2 ∈ setup.jury_members → m1 ≠ m2 → m1.solved_problems ≠ m2.solved_problems) :
  ∃ p : Nat, p < setup.total_problems ∧ 
    (20 ≤ (setup.jury_members.filter (λ m => p ∈ m.solved_problems)).card) ∧
    ((setup.jury_members.filter (λ m => p ∈ m.solved_problems)).card < 40) := by
  sorry


end exists_problem_solved_by_half_not_all_l3296_329667


namespace inequality_solution_set_l3296_329617

theorem inequality_solution_set (x : ℝ) :
  (∀ y : ℝ, y > 0 → (4 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y)) / (x + y) > 3 * x^2 * y) ↔
  0 ≤ x ∧ x < 2 := by
sorry

end inequality_solution_set_l3296_329617


namespace singer_work_hours_l3296_329683

/-- Calculates the total hours taken to complete multiple songs given the daily work hours, days per song, and number of songs. -/
def totalHours (hoursPerDay : ℕ) (daysPerSong : ℕ) (numberOfSongs : ℕ) : ℕ :=
  hoursPerDay * daysPerSong * numberOfSongs

/-- Proves that a singer working 10 hours a day for 10 days on each of 3 songs will take 300 hours in total. -/
theorem singer_work_hours : totalHours 10 10 3 = 300 := by
  sorry

end singer_work_hours_l3296_329683


namespace equation_solution_l3296_329693

theorem equation_solution (n k l m : ℕ) : 
  l > 1 → 
  (1 + n^k)^l = 1 + n^m →
  n = 2 ∧ k = 1 ∧ l = 2 ∧ m = 3 :=
by sorry

end equation_solution_l3296_329693


namespace mary_sugar_already_added_l3296_329669

/-- Given a recipe that requires a total amount of sugar and the amount still needed to be added,
    calculate the amount of sugar already put in. -/
def sugar_already_added (total_required : ℕ) (still_needed : ℕ) : ℕ :=
  total_required - still_needed

/-- Theorem stating that given the specific values from the problem,
    the amount of sugar already added is 2 cups. -/
theorem mary_sugar_already_added :
  sugar_already_added 13 11 = 2 := by sorry

end mary_sugar_already_added_l3296_329669


namespace probability_sum_20_l3296_329666

/-- A die is represented as a finite set of natural numbers from 1 to 12 -/
def TwelveSidedDie : Finset ℕ := Finset.range 12 

/-- The sum of two dice rolls -/
def DiceSum (roll1 roll2 : ℕ) : ℕ := roll1 + roll2

/-- The set of all possible outcomes when rolling two dice -/
def AllOutcomes : Finset (ℕ × ℕ) := TwelveSidedDie.product TwelveSidedDie

/-- The set of favorable outcomes (sum of 20) -/
def FavorableOutcomes : Finset (ℕ × ℕ) :=
  AllOutcomes.filter (fun p => DiceSum p.1 p.2 = 20)

/-- The probability of rolling a sum of 20 with two twelve-sided dice -/
theorem probability_sum_20 : 
  (FavorableOutcomes.card : ℚ) / AllOutcomes.card = 5 / 144 := by
  sorry


end probability_sum_20_l3296_329666


namespace three_digit_powers_of_three_l3296_329663

theorem three_digit_powers_of_three (n : ℕ) : 
  (∃ k, 100 ≤ 3^k ∧ 3^k ≤ 999) ∧ (∀ m, 100 ≤ 3^m ∧ 3^m ≤ 999 → m = n ∨ m = n+1) :=
sorry

end three_digit_powers_of_three_l3296_329663


namespace max_distance_between_sine_cosine_curves_l3296_329615

theorem max_distance_between_sine_cosine_curves : 
  ∃ (C : ℝ), C = (Real.sqrt 3 / 2) * Real.sqrt 2 ∧ 
  ∀ (x : ℝ), |Real.sin (x + π/6) - 2 * Real.cos x| ≤ C ∧
  ∃ (a : ℝ), |Real.sin (a + π/6) - 2 * Real.cos a| = C :=
by sorry

end max_distance_between_sine_cosine_curves_l3296_329615


namespace renovation_project_dirt_required_l3296_329600

theorem renovation_project_dirt_required (sand cement total : ℚ)
  (h1 : sand = 0.16666666666666666)
  (h2 : cement = 0.16666666666666666)
  (h3 : total = 0.6666666666666666) :
  total - (sand + cement) = 0.3333333333333333 :=
by sorry

end renovation_project_dirt_required_l3296_329600


namespace max_ab_min_3x_4y_max_f_l3296_329660

-- Part 1
theorem max_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 1) :
  a * b ≤ 1 / 16 := by sorry

-- Part 2
theorem min_3x_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 := by sorry

-- Part 3
theorem max_f (x : ℝ) (h : x < 5 / 4) :
  4 * x - 2 + 1 / (4 * x - 5) ≤ 1 := by sorry

end max_ab_min_3x_4y_max_f_l3296_329660


namespace jack_christina_lindy_meeting_l3296_329694

/-- The problem of Jack, Christina, and Lindy meeting --/
theorem jack_christina_lindy_meeting 
  (jack_speed Christina_speed lindy_speed : ℝ)
  (lindy_distance : ℝ)
  (h1 : jack_speed = 7)
  (h2 : Christina_speed = 8)
  (h3 : lindy_speed = 10)
  (h4 : lindy_distance = 100) :
  (jack_speed + Christina_speed) * (lindy_distance / lindy_speed) = 150 := by
  sorry

end jack_christina_lindy_meeting_l3296_329694


namespace no_x_squared_term_l3296_329648

theorem no_x_squared_term (m : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + 2 * x * y + y^2 + m * x^2 = 2 * x * y + y^2) ↔ m = -3 :=
by sorry

end no_x_squared_term_l3296_329648


namespace sqrt_twelve_minus_three_sqrt_one_third_l3296_329696

theorem sqrt_twelve_minus_three_sqrt_one_third (x : ℝ) : 
  Real.sqrt 12 - 3 * Real.sqrt (1/3) = Real.sqrt 3 := by
  sorry

end sqrt_twelve_minus_three_sqrt_one_third_l3296_329696


namespace lauryns_company_men_count_l3296_329631

/-- The number of men employed by Lauryn's computer company. -/
def num_men : ℕ := 80

/-- The number of women employed by Lauryn's computer company. -/
def num_women : ℕ := num_men + 20

/-- The total number of employees in Lauryn's computer company. -/
def total_employees : ℕ := 180

/-- Theorem stating that the number of men employed by Lauryn is 80,
    given the conditions of the problem. -/
theorem lauryns_company_men_count :
  (num_men + num_women = total_employees) ∧ 
  (num_women = num_men + 20) →
  num_men = 80 := by
  sorry

end lauryns_company_men_count_l3296_329631


namespace simplify_expression_l3296_329603

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 9) - (x + 6)*(3*x - 2) = 7*x - 24 := by
  sorry

end simplify_expression_l3296_329603


namespace cookie_remainder_l3296_329650

theorem cookie_remainder (whole : ℝ) (person_a_fraction : ℝ) (person_b_fraction : ℝ) :
  person_a_fraction = 0.7 →
  person_b_fraction = 1/3 →
  (whole - person_a_fraction * whole) * (1 - person_b_fraction) = 0.2 * whole := by
  sorry

end cookie_remainder_l3296_329650


namespace photocopy_pages_theorem_l3296_329618

/-- The number of team members -/
def team_members : ℕ := 23

/-- The cost per page for the first 300 pages (in tenths of yuan) -/
def cost_first_300 : ℕ := 15

/-- The cost per page for additional pages beyond 300 (in tenths of yuan) -/
def cost_additional : ℕ := 10

/-- The threshold number of pages for price change -/
def threshold : ℕ := 300

/-- The ratio of total cost to single set cost -/
def cost_ratio : ℕ := 20

/-- The function to calculate the cost of photocopying a single set of materials -/
def single_set_cost (pages : ℕ) : ℕ :=
  if pages ≤ threshold then
    pages * cost_first_300
  else
    threshold * cost_first_300 + (pages - threshold) * cost_additional

/-- The function to calculate the cost of photocopying all sets of materials -/
def total_cost (pages : ℕ) : ℕ :=
  if team_members * pages ≤ threshold then
    team_members * pages * cost_first_300
  else
    threshold * cost_first_300 + (team_members * pages - threshold) * cost_additional

/-- The theorem stating that 950 pages satisfies the given conditions -/
theorem photocopy_pages_theorem :
  ∃ (pages : ℕ), pages = 950 ∧ total_cost pages = cost_ratio * single_set_cost pages :=
sorry

end photocopy_pages_theorem_l3296_329618


namespace largest_expression_l3296_329627

def P : ℕ := 3 * 2024^2025
def Q : ℕ := 2024^2025
def R : ℕ := 2023 * 2024^2024
def S : ℕ := 3 * 2024^2024
def T : ℕ := 2024^2024
def U : ℕ := 2024^2023

theorem largest_expression :
  (P - Q ≥ Q - R) ∧
  (P - Q ≥ R - S) ∧
  (P - Q ≥ S - T) ∧
  (P - Q ≥ T - U) :=
by sorry

end largest_expression_l3296_329627


namespace vector_properties_l3296_329624

def e₁ : ℝ × ℝ := (1, 0)
def e₂ : ℝ × ℝ := (0, 1)
def a : ℝ × ℝ := (3 * e₁.1 - 2 * e₂.1, 3 * e₁.2 - 2 * e₂.2)
def b : ℝ × ℝ := (4 * e₁.1 + e₂.1, 4 * e₁.2 + e₂.2)

theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 = 10) ∧
  ((a.1 + b.1)^2 + (a.2 + b.2)^2 = 50) ∧
  ((a.1 * b.1 + a.2 * b.2)^2 = 100 * ((a.1^2 + a.2^2) * (b.1^2 + b.2^2)) / 221) := by
  sorry

end vector_properties_l3296_329624


namespace ellipse_hyperbola_intersection_eccentricity_l3296_329699

/-- An ellipse with foci and eccentricity -/
structure Ellipse :=
  (F₁ F₂ : ℝ × ℝ)
  (e : ℝ)

/-- A hyperbola with foci and eccentricity -/
structure Hyperbola :=
  (F₁ F₂ : ℝ × ℝ)
  (e : ℝ)

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1) + (v.2 * w.2)

theorem ellipse_hyperbola_intersection_eccentricity 
  (C₁ : Ellipse) (C₂ : Hyperbola) (P : ℝ × ℝ) :
  C₁.F₁ = C₂.F₁ →
  C₁.F₂ = C₂.F₂ →
  dot_product (P.1 - C₁.F₁.1, P.2 - C₁.F₁.2) (P.1 - C₁.F₂.1, P.2 - C₁.F₂.2) = 0 →
  (1 / C₁.e^2) + (1 / C₂.e^2) = 2 := by
  sorry

end ellipse_hyperbola_intersection_eccentricity_l3296_329699


namespace bus_students_count_l3296_329686

theorem bus_students_count (initial_students : Real) (students_boarding : Real) : 
  initial_students = 10.0 → students_boarding = 3.0 → initial_students + students_boarding = 13.0 := by
  sorry

end bus_students_count_l3296_329686


namespace expected_min_swaps_value_l3296_329601

/-- Represents a pair of twins -/
structure TwinPair :=
  (twin1 : ℕ)
  (twin2 : ℕ)

/-- Represents an arrangement of twin pairs around a circle -/
def Arrangement := List TwinPair

/-- Computes whether an arrangement has adjacent twins -/
def has_adjacent_twins (arr : Arrangement) : Prop :=
  sorry

/-- Performs a swap between two adjacent positions in the arrangement -/
def swap (arr : Arrangement) (pos : ℕ) : Arrangement :=
  sorry

/-- Computes the minimum number of swaps needed to separate all twins -/
def min_swaps (arr : Arrangement) : ℕ :=
  sorry

/-- Generates all possible arrangements of 5 pairs of twins -/
def all_arrangements : List Arrangement :=
  sorry

/-- Computes the expected value of the minimum number of swaps -/
def expected_min_swaps : ℚ :=
  sorry

theorem expected_min_swaps_value : 
  expected_min_swaps = 926 / 945 :=
sorry

end expected_min_swaps_value_l3296_329601


namespace rachel_age_proof_l3296_329664

/-- Rachel's age in years -/
def rachel_age : ℕ := 12

/-- Rachel's grandfather's age in years -/
def grandfather_age (r : ℕ) : ℕ := 7 * r

/-- Rachel's mother's age in years -/
def mother_age (r : ℕ) : ℕ := grandfather_age r / 2

/-- Rachel's father's age in years -/
def father_age (r : ℕ) : ℕ := mother_age r + 5

theorem rachel_age_proof :
  rachel_age = 12 ∧
  grandfather_age rachel_age = 7 * rachel_age ∧
  mother_age rachel_age = grandfather_age rachel_age / 2 ∧
  father_age rachel_age = mother_age rachel_age + 5 ∧
  father_age rachel_age = rachel_age + 35 ∧
  father_age 25 = 60 :=
by sorry

end rachel_age_proof_l3296_329664


namespace data_set_average_l3296_329623

theorem data_set_average (a : ℝ) : 
  let data_set := [4, 2*a, 3-a, 5, 6]
  (data_set.sum / data_set.length = 4) → a = 2 := by
sorry

end data_set_average_l3296_329623


namespace cookout_buns_per_pack_alex_cookout_buns_per_pack_l3296_329616

/-- Calculates the number of buns in each pack given the cookout conditions -/
theorem cookout_buns_per_pack (total_guests : ℕ) (burgers_per_guest : ℕ) 
  (non_meat_guests : ℕ) (non_bread_guests : ℕ) (bun_packs : ℕ) : ℕ :=
  let guests_eating_meat := total_guests - non_meat_guests
  let guests_eating_bread := guests_eating_meat - non_bread_guests
  let total_buns_needed := guests_eating_bread * burgers_per_guest
  total_buns_needed / bun_packs

/-- Proves that the number of buns in each pack for Alex's cookout is 8 -/
theorem alex_cookout_buns_per_pack : 
  cookout_buns_per_pack 10 3 1 1 3 = 8 := by
  sorry

end cookout_buns_per_pack_alex_cookout_buns_per_pack_l3296_329616


namespace inequality_solution_set_l3296_329681

theorem inequality_solution_set : 
  {x : ℝ | x^2 - 7*x + 12 < 0} = Set.Ioo 3 4 := by sorry

end inequality_solution_set_l3296_329681


namespace quadratic_inequality_solution_l3296_329640

theorem quadratic_inequality_solution (a : ℝ) (h : a > 0) :
  let f := fun x => a * x^2 - (a^2 + 1) * x + a
  (∀ x, f x > 0 ↔
    (a > 1 ∧ (x < 1/a ∨ x > a)) ∨
    (a = 1 ∧ x ≠ 1) ∨
    (0 < a ∧ a < 1 ∧ (x < a ∨ x > 1/a))) :=
by sorry

end quadratic_inequality_solution_l3296_329640


namespace smallest_integer_satisfying_inequality_l3296_329674

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x^2 < 2*x + 3 → x ≥ 0 ∧ 0^2 < 2*0 + 3 := by
  sorry

end smallest_integer_satisfying_inequality_l3296_329674


namespace cone_volume_with_plane_intersection_l3296_329685

/-- The volume of a cone given specific plane intersections -/
theorem cone_volume_with_plane_intersection 
  (p q : ℝ) (a α : ℝ) (hp : p > 0) (hq : q > 0) (ha : a > 0) (hα : 0 < α ∧ α < π / 2) :
  let V := (π * a^3) / (3 * Real.sin α * Real.cos α^2 * Real.cos (π * q / (p + q))^2)
  ∃ (r h : ℝ), r > 0 ∧ h > 0 ∧ V = (1/3) * π * r^2 * h :=
by sorry

end cone_volume_with_plane_intersection_l3296_329685


namespace tan_75_degrees_l3296_329604

theorem tan_75_degrees : Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  -- We define tan 75° as tan(90° - 15°)
  have h1 : Real.tan (75 * π / 180) = Real.tan ((90 - 15) * π / 180) := by sorry
  
  -- The rest of the proof would go here
  sorry

end tan_75_degrees_l3296_329604


namespace solution_value_l3296_329697

theorem solution_value (a : ℝ) : (2 * a = 4) → a = 2 := by
  sorry

end solution_value_l3296_329697


namespace share_distribution_l3296_329653

theorem share_distribution (total : ℕ) (a b c : ℚ) 
  (h1 : total = 880)
  (h2 : a + b + c = total)
  (h3 : 4 * a = 5 * b)
  (h4 : 5 * b = 10 * c) :
  c = 160 := by
  sorry

end share_distribution_l3296_329653


namespace regular_polygon_interior_angle_sum_l3296_329656

theorem regular_polygon_interior_angle_sum (n : ℕ) (h1 : n > 2) (h2 : 360 / n = 18) :
  (n - 2) * 180 = 3240 := by
  sorry

end regular_polygon_interior_angle_sum_l3296_329656


namespace even_odd_sum_difference_l3296_329628

def first_n_even_sum (n : ℕ) : ℕ := n * (n + 1)

def first_n_odd_sum (n : ℕ) : ℕ := n^2

theorem even_odd_sum_difference : 
  first_n_even_sum 1500 - first_n_odd_sum 1500 = 1500 :=
by sorry

end even_odd_sum_difference_l3296_329628


namespace roots_square_sum_l3296_329647

-- Define the quadratic equation
def quadratic (x : ℝ) : ℝ := x^2 - 3*x + 1

-- Define the roots
theorem roots_square_sum : 
  ∀ r s : ℝ, quadratic r = 0 → quadratic s = 0 → r^2 + s^2 = 7 :=
by
  sorry

#check roots_square_sum

end roots_square_sum_l3296_329647


namespace whale_consumption_increase_l3296_329646

/-- Represents the whale's plankton consumption pattern over 9 hours -/
structure WhaleConsumption where
  initial : ℝ  -- Initial consumption in the first hour
  increase : ℝ  -- Increase in consumption each hour
  total : ℝ     -- Total consumption over 9 hours
  sixth_hour : ℝ -- Consumption in the sixth hour

/-- The whale's consumption satisfies the given conditions -/
def satisfies_conditions (w : WhaleConsumption) : Prop :=
  w.total = 270 ∧ 
  w.sixth_hour = 33 ∧ 
  w.total = (9 * w.initial + 36 * w.increase) ∧
  w.sixth_hour = w.initial + 5 * w.increase

/-- The theorem stating that the increase in consumption is 3 kilos per hour -/
theorem whale_consumption_increase (w : WhaleConsumption) 
  (h : satisfies_conditions w) : w.increase = 3 := by
  sorry

end whale_consumption_increase_l3296_329646


namespace softball_team_ratio_l3296_329657

/-- Proves that for a team with 4 more women than men and 20 total players, the ratio of men to women is 2:3 -/
theorem softball_team_ratio : 
  ∀ (men women : ℕ), 
  women = men + 4 →
  men + women = 20 →
  (men : ℚ) / women = 2 / 3 := by
sorry

end softball_team_ratio_l3296_329657


namespace arithmetic_calculations_l3296_329644

theorem arithmetic_calculations :
  (58 + 15 * 4 = 118) ∧
  (216 - 72 / 8 = 207) ∧
  ((358 - 295) / 7 = 9) := by
  sorry

end arithmetic_calculations_l3296_329644


namespace angle_sum_equality_l3296_329641

theorem angle_sum_equality (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (h4 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π / 2 := by
sorry

end angle_sum_equality_l3296_329641


namespace carrots_in_second_bed_l3296_329639

/-- Given Kelly's carrot harvest information, prove the number of carrots in the second bed --/
theorem carrots_in_second_bed 
  (total_pounds : ℕ)
  (carrots_per_pound : ℕ)
  (first_bed : ℕ)
  (third_bed : ℕ)
  (h1 : total_pounds = 39)
  (h2 : carrots_per_pound = 6)
  (h3 : first_bed = 55)
  (h4 : third_bed = 78) :
  total_pounds * carrots_per_pound - first_bed - third_bed = 101 := by
  sorry

#check carrots_in_second_bed

end carrots_in_second_bed_l3296_329639


namespace parabola_translation_l3296_329608

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
  , b := p.b - 2 * p.a * h
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation :
  let original := Parabola.mk 1 (-6) 5
  let translated := translate original 1 2
  translated = Parabola.mk 1 (-8) 14 := by sorry

end parabola_translation_l3296_329608


namespace line_through_intersection_and_parallel_l3296_329673

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x + 3*y - 7 = 0
def line2 (x y : ℝ) : Prop := 7*x + 15*y + 1 = 0
def line3 (x y : ℝ) : Prop := x + 2*y - 3 = 0
def target_line (x y : ℝ) : Prop := 3*x + 6*y - 2 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Theorem statement
theorem line_through_intersection_and_parallel :
  ∃ (x y : ℝ), intersection_point x y ∧ target_line x y ∧
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), target_line x y ↔ line3 (k*x) (k*y) :=
sorry

end line_through_intersection_and_parallel_l3296_329673


namespace inequality_system_solution_l3296_329645

theorem inequality_system_solution :
  let S := {x : ℝ | (x - 1 < 2) ∧ (2*x + 3 ≥ x - 1)}
  S = {x : ℝ | -4 ≤ x ∧ x < 3} :=
by sorry

end inequality_system_solution_l3296_329645


namespace largest_number_in_systematic_sample_l3296_329636

/-- The largest number in a systematic sample --/
theorem largest_number_in_systematic_sample :
  let population_size : ℕ := 60
  let sample_size : ℕ := 10
  let remainder : ℕ := 3
  let divisor : ℕ := 6
  let sampling_interval : ℕ := population_size / sample_size
  let first_sample : ℕ := remainder
  let last_sample : ℕ := first_sample + sampling_interval * (sample_size - 1)
  last_sample = 57 := by sorry

end largest_number_in_systematic_sample_l3296_329636


namespace simplify_fraction_with_sqrt_three_l3296_329609

theorem simplify_fraction_with_sqrt_three : 
  (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 := by sorry

end simplify_fraction_with_sqrt_three_l3296_329609


namespace equation_solution_l3296_329630

theorem equation_solution : 
  let f : ℝ → ℝ := λ x => (x^2 - x + 1) * (3*x^2 - 10*x + 3) - 20*x^2
  ∀ x : ℝ, f x = 0 ↔ x = (5 + Real.sqrt 21) / 2 ∨ x = (5 - Real.sqrt 21) / 2 := by
sorry

end equation_solution_l3296_329630


namespace choose_three_from_seven_l3296_329613

theorem choose_three_from_seven (n : ℕ) (k : ℕ) : n = 7 ∧ k = 3 → Nat.choose n k = 35 := by
  sorry

end choose_three_from_seven_l3296_329613


namespace max_sum_of_four_digit_integers_l3296_329654

/-- A function that returns true if a number is a 4-digit integer -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that returns the set of digits in a number -/
def digits (n : ℕ) : Finset ℕ :=
  Finset.filter (fun d => d < 10) (Finset.range (n + 1))

/-- The theorem statement -/
theorem max_sum_of_four_digit_integers (a c : ℕ) :
  is_four_digit a ∧ is_four_digit c ∧
  (digits a ∪ digits c = Finset.range 10) →
  a + c ≤ 18395 :=
sorry

end max_sum_of_four_digit_integers_l3296_329654


namespace equation_relation_l3296_329658

theorem equation_relation (x y z w : ℝ) :
  (x + 2 * y) / (2 * y + 3 * z) = (3 * z + 4 * w) / (4 * w + x) →
  x = 3 * z ∨ x + 2 * y + 4 * w + 3 * z = 0 :=
by sorry

end equation_relation_l3296_329658


namespace cosine_sine_identity_l3296_329634

theorem cosine_sine_identity : 
  Real.cos (80 * π / 180) * Real.cos (35 * π / 180) + 
  Real.sin (80 * π / 180) * Real.cos (55 * π / 180) = 
  (1 / 2) * (Real.sin (65 * π / 180) + Real.sin (25 * π / 180)) := by
  sorry

end cosine_sine_identity_l3296_329634


namespace odd_sum_of_squares_implies_odd_sum_l3296_329655

theorem odd_sum_of_squares_implies_odd_sum (n m : ℤ) 
  (h : Odd (n^2 + m^2)) : Odd (n + m) := by
  sorry

end odd_sum_of_squares_implies_odd_sum_l3296_329655


namespace union_of_sets_l3296_329619

theorem union_of_sets : 
  let A : Set ℕ := {1, 3, 5}
  let B : Set ℕ := {3, 5, 7}
  A ∪ B = {1, 3, 5, 7} := by sorry

end union_of_sets_l3296_329619


namespace parabola_coefficients_l3296_329612

/-- A parabola with given properties has specific coefficients -/
theorem parabola_coefficients :
  ∀ (a b c : ℝ),
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (3 = a * 4^2 + b * 4 + c) →
  (4 = -b / (2 * a)) →
  (7 = a * 2^2 + b * 2 + c) →
  a = 1 ∧ b = -8 ∧ c = 19 :=
by sorry

end parabola_coefficients_l3296_329612


namespace total_followers_count_l3296_329606

def instagram_followers : ℕ := 240
def facebook_followers : ℕ := 500

def twitter_followers : ℕ := (instagram_followers + facebook_followers) / 2
def tiktok_followers : ℕ := 3 * twitter_followers
def youtube_followers : ℕ := tiktok_followers + 510

def total_followers : ℕ := instagram_followers + facebook_followers + twitter_followers + tiktok_followers + youtube_followers

theorem total_followers_count : total_followers = 3840 := by
  sorry

end total_followers_count_l3296_329606


namespace marathon_distance_theorem_l3296_329625

/-- The length of a marathon in miles -/
def marathon_miles : ℕ := 26

/-- The additional length of a marathon in yards -/
def marathon_yards : ℕ := 312

/-- The number of yards in a mile -/
def yards_per_mile : ℕ := 1760

/-- The number of marathons Joanna has run -/
def num_marathons : ℕ := 8

/-- The total distance Joanna has run in yards -/
def total_distance : ℕ := num_marathons * (marathon_miles * yards_per_mile + marathon_yards)

theorem marathon_distance_theorem :
  ∃ (m : ℕ) (y : ℕ), total_distance = m * yards_per_mile + y ∧ y = 736 ∧ y < yards_per_mile :=
by sorry

end marathon_distance_theorem_l3296_329625


namespace correct_mean_calculation_l3296_329620

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 30 ∧ initial_mean = 180 ∧ incorrect_value = 135 ∧ correct_value = 155 →
  (n * initial_mean + (correct_value - incorrect_value)) / n = 180.67 := by
  sorry

end correct_mean_calculation_l3296_329620


namespace choose_three_from_nine_l3296_329670

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end choose_three_from_nine_l3296_329670


namespace complement_A_in_U_l3296_329643

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define set A
def A : Set ℝ := {x | x * (x - 1) < 0}

-- Theorem statement
theorem complement_A_in_U : 
  {x ∈ U | x ∉ A} = {x : ℝ | x ≥ 1} := by sorry

end complement_A_in_U_l3296_329643


namespace no_positive_integer_solutions_l3296_329675

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 + y^2 = x^4 := by
sorry

end no_positive_integer_solutions_l3296_329675


namespace total_different_movies_l3296_329661

-- Define the number of people
def num_people : ℕ := 5

-- Define the number of movies watched by each person
def dalton_movies : ℕ := 15
def hunter_movies : ℕ := 19
def alex_movies : ℕ := 25
def bella_movies : ℕ := 21
def chris_movies : ℕ := 11

-- Define the number of movies watched together
def all_together : ℕ := 5
def dalton_hunter_alex : ℕ := 3
def bella_chris : ℕ := 2

-- Theorem to prove
theorem total_different_movies : 
  dalton_movies + hunter_movies + alex_movies + bella_movies + chris_movies
  - (num_people - 1) * all_together
  - (3 - 1) * dalton_hunter_alex
  - (2 - 1) * bella_chris = 63 := by
sorry

end total_different_movies_l3296_329661


namespace parabola_vertex_coordinates_l3296_329607

/-- The vertex of the parabola y = x^2 - 9 has coordinates (0, -9) -/
theorem parabola_vertex_coordinates :
  let f : ℝ → ℝ := λ x => x^2 - 9
  ∃ (x y : ℝ), (∀ t, f t ≥ f x) ∧ y = f x ∧ x = 0 ∧ y = -9 :=
by sorry

end parabola_vertex_coordinates_l3296_329607


namespace set_A_equals_circle_B_l3296_329687

-- Define the circle D
def circle_D (Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | dist P Q = 10}

-- Define point B
def point_B (Q : ℝ × ℝ) : ℝ × ℝ :=
  let v : ℝ × ℝ := (6, 0)  -- Arbitrary direction, 6 units from Q
  (Q.1 + v.1, Q.2 + v.2)

-- Define the set of points A satisfying the condition
def set_A (Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {A | ∀ P ∈ circle_D Q, dist A (point_B Q) ≤ dist A P}

-- Define the circle with center B and radius 4
def circle_B (Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | dist P (point_B Q) ≤ 4}

-- The theorem to prove
theorem set_A_equals_circle_B (Q : ℝ × ℝ) : set_A Q = circle_B Q := by
  sorry

end set_A_equals_circle_B_l3296_329687


namespace similar_triangle_leg_length_l3296_329622

theorem similar_triangle_leg_length
  (a b c : ℝ)  -- sides of the first triangle
  (d e f : ℝ)  -- sides of the second triangle
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hd : d > 0) (he : e > 0) (hf : f > 0)
  (right_triangle1 : a^2 + b^2 = c^2)  -- first triangle is right triangle
  (right_triangle2 : d^2 + e^2 = f^2)  -- second triangle is right triangle
  (similar : a / d = b / e ∧ b / e = c / f)  -- triangles are similar
  (leg1 : a = 15)  -- one leg of first triangle
  (hyp1 : c = 17)  -- hypotenuse of first triangle
  (hyp2 : f = 51)  -- hypotenuse of second triangle
  : e = 24 :=  -- corresponding leg in second triangle
by sorry

end similar_triangle_leg_length_l3296_329622


namespace solution_set_f_max_value_g_range_of_m_l3296_329695

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Define the function g
def g (x : ℝ) : ℝ := f x - x^2 + x

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f (x : ℝ) : f x ≥ 1 ↔ x ≥ 1 := by sorry

-- Theorem for the maximum value of g(x)
theorem max_value_g : ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 5/4 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : (∃ (x : ℝ), f x ≥ x^2 - x + m) ↔ m ≤ 5/4 := by sorry

end solution_set_f_max_value_g_range_of_m_l3296_329695


namespace common_volume_formula_l3296_329678

/-- Represents a cube with edge length a -/
structure Cube where
  a : ℝ
  a_pos : a > 0

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- Represents the configuration of a cube and a tetrahedron with aligned edges and coinciding midpoints -/
structure CubeTetrahedronConfig where
  cube : Cube
  tetrahedron : RegularTetrahedron
  aligned_edges : Bool
  coinciding_midpoints : Bool

/-- Calculates the volume of the common part of a cube and a tetrahedron in the given configuration -/
def common_volume (config : CubeTetrahedronConfig) : ℝ := sorry

/-- Theorem stating the volume of the common part of the cube and tetrahedron -/
theorem common_volume_formula (config : CubeTetrahedronConfig) 
  (h_aligned : config.aligned_edges = true) 
  (h_coincide : config.coinciding_midpoints = true) :
  common_volume config = (config.cube.a^3 * Real.sqrt 2 / 12) * (16 * Real.sqrt 2 - 17) := by
  sorry

end common_volume_formula_l3296_329678


namespace quadratic_inequality_range_l3296_329602

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + a ≥ 0) → a ≥ 1 := by
  sorry

end quadratic_inequality_range_l3296_329602


namespace power_product_equals_four_digit_l3296_329637

/-- Given that 2^x × 9^y equals the four-digit number 2x9y, prove that x^2 * y^3 = 200 -/
theorem power_product_equals_four_digit (x y : ℕ) : 
  (2^x * 9^y = 2000 + 100*x + 10*y + 9) → 
  (1000 ≤ 2000 + 100*x + 10*y + 9) → 
  (2000 + 100*x + 10*y + 9 < 10000) → 
  x^2 * y^3 = 200 := by sorry

end power_product_equals_four_digit_l3296_329637


namespace condition_sufficient_not_necessary_l3296_329649

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > |a n|

theorem condition_sufficient_not_necessary :
  (∀ a : ℕ → ℝ, satisfies_condition a → is_increasing a) ∧
  (∃ a : ℕ → ℝ, is_increasing a ∧ ¬satisfies_condition a) :=
by sorry

end condition_sufficient_not_necessary_l3296_329649


namespace ellipse_properties_l3296_329610

/-- Definition of an ellipse C with given parameters -/
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- Definition of an equilateral triangle with given side length -/
def EquilateralTriangle (side : ℝ) (p1 p2 p3 : ℝ × ℝ) :=
  ‖p1 - p2‖ = side ∧ ‖p2 - p3‖ = side ∧ ‖p3 - p1‖ = side

/-- Theorem about the properties of a specific ellipse -/
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (F1 F2 B : ℝ × ℝ) (h3 : EquilateralTriangle 2 B F1 F2) :
  ∃ (C : Set (ℝ × ℝ)) (e : ℝ) (l1 l2 : ℝ → ℝ),
    C = Ellipse 2 (Real.sqrt 3) ∧
    e = (1 : ℝ) / 2 ∧
    (∀ x, l1 x = (Real.sqrt 5 * x - Real.sqrt 5) / 2) ∧
    (∀ x, l2 x = (-Real.sqrt 5 * x + Real.sqrt 5) / 2) ∧
    (∃ P Q : ℝ × ℝ, P ∈ C ∧ Q ∈ C ∧
      (P.2 = l1 P.1 ∨ P.2 = l2 P.1) ∧
      (Q.2 = l1 Q.1 ∨ Q.2 = l2 Q.1) ∧
      ((P.1 - 2) * (Q.2 + 1) = (P.2) * (Q.1 + 1))) :=
by
  sorry

end ellipse_properties_l3296_329610


namespace probability_green_ball_l3296_329659

/-- The probability of drawing a green ball from a bag with specified contents -/
theorem probability_green_ball (green black red : ℕ) : 
  green = 3 → black = 3 → red = 6 → 
  (green : ℚ) / (green + black + red : ℚ) = 1/4 := by
  sorry

end probability_green_ball_l3296_329659


namespace algebraic_division_l3296_329621

theorem algebraic_division (m : ℝ) : -20 * m^6 / (5 * m^2) = -4 * m^4 := by
  sorry

end algebraic_division_l3296_329621


namespace tan_theta_value_l3296_329689

theorem tan_theta_value (θ : Real) 
  (h1 : 2 * Real.sin θ + Real.cos θ = Real.sqrt 2 / 3)
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.tan θ = -(90 + 5 * Real.sqrt 86) / 168 := by
  sorry

end tan_theta_value_l3296_329689


namespace intersection_condition_l3296_329692

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M (a : ℝ) : Set ℝ := {x | x + a ≥ 0}

-- Define set N
def N : Set ℝ := {x | x - 2 < 1}

-- Theorem statement
theorem intersection_condition (a : ℝ) :
  M a ∩ (Set.compl N) = {x | x ≥ 3} → a ≥ 3 := by
  sorry

end intersection_condition_l3296_329692


namespace jerry_needs_72_dollars_l3296_329677

/-- The amount of money Jerry needs to finish his action figure collection -/
def jerryNeedsMoney (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Proof that Jerry needs $72 to finish his collection -/
theorem jerry_needs_72_dollars :
  jerryNeedsMoney 7 16 8 = 72 := by
  sorry

end jerry_needs_72_dollars_l3296_329677


namespace card_area_theorem_l3296_329671

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem card_area_theorem (original : Rectangle) 
  (h1 : original.length = 3 ∧ original.width = 7)
  (h2 : ∃ (shortened : Rectangle), 
    (shortened.length = original.length ∧ shortened.width = original.width - 2) ∨
    (shortened.length = original.length - 2 ∧ shortened.width = original.width) ∧
    area shortened = 15) :
  ∃ (other_shortened : Rectangle),
    ((other_shortened.length = original.length - 1 ∧ other_shortened.width = original.width) ∨
     (other_shortened.length = original.length ∧ other_shortened.width = original.width - 1)) ∧
    area other_shortened = 10 := by
  sorry

end card_area_theorem_l3296_329671


namespace travel_time_ratio_is_one_to_one_l3296_329662

-- Define the time spent on each leg of the journey
def walk_to_bus : ℕ := 5
def bus_ride : ℕ := 20
def walk_to_job : ℕ := 5

-- Define the total travel time per year in hours
def total_travel_time_per_year : ℕ := 365

-- Define the number of days worked per year
def days_per_year : ℕ := 365

-- Define the total travel time for one way (morning or evening)
def one_way_travel_time : ℕ := walk_to_bus + bus_ride + walk_to_job

-- Theorem to prove
theorem travel_time_ratio_is_one_to_one :
  one_way_travel_time = (total_travel_time_per_year * 60) / (2 * days_per_year) :=
by
  sorry

#check travel_time_ratio_is_one_to_one

end travel_time_ratio_is_one_to_one_l3296_329662


namespace art_class_selection_l3296_329672

theorem art_class_selection (n m k : ℕ) (hn : n = 10) (hm : m = 4) (hk : k = 2) :
  (Nat.choose (n - k + 1) (m - k + 1)) = 56 := by
  sorry

end art_class_selection_l3296_329672


namespace binomial_product_factorial_equals_l3296_329651

theorem binomial_product_factorial_equals : (
  Nat.choose 10 3 * Nat.choose 8 3 * (Nat.factorial 7 / Nat.factorial 4)
) = 235200 := by
  sorry

end binomial_product_factorial_equals_l3296_329651


namespace undecagon_diagonals_l3296_329698

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular undecagon has 11 sides -/
def undecagon_sides : ℕ := 11

/-- Theorem: A regular undecagon (11-sided polygon) has 44 diagonals -/
theorem undecagon_diagonals :
  num_diagonals undecagon_sides = 44 := by sorry

end undecagon_diagonals_l3296_329698


namespace radio_price_rank_l3296_329642

theorem radio_price_rank (n : ℕ) (prices : Finset ℕ) (radio_price : ℕ) :
  n = 43 →
  prices.card = n →
  (∀ (p q : ℕ), p ∈ prices → q ∈ prices → p ≠ q) →
  radio_price ∈ prices →
  (prices.filter (λ p => p > radio_price)).card = 8 →
  ∃ (m : ℕ), (prices.filter (λ p => p < radio_price)).card = m - 1 →
  (prices.filter (λ p => p ≤ radio_price)).card = 35 :=
by sorry

end radio_price_rank_l3296_329642


namespace inequality_solution_sum_l3296_329605

theorem inequality_solution_sum (m n : ℝ) : 
  (∀ x, x ∈ Set.Ioo m n ↔ (m * x - 1) / (x + 3) > 0) →
  m + n = -10/3 :=
by sorry

end inequality_solution_sum_l3296_329605
