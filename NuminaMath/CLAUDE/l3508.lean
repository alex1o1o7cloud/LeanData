import Mathlib

namespace charity_pastries_count_l3508_350849

theorem charity_pastries_count (total_volunteers : ℕ) 
  (group_a_percent group_b_percent group_c_percent : ℚ)
  (group_a_batches group_b_batches group_c_batches : ℕ)
  (group_a_trays group_b_trays group_c_trays : ℕ)
  (group_a_pastries group_b_pastries group_c_pastries : ℕ) :
  total_volunteers = 1500 →
  group_a_percent = 2/5 →
  group_b_percent = 7/20 →
  group_c_percent = 1/4 →
  group_a_batches = 10 →
  group_b_batches = 15 →
  group_c_batches = 8 →
  group_a_trays = 6 →
  group_b_trays = 4 →
  group_c_trays = 5 →
  group_a_pastries = 20 →
  group_b_pastries = 12 →
  group_c_pastries = 15 →
  (↑total_volunteers * group_a_percent).floor * group_a_batches * group_a_trays * group_a_pastries +
  (↑total_volunteers * group_b_percent).floor * group_b_batches * group_b_trays * group_b_pastries +
  (↑total_volunteers * group_c_percent).floor * group_c_batches * group_c_trays * group_c_pastries = 1323000 := by
  sorry


end charity_pastries_count_l3508_350849


namespace typing_speed_ratio_l3508_350893

-- Define Tim's typing speed
def tim_speed : ℝ := 2

-- Define Tom's normal typing speed
def tom_speed : ℝ := 10

-- Define Tom's increased typing speed (30% increase)
def tom_increased_speed : ℝ := tom_speed * 1.3

-- Theorem to prove
theorem typing_speed_ratio :
  -- Condition 1: Tim and Tom can type 12 pages in one hour together
  tim_speed + tom_speed = 12 →
  -- Condition 2: With Tom's increased speed, they can type 15 pages in one hour
  tim_speed + tom_increased_speed = 15 →
  -- Conclusion: The ratio of Tom's normal speed to Tim's is 5:1
  tom_speed / tim_speed = 5 := by
  sorry

end typing_speed_ratio_l3508_350893


namespace andys_basketball_team_size_l3508_350807

/-- The number of cookies Andy had initially -/
def initial_cookies : ℕ := 72

/-- The number of cookies Andy ate -/
def cookies_eaten : ℕ := 3

/-- The number of cookies Andy gave to his little brother -/
def cookies_given : ℕ := 5

/-- Calculate the remaining cookies after Andy ate some and gave some to his brother -/
def remaining_cookies : ℕ := initial_cookies - (cookies_eaten + cookies_given)

/-- Function to calculate the sum of the first n odd numbers -/
def sum_odd_numbers (n : ℕ) : ℕ := n * n

theorem andys_basketball_team_size :
  ∃ (team_size : ℕ), team_size > 0 ∧ sum_odd_numbers team_size = remaining_cookies :=
sorry

end andys_basketball_team_size_l3508_350807


namespace marble_probability_difference_l3508_350897

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 2000

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 2000

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def P_s : ℚ := (red_marbles * (red_marbles - 1) + black_marbles * (black_marbles - 1)) / (total_marbles * (total_marbles - 1))

/-- The probability of drawing two marbles of different colors -/
def P_d : ℚ := (2 * red_marbles * black_marbles) / (total_marbles * (total_marbles - 1))

/-- The theorem stating the absolute difference between P_s and P_d -/
theorem marble_probability_difference : |P_s - P_d| = 1 / 3999 := by sorry

end marble_probability_difference_l3508_350897


namespace dividend_calculation_l3508_350899

/-- The dividend calculation problem -/
theorem dividend_calculation (divisor quotient remainder : ℝ) 
  (h_divisor : divisor = 176.22471910112358)
  (h_quotient : quotient = 89)
  (h_remainder : remainder = 14) :
  divisor * quotient + remainder = 15697.799999999998 := by
  sorry

end dividend_calculation_l3508_350899


namespace caramel_candies_count_l3508_350816

theorem caramel_candies_count (total : ℕ) (lemon : ℕ) (caramel : ℕ) : 
  lemon = 4 →
  (caramel : ℚ) / (total : ℚ) = 3 / 7 →
  total = lemon + caramel →
  caramel = 3 := by
sorry

end caramel_candies_count_l3508_350816


namespace reverse_divisibility_implies_divides_99_l3508_350859

/-- Given a natural number, return the number formed by reversing its digits -/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- A natural number k has the property that if it divides n, it also divides the reverse of n -/
def has_reverse_divisibility_property (k : ℕ) : Prop :=
  ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n

theorem reverse_divisibility_implies_divides_99 (k : ℕ) :
  has_reverse_divisibility_property k → 99 ∣ k := by sorry

end reverse_divisibility_implies_divides_99_l3508_350859


namespace triangle_properties_l3508_350837

theorem triangle_properties (x : ℝ) (h : x > 0) :
  let a := 5*x
  let b := 12*x
  let c := 13*x
  (a^2 + b^2 = c^2) ∧ (∃ q : ℚ, (a / b : ℝ) = q) :=
by sorry

end triangle_properties_l3508_350837


namespace monotone_sine_function_l3508_350817

/-- The function f(x) = x + t*sin(2x) is monotonically increasing on ℝ if and only if t ∈ [-1/2, 1/2] -/
theorem monotone_sine_function (t : ℝ) :
  (∀ x : ℝ, Monotone (λ x => x + t * Real.sin (2 * x))) ↔ t ∈ Set.Icc (-1/2) (1/2) := by
  sorry

end monotone_sine_function_l3508_350817


namespace expression_equals_six_l3508_350829

-- Define the expression
def expression : ℚ := 3 * (3 + 3) / 3

-- Theorem statement
theorem expression_equals_six : expression = 6 := by
  sorry

end expression_equals_six_l3508_350829


namespace tan_equation_l3508_350877

theorem tan_equation (α : Real) (h : Real.tan α = 2) :
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 1 := by
  sorry

end tan_equation_l3508_350877


namespace bee_flight_count_l3508_350848

/-- Represents the energy content of honey in terms of bee flight distance -/
def honey_energy : ℕ := 7000

/-- Represents the amount of honey available -/
def honey_amount : ℕ := 10

/-- Represents the distance each bee should fly -/
def flight_distance : ℕ := 1

/-- Theorem: Given the energy content of honey and the amount available,
    calculate the number of bees that can fly a specified distance -/
theorem bee_flight_count :
  (honey_energy * honey_amount) / flight_distance = 70000 := by
  sorry

end bee_flight_count_l3508_350848


namespace perfect_square_bc_l3508_350863

theorem perfect_square_bc (a b c : ℕ) 
  (h : (a^2 / (a^2 + b^2) : ℚ) + (c^2 / (a^2 + c^2) : ℚ) = 2 * c / (b + c)) : 
  ∃ k : ℕ, b * c = k^2 := by
sorry

end perfect_square_bc_l3508_350863


namespace emmas_drive_speed_l3508_350887

/-- Proves that given the conditions of Emma's drive, her average speed during the last 40 minutes was 75 mph -/
theorem emmas_drive_speed (total_distance : ℝ) (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 120)
  (h2 : total_time = 2)
  (h3 : speed1 = 50)
  (h4 : speed2 = 55) :
  let segment_time := total_time / 3
  let speed3 := (total_distance - (speed1 + speed2) * segment_time) / segment_time
  speed3 = 75 := by sorry

end emmas_drive_speed_l3508_350887


namespace odd_function_periodic_l3508_350809

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_periodic 
  (f : ℝ → ℝ) 
  (h1 : IsOdd f) 
  (h2 : ∀ x, f x = f (2 - x)) : 
  IsPeriodic f 4 := by
sorry

end odd_function_periodic_l3508_350809


namespace right_triangle_circles_l3508_350874

theorem right_triangle_circles (a b : ℝ) (R r : ℝ) : 
  a = 16 → b = 30 → 
  R = (a^2 + b^2).sqrt / 2 → 
  r = (a * b) / (a + b + (a^2 + b^2).sqrt) → 
  R + r = 23 := by sorry

end right_triangle_circles_l3508_350874


namespace frisbee_committee_formations_l3508_350818

def num_teams : Nat := 5
def team_size : Nat := 8
def host_committee_size : Nat := 4
def non_host_committee_size : Nat := 2

theorem frisbee_committee_formations :
  (num_teams * (Nat.choose team_size host_committee_size) *
   (Nat.choose team_size non_host_committee_size) ^ (num_teams - 1)) =
  215134600 := by
  sorry

end frisbee_committee_formations_l3508_350818


namespace division_problem_l3508_350852

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 122 → quotient = 6 → remainder = 2 → 
  dividend = divisor * quotient + remainder →
  divisor = 20 := by sorry

end division_problem_l3508_350852


namespace equation_solution_l3508_350875

theorem equation_solution (a b c d p : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |p| = 3) :
  ∃! x : ℝ, (a + b) * x^2 + 4 * c * d * x + p^2 = x ∧ x = -3 := by
sorry

end equation_solution_l3508_350875


namespace point_not_on_line_l3508_350890

theorem point_not_on_line (p q : ℝ) (h : p * q < 0) :
  -101 ≠ 21 * p + q := by sorry

end point_not_on_line_l3508_350890


namespace percentage_problem_l3508_350862

/-- Proves that the percentage is 50% given the problem conditions -/
theorem percentage_problem (x : ℝ) : 
  (x / 100) * 150 = 75 / 100 → x = 50 := by
  sorry

end percentage_problem_l3508_350862


namespace min_distance_point_is_circumcenter_l3508_350879

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop :=
  sorry

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Finds the foot of the perpendicular from a point to a line segment -/
def perpendicularFoot (p : Point) (a b : Point) : Point :=
  sorry

/-- Calculates the circumcenter of a triangle -/
def circumcenter (t : Triangle) : Point :=
  sorry

/-- Main theorem: The point that minimizes the sum of squared distances to the sides
    of an acute triangle is its circumcenter -/
theorem min_distance_point_is_circumcenter (t : Triangle) (h : isAcute t) :
  ∀ P : Point,
    let L := perpendicularFoot P t.B t.C
    let M := perpendicularFoot P t.C t.A
    let N := perpendicularFoot P t.A t.B
    squaredDistance P L + squaredDistance P M + squaredDistance P N ≥
    let C := circumcenter t
    let CL := perpendicularFoot C t.B t.C
    let CM := perpendicularFoot C t.C t.A
    let CN := perpendicularFoot C t.A t.B
    squaredDistance C CL + squaredDistance C CM + squaredDistance C CN :=
by
  sorry

end min_distance_point_is_circumcenter_l3508_350879


namespace y_value_proof_l3508_350811

theorem y_value_proof (y : ℚ) (h : 2/3 - 1/4 = 4/y) : y = 48/5 := by
  sorry

end y_value_proof_l3508_350811


namespace curve_equation_l3508_350831

/-- Given a curve ax² + by² = 2 passing through (0, 5/3) and (1, 1), with a + b = 2,
    prove that the equation of the curve is 16/25 * x² + 9/25 * y² = 1 -/
theorem curve_equation (a b : ℝ) :
  (∀ x y : ℝ, a * x^2 + b * y^2 = 2) →
  (a * 0^2 + b * (5/3)^2 = 2) →
  (a * 1^2 + b * 1^2 = 2) →
  (a + b = 2) →
  (∀ x y : ℝ, 16/25 * x^2 + 9/25 * y^2 = 1) :=
by sorry

end curve_equation_l3508_350831


namespace one_time_cost_correct_l3508_350846

/-- The one-time product cost for editing and printing --/
def one_time_cost : ℝ := 56430

/-- The variable cost per book --/
def variable_cost : ℝ := 8.25

/-- The selling price per book --/
def selling_price : ℝ := 21.75

/-- The number of books at the break-even point --/
def break_even_books : ℕ := 4180

/-- Theorem stating that the one-time cost is correct given the conditions --/
theorem one_time_cost_correct :
  one_time_cost = (selling_price - variable_cost) * break_even_books :=
by sorry

end one_time_cost_correct_l3508_350846


namespace truck_toll_theorem_l3508_350857

/-- Calculates the toll for a truck given the number of axles -/
def toll (x : ℕ) : ℚ := 2.5 + 0.5 * (x - 2)

/-- Calculates the number of axles for a truck given the total number of wheels,
    the number of wheels on the front axle, and the number of wheels on each other axle -/
def calculateAxles (totalWheels frontAxleWheels otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

theorem truck_toll_theorem :
  let totalWheels : ℕ := 18
  let frontAxleWheels : ℕ := 2
  let otherAxleWheels : ℕ := 4
  let numAxles : ℕ := calculateAxles totalWheels frontAxleWheels otherAxleWheels
  toll numAxles = 4 := by
  sorry

end truck_toll_theorem_l3508_350857


namespace sector_area_l3508_350854

theorem sector_area (r : ℝ) (α : ℝ) (h1 : r = 3) (h2 : α = 2) :
  (1 / 2 : ℝ) * r^2 * α = 9 := by
  sorry

end sector_area_l3508_350854


namespace hyperbola_asymptote_ratio_l3508_350835

/-- For a hyperbola with equation x^2/a^2 - y^2/b^2 = 1, where a > b and the angle between
    the asymptotes is 30°, the ratio a/b = 2 - √3. -/
theorem hyperbola_asymptote_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.pi / 6 = Real.arctan ((2 * b / a) / (1 - (b / a)^2))) →
  a / b = 2 - Real.sqrt 3 :=
by sorry

end hyperbola_asymptote_ratio_l3508_350835


namespace expression_equals_four_l3508_350827

theorem expression_equals_four :
  (-2022)^0 - 2 * Real.tan (45 * π / 180) + |(-2)| + Real.sqrt 9 = 4 := by
  sorry

end expression_equals_four_l3508_350827


namespace jerseys_sold_is_two_l3508_350822

/-- The profit made from selling one jersey -/
def profit_per_jersey : ℕ := 76

/-- The total profit made from selling jerseys during the game -/
def total_profit : ℕ := 152

/-- The number of jerseys sold during the game -/
def jerseys_sold : ℕ := total_profit / profit_per_jersey

theorem jerseys_sold_is_two : jerseys_sold = 2 := by sorry

end jerseys_sold_is_two_l3508_350822


namespace season_games_count_l3508_350802

/-- The number of teams in the conference -/
def num_teams : ℕ := 10

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 6

/-- The total number of games in a season -/
def total_games : ℕ := num_teams * (num_teams - 1) + num_teams * non_conference_games

theorem season_games_count :
  total_games = 150 :=
by sorry

end season_games_count_l3508_350802


namespace residential_ratio_is_half_l3508_350851

/-- Represents a building with residential, office, and restaurant units. -/
structure Building where
  total_units : ℕ
  restaurant_units : ℕ
  office_units : ℕ
  residential_units : ℕ

/-- The ratio of residential units to total units in a building. -/
def residential_ratio (b : Building) : ℚ :=
  b.residential_units / b.total_units

/-- Theorem stating the residential ratio for a specific building configuration. -/
theorem residential_ratio_is_half (b : Building) 
    (h1 : b.total_units = 300)
    (h2 : b.restaurant_units = 75)
    (h3 : b.office_units = b.restaurant_units)
    (h4 : b.residential_units = b.total_units - (b.restaurant_units + b.office_units)) :
    residential_ratio b = 1 / 2 := by
  sorry

end residential_ratio_is_half_l3508_350851


namespace total_hours_worked_l3508_350876

theorem total_hours_worked (saturday_hours sunday_hours : ℕ) 
  (h1 : saturday_hours = 6) 
  (h2 : sunday_hours = 4) : 
  saturday_hours + sunday_hours = 10 := by
  sorry

end total_hours_worked_l3508_350876


namespace bob_sister_time_relation_l3508_350865

/-- Bob's current time for a mile in seconds -/
def bob_current_time : ℝ := 640

/-- The percentage improvement Bob needs to make -/
def improvement_percentage : ℝ := 9.062499999999996

/-- Bob's sister's time for a mile in seconds -/
def sister_time : ℝ := 582

/-- Theorem stating the relationship between Bob's current time, improvement percentage, and his sister's time -/
theorem bob_sister_time_relation :
  sister_time = bob_current_time * (1 - improvement_percentage / 100) := by
  sorry

end bob_sister_time_relation_l3508_350865


namespace square_ending_four_identical_digits_l3508_350895

theorem square_ending_four_identical_digits (n : ℕ) (d : ℕ) 
  (h1 : d ≤ 9) 
  (h2 : ∃ k : ℕ, n^2 = 10000 * k + d * 1111) : 
  d = 0 := by
sorry

end square_ending_four_identical_digits_l3508_350895


namespace stuffed_animals_count_l3508_350813

/-- The number of stuffed animals McKenna has -/
def mckenna_stuffed_animals : ℕ := 34

/-- The number of stuffed animals Kenley has -/
def kenley_stuffed_animals : ℕ := 2 * mckenna_stuffed_animals

/-- The number of stuffed animals Tenly has -/
def tenly_stuffed_animals : ℕ := kenley_stuffed_animals + 5

/-- The total number of stuffed animals the three girls have -/
def total_stuffed_animals : ℕ := mckenna_stuffed_animals + kenley_stuffed_animals + tenly_stuffed_animals

theorem stuffed_animals_count : total_stuffed_animals = 175 := by
  sorry

end stuffed_animals_count_l3508_350813


namespace heather_oranges_l3508_350810

theorem heather_oranges (initial : Real) (received : Real) :
  initial = 60.0 → received = 35.0 → initial + received = 95.0 := by
  sorry

end heather_oranges_l3508_350810


namespace tan_alpha_value_l3508_350826

theorem tan_alpha_value (α : ℝ) (h : Real.tan (α - 5 * Real.pi / 4) = 1 / 5) :
  Real.tan α = 3 / 2 := by
  sorry

end tan_alpha_value_l3508_350826


namespace janet_stickers_l3508_350823

theorem janet_stickers (S : ℕ) : 
  S > 2 ∧ 
  S % 5 = 2 ∧ 
  S % 11 = 2 ∧ 
  S % 13 = 2 ∧ 
  (∀ T : ℕ, T > 2 ∧ T % 5 = 2 ∧ T % 11 = 2 ∧ T % 13 = 2 → S ≤ T) → 
  S = 717 := by
sorry

end janet_stickers_l3508_350823


namespace specific_wall_rows_l3508_350824

/-- Represents a brick wall with a specific structure -/
structure BrickWall where
  totalBricks : ℕ
  bottomRowBricks : ℕ
  (total_positive : 0 < totalBricks)
  (bottom_positive : 0 < bottomRowBricks)
  (bottom_leq_total : bottomRowBricks ≤ totalBricks)

/-- Calculates the number of rows in a brick wall -/
def numberOfRows (wall : BrickWall) : ℕ :=
  sorry

/-- Theorem stating that a wall with 100 total bricks and 18 bricks in the bottom row has 8 rows -/
theorem specific_wall_rows :
  ∀ (wall : BrickWall),
    wall.totalBricks = 100 →
    wall.bottomRowBricks = 18 →
    numberOfRows wall = 8 :=
  sorry

end specific_wall_rows_l3508_350824


namespace polynomial_identity_sum_l3508_350820

theorem polynomial_identity_sum (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) :
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + a₁*x + d₁) * (x^2 + a₂*x + d₂) * (x^2 + a₃*x + d₃) * (x^2 - x + 1)) →
  a₁*d₁ + a₂*d₂ + a₃*d₃ = 0 := by
sorry

end polynomial_identity_sum_l3508_350820


namespace polar_curve_length_l3508_350867

noncomputable def curve_length (ρ : Real → Real) (φ₁ φ₂ : Real) : Real :=
  ∫ x in φ₁..φ₂, Real.sqrt (ρ x ^ 2 + (deriv ρ x) ^ 2)

theorem polar_curve_length :
  let ρ : Real → Real := fun φ ↦ 2 * (1 - Real.cos φ)
  curve_length ρ (-Real.pi) (-Real.pi/2) = -4 * Real.sqrt 2 := by
  sorry

end polar_curve_length_l3508_350867


namespace complex_equation_solution_l3508_350842

def complex_one_plus_i : ℂ := Complex.mk 1 1

theorem complex_equation_solution (a b : ℝ) : 
  let z : ℂ := complex_one_plus_i
  (z^2 + a*z + b) / (z^2 - z + 1) = 1 - Complex.I → a = -1 ∧ b = 2 := by
  sorry

end complex_equation_solution_l3508_350842


namespace sum_of_digits_9ab_l3508_350871

/-- The number of digits in a and b -/
def n : ℕ := 1984

/-- The integer a consisting of n nines in base 10 -/
def a : ℕ := (10^n - 1) / 9

/-- The integer b consisting of n fives in base 10 -/
def b : ℕ := (5 * (10^n - 1)) / 9

/-- Function to calculate the sum of digits of a natural number in base 10 -/
def sumOfDigits (k : ℕ) : ℕ :=
  if k < 10 then k else k % 10 + sumOfDigits (k / 10)

/-- Theorem stating that the sum of digits of 9ab is 27779 -/
theorem sum_of_digits_9ab : sumOfDigits (9 * a * b) = 27779 := by
  sorry

end sum_of_digits_9ab_l3508_350871


namespace movie_of_the_year_requirement_l3508_350836

theorem movie_of_the_year_requirement (total_members : ℕ) (fraction : ℚ) : total_members = 775 → fraction = 1/4 → ↑(⌈total_members * fraction⌉) = 194 := by
  sorry

end movie_of_the_year_requirement_l3508_350836


namespace competition_result_l3508_350847

structure Athlete where
  longJump : ℝ
  tripleJump : ℝ
  highJump : ℝ

def totalDistance (a : Athlete) : ℝ :=
  a.longJump + a.tripleJump + a.highJump

def isWinner (a : Athlete) : Prop :=
  totalDistance a = 22 * 3

theorem competition_result (x : ℝ) :
  let athlete1 := Athlete.mk x 30 7
  let athlete2 := Athlete.mk 24 34 8
  isWinner athlete2 ∧ ¬∃y, y = x ∧ isWinner (Athlete.mk y 30 7) := by
  sorry

end competition_result_l3508_350847


namespace lamp_cost_l3508_350806

/-- Proves the cost of the lamp given Daria's furniture purchase scenario -/
theorem lamp_cost (savings : ℕ) (couch_cost table_cost remaining_debt : ℕ) : 
  savings = 500 → 
  couch_cost = 750 → 
  table_cost = 100 → 
  remaining_debt = 400 → 
  ∃ (lamp_cost : ℕ), 
    lamp_cost = remaining_debt - (couch_cost + table_cost - savings) ∧ 
    lamp_cost = 50 := by
sorry

end lamp_cost_l3508_350806


namespace complement_intersection_theorem_l3508_350855

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set Nat := {3, 4, 5}
def B : Set Nat := {1, 3, 6}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {2, 7, 8} := by sorry

end complement_intersection_theorem_l3508_350855


namespace fraction_equality_l3508_350839

theorem fraction_equality : (1/3 - 1/4) / (1/2 - 1/3) = 1/2 := by
  sorry

end fraction_equality_l3508_350839


namespace twirly_tea_cups_capacity_l3508_350884

/-- The number of people that can fit in one teacup -/
def people_per_teacup : ℕ := 9

/-- The number of teacups on the ride -/
def number_of_teacups : ℕ := 7

/-- The total number of people that can ride at a time -/
def total_riders : ℕ := people_per_teacup * number_of_teacups

theorem twirly_tea_cups_capacity :
  total_riders = 63 := by sorry

end twirly_tea_cups_capacity_l3508_350884


namespace fraction_simplification_l3508_350844

theorem fraction_simplification :
  (5 - Real.sqrt 4) / (5 + Real.sqrt 4) = 3 / 7 :=
by
  sorry

end fraction_simplification_l3508_350844


namespace tabs_per_window_l3508_350869

theorem tabs_per_window (num_browsers : ℕ) (windows_per_browser : ℕ) (total_tabs : ℕ) :
  num_browsers = 2 →
  windows_per_browser = 3 →
  total_tabs = 60 →
  total_tabs / (num_browsers * windows_per_browser) = 10 :=
by sorry

end tabs_per_window_l3508_350869


namespace bus_tour_sales_l3508_350858

/-- Given a bus tour with senior and regular tickets, calculate the total sales amount. -/
theorem bus_tour_sales (total_tickets : ℕ) (senior_price regular_price : ℕ) (senior_tickets : ℕ) 
  (h1 : total_tickets = 65)
  (h2 : senior_price = 10)
  (h3 : regular_price = 15)
  (h4 : senior_tickets = 24)
  (h5 : senior_tickets ≤ total_tickets) :
  senior_tickets * senior_price + (total_tickets - senior_tickets) * regular_price = 855 := by
  sorry


end bus_tour_sales_l3508_350858


namespace arithmetic_iff_straight_line_l3508_350833

/-- A sequence of real numbers -/
def Sequence := ℕ+ → ℝ

/-- A sequence of points in 2D space -/
def PointSequence := ℕ+ → ℝ × ℝ

/-- Predicate for arithmetic sequences -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

/-- Predicate for points lying on a straight line -/
def on_straight_line (P : PointSequence) : Prop :=
  ∃ m b : ℝ, ∀ n : ℕ+, (P n).2 = m * (P n).1 + b

/-- Main theorem: equivalence between arithmetic sequence and points on a straight line -/
theorem arithmetic_iff_straight_line (a : Sequence) (P : PointSequence) :
  is_arithmetic a ↔ on_straight_line P :=
sorry

end arithmetic_iff_straight_line_l3508_350833


namespace derivative_symmetry_l3508_350832

/-- Given a function f(x) = ax^4 + bx^2 + c, if f'(1) = 2, then f'(-1) = -2 -/
theorem derivative_symmetry (a b c : ℝ) : 
  let f := fun (x : ℝ) => a * x^4 + b * x^2 + c
  let f' := fun (x : ℝ) => 4 * a * x^3 + 2 * b * x
  f' 1 = 2 → f' (-1) = -2 := by
  sorry

end derivative_symmetry_l3508_350832


namespace geometric_sum_example_l3508_350889

/-- Sum of a geometric sequence -/
def geometric_sum (a₀ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₀ * (1 - r^n) / (1 - r)

/-- Proof of the sum of the first eight terms of a specific geometric sequence -/
theorem geometric_sum_example : geometric_sum (1/3) (1/3) 8 = 3280/6561 := by
  sorry

end geometric_sum_example_l3508_350889


namespace find_a_value_l3508_350886

theorem find_a_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = a * x^3 + 9 * x^2 + 6 * x - 7) →
  (((fun x ↦ 3 * a * x^2 + 18 * x + 6) (-1)) = 4) →
  a = 16/3 := by
  sorry

end find_a_value_l3508_350886


namespace perpendicular_lines_from_perpendicular_planes_l3508_350883

structure Plane
structure Line

-- Define the perpendicular relationship between planes
def perp_planes (α β : Plane) : Prop := sorry

-- Define the intersection of two planes
def intersect_planes (α β : Plane) : Line := sorry

-- Define a line parallel to a plane
def parallel_line_plane (a : Line) (α : Plane) : Prop := sorry

-- Define a line perpendicular to a plane
def perp_line_plane (b : Line) (β : Plane) : Prop := sorry

-- Define a line perpendicular to another line
def perp_lines (b l : Line) : Prop := sorry

theorem perpendicular_lines_from_perpendicular_planes 
  (α β : Plane) (a b l : Line) 
  (h1 : perp_planes α β) 
  (h2 : intersect_planes α β = l) 
  (h3 : parallel_line_plane a α) 
  (h4 : perp_line_plane b β) : 
  perp_lines b l := sorry

end perpendicular_lines_from_perpendicular_planes_l3508_350883


namespace daughters_age_l3508_350845

theorem daughters_age (mother_age : ℕ) (daughter_age : ℕ) : 
  mother_age = 42 → 
  (mother_age + 9) = 3 * (daughter_age + 9) → 
  daughter_age = 8 := by
  sorry

end daughters_age_l3508_350845


namespace sin_n_eq_cos_390_l3508_350864

theorem sin_n_eq_cos_390 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.cos (390 * π / 180) →
  n = 60 ∨ n = 120 := by
sorry

end sin_n_eq_cos_390_l3508_350864


namespace complex_roots_theorem_l3508_350828

theorem complex_roots_theorem (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (a + 5 * Complex.I) * (b + 6 * Complex.I) = 9 + 61 * Complex.I →
  (a + 5 * Complex.I) + (b + 6 * Complex.I) = 12 + 11 * Complex.I →
  (a, b) = (9, 3) := by
  sorry

end complex_roots_theorem_l3508_350828


namespace solve_for_y_l3508_350892

theorem solve_for_y (x y z : ℝ) (h1 : x = 1) (h2 : z = 3) (h3 : x^2 * y * z - x * y * z^2 = 6) : y = -1/4 := by
  sorry

end solve_for_y_l3508_350892


namespace even_sum_difference_l3508_350898

def sum_even_range (a b : ℕ) : ℕ :=
  let n := (b - a) / 2 + 1
  n * (a + b) / 2

theorem even_sum_difference : sum_even_range 102 150 - sum_even_range 2 50 = 2500 := by
  sorry

end even_sum_difference_l3508_350898


namespace peters_situps_l3508_350801

theorem peters_situps (greg_situps : ℕ) (ratio : ℚ) : 
  greg_situps = 32 →
  ratio = 3 / 4 →
  ∃ peter_situps : ℕ, peter_situps * 4 = greg_situps * 3 ∧ peter_situps = 24 :=
by sorry

end peters_situps_l3508_350801


namespace triangle_angle_measure_l3508_350819

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (a * Real.cos B - b * Real.cos A = c) →
  (C = π / 5) →
  (B = 3 * π / 10) := by
sorry

end triangle_angle_measure_l3508_350819


namespace average_trees_planted_l3508_350821

def tree_data : List ℕ := [10, 8, 9, 9]

theorem average_trees_planted : 
  (List.sum tree_data) / (List.length tree_data : ℚ) = 9 := by
  sorry

end average_trees_planted_l3508_350821


namespace principal_amount_calculation_l3508_350843

/-- Given a principal amount and an interest rate, if increasing the rate by 1%
    results in an additional interest of 63 over 3 years, then the principal amount is 2100. -/
theorem principal_amount_calculation (P R : ℝ) (h : P * (R + 1) * 3 / 100 - P * R * 3 / 100 = 63) :
  P = 2100 := by
  sorry

end principal_amount_calculation_l3508_350843


namespace interest_difference_l3508_350856

/-- Calculates the difference between compound interest and simple interest -/
theorem interest_difference (principal : ℝ) (rate : ℝ) (time : ℝ) :
  let compound_interest := principal * (1 + rate)^time - principal
  let simple_interest := principal * rate * time
  principal = 6500 ∧ rate = 0.04 ∧ time = 2 →
  compound_interest - simple_interest = 9.40 := by sorry

end interest_difference_l3508_350856


namespace sarahs_brother_apples_l3508_350872

theorem sarahs_brother_apples (sarah_apples : ℝ) (ratio : ℝ) (brother_apples : ℝ) : 
  sarah_apples = 45.0 →
  sarah_apples = ratio * brother_apples →
  ratio = 5 →
  brother_apples = 9.0 := by
sorry

end sarahs_brother_apples_l3508_350872


namespace train_speed_problem_l3508_350870

theorem train_speed_problem (v : ℝ) : 
  v > 0 → -- The speed of the second train is positive
  (∃ t : ℝ, t > 0 ∧ -- There exists a positive time t
    16 * t + v * t = 444 ∧ -- Total distance traveled equals the distance between stations
    v * t = 16 * t + 60) -- The second train travels 60 km more than the first
  → v = 21 := by
sorry

end train_speed_problem_l3508_350870


namespace x_minus_y_equals_negative_twelve_l3508_350882

theorem x_minus_y_equals_negative_twelve (x y : ℝ) 
  (hx : 2 = 0.25 * x) (hy : 2 = 0.10 * y) : x - y = -12 := by
  sorry

end x_minus_y_equals_negative_twelve_l3508_350882


namespace vitya_older_probability_l3508_350868

/-- The number of days in June -/
def june_days : ℕ := 30

/-- The probability that Vitya is at least one day older than Masha -/
def probability_vitya_older (june_days : ℕ) : ℚ :=
  (june_days - 1).choose 2 / (june_days * june_days)

/-- Theorem stating the probability that Vitya is at least one day older than Masha -/
theorem vitya_older_probability :
  probability_vitya_older june_days = 29 / 60 := by
  sorry

end vitya_older_probability_l3508_350868


namespace arithmetic_calculation_l3508_350804

theorem arithmetic_calculation : 2 + 3 * 4 - 5 + 6 = 15 := by
  sorry

end arithmetic_calculation_l3508_350804


namespace total_cost_is_30_l3508_350861

def silverware_cost : ℝ := 20
def dinner_plates_cost_ratio : ℝ := 0.5

def total_cost : ℝ :=
  silverware_cost + (silverware_cost * dinner_plates_cost_ratio)

theorem total_cost_is_30 :
  total_cost = 30 := by sorry

end total_cost_is_30_l3508_350861


namespace matrix_is_own_inverse_l3508_350880

def A (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![4, -2; a, b]

theorem matrix_is_own_inverse (a b : ℚ) :
  A a b * A a b = 1 ↔ a = 15/2 ∧ b = -4 := by
  sorry

end matrix_is_own_inverse_l3508_350880


namespace street_number_painting_cost_l3508_350825

/-- Calculates the sum of digits for a given range of numbers in an arithmetic sequence -/
def sumDigits (start : ℕ) (diff : ℕ) (count : ℕ) : ℕ :=
  sorry

/-- Calculates the total cost of painting house numbers on a street -/
def totalCost (eastStart eastDiff westStart westDiff houseCount : ℕ) : ℕ :=
  sorry

theorem street_number_painting_cost :
  totalCost 5 5 2 4 25 = 88 :=
sorry

end street_number_painting_cost_l3508_350825


namespace plates_needed_is_38_l3508_350814

/-- The number of plates needed for a week given the specified eating patterns -/
def plates_needed : ℕ :=
  let days_with_son := 3
  let days_with_parents := 7 - days_with_son
  let people_with_son := 2
  let people_with_parents := 4
  let plates_per_person_with_son := 1
  let plates_per_person_with_parents := 2
  
  (days_with_son * people_with_son * plates_per_person_with_son) +
  (days_with_parents * people_with_parents * plates_per_person_with_parents)

theorem plates_needed_is_38 : plates_needed = 38 := by
  sorry

end plates_needed_is_38_l3508_350814


namespace ammonium_hydroxide_formation_l3508_350878

-- Define the chemicals involved in the reaction
inductive Chemical
| NH4Cl
| NaOH
| NH4OH
| NaCl

-- Define a function to represent the reaction
def reaction (nh4cl : ℚ) (naoh : ℚ) : ℚ × ℚ × ℚ × ℚ :=
  (nh4cl - min nh4cl naoh, naoh - min nh4cl naoh, min nh4cl naoh, min nh4cl naoh)

-- Theorem stating the result of the reaction
theorem ammonium_hydroxide_formation 
  (nh4cl_moles naoh_moles : ℚ) 
  (h1 : nh4cl_moles = 1) 
  (h2 : naoh_moles = 1) : 
  (reaction nh4cl_moles naoh_moles).2.1 = 1 := by
  sorry

-- Note: The theorem states that the third component of the reaction result
-- (which represents NH4OH) is equal to 1 when both input moles are 1.

end ammonium_hydroxide_formation_l3508_350878


namespace compound_interest_with_contributions_l3508_350866

theorem compound_interest_with_contributions
  (initial_amount : ℝ)
  (interest_rate : ℝ)
  (annual_contribution : ℝ)
  (years : ℕ)
  (h1 : initial_amount = 76800)
  (h2 : interest_rate = 0.125)
  (h3 : annual_contribution = 5000)
  (h4 : years = 2) :
  let amount_after_first_year := initial_amount * (1 + interest_rate)
  let total_after_first_year := amount_after_first_year + annual_contribution
  let amount_after_second_year := total_after_first_year * (1 + interest_rate)
  let final_amount := amount_after_second_year + annual_contribution
  final_amount = 107825 := by
  sorry

end compound_interest_with_contributions_l3508_350866


namespace orchid_rose_difference_l3508_350873

-- Define the initial and final counts of roses and orchids
def initial_roses : ℕ := 7
def initial_orchids : ℕ := 12
def final_roses : ℕ := 11
def final_orchids : ℕ := 20

-- Theorem to prove
theorem orchid_rose_difference :
  final_orchids - final_roses = 9 :=
by sorry

end orchid_rose_difference_l3508_350873


namespace log_equation_solution_l3508_350888

theorem log_equation_solution (x : ℝ) (h : Real.log 729 / Real.log (3 * x) = x) : x = 3 := by
  sorry

end log_equation_solution_l3508_350888


namespace solution_set_when_a_is_2_range_of_a_when_f_geq_1_l3508_350840

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * |x - 1| + |x - a|

-- Statement 1
theorem solution_set_when_a_is_2 :
  let f2 := f 2
  {x : ℝ | f2 x ≤ 4} = {x : ℝ | 0 ≤ x ∧ x ≤ 8/3} := by sorry

-- Statement 2
theorem range_of_a_when_f_geq_1 :
  {a : ℝ | a > 0 ∧ ∀ x, f a x ≥ 1} = {a : ℝ | a ≥ 2} := by sorry

end solution_set_when_a_is_2_range_of_a_when_f_geq_1_l3508_350840


namespace x_intercept_distance_l3508_350812

/-- Given two lines with slopes 4 and -2 intersecting at (8, 20),
    the distance between their x-intercepts is 15. -/
theorem x_intercept_distance (line1 line2 : ℝ → ℝ) : 
  (∀ x, line1 x = 4 * x - 12) →  -- Equation of line1
  (∀ x, line2 x = -2 * x + 36) →  -- Equation of line2
  line1 8 = 20 →  -- Intersection point
  line2 8 = 20 →  -- Intersection point
  |((36 : ℝ) / 2) - (12 / 4)| = 15 := by
  sorry


end x_intercept_distance_l3508_350812


namespace quadratic_always_positive_l3508_350896

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 5)*x - k + 8 > 0) ↔ k > -1 ∧ k < 7 := by
  sorry

end quadratic_always_positive_l3508_350896


namespace gym_guests_first_hour_l3508_350841

/-- The number of guests who entered the gym in the first hour -/
def first_hour_guests : ℕ := 50

/-- The total number of towels available -/
def total_towels : ℕ := 300

/-- The number of hours the gym is open -/
def open_hours : ℕ := 4

/-- The increase rate for the second hour -/
def second_hour_rate : ℚ := 1.2

/-- The increase rate for the third hour -/
def third_hour_rate : ℚ := 1.25

/-- The increase rate for the fourth hour -/
def fourth_hour_rate : ℚ := 4/3

/-- The number of towels that need to be washed at the end of the day -/
def towels_to_wash : ℕ := 285

theorem gym_guests_first_hour :
  first_hour_guests * (1 + second_hour_rate + second_hour_rate * third_hour_rate +
    second_hour_rate * third_hour_rate * fourth_hour_rate) = towels_to_wash :=
sorry

end gym_guests_first_hour_l3508_350841


namespace equation_solution_l3508_350838

theorem equation_solution :
  ∀ x : ℝ, (2*x - 3)^2 = (x - 2)^2 ↔ x = 1 ∨ x = 5/3 := by
sorry

end equation_solution_l3508_350838


namespace fried_green_tomatoes_l3508_350815

/-- Given that each tomato is cut into 8 slices and 20 tomatoes are needed to feed a family of 8 for a single meal, 
    prove that 20 slices are needed for a single person's meal. -/
theorem fried_green_tomatoes (slices_per_tomato : ℕ) (tomatoes_for_family : ℕ) (family_size : ℕ) 
  (h1 : slices_per_tomato = 8)
  (h2 : tomatoes_for_family = 20)
  (h3 : family_size = 8) :
  (slices_per_tomato * tomatoes_for_family) / family_size = 20 := by
  sorry

#check fried_green_tomatoes

end fried_green_tomatoes_l3508_350815


namespace polynomial_simplification_l3508_350891

theorem polynomial_simplification (m : ℝ) : 
  (∀ x y : ℝ, (2 * m * x^2 + 4 * x^2 + 3 * x + 1) - (6 * x^2 - 4 * y^2 + 3 * x) = 4 * y^2 + 1) ↔ 
  m = 1 := by
sorry

end polynomial_simplification_l3508_350891


namespace x_equals_5y_when_squared_difference_equal_l3508_350803

theorem x_equals_5y_when_squared_difference_equal
  (x y : ℕ) -- x and y are natural numbers
  (h : x^2 - 3*x = 25*y^2 - 15*y) -- given equation
  : x = 5*y := by
sorry

end x_equals_5y_when_squared_difference_equal_l3508_350803


namespace train_departure_time_l3508_350860

/-- Proves that the first train left Mumbai 2 hours before the meeting point -/
theorem train_departure_time 
  (first_train_speed : ℝ) 
  (second_train_speed : ℝ) 
  (time_difference : ℝ) 
  (meeting_distance : ℝ) 
  (h1 : first_train_speed = 45)
  (h2 : second_train_speed = 90)
  (h3 : time_difference = 1)
  (h4 : meeting_distance = 90) :
  ∃ (departure_time : ℝ), 
    departure_time = 2 ∧ 
    first_train_speed * (departure_time + time_difference) = 
    second_train_speed * time_difference ∧
    first_train_speed * departure_time = meeting_distance :=
by sorry


end train_departure_time_l3508_350860


namespace quadratic_triple_root_l3508_350805

/-- For a quadratic equation ax^2 + bx + c = 0, if one root is triple the other, 
    then 3b^2 = 16ac -/
theorem quadratic_triple_root (a b c : ℝ) (h : ∃ x y : ℝ, x ≠ 0 ∧ 
  a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) : 
  3 * b^2 = 16 * a * c := by
  sorry

end quadratic_triple_root_l3508_350805


namespace sum_of_roots_l3508_350834

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 1716 := by
sorry

end sum_of_roots_l3508_350834


namespace quadratic_equations_solutions_l3508_350808

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x ↦ x^2 - 4*x + 3 = 0
  let eq2 : ℝ → Prop := λ x ↦ x^2 - x - 3 = 0
  let sol1 : Set ℝ := {3, 1}
  let sol2 : Set ℝ := {(1 + Real.sqrt 13) / 2, (1 - Real.sqrt 13) / 2}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ x, eq1 x → x ∈ sol1) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ x, eq2 x → x ∈ sol2) := by
sorry

end quadratic_equations_solutions_l3508_350808


namespace bus_length_calculation_l3508_350885

/-- Calculates the length of a bus given its speed, the speed of a person moving in the opposite direction, and the time it takes for the bus to pass the person. -/
theorem bus_length_calculation (bus_speed : ℝ) (skater_speed : ℝ) (passing_time : ℝ) :
  bus_speed = 40 ∧ skater_speed = 8 ∧ passing_time = 1.125 →
  (bus_speed + skater_speed) * passing_time * (5 / 18) = 45 :=
by sorry

end bus_length_calculation_l3508_350885


namespace kopeck_ruble_exchange_l3508_350894

/-- Represents the denominations of coins available in kopecks -/
def Denominations : List ℕ := [1, 2, 5, 10, 20, 50, 100]

/-- Represents a valid coin exchange -/
def IsValidExchange (amount : ℕ) (coinCount : ℕ) : Prop :=
  ∃ (coins : List ℕ), 
    (coins.length = coinCount) ∧ 
    (coins.sum = amount) ∧
    (∀ c ∈ coins, c ∈ Denominations)

/-- The main theorem: if A kopecks can be exchanged with B coins,
    then B rubles can be exchanged with A coins -/
theorem kopeck_ruble_exchange 
  (A B : ℕ) 
  (h : IsValidExchange A B) : 
  IsValidExchange (100 * B) A := by
  sorry

#check kopeck_ruble_exchange

end kopeck_ruble_exchange_l3508_350894


namespace confucius_wine_consumption_l3508_350830

theorem confucius_wine_consumption :
  let wine_sequence : List ℚ := [1, 1, 1/2, 1/4, 1/8, 1/16]
  List.sum wine_sequence = 47/16 := by
  sorry

end confucius_wine_consumption_l3508_350830


namespace sally_pokemon_cards_l3508_350853

theorem sally_pokemon_cards (initial : ℕ) (dan_gift : ℕ) (sally_bought : ℕ) : 
  initial = 27 → dan_gift = 41 → sally_bought = 20 → 
  initial + dan_gift + sally_bought = 88 := by
sorry

end sally_pokemon_cards_l3508_350853


namespace amy_albums_count_l3508_350850

/-- The number of photos Amy uploaded to Facebook -/
def total_photos : ℕ := 180

/-- The number of photos in each album -/
def photos_per_album : ℕ := 20

/-- The number of albums Amy created -/
def num_albums : ℕ := total_photos / photos_per_album

theorem amy_albums_count : num_albums = 9 := by
  sorry

end amy_albums_count_l3508_350850


namespace triangle_side_range_l3508_350881

theorem triangle_side_range (a : ℝ) : 
  (3 : ℝ) > 0 ∧ (5 : ℝ) > 0 ∧ (1 - 2*a : ℝ) > 0 ∧
  3 + 5 > 1 - 2*a ∧
  3 + (1 - 2*a) > 5 ∧
  5 + (1 - 2*a) > 3 →
  -7/2 < a ∧ a < -1/2 := by
sorry

end triangle_side_range_l3508_350881


namespace max_value_xyz_l3508_350800

/-- Given real numbers x, y, and z that are non-negative and satisfy the equation
    2x + 3xy² + 2z = 36, the maximum value of x²y²z is 144. -/
theorem max_value_xyz (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
    (h_eq : 2*x + 3*x*y^2 + 2*z = 36) :
    x^2 * y^2 * z ≤ 144 :=
  sorry

end max_value_xyz_l3508_350800
