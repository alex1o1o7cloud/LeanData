import Mathlib

namespace two_segment_trip_avg_speed_l2604_260457

/-- Calculates the average speed for a two-segment trip -/
theorem two_segment_trip_avg_speed 
  (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) 
  (h1 : distance1 = 40) 
  (h2 : speed1 = 30) 
  (h3 : distance2 = 40) 
  (h4 : speed2 = 15) : 
  (distance1 + distance2) / ((distance1 / speed1) + (distance2 / speed2)) = 20 := by
  sorry

#check two_segment_trip_avg_speed

end two_segment_trip_avg_speed_l2604_260457


namespace one_neither_prime_nor_composite_l2604_260403

theorem one_neither_prime_nor_composite : 
  ¬(Nat.Prime 1) ∧ ¬(∃ a b : Nat, a > 1 ∧ b > 1 ∧ a * b = 1) := by
  sorry

end one_neither_prime_nor_composite_l2604_260403


namespace ruth_shared_apples_l2604_260426

/-- The number of apples Ruth shared with Peter -/
def apples_shared (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that Ruth shared 5 apples with Peter -/
theorem ruth_shared_apples : apples_shared 89 84 = 5 := by
  sorry

end ruth_shared_apples_l2604_260426


namespace allocation_schemes_count_l2604_260472

/-- Represents a student with their skills -/
structure Student where
  hasExcellentEnglish : Bool
  hasStrongComputer : Bool

/-- The total number of students -/
def totalStudents : Nat := 8

/-- The number of students with excellent English scores -/
def excellentEnglishCount : Nat := 2

/-- The number of students with strong computer skills -/
def strongComputerCount : Nat := 3

/-- The number of students to be allocated to each company -/
def studentsPerCompany : Nat := 4

/-- Calculates the number of valid allocation schemes -/
def countAllocationSchemes (students : List Student) : Nat :=
  sorry

/-- Theorem stating the number of valid allocation schemes -/
theorem allocation_schemes_count :
  ∀ (students : List Student),
    students.length = totalStudents →
    (students.filter (·.hasExcellentEnglish)).length = excellentEnglishCount →
    (students.filter (·.hasStrongComputer)).length = strongComputerCount →
    countAllocationSchemes students = 36 := by
  sorry

end allocation_schemes_count_l2604_260472


namespace sin_double_angle_special_case_l2604_260410

theorem sin_double_angle_special_case (φ : ℝ) :
  (7 : ℝ) / 13 + Real.sin φ = Real.cos φ →
  Real.sin (2 * φ) = 120 / 169 := by
sorry

end sin_double_angle_special_case_l2604_260410


namespace sum_squares_lengths_eq_k_squared_l2604_260463

/-- A regular k-gon inscribed in a unit circle -/
structure RegularKGon (k : ℕ) where
  (k_pos : k > 0)

/-- The sum of squares of lengths of all sides and diagonals of a regular k-gon -/
def sum_squares_lengths (k : ℕ) (P : RegularKGon k) : ℝ :=
  sorry

/-- Theorem: The sum of squares of lengths of all sides and diagonals of a regular k-gon
    inscribed in a unit circle is equal to k^2 -/
theorem sum_squares_lengths_eq_k_squared (k : ℕ) (P : RegularKGon k) :
  sum_squares_lengths k P = k^2 :=
sorry

end sum_squares_lengths_eq_k_squared_l2604_260463


namespace income_left_is_2_15_percent_l2604_260448

/-- Calculates the percentage of income left after one year given initial expenses and yearly changes. -/
def income_left_after_one_year (
  food_expense : ℝ)
  (education_expense : ℝ)
  (transportation_expense : ℝ)
  (medical_expense : ℝ)
  (rent_percentage_of_remaining : ℝ)
  (expense_increase_rate : ℝ)
  (income_increase_rate : ℝ) : ℝ :=
  let initial_expenses := food_expense + education_expense + transportation_expense + medical_expense
  let remaining_after_initial := 1 - initial_expenses
  let initial_rent := remaining_after_initial * rent_percentage_of_remaining
  let increased_expenses := initial_expenses * (1 + expense_increase_rate)
  let new_remaining := 1 - increased_expenses
  let new_rent := new_remaining * rent_percentage_of_remaining
  1 - (increased_expenses + new_rent)

/-- Theorem stating that given the specified conditions, the percentage of income left after one year is 2.15%. -/
theorem income_left_is_2_15_percent :
  income_left_after_one_year 0.35 0.25 0.15 0.10 0.80 0.05 0.10 = 0.0215 := by
  sorry

end income_left_is_2_15_percent_l2604_260448


namespace factory_production_l2604_260414

/-- Represents a machine in the factory -/
structure Machine where
  rate : Nat  -- shirts produced per minute
  time_yesterday : Nat  -- minutes worked yesterday
  time_today : Nat  -- minutes worked today

/-- Calculates the total number of shirts produced by all machines -/
def total_shirts (machines : List Machine) : Nat :=
  machines.foldl (fun acc m => acc + m.rate * (m.time_yesterday + m.time_today)) 0

/-- Theorem: Given the specified machines, the total number of shirts produced is 432 -/
theorem factory_production : 
  let machines : List Machine := [
    { rate := 6, time_yesterday := 12, time_today := 10 },  -- Machine A
    { rate := 8, time_yesterday := 10, time_today := 15 },  -- Machine B
    { rate := 5, time_yesterday := 20, time_today := 0 }    -- Machine C
  ]
  total_shirts machines = 432 := by
  sorry


end factory_production_l2604_260414


namespace tank_filling_time_l2604_260419

/-- The time required to fill a tank with different valve combinations -/
theorem tank_filling_time 
  (fill_time_xyz : Real) 
  (fill_time_xz : Real) 
  (fill_time_yz : Real) 
  (h1 : fill_time_xyz = 2)
  (h2 : fill_time_xz = 4)
  (h3 : fill_time_yz = 3) :
  let rate_x := 1 / fill_time_xz - 1 / fill_time_xyz
  let rate_y := 1 / fill_time_yz - 1 / fill_time_xyz
  1 / (rate_x + rate_y) = 2.4 := by
  sorry

end tank_filling_time_l2604_260419


namespace fred_marbles_l2604_260473

theorem fred_marbles (total : ℕ) (dark_blue : ℕ) (green : ℕ) (yellow : ℕ) (red : ℕ) : 
  total = 120 →
  dark_blue ≥ total / 3 →
  green = 10 →
  yellow = 5 →
  red = total - (dark_blue + green + yellow) →
  red = 65 := by
  sorry

end fred_marbles_l2604_260473


namespace red_shirt_pairs_l2604_260489

theorem red_shirt_pairs 
  (total_students : ℕ) 
  (green_students : ℕ) 
  (red_students : ℕ) 
  (total_pairs : ℕ) 
  (green_green_pairs : ℕ) : 
  total_students = 132 →
  green_students = 64 →
  red_students = 68 →
  total_pairs = 66 →
  green_green_pairs = 28 →
  ∃ (red_red_pairs : ℕ), red_red_pairs = 30 :=
by sorry

end red_shirt_pairs_l2604_260489


namespace man_in_dark_probability_l2604_260443

/-- The number of revolutions per minute made by the searchlight -/
def revolutions_per_minute : ℝ := 3

/-- The probability that a man will stay in the dark for at least some seconds -/
def probability_in_dark : ℝ := 0.25

/-- Theorem stating the probability of a man staying in the dark -/
theorem man_in_dark_probability :
  probability_in_dark = 0.25 := by sorry

end man_in_dark_probability_l2604_260443


namespace pentagon_area_l2604_260469

/-- The area of a specific pentagon -/
theorem pentagon_area : 
  ∀ (s₁ s₂ s₃ s₄ s₅ : ℝ) (θ : ℝ),
  s₁ = 18 → s₂ = 20 → s₃ = 27 → s₄ = 24 → s₅ = 20 →
  θ = Real.pi / 2 →
  ∃ (A : ℝ),
  A = (1/2 * s₁ * s₂) + (1/2 * (s₃ + s₄) * s₅) ∧
  A = 690 :=
by sorry

end pentagon_area_l2604_260469


namespace sqrt_of_nine_l2604_260440

theorem sqrt_of_nine : Real.sqrt 9 = 3 := by
  sorry

end sqrt_of_nine_l2604_260440


namespace counterexample_exists_l2604_260445

theorem counterexample_exists : ∃ (a b c : ℝ), 0 < a ∧ a < b ∧ b < c ∧ a ≥ b * c := by
  sorry

end counterexample_exists_l2604_260445


namespace gcd_of_lcm_and_ratio_l2604_260450

theorem gcd_of_lcm_and_ratio (X Y : ℕ) : 
  X ≠ 0 → Y ≠ 0 → 
  lcm X Y = 180 → 
  ∃ (k : ℕ), X = 2 * k ∧ Y = 5 * k → 
  gcd X Y = 18 := by
sorry

end gcd_of_lcm_and_ratio_l2604_260450


namespace brock_cookies_proof_l2604_260490

/-- Represents the number of cookies Brock bought -/
def brock_cookies : ℕ := 7

theorem brock_cookies_proof (total_cookies : ℕ) (stone_cookies : ℕ) (remaining_cookies : ℕ) 
  (h1 : total_cookies = 5 * 12)
  (h2 : stone_cookies = 2 * 12)
  (h3 : remaining_cookies = 15)
  (h4 : total_cookies = stone_cookies + 3 * brock_cookies + remaining_cookies) :
  brock_cookies = 7 := by
  sorry

end brock_cookies_proof_l2604_260490


namespace complex_magnitude_product_l2604_260487

theorem complex_magnitude_product : Complex.abs ((12 - 9*Complex.I) * (8 + 15*Complex.I)) = 255 := by
  sorry

end complex_magnitude_product_l2604_260487


namespace line_direction_vector_l2604_260411

/-- The direction vector of a line y = (2x - 6)/5 parameterized as [x, y] = [4, 0] + t * d,
    where t is the distance between [x, y] and [4, 0] for x ≥ 4. -/
theorem line_direction_vector :
  ∃ (d : ℝ × ℝ),
    (∀ (x y t : ℝ), x ≥ 4 →
      y = (2 * x - 6) / 5 →
      (x, y) = (4, 0) + t • d →
      t = Real.sqrt ((x - 4)^2 + y^2)) →
    d = (5 / Real.sqrt 29, 10 / Real.sqrt 29) := by
  sorry

end line_direction_vector_l2604_260411


namespace work_completion_time_l2604_260495

theorem work_completion_time (a b : ℝ) 
  (h1 : a + b = 1 / 16)  -- A and B together finish in 16 days
  (h2 : a = 1 / 32)      -- A alone finishes in 32 days
  : 1 / b = 32 :=        -- B alone finishes in 32 days
by sorry

end work_completion_time_l2604_260495


namespace card_value_decrease_l2604_260437

theorem card_value_decrease (v : ℝ) (h : v > 0) : 
  let value_after_first_year := v * (1 - 0.5)
  let value_after_second_year := value_after_first_year * (1 - 0.1)
  let total_decrease := (v - value_after_second_year) / v
  total_decrease = 0.55
:= by sorry

end card_value_decrease_l2604_260437


namespace common_roots_of_polynomials_l2604_260475

theorem common_roots_of_polynomials :
  let f (x : ℝ) := x^4 + 2*x^3 - x^2 - 2*x - 3
  let g (x : ℝ) := x^4 + 3*x^3 + x^2 - 4*x - 6
  let r₁ := (-1 + Real.sqrt 13) / 2
  let r₂ := (-1 - Real.sqrt 13) / 2
  (f r₁ = 0 ∧ f r₂ = 0) ∧ (g r₁ = 0 ∧ g r₂ = 0) :=
by
  sorry

end common_roots_of_polynomials_l2604_260475


namespace complex_cube_root_sum_l2604_260496

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the theorem
theorem complex_cube_root_sum (a b : ℝ) (h : i^3 = a - b*i) : a + b = 1 := by
  sorry

end complex_cube_root_sum_l2604_260496


namespace f_odd_iff_l2604_260422

/-- A function f is odd if f(-x) = -f(x) for all x in the domain of f. -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = x|x + a| + b -/
def f (a b : ℝ) (x : ℝ) : ℝ :=
  x * |x + a| + b

/-- The necessary and sufficient condition for f to be an odd function -/
theorem f_odd_iff (a b : ℝ) :
  IsOdd (f a b) ↔ a^2 + b^2 = 0 := by
  sorry

end f_odd_iff_l2604_260422


namespace photo_selection_choices_l2604_260477

-- Define the number of items to choose from
def n : ℕ := 10

-- Define the possible numbers of items to be chosen
def k₁ : ℕ := 5
def k₂ : ℕ := 6

-- Define the combination function
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem photo_selection_choices : 
  combination n k₁ + combination n k₂ = 462 := by sorry

end photo_selection_choices_l2604_260477


namespace ground_school_cost_proof_l2604_260447

/-- Represents the cost of a private pilot course -/
def total_cost : ℕ := 1275

/-- Represents the additional cost of the flight portion compared to the ground school portion -/
def flight_additional_cost : ℕ := 625

/-- Represents the cost of the flight portion -/
def flight_cost : ℕ := 950

/-- Represents the cost of the ground school portion -/
def ground_school_cost : ℕ := total_cost - flight_cost

theorem ground_school_cost_proof : ground_school_cost = 325 := by
  sorry

end ground_school_cost_proof_l2604_260447


namespace f_composition_value_l2604_260442

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.log (-x) else Real.tan x

theorem f_composition_value : f (f (3 * Real.pi / 4)) = 0 := by
  sorry

end f_composition_value_l2604_260442


namespace tan_660_degrees_l2604_260451

theorem tan_660_degrees : Real.tan (660 * π / 180) = -Real.sqrt 3 := by
  sorry

end tan_660_degrees_l2604_260451


namespace family_ages_solution_l2604_260415

/-- Represents the current ages of Jennifer, Jordana, and James -/
structure FamilyAges where
  jennifer : ℕ
  jordana : ℕ
  james : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.jennifer + 20 = 40 ∧
  ages.jordana + 20 = 2 * (ages.jennifer + 20) ∧
  ages.james + 20 = (ages.jennifer + 20) + (ages.jordana + 20) - 10

theorem family_ages_solution :
  ∃ (ages : FamilyAges), satisfiesConditions ages ∧ ages.jordana = 60 ∧ ages.james = 90 :=
sorry

end family_ages_solution_l2604_260415


namespace ball_probabilities_l2604_260406

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := 6

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 4

/-- Represents the number of red balls in the bag -/
def red_balls : ℕ := 2

/-- Represents the number of balls drawn -/
def drawn_balls : ℕ := 2

/-- Calculates the probability of drawing two red balls -/
def prob_two_red : ℚ := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))

/-- Calculates the probability of drawing at least one red ball -/
def prob_at_least_one_red : ℚ := 1 - (white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1))

theorem ball_probabilities :
  prob_two_red = 1/15 ∧ prob_at_least_one_red = 3/5 := by sorry

end ball_probabilities_l2604_260406


namespace pentagon_area_relationship_l2604_260479

/-- Represents the areas of different parts of a pentagon -/
structure PentagonAreas where
  x : ℝ  -- Area of the smaller similar pentagon
  y : ℝ  -- Area of one type of surrounding region
  z : ℝ  -- Area of another type of surrounding region
  total : ℝ  -- Total area of the larger pentagon

/-- Theorem about the relationship between areas in a specially divided pentagon -/
theorem pentagon_area_relationship (p : PentagonAreas) 
  (h_positive : p.x > 0 ∧ p.y > 0 ∧ p.z > 0 ∧ p.total > 0)
  (h_similar : ∃ (k : ℝ), k > 0 ∧ p.x = k^2 * p.total)
  (h_total : p.total = p.x + 5*p.y + 5*p.z) :
  p.y = p.z ∧ 
  p.y = (p.total - p.x) / 10 ∧
  p.total = p.x + 10*p.y := by
  sorry


end pentagon_area_relationship_l2604_260479


namespace ellipse_and_line_intersection_l2604_260401

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - Real.sqrt 3

theorem ellipse_and_line_intersection :
  -- Conditions
  (∀ x y, ellipse_C x y → (x = 0 ∧ y = 0) → False) →  -- center at origin
  (∃ c > 0, ∀ x y, ellipse_C x y → x^2 / 4 + y^2 / c^2 = 1) →  -- standard form
  (ellipse_C 1 (Real.sqrt 3 / 2)) →  -- point on ellipse
  -- Conclusions
  (∀ x y, ellipse_C x y ↔ x^2 / 4 + y^2 = 1) ∧  -- equation of C
  (∃ x₁ x₂ y₁ y₂ : ℝ,
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (8/5)^2) :=  -- length of AB
by sorry

end ellipse_and_line_intersection_l2604_260401


namespace lcm_of_10_and_21_l2604_260488

theorem lcm_of_10_and_21 : Nat.lcm 10 21 = 210 := by
  sorry

end lcm_of_10_and_21_l2604_260488


namespace simplification_and_exponent_sum_l2604_260474

-- Define the original expression
def original_expression (x y z : ℝ) : ℝ := (40 * x^5 * y^9 * z^14)^(1/3)

-- Define the simplified expression
def simplified_expression (x y z : ℝ) : ℝ := 2 * x * y * z^3 * (5 * x^2 * z^5)^(1/3)

-- Define the sum of exponents outside the radical
def sum_of_exponents : ℕ := 1 + 1 + 3

-- Theorem statement
theorem simplification_and_exponent_sum (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) : 
  original_expression x y z = simplified_expression x y z ∧ 
  sum_of_exponents = 5 := by
  sorry

end simplification_and_exponent_sum_l2604_260474


namespace smallest_x_satisfying_equation_l2604_260493

theorem smallest_x_satisfying_equation : 
  ∃ (x : ℝ), x = -6 ∧ 
  (∀ y : ℝ, (y^2 - y - 30) / (y - 6) = 2 / (y + 4) → y ≥ x) ∧
  (x^2 - x - 30) / (x - 6) = 2 / (x + 4) := by
  sorry

end smallest_x_satisfying_equation_l2604_260493


namespace jose_profit_share_l2604_260409

/-- Calculates the share of profit for an investor given their investment amount, 
    investment duration, total investment-months, and total profit. -/
def shareOfProfit (investment : ℕ) (duration : ℕ) (totalInvestmentMonths : ℕ) (totalProfit : ℕ) : ℚ :=
  (investment * duration : ℚ) / totalInvestmentMonths * totalProfit

theorem jose_profit_share 
  (tom_investment : ℕ) (tom_duration : ℕ) 
  (jose_investment : ℕ) (jose_duration : ℕ) 
  (total_profit : ℕ) : 
  tom_investment = 30000 → 
  tom_duration = 12 → 
  jose_investment = 45000 → 
  jose_duration = 10 → 
  total_profit = 45000 → 
  shareOfProfit jose_investment jose_duration 
    (tom_investment * tom_duration + jose_investment * jose_duration) total_profit = 25000 := by
  sorry

#check jose_profit_share

end jose_profit_share_l2604_260409


namespace gumball_sale_revenue_l2604_260424

theorem gumball_sale_revenue (num_gumballs : ℕ) (price_per_gumball : ℕ) : 
  num_gumballs = 4 → price_per_gumball = 8 → num_gumballs * price_per_gumball = 32 := by
  sorry

end gumball_sale_revenue_l2604_260424


namespace isosceles_triangle_proof_l2604_260465

theorem isosceles_triangle_proof (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_equation : 2 * Real.sin A * Real.cos B = Real.sin C) : A = B :=
sorry

end isosceles_triangle_proof_l2604_260465


namespace total_hockey_games_l2604_260436

/-- The number of hockey games in a season -/
def hockey_games_in_season (games_per_month : ℕ) (months_in_season : ℕ) : ℕ :=
  games_per_month * months_in_season

/-- Theorem stating that there are 182 hockey games in the season -/
theorem total_hockey_games :
  hockey_games_in_season 13 14 = 182 := by
  sorry

end total_hockey_games_l2604_260436


namespace prove_income_expenditure_ratio_l2604_260449

def income_expenditure_ratio (income savings : ℕ) : Prop :=
  ∃ (expenditure : ℕ),
    savings = income - expenditure ∧
    income * 8 = expenditure * 15

theorem prove_income_expenditure_ratio :
  income_expenditure_ratio 15000 7000 := by
  sorry

end prove_income_expenditure_ratio_l2604_260449


namespace distance_sum_equals_radii_sum_l2604_260498

/-- An acute-angled triangle with its circumscribed and inscribed circles -/
structure AcuteTriangle where
  /-- The radius of the circumscribed circle -/
  R : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The distance from the circumcenter to side a -/
  da : ℝ
  /-- The distance from the circumcenter to side b -/
  db : ℝ
  /-- The distance from the circumcenter to side c -/
  dc : ℝ
  /-- The triangle is acute-angled -/
  acute : R > 0
  /-- The radii and distances are positive -/
  positive : r > 0 ∧ da > 0 ∧ db > 0 ∧ dc > 0

/-- The sum of distances from the circumcenter to the sides equals the sum of circumradius and inradius -/
theorem distance_sum_equals_radii_sum (t : AcuteTriangle) : t.da + t.db + t.dc = t.R + t.r := by
  sorry

end distance_sum_equals_radii_sum_l2604_260498


namespace greatest_multiple_of_5_and_7_less_than_700_l2604_260491

theorem greatest_multiple_of_5_and_7_less_than_700 :
  (∃ n : ℕ, n * 5 * 7 < 700 ∧ 
    ∀ m : ℕ, m * 5 * 7 < 700 → m * 5 * 7 ≤ n * 5 * 7) →
  (∃ n : ℕ, n * 5 * 7 = 695) :=
by sorry

end greatest_multiple_of_5_and_7_less_than_700_l2604_260491


namespace only_one_true_statement_l2604_260430

/-- Two lines are non-coincident -/
def NonCoincidentLines (m n : Line) : Prop :=
  m ≠ n

/-- Two planes are non-coincident -/
def NonCoincidentPlanes (α β : Plane) : Prop :=
  α ≠ β

/-- A line is parallel to a plane -/
def LineParallelToPlane (l : Line) (p : Plane) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def LinePerpendicularToPlane (l : Line) (p : Plane) : Prop :=
  sorry

/-- Two lines are parallel -/
def ParallelLines (l1 l2 : Line) : Prop :=
  sorry

/-- Two lines intersect -/
def LinesIntersect (l1 l2 : Line) : Prop :=
  sorry

/-- Two planes are perpendicular -/
def PerpendicularPlanes (p1 p2 : Plane) : Prop :=
  sorry

/-- Two lines are perpendicular -/
def PerpendicularLines (l1 l2 : Line) : Prop :=
  sorry

/-- Projection of a line onto a plane -/
def ProjectionOntoPlane (l : Line) (p : Plane) : Line :=
  sorry

theorem only_one_true_statement
  (m n : Line) (α β : Plane)
  (h_lines : NonCoincidentLines m n)
  (h_planes : NonCoincidentPlanes α β) :
  (LineParallelToPlane m α ∧ LineParallelToPlane n α → ¬LinesIntersect m n) ∨
  (LinePerpendicularToPlane m α ∧ LinePerpendicularToPlane n α → ParallelLines m n) ∨
  (PerpendicularPlanes α β ∧ PerpendicularLines m n ∧ LinePerpendicularToPlane m α → LinePerpendicularToPlane n β) ∨
  (PerpendicularLines (ProjectionOntoPlane m α) (ProjectionOntoPlane n α) → PerpendicularLines m n) :=
by sorry

end only_one_true_statement_l2604_260430


namespace cube_surface_area_increase_l2604_260499

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_area := 6 * L^2
  let new_edge_length := 1.5 * L
  let new_area := 6 * new_edge_length^2
  (new_area - original_area) / original_area * 100 = 125 :=
by sorry

end cube_surface_area_increase_l2604_260499


namespace average_age_is_35_l2604_260435

-- Define the ages of John, Mary, and Tonya
def john_age : ℕ := 30
def mary_age : ℕ := 15
def tonya_age : ℕ := 60

-- State the theorem
theorem average_age_is_35 :
  (john_age = 2 * mary_age) ∧  -- John is twice as old as Mary
  (2 * john_age = tonya_age) ∧  -- John is half as old as Tonya
  (tonya_age = 60) →  -- Tonya is 60 years old
  (john_age + mary_age + tonya_age) / 3 = 35 := by
  sorry

#check average_age_is_35

end average_age_is_35_l2604_260435


namespace chores_ratio_l2604_260485

/-- Proves that the ratio of time spent on other chores to vacuuming is 3:1 -/
theorem chores_ratio (vacuum_time other_chores_time total_time : ℕ) : 
  vacuum_time = 3 → 
  total_time = 12 → 
  other_chores_time = total_time - vacuum_time →
  (other_chores_time : ℚ) / vacuum_time = 3 := by
  sorry

end chores_ratio_l2604_260485


namespace inverse_proportion_decrease_l2604_260413

theorem inverse_proportion_decrease (x y : ℝ) (k : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = k) :
  let x_new := 1.1 * x
  let y_new := k / x_new
  (y - y_new) / y = 1 / 11 := by sorry

end inverse_proportion_decrease_l2604_260413


namespace conic_section_focus_l2604_260453

/-- The conic section defined by parametric equations x = t^2 and y = 2t -/
def conic_section (t : ℝ) : ℝ × ℝ := (t^2, 2*t)

/-- The focus of the conic section -/
def focus : ℝ × ℝ := (1, 0)

/-- Theorem: The focus of the conic section defined by parametric equations x = t^2 and y = 2t is (1, 0) -/
theorem conic_section_focus :
  ∀ t : ℝ, ∃ a : ℝ, a > 0 ∧ (conic_section t).2^2 = 4 * a * (conic_section t).1 ∧ focus = (a, 0) :=
sorry

end conic_section_focus_l2604_260453


namespace triangle_perimeter_l2604_260407

theorem triangle_perimeter (a b c : ℝ) : 
  a = 2 → (b - 2)^2 + |c - 3| = 0 → a + b + c = 7 := by sorry

end triangle_perimeter_l2604_260407


namespace correct_division_result_l2604_260418

theorem correct_division_result (dividend : ℕ) 
  (h1 : dividend / 87 = 24) 
  (h2 : dividend % 87 = 0) : 
  dividend / 36 = 58 := by
sorry

end correct_division_result_l2604_260418


namespace specific_tetrahedron_volume_l2604_260454

/-- The volume of a tetrahedron given its edge lengths -/
def tetrahedron_volume (ab ac ad cd bd bc : ℝ) : ℝ :=
  -- Definition to be filled
  sorry

/-- Theorem: The volume of the specific tetrahedron is 48 cubic units -/
theorem specific_tetrahedron_volume :
  tetrahedron_volume 6 7 8 9 10 11 = 48 := by
  sorry

end specific_tetrahedron_volume_l2604_260454


namespace t_shape_perimeter_l2604_260482

/-- A "T" shaped figure composed of unit squares -/
structure TShape :=
  (top_row : Fin 3 → Unit)
  (bottom_column : Fin 2 → Unit)

/-- The perimeter of a TShape -/
def perimeter (t : TShape) : ℕ :=
  14

theorem t_shape_perimeter :
  ∀ (t : TShape), perimeter t = 14 :=
by
  sorry

end t_shape_perimeter_l2604_260482


namespace max_profit_theorem_profit_range_theorem_l2604_260467

/-- The daily sales volume as a function of the selling price -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 1000

/-- The daily profit as a function of the selling price -/
def profit (x : ℝ) : ℝ := (x - 40) * (sales_volume x)

/-- The theorem stating the maximum profit and the price at which it occurs -/
theorem max_profit_theorem :
  ∃ (max_profit : ℝ) (optimal_price : ℝ),
    (∀ x : ℝ, 50 ≤ x ∧ x ≤ 65 → profit x ≤ max_profit) ∧
    profit optimal_price = max_profit ∧
    optimal_price = 65 ∧
    max_profit = 8750 :=
sorry

/-- The theorem stating the range of prices for which the profit is at least 8000 -/
theorem profit_range_theorem :
  ∀ x : ℝ, (60 ≤ x ∧ x ≤ 65) ↔ profit x ≥ 8000 :=
sorry

end max_profit_theorem_profit_range_theorem_l2604_260467


namespace divisibility_equation_solutions_l2604_260444

theorem divisibility_equation_solutions (n x y z t : ℕ+) :
  (n ^ x.val ∣ n ^ y.val + n ^ z.val) ∧ (n ^ y.val + n ^ z.val = n ^ t.val) →
  ((n = 2 ∧ y = x ∧ z = x + 1 ∧ t = x + 2) ∨
   (n = 3 ∧ y = x ∧ z = x ∧ t = x + 1)) :=
by sorry

end divisibility_equation_solutions_l2604_260444


namespace system_solution_equation_solution_l2604_260492

-- System of equations
theorem system_solution :
  ∃! (x y : ℝ), x + 2*y = 9 ∧ 3*x - 2*y = 3 ∧ x = 3 ∧ y = 3 := by sorry

-- Single equation
theorem equation_solution :
  ∃! (x : ℝ), x ≠ 3 ∧ (2-x)/(x-3) + 3 = 2/(3-x) ∧ x = 5/2 := by sorry

end system_solution_equation_solution_l2604_260492


namespace remainder_thirteen_power_thirteen_plus_thirteen_l2604_260432

theorem remainder_thirteen_power_thirteen_plus_thirteen (n : ℕ) :
  (13^13 + 13) % 14 = 12 := by
  sorry

end remainder_thirteen_power_thirteen_plus_thirteen_l2604_260432


namespace exp_two_log_five_equals_twentyfive_l2604_260461

theorem exp_two_log_five_equals_twentyfive : 
  Real.exp (2 * Real.log 5) = 25 := by sorry

end exp_two_log_five_equals_twentyfive_l2604_260461


namespace quadratic_inequality_solution_l2604_260455

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2 := by
  sorry

end quadratic_inequality_solution_l2604_260455


namespace sin_40_tan_10_minus_sqrt_3_l2604_260421

theorem sin_40_tan_10_minus_sqrt_3 : 
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end sin_40_tan_10_minus_sqrt_3_l2604_260421


namespace smallest_perimeter_circle_circle_center_on_L_l2604_260420

-- Define the points A and B
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (-1, 4)

-- Define the line L
def L (x y : ℝ) : Prop := 2 * x - y - 4 = 0

-- Define a general circle equation
def isCircle (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

-- Define a circle passing through two points
def circlePassingThrough (h k r : ℝ) : Prop :=
  isCircle h k r A.1 A.2 ∧ isCircle h k r B.1 B.2

-- Theorem for the smallest perimeter circle
theorem smallest_perimeter_circle :
  ∃ (h k r : ℝ), circlePassingThrough h k r ∧
  isCircle h k r = fun x y => x^2 + (y - 1)^2 = 10 :=
sorry

-- Theorem for the circle with center on line L
theorem circle_center_on_L :
  ∃ (h k r : ℝ), circlePassingThrough h k r ∧
  L h k ∧
  isCircle h k r = fun x y => (x - 3)^2 + (y - 2)^2 = 20 :=
sorry

end smallest_perimeter_circle_circle_center_on_L_l2604_260420


namespace number_problem_l2604_260470

theorem number_problem (x : ℚ) (h : (7/8) * x = 28) : (x + 16) * (5/16) = 15 := by
  sorry

end number_problem_l2604_260470


namespace f_maximized_at_three_tenths_l2604_260483

/-- The probability that exactly k out of n items are defective, given probability p for each item -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability that exactly 3 out of 10 items are defective -/
def f (p : ℝ) : ℝ := binomial_probability 10 3 p

/-- The theorem stating that f(p) is maximized when p = 3/10 -/
theorem f_maximized_at_three_tenths (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  ∃ (max_p : ℝ), max_p = 3/10 ∧ ∀ q, 0 < q → q < 1 → f q ≤ f max_p :=
sorry

end f_maximized_at_three_tenths_l2604_260483


namespace number_calculation_l2604_260478

theorem number_calculation (x : ℚ) : (30 / 100 * x = 25 / 100 * 50) → x = 125 / 3 := by
  sorry

end number_calculation_l2604_260478


namespace journey_distance_l2604_260405

theorem journey_distance : ∀ (D : ℝ),
  (1/5 : ℝ) * D + (2/3 : ℝ) * D + 12 = D →
  D = 90 := by sorry

end journey_distance_l2604_260405


namespace bills_final_money_is_411_l2604_260427

/-- Calculates the final amount of money Bill has after all transactions and expenses. -/
def bills_final_money : ℝ :=
  let merchant_a_sale := 8 * 9
  let merchant_b_sale := 15 * 11
  let merchant_c_sale := 25 * 8
  let passerby_sale := 12 * 7
  let total_income := merchant_a_sale + merchant_b_sale + merchant_c_sale + passerby_sale
  let fine := 80
  let protection_cost := 30
  let total_expenses := fine + protection_cost
  total_income - total_expenses

/-- Theorem stating that Bill's final amount of money is $411. -/
theorem bills_final_money_is_411 : bills_final_money = 411 := by
  sorry

end bills_final_money_is_411_l2604_260427


namespace combined_mass_of_individuals_l2604_260429

/-- The density of water in kg/m³ -/
def water_density : ℝ := 1000

/-- The length of the boat in meters -/
def boat_length : ℝ := 4

/-- The breadth of the boat in meters -/
def boat_breadth : ℝ := 3

/-- The depth the boat sinks when the first person gets on, in meters -/
def first_person_depth : ℝ := 0.01

/-- The additional depth the boat sinks when the second person gets on, in meters -/
def second_person_depth : ℝ := 0.02

/-- Calculates the mass of water displaced by the boat sinking to a given depth -/
def water_mass (depth : ℝ) : ℝ :=
  boat_length * boat_breadth * depth * water_density

theorem combined_mass_of_individuals :
  water_mass (first_person_depth + second_person_depth) = 360 := by
  sorry

end combined_mass_of_individuals_l2604_260429


namespace zero_in_interval_l2604_260431

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log (1/2) + x - a

theorem zero_in_interval (a : ℝ) :
  a ∈ Set.Ioo 1 3 →
  ∃ x ∈ Set.Ioo 2 8, f a x = 0 ∧
  ¬(∀ a : ℝ, (∃ x ∈ Set.Ioo 2 8, f a x = 0) → a ∈ Set.Ioo 1 3) :=
by sorry

end zero_in_interval_l2604_260431


namespace mn_length_l2604_260494

/-- Triangle XYZ with given side lengths -/
structure Triangle (X Y Z : ℝ × ℝ) where
  xy_length : dist X Y = 130
  xz_length : dist X Z = 112
  yz_length : dist Y Z = 125

/-- L is the intersection of angle bisector of X with YZ -/
def L (X Y Z : ℝ × ℝ) : ℝ × ℝ := sorry

/-- K is the intersection of angle bisector of Y with XZ -/
def K (X Y Z : ℝ × ℝ) : ℝ × ℝ := sorry

/-- M is the foot of the perpendicular from Z to YK -/
def M (X Y Z : ℝ × ℝ) : ℝ × ℝ := sorry

/-- N is the foot of the perpendicular from Z to XL -/
def N (X Y Z : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The main theorem -/
theorem mn_length (X Y Z : ℝ × ℝ) (t : Triangle X Y Z) : 
  dist (M X Y Z) (N X Y Z) = 53.5 := by sorry

end mn_length_l2604_260494


namespace expansion_equals_cube_l2604_260400

theorem expansion_equals_cube : 16^3 + 3*(16^2)*2 + 3*16*(2^2) + 2^3 = 5832 := by
  sorry

end expansion_equals_cube_l2604_260400


namespace uniform_rod_weight_l2604_260459

/-- Represents the weight of a uniform rod -/
def rod_weight (length : ℝ) (weight_per_meter : ℝ) : ℝ :=
  length * weight_per_meter

/-- Theorem: For a uniform rod where 9 m weighs 34.2 kg, 11.25 m of the same rod weighs 42.75 kg -/
theorem uniform_rod_weight :
  ∀ (weight_per_meter : ℝ),
    rod_weight 9 weight_per_meter = 34.2 →
    rod_weight 11.25 weight_per_meter = 42.75 := by
  sorry

end uniform_rod_weight_l2604_260459


namespace larger_number_proof_l2604_260434

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 2342) (h3 : L = 9 * S + 23) : L = 2624 := by
  sorry

end larger_number_proof_l2604_260434


namespace matinee_ticket_cost_l2604_260433

/-- The cost of a matinee ticket in dollars -/
def matinee_cost : ℚ := 5

/-- The cost of an evening ticket in dollars -/
def evening_cost : ℚ := 7

/-- The cost of an opening night ticket in dollars -/
def opening_night_cost : ℚ := 10

/-- The cost of a bucket of popcorn in dollars -/
def popcorn_cost : ℚ := 10

/-- The number of matinee customers -/
def matinee_customers : ℕ := 32

/-- The number of evening customers -/
def evening_customers : ℕ := 40

/-- The number of opening night customers -/
def opening_night_customers : ℕ := 58

/-- The total revenue in dollars -/
def total_revenue : ℚ := 1670

theorem matinee_ticket_cost :
  matinee_cost * matinee_customers +
  evening_cost * evening_customers +
  opening_night_cost * opening_night_customers +
  popcorn_cost * ((matinee_customers + evening_customers + opening_night_customers) / 2) =
  total_revenue :=
sorry

end matinee_ticket_cost_l2604_260433


namespace inequalities_comparison_l2604_260428

theorem inequalities_comparison (a b : ℝ) (h : a > b) : (a - 3 > b - 3) ∧ (-4*a < -4*b) := by
  sorry

end inequalities_comparison_l2604_260428


namespace triangle_division_ratio_l2604_260425

/-- Given a triangle ABC, this theorem proves that if point F divides side AC in the ratio 2:3,
    G is the midpoint of BF, and E is the point of intersection of side BC and AG,
    then E divides side BC in the ratio 2:5. -/
theorem triangle_division_ratio (A B C F G E : ℝ × ℝ) : 
  (∃ k : ℝ, F = A + (2/5 : ℝ) • (C - A)) →  -- F divides AC in ratio 2:3
  G = B + (1/2 : ℝ) • (F - B) →              -- G is midpoint of BF
  (∃ t : ℝ, E = B + t • (C - B) ∧ 
            E = A + t • (G - A)) →           -- E is intersection of BC and AG
  (∃ s : ℝ, E = B + (2/7 : ℝ) • (C - B)) :=   -- E divides BC in ratio 2:5
by sorry


end triangle_division_ratio_l2604_260425


namespace sequence_property_l2604_260408

theorem sequence_property (a : ℕ → ℤ) 
  (h : ∀ n : ℕ, a 1 + a (2 * n - 1) = 4 * n - 6) : 
  a 7 = 11 := by
sorry

end sequence_property_l2604_260408


namespace machinery_expenditure_l2604_260446

theorem machinery_expenditure (total : ℝ) (raw_materials : ℝ) (cash_percentage : ℝ) :
  total = 137500 →
  raw_materials = 80000 →
  cash_percentage = 0.20 →
  ∃ machinery : ℝ,
    machinery = 30000 ∧
    raw_materials + machinery + (cash_percentage * total) = total :=
by sorry

end machinery_expenditure_l2604_260446


namespace function_c_injective_l2604_260412

theorem function_c_injective (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + 2 / a = b + 2 / b → a = b := by sorry

end function_c_injective_l2604_260412


namespace fabric_sale_meters_l2604_260486

-- Define the price per meter in kopecks
def price_per_meter : ℕ := 436

-- Define the maximum revenue in kopecks
def max_revenue : ℕ := 50000

-- Define a predicate for valid revenue
def valid_revenue (x : ℕ) : Prop :=
  (price_per_meter * x) % 1000 = 728 ∧
  price_per_meter * x ≤ max_revenue

-- Theorem statement
theorem fabric_sale_meters :
  ∃ (x : ℕ), valid_revenue x ∧ x = 98 := by sorry

end fabric_sale_meters_l2604_260486


namespace quadratic_roots_product_l2604_260484

theorem quadratic_roots_product (p q P Q : ℝ) (α β γ δ : ℝ) 
  (h1 : α^2 + p*α + q = 0)
  (h2 : β^2 + p*β + q = 0)
  (h3 : γ^2 + P*γ + Q = 0)
  (h4 : δ^2 + P*δ + Q = 0) :
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = P^2 * q - P * p * Q + Q^2 := by
  sorry

end quadratic_roots_product_l2604_260484


namespace x_squared_positive_necessary_not_sufficient_l2604_260460

theorem x_squared_positive_necessary_not_sufficient :
  (∀ x : ℝ, x > 0 → x^2 > 0) ∧
  (∃ x : ℝ, x^2 > 0 ∧ x ≤ 0) := by
  sorry

end x_squared_positive_necessary_not_sufficient_l2604_260460


namespace min_eating_time_is_23_5_l2604_260497

/-- Represents the eating rates and constraints for Amy and Ben -/
structure EatingProblem where
  total_carrots : ℕ
  total_muffins : ℕ
  wait_time : ℕ
  amy_carrot_rate : ℕ
  amy_muffin_rate : ℕ
  ben_carrot_rate : ℕ
  ben_muffin_rate : ℕ

/-- Calculates the minimum time to eat all food given the problem constraints -/
def min_eating_time (problem : EatingProblem) : ℚ :=
  sorry

/-- Theorem stating that the minimum eating time for the given problem is 23.5 minutes -/
theorem min_eating_time_is_23_5 : 
  let problem : EatingProblem := {
    total_carrots := 1000
    total_muffins := 1000
    wait_time := 5
    amy_carrot_rate := 40
    amy_muffin_rate := 70
    ben_carrot_rate := 60
    ben_muffin_rate := 30
  }
  min_eating_time problem = 47/2 := by
  sorry

end min_eating_time_is_23_5_l2604_260497


namespace even_function_implies_m_equals_one_l2604_260452

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- The function f(x) = x^4 + (m - 1)x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^4 + (m - 1) * x + 1

/-- If f(x) = x^4 + (m - 1)x + 1 is an even function, then m = 1 -/
theorem even_function_implies_m_equals_one (m : ℝ) : IsEven (f m) → m = 1 := by
  sorry

end even_function_implies_m_equals_one_l2604_260452


namespace schedule_five_courses_nine_periods_l2604_260466

/-- The number of ways to schedule courses -/
def schedule_ways (n_courses n_periods : ℕ) : ℕ :=
  Nat.choose n_periods n_courses * Nat.factorial n_courses

/-- Theorem stating the number of ways to schedule 5 courses in 9 periods -/
theorem schedule_five_courses_nine_periods :
  schedule_ways 5 9 = 15120 := by
  sorry

end schedule_five_courses_nine_periods_l2604_260466


namespace arithmetic_sum_specific_l2604_260468

/-- Sum of arithmetic sequence with given parameters -/
def arithmetic_sum (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) : ℤ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: The sum of the arithmetic sequence with first term -45, last term 0, and common difference 3 is -360 -/
theorem arithmetic_sum_specific : arithmetic_sum (-45) 0 3 = -360 := by
  sorry

end arithmetic_sum_specific_l2604_260468


namespace water_left_proof_l2604_260480

-- Define the initial amount of water
def initial_water : ℚ := 7/2

-- Define the amount of water used
def water_used : ℚ := 9/4

-- Theorem statement
theorem water_left_proof : initial_water - water_used = 5/4 := by
  sorry

end water_left_proof_l2604_260480


namespace max_exchanges_theorem_l2604_260417

/-- Represents a student with a height -/
structure Student where
  height : ℕ

/-- Represents a circle of students -/
def StudentCircle := List Student

/-- Condition for a student to switch places -/
def canSwitch (s₁ s₂ : Student) (prevHeight : ℕ) : Prop :=
  s₁.height > s₂.height ∧ s₂.height ≤ prevHeight

/-- The maximum number of exchanges possible -/
def maxExchanges (n : ℕ) : ℕ := n * (n - 1) * (n - 2) / 6

/-- Theorem stating the maximum number of exchanges -/
theorem max_exchanges_theorem (n : ℕ) (circle : StudentCircle) :
  (circle.length = n) →
  (∀ i j, i < j → (circle.get i).height < (circle.get j).height) →
  (∀ exchanges, exchanges ≤ maxExchanges n) := by
  sorry

end max_exchanges_theorem_l2604_260417


namespace absolute_value_inequality_l2604_260416

theorem absolute_value_inequality (x : ℝ) : 
  (1 ≤ |x - 2| ∧ |x - 2| ≤ 7) ↔ ((-5 ≤ x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x ≤ 9)) := by
  sorry

end absolute_value_inequality_l2604_260416


namespace sheila_picnic_probability_l2604_260439

/-- The probability of Sheila attending the picnic given the weather conditions and transport strike possibility. -/
theorem sheila_picnic_probability :
  let p_rain : ℝ := 0.5
  let p_sunny : ℝ := 1 - p_rain
  let p_attend_if_rain : ℝ := 0.25
  let p_attend_if_sunny : ℝ := 0.8
  let p_transport_strike : ℝ := 0.1
  let p_attend : ℝ := (p_rain * p_attend_if_rain + p_sunny * p_attend_if_sunny) * (1 - p_transport_strike)
  p_attend = 0.4725 := by
  sorry

end sheila_picnic_probability_l2604_260439


namespace least_multiple_of_next_three_primes_after_5_l2604_260438

def next_three_primes_after_5 : List Nat := [7, 11, 13]

theorem least_multiple_of_next_three_primes_after_5 :
  (∀ p ∈ next_three_primes_after_5, Nat.Prime p) →
  (∀ p ∈ next_three_primes_after_5, p > 5) →
  (∀ n < 1001, ∃ p ∈ next_three_primes_after_5, ¬(p ∣ n)) →
  (∀ p ∈ next_three_primes_after_5, p ∣ 1001) :=
by sorry

end least_multiple_of_next_three_primes_after_5_l2604_260438


namespace apple_pile_count_l2604_260462

theorem apple_pile_count : ∃! n : ℕ,
  50 < n ∧ n < 70 ∧
  n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧
  n % 1 = 0 ∧ n % 2 = 0 ∧ n % 4 = 0 ∧ n % 6 = 0 ∧ n % 10 = 0 ∧
  n % 12 = 0 ∧ n % 15 = 0 ∧ n % 20 = 0 ∧ n % 30 = 0 ∧ n % 60 = 0 ∧
  n = 60 := by
sorry

end apple_pile_count_l2604_260462


namespace fibonacci_6_l2604_260464

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_6 : fibonacci 5 = 8 := by
  sorry

end fibonacci_6_l2604_260464


namespace river_current_speed_l2604_260481

/-- The speed of a river's current given a swimmer's performance -/
theorem river_current_speed 
  (swimmer_still_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : swimmer_still_speed = 3)
  (h2 : distance = 8)
  (h3 : time = 5) :
  swimmer_still_speed - (distance / time) = (1.4 : ℝ) := by
sorry

end river_current_speed_l2604_260481


namespace rational_segment_existence_l2604_260471

theorem rational_segment_existence (f : ℚ → ℤ) :
  ∃ a b : ℚ, f a + f b ≤ 2 * f ((a + b) / 2) := by
  sorry

end rational_segment_existence_l2604_260471


namespace sandy_average_book_price_l2604_260456

/-- The average price Sandy paid per book given her purchases from two shops -/
theorem sandy_average_book_price (books1 : ℕ) (price1 : ℚ) (books2 : ℕ) (price2 : ℚ) 
  (h1 : books1 = 65)
  (h2 : price1 = 1380)
  (h3 : books2 = 55)
  (h4 : price2 = 900) :
  (price1 + price2) / (books1 + books2 : ℚ) = 19 := by
  sorry

end sandy_average_book_price_l2604_260456


namespace sequence_properties_l2604_260441

/-- Definition of an arithmetic sequence -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Definition of a geometric sequence -/
def geometric_seq (g : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, g (n + 1) = g n * q

/-- Main theorem -/
theorem sequence_properties
  (a g : ℕ → ℝ)
  (ha : arithmetic_seq a)
  (hg : geometric_seq g)
  (h1 : a 1 = 1)
  (h2 : g 1 = 1)
  (h3 : a 2 = g 2)
  (h4 : a 2 ≠ 1)
  (h5 : ∃ m : ℕ, m > 3 ∧ a m = g 3) :
  (∃ m : ℕ, m > 3 ∧
    (∃ d q : ℝ, d = m - 3 ∧ q = m - 2 ∧
      (∀ n : ℕ, a (n + 1) = a n + d) ∧
      (∀ n : ℕ, g (n + 1) = g n * q))) ∧
  (∃ k : ℕ, a k = g 4) ∧
  (∀ j : ℕ, ∃ k : ℕ, g (j + 1) = a k) :=
sorry

end sequence_properties_l2604_260441


namespace quadratic_inequality_solution_set_l2604_260458

theorem quadratic_inequality_solution_set (a b c : ℝ) (h1 : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c < 0) → (a < 0 ∧ b^2 - 4*a*c < 0) :=
by sorry

end quadratic_inequality_solution_set_l2604_260458


namespace complex_magnitude_l2604_260404

theorem complex_magnitude (z : ℂ) (h : z + Complex.abs z = 2 + 8 * Complex.I) : Complex.abs z = 17 := by
  sorry

end complex_magnitude_l2604_260404


namespace element_in_intersection_complement_l2604_260423

theorem element_in_intersection_complement (S : Type) (A B : Set S) (a : S) :
  Set.Nonempty A →
  Set.Nonempty B →
  A ⊂ Set.univ →
  B ⊂ Set.univ →
  a ∈ A →
  a ∉ B →
  a ∈ A ∩ (Set.univ \ B) :=
by sorry

end element_in_intersection_complement_l2604_260423


namespace route_time_proof_l2604_260402

/-- Proves that the time to run a 5-mile route one way is 1 hour, given the round trip average speed and return speed. -/
theorem route_time_proof (route_length : ℝ) (avg_speed : ℝ) (return_speed : ℝ) 
  (h1 : route_length = 5)
  (h2 : avg_speed = 8)
  (h3 : return_speed = 20) :
  let t := (2 * route_length / avg_speed - route_length / return_speed)
  t = 1 := by sorry

end route_time_proof_l2604_260402


namespace negation_equivalence_l2604_260476

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0) := by
  sorry

end negation_equivalence_l2604_260476
