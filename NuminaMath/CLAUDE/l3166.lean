import Mathlib

namespace NUMINAMATH_CALUDE_car_distance_l3166_316665

theorem car_distance (train_speed : ℝ) (car_speed_ratio : ℝ) (time_minutes : ℝ) : 
  train_speed = 120 →
  car_speed_ratio = 2/3 →
  time_minutes = 15 →
  (car_speed_ratio * train_speed) * (time_minutes / 60) = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_l3166_316665


namespace NUMINAMATH_CALUDE_sixth_game_score_l3166_316680

theorem sixth_game_score (scores : List ℕ) (mean : ℚ) : 
  scores.length = 7 ∧
  scores = [69, 68, 70, 61, 74, 65, 74] ∧
  mean = 67.9 ∧
  (∃ x : ℕ, (scores.sum + x) / 8 = mean) →
  ∃ x : ℕ, x = 62 ∧ (scores.sum + x) / 8 = mean := by
sorry

end NUMINAMATH_CALUDE_sixth_game_score_l3166_316680


namespace NUMINAMATH_CALUDE_average_headcount_l3166_316677

-- Define the list of spring term headcounts
def spring_headcounts : List Nat := [11000, 10200, 10800, 11300]

-- Define the number of terms
def num_terms : Nat := 4

-- Theorem to prove the average headcount
theorem average_headcount :
  (spring_headcounts.sum / num_terms : ℚ) = 10825 := by
  sorry

end NUMINAMATH_CALUDE_average_headcount_l3166_316677


namespace NUMINAMATH_CALUDE_scrap_cookie_radius_l3166_316618

theorem scrap_cookie_radius 
  (R : ℝ) 
  (r : ℝ) 
  (n : ℕ) 
  (h1 : R = 3.5) 
  (h2 : r = 1) 
  (h3 : n = 9) : 
  ∃ (x : ℝ), x^2 = R^2 * π - n * r^2 * π ∧ x = Real.sqrt 3.25 := by
  sorry

end NUMINAMATH_CALUDE_scrap_cookie_radius_l3166_316618


namespace NUMINAMATH_CALUDE_necklace_count_l3166_316603

/-- The number of unique necklaces made from 5 red and 2 blue beads -/
def unique_necklaces : ℕ := 3

/-- The number of red beads in each necklace -/
def red_beads : ℕ := 5

/-- The number of blue beads in each necklace -/
def blue_beads : ℕ := 2

/-- The total number of beads in each necklace -/
def total_beads : ℕ := red_beads + blue_beads

/-- Theorem stating that the number of unique necklaces is 3 -/
theorem necklace_count : unique_necklaces = 3 := by sorry

end NUMINAMATH_CALUDE_necklace_count_l3166_316603


namespace NUMINAMATH_CALUDE_cyclist_distance_l3166_316630

/-- Proves that a cyclist traveling at 18 km/hr for 2 minutes and 30 seconds covers a distance of 750 meters. -/
theorem cyclist_distance (speed : ℝ) (time_min : ℝ) (time_sec : ℝ) (distance : ℝ) :
  speed = 18 →
  time_min = 2 →
  time_sec = 30 →
  distance = speed * (time_min / 60 + time_sec / 3600) * 1000 →
  distance = 750 := by
sorry


end NUMINAMATH_CALUDE_cyclist_distance_l3166_316630


namespace NUMINAMATH_CALUDE_circle_parabola_height_difference_l3166_316632

/-- The height difference between the center of a circle and its points of tangency with the parabola y = 2x^2 -/
theorem circle_parabola_height_difference (a : ℝ) : 
  ∃ (b r : ℝ), 
    (∀ x y : ℝ, y = 2 * x^2 → x^2 + (y - b)^2 = r^2 → x = a ∨ x = -a) →
    (b - 2 * a^2 = 1/4 - a^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_parabola_height_difference_l3166_316632


namespace NUMINAMATH_CALUDE_x_value_l3166_316643

theorem x_value : ∀ (x y z w : ℤ), 
  x = y + 5 →
  y = z + 10 →
  z = w + 20 →
  w = 80 →
  x = 115 := by
sorry

end NUMINAMATH_CALUDE_x_value_l3166_316643


namespace NUMINAMATH_CALUDE_sum_of_squared_pairs_l3166_316606

theorem sum_of_squared_pairs (a b c d : ℝ) : 
  (a^4 - 24*a^3 + 50*a^2 - 35*a + 10 = 0) →
  (b^4 - 24*b^3 + 50*b^2 - 35*b + 10 = 0) →
  (c^4 - 24*c^3 + 50*c^2 - 35*c + 10 = 0) →
  (d^4 - 24*d^3 + 50*d^2 - 35*d + 10 = 0) →
  (a+b)^2 + (b+c)^2 + (c+d)^2 + (d+a)^2 = 541 := by sorry

end NUMINAMATH_CALUDE_sum_of_squared_pairs_l3166_316606


namespace NUMINAMATH_CALUDE_shortest_path_on_sphere_intersection_l3166_316688

/-- The shortest path on a sphere's surface between the two most distant points of its intersection with a plane --/
theorem shortest_path_on_sphere_intersection (R d : ℝ) (h1 : R = 2) (h2 : d = 1) :
  let r := Real.sqrt (R^2 - d^2)
  let θ := 2 * Real.arccos (d / R)
  θ / (2 * Real.pi) * (2 * Real.pi * r) = Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_shortest_path_on_sphere_intersection_l3166_316688


namespace NUMINAMATH_CALUDE_min_value_of_f_l3166_316649

-- Define the function
def f (y : ℝ) : ℝ := 3 * y^2 - 18 * y + 11

-- State the theorem
theorem min_value_of_f :
  ∃ (y_min : ℝ), ∀ (y : ℝ), f y ≥ f y_min ∧ f y_min = -16 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3166_316649


namespace NUMINAMATH_CALUDE_probability_x_plus_y_leq_6_l3166_316695

/-- The probability that x + y ≤ 6 when (x, y) is randomly selected from a rectangle where 0 ≤ x ≤ 4 and 0 ≤ y ≤ 5 -/
theorem probability_x_plus_y_leq_6 :
  let rectangle_area : ℝ := 4 * 5
  let favorable_area : ℝ := 15
  favorable_area / rectangle_area = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_leq_6_l3166_316695


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_root_l3166_316666

theorem quadratic_equation_unique_root (b c : ℝ) :
  (∀ x : ℝ, 3 * x^2 + b * x + c = 0 ↔ x = -4) →
  b = 24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_root_l3166_316666


namespace NUMINAMATH_CALUDE_tommys_quarters_l3166_316633

/-- Tommy's coin collection problem -/
theorem tommys_quarters (P D N Q : ℕ) 
  (dimes_pennies : D = P + 10)
  (nickels_dimes : N = 2 * D)
  (pennies_quarters : P = 10 * Q)
  (total_nickels : N = 100) : Q = 4 := by
  sorry

end NUMINAMATH_CALUDE_tommys_quarters_l3166_316633


namespace NUMINAMATH_CALUDE_binomial_square_example_l3166_316658

theorem binomial_square_example : 34^2 + 2*(34*5) + 5^2 = 1521 := by sorry

end NUMINAMATH_CALUDE_binomial_square_example_l3166_316658


namespace NUMINAMATH_CALUDE_total_amount_calculation_l3166_316600

-- Define the given parameters
def interest_rate : ℚ := 8 / 100
def time_period : ℕ := 2
def compound_interest : ℚ := 2828.80

-- Define the compound interest formula
def compound_interest_formula (P : ℚ) : ℚ :=
  P * (1 + interest_rate) ^ time_period - P

-- Define the total amount formula
def total_amount (P : ℚ) : ℚ :=
  P + compound_interest

-- Theorem statement
theorem total_amount_calculation :
  ∃ P : ℚ, compound_interest_formula P = compound_interest ∧
           total_amount P = 19828.80 :=
by sorry

end NUMINAMATH_CALUDE_total_amount_calculation_l3166_316600


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l3166_316631

-- Define the sets S and T
def S : Set ℝ := {x : ℝ | x < -5 ∨ x > 5}
def T : Set ℝ := {x : ℝ | -7 < x ∧ x < 3}

-- State the theorem
theorem set_intersection_theorem : S ∩ T = {x : ℝ | -7 < x ∧ x < -5} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l3166_316631


namespace NUMINAMATH_CALUDE_hallie_monday_tips_l3166_316628

/-- Represents Hallie's work and earnings over three days --/
structure WaitressEarnings where
  hourly_rate : ℝ
  monday_hours : ℝ
  tuesday_hours : ℝ
  wednesday_hours : ℝ
  tuesday_tips : ℝ
  wednesday_tips : ℝ
  total_earnings : ℝ

/-- Calculates Hallie's tips on Monday given her work schedule and earnings --/
def monday_tips (e : WaitressEarnings) : ℝ :=
  e.total_earnings -
  (e.hourly_rate * (e.monday_hours + e.tuesday_hours + e.wednesday_hours)) -
  e.tuesday_tips - e.wednesday_tips

/-- Theorem stating that Hallie's tips on Monday were $18 --/
theorem hallie_monday_tips (e : WaitressEarnings)
  (h1 : e.hourly_rate = 10)
  (h2 : e.monday_hours = 7)
  (h3 : e.tuesday_hours = 5)
  (h4 : e.wednesday_hours = 7)
  (h5 : e.tuesday_tips = 12)
  (h6 : e.wednesday_tips = 20)
  (h7 : e.total_earnings = 240) :
  monday_tips e = 18 := by
  sorry

end NUMINAMATH_CALUDE_hallie_monday_tips_l3166_316628


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l3166_316698

theorem empty_solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 1| + |x - 2| ≤ a^2 + a + 1)) → 
  -1 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l3166_316698


namespace NUMINAMATH_CALUDE_solve_amusement_park_problem_l3166_316611

def amusement_park_problem (ticket_price : ℕ) (weekday_visitors : ℕ) (saturday_visitors : ℕ) (total_revenue : ℕ) : Prop :=
  let weekday_total := weekday_visitors * 5
  let sunday_visitors := (total_revenue - ticket_price * (weekday_total + saturday_visitors)) / ticket_price
  sunday_visitors = 300

theorem solve_amusement_park_problem :
  amusement_park_problem 3 100 200 3000 := by
  sorry

end NUMINAMATH_CALUDE_solve_amusement_park_problem_l3166_316611


namespace NUMINAMATH_CALUDE_roots_are_imaginary_l3166_316661

theorem roots_are_imaginary (m : ℝ) : 
  (∃ x y : ℂ, x^2 - 4*m*x + 5*m^2 + 2 = 0 ∧ y^2 - 4*m*y + 5*m^2 + 2 = 0 ∧ x*y = 9) →
  (∃ a b : ℝ, a ≠ 0 ∧ (∀ z : ℂ, z^2 - 4*m*z + 5*m^2 + 2 = 0 → ∃ r : ℝ, z = Complex.mk r (a*r + b) ∨ z = Complex.mk r (-a*r - b))) :=
by sorry

end NUMINAMATH_CALUDE_roots_are_imaginary_l3166_316661


namespace NUMINAMATH_CALUDE_total_apples_is_33_l3166_316686

/-- The number of apples picked by each person -/
structure ApplePickers where
  mike : ℕ
  nancy : ℕ
  keith : ℕ
  jennifer : ℕ
  tom : ℕ
  stacy : ℕ

/-- The total number of apples picked -/
def total_apples (pickers : ApplePickers) : ℕ :=
  pickers.mike + pickers.nancy + pickers.keith + pickers.jennifer + pickers.tom + pickers.stacy

/-- Theorem stating that the total number of apples picked is 33 -/
theorem total_apples_is_33 (pickers : ApplePickers) 
    (h_mike : pickers.mike = 7)
    (h_nancy : pickers.nancy = 3)
    (h_keith : pickers.keith = 6)
    (h_jennifer : pickers.jennifer = 5)
    (h_tom : pickers.tom = 8)
    (h_stacy : pickers.stacy = 4) : 
  total_apples pickers = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_is_33_l3166_316686


namespace NUMINAMATH_CALUDE_f_periodic_l3166_316607

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.tan (x / 2) + 1

theorem f_periodic (a : ℝ) (h : f (-a) = 11) : f (2 * Real.pi + a) = -9 := by
  sorry

end NUMINAMATH_CALUDE_f_periodic_l3166_316607


namespace NUMINAMATH_CALUDE_election_total_votes_l3166_316679

/-- An election with two candidates -/
structure Election :=
  (totalValidVotes : ℕ)
  (losingCandidatePercentage : ℚ)
  (voteDifference : ℕ)
  (invalidVotes : ℕ)

/-- The total number of polled votes in an election -/
def totalPolledVotes (e : Election) : ℕ :=
  e.totalValidVotes + e.invalidVotes

/-- Theorem stating the total number of polled votes in the given election -/
theorem election_total_votes (e : Election) 
  (h1 : e.losingCandidatePercentage = 45/100)
  (h2 : e.voteDifference = 9000)
  (h3 : e.invalidVotes = 83)
  : totalPolledVotes e = 90083 := by
  sorry

#eval totalPolledVotes { totalValidVotes := 90000, losingCandidatePercentage := 45/100, voteDifference := 9000, invalidVotes := 83 }

end NUMINAMATH_CALUDE_election_total_votes_l3166_316679


namespace NUMINAMATH_CALUDE_football_cost_l3166_316621

/-- The cost of a football given the total cost of a football and baseball, and the cost of the baseball. -/
theorem football_cost (total_cost baseball_cost : ℚ) : 
  total_cost = 20 - (4 + 5/100) → 
  baseball_cost = 6 + 81/100 → 
  total_cost - baseball_cost = 9 + 14/100 := by
sorry

end NUMINAMATH_CALUDE_football_cost_l3166_316621


namespace NUMINAMATH_CALUDE_absolute_value_theorem_l3166_316612

theorem absolute_value_theorem (a : ℝ) (h : a = -1) : |a + 3| = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_theorem_l3166_316612


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_15_l3166_316690

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |> List.sum

theorem units_digit_factorial_sum_15 :
  units_digit (factorial_sum 15) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_15_l3166_316690


namespace NUMINAMATH_CALUDE_derivative_sin_squared_minus_cos_squared_l3166_316617

theorem derivative_sin_squared_minus_cos_squared (x : ℝ) : 
  deriv (λ x => Real.sin x ^ 2 - Real.cos x ^ 2) x = 2 * Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_squared_minus_cos_squared_l3166_316617


namespace NUMINAMATH_CALUDE_fruit_vendor_lemons_sold_l3166_316671

/-- Proves that a fruit vendor who sold 5 dozens of avocados and a total of 90 fruits sold 2.5 dozens of lemons -/
theorem fruit_vendor_lemons_sold (total_fruits : ℕ) (avocado_dozens : ℕ) (lemon_dozens : ℚ) : 
  total_fruits = 90 → avocado_dozens = 5 → lemon_dozens = 2.5 → 
  total_fruits = 12 * avocado_dozens + 12 * lemon_dozens := by
  sorry

#check fruit_vendor_lemons_sold

end NUMINAMATH_CALUDE_fruit_vendor_lemons_sold_l3166_316671


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l3166_316692

/-- The curve function f(x) = x^2 + 2x - 2 --/
def f (x : ℝ) : ℝ := x^2 + 2*x - 2

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 2*x + 2

theorem tangent_parallel_to_x_axis :
  ∃ (x y : ℝ), f x = y ∧ f' x = 0 ∧ x = -1 ∧ y = -3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l3166_316692


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3166_316639

/-- An arithmetic sequence with common difference 3 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 3

/-- a_2, a_4, and a_8 form a geometric sequence -/
def geometric_subseq (a : ℕ → ℝ) : Prop :=
  (a 4) ^ 2 = a 2 * a 8

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  arithmetic_seq a → geometric_subseq a → a 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3166_316639


namespace NUMINAMATH_CALUDE_annas_number_l3166_316623

theorem annas_number : ∃ x : ℚ, 5 * ((3 * x + 20) - 5) = 200 ∧ x = 25 / 3 := by sorry

end NUMINAMATH_CALUDE_annas_number_l3166_316623


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l3166_316610

theorem like_terms_exponent_sum (m n : ℕ) : 
  (∃ (x y : ℝ), 3 * x^(2*m) * y^3 = -2 * x^2 * y^n) → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l3166_316610


namespace NUMINAMATH_CALUDE_combined_salaries_l3166_316637

/-- The problem of calculating combined salaries -/
theorem combined_salaries 
  (salary_C : ℕ) 
  (average_salary : ℕ) 
  (num_individuals : ℕ) 
  (h1 : salary_C = 11000)
  (h2 : average_salary = 8200)
  (h3 : num_individuals = 5) :
  average_salary * num_individuals - salary_C = 30000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_l3166_316637


namespace NUMINAMATH_CALUDE_mode_most_effective_l3166_316652

/-- Represents different statistical measures -/
inductive StatisticalMeasure
  | Variance
  | Mean
  | Median
  | Mode

/-- Represents a shoe model -/
structure ShoeModel where
  id : Nat
  sales : Nat

/-- Represents a shoe store -/
structure ShoeStore where
  models : List ShoeModel
  
/-- Determines the most effective statistical measure for increasing sales -/
def mostEffectiveMeasure (store : ShoeStore) : StatisticalMeasure :=
  StatisticalMeasure.Mode

/-- Theorem: The mode is the most effective statistical measure for increasing sales -/
theorem mode_most_effective (store : ShoeStore) :
  mostEffectiveMeasure store = StatisticalMeasure.Mode :=
by sorry

end NUMINAMATH_CALUDE_mode_most_effective_l3166_316652


namespace NUMINAMATH_CALUDE_linear_system_sum_a_d_l3166_316627

theorem linear_system_sum_a_d :
  ∀ (a b c d e : ℝ),
    a + b = 14 →
    b + c = 9 →
    c + d = 3 →
    d + e = 6 →
    a - 2 * e = 1 →
    a + d = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_linear_system_sum_a_d_l3166_316627


namespace NUMINAMATH_CALUDE_problem_solution_l3166_316646

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (3 / x) * (2 / y) = 1 / 3) : x * y = 18 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3166_316646


namespace NUMINAMATH_CALUDE_perez_class_cans_collected_l3166_316624

/-- Calculates the total number of cans collected by a class during a food drive. -/
def totalCansCollected (totalStudents : ℕ) (halfStudentsCans : ℕ) (nonCollectingStudents : ℕ) (remainingStudentsCans : ℕ) : ℕ :=
  let halfStudents := totalStudents / 2
  let remainingStudents := totalStudents - halfStudents - nonCollectingStudents
  halfStudents * halfStudentsCans + remainingStudents * remainingStudentsCans

/-- Proves that Ms. Perez's class collected 232 cans in total. -/
theorem perez_class_cans_collected :
  totalCansCollected 30 12 2 4 = 232 := by
  sorry

#eval totalCansCollected 30 12 2 4

end NUMINAMATH_CALUDE_perez_class_cans_collected_l3166_316624


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3166_316685

-- Define the inequality
def inequality (x : ℝ) : Prop := -x^2 - 5*x + 6 ≥ 0

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | -6 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3166_316685


namespace NUMINAMATH_CALUDE_problem_statement_l3166_316687

theorem problem_statement (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) : 
  (ab + cd ≤ 1) ∧ (-2 ≤ a + Real.sqrt 3 * b) ∧ (a + Real.sqrt 3 * b ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3166_316687


namespace NUMINAMATH_CALUDE_tetrahedral_pyramid_marbles_hypertetrahedron_marbles_formula_l3166_316629

/-- The number of marbles in a d-dimensional hypertetrahedron with N layers -/
def hypertetrahedron_marbles (d : ℕ) (N : ℕ) : ℕ := Nat.choose (N + d - 1) d

/-- Theorem: The number of marbles in a tetrahedral pyramid with N layers is (N + 2) choose 3 -/
theorem tetrahedral_pyramid_marbles (N : ℕ) : 
  hypertetrahedron_marbles 3 N = Nat.choose (N + 2) 3 := by sorry

/-- Theorem: The number of marbles in a d-dimensional hypertetrahedron with N layers is (N + d - 1) choose d -/
theorem hypertetrahedron_marbles_formula (d : ℕ) (N : ℕ) : 
  hypertetrahedron_marbles d N = Nat.choose (N + d - 1) d := by sorry

end NUMINAMATH_CALUDE_tetrahedral_pyramid_marbles_hypertetrahedron_marbles_formula_l3166_316629


namespace NUMINAMATH_CALUDE_C₁_cartesian_polar_equiv_l3166_316619

/-- The curve C₁ in Cartesian coordinates -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 10*y + 16 = 0

/-- The curve C₁ in polar coordinates -/
def C₁_polar (ρ θ : ℝ) : Prop := ρ^2 - 8*ρ*Real.cos θ - 10*ρ*Real.sin θ + 16 = 0

/-- Theorem stating the equivalence of Cartesian and polar representations of C₁ -/
theorem C₁_cartesian_polar_equiv :
  ∀ (x y ρ θ : ℝ), 
    x = ρ * Real.cos θ → 
    y = ρ * Real.sin θ → 
    (C₁ x y ↔ C₁_polar ρ θ) :=
by
  sorry

end NUMINAMATH_CALUDE_C₁_cartesian_polar_equiv_l3166_316619


namespace NUMINAMATH_CALUDE_determinant_of_roots_l3166_316609

/-- Given a, b, c are roots of x^3 + px^2 + qx + r = 0, 
    the determinant of [[a, c, b], [c, b, a], [b, a, c]] is -c^3 + b^2c -/
theorem determinant_of_roots (p q r a b c : ℝ) : 
  a^3 + p*a^2 + q*a + r = 0 →
  b^3 + p*b^2 + q*b + r = 0 →
  c^3 + p*c^2 + q*c + r = 0 →
  Matrix.det !![a, c, b; c, b, a; b, a, c] = -c^3 + b^2*c := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_roots_l3166_316609


namespace NUMINAMATH_CALUDE_garden_shape_is_square_l3166_316682

theorem garden_shape_is_square (cabbages_this_year : ℕ) (cabbage_increase : ℕ) 
  (h1 : cabbages_this_year = 11236)
  (h2 : cabbage_increase = 211)
  (h3 : ∃ (n : ℕ), n ^ 2 = cabbages_this_year)
  (h4 : ∃ (m : ℕ), m ^ 2 = cabbages_this_year - cabbage_increase) :
  ∃ (side : ℕ), side ^ 2 = cabbages_this_year := by
  sorry

end NUMINAMATH_CALUDE_garden_shape_is_square_l3166_316682


namespace NUMINAMATH_CALUDE_quadratic_sum_of_constants_l3166_316667

-- Define the quadratic function
def f (x : ℝ) : ℝ := 4*x^2 - 16*x - 64

-- Define the completed square form
def g (x a b c : ℝ) : ℝ := a*(x+b)^2 + c

-- Theorem statement
theorem quadratic_sum_of_constants :
  ∃ (a b c : ℝ), (∀ x, f x = g x a b c) ∧ (a + b + c = -78) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_constants_l3166_316667


namespace NUMINAMATH_CALUDE_geometric_mean_a2_a8_l3166_316659

theorem geometric_mean_a2_a8 (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 3 →                     -- first term
  q = 2 →                       -- common ratio
  (a 2 * a 8).sqrt = 48 ∨ (a 2 * a 8).sqrt = -48 :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_a2_a8_l3166_316659


namespace NUMINAMATH_CALUDE_school_sections_l3166_316625

theorem school_sections (num_boys num_girls : ℕ) (h1 : num_boys = 408) (h2 : num_girls = 240) :
  let section_size := Nat.gcd num_boys num_girls
  let boys_sections := num_boys / section_size
  let girls_sections := num_girls / section_size
  boys_sections + girls_sections = 27 := by
sorry

end NUMINAMATH_CALUDE_school_sections_l3166_316625


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3166_316601

theorem necessary_not_sufficient_condition :
  (∀ a b c d : ℝ, a + b < c + d → (a < c ∨ b < d)) ∧
  (∃ a b c d : ℝ, (a < c ∨ b < d) ∧ ¬(a + b < c + d)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3166_316601


namespace NUMINAMATH_CALUDE_x_2023_minus_1_values_l3166_316615

theorem x_2023_minus_1_values (x : ℝ) : 
  (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 → 
  x^2023 - 1 = 0 ∨ x^2023 - 1 = -2 := by
sorry

end NUMINAMATH_CALUDE_x_2023_minus_1_values_l3166_316615


namespace NUMINAMATH_CALUDE_probability_theorem_l3166_316694

def total_candidates : ℕ := 9
def boys : ℕ := 5
def girls : ℕ := 4
def volunteers : ℕ := 4

def probability_1girl_3boys : ℚ := 20 / 63

def P (n : ℕ) : ℚ := 
  (Nat.choose boys n * Nat.choose girls (volunteers - n)) / Nat.choose total_candidates volunteers

theorem probability_theorem :
  (probability_1girl_3boys = P 3) ∧
  (∀ n : ℕ, P n ≥ 3/4 → n ≤ 2) ∧
  (P 2 ≥ 3/4) :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l3166_316694


namespace NUMINAMATH_CALUDE_li_fang_outfits_l3166_316641

/-- The number of unique outfit combinations given a set of shirts, skirts, and dresses -/
def outfit_combinations (num_shirts num_skirts num_dresses : ℕ) : ℕ :=
  num_shirts * num_skirts + num_dresses

/-- Theorem: Given 4 shirts, 3 skirts, and 2 dresses, the total number of unique outfit combinations is 14 -/
theorem li_fang_outfits : outfit_combinations 4 3 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_li_fang_outfits_l3166_316641


namespace NUMINAMATH_CALUDE_combine_like_terms_1_combine_like_terms_2_l3166_316684

-- Define variables
variable (x y : ℝ)

-- Theorem 1
theorem combine_like_terms_1 : 2*x - (x - y) + (x + y) = 2*x + 2*y := by
  sorry

-- Theorem 2
theorem combine_like_terms_2 : 3*x^2 - 9*x + 2 - x^2 + 4*x - 6 = 2*x^2 - 5*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_1_combine_like_terms_2_l3166_316684


namespace NUMINAMATH_CALUDE_both_reunions_count_l3166_316693

/-- The number of people attending both reunions -/
def both_reunions (total guests : ℕ) (oates hall : ℕ) : ℕ :=
  total - (oates + hall - total)

theorem both_reunions_count : both_reunions 150 70 52 = 28 := by
  sorry

end NUMINAMATH_CALUDE_both_reunions_count_l3166_316693


namespace NUMINAMATH_CALUDE_not_equal_vectors_not_both_zero_l3166_316654

theorem not_equal_vectors_not_both_zero {n : Type*} [NormedAddCommGroup n] 
  (a b : n) (h : a ≠ b) : ¬(a = 0 ∧ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_not_equal_vectors_not_both_zero_l3166_316654


namespace NUMINAMATH_CALUDE_mehki_age_l3166_316602

/-- Given the ages of Mehki, Jordyn, and Zrinka, prove Mehki's age -/
theorem mehki_age (zrinka jordyn mehki : ℕ) 
  (h1 : mehki = jordyn + 10)
  (h2 : jordyn = 2 * zrinka)
  (h3 : zrinka = 6) :
  mehki = 22 := by sorry

end NUMINAMATH_CALUDE_mehki_age_l3166_316602


namespace NUMINAMATH_CALUDE_problem_statement_l3166_316634

theorem problem_statement (a b c t : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : t ≥ 1)
  (h5 : a + b + c = 1/2)
  (h6 : Real.sqrt (a + 1/2 * (b - c)^2) + Real.sqrt b + Real.sqrt c = Real.sqrt (6*t) / 2) :
  a^(2*t) + b^(2*t) + c^(2*t) = 1/12 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3166_316634


namespace NUMINAMATH_CALUDE_average_of_ten_numbers_l3166_316691

theorem average_of_ten_numbers (first_six_avg : ℝ) (last_six_avg : ℝ) (seventh_num : ℝ) 
  (h1 : first_six_avg = 68)
  (h2 : last_six_avg = 75)
  (h3 : seventh_num = 258) :
  (6 * first_six_avg + 6 * last_six_avg - seventh_num) / 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_of_ten_numbers_l3166_316691


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l3166_316645

/-- Given a parallelogram with height 6 meters and area 72 square meters, its base is 12 meters. -/
theorem parallelogram_base_length (height : ℝ) (area : ℝ) (base : ℝ) : 
  height = 6 → area = 72 → area = base * height → base = 12 := by sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l3166_316645


namespace NUMINAMATH_CALUDE_triangle_angle_F_l3166_316644

theorem triangle_angle_F (D E : Real) (h1 : 2 * Real.sin D + 5 * Real.cos E = 7)
                         (h2 : 5 * Real.sin E + 2 * Real.cos D = 4) :
  Real.sin (π - D - E) = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_F_l3166_316644


namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_achievable_l3166_316674

theorem max_value_inequality (x y : ℝ) :
  (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 35 :=
by sorry

theorem max_value_achievable :
  ∃ (x y : ℝ), (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4) = Real.sqrt 35 :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_achievable_l3166_316674


namespace NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l3166_316675

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l3166_316675


namespace NUMINAMATH_CALUDE_solve_and_prove_l3166_316622

-- Given that |x+a| ≤ b has the solution set [-6, 2]
def has_solution_set (a b : ℝ) : Prop :=
  ∀ x, |x + a| ≤ b ↔ -6 ≤ x ∧ x ≤ 2

-- Define the conditions |am+n| < 1/3 and |m-bn| < 1/6
def conditions (a b m n : ℝ) : Prop :=
  |a * m + n| < 1/3 ∧ |m - b * n| < 1/6

theorem solve_and_prove (a b m n : ℝ) 
  (h1 : has_solution_set a b) 
  (h2 : conditions a b m n) : 
  (a = 2 ∧ b = 4) ∧ |n| < 2/27 :=
sorry

end NUMINAMATH_CALUDE_solve_and_prove_l3166_316622


namespace NUMINAMATH_CALUDE_sum_first_seven_odd_numbers_l3166_316656

def sum_odd_numbers (n : ℕ) : ℕ := (2 * n - 1) * n

theorem sum_first_seven_odd_numbers :
  (sum_odd_numbers 2 = 2^2) →
  (sum_odd_numbers 5 = 5^2) →
  (sum_odd_numbers 7 = 7^2) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_first_seven_odd_numbers_l3166_316656


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l3166_316663

theorem min_value_and_inequality (x y z : ℝ) (h : x + y + z = 1) :
  ((x - 1)^2 + (y + 1)^2 + (z + 1)^2 ≥ 4/3) ∧
  (∀ a : ℝ, (x - 2)^2 + (y - 1)^2 + (z - a)^2 ≥ 1/3 → a ≤ -3 ∨ a ≥ -1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l3166_316663


namespace NUMINAMATH_CALUDE_shopkeeper_percentage_gain_l3166_316650

/-- The percentage gain of a shopkeeper using a false weight --/
theorem shopkeeper_percentage_gain :
  let actual_weight : ℝ := 970
  let claimed_weight : ℝ := 1000
  let gain : ℝ := claimed_weight - actual_weight
  let percentage_gain : ℝ := (gain / actual_weight) * 100
  ∃ ε > 0, abs (percentage_gain - 3.09) < ε :=
by sorry

end NUMINAMATH_CALUDE_shopkeeper_percentage_gain_l3166_316650


namespace NUMINAMATH_CALUDE_michael_pizza_portion_l3166_316638

theorem michael_pizza_portion
  (total_pizza : ℚ)
  (treshawn_portion : ℚ)
  (lamar_portion : ℚ)
  (h1 : total_pizza = 1)
  (h2 : treshawn_portion = 1 / 2)
  (h3 : lamar_portion = 1 / 6)
  : total_pizza - (treshawn_portion + lamar_portion) = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_michael_pizza_portion_l3166_316638


namespace NUMINAMATH_CALUDE_charlie_widget_production_l3166_316668

/-- Charlie's widget production problem -/
theorem charlie_widget_production 
  (w t : ℕ) -- w: widgets per hour, t: hours worked on Thursday
  (h1 : w = 3 * t) -- Condition: w = 3t
  : w * t - (w + 6) * (t - 3) = 3 * t + 18 := by
  sorry


end NUMINAMATH_CALUDE_charlie_widget_production_l3166_316668


namespace NUMINAMATH_CALUDE_a_greater_than_b_squared_l3166_316613

theorem a_greater_than_b_squared {a b : ℝ} (h1 : a > 1) (h2 : 1 > b) (h3 : b > -1) : a > b^2 := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_squared_l3166_316613


namespace NUMINAMATH_CALUDE_difference_30th_28th_triangular_l3166_316672

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_30th_28th_triangular : 
  triangular_number 30 - triangular_number 28 = 59 := by
  sorry

end NUMINAMATH_CALUDE_difference_30th_28th_triangular_l3166_316672


namespace NUMINAMATH_CALUDE_paths_7x8_grid_l3166_316662

/-- The number of distinct paths on a rectangular grid -/
def gridPaths (width height : ℕ) : ℕ :=
  Nat.choose (width + height) height

/-- Theorem: The number of distinct paths on a 7x8 grid is 6435 -/
theorem paths_7x8_grid :
  gridPaths 7 8 = 6435 := by
  sorry

end NUMINAMATH_CALUDE_paths_7x8_grid_l3166_316662


namespace NUMINAMATH_CALUDE_notebook_cost_l3166_316657

def total_spent : ℕ := 74
def ruler_cost : ℕ := 18
def pencil_cost : ℕ := 7
def num_pencils : ℕ := 3

theorem notebook_cost :
  total_spent - (ruler_cost + num_pencils * pencil_cost) = 35 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l3166_316657


namespace NUMINAMATH_CALUDE_e_4i_in_third_quadrant_l3166_316651

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Real.cos z.im + Complex.I * Real.sin z.im)

-- Define Euler's formula
axiom eulers_formula (x : ℝ) : cexp (Complex.I * x) = Complex.cos x + Complex.I * Complex.sin x

-- Define the third quadrant
def third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- Theorem statement
theorem e_4i_in_third_quadrant :
  third_quadrant (cexp (4 * Complex.I)) :=
sorry

end NUMINAMATH_CALUDE_e_4i_in_third_quadrant_l3166_316651


namespace NUMINAMATH_CALUDE_loan_division_l3166_316626

theorem loan_division (total : ℝ) (rate1 rate2 years1 years2 : ℝ) : 
  total = 2665 ∧ rate1 = 3/100 ∧ rate2 = 5/100 ∧ years1 = 5 ∧ years2 = 3 →
  ∃ (part1 part2 : ℝ), 
    part1 + part2 = total ∧
    part1 * rate1 * years1 = part2 * rate2 * years2 ∧
    part2 = 1332.5 := by
  sorry

end NUMINAMATH_CALUDE_loan_division_l3166_316626


namespace NUMINAMATH_CALUDE_grade_assignments_count_l3166_316647

/-- The number of possible grades a professor can assign to each student. -/
def num_grades : ℕ := 4

/-- The number of students in the class. -/
def num_students : ℕ := 15

/-- The number of ways to assign grades to all students in the class. -/
def num_grade_assignments : ℕ := num_grades ^ num_students

/-- Theorem stating that the number of ways to assign grades is 4^15. -/
theorem grade_assignments_count :
  num_grade_assignments = 1073741824 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignments_count_l3166_316647


namespace NUMINAMATH_CALUDE_domino_tiling_theorem_l3166_316673

/-- Represents a rectangle tiled with dominoes -/
structure DominoRectangle where
  width : ℕ
  height : ℕ
  dominoes : ℕ

/-- Condition that any grid line intersects a multiple of four dominoes -/
def grid_line_condition (r : DominoRectangle) : Prop :=
  ∀ (line : ℕ), line ≤ r.width ∨ line ≤ r.height → 
    (if line ≤ r.width then r.height else r.width) % 4 = 0

/-- Main theorem: If the grid line condition holds, then one side is divisible by 4 -/
theorem domino_tiling_theorem (r : DominoRectangle) 
  (h : grid_line_condition r) : 
  r.width % 4 = 0 ∨ r.height % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_domino_tiling_theorem_l3166_316673


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l3166_316660

theorem largest_n_for_factorization : ∃ (n : ℤ),
  (∀ m : ℤ, (∃ (a b c d : ℤ), 7 * X^2 + m * X + 56 = (a * X + b) * (c * X + d)) → m ≤ n) ∧
  (∃ (a b c d : ℤ), 7 * X^2 + n * X + 56 = (a * X + b) * (c * X + d)) ∧
  n = 393 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l3166_316660


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_squared_l3166_316699

theorem cube_sum_reciprocal_squared (x : ℝ) (h : 53 = x^6 + 1/x^6) : (x^3 + 1/x^3)^2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_squared_l3166_316699


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3166_316620

theorem sqrt_equation_solution :
  ∃! (x : ℝ), Real.sqrt x + Real.sqrt (x + 8) = 8 ∧ x = 49/4 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3166_316620


namespace NUMINAMATH_CALUDE_num_triangles_on_square_l3166_316696

/-- The number of points on each side of the square (excluding corners) -/
def points_per_side : ℕ := 7

/-- The total number of points on all sides of the square -/
def total_points : ℕ := 4 * points_per_side

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of different triangles formed by selecting three distinct points
    from a set of points on the sides of a square (excluding corners) -/
theorem num_triangles_on_square : 
  choose total_points 3 - 4 * (choose points_per_side 3) = 3136 := by
  sorry

end NUMINAMATH_CALUDE_num_triangles_on_square_l3166_316696


namespace NUMINAMATH_CALUDE_sara_height_l3166_316689

/-- Proves that Sara's height is 45 inches given the relative heights of Sara, Joe, Roy, Mark, and Julie. -/
theorem sara_height (
  julie_height : ℕ)
  (mark_taller_than_julie : ℕ)
  (roy_taller_than_mark : ℕ)
  (joe_taller_than_roy : ℕ)
  (sara_taller_than_joe : ℕ)
  (h_julie : julie_height = 33)
  (h_mark : mark_taller_than_julie = 1)
  (h_roy : roy_taller_than_mark = 2)
  (h_joe : joe_taller_than_roy = 3)
  (h_sara : sara_taller_than_joe = 6) :
  julie_height + mark_taller_than_julie + roy_taller_than_mark + joe_taller_than_roy + sara_taller_than_joe = 45 := by
  sorry

end NUMINAMATH_CALUDE_sara_height_l3166_316689


namespace NUMINAMATH_CALUDE_incorrect_statement_about_converses_l3166_316642

/-- A proposition in mathematics -/
structure Proposition where
  statement : Prop

/-- A theorem in mathematics -/
structure Theorem where
  statement : Prop
  proof : statement

/-- The converse of a proposition -/
def converse (p : Proposition) : Proposition :=
  ⟨¬p.statement⟩

theorem incorrect_statement_about_converses :
  ¬(∀ (p : Proposition), ∃ (c : Proposition), c = converse p ∧
     ∃ (t : Theorem), ¬∃ (c : Proposition), c = converse ⟨t.statement⟩) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_about_converses_l3166_316642


namespace NUMINAMATH_CALUDE_chair_price_proof_l3166_316670

/-- The normal price of a chair -/
def normal_price : ℝ := 20

/-- The discounted price for the first 5 chairs -/
def discounted_price_first_5 : ℝ := 0.75 * normal_price

/-- The discounted price for chairs after the first 5 -/
def discounted_price_after_5 : ℝ := 0.5 * normal_price

/-- The number of chairs bought -/
def chairs_bought : ℕ := 8

/-- The total cost of all chairs bought -/
def total_cost : ℝ := 105

theorem chair_price_proof :
  5 * discounted_price_first_5 + (chairs_bought - 5) * discounted_price_after_5 = total_cost :=
sorry

end NUMINAMATH_CALUDE_chair_price_proof_l3166_316670


namespace NUMINAMATH_CALUDE_tyson_basketball_score_l3166_316683

theorem tyson_basketball_score (three_point_shots two_point_shots one_point_shots : ℕ) 
  (h1 : three_point_shots = 15)
  (h2 : two_point_shots = 12)
  (h3 : 3 * three_point_shots + 2 * two_point_shots + one_point_shots = 75) :
  one_point_shots = 6 := by
  sorry

end NUMINAMATH_CALUDE_tyson_basketball_score_l3166_316683


namespace NUMINAMATH_CALUDE_same_solution_implies_c_value_l3166_316608

theorem same_solution_implies_c_value (x : ℝ) (c : ℝ) :
  (3 * x + 9 = 6) ∧ (c * x - 15 = -5) → c = -10 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_value_l3166_316608


namespace NUMINAMATH_CALUDE_cubic_polynomial_problem_l3166_316635

-- Define the cubic polynomial whose roots are a, b, c
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 2*x + 3

-- Define the properties of P
def is_valid_P (P : ℝ → ℝ) (a b c : ℝ) : Prop :=
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
  P a = b + c ∧ P b = a + c ∧ P c = a + b ∧
  P (a + b + c) = -20

-- The theorem to prove
theorem cubic_polynomial_problem :
  ∃ (P : ℝ → ℝ) (a b c : ℝ),
    is_valid_P P a b c ∧
    (∀ x, P x = -17/3 * x^3 + 68/3 * x^2 - 31/3 * x - 18) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_problem_l3166_316635


namespace NUMINAMATH_CALUDE_boys_camp_total_l3166_316614

theorem boys_camp_total (total : ℝ) 
  (h1 : 0.2 * total = total_school_A)
  (h2 : 0.3 * total_school_A = science_school_A)
  (h3 : total_school_A - science_school_A = 77) : 
  total = 550 := by
sorry

end NUMINAMATH_CALUDE_boys_camp_total_l3166_316614


namespace NUMINAMATH_CALUDE_mixture_quantity_is_three_l3166_316678

/-- Represents the cost and quantity of a tea and coffee mixture --/
structure TeaCoffeeMixture where
  june_cost : ℝ  -- Cost per pound of both tea and coffee in June
  july_tea_cost : ℝ  -- Cost per pound of tea in July
  july_coffee_cost : ℝ  -- Cost per pound of coffee in July
  mixture_cost : ℝ  -- Total cost of the mixture in July
  mixture_quantity : ℝ  -- Quantity of the mixture in pounds

/-- Theorem stating the quantity of mixture bought given the conditions --/
theorem mixture_quantity_is_three (m : TeaCoffeeMixture) : 
  m.june_cost > 0 ∧ 
  m.july_coffee_cost = 2 * m.june_cost ∧ 
  m.july_tea_cost = 0.3 * m.june_cost ∧ 
  m.july_tea_cost = 0.3 ∧ 
  m.mixture_cost = 3.45 ∧ 
  m.mixture_quantity = m.mixture_cost / ((m.july_tea_cost + m.july_coffee_cost) / 2) →
  m.mixture_quantity = 3 := by
  sorry

#check mixture_quantity_is_three

end NUMINAMATH_CALUDE_mixture_quantity_is_three_l3166_316678


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l3166_316604

theorem r_value_when_n_is_3 (n : ℕ) (s : ℕ) (r : ℕ) 
  (h1 : s = 2^n - 1) 
  (h2 : r = 3^s - s) 
  (h3 : n = 3) : 
  r = 2180 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l3166_316604


namespace NUMINAMATH_CALUDE_number_of_chip_bags_l3166_316636

/-- Given the weight of chips and juice, prove the number of bags of chips -/
theorem number_of_chip_bags (C J n : ℕ) : 
  2 * C = 800 →
  C = J + 350 →
  n * C + 4 * J = 2200 →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_number_of_chip_bags_l3166_316636


namespace NUMINAMATH_CALUDE_algebraic_identity_l3166_316664

theorem algebraic_identity (a b : ℝ) : 
  a = (Real.sqrt 5 + Real.sqrt 3) / (Real.sqrt 5 - Real.sqrt 3) →
  b = (Real.sqrt 5 - Real.sqrt 3) / (Real.sqrt 5 + Real.sqrt 3) →
  a^4 + b^4 + (a + b)^4 = 7938 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identity_l3166_316664


namespace NUMINAMATH_CALUDE_wendy_sales_l3166_316648

/-- Represents the sales data for a fruit vendor --/
structure FruitSales where
  apple_price : ℝ
  orange_price : ℝ
  morning_apples : ℕ
  morning_oranges : ℕ
  afternoon_apples : ℕ
  afternoon_oranges : ℕ

/-- Calculates the total sales for a given FruitSales instance --/
def total_sales (sales : FruitSales) : ℝ :=
  let total_apples := sales.morning_apples + sales.afternoon_apples
  let total_oranges := sales.morning_oranges + sales.afternoon_oranges
  (total_apples : ℝ) * sales.apple_price + (total_oranges : ℝ) * sales.orange_price

/-- Theorem stating that the total sales for the given conditions equal $205 --/
theorem wendy_sales : 
  let sales := FruitSales.mk 1.5 1 40 30 50 40
  total_sales sales = 205 := by
  sorry


end NUMINAMATH_CALUDE_wendy_sales_l3166_316648


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l3166_316676

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.20 * last_year_earnings
  let this_year_earnings := 1.20 * last_year_earnings
  let this_year_rent := 0.30 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 180 := by
sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l3166_316676


namespace NUMINAMATH_CALUDE_smallest_positive_integer_3003m_55555n_specific_solution_3003m_55555n_l3166_316655

theorem smallest_positive_integer_3003m_55555n :
  ∃ (m n : ℤ), 3003 * m + 55555 * n = 1 ∧
  ∀ (k l : ℤ), 3003 * k + 55555 * l > 0 → 3003 * k + 55555 * l ≥ 1 :=
by sorry

theorem specific_solution_3003m_55555n :
  3003 * 37 + 55555 * (-2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_3003m_55555n_specific_solution_3003m_55555n_l3166_316655


namespace NUMINAMATH_CALUDE_at_least_two_thirds_covered_l3166_316605

/-- Represents a chessboard with dominoes -/
structure ChessboardWithDominoes where
  m : Nat
  n : Nat
  dominoes : Finset (Nat × Nat)
  m_ge_two : m ≥ 2
  n_ge_two : n ≥ 2
  valid_placement : ∀ (i j : Nat), (i, j) ∈ dominoes → 
    (i < m ∧ j < n) ∧ (
      ((i + 1, j) ∈ dominoes ∧ (i + 1) < m) ∨
      ((i, j + 1) ∈ dominoes ∧ (j + 1) < n)
    )
  no_overlap : ∀ (i j k l : Nat), (i, j) ∈ dominoes → (k, l) ∈ dominoes → 
    (i = k ∧ j = l) ∨ (i + 1 = k ∧ j = l) ∨ (i = k ∧ j + 1 = l) ∨
    (k + 1 = i ∧ j = l) ∨ (k = i ∧ l + 1 = j)
  no_more_addable : ∀ (i j : Nat), i < m → j < n → 
    (i, j) ∉ dominoes → (i + 1 < m → (i + 1, j) ∈ dominoes) ∧
    (j + 1 < n → (i, j + 1) ∈ dominoes)

/-- The main theorem stating that at least 2/3 of the chessboard is covered by dominoes -/
theorem at_least_two_thirds_covered (board : ChessboardWithDominoes) : 
  (2 : ℚ) / 3 * (board.m * board.n : ℚ) ≤ (board.dominoes.card * 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_thirds_covered_l3166_316605


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3166_316669

theorem cubic_equation_roots : ∃ (x₁ x₂ x₃ : ℝ),
  (x₁ = 3 ∧ x₂ = -3 ∧ x₃ = 5) ∧
  (∀ x : ℝ, x^3 - 5*x^2 - 9*x + 45 = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
  (x₁ = -x₂) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3166_316669


namespace NUMINAMATH_CALUDE_min_value_expression_l3166_316616

theorem min_value_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + c = 2 * b) (h4 : a ≠ 0) :
  ((a + b)^2 + (b - c)^2 + (c - a)^2) / a^2 ≥ 7/2 ∧
  ∃ a b c : ℝ, a > b ∧ b > c ∧ a + c = 2 * b ∧ a ≠ 0 ∧
    ((a + b)^2 + (b - c)^2 + (c - a)^2) / a^2 = 7/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3166_316616


namespace NUMINAMATH_CALUDE_hannah_games_played_l3166_316653

/-- Given Hannah's total points and average points per game, calculate the number of games played. -/
theorem hannah_games_played (total_points : ℕ) (average_points : ℕ) (h1 : total_points = 312) (h2 : average_points = 13) :
  total_points / average_points = 24 := by
  sorry

#check hannah_games_played

end NUMINAMATH_CALUDE_hannah_games_played_l3166_316653


namespace NUMINAMATH_CALUDE_stability_comparison_l3166_316681

/-- Represents a set of data with its variance -/
structure DataSet where
  variance : ℝ

/-- Stability comparison between two data sets -/
def more_stable (a b : DataSet) : Prop := a.variance < b.variance

/-- Theorem: If two data sets have the same average and set A has lower variance,
    then set A is more stable than set B -/
theorem stability_comparison (A B : DataSet) 
  (h1 : A.variance = 2)
  (h2 : B.variance = 2.5)
  : more_stable A B := by
  sorry

end NUMINAMATH_CALUDE_stability_comparison_l3166_316681


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3166_316640

theorem quadratic_roots_sum (a b : ℝ) : 
  (a^2 + a - 2024 = 0) → 
  (b^2 + b - 2024 = 0) → 
  (a^2 + 2*a + b = 2023) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3166_316640


namespace NUMINAMATH_CALUDE_tank_capacity_l3166_316697

/-- Proves that a tank with given leak and inlet rates has a capacity of 1728 litres -/
theorem tank_capacity (leak_empty_time : ℝ) (inlet_rate : ℝ) (combined_empty_time : ℝ) 
  (h1 : leak_empty_time = 8) 
  (h2 : inlet_rate = 6) 
  (h3 : combined_empty_time = 12) : ℝ :=
by
  -- Define the capacity of the tank
  let capacity : ℝ := 1728

  -- State that the capacity is equal to 1728 litres
  have capacity_eq : capacity = 1728 := by rfl

  -- The proof would go here
  sorry


end NUMINAMATH_CALUDE_tank_capacity_l3166_316697
