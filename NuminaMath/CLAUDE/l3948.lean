import Mathlib

namespace S_31_primes_less_than_20000_l3948_394809

/-- Sum of digits in base k -/
def S (k : ℕ) (n : ℕ) : ℕ := sorry

/-- The theorem to be proved -/
theorem S_31_primes_less_than_20000 (p : ℕ) (h_prime : Nat.Prime p) (h_bound : p < 20000) :
  S 31 p = 49 ∨ S 31 p = 77 := by sorry

end S_31_primes_less_than_20000_l3948_394809


namespace infinite_solutions_l3948_394834

/-- The equation that x, y, and z must satisfy -/
def satisfies_equation (x y z : ℕ+) : Prop :=
  (x + y + z)^2 + 2*(x + y + z) = 5*(x*y + y*z + z*x)

/-- The set of all positive integer solutions to the equation -/
def solution_set : Set (ℕ+ × ℕ+ × ℕ+) :=
  {xyz | satisfies_equation xyz.1 xyz.2.1 xyz.2.2}

/-- The main theorem stating that the solution set is infinite -/
theorem infinite_solutions : Set.Infinite solution_set := by
  sorry

end infinite_solutions_l3948_394834


namespace hardcover_books_purchased_l3948_394885

/-- The number of hardcover books purchased -/
def num_hardcover : ℕ := 8

/-- The number of paperback books purchased -/
def num_paperback : ℕ := 12 - num_hardcover

/-- The price of a paperback book -/
def price_paperback : ℕ := 18

/-- The price of a hardcover book -/
def price_hardcover : ℕ := 30

/-- The total amount spent -/
def total_spent : ℕ := 312

/-- Theorem stating that the number of hardcover books purchased is 8 -/
theorem hardcover_books_purchased :
  num_hardcover = 8 ∧
  num_hardcover + num_paperback = 12 ∧
  price_hardcover * num_hardcover + price_paperback * num_paperback = total_spent :=
by sorry

end hardcover_books_purchased_l3948_394885


namespace prime_square_plus_two_prime_l3948_394870

theorem prime_square_plus_two_prime (P : ℕ) (h1 : Nat.Prime P) (h2 : Nat.Prime (P^2 + 2)) :
  P^4 + 1921 = 2002 := by
sorry

end prime_square_plus_two_prime_l3948_394870


namespace james_total_vegetables_l3948_394877

/-- The total number of vegetables James ate -/
def total_vegetables (before_carrot before_cucumber after_carrot after_cucumber after_celery : ℕ) : ℕ :=
  before_carrot + before_cucumber + after_carrot + after_cucumber + after_celery

/-- Theorem stating that James ate 77 vegetables in total -/
theorem james_total_vegetables :
  total_vegetables 22 18 15 10 12 = 77 := by
  sorry

end james_total_vegetables_l3948_394877


namespace calculate_salary_e_l3948_394849

/-- Calculates the salary of person E given the salaries of A, B, C, D, and the average salary of all five people. -/
theorem calculate_salary_e (salary_a salary_b salary_c salary_d avg_salary : ℕ) :
  salary_a = 8000 →
  salary_b = 5000 →
  salary_c = 11000 →
  salary_d = 7000 →
  avg_salary = 8000 →
  (salary_a + salary_b + salary_c + salary_d + (avg_salary * 5 - (salary_a + salary_b + salary_c + salary_d))) / 5 = avg_salary →
  avg_salary * 5 - (salary_a + salary_b + salary_c + salary_d) = 9000 := by
sorry

end calculate_salary_e_l3948_394849


namespace sum_base3_equals_100212_l3948_394863

/-- Converts a base-3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 3 * acc + d) 0

/-- Converts a decimal number to its base-3 representation as a list of digits -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
    aux n []

theorem sum_base3_equals_100212 :
  let a := base3ToDecimal [1]
  let b := base3ToDecimal [1, 0, 2]
  let c := base3ToDecimal [2, 0, 2, 1]
  let d := base3ToDecimal [1, 1, 0, 1, 2]
  let e := base3ToDecimal [2, 2, 1, 1, 1]
  decimalToBase3 (a + b + c + d + e) = [1, 0, 0, 2, 1, 2] := by
  sorry

end sum_base3_equals_100212_l3948_394863


namespace sin_continuous_l3948_394880

theorem sin_continuous : ContinuousOn Real.sin Set.univ := by
  sorry

end sin_continuous_l3948_394880


namespace hammond_discarded_marble_l3948_394810

/-- The weight of discarded marble after carving statues -/
def discarded_marble (initial_block : ℕ) (statue1 statue2 statue3 statue4 : ℕ) : ℕ :=
  initial_block - (statue1 + statue2 + statue3 + statue4)

/-- Theorem stating the amount of discarded marble for Hammond's statues -/
theorem hammond_discarded_marble :
  discarded_marble 80 10 18 15 15 = 22 := by
  sorry

end hammond_discarded_marble_l3948_394810


namespace solution_correctness_l3948_394868

/-- The set of solutions for x^2 - y^2 = 105 where x and y are natural numbers -/
def SolutionsA : Set (ℕ × ℕ) :=
  {(53, 52), (19, 16), (13, 8), (11, 4)}

/-- The set of solutions for 2x^2 + 5xy - 12y^2 = 28 where x and y are natural numbers -/
def SolutionsB : Set (ℕ × ℕ) :=
  {(8, 5)}

theorem solution_correctness :
  (∀ (x y : ℕ), x^2 - y^2 = 105 ↔ (x, y) ∈ SolutionsA) ∧
  (∀ (x y : ℕ), 2*x^2 + 5*x*y - 12*y^2 = 28 ↔ (x, y) ∈ SolutionsB) := by
  sorry

end solution_correctness_l3948_394868


namespace find_a_l3948_394806

theorem find_a : ∃ a : ℚ, 3 * a - 2 = 2 / 2 + 3 → a = 2 := by
  sorry

end find_a_l3948_394806


namespace slope_angle_of_negative_sqrt3_over_3_l3948_394881

/-- The slope angle of a line with slope -√3/3 is 5π/6 -/
theorem slope_angle_of_negative_sqrt3_over_3 :
  let slope : ℝ := -Real.sqrt 3 / 3
  let slope_angle : ℝ := Real.arctan slope
  slope_angle = 5 * Real.pi / 6 := by sorry

end slope_angle_of_negative_sqrt3_over_3_l3948_394881


namespace kite_to_square_area_ratio_l3948_394840

/-- The ratio of the area of a kite formed by the diagonals of four central
    smaller squares to the area of a large square --/
theorem kite_to_square_area_ratio :
  let large_side : ℝ := 60
  let small_side : ℝ := 10
  let large_area : ℝ := large_side ^ 2
  let kite_diagonal1 : ℝ := 2 * small_side
  let kite_diagonal2 : ℝ := 2 * small_side * Real.sqrt 2
  let kite_area : ℝ := (1 / 2) * kite_diagonal1 * kite_diagonal2
  kite_area / large_area = 100 * Real.sqrt 2 / 3600 := by
sorry

end kite_to_square_area_ratio_l3948_394840


namespace smallest_X_l3948_394830

/-- A function that checks if a natural number consists only of 0s and 1s --/
def onlyZerosAndOnes (n : ℕ) : Prop := sorry

/-- The smallest positive integer T consisting of only 0s and 1s that is divisible by 18 --/
def T : ℕ := 111111111000

/-- X is defined as T divided by 18 --/
def X : ℕ := T / 18

/-- Main theorem: X is the smallest positive integer satisfying the given conditions --/
theorem smallest_X : 
  (onlyZerosAndOnes T) ∧ 
  (X * 18 = T) ∧ 
  (∀ Y : ℕ, (∃ S : ℕ, onlyZerosAndOnes S ∧ Y * 18 = S) → X ≤ Y) ∧
  X = 6172839500 := by sorry

end smallest_X_l3948_394830


namespace number_relation_l3948_394847

theorem number_relation (A B : ℝ) (h : A = B - (4/5) * B) : B = (A + B) / (4/5) := by
  sorry

end number_relation_l3948_394847


namespace track_length_is_360_l3948_394817

/-- Represents a circular running track with two runners -/
structure RunningTrack where
  length : ℝ
  sally_first_meeting : ℝ
  john_second_meeting : ℝ

/-- Theorem stating that given the conditions, the track length is 360 meters -/
theorem track_length_is_360 (track : RunningTrack) 
  (h1 : track.sally_first_meeting = 90)
  (h2 : track.john_second_meeting = 200)
  (h3 : track.sally_first_meeting > 0)
  (h4 : track.john_second_meeting > 0)
  (h5 : track.length > 0) :
  track.length = 360 := by
  sorry

#check track_length_is_360

end track_length_is_360_l3948_394817


namespace intersection_line_canonical_form_l3948_394828

/-- Given two planes in 3D space, prove that their intersection forms a line with specific canonical equations. -/
theorem intersection_line_canonical_form (x y z : ℝ) :
  (2 * x - 3 * y - 2 * z + 6 = 0) →
  (x - 3 * y + z + 3 = 0) →
  ∃ (t : ℝ), x = 9 * t - 3 ∧ y = 4 * t ∧ z = 3 * t :=
by sorry

end intersection_line_canonical_form_l3948_394828


namespace fixed_point_sum_l3948_394838

/-- The function f(x) = a^(x-2) + 2 with a > 0 and a ≠ 1 has a fixed point (m, n) such that m + n = 5 -/
theorem fixed_point_sum (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  ∃ (m n : ℝ), (∀ x : ℝ, a^(x - 2) + 2 = a^(m - 2) + 2) ∧ m + n = 5 := by
  sorry

end fixed_point_sum_l3948_394838


namespace replaced_man_age_l3948_394841

theorem replaced_man_age (A B C D : ℝ) (new_avg : ℝ) :
  A = 23 →
  (A + B + C + D) / 4 < (52 + C + D) / 4 →
  B < 29 := by
sorry

end replaced_man_age_l3948_394841


namespace colored_regions_bound_l3948_394851

/-- A structure representing a plane with n lines and colored regions -/
structure ColoredPlane where
  n : ℕ
  n_ge_2 : n ≥ 2

/-- The number of colored regions in a ColoredPlane -/
def num_colored_regions (p : ColoredPlane) : ℕ := sorry

/-- Theorem stating that the number of colored regions is bounded -/
theorem colored_regions_bound (p : ColoredPlane) :
  num_colored_regions p ≤ (p.n^2 + p.n) / 3 := by sorry

end colored_regions_bound_l3948_394851


namespace yasmin_children_count_l3948_394836

def john_children (yasmin_children : ℕ) : ℕ := 2 * yasmin_children

theorem yasmin_children_count :
  ∃ (yasmin_children : ℕ),
    yasmin_children = 2 ∧
    john_children yasmin_children + yasmin_children = 6 :=
by
  sorry

end yasmin_children_count_l3948_394836


namespace percentage_of_eight_l3948_394802

theorem percentage_of_eight : ∃ p : ℝ, (p / 100) * 8 = 0.06 ∧ p = 0.75 := by
  sorry

end percentage_of_eight_l3948_394802


namespace gcd_of_three_numbers_l3948_394823

theorem gcd_of_three_numbers : Nat.gcd 12903 (Nat.gcd 18239 37422) = 1 := by
  sorry

end gcd_of_three_numbers_l3948_394823


namespace x_plus_one_equals_four_l3948_394867

theorem x_plus_one_equals_four (x : ℝ) (h : x = 3) : x + 1 = 4 := by
  sorry

end x_plus_one_equals_four_l3948_394867


namespace stream_speed_l3948_394872

/-- The speed of a stream given downstream and upstream speeds -/
theorem stream_speed (downstream upstream : ℝ) (h1 : downstream = 13) (h2 : upstream = 8) :
  (downstream - upstream) / 2 = 2.5 := by
  sorry

end stream_speed_l3948_394872


namespace mississippi_arrangements_l3948_394852

def word : String := "MISSISSIPPI"

def letter_count (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

theorem mississippi_arrangements :
  (Nat.factorial 11) / 
  (Nat.factorial (letter_count word 'S') * 
   Nat.factorial (letter_count word 'I') * 
   Nat.factorial (letter_count word 'P') * 
   Nat.factorial (letter_count word 'M')) = 34650 := by
  sorry

end mississippi_arrangements_l3948_394852


namespace team_probability_l3948_394842

/-- Given 27 players randomly split into 3 teams of 9 each, with Zack, Mihir, and Andrew on different teams,
    the probability that Zack and Andrew are on the same team is 8/17. -/
theorem team_probability (total_players : Nat) (teams : Nat) (players_per_team : Nat)
  (h1 : total_players = 27)
  (h2 : teams = 3)
  (h3 : players_per_team = 9)
  (h4 : total_players = teams * players_per_team)
  (zack mihir andrew : Nat)
  (h5 : zack ≠ mihir)
  (h6 : mihir ≠ andrew)
  (h7 : zack ≠ andrew) :
  (8 : ℚ) / 17 = (players_per_team - 1 : ℚ) / (total_players - 2 * players_per_team) :=
sorry

end team_probability_l3948_394842


namespace melissa_driving_hours_l3948_394891

/-- Calculates the total driving hours in a year given the number of trips per month,
    hours per trip, and months in a year. -/
def annual_driving_hours (trips_per_month : ℕ) (hours_per_trip : ℕ) (months_in_year : ℕ) : ℕ :=
  trips_per_month * hours_per_trip * months_in_year

/-- Proves that Melissa spends 72 hours driving in a year given the specified conditions. -/
theorem melissa_driving_hours :
  let trips_per_month : ℕ := 2
  let hours_per_trip : ℕ := 3
  let months_in_year : ℕ := 12
  annual_driving_hours trips_per_month hours_per_trip months_in_year = 72 :=
by
  sorry


end melissa_driving_hours_l3948_394891


namespace marta_number_proof_l3948_394800

/-- A function that checks if a number has three different non-zero digits -/
def has_three_different_nonzero_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100) ≠ ((n / 10) % 10) ∧
  (n / 100) ≠ (n % 10) ∧
  ((n / 10) % 10) ≠ (n % 10) ∧
  (n / 100) ≠ 0 ∧ ((n / 10) % 10) ≠ 0 ∧ (n % 10) ≠ 0

/-- A function that checks if a number has three identical digits -/
def has_three_identical_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100) = ((n / 10) % 10) ∧
  (n / 100) = (n % 10)

theorem marta_number_proof :
  ∀ n : ℕ,
  has_three_different_nonzero_digits n →
  has_three_identical_digits (3 * n) →
  ((n / 10) % 10) = (3 * n / 100) →
  n = 148 :=
by
  sorry

#check marta_number_proof

end marta_number_proof_l3948_394800


namespace regression_correlation_zero_l3948_394850

/-- Regression coefficient -/
def regression_coefficient (X Y : List ℝ) : ℝ := sorry

/-- Correlation coefficient -/
def correlation_coefficient (X Y : List ℝ) : ℝ := sorry

theorem regression_correlation_zero (X Y : List ℝ) :
  regression_coefficient X Y = 0 → correlation_coefficient X Y = 0 := by
  sorry

end regression_correlation_zero_l3948_394850


namespace smallest_positive_integer_with_remainders_l3948_394897

theorem smallest_positive_integer_with_remainders : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n % 4 = 1) ∧ 
  (n % 5 = 2) ∧ 
  (n % 6 = 3) ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 2 ∧ m % 6 = 3 → m ≥ n) ∧
  n = 21 := by
sorry

end smallest_positive_integer_with_remainders_l3948_394897


namespace square_fraction_above_line_l3948_394804

-- Define the square
def square_vertices : List (ℝ × ℝ) := [(4, 1), (7, 1), (7, 4), (4, 4)]

-- Define the line passing through two points
def line_points : List (ℝ × ℝ) := [(4, 3), (7, 1)]

-- Function to calculate the fraction of square area above the line
def fraction_above_line (square : List (ℝ × ℝ)) (line : List (ℝ × ℝ)) : ℚ :=
  sorry

-- Theorem statement
theorem square_fraction_above_line :
  fraction_above_line square_vertices line_points = 1/2 :=
sorry

end square_fraction_above_line_l3948_394804


namespace a_percentage_less_than_b_l3948_394886

def full_marks : ℕ := 500
def d_marks : ℕ := (80 * full_marks) / 100
def c_marks : ℕ := (80 * d_marks) / 100
def b_marks : ℕ := (125 * c_marks) / 100
def a_marks : ℕ := 360

theorem a_percentage_less_than_b :
  (b_marks - a_marks) * 100 / b_marks = 10 := by sorry

end a_percentage_less_than_b_l3948_394886


namespace rational_equation_zero_solution_l3948_394801

theorem rational_equation_zero_solution (x y z : ℚ) :
  x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end rational_equation_zero_solution_l3948_394801


namespace complex_sum_to_polar_l3948_394871

theorem complex_sum_to_polar : ∃ (r θ : ℝ), 
  5 * Complex.exp (3 * Real.pi * Complex.I / 4) + 5 * Complex.exp (-3 * Real.pi * Complex.I / 4) = r * Complex.exp (θ * Complex.I) ∧ 
  r = -5 * Real.sqrt 2 ∧ 
  θ = Real.pi := by
sorry

end complex_sum_to_polar_l3948_394871


namespace max_m_value_l3948_394808

def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ m + 1}

theorem max_m_value (m : ℝ) (h : B m ⊆ A) : m ≤ 3 := by
  sorry

end max_m_value_l3948_394808


namespace smaller_number_problem_l3948_394875

theorem smaller_number_problem (x y : ℝ) : 
  y = 3 * x + 11 → x + y = 55 → x = 11 := by sorry

end smaller_number_problem_l3948_394875


namespace line_inclination_l3948_394899

/-- Given a line with equation y = √3x + 2, its angle of inclination is π/3 -/
theorem line_inclination (x y : ℝ) :
  y = Real.sqrt 3 * x + 2 → 
  ∃ θ : ℝ, θ ∈ Set.Icc 0 π ∧ θ = π / 3 ∧ Real.tan θ = Real.sqrt 3 := by
sorry

end line_inclination_l3948_394899


namespace quadratic_equation_solution_l3948_394879

theorem quadratic_equation_solution (x : ℝ) :
  2 * x^2 + 2 * x - 1 = 0 ↔ x = (-1 + Real.sqrt 3) / 2 ∨ x = (-1 - Real.sqrt 3) / 2 := by
sorry

end quadratic_equation_solution_l3948_394879


namespace exponent_equation_solution_l3948_394816

theorem exponent_equation_solution : ∃ n : ℤ, 5^3 - 7 = 6^2 + n ∧ n = 82 := by
  sorry

end exponent_equation_solution_l3948_394816


namespace reflection_line_equation_l3948_394866

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Triangle PQR -/
def triangle_PQR : (Point2D × Point2D × Point2D) :=
  (⟨1, 2⟩, ⟨6, 7⟩, ⟨-3, 5⟩)

/-- Reflected triangle P'Q'R' -/
def reflected_triangle : (Point2D × Point2D × Point2D) :=
  (⟨1, -4⟩, ⟨6, -9⟩, ⟨-3, -7⟩)

/-- Line of reflection -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: The line of reflection for the given triangle is y = -1 -/
theorem reflection_line_equation : 
  ∃ (M : Line), M.a = 0 ∧ M.b = 1 ∧ M.c = 1 ∧
  (∀ (P : Point2D) (P' : Point2D), 
    (P = triangle_PQR.1 ∧ P' = reflected_triangle.1) ∨
    (P = triangle_PQR.2.1 ∧ P' = reflected_triangle.2.1) ∨
    (P = triangle_PQR.2.2 ∧ P' = reflected_triangle.2.2) →
    M.a * P.x + M.b * P.y + M.c = M.a * P'.x + M.b * P'.y + M.c) :=
sorry

end reflection_line_equation_l3948_394866


namespace geometric_sequence_fourth_term_l3948_394853

/-- Given a geometric sequence where the first term is 512 and the 6th term is 8,
    the 4th term is 64. -/
theorem geometric_sequence_fourth_term : ∀ (a : ℕ → ℝ),
  (∀ n, a (n + 1) = a n * (a 1)⁻¹ * a 0) →  -- Geometric sequence definition
  a 0 = 512 →                               -- First term is 512
  a 5 = 8 →                                 -- 6th term is 8
  a 3 = 64 :=                               -- 4th term is 64
by
  sorry

end geometric_sequence_fourth_term_l3948_394853


namespace choose_officers_count_l3948_394857

/-- Represents a club with boys and girls -/
structure Club where
  total_members : ℕ
  boys : ℕ
  girls : ℕ

/-- Calculates the number of ways to choose officers in the club -/
def choose_officers (club : Club) : ℕ :=
  club.boys * club.girls * (club.boys - 1) +
  club.girls * club.boys * (club.girls - 1)

/-- The main theorem stating the number of ways to choose officers -/
theorem choose_officers_count (club : Club)
  (h1 : club.total_members = 25)
  (h2 : club.boys = 12)
  (h3 : club.girls = 13)
  (h4 : club.total_members = club.boys + club.girls) :
  choose_officers club = 3588 := by
  sorry

#eval choose_officers ⟨25, 12, 13⟩

end choose_officers_count_l3948_394857


namespace harry_green_weights_l3948_394856

/-- Represents the weight configuration of Harry's custom creation at the gym -/
structure WeightConfiguration where
  blue_weight : ℕ        -- Weight of each blue weight in pounds
  green_weight : ℕ       -- Weight of each green weight in pounds
  bar_weight : ℕ         -- Weight of the bar in pounds
  num_blue : ℕ           -- Number of blue weights used
  total_weight : ℕ       -- Total weight of the custom creation in pounds

/-- Calculates the number of green weights in Harry's custom creation -/
def num_green_weights (config : WeightConfiguration) : ℕ :=
  (config.total_weight - config.bar_weight - config.num_blue * config.blue_weight) / config.green_weight

/-- Theorem stating that Harry put 5 green weights on the bar -/
theorem harry_green_weights :
  let config : WeightConfiguration := {
    blue_weight := 2,
    green_weight := 3,
    bar_weight := 2,
    num_blue := 4,
    total_weight := 25
  }
  num_green_weights config = 5 := by
  sorry

end harry_green_weights_l3948_394856


namespace january_rainfall_l3948_394829

theorem january_rainfall (first_week : ℝ) (second_week : ℝ) :
  second_week = 1.5 * first_week →
  second_week = 21 →
  first_week + second_week = 35 := by
sorry

end january_rainfall_l3948_394829


namespace minute_hand_rotation_1_to_3_20_l3948_394837

/-- The number of radians a clock's minute hand turns through in a given time interval -/
def minute_hand_rotation (start_hour start_minute end_hour end_minute : ℕ) : ℝ :=
  sorry

/-- The number of radians a clock's minute hand turns through from 1:00 to 3:20 -/
theorem minute_hand_rotation_1_to_3_20 :
  minute_hand_rotation 1 0 3 20 = -14/3 * π :=
sorry

end minute_hand_rotation_1_to_3_20_l3948_394837


namespace equation_solution_l3948_394884

theorem equation_solution : ∃ x : ℝ, (24 - 5 = 3 + x) ∧ (x = 16) := by
  sorry

end equation_solution_l3948_394884


namespace complex_equation_sum_l3948_394865

theorem complex_equation_sum (a b : ℝ) :
  (1 - Complex.I) * (a + Complex.I) = 3 - b * Complex.I →
  a + b = 3 := by
  sorry

end complex_equation_sum_l3948_394865


namespace sum_of_altitudes_for_specific_line_l3948_394843

/-- A line in 2D space represented by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A triangle in 2D space represented by its three vertices -/
structure Triangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ

/-- Function to create a triangle from a line that intersects the coordinate axes -/
def triangleFromLine (l : Line) : Triangle := sorry

/-- Function to calculate the sum of altitudes of a triangle -/
def sumOfAltitudes (t : Triangle) : ℝ := sorry

/-- Theorem stating that for the given line, the sum of altitudes of the formed triangle
    is equal to 23 + 60/√409 -/
theorem sum_of_altitudes_for_specific_line :
  let l : Line := { a := 20, b := 3, c := 60 }
  let t : Triangle := triangleFromLine l
  sumOfAltitudes t = 23 + 60 / Real.sqrt 409 := by sorry

end sum_of_altitudes_for_specific_line_l3948_394843


namespace birds_joining_fence_l3948_394895

theorem birds_joining_fence (initial_birds : ℕ) (total_birds : ℕ) (joined_birds : ℕ) : 
  initial_birds = 1 → total_birds = 5 → joined_birds = total_birds - initial_birds → joined_birds = 4 := by
  sorry

end birds_joining_fence_l3948_394895


namespace arithmetic_calculations_l3948_394822

theorem arithmetic_calculations :
  ((-3) - (-5) - 6 + (-4) = -8) ∧
  ((1/9 + 1/6 - 1/2) / (-1/18) = 4) ∧
  (-1^4 + |3-6| - 2 * (-2)^2 = -6) :=
by
  sorry

end arithmetic_calculations_l3948_394822


namespace range_of_m_l3948_394844

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > b ∧ b > 0 ∧ a = 8 - m ∧ b = 2*m - 1

def q (m : ℝ) : Prop := (m + 1) * (m - 2) < 0

-- Define the theorem
theorem range_of_m : 
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ∈ Set.Ioo (-1 : ℝ) (1/2) ∪ Set.Icc 2 3 :=
sorry

end range_of_m_l3948_394844


namespace raspberry_pies_count_l3948_394811

/-- The total number of pies -/
def total_pies : ℕ := 36

/-- The ratio of apple pies -/
def apple_ratio : ℕ := 1

/-- The ratio of blueberry pies -/
def blueberry_ratio : ℕ := 3

/-- The ratio of cherry pies -/
def cherry_ratio : ℕ := 2

/-- The ratio of raspberry pies -/
def raspberry_ratio : ℕ := 4

/-- The sum of all ratios -/
def total_ratio : ℕ := apple_ratio + blueberry_ratio + cherry_ratio + raspberry_ratio

/-- Theorem: The number of raspberry pies is 14.4 -/
theorem raspberry_pies_count : 
  (total_pies : ℚ) * raspberry_ratio / total_ratio = 14.4 := by
  sorry

end raspberry_pies_count_l3948_394811


namespace inequality_solution_l3948_394824

theorem inequality_solution (x : ℝ) :
  (2 / (x^2 + 1) > 4 / x + 5 / 2) ↔ -2 < x ∧ x < 0 := by
  sorry

end inequality_solution_l3948_394824


namespace prism_volume_is_six_times_pyramid_volume_l3948_394858

/-- A regular quadrilateral prism with an inscribed pyramid -/
structure PrismWithPyramid where
  /-- Side length of the prism's base -/
  a : ℝ
  /-- Height of the prism -/
  h : ℝ
  /-- Volume of the inscribed pyramid -/
  V : ℝ
  /-- The inscribed pyramid has vertices at the center of the upper base
      and the midpoints of the sides of the lower base -/
  pyramid_vertices : Unit

/-- The volume of the prism is 6 times the volume of the inscribed pyramid -/
theorem prism_volume_is_six_times_pyramid_volume (p : PrismWithPyramid) :
  p.a^2 * p.h = 6 * p.V := by
  sorry

end prism_volume_is_six_times_pyramid_volume_l3948_394858


namespace partnership_profit_share_l3948_394893

/-- Given a partnership with three investors A, B, and C, where A invests 3 times as much as B
    and 2/3 of what C invests, prove that C's share of a total profit of 11000 is (9/17) * 11000. -/
theorem partnership_profit_share (a b c : ℝ) (profit : ℝ) : 
  a = 3 * b → 
  a = (2/3) * c → 
  profit = 11000 → 
  c * profit / (a + b + c) = (9/17) * 11000 := by
  sorry

end partnership_profit_share_l3948_394893


namespace perfect_square_sequence_l3948_394876

theorem perfect_square_sequence (a b : ℤ) 
  (h : ∀ n : ℕ, ∃ x : ℤ, 2^n * a + b = x^2) : 
  a = 0 := by
  sorry

end perfect_square_sequence_l3948_394876


namespace semicircle_perimeter_specific_semicircle_perimeter_l3948_394896

/-- The perimeter of a semi-circle with radius r is equal to π * r + 2 * r -/
theorem semicircle_perimeter (r : ℝ) (h : r > 0) :
  let perimeter := π * r + 2 * r
  perimeter = π * r + 2 * r :=
by sorry

/-- The perimeter of a semi-circle with radius 6.7 cm is approximately 34.45 cm -/
theorem specific_semicircle_perimeter :
  let r : ℝ := 6.7
  let perimeter := π * r + 2 * r
  ∃ ε > 0, |perimeter - 34.45| < ε :=
by sorry

end semicircle_perimeter_specific_semicircle_perimeter_l3948_394896


namespace graham_younger_than_mark_l3948_394883

/-- Represents a person with a birth year and month -/
structure Person where
  birthYear : ℕ
  birthMonth : ℕ
  deriving Repr

def currentYear : ℕ := 2021
def currentMonth : ℕ := 2

def Mark : Person := { birthYear := 1976, birthMonth := 1 }

def JaniceAge : ℕ := 21

/-- Calculates the age of a person in years -/
def age (p : Person) : ℕ :=
  if currentMonth >= p.birthMonth then
    currentYear - p.birthYear
  else
    currentYear - p.birthYear - 1

/-- Calculates Graham's age based on Janice's age -/
def GrahamAge : ℕ := 2 * JaniceAge

theorem graham_younger_than_mark :
  age Mark - GrahamAge = 3 := by
  sorry

end graham_younger_than_mark_l3948_394883


namespace unit_circle_point_x_coordinate_l3948_394898

theorem unit_circle_point_x_coordinate 
  (P : ℝ × ℝ) 
  (α : ℝ) 
  (h1 : P.1 ≥ 0 ∧ P.2 ≥ 0) -- P is in the first quadrant
  (h2 : P.1^2 + P.2^2 = 1) -- P is on the unit circle
  (h3 : P.1 = Real.cos α ∧ P.2 = Real.sin α) -- Definition of α
  (h4 : Real.cos (α + π/3) = -11/13) -- Given condition
  : P.1 = 1/26 := by
  sorry

end unit_circle_point_x_coordinate_l3948_394898


namespace chord_bisected_by_point_one_one_l3948_394890

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define a chord of the ellipse
def is_chord (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  is_on_ellipse x₁ y₁ ∧ is_on_ellipse x₂ y₂

-- Define the midpoint of a chord
def is_midpoint (x y x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2

-- Define a line equation
def line_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Theorem statement
theorem chord_bisected_by_point_one_one :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  is_chord x₁ y₁ x₂ y₂ →
  is_midpoint 1 1 x₁ y₁ x₂ y₂ →
  line_equation 4 9 (-13) x₁ y₁ ∧ line_equation 4 9 (-13) x₂ y₂ :=
sorry

end chord_bisected_by_point_one_one_l3948_394890


namespace problem_1_l3948_394813

theorem problem_1 : 2^2 - 2023^0 + |3 - Real.pi| = Real.pi := by sorry

end problem_1_l3948_394813


namespace square_inequality_l3948_394859

theorem square_inequality (x : ℝ) : (x^2 + x + 1)^2 ≤ 3 * (x^4 + x^2 + 1) := by
  sorry

end square_inequality_l3948_394859


namespace dogwood_tree_count_l3948_394831

/-- The number of dogwood trees currently in the park -/
def current_trees : ℕ := 39

/-- The number of trees to be planted today -/
def planted_today : ℕ := 41

/-- The number of trees to be planted tomorrow -/
def planted_tomorrow : ℕ := 20

/-- The total number of trees after planting -/
def total_trees : ℕ := 100

/-- Theorem stating that the current number of trees plus the trees to be planted
    equals the total number of trees after planting -/
theorem dogwood_tree_count : 
  current_trees + planted_today + planted_tomorrow = total_trees := by
  sorry

end dogwood_tree_count_l3948_394831


namespace gini_coefficient_change_l3948_394827

/-- Represents a region in the country -/
structure Region where
  population : ℕ
  ppc : ℝ → ℝ
  maxKits : ℝ

/-- Calculates the Gini coefficient given two regions -/
def giniCoefficient (r1 r2 : Region) : ℝ :=
  sorry

/-- Calculates the new Gini coefficient after collaboration -/
def newGiniCoefficient (r1 r2 : Region) (compensation : ℝ) : ℝ :=
  sorry

theorem gini_coefficient_change
  (north : Region)
  (south : Region)
  (h1 : north.population = 24)
  (h2 : south.population = 6)
  (h3 : north.ppc = fun x => 13.5 - 9 * x)
  (h4 : south.ppc = fun x => 24 - 1.5 * x^2)
  (h5 : north.maxKits = 18)
  (h6 : south.maxKits = 12)
  (setPrice : ℝ)
  (h7 : setPrice = 6000)
  (compensation : ℝ)
  (h8 : compensation = 109983) :
  (giniCoefficient north south = 0.2) ∧
  (newGiniCoefficient north south compensation = 0.199) :=
sorry

end gini_coefficient_change_l3948_394827


namespace translation_problem_l3948_394878

/-- A translation in the complex plane. -/
def Translation (w : ℂ) : ℂ → ℂ := fun z ↦ z + w

/-- The theorem statement -/
theorem translation_problem (T : ℂ → ℂ) (h : T (1 + 3*I) = 4 + 6*I) :
  T (2 - I) = 5 + 2*I := by
  sorry

end translation_problem_l3948_394878


namespace perpendicular_vectors_x_value_l3948_394839

/-- Given two vectors a and b in ℝ², where a = (x, x+1) and b = (1, 2),
    if a is perpendicular to b, then x = -2/3 -/
theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![x, x + 1]
  let b : Fin 2 → ℝ := ![1, 2]
  (∀ i, i < 2 → a i * b i = 0) → x = -2/3 := by
sorry

end perpendicular_vectors_x_value_l3948_394839


namespace floor_times_self_eq_54_l3948_394812

theorem floor_times_self_eq_54 (x : ℝ) :
  x > 0 ∧ (⌊x⌋ : ℝ) * x = 54 → x = 54 / 7 := by
  sorry

end floor_times_self_eq_54_l3948_394812


namespace intersection_A_complement_B_l3948_394819

def A : Set ℤ := {1, 2, 3, 5, 7}
def B : Set ℤ := {x : ℤ | 1 < x ∧ x ≤ 6}
def U : Set ℤ := A ∪ B

theorem intersection_A_complement_B : A ∩ (U \ B) = {1, 7} := by
  sorry

end intersection_A_complement_B_l3948_394819


namespace margin_formula_in_terms_of_selling_price_l3948_394854

/-- Prove that the margin formula can be expressed in terms of selling price -/
theorem margin_formula_in_terms_of_selling_price 
  (n : ℝ) (C S M : ℝ) 
  (h1 : M = (C + S) / n) 
  (h2 : M = S - C) : 
  M = 2 * S / (n + 1) :=
sorry

end margin_formula_in_terms_of_selling_price_l3948_394854


namespace parallel_line_distance_l3948_394861

/-- Represents a circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  /-- Radius of the circle -/
  radius : ℝ
  /-- Distance between adjacent parallel lines -/
  line_distance : ℝ
  /-- Length of the first chord -/
  chord1 : ℝ
  /-- Length of the second chord -/
  chord2 : ℝ
  /-- Length of the third chord -/
  chord3 : ℝ
  /-- Assertion that the first and third chords are equal -/
  chord1_eq_chord3 : chord1 = chord3
  /-- Assertion that the first chord has length 42 -/
  chord1_length : chord1 = 42
  /-- Assertion that the second chord has length 40 -/
  chord2_length : chord2 = 40

/-- Theorem stating that the distance between adjacent parallel lines is √(92/11) -/
theorem parallel_line_distance (c : CircleWithParallelLines) : 
  c.line_distance = Real.sqrt (92 / 11) := by
  sorry

end parallel_line_distance_l3948_394861


namespace bus_journey_distance_l3948_394887

/-- Given a bus journey with the following parameters:
  * total_distance: The total distance covered by the bus
  * speed1: The first speed at which the bus travels for part of the journey
  * speed2: The second speed at which the bus travels for the remaining part of the journey
  * total_time: The total time taken for the entire journey

  This theorem proves that the distance covered at the first speed (speed1) is equal to
  the calculated value when the given conditions are met.
-/
theorem bus_journey_distance (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ)
  (h1 : total_distance = 250)
  (h2 : speed1 = 40)
  (h3 : speed2 = 60)
  (h4 : total_time = 5)
  (h5 : speed1 > 0)
  (h6 : speed2 > 0) :
  ∃ (distance1 : ℝ),
    distance1 / speed1 + (total_distance - distance1) / speed2 = total_time ∧
    distance1 = 100 := by
  sorry


end bus_journey_distance_l3948_394887


namespace ratio_common_value_l3948_394835

theorem ratio_common_value (x y z : ℝ) (k : ℝ) 
  (h1 : (x + y) / z = k)
  (h2 : (x + z) / y = k)
  (h3 : (y + z) / x = k)
  (h4 : x ≠ 0)
  (h5 : y ≠ 0)
  (h6 : z ≠ 0) :
  k = -1 ∨ k = 2 := by
  sorry

end ratio_common_value_l3948_394835


namespace problem_solution_l3948_394864

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

theorem problem_solution (n : ℕ) (h1 : n = 1221) :
  (∃ (d : Finset ℕ), d = {x : ℕ | x ∣ n} ∧ d.card = 8) ∧
  (∃ (d1 d2 d3 : ℕ), d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n ∧
    d1 < d2 ∧ d2 < d3 ∧
    ∀ (x : ℕ), x ∣ n → x ≤ d1 ∨ x = d2 ∨ x ≥ d3 ∧
    d1 + d2 + d3 = 15) ∧
  is_four_digit n ∧
  (∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    n = p * q * r ∧ p - 5 * q = 2 * r) :=
by sorry

end problem_solution_l3948_394864


namespace beta_value_l3948_394814

/-- Given α = 2023°, if β has the same terminal side as α and β ∈ (0, 2π), then β = 223π/180 -/
theorem beta_value (α β : Real) : 
  α = 2023 * (π / 180) →
  (∃ k : ℤ, β = α + k * 2 * π) →
  β ∈ Set.Ioo 0 (2 * π) →
  β = 223 * (π / 180) := by
  sorry

end beta_value_l3948_394814


namespace prob_HHT_fair_coin_l3948_394846

/-- A fair coin has equal probability of heads and tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of a sequence of independent events is the product of their individual probabilities -/
def prob_independent_events (p q r : ℝ) : ℝ := p * q * r

/-- The probability of getting heads on first two flips and tails on third flip for a fair coin -/
theorem prob_HHT_fair_coin (p : ℝ) (h : fair_coin p) : 
  prob_independent_events p p (1 - p) = 1/8 := by
  sorry

end prob_HHT_fair_coin_l3948_394846


namespace smallest_composite_no_small_factors_l3948_394869

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 20 → p.Prime → ¬(p ∣ n)

theorem smallest_composite_no_small_factors : 
  (is_composite 667 ∧ has_no_small_prime_factors 667) ∧ 
  (∀ m : ℕ, m < 667 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end smallest_composite_no_small_factors_l3948_394869


namespace probability_opposite_rooms_is_one_fifth_l3948_394845

/-- Represents a hotel with 6 rooms -/
structure Hotel :=
  (rooms : Fin 6 → ℕ)
  (opposite : Fin 3 → Fin 2 → Fin 6)
  (opposite_bijective : ∀ i, Function.Bijective (opposite i))

/-- Represents the random selection of room keys by 6 people -/
def RoomSelection := Fin 6 → Fin 6

/-- The probability of two specific people selecting opposite rooms -/
def probability_opposite_rooms (h : Hotel) : ℚ :=
  1 / 5

/-- Theorem stating that the probability of two specific people
    selecting opposite rooms is 1/5 -/
theorem probability_opposite_rooms_is_one_fifth (h : Hotel) :
  probability_opposite_rooms h = 1 / 5 := by
  sorry

end probability_opposite_rooms_is_one_fifth_l3948_394845


namespace inequality_proof_l3948_394805

theorem inequality_proof (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1/3 := by
sorry

end inequality_proof_l3948_394805


namespace calculate_expression_l3948_394873

theorem calculate_expression : (81 : ℝ) ^ (1/4) * (81 : ℝ) ^ (1/5) * 2 = 20.09 := by
  sorry

end calculate_expression_l3948_394873


namespace value_of_n_l3948_394860

theorem value_of_n (x : ℝ) (n : ℝ) 
  (h1 : Real.log (Real.sin x) + Real.log (Real.cos x) = -1)
  (h2 : Real.tan x = Real.sqrt 3)
  (h3 : Real.log (Real.sin x + Real.cos x) = (1/3) * (Real.log n - 1)) :
  n = Real.exp (3 * (-1/2 + Real.log (1 + 1 / Real.sqrt (Real.sqrt 3))) + 1) := by
sorry

end value_of_n_l3948_394860


namespace installation_charge_company_x_l3948_394832

/-- Represents a company's pricing for an air conditioner --/
structure CompanyPricing where
  price : ℝ
  surcharge_rate : ℝ
  installation_charge : ℝ

/-- Calculates the total cost for a company --/
def total_cost (c : CompanyPricing) : ℝ :=
  c.price + c.price * c.surcharge_rate + c.installation_charge

theorem installation_charge_company_x (
  company_x : CompanyPricing)
  (company_y : CompanyPricing)
  (h1 : company_x.price = 575)
  (h2 : company_x.surcharge_rate = 0.04)
  (h3 : company_y.price = 530)
  (h4 : company_y.surcharge_rate = 0.03)
  (h5 : company_y.installation_charge = 93)
  (h6 : total_cost company_x - total_cost company_y = 41.60) :
  company_x.installation_charge = 82.50 := by
  sorry

end installation_charge_company_x_l3948_394832


namespace line_slope_angle_l3948_394818

theorem line_slope_angle : 
  let x : ℝ → ℝ := λ t => 3 + t * Real.sin (π / 6)
  let y : ℝ → ℝ := λ t => -t * Real.cos (π / 6)
  (∃ m : ℝ, ∀ t₁ t₂ : ℝ, t₁ ≠ t₂ → 
    (y t₂ - y t₁) / (x t₂ - x t₁) = m ∧ 
    Real.arctan m = 2 * π / 3) :=
by sorry

end line_slope_angle_l3948_394818


namespace inscribed_parallelogram_sides_l3948_394825

/-- Triangle ABC with inscribed parallelogram BKLM -/
structure InscribedParallelogram where
  -- Side lengths of triangle ABC
  AB : ℝ
  BC : ℝ
  -- Sides of parallelogram BKLM
  BM : ℝ
  BK : ℝ
  -- Condition that BKLM is inscribed in ABC
  inscribed : BM ≤ BC ∧ BK ≤ AB

/-- The theorem stating the possible side lengths of the inscribed parallelogram -/
theorem inscribed_parallelogram_sides
  (T : InscribedParallelogram)
  (h_AB : T.AB = 18)
  (h_BC : T.BC = 12)
  (h_area : T.BM * T.BK = 48) :
  (T.BM = 8 ∧ T.BK = 6) ∨ (T.BM = 4 ∧ T.BK = 12) := by
  sorry

#check inscribed_parallelogram_sides

end inscribed_parallelogram_sides_l3948_394825


namespace sum_inequality_l3948_394888

theorem sum_inequality (a b c : ℕ+) (h : (a * b * c : ℚ) = 1) :
  (1 / (b * (a + b)) + 1 / (c * (b + c)) + 1 / (a * (c + a)) : ℚ) ≥ 3/2 := by
  sorry

end sum_inequality_l3948_394888


namespace monica_savings_l3948_394855

theorem monica_savings (weekly_saving : ℕ) (weeks_per_cycle : ℕ) (num_cycles : ℕ) 
  (h1 : weekly_saving = 15)
  (h2 : weeks_per_cycle = 60)
  (h3 : num_cycles = 5) :
  weekly_saving * weeks_per_cycle * num_cycles = 4500 := by
  sorry

end monica_savings_l3948_394855


namespace smallest_three_digit_sum_product_l3948_394889

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digits_sum_product (a b c : ℕ) : ℕ := a + b + c + a*b + b*c + a*c + a*b*c

theorem smallest_three_digit_sum_product :
  ∃ (n : ℕ) (a b c : ℕ),
    is_three_digit n ∧
    n = 100*a + 10*b + c ∧
    n = digits_sum_product a b c ∧
    (∀ (m : ℕ) (x y z : ℕ),
      is_three_digit m ∧
      m = 100*x + 10*y + z ∧
      m = digits_sum_product x y z →
      n ≤ m) ∧
    n = 199 := by
  sorry

end smallest_three_digit_sum_product_l3948_394889


namespace base_conversion_and_arithmetic_l3948_394826

-- Define the base conversion functions
def to_base_10 (digits : List Nat) (base : Nat) : Rat :=
  (digits.reverse.enum.map (λ (i, d) => d * base^i)).sum

-- Define the given numbers in their respective bases
def num1 : Rat := 2468
def num2 : Rat := to_base_10 [1, 2, 1] 3
def num3 : Rat := to_base_10 [6, 5, 4, 3] 7
def num4 : Rat := to_base_10 [6, 7, 8, 9] 9

-- State the theorem
theorem base_conversion_and_arithmetic :
  num1 / num2 + num3 - num4 = -5857.75 := by sorry

end base_conversion_and_arithmetic_l3948_394826


namespace unique_solution_ceiling_equation_l3948_394892

theorem unique_solution_ceiling_equation :
  ∃! b : ℝ, b + ⌈b⌉ = 17.8 ∧ b = 8.8 := by sorry

end unique_solution_ceiling_equation_l3948_394892


namespace hemisphere_surface_area_l3948_394803

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 144 * π) :
  2 * π * r^2 + π * r^2 = 432 * π := by
  sorry

end hemisphere_surface_area_l3948_394803


namespace part_one_part_two_l3948_394820

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Part 1 of the problem -/
theorem part_one (t : Triangle) (h1 : t.A = π / 3) (h2 : t.a = 4 * Real.sqrt 3) (h3 : t.b = 4 * Real.sqrt 2) :
  t.B = π / 4 := by
  sorry

/-- Part 2 of the problem -/
theorem part_two (t : Triangle) (h1 : t.a = 3 * Real.sqrt 3) (h2 : t.c = 2) (h3 : t.B = 5 * π / 6) :
  t.b = 7 := by
  sorry

end part_one_part_two_l3948_394820


namespace speed_of_A_is_correct_l3948_394882

/-- Represents the speed of boy A in mph -/
def speed_A : ℝ := 7.5

/-- Represents the speed of boy B in mph -/
def speed_B : ℝ := speed_A + 5

/-- Represents the speed of boy C in mph -/
def speed_C : ℝ := speed_A + 3

/-- Represents the total distance between Port Jervis and Poughkeepsie in miles -/
def total_distance : ℝ := 80

/-- Represents the distance from Poughkeepsie where A and B meet in miles -/
def meeting_distance : ℝ := 20

theorem speed_of_A_is_correct :
  speed_A * (total_distance / speed_B + meeting_distance / speed_B) =
  total_distance - meeting_distance := by sorry

end speed_of_A_is_correct_l3948_394882


namespace globe_surface_parts_l3948_394821

/-- Represents a globe with a given number of parallels and meridians. -/
structure Globe where
  parallels : ℕ
  meridians : ℕ

/-- Calculates the number of parts the surface of a globe is divided into. -/
def surfaceParts (g : Globe) : ℕ :=
  (g.parallels + 1) * g.meridians

/-- Theorem: A globe with 17 parallels and 24 meridians has its surface divided into 432 parts. -/
theorem globe_surface_parts :
  let g : Globe := { parallels := 17, meridians := 24 }
  surfaceParts g = 432 := by
  sorry

end globe_surface_parts_l3948_394821


namespace quadrupled_base_exponent_l3948_394894

theorem quadrupled_base_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (4 * a)^(4 * b) = a^b * x^2 → x = 16^b * a^(3/2 * b) := by
  sorry

end quadrupled_base_exponent_l3948_394894


namespace parakeet_cost_graph_is_finite_distinct_points_l3948_394862

def parakeet_cost (n : ℕ) : ℚ :=
  if n ≤ 10 then 20 * n
  else if n ≤ 20 then 18 * n
  else if n ≤ 25 then 18 * n
  else 0

def cost_graph : Set (ℕ × ℚ) :=
  {p | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 25 ∧ p = (n, parakeet_cost n)}

theorem parakeet_cost_graph_is_finite_distinct_points :
  Finite cost_graph ∧ ∀ p q : ℕ × ℚ, p ∈ cost_graph → q ∈ cost_graph → p ≠ q → p.1 ≠ q.1 :=
sorry

end parakeet_cost_graph_is_finite_distinct_points_l3948_394862


namespace direct_proportion_points_l3948_394807

/-- A direct proportion function passing through (-1, 2) also passes through (1, -2) -/
theorem direct_proportion_points : 
  ∀ (f : ℝ → ℝ), 
  (∃ k : ℝ, ∀ x, f x = k * x) → -- f is a direct proportion function
  f (-1) = 2 →                  -- f passes through (-1, 2)
  f 1 = -2                      -- f passes through (1, -2)
:= by sorry

end direct_proportion_points_l3948_394807


namespace flour_sugar_difference_l3948_394874

theorem flour_sugar_difference (recipe_sugar : ℕ) (recipe_flour : ℕ) (recipe_salt : ℕ) (flour_added : ℕ) :
  recipe_sugar = 9 →
  recipe_flour = 14 →
  recipe_salt = 40 →
  flour_added = 4 →
  recipe_flour - flour_added - recipe_sugar = 1 := by
sorry

end flour_sugar_difference_l3948_394874


namespace nine_steps_climb_l3948_394833

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

def ways_to_climb (n : ℕ) : ℕ := fibonacci (n + 1)

theorem nine_steps_climb : ways_to_climb 9 = 55 := by
  sorry

end nine_steps_climb_l3948_394833


namespace division_remainder_l3948_394815

theorem division_remainder : ∃ q : ℕ, 1234567 = 256 * q + 229 ∧ 229 < 256 := by
  sorry

end division_remainder_l3948_394815


namespace blueberry_bonnie_ratio_l3948_394848

/-- Represents the number of fruits eaten by each dog -/
structure DogFruits where
  apples : ℕ
  blueberries : ℕ
  bonnies : ℕ

/-- The problem setup -/
def fruitProblem (dogs : Vector DogFruits 3) : Prop :=
  let d1 := dogs.get 0
  let d2 := dogs.get 1
  let d3 := dogs.get 2
  d1.apples = 3 * d2.blueberries ∧
  d3.bonnies = 60 ∧
  d1.apples + d2.blueberries + d3.bonnies = 240

/-- The theorem to prove -/
theorem blueberry_bonnie_ratio (dogs : Vector DogFruits 3) 
  (h : fruitProblem dogs) : 
  (dogs.get 1).blueberries * 4 = (dogs.get 2).bonnies * 3 := by
  sorry


end blueberry_bonnie_ratio_l3948_394848
