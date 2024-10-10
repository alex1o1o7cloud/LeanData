import Mathlib

namespace paul_sandwich_consumption_l604_60488

def sandwiches_per_cycle : ℕ := 2 + 4 + 8

def study_days : ℕ := 6

def cycles : ℕ := study_days / 3

theorem paul_sandwich_consumption :
  cycles * sandwiches_per_cycle = 28 := by
  sorry

end paul_sandwich_consumption_l604_60488


namespace cookies_eaten_l604_60496

theorem cookies_eaten (initial : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  initial = 18 → remaining = 9 → eaten = initial - remaining → eaten = 9 := by
  sorry

end cookies_eaten_l604_60496


namespace continuity_point_sum_l604_60400

theorem continuity_point_sum (g : ℝ → ℝ) : 
  (∃ m₁ m₂ : ℝ, 
    (∀ x < m₁, g x = x^2 + 4) ∧ 
    (∀ x ≥ m₁, g x = 3*x + 6) ∧
    (∀ x < m₂, g x = x^2 + 4) ∧ 
    (∀ x ≥ m₂, g x = 3*x + 6) ∧
    (m₁^2 + 4 = 3*m₁ + 6) ∧
    (m₂^2 + 4 = 3*m₂ + 6) ∧
    (m₁ ≠ m₂)) →
  (∃ m₁ m₂ : ℝ, m₁ + m₂ = 3 ∧ 
    (∀ x < m₁, g x = x^2 + 4) ∧ 
    (∀ x ≥ m₁, g x = 3*x + 6) ∧
    (∀ x < m₂, g x = x^2 + 4) ∧ 
    (∀ x ≥ m₂, g x = 3*x + 6) ∧
    (m₁^2 + 4 = 3*m₁ + 6) ∧
    (m₂^2 + 4 = 3*m₂ + 6) ∧
    (m₁ ≠ m₂)) :=
by sorry

end continuity_point_sum_l604_60400


namespace initial_men_count_l604_60481

/-- Given a piece of work that can be completed by some number of men in 25 hours,
    or by 12 men in 75 hours, prove that the initial number of men is 36. -/
theorem initial_men_count : ℕ :=
  let initial_time : ℕ := 25
  let new_men_count : ℕ := 12
  let new_time : ℕ := 75
  36

#check initial_men_count

end initial_men_count_l604_60481


namespace hyperbola_condition_l604_60441

/-- The statement "ab < 0" is a necessary but not sufficient condition for "b < 0 < a" -/
theorem hyperbola_condition (a b : ℝ) : 
  (∃ (p q : Prop), (p ↔ a * b < 0) ∧ (q ↔ b < 0 ∧ 0 < a) ∧ 
  (q → p) ∧ ¬(p → q)) :=
sorry

end hyperbola_condition_l604_60441


namespace lisa_spoons_l604_60461

/-- The total number of spoons Lisa has -/
def total_spoons (num_children : ℕ) (spoons_per_child : ℕ) (decorative_spoons : ℕ) 
                 (large_spoons : ℕ) (teaspoons : ℕ) : ℕ :=
  num_children * spoons_per_child + decorative_spoons + large_spoons + teaspoons

/-- Proof that Lisa has 39 spoons in total -/
theorem lisa_spoons : 
  total_spoons 4 3 2 10 15 = 39 := by
  sorry

end lisa_spoons_l604_60461


namespace increasing_root_m_value_l604_60499

theorem increasing_root_m_value (m : ℝ) : 
  (∃ x : ℝ, (2 * x + 1) / (x - 3) = m / (3 - x) + 1 ∧ 
   ∀ y : ℝ, y > x → (2 * y + 1) / (y - 3) > m / (3 - y) + 1) → 
  m = -7 := by
sorry

end increasing_root_m_value_l604_60499


namespace inequality_solution_l604_60437

noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if a < -1 ∨ (0 < a ∧ a < 1) then
    {x | a < x ∧ x < 1/a}
  else if a = 1 ∨ a = -1 then
    ∅
  else if a > 1 ∨ (-1 < a ∧ a < 0) then
    {x | 1/a < x ∧ x < a}
  else
    ∅

theorem inequality_solution (a : ℝ) (h : a ≠ 0) :
  {x : ℝ | x^2 - (a + 1/a)*x + 1 < 0} = solution_set a :=
by sorry

end inequality_solution_l604_60437


namespace determinant_equality_l604_60477

theorem determinant_equality (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = 7 →
  Matrix.det !![p + r, q + s; r, s] = 7 := by
  sorry

end determinant_equality_l604_60477


namespace original_number_proof_l604_60440

theorem original_number_proof (x : ℝ) : 
  (x * 1.375 - x * 0.575 = 85) → x = 106.25 := by
  sorry

end original_number_proof_l604_60440


namespace car_growth_rates_l604_60451

/-- The number of cars in millions at the end of 2010 -/
def cars_2010 : ℝ := 1

/-- The number of cars in millions at the end of 2012 -/
def cars_2012 : ℝ := 1.44

/-- The maximum allowed number of cars in millions at the end of 2013 -/
def max_cars_2013 : ℝ := 1.5552

/-- The proportion of cars scrapped in 2013 -/
def scrap_rate : ℝ := 0.1

/-- The average annual growth rate of cars from 2010 to 2012 -/
def growth_rate_2010_2012 : ℝ := 0.2

/-- The maximum annual growth rate from 2012 to 2013 -/
def max_growth_rate_2012_2013 : ℝ := 0.18

theorem car_growth_rates :
  (cars_2010 * (1 + growth_rate_2010_2012)^2 = cars_2012) ∧
  (cars_2012 * (1 + max_growth_rate_2012_2013) * (1 - scrap_rate) ≤ max_cars_2013) := by
  sorry

end car_growth_rates_l604_60451


namespace f_one_root_iff_a_in_set_l604_60486

/-- A quadratic function f(x) = ax^2 + (3-a)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (3 - a) * x + 1

/-- The condition for a quadratic function to have exactly one root -/
def has_one_root (a : ℝ) : Prop :=
  (a = 0 ∧ ∃! x, f a x = 0) ∨
  (a ≠ 0 ∧ (3 - a)^2 - 4*a = 0)

/-- The theorem stating that f has only one common point with the x-axis iff a ∈ {0, 1, 9} -/
theorem f_one_root_iff_a_in_set :
  ∀ a : ℝ, has_one_root a ↔ a ∈ ({0, 1, 9} : Set ℝ) := by sorry

end f_one_root_iff_a_in_set_l604_60486


namespace total_campers_count_l604_60410

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 36

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 13

/-- The number of campers who went rowing in the evening -/
def evening_campers : ℕ := 49

/-- The total number of campers who went rowing -/
def total_campers : ℕ := morning_campers + afternoon_campers + evening_campers

theorem total_campers_count : total_campers = 98 := by
  sorry

end total_campers_count_l604_60410


namespace polynomial_simplification_l604_60456

theorem polynomial_simplification (y : ℝ) :
  (3 * y - 2) * (6 * y^12 + 3 * y^11 + 6 * y^10 + 3 * y^9) =
  18 * y^13 - 3 * y^12 + 12 * y^11 - 3 * y^10 - 6 * y^9 := by sorry

end polynomial_simplification_l604_60456


namespace item_pricing_and_profit_l604_60462

/-- Represents the pricing and profit calculation for an item -/
theorem item_pricing_and_profit (a : ℝ) :
  let original_price := a * (1 + 0.2)
  let current_price := original_price * 0.9
  let profit_per_unit := current_price - a
  (current_price = 1.08 * a) ∧
  (1000 * profit_per_unit = 80 * a) := by
  sorry

end item_pricing_and_profit_l604_60462


namespace same_university_probability_l604_60407

theorem same_university_probability (n : ℕ) (h : n = 5) :
  let total_outcomes := n * n
  let favorable_outcomes := n
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 5 := by
  sorry

end same_university_probability_l604_60407


namespace sixth_term_of_arithmetic_sequence_l604_60422

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sixth_term_of_arithmetic_sequence (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_3 : a 3 = 4) 
  (h_7 : a 7 = 10) : 
  a 6 = 17/2 := by
sorry

end sixth_term_of_arithmetic_sequence_l604_60422


namespace multiply_fractions_equals_thirty_l604_60421

theorem multiply_fractions_equals_thirty : 15 * (1 / 17) * 34 = 30 := by
  sorry

end multiply_fractions_equals_thirty_l604_60421


namespace domino_path_count_l604_60413

/-- The number of distinct paths from (0,0) to (m,n) on a grid -/
def grid_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- The grid dimensions -/
def grid_width : ℕ := 5
def grid_height : ℕ := 6

/-- The number of right and down steps required -/
def right_steps : ℕ := grid_width - 1
def down_steps : ℕ := grid_height - 1

theorem domino_path_count : grid_paths right_steps down_steps = 126 := by
  sorry

end domino_path_count_l604_60413


namespace megans_earnings_l604_60472

/-- Calculates the total earnings for a worker given their work schedule and hourly rate. -/
def total_earnings (hours_per_day : ℕ) (hourly_rate : ℚ) (days_per_month : ℕ) (num_months : ℕ) : ℚ :=
  (hours_per_day : ℚ) * hourly_rate * (days_per_month : ℚ) * (num_months : ℚ)

/-- Proves that Megan's total earnings for two months of work is $2400. -/
theorem megans_earnings :
  let hours_per_day : ℕ := 8
  let hourly_rate : ℚ := 15/2  -- $7.50 expressed as a rational number
  let days_per_month : ℕ := 20
  let num_months : ℕ := 2
  total_earnings hours_per_day hourly_rate days_per_month num_months = 2400 := by
  sorry


end megans_earnings_l604_60472


namespace teachers_present_l604_60420

/-- The number of teachers present in a program --/
def num_teachers (parents pupils total : ℕ) : ℕ :=
  total - (parents + pupils)

/-- Theorem: Given 73 parents, 724 pupils, and 1541 total people,
    there were 744 teachers present in the program --/
theorem teachers_present :
  num_teachers 73 724 1541 = 744 := by
  sorry

end teachers_present_l604_60420


namespace chess_team_selection_l604_60406

def boys : ℕ := 10
def girls : ℕ := 12
def team_boys : ℕ := 5
def team_girls : ℕ := 3

theorem chess_team_selection :
  (Nat.choose boys team_boys) * (Nat.choose girls team_girls) = 55440 :=
by sorry

end chess_team_selection_l604_60406


namespace circle_equation_from_diameter_l604_60403

/-- Given two points A and B as the diameter of a circle, 
    prove that the equation of the circle is as stated. -/
theorem circle_equation_from_diameter 
  (A B : ℝ × ℝ) 
  (hA : A = (4, 9)) 
  (hB : B = (6, -3)) : 
  ∃ (C : ℝ × ℝ) (r : ℝ), 
    C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ 
    r^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4 ∧
    ∀ (x y : ℝ), (x - C.1)^2 + (y - C.2)^2 = r^2 ↔ 
      (x - 5)^2 + (y - 3)^2 = 37 :=
by sorry

end circle_equation_from_diameter_l604_60403


namespace arithmetic_sequence_problem_l604_60411

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S₃ : ℝ) :
  (∀ n, a (n + 1) - a n = 4) →  -- Common difference is 4
  ((a 3 + 2) / 2 = Real.sqrt (2 * S₃)) →  -- Arithmetic mean = Geometric mean condition
  (S₃ = a 1 + a 2 + a 3) →  -- Definition of S₃
  (a 10 = 38) :=
by
  sorry

end arithmetic_sequence_problem_l604_60411


namespace platform_length_l604_60464

/-- The length of a platform given train specifications -/
theorem platform_length
  (train_length : ℝ)
  (time_tree : ℝ)
  (time_platform : ℝ)
  (h1 : train_length = 1200)
  (h2 : time_tree = 120)
  (h3 : time_platform = 160) :
  let train_speed := train_length / time_tree
  let platform_length := train_speed * time_platform - train_length
  platform_length = 400 :=
by sorry

end platform_length_l604_60464


namespace range_of_m_l604_60439

/-- The function f(x) = x² + mx - 1 --/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 1

/-- Theorem stating the range of m given the conditions --/
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc m (m + 1), f m x < 0) →
  m ∈ Set.Ioo (-Real.sqrt 2 / 2) 0 :=
by sorry

end range_of_m_l604_60439


namespace binomial_sum_theorem_l604_60452

theorem binomial_sum_theorem :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (5*x - 4)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 25 := by
sorry

end binomial_sum_theorem_l604_60452


namespace root_equation_solution_l604_60417

theorem root_equation_solution (a : ℝ) (n : ℕ) : 
  a^11 + a^7 + a^3 = 1 → (a^4 + a^3 = a^n + 1 ↔ n = 15) :=
by sorry

end root_equation_solution_l604_60417


namespace emma_toast_pieces_l604_60494

/-- Given a loaf of bread with an initial number of slices, 
    calculate the number of toast pieces that can be made 
    after some slices are eaten and leaving one slice remaining. --/
def toastPieces (initialSlices : ℕ) (eatenSlices : ℕ) (slicesPerToast : ℕ) : ℕ :=
  ((initialSlices - eatenSlices - 1) / slicesPerToast : ℕ)

/-- Theorem stating that given the specific conditions of the problem,
    the number of toast pieces made is 10. --/
theorem emma_toast_pieces : 
  toastPieces 27 6 2 = 10 := by sorry

end emma_toast_pieces_l604_60494


namespace RS_length_l604_60434

-- Define the triangle RFS
structure Triangle :=
  (R F S : ℝ × ℝ)

-- Define the given lengths
def FD : ℝ := 5
def DR : ℝ := 8
def FR : ℝ := 6
def FS : ℝ := 9

-- Define the angles
def angle_RFS (t : Triangle) : ℝ := sorry
def angle_FDR : ℝ := sorry

-- State the theorem
theorem RS_length (t : Triangle) :
  angle_RFS t = angle_FDR →
  FR = 6 →
  FS = 9 →
  ∃ (RS : ℝ), abs (RS - 10.25) < 0.01 := by sorry

end RS_length_l604_60434


namespace invisible_dots_sum_l604_60495

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The total number of dice -/
def num_dice : ℕ := 4

/-- The visible numbers on the stacked dice -/
def visible_numbers : List ℕ := [1, 1, 2, 3, 3, 4, 5, 5, 6]

/-- Theorem: The total number of dots not visible is 54 -/
theorem invisible_dots_sum : 
  num_dice * die_sum - visible_numbers.sum = 54 := by sorry

end invisible_dots_sum_l604_60495


namespace min_value_E_p_l604_60427

/-- Given an odd prime p and positive integers x and y, 
    the function E_p(x,y) has a lower bound. -/
theorem min_value_E_p (p : ℕ) (x y : ℕ) 
  (hp : Nat.Prime p ∧ Odd p) (hx : x > 0) (hy : y > 0) : 
  Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≥ 
  Real.sqrt (2 * p) - (Real.sqrt ((p - 1) / 2) + Real.sqrt ((p + 1) / 2)) :=
sorry

end min_value_E_p_l604_60427


namespace integer_solution_system_l604_60433

theorem integer_solution_system :
  ∀ A B C : ℤ,
  (A^2 - B^2 - C^2 = 1 ∧ B + C - A = 3) ↔
  ((A = 9 ∧ B = 8 ∧ C = 4) ∨
   (A = 9 ∧ B = 4 ∧ C = 8) ∨
   (A = -3 ∧ B = 2 ∧ C = -2) ∨
   (A = -3 ∧ B = -2 ∧ C = 2)) :=
by sorry


end integer_solution_system_l604_60433


namespace remaining_surface_area_l604_60478

/-- The surface area of the remaining part of a cube after cutting a smaller cube from its vertex -/
theorem remaining_surface_area (original_edge : ℝ) (small_edge : ℝ) 
  (h1 : original_edge = 9) 
  (h2 : small_edge = 2) : 
  6 * original_edge^2 - 3 * small_edge^2 + 3 * small_edge^2 = 486 :=
by sorry

end remaining_surface_area_l604_60478


namespace ship_speed_ratio_l604_60492

theorem ship_speed_ratio (downstream_speed upstream_speed average_speed : ℝ) 
  (h1 : downstream_speed / upstream_speed = 5 / 2) 
  (h2 : average_speed = (2 * downstream_speed * upstream_speed) / (downstream_speed + upstream_speed)) : 
  average_speed / downstream_speed = 4 / 7 := by
  sorry

end ship_speed_ratio_l604_60492


namespace smallest_fourth_number_l604_60431

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ := 
  (n / 10) + (n % 10)

def theorem_smallest_fourth_number (fourth : ℕ) : Prop :=
  is_two_digit fourth ∧
  (sum_of_digits 24 + sum_of_digits 58 + sum_of_digits 63 + sum_of_digits fourth) * 4 =
  (24 + 58 + 63 + fourth)

theorem smallest_fourth_number :
  ∃ (fourth : ℕ), theorem_smallest_fourth_number fourth ∧
  (∀ (n : ℕ), theorem_smallest_fourth_number n → fourth ≤ n) ∧
  fourth = 35 :=
sorry

end smallest_fourth_number_l604_60431


namespace circles_tangent_internally_l604_60491

/-- Two circles are tangent internally if the distance between their centers
    is equal to the difference of their radii --/
def are_tangent_internally (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2) = r1 - r2

/-- Given two circles with specified centers and radii, prove they are tangent internally --/
theorem circles_tangent_internally :
  let c1 : ℝ × ℝ := (0, 8)
  let c2 : ℝ × ℝ := (-6, 0)
  let r1 : ℝ := 12
  let r2 : ℝ := 2
  are_tangent_internally c1 c2 r1 r2 := by
  sorry


end circles_tangent_internally_l604_60491


namespace triangle_at_most_one_obtuse_l604_60482

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180

-- Define an obtuse angle
def is_obtuse (angle : ℝ) : Prop := angle > 90

-- Theorem statement
theorem triangle_at_most_one_obtuse (T : Triangle) :
  ¬ (∃ i j : Fin 3, i ≠ j ∧ is_obtuse (T.angles i) ∧ is_obtuse (T.angles j)) :=
sorry

end triangle_at_most_one_obtuse_l604_60482


namespace ab_max_and_reciprocal_sum_min_l604_60485

theorem ab_max_and_reciprocal_sum_min (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 10 * b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 10 * y = 1 ∧ a * b ≤ x * y) ∧
  (a * b ≤ 1 / 40) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 10 * y = 1 ∧ 1 / x + 1 / y ≥ 1 / a + 1 / b) ∧
  (1 / a + 1 / b ≥ 11 + 2 * Real.sqrt 10) :=
by sorry

end ab_max_and_reciprocal_sum_min_l604_60485


namespace solution_exists_l604_60463

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the equation
def equation (x : ℝ) : Prop :=
  cubeRoot (24 * x + cubeRoot (24 * x + 16)) = 14

-- Theorem statement
theorem solution_exists : ∃ x : ℝ, equation x ∧ x = 114 := by
  sorry

end solution_exists_l604_60463


namespace not_necessarily_no_mass_infection_l604_60471

/-- Represents the daily increase in suspected cases over 10 days -/
def DailyIncrease := Fin 10 → ℕ

/-- The sign of no mass infection -/
def NoMassInfection (d : DailyIncrease) : Prop :=
  ∀ i, d i ≤ 7

/-- The median of a DailyIncrease is 2 -/
def MedianIsTwo (d : DailyIncrease) : Prop :=
  ∃ (sorted : Fin 10 → ℕ), (∀ i j, i ≤ j → sorted i ≤ sorted j) ∧
    (∀ i, ∃ j, d j = sorted i) ∧
    sorted 4 = 2 ∧ sorted 5 = 2

/-- The mode of a DailyIncrease is 3 -/
def ModeIsThree (d : DailyIncrease) : Prop :=
  ∃ (count : ℕ → ℕ), (∀ n, count n = (Finset.univ.filter (λ i => d i = n)).card) ∧
    ∀ n, count 3 ≥ count n

theorem not_necessarily_no_mass_infection :
  ∃ d : DailyIncrease, MedianIsTwo d ∧ ModeIsThree d ∧ ¬NoMassInfection d :=
sorry

end not_necessarily_no_mass_infection_l604_60471


namespace min_sum_inequality_l604_60428

theorem min_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ 3 / Real.rpow 162 (1/3) :=
sorry

end min_sum_inequality_l604_60428


namespace job_completion_theorem_l604_60444

/-- The number of days it takes the initial group of machines to finish the job -/
def initial_days : ℕ := 40

/-- The number of additional machines added -/
def additional_machines : ℕ := 4

/-- The number of days it takes after adding more machines -/
def reduced_days : ℕ := 30

/-- The number of machines initially working on the job -/
def initial_machines : ℕ := 16

theorem job_completion_theorem :
  (initial_machines : ℚ) / initial_days = (initial_machines + additional_machines : ℚ) / reduced_days :=
by sorry

#check job_completion_theorem

end job_completion_theorem_l604_60444


namespace max_profit_fruit_transport_l604_60457

/-- Represents the fruit transportation problem --/
structure FruitTransport where
  totalCars : Nat
  totalCargo : Nat
  minCarsPerFruit : Nat
  cargoA : Nat
  cargoB : Nat
  cargoC : Nat
  profitA : Nat
  profitB : Nat
  profitC : Nat

/-- Calculates the profit for a given arrangement of cars --/
def calculateProfit (ft : FruitTransport) (x y : Nat) : Nat :=
  ft.profitA * ft.cargoA * x + ft.profitB * ft.cargoB * y + ft.profitC * ft.cargoC * (ft.totalCars - x - y)

/-- Theorem stating the maximum profit and optimal arrangement --/
theorem max_profit_fruit_transport (ft : FruitTransport)
  (h1 : ft.totalCars = 20)
  (h2 : ft.totalCargo = 120)
  (h3 : ft.minCarsPerFruit = 3)
  (h4 : ft.cargoA = 7)
  (h5 : ft.cargoB = 6)
  (h6 : ft.cargoC = 5)
  (h7 : ft.profitA = 1200)
  (h8 : ft.profitB = 1800)
  (h9 : ft.profitC = 1500)
  (h10 : ∀ x y, x + y ≤ ft.totalCars → x ≥ ft.minCarsPerFruit → y ≥ ft.minCarsPerFruit → 
    ft.totalCars - x - y ≥ ft.minCarsPerFruit → ft.cargoA * x + ft.cargoB * y + ft.cargoC * (ft.totalCars - x - y) = ft.totalCargo) :
  ∃ (x y : Nat), x = 3 ∧ y = 14 ∧ calculateProfit ft x y = 198900 ∧
    ∀ (a b : Nat), a + b ≤ ft.totalCars → a ≥ ft.minCarsPerFruit → b ≥ ft.minCarsPerFruit → 
      ft.totalCars - a - b ≥ ft.minCarsPerFruit → calculateProfit ft a b ≤ 198900 :=
by sorry


end max_profit_fruit_transport_l604_60457


namespace matrix_not_invertible_l604_60424

def A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![2 + x, 9; 4 - x, 10]

theorem matrix_not_invertible (x : ℝ) : 
  ¬(IsUnit (A x).det) ↔ x = 16/19 := by
  sorry

end matrix_not_invertible_l604_60424


namespace max_m_is_maximum_l604_60455

/-- The maximum value of m for which the given conditions hold --/
def max_m : ℝ := 9

/-- Condition that abc ≤ 1/4 --/
def condition_product (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c ≤ 1/4

/-- Condition that 1/a² + 1/b² + 1/c² < m --/
def condition_sum (a b c m : ℝ) : Prop :=
  1/a^2 + 1/b^2 + 1/c^2 < m

/-- Condition that a, b, c can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem stating that max_m is the maximum value satisfying all conditions --/
theorem max_m_is_maximum :
  ∀ m : ℝ, m > 0 →
  (∀ a b c : ℝ, condition_product a b c → condition_sum a b c m → can_form_triangle a b c) →
  m ≤ max_m :=
sorry

end max_m_is_maximum_l604_60455


namespace exists_winning_strategy_l604_60483

/-- Represents the state of the candy game -/
structure GameState where
  pile1 : Nat
  pile2 : Nat

/-- Defines a valid move in the game -/
def ValidMove (state : GameState) (newState : GameState) : Prop :=
  (newState.pile1 = state.pile1 ∧ newState.pile2 < state.pile2 ∧ (state.pile2 - newState.pile2) % state.pile1 = 0) ∨
  (newState.pile2 = state.pile2 ∧ newState.pile1 < state.pile1 ∧ (state.pile1 - newState.pile1) % state.pile2 = 0)

/-- Defines a winning state -/
def WinningState (state : GameState) : Prop :=
  state.pile1 = 0 ∨ state.pile2 = 0

/-- Theorem stating that there exists a winning strategy -/
theorem exists_winning_strategy :
  ∃ (strategy : GameState → GameState),
    let initialState := GameState.mk 1000 2357
    ∀ (state : GameState),
      state = initialState ∨ (∃ (prevState : GameState), ValidMove prevState state) →
      WinningState state ∨ (ValidMove state (strategy state) ∧ 
        ¬∃ (nextState : GameState), ValidMove (strategy state) nextState ∧ ¬WinningState nextState) :=
sorry


end exists_winning_strategy_l604_60483


namespace givenPointInFirstQuadrant_l604_60416

/-- A point in the Cartesian coordinate system. -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant in the Cartesian coordinate system. -/
def isInFirstQuadrant (p : CartesianPoint) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The given point (3,2) in the Cartesian coordinate system. -/
def givenPoint : CartesianPoint :=
  { x := 3, y := 2 }

/-- Theorem stating that the given point (3,2) lies in the first quadrant. -/
theorem givenPointInFirstQuadrant : isInFirstQuadrant givenPoint := by
  sorry

end givenPointInFirstQuadrant_l604_60416


namespace sum_of_divisors_450_prime_factors_l604_60465

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_450_prime_factors :
  ∃ (p q r : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    sum_of_divisors 450 = p * q * r ∧
    ∀ (s : ℕ), Nat.Prime s → s ∣ sum_of_divisors 450 → (s = p ∨ s = q ∨ s = r) :=
by sorry

end sum_of_divisors_450_prime_factors_l604_60465


namespace first_purchase_amount_l604_60438

/-- Represents the student-entrepreneur's mask selling scenario -/
structure MaskSelling where
  /-- Cost price of each package of masks (in rubles) -/
  cost_price : ℝ
  /-- Selling price of each package of masks (in rubles) -/
  selling_price : ℝ
  /-- Number of packages bought in the first purchase -/
  initial_quantity : ℝ
  /-- Profit from the first sale (in rubles) -/
  first_profit : ℝ
  /-- Profit from the second sale (in rubles) -/
  second_profit : ℝ

/-- Theorem stating the amount spent on the first purchase -/
theorem first_purchase_amount (m : MaskSelling)
  (h1 : m.first_profit = 1000)
  (h2 : m.second_profit = 1500)
  (h3 : m.selling_price > m.cost_price)
  (h4 : m.initial_quantity * m.selling_price = 
        (m.initial_quantity * m.selling_price / m.cost_price) * m.cost_price) :
  m.initial_quantity * m.cost_price = 2000 := by
  sorry


end first_purchase_amount_l604_60438


namespace snowboard_final_price_l604_60448

/-- 
Given a snowboard with an original price and two successive discounts,
calculate the final price after both discounts are applied.
-/
theorem snowboard_final_price 
  (original_price : ℝ)
  (friday_discount : ℝ)
  (monday_discount : ℝ)
  (h1 : original_price = 200)
  (h2 : friday_discount = 0.4)
  (h3 : monday_discount = 0.25) :
  original_price * (1 - friday_discount) * (1 - monday_discount) = 90 :=
by sorry

#check snowboard_final_price

end snowboard_final_price_l604_60448


namespace new_person_weight_l604_60497

/-- Given a group of 8 people where one person weighing 65 kg is replaced by a new person,
    and the average weight of the group increases by 4.2 kg,
    prove that the weight of the new person is 98.6 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_increase : Real) (replaced_weight : Real) :
  initial_count = 8 →
  weight_increase = 4.2 →
  replaced_weight = 65 →
  (initial_count : Real) * weight_increase + replaced_weight = 98.6 :=
by sorry

end new_person_weight_l604_60497


namespace smallest_valid_n_l604_60458

def is_valid_sequence (n : ℕ) (xs : List ℕ) : Prop :=
  xs.length = n ∧
  (∀ x ∈ xs, 1 ≤ x ∧ x ≤ n) ∧
  xs.sum = n * (n + 1) / 2 ∧
  xs.prod = Nat.factorial n ∧
  xs.toFinset ≠ Finset.range n

theorem smallest_valid_n : 
  (∀ m < 9, ¬ ∃ xs : List ℕ, is_valid_sequence m xs) ∧
  (∃ xs : List ℕ, is_valid_sequence 9 xs) :=
by sorry

end smallest_valid_n_l604_60458


namespace problem_solution_l604_60490

theorem problem_solution (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 2)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 15)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 130)
  (eq4 : 16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 550) :
  25*x₁ + 36*x₂ + 49*x₃ + 64*x₄ + 81*x₅ + 100*x₆ + 121*x₇ = 1492 := by
  sorry

end problem_solution_l604_60490


namespace two_players_goals_l604_60498

theorem two_players_goals (total_goals : ℕ) (player1_goals player2_goals : ℕ) : 
  total_goals = 300 →
  player1_goals = player2_goals →
  player1_goals + player2_goals = total_goals / 5 →
  player1_goals = 30 ∧ player2_goals = 30 := by
  sorry

end two_players_goals_l604_60498


namespace displacement_increment_formula_l604_60466

/-- The equation of motion for an object -/
def equation_of_motion (t : ℝ) : ℝ := 2 * t^2

/-- The increment of displacement -/
def displacement_increment (d : ℝ) : ℝ :=
  equation_of_motion (2 + d) - equation_of_motion 2

theorem displacement_increment_formula (d : ℝ) :
  displacement_increment d = 8 * d + 2 * d^2 := by
  sorry

end displacement_increment_formula_l604_60466


namespace exponential_properties_l604_60418

noncomputable def f (x : ℝ) : ℝ := 2^x

theorem exponential_properties (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  (f (x₁ + x₂) = f x₁ * f x₂) ∧ (f (-x₁) = 1 / f x₁) :=
by sorry

end exponential_properties_l604_60418


namespace circle_line_distance_l604_60473

theorem circle_line_distance (M : ℝ × ℝ) :
  (M.1 - 5)^2 + (M.2 - 3)^2 = 9 →
  (∃ d : ℝ, d = |3 * M.1 + 4 * M.2 - 2| / (3^2 + 4^2).sqrt ∧ d = 2) :=
sorry

end circle_line_distance_l604_60473


namespace inverse_function_theorem_l604_60470

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem inverse_function_theorem (p q r s : ℝ) :
  p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 →
  (∀ x, g (g p q r s x) p q r s = x) →
  p + s = 2 * q →
  p + s = 0 := by sorry

end inverse_function_theorem_l604_60470


namespace greatest_three_digit_multiple_of_27_l604_60404

theorem greatest_three_digit_multiple_of_27 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 27 ∣ n → n ≤ 999 ∧ 27 ∣ 999 := by
  sorry

end greatest_three_digit_multiple_of_27_l604_60404


namespace geometric_sequence_sum_l604_60419

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a 2 →
  a 1 + a 3 = 5 →
  a 3 + a 5 = 20 := by
  sorry

end geometric_sequence_sum_l604_60419


namespace function_nonnegative_m_range_l604_60430

theorem function_nonnegative_m_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 1 ≥ 0) → -2 ≤ m ∧ m ≤ 2 := by
  sorry

end function_nonnegative_m_range_l604_60430


namespace equation_system_solution_nature_l604_60484

/-- Given a system of equations:
    x - y + z - w = 2
    x^2 - y^2 + z^2 - w^2 = 6
    x^3 - y^3 + z^3 - w^3 = 20
    x^4 - y^4 + z^4 - w^4 = 66
    Prove that this system either has no solutions or infinitely many solutions. -/
theorem equation_system_solution_nature :
  let s₁ : ℝ := 2
  let s₂ : ℝ := 6
  let s₃ : ℝ := 20
  let s₄ : ℝ := 66
  let b₁ : ℝ := s₁
  let b₂ : ℝ := (s₁^2 - s₂) / 2
  let b₃ : ℝ := (s₁^3 - 3*s₁*s₂ + 2*s₃) / 6
  let b₄ : ℝ := (s₁^4 - 6*s₁^2*s₂ + 3*s₂^2 + 8*s₁*s₃ - 6*s₄) / 24
  b₂^2 - b₁*b₃ = 0 →
  (∀ x y z w : ℝ, 
    x - y + z - w = s₁ ∧
    x^2 - y^2 + z^2 - w^2 = s₂ ∧
    x^3 - y^3 + z^3 - w^3 = s₃ ∧
    x^4 - y^4 + z^4 - w^4 = s₄ →
    (∀ ε > 0, ∃ x' y' z' w' : ℝ,
      x' - y' + z' - w' = s₁ ∧
      x'^2 - y'^2 + z'^2 - w'^2 = s₂ ∧
      x'^3 - y'^3 + z'^3 - w'^3 = s₃ ∧
      x'^4 - y'^4 + z'^4 - w'^4 = s₄ ∧
      ((x' - x)^2 + (y' - y)^2 + (z' - z)^2 + (w' - w)^2 < ε^2) ∧
      (x' ≠ x ∨ y' ≠ y ∨ z' ≠ z ∨ w' ≠ w))) ∨
  (¬∃ x y z w : ℝ,
    x - y + z - w = s₁ ∧
    x^2 - y^2 + z^2 - w^2 = s₂ ∧
    x^3 - y^3 + z^3 - w^3 = s₃ ∧
    x^4 - y^4 + z^4 - w^4 = s₄) :=
by
  sorry

end equation_system_solution_nature_l604_60484


namespace equation_solution_unique_solution_l604_60449

theorem equation_solution : ∃ (x : ℝ), x = 2 ∧ -2 * x + 4 = 0 := by
  sorry

-- Definitions of the given equations
def eq1 (x : ℝ) : Prop := 3 * x + 6 = 0
def eq2 (x : ℝ) : Prop := -2 * x + 4 = 0
def eq3 (x : ℝ) : Prop := (1 / 2) * x = 2
def eq4 (x : ℝ) : Prop := 2 * x + 4 = 0

-- Theorem stating that eq2 is the only equation satisfied by x = 2
theorem unique_solution :
  ∃! (i : Fin 4), (match i with
    | 0 => eq1
    | 1 => eq2
    | 2 => eq3
    | 3 => eq4) 2 := by
  sorry

end equation_solution_unique_solution_l604_60449


namespace equal_population_time_l604_60459

/-- The number of years it takes for two villages' populations to be equal -/
def yearsToEqualPopulation (initialX initialY decreaseRateX increaseRateY : ℕ) : ℕ :=
  (initialX - initialY) / (decreaseRateX + increaseRateY)

theorem equal_population_time :
  yearsToEqualPopulation 70000 42000 1200 800 = 14 := by
  sorry

end equal_population_time_l604_60459


namespace power_equation_solution_l604_60460

theorem power_equation_solution : ∃ x : ℝ, (1/8 : ℝ) * 2^36 = 4^x ∧ x = 16.5 := by
  sorry

end power_equation_solution_l604_60460


namespace square_sum_geq_product_sum_l604_60489

theorem square_sum_geq_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + a*c + b*c := by
  sorry

end square_sum_geq_product_sum_l604_60489


namespace marks_lost_is_one_l604_60401

/-- Represents an examination with given parameters -/
structure Examination where
  totalQuestions : Nat
  correctAnswers : Nat
  marksPerCorrect : Nat
  totalScore : Int

/-- Calculates the marks lost per wrong answer -/
def marksLostPerWrongAnswer (exam : Examination) : Rat :=
  let wrongAnswers := exam.totalQuestions - exam.correctAnswers
  let totalCorrectMarks := exam.correctAnswers * exam.marksPerCorrect
  let totalLostMarks := totalCorrectMarks - exam.totalScore
  totalLostMarks / wrongAnswers

/-- Theorem stating that for the given examination parameters, 
    the marks lost per wrong answer is 1 -/
theorem marks_lost_is_one : 
  let exam : Examination := {
    totalQuestions := 80,
    correctAnswers := 42,
    marksPerCorrect := 4,
    totalScore := 130
  }
  marksLostPerWrongAnswer exam = 1 := by
  sorry

end marks_lost_is_one_l604_60401


namespace rectangular_field_width_l604_60476

theorem rectangular_field_width (width length : ℝ) (h1 : length = (7/5) * width) (h2 : 2 * (length + width) = 432) : width = 90 :=
by sorry

end rectangular_field_width_l604_60476


namespace kamal_english_marks_l604_60475

/-- Represents the marks of a student in various subjects -/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average of marks -/
def average (m : Marks) : ℚ :=
  (m.english + m.mathematics + m.physics + m.chemistry + m.biology) / 5

theorem kamal_english_marks :
  ∃ (m : Marks),
    m.mathematics = 60 ∧
    m.physics = 82 ∧
    m.chemistry = 67 ∧
    m.biology = 85 ∧
    average m = 74 ∧
    m.english = 76 := by
  sorry

end kamal_english_marks_l604_60475


namespace calories_in_box_is_1600_l604_60425

/-- Represents the number of cookies in a bag -/
def cookies_per_bag : ℕ := 20

/-- Represents the number of bags in a box -/
def bags_per_box : ℕ := 4

/-- Represents the number of calories in a cookie -/
def calories_per_cookie : ℕ := 20

/-- Calculates the total number of calories in a box of cookies -/
def total_calories_in_box : ℕ := cookies_per_bag * bags_per_box * calories_per_cookie

/-- Theorem stating that the total calories in a box of cookies is 1600 -/
theorem calories_in_box_is_1600 : total_calories_in_box = 1600 := by
  sorry

end calories_in_box_is_1600_l604_60425


namespace triangle_side_lengths_l604_60468

theorem triangle_side_lengths (a b c : ℝ) (C : ℝ) (area : ℝ) :
  a = 3 →
  C = 2 * Real.pi / 3 →
  area = 3 * Real.sqrt 3 / 4 →
  area = 1 / 2 * a * b * Real.sin C →
  c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C →
  b = 1 ∧ c = Real.sqrt 13 := by
  sorry

end triangle_side_lengths_l604_60468


namespace fraction_simplification_l604_60408

theorem fraction_simplification :
  (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := by
  sorry

end fraction_simplification_l604_60408


namespace function_inequality_implies_positive_a_l604_60446

open Real

theorem function_inequality_implies_positive_a (a : ℝ) :
  (∃ x₀ ∈ Set.Icc 1 (Real.exp 1), a * (x₀ - 1 / x₀) - 2 * log x₀ > -a / x₀) →
  a > 0 := by
  sorry

end function_inequality_implies_positive_a_l604_60446


namespace jack_jill_water_fetching_l604_60435

/-- A problem about Jack and Jill fetching water --/
theorem jack_jill_water_fetching :
  -- Tank capacity
  ∀ (tank_capacity : ℕ),
  -- Bucket capacity
  ∀ (bucket_capacity : ℕ),
  -- Jack's bucket carrying capacity
  ∀ (jack_buckets : ℕ),
  -- Jill's bucket carrying capacity
  ∀ (jill_buckets : ℕ),
  -- Number of trips Jill made
  ∀ (jill_trips : ℕ),
  -- Conditions
  tank_capacity = 600 →
  bucket_capacity = 5 →
  jack_buckets = 2 →
  jill_buckets = 1 →
  jill_trips = 30 →
  -- Conclusion: Jack's trips in the time Jill makes two trips
  ∃ (jack_trips : ℕ), jack_trips = 9 :=
by
  sorry

end jack_jill_water_fetching_l604_60435


namespace inequality_proof_l604_60474

theorem inequality_proof (x y : ℝ) : 2 * (x^2 + y^2) - (x + y)^2 ≥ 0 := by
  sorry

end inequality_proof_l604_60474


namespace canteen_seat_count_l604_60423

/-- Represents the seating arrangements in the office canteen -/
structure CanteenSeating where
  round_tables : Nat
  rectangular_tables : Nat
  square_tables : Nat
  couches : Nat
  benches : Nat
  extra_chairs : Nat
  round_table_capacity : Nat
  rectangular_table_capacity : Nat
  square_table_capacity : Nat
  couch_capacity : Nat
  bench_capacity : Nat

/-- Calculates the total number of seats available in the canteen -/
def total_seats (s : CanteenSeating) : Nat :=
  s.round_tables * s.round_table_capacity +
  s.rectangular_tables * s.rectangular_table_capacity +
  s.square_tables * s.square_table_capacity +
  s.couches * s.couch_capacity +
  s.benches * s.bench_capacity +
  s.extra_chairs

/-- Theorem stating that the total number of seats in the given arrangement is 80 -/
theorem canteen_seat_count :
  let s : CanteenSeating := {
    round_tables := 3,
    rectangular_tables := 4,
    square_tables := 2,
    couches := 2,
    benches := 3,
    extra_chairs := 5,
    round_table_capacity := 6,
    rectangular_table_capacity := 7,
    square_table_capacity := 4,
    couch_capacity := 3,
    bench_capacity := 5
  }
  total_seats s = 80 := by
  sorry

end canteen_seat_count_l604_60423


namespace max_consecutive_semi_primes_l604_60436

/-- A natural number is semi-prime if it is greater than 25 and is the sum of two distinct prime numbers. -/
def IsSemiPrime (n : ℕ) : Prop :=
  n > 25 ∧ ∃ p q : ℕ, p.Prime ∧ q.Prime ∧ p ≠ q ∧ n = p + q

/-- The maximum number of consecutive semi-prime natural numbers is 5. -/
theorem max_consecutive_semi_primes :
  ∀ n : ℕ, (∀ k : ℕ, k ∈ Finset.range 6 → IsSemiPrime (n + k)) →
    ¬∀ k : ℕ, k ∈ Finset.range 7 → IsSemiPrime (n + k) :=
by sorry

end max_consecutive_semi_primes_l604_60436


namespace red_marbles_count_l604_60443

-- Define the number of marbles of each color
def blue_marbles : ℕ := 10
def yellow_marbles : ℕ := 6

-- Define the probability of selecting a blue marble from either bag
def prob_blue : ℚ := 3/4

-- Define the function to calculate the probability of selecting a blue marble
def prob_select_blue (red_marbles : ℕ) : ℚ :=
  1 - (1 - blue_marbles / (red_marbles + blue_marbles + yellow_marbles : ℚ))^2

-- Theorem statement
theorem red_marbles_count :
  ∃ (red_marbles : ℕ), prob_select_blue red_marbles = prob_blue :=
sorry

end red_marbles_count_l604_60443


namespace john_needs_more_money_l604_60414

theorem john_needs_more_money (total_needed : ℝ) (amount_has : ℝ) (h1 : total_needed = 2.50) (h2 : amount_has = 0.75) :
  total_needed - amount_has = 1.75 := by
  sorry

end john_needs_more_money_l604_60414


namespace hardware_contract_probability_l604_60454

theorem hardware_contract_probability 
  (p_not_software : ℝ) 
  (p_at_least_one : ℝ) 
  (p_both : ℝ) 
  (h1 : p_not_software = 3/5) 
  (h2 : p_at_least_one = 9/10) 
  (h3 : p_both = 0.3) : 
  ∃ p_hardware : ℝ, p_hardware = 0.8 := by
  sorry

end hardware_contract_probability_l604_60454


namespace derivative_implies_limit_l604_60445

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define x₀ as a real number
variable (x₀ : ℝ)

-- State the theorem
theorem derivative_implies_limit 
  (h₁ : HasDerivAt f (-2) x₀) :
  ∀ ε > 0, ∃ δ > 0, ∀ h ≠ 0, |h| < δ → 
    |((f (x₀ - 1/2 * h) - f x₀) / h) - 1| < ε :=
sorry

end derivative_implies_limit_l604_60445


namespace incenter_representation_l604_60409

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle with vertices P, Q, R and side lengths p, q, r -/
structure Triangle where
  P : Point2D
  Q : Point2D
  R : Point2D
  p : ℝ
  q : ℝ
  r : ℝ

/-- The incenter of a triangle -/
def incenter (t : Triangle) : Point2D := sorry

/-- Theorem: The incenter of the specific triangle can be represented as a linear combination
    of its vertices with coefficients (1/3, 1/4, 5/12) -/
theorem incenter_representation (t : Triangle) 
  (h1 : t.p = 8) (h2 : t.q = 6) (h3 : t.r = 10) : 
  ∃ (J : Point2D), J = incenter t ∧ 
    J.x = (1/3) * t.P.x + (1/4) * t.Q.x + (5/12) * t.R.x ∧
    J.y = (1/3) * t.P.y + (1/4) * t.Q.y + (5/12) * t.R.y :=
sorry

end incenter_representation_l604_60409


namespace center_of_mass_distance_three_points_l604_60487

/-- Given three material points with masses and distances from a line,
    prove the formula for the distance of their center of mass from the line. -/
theorem center_of_mass_distance_three_points
  (m₁ m₂ m₃ y₁ y₂ y₃ : ℝ)
  (hm : m₁ > 0 ∧ m₂ > 0 ∧ m₃ > 0) :
  let z := (m₁ * y₁ + m₂ * y₂ + m₃ * y₃) / (m₁ + m₂ + m₃)
  ∃ (com : ℝ), com = z ∧ 
    com * (m₁ + m₂ + m₃) = m₁ * y₁ + m₂ * y₂ + m₃ * y₃ :=
by sorry

end center_of_mass_distance_three_points_l604_60487


namespace slope_angle_of_line_l604_60426

theorem slope_angle_of_line (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ θ : ℝ, θ ∈ Set.Icc 0 π ∧ θ = π - Real.arctan (a / b) := by
  sorry

end slope_angle_of_line_l604_60426


namespace quadratic_one_root_l604_60450

theorem quadratic_one_root (m : ℝ) : 
  (∃! x : ℝ, x^2 + 6*m*x + 2*m = 0) ↔ m = 2/9 := by sorry

end quadratic_one_root_l604_60450


namespace quadratic_root_l604_60447

theorem quadratic_root (m : ℝ) : 
  (2 : ℝ)^2 + m * 2 - 6 = 0 → (-3 : ℝ)^2 + m * (-3) - 6 = 0 := by
  sorry

end quadratic_root_l604_60447


namespace product_digits_sum_base9_l604_60453

/-- Converts a base 9 number to decimal --/
def base9ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 9 --/
def decimalToBase9 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a base 9 number --/
def sumOfDigitsBase9 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem product_digits_sum_base9 :
  let a := 36
  let b := 21
  let product := (base9ToDecimal a) * (base9ToDecimal b)
  sumOfDigitsBase9 (decimalToBase9 product) = 19 := by sorry

end product_digits_sum_base9_l604_60453


namespace proportional_relationship_l604_60429

/-- Given that y-2 is directly proportional to x-3, and when x=4, y=8,
    prove the functional relationship and a specific point. -/
theorem proportional_relationship (k : ℝ) :
  (∀ x y : ℝ, y - 2 = k * (x - 3)) →  -- Condition 1
  (8 - 2 = k * (4 - 3)) →             -- Condition 2
  (∀ x y : ℝ, y = 6 * x - 16) ∧       -- Conclusion 1
  (-6 = 6 * (5/3) - 16) :=            -- Conclusion 2
by sorry

end proportional_relationship_l604_60429


namespace complex_modulus_problem_l604_60493

theorem complex_modulus_problem (a : ℝ) (z : ℂ) : 
  z = (a + Complex.I) / (2 - Complex.I) + a ∧ z.re = 0 → Complex.abs z = 3/7 := by
  sorry

end complex_modulus_problem_l604_60493


namespace min_value_theorem_l604_60402

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 64) :
  x^2 + 8*x*y + 16*y^2 + 4*z^2 ≥ 192 ∧
  (x^2 + 8*x*y + 16*y^2 + 4*z^2 = 192 ↔ x = 8 ∧ y = 2 ∧ z = 8) :=
by sorry

end min_value_theorem_l604_60402


namespace largest_gcd_of_sum_221_l604_60442

theorem largest_gcd_of_sum_221 :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a + b = 221 ∧
  (∀ (c d : ℕ), c > 0 → d > 0 → c + d = 221 → Nat.gcd c d ≤ Nat.gcd a b) ∧
  Nat.gcd a b = 17 :=
by sorry

end largest_gcd_of_sum_221_l604_60442


namespace arrangements_with_restriction_l604_60480

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where two specific people are together -/
def arrangementsWithTwoTogether (n : ℕ) : ℕ := Nat.factorial (n - 1) * Nat.factorial 2

/-- The number of ways to arrange n people in a row where three specific people are together -/
def arrangementsWithThreeTogether (n : ℕ) : ℕ := Nat.factorial (n - 2) * Nat.factorial 3

/-- The number of ways to arrange 9 people in a row where three specific people cannot sit next to each other -/
theorem arrangements_with_restriction : 
  totalArrangements 9 - (3 * arrangementsWithTwoTogether 9 - arrangementsWithThreeTogether 9) = 181200 := by
  sorry

end arrangements_with_restriction_l604_60480


namespace final_milk_amount_l604_60432

-- Define the initial amount of milk
def initial_milk : ℚ := 5

-- Define the amount given away
def given_away : ℚ := 18/4

-- Define the amount received back
def received_back : ℚ := 7/4

-- Theorem statement
theorem final_milk_amount :
  initial_milk - given_away + received_back = 9/4 :=
by sorry

end final_milk_amount_l604_60432


namespace probability_smaller_triangle_l604_60412

/-- The probability that a randomly chosen point in a right triangle
    forms a smaller triangle with area less than one-third of the original -/
theorem probability_smaller_triangle (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let triangle_area := a * b / 2
  let probability := (a * (b / 3)) / (2 * triangle_area)
  probability = 1 / 3 := by
  sorry

end probability_smaller_triangle_l604_60412


namespace triangle_properties_l604_60415

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.a^2 - t.c^2 - 1/2 * t.b * t.c = t.a * t.b * Real.cos t.C ∧
  t.a = 2 * Real.sqrt 3

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.A = 2 * Real.pi / 3 ∧
  4 * Real.sqrt 3 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 4 + 2 * Real.sqrt 3 :=
by sorry

end triangle_properties_l604_60415


namespace coin_count_l604_60469

/-- The total value of coins in cents -/
def total_value : ℕ := 240

/-- The number of nickels -/
def num_nickels : ℕ := 12

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The total number of coins -/
def total_coins : ℕ := num_nickels + (total_value - num_nickels * nickel_value) / dime_value

theorem coin_count : total_coins = 30 := by
  sorry

end coin_count_l604_60469


namespace inequality_proof_l604_60479

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  b^2 / a + a^2 / b ≥ a + b :=
by sorry

end inequality_proof_l604_60479


namespace arithmetic_sequence_part_1_arithmetic_sequence_part_2_l604_60467

/-- An arithmetic sequence with its sum of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- Sum of first n terms

/-- Theorem for part I -/
theorem arithmetic_sequence_part_1 (seq : ArithmeticSequence) 
  (h1 : seq.a 1 = 1) (h2 : seq.S 10 = 100) :
  ∀ n : ℕ, seq.a n = 2 * n - 1 := by sorry

/-- Theorem for part II -/
theorem arithmetic_sequence_part_2 (seq : ArithmeticSequence) 
  (h : ∀ n : ℕ, seq.S n = n^2 - 6*n) :
  ∀ n : ℕ, (seq.S n + seq.a n > 2*n) ↔ (n > 7) := by sorry

end arithmetic_sequence_part_1_arithmetic_sequence_part_2_l604_60467


namespace inequality_solution_l604_60405

theorem inequality_solution (m : ℝ) : 
  (∀ x : ℝ, (x + m) / 2 - 1 > 2 * m ↔ x > 5) → m = 1 := by
  sorry

end inequality_solution_l604_60405
