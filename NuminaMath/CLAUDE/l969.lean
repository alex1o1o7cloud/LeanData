import Mathlib

namespace no_solution_for_equation_l969_96925

theorem no_solution_for_equation : ¬∃ (x : ℝ), (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) := by
  sorry

end no_solution_for_equation_l969_96925


namespace price_increase_2008_2009_l969_96920

/-- Given a 60% increase from 2006 to 2008 and a 20% annual average growth rate
    from 2006 to 2009, the increase from 2008 to 2009 is 8%. -/
theorem price_increase_2008_2009 
  (price_2006 : ℝ) 
  (price_2008 : ℝ) 
  (price_2009 : ℝ) 
  (h1 : price_2008 = price_2006 * (1 + 0.60))
  (h2 : price_2009 = price_2006 * (1 + 0.20)^3) :
  price_2009 = price_2008 * (1 + 0.08) :=
by sorry

end price_increase_2008_2009_l969_96920


namespace problem_1_problem_2_problem_3_problem_4_l969_96996

-- Problem 1
theorem problem_1 : 211 * (-455) + 365 * 455 - 211 * 545 + 545 * 365 = 154000 := by
  sorry

-- Problem 2
theorem problem_2 : (-7/5 * (-5/2) - 1) / 9 / (1/(-3/4)^2) - |2 + (-1/2)^3 * 5^2| = -31/32 := by
  sorry

-- Problem 3
theorem problem_3 (x : ℝ) : (3*x + 2)*(x + 1) + 2*(x - 3)*(x + 2) = 5*x^2 + 3*x - 10 := by
  sorry

-- Problem 4
theorem problem_4 : ∃ (x : ℚ), (2*x + 3)/6 - (2*x - 1)/4 = 1 ∧ x = -3/2 := by
  sorry

end problem_1_problem_2_problem_3_problem_4_l969_96996


namespace curve_translation_l969_96987

-- Define the original curve
def original_curve (x y : ℝ) : Prop :=
  y * Real.cos x + 2 * y - 1 = 0

-- Define the translated curve
def translated_curve (x y : ℝ) : Prop :=
  (y - 1) * Real.sin x + 2 * y - 3 = 0

-- State the theorem
theorem curve_translation :
  ∀ (x y : ℝ),
    original_curve (x - π/2) (y + 1) ↔ translated_curve x y :=
by sorry

end curve_translation_l969_96987


namespace sum_of_extremal_x_values_l969_96934

theorem sum_of_extremal_x_values (x y z : ℝ) 
  (sum_condition : x + y + z = 5)
  (square_sum_condition : x^2 + y^2 + z^2 = 11) :
  ∃ (m M : ℝ), 
    (∀ x', (∃ y' z', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 11) → m ≤ x' ∧ x' ≤ M) ∧
    m + M = 10/3 := by
  sorry

end sum_of_extremal_x_values_l969_96934


namespace red_ball_certain_l969_96943

/-- Represents the number of balls of each color in the box -/
structure BallCount where
  red : Nat
  yellow : Nat

/-- Represents the number of balls drawn from the box -/
def BallsDrawn : Nat := 3

/-- The initial state of the box -/
def initialBox : BallCount where
  red := 3
  yellow := 2

/-- A function to check if drawing at least one red ball is certain -/
def isRedBallCertain (box : BallCount) : Prop :=
  box.yellow < BallsDrawn

/-- Theorem stating that drawing at least one red ball is certain -/
theorem red_ball_certain :
  isRedBallCertain initialBox := by
  sorry

end red_ball_certain_l969_96943


namespace sum_of_roots_is_negative_one_l969_96944

-- Define the ∇ operation
def nabla (a b : ℝ) : ℝ := a * b - b * a^2

-- Theorem statement
theorem sum_of_roots_is_negative_one :
  let f : ℝ → ℝ := λ x => (nabla 2 x) - 8 - (nabla x 6)
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0) ∧ x₁ + x₂ = -1 :=
sorry

end sum_of_roots_is_negative_one_l969_96944


namespace inequality_solution_sets_l969_96918

-- Define the solution set of the first inequality
def solution_set_1 : Set ℝ := {x : ℝ | 2 < x ∧ x < 3}

-- Define the coefficients a and b
def a : ℝ := 5
def b : ℝ := -6

-- Define the solution set of the second inequality
def solution_set_2 : Set ℝ := {x : ℝ | -1/2 < x ∧ x < -1/3}

theorem inequality_solution_sets :
  (∀ x : ℝ, x ∈ solution_set_1 ↔ x^2 - a*x - b < 0) →
  (∀ x : ℝ, x ∈ solution_set_2 ↔ b*x^2 - a*x - 1 > 0) :=
by sorry

end inequality_solution_sets_l969_96918


namespace parabola_max_value_l969_96919

theorem parabola_max_value :
  let f : ℝ → ℝ := fun x ↦ -2 * x^2 + 4 * x + 3
  ∃ (max : ℝ), ∀ (x : ℝ), f x ≤ max ∧ ∃ (x_max : ℝ), f x_max = max ∧ max = 5 := by
  sorry

end parabola_max_value_l969_96919


namespace tangent_parallel_to_x_axis_l969_96957

/-- The curve function f(x) = x^2 + 2x - 2 --/
def f (x : ℝ) : ℝ := x^2 + 2*x - 2

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 2*x + 2

theorem tangent_parallel_to_x_axis :
  ∃ (x y : ℝ), f x = y ∧ f' x = 0 ∧ x = -1 ∧ y = -3 :=
by sorry

end tangent_parallel_to_x_axis_l969_96957


namespace rectangle_to_square_cut_l969_96976

theorem rectangle_to_square_cut (rectangle_length : ℝ) (rectangle_width : ℝ) (num_parts : ℕ) :
  rectangle_length = 2 ∧ rectangle_width = 1 ∧ num_parts = 3 →
  ∃ (square_side : ℝ), square_side = Real.sqrt 2 ∧
    rectangle_length * rectangle_width = square_side * square_side :=
by sorry

end rectangle_to_square_cut_l969_96976


namespace least_k_factorial_multiple_of_315_l969_96946

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem least_k_factorial_multiple_of_315 (k : ℕ) (h1 : k > 1) (h2 : 315 ∣ factorial k) :
  k ≥ 7 ∧ 315 ∣ factorial 7 := by
  sorry

end least_k_factorial_multiple_of_315_l969_96946


namespace quadratic_roots_theorem_l969_96936

/-- Quadratic equation with parameter k -/
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 + (2*k + 1)*x + k^2 + 1

/-- Discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ := (2*k + 1)^2 - 4*(k^2 + 1)

/-- Theorem stating the conditions for distinct real roots and the value of k -/
theorem quadratic_roots_theorem (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0) ↔ k > 3/4 ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0 ∧
   |x₁| + |x₂| = x₁ * x₂ → k = 2) :=
by sorry

end quadratic_roots_theorem_l969_96936


namespace sequence_inequality_l969_96979

def a : ℕ → ℚ
  | 0 => 1
  | (n + 1) => a n - (a n)^2 / 2019

theorem sequence_inequality : a 2019 < 1/2 ∧ 1/2 < a 2018 := by
  sorry

end sequence_inequality_l969_96979


namespace expected_heads_is_40_l969_96942

/-- A coin toss simulation with specific rules --/
def CoinTossSimulation :=
  { n : ℕ  // n = 80 }

/-- The probability of a coin showing heads after all tosses --/
def prob_heads (c : CoinTossSimulation) : ℚ :=
  1 / 2

/-- The expected number of heads in the simulation --/
def expected_heads (c : CoinTossSimulation) : ℚ :=
  c.val * prob_heads c

/-- Theorem stating that the expected number of heads is 40 --/
theorem expected_heads_is_40 (c : CoinTossSimulation) :
  expected_heads c = 40 := by
  sorry

#check expected_heads_is_40

end expected_heads_is_40_l969_96942


namespace complex_equation_roots_l969_96990

theorem complex_equation_roots : 
  let z₁ : ℂ := -1 + Real.sqrt 5 - (2 * Real.sqrt 5 / 5) * I
  let z₂ : ℂ := -1 - Real.sqrt 5 + (2 * Real.sqrt 5 / 5) * I
  (z₁^2 + 2*z₁ = 3 - 4*I) ∧ (z₂^2 + 2*z₂ = 3 - 4*I) := by
  sorry

end complex_equation_roots_l969_96990


namespace nonagon_diagonal_intersection_probability_l969_96965

/-- The number of sides in a regular nonagon -/
def n : ℕ := 9

/-- The total number of line segments (sides and diagonals) in a regular nonagon -/
def total_segments : ℕ := n.choose 2

/-- The number of diagonals in a regular nonagon -/
def num_diagonals : ℕ := total_segments - n

/-- The number of ways to choose two diagonals -/
def ways_to_choose_diagonals : ℕ := num_diagonals.choose 2

/-- The number of ways to choose four points that form intersecting diagonals -/
def intersecting_diagonals : ℕ := n.choose 4

/-- The probability that two randomly chosen diagonals intersect inside the nonagon -/
def probability_intersect : ℚ := intersecting_diagonals / ways_to_choose_diagonals

theorem nonagon_diagonal_intersection_probability :
  probability_intersect = 6 / 13 := by sorry

end nonagon_diagonal_intersection_probability_l969_96965


namespace quadratic_inequality_equivalence_l969_96923

theorem quadratic_inequality_equivalence (a b c : ℝ) :
  (a > 0 ∧ b^2 - 4*a*c < 0) ↔ (∀ x : ℝ, a*x^2 + b*x + c > 0) :=
by sorry

end quadratic_inequality_equivalence_l969_96923


namespace total_area_three_shapes_l969_96909

theorem total_area_three_shapes 
  (rect_area square_area tri_area : ℝ)
  (rect_square_overlap rect_tri_overlap square_tri_overlap : ℝ)
  (all_overlap : ℝ) :
  let total_area := rect_area + square_area + tri_area - 
                    rect_square_overlap - rect_tri_overlap - square_tri_overlap + 
                    all_overlap
  total_area = 66 :=
by sorry

end total_area_three_shapes_l969_96909


namespace solutions_count_l969_96911

/-- The number of solutions to the equation √(x+3) = ax + 2 depends on the value of parameter a -/
theorem solutions_count (a : ℝ) : 
  (∀ x, Real.sqrt (x + 3) ≠ a * x + 2) ∨ 
  (∃! x, Real.sqrt (x + 3) = a * x + 2) ∨
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ Real.sqrt (x₁ + 3) = a * x₁ + 2 ∧ Real.sqrt (x₂ + 3) = a * x₂ + 2 ∧ 
    ∀ x, Real.sqrt (x + 3) = a * x + 2 → x = x₁ ∨ x = x₂) ∨
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    Real.sqrt (x₁ + 3) = a * x₁ + 2 ∧ Real.sqrt (x₂ + 3) = a * x₂ + 2 ∧ Real.sqrt (x₃ + 3) = a * x₃ + 2 ∧
    ∀ x, Real.sqrt (x + 3) = a * x + 2 → x = x₁ ∨ x = x₂ ∨ x = x₃) ∨
  (∃ x₁ x₂ x₃ x₄, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    Real.sqrt (x₁ + 3) = a * x₁ + 2 ∧ Real.sqrt (x₂ + 3) = a * x₂ + 2 ∧ 
    Real.sqrt (x₃ + 3) = a * x₃ + 2 ∧ Real.sqrt (x₄ + 3) = a * x₄ + 2 ∧
    ∀ x, Real.sqrt (x + 3) = a * x + 2 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) :=
by
  sorry

end solutions_count_l969_96911


namespace classroom_pencils_l969_96932

theorem classroom_pencils (num_children : ℕ) (pencils_per_child : ℕ) 
  (h1 : num_children = 10) (h2 : pencils_per_child = 5) : 
  num_children * pencils_per_child = 50 := by
  sorry

end classroom_pencils_l969_96932


namespace volunteer_allocation_schemes_l969_96937

def num_volunteers : ℕ := 5
def num_projects : ℕ := 4

theorem volunteer_allocation_schemes :
  (num_volunteers.choose 2) * (num_projects!) = 240 :=
by sorry

end volunteer_allocation_schemes_l969_96937


namespace r_value_when_n_is_3_l969_96967

theorem r_value_when_n_is_3 (n : ℕ) (s : ℕ) (r : ℕ) 
  (h1 : s = 2^n - 1) 
  (h2 : r = 3^s - s) 
  (h3 : n = 3) : 
  r = 2180 := by
  sorry

end r_value_when_n_is_3_l969_96967


namespace sqrt_x_minus_four_defined_l969_96928

theorem sqrt_x_minus_four_defined (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 4) ↔ x ≥ 4 := by sorry

end sqrt_x_minus_four_defined_l969_96928


namespace three_digit_permutation_property_l969_96933

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_permutations (n : ℕ) : List ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  [100*a + 10*b + c, 100*a + 10*c + b, 100*b + 10*a + c, 100*b + 10*c + a, 100*c + 10*a + b, 100*c + 10*b + a]

def satisfies_property (n : ℕ) : Prop :=
  is_three_digit n ∧ (List.sum (digit_permutations n)) / 6 = n

def solution_set : List ℕ := [111, 222, 333, 444, 555, 666, 777, 888, 999, 407, 518, 629, 370, 481, 592]

theorem three_digit_permutation_property :
  ∀ n : ℕ, satisfies_property n ↔ n ∈ solution_set := by sorry

end three_digit_permutation_property_l969_96933


namespace derivative_f_at_pi_l969_96929

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem derivative_f_at_pi : 
  deriv f π = -π := by sorry

end derivative_f_at_pi_l969_96929


namespace election_total_votes_l969_96992

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

end election_total_votes_l969_96992


namespace berry_difference_change_l969_96939

/-- Represents the number of berries in a box -/
structure Berry where
  count : ℕ

/-- Represents a box of berries -/
inductive Box
  | Red : Berry → Box
  | Blue : Berry → Box

/-- The problem setup -/
structure BerryProblem where
  blue_berry_count : ℕ
  red_berry_count : ℕ
  berry_increase : ℕ
  blue_box_count : ℕ
  red_box_count : ℕ

/-- The theorem to prove -/
theorem berry_difference_change (problem : BerryProblem) 
  (h1 : problem.blue_berry_count = 36)
  (h2 : problem.red_berry_count = problem.blue_berry_count + problem.berry_increase)
  (h3 : problem.berry_increase = 15) :
  problem.red_berry_count - problem.blue_berry_count = 15 := by
  sorry

#check berry_difference_change

end berry_difference_change_l969_96939


namespace complex_sum_simplification_l969_96949

theorem complex_sum_simplification : 
  let z₁ : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  let z₂ : ℂ := (-1 - Complex.I * Real.sqrt 3) / 2
  z₁^12 + z₂^12 = 2 := by sorry

end complex_sum_simplification_l969_96949


namespace sixth_game_score_l969_96993

theorem sixth_game_score (scores : List ℕ) (mean : ℚ) : 
  scores.length = 7 ∧
  scores = [69, 68, 70, 61, 74, 65, 74] ∧
  mean = 67.9 ∧
  (∃ x : ℕ, (scores.sum + x) / 8 = mean) →
  ∃ x : ℕ, x = 62 ∧ (scores.sum + x) / 8 = mean := by
sorry

end sixth_game_score_l969_96993


namespace odds_against_C_winning_l969_96917

-- Define the type for horses
inductive Horse : Type
| A
| B
| C

-- Define the function for odds against winning
def oddsAgainst (h : Horse) : ℚ :=
  match h with
  | Horse.A => 4 / 1
  | Horse.B => 3 / 2
  | Horse.C => 3 / 2  -- This is what we want to prove

-- State the theorem
theorem odds_against_C_winning :
  (∀ h₁ h₂ : Horse, h₁ ≠ h₂ → oddsAgainst h₁ ≠ oddsAgainst h₂) →  -- No ties
  oddsAgainst Horse.A = 4 / 1 →
  oddsAgainst Horse.B = 3 / 2 →
  oddsAgainst Horse.C = 3 / 2 :=
by
  sorry


end odds_against_C_winning_l969_96917


namespace m_range_l969_96916

/-- The range of m given the specified conditions -/
theorem m_range (m : ℝ) : 
  (¬ ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∧ 
  ((∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∨ 
   (∀ x : ℝ, x ≥ 2 → x + m/x - 2 > 0)) →
  0 < m ∧ m ≤ 2 := by
sorry

end m_range_l969_96916


namespace average_of_combined_results_l969_96991

theorem average_of_combined_results :
  let n₁ : ℕ := 40
  let avg₁ : ℚ := 30
  let n₂ : ℕ := 30
  let avg₂ : ℚ := 40
  let total_sum := n₁ * avg₁ + n₂ * avg₂
  let total_count := n₁ + n₂
  (total_sum / total_count : ℚ) = 2400 / 70 := by
sorry

end average_of_combined_results_l969_96991


namespace hypothetical_town_population_l969_96910

theorem hypothetical_town_population : ∃ n : ℕ, 
  (∃ m k : ℕ, 
    n^2 + 150 = m^2 + 1 ∧ 
    n^2 + 300 = k^2) ∧ 
  n^2 = 5476 := by
  sorry

end hypothetical_town_population_l969_96910


namespace correct_match_probability_l969_96912

theorem correct_match_probability (n : ℕ) (h : n = 6) :
  (1 : ℚ) / (Nat.factorial n) = (1 : ℚ) / 720 :=
sorry

end correct_match_probability_l969_96912


namespace time_to_plant_trees_l969_96948

-- Define the rate of planting trees
def trees_per_minute : ℚ := 10 / 3

-- Define the total number of trees to be planted
def total_trees : ℕ := 2500

-- Define the time it takes to plant all trees in hours
def planting_time : ℚ := 12.5

-- Theorem to prove
theorem time_to_plant_trees :
  trees_per_minute * 60 * planting_time = total_trees :=
sorry

end time_to_plant_trees_l969_96948


namespace jose_investment_is_45000_l969_96931

/-- Represents the investment and profit scenario of Tom and Jose's shop --/
structure ShopInvestment where
  tom_investment : ℕ
  tom_months : ℕ
  jose_months : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Jose's investment based on the given conditions --/
def calculate_jose_investment (s : ShopInvestment) : ℕ :=
  let tom_time_investment := s.tom_investment * s.tom_months
  let tom_profit := s.total_profit - s.jose_profit
  (tom_time_investment * s.jose_profit) / (tom_profit * s.jose_months)

/-- Theorem stating that Jose's investment is 45000 given the problem conditions --/
theorem jose_investment_is_45000 :
  let s : ShopInvestment := {
    tom_investment := 30000,
    tom_months := 12,
    jose_months := 10,
    total_profit := 63000,
    jose_profit := 35000
  }
  calculate_jose_investment s = 45000 := by sorry


end jose_investment_is_45000_l969_96931


namespace right_triangle_sides_l969_96905

theorem right_triangle_sides : ∀ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a = 1 ∧ b = 2 ∧ c = Real.sqrt 3) ↔ a * a + b * b = c * c :=
by sorry

end right_triangle_sides_l969_96905


namespace two_numbers_satisfy_property_l969_96984

/-- Given a two-digit positive integer, return the integer obtained by reversing its digits -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The property that we're checking for each two-digit number -/
def has_property (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ is_perfect_square (n + (reverse_digits n)^3)

/-- The main theorem stating that exactly two numbers satisfy the property -/
theorem two_numbers_satisfy_property :
  ∃! (s : Finset ℕ), s.card = 2 ∧ ∀ n, n ∈ s ↔ has_property n :=
sorry

end two_numbers_satisfy_property_l969_96984


namespace unique_solution_cube_equation_l969_96915

theorem unique_solution_cube_equation (y : ℝ) (hy : y ≠ 0) :
  (3 * y)^5 = (9 * y)^4 ↔ y = 27 := by
sorry

end unique_solution_cube_equation_l969_96915


namespace hyperbola_eccentricity_l969_96900

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptote 3x + y = 0 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b = 3 * a) : Real.sqrt 10 = 
  Real.sqrt ((a^2 + b^2) / a^2) := by sorry

end hyperbola_eccentricity_l969_96900


namespace total_amount_calculation_l969_96971

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

end total_amount_calculation_l969_96971


namespace quadratic_equation_solution_l969_96921

/-- The equation 7x^2 + 13x + d = 0 has rational solutions for d -/
def has_rational_solution (d : ℕ+) : Prop :=
  ∃ x : ℚ, 7 * x^2 + 13 * x + d.val = 0

/-- The set of positive integers d for which the equation has rational solutions -/
def solution_set : Set ℕ+ :=
  {d | has_rational_solution d}

theorem quadratic_equation_solution :
  ∃ (d₁ d₂ : ℕ+), d₁ ≠ d₂ ∧ 
    solution_set = {d₁, d₂} ∧
    d₁.val * d₂.val = 2 :=
sorry

end quadratic_equation_solution_l969_96921


namespace diophantine_equation_solution_l969_96952

theorem diophantine_equation_solution : ∃ (a b c d : ℕ+), 
  (a^3 + b^4 + c^5 = d^11) ∧ (a * b * c < 10^5) := by
  sorry

end diophantine_equation_solution_l969_96952


namespace consecutive_integers_permutation_divisibility_l969_96926

theorem consecutive_integers_permutation_divisibility
  (p : ℕ) (h_prime : Nat.Prime p)
  (m : ℕ → ℕ) (h_consecutive : ∀ i ∈ Finset.range p, m (i + 1) = m i + 1)
  (σ : Fin p → Fin p) (h_perm : Function.Bijective σ) :
  ∃ (k l : Fin p), k ≠ l ∧ p ∣ (m k * m (σ k) - m l * m (σ l)) := by
  sorry

end consecutive_integers_permutation_divisibility_l969_96926


namespace polygon_deformation_to_triangle_l969_96999

/-- A planar polygon represented by its vertices -/
structure PlanarPolygon where
  vertices : List (ℝ × ℝ)
  is_planar : sorry
  is_closed : sorry

/-- A function that checks if a polygon can be deformed into a triangle -/
def can_deform_to_triangle (p : PlanarPolygon) : Prop :=
  sorry

/-- The main theorem stating that any planar polygon with more than 4 sides
    can be deformed into a triangle -/
theorem polygon_deformation_to_triangle 
  (p : PlanarPolygon) (h : p.vertices.length > 4) :
  can_deform_to_triangle p :=
sorry

end polygon_deformation_to_triangle_l969_96999


namespace problem_statement_l969_96982

theorem problem_statement (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = {0, a^2, a+b} → a^2015 + b^2014 = -1 := by sorry

end problem_statement_l969_96982


namespace g_of_4_l969_96956

def g (x : ℝ) : ℝ := 5 * x - 2

theorem g_of_4 : g 4 = 18 := by sorry

end g_of_4_l969_96956


namespace product_expansion_l969_96903

theorem product_expansion (x y : ℝ) :
  (3 * x^4 - 7 * y^3) * (9 * x^8 + 21 * x^4 * y^3 + 49 * y^6) = 27 * x^12 - 343 * y^9 := by
  sorry

end product_expansion_l969_96903


namespace mean_proportional_problem_l969_96983

theorem mean_proportional_problem (B : ℝ) :
  (56.5 : ℝ) = Real.sqrt (49 * B) → B = 64.9 := by
  sorry

end mean_proportional_problem_l969_96983


namespace macaroon_ratio_is_two_to_one_l969_96988

/-- Represents the numbers of macaroons in different states --/
structure MacaroonCounts where
  initial_red : ℕ
  initial_green : ℕ
  green_eaten : ℕ
  total_remaining : ℕ

/-- Calculates the ratio of red macaroons eaten to green macaroons eaten --/
def macaroon_ratio (m : MacaroonCounts) : ℚ :=
  let red_eaten := m.initial_red - (m.total_remaining - (m.initial_green - m.green_eaten))
  red_eaten / m.green_eaten

/-- Theorem stating that given the specific conditions, the ratio is 2:1 --/
theorem macaroon_ratio_is_two_to_one (m : MacaroonCounts) 
  (h1 : m.initial_red = 50)
  (h2 : m.initial_green = 40)
  (h3 : m.green_eaten = 15)
  (h4 : m.total_remaining = 45) :
  macaroon_ratio m = 2 := by
  sorry

end macaroon_ratio_is_two_to_one_l969_96988


namespace intersection_point_l969_96986

-- Define the line using a parameter t
def line (t : ℝ) : ℝ × ℝ × ℝ := (1 - 2*t, 2 + t, -1 - t)

-- Define the plane equation
def plane (p : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  x - 2*y + 5*z + 17 = 0

-- Theorem statement
theorem intersection_point :
  ∃! p : ℝ × ℝ × ℝ, (∃ t : ℝ, line t = p) ∧ plane p ∧ p = (-1, 3, -2) := by
  sorry

end intersection_point_l969_96986


namespace tree_height_difference_l969_96947

/-- The height difference between two trees -/
theorem tree_height_difference (pine_height maple_height : ℚ) 
  (h_pine : pine_height = 49/4)
  (h_maple : maple_height = 37/2) :
  maple_height - pine_height = 25/4 := by
  sorry

#eval (37/2 : ℚ) - (49/4 : ℚ)  -- Should output 25/4

end tree_height_difference_l969_96947


namespace polar_to_cartesian_parabola_l969_96981

/-- The curve defined by the polar equation r = 1 / (1 - sin θ) is a parabola -/
theorem polar_to_cartesian_parabola :
  ∃ (x y : ℝ), (∃ (r θ : ℝ), r = 1 / (1 - Real.sin θ) ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) →
  x^2 = 2*y + 1 :=
sorry

end polar_to_cartesian_parabola_l969_96981


namespace g_of_5_l969_96940

/-- The function g satisfies the given functional equation for all real x -/
axiom functional_equation (g : ℝ → ℝ) :
  ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 3 * x + 2

/-- The value of g(5) is -20.01 -/
theorem g_of_5 (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 3 * x + 2) :
  g 5 = -20.01 := by
  sorry

end g_of_5_l969_96940


namespace both_reunions_count_l969_96958

/-- The number of people attending both reunions -/
def both_reunions (total guests : ℕ) (oates hall : ℕ) : ℕ :=
  total - (oates + hall - total)

theorem both_reunions_count : both_reunions 150 70 52 = 28 := by
  sorry

end both_reunions_count_l969_96958


namespace complex_number_properties_l969_96922

theorem complex_number_properties (w : ℂ) (h : w^2 = 16 - 48*I) : 
  Complex.abs w = 4 * (10 : ℝ)^(1/4) ∧ 
  Complex.arg w = (Real.arctan (-3) / 2 + Real.pi / 2) :=
by sorry

end complex_number_properties_l969_96922


namespace octal_to_decimal_l969_96935

-- Define the octal number
def octal_number : ℕ := 724

-- Define the decimal number
def decimal_number : ℕ := 468

-- Theorem stating that the octal number 724 is equal to the decimal number 468
theorem octal_to_decimal :
  octal_number.digits 8 = [4, 2, 7] ∧ 
  decimal_number = 4 * 8^0 + 2 * 8^1 + 7 * 8^2 := by
  sorry

#check octal_to_decimal

end octal_to_decimal_l969_96935


namespace tenth_term_of_sequence_l969_96998

def inversely_proportional_sequence (a : ℕ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) * a n = k

theorem tenth_term_of_sequence (a : ℕ → ℝ) :
  inversely_proportional_sequence a →
  a 1 = 3 →
  a 2 = 4 →
  a 10 = 4 :=
by
  sorry

end tenth_term_of_sequence_l969_96998


namespace survey_result_l969_96968

def teachers_survey (total : ℕ) (high_bp : ℕ) (heart : ℕ) (diabetes : ℕ) 
  (high_bp_heart : ℕ) (diabetes_heart : ℕ) (diabetes_high_bp : ℕ) (all_three : ℕ) : Prop :=
  let teachers_with_condition := 
    high_bp + heart + diabetes - high_bp_heart - diabetes_heart - diabetes_high_bp + all_three
  let teachers_without_condition := total - teachers_with_condition
  (teachers_without_condition : ℚ) / total * 100 = 28

theorem survey_result : 
  teachers_survey 150 90 60 10 30 5 8 3 :=
by
  sorry

end survey_result_l969_96968


namespace solution_verification_l969_96970

theorem solution_verification (x y : ℝ) : x = 2 ∧ x + y = 3 → y = 1 := by
  sorry

end solution_verification_l969_96970


namespace a_fourth_plus_inverse_a_fourth_l969_96908

theorem a_fourth_plus_inverse_a_fourth (a : ℝ) (h : a - 1/a = -2) : 
  a^4 + 1/a^4 = 34 := by
sorry

end a_fourth_plus_inverse_a_fourth_l969_96908


namespace unique_fixed_point_for_rotation_invariant_function_l969_96907

/-- A function is invariant under π rotation around the origin -/
def RotationInvariant (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (-x) = -y

/-- The main theorem -/
theorem unique_fixed_point_for_rotation_invariant_function (f : ℝ → ℝ) 
    (h : RotationInvariant f) : 
    ∃! x, f x = x :=
  sorry

end unique_fixed_point_for_rotation_invariant_function_l969_96907


namespace circle_parabola_height_difference_l969_96951

/-- The height difference between the center of a circle and its points of tangency with the parabola y = 2x^2 -/
theorem circle_parabola_height_difference (a : ℝ) : 
  ∃ (b r : ℝ), 
    (∀ x y : ℝ, y = 2 * x^2 → x^2 + (y - b)^2 = r^2 → x = a ∨ x = -a) →
    (b - 2 * a^2 = 1/4 - a^2) := by
  sorry

end circle_parabola_height_difference_l969_96951


namespace quadratic_root_property_l969_96963

theorem quadratic_root_property (a b s t : ℝ) (h_neq : s ≠ t) 
  (h_ps : s^2 + a*s + b = t) (h_pt : t^2 + a*t + b = s) : 
  (b - s*t)^2 + a*(b - s*t) + b - s*t = 0 := by
sorry

end quadratic_root_property_l969_96963


namespace parallel_tangents_imply_m_values_l969_96954

-- Define the line and curve
def line (x y : ℝ) : Prop := x - 9 * y - 8 = 0
def curve (x y m : ℝ) : Prop := y = x^3 - m * x^2 + 3 * x

-- Define the tangent slope at a point on the curve
def tangent_slope (x m : ℝ) : ℝ := 3 * x^2 - 2 * m * x + 3

-- State the theorem
theorem parallel_tangents_imply_m_values (m : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    line x₁ y₁ ∧ line x₂ y₂ ∧
    curve x₁ y₁ m ∧ curve x₂ y₂ m ∧
    x₁ ≠ x₂ ∧
    tangent_slope x₁ m = tangent_slope x₂ m) →
  m = 4 ∨ m = -3 :=
sorry

end parallel_tangents_imply_m_values_l969_96954


namespace base2_digit_difference_l969_96927

-- Function to calculate the number of digits in base-2 representation
def base2Digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

-- Theorem statement
theorem base2_digit_difference : base2Digits 1800 - base2Digits 500 = 2 := by
  sorry

end base2_digit_difference_l969_96927


namespace line_inclination_l969_96904

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 1 = 0

-- Define the angle of inclination
def angle_of_inclination (θ : ℝ) : Prop := Real.tan θ = 1 / Real.sqrt 3

-- Theorem statement
theorem line_inclination :
  ∃ θ, angle_of_inclination θ ∧ θ = 30 * π / 180 :=
sorry

end line_inclination_l969_96904


namespace smallest_three_digit_multiple_of_17_l969_96973

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≥ 100 → n.mod 17 = 0 → n ≥ 102 :=
by
  sorry

end smallest_three_digit_multiple_of_17_l969_96973


namespace sum_of_roots_equals_fourteen_l969_96964

theorem sum_of_roots_equals_fourteen : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 7)^2 = 16 ∧ (x₂ - 7)^2 = 16 ∧ x₁ + x₂ = 14 := by
  sorry

end sum_of_roots_equals_fourteen_l969_96964


namespace family_ages_exist_and_unique_l969_96902

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem family_ages_exist_and_unique :
  ∃! (father mother daughter son : ℕ),
    is_perfect_square father ∧
    digit_product father = mother ∧
    digit_sum father = daughter ∧
    digit_sum mother = son ∧
    father ≤ 121 ∧
    mother > 0 ∧
    daughter > 0 ∧
    son > 0 :=
by sorry

end family_ages_exist_and_unique_l969_96902


namespace sally_payment_l969_96906

/-- The amount Sally needs to pay out of pocket to buy books for her students -/
def sally_out_of_pocket (budget : ℚ) (num_students : ℕ) (reading_book_price : ℚ) 
  (math_book_price : ℚ) (discount_rate : ℚ) (discount_threshold : ℕ) : ℚ :=
  let total_reading_books := num_students * reading_book_price
  let discounted_reading_books := if num_students ≥ discount_threshold
    then total_reading_books * (1 - discount_rate)
    else total_reading_books
  let total_math_books := num_students * math_book_price
  let total_cost := discounted_reading_books + total_math_books
  max (total_cost - budget) 0

/-- Theorem stating that Sally needs to pay $467.50 out of pocket -/
theorem sally_payment : 
  sally_out_of_pocket 320 35 15 9 (1/10) 25 = 467.5 := by
  sorry

end sally_payment_l969_96906


namespace marble_weight_proof_l969_96955

/-- The weight of one marble in pounds -/
def marble_weight : ℚ := 100 / 9

/-- The weight of one waffle iron in pounds -/
def waffle_iron_weight : ℚ := 25

theorem marble_weight_proof :
  (9 * marble_weight = 4 * waffle_iron_weight) ∧
  (3 * waffle_iron_weight = 75) →
  marble_weight = 100 / 9 := by
sorry

end marble_weight_proof_l969_96955


namespace page_number_digit_difference_l969_96969

/-- Counts the occurrences of a digit in a range of numbers -/
def countDigit (d : Nat) (start finish : Nat) : Nat :=
  sorry

/-- The difference between the number of 3's and 7's in page numbers of a book -/
def digitDifference (pages : Nat) : Nat :=
  (countDigit 3 1 pages) - (countDigit 7 1 pages)

theorem page_number_digit_difference :
  digitDifference 350 = 56 := by sorry

end page_number_digit_difference_l969_96969


namespace second_file_size_is_90_l969_96980

/-- Represents the download scenario with given conditions -/
structure DownloadScenario where
  internetSpeed : ℕ  -- in megabits per minute
  totalTime : ℕ      -- in minutes
  fileCount : ℕ
  firstFileSize : ℕ  -- in megabits
  thirdFileSize : ℕ  -- in megabits

/-- Calculates the size of the second file given a download scenario -/
def secondFileSize (scenario : DownloadScenario) : ℕ :=
  scenario.internetSpeed * scenario.totalTime - scenario.firstFileSize - scenario.thirdFileSize

/-- Theorem stating that the size of the second file is 90 megabits -/
theorem second_file_size_is_90 (scenario : DownloadScenario) 
  (h1 : scenario.internetSpeed = 2)
  (h2 : scenario.totalTime = 120)
  (h3 : scenario.fileCount = 3)
  (h4 : scenario.firstFileSize = 80)
  (h5 : scenario.thirdFileSize = 70) :
  secondFileSize scenario = 90 := by
  sorry

#eval secondFileSize { 
  internetSpeed := 2, 
  totalTime := 120, 
  fileCount := 3, 
  firstFileSize := 80, 
  thirdFileSize := 70 
}

end second_file_size_is_90_l969_96980


namespace inverse_composition_l969_96975

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv g_inv : ℝ → ℝ)

-- Assumption: f and g are bijective (to ensure inverses exist)
variable (hf : Function.Bijective f)
variable (hg : Function.Bijective g)

-- Define the relationship between f_inv and g
axiom relation (x : ℝ) : f_inv (g x) = 4 * x - 2

-- Theorem to prove
theorem inverse_composition :
  g_inv (f 5) = 7/4 :=
sorry

end inverse_composition_l969_96975


namespace complex_fraction_simplification_l969_96930

theorem complex_fraction_simplification :
  let z₁ : ℂ := 2 + 4*I
  let z₂ : ℂ := 2 - 4*I
  (z₁ / z₂ - z₂ / z₁) = (4:ℝ)/5 * I :=
by sorry

end complex_fraction_simplification_l969_96930


namespace direct_proportion_point_value_l969_96953

/-- A directly proportional function passing through points (-2, 3) and (a, -3) has a = 2 -/
theorem direct_proportion_point_value (k a : ℝ) : 
  (∃ k : ℝ, k * (-2) = 3 ∧ k * a = -3) → a = 2 := by
  sorry

end direct_proportion_point_value_l969_96953


namespace convex_polygon_sides_l969_96914

theorem convex_polygon_sides (n : ℕ) : n > 2 → (n - 1) * 180 - 2008 < 180 ∧ 2008 < (n - 1) * 180 → n = 14 := by
  sorry

end convex_polygon_sides_l969_96914


namespace triangle_theorem_l969_96997

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (t.b - 2 * t.a) * Real.cos t.C + t.c * Real.cos t.B = 0)
  (h2 : t.c = Real.sqrt 7)
  (h3 : t.b = 3 * t.a) :
  t.C = π / 3 ∧ 
  (1 / 2 : ℝ) * t.a * t.b * Real.sin t.C = 3 * Real.sqrt 3 / 4 := by
  sorry


end triangle_theorem_l969_96997


namespace triangle_angle_F_l969_96959

theorem triangle_angle_F (D E : Real) (h1 : 2 * Real.sin D + 5 * Real.cos E = 7)
                         (h2 : 5 * Real.sin E + 2 * Real.cos D = 4) :
  Real.sin (π - D - E) = 9 / 10 := by
  sorry

end triangle_angle_F_l969_96959


namespace set_intersection_theorem_l969_96950

-- Define the sets S and T
def S : Set ℝ := {x : ℝ | x < -5 ∨ x > 5}
def T : Set ℝ := {x : ℝ | -7 < x ∧ x < 3}

-- State the theorem
theorem set_intersection_theorem : S ∩ T = {x : ℝ | -7 < x ∧ x < -5} := by
  sorry

end set_intersection_theorem_l969_96950


namespace total_students_is_1076_l969_96962

/-- Represents the number of students in a school --/
structure School where
  girls : ℕ
  boys : ℕ

/-- The total number of students in the school --/
def School.total (s : School) : ℕ := s.girls + s.boys

/-- A school with 402 more girls than boys and 739 girls --/
def our_school : School := {
  girls := 739,
  boys := 739 - 402
}

/-- Theorem stating that the total number of students in our_school is 1076 --/
theorem total_students_is_1076 : our_school.total = 1076 := by
  sorry

end total_students_is_1076_l969_96962


namespace amin_iff_ali_can_color_all_red_l969_96966

-- Define a type for cell colors
inductive CellColor
| Black
| White
| Red

-- Define the table as a function from coordinates to cell colors
def Table (n : ℕ) := Fin n → Fin n → CellColor

-- Define Amin's move
def AminMove (t : Table n) (row : Fin n) : Table n :=
  sorry

-- Define Ali's move
def AliMove (t : Table n) (col : Fin n) : Table n :=
  sorry

-- Define a predicate to check if all cells are red
def AllRed (t : Table n) : Prop :=
  ∀ i j, t i j = CellColor.Red

-- Define a predicate to check if Amin can color all cells red
def AminCanColorAllRed (t : Table n) : Prop :=
  sorry

-- Define a predicate to check if Ali can color all cells red
def AliCanColorAllRed (t : Table n) : Prop :=
  sorry

-- The main theorem
theorem amin_iff_ali_can_color_all_red (n : ℕ) (t : Table n) :
  AminCanColorAllRed t ↔ AliCanColorAllRed t :=
sorry

end amin_iff_ali_can_color_all_red_l969_96966


namespace stability_comparison_l969_96994

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

end stability_comparison_l969_96994


namespace same_type_as_3a2b_l969_96978

/-- Two terms are of the same type if they have the same variables with the same exponents. -/
def same_type (t1 t2 : ℕ → ℕ → ℕ) : Prop :=
  ∀ a b, t1 a b ≠ 0 ∧ t2 a b ≠ 0 → 
    (t1 a b).factors.toFinset = (t2 a b).factors.toFinset

/-- The term $3a^2b$ -/
def term1 (a b : ℕ) : ℕ := 3 * a^2 * b

/-- The term $2ab^2$ -/
def term2 (a b : ℕ) : ℕ := 2 * a * b^2

/-- The term $-a^2b$ -/
def term3 (a b : ℕ) : ℕ := a^2 * b

/-- The term $-2ab$ -/
def term4 (a b : ℕ) : ℕ := 2 * a * b

/-- The term $5a^2$ -/
def term5 (a b : ℕ) : ℕ := 5 * a^2

theorem same_type_as_3a2b :
  same_type term1 term3 ∧
  ¬ same_type term1 term2 ∧
  ¬ same_type term1 term4 ∧
  ¬ same_type term1 term5 :=
sorry

end same_type_as_3a2b_l969_96978


namespace inequality_solution_set_l969_96989

theorem inequality_solution_set (x : ℝ) : 
  (2 / (x - 1) - 3 / (x - 3) + 5 / (x - 5) - 2 / (x - 7) < 1 / 15) ↔ 
  (x < -8 ∨ (-7 < x ∧ x < -1) ∨ (1 < x ∧ x < 3) ∨ (5 < x ∧ x < 7) ∨ x > 8) :=
by sorry

end inequality_solution_set_l969_96989


namespace total_payment_is_correct_l969_96938

-- Define the payment per lawn
def payment_per_lawn : ℚ := 13 / 3

-- Define the number of lawns mowed
def lawns_mowed : ℚ := 8 / 5

-- Define the base fee
def base_fee : ℚ := 5

-- Theorem statement
theorem total_payment_is_correct :
  payment_per_lawn * lawns_mowed + base_fee = 179 / 15 := by
  sorry

end total_payment_is_correct_l969_96938


namespace max_curved_sides_is_2n_minus_2_l969_96985

/-- A type representing a figure formed by the intersection of circles -/
structure IntersectionFigure where
  n : ℕ
  h_n : n ≥ 2

/-- The number of curved sides in an intersection figure -/
def curved_sides (F : IntersectionFigure) : ℕ := sorry

/-- The maximum number of curved sides for a given number of circles -/
def max_curved_sides (n : ℕ) : ℕ := 2 * n - 2

/-- Theorem stating that the maximum number of curved sides is 2n - 2 -/
theorem max_curved_sides_is_2n_minus_2 (F : IntersectionFigure) :
  curved_sides F ≤ max_curved_sides F.n := by
  sorry

end max_curved_sides_is_2n_minus_2_l969_96985


namespace combined_salaries_l969_96945

/-- The combined salaries of four employees given the salary of the fifth and the average of all five -/
theorem combined_salaries 
  (c_salary : ℕ) 
  (average_salary : ℕ) 
  (h1 : c_salary = 15000)
  (h2 : average_salary = 8800) :
  c_salary + 4 * average_salary - 5 * average_salary = 29000 :=
by sorry

end combined_salaries_l969_96945


namespace average_chapters_per_book_l969_96913

theorem average_chapters_per_book 
  (total_chapters : Float) 
  (total_books : Float) 
  (h1 : total_chapters = 17.0) 
  (h2 : total_books = 4.0) : 
  total_chapters / total_books = 4.25 := by
sorry

end average_chapters_per_book_l969_96913


namespace cosine_theorem_trirectangular_angle_l969_96941

-- Define the trirectangular angle
structure TrirectangularAngle where
  α : Real  -- plane angle opposite to SA
  β : Real  -- plane angle opposite to SB
  γ : Real  -- plane angle opposite to SC
  A : Real  -- dihedral angle at SA
  B : Real  -- dihedral angle at SB
  C : Real  -- dihedral angle at SC

-- State the theorem
theorem cosine_theorem_trirectangular_angle (t : TrirectangularAngle) :
  Real.cos t.α = Real.cos t.A * Real.cos t.B + Real.cos t.B * Real.cos t.C + Real.cos t.C * Real.cos t.A := by
  sorry

end cosine_theorem_trirectangular_angle_l969_96941


namespace rotation_of_doubled_complex_l969_96924

theorem rotation_of_doubled_complex :
  let z : ℂ := 3 - 4*I
  let doubled : ℂ := 2 * z
  let rotated : ℂ := -doubled
  rotated = -6 + 8*I :=
by
  sorry

end rotation_of_doubled_complex_l969_96924


namespace remainder_3_250_mod_11_l969_96974

theorem remainder_3_250_mod_11 : 3^250 % 11 = 1 := by
  sorry

end remainder_3_250_mod_11_l969_96974


namespace find_b_l969_96995

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then 3 * x - b else 2^x

theorem find_b : ∃ b : ℝ, f b (f b (5/6)) = 4 ∧ b = 1/2 := by
  sorry

end find_b_l969_96995


namespace matrix_vector_product_l969_96960

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; -6, 5]
def v : Matrix (Fin 2) (Fin 1) ℝ := !![2; -3]

theorem matrix_vector_product :
  A * v = !![14; -27] := by sorry

end matrix_vector_product_l969_96960


namespace local_min_implies_a_half_subset_implies_a_range_l969_96977

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * Real.log x - a * (x^2 - 1)

-- Part 1: Local minimum at x = 1 implies a = 1/2
theorem local_min_implies_a_half (a : ℝ) :
  (∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → f a x ≥ f a 1) →
  a = 1/2 :=
sorry

-- Part 2: N ⊆ M implies a ∈ (-∞, 1/2]
theorem subset_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≥ 0) →
  a ≤ 1/2 :=
sorry

end local_min_implies_a_half_subset_implies_a_range_l969_96977


namespace exactly_two_valid_A_values_l969_96901

/-- A function that checks if a number is divisible by 8 based on its last three digits -/
def isDivisibleBy8 (n : ℕ) : Prop :=
  n % 8 = 0

/-- A function that constructs the number 451,2A8 given A -/
def constructNumber (A : ℕ) : ℕ :=
  451200 + A * 10 + 8

/-- The main theorem stating that there are exactly 2 single-digit values of A satisfying both conditions -/
theorem exactly_two_valid_A_values :
  ∃! (S : Finset ℕ), S.card = 2 ∧ 
    (∀ A ∈ S, A < 10 ∧ 120 % A = 0 ∧ isDivisibleBy8 (constructNumber A)) ∧
    (∀ A < 10, 120 % A = 0 ∧ isDivisibleBy8 (constructNumber A) → A ∈ S) :=
sorry

end exactly_two_valid_A_values_l969_96901


namespace race_earnings_theorem_l969_96961

/-- Calculates the average earnings per minute for the race winner -/
def average_earnings_per_minute (race_duration : ℕ) (lap_distance : ℕ) (gift_rate : ℚ) (winner_laps : ℕ) : ℚ :=
  let total_distance := winner_laps * lap_distance
  let total_earnings := (total_distance / 100) * gift_rate
  total_earnings / race_duration

/-- Theorem stating that the average earnings per minute is $7 given the race conditions -/
theorem race_earnings_theorem :
  average_earnings_per_minute 12 100 (7/2) 24 = 7 := by
  sorry

end race_earnings_theorem_l969_96961


namespace absolute_value_equation_l969_96972

theorem absolute_value_equation (a : ℝ) : 
  |2*a + 1| = 3*|a| - 2 → a = -1 ∨ a = 3 := by
sorry

end absolute_value_equation_l969_96972
