import Mathlib

namespace NUMINAMATH_CALUDE_congruence_problem_l1009_100969

theorem congruence_problem : ∃ n : ℕ, 0 ≤ n ∧ n < 9 ∧ 
  (3 * (2 + 44 + 666 + 8888 + 111110 + 13131312 + 1515151514)) % 9 = n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1009_100969


namespace NUMINAMATH_CALUDE_molecular_weight_is_265_21_l1009_100962

/-- Atomic weight of Aluminium in amu -/
def Al_weight : ℝ := 26.98

/-- Atomic weight of Oxygen in amu -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Hydrogen in amu -/
def H_weight : ℝ := 1.01

/-- Atomic weight of Silicon in amu -/
def Si_weight : ℝ := 28.09

/-- Atomic weight of Nitrogen in amu -/
def N_weight : ℝ := 14.01

/-- Number of Aluminium atoms in the compound -/
def Al_count : ℕ := 2

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 6

/-- Number of Hydrogen atoms in the compound -/
def H_count : ℕ := 3

/-- Number of Silicon atoms in the compound -/
def Si_count : ℕ := 2

/-- Number of Nitrogen atoms in the compound -/
def N_count : ℕ := 4

/-- The molecular weight of the compound -/
def molecular_weight : ℝ :=
  Al_count * Al_weight + O_count * O_weight + H_count * H_weight +
  Si_count * Si_weight + N_count * N_weight

theorem molecular_weight_is_265_21 : molecular_weight = 265.21 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_is_265_21_l1009_100962


namespace NUMINAMATH_CALUDE_soccer_game_scoring_l1009_100964

theorem soccer_game_scoring (team_a_first_half : ℕ) : 
  (team_a_first_half : ℝ) + (team_a_first_half : ℝ) / 2 + 
  (team_a_first_half : ℝ) + (team_a_first_half : ℝ) - 2 = 26 →
  team_a_first_half = 8 := by
sorry

end NUMINAMATH_CALUDE_soccer_game_scoring_l1009_100964


namespace NUMINAMATH_CALUDE_line_equation_l1009_100992

/-- Given a line y = kx + b passing through points (-1, 0) and (0, 3),
    prove that its equation is y = 3x + 3 -/
theorem line_equation (k b : ℝ) : 
  (k * (-1) + b = 0) → (b = 3) → (k = 3 ∧ b = 3) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l1009_100992


namespace NUMINAMATH_CALUDE_tree_height_difference_l1009_100931

def maple_height : ℚ := 10 + 3/4
def pine_height : ℚ := 12 + 7/8

theorem tree_height_difference :
  pine_height - maple_height = 2 + 1/8 := by sorry

end NUMINAMATH_CALUDE_tree_height_difference_l1009_100931


namespace NUMINAMATH_CALUDE_part_one_part_two_l1009_100932

-- Define sets A and B
def A : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Part 1: Prove that (C_R B) ∩ A = {x | 3 ≤ x ≤ 5} when m = 3
theorem part_one : (Set.compl (B 3) ∩ A) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

-- Part 2: Prove that if A ∩ B = {x | -1 < x < 4}, then m = 8
theorem part_two : (∃ m : ℝ, A ∩ B m = {x | -1 < x ∧ x < 4}) → (∃ m : ℝ, m = 8) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1009_100932


namespace NUMINAMATH_CALUDE_age_difference_l1009_100960

theorem age_difference (younger_age elder_age : ℕ) 
  (h1 : younger_age = 33)
  (h2 : elder_age = 53) : 
  elder_age - younger_age = 20 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l1009_100960


namespace NUMINAMATH_CALUDE_f_value_at_2_l1009_100963

/-- Given a function f(x) = a*sin(x) + b*x*cos(x) - 2c*tan(x) + x^2 where f(-2) = 3,
    prove that f(2) = 5 -/
theorem f_value_at_2 (a b c : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * Real.sin x + b * x * Real.cos x - 2 * c * Real.tan x + x^2)
  (h2 : f (-2) = 3) :
  f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l1009_100963


namespace NUMINAMATH_CALUDE_series_growth_l1009_100953

theorem series_growth (n : ℕ) (h : n > 1) :
  (Finset.range (2^(n+1) - 1)).card - (Finset.range (2^n - 1)).card = 2^n :=
sorry

end NUMINAMATH_CALUDE_series_growth_l1009_100953


namespace NUMINAMATH_CALUDE_root_in_interval_implies_a_range_l1009_100990

-- Define the function f
def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 2

-- Define the theorem
theorem root_in_interval_implies_a_range :
  ∀ a : ℝ, (∃ x ∈ Set.Icc (-1) 1, f a x = 0) → a ∈ Set.Iic (-1) ∪ Set.Ici 1 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_a_range_l1009_100990


namespace NUMINAMATH_CALUDE_daughters_age_l1009_100906

theorem daughters_age (father_age : ℕ) (daughter_age : ℕ) : 
  father_age = 40 → 
  father_age = 4 * daughter_age → 
  father_age + 20 = 2 * (daughter_age + 20) → 
  daughter_age = 10 := by
sorry

end NUMINAMATH_CALUDE_daughters_age_l1009_100906


namespace NUMINAMATH_CALUDE_inequality_implies_range_l1009_100957

theorem inequality_implies_range (a : ℝ) : (1^2 * a + 2 * 1 + 1 < 0) → a < -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_range_l1009_100957


namespace NUMINAMATH_CALUDE_equation_equivalence_l1009_100916

theorem equation_equivalence (a b c : ℕ) 
  (ha : 0 < a ∧ a < 12) 
  (hb : 0 < b ∧ b < 12) 
  (hc : 0 < c ∧ c < 12) : 
  ((12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c) ↔ (b + c = 12) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1009_100916


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1009_100959

/-- The equation of a hyperbola with given conditions -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →
  (∃ c : ℝ, c / a = Real.sqrt 3) →
  (∃ k : ℝ, ∀ x : ℝ, k = -1 ∧ k = a^2 / (a * Real.sqrt 3)) →
  (x^2 / 3 - y^2 / 6 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1009_100959


namespace NUMINAMATH_CALUDE_exponential_decreasing_range_l1009_100983

/-- A function f: ℝ → ℝ is strictly decreasing -/
def StrictlyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem exponential_decreasing_range (a : ℝ) :
  StrictlyDecreasing (fun x ↦ (a - 1) ^ x) → 1 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_decreasing_range_l1009_100983


namespace NUMINAMATH_CALUDE_certain_number_proof_l1009_100919

theorem certain_number_proof (h : 213 * 16 = 3408) : 
  ∃ x : ℝ, 0.016 * x = 0.03408 ∧ x = 2.13 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1009_100919


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l1009_100937

theorem geometric_sequence_second_term 
  (a : ℕ → ℚ) -- a is the sequence
  (h1 : a 3 = 12) -- third term is 12
  (h2 : a 4 = 18) -- fourth term is 18
  (h3 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * (a 4 / a 3)) -- definition of geometric sequence
  : a 2 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l1009_100937


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l1009_100933

/-- A circle with equation x^2 + y^2 = 4m is tangent to a line with equation x - y = 2√m if and only if m = 0 -/
theorem circle_tangent_to_line (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 = 4*m ∧ x - y = 2*Real.sqrt m) →
  (∀ (x y : ℝ), x^2 + y^2 = 4*m → (x - y ≠ 2*Real.sqrt m ∨ (x - y = 2*Real.sqrt m ∧ 
    ∀ ε > 0, ∃ x' y', (x' - x)^2 + (y' - y)^2 < ε^2 ∧ x'^2 + y'^2 > 4*m))) →
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l1009_100933


namespace NUMINAMATH_CALUDE_tan_eleven_pi_over_four_l1009_100914

theorem tan_eleven_pi_over_four : Real.tan (11 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_eleven_pi_over_four_l1009_100914


namespace NUMINAMATH_CALUDE_sum_first_15_even_positive_l1009_100988

/-- The sum of the first n even positive integers -/
def sum_first_n_even_positive (n : ℕ) : ℕ :=
  n * (n + 1)

/-- Theorem: The sum of the first 15 even positive integers is 240 -/
theorem sum_first_15_even_positive :
  sum_first_n_even_positive 15 = 240 := by
  sorry

#eval sum_first_n_even_positive 15  -- This should output 240

end NUMINAMATH_CALUDE_sum_first_15_even_positive_l1009_100988


namespace NUMINAMATH_CALUDE_total_cost_is_2000_l1009_100993

/-- The cost of buying two laptops, where the first laptop costs $500 and the second laptop is 3 times as costly as the first laptop. -/
def total_cost (first_laptop_cost : ℕ) (cost_multiplier : ℕ) : ℕ :=
  first_laptop_cost + (cost_multiplier * first_laptop_cost)

/-- Theorem stating that the total cost of buying both laptops is $2000. -/
theorem total_cost_is_2000 : total_cost 500 3 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_2000_l1009_100993


namespace NUMINAMATH_CALUDE_second_smallest_sum_of_two_cubes_l1009_100954

-- Define a function to check if a number is the sum of two cubes
def isSumOfTwoCubes (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ a^3 + b^3 = n

-- Define a function to check if a number can be written as the sum of two cubes in two different ways
def hasTwoDifferentCubeRepresentations (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a^3 + b^3 = n ∧ c^3 + d^3 = n ∧
    (a ≠ c ∨ b ≠ d) ∧ (a ≠ d ∨ b ≠ c)

-- Define the theorem
theorem second_smallest_sum_of_two_cubes : 
  (∃ m : ℕ, m < 4104 ∧ hasTwoDifferentCubeRepresentations m) ∧
  (∀ k : ℕ, k < 4104 → k ≠ 1729 → ¬hasTwoDifferentCubeRepresentations k) ∧
  hasTwoDifferentCubeRepresentations 4104 :=
sorry

end NUMINAMATH_CALUDE_second_smallest_sum_of_two_cubes_l1009_100954


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l1009_100987

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number we want to represent in scientific notation -/
def target_number : ℕ := 101000

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 1.01
    exponent := 5
    coeff_range := by sorry }

theorem scientific_notation_correct :
  (proposed_notation.coefficient * (10 : ℝ) ^ proposed_notation.exponent) = target_number := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l1009_100987


namespace NUMINAMATH_CALUDE_five_solutions_l1009_100909

/-- The number of integer solution pairs (x, y) to the equation √x + √y = √336 -/
def num_solutions : ℕ := 5

/-- A predicate that checks if a pair of natural numbers satisfies the equation -/
def is_solution (x y : ℕ) : Prop :=
  Real.sqrt (x : ℝ) + Real.sqrt (y : ℝ) = Real.sqrt 336

/-- The theorem stating that there are exactly 5 solution pairs -/
theorem five_solutions :
  ∃! (s : Finset (ℕ × ℕ)), s.card = num_solutions ∧ ∀ (x y : ℕ), (x, y) ∈ s ↔ is_solution x y :=
sorry

end NUMINAMATH_CALUDE_five_solutions_l1009_100909


namespace NUMINAMATH_CALUDE_f_min_value_f_attains_min_l1009_100910

def f (x : ℝ) : ℝ := (x - 1)^2 + 2

theorem f_min_value : ∀ x : ℝ, f x ≥ 2 := by sorry

theorem f_attains_min : ∃ x : ℝ, f x = 2 := by sorry

end NUMINAMATH_CALUDE_f_min_value_f_attains_min_l1009_100910


namespace NUMINAMATH_CALUDE_remainder_mod_six_l1009_100997

theorem remainder_mod_six (n : ℕ) (h : n % 12 = 8) : n % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_six_l1009_100997


namespace NUMINAMATH_CALUDE_remaining_credit_l1009_100917

/-- Calculates the remaining credit to be paid given a credit limit and two payments -/
theorem remaining_credit (credit_limit : ℕ) (payment1 : ℕ) (payment2 : ℕ) :
  credit_limit = 100 →
  payment1 = 15 →
  payment2 = 23 →
  credit_limit - (payment1 + payment2) = 62 := by
  sorry

#check remaining_credit

end NUMINAMATH_CALUDE_remaining_credit_l1009_100917


namespace NUMINAMATH_CALUDE_sequence_periodicity_l1009_100955

def sequence_rule (a : ℕ) (u : ℕ → ℕ) : Prop :=
  ∀ n, (Even (u n) → u (n + 1) = (u n) / 2) ∧
       (Odd (u n) → u (n + 1) = a + u n)

theorem sequence_periodicity (a : ℕ) (u : ℕ → ℕ) 
  (h1 : Odd a) 
  (h2 : a > 0) 
  (h3 : sequence_rule a u) :
  ∃ k : ℕ, ∃ p : ℕ, p > 0 ∧ ∀ n ≥ k, u (n + p) = u n :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l1009_100955


namespace NUMINAMATH_CALUDE_work_completion_time_l1009_100904

theorem work_completion_time (a b : ℝ) (h1 : a = 2 * b) (h2 : 1 / a + 1 / b = 3 / 10) : 1 / b = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1009_100904


namespace NUMINAMATH_CALUDE_north_southland_population_increase_l1009_100958

/-- The number of hours between each birth in North Southland -/
def hours_between_births : ℕ := 6

/-- The number of deaths per day in North Southland -/
def deaths_per_day : ℕ := 2

/-- The number of days in a year -/
def days_per_year : ℕ := 365

/-- The population increase in North Southland per year -/
def population_increase : ℕ := 
  (24 / hours_between_births - deaths_per_day) * days_per_year

/-- The population increase in North Southland rounded to the nearest hundred -/
def rounded_population_increase : ℕ := 
  (population_increase + 50) / 100 * 100

theorem north_southland_population_increase :
  rounded_population_increase = 700 := by sorry

end NUMINAMATH_CALUDE_north_southland_population_increase_l1009_100958


namespace NUMINAMATH_CALUDE_cereal_spending_l1009_100905

theorem cereal_spending (total : ℝ) (snap crackle pop : ℝ) : 
  total = 150 ∧ 
  snap = 2 * crackle ∧ 
  crackle = 3 * pop ∧ 
  total = snap + crackle + pop → 
  pop = 15 := by
  sorry

end NUMINAMATH_CALUDE_cereal_spending_l1009_100905


namespace NUMINAMATH_CALUDE_circle_equation_l1009_100982

/-- The equation (x - 3)^2 + (y + 4)^2 = 9 represents a circle centered at (3, -4) with radius 3 -/
theorem circle_equation (x y : ℝ) : 
  (x - 3)^2 + (y + 4)^2 = 9 ↔ 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (3, -4) ∧ 
    radius = 3 ∧ 
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1009_100982


namespace NUMINAMATH_CALUDE_plan1_more_profitable_l1009_100968

/-- Represents the monthly production and profit of a factory with two wastewater treatment plans -/
structure FactoryProduction where
  x : ℕ  -- Number of products produced per month
  y1 : ℤ -- Monthly profit for Plan 1 in yuan
  y2 : ℤ -- Monthly profit for Plan 2 in yuan

/-- Calculates the monthly profit for Plan 1 -/
def plan1Profit (x : ℕ) : ℤ :=
  24 * x - 30000

/-- Calculates the monthly profit for Plan 2 -/
def plan2Profit (x : ℕ) : ℤ :=
  18 * x

/-- Theorem stating that Plan 1 yields more profit when producing 6000 products per month -/
theorem plan1_more_profitable :
  let production : FactoryProduction := {
    x := 6000,
    y1 := plan1Profit 6000,
    y2 := plan2Profit 6000
  }
  production.y1 > production.y2 :=
by sorry

end NUMINAMATH_CALUDE_plan1_more_profitable_l1009_100968


namespace NUMINAMATH_CALUDE_valid_triples_l1009_100966

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

def isGeometricSequence (x y z : Nat) : Prop := ∃ r : ℚ, y = x * r ∧ z = y * r

def validTriple (a b c : Nat) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧
  a < b ∧ b < c ∧ c < 100 ∧
  isGeometricSequence (a + 1) (b + 1) (c + 1)

theorem valid_triples :
  {t : Nat × Nat × Nat | validTriple t.1 t.2.1 t.2.2} =
  {(2, 5, 11), (2, 11, 47), (5, 11, 23), (5, 17, 53), (7, 23, 71), (11, 23, 47)} :=
by sorry

end NUMINAMATH_CALUDE_valid_triples_l1009_100966


namespace NUMINAMATH_CALUDE_exists_158_consecutive_not_div_17_exists_div_17_in_159_consecutive_l1009_100996

-- Define a function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem 1: There exists a sequence of 158 consecutive integers where the sum of digits of each number is not divisible by 17
theorem exists_158_consecutive_not_div_17 : 
  ∃ (start : ℕ), ∀ (i : ℕ), i < 158 → ¬(17 ∣ sum_of_digits (start + i)) :=
sorry

-- Theorem 2: For any sequence of 159 consecutive integers, there exists at least one integer in the sequence whose sum of digits is divisible by 17
theorem exists_div_17_in_159_consecutive (start : ℕ) : 
  ∃ (i : ℕ), i < 159 ∧ (17 ∣ sum_of_digits (start + i)) :=
sorry

end NUMINAMATH_CALUDE_exists_158_consecutive_not_div_17_exists_div_17_in_159_consecutive_l1009_100996


namespace NUMINAMATH_CALUDE_amy_connor_score_difference_l1009_100943

theorem amy_connor_score_difference
  (connor_score : ℕ)
  (amy_score : ℕ)
  (jason_score : ℕ)
  (connor_scored_two : connor_score = 2)
  (amy_scored_more : amy_score > connor_score)
  (jason_scored_twice_amy : jason_score = 2 * amy_score)
  (team_total_score : connor_score + amy_score + jason_score = 20) :
  amy_score - connor_score = 4 := by
sorry

end NUMINAMATH_CALUDE_amy_connor_score_difference_l1009_100943


namespace NUMINAMATH_CALUDE_infinite_solutions_equation_l1009_100985

theorem infinite_solutions_equation :
  ∃ (x y z : ℕ → ℕ), ∀ n : ℕ,
    (x n)^2 + (x n + 1)^2 = (y n)^2 ∧
    z n = 2 * (x n) + 1 ∧
    (z n)^2 = 2 * (y n)^2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_equation_l1009_100985


namespace NUMINAMATH_CALUDE_logarithmic_function_fixed_point_l1009_100900

theorem logarithmic_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ Real.log x / Real.log a + 1
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_logarithmic_function_fixed_point_l1009_100900


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l1009_100973

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  side_count : ℕ
  total_cubes : ℕ
  inner_cubes : ℕ

/-- The number of smaller cubes with no faces colored in a painted cube cut into 64 equal parts -/
def painted_cube_inner_count : ℕ := 8

/-- Theorem: In a cube cut into 64 equal smaller cubes, 
    the number of smaller cubes with no faces touching the original cube's surface is 8 -/
theorem painted_cube_theorem (c : CutCube) 
  (h1 : c.side_count = 4)
  (h2 : c.total_cubes = 64)
  (h3 : c.inner_cubes = (c.side_count - 2)^3) :
  c.inner_cubes = painted_cube_inner_count := by sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l1009_100973


namespace NUMINAMATH_CALUDE_min_white_surface_area_l1009_100986

/-- Represents a cube with some faces painted gray and others white -/
structure PaintedCube where
  grayFaces : Fin 6 → Bool

/-- The number of identical structures -/
def numStructures : Nat := 7

/-- The number of cubes in each structure -/
def cubesPerStructure : Nat := 8

/-- The number of additional white cubes -/
def additionalWhiteCubes : Nat := 8

/-- The edge length of each small cube in cm -/
def smallCubeEdgeLength : ℝ := 1

/-- The total number of cubes used to construct the large cube -/
def totalCubes : Nat := numStructures * cubesPerStructure + additionalWhiteCubes

/-- The edge length of the large cube in terms of small cubes -/
def largeCubeEdgeLength : Nat := 4

/-- The surface area of the large cube in cm² -/
def largeCubeSurfaceArea : ℝ := 6 * (largeCubeEdgeLength * largeCubeEdgeLength : ℝ) * smallCubeEdgeLength ^ 2

/-- A function to calculate the maximum possible gray surface area -/
def maxGraySurfaceArea : ℝ := 84

/-- Theorem stating that the minimum white surface area is 12 cm² -/
theorem min_white_surface_area :
  largeCubeSurfaceArea - maxGraySurfaceArea = 12 := by sorry

end NUMINAMATH_CALUDE_min_white_surface_area_l1009_100986


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1009_100926

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) ↔ a ∈ Set.Ioi 3 ∪ Set.Iio 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1009_100926


namespace NUMINAMATH_CALUDE_probability_all_white_balls_l1009_100924

/-- The probability of drawing 7 white balls from a box containing 7 white and 8 black balls -/
theorem probability_all_white_balls (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (drawn_balls : ℕ) :
  total_balls = white_balls + black_balls →
  total_balls = 15 →
  white_balls = 7 →
  black_balls = 8 →
  drawn_balls = 7 →
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 1 / 6435 :=
by sorry

end NUMINAMATH_CALUDE_probability_all_white_balls_l1009_100924


namespace NUMINAMATH_CALUDE_only_two_random_events_l1009_100939

-- Define the universe of events
inductive Event : Type
| real_number_multiplication : Event
| draw_odd_numbered_ball : Event
| win_lottery : Event
| number_inequality : Event

-- Define a predicate for random events
def is_random_event : Event → Prop
| Event.real_number_multiplication => False
| Event.draw_odd_numbered_ball => True
| Event.win_lottery => True
| Event.number_inequality => False

-- Theorem statement
theorem only_two_random_events :
  (∀ e : Event, is_random_event e ↔ (e = Event.draw_odd_numbered_ball ∨ e = Event.win_lottery)) :=
by sorry

end NUMINAMATH_CALUDE_only_two_random_events_l1009_100939


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l1009_100923

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- The theorem states that if the points (1,-2), (3,4), and (6,k/3) are collinear, then k = 39. -/
theorem collinear_points_k_value :
  ∀ k : ℝ, collinear 1 (-2) 3 4 6 (k/3) → k = 39 := by
  sorry

#check collinear_points_k_value

end NUMINAMATH_CALUDE_collinear_points_k_value_l1009_100923


namespace NUMINAMATH_CALUDE_students_playing_cricket_l1009_100925

theorem students_playing_cricket 
  (total_students : ℕ) 
  (football_players : ℕ) 
  (neither_players : ℕ) 
  (both_players : ℕ) 
  (h1 : total_students = 450)
  (h2 : football_players = 325)
  (h3 : neither_players = 50)
  (h4 : both_players = 100)
  : ∃ cricket_players : ℕ, cricket_players = 175 :=
by
  sorry

end NUMINAMATH_CALUDE_students_playing_cricket_l1009_100925


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1009_100912

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- Define the equation
def equation (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- Define the theorem
theorem arithmetic_sequence_properties
  (a : ℝ) (d : ℝ)
  (h1 : equation a 1 = 0)
  (h2 : equation a d = 0) :
  ∃ (S_n : ℕ → ℝ) (T_n : ℕ → ℝ),
    (∀ n : ℕ, arithmetic_sequence a d n = n + 1) ∧
    (∀ n : ℕ, S_n n = (n^2 + 3*n) / 2) ∧
    (∀ n : ℕ, T_n n = 1 + (n - 1) * 3^n + (3^n - 1) / 2) :=
by sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1009_100912


namespace NUMINAMATH_CALUDE_smallest_n_square_cube_l1009_100977

theorem smallest_n_square_cube : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^3) ∧ 
  (∀ (x : ℕ), x > 0 → 
    (∃ (y : ℕ), 4 * x = y^2) → 
    (∃ (z : ℕ), 5 * x = z^3) → 
    n ≤ x) ∧
  n = 25 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_square_cube_l1009_100977


namespace NUMINAMATH_CALUDE_ellipse_focus_k_l1009_100908

-- Define the ellipse
def ellipse (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k + 2) + y^2 / 9 = 1

-- Define the focus
def focus : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem ellipse_focus_k (k : ℝ) :
  (∀ x y, ellipse k x y) → (focus.1 = 0 ∧ focus.2 = 2) → k = 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_k_l1009_100908


namespace NUMINAMATH_CALUDE_ellipse_and_trajectory_l1009_100935

/-- Definition of the ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Point A is on the ellipse C -/
def point_A_on_C (a b : ℝ) : Prop := ellipse_C 1 (3/2) a b

/-- Sum of distances from A to foci equals 4 -/
def sum_distances_4 (a : ℝ) : Prop := 2 * a = 4

/-- Conditions on a and b -/
def a_b_conditions (a b : ℝ) : Prop := a > b ∧ b > 0

/-- Theorem stating the equation of ellipse C and the trajectory of midpoint M -/
theorem ellipse_and_trajectory (a b : ℝ) 
  (h1 : a_b_conditions a b) 
  (h2 : point_A_on_C a b) 
  (h3 : sum_distances_4 a) : 
  (∀ x y, ellipse_C x y a b ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∃ F1 : ℝ × ℝ, ∀ x y, 
    (∃ x1 y1, ellipse_C x1 y1 a b ∧ x = (F1.1 + x1) / 2 ∧ y = (F1.2 + y1) / 2) ↔ 
    (x + 1/2)^2 + 4*y^2 / 3 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_trajectory_l1009_100935


namespace NUMINAMATH_CALUDE_toy_car_gift_difference_l1009_100929

/-- Proves the difference between Mum's and Dad's toy car gifts --/
theorem toy_car_gift_difference :
  ∀ (initial final dad uncle auntie grandpa : ℕ),
  initial = 150 →
  final = 196 →
  dad = 10 →
  auntie = uncle + 1 →
  auntie = 6 →
  grandpa = 2 * uncle →
  ∃ (mum : ℕ),
    final = initial + dad + uncle + auntie + grandpa + mum ∧
    mum - dad = 5 := by
  sorry

end NUMINAMATH_CALUDE_toy_car_gift_difference_l1009_100929


namespace NUMINAMATH_CALUDE_abs_eq_self_not_negative_l1009_100941

theorem abs_eq_self_not_negative (x : ℝ) : |x| = x → x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_self_not_negative_l1009_100941


namespace NUMINAMATH_CALUDE_parallel_line_point_slope_form_l1009_100942

/-- Given points A, B, and C in the plane, this theorem states that the line passing through A
    and parallel to BC has the specified point-slope form. -/
theorem parallel_line_point_slope_form 
  (A B C : ℝ × ℝ) 
  (hA : A = (4, 6)) 
  (hB : B = (-3, -1)) 
  (hC : C = (5, -5)) : 
  ∃ (m : ℝ), m = -1/2 ∧ 
  ∀ (x y : ℝ), (y - 6 = m * (x - 4) ↔ 
    (∃ (t : ℝ), (x, y) = A + t • (C - B) ∧ (x, y) ≠ A)) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_point_slope_form_l1009_100942


namespace NUMINAMATH_CALUDE_fraction_irreducible_l1009_100903

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l1009_100903


namespace NUMINAMATH_CALUDE_infinite_pairs_exist_l1009_100950

/-- C(n) is the number of distinct prime divisors of n -/
def C (n : ℕ) : ℕ := sorry

/-- There exist infinitely many pairs of natural numbers (a,b) satisfying the given conditions -/
theorem infinite_pairs_exist : ∀ k : ℕ, ∃ a b : ℕ, a ≠ b ∧ a > k ∧ b > k ∧ C (a + b) = C a + C b := by
  sorry

end NUMINAMATH_CALUDE_infinite_pairs_exist_l1009_100950


namespace NUMINAMATH_CALUDE_solution_value_l1009_100936

theorem solution_value (a : ℝ) : (∃ x : ℝ, x = -2 ∧ a * x - 6 = a + 3) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1009_100936


namespace NUMINAMATH_CALUDE_pie_arrangement_rows_l1009_100965

/-- Given the number of pecan and apple pies, calculates the number of complete rows when arranged with a fixed number of pies per row. -/
def calculate_rows (pecan_pies apple_pies pies_per_row : ℕ) : ℕ :=
  (pecan_pies + apple_pies) / pies_per_row

/-- Proves that 16 pecan pies and 14 apple pies, when arranged in rows of 5 pies each, result in 6 complete rows. -/
theorem pie_arrangement_rows : calculate_rows 16 14 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pie_arrangement_rows_l1009_100965


namespace NUMINAMATH_CALUDE_six_pairs_l1009_100947

/-- The number of distinct pairs of integers (x, y) satisfying the conditions -/
def num_pairs : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    0 < p.1 ∧ p.1 < p.2 ∧ p.1 * p.2 = 2025
  ) (Finset.product (Finset.range 2026) (Finset.range 2026))).card

/-- Theorem stating that there are exactly 6 pairs satisfying the conditions -/
theorem six_pairs : num_pairs = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_pairs_l1009_100947


namespace NUMINAMATH_CALUDE_sin_2x_value_l1009_100920

theorem sin_2x_value (x : ℝ) (h : Real.cos (x - π/4) = 4/5) : Real.sin (2*x) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_value_l1009_100920


namespace NUMINAMATH_CALUDE_paul_sunday_bags_l1009_100980

/-- The number of bags Paul filled on Saturday -/
def saturday_bags : ℕ := 6

/-- The number of cans in each bag -/
def cans_per_bag : ℕ := 8

/-- The total number of cans collected -/
def total_cans : ℕ := 72

/-- The number of bags Paul filled on Sunday -/
def sunday_bags : ℕ := (total_cans - saturday_bags * cans_per_bag) / cans_per_bag

theorem paul_sunday_bags :
  sunday_bags = 3 := by sorry

end NUMINAMATH_CALUDE_paul_sunday_bags_l1009_100980


namespace NUMINAMATH_CALUDE_semicircle_area_l1009_100930

theorem semicircle_area (rectangle_width : Real) (rectangle_length : Real)
  (triangle_leg1 : Real) (triangle_leg2 : Real) :
  rectangle_width = 1 →
  rectangle_length = 3 →
  triangle_leg1 = 1 →
  triangle_leg2 = 2 →
  triangle_leg1^2 + triangle_leg2^2 = rectangle_length^2 →
  (π * rectangle_length^2) / 8 = 9 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_l1009_100930


namespace NUMINAMATH_CALUDE_f_f_zero_l1009_100989

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 3 * x^2 - 4
  else if x = 0 then Real.pi
  else 0

theorem f_f_zero : f (f 0) = 3 * Real.pi^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_f_f_zero_l1009_100989


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l1009_100902

theorem factorial_equation_solutions :
  ∀ x y z : ℕ, 2^x + 5^y + 63 = z.factorial → 
    ((x = 5 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6)) := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solutions_l1009_100902


namespace NUMINAMATH_CALUDE_fermat_prime_power_of_two_l1009_100971

theorem fermat_prime_power_of_two (n : ℕ+) : 
  (Nat.Prime (2^(n : ℕ) + 1)) → (∃ k : ℕ, n = 2^k) := by
  sorry

end NUMINAMATH_CALUDE_fermat_prime_power_of_two_l1009_100971


namespace NUMINAMATH_CALUDE_function_composition_properties_l1009_100961

theorem function_composition_properties :
  (¬ ∃ (f g : ℝ → ℝ), ∀ x, f (g x) = x^2 ∧ g (f x) = x^3) ∧
  (∃ (f g : ℝ → ℝ), ∀ x, f (g x) = x^2 ∧ g (f x) = x^4) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_properties_l1009_100961


namespace NUMINAMATH_CALUDE_days_to_pay_for_register_is_8_l1009_100938

/-- The number of days required to pay for a cash register given daily sales and costs -/
def days_to_pay_for_register (register_cost : ℕ) (bread_sold : ℕ) (bread_price : ℕ) 
  (cakes_sold : ℕ) (cake_price : ℕ) (rent : ℕ) (electricity : ℕ) : ℕ :=
  let daily_revenue := bread_sold * bread_price + cakes_sold * cake_price
  let daily_expenses := rent + electricity
  let daily_profit := daily_revenue - daily_expenses
  (register_cost + daily_profit - 1) / daily_profit

theorem days_to_pay_for_register_is_8 :
  days_to_pay_for_register 1040 40 2 6 12 20 2 = 8 := by sorry

end NUMINAMATH_CALUDE_days_to_pay_for_register_is_8_l1009_100938


namespace NUMINAMATH_CALUDE_gloria_tickets_count_l1009_100915

/-- Given that Gloria has 9 boxes of tickets and each box contains 5 tickets,
    prove that the total number of tickets is 45. -/
theorem gloria_tickets_count :
  let num_boxes : ℕ := 9
  let tickets_per_box : ℕ := 5
  num_boxes * tickets_per_box = 45 := by
  sorry

end NUMINAMATH_CALUDE_gloria_tickets_count_l1009_100915


namespace NUMINAMATH_CALUDE_money_problem_l1009_100984

theorem money_problem (a b : ℚ) : 
  (4 * a + 2 * b = 92) ∧ (6 * a - 4 * b = 60) → 
  (a = 122 / 7) ∧ (b = 78 / 7) := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l1009_100984


namespace NUMINAMATH_CALUDE_bugs_eat_seventeen_flowers_l1009_100979

/-- Represents the number of flowers eaten by each type of bug -/
structure BugEating where
  typeA : Nat
  typeB : Nat
  typeC : Nat

/-- Represents the number of bugs of each type -/
structure BugCount where
  typeA : Nat
  typeB : Nat
  typeC : Nat

/-- Calculates the total number of flowers eaten by all bugs -/
def totalFlowersEaten (eating : BugEating) (count : BugCount) : Nat :=
  eating.typeA * count.typeA + eating.typeB * count.typeB + eating.typeC * count.typeC

theorem bugs_eat_seventeen_flowers : 
  let eating : BugEating := { typeA := 2, typeB := 3, typeC := 5 }
  let count : BugCount := { typeA := 3, typeB := 2, typeC := 1 }
  totalFlowersEaten eating count = 17 := by
  sorry

end NUMINAMATH_CALUDE_bugs_eat_seventeen_flowers_l1009_100979


namespace NUMINAMATH_CALUDE_cos_4alpha_minus_9pi_over_2_l1009_100975

theorem cos_4alpha_minus_9pi_over_2 (α : ℝ) : 
  4.53 * (1 + Real.cos (2 * α - 2 * Real.pi) + Real.cos (4 * α + 2 * Real.pi) - Real.cos (6 * α - Real.pi)) / 
  (Real.cos (2 * Real.pi - 2 * α) + 2 * (Real.cos (2 * α + Real.pi))^2 - 1) = 2 * Real.cos (2 * α) →
  Real.cos (4 * α - 9 * Real.pi / 2) = Real.cos (4 * α - Real.pi / 2) := by
sorry

end NUMINAMATH_CALUDE_cos_4alpha_minus_9pi_over_2_l1009_100975


namespace NUMINAMATH_CALUDE_class_average_weight_l1009_100946

theorem class_average_weight (students_A students_B : ℕ) (avg_weight_A avg_weight_B : ℝ) :
  students_A = 36 →
  students_B = 44 →
  avg_weight_A = 40 →
  avg_weight_B = 35 →
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B : ℝ) = 37.25 := by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l1009_100946


namespace NUMINAMATH_CALUDE_four_digit_perfect_square_same_digits_l1009_100918

theorem four_digit_perfect_square_same_digits : ∃ n : ℕ,
  (1000 ≤ n) ∧ (n < 10000) ∧  -- four-digit number
  (∃ m : ℕ, n = m^2) ∧  -- perfect square
  (∃ a b : ℕ, n = 1100 * a + 11 * b) ∧  -- first two digits same, last two digits same
  (n = 7744) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_perfect_square_same_digits_l1009_100918


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1009_100928

theorem smallest_positive_integer_with_remainders : ∃ (x : ℕ), 
  (x > 0) ∧
  (x % 5 = 4) ∧ 
  (x % 7 = 6) ∧ 
  (x % 9 = 8) ∧
  (∀ y : ℕ, y > 0 → y % 5 = 4 → y % 7 = 6 → y % 9 = 8 → x ≤ y) ∧
  (x = 314) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1009_100928


namespace NUMINAMATH_CALUDE_base_10_to_base_2_l1009_100940

theorem base_10_to_base_2 (n : Nat) (h : n = 123) :
  ∃ (a b c d e f g : Nat),
    a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 0 ∧ f = 1 ∧ g = 1 ∧
    n = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_2_l1009_100940


namespace NUMINAMATH_CALUDE_games_planned_this_month_l1009_100952

theorem games_planned_this_month
  (total_attended : ℕ)
  (planned_last_month : ℕ)
  (missed : ℕ)
  (h1 : total_attended = 12)
  (h2 : planned_last_month = 17)
  (h3 : missed = 16)
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_games_planned_this_month_l1009_100952


namespace NUMINAMATH_CALUDE_inequality_proof_l1009_100907

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : x - Real.sqrt x ≤ y - 1/4 ∧ y - 1/4 ≤ x + Real.sqrt x) :
  y - Real.sqrt y ≤ x - 1/4 ∧ x - 1/4 ≤ y + Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1009_100907


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l1009_100944

/-- A point on a parabola with a specific distance to its focus -/
def PointOnParabola (x y : ℝ) : Prop :=
  x^2 = 4*y ∧ (x - 0)^2 + (y - 1/4)^2 = 10^2

/-- The coordinates of the point satisfy the given conditions -/
theorem parabola_point_coordinates :
  ∀ x y : ℝ, PointOnParabola x y → (x = 6 ∨ x = -6) ∧ y = 9 :=
by sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l1009_100944


namespace NUMINAMATH_CALUDE_hayley_meatballs_l1009_100934

/-- The number of meatballs Hayley has left after Kirsten stole some -/
def meatballs_left (initial : ℕ) (stolen : ℕ) : ℕ :=
  initial - stolen

/-- Theorem stating that Hayley has 11 meatballs left -/
theorem hayley_meatballs : meatballs_left 25 14 = 11 := by
  sorry

end NUMINAMATH_CALUDE_hayley_meatballs_l1009_100934


namespace NUMINAMATH_CALUDE_no_2000_digit_square_with_1999_fives_l1009_100978

theorem no_2000_digit_square_with_1999_fives : 
  ¬ ∃ n : ℕ, 
    (10^1999 ≤ n) ∧ (n < 10^2000) ∧  -- 2000-digit integer
    (∃ k : ℕ, n = k^2) ∧              -- perfect square
    (∃ d : ℕ, d < 10 ∧                -- at least 1999 digits of "5"
      (n / 10 = 5 * (10^1998 - 1) / 9 + d * 10^1998 ∨
       n % 10 ≠ 5 ∧ n / 10 = 5 * (10^1999 - 1) / 9)) :=
by sorry

end NUMINAMATH_CALUDE_no_2000_digit_square_with_1999_fives_l1009_100978


namespace NUMINAMATH_CALUDE_probability_select_four_or_five_l1009_100911

/-- The probability of selecting a product with a number not less than 4 from 5 products -/
theorem probability_select_four_or_five (n : ℕ) (h : n = 5) :
  (Finset.filter (λ i => i ≥ 4) (Finset.range n)).card / n = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_select_four_or_five_l1009_100911


namespace NUMINAMATH_CALUDE_barney_towel_shortage_l1009_100995

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents Barney's towel situation -/
structure TowelSituation where
  totalTowels : ℕ
  towelsPerDay : ℕ
  extraTowelsUsed : ℕ
  expectedGuests : ℕ

/-- Calculates the number of days without clean towels -/
def daysWithoutCleanTowels (s : TowelSituation) : ℕ :=
  daysInWeek

/-- Theorem stating that Barney will not have clean towels for 7 days -/
theorem barney_towel_shortage (s : TowelSituation)
  (h1 : s.totalTowels = 18)
  (h2 : s.towelsPerDay = 2)
  (h3 : s.extraTowelsUsed = 5)
  (h4 : s.expectedGuests = 3) :
  daysWithoutCleanTowels s = daysInWeek :=
by sorry

#check barney_towel_shortage

end NUMINAMATH_CALUDE_barney_towel_shortage_l1009_100995


namespace NUMINAMATH_CALUDE_tax_calculation_l1009_100901

/-- Calculates the tax paid given gross pay and net pay -/
def tax_paid (gross_pay : ℕ) (net_pay : ℕ) : ℕ :=
  gross_pay - net_pay

/-- Theorem stating that the tax paid is 135 dollars given the conditions -/
theorem tax_calculation (gross_pay net_pay : ℕ) 
  (h1 : gross_pay = 450)
  (h2 : net_pay = 315)
  (h3 : tax_paid gross_pay net_pay = gross_pay - net_pay) :
  tax_paid gross_pay net_pay = 135 := by
  sorry

end NUMINAMATH_CALUDE_tax_calculation_l1009_100901


namespace NUMINAMATH_CALUDE_zero_unique_for_multiplication_and_division_l1009_100956

theorem zero_unique_for_multiplication_and_division :
  ∀ x : ℝ, (∀ a : ℝ, x * a = x ∧ (a ≠ 0 → x / a = x)) → x = 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_unique_for_multiplication_and_division_l1009_100956


namespace NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l1009_100999

theorem tan_45_degrees_equals_one :
  let θ : Real := 45 * π / 180  -- Convert 45 degrees to radians
  let tan_θ := Real.tan θ
  let sin_θ := Real.sin θ
  let cos_θ := Real.cos θ
  (∀ α, Real.tan α = Real.sin α / Real.cos α) →  -- General tangent identity
  sin_θ = Real.sqrt 2 / 2 →  -- Given value of sin 45°
  cos_θ = Real.sqrt 2 / 2 →  -- Given value of cos 45°
  tan_θ = 1 := by
sorry

end NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l1009_100999


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l1009_100970

/-- A trapezoid with specific side lengths -/
structure Trapezoid where
  EG : ℝ
  FH : ℝ
  GH : ℝ
  EF : ℝ
  is_trapezoid : EF > GH
  parallel_bases : EF = 2 * GH

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.EG + t.FH + t.GH + t.EF

/-- Theorem: The perimeter of the given trapezoid is 183 units -/
theorem trapezoid_perimeter :
  ∃ t : Trapezoid, t.EG = 35 ∧ t.FH = 40 ∧ t.GH = 36 ∧ perimeter t = 183 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l1009_100970


namespace NUMINAMATH_CALUDE_inequality_of_positive_reals_l1009_100951

theorem inequality_of_positive_reals (a b c d e f : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f) :
  (a * b) / (a + b) + (c * d) / (c + d) + (e * f) / (e + f) ≤ 
  ((a + c + e) * (b + d + f)) / (a + b + c + d + e + f) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_positive_reals_l1009_100951


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1009_100991

-- Define the property that f must satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = (x + y) * (f x - f y)

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1009_100991


namespace NUMINAMATH_CALUDE_sum_of_squares_of_cubic_roots_l1009_100976

/-- Given a cubic equation dx³ - ex² + fx - g = 0 with real coefficients,
    the sum of squares of its roots is (e² - 2df) / d². -/
theorem sum_of_squares_of_cubic_roots
  (d e f g : ℝ) (a b c : ℝ)
  (hroots : d * (X - a) * (X - b) * (X - c) = d * X^3 - e * X^2 + f * X - g) :
  a^2 + b^2 + c^2 = (e^2 - 2*d*f) / d^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_cubic_roots_l1009_100976


namespace NUMINAMATH_CALUDE_skew_iff_b_neq_4_l1009_100927

def line1 (b t : ℝ) : ℝ × ℝ × ℝ := (2 + 3*t, 3 + 4*t, b + 5*t)
def line2 (u : ℝ) : ℝ × ℝ × ℝ := (5 + 6*u, 2 + 3*u, 1 + 2*u)

def are_skew (b : ℝ) : Prop :=
  ∀ t u : ℝ, line1 b t ≠ line2 u

theorem skew_iff_b_neq_4 (b : ℝ) :
  are_skew b ↔ b ≠ 4 :=
sorry

end NUMINAMATH_CALUDE_skew_iff_b_neq_4_l1009_100927


namespace NUMINAMATH_CALUDE_sum_of_zeros_greater_than_one_l1009_100981

noncomputable def g (x m : ℝ) : ℝ := Real.log x - x + x + 1 / (2 * x) - m

theorem sum_of_zeros_greater_than_one (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : x₁ < x₂) 
  (hz₁ : g x₁ m = 0) (hz₂ : g x₂ m = 0) : 
  x₁ + x₂ > 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_zeros_greater_than_one_l1009_100981


namespace NUMINAMATH_CALUDE_consecutive_natural_numbers_sum_l1009_100972

theorem consecutive_natural_numbers_sum (a : ℕ) : 
  (∃ (x : ℕ), x + (x + 1) + (x + 2) + (x + 3) + (x + 4) = 50) → 
  (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) = 50) → 
  (a + 2 = 10) := by
  sorry

#check consecutive_natural_numbers_sum

end NUMINAMATH_CALUDE_consecutive_natural_numbers_sum_l1009_100972


namespace NUMINAMATH_CALUDE_isosceles_triangle_l1009_100949

/-- Given a triangle ABC where sin A = 2 sin C cos B, prove that B = C -/
theorem isosceles_triangle (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_sin : Real.sin A = 2 * Real.sin C * Real.cos B) : B = C := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l1009_100949


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1009_100948

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y = x * y) :
  x + y ≥ 9 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 4 * x₀ + y₀ = x₀ * y₀ ∧ x₀ + y₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1009_100948


namespace NUMINAMATH_CALUDE_wickets_before_last_match_value_l1009_100922

/-- The number of wickets taken by a bowler before his last match -/
def wickets_before_last_match (initial_average : ℚ) (wickets_last_match : ℕ) 
  (runs_last_match : ℕ) (average_decrease : ℚ) : ℕ :=
  sorry

/-- Theorem stating the number of wickets taken before the last match -/
theorem wickets_before_last_match_value :
  wickets_before_last_match 12.4 3 26 0.4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_wickets_before_last_match_value_l1009_100922


namespace NUMINAMATH_CALUDE_william_has_45_napkins_l1009_100967

/-- The number of napkins William has now -/
def williams_napkins (original : ℕ) (from_olivia : ℕ) (amelia_multiplier : ℕ) : ℕ :=
  original + from_olivia + amelia_multiplier * from_olivia

/-- Proof that William has 45 napkins given the conditions -/
theorem william_has_45_napkins :
  williams_napkins 15 10 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_william_has_45_napkins_l1009_100967


namespace NUMINAMATH_CALUDE_village_foods_monthly_sales_l1009_100998

/-- Represents the monthly sales data for Village Foods --/
structure VillageFoodsSales where
  customers : ℕ
  lettucePerCustomer : ℕ
  lettucePrice : ℚ
  tomatoesPerCustomer : ℕ
  tomatoPrice : ℚ

/-- Calculates the total monthly sales from lettuce and tomatoes --/
def totalMonthlySales (s : VillageFoodsSales) : ℚ :=
  s.customers * (s.lettucePerCustomer * s.lettucePrice + s.tomatoesPerCustomer * s.tomatoPrice)

/-- Theorem stating that the total monthly sales for the given conditions is $2000 --/
theorem village_foods_monthly_sales :
  let sales := VillageFoodsSales.mk 500 2 1 4 (1/2)
  totalMonthlySales sales = 2000 := by sorry

end NUMINAMATH_CALUDE_village_foods_monthly_sales_l1009_100998


namespace NUMINAMATH_CALUDE_line_circle_intersection_a_values_l1009_100974

/-- A line intersecting a circle -/
structure LineCircleIntersection where
  /-- The parameter of the line equation 4x + 3y + a = 0 -/
  a : ℝ
  /-- The line 4x + 3y + a = 0 intersects the circle (x-1)^2 + (y-2)^2 = 9 -/
  intersects : ∃ (x y : ℝ), 4*x + 3*y + a = 0 ∧ (x-1)^2 + (y-2)^2 = 9
  /-- The distance between intersection points is 4√2 -/
  chord_length : ∃ (A B : ℝ × ℝ), 
    (4*(A.1) + 3*(A.2) + a = 0) ∧ ((A.1-1)^2 + (A.2-2)^2 = 9) ∧
    (4*(B.1) + 3*(B.2) + a = 0) ∧ ((B.1-1)^2 + (B.2-2)^2 = 9) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 32

/-- The theorem stating the possible values of a -/
theorem line_circle_intersection_a_values (lci : LineCircleIntersection) :
  lci.a = -5 ∨ lci.a = -15 := by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_a_values_l1009_100974


namespace NUMINAMATH_CALUDE_contractor_payment_l1009_100945

/-- A contractor's payment problem -/
theorem contractor_payment
  (total_days : ℕ)
  (payment_per_day : ℚ)
  (fine_per_day : ℚ)
  (absent_days : ℕ)
  (h1 : total_days = 30)
  (h2 : payment_per_day = 25)
  (h3 : fine_per_day = 7.5)
  (h4 : absent_days = 10)
  (h5 : absent_days ≤ total_days) :
  (total_days - absent_days) * payment_per_day - absent_days * fine_per_day = 425 :=
by sorry

end NUMINAMATH_CALUDE_contractor_payment_l1009_100945


namespace NUMINAMATH_CALUDE_sin_negative_1560_degrees_l1009_100994

theorem sin_negative_1560_degrees : Real.sin ((-1560 : ℝ) * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_1560_degrees_l1009_100994


namespace NUMINAMATH_CALUDE_person_B_lap_time_l1009_100913

/-- The time it takes for person B to complete a lap on a circular track -/
def time_B_lap : ℝ :=
  let time_A_lap : ℝ := 80  -- 1 minute and 20 seconds in seconds
  let meeting_interval : ℝ := 30
  48  -- The time we want to prove

theorem person_B_lap_time :
  let time_A_lap : ℝ := 80  -- 1 minute and 20 seconds in seconds
  let meeting_interval : ℝ := 30
  (1 / time_B_lap + 1 / time_A_lap) * meeting_interval = 1 ∧
  time_B_lap > 0 :=
by sorry

end NUMINAMATH_CALUDE_person_B_lap_time_l1009_100913


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_two_l1009_100921

/-- Two lines in the plane -/
structure TwoLines where
  line1 : ℝ → ℝ → ℝ  -- represents ax + 2y = 0
  line2 : ℝ → ℝ → ℝ  -- represents x + y = 1

/-- The condition for parallelism -/
def parallel (l : TwoLines) (a : ℝ) : Prop :=
  ∀ x y, l.line1 x y = 0 ∧ l.line2 x y = 1 → 
    ∃ k : ℝ, k ≠ 0 ∧ (a = k ∧ 2 = k)

/-- The theorem stating that a=2 is necessary and sufficient for parallelism -/
theorem parallel_iff_a_eq_two (l : TwoLines) : 
  (∀ a, parallel l a ↔ a = 2) := by sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_two_l1009_100921
