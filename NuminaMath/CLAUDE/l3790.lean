import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3790_379030

theorem quadratic_one_solution (n : ℝ) : 
  (n > 0 ∧ ∃! x : ℝ, 4 * x^2 + n * x + 4 = 0) ↔ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3790_379030


namespace NUMINAMATH_CALUDE_symmetric_increasing_function_property_l3790_379057

/-- A function that is increasing on (-∞, 2) and its graph shifted by 2 is symmetric about x=0 -/
def symmetric_increasing_function (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y ∧ y < 2 → f x < f y) ∧
  (∀ x, f (x + 2) = f (2 - x))

/-- If f is a symmetric increasing function, then f(0) < f(3) -/
theorem symmetric_increasing_function_property (f : ℝ → ℝ) 
  (h : symmetric_increasing_function f) : f 0 < f 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_increasing_function_property_l3790_379057


namespace NUMINAMATH_CALUDE_min_value_expression_l3790_379054

theorem min_value_expression (a : ℝ) (h : a > 0) :
  (a - 1) * (4 * a - 1) / a ≥ -1 ∧
  ∃ a₀ > 0, (a₀ - 1) * (4 * a₀ - 1) / a₀ = -1 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3790_379054


namespace NUMINAMATH_CALUDE_wade_average_points_l3790_379063

/-- Represents a basketball team with Wade and his teammates -/
structure BasketballTeam where
  wade_avg : ℝ
  teammates_avg : ℝ
  total_points : ℝ
  num_games : ℝ

/-- Theorem stating Wade's average points per game -/
theorem wade_average_points (team : BasketballTeam)
  (h1 : team.teammates_avg = 40)
  (h2 : team.total_points = 300)
  (h3 : team.num_games = 5) :
  team.wade_avg = 20 := by
  sorry

#check wade_average_points

end NUMINAMATH_CALUDE_wade_average_points_l3790_379063


namespace NUMINAMATH_CALUDE_fraction_equals_93_l3790_379045

theorem fraction_equals_93 : (3025 - 2880)^2 / 225 = 93 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_93_l3790_379045


namespace NUMINAMATH_CALUDE_EL_length_l3790_379097

-- Define the rectangle
def rectangle_EFGH : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define points E and K
def E : ℝ × ℝ := (0, 1)
def K : ℝ × ℝ := (1, 0)

-- Define the inscribed circle ω
def ω : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + (p.2 - 0.5)^2 = 0.25}

-- Define the line EK
def line_EK (x : ℝ) : ℝ := -x + 1

-- Define point L as the intersection of EK and ω (different from K)
def L : ℝ × ℝ :=
  let x := 0.5
  (x, line_EK x)

-- Theorem statement
theorem EL_length :
  let el_length := Real.sqrt ((L.1 - E.1)^2 + (L.2 - E.2)^2)
  el_length = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_EL_length_l3790_379097


namespace NUMINAMATH_CALUDE_cattle_milk_production_l3790_379056

/-- Given a herd of dairy cows, calculates the daily milk production per cow -/
def daily_milk_per_cow (num_cows : ℕ) (weekly_milk : ℕ) : ℚ :=
  (weekly_milk : ℚ) / 7 / num_cows

theorem cattle_milk_production :
  daily_milk_per_cow 52 364000 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cattle_milk_production_l3790_379056


namespace NUMINAMATH_CALUDE_max_radius_circle_x_value_l3790_379014

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the theorem
theorem max_radius_circle_x_value 
  (C : ℝ × ℝ → ℝ → Set (ℝ × ℝ)) 
  (max_radius : ℝ) 
  (x : ℝ) :
  (∀ r : ℝ, r ≤ max_radius) →
  ((8, 0) ∈ C (0, 0) max_radius) →
  ((x, 0) ∈ C (0, 0) max_radius) →
  (max_radius = 8) →
  (x = -8) :=
by sorry

end NUMINAMATH_CALUDE_max_radius_circle_x_value_l3790_379014


namespace NUMINAMATH_CALUDE_prime_power_sum_implies_power_of_three_l3790_379006

theorem prime_power_sum_implies_power_of_three (n : ℕ) :
  Nat.Prime (1 + 2^n + 4^n) → ∃ k : ℕ+, n = 3^(k : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_implies_power_of_three_l3790_379006


namespace NUMINAMATH_CALUDE_charles_milk_amount_l3790_379009

/-- The amount of chocolate milk in each glass (in ounces) -/
def glass_size : ℝ := 8

/-- The amount of milk in each glass (in ounces) -/
def milk_per_glass : ℝ := 6.5

/-- The amount of chocolate syrup in each glass (in ounces) -/
def syrup_per_glass : ℝ := 1.5

/-- The total amount of chocolate syrup Charles has (in ounces) -/
def total_syrup : ℝ := 60

/-- The total amount of chocolate milk Charles will drink (in ounces) -/
def total_milk : ℝ := 160

/-- Theorem stating that Charles has 130 ounces of milk -/
theorem charles_milk_amount : 
  ∃ (num_glasses : ℝ),
    num_glasses * glass_size = total_milk ∧
    num_glasses * syrup_per_glass ≤ total_syrup ∧
    num_glasses * milk_per_glass = 130 := by
  sorry

end NUMINAMATH_CALUDE_charles_milk_amount_l3790_379009


namespace NUMINAMATH_CALUDE_distance_on_line_l3790_379080

/-- The distance between two points (5, b) and (10, d) on the line y = 2x + 3 is 5√5. -/
theorem distance_on_line : ∀ b d : ℝ,
  b = 2 * 5 + 3 →
  d = 2 * 10 + 3 →
  Real.sqrt ((10 - 5)^2 + (d - b)^2) = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_on_line_l3790_379080


namespace NUMINAMATH_CALUDE_cats_in_center_l3790_379064

/-- The number of cats that can jump -/
def jump : ℕ := 60

/-- The number of cats that can fetch -/
def fetch : ℕ := 35

/-- The number of cats that can spin -/
def spin : ℕ := 40

/-- The number of cats that can jump and fetch -/
def jump_fetch : ℕ := 20

/-- The number of cats that can fetch and spin -/
def fetch_spin : ℕ := 15

/-- The number of cats that can jump and spin -/
def jump_spin : ℕ := 25

/-- The number of cats that can do all three tricks -/
def all_three : ℕ := 10

/-- The number of cats that can do no tricks -/
def no_tricks : ℕ := 8

/-- The total number of cats in the center -/
def total_cats : ℕ := 93

theorem cats_in_center : 
  jump + fetch + spin - jump_fetch - fetch_spin - jump_spin + all_three + no_tricks = total_cats :=
by sorry

end NUMINAMATH_CALUDE_cats_in_center_l3790_379064


namespace NUMINAMATH_CALUDE_niles_collection_l3790_379019

/-- The total amount collected by Niles from the book club members -/
def total_collected (num_members : ℕ) (snack_fee : ℕ) (num_hardcover : ℕ) (hardcover_price : ℕ) (num_paperback : ℕ) (paperback_price : ℕ) : ℕ :=
  num_members * (snack_fee + num_hardcover * hardcover_price + num_paperback * paperback_price)

/-- Theorem stating the total amount collected by Niles -/
theorem niles_collection : 
  total_collected 6 150 6 30 6 12 = 2412 := by
  sorry

end NUMINAMATH_CALUDE_niles_collection_l3790_379019


namespace NUMINAMATH_CALUDE_cake_frosting_time_difference_l3790_379061

/-- The time difference in frosting cakes with normal and sprained conditions -/
theorem cake_frosting_time_difference 
  (normal_time : ℕ) -- Time to frost one cake under normal conditions
  (sprained_time : ℕ) -- Time to frost one cake with sprained wrist
  (num_cakes : ℕ) -- Number of cakes to frost
  (h1 : normal_time = 5) -- Normal frosting time is 5 minutes
  (h2 : sprained_time = 8) -- Sprained wrist frosting time is 8 minutes
  (h3 : num_cakes = 10) -- Number of cakes to frost is 10
  : sprained_time * num_cakes - normal_time * num_cakes = 30 := by
  sorry

end NUMINAMATH_CALUDE_cake_frosting_time_difference_l3790_379061


namespace NUMINAMATH_CALUDE_train_crossing_bridge_time_l3790_379051

/-- Proves the time taken for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 165) 
  (h2 : train_speed_kmph = 72) 
  (h3 : bridge_length = 660) : 
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 41.25 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_bridge_time_l3790_379051


namespace NUMINAMATH_CALUDE_circles_are_tangent_l3790_379005

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 16*y - 48 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 44 = 0

-- Define what it means for two circles to be tangent
def are_tangent (c1 c2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), c1 x y ∧ c2 x y ∧ 
  ∀ (x' y' : ℝ), (x' ≠ x ∨ y' ≠ y) → ¬(c1 x' y' ∧ c2 x' y')

-- Theorem statement
theorem circles_are_tangent : are_tangent circle1 circle2 := by
  sorry

end NUMINAMATH_CALUDE_circles_are_tangent_l3790_379005


namespace NUMINAMATH_CALUDE_lambda_positive_infinite_lambda_negative_infinite_l3790_379015

/-- Definition of Ω(n) -/
def Omega (n : ℕ) : ℕ := sorry

/-- Definition of λ(n) -/
def lambda (n : ℕ) : Int := (-1) ^ (Omega n)

/-- The set of positive integers n such that λ(n) = λ(n+1) = 1 is infinite -/
theorem lambda_positive_infinite : Set.Infinite {n : ℕ | lambda n = 1 ∧ lambda (n + 1) = 1} := by sorry

/-- The set of positive integers n such that λ(n) = λ(n+1) = -1 is infinite -/
theorem lambda_negative_infinite : Set.Infinite {n : ℕ | lambda n = -1 ∧ lambda (n + 1) = -1} := by sorry

end NUMINAMATH_CALUDE_lambda_positive_infinite_lambda_negative_infinite_l3790_379015


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3790_379089

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3790_379089


namespace NUMINAMATH_CALUDE_carpet_coverage_theorem_l3790_379043

/-- Represents the problem of covering a corridor with carpets --/
structure CarpetProblem where
  totalCarpetLength : ℕ
  numCarpets : ℕ
  corridorLength : ℕ

/-- Calculates the maximum number of uncovered sections in a carpet problem --/
def maxUncoveredSections (problem : CarpetProblem) : ℕ :=
  sorry

/-- Theorem stating that for the given problem, the maximum number of uncovered sections is 11 --/
theorem carpet_coverage_theorem (problem : CarpetProblem) 
  (h1 : problem.totalCarpetLength = 1000)
  (h2 : problem.numCarpets = 20)
  (h3 : problem.corridorLength = 100) :
  maxUncoveredSections problem = 11 :=
sorry

end NUMINAMATH_CALUDE_carpet_coverage_theorem_l3790_379043


namespace NUMINAMATH_CALUDE_product_of_numbers_l3790_379055

theorem product_of_numbers (x₁ x₂ : ℝ) 
  (h1 : x₁ + x₂ = 2 * Real.sqrt 1703)
  (h2 : |x₁ - x₂| = 90) : 
  x₁ * x₂ = -322 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3790_379055


namespace NUMINAMATH_CALUDE_least_intersection_size_l3790_379087

theorem least_intersection_size (total students_with_glasses students_with_pets : ℕ) 
  (h_total : total = 35)
  (h_glasses : students_with_glasses = 18)
  (h_pets : students_with_pets = 25) :
  (students_with_glasses + students_with_pets - total : ℤ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_least_intersection_size_l3790_379087


namespace NUMINAMATH_CALUDE_max_triangles_three_families_ten_lines_l3790_379053

/-- Represents a family of parallel lines -/
structure ParallelLineFamily :=
  (num_lines : ℕ)

/-- Represents the configuration of three families of parallel lines -/
structure ThreeParallelLineFamilies :=
  (family1 : ParallelLineFamily)
  (family2 : ParallelLineFamily)
  (family3 : ParallelLineFamily)

/-- Calculates the maximum number of triangles formed by three families of parallel lines -/
def max_triangles (config : ThreeParallelLineFamilies) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of triangles formed by three families of 10 parallel lines is 150 -/
theorem max_triangles_three_families_ten_lines :
  ∀ (config : ThreeParallelLineFamilies),
    config.family1.num_lines = 10 →
    config.family2.num_lines = 10 →
    config.family3.num_lines = 10 →
    max_triangles config = 150 :=
  sorry

end NUMINAMATH_CALUDE_max_triangles_three_families_ten_lines_l3790_379053


namespace NUMINAMATH_CALUDE_additive_multiplicative_inverse_sum_l3790_379058

theorem additive_multiplicative_inverse_sum (a b : ℝ) : 
  (a + a = 0) → (b * b = 1) → (a + b = 1 ∨ a + b = -1) := by sorry

end NUMINAMATH_CALUDE_additive_multiplicative_inverse_sum_l3790_379058


namespace NUMINAMATH_CALUDE_apple_distribution_theorem_l3790_379042

def distribute_apples (total_apples : ℕ) (num_people : ℕ) (min_apples : ℕ) : ℕ :=
  Nat.choose (total_apples - num_people * min_apples + num_people - 1) (num_people - 1)

theorem apple_distribution_theorem :
  distribute_apples 30 3 3 = 253 :=
sorry

end NUMINAMATH_CALUDE_apple_distribution_theorem_l3790_379042


namespace NUMINAMATH_CALUDE_knowledge_competition_probability_l3790_379025

/-- Represents the probability of correctly answering a single question -/
def p : ℝ := 0.8

/-- Represents the number of preset questions -/
def n : ℕ := 5

/-- Represents the probability of answering exactly 4 questions before advancing -/
def prob_4_questions : ℝ := 2 * p^3 * (1 - p)

theorem knowledge_competition_probability : prob_4_questions = 0.128 := by
  sorry


end NUMINAMATH_CALUDE_knowledge_competition_probability_l3790_379025


namespace NUMINAMATH_CALUDE_inequality_proof_l3790_379000

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3790_379000


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3790_379031

theorem sin_alpha_value (α : Real) : 
  α ∈ Set.Ioo (π) (3*π/2) →  -- α is in the third quadrant
  Real.tan (α + π/4) = 3 → 
  Real.sin α = -Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3790_379031


namespace NUMINAMATH_CALUDE_tennis_balls_count_l3790_379026

theorem tennis_balls_count (baskets : ℕ) (soccer_balls : ℕ) (students_8 : ℕ) (students_10 : ℕ) 
  (balls_removed_8 : ℕ) (balls_removed_10 : ℕ) (balls_remaining : ℕ) :
  baskets = 5 →
  soccer_balls = 5 →
  students_8 = 3 →
  students_10 = 2 →
  balls_removed_8 = 8 →
  balls_removed_10 = 10 →
  balls_remaining = 56 →
  ∃ T : ℕ, 
    baskets * (T + soccer_balls) - (students_8 * balls_removed_8 + students_10 * balls_removed_10) = balls_remaining ∧
    T = 15 :=
by sorry

end NUMINAMATH_CALUDE_tennis_balls_count_l3790_379026


namespace NUMINAMATH_CALUDE_bakers_new_cakes_l3790_379018

/-- Baker's cake problem -/
theorem bakers_new_cakes 
  (initial_cakes : ℕ) 
  (sold_cakes : ℕ) 
  (difference : ℕ) 
  (h1 : initial_cakes = 13) 
  (h2 : sold_cakes = 91) 
  (h3 : difference = 63) : 
  sold_cakes + difference = 154 := by
  sorry

end NUMINAMATH_CALUDE_bakers_new_cakes_l3790_379018


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l3790_379034

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  totalStudents : Nat
  sampleSize : Nat
  startingNumber : Nat

/-- Generates the sequence of selected student numbers. -/
def generateSequence (s : SystematicSampling) : List Nat :=
  List.range s.sampleSize |>.map (fun i => s.startingNumber + i * (s.totalStudents / s.sampleSize))

/-- Theorem stating the properties of systematic sampling for the given problem. -/
theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.totalStudents = 50)
  (h2 : s.sampleSize = 5)
  (h3 : 1 ≤ s.startingNumber)
  (h4 : s.startingNumber ≤ 10) :
  ∃ (a : Nat), 1 ≤ a ∧ a ≤ 10 ∧ 
  generateSequence s = [a, a + 10, a + 20, a + 30, a + 40] :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l3790_379034


namespace NUMINAMATH_CALUDE_julies_savings_l3790_379066

theorem julies_savings (monthly_salary : ℝ) (savings_fraction : ℝ) : 
  monthly_salary > 0 →
  savings_fraction > 0 →
  savings_fraction < 1 →
  12 * monthly_salary * savings_fraction = 4 * monthly_salary * (1 - savings_fraction) →
  1 - savings_fraction = 3/4 := by
sorry

end NUMINAMATH_CALUDE_julies_savings_l3790_379066


namespace NUMINAMATH_CALUDE_f_five_zeros_a_range_l3790_379068

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then
    Real.log x ^ 2 - floor (Real.log x) - 2
  else if x ≤ 0 then
    Real.exp (-x) - a * x - 1
  else
    0  -- This case is not specified in the original problem, so we set it to 0

-- State the theorem
theorem f_five_zeros_a_range (a : ℝ) :
  (∃ (s : Finset ℝ), s.card = 5 ∧ ∀ x ∈ s, f a x = 0) →
  a ∈ Set.Iic (-1 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_f_five_zeros_a_range_l3790_379068


namespace NUMINAMATH_CALUDE_projectile_max_height_l3790_379048

/-- The height function of the projectile -/
def f (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- Theorem stating that the maximum height of the projectile is 161 meters -/
theorem projectile_max_height :
  ∃ (t : ℝ), ∀ (s : ℝ), f s ≤ f t ∧ f t = 161 := by
  sorry

end NUMINAMATH_CALUDE_projectile_max_height_l3790_379048


namespace NUMINAMATH_CALUDE_alligators_not_hiding_l3790_379024

theorem alligators_not_hiding (total_alligators hiding_alligators : ℕ) 
  (h1 : total_alligators = 75)
  (h2 : hiding_alligators = 19) : 
  total_alligators - hiding_alligators = 56 := by
  sorry

end NUMINAMATH_CALUDE_alligators_not_hiding_l3790_379024


namespace NUMINAMATH_CALUDE_distance_p_ran_l3790_379012

/-- A race between two runners p and q, where p is faster but q gets a head start. -/
structure Race where
  /-- The speed of runner q in meters per minute -/
  v : ℝ
  /-- The time of the race in minutes -/
  t : ℝ
  /-- The head start distance given to runner q in meters -/
  d : ℝ
  /-- Assumption that the speeds and time are positive -/
  hv : v > 0
  ht : t > 0
  /-- Assumption that p runs 30% faster than q -/
  hp_speed : ℝ := 1.3 * v
  /-- Assumption that the race ends in a tie -/
  h_tie : d + v * t = hp_speed * t

/-- The theorem stating the distance p ran in the race -/
theorem distance_p_ran (race : Race) : ℝ := by
  sorry

end NUMINAMATH_CALUDE_distance_p_ran_l3790_379012


namespace NUMINAMATH_CALUDE_charlie_area_is_72_l3790_379020

-- Define the total area to be painted
def total_area : ℝ := 360

-- Define the ratio of work done by each person
def allen_ratio : ℝ := 3
def ben_ratio : ℝ := 5
def charlie_ratio : ℝ := 2

-- Define the total ratio
def total_ratio : ℝ := allen_ratio + ben_ratio + charlie_ratio

-- Theorem to prove
theorem charlie_area_is_72 : 
  charlie_ratio / total_ratio * total_area = 72 := by
  sorry

end NUMINAMATH_CALUDE_charlie_area_is_72_l3790_379020


namespace NUMINAMATH_CALUDE_complex_equality_l3790_379059

theorem complex_equality (z : ℂ) : z = -1.5 - (1/6)*I →
  Complex.abs (z - 2) = Complex.abs (z + 4) ∧ 
  Complex.abs (z - 2) = Complex.abs (z - 3*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l3790_379059


namespace NUMINAMATH_CALUDE_sixty_first_term_is_201_l3790_379039

/-- An arithmetic sequence with a_5 = 33 and common difference d = 3 -/
def arithmetic_sequence (n : ℕ) : ℝ :=
  33 + 3 * (n - 5)

/-- Theorem: The 61st term of the sequence is 201 -/
theorem sixty_first_term_is_201 : arithmetic_sequence 61 = 201 := by
  sorry

end NUMINAMATH_CALUDE_sixty_first_term_is_201_l3790_379039


namespace NUMINAMATH_CALUDE_coin_toss_probability_l3790_379021

theorem coin_toss_probability : 
  let n : ℕ := 5  -- Total number of coins
  let k : ℕ := 3  -- Number of tails (or heads, whichever is smaller)
  let p : ℚ := 1/2  -- Probability of getting tails (or heads) on a single toss
  (n.choose k) * p^n = 5/16 :=
sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l3790_379021


namespace NUMINAMATH_CALUDE_equation_solutions_range_l3790_379077

-- Define the equation
def equation (x a : ℝ) : Prop := |2^x - a| = 1

-- Define the condition of having two unequal real solutions
def has_two_unequal_solutions (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ equation x a ∧ equation y a

-- State the theorem
theorem equation_solutions_range :
  ∀ a : ℝ, has_two_unequal_solutions a ↔ a > 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_range_l3790_379077


namespace NUMINAMATH_CALUDE_vertex_landing_probability_l3790_379079

/-- Square vertices -/
def square_vertices : List (Int × Int) := [(2, 2), (-2, 2), (-2, -2), (2, -2)]

/-- All boundary points of the square -/
def boundary_points : List (Int × Int) := [
  (2, 2), (-2, 2), (-2, -2), (2, -2),  -- vertices
  (1, 2), (0, 2), (-1, 2),             -- top edge
  (1, -2), (0, -2), (-1, -2),          -- bottom edge
  (2, 1), (2, 0), (2, -1),             -- right edge
  (-2, 1), (-2, 0), (-2, -1)           -- left edge
]

/-- Neighboring points function -/
def neighbors (x y : Int) : List (Int × Int) := [
  (x, y+1), (x+1, y+1), (x+1, y),
  (x+1, y-1), (x, y-1), (x-1, y-1),
  (x-1, y), (x-1, y+1)
]

/-- Theorem: Probability of landing on a vertex is 1/4 -/
theorem vertex_landing_probability :
  let start := (0, 0)
  let p_vertex := (square_vertices.length : ℚ) / (boundary_points.length : ℚ)
  p_vertex = 1/4 := by sorry

end NUMINAMATH_CALUDE_vertex_landing_probability_l3790_379079


namespace NUMINAMATH_CALUDE_jackson_and_williams_money_l3790_379069

theorem jackson_and_williams_money (jackson_money : ℝ) (williams_money : ℝ) :
  jackson_money = 125 →
  jackson_money = 5 * williams_money →
  jackson_money + williams_money = 145.83 :=
by sorry

end NUMINAMATH_CALUDE_jackson_and_williams_money_l3790_379069


namespace NUMINAMATH_CALUDE_probability_sum_12_probability_sum_12_is_19_216_l3790_379062

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ 3

/-- The number of ways to roll a sum of 12 with three dice -/
def waysToRoll12 : ℕ := 19

/-- The probability of rolling a sum of 12 with three standard six-faced dice -/
theorem probability_sum_12 : ℚ :=
  waysToRoll12 / totalOutcomes

/-- Proof that the probability of rolling a sum of 12 with three standard six-faced dice is 19/216 -/
theorem probability_sum_12_is_19_216 : probability_sum_12 = 19 / 216 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_12_probability_sum_12_is_19_216_l3790_379062


namespace NUMINAMATH_CALUDE_smallest_number_with_weight_2000_l3790_379036

/-- The weight of a number is the sum of its digits -/
def weight (n : ℕ) : ℕ := sorry

/-- Construct a number with a leading digit followed by a sequence of nines -/
def constructNumber (lead : ℕ) (nines : ℕ) : ℕ := sorry

theorem smallest_number_with_weight_2000 :
  ∀ n : ℕ, weight n = 2000 → n ≥ constructNumber 2 222 := by sorry

end NUMINAMATH_CALUDE_smallest_number_with_weight_2000_l3790_379036


namespace NUMINAMATH_CALUDE_max_valid_config_l3790_379003

/-- Represents a chessboard configuration -/
structure ChessboardConfig where
  white : Nat
  black : Nat

/-- Checks if a configuration is valid for an 8x8 chessboard -/
def is_valid_config (config : ChessboardConfig) : Prop :=
  config.white + config.black ≤ 64 ∧
  config.white = 2 * config.black ∧
  (config.white + config.black) % 8 = 0

/-- The maximum valid configuration -/
def max_config : ChessboardConfig :=
  ⟨32, 16⟩

/-- Theorem: The maximum valid configuration is (32, 16) -/
theorem max_valid_config :
  is_valid_config max_config ∧
  ∀ (c : ChessboardConfig), is_valid_config c → c.white ≤ max_config.white ∧ c.black ≤ max_config.black :=
by sorry


end NUMINAMATH_CALUDE_max_valid_config_l3790_379003


namespace NUMINAMATH_CALUDE_police_chase_distance_l3790_379065

/-- Calculates the distance between a police station and a thief's starting location
    given their speeds and chase duration. -/
def police_station_distance (thief_speed : ℝ) (police_speed : ℝ) 
                             (head_start : ℝ) (chase_duration : ℝ) : ℝ :=
  police_speed * chase_duration - 
  (thief_speed * head_start + thief_speed * chase_duration)

/-- Theorem stating that given specific chase parameters, 
    the police station is 60 km away from the thief's starting point. -/
theorem police_chase_distance : 
  police_station_distance 20 40 1 4 = 60 := by sorry

end NUMINAMATH_CALUDE_police_chase_distance_l3790_379065


namespace NUMINAMATH_CALUDE_total_marbles_eq_4_9r_l3790_379091

/-- The total number of marbles in a bag given the number of red marbles -/
def total_marbles (r : ℝ) : ℝ :=
  let blue := 1.3 * r
  let green := 2 * blue
  r + blue + green

/-- Theorem stating that the total number of marbles is 4.9 times the number of red marbles -/
theorem total_marbles_eq_4_9r (r : ℝ) : total_marbles r = 4.9 * r := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_eq_4_9r_l3790_379091


namespace NUMINAMATH_CALUDE_investment_profit_sharing_l3790_379038

/-- Represents the capital contribution of an investor over a year -/
def capital_contribution (initial_investment : ℕ) (doubled_after_six_months : Bool) : ℕ :=
  if doubled_after_six_months
  then initial_investment * 6 + (initial_investment * 2) * 6
  else initial_investment * 12

/-- Represents the profit-sharing ratio between two investors -/
def profit_sharing_ratio (a_contribution : ℕ) (b_contribution : ℕ) : Prop :=
  a_contribution = b_contribution

theorem investment_profit_sharing :
  let a_initial_investment : ℕ := 3000
  let b_initial_investment : ℕ := 4500
  let a_doubles_capital : Bool := true
  let b_doubles_capital : Bool := false
  
  let a_contribution := capital_contribution a_initial_investment a_doubles_capital
  let b_contribution := capital_contribution b_initial_investment b_doubles_capital
  
  profit_sharing_ratio a_contribution b_contribution :=
by
  sorry

end NUMINAMATH_CALUDE_investment_profit_sharing_l3790_379038


namespace NUMINAMATH_CALUDE_correct_international_letters_l3790_379081

/-- The number of international letters in a mailing scenario. -/
def num_international_letters : ℕ :=
  let total_letters : ℕ := 4
  let standard_postage : ℚ := 108 / 100  -- $1.08
  let international_charge : ℚ := 14 / 100  -- $0.14
  let total_cost : ℚ := 460 / 100  -- $4.60
  2

/-- Proof that the number of international letters is correct. -/
theorem correct_international_letters : 
  let total_letters : ℕ := 4
  let standard_postage : ℚ := 108 / 100  -- $1.08
  let international_charge : ℚ := 14 / 100  -- $0.14
  let total_cost : ℚ := 460 / 100  -- $4.60
  num_international_letters = 2 ∧
  (num_international_letters : ℚ) * (standard_postage + international_charge) + 
  (total_letters - num_international_letters : ℚ) * standard_postage = total_cost := by
  sorry

end NUMINAMATH_CALUDE_correct_international_letters_l3790_379081


namespace NUMINAMATH_CALUDE_merchant_revenue_l3790_379044

/-- Calculates the total revenue for a set of vegetables --/
def total_revenue (quantities : List ℝ) (prices : List ℝ) (sold_percentages : List ℝ) : ℝ :=
  List.sum (List.zipWith3 (fun q p s => q * p * s) quantities prices sold_percentages)

/-- The total revenue generated by the merchant is $134.1 --/
theorem merchant_revenue : 
  let quantities : List ℝ := [20, 18, 12, 25, 10]
  let prices : List ℝ := [2, 3, 4, 1, 5]
  let sold_percentages : List ℝ := [0.6, 0.4, 0.75, 0.5, 0.8]
  total_revenue quantities prices sold_percentages = 134.1 := by
  sorry

end NUMINAMATH_CALUDE_merchant_revenue_l3790_379044


namespace NUMINAMATH_CALUDE_max_remainder_div_by_nine_l3790_379028

theorem max_remainder_div_by_nine (n : ℕ) (h : n % 9 = 6) : 
  ∀ m : ℕ, m % 9 < 9 ∧ m % 9 ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_remainder_div_by_nine_l3790_379028


namespace NUMINAMATH_CALUDE_circle_product_theorem_l3790_379076

/-- A circular permutation of five elements -/
def CircularPerm (α : Type) := Fin 5 → α

/-- The condition for the first part of the problem -/
def FirstCondition (a b c d e : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧
  a + b + c + d + e = 1 ∧
  ∀ π : CircularPerm ℝ, π 0 = a ∧ π 1 = b ∧ π 2 = c ∧ π 3 = d ∧ π 4 = e →
    ∃ i : Fin 5, π i * π ((i + 1) % 5) ≥ 1/9

/-- The condition for the second part of the problem -/
def SecondCondition (a b c d e : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧
  a + b + c + d + e = 1

/-- The theorem statement combining both parts of the problem -/
theorem circle_product_theorem :
  (∃ a b c d e : ℝ, FirstCondition a b c d e) ∧
  (∀ a b c d e : ℝ, SecondCondition a b c d e →
    ∃ π : CircularPerm ℝ, π 0 = a ∧ π 1 = b ∧ π 2 = c ∧ π 3 = d ∧ π 4 = e ∧
      ∀ i : Fin 5, π i * π ((i + 1) % 5) ≤ 1/9) :=
by sorry

end NUMINAMATH_CALUDE_circle_product_theorem_l3790_379076


namespace NUMINAMATH_CALUDE_revenue_decrease_l3790_379085

def previous_revenue : ℝ := 69.0
def decrease_percentage : ℝ := 30.434782608695656

theorem revenue_decrease (previous_revenue : ℝ) (decrease_percentage : ℝ) :
  previous_revenue * (1 - decrease_percentage / 100) = 48.0 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_l3790_379085


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l3790_379073

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_prime_divisor_of_factorial_sum :
  ∃ (p : ℕ), is_prime p ∧ 
    p ∣ (factorial 10 + factorial 11) ∧ 
    ∀ (q : ℕ), is_prime q → q ∣ (factorial 10 + factorial 11) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l3790_379073


namespace NUMINAMATH_CALUDE_hotel_has_21_rooms_l3790_379060

/-- Represents the inventory and room requirements for a hotel. -/
structure HotelInventory where
  total_lamps : ℕ
  total_chairs : ℕ
  total_bed_sheets : ℕ
  lamps_per_room : ℕ
  chairs_per_room : ℕ
  bed_sheets_per_room : ℕ

/-- Calculates the number of rooms in a hotel based on its inventory and room requirements. -/
def calculateRooms (inventory : HotelInventory) : ℕ :=
  min (inventory.total_lamps / inventory.lamps_per_room)
    (min (inventory.total_chairs / inventory.chairs_per_room)
      (inventory.total_bed_sheets / inventory.bed_sheets_per_room))

/-- Theorem stating that the hotel has 21 rooms based on the given inventory. -/
theorem hotel_has_21_rooms (inventory : HotelInventory)
    (h1 : inventory.total_lamps = 147)
    (h2 : inventory.total_chairs = 84)
    (h3 : inventory.total_bed_sheets = 210)
    (h4 : inventory.lamps_per_room = 7)
    (h5 : inventory.chairs_per_room = 4)
    (h6 : inventory.bed_sheets_per_room = 10) :
    calculateRooms inventory = 21 := by
  sorry

end NUMINAMATH_CALUDE_hotel_has_21_rooms_l3790_379060


namespace NUMINAMATH_CALUDE_min_value_function_l3790_379037

theorem min_value_function (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (6 * x^2 + 9 * x + 2 * y^2 + 3 * y + 20) / (9 * (x + y + 2)) ≥ 4 * Real.sqrt 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_function_l3790_379037


namespace NUMINAMATH_CALUDE_pure_imaginary_quotient_l3790_379050

/-- Given a real number a and i as the imaginary unit, if (a-i)/(1+i) is a pure imaginary number, then a = 1 -/
theorem pure_imaginary_quotient (a : ℝ) : 
  (∃ (b : ℝ), (a - Complex.I) / (1 + Complex.I) = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_quotient_l3790_379050


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_achieved_l3790_379027

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + b = 1) :
  1 / a + 27 / b ≥ 48 := by
  sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 3 * a + b = 1 ∧ 1 / a + 27 / b < 48 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_achieved_l3790_379027


namespace NUMINAMATH_CALUDE_customers_served_today_l3790_379088

theorem customers_served_today (x : ℕ) 
  (h1 : (65 : ℝ) = (65 * x) / x) 
  (h2 : (90 : ℝ) = (65 * x + C) / (x + 1)) 
  (h3 : x = 1) : C = 115 := by
  sorry

end NUMINAMATH_CALUDE_customers_served_today_l3790_379088


namespace NUMINAMATH_CALUDE_absolute_value_ab_l3790_379010

-- Define the constants for the foci locations
def ellipse_focus : ℝ := 5
def hyperbola_focus : ℝ := 7

-- Define the equations for the ellipse and hyperbola
def ellipse_equation (a b : ℝ) : Prop := b^2 - a^2 = ellipse_focus^2
def hyperbola_equation (a b : ℝ) : Prop := a^2 + b^2 = hyperbola_focus^2

-- Theorem statement
theorem absolute_value_ab (a b : ℝ) 
  (h_ellipse : ellipse_equation a b) 
  (h_hyperbola : hyperbola_equation a b) : 
  |a * b| = 2 * Real.sqrt 111 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_ab_l3790_379010


namespace NUMINAMATH_CALUDE_final_painting_width_l3790_379046

theorem final_painting_width (total_paintings : ℕ) (total_area : ℕ) 
  (small_paintings : ℕ) (small_painting_side : ℕ) 
  (large_painting_length : ℕ) (large_painting_width : ℕ)
  (final_painting_height : ℕ) :
  total_paintings = 5 →
  total_area = 200 →
  small_paintings = 3 →
  small_painting_side = 5 →
  large_painting_length = 10 →
  large_painting_width = 8 →
  final_painting_height = 5 →
  (total_area - 
    (small_paintings * small_painting_side * small_painting_side + 
     large_painting_length * large_painting_width)) / final_painting_height = 9 := by
  sorry

#check final_painting_width

end NUMINAMATH_CALUDE_final_painting_width_l3790_379046


namespace NUMINAMATH_CALUDE_c_younger_than_a_l3790_379007

-- Define variables for the ages of A, B, and C
variable (a b c : ℕ)

-- Define the condition given in the problem
def age_difference : Prop := a + b = b + c + 11

-- Theorem to prove
theorem c_younger_than_a (h : age_difference a b c) : a - c = 11 := by
  sorry

end NUMINAMATH_CALUDE_c_younger_than_a_l3790_379007


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3790_379070

theorem absolute_value_inequality (m : ℝ) : 
  (∀ x : ℝ, x + |x - 1| > m) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3790_379070


namespace NUMINAMATH_CALUDE_problem_solution_l3790_379017

theorem problem_solution (x y : ℝ) (h1 : 3 * x + y = 6) (h2 : x + 3 * y = 6) :
  3 * x^2 + 5 * x * y + 3 * y^2 = 99 / 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3790_379017


namespace NUMINAMATH_CALUDE_tea_party_waiting_time_l3790_379099

/-- Mad Hatter's clock speed relative to real time -/
def mad_hatter_clock_speed : ℚ := 5/4

/-- March Hare's clock speed relative to real time -/
def march_hare_clock_speed : ℚ := 5/6

/-- Time shown on both clocks when they meet (in hours after noon) -/
def meeting_time : ℚ := 5

theorem tea_party_waiting_time :
  let mad_hatter_arrival_time := meeting_time / mad_hatter_clock_speed
  let march_hare_arrival_time := meeting_time / march_hare_clock_speed
  march_hare_arrival_time - mad_hatter_arrival_time = 2 := by sorry

end NUMINAMATH_CALUDE_tea_party_waiting_time_l3790_379099


namespace NUMINAMATH_CALUDE_homework_percentage_l3790_379093

theorem homework_percentage (total_angle : ℝ) (less_than_one_hour_angle : ℝ) :
  total_angle = 360 →
  less_than_one_hour_angle = 90 →
  (1 - less_than_one_hour_angle / total_angle) * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_homework_percentage_l3790_379093


namespace NUMINAMATH_CALUDE_smallest_bases_sum_is_correct_l3790_379086

/-- Represents a number in a given base -/
def representationInBase (n : ℕ) (base : ℕ) : ℕ := 
  (n / base) * base + (n % base)

/-- The smallest possible sum of bases c and d where 83 in base c equals 38 in base d -/
def smallestBasesSum : ℕ := 27

theorem smallest_bases_sum_is_correct :
  ∀ c d : ℕ, c ≥ 2 → d ≥ 2 →
  representationInBase 83 c = representationInBase 38 d →
  c + d ≥ smallestBasesSum :=
sorry

end NUMINAMATH_CALUDE_smallest_bases_sum_is_correct_l3790_379086


namespace NUMINAMATH_CALUDE_customers_left_l3790_379095

theorem customers_left (initial_customers : ℕ) (remaining_tables : ℕ) (people_per_table : ℕ) : 
  initial_customers = 21 → 
  remaining_tables = 3 → 
  people_per_table = 3 → 
  initial_customers - (remaining_tables * people_per_table) = 12 := by
sorry

end NUMINAMATH_CALUDE_customers_left_l3790_379095


namespace NUMINAMATH_CALUDE_initial_passengers_l3790_379023

theorem initial_passengers (remaining : ℕ) : 
  remaining = 216 →
  ∃ initial : ℕ,
    initial > 0 ∧
    remaining = initial - 
      (initial / 10 + 
       (initial - initial / 10) / 7 + 
       (initial - initial / 10 - (initial - initial / 10) / 7) / 5) ∧
    initial = 350 := by
  sorry

end NUMINAMATH_CALUDE_initial_passengers_l3790_379023


namespace NUMINAMATH_CALUDE_slope_range_l3790_379049

-- Define the points
def A₁ : ℝ × ℝ := (-2, 0)
def A₂ : ℝ × ℝ := (2, 0)
def B₁ (x : ℝ) : ℝ × ℝ := (x, 2)
def B₂ (x : ℝ) : ℝ × ℝ := (x, -2)
def P (x y : ℝ) : ℝ × ℝ := (x, y)
def O : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 2)

-- Define the dot product
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

-- Define the equation of the ellipse
def on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

-- Define the condition for the line passing through B and intersecting the ellipse
def intersects_ellipse (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    on_ellipse x₁ y₁ ∧
    on_ellipse x₂ y₂ ∧
    y₁ = k * x₁ + 2 ∧
    y₂ = k * x₂ + 2 ∧
    x₁ ≠ x₂

-- Define the condition for the ratio of triangle areas
def area_ratio_condition (x₁ x₂ : ℝ) : Prop :=
  1/2 < |x₁| / |x₂| ∧ |x₁| / |x₂| < 1

-- Main theorem
theorem slope_range :
  ∀ k : ℝ,
    intersects_ellipse k ∧
    (∃ x₁ x₂ : ℝ, area_ratio_condition x₁ x₂) ↔
    (k > Real.sqrt 2 / 2 ∧ k < 3 * Real.sqrt 14 / 14) ∨
    (k < -Real.sqrt 2 / 2 ∧ k > -3 * Real.sqrt 14 / 14) :=
sorry

end NUMINAMATH_CALUDE_slope_range_l3790_379049


namespace NUMINAMATH_CALUDE_y2_greater_than_y1_l3790_379033

-- Define the linear function
def f (x : ℝ) : ℝ := -2 * x + 1

-- Define the points A and B
def A : ℝ × ℝ := (-1, f (-1))
def B : ℝ × ℝ := (-2, f (-2))

-- Theorem statement
theorem y2_greater_than_y1 : A.2 < B.2 := by
  sorry

end NUMINAMATH_CALUDE_y2_greater_than_y1_l3790_379033


namespace NUMINAMATH_CALUDE_lines_intersect_at_point_l3790_379011

/-- Represents a 2D point or vector -/
structure Vec2 where
  x : ℚ
  y : ℚ

/-- Represents a parametric line in 2D -/
structure ParamLine where
  origin : Vec2
  direction : Vec2

/-- The first line -/
def line1 : ParamLine := {
  origin := { x := 1, y := 4 },
  direction := { x := -2, y := 3 }
}

/-- The second line -/
def line2 : ParamLine := {
  origin := { x := 0, y := 5 },
  direction := { x := -1, y := 6 }
}

/-- The intersection point of the two lines -/
def intersection : Vec2 := { x := -1/9, y := 17/3 }

/-- Theorem stating that the given point is the intersection of the two lines -/
theorem lines_intersect_at_point : 
  ∃ (s v : ℚ), 
    line1.origin.x + s * line1.direction.x = intersection.x ∧
    line1.origin.y + s * line1.direction.y = intersection.y ∧
    line2.origin.x + v * line2.direction.x = intersection.x ∧
    line2.origin.y + v * line2.direction.y = intersection.y :=
by
  sorry

end NUMINAMATH_CALUDE_lines_intersect_at_point_l3790_379011


namespace NUMINAMATH_CALUDE_animal_shelter_count_l3790_379013

/-- The number of cats received by the animal shelter -/
def num_cats : ℕ := 40

/-- The difference between the number of cats and dogs -/
def cat_dog_difference : ℕ := 20

/-- The total number of animals received by the shelter -/
def total_animals : ℕ := num_cats + (num_cats - cat_dog_difference)

theorem animal_shelter_count : total_animals = 60 := by
  sorry

end NUMINAMATH_CALUDE_animal_shelter_count_l3790_379013


namespace NUMINAMATH_CALUDE_b_minus_c_subscription_l3790_379094

/-- Represents the business subscription problem -/
structure BusinessSubscription where
  total_subscription : ℕ
  total_profit : ℕ
  c_profit : ℕ
  a_more_than_b : ℕ

/-- Theorem stating the difference between B's and C's subscriptions -/
theorem b_minus_c_subscription (bs : BusinessSubscription)
  (h1 : bs.total_subscription = 50000)
  (h2 : bs.total_profit = 35000)
  (h3 : bs.c_profit = 8400)
  (h4 : bs.a_more_than_b = 4000) :
  ∃ (b_sub c_sub : ℕ), b_sub - c_sub = 10000 ∧
    ∃ (a_sub : ℕ), a_sub + b_sub + c_sub = bs.total_subscription ∧
    a_sub = b_sub + bs.a_more_than_b ∧
    bs.c_profit * bs.total_subscription = c_sub * bs.total_profit :=
by sorry

end NUMINAMATH_CALUDE_b_minus_c_subscription_l3790_379094


namespace NUMINAMATH_CALUDE_opposite_of_negative_seven_l3790_379032

-- Definition of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_seven :
  opposite (-7) = 7 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_seven_l3790_379032


namespace NUMINAMATH_CALUDE_expression_evaluation_l3790_379082

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 2
  (1/2 * x - 2 * (x - 1/3 * y^2) + (-3/2 * x + 1/3 * y^2)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3790_379082


namespace NUMINAMATH_CALUDE_function_upper_bound_l3790_379004

theorem function_upper_bound (x : ℝ) (h : x > 0) : (1 + Real.log x) / x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_upper_bound_l3790_379004


namespace NUMINAMATH_CALUDE_valerie_light_bulb_shortage_l3790_379008

structure LightBulb where
  price : Float
  quantity : Nat

def small_bulb : LightBulb := { price := 8.75, quantity := 3 }
def medium_bulb : LightBulb := { price := 11.25, quantity := 4 }
def large_bulb : LightBulb := { price := 15.50, quantity := 3 }
def extra_small_bulb : LightBulb := { price := 6.10, quantity := 4 }

def budget : Float := 120.00

def total_cost : Float :=
  small_bulb.price * small_bulb.quantity.toFloat +
  medium_bulb.price * medium_bulb.quantity.toFloat +
  large_bulb.price * large_bulb.quantity.toFloat +
  extra_small_bulb.price * extra_small_bulb.quantity.toFloat

theorem valerie_light_bulb_shortage :
  total_cost - budget = 22.15 := by
  sorry


end NUMINAMATH_CALUDE_valerie_light_bulb_shortage_l3790_379008


namespace NUMINAMATH_CALUDE_unique_root_implies_specific_function_max_min_on_interval_l3790_379035

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

-- Theorem 1
theorem unique_root_implies_specific_function (a b : ℝ) (h1 : a ≠ 0) (h2 : f a b 2 = 0) 
  (h3 : ∃! x, f a b x - x = 0) : 
  ∀ x, f (-1/2) 1 x = f a b x := by sorry

-- Theorem 2
theorem max_min_on_interval (x : ℝ) (h : x ∈ Set.Icc (-1) 2) : 
  f 1 (-2) x ≤ 3 ∧ f 1 (-2) x ≥ -1 ∧ 
  (∃ x₁ ∈ Set.Icc (-1) 2, f 1 (-2) x₁ = 3) ∧ 
  (∃ x₂ ∈ Set.Icc (-1) 2, f 1 (-2) x₂ = -1) := by sorry

end NUMINAMATH_CALUDE_unique_root_implies_specific_function_max_min_on_interval_l3790_379035


namespace NUMINAMATH_CALUDE_prob_two_consecutive_wins_l3790_379084

/-- The probability of player A winning exactly two consecutive games in a three-game series -/
theorem prob_two_consecutive_wins (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/4) (h2 : p2 = 1/3) (h3 : p3 = 1/3) : 
  p1 * p2 * (1 - p3) + (1 - p1) * p2 * p3 = 5/36 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_consecutive_wins_l3790_379084


namespace NUMINAMATH_CALUDE_min_value_theorem_l3790_379078

theorem min_value_theorem (m : ℝ) (a b : ℝ) :
  0 < m → m < 1 →
  ({x : ℝ | x^2 - 2*x + 1 - m^2 < 0} = {x : ℝ | a < x ∧ x < b}) →
  (∀ x : ℝ, x^2 - 2*x + 1 - m^2 < 0 ↔ a < x ∧ x < b) →
  (∀ x : ℝ, 1/(8*a + 2*b) - 1/(3*a - 3*b) ≥ 2/5) ∧
  (∃ x : ℝ, 1/(8*a + 2*b) - 1/(3*a - 3*b) = 2/5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3790_379078


namespace NUMINAMATH_CALUDE_sector_area_l3790_379002

/-- The area of a circular sector with radius R and circumference 4R is R^2 -/
theorem sector_area (R : ℝ) (R_pos : R > 0) : 
  let circumference := 4 * R
  let arc_length := circumference - 2 * R
  let sector_area := (1 / 2) * arc_length * R
  sector_area = R^2 := by
sorry

end NUMINAMATH_CALUDE_sector_area_l3790_379002


namespace NUMINAMATH_CALUDE_nested_series_sum_l3790_379098

def nested_series : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * (1 + nested_series n)

theorem nested_series_sum : nested_series 5 = 126 := by
  sorry

end NUMINAMATH_CALUDE_nested_series_sum_l3790_379098


namespace NUMINAMATH_CALUDE_absolute_value_theorem_l3790_379016

theorem absolute_value_theorem (x : ℝ) (h : x < -1) :
  |x - Real.sqrt ((x + 2)^2)| = -2*x - 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_theorem_l3790_379016


namespace NUMINAMATH_CALUDE_f_monotonic_increase_interval_l3790_379083

open Real

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 6) + cos (2 * x - π / 3)

theorem f_monotonic_increase_interval :
  ∀ k : ℤ, StrictMonoOn f (Set.Ioo (k * π - π / 3) (k * π + π / 6)) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonic_increase_interval_l3790_379083


namespace NUMINAMATH_CALUDE_star_example_l3790_379075

-- Define the star operation
def star (m n p q : ℚ) : ℚ := m * p * (n / q)

-- Theorem statement
theorem star_example : star (5/9) (6/4) = 135/2 := by sorry

end NUMINAMATH_CALUDE_star_example_l3790_379075


namespace NUMINAMATH_CALUDE_intercept_sum_l3790_379040

/-- Given a line with equation y - 3 = -3(x - 5), prove that the sum of its x-intercept and y-intercept is 24 -/
theorem intercept_sum (x y : ℝ) : 
  (y - 3 = -3 * (x - 5)) → 
  ∃ (x_int y_int : ℝ), 
    (0 - 3 = -3 * (x_int - 5)) ∧ 
    (y_int - 3 = -3 * (0 - 5)) ∧ 
    (x_int + y_int = 24) := by
  sorry

end NUMINAMATH_CALUDE_intercept_sum_l3790_379040


namespace NUMINAMATH_CALUDE_gym_membership_cost_theorem_l3790_379092

/-- Calculates the total cost of a gym membership for a given number of years -/
def gymMembershipCost (monthlyFee : ℕ) (downPayment : ℕ) (years : ℕ) : ℕ :=
  monthlyFee * 12 * years + downPayment

/-- Theorem: The total cost for a 3-year gym membership with a $12 monthly fee and $50 down payment is $482 -/
theorem gym_membership_cost_theorem :
  gymMembershipCost 12 50 3 = 482 := by
  sorry

end NUMINAMATH_CALUDE_gym_membership_cost_theorem_l3790_379092


namespace NUMINAMATH_CALUDE_smallest_two_base_representation_l3790_379096

/-- Represents a number in a given base with two identical digits --/
def twoDigitNumber (digit : Nat) (base : Nat) : Nat :=
  digit * base + digit

/-- Checks if a number is valid in a given base --/
def isValidInBase (n : Nat) (base : Nat) : Prop :=
  n < base

theorem smallest_two_base_representation : 
  ∀ n : Nat, n < 24 → 
  ¬(∃ (a b : Nat), 
    isValidInBase a 5 ∧ 
    isValidInBase b 7 ∧ 
    n = twoDigitNumber a 5 ∧ 
    n = twoDigitNumber b 7) ∧
  (∃ (a b : Nat),
    isValidInBase a 5 ∧
    isValidInBase b 7 ∧
    24 = twoDigitNumber a 5 ∧
    24 = twoDigitNumber b 7) :=
by sorry

#check smallest_two_base_representation

end NUMINAMATH_CALUDE_smallest_two_base_representation_l3790_379096


namespace NUMINAMATH_CALUDE_price_ratio_theorem_l3790_379041

theorem price_ratio_theorem (cost : ℝ) (price1 price2 : ℝ) 
  (h1 : price1 = cost * (1 + 0.35))
  (h2 : price2 = cost * (1 - 0.10)) :
  price2 / price1 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_price_ratio_theorem_l3790_379041


namespace NUMINAMATH_CALUDE_partner_b_profit_share_l3790_379047

/-- Calculates the share of profit for partner B given the investment ratios and total profit -/
theorem partner_b_profit_share 
  (invest_a invest_b invest_c : ℚ) 
  (total_profit : ℚ)
  (h1 : invest_a = 3 * invest_b)
  (h2 : invest_b = (2/3) * invest_c)
  (h3 : total_profit = 5500) :
  (invest_b / (invest_a + invest_b + invest_c)) * total_profit = 1000 := by
  sorry

end NUMINAMATH_CALUDE_partner_b_profit_share_l3790_379047


namespace NUMINAMATH_CALUDE_zero_is_root_of_polynomial_l3790_379052

theorem zero_is_root_of_polynomial : ∃ (x : ℝ), 12 * x^4 + 38 * x^3 - 51 * x^2 + 40 * x = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_is_root_of_polynomial_l3790_379052


namespace NUMINAMATH_CALUDE_identical_cuts_different_shapes_l3790_379067

/-- Represents a polygon --/
structure Polygon where
  area : ℝ

/-- Represents a square --/
structure Square where
  side : ℝ

/-- Represents a triangle --/
structure Triangle where
  base : ℝ
  height : ℝ

/-- The theorem stating that it's possible to cut identical pieces from two identical polygons
    such that one remaining shape is a square and the other is a triangle --/
theorem identical_cuts_different_shapes (original : Polygon) :
  ∃ (cut_piece : ℝ) (square : Square) (triangle : Triangle),
    original.area = square.side ^ 2 + cut_piece ∧
    original.area = (1 / 2) * triangle.base * triangle.height + cut_piece ∧
    square.side ^ 2 = (1 / 2) * triangle.base * triangle.height :=
sorry

end NUMINAMATH_CALUDE_identical_cuts_different_shapes_l3790_379067


namespace NUMINAMATH_CALUDE_cuboid_dimensions_l3790_379001

/-- Represents a cuboid with side areas a, b, and c, and dimensions l, w, h -/
structure Cuboid where
  a : ℝ
  b : ℝ
  c : ℝ
  l : ℝ
  w : ℝ
  h : ℝ

/-- The theorem stating that a cuboid with side areas 5, 8, and 10 has dimensions 4, 2.5, and 2 -/
theorem cuboid_dimensions (cube : Cuboid) 
  (h1 : cube.a = 5) 
  (h2 : cube.b = 8) 
  (h3 : cube.c = 10) 
  (h4 : cube.l * cube.w = cube.a) 
  (h5 : cube.l * cube.h = cube.b) 
  (h6 : cube.w * cube.h = cube.c) :
  cube.l = 4 ∧ cube.w = 2.5 ∧ cube.h = 2 := by
  sorry


end NUMINAMATH_CALUDE_cuboid_dimensions_l3790_379001


namespace NUMINAMATH_CALUDE_egypt_traditional_growth_l3790_379071

-- Define the set of countries
inductive Country
| UnitedStates
| Japan
| France
| Egypt

-- Define the development status of a country
inductive DevelopmentStatus
| Developed
| Developing

-- Define the population growth pattern
inductive PopulationGrowthPattern
| Modern
| Traditional

-- Function to determine the development status of a country
def developmentStatus (c : Country) : DevelopmentStatus :=
  match c with
  | Country.Egypt => DevelopmentStatus.Developing
  | _ => DevelopmentStatus.Developed

-- Function to determine the population growth pattern based on development status
def growthPattern (s : DevelopmentStatus) : PopulationGrowthPattern :=
  match s with
  | DevelopmentStatus.Developed => PopulationGrowthPattern.Modern
  | DevelopmentStatus.Developing => PopulationGrowthPattern.Traditional

-- Theorem: Egypt is the only country with a traditional population growth pattern
theorem egypt_traditional_growth : 
  ∀ c : Country, 
    growthPattern (developmentStatus c) = PopulationGrowthPattern.Traditional ↔ 
    c = Country.Egypt :=
  sorry


end NUMINAMATH_CALUDE_egypt_traditional_growth_l3790_379071


namespace NUMINAMATH_CALUDE_students_passed_at_least_one_subject_l3790_379074

theorem students_passed_at_least_one_subject 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 32) 
  (h2 : failed_english = 56) 
  (h3 : failed_both = 12) : 
  100 - (failed_hindi + failed_english - failed_both) = 24 := by
  sorry

end NUMINAMATH_CALUDE_students_passed_at_least_one_subject_l3790_379074


namespace NUMINAMATH_CALUDE_restaurant_ratio_change_l3790_379029

theorem restaurant_ratio_change (initial_cooks : ℕ) (initial_waiters : ℕ) 
  (additional_waiters : ℕ) :
  initial_cooks = 9 →
  initial_cooks * 10 = initial_waiters * 3 →
  additional_waiters = 12 →
  (initial_cooks : ℚ) / (initial_waiters + additional_waiters : ℚ) = 3 / 14 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_ratio_change_l3790_379029


namespace NUMINAMATH_CALUDE_condition_a_equals_one_sufficient_not_necessary_l3790_379072

-- Define the quadratic equation
def has_real_roots (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a = 2*x

-- Theorem statement
theorem condition_a_equals_one_sufficient_not_necessary :
  (has_real_roots 1) ∧ (∃ a : ℝ, a ≠ 1 ∧ has_real_roots a) :=
sorry

end NUMINAMATH_CALUDE_condition_a_equals_one_sufficient_not_necessary_l3790_379072


namespace NUMINAMATH_CALUDE_sequence_ratio_proof_l3790_379022

theorem sequence_ratio_proof (a : ℕ → ℕ+) 
  (h1 : a 1 = 1)
  (h2 : a 2 = 2009)
  (h3 : ∀ n : ℕ, n ≥ 1 → (a (n + 2)) * (a n) = (a (n + 1))^2 + (a (n + 1)) * (a n)) :
  (a 993 : ℚ) / (100 * (a 991)) = 89970 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_proof_l3790_379022


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3790_379090

open Set

/-- The solution set of the inequality -x^2 + ax + b ≥ 0 -/
def SolutionSet (a b : ℝ) : Set ℝ := {x | -x^2 + a*x + b ≥ 0}

/-- The theorem stating the equivalence of the solution sets -/
theorem solution_set_equivalence (a b : ℝ) :
  SolutionSet a b = Icc (-2) 3 →
  {x : ℝ | x^2 - 5*a*x + b > 0} = {x : ℝ | x < 2 ∨ x > 3} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l3790_379090
