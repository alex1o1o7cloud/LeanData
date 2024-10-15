import Mathlib

namespace NUMINAMATH_CALUDE_gcd_7429_13356_l3549_354973

theorem gcd_7429_13356 : Nat.gcd 7429 13356 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7429_13356_l3549_354973


namespace NUMINAMATH_CALUDE_three_balls_per_can_l3549_354997

/-- Represents a tennis tournament with a given structure and ball usage. -/
structure TennisTournament where
  rounds : Nat
  games_per_round : List Nat
  cans_per_game : Nat
  total_balls : Nat

/-- Calculates the number of tennis balls per can in a given tournament. -/
def balls_per_can (t : TennisTournament) : Nat :=
  let total_games := t.games_per_round.sum
  let total_cans := total_games * t.cans_per_game
  t.total_balls / total_cans

/-- Theorem stating that for the given tournament structure, there are 3 balls per can. -/
theorem three_balls_per_can :
  let t : TennisTournament := {
    rounds := 4,
    games_per_round := [8, 4, 2, 1],
    cans_per_game := 5,
    total_balls := 225
  }
  balls_per_can t = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_balls_per_can_l3549_354997


namespace NUMINAMATH_CALUDE_intersection_empty_iff_b_in_range_l3549_354936

def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

theorem intersection_empty_iff_b_in_range :
  ∀ b : ℝ, (∀ m : ℝ, M ∩ N m b = ∅) ↔ b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_b_in_range_l3549_354936


namespace NUMINAMATH_CALUDE_sequence_problem_l3549_354905

def arithmetic_sequence (a b c d : ℝ) : Prop :=
  (b - a = c - b) ∧ (c - b = d - c)

def geometric_sequence (a b c d e : ℝ) : Prop :=
  (b / a = c / b) ∧ (c / b = d / c) ∧ (d / c = e / d)

theorem sequence_problem (a₁ a₂ b₁ b₂ b₃ : ℝ) :
  arithmetic_sequence (-7) a₁ a₂ (-1) →
  geometric_sequence (-4) b₁ b₂ b₃ (-1) →
  (a₂ - a₁) / b₂ = -1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l3549_354905


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3549_354910

theorem simplify_square_roots : 
  (Real.sqrt 392 / Real.sqrt 336) + (Real.sqrt 192 / Real.sqrt 144) = 11/6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3549_354910


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l3549_354994

theorem smallest_n_for_inequality : ∃ (n : ℕ), (∀ (w x y z : ℝ), (w^2 + x^2 + y^2 + z^2)^2 ≤ n * (w^4 + x^4 + y^4 + z^4)) ∧
  (∀ (m : ℕ), m < n → ∃ (w x y z : ℝ), (w^2 + x^2 + y^2 + z^2)^2 > m * (w^4 + x^4 + y^4 + z^4)) ∧
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l3549_354994


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l3549_354911

theorem factor_difference_of_squares (x : ℝ) : 49 - 16 * x^2 = (7 - 4*x) * (7 + 4*x) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l3549_354911


namespace NUMINAMATH_CALUDE_exactly_two_partitions_l3549_354912

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- A function that represents a valid partition of 100 into three distinct positive perfect squares -/
def valid_partition (a b c : ℕ) : Prop :=
  a + b + c = 100 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_square c

/-- The main theorem stating that there are exactly 2 valid partitions -/
theorem exactly_two_partitions :
  ∃! (s : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ s ↔ valid_partition a b c) ∧
    s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_partitions_l3549_354912


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3549_354972

/-- Proves that the complex fraction (3+8i)/(1-4i) simplifies to -29/17 + 20/17*i -/
theorem complex_fraction_simplification :
  (3 + 8 * Complex.I) / (1 - 4 * Complex.I) = -29/17 + 20/17 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3549_354972


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l3549_354915

/-- Given an ellipse and a line, this theorem states the conditions for their intersection at two distinct points. -/
theorem ellipse_line_intersection (m : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
    (x₁^2 / 3 + y₁^2 / m = 1) ∧
    (x₁ + 2*y₁ - 2 = 0) ∧
    (x₂^2 / 3 + y₂^2 / m = 1) ∧
    (x₂ + 2*y₂ - 2 = 0)) ↔ 
  (m > 1/12 ∧ m < 3) ∨ m > 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l3549_354915


namespace NUMINAMATH_CALUDE_finite_cuboidal_blocks_l3549_354949

theorem finite_cuboidal_blocks :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), ∀ a b c : ℕ,
    0 < c ∧ c ≤ b ∧ b ≤ a ∧ a * b * c = 2 * (a - 2) * (b - 2) * (c - 2) →
    (a, b, c) ∈ S := by
  sorry

end NUMINAMATH_CALUDE_finite_cuboidal_blocks_l3549_354949


namespace NUMINAMATH_CALUDE_rent_share_ratio_l3549_354903

/-- Proves that the ratio of Sheila's share to Purity's share is 5:1 given the rent conditions --/
theorem rent_share_ratio (total_rent : ℝ) (rose_share : ℝ) (purity_share : ℝ) (sheila_share : ℝ) :
  total_rent = 5400 →
  rose_share = 1800 →
  rose_share = 3 * purity_share →
  total_rent = purity_share + rose_share + sheila_share →
  sheila_share / purity_share = 5 := by
  sorry

#check rent_share_ratio

end NUMINAMATH_CALUDE_rent_share_ratio_l3549_354903


namespace NUMINAMATH_CALUDE_largest_prime_for_prime_check_l3549_354908

theorem largest_prime_for_prime_check : ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 1050 →
  ∀ p : ℕ, Prime p ∧ p ≤ Real.sqrt n → p ≤ 31 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_for_prime_check_l3549_354908


namespace NUMINAMATH_CALUDE_min_distance_tangent_line_l3549_354985

/-- Circle M -/
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 2 = 0

/-- Line l -/
def line_l (x y : ℝ) : Prop := 2*x + y + 2 = 0

/-- Point P on line l -/
def point_P (x y : ℝ) : Prop := line_l x y

/-- Tangent points A and B on circle M -/
def tangent_points (xA yA xB yB : ℝ) : Prop := 
  circle_M xA yA ∧ circle_M xB yB

/-- Line AB -/
def line_AB (x y : ℝ) : Prop := 2*x + y + 1 = 0

theorem min_distance_tangent_line : 
  ∃ (xP yP xA yA xB yB : ℝ),
    point_P xP yP ∧
    tangent_points xA yA xB yB ∧
    (∀ (x'P y'P x'A y'A x'B y'B : ℝ),
      point_P x'P y'P → 
      tangent_points x'A y'A x'B y'B →
      (xP - 1)^2 + (yP - 1)^2 ≤ (x'P - 1)^2 + (y'P - 1)^2) →
    line_AB xA yA ∧ line_AB xB yB :=
sorry

end NUMINAMATH_CALUDE_min_distance_tangent_line_l3549_354985


namespace NUMINAMATH_CALUDE_parabola_transformation_l3549_354937

/-- The original parabola function -/
def f (x : ℝ) : ℝ := -(x + 3) * (x - 2)

/-- The transformed parabola function -/
def g (x : ℝ) : ℝ := -(x - 3) * (x + 2)

/-- The transformation function -/
def T (x : ℝ) : ℝ := x + 1

theorem parabola_transformation :
  ∀ x : ℝ, f x = g (T x) :=
sorry

end NUMINAMATH_CALUDE_parabola_transformation_l3549_354937


namespace NUMINAMATH_CALUDE_peter_lost_marbles_l3549_354993

/-- The number of marbles Peter lost -/
def marbles_lost (initial : ℕ) (current : ℕ) : ℕ := initial - current

/-- Theorem stating that the number of marbles Peter lost is the difference between his initial and current marbles -/
theorem peter_lost_marbles (initial : ℕ) (current : ℕ) (h : initial ≥ current) :
  marbles_lost initial current = initial - current :=
by sorry

end NUMINAMATH_CALUDE_peter_lost_marbles_l3549_354993


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3549_354914

theorem trigonometric_identities (θ : Real) 
  (h : Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5) : 
  (1 / Real.sin θ + 1 / Real.cos θ = 2 * Real.sqrt 10 / 3) ∧ 
  (Real.tan θ = -1/3 ∨ Real.tan θ = -3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3549_354914


namespace NUMINAMATH_CALUDE_hcl_formation_l3549_354925

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents a chemical reaction between NaCl and HNO3 -/
structure Reaction where
  nacl : Moles
  hno3 : Moles
  nano3 : Moles
  hcl : Moles

/-- Defines a balanced reaction where NaCl and HNO3 react in a 1:1 ratio -/
def balanced_reaction (r : Reaction) : Prop :=
  r.nacl = r.hno3 ∧ r.nacl = r.hcl ∧ r.nacl = r.nano3

/-- Theorem: In a balanced reaction, the number of moles of HCl formed
    is equal to the number of moles of NaCl used -/
theorem hcl_formation (r : Reaction) (h : balanced_reaction r) :
  r.hcl = r.nacl := by sorry

end NUMINAMATH_CALUDE_hcl_formation_l3549_354925


namespace NUMINAMATH_CALUDE_ten_percent_of_400_minus_25_l3549_354933

theorem ten_percent_of_400_minus_25 : 
  (400 * (10 : ℝ) / 100) - 25 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ten_percent_of_400_minus_25_l3549_354933


namespace NUMINAMATH_CALUDE_train_arrival_time_correct_l3549_354932

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a train journey -/
structure TrainJourney where
  distance : Nat  -- in miles
  speed : Nat     -- in miles per hour
  departure : Time
  timeDifference : Int  -- time difference between departure and arrival time zones

def arrivalTime (journey : TrainJourney) : Time :=
  sorry

theorem train_arrival_time_correct (journey : TrainJourney) 
  (h1 : journey.distance = 480)
  (h2 : journey.speed = 60)
  (h3 : journey.departure = ⟨10, 0⟩)
  (h4 : journey.timeDifference = -1) :
  arrivalTime journey = ⟨17, 0⟩ :=
  sorry

end NUMINAMATH_CALUDE_train_arrival_time_correct_l3549_354932


namespace NUMINAMATH_CALUDE_equation_solution_l3549_354986

theorem equation_solution : ∃ x : ℚ, 2 * x - 5 = 10 + 4 * x ∧ x = -7.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3549_354986


namespace NUMINAMATH_CALUDE_sin_negative_270_degrees_l3549_354989

theorem sin_negative_270_degrees : Real.sin ((-270 : ℝ) * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_270_degrees_l3549_354989


namespace NUMINAMATH_CALUDE_ben_initial_eggs_l3549_354907

/-- The number of eggs Ben had initially -/
def initial_eggs : ℕ := sorry

/-- The number of eggs Ben ate in the morning -/
def morning_eggs : ℕ := 4

/-- The number of eggs Ben ate in the afternoon -/
def afternoon_eggs : ℕ := 3

/-- The number of eggs Ben has left -/
def remaining_eggs : ℕ := 13

/-- Theorem stating that Ben initially had 20 eggs -/
theorem ben_initial_eggs : initial_eggs = 20 := by
  sorry

end NUMINAMATH_CALUDE_ben_initial_eggs_l3549_354907


namespace NUMINAMATH_CALUDE_unique_prime_exponent_l3549_354929

theorem unique_prime_exponent : ∃! (n : ℕ), Nat.Prime (3^(2*n) - 2^n) :=
  sorry

end NUMINAMATH_CALUDE_unique_prime_exponent_l3549_354929


namespace NUMINAMATH_CALUDE_total_labor_tools_l3549_354961

/-- Given a school with 3 grades, where each grade receives n sets of labor tools,
    prove that the total number of sets needed is 3n. -/
theorem total_labor_tools (n : ℕ) : 3 * n = 3 * n := by sorry

end NUMINAMATH_CALUDE_total_labor_tools_l3549_354961


namespace NUMINAMATH_CALUDE_armchair_price_l3549_354928

/-- Calculates the price of each armchair in a living room set purchase --/
theorem armchair_price (sofa_price : ℕ) (num_armchairs : ℕ) (coffee_table_price : ℕ) (total_invoice : ℕ) :
  sofa_price = 1250 →
  num_armchairs = 2 →
  coffee_table_price = 330 →
  total_invoice = 2430 →
  (total_invoice - sofa_price - coffee_table_price) / num_armchairs = 425 := by
  sorry

end NUMINAMATH_CALUDE_armchair_price_l3549_354928


namespace NUMINAMATH_CALUDE_worker_completion_times_l3549_354999

/-- 
Given two positive real numbers p and q, where p < q, and three workers with the following properties:
1. The first worker takes p more days than the second worker to complete a job.
2. The first worker takes q more days than the third worker to complete the job.
3. The first two workers together can complete the job in the same amount of time as the third worker alone.

This theorem proves the time needed for each worker to complete the job individually.
-/
theorem worker_completion_times (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (hpq : p < q) :
  let x := q + Real.sqrt (q * (q - p))
  let y := q - p + Real.sqrt (q * (q - p))
  let z := Real.sqrt (q * (q - p))
  (1 / x + 1 / (x - p) = 1 / (x - q)) ∧
  (x > 0) ∧ (x - p > 0) ∧ (x - q > 0) ∧
  (x = q + Real.sqrt (q * (q - p))) ∧
  (y = q - p + Real.sqrt (q * (q - p))) ∧
  (z = Real.sqrt (q * (q - p))) := by
  sorry

#check worker_completion_times

end NUMINAMATH_CALUDE_worker_completion_times_l3549_354999


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l3549_354960

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : a + b + c = 6)
  (h2 : 2 * (a * b + b * c + a * c) = 24) :
  Real.sqrt (a^2 + b^2 + c^2) = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l3549_354960


namespace NUMINAMATH_CALUDE_sin_product_equality_l3549_354944

theorem sin_product_equality : 
  Real.sin (9 * π / 180) * Real.sin (45 * π / 180) * Real.sin (69 * π / 180) * Real.sin (81 * π / 180) = 
  (Real.sin (39 * π / 180) * Real.sqrt 2) / 4 := by
sorry

end NUMINAMATH_CALUDE_sin_product_equality_l3549_354944


namespace NUMINAMATH_CALUDE_symmetry_of_M_and_N_l3549_354954

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the z-axis -/
def symmetricAboutZAxis (p q : Point3D) : Prop :=
  p.x = -q.x ∧ p.y = -q.y ∧ p.z = q.z

theorem symmetry_of_M_and_N :
  let M : Point3D := ⟨1, -2, 3⟩
  let N : Point3D := ⟨-1, 2, 3⟩
  symmetricAboutZAxis M N := by sorry

end NUMINAMATH_CALUDE_symmetry_of_M_and_N_l3549_354954


namespace NUMINAMATH_CALUDE_no_ten_digit_square_plus_three_with_distinct_digits_l3549_354941

theorem no_ten_digit_square_plus_three_with_distinct_digits :
  ¬ ∃ (n : ℕ), 
    (10^9 ≤ n^2 + 3) ∧ 
    (n^2 + 3 < 10^10) ∧ 
    (∀ (i j : Fin 10), i ≠ j → 
      (((n^2 + 3) / 10^i.val) % 10 ≠ ((n^2 + 3) / 10^j.val) % 10)) :=
sorry

end NUMINAMATH_CALUDE_no_ten_digit_square_plus_three_with_distinct_digits_l3549_354941


namespace NUMINAMATH_CALUDE_haydens_tank_water_remaining_l3549_354931

/-- Calculates the amount of water remaining in a tank after a given time period,
    considering initial volume, loss rate, and water additions. -/
def water_remaining (initial_volume : ℝ) (loss_rate : ℝ) (time : ℕ) (additions : List ℝ) : ℝ :=
  initial_volume - loss_rate * time + additions.sum

/-- Theorem stating that given the specific conditions of Hayden's tank,
    the amount of water remaining after 4 hours is 36 gallons. -/
theorem haydens_tank_water_remaining :
  let initial_volume : ℝ := 40
  let loss_rate : ℝ := 2
  let time : ℕ := 4
  let additions : List ℝ := [0, 0, 1, 3]
  water_remaining initial_volume loss_rate time additions = 36 := by
  sorry

end NUMINAMATH_CALUDE_haydens_tank_water_remaining_l3549_354931


namespace NUMINAMATH_CALUDE_inequality_proof_l3549_354956

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : d - a < c - b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3549_354956


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3549_354942

theorem complex_fraction_simplification :
  let z₁ : ℂ := 3 + 5*I
  let z₂ : ℂ := -2 + 7*I
  z₁ / z₂ = 29/53 - (31/53)*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3549_354942


namespace NUMINAMATH_CALUDE_mrs_sheridan_fish_count_l3549_354970

/-- The number of fish Mrs. Sheridan initially had -/
def initial_fish : ℕ := 22

/-- The number of fish Mrs. Sheridan's sister gave her -/
def additional_fish : ℕ := 47

/-- The total number of fish Mrs. Sheridan has now -/
def total_fish : ℕ := initial_fish + additional_fish

theorem mrs_sheridan_fish_count : total_fish = 69 := by
  sorry

end NUMINAMATH_CALUDE_mrs_sheridan_fish_count_l3549_354970


namespace NUMINAMATH_CALUDE_math_team_combinations_l3549_354934

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem math_team_combinations : 
  let girls := 4
  let boys := 7
  let team_girls := 3
  let team_boys := 3
  (choose girls team_girls) * (choose boys team_boys) = 140 := by
sorry

end NUMINAMATH_CALUDE_math_team_combinations_l3549_354934


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3549_354935

/-- The sum of the infinite series ∑(n=1 to ∞) (3n+2)/(2^n) is equal to 8 -/
theorem infinite_series_sum : ∑' n, (3 * n + 2) / (2 ^ n) = 8 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3549_354935


namespace NUMINAMATH_CALUDE_wilson_sledding_l3549_354966

/-- The number of times Wilson sleds down each tall hill -/
def T : ℕ := sorry

/-- The number of times Wilson sleds down each small hill -/
def S : ℕ := sorry

/-- There are 2 tall hills and 3 small hills -/
axiom hill_counts : 2 * T + 3 * S = 14

/-- The number of times he sleds down each small hill is half the number of times he sleds down each tall hill -/
axiom small_hill_frequency : S = T / 2

theorem wilson_sledding :
  T = 4 := by sorry

end NUMINAMATH_CALUDE_wilson_sledding_l3549_354966


namespace NUMINAMATH_CALUDE_sexagenary_cycle_after_80_years_l3549_354923

/-- Represents the Chinese sexagenary cycle -/
structure SexagenaryCycle where
  heavenly_stems : Fin 10
  earthly_branches : Fin 12

/-- Advances the cycle by n years -/
def advance_cycle (cycle : SexagenaryCycle) (n : ℕ) : SexagenaryCycle :=
  { heavenly_stems := (cycle.heavenly_stems + n) % 10,
    earthly_branches := (cycle.earthly_branches + n) % 12 }

/-- Represents the specific combinations in the problem -/
def ji_chou : SexagenaryCycle := ⟨5, 1⟩  -- 己丑
def ji_you : SexagenaryCycle := ⟨5, 9⟩   -- 己酉

/-- The main theorem to prove -/
theorem sexagenary_cycle_after_80_years :
  ∀ (year : ℕ), advance_cycle ji_chou 80 = ji_you := by
  sorry

end NUMINAMATH_CALUDE_sexagenary_cycle_after_80_years_l3549_354923


namespace NUMINAMATH_CALUDE_shifted_function_point_l3549_354982

-- Define a function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem shifted_function_point (h : f 1 = 1) : 
  f (5 - 4) = 1 := by sorry

end NUMINAMATH_CALUDE_shifted_function_point_l3549_354982


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3549_354947

/-- An arithmetic sequence with common difference -2 and S_5 = 10 has a_100 = -192 -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n, a (n + 1) - a n = -2) →  -- arithmetic sequence with common difference -2
  (S 5 = 10) →                   -- sum of first 5 terms is 10
  (∀ n, S n = n * a 1 + n * (n - 1) * (-1)) →  -- formula for sum of arithmetic sequence
  (a 100 = -192) :=              -- a_100 = -192
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3549_354947


namespace NUMINAMATH_CALUDE_farrah_match_sticks_l3549_354984

/-- Calculates the total number of match sticks ordered given the number of boxes, 
    matchboxes per box, and sticks per matchbox. -/
def total_match_sticks (boxes : ℕ) (matchboxes_per_box : ℕ) (sticks_per_matchbox : ℕ) : ℕ :=
  boxes * matchboxes_per_box * sticks_per_matchbox

/-- Proves that the total number of match sticks ordered by Farrah is 122,500. -/
theorem farrah_match_sticks : 
  total_match_sticks 7 35 500 = 122500 := by
  sorry

end NUMINAMATH_CALUDE_farrah_match_sticks_l3549_354984


namespace NUMINAMATH_CALUDE_distance_ratio_bound_l3549_354906

/-- Given n points on a plane with maximum distance D and minimum distance d between any two points,
    the ratio of maximum to minimum distance is greater than (√(nπ)/2) - 1. -/
theorem distance_ratio_bound (n : ℕ) (D d : ℝ) (h_pos : 0 < d) (h_max : d ≤ D) :
  D / d > Real.sqrt (n * Real.pi) / 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_ratio_bound_l3549_354906


namespace NUMINAMATH_CALUDE_unit_digit_of_3_power_2023_l3549_354916

def unit_digit_pattern : List Nat := [3, 9, 7, 1]

theorem unit_digit_of_3_power_2023 :
  (3^2023 : ℕ) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_3_power_2023_l3549_354916


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3549_354926

theorem coin_flip_probability (p : ℝ) (n : ℕ) (h_p : p = 1 / 2) (h_n : n = 5) :
  p ^ 4 * (1 - p) = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3549_354926


namespace NUMINAMATH_CALUDE_population_reaches_capacity_in_90_years_l3549_354920

def usable_land : ℕ := 32500
def acres_per_person : ℕ := 2
def initial_population : ℕ := 500
def growth_factor : ℕ := 4
def growth_period : ℕ := 30

def max_capacity : ℕ := usable_land / acres_per_person

def population_after_years (years : ℕ) : ℕ :=
  initial_population * (growth_factor ^ (years / growth_period))

theorem population_reaches_capacity_in_90_years :
  population_after_years 90 ≥ max_capacity ∧
  population_after_years 60 < max_capacity :=
sorry

end NUMINAMATH_CALUDE_population_reaches_capacity_in_90_years_l3549_354920


namespace NUMINAMATH_CALUDE_opposite_of_negative_six_l3549_354965

theorem opposite_of_negative_six : 
  -((-6) : ℤ) = 6 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_six_l3549_354965


namespace NUMINAMATH_CALUDE_acute_angles_sum_l3549_354939

theorem acute_angles_sum (α β : Real) (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_β : 0 < β ∧ β < π / 2)
  (h_condition : (1 + Real.sqrt 3 * Real.tan α) * (1 + Real.sqrt 3 * Real.tan β) = 4) :
  α + β = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_acute_angles_sum_l3549_354939


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l3549_354996

open Matrix

theorem matrix_equation_solution {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) 
  (h_inv : IsUnit A) 
  (h_eq : (A - 3 • 1) * (A - 5 • 1) = -1) : 
  A + 10 • A⁻¹ = (6.5 : ℝ) • 1 := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l3549_354996


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3549_354951

theorem polynomial_factorization (a : ℝ) : a^2 + 2*a = a*(a+2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3549_354951


namespace NUMINAMATH_CALUDE_josh_pencils_left_josh_pencils_left_proof_l3549_354991

/-- Given that Josh initially had 142 pencils and gave away 31 pencils,
    prove that he has 111 pencils left. -/
theorem josh_pencils_left : ℕ → ℕ → ℕ → Prop :=
  fun initial_pencils pencils_given_away pencils_left =>
    initial_pencils = 142 →
    pencils_given_away = 31 →
    pencils_left = initial_pencils - pencils_given_away →
    pencils_left = 111

/-- Proof of the theorem -/
theorem josh_pencils_left_proof : josh_pencils_left 142 31 111 := by
  sorry

end NUMINAMATH_CALUDE_josh_pencils_left_josh_pencils_left_proof_l3549_354991


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l3549_354963

theorem cricket_team_average_age :
  let team_size : ℕ := 11
  let captain_age : ℕ := 26
  let wicket_keeper_age : ℕ := captain_age + 3
  let average_age : ℚ := (team_size : ℚ)⁻¹ * (captain_age + wicket_keeper_age + (team_size - 2) * (average_age - 1))
  average_age = 23 := by sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l3549_354963


namespace NUMINAMATH_CALUDE_candidate_vote_difference_l3549_354922

theorem candidate_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 8000 → 
  candidate_percentage = 35 / 100 → 
  (total_votes : ℚ) * candidate_percentage + 
  (total_votes : ℚ) * (1 - candidate_percentage) = total_votes → 
  (total_votes : ℚ) * (1 - candidate_percentage) - 
  (total_votes : ℚ) * candidate_percentage = 2400 := by
sorry

end NUMINAMATH_CALUDE_candidate_vote_difference_l3549_354922


namespace NUMINAMATH_CALUDE_probability_divisor_of_8_l3549_354957

/-- A fair 8-sided die -/
def fair_8_sided_die : Finset ℕ := Finset.range 8

/-- The set of divisors of 8 -/
def divisors_of_8 : Finset ℕ := {1, 2, 4, 8}

/-- The probability of an event occurring when rolling a fair die -/
def probability (event : Finset ℕ) (die : Finset ℕ) : ℚ :=
  (event ∩ die).card / die.card

theorem probability_divisor_of_8 :
  probability divisors_of_8 fair_8_sided_die = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_divisor_of_8_l3549_354957


namespace NUMINAMATH_CALUDE_expression_eval_at_two_l3549_354952

theorem expression_eval_at_two : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*x + 2
  f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_eval_at_two_l3549_354952


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3549_354975

theorem quadratic_inequality_solution (a b c : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → 0 ≤ a * x^2 + b * x + c ∧ a * x^2 + b * x + c ≤ 1) →
  a > 0 →
  b = -a →
  c = -2 * a + 1 →
  0 < a ∧ a ≤ 4/9 →
  3 * a + 2 * b + c ≠ 1/3 ∧ 3 * a + 2 * b + c ≠ 5/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3549_354975


namespace NUMINAMATH_CALUDE_max_profit_optimal_plan_model_b_units_l3549_354964

/-- Represents the profit function for tablet sales -/
def profit_function (x : ℕ) : ℝ := -100 * x + 10000

/-- Represents the total cost function for tablet purchases -/
def total_cost (x : ℕ) : ℝ := 1600 * x + 2500 * (20 - x)

/-- Theorem stating the maximum profit and optimal purchasing plan -/
theorem max_profit_optimal_plan :
  ∃ (x : ℕ),
    x ≤ 20 ∧
    total_cost x ≤ 39200 ∧
    profit_function x ≥ 8500 ∧
    (∀ (y : ℕ), y ≤ 20 → total_cost y ≤ 39200 → profit_function y ≥ 8500 →
      profit_function x ≥ profit_function y) ∧
    x = 12 ∧
    profit_function x = 8800 :=
by sorry

/-- Corollary stating the number of units for model B tablets -/
theorem model_b_units (x : ℕ) (h : x = 12) : 20 - x = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_profit_optimal_plan_model_b_units_l3549_354964


namespace NUMINAMATH_CALUDE_min_value_theorem_l3549_354927

def is_arithmetic_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r q : ℝ, r > 0 ∧ q > 1 ∧ ∀ n : ℕ, a (n + 1) - a n = r * q^n

theorem min_value_theorem (a : ℕ → ℝ) (h1 : is_arithmetic_geometric a)
  (h2 : a 9 = a 8 + 2 * a 7) (p q : ℕ) (h3 : a p * a q = 8 * (a 1)^2) :
  (1 : ℝ) / p + 4 / q ≥ 9 / 5 ∧ ∃ p₀ q₀ : ℕ, (1 : ℝ) / p₀ + 4 / q₀ = 9 / 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3549_354927


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l3549_354976

theorem divisibility_equivalence (m n : ℤ) :
  (17 ∣ (2 * m + 3 * n)) ↔ (17 ∣ (9 * m + 5 * n)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l3549_354976


namespace NUMINAMATH_CALUDE_complement_A_in_U_eq_l3549_354987

-- Define the universal set U
def U : Set ℝ := {x : ℝ | x > 1}

-- Define the set A
def A : Set ℝ := {x : ℝ | x > 2}

-- Define the complement of A in U
def complement_A_in_U : Set ℝ := {x : ℝ | x ∈ U ∧ x ∉ A}

-- Theorem statement
theorem complement_A_in_U_eq : complement_A_in_U = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_eq_l3549_354987


namespace NUMINAMATH_CALUDE_shortest_distance_l3549_354958

/-- The shortest distance between two points given their x and y displacements -/
theorem shortest_distance (x_displacement y_displacement : ℝ) :
  x_displacement = 4 →
  y_displacement = 3 →
  Real.sqrt (x_displacement ^ 2 + y_displacement ^ 2) = 5 := by
sorry

end NUMINAMATH_CALUDE_shortest_distance_l3549_354958


namespace NUMINAMATH_CALUDE_intersection_coordinates_l3549_354917

/-- A line perpendicular to the x-axis passing through a point -/
structure VerticalLine where
  x : ℝ

/-- A line perpendicular to the y-axis passing through a point -/
structure HorizontalLine where
  y : ℝ

/-- The intersection point of a vertical and a horizontal line -/
def intersectionPoint (v : VerticalLine) (h : HorizontalLine) : ℝ × ℝ :=
  (v.x, h.y)

/-- Theorem: The intersection of the specific vertical and horizontal lines -/
theorem intersection_coordinates :
  let v := VerticalLine.mk (-3)
  let h := HorizontalLine.mk (-3)
  intersectionPoint v h = (-3, -3) := by
  sorry

end NUMINAMATH_CALUDE_intersection_coordinates_l3549_354917


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_lines_l3549_354988

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m - 1) * x + 2 * y - m = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := x + m * y + m - 2 = 0

-- Define perpendicularity condition
def perpendicular (m : ℝ) : Prop := (m - 1) / 2 * (1 / m) = -1

-- Define parallelism condition
def parallel (m : ℝ) : Prop := (m - 1) / 2 = 1 / m

-- Theorem for perpendicular lines
theorem perpendicular_lines (m : ℝ) : perpendicular m → m = 1/3 := by sorry

-- Theorem for parallel lines
theorem parallel_lines (m : ℝ) : parallel m → m = 2 ∨ m = -1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_lines_l3549_354988


namespace NUMINAMATH_CALUDE_sequence_formula_l3549_354919

def sequence_a (n : ℕ) : ℝ := 2 * n - 1

theorem sequence_formula :
  (sequence_a 1 = 1) ∧
  (∀ n : ℕ, sequence_a n - sequence_a (n + 1) + 2 = 0) →
  ∀ n : ℕ, sequence_a n = 2 * n - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l3549_354919


namespace NUMINAMATH_CALUDE_triangular_pyramids_from_prism_l3549_354995

/-- The number of vertices in a triangular prism -/
def triangular_prism_vertices : ℕ := 6

/-- The number of vertices required to form a triangular pyramid -/
def triangular_pyramid_vertices : ℕ := 4

/-- The number of distinct triangular pyramids that can be formed using the vertices of a triangular prism -/
def distinct_triangular_pyramids : ℕ := 12

theorem triangular_pyramids_from_prism :
  distinct_triangular_pyramids = 12 :=
sorry

end NUMINAMATH_CALUDE_triangular_pyramids_from_prism_l3549_354995


namespace NUMINAMATH_CALUDE_time_after_2017_minutes_l3549_354990

def add_minutes (hours minutes add_minutes : ℕ) : ℕ × ℕ :=
  let total_minutes := hours * 60 + minutes + add_minutes
  let new_hours := (total_minutes / 60) % 24
  let new_minutes := total_minutes % 60
  (new_hours, new_minutes)

theorem time_after_2017_minutes : 
  add_minutes 20 17 2017 = (5, 54) := by
sorry

end NUMINAMATH_CALUDE_time_after_2017_minutes_l3549_354990


namespace NUMINAMATH_CALUDE_chord_length_squared_l3549_354959

/-- Two circles with given radii and center distance, intersecting at point P with equal chords QP and PR --/
structure IntersectingCircles where
  r1 : ℝ
  r2 : ℝ
  center_distance : ℝ
  chord_length : ℝ
  h1 : r1 = 10
  h2 : r2 = 7
  h3 : center_distance = 15
  h4 : chord_length > 0

/-- The square of the chord length in the given configuration is 154 --/
theorem chord_length_squared (ic : IntersectingCircles) : ic.chord_length ^ 2 = 154 := by
  sorry

#check chord_length_squared

end NUMINAMATH_CALUDE_chord_length_squared_l3549_354959


namespace NUMINAMATH_CALUDE_company_employees_ratio_salary_increase_impact_l3549_354902

theorem company_employees_ratio (M F N : ℕ) : 
  (M : ℚ) / F = 7 / 8 ∧ 
  (N : ℚ) / F = 6 / 8 ∧ 
  ((M + 5 : ℚ) / F = 8 / 9 ∧ (N + 3 : ℚ) / F = 7 / 9) → 
  M = 315 ∧ F = 360 ∧ N = 270 := by
  sorry

theorem salary_increase_impact (T : ℚ) :
  T > 0 → T * (110 / 100) - T = T / 10 := by
  sorry

end NUMINAMATH_CALUDE_company_employees_ratio_salary_increase_impact_l3549_354902


namespace NUMINAMATH_CALUDE_painter_problem_l3549_354967

/-- Given a painting job with a total number of rooms, rooms already painted, and time per room,
    calculate the time needed to paint the remaining rooms. -/
def time_to_paint_remaining (total_rooms : ℕ) (painted_rooms : ℕ) (time_per_room : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

theorem painter_problem :
  let total_rooms : ℕ := 9
  let painted_rooms : ℕ := 5
  let time_per_room : ℕ := 8
  time_to_paint_remaining total_rooms painted_rooms time_per_room = 32 := by
  sorry

end NUMINAMATH_CALUDE_painter_problem_l3549_354967


namespace NUMINAMATH_CALUDE_quilt_remaining_squares_l3549_354921

/-- Given a quilt with 16 squares on each side and 25% of it already sewn,
    prove that the number of remaining squares to sew is 24. -/
theorem quilt_remaining_squares (squares_per_side : ℕ) (percent_sewn : ℚ) : 
  squares_per_side = 16 →
  percent_sewn = 1/4 →
  (2 * squares_per_side : ℕ) - (percent_sewn * (2 * squares_per_side : ℕ) : ℚ).num = 24 := by
  sorry

end NUMINAMATH_CALUDE_quilt_remaining_squares_l3549_354921


namespace NUMINAMATH_CALUDE_average_marks_of_combined_classes_l3549_354946

theorem average_marks_of_combined_classes 
  (class1_size : ℕ) (class1_avg : ℝ) 
  (class2_size : ℕ) (class2_avg : ℝ) : 
  class1_size = 30 → 
  class1_avg = 40 → 
  class2_size = 50 → 
  class2_avg = 60 → 
  let total_students := class1_size + class2_size
  let total_marks := class1_size * class1_avg + class2_size * class2_avg
  (total_marks / total_students : ℝ) = 52.5 := by
sorry

end NUMINAMATH_CALUDE_average_marks_of_combined_classes_l3549_354946


namespace NUMINAMATH_CALUDE_cookies_calculation_l3549_354983

/-- The number of people receiving cookies -/
def num_people : ℕ := 14

/-- The number of cookies each person receives -/
def cookies_per_person : ℕ := 30

/-- The total number of cookies prepared -/
def total_cookies : ℕ := num_people * cookies_per_person

theorem cookies_calculation : total_cookies = 420 := by
  sorry

end NUMINAMATH_CALUDE_cookies_calculation_l3549_354983


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l3549_354943

theorem expression_equals_negative_one (a : ℝ) (ha : a ≠ 0) :
  ∀ y : ℝ, y ≠ a ∧ y ≠ -a →
    (a / (a + y) + y / (a - y)) / (y / (a + y) - a / (a - y)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l3549_354943


namespace NUMINAMATH_CALUDE_circle_point_marking_l3549_354969

/-- The number of points on the circle -/
def n : ℕ := 2021

/-- 
Given n points on a circle, prove that the smallest positive integer b 
such that b(b+1)/2 is divisible by n is 67.
-/
theorem circle_point_marking (b : ℕ) : 
  (∀ k < b, ¬(2 ∣ k * (k + 1) ∧ n ∣ k * (k + 1))) ∧ 
  (2 ∣ b * (b + 1) ∧ n ∣ b * (b + 1)) → 
  b = 67 := by sorry

end NUMINAMATH_CALUDE_circle_point_marking_l3549_354969


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l3549_354938

theorem increase_by_percentage (initial : ℕ) (percentage : ℚ) :
  initial + (initial * percentage) = initial * (1 + percentage) := by sorry

theorem increase_80_by_150_percent :
  80 + (80 * (150 / 100)) = 200 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l3549_354938


namespace NUMINAMATH_CALUDE_arrangement_ratio_l3549_354992

def C : ℕ := Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2)
def T : ℕ := Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2)
def S : ℕ := Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2)
def M : ℕ := Nat.factorial 6 / Nat.factorial 2

theorem arrangement_ratio : (C - T + S) / M = 126 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_ratio_l3549_354992


namespace NUMINAMATH_CALUDE_arithmetic_mean_sum_l3549_354977

theorem arithmetic_mean_sum (x y : ℝ) : 
  let s : Finset ℝ := {6, 13, 18, 4, x, y}
  (s.sum id) / s.card = 12 → x + y = 31 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_sum_l3549_354977


namespace NUMINAMATH_CALUDE_unique_matches_exist_l3549_354940

/-- A graph with 20 vertices and 14 edges where each vertex has degree at least 1 -/
structure TennisGraph where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  edge_count : edges.card = 14
  degree_at_least_one : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≥ 1

/-- A subgraph where each vertex has degree at most 1 -/
def UniqueMatchesSubgraph (G : TennisGraph) :=
  { edges : Finset (Fin 20 × Fin 20) //
    edges ⊆ G.edges ∧
    ∀ v ∈ G.vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≤ 1 }

/-- The main theorem -/
theorem unique_matches_exist (G : TennisGraph) :
  ∃ (S : UniqueMatchesSubgraph G), S.val.card ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_matches_exist_l3549_354940


namespace NUMINAMATH_CALUDE_total_flowers_received_l3549_354901

/-- The number of types of flowers bought -/
def num_flower_types : ℕ := 4

/-- The number of pieces bought for each type of flower -/
def pieces_per_type : ℕ := 40

/-- Theorem: The total number of flowers received by the orphanage is 160 -/
theorem total_flowers_received : num_flower_types * pieces_per_type = 160 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_received_l3549_354901


namespace NUMINAMATH_CALUDE_pictures_in_new_galleries_l3549_354953

/-- The number of pictures Alexander draws for the initial exhibition -/
def initial_pictures : ℕ := 9

/-- The number of new galleries -/
def new_galleries : ℕ := 7

/-- The number of pencils Alexander needs for each picture -/
def pencils_per_picture : ℕ := 5

/-- The number of pencils Alexander needs for signing at each exhibition -/
def pencils_for_signing : ℕ := 3

/-- The total number of pencils Alexander uses -/
def total_pencils : ℕ := 218

/-- The list of pictures requested by each new gallery -/
def new_gallery_requests : List ℕ := [4, 6, 8, 5, 7, 3, 9]

/-- Theorem: The number of pictures hung in the new galleries is 29 -/
theorem pictures_in_new_galleries : 
  (total_pencils - (pencils_for_signing * (new_galleries + 1))) / pencils_per_picture - initial_pictures = 29 := by
  sorry

end NUMINAMATH_CALUDE_pictures_in_new_galleries_l3549_354953


namespace NUMINAMATH_CALUDE_no_solution_exponential_equation_l3549_354962

theorem no_solution_exponential_equation :
  ¬ ∃ y : ℝ, (16 : ℝ) ^ (3 * y) = (64 : ℝ) ^ (2 * y + 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_exponential_equation_l3549_354962


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3549_354980

theorem linear_equation_solution :
  ∃ (x y : ℝ), 2 * x - 3 * y = 5 ∧ x = 1 ∧ y = -1 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3549_354980


namespace NUMINAMATH_CALUDE_certain_number_proof_l3549_354950

theorem certain_number_proof (x : ℝ) (n : ℝ) (h1 : x^2 - 3*x = n) (h2 : x - 4 = 2) : n = 18 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3549_354950


namespace NUMINAMATH_CALUDE_shirts_remaining_l3549_354968

theorem shirts_remaining (initial_shirts sold_shirts : ℕ) 
  (h1 : initial_shirts = 49)
  (h2 : sold_shirts = 21) :
  initial_shirts - sold_shirts = 28 := by
  sorry

end NUMINAMATH_CALUDE_shirts_remaining_l3549_354968


namespace NUMINAMATH_CALUDE_library_wall_leftover_space_l3549_354909

theorem library_wall_leftover_space 
  (wall_length : ℝ) 
  (desk_length : ℝ) 
  (bookcase_length : ℝ) 
  (h1 : wall_length = 15) 
  (h2 : desk_length = 2) 
  (h3 : bookcase_length = 1.5) : 
  ∃ (n : ℕ), 
    n * desk_length + n * bookcase_length ≤ wall_length ∧ 
    (n + 1) * desk_length + (n + 1) * bookcase_length > wall_length ∧
    wall_length - (n * desk_length + n * bookcase_length) = 1 := by
  sorry

#check library_wall_leftover_space

end NUMINAMATH_CALUDE_library_wall_leftover_space_l3549_354909


namespace NUMINAMATH_CALUDE_rational_function_value_l3549_354974

-- Define the property of the function f
def satisfies_equation (f : ℚ → ℚ) : Prop :=
  ∀ x : ℚ, x ≠ 0 → 4 * f (1 / x) + 3 * f x / x = x^3

-- State the theorem
theorem rational_function_value (f : ℚ → ℚ) (h : satisfies_equation f) : 
  f (-3) = -6565 / 189 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l3549_354974


namespace NUMINAMATH_CALUDE_minkyung_height_calculation_l3549_354948

def haeun_height : ℝ := 1.56
def nayeon_height : ℝ := haeun_height - 0.14
def minkyung_height : ℝ := nayeon_height + 0.27

theorem minkyung_height_calculation : minkyung_height = 1.69 := by
  sorry

end NUMINAMATH_CALUDE_minkyung_height_calculation_l3549_354948


namespace NUMINAMATH_CALUDE_sues_mix_nuts_percent_l3549_354945

/-- Sue's trail mix composition -/
structure SuesMix where
  nuts : ℝ
  dried_fruit : ℝ
  dried_fruit_percent : dried_fruit = 70
  sum_to_100 : nuts + dried_fruit = 100

/-- Jane's trail mix composition -/
structure JanesMix where
  nuts : ℝ
  chocolate_chips : ℝ
  nuts_percent : nuts = 60
  chocolate_chips_percent : chocolate_chips = 40
  sum_to_100 : nuts + chocolate_chips = 100

/-- Combined mixture composition -/
structure CombinedMix where
  nuts : ℝ
  dried_fruit : ℝ
  nuts_percent : nuts = 45
  dried_fruit_percent : dried_fruit = 35

/-- Theorem stating that Sue's trail mix contains 30% nuts -/
theorem sues_mix_nuts_percent (s : SuesMix) (j : JanesMix) (c : CombinedMix) : s.nuts = 30 :=
sorry

end NUMINAMATH_CALUDE_sues_mix_nuts_percent_l3549_354945


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l3549_354918

theorem roots_sum_and_product (a b : ℝ) : 
  (a^4 - 6*a^2 - 4*a + 1 = 0) → 
  (b^4 - 6*b^2 - 4*b + 1 = 0) → 
  (a ≠ b) →
  (∀ x : ℝ, x^4 - 6*x^2 - 4*x + 1 = 0 → x = a ∨ x = b) →
  a * b + a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l3549_354918


namespace NUMINAMATH_CALUDE_sequence_sum_l3549_354924

theorem sequence_sum (A B C D E F G H I : ℝ) : 
  D = 7 →
  I = 10 →
  B + C + D = 36 →
  C + D + E = 36 →
  D + E + F = 36 →
  E + F + G = 36 →
  F + G + H = 36 →
  G + H + I = 36 →
  A + I = 17 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l3549_354924


namespace NUMINAMATH_CALUDE_factorization_condition_l3549_354979

theorem factorization_condition (a b c : ℤ) : 
  (∀ x : ℝ, (x - a) * (x - 10) + 1 = (x + b) * (x + c)) ↔ 
  ((a = 8 ∧ b = -9 ∧ c = -9) ∨ (a = 12 ∧ b = -11 ∧ c = -11)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_condition_l3549_354979


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3549_354930

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (4 + 3*I) / (1 + 2*I)
  Complex.im z = -1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3549_354930


namespace NUMINAMATH_CALUDE_samson_utility_solution_l3549_354978

/-- Samson's utility function --/
def utility (math : ℝ) (frisbee : ℝ) : ℝ := math^2 * frisbee

/-- Monday's frisbee hours --/
def monday_frisbee (t : ℝ) : ℝ := t

/-- Monday's math hours --/
def monday_math (t : ℝ) : ℝ := 10 - 2*t

/-- Tuesday's frisbee hours --/
def tuesday_frisbee (t : ℝ) : ℝ := 3 - t

/-- Tuesday's math hours --/
def tuesday_math (t : ℝ) : ℝ := 2*t + 4

theorem samson_utility_solution :
  ∃ (t : ℝ), 
    t > 0 ∧ 
    t < 5 ∧
    utility (monday_math t) (monday_frisbee t) = utility (tuesday_math t) (tuesday_frisbee t) ∧
    t = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_samson_utility_solution_l3549_354978


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3549_354913

theorem rectangle_dimensions (area perimeter long_side : ℝ) 
  (h_area : area = 300)
  (h_perimeter : perimeter = 70)
  (h_long_side : long_side = 20) : 
  ∃ (width : ℝ), 
    area = long_side * width ∧ 
    perimeter = 2 * (long_side + width) ∧
    width = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3549_354913


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l3549_354900

/-- The probability of selecting at least one woman when choosing 4 people at random from a group of 10 men and 5 women -/
theorem probability_at_least_one_woman (total_people : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) :
  total_people = men + women →
  men = 10 →
  women = 5 →
  selected = 4 →
  (1 : ℚ) - (men.choose selected : ℚ) / (total_people.choose selected : ℚ) = 84 / 91 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l3549_354900


namespace NUMINAMATH_CALUDE_paint_needed_for_one_door_l3549_354998

theorem paint_needed_for_one_door 
  (total_doors : ℕ) 
  (pint_cost : ℚ) 
  (gallon_cost : ℚ) 
  (pints_per_gallon : ℕ) 
  (savings : ℚ) 
  (h1 : total_doors = 8)
  (h2 : pint_cost = 8)
  (h3 : gallon_cost = 55)
  (h4 : pints_per_gallon = 8)
  (h5 : savings = 9)
  (h6 : total_doors * pint_cost - gallon_cost = savings) :
  (1 : ℚ) = pints_per_gallon / total_doors := by
sorry

end NUMINAMATH_CALUDE_paint_needed_for_one_door_l3549_354998


namespace NUMINAMATH_CALUDE_water_depth_l3549_354904

/-- The depth of water given heights of two people -/
theorem water_depth (ron_height dean_height water_depth : ℕ) : 
  ron_height = 13 →
  dean_height = ron_height + 4 →
  water_depth = 15 * dean_height →
  water_depth = 255 :=
by
  sorry

#check water_depth

end NUMINAMATH_CALUDE_water_depth_l3549_354904


namespace NUMINAMATH_CALUDE_average_difference_l3549_354955

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 115) 
  (h2 : (b + c) / 2 = 160) : 
  a - c = -90 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l3549_354955


namespace NUMINAMATH_CALUDE_g_composition_of_2_l3549_354971

def g (x : ℕ) : ℕ :=
  if x % 3 = 0 then x / 3 else 5 * x + 2

theorem g_composition_of_2 : g (g (g (g 2))) = 112 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_2_l3549_354971


namespace NUMINAMATH_CALUDE_sheet_length_is_48_l3549_354981

/-- Represents the dimensions and volume of a box made from a rectangular sheet. -/
structure BoxDimensions where
  sheet_length : ℝ
  sheet_width : ℝ
  cut_length : ℝ
  box_volume : ℝ

/-- Calculates the volume of a box given its dimensions. -/
def calculate_box_volume (d : BoxDimensions) : ℝ :=
  (d.sheet_length - 2 * d.cut_length) * (d.sheet_width - 2 * d.cut_length) * d.cut_length

/-- Theorem stating that given the specified conditions, the sheet length must be 48 meters. -/
theorem sheet_length_is_48 (d : BoxDimensions) 
  (h_width : d.sheet_width = 36)
  (h_cut : d.cut_length = 7)
  (h_volume : d.box_volume = 5236)
  (h_vol_calc : d.box_volume = calculate_box_volume d) : 
  d.sheet_length = 48 := by
  sorry

#check sheet_length_is_48

end NUMINAMATH_CALUDE_sheet_length_is_48_l3549_354981
