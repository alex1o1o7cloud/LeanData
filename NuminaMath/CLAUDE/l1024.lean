import Mathlib

namespace NUMINAMATH_CALUDE_pushup_difference_l1024_102446

theorem pushup_difference (david_pushups : ℕ) (total_pushups : ℕ) (zachary_pushups : ℕ) :
  david_pushups = 51 →
  total_pushups = 53 →
  david_pushups > zachary_pushups →
  total_pushups = david_pushups + zachary_pushups →
  david_pushups - zachary_pushups = 49 := by
  sorry

end NUMINAMATH_CALUDE_pushup_difference_l1024_102446


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1024_102469

def i : ℂ := Complex.I

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7 * i
  let z₂ : ℂ := 4 - 7 * i
  (z₁ / z₂) + (z₂ / z₁) = -66 / 65 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1024_102469


namespace NUMINAMATH_CALUDE_science_team_selection_l1024_102481

def number_of_boys : ℕ := 7
def number_of_girls : ℕ := 9
def boys_in_team : ℕ := 2
def girls_in_team : ℕ := 3

theorem science_team_selection :
  (number_of_boys.choose boys_in_team) * (number_of_girls.choose girls_in_team) = 1764 := by
  sorry

end NUMINAMATH_CALUDE_science_team_selection_l1024_102481


namespace NUMINAMATH_CALUDE_frustum_volume_l1024_102477

theorem frustum_volume (r₁ r₂ : ℝ) (h l : ℝ) : 
  r₂ = 4 * r₁ →
  l = 5 →
  π * (r₁ + r₂) * l = 25 * π →
  (1/3) * π * h * (r₁^2 + r₂^2 + r₁*r₂) = 28 * π :=
by sorry

end NUMINAMATH_CALUDE_frustum_volume_l1024_102477


namespace NUMINAMATH_CALUDE_rope_length_problem_l1024_102408

theorem rope_length_problem (short_rope : ℝ) (long_rope : ℝ) : 
  short_rope = 150 →
  short_rope = long_rope * (1 - 1/8) →
  long_rope = 1200/7 := by
sorry

end NUMINAMATH_CALUDE_rope_length_problem_l1024_102408


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l1024_102483

theorem unique_congruence_in_range : ∃! n : ℕ, 3 ≤ n ∧ n ≤ 10 ∧ n % 7 = 10573 % 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l1024_102483


namespace NUMINAMATH_CALUDE_least_value_x_minus_y_plus_z_l1024_102490

theorem least_value_x_minus_y_plus_z (x y z : ℕ+) (h : (3 : ℕ) * x.val = (4 : ℕ) * y.val ∧ (4 : ℕ) * y.val = (7 : ℕ) * z.val) :
  (x.val - y.val + z.val : ℤ) ≥ 19 ∧ ∃ (x₀ y₀ z₀ : ℕ+), (3 : ℕ) * x₀.val = (4 : ℕ) * y₀.val ∧ (4 : ℕ) * y₀.val = (7 : ℕ) * z₀.val ∧ (x₀.val - y₀.val + z₀.val : ℤ) = 19 :=
sorry

end NUMINAMATH_CALUDE_least_value_x_minus_y_plus_z_l1024_102490


namespace NUMINAMATH_CALUDE_parabola_intersection_fixed_point_l1024_102414

-- Define the parabola E
def E (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the lines l₁ and l₂
def l₁ (k₁ x y : ℝ) : Prop := y = k₁*(x - 1)
def l₂ (k₂ x y : ℝ) : Prop := y = k₂*(x - 1)

-- Define the line l
def l (k k₁ k₂ x y : ℝ) : Prop := k*x - y - k*k₁ - k*k₂ = 0

theorem parabola_intersection_fixed_point 
  (p : ℝ) (k₁ k₂ k : ℝ) :
  E p 4 0 ∧ -- This represents y² = 8x, derived from the minimum value condition
  k₁ * k₂ = -3/2 ∧
  k = (4/k₁ - 4/k₂) / ((k₁^2 + 4)/k₁^2 - (k₂^2 + 4)/k₂^2) →
  l k k₁ k₂ 0 (3/2) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_fixed_point_l1024_102414


namespace NUMINAMATH_CALUDE_scheduleArrangements_eq_180_l1024_102418

/-- The number of ways to schedule 4 out of 6 people over 3 days -/
def scheduleArrangements : ℕ :=
  Nat.choose 6 1 * Nat.choose 5 1 * Nat.choose 4 2

/-- Theorem stating that the number of scheduling arrangements is 180 -/
theorem scheduleArrangements_eq_180 : scheduleArrangements = 180 := by
  sorry

end NUMINAMATH_CALUDE_scheduleArrangements_eq_180_l1024_102418


namespace NUMINAMATH_CALUDE_positive_integer_solutions_count_l1024_102452

theorem positive_integer_solutions_count : 
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s ↔ p.1 > 0 ∧ p.2 > 0 ∧ (4 : ℚ) / p.1 + (2 : ℚ) / p.2 = 1) ∧
    s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_count_l1024_102452


namespace NUMINAMATH_CALUDE_probability_five_green_marbles_l1024_102406

/-- The probability of drawing exactly k successes in n trials 
    with probability p for each success -/
def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

/-- The number of green marbles in the bag -/
def greenMarbles : ℕ := 9

/-- The number of purple marbles in the bag -/
def purpleMarbles : ℕ := 6

/-- The total number of marbles in the bag -/
def totalMarbles : ℕ := greenMarbles + purpleMarbles

/-- The probability of drawing a green marble -/
def pGreen : ℚ := greenMarbles / totalMarbles

/-- The number of marbles drawn -/
def numDraws : ℕ := 8

/-- The number of green marbles we want to draw -/
def numGreenDraws : ℕ := 5

theorem probability_five_green_marbles :
  binomialProbability numDraws numGreenDraws pGreen = 108864 / 390625 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_green_marbles_l1024_102406


namespace NUMINAMATH_CALUDE_intersection_eq_A_intersection_nonempty_l1024_102475

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a + 1}
def B : Set ℝ := {x | x < -1 ∨ x > 3}

-- Theorem for the first question
theorem intersection_eq_A (a : ℝ) : A a ∩ B = A a ↔ a ≤ -2 ∨ a ≥ 1 := by sorry

-- Theorem for the second question
theorem intersection_nonempty (a : ℝ) : (A a ∩ B).Nonempty ↔ a < -1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_eq_A_intersection_nonempty_l1024_102475


namespace NUMINAMATH_CALUDE_project_completion_time_l1024_102493

/-- Given a project that A can complete in 20 days, and A and B together can complete in 15 days
    with A quitting 5 days before completion, prove that B can complete the project alone in 30 days. -/
theorem project_completion_time (a_rate b_rate : ℚ) : 
  a_rate = (1 : ℚ) / 20 →                          -- A's work rate
  a_rate + b_rate = (1 : ℚ) / 15 →                 -- A and B's combined work rate
  10 * (a_rate + b_rate) + 5 * b_rate = 1 →        -- Total work done
  b_rate = (1 : ℚ) / 30                            -- B's work rate (reciprocal of completion time)
  := by sorry

end NUMINAMATH_CALUDE_project_completion_time_l1024_102493


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l1024_102401

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1/a > 1/b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l1024_102401


namespace NUMINAMATH_CALUDE_polly_tweet_time_l1024_102453

/-- Represents the number of tweets per minute in different emotional states -/
structure TweetsPerMinute where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Represents the total number of tweets and the time spent in each state -/
structure TweetData where
  tweets_per_minute : TweetsPerMinute
  total_tweets : ℕ
  time_per_state : ℕ

/-- Theorem: Given Polly's tweet rates and total tweets, prove the time spent in each state -/
theorem polly_tweet_time (data : TweetData)
  (h1 : data.tweets_per_minute.happy = 18)
  (h2 : data.tweets_per_minute.hungry = 4)
  (h3 : data.tweets_per_minute.mirror = 45)
  (h4 : data.total_tweets = 1340)
  (h5 : data.time_per_state * (data.tweets_per_minute.happy + data.tweets_per_minute.hungry + data.tweets_per_minute.mirror) = data.total_tweets) :
  data.time_per_state = 20 := by
  sorry


end NUMINAMATH_CALUDE_polly_tweet_time_l1024_102453


namespace NUMINAMATH_CALUDE_equation_solution_l1024_102444

theorem equation_solution : ∃ x : ℝ, (4 : ℝ) ^ (x + 3) = 64 ^ x ∧ x = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1024_102444


namespace NUMINAMATH_CALUDE_simplify_power_sum_l1024_102430

theorem simplify_power_sum : 
  -(2^2004) + (-2)^2005 + 2^2006 - 2^2007 = -(2^2004) - 2^2005 + 2^2006 - 2^2007 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_sum_l1024_102430


namespace NUMINAMATH_CALUDE_chord_length_polar_l1024_102466

theorem chord_length_polar (ρ θ : ℝ) : 
  ρ = 4 * Real.sin θ → θ = π / 4 → ρ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_polar_l1024_102466


namespace NUMINAMATH_CALUDE_sum_divides_ten_n_count_l1024_102464

theorem sum_divides_ten_n_count : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ (10 * n) % (n * (n + 1) / 2) = 0) ∧
    (∀ n : ℕ, n > 0 ∧ (10 * n) % (n * (n + 1) / 2) = 0 → n ∈ S) ∧
    Finset.card S = 5 :=
by sorry

end NUMINAMATH_CALUDE_sum_divides_ten_n_count_l1024_102464


namespace NUMINAMATH_CALUDE_n_pointed_star_angle_sum_l1024_102476

/-- A structure representing an n-pointed star formed from a convex polygon. -/
structure NPointedStar where
  n : ℕ
  n_ge_5 : n ≥ 5

/-- The sum of interior angles at the n points of an n-pointed star. -/
def interior_angle_sum (star : NPointedStar) : ℝ :=
  180 * (star.n - 4)

/-- Theorem stating that the sum of interior angles of an n-pointed star is 180(n-4) degrees. -/
theorem n_pointed_star_angle_sum (star : NPointedStar) :
  interior_angle_sum star = 180 * (star.n - 4) := by
  sorry

end NUMINAMATH_CALUDE_n_pointed_star_angle_sum_l1024_102476


namespace NUMINAMATH_CALUDE_decimal_21_equals_binary_10101_l1024_102442

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem decimal_21_equals_binary_10101 : 
  to_binary 21 = [true, false, true, false, true] ∧ from_binary [true, false, true, false, true] = 21 := by
  sorry

end NUMINAMATH_CALUDE_decimal_21_equals_binary_10101_l1024_102442


namespace NUMINAMATH_CALUDE_triangle_abc_area_l1024_102437

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is (√3 + 1) / 2 under the given conditions. -/
theorem triangle_abc_area (A B C : Real) (a b c : Real) :
  b = Real.sqrt 2 * a →
  Real.sqrt 3 * Real.cos B = Real.sqrt 2 * Real.cos A →
  c = Real.sqrt 3 + 1 →
  (1/2) * a * c * Real.sin B = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_area_l1024_102437


namespace NUMINAMATH_CALUDE_dogwood_trees_to_cut_l1024_102450

/-- The number of dogwood trees in the first part of the park -/
def trees_part1 : ℝ := 5.0

/-- The number of dogwood trees in the second part of the park -/
def trees_part2 : ℝ := 4.0

/-- The number of dogwood trees that will be left after the work is done -/
def trees_left : ℝ := 2.0

/-- The number of dogwood trees to be cut down -/
def trees_to_cut : ℝ := trees_part1 + trees_part2 - trees_left

theorem dogwood_trees_to_cut :
  trees_to_cut = 7.0 := by sorry

end NUMINAMATH_CALUDE_dogwood_trees_to_cut_l1024_102450


namespace NUMINAMATH_CALUDE_box_width_l1024_102417

/-- The width of a rectangular box with given dimensions and filling rate -/
theorem box_width (fill_rate : ℝ) (length depth : ℝ) (fill_time : ℝ) :
  fill_rate = 4 →
  length = 7 →
  depth = 2 →
  fill_time = 21 →
  (fill_rate * fill_time) / (length * depth) = 6 :=
by sorry

end NUMINAMATH_CALUDE_box_width_l1024_102417


namespace NUMINAMATH_CALUDE_intersection_empty_implies_t_geq_one_l1024_102471

theorem intersection_empty_implies_t_geq_one (t : ℝ) : 
  let M : Set ℝ := {x | x ≤ 1}
  let P : Set ℝ := {x | x > t}
  (M ∩ P = ∅) → t ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_t_geq_one_l1024_102471


namespace NUMINAMATH_CALUDE_smallest_four_digit_congruent_to_one_mod_23_l1024_102447

theorem smallest_four_digit_congruent_to_one_mod_23 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≡ 1 [MOD 23] → 1013 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_congruent_to_one_mod_23_l1024_102447


namespace NUMINAMATH_CALUDE_smallest_n_proof_l1024_102429

/-- The smallest possible value of n given the conditions -/
def smallest_n : ℕ := 400

theorem smallest_n_proof (a b c m n : ℕ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : a + b + c = 2003)
  (h3 : a = 2 * b)
  (h4 : a.factorial * b.factorial * c.factorial = m * (10 ^ n))
  (h5 : ¬(10 ∣ m)) : 
  n ≥ smallest_n := by
  sorry

#check smallest_n_proof

end NUMINAMATH_CALUDE_smallest_n_proof_l1024_102429


namespace NUMINAMATH_CALUDE_exists_subset_with_unique_adjacency_l1024_102410

def adjacent (p q : ℤ × ℤ × ℤ) : Prop :=
  let (x, y, z) := p
  let (u, v, w) := q
  abs (x - u) + abs (y - v) + abs (z - w) = 1

theorem exists_subset_with_unique_adjacency :
  ∃ (S : Set (ℤ × ℤ × ℤ)), ∀ p : ℤ × ℤ × ℤ,
    (p ∈ S ∧ ∀ q, adjacent p q → q ∉ S) ∨
    (p ∉ S ∧ ∃! q, adjacent p q ∧ q ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_exists_subset_with_unique_adjacency_l1024_102410


namespace NUMINAMATH_CALUDE_solution_equation_one_solution_equation_two_solution_system_equations_l1024_102420

-- Problem 1
theorem solution_equation_one (x : ℝ) : 4 - 3 * x = 6 - 5 * x ↔ x = 1 := by sorry

-- Problem 2
theorem solution_equation_two (x : ℝ) : (x + 1) / 2 - 1 = (2 - x) / 3 ↔ x = 7 / 5 := by sorry

-- Problem 3
theorem solution_system_equations (x y : ℝ) : 3 * x - y = 7 ∧ x + 3 * y = -1 ↔ x = 2 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_solution_equation_one_solution_equation_two_solution_system_equations_l1024_102420


namespace NUMINAMATH_CALUDE_haploid_breeding_shortens_cycle_l1024_102428

/-- Represents a haploid organism -/
structure Haploid where
  chromosomeSets : ℕ
  derivedFromGamete : Bool

/-- Represents the process of haploid breeding -/
structure HaploidBreeding where
  usesPollenGrains : Bool
  inducesChromosomeDoubling : Bool

/-- Represents the outcome of breeding -/
structure BreedingOutcome where
  cycleLength : ℕ
  homozygosity : Bool

/-- Theorem stating that haploid breeding shortens the breeding cycle -/
theorem haploid_breeding_shortens_cycle 
  (h : Haploid) 
  (hb : HaploidBreeding) 
  (outcome : BreedingOutcome) : 
  h.derivedFromGamete ∧ 
  hb.usesPollenGrains ∧ 
  hb.inducesChromosomeDoubling ∧
  outcome.homozygosity → 
  outcome.cycleLength < regular_breeding_cycle_length :=
sorry

/-- The length of a regular breeding cycle -/
def regular_breeding_cycle_length : ℕ := sorry

end NUMINAMATH_CALUDE_haploid_breeding_shortens_cycle_l1024_102428


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l1024_102488

theorem chord_length_concentric_circles (a b : ℝ) (h1 : a > b) (h2 : a^2 - b^2 = 20) :
  ∃ c : ℝ, c = 4 * Real.sqrt 5 ∧ c^2 / 4 + b^2 = a^2 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l1024_102488


namespace NUMINAMATH_CALUDE_max_value_expression_l1024_102427

theorem max_value_expression (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  ∃ (m : ℝ), m = 2 ∧ ∀ x y z w, 
    (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → (0 ≤ z ∧ z ≤ 1) → (0 ≤ w ∧ w ≤ 1) →
    x + y + z + w - x*y - y*z - z*w - w*x ≤ m :=
by
  sorry

#check max_value_expression

end NUMINAMATH_CALUDE_max_value_expression_l1024_102427


namespace NUMINAMATH_CALUDE_oranges_left_l1024_102419

theorem oranges_left (initial_oranges : ℕ) (taken_oranges : ℕ) : 
  initial_oranges = 60 → taken_oranges = 35 → initial_oranges - taken_oranges = 25 := by
  sorry

end NUMINAMATH_CALUDE_oranges_left_l1024_102419


namespace NUMINAMATH_CALUDE_fraction_simplification_l1024_102434

theorem fraction_simplification (x : ℝ) (h : x ≠ 4) :
  (x^2 - 4*x) / (x^2 - 8*x + 16) = x / (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1024_102434


namespace NUMINAMATH_CALUDE_copy_machine_rate_l1024_102486

/-- Given two copy machines working together for 30 minutes to produce 2850 copies,
    with one machine producing 55 copies per minute, prove that the other machine
    produces 40 copies per minute. -/
theorem copy_machine_rate : ∀ (rate1 : ℕ),
  (30 * rate1 + 30 * 55 = 2850) → rate1 = 40 := by
  sorry

end NUMINAMATH_CALUDE_copy_machine_rate_l1024_102486


namespace NUMINAMATH_CALUDE_equation_solution_l1024_102413

theorem equation_solution : ∃! y : ℚ, 7 * (4 * y - 3) + 5 = 3 * (-2 + 8 * y) :=
  by sorry

end NUMINAMATH_CALUDE_equation_solution_l1024_102413


namespace NUMINAMATH_CALUDE_f_divisible_by_8_l1024_102458

def f (n : ℕ) : ℤ := 5 * n + 2 * (-1)^n + 1

theorem f_divisible_by_8 : ∀ n : ℕ, ∃ k : ℤ, f n = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_f_divisible_by_8_l1024_102458


namespace NUMINAMATH_CALUDE_youtube_dislikes_difference_l1024_102421

theorem youtube_dislikes_difference (D : ℕ) : 
  D + 1000 = 2600 → D - D / 2 = 800 := by
  sorry

end NUMINAMATH_CALUDE_youtube_dislikes_difference_l1024_102421


namespace NUMINAMATH_CALUDE_sequence_properties_l1024_102422

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

def T (n : ℕ) : ℝ := sorry

theorem sequence_properties (n : ℕ) :
  n > 0 →
  (S n = 2 * sequence_a n - 2) →
  (sequence_a n = 2^n ∧ T n = 2^(n+2) - 4 - 2*n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1024_102422


namespace NUMINAMATH_CALUDE_two_person_apartments_count_l1024_102400

/-- Represents an apartment complex with identical buildings -/
structure ApartmentComplex where
  numBuildings : ℕ
  studioPerBuilding : ℕ
  fourPersonPerBuilding : ℕ
  totalOccupants : ℕ
  occupancyRate : ℚ

/-- Calculate the number of 2-person apartments in each building -/
def calculateTwoPersonApartments (complex : ApartmentComplex) : ℕ :=
  sorry

/-- Theorem stating that the number of 2-person apartments is 20 for the given complex -/
theorem two_person_apartments_count (complex : ApartmentComplex) :
  complex.numBuildings = 4 →
  complex.studioPerBuilding = 10 →
  complex.fourPersonPerBuilding = 5 →
  complex.occupancyRate = 3/4 →
  complex.totalOccupants = 210 →
  calculateTwoPersonApartments complex = 20 := by
  sorry

end NUMINAMATH_CALUDE_two_person_apartments_count_l1024_102400


namespace NUMINAMATH_CALUDE_olivia_payment_l1024_102457

/-- The number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- The number of quarters Olivia pays for chips -/
def chips_quarters : ℕ := 4

/-- The number of quarters Olivia pays for soda -/
def soda_quarters : ℕ := 12

/-- The total amount Olivia pays in dollars -/
def total_dollars : ℚ := (chips_quarters + soda_quarters) / quarters_per_dollar

theorem olivia_payment :
  total_dollars = 4 := by sorry

end NUMINAMATH_CALUDE_olivia_payment_l1024_102457


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1024_102407

/-- If an arc of 60° on circle X has the same length as an arc of 40° on circle Y,
    then the ratio of the area of circle X to the area of circle Y is 4/9. -/
theorem circle_area_ratio (X Y : Real) (hX : X > 0) (hY : Y > 0) :
  (60 / 360) * (2 * Real.pi * X) = (40 / 360) * (2 * Real.pi * Y) →
  (Real.pi * X^2) / (Real.pi * Y^2) = 4 / 9 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1024_102407


namespace NUMINAMATH_CALUDE_vector_expression_simplification_l1024_102474

variable {V : Type*} [AddCommGroup V]

/-- For any four points A, B, C, D in a vector space,
    the expression AC - BD + CD - AB equals the zero vector. -/
theorem vector_expression_simplification
  (A B C D : V) : (C - A) - (D - B) + (D - C) - (B - A) = (0 : V) := by
  sorry

end NUMINAMATH_CALUDE_vector_expression_simplification_l1024_102474


namespace NUMINAMATH_CALUDE_rosie_pies_theorem_l1024_102478

/-- Represents the number of pies that can be made from a given number of apples -/
def pies_from_apples (apples : ℕ) : ℕ :=
  3 * (apples / 12)

theorem rosie_pies_theorem :
  pies_from_apples 36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pies_theorem_l1024_102478


namespace NUMINAMATH_CALUDE_ring_arrangements_count_l1024_102472

/-- The number of ways to arrange 6 rings out of 10 distinguishable rings on 4 fingers. -/
def ring_arrangements : ℕ :=
  (Nat.choose 10 6) * (Nat.factorial 6) * (Nat.choose 9 3)

/-- The correct number of arrangements is 12672000. -/
theorem ring_arrangements_count : ring_arrangements = 12672000 := by
  sorry

end NUMINAMATH_CALUDE_ring_arrangements_count_l1024_102472


namespace NUMINAMATH_CALUDE_gcd_105_210_l1024_102491

theorem gcd_105_210 : Nat.gcd 105 210 = 105 := by
  sorry

end NUMINAMATH_CALUDE_gcd_105_210_l1024_102491


namespace NUMINAMATH_CALUDE_input_is_input_statement_l1024_102449

-- Define the type for programming language statements
inductive Statement
  | Print
  | Input
  | If
  | Let

-- Define a predicate for input statements
def isInputStatement : Statement → Prop
  | Statement.Input => True
  | _ => False

-- Theorem: INPUT is an input statement
theorem input_is_input_statement : isInputStatement Statement.Input := by
  sorry

end NUMINAMATH_CALUDE_input_is_input_statement_l1024_102449


namespace NUMINAMATH_CALUDE_remaining_ribbon_l1024_102492

/-- Calculates the remaining ribbon length after wrapping gifts -/
theorem remaining_ribbon (num_gifts : ℕ) (ribbon_per_gift : ℚ) (total_ribbon : ℚ) :
  num_gifts = 8 →
  ribbon_per_gift = 3/2 →
  total_ribbon = 15 →
  total_ribbon - (num_gifts : ℚ) * ribbon_per_gift = 3 := by
  sorry

end NUMINAMATH_CALUDE_remaining_ribbon_l1024_102492


namespace NUMINAMATH_CALUDE_y₁_less_than_y₂_l1024_102496

/-- A linear function y = 2x + 1 passing through points (-3, y₁) and (4, y₂) -/
def linear_function (x : ℝ) : ℝ := 2 * x + 1

/-- y₁ is the y-coordinate when x = -3 -/
def y₁ : ℝ := linear_function (-3)

/-- y₂ is the y-coordinate when x = 4 -/
def y₂ : ℝ := linear_function 4

/-- Theorem stating that y₁ < y₂ -/
theorem y₁_less_than_y₂ : y₁ < y₂ := by sorry

end NUMINAMATH_CALUDE_y₁_less_than_y₂_l1024_102496


namespace NUMINAMATH_CALUDE_sales_revenue_error_l1024_102411

theorem sales_revenue_error (x z : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →
  (10 ≤ z ∧ z ≤ 99) →
  (1000 * z + 10 * x) - (1000 * x + 10 * z) = 2920 →
  z = x + 3 ∧ 10 ≤ x ∧ x ≤ 96 :=
by sorry

end NUMINAMATH_CALUDE_sales_revenue_error_l1024_102411


namespace NUMINAMATH_CALUDE_probability_of_selecting_particular_student_l1024_102412

/-- The probability of selecting a particular student from an institute with multiple classes. -/
theorem probability_of_selecting_particular_student
  (total_classes : ℕ)
  (students_per_class : ℕ)
  (selected_students : ℕ)
  (h1 : total_classes = 8)
  (h2 : students_per_class = 40)
  (h3 : selected_students = 3)
  (h4 : selected_students ≤ total_classes) :
  (selected_students : ℚ) / (total_classes * students_per_class : ℚ) = 3 / 320 := by
  sorry

#check probability_of_selecting_particular_student

end NUMINAMATH_CALUDE_probability_of_selecting_particular_student_l1024_102412


namespace NUMINAMATH_CALUDE_point_movement_l1024_102497

/-- Given a point P(-5, -2) in the Cartesian coordinate system, 
    moving it 3 units left and 2 units up results in the point (-8, 0). -/
theorem point_movement : 
  let P : ℝ × ℝ := (-5, -2)
  let left_movement : ℝ := 3
  let up_movement : ℝ := 2
  let new_x : ℝ := P.1 - left_movement
  let new_y : ℝ := P.2 + up_movement
  (new_x, new_y) = (-8, 0) := by sorry

end NUMINAMATH_CALUDE_point_movement_l1024_102497


namespace NUMINAMATH_CALUDE_spadesuit_inequality_not_always_true_l1024_102461

def spadesuit (x y : ℝ) : ℝ := x^2 - y^2

theorem spadesuit_inequality_not_always_true :
  ¬ (∀ x y : ℝ, x ≥ y → spadesuit x y ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_spadesuit_inequality_not_always_true_l1024_102461


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1024_102463

/-- A line is represented by its slope and y-intercept -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Check if a line passes through a point -/
def Line.passes_through (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Convert an equation of the form ax + by = c to slope-intercept form -/
def to_slope_intercept (a b c : ℚ) : Line :=
  { slope := -a / b, intercept := c / b }

theorem parallel_line_through_point 
  (l1 : Line) (p : Point) :
  ∃ (l2 : Line), 
    parallel l1 l2 ∧ 
    l2.passes_through p ∧
    l2.slope = 1/2 ∧ 
    l2.intercept = -2 :=
  sorry

#check parallel_line_through_point

end NUMINAMATH_CALUDE_parallel_line_through_point_l1024_102463


namespace NUMINAMATH_CALUDE_diagonal_length_from_area_and_offsets_l1024_102431

/-- The length of a quadrilateral's diagonal given its area and offsets -/
theorem diagonal_length_from_area_and_offsets (area : ℝ) (offset1 : ℝ) (offset2 : ℝ) :
  area = 90 ∧ offset1 = 5 ∧ offset2 = 4 →
  ∃ (diagonal : ℝ), diagonal = 20 ∧ area = (offset1 + offset2) * diagonal / 2 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_from_area_and_offsets_l1024_102431


namespace NUMINAMATH_CALUDE_grades_theorem_l1024_102438

structure Student :=
  (name : String)
  (gotA : Prop)

def Emily : Student := ⟨"Emily", true⟩
def Fran : Student := ⟨"Fran", true⟩
def George : Student := ⟨"George", true⟩
def Hailey : Student := ⟨"Hailey", false⟩

theorem grades_theorem :
  (Emily.gotA → Fran.gotA) ∧
  (Fran.gotA → George.gotA) ∧
  (George.gotA → ¬Hailey.gotA) ∧
  (Emily.gotA ∧ Fran.gotA ∧ George.gotA ∧ ¬Hailey.gotA) ∧
  (∃! (s : Finset Student), s.card = 3 ∧ ∀ student ∈ s, student.gotA) →
  ∃! (s : Finset Student),
    s.card = 3 ∧
    Emily ∈ s ∧ Fran ∈ s ∧ George ∈ s ∧ Hailey ∉ s ∧
    (∀ student ∈ s, student.gotA) :=
by sorry

end NUMINAMATH_CALUDE_grades_theorem_l1024_102438


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l1024_102402

/-- Given two vectors a and b in ℝ², where a is parallel to (b - a), prove that the x-coordinate of a is -2. -/
theorem vector_parallel_condition (a b : ℝ × ℝ) (h : a.1 = m ∧ a.2 = 1 ∧ b.1 = 2 ∧ b.2 = -1) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a = k • (b - a)) : m = -2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l1024_102402


namespace NUMINAMATH_CALUDE_f_minimum_value_l1024_102485

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp x / x + x - Real.log x

-- State the theorem
theorem f_minimum_value :
  ∃ (min_value : ℝ), min_value = Real.exp 1 + 1 ∧
  ∀ (x : ℝ), x > 0 → f x ≥ min_value :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_l1024_102485


namespace NUMINAMATH_CALUDE_inequality_proof_l1024_102416

theorem inequality_proof (u v w : ℝ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h : u + v + w + Real.sqrt (u * v * w) = 4) :
  Real.sqrt (v * w / u) + Real.sqrt (u * w / v) + Real.sqrt (u * v / w) ≥ u + v + w := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1024_102416


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1024_102424

theorem arithmetic_mean_problem : 
  let a := 3 / 4
  let b := 5 / 8
  let mean := (a + b) / 2
  3 * mean = 33 / 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1024_102424


namespace NUMINAMATH_CALUDE_special_line_equation_l1024_102462

/-- A line passing through (5, 2) with y-intercept twice the x-intercept -/
structure SpecialLine where
  -- Slope-intercept form: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (5, 2)
  point_condition : 2 = m * 5 + b
  -- The y-intercept is twice the x-intercept
  intercept_condition : b = -2 * (b / m)

/-- The equation of the special line is either 2x + y - 12 = 0 or 2x - 5y = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (∀ x y, 2 * x + y - 12 = 0 ↔ y = l.m * x + l.b) ∨
  (∀ x y, 2 * x - 5 * y = 0 ↔ y = l.m * x + l.b) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l1024_102462


namespace NUMINAMATH_CALUDE_bill_fraction_l1024_102436

theorem bill_fraction (total_stickers : ℕ) (andrew_fraction : ℚ) (total_given : ℕ) 
  (h1 : total_stickers = 100)
  (h2 : andrew_fraction = 1 / 5)
  (h3 : total_given = 44) :
  let andrew_stickers := andrew_fraction * total_stickers
  let remaining_after_andrew := total_stickers - andrew_stickers
  let bill_stickers := total_given - andrew_stickers
  bill_stickers / remaining_after_andrew = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_bill_fraction_l1024_102436


namespace NUMINAMATH_CALUDE_hanoi_theorem_l1024_102403

/-- The minimum number of moves to solve the Towers of Hanoi puzzle with n disks -/
def hanoi_moves (n : ℕ) : ℕ := 2^n - 1

/-- The minimum number of moves to solve the Towers of Hanoi puzzle with n disks
    when direct movement between pegs 1 and 3 is prohibited -/
def hanoi_moves_restricted (n : ℕ) : ℕ := 3^n - 1

/-- The minimum number of moves to solve the Towers of Hanoi puzzle with n disks
    when the smallest disk cannot be placed on peg 2 -/
def hanoi_moves_no_small_on_middle (n : ℕ) : ℕ := 2 * 3^(n-1) - 1

theorem hanoi_theorem (n : ℕ) :
  (hanoi_moves n = 2^n - 1) ∧
  (hanoi_moves_restricted n = 3^n - 1) ∧
  (hanoi_moves_no_small_on_middle n = 2 * 3^(n-1) - 1) :=
by sorry

end NUMINAMATH_CALUDE_hanoi_theorem_l1024_102403


namespace NUMINAMATH_CALUDE_complex_absolute_value_l1024_102445

theorem complex_absolute_value (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 2 - I
  (∃ (b : ℝ), z₁ / z₂ = b*I) → Complex.abs z₁ = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l1024_102445


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l1024_102454

theorem complex_product_pure_imaginary (a : ℝ) : 
  (Complex.I + 1) * (Complex.I * a + 1) = Complex.I * (Complex.I.im * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l1024_102454


namespace NUMINAMATH_CALUDE_sinusoidal_function_properties_l1024_102465

/-- Given a sinusoidal function y = A sin(ωx + φ) with A > 0 and ω > 0,
    passing through (π/12, 0) and with nearest highest point at (π/3, 5),
    prove its properties. -/
theorem sinusoidal_function_properties (A ω φ : ℝ) (h_A : A > 0) (h_ω : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ A * Real.sin (ω * x + φ)
  (f (π/12) = 0) →
  (f (π/3) = 5) →
  (∀ x, x ∈ Set.Ioo (π/12) (π/3) → f x < 5) →
  (∃ k : ℤ, ∀ x, f x ≤ 0 ↔ k * π - 5*π/12 ≤ x ∧ x ≤ k * π + π/12) →
  (A = 5 ∧ ω = 2 ∧ φ = -π/6) ∧
  (∀ k : ℤ, Set.Icc (k*π + π/3) (k*π + 5*π/6) ⊆ { x | f x ≥ f (x + π/2) }) ∧
  (∀ x ∈ Set.Icc 0 π, f x ≤ 5) ∧
  (∀ x ∈ Set.Icc 0 π, f x ≥ -5) ∧
  (f (π/3) = 5) ∧
  (f (5*π/6) = -5) := by
sorry

end NUMINAMATH_CALUDE_sinusoidal_function_properties_l1024_102465


namespace NUMINAMATH_CALUDE_books_loaned_out_l1024_102405

theorem books_loaned_out (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : return_rate = 65 / 100)
  (h3 : final_books = 68) : 
  ∃ (loaned_books : ℕ), loaned_books = 20 ∧ 
    final_books = initial_books - (1 - return_rate) * loaned_books :=
by sorry

end NUMINAMATH_CALUDE_books_loaned_out_l1024_102405


namespace NUMINAMATH_CALUDE_cannot_divide_rectangle_l1024_102404

theorem cannot_divide_rectangle : ¬ ∃ (m n : ℕ), 55 = m * 5 ∧ 39 = n * 11 := by
  sorry

end NUMINAMATH_CALUDE_cannot_divide_rectangle_l1024_102404


namespace NUMINAMATH_CALUDE_sin_330_degrees_l1024_102470

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l1024_102470


namespace NUMINAMATH_CALUDE_find_N_l1024_102479

theorem find_N : ∃ N : ℕ, 
  (87^2 - 78^2) % N = 0 ∧ 
  45 < N ∧ 
  N < 100 ∧ 
  (N = 55 ∨ N = 99) := by
sorry

end NUMINAMATH_CALUDE_find_N_l1024_102479


namespace NUMINAMATH_CALUDE_ed_pets_problem_l1024_102432

theorem ed_pets_problem (dogs : ℕ) (cats : ℕ) (fish : ℕ) : 
  cats = 3 → 
  fish = 2 * (dogs + cats) → 
  dogs + cats + fish = 15 → 
  dogs = 2 := by
sorry

end NUMINAMATH_CALUDE_ed_pets_problem_l1024_102432


namespace NUMINAMATH_CALUDE_two_distinct_decorations_l1024_102456

/-- Represents the two types of decorations --/
inductive Decoration
| A
| B

/-- Represents a triangle decoration --/
structure TriangleDecoration :=
  (v1 v2 v3 : Decoration)

/-- Checks if a triangle decoration is valid according to the rules --/
def isValidDecoration (td : TriangleDecoration) : Prop :=
  (td.v1 = td.v2 ∧ td.v3 ≠ td.v1) ∨
  (td.v1 = td.v3 ∧ td.v2 ≠ td.v1) ∨
  (td.v2 = td.v3 ∧ td.v1 ≠ td.v2)

/-- Checks if two triangle decorations are equivalent under rotation or flipping --/
def areEquivalentDecorations (td1 td2 : TriangleDecoration) : Prop :=
  td1 = td2 ∨
  td1 = {v1 := td2.v2, v2 := td2.v3, v3 := td2.v1} ∨
  td1 = {v1 := td2.v3, v2 := td2.v1, v3 := td2.v2} ∨
  td1 = {v1 := td2.v1, v2 := td2.v3, v3 := td2.v2} ∨
  td1 = {v1 := td2.v3, v2 := td2.v2, v3 := td2.v1} ∨
  td1 = {v1 := td2.v2, v2 := td2.v1, v3 := td2.v3}

/-- The main theorem stating that there are exactly two distinct decorations --/
theorem two_distinct_decorations :
  ∃ (d1 d2 : TriangleDecoration),
    isValidDecoration d1 ∧
    isValidDecoration d2 ∧
    ¬(areEquivalentDecorations d1 d2) ∧
    (∀ d : TriangleDecoration, isValidDecoration d →
      (areEquivalentDecorations d d1 ∨ areEquivalentDecorations d d2)) :=
  sorry

end NUMINAMATH_CALUDE_two_distinct_decorations_l1024_102456


namespace NUMINAMATH_CALUDE_inverse_function_sum_l1024_102482

-- Define the functions g and g_inv
def g (c d : ℝ) (x : ℝ) : ℝ := c * x + d
def g_inv (c d : ℝ) (x : ℝ) : ℝ := d * x + c

-- State the theorem
theorem inverse_function_sum (c d : ℝ) :
  (∀ x : ℝ, g c d (g_inv c d x) = x) →
  (∀ x : ℝ, g_inv c d (g c d x) = x) →
  c + d = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_sum_l1024_102482


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1024_102409

theorem polynomial_simplification (x : ℝ) :
  (3 * x^5 - 2 * x^3 + 5 * x^2 - 8 * x + 6) + (7 * x^4 + x^3 - 3 * x^2 + x - 9) =
  3 * x^5 + 7 * x^4 - x^3 + 2 * x^2 - 7 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1024_102409


namespace NUMINAMATH_CALUDE_function_properties_l1024_102489

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b - 8) * x - a - a * b

-- Theorem statement
theorem function_properties :
  ∀ (a b : ℝ),
  (∀ x ∈ Set.Ioo (-3) 2, f a b x > 0) ∧
  (∀ x ∈ Set.Iic (-3) ∪ Set.Ici 2, f a b x < 0) →
  (∃ (c : ℝ), ∀ x : ℝ, -3 * x^2 + 5 * x + c ≤ 0) →
  (∃ (y : ℝ), ∀ x > -1, (f (-3) 5 x - 21) / (x + 1) ≤ y ∧ 
    ∃ x₀ > -1, (f (-3) 5 x₀ - 21) / (x₀ + 1) = y) →
  (∀ x : ℝ, f a b x = -3 * x^2 + 3 * x + 18) ∧
  (∀ c : ℝ, (∀ x : ℝ, -3 * x^2 + 5 * x + c ≤ 0) → c ≤ -25/12) ∧
  (∀ x > -1, (f (-3) 5 x - 21) / (x + 1) ≤ -3 ∧ 
    ∃ x₀ > -1, (f (-3) 5 x₀ - 21) / (x₀ + 1) = -3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1024_102489


namespace NUMINAMATH_CALUDE_parabola_point_distance_l1024_102425

theorem parabola_point_distance (x y : ℝ) :
  x^2 = 4*y →  -- Point (x, y) is on the parabola
  (x^2 + (y - 1)^2 = 9) →  -- Distance from (x, y) to focus (0, 1) is 3
  y = 2 := by  -- The y-coordinate of the point is 2
sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l1024_102425


namespace NUMINAMATH_CALUDE_local_extrema_of_f_l1024_102487

def f (x : ℝ) : ℝ := 1 + 3*x - x^3

theorem local_extrema_of_f :
  (∃ δ₁ > 0, ∀ x ∈ Set.Ioo (-1 - δ₁) (-1 + δ₁), f x ≥ f (-1)) ∧
  (∃ δ₂ > 0, ∀ x ∈ Set.Ioo (1 - δ₂) (1 + δ₂), f x ≤ f 1) ∧
  f (-1) = -1 ∧
  f 1 = 3 :=
sorry

end NUMINAMATH_CALUDE_local_extrema_of_f_l1024_102487


namespace NUMINAMATH_CALUDE_store_pricing_l1024_102423

theorem store_pricing (shirts_total : ℝ) (sweaters_total : ℝ) (jeans_total : ℝ)
  (shirts_count : ℕ) (sweaters_count : ℕ) (jeans_count : ℕ)
  (shirt_discount : ℝ) (sweater_discount : ℝ) (jeans_discount : ℝ)
  (h1 : shirts_total = 360)
  (h2 : sweaters_total = 900)
  (h3 : jeans_total = 1200)
  (h4 : shirts_count = 20)
  (h5 : sweaters_count = 45)
  (h6 : jeans_count = 30)
  (h7 : shirt_discount = 2)
  (h8 : sweater_discount = 4)
  (h9 : jeans_discount = 3) :
  let shirt_avg := (shirts_total / shirts_count) - shirt_discount
  let sweater_avg := (sweaters_total / sweaters_count) - sweater_discount
  let jeans_avg := (jeans_total / jeans_count) - jeans_discount
  shirt_avg = sweater_avg ∧ jeans_avg - sweater_avg = 21 := by
  sorry

end NUMINAMATH_CALUDE_store_pricing_l1024_102423


namespace NUMINAMATH_CALUDE_bisected_polyhedron_edges_l1024_102494

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ

/-- Represents the new polyhedron after bisection -/
structure BisectedPolyhedron where
  original : ConvexPolyhedron
  planes : ℕ

/-- Calculate the number of edges in the bisected polyhedron -/
def edges_after_bisection (T : BisectedPolyhedron) : ℕ :=
  T.original.edges + 2 * T.original.edges

/-- Theorem stating the number of edges in the bisected polyhedron -/
theorem bisected_polyhedron_edges 
  (P : ConvexPolyhedron) 
  (h_vertices : P.vertices = 0)  -- placeholder for actual number of vertices
  (h_edges : P.edges = 150)
  (T : BisectedPolyhedron)
  (h_T : T.original = P)
  (h_planes : T.planes = P.vertices)
  : edges_after_bisection T = 450 := by
  sorry

#check bisected_polyhedron_edges

end NUMINAMATH_CALUDE_bisected_polyhedron_edges_l1024_102494


namespace NUMINAMATH_CALUDE_NaClO_molecular_weight_l1024_102443

/-- The atomic weight of sodium in g/mol -/
def sodium_weight : ℝ := 22.99

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := 35.45

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of NaClO in g/mol -/
def NaClO_weight : ℝ := sodium_weight + chlorine_weight + oxygen_weight

/-- Theorem stating that the molecular weight of NaClO is approximately 74.44 g/mol -/
theorem NaClO_molecular_weight : 
  ‖NaClO_weight - 74.44‖ < 0.01 := by sorry

end NUMINAMATH_CALUDE_NaClO_molecular_weight_l1024_102443


namespace NUMINAMATH_CALUDE_charity_fundraising_l1024_102460

theorem charity_fundraising (total : ℕ) (people : ℕ) (raised : ℕ) (target : ℕ) :
  total = 2100 →
  people = 8 →
  raised = 150 →
  target = 279 →
  (total - raised) / (people - 1) = target := by
  sorry

end NUMINAMATH_CALUDE_charity_fundraising_l1024_102460


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_under_hundred_l1024_102440

theorem largest_multiple_of_seven_under_hundred : 
  ∃ (n : ℕ), n = 98 ∧ 
  7 ∣ n ∧ 
  n < 100 ∧ 
  ∀ (m : ℕ), 7 ∣ m → m < 100 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_under_hundred_l1024_102440


namespace NUMINAMATH_CALUDE_vector_simplification_l1024_102435

/-- Given points A, B, C, and O in 3D space, 
    prove that AB + OC - OB = AC -/
theorem vector_simplification 
  (A B C O : EuclideanSpace ℝ (Fin 3)) : 
  (B - A) + (C - O) - (B - O) = C - A := by sorry

end NUMINAMATH_CALUDE_vector_simplification_l1024_102435


namespace NUMINAMATH_CALUDE_polynomial_subtraction_simplification_l1024_102473

theorem polynomial_subtraction_simplification :
  ∀ x : ℝ, (2 * x^6 + x^5 + 3 * x^4 + x^2 + 15) - (x^6 + 2 * x^5 - x^4 + x^3 + 17) = 
            x^6 - x^5 + 4 * x^4 - x^3 + x^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_subtraction_simplification_l1024_102473


namespace NUMINAMATH_CALUDE_cubic_inequality_false_l1024_102455

theorem cubic_inequality_false : 
  ¬(∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
sorry

end NUMINAMATH_CALUDE_cubic_inequality_false_l1024_102455


namespace NUMINAMATH_CALUDE_number_of_bowls_l1024_102441

/-- The number of bowls on the table -/
def num_bowls : ℕ := sorry

/-- The initial number of grapes in each bowl -/
def initial_grapes : ℕ → ℕ := sorry

/-- The total number of grapes initially -/
def total_initial_grapes : ℕ := sorry

/-- The number of bowls that receive additional grapes -/
def bowls_with_added_grapes : ℕ := 12

/-- The number of grapes added to each of the specified bowls -/
def grapes_added_per_bowl : ℕ := 8

/-- The increase in the average number of grapes across all bowls -/
def average_increase : ℕ := 6

theorem number_of_bowls :
  (total_initial_grapes + bowls_with_added_grapes * grapes_added_per_bowl) / num_bowls =
  total_initial_grapes / num_bowls + average_increase →
  num_bowls = 16 := by sorry

end NUMINAMATH_CALUDE_number_of_bowls_l1024_102441


namespace NUMINAMATH_CALUDE_joan_attended_games_l1024_102499

theorem joan_attended_games (total_games missed_games : ℕ) 
  (h1 : total_games = 864) 
  (h2 : missed_games = 469) : 
  total_games - missed_games = 395 := by
  sorry

end NUMINAMATH_CALUDE_joan_attended_games_l1024_102499


namespace NUMINAMATH_CALUDE_mod_equivalence_l1024_102468

theorem mod_equivalence (n : ℕ) (h1 : n < 41) (h2 : (5 * n) % 41 = 1) :
  (((2 ^ n) ^ 3) - 3) % 41 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_l1024_102468


namespace NUMINAMATH_CALUDE_statement_equivalence_l1024_102467

theorem statement_equivalence (x y : ℝ) : 
  ((abs y < abs x) ↔ (x^2 > y^2)) ∧ 
  ((x^3 - y^3 = 0) ↔ (x - y = 0)) ∧ 
  ((x^3 - y^3 ≠ 0) ↔ (x - y ≠ 0)) ∧ 
  ¬((x^2 - y^2 ≠ 0 ∧ x^3 - y^3 ≠ 0) ↔ (x^2 - y^2 ≠ 0 ∨ x^3 - y^3 ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_statement_equivalence_l1024_102467


namespace NUMINAMATH_CALUDE_sum_reciprocal_inequality_l1024_102448

theorem sum_reciprocal_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c = 1/a + 1/b + 1/c) : a + b + c ≥ 3/(a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_inequality_l1024_102448


namespace NUMINAMATH_CALUDE_complex_product_real_l1024_102498

theorem complex_product_real (a : ℝ) : 
  let z₁ : ℂ := 2 - I
  let z₂ : ℂ := a + 2*I
  (z₁ * z₂).im = 0 → a = 4 := by sorry

end NUMINAMATH_CALUDE_complex_product_real_l1024_102498


namespace NUMINAMATH_CALUDE_equality_proof_l1024_102484

theorem equality_proof : 2017 - 1 / 2017 = (2018 * 2016) / 2017 := by
  sorry

end NUMINAMATH_CALUDE_equality_proof_l1024_102484


namespace NUMINAMATH_CALUDE_solve_for_k_l1024_102459

/-- A function that represents the linearity condition of the equation -/
def is_linear (k : ℤ) : Prop := ∀ x y : ℝ, ∃ a b c : ℝ, 2 * x^(|k|) + (k - 1) * y = a * x + b * y + c

/-- The main theorem stating the conditions and the conclusion about the value of k -/
theorem solve_for_k (k : ℤ) (h1 : is_linear k) (h2 : k - 1 ≠ 0) : k = -1 := by
  sorry


end NUMINAMATH_CALUDE_solve_for_k_l1024_102459


namespace NUMINAMATH_CALUDE_circle_area_theorem_l1024_102426

theorem circle_area_theorem (c : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ π * r^2 = (c + 4 * Real.sqrt 3) * π / 3) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l1024_102426


namespace NUMINAMATH_CALUDE_positive_interval_l1024_102480

theorem positive_interval (x : ℝ) : (x + 3) * (x - 2) > 0 ↔ x < -3 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_interval_l1024_102480


namespace NUMINAMATH_CALUDE_expansion_coefficient_l1024_102451

/-- The coefficient of x^5 in the expansion of (ax^2 + 1/√x)^5 -/
def coefficient_x5 (a : ℝ) : ℝ := 10 * a^3

theorem expansion_coefficient (a : ℝ) : 
  coefficient_x5 a = -80 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l1024_102451


namespace NUMINAMATH_CALUDE_parallelogram_height_l1024_102439

/-- The height of a parallelogram with given area and base -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (h_area : area = 288) (h_base : base = 18) :
  area / base = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l1024_102439


namespace NUMINAMATH_CALUDE_seagull_fraction_l1024_102415

theorem seagull_fraction (initial_seagulls : ℕ) (scared_fraction : ℚ) (remaining_seagulls : ℕ) :
  initial_seagulls = 36 →
  scared_fraction = 1/4 →
  remaining_seagulls = 18 →
  (initial_seagulls - initial_seagulls * scared_fraction : ℚ) - remaining_seagulls = 
  (1/3) * (initial_seagulls - initial_seagulls * scared_fraction) :=
by
  sorry

end NUMINAMATH_CALUDE_seagull_fraction_l1024_102415


namespace NUMINAMATH_CALUDE_min_sum_products_l1024_102495

theorem min_sum_products (m n : ℕ) : 
  (m * (m - 1)) / ((m + n) * (m + n - 1)) = 1 / 2 →
  m ≥ 1 →
  n ≥ 1 →
  m + n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_products_l1024_102495


namespace NUMINAMATH_CALUDE_equation_exists_solution_l1024_102433

theorem equation_exists_solution (x : ℝ) (hx : x = 2407) :
  ∃ (y z : ℝ), x^y + y^x = z :=
sorry

end NUMINAMATH_CALUDE_equation_exists_solution_l1024_102433
