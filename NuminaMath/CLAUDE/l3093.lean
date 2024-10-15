import Mathlib

namespace NUMINAMATH_CALUDE_equal_exchange_ways_l3093_309305

/-- Represents the number of ways to exchange money -/
def exchange_ways (n a b : ℕ) (use_blue : Bool) : ℕ :=
  sorry

/-- The main theorem stating that the number of ways to exchange is equal for both scenarios -/
theorem equal_exchange_ways (n a b : ℕ) :
  exchange_ways n a b true = exchange_ways n a b false :=
sorry

end NUMINAMATH_CALUDE_equal_exchange_ways_l3093_309305


namespace NUMINAMATH_CALUDE_truck_speed_problem_l3093_309369

theorem truck_speed_problem (v : ℝ) : 
  v > 0 →  -- Truck speed is positive
  (60 * 4 = v * 5) →  -- Car catches up after 4 hours
  v = 48 := by
sorry

end NUMINAMATH_CALUDE_truck_speed_problem_l3093_309369


namespace NUMINAMATH_CALUDE_dog_roaming_area_l3093_309366

/-- The area a dog can roam when tied to a circular pillar -/
theorem dog_roaming_area (leash_length : ℝ) (pillar_radius : ℝ) (roaming_area : ℝ) : 
  leash_length = 10 →
  pillar_radius = 2 →
  roaming_area = π * (leash_length + pillar_radius)^2 →
  roaming_area = 144 * π :=
by sorry

end NUMINAMATH_CALUDE_dog_roaming_area_l3093_309366


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3093_309349

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- Theorem: In a geometric sequence, if a_7 · a_19 = 8, then a_3 · a_23 = 8 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_prod : a 7 * a 19 = 8) : a 3 * a 23 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3093_309349


namespace NUMINAMATH_CALUDE_three_unit_fractions_sum_to_one_l3093_309328

theorem three_unit_fractions_sum_to_one :
  ∀ a b c : ℕ+,
    a ≠ b → b ≠ c → a ≠ c →
    (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ = 1 →
    ({a, b, c} : Set ℕ+) = {2, 3, 6} := by
  sorry

end NUMINAMATH_CALUDE_three_unit_fractions_sum_to_one_l3093_309328


namespace NUMINAMATH_CALUDE_machine_working_time_l3093_309395

/-- The number of shirts made by the machine -/
def total_shirts : ℕ := 196

/-- The number of shirts the machine can make per minute -/
def shirts_per_minute : ℕ := 7

/-- The time worked by the machine in minutes -/
def time_worked : ℕ := total_shirts / shirts_per_minute

theorem machine_working_time : time_worked = 28 := by
  sorry

end NUMINAMATH_CALUDE_machine_working_time_l3093_309395


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3093_309342

def num_flips : ℕ := 12

def favorable_outcomes : ℕ := (
  Nat.choose num_flips 7 + 
  Nat.choose num_flips 8 + 
  Nat.choose num_flips 9 + 
  Nat.choose num_flips 10 + 
  Nat.choose num_flips 11 + 
  Nat.choose num_flips 12
)

def total_outcomes : ℕ := 2^num_flips

theorem coin_flip_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 793 / 2048 :=
by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3093_309342


namespace NUMINAMATH_CALUDE_number_problem_l3093_309387

theorem number_problem (x : ℝ) : 0.4 * x - 11 = 23 → x = 85 := by sorry

end NUMINAMATH_CALUDE_number_problem_l3093_309387


namespace NUMINAMATH_CALUDE_trig_identity_l3093_309358

theorem trig_identity (α : Real) (h : Real.sin (α - π/12) = 1/3) : 
  Real.cos (α + 5*π/12) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3093_309358


namespace NUMINAMATH_CALUDE_root_product_of_equation_l3093_309371

theorem root_product_of_equation : ∃ (x y : ℝ), 
  (Real.sqrt (2 * x^2 + 8 * x + 1) - x = 3) ∧
  (Real.sqrt (2 * y^2 + 8 * y + 1) - y = 3) ∧
  (x ≠ y) ∧ (x * y = -8) := by
  sorry

end NUMINAMATH_CALUDE_root_product_of_equation_l3093_309371


namespace NUMINAMATH_CALUDE_alice_purchases_cost_l3093_309302

/-- The exchange rate from British Pounds to USD -/
def gbp_to_usd : ℝ := 1.25

/-- The exchange rate from Euros to USD -/
def eur_to_usd : ℝ := 1.10

/-- The cost of the book in British Pounds -/
def book_cost_gbp : ℝ := 15

/-- The cost of the souvenir in Euros -/
def souvenir_cost_eur : ℝ := 20

/-- The total cost of Alice's purchases in USD -/
def total_cost_usd : ℝ := book_cost_gbp * gbp_to_usd + souvenir_cost_eur * eur_to_usd

theorem alice_purchases_cost : total_cost_usd = 40.75 := by
  sorry

end NUMINAMATH_CALUDE_alice_purchases_cost_l3093_309302


namespace NUMINAMATH_CALUDE_combination_equality_l3093_309379

theorem combination_equality (x : ℕ) : 
  (Nat.choose 7 x = Nat.choose 6 5 + Nat.choose 6 4) → (x = 2 ∨ x = 5) := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l3093_309379


namespace NUMINAMATH_CALUDE_range_of_a_l3093_309396

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 > -2 * a * x - 8

-- Define proposition q
def q (a : ℝ) : Prop := ∃ (h k r : ℝ), ∀ (x y : ℝ), 
  x^2 + y^2 - 4*x + a = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

-- Main theorem
theorem range_of_a : 
  (∃ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) → 
  {a : ℝ | a < 0 ∨ (a ≥ 4 ∧ a < 8)} = {a : ℝ | p a ∨ q a} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3093_309396


namespace NUMINAMATH_CALUDE_new_basis_from_old_l3093_309390

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem new_basis_from_old (a b c : V) 
  (h : LinearIndependent ℝ ![a, b, c]) 
  (h_span : Submodule.span ℝ {a, b, c} = ⊤) :
  LinearIndependent ℝ ![a + b, b + c, c + a] ∧ 
  Submodule.span ℝ {a + b, b + c, c + a} = ⊤ := by
sorry

end NUMINAMATH_CALUDE_new_basis_from_old_l3093_309390


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3093_309301

theorem fractional_equation_solution :
  ∃ x : ℝ, (1 / x = 2 / (x + 3)) ∧ (x = 3) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3093_309301


namespace NUMINAMATH_CALUDE_solution_to_equation_l3093_309325

theorem solution_to_equation : ∃ x : ℝ, 0.2 * x + (0.6 * 0.8) = 0.56 ∧ x = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3093_309325


namespace NUMINAMATH_CALUDE_eighth_pitch_frequency_l3093_309377

/-- Twelve-tone Equal Temperament system -/
structure TwelveToneEqualTemperament where
  /-- The frequency ratio between consecutive pitches -/
  ratio : ℝ
  /-- The ratio is the twelfth root of 2 -/
  ratio_def : ratio = Real.rpow 2 (1/12)

/-- The frequency of a pitch in the Twelve-tone Equal Temperament system -/
def frequency (system : TwelveToneEqualTemperament) (first_pitch : ℝ) (n : ℕ) : ℝ :=
  first_pitch * (system.ratio ^ (n - 1))

/-- Theorem: The frequency of the eighth pitch is the seventh root of 2 times the first pitch -/
theorem eighth_pitch_frequency (system : TwelveToneEqualTemperament) (f : ℝ) :
  frequency system f 8 = f * Real.rpow 2 (7/12) := by
  sorry

end NUMINAMATH_CALUDE_eighth_pitch_frequency_l3093_309377


namespace NUMINAMATH_CALUDE_line_slope_problem_l3093_309341

theorem line_slope_problem (n : ℝ) : 
  n > 0 → 
  (n - 5) / (2 - n) = 2 * n → 
  n = 2.5 := by
sorry

end NUMINAMATH_CALUDE_line_slope_problem_l3093_309341


namespace NUMINAMATH_CALUDE_negation_distribution_l3093_309356

theorem negation_distribution (m : ℝ) : -(m - 2) = -m + 2 := by
  sorry

end NUMINAMATH_CALUDE_negation_distribution_l3093_309356


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3093_309339

open Real

theorem min_value_trig_expression (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) :
  ∃ (min_val : ℝ), min_val = 9 * sqrt 3 ∧
  ∀ θ', 0 < θ' ∧ θ' < π / 2 →
    3 * sin θ' + 4 * (1 / cos θ') + 2 * sqrt 3 * tan θ' ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3093_309339


namespace NUMINAMATH_CALUDE_decimal_places_in_expression_l3093_309344

-- Define the original number
def original_number : ℝ := 3.456789

-- Define the expression
def expression : ℝ := ((10^4 : ℝ) * original_number)^9

-- Function to count decimal places
def count_decimal_places (x : ℝ) : ℕ :=
  sorry

-- Theorem stating that the number of decimal places in the expression is 2
theorem decimal_places_in_expression :
  count_decimal_places expression = 2 := by
  sorry

end NUMINAMATH_CALUDE_decimal_places_in_expression_l3093_309344


namespace NUMINAMATH_CALUDE_complex_product_real_l3093_309338

theorem complex_product_real (m : ℝ) : 
  (Complex.I : ℂ) * (1 - m * Complex.I) + (m^2 : ℂ) * (1 - m * Complex.I) ∈ Set.range Complex.ofReal → 
  m = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_product_real_l3093_309338


namespace NUMINAMATH_CALUDE_equation_solution_l3093_309362

theorem equation_solution : ∀ (x : ℝ) (number : ℝ),
  x = 4 →
  7 * (x - 1) = number →
  number = 21 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3093_309362


namespace NUMINAMATH_CALUDE_basketball_probabilities_l3093_309348

/-- A series of 6 independent Bernoulli trials with probability of success 1/3 -/
def bernoulli_trials (n : ℕ) (p : ℝ) := n = 6 ∧ p = 1/3

/-- Probability of two failures before the first success -/
def prob_two_failures_before_success (p : ℝ) : ℝ := (1 - p)^2 * p

/-- Probability of exactly k successes in n trials -/
def prob_exactly_k_successes (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- Expected number of successes -/
def expected_successes (n : ℕ) (p : ℝ) : ℝ := n * p

/-- Variance of the number of successes -/
def variance_successes (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem basketball_probabilities (n : ℕ) (p : ℝ) 
  (h : bernoulli_trials n p) : 
  prob_two_failures_before_success p = 4/27 ∧
  prob_exactly_k_successes n 3 p = 160/729 ∧
  expected_successes n p = 2 ∧
  variance_successes n p = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_basketball_probabilities_l3093_309348


namespace NUMINAMATH_CALUDE_division_threefold_change_l3093_309337

theorem division_threefold_change (a b c d : ℤ) (h : a = b * c + d) :
  ∃ (d' : ℤ), (3 * a) = (3 * b) * c + d' ∧ d' = 3 * d :=
sorry

end NUMINAMATH_CALUDE_division_threefold_change_l3093_309337


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l3093_309392

/-- The set of functions satisfying the given conditions -/
def S : Set (ℕ → ℝ) :=
  {f | f 1 = 2 ∧ ∀ n, f (n + 1) ≥ f n ∧ f n ≥ (n / (n + 1 : ℝ)) * f (2 * n)}

/-- The smallest natural number M such that f(n) < M for all f ∈ S and n ∈ ℕ -/
theorem smallest_upper_bound : ∃! M : ℕ, 
  (∀ f ∈ S, ∀ n, f n < M) ∧ 
  (∀ M' : ℕ, (∀ f ∈ S, ∀ n, f n < M') → M ≤ M') :=
by
  use 10
  sorry

#check smallest_upper_bound

end NUMINAMATH_CALUDE_smallest_upper_bound_l3093_309392


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l3093_309389

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 2142 → n + (n + 1) = -93 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l3093_309389


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_2_l3093_309361

theorem factorization_of_2x_squared_minus_2 (x : ℝ) : 2 * x^2 - 2 = 2 * (x - 1) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_2_l3093_309361


namespace NUMINAMATH_CALUDE_value_of_expression_l3093_309383

theorem value_of_expression (a b : ℝ) (h : 2 * a + 4 * b = 3) :
  4 * a + 8 * b - 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3093_309383


namespace NUMINAMATH_CALUDE_no_nonsquare_triple_divisors_l3093_309321

theorem no_nonsquare_triple_divisors : 
  ¬ ∃ (N : ℕ+), (¬ ∃ (m : ℕ+), N = m * m) ∧ 
  (∃ (t : ℕ+), ∀ d : ℕ+, d ∣ N → ∃ (a b : ℕ+), (a ∣ N) ∧ (b ∣ N) ∧ (d * a * b = t)) :=
by sorry

end NUMINAMATH_CALUDE_no_nonsquare_triple_divisors_l3093_309321


namespace NUMINAMATH_CALUDE_fermat_1000_units_digit_l3093_309388

/-- Fermat number F_n is defined as 2^(2^n) + 1 -/
def fermat_number (n : ℕ) : ℕ := 2^(2^n) + 1

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem fermat_1000_units_digit :
  units_digit (fermat_number 1000) = 7 := by sorry

end NUMINAMATH_CALUDE_fermat_1000_units_digit_l3093_309388


namespace NUMINAMATH_CALUDE_one_third_percent_of_150_l3093_309381

theorem one_third_percent_of_150 : (1 / 3 * 1 / 100) * 150 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_one_third_percent_of_150_l3093_309381


namespace NUMINAMATH_CALUDE_minimum_score_needed_l3093_309350

def current_scores : List ℕ := [90, 80, 70, 60, 85]
def score_count : ℕ := current_scores.length
def current_average : ℚ := (current_scores.sum : ℚ) / score_count
def target_increase : ℚ := 3
def new_score_count : ℕ := score_count + 1

theorem minimum_score_needed (x : ℕ) : 
  (((current_scores.sum + x) : ℚ) / new_score_count ≥ current_average + target_increase) ↔ 
  (x ≥ 95) :=
sorry

end NUMINAMATH_CALUDE_minimum_score_needed_l3093_309350


namespace NUMINAMATH_CALUDE_min_value_sum_squares_and_reciprocal_cube_min_value_achievable_l3093_309330

theorem min_value_sum_squares_and_reciprocal_cube (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a^2 + b^2 + c^2 + 1 / (a + b + c)^3 ≥ (1/12)^(1/3) := by
  sorry

theorem min_value_achievable (ε : ℝ) (hε : ε > 0) : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a^2 + b^2 + c^2 + 1 / (a + b + c)^3 < (1/12)^(1/3) + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_and_reciprocal_cube_min_value_achievable_l3093_309330


namespace NUMINAMATH_CALUDE_total_blue_balloons_l3093_309343

/-- The number of blue balloons Joan and Melanie have in total -/
def total_balloons (joan_balloons melanie_balloons : ℕ) : ℕ :=
  joan_balloons + melanie_balloons

/-- Theorem stating that Joan and Melanie have 81 blue balloons in total -/
theorem total_blue_balloons :
  total_balloons 40 41 = 81 := by
  sorry

end NUMINAMATH_CALUDE_total_blue_balloons_l3093_309343


namespace NUMINAMATH_CALUDE_sum_greater_than_double_smaller_l3093_309335

theorem sum_greater_than_double_smaller (a b c : ℝ) 
  (h1 : a > c) (h2 : b > c) : a + b > 2 * c := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_double_smaller_l3093_309335


namespace NUMINAMATH_CALUDE_zero_point_in_interval_l3093_309394

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem zero_point_in_interval :
  ∃ (c : ℝ), 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zero_point_in_interval_l3093_309394


namespace NUMINAMATH_CALUDE_probability_of_event_a_l3093_309386

theorem probability_of_event_a 
  (prob_b : ℝ) 
  (prob_a_and_b : ℝ) 
  (prob_neither_a_nor_b : ℝ) 
  (h1 : prob_b = 0.40)
  (h2 : prob_a_and_b = 0.15)
  (h3 : prob_neither_a_nor_b = 0.5499999999999999) : 
  ∃ (prob_a : ℝ), prob_a = 0.20 := by
  sorry

#check probability_of_event_a

end NUMINAMATH_CALUDE_probability_of_event_a_l3093_309386


namespace NUMINAMATH_CALUDE_no_real_solutions_l3093_309382

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  (3 * x^2) / (x - 2) - (3 * x + 10) / 4 + (9 - 9 * x) / (x - 2) - 3 = 0

-- Theorem stating that the equation has no real solutions
theorem no_real_solutions : ¬∃ x : ℝ, original_equation x :=
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3093_309382


namespace NUMINAMATH_CALUDE_at_least_one_survives_one_of_each_type_survives_l3093_309345

-- Define the survival probabilities
def survival_rate_A : ℚ := 5/6
def survival_rate_B : ℚ := 4/5

-- Define the number of trees of each type
def num_trees_A : ℕ := 2
def num_trees_B : ℕ := 2

-- Define the total number of trees
def total_trees : ℕ := num_trees_A + num_trees_B

-- Theorem for the probability that at least one tree survives
theorem at_least_one_survives :
  1 - (1 - survival_rate_A)^num_trees_A * (1 - survival_rate_B)^num_trees_B = 899/900 := by
  sorry

-- Theorem for the probability that one tree of each type survives
theorem one_of_each_type_survives :
  num_trees_A * survival_rate_A * (1 - survival_rate_A) *
  num_trees_B * survival_rate_B * (1 - survival_rate_B) = 4/45 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_survives_one_of_each_type_survives_l3093_309345


namespace NUMINAMATH_CALUDE_brown_mm_averages_l3093_309399

def brown_smiley_counts : List Nat := [9, 12, 8, 8, 3]
def brown_star_counts : List Nat := [7, 14, 11, 6, 10]

theorem brown_mm_averages :
  let smiley_avg := (brown_smiley_counts.sum : ℚ) / brown_smiley_counts.length
  let star_avg := (brown_star_counts.sum : ℚ) / brown_star_counts.length
  smiley_avg = 8 ∧ star_avg = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_brown_mm_averages_l3093_309399


namespace NUMINAMATH_CALUDE_cos_75_degrees_l3093_309304

theorem cos_75_degrees :
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_degrees_l3093_309304


namespace NUMINAMATH_CALUDE_net_rate_of_pay_l3093_309333

/-- Calculates the net rate of pay for a driver given travel conditions and expenses. -/
theorem net_rate_of_pay
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (pay_rate : ℝ)
  (gasoline_cost : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_rate = 0.60)
  (h5 : gasoline_cost = 2.50) :
  (pay_rate * speed * travel_time - (speed * travel_time / fuel_efficiency) * gasoline_cost) / travel_time = 25 := by
  sorry


end NUMINAMATH_CALUDE_net_rate_of_pay_l3093_309333


namespace NUMINAMATH_CALUDE_find_m_l3093_309373

theorem find_m : ∃ m : ℚ, 
  (∀ x : ℚ, 4 * x + 2 * m = 5 * x + 1 ↔ 3 * x = 6 * x - 1) → 
  m = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3093_309373


namespace NUMINAMATH_CALUDE_hyperbola_trisect_foci_eccentricity_l3093_309363

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Theorem: If the vertices of a hyperbola trisect the line segment between its foci,
    then its eccentricity is 3 -/
theorem hyperbola_trisect_foci_eccentricity (a b : ℝ) (h : Hyperbola a b) 
    (trisect : ∃ (c : ℝ), c = 3 * a) : eccentricity h = 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_trisect_foci_eccentricity_l3093_309363


namespace NUMINAMATH_CALUDE_additional_as_needed_l3093_309332

/-- Given initial grades and A's, and subsequent increases in A proportion,
    calculate additional A's needed for a further increase. -/
theorem additional_as_needed
  (n k : ℕ)  -- Initial number of grades and A's
  (h1 : (k + 1 : ℚ) / (n + 1) - k / n = 15 / 100)  -- First increase
  (h2 : (k + 2 : ℚ) / (n + 2) - (k + 1) / (n + 1) = 1 / 10)  -- Second increase
  (h3 : (k + 2 : ℚ) / (n + 2) = 2 / 3)  -- Current proportion
  : ∃ m : ℕ, (k + 2 + m : ℚ) / (n + 2 + m) = 7 / 10 ∧ m = 4 := by
  sorry


end NUMINAMATH_CALUDE_additional_as_needed_l3093_309332


namespace NUMINAMATH_CALUDE_pure_imaginary_iff_x_eq_one_l3093_309378

theorem pure_imaginary_iff_x_eq_one (x : ℝ) :
  x = 1 ↔ (Complex.mk (x^2 - 1) (x + 1)).im ≠ 0 ∧ (Complex.mk (x^2 - 1) (x + 1)).re = 0 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_iff_x_eq_one_l3093_309378


namespace NUMINAMATH_CALUDE_max_sum_cubes_l3093_309393

theorem max_sum_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  ∃ (M : ℝ), M = 5 * Real.sqrt 5 ∧ a^3 + b^3 + c^3 + d^3 + e^3 ≤ M ∧
  ∃ (a' b' c' d' e' : ℝ), a'^2 + b'^2 + c'^2 + d'^2 + e'^2 = 5 ∧
                           a'^3 + b'^3 + c'^3 + d'^3 + e'^3 = M :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_cubes_l3093_309393


namespace NUMINAMATH_CALUDE_vector_problem_l3093_309307

theorem vector_problem (α β : ℝ) (a b c : ℝ × ℝ) :
  a = (Real.cos α, Real.sin α) →
  b = (Real.cos β, Real.sin β) →
  c = (1, 2) →
  a.1 * b.1 + a.2 * b.2 = Real.sqrt 2 / 2 →
  ∃ (k : ℝ), a = k • c →
  0 < β →
  β < α →
  α < Real.pi / 2 →
  Real.cos (α - β) = Real.sqrt 2 / 2 ∧ Real.cos β = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3093_309307


namespace NUMINAMATH_CALUDE_farm_animals_l3093_309384

theorem farm_animals (horses cows : ℕ) : 
  horses = 6 * cows →  -- Initial ratio of horses to cows is 6:1
  (horses - 15) = 3 * (cows + 15) →  -- New ratio after transaction is 3:1
  (horses - 15) - (cows + 15) = 70 := by  -- Difference after transaction is 70
sorry

end NUMINAMATH_CALUDE_farm_animals_l3093_309384


namespace NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l3093_309303

theorem smallest_x_for_perfect_cube (x : ℕ) : x = 36 ↔ 
  (x > 0 ∧ ∃ y : ℕ, 1152 * x = y^3 ∧ ∀ z < x, z > 0 → ¬∃ w : ℕ, 1152 * z = w^3) :=
sorry

end NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l3093_309303


namespace NUMINAMATH_CALUDE_population_change_l3093_309370

/-- The initial population of a village that underwent several population changes --/
def initial_population : ℕ :=
  -- Define the initial population (to be proved)
  6496

/-- The final population after a series of events --/
def final_population : ℕ :=
  -- Given final population
  4555

/-- Theorem stating the relationship between initial and final population --/
theorem population_change (P : ℕ) :
  P = initial_population →
  (1.10 : ℝ) * ((0.75 : ℝ) * ((0.85 : ℝ) * P)) = final_population := by
  sorry


end NUMINAMATH_CALUDE_population_change_l3093_309370


namespace NUMINAMATH_CALUDE_black_area_after_four_changes_l3093_309353

/-- Represents the fraction of black area remaining after a certain number of changes --/
def blackAreaFraction (changes : ℕ) : ℚ :=
  (3/4) ^ changes

/-- The number of changes applied to the triangle --/
def totalChanges : ℕ := 4

/-- Theorem stating that after four changes, the fraction of the original area that remains black is 81/256 --/
theorem black_area_after_four_changes :
  blackAreaFraction totalChanges = 81/256 := by
  sorry

#eval blackAreaFraction totalChanges

end NUMINAMATH_CALUDE_black_area_after_four_changes_l3093_309353


namespace NUMINAMATH_CALUDE_tourist_cyclist_speed_l3093_309367

/-- Given the conditions of a tourist and cyclist problem, prove their speeds -/
theorem tourist_cyclist_speed :
  -- Distance from A to B
  let distance : ℝ := 24

  -- Time difference between tourist and cyclist start
  let time_diff : ℝ := 4/3

  -- Time for cyclist to overtake tourist
  let overtake_time : ℝ := 1/2

  -- Time between first and second encounter
  let encounter_interval : ℝ := 3/2

  -- Speed of cyclist
  let v_cyclist : ℝ := 16.5

  -- Speed of tourist
  let v_tourist : ℝ := 4.5

  -- Equations based on the problem conditions
  (v_cyclist * overtake_time = v_tourist * (time_diff + overtake_time)) ∧
  (v_cyclist * 2 + v_tourist * (time_diff + overtake_time + encounter_interval) = 2 * distance)

  -- Conclusion: The speeds satisfy the equations
  → v_cyclist = 16.5 ∧ v_tourist = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_tourist_cyclist_speed_l3093_309367


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3093_309376

theorem triangle_perimeter (a b c : ℝ) : 
  a = 2 ∧ b = 5 ∧ 
  c^2 - 8*c + 12 = 0 ∧
  a + b > c ∧ a + c > b ∧ b + c > a →
  a + b + c = 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3093_309376


namespace NUMINAMATH_CALUDE_negation_equivalence_l3093_309323

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 5*x₀ + 6 > 0) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5*x + 6 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3093_309323


namespace NUMINAMATH_CALUDE_solve_for_x_l3093_309391

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l3093_309391


namespace NUMINAMATH_CALUDE_value_of_T_l3093_309309

theorem value_of_T : ∃ T : ℝ, (1/3 : ℝ) * (1/6 : ℝ) * T = (1/4 : ℝ) * (1/8 : ℝ) * 120 ∧ T = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_T_l3093_309309


namespace NUMINAMATH_CALUDE_power_function_property_l3093_309372

/-- A power function is a function of the form f(x) = x^n for some real number n. -/
def PowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℝ, ∀ x : ℝ, f x = x ^ n

/-- A function lies in the first and third quadrants if it's positive for positive x
    and negative for negative x. -/
def LiesInFirstAndThirdQuadrants (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (x > 0 → f x > 0) ∧ (x < 0 → f x < 0)

theorem power_function_property
  (f : ℝ → ℝ)
  (h_power : PowerFunction f)
  (h_quadrants : LiesInFirstAndThirdQuadrants f)
  (h_inequality : f 3 < f 2) :
  f (-3) > f (-2) := by
  sorry

end NUMINAMATH_CALUDE_power_function_property_l3093_309372


namespace NUMINAMATH_CALUDE_special_triangle_angles_l3093_309365

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the specific triangle
def SpecialTriangle (t : Triangle) : Prop :=
  t.C = 2 * t.A ∧ t.b = 2 * t.a

-- Theorem statement
theorem special_triangle_angles (t : Triangle) 
  (h : SpecialTriangle t) : 
  t.A = 30 ∧ t.B = 90 ∧ t.C = 60 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_angles_l3093_309365


namespace NUMINAMATH_CALUDE_angle_covered_in_three_layers_l3093_309352

/-- Given a 90-degree angle covered by some angles with the same vertex in two or three layers,
    if the sum of the angles is 290 degrees, then the measure of the angle covered in three layers is 20 degrees. -/
theorem angle_covered_in_three_layers 
  (total_angle : ℝ) 
  (sum_of_angles : ℝ) 
  (angle_covered_three_layers : ℝ) 
  (angle_covered_two_layers : ℝ) 
  (h1 : total_angle = 90)
  (h2 : sum_of_angles = 290)
  (h3 : angle_covered_three_layers + angle_covered_two_layers = total_angle)
  (h4 : 3 * angle_covered_three_layers + 2 * angle_covered_two_layers = sum_of_angles) :
  angle_covered_three_layers = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_covered_in_three_layers_l3093_309352


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l3093_309334

theorem geometric_series_ratio (a r : ℝ) (h1 : r ≠ 1) : 
  (∃ (S : ℝ), S = a / (1 - r) ∧ S = 18) →
  (∃ (S_odd : ℝ), S_odd = a * r / (1 - r^2) ∧ S_odd = 6) →
  r = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l3093_309334


namespace NUMINAMATH_CALUDE_distance_from_origin_l3093_309357

theorem distance_from_origin (x y : ℝ) (h1 : y = 15) 
  (h2 : Real.sqrt ((x - 2)^2 + (y - 8)^2) = 13) (h3 : x > 2) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (349 + 8 * Real.sqrt 30) := by
sorry

end NUMINAMATH_CALUDE_distance_from_origin_l3093_309357


namespace NUMINAMATH_CALUDE_half_correct_probability_l3093_309312

def num_questions : ℕ := 10
def num_correct : ℕ := 5
def probability_correct : ℚ := 1/2

theorem half_correct_probability :
  (Nat.choose num_questions num_correct) * (probability_correct ^ num_correct) * ((1 - probability_correct) ^ (num_questions - num_correct)) = 63/256 := by
  sorry

end NUMINAMATH_CALUDE_half_correct_probability_l3093_309312


namespace NUMINAMATH_CALUDE_average_stickers_per_pack_l3093_309329

def sticker_counts : List ℕ := [5, 7, 9, 9, 11, 15, 15, 17, 19, 21]

def total_stickers : ℕ := sticker_counts.sum

def num_packs : ℕ := sticker_counts.length

theorem average_stickers_per_pack :
  (total_stickers : ℚ) / num_packs = 12.8 := by
  sorry

end NUMINAMATH_CALUDE_average_stickers_per_pack_l3093_309329


namespace NUMINAMATH_CALUDE_function_determination_l3093_309320

theorem function_determination (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → f ((x - 2) / (x + 1)) + f ((3 + x) / (1 - x)) = x) →
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → f x = (x^3 + 7*x) / (2 - 2*x^2)) := by
sorry

end NUMINAMATH_CALUDE_function_determination_l3093_309320


namespace NUMINAMATH_CALUDE_suitcase_weight_problem_l3093_309322

/-- Proves that given the initial ratio of books : clothes : electronics as 5 : 4 : 2, 
    and after removing 9 pounds of clothing, which doubles the ratio of books to clothes, 
    the weight of electronics is 9 pounds. -/
theorem suitcase_weight_problem (B C E : ℝ) : 
  B / C = 5 / 4 →  -- Initial ratio of books to clothes
  B / E = 5 / 2 →  -- Initial ratio of books to electronics
  B / (C - 9) = 10 / 4 →  -- New ratio after removing 9 pounds of clothes
  E = 9 := by
  sorry


end NUMINAMATH_CALUDE_suitcase_weight_problem_l3093_309322


namespace NUMINAMATH_CALUDE_franks_books_l3093_309398

theorem franks_books (a b c : ℤ) (n : ℕ) (p d t : ℕ) :
  p = 2 * a →
  d = 3 * b →
  t = 2 * c * (3 * b) →
  n * p = t →
  n * d = t →
  ∃ (k : ℤ), n = 2 * k ∧ k = c := by
  sorry

end NUMINAMATH_CALUDE_franks_books_l3093_309398


namespace NUMINAMATH_CALUDE_difference_of_max_min_F_l3093_309346

-- Define the function F
def F (x y : ℝ) : ℝ := 4 * x + y

-- State the theorem
theorem difference_of_max_min_F :
  ∀ x y : ℝ, x > 0 → y > 0 → 4 * x + 1 / x + y + 9 / y = 26 →
  (∃ (max min : ℝ), (∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 * x' + 1 / x' + y' + 9 / y' = 26 → F x' y' ≤ max) ∧
                    (∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 * x' + 1 / x' + y' + 9 / y' = 26 → F x' y' ≥ min) ∧
                    (max - min = 24)) :=
by sorry

end NUMINAMATH_CALUDE_difference_of_max_min_F_l3093_309346


namespace NUMINAMATH_CALUDE_cricket_average_increase_l3093_309359

/-- Proves that the increase in average runs per innings is 5 -/
theorem cricket_average_increase
  (initial_average : ℝ)
  (initial_innings : ℕ)
  (next_innings_runs : ℝ)
  (h1 : initial_average = 32)
  (h2 : initial_innings = 20)
  (h3 : next_innings_runs = 137) :
  let total_runs := initial_average * initial_innings
  let new_innings := initial_innings + 1
  let new_total_runs := total_runs + next_innings_runs
  let new_average := new_total_runs / new_innings
  new_average - initial_average = 5 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_increase_l3093_309359


namespace NUMINAMATH_CALUDE_right_triangle_count_l3093_309311

/-- Count of right triangles with integer leg lengths a and b, hypotenuse b+2, and b < 50 -/
theorem right_triangle_count : 
  (Finset.filter (fun p : ℕ × ℕ => 
    let a := p.1
    let b := p.2
    a * a + b * b = (b + 2) * (b + 2) ∧ 
    0 < a ∧ 
    0 < b ∧ 
    b < 50
  ) (Finset.product (Finset.range 200) (Finset.range 50))).card = 7 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_count_l3093_309311


namespace NUMINAMATH_CALUDE_systematic_sampling_l3093_309317

theorem systematic_sampling (total_students : Nat) (sample_size : Nat) (part_size : Nat) (first_drawn : Nat) :
  total_students = 1000 →
  sample_size = 50 →
  part_size = 20 →
  first_drawn = 15 →
  (third_drawn : Nat) = 55 :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_l3093_309317


namespace NUMINAMATH_CALUDE_parabola_symmetric_points_l3093_309375

/-- A parabola with parameter p > 0 has two distinct points symmetrical with respect to the line x + y = 1 if and only if 0 < p < 2/3 -/
theorem parabola_symmetric_points (p : ℝ) :
  (p > 0) →
  (∃ (A B : ℝ × ℝ),
    A ≠ B ∧
    (A.2)^2 = 2*p*A.1 ∧
    (B.2)^2 = 2*p*B.1 ∧
    (∃ (C : ℝ × ℝ),
      C.1 + C.2 = 1 ∧
      C.1 = (A.1 + B.1) / 2 ∧
      C.2 = (A.2 + B.2) / 2)) ↔
  (0 < p ∧ p < 2/3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetric_points_l3093_309375


namespace NUMINAMATH_CALUDE_vector_sum_equality_l3093_309331

variable (V : Type*) [AddCommGroup V]

theorem vector_sum_equality (a : V) : a + 2 • a = 3 • a := by sorry

end NUMINAMATH_CALUDE_vector_sum_equality_l3093_309331


namespace NUMINAMATH_CALUDE_set_partition_real_line_l3093_309380

theorem set_partition_real_line (m : ℝ) : 
  let A := {x : ℝ | x ≥ 3}
  let B := {x : ℝ | x < m}
  (A ∪ B = Set.univ) → (A ∩ B = ∅) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_set_partition_real_line_l3093_309380


namespace NUMINAMATH_CALUDE_average_book_width_l3093_309308

def book_widths : List ℝ := [6, 50, 1, 35, 3, 5, 75, 20]

theorem average_book_width :
  let total_width := book_widths.sum
  let num_books := book_widths.length
  total_width / num_books = 24.375 := by
sorry

end NUMINAMATH_CALUDE_average_book_width_l3093_309308


namespace NUMINAMATH_CALUDE_never_sunday_date_l3093_309360

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a month of the year -/
inductive Month
| January
| February
| March
| April
| May
| June
| July
| August
| September
| October
| November
| December

/-- Function to determine the day of the week for a given date in a month -/
def dayOfWeek (date : Nat) (month : Month) (isLeapYear : Bool) : DayOfWeek :=
  sorry

/-- Theorem stating that 31 is the only date that can never be a Sunday in any month of a year -/
theorem never_sunday_date :
  ∀ (date : Nat),
    (∀ (month : Month) (isLeapYear : Bool),
      dayOfWeek date month isLeapYear ≠ DayOfWeek.Sunday) ↔ date = 31 :=
by sorry

end NUMINAMATH_CALUDE_never_sunday_date_l3093_309360


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l3093_309315

theorem cubic_roots_sum (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) →
  (b^3 - 2*b^2 + 3*b - 4 = 0) →
  (c^3 - 2*c^2 + 3*c - 4 = 0) →
  a ≠ b → b ≠ c → a ≠ c →
  1/(a*(b^2 + c^2 - a^2)) + 1/(b*(c^2 + a^2 - b^2)) + 1/(c*(a^2 + b^2 - c^2)) = -1/8 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l3093_309315


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_three_l3093_309310

theorem subset_implies_m_equals_three (A B : Set ℝ) (m : ℝ) :
  A = {1, 3} →
  B = {1, 2, m} →
  A ⊆ B →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_three_l3093_309310


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l3093_309364

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 2 ∧
  (∀ (y : ℝ), y * |y| = 3 * y + 2 → x ≤ y) ∧
  x = -2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l3093_309364


namespace NUMINAMATH_CALUDE_exists_abs_neq_self_l3093_309300

theorem exists_abs_neq_self : ∃ a : ℝ, |a| ≠ a := by
  sorry

end NUMINAMATH_CALUDE_exists_abs_neq_self_l3093_309300


namespace NUMINAMATH_CALUDE_max_cube_sum_under_constraints_l3093_309368

theorem max_cube_sum_under_constraints {a b c d : ℝ} 
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 20)
  (sum_linear : a + b + c + d = 10) :
  a^3 + b^3 + c^3 + d^3 ≤ 500 ∧ 
  ∃ (x y z w : ℝ), x^2 + y^2 + z^2 + w^2 = 20 ∧ 
                   x + y + z + w = 10 ∧ 
                   x^3 + y^3 + z^3 + w^3 = 500 :=
by sorry

end NUMINAMATH_CALUDE_max_cube_sum_under_constraints_l3093_309368


namespace NUMINAMATH_CALUDE_super_k_conference_l3093_309318

theorem super_k_conference (n : ℕ) : 
  (n * (n - 1)) / 2 = 45 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_super_k_conference_l3093_309318


namespace NUMINAMATH_CALUDE_m_greater_than_one_l3093_309314

theorem m_greater_than_one (m : ℝ) : (∀ x : ℝ, |x| ≤ 1 → x < m) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_one_l3093_309314


namespace NUMINAMATH_CALUDE_unique_divisor_perfect_square_l3093_309319

theorem unique_divisor_perfect_square (p n : ℕ) (hp : Prime p) (hp2 : p > 2) :
  ∃! d : ℕ, d ∣ (p * n^2) ∧ ∃ m : ℕ, n^2 + d = m^2 :=
sorry

end NUMINAMATH_CALUDE_unique_divisor_perfect_square_l3093_309319


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3093_309397

theorem sphere_surface_area (r : ℝ) (h : r = 3) : 4 * π * r^2 = 36 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3093_309397


namespace NUMINAMATH_CALUDE_equation_solution_set_l3093_309326

theorem equation_solution_set : ∃ (S : Set ℝ), 
  S = {x : ℝ | (1 / (x^2 + 8*x - 12) + 1 / (x^2 + 5*x - 12) + 1 / (x^2 - 10*x - 12) = 0)} ∧ 
  S = {Real.sqrt 12, -Real.sqrt 12, 4, 3} := by
sorry

end NUMINAMATH_CALUDE_equation_solution_set_l3093_309326


namespace NUMINAMATH_CALUDE_bonnets_per_orphanage_l3093_309374

/-- The number of bonnets made on Monday -/
def monday_bonnets : ℕ := 10

/-- The number of bonnets made on Tuesday and Wednesday combined -/
def tuesday_wednesday_bonnets : ℕ := 2 * monday_bonnets

/-- The number of bonnets made on Thursday -/
def thursday_bonnets : ℕ := monday_bonnets + 5

/-- The number of bonnets made on Friday -/
def friday_bonnets : ℕ := thursday_bonnets - 5

/-- The total number of bonnets made -/
def total_bonnets : ℕ := monday_bonnets + tuesday_wednesday_bonnets + thursday_bonnets + friday_bonnets

/-- The number of orphanages -/
def num_orphanages : ℕ := 5

/-- Theorem stating the number of bonnets sent to each orphanage -/
theorem bonnets_per_orphanage : total_bonnets / num_orphanages = 11 := by
  sorry

end NUMINAMATH_CALUDE_bonnets_per_orphanage_l3093_309374


namespace NUMINAMATH_CALUDE_banana_arrangements_count_l3093_309324

/-- The number of distinct arrangements of the letters in the word BANANA -/
def banana_arrangements : ℕ := 180

/-- The total number of letters in the word BANANA -/
def total_letters : ℕ := 6

/-- The number of A's in the word BANANA -/
def num_a : ℕ := 3

/-- The number of N's in the word BANANA -/
def num_n : ℕ := 2

/-- The number of B's in the word BANANA -/
def num_b : ℕ := 1

/-- Theorem stating that the number of distinct arrangements of the letters in BANANA is 180 -/
theorem banana_arrangements_count :
  banana_arrangements = Nat.factorial total_letters / (Nat.factorial num_a * Nat.factorial num_n) :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangements_count_l3093_309324


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3093_309313

-- Define the equation
def equation (m : ℝ) (x : ℝ) : Prop := x^2 - 2*m*x + 3*m = 0

-- Define the isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  base_positive : base > 0
  side_positive : side > 0
  triangle_inequality : 2 * side > base

-- Theorem statement
theorem isosceles_triangle_perimeter : ∃ (m : ℝ) (t : IsoscelesTriangle),
  equation m 2 ∧ 
  (equation m t.base ∨ equation m t.side) ∧
  (t.base = 2 ∨ t.side = 2) ∧
  t.base + 2 * t.side = 14 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3093_309313


namespace NUMINAMATH_CALUDE_max_value_expression_l3093_309347

theorem max_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 + y - Real.sqrt (x^4 + y^2)) / x ≤ (1 : ℝ) / 2 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ (x₀^2 + y₀ - Real.sqrt (x₀^4 + y₀^2)) / x₀ = (1 : ℝ) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l3093_309347


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3093_309385

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  2 * X^4 + 10 * X^3 - 45 * X^2 - 55 * X + 52 = 
  (X^2 + 8 * X - 6) * q + (-211 * X + 142) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3093_309385


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l3093_309340

theorem smallest_solution_of_equation (x : ℝ) : 
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → 
  (x ≥ 4 - Real.sqrt 2 ∧ 
   (1 / ((4 - Real.sqrt 2) - 3) + 1 / ((4 - Real.sqrt 2) - 5) = 4 / ((4 - Real.sqrt 2) - 4))) := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l3093_309340


namespace NUMINAMATH_CALUDE_sum_of_f_symmetric_points_sum_of_roots_l3093_309354

-- Define the cubic function f
def f (x : ℝ) : ℝ := x^3 + 2*x - 1

-- Theorem 1
theorem sum_of_f_symmetric_points (x₁ x₂ : ℝ) (h : x₁ + x₂ = 0) : 
  f x₁ + f x₂ = -2 := by sorry

-- Theorem 2
theorem sum_of_roots (m n : ℝ) 
  (hm : m^3 - 3*m^2 + 5*m - 4 = 0) 
  (hn : n^3 - 3*n^2 + 5*n - 2 = 0) : 
  m + n = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_f_symmetric_points_sum_of_roots_l3093_309354


namespace NUMINAMATH_CALUDE_log_ratio_squared_l3093_309355

theorem log_ratio_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, x = Real.log a ∧ y = Real.log b ∧ 2 * x^2 - 4 * x + 1 = 0 ∧ 2 * y^2 - 4 * y + 1 = 0) →
  (Real.log (a / b))^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l3093_309355


namespace NUMINAMATH_CALUDE_max_value_of_complex_sum_l3093_309351

theorem max_value_of_complex_sum (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z + 1 + Complex.I * Real.sqrt 3) ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_complex_sum_l3093_309351


namespace NUMINAMATH_CALUDE_xiaoming_pe_grade_l3093_309336

/-- Calculates the semester physical education grade based on given scores and weights -/
def calculate_semester_grade (extracurricular_score midterm_score final_score : ℚ) 
  (extracurricular_weight midterm_weight final_weight : ℕ) : ℚ :=
  (extracurricular_score * extracurricular_weight + 
   midterm_score * midterm_weight + 
   final_score * final_weight) / 
  (extracurricular_weight + midterm_weight + final_weight)

/-- Xiaoming's physical education grade theorem -/
theorem xiaoming_pe_grade :
  let max_score : ℚ := 100
  let extracurricular_score : ℚ := 95
  let midterm_score : ℚ := 90
  let final_score : ℚ := 85
  let extracurricular_weight : ℕ := 2
  let midterm_weight : ℕ := 4
  let final_weight : ℕ := 4
  calculate_semester_grade extracurricular_score midterm_score final_score
    extracurricular_weight midterm_weight final_weight = 89 := by
  sorry


end NUMINAMATH_CALUDE_xiaoming_pe_grade_l3093_309336


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3093_309306

def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3093_309306


namespace NUMINAMATH_CALUDE_problem_1_l3093_309327

theorem problem_1 : (1) - 2 + 8 - (-30) = 36 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3093_309327


namespace NUMINAMATH_CALUDE_range_of_sum_l3093_309316

theorem range_of_sum (a b c : ℝ) 
  (sum_condition : a + b + c = 1) 
  (square_sum_condition : a^2 + b^2 + c^2 = 1) : 
  0 ≤ a + b ∧ a + b ≤ 4/3 := by
sorry

end NUMINAMATH_CALUDE_range_of_sum_l3093_309316
