import Mathlib

namespace NUMINAMATH_CALUDE_weight_loss_challenge_l843_84344

theorem weight_loss_challenge (initial_weight : ℝ) (h_initial_weight_pos : initial_weight > 0) :
  let weight_after_loss := initial_weight * (1 - 0.11)
  let measured_weight_loss_percentage := 0.0922
  ∃ (clothes_weight_percentage : ℝ),
    weight_after_loss * (1 + clothes_weight_percentage) = initial_weight * (1 - measured_weight_loss_percentage) ∧
    clothes_weight_percentage = 0.02 :=
by sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l843_84344


namespace NUMINAMATH_CALUDE_sqrt_calculations_l843_84359

theorem sqrt_calculations : 
  (2 * Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 3 * Real.sqrt 2) ∧ 
  ((Real.sqrt 12 - Real.sqrt 24) / Real.sqrt 6 - 2 * Real.sqrt (1/2) = -2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l843_84359


namespace NUMINAMATH_CALUDE_parking_cost_theorem_l843_84380

/-- The number of hours for the initial parking cost -/
def initial_hours : ℝ := 2

/-- The initial parking cost -/
def initial_cost : ℝ := 9

/-- The cost per hour for excess hours -/
def excess_cost_per_hour : ℝ := 1.75

/-- The total number of hours parked -/
def total_hours : ℝ := 9

/-- The average cost per hour for the total parking time -/
def average_cost_per_hour : ℝ := 2.361111111111111

theorem parking_cost_theorem :
  initial_hours = 2 ∧
  initial_cost + excess_cost_per_hour * (total_hours - initial_hours) =
    average_cost_per_hour * total_hours :=
by sorry

end NUMINAMATH_CALUDE_parking_cost_theorem_l843_84380


namespace NUMINAMATH_CALUDE_cosine_BHD_value_l843_84335

structure RectangularPrism where
  DHG : Real
  FHB : Real

def cosine_BHD (prism : RectangularPrism) : Real :=
  sorry

theorem cosine_BHD_value (prism : RectangularPrism) 
  (h1 : prism.DHG = Real.pi / 4)
  (h2 : prism.FHB = Real.pi / 3) :
  cosine_BHD prism = Real.sqrt 6 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_BHD_value_l843_84335


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l843_84390

theorem simplify_and_rationalize : 
  (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 9) * (Real.sqrt 10 / Real.sqrt 11) = 
  (10 * Real.sqrt 66) / 99 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l843_84390


namespace NUMINAMATH_CALUDE_audio_channel_bandwidth_l843_84324

/-- Represents the parameters for an audio channel --/
structure AudioChannelParams where
  session_duration : ℕ  -- in minutes
  sampling_rate : ℕ     -- in Hz
  bit_depth : ℕ         -- in bits
  metadata_size : ℕ     -- in bytes
  metadata_per : ℕ      -- in kilobits of audio
  is_stereo : Bool

/-- Calculates the required bandwidth for an audio channel --/
def calculate_bandwidth (params : AudioChannelParams) : ℝ :=
  sorry

/-- Theorem stating the required bandwidth for the given audio channel parameters --/
theorem audio_channel_bandwidth 
  (params : AudioChannelParams)
  (h1 : params.session_duration = 51)
  (h2 : params.sampling_rate = 63)
  (h3 : params.bit_depth = 17)
  (h4 : params.metadata_size = 47)
  (h5 : params.metadata_per = 5)
  (h6 : params.is_stereo = true) :
  ∃ (ε : ℝ), ε > 0 ∧ abs (calculate_bandwidth params - 2.25) < ε :=
sorry

end NUMINAMATH_CALUDE_audio_channel_bandwidth_l843_84324


namespace NUMINAMATH_CALUDE_log_plus_fraction_gt_one_l843_84316

theorem log_plus_fraction_gt_one (x a : ℝ) (hx : x > 1) (ha : a ≥ 1/2) :
  Real.log x + a / (x - 1) > 1 := by sorry

end NUMINAMATH_CALUDE_log_plus_fraction_gt_one_l843_84316


namespace NUMINAMATH_CALUDE_converse_not_always_true_l843_84394

theorem converse_not_always_true : 
  ¬ (∀ (a b m : ℝ), a < b → a * m^2 < b * m^2) :=
by sorry

end NUMINAMATH_CALUDE_converse_not_always_true_l843_84394


namespace NUMINAMATH_CALUDE_reflection_composition_l843_84313

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_line (p : ℝ × ℝ) : ℝ × ℝ := 
  let p' := (p.1, p.2 - 2)
  let p'' := (p'.2, p'.1)
  (p''.1, p''.2 + 2)

theorem reflection_composition (D : ℝ × ℝ) (h : D = (5, 2)) : 
  reflect_line (reflect_x D) = (-4, 7) := by sorry

end NUMINAMATH_CALUDE_reflection_composition_l843_84313


namespace NUMINAMATH_CALUDE_first_group_men_count_l843_84355

/-- Represents the amount of work that can be done by one person in one day -/
structure WorkRate where
  men : ℝ
  boys : ℝ

/-- Represents a group of workers -/
structure WorkGroup where
  men : ℕ
  boys : ℕ

/-- Represents a work scenario -/
structure WorkScenario where
  group : WorkGroup
  days : ℕ

theorem first_group_men_count (rate : WorkRate) 
  (scenario1 : WorkScenario)
  (scenario2 : WorkScenario)
  (scenario3 : WorkScenario) :
  scenario1.group.men = 6 :=
by
  sorry

#check first_group_men_count

end NUMINAMATH_CALUDE_first_group_men_count_l843_84355


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l843_84392

open Set

def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 6}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 7}

theorem intersection_A_complement_B : A ∩ (𝒰 \ B) = Ioo (-3) 2 ∪ {2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l843_84392


namespace NUMINAMATH_CALUDE_exists_periodic_nonconstant_sequence_l843_84386

/-- A sequence is periodic if there exists a positive integer p such that
    x_{n+p} = x_n for all integers n -/
def IsPeriodic (x : ℤ → ℝ) : Prop :=
  ∃ p : ℕ+, ∀ n : ℤ, x (n + p) = x n

/-- A sequence is constant if all its terms are equal -/
def IsConstant (x : ℤ → ℝ) : Prop :=
  ∀ m n : ℤ, x m = x n

/-- The recurrence relation for the sequence -/
def SatisfiesRecurrence (x : ℤ → ℝ) : Prop :=
  ∀ n : ℤ, x (n + 1) = 3 * x n + 4 * x (n - 1)

theorem exists_periodic_nonconstant_sequence :
  ∃ x : ℤ → ℝ, SatisfiesRecurrence x ∧ IsPeriodic x ∧ ¬IsConstant x := by
  sorry

end NUMINAMATH_CALUDE_exists_periodic_nonconstant_sequence_l843_84386


namespace NUMINAMATH_CALUDE_worker_payment_l843_84378

/-- Calculate the total amount paid to a worker for a week -/
theorem worker_payment (daily_wage : ℝ) (days_worked : List ℝ) : 
  daily_wage = 20 →
  days_worked = [11, 32, 31, 8.3, 4] →
  (daily_wage * (days_worked.sum)) = 1726 := by
sorry

end NUMINAMATH_CALUDE_worker_payment_l843_84378


namespace NUMINAMATH_CALUDE_product_a2_a6_l843_84301

def S (n : ℕ) : ℕ := 2^n - 1

def a (n : ℕ) : ℕ := S n - S (n-1)

theorem product_a2_a6 : a 2 * a 6 = 64 := by
  sorry

end NUMINAMATH_CALUDE_product_a2_a6_l843_84301


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l843_84302

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  arith : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1 : ℕ) * seq.d) / 2

theorem arithmetic_sequence_first_term 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 3 + seq.a 6 = 40) 
  (h2 : S seq 2 = 10) : 
  seq.a 1 = 5/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l843_84302


namespace NUMINAMATH_CALUDE_lemon_cupcakes_left_at_home_l843_84321

/-- Proves that the number of lemon cupcakes left at home is 2 -/
theorem lemon_cupcakes_left_at_home 
  (total_baked : ℕ) 
  (boxes_given : ℕ) 
  (cupcakes_per_box : ℕ) 
  (h1 : total_baked = 53) 
  (h2 : boxes_given = 17) 
  (h3 : cupcakes_per_box = 3) : 
  total_baked - (boxes_given * cupcakes_per_box) = 2 := by
  sorry

end NUMINAMATH_CALUDE_lemon_cupcakes_left_at_home_l843_84321


namespace NUMINAMATH_CALUDE_three_million_squared_l843_84300

theorem three_million_squared :
  (3000000 : ℕ) * 3000000 = 9000000000000 := by
  sorry

end NUMINAMATH_CALUDE_three_million_squared_l843_84300


namespace NUMINAMATH_CALUDE_simplify_expression_l843_84341

theorem simplify_expression (x : ℝ) : 2 * (x - 3) - (-x + 4) = 3 * x - 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l843_84341


namespace NUMINAMATH_CALUDE_fraction_simplification_l843_84362

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) :
  2 / (x + y) - (x - 3*y) / (x^2 - y^2) = 1 / (x - y) := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l843_84362


namespace NUMINAMATH_CALUDE_equal_powers_implies_equality_l843_84383

theorem equal_powers_implies_equality (a b : ℝ) : 
  0 < a → 0 < b → a^b = b^a → a < 1 → a = b := by
sorry

end NUMINAMATH_CALUDE_equal_powers_implies_equality_l843_84383


namespace NUMINAMATH_CALUDE_product_96_104_l843_84350

theorem product_96_104 : 96 * 104 = 9984 := by
  sorry

end NUMINAMATH_CALUDE_product_96_104_l843_84350


namespace NUMINAMATH_CALUDE_pass_rate_two_steps_l843_84399

/-- The pass rate of a product going through two independent processing steps -/
def product_pass_rate (a b : ℝ) : ℝ := (1 - a) * (1 - b)

/-- Theorem stating that the pass rate of a product going through two independent
    processing steps with defect rates a and b is (1-a) * (1-b) -/
theorem pass_rate_two_steps (a b : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : 
  product_pass_rate a b = (1 - a) * (1 - b) := by
  sorry

#check pass_rate_two_steps

end NUMINAMATH_CALUDE_pass_rate_two_steps_l843_84399


namespace NUMINAMATH_CALUDE_extremum_maximum_at_negative_one_l843_84373

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f_derivative (x : ℝ) : ℝ := 3*x^2 - 3

/-- Theorem stating that x = -1 is the extremum maximum point of f(x) --/
theorem extremum_maximum_at_negative_one :
  ∃ (a : ℝ), a = -1 ∧ 
  (∀ x : ℝ, f x ≤ f a) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - a| ∧ |x - a| < δ → f x < f a) :=
sorry

end NUMINAMATH_CALUDE_extremum_maximum_at_negative_one_l843_84373


namespace NUMINAMATH_CALUDE_team_b_city_a_matches_l843_84303

/-- Represents a team in the tournament -/
structure Team where
  city : Fin 16
  isTeamA : Bool

/-- The number of matches played by a team -/
def matchesPlayed (t : Team) : ℕ := sorry

/-- The tournament satisfies the given conditions -/
axiom tournament_conditions :
  ∀ t1 t2 : Team,
    (t1 ≠ t2) →
    (t1.city ≠ t2.city ∨ t1.isTeamA ≠ t2.isTeamA) →
    (t1 ≠ ⟨0, true⟩) →
    (t2 ≠ ⟨0, true⟩) →
    matchesPlayed t1 ≠ matchesPlayed t2

/-- All teams except one have played between 0 and 30 matches -/
axiom matches_range :
  ∀ t : Team, t ≠ ⟨0, true⟩ → matchesPlayed t ≤ 30

/-- The theorem to be proved -/
theorem team_b_city_a_matches :
  matchesPlayed ⟨0, false⟩ = 15 := by sorry

end NUMINAMATH_CALUDE_team_b_city_a_matches_l843_84303


namespace NUMINAMATH_CALUDE_new_average_after_grace_marks_l843_84310

/-- Represents the grace marks distribution for different percentile ranges -/
structure GraceMarksDistribution where
  below_25th : ℕ
  between_25th_50th : ℕ
  between_50th_75th : ℕ
  above_75th : ℕ

/-- Represents the class statistics -/
structure ClassStats where
  size : ℕ
  original_average : ℝ
  standard_deviation : ℝ
  percentile_25th : ℝ
  percentile_50th : ℝ
  percentile_75th : ℝ

def calculate_new_average (stats : ClassStats) (grace_marks : GraceMarksDistribution) : ℝ :=
  sorry

theorem new_average_after_grace_marks
  (stats : ClassStats)
  (grace_marks : GraceMarksDistribution)
  (h_size : stats.size = 35)
  (h_original_avg : stats.original_average = 37)
  (h_std_dev : stats.standard_deviation = 6)
  (h_25th : stats.percentile_25th = 32)
  (h_50th : stats.percentile_50th = 37)
  (h_75th : stats.percentile_75th = 42)
  (h_grace_below_25th : grace_marks.below_25th = 6)
  (h_grace_25th_50th : grace_marks.between_25th_50th = 4)
  (h_grace_50th_75th : grace_marks.between_50th_75th = 2)
  (h_grace_above_75th : grace_marks.above_75th = 0) :
  abs (calculate_new_average stats grace_marks - 40.09) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_grace_marks_l843_84310


namespace NUMINAMATH_CALUDE_roses_flats_is_three_l843_84326

/-- Represents the plant shop inventory and fertilizer requirements --/
structure PlantShop where
  petunia_flats : ℕ
  petunias_per_flat : ℕ
  roses_per_flat : ℕ
  venus_flytraps : ℕ
  petunia_fertilizer : ℕ
  rose_fertilizer : ℕ
  venus_flytrap_fertilizer : ℕ
  total_fertilizer : ℕ

/-- Calculates the number of flats of roses in the shop --/
def roses_flats (shop : PlantShop) : ℕ :=
  let petunia_total := shop.petunia_flats * shop.petunias_per_flat * shop.petunia_fertilizer
  let venus_total := shop.venus_flytraps * shop.venus_flytrap_fertilizer
  let roses_total := shop.total_fertilizer - petunia_total - venus_total
  roses_total / (shop.roses_per_flat * shop.rose_fertilizer)

/-- Theorem stating that the number of rose flats is 3 --/
theorem roses_flats_is_three (shop : PlantShop)
  (h1 : shop.petunia_flats = 4)
  (h2 : shop.petunias_per_flat = 8)
  (h3 : shop.roses_per_flat = 6)
  (h4 : shop.venus_flytraps = 2)
  (h5 : shop.petunia_fertilizer = 8)
  (h6 : shop.rose_fertilizer = 3)
  (h7 : shop.venus_flytrap_fertilizer = 2)
  (h8 : shop.total_fertilizer = 314) :
  roses_flats shop = 3 := by
  sorry

end NUMINAMATH_CALUDE_roses_flats_is_three_l843_84326


namespace NUMINAMATH_CALUDE_father_twice_as_old_father_four_times_now_l843_84336

/-- Represents the current age of the father -/
def father_age : ℕ := 40

/-- Represents the current age of the daughter -/
def daughter_age : ℕ := 10

/-- Represents the number of years until the father is twice as old as the daughter -/
def years_until_twice : ℕ := 20

/-- Theorem stating that after the specified number of years, the father will be twice as old as the daughter -/
theorem father_twice_as_old :
  father_age + years_until_twice = 2 * (daughter_age + years_until_twice) :=
sorry

/-- Theorem stating that the father is currently 4 times as old as the daughter -/
theorem father_four_times_now :
  father_age = 4 * daughter_age :=
sorry

end NUMINAMATH_CALUDE_father_twice_as_old_father_four_times_now_l843_84336


namespace NUMINAMATH_CALUDE_amelia_wins_probability_l843_84384

/-- Probability of Amelia's coin landing heads -/
def p_amelia : ℚ := 1/4

/-- Probability of Blaine's coin landing heads -/
def p_blaine : ℚ := 3/7

/-- Maximum number of rounds -/
def max_rounds : ℕ := 5

/-- The probability that Amelia wins the coin toss game -/
def amelia_wins_prob : ℚ := 223/784

/-- Theorem stating that the probability of Amelia winning is 223/784 -/
theorem amelia_wins_probability : 
  amelia_wins_prob = p_amelia * (1 - p_blaine) + 
    (1 - p_amelia) * (1 - p_blaine) * p_amelia * (1 - p_blaine) + 
    (1 - p_amelia) * (1 - p_blaine) * (1 - p_amelia) * (1 - p_blaine) * p_amelia := by
  sorry

#check amelia_wins_probability

end NUMINAMATH_CALUDE_amelia_wins_probability_l843_84384


namespace NUMINAMATH_CALUDE_smallest_N_bound_l843_84308

theorem smallest_N_bound (x : ℝ) (h : |x - 2| < 0.01) : 
  |x^2 - 4| < 0.0401 ∧ 
  ∀ ε > 0, ∃ y : ℝ, |y - 2| < 0.01 ∧ |y^2 - 4| ≥ 0.0401 - ε :=
sorry

end NUMINAMATH_CALUDE_smallest_N_bound_l843_84308


namespace NUMINAMATH_CALUDE_point_on_linear_graph_l843_84371

/-- Given that the point (a, -1) lies on the graph of y = -2x + 1, prove that a = 1 -/
theorem point_on_linear_graph (a : ℝ) : 
  -1 = -2 * a + 1 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_linear_graph_l843_84371


namespace NUMINAMATH_CALUDE_sum_of_squares_coefficients_l843_84320

theorem sum_of_squares_coefficients 
  (a b c d e f : ℤ) 
  (h : ∀ x : ℝ, 729 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) : 
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_coefficients_l843_84320


namespace NUMINAMATH_CALUDE_binomial_inequality_l843_84379

theorem binomial_inequality (x : ℝ) (m : ℕ) (h : x > -1) :
  (1 + x)^m ≥ 1 + m * x := by
  sorry

end NUMINAMATH_CALUDE_binomial_inequality_l843_84379


namespace NUMINAMATH_CALUDE_residue_of_7_pow_1234_mod_19_l843_84375

theorem residue_of_7_pow_1234_mod_19 : 7^1234 % 19 = 9 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_7_pow_1234_mod_19_l843_84375


namespace NUMINAMATH_CALUDE_base_conversion_sum_l843_84356

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 13 to base 10 -/
def base13_to_base10 (n : ℕ) : ℕ := sorry

theorem base_conversion_sum :
  let base8_num := 357
  let base13_num := 4 * 13^2 + 12 * 13 + 13
  (base8_to_base10 base8_num) + (base13_to_base10 base13_num) = 1084 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l843_84356


namespace NUMINAMATH_CALUDE_inequality_proof_l843_84377

theorem inequality_proof (x y z w : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0) 
  (h_eq : (x^3 + y^3)^4 = z^3 + w^3) : 
  x^4*z + y^4*w ≥ z*w := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l843_84377


namespace NUMINAMATH_CALUDE_fraction_sum_to_decimal_l843_84325

theorem fraction_sum_to_decimal : 3/8 + 5/32 = 0.53125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_to_decimal_l843_84325


namespace NUMINAMATH_CALUDE_log_ratio_equality_l843_84361

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem log_ratio_equality (m n : ℝ) 
  (h1 : log10 2 = m) 
  (h2 : log10 3 = n) : 
  (log10 12) / (log10 15) = (2*m + n) / (1 - m + n) := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_equality_l843_84361


namespace NUMINAMATH_CALUDE_min_value_theorem_l843_84327

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 27) :
  18 ≤ 3 * a + 2 * b + c ∧ ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ * b₀ * c₀ = 27 ∧ 3 * a₀ + 2 * b₀ + c₀ = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l843_84327


namespace NUMINAMATH_CALUDE_min_value_expression_l843_84395

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (y/x) + (1/y) ≥ 4 ∧ ((y/x) + (1/y) = 4 ↔ x = 1/3 ∧ y = 1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l843_84395


namespace NUMINAMATH_CALUDE_sparrow_grains_l843_84342

theorem sparrow_grains : ∃ (x : ℕ), 
  (9 * x < 1001) ∧ 
  (10 * x > 1100) ∧ 
  (x = 111) := by
sorry

end NUMINAMATH_CALUDE_sparrow_grains_l843_84342


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l843_84368

def M : Set ℝ := {y | 0 < y ∧ y < 1}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem sufficient_not_necessary : 
  (∀ x, x ∈ M → x ∈ N) ∧ 
  (∃ x, x ∈ N ∧ x ∉ M) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l843_84368


namespace NUMINAMATH_CALUDE_power_division_l843_84311

theorem power_division (a b c d : ℕ) (h : b = a^2) :
  a^(2*c+1) / b^c = a :=
sorry

end NUMINAMATH_CALUDE_power_division_l843_84311


namespace NUMINAMATH_CALUDE_birds_and_storks_l843_84389

theorem birds_and_storks (initial_birds : ℕ) (initial_storks : ℕ) (joining_storks : ℕ) :
  initial_birds = 6 →
  initial_storks = 3 →
  joining_storks = 2 →
  initial_birds - (initial_storks + joining_storks) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_birds_and_storks_l843_84389


namespace NUMINAMATH_CALUDE_probability_of_purple_marble_l843_84348

theorem probability_of_purple_marble (blue_prob green_prob purple_prob : ℝ) : 
  blue_prob = 0.35 →
  green_prob = 0.45 →
  blue_prob + green_prob + purple_prob = 1 →
  purple_prob = 0.2 := by
sorry

end NUMINAMATH_CALUDE_probability_of_purple_marble_l843_84348


namespace NUMINAMATH_CALUDE_unique_x_with_rational_sums_l843_84366

theorem unique_x_with_rational_sums (x : ℝ) :
  (∃ a : ℚ, x + Real.sqrt 3 = a) ∧ 
  (∃ b : ℚ, x^2 + Real.sqrt 3 = b) →
  x = 1/2 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_unique_x_with_rational_sums_l843_84366


namespace NUMINAMATH_CALUDE_cubic_polynomial_d_value_l843_84345

/-- Represents a cubic polynomial of the form 3x^3 + dx^2 + ex - 6 -/
structure CubicPolynomial where
  d : ℝ
  e : ℝ

def CubicPolynomial.eval (p : CubicPolynomial) (x : ℝ) : ℝ :=
  3 * x^3 + p.d * x^2 + p.e * x - 6

def CubicPolynomial.productOfZeros (p : CubicPolynomial) : ℝ := 2

def CubicPolynomial.sumOfCoefficients (p : CubicPolynomial) : ℝ :=
  3 + p.d + p.e - 6

theorem cubic_polynomial_d_value (p : CubicPolynomial) :
  p.productOfZeros = 9 →
  p.sumOfCoefficients = 9 →
  p.d = -18 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_d_value_l843_84345


namespace NUMINAMATH_CALUDE_negation_of_proposition_l843_84314

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x ≥ 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + x₀ < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l843_84314


namespace NUMINAMATH_CALUDE_february_greatest_difference_l843_84398

-- Define the sales data for drummers and bugle players
def drummer_sales : Fin 5 → ℕ
  | 0 => 4  -- January
  | 1 => 5  -- February
  | 2 => 4  -- March
  | 3 => 3  -- April
  | 4 => 2  -- May

def bugle_sales : Fin 5 → ℕ
  | 0 => 3  -- January
  | 1 => 3  -- February
  | 2 => 4  -- March
  | 3 => 4  -- April
  | 4 => 3  -- May

-- Define the percentage difference function
def percentage_difference (a b : ℕ) : ℚ :=
  (max a b - min a b : ℚ) / (min a b : ℚ) * 100

-- Define a function to calculate the percentage difference for each month
def month_percentage_difference (i : Fin 5) : ℚ :=
  percentage_difference (drummer_sales i) (bugle_sales i)

-- Theorem: February has the greatest percentage difference
theorem february_greatest_difference :
  ∀ i : Fin 5, i ≠ 1 → month_percentage_difference 1 ≥ month_percentage_difference i :=
by sorry

end NUMINAMATH_CALUDE_february_greatest_difference_l843_84398


namespace NUMINAMATH_CALUDE_third_month_sales_l843_84315

def sales_1 : ℕ := 6435
def sales_2 : ℕ := 6927
def sales_4 : ℕ := 7230
def sales_5 : ℕ := 6562
def sales_6 : ℕ := 6191
def average_sale : ℕ := 6700
def num_months : ℕ := 6

theorem third_month_sales :
  ∃ (sales_3 : ℕ),
    sales_3 = average_sale * num_months - (sales_1 + sales_2 + sales_4 + sales_5 + sales_6) ∧
    sales_3 = 6855 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sales_l843_84315


namespace NUMINAMATH_CALUDE_probability_select_leaders_l843_84381

def club_sizes : List Nat := [6, 8, 9]

def num_clubs : Nat := 3

def num_selected : Nat := 4

def num_co_presidents : Nat := 2

def num_vice_presidents : Nat := 1

theorem probability_select_leaders (club_sizes : List Nat) 
  (h1 : club_sizes = [6, 8, 9]) 
  (h2 : num_clubs = 3) 
  (h3 : num_selected = 4) 
  (h4 : num_co_presidents = 2) 
  (h5 : num_vice_presidents = 1) : 
  (1 / num_clubs) * (club_sizes.map (λ n => Nat.choose (n - (num_co_presidents + num_vice_presidents)) 1 / Nat.choose n num_selected)).sum = 67 / 630 := by
  sorry

end NUMINAMATH_CALUDE_probability_select_leaders_l843_84381


namespace NUMINAMATH_CALUDE_baking_scoop_size_l843_84382

theorem baking_scoop_size (total_ingredients : ℚ) (num_scoops : ℕ) (scoop_size : ℚ) :
  total_ingredients = 3.75 ∧ num_scoops = 15 ∧ total_ingredients = num_scoops * scoop_size →
  scoop_size = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_baking_scoop_size_l843_84382


namespace NUMINAMATH_CALUDE_curve_C_symmetry_l843_84353

/-- The curve C is defined by the equation x^2*y + x*y^2 = 1 --/
def C (x y : ℝ) : Prop := x^2*y + x*y^2 = 1

/-- A point (x, y) is symmetric to (a, b) with respect to the line y=x --/
def symmetric_y_eq_x (x y a b : ℝ) : Prop := x = b ∧ y = a

theorem curve_C_symmetry :
  (∀ x y : ℝ, C x y → C y x) ∧ 
  (∃ x y : ℝ, C x y ∧ ¬C x (-y)) ∧ 
  (∃ x y : ℝ, C x y ∧ ¬C (-x) y) ∧ 
  (∃ x y : ℝ, C x y ∧ ¬C (-x) (-y)) ∧ 
  (∃ x y : ℝ, C x y ∧ ¬C (-y) (-x)) :=
sorry

end NUMINAMATH_CALUDE_curve_C_symmetry_l843_84353


namespace NUMINAMATH_CALUDE_six_at_three_equals_six_l843_84340

/-- The @ operation for positive integers a and b where a > b -/
def at_op (a b : ℕ+) (h : a > b) : ℚ :=
  (a * b : ℚ) / (a - b)

/-- Theorem: 6 @ 3 = 6 -/
theorem six_at_three_equals_six :
  ∀ (h : (6 : ℕ+) > (3 : ℕ+)), at_op 6 3 h = 6 := by sorry

end NUMINAMATH_CALUDE_six_at_three_equals_six_l843_84340


namespace NUMINAMATH_CALUDE_pencil_difference_l843_84305

/-- The number of pencils each person has -/
structure PencilCounts where
  candy : ℕ
  caleb : ℕ
  calen : ℕ

/-- The conditions of the problem -/
def problem_conditions (p : PencilCounts) : Prop :=
  p.candy = 9 ∧
  p.calen = p.caleb + 5 ∧
  p.caleb < 2 * p.candy ∧
  p.calen - 10 = 10

/-- The theorem to be proved -/
theorem pencil_difference (p : PencilCounts) 
  (h : problem_conditions p) : 2 * p.candy - p.caleb = 3 := by
  sorry


end NUMINAMATH_CALUDE_pencil_difference_l843_84305


namespace NUMINAMATH_CALUDE_evie_shell_collection_l843_84376

theorem evie_shell_collection (daily_shells : ℕ) : 
  (6 * daily_shells - 2 = 58) → daily_shells = 10 := by
  sorry

end NUMINAMATH_CALUDE_evie_shell_collection_l843_84376


namespace NUMINAMATH_CALUDE_number_operation_proof_l843_84309

theorem number_operation_proof (x : ℝ) : x = 115 → (((x + 45) / 2) / 2) + 45 = 85 := by
  sorry

end NUMINAMATH_CALUDE_number_operation_proof_l843_84309


namespace NUMINAMATH_CALUDE_octal_arithmetic_equality_l843_84372

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Addition operation for octal numbers --/
def octal_add : OctalNumber → OctalNumber → OctalNumber := sorry

/-- Subtraction operation for octal numbers --/
def octal_sub : OctalNumber → OctalNumber → OctalNumber := sorry

/-- Conversion from decimal to octal --/
def to_octal : ℕ → OctalNumber := sorry

/-- Theorem: In base 8, 5234₈ - 127₈ + 235₈ = 5344₈ --/
theorem octal_arithmetic_equality :
  octal_sub (octal_add (to_octal 5234) (to_octal 235)) (to_octal 127) = to_octal 5344 := by
  sorry

end NUMINAMATH_CALUDE_octal_arithmetic_equality_l843_84372


namespace NUMINAMATH_CALUDE_prime_numbers_existence_l843_84393

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem prime_numbers_existence : 
  ∃ (a : ℕ), 
    a < 10 ∧ 
    is_prime (11*a - 1) ∧ 
    is_prime (10*a + 1) ∧ 
    is_prime (10*a + 7) ∧ 
    a = 4 :=
sorry

end NUMINAMATH_CALUDE_prime_numbers_existence_l843_84393


namespace NUMINAMATH_CALUDE_basketball_series_probability_l843_84332

/-- The probability of at least k successes in n independent trials with probability p -/
def prob_at_least (n k : ℕ) (p : ℝ) : ℝ := sorry

theorem basketball_series_probability :
  prob_at_least 9 5 (1/2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_basketball_series_probability_l843_84332


namespace NUMINAMATH_CALUDE_function_positivity_implies_m_range_l843_84307

/-- Given two functions f and g defined on real numbers, 
    prove that if at least one of f(x) or g(x) is positive for all real x,
    then the parameter m is in the open interval (0, 8) -/
theorem function_positivity_implies_m_range 
  (f g : ℝ → ℝ) 
  (m : ℝ) 
  (hf : f = fun x ↦ 2 * m * x^2 - 2 * (4 - m) * x + 1) 
  (hg : g = fun x ↦ m * x) 
  (h : ∀ x : ℝ, 0 < f x ∨ 0 < g x) : 
  0 < m ∧ m < 8 := by
  sorry

end NUMINAMATH_CALUDE_function_positivity_implies_m_range_l843_84307


namespace NUMINAMATH_CALUDE_min_tuple_c_value_l843_84349

def is_valid_tuple (a b c d e f : ℕ) : Prop :=
  a + 2*b + 6*c + 30*d + 210*e + 2310*f = 2^15

def tuple_sum (a b c d e f : ℕ) : ℕ :=
  a + b + c + d + e + f

theorem min_tuple_c_value :
  ∃ (a b c d e f : ℕ),
    is_valid_tuple a b c d e f ∧
    (∀ (a' b' c' d' e' f' : ℕ),
      is_valid_tuple a' b' c' d' e' f' →
      tuple_sum a b c d e f ≤ tuple_sum a' b' c' d' e' f') ∧
    c = 1 := by sorry

end NUMINAMATH_CALUDE_min_tuple_c_value_l843_84349


namespace NUMINAMATH_CALUDE_variance_sum_random_nonrandom_l843_84306

/-- A random function -/
def RandomFunction (α : Type*) := α → ℝ

/-- A non-random function -/
def NonRandomFunction (α : Type*) := α → ℝ

/-- Variance of a random function -/
noncomputable def variance (X : RandomFunction ℝ) (t : ℝ) : ℝ := sorry

/-- The sum of a random function and a non-random function -/
def sumFunction (X : RandomFunction ℝ) (φ : NonRandomFunction ℝ) : RandomFunction ℝ :=
  fun t => X t + φ t

/-- Theorem: The variance of the sum of a random function and a non-random function
    is equal to the variance of the random function -/
theorem variance_sum_random_nonrandom
  (X : RandomFunction ℝ) (φ : NonRandomFunction ℝ) (t : ℝ) :
  variance (sumFunction X φ) t = variance X t := by sorry

end NUMINAMATH_CALUDE_variance_sum_random_nonrandom_l843_84306


namespace NUMINAMATH_CALUDE_cut_square_equation_l843_84346

/-- Represents the dimensions of a rectangular sheet and the side length of squares cut from its corners. -/
structure SheetDimensions where
  length : ℝ
  width : ℝ
  cutSide : ℝ

/-- Calculates the area of the base of a box formed by cutting squares from a sheet's corners. -/
def baseArea (d : SheetDimensions) : ℝ :=
  (d.length - 2 * d.cutSide) * (d.width - 2 * d.cutSide)

/-- Calculates the original area of a rectangular sheet. -/
def originalArea (d : SheetDimensions) : ℝ :=
  d.length * d.width

/-- Theorem stating the relationship between the cut side length and the resulting box dimensions. -/
theorem cut_square_equation (d : SheetDimensions) 
    (h1 : d.length = 8)
    (h2 : d.width = 6)
    (h3 : baseArea d = (2/3) * originalArea d) :
  d.cutSide ^ 2 - 7 * d.cutSide + 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cut_square_equation_l843_84346


namespace NUMINAMATH_CALUDE_double_plus_five_l843_84304

theorem double_plus_five (x : ℝ) (h : x = 6) : 2 * x + 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_double_plus_five_l843_84304


namespace NUMINAMATH_CALUDE_six_meter_logs_more_advantageous_l843_84364

-- Define the length of logs and the target length of chunks
def log_length_6 : ℕ := 6
def log_length_7 : ℕ := 7
def chunk_length : ℕ := 1
def total_length : ℕ := 42

-- Define the number of cuts needed for each log type
def cuts_per_log_6 : ℕ := log_length_6 - 1
def cuts_per_log_7 : ℕ := log_length_7 - 1

-- Define the number of logs needed for each type
def logs_needed_6 : ℕ := (total_length + log_length_6 - 1) / log_length_6
def logs_needed_7 : ℕ := (total_length + log_length_7 - 1) / log_length_7

-- Define the total number of cuts for each log type
def total_cuts_6 : ℕ := logs_needed_6 * cuts_per_log_6
def total_cuts_7 : ℕ := logs_needed_7 * cuts_per_log_7

-- Theorem statement
theorem six_meter_logs_more_advantageous :
  total_cuts_6 < total_cuts_7 :=
by sorry

end NUMINAMATH_CALUDE_six_meter_logs_more_advantageous_l843_84364


namespace NUMINAMATH_CALUDE_remaining_coin_value_l843_84387

def initial_quarters : Nat := 11
def initial_dimes : Nat := 15
def initial_nickels : Nat := 7

def purchased_quarters : Nat := 1
def purchased_dimes : Nat := 8
def purchased_nickels : Nat := 3

def quarter_value : Nat := 25
def dime_value : Nat := 10
def nickel_value : Nat := 5

theorem remaining_coin_value :
  (initial_quarters - purchased_quarters) * quarter_value +
  (initial_dimes - purchased_dimes) * dime_value +
  (initial_nickels - purchased_nickels) * nickel_value = 340 := by
  sorry

end NUMINAMATH_CALUDE_remaining_coin_value_l843_84387


namespace NUMINAMATH_CALUDE_solve_for_d_l843_84388

theorem solve_for_d (c a m d : ℝ) (h : m = (c * a * d) / (a - d)) : 
  d = (m * a) / (m + c * a) := by
sorry

end NUMINAMATH_CALUDE_solve_for_d_l843_84388


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l843_84397

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := |3 * x - 1| + a * x + 3

-- Part 1: Prove the solution set for f(x) ≤ 5 when a = 1
theorem solution_set_part1 : 
  {x : ℝ | f 1 x ≤ 5} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 3/4} := by sorry

-- Part 2: Prove the range of a for which f(x) has a minimum value
theorem range_of_a_part2 :
  ∀ a : ℝ, (∃ x : ℝ, ∀ y : ℝ, f a x ≤ f a y) → -3 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l843_84397


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l843_84319

theorem geometric_sequence_problem (a b c r : ℤ) : 
  (b = a * r ∧ c = a * r^2) →  -- geometric sequence condition
  (r ≠ 0) →                   -- non-zero ratio
  (c = a + 56) →              -- given condition
  b = 21 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l843_84319


namespace NUMINAMATH_CALUDE_no_integer_solution_l843_84352

theorem no_integer_solution : ∀ x y : ℤ, x^2 + 3*x*y - 2*y^2 ≠ 122 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l843_84352


namespace NUMINAMATH_CALUDE_exists_n_where_B_less_than_A_largest_n_where_B_leq_A_l843_84360

-- Define A(n) for Alphonse's jumps
def A (n : ℕ) : ℕ :=
  n / 8 + n % 8

-- Define B(n) for Beryl's jumps
def B (n : ℕ) : ℕ :=
  n / 7 + n % 7

-- Part (a)
theorem exists_n_where_B_less_than_A :
  ∃ n : ℕ, n > 200 ∧ B n < A n :=
sorry

-- Part (b)
theorem largest_n_where_B_leq_A :
  ∀ n : ℕ, B n ≤ A n → n ≤ 343 :=
sorry

end NUMINAMATH_CALUDE_exists_n_where_B_less_than_A_largest_n_where_B_leq_A_l843_84360


namespace NUMINAMATH_CALUDE_constant_c_value_l843_84354

theorem constant_c_value (b c : ℝ) : 
  (∀ x : ℝ, (x + 2) * (x + b) = x^2 + c*x + 6) → c = 5 := by
  sorry

end NUMINAMATH_CALUDE_constant_c_value_l843_84354


namespace NUMINAMATH_CALUDE_division_simplification_l843_84374

theorem division_simplification (a : ℝ) (h : a ≠ 0) :
  (a - 1/a) / ((a - 1)/a) = a + 1 := by
sorry

end NUMINAMATH_CALUDE_division_simplification_l843_84374


namespace NUMINAMATH_CALUDE_like_terms_imply_m_minus_2n_equals_1_l843_84357

/-- Two monomials are like terms if they have the same variables with the same exponents. -/
def are_like_terms (m n : ℕ) : Prop :=
  m = 3 ∧ n = 1

/-- The theorem states that if 3x^m*y and -5x^3*y^n are like terms, then m - 2n = 1. -/
theorem like_terms_imply_m_minus_2n_equals_1 (m n : ℕ) :
  are_like_terms m n → m - 2*n = 1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_m_minus_2n_equals_1_l843_84357


namespace NUMINAMATH_CALUDE_ellipse_intersection_slope_l843_84338

/-- Given an ellipse mx^2 + ny^2 = 1 intersecting with y = 1 - x at A and B,
    if the slope of the line through origin and midpoint of AB is √2, then m/n = √2 -/
theorem ellipse_intersection_slope (m n : ℝ) (A B : ℝ × ℝ) :
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  (m * x₁^2 + n * y₁^2 = 1) →
  (m * x₂^2 + n * y₂^2 = 1) →
  (y₁ = 1 - x₁) →
  (y₂ = 1 - x₂) →
  ((y₁ + y₂) / (x₁ + x₂) = Real.sqrt 2) →
  m / n = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_slope_l843_84338


namespace NUMINAMATH_CALUDE_binomial_rv_p_value_l843_84358

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  mean : ℝ
  std_dev : ℝ

/-- Theorem: For a binomial random variable with mean 200 and standard deviation 10, p = 1/2 -/
theorem binomial_rv_p_value (X : BinomialRV) 
  (h_mean : X.mean = 200)
  (h_std_dev : X.std_dev = 10) :
  X.p = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_rv_p_value_l843_84358


namespace NUMINAMATH_CALUDE_solution_set_equivalence_minimum_value_l843_84347

-- Define the function f
def f (m n : ℝ) (x : ℝ) : ℝ := m * x^2 - n * x

-- Part 1
theorem solution_set_equivalence
  (m n t : ℝ)
  (h1 : ∀ x, f m n x ≥ t ↔ -3 ≤ x ∧ x ≤ 2) :
  ∀ x, n * x^2 + m * x + t ≤ 0 ↔ -2 ≤ x ∧ x ≤ 3 :=
sorry

-- Part 2
theorem minimum_value
  (m n : ℝ)
  (h1 : f m n 1 > 0)
  (h2 : 1 ≤ m ∧ m ≤ 3) :
  ∃ (m₀ n₀ : ℝ), 1/(m₀-n₀) + 9/m₀ - n₀ = 2 ∧
    ∀ m n, f m n 1 > 0 → 1 ≤ m ∧ m ≤ 3 → 1/(m-n) + 9/m - n ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_minimum_value_l843_84347


namespace NUMINAMATH_CALUDE_series_sum_equals_one_over_200_l843_84339

/-- The nth term of the series -/
def seriesTerm (n : ℕ) : ℚ :=
  (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

/-- The sum of the series -/
noncomputable def seriesSum : ℚ := ∑' n, seriesTerm n

/-- Theorem stating that the sum of the series is 1/200 -/
theorem series_sum_equals_one_over_200 : seriesSum = 1 / 200 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_over_200_l843_84339


namespace NUMINAMATH_CALUDE_bottles_not_in_crates_l843_84333

/-- Represents the number of bottles that can be held by each crate size -/
structure CrateCapacity where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the number of crates of each size -/
structure CrateCount where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculate the total capacity of all crates -/
def totalCrateCapacity (capacity : CrateCapacity) (count : CrateCount) : Nat :=
  capacity.small * count.small + capacity.medium * count.medium + capacity.large * count.large

/-- Calculate the number of bottles that will not be placed in a crate -/
def bottlesNotInCrates (totalBottles : Nat) (capacity : CrateCapacity) (count : CrateCount) : Nat :=
  totalBottles - totalCrateCapacity capacity count

/-- Theorem stating that 50 bottles will not be placed in a crate -/
theorem bottles_not_in_crates : 
  let totalBottles : Nat := 250
  let capacity : CrateCapacity := { small := 8, medium := 12, large := 20 }
  let count : CrateCount := { small := 5, medium := 5, large := 5 }
  bottlesNotInCrates totalBottles capacity count = 50 := by
  sorry

end NUMINAMATH_CALUDE_bottles_not_in_crates_l843_84333


namespace NUMINAMATH_CALUDE_expression_equals_two_l843_84323

theorem expression_equals_two (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt (a * b / 2) + Real.sqrt 8) / Real.sqrt ((a * b + 16) / 8 + Real.sqrt (a * b)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l843_84323


namespace NUMINAMATH_CALUDE_triangle_circles_area_sum_l843_84391

theorem triangle_circles_area_sum (r s t : ℝ) : 
  r > 0 ∧ s > 0 ∧ t > 0 →
  r + s = 5 →
  r + t = 12 →
  s + t = 13 →
  π * (r^2 + s^2 + t^2) = 113 * π :=
by sorry

end NUMINAMATH_CALUDE_triangle_circles_area_sum_l843_84391


namespace NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l843_84351

/-- The value of p for which a circle (x-1)^2 + y^2 = 4 is tangent to the directrix of a parabola y^2 = 2px -/
theorem circle_tangent_to_parabola_directrix (p : ℝ) : 
  p > 0 → 
  (∃ (x y : ℝ), (x - 1)^2 + y^2 = 4 ∧ y^2 = 2*p*x) →
  (∀ (x y : ℝ), (x - 1)^2 + y^2 = 4 → x ≥ -p/2) →
  (∃ (x y : ℝ), (x - 1)^2 + y^2 = 4 ∧ x = -p/2) →
  p = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l843_84351


namespace NUMINAMATH_CALUDE_sum_of_seven_squares_not_perfect_square_l843_84363

theorem sum_of_seven_squares_not_perfect_square (n : ℤ) : 
  ¬∃ (m : ℤ), 7 * (n ^ 2 + 4) = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_squares_not_perfect_square_l843_84363


namespace NUMINAMATH_CALUDE_evaluate_expression_l843_84367

theorem evaluate_expression (a x : ℝ) (h : x = a + 5) : x - a + 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l843_84367


namespace NUMINAMATH_CALUDE_symmetric_polynomial_value_l843_84331

/-- Given a function f(x) = (x² + 3x)(x² + ax + b) where f(x) = f(2-x) for all real x, prove f(3) = -18 -/
theorem symmetric_polynomial_value (a b : ℝ) :
  (∀ x : ℝ, (x^2 + 3*x) * (x^2 + a*x + b) = ((2-x)^2 + 3*(2-x)) * ((2-x)^2 + a*(2-x) + b)) →
  (3^2 + 3*3) * (3^2 + a*3 + b) = -18 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_polynomial_value_l843_84331


namespace NUMINAMATH_CALUDE_sum_congruence_l843_84328

theorem sum_congruence : (1 + 23 + 456 + 7890) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_l843_84328


namespace NUMINAMATH_CALUDE_sqrt_inequality_l843_84317

theorem sqrt_inequality (x : ℝ) (h : x ≥ -3) :
  Real.sqrt (x + 5) - Real.sqrt (x + 3) > Real.sqrt (x + 6) - Real.sqrt (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l843_84317


namespace NUMINAMATH_CALUDE_sisters_age_when_kolya_was_her_current_age_l843_84370

/- Define the current ages of the brother, sister, and Kolya -/
variable (x y k : ℕ)

/- Define the time differences -/
variable (t₁ t₂ : ℕ)

/- First condition: When Kolya was as old as they both are now, the sister was as old as the brother is now -/
axiom condition1 : k - t₁ = x + y ∧ y - t₁ = x

/- Second condition: When Kolya was as old as the sister is now, the sister's age was to be determined -/
axiom condition2 : k - t₂ = y

/- The theorem to prove -/
theorem sisters_age_when_kolya_was_her_current_age : y - t₂ = 0 :=
sorry

end NUMINAMATH_CALUDE_sisters_age_when_kolya_was_her_current_age_l843_84370


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l843_84334

/-- The cubic polynomial q(x) that satisfies given conditions -/
def q (x : ℝ) : ℝ := 4 * x^3 - 19 * x^2 + 5 * x + 6

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions :
  q 0 = 6 ∧ q 1 = -4 ∧ q 2 = 0 ∧ q 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l843_84334


namespace NUMINAMATH_CALUDE_problem_statement_l843_84343

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -9)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 10) :
  b / (a + b) + c / (b + c) + a / (c + a) = 11 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l843_84343


namespace NUMINAMATH_CALUDE_remainder_problem_l843_84330

theorem remainder_problem : 29 * 169^1990 ≡ 7 [MOD 11] := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l843_84330


namespace NUMINAMATH_CALUDE_modular_inverse_30_mod_31_l843_84329

theorem modular_inverse_30_mod_31 : ∃ x : ℕ, x ≤ 31 ∧ (30 * x) % 31 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_30_mod_31_l843_84329


namespace NUMINAMATH_CALUDE_symmetric_point_xoz_l843_84318

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The plane xOz in three-dimensional space -/
def PlaneXOZ : Set Point3D :=
  {p : Point3D | p.y = 0}

/-- Symmetry with respect to the plane xOz -/
def symmetricXOZ (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, p.z⟩

theorem symmetric_point_xoz :
  let A : Point3D := ⟨-3, 2, -4⟩
  symmetricXOZ A = ⟨-3, -2, -4⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_xoz_l843_84318


namespace NUMINAMATH_CALUDE_arctangent_inequalities_l843_84369

theorem arctangent_inequalities (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (Real.arctan x + Real.arctan y < π / 2 ↔ x * y < 1) ∧
  (Real.arctan x + Real.arctan y + Real.arctan z < π ↔ x * y * z < x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_arctangent_inequalities_l843_84369


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_240_l843_84365

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  rectangle_length * rectangle_breadth

theorem rectangle_area_is_240 :
  rectangle_area 3600 10 = 240 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_240_l843_84365


namespace NUMINAMATH_CALUDE_nikola_ant_farm_problem_l843_84385

/-- Nikola's ant farm problem -/
theorem nikola_ant_farm_problem 
  (num_ants : ℕ) 
  (food_per_ant : ℕ) 
  (food_cost_per_oz : ℚ) 
  (leaf_cost : ℚ) 
  (num_leaves : ℕ) 
  (num_jobs : ℕ) : 
  num_ants = 400 →
  food_per_ant = 2 →
  food_cost_per_oz = 1/10 →
  leaf_cost = 1/100 →
  num_leaves = 6000 →
  num_jobs = 4 →
  (num_ants * food_per_ant * food_cost_per_oz - num_leaves * leaf_cost) / num_jobs = 5 :=
by sorry

end NUMINAMATH_CALUDE_nikola_ant_farm_problem_l843_84385


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l843_84312

theorem line_slope_intercept_product (m b : ℚ) : 
  m = 3/4 → b = 6/5 → 0 < m * b ∧ m * b < 1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l843_84312


namespace NUMINAMATH_CALUDE_solution_set_x_one_minus_x_l843_84396

theorem solution_set_x_one_minus_x (x : ℝ) : x * (1 - x) > 0 ↔ 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_x_one_minus_x_l843_84396


namespace NUMINAMATH_CALUDE_factorization_proof_l843_84322

theorem factorization_proof (a b x y : ℝ) : x * (a + b) - 2 * y * (a + b) = (a + b) * (x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l843_84322


namespace NUMINAMATH_CALUDE_tree_space_for_given_conditions_l843_84337

/-- Calculates the sidewalk space taken by each tree given the street length, number of trees, and space between trees. -/
def tree_space (street_length : ℕ) (num_trees : ℕ) (space_between : ℕ) : ℚ :=
  let total_gap_space := (num_trees - 1) * space_between
  let total_tree_space := street_length - total_gap_space
  (total_tree_space : ℚ) / num_trees

/-- Theorem stating that for a 151-foot street with 16 trees and 9 feet between each tree, each tree takes up 1 square foot of sidewalk space. -/
theorem tree_space_for_given_conditions :
  tree_space 151 16 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_tree_space_for_given_conditions_l843_84337
