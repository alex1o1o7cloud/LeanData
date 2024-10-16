import Mathlib

namespace NUMINAMATH_CALUDE_josephus_69_l39_3988

/-- The Josephus function that returns the last remaining number given n. -/
def josephus (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the Josephus number for n = 69 is 10. -/
theorem josephus_69 : josephus 69 = 10 := by
  sorry

end NUMINAMATH_CALUDE_josephus_69_l39_3988


namespace NUMINAMATH_CALUDE_sculpture_and_base_height_l39_3935

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Represents the height of an object in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ

/-- Converts a Height to total inches -/
def heightToInches (h : Height) : ℕ := feetToInches h.feet + h.inches

/-- Theorem: The combined height of the sculpture and base is 42 inches -/
theorem sculpture_and_base_height :
  let sculpture : Height := { feet := 2, inches := 10 }
  let base_height : ℕ := 8
  heightToInches sculpture + base_height = 42 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_and_base_height_l39_3935


namespace NUMINAMATH_CALUDE_alpha_value_when_beta_is_36_l39_3983

/-- Given that α² is inversely proportional to β, and α = 4 when β = 9,
    prove that α = ±2 when β = 36. -/
theorem alpha_value_when_beta_is_36
  (k : ℝ)  -- Constant of proportionality
  (h1 : ∀ α β : ℝ, α ^ 2 * β = k)  -- α² is inversely proportional to β
  (h2 : 4 ^ 2 * 9 = k)  -- α = 4 when β = 9
  : {α : ℝ | α ^ 2 * 36 = k} = {2, -2} := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_when_beta_is_36_l39_3983


namespace NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l39_3985

theorem greatest_integer_for_all_real_domain : 
  ∃ (b : ℤ), (∀ (x : ℝ), x^2 + b * x + 12 ≠ 0) ∧ 
  (∀ (c : ℤ), c > b → ∃ (x : ℝ), x^2 + c * x + 12 = 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l39_3985


namespace NUMINAMATH_CALUDE_q_unique_l39_3971

-- Define the numerator polynomial
def p (x : ℝ) : ℝ := x^4 - 3*x^3 - 4*x + 12

-- Define the properties of q(x)
def has_vertical_asymptotes (q : ℝ → ℝ) : Prop :=
  q 3 = 0 ∧ q (-1) = 0

def no_horizontal_asymptote (q : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, ∀ x : ℝ, |q x| ≤ |x|^n

-- Main theorem
theorem q_unique (q : ℝ → ℝ) :
  has_vertical_asymptotes q →
  no_horizontal_asymptote q →
  q (-2) = 20 →
  ∀ x, q x = 4*x^2 - 8*x - 12 :=
sorry

end NUMINAMATH_CALUDE_q_unique_l39_3971


namespace NUMINAMATH_CALUDE_abs_value_sum_difference_l39_3915

theorem abs_value_sum_difference (a b : ℝ) : 
  (|a| = 2) → (|b| = 4) → (a + b < 0) → (a - b = 2 ∨ a - b = 6) := by
  sorry

end NUMINAMATH_CALUDE_abs_value_sum_difference_l39_3915


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l39_3914

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- An arithmetic sequence -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The property that a_{n+1}^2 = a_n * a_{n+2} for all n -/
def has_square_middle_property (a : Sequence) : Prop :=
  ∀ n : ℕ, (a (n + 1))^2 = a n * a (n + 2)

theorem arithmetic_sequence_property :
  (∀ a : Sequence, is_arithmetic a → has_square_middle_property a) ∧
  (∃ a : Sequence, has_square_middle_property a ∧ ¬is_arithmetic a) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l39_3914


namespace NUMINAMATH_CALUDE_vasya_lowest_position_l39_3961

/-- Represents a cyclist in the race -/
structure Cyclist :=
  (id : Nat)

/-- Represents a stage in the race -/
structure Stage :=
  (number : Nat)

/-- Represents the time a cyclist takes to complete a stage -/
structure StageTime :=
  (cyclist : Cyclist)
  (stage : Stage)
  (time : ℝ)

/-- Represents the total time a cyclist takes to complete all stages -/
structure TotalTime :=
  (cyclist : Cyclist)
  (time : ℝ)

/-- The number of cyclists in the race -/
def numCyclists : Nat := 500

/-- The number of stages in the race -/
def numStages : Nat := 15

/-- Vasya's position in each stage -/
def vasyaStagePosition : Nat := 7

/-- Function to get a cyclist's position in a stage -/
def stagePosition (c : Cyclist) (s : Stage) : Nat := sorry

/-- Function to get a cyclist's overall position -/
def overallPosition (c : Cyclist) : Nat := sorry

/-- Vasya's cyclist object -/
def vasya : Cyclist := ⟨0⟩  -- Assuming Vasya's ID is 0

/-- The main theorem -/
theorem vasya_lowest_position :
  (∀ s : Stage, stagePosition vasya s = vasyaStagePosition) →
  (∀ c1 c2 : Cyclist, ∀ s : Stage, c1 ≠ c2 → stagePosition c1 s ≠ stagePosition c2 s) →
  (∀ c1 c2 : Cyclist, c1 ≠ c2 → overallPosition c1 ≠ overallPosition c2) →
  overallPosition vasya ≤ 91 := sorry

end NUMINAMATH_CALUDE_vasya_lowest_position_l39_3961


namespace NUMINAMATH_CALUDE_suitcase_electronics_weight_l39_3903

/-- Proves that the weight of electronics is 12 pounds given the conditions of the suitcase problem -/
theorem suitcase_electronics_weight 
  (B C E : ℝ) -- Weights of books, clothes, and electronics
  (h1 : B / C = 7 / 4) -- Initial ratio of books to clothes
  (h2 : C / E = 4 / 3) -- Initial ratio of clothes to electronics
  (h3 : B / (C - 8) = 2 * (B / C)) -- Ratio doubles after removing 8 pounds of clothes
  : E = 12 := by
  sorry

end NUMINAMATH_CALUDE_suitcase_electronics_weight_l39_3903


namespace NUMINAMATH_CALUDE_gina_purse_value_l39_3953

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels dimes : ℕ) : ℕ :=
  pennies * coin_value "penny" +
  nickels * coin_value "nickel" +
  dimes * coin_value "dime"

/-- Converts cents to percentage of a dollar -/
def cents_to_percentage (cents : ℕ) : ℚ :=
  (cents : ℚ) / 100

theorem gina_purse_value :
  cents_to_percentage (total_value 2 3 2) = 37 / 100 := by
  sorry

end NUMINAMATH_CALUDE_gina_purse_value_l39_3953


namespace NUMINAMATH_CALUDE_even_function_derivative_zero_l39_3922

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- Theorem: If f is an even function and its derivative exists, then f'(0) = 0 -/
theorem even_function_derivative_zero (f : ℝ → ℝ) (hf : IsEven f) (hf' : Differentiable ℝ f) :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_derivative_zero_l39_3922


namespace NUMINAMATH_CALUDE_intersection_M_N_l39_3932

def M : Set ℝ := {x | -2 < x ∧ x < 3}
def N : Set ℝ := {-2, -1, 0, 1}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l39_3932


namespace NUMINAMATH_CALUDE_lily_score_l39_3912

/-- Represents the score for hitting a specific ring -/
structure RingScore where
  inner : ℕ
  middle : ℕ
  outer : ℕ

/-- Represents the number of hits for each ring -/
structure Hits where
  inner : ℕ
  middle : ℕ
  outer : ℕ

/-- Calculates the total score given ring scores and hits -/
def totalScore (rs : RingScore) (h : Hits) : ℕ :=
  rs.inner * h.inner + rs.middle * h.middle + rs.outer * h.outer

theorem lily_score 
  (rs : RingScore) 
  (tom_hits john_hits : Hits) 
  (h1 : tom_hits.inner + tom_hits.middle + tom_hits.outer = 6)
  (h2 : john_hits.inner + john_hits.middle + john_hits.outer = 6)
  (h3 : totalScore rs tom_hits = 46)
  (h4 : totalScore rs john_hits = 34)
  (h5 : totalScore rs { inner := 4, middle := 4, outer := 4 } = 80) :
  totalScore rs { inner := 2, middle := 2, outer := 2 } = 40 := by
  sorry

#check lily_score

end NUMINAMATH_CALUDE_lily_score_l39_3912


namespace NUMINAMATH_CALUDE_cab_journey_time_l39_3990

/-- Given a cab walking at 5/6 of its usual speed and arriving 8 minutes late,
    prove that its usual time to cover the journey is 40 minutes. -/
theorem cab_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (5 / 6 * usual_speed) * (usual_time + 8) = usual_speed * usual_time → 
  usual_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_cab_journey_time_l39_3990


namespace NUMINAMATH_CALUDE_population_increase_l39_3986

theorem population_increase (x : ℝ) : 
  (3 + 3 * x / 100 = 12) → x = 300 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_l39_3986


namespace NUMINAMATH_CALUDE_solve_system_l39_3904

theorem solve_system (x y : ℚ) (eq1 : 3 * x - 2 * y = 7) (eq2 : 2 * x + 3 * y = 8) : x = 37 / 13 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l39_3904


namespace NUMINAMATH_CALUDE_hyperbola_sum_l39_3956

/-- Given a hyperbola with center (1, 0), one focus at (1 + √41, 0), and one vertex at (-2, 0),
    prove that h + k + a + b = 1 + 0 + 3 + 4√2, where (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1
    is the equation of the hyperbola. -/
theorem hyperbola_sum (h k a b : ℝ) : 
  (1 : ℝ) = h ∧ (0 : ℝ) = k ∧  -- center at (1, 0)
  (1 + Real.sqrt 41 : ℝ) = h + Real.sqrt (c^2) ∧ -- focus at (1 + √41, 0)
  (-2 : ℝ) = h - a ∧ -- vertex at (-2, 0)
  (∀ x y : ℝ, (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1) → -- equation of hyperbola
  h + k + a + b = 1 + 0 + 3 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l39_3956


namespace NUMINAMATH_CALUDE_arithmetic_sequence_special_condition_l39_3979

/-- An arithmetic sequence with positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Theorem stating that if 3a_6 - a_7^2 + 3a_8 = 0 in an arithmetic sequence with positive terms, then a_7 = 6 -/
theorem arithmetic_sequence_special_condition
  (seq : ArithmeticSequence)
  (h : 3 * seq.a 6 - (seq.a 7)^2 + 3 * seq.a 8 = 0) :
  seq.a 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_special_condition_l39_3979


namespace NUMINAMATH_CALUDE_tiffany_bags_on_monday_l39_3973

theorem tiffany_bags_on_monday :
  ∀ (bags_monday : ℕ),
  bags_monday + 8 = 12 →
  bags_monday = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_tiffany_bags_on_monday_l39_3973


namespace NUMINAMATH_CALUDE_lcm_of_5_6_10_18_l39_3968

theorem lcm_of_5_6_10_18 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 18)) = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_5_6_10_18_l39_3968


namespace NUMINAMATH_CALUDE_worker_completion_time_l39_3937

/-- Given two workers A and B, proves that A can complete a job in 14 days 
    when A and B together can complete the job in 10 days, 
    and B alone can complete the job in 35 days. -/
theorem worker_completion_time 
  (joint_completion_time : ℝ) 
  (b_alone_completion_time : ℝ) 
  (h1 : joint_completion_time = 10) 
  (h2 : b_alone_completion_time = 35) : 
  ∃ (a_alone_completion_time : ℝ), 
    a_alone_completion_time = 14 ∧ 
    (1 / a_alone_completion_time + 1 / b_alone_completion_time = 1 / joint_completion_time) :=
by sorry

end NUMINAMATH_CALUDE_worker_completion_time_l39_3937


namespace NUMINAMATH_CALUDE_correct_ordering_l39_3919

/-- Represents the labels of the conjectures -/
inductive ConjLabel
  | A | C | G | P | R | E | S

/-- The smallest counterexample for each conjecture -/
def smallest_counterexample : ConjLabel → ℕ
  | ConjLabel.A => 44
  | ConjLabel.C => 105
  | ConjLabel.G => 5777
  | ConjLabel.P => 906150257
  | ConjLabel.R => 23338590792
  | ConjLabel.E => 31858749840007945920321
  | ConjLabel.S => 8424432925592889329288197322308900672459420460792433

/-- Checks if a list of ConjLabels is in ascending order based on their smallest counterexamples -/
def is_ascending (labels : List ConjLabel) : Prop :=
  labels.Pairwise (λ l1 l2 => smallest_counterexample l1 < smallest_counterexample l2)

/-- The theorem to be proved -/
theorem correct_ordering :
  is_ascending [ConjLabel.A, ConjLabel.C, ConjLabel.G, ConjLabel.P, ConjLabel.R, ConjLabel.E, ConjLabel.S] :=
by sorry

end NUMINAMATH_CALUDE_correct_ordering_l39_3919


namespace NUMINAMATH_CALUDE_second_day_percentage_l39_3900

def puzzle_pieces : ℕ := 1000
def first_day_percentage : ℚ := 10 / 100
def third_day_percentage : ℚ := 30 / 100
def pieces_left_after_third_day : ℕ := 504

theorem second_day_percentage :
  ∃ (p : ℚ),
    p > 0 ∧
    p < 1 ∧
    (puzzle_pieces * (1 - first_day_percentage) * (1 - p) * (1 - third_day_percentage) : ℚ) =
      pieces_left_after_third_day ∧
    p = 20 / 100 := by
  sorry

end NUMINAMATH_CALUDE_second_day_percentage_l39_3900


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l39_3960

/-- Given a line L1 with equation 3x + 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -1/2x - 2 is parallel to L1 and passes through P. -/
theorem parallel_line_through_point (x y : ℝ) :
  (3 * x + 6 * y = 9) →  -- Equation of L1
  (y = -1/2 * x - 2) →   -- Equation of L2
  (∃ m b : ℝ, 3 * x + 6 * y = 9 ↔ y = m * x + b) →  -- L1 can be written in slope-intercept form
  ((-1/2) = m) →  -- Slopes are equal
  ((-1/2) * 2 - 2 = -3) →  -- L2 passes through (2, -3)
  (y = -1/2 * x - 2) ∧ (3 * 2 + 6 * (-3) = 9)  -- L2 is parallel to L1 and passes through (2, -3)
:= by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l39_3960


namespace NUMINAMATH_CALUDE_multiples_of_six_ending_in_four_l39_3907

theorem multiples_of_six_ending_in_four (n : ℕ) : 
  (∃ k, k ∈ Finset.range 1000 ∧ k % 10 = 4 ∧ k % 6 = 0) ↔ n = 17 :=
by sorry

end NUMINAMATH_CALUDE_multiples_of_six_ending_in_four_l39_3907


namespace NUMINAMATH_CALUDE_circle_radius_from_polar_equation_l39_3948

/-- The radius of a circle given by the polar equation ρ = 2cosθ is 1 -/
theorem circle_radius_from_polar_equation : 
  ∃ (center : ℝ × ℝ) (r : ℝ), 
    (∀ θ : ℝ, (2 * Real.cos θ * Real.cos θ, 2 * Real.cos θ * Real.sin θ) ∈ 
      {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2}) ∧ 
    r = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_from_polar_equation_l39_3948


namespace NUMINAMATH_CALUDE_radiator_problem_l39_3984

/-- Represents the fraction of original substance remaining after multiple replacements -/
def fractionRemaining (totalVolume : ℚ) (replacementVolume : ℚ) (numberOfReplacements : ℕ) : ℚ :=
  (1 - replacementVolume / totalVolume) ^ numberOfReplacements

/-- The radiator problem -/
theorem radiator_problem :
  let totalVolume : ℚ := 25
  let replacementVolume : ℚ := 5
  let numberOfReplacements : ℕ := 3
  fractionRemaining totalVolume replacementVolume numberOfReplacements = 64 / 125 := by
  sorry

end NUMINAMATH_CALUDE_radiator_problem_l39_3984


namespace NUMINAMATH_CALUDE_average_temp_bucyrus_l39_3959

/-- The average temperature in Bucyrus, Ohio over three days -/
def average_temperature (temp1 temp2 temp3 : ℤ) : ℚ :=
  (temp1 + temp2 + temp3) / 3

/-- Theorem stating that the average of the given temperatures is -7 -/
theorem average_temp_bucyrus :
  average_temperature (-14) (-8) 1 = -7 := by
  sorry

end NUMINAMATH_CALUDE_average_temp_bucyrus_l39_3959


namespace NUMINAMATH_CALUDE_cabin_rental_duration_l39_3916

/-- Proves that the number of days for which the cabin is rented is 14, given the specified conditions. -/
theorem cabin_rental_duration :
  let daily_rate : ℚ := 125
  let pet_fee : ℚ := 100
  let service_fee_rate : ℚ := 0.2
  let security_deposit_rate : ℚ := 0.5
  let security_deposit : ℚ := 1110
  ∃ (days : ℕ), 
    security_deposit = security_deposit_rate * (daily_rate * days + pet_fee + service_fee_rate * (daily_rate * days + pet_fee)) ∧
    days = 14 := by
  sorry

end NUMINAMATH_CALUDE_cabin_rental_duration_l39_3916


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l39_3958

/-- A random variable following a normal distribution with mean μ and standard deviation σ. -/
structure NormalRV (μ σ : ℝ) where
  (σ_pos : σ > 0)

/-- The probability that a normal random variable falls within a given interval. -/
def prob_interval (X : NormalRV μ σ) (a b : ℝ) : ℝ := sorry

/-- Theorem: For a normal distribution N(4, 1²), given specific probabilities for certain intervals,
    the probability P(5 < X < 6) is equal to 0.1359. -/
theorem normal_distribution_probability (X : NormalRV 4 1) :
  prob_interval X 2 6 = 0.9544 →
  prob_interval X 3 5 = 0.6826 →
  prob_interval X 5 6 = 0.1359 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l39_3958


namespace NUMINAMATH_CALUDE_fraction_subtraction_l39_3902

theorem fraction_subtraction : (4 + 6 + 8) / (3 + 5 + 7) - (3 + 5 + 7) / (4 + 6 + 8) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l39_3902


namespace NUMINAMATH_CALUDE_pens_given_to_sharon_l39_3911

/-- The number of pens given to Sharon in a pen collection scenario --/
theorem pens_given_to_sharon (initial_pens : ℕ) (mike_pens : ℕ) (final_pens : ℕ) : 
  initial_pens = 25 →
  mike_pens = 22 →
  final_pens = 75 →
  (initial_pens + mike_pens) * 2 - final_pens = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_pens_given_to_sharon_l39_3911


namespace NUMINAMATH_CALUDE_price_increase_percentage_l39_3982

theorem price_increase_percentage (new_price : ℝ) (h1 : new_price - 0.8 * new_price = 4) : 
  (new_price - (0.8 * new_price)) / (0.8 * new_price) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l39_3982


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l39_3992

def A : Set ℝ := {x | |x - 1| ≤ 1}
def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -Real.sqrt 2 ≤ x ∧ x < 1}

theorem complement_intersection_A_B : 
  (A ∩ B)ᶜ = {x : ℝ | x ≠ 0} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l39_3992


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l39_3964

def arithmetic_sequence (c d : ℤ) : Fin 5 → ℤ
  | ⟨0, _⟩ => c - 2*d
  | ⟨1, _⟩ => c - d
  | ⟨2, _⟩ => c
  | ⟨3, _⟩ => c + d
  | ⟨4, _⟩ => c + 2*d

def sum_of_cubes (f : Fin 4 → ℤ) : ℤ :=
  (f 0)^3 + (f 1)^3 + (f 2)^3 + (f 3)^3

def sum_of_terms (f : Fin 4 → ℤ) : ℤ :=
  f 0 + f 1 + f 2 + f 3

theorem arithmetic_sequence_property (c d : ℤ) :
  (sum_of_cubes (λ i => arithmetic_sequence c d i) = 
   16 * (sum_of_terms (λ i => arithmetic_sequence c d i))^2) ∧
  (sum_of_cubes (λ i => arithmetic_sequence c d ⟨i.val + 1, sorry⟩) = 
   16 * (sum_of_terms (λ i => arithmetic_sequence c d ⟨i.val + 1, sorry⟩))^2) →
  c = 32 ∧ d = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l39_3964


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_lower_bound_l39_3999

/-- Given plane vectors a, b, and c satisfying certain dot product conditions,
    prove that the magnitude of their sum is at least 4. -/
theorem vector_sum_magnitude_lower_bound
  (a b c : ℝ × ℝ)
  (ha : a.1 * a.1 + a.2 * a.2 = 1)
  (hab : a.1 * b.1 + a.2 * b.2 = 1)
  (hac : a.1 * c.1 + a.2 * c.2 = 2)
  (hbc : b.1 * c.1 + b.2 * c.2 = 1) :
  (a.1 + b.1 + c.1)^2 + (a.2 + b.2 + c.2)^2 ≥ 16 := by
  sorry

#check vector_sum_magnitude_lower_bound

end NUMINAMATH_CALUDE_vector_sum_magnitude_lower_bound_l39_3999


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l39_3957

theorem real_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  (Complex.re ((1 - 2*i) / (2 + i^5))) = 0 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l39_3957


namespace NUMINAMATH_CALUDE_triangle_vertices_l39_3965

-- Define the lines
def d₁ (x y : ℝ) : Prop := 2 * x - y - 2 = 0
def d₂ (x y : ℝ) : Prop := x + y - 4 = 0
def d₃ (x y : ℝ) : Prop := y = 2
def d₄ (x y : ℝ) : Prop := x - 4 * y + 3 = 0

-- Define the points
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (5, 2)

-- Define what it means for a line to be a median
def is_median (line : (ℝ → ℝ → Prop)) (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  sorry

-- Define what it means for a line to be an altitude
def is_altitude (line : (ℝ → ℝ → Prop)) (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  sorry

theorem triangle_vertices : 
  is_median d₁ (A, B, C) ∧ 
  is_median d₂ (A, B, C) ∧ 
  is_median d₃ (A, B, C) ∧ 
  is_altitude d₄ (A, B, C) → 
  (A = (1, 0) ∧ B = (0, 4) ∧ C = (5, 2)) :=
sorry

end NUMINAMATH_CALUDE_triangle_vertices_l39_3965


namespace NUMINAMATH_CALUDE_shirt_price_markdown_l39_3929

/-- Given a shirt price that goes through two markdowns, prove that the initial sale price
    was 70% of the original price if the second markdown is 10% and the final price
    is 63% of the original price. -/
theorem shirt_price_markdown (original_price : ℝ) (initial_sale_price : ℝ) :
  initial_sale_price > 0 →
  original_price > 0 →
  initial_sale_price * 0.9 = original_price * 0.63 →
  initial_sale_price / original_price = 0.7 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_markdown_l39_3929


namespace NUMINAMATH_CALUDE_inequality_proof_l39_3989

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^2 + 4*b^2 + 1/(a*b) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l39_3989


namespace NUMINAMATH_CALUDE_nth_prime_upper_bound_and_prime_counting_lower_bound_l39_3928

-- Define the n-th prime number
def nth_prime (n : ℕ) : ℕ := sorry

-- Define the prime counting function
def prime_counting_function (x : ℝ) : ℝ := sorry

theorem nth_prime_upper_bound_and_prime_counting_lower_bound :
  (∀ n : ℕ, nth_prime n ≤ 2^(2^n)) ∧
  (∃ c : ℝ, c > 0 ∧ ∀ x : ℝ, x > Real.exp 1 → prime_counting_function x ≥ c * Real.log (Real.log x)) :=
sorry

end NUMINAMATH_CALUDE_nth_prime_upper_bound_and_prime_counting_lower_bound_l39_3928


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l39_3970

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_example : parallelogram_area 12 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l39_3970


namespace NUMINAMATH_CALUDE_functional_equation_implies_identity_l39_3991

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

/-- The main theorem: if f satisfies the equation, then f is the identity function -/
theorem functional_equation_implies_identity (f : ℝ → ℝ) 
  (h : SatisfiesEquation f) : ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_implies_identity_l39_3991


namespace NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_cylinder_l39_3920

/-- The surface area of a sphere circumscribing a right circular cylinder with edge length 6 -/
theorem sphere_surface_area_circumscribing_cylinder (r : ℝ) : r^2 = 21 → 4 * Real.pi * r^2 = 84 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_cylinder_l39_3920


namespace NUMINAMATH_CALUDE_scaling_transform_line_l39_3910

/-- Scaling transformation that maps (x, y) to (x', y') -/
def scaling_transform (x y : ℝ) : ℝ × ℝ :=
  (3 * x, 2 * y)

theorem scaling_transform_line : 
  ∀ (x y : ℝ), x + y = 1 → 
  let (x', y') := scaling_transform x y
  2 * x' + 3 * y' = 6 := by
sorry

end NUMINAMATH_CALUDE_scaling_transform_line_l39_3910


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l39_3942

-- Define a sequence as a function from ℕ to ℝ
def Sequence := ℕ → ℝ

-- Define what it means for a sequence to be increasing
def IsIncreasing (a : Sequence) : Prop :=
  ∀ n : ℕ, a n ≤ a (n + 1)

-- Define the condition a_{n+1} > |a_n|
def StrictlyGreaterThanAbs (a : Sequence) : Prop :=
  ∀ n : ℕ, a (n + 1) > |a n|

-- Theorem statement
theorem condition_sufficient_not_necessary :
  (∀ a : Sequence, StrictlyGreaterThanAbs a → IsIncreasing a) ∧
  (∃ a : Sequence, IsIncreasing a ∧ ¬StrictlyGreaterThanAbs a) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l39_3942


namespace NUMINAMATH_CALUDE_min_value_of_expression_l39_3940

theorem min_value_of_expression (x : ℝ) (h : x > 0) :
  x + 4 / (x + 1) ≥ 3 ∧ ∃ y > 0, y + 4 / (y + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l39_3940


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l39_3945

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 6 * x + c = 0) →  -- exactly one solution
  (a + c = 12) →                     -- sum condition
  (a < c) →                          -- order condition
  (a = 6 - 3 * Real.sqrt 3 ∧ c = 6 + 3 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l39_3945


namespace NUMINAMATH_CALUDE_elizabeth_pencil_purchase_l39_3954

/-- Given Elizabeth's shopping scenario, prove she can buy exactly 5 pencils. -/
theorem elizabeth_pencil_purchase (
  initial_money : ℚ)
  (pen_cost : ℚ)
  (pencil_cost : ℚ)
  (pens_to_buy : ℕ)
  (h1 : initial_money = 20)
  (h2 : pen_cost = 2)
  (h3 : pencil_cost = 1.6)
  (h4 : pens_to_buy = 6) :
  (initial_money - pens_to_buy * pen_cost) / pencil_cost = 5 := by
sorry

end NUMINAMATH_CALUDE_elizabeth_pencil_purchase_l39_3954


namespace NUMINAMATH_CALUDE_negation_equal_area_congruent_is_true_l39_3933

-- Define a type for triangles
def Triangle : Type := sorry

-- Define a function for the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Theorem stating that the negation of "Triangles with equal areas are congruent" is true
theorem negation_equal_area_congruent_is_true :
  ¬(∀ t1 t2 : Triangle, area t1 = area t2 → congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_negation_equal_area_congruent_is_true_l39_3933


namespace NUMINAMATH_CALUDE_tiffany_towels_l39_3944

theorem tiffany_towels (packs : ℕ) (towels_per_pack : ℕ) (h1 : packs = 9) (h2 : towels_per_pack = 3) :
  packs * towels_per_pack = 27 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_towels_l39_3944


namespace NUMINAMATH_CALUDE_distance_to_CD_l39_3950

/-- A square with semi-circle arcs -/
structure SquareWithArcs (s : ℝ) where
  -- Square ABCD
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- Ensure it's a square with side length s
  square_side : dist A B = s ∧ dist B C = s ∧ dist C D = s ∧ dist D A = s
  -- Semi-circle arcs
  arc_A : Set (ℝ × ℝ)
  arc_B : Set (ℝ × ℝ)
  -- Ensure arcs have correct radii and centers
  arc_A_def : arc_A = {p : ℝ × ℝ | dist p A = s/2 ∧ p.1 ≥ A.1 ∧ p.2 ≤ C.2}
  arc_B_def : arc_B = {p : ℝ × ℝ | dist p B = s/2 ∧ p.1 ≤ B.1 ∧ p.2 ≤ C.2}
  -- Intersection point X
  X : ℝ × ℝ
  X_def : X ∈ arc_A ∧ X ∈ arc_B

/-- The main theorem -/
theorem distance_to_CD (s : ℝ) (h : s > 0) (sq : SquareWithArcs s) :
  dist sq.X (sq.C.1, sq.X.2) = s :=
sorry

end NUMINAMATH_CALUDE_distance_to_CD_l39_3950


namespace NUMINAMATH_CALUDE_transistor_count_scientific_notation_l39_3934

/-- The number of transistors in a Huawei Kirin 990 processor -/
def transistor_count : ℝ := 12000000000

/-- The scientific notation representation of the transistor count -/
def scientific_notation : ℝ := 1.2 * (10 ^ 10)

theorem transistor_count_scientific_notation : 
  transistor_count = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_transistor_count_scientific_notation_l39_3934


namespace NUMINAMATH_CALUDE_singles_percentage_l39_3995

def total_hits : ℕ := 40
def home_runs : ℕ := 2
def triples : ℕ := 3
def doubles : ℕ := 6

def singles : ℕ := total_hits - (home_runs + triples + doubles)

def percentage_singles : ℚ := singles / total_hits * 100

theorem singles_percentage : percentage_singles = 72.5 := by
  sorry

end NUMINAMATH_CALUDE_singles_percentage_l39_3995


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l39_3936

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^11 + i^16 + i^21 + i^26 + i^31 + i^36 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l39_3936


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l39_3930

/-- A quadratic function passing through points (0,2) and (1,0) -/
def quadratic_function (x b c : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_function_properties :
  ∃ (b c : ℝ),
    (quadratic_function 0 b c = 2) ∧
    (quadratic_function 1 b c = 0) ∧
    (b = -3) ∧
    (c = 2) ∧
    (∀ x, quadratic_function x b c = (x - 3/2)^2 - 1/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l39_3930


namespace NUMINAMATH_CALUDE_no_food_left_for_dog_l39_3925

theorem no_food_left_for_dog (N : ℕ) (prepared_food : ℝ) : 
  let stayed := N / 3
  let excursion := 2 * N / 3
  let lunch_portion := prepared_food / 4
  let excursion_portion := 1.5 * lunch_portion
  stayed * lunch_portion + excursion * excursion_portion = prepared_food :=
by sorry

end NUMINAMATH_CALUDE_no_food_left_for_dog_l39_3925


namespace NUMINAMATH_CALUDE_one_ta_grading_time_l39_3994

/-- The number of initial teaching assistants -/
def N : ℕ := 5

/-- The time it takes N teaching assistants to grade all homework -/
def initial_time : ℕ := 5

/-- The time it takes N+1 teaching assistants to grade all homework -/
def new_time : ℕ := 4

/-- The total work required to grade all homework -/
def total_work : ℕ := N * initial_time

theorem one_ta_grading_time :
  (total_work : ℚ) = 20 :=
by sorry

end NUMINAMATH_CALUDE_one_ta_grading_time_l39_3994


namespace NUMINAMATH_CALUDE_lexis_cement_is_10_l39_3966

/-- The amount of cement (in tons) used for Lexi's street -/
def lexis_cement : ℝ := 15.1 - 5.1

/-- Theorem stating that the amount of cement used for Lexi's street is 10 tons -/
theorem lexis_cement_is_10 : lexis_cement = 10 := by
  sorry

end NUMINAMATH_CALUDE_lexis_cement_is_10_l39_3966


namespace NUMINAMATH_CALUDE_negation_equivalence_l39_3967

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (3 : ℝ) ^ x + x < 0) ↔ (∀ x : ℝ, (3 : ℝ) ^ x + x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l39_3967


namespace NUMINAMATH_CALUDE_combined_annual_income_l39_3909

def monthly_income_problem (A B C D : ℝ) : Prop :=
  -- Ratio condition
  A / B = 5 / 3 ∧ B / C = 3 / 2 ∧
  -- B's income is 12% more than C's
  B = 1.12 * C ∧
  -- D's income is 15% less than A's
  D = 0.85 * A ∧
  -- C's income is 17000
  C = 17000

theorem combined_annual_income 
  (A B C D : ℝ) 
  (h : monthly_income_problem A B C D) : 
  (A + B + C + D) * 12 = 1375980 := by
  sorry

#check combined_annual_income

end NUMINAMATH_CALUDE_combined_annual_income_l39_3909


namespace NUMINAMATH_CALUDE_line_x_intercept_m_values_l39_3998

theorem line_x_intercept_m_values (m : ℝ) : 
  (∃ y : ℝ, (2 * m^2 - m + 3) * 1 + (m^2 + 2*m) * y = 4*m + 1) → 
  (m = 2 ∨ m = 1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_line_x_intercept_m_values_l39_3998


namespace NUMINAMATH_CALUDE_complex_roots_quadratic_l39_3977

theorem complex_roots_quadratic (p q : ℝ) : 
  (p + 3*I : ℂ) * (p + 3*I : ℂ) - (12 + 11*I : ℂ) * (p + 3*I : ℂ) + (9 + 63*I : ℂ) = 0 ∧
  (q + 6*I : ℂ) * (q + 6*I : ℂ) - (12 + 11*I : ℂ) * (q + 6*I : ℂ) + (9 + 63*I : ℂ) = 0 →
  p = 9 ∧ q = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_roots_quadratic_l39_3977


namespace NUMINAMATH_CALUDE_min_value_of_f_l39_3978

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 2023

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = 1996) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l39_3978


namespace NUMINAMATH_CALUDE_point_reflection_x_axis_l39_3947

/-- Given a point P(-1,2) in the Cartesian coordinate system, 
    its coordinates with respect to the x-axis are (-1,-2). -/
theorem point_reflection_x_axis : 
  let P : ℝ × ℝ := (-1, 2)
  let reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  reflect_x P = (-1, -2) := by sorry

end NUMINAMATH_CALUDE_point_reflection_x_axis_l39_3947


namespace NUMINAMATH_CALUDE_dans_remaining_marbles_l39_3908

-- Define the initial number of green marbles Dan has
def initial_green_marbles : ℝ := 32.0

-- Define the number of green marbles Mike took
def marbles_taken : ℝ := 23.0

-- Define the number of green marbles Dan has now
def remaining_green_marbles : ℝ := initial_green_marbles - marbles_taken

-- Theorem to prove
theorem dans_remaining_marbles :
  remaining_green_marbles = 9.0 := by sorry

end NUMINAMATH_CALUDE_dans_remaining_marbles_l39_3908


namespace NUMINAMATH_CALUDE_power_product_equality_l39_3975

theorem power_product_equality : (3^5 * 4^5) * 6^2 = 8957952 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l39_3975


namespace NUMINAMATH_CALUDE_unique_digit_product_solution_l39_3981

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem unique_digit_product_solution :
  ∃! n : ℕ, digit_product n = n^2 - 10*n - 22 :=
sorry

end NUMINAMATH_CALUDE_unique_digit_product_solution_l39_3981


namespace NUMINAMATH_CALUDE_age_ratio_problem_l39_3941

theorem age_ratio_problem (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 24) :
  (x : ℚ) / y = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l39_3941


namespace NUMINAMATH_CALUDE_bob_distance_when_met_l39_3943

/-- The distance between points X and Y in miles -/
def total_distance : ℝ := 17

/-- Yolanda's speed for the first half of the journey in miles per hour -/
def yolanda_speed1 : ℝ := 3

/-- Yolanda's speed for the second half of the journey in miles per hour -/
def yolanda_speed2 : ℝ := 4

/-- Bob's speed for the first half of the journey in miles per hour -/
def bob_speed1 : ℝ := 4

/-- Bob's speed for the second half of the journey in miles per hour -/
def bob_speed2 : ℝ := 3

/-- The time in hours that Yolanda starts walking before Bob -/
def head_start : ℝ := 1

/-- The distance Bob walked when they met -/
def bob_distance : ℝ := 8.5004

theorem bob_distance_when_met :
  ∃ (t : ℝ), 
    t > 0 ∧ 
    t < total_distance / 2 / bob_speed1 ∧
    bob_distance = bob_speed1 * t ∧
    total_distance = 
      yolanda_speed1 * (total_distance / 2 / yolanda_speed1) +
      yolanda_speed2 * (total_distance / 2 / yolanda_speed2) +
      bob_speed1 * t +
      bob_speed2 * ((total_distance / 2 / bob_speed1 + total_distance / 2 / bob_speed2 - head_start) - t) :=
by sorry

end NUMINAMATH_CALUDE_bob_distance_when_met_l39_3943


namespace NUMINAMATH_CALUDE_negative_square_two_l39_3913

theorem negative_square_two : -2^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_two_l39_3913


namespace NUMINAMATH_CALUDE_hyperbola_equation_l39_3972

/-- The standard equation of a hyperbola passing through a given point with a given eccentricity -/
theorem hyperbola_equation (x y a b c : ℝ) (h1 : x = 2 * Real.sqrt 2) (h2 : y = -Real.sqrt 2) 
  (h3 : c / a = Real.sqrt 3) (h4 : b^2 = c^2 - a^2) (h5 : a^2 = 7) :
  x^2 / 7 - y^2 / 14 = 1 := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_hyperbola_equation_l39_3972


namespace NUMINAMATH_CALUDE_constant_term_expansion_l39_3997

theorem constant_term_expansion (n : ℕ) : 
  (∃ k : ℕ, k = n / 3 ∧ Nat.choose n k = 15) ↔ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l39_3997


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l39_3905

/-- The correct quadratic equation given erroneous roots -/
theorem correct_quadratic_equation 
  (root1_student1 root2_student1 : ℝ)
  (root1_student2 root2_student2 : ℝ)
  (h1 : root1_student1 = 5 ∧ root2_student1 = 3)
  (h2 : root1_student2 = -12 ∧ root2_student2 = -4) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
    (∀ x, a * x^2 + b * x + c = 0 ↔ x^2 - 8 * x + 48 = 0) :=
sorry

#check correct_quadratic_equation

end NUMINAMATH_CALUDE_correct_quadratic_equation_l39_3905


namespace NUMINAMATH_CALUDE_bakers_pastry_problem_l39_3962

/-- Baker's pastry problem -/
theorem bakers_pastry_problem 
  (total_cakes : ℕ) 
  (total_pastries : ℕ) 
  (sold_pastries : ℕ) 
  (remaining_pastries : ℕ) 
  (h1 : total_cakes = 7)
  (h2 : total_pastries = 148)
  (h3 : sold_pastries = 103)
  (h4 : remaining_pastries = 45)
  (h5 : total_pastries = sold_pastries + remaining_pastries) :
  ¬∃! sold_cakes : ℕ, sold_cakes ≤ total_cakes :=
sorry

end NUMINAMATH_CALUDE_bakers_pastry_problem_l39_3962


namespace NUMINAMATH_CALUDE_physics_class_size_l39_3906

theorem physics_class_size (total_students : ℕ) (both_classes : ℕ) :
  total_students = 75 →
  both_classes = 9 →
  ∃ (math_only : ℕ) (phys_only : ℕ),
    total_students = math_only + phys_only + both_classes ∧
    phys_only + both_classes = 2 * (math_only + both_classes) →
  phys_only + both_classes = 56 := by
  sorry

end NUMINAMATH_CALUDE_physics_class_size_l39_3906


namespace NUMINAMATH_CALUDE_team_selection_problem_l39_3926

def num_players : ℕ := 6
def team_size : ℕ := 3

def ways_to_select (n k : ℕ) : ℕ := (n.factorial) / ((n - k).factorial)

theorem team_selection_problem :
  ways_to_select num_players team_size - ways_to_select (num_players - 1) (team_size - 1) = 100 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_problem_l39_3926


namespace NUMINAMATH_CALUDE_berry_average_temperature_l39_3955

def berry_temperatures : List (List Float) := [
  [37.3, 37.2, 36.9],  -- Sunday
  [36.6, 36.9, 37.1],  -- Monday
  [37.1, 37.3, 37.2],  -- Tuesday
  [36.8, 37.3, 37.5],  -- Wednesday
  [37.1, 37.7, 37.3],  -- Thursday
  [37.5, 37.4, 36.9],  -- Friday
  [36.9, 37.0, 37.1]   -- Saturday
]

def average_temperature (temperatures : List (List Float)) : Float :=
  let total_sum := temperatures.map (·.sum) |>.sum
  let total_count := temperatures.length * temperatures.head!.length
  total_sum / total_count.toFloat

theorem berry_average_temperature :
  (average_temperature berry_temperatures).floor = 37 ∧
  (average_temperature berry_temperatures - (average_temperature berry_temperatures).floor) * 100 ≥ 62 :=
by sorry

end NUMINAMATH_CALUDE_berry_average_temperature_l39_3955


namespace NUMINAMATH_CALUDE_runner_problem_l39_3993

theorem runner_problem (v : ℝ) (h1 : v > 0) :
  let t1 := 20 / v
  let t2 := 40 / v
  t2 = t1 + 4 →
  t2 = 8 := by
sorry

end NUMINAMATH_CALUDE_runner_problem_l39_3993


namespace NUMINAMATH_CALUDE_sunglasses_and_hats_probability_l39_3923

theorem sunglasses_and_hats_probability 
  (total_sunglasses : ℕ) 
  (total_hats : ℕ) 
  (prob_sunglasses_given_hat : ℚ) :
  total_sunglasses = 75 →
  total_hats = 50 →
  prob_sunglasses_given_hat = 1 / 5 →
  (total_hats * prob_sunglasses_given_hat : ℚ) / total_sunglasses = 2 / 15 :=
by sorry

end NUMINAMATH_CALUDE_sunglasses_and_hats_probability_l39_3923


namespace NUMINAMATH_CALUDE_increasing_cubic_function_condition_l39_3917

/-- A function f(x) = x³ - ax - 1 is increasing for all real x if and only if a ≤ 0 -/
theorem increasing_cubic_function_condition (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => x^3 - a*x - 1)) ↔ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_increasing_cubic_function_condition_l39_3917


namespace NUMINAMATH_CALUDE_shaded_area_between_tangent_circles_l39_3927

theorem shaded_area_between_tangent_circles 
  (r₁ : ℝ) (r₂ : ℝ) (d : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 8) (h₃ : d = 4) :
  let area_shaded := π * r₂^2 - π * r₁^2
  area_shaded = 48 * π := by
sorry

end NUMINAMATH_CALUDE_shaded_area_between_tangent_circles_l39_3927


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_341_l39_3939

theorem greatest_prime_factor_of_341 :
  ∃ (p : ℕ), p.Prime ∧ p ∣ 341 ∧ p = 19 ∧ ∀ (q : ℕ), q.Prime → q ∣ 341 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_341_l39_3939


namespace NUMINAMATH_CALUDE_walk_distance_l39_3969

/-- The total distance walked by Erin and Susan -/
def total_distance (susan_distance erin_distance : ℕ) : ℕ :=
  susan_distance + erin_distance

/-- Theorem stating the total distance walked by Erin and Susan -/
theorem walk_distance :
  ∀ (susan_distance erin_distance : ℕ),
    susan_distance = 9 →
    erin_distance = susan_distance - 3 →
    total_distance susan_distance erin_distance = 15 := by
  sorry

end NUMINAMATH_CALUDE_walk_distance_l39_3969


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l39_3996

/-- A monotonically decreasing geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 0 < q ∧ q < 1 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_a3 : a 3 = 1)
  (h_sum : a 2 + a 4 = 5/2) :
  a 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l39_3996


namespace NUMINAMATH_CALUDE_cubic_fraction_factorization_l39_3987

theorem cubic_fraction_factorization (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)
  = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_fraction_factorization_l39_3987


namespace NUMINAMATH_CALUDE_cost_of_skirt_l39_3951

/-- Proves that the cost of each skirt is $15 --/
theorem cost_of_skirt (total_spent art_supplies_cost number_of_skirts : ℕ) 
  (h1 : total_spent = 50)
  (h2 : art_supplies_cost = 20)
  (h3 : number_of_skirts = 2) :
  (total_spent - art_supplies_cost) / number_of_skirts = 15 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_skirt_l39_3951


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_linear_equation_solutions_l39_3952

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x ↦ 2*x^2 - 6*x - 5
  ∃ x₁ x₂ : ℝ, x₁ = (3 + Real.sqrt 19) / 2 ∧ 
              x₂ = (3 - Real.sqrt 19) / 2 ∧ 
              f x₁ = 0 ∧ f x₂ = 0 :=
sorry

theorem linear_equation_solutions :
  let g : ℝ → ℝ := λ x ↦ 3*x*(4-x) - 2*(x-4)
  ∃ x₁ x₂ : ℝ, x₁ = 4 ∧ 
              x₂ = -2/3 ∧ 
              g x₁ = 0 ∧ g x₂ = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_linear_equation_solutions_l39_3952


namespace NUMINAMATH_CALUDE_square_preserves_geometric_sequence_sqrt_abs_preserves_geometric_sequence_l39_3931

-- Define the domain for the functions
def Domain : Set ℝ := {x : ℝ | x < 0 ∨ x > 0}

-- Define the property of being a geometric sequence
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the property of being a geometric sequence preserving function
def IsGeometricSequencePreserving (f : ℝ → ℝ) : Prop :=
  ∀ a : ℕ → ℝ, (∀ n : ℕ, a n ∈ Domain) →
    IsGeometricSequence a → IsGeometricSequence (f ∘ a)

-- State the theorem for f(x) = x^2
theorem square_preserves_geometric_sequence :
  IsGeometricSequencePreserving (fun x ↦ x^2) :=
sorry

-- State the theorem for f(x) = √|x|
theorem sqrt_abs_preserves_geometric_sequence :
  IsGeometricSequencePreserving (fun x ↦ Real.sqrt (abs x)) :=
sorry

end NUMINAMATH_CALUDE_square_preserves_geometric_sequence_sqrt_abs_preserves_geometric_sequence_l39_3931


namespace NUMINAMATH_CALUDE_abs_m_minus_n_equals_five_l39_3918

theorem abs_m_minus_n_equals_five (m n : ℝ) 
  (h1 : m * n = 6) 
  (h2 : m + n = 7) : 
  |m - n| = 5 := by
sorry

end NUMINAMATH_CALUDE_abs_m_minus_n_equals_five_l39_3918


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l39_3974

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_first_term 
  (seq : ArithmeticSequence) 
  (h3 : seq.a 2 * seq.a 3 = seq.a 4 * seq.a 5) 
  (h4 : S seq 4 = 27) : 
  seq.a 1 = 135 / 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l39_3974


namespace NUMINAMATH_CALUDE_rectangle_sequence_area_stage_6_l39_3901

/-- Calculates the area of a rectangle sequence up to a given stage -/
def rectangleSequenceArea (stage : ℕ) : ℕ :=
  let baseWidth := 2
  let length := 3
  List.range stage |>.map (fun i => (baseWidth + i) * length) |>.sum

/-- The area of the rectangle sequence at Stage 6 is 81 square inches -/
theorem rectangle_sequence_area_stage_6 :
  rectangleSequenceArea 6 = 81 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_sequence_area_stage_6_l39_3901


namespace NUMINAMATH_CALUDE_origin_and_point_same_side_l39_3963

def line_equation (x y : ℝ) : ℝ := 3 * x + 2 * y + 5

def same_side (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  line_equation x₁ y₁ * line_equation x₂ y₂ > 0

theorem origin_and_point_same_side : same_side 0 0 (-3) 4 := by sorry

end NUMINAMATH_CALUDE_origin_and_point_same_side_l39_3963


namespace NUMINAMATH_CALUDE_least_number_divisibility_l39_3946

theorem least_number_divisibility (n : ℕ) (h : n = 59789) : 
  let m := 16142
  (∀ k : ℕ, k < m → ¬((n + k) % 7 = 0 ∧ (n + k) % 11 = 0 ∧ (n + k) % 13 = 0 ∧ (n + k) % 17 = 0)) ∧ 
  ((n + m) % 7 = 0 ∧ (n + m) % 11 = 0 ∧ (n + m) % 13 = 0 ∧ (n + m) % 17 = 0) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisibility_l39_3946


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l39_3921

theorem smallest_lcm_with_gcd_five (m n : ℕ) : 
  10000 ≤ m ∧ m < 100000 ∧ 
  10000 ≤ n ∧ n < 100000 ∧ 
  Nat.gcd m n = 5 →
  20030010 ≤ Nat.lcm m n :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l39_3921


namespace NUMINAMATH_CALUDE_linear_function_property_l39_3949

-- Define a linear function
def LinearFunction (g : ℝ → ℝ) : Prop :=
  ∀ x y t : ℝ, g (x + t * (y - x)) = g x + t * (g y - g x)

-- State the theorem
theorem linear_function_property (g : ℝ → ℝ) 
  (h_linear : LinearFunction g)
  (h1 : g 8 - g 3 = 15)
  (h2 : g 4 - g 1 = 9) :
  g 10 - g 1 = 27 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l39_3949


namespace NUMINAMATH_CALUDE_power_sum_equality_l39_3976

theorem power_sum_equality : (2 : ℕ)^(3^2) + (-1 : ℤ)^(2^3) = 513 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l39_3976


namespace NUMINAMATH_CALUDE_teresas_current_age_l39_3924

/-- Given the ages of family members at different points in time, 
    prove Teresa's current age. -/
theorem teresas_current_age 
  (morio_current_age : ℕ)
  (morio_age_at_birth : ℕ)
  (teresa_age_at_birth : ℕ)
  (h1 : morio_current_age = 71)
  (h2 : morio_age_at_birth = 38)
  (h3 : teresa_age_at_birth = 26) :
  teresa_age_at_birth + (morio_current_age - morio_age_at_birth) = 59 :=
by sorry

end NUMINAMATH_CALUDE_teresas_current_age_l39_3924


namespace NUMINAMATH_CALUDE_pyramid_volume_theorem_l39_3980

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- A right pyramid with a regular hexagon base -/
structure RightPyramid where
  base : RegularHexagon
  apex : ℝ × ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Calculate the volume of a right pyramid -/
def pyramidVolume (p : RightPyramid) : ℝ := sorry

/-- Check if a triangle is equilateral with given side length -/
def isEquilateralWithSideLength (t : EquilateralTriangle) (s : ℝ) : Prop := sorry

theorem pyramid_volume_theorem (p : RightPyramid) (t : EquilateralTriangle) :
  isEquilateralWithSideLength t 10 →
  pyramidVolume p = 187.5 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_theorem_l39_3980


namespace NUMINAMATH_CALUDE_max_area_right_triangle_l39_3938

/-- The maximum area of a right-angled triangle with perimeter √2 + 1 is 1/4 -/
theorem max_area_right_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a^2 + b^2 = c^2 → 
  a + b + c = Real.sqrt 2 + 1 → 
  (1/2 * a * b) ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_max_area_right_triangle_l39_3938
