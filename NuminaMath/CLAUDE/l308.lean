import Mathlib

namespace simplify_fraction_l308_30813

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end simplify_fraction_l308_30813


namespace polynomial_product_expansion_l308_30883

theorem polynomial_product_expansion (x : ℝ) :
  (3 * x^2 + 4) * (2 * x^3 + x^2 + 5) = 6 * x^5 + 3 * x^4 + 8 * x^3 + 19 * x^2 + 20 := by
  sorry

end polynomial_product_expansion_l308_30883


namespace table_height_is_five_l308_30840

/-- Represents the configuration of blocks and table -/
structure Configuration where
  total_length : ℝ

/-- Represents the table and blocks setup -/
structure TableSetup where
  block_length : ℝ
  block_width : ℝ
  table_height : ℝ
  config1 : Configuration
  config2 : Configuration

/-- The theorem stating the height of the table given the configurations -/
theorem table_height_is_five (setup : TableSetup)
  (h1 : setup.config1.total_length = setup.block_length + setup.table_height + setup.block_width)
  (h2 : setup.config2.total_length = 2 * setup.block_width + setup.table_height)
  (h3 : setup.config1.total_length = 45)
  (h4 : setup.config2.total_length = 40) :
  setup.table_height = 5 := by
  sorry

#check table_height_is_five

end table_height_is_five_l308_30840


namespace common_ratio_of_geometric_sequence_l308_30887

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = q * a n

theorem common_ratio_of_geometric_sequence (a : ℕ → ℚ) 
  (h_geometric : geometric_sequence a) 
  (h_a1 : a 1 = 1/8)
  (h_a4 : a 4 = -1) :
  ∃ q : ℚ, q = -2 ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
by
  sorry

end common_ratio_of_geometric_sequence_l308_30887


namespace value_of_m_l308_30854

theorem value_of_m (m : ℝ) (h1 : m ≠ 0) 
  (h2 : ∀ x : ℝ, (x^2 - m) * (x + m) = x^3 + m * (x^2 - x - 12)) : 
  m = 12 := by
sorry

end value_of_m_l308_30854


namespace rationalize_denominator_l308_30816

theorem rationalize_denominator :
  ∃ (A B C D : ℚ),
    (1 / (2 - Real.rpow 7 (1/3 : ℚ)) = Real.rpow A (1/3 : ℚ) + Real.rpow B (1/3 : ℚ) + Real.rpow C (1/3 : ℚ)) ∧
    (A = 4) ∧ (B = 2) ∧ (C = 7) ∧ (D = 1) ∧
    (A + B + C + D = 14) := by
  sorry

end rationalize_denominator_l308_30816


namespace delta_takes_five_hours_prove_delta_time_l308_30839

/-- Represents the time taken by Delta and Epsilon to complete a landscaping job -/
structure LandscapingJob where
  delta : ℝ  -- Time taken by Delta alone
  epsilon : ℝ  -- Time taken by Epsilon alone
  together : ℝ  -- Time taken by both working together

/-- The conditions of the landscaping job -/
def job_conditions (job : LandscapingJob) : Prop :=
  job.together = job.delta - 3 ∧
  job.together = job.epsilon - 4 ∧
  job.together = 2

/-- The theorem stating that Delta takes 5 hours to complete the job alone -/
theorem delta_takes_five_hours (job : LandscapingJob) 
  (h : job_conditions job) : job.delta = 5 := by
  sorry

/-- The main theorem proving Delta's time based on the given conditions -/
theorem prove_delta_time : ∃ (job : LandscapingJob), job_conditions job ∧ job.delta = 5 := by
  sorry

end delta_takes_five_hours_prove_delta_time_l308_30839


namespace arc_length_ninety_degrees_radius_three_l308_30832

/-- The arc length of a sector with a central angle of 90° and a radius of 3 is equal to (3/2)π. -/
theorem arc_length_ninety_degrees_radius_three :
  let central_angle : ℝ := 90
  let radius : ℝ := 3
  let arc_length : ℝ := (central_angle * π * radius) / 180
  arc_length = (3/2) * π := by sorry

end arc_length_ninety_degrees_radius_three_l308_30832


namespace distance_P_to_x_axis_l308_30804

/-- The distance from a point to the x-axis in a Cartesian coordinate system --/
def distance_to_x_axis (y : ℝ) : ℝ := |y|

/-- Point P in the Cartesian coordinate system --/
def P : ℝ × ℝ := (2, -3)

/-- Theorem: The distance from point P(2, -3) to the x-axis is 3 --/
theorem distance_P_to_x_axis :
  distance_to_x_axis P.2 = 3 := by sorry

end distance_P_to_x_axis_l308_30804


namespace union_complement_A_with_B_l308_30800

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 5}
def B : Set Nat := {1, 3, 5}

theorem union_complement_A_with_B :
  (U \ A) ∪ B = {1, 3, 4, 5} := by sorry

end union_complement_A_with_B_l308_30800


namespace three_numbers_sum_l308_30842

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →  -- Ascending order
  b = 10 →  -- Median is 10
  (a + b + c) / 3 = a + 20 →  -- Mean is 20 more than least
  (a + b + c) / 3 = c - 25 →  -- Mean is 25 less than greatest
  a + b + c = 45 := by
sorry

end three_numbers_sum_l308_30842


namespace gold_bars_theorem_l308_30815

/-- Represents the masses of five gold bars -/
structure GoldBars where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e
  h2 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e
  h3 : (a = 1 ∧ b = 2) ∨ (a = 1 ∧ c = 2) ∨ (a = 1 ∧ d = 2) ∨ (a = 1 ∧ e = 2) ∨
       (b = 1 ∧ c = 2) ∨ (b = 1 ∧ d = 2) ∨ (b = 1 ∧ e = 2) ∨
       (c = 1 ∧ d = 2) ∨ (c = 1 ∧ e = 2) ∨ (d = 1 ∧ e = 2)

/-- Condition for equal division of remaining bars -/
def canDivideEqually (bars : GoldBars) : Prop :=
  (bars.c + bars.d + bars.e = bars.a + bars.b) ∧
  (bars.b + bars.d + bars.e = bars.a + bars.c) ∧
  (bars.b + bars.c + bars.e = bars.a + bars.d) ∧
  (bars.a + bars.d + bars.e = bars.b + bars.c) ∧
  (bars.a + bars.c + bars.e = bars.b + bars.d) ∧
  (bars.a + bars.b + bars.e = bars.c + bars.d) ∧
  (bars.a + bars.c + bars.d = bars.b + bars.e) ∧
  (bars.a + bars.b + bars.d = bars.c + bars.e) ∧
  (bars.a + bars.b + bars.c = bars.d + bars.e)

/-- The main theorem -/
theorem gold_bars_theorem (bars : GoldBars) (h : canDivideEqually bars) :
  (bars.a = 1 ∧ bars.b = 1 ∧ bars.c = 2 ∧ bars.d = 2 ∧ bars.e = 2) ∨
  (bars.a = 1 ∧ bars.b = 2 ∧ bars.c = 3 ∧ bars.d = 3 ∧ bars.e = 3) ∨
  (bars.a = 1 ∧ bars.b = 1 ∧ bars.c = 1 ∧ bars.d = 1 ∧ bars.e = 2) :=
by sorry

end gold_bars_theorem_l308_30815


namespace ratio_from_equation_l308_30825

theorem ratio_from_equation (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end ratio_from_equation_l308_30825


namespace appropriate_sampling_methods_l308_30828

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Random
  | Stratified
  | Systematic

/-- Represents a population with subgroups -/
structure Population where
  total : ℕ
  subgroups : List ℕ

/-- Represents a sampling scenario -/
structure SamplingScenario where
  population : Population
  sample_size : ℕ

/-- Determines the most appropriate sampling method for a given scenario -/
def most_appropriate_method (scenario : SamplingScenario) : SamplingMethod :=
  sorry

/-- The student population -/
def student_population : Population :=
  { total := 10000, subgroups := [2000, 4500, 3500] }

/-- The product population -/
def product_population : Population :=
  { total := 1002, subgroups := [1002] }

/-- The student sampling scenario -/
def student_scenario : SamplingScenario :=
  { population := student_population, sample_size := 200 }

/-- The product sampling scenario -/
def product_scenario : SamplingScenario :=
  { population := product_population, sample_size := 20 }

theorem appropriate_sampling_methods :
  (most_appropriate_method student_scenario = SamplingMethod.Stratified) ∧
  (most_appropriate_method product_scenario = SamplingMethod.Systematic) :=
sorry

end appropriate_sampling_methods_l308_30828


namespace slope_from_sin_cos_sum_l308_30882

theorem slope_from_sin_cos_sum (θ : Real) 
  (h : Real.sin θ + Real.cos θ = Real.sqrt 5 / 5) : 
  Real.tan θ = -2 := by
  sorry

end slope_from_sin_cos_sum_l308_30882


namespace E_Z_eq_l308_30808

variable (p : ℝ)

-- Assumption that p is a probability
axiom h_p_prob : 0 < p ∧ p < 1

-- Definition of the probability mass function for Y
def P_Y (k : ℕ) : ℝ := p * (1 - p) ^ (k - 1)

-- Definition of the probability mass function for Z
def P_Z (k : ℕ) : ℝ := 
  if k ≥ 2 then p * (1 - p) ^ (k - 1) + (1 - p) * p ^ (k - 1) else 0

-- Expected value of Y
axiom E_Y : ∑' k, k * P_Y p k = 1 / p

-- Theorem to prove
theorem E_Z_eq : ∑' k, k * P_Z p k = 1 / (p * (1 - p)) - 1 := by sorry

end E_Z_eq_l308_30808


namespace perfect_squares_theorem_l308_30822

theorem perfect_squares_theorem :
  -- Part 1: Infinitely many n such that 2n+1 and 3n+1 are perfect squares
  (∃ f : ℕ → ℤ, ∀ k, ∃ a b : ℤ, 2 * f k + 1 = a^2 ∧ 3 * f k + 1 = b^2) ∧
  -- Part 2: Such n are multiples of 40
  (∀ n : ℤ, (∃ a b : ℤ, 2 * n + 1 = a^2 ∧ 3 * n + 1 = b^2) → ∃ k : ℤ, n = 40 * k) ∧
  -- Part 3: Generalization for any positive integer m
  (∀ m : ℕ, m > 0 →
    ∃ g : ℕ → ℤ, ∀ k, ∃ a b : ℤ, m * g k + 1 = a^2 ∧ (m + 1) * g k + 1 = b^2) :=
by sorry

end perfect_squares_theorem_l308_30822


namespace price_difference_l308_30881

/-- The original price of the toy rabbit -/
def original_price : ℝ := 25

/-- The price increase percentage for Store A -/
def increase_percentage : ℝ := 0.1

/-- The price decrease percentage for Store A -/
def decrease_percentage_A : ℝ := 0.2

/-- The price decrease percentage for Store B -/
def decrease_percentage_B : ℝ := 0.1

/-- The final price of the toy rabbit in Store A -/
def price_A : ℝ := original_price * (1 + increase_percentage) * (1 - decrease_percentage_A)

/-- The final price of the toy rabbit in Store B -/
def price_B : ℝ := original_price * (1 - decrease_percentage_B)

/-- Theorem stating that the price in Store A is 0.5 yuan less than in Store B -/
theorem price_difference : price_B - price_A = 0.5 := by
  sorry

end price_difference_l308_30881


namespace complex_modulus_l308_30870

theorem complex_modulus (z : ℂ) (h : z * (1 - Complex.I) = -Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end complex_modulus_l308_30870


namespace ceiling_sum_sqrt_l308_30850

theorem ceiling_sum_sqrt : ⌈Real.sqrt 20⌉ + ⌈Real.sqrt 200⌉ + ⌈Real.sqrt 2000⌉ = 65 := by
  sorry

end ceiling_sum_sqrt_l308_30850


namespace smallest_gcd_bc_l308_30819

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 130) (h2 : Nat.gcd a c = 770) :
  ∃ (b' c' : ℕ+), Nat.gcd a b' = 130 ∧ Nat.gcd a c' = 770 ∧ 
  Nat.gcd b' c' = 10 ∧ ∀ (b'' c'' : ℕ+), Nat.gcd a b'' = 130 → Nat.gcd a c'' = 770 → 
  Nat.gcd b'' c'' ≥ 10 :=
sorry

end smallest_gcd_bc_l308_30819


namespace product_not_divisible_by_sum_l308_30898

theorem product_not_divisible_by_sum (a b : ℕ) (h : a + b = 201) : ¬(201 ∣ (a * b)) := by
  sorry

end product_not_divisible_by_sum_l308_30898


namespace geometric_sequence_sum_l308_30856

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + a 3 = 3 →
  a 4 + a 6 = 6 →
  a 1 * a 3 + a 2 * a 4 + a 3 * a 5 + a 4 * a 6 + a 5 * a 7 = 62 := by
  sorry

end geometric_sequence_sum_l308_30856


namespace hyperbola_focus_k_value_l308_30831

/-- Given a hyperbola with equation x^2 - ky^2 = 1 and one focus at (3,0), prove that k = 1/8 -/
theorem hyperbola_focus_k_value (k : ℝ) : 
  (∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 - k*(y t)^2 = 1) →  -- Hyperbola equation
  (∃ (x₀ y₀ : ℝ), x₀^2 - k*y₀^2 = 1 ∧ x₀ = 3 ∧ y₀ = 0) →  -- Focus at (3,0)
  k = 1/8 := by
sorry

end hyperbola_focus_k_value_l308_30831


namespace semicircle_area_shaded_area_calculation_l308_30890

theorem semicircle_area (r : ℝ) (h : r = 2.5) : 
  (π * r^2) / 2 = 3.125 * π := by
  sorry

/- Definitions based on problem conditions -/
def semicircle_ADB_radius : ℝ := 2
def semicircle_BEC_radius : ℝ := 1
def point_D : ℝ × ℝ := (1, 2)  -- midpoint of arc ADB
def point_E : ℝ × ℝ := (3, 1)  -- midpoint of arc BEC
def point_F : ℝ × ℝ := (3, 2.5)  -- midpoint of arc DFE

/- Main theorem -/
theorem shaded_area_calculation : 
  let r : ℝ := semicircle_ADB_radius + semicircle_BEC_radius / 2
  (π * r^2) / 2 = 3.125 * π := by
  sorry

end semicircle_area_shaded_area_calculation_l308_30890


namespace min_good_pairs_l308_30895

/-- A circular arrangement of integers from 1 to 100 -/
def CircularArrangement := Fin 100 → ℕ

/-- Property that each number is either greater than both neighbors or less than both neighbors -/
def ValidArrangement (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 100, (arr i > arr (i - 1) ∧ arr i > arr (i + 1)) ∨ 
                 (arr i < arr (i - 1) ∧ arr i < arr (i + 1))

/-- Definition of a "good" pair -/
def GoodPair (arr : CircularArrangement) (i : Fin 100) : Prop :=
  ValidArrangement (Function.update (Function.update arr i (arr (i + 1))) (i + 1) (arr i))

/-- The main theorem stating that any valid arrangement has at least 51 good pairs -/
theorem min_good_pairs (arr : CircularArrangement) (h : ValidArrangement arr) :
  ∃ (s : Finset (Fin 100)), s.card ≥ 51 ∧ ∀ i ∈ s, GoodPair arr i :=
sorry

end min_good_pairs_l308_30895


namespace james_room_area_l308_30877

/-- Calculates the total area of rooms given initial dimensions and modifications --/
def total_area (initial_length initial_width increase : ℕ) : ℕ :=
  let new_length := initial_length + increase
  let new_width := initial_width + increase
  let single_room_area := new_length * new_width
  4 * single_room_area + 2 * single_room_area

/-- Theorem stating the total area for the given problem --/
theorem james_room_area :
  total_area 13 18 2 = 1800 := by
  sorry

end james_room_area_l308_30877


namespace point_on_graph_l308_30868

def f (x : ℝ) : ℝ := x + 1

theorem point_on_graph :
  f 0 = 1 ∧ 
  f 1 ≠ 1 ∧
  f 2 ≠ 0 ∧
  f (-1) ≠ 1 := by
sorry

end point_on_graph_l308_30868


namespace toms_cat_surgery_savings_l308_30809

/-- Calculates the savings made by having insurance for a pet's surgery --/
def calculate_insurance_savings (
  insurance_duration : ℕ
  ) (insurance_monthly_cost : ℝ
  ) (procedure_cost : ℝ
  ) (insurance_coverage_percentage : ℝ
  ) : ℝ :=
  let total_insurance_cost := insurance_duration * insurance_monthly_cost
  let out_of_pocket_cost := procedure_cost * (1 - insurance_coverage_percentage)
  let total_cost_with_insurance := out_of_pocket_cost + total_insurance_cost
  procedure_cost - total_cost_with_insurance

/-- Theorem stating that the savings made by having insurance for Tom's cat surgery is $3520 --/
theorem toms_cat_surgery_savings :
  calculate_insurance_savings 24 20 5000 0.8 = 3520 := by
  sorry

end toms_cat_surgery_savings_l308_30809


namespace max_k_for_no_real_roots_l308_30827

theorem max_k_for_no_real_roots (k : ℤ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k ≤ -2 :=
by sorry

end max_k_for_no_real_roots_l308_30827


namespace factory_bulb_supply_percentage_l308_30838

theorem factory_bulb_supply_percentage 
  (prob_x : ℝ) 
  (prob_y : ℝ) 
  (prob_total : ℝ) 
  (h1 : prob_x = 0.59) 
  (h2 : prob_y = 0.65) 
  (h3 : prob_total = 0.62) : 
  ∃ (p : ℝ), p * prob_x + (1 - p) * prob_y = prob_total ∧ p = 0.5 :=
by sorry

end factory_bulb_supply_percentage_l308_30838


namespace intersection_condition_area_condition_l308_30806

/-- The hyperbola C: x² - y² = 1 -/
def C (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- The line L: y = kx - 1 -/
def L (k x y : ℝ) : Prop := y = k * x - 1

/-- L intersects C at two distinct points -/
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ C x₁ y₁ ∧ C x₂ y₂ ∧ L k x₁ y₁ ∧ L k x₂ y₂

/-- The area of triangle AOB is √2 -/
def triangle_area_sqrt_2 (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, C x₁ y₁ ∧ C x₂ y₂ ∧ L k x₁ y₁ ∧ L k x₂ y₂ ∧
    (1/2 : ℝ) * |x₁ - x₂| = Real.sqrt 2

theorem intersection_condition (k : ℝ) :
  intersects_at_two_points k ↔ -Real.sqrt 2 < k ∧ k < -1 :=
sorry

theorem area_condition (k : ℝ) :
  triangle_area_sqrt_2 k ↔ k = 0 ∨ k = Real.sqrt 6 / 2 ∨ k = -Real.sqrt 6 / 2 :=
sorry

end intersection_condition_area_condition_l308_30806


namespace cube_root_approximation_l308_30885

-- Define k as the real cube root of 2
noncomputable def k : ℝ := Real.rpow 2 (1/3)

-- Define the inequality function
def inequality (A B C a b c : ℤ) (x : ℚ) : Prop :=
  x ≥ 0 → |((A * x^2 + B * x + C) / (a * x^2 + b * x + c) : ℝ) - k| < |x - k|

-- State the theorem
theorem cube_root_approximation :
  ∀ x : ℚ, inequality 2 2 2 1 2 2 x := by sorry

end cube_root_approximation_l308_30885


namespace deductive_reasoning_validity_l308_30857

/-- Represents a deductive reasoning argument -/
structure DeductiveArgument where
  major_premise : Prop
  minor_premise : Prop
  form_of_reasoning : Prop
  conclusion : Prop

/-- States that if all components of a deductive argument are correct, the conclusion must be correct -/
theorem deductive_reasoning_validity (arg : DeductiveArgument) :
  arg.major_premise → arg.minor_premise → arg.form_of_reasoning → arg.conclusion :=
by sorry

end deductive_reasoning_validity_l308_30857


namespace dollar_sum_squared_zero_l308_30860

/-- The dollar operation for real numbers -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem: For real numbers x and y, (x + y)²$(y + x)² = 0 -/
theorem dollar_sum_squared_zero (x y : ℝ) : dollar ((x + y)^2) ((y + x)^2) = 0 := by
  sorry

end dollar_sum_squared_zero_l308_30860


namespace abs_4y_minus_7_not_positive_l308_30864

theorem abs_4y_minus_7_not_positive (y : ℚ) :
  (|4 * y - 7| ≤ 0) ↔ (y = 7 / 4) := by sorry

end abs_4y_minus_7_not_positive_l308_30864


namespace tic_tac_toe_tie_probability_l308_30807

theorem tic_tac_toe_tie_probability (ben_win_prob tom_win_prob tie_prob : ℚ) : 
  ben_win_prob = 1/4 → tom_win_prob = 2/5 → tie_prob = 1 - (ben_win_prob + tom_win_prob) → 
  tie_prob = 7/20 := by
  sorry

end tic_tac_toe_tie_probability_l308_30807


namespace cosine_sine_identity_l308_30893

theorem cosine_sine_identity : 
  Real.cos (32 * π / 180) * Real.sin (62 * π / 180) - 
  Real.sin (32 * π / 180) * Real.sin (28 * π / 180) = 1 / 2 := by
  sorry

end cosine_sine_identity_l308_30893


namespace light_travel_distance_l308_30846

/-- The distance light travels in one year in a vacuum (in miles) -/
def light_speed_vacuum : ℝ := 5870000000000

/-- The factor by which light speed is reduced in the medium -/
def speed_reduction_factor : ℝ := 2

/-- The number of years we're considering -/
def years : ℝ := 1000

/-- The theorem stating the distance light travels in the given conditions -/
theorem light_travel_distance :
  (light_speed_vacuum / speed_reduction_factor) * years = 2935 * (10 ^ 12) := by
  sorry

end light_travel_distance_l308_30846


namespace students_taking_german_german_count_l308_30843

theorem students_taking_german (total : ℕ) (french : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  let students_taking_language := total - neither
  let only_french := french - both
  let only_german := students_taking_language - only_french - both
  only_german + both
  
theorem german_count :
  students_taking_german 94 41 9 40 = 22 := by sorry

end students_taking_german_german_count_l308_30843


namespace cubic_polynomial_coefficients_l308_30851

/-- Given a cubic polynomial with real coefficients that has 2 - 3i as a root,
    prove that its coefficients are a = -3, b = 1, and c = -39. -/
theorem cubic_polynomial_coefficients 
  (p : ℂ → ℂ) 
  (h1 : ∀ x, p x = x^3 + a*x^2 + b*x - c) 
  (h2 : p (2 - 3*I) = 0) 
  (a b c : ℝ) :
  a = -3 ∧ b = 1 ∧ c = -39 :=
sorry

end cubic_polynomial_coefficients_l308_30851


namespace sin_alpha_plus_5pi_12_l308_30820

theorem sin_alpha_plus_5pi_12 (α : Real) (h1 : 0 < α) (h2 : α < π / 2)
  (h3 : Real.cos α - Real.sin α = 2 * Real.sqrt 2 / 3) :
  Real.sin (α + 5 * π / 12) = (2 + Real.sqrt 15) / 6 := by
sorry

end sin_alpha_plus_5pi_12_l308_30820


namespace complex_number_calculation_l308_30878

theorem complex_number_calculation : 
  (Complex.mk 1 3) * (Complex.mk 2 (-4)) + (Complex.mk 2 5) * (Complex.mk 2 (-1)) = Complex.mk 13 10 := by
sorry

end complex_number_calculation_l308_30878


namespace problem_solution_l308_30803

theorem problem_solution (x y A : ℝ) 
  (h1 : 2^x = A) 
  (h2 : 7^(2*y) = A) 
  (h3 : 1/x + 1/y = 2) : 
  A = 7 * Real.sqrt 2 := by
sorry

end problem_solution_l308_30803


namespace annulus_area_dead_grass_area_l308_30852

/-- The area of an annulus formed by two concentric circles -/
theorem annulus_area (r₁ r₂ : ℝ) (h : 0 < r₁ ∧ r₁ < r₂) : 
  π * (r₂^2 - r₁^2) = π * (r₂ + r₁) * (r₂ - r₁) :=
by sorry

/-- The area of dead grass caused by a walking man with a sombrero -/
theorem dead_grass_area (r_walk r_sombrero : ℝ) 
  (h_walk : r_walk = 5)
  (h_sombrero : r_sombrero = 3) : 
  π * ((r_walk + r_sombrero)^2 - (r_walk - r_sombrero)^2) = 60 * π :=
by sorry

end annulus_area_dead_grass_area_l308_30852


namespace ellipse_intercept_inequality_l308_30862

-- Define the ellipse E
def E (m : ℝ) (x y : ℝ) : Prop := x^2 / m + y^2 / 4 = 1

-- Define the discriminant for the line y = kx + 1
def discriminant1 (m k : ℝ) : ℝ := 16 * m^2 * k^2 + 48 * m

-- Define the discriminant for the line kx + y - 2 = 0
def discriminant2 (m k : ℝ) : ℝ := 16 * m^2 * k^2

-- Theorem statement
theorem ellipse_intercept_inequality (m : ℝ) (h : m > 0) :
  ∀ k : ℝ, discriminant1 m k ≠ discriminant2 m k :=
sorry

end ellipse_intercept_inequality_l308_30862


namespace worker_payment_l308_30866

/-- The daily wage in rupees -/
def daily_wage : ℚ := 20

/-- The number of days worked in a week -/
def days_worked : ℚ := 11/3 + 2/3 + 1/8 + 3/4

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem worker_payment :
  round_to_nearest (daily_wage * days_worked) = 104 := by
  sorry

end worker_payment_l308_30866


namespace eight_monkeys_eat_fortyeight_bananas_l308_30880

/-- Given a rate at which monkeys eat bananas, calculate the number of monkeys needed to eat a certain number of bananas -/
def monkeys_needed (initial_monkeys initial_bananas target_bananas : ℕ) : ℕ :=
  initial_monkeys

/-- Theorem: Given that 8 monkeys take 8 minutes to eat 8 bananas, 8 monkeys are needed to eat 48 bananas -/
theorem eight_monkeys_eat_fortyeight_bananas :
  monkeys_needed 8 8 48 = 8 := by
  sorry

end eight_monkeys_eat_fortyeight_bananas_l308_30880


namespace max_value_cubic_function_l308_30853

theorem max_value_cubic_function :
  ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → x^3 - 3*x^2 + 2 ≤ 2 := by
  sorry

end max_value_cubic_function_l308_30853


namespace fixed_equidistant_point_l308_30835

-- Define the circles
variable (k₁ k₂ : Set (ℝ × ℝ))

-- Define the intersection point A
variable (A : ℝ × ℝ)

-- Define the particles P₁ and P₂ as functions of time
variable (P₁ P₂ : ℝ → ℝ × ℝ)

-- Define the constant angular speeds
variable (ω₁ ω₂ : ℝ)

-- Axioms
axiom circles_intersect : A ∈ k₁ ∩ k₂

axiom P₁_on_k₁ : ∀ t, P₁ t ∈ k₁
axiom P₂_on_k₂ : ∀ t, P₂ t ∈ k₂

axiom start_at_A : P₁ 0 = A ∧ P₂ 0 = A

axiom constant_speed : ∀ t, ‖(P₁ t).fst - (P₁ 0).fst‖ = ω₁ * t
                    ∧ ‖(P₂ t).fst - (P₂ 0).fst‖ = ω₂ * t

axiom same_direction : ω₁ * ω₂ > 0

axiom simultaneous_arrival : ∃ T > 0, P₁ T = A ∧ P₂ T = A

-- Theorem
theorem fixed_equidistant_point :
  ∃ P : ℝ × ℝ, ∀ t, ‖P - P₁ t‖ = ‖P - P₂ t‖ :=
sorry

end fixed_equidistant_point_l308_30835


namespace complex_cut_cube_edges_l308_30897

/-- A cube with complex cuts at each vertex -/
structure ComplexCutCube where
  /-- The number of vertices in the original cube -/
  originalVertices : Nat
  /-- The number of edges in the original cube -/
  originalEdges : Nat
  /-- The number of cuts per vertex -/
  cutsPerVertex : Nat
  /-- The number of new edges introduced per vertex due to cuts -/
  newEdgesPerVertex : Nat

/-- Theorem stating that a cube with complex cuts results in 60 edges -/
theorem complex_cut_cube_edges (c : ComplexCutCube) 
  (h1 : c.originalVertices = 8)
  (h2 : c.originalEdges = 12)
  (h3 : c.cutsPerVertex = 2)
  (h4 : c.newEdgesPerVertex = 6) : 
  c.originalEdges + c.originalVertices * c.newEdgesPerVertex = 60 := by
  sorry

/-- The total number of edges in a complex cut cube is 60 -/
def total_edges (c : ComplexCutCube) : Nat :=
  c.originalEdges + c.originalVertices * c.newEdgesPerVertex

#check complex_cut_cube_edges

end complex_cut_cube_edges_l308_30897


namespace absolute_value_equation_solution_l308_30837

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|x| - 2 = 1) ↔ (x = 3 ∨ x = -3) :=
sorry

end absolute_value_equation_solution_l308_30837


namespace eighth_term_is_22_l308_30884

/-- An arithmetic sequence with a_1 = 1 and sum of first 5 terms = 35 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (a 1 + a 2 + a 3 + a 4 + a 5 = 35)

/-- The 8th term of the sequence is 22 -/
theorem eighth_term_is_22 (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  a 8 = 22 := by
  sorry


end eighth_term_is_22_l308_30884


namespace platform_length_l308_30889

/-- Given a train with speed 72 km/hr and length 220 m, crossing a platform in 26 seconds,
    the length of the platform is 300 m. -/
theorem platform_length (train_speed : ℝ) (train_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 →
  train_length = 220 →
  crossing_time = 26 →
  (train_speed * (5/18) * crossing_time) - train_length = 300 := by
  sorry

#check platform_length

end platform_length_l308_30889


namespace problem_solution_l308_30869

theorem problem_solution : ∃! x : ℝ, 3 ∈ ({x + 2, x^2 + 2*x} : Set ℝ) ∧ x = -3 := by
  sorry

end problem_solution_l308_30869


namespace triangle_inequality_theorem_triangle_equality_condition_l308_30855

/-- Represents a triangle with side lengths and area -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_area : 0 < S
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The inequality holds for any triangle -/
theorem triangle_inequality_theorem (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 ≥ 4 * Real.sqrt 3 * t.S :=
sorry

/-- The equality condition for the theorem -/
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- The equality holds if and only if the triangle is equilateral -/
theorem triangle_equality_condition (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 = 4 * Real.sqrt 3 * t.S ↔ is_equilateral t :=
sorry

end triangle_inequality_theorem_triangle_equality_condition_l308_30855


namespace juice_bar_solution_l308_30821

/-- Represents the juice bar problem --/
def juice_bar_problem (total_spent : ℕ) (mango_price : ℕ) (pineapple_price : ℕ) (pineapple_spent : ℕ) : Prop :=
  ∃ (mango_count pineapple_count : ℕ),
    mango_count * mango_price + pineapple_count * pineapple_price = total_spent ∧
    pineapple_count * pineapple_price = pineapple_spent ∧
    mango_count + pineapple_count = 17

/-- The theorem stating the solution to the juice bar problem --/
theorem juice_bar_solution :
  juice_bar_problem 94 5 6 54 :=
sorry

end juice_bar_solution_l308_30821


namespace unique_last_digit_for_multiple_of_6_l308_30872

def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def last_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

theorem unique_last_digit_for_multiple_of_6 :
  ∃! d : ℕ, d < 10 ∧ is_multiple_of_6 (64310 + d) :=
by sorry

end unique_last_digit_for_multiple_of_6_l308_30872


namespace not_always_geometric_sequence_l308_30836

def is_geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ+, a (n + 1) = q * a n

theorem not_always_geometric_sequence :
  ¬ (∀ a : ℕ+ → ℝ, (∃ q : ℝ, ∀ n : ℕ+, a (n + 1) = q * a n) → is_geometric_sequence a) :=
by
  sorry

end not_always_geometric_sequence_l308_30836


namespace sum_of_first_and_last_l308_30886

/-- A sequence of eight terms -/
structure EightTermSequence where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ
  U : ℝ
  V : ℝ
  W : ℝ

/-- The sum of any four consecutive terms is 40 -/
def consecutive_sum_40 (seq : EightTermSequence) : Prop :=
  seq.P + seq.Q + seq.R + seq.S = 40 ∧
  seq.Q + seq.R + seq.S + seq.T = 40 ∧
  seq.R + seq.S + seq.T + seq.U = 40 ∧
  seq.S + seq.T + seq.U + seq.V = 40 ∧
  seq.T + seq.U + seq.V + seq.W = 40

theorem sum_of_first_and_last (seq : EightTermSequence) 
  (h1 : seq.S = 10)
  (h2 : consecutive_sum_40 seq) : 
  seq.P + seq.W = 40 := by
  sorry

end sum_of_first_and_last_l308_30886


namespace last_even_number_in_sequence_l308_30829

theorem last_even_number_in_sequence (n : ℕ) : 
  (4 * (n * (n + 1) * (2 * n + 1)) / 6 = 560) → n = 7 :=
by sorry

end last_even_number_in_sequence_l308_30829


namespace total_votes_polled_l308_30858

-- Define the total number of votes
variable (V : ℝ)

-- Define the number of votes for each candidate
variable (T S R F : ℝ)

-- Define the conditions
def condition1 : Prop := T = S + 0.15 * V
def condition2 : Prop := S = R + 0.05 * V
def condition3 : Prop := R = F + 0.07 * V
def condition4 : Prop := T + S + R + F = V
def condition5 : Prop := T - 2500 - 2000 = S + 2500
def condition6 : Prop := S + 2500 = R + 2000 + 0.05 * V

-- State the theorem
theorem total_votes_polled
  (h1 : condition1 V T S)
  (h2 : condition2 V S R)
  (h3 : condition3 V R F)
  (h4 : condition4 V T S R F)
  (h5 : condition5 T S)
  (h6 : condition6 V S R) :
  V = 30000 := by
  sorry


end total_votes_polled_l308_30858


namespace second_number_value_l308_30845

theorem second_number_value (x y : ℝ) 
  (h1 : (1/5) * x = (5/8) * y) 
  (h2 : x + 35 = 4 * y) : 
  y = 40 := by
sorry

end second_number_value_l308_30845


namespace negation_of_universal_proposition_l308_30875

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 + x - 1 ≤ 0) ↔ (∃ x : ℝ, 2 * x^2 + x - 1 > 0) := by
  sorry

end negation_of_universal_proposition_l308_30875


namespace second_number_value_l308_30844

theorem second_number_value (a b c : ℝ) : 
  a + b + c = 660 ∧ 
  a = 2 * b ∧ 
  c = (1/3) * a →
  b = 180 := by sorry

end second_number_value_l308_30844


namespace real_solutions_of_equation_l308_30826

theorem real_solutions_of_equation (x : ℝ) : 
  x^4 + (3 - x)^4 = 82 ↔ x = 2.5 ∨ x = 0.5 := by
sorry

end real_solutions_of_equation_l308_30826


namespace no_real_sqrt_negative_l308_30830

theorem no_real_sqrt_negative : ∃ (a b c d : ℝ), 
  (a = (-3)^2 ∧ ∃ x : ℝ, x^2 = a) ∧ 
  (b = 0 ∧ ∃ x : ℝ, x^2 = b) ∧ 
  (c = 1/8 ∧ ∃ x : ℝ, x^2 = c) ∧ 
  (d = -6^3 ∧ ¬∃ x : ℝ, x^2 = d) :=
by sorry

end no_real_sqrt_negative_l308_30830


namespace sachins_age_l308_30865

theorem sachins_age (sachin rahul : ℕ) : 
  rahul = sachin + 18 →
  sachin * 9 = rahul * 7 →
  sachin = 63 := by sorry

end sachins_age_l308_30865


namespace playstation_value_l308_30879

theorem playstation_value (computer_cost accessories_cost out_of_pocket : ℝ) 
  (h1 : computer_cost = 700)
  (h2 : accessories_cost = 200)
  (h3 : out_of_pocket = 580) : 
  ∃ (playstation_value : ℝ), 
    playstation_value = 400 ∧ 
    computer_cost + accessories_cost = out_of_pocket + playstation_value * 0.8 := by
  sorry

end playstation_value_l308_30879


namespace deepak_age_l308_30867

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 6 = 22 →
  deepak_age = 12 := by
sorry

end deepak_age_l308_30867


namespace simplify_fraction_simplify_and_evaluate_evaluate_at_two_l308_30812

-- Part 1
theorem simplify_fraction (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  (x - 1) / x / ((2 * x - 2) / x^2) = x / 2 := by sorry

-- Part 2
theorem simplify_and_evaluate (a : ℝ) (ha : a ≠ -1) :
  (2 - (a - 1) / (a + 1)) / ((a^2 + 6*a + 9) / (a + 1)) = 1 / (a + 3) := by sorry

theorem evaluate_at_two :
  (2 - (2 - 1) / (2 + 1)) / ((2^2 + 6*2 + 9) / (2 + 1)) = 1 / 5 := by sorry

end simplify_fraction_simplify_and_evaluate_evaluate_at_two_l308_30812


namespace quadratic_value_theorem_l308_30874

theorem quadratic_value_theorem (m : ℝ) (h : m^2 - 2*m - 1 = 0) :
  3*m^2 - 6*m + 2020 = 2023 := by
  sorry

end quadratic_value_theorem_l308_30874


namespace division_problem_l308_30801

theorem division_problem (dividend quotient remainder : ℕ) (h : dividend = 162 ∧ quotient = 9 ∧ remainder = 9) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 17 := by
sorry

end division_problem_l308_30801


namespace insect_jump_coordinates_l308_30863

/-- A point in a 2D plane represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a jump to the right by a certain distance -/
def jumpRight (p : Point) (distance : ℝ) : Point :=
  ⟨p.x + distance, p.y⟩

theorem insect_jump_coordinates :
  let A : Point := ⟨-2, 1⟩
  let B : Point := jumpRight A 4
  B.x = 2 ∧ B.y = 1 := by
  sorry

end insect_jump_coordinates_l308_30863


namespace odd_function_sum_zero_l308_30848

/-- A function v is odd if v(-x) = -v(x) for all x in its domain -/
def IsOdd (v : ℝ → ℝ) : Prop := ∀ x, v (-x) = -v x

/-- The sum of v(-3.14), v(-1.57), v(1.57), and v(3.14) is zero for any odd function v -/
theorem odd_function_sum_zero (v : ℝ → ℝ) (h : IsOdd v) :
  v (-3.14) + v (-1.57) + v 1.57 + v 3.14 = 0 :=
sorry

end odd_function_sum_zero_l308_30848


namespace gustran_facial_cost_l308_30861

/-- Represents the prices of services at a salon -/
structure SalonPrices where
  haircut : ℕ
  nails : ℕ
  facial : ℕ

/-- Calculates the total cost of services at a salon -/
def totalCost (prices : SalonPrices) : ℕ :=
  prices.haircut + prices.nails + prices.facial

theorem gustran_facial_cost (gustran : SalonPrices) (barbara : SalonPrices) (fancy : SalonPrices)
  (h1 : gustran.haircut = 45)
  (h2 : gustran.nails = 30)
  (h3 : barbara.haircut = 30)
  (h4 : barbara.nails = 40)
  (h5 : barbara.facial = 28)
  (h6 : fancy.haircut = 34)
  (h7 : fancy.nails = 20)
  (h8 : fancy.facial = 30)
  (h9 : totalCost fancy = 84)
  (h10 : totalCost fancy ≤ totalCost barbara)
  (h11 : totalCost fancy ≤ totalCost gustran) :
  gustran.facial = 9 := by
  sorry

end gustran_facial_cost_l308_30861


namespace factorization_proof_l308_30891

theorem factorization_proof (a : ℝ) : 2 * a^2 - 2 * a + (1/2 : ℝ) = 2 * (a - 1/2)^2 := by
  sorry

end factorization_proof_l308_30891


namespace existence_of_m_n_l308_30818

theorem existence_of_m_n (p : Nat) (hp : p.Prime) (hp10 : p > 10) :
  ∃ m n : Nat, m > 0 ∧ n > 0 ∧ m + n < p ∧ (5^m * 7^n - 1) % p = 0 := by
  sorry

end existence_of_m_n_l308_30818


namespace no_collision_after_jumps_l308_30824

/-- Represents the position of a grasshopper -/
structure Position where
  x : Int
  y : Int

/-- Represents the state of the system with four grasshoppers -/
structure GrasshopperSystem where
  positions : Fin 4 → Position

/-- Performs a symmetric jump for one grasshopper -/
def symmetricJump (system : GrasshopperSystem) (jumper : Fin 4) : GrasshopperSystem :=
  sorry

/-- Checks if any two grasshoppers occupy the same position -/
def hasCollision (system : GrasshopperSystem) : Bool :=
  sorry

/-- Initial configuration of the grasshoppers on a square -/
def initialSquare : GrasshopperSystem :=
  { positions := λ i => match i with
    | 0 => ⟨0, 0⟩
    | 1 => ⟨0, 1⟩
    | 2 => ⟨1, 1⟩
    | 3 => ⟨1, 0⟩ }

theorem no_collision_after_jumps :
  ∀ (jumps : List (Fin 4)), ¬(hasCollision (jumps.foldl symmetricJump initialSquare)) :=
  sorry

end no_collision_after_jumps_l308_30824


namespace square_sum_and_product_l308_30817

theorem square_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 1) 
  (h2 : (x - y)^2 = 49) : 
  x^2 + y^2 = 25 ∧ x * y = -12 := by
  sorry

end square_sum_and_product_l308_30817


namespace prime_sum_of_powers_l308_30899

theorem prime_sum_of_powers (a b p : ℕ) : 
  Nat.Prime a → Nat.Prime b → Nat.Prime p → p = a^b + b^a → p = 17 := by
  sorry

end prime_sum_of_powers_l308_30899


namespace least_k_for_subset_sum_l308_30871

theorem least_k_for_subset_sum (n : ℕ) :
  let k := if n % 2 = 1 then 2 * n else n + 1
  ∀ (A : Finset ℕ), A.card ≥ k →
    ∃ (S : Finset ℕ), S ⊆ A ∧ S.card % 2 = 0 ∧ (S.sum id) % n = 0 :=
by sorry

end least_k_for_subset_sum_l308_30871


namespace saree_sale_price_l308_30823

/-- Applies a discount to a given price -/
def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

/-- Calculates the final price after applying multiple discounts -/
def final_price (original_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount original_price

theorem saree_sale_price :
  let original_price : ℝ := 175
  let discounts : List ℝ := [0.30, 0.25, 0.15, 0.10]
  let result := final_price original_price discounts
  ∃ ε > 0, |result - 70.28| < ε :=
sorry

end saree_sale_price_l308_30823


namespace f_is_even_l308_30888

def f (x : ℝ) : ℝ := (x + 2)^2 + (2*x - 1)^2

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end f_is_even_l308_30888


namespace chord_length_inequality_l308_30849

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

-- Define the line y = kx + 1
def line1 (k x y : ℝ) : Prop := y = k*x + 1

-- Define the line kx + y - 2 = 0
def line2 (k x y : ℝ) : Prop := k*x + y - 2 = 0

-- Define a function to calculate the chord length
noncomputable def chordLength (k : ℝ) (line : ℝ → ℝ → ℝ → Prop) : ℝ :=
  sorry -- Actual calculation of chord length would go here

-- Theorem statement
theorem chord_length_inequality (k : ℝ) :
  chordLength k line1 ≠ chordLength k line2 :=
sorry

end chord_length_inequality_l308_30849


namespace files_remaining_l308_30811

theorem files_remaining (music_files : ℕ) (video_files : ℕ) (deleted_files : ℕ)
  (h1 : music_files = 4)
  (h2 : video_files = 21)
  (h3 : deleted_files = 23) :
  music_files + video_files - deleted_files = 2 :=
by
  sorry

end files_remaining_l308_30811


namespace unique_remainder_modulo_8_and_13_l308_30847

theorem unique_remainder_modulo_8_and_13 : 
  ∃! n : ℕ, n < 180 ∧ n % 8 = 2 ∧ n % 13 = 2 :=
sorry

end unique_remainder_modulo_8_and_13_l308_30847


namespace total_spent_calculation_l308_30810

-- Define currency exchange rates
def gbp_to_usd : ℝ := 1.38
def eur_to_usd : ℝ := 1.12
def jpy_to_usd : ℝ := 0.0089

-- Define purchases
def speakers_cost : ℝ := 136.01
def cd_player_cost : ℝ := 139.38
def tires_cost_gbp : ℝ := 85.62
def tires_quantity : ℕ := 4
def printer_cables_cost_eur : ℝ := 12.54
def printer_cables_quantity : ℕ := 2
def blank_cds_cost_jpy : ℝ := 9800

-- Define sales tax rate
def sales_tax_rate : ℝ := 0.0825

-- Theorem statement
theorem total_spent_calculation :
  let usd_taxable := speakers_cost + cd_player_cost
  let usd_tax := usd_taxable * sales_tax_rate
  let usd_with_tax := usd_taxable + usd_tax
  let tires_usd := (tires_cost_gbp * tires_quantity) * gbp_to_usd
  let cables_usd := (printer_cables_cost_eur * printer_cables_quantity) * eur_to_usd
  let cds_usd := blank_cds_cost_jpy * jpy_to_usd
  usd_with_tax + tires_usd + cables_usd + cds_usd = 886.04 := by
  sorry

end total_spent_calculation_l308_30810


namespace expression_simplification_and_evaluation_expression_evaluation_at_one_l308_30859

theorem expression_simplification_and_evaluation (x : ℝ) (h : x ≠ 3) :
  (x^2 - 5) / (x - 3) - 4 / (x - 3) = x + 3 :=
by sorry

theorem expression_evaluation_at_one :
  ((1 : ℝ)^2 - 5) / (1 - 3) - 4 / (1 - 3) = 4 :=
by sorry

end expression_simplification_and_evaluation_expression_evaluation_at_one_l308_30859


namespace meet_five_times_l308_30841

/-- Represents the meeting problem between Michael and the garbage truck --/
structure MeetingProblem where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def number_of_meetings (problem : MeetingProblem) : ℕ :=
  sorry

/-- The theorem stating that Michael and the truck meet exactly 5 times --/
theorem meet_five_times (problem : MeetingProblem) 
  (h1 : problem.michael_speed = 5)
  (h2 : problem.truck_speed = 10)
  (h3 : problem.pail_distance = 200)
  (h4 : problem.truck_stop_time = 30) :
  number_of_meetings problem = 5 :=
  sorry

end meet_five_times_l308_30841


namespace existence_of_composite_nx_plus_one_l308_30873

theorem existence_of_composite_nx_plus_one (n : ℤ) : ∃ x : ℤ, ∃ a b : ℤ, a > 1 ∧ b > 1 ∧ n * x + 1 = a * b := by
  sorry

end existence_of_composite_nx_plus_one_l308_30873


namespace gcd_of_squares_sum_l308_30876

theorem gcd_of_squares_sum : Nat.gcd (130^2 + 241^2 + 352^2) (129^2 + 240^2 + 353^2 + 2^3) = 1 := by
  sorry

end gcd_of_squares_sum_l308_30876


namespace mailman_theorem_l308_30814

def mailman_problem (mails_per_block : ℕ) (houses_per_block : ℕ) : Prop :=
  mails_per_block / houses_per_block = 8

theorem mailman_theorem :
  mailman_problem 32 4 :=
by
  sorry

end mailman_theorem_l308_30814


namespace nina_money_problem_l308_30834

theorem nina_money_problem (W : ℝ) (M : ℝ) :
  (10 * W = M) →
  (14 * (W - 1.75) = M) →
  M = 61.25 := by
sorry

end nina_money_problem_l308_30834


namespace percentage_problem_l308_30833

theorem percentage_problem (x : ℝ) (p : ℝ) : 
  x = 780 →
  (25 / 100) * x = (p / 100) * 1500 - 30 →
  p = 15 := by
sorry

end percentage_problem_l308_30833


namespace smallest_five_digit_congruent_to_21_mod_30_l308_30896

theorem smallest_five_digit_congruent_to_21_mod_30 :
  ∀ n : ℕ, 
    n ≥ 10000 ∧ n ≤ 99999 ∧ n ≡ 21 [MOD 30] → 
    n ≥ 10011 :=
by sorry

end smallest_five_digit_congruent_to_21_mod_30_l308_30896


namespace initial_plus_rainfall_equals_final_l308_30894

/-- Represents the rainfall data for a single day -/
structure RainfallData where
  rate1 : Real  -- rainfall rate from 2pm to 4pm in inches per hour
  duration1 : Real  -- duration of rainfall from 2pm to 4pm in hours
  rate2 : Real  -- rainfall rate from 4pm to 7pm in inches per hour
  duration2 : Real  -- duration of rainfall from 4pm to 7pm in hours
  rate3 : Real  -- rainfall rate from 7pm to 9pm in inches per hour
  duration3 : Real  -- duration of rainfall from 7pm to 9pm in hours
  final_amount : Real  -- amount of water in the gauge at 9pm in inches

/-- Calculates the total rainfall during the day -/
def total_rainfall (data : RainfallData) : Real :=
  data.rate1 * data.duration1 + data.rate2 * data.duration2 + data.rate3 * data.duration3

/-- Theorem stating that the initial amount plus total rainfall equals the final amount -/
theorem initial_plus_rainfall_equals_final (data : RainfallData) 
    (h1 : data.rate1 = 4) (h2 : data.duration1 = 2)
    (h3 : data.rate2 = 3) (h4 : data.duration2 = 3)
    (h5 : data.rate3 = 0.5) (h6 : data.duration3 = 2)
    (h7 : data.final_amount = 20) :
    ∃ initial_amount : Real, initial_amount + total_rainfall data = data.final_amount := by
  sorry

end initial_plus_rainfall_equals_final_l308_30894


namespace triangle_inequality_for_positive_reals_l308_30892

theorem triangle_inequality_for_positive_reals (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  Real.sqrt (a^2 + b^2) ≤ a + b := by sorry

end triangle_inequality_for_positive_reals_l308_30892


namespace line_passes_through_point_l308_30805

/-- A line in the form kx - y + 1 - 3k = 0 always passes through (3, 1) -/
theorem line_passes_through_point :
  ∀ (k : ℝ), (3 * k : ℝ) - 1 + 1 - 3 * k = 0 := by
  sorry

end line_passes_through_point_l308_30805


namespace rectangle_area_difference_rectangle_area_difference_is_196_l308_30802

theorem rectangle_area_difference : ℕ → Prop :=
  fun diff =>
    ∀ l w : ℕ,
      (l > 0 ∧ w > 0) →  -- Ensure positive side lengths
      (2 * l + 2 * w = 60) →  -- Perimeter condition
      ∃ l_max w_max l_min w_min : ℕ,
        (l_max > 0 ∧ w_max > 0 ∧ l_min > 0 ∧ w_min > 0) →
        (2 * l_max + 2 * w_max = 60) →
        (2 * l_min + 2 * w_min = 60) →
        (∀ l' w' : ℕ, (l' > 0 ∧ w' > 0) → (2 * l' + 2 * w' = 60) → 
          l' * w' ≤ l_max * w_max ∧ l' * w' ≥ l_min * w_min) →
        diff = l_max * w_max - l_min * w_min

theorem rectangle_area_difference_is_196 : rectangle_area_difference 196 := by
  sorry

end rectangle_area_difference_rectangle_area_difference_is_196_l308_30802
