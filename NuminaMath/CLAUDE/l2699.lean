import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_l2699_269940

theorem inequality_solution (x : ℝ) :
  -2 < (x^2 - 18*x + 35) / (x^2 - 4*x + 8) ∧
  (x^2 - 18*x + 35) / (x^2 - 4*x + 8) < 2 →
  3 < x ∧ x < 17/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2699_269940


namespace NUMINAMATH_CALUDE_pencil_count_l2699_269917

/-- Proves that given the ratio of pens to pencils is 5:6 and there are 9 more pencils than pens, the number of pencils is 54. -/
theorem pencil_count (pens pencils : ℕ) : 
  (pens : ℚ) / (pencils : ℚ) = 5 / 6 →
  pencils = pens + 9 →
  pencils = 54 := by
sorry

end NUMINAMATH_CALUDE_pencil_count_l2699_269917


namespace NUMINAMATH_CALUDE_g_of_three_l2699_269918

/-- Given a function g such that g(x-1) = 2x + 6 for all x, prove that g(3) = 14 -/
theorem g_of_three (g : ℝ → ℝ) (h : ∀ x, g (x - 1) = 2 * x + 6) : g 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_g_of_three_l2699_269918


namespace NUMINAMATH_CALUDE_symmetric_distribution_within_one_std_dev_l2699_269969

/-- A symmetric distribution about a mean -/
structure SymmetricDistribution where
  mean : ℝ
  std_dev : ℝ
  is_symmetric : Bool
  percent_less_than_mean_plus_std_dev : ℝ

/-- The percentage of a symmetric distribution within one standard deviation of the mean -/
def percent_within_one_std_dev (dist : SymmetricDistribution) : ℝ :=
  2 * (dist.percent_less_than_mean_plus_std_dev - 50)

theorem symmetric_distribution_within_one_std_dev 
  (dist : SymmetricDistribution) 
  (h1 : dist.is_symmetric = true) 
  (h2 : dist.percent_less_than_mean_plus_std_dev = 82) :
  percent_within_one_std_dev dist = 64 := by
  sorry

#check symmetric_distribution_within_one_std_dev

end NUMINAMATH_CALUDE_symmetric_distribution_within_one_std_dev_l2699_269969


namespace NUMINAMATH_CALUDE_max_participants_l2699_269977

/-- Represents a round-robin chess tournament. -/
structure ChessTournament where
  n : ℕ  -- number of players
  a : ℕ  -- number of draws
  b : ℕ  -- number of wins (and losses)

/-- The conditions of the tournament are met. -/
def validTournament (t : ChessTournament) : Prop :=
  t.n > 0 ∧
  2 * t.a + 3 * t.b = 120 ∧
  t.a + t.b = t.n * (t.n - 1) / 2

/-- The maximum number of participants in the tournament is 11. -/
theorem max_participants :
  ∀ t : ChessTournament, validTournament t → t.n ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_participants_l2699_269977


namespace NUMINAMATH_CALUDE_oil_leak_calculation_l2699_269932

/-- The total amount of oil leaked from a broken pipe -/
def total_oil_leaked (before_fixing : ℕ) (while_fixing : ℕ) : ℕ :=
  before_fixing + while_fixing

/-- Theorem: Given the specific amounts of oil leaked before and during fixing,
    the total amount of oil leaked is 6206 gallons -/
theorem oil_leak_calculation :
  total_oil_leaked 2475 3731 = 6206 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_calculation_l2699_269932


namespace NUMINAMATH_CALUDE_root_equation_value_l2699_269957

theorem root_equation_value (a : ℝ) : 
  a^2 + 3*a - 1 = 0 → 2*a^2 + 6*a + 2021 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l2699_269957


namespace NUMINAMATH_CALUDE_larger_number_problem_l2699_269911

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : max x y = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2699_269911


namespace NUMINAMATH_CALUDE_star_eight_four_l2699_269992

-- Define the & operation
def amp (a b : ℝ) : ℝ := (a + b) * (a - b)

-- Define the ★ operation
def star (c d : ℝ) : ℝ := amp c d + 2 * (c + d)

-- Theorem to prove
theorem star_eight_four : star 8 4 = 72 := by
  sorry

end NUMINAMATH_CALUDE_star_eight_four_l2699_269992


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l2699_269922

/-- The volume of a rectangular box with face areas 36, 18, and 12 square inches -/
theorem rectangular_box_volume (l w h : ℝ) 
  (face1 : l * w = 36)
  (face2 : w * h = 18)
  (face3 : l * h = 12) :
  l * w * h = 36 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l2699_269922


namespace NUMINAMATH_CALUDE_interest_rate_is_twelve_percent_l2699_269968

/-- Given a banker's gain and true discount on a bill due in 1 year,
    calculate the rate of interest per annum. -/
def calculate_interest_rate (bankers_gain : ℚ) (true_discount : ℚ) : ℚ :=
  bankers_gain / true_discount

/-- Theorem stating that for the given banker's gain and true discount,
    the calculated interest rate is 12% -/
theorem interest_rate_is_twelve_percent :
  let bankers_gain : ℚ := 78/10
  let true_discount : ℚ := 65
  calculate_interest_rate bankers_gain true_discount = 12/100 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_twelve_percent_l2699_269968


namespace NUMINAMATH_CALUDE_line_passes_through_point_l2699_269923

/-- Given a line equation 3x + ay - 5 = 0 passing through point A(1, 2), prove that a = 1 -/
theorem line_passes_through_point (a : ℝ) : 
  (3 * 1 + a * 2 - 5 = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l2699_269923


namespace NUMINAMATH_CALUDE_total_turnips_l2699_269997

theorem total_turnips (keith_turnips alyssa_turnips : ℕ) 
  (h1 : keith_turnips = 6) 
  (h2 : alyssa_turnips = 9) : 
  keith_turnips + alyssa_turnips = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_l2699_269997


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2699_269937

theorem smallest_positive_integer_with_remainders : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (x % 4 = 3) ∧ 
  (x % 6 = 5) ∧ 
  (∀ (y : ℕ), y > 0 → y % 4 = 3 → y % 6 = 5 → x ≤ y) ∧
  (x = 11) := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2699_269937


namespace NUMINAMATH_CALUDE_donut_area_l2699_269928

/-- The area of a donut shape formed by two concentric circles -/
theorem donut_area (r₁ r₂ : ℝ) (h₁ : r₁ = 7) (h₂ : r₂ = 10) :
  (r₂^2 - r₁^2) * π = 51 * π := by
  sorry

#check donut_area

end NUMINAMATH_CALUDE_donut_area_l2699_269928


namespace NUMINAMATH_CALUDE_rope_cutting_ratio_l2699_269963

/-- Given a rope of initial length 100 feet, prove that after specific cuts,
    the ratio of the final piece to its parent piece is 1:5 -/
theorem rope_cutting_ratio :
  ∀ (initial_length : ℝ) (final_piece_length : ℝ),
    initial_length = 100 →
    final_piece_length = 5 →
    ∃ (second_cut_length : ℝ),
      second_cut_length = initial_length / 4 ∧
      final_piece_length / second_cut_length = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_ratio_l2699_269963


namespace NUMINAMATH_CALUDE_arrange_objects_count_l2699_269943

/-- The number of ways to arrange 7 indistinguishable objects of one type
    and 3 indistinguishable objects of another type in a row of 10 positions -/
def arrangeObjects : ℕ := Nat.choose 10 3

/-- Theorem stating that the number of arrangements is equal to binomial coefficient (10 choose 3) -/
theorem arrange_objects_count : arrangeObjects = 120 := by
  sorry

end NUMINAMATH_CALUDE_arrange_objects_count_l2699_269943


namespace NUMINAMATH_CALUDE_optimal_profit_distribution_l2699_269941

/-- Represents the profit and production setup for handicrafts A and B --/
structure HandicraftSetup where
  profit_diff : ℝ  -- Profit difference between B and A
  profit_A_equal : ℝ  -- Profit of A when quantities are equal
  profit_B_equal : ℝ  -- Profit of B when quantities are equal
  total_workers : ℕ  -- Total number of workers
  A_production_rate : ℕ  -- Number of A pieces one worker can produce
  B_production_rate : ℕ  -- Number of B pieces one worker can produce
  min_B_production : ℕ  -- Minimum number of B pieces to be produced
  profit_decrease_rate : ℝ  -- Rate of profit decrease per extra B piece

/-- Calculates the maximum profit for the given handicraft setup --/
def max_profit (setup : HandicraftSetup) : ℝ :=
  let profit_A := setup.profit_A_equal * setup.profit_B_equal / (setup.profit_B_equal - setup.profit_diff)
  let profit_B := profit_A + setup.profit_diff
  let m := setup.total_workers / 2  -- Approximate midpoint for worker distribution
  (-2) * (m - 25)^2 + 3200

/-- Theorem stating the maximum profit and optimal worker distribution --/
theorem optimal_profit_distribution (setup : HandicraftSetup) :
  setup.profit_diff = 105 ∧
  setup.profit_A_equal = 30 ∧
  setup.profit_B_equal = 240 ∧
  setup.total_workers = 65 ∧
  setup.A_production_rate = 2 ∧
  setup.B_production_rate = 1 ∧
  setup.min_B_production = 5 ∧
  setup.profit_decrease_rate = 2 →
  max_profit setup = 3200 ∧
  ∃ (workers_A workers_B : ℕ),
    workers_A = 40 ∧
    workers_B = 25 ∧
    workers_A + workers_B = setup.total_workers :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_profit_distribution_l2699_269941


namespace NUMINAMATH_CALUDE_minutes_before_noon_l2699_269952

theorem minutes_before_noon (x : ℕ) : x = 65 :=
  -- Define the conditions
  let minutes_between_9am_and_12pm := 180
  let minutes_ago := 20
  -- The equation: 180 - (x - 20) = 3 * (x - 20)
  have h : minutes_between_9am_and_12pm - (x - minutes_ago) = 3 * (x - minutes_ago) := by sorry
  -- Prove that x = 65
  sorry

#check minutes_before_noon

end NUMINAMATH_CALUDE_minutes_before_noon_l2699_269952


namespace NUMINAMATH_CALUDE_equation_solution_l2699_269990

theorem equation_solution (x : ℝ) : 14 * x + 5 - 21 * x^2 = -2 → 6 * x^2 - 4 * x + 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2699_269990


namespace NUMINAMATH_CALUDE_product_of_fractions_l2699_269975

theorem product_of_fractions : (2 : ℚ) / 3 * (5 : ℚ) / 11 = 10 / 33 := by sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2699_269975


namespace NUMINAMATH_CALUDE_age_difference_l2699_269965

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 17) : A - C = 17 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2699_269965


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_five_l2699_269942

theorem largest_four_digit_divisible_by_five : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 5 = 0 → n ≤ 9995 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_five_l2699_269942


namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l2699_269903

theorem quadratic_form_k_value (a h k : ℚ) :
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) →
  k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l2699_269903


namespace NUMINAMATH_CALUDE_log_property_l2699_269994

theorem log_property (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.log x) (h2 : f (a * b) = 1) :
  f (a ^ 2) + f (b ^ 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_property_l2699_269994


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2699_269914

theorem rectangular_prism_diagonal 
  (a b c : ℝ) 
  (h1 : 2 * a * b + 2 * b * c + 2 * c * a = 11) 
  (h2 : 4 * (a + b + c) = 24) : 
  Real.sqrt (a^2 + b^2 + c^2) = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2699_269914


namespace NUMINAMATH_CALUDE_expression_equivalence_l2699_269985

theorem expression_equivalence : 
  (4+5)*(4^2+5^2)*(4^4+5^4)*(4^8+5^8)*(4^16+5^16)*(4^32+5^32)*(4^64+5^64) = 5^128 - 4^128 := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l2699_269985


namespace NUMINAMATH_CALUDE_class_size_l2699_269967

/-- Represents the number of students in the class -/
def n : ℕ := sorry

/-- Represents the number of leftover erasers -/
def x : ℕ := sorry

/-- The number of gel pens bought -/
def gel_pens : ℕ := 2 * n + 2 * x

/-- The number of ballpoint pens bought -/
def ballpoint_pens : ℕ := 3 * n + 48

/-- The number of erasers bought -/
def erasers : ℕ := 4 * n + x

theorem class_size : 
  gel_pens = ballpoint_pens ∧ 
  ballpoint_pens = erasers ∧ 
  x = 2 * n → 
  n = 16 := by sorry

end NUMINAMATH_CALUDE_class_size_l2699_269967


namespace NUMINAMATH_CALUDE_green_hats_count_l2699_269904

theorem green_hats_count (total_hats : ℕ) (blue_cost green_cost total_cost : ℚ) :
  total_hats = 85 →
  blue_cost = 6 →
  green_cost = 7 →
  total_cost = 540 →
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_cost * blue_hats + green_cost * green_hats = total_cost ∧
    green_hats = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_green_hats_count_l2699_269904


namespace NUMINAMATH_CALUDE_jungkook_english_score_l2699_269901

/-- Jungkook's average score in Korean, math, and science -/
def initial_average : ℝ := 92

/-- The increase in average score after taking the English test -/
def average_increase : ℝ := 2

/-- The number of subjects before taking the English test -/
def initial_subjects : ℕ := 3

/-- The number of subjects after taking the English test -/
def total_subjects : ℕ := 4

/-- Jungkook's English score -/
def english_score : ℝ := total_subjects * (initial_average + average_increase) - initial_subjects * initial_average

theorem jungkook_english_score : english_score = 100 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_english_score_l2699_269901


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l2699_269933

theorem quadratic_equation_problem (k : ℝ) (α β : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + 3 - k = 0 ∧ y^2 + 2*y + 3 - k = 0) →
  (k > 2 ∧ 
   (k^2 = α*β + 3*k ∧ α^2 + 2*α + 3 - k = 0 ∧ β^2 + 2*β + 3 - k = 0) → k = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l2699_269933


namespace NUMINAMATH_CALUDE_cloth_meters_sold_l2699_269947

/-- Proves that the number of meters of cloth sold is 66, given the selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_meters_sold
  (selling_price : ℕ)
  (profit_per_meter : ℕ)
  (cost_per_meter : ℕ)
  (h1 : selling_price = 660)
  (h2 : profit_per_meter = 5)
  (h3 : cost_per_meter = 5) :
  selling_price / (profit_per_meter + cost_per_meter) = 66 := by
  sorry

#check cloth_meters_sold

end NUMINAMATH_CALUDE_cloth_meters_sold_l2699_269947


namespace NUMINAMATH_CALUDE_screen_area_difference_l2699_269916

/-- The area difference between two square screens given their diagonal lengths -/
theorem screen_area_difference (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 18) :
  d1^2 - d2^2 = 76 := by
  sorry

#check screen_area_difference

end NUMINAMATH_CALUDE_screen_area_difference_l2699_269916


namespace NUMINAMATH_CALUDE_incorrect_parentheses_removal_l2699_269936

theorem incorrect_parentheses_removal (a b c : ℝ) : c - 2*(a + b) ≠ c - 2*a + 2*b := by
  sorry

end NUMINAMATH_CALUDE_incorrect_parentheses_removal_l2699_269936


namespace NUMINAMATH_CALUDE_limit_of_sequence_l2699_269945

/-- The sequence a_n defined as (1 + 3n) / (6 - n) converges to -3 as n approaches infinity. -/
theorem limit_of_sequence (ε : ℝ) (hε : ε > 0) : 
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → |((1 : ℝ) + 3 * n) / (6 - n) + 3| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_sequence_l2699_269945


namespace NUMINAMATH_CALUDE_negation_of_parallelogram_is_rhombus_is_true_l2699_269984

-- Define the property of being a parallelogram
def is_parallelogram (shape : Type) : Prop := sorry

-- Define the property of being a rhombus
def is_rhombus (shape : Type) : Prop := sorry

-- The statement we want to prove
theorem negation_of_parallelogram_is_rhombus_is_true :
  ∃ (shape : Type), is_parallelogram shape ∧ ¬is_rhombus shape := by sorry

end NUMINAMATH_CALUDE_negation_of_parallelogram_is_rhombus_is_true_l2699_269984


namespace NUMINAMATH_CALUDE_sin_135_degrees_l2699_269951

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l2699_269951


namespace NUMINAMATH_CALUDE_system_solvability_l2699_269946

-- Define the system of equations
def system (x y a b : ℝ) : Prop :=
  x * Real.cos a + y * Real.sin a + 4 ≤ 0 ∧
  x^2 + y^2 + 10*x + 2*y - b^2 - 8*b + 10 = 0

-- Define the set of valid b values
def valid_b_set (b : ℝ) : Prop :=
  b ≤ -8 - Real.sqrt 26 ∨ b ≥ Real.sqrt 26

-- Theorem statement
theorem system_solvability (b : ℝ) :
  (∀ a, ∃ x y, system x y a b) ↔ valid_b_set b :=
sorry

end NUMINAMATH_CALUDE_system_solvability_l2699_269946


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l2699_269915

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define the transformed function
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (2*x + 1)

-- Theorem statement
theorem axis_of_symmetry (f : ℝ → ℝ) (h : even_function f) :
  ∀ x : ℝ, g f ((-1) + x) = g f ((-1) - x) :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l2699_269915


namespace NUMINAMATH_CALUDE_sample_is_sixteen_l2699_269999

/-- Represents a stratified sampling scenario in a factory -/
structure StratifiedSampling where
  totalSample : ℕ
  totalProducts : ℕ
  workshopProducts : ℕ
  h_positive : 0 < totalSample ∧ 0 < totalProducts ∧ 0 < workshopProducts
  h_valid : workshopProducts ≤ totalProducts

/-- Calculates the number of items sampled from a specific workshop -/
def sampleFromWorkshop (s : StratifiedSampling) : ℕ :=
  (s.totalSample * s.workshopProducts) / s.totalProducts

/-- Theorem stating that for the given scenario, the sample from the workshop is 16 -/
theorem sample_is_sixteen (s : StratifiedSampling) 
  (h_total_sample : s.totalSample = 128)
  (h_total_products : s.totalProducts = 2048)
  (h_workshop_products : s.workshopProducts = 256) : 
  sampleFromWorkshop s = 16 := by
  sorry

#eval sampleFromWorkshop { 
  totalSample := 128, 
  totalProducts := 2048, 
  workshopProducts := 256, 
  h_positive := by norm_num, 
  h_valid := by norm_num 
}

end NUMINAMATH_CALUDE_sample_is_sixteen_l2699_269999


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_l2699_269927

theorem quadratic_inequality_solutions (b : ℤ) : 
  (∃ (x₁ x₂ x₃ x₄ : ℤ), 
    (∀ x : ℤ, x^2 + b*x + 1 ≤ 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) ∧
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄)) →
  b = 4 ∨ b = -4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_l2699_269927


namespace NUMINAMATH_CALUDE_decreasing_interval_of_f_l2699_269906

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f(x)
def f_deriv (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem statement
theorem decreasing_interval_of_f :
  ∀ x : ℝ, (x ∈ Set.Ioo (-1) 1) ↔ (f_deriv x < 0) :=
sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_f_l2699_269906


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l2699_269912

/-- Represents different sampling methods -/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Represents a community with different income groups -/
structure Community where
  total_households : ℕ
  high_income : ℕ
  middle_income : ℕ
  low_income : ℕ
  sample_size : ℕ

/-- Represents a group of volleyball players -/
structure VolleyballTeam where
  total_players : ℕ
  players_to_select : ℕ

/-- Determines the most appropriate sampling method for a community survey -/
def best_community_sampling_method (c : Community) : SamplingMethod :=
  sorry

/-- Determines the most appropriate sampling method for a volleyball team survey -/
def best_volleyball_sampling_method (v : VolleyballTeam) : SamplingMethod :=
  sorry

/-- Theorem stating the most appropriate sampling methods for the given scenarios -/
theorem appropriate_sampling_methods 
  (community : Community)
  (volleyball_team : VolleyballTeam)
  (h_community : community = { 
    total_households := 400,
    high_income := 120,
    middle_income := 180,
    low_income := 100,
    sample_size := 100
  })
  (h_volleyball : volleyball_team = {
    total_players := 12,
    players_to_select := 3
  }) :
  best_community_sampling_method community = SamplingMethod.StratifiedSampling ∧
  best_volleyball_sampling_method volleyball_team = SamplingMethod.SimpleRandomSampling :=
sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l2699_269912


namespace NUMINAMATH_CALUDE_fifteen_non_congruent_triangles_l2699_269949

-- Define the points
variable (A B C M N P : ℝ × ℝ)

-- Define the isosceles triangle
def is_isosceles_triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist A C

-- Define M as midpoint of AB
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define N on AC with 1:2 ratio
def divides_in_ratio_one_two (N A C : ℝ × ℝ) : Prop :=
  dist A N = (1/3) * dist A C

-- Define P on BC with 1:3 ratio
def divides_in_ratio_one_three (P B C : ℝ × ℝ) : Prop :=
  dist B P = (1/4) * dist B C

-- Define a function to count non-congruent triangles
def count_non_congruent_triangles (A B C M N P : ℝ × ℝ) : ℕ := sorry

-- State the theorem
theorem fifteen_non_congruent_triangles
  (h1 : is_isosceles_triangle A B C)
  (h2 : is_midpoint M A B)
  (h3 : divides_in_ratio_one_two N A C)
  (h4 : divides_in_ratio_one_three P B C) :
  count_non_congruent_triangles A B C M N P = 15 := by sorry

end NUMINAMATH_CALUDE_fifteen_non_congruent_triangles_l2699_269949


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_l2699_269929

-- Define what it means for a number to be rational
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define irrationality as the negation of rationality
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- State the theorem
theorem sqrt_two_irrational : IsIrrational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_l2699_269929


namespace NUMINAMATH_CALUDE_initial_bedbug_count_l2699_269902

/-- The number of bedbugs after n days, given an initial population -/
def bedbug_population (initial : ℕ) (days : ℕ) : ℕ :=
  initial * (3 ^ days)

/-- Theorem: If the number of bedbugs triples every day and there are 810 bedbugs after four days, 
    then the initial number of bedbugs was 30. -/
theorem initial_bedbug_count : bedbug_population 30 4 = 810 := by
  sorry

#check initial_bedbug_count

end NUMINAMATH_CALUDE_initial_bedbug_count_l2699_269902


namespace NUMINAMATH_CALUDE_supermarket_promotion_cost_l2699_269987

/-- Represents the cost calculation for a supermarket promotion --/
def supermarket_promotion (x : ℕ) : Prop :=
  let teapot_price : ℕ := 20
  let teacup_price : ℕ := 6
  let num_teapots : ℕ := 5
  x > 5 →
  (num_teapots * teapot_price + (x - num_teapots) * teacup_price = 6 * x + 70) ∧
  ((num_teapots * teapot_price + x * teacup_price) * 9 / 10 = 54 * x / 10 + 90)

theorem supermarket_promotion_cost (x : ℕ) : supermarket_promotion x :=
by sorry

end NUMINAMATH_CALUDE_supermarket_promotion_cost_l2699_269987


namespace NUMINAMATH_CALUDE_tulip_lilac_cost_comparison_l2699_269972

/-- Given that 4 tulips and 5 lilacs cost less than 22 yuan, and 6 tulips and 3 lilacs cost more than 24 yuan, prove that 2 tulips cost more than 3 lilacs. -/
theorem tulip_lilac_cost_comparison (x y : ℝ) 
  (h1 : 4 * x + 5 * y < 22) 
  (h2 : 6 * x + 3 * y > 24) : 
  2 * x > 3 * y := by
  sorry

end NUMINAMATH_CALUDE_tulip_lilac_cost_comparison_l2699_269972


namespace NUMINAMATH_CALUDE_a_range_l2699_269979

-- Define the line equation
def line_equation (a x y : ℝ) : ℝ := a * x + 2 * y - 1

-- Define the points A and B
def point_A : ℝ × ℝ := (3, -1)
def point_B : ℝ × ℝ := (-1, 2)

-- Define the condition for points being on the same side of the line
def same_side (a : ℝ) : Prop :=
  (line_equation a point_A.1 point_A.2) * (line_equation a point_B.1 point_B.2) > 0

-- Theorem stating the range of a
theorem a_range : 
  ∀ a : ℝ, same_side a ↔ a ∈ Set.Ioo 1 3 :=
sorry

end NUMINAMATH_CALUDE_a_range_l2699_269979


namespace NUMINAMATH_CALUDE_sixty_seven_in_one_row_l2699_269953

def pascal_coefficient (n k : ℕ) : ℕ := Nat.choose n k

theorem sixty_seven_in_one_row :
  ∃! row : ℕ, ∃ k : ℕ, pascal_coefficient row k = 67 :=
by
  sorry

end NUMINAMATH_CALUDE_sixty_seven_in_one_row_l2699_269953


namespace NUMINAMATH_CALUDE_dodecahedron_edge_probability_l2699_269973

/-- A regular dodecahedron with 20 vertices -/
structure Dodecahedron :=
  (vertices : Finset Nat)
  (h_card : vertices.card = 20)

/-- The probability of two randomly chosen vertices being endpoints of an edge -/
def edge_probability (d : Dodecahedron) : ℚ :=
  3 / 19

theorem dodecahedron_edge_probability (d : Dodecahedron) :
  edge_probability d = 3 / 19 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_edge_probability_l2699_269973


namespace NUMINAMATH_CALUDE_tom_filled_33_balloons_l2699_269961

/-- The number of water balloons filled up by Anthony -/
def anthony_balloons : ℕ := 44

/-- The number of water balloons filled up by Luke -/
def luke_balloons : ℕ := anthony_balloons / 4

/-- The number of water balloons filled up by Tom -/
def tom_balloons : ℕ := 3 * luke_balloons

/-- Theorem stating that Tom filled up 33 water balloons -/
theorem tom_filled_33_balloons : tom_balloons = 33 := by
  sorry

end NUMINAMATH_CALUDE_tom_filled_33_balloons_l2699_269961


namespace NUMINAMATH_CALUDE_simplify_fraction_l2699_269995

theorem simplify_fraction : 5 * (21 / 6) * (18 / -63) = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2699_269995


namespace NUMINAMATH_CALUDE_cosine_symmetry_center_l2699_269944

/-- Given a cosine function y = 2cos(2x) translated π/12 units to the right,
    prove that (5π/6, 0) is one of its symmetry centers. -/
theorem cosine_symmetry_center :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.cos (2 * (x - π/12))
  ∃ (k : ℤ), (5*π/6 : ℝ) = k*π/2 + π/3 ∧ 
    (∀ x : ℝ, f (5*π/6 + x) = f (5*π/6 - x)) :=
by sorry

end NUMINAMATH_CALUDE_cosine_symmetry_center_l2699_269944


namespace NUMINAMATH_CALUDE_possible_values_of_P_zero_l2699_269991

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The property that the polynomial P must satisfy -/
def SatisfiesProperty (P : RealPolynomial) : Prop :=
  ∀ x y : ℝ, |y^2 - P x| ≤ 2 * |x| ↔ |x^2 - P y| ≤ 2 * |y|

/-- The theorem stating the possible values of P(0) -/
theorem possible_values_of_P_zero (P : RealPolynomial) 
  (h : SatisfiesProperty P) : 
  P 0 < 0 ∨ P 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_P_zero_l2699_269991


namespace NUMINAMATH_CALUDE_blasting_safety_condition_l2699_269962

/-- Represents the parameters of a blasting operation safety scenario -/
structure BlastingSafety where
  safetyDistance : ℝ
  fuseSpeed : ℝ
  blasterSpeed : ℝ

/-- Defines the safety condition for a blasting operation -/
def isSafe (params : BlastingSafety) (fuseLength : ℝ) : Prop :=
  fuseLength / params.fuseSpeed > (params.safetyDistance - fuseLength) / params.blasterSpeed

/-- Theorem stating the safety condition for a specific blasting scenario -/
theorem blasting_safety_condition :
  let params : BlastingSafety := {
    safetyDistance := 50,
    fuseSpeed := 0.2,
    blasterSpeed := 3
  }
  ∀ x : ℝ, isSafe params x ↔ x / 0.2 > (50 - x) / 3 := by
  sorry


end NUMINAMATH_CALUDE_blasting_safety_condition_l2699_269962


namespace NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l2699_269978

theorem richmond_tigers_ticket_sales (total_tickets : ℕ) (first_half_tickets : ℕ) 
  (h1 : total_tickets = 9570) (h2 : first_half_tickets = 3867) :
  total_tickets - first_half_tickets = 5703 := by
  sorry

end NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l2699_269978


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2699_269934

theorem max_value_of_expression (a b c : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c)
  (sum_squares : a^2 + b^2 + c^2 = 1) :
  2 * a * b + 2 * b * c * Real.sqrt 2 ≤ Real.sqrt (3/2) ∧ 
  ∃ a₀ b₀ c₀ : ℝ, 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ 
    a₀^2 + b₀^2 + c₀^2 = 1 ∧
    2 * a₀ * b₀ + 2 * b₀ * c₀ * Real.sqrt 2 = Real.sqrt (3/2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2699_269934


namespace NUMINAMATH_CALUDE_first_fish_length_is_0_3_l2699_269924

/-- The length of the second fish in feet -/
def second_fish_length : ℝ := 0.2

/-- The difference in length between the first and second fish in feet -/
def length_difference : ℝ := 0.1

/-- The length of the first fish in feet -/
def first_fish_length : ℝ := second_fish_length + length_difference

theorem first_fish_length_is_0_3 : first_fish_length = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_first_fish_length_is_0_3_l2699_269924


namespace NUMINAMATH_CALUDE_spicy_hot_noodles_count_l2699_269989

theorem spicy_hot_noodles_count (total_plates lobster_rolls seafood_noodles : ℕ) 
  (h1 : total_plates = 55)
  (h2 : lobster_rolls = 25)
  (h3 : seafood_noodles = 16) :
  total_plates - (lobster_rolls + seafood_noodles) = 14 := by
  sorry

end NUMINAMATH_CALUDE_spicy_hot_noodles_count_l2699_269989


namespace NUMINAMATH_CALUDE_boys_in_art_class_l2699_269998

theorem boys_in_art_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) 
  (h1 : total = 35) 
  (h2 : ratio_girls = 4) 
  (h3 : ratio_boys = 3) : 
  (ratio_boys * total) / (ratio_girls + ratio_boys) = 15 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_art_class_l2699_269998


namespace NUMINAMATH_CALUDE_roots_not_analytically_determinable_l2699_269908

/-- The polynomial equation whose roots we want to determine -/
def f (x : ℝ) : ℝ := (x - 2) * (x + 5)^3 * (5 - x) - 8

/-- Theorem stating that the roots of the polynomial equation cannot be determined analytically -/
theorem roots_not_analytically_determinable :
  ¬ ∃ (roots : Set ℝ), ∀ (x : ℝ), x ∈ roots ↔ f x = 0 ∧ 
  ∃ (formula : ℝ → ℝ), ∀ (x : ℝ), x ∈ roots → ∃ (n : ℕ), formula x = x ∧ 
  (∀ (y : ℝ), formula y = y → y ∈ roots) :=
sorry

end NUMINAMATH_CALUDE_roots_not_analytically_determinable_l2699_269908


namespace NUMINAMATH_CALUDE_tim_cabinet_price_l2699_269938

/-- The amount Tim paid for a cabinet with a discount -/
def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

/-- Proof that Tim paid $1020 for the cabinet -/
theorem tim_cabinet_price :
  let original_price : ℝ := 1200
  let discount_rate : ℝ := 0.15
  discounted_price original_price discount_rate = 1020 := by
sorry

end NUMINAMATH_CALUDE_tim_cabinet_price_l2699_269938


namespace NUMINAMATH_CALUDE_min_value_theorem_l2699_269907

theorem min_value_theorem (x y : ℝ) (hx : x > 1) (hy : y > 1) (h_sum : x + 2*y = 5) :
  1/(x-1) + 1/(y-1) ≥ 3/2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2699_269907


namespace NUMINAMATH_CALUDE_school_teachers_l2699_269983

/-- Calculates the number of teachers in a school given specific conditions -/
theorem school_teachers (students : ℕ) (classes_per_student : ℕ) (classes_per_teacher : ℕ) (students_per_class : ℕ) :
  students = 2400 →
  classes_per_student = 5 →
  classes_per_teacher = 4 →
  students_per_class = 30 →
  (students * classes_per_student) / (classes_per_teacher * students_per_class) = 100 := by
  sorry

end NUMINAMATH_CALUDE_school_teachers_l2699_269983


namespace NUMINAMATH_CALUDE_cubic_three_roots_m_range_l2699_269939

/-- Given a cubic function f(x) = x³ - 6x² + 9x + m, if there exist three distinct
    real roots, then the parameter m must be in the open interval (-4, 0). -/
theorem cubic_three_roots_m_range (m : ℝ) :
  (∃ a b c : ℝ, a < b ∧ b < c ∧
    (a^3 - 6*a^2 + 9*a + m = 0) ∧
    (b^3 - 6*b^2 + 9*b + m = 0) ∧
    (c^3 - 6*c^2 + 9*c + m = 0)) →
  -4 < m ∧ m < 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_three_roots_m_range_l2699_269939


namespace NUMINAMATH_CALUDE_sunflower_majority_on_friday_friday_first_sunflower_majority_l2699_269935

/-- Represents the amount of sunflower seeds in the feeder on a given day -/
def sunflower_seeds (day : ℕ) : ℝ :=
  0.9 + (0.63 ^ (day - 1)) * 0.9

/-- Represents the total amount of seeds in the feeder on a given day -/
def total_seeds (day : ℕ) : ℝ :=
  3 * day

/-- The theorem states that on the 5th day, sunflower seeds exceed half of the total seeds -/
theorem sunflower_majority_on_friday :
  sunflower_seeds 5 > (total_seeds 5) / 2 := by
  sorry

/-- Helper function to check if sunflower seeds exceed half of total seeds on a given day -/
def is_sunflower_majority (day : ℕ) : Prop :=
  sunflower_seeds day > (total_seeds day) / 2

/-- The theorem states that Friday (day 5) is the first day when sunflower seeds exceed half of the total seeds -/
theorem friday_first_sunflower_majority :
  is_sunflower_majority 5 ∧ 
  (∀ d : ℕ, d < 5 → ¬is_sunflower_majority d) := by
  sorry

end NUMINAMATH_CALUDE_sunflower_majority_on_friday_friday_first_sunflower_majority_l2699_269935


namespace NUMINAMATH_CALUDE_parallel_lines_sum_l2699_269982

/-- Two parallel lines with a given distance between them -/
structure ParallelLines where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  distance : ℝ
  parallel : a / b = c / d
  dist_formula : distance = |c - a| / Real.sqrt (c^2 + d^2)

/-- The theorem to be proved -/
theorem parallel_lines_sum (lines : ParallelLines) 
  (h1 : lines.a = 3 ∧ lines.b = 4)
  (h2 : lines.c = 6)
  (h3 : lines.distance = 3) :
  (lines.d + lines.c = -12) ∨ (lines.d + lines.c = 48) := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_sum_l2699_269982


namespace NUMINAMATH_CALUDE_complex_fraction_equals_minus_one_plus_i_l2699_269976

theorem complex_fraction_equals_minus_one_plus_i :
  (2 * Complex.I) / (1 - Complex.I) = -1 + Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_minus_one_plus_i_l2699_269976


namespace NUMINAMATH_CALUDE_total_cartons_packed_l2699_269956

/-- Proves the total number of cartons packed given the conditions -/
theorem total_cartons_packed 
  (cans_per_carton : ℕ) 
  (loaded_cartons : ℕ) 
  (remaining_cans : ℕ) : 
  cans_per_carton = 20 → 
  loaded_cartons = 40 → 
  remaining_cans = 200 → 
  loaded_cartons + (remaining_cans / cans_per_carton) = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_cartons_packed_l2699_269956


namespace NUMINAMATH_CALUDE_integer_square_four_l2699_269913

theorem integer_square_four (x : ℝ) (y : ℤ) 
  (eq1 : 4 * x + y = 34)
  (eq2 : 2 * x - y = 20) : 
  y = -2 ∧ y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_integer_square_four_l2699_269913


namespace NUMINAMATH_CALUDE_treaty_signing_day_l2699_269955

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def advance_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advance_days (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => advance_day (advance_days d m)

theorem treaty_signing_day :
  advance_days DayOfWeek.Wednesday 1566 = DayOfWeek.Wednesday :=
by sorry


end NUMINAMATH_CALUDE_treaty_signing_day_l2699_269955


namespace NUMINAMATH_CALUDE_similar_triangle_shortest_side_l2699_269996

theorem similar_triangle_shortest_side 
  (side1 : ℝ) 
  (hyp1 : ℝ) 
  (hyp2 : ℝ) 
  (is_right_triangle : side1^2 + (hyp1^2 - side1^2) = hyp1^2)
  (hyp1_positive : hyp1 > 0)
  (hyp2_positive : hyp2 > 0)
  (h_similar : hyp2 / hyp1 * side1 = 72) :
  ∃ (side2 : ℝ), side2 = 72 ∧ side2 ≤ hyp2 := by sorry

end NUMINAMATH_CALUDE_similar_triangle_shortest_side_l2699_269996


namespace NUMINAMATH_CALUDE_santino_papaya_trees_l2699_269958

/-- The number of papaya trees Santino has -/
def num_papaya_trees : ℕ := sorry

/-- The number of mango trees Santino has -/
def num_mango_trees : ℕ := 3

/-- The number of papayas produced by each papaya tree -/
def papayas_per_tree : ℕ := 10

/-- The number of mangos produced by each mango tree -/
def mangos_per_tree : ℕ := 20

/-- The total number of fruits Santino has -/
def total_fruits : ℕ := 80

/-- Theorem stating that Santino has 2 papaya trees -/
theorem santino_papaya_trees :
  num_papaya_trees * papayas_per_tree + num_mango_trees * mangos_per_tree = total_fruits ∧
  num_papaya_trees = 2 := by sorry

end NUMINAMATH_CALUDE_santino_papaya_trees_l2699_269958


namespace NUMINAMATH_CALUDE_count_D_two_eq_30_l2699_269905

/-- The number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- The count of positive integers n ≤ 127 for which D(n) = 2 -/
def count_D_two : ℕ := sorry

theorem count_D_two_eq_30 : count_D_two = 30 := by sorry

end NUMINAMATH_CALUDE_count_D_two_eq_30_l2699_269905


namespace NUMINAMATH_CALUDE_merchant_profit_comparison_l2699_269925

/-- Represents the profit calculation for two merchants selling goods --/
theorem merchant_profit_comparison
  (x : ℝ) -- cost price of goods for each merchant
  (h_pos : x > 0) -- assumption that cost price is positive
  : x < 1.08 * x := by
  sorry

#check merchant_profit_comparison

end NUMINAMATH_CALUDE_merchant_profit_comparison_l2699_269925


namespace NUMINAMATH_CALUDE_gcd_459_357_l2699_269921

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l2699_269921


namespace NUMINAMATH_CALUDE_function_properties_l2699_269960

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2*a - x) = f x

theorem function_properties (f : ℝ → ℝ) :
  (∀ x, f (x + 2) = f (x - 2)) →
  (∀ x, f (4 - x) = f x) →
  (is_periodic f 4 ∧ is_symmetric_about f 2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2699_269960


namespace NUMINAMATH_CALUDE_bathroom_area_is_eight_l2699_269970

/-- The area of a rectangular bathroom -/
def bathroom_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of a bathroom with length 4 feet and width 2 feet is 8 square feet -/
theorem bathroom_area_is_eight :
  bathroom_area 4 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_area_is_eight_l2699_269970


namespace NUMINAMATH_CALUDE_museum_paintings_ratio_l2699_269900

theorem museum_paintings_ratio (total_paintings portraits : ℕ) 
  (h1 : total_paintings = 80)
  (h2 : portraits = 16)
  (h3 : ∃ k : ℕ, total_paintings - portraits = k * portraits) :
  (total_paintings - portraits) / portraits = 4 := by
  sorry

end NUMINAMATH_CALUDE_museum_paintings_ratio_l2699_269900


namespace NUMINAMATH_CALUDE_equation_equivalence_l2699_269988

-- Define the original equation
def original_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + y^2) + Real.sqrt ((x + 4)^2 + y^2) = 10

-- Define the simplified ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

-- Theorem stating the equivalence of the two equations
theorem equation_equivalence :
  ∀ x y : ℝ, original_equation x y ↔ ellipse_equation x y :=
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2699_269988


namespace NUMINAMATH_CALUDE_inequality_proof_l2699_269981

theorem inequality_proof (x y z : ℝ) : (x^2 - y^2)^2 + (z - x)^2 + (x - 1)^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2699_269981


namespace NUMINAMATH_CALUDE_january_salary_solve_salary_problem_l2699_269966

/-- Represents the monthly salary structure -/
structure MonthlySalary where
  january : ℕ
  february : ℕ
  march : ℕ
  april : ℕ
  may : ℕ

/-- Theorem stating the salary for January given the conditions -/
theorem january_salary (s : MonthlySalary) :
  (s.january + s.february + s.march + s.april) / 4 = 8000 →
  (s.february + s.march + s.april + s.may) / 4 = 8400 →
  s.may = 6500 →
  s.january = 4900 := by
  sorry

/-- Main theorem proving the salary calculation -/
theorem solve_salary_problem :
  ∃ s : MonthlySalary,
    (s.january + s.february + s.march + s.april) / 4 = 8000 ∧
    (s.february + s.march + s.april + s.may) / 4 = 8400 ∧
    s.may = 6500 ∧
    s.january = 4900 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_solve_salary_problem_l2699_269966


namespace NUMINAMATH_CALUDE_katya_problems_l2699_269930

theorem katya_problems (p_katya : ℝ) (p_pen : ℝ) (total_problems : ℕ) (good_grade : ℝ) 
  (h_katya : p_katya = 4/5)
  (h_pen : p_pen = 1/2)
  (h_total : total_problems = 20)
  (h_good : good_grade = 13) :
  ∃ x : ℝ, x ≥ 10 ∧ 
    x * p_katya + (total_problems - x) * p_pen ≥ good_grade ∧
    ∀ y : ℝ, y < 10 → y * p_katya + (total_problems - y) * p_pen < good_grade := by
  sorry

end NUMINAMATH_CALUDE_katya_problems_l2699_269930


namespace NUMINAMATH_CALUDE_no_real_roots_x_squared_plus_three_l2699_269974

theorem no_real_roots_x_squared_plus_three : 
  ∀ x : ℝ, x^2 + 3 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_x_squared_plus_three_l2699_269974


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2699_269954

theorem three_numbers_sum (x y z : ℝ) : 
  x ≤ y ∧ y ≤ z →  -- x is the least, z is the greatest
  y = 9 →  -- median is 9
  (x + y + z) / 3 = x + 20 →  -- mean is 20 more than least
  (x + y + z) / 3 = z - 18 →  -- mean is 18 less than greatest
  x + y + z = 21 := by sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2699_269954


namespace NUMINAMATH_CALUDE_woman_work_time_l2699_269931

/-- Represents the time taken to complete a work unit -/
structure WorkTime where
  men : ℕ
  women : ℕ
  days : ℚ

/-- The work rate of a single person -/
def work_rate (wt : WorkTime) : ℚ := 1 / wt.days

theorem woman_work_time (wt1 wt2 : WorkTime) : 
  wt1.men = 10 ∧ 
  wt1.women = 15 ∧ 
  wt1.days = 6 ∧
  wt2.men = 1 ∧ 
  wt2.women = 0 ∧ 
  wt2.days = 100 →
  ∃ wt3 : WorkTime, wt3.men = 0 ∧ wt3.women = 1 ∧ wt3.days = 225 :=
by sorry

#check woman_work_time

end NUMINAMATH_CALUDE_woman_work_time_l2699_269931


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l2699_269948

theorem greatest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l2699_269948


namespace NUMINAMATH_CALUDE_basketball_only_count_l2699_269909

theorem basketball_only_count (total students_basketball students_table_tennis students_neither : ℕ) :
  total = 30 ∧
  students_basketball = 15 ∧
  students_table_tennis = 10 ∧
  students_neither = 8 →
  ∃ (students_both : ℕ),
    students_basketball - students_both = 12 ∧
    students_both + (students_basketball - students_both) + (students_table_tennis - students_both) + students_neither = total :=
by sorry

end NUMINAMATH_CALUDE_basketball_only_count_l2699_269909


namespace NUMINAMATH_CALUDE_total_packs_eq_sum_l2699_269971

/-- The number of glue stick packs Emily's mom bought -/
def total_packs : ℕ := sorry

/-- The number of glue stick packs Emily received -/
def emily_packs : ℕ := 6

/-- The number of glue stick packs Emily's sister received -/
def sister_packs : ℕ := 7

/-- Theorem: The total number of glue stick packs is the sum of packs given to Emily and her sister -/
theorem total_packs_eq_sum : total_packs = emily_packs + sister_packs := by sorry

end NUMINAMATH_CALUDE_total_packs_eq_sum_l2699_269971


namespace NUMINAMATH_CALUDE_rook_paths_eq_catalan_l2699_269926

/-- The number of valid paths for a rook on an n × n chessboard -/
def rookPaths (n : ℕ) : ℕ :=
  if n = 0 then 1
  else (Nat.choose (2 * n - 2) (n - 1)) / n

/-- The Catalan number C_n -/
def catalanNumber (n : ℕ) : ℕ :=
  if n = 0 then 1
  else (Nat.choose (2 * n) n) / (n + 1)

/-- Theorem: The number of valid rook paths on an n × n chessboard
    is equal to the (n-1)th Catalan number -/
theorem rook_paths_eq_catalan (n : ℕ) :
  rookPaths n = catalanNumber (n - 1) :=
sorry

end NUMINAMATH_CALUDE_rook_paths_eq_catalan_l2699_269926


namespace NUMINAMATH_CALUDE_unique_positive_number_l2699_269980

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 8 = 128 / x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l2699_269980


namespace NUMINAMATH_CALUDE_parentheses_equivalence_l2699_269919

theorem parentheses_equivalence (a b c : ℝ) : a - b + c = a - (b - c) := by
  sorry

end NUMINAMATH_CALUDE_parentheses_equivalence_l2699_269919


namespace NUMINAMATH_CALUDE_max_internet_days_l2699_269993

/-- Represents the tiered pricing structure for internet service -/
def daily_rate (day : ℕ) : ℚ :=
  if day ≤ 3 then 1/2
  else if day ≤ 7 then 7/10
  else 9/10

/-- Calculates the additional fee for every 5 days -/
def additional_fee (day : ℕ) : ℚ :=
  if day % 5 = 0 then 1 else 0

/-- Calculates the total cost for a given number of days -/
def total_cost (days : ℕ) : ℚ :=
  (Finset.range days).sum (λ d => daily_rate (d + 1) + additional_fee (d + 1))

/-- Theorem stating that 8 is the maximum number of days of internet connection -/
theorem max_internet_days : 
  ∀ n : ℕ, n ≤ 8 → total_cost n ≤ 7 ∧ 
  (n < 8 → total_cost (n + 1) ≤ 7) ∧
  (total_cost 9 > 7) :=
sorry

end NUMINAMATH_CALUDE_max_internet_days_l2699_269993


namespace NUMINAMATH_CALUDE_sunday_sales_proof_l2699_269950

def price_per_caricature : ℕ := 20
def saturday_sales : ℕ := 24
def total_revenue : ℕ := 800

theorem sunday_sales_proof : 
  ∃ (sunday_sales : ℕ), 
    price_per_caricature * (saturday_sales + sunday_sales) = total_revenue ∧ 
    sunday_sales = 16 := by
  sorry

end NUMINAMATH_CALUDE_sunday_sales_proof_l2699_269950


namespace NUMINAMATH_CALUDE_smallest_n_with_seven_and_terminating_l2699_269910

/-- A function that checks if a number contains the digit 7 -/
def contains_seven (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < n ∧ n % 10^(d+1) / 10^d = 7

/-- A function that checks if a fraction 1/n is a terminating decimal -/
def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2^a * 5^b

/-- Theorem stating that 128 is the smallest positive integer n such that
    1/n is a terminating decimal and n contains the digit 7 -/
theorem smallest_n_with_seven_and_terminating :
  ∀ n : ℕ, n > 0 → is_terminating_decimal n → contains_seven n → n ≥ 128 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_seven_and_terminating_l2699_269910


namespace NUMINAMATH_CALUDE_equation_solution_l2699_269964

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem equation_solution (a : ℕ+) :
  (∃ n : ℕ+, 7 * a * n - 3 * factorial n = 2020) ↔ (a = 68 ∨ a = 289) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l2699_269964


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2699_269986

-- Define the universal set U
def U : Set Int := {x | -3 ≤ x ∧ x ≤ 3}

-- Define set A
def A : Set Int := {0, 1, 2, 3}

-- State the theorem
theorem complement_of_A_in_U : 
  (U \ A) = {-3, -2, -1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2699_269986


namespace NUMINAMATH_CALUDE_reduced_price_is_80_l2699_269959

/-- Represents the price reduction percentage -/
def price_reduction : ℚ := 1/2

/-- Represents the additional amount of oil obtained after price reduction -/
def additional_oil : ℕ := 5

/-- Represents the fixed amount of money spent -/
def fixed_cost : ℕ := 800

/-- Theorem stating that given the conditions, the reduced price per kg is 80 -/
theorem reduced_price_is_80 :
  ∀ (original_price : ℚ) (original_amount : ℚ),
  original_amount * original_price = fixed_cost →
  (original_amount + additional_oil) * (original_price * (1 - price_reduction)) = fixed_cost →
  original_price * (1 - price_reduction) = 80 :=
by sorry

end NUMINAMATH_CALUDE_reduced_price_is_80_l2699_269959


namespace NUMINAMATH_CALUDE_coursework_materials_expense_l2699_269920

def budget : ℝ := 1000
def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

theorem coursework_materials_expense : 
  budget * (1 - (food_percentage + accommodation_percentage + entertainment_percentage)) = 300 := by
  sorry

end NUMINAMATH_CALUDE_coursework_materials_expense_l2699_269920
