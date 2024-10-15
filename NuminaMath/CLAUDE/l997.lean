import Mathlib

namespace NUMINAMATH_CALUDE_stating_probability_of_target_sequence_l997_99777

/-- The number of balls in the box -/
def total_balls : ℕ := 500

/-- The number of balls selected -/
def selections : ℕ := 5

/-- The probability of selecting an odd-numbered ball -/
def prob_odd : ℚ := 1 / 2

/-- The probability of selecting an even-numbered ball -/
def prob_even : ℚ := 1 / 2

/-- The sequence of selections we're interested in (odd, even, odd, even, odd) -/
def target_sequence : List Bool := [true, false, true, false, true]

/-- 
Theorem stating that the probability of selecting the target sequence 
(odd, even, odd, even, odd) from a box of 500 balls numbered 1 to 500, 
with 5 selections and replacement, is 1/32.
-/
theorem probability_of_target_sequence : 
  (List.prod (target_sequence.map (fun b => if b then prob_odd else prob_even))) = (1 : ℚ) / 32 := by
  sorry

end NUMINAMATH_CALUDE_stating_probability_of_target_sequence_l997_99777


namespace NUMINAMATH_CALUDE_exchange_properties_l997_99718

/-- Represents a box containing red and yellow balls -/
structure Box where
  red : ℕ
  yellow : ℕ

/-- Calculates the expected number of red balls after exchanging i balls -/
noncomputable def expected_red (box_a box_b : Box) (i : ℕ) : ℚ :=
  sorry

/-- Box A initially contains 3 red balls and 1 yellow ball -/
def initial_box_a : Box := ⟨3, 1⟩

/-- Box B initially contains 1 red ball and 3 yellow balls -/
def initial_box_b : Box := ⟨1, 3⟩

theorem exchange_properties :
  let E₁ := expected_red initial_box_a initial_box_b
  let E₂ := expected_red initial_box_b initial_box_a
  (E₁ 1 > E₂ 1) ∧
  (E₁ 2 = E₂ 2) ∧
  (E₁ 2 = 2) ∧
  (E₁ 1 + E₂ 1 = 4) ∧
  (E₁ 3 = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_exchange_properties_l997_99718


namespace NUMINAMATH_CALUDE_difference_between_fractions_l997_99724

theorem difference_between_fractions (n : ℝ) : n = 100 → (3/5 * n) - (1/2 * n) = 10 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_fractions_l997_99724


namespace NUMINAMATH_CALUDE_ellipse_property_l997_99769

/-- Given an ellipse with foci at (2, 0) and (8, 0) passing through (5, 3),
    prove that the sum of its semi-major axis length and the y-coordinate of its center is 3√2. -/
theorem ellipse_property (a b h k : ℝ) : 
  a > 0 → b > 0 →
  (∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ↔ 
    (x - 2)^2 + y^2 + (x - 8)^2 + y^2 = ((x - 2)^2 + y^2 + (x - 8)^2 + y^2)) →
  (5 - h)^2 / a^2 + (3 - k)^2 / b^2 = 1 →
  a + k = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_property_l997_99769


namespace NUMINAMATH_CALUDE_two_over_x_is_proper_convert_improper_to_mixed_integer_values_for_integer_result_l997_99794

-- Define proper and improper expressions
def is_proper_expression (num denom : Polynomial ℚ) : Prop :=
  num.degree < denom.degree

def is_improper_expression (num denom : Polynomial ℚ) : Prop :=
  num.degree ≥ denom.degree

-- Statement 1
theorem two_over_x_is_proper :
  is_proper_expression (2 : Polynomial ℚ) (X : Polynomial ℚ) :=
sorry

-- Statement 2
theorem convert_improper_to_mixed :
  (X^2 - 1) / (X + 2) = X - 2 + 3 / (X + 2) :=
sorry

-- Statement 3
theorem integer_values_for_integer_result :
  {x : ℤ | ∃ (y : ℤ), (2*x - 1) / (x + 1) = y} = {0, -2, 2, -4} :=
sorry

end NUMINAMATH_CALUDE_two_over_x_is_proper_convert_improper_to_mixed_integer_values_for_integer_result_l997_99794


namespace NUMINAMATH_CALUDE_abc_sum_l997_99716

theorem abc_sum (a b c : ℚ) : 
  (a : ℚ) / 3 = (b : ℚ) / 5 ∧ (b : ℚ) / 5 = (c : ℚ) / 7 ∧ 
  3 * a + 2 * b - 4 * c = -9 → 
  a + b - c = 1 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_l997_99716


namespace NUMINAMATH_CALUDE_sum_of_roots_sum_of_roots_is_twenty_l997_99745

/-- Square with sides parallel to coordinate axes -/
structure Square :=
  (side_length : ℝ)
  (bottom_left : ℝ × ℝ)

/-- Parabola defined by y = (1/5)x^2 + ax + b -/
structure Parabola :=
  (a : ℝ)
  (b : ℝ)

/-- Configuration of square and parabola -/
structure Configuration :=
  (square : Square)
  (parabola : Parabola)
  (passes_through_B : Bool)
  (passes_through_C : Bool)
  (vertex_on_AD : Bool)

/-- Theorem: Sum of roots of quadratic polynomial -/
theorem sum_of_roots (config : Configuration) : ℝ :=
  20

/-- Main theorem: Sum of roots is 20 -/
theorem sum_of_roots_is_twenty (config : Configuration) :
  sum_of_roots config = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_sum_of_roots_is_twenty_l997_99745


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l997_99726

/-- Calculates the cost of plastering a rectangular tank's walls and bottom -/
def plasteringCost (length width depth rate : ℝ) : ℝ :=
  let wallArea := 2 * (length * depth + width * depth)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * rate

/-- Theorem: The cost of plastering a 35m x 18m x 10m tank at ₹135 per sq m is ₹228,150 -/
theorem tank_plastering_cost :
  plasteringCost 35 18 10 135 = 228150 := by
  sorry

end NUMINAMATH_CALUDE_tank_plastering_cost_l997_99726


namespace NUMINAMATH_CALUDE_olivia_payment_l997_99744

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

end NUMINAMATH_CALUDE_olivia_payment_l997_99744


namespace NUMINAMATH_CALUDE_xy_values_l997_99730

theorem xy_values (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 108) : 
  xy = 0 ∨ xy = 72 := by
sorry

end NUMINAMATH_CALUDE_xy_values_l997_99730


namespace NUMINAMATH_CALUDE_arithmetic_sequence_m_l997_99746

/-- An arithmetic sequence with its first n terms sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- The main theorem -/
theorem arithmetic_sequence_m (seq : ArithmeticSequence) (m : ℕ) :
  m ≥ 2 →
  seq.S (m - 1) = -2 →
  seq.S m = 0 →
  seq.S (m + 1) = 3 →
  m = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_m_l997_99746


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l997_99727

theorem polynomial_division_quotient :
  let dividend := fun (z : ℚ) => 4 * z^5 + 2 * z^4 - 7 * z^3 + 5 * z^2 - 3 * z + 8
  let divisor := fun (z : ℚ) => 3 * z + 1
  let quotient := fun (z : ℚ) => (4/3) * z^4 - (19/3) * z^3 + (34/3) * z^2 - (61/9) * z - 1
  ∀ z : ℚ, dividend z = divisor z * quotient z + (275/27) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l997_99727


namespace NUMINAMATH_CALUDE_new_person_weight_l997_99735

/-- Given a group of 8 people where one person weighing 40 kg is replaced,
    if the average weight increases by 2.5 kg, then the new person weighs 60 kg. -/
theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 40 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 60 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l997_99735


namespace NUMINAMATH_CALUDE_cubic_inequality_false_l997_99742

theorem cubic_inequality_false : 
  ¬(∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
sorry

end NUMINAMATH_CALUDE_cubic_inequality_false_l997_99742


namespace NUMINAMATH_CALUDE_complex_number_magnitude_squared_l997_99715

theorem complex_number_magnitude_squared : 
  ∀ z : ℂ, z + Complex.abs z = 5 - 3*I → Complex.abs z^2 = 11.56 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_squared_l997_99715


namespace NUMINAMATH_CALUDE_partnership_profit_l997_99773

/-- Calculates the profit of a business partnership given the investments and profit sharing rules -/
theorem partnership_profit (mary_investment mike_investment : ℚ) 
  (h1 : mary_investment = 700)
  (h2 : mike_investment = 300)
  (h3 : mary_investment + mike_investment > 0) :
  ∃ (P : ℚ), 
    (P / 6 + 7 * (2 * P / 3) / 10) - (P / 6 + 3 * (2 * P / 3) / 10) = 800 ∧ 
    P = 3000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_l997_99773


namespace NUMINAMATH_CALUDE_f_monotonicity_and_inequality_l997_99741

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * Real.log x + 11

theorem f_monotonicity_and_inequality :
  (∀ x ∈ Set.Ioo (0 : ℝ) 1, StrictMonoOn f (Set.Ioo 0 1)) ∧
  (∀ x ∈ Set.Ioi 1, StrictMonoOn f (Set.Ioi 1)) ∧
  (∀ x > 0, f x > -x^3 + 3*x^2 + (3 - x) * Real.exp x) := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_inequality_l997_99741


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l997_99793

/-- Calculates the sampling interval for systematic sampling -/
def sampling_interval (N : ℕ) (n : ℕ) : ℕ := N / n

/-- Theorem: The sampling interval for a population of 1000 and sample size of 20 is 50 -/
theorem systematic_sampling_interval :
  sampling_interval 1000 20 = 50 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l997_99793


namespace NUMINAMATH_CALUDE_solution_pairs_count_l997_99704

theorem solution_pairs_count : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    4 * p.1 + 7 * p.2 = 600 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 601) (Finset.range 601))).card ∧ n = 22 :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_count_l997_99704


namespace NUMINAMATH_CALUDE_train_speed_and_length_l997_99728

/-- Proves that given a bridge of length 1000m, if a train takes 60s to pass from the beginning
    to the end of the bridge and spends 40s on the bridge, then the speed of the train is 20 m/s
    and its length is 200m. -/
theorem train_speed_and_length
  (bridge_length : ℝ)
  (time_to_pass : ℝ)
  (time_on_bridge : ℝ)
  (h1 : bridge_length = 1000)
  (h2 : time_to_pass = 60)
  (h3 : time_on_bridge = 40)
  : ∃ (speed length : ℝ),
    speed = 20 ∧
    length = 200 ∧
    time_to_pass * speed = bridge_length + length ∧
    time_on_bridge * speed = bridge_length - length :=
by sorry

end NUMINAMATH_CALUDE_train_speed_and_length_l997_99728


namespace NUMINAMATH_CALUDE_olympiad_numbers_equal_divisors_of_1998_l997_99752

/-- The year of the first Olympiad -/
def firstOlympiadYear : ℕ := 1999

/-- The year of the n-th Olympiad -/
def olympiadYear (n : ℕ) : ℕ := firstOlympiadYear + n - 1

/-- The set of positive integers n such that n divides the year of the n-th Olympiad -/
def validOlympiadNumbers : Set ℕ :=
  {n : ℕ | n > 0 ∧ n ∣ olympiadYear n}

/-- The set of divisors of 1998 -/
def divisorsOf1998 : Set ℕ :=
  {n : ℕ | n > 0 ∧ n ∣ 1998}

theorem olympiad_numbers_equal_divisors_of_1998 :
  validOlympiadNumbers = divisorsOf1998 :=
by sorry

end NUMINAMATH_CALUDE_olympiad_numbers_equal_divisors_of_1998_l997_99752


namespace NUMINAMATH_CALUDE_function_zero_implies_a_range_l997_99749

theorem function_zero_implies_a_range (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioo (-1) 1 ∧ 2 * a * x₀ - a + 3 = 0) →
  a ∈ Set.Iio (-3) ∪ Set.Ioi 1 := by
sorry

end NUMINAMATH_CALUDE_function_zero_implies_a_range_l997_99749


namespace NUMINAMATH_CALUDE_distinct_colorings_count_l997_99795

/-- The number of distinct colorings of n points on a circle with k blue points
    and at least p red points between each pair of consecutive blue points. -/
def distinctColorings (n k p : ℕ) : ℚ :=
  if 2 ≤ k ∧ k ≤ n / (p + 1) then
    (1 : ℚ) / k * (Nat.choose (n - k * p - 1) (k - 1) : ℚ)
  else
    0

theorem distinct_colorings_count
  (n k p : ℕ)
  (h1 : 0 < n ∧ 0 < k ∧ 0 < p)
  (h2 : 2 ≤ k)
  (h3 : k ≤ n / (p + 1)) :
  distinctColorings n k p = (1 : ℚ) / k * (Nat.choose (n - k * p - 1) (k - 1) : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_distinct_colorings_count_l997_99795


namespace NUMINAMATH_CALUDE_simplify_sqrt_240_l997_99738

theorem simplify_sqrt_240 : Real.sqrt 240 = 4 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_240_l997_99738


namespace NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l997_99754

def num_maple_trees : ℕ := 5
def num_oak_trees : ℕ := 4
def num_birch_trees : ℕ := 6

def total_trees : ℕ := num_maple_trees + num_oak_trees + num_birch_trees

def num_non_birch_trees : ℕ := num_maple_trees + num_oak_trees

def num_slots_for_birch : ℕ := num_non_birch_trees + 1

theorem birch_tree_arrangement_probability :
  (Nat.choose num_slots_for_birch num_birch_trees : ℚ) / (Nat.choose total_trees num_birch_trees) = 2 / 45 := by
  sorry

end NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l997_99754


namespace NUMINAMATH_CALUDE_stove_and_wall_repair_cost_l997_99796

/-- The total cost of replacing a stove and repairing wall damage -/
theorem stove_and_wall_repair_cost :
  let stove_cost : ℚ := 1200
  let wall_repair_cost : ℚ := stove_cost / 6
  let total_cost : ℚ := stove_cost + wall_repair_cost
  total_cost = 1400 := by sorry

end NUMINAMATH_CALUDE_stove_and_wall_repair_cost_l997_99796


namespace NUMINAMATH_CALUDE_gino_bears_count_l997_99790

/-- The number of brown bears Gino has -/
def brown_bears : ℕ := 15

/-- The number of white bears Gino has -/
def white_bears : ℕ := 24

/-- The number of black bears Gino has -/
def black_bears : ℕ := 27

/-- The total number of bears Gino has -/
def total_bears : ℕ := brown_bears + white_bears + black_bears

theorem gino_bears_count : total_bears = 66 := by
  sorry

end NUMINAMATH_CALUDE_gino_bears_count_l997_99790


namespace NUMINAMATH_CALUDE_complex_equation_l997_99784

theorem complex_equation : Complex.I ^ 3 + 2 * Complex.I = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_l997_99784


namespace NUMINAMATH_CALUDE_sum_of_abs_values_l997_99772

theorem sum_of_abs_values (a b : ℝ) : 
  (|a| = 3 ∧ |b| = 5 ∧ a > b) → (a + b = -2 ∨ a + b = -8) := by
sorry

end NUMINAMATH_CALUDE_sum_of_abs_values_l997_99772


namespace NUMINAMATH_CALUDE_no_integer_solutions_l997_99753

theorem no_integer_solutions : ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l997_99753


namespace NUMINAMATH_CALUDE_garden_breadth_calculation_l997_99725

/-- Represents a rectangular garden --/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden --/
def perimeter (g : RectangularGarden) : ℝ :=
  2 * (g.length + g.breadth)

theorem garden_breadth_calculation :
  ∀ g : RectangularGarden,
    g.length = 205 →
    perimeter g = 600 →
    g.breadth = 95 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_calculation_l997_99725


namespace NUMINAMATH_CALUDE_optimal_sampling_methods_for_given_surveys_l997_99734

/-- Represents different sampling methods -/
inductive SamplingMethod
| Stratified
| Random
| Systematic

/-- Represents a survey with its characteristics -/
structure Survey where
  population_size : ℕ
  sample_size : ℕ
  has_distinct_groups : Bool

/-- Determines the optimal sampling method for a given survey -/
def optimal_sampling_method (s : Survey) : SamplingMethod :=
  if s.has_distinct_groups then SamplingMethod.Stratified
  else if s.population_size ≤ 20 then SamplingMethod.Random
  else SamplingMethod.Systematic

/-- The first survey from the problem -/
def survey1 : Survey :=
  { population_size := 500
  , sample_size := 100
  , has_distinct_groups := true }

/-- The second survey from the problem -/
def survey2 : Survey :=
  { population_size := 12
  , sample_size := 3
  , has_distinct_groups := false }

theorem optimal_sampling_methods_for_given_surveys :
  optimal_sampling_method survey1 = SamplingMethod.Stratified ∧
  optimal_sampling_method survey2 = SamplingMethod.Random :=
sorry


end NUMINAMATH_CALUDE_optimal_sampling_methods_for_given_surveys_l997_99734


namespace NUMINAMATH_CALUDE_additional_machines_needed_l997_99771

/-- Given that 15 machines can finish a job in 36 days, prove that 5 additional machines
    are needed to finish the job in one-fourth less time. -/
theorem additional_machines_needed (machines : ℕ) (days : ℕ) (job : ℕ) :
  machines = 15 →
  days = 36 →
  job = machines * days →
  (machines + 5) * (days - days / 4) = job :=
by sorry

end NUMINAMATH_CALUDE_additional_machines_needed_l997_99771


namespace NUMINAMATH_CALUDE_max_area_triangle_area_equals_perimeter_l997_99786

theorem max_area_triangle_area_equals_perimeter : ∃ (a b c : ℕ+),
  (∃ (s : ℝ), s = (a + b + c : ℝ) / 2 ∧ 
   (s * (s - a) * (s - b) * (s - c) : ℝ) = ((a + b + c) ^ 2 : ℝ) / 4) ∧
  (∀ (x y z : ℕ+), 
    (∃ (t : ℝ), t = (x + y + z : ℝ) / 2 ∧ 
     (t * (t - x) * (t - y) * (t - z) : ℝ) = ((x + y + z) ^ 2 : ℝ) / 4) →
    (x + y + z : ℝ) ≤ (a + b + c : ℝ)) :=
by sorry

#check max_area_triangle_area_equals_perimeter

end NUMINAMATH_CALUDE_max_area_triangle_area_equals_perimeter_l997_99786


namespace NUMINAMATH_CALUDE_sixth_group_frequency_is_one_tenth_l997_99787

/-- Represents the distribution of students across six groups in a mathematics competition. -/
structure StudentDistribution where
  total : ℕ
  group1 : ℕ
  group2 : ℕ
  group3 : ℕ
  group4 : ℕ
  freq5 : ℚ

/-- Calculates the frequency of the sixth group given a student distribution. -/
def sixthGroupFrequency (d : StudentDistribution) : ℚ :=
  1 - (d.group1 + d.group2 + d.group3 + d.group4 : ℚ) / d.total - d.freq5

/-- Theorem stating that for the given distribution, the frequency of the sixth group is 0.1. -/
theorem sixth_group_frequency_is_one_tenth 
  (d : StudentDistribution)
  (h1 : d.total = 40)
  (h2 : d.group1 = 10)
  (h3 : d.group2 = 5)
  (h4 : d.group3 = 7)
  (h5 : d.group4 = 6)
  (h6 : d.freq5 = 1/5) :
  sixthGroupFrequency d = 1/10 := by
  sorry


end NUMINAMATH_CALUDE_sixth_group_frequency_is_one_tenth_l997_99787


namespace NUMINAMATH_CALUDE_inequality_proof_l997_99782

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c) / a + (c + a) / b + (a + b) / c ≥ 
  ((a^2 + b^2 + c^2) * (a*b + b*c + c*a)) / (a*b*c * (a + b + c)) + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l997_99782


namespace NUMINAMATH_CALUDE_rectangle_area_l997_99762

theorem rectangle_area (length width diagonal : ℝ) (h1 : length / width = 5 / 2) (h2 : length^2 + width^2 = diagonal^2) (h3 : diagonal = 13) : 
  length * width = (10 / 29) * diagonal^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l997_99762


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l997_99757

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem quadratic_equation_roots (p q : ℕ) (hp : is_prime p) (hq : is_prime q) :
  ∃ (x y : ℕ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ x^2 - p*x + 2*q = 0 ∧ y^2 - p*y + 2*q = 0 →
  (∃ r : ℕ, (r = x ∨ r = y) ∧ is_prime r) ∧
  is_prime (p - q) ∧
  ¬(∀ x y : ℕ, x^2 - p*x + 2*q = 0 → y^2 - p*y + 2*q = 0 → x ≠ y → Even (x - y)) ∧
  ¬(is_prime (p^2 + 2*q)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l997_99757


namespace NUMINAMATH_CALUDE_donny_gas_station_payment_l997_99747

/-- Given the conditions of Donny's gas station visit, prove that he paid $350. -/
theorem donny_gas_station_payment (tank_capacity : ℝ) (initial_fuel : ℝ) (fuel_cost : ℝ) (change : ℝ)
  (h1 : tank_capacity = 150)
  (h2 : initial_fuel = 38)
  (h3 : fuel_cost = 3)
  (h4 : change = 14) :
  (tank_capacity - initial_fuel) * fuel_cost + change = 350 := by
  sorry

#check donny_gas_station_payment

end NUMINAMATH_CALUDE_donny_gas_station_payment_l997_99747


namespace NUMINAMATH_CALUDE_cost_of_birdhouses_l997_99767

-- Define the constants
def planks_per_birdhouse : ℕ := 7
def nails_per_birdhouse : ℕ := 20
def cost_per_nail : ℚ := 0.05
def cost_per_plank : ℕ := 3
def num_birdhouses : ℕ := 4

-- Define the theorem
theorem cost_of_birdhouses :
  (num_birdhouses * (planks_per_birdhouse * cost_per_plank +
   nails_per_birdhouse * cost_per_nail) : ℚ) = 88 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_birdhouses_l997_99767


namespace NUMINAMATH_CALUDE_problem_solution_l997_99755

/-- Permutations of n items taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := sorry

/-- Combinations of n items taken r at a time -/
def combinations (n : ℕ) (r : ℕ) : ℕ := sorry

theorem problem_solution (r : ℕ) (k : ℕ) : 
  permutations 32 r = k * combinations 32 r → k = 720 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l997_99755


namespace NUMINAMATH_CALUDE_units_digit_of_sum_even_factorials_l997_99712

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def sum_of_even_factorials : ℕ := 
  factorial 2 + factorial 4 + factorial 6 + factorial 8 + factorial 10

theorem units_digit_of_sum_even_factorials :
  units_digit sum_of_even_factorials = 6 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_even_factorials_l997_99712


namespace NUMINAMATH_CALUDE_unique_integer_sequence_l997_99722

theorem unique_integer_sequence : ∃! x : ℤ, x = ((x + 2)/2 + 2)/2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_sequence_l997_99722


namespace NUMINAMATH_CALUDE_largest_divisor_of_m_l997_99717

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 216 ∣ m^2) : 
  36 = Nat.gcd m 36 ∧ ∀ k : ℕ, k > 36 → k ∣ m → k ∣ 36 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_m_l997_99717


namespace NUMINAMATH_CALUDE_statement_two_is_false_l997_99766

/-- Definition of the heart operation -/
def heart (x y : ℝ) : ℝ := 2 * |x - y| + 1

/-- Theorem stating that Statement 2 is false -/
theorem statement_two_is_false :
  ∃ x y : ℝ, 3 * (heart x y) ≠ heart (3 * x) (3 * y) :=
sorry

end NUMINAMATH_CALUDE_statement_two_is_false_l997_99766


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l997_99719

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallelLines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point 
  (given_line : Line) 
  (point : Point) 
  (h_given_line : given_line = ⟨2, 1, -5⟩) 
  (h_point : point = ⟨1, 0⟩) : 
  ∃ (parallel_line : Line), 
    parallelLines given_line parallel_line ∧ 
    pointOnLine point parallel_line ∧ 
    parallel_line = ⟨2, 1, -2⟩ :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l997_99719


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l997_99748

theorem triangle_angle_calculation (a b : ℝ) (B : ℝ) (hA : 0 < a) (hB : 0 < b) (hC : 0 < B) (hD : B < π) 
  (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 3) (hB : B = π / 3) :
  ∃ (A : ℝ), 
    0 < A ∧ A < π / 2 ∧ 
    Real.sin A = (a * Real.sin B) / b ∧
    A = π / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l997_99748


namespace NUMINAMATH_CALUDE_joe_count_l997_99706

theorem joe_count (barry_count kevin_count julie_count : ℕ) 
  (nice_count : ℕ) (joe_nice_ratio : ℚ) :
  barry_count = 24 →
  kevin_count = 20 →
  julie_count = 80 →
  nice_count = 99 →
  joe_nice_ratio = 1/10 →
  ∃ (joe_count : ℕ),
    joe_count = 50 ∧
    nice_count = barry_count + 
                 (kevin_count / 2) + 
                 (julie_count * 3 / 4) + 
                 (joe_count * joe_nice_ratio) :=
by sorry

end NUMINAMATH_CALUDE_joe_count_l997_99706


namespace NUMINAMATH_CALUDE_two_distinct_decorations_l997_99743

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

end NUMINAMATH_CALUDE_two_distinct_decorations_l997_99743


namespace NUMINAMATH_CALUDE_difference_sum_of_powers_of_three_l997_99709

def S : Finset ℕ := Finset.range 9

def difference_sum (S : Finset ℕ) : ℕ :=
  S.sum (λ i => S.sum (λ j => if i > j then (3^i - 3^j) else 0))

theorem difference_sum_of_powers_of_three : difference_sum S = 68896 := by
  sorry

end NUMINAMATH_CALUDE_difference_sum_of_powers_of_three_l997_99709


namespace NUMINAMATH_CALUDE_solve_eggs_problem_l997_99799

def eggs_problem (breakfast_eggs lunch_eggs total_eggs : ℕ) : Prop :=
  let dinner_eggs := total_eggs - (breakfast_eggs + lunch_eggs)
  dinner_eggs = 1

theorem solve_eggs_problem :
  eggs_problem 2 3 6 :=
sorry

end NUMINAMATH_CALUDE_solve_eggs_problem_l997_99799


namespace NUMINAMATH_CALUDE_sandwich_shop_period_length_l997_99737

/-- Represents the Eat "N Go Mobile Sausage Sandwich Shop scenario -/
structure SandwichShop where
  jalapeno_strips_per_sandwich : ℕ
  minutes_per_sandwich : ℕ
  total_jalapeno_strips : ℕ

/-- Calculates the period length in minutes for a given SandwichShop scenario -/
def period_length (shop : SandwichShop) : ℕ :=
  (shop.total_jalapeno_strips / shop.jalapeno_strips_per_sandwich) * shop.minutes_per_sandwich

/-- Theorem stating that under the given conditions, the period length is 60 minutes -/
theorem sandwich_shop_period_length :
  ∀ (shop : SandwichShop),
    shop.jalapeno_strips_per_sandwich = 4 →
    shop.minutes_per_sandwich = 5 →
    shop.total_jalapeno_strips = 48 →
    period_length shop = 60 := by
  sorry


end NUMINAMATH_CALUDE_sandwich_shop_period_length_l997_99737


namespace NUMINAMATH_CALUDE_comic_books_calculation_l997_99702

theorem comic_books_calculation (initial : ℕ) (bought : ℕ) : 
  initial = 14 → bought = 6 → initial / 2 + bought = 13 := by
  sorry

end NUMINAMATH_CALUDE_comic_books_calculation_l997_99702


namespace NUMINAMATH_CALUDE_assembly_time_proof_l997_99789

/-- Calculates the total time spent assembling furniture -/
def total_assembly_time (chairs tables time_per_piece : ℕ) : ℕ :=
  (chairs + tables) * time_per_piece

/-- Proves that given 20 chairs, 8 tables, and 6 minutes per piece, 
    the total assembly time is 168 minutes -/
theorem assembly_time_proof :
  total_assembly_time 20 8 6 = 168 := by
  sorry

end NUMINAMATH_CALUDE_assembly_time_proof_l997_99789


namespace NUMINAMATH_CALUDE_enrollment_difference_l997_99785

def highest_enrollment : ℕ := 2150
def lowest_enrollment : ℕ := 980

theorem enrollment_difference : highest_enrollment - lowest_enrollment = 1170 := by
  sorry

end NUMINAMATH_CALUDE_enrollment_difference_l997_99785


namespace NUMINAMATH_CALUDE_no_positive_reals_satisfy_inequalities_l997_99791

theorem no_positive_reals_satisfy_inequalities :
  ¬ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (4 * (a * b + b * c + c * a) - 1 ≥ a^2 + b^2 + c^2) ∧
    (a^2 + b^2 + c^2 ≥ 3 * (a^3 + b^3 + c^3)) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_reals_satisfy_inequalities_l997_99791


namespace NUMINAMATH_CALUDE_square_difference_simplification_l997_99721

theorem square_difference_simplification (y : ℝ) (h : y^2 ≥ 16) :
  (4 - Real.sqrt (y^2 - 16))^2 = y^2 - 8 * Real.sqrt (y^2 - 16) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_simplification_l997_99721


namespace NUMINAMATH_CALUDE_factorial_sum_div_l997_99764

theorem factorial_sum_div (n : ℕ) : (8 * n.factorial + 9 * 8 * n.factorial + 10 * 9 * 8 * n.factorial) / n.factorial = 800 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_div_l997_99764


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l997_99768

-- Define the conditions
def p (x : ℝ) : Prop := |x - 4| ≤ 6
def q (x m : ℝ) : Prop := x ≤ 1 + m

-- State the theorem
theorem sufficient_condition_range (m : ℝ) :
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) ↔ m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l997_99768


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l997_99740

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (1 : ℂ) / (1 - Complex.I) = ↑a + ↑b * Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l997_99740


namespace NUMINAMATH_CALUDE_round_robin_tournament_l997_99751

theorem round_robin_tournament (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_round_robin_tournament_l997_99751


namespace NUMINAMATH_CALUDE_largest_value_l997_99792

theorem largest_value (a b : ℝ) (ha : 0 < a) (ha1 : a < 1) (hb : 0 < b) (hb1 : b < 1) (hab : a ≠ b) :
  (a + b) = max (a + b) (max (2 * Real.sqrt (a * b)) (max (a^2 + b^2) (2 * a * b))) :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l997_99792


namespace NUMINAMATH_CALUDE_man_speed_calculation_man_speed_approx_7_9916_l997_99781

/-- Calculates the speed of a man given the parameters of a passing train -/
theorem man_speed_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let man_speed_mps := train_speed_mps - relative_speed
  let man_speed_kmph := man_speed_mps * (3600 / 1000)
  man_speed_kmph

/-- The speed of the man is approximately 7.9916 kmph -/
theorem man_speed_approx_7_9916 : 
  ∃ ε > 0, |man_speed_calculation 350 68 20.99832013438925 - 7.9916| < ε :=
sorry

end NUMINAMATH_CALUDE_man_speed_calculation_man_speed_approx_7_9916_l997_99781


namespace NUMINAMATH_CALUDE_edges_after_intersection_l997_99710

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ

/-- Represents the result of intersecting a polyhedron with planes -/
def intersect_with_planes (Q : ConvexPolyhedron) (num_planes : ℕ) : ℕ := sorry

/-- Theorem: The number of edges after intersection is 450 -/
theorem edges_after_intersection (Q : ConvexPolyhedron) (h1 : Q.edges = 150) :
  intersect_with_planes Q Q.vertices = 450 := by sorry

end NUMINAMATH_CALUDE_edges_after_intersection_l997_99710


namespace NUMINAMATH_CALUDE_alan_phone_price_l997_99756

theorem alan_phone_price (john_price : ℝ) (percentage : ℝ) (alan_price : ℝ) :
  john_price = 2040 →
  percentage = 0.02 →
  john_price = alan_price * (1 + percentage) →
  alan_price = 1999.20 := by
sorry

end NUMINAMATH_CALUDE_alan_phone_price_l997_99756


namespace NUMINAMATH_CALUDE_expand_expression_l997_99780

theorem expand_expression (x : ℝ) : 5 * (4 * x^3 - 3 * x^2 + 2 * x - 7) = 20 * x^3 - 15 * x^2 + 10 * x - 35 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l997_99780


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_composite_l997_99776

theorem quadratic_roots_imply_composite (a b : ℤ) :
  (∃ x₁ x₂ : ℕ+, x₁^2 + a * x₁ + b + 1 = 0 ∧ x₂^2 + a * x₂ + b + 1 = 0) →
  ∃ m n : ℕ, m > 1 ∧ n > 1 ∧ a^2 + b^2 = m * n :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_composite_l997_99776


namespace NUMINAMATH_CALUDE_sum_of_cubes_l997_99720

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 7) (h2 : a * b = 3) : a^3 + b^3 = 280 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l997_99720


namespace NUMINAMATH_CALUDE_teaching_ratio_l997_99770

def total_years : ℕ := 52
def calculus_years : ℕ := 4

def algebra_years (c : ℕ) : ℕ := 2 * c

def statistics_years (t a c : ℕ) : ℕ := t - a - c

theorem teaching_ratio :
  let c := calculus_years
  let a := algebra_years c
  let s := statistics_years total_years a c
  (s : ℚ) / a = 5 / 1 :=
sorry

end NUMINAMATH_CALUDE_teaching_ratio_l997_99770


namespace NUMINAMATH_CALUDE_hot_pot_restaurant_problem_l997_99775

-- Define variables for set prices
variable (price_A price_B : ℚ)

-- Define variables for daily quantities and income
variable (day1_A day1_B day2_A day2_B : ℕ)
variable (income1 income2 : ℚ)

-- Define variables for costs and constraints
variable (cost_A cost_B : ℚ)
variable (max_sets : ℕ)
variable (set_A_ratio : ℚ)

-- Define variables for extra ingredients
variable (extra_cost : ℚ)

-- Define variables for Xiaoming's spending
variable (xiaoming_total : ℚ)
variable (xiaoming_set_A_ratio : ℚ)

-- Theorem statement
theorem hot_pot_restaurant_problem 
  (h1 : day1_A * price_A + day1_B * price_B = income1)
  (h2 : day2_A * price_A + day2_B * price_B = income2)
  (h3 : day1_A = 20 ∧ day1_B = 10 ∧ income1 = 2800)
  (h4 : day2_A = 15 ∧ day2_B = 20 ∧ income2 = 3350)
  (h5 : cost_A = 45 ∧ cost_B = 50)
  (h6 : max_sets = 50)
  (h7 : set_A_ratio = 1/5)
  (h8 : extra_cost = 10)
  (h9 : xiaoming_total = 1610)
  (h10 : xiaoming_set_A_ratio = 1/4) :
  price_A = 90 ∧ 
  price_B = 100 ∧ 
  (∃ (m : ℕ), m ≥ max_sets * set_A_ratio ∧ 
              m ≤ max_sets ∧ 
              (price_A - cost_A) * m + (price_B - cost_B) * (max_sets - m) = 2455) ∧
  (∃ (x y : ℕ), x = xiaoming_set_A_ratio * (x + y) ∧
                90 * x + 100 * y + 110 * (3 * x - y) = xiaoming_total ∧
                3 * x - y = 5) := by
  sorry

end NUMINAMATH_CALUDE_hot_pot_restaurant_problem_l997_99775


namespace NUMINAMATH_CALUDE_min_distance_complex_l997_99711

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + Real.sqrt 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 1 + Real.sqrt 2 ∧ 
  ∀ (w : ℂ), Complex.abs (w + Real.sqrt 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_l997_99711


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l997_99705

theorem quadratic_equation_solution (c : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + c = 0 ∧ (-x)^2 + 3*(-x) - c = 0) →
  (∀ x : ℝ, x^2 - 3*x + c = 0 ↔ (x = 0 ∨ x = 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l997_99705


namespace NUMINAMATH_CALUDE_square_perimeter_sum_l997_99750

theorem square_perimeter_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^2 + b^2 = 65) (h4 : a^2 - b^2 = 33) : 
  4*a + 4*b = 44 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_sum_l997_99750


namespace NUMINAMATH_CALUDE_number_of_nephews_l997_99701

-- Define the price of a candy as the base unit
def candy_price : ℚ := 1

-- Define the prices of other items in terms of candy price
def orange_price : ℚ := 2 * candy_price
def cake_price : ℚ := 4 * candy_price
def chocolate_price : ℚ := 7 * candy_price
def book_price : ℚ := 14 * candy_price

-- Define the cost of one gift
def gift_cost : ℚ := candy_price + orange_price + cake_price + chocolate_price + book_price

-- Define the total number of each item if all money was spent on that item
def total_candies : ℕ := 224
def total_oranges : ℕ := 112
def total_cakes : ℕ := 56
def total_chocolates : ℕ := 32
def total_books : ℕ := 16

-- Theorem: The number of nephews is 8
theorem number_of_nephews : ℕ := by
  sorry

end NUMINAMATH_CALUDE_number_of_nephews_l997_99701


namespace NUMINAMATH_CALUDE_decimal_24_equals_binary_11000_l997_99774

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: aux (m / 2)
    aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem decimal_24_equals_binary_11000 : 
  to_binary 24 = [false, false, false, true, true] ∧ 
  from_binary [false, false, false, true, true] = 24 := by
  sorry

end NUMINAMATH_CALUDE_decimal_24_equals_binary_11000_l997_99774


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l997_99729

theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 2 →
  a 5 = 8 →
  a 3 = 4 ∨ a 3 = -4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l997_99729


namespace NUMINAMATH_CALUDE_equation_solution_l997_99783

theorem equation_solution :
  ∃! y : ℝ, (4 * y - 2) / (5 * y - 5) = 3 / 4 ∧ y = -7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l997_99783


namespace NUMINAMATH_CALUDE_compare_magnitudes_l997_99797

theorem compare_magnitudes (a b : ℝ) (ha : a ≠ 1) :
  a^2 + b^2 > 2*(a - b - 1) := by
sorry

end NUMINAMATH_CALUDE_compare_magnitudes_l997_99797


namespace NUMINAMATH_CALUDE_abc_product_l997_99739

theorem abc_product (a b c : ℤ) 
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : a + b + c = 30)
  (h3 : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + 630 / (a * b * c) = 1) :
  a * b * c = 483 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l997_99739


namespace NUMINAMATH_CALUDE_total_ways_from_A_to_C_l997_99736

/-- The number of roads from village A to village B -/
def roads_A_to_B : ℕ := 3

/-- The number of roads from village B to village C -/
def roads_B_to_C : ℕ := 2

/-- The total number of different ways to go from village A to village C via village B -/
def total_ways : ℕ := roads_A_to_B * roads_B_to_C

theorem total_ways_from_A_to_C : total_ways = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_from_A_to_C_l997_99736


namespace NUMINAMATH_CALUDE_camryn_practice_schedule_l997_99765

theorem camryn_practice_schedule :
  let trumpet := 11
  let flute := 3
  let piano := 7
  let violin := 13
  let guitar := 5
  Nat.lcm trumpet (Nat.lcm flute (Nat.lcm piano (Nat.lcm violin guitar))) = 15015 := by
  sorry

end NUMINAMATH_CALUDE_camryn_practice_schedule_l997_99765


namespace NUMINAMATH_CALUDE_square_side_length_l997_99732

theorem square_side_length 
  (total_wire : ℝ) 
  (triangle_perimeter : ℝ) 
  (h1 : total_wire = 78) 
  (h2 : triangle_perimeter = 46) : 
  (total_wire - triangle_perimeter) / 4 = 8 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l997_99732


namespace NUMINAMATH_CALUDE_problem_solution_l997_99798

theorem problem_solution (a : ℝ) (f g : ℝ → ℝ) 
  (h1 : a > 0) 
  (h2 : f (g a) = 18)
  (h3 : ∀ x, f x = x^2 - 2)
  (h4 : ∀ x, g x = x^2 + 6) : 
  a = Real.sqrt 14 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l997_99798


namespace NUMINAMATH_CALUDE_symmetric_point_xoz_l997_99788

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOz plane -/
def xOzPlane : Set Point3D := {p : Point3D | p.y = 0}

/-- Symmetry with respect to the xOz plane -/
def symmetricPointXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

theorem symmetric_point_xoz (p : Point3D) :
  p.x = 2 ∧ p.y = 1 ∧ p.z = 3 →
  symmetricPointXOZ p = Point3D.mk 2 (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_xoz_l997_99788


namespace NUMINAMATH_CALUDE_log_base_value_l997_99703

theorem log_base_value (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x > 0, f x = Real.log x / Real.log a) (h2 : f 4 = 2) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_base_value_l997_99703


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l997_99731

theorem quadratic_equation_unique_solution (b : ℝ) (hb : b ≠ 0) :
  (∃! x, b * x^2 + 15 * x + 4 = 0) →
  (∃ x, b * x^2 + 15 * x + 4 = 0 ∧ x = -8/15) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l997_99731


namespace NUMINAMATH_CALUDE_girls_combined_average_l997_99778

/-- Represents a high school with exam scores -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- The average score calculation problem -/
theorem girls_combined_average 
  (cedar : School)
  (drake : School)
  (boys_combined_avg : ℝ)
  (h_cedar : cedar.combined_avg = 78)
  (h_drake : drake.combined_avg = 88)
  (h_cedar_boys : cedar.boys_avg = 75)
  (h_cedar_girls : cedar.girls_avg = 80)
  (h_drake_boys : drake.boys_avg = 85)
  (h_drake_girls : drake.girls_avg = 92)
  (h_boys_combined : boys_combined_avg = 83) :
  ∃ (girls_combined_avg : ℝ), girls_combined_avg = 88 := by
  sorry


end NUMINAMATH_CALUDE_girls_combined_average_l997_99778


namespace NUMINAMATH_CALUDE_vovochka_max_candies_l997_99758

/-- Represents the candy distribution problem --/
structure CandyDistribution where
  total_candies : ℕ
  num_classmates : ℕ
  min_group_size : ℕ
  min_group_candies : ℕ

/-- Calculates the maximum number of candies Vovochka can keep --/
def max_candies_kept (cd : CandyDistribution) : ℕ :=
  cd.total_candies - (cd.num_classmates * (cd.min_group_candies / cd.min_group_size))

/-- Theorem stating the maximum number of candies Vovochka can keep --/
theorem vovochka_max_candies :
  let cd := CandyDistribution.mk 200 25 16 100
  max_candies_kept cd = 37 := by
  sorry

end NUMINAMATH_CALUDE_vovochka_max_candies_l997_99758


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l997_99708

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

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l997_99708


namespace NUMINAMATH_CALUDE_tiffany_bags_on_monday_l997_99707

theorem tiffany_bags_on_monday :
  ∀ (bags_monday : ℕ),
  bags_monday + 8 = 12 →
  bags_monday = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_tiffany_bags_on_monday_l997_99707


namespace NUMINAMATH_CALUDE_combined_tax_rate_approx_l997_99759

/-- Represents the tax system for a group of individuals in a fictional universe. -/
structure TaxSystem where
  mork_tax_rate : ℝ
  mork_deduction : ℝ
  mindy_tax_rate : ℝ
  mindy_income_multiple : ℝ
  mindy_tax_break : ℝ
  bickley_income_multiple : ℝ
  bickley_tax_rate : ℝ
  bickley_deduction : ℝ
  exidor_income_fraction : ℝ
  exidor_tax_rate : ℝ
  exidor_tax_break : ℝ

/-- Calculates the combined tax rate for the group. -/
def combined_tax_rate (ts : TaxSystem) : ℝ :=
  sorry

/-- Theorem stating that the combined tax rate is approximately 23.57% -/
theorem combined_tax_rate_approx (ts : TaxSystem) 
  (h1 : ts.mork_tax_rate = 0.45)
  (h2 : ts.mork_deduction = 0.10)
  (h3 : ts.mindy_tax_rate = 0.20)
  (h4 : ts.mindy_income_multiple = 4)
  (h5 : ts.mindy_tax_break = 0.05)
  (h6 : ts.bickley_income_multiple = 2)
  (h7 : ts.bickley_tax_rate = 0.25)
  (h8 : ts.bickley_deduction = 0.07)
  (h9 : ts.exidor_income_fraction = 0.5)
  (h10 : ts.exidor_tax_rate = 0.30)
  (h11 : ts.exidor_tax_break = 0.08) :
  abs (combined_tax_rate ts - 0.2357) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_approx_l997_99759


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l997_99733

/-- The equation represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  (k - 3) * (k + 3) > 0

/-- The condition k > 3 is sufficient for the equation to represent a hyperbola -/
theorem sufficient_condition (k : ℝ) (h : k > 3) : is_hyperbola k := by sorry

/-- The condition k > 3 is not necessary for the equation to represent a hyperbola -/
theorem not_necessary_condition : ∃ k, k ≤ 3 ∧ is_hyperbola k := by sorry

/-- k > 3 is a sufficient but not necessary condition for the equation to represent a hyperbola -/
theorem sufficient_but_not_necessary : 
  (∀ k, k > 3 → is_hyperbola k) ∧ (∃ k, k ≤ 3 ∧ is_hyperbola k) := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l997_99733


namespace NUMINAMATH_CALUDE_not_sum_solution_equation_example_sum_solution_equation_condition_l997_99760

/-- Definition of a sum solution equation -/
def is_sum_solution_equation (a b : ℝ) : Prop :=
  (b / a) = b + a

/-- Theorem 1: 3x = 4.5 is not a sum solution equation -/
theorem not_sum_solution_equation_example : ¬ is_sum_solution_equation 3 4.5 := by
  sorry

/-- Theorem 2: 5x = m + 1 is a sum solution equation iff m = -29/4 -/
theorem sum_solution_equation_condition (m : ℝ) : 
  is_sum_solution_equation 5 (m + 1) ↔ m = -29/4 := by
  sorry

end NUMINAMATH_CALUDE_not_sum_solution_equation_example_sum_solution_equation_condition_l997_99760


namespace NUMINAMATH_CALUDE_eggs_last_24_days_l997_99700

/-- Calculates the number of days eggs will last given initial eggs, daily egg laying, and daily consumption. -/
def days_eggs_last (initial_eggs : ℕ) (daily_laid : ℕ) (daily_consumed : ℕ) : ℕ :=
  initial_eggs / (daily_consumed - daily_laid)

/-- Theorem: Given 72 initial eggs, a hen laying 1 egg per day, and a family consuming 4 eggs per day, the eggs will last for 24 days. -/
theorem eggs_last_24_days :
  days_eggs_last 72 1 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_eggs_last_24_days_l997_99700


namespace NUMINAMATH_CALUDE_three_digit_number_theorem_l997_99714

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧
  n / 100 + n % 10 = 8

theorem three_digit_number_theorem :
  ∀ n : ℕ, is_valid_number n → (n = 810 ∨ n = 840 ∨ n = 870) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_theorem_l997_99714


namespace NUMINAMATH_CALUDE_quadratic_roots_when_k_negative_l997_99779

theorem quadratic_roots_when_k_negative (k : ℝ) (h : k < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 + x₁ + k - 1 = 0) ∧ 
  (x₂^2 + x₂ + k - 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_when_k_negative_l997_99779


namespace NUMINAMATH_CALUDE_standard_poodle_height_difference_l997_99761

/-- The height difference between the standard poodle and the miniature poodle -/
def height_difference (standard_height miniature_height : ℕ) : ℕ :=
  standard_height - miniature_height

/-- Theorem: The standard poodle is 8 inches taller than the miniature poodle -/
theorem standard_poodle_height_difference :
  let toy_height : ℕ := 14
  let standard_height : ℕ := 28
  let miniature_height : ℕ := toy_height + 6
  height_difference standard_height miniature_height = 8 := by
  sorry

end NUMINAMATH_CALUDE_standard_poodle_height_difference_l997_99761


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l997_99713

theorem complex_magnitude_proof : Complex.abs (-4 + (7/6) * Complex.I) = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l997_99713


namespace NUMINAMATH_CALUDE_sin_330_degrees_l997_99763

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l997_99763


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l997_99723

-- Define the set of real numbers except 1
def RealExceptOne : Set ℝ := {x : ℝ | x ≠ 1}

-- Define the property of the expression being meaningful
def IsMeaningful (x : ℝ) : Prop := x - 1 ≠ 0

-- Theorem statement
theorem meaningful_expression_range :
  ∀ x : ℝ, IsMeaningful x ↔ x ∈ RealExceptOne :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l997_99723
