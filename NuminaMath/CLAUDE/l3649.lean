import Mathlib

namespace calculate_at_20at_l3649_364917

-- Define the @ operation (postfix)
def at_post (x : ℤ) : ℤ := 9 - x

-- Define the @ operation (prefix)
def at_pre (x : ℤ) : ℤ := x - 9

-- Theorem statement
theorem calculate_at_20at : at_pre (at_post 20) = -20 := by
  sorry

end calculate_at_20at_l3649_364917


namespace distance_on_line_l3649_364930

/-- Given two points on a line, prove that their distance is |x₁ - x₂|√(1 + k²) -/
theorem distance_on_line (k b x₁ x₂ : ℝ) :
  let y₁ := k * x₁ + b
  let y₂ := k * x₂ + b
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = |x₁ - x₂| * Real.sqrt (1 + k^2) := by
  sorry

end distance_on_line_l3649_364930


namespace boric_acid_mixture_volume_l3649_364918

theorem boric_acid_mixture_volume 
  (volume_1_percent : ℝ) 
  (volume_5_percent : ℝ) 
  (h1 : volume_1_percent = 15) 
  (h2 : volume_5_percent = 15) : 
  volume_1_percent + volume_5_percent = 30 := by
  sorry

end boric_acid_mixture_volume_l3649_364918


namespace lottery_distribution_l3649_364902

theorem lottery_distribution (lottery_win : ℝ) (num_students : ℕ) : 
  lottery_win = 155250 →
  num_students = 100 →
  (lottery_win / 1000) * num_students = 15525 := by
  sorry

end lottery_distribution_l3649_364902


namespace least_three_digit_multiple_l3649_364914

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 3 = 0 ∧ n % 4 = 0 ∧ n % 9 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 3 = 0 ∧ m % 4 = 0 ∧ m % 9 = 0 → m ≥ n) ∧
  n = 108 := by
sorry

end least_three_digit_multiple_l3649_364914


namespace inequality_proof_l3649_364925

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end inequality_proof_l3649_364925


namespace max_silver_tokens_l3649_364984

/-- Represents the number of tokens of each color --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange booth --/
structure Booth where
  inputRed : ℕ
  inputBlue : ℕ
  outputSilver : ℕ
  outputRed : ℕ
  outputBlue : ℕ

/-- The initial token count --/
def initialTokens : TokenCount :=
  { red := 100, blue := 50, silver := 0 }

/-- The first exchange booth --/
def booth1 : Booth :=
  { inputRed := 3, inputBlue := 0, outputSilver := 1, outputRed := 0, outputBlue := 2 }

/-- The second exchange booth --/
def booth2 : Booth :=
  { inputRed := 0, inputBlue := 4, outputSilver := 1, outputRed := 2, outputBlue := 0 }

/-- Predicate to check if an exchange is possible --/
def canExchange (tokens : TokenCount) (booth : Booth) : Prop :=
  tokens.red ≥ booth.inputRed ∧ tokens.blue ≥ booth.inputBlue

/-- The final token count after all possible exchanges --/
noncomputable def finalTokens : TokenCount :=
  sorry

/-- Theorem stating that the maximum number of silver tokens is 103 --/
theorem max_silver_tokens : finalTokens.silver = 103 := by
  sorry

end max_silver_tokens_l3649_364984


namespace lcm_sum_triplet_l3649_364959

theorem lcm_sum_triplet (a b c : ℕ+) :
  a + b + c = Nat.lcm (Nat.lcm a.val b.val) c.val ↔ b = 2 * a ∧ c = 3 * a := by
  sorry

end lcm_sum_triplet_l3649_364959


namespace p_range_nonnegative_reals_l3649_364904

/-- The function p(x) = x^4 - 6x^2 + 9 -/
def p (x : ℝ) : ℝ := x^4 - 6*x^2 + 9

theorem p_range_nonnegative_reals :
  Set.range (fun (x : ℝ) ↦ p x) = Set.Ici (0 : ℝ) := by sorry

end p_range_nonnegative_reals_l3649_364904


namespace product_of_functions_l3649_364981

theorem product_of_functions (x : ℝ) (h : x > 0) :
  Real.sqrt (x * (x + 1)) * (1 / Real.sqrt x) = Real.sqrt (x + 1) := by
  sorry

end product_of_functions_l3649_364981


namespace medication_expiration_time_l3649_364927

-- Define the number of seconds in 8!
def medication_duration : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1

-- Define the number of seconds in a minute in this system
def seconds_per_minute : ℕ := 50

-- Define the release time
def release_time : String := "3 PM on February 14"

-- Define a function to calculate the expiration time
def calculate_expiration_time (duration : ℕ) (seconds_per_min : ℕ) (start_time : String) : String :=
  sorry

-- Theorem statement
theorem medication_expiration_time :
  calculate_expiration_time medication_duration seconds_per_minute release_time = "February 15, around 4 AM" :=
sorry

end medication_expiration_time_l3649_364927


namespace orphanage_donation_percentage_l3649_364946

def total_income : ℝ := 200000
def children_percentage : ℝ := 0.15
def num_children : ℕ := 3
def wife_percentage : ℝ := 0.30
def final_amount : ℝ := 40000

theorem orphanage_donation_percentage :
  let children_total_percentage := children_percentage * num_children
  let total_given_percentage := children_total_percentage + wife_percentage
  let remaining_amount := total_income * (1 - total_given_percentage)
  let donated_amount := remaining_amount - final_amount
  donated_amount / remaining_amount = 0.20 := by
sorry

end orphanage_donation_percentage_l3649_364946


namespace triangle_property_l3649_364942

theorem triangle_property (a b c A B C : Real) (h1 : b = a * (Real.cos C - Real.sin C))
  (h2 : a = Real.sqrt 10) (h3 : Real.sin B = Real.sqrt 2 * Real.sin C) :
  A = 3 * Real.pi / 4 ∧ 1/2 * b * c * Real.sin A = 1 := by
  sorry

end triangle_property_l3649_364942


namespace union_covers_reals_l3649_364998

def A : Set ℝ := {x | x ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x > a}

theorem union_covers_reals (a : ℝ) : A ∪ B a = Set.univ → a ≤ 0 := by
  sorry

end union_covers_reals_l3649_364998


namespace orchestra_members_count_l3649_364997

theorem orchestra_members_count :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 300 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 1 ∧ 
  n % 7 = 5 ∧
  n = 651 := by sorry

end orchestra_members_count_l3649_364997


namespace profit_percent_approximation_l3649_364960

def selling_price : ℝ := 2524.36
def cost_price : ℝ := 2400

theorem profit_percent_approximation :
  let profit := selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  ∃ ε > 0, abs (profit_percent - 5.18) < ε :=
by sorry

end profit_percent_approximation_l3649_364960


namespace f_10_equals_756_l3649_364967

def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 6

theorem f_10_equals_756 : f 10 = 756 := by sorry

end f_10_equals_756_l3649_364967


namespace remainder_when_c_divided_by_b_l3649_364999

theorem remainder_when_c_divided_by_b (a b c : ℕ) 
  (h1 : b = 3 * a + 3) 
  (h2 : c = 9 * a + 11) : 
  c % b = 2 := by
sorry

end remainder_when_c_divided_by_b_l3649_364999


namespace equation_solution_l3649_364935

theorem equation_solution :
  ∃ x : ℝ, 3 * x - 2 * x = 7 ∧ x = 7 :=
by sorry

end equation_solution_l3649_364935


namespace amys_tomato_soup_cans_l3649_364964

/-- Amy's soup purchase problem -/
theorem amys_tomato_soup_cans (total_soups chicken_soups tomato_soups : ℕ) : 
  total_soups = 9 →
  chicken_soups = 6 →
  total_soups = chicken_soups + tomato_soups →
  tomato_soups = 3 := by
sorry

end amys_tomato_soup_cans_l3649_364964


namespace poverty_education_relationship_l3649_364989

/-- Regression line for poverty and education data -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Define a point on the regression line -/
structure Point where
  x : ℝ
  y : ℝ

/-- The regression line equation -/
def on_regression_line (line : RegressionLine) (p : Point) : Prop :=
  p.y = line.slope * p.x + line.intercept

theorem poverty_education_relationship (line : RegressionLine) 
    (h_slope : line.slope = 0.8) (h_intercept : line.intercept = 4.6)
    (p1 p2 : Point) (h_on_line1 : on_regression_line line p1) 
    (h_on_line2 : on_regression_line line p2) (h_x_diff : p2.x - p1.x = 1) :
    p2.y - p1.y = 0.8 := by
  sorry

end poverty_education_relationship_l3649_364989


namespace bee_closest_point_to_flower_l3649_364976

/-- The point where the bee starts moving away from the flower -/
def closest_point : ℝ × ℝ := (4.6, 13.8)

/-- The location of the flower -/
def flower_location : ℝ × ℝ := (10, 12)

/-- The path of the bee -/
def bee_path (x : ℝ) : ℝ := 3 * x

theorem bee_closest_point_to_flower :
  let (c, d) := closest_point
  -- The point is on the bee's path
  (d = bee_path c) ∧
  -- This point is the closest to the flower
  (∀ x y, y = bee_path x → (x - 10)^2 + (y - 12)^2 ≥ (c - 10)^2 + (d - 12)^2) ∧
  -- The sum of coordinates is 18.4
  (c + d = 18.4) := by sorry

end bee_closest_point_to_flower_l3649_364976


namespace f_odd_a_range_l3649_364958

variable (f : ℝ → ℝ)

/-- f is an increasing function -/
axiom f_increasing : ∀ x y, x < y → f x < f y

/-- f satisfies the functional equation f(x+y) = f(x) + f(y) for all x, y ∈ ℝ -/
axiom f_add : ∀ x y, f (x + y) = f x + f y

/-- f is an odd function -/
theorem f_odd : ∀ x, f (-x) = -f x := by sorry

/-- The range of a for which f(x²) - 2f(x) < f(ax) - 2f(a) has exactly 3 positive integer solutions -/
theorem a_range : 
  {a : ℝ | 5 < a ∧ a ≤ 6} = 
  {a : ℝ | ∃! (x₁ x₂ x₃ : ℕ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    (∀ x : ℝ, x > 0 → (f (x^2) - 2*f x < f (a*x) - 2*f a ↔ x = x₁ ∨ x = x₂ ∨ x = x₃))} := by sorry

end f_odd_a_range_l3649_364958


namespace bike_riding_average_l3649_364906

/-- Calculates the average miles ridden per day given total miles and years --/
def average_miles_per_day (total_miles : ℕ) (years : ℕ) : ℚ :=
  total_miles / (years * 365)

/-- Theorem stating that riding 3,285 miles over 3 years averages to 3 miles per day --/
theorem bike_riding_average :
  average_miles_per_day 3285 3 = 3 := by
  sorry

end bike_riding_average_l3649_364906


namespace two_valid_configurations_l3649_364909

-- Define a 4x4 table as a function from (Fin 4 × Fin 4) to Char
def Table := Fin 4 → Fin 4 → Char

-- Define the swap operations
def swapFirstTwoRows (t : Table) : Table :=
  fun i j => if i = 0 then t 1 j else if i = 1 then t 0 j else t i j

def swapFirstTwoCols (t : Table) : Table :=
  fun i j => if j = 0 then t i 1 else if j = 1 then t i 0 else t i j

def swapLastTwoCols (t : Table) : Table :=
  fun i j => if j = 2 then t i 3 else if j = 3 then t i 2 else t i j

-- Define the property of identical letters in corresponding quadrants
def maintainsQuadrantProperty (t1 t2 : Table) : Prop :=
  ∀ i j, (t1 i j = t1 (i + 2) j ∧ t1 i j = t1 i (j + 2) ∧ t1 i j = t1 (i + 2) (j + 2)) →
         (t2 i j = t2 (i + 2) j ∧ t2 i j = t2 i (j + 2) ∧ t2 i j = t2 (i + 2) (j + 2))

-- Define the initial table
def initialTable : Table :=
  fun i j => match (i, j) with
  | (0, 0) => 'A' | (0, 1) => 'B' | (0, 2) => 'C' | (0, 3) => 'D'
  | (1, 0) => 'D' | (1, 1) => 'C' | (1, 2) => 'B' | (1, 3) => 'A'
  | (2, 0) => 'C' | (2, 1) => 'A' | (2, 2) => 'C' | (2, 3) => 'A'
  | (3, 0) => 'B' | (3, 1) => 'D' | (3, 2) => 'B' | (3, 3) => 'D'

-- The main theorem
theorem two_valid_configurations :
  ∃! (validConfigs : Finset Table),
    validConfigs.card = 2 ∧
    (∀ t ∈ validConfigs,
      maintainsQuadrantProperty initialTable
        (swapLastTwoCols (swapFirstTwoCols (swapFirstTwoRows t)))) := by
  sorry

end two_valid_configurations_l3649_364909


namespace sachins_age_l3649_364983

theorem sachins_age (sachin rahul : ℝ) 
  (h1 : rahul = sachin + 7)
  (h2 : sachin / rahul = 7 / 9) :
  sachin = 24.5 := by
sorry

end sachins_age_l3649_364983


namespace data_median_and_mode_l3649_364912

def data : List Int := [15, 17, 14, 10, 15, 17, 17, 16, 14, 12]

def median (l : List Int) : ℚ := sorry

def mode (l : List Int) : Int := sorry

theorem data_median_and_mode :
  median data = 14.5 ∧ mode data = 17 := by sorry

end data_median_and_mode_l3649_364912


namespace bamboo_pole_sections_l3649_364933

/-- Represents the properties of a bamboo pole with n sections -/
structure BambooPole (n : ℕ) where
  -- The common difference of the arithmetic sequence
  d : ℝ
  -- The number of sections is at least 6
  h_n_ge_6 : n ≥ 6
  -- The length of the top section is 10 cm
  h_top_length : 10 = 10
  -- The total length of the last three sections is 114 cm
  h_last_three : (10 + (n - 3) * d) + (10 + (n - 2) * d) + (10 + (n - 1) * d) = 114
  -- The length of the 6th section is the geometric mean of the lengths of the first and last sections
  h_geometric_mean : (10 + 5 * d)^2 = 10 * (10 + (n - 1) * d)

/-- The number of sections in the bamboo pole is 16 -/
theorem bamboo_pole_sections : ∃ (n : ℕ), ∃ (p : BambooPole n), n = 16 :=
sorry

end bamboo_pole_sections_l3649_364933


namespace simplify_expression_l3649_364955

theorem simplify_expression (x : ℝ) : 4*x - 3*x^2 + 6 + (8 - 5*x + 2*x^2) = -x^2 - x + 14 := by
  sorry

end simplify_expression_l3649_364955


namespace max_value_of_expression_l3649_364949

theorem max_value_of_expression (x y : ℝ) 
  (hx : |x - 1| ≤ 2) (hy : |y - 1| ≤ 2) : 
  ∃ (a b : ℝ), |a - 2*b + 1| = 6 ∧ |x - 2*y + 1| ≤ 6 := by
  sorry

end max_value_of_expression_l3649_364949


namespace min_squares_to_exceed_10000_l3649_364985

/-- Represents the squaring operation on a calculator --/
def square (x : ℕ) : ℕ := x * x

/-- Represents n iterations of squaring, starting from x --/
def iterate_square (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => square (iterate_square x n)

/-- The theorem to be proved --/
theorem min_squares_to_exceed_10000 :
  (∃ n : ℕ, iterate_square 5 n > 10000) ∧
  (∀ n : ℕ, iterate_square 5 n > 10000 → n ≥ 3) ∧
  (iterate_square 5 3 > 10000) :=
sorry

end min_squares_to_exceed_10000_l3649_364985


namespace line_intersection_y_axis_l3649_364943

/-- A line passing through two points (2, 9) and (5, 17) intersects the y-axis at (0, 11/3) -/
theorem line_intersection_y_axis : 
  ∃ (m b : ℚ), 
    (9 = m * 2 + b) ∧ 
    (17 = m * 5 + b) ∧ 
    (11/3 = b) := by sorry

end line_intersection_y_axis_l3649_364943


namespace smallest_non_prime_non_square_no_small_factors_l3649_364986

def is_prime (n : ℕ) : Prop := sorry

def is_square (n : ℕ) : Prop := sorry

def has_prime_factor_less_than (n m : ℕ) : Prop := sorry

theorem smallest_non_prime_non_square_no_small_factors : 
  (∀ k < 4087, k > 0 → is_prime k ∨ is_square k ∨ has_prime_factor_less_than k 60) ∧
  ¬ is_prime 4087 ∧
  ¬ is_square 4087 ∧
  ¬ has_prime_factor_less_than 4087 60 := by sorry

end smallest_non_prime_non_square_no_small_factors_l3649_364986


namespace max_product_sum_l3649_364919

def values : Finset ℕ := {1, 3, 5, 7}

theorem max_product_sum (a b c d : ℕ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_in_values : a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values) :
  (a * b + b * c + c * d + d * a) ≤ 64 :=
sorry

end max_product_sum_l3649_364919


namespace soybean_experiment_results_l3649_364908

/-- Represents the weight distribution of soybean samples -/
structure WeightDistribution :=
  (low : ℕ) -- count in [100, 150) range
  (mid : ℕ) -- count in [150, 200) range
  (high : ℕ) -- count in [200, 250] range

/-- Represents the experimental setup for soybean fields -/
structure SoybeanExperiment :=
  (field_A : WeightDistribution)
  (field_B : WeightDistribution)
  (sample_size : ℕ)
  (critical_value : ℝ)

/-- Calculates the chi-square statistic for the experiment -/
def calculate_chi_square (exp : SoybeanExperiment) : ℝ :=
  sorry

/-- Calculates the probability of selecting at least one full grain from both fields -/
def probability_full_grain (exp : SoybeanExperiment) : ℚ :=
  sorry

/-- Calculates the expected number of full grains in 100 samples from field A -/
def expected_full_grains (exp : SoybeanExperiment) : ℕ :=
  sorry

/-- Calculates the variance of full grains in 100 samples from field A -/
def variance_full_grains (exp : SoybeanExperiment) : ℚ :=
  sorry

/-- Main theorem about the soybean experiment -/
theorem soybean_experiment_results (exp : SoybeanExperiment) 
  (h1 : exp.field_A = ⟨3, 6, 11⟩)
  (h2 : exp.field_B = ⟨6, 10, 4⟩)
  (h3 : exp.sample_size = 20)
  (h4 : exp.critical_value = 5.024) :
  calculate_chi_square exp > exp.critical_value ∧
  probability_full_grain exp = 89 / 100 ∧
  expected_full_grains exp = 55 ∧
  variance_full_grains exp = 99 / 4 :=
sorry

end soybean_experiment_results_l3649_364908


namespace problem_statement_l3649_364940

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) : 
  (-1 < x - y ∧ x - y < 1) ∧ 
  ((1 / x + x / y) ≥ 3 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 1 ∧ 1 / a + a / b = 3) :=
by sorry

end problem_statement_l3649_364940


namespace race_has_six_laps_l3649_364979

/-- Represents a cyclist in the race -/
structure Cyclist where
  name : String
  lap_time : ℕ

/-- Represents the race setup -/
structure Race where
  total_laps : ℕ
  vasya : Cyclist
  petya : Cyclist
  kolya : Cyclist

/-- The race conditions are satisfied -/
def race_conditions (r : Race) : Prop :=
  r.vasya.lap_time + 2 = r.petya.lap_time ∧
  r.petya.lap_time + 3 = r.kolya.lap_time ∧
  r.vasya.lap_time * r.total_laps = r.petya.lap_time * (r.total_laps - 1) ∧
  r.vasya.lap_time * r.total_laps = r.kolya.lap_time * (r.total_laps - 2)

/-- The theorem stating that the race has 6 laps -/
theorem race_has_six_laps :
  ∃ (r : Race), race_conditions r ∧ r.total_laps = 6 := by sorry

end race_has_six_laps_l3649_364979


namespace empty_bucket_weight_l3649_364928

/-- Given a bucket with known weights at different fill levels, 
    calculate the weight of the empty bucket -/
theorem empty_bucket_weight (M N P : ℝ) 
  (h_full : ∃ x y : ℝ, x + y = M ∧ x + 3/4 * y = N ∧ x + 1/3 * y = P) : 
  ∃ x : ℝ, x = 4 * N - 3 * M := by
  sorry

end empty_bucket_weight_l3649_364928


namespace b_is_eighteen_l3649_364936

/-- Represents the ages of three people a, b, and c. -/
structure Ages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.a = ages.b + 2 ∧
  ages.b = 2 * ages.c ∧
  ages.a + ages.b + ages.c = 47

/-- The theorem statement -/
theorem b_is_eighteen (ages : Ages) (h : satisfiesConditions ages) : ages.b = 18 := by
  sorry

end b_is_eighteen_l3649_364936


namespace find_missing_number_l3649_364990

/-- Given two sets of numbers with known means, find the missing number in the second set. -/
theorem find_missing_number (x : ℝ) (missing : ℝ) : 
  (12 + x + 42 + 78 + 104) / 5 = 62 →
  (128 + 255 + 511 + missing + x) / 5 = 398.2 →
  missing = 1023 := by
sorry

end find_missing_number_l3649_364990


namespace count_multiples_eq_42_l3649_364980

/-- The number of positive integers less than 201 that are multiples of either 6 or 8, but not both -/
def count_multiples : ℕ :=
  (Finset.filter (fun n => n % 6 = 0 ∨ n % 8 = 0) (Finset.range 201)).card -
  (Finset.filter (fun n => n % 6 = 0 ∧ n % 8 = 0) (Finset.range 201)).card

theorem count_multiples_eq_42 : count_multiples = 42 := by
  sorry

end count_multiples_eq_42_l3649_364980


namespace evaluate_expression_l3649_364905

theorem evaluate_expression : 3 * (-3)^4 + 3 * (-3)^3 + 3 * (-3)^2 + 3 * 3^2 + 3 * 3^3 + 3 * 3^4 = 540 := by
  sorry

end evaluate_expression_l3649_364905


namespace newton_sports_club_membership_ratio_l3649_364982

/-- Proves that given the average ages of female and male members, and the overall average age,
    the ratio of female to male members is 8:17 --/
theorem newton_sports_club_membership_ratio
  (avg_age_female : ℝ)
  (avg_age_male : ℝ)
  (avg_age_all : ℝ)
  (h_female : avg_age_female = 45)
  (h_male : avg_age_male = 20)
  (h_all : avg_age_all = 28)
  : ∃ (f m : ℝ), f > 0 ∧ m > 0 ∧ f / m = 8 / 17 := by
  sorry

end newton_sports_club_membership_ratio_l3649_364982


namespace expression_one_expression_two_expression_three_expression_four_l3649_364988

-- 1. 75 + 7 × 5
theorem expression_one : 75 + 7 * 5 = 110 := by sorry

-- 2. 148 - 48 ÷ 2
theorem expression_two : 148 - 48 / 2 = 124 := by sorry

-- 3. (400 - 160) ÷ 8
theorem expression_three : (400 - 160) / 8 = 30 := by sorry

-- 4. 4 × 25 × 7
theorem expression_four : 4 * 25 * 7 = 700 := by sorry

end expression_one_expression_two_expression_three_expression_four_l3649_364988


namespace computer_cost_l3649_364996

theorem computer_cost (C : ℝ) : 
  C + (1/5) * C + 300 = 2100 → C = 1500 := by
  sorry

end computer_cost_l3649_364996


namespace sum_remainder_mod_17_l3649_364948

theorem sum_remainder_mod_17 : ∃ k : ℕ, (78 + 79 + 80 + 81 + 82 + 83 + 84 + 85) = 17 * k + 6 := by
  sorry

end sum_remainder_mod_17_l3649_364948


namespace no_perfect_square_sum_l3649_364945

theorem no_perfect_square_sum (n : ℕ) : n ≥ 1 → ¬∃ (m : ℕ), 2^n + 12^n + 2014^n = m^2 := by
  sorry

end no_perfect_square_sum_l3649_364945


namespace milk_production_l3649_364903

theorem milk_production (M : ℝ) 
  (h1 : M > 0) 
  (h2 : M * 0.25 * 0.5 = 2) : M = 16 := by
  sorry

end milk_production_l3649_364903


namespace exam_students_count_l3649_364915

theorem exam_students_count (total_average : ℝ) (excluded_average : ℝ) (remaining_average : ℝ) (excluded_count : ℕ) :
  total_average = 80 →
  excluded_average = 20 →
  remaining_average = 92 →
  excluded_count = 5 →
  ∃ N : ℕ, 
    N * total_average = (N - excluded_count) * remaining_average + excluded_count * excluded_average ∧
    N = 30 := by
  sorry

end exam_students_count_l3649_364915


namespace quadratic_root_sum_minus_product_l3649_364900

theorem quadratic_root_sum_minus_product (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 5 = 0 → 
  x₂^2 - 3*x₂ - 5 = 0 → 
  x₁ + x₂ - x₁ * x₂ = 8 := by
sorry

end quadratic_root_sum_minus_product_l3649_364900


namespace necessary_sufficient_condition_for_ax0_eq_b_l3649_364994

theorem necessary_sufficient_condition_for_ax0_eq_b 
  (a b x₀ : ℝ) (h : a < 0) :
  (a * x₀ = b) ↔ 
  (∀ x : ℝ, (1/2) * a * x^2 - b * x ≤ (1/2) * a * x₀^2 - b * x₀) :=
by sorry

end necessary_sufficient_condition_for_ax0_eq_b_l3649_364994


namespace frank_reading_total_l3649_364934

theorem frank_reading_total (book1_pages_per_day book1_days book2_pages_per_day book2_days book3_pages_per_day book3_days : ℕ) :
  book1_pages_per_day = 22 →
  book1_days = 569 →
  book2_pages_per_day = 35 →
  book2_days = 315 →
  book3_pages_per_day = 18 →
  book3_days = 450 →
  book1_pages_per_day * book1_days + book2_pages_per_day * book2_days + book3_pages_per_day * book3_days = 31643 :=
by
  sorry

end frank_reading_total_l3649_364934


namespace inverse_and_determinant_properties_l3649_364975

/-- Given a 2x2 matrix A with its inverse, prove properties about A^2 and (A^(-1))^2 -/
theorem inverse_and_determinant_properties (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = ![![3, 4], ![-2, -2]]) : 
  (A^2)⁻¹ = ![![1, 4], ![-2, 0]] ∧ 
  Matrix.det ((A⁻¹)^2) = 8 := by
  sorry


end inverse_and_determinant_properties_l3649_364975


namespace jia_jia_clovers_l3649_364929

theorem jia_jia_clovers : 
  ∀ (total_leaves : ℕ) (four_leaf_clovers : ℕ) (three_leaf_clovers : ℕ),
  total_leaves = 100 →
  four_leaf_clovers = 1 →
  total_leaves = 4 * four_leaf_clovers + 3 * three_leaf_clovers →
  three_leaf_clovers = 32 := by
sorry

end jia_jia_clovers_l3649_364929


namespace petyas_calculation_error_l3649_364977

theorem petyas_calculation_error (a : ℕ) (h1 : a > 2) : 
  ¬ (∃ (n : ℕ), 
    (a - 2) * (a + 3) - a = n ∧ 
    (∃ (k : ℕ), n.digits 10 = List.replicate 2023 8 ++ List.replicate 2023 3 ++ List.replicate k 0)) :=
by sorry

end petyas_calculation_error_l3649_364977


namespace quadratic_value_l3649_364932

/-- A quadratic function with vertex (2,7) passing through (0,-7) -/
def f (x : ℝ) : ℝ :=
  let a : ℝ := -3.5
  a * (x - 2)^2 + 7

theorem quadratic_value : f 5 = -24.5 := by
  sorry

end quadratic_value_l3649_364932


namespace min_value_theorem_l3649_364944

theorem min_value_theorem (a₁ a₂ : ℝ) 
  (h : (3 / (3 + 2 * Real.sin a₁)) + (2 / (4 - Real.sin (2 * a₂))) = 1) :
  ∃ (m : ℝ), m = π / 4 ∧ ∀ (x : ℝ), |4 * π - a₁ + a₂| ≥ m :=
sorry

end min_value_theorem_l3649_364944


namespace weight_of_8_moles_AlI3_l3649_364907

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of Aluminum atoms in AlI3 -/
def num_Al_atoms : ℕ := 1

/-- The number of Iodine atoms in AlI3 -/
def num_I_atoms : ℕ := 3

/-- The number of moles of AlI3 -/
def num_moles_AlI3 : ℝ := 8

/-- The molecular weight of AlI3 in g/mol -/
def molecular_weight_AlI3 : ℝ := 
  num_Al_atoms * atomic_weight_Al + num_I_atoms * atomic_weight_I

/-- The weight of a given number of moles of AlI3 in grams -/
def weight_AlI3 (moles : ℝ) : ℝ := moles * molecular_weight_AlI3

theorem weight_of_8_moles_AlI3 : 
  weight_AlI3 num_moles_AlI3 = 3261.44 := by
  sorry


end weight_of_8_moles_AlI3_l3649_364907


namespace arithmetic_square_root_of_nine_l3649_364968

theorem arithmetic_square_root_of_nine : ∃ x : ℝ, x ≥ 0 ∧ x^2 = 9 ∧ ∀ y : ℝ, y ≥ 0 ∧ y^2 = 9 → y = x :=
by sorry

end arithmetic_square_root_of_nine_l3649_364968


namespace root_sum_reciprocal_l3649_364963

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 7*x^2 + 7*x - 1

-- Define the roots
noncomputable def a : ℝ := 3 + Real.sqrt 8
noncomputable def b : ℝ := 3 - Real.sqrt 8

-- Theorem statement
theorem root_sum_reciprocal : 
  f a = 0 ∧ f b = 0 ∧ (∀ x, f x = 0 → b ≤ x ∧ x ≤ a) → a / b + b / a = 34 := by
  sorry

end root_sum_reciprocal_l3649_364963


namespace science_competition_accuracy_l3649_364993

theorem science_competition_accuracy (correct : ℕ) (wrong : ℕ) (target_accuracy : ℚ) (additional : ℕ) : 
  correct = 30 →
  wrong = 6 →
  target_accuracy = 85/100 →
  (correct + additional) / (correct + wrong + additional) = target_accuracy →
  additional = 4 := by
sorry

end science_competition_accuracy_l3649_364993


namespace exactly_one_machine_maintenance_probability_l3649_364957

/-- The probability that exactly one of three independent machines needs maintenance,
    given their individual maintenance probabilities. -/
theorem exactly_one_machine_maintenance_probability
  (p_A p_B p_C : ℝ)
  (h_A : 0 ≤ p_A ∧ p_A ≤ 1)
  (h_B : 0 ≤ p_B ∧ p_B ≤ 1)
  (h_C : 0 ≤ p_C ∧ p_C ≤ 1)
  (h_p_A : p_A = 0.1)
  (h_p_B : p_B = 0.2)
  (h_p_C : p_C = 0.4) :
  p_A * (1 - p_B) * (1 - p_C) +
  (1 - p_A) * p_B * (1 - p_C) +
  (1 - p_A) * (1 - p_B) * p_C = 0.444 :=
sorry

end exactly_one_machine_maintenance_probability_l3649_364957


namespace initial_value_exists_and_unique_l3649_364947

theorem initial_value_exists_and_unique : 
  ∃! x : ℤ, ∃ k : ℤ, x + 7 = k * 456 := by sorry

end initial_value_exists_and_unique_l3649_364947


namespace function_property_l3649_364941

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x + 2

theorem function_property (a : ℝ) :
  (∀ x₁ ∈ Set.Icc 1 (Real.exp 1), ∃ x₂ ∈ Set.Icc 1 (Real.exp 1), f a x₁ + f a x₂ = 4) ↔
  a = Real.exp 1 + 1 :=
by sorry

end function_property_l3649_364941


namespace second_shift_size_l3649_364901

/-- The number of members in each shift of a company and their participation in a pension program. -/
structure CompanyShifts where
  first_shift : ℕ
  second_shift : ℕ
  third_shift : ℕ
  first_participation : ℚ
  second_participation : ℚ
  third_participation : ℚ
  total_participation : ℚ

/-- The company shifts satisfy the given conditions -/
def satisfies_conditions (c : CompanyShifts) : Prop :=
  c.first_shift = 60 ∧
  c.third_shift = 40 ∧
  c.first_participation = 1/5 ∧
  c.second_participation = 2/5 ∧
  c.third_participation = 1/10 ∧
  c.total_participation = 6/25 ∧
  (c.first_shift * c.first_participation + c.second_shift * c.second_participation + c.third_shift * c.third_participation : ℚ) = 
    c.total_participation * (c.first_shift + c.second_shift + c.third_shift)

/-- The theorem stating that the second shift has 50 members -/
theorem second_shift_size (c : CompanyShifts) (h : satisfies_conditions c) : c.second_shift = 50 := by
  sorry

end second_shift_size_l3649_364901


namespace total_sleep_is_53_l3649_364926

/-- Represents Janna's sleep schedule and calculates total weekly sleep hours -/
def weekly_sleep_hours : ℝ :=
  let weekday_sleep := 7
  let weekend_sleep := 8
  let nap_hours := 0.5
  let friday_extra := 1
  
  -- Monday, Wednesday
  2 * weekday_sleep +
  -- Tuesday, Thursday (with naps)
  2 * (weekday_sleep + nap_hours) +
  -- Friday (with extra hour)
  (weekday_sleep + friday_extra) +
  -- Saturday, Sunday
  2 * weekend_sleep

/-- Theorem stating that Janna's total sleep hours in a week is 53 -/
theorem total_sleep_is_53 : weekly_sleep_hours = 53 := by
  sorry

end total_sleep_is_53_l3649_364926


namespace problem_statement_l3649_364995

theorem problem_statement (x y z : ℝ) 
  (hx : x ≠ 1) (hy : y ≠ 1) (hxy : x ≠ y)
  (h : (y * z - x^2) / (1 - x) = (x * z - y^2) / (1 - y)) :
  (y * z - x^2) / (1 - x) = x + y + z := by
  sorry

end problem_statement_l3649_364995


namespace smallest_four_digit_divisible_by_smallest_odd_primes_l3649_364939

theorem smallest_four_digit_divisible_by_smallest_odd_primes : 
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 → (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ (11 ∣ n) → n ≥ 1155 :=
by sorry

end smallest_four_digit_divisible_by_smallest_odd_primes_l3649_364939


namespace line_plane_perpendicularity_l3649_364969

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the parallel and perpendicular relations
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

theorem line_plane_perpendicularity 
  (a b : Line) (α : Plane) 
  (h1 : parallel a α) 
  (h2 : perpendicular b α) : 
  perpendicular_lines a b :=
sorry

end line_plane_perpendicularity_l3649_364969


namespace min_sum_of_bases_l3649_364987

theorem min_sum_of_bases (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  (3 * a + 6 = 6 * b + 3) → (∀ x y : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 6 = 6 * y + 3 → a + b ≤ x + y) → 
  a + b = 20 := by sorry

end min_sum_of_bases_l3649_364987


namespace product_of_difference_and_sum_of_squares_l3649_364970

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 4) 
  (h2 : a^2 + b^2 = 80) : 
  a * b = 32 := by sorry

end product_of_difference_and_sum_of_squares_l3649_364970


namespace distance_between_5th_and_25th_red_light_l3649_364973

/-- Represents the color of a light -/
inductive LightColor
| Red
| Green

/-- Represents the pattern of lights -/
def lightPattern : List LightColor :=
  [LightColor.Red, LightColor.Red, LightColor.Red, LightColor.Green, LightColor.Green]

/-- The distance between adjacent lights in inches -/
def lightDistance : ℕ := 8

/-- The number of inches in a foot -/
def inchesPerFoot : ℕ := 12

/-- The position of a red light given its index -/
def redLightPosition (n : ℕ) : ℕ :=
  (n - 1) / 3 * lightPattern.length + (n - 1) % 3 + 1

/-- The distance between two red lights given their indices -/
def distanceBetweenRedLights (n m : ℕ) : ℕ :=
  (redLightPosition m - redLightPosition n) * lightDistance

/-- The theorem to be proved -/
theorem distance_between_5th_and_25th_red_light :
  distanceBetweenRedLights 5 25 / inchesPerFoot = 56 := by
  sorry


end distance_between_5th_and_25th_red_light_l3649_364973


namespace train_length_calculation_l3649_364952

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length_calculation (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 ∧ 
  crossing_time = 30 ∧ 
  bridge_length = 255 →
  (train_speed * 1000 / 3600) * crossing_time - bridge_length = 120 := by
  sorry

#check train_length_calculation

end train_length_calculation_l3649_364952


namespace monday_tuesday_widget_difference_l3649_364938

/-- The number of widgets David produces on Monday minus the number of widgets he produces on Tuesday -/
def widget_difference (w t : ℕ) : ℕ :=
  w * t - (w + 5) * (t - 3)

theorem monday_tuesday_widget_difference (t : ℕ) (h : t ≥ 3) :
  widget_difference (2 * t) t = t + 15 := by
  sorry

end monday_tuesday_widget_difference_l3649_364938


namespace triangle_side_length_l3649_364916

theorem triangle_side_length 
  (x y z : ℝ) 
  (X Y Z : ℝ) 
  (h1 : y = 7)
  (h2 : z = 3)
  (h3 : Real.cos (Y - Z) = 7/8)
  (h4 : x > 0 ∧ y > 0 ∧ z > 0)
  (h5 : X + Y + Z = Real.pi)
  (h6 : x / Real.sin X = y / Real.sin Y)
  (h7 : y / Real.sin Y = z / Real.sin Z) :
  x = Real.sqrt 18.625 := by
sorry

end triangle_side_length_l3649_364916


namespace jason_borrowed_amount_l3649_364911

/-- Represents the value of a chore based on its position in the cycle -/
def chore_value (n : ℕ) : ℕ :=
  match n % 6 with
  | 1 => 1
  | 2 => 3
  | 3 => 5
  | 4 => 7
  | 5 => 9
  | 0 => 11
  | _ => 0  -- This case should never occur

/-- Calculates the total value of a complete cycle of 6 chores -/
def cycle_value : ℕ := 
  (chore_value 1) + (chore_value 2) + (chore_value 3) + 
  (chore_value 4) + (chore_value 5) + (chore_value 6)

/-- Theorem: Jason borrowed $288 -/
theorem jason_borrowed_amount : 
  (cycle_value * (48 / 6) = 288) := by
  sorry

end jason_borrowed_amount_l3649_364911


namespace stability_promotion_criterion_l3649_364965

/-- Represents a rice variety with its yield statistics -/
structure RiceVariety where
  name : String
  average_yield : ℝ
  variance : ℝ

/-- Determines if a rice variety is more stable than another -/
def is_more_stable (a b : RiceVariety) : Prop :=
  a.variance < b.variance

/-- Determines if a rice variety is suitable for promotion based on stability -/
def suitable_for_promotion (a b : RiceVariety) : Prop :=
  is_more_stable a b

theorem stability_promotion_criterion 
  (a b : RiceVariety) 
  (h1 : a.average_yield = b.average_yield) 
  (h2 : a.variance < b.variance) : 
  suitable_for_promotion a b :=
sorry

end stability_promotion_criterion_l3649_364965


namespace opposite_of_2023_l3649_364966

theorem opposite_of_2023 : 
  (∀ x : ℤ, x + 2023 = 0 → x = -2023) ∧ (-2023 + 2023 = 0) := by
  sorry

end opposite_of_2023_l3649_364966


namespace center_in_triangle_probability_l3649_364913

theorem center_in_triangle_probability (n : ℕ) (hn : n > 0) :
  let sides := 2 * n + 1
  (n + 1 : ℚ) / (4 * n - 2) =
    1 - (sides * (n.choose 2) : ℚ) / (sides.choose 3) :=
by sorry

end center_in_triangle_probability_l3649_364913


namespace phi_difference_bound_l3649_364910

/-- The n-th iterate of a function -/
def iterate (f : ℝ → ℝ) (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => id
  | n + 1 => f ∘ (iterate f n)

/-- The main theorem -/
theorem phi_difference_bound
  (f : ℝ → ℝ)
  (h_mono : ∀ x y, x ≤ y → f x ≤ f y)
  (h_period : ∀ x, f (x + 1) = f x + 1)
  (n : ℕ)
  (φ : ℝ → ℝ)
  (h_phi : ∀ x, φ x = iterate f n x - x) :
  ∀ x y, |φ x - φ y| < 1 :=
sorry

end phi_difference_bound_l3649_364910


namespace reciprocal_of_fraction_difference_l3649_364962

theorem reciprocal_of_fraction_difference : 
  (((2 : ℚ) / 5 - (3 : ℚ) / 4)⁻¹ : ℚ) = -(20 : ℚ) / 7 := by
  sorry

end reciprocal_of_fraction_difference_l3649_364962


namespace ancient_chinese_journey_l3649_364956

/-- Represents the distance walked on each day of a 6-day journey -/
structure JourneyDistances where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ
  day6 : ℝ

/-- The theorem statement for the ancient Chinese mathematical problem -/
theorem ancient_chinese_journey 
  (j : JourneyDistances) 
  (total_distance : j.day1 + j.day2 + j.day3 + j.day4 + j.day5 + j.day6 = 378)
  (day2_half : j.day2 = j.day1 / 2)
  (day3_half : j.day3 = j.day2 / 2)
  (day4_half : j.day4 = j.day3 / 2)
  (day5_half : j.day5 = j.day4 / 2)
  (day6_half : j.day6 = j.day5 / 2) :
  j.day3 = 48 := by
  sorry


end ancient_chinese_journey_l3649_364956


namespace problem_statement_l3649_364971

theorem problem_statement :
  (¬ (∃ x : ℝ, Real.tan x = 1 ∧ ∃ x : ℝ, x^2 - x + 1 ≤ 0)) ∧
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ↔ (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) :=
by sorry

end problem_statement_l3649_364971


namespace min_value_of_function_min_value_achieved_l3649_364953

theorem min_value_of_function (x : ℝ) : 
  (x^2 + 3) / Real.sqrt (x^2 + 2) ≥ 3 * Real.sqrt 2 / 2 :=
by sorry

theorem min_value_achieved : 
  ∃ x : ℝ, (x^2 + 3) / Real.sqrt (x^2 + 2) = 3 * Real.sqrt 2 / 2 :=
by sorry

end min_value_of_function_min_value_achieved_l3649_364953


namespace x_value_for_given_y_z_exists_constant_k_l3649_364931

/-- Given a relationship between x, y, and z, prove that x equals 5/8 for specific values of y and z -/
theorem x_value_for_given_y_z : ∀ (x y z k : ℝ), 
  (x = k * (z / y^2)) →  -- Relationship between x, y, and z
  (1 = k * (2 / 3^2)) →  -- Initial condition
  (y = 6 ∧ z = 5) →      -- New values for y and z
  x = 5/8 := by
    sorry

/-- There exists a constant k that satisfies the given conditions -/
theorem exists_constant_k : ∃ (k : ℝ), 
  (1 = k * (2 / 3^2)) ∧
  (∀ (x y z : ℝ), x = k * (z / y^2)) := by
    sorry

end x_value_for_given_y_z_exists_constant_k_l3649_364931


namespace g_range_l3649_364950

noncomputable def g (x : ℝ) : ℝ :=
  (Real.sin x ^ 3 + 4 * Real.sin x ^ 2 - 3 * Real.sin x + 3 * Real.cos x ^ 2 - 9) / (Real.sin x - 1)

theorem g_range :
  Set.range (fun x : ℝ => g x) = Set.Icc 5 9 \ {9} :=
by
  sorry

end g_range_l3649_364950


namespace equal_integers_from_cyclic_equation_l3649_364961

theorem equal_integers_from_cyclic_equation (n : ℕ+) (p : ℕ) (a b c : ℤ) 
  (hp : Nat.Prime p) 
  (h : a^n.val + p * b = b^n.val + p * c ∧ b^n.val + p * c = c^n.val + p * a) : 
  a = b ∧ b = c := by
  sorry

end equal_integers_from_cyclic_equation_l3649_364961


namespace greatest_power_of_three_in_factorial_l3649_364991

theorem greatest_power_of_three_in_factorial :
  (∃ n : ℕ, n = 6 ∧ 
   ∀ k : ℕ, 3^k ∣ Nat.factorial 16 → k ≤ n) ∧
   3^6 ∣ Nat.factorial 16 :=
by sorry

end greatest_power_of_three_in_factorial_l3649_364991


namespace quadratic_one_solution_sum_l3649_364923

theorem quadratic_one_solution_sum (b : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + b * x + 5 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 + b * y + 5 * y + 12 = 0 → y = x) →
  (∃ c : ℝ, 3 * c^2 + (-b) * c + 5 * c + 12 = 0 ∧ 
   ∀ z : ℝ, 3 * z^2 + (-b) * z + 5 * z + 12 = 0 → z = c) →
  b + (-b) = -10 :=
by sorry

end quadratic_one_solution_sum_l3649_364923


namespace cos_two_pi_seventh_inequality_l3649_364922

theorem cos_two_pi_seventh_inequality (a : ℝ) (h : a = Real.cos (2 * Real.pi / 7)) : 
  2^(a - 1/2) < 2 * a := by sorry

end cos_two_pi_seventh_inequality_l3649_364922


namespace intersection_implies_a_value_l3649_364972

def set_A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

def set_B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + (a^2-5) = 0}

theorem intersection_implies_a_value :
  ∀ a : ℝ, set_A ∩ set_B a = {2} → a = -1 ∨ a = -3 := by
  sorry

end intersection_implies_a_value_l3649_364972


namespace total_accepted_cartons_is_990_l3649_364954

/-- Represents the number of cartons delivered to a customer -/
def delivered_cartons (customer : Fin 5) : ℕ :=
  if customer.val < 2 then 300 else 200

/-- Represents the number of damaged cartons for a customer -/
def damaged_cartons (customer : Fin 5) : ℕ :=
  match customer.val with
  | 0 => 70
  | 1 => 50
  | 2 => 40
  | 3 => 30
  | 4 => 20
  | _ => 0  -- This case should never occur due to Fin 5

/-- Calculates the number of accepted cartons for a customer -/
def accepted_cartons (customer : Fin 5) : ℕ :=
  delivered_cartons customer - damaged_cartons customer

/-- The main theorem stating that the total number of accepted cartons is 990 -/
theorem total_accepted_cartons_is_990 :
  (Finset.sum Finset.univ accepted_cartons) = 990 := by
  sorry


end total_accepted_cartons_is_990_l3649_364954


namespace complex_square_on_positive_imaginary_axis_l3649_364924

theorem complex_square_on_positive_imaginary_axis (a : ℝ) :
  let z : ℂ := a + 2 * Complex.I
  (∃ (y : ℝ), y > 0 ∧ z^2 = Complex.I * y) → a = 2 := by
  sorry

end complex_square_on_positive_imaginary_axis_l3649_364924


namespace baker_remaining_cakes_l3649_364937

/-- The number of cakes remaining after a sale -/
def cakes_remaining (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

/-- Proof that the baker has 32 cakes remaining -/
theorem baker_remaining_cakes :
  let initial_cakes : ℕ := 169
  let sold_cakes : ℕ := 137
  cakes_remaining initial_cakes sold_cakes = 32 := by
sorry

end baker_remaining_cakes_l3649_364937


namespace seed_testing_methods_eq_18_l3649_364978

/-- The number of ways to select and arrange seeds for testing -/
def seed_testing_methods : ℕ :=
  (Nat.choose 3 2) * (Nat.factorial 3)

/-- Theorem stating that the number of seed testing methods is 18 -/
theorem seed_testing_methods_eq_18 : seed_testing_methods = 18 := by
  sorry

end seed_testing_methods_eq_18_l3649_364978


namespace negation_distribution_l3649_364974

theorem negation_distribution (x y : ℝ) : -(x + y) = -x + -y := by sorry

end negation_distribution_l3649_364974


namespace smallest_unsubmitted_integer_l3649_364921

/-- Represents the HMMT tournament --/
structure HMMTTournament where
  num_questions : ℕ
  max_expected_answer : ℕ

/-- Defines the property of an integer being submitted as an answer --/
def is_submitted (t : HMMTTournament) (n : ℕ) : Prop := sorry

/-- Theorem stating the smallest unsubmitted positive integer in the HMMT tournament --/
theorem smallest_unsubmitted_integer (t : HMMTTournament) 
  (h1 : t.num_questions = 66)
  (h2 : t.max_expected_answer = 150) :
  ∃ (N : ℕ), N = 139 ∧ 
  (∀ m : ℕ, m < N → is_submitted t m) ∧
  ¬(is_submitted t N) := by
  sorry

end smallest_unsubmitted_integer_l3649_364921


namespace quadratic_root_range_l3649_364951

theorem quadratic_root_range (a : ℝ) (α β : ℝ) : 
  (∃ x, x^2 - 2*a*x + a + 2 = 0) ∧ 
  (α^2 - 2*a*α + a + 2 = 0) ∧ 
  (β^2 - 2*a*β + a + 2 = 0) ∧ 
  (1 < α) ∧ (α < 2) ∧ (2 < β) ∧ (β < 3) →
  (2 < a) ∧ (a < 11/5) :=
by sorry

end quadratic_root_range_l3649_364951


namespace calculator_sales_loss_l3649_364920

theorem calculator_sales_loss (price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) :
  price = 135 ∧ profit_percent = 25 ∧ loss_percent = 25 →
  ∃ (cost1 cost2 : ℝ),
    cost1 + (profit_percent / 100) * cost1 = price ∧
    cost2 - (loss_percent / 100) * cost2 = price ∧
    2 * price - (cost1 + cost2) = -18 :=
by sorry

end calculator_sales_loss_l3649_364920


namespace pyramid_volume_l3649_364992

/-- Volume of a pyramid with a right triangular base -/
theorem pyramid_volume (c α β : ℝ) (hc : c > 0) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) :
  let volume := c^3 * Real.sin (2*α) * Real.tan β / 24
  ∃ (V : ℝ), V = volume ∧ V > 0 :=
by sorry

end pyramid_volume_l3649_364992
