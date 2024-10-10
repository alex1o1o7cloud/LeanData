import Mathlib

namespace quadratic_radical_condition_l2866_286613

theorem quadratic_radical_condition (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 3) ↔ x ≥ -3 := by sorry

end quadratic_radical_condition_l2866_286613


namespace base_conversion_sum_equality_l2866_286687

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem base_conversion_sum_equality : 
  let num1 := base_to_decimal [2, 5, 3] 8
  let den1 := base_to_decimal [1, 3] 4
  let num2 := base_to_decimal [1, 4, 4] 5
  let den2 := base_to_decimal [3, 3] 3
  (num1 : ℚ) / den1 + (num2 : ℚ) / den2 = 28.511904 := by
  sorry

end base_conversion_sum_equality_l2866_286687


namespace sugar_price_inflation_rate_l2866_286614

/-- Proves that the inflation rate is 12% given the conditions of sugar price increase --/
theorem sugar_price_inflation_rate 
  (initial_price : ℝ) 
  (final_price : ℝ) 
  (sugar_rate_increase : ℝ → ℝ) 
  (inflation_rate : ℝ) :
  initial_price = 25 →
  final_price = 33.0625 →
  (∀ x, sugar_rate_increase x = x + 0.03) →
  initial_price * (1 + sugar_rate_increase inflation_rate)^2 = final_price →
  inflation_rate = 0.12 := by
sorry

end sugar_price_inflation_rate_l2866_286614


namespace fern_leaves_count_l2866_286679

/-- The number of ferns Karen hangs -/
def num_ferns : ℕ := 6

/-- The number of fronds each fern has -/
def fronds_per_fern : ℕ := 7

/-- The number of leaves each frond has -/
def leaves_per_frond : ℕ := 30

/-- The total number of leaves on all ferns -/
def total_leaves : ℕ := num_ferns * fronds_per_fern * leaves_per_frond

theorem fern_leaves_count : total_leaves = 1260 := by
  sorry

end fern_leaves_count_l2866_286679


namespace perpendicular_lines_parallel_lines_l2866_286677

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l₂ (a x y : ℝ) : Prop := x + (a - 1) * y + a^2 - 1 = 0

-- Theorem for perpendicular lines
theorem perpendicular_lines (a : ℝ) :
  (∀ x y, l₁ a x y ∧ l₂ a x y → (a * 1 + 2 * (a - 1) = 0)) →
  a = 2/3 :=
sorry

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) :
  (∀ x y, l₁ a x y ∧ l₂ a x y → (a / 1 = 2 / (a - 1) ∧ a / 1 ≠ 6 / (a^2 - 1))) →
  a = -1 :=
sorry

end perpendicular_lines_parallel_lines_l2866_286677


namespace necessary_sufficient_condition_l2866_286692

theorem necessary_sufficient_condition (a b : ℝ) :
  (a > 1 ∧ b > 1) ↔ (a + b > 2 ∧ a * b - a - b + 1 > 0) := by
  sorry

end necessary_sufficient_condition_l2866_286692


namespace two_std_dev_below_mean_l2866_286659

/-- For a normal distribution with mean 14.5 and standard deviation 1.7,
    the value that is exactly 2 standard deviations less than the mean is 11.1. -/
theorem two_std_dev_below_mean (μ σ : ℝ) (h1 : μ = 14.5) (h2 : σ = 1.7) :
  μ - 2 * σ = 11.1 := by
  sorry

end two_std_dev_below_mean_l2866_286659


namespace simplify_expression_l2866_286631

theorem simplify_expression (a : ℝ) : a^4 * (-a)^3 = -a^7 := by
  sorry

end simplify_expression_l2866_286631


namespace ceiling_tiling_count_l2866_286661

/-- Represents a rectangular region -/
structure Rectangle :=
  (length : ℕ)
  (width : ℕ)

/-- Represents a tile -/
structure Tile :=
  (length : ℕ)
  (width : ℕ)

/-- Counts the number of ways to tile a rectangle with given tiles -/
def count_tilings (r : Rectangle) (t : Tile) : ℕ :=
  sorry

/-- Counts the number of ways to tile a rectangle with a beam -/
def count_tilings_with_beam (r : Rectangle) (t : Tile) (beam_pos : ℕ) : ℕ :=
  sorry

theorem ceiling_tiling_count :
  let ceiling := Rectangle.mk 6 4
  let tile := Tile.mk 2 1
  let beam_pos := 2
  count_tilings_with_beam ceiling tile beam_pos = 180 :=
sorry

end ceiling_tiling_count_l2866_286661


namespace xyz_value_l2866_286670

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12)
  (h3 : (x + y + z)^2 = x^2 + y^2 + z^2 + 12) :
  x * y * z = 8 := by
  sorry

end xyz_value_l2866_286670


namespace smallest_number_l2866_286675

theorem smallest_number (a b c d : ℚ) (ha : a = 0) (hb : b = -2/3) (hc : c = 1) (hd : d = -3) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c := by
  sorry

end smallest_number_l2866_286675


namespace power_equality_l2866_286629

theorem power_equality (y : ℝ) (h : (10 : ℝ) ^ (4 * y) = 100) : (10 : ℝ) ^ (y / 2) = (10 : ℝ) ^ (1 / 4) := by
  sorry

end power_equality_l2866_286629


namespace max_reflections_theorem_l2866_286635

/-- The angle between two lines in degrees -/
def angle_between_lines : ℝ := 6

/-- The maximum number of reflections before perpendicular incidence -/
def max_reflections : ℕ := 15

/-- Theorem: Given the angle between two lines is 6°, the maximum number of reflections
    before perpendicular incidence is 15 -/
theorem max_reflections_theorem (angle : ℝ) (n : ℕ) 
  (h1 : angle = angle_between_lines)
  (h2 : n = max_reflections) :
  n * angle = 90 ∧ ∀ m : ℕ, m > n → m * angle > 90 := by
  sorry

#check max_reflections_theorem

end max_reflections_theorem_l2866_286635


namespace matthews_income_l2866_286606

/-- Represents the state income tax calculation function -/
def state_tax (q : ℝ) (income : ℝ) : ℝ :=
  0.01 * q * 50000 + 0.01 * (q + 3) * (income - 50000)

/-- Represents the condition that the total tax is (q + 0.5)% of the total income -/
def tax_condition (q : ℝ) (income : ℝ) : Prop :=
  state_tax q income = 0.01 * (q + 0.5) * income

/-- Theorem stating that given the tax calculation method and condition, 
    Matthew's annual income is $60000 -/
theorem matthews_income (q : ℝ) : 
  ∃ (income : ℝ), tax_condition q income ∧ income = 60000 := by
  sorry

end matthews_income_l2866_286606


namespace intersected_cubes_count_l2866_286658

/-- Represents a cube composed of unit cubes -/
structure LargeCube where
  size : ℕ
  total_units : ℕ

/-- Represents a plane intersecting the large cube -/
structure IntersectingPlane where
  perpendicular_to_diagonal : Bool
  bisects_diagonal : Bool

/-- Counts the number of unit cubes intersected by the plane -/
def count_intersected_cubes (cube : LargeCube) (plane : IntersectingPlane) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem intersected_cubes_count 
  (cube : LargeCube) 
  (plane : IntersectingPlane) 
  (h1 : cube.size = 4) 
  (h2 : cube.total_units = 64) 
  (h3 : plane.perpendicular_to_diagonal) 
  (h4 : plane.bisects_diagonal) : 
  count_intersected_cubes cube plane = 56 :=
sorry

end intersected_cubes_count_l2866_286658


namespace weekend_rain_probability_l2866_286691

theorem weekend_rain_probability
  (p_rain_saturday : ℝ)
  (p_rain_sunday : ℝ)
  (p_rain_sunday_given_no_saturday : ℝ)
  (h1 : p_rain_saturday = 0.6)
  (h2 : p_rain_sunday = 0.4)
  (h3 : p_rain_sunday_given_no_saturday = 0.7)
  : ℝ :=
by
  -- Probability of rain over the weekend
  sorry

#check weekend_rain_probability

end weekend_rain_probability_l2866_286691


namespace count_triples_eq_200_l2866_286697

/-- Counts the number of ways to partition a positive integer into two positive integers -/
def partitionCount (n : ℕ) : ℕ := if n ≤ 1 then 0 else n - 1

/-- Counts the number of ordered triples (a,b,c) satisfying the given conditions -/
def countTriples : ℕ :=
  (partitionCount 3) + (partitionCount 4) + (partitionCount 9) +
  (partitionCount 19) + (partitionCount 24) + (partitionCount 49) +
  (partitionCount 99)

theorem count_triples_eq_200 :
  countTriples = 200 :=
sorry

end count_triples_eq_200_l2866_286697


namespace quadratic_equation_propositions_l2866_286695

theorem quadratic_equation_propositions (a b : ℝ) : 
  ∃! (prop_a prop_b prop_c prop_d : Prop),
    prop_a = (1 ^ 2 + a * 1 + b = 0) ∧
    prop_b = (∃ x y : ℝ, x ^ 2 + a * x + b = 0 ∧ y ^ 2 + a * y + b = 0 ∧ x + y = 2) ∧
    prop_c = (3 ^ 2 + a * 3 + b = 0) ∧
    prop_d = (∃ x y : ℝ, x ^ 2 + a * x + b = 0 ∧ y ^ 2 + a * y + b = 0 ∧ x * y < 0) ∧
    (¬prop_a ∧ prop_b ∧ prop_c ∧ prop_d) ∨
    (prop_a ∧ ¬prop_b ∧ prop_c ∧ prop_d) ∨
    (prop_a ∧ prop_b ∧ ¬prop_c ∧ prop_d) ∨
    (prop_a ∧ prop_b ∧ prop_c ∧ ¬prop_d) :=
by
  sorry

end quadratic_equation_propositions_l2866_286695


namespace traffic_light_probability_l2866_286669

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  total_duration : ℕ
  red_duration : ℕ
  yellow_duration : ℕ
  green_duration : ℕ

/-- Calculates the probability of waiting no more than a given time -/
def probability_of_waiting (cycle : TrafficLightCycle) (max_wait : ℕ) : ℚ :=
  let proceed_duration := cycle.yellow_duration + cycle.green_duration
  let favorable_duration := min max_wait cycle.red_duration + proceed_duration
  favorable_duration / cycle.total_duration

/-- The main theorem to be proved -/
theorem traffic_light_probability (cycle : TrafficLightCycle) 
  (h1 : cycle.total_duration = 80)
  (h2 : cycle.red_duration = 40)
  (h3 : cycle.yellow_duration = 10)
  (h4 : cycle.green_duration = 30) :
  probability_of_waiting cycle 10 = 5/8 := by
  sorry

end traffic_light_probability_l2866_286669


namespace hot_chocolate_consumption_l2866_286650

/-- The number of cups of hot chocolate Tom can drink in 5 hours -/
def cups_in_five_hours : ℕ := 15

/-- The time interval between each cup of hot chocolate in minutes -/
def interval_minutes : ℕ := 20

/-- The total time in hours -/
def total_time_hours : ℕ := 5

/-- Theorem stating the number of cups Tom can drink in 5 hours -/
theorem hot_chocolate_consumption :
  cups_in_five_hours = (total_time_hours * 60) / interval_minutes :=
by sorry

end hot_chocolate_consumption_l2866_286650


namespace distance_between_walkers_l2866_286694

/-- Proves the distance between two people walking towards each other after a given time -/
theorem distance_between_walkers 
  (playground_length : ℝ) 
  (speed_hyosung : ℝ) 
  (speed_mimi : ℝ) 
  (time : ℝ) 
  (h1 : playground_length = 2.5)
  (h2 : speed_hyosung = 0.08)
  (h3 : speed_mimi = 2.4 / 60)
  (h4 : time = 15) :
  playground_length - (speed_hyosung + speed_mimi) * time = 0.7 := by
  sorry

end distance_between_walkers_l2866_286694


namespace x_range_for_given_equation_l2866_286622

theorem x_range_for_given_equation (x y : ℝ) :
  x - 4 * Real.sqrt y = 2 * Real.sqrt (x - y) →
  x = 0 ∨ (4 ≤ x ∧ x ≤ 20) := by
  sorry

end x_range_for_given_equation_l2866_286622


namespace sequence_ratio_l2866_286653

-- Define arithmetic sequence
def is_arithmetic_sequence (s : Fin 4 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : Fin 3, s (i + 1) - s i = d

-- Define geometric sequence
def is_geometric_sequence (s : Fin 5 → ℝ) : Prop :=
  ∃ r : ℝ, ∀ i : Fin 4, s (i + 1) / s i = r

theorem sequence_ratio :
  ∀ a₁ a₂ b₁ b₂ b₃ : ℝ,
  let s₁ : Fin 4 → ℝ := ![1, a₁, a₂, 9]
  let s₂ : Fin 5 → ℝ := ![1, b₁, b₂, b₃, 9]
  is_arithmetic_sequence s₁ →
  is_geometric_sequence s₂ →
  b₂ / (a₁ + a₂) = 3/10 :=
by sorry

end sequence_ratio_l2866_286653


namespace magnitude_of_complex_number_l2866_286665

theorem magnitude_of_complex_number : 
  Complex.abs ((1 + Complex.I)^2 / (1 - 2 * Complex.I)) = 2 * Real.sqrt 5 / 5 := by
  sorry

end magnitude_of_complex_number_l2866_286665


namespace newspaper_circulation_estimate_l2866_286699

/-- Estimated circulation of a newspaper given survey results -/
theorem newspaper_circulation_estimate 
  (city_population : ℕ) 
  (survey_size : ℕ) 
  (buyers_in_survey : ℕ) 
  (h1 : city_population = 8000000)
  (h2 : survey_size = 2500)
  (h3 : buyers_in_survey = 500) :
  (buyers_in_survey : ℚ) / survey_size * (city_population / 10000) = 160 := by
  sorry

#check newspaper_circulation_estimate

end newspaper_circulation_estimate_l2866_286699


namespace lateral_surface_area_cylinder_l2866_286633

/-- The lateral surface area of a cylinder with radius 1 and height 2 is 4π. -/
theorem lateral_surface_area_cylinder : 
  let r : ℝ := 1
  let h : ℝ := 2
  2 * Real.pi * r * h = 4 * Real.pi :=
by sorry

end lateral_surface_area_cylinder_l2866_286633


namespace fraction_simplification_l2866_286662

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) :
  (x + y) / (x - y) - (2 * y) / (x - y) = 1 := by
  sorry

end fraction_simplification_l2866_286662


namespace quadratic_equation_properties_l2866_286610

theorem quadratic_equation_properties (k m : ℝ) 
  (h : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (1 + 2*k^2)*x₁^2 - 4*k*m*x₁ + 2*m^2 - 2 = 0 
                   ∧ (1 + 2*k^2)*x₂^2 - 4*k*m*x₂ + 2*m^2 - 2 = 0) : 
  m^2 < 1 + 2*k^2 
  ∧ (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (1 + 2*k^2)*x₁^2 - 4*k*m*x₁ + 2*m^2 - 2 = 0 
                 → (1 + 2*k^2)*x₂^2 - 4*k*m*x₂ + 2*m^2 - 2 = 0 
                 → x₁*x₂ < 2)
  ∧ (∃ S : ℝ → ℝ, (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (1 + 2*k^2)*x₁^2 - 4*k*m*x₁ + 2*m^2 - 2 = 0 
                                       → (1 + 2*k^2)*x₂^2 - 4*k*m*x₂ + 2*m^2 - 2 = 0 
                                       → S m = |m| * Real.sqrt ((x₁ + x₂)^2 - 4*x₁*x₂))
     ∧ (∀ m : ℝ, S m ≤ Real.sqrt 2)
     ∧ (∃ m : ℝ, S m = Real.sqrt 2)) :=
by sorry

end quadratic_equation_properties_l2866_286610


namespace worker_y_fraction_l2866_286689

theorem worker_y_fraction (total_products : ℝ) (x_products y_products : ℝ) 
  (h1 : x_products + y_products = total_products)
  (h2 : 0.005 * x_products + 0.008 * y_products = 0.007 * total_products) :
  y_products / total_products = 2 / 3 :=
by sorry

end worker_y_fraction_l2866_286689


namespace profit_difference_exists_l2866_286604

/-- Represents the strategy of selling or renting a movie -/
inductive SaleStrategy
  | Forever
  | Rental

/-- Represents the economic factors affecting movie sales -/
structure EconomicFactors where
  price : ℝ
  customerBase : ℕ
  sharingRate : ℝ
  rentalFrequency : ℕ
  adminCosts : ℝ
  piracyRisk : ℝ

/-- Calculates the total profit for a given sale strategy and economic factors -/
def totalProfit (strategy : SaleStrategy) (factors : EconomicFactors) : ℝ :=
  sorry

/-- Theorem stating that the total profit from selling a movie "forever" 
    may be different from the total profit from temporary rentals -/
theorem profit_difference_exists :
  ∃ (f₁ f₂ : EconomicFactors), 
    totalProfit SaleStrategy.Forever f₁ ≠ totalProfit SaleStrategy.Rental f₂ :=
  sorry

end profit_difference_exists_l2866_286604


namespace weight_loss_problem_l2866_286684

theorem weight_loss_problem (x : ℝ) : 
  (x - 12 = 2 * (x - 7) - 80) → x = 82 := by
  sorry

end weight_loss_problem_l2866_286684


namespace base2_to_base4_conversion_l2866_286617

def base2_to_decimal (n : List Bool) : ℕ :=
  n.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

def decimal_to_base4 (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List (Fin 4) :=
      if m = 0 then [] else (m % 4) :: aux (m / 4)
    aux n |>.reverse

theorem base2_to_base4_conversion :
  let base2 : List Bool := [true, false, true, false, true, false, true, false, true]
  let base4 : List (Fin 4) := [1, 1, 1, 1, 1]
  decimal_to_base4 (base2_to_decimal base2) = base4 := by
  sorry

end base2_to_base4_conversion_l2866_286617


namespace k_range_for_three_elements_l2866_286666

def P (k : ℝ) : Set ℕ := {x : ℕ | 2 < x ∧ x < k}

theorem k_range_for_three_elements (k : ℝ) :
  (∃ (a b c : ℕ), P k = {a, b, c} ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c) →
  5 < k ∧ k ≤ 6 :=
by sorry

end k_range_for_three_elements_l2866_286666


namespace rice_and_grain_separation_l2866_286634

/-- Represents the amount of rice in dan -/
def total_rice : ℕ := 1536

/-- Represents the sample size in grains -/
def sample_size : ℕ := 256

/-- Represents the number of mixed grain in the sample -/
def mixed_grain_sample : ℕ := 18

/-- Calculates the amount of mixed grain in the entire batch -/
def mixed_grain_total : ℕ := total_rice * mixed_grain_sample / sample_size

theorem rice_and_grain_separation :
  mixed_grain_total = 108 := by
  sorry

end rice_and_grain_separation_l2866_286634


namespace quadratic_equation_m_range_l2866_286625

/-- Given a quadratic equation (m-1)x² + x + 1 = 0 with real roots, 
    prove that the range of m is m ≤ 5/4 and m ≠ 1 -/
theorem quadratic_equation_m_range (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + x + 1 = 0) → 
  (m ≤ 5/4 ∧ m ≠ 1) :=
by sorry

end quadratic_equation_m_range_l2866_286625


namespace coefficient_of_x_squared_l2866_286621

def polynomial (x : ℝ) : ℝ := 5*(x - x^4) - 4*(x^2 - 2*x^4 + x^6) + 3*(2*x^2 - x^8)

theorem coefficient_of_x_squared (x : ℝ) : 
  ∃ (a b c : ℝ), polynomial x = 2*x^2 + a*x + b*x^3 + c*x^4 + 
    (-5)*x^4 + 8*x^4 + (-4)*x^6 + (-3)*x^8 := by
  sorry

end coefficient_of_x_squared_l2866_286621


namespace unique_seq_largest_gt_100_l2866_286608

/-- A sequence of 9 positive integers with unique sums property -/
def UniqueSeq (a : Fin 9 → ℕ) : Prop :=
  (∀ i j, i < j → a i < a j) ∧
  (∀ s₁ s₂ : Finset (Fin 9), s₁ ≠ s₂ → s₁.sum a ≠ s₂.sum a)

/-- Theorem: In a sequence with unique sums property, the largest element is greater than 100 -/
theorem unique_seq_largest_gt_100 (a : Fin 9 → ℕ) (h : UniqueSeq a) : a 8 > 100 := by
  sorry

end unique_seq_largest_gt_100_l2866_286608


namespace ren_faire_amulet_sales_l2866_286602

/-- Represents the problem of calculating amulets sold per day at a Ren Faire --/
theorem ren_faire_amulet_sales (selling_price : ℕ) (cost_price : ℕ) (revenue_share : ℚ)
  (total_days : ℕ) (total_profit : ℕ) :
  selling_price = 40 →
  cost_price = 30 →
  revenue_share = 1/10 →
  total_days = 2 →
  total_profit = 300 →
  (selling_price - cost_price - (revenue_share * selling_price)) * total_days * 
    (total_profit / ((selling_price - cost_price - (revenue_share * selling_price)) * total_days)) = 25 :=
by sorry

end ren_faire_amulet_sales_l2866_286602


namespace dog_cleaner_amount_l2866_286619

/-- The amount of cleaner used for a cat stain in ounces -/
def cat_cleaner : ℝ := 4

/-- The amount of cleaner used for a rabbit stain in ounces -/
def rabbit_cleaner : ℝ := 1

/-- The total amount of cleaner used for all stains in ounces -/
def total_cleaner : ℝ := 49

/-- The number of dog stains -/
def num_dogs : ℕ := 6

/-- The number of cat stains -/
def num_cats : ℕ := 3

/-- The number of rabbit stains -/
def num_rabbits : ℕ := 1

/-- The amount of cleaner used for a dog stain in ounces -/
def dog_cleaner : ℝ := 6

theorem dog_cleaner_amount :
  num_dogs * dog_cleaner + num_cats * cat_cleaner + num_rabbits * rabbit_cleaner = total_cleaner :=
by sorry

end dog_cleaner_amount_l2866_286619


namespace polynomial_remainder_l2866_286686

def p (x : ℝ) : ℝ := 5*x^9 - 3*x^7 + 4*x^6 - 8*x^4 + 3*x^3 - 6*x + 5

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, p = λ x => (3*x - 6) * q x + 2321 :=
sorry

end polynomial_remainder_l2866_286686


namespace class_size_l2866_286627

/-- The number of girls in Jungkook's class -/
def num_girls : ℕ := 9

/-- The number of boys in Jungkook's class -/
def num_boys : ℕ := 16

/-- The total number of students in Jungkook's class -/
def total_students : ℕ := num_girls + num_boys

theorem class_size : total_students = 25 := by
  sorry

end class_size_l2866_286627


namespace fraction_problem_l2866_286643

theorem fraction_problem (N : ℝ) (f : ℝ) (h1 : N = 12) (h2 : 1 + f * N = 0.75 * N) : f = 2/3 := by
  sorry

end fraction_problem_l2866_286643


namespace inequality_proof_l2866_286685

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  a * b > a * c := by
  sorry

end inequality_proof_l2866_286685


namespace main_theorem_l2866_286615

/-- Definition of the function f --/
def f (a b k : ℤ) : ℤ := a * k^3 + b * k

/-- Definition of n-good --/
def is_n_good (a b n : ℤ) : Prop :=
  ∀ m k : ℤ, n ∣ (f a b k - f a b m) → n ∣ (k - m)

/-- Definition of very good --/
def is_very_good (a b : ℤ) : Prop :=
  ∀ n : ℤ, ∃ m : ℤ, m > n ∧ is_n_good a b m

/-- Main theorem --/
theorem main_theorem :
  (is_n_good 1 (-51^2) 51 ∧ ¬ is_very_good 1 (-51^2)) ∧
  (∀ a b : ℤ, is_n_good a b 2013 → is_very_good a b) := by
  sorry

end main_theorem_l2866_286615


namespace lemonade_pitchers_sum_l2866_286603

theorem lemonade_pitchers_sum : 
  let first_intermission : ℚ := 0.25
  let second_intermission : ℚ := 0.42
  let third_intermission : ℚ := 0.25
  first_intermission + second_intermission + third_intermission = 0.92 := by
sorry

end lemonade_pitchers_sum_l2866_286603


namespace sum_of_cubes_l2866_286648

theorem sum_of_cubes (x y z : ℝ) 
  (sum_eq : x + y + z = 7)
  (sum_prod_eq : x*y + x*z + y*z = 9)
  (prod_eq : x*y*z = -18) : 
  x^3 + y^3 + z^3 = 100 := by
  sorry

end sum_of_cubes_l2866_286648


namespace unique_special_number_l2866_286618

/-- A three-digit number ending with 2 that, when the 2 is moved to the front,
    results in a number 18 greater than the original. -/
def special_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  n % 10 = 2 ∧  -- ends with 2
  200 + (n / 10) = n + 18  -- moving 2 to front increases by 18

theorem unique_special_number :
  ∃! n : ℕ, special_number n ∧ n = 202 :=
sorry

end unique_special_number_l2866_286618


namespace min_value_expression_l2866_286667

theorem min_value_expression (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 10)
  (hb : 1 ≤ b ∧ b ≤ 10)
  (hc : 1 ≤ c ∧ c ≤ 10)
  (hbc : b < c) :
  4 ≤ 3*a - a*b + a*c ∧ ∃ (a' b' c' : ℕ), 
    1 ≤ a' ∧ a' ≤ 10 ∧
    1 ≤ b' ∧ b' ≤ 10 ∧
    1 ≤ c' ∧ c' ≤ 10 ∧
    b' < c' ∧
    3*a' - a'*b' + a'*c' = 4 :=
by sorry

end min_value_expression_l2866_286667


namespace expression_simplification_l2866_286632

theorem expression_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) (h3 : x ≠ 4) :
  (x^2 - 4) * ((x + 2) / (x^2 - 2*x) - (x - 1) / (x^2 - 4*x + 4)) / ((x - 4) / x) = (x + 2) / (x - 2) := by
  sorry

end expression_simplification_l2866_286632


namespace transformation_maps_curve_to_ellipse_l2866_286623

/-- The transformation that maps a curve to an ellipse -/
def transformation (x' y' : ℝ) : ℝ × ℝ :=
  (2 * x', y')

/-- The original curve equation -/
def original_curve (x y : ℝ) : Prop :=
  y^2 = 4

/-- The transformed ellipse equation -/
def transformed_ellipse (x' y' : ℝ) : Prop :=
  x'^2 + y'^2 / 4 = 1

/-- Theorem stating that the transformation maps the original curve to the ellipse -/
theorem transformation_maps_curve_to_ellipse :
  ∀ x' y', original_curve (transformation x' y').1 (transformation x' y').2 ↔ transformed_ellipse x' y' :=
sorry

end transformation_maps_curve_to_ellipse_l2866_286623


namespace sum_of_squares_l2866_286616

theorem sum_of_squares (x y z : ℤ) 
  (sum_eq : x + y + z = 3) 
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 := by
  sorry

end sum_of_squares_l2866_286616


namespace equation_rewrite_l2866_286656

theorem equation_rewrite (x y : ℝ) : 
  (2 * x - y = 4) → (y = 2 * x - 4) := by
  sorry

end equation_rewrite_l2866_286656


namespace ab_value_l2866_286644

theorem ab_value (a b c : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 27) 
  (h3 : a + b + c = 10) : 
  a * b = 9 := by
sorry

end ab_value_l2866_286644


namespace orthogonal_vectors_sum_magnitude_l2866_286680

/-- Prove that given planar vectors a and b, where a and b are orthogonal, 
    a = (-1, 1), and |b| = 1, |a + 2b| = √6. -/
theorem orthogonal_vectors_sum_magnitude 
  (a b : ℝ × ℝ) 
  (h_orthogonal : a.1 * b.1 + a.2 * b.2 = 0)
  (h_a : a = (-1, 1))
  (h_b_norm : Real.sqrt (b.1^2 + b.2^2) = 1) :
  Real.sqrt ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2) = Real.sqrt 6 := by
  sorry

end orthogonal_vectors_sum_magnitude_l2866_286680


namespace x_minus_y_equals_two_l2866_286638

theorem x_minus_y_equals_two (x y : ℝ) 
  (sum_eq : x + y = 6) 
  (diff_squares_eq : x^2 - y^2 = 12) : 
  x - y = 2 := by
sorry

end x_minus_y_equals_two_l2866_286638


namespace average_equation_l2866_286668

theorem average_equation (y : ℝ) : 
  (55 + 48 + 507 + 2 + 684 + y) / 6 = 223 → y = 42 := by
  sorry

end average_equation_l2866_286668


namespace arithmetic_sequence_sum_l2866_286612

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) 
  (h_arith : ArithmeticSequence a)
  (h_sum1 : a 2 + a 3 = 4)
  (h_sum2 : a 4 + a 5 = 6) :
  a 9 + a 10 = 11 := by
  sorry

end arithmetic_sequence_sum_l2866_286612


namespace time_interval_for_population_change_l2866_286664

/-- Proves that given the specified birth and death rates and net population increase,
    the time interval is 2 seconds. -/
theorem time_interval_for_population_change (t : ℝ) : 
  (5 : ℝ) / t - (3 : ℝ) / t > 0 →  -- Ensure positive net change
  (5 - 3) * (86400 / t) = 86400 →
  t = 2 := by
  sorry

end time_interval_for_population_change_l2866_286664


namespace remainder_98_pow_50_mod_50_l2866_286640

theorem remainder_98_pow_50_mod_50 : 98^50 % 50 = 0 := by
  sorry

end remainder_98_pow_50_mod_50_l2866_286640


namespace cabinet_ratio_proof_l2866_286696

/-- Proves the ratio of new cabinets per counter to initial cabinets is 2:1 --/
theorem cabinet_ratio_proof (initial_cabinets : ℕ) (total_cabinets : ℕ) (additional_cabinets : ℕ) 
  (h1 : initial_cabinets = 3)
  (h2 : total_cabinets = 26)
  (h3 : additional_cabinets = 5)
  : ∃ (new_cabinets_per_counter : ℕ), 
    initial_cabinets + 3 * new_cabinets_per_counter + additional_cabinets = total_cabinets ∧ 
    new_cabinets_per_counter = 2 * initial_cabinets :=
by
  sorry


end cabinet_ratio_proof_l2866_286696


namespace five_hour_pay_calculation_l2866_286641

/-- Represents the hourly pay rate in dollars -/
def hourly_rate (three_hour_pay six_hour_pay : ℚ) : ℚ :=
  three_hour_pay / 3

/-- Calculates the pay for a given number of hours -/
def calculate_pay (rate : ℚ) (hours : ℚ) : ℚ :=
  rate * hours

theorem five_hour_pay_calculation 
  (three_hour_pay six_hour_pay : ℚ) 
  (h1 : three_hour_pay = 24.75)
  (h2 : six_hour_pay = 49.50)
  (h3 : hourly_rate three_hour_pay six_hour_pay = hourly_rate three_hour_pay six_hour_pay) :
  calculate_pay (hourly_rate three_hour_pay six_hour_pay) 5 = 41.25 := by
  sorry

end five_hour_pay_calculation_l2866_286641


namespace function_properties_l2866_286663

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x

theorem function_properties :
  (∀ x ≠ 0, f x = x^2 + 1/x) →
  f 1 = 2 →
  (¬ (∀ x ≠ 0, f (-x) = f x) ∧ ¬ (∀ x ≠ 0, f (-x) = -f x)) ∧
  (∀ x y, 2 ≤ x ∧ x < y → f x < f y) :=
by sorry

end function_properties_l2866_286663


namespace cases_in_1990_l2866_286639

/-- Calculates the number of disease cases in a given year, assuming a linear decrease from 1960 to 2000 --/
def diseaseCases (year : ℕ) : ℕ :=
  let initialCases : ℕ := 600000
  let finalCases : ℕ := 600
  let initialYear : ℕ := 1960
  let finalYear : ℕ := 2000
  let totalYears : ℕ := finalYear - initialYear
  let yearlyDecrease : ℕ := (initialCases - finalCases) / totalYears
  initialCases - yearlyDecrease * (year - initialYear)

theorem cases_in_1990 :
  diseaseCases 1990 = 150450 := by
  sorry

end cases_in_1990_l2866_286639


namespace parabola_point_order_l2866_286611

/-- The parabola function -/
def f (x : ℝ) : ℝ := -2 * (x + 1)^2 - 1

/-- Point A -/
def A : ℝ × ℝ := (-3, f (-3))

/-- Point B -/
def B : ℝ × ℝ := (-2, f (-2))

/-- Point C -/
def C : ℝ × ℝ := (2, f 2)

theorem parabola_point_order :
  A.2 < B.2 ∧ C.2 < A.2 := by sorry

end parabola_point_order_l2866_286611


namespace increase_by_percentage_l2866_286693

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 80 → percentage = 150 → result = initial * (1 + percentage / 100) → result = 200 := by
  sorry

end increase_by_percentage_l2866_286693


namespace ratatouille_price_proof_l2866_286626

def ratatouille_problem (eggplant_weight : ℝ) (zucchini_weight : ℝ) 
  (tomato_price : ℝ) (tomato_weight : ℝ)
  (onion_price : ℝ) (onion_weight : ℝ)
  (basil_price : ℝ) (basil_weight : ℝ)
  (total_quarts : ℝ) (price_per_quart : ℝ) : Prop :=
  let total_weight := eggplant_weight + zucchini_weight
  let other_ingredients_cost := tomato_price * tomato_weight + 
                                onion_price * onion_weight + 
                                basil_price * basil_weight * 2
  let total_cost := total_quarts * price_per_quart
  let eggplant_zucchini_cost := total_cost - other_ingredients_cost
  let price_per_pound := eggplant_zucchini_cost / total_weight
  price_per_pound = 2

theorem ratatouille_price_proof :
  ratatouille_problem 5 4 3.5 4 1 3 2.5 1 4 10 := by
  sorry

end ratatouille_price_proof_l2866_286626


namespace min_reciprocal_sum_l2866_286681

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = 2 :=
by sorry

end min_reciprocal_sum_l2866_286681


namespace second_tract_width_l2866_286620

/-- Given two rectangular tracts of land, prove that the width of the second tract is 630 meters -/
theorem second_tract_width (length1 width1 length2 combined_area : ℝ)
  (h1 : length1 = 300)
  (h2 : width1 = 500)
  (h3 : length2 = 250)
  (h4 : combined_area = 307500)
  (h5 : combined_area = length1 * width1 + length2 * (combined_area - length1 * width1) / length2) :
  (combined_area - length1 * width1) / length2 = 630 := by
  sorry

end second_tract_width_l2866_286620


namespace min_abs_z_on_line_segment_l2866_286671

theorem min_abs_z_on_line_segment (z : ℂ) (h : Complex.abs (z - 6) + Complex.abs (z - Complex.I * 5) = 7) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs (w - 6) + Complex.abs (w - Complex.I * 5) = 7 ∧ Complex.abs w = 30 / 7 :=
by sorry

end min_abs_z_on_line_segment_l2866_286671


namespace min_colors_2016_board_l2866_286624

/-- A color assignment for a square board. -/
def ColorAssignment (n : ℕ) := Fin n → Fin n → ℕ

/-- Predicate for a valid coloring of a square board. -/
def ValidColoring (n k : ℕ) (c : ColorAssignment n) : Prop :=
  -- One diagonal is colored with the first color
  (∀ i, c i i = 0) ∧
  -- Symmetric cells have the same color
  (∀ i j, c i j = c j i) ∧
  -- Cells in the same row on different sides of the diagonal have different colors
  (∀ i j₁ j₂, i < j₁ ∧ j₂ < i → c i j₁ ≠ c i j₂)

/-- Theorem stating the minimum number of colors needed for a 2016 × 2016 board. -/
theorem min_colors_2016_board :
  (∃ (c : ColorAssignment 2016), ValidColoring 2016 11 c) ∧
  (∀ k < 11, ¬ ∃ (c : ColorAssignment 2016), ValidColoring 2016 k c) :=
sorry

end min_colors_2016_board_l2866_286624


namespace english_score_is_67_l2866_286630

def mathematics_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def biology_score : ℕ := 85
def average_marks : ℕ := 75
def total_subjects : ℕ := 5

def english_score : ℕ := average_marks * total_subjects - (mathematics_score + science_score + social_studies_score + biology_score)

theorem english_score_is_67 : english_score = 67 := by
  sorry

end english_score_is_67_l2866_286630


namespace two_correct_statements_l2866_286601

theorem two_correct_statements :
  let statement1 := ∀ a b : ℝ, (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0) → a + b = 0
  let statement2 := ∀ a : ℝ, -a < 0
  let statement3 := ∀ n : ℤ, n ≠ 0
  let statement4 := ∀ a b : ℝ, |a| > |b| → |a| > |b - 0|
  let statement5 := ∀ a : ℝ, a ≠ 0 → |a| > 0
  (¬statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ statement4 ∧ statement5) :=
by
  sorry

end two_correct_statements_l2866_286601


namespace characteristic_vector_of_g_sin_value_for_associated_function_l2866_286682

def associated_characteristic_vector (f : ℝ → ℝ) : ℝ × ℝ :=
  sorry

def associated_function (v : ℝ × ℝ) : ℝ → ℝ :=
  sorry

theorem characteristic_vector_of_g :
  let g : ℝ → ℝ := λ x => Real.sin (x + 5 * Real.pi / 6) - Real.sin (3 * Real.pi / 2 - x)
  associated_characteristic_vector g = (-Real.sqrt 3 / 2, 3 / 2) :=
sorry

theorem sin_value_for_associated_function :
  let f := associated_function (1, Real.sqrt 3)
  ∀ x, f x = 8 / 5 → x > -Real.pi / 3 → x < Real.pi / 6 →
    Real.sin x = (4 - 3 * Real.sqrt 3) / 10 :=
sorry

end characteristic_vector_of_g_sin_value_for_associated_function_l2866_286682


namespace probability_all_girls_chosen_l2866_286642

def total_members : ℕ := 15
def num_boys : ℕ := 8
def num_girls : ℕ := 7
def num_chosen : ℕ := 3

theorem probability_all_girls_chosen :
  (Nat.choose num_girls num_chosen : ℚ) / (Nat.choose total_members num_chosen) = 1 / 13 :=
by sorry

end probability_all_girls_chosen_l2866_286642


namespace square_area_7m_l2866_286647

theorem square_area_7m (side_length : ℝ) (area : ℝ) : 
  side_length = 7 → area = side_length ^ 2 → area = 49 := by
  sorry

end square_area_7m_l2866_286647


namespace max_value_of_g_l2866_286652

-- Define the function g(x)
def g (x : ℝ) : ℝ := 9*x - 2*x^3

-- State the theorem
theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 9 :=
sorry

end max_value_of_g_l2866_286652


namespace parabola_vertex_range_l2866_286676

/-- Represents a parabola with equation y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The vertex of a parabola -/
structure Vertex where
  s : ℝ
  t : ℝ

theorem parabola_vertex_range 
  (p : Parabola) 
  (v : Vertex) 
  (y₁ y₂ : ℝ)
  (h1 : p.a * (-2)^2 + p.b * (-2) + p.c = y₁)
  (h2 : p.a * 4^2 + p.b * 4 + p.c = y₂)
  (h3 : y₁ > y₂)
  (h4 : y₂ > v.t)
  : v.s > 1 ∧ v.s ≠ 4 := by
  sorry

end parabola_vertex_range_l2866_286676


namespace hidden_cave_inventory_sum_l2866_286649

/-- Converts a number from base 5 to base 10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The problem statement --/
theorem hidden_cave_inventory_sum : 
  let artifact := base5ToBase10 [3, 1, 2, 4]
  let sculpture := base5ToBase10 [1, 3, 4, 2]
  let coins := base5ToBase10 [3, 1, 2]
  artifact + sculpture + coins = 982 := by
sorry

end hidden_cave_inventory_sum_l2866_286649


namespace squares_sum_difference_l2866_286654

theorem squares_sum_difference : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 241 := by
  sorry

end squares_sum_difference_l2866_286654


namespace geometric_sequence_sum_l2866_286678

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 3 + a 5 = 7) →
  (a 5 + a 7 + a 9 = 28) →
  (a 9 + a 11 + a 13 = 112) := by
  sorry

end geometric_sequence_sum_l2866_286678


namespace value_of_E_l2866_286673

theorem value_of_E (a b c : ℝ) (h1 : a ≠ b) (h2 : a^2 * (b + c) = 2023) (h3 : b^2 * (c + a) = 2023) :
  c^2 * (a + b) = 2023 := by
sorry

end value_of_E_l2866_286673


namespace solve_equation_l2866_286690

theorem solve_equation (x : ℝ) : 3*x - 4*x + 7*x = 210 → x = 35 := by
  sorry

end solve_equation_l2866_286690


namespace gasoline_quantity_reduction_l2866_286660

/-- Proves that a 25% price increase and 10% budget increase results in a 12% reduction in quantity --/
theorem gasoline_quantity_reduction 
  (P : ℝ) -- Original price of gasoline
  (Q : ℝ) -- Original quantity of gasoline
  (h1 : P > 0) -- Assumption: Price is positive
  (h2 : Q > 0) -- Assumption: Quantity is positive
  : 
  let new_price := 1.25 * P -- 25% price increase
  let new_budget := 1.10 * (P * Q) -- 10% budget increase
  let new_quantity := new_budget / new_price -- New quantity calculation
  (1 - new_quantity / Q) * 100 = 12 -- Percentage reduction in quantity
  := by sorry

end gasoline_quantity_reduction_l2866_286660


namespace students_passed_both_tests_l2866_286683

theorem students_passed_both_tests 
  (total : Nat) 
  (passed_long_jump : Nat) 
  (passed_shot_put : Nat) 
  (failed_both : Nat) : 
  total = 50 → 
  passed_long_jump = 40 → 
  passed_shot_put = 31 → 
  failed_both = 4 → 
  ∃ (passed_both : Nat), 
    passed_both = 25 ∧ 
    total = passed_both + (passed_long_jump - passed_both) + (passed_shot_put - passed_both) + failed_both :=
by sorry

end students_passed_both_tests_l2866_286683


namespace no_lower_bound_l2866_286688

/-- An arithmetic sequence with three terms -/
structure ArithmeticSequence :=
  (a₁ : ℝ)
  (d : ℝ)

/-- The second term of the arithmetic sequence -/
def ArithmeticSequence.a₂ (seq : ArithmeticSequence) : ℝ := seq.a₁ + seq.d

/-- The third term of the arithmetic sequence -/
def ArithmeticSequence.a₃ (seq : ArithmeticSequence) : ℝ := seq.a₁ + 2 * seq.d

/-- The expression to be minimized -/
def expression (seq : ArithmeticSequence) : ℝ := 3 * seq.a₂ + 7 * seq.a₃

/-- The theorem stating that the expression has no lower bound -/
theorem no_lower_bound :
  ∀ (b : ℝ), ∃ (seq : ArithmeticSequence), seq.a₁ = 3 ∧ expression seq < b :=
sorry

end no_lower_bound_l2866_286688


namespace arithmetic_sequence_sum_12_l2866_286628

def arithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) - a n = 1

def geometricMean (x y z : ℚ) : Prop :=
  z * z = x * y

def sumOfTerms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_12 (a : ℕ → ℚ) :
  arithmeticSequence a →
  geometricMean (a 3) (a 11) (a 6) →
  sumOfTerms a 12 = 96 := by
sorry

end arithmetic_sequence_sum_12_l2866_286628


namespace pizzas_served_during_lunch_l2866_286605

theorem pizzas_served_during_lunch (total_pizzas dinner_pizzas lunch_pizzas : ℕ) : 
  total_pizzas = 15 → 
  dinner_pizzas = 6 → 
  lunch_pizzas = total_pizzas - dinner_pizzas → 
  lunch_pizzas = 9 := by
sorry

end pizzas_served_during_lunch_l2866_286605


namespace product_of_primes_summing_to_91_l2866_286645

theorem product_of_primes_summing_to_91 (p q : ℕ) : 
  Prime p → Prime q → p + q = 91 → p * q = 178 := by
  sorry

end product_of_primes_summing_to_91_l2866_286645


namespace lily_of_valley_cost_price_l2866_286698

/-- The cost price of a pot of lily of the valley -/
def cost_price : ℝ := 2.4

/-- The selling price of a pot of lily of the valley -/
def selling_price : ℝ := cost_price * 1.25

/-- The number of pots sold -/
def num_pots : ℕ := 150

/-- The total revenue from selling the pots -/
def total_revenue : ℝ := 450

theorem lily_of_valley_cost_price :
  cost_price = 2.4 ∧
  selling_price = cost_price * 1.25 ∧
  (num_pots : ℝ) * selling_price = total_revenue :=
sorry

end lily_of_valley_cost_price_l2866_286698


namespace unique_integer_point_implies_c_value_l2866_286607

/-- The x-coordinate of the first point -/
def x1 : ℚ := 22

/-- The y-coordinate of the first point -/
def y1 : ℚ := 38/3

/-- The y-coordinate of the second point -/
def y2 : ℚ := 53/3

/-- The number of integer points on the line segment -/
def num_integer_points : ℕ := 1

/-- The x-coordinate of the second point -/
def c : ℚ := 23

theorem unique_integer_point_implies_c_value :
  (∃! p : ℤ × ℤ, (x1 : ℚ) < p.1 ∧ p.1 < c ∧
    (p.2 : ℚ) = y1 + (y2 - y1) / (c - x1) * ((p.1 : ℚ) - x1)) →
  c = 23 := by sorry

end unique_integer_point_implies_c_value_l2866_286607


namespace problem_solution_l2866_286609

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = 3 * t + 6) 
  (h3 : x = -6) : 
  y = 19.5 := by
sorry

end problem_solution_l2866_286609


namespace ratio_equality_l2866_286672

theorem ratio_equality (x y : ℝ) (h1 : 3 * x = 5 * y) (h2 : x ≠ 0) (h3 : y ≠ 0) :
  x / y = 5 / 3 := by sorry

end ratio_equality_l2866_286672


namespace jose_land_share_l2866_286637

def total_land_area : ℝ := 20000
def num_siblings : ℕ := 4

theorem jose_land_share :
  let total_people := num_siblings + 1
  let share := total_land_area / total_people
  share = 4000 := by
  sorry

end jose_land_share_l2866_286637


namespace perpendicular_tangents_range_l2866_286651

open Real

theorem perpendicular_tangents_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    let f : ℝ → ℝ := λ x => a * x + sin x + cos x
    let f' : ℝ → ℝ := λ x => a + cos x - sin x
    (f' x₁) * (f' x₂) = -1) →
  -1 ≤ a ∧ a ≤ 1 := by
sorry

end perpendicular_tangents_range_l2866_286651


namespace smallest_m_is_correct_l2866_286674

/-- The smallest positive value of m for which 15x^2 - mx + 630 = 0 has integral solutions -/
def smallest_m : ℕ := 195

/-- The equation 15x^2 - mx + 630 = 0 has integral solutions -/
def has_integral_solutions (m : ℕ) : Prop :=
  ∃ x : ℤ, 15 * x^2 - m * x + 630 = 0

/-- The main theorem: smallest_m is the smallest positive value of m for which
    the equation 15x^2 - mx + 630 = 0 has integral solutions -/
theorem smallest_m_is_correct :
  has_integral_solutions smallest_m ∧
  ∀ m : ℕ, 0 < m → m < smallest_m → ¬(has_integral_solutions m) :=
sorry

end smallest_m_is_correct_l2866_286674


namespace range_of_s_l2866_286636

-- Define a composite positive integer not divisible by 3
def IsCompositeNotDivisibleBy3 (n : ℕ) : Prop :=
  n > 1 ∧ ¬ (∃ k : ℕ, n = k * k) ∧ ¬ (3 ∣ n)

-- Define the function s
def s (n : ℕ) (h : IsCompositeNotDivisibleBy3 n) : ℕ :=
  sorry -- Implementation of s is not required for the statement

-- The main theorem
theorem range_of_s :
  ∀ m : ℤ, m > 3 ↔ ∃ (n : ℕ) (h : IsCompositeNotDivisibleBy3 n), s n h = m :=
sorry

end range_of_s_l2866_286636


namespace upper_bound_of_expression_l2866_286600

theorem upper_bound_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  -1/(2*a) - 2/b ≤ -9/2 ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 1 ∧ -1/(2*a₀) - 2/b₀ = -9/2 :=
by sorry

end upper_bound_of_expression_l2866_286600


namespace hapok_guarantee_l2866_286657

/-- Represents the coin division game between Hapok and Glazok -/
structure CoinGame where
  total_coins : ℕ
  max_handfuls : ℕ

/-- Represents a strategy for Hapok -/
def HapokStrategy := ℕ → ℕ

/-- Represents a strategy for Glazok -/
def GlazokStrategy := ℕ → Bool

/-- The outcome of the game given strategies for both players -/
def gameOutcome (game : CoinGame) (hapok_strat : HapokStrategy) (glazok_strat : GlazokStrategy) : ℕ := sorry

/-- Hapok's guaranteed minimum coins -/
def hapokGuaranteedCoins (game : CoinGame) : ℕ := sorry

/-- The main theorem stating Hapok can guarantee at least 46 coins -/
theorem hapok_guarantee (game : CoinGame) (h1 : game.total_coins = 100) (h2 : game.max_handfuls = 9) :
  hapokGuaranteedCoins game ≥ 46 := sorry

end hapok_guarantee_l2866_286657


namespace no_real_solutions_l2866_286655

theorem no_real_solutions : ∀ x : ℝ, (x^3 - 8) / (x - 2) ≠ 3*x :=
sorry

end no_real_solutions_l2866_286655


namespace turtles_on_log_l2866_286646

theorem turtles_on_log (initial_turtles : ℕ) : initial_turtles = 50 → 226 = initial_turtles + (7 * initial_turtles - 6) - (3 * (initial_turtles + (7 * initial_turtles - 6)) / 7) := by
  sorry

end turtles_on_log_l2866_286646
