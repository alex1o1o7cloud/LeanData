import Mathlib

namespace NUMINAMATH_CALUDE_monotonic_increase_interval_l39_3968

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem monotonic_increase_interval (x : ℝ) :
  StrictMonoOn f (Set.Ici 2) :=
sorry

end NUMINAMATH_CALUDE_monotonic_increase_interval_l39_3968


namespace NUMINAMATH_CALUDE_parallelogram_count_l39_3927

/-- 
Given an equilateral triangle ABC where each side is divided into n equal parts
and lines are drawn parallel to each side through these division points,
the total number of parallelograms formed is 3 * (n+1)^2 * n^2 / 4.
-/
theorem parallelogram_count (n : ℕ) : 
  (3 : ℚ) * (n + 1)^2 * n^2 / 4 = 3 * Nat.choose (n + 2) 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_count_l39_3927


namespace NUMINAMATH_CALUDE_parallel_transitivity_counterexample_l39_3911

-- Define the types for lines and planes in 3D space
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Define the subset relation for a line being in a plane
variable (subset : Line → Plane → Prop)

theorem parallel_transitivity_counterexample 
  (m n : Line) (α β : Plane) :
  ¬(∀ (m n : Line) (α β : Plane), 
    parallel m n → 
    parallel_line_plane n α → 
    parallel_plane α β → 
    parallel_line_plane m β) :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_counterexample_l39_3911


namespace NUMINAMATH_CALUDE_sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l39_3934

theorem sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds :
  7 < Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) ∧
  Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) < 8 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l39_3934


namespace NUMINAMATH_CALUDE_at_least_one_women_pair_probability_l39_3964

/-- The number of young men in the group -/
def num_men : ℕ := 5

/-- The number of young women in the group -/
def num_women : ℕ := 5

/-- The total number of people in the group -/
def total_people : ℕ := num_men + num_women

/-- The number of pairs formed -/
def num_pairs : ℕ := total_people / 2

/-- The total number of ways to form pairs -/
def total_pairings : ℕ := (total_people.factorial) / ((2^num_pairs) * num_pairs.factorial)

/-- The number of ways to form pairs with at least one pair of two women -/
def favorable_pairings : ℕ := total_pairings - num_pairs.factorial

/-- The probability of at least one pair consisting of two young women -/
def probability : ℚ := favorable_pairings / total_pairings

theorem at_least_one_women_pair_probability :
  probability = 825 / 945 :=
sorry

end NUMINAMATH_CALUDE_at_least_one_women_pair_probability_l39_3964


namespace NUMINAMATH_CALUDE_smaller_angle_at_8_is_120_l39_3940

/-- The number of hour marks on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full circle -/
def full_circle_degrees : ℕ := 360

/-- The hour being considered (8 o'clock) -/
def current_hour : ℕ := 8

/-- The angle between adjacent hour marks on a clock face -/
def angle_between_hours : ℚ := full_circle_degrees / clock_hours

/-- The position of the hour hand at the current hour -/
def hour_hand_position : ℚ := current_hour * angle_between_hours

/-- The smaller angle between clock hands at the given hour -/
def smaller_angle_at_hour (h : ℕ) : ℚ :=
  min (h * angle_between_hours) (full_circle_degrees - h * angle_between_hours)

theorem smaller_angle_at_8_is_120 :
  smaller_angle_at_hour current_hour = 120 :=
sorry

end NUMINAMATH_CALUDE_smaller_angle_at_8_is_120_l39_3940


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l39_3932

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 4 = 0 → x₂^2 - 3*x₂ - 4 = 0 → x₁^2 - 4*x₁ - x₂ + 2*x₁*x₂ = -7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l39_3932


namespace NUMINAMATH_CALUDE_max_min_product_l39_3989

theorem max_min_product (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 12) (h_prod_sum : a * b + b * c + c * a = 35) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 3 ∧
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l39_3989


namespace NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l39_3952

theorem polynomial_sum_of_coefficients 
  (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ x : ℝ, x^5 + 2 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) : 
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l39_3952


namespace NUMINAMATH_CALUDE_second_supply_cost_l39_3967

def first_supply_cost : ℕ := 13
def total_budget : ℕ := 56
def remaining_budget : ℕ := 19

theorem second_supply_cost :
  total_budget - remaining_budget - first_supply_cost = 24 :=
by sorry

end NUMINAMATH_CALUDE_second_supply_cost_l39_3967


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l39_3958

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ n : ℕ+, (n : ℕ) = 210 ∧ 
  (∀ m : ℕ+, m < n → ¬(∃ p q r s : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧ 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧ 
  (m : ℕ) % p = 0 ∧ (m : ℕ) % q = 0 ∧ (m : ℕ) % r = 0 ∧ (m : ℕ) % s = 0)) ∧
  (∃ p q r s : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧ 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧ 
  210 % p = 0 ∧ 210 % q = 0 ∧ 210 % r = 0 ∧ 210 % s = 0) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l39_3958


namespace NUMINAMATH_CALUDE_catherine_friends_count_l39_3981

def total_bottle_caps : ℕ := 18
def caps_per_friend : ℕ := 3

theorem catherine_friends_count : 
  total_bottle_caps / caps_per_friend = 6 := by sorry

end NUMINAMATH_CALUDE_catherine_friends_count_l39_3981


namespace NUMINAMATH_CALUDE_no_three_naturals_with_pairwise_sums_as_power_of_three_l39_3950

theorem no_three_naturals_with_pairwise_sums_as_power_of_three :
  ¬ ∃ (a b c : ℕ), 
    (∃ m : ℕ, a + b = 3^m) ∧ 
    (∃ n : ℕ, b + c = 3^n) ∧ 
    (∃ p : ℕ, c + a = 3^p) :=
sorry

end NUMINAMATH_CALUDE_no_three_naturals_with_pairwise_sums_as_power_of_three_l39_3950


namespace NUMINAMATH_CALUDE_bus_interval_theorem_l39_3972

/-- The interval between buses on a circular route -/
def interval (num_buses : ℕ) (total_time : ℕ) : ℕ :=
  total_time / num_buses

/-- The theorem stating the relationship between intervals for 2 and 3 buses -/
theorem bus_interval_theorem (initial_interval : ℕ) :
  initial_interval = 21 →
  interval 2 (2 * initial_interval) = 21 →
  interval 3 (2 * initial_interval) = 14 :=
by
  sorry

#check bus_interval_theorem

end NUMINAMATH_CALUDE_bus_interval_theorem_l39_3972


namespace NUMINAMATH_CALUDE_group_average_calculation_l39_3910

theorem group_average_calculation (initial_group_size : ℕ) 
  (new_member_amount : ℚ) (new_average : ℚ) : 
  initial_group_size = 7 → 
  new_member_amount = 56 → 
  new_average = 20 → 
  (initial_group_size * new_average + new_member_amount) / (initial_group_size + 1) = new_average → 
  new_average = 20 := by
  sorry

end NUMINAMATH_CALUDE_group_average_calculation_l39_3910


namespace NUMINAMATH_CALUDE_abc_inequality_l39_3920

theorem abc_inequality (a b c t : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c * (a^t + b^t + c^t) ≥ 
  a^(t+2) * (-a + b + c) + b^(t+2) * (a - b + c) + c^(t+2) * (a + b - c) ∧
  (a * b * c * (a^t + b^t + c^t) = 
   a^(t+2) * (-a + b + c) + b^(t+2) * (a - b + c) + c^(t+2) * (a + b - c) ↔ 
   a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l39_3920


namespace NUMINAMATH_CALUDE_equation_solution_l39_3906

theorem equation_solution :
  ∀ y : ℝ, (((36 * y + (36 * y + 55) ^ (1/3)) ^ (1/4)) = 11) → y = 7315/18 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l39_3906


namespace NUMINAMATH_CALUDE_unique_products_count_l39_3982

def set_a : Finset ℕ := {2, 3, 5, 7, 11}
def set_b : Finset ℕ := {2, 4, 6, 19}

theorem unique_products_count : 
  Finset.card ((set_a.product set_b).image (λ (x : ℕ × ℕ) => x.1 * x.2)) = 19 := by
  sorry

end NUMINAMATH_CALUDE_unique_products_count_l39_3982


namespace NUMINAMATH_CALUDE_subset_implies_membership_condition_l39_3933

theorem subset_implies_membership_condition (A B : Set α) (h : A ⊆ B) :
  ∀ x, x ∈ A → x ∈ B := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_membership_condition_l39_3933


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l39_3983

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) = (a^2 - 4*a + 3) + Complex.I * (a - 1)) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l39_3983


namespace NUMINAMATH_CALUDE_mantou_distribution_theorem_l39_3926

/-- Represents the distribution of mantou among monks -/
structure MantouDistribution where
  bigMonks : ℕ
  smallMonks : ℕ
  totalMonks : ℕ
  totalMantou : ℕ

/-- The mantou distribution satisfies the problem conditions -/
def isValidDistribution (d : MantouDistribution) : Prop :=
  d.bigMonks + d.smallMonks = d.totalMonks ∧
  d.totalMonks = 100 ∧
  d.totalMantou = 100 ∧
  3 * d.bigMonks + (1/3) * d.smallMonks = d.totalMantou

/-- The system of equations correctly represents the mantou distribution -/
theorem mantou_distribution_theorem (d : MantouDistribution) :
  isValidDistribution d ↔
  d.bigMonks + d.smallMonks = 100 ∧
  3 * d.bigMonks + (1/3) * d.smallMonks = 100 :=
sorry

end NUMINAMATH_CALUDE_mantou_distribution_theorem_l39_3926


namespace NUMINAMATH_CALUDE_range_of_a_l39_3953

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1 - 2*a

def g (a : ℝ) (x : ℝ) : ℝ := |x - a| - a*x

def has_two_distinct_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ f a x₁ = 0 ∧ f a x₂ = 0

def has_minimum_value (a : ℝ) : Prop :=
  ∃ x₀, ∀ x, g a x₀ ≤ g a x

theorem range_of_a (a : ℝ) :
  a > 0 ∧ ¬(has_two_distinct_intersections a) ∧ has_minimum_value a →
  a ∈ Set.Ioo 0 (Real.sqrt 2 - 1) ∪ Set.Ioo (1/2) 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l39_3953


namespace NUMINAMATH_CALUDE_function_upper_bound_condition_l39_3948

theorem function_upper_bound_condition (a : ℝ) (h_a : a > 0) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → a * x - x^2 ≤ 1) ↔ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_function_upper_bound_condition_l39_3948


namespace NUMINAMATH_CALUDE_odd_function_sum_zero_l39_3999

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem odd_function_sum_zero (f : ℝ → ℝ) (h : OddFunction f) :
  f (-2012) + f (-2011) + f 0 + f 2011 + f 2012 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_zero_l39_3999


namespace NUMINAMATH_CALUDE_stamp_reorganization_l39_3977

/-- Represents the stamp reorganization problem --/
theorem stamp_reorganization (
  initial_books : Nat)
  (pages_per_book : Nat)
  (initial_stamps_per_page : Nat)
  (new_stamps_per_page : Nat)
  (filled_books : Nat)
  (filled_pages_in_last_book : Nat)
  (h1 : initial_books = 10)
  (h2 : pages_per_book = 36)
  (h3 : initial_stamps_per_page = 5)
  (h4 : new_stamps_per_page = 8)
  (h5 : filled_books = 7)
  (h6 : filled_pages_in_last_book = 28) :
  (initial_books * pages_per_book * initial_stamps_per_page) -
  (filled_books * pages_per_book + filled_pages_in_last_book) * new_stamps_per_page = 8 := by
  sorry

#check stamp_reorganization

end NUMINAMATH_CALUDE_stamp_reorganization_l39_3977


namespace NUMINAMATH_CALUDE_triangle_side_length_l39_3969

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a = 2)
  (h2 : A = π / 6)  -- 30° in radians
  (h3 : C = 3 * π / 4)  -- 135° in radians
  (h4 : A + B + C = π)  -- sum of angles in a triangle
  (h5 : a / Real.sin A = b / Real.sin B)  -- Law of Sines
  : b = (Real.sqrt 2 - Real.sqrt 6) / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l39_3969


namespace NUMINAMATH_CALUDE_vector_equality_l39_3914

/-- Given vectors a, b, and c in R², prove that c = a - 3b -/
theorem vector_equality (a b c : Fin 2 → ℝ) 
  (ha : a = ![1, 1]) 
  (hb : b = ![1, -1]) 
  (hc : c = ![-2, 4]) : 
  c = a - 3 • b := by sorry

end NUMINAMATH_CALUDE_vector_equality_l39_3914


namespace NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_110_l39_3938

theorem greatest_common_multiple_9_15_under_110 : ∃ n : ℕ, 
  (∀ m : ℕ, m < 110 ∧ 9 ∣ m ∧ 15 ∣ m → m ≤ n) ∧
  n < 110 ∧ 9 ∣ n ∧ 15 ∣ n ∧
  n = 90 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_110_l39_3938


namespace NUMINAMATH_CALUDE_molar_ratio_h2_ch4_l39_3915

/-- Represents the heat of reaction for H₂ combustion in kJ/mol -/
def heat_h2 : ℝ := -571.6

/-- Represents the heat of reaction for CH₄ combustion in kJ/mol -/
def heat_ch4 : ℝ := -890

/-- Represents the volume of the gas mixture in liters -/
def mixture_volume : ℝ := 112

/-- Represents the molar volume of gas under standard conditions in L/mol -/
def molar_volume : ℝ := 22.4

/-- Represents the total heat released in kJ -/
def total_heat_released : ℝ := 3695

/-- Theorem stating that the molar ratio of H₂ to CH₄ in the original mixture is 1:3 -/
theorem molar_ratio_h2_ch4 :
  ∃ (x y : ℝ),
    x + y = mixture_volume / molar_volume ∧
    (heat_h2 / 2) * x + heat_ch4 * y = total_heat_released ∧
    x / y = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_molar_ratio_h2_ch4_l39_3915


namespace NUMINAMATH_CALUDE_ten_digit_number_exists_l39_3937

def product_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * product_of_digits (n / 10)

theorem ten_digit_number_exists : ∃ n : ℕ, 
  1000000000 ≤ n ∧ n < 10000000000 ∧ 
  (∀ d, d ∣ n → d ≠ 0) ∧
  product_of_digits (n + product_of_digits n) = product_of_digits n :=
sorry

end NUMINAMATH_CALUDE_ten_digit_number_exists_l39_3937


namespace NUMINAMATH_CALUDE_monotonically_decreasing_interval_of_f_l39_3902

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

theorem monotonically_decreasing_interval_of_f :
  ∀ x ∈ Set.Ioo (0 : ℝ) (Real.sqrt 2 / 2),
    ∀ y ∈ Set.Ioo (0 : ℝ) (Real.sqrt 2 / 2),
      x < y → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_interval_of_f_l39_3902


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l39_3966

theorem cube_volume_ratio :
  let cube1_edge : ℚ := 8
  let cube2_edge : ℚ := 16
  let volume_ratio := (cube1_edge ^ 3) / (cube2_edge ^ 3)
  volume_ratio = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l39_3966


namespace NUMINAMATH_CALUDE_sin_585_degrees_l39_3993

theorem sin_585_degrees : Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l39_3993


namespace NUMINAMATH_CALUDE_equation_solutions_l39_3917

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 6*x = -1 ↔ x = 3 - 2*Real.sqrt 2 ∨ x = 3 + 2*Real.sqrt 2) ∧
  (∀ x : ℝ, x*(2*x - 1) = 2*(2*x - 1) ↔ x = 1/2 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l39_3917


namespace NUMINAMATH_CALUDE_toad_ratio_proof_l39_3922

/-- Proves that the ratio of Sarah's toads to Jim's toads is 2 --/
theorem toad_ratio_proof (tim_toads jim_toads sarah_toads : ℕ) : 
  tim_toads = 30 →
  jim_toads = tim_toads + 20 →
  sarah_toads = 100 →
  sarah_toads / jim_toads = 2 := by
sorry

end NUMINAMATH_CALUDE_toad_ratio_proof_l39_3922


namespace NUMINAMATH_CALUDE_line_passes_through_intercepts_l39_3941

/-- A line that intersects the x-axis at (3, 0) and the y-axis at (0, -5) -/
def line_equation (x y : ℝ) : Prop :=
  x / 3 - y / 5 = 1

/-- The x-intercept of the line -/
def x_intercept : ℝ := 3

/-- The y-intercept of the line -/
def y_intercept : ℝ := -5

theorem line_passes_through_intercepts :
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_intercepts_l39_3941


namespace NUMINAMATH_CALUDE_shelter_dogs_l39_3980

theorem shelter_dogs (cat_count : ℕ) (cat_ratio : ℕ) (dog_ratio : ℕ) : 
  cat_count = 21 → cat_ratio = 7 → dog_ratio = 5 → 
  (cat_count * dog_ratio) / cat_ratio = 15 :=
by sorry

end NUMINAMATH_CALUDE_shelter_dogs_l39_3980


namespace NUMINAMATH_CALUDE_marbles_won_l39_3903

theorem marbles_won (initial : ℕ) (lost : ℕ) (final : ℕ) : 
  initial = 57 → lost = 18 → final = 64 → final - (initial - lost) = 25 := by
  sorry

end NUMINAMATH_CALUDE_marbles_won_l39_3903


namespace NUMINAMATH_CALUDE_last_locker_opened_l39_3992

/-- Represents the locker opening pattern described in the problem -/
def lockerOpeningPattern (n : ℕ) : ℕ → Prop :=
  sorry

/-- The number of lockers -/
def totalLockers : ℕ := 2048

/-- Theorem stating that the last locker opened is number 2046 -/
theorem last_locker_opened :
  ∃ (last : ℕ), last = 2046 ∧ 
  (∀ (k : ℕ), k ≤ totalLockers → lockerOpeningPattern totalLockers k → k ≤ last) ∧
  lockerOpeningPattern totalLockers last :=
sorry

end NUMINAMATH_CALUDE_last_locker_opened_l39_3992


namespace NUMINAMATH_CALUDE_price_reduction_achieves_profit_l39_3955

/-- Represents the daily sales and profit scenario of a store --/
structure StoreSales where
  initial_sales : ℕ := 20
  initial_profit_per_item : ℝ := 40
  sales_increase_rate : ℝ := 2
  min_profit_per_item : ℝ := 25

/-- Calculates the daily sales after a price reduction --/
def daily_sales (s : StoreSales) (price_reduction : ℝ) : ℝ :=
  s.initial_sales + s.sales_increase_rate * price_reduction

/-- Calculates the profit per item after a price reduction --/
def profit_per_item (s : StoreSales) (price_reduction : ℝ) : ℝ :=
  s.initial_profit_per_item - price_reduction

/-- Calculates the total daily profit after a price reduction --/
def daily_profit (s : StoreSales) (price_reduction : ℝ) : ℝ :=
  (daily_sales s price_reduction) * (profit_per_item s price_reduction)

/-- Theorem stating that a price reduction of 10 achieves the desired profit --/
theorem price_reduction_achieves_profit (s : StoreSales) :
  daily_profit s 10 = 1200 ∧ profit_per_item s 10 ≥ s.min_profit_per_item := by
  sorry


end NUMINAMATH_CALUDE_price_reduction_achieves_profit_l39_3955


namespace NUMINAMATH_CALUDE_infinite_primes_quadratic_equation_l39_3991

theorem infinite_primes_quadratic_equation :
  ∀ (S : Finset Nat), ∃ (p : Nat), Prime p ∧ p ∉ S ∧ ∃ (x y : ℤ), x^2 + x + 1 = p * y := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_quadratic_equation_l39_3991


namespace NUMINAMATH_CALUDE_real_y_condition_l39_3985

theorem real_y_condition (x : ℝ) : 
  (∃ y : ℝ, 3 * y^2 + 6 * x * y + 2 * x + 2 = 0) ↔ 
  (x ≤ (1 - Real.sqrt 7) / 3 ∨ x ≥ (1 + Real.sqrt 7) / 3) :=
by sorry

end NUMINAMATH_CALUDE_real_y_condition_l39_3985


namespace NUMINAMATH_CALUDE_back_seat_holds_eight_l39_3907

/-- Represents the seating capacity of a bus with specific arrangements --/
structure BusSeating where
  left_seats : Nat
  right_seats : Nat
  people_per_seat : Nat
  total_capacity : Nat

/-- Calculates the number of people that can be seated at the back of the bus --/
def back_seat_capacity (bus : BusSeating) : Nat :=
  bus.total_capacity - (bus.left_seats + bus.right_seats) * bus.people_per_seat

/-- Theorem stating that for the given bus configuration, the back seat can hold 8 people --/
theorem back_seat_holds_eight :
  let bus : BusSeating := {
    left_seats := 15,
    right_seats := 12,
    people_per_seat := 3,
    total_capacity := 89
  }
  back_seat_capacity bus = 8 := by
  sorry

#eval back_seat_capacity {
  left_seats := 15,
  right_seats := 12,
  people_per_seat := 3,
  total_capacity := 89
}

end NUMINAMATH_CALUDE_back_seat_holds_eight_l39_3907


namespace NUMINAMATH_CALUDE_alice_ball_drawing_l39_3925

/-- The number of balls in the bin -/
def n : ℕ := 20

/-- The number of draws -/
def k : ℕ := 4

/-- The number of possible lists when drawing k balls from n balls with replacement -/
def num_possible_lists (n k : ℕ) : ℕ := n ^ k

theorem alice_ball_drawing :
  num_possible_lists n k = 160000 := by
  sorry

end NUMINAMATH_CALUDE_alice_ball_drawing_l39_3925


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l39_3986

theorem rectangle_circle_area_ratio (w : ℝ) (r : ℝ) (h1 : w > 0) (h2 : r > 0) :
  let l := 2 * w
  let rectangle_perimeter := 2 * l + 2 * w
  let circle_circumference := 2 * Real.pi * r
  rectangle_perimeter = circle_circumference →
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
    sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l39_3986


namespace NUMINAMATH_CALUDE_seashells_given_proof_l39_3971

def seashells_given_to_jessica (initial_seashells remaining_seashells : ℝ) : ℝ :=
  initial_seashells - remaining_seashells

theorem seashells_given_proof (initial_seashells remaining_seashells : ℝ) 
  (h1 : initial_seashells = 62.5) 
  (h2 : remaining_seashells = 30.75) : 
  seashells_given_to_jessica initial_seashells remaining_seashells = 31.75 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_proof_l39_3971


namespace NUMINAMATH_CALUDE_last_digit_of_sum_l39_3946

theorem last_digit_of_sum (n : ℕ) : 
  (54^2019 + 28^2021) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_sum_l39_3946


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l39_3944

theorem sum_of_solutions_is_zero (y : ℝ) (x₁ x₂ : ℝ) : 
  y = 10 → 
  x₁^2 + y^2 = 200 → 
  x₂^2 + y^2 = 200 → 
  x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l39_3944


namespace NUMINAMATH_CALUDE_dvd_pack_cost_l39_3987

theorem dvd_pack_cost (total_amount : ℕ) (num_packs : ℕ) (cost_per_pack : ℕ) :
  total_amount = 104 →
  num_packs = 4 →
  cost_per_pack = total_amount / num_packs →
  cost_per_pack = 26 := by
  sorry

end NUMINAMATH_CALUDE_dvd_pack_cost_l39_3987


namespace NUMINAMATH_CALUDE_P_no_real_roots_l39_3970

/-- Recursive definition of the polynomial sequence P_n(x) -/
def P (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => λ _ => 1
  | n + 1 => λ x => x^(11*(n+1)) - P n x

/-- Theorem stating that P_n(x) has no real roots for all n ≥ 0 -/
theorem P_no_real_roots (n : ℕ) : ∀ x : ℝ, P n x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_P_no_real_roots_l39_3970


namespace NUMINAMATH_CALUDE_soccer_league_female_fraction_l39_3994

theorem soccer_league_female_fraction :
  -- Last year's male participants
  ∀ (male_last_year : ℕ),
  male_last_year = 30 →
  -- Male increase rate
  ∀ (male_increase_rate : ℝ),
  male_increase_rate = 0.1 →
  -- Female increase rate
  ∀ (female_increase_rate : ℝ),
  female_increase_rate = 0.25 →
  -- Overall increase rate
  ∀ (total_increase_rate : ℝ),
  total_increase_rate = 0.1 →
  -- This year's female participants fraction
  ∃ (female_fraction : ℚ),
  female_fraction = 50 / 83 :=
by sorry

end NUMINAMATH_CALUDE_soccer_league_female_fraction_l39_3994


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l39_3973

theorem not_necessarily_right_triangle (A B C : ℝ) : 
  A + B + C = 180 → A = B → A = 2 * C → ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
by sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l39_3973


namespace NUMINAMATH_CALUDE_decimal_rep_1_13_150th_digit_l39_3957

/-- The decimal representation of 1/13 as a sequence of digits -/
def decimalRep : ℕ → Fin 10 := 
  fun n => match n % 6 with
  | 0 => 0
  | 1 => 7
  | 2 => 6
  | 3 => 9
  | 4 => 2
  | 5 => 3
  | _ => 0  -- This case is unreachable, but needed for exhaustiveness

/-- The 150th digit after the decimal point in the decimal representation of 1/13 is 3 -/
theorem decimal_rep_1_13_150th_digit : decimalRep 150 = 3 := by
  sorry

end NUMINAMATH_CALUDE_decimal_rep_1_13_150th_digit_l39_3957


namespace NUMINAMATH_CALUDE_shaded_area_of_circles_l39_3905

theorem shaded_area_of_circles (r : ℝ) (h1 : r > 0) (h2 : π * r^2 = 81 * π) : 
  (π * r^2) / 2 + (π * (r/2)^2) / 2 = 50.625 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_of_circles_l39_3905


namespace NUMINAMATH_CALUDE_total_liquid_consumed_l39_3962

/-- Proves that the total amount of liquid consumed by Yurim and Ji-in is 6300 milliliters -/
theorem total_liquid_consumed (yurim_liters : ℕ) (yurim_ml : ℕ) (jiin_ml : ℕ) :
  yurim_liters = 2 →
  yurim_ml = 600 →
  jiin_ml = 3700 →
  yurim_liters * 1000 + yurim_ml + jiin_ml = 6300 :=
by
  sorry

end NUMINAMATH_CALUDE_total_liquid_consumed_l39_3962


namespace NUMINAMATH_CALUDE_no_prime_square_diff_4048_l39_3931

theorem no_prime_square_diff_4048 : ¬ ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p^2 - q^2 = 4048 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_square_diff_4048_l39_3931


namespace NUMINAMATH_CALUDE_solve_equation_l39_3988

theorem solve_equation (x : ℤ) (h : 9873 + x = 13200) : x = 3327 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l39_3988


namespace NUMINAMATH_CALUDE_election_winner_percentage_l39_3913

theorem election_winner_percentage (total_votes : ℕ) (majority : ℕ) : 
  total_votes = 470 → majority = 188 → 
  (70 : ℚ) * total_votes / 100 - ((100 : ℚ) - 70) * total_votes / 100 = majority := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l39_3913


namespace NUMINAMATH_CALUDE_secret_santa_five_friends_l39_3901

/-- The number of derangements for n elements -/
def derangement (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement (n - 1) + derangement (n - 2))

/-- The number of ways to distribute gifts in a Secret Santa game -/
def secretSantaDistributions (n : ℕ) : ℕ := derangement n

theorem secret_santa_five_friends :
  secretSantaDistributions 5 = 44 := by
  sorry

#eval secretSantaDistributions 5

end NUMINAMATH_CALUDE_secret_santa_five_friends_l39_3901


namespace NUMINAMATH_CALUDE_income_comparison_l39_3998

theorem income_comparison (tim mary juan : ℝ) 
  (h1 : mary = 1.6 * tim) 
  (h2 : mary = 0.8 * juan) : 
  tim = 0.5 * juan := by
  sorry

end NUMINAMATH_CALUDE_income_comparison_l39_3998


namespace NUMINAMATH_CALUDE_four_integers_product_2002_sum_less_40_l39_3945

theorem four_integers_product_2002_sum_less_40 :
  ∀ a b c d : ℕ+,
  a * b * c * d = 2002 →
  (a : ℕ) + b + c + d < 40 →
  ((a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨
   (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) ∨
   (a = 1 ∧ b = 11 ∧ c = 14 ∧ d = 13) ∨
   (a = 1 ∧ b = 11 ∧ c = 13 ∧ d = 14) ∨
   (a = 2 ∧ b = 11 ∧ c = 7 ∧ d = 13) ∨
   (a = 2 ∧ b = 11 ∧ c = 13 ∧ d = 7) ∨
   (a = 2 ∧ b = 13 ∧ c = 7 ∧ d = 11) ∨
   (a = 2 ∧ b = 13 ∧ c = 11 ∧ d = 7) ∨
   (a = 7 ∧ b = 2 ∧ c = 11 ∧ d = 13) ∨
   (a = 7 ∧ b = 2 ∧ c = 13 ∧ d = 11) ∨
   (a = 7 ∧ b = 11 ∧ c = 2 ∧ d = 13) ∨
   (a = 7 ∧ b = 11 ∧ c = 13 ∧ d = 2) ∨
   (a = 7 ∧ b = 13 ∧ c = 2 ∧ d = 11) ∧
   (a = 7 ∧ b = 13 ∧ c = 11 ∧ d = 2)) :=
by sorry

end NUMINAMATH_CALUDE_four_integers_product_2002_sum_less_40_l39_3945


namespace NUMINAMATH_CALUDE_opposite_values_l39_3900

theorem opposite_values (x y : ℝ) : 
  |x + y - 9| + (2*x - y + 3)^2 = 0 → x = 2 ∧ y = 7 := by
  sorry

end NUMINAMATH_CALUDE_opposite_values_l39_3900


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l39_3928

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 1)^2 = 4 ∧ (x₂ - 1)^2 = 4 ∧ x₁ = 3 ∧ x₂ = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l39_3928


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l39_3990

theorem right_rectangular_prism_volume 
  (a b c : ℝ) 
  (h1 : a * b = 36) 
  (h2 : a * c = 54) 
  (h3 : b * c = 72) : 
  a * b * c = 648 := by
sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l39_3990


namespace NUMINAMATH_CALUDE_cranberry_calculation_l39_3923

/-- The initial number of cranberries in the bog -/
def initial_cranberries : ℕ := 60000

/-- The fraction of cranberries harvested by humans -/
def human_harvest_fraction : ℚ := 2/5

/-- The number of cranberries eaten by elk -/
def elk_eaten : ℕ := 20000

/-- The number of cranberries left after harvesting and elk eating -/
def remaining_cranberries : ℕ := 16000

/-- Theorem stating that the initial number of cranberries is correct given the conditions -/
theorem cranberry_calculation :
  (1 - human_harvest_fraction) * initial_cranberries - elk_eaten = remaining_cranberries :=
by sorry

end NUMINAMATH_CALUDE_cranberry_calculation_l39_3923


namespace NUMINAMATH_CALUDE_minimize_expression_l39_3947

theorem minimize_expression (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 3 + 27 / n ≥ 6 ∧
  ((n : ℝ) / 3 + 27 / n = 6 ↔ n = 9) :=
sorry

end NUMINAMATH_CALUDE_minimize_expression_l39_3947


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l39_3956

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_pos : ∀ x > 0, f x = 2 * x - 3) : 
  ∀ x < 0, f x = 2 * x + 3 := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l39_3956


namespace NUMINAMATH_CALUDE_parallelogram_area_l39_3921

/-- The area of a parallelogram with longer diagonal 5 and heights 2 and 3 -/
theorem parallelogram_area (d : ℝ) (h₁ h₂ : ℝ) (hd : d = 5) (hh₁ : h₁ = 2) (hh₂ : h₂ = 3) :
  (h₁ * h₂) / (((3 * Real.sqrt 21 + 8) / 25) : ℝ) = 150 / (3 * Real.sqrt 21 + 8) := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l39_3921


namespace NUMINAMATH_CALUDE_paco_salty_cookies_left_l39_3963

/-- The number of salty cookies Paco has left after sharing with friends -/
def salty_cookies_left (initial_salty : ℕ) (shared_ana : ℕ) (shared_juan : ℕ) : ℕ :=
  initial_salty - (shared_ana + shared_juan)

/-- Theorem stating that Paco has 12 salty cookies left -/
theorem paco_salty_cookies_left :
  salty_cookies_left 26 11 3 = 12 := by sorry

end NUMINAMATH_CALUDE_paco_salty_cookies_left_l39_3963


namespace NUMINAMATH_CALUDE_biggest_number_l39_3978

theorem biggest_number (jungkook yoongi yuna : ℚ) : 
  jungkook = 6 / 3 → yoongi = 4 → yuna = 5 → 
  max (max jungkook yoongi) yuna = 5 := by
sorry

end NUMINAMATH_CALUDE_biggest_number_l39_3978


namespace NUMINAMATH_CALUDE_probability_prime_and_power_of_2_l39_3959

/-- The set of prime numbers between 1 and 8 (inclusive) -/
def primes_1_to_8 : Finset Nat := {2, 3, 5, 7}

/-- The set of powers of 2 between 1 and 8 (inclusive) -/
def powers_of_2_1_to_8 : Finset Nat := {1, 2, 4, 8}

/-- The number of sides on each die -/
def die_sides : Nat := 8

theorem probability_prime_and_power_of_2 :
  (Finset.card primes_1_to_8 * Finset.card powers_of_2_1_to_8) / (die_sides * die_sides) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_prime_and_power_of_2_l39_3959


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l39_3951

def A : Set ℝ := {x | ∃ y, y = Real.log x}
def B : Set ℝ := {-2, -1, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l39_3951


namespace NUMINAMATH_CALUDE_unique_solution_for_system_l39_3949

/-- The system of inequalities has a unique solution for specific values of a -/
theorem unique_solution_for_system (a : ℝ) :
  (∃! x y : ℝ, x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) ↔ 
  (a = 1 ∧ ∃! x y : ℝ, x = -1 ∧ y = 0 ∧ x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) ∨
  (a = -3 ∧ ∃! x y : ℝ, x = 1 ∧ y = 2 ∧ x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_system_l39_3949


namespace NUMINAMATH_CALUDE_equation_solution_l39_3995

theorem equation_solution : ∃! x : ℝ, (x - 12) / 3 = (3 * x + 9) / 8 ∧ x = -123 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l39_3995


namespace NUMINAMATH_CALUDE_fourth_root_of_105413504_l39_3908

theorem fourth_root_of_105413504 : (105413504 : ℝ) ^ (1/4 : ℝ) = 101 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_105413504_l39_3908


namespace NUMINAMATH_CALUDE_exists_prime_triplet_l39_3976

/-- A structure representing a prime triplet (a, b, c) -/
structure PrimeTriplet where
  a : Nat
  b : Nat
  c : Nat
  h_prime_a : Nat.Prime a
  h_prime_b : Nat.Prime b
  h_prime_c : Nat.Prime c
  h_order : a < b ∧ b < c ∧ c < 100
  h_geometric : (b + 1)^2 = (a + 1) * (c + 1)

/-- Theorem stating the existence of prime triplets satisfying the given conditions -/
theorem exists_prime_triplet : ∃ t : PrimeTriplet, True := by
  sorry

end NUMINAMATH_CALUDE_exists_prime_triplet_l39_3976


namespace NUMINAMATH_CALUDE_mrs_hilt_chickens_l39_3960

theorem mrs_hilt_chickens (total_legs : ℕ) (num_dogs : ℕ) (dog_legs : ℕ) (chicken_legs : ℕ) 
  (h1 : total_legs = 12)
  (h2 : num_dogs = 2)
  (h3 : dog_legs = 4)
  (h4 : chicken_legs = 2) :
  (total_legs - num_dogs * dog_legs) / chicken_legs = 2 := by
sorry

end NUMINAMATH_CALUDE_mrs_hilt_chickens_l39_3960


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l39_3904

theorem rational_inequality_solution (x : ℝ) : x / (x + 5) ≥ 0 ↔ x ∈ Set.Ici 0 ∪ Set.Iio (-5) := by
  sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l39_3904


namespace NUMINAMATH_CALUDE_min_value_ab_minus_cd_l39_3979

theorem min_value_ab_minus_cd (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 9)
  (h5 : a^2 + b^2 + c^2 + d^2 = 21) :
  a * b - c * d ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_minus_cd_l39_3979


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_geq_5_l39_3965

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | x ≤ a}
def B : Set ℝ := {x | x^2 - 5*x < 0}

-- State the theorem
theorem intersection_equality_implies_a_geq_5 (a : ℝ) :
  A a ∩ B = B → a ≥ 5 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_geq_5_l39_3965


namespace NUMINAMATH_CALUDE_expression_evaluation_l39_3930

theorem expression_evaluation :
  let a : ℝ := 1
  let b : ℝ := -1
  5 * (3 * a^2 * b - a * b^2) - (a * b^2 + 3 * a^2 * b) + 1 = -17 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l39_3930


namespace NUMINAMATH_CALUDE_only_one_and_two_satisfy_property_l39_3984

/-- A function that checks if a number has n-1 digits of 1 and one digit of 7 -/
def has_n_minus_1_ones_and_one_seven (x : ℕ) (n : ℕ) : Prop := sorry

/-- A function that generates all numbers with n-1 digits of 1 and one digit of 7 -/
def numbers_with_n_minus_1_ones_and_one_seven (n : ℕ) : Set ℕ := sorry

theorem only_one_and_two_satisfy_property :
  ∀ n : ℕ, (∀ x ∈ numbers_with_n_minus_1_ones_and_one_seven n, Nat.Prime x) ↔ (n = 1 ∨ n = 2) :=
sorry

end NUMINAMATH_CALUDE_only_one_and_two_satisfy_property_l39_3984


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l39_3997

/-- The coefficient of x^2 in the expansion of (1+x)^7(1-x) -/
def coefficient_x_squared : ℤ := 14

/-- The expansion of (1+x)^7(1-x) -/
def expansion (x : ℝ) : ℝ := (1 + x)^7 * (1 - x)

theorem coefficient_x_squared_in_expansion :
  (∃ f : ℝ → ℝ, ∃ g : ℝ → ℝ, expansion = λ x => coefficient_x_squared * x^2 + x * f x + g x) :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l39_3997


namespace NUMINAMATH_CALUDE_two_children_gender_combinations_l39_3942

-- Define the possible genders
inductive Gender
| Male
| Female

-- Define a type for a pair of children's genders
def ChildrenGenders := (Gender × Gender)

-- Define the set of all possible gender combinations
def allGenderCombinations : Set ChildrenGenders :=
  {(Gender.Male, Gender.Male), (Gender.Male, Gender.Female),
   (Gender.Female, Gender.Male), (Gender.Female, Gender.Female)}

-- Theorem statement
theorem two_children_gender_combinations :
  ∀ (family : ChildrenGenders), family ∈ allGenderCombinations :=
by sorry

end NUMINAMATH_CALUDE_two_children_gender_combinations_l39_3942


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l39_3924

theorem complex_fraction_equality (a b : ℂ) 
  (h : (a + b) / (a - b) - (a - b) / (a + b) = 0) : 
  (a^4 + b^4) / (a^4 - b^4) - (a^4 - b^4) / (a^4 + b^4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l39_3924


namespace NUMINAMATH_CALUDE_average_age_of_first_seven_students_l39_3954

theorem average_age_of_first_seven_students 
  (total_students : Nat) 
  (average_age_all : ℚ) 
  (second_group_size : Nat) 
  (average_age_second_group : ℚ) 
  (age_last_student : ℚ) 
  (h1 : total_students = 15)
  (h2 : average_age_all = 15)
  (h3 : second_group_size = 7)
  (h4 : average_age_second_group = 16)
  (h5 : age_last_student = 15) :
  let first_group_size := total_students - second_group_size - 1
  let total_age := average_age_all * total_students
  let second_group_total_age := average_age_second_group * second_group_size
  let first_group_total_age := total_age - second_group_total_age - age_last_student
  first_group_total_age / first_group_size = 14 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_first_seven_students_l39_3954


namespace NUMINAMATH_CALUDE_renatas_transactions_l39_3909

/-- Represents Renata's financial transactions and final balance --/
theorem renatas_transactions (initial_amount casino_and_water_cost lottery_win final_balance : ℚ) :
  initial_amount = 10 →
  lottery_win = 65 →
  final_balance = 94 →
  casino_and_water_cost = 67 →
  initial_amount - 4 + 90 - casino_and_water_cost + lottery_win = final_balance :=
by sorry

end NUMINAMATH_CALUDE_renatas_transactions_l39_3909


namespace NUMINAMATH_CALUDE_equal_balance_after_10_days_l39_3961

/-- Carol's initial borrowing in clams -/
def carol_initial : ℝ := 200

/-- Emily's initial borrowing in clams -/
def emily_initial : ℝ := 250

/-- Carol's daily interest rate -/
def carol_rate : ℝ := 0.15

/-- Emily's daily interest rate -/
def emily_rate : ℝ := 0.10

/-- Number of days after which Carol and Emily owe the same amount -/
def days_equal : ℕ := 10

/-- Carol's balance after t days -/
def carol_balance (t : ℝ) : ℝ := carol_initial * (1 + carol_rate * t)

/-- Emily's balance after t days -/
def emily_balance (t : ℝ) : ℝ := emily_initial * (1 + emily_rate * t)

theorem equal_balance_after_10_days :
  carol_balance days_equal = emily_balance days_equal :=
by sorry

end NUMINAMATH_CALUDE_equal_balance_after_10_days_l39_3961


namespace NUMINAMATH_CALUDE_dinitrogen_monoxide_molecular_weight_l39_3929

/-- The atomic weight of nitrogen in atomic mass units (amu) -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of a compound consisting of two nitrogen atoms and one oxygen atom -/
def dinitrogen_monoxide_weight : ℝ := 2 * nitrogen_weight + oxygen_weight

/-- Theorem stating that the molecular weight of Dinitrogen monoxide is 44.02 amu -/
theorem dinitrogen_monoxide_molecular_weight :
  dinitrogen_monoxide_weight = 44.02 := by sorry

end NUMINAMATH_CALUDE_dinitrogen_monoxide_molecular_weight_l39_3929


namespace NUMINAMATH_CALUDE_max_m2_plus_n2_l39_3916

theorem max_m2_plus_n2 : ∃ (m n : ℕ),
  1 ≤ m ∧ m ≤ 2005 ∧
  1 ≤ n ∧ n ≤ 2005 ∧
  (n^2 + 2*m*n - 2*m^2)^2 = 1 ∧
  m^2 + n^2 = 702036 ∧
  ∀ (m' n' : ℕ),
    1 ≤ m' ∧ m' ≤ 2005 ∧
    1 ≤ n' ∧ n' ≤ 2005 ∧
    (n'^2 + 2*m'*n' - 2*m'^2)^2 = 1 →
    m'^2 + n'^2 ≤ 702036 :=
by sorry

end NUMINAMATH_CALUDE_max_m2_plus_n2_l39_3916


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l39_3996

theorem minimum_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  2 / x + 1 / y ≥ 9 / 2 ∧ (2 / x + 1 / y = 9 / 2 ↔ x = 2 / 3 ∧ y = 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l39_3996


namespace NUMINAMATH_CALUDE_strawberry_weight_calculation_l39_3935

/-- Calculates the weight of Marco's dad's strawberries after losing some. -/
def dads_strawberry_weight (total_initial : ℕ) (lost : ℕ) (marcos : ℕ) : ℕ :=
  total_initial - lost - marcos

/-- Theorem: Given the initial total weight of strawberries, the weight of strawberries lost,
    and Marco's current weight of strawberries, Marco's dad's current weight of strawberries
    is equal to the difference between the remaining total weight and Marco's current weight. -/
theorem strawberry_weight_calculation
  (total_initial : ℕ)
  (lost : ℕ)
  (marcos : ℕ)
  (h1 : total_initial = 36)
  (h2 : lost = 8)
  (h3 : marcos = 12) :
  dads_strawberry_weight total_initial lost marcos = 16 :=
by sorry

end NUMINAMATH_CALUDE_strawberry_weight_calculation_l39_3935


namespace NUMINAMATH_CALUDE_last_bead_is_white_l39_3912

/-- Represents the color of a bead -/
inductive BeadColor
| White
| Black
| Red

/-- Returns the color of the nth bead in the pattern -/
def nthBeadColor (n : ℕ) : BeadColor :=
  match n % 6 with
  | 1 => BeadColor.White
  | 2 | 3 => BeadColor.Black
  | _ => BeadColor.Red

/-- The total number of beads in the necklace -/
def totalBeads : ℕ := 85

theorem last_bead_is_white :
  nthBeadColor totalBeads = BeadColor.White := by
  sorry

end NUMINAMATH_CALUDE_last_bead_is_white_l39_3912


namespace NUMINAMATH_CALUDE_lines_planes_perpendicular_l39_3974

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem lines_planes_perpendicular
  (m n : Line) (α β : Plane)
  (h1 : parallel_lines m n)
  (h2 : parallel_line_plane m α)
  (h3 : perpendicular_line_plane n β) :
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_lines_planes_perpendicular_l39_3974


namespace NUMINAMATH_CALUDE_sequence_determination_l39_3975

theorem sequence_determination (p : ℕ) (hp : p.Prime ∧ p > 5) :
  ∀ (a : Fin ((p - 1) / 2) → ℕ),
  (∀ i, a i ∈ Finset.range ((p - 1) / 2 + 1) \ {0}) →
  (∀ i j, i ≠ j → ∃ r, (a i * a j) % p = r) →
  Function.Injective a :=
sorry

end NUMINAMATH_CALUDE_sequence_determination_l39_3975


namespace NUMINAMATH_CALUDE_three_divided_by_p_l39_3943

theorem three_divided_by_p (p q : ℝ) 
  (h1 : 3 / q = 18) 
  (h2 : p - q = 0.33333333333333337) : 
  3 / p = 6 := by
  sorry

end NUMINAMATH_CALUDE_three_divided_by_p_l39_3943


namespace NUMINAMATH_CALUDE_intersection_distance_l39_3939

/-- The circle C₁ -/
def C₁ (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 1

/-- The line l passing through (-4, 0) with slope angle π/4 -/
def l (x y : ℝ) : Prop := y = x + 4

/-- The intersection points of l and C₁ -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | C₁ p.1 p.2 ∧ l p.1 p.2}

/-- The theorem stating that the distance between the intersection points is √2 -/
theorem intersection_distance : 
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧ 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l39_3939


namespace NUMINAMATH_CALUDE_bud_is_eight_years_old_l39_3919

def buds_age (uncle_age : ℕ) : ℕ :=
  uncle_age / 3

theorem bud_is_eight_years_old (uncle_age : ℕ) (h : uncle_age = 24) :
  buds_age uncle_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_bud_is_eight_years_old_l39_3919


namespace NUMINAMATH_CALUDE_negate_positive_negate_negative_positive_negative_positive_positive_l39_3918

-- Define the operations
def negate (x : ℝ) : ℝ := -x
def positive (x : ℝ) : ℝ := x

-- Theorem statements
theorem negate_positive (x : ℝ) : negate (positive x) = -x := by sorry

theorem negate_negative (x : ℝ) : negate (negate x) = x := by sorry

theorem positive_negative (x : ℝ) : positive (negate x) = -x := by sorry

theorem positive_positive (x : ℝ) : positive (positive x) = x := by sorry

end NUMINAMATH_CALUDE_negate_positive_negate_negative_positive_negative_positive_positive_l39_3918


namespace NUMINAMATH_CALUDE_random_number_table_sampling_sequence_l39_3936

-- Define the steps as an enumeration
inductive SamplingStep
  | NumberIndividuals
  | ObtainSampleNumbers
  | SelectStartingNumber

-- Define the correct sequence
def correctSequence : List SamplingStep :=
  [SamplingStep.NumberIndividuals, SamplingStep.SelectStartingNumber, SamplingStep.ObtainSampleNumbers]

-- Theorem statement
theorem random_number_table_sampling_sequence :
  correctSequence = [SamplingStep.NumberIndividuals, SamplingStep.SelectStartingNumber, SamplingStep.ObtainSampleNumbers] :=
by sorry

end NUMINAMATH_CALUDE_random_number_table_sampling_sequence_l39_3936
