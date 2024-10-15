import Mathlib

namespace NUMINAMATH_CALUDE_min_abs_z_l2192_219219

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 16) + Complex.abs (z - 8 * Complex.I) = 18) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs (w - 16) + Complex.abs (w - 8 * Complex.I) = 18 ∧ Complex.abs w = 64 / 9 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_l2192_219219


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l2192_219289

theorem triangle_third_side_length (a b c : ℝ) (θ : ℝ) : 
  a = 10 → b = 12 → θ = π / 3 → c = Real.sqrt 124 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l2192_219289


namespace NUMINAMATH_CALUDE_change_received_l2192_219251

-- Define the number of apples
def num_apples : ℕ := 5

-- Define the cost per apple in cents
def cost_per_apple : ℕ := 30

-- Define the amount paid in dollars
def amount_paid : ℚ := 10

-- Define the function to calculate change
def calculate_change (num_apples : ℕ) (cost_per_apple : ℕ) (amount_paid : ℚ) : ℚ :=
  amount_paid - (num_apples * cost_per_apple : ℚ) / 100

-- Theorem statement
theorem change_received :
  calculate_change num_apples cost_per_apple amount_paid = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_change_received_l2192_219251


namespace NUMINAMATH_CALUDE_capital_ratio_l2192_219224

-- Define the partners' capitals
variable (a b c : ℝ)

-- Define the total profit and b's share
def total_profit : ℝ := 16500
def b_share : ℝ := 6000

-- State the theorem
theorem capital_ratio (h1 : b = 4 * c) (h2 : b_share / total_profit = b / (a + b + c)) :
  a / b = 17 / 4 := by
  sorry

end NUMINAMATH_CALUDE_capital_ratio_l2192_219224


namespace NUMINAMATH_CALUDE_unique_square_with_property_l2192_219208

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def all_digits_less_than_7 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ (n.digits 10) → d < 7

def add_3_to_digits (n : ℕ) : ℕ :=
  n.digits 10
   |> List.map (· + 3)
   |> List.foldl (λ acc d => acc * 10 + d) 0

theorem unique_square_with_property :
  ∃! N : ℕ,
    1000 ≤ N ∧ N < 10000 ∧
    is_perfect_square N ∧
    all_digits_less_than_7 N ∧
    is_perfect_square (add_3_to_digits N) ∧
    N = 1156 :=
sorry

end NUMINAMATH_CALUDE_unique_square_with_property_l2192_219208


namespace NUMINAMATH_CALUDE_remainder_problem_l2192_219272

theorem remainder_problem (x : ℤ) : (x + 11) % 31 = 18 → x % 62 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2192_219272


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2192_219283

theorem quadratic_roots_relation (b c : ℚ) : 
  (∃ r₁ r₂ : ℚ, r₁ ≠ r₂ ∧ 
    (∀ x : ℚ, x^2 + b*x + c = 0 ↔ x = r₁ ∨ x = r₂) ∧
    (∃ s₁ s₂ : ℚ, s₁ ≠ s₂ ∧ 
      (∀ x : ℚ, 3*x^2 - 5*x - 7 = 0 ↔ x = s₁ ∨ x = s₂) ∧
      r₁ = s₁ + 3 ∧ r₂ = s₂ + 3)) →
  c = 35/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2192_219283


namespace NUMINAMATH_CALUDE_sphere_cone_equal_volume_l2192_219239

/-- Given a cone with radius 2 inches and height 6 inches, prove that a sphere
    with radius ∛6 inches has the same volume as the cone. -/
theorem sphere_cone_equal_volume :
  let cone_radius : ℝ := 2
  let cone_height : ℝ := 6
  let sphere_radius : ℝ := (6 : ℝ) ^ (1/3)
  (1/3 : ℝ) * Real.pi * cone_radius^2 * cone_height = (4/3 : ℝ) * Real.pi * sphere_radius^3 :=
by sorry

end NUMINAMATH_CALUDE_sphere_cone_equal_volume_l2192_219239


namespace NUMINAMATH_CALUDE_monotonicity_of_f_l2192_219254

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x / (x - 1)

theorem monotonicity_of_f (a : ℝ) (h_a : a ≠ 0) :
  (∀ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → 
    (a > 0 → f a x₁ > f a x₂) ∧ 
    (a < 0 → f a x₁ < f a x₂)) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_of_f_l2192_219254


namespace NUMINAMATH_CALUDE_systematic_sampling_milk_powder_l2192_219245

/-- Represents a systematic sampling selection -/
def SystematicSample (totalItems : ℕ) (sampleSize : ℕ) : List ℕ :=
  let interval := totalItems / sampleSize
  List.range sampleSize |>.map (fun i => (i + 1) * interval)

/-- The problem statement -/
theorem systematic_sampling_milk_powder :
  SystematicSample 50 5 = [5, 15, 25, 35, 45] := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_milk_powder_l2192_219245


namespace NUMINAMATH_CALUDE_third_term_base_l2192_219285

theorem third_term_base (x : ℝ) (some_number : ℝ) 
  (h1 : 625^(-x) + 25^(-2*x) + some_number^(-4*x) = 14)
  (h2 : x = 0.25) : 
  some_number = 125/1744 := by
sorry

end NUMINAMATH_CALUDE_third_term_base_l2192_219285


namespace NUMINAMATH_CALUDE_john_volunteer_hours_l2192_219222

/-- Represents John's volunteering schedule for a year -/
structure VolunteerSchedule where
  first_six_months_frequency : Nat
  first_six_months_hours : Nat
  next_five_months_frequency : Nat
  next_five_months_hours : Nat
  december_days : Nat
  december_total_hours : Nat

/-- Calculates the total volunteering hours for a year given a schedule -/
def total_volunteer_hours (schedule : VolunteerSchedule) : Nat :=
  (schedule.first_six_months_frequency * schedule.first_six_months_hours * 6) +
  (schedule.next_five_months_frequency * schedule.next_five_months_hours * 4 * 5) +
  schedule.december_total_hours

/-- Theorem stating that John's volunteering schedule results in 82 hours for the year -/
theorem john_volunteer_hours :
  ∃ (schedule : VolunteerSchedule),
    schedule.first_six_months_frequency = 2 ∧
    schedule.first_six_months_hours = 3 ∧
    schedule.next_five_months_frequency = 1 ∧
    schedule.next_five_months_hours = 2 ∧
    schedule.december_days = 3 ∧
    schedule.december_total_hours = 6 ∧
    total_volunteer_hours schedule = 82 := by
  sorry

end NUMINAMATH_CALUDE_john_volunteer_hours_l2192_219222


namespace NUMINAMATH_CALUDE_at_least_two_heads_probability_l2192_219298

def coin_toss_probability : ℕ → ℕ → ℚ
  | n, k => (Nat.choose n k : ℚ) * (1/2)^k * (1/2)^(n-k)

theorem at_least_two_heads_probability :
  coin_toss_probability 4 2 + coin_toss_probability 4 3 + coin_toss_probability 4 4 = 11/16 := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_heads_probability_l2192_219298


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l2192_219223

theorem polynomial_root_sum (a b : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + a * (Complex.I * Real.sqrt 2 + 2) + b = 0 → 
  a + b = 14 := by sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l2192_219223


namespace NUMINAMATH_CALUDE_change_parity_mismatch_l2192_219260

theorem change_parity_mismatch (bills : List ℕ) (denominations : List ℕ) :
  (bills.length = 10) →
  (∀ d ∈ denominations, d % 2 = 1) →
  (∀ b ∈ bills, b ∈ denominations) →
  (bills.sum ≠ 31) :=
sorry

end NUMINAMATH_CALUDE_change_parity_mismatch_l2192_219260


namespace NUMINAMATH_CALUDE_mango_problem_l2192_219220

theorem mango_problem (alexis_mangoes : ℕ) (dilan_ashley_mangoes : ℕ) : 
  alexis_mangoes = 60 → 
  alexis_mangoes = 4 * dilan_ashley_mangoes → 
  alexis_mangoes + dilan_ashley_mangoes = 75 := by
sorry

end NUMINAMATH_CALUDE_mango_problem_l2192_219220


namespace NUMINAMATH_CALUDE_x_squared_minus_4x_geq_m_l2192_219252

theorem x_squared_minus_4x_geq_m (m : ℝ) : 
  (∀ x ∈ Set.Icc 3 4, x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_4x_geq_m_l2192_219252


namespace NUMINAMATH_CALUDE_tv_clients_count_l2192_219235

def total_clients : ℕ := 180
def radio_clients : ℕ := 110
def magazine_clients : ℕ := 130
def tv_and_magazine_clients : ℕ := 85
def tv_and_radio_clients : ℕ := 75
def radio_and_magazine_clients : ℕ := 95
def all_three_clients : ℕ := 80

theorem tv_clients_count :
  ∃ (tv_clients : ℕ),
    tv_clients = total_clients + all_three_clients - radio_clients - magazine_clients + 
                 tv_and_magazine_clients + tv_and_radio_clients + radio_and_magazine_clients ∧
    tv_clients = 130 := by
  sorry

end NUMINAMATH_CALUDE_tv_clients_count_l2192_219235


namespace NUMINAMATH_CALUDE_expected_hits_value_l2192_219276

/-- The probability of hitting the target -/
def hit_probability : ℝ := 0.97

/-- The total number of shots -/
def total_shots : ℕ := 1000

/-- The expected number of hits -/
def expected_hits : ℝ := hit_probability * total_shots

theorem expected_hits_value : expected_hits = 970 := by
  sorry

end NUMINAMATH_CALUDE_expected_hits_value_l2192_219276


namespace NUMINAMATH_CALUDE_toy_sales_profit_maximization_l2192_219290

def weekly_sales (x : ℤ) (k : ℚ) (b : ℚ) : ℚ := k * x + b

theorem toy_sales_profit_maximization 
  (k : ℚ) (b : ℚ) 
  (h1 : weekly_sales 120 k b = 80) 
  (h2 : weekly_sales 140 k b = 40) 
  (h3 : ∀ x : ℤ, 100 ≤ x ∧ x ≤ 160) :
  (k = -2 ∧ b = 320) ∧
  (∀ x : ℤ, (x - 100) * (weekly_sales x k b) ≤ 1800) ∧
  ((130 - 100) * (weekly_sales 130 k b) = 1800) :=
sorry

end NUMINAMATH_CALUDE_toy_sales_profit_maximization_l2192_219290


namespace NUMINAMATH_CALUDE_jonas_bookshelves_l2192_219210

/-- Calculates the maximum number of bookshelves that can fit in a room. -/
def max_bookshelves (total_space : ℕ) (reserved_space : ℕ) (shelf_space : ℕ) : ℕ :=
  (total_space - reserved_space) / shelf_space

/-- Proves that given the specific conditions, the maximum number of bookshelves is 3. -/
theorem jonas_bookshelves :
  max_bookshelves 400 160 80 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jonas_bookshelves_l2192_219210


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l2192_219293

/-- The position function of a particle -/
def position (t : ℝ) : ℝ := t^3 - 2*t

/-- The velocity function of a particle -/
def velocity (t : ℝ) : ℝ := 3*t^2 - 2

theorem instantaneous_velocity_at_3 : velocity 3 = 25 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l2192_219293


namespace NUMINAMATH_CALUDE_passes_through_fixed_point_not_in_fourth_quadrant_min_area_min_area_line_eq_l2192_219294

/-- Given a line l: kx - 3y + 2k + 3 = 0, where k ∈ ℝ -/
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - 3 * y + 2 * k + 3 = 0

/-- The point (-2, 1) -/
def fixed_point : ℝ × ℝ := (-2, 1)

/-- The line passes through the fixed point for all values of k -/
theorem passes_through_fixed_point (k : ℝ) :
  line_l k (fixed_point.1) (fixed_point.2) := by sorry

/-- The line does not pass through the fourth quadrant when k ∈ [0, +∞) -/
theorem not_in_fourth_quadrant (k : ℝ) (hk : k ≥ 0) :
  ∀ x y, line_l k x y → (x ≤ 0 ∧ y ≥ 0) ∨ (x ≥ 0 ∧ y ≥ 0) := by sorry

/-- The area of triangle AOB formed by the line's intersections with the x and y axes -/
noncomputable def triangle_area (k : ℝ) : ℝ :=
  (1/6) * (4 * k + 9 / k + 12)

/-- The minimum area of triangle AOB is 4, occurring when k = 3/2 -/
theorem min_area :
  ∃ k, k > 0 ∧ triangle_area k = 4 ∧ ∀ k', k' > 0 → triangle_area k' ≥ 4 := by sorry

/-- The line equation at the minimum area point -/
def min_area_line (x y : ℝ) : Prop := x - 2 * y + 4 = 0

/-- The line equation at the minimum area point is x - 2y + 4 = 0 -/
theorem min_area_line_eq :
  ∃ k, k > 0 ∧ triangle_area k = 4 ∧ ∀ x y, line_l k x y ↔ min_area_line x y := by sorry

end NUMINAMATH_CALUDE_passes_through_fixed_point_not_in_fourth_quadrant_min_area_min_area_line_eq_l2192_219294


namespace NUMINAMATH_CALUDE_competition_participants_l2192_219273

theorem competition_participants (right_rank left_rank : ℕ) 
  (h1 : right_rank = 18) 
  (h2 : left_rank = 12) : 
  right_rank + left_rank - 1 = 29 := by
  sorry

end NUMINAMATH_CALUDE_competition_participants_l2192_219273


namespace NUMINAMATH_CALUDE_tenth_toss_probability_l2192_219275

/-- A fair coin is a coin with equal probability of landing heads or tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of getting heads on a single toss of a fair coin -/
def prob_heads (p : ℝ) : Prop := p = 1/2

/-- The number of times the coin has been tossed -/
def num_tosses : ℕ := 9

/-- The number of heads obtained in the previous tosses -/
def num_heads : ℕ := 7

/-- The number of tails obtained in the previous tosses -/
def num_tails : ℕ := 2

theorem tenth_toss_probability (p : ℝ) 
  (h_fair : fair_coin p) 
  (h_prev_tosses : num_tosses = num_heads + num_tails) :
  prob_heads p := by sorry

end NUMINAMATH_CALUDE_tenth_toss_probability_l2192_219275


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l2192_219258

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 5)
  (hdb : d / b = 2 / 5) :
  a / c = 125 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l2192_219258


namespace NUMINAMATH_CALUDE_cubic_decreasing_l2192_219232

-- Define the cubic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 1

-- State the theorem
theorem cubic_decreasing (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_decreasing_l2192_219232


namespace NUMINAMATH_CALUDE_trailing_zeros_302_factorial_l2192_219214

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeros in 302! is 74 -/
theorem trailing_zeros_302_factorial :
  trailingZeros 302 = 74 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_302_factorial_l2192_219214


namespace NUMINAMATH_CALUDE_total_methods_is_fifteen_l2192_219211

/-- A two-stage test with options for each stage -/
structure TwoStageTest where
  first_stage_options : Nat
  second_stage_options : Nat

/-- Calculate the total number of testing methods for a two-stage test -/
def total_testing_methods (test : TwoStageTest) : Nat :=
  test.first_stage_options * test.second_stage_options

/-- The specific test configuration -/
def our_test : TwoStageTest :=
  { first_stage_options := 3
  , second_stage_options := 5 }

theorem total_methods_is_fifteen :
  total_testing_methods our_test = 15 := by
  sorry

#eval total_testing_methods our_test

end NUMINAMATH_CALUDE_total_methods_is_fifteen_l2192_219211


namespace NUMINAMATH_CALUDE_number_2005_location_l2192_219207

/-- The sum of the first n positive integers -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The last number in the nth row of the pattern -/
def last_in_row (n : ℕ) : ℕ := n^2

/-- The first number in the nth row of the pattern -/
def first_in_row (n : ℕ) : ℕ := last_in_row (n - 1) + 1

/-- The number of elements in the nth row of the pattern -/
def elements_in_row (n : ℕ) : ℕ := 2 * n - 1

/-- The position of a number within its row in the pattern -/
def position_in_row (n : ℕ) (target : ℕ) : ℕ :=
  target - first_in_row n + 1

theorem number_2005_location :
  ∃ (i j : ℕ), i = 45 ∧ j = 20 ∧ 
  first_in_row i ≤ 2005 ∧
  2005 ≤ last_in_row i ∧
  position_in_row i 2005 = j :=
sorry

end NUMINAMATH_CALUDE_number_2005_location_l2192_219207


namespace NUMINAMATH_CALUDE_candy_distribution_l2192_219263

theorem candy_distribution (total_candies : ℕ) 
  (lollipops_per_boy : ℕ) (candy_canes_per_girl : ℕ) : 
  total_candies = 90 →
  lollipops_per_boy = 3 →
  candy_canes_per_girl = 2 →
  ∃ (num_boys num_girls : ℕ),
    num_boys * lollipops_per_boy = total_candies / 3 ∧
    num_girls * candy_canes_per_girl = total_candies * 2 / 3 ∧
    num_boys + num_girls = 40 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l2192_219263


namespace NUMINAMATH_CALUDE_triangle_properties_l2192_219266

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a + b = 13 →
  c = 7 →
  4 * (Real.sin ((A + B) / 2))^2 - Real.cos (2 * C) = 7/2 →
  C = π/3 ∧ 
  π * (2 * (1/2 * a * b * Real.sin C) / (a + b + c))^2 = 3*π :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2192_219266


namespace NUMINAMATH_CALUDE_a_seq_divisibility_l2192_219213

/-- Given a natural number a ≥ 2, define the sequence a_n recursively -/
def a_seq (a : ℕ) : ℕ → ℕ
  | 0 => a
  | n + 1 => a ^ (a_seq a n)

/-- The main theorem stating the divisibility property of the sequence -/
theorem a_seq_divisibility (a : ℕ) (h : a ≥ 2) (n : ℕ) :
  (a_seq a (n + 1) - a_seq a n) ∣ (a_seq a (n + 2) - a_seq a (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_a_seq_divisibility_l2192_219213


namespace NUMINAMATH_CALUDE_max_expression_value_l2192_219242

def expression (a b c d : ℕ) : ℕ := c * a^b - d

theorem max_expression_value :
  ∃ (a b c d : ℕ), 
    a ∈ ({0, 1, 2, 3} : Set ℕ) ∧ 
    b ∈ ({0, 1, 2, 3} : Set ℕ) ∧ 
    c ∈ ({0, 1, 2, 3} : Set ℕ) ∧ 
    d ∈ ({0, 1, 2, 3} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    expression a b c d = 9 ∧
    ∀ (w x y z : ℕ), 
      w ∈ ({0, 1, 2, 3} : Set ℕ) → 
      x ∈ ({0, 1, 2, 3} : Set ℕ) → 
      y ∈ ({0, 1, 2, 3} : Set ℕ) → 
      z ∈ ({0, 1, 2, 3} : Set ℕ) →
      w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
      expression w x y z ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_expression_value_l2192_219242


namespace NUMINAMATH_CALUDE_min_dials_for_equal_sums_l2192_219286

/-- A type representing a 12-sided dial with numbers from 1 to 12 -/
def Dial := Fin 12 → Fin 12

/-- A stack of dials -/
def Stack := ℕ → Dial

/-- The sum of numbers in a column of the stack -/
def columnSum (s : Stack) (col : Fin 12) (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => (s i col).val + 1)

/-- Whether all column sums have the same remainder modulo 12 -/
def allColumnSumsEqualMod12 (s : Stack) (n : ℕ) : Prop :=
  ∀ (c₁ c₂ : Fin 12), columnSum s c₁ n % 12 = columnSum s c₂ n % 12

/-- The theorem stating that 12 is the minimum number of dials required -/
theorem min_dials_for_equal_sums :
  ∀ (s : Stack), (∃ (n : ℕ), allColumnSumsEqualMod12 s n) →
  (∃ (n : ℕ), n ≥ 12 ∧ allColumnSumsEqualMod12 s n) :=
by sorry

end NUMINAMATH_CALUDE_min_dials_for_equal_sums_l2192_219286


namespace NUMINAMATH_CALUDE_cauchy_schwarz_inequality_l2192_219204

theorem cauchy_schwarz_inequality (a b a₁ b₁ : ℝ) :
  (a * a₁ + b * b₁)^2 ≤ (a^2 + b^2) * (a₁^2 + b₁^2) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_inequality_l2192_219204


namespace NUMINAMATH_CALUDE_problem_solution_l2192_219233

theorem problem_solution (a b c : ℝ) : 
  8 = 0.06 * a → 
  6 = 0.08 * b → 
  c = b / a → 
  c = 0.5625 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2192_219233


namespace NUMINAMATH_CALUDE_li_shuang_walking_speed_l2192_219206

/-- The problem of finding Li Shuang's walking speed -/
theorem li_shuang_walking_speed 
  (initial_speed : ℝ) 
  (walking_time : ℝ) 
  (repair_distance : ℝ) 
  (repair_time : ℝ) 
  (speed_multiplier : ℝ) 
  (delay : ℝ)
  (h1 : initial_speed = 320)
  (h2 : walking_time = 5)
  (h3 : repair_distance = 1800)
  (h4 : repair_time = 15)
  (h5 : speed_multiplier = 1.5)
  (h6 : delay = 17) :
  ∃ (walking_speed : ℝ), walking_speed = 72 ∧ 
  (∃ (total_distance : ℝ), 
    total_distance / initial_speed + delay = 
    walking_time + repair_time + 
    (total_distance - repair_distance - walking_speed * walking_time) / (initial_speed * speed_multiplier)) := by
  sorry

end NUMINAMATH_CALUDE_li_shuang_walking_speed_l2192_219206


namespace NUMINAMATH_CALUDE_jane_baking_time_l2192_219234

/-- Represents the time it takes Jane to bake cakes individually -/
def jane_time : ℝ := 4

/-- Represents the time it takes Roy to bake cakes individually -/
def roy_time : ℝ := 5

/-- The time Jane and Roy work together -/
def joint_work_time : ℝ := 2

/-- The time Jane works alone after Roy leaves -/
def jane_solo_time : ℝ := 0.4

/-- The theorem stating that Jane's individual baking time is 4 hours -/
theorem jane_baking_time :
  (joint_work_time * (1 / jane_time + 1 / roy_time)) + 
  (jane_solo_time * (1 / jane_time)) = 1 :=
sorry

end NUMINAMATH_CALUDE_jane_baking_time_l2192_219234


namespace NUMINAMATH_CALUDE_inequality_multiplication_l2192_219268

theorem inequality_multiplication (m n : ℝ) (h : m > n) : 2 * m > 2 * n := by
  sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l2192_219268


namespace NUMINAMATH_CALUDE_largest_divisible_by_seven_l2192_219280

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  ∃ (A B C : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    n = A * 10000 + B * 1000 + B * 100 + C * 10 + A

theorem largest_divisible_by_seven :
  ∀ n : ℕ, is_valid_number n → n % 7 = 0 → n ≤ 98879 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_by_seven_l2192_219280


namespace NUMINAMATH_CALUDE_triangle_side_length_l2192_219226

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 60 * π / 180 →  -- Convert 60° to radians
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  c = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2192_219226


namespace NUMINAMATH_CALUDE_kids_difference_l2192_219265

def kids_monday : ℕ := 6
def kids_wednesday : ℕ := 4

theorem kids_difference : kids_monday - kids_wednesday = 2 := by
  sorry

end NUMINAMATH_CALUDE_kids_difference_l2192_219265


namespace NUMINAMATH_CALUDE_ice_cream_group_size_l2192_219238

/-- The number of days it takes one person to eat a gallon of ice cream -/
def days_per_person : ℕ := 5 * 16

/-- The number of days it takes the group to eat a gallon of ice cream -/
def days_for_group : ℕ := 10

/-- The number of people in the group -/
def people_in_group : ℕ := days_per_person / days_for_group

theorem ice_cream_group_size :
  people_in_group = 8 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_group_size_l2192_219238


namespace NUMINAMATH_CALUDE_infinite_male_lineage_l2192_219215

/-- Represents a male person --/
structure Male where
  name : String

/-- Represents the son relationship between two males --/
def is_son_of (son father : Male) : Prop := sorry

/-- Adam, the first male --/
def adam : Male := ⟨"Adam"⟩

/-- An infinite sequence of males --/
def male_sequence : ℕ → Male
| 0 => adam
| n + 1 => sorry

/-- Theorem stating the existence of an infinite male lineage starting from Adam --/
theorem infinite_male_lineage :
  (∀ n : ℕ, is_son_of (male_sequence (n + 1)) (male_sequence n)) ∧
  (∀ n : ℕ, ∃ m : ℕ, m > n) :=
sorry

end NUMINAMATH_CALUDE_infinite_male_lineage_l2192_219215


namespace NUMINAMATH_CALUDE_symmetric_even_function_implies_odd_l2192_219267

/-- A function satisfying certain symmetry and evenness properties -/
def SymmetricEvenFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x - 1) + f (1 - x) = 0) ∧ 
  (∀ x, f (x + 1) = f (-x + 1)) ∧
  (f (-3/2) = 1)

/-- The main theorem stating that f(x-2) is an odd function -/
theorem symmetric_even_function_implies_odd (f : ℝ → ℝ) 
  (h : SymmetricEvenFunction f) : 
  ∀ x, f (-(x - 2)) = -f (x - 2) := by
sorry

end NUMINAMATH_CALUDE_symmetric_even_function_implies_odd_l2192_219267


namespace NUMINAMATH_CALUDE_generatable_pairs_theorem_l2192_219225

/-- Given two positive integers and the operations of sum, product, and integer ratio,
    this function determines which pairs of positive integers can be generated. -/
def generatable_pairs (m n : ℕ+) : Set (ℕ+ × ℕ+) :=
  if m = 1 ∧ n = 1 then Set.univ
  else Set.univ \ {(1, 1)}

/-- The main theorem stating which pairs can be generated based on the initial values -/
theorem generatable_pairs_theorem (m n : ℕ+) :
  (∀ (a b : ℕ+), (a, b) ∈ generatable_pairs m n) ∨
  (∀ (a b : ℕ+), (a, b) ≠ (1, 1) → (a, b) ∈ generatable_pairs m n) :=
sorry

end NUMINAMATH_CALUDE_generatable_pairs_theorem_l2192_219225


namespace NUMINAMATH_CALUDE_xy_minimum_value_l2192_219212

theorem xy_minimum_value (x y : ℝ) (hx : x > 1) (hy : y > 1) 
  (h_geom : (1/4 * Real.log x) * (Real.log y) = (1/4)^2) : x * y ≥ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_minimum_value_l2192_219212


namespace NUMINAMATH_CALUDE_prime_arithmetic_progression_difference_divisibility_l2192_219221

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def arithmetic_progression (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem prime_arithmetic_progression_difference_divisibility
  (p : ℕ → ℕ)
  (d : ℕ)
  (h_prime : ∀ n, n ∈ Finset.range 15 → is_prime (p n))
  (h_increasing : ∀ n, n ∈ Finset.range 14 → p n < p (n + 1))
  (h_arith_prog : arithmetic_progression p d) :
  ∃ k : ℕ, d = k * (2 * 3 * 5 * 7 * 11 * 13) :=
sorry

end NUMINAMATH_CALUDE_prime_arithmetic_progression_difference_divisibility_l2192_219221


namespace NUMINAMATH_CALUDE_total_spent_theorem_l2192_219236

/-- Calculates the total amount spent on pens by Dorothy, Julia, and Robert --/
def total_spent_on_pens (robert_pens : ℕ) (julia_factor : ℕ) (dorothy_factor : ℚ) (cost_per_pen : ℚ) : ℚ :=
  let julia_pens := julia_factor * robert_pens
  let dorothy_pens := dorothy_factor * julia_pens
  let total_pens := robert_pens + julia_pens + dorothy_pens
  total_pens * cost_per_pen

/-- Theorem stating the total amount spent on pens by Dorothy, Julia, and Robert --/
theorem total_spent_theorem :
  total_spent_on_pens 4 3 (1/2) (3/2) = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_theorem_l2192_219236


namespace NUMINAMATH_CALUDE_apex_angle_of_regular_quad_pyramid_l2192_219246

-- Define a regular quadrilateral pyramid
structure RegularQuadPyramid where
  -- We don't need to specify all properties, just the relevant ones
  apex_angle : ℝ
  dihedral_angle : ℝ

-- State the theorem
theorem apex_angle_of_regular_quad_pyramid 
  (pyramid : RegularQuadPyramid)
  (h : pyramid.dihedral_angle = 2 * pyramid.apex_angle) :
  pyramid.apex_angle = Real.arccos ((Real.sqrt 5 - 1) / 2) := by
  sorry


end NUMINAMATH_CALUDE_apex_angle_of_regular_quad_pyramid_l2192_219246


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_expression_l2192_219255

theorem min_value_of_quadratic_expression (x y : ℝ) :
  2 * x^2 + 3 * y^2 - 12 * x + 9 * y + 35 ≥ 41 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_expression_l2192_219255


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2192_219278

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 9*x*y - x^2 - 8*y^2 = 2005 ↔ 
    (x = 63 ∧ y = 58) ∨ (x = -63 ∧ y = -58) ∨ 
    (x = 459 ∧ y = 58) ∨ (x = -459 ∧ y = -58) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2192_219278


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2192_219274

def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3 * p.1 - 2}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1}

theorem intersection_of_A_and_B : A ∩ B = {(1, 1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2192_219274


namespace NUMINAMATH_CALUDE_maggies_income_l2192_219240

/-- Maggie's weekly income calculation -/
theorem maggies_income
  (office_rate : ℝ)
  (tractor_rate : ℝ)
  (tractor_hours : ℝ)
  (total_income : ℝ)
  (h1 : tractor_rate = 12)
  (h2 : tractor_hours = 13)
  (h3 : total_income = 416)
  (h4 : office_rate * (2 * tractor_hours) + tractor_rate * tractor_hours = total_income) :
  office_rate = 10 := by
sorry

end NUMINAMATH_CALUDE_maggies_income_l2192_219240


namespace NUMINAMATH_CALUDE_number_of_pots_l2192_219284

/-- Given a collection of pots where each pot contains 71 flowers,
    and there are 10011 flowers in total, prove that there are 141 pots. -/
theorem number_of_pots (flowers_per_pot : ℕ) (total_flowers : ℕ) (h1 : flowers_per_pot = 71) (h2 : total_flowers = 10011) :
  total_flowers / flowers_per_pot = 141 := by
  sorry


end NUMINAMATH_CALUDE_number_of_pots_l2192_219284


namespace NUMINAMATH_CALUDE_g_inequality_l2192_219218

/-- A quadratic function f(x) = ax^2 + a that is even on the interval [-a, a^2] -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a

/-- The function g(x) = f(x-1) -/
def g (a : ℝ) (x : ℝ) : ℝ := f a (x - 1)

/-- Theorem stating the relationship between g(3/2), g(0), and g(3) -/
theorem g_inequality (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∀ x ∈ Set.Icc (-a) (a^2), f a x = f a (-x)) : 
  g a (3/2) < g a 0 ∧ g a 0 < g a 3 := by
  sorry

end NUMINAMATH_CALUDE_g_inequality_l2192_219218


namespace NUMINAMATH_CALUDE_peters_children_l2192_219259

theorem peters_children (initial_savings : ℕ) (addition : ℕ) (num_children : ℕ) : 
  initial_savings = 642986 →
  addition = 642987 →
  (initial_savings + addition) % num_children = 0 →
  num_children = 642987 := by
sorry

end NUMINAMATH_CALUDE_peters_children_l2192_219259


namespace NUMINAMATH_CALUDE_well_digging_hours_l2192_219295

/-- The number of hours worked on the first day by two men digging a well -/
def first_day_hours : ℕ := 20

/-- The total payment for both men over three days of work -/
def total_payment : ℕ := 660

/-- The hourly rate paid to each man -/
def hourly_rate : ℕ := 10

/-- The number of hours worked by both men on the second day -/
def second_day_hours : ℕ := 16

/-- The number of hours worked by both men on the third day -/
def third_day_hours : ℕ := 30

theorem well_digging_hours : 
  hourly_rate * (first_day_hours + second_day_hours + third_day_hours) = total_payment :=
by sorry

end NUMINAMATH_CALUDE_well_digging_hours_l2192_219295


namespace NUMINAMATH_CALUDE_product_line_size_l2192_219282

/-- Represents the product line of Company C -/
structure ProductLine where
  n : ℕ                  -- number of products
  prices : Fin n → ℝ     -- prices of products
  avg_price : ℝ          -- average price
  min_price : ℝ          -- minimum price
  max_price : ℝ          -- maximum price
  low_price_count : ℕ    -- count of products below $1000

/-- The product line satisfies the given conditions -/
def satisfies_conditions (pl : ProductLine) : Prop :=
  pl.avg_price = 1200 ∧
  (∀ i, pl.prices i ≥ 400) ∧
  pl.low_price_count = 10 ∧
  (∀ i, pl.prices i < 1000 ∨ pl.prices i ≥ 1000) ∧
  pl.max_price = 11000 ∧
  (∃ i, pl.prices i = pl.max_price)

/-- The theorem to be proved -/
theorem product_line_size (pl : ProductLine) 
  (h : satisfies_conditions pl) : pl.n = 20 := by
  sorry


end NUMINAMATH_CALUDE_product_line_size_l2192_219282


namespace NUMINAMATH_CALUDE_katie_earnings_l2192_219257

/-- The number of bead necklaces Katie sold -/
def bead_necklaces : ℕ := 4

/-- The number of gem stone necklaces Katie sold -/
def gem_necklaces : ℕ := 3

/-- The cost of each necklace in dollars -/
def necklace_cost : ℕ := 3

/-- The total money Katie earned from selling necklaces -/
def total_earned : ℕ := (bead_necklaces + gem_necklaces) * necklace_cost

theorem katie_earnings : total_earned = 21 := by
  sorry

end NUMINAMATH_CALUDE_katie_earnings_l2192_219257


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2192_219229

theorem inequality_solution_range (k : ℝ) : 
  (∀ x : ℝ, -x^2 + k*x - 4 < 0) → -4 < k ∧ k < 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2192_219229


namespace NUMINAMATH_CALUDE_wholesale_price_proof_l2192_219231

def retail_price : ℝ := 144

theorem wholesale_price_proof :
  ∃ (wholesale_price : ℝ),
    wholesale_price = 108 ∧
    retail_price = 144 ∧
    retail_price * 0.9 = wholesale_price + wholesale_price * 0.2 :=
by sorry

end NUMINAMATH_CALUDE_wholesale_price_proof_l2192_219231


namespace NUMINAMATH_CALUDE_added_number_after_doubling_l2192_219297

theorem added_number_after_doubling (original : ℕ) (added : ℕ) : 
  original = 6 →
  3 * (2 * original + added) = 63 →
  added = 9 := by
sorry

end NUMINAMATH_CALUDE_added_number_after_doubling_l2192_219297


namespace NUMINAMATH_CALUDE_fraction_equality_l2192_219228

theorem fraction_equality : (5 + 4 - 3) / (5 + 4 + 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2192_219228


namespace NUMINAMATH_CALUDE_three_roots_and_minimum_implies_ratio_l2192_219261

/-- Given positive real numbers a, b, c with a > c, if the equation |x²-ax+b| = cx 
    has exactly three distinct real roots, and the function f(x) = |x²-ax+b| + cx 
    has a minimum value of c², then a/c = 5 -/
theorem three_roots_and_minimum_implies_ratio (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hac : a > c)
  (h_three_roots : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    |x^2 - a*x + b| = c*x ∧ |y^2 - a*y + b| = c*y ∧ |z^2 - a*z + b| = c*z)
  (h_min : ∃ m : ℝ, ∀ x : ℝ, |x^2 - a*x + b| + c*x ≥ c^2 ∧ 
    ∃ x₀ : ℝ, |x₀^2 - a*x₀ + b| + c*x₀ = c^2) :
  a / c = 5 := by
sorry

end NUMINAMATH_CALUDE_three_roots_and_minimum_implies_ratio_l2192_219261


namespace NUMINAMATH_CALUDE_bookseller_display_windows_l2192_219217

/-- Given the conditions of the bookseller's display windows problem, prove that the number of non-fiction books is 2. -/
theorem bookseller_display_windows (fiction_books : ℕ) (display_fiction : ℕ) (total_configs : ℕ) :
  fiction_books = 4 →
  display_fiction = 3 →
  total_configs = 36 →
  ∃ n : ℕ, n = 2 ∧ (Nat.factorial fiction_books / Nat.factorial (fiction_books - display_fiction)) * Nat.factorial n = total_configs :=
by sorry

end NUMINAMATH_CALUDE_bookseller_display_windows_l2192_219217


namespace NUMINAMATH_CALUDE_lulu_poptarts_count_l2192_219241

/-- Represents the number of pastries baked by Lola and Lulu -/
structure PastryCounts where
  lola_cupcakes : ℕ
  lola_poptarts : ℕ
  lola_pies : ℕ
  lulu_cupcakes : ℕ
  lulu_poptarts : ℕ
  lulu_pies : ℕ

/-- The total number of pastries baked by Lola and Lulu -/
def total_pastries (counts : PastryCounts) : ℕ :=
  counts.lola_cupcakes + counts.lola_poptarts + counts.lola_pies +
  counts.lulu_cupcakes + counts.lulu_poptarts + counts.lulu_pies

/-- Theorem stating that Lulu baked 12 pop tarts -/
theorem lulu_poptarts_count (counts : PastryCounts) 
  (h1 : counts.lola_cupcakes = 13)
  (h2 : counts.lola_poptarts = 10)
  (h3 : counts.lola_pies = 8)
  (h4 : counts.lulu_cupcakes = 16)
  (h5 : counts.lulu_pies = 14)
  (h6 : total_pastries counts = 73) :
  counts.lulu_poptarts = 12 := by
  sorry

end NUMINAMATH_CALUDE_lulu_poptarts_count_l2192_219241


namespace NUMINAMATH_CALUDE_inequality_proof_l2192_219202

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a^4 + b^4 ≥ a^3 * b + a * b^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2192_219202


namespace NUMINAMATH_CALUDE_quadratic_minimum_positive_l2192_219253

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 5

-- Theorem statement
theorem quadratic_minimum_positive :
  ∃ (x_min : ℝ), x_min > 0 ∧ 
  (∀ (x : ℝ), f x ≥ f x_min) ∧
  (∀ (ε : ℝ), ε > 0 → ∃ (x : ℝ), x ≠ x_min ∧ |x - x_min| < ε ∧ f x > f x_min) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_positive_l2192_219253


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l2192_219209

theorem cylinder_volume_change (r h V : ℝ) : 
  V = π * r^2 * h → (π * (3*r)^2 * (4*h) = 36 * V) := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l2192_219209


namespace NUMINAMATH_CALUDE_sally_and_fred_onions_l2192_219299

/-- The number of onions Sally and Fred have after giving some to Sara -/
def remaining_onions (sally_onions fred_onions given_onions : ℕ) : ℕ :=
  sally_onions + fred_onions - given_onions

/-- Theorem stating that Sally and Fred have 10 onions after giving some to Sara -/
theorem sally_and_fred_onions :
  remaining_onions 5 9 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sally_and_fred_onions_l2192_219299


namespace NUMINAMATH_CALUDE_obtuse_triangle_count_l2192_219237

/-- A function that determines if a triangle with sides a, b, and c is obtuse -/
def is_obtuse (a b c : ℕ) : Prop :=
  (a ^ 2 > b ^ 2 + c ^ 2) ∨ (b ^ 2 > a ^ 2 + c ^ 2) ∨ (c ^ 2 > a ^ 2 + b ^ 2)

/-- A function that determines if three lengths can form a valid triangle -/
def is_valid_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

/-- The main theorem stating that there are exactly 14 positive integer values of k
    for which a triangle with side lengths 13, 17, and k is obtuse -/
theorem obtuse_triangle_count :
  (∃! (s : Finset ℕ), s.card = 14 ∧ 
    (∀ k, k ∈ s ↔ (k > 0 ∧ is_valid_triangle 13 17 k ∧ is_obtuse 13 17 k))) :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_count_l2192_219237


namespace NUMINAMATH_CALUDE_binary_multiplication_l2192_219227

def binary_to_nat : List Bool → Nat
  | [] => 0
  | b::bs => (if b then 1 else 0) + 2 * binary_to_nat bs

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then
    []
  else
    (n % 2 = 1) :: nat_to_binary (n / 2)

def binary_num1 : List Bool := [true, true, false, true, true]
def binary_num2 : List Bool := [true, true, true, true]
def binary_result : List Bool := [true, false, true, true, true, true, false, true]

theorem binary_multiplication :
  binary_to_nat binary_num1 * binary_to_nat binary_num2 = binary_to_nat binary_result := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_l2192_219227


namespace NUMINAMATH_CALUDE_team_selection_ways_l2192_219247

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem team_selection_ways : 
  let group_size := 6
  let selection_size := 3
  let num_groups := 2
  (choose group_size selection_size) ^ num_groups = 400 := by sorry

end NUMINAMATH_CALUDE_team_selection_ways_l2192_219247


namespace NUMINAMATH_CALUDE_pythagorean_triple_for_odd_integer_l2192_219201

theorem pythagorean_triple_for_odd_integer (x : ℕ) 
  (h1 : x > 1) 
  (h2 : Odd x) : 
  ∃ y z : ℕ, 
    y > 0 ∧ 
    z > 0 ∧ 
    y = (x^2 - 1) / 2 ∧ 
    z = (x^2 + 1) / 2 ∧ 
    x^2 + y^2 = z^2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_for_odd_integer_l2192_219201


namespace NUMINAMATH_CALUDE_parallelogram_area_l2192_219216

/-- Represents a parallelogram with given dimensions -/
structure Parallelogram where
  base : ℝ
  height : ℝ
  shift : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- Theorem: The area of the specific parallelogram is 140 square feet -/
theorem parallelogram_area :
  let p := Parallelogram.mk 20 7 8
  area p = 140 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2192_219216


namespace NUMINAMATH_CALUDE_triangle_square_side_ratio_l2192_219250

theorem triangle_square_side_ratio (t s : ℝ) : 
  t > 0 ∧ s > 0 ∧ 3 * t = 12 ∧ 4 * s = 12 → t / s = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_square_side_ratio_l2192_219250


namespace NUMINAMATH_CALUDE_real_part_of_z_l2192_219244

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z - 3) = -1 + 3 * Complex.I) : 
  z.re = 6 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l2192_219244


namespace NUMINAMATH_CALUDE_total_rainfall_sum_l2192_219200

/-- The total rainfall recorded over three days equals the sum of individual daily rainfall amounts. -/
theorem total_rainfall_sum (monday tuesday wednesday : Real) 
  (h1 : monday = 0.17)
  (h2 : tuesday = 0.42)
  (h3 : wednesday = 0.08) :
  monday + tuesday + wednesday = 0.67 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_sum_l2192_219200


namespace NUMINAMATH_CALUDE_octal_subtraction_correct_l2192_219264

/-- Represents a number in base 8 -/
def OctalNum := Nat

/-- Addition in base 8 -/
def octal_add (a b : OctalNum) : OctalNum :=
  sorry

/-- Subtraction in base 8 -/
def octal_sub (a b : OctalNum) : OctalNum :=
  sorry

/-- Conversion from decimal to octal -/
def to_octal (n : Nat) : OctalNum :=
  sorry

theorem octal_subtraction_correct :
  let a : OctalNum := to_octal 537
  let b : OctalNum := to_octal 261
  let c : OctalNum := to_octal 256
  octal_sub a b = c ∧ octal_add b c = a := by
  sorry

end NUMINAMATH_CALUDE_octal_subtraction_correct_l2192_219264


namespace NUMINAMATH_CALUDE_log4_one_sixteenth_eq_neg_two_l2192_219291

-- Define the logarithm function for base 4
noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4

-- Theorem statement
theorem log4_one_sixteenth_eq_neg_two : log4 (1/16) = -2 := by
  sorry

end NUMINAMATH_CALUDE_log4_one_sixteenth_eq_neg_two_l2192_219291


namespace NUMINAMATH_CALUDE_triangle_ABC_is_right_l2192_219243

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line passing through (5,-2)
def line_through_point (x y : ℝ) : Prop := ∃ (m b : ℝ), y = m*x + b ∧ -2 = m*5 + b

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop :=
  let (x₁, y₁) := t.A
  let (x₂, y₂) := t.B
  let (x₃, y₃) := t.C
  (x₂ - x₁) * (x₃ - x₁) + (y₂ - y₁) * (y₃ - y₁) = 0

-- Theorem statement
theorem triangle_ABC_is_right :
  ∀ (B C : ℝ × ℝ),
  parabola B.1 B.2 →
  parabola C.1 C.2 →
  line_through_point B.1 B.2 →
  line_through_point C.1 C.2 →
  is_right_triangle { A := (1, 2), B := B, C := C } :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_is_right_l2192_219243


namespace NUMINAMATH_CALUDE_combined_average_age_l2192_219248

theorem combined_average_age (people_c : ℕ) (avg_c : ℚ) (people_d : ℕ) (avg_d : ℚ) :
  people_c = 8 →
  avg_c = 35 →
  people_d = 6 →
  avg_d = 30 →
  (people_c * avg_c + people_d * avg_d) / (people_c + people_d : ℚ) = 33 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_age_l2192_219248


namespace NUMINAMATH_CALUDE_difference_of_numbers_l2192_219287

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 275) : |x - y| = 14 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l2192_219287


namespace NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l2192_219205

/-- The value of c for which the line 3x - y = c is a perpendicular bisector
    of the line segment from (2,4) to (6,8) -/
theorem perpendicular_bisector_c_value :
  ∃ c : ℝ,
    (∀ x y : ℝ, 3 * x - y = c → 
      ((x - 4) ^ 2 + (y - 6) ^ 2 = 8) ∧ 
      (3 * (x - 4) + (y - 6) = 0)) →
    c = 6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l2192_219205


namespace NUMINAMATH_CALUDE_discount_calculation_l2192_219277

theorem discount_calculation (original_price : ℝ) (original_price_pos : original_price > 0) :
  let first_discount := 0.3
  let second_discount := 0.25
  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  final_price / original_price = 0.525 := by
sorry

end NUMINAMATH_CALUDE_discount_calculation_l2192_219277


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2192_219230

-- Define the quadratic function
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- Define the solution set condition
def solution_set (b c : ℝ) : Set ℝ := {x | x > 2 ∨ x < 1}

-- Theorem statement
theorem quadratic_inequality (b c : ℝ) :
  (∀ x, x ∈ solution_set b c ↔ f b c x > 0) →
  (b = -3 ∧ c = 2) ∧
  (∀ x, x ∈ {x | 1/2 ≤ x ∧ x ≤ 1} ↔ 2*x^2 - 3*x + 1 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2192_219230


namespace NUMINAMATH_CALUDE_screenwriter_speed_l2192_219281

/-- Calculates the average words per minute for a given script and writing duration -/
def average_words_per_minute (total_words : ℕ) (total_hours : ℕ) : ℚ :=
  (total_words : ℚ) / (total_hours * 60 : ℚ)

/-- Theorem stating that a 30,000-word script written in 100 hours has an average writing speed of 5 words per minute -/
theorem screenwriter_speed : average_words_per_minute 30000 100 = 5 := by
  sorry

#eval average_words_per_minute 30000 100

end NUMINAMATH_CALUDE_screenwriter_speed_l2192_219281


namespace NUMINAMATH_CALUDE_cousin_age_l2192_219288

/-- Given the ages of Rick and his brothers, prove the age of their cousin -/
theorem cousin_age (rick_age : ℕ) (oldest_brother_age : ℕ) (middle_brother_age : ℕ) 
  (smallest_brother_age : ℕ) (youngest_brother_age : ℕ) (cousin_age : ℕ) 
  (h1 : rick_age = 15)
  (h2 : oldest_brother_age = 2 * rick_age)
  (h3 : middle_brother_age = oldest_brother_age / 3)
  (h4 : smallest_brother_age = middle_brother_age / 2)
  (h5 : youngest_brother_age = smallest_brother_age - 2)
  (h6 : cousin_age = 5 * youngest_brother_age) :
  cousin_age = 15 := by
  sorry

#check cousin_age

end NUMINAMATH_CALUDE_cousin_age_l2192_219288


namespace NUMINAMATH_CALUDE_company_fund_problem_l2192_219279

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  (initial_fund = 60 * n - 10) →    -- Fund was $10 short for $60 bonuses
  (50 * n + 120 = initial_fund) →   -- $50 bonuses given, $120 remained
  (initial_fund = 770) :=           -- Prove initial fund was $770
by
  sorry

end NUMINAMATH_CALUDE_company_fund_problem_l2192_219279


namespace NUMINAMATH_CALUDE_all_gp_lines_through_origin_l2192_219203

/-- A line in the 2D plane represented by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three real numbers form a geometric progression -/
def isGeometricProgression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = a * r^2

/-- The point (0, 0) in the 2D plane -/
def origin : ℝ × ℝ := (0, 0)

/-- Checks if a point lies on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 = l.c

theorem all_gp_lines_through_origin :
  ∀ l : Line, isGeometricProgression l.a l.b l.c → pointOnLine origin l :=
sorry

end NUMINAMATH_CALUDE_all_gp_lines_through_origin_l2192_219203


namespace NUMINAMATH_CALUDE_major_axis_endpoints_of_ellipse_l2192_219292

/-- An ellipse is defined by the equation 6x^2 + y^2 = 36 -/
def ellipse (x y : ℝ) : Prop := 6 * x^2 + y^2 = 36

/-- The endpoints of the major axis of the ellipse -/
def major_axis_endpoints : Set (ℝ × ℝ) := {(-6, 0), (6, 0)}

/-- Theorem: The coordinates of the endpoints of the major axis of the ellipse 6x^2 + y^2 = 36 are (0, -6) and (0, 6) -/
theorem major_axis_endpoints_of_ellipse :
  major_axis_endpoints = {(0, -6), (0, 6)} :=
sorry

end NUMINAMATH_CALUDE_major_axis_endpoints_of_ellipse_l2192_219292


namespace NUMINAMATH_CALUDE_smallest_positive_solution_tan_cos_l2192_219296

theorem smallest_positive_solution_tan_cos (x : ℝ) : 
  (x > 0 ∧ x = Real.pi / 8 ∧ Real.tan (2 * x) + Real.tan (4 * x) = Real.cos (2 * x)) ∧
  (∀ y : ℝ, y > 0 ∧ y < x → Real.tan (2 * y) + Real.tan (4 * y) ≠ Real.cos (2 * y)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_tan_cos_l2192_219296


namespace NUMINAMATH_CALUDE_x_power_expression_l2192_219249

theorem x_power_expression (x : ℝ) (h : x + 1/x = 3) :
  x^7 - 5*x^5 + 3*x^3 = 126*x - 48 := by
  sorry

end NUMINAMATH_CALUDE_x_power_expression_l2192_219249


namespace NUMINAMATH_CALUDE_max_value_polynomial_l2192_219262

theorem max_value_polynomial (x y : ℝ) (h : x + y = 3) :
  ∃ M : ℝ, M = 400 / 11 ∧ 
  ∀ a b : ℝ, a + b = 3 → 
  a^4 * b + a^3 * b + a^2 * b + a * b + a * b^2 + a * b^3 + a * b^4 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l2192_219262


namespace NUMINAMATH_CALUDE_min_value_sum_l2192_219269

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  4*x + 9*y ≥ 25 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 1/y₀ = 1 ∧ 4*x₀ + 9*y₀ = 25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l2192_219269


namespace NUMINAMATH_CALUDE_min_students_in_class_l2192_219256

theorem min_students_in_class (b g : ℕ) : 
  (2 * b / 3 : ℚ) = (3 * g / 4 : ℚ) →
  b + g ≥ 17 ∧ 
  ∃ (b' g' : ℕ), b' + g' = 17 ∧ (2 * b' / 3 : ℚ) = (3 * g' / 4 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_min_students_in_class_l2192_219256


namespace NUMINAMATH_CALUDE_workshop_workers_l2192_219271

/-- The total number of workers in the workshop -/
def total_workers : ℕ := 15

/-- The number of technicians in the workshop -/
def num_technicians : ℕ := 5

/-- The average salary of all workers in the workshop -/
def avg_salary_all : ℚ := 700

/-- The average salary of technicians -/
def avg_salary_technicians : ℚ := 800

/-- The average salary of non-technician workers -/
def avg_salary_rest : ℚ := 650

theorem workshop_workers :
  total_workers = num_technicians + 
    (avg_salary_all * total_workers - avg_salary_technicians * num_technicians) / 
    (avg_salary_rest - avg_salary_all) := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_l2192_219271


namespace NUMINAMATH_CALUDE_leonie_cats_l2192_219270

theorem leonie_cats : ∃ n : ℚ, n = (4 / 5) * n + (4 / 5) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_leonie_cats_l2192_219270
