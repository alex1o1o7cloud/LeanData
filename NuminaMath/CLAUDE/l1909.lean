import Mathlib

namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1909_190999

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (1 / a + 4 / b) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1909_190999


namespace NUMINAMATH_CALUDE_star_three_four_l1909_190939

-- Define the * operation
def star (a b : ℝ) : ℝ := 4*a + 5*b - 2*a*b

-- Theorem statement
theorem star_three_four : star 3 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_star_three_four_l1909_190939


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1909_190910

/-- Given a geometric sequence {aₙ} where a₁ + a₂ + a₃ = 1 and a₂ + a₃ + a₄ = 2,
    prove that a₈ + a₉ + a₁₀ = 128 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) →  -- {aₙ} is a geometric sequence
  a 1 + a 2 + a 3 = 1 →                      -- First condition
  a 2 + a 3 + a 4 = 2 →                      -- Second condition
  a 8 + a 9 + a 10 = 128 :=                  -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1909_190910


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l1909_190953

theorem fraction_ratio_equality : ∃ (X Y : ℚ), (X / Y) / (2 / 6) = (1 / 2) / (1 / 2) → X / Y = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l1909_190953


namespace NUMINAMATH_CALUDE_dr_jones_remaining_money_l1909_190947

theorem dr_jones_remaining_money :
  let monthly_earnings : ℕ := 6000
  let house_rental : ℕ := 640
  let food_expense : ℕ := 380
  let electric_water_bill : ℕ := monthly_earnings / 4
  let insurance_cost : ℕ := monthly_earnings / 5
  let total_expenses : ℕ := house_rental + food_expense + electric_water_bill + insurance_cost
  let remaining_money : ℕ := monthly_earnings - total_expenses
  remaining_money = 2280 := by
  sorry

end NUMINAMATH_CALUDE_dr_jones_remaining_money_l1909_190947


namespace NUMINAMATH_CALUDE_stratified_sampling_result_count_l1909_190931

def junior_population : ℕ := 400
def senior_population : ℕ := 200
def total_sample_size : ℕ := 60

def stratified_proportional_sample_count (n1 n2 k : ℕ) : ℕ :=
  Nat.choose n1 ((k * n1) / (n1 + n2)) * Nat.choose n2 ((k * n2) / (n1 + n2))

theorem stratified_sampling_result_count :
  stratified_proportional_sample_count junior_population senior_population total_sample_size =
  Nat.choose junior_population 40 * Nat.choose senior_population 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_result_count_l1909_190931


namespace NUMINAMATH_CALUDE_factorial_ten_base_twelve_zeros_l1909_190996

theorem factorial_ten_base_twelve_zeros (n : ℕ) (h : n = 10) :
  ∃ k : ℕ, k = 4 ∧ 12^k ∣ n! ∧ ¬(12^(k+1) ∣ n!) :=
sorry

end NUMINAMATH_CALUDE_factorial_ten_base_twelve_zeros_l1909_190996


namespace NUMINAMATH_CALUDE_max_a_items_eleven_a_items_possible_l1909_190972

/-- Represents the number of items purchased for each stationery type -/
structure Stationery where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total cost of the stationery purchase -/
def totalCost (s : Stationery) : ℕ :=
  3 * s.a + 2 * s.b + s.c

/-- Checks if the purchase satisfies all conditions -/
def isValidPurchase (s : Stationery) : Prop :=
  s.b = s.a - 2 ∧
  3 * s.a ≤ 33 ∧
  totalCost s = 66

/-- Theorem stating that the maximum number of A items that can be purchased is 11 -/
theorem max_a_items : ∀ s : Stationery, isValidPurchase s → s.a ≤ 11 :=
  sorry

/-- Theorem stating that 11 A items can actually be purchased -/
theorem eleven_a_items_possible : ∃ s : Stationery, isValidPurchase s ∧ s.a = 11 :=
  sorry

end NUMINAMATH_CALUDE_max_a_items_eleven_a_items_possible_l1909_190972


namespace NUMINAMATH_CALUDE_projectile_max_height_l1909_190925

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 161

/-- Theorem stating that the maximum height of the projectile is 161 meters -/
theorem projectile_max_height : 
  ∀ t : ℝ, h t ≤ max_height :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l1909_190925


namespace NUMINAMATH_CALUDE_increasing_function_implies_a_geq_neg_three_l1909_190998

/-- A function f(x) = x^2 + 2(a-1)x is increasing on [4, +∞) -/
def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ≥ 4 → y ≥ 4 → x < y → x^2 + 2*(a-1)*x < y^2 + 2*(a-1)*y

/-- If f(x) = x^2 + 2(a-1)x is increasing on [4, +∞), then a ≥ -3 -/
theorem increasing_function_implies_a_geq_neg_three :
  ∀ a : ℝ, is_increasing_on_interval a → a ≥ -3 :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_implies_a_geq_neg_three_l1909_190998


namespace NUMINAMATH_CALUDE_train_platform_lengths_l1909_190903

/-- Two trains with different constant velocities passing platforms -/
theorem train_platform_lengths 
  (V1 V2 L1 L2 T1 T2 : ℝ) 
  (h_diff_vel : V1 ≠ V2) 
  (h_pos_V1 : V1 > 0) 
  (h_pos_V2 : V2 > 0)
  (h_pos_T1 : T1 > 0)
  (h_pos_T2 : T2 > 0)
  (h_L1 : L1 = V1 * T1)
  (h_L2 : L2 = V2 * T2) :
  ∃ (P1 P2 : ℝ), 
    P1 = 3 * V1 * T1 ∧ 
    P2 = 2 * V2 * T2 ∧
    V1 * (4 * T1) = L1 + P1 ∧
    V2 * (3 * T2) = L2 + P2 := by
  sorry


end NUMINAMATH_CALUDE_train_platform_lengths_l1909_190903


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_difference_bound_l1909_190967

theorem arithmetic_geometric_mean_difference_bound (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a < b) : 
  (a + b) / 2 - Real.sqrt (a * b) < (b - a)^2 / (8 * a) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_difference_bound_l1909_190967


namespace NUMINAMATH_CALUDE_percentage_difference_l1909_190983

theorem percentage_difference (x y : ℝ) (P : ℝ) : 
  x = y * 0.9 →                 -- x is 10% less than y
  y = 125 * (1 + P / 100) →     -- y is P% more than 125
  x = 123.75 →                  -- x is equal to 123.75
  P = 10 :=                     -- P is equal to 10
by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1909_190983


namespace NUMINAMATH_CALUDE_discriminant_of_5x2_minus_9x_plus_4_l1909_190902

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Proof that the discriminant of 5x^2 - 9x + 4 is 1 -/
theorem discriminant_of_5x2_minus_9x_plus_4 :
  discriminant 5 (-9) 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_5x2_minus_9x_plus_4_l1909_190902


namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_odds_with_product_l1909_190937

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def consecutive_odds (a b c d e : ℕ) : Prop :=
  is_odd a ∧ is_odd b ∧ is_odd c ∧ is_odd d ∧ is_odd e ∧
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2 ∧ e = d + 2

theorem smallest_of_five_consecutive_odds_with_product (a b c d e : ℕ) :
  consecutive_odds a b c d e →
  a * b * c * d * e = 135135 →
  a = 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_odds_with_product_l1909_190937


namespace NUMINAMATH_CALUDE_factors_4k_plus_1_ge_4k_plus_3_infinitely_many_equal_factors_infinitely_many_more_4k_plus_1_factors_l1909_190984

/-- The number of prime factors of n of the form 4k+1 -/
def f (n : ℕ+) : ℕ := sorry

/-- The number of prime factors of n of the form 4k+3 -/
def g (n : ℕ+) : ℕ := sorry

/-- For every positive integer, the number of factors of the form 4k+1 is at least as many as the number of factors of the form 4k+3 -/
theorem factors_4k_plus_1_ge_4k_plus_3 : ∀ n : ℕ+, f n ≥ g n := by sorry

/-- There are infinitely many positive integers for which the number of factors of the form 4k+1 is equal to the number of factors of the form 4k+3 -/
theorem infinitely_many_equal_factors : ∃ S : Set ℕ+, Set.Infinite S ∧ ∀ n ∈ S, f n = g n := by sorry

/-- There are infinitely many positive integers for which the number of factors of the form 4k+1 is greater than the number of factors of the form 4k+3 -/
theorem infinitely_many_more_4k_plus_1_factors : ∃ S : Set ℕ+, Set.Infinite S ∧ ∀ n ∈ S, f n > g n := by sorry

end NUMINAMATH_CALUDE_factors_4k_plus_1_ge_4k_plus_3_infinitely_many_equal_factors_infinitely_many_more_4k_plus_1_factors_l1909_190984


namespace NUMINAMATH_CALUDE_james_toys_problem_l1909_190987

theorem james_toys_problem (sell_percentage : Real) (buy_price : Real) (sell_price : Real) (total_profit : Real) :
  sell_percentage = 0.8 →
  buy_price = 20 →
  sell_price = 30 →
  total_profit = 800 →
  ∃ initial_toys : Real, initial_toys = 100 ∧ 
    sell_percentage * initial_toys * (sell_price - buy_price) = total_profit := by
  sorry

end NUMINAMATH_CALUDE_james_toys_problem_l1909_190987


namespace NUMINAMATH_CALUDE_function_decomposition_l1909_190982

/-- Given a function φ: ℝ³ → ℝ and two functions f, g: ℝ² → ℝ satisfying certain conditions,
    prove the existence of a function h: ℝ → ℝ with a specific property. -/
theorem function_decomposition
  (φ : ℝ → ℝ → ℝ → ℝ)
  (f g : ℝ → ℝ → ℝ)
  (h1 : ∀ x y z, φ x y z = f (x + y) z)
  (h2 : ∀ x y z, φ x y z = g x (y + z)) :
  ∃ h : ℝ → ℝ, ∀ x y z, φ x y z = h (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_function_decomposition_l1909_190982


namespace NUMINAMATH_CALUDE_range_of_x_range_of_m_l1909_190911

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 6*x + 8 < 0
def q (x m : ℝ) : Prop := m - 2 < x ∧ x < m + 1

-- Theorem for the range of x when p is true
theorem range_of_x : Set.Ioo 2 4 = {x : ℝ | p x} := by sorry

-- Theorem for the range of m when p is a sufficient condition for q
theorem range_of_m : 
  (∀ x, p x → ∃ m, q x m) → 
  Set.Icc 3 4 = {m : ℝ | ∀ x, p x → q x m} := by sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_m_l1909_190911


namespace NUMINAMATH_CALUDE_comic_book_frames_l1909_190961

/-- The number of frames per page in Julian's comic book -/
def frames_per_page : ℝ := 143.0

/-- The number of pages in Julian's comic book -/
def pages : ℝ := 11.0

/-- The total number of frames in Julian's comic book -/
def total_frames : ℝ := frames_per_page * pages

theorem comic_book_frames :
  total_frames = 1573.0 := by sorry

end NUMINAMATH_CALUDE_comic_book_frames_l1909_190961


namespace NUMINAMATH_CALUDE_cupcakes_frosted_l1909_190920

def cagney_rate : ℚ := 1 / 24
def lacey_rate : ℚ := 1 / 30
def casey_rate : ℚ := 1 / 40
def working_time : ℕ := 6 * 60  -- 6 minutes in seconds

theorem cupcakes_frosted :
  (cagney_rate + lacey_rate + casey_rate) * working_time = 36 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_frosted_l1909_190920


namespace NUMINAMATH_CALUDE_special_geometric_sequence_ratio_l1909_190926

/-- A geometric sequence with a special property -/
def SpecialGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) ∧ a 1 + a 5 = a 1 * a 5

/-- The ratio of the 13th to the 9th term is 9 -/
theorem special_geometric_sequence_ratio
  (a : ℕ → ℝ) (h : SpecialGeometricSequence a) :
  a 13 / a 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_special_geometric_sequence_ratio_l1909_190926


namespace NUMINAMATH_CALUDE_k_range_for_equation_solution_l1909_190932

theorem k_range_for_equation_solution (k : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ k * 4^x - k * 2^(x + 1) + 6 * (k - 5) = 0) →
  k ∈ Set.Icc 5 6 :=
by sorry

end NUMINAMATH_CALUDE_k_range_for_equation_solution_l1909_190932


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_divisible_by_12_l1909_190974

theorem consecutive_integers_sum_divisible_by_12 (a b c d : ℤ) :
  (b = a + 1) → (c = b + 1) → (d = c + 1) →
  ∃ k : ℤ, ab + ac + ad + bc + bd + cd + 1 = 12 * k :=
by
  sorry
where
  ab := a * b
  ac := a * c
  ad := a * d
  bc := b * c
  bd := b * d
  cd := c * d

end NUMINAMATH_CALUDE_consecutive_integers_sum_divisible_by_12_l1909_190974


namespace NUMINAMATH_CALUDE_fill_2004_2006_not_fill_2005_2006_l1909_190997

/-- Represents the possible marble placement patterns -/
inductive Pattern
| L
| T
| S
| I

/-- Represents a table with marbles -/
structure MarbleTable (m n : ℕ) where
  grid : Matrix (Fin m) (Fin n) ℕ

/-- Defines the operation of placing marbles according to a pattern -/
def place_marbles (t : MarbleTable m n) (p : Pattern) (i j : ℕ) : MarbleTable m n :=
  sorry

/-- Checks if all squares in the table have the same number of marbles -/
def all_squares_equal (t : MarbleTable m n) : Prop :=
  sorry

/-- Theorem: For a 2004 × 2006 table, it's possible to fill all squares equally -/
theorem fill_2004_2006 :
  ∃ (t : MarbleTable 2004 2006), all_squares_equal t :=
sorry

/-- Theorem: For a 2005 × 2006 table, it's impossible to fill all squares equally -/
theorem not_fill_2005_2006 :
  ¬∃ (t : MarbleTable 2005 2006), all_squares_equal t :=
sorry

end NUMINAMATH_CALUDE_fill_2004_2006_not_fill_2005_2006_l1909_190997


namespace NUMINAMATH_CALUDE_houses_not_yellow_l1909_190906

theorem houses_not_yellow (green yellow red : ℕ) : 
  green = 3 * yellow →
  yellow + 40 = red →
  green = 90 →
  green + red = 160 :=
by sorry

end NUMINAMATH_CALUDE_houses_not_yellow_l1909_190906


namespace NUMINAMATH_CALUDE_trigonometric_problem_l1909_190942

theorem trigonometric_problem (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.sin α + Real.cos α = 1/5) :
  (Real.sin α - Real.cos α = 7/5) ∧
  (Real.sin (2 * α + π/3) = -12/25 - 7 * Real.sqrt 3 / 50) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l1909_190942


namespace NUMINAMATH_CALUDE_inequality_and_constraint_solution_l1909_190978

-- Define the inequality and its solution set
def inequality (a : ℝ) (x : ℝ) : Prop := 2 * a * x^2 - 8 * x - 3 * a^2 < 0
def solution_set (a b : ℝ) : Set ℝ := {x | -1 < x ∧ x < b ∧ inequality a x}

-- Define the constraint equation
def constraint (a b x y : ℝ) : Prop := a / x + b / y = 1

-- State the theorem
theorem inequality_and_constraint_solution :
  ∃ (a b : ℝ),
    (∀ x, x ∈ solution_set a b ↔ inequality a x) ∧
    a > 0 ∧
    (∀ x y, x > 0 → y > 0 → constraint a b x y →
      3 * x + 2 * y ≥ 24 ∧
      (∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ constraint a b x₀ y₀ ∧ 3 * x₀ + 2 * y₀ = 24)) ∧
    a = 2 ∧
    b = 3 :=
  sorry

end NUMINAMATH_CALUDE_inequality_and_constraint_solution_l1909_190978


namespace NUMINAMATH_CALUDE_second_number_is_sixteen_l1909_190944

theorem second_number_is_sixteen (first_number second_number third_number : ℤ) : 
  first_number = 17 →
  third_number = 20 →
  3 * first_number + 3 * second_number + 3 * third_number + 11 = 170 →
  second_number = 16 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_sixteen_l1909_190944


namespace NUMINAMATH_CALUDE_derivative_zero_necessary_not_sufficient_l1909_190941

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Define what it means for a function to be differentiable
def IsDifferentiable (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f

-- Define what it means for a point to be an extremum
def IsExtremum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

-- Theorem stating that f'(x) = 0 is necessary but not sufficient for x to be an extremum
theorem derivative_zero_necessary_not_sufficient (h : IsDifferentiable f) :
  (IsExtremum f x → deriv f x = 0) ∧
  ¬(deriv f x = 0 → IsExtremum f x) :=
sorry

end NUMINAMATH_CALUDE_derivative_zero_necessary_not_sufficient_l1909_190941


namespace NUMINAMATH_CALUDE_equation_satisfied_when_m_is_34_l1909_190913

theorem equation_satisfied_when_m_is_34 :
  let m : ℕ := 34
  (((1 : ℚ) ^ (m + 1)) / ((5 : ℚ) ^ (m + 1))) * (((1 : ℚ) ^ 18) / ((4 : ℚ) ^ 18)) = 1 / (2 * ((10 : ℚ) ^ 35)) :=
by sorry

end NUMINAMATH_CALUDE_equation_satisfied_when_m_is_34_l1909_190913


namespace NUMINAMATH_CALUDE_collinear_vectors_sum_l1909_190989

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ m : ℝ, b = (m * a.1, m * a.2.1, m * a.2.2)

/-- The problem statement -/
theorem collinear_vectors_sum (x y : ℝ) :
  let a : ℝ × ℝ × ℝ := (x, 2, 2)
  let b : ℝ × ℝ × ℝ := (2, y, 4)
  collinear a b → x + y = 5 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_sum_l1909_190989


namespace NUMINAMATH_CALUDE_probability_penny_dime_heads_l1909_190968

-- Define the coin flip experiment
def coin_flip_experiment : ℕ := 5

-- Define the number of coins we're interested in (penny and dime)
def target_coins : ℕ := 2

-- Define the probability of a single coin coming up heads
def prob_heads : ℚ := 1/2

-- Theorem statement
theorem probability_penny_dime_heads :
  (prob_heads ^ target_coins) * (2 ^ (coin_flip_experiment - target_coins)) / (2 ^ coin_flip_experiment) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_probability_penny_dime_heads_l1909_190968


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_subsequence_l1909_190992

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (x y z : ℝ) : Prop :=
  y / x = z / y

theorem arithmetic_sequence_with_geometric_subsequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_first : a 1 = 1)
  (h_geom : is_geometric_sequence (a 1) (a 3) (a 9)) :
  (∀ n : ℕ, a n = 1) ∨ (∀ n : ℕ, a n = n) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_subsequence_l1909_190992


namespace NUMINAMATH_CALUDE_function_equality_implies_a_value_l1909_190965

open Real

theorem function_equality_implies_a_value (a : ℝ) :
  (∃ x₀ : ℝ, x₀ + exp (x₀ - a) - (log (x₀ + 2) - 4 * exp (a - x₀)) = 3) →
  a = -log 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_function_equality_implies_a_value_l1909_190965


namespace NUMINAMATH_CALUDE_walter_school_allocation_l1909_190995

/-- Represents Walter's work and allocation details -/
structure WorkDetails where
  daysPerWeek : ℕ
  hourlyWage : ℚ
  hoursPerDay : ℕ
  allocationRatio : ℚ

/-- Calculates the amount allocated for school based on work details -/
def schoolAllocation (w : WorkDetails) : ℚ :=
  w.daysPerWeek * w.hourlyWage * w.hoursPerDay * w.allocationRatio

/-- Theorem stating that Walter's school allocation is $75 -/
theorem walter_school_allocation :
  let w : WorkDetails := {
    daysPerWeek := 5,
    hourlyWage := 5,
    hoursPerDay := 4,
    allocationRatio := 3/4
  }
  schoolAllocation w = 75 := by sorry

end NUMINAMATH_CALUDE_walter_school_allocation_l1909_190995


namespace NUMINAMATH_CALUDE_vasyas_capital_decreases_l1909_190930

/-- Represents the change in Vasya's capital after a series of trading days -/
def vasyas_capital_change (num_unsuccessful_days : ℕ) : ℝ :=
  (1.1^2 * 0.8)^num_unsuccessful_days

/-- Theorem stating that Vasya's capital decreases -/
theorem vasyas_capital_decreases (num_unsuccessful_days : ℕ) :
  vasyas_capital_change num_unsuccessful_days < 1 := by
  sorry

#check vasyas_capital_decreases

end NUMINAMATH_CALUDE_vasyas_capital_decreases_l1909_190930


namespace NUMINAMATH_CALUDE_a_less_than_one_l1909_190966

/-- The sequence a_n defined recursively -/
def a (k : ℕ) : ℕ → ℚ
  | 0 => 1 / k
  | n + 1 => a k n + (1 / (n + 1)^2) * (a k n)^2

/-- The theorem stating the condition for a_n < 1 for all n -/
theorem a_less_than_one (k : ℕ) : (∀ n : ℕ, a k n < 1) ↔ k ≥ 3 := by sorry

end NUMINAMATH_CALUDE_a_less_than_one_l1909_190966


namespace NUMINAMATH_CALUDE_sum_of_specific_S_values_l1909_190959

-- Define the sequence Sn
def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

-- State the theorem
theorem sum_of_specific_S_values : S 19 + S 37 + S 52 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_S_values_l1909_190959


namespace NUMINAMATH_CALUDE_go_stones_perimeter_l1909_190970

/-- Calculates the number of stones on the perimeter of a rectangle made of Go stones -/
def perimeter_stones (width : ℕ) (height : ℕ) : ℕ :=
  2 * (width + height) - 4

theorem go_stones_perimeter :
  let width : ℕ := 4
  let height : ℕ := 8
  perimeter_stones width height = 20 := by sorry

end NUMINAMATH_CALUDE_go_stones_perimeter_l1909_190970


namespace NUMINAMATH_CALUDE_geometric_mean_of_one_and_four_l1909_190993

theorem geometric_mean_of_one_and_four :
  ∀ x : ℝ, x^2 = 1 * 4 → x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_one_and_four_l1909_190993


namespace NUMINAMATH_CALUDE_special_ellipse_properties_l1909_190918

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_angle : c / a = Real.sqrt 3 / 2
  h_dist : a + c = 2 + Real.sqrt 3

/-- The line passing through a focus of the ellipse -/
structure FocusLine where
  m : ℝ

/-- The theorem statement -/
theorem special_ellipse_properties (e : SpecialEllipse) (l : FocusLine) :
  (e.a = 2 ∧ e.b = 1) ∧
  (l.m = Real.sqrt 2 ∨ l.m = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_properties_l1909_190918


namespace NUMINAMATH_CALUDE_line_through_three_points_l1909_190938

/-- Given a line containing points (0, 5), (7, k), and (25, 2), prove that k = 104/25 -/
theorem line_through_three_points (k : ℝ) : 
  (∃ m b : ℝ, (0 = m * 0 + b ∧ 5 = m * 0 + b) ∧ 
              (k = m * 7 + b) ∧ 
              (2 = m * 25 + b)) → 
  k = 104 / 25 := by
sorry

end NUMINAMATH_CALUDE_line_through_three_points_l1909_190938


namespace NUMINAMATH_CALUDE_negation_of_square_positivity_l1909_190909

theorem negation_of_square_positivity :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, ¬(x^2 > 0)) :=
sorry

end NUMINAMATH_CALUDE_negation_of_square_positivity_l1909_190909


namespace NUMINAMATH_CALUDE_sequence_sum_l1909_190904

theorem sequence_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l1909_190904


namespace NUMINAMATH_CALUDE_largest_power_of_ten_in_factorial_l1909_190943

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def count_multiples (n : ℕ) (d : ℕ) : ℕ := n / d

def count_factors_of_five (n : ℕ) : ℕ :=
  (count_multiples n 5) + (count_multiples n 25) + (count_multiples n 125)

theorem largest_power_of_ten_in_factorial :
  (∀ k : ℕ, k ≤ 41 → (factorial 170) % (10^k) = 0) ∧
  ¬((factorial 170) % (10^42) = 0) := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_ten_in_factorial_l1909_190943


namespace NUMINAMATH_CALUDE_existence_of_sequence_l1909_190946

theorem existence_of_sequence (α : ℝ) (n : ℕ) (h_α : 0 < α ∧ α < 1) (h_n : 0 < n) :
  ∃ (a : ℕ → ℕ), 
    (∀ i ∈ Finset.range n, 1 ≤ a i) ∧
    (∀ i ∈ Finset.range (n-1), a i < a (i+1)) ∧
    (∀ i ∈ Finset.range n, a i ≤ 2^(n-1)) ∧
    (∀ i ∈ Finset.range (n-1), ⌊(α^(i+1) : ℝ) * (a (i+1) : ℝ)⌋ ≥ ⌊(α^i : ℝ) * (a i : ℝ)⌋) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_sequence_l1909_190946


namespace NUMINAMATH_CALUDE_angle_between_clock_hands_at_9_15_angle_between_clock_hands_at_9_15_is_82_point_5_l1909_190994

/-- The angle between clock hands at 9:15 --/
theorem angle_between_clock_hands_at_9_15 : ℝ :=
  let full_rotation : ℝ := 360
  let hours_on_clock_face : ℕ := 12
  let minutes_on_clock_face : ℕ := 60
  let current_hour : ℕ := 9
  let current_minute : ℕ := 15

  let angle_per_hour : ℝ := full_rotation / hours_on_clock_face
  let angle_per_minute : ℝ := full_rotation / minutes_on_clock_face

  let minute_hand_angle : ℝ := current_minute * angle_per_minute
  let hour_hand_angle : ℝ := current_hour * angle_per_hour + (current_minute * angle_per_hour / minutes_on_clock_face)

  let angle_between_hands : ℝ := abs (minute_hand_angle - hour_hand_angle)

  82.5

theorem angle_between_clock_hands_at_9_15_is_82_point_5 :
  angle_between_clock_hands_at_9_15 = 82.5 := by sorry

end NUMINAMATH_CALUDE_angle_between_clock_hands_at_9_15_angle_between_clock_hands_at_9_15_is_82_point_5_l1909_190994


namespace NUMINAMATH_CALUDE_tan_45_degrees_l1909_190922

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l1909_190922


namespace NUMINAMATH_CALUDE_power_of_product_equals_power_l1909_190954

theorem power_of_product_equals_power (n : ℕ) : 3^12 * 3^18 = 243^6 := by sorry

end NUMINAMATH_CALUDE_power_of_product_equals_power_l1909_190954


namespace NUMINAMATH_CALUDE_extended_annuity_period_is_34_l1909_190985

/-- Calculates the extended annuity period given the initial parameters -/
def extended_annuity_period (initial_period : ℕ) (delay : ℕ) (interest_rate : ℚ) (annuity_payment : ℕ) : ℕ :=
  34

/-- Theorem stating that the extended annuity period is 34 years given the initial conditions -/
theorem extended_annuity_period_is_34 :
  extended_annuity_period 26 3 (4.5 / 100) 5000 = 34 := by
  sorry

end NUMINAMATH_CALUDE_extended_annuity_period_is_34_l1909_190985


namespace NUMINAMATH_CALUDE_circle_plus_four_three_l1909_190928

-- Define the operation ⊕
def circle_plus (a b : ℚ) : ℚ := a * (1 + a / b^2)

-- Theorem statement
theorem circle_plus_four_three : circle_plus 4 3 = 52 / 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_four_three_l1909_190928


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l1909_190955

theorem smallest_sum_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15 → 
  ∀ a b : ℕ+, 
    a ≠ b → 
    (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → 
    (x : ℤ) + y ≤ (a : ℤ) + b :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l1909_190955


namespace NUMINAMATH_CALUDE_inequality_sign_change_l1909_190908

theorem inequality_sign_change (a b : ℝ) (c : ℝ) (h1 : c < 0) (h2 : a < b) : c * b < c * a := by
  sorry

end NUMINAMATH_CALUDE_inequality_sign_change_l1909_190908


namespace NUMINAMATH_CALUDE_sandra_savings_l1909_190964

-- Define the number of notepads
def num_notepads : ℕ := 8

-- Define the original price per notepad
def original_price : ℚ := 375 / 100

-- Define the discount rate
def discount_rate : ℚ := 25 / 100

-- Define the savings calculation
def savings : ℚ :=
  num_notepads * original_price - num_notepads * (original_price * (1 - discount_rate))

-- Theorem to prove
theorem sandra_savings : savings = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sandra_savings_l1909_190964


namespace NUMINAMATH_CALUDE_wrong_number_calculation_l1909_190945

theorem wrong_number_calculation (n : ℕ) (initial_avg correct_avg correct_num wrong_num : ℚ) : 
  n = 10 ∧ 
  initial_avg = 15 ∧ 
  correct_avg = 16 ∧ 
  correct_num = 36 → 
  (n : ℚ) * correct_avg - (n : ℚ) * initial_avg = correct_num - wrong_num →
  wrong_num = 26 := by sorry

end NUMINAMATH_CALUDE_wrong_number_calculation_l1909_190945


namespace NUMINAMATH_CALUDE_daughter_weight_l1909_190933

/-- Represents the weights of family members in kilograms -/
structure FamilyWeights where
  mother : ℝ
  daughter : ℝ
  grandchild : ℝ

/-- Conditions for the family weights problem -/
def FamilyWeightsProblem (w : FamilyWeights) : Prop :=
  w.mother + w.daughter + w.grandchild = 150 ∧
  w.daughter + w.grandchild = 60 ∧
  w.grandchild = (1 / 5) * w.mother

/-- The weight of the daughter is 42 kg given the conditions -/
theorem daughter_weight (w : FamilyWeights) 
  (h : FamilyWeightsProblem w) : w.daughter = 42 := by
  sorry

end NUMINAMATH_CALUDE_daughter_weight_l1909_190933


namespace NUMINAMATH_CALUDE_divisibility_of_polynomial_l1909_190981

theorem divisibility_of_polynomial (n : ℤ) : 
  120 ∣ (n^5 - 5*n^3 + 4*n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_polynomial_l1909_190981


namespace NUMINAMATH_CALUDE_exists_non_isosceles_with_four_equal_subtriangles_l1909_190901

/-- A triangle represented by its three vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- An interior point of a triangle -/
def InteriorPoint (t : Triangle) := ℝ × ℝ

/-- Predicate to check if a triangle is isosceles -/
def IsIsosceles (t : Triangle) : Prop := sorry

/-- Predicate to check if a point is inside a triangle -/
def IsInside (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Function to create 6 triangles by connecting an interior point to vertices and drawing perpendiculars -/
def CreateSubTriangles (t : Triangle) (p : InteriorPoint t) : List Triangle := sorry

/-- Predicate to check if 4 out of 6 triangles in a list are equal -/
def FourOutOfSixEqual (triangles : List Triangle) : Prop := sorry

/-- Theorem stating that there exists a non-isosceles triangle with an interior point
    such that 4 out of 6 resulting triangles are equal -/
theorem exists_non_isosceles_with_four_equal_subtriangles :
  ∃ (t : Triangle) (p : InteriorPoint t),
    ¬IsIsosceles t ∧
    IsInside p t ∧
    FourOutOfSixEqual (CreateSubTriangles t p) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_isosceles_with_four_equal_subtriangles_l1909_190901


namespace NUMINAMATH_CALUDE_line_points_k_value_l1909_190962

/-- A line contains the points (3, 10), (1, k), and (-7, 2). Prove that k = 8.4. -/
theorem line_points_k_value (k : ℝ) : 
  (∃ (m b : ℝ), 
    (3 * m + b = 10) ∧ 
    (1 * m + b = k) ∧ 
    (-7 * m + b = 2)) → 
  k = 8.4 := by
sorry

end NUMINAMATH_CALUDE_line_points_k_value_l1909_190962


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l1909_190957

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 20 → p.Prime → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 529) ∧
  (has_no_small_prime_factors 529) ∧
  (∀ m : ℕ, m < 529 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l1909_190957


namespace NUMINAMATH_CALUDE_triangle_to_pentagon_area_ratio_l1909_190991

/-- The ratio of the area of an isosceles right triangle to the area of a pentagon formed by the triangle and a rectangle -/
theorem triangle_to_pentagon_area_ratio :
  let triangle_leg : ℝ := 2
  let triangle_hypotenuse : ℝ := triangle_leg * Real.sqrt 2
  let rectangle_width : ℝ := triangle_hypotenuse
  let rectangle_height : ℝ := 2 * triangle_leg
  let triangle_area : ℝ := (1 / 2) * triangle_leg ^ 2
  let rectangle_area : ℝ := rectangle_width * rectangle_height
  let pentagon_area : ℝ := triangle_area + rectangle_area
  triangle_area / pentagon_area = 2 / (2 + 8 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_to_pentagon_area_ratio_l1909_190991


namespace NUMINAMATH_CALUDE_train_length_l1909_190929

/-- The length of a train given its speed, time to cross a bridge, and the bridge's length -/
theorem train_length (v : ℝ) (t : ℝ) (bridge_length : ℝ) (h1 : v = 65 * 1000 / 3600) 
    (h2 : t = 15.506451791548985) (h3 : bridge_length = 150) : 
    v * t - bridge_length = 130 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1909_190929


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1909_190950

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1909_190950


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l1909_190905

theorem product_remainder_mod_five : (14452 * 15652 * 16781) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l1909_190905


namespace NUMINAMATH_CALUDE_total_spending_is_638_l1909_190916

/-- The total spending of Elizabeth, Emma, and Elsa -/
def total_spending (emma_spending : ℕ) : ℕ :=
  let elsa_spending := 2 * emma_spending
  let elizabeth_spending := 4 * elsa_spending
  emma_spending + elsa_spending + elizabeth_spending

/-- Theorem: The total spending is $638 when Emma spent $58 -/
theorem total_spending_is_638 : total_spending 58 = 638 := by
  sorry

end NUMINAMATH_CALUDE_total_spending_is_638_l1909_190916


namespace NUMINAMATH_CALUDE_airplane_seats_l1909_190934

/-- Represents the total number of seats on an airplane -/
def total_seats : ℕ := 216

/-- Represents the number of seats in First Class -/
def first_class_seats : ℕ := 36

/-- Theorem stating that the total number of seats on the airplane is 216 -/
theorem airplane_seats :
  (first_class_seats : ℚ) + (1 : ℚ) / 3 * total_seats + (1 : ℚ) / 2 * total_seats = total_seats :=
by sorry

end NUMINAMATH_CALUDE_airplane_seats_l1909_190934


namespace NUMINAMATH_CALUDE_no_four_distinct_naturals_power_sum_equality_l1909_190919

theorem no_four_distinct_naturals_power_sum_equality :
  ¬∃ (x y z t : ℕ), x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t ∧ x^x + y^y = z^z + t^t :=
by sorry

end NUMINAMATH_CALUDE_no_four_distinct_naturals_power_sum_equality_l1909_190919


namespace NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l1909_190969

theorem least_positive_integer_with_given_remainders : ∃! N : ℕ,
  (N % 11 = 10) ∧
  (N % 12 = 11) ∧
  (N % 13 = 12) ∧
  (N % 14 = 13) ∧
  (∀ M : ℕ, M < N →
    ¬((M % 11 = 10) ∧
      (M % 12 = 11) ∧
      (M % 13 = 12) ∧
      (M % 14 = 13))) ∧
  N = 12011 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l1909_190969


namespace NUMINAMATH_CALUDE_local_minimum_condition_l1909_190935

/-- The function f(x) defined as x^3 - 3bx + b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + b

/-- Theorem stating the condition for f(x) to have a local minimum in (0,1) -/
theorem local_minimum_condition (b : ℝ) :
  (∃ x, x ∈ Set.Ioo 0 1 ∧ IsLocalMin (f b) x) ↔ b ∈ Set.Ioo 0 1 := by
  sorry

#check local_minimum_condition

end NUMINAMATH_CALUDE_local_minimum_condition_l1909_190935


namespace NUMINAMATH_CALUDE_sphere_radius_equal_cylinder_surface_area_l1909_190975

/-- The radius of a sphere given that its surface area is equal to the curved surface area of a right circular cylinder with height and diameter both 6 cm -/
theorem sphere_radius_equal_cylinder_surface_area (h : ℝ) (d : ℝ) (r : ℝ) : 
  h = 6 →
  d = 6 →
  4 * Real.pi * r^2 = 2 * Real.pi * (d/2) * h →
  r = 3 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_equal_cylinder_surface_area_l1909_190975


namespace NUMINAMATH_CALUDE_sin_angle_through_point_l1909_190980

theorem sin_angle_through_point (α : Real) :
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 2 ∧ r * Real.sin α = -1) →
  Real.sin α = -Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_angle_through_point_l1909_190980


namespace NUMINAMATH_CALUDE_inequality_range_l1909_190973

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ≥ -1 → Real.log (x + 2) + a * (x^2 + x) ≥ 0) ↔ 0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1909_190973


namespace NUMINAMATH_CALUDE_power_equation_solution_l1909_190940

theorem power_equation_solution (n : ℕ) : 
  2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^22 → n = 21 := by
sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1909_190940


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1909_190979

/-- A rectangular parallelepiped with edge lengths 1, 2, and 3 -/
structure Parallelepiped where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ
  edge1_eq : edge1 = 1
  edge2_eq : edge2 = 2
  edge3_eq : edge3 = 3

/-- A sphere containing all vertices of a rectangular parallelepiped -/
structure Sphere where
  radius : ℝ
  contains_parallelepiped : Parallelepiped → Prop

/-- The surface area of a sphere is 14π given the conditions -/
theorem sphere_surface_area (s : Sphere) (p : Parallelepiped) 
  (h : s.contains_parallelepiped p) : 
  s.radius^2 * (4 * Real.pi) = 14 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1909_190979


namespace NUMINAMATH_CALUDE_harkamal_fruit_purchase_cost_l1909_190960

/-- The total cost of fruits purchased by Harkamal -/
def total_cost (grapes_kg : ℕ) (grapes_price : ℕ) 
               (mangoes_kg : ℕ) (mangoes_price : ℕ)
               (apples_kg : ℕ) (apples_price : ℕ)
               (strawberries_kg : ℕ) (strawberries_price : ℕ) : ℕ :=
  grapes_kg * grapes_price + 
  mangoes_kg * mangoes_price + 
  apples_kg * apples_price + 
  strawberries_kg * strawberries_price

/-- Theorem stating the total cost of fruits purchased by Harkamal -/
theorem harkamal_fruit_purchase_cost : 
  total_cost 8 70 9 45 5 30 3 100 = 1415 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_fruit_purchase_cost_l1909_190960


namespace NUMINAMATH_CALUDE_circumradii_ratio_eq_side_ratio_l1909_190924

/-- Represents the properties of an equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  perimeter : ℝ
  area : ℝ
  circumradius : ℝ
  perimeter_eq : perimeter = 3 * side
  area_eq : area = (side^2 * Real.sqrt 3) / 4
  circumradius_eq : circumradius = (side * Real.sqrt 3) / 3

/-- Theorem stating the relationship between circumradii of two equilateral triangles -/
theorem circumradii_ratio_eq_side_ratio 
  (n m : ℝ) 
  (fore back : EquilateralTriangle) 
  (h_perimeter_ratio : fore.perimeter / back.perimeter = n / m)
  (h_area_ratio : fore.area / back.area = n / m) :
  fore.circumradius / back.circumradius = fore.side / back.side := by
  sorry

#check circumradii_ratio_eq_side_ratio

end NUMINAMATH_CALUDE_circumradii_ratio_eq_side_ratio_l1909_190924


namespace NUMINAMATH_CALUDE_prob_of_specific_sums_is_five_eighteenths_l1909_190976

/-- Represents the faces of a die -/
def Die := List Nat

/-- The first die with faces 1, 3, 3, 5, 5, 7 -/
def die1 : Die := [1, 3, 3, 5, 5, 7]

/-- The second die with faces 2, 4, 4, 6, 6, 8 -/
def die2 : Die := [2, 4, 4, 6, 6, 8]

/-- Calculates the probability of a specific sum occurring when rolling two dice -/
def probOfSum (d1 d2 : Die) (sum : Nat) : Rat :=
  sorry

/-- Calculates the probability of the sum being 8, 10, or 12 when rolling the two specified dice -/
def probOfSpecificSums (d1 d2 : Die) : Rat :=
  (probOfSum d1 d2 8) + (probOfSum d1 d2 10) + (probOfSum d1 d2 12)

/-- Theorem stating that the probability of getting a sum of 8, 10, or 12 with the given dice is 5/18 -/
theorem prob_of_specific_sums_is_five_eighteenths :
  probOfSpecificSums die1 die2 = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_prob_of_specific_sums_is_five_eighteenths_l1909_190976


namespace NUMINAMATH_CALUDE_joan_football_games_l1909_190988

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := sorry

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := 9

/-- The total number of football games Joan went to -/
def total_games : ℕ := 13

theorem joan_football_games : games_this_year = 4 := by sorry

end NUMINAMATH_CALUDE_joan_football_games_l1909_190988


namespace NUMINAMATH_CALUDE_perfect_square_prime_exponents_l1909_190907

theorem perfect_square_prime_exponents (p q r : Nat) : 
  Prime p ∧ Prime q ∧ Prime r → 
  (∃ (n : Nat), p^q + p^r = n^2) ↔ 
  ((p = 2 ∧ q = 2 ∧ r = 5) ∨ 
   (p = 2 ∧ q = 5 ∧ r = 2) ∨ 
   (p = 3 ∧ q = 2 ∧ r = 3) ∨ 
   (p = 3 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 2 ∧ q = r ∧ q ≥ 3 ∧ Odd q)) := by
  sorry

#check perfect_square_prime_exponents

end NUMINAMATH_CALUDE_perfect_square_prime_exponents_l1909_190907


namespace NUMINAMATH_CALUDE_minimum_age_vasily_l1909_190958

theorem minimum_age_vasily (n : ℕ) (h_n : n = 64) :
  ∃ (V F : ℕ),
    V = F + 2 ∧
    F ≥ 5 ∧
    (∀ k : ℕ, k ≥ F → Nat.choose n k > Nat.choose n (k + 2)) ∧
    (∀ V' F' : ℕ, V' = F' + 2 → F' ≥ 5 → 
      (∀ k : ℕ, k ≥ F' → Nat.choose n k > Nat.choose n (k + 2)) → V' ≥ V) ∧
    V = 34 := by
  sorry

end NUMINAMATH_CALUDE_minimum_age_vasily_l1909_190958


namespace NUMINAMATH_CALUDE_only_five_students_l1909_190971

/-- Represents the number of students -/
def n : ℕ := sorry

/-- Represents the total number of problems solved -/
def S : ℕ := sorry

/-- Represents the number of problems solved by one student -/
def a : ℕ := sorry

/-- The condition that each student solved more than one-fifth of the problems solved by others -/
axiom condition1 : a > (S - a) / 5

/-- The condition that each student solved less than one-third of the problems solved by others -/
axiom condition2 : a < (S - a) / 3

/-- The total number of problems is the sum of problems solved by all students -/
axiom total_problems : S = n * a

/-- The theorem stating that the only possible number of students is 5 -/
theorem only_five_students : n = 5 := by sorry

end NUMINAMATH_CALUDE_only_five_students_l1909_190971


namespace NUMINAMATH_CALUDE_ABCD_equals_one_l1909_190956

theorem ABCD_equals_one :
  let A := Real.sqrt 3003 + Real.sqrt 3004
  let B := -Real.sqrt 3003 - Real.sqrt 3004
  let C := Real.sqrt 3003 - Real.sqrt 3004
  let D := Real.sqrt 3004 - Real.sqrt 3003
  A * B * C * D = 1 := by
  sorry

end NUMINAMATH_CALUDE_ABCD_equals_one_l1909_190956


namespace NUMINAMATH_CALUDE_three_pipes_fill_time_l1909_190914

/-- Represents the time taken to fill a tank given a number of pipes -/
def fill_time (num_pipes : ℕ) (time : ℝ) : Prop :=
  num_pipes > 0 ∧ time > 0 ∧ num_pipes * time = 36

theorem three_pipes_fill_time :
  fill_time 2 18 → fill_time 3 12 := by
  sorry

end NUMINAMATH_CALUDE_three_pipes_fill_time_l1909_190914


namespace NUMINAMATH_CALUDE_binary_search_sixteen_people_l1909_190952

/-- The number of tests required to identify one infected person in a group of size n using binary search. -/
def numTestsBinarySearch (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else 1 + numTestsBinarySearch (n / 2)

/-- Theorem: For a group of 16 people with one infected person, 4 tests are required using binary search. -/
theorem binary_search_sixteen_people :
  numTestsBinarySearch 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_binary_search_sixteen_people_l1909_190952


namespace NUMINAMATH_CALUDE_faye_bought_30_songs_l1909_190948

/-- The number of songs Faye bought -/
def total_songs (country_albums pop_albums songs_per_album : ℕ) : ℕ :=
  (country_albums + pop_albums) * songs_per_album

/-- Proof that Faye bought 30 songs -/
theorem faye_bought_30_songs :
  total_songs 2 3 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_faye_bought_30_songs_l1909_190948


namespace NUMINAMATH_CALUDE_board_cutting_l1909_190917

theorem board_cutting (total_length shorter_length : ℝ) 
  (h1 : total_length = 120)
  (h2 : shorter_length = 35)
  (h3 : ∃ longer_length, longer_length + shorter_length = total_length ∧ 
    ∃ x, longer_length = 2 * shorter_length + x) :
  ∃ longer_length x, 
    longer_length + shorter_length = total_length ∧ 
    longer_length = 2 * shorter_length + x ∧ 
    x = 15 :=
by sorry

end NUMINAMATH_CALUDE_board_cutting_l1909_190917


namespace NUMINAMATH_CALUDE_value_exceeds_initial_in_7th_year_min_avg_value_at_4th_year_l1909_190949

-- Define the value of M at the beginning of the nth year
def value (n : ℕ) : ℚ :=
  if n ≤ 3 then
    20 * (1/2)^(n-1)
  else
    4*n - 7

-- Define the sum of values for the first n years
def sum_values (n : ℕ) : ℚ :=
  if n ≤ 3 then
    40 - 5 * 2^(3-n)
  else
    2*n^2 - 5*n + 32

-- Define the average value over n years
def avg_value (n : ℕ) : ℚ :=
  sum_values n / n

-- Theorem 1: The value exceeds the initial price in the 7th year
theorem value_exceeds_initial_in_7th_year :
  ∀ n : ℕ, n < 7 → value n ≤ 20 ∧ value 7 > 20 :=
sorry

-- Theorem 2: The minimum average value is 11, occurring at n = 4
theorem min_avg_value_at_4th_year :
  ∀ n : ℕ, n ≥ 1 → avg_value n ≥ 11 ∧ avg_value 4 = 11 :=
sorry

end NUMINAMATH_CALUDE_value_exceeds_initial_in_7th_year_min_avg_value_at_4th_year_l1909_190949


namespace NUMINAMATH_CALUDE_expressions_not_equivalent_l1909_190963

theorem expressions_not_equivalent :
  ∃ x : ℝ, (x^2 + 1 ≠ 0 ∧ x^2 + 2*x + 1 ≠ 0) →
    (x^2 + x + 1) / (x^2 + 1) ≠ ((x + 1)^2) / (x^2 + 2*x + 1) :=
by sorry

end NUMINAMATH_CALUDE_expressions_not_equivalent_l1909_190963


namespace NUMINAMATH_CALUDE_problem_statement_l1909_190951

theorem problem_statement : 
  (3 * (0.6 * 40) - (4/5 * 25) / 2) * (Real.sqrt 16 - 3) = 62 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1909_190951


namespace NUMINAMATH_CALUDE_S_128_eq_half_l1909_190921

/-- Best decomposition of a positive integer -/
def best_decomposition (n : ℕ+) : ℕ+ × ℕ+ :=
  sorry

/-- S function for a positive integer -/
def S (n : ℕ+) : ℚ :=
  let (p, q) := best_decomposition n
  (p : ℚ) / q

/-- Theorem: S(128) = 1/2 -/
theorem S_128_eq_half : S 128 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_S_128_eq_half_l1909_190921


namespace NUMINAMATH_CALUDE_right_triangle_30_hypotenuse_l1909_190986

/-- A right triangle with one angle of 30 degrees -/
structure RightTriangle30 where
  /-- The length of side XZ -/
  xz : ℝ
  /-- XZ is positive -/
  xz_pos : 0 < xz

/-- The length of the hypotenuse in a right triangle with a 30-degree angle -/
def hypotenuse (t : RightTriangle30) : ℝ := 2 * t.xz

/-- Theorem: In a right triangle XYZ with right angle at X, if angle YZX = 30° and XZ = 15, then XY = 30 -/
theorem right_triangle_30_hypotenuse :
  ∀ t : RightTriangle30, t.xz = 15 → hypotenuse t = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_30_hypotenuse_l1909_190986


namespace NUMINAMATH_CALUDE_smallest_root_of_equation_l1909_190936

theorem smallest_root_of_equation : 
  let eq := fun x : ℝ => 2 * (x - 3 * Real.sqrt 5) * (x - 5 * Real.sqrt 3)
  ∃ (r : ℝ), eq r = 0 ∧ r = 3 * Real.sqrt 5 ∧ ∀ (s : ℝ), eq s = 0 → r ≤ s :=
by sorry

end NUMINAMATH_CALUDE_smallest_root_of_equation_l1909_190936


namespace NUMINAMATH_CALUDE_circle_line_intersection_theorem_l1909_190912

/-- Circle C with equation x^2 + (y-4)^2 = 4 -/
def C (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 4

/-- Line l with equation y = kx -/
def l (k x y : ℝ) : Prop := y = k * x

/-- Point Q(m, n) is on segment MN -/
def Q_on_MN (m n x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ m = t * x₁ + (1 - t) * x₂ ∧ n = t * y₁ + (1 - t) * y₂

/-- The condition 2/|OQ|^2 = 1/|OM|^2 + 1/|ON|^2 -/
def harmonic_condition (m n x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  2 / (m^2 + n^2) = 1 / (x₁^2 + y₁^2) + 1 / (x₂^2 + y₂^2)

theorem circle_line_intersection_theorem
  (k m n x₁ y₁ x₂ y₂ : ℝ)
  (hC₁ : C x₁ y₁)
  (hC₂ : C x₂ y₂)
  (hl₁ : l k x₁ y₁)
  (hl₂ : l k x₂ y₂)
  (hQ : Q_on_MN m n x₁ y₁ x₂ y₂)
  (hHarmonic : harmonic_condition m n x₁ y₁ x₂ y₂)
  (hm : m ∈ Set.Ioo (-Real.sqrt 3) 0 ∪ Set.Ioo 0 (Real.sqrt 3)) :
  n = Real.sqrt (15 * m^2 + 180) / 5 :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_theorem_l1909_190912


namespace NUMINAMATH_CALUDE_white_ring_weight_l1909_190900

/-- Given the weights of three plastic rings (orange, purple, and white) and their total weight,
    this theorem proves that the weight of the white ring is equal to the total weight
    minus the sum of the orange and purple ring weights. -/
theorem white_ring_weight 
  (orange_weight : ℝ) 
  (purple_weight : ℝ) 
  (total_weight : ℝ) 
  (h1 : orange_weight = 0.08333333333333333)
  (h2 : purple_weight = 0.3333333333333333)
  (h3 : total_weight = 0.8333333333) :
  total_weight - (orange_weight + purple_weight) = 0.41666666663333337 := by
  sorry

#eval Float.toString (0.8333333333 - (0.08333333333333333 + 0.3333333333333333))

end NUMINAMATH_CALUDE_white_ring_weight_l1909_190900


namespace NUMINAMATH_CALUDE_min_circle_and_common_chord_for_given_points_l1909_190927

/-- The circle with the smallest circumference passing through two given points -/
structure MinCircle where
  center : ℝ × ℝ
  radius : ℝ

/-- The common chord between two intersecting circles -/
structure CommonChord where
  length : ℝ

/-- Given points A and B, find the circle with smallest circumference passing through them
    and calculate its common chord length with another given circle -/
def find_min_circle_and_common_chord 
  (A B : ℝ × ℝ) 
  (C₂ : ℝ → ℝ → Prop) : MinCircle × CommonChord :=
sorry

theorem min_circle_and_common_chord_for_given_points :
  let A : ℝ × ℝ := (0, 2)
  let B : ℝ × ℝ := (2, -2)
  let C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 2*y + 5 = 0
  let (min_circle, common_chord) := find_min_circle_and_common_chord A B C₂
  min_circle.center = (1, 0) ∧
  min_circle.radius = Real.sqrt 5 ∧
  common_chord.length = Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_min_circle_and_common_chord_for_given_points_l1909_190927


namespace NUMINAMATH_CALUDE_unique_triple_solution_l1909_190923

theorem unique_triple_solution : 
  ∃! (x y z : ℕ+), 
    x ≤ y ∧ y ≤ z ∧ 
    x^3 * (y^3 + z^3) = 2012 * (x * y * z + 2) ∧
    x = 2 ∧ y = 251 ∧ z = 252 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l1909_190923


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_5_l1909_190977

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem units_digit_of_7_pow_5 : unitsDigit (7^5) = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_5_l1909_190977


namespace NUMINAMATH_CALUDE_triangle_conjugates_l1909_190990

/-- Triangle ABC with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

/-- Barycentric coordinates -/
structure BarycentricCoord where
  α : ℝ
  β : ℝ
  γ : ℝ
  h_pos : α > 0 ∧ β > 0 ∧ γ > 0

/-- Isotomically conjugate points -/
def isIsotomicallyConjugate (tri : Triangle) (p q : BarycentricCoord) : Prop :=
  p.α * q.α = p.β * q.β ∧ p.β * q.β = p.γ * q.γ ∧ p.γ * q.γ = p.α * q.α

/-- Isogonally conjugate points -/
def isIsogonallyConjugate (tri : Triangle) (p q : BarycentricCoord) : Prop :=
  p.α * q.α = tri.a^2 ∧ p.β * q.β = tri.b^2 ∧ p.γ * q.γ = tri.c^2

/-- Main theorem -/
theorem triangle_conjugates (tri : Triangle) (p : BarycentricCoord) :
  let q₁ : BarycentricCoord := ⟨p.α⁻¹, p.β⁻¹, p.γ⁻¹, sorry⟩
  let q₂ : BarycentricCoord := ⟨tri.a^2 / p.α, tri.b^2 / p.β, tri.c^2 / p.γ, sorry⟩
  isIsotomicallyConjugate tri p q₁ ∧ isIsogonallyConjugate tri p q₂ := by
  sorry

end NUMINAMATH_CALUDE_triangle_conjugates_l1909_190990


namespace NUMINAMATH_CALUDE_probability_at_least_six_heads_in_eight_flips_probability_at_least_six_heads_in_eight_flips_proof_l1909_190915

/-- The probability of getting at least 6 heads in 8 flips of a fair coin -/
theorem probability_at_least_six_heads_in_eight_flips : ℚ :=
  37 / 256

/-- Proof that the probability of getting at least 6 heads in 8 flips of a fair coin is 37/256 -/
theorem probability_at_least_six_heads_in_eight_flips_proof :
  probability_at_least_six_heads_in_eight_flips = 37 / 256 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_six_heads_in_eight_flips_probability_at_least_six_heads_in_eight_flips_proof_l1909_190915
