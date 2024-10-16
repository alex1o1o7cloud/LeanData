import Mathlib

namespace NUMINAMATH_CALUDE_board_cut_ratio_l295_29511

/-- Proves that the ratio of the shorter piece to the longer piece is 1:1 for a 20-foot board cut into two pieces -/
theorem board_cut_ratio (total_length : ℝ) (shorter_length : ℝ) (longer_length : ℝ) :
  total_length = 20 →
  shorter_length = 8 →
  shorter_length = longer_length + 4 →
  shorter_length / longer_length = 1 := by
  sorry

end NUMINAMATH_CALUDE_board_cut_ratio_l295_29511


namespace NUMINAMATH_CALUDE_sum_of_numbers_l295_29593

theorem sum_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_prod : x * y = 12) (h_recip : 1 / x = 3 * (1 / y)) : x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l295_29593


namespace NUMINAMATH_CALUDE_total_two_month_revenue_l295_29539

/-- Represents the amount of money raised in a single telethon day -/
def telethon_day_revenue (base_rate : ℝ) (first_period : ℝ) (second_period : ℝ) 
  (first_hours : ℝ) (second_hours : ℝ) : ℝ :=
  base_rate * first_hours + (base_rate * (1 + second_period)) * second_hours

/-- Represents the amount of money raised in a Sunday telethon -/
def sunday_telethon_revenue (base_rate : ℝ) : ℝ :=
  let initial_rate := base_rate * 0.85
  let first_10_hours := initial_rate * 10
  let fluctuating_16_hours := 
    (initial_rate * 1.05 * 2) + 
    (initial_rate * 1.3 * 4) + 
    (initial_rate * 0.9 * 2) + 
    (initial_rate * 1.2 * 1) + 
    (initial_rate * 0.75 * 7)
  first_10_hours + fluctuating_16_hours

/-- Represents the total amount of money raised in one weekend -/
def weekend_revenue : ℝ :=
  telethon_day_revenue 4000 0 0.1 12 14 + 
  telethon_day_revenue 5000 0 0.2 12 14 + 
  sunday_telethon_revenue 5000

/-- The main theorem stating the total amount raised over two months -/
theorem total_two_month_revenue : 
  weekend_revenue * 8 = 2849500 :=
sorry

end NUMINAMATH_CALUDE_total_two_month_revenue_l295_29539


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l295_29564

theorem fraction_sum_equality : (3 : ℚ) / 5 - 2 / 15 + 1 / 3 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l295_29564


namespace NUMINAMATH_CALUDE_f_has_zero_at_two_two_is_zero_point_of_f_l295_29500

/-- A function that has a zero point at 2 -/
def f (x : ℝ) : ℝ := x - 2

/-- Theorem stating that f has a zero point at 2 -/
theorem f_has_zero_at_two : f 2 = 0 := by
  sorry

/-- Definition of a zero point -/
def is_zero_point (g : ℝ → ℝ) (x : ℝ) : Prop := g x = 0

/-- Theorem stating that 2 is a zero point of f -/
theorem two_is_zero_point_of_f : is_zero_point f 2 := by
  sorry

end NUMINAMATH_CALUDE_f_has_zero_at_two_two_is_zero_point_of_f_l295_29500


namespace NUMINAMATH_CALUDE_gcd_power_minus_identity_gcd_power_minus_identity_general_l295_29566

theorem gcd_power_minus_identity (a : ℕ) (h : a ≥ 2) : 
  13530 ∣ a^41 - a :=
sorry

/- More general version for any natural number n -/
theorem gcd_power_minus_identity_general (n : ℕ) (a : ℕ) (h : a ≥ 2) : 
  ∃ k : ℕ, k ∣ a^n - a :=
sorry

end NUMINAMATH_CALUDE_gcd_power_minus_identity_gcd_power_minus_identity_general_l295_29566


namespace NUMINAMATH_CALUDE_butterfly_collection_l295_29561

theorem butterfly_collection (total : ℕ) (black : ℕ) : 
  total = 19 → 
  black = 10 → 
  ∃ (blue yellow : ℕ), 
    blue = 2 * yellow ∧ 
    blue + yellow + black = total ∧ 
    blue = 6 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_collection_l295_29561


namespace NUMINAMATH_CALUDE_committee_selection_ways_l295_29524

-- Define the total number of members
def total_members : ℕ := 30

-- Define the number of ineligible members
def ineligible_members : ℕ := 3

-- Define the size of the committee
def committee_size : ℕ := 5

-- Define the number of eligible members
def eligible_members : ℕ := total_members - ineligible_members

-- Theorem statement
theorem committee_selection_ways :
  Nat.choose eligible_members committee_size = 80730 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l295_29524


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_l295_29501

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_l295_29501


namespace NUMINAMATH_CALUDE_equal_sum_parallel_segments_l295_29544

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (AB : ℝ) (BC : ℝ) (CA : ℝ)

/-- Points on the sides of the triangle -/
structure Points (t : Triangle) :=
  (D : ℝ) (E : ℝ) (F : ℝ) (G : ℝ)
  (h₁ : 0 ≤ D ∧ D ≤ E ∧ E ≤ t.AB)
  (h₂ : 0 ≤ F ∧ F ≤ G ∧ G ≤ t.CA)

/-- Perimeter of triangle ADF -/
def perim_ADF (t : Triangle) (p : Points t) : ℝ :=
  p.D + (p.F - p.D) + p.F

/-- Perimeter of trapezoid DEFG -/
def perim_DEFG (t : Triangle) (p : Points t) : ℝ :=
  (p.E - p.D) + (p.G - p.F) + p.G + (p.F - p.D)

/-- Perimeter of trapezoid EBCG -/
def perim_EBCG (t : Triangle) (p : Points t) : ℝ :=
  (t.AB - p.E) + t.BC + (t.CA - p.G) + (p.G - p.F)

theorem equal_sum_parallel_segments (t : Triangle) (p : Points t) 
    (h_sides : t.AB = 2 ∧ t.BC = 3 ∧ t.CA = 4)
    (h_parallel : (p.E - p.D) / t.BC = (p.G - p.F) / t.BC)
    (h_perims : perim_ADF t p = perim_DEFG t p ∧ perim_DEFG t p = perim_EBCG t p) :
    (p.E - p.D) + (p.G - p.F) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_equal_sum_parallel_segments_l295_29544


namespace NUMINAMATH_CALUDE_projectile_height_l295_29581

theorem projectile_height (t : ℝ) : 
  (∃ t₀ : ℝ, t₀ > 0 ∧ -4.9 * t₀^2 + 30 * t₀ = 35 ∧ 
   ∀ t' : ℝ, t' > 0 ∧ -4.9 * t'^2 + 30 * t' = 35 → t₀ ≤ t') → 
  t = 10/7 := by
sorry

end NUMINAMATH_CALUDE_projectile_height_l295_29581


namespace NUMINAMATH_CALUDE_sum_of_roots_l295_29523

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

-- State the theorem
theorem sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = 19) : a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l295_29523


namespace NUMINAMATH_CALUDE_investment_sum_proof_l295_29558

/-- Proves that a sum invested at 15% p.a. simple interest for two years yields
    Rs. 420 more interest than if invested at 12% p.a. for the same period,
    then the sum is Rs. 7000. -/
theorem investment_sum_proof (P : ℚ) 
  (h1 : P * (15 / 100) * 2 - P * (12 / 100) * 2 = 420) : P = 7000 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_proof_l295_29558


namespace NUMINAMATH_CALUDE_fran_required_speed_l295_29557

/-- Given Joann's bike ride parameters and Fran's ride time, calculate Fran's required speed -/
theorem fran_required_speed 
  (joann_speed : ℝ) 
  (joann_time : ℝ) 
  (fran_time : ℝ) 
  (h1 : joann_speed = 15) 
  (h2 : joann_time = 4) 
  (h3 : fran_time = 2.5) : 
  (joann_speed * joann_time) / fran_time = 24 := by
sorry

end NUMINAMATH_CALUDE_fran_required_speed_l295_29557


namespace NUMINAMATH_CALUDE_function_domain_range_sum_l295_29547

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 2*x

-- Define the domain and range
def is_valid_domain_and_range (m n : ℝ) : Prop :=
  (∀ x, m ≤ x ∧ x ≤ n → 3*m ≤ f x ∧ f x ≤ 3*n) ∧
  (∃ x, m ≤ x ∧ x ≤ n ∧ f x = 3*m) ∧
  (∃ x, m ≤ x ∧ x ≤ n ∧ f x = 3*n)

-- State the theorem
theorem function_domain_range_sum :
  ∃ m n : ℝ, is_valid_domain_and_range m n ∧ m = -1 ∧ n = 0 ∧ m + n = -1 :=
sorry

end NUMINAMATH_CALUDE_function_domain_range_sum_l295_29547


namespace NUMINAMATH_CALUDE_exponential_function_property_l295_29589

theorem exponential_function_property (a : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = a^x) →
  (a > 0) →
  (abs (f 2 - f 1) = a / 2) →
  (a = 1/2 ∨ a = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_property_l295_29589


namespace NUMINAMATH_CALUDE_exists_cell_with_same_color_in_all_directions_l295_29579

/-- A color type with four possible values -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- A type representing a 50x50 grid colored with four colors -/
def ColoredGrid := Fin 50 → Fin 50 → Color

/-- A function to check if a cell has the same color in all four directions -/
def hasSameColorInAllDirections (grid : ColoredGrid) (row col : Fin 50) : Prop :=
  ∃ (r1 r2 : Fin 50) (c1 c2 : Fin 50),
    r1 < row ∧ row < r2 ∧ c1 < col ∧ col < c2 ∧
    grid row col = grid r1 col ∧
    grid row col = grid r2 col ∧
    grid row col = grid row c1 ∧
    grid row col = grid row c2

/-- Theorem stating that there exists a cell with the same color in all four directions -/
theorem exists_cell_with_same_color_in_all_directions (grid : ColoredGrid) :
  ∃ (row col : Fin 50), hasSameColorInAllDirections grid row col := by
  sorry


end NUMINAMATH_CALUDE_exists_cell_with_same_color_in_all_directions_l295_29579


namespace NUMINAMATH_CALUDE_four_prime_pairs_sum_50_l295_29533

/-- A function that returns true if a natural number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the number of unordered pairs of prime numbers that sum to a given number -/
def countPrimePairs (sum : ℕ) : ℕ := sorry

/-- Theorem stating that there are exactly 4 unordered pairs of prime numbers that sum to 50 -/
theorem four_prime_pairs_sum_50 : countPrimePairs 50 = 4 := by sorry

end NUMINAMATH_CALUDE_four_prime_pairs_sum_50_l295_29533


namespace NUMINAMATH_CALUDE_amy_bob_games_l295_29520

theorem amy_bob_games (n : ℕ) (h : n = 9) :
  let total_combinations := Nat.choose n 3
  let games_per_player := total_combinations / n
  let games_together := games_per_player / 4
  games_together = 7 := by
  sorry

end NUMINAMATH_CALUDE_amy_bob_games_l295_29520


namespace NUMINAMATH_CALUDE_larger_number_proof_l295_29508

theorem larger_number_proof (A B : ℕ+) (h1 : Nat.gcd A B = 28) 
  (h2 : Nat.lcm A B = 28 * 12 * 15) : max A B = 420 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l295_29508


namespace NUMINAMATH_CALUDE_hyperbola_and_slopes_l295_29590

-- Define the hyperbola E
def E (b : ℝ) (x y : ℝ) : Prop := x^2 - y^2/b^2 = 1

-- Define point P
def P : ℝ × ℝ := (-2, -3)

-- Define point Q
def Q : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem hyperbola_and_slopes 
  (b : ℝ) 
  (h1 : b > 0) 
  (h2 : E b P.1 P.2) 
  (A B : ℝ × ℝ) 
  (h3 : A ≠ P ∧ B ≠ P ∧ A ≠ B) 
  (h4 : ∃ k : ℝ, A.2 = k * A.1 - 1 ∧ B.2 = k * B.1 - 1) 
  (h5 : E b A.1 A.2 ∧ E b B.1 B.2) :
  (b^2 = 3) ∧ 
  (((A.2 - P.2) / (A.1 - P.1)) + ((B.2 - P.2) / (B.1 - P.1)) = 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_and_slopes_l295_29590


namespace NUMINAMATH_CALUDE_sales_tax_percentage_l295_29546

theorem sales_tax_percentage (total_before_tax : ℝ) (total_with_tax : ℝ) : 
  total_before_tax = 150 → 
  total_with_tax = 162 → 
  (total_with_tax - total_before_tax) / total_before_tax * 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_percentage_l295_29546


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_four_with_digit_sum_twelve_l295_29541

/-- The largest three-digit multiple of 4 whose digits' sum is 12 -/
def largest_multiple : ℕ := 912

/-- Function to calculate the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Function to check if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_four_with_digit_sum_twelve :
  (is_three_digit largest_multiple) ∧ 
  (largest_multiple % 4 = 0) ∧
  (digit_sum largest_multiple = 12) ∧
  (∀ n : ℕ, is_three_digit n → n % 4 = 0 → digit_sum n = 12 → n ≤ largest_multiple) := by
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_four_with_digit_sum_twelve_l295_29541


namespace NUMINAMATH_CALUDE_equality_iff_inequality_l295_29594

theorem equality_iff_inequality (x : ℝ) : (x - 2) * (x - 3) ≤ 0 ↔ |x - 2| + |x - 3| = 1 := by
  sorry

end NUMINAMATH_CALUDE_equality_iff_inequality_l295_29594


namespace NUMINAMATH_CALUDE_carly_dogs_worked_on_l295_29534

/-- The number of dogs Carly worked on given the number of nails trimmed,
    nails per paw, and number of three-legged dogs. -/
def dogs_worked_on (total_nails : ℕ) (nails_per_paw : ℕ) (three_legged_dogs : ℕ) : ℕ :=
  let total_paws := total_nails / nails_per_paw
  let three_legged_paws := three_legged_dogs * 3
  let four_legged_paws := total_paws - three_legged_paws
  let four_legged_dogs := four_legged_paws / 4
  four_legged_dogs + three_legged_dogs

theorem carly_dogs_worked_on :
  dogs_worked_on 164 4 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_carly_dogs_worked_on_l295_29534


namespace NUMINAMATH_CALUDE_max_k_value_l295_29577

/-- Sum of first n terms of sequence a_n -/
def S (n : ℕ) : ℚ := (n^2 + 11*n) / 2

/-- Sequence a_n -/
def a (n : ℕ) : ℚ := n + 5

/-- Sequence b_n -/
def b (n : ℕ) : ℚ := 3*n + 2

/-- Sequence c_n -/
def c (n : ℕ) : ℚ := 6 / ((2*a n - 11) * (2*b n - 1))

/-- Sum of first n terms of sequence c_n -/
def T (n : ℕ) : ℚ := 1 - 1 / (2*n + 1)

theorem max_k_value (n : ℕ+) :
  ∃ (k : ℕ), k = 37 ∧ 
  (∀ (m : ℕ+), T m > (m : ℚ) / 57) ∧
  (∀ (l : ℕ), l > k → ∃ (m : ℕ+), T m ≤ (l : ℚ) / 57) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l295_29577


namespace NUMINAMATH_CALUDE_inequality_proof_l295_29510

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l295_29510


namespace NUMINAMATH_CALUDE_probability_x_plus_y_less_than_4_l295_29504

/-- A square with vertices at (0, 0), (0, 3), (3, 3), and (3, 0) -/
def Square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

/-- The region where x + y < 4 within the square -/
def RegionXPlusYLessThan4 : Set (ℝ × ℝ) :=
  {p ∈ Square | p.1 + p.2 < 4}

/-- The area of the square -/
def squareArea : ℝ := 9

/-- The area of the region where x + y < 4 within the square -/
def regionArea : ℝ := 7

theorem probability_x_plus_y_less_than_4 :
  (regionArea / squareArea : ℝ) = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_less_than_4_l295_29504


namespace NUMINAMATH_CALUDE_binomial_n_minus_two_l295_29552

theorem binomial_n_minus_two (n : ℕ+) : Nat.choose n (n - 2) = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_n_minus_two_l295_29552


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l295_29550

/-- The equation of a line symmetric to y = 3x + 1 with respect to the y-axis -/
theorem symmetric_line_equation : 
  ∀ (x y : ℝ), (∃ (m n : ℝ), n = 3 * m + 1 ∧ x + m = 0 ∧ y = n) → y = -3 * x + 1 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l295_29550


namespace NUMINAMATH_CALUDE_sum_9_is_27_l295_29592

/-- An arithmetic sequence on a line through (5,3) -/
structure ArithmeticSequenceOnLine where
  a : ℕ+ → ℚ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1
  on_line : ∀ n : ℕ+, ∃ k m : ℚ, a n = k * n + m ∧ 3 = k * 5 + m

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequenceOnLine) (n : ℕ+) : ℚ :=
  (n : ℚ) * seq.a n

/-- The sum of the first 9 terms of an arithmetic sequence on a line through (5,3) is 27 -/
theorem sum_9_is_27 (seq : ArithmeticSequenceOnLine) : sum_n seq 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_sum_9_is_27_l295_29592


namespace NUMINAMATH_CALUDE_bus_profit_maximization_l295_29505

/-- The profit function for a bus operating for x years -/
def profit (x : ℕ+) : ℚ := -x^2 + 18*x - 36

/-- The average profit function for a bus operating for x years -/
def avgProfit (x : ℕ+) : ℚ := (profit x) / x

theorem bus_profit_maximization :
  (∃ (x : ℕ+), ∀ (y : ℕ+), profit x ≥ profit y) ∧
  (∃ (x : ℕ+), profit x = 45) ∧
  (∃ (x : ℕ+), ∀ (y : ℕ+), avgProfit x ≥ avgProfit y) ∧
  (∃ (x : ℕ+), avgProfit x = 6) :=
sorry

end NUMINAMATH_CALUDE_bus_profit_maximization_l295_29505


namespace NUMINAMATH_CALUDE_smallest_ten_digit_max_sum_l295_29599

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_ten_digit (n : Nat) : Prop :=
  1000000000 ≤ n ∧ n < 10000000000

theorem smallest_ten_digit_max_sum : 
  ∀ n : Nat, is_ten_digit n → n < 1999999999 → sum_of_digits n < sum_of_digits 1999999999 :=
sorry

#eval sum_of_digits 1999999999

end NUMINAMATH_CALUDE_smallest_ten_digit_max_sum_l295_29599


namespace NUMINAMATH_CALUDE_arithmetic_triangle_third_side_l295_29530

/-- Represents a triangle with sides in arithmetic progression -/
structure ArithmeticTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angles_sum : ℝ
  is_arithmetic_sides : b - a = c - b
  is_arithmetic_angles : ∃ (θ d : ℝ), angles_sum = 3 * θ ∧ θ = 60

/-- The theorem stating that for a specific arithmetic triangle, the third side is 5 -/
theorem arithmetic_triangle_third_side :
  ∀ (t : ArithmeticTriangle), t.a = 3 → t.b = 4 → t.angles_sum = 180 → t.c = 5 := by
  sorry

#check arithmetic_triangle_third_side

end NUMINAMATH_CALUDE_arithmetic_triangle_third_side_l295_29530


namespace NUMINAMATH_CALUDE_renatas_final_amount_is_77_l295_29538

/-- Calculates Renata's final amount after a series of transactions --/
def renatas_final_amount (initial_amount charity_donation charity_prize 
  slot_loss1 slot_loss2 slot_loss3 sunglasses_price sunglasses_discount
  water_price lottery_ticket_price lottery_prize sandwich_price 
  sandwich_discount latte_price : ℚ) : ℚ :=
  let after_charity := initial_amount - charity_donation + charity_prize
  let after_slots := after_charity - slot_loss1 - slot_loss2 - slot_loss3
  let sunglasses_cost := sunglasses_price * (1 - sunglasses_discount)
  let after_sunglasses := after_slots - sunglasses_cost
  let after_water_lottery := after_sunglasses - water_price - lottery_ticket_price + lottery_prize
  let meal_cost := (sandwich_price * (1 - sandwich_discount) + latte_price) / 2
  after_water_lottery - meal_cost

/-- Theorem stating that Renata's final amount is $77 --/
theorem renatas_final_amount_is_77 :
  renatas_final_amount 10 4 90 50 10 5 15 0.2 1 1 65 8 0.25 4 = 77 := by
  sorry

end NUMINAMATH_CALUDE_renatas_final_amount_is_77_l295_29538


namespace NUMINAMATH_CALUDE_total_vegetarian_eaters_l295_29597

/-- Represents the dietary preferences in a family -/
structure DietaryPreferences where
  vegetarian : ℕ
  nonVegetarian : ℕ
  bothVegNonVeg : ℕ
  vegan : ℕ
  veganAndVegetarian : ℕ
  pescatarian : ℕ
  pescatarianAndBoth : ℕ

/-- Theorem stating the total number of people eating vegetarian meals -/
theorem total_vegetarian_eaters (prefs : DietaryPreferences)
  (h1 : prefs.vegetarian = 13)
  (h2 : prefs.nonVegetarian = 7)
  (h3 : prefs.bothVegNonVeg = 8)
  (h4 : prefs.vegan = 5)
  (h5 : prefs.veganAndVegetarian = 3)
  (h6 : prefs.pescatarian = 4)
  (h7 : prefs.pescatarianAndBoth = 2) :
  prefs.vegetarian + prefs.bothVegNonVeg + (prefs.vegan - prefs.veganAndVegetarian) = 23 := by
  sorry

end NUMINAMATH_CALUDE_total_vegetarian_eaters_l295_29597


namespace NUMINAMATH_CALUDE_inequalities_always_satisfied_l295_29559

theorem inequalities_always_satisfied (a b c x y z : ℝ) 
  (hx : x < a) (hy : y < b) (hz : z < c) : 
  (x * y * c < a * b * z) ∧ 
  (x^2 + c < a^2 + z) ∧ 
  (x^2 * y^2 * z^2 < a^2 * b^2 * c^2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_always_satisfied_l295_29559


namespace NUMINAMATH_CALUDE_equation_solution_l295_29527

theorem equation_solution (r : ℝ) : 
  (r^2 - 6*r + 8)/(r^2 - 9*r + 20) = (r^2 - 3*r - 10)/(r^2 - 2*r - 15) ↔ r = 2*Real.sqrt 2 ∨ r = -2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l295_29527


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l295_29509

theorem smallest_integer_solution : ∃ x : ℤ, 
  (∀ y : ℤ, 10 * y^2 - 40 * y + 36 = 0 → x ≤ y) ∧ 
  (10 * x^2 - 40 * x + 36 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l295_29509


namespace NUMINAMATH_CALUDE_new_players_count_new_players_proof_l295_29512

theorem new_players_count (returning_players : ℕ) (players_per_group : ℕ) (num_groups : ℕ) : ℕ :=
  let total_players := num_groups * players_per_group
  total_players - returning_players

theorem new_players_proof :
  new_players_count 6 6 9 = 48 := by
  sorry

end NUMINAMATH_CALUDE_new_players_count_new_players_proof_l295_29512


namespace NUMINAMATH_CALUDE_mean_of_xyz_l295_29568

theorem mean_of_xyz (x y z : ℝ) 
  (eq1 : 9*x + 3*y - 5*z = -4)
  (eq2 : 5*x + 2*y - 2*z = 13) :
  (x + y + z) / 3 = 10 := by
sorry

end NUMINAMATH_CALUDE_mean_of_xyz_l295_29568


namespace NUMINAMATH_CALUDE_game_result_l295_29513

def g (m : ℕ) : ℕ :=
  if m % 3 = 0 then 8
  else if m = 2 ∨ m = 3 ∨ m = 5 then 3
  else if m % 2 = 0 then 1
  else 0

def jack_rolls : List ℕ := [2, 5, 6, 4, 3]
def jill_rolls : List ℕ := [1, 6, 3, 2, 5]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_result : total_points jack_rolls * total_points jill_rolls = 420 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l295_29513


namespace NUMINAMATH_CALUDE_ratio_problem_l295_29551

theorem ratio_problem (a b : ℝ) (h1 : a ≠ b) (h2 : (a + b) / (a - b) = 3) : a / b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l295_29551


namespace NUMINAMATH_CALUDE_count_master_sudokus_master_sudoku_count_l295_29529

/-- The number of Master Sudokus for a given n -/
def masterSudokuCount (n : ℕ) : ℕ :=
  2^(n-1)

/-- Theorem: The number of Master Sudokus for n is 2^(n-1) -/
theorem count_master_sudokus (n : ℕ) :
  (∀ k : ℕ, k < n → masterSudokuCount k = 2^(k-1)) →
  masterSudokuCount n = 2^(n-1) := by
  sorry

/-- The main theorem stating the number of Master Sudokus -/
theorem master_sudoku_count (n : ℕ) :
  masterSudokuCount n = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_count_master_sudokus_master_sudoku_count_l295_29529


namespace NUMINAMATH_CALUDE_similar_triangles_perimeter_area_l295_29587

/-- Given two similar triangles with corresponding median lengths and sum of perimeters,
    prove their individual perimeters and area ratio -/
theorem similar_triangles_perimeter_area (median1 median2 perimeter_sum : ℝ) :
  median1 = 10 →
  median2 = 4 →
  perimeter_sum = 140 →
  ∃ (perimeter1 perimeter2 area1 area2 : ℝ),
    perimeter1 = 100 ∧
    perimeter2 = 40 ∧
    perimeter1 + perimeter2 = perimeter_sum ∧
    (area1 / area2 = 25 / 4) ∧
    (median1 / median2)^2 = area1 / area2 ∧
    median1 / median2 = perimeter1 / perimeter2 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_perimeter_area_l295_29587


namespace NUMINAMATH_CALUDE_product_repeating_decimal_and_seven_l295_29554

theorem product_repeating_decimal_and_seven :
  let x : ℚ := 152 / 333
  x * 7 = 1064 / 333 := by sorry

end NUMINAMATH_CALUDE_product_repeating_decimal_and_seven_l295_29554


namespace NUMINAMATH_CALUDE_circle_area_below_line_l295_29535

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 - 14*y + 33 = 0

-- Define the line equation
def line_equation (y : ℝ) : Prop :=
  y = 7

-- Theorem statement
theorem circle_area_below_line :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    center_y = 7 ∧
    radius > 0 ∧
    (π * radius^2 / 2 : ℝ) = 25 * π / 2 :=
sorry

end NUMINAMATH_CALUDE_circle_area_below_line_l295_29535


namespace NUMINAMATH_CALUDE_solution_set_for_even_monotonic_function_l295_29560

-- Define the properties of the function f
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_monotonic_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- Define the set of solutions
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f (x + 1) = f (2 * x)}

-- Theorem statement
theorem solution_set_for_even_monotonic_function
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_monotonic : is_monotonic_on_positive f) :
  solution_set f = {1, -1/3} := by
sorry

end NUMINAMATH_CALUDE_solution_set_for_even_monotonic_function_l295_29560


namespace NUMINAMATH_CALUDE_stamps_problem_l295_29536

theorem stamps_problem (cj kj aj : ℕ) : 
  cj = 2 * kj + 5 →  -- CJ has 5 more than twice the number of stamps that KJ has
  kj * 2 = aj →      -- KJ has half as many stamps as AJ
  cj + kj + aj = 930 →  -- The three boys have 930 stamps in total
  aj = 370 :=         -- Prove that AJ has 370 stamps
by
  sorry


end NUMINAMATH_CALUDE_stamps_problem_l295_29536


namespace NUMINAMATH_CALUDE_n3_equals_9_l295_29563

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k, n = 9 * k

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem n3_equals_9 
  (N : ℕ) 
  (h1 : 10^1989 ≤ 16*N ∧ 16*N < 10^1990) 
  (h2 : is_multiple_of_9 (16*N)) 
  (N1 : ℕ) (h3 : N1 = sum_of_digits N)
  (N2 : ℕ) (h4 : N2 = sum_of_digits N1)
  (N3 : ℕ) (h5 : N3 = sum_of_digits N2) :
  N3 = 9 :=
sorry

end NUMINAMATH_CALUDE_n3_equals_9_l295_29563


namespace NUMINAMATH_CALUDE_med_school_acceptances_l295_29574

theorem med_school_acceptances 
  (total_researched : ℕ) 
  (applied_fraction : ℚ) 
  (accepted_fraction : ℚ) 
  (h1 : total_researched = 42)
  (h2 : applied_fraction = 1 / 3)
  (h3 : accepted_fraction = 1 / 2) :
  ↑⌊(total_researched : ℚ) * applied_fraction * accepted_fraction⌋ = 7 :=
by sorry

end NUMINAMATH_CALUDE_med_school_acceptances_l295_29574


namespace NUMINAMATH_CALUDE_h_k_equality_implies_m_value_l295_29582

/-- The function h(x) = x^2 - 3x + m -/
def h (x m : ℝ) : ℝ := x^2 - 3*x + m

/-- The function k(x) = x^2 - 3x + 5m -/
def k (x m : ℝ) : ℝ := x^2 - 3*x + 5*m

/-- Theorem stating that if 3h(5) = 2k(5), then m = 10/7 -/
theorem h_k_equality_implies_m_value :
  ∀ m : ℝ, 3 * (h 5 m) = 2 * (k 5 m) → m = 10/7 := by
  sorry

end NUMINAMATH_CALUDE_h_k_equality_implies_m_value_l295_29582


namespace NUMINAMATH_CALUDE_count_distinct_terms_l295_29567

/-- The number of distinct terms in the expansion of (x+y+z)^2026 + (x-y-z)^2026 -/
def num_distinct_terms : ℕ := 1028196

/-- The exponent in the original expression -/
def exponent : ℕ := 2026

-- Theorem stating the number of distinct terms
theorem count_distinct_terms : 
  num_distinct_terms = (exponent / 2 + 1)^2 := by sorry

end NUMINAMATH_CALUDE_count_distinct_terms_l295_29567


namespace NUMINAMATH_CALUDE_age_ratio_l295_29542

/-- Given the ages of Albert, Mary, and Betty, prove the ratio of Albert's age to Betty's age -/
theorem age_ratio (albert mary betty : ℕ) (h1 : albert = 2 * mary) 
  (h2 : mary = albert - 10) (h3 : betty = 5) : albert / betty = 4 := by
  sorry


end NUMINAMATH_CALUDE_age_ratio_l295_29542


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l295_29528

/-- Represents different income levels of families -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic

/-- Represents a survey with its population and sample size -/
structure Survey where
  population : ℕ
  sampleSize : ℕ
  incomeGroups : Option (ℕ × ℕ × ℕ)

/-- Determines the appropriate sampling method for a given survey -/
def appropriateSamplingMethod (s : Survey) : SamplingMethod :=
  sorry

/-- The first survey from the problem -/
def survey1 : Survey :=
  { population := 430 + 980 + 290
  , sampleSize := 170
  , incomeGroups := some (430, 980, 290) }

/-- The second survey from the problem -/
def survey2 : Survey :=
  { population := 12
  , sampleSize := 5
  , incomeGroups := none }

/-- Theorem stating the appropriate sampling methods for the two surveys -/
theorem appropriate_sampling_methods :
  appropriateSamplingMethod survey1 = SamplingMethod.Stratified ∧
  appropriateSamplingMethod survey2 = SamplingMethod.SimpleRandom :=
sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l295_29528


namespace NUMINAMATH_CALUDE_sum_of_numbers_l295_29515

theorem sum_of_numbers (x y : ℝ) (h1 : x ≠ y) (h2 : x^2 - 2000*x = y^2 - 2000*y) : x + y = 2000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l295_29515


namespace NUMINAMATH_CALUDE_intersection_A_B_l295_29502

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 1}
def B : Set ℝ := {-2, -1, 0, 1}

theorem intersection_A_B : A ∩ B = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l295_29502


namespace NUMINAMATH_CALUDE_investment_growth_l295_29562

/-- Calculates the total amount after compound interest is applied for a given number of periods -/
def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- The problem statement -/
theorem investment_growth : compound_interest 300 0.1 2 = 363 := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l295_29562


namespace NUMINAMATH_CALUDE_odd_factors_of_450_l295_29583

/-- The number of odd factors of a natural number n -/
def num_odd_factors (n : ℕ) : ℕ := sorry

/-- 450 has exactly 9 odd factors -/
theorem odd_factors_of_450 : num_odd_factors 450 = 9 := by sorry

end NUMINAMATH_CALUDE_odd_factors_of_450_l295_29583


namespace NUMINAMATH_CALUDE_total_wrapping_cost_l295_29570

/-- Represents a wrapping paper design with its cost and wrapping capacities -/
structure WrappingPaper where
  cost : ℝ
  shirtBoxCapacity : ℕ
  xlBoxCapacity : ℕ
  xxlBoxCapacity : ℕ

/-- Calculates the number of rolls needed for a given number of boxes -/
def rollsNeeded (boxes : ℕ) (capacity : ℕ) : ℕ :=
  (boxes + capacity - 1) / capacity

/-- Calculates the cost for wrapping a specific type of box -/
def costForBoxType (paper : WrappingPaper) (boxes : ℕ) (capacity : ℕ) : ℝ :=
  paper.cost * (rollsNeeded boxes capacity : ℝ)

/-- Theorem stating the total cost of wrapping all boxes -/
theorem total_wrapping_cost (design1 design2 design3 : WrappingPaper)
    (shirtBoxes xlBoxes xxlBoxes : ℕ) :
    design1.cost = 4 →
    design1.shirtBoxCapacity = 5 →
    design2.cost = 8 →
    design2.xlBoxCapacity = 4 →
    design3.cost = 12 →
    design3.xxlBoxCapacity = 4 →
    shirtBoxes = 20 →
    xlBoxes = 12 →
    xxlBoxes = 6 →
    costForBoxType design1 shirtBoxes design1.shirtBoxCapacity +
    costForBoxType design2 xlBoxes design2.xlBoxCapacity +
    costForBoxType design3 xxlBoxes design3.xxlBoxCapacity = 76 := by
  sorry

end NUMINAMATH_CALUDE_total_wrapping_cost_l295_29570


namespace NUMINAMATH_CALUDE_binary_property_l295_29537

-- Define the number in base 10
def base10Number : Nat := 235

-- Define a function to convert a number to binary
def toBinary (n : Nat) : List Nat := sorry

-- Define a function to count zeros in a binary representation
def countZeros (binary : List Nat) : Nat := sorry

-- Define a function to count ones in a binary representation
def countOnes (binary : List Nat) : Nat := sorry

-- Theorem statement
theorem binary_property :
  let binary := toBinary base10Number
  let x := countZeros binary
  let y := countOnes binary
  y^2 - 2*x = 32 := by sorry

end NUMINAMATH_CALUDE_binary_property_l295_29537


namespace NUMINAMATH_CALUDE_derivative_F_at_one_l295_29531

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define F in terms of f
def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x^3 - 1) + f (1 - x^3)

-- State the theorem
theorem derivative_F_at_one (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  deriv (F f) 1 = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_F_at_one_l295_29531


namespace NUMINAMATH_CALUDE_strictly_increasing_implies_monotone_increasing_l295_29578

-- Define a function f from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property that for any x₁ < x₂, f(x₁) < f(x₂)
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

-- Theorem statement
theorem strictly_increasing_implies_monotone_increasing
  (h : StrictlyIncreasing f) : MonotoneOn f Set.univ :=
sorry

end NUMINAMATH_CALUDE_strictly_increasing_implies_monotone_increasing_l295_29578


namespace NUMINAMATH_CALUDE_white_surface_area_fraction_l295_29576

theorem white_surface_area_fraction (cube_edge : ℕ) (small_cube_edge : ℕ) 
  (total_cubes : ℕ) (white_cubes : ℕ) :
  cube_edge = 4 →
  small_cube_edge = 1 →
  total_cubes = 64 →
  white_cubes = 16 →
  (white_cubes : ℚ) / ((cube_edge ^ 2 * 6) : ℚ) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_area_fraction_l295_29576


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_l295_29548

theorem sqrt_sum_difference (x : ℝ) : 
  Real.sqrt 8 + Real.sqrt 18 - 4 * Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_l295_29548


namespace NUMINAMATH_CALUDE_multiply_powers_of_x_l295_29549

theorem multiply_powers_of_x (x : ℝ) : 2 * x * (3 * x^2) = 6 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_of_x_l295_29549


namespace NUMINAMATH_CALUDE_divisibility_by_19_l295_29521

theorem divisibility_by_19 (n : ℕ) : ∃ k : ℤ, 
  120 * 10^(n+2) + 3 * ((10^(n+1) - 1) / 9) * 100 + 8 = 19 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_19_l295_29521


namespace NUMINAMATH_CALUDE_factorization_equality_l295_29514

theorem factorization_equality (x : ℝ) :
  3 * x^2 * (x - 4) + 5 * x * (x - 4) = (3 * x^2 + 5 * x) * (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l295_29514


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l295_29591

theorem chess_tournament_participants (total_games : ℕ) 
  (h1 : total_games = 105) : ∃ n : ℕ, n > 0 ∧ n * (n - 1) / 2 = total_games := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l295_29591


namespace NUMINAMATH_CALUDE_tangent_line_determines_function_l295_29571

/-- Given a function f(x) = (mx-6)/(x^2+n), prove that if the tangent line
    at P(-1,f(-1)) is x+2y+5=0, then f(x) = (2x-6)/(x^2+3) -/
theorem tangent_line_determines_function (m n : ℝ) :
  let f : ℝ → ℝ := λ x => (m*x - 6) / (x^2 + n)
  let tangent_line : ℝ → ℝ := λ x => -(1/2)*x - 5/2
  (f (-1) = tangent_line (-1) ∧ 
   (deriv f) (-1) = (deriv tangent_line) (-1)) →
  f = λ x => (2*x - 6) / (x^2 + 3) :=
by
  sorry


end NUMINAMATH_CALUDE_tangent_line_determines_function_l295_29571


namespace NUMINAMATH_CALUDE_pyramid_lateral_angle_l295_29555

/-- Given a pyramid with an isosceles triangular base of area S, angle α between the equal sides,
    and volume V, the angle θ between the lateral edges and the base plane is:
    θ = arctan((3V * cos(α/2) / S) * sqrt(2 * sin(α) / S)) -/
theorem pyramid_lateral_angle (S V : ℝ) (α : ℝ) (hS : S > 0) (hV : V > 0) (hα : 0 < α ∧ α < π) :
  ∃ θ : ℝ, θ = Real.arctan ((3 * V * Real.cos (α / 2) / S) * Real.sqrt (2 * Real.sin α / S)) :=
sorry

end NUMINAMATH_CALUDE_pyramid_lateral_angle_l295_29555


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l295_29519

theorem rectangle_dimension_change (L B : ℝ) (h₁ : L > 0) (h₂ : B > 0) : 
  let new_B := 1.25 * B
  let new_area := 1.375 * (L * B)
  ∃ x : ℝ, x = 10 ∧ new_area = (L * (1 + x / 100)) * new_B := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l295_29519


namespace NUMINAMATH_CALUDE_one_common_color_l295_29525

/-- Given a set of n ≥ 5 colors and n+1 distinct 3-element subsets,
    there exist two subsets that share exactly one element. -/
theorem one_common_color (n : ℕ) (C : Finset ℕ) (A : Fin (n + 1) → Finset ℕ)
  (h_n : n ≥ 5)
  (h_C : C.card = n)
  (h_A_subset : ∀ i, A i ⊆ C)
  (h_A_card : ∀ i, (A i).card = 3)
  (h_A_distinct : ∀ i j, i ≠ j → A i ≠ A j) :
  ∃ i j, i ≠ j ∧ (A i ∩ A j).card = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_common_color_l295_29525


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l295_29503

/-- The eccentricity of a hyperbola with the given conditions is √5 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let right_vertex := (a, 0)
  let line := fun (x : ℝ) => -x + a
  let asymptote1 := fun (x : ℝ) => (b / a) * x
  let asymptote2 := fun (x : ℝ) => -(b / a) * x
  let B := (a^2 / (a + b), a * b / (a + b))
  let C := (a^2 / (a - b), -a * b / (a - b))
  let vector_AB := (B.1 - right_vertex.1, B.2 - right_vertex.2)
  let vector_BC := (C.1 - B.1, C.2 - B.2)
  vector_AB = (1/2 : ℝ) • vector_BC →
  ∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c / a = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l295_29503


namespace NUMINAMATH_CALUDE_equation_solution_l295_29596

theorem equation_solution (x : ℝ) : 
  x ≠ -3 → (-x^2 = (3*x + 1) / (x + 3) ↔ x = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l295_29596


namespace NUMINAMATH_CALUDE_single_draw_probability_triple_draw_probability_l295_29545

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Black

/-- Represents the outcome of drawing a single ball -/
def SingleDrawOutcome := BallColor

/-- Represents the outcome of drawing three balls -/
def TripleDrawOutcome := (BallColor × BallColor × BallColor)

/-- The total number of balls in the box -/
def totalBalls : Nat := 5 + 2

/-- The number of white balls in the box -/
def whiteBalls : Nat := 5

/-- The number of black balls in the box -/
def blackBalls : Nat := 2

/-- A function that simulates drawing a single ball -/
noncomputable def simulateSingleDraw : SingleDrawOutcome := sorry

/-- A function that simulates drawing three balls -/
noncomputable def simulateTripleDraw : TripleDrawOutcome := sorry

/-- Checks if a single draw outcome is favorable (white ball) -/
def isFavorableSingleDraw (outcome : SingleDrawOutcome) : Bool :=
  match outcome with
  | BallColor.White => true
  | BallColor.Black => false

/-- Checks if a triple draw outcome is favorable (all white balls) -/
def isFavorableTripleDraw (outcome : TripleDrawOutcome) : Bool :=
  match outcome with
  | (BallColor.White, BallColor.White, BallColor.White) => true
  | _ => false

/-- Theorem: The probability of drawing a white ball is equal to the ratio of favorable outcomes to total outcomes in a random simulation -/
theorem single_draw_probability (n : Nat) (m : Nat) :
  m ≤ n →
  (m : ℚ) / n = whiteBalls / totalBalls :=
sorry

/-- Theorem: The probability of drawing three white balls is equal to the ratio of favorable outcomes to total outcomes in a random simulation -/
theorem triple_draw_probability (n : Nat) (m : Nat) :
  m ≤ n →
  (m : ℚ) / n = (whiteBalls / totalBalls) * ((whiteBalls - 1) / (totalBalls - 1)) * ((whiteBalls - 2) / (totalBalls - 2)) :=
sorry

end NUMINAMATH_CALUDE_single_draw_probability_triple_draw_probability_l295_29545


namespace NUMINAMATH_CALUDE_cookies_theorem_l295_29580

def cookies_problem (total_cookies : ℕ) : Prop :=
  let father_cookies := (total_cookies : ℚ) * (1 / 10)
  let mother_cookies := father_cookies / 2
  let brother_cookies := mother_cookies + 2
  let sister_cookies := brother_cookies * (3 / 2)
  let aunt_cookies := father_cookies * 2
  let cousin_cookies := aunt_cookies * (4 / 5)
  let grandmother_cookies := cousin_cookies / 3
  let eaten_cookies := father_cookies + mother_cookies + brother_cookies + 
                       sister_cookies + aunt_cookies + cousin_cookies + 
                       grandmother_cookies
  let monica_cookies := total_cookies - eaten_cookies.floor
  monica_cookies = 120

theorem cookies_theorem : cookies_problem 400 := by
  sorry

end NUMINAMATH_CALUDE_cookies_theorem_l295_29580


namespace NUMINAMATH_CALUDE_vertical_asymptote_at_five_l295_29553

/-- The function f(x) = (x^2 + 3x + 4) / (x - 5) has a vertical asymptote at x = 5 -/
theorem vertical_asymptote_at_five (x : ℝ) :
  let f := fun (x : ℝ) => (x^2 + 3*x + 4) / (x - 5)
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), 0 < δ ∧ δ < ε →
    (abs (f (5 + δ)) > 1/δ) ∧ (abs (f (5 - δ)) > 1/δ) :=
by sorry

end NUMINAMATH_CALUDE_vertical_asymptote_at_five_l295_29553


namespace NUMINAMATH_CALUDE_maximize_x2y5_l295_29569

theorem maximize_x2y5 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 50) :
  x^2 * y^5 ≤ (100/7)^2 * (250/7)^5 ∧ 
  (x^2 * y^5 = (100/7)^2 * (250/7)^5 ↔ x = 100/7 ∧ y = 250/7) := by
sorry

end NUMINAMATH_CALUDE_maximize_x2y5_l295_29569


namespace NUMINAMATH_CALUDE_topsoil_cost_for_seven_cubic_yards_l295_29585

/-- The cost of topsoil in dollars per cubic foot -/
def topsoil_cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil -/
def cubic_yards_of_topsoil : ℝ := 7

/-- The cost of topsoil for a given number of cubic yards -/
def topsoil_cost (cubic_yards : ℝ) : ℝ :=
  cubic_yards * cubic_feet_per_cubic_yard * topsoil_cost_per_cubic_foot

theorem topsoil_cost_for_seven_cubic_yards :
  topsoil_cost cubic_yards_of_topsoil = 1512 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_for_seven_cubic_yards_l295_29585


namespace NUMINAMATH_CALUDE_contest_paths_count_l295_29572

/-- Represents a grid where the word "CONTEST" can be spelled out -/
structure ContestGrid where
  word : String
  start_letter : Char
  end_letter : Char

/-- Calculates the number of valid paths to spell out the word on the grid -/
def count_paths (grid : ContestGrid) : ℕ :=
  2^(grid.word.length - 1) - 1

/-- Theorem stating that the number of valid paths to spell "CONTEST" is 127 -/
theorem contest_paths_count :
  ∀ (grid : ContestGrid),
    grid.word = "CONTEST" →
    grid.start_letter = 'C' →
    grid.end_letter = 'T' →
    count_paths grid = 127 := by
  sorry


end NUMINAMATH_CALUDE_contest_paths_count_l295_29572


namespace NUMINAMATH_CALUDE_linear_composition_solution_l295_29573

/-- A linear function f that satisfies f[f(x)] = 4x - 1 -/
def LinearComposition (f : ℝ → ℝ) : Prop :=
  (∃ k b : ℝ, ∀ x, f x = k * x + b) ∧ 
  (∀ x, f (f x) = 4 * x - 1)

/-- The theorem stating that a linear function satisfying the composition condition
    must be one of two specific linear functions -/
theorem linear_composition_solution (f : ℝ → ℝ) (h : LinearComposition f) :
  (∀ x, f x = 2 * x - 1/3) ∨ (∀ x, f x = -2 * x + 1) :=
sorry

end NUMINAMATH_CALUDE_linear_composition_solution_l295_29573


namespace NUMINAMATH_CALUDE_circle_tangent_and_center_range_l295_29526

-- Define the given points and lines
def A : ℝ × ℝ := (0, 3)
def l (x : ℝ) : ℝ := 2 * x - 4

-- Define the circle C
def C (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 1}

-- Define the condition for the center of C
def center_condition (center : ℝ × ℝ) : Prop :=
  center.2 = l center.1 ∧ center.2 = center.1 - 1

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop :=
  3 * x + 4 * y - 12 = 0

-- Define the condition for point M
def M_condition (center : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
  M ∈ C center ∧ (M.1 - A.1)^2 + (M.2 - A.2)^2 = 4 * ((M.1 - center.1)^2 + (M.2 - center.2)^2)

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 12/5

-- State the theorem
theorem circle_tangent_and_center_range :
  ∃ (center : ℝ × ℝ),
    center_condition center ∧
    (∀ (x y : ℝ), (x, y) ∈ C center → tangent_line x y) ∧
    (∃ (M : ℝ × ℝ), M_condition center M) ∧
    a_range center.1 := by sorry

end NUMINAMATH_CALUDE_circle_tangent_and_center_range_l295_29526


namespace NUMINAMATH_CALUDE_reflection_of_point_l295_29595

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Theorem: The reflection of the point (5,2) across the x-axis is (5,-2) -/
theorem reflection_of_point : reflect_x (5, 2) = (5, -2) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_point_l295_29595


namespace NUMINAMATH_CALUDE_not_prime_n_pow_n_minus_4n_plus_3_l295_29507

theorem not_prime_n_pow_n_minus_4n_plus_3 (n : ℕ) : ¬ Nat.Prime (n^n - 4*n + 3) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_n_pow_n_minus_4n_plus_3_l295_29507


namespace NUMINAMATH_CALUDE_distance_between_x_intercepts_l295_29575

-- Define the slopes and intersection point
def m1 : ℝ := 4
def m2 : ℝ := -2
def intersection : ℝ × ℝ := (8, 20)

-- Define the lines using point-slope form
def line1 (x : ℝ) : ℝ := m1 * (x - intersection.1) + intersection.2
def line2 (x : ℝ) : ℝ := m2 * (x - intersection.1) + intersection.2

-- Define x-intercepts
noncomputable def x_intercept1 : ℝ := (intersection.2 - m1 * intersection.1) / (-m1)
noncomputable def x_intercept2 : ℝ := (intersection.2 - m2 * intersection.1) / (-m2)

-- Theorem statement
theorem distance_between_x_intercepts :
  |x_intercept2 - x_intercept1| = 15 := by sorry

end NUMINAMATH_CALUDE_distance_between_x_intercepts_l295_29575


namespace NUMINAMATH_CALUDE_solution_set_and_minimum_t_l295_29506

/-- The set of all numerical values of the real number a -/
def M : Set ℝ := {a | ∀ x : ℝ, a * x^2 + a * x + 2 > 0}

theorem solution_set_and_minimum_t :
  (M = {a : ℝ | 0 ≤ a ∧ a < 4}) ∧
  (∃ t₀ : ℝ, t₀ > 0 ∧ ∀ t : ℝ, t > 0 → (∀ a ∈ M, (a^2 - 2*a) * t ≤ t^2 + 3*t - 46) → t ≥ t₀) ∧
  (∀ t : ℝ, t > 0 → (∀ a ∈ M, (a^2 - 2*a) * t ≤ t^2 + 3*t - 46) → t ≥ 46) :=
by sorry

#check solution_set_and_minimum_t

end NUMINAMATH_CALUDE_solution_set_and_minimum_t_l295_29506


namespace NUMINAMATH_CALUDE_males_band_not_orchestra_l295_29586

/-- Represents the number of students in various categories of the school's music program -/
structure MusicProgram where
  female_band : ℕ
  male_band : ℕ
  female_orchestra : ℕ
  male_orchestra : ℕ
  female_both : ℕ
  left_band : ℕ
  total_either : ℕ

/-- Theorem stating the number of males in the band who are not in the orchestra -/
theorem males_band_not_orchestra (mp : MusicProgram) : 
  mp.female_band = 120 →
  mp.male_band = 90 →
  mp.female_orchestra = 70 →
  mp.male_orchestra = 110 →
  mp.female_both = 55 →
  mp.left_band = 10 →
  mp.total_either = 250 →
  mp.male_band - (mp.male_band + mp.male_orchestra - (mp.total_either - ((mp.female_band + mp.female_orchestra - mp.female_both) + mp.left_band))) = 15 := by
  sorry


end NUMINAMATH_CALUDE_males_band_not_orchestra_l295_29586


namespace NUMINAMATH_CALUDE_count_marquis_duels_l295_29522

theorem count_marquis_duels (counts dukes marquises : ℕ) 
  (h1 : counts > 0) (h2 : dukes > 0) (h3 : marquises > 0)
  (h4 : 3 * counts = 2 * dukes)
  (h5 : 6 * dukes = 3 * marquises)
  (h6 : 2 * marquises = 2 * counts * k)
  (h7 : k > 0) :
  k = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_marquis_duels_l295_29522


namespace NUMINAMATH_CALUDE_consecutive_product_divisible_by_six_product_over_six_is_integer_l295_29532

theorem consecutive_product_divisible_by_six (n : ℤ) : ∃ k : ℤ, n * (n + 1) * (n + 2) = 6 * k := by
  sorry

theorem product_over_six_is_integer (n : ℤ) : ∃ m : ℤ, n * (n + 1) * (n + 2) / 6 = m := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_divisible_by_six_product_over_six_is_integer_l295_29532


namespace NUMINAMATH_CALUDE_vector_orthogonality_l295_29516

def a (k : ℝ) : ℝ × ℝ := (k, 3)
def b : ℝ × ℝ := (1, 4)
def c : ℝ × ℝ := (2, 1)

theorem vector_orthogonality (k : ℝ) :
  (2 • a k - 3 • b) • c = 0 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_orthogonality_l295_29516


namespace NUMINAMATH_CALUDE_house_selling_price_l295_29584

/-- Proves that the selling price of each house is $120,000 given the problem conditions -/
theorem house_selling_price (C S : ℝ) : 
  (C + 100000 = 1.5 * S - 60000) → -- Construction cost of certain house equals its selling price minus profit
  (C = S - 100000) →               -- Construction cost difference between certain house and others
  S = 120000 := by
sorry

end NUMINAMATH_CALUDE_house_selling_price_l295_29584


namespace NUMINAMATH_CALUDE_lawn_width_proof_l295_29588

/-- The width of a rectangular lawn with specific conditions -/
def lawn_width : ℝ := 50

theorem lawn_width_proof (length : ℝ) (road_width : ℝ) (total_road_area : ℝ) :
  length = 80 →
  road_width = 10 →
  total_road_area = 1200 →
  lawn_width = (total_road_area - (length * road_width) + (road_width * road_width)) / road_width :=
by
  sorry

#check lawn_width_proof

end NUMINAMATH_CALUDE_lawn_width_proof_l295_29588


namespace NUMINAMATH_CALUDE_at_least_one_fails_l295_29517

-- Define the propositions
variable (p : Prop) -- "Student A passes the driving test"
variable (q : Prop) -- "Student B passes the driving test"

-- Define the theorem
theorem at_least_one_fails : (¬p ∨ ¬q) ↔ (∃ student, student = p ∨ student = q) ∧ (¬student) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_fails_l295_29517


namespace NUMINAMATH_CALUDE_sum_of_coordinates_of_D_l295_29518

/-- Given that N is the midpoint of CD and C's coordinates, prove the sum of D's coordinates -/
theorem sum_of_coordinates_of_D (N C : ℝ × ℝ) (h1 : N = (2, 6)) (h2 : C = (6, 2)) :
  ∃ D : ℝ × ℝ, N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) ∧ D.1 + D.2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_of_D_l295_29518


namespace NUMINAMATH_CALUDE_davids_average_marks_l295_29598

def english_marks : ℝ := 90
def mathematics_marks : ℝ := 92
def physics_marks : ℝ := 85
def chemistry_marks : ℝ := 87
def biology_marks : ℝ := 85

def total_marks : ℝ := english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks
def number_of_subjects : ℝ := 5

theorem davids_average_marks :
  total_marks / number_of_subjects = 87.8 := by
  sorry

end NUMINAMATH_CALUDE_davids_average_marks_l295_29598


namespace NUMINAMATH_CALUDE_highest_water_level_on_tuesday_l295_29543

/-- Water level changes for each day of the week -/
def water_level_changes : List ℝ := [0.03, 0.41, 0.25, 0.10, 0, -0.13, -0.2]

/-- Days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- The day with the highest water level change -/
def highest_water_level_day : Day := Day.Tuesday

theorem highest_water_level_on_tuesday :
  water_level_changes[1] = (List.maximum water_level_changes).get! :=
by sorry

end NUMINAMATH_CALUDE_highest_water_level_on_tuesday_l295_29543


namespace NUMINAMATH_CALUDE_sum_reciprocal_inequality_l295_29565

theorem sum_reciprocal_inequality (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c ≤ 3) : 
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) ≥ 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocal_inequality_l295_29565


namespace NUMINAMATH_CALUDE_rectangle_area_relation_l295_29556

/-- Given a rectangle with area 10 and adjacent sides x and y, 
    prove that the relationship between x and y is y = 10/x -/
theorem rectangle_area_relation (x y : ℝ) (h : x * y = 10) : y = 10 / x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_relation_l295_29556


namespace NUMINAMATH_CALUDE_smallest_positive_period_of_f_l295_29540

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x * Real.cos (2 * x)

theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧
  (∀ q, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_period_of_f_l295_29540
