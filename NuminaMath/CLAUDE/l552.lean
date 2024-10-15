import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_equation_l552_55281

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := ∃ (a b : ℝ), x^2/a^2 - y^2/b^2 = 1

-- Define the condition of shared foci
def shared_foci (e h : (ℝ → ℝ → Prop)) : Prop := 
  ∃ (c : ℝ), c^2 = 5 ∧ 
    (∀ x y, e x y ↔ x^2/(c^2+4) + y^2/4 = 1) ∧
    (∀ x y, h x y ↔ ∃ (a b : ℝ), x^2/a^2 - y^2/b^2 = 1 ∧ a^2 - b^2 = c^2)

-- Define the asymptote condition
def asymptote_condition (h : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), h x y ∧ x - 2*y = 0

-- The theorem to prove
theorem hyperbola_equation 
  (h : shared_foci ellipse hyperbola_C)
  (a : asymptote_condition hyperbola_C) :
  ∀ x y, hyperbola_C x y ↔ x^2/4 - y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l552_55281


namespace NUMINAMATH_CALUDE_oliver_stickers_l552_55283

theorem oliver_stickers (initial_stickers : ℕ) (used_fraction : ℚ) (kept_stickers : ℕ) 
  (h1 : initial_stickers = 135)
  (h2 : used_fraction = 1/3)
  (h3 : kept_stickers = 54) :
  let remaining_stickers := initial_stickers - (used_fraction * initial_stickers).num
  let given_stickers := remaining_stickers - kept_stickers
  (given_stickers : ℚ) / remaining_stickers = 2/5 := by
sorry

end NUMINAMATH_CALUDE_oliver_stickers_l552_55283


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_squares_l552_55207

theorem product_of_sum_and_sum_of_squares (a b : ℝ) 
  (sum_of_squares : a^2 + b^2 = 26) 
  (sum : a + b = 7) : 
  a * b = 23 / 2 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_squares_l552_55207


namespace NUMINAMATH_CALUDE_value_of_a_l552_55277

theorem value_of_a (a : ℝ) : -3 ∈ ({a - 3, 2 * a - 1, a^2 + 1} : Set ℝ) → a = 0 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l552_55277


namespace NUMINAMATH_CALUDE_boundary_is_pentagon_l552_55237

/-- The set S of points (x, y) satisfying the given conditions -/
def S (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               a / 2 ≤ x ∧ x ≤ 2 * a ∧
               a / 2 ≤ y ∧ y ≤ 2 * a ∧
               x + y ≥ 3 * a ∧
               x + a ≥ y ∧
               y + a ≥ x}

/-- The boundary of set S -/
def boundary (a : ℝ) : Set (ℝ × ℝ) :=
  frontier (S a)

/-- The number of sides of the polygon formed by the boundary of S -/
def numSides (a : ℝ) : ℕ :=
  sorry

theorem boundary_is_pentagon (a : ℝ) (h : a > 0) : numSides a = 5 :=
  sorry

end NUMINAMATH_CALUDE_boundary_is_pentagon_l552_55237


namespace NUMINAMATH_CALUDE_magic_square_base_5_l552_55204

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a b c d e f g h i : ℕ)

/-- Converts a number from base 5 to base 10 -/
def toBase10 (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 4
  | 10 => 5
  | 11 => 6
  | 12 => 7
  | 13 => 8
  | 14 => 9
  | _ => 0  -- For simplicity, we only define the conversion for numbers used in the square

/-- Checks if the given square is magic in base 5 -/
def isMagicSquare (s : MagicSquare) : Prop :=
  let a' := toBase10 s.a
  let b' := toBase10 s.b
  let c' := toBase10 s.c
  let d' := toBase10 s.d
  let e' := toBase10 s.e
  let f' := toBase10 s.f
  let g' := toBase10 s.g
  let h' := toBase10 s.h
  let i' := toBase10 s.i
  -- Row sums are equal
  (a' + b' + c' = d' + e' + f') ∧
  (d' + e' + f' = g' + h' + i') ∧
  -- Column sums are equal
  (a' + d' + g' = b' + e' + h') ∧
  (b' + e' + h' = c' + f' + i') ∧
  -- Diagonal sums are equal
  (a' + e' + i' = c' + e' + g')

theorem magic_square_base_5 : 
  isMagicSquare ⟨13, 1, 11, 3, 10, 12, 4, 14, 2⟩ := by
  sorry


end NUMINAMATH_CALUDE_magic_square_base_5_l552_55204


namespace NUMINAMATH_CALUDE_different_sum_of_digits_l552_55231

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Statement: For any natural number N, the sum of digits of N(N-1) is not equal to the sum of digits of (N+1)² -/
theorem different_sum_of_digits (N : ℕ) : 
  sum_of_digits (N * (N - 1)) ≠ sum_of_digits ((N + 1) ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_different_sum_of_digits_l552_55231


namespace NUMINAMATH_CALUDE_article_selling_price_l552_55268

theorem article_selling_price (cost_price : ℝ) (gain_percent : ℝ) (selling_price : ℝ) : 
  cost_price = 10 →
  gain_percent = 150 →
  selling_price = cost_price * (1 + gain_percent / 100) →
  selling_price = 25 := by
sorry

end NUMINAMATH_CALUDE_article_selling_price_l552_55268


namespace NUMINAMATH_CALUDE_average_daily_temp_range_l552_55266

def high_temps : List ℝ := [49, 62, 58, 57, 46, 60, 55]
def low_temps : List ℝ := [40, 47, 45, 41, 39, 42, 44]

def daily_range (high low : List ℝ) : List ℝ :=
  List.zipWith (·-·) high low

theorem average_daily_temp_range :
  let ranges := daily_range high_temps low_temps
  (ranges.sum / ranges.length : ℝ) = 89 / 7 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_temp_range_l552_55266


namespace NUMINAMATH_CALUDE_four_of_a_kind_count_l552_55276

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- Represents a "Four of a Kind" combination -/
structure FourOfAKind :=
  (number : ℕ)
  (fifth_card : ℕ)
  (h_number_valid : number ≤ 13)
  (h_fifth_card_valid : fifth_card ≤ 52)
  (h_fifth_card_diff : fifth_card ≠ number)

/-- The number of different "Four of a Kind" combinations in a standard deck -/
def count_four_of_a_kind (d : Deck) : ℕ :=
  13 * (d.total_cards - d.num_suits)

/-- Theorem stating that the number of "Four of a Kind" combinations is 624 -/
theorem four_of_a_kind_count (d : Deck) 
  (h_standard : d.total_cards = 52 ∧ d.num_suits = 4 ∧ d.cards_per_suit = 13) : 
  count_four_of_a_kind d = 624 := by
  sorry

end NUMINAMATH_CALUDE_four_of_a_kind_count_l552_55276


namespace NUMINAMATH_CALUDE_largest_certain_divisor_l552_55273

def is_valid_roll (roll : Finset Nat) : Prop :=
  roll.card = 7 ∧ roll ⊆ Finset.range 9 \ {0}

def product_of_roll (roll : Finset Nat) : Nat :=
  roll.prod id

theorem largest_certain_divisor :
  ∃ (n : Nat), n = 192 ∧
  (∀ (roll : Finset Nat), is_valid_roll roll → n ∣ product_of_roll roll) ∧
  (∀ (m : Nat), m > n →
    ∃ (roll : Finset Nat), is_valid_roll roll ∧ ¬(m ∣ product_of_roll roll)) :=
sorry

end NUMINAMATH_CALUDE_largest_certain_divisor_l552_55273


namespace NUMINAMATH_CALUDE_lottery_probability_l552_55256

theorem lottery_probability : 
  let powerball_count : ℕ := 30
  let luckyball_count : ℕ := 49
  let luckyball_picks : ℕ := 6
  let powerball_prob : ℚ := 1 / powerball_count
  let luckyball_prob : ℚ := 1 / (Nat.choose luckyball_count luckyball_picks)
  powerball_prob * luckyball_prob = 1 / 419512480 :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l552_55256


namespace NUMINAMATH_CALUDE_food_distribution_l552_55200

/-- Represents the number of days the initial food supply lasts -/
def initial_days : ℕ := 22

/-- Represents the number of days that pass before additional men join -/
def days_passed : ℕ := 2

/-- Represents the number of additional men who join -/
def additional_men : ℕ := 2280

/-- Represents the number of days the food lasts after additional men join -/
def remaining_days : ℕ := 5

/-- Represents the initial number of men -/
def initial_men : ℕ := 760

theorem food_distribution (M : ℕ) :
  M * (initial_days - days_passed) = (M + additional_men) * remaining_days →
  M = initial_men := by sorry

end NUMINAMATH_CALUDE_food_distribution_l552_55200


namespace NUMINAMATH_CALUDE_triangle_inequality_l552_55298

theorem triangle_inequality (a b c S r R : ℝ) (ha hb hc : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < S ∧ 0 < r ∧ 0 < R →
  9 * r ≤ ha + hb + hc →
  ha + hb + hc ≤ 9 * R / 2 →
  1 / a + 1 / b + 1 / c = (ha + hb + hc) / (2 * S) →
  9 * r / (2 * S) ≤ 1 / a + 1 / b + 1 / c ∧ 1 / a + 1 / b + 1 / c ≤ 9 * R / (4 * S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l552_55298


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l552_55223

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 0 → a^2 + a ≥ 0) ∧ 
  (∃ a, a^2 + a ≥ 0 ∧ ¬(a > 0)) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l552_55223


namespace NUMINAMATH_CALUDE_tan_and_cos_relations_l552_55249

theorem tan_and_cos_relations (θ : Real) (h : Real.tan θ = 2) :
  Real.tan (π / 4 - θ) = -1 / 3 ∧ Real.cos (2 * θ) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_and_cos_relations_l552_55249


namespace NUMINAMATH_CALUDE_xy_sum_product_l552_55239

theorem xy_sum_product (x y : ℝ) (h1 : x * y = 3) (h2 : x + y = 5) :
  x^2 * y + x * y^2 = 15 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_product_l552_55239


namespace NUMINAMATH_CALUDE_waiter_tables_l552_55271

theorem waiter_tables (total_customers : ℕ) (people_per_table : ℕ) (h1 : total_customers = 90) (h2 : people_per_table = 10) :
  total_customers / people_per_table = 9 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tables_l552_55271


namespace NUMINAMATH_CALUDE_night_heads_count_l552_55248

/-- Represents the number of animals of each type -/
structure AnimalCounts where
  chickens : ℕ
  rabbits : ℕ
  geese : ℕ

/-- Calculates the total number of legs during the day -/
def totalDayLegs (counts : AnimalCounts) : ℕ :=
  2 * counts.chickens + 4 * counts.rabbits + 2 * counts.geese

/-- Calculates the total number of heads -/
def totalHeads (counts : AnimalCounts) : ℕ :=
  counts.chickens + counts.rabbits + counts.geese

/-- Calculates the total number of legs at night -/
def totalNightLegs (counts : AnimalCounts) : ℕ :=
  2 * counts.chickens + 4 * counts.rabbits + counts.geese

/-- The main theorem to prove -/
theorem night_heads_count (counts : AnimalCounts) 
  (h1 : totalDayLegs counts = 56)
  (h2 : totalDayLegs counts - totalHeads counts = totalNightLegs counts - totalHeads counts) :
  totalHeads counts = 14 := by
  sorry


end NUMINAMATH_CALUDE_night_heads_count_l552_55248


namespace NUMINAMATH_CALUDE_abc_product_l552_55282

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 165)
  (h2 : b * (c + a) = 156)
  (h3 : c * (a + b) = 180) :
  a * b * c = 100 * Real.sqrt 39 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l552_55282


namespace NUMINAMATH_CALUDE_solve_equation_l552_55258

theorem solve_equation (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 8 → y = 40 / 3 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l552_55258


namespace NUMINAMATH_CALUDE_two_from_five_permutation_l552_55275

def permutations (n : ℕ) (r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

theorem two_from_five_permutation : 
  permutations 5 2 = 20 := by sorry

end NUMINAMATH_CALUDE_two_from_five_permutation_l552_55275


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_greater_than_five_l552_55279

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 6*x + 5) / Real.log (Real.sin 1)

theorem decreasing_f_implies_a_greater_than_five (a : ℝ) :
  (∀ x y, a < x ∧ x < y → f y < f x) →
  a > 5 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_f_implies_a_greater_than_five_l552_55279


namespace NUMINAMATH_CALUDE_division_remainder_proof_l552_55228

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 690 →
  divisor = 36 →
  quotient = 19 →
  dividend = divisor * quotient + remainder →
  remainder = 6 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l552_55228


namespace NUMINAMATH_CALUDE_debt_installments_l552_55263

theorem debt_installments (x : ℝ) : 
  (8 * x + 44 * (x + 65)) / 52 = 465 → x = 410 := by
  sorry

end NUMINAMATH_CALUDE_debt_installments_l552_55263


namespace NUMINAMATH_CALUDE_min_value_theorem_l552_55284

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 2) :
  x^2 / (2*y) + 4*y^2 / x ≥ 2 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 2*y = 2 ∧ x^2 / (2*y) + 4*y^2 / x = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l552_55284


namespace NUMINAMATH_CALUDE_georgesBirthdayMoneyIs12_l552_55288

/-- Calculates the amount George will receive on his 25th birthday --/
def georgesBirthdayMoney (currentAge : ℕ) (startAge : ℕ) (spendPercentage : ℚ) (exchangeRate : ℚ) : ℚ :=
  let totalBills : ℕ := currentAge - startAge
  let remainingBills : ℚ := (1 - spendPercentage) * totalBills
  exchangeRate * remainingBills

/-- Theorem stating the amount George will receive --/
theorem georgesBirthdayMoneyIs12 : 
  georgesBirthdayMoney 25 15 (1/5) (3/2) = 12 := by
  sorry


end NUMINAMATH_CALUDE_georgesBirthdayMoneyIs12_l552_55288


namespace NUMINAMATH_CALUDE_remainder_3056_div_32_l552_55230

theorem remainder_3056_div_32 : 3056 % 32 = 16 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3056_div_32_l552_55230


namespace NUMINAMATH_CALUDE_functional_eq_solution_l552_55215

/-- A function satisfying the given functional equation -/
def FunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - y) = 2009 * f x * f y

/-- The main theorem -/
theorem functional_eq_solution (f : ℝ → ℝ) 
  (h1 : FunctionalEq f) 
  (h2 : ∀ x : ℝ, f x ≠ 0) : 
  f (Real.sqrt 2009) = 1 / 2009 := by
  sorry

end NUMINAMATH_CALUDE_functional_eq_solution_l552_55215


namespace NUMINAMATH_CALUDE_bench_cost_is_150_l552_55232

/-- The cost of a bench and garden table, where the table costs twice as much as the bench. -/
def BenchAndTableCost (bench_cost : ℝ) : ℝ := bench_cost + 2 * bench_cost

/-- Theorem stating that the bench costs 150 dollars given the conditions. -/
theorem bench_cost_is_150 :
  ∃ (bench_cost : ℝ), BenchAndTableCost bench_cost = 450 ∧ bench_cost = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_bench_cost_is_150_l552_55232


namespace NUMINAMATH_CALUDE_function_value_l552_55212

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt (a * x - 1) else -x^2 - 4*x

theorem function_value (a : ℝ) : f a (f a (-2)) = 3 → a = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_l552_55212


namespace NUMINAMATH_CALUDE_wrong_to_right_exists_l552_55240

-- Define a type for single-digit numbers (1-9)
def Digit := {n : ℕ | 1 ≤ n ∧ n ≤ 9}

-- Define a function to convert a 5-digit number to its numerical value
def to_number (a b c d e : Digit) : ℕ :=
  10000 * a + 1000 * b + 100 * c + 10 * d + e

-- State the theorem
theorem wrong_to_right_exists :
  ∃ (W R O N G I H T : Digit),
    (W ≠ R) ∧ (W ≠ O) ∧ (W ≠ N) ∧ (W ≠ G) ∧ (W ≠ I) ∧ (W ≠ H) ∧ (W ≠ T) ∧
    (R ≠ O) ∧ (R ≠ N) ∧ (R ≠ G) ∧ (R ≠ I) ∧ (R ≠ H) ∧ (R ≠ T) ∧
    (O ≠ N) ∧ (O ≠ G) ∧ (O ≠ I) ∧ (O ≠ H) ∧ (O ≠ T) ∧
    (N ≠ G) ∧ (N ≠ I) ∧ (N ≠ H) ∧ (N ≠ T) ∧
    (G ≠ I) ∧ (G ≠ H) ∧ (G ≠ T) ∧
    (I ≠ H) ∧ (I ≠ T) ∧
    (H ≠ T) ∧
    to_number W R O N G + to_number W R O N G = to_number R I G H T :=
by sorry

end NUMINAMATH_CALUDE_wrong_to_right_exists_l552_55240


namespace NUMINAMATH_CALUDE_sqrt_two_plus_sqrt_three_gt_sqrt_five_l552_55222

theorem sqrt_two_plus_sqrt_three_gt_sqrt_five :
  Real.sqrt 2 + Real.sqrt 3 > Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_plus_sqrt_three_gt_sqrt_five_l552_55222


namespace NUMINAMATH_CALUDE_derivative_of_power_function_l552_55286

open Real

/-- Given differentiable functions u and v, where u is positive,
    f(x) = u(x)^(v(x)) is differentiable and its derivative is as stated. -/
theorem derivative_of_power_function (u v : ℝ → ℝ) (hu : Differentiable ℝ u)
    (hv : Differentiable ℝ v) (hup : ∀ x, u x > 0) :
  let f := λ x => (u x) ^ (v x)
  Differentiable ℝ f ∧ 
  ∀ x, deriv f x = (u x)^(v x) * (deriv v x * log (u x) + v x * deriv u x / u x) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_power_function_l552_55286


namespace NUMINAMATH_CALUDE_acute_triangle_median_inequality_l552_55280

/-- For an acute triangle ABC with side lengths a, b, c and corresponding median lengths m_a, m_b, m_c,
    the sum of the squared medians divided by the sum of two squared sides minus the third squared side
    is greater than or equal to 9/4. -/
theorem acute_triangle_median_inequality (a b c m_a m_b m_c : ℝ) 
  (h_acute : 0 < a ∧ 0 < b ∧ 0 < c ∧ a < b + c ∧ b < a + c ∧ c < a + b)
  (h_medians : m_a^2 = (2*b^2 + 2*c^2 - a^2)/4 ∧ 
               m_b^2 = (2*a^2 + 2*c^2 - b^2)/4 ∧ 
               m_c^2 = (2*a^2 + 2*b^2 - c^2)/4) :
  m_a^2 / (-a^2 + b^2 + c^2) + m_b^2 / (-b^2 + a^2 + c^2) + m_c^2 / (-c^2 + a^2 + b^2) ≥ 9/4 :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_median_inequality_l552_55280


namespace NUMINAMATH_CALUDE_integral_cube_root_problem_l552_55285

open Real MeasureTheory

theorem integral_cube_root_problem :
  ∫ x in (1 : ℝ)..64, (2 + x^(1/3)) / ((x^(1/6) + 2*x^(1/3) + x^(1/2)) * x^(1/2)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_integral_cube_root_problem_l552_55285


namespace NUMINAMATH_CALUDE_xy_term_vanishes_l552_55234

/-- The polynomial in question -/
def polynomial (k x y : ℝ) : ℝ := x^2 + (k-1)*x*y - 3*y^2 - 2*x*y - 5

/-- The coefficient of xy in the polynomial -/
def xy_coefficient (k : ℝ) : ℝ := k - 3

theorem xy_term_vanishes (k : ℝ) :
  xy_coefficient k = 0 ↔ k = 3 := by sorry

end NUMINAMATH_CALUDE_xy_term_vanishes_l552_55234


namespace NUMINAMATH_CALUDE_original_savings_l552_55202

/-- Proves that if a person spends 4/5 of their savings on furniture and the remaining 1/5 on a TV that costs $100, their original savings were $500. -/
theorem original_savings (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  furniture_fraction = 4/5 → 
  tv_cost = 100 → 
  (1 - furniture_fraction) * savings = tv_cost → 
  savings = 500 := by
sorry

end NUMINAMATH_CALUDE_original_savings_l552_55202


namespace NUMINAMATH_CALUDE_onion_problem_l552_55225

theorem onion_problem (initial : ℕ) (removed : ℕ) : 
  initial + 4 - removed + 9 = initial + 8 → removed = 5 := by
  sorry

end NUMINAMATH_CALUDE_onion_problem_l552_55225


namespace NUMINAMATH_CALUDE_human_habitable_area_l552_55290

/-- The fraction of Earth's surface that is not covered by water -/
def land_fraction : ℚ := 1/3

/-- The fraction of land that is inhabitable for humans -/
def inhabitable_land_fraction : ℚ := 2/3

/-- The fraction of Earth's surface that humans can live on -/
def human_habitable_fraction : ℚ := land_fraction * inhabitable_land_fraction

theorem human_habitable_area :
  human_habitable_fraction = 2/9 := by sorry

end NUMINAMATH_CALUDE_human_habitable_area_l552_55290


namespace NUMINAMATH_CALUDE_unused_edge_exists_l552_55272

/-- Represents a token on a vertex of the 2n-gon -/
structure Token (n : ℕ) where
  position : Fin (2 * n)

/-- Represents a move (swapping tokens on an edge) -/
structure Move (n : ℕ) where
  edge : Fin (2 * n) × Fin (2 * n)

/-- Represents the state of the 2n-gon after some moves -/
structure GameState (n : ℕ) where
  tokens : Fin (2 * n) → Token n
  moves : List (Move n)

/-- Predicate to check if two tokens have been swapped -/
def haveBeenSwapped (n : ℕ) (t1 t2 : Token n) (moves : List (Move n)) : Prop :=
  sorry

/-- Predicate to check if an edge has been used for swapping -/
def edgeUsed (n : ℕ) (edge : Fin (2 * n) × Fin (2 * n)) (moves : List (Move n)) : Prop :=
  sorry

/-- The main theorem -/
theorem unused_edge_exists (n : ℕ) (finalState : GameState n) :
  (∀ t1 t2 : Token n, t1 ≠ t2 → haveBeenSwapped n t1 t2 finalState.moves) →
  ∃ edge : Fin (2 * n) × Fin (2 * n), ¬edgeUsed n edge finalState.moves :=
sorry

end NUMINAMATH_CALUDE_unused_edge_exists_l552_55272


namespace NUMINAMATH_CALUDE_integer_terms_count_l552_55278

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a₃ : ℝ  -- 3rd term
  a₁₈ : ℝ -- 18th term
  h₃ : a₃ = 14
  h₁₈ : a₁₈ = 23

/-- The number of integer terms in the first 2010 terms of the sequence -/
def integerTermCount (seq : ArithmeticSequence) : ℕ :=
  402

/-- Theorem stating the number of integer terms in the first 2010 terms -/
theorem integer_terms_count (seq : ArithmeticSequence) :
  integerTermCount seq = 402 := by
  sorry

end NUMINAMATH_CALUDE_integer_terms_count_l552_55278


namespace NUMINAMATH_CALUDE_rachel_and_sarah_return_trip_money_l552_55208

theorem rachel_and_sarah_return_trip_money :
  let initial_amount : ℚ := 50
  let gasoline_cost : ℚ := 8
  let lunch_cost : ℚ := 15.65
  let gift_cost_per_person : ℚ := 5
  let grandma_gift_per_person : ℚ := 10
  let num_people : ℕ := 2

  let total_spent : ℚ := gasoline_cost + lunch_cost + (gift_cost_per_person * num_people)
  let total_received_from_grandma : ℚ := grandma_gift_per_person * num_people
  let remaining_amount : ℚ := initial_amount - total_spent + total_received_from_grandma

  remaining_amount = 36.35 :=
by
  sorry

end NUMINAMATH_CALUDE_rachel_and_sarah_return_trip_money_l552_55208


namespace NUMINAMATH_CALUDE_gcf_72_108_l552_55296

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by sorry

end NUMINAMATH_CALUDE_gcf_72_108_l552_55296


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l552_55287

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 →
  a = 30 →
  a^2 + b^2 = c^2 →
  a + b + c = 40 + 10 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l552_55287


namespace NUMINAMATH_CALUDE_fraction_product_equality_l552_55255

theorem fraction_product_equality : (3 / 4) * (5 / 9) * (8 / 13) * (3 / 7) = 10 / 91 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equality_l552_55255


namespace NUMINAMATH_CALUDE_rectangle_width_equality_l552_55238

/-- Given two rectangles of equal area, where one rectangle measures 5 inches by 24 inches
    and the other rectangle is 4 inches long, prove that the width of the second rectangle
    is 30 inches. -/
theorem rectangle_width_equality (area carol_length carol_width jordan_length : ℝ)
    (h1 : area = carol_length * carol_width)
    (h2 : carol_length = 5)
    (h3 : carol_width = 24)
    (h4 : jordan_length = 4)
    (h5 : area = jordan_length * (area / jordan_length)) :
    area / jordan_length = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_equality_l552_55238


namespace NUMINAMATH_CALUDE_zero_subset_X_l552_55260

def X : Set ℝ := {x | x > -1}

theorem zero_subset_X : {0} ⊆ X := by
  sorry

end NUMINAMATH_CALUDE_zero_subset_X_l552_55260


namespace NUMINAMATH_CALUDE_total_drivers_l552_55241

theorem total_drivers (N : ℕ) 
  (drivers_A : ℕ) 
  (sample_A sample_B sample_C sample_D : ℕ) :
  drivers_A = 96 →
  sample_A = 8 →
  sample_B = 23 →
  sample_C = 27 →
  sample_D = 43 →
  (sample_A : ℚ) / drivers_A = (sample_A + sample_B + sample_C + sample_D : ℚ) / N →
  N = 1212 :=
by sorry

end NUMINAMATH_CALUDE_total_drivers_l552_55241


namespace NUMINAMATH_CALUDE_sequence_property_l552_55245

def sequence_a (n : ℕ+) : ℚ :=
  1 / (2 * n - 1)

theorem sequence_property (n : ℕ+) :
  let a : ℕ+ → ℚ := sequence_a
  (n = 1 → a n = 1) ∧
  (∀ k : ℕ+, a k ≠ 0) ∧
  (∀ k : ℕ+, k ≥ 2 → a k + 2 * a k * a (k - 1) - a (k - 1) = 0) →
  a n = 1 / (2 * n - 1) :=
by
  sorry

#check sequence_property

end NUMINAMATH_CALUDE_sequence_property_l552_55245


namespace NUMINAMATH_CALUDE_partition_naturals_l552_55253

theorem partition_naturals (c : ℚ) (hc : c > 0) (hc_ne_one : c ≠ 1) :
  ∃ (A B : Set ℕ), (A ∪ B = Set.univ) ∧ (A ∩ B = ∅) ∧
  (∀ (a₁ a₂ : ℕ), a₁ ∈ A → a₂ ∈ A → a₁ ≠ 0 → a₂ ≠ 0 → (a₁ : ℚ) / a₂ ≠ c) ∧
  (∀ (b₁ b₂ : ℕ), b₁ ∈ B → b₂ ∈ B → b₁ ≠ 0 → b₂ ≠ 0 → (b₁ : ℚ) / b₂ ≠ c) :=
by sorry

end NUMINAMATH_CALUDE_partition_naturals_l552_55253


namespace NUMINAMATH_CALUDE_book_page_numbering_l552_55289

def total_digits (n : ℕ) : ℕ :=
  let d1 := min n 9
  let d2 := min (n - 9) 90
  let d3 := min (n - 99) 900
  let d4 := max (n - 999) 0
  d1 + 2 * d2 + 3 * d3 + 4 * d4

theorem book_page_numbering :
  total_digits 5000 = 18893 := by
sorry

end NUMINAMATH_CALUDE_book_page_numbering_l552_55289


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l552_55219

theorem min_value_quadratic_form (x y z : ℝ) :
  x^2 + x*y + y^2 + y*z + z^2 ≥ 0 ∧
  (x^2 + x*y + y^2 + y*z + z^2 = 0 ↔ x = 0 ∧ y = 0 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l552_55219


namespace NUMINAMATH_CALUDE_expression_evaluation_l552_55229

theorem expression_evaluation :
  let x : ℚ := 1/2
  let y : ℤ := -3
  3 * (x^2 - 2*x^2*y) - 3*x^2 + 2*y - 2*(x^2*y + y) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l552_55229


namespace NUMINAMATH_CALUDE_sqrt_real_range_l552_55205

theorem sqrt_real_range (x : ℝ) : (∃ y : ℝ, y^2 = x - 3) → x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_real_range_l552_55205


namespace NUMINAMATH_CALUDE_fraction_subtraction_l552_55261

theorem fraction_subtraction : (8 : ℚ) / 19 - (5 : ℚ) / 57 = (1 : ℚ) / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l552_55261


namespace NUMINAMATH_CALUDE_difference_of_squares_601_599_l552_55259

theorem difference_of_squares_601_599 : 601^2 - 599^2 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_601_599_l552_55259


namespace NUMINAMATH_CALUDE_meeting_probability_for_seven_steps_l552_55216

/-- Represents a position on the coordinate plane -/
structure Position where
  x : ℕ
  y : ℕ

/-- Represents the possible movements for an object -/
inductive Movement
  | Right
  | Up
  | Left
  | Down

/-- Represents an object on the coordinate plane -/
structure Object where
  position : Position
  allowedMovements : List Movement

/-- Calculates the number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Calculates the number of intersection paths for given number of steps -/
def intersectionPaths (steps : ℕ) : ℕ := sorry

/-- The probability of two objects meeting given their initial positions and movement constraints -/
def meetingProbability (obj1 obj2 : Object) (steps : ℕ) : ℚ := sorry

theorem meeting_probability_for_seven_steps :
  let c : Object := ⟨⟨1, 1⟩, [Movement.Right, Movement.Up]⟩
  let d : Object := ⟨⟨6, 7⟩, [Movement.Left, Movement.Down]⟩
  meetingProbability c d 7 = 1715 / 16384 := by sorry

end NUMINAMATH_CALUDE_meeting_probability_for_seven_steps_l552_55216


namespace NUMINAMATH_CALUDE_digit_sum_l552_55233

theorem digit_sum (w x y z : ℕ) : 
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
  w < 10 ∧ x < 10 ∧ y < 10 ∧ z < 10 →
  y + w = 11 →
  x + y + 1 = 10 →
  w + z + 1 = 11 →
  w + x + y + z = 20 :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_l552_55233


namespace NUMINAMATH_CALUDE_inequality_solution_range_l552_55247

theorem inequality_solution_range (m : ℝ) : 
  (∃ x : ℝ, |x + 1| + |x - 1| < m) → m > 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l552_55247


namespace NUMINAMATH_CALUDE_perfect_square_difference_l552_55221

theorem perfect_square_difference (x y : ℕ) (h : x > 0 ∧ y > 0) 
  (eq : 3 * x^2 + x = 4 * y^2 + y) : 
  ∃ (k : ℕ), x - y = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_difference_l552_55221


namespace NUMINAMATH_CALUDE_third_column_sum_l552_55206

/-- Represents a 3x3 grid of numbers -/
def Grid := Matrix (Fin 3) (Fin 3) ℤ

/-- The sum of a row in the grid -/
def row_sum (g : Grid) (i : Fin 3) : ℤ :=
  (g i 0) + (g i 1) + (g i 2)

/-- The sum of a column in the grid -/
def col_sum (g : Grid) (j : Fin 3) : ℤ :=
  (g 0 j) + (g 1 j) + (g 2 j)

/-- The theorem statement -/
theorem third_column_sum (g : Grid) 
  (h1 : row_sum g 0 = 24)
  (h2 : row_sum g 1 = 26)
  (h3 : row_sum g 2 = 40)
  (h4 : col_sum g 0 = 27)
  (h5 : col_sum g 1 = 20) :
  col_sum g 2 = 43 := by
  sorry


end NUMINAMATH_CALUDE_third_column_sum_l552_55206


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l552_55254

theorem closest_integer_to_cube_root (n : ℕ) : 
  ∃ (m : ℤ), ∀ (k : ℤ), |k - (5^3 + 9^3 : ℝ)^(1/3)| ≥ |m - (5^3 + 9^3 : ℝ)^(1/3)| ∧ m = 9 := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l552_55254


namespace NUMINAMATH_CALUDE_linear_increase_l552_55267

theorem linear_increase (f : ℝ → ℝ) (h : ∀ x, f (x + 4) - f x = 6) :
  ∀ x, f (x + 12) - f x = 18 := by
  sorry

end NUMINAMATH_CALUDE_linear_increase_l552_55267


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l552_55269

theorem reciprocal_of_negative_three :
  (1 : ℚ) / (-3 : ℚ) = -1/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l552_55269


namespace NUMINAMATH_CALUDE_rectangle_length_l552_55251

/-- Given a rectangle with width 5 feet and perimeter 22 feet, prove that its length is 6 feet. -/
theorem rectangle_length (width : ℝ) (perimeter : ℝ) (length : ℝ) : 
  width = 5 → perimeter = 22 → 2 * (length + width) = perimeter → length = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l552_55251


namespace NUMINAMATH_CALUDE_infinite_inequality_occurrences_l552_55252

theorem infinite_inequality_occurrences (a : ℕ → ℝ) (h : ∀ n, a n > 0) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, 1 + a n > a (n - 1) * (2 : ℝ)^(1/5) :=
sorry

end NUMINAMATH_CALUDE_infinite_inequality_occurrences_l552_55252


namespace NUMINAMATH_CALUDE_chessboard_coverage_l552_55214

/-- An L-shaped tetromino covers exactly 4 squares. -/
def LTetromino : ℕ := 4

/-- Represents an m × n chessboard. -/
structure Chessboard where
  m : ℕ
  n : ℕ

/-- Predicate to check if a number is divisible by 8. -/
def divisible_by_eight (x : ℕ) : Prop := ∃ k, x = 8 * k

/-- Predicate to check if a chessboard can be covered by L-shaped tetrominoes. -/
def can_cover (board : Chessboard) : Prop :=
  divisible_by_eight (board.m * board.n) ∧ board.m ≠ 1 ∧ board.n ≠ 1

theorem chessboard_coverage (board : Chessboard) :
  (∃ (tiles : ℕ), board.m * board.n = tiles * LTetromino) ↔ can_cover board :=
sorry

end NUMINAMATH_CALUDE_chessboard_coverage_l552_55214


namespace NUMINAMATH_CALUDE_parabola_vertex_l552_55274

/-- Given a quadratic function f(x) = -x^2 + cx + d whose inequality f(x) ≤ 0
    has the solution set (-∞, -4] ∪ [6, ∞), prove that its vertex is (5, 1) -/
theorem parabola_vertex (c d : ℝ) : 
  (∀ x, -x^2 + c*x + d ≤ 0 ↔ x ≤ -4 ∨ x ≥ 6) → 
  ∃ x y, x = 5 ∧ y = 1 ∧ ∀ t, -t^2 + c*t + d ≤ -(-t + x)^2 + y :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l552_55274


namespace NUMINAMATH_CALUDE_copperfield_numbers_l552_55213

theorem copperfield_numbers :
  ∃ (x₁ x₂ x₃ : ℕ) (k₁ k₂ k₃ : ℕ+),
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    x₁ * 3^(k₁.val) = x₁ + 2500 * k₁.val ∧
    x₂ * 3^(k₂.val) = x₂ + 2500 * k₂.val ∧
    x₃ * 3^(k₃.val) = x₃ + 2500 * k₃.val :=
by sorry

end NUMINAMATH_CALUDE_copperfield_numbers_l552_55213


namespace NUMINAMATH_CALUDE_min_teams_for_athletes_l552_55291

theorem min_teams_for_athletes (total_athletes : ℕ) (max_per_team : ℕ) (h1 : total_athletes = 30) (h2 : max_per_team = 9) :
  ∃ (num_teams : ℕ) (athletes_per_team : ℕ),
    num_teams * athletes_per_team = total_athletes ∧
    athletes_per_team ≤ max_per_team ∧
    num_teams = 5 ∧
    ∀ (other_num_teams : ℕ) (other_athletes_per_team : ℕ),
      other_num_teams * other_athletes_per_team = total_athletes →
      other_athletes_per_team ≤ max_per_team →
      other_num_teams ≥ num_teams :=
by sorry

end NUMINAMATH_CALUDE_min_teams_for_athletes_l552_55291


namespace NUMINAMATH_CALUDE_divisors_of_2160_l552_55201

def n : ℕ := 2160

-- Define the prime factorization of n
axiom n_factorization : n = 2^4 * 3^3 * 5

-- Define the number of positive divisors
def num_divisors (m : ℕ) : ℕ := sorry

-- Define the sum of positive divisors
def sum_divisors (m : ℕ) : ℕ := sorry

theorem divisors_of_2160 :
  (num_divisors n = 40) ∧ (sum_divisors n = 7440) := by sorry

end NUMINAMATH_CALUDE_divisors_of_2160_l552_55201


namespace NUMINAMATH_CALUDE_problem_solution_l552_55294

def f (x : ℝ) := abs (2*x - 1) - abs (2*x - 2)

theorem problem_solution :
  (∃ k : ℝ, ∀ x : ℝ, f x ≤ k) ∧
  ({x : ℝ | f x ≥ x} = {x : ℝ | x ≤ -1 ∨ x = 1}) ∧
  (∀ x : ℝ, f x ≤ 1) ∧
  (¬∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 2*b = 1 ∧ 2/a + 1/b = 4 - 1/(a*b)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l552_55294


namespace NUMINAMATH_CALUDE_person_b_work_days_l552_55292

/-- Given that person A can complete a work in 30 days, and together with person B
    they complete 2/9 of the work in 4 days, prove that person B can complete
    the work alone in 45 days. -/
theorem person_b_work_days (a_days : ℕ) (combined_work : ℚ) (combined_days : ℕ) :
  a_days = 30 →
  combined_work = 2 / 9 →
  combined_days = 4 →
  ∃ b_days : ℕ,
    b_days = 45 ∧
    combined_work = combined_days * (1 / a_days + 1 / b_days) :=
by sorry

end NUMINAMATH_CALUDE_person_b_work_days_l552_55292


namespace NUMINAMATH_CALUDE_coffee_stock_problem_l552_55235

/-- Represents the coffee stock problem --/
theorem coffee_stock_problem 
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (new_decaf_percent : ℝ)
  (final_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 0.4)
  (h3 : new_decaf_percent = 0.6)
  (h4 : final_decaf_percent = 0.44)
  : ∃ (additional_coffee : ℝ),
    additional_coffee = 100 ∧
    (initial_stock * initial_decaf_percent + additional_coffee * new_decaf_percent) / (initial_stock + additional_coffee) = final_decaf_percent :=
by sorry


end NUMINAMATH_CALUDE_coffee_stock_problem_l552_55235


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l552_55295

/-- The radius of the inscribed circle in a rhombus with given diagonals -/
theorem inscribed_circle_radius_rhombus (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) :
  let a := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  let area := d1 * d2 / 2
  area / (4 * a) = 30 / Real.sqrt 241 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l552_55295


namespace NUMINAMATH_CALUDE_ceil_e_plus_pi_l552_55209

theorem ceil_e_plus_pi : ⌈Real.exp 1 + Real.pi⌉ = 6 := by sorry

end NUMINAMATH_CALUDE_ceil_e_plus_pi_l552_55209


namespace NUMINAMATH_CALUDE_student_failed_marks_l552_55217

def total_marks : ℕ := 300
def passing_percentage : ℚ := 60 / 100
def student_marks : ℕ := 160

theorem student_failed_marks :
  (passing_percentage * total_marks : ℚ).ceil - student_marks = 20 := by
  sorry

end NUMINAMATH_CALUDE_student_failed_marks_l552_55217


namespace NUMINAMATH_CALUDE_journey_time_ratio_l552_55250

/-- Represents a two-part journey with given speeds and times -/
structure Journey where
  v : ℝ  -- Initial speed
  t : ℝ  -- Initial time
  total_distance : ℝ  -- Total distance traveled

/-- The theorem statement -/
theorem journey_time_ratio 
  (j : Journey) 
  (h1 : j.v = 30)  -- Initial speed is 30 mph
  (h2 : j.v * j.t + (2 * j.v) * (2 * j.t) = j.total_distance)  -- Total distance equation
  (h3 : j.total_distance = 75)  -- Total distance is 75 miles
  : j.t / (2 * j.t) = 1 / 2 := by
  sorry

#check journey_time_ratio

end NUMINAMATH_CALUDE_journey_time_ratio_l552_55250


namespace NUMINAMATH_CALUDE_new_boat_travel_distance_l552_55218

/-- Calculates the distance traveled by a new boat given the speed increase and the distance traveled by an old boat -/
def new_boat_distance (speed_increase : ℝ) (old_distance : ℝ) : ℝ :=
  old_distance * (1 + speed_increase)

/-- Theorem: Given a new boat traveling 30% faster than an old boat, and the old boat traveling 150 miles,
    the new boat will travel 195 miles in the same time -/
theorem new_boat_travel_distance :
  new_boat_distance 0.3 150 = 195 := by
  sorry

#eval new_boat_distance 0.3 150

end NUMINAMATH_CALUDE_new_boat_travel_distance_l552_55218


namespace NUMINAMATH_CALUDE_projected_revenue_increase_l552_55299

theorem projected_revenue_increase (actual_decrease : Real) (actual_to_projected_ratio : Real) 
  (h1 : actual_decrease = 0.3)
  (h2 : actual_to_projected_ratio = 0.5) :
  ∃ (projected_increase : Real), 
    (1 - actual_decrease) = actual_to_projected_ratio * (1 + projected_increase) ∧ 
    projected_increase = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_projected_revenue_increase_l552_55299


namespace NUMINAMATH_CALUDE_unique_triple_solution_l552_55210

theorem unique_triple_solution (a b c : ℝ) : 
  a > 2 ∧ b > 2 ∧ c > 2 ∧ 
  ((a + 3)^2) / (b + c - 3) + ((b + 5)^2) / (c + a - 5) + ((c + 7)^2) / (a + b - 7) = 45 →
  a = 13 ∧ b = 11 ∧ c = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l552_55210


namespace NUMINAMATH_CALUDE_standard_deviation_of_commute_times_l552_55224

def commute_times : List ℝ := [12, 8, 10, 11, 9]

theorem standard_deviation_of_commute_times :
  let n : ℕ := commute_times.length
  let mean : ℝ := (commute_times.sum) / n
  let variance : ℝ := (commute_times.map (λ x => (x - mean)^2)).sum / n
  Real.sqrt variance = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_standard_deviation_of_commute_times_l552_55224


namespace NUMINAMATH_CALUDE_jerry_color_cartridges_l552_55226

/-- Represents the cost of a color cartridge in dollars -/
def color_cartridge_cost : ℕ := 32

/-- Represents the cost of a black-and-white cartridge in dollars -/
def bw_cartridge_cost : ℕ := 27

/-- Represents the total amount Jerry pays in dollars -/
def total_cost : ℕ := 123

/-- Represents the number of black-and-white cartridges Jerry needs -/
def bw_cartridges : ℕ := 1

theorem jerry_color_cartridges :
  ∃ (c : ℕ), c * color_cartridge_cost + bw_cartridges * bw_cartridge_cost = total_cost ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_jerry_color_cartridges_l552_55226


namespace NUMINAMATH_CALUDE_zeros_of_f_l552_55265

def f (x : ℝ) : ℝ := (x^2 - 3*x) * (x + 4)

theorem zeros_of_f : {x : ℝ | f x = 0} = {0, 3, -4} := by sorry

end NUMINAMATH_CALUDE_zeros_of_f_l552_55265


namespace NUMINAMATH_CALUDE_largest_coefficient_binomial_expansion_l552_55220

theorem largest_coefficient_binomial_expansion :
  ∀ n : ℕ, 
    n ≤ 11 → 
    (Nat.choose 11 n : ℚ) ≤ (Nat.choose 11 6 : ℚ) ∧
    (Nat.choose 11 6 : ℚ) = (Nat.choose 11 5 : ℚ) ∧
    (∀ k : ℕ, k < 5 → (Nat.choose 11 k : ℚ) < (Nat.choose 11 6 : ℚ)) :=
by
  sorry

#check largest_coefficient_binomial_expansion

end NUMINAMATH_CALUDE_largest_coefficient_binomial_expansion_l552_55220


namespace NUMINAMATH_CALUDE_max_value_a_l552_55257

theorem max_value_a (a b c d : ℝ) 
  (h1 : b + c + d = 3 - a) 
  (h2 : 2 * b^2 + 3 * c^2 + 6 * d^2 = 5 - a^2) : 
  a ≤ 2 ∧ ∃ (b c d : ℝ), b + c + d = 3 - 2 ∧ 2 * b^2 + 3 * c^2 + 6 * d^2 = 5 - 2^2 :=
sorry

end NUMINAMATH_CALUDE_max_value_a_l552_55257


namespace NUMINAMATH_CALUDE_fisherman_catch_l552_55270

theorem fisherman_catch (bass : ℕ) (trout : ℕ) (blue_gill : ℕ) : 
  bass = 32 → 
  trout = bass / 4 → 
  blue_gill = 2 * bass → 
  bass + trout + blue_gill = 104 := by
  sorry

end NUMINAMATH_CALUDE_fisherman_catch_l552_55270


namespace NUMINAMATH_CALUDE_calculation_proof_l552_55242

theorem calculation_proof : (30 / (7 + 2 - 6)) * 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l552_55242


namespace NUMINAMATH_CALUDE_union_complement_problem_l552_55297

theorem union_complement_problem (U A B : Set Nat) :
  U = {1, 2, 3, 4} →
  A = {1, 2} →
  B = {2, 3} →
  A ∪ (U \ B) = {1, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_problem_l552_55297


namespace NUMINAMATH_CALUDE_third_term_binomial_expansion_l552_55246

theorem third_term_binomial_expansion (x : ℝ) : 
  let n : ℕ := 4
  let a : ℝ := x
  let b : ℝ := 2
  let r : ℕ := 2
  let binomial_coeff := Nat.choose n r
  let power_term := a^(n - r) * b^r
  binomial_coeff * power_term = 24 * x^2 := by
sorry


end NUMINAMATH_CALUDE_third_term_binomial_expansion_l552_55246


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l552_55293

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n.choose 2) - n

/-- Theorem: A regular nine-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l552_55293


namespace NUMINAMATH_CALUDE_divisibility_problem_l552_55203

theorem divisibility_problem (N : ℕ) : 
  N = 7 * 13 + 1 → (N / 8 + N % 8 = 15) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l552_55203


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l552_55262

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 1) : 
  Complex.im z = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l552_55262


namespace NUMINAMATH_CALUDE_segment_length_ratio_l552_55243

/-- Given two line segments with points placed at equal intervals, 
    prove that the longer segment is 101 times the length of the shorter segment. -/
theorem segment_length_ratio 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_points_a : ∃ (d : ℝ), a = 99 * d ∧ d > 0) 
  (h_points_b : ∃ (d : ℝ), b = 9999 * d ∧ d > 0) 
  (h_same_interval : ∀ (d1 d2 : ℝ), (a = 99 * d1 ∧ d1 > 0) → (b = 9999 * d2 ∧ d2 > 0) → d1 = d2) :
  b = 101 * a := by
  sorry

end NUMINAMATH_CALUDE_segment_length_ratio_l552_55243


namespace NUMINAMATH_CALUDE_units_digit_of_sqrt_product_sum_last_three_digits_of_2012_cubed_l552_55264

-- Define the function to calculate the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the function to calculate the sum of the last 3 digits
def sumLastThreeDigits (n : ℕ) : ℕ := (n % 1000) / 100 + ((n % 100) / 10) + (n % 10)

-- Theorem 1
theorem units_digit_of_sqrt_product (Q : ℤ) :
  ∃ X : ℕ, X^2 = (100 * 102 * 103 * 105 + (Q - 3)) ∧ unitsDigit X = 3 := by sorry

-- Theorem 2
theorem sum_last_three_digits_of_2012_cubed :
  sumLastThreeDigits (2012^3) = 17 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sqrt_product_sum_last_three_digits_of_2012_cubed_l552_55264


namespace NUMINAMATH_CALUDE_shoes_per_person_l552_55227

theorem shoes_per_person (num_pairs : ℕ) (num_people : ℕ) : 
  num_pairs = 36 → num_people = 36 → (num_pairs * 2) / num_people = 2 := by
  sorry

end NUMINAMATH_CALUDE_shoes_per_person_l552_55227


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l552_55236

theorem expression_simplification_and_evaluation :
  ∀ a : ℤ, -Real.sqrt 2 < (a : ℝ) ∧ (a : ℝ) < Real.sqrt 5 →
  (a ≠ -1 ∧ a ≠ 2) →
  ((a - 1 - 3 / (a + 1)) / ((a^2 - 4*a + 4) / (a + 1)) = (a + 2) / (a - 2)) ∧
  ((a + 2) / (a - 2) = -1 ∨ (a + 2) / (a - 2) = -3) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l552_55236


namespace NUMINAMATH_CALUDE_distribution_scheme_count_l552_55244

/-- The number of ways to distribute spots among schools -/
def distribute_spots (total_spots : ℕ) (num_schools : ℕ) (distribution : List ℕ) : ℕ :=
  if total_spots = distribution.sum ∧ num_schools = distribution.length
  then Nat.factorial num_schools
  else 0

theorem distribution_scheme_count :
  distribute_spots 10 4 [1, 2, 3, 4] = 24 := by
  sorry

end NUMINAMATH_CALUDE_distribution_scheme_count_l552_55244


namespace NUMINAMATH_CALUDE_trapezoid_bases_l552_55211

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- Diameter of the inscribed circle -/
  diameter : ℝ
  /-- Length of the leg (non-parallel side) -/
  leg : ℝ
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- Length of the shorter base -/
  shorterBase : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The trapezoid is circumscribed around the circle -/
  isCircumscribed : True

/-- Theorem stating the lengths of the bases for the given trapezoid -/
theorem trapezoid_bases (t : IsoscelesTrapezoid) 
    (h1 : t.diameter = 15)
    (h2 : t.leg = 17) :
    t.longerBase = 25 ∧ t.shorterBase = 9 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_bases_l552_55211
