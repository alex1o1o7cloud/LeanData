import Mathlib

namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l3671_367195

def A (x y : ℝ) : Set ℝ := {x, y / x, 1}
def B (x y : ℝ) : Set ℝ := {x^2, x + y, 0}

theorem set_equality_implies_sum (x y : ℝ) (h : A x y = B x y) : x^2023 + y^2024 = -1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l3671_367195


namespace NUMINAMATH_CALUDE_set_B_determination_l3671_367103

theorem set_B_determination (U A B : Set ℕ) : 
  U = A ∪ B ∧ 
  U = {x : ℕ | 0 ≤ x ∧ x ≤ 10} ∧ 
  A ∩ (U \ B) = {1, 3, 5, 7} → 
  B = {0, 2, 4, 6, 8, 9, 10} := by
  sorry

end NUMINAMATH_CALUDE_set_B_determination_l3671_367103


namespace NUMINAMATH_CALUDE_rational_number_equation_l3671_367149

theorem rational_number_equation (A B : ℝ) (x : ℚ) :
  x = (1 / 2) * x + (1 / 5) * ((3 / 4) * (A + B) - (2 / 3) * (A + B)) →
  x = (1 / 30) * (A + B) := by
  sorry

end NUMINAMATH_CALUDE_rational_number_equation_l3671_367149


namespace NUMINAMATH_CALUDE_dave_final_tickets_l3671_367101

def arcade_tickets (initial_tickets : ℕ) (candy_cost : ℕ) (beanie_cost : ℕ) (racing_game_win : ℕ) : ℕ :=
  let remaining_tickets := initial_tickets - (candy_cost + beanie_cost)
  let tickets_before_challenge := remaining_tickets + racing_game_win
  2 * tickets_before_challenge

theorem dave_final_tickets :
  arcade_tickets 11 3 5 10 = 26 := by
  sorry

end NUMINAMATH_CALUDE_dave_final_tickets_l3671_367101


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l3671_367172

theorem min_value_x_plus_2y (x y : ℝ) 
  (h1 : x > -1) 
  (h2 : y > 0) 
  (h3 : 1 / (x + 1) + 2 / y = 1) : 
  ∀ z, x + 2 * y ≤ z → 8 ≤ z :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l3671_367172


namespace NUMINAMATH_CALUDE_leading_coefficient_is_negative_fourteen_l3671_367138

def polynomial (x : ℝ) : ℝ := -5 * (x^5 - x^4 + 2*x) + 9 * (x^5 + 3) - 6 * (3*x^5 + x^3 + 2)

theorem leading_coefficient_is_negative_fourteen :
  ∃ (a : ℝ) (p : ℝ → ℝ), (∀ x, polynomial x = a * x^5 + p x) ∧ (∀ x, x ≠ 0 → |p x| / |x|^5 < 1) ∧ a = -14 :=
sorry

end NUMINAMATH_CALUDE_leading_coefficient_is_negative_fourteen_l3671_367138


namespace NUMINAMATH_CALUDE_max_value_z_l3671_367168

theorem max_value_z (x y : ℝ) (h1 : x - y + 1 ≤ 0) (h2 : x - 2*y ≤ 0) (h3 : x + 2*y - 2 ≤ 0) :
  ∀ z, z = x + y → z ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_z_l3671_367168


namespace NUMINAMATH_CALUDE_radius_vector_coordinates_l3671_367189

/-- Given a point M with coordinates (-2, 5, 0) in a rectangular coordinate system,
    prove that the coordinates of its radius vector OM are equal to (-2, 5, 0). -/
theorem radius_vector_coordinates (M : ℝ × ℝ × ℝ) (h : M = (-2, 5, 0)) :
  M = (-2, 5, 0) := by sorry

end NUMINAMATH_CALUDE_radius_vector_coordinates_l3671_367189


namespace NUMINAMATH_CALUDE_exists_zero_of_f_n_l3671_367173

/-- The function f(x) = x^2 + 2017x + 1 -/
def f (x : ℝ) : ℝ := x^2 + 2017*x + 1

/-- n-fold composition of f -/
def f_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n+1 => f (f_n n x)

/-- For any positive integer n, there exists a real x such that f_n(x) = 0 -/
theorem exists_zero_of_f_n (n : ℕ) (hn : n ≥ 1) : ∃ x : ℝ, f_n n x = 0 := by
  sorry


end NUMINAMATH_CALUDE_exists_zero_of_f_n_l3671_367173


namespace NUMINAMATH_CALUDE_largest_six_digit_number_with_divisibility_l3671_367118

theorem largest_six_digit_number_with_divisibility (A : ℕ) : 
  A ≤ 999999 ∧ 
  A ≥ 100000 ∧
  A % 19 = 0 ∧ 
  (A / 10) % 17 = 0 ∧ 
  (A / 100) % 13 = 0 →
  A ≤ 998412 :=
by sorry

end NUMINAMATH_CALUDE_largest_six_digit_number_with_divisibility_l3671_367118


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_32_l3671_367115

theorem smallest_positive_multiple_of_32 :
  ∀ n : ℕ, n > 0 → 32 * n ≥ 32 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_32_l3671_367115


namespace NUMINAMATH_CALUDE_round_table_seats_l3671_367160

/-- A round table with equally spaced seats numbered clockwise. -/
structure RoundTable where
  num_seats : ℕ
  seat_numbers : Fin num_seats → ℕ
  seat_numbers_clockwise : ∀ (i j : Fin num_seats), i < j → seat_numbers i < seat_numbers j

/-- Two seats are opposite if they are half the total number of seats apart. -/
def are_opposite (t : RoundTable) (s1 s2 : Fin t.num_seats) : Prop :=
  (s2.val + t.num_seats / 2) % t.num_seats = s1.val

theorem round_table_seats (t : RoundTable) (s1 s2 : Fin t.num_seats) :
  t.seat_numbers s1 = 10 →
  t.seat_numbers s2 = 29 →
  are_opposite t s1 s2 →
  t.num_seats = 38 := by
  sorry

end NUMINAMATH_CALUDE_round_table_seats_l3671_367160


namespace NUMINAMATH_CALUDE_tens_digit_of_3_to_100_l3671_367199

/-- The function that computes the last two digits of 3^n -/
def lastTwoDigits (n : ℕ) : ℕ := (3^n) % 100

/-- The cycle length of the last two digits of 3^n -/
def cycleLengthLastTwoDigits : ℕ := 20

/-- The tens digit of a number -/
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

theorem tens_digit_of_3_to_100 : tensDigit (lastTwoDigits 100) = 0 := by sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_to_100_l3671_367199


namespace NUMINAMATH_CALUDE_arithmetic_sequence_reaches_negative_27_l3671_367191

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_reaches_negative_27 :
  ∃ n : ℕ, arithmetic_sequence 1 (-2) n = -27 ∧ n = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_reaches_negative_27_l3671_367191


namespace NUMINAMATH_CALUDE_seventieth_pair_is_4_9_l3671_367143

/-- Represents a pair of integers -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- Calculates the sum of the numbers in a pair -/
def pairSum (p : IntPair) : ℕ := p.first + p.second

/-- Generates the nth pair in the sequence -/
def nthPair (n : ℕ) : IntPair :=
  sorry

/-- Calculates the total number of pairs up to and including pairs with sum k -/
def totalPairsUpToSum (k : ℕ) : ℕ :=
  sorry

theorem seventieth_pair_is_4_9 : nthPair 70 = IntPair.mk 4 9 := by
  sorry

end NUMINAMATH_CALUDE_seventieth_pair_is_4_9_l3671_367143


namespace NUMINAMATH_CALUDE_balance_after_transactions_l3671_367198

def football_club_balance (initial_balance : ℝ) (players_sold : ℕ) (selling_price : ℝ) (players_bought : ℕ) (buying_price : ℝ) : ℝ :=
  initial_balance + players_sold * selling_price - players_bought * buying_price

theorem balance_after_transactions :
  football_club_balance 100 2 10 4 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_balance_after_transactions_l3671_367198


namespace NUMINAMATH_CALUDE_min_max_values_l3671_367136

theorem min_max_values (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 1) :
  (1 / a + 1 / b ≥ 9) ∧ (a * b ≤ 1 / 16) := by sorry

end NUMINAMATH_CALUDE_min_max_values_l3671_367136


namespace NUMINAMATH_CALUDE_exclusive_or_implication_l3671_367176

theorem exclusive_or_implication :
  let statement1 := ¬p ∧ ¬q
  let statement2 := ¬p ∧ q
  let statement3 := p ∧ ¬q
  let statement4 := p ∧ q
  let exclusive_condition := ¬(p ∧ q)
  (statement1 → exclusive_condition) ∧
  (statement2 → exclusive_condition) ∧
  (statement3 → exclusive_condition) ∧
  ¬(statement4 → exclusive_condition) := by
  sorry

end NUMINAMATH_CALUDE_exclusive_or_implication_l3671_367176


namespace NUMINAMATH_CALUDE_symmetric_periodic_function_properties_l3671_367126

open Real

/-- A function satisfying specific symmetry and periodicity properties -/
structure SymmetricPeriodicFunction (a c d : ℝ) where
  f : ℝ → ℝ
  even_at_a : ∀ x, f (a + x) = f (a - x)
  sum_at_c : ∀ x, f (c + x) + f (c - x) = 2 * d
  a_neq_c : a ≠ c

theorem symmetric_periodic_function_properties
  {a c d : ℝ} (spf : SymmetricPeriodicFunction a c d) :
  (∀ x, (deriv spf.f) (c + x) = (deriv spf.f) (c - x)) ∧
  (∀ x, spf.f (x + 2 * |c - a|) = 2 * d - spf.f x) ∧
  (∀ x, spf.f (spf.f (a + x)) = spf.f (spf.f (a - x))) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_periodic_function_properties_l3671_367126


namespace NUMINAMATH_CALUDE_focus_of_our_parabola_l3671_367129

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  /-- The equation of the parabola in the form y = ax² + bx + c -/
  equation : ℝ → ℝ → Prop

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of a parabola -/
def focus (p : Parabola) : Point :=
  sorry

/-- The parabola defined by x² = y -/
def our_parabola : Parabola :=
  { equation := fun x y ↦ x^2 = y }

/-- Theorem stating that the focus of our parabola is at (0, 1) -/
theorem focus_of_our_parabola :
  focus our_parabola = Point.mk 0 1 := by
  sorry

end NUMINAMATH_CALUDE_focus_of_our_parabola_l3671_367129


namespace NUMINAMATH_CALUDE_jim_caught_two_fish_l3671_367171

def fish_problem (ben judy billy susie jim : ℕ) : Prop :=
  ben = 4 ∧
  judy = 1 ∧
  billy = 3 ∧
  susie = 5 ∧
  ∃ (thrown_back : ℕ), thrown_back = 3 ∧
  ∃ (total_filets : ℕ), total_filets = 24 ∧
  (ben + judy + billy + susie + jim - thrown_back) * 2 = total_filets

theorem jim_caught_two_fish :
  ∀ ben judy billy susie jim : ℕ,
  fish_problem ben judy billy susie jim →
  jim = 2 :=
by sorry

end NUMINAMATH_CALUDE_jim_caught_two_fish_l3671_367171


namespace NUMINAMATH_CALUDE_calculation_difference_l3671_367154

def correct_calculation : ℤ := 12 - (3 * 4 + 2)

def incorrect_calculation : ℤ := 12 - 3 * 4 + 2

theorem calculation_difference :
  correct_calculation - incorrect_calculation = -4 := by
  sorry

end NUMINAMATH_CALUDE_calculation_difference_l3671_367154


namespace NUMINAMATH_CALUDE_evaluate_expression_l3671_367146

theorem evaluate_expression (a : ℝ) : 
  let x := a + 9
  (x - a + 3)^2 = 144 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3671_367146


namespace NUMINAMATH_CALUDE_right_triangle_equation_roots_l3671_367137

theorem right_triangle_equation_roots (a b c : ℝ) (h_right_angle : a^2 + c^2 = b^2) :
  ∃ (x : ℝ), ¬ (∀ (y : ℝ), a * (y^2 - 1) - 2 * y + b * (y^2 + 1) = 0 ↔ x = y) ∧
             ¬ (∀ (y z : ℝ), a * (y^2 - 1) - 2 * y + b * (y^2 + 1) = 0 ∧
                             a * (z^2 - 1) - 2 * z + b * (z^2 + 1) = 0 → y ≠ z) ∧
             ¬ (¬ ∃ (y : ℝ), a * (y^2 - 1) - 2 * y + b * (y^2 + 1) = 0) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_equation_roots_l3671_367137


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l3671_367147

theorem trigonometric_expression_value (α : Real) (h : α = -35 * π / 6) :
  (2 * Real.sin (π + α) * Real.cos (π - α) - Real.cos (π + α)) /
  (1 + Real.sin α ^ 2 + Real.sin (π - α) - Real.cos (π + α) ^ 2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l3671_367147


namespace NUMINAMATH_CALUDE_fruit_juice_mixture_proof_l3671_367144

/-- Proves that adding 0.4 liters of pure fruit juice to a 2-liter mixture
    that is 10% pure fruit juice results in a final mixture that is 25% pure fruit juice. -/
theorem fruit_juice_mixture_proof :
  let initial_volume : ℝ := 2
  let initial_concentration : ℝ := 0.1
  let target_concentration : ℝ := 0.25
  let added_juice : ℝ := 0.4
  let final_volume : ℝ := initial_volume + added_juice
  let final_juice_amount : ℝ := initial_volume * initial_concentration + added_juice
  final_juice_amount / final_volume = target_concentration :=
by sorry

end NUMINAMATH_CALUDE_fruit_juice_mixture_proof_l3671_367144


namespace NUMINAMATH_CALUDE_object_length_increase_l3671_367169

/-- The daily increase factor for the object's length on day n -/
def daily_factor (n : ℕ) : ℚ := (n + 3) / (n + 2)

/-- The total multiplication factor after n days -/
def total_factor (n : ℕ) : ℚ := (n + 3) / 3

theorem object_length_increase (n : ℕ) : 
  n = 147 → total_factor n = 50 := by
  sorry

end NUMINAMATH_CALUDE_object_length_increase_l3671_367169


namespace NUMINAMATH_CALUDE_function_triple_is_linear_l3671_367177

/-- A triple of injective functions from ℝ to ℝ satisfying specific conditions -/
structure FunctionTriple where
  f : ℝ → ℝ
  g : ℝ → ℝ
  h : ℝ → ℝ
  f_injective : Function.Injective f
  g_injective : Function.Injective g
  h_injective : Function.Injective h
  eq1 : ∀ x y, f (x + f y) = g x + h y
  eq2 : ∀ x y, g (x + g y) = h x + f y
  eq3 : ∀ x y, h (x + h y) = f x + g y

/-- The main theorem stating that any FunctionTriple consists of linear functions with the same constant term -/
theorem function_triple_is_linear (t : FunctionTriple) : 
  ∃ C : ℝ, ∀ x : ℝ, t.f x = x + C ∧ t.g x = x + C ∧ t.h x = x + C := by
  sorry


end NUMINAMATH_CALUDE_function_triple_is_linear_l3671_367177


namespace NUMINAMATH_CALUDE_isabel_earnings_l3671_367139

def bead_necklaces : ℕ := 3
def gemstone_necklaces : ℕ := 3
def bead_price : ℚ := 4
def gemstone_price : ℚ := 8
def sales_tax_rate : ℚ := 0.05
def discount_rate : ℚ := 0.10

def total_earned : ℚ :=
  let total_before_tax := bead_necklaces * bead_price + gemstone_necklaces * gemstone_price
  let tax_amount := total_before_tax * sales_tax_rate
  let total_after_tax := total_before_tax + tax_amount
  let discount_amount := total_after_tax * discount_rate
  total_after_tax - discount_amount

theorem isabel_earnings : total_earned = 34.02 := by
  sorry

end NUMINAMATH_CALUDE_isabel_earnings_l3671_367139


namespace NUMINAMATH_CALUDE_sum_of_abc_l3671_367174

theorem sum_of_abc (a b c : ℝ) 
  (h1 : a^2 - 2*b = -2) 
  (h2 : b^2 + 6*c = 7) 
  (h3 : c^2 - 8*a = -31) : 
  a + b + c = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_abc_l3671_367174


namespace NUMINAMATH_CALUDE_four_digit_number_count_special_four_digit_number_count_l3671_367184

/-- The set of available digits --/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- A four-digit number with no repeating digits --/
structure FourDigitNumber where
  d₁ : Nat
  d₂ : Nat
  d₃ : Nat
  d₄ : Nat
  h₁ : d₁ ∈ digits
  h₂ : d₂ ∈ digits
  h₃ : d₃ ∈ digits
  h₄ : d₄ ∈ digits
  h₅ : d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄
  h₆ : d₁ ≠ 0  -- Ensures it's a four-digit number

/-- The set of all valid four-digit numbers --/
def allFourDigitNumbers : Finset FourDigitNumber := sorry

/-- Four-digit numbers with tens digit larger than both units and hundreds digits --/
def specialFourDigitNumbers : Finset FourDigitNumber :=
  allFourDigitNumbers.filter (fun n => n.d₃ > n.d₂ ∧ n.d₃ > n.d₄)

theorem four_digit_number_count :
  Finset.card allFourDigitNumbers = 300 := by sorry

theorem special_four_digit_number_count :
  Finset.card specialFourDigitNumbers = 100 := by sorry

end NUMINAMATH_CALUDE_four_digit_number_count_special_four_digit_number_count_l3671_367184


namespace NUMINAMATH_CALUDE_cost_calculation_l3671_367180

/-- The total cost of buying bread and drinks -/
def total_cost (a b : ℝ) : ℝ := 2 * a + 3 * b

/-- Theorem: The total cost of buying 2 pieces of bread at 'a' yuan each 
    and 3 bottles of drink at 'b' yuan each is equal to 2a+3b yuan -/
theorem cost_calculation (a b : ℝ) : 
  total_cost a b = 2 * a + 3 * b := by sorry

end NUMINAMATH_CALUDE_cost_calculation_l3671_367180


namespace NUMINAMATH_CALUDE_mod_inverse_of_5_mod_33_l3671_367153

theorem mod_inverse_of_5_mod_33 : ∃ x : ℕ, x ≥ 0 ∧ x ≤ 32 ∧ (5 * x) % 33 = 1 := by
  sorry

end NUMINAMATH_CALUDE_mod_inverse_of_5_mod_33_l3671_367153


namespace NUMINAMATH_CALUDE_circle_largest_area_l3671_367112

-- Define the shapes
def triangle_area (side : Real) (angle1 : Real) (angle2 : Real) : Real :=
  -- Area calculation for triangle
  sorry

def rhombus_area (d1 : Real) (d2 : Real) (angle : Real) : Real :=
  -- Area calculation for rhombus
  sorry

def circle_area (radius : Real) : Real :=
  -- Area calculation for circle
  sorry

def square_area (diagonal : Real) : Real :=
  -- Area calculation for square
  sorry

-- Theorem statement
theorem circle_largest_area :
  let triangle_a := triangle_area (Real.sqrt 2) (60 * π / 180) (45 * π / 180)
  let rhombus_a := rhombus_area (Real.sqrt 2) (Real.sqrt 3) (75 * π / 180)
  let circle_a := circle_area 1
  let square_a := square_area 2.5
  circle_a > triangle_a ∧ circle_a > rhombus_a ∧ circle_a > square_a :=
by sorry


end NUMINAMATH_CALUDE_circle_largest_area_l3671_367112


namespace NUMINAMATH_CALUDE_cost_per_load_is_25_cents_l3671_367133

/-- Calculates the cost per load in cents when buying detergent on sale -/
def cost_per_load_cents (loads_per_bottle : ℕ) (sale_price_per_bottle : ℚ) : ℚ :=
  let total_cost := 2 * sale_price_per_bottle
  let total_loads := 2 * loads_per_bottle
  (total_cost / total_loads) * 100

/-- Theorem stating that the cost per load is 25 cents under given conditions -/
theorem cost_per_load_is_25_cents (loads_per_bottle : ℕ) (sale_price_per_bottle : ℚ) 
    (h1 : loads_per_bottle = 80)
    (h2 : sale_price_per_bottle = 20) :
  cost_per_load_cents loads_per_bottle sale_price_per_bottle = 25 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_load_is_25_cents_l3671_367133


namespace NUMINAMATH_CALUDE_investment_interest_proof_l3671_367183

/-- Calculates the simple interest earned on an investment. -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_interest_proof (total_investment : ℝ) (rate1 : ℝ) (rate2 : ℝ) 
  (investment1 : ℝ) (time : ℝ) 
  (h1 : total_investment = 15000)
  (h2 : rate1 = 0.06)
  (h3 : rate2 = 0.075)
  (h4 : investment1 = 8200)
  (h5 : time = 1)
  (h6 : investment1 ≤ total_investment) :
  simple_interest investment1 rate1 time + 
  simple_interest (total_investment - investment1) rate2 time = 1002 := by
  sorry

#check investment_interest_proof

end NUMINAMATH_CALUDE_investment_interest_proof_l3671_367183


namespace NUMINAMATH_CALUDE_max_d_is_one_l3671_367186

def a (n : ℕ+) : ℕ := 100 + n^3

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_is_one : ∀ n : ℕ+, d n = 1 := by sorry

end NUMINAMATH_CALUDE_max_d_is_one_l3671_367186


namespace NUMINAMATH_CALUDE_money_division_l3671_367167

/-- Proof that the total sum of money is $320 given the specified conditions -/
theorem money_division (a b c d : ℝ) : 
  (∀ (x : ℝ), b = 0.75 * x → c = 0.5 * x → d = 0.25 * x → a = x) →
  c = 64 →
  a + b + c + d = 320 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l3671_367167


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3671_367178

theorem product_of_three_numbers (a b c : ℝ) : 
  a + b + c = 45 ∧ 
  a = 2 * (b + c) ∧ 
  c = 4 * b → 
  a * b * c = 1080 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3671_367178


namespace NUMINAMATH_CALUDE_work_completion_time_l3671_367152

theorem work_completion_time 
  (total_pay : ℝ) 
  (a_time : ℝ) 
  (b_time : ℝ) 
  (c_pay : ℝ) : 
  total_pay = 500 →
  a_time = 5 →
  b_time = 10 →
  c_pay = 200 →
  ∃ (completion_time : ℝ),
    completion_time = 2 ∧
    (1 / completion_time) = (1 / a_time) + (1 / b_time) + (c_pay / total_pay) / completion_time :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3671_367152


namespace NUMINAMATH_CALUDE_square_root_division_l3671_367163

theorem square_root_division (x : ℝ) : (Real.sqrt 5776 / x = 4) → x = 19 := by
  sorry

end NUMINAMATH_CALUDE_square_root_division_l3671_367163


namespace NUMINAMATH_CALUDE_girl_scout_pool_trip_expenses_l3671_367123

/-- Girl Scout Pool Trip Expenses Theorem -/
theorem girl_scout_pool_trip_expenses
  (earnings : ℝ)
  (pool_entry_cost : ℝ)
  (transportation_fee : ℝ)
  (snack_cost : ℝ)
  (num_people : ℕ)
  (h1 : earnings = 30)
  (h2 : pool_entry_cost = 2.5)
  (h3 : transportation_fee = 1.25)
  (h4 : snack_cost = 3)
  (h5 : num_people = 10) :
  earnings - (pool_entry_cost + transportation_fee + snack_cost) * num_people = -37.5 :=
sorry

end NUMINAMATH_CALUDE_girl_scout_pool_trip_expenses_l3671_367123


namespace NUMINAMATH_CALUDE_water_bottle_shape_l3671_367100

/-- Represents the volume of water in a bottle as a function of height -/
noncomputable def VolumeFunction := ℝ → ℝ

/-- A water bottle with a given height and volume function -/
structure WaterBottle where
  height : ℝ
  volume : VolumeFunction
  height_pos : height > 0

/-- The shape of a water bottle is non-linear and increases faster than linear growth -/
def IsNonLinearIncreasing (b : WaterBottle) : Prop :=
  b.volume (b.height / 2) > (1 / 2) * b.volume b.height

theorem water_bottle_shape (b : WaterBottle) 
  (h : IsNonLinearIncreasing b) : 
  ∃ (k : ℝ), k > 0 ∧ ∀ h, 0 ≤ h ∧ h ≤ b.height → b.volume h = k * h^2 :=
sorry

end NUMINAMATH_CALUDE_water_bottle_shape_l3671_367100


namespace NUMINAMATH_CALUDE_wall_bricks_l3671_367104

/-- Represents the time taken by Ben to build the wall alone -/
def ben_time : ℝ := 12

/-- Represents the time taken by Jerry to build the wall alone -/
def jerry_time : ℝ := 8

/-- Represents the decrease in combined output when working together -/
def output_decrease : ℝ := 15

/-- Represents the time taken to complete the job together -/
def combined_time : ℝ := 6

/-- Theorem stating that the number of bricks in the wall is 240 -/
theorem wall_bricks : ℝ := by
  sorry

end NUMINAMATH_CALUDE_wall_bricks_l3671_367104


namespace NUMINAMATH_CALUDE_white_surface_area_fraction_l3671_367156

theorem white_surface_area_fraction (cube_edge : ℕ) (total_cubes : ℕ) (white_cubes : ℕ) (black_cubes : ℕ) : 
  cube_edge = 4 →
  total_cubes = 64 →
  white_cubes = 48 →
  black_cubes = 16 →
  (cube_edge : ℚ) * (cube_edge : ℚ) * 6 / ((cube_edge : ℚ) * (cube_edge : ℚ) * 6) - 
  ((3 * 8 + cube_edge * cube_edge) : ℚ) / ((cube_edge : ℚ) * (cube_edge : ℚ) * 6) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_area_fraction_l3671_367156


namespace NUMINAMATH_CALUDE_sum_of_roots_l3671_367187

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3671_367187


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_is_two_min_value_achieved_l3671_367119

theorem min_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + x*y + 3*y^2 = 10) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a^2 + a*b + 3*b^2 = 10 → x^2 - x*y + y^2 ≤ a^2 - a*b + b^2 :=
by sorry

theorem min_value_is_two (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + x*y + 3*y^2 = 10) : 
  x^2 - x*y + y^2 ≥ 2 :=
by sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x^2 + x*y + 3*y^2 = 10 ∧ x^2 - x*y + y^2 < 2 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_is_two_min_value_achieved_l3671_367119


namespace NUMINAMATH_CALUDE_students_not_both_count_l3671_367182

/-- Given information about students taking chemistry and physics classes -/
structure ClassData where
  both : ℕ         -- Number of students taking both chemistry and physics
  chemistry : ℕ    -- Total number of students taking chemistry
  only_physics : ℕ -- Number of students taking only physics

/-- Calculate the number of students taking chemistry or physics but not both -/
def students_not_both (data : ClassData) : ℕ :=
  (data.chemistry - data.both) + data.only_physics

/-- Theorem stating the number of students taking chemistry or physics but not both -/
theorem students_not_both_count (data : ClassData) 
  (h1 : data.both = 12)
  (h2 : data.chemistry = 30)
  (h3 : data.only_physics = 18) :
  students_not_both data = 36 := by
  sorry

#eval students_not_both ⟨12, 30, 18⟩

end NUMINAMATH_CALUDE_students_not_both_count_l3671_367182


namespace NUMINAMATH_CALUDE_pipe_fill_time_l3671_367131

/-- The time (in hours) it takes for Pipe A to fill the tank without the leak -/
def fill_time_without_leak : ℝ := 8

/-- The time (in hours) it takes for Pipe A to fill the tank with the leak -/
def fill_time_with_leak : ℝ := 12

/-- The time (in hours) it takes for the leak to empty the full tank -/
def empty_time : ℝ := 24

theorem pipe_fill_time :
  (1 / fill_time_without_leak) - (1 / empty_time) = (1 / fill_time_with_leak) :=
sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l3671_367131


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3671_367190

theorem simplify_sqrt_expression (x : ℝ) (h : x ≠ 0) :
  Real.sqrt (4 + ((x^6 - 3*x^3 + 2) / (3*x^3))^2) = (x^6 - 3*x^3 + 2) / (3*x^3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3671_367190


namespace NUMINAMATH_CALUDE_machining_defective_rate_l3671_367166

theorem machining_defective_rate 
  (p1 p2 p3 : ℚ) 
  (h1 : p1 = 1 / 70)
  (h2 : p2 = 1 / 69)
  (h3 : p3 = 1 / 68)
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - p1) * (1 - p2) * (1 - p3) = 3 / 70 :=
sorry

end NUMINAMATH_CALUDE_machining_defective_rate_l3671_367166


namespace NUMINAMATH_CALUDE_fraction_division_l3671_367128

theorem fraction_division (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (1 / y) / (1 / x) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l3671_367128


namespace NUMINAMATH_CALUDE_tens_digit_of_13_pow_2023_l3671_367164

theorem tens_digit_of_13_pow_2023 : ∃ n : ℕ, 13^2023 ≡ 90 + n [ZMOD 100] ∧ n < 10 :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_13_pow_2023_l3671_367164


namespace NUMINAMATH_CALUDE_original_painting_height_l3671_367150

/-- Proves that given a painting with width 15 inches and a print of the painting with width 37.5 inches and height 25 inches, the height of the original painting is 10 inches. -/
theorem original_painting_height
  (original_width : ℝ)
  (print_width : ℝ)
  (print_height : ℝ)
  (h_original_width : original_width = 15)
  (h_print_width : print_width = 37.5)
  (h_print_height : print_height = 25) :
  print_height / (print_width / original_width) = 10 := by
  sorry


end NUMINAMATH_CALUDE_original_painting_height_l3671_367150


namespace NUMINAMATH_CALUDE_farm_has_six_cows_l3671_367145

/-- Represents the number of animals of each type on the farm -/
structure FarmAnimals where
  cows : ℕ
  chickens : ℕ
  sheep : ℕ

/-- Calculates the total number of legs for given farm animals -/
def totalLegs (animals : FarmAnimals) : ℕ :=
  4 * animals.cows + 2 * animals.chickens + 4 * animals.sheep

/-- Calculates the total number of heads for given farm animals -/
def totalHeads (animals : FarmAnimals) : ℕ :=
  animals.cows + animals.chickens + animals.sheep

/-- Theorem stating that the farm with the given conditions has 6 cows -/
theorem farm_has_six_cows :
  ∃ (animals : FarmAnimals),
    totalLegs animals = 100 ∧
    totalLegs animals = 3 * totalHeads animals + 20 ∧
    animals.cows = 6 := by
  sorry


end NUMINAMATH_CALUDE_farm_has_six_cows_l3671_367145


namespace NUMINAMATH_CALUDE_remainder_problem_l3671_367106

theorem remainder_problem (n : ℤ) (h : n % 20 = 11) : (2 * n) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3671_367106


namespace NUMINAMATH_CALUDE_no_digit_reversal_double_l3671_367121

theorem no_digit_reversal_double :
  (∀ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → 10 * b + a ≠ 2 * (10 * a + b)) ∧
  (∀ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
    100 * c + 10 * b + a ≠ 2 * (100 * a + 10 * b + c)) := by
  sorry

end NUMINAMATH_CALUDE_no_digit_reversal_double_l3671_367121


namespace NUMINAMATH_CALUDE_frames_per_page_l3671_367197

theorem frames_per_page (total_frames : ℕ) (num_pages : ℕ) (h1 : total_frames = 143) (h2 : num_pages = 13) :
  total_frames / num_pages = 11 := by
  sorry

end NUMINAMATH_CALUDE_frames_per_page_l3671_367197


namespace NUMINAMATH_CALUDE_jamies_mean_is_88_5_l3671_367109

/-- Represents a test score series for two students -/
structure TestScores where
  scores : List Nat
  alex_count : Nat
  jamie_count : Nat
  alex_mean : Rat

/-- Calculates Jamie's mean score given the test scores -/
def jamies_mean (ts : TestScores) : Rat :=
  let total_sum := ts.scores.sum
  let alex_sum := ts.alex_mean * ts.alex_count
  let jamie_sum := total_sum - alex_sum
  jamie_sum / ts.jamie_count

/-- Theorem: Jamie's mean score is 88.5 given the conditions -/
theorem jamies_mean_is_88_5 (ts : TestScores) 
  (h1 : ts.scores = [75, 80, 85, 90, 92, 97])
  (h2 : ts.alex_count = 4)
  (h3 : ts.jamie_count = 2)
  (h4 : ts.alex_mean = 85.5)
  : jamies_mean ts = 88.5 := by
  sorry

end NUMINAMATH_CALUDE_jamies_mean_is_88_5_l3671_367109


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l3671_367159

theorem isosceles_right_triangle_hypotenuse (a : ℝ) (h : a = 10) :
  let hypotenuse := a * Real.sqrt 2
  hypotenuse = 10 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l3671_367159


namespace NUMINAMATH_CALUDE_min_k_for_A_cannot_win_l3671_367113

/-- Represents a position on the infinite hexagonal grid --/
structure HexPosition

/-- Represents the game state --/
structure GameState where
  board : HexPosition → Option Bool  -- True for A's counter, False for empty
  turn : Bool  -- True for A's turn, False for B's turn

/-- Checks if two positions are adjacent --/
def adjacent (p1 p2 : HexPosition) : Prop := sorry

/-- Checks if there are k consecutive counters in a line --/
def consecutive_counters (state : GameState) (k : ℕ) : Prop := sorry

/-- Represents a valid move by player A --/
def valid_move_A (state : GameState) (p1 p2 : HexPosition) : Prop :=
  adjacent p1 p2 ∧ state.board p1 = none ∧ state.board p2 = none ∧ state.turn

/-- Represents a valid move by player B --/
def valid_move_B (state : GameState) (p : HexPosition) : Prop :=
  state.board p = some true ∧ ¬state.turn

/-- The main theorem stating that 6 is the minimum k for which A cannot win --/
theorem min_k_for_A_cannot_win :
  (∀ k < 6, ∃ (strategy : GameState → HexPosition × HexPosition),
    ∀ (counter_strategy : GameState → HexPosition),
      ∃ (n : ℕ), ∃ (final_state : GameState),
        consecutive_counters final_state k) ∧
  (∀ (strategy : GameState → HexPosition × HexPosition),
    ∃ (counter_strategy : GameState → HexPosition),
      ∀ (n : ℕ), ∀ (final_state : GameState),
        ¬consecutive_counters final_state 6) :=
sorry

end NUMINAMATH_CALUDE_min_k_for_A_cannot_win_l3671_367113


namespace NUMINAMATH_CALUDE_square_reassembly_l3671_367192

/-- Given two squares with side lengths a and b (where a > b), 
    they can be cut and reassembled into a single square with side length √(a² + b²) -/
theorem square_reassembly (a b : ℝ) (h : a > b) (h' : a > 0) (h'' : b > 0) :
  ∃ (new_side : ℝ), 
    new_side = Real.sqrt (a^2 + b^2) ∧ 
    new_side^2 = a^2 + b^2 :=
by sorry

end NUMINAMATH_CALUDE_square_reassembly_l3671_367192


namespace NUMINAMATH_CALUDE_min_cards_xiaohua_l3671_367158

def greeting_cards (x y z : ℕ) : Prop :=
  (Nat.lcm x (Nat.lcm y z) = 60) ∧
  (Nat.gcd x y = 4) ∧
  (Nat.gcd y z = 3) ∧
  (x ≥ 5)

theorem min_cards_xiaohua :
  ∀ x y z : ℕ, greeting_cards x y z → x ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_min_cards_xiaohua_l3671_367158


namespace NUMINAMATH_CALUDE_equivalent_form_l3671_367196

theorem equivalent_form (x : ℝ) (h : x < 0) :
  Real.sqrt (x / (1 - (x + 1) / x)) = -x * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_equivalent_form_l3671_367196


namespace NUMINAMATH_CALUDE_grocery_problem_l3671_367157

theorem grocery_problem (total_packs : ℕ) (cookie_packs : ℕ) (noodle_packs : ℕ) :
  total_packs = 28 →
  cookie_packs = 12 →
  total_packs = cookie_packs + noodle_packs →
  noodle_packs = 16 := by
sorry

end NUMINAMATH_CALUDE_grocery_problem_l3671_367157


namespace NUMINAMATH_CALUDE_bus_passengers_l3671_367188

theorem bus_passengers (initial_students : ℕ) (num_stops : ℕ) : 
  initial_students = 60 → num_stops = 4 → 
  ⌊(initial_students : ℚ) * (2/3)^num_stops⌋ = 11 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l3671_367188


namespace NUMINAMATH_CALUDE_prob_ice_skating_given_skiing_l3671_367107

/-- The probability that a randomly selected student likes ice skating -/
def P_ice_skating : ℝ := 0.6

/-- The probability that a randomly selected student likes skiing -/
def P_skiing : ℝ := 0.5

/-- The probability that a randomly selected student likes either ice skating or skiing -/
def P_ice_skating_or_skiing : ℝ := 0.7

/-- Theorem stating that the probability of a student liking ice skating given that they like skiing is 0.8 -/
theorem prob_ice_skating_given_skiing :
  (P_ice_skating + P_skiing - P_ice_skating_or_skiing) / P_skiing = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_prob_ice_skating_given_skiing_l3671_367107


namespace NUMINAMATH_CALUDE_compare_quadratic_expressions_range_of_linear_combination_l3671_367140

-- Part 1
theorem compare_quadratic_expressions (a : ℝ) : 
  (a - 2) * (a - 6) < (a - 3) * (a - 5) := by sorry

-- Part 2
theorem range_of_linear_combination (x y : ℝ) 
  (hx : -2 < x ∧ x < 1) (hy : 1 < y ∧ y < 2) : 
  -6 < 2 * x - y ∧ 2 * x - y < 1 := by sorry

end NUMINAMATH_CALUDE_compare_quadratic_expressions_range_of_linear_combination_l3671_367140


namespace NUMINAMATH_CALUDE_discount_difference_l3671_367141

theorem discount_difference : 
  let first_discount : ℝ := 0.25
  let second_discount : ℝ := 0.15
  let third_discount : ℝ := 0.10
  let claimed_discount : ℝ := 0.45
  let true_discount : ℝ := 1 - (1 - first_discount) * (1 - second_discount) * (1 - third_discount)
  claimed_discount - true_discount = 0.02375 := by
sorry

end NUMINAMATH_CALUDE_discount_difference_l3671_367141


namespace NUMINAMATH_CALUDE_equation_value_l3671_367155

theorem equation_value : 
  let Y : ℝ := (180 * 0.15 - (180 * 0.15) / 3) + 0.245 * (2 / 3 * 270) - (5.4 * 2) / (0.25^2)
  Y = -110.7 := by
  sorry

end NUMINAMATH_CALUDE_equation_value_l3671_367155


namespace NUMINAMATH_CALUDE_quadratic_extrema_l3671_367181

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 3)^2 - 1

-- Define the interval
def I : Set ℝ := Set.Icc 1 4

-- Theorem statement
theorem quadratic_extrema :
  (∃ (x : ℝ), x ∈ I ∧ ∀ (y : ℝ), y ∈ I → f y ≥ f x) ∧
  (∃ (x : ℝ), x ∈ I ∧ ∀ (y : ℝ), y ∈ I → f y ≤ f x) ∧
  (∀ (x : ℝ), x ∈ I → f x ≥ -1) ∧
  (∀ (x : ℝ), x ∈ I → f x ≤ 3) ∧
  (∃ (x : ℝ), x ∈ I ∧ f x = -1) ∧
  (∃ (x : ℝ), x ∈ I ∧ f x = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_extrema_l3671_367181


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_57_l3671_367120

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define divisibility
def divides (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

-- State the theorem
theorem no_primes_divisible_by_57 :
  ¬∃ p : ℕ, isPrime p ∧ divides 57 p :=
sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_57_l3671_367120


namespace NUMINAMATH_CALUDE_rajesh_savings_percentage_l3671_367151

theorem rajesh_savings_percentage (monthly_salary : ℝ) (food_percentage : ℝ) (medicine_percentage : ℝ) (savings : ℝ) : 
  monthly_salary = 15000 →
  food_percentage = 40 →
  medicine_percentage = 20 →
  savings = 4320 →
  let remaining := monthly_salary - (food_percentage / 100 * monthly_salary) - (medicine_percentage / 100 * monthly_salary)
  (savings / remaining) * 100 = 72 := by
sorry

end NUMINAMATH_CALUDE_rajesh_savings_percentage_l3671_367151


namespace NUMINAMATH_CALUDE_pairwise_sums_problem_l3671_367110

theorem pairwise_sums_problem (a b c d e x y : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℤ) = 
    {183, 186, 187, 190, 191, 192, 193, 194, 196, x} ∧
  x > 196 ∧
  y = 10 * x + 3 →
  a = 91 ∧ b = 92 ∧ c = 95 ∧ d = 99 ∧ e = 101 ∧ x = 200 ∧ y = 2003 :=
by sorry

end NUMINAMATH_CALUDE_pairwise_sums_problem_l3671_367110


namespace NUMINAMATH_CALUDE_solve_equation_l3671_367185

theorem solve_equation : ∃ x : ℝ, (3 * x) / 4 = 24 ∧ x = 32 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3671_367185


namespace NUMINAMATH_CALUDE_race_time_difference_l3671_367105

theorem race_time_difference (apple_rate mac_rate : ℝ) (race_distance : ℝ) : 
  apple_rate = 3 ∧ mac_rate = 4 ∧ race_distance = 24 → 
  (race_distance / apple_rate - race_distance / mac_rate) * 60 = 120 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l3671_367105


namespace NUMINAMATH_CALUDE_complex_number_equality_l3671_367108

theorem complex_number_equality : (((1 : ℂ) + I) * ((3 : ℂ) + 4*I)) / I = (7 : ℂ) + I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l3671_367108


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3671_367132

theorem arithmetic_mean_problem (x : ℝ) : 
  (8 + 16 + 21 + 7 + x) / 5 = 12 → x = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3671_367132


namespace NUMINAMATH_CALUDE_largest_digit_sum_l3671_367194

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem largest_digit_sum (a b c y : ℕ) (ha : is_digit a) (hb : is_digit b) (hc : is_digit c)
  (hy : 0 < y ∧ y ≤ 15) (h_frac : (a * 100 + b * 10 + c : ℚ) / 1000 = 1 / y) :
  a + b + c ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l3671_367194


namespace NUMINAMATH_CALUDE_quotient_in_fourth_quadrant_l3671_367122

/-- Given two complex numbers z₁ and z₂, prove that their quotient lies in the fourth quadrant. -/
theorem quotient_in_fourth_quadrant (z₁ z₂ : ℂ) 
  (hz₁ : z₁ = 2 + Complex.I) 
  (hz₂ : z₂ = 1 + Complex.I) : 
  let q := z₁ / z₂
  0 < q.re ∧ q.im < 0 :=
by sorry


end NUMINAMATH_CALUDE_quotient_in_fourth_quadrant_l3671_367122


namespace NUMINAMATH_CALUDE_inequality_proof_l3671_367127

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a*c) :
  (a*f - c*d)^2 ≥ (a*e - b*d)*(b*f - c*e) := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3671_367127


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3671_367148

theorem line_passes_through_fixed_point (m : ℝ) :
  (m + 1) * 1 + (2 * m - 1) * (-1) + m - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3671_367148


namespace NUMINAMATH_CALUDE_polynomial_independence_l3671_367130

theorem polynomial_independence (x m : ℝ) : 
  (∀ m, 6 * x^2 + (1 - 2*m) * x + 7*m = 6 * x^2 + x) → x = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_independence_l3671_367130


namespace NUMINAMATH_CALUDE_weight_of_e_l3671_367117

/-- Given three weights d, e, and f, prove that e equals 82 when their average is 42,
    the average of d and e is 35, and the average of e and f is 41. -/
theorem weight_of_e (d e f : ℝ) 
  (h1 : (d + e + f) / 3 = 42)
  (h2 : (d + e) / 2 = 35)
  (h3 : (e + f) / 2 = 41) : 
  e = 82 := by
  sorry

#check weight_of_e

end NUMINAMATH_CALUDE_weight_of_e_l3671_367117


namespace NUMINAMATH_CALUDE_binomial_15_choose_3_l3671_367179

theorem binomial_15_choose_3 : Nat.choose 15 3 = 455 := by sorry

end NUMINAMATH_CALUDE_binomial_15_choose_3_l3671_367179


namespace NUMINAMATH_CALUDE_triangle_returns_after_six_rotations_l3671_367170

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents a rotation around a point by a given angle -/
def rotate (center : ℝ × ℝ) (angle : ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Performs a single rotation of the triangle around one of its vertices -/
def rotateTriangle (t : Triangle) (vertex : Fin 3) : Triangle := sorry

/-- Performs six successive rotations of the triangle -/
def sixRotations (t : Triangle) : Triangle := sorry

/-- Theorem stating that after six rotations, the triangle returns to its original position -/
theorem triangle_returns_after_six_rotations (t : Triangle) : 
  sixRotations t = t := by sorry

end NUMINAMATH_CALUDE_triangle_returns_after_six_rotations_l3671_367170


namespace NUMINAMATH_CALUDE_arithmetic_seq_common_diff_l3671_367114

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  seq_def : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: If (S_3/3) - (S_2/2) = 1 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_seq_common_diff
  (seq : ArithmeticSequence)
  (h : seq.S 3 / 3 - seq.S 2 / 2 = 1) :
  seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_common_diff_l3671_367114


namespace NUMINAMATH_CALUDE_ratio_composition_l3671_367175

theorem ratio_composition (a b c : ℚ) 
  (h1 : a / b = 2 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 2 / 15 := by
sorry

end NUMINAMATH_CALUDE_ratio_composition_l3671_367175


namespace NUMINAMATH_CALUDE_fuchsia_to_mauve_l3671_367125

/-- Represents the composition of paint mixtures -/
structure PaintMix where
  red : ℚ
  blue : ℚ

/-- The ratio of red to blue in fuchsia paint -/
def fuchsia : PaintMix := { red := 6, blue := 3 }

/-- The ratio of red to blue in mauve paint -/
def mauve : PaintMix := { red := 4, blue := 5 }

/-- The amount of blue paint needed to change fuchsia to mauve -/
def blue_paint_needed (F : ℚ) : ℚ := F / 2

theorem fuchsia_to_mauve (F : ℚ) (F_pos : F > 0) :
  let original_red := F * fuchsia.red / (fuchsia.red + fuchsia.blue)
  let original_blue := F * fuchsia.blue / (fuchsia.red + fuchsia.blue)
  let added_blue := blue_paint_needed F
  original_red / (original_blue + added_blue) = mauve.red / mauve.blue :=
by sorry

end NUMINAMATH_CALUDE_fuchsia_to_mauve_l3671_367125


namespace NUMINAMATH_CALUDE_dog_area_theorem_l3671_367162

/-- Represents a rectangular obstruction -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the position where the dog is tied -/
structure TiePoint where
  distance_from_midpoint : ℝ

/-- Calculates the area accessible by a dog tied to a point near a rectangular obstruction -/
def accessible_area (rect : Rectangle) (tie : TiePoint) (rope_length : ℝ) : ℝ :=
  sorry

/-- Theorem stating the accessible area for the given problem -/
theorem dog_area_theorem (rect : Rectangle) (tie : TiePoint) (rope_length : ℝ) :
  rect.length = 20 ∧ rect.width = 10 ∧ tie.distance_from_midpoint = 5 ∧ rope_length = 10 →
  accessible_area rect tie rope_length = 62.5 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_dog_area_theorem_l3671_367162


namespace NUMINAMATH_CALUDE_cannot_form_triangle_l3671_367165

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that line segments of lengths 2, 4, and 6 cannot form a triangle -/
theorem cannot_form_triangle : ¬(can_form_triangle 2 4 6) := by
  sorry

end NUMINAMATH_CALUDE_cannot_form_triangle_l3671_367165


namespace NUMINAMATH_CALUDE_sum_even_integers_minus15_to_5_l3671_367111

def sum_even_integers (a b : Int) : Int :=
  let first_even := if a % 2 = 0 then a else a + 1
  let last_even := if b % 2 = 0 then b else b - 1
  let num_terms := (last_even - first_even) / 2 + 1
  (first_even + last_even) * num_terms / 2

theorem sum_even_integers_minus15_to_5 :
  sum_even_integers (-15) 5 = -50 := by
sorry

end NUMINAMATH_CALUDE_sum_even_integers_minus15_to_5_l3671_367111


namespace NUMINAMATH_CALUDE_set_intersection_equality_l3671_367134

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem set_intersection_equality : M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l3671_367134


namespace NUMINAMATH_CALUDE_factorization_equality_l3671_367116

theorem factorization_equality (x y : ℝ) : 6*x^2*y - 3*x*y = 3*x*y*(2*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3671_367116


namespace NUMINAMATH_CALUDE_percentage_excess_l3671_367135

theorem percentage_excess (x y : ℝ) (h : x = 0.8 * y) : y = 1.25 * x := by
  sorry

end NUMINAMATH_CALUDE_percentage_excess_l3671_367135


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3671_367161

theorem min_value_trig_expression : 
  ∀ x : ℝ, (Real.sin x)^8 + (Real.cos x)^8 + 1 ≥ (9/10) * ((Real.sin x)^6 + (Real.cos x)^6 + 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3671_367161


namespace NUMINAMATH_CALUDE_a_squared_ge_three_l3671_367142

theorem a_squared_ge_three (a b c : ℝ) (h1 : a ≠ 0) (h2 : a + b + c = a * b * c) (h3 : a^2 = b * c) : a^2 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_a_squared_ge_three_l3671_367142


namespace NUMINAMATH_CALUDE_opposite_of_negative_sqrt_two_l3671_367124

theorem opposite_of_negative_sqrt_two : -((-Real.sqrt 2)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_sqrt_two_l3671_367124


namespace NUMINAMATH_CALUDE_total_snakes_l3671_367102

theorem total_snakes (boa_constrictors python rattlesnakes : ℕ) : 
  boa_constrictors = 40 →
  python = 3 * boa_constrictors →
  rattlesnakes = 40 →
  boa_constrictors + python + rattlesnakes = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_snakes_l3671_367102


namespace NUMINAMATH_CALUDE_calculation_proof_l3671_367193

theorem calculation_proof : (1/2)⁻¹ + (Real.sqrt 2)^2 - 4 * |(-(1/2))| = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3671_367193
