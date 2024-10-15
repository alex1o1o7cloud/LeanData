import Mathlib

namespace NUMINAMATH_CALUDE_rice_yield_and_conversion_l289_28912

-- Define the yield per acre of ordinary rice
def ordinary_yield : ℝ := 600

-- Define the yield per acre of hybrid rice
def hybrid_yield : ℝ := 2 * ordinary_yield

-- Define the acreage difference between fields
def acreage_difference : ℝ := 4

-- Define the total yield of field A
def field_A_yield : ℝ := 9600

-- Define the total yield of field B
def field_B_yield : ℝ := 7200

-- Define the minimum total yield after conversion
def min_total_yield : ℝ := 17700

-- Theorem statement
theorem rice_yield_and_conversion :
  -- Prove that the ordinary yield is 600 kg/acre
  ordinary_yield = 600 ∧
  -- Prove that the hybrid yield is 1200 kg/acre
  hybrid_yield = 1200 ∧
  -- Prove that at least 1.5 acres of field B should be converted
  ∃ (converted_acres : ℝ),
    converted_acres ≥ 1.5 ∧
    field_A_yield + 
    ordinary_yield * (field_B_yield / ordinary_yield - converted_acres) + 
    hybrid_yield * converted_acres ≥ min_total_yield :=
by sorry

end NUMINAMATH_CALUDE_rice_yield_and_conversion_l289_28912


namespace NUMINAMATH_CALUDE_f_min_and_g_zeros_l289_28908

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 3*x + 2

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f x + (3 - m) * x

theorem f_min_and_g_zeros (h : ∀ x, x > 0 → f x ≥ 0) :
  (∃ x > 0, f x = 0) ∧
  (∀ m < 3, ¬∃ x₁ x₂, x₁ ≠ x₂ ∧ g m x₁ = 0 ∧ g m x₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_min_and_g_zeros_l289_28908


namespace NUMINAMATH_CALUDE_quadratic_general_form_l289_28925

theorem quadratic_general_form :
  ∀ x : ℝ, (6 * x^2 = 5 * x - 4) ↔ (6 * x^2 - 5 * x + 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_general_form_l289_28925


namespace NUMINAMATH_CALUDE_point_not_on_transformed_plane_l289_28921

/-- The original plane equation -/
def plane_equation (x y z : ℝ) : Prop := x - 3*y + 5*z - 1 = 0

/-- The similarity transformation coefficient -/
def k : ℝ := -1

/-- The point A -/
def point_A : ℝ × ℝ × ℝ := (2, 0, -1)

/-- The transformed plane equation -/
def transformed_plane_equation (x y z : ℝ) : Prop := x - 3*y + 5*z + 1 = 0

/-- Theorem stating that point A does not belong to the transformed plane -/
theorem point_not_on_transformed_plane :
  ¬(transformed_plane_equation point_A.1 point_A.2.1 point_A.2.2) :=
sorry

end NUMINAMATH_CALUDE_point_not_on_transformed_plane_l289_28921


namespace NUMINAMATH_CALUDE_product_positive_l289_28986

theorem product_positive (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x^4 - y^4 > x) (h2 : y^4 - x^4 > y) : x * y > 0 :=
by sorry

end NUMINAMATH_CALUDE_product_positive_l289_28986


namespace NUMINAMATH_CALUDE_opposite_vector_with_magnitude_l289_28947

/-- Given two vectors a and b in ℝ², where a is (-1, 2) and b is in the opposite direction
    to a with magnitude √5, prove that b = (1, -2) -/
theorem opposite_vector_with_magnitude (a b : ℝ × ℝ) : 
  a = (-1, 2) →
  ∃ k : ℝ, k < 0 ∧ b = k • a →
  ‖b‖ = Real.sqrt 5 →
  b = (1, -2) :=
by sorry

end NUMINAMATH_CALUDE_opposite_vector_with_magnitude_l289_28947


namespace NUMINAMATH_CALUDE_base5_subtraction_l289_28999

/-- Converts a base 5 number represented as a list of digits to its decimal equivalent -/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 5 * acc + d) 0

/-- Converts a decimal number to its base 5 representation as a list of digits -/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The first number in base 5 -/
def num1 : List Nat := [1, 2, 3, 4]

/-- The second number in base 5 -/
def num2 : List Nat := [2, 3, 4]

/-- The expected difference in base 5 -/
def expected_diff : List Nat := [1, 0, 0, 0]

theorem base5_subtraction :
  decimalToBase5 (base5ToDecimal num1 - base5ToDecimal num2) = expected_diff := by
  sorry

end NUMINAMATH_CALUDE_base5_subtraction_l289_28999


namespace NUMINAMATH_CALUDE_subtraction_from_percentage_l289_28903

theorem subtraction_from_percentage (n : ℝ) : n = 85 → 0.4 * n - 11 = 23 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_from_percentage_l289_28903


namespace NUMINAMATH_CALUDE_sin_cos_difference_21_81_l289_28961

theorem sin_cos_difference_21_81 :
  Real.sin (21 * π / 180) * Real.cos (81 * π / 180) -
  Real.cos (21 * π / 180) * Real.sin (81 * π / 180) =
  -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_21_81_l289_28961


namespace NUMINAMATH_CALUDE_remainder_101_pow_37_mod_100_l289_28933

theorem remainder_101_pow_37_mod_100 : 101^37 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_pow_37_mod_100_l289_28933


namespace NUMINAMATH_CALUDE_zain_coin_count_l289_28927

/-- Represents the number of coins Emerie has of each type -/
structure EmerieCoins where
  quarters : Nat
  dimes : Nat
  nickels : Nat

/-- Calculates the total number of coins Zain has given Emerie's coin counts -/
def zainTotalCoins (e : EmerieCoins) : Nat :=
  (e.quarters + 10) + (e.dimes + 10) + (e.nickels + 10)

theorem zain_coin_count (e : EmerieCoins) 
  (hq : e.quarters = 6) 
  (hd : e.dimes = 7) 
  (hn : e.nickels = 5) : 
  zainTotalCoins e = 48 := by
  sorry

end NUMINAMATH_CALUDE_zain_coin_count_l289_28927


namespace NUMINAMATH_CALUDE_max_m_and_a_value_l289_28917

/-- The function f(x) = |x+3| -/
def f (x : ℝ) : ℝ := |x + 3|

/-- The function g(x) = m - 2|x-11| -/
def g (m : ℝ) (x : ℝ) : ℝ := m - 2*|x - 11|

/-- The theorem stating the maximum value of m and the value of a -/
theorem max_m_and_a_value :
  (∀ x : ℝ, 2 * f x ≥ g m (x + 4)) →
  (∃ t : ℝ, t = 20 ∧ 
    (∀ m' : ℝ, (∀ x : ℝ, 2 * f x ≥ g m' (x + 4)) → m' ≤ t) ∧
    (∀ a : ℝ, a > 0 →
      (∃ x y z : ℝ, 2*x^2 + 3*y^2 + 6*z^2 = a ∧ 
        x + y + z = t/20 ∧
        (∀ x' y' z' : ℝ, 2*x'^2 + 3*y'^2 + 6*z'^2 = a → x' + y' + z' ≤ t/20)) →
      a = 1)) :=
sorry

end NUMINAMATH_CALUDE_max_m_and_a_value_l289_28917


namespace NUMINAMATH_CALUDE_defective_units_shipped_percentage_l289_28970

theorem defective_units_shipped_percentage
  (total_units : ℝ)
  (defective_rate : ℝ)
  (defective_shipped_rate : ℝ)
  (h1 : defective_rate = 0.1)
  (h2 : defective_shipped_rate = 0.005)
  (h3 : total_units > 0) :
  (defective_shipped_rate / defective_rate) * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_defective_units_shipped_percentage_l289_28970


namespace NUMINAMATH_CALUDE_celias_budget_weeks_l289_28995

/-- Celia's budget problem -/
theorem celias_budget_weeks (weekly_food_budget : ℝ) (rent : ℝ) (streaming : ℝ) (cell_phone : ℝ) 
  (savings_percent : ℝ) (savings_amount : ℝ) :
  weekly_food_budget = 100 →
  rent = 1500 →
  streaming = 30 →
  cell_phone = 50 →
  savings_percent = 0.1 →
  savings_amount = 198 →
  ∃ (weeks : ℕ), 
    savings_amount = savings_percent * (weekly_food_budget * ↑weeks + rent + streaming + cell_phone) ∧
    weeks = 4 := by
  sorry

end NUMINAMATH_CALUDE_celias_budget_weeks_l289_28995


namespace NUMINAMATH_CALUDE_min_a_for_p_true_l289_28987

-- Define the set of x
def X : Set ℝ := { x | 1 ≤ x ∧ x ≤ 9 }

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x ∈ X, x^2 - a*x + 36 ≤ 0

-- Theorem statement
theorem min_a_for_p_true : 
  (∃ a : ℝ, p a) → (∀ a : ℝ, p a → a ≥ 12) ∧ p 12 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_p_true_l289_28987


namespace NUMINAMATH_CALUDE_worm_length_difference_l289_28974

theorem worm_length_difference (long_worm short_worm : Real) 
  (h1 : long_worm = 0.8)
  (h2 : short_worm = 0.1) :
  long_worm - short_worm = 0.7 := by
sorry

end NUMINAMATH_CALUDE_worm_length_difference_l289_28974


namespace NUMINAMATH_CALUDE_square_root_pattern_l289_28996

theorem square_root_pattern (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h2 : Real.sqrt (2 + 2/3) = 2 * Real.sqrt (2/3))
  (h3 : Real.sqrt (3 + 3/8) = 3 * Real.sqrt (3/8))
  (h4 : Real.sqrt (4 + 4/15) = 4 * Real.sqrt (4/15))
  (h7 : Real.sqrt (7 + a/b) = 7 * Real.sqrt (a/b)) :
  a + b = 55 := by sorry

end NUMINAMATH_CALUDE_square_root_pattern_l289_28996


namespace NUMINAMATH_CALUDE_game_winning_probability_l289_28972

/-- A game with consecutive integers from 2 to 2020 -/
def game_range : Set ℕ := {n | 2 ≤ n ∧ n ≤ 2020}

/-- The total number of integers in the game -/
def total_numbers : ℕ := 2019

/-- Two numbers are coprime if their greatest common divisor is 1 -/
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- The number of coprime pairs in the game range -/
def coprime_pairs : ℕ := 1010

/-- The probability of winning is the number of coprime pairs divided by the total numbers -/
theorem game_winning_probability :
  (coprime_pairs : ℚ) / total_numbers = 1010 / 2019 := by sorry

end NUMINAMATH_CALUDE_game_winning_probability_l289_28972


namespace NUMINAMATH_CALUDE_product_value_l289_28982

theorem product_value (x y : ℤ) (h1 : x = 12) (h2 : y = 7) :
  (x - y) * (2 * x + 2 * y) = 190 := by
  sorry

end NUMINAMATH_CALUDE_product_value_l289_28982


namespace NUMINAMATH_CALUDE_day_150_of_year_N_minus_1_is_friday_l289_28930

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific day in a year -/
structure YearDay where
  year : Int
  day : Nat

/-- Function to determine the day of the week for a given YearDay -/
def dayOfWeek (yd : YearDay) : DayOfWeek := sorry

/-- Theorem stating the problem conditions and the result to be proved -/
theorem day_150_of_year_N_minus_1_is_friday 
  (N : Int) 
  (h1 : dayOfWeek ⟨N, 250⟩ = DayOfWeek.Monday) 
  (h2 : dayOfWeek ⟨N + 2, 300⟩ = DayOfWeek.Monday) :
  dayOfWeek ⟨N - 1, 150⟩ = DayOfWeek.Friday := by sorry

end NUMINAMATH_CALUDE_day_150_of_year_N_minus_1_is_friday_l289_28930


namespace NUMINAMATH_CALUDE_parallel_angle_theorem_l289_28984

theorem parallel_angle_theorem (α β : Real) :
  (α = 60 ∨ β = 60) →  -- One angle is 60°
  (α = β ∨ α + β = 180) →  -- Angles are either equal or supplementary (parallel sides condition)
  (α = 60 ∧ β = 60) ∨ (α = 60 ∧ β = 120) ∨ (α = 120 ∧ β = 60) :=
by sorry

end NUMINAMATH_CALUDE_parallel_angle_theorem_l289_28984


namespace NUMINAMATH_CALUDE_quadruplet_solution_l289_28929

theorem quadruplet_solution (x y z w : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0)
  (h_eq : (x*y*z + 1)/(x + 1) = (y*z*w + 1)/(y + 1) ∧
          (y*z*w + 1)/(y + 1) = (z*w*x + 1)/(z + 1) ∧
          (z*w*x + 1)/(z + 1) = (w*x*y + 1)/(w + 1))
  (h_sum : x + y + z + w = 48) :
  x = 12 ∧ y = 12 ∧ z = 12 ∧ w = 12 := by
sorry

end NUMINAMATH_CALUDE_quadruplet_solution_l289_28929


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_solution_l289_28935

/-- A complex number is purely imaginary if its real part is zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

/-- Given that z and (z+2)^2 - 8i are both purely imaginary, prove that z = -2i -/
theorem complex_purely_imaginary_solution (z : ℂ) 
  (h1 : isPurelyImaginary z) 
  (h2 : isPurelyImaginary ((z + 2)^2 - 8*I)) : 
  z = -2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_solution_l289_28935


namespace NUMINAMATH_CALUDE_olyas_numbers_proof_l289_28976

def first_number : ℕ := 929
def second_number : ℕ := 20
def third_number : ℕ := 2

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem olyas_numbers_proof :
  (100 ≤ first_number ∧ first_number < 1000) ∧
  (second_number = sum_of_digits first_number) ∧
  (third_number = sum_of_digits second_number) ∧
  (∃ (a b : ℕ), first_number = 100 * a + 10 * b + a ∧
                second_number = 10 * b + 0 ∧
                third_number = b) :=
by sorry

end NUMINAMATH_CALUDE_olyas_numbers_proof_l289_28976


namespace NUMINAMATH_CALUDE_negative_cube_squared_l289_28978

theorem negative_cube_squared (x : ℝ) : (-x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l289_28978


namespace NUMINAMATH_CALUDE_collinear_vectors_y_value_l289_28957

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (-4, y)
  collinear a b → y = -6 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_y_value_l289_28957


namespace NUMINAMATH_CALUDE_prime_count_inequality_l289_28937

/-- p_n denotes the nth prime number -/
def p (n : ℕ) : ℕ := sorry

/-- π(x) denotes the number of primes less than or equal to x -/
def π (x : ℝ) : ℕ := sorry

/-- The product of the first n primes -/
def primeProduct (n : ℕ) : ℕ := sorry

theorem prime_count_inequality (n : ℕ) (h : n ≥ 6) :
  π (Real.sqrt (primeProduct n : ℝ)) > 2 * n := by
  sorry

end NUMINAMATH_CALUDE_prime_count_inequality_l289_28937


namespace NUMINAMATH_CALUDE_committee_selection_l289_28990

theorem committee_selection (n m : ℕ) (h1 : n = 15) (h2 : m = 5) : 
  Nat.choose n m = 3003 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l289_28990


namespace NUMINAMATH_CALUDE_ring_toss_total_earnings_l289_28959

/-- The ring toss game at a carnival earns a certain amount per day for a given number of days. -/
def carnival_earnings (daily_earnings : ℕ) (num_days : ℕ) : ℕ :=
  daily_earnings * num_days

/-- Theorem: The ring toss game earns 3168 dollars in total when it makes 144 dollars per day for 22 days. -/
theorem ring_toss_total_earnings :
  carnival_earnings 144 22 = 3168 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_total_earnings_l289_28959


namespace NUMINAMATH_CALUDE_range_of_a_l289_28979

theorem range_of_a (a : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |a - 1| ≥ x + 2*y + 2*z) →
  (a ≤ -2 ∨ a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l289_28979


namespace NUMINAMATH_CALUDE_card_problem_solution_l289_28911

/-- Represents the types of cards -/
inductive CardType
  | WW  -- White-White
  | BB  -- Black-Black
  | BW  -- Black-White

/-- Represents the state of a set of cards -/
structure CardSet where
  total : Nat
  blackUp : Nat

/-- Represents the problem setup -/
structure CardProblem where
  initialState : CardSet
  afterFirst : CardSet
  afterSecond : CardSet
  afterThird : CardSet

/-- The main theorem to prove -/
theorem card_problem_solution (p : CardProblem) : 
  p.initialState.total = 12 ∧ 
  p.initialState.blackUp = 9 ∧
  p.afterFirst.blackUp = 4 ∧
  p.afterSecond.blackUp = 6 ∧
  p.afterThird.blackUp = 5 →
  ∃ (bw ww : Nat), bw = 9 ∧ ww = 3 ∧ bw + ww = p.initialState.total := by
  sorry


end NUMINAMATH_CALUDE_card_problem_solution_l289_28911


namespace NUMINAMATH_CALUDE_spoon_fork_sale_price_comparison_l289_28943

theorem spoon_fork_sale_price_comparison :
  ∃ (initial_price : ℕ),
    initial_price % 10 = 0 ∧
    initial_price > 100 ∧
    initial_price - 100 < initial_price / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_spoon_fork_sale_price_comparison_l289_28943


namespace NUMINAMATH_CALUDE_sin_6theta_l289_28909

theorem sin_6theta (θ : ℝ) : 
  Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 5 →
  Real.sin (6 * θ) = -630 * Real.sqrt 8 / 15625 := by
sorry

end NUMINAMATH_CALUDE_sin_6theta_l289_28909


namespace NUMINAMATH_CALUDE_smallest_part_of_three_way_division_l289_28967

theorem smallest_part_of_three_way_division (total : ℕ) (a b c : ℕ) : 
  total = 2340 →
  a + b + c = total →
  ∃ (x : ℕ), a = 5 * x ∧ b = 7 * x ∧ c = 11 * x →
  min a (min b c) = 510 :=
by sorry

end NUMINAMATH_CALUDE_smallest_part_of_three_way_division_l289_28967


namespace NUMINAMATH_CALUDE_seventh_term_is_13_l289_28906

def fibonacci_like : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci_like n + fibonacci_like (n + 1)

theorem seventh_term_is_13 : fibonacci_like 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_13_l289_28906


namespace NUMINAMATH_CALUDE_problem_solution_l289_28985

theorem problem_solution (y : ℝ) (h : 9 / y^2 = y / 81) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l289_28985


namespace NUMINAMATH_CALUDE_angle_rotation_l289_28951

theorem angle_rotation (initial_angle rotation : ℝ) (h1 : initial_angle = 60) (h2 : rotation = 420) :
  (initial_angle - (rotation % 360)) % 360 = 0 :=
by sorry

end NUMINAMATH_CALUDE_angle_rotation_l289_28951


namespace NUMINAMATH_CALUDE_total_weight_is_seven_pounds_l289_28989

-- Define the weights of items
def brie_cheese : ℚ := 8 / 16  -- in pounds
def bread : ℚ := 1
def tomatoes : ℚ := 1
def zucchini : ℚ := 2
def chicken_breasts : ℚ := 3 / 2
def raspberries : ℚ := 8 / 16  -- in pounds
def blueberries : ℚ := 8 / 16  -- in pounds

-- Define the conversion factor
def ounces_per_pound : ℚ := 16

-- Theorem statement
theorem total_weight_is_seven_pounds :
  brie_cheese + bread + tomatoes + zucchini + chicken_breasts + raspberries + blueberries = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_seven_pounds_l289_28989


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l289_28993

/-- Proves that a cement mixture with the given composition weighs 40 pounds -/
theorem cement_mixture_weight :
  ∀ (W : ℝ),
  (1/4 : ℝ) * W + (2/5 : ℝ) * W + 14 = W →
  W = 40 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l289_28993


namespace NUMINAMATH_CALUDE_d_equals_25_l289_28980

theorem d_equals_25 (x : ℝ) (h : x^2 - 2*x - 5 = 0) : 
  x^4 - 2*x^3 + x^2 - 12*x - 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_d_equals_25_l289_28980


namespace NUMINAMATH_CALUDE_greatest_y_value_l289_28942

theorem greatest_y_value (x y : ℤ) (h : x * y + 7 * x + 2 * y = -8) :
  y ≤ -1 ∧ ∃ (x₀ y₀ : ℤ), x₀ * y₀ + 7 * x₀ + 2 * y₀ = -8 ∧ y₀ = -1 := by
  sorry

end NUMINAMATH_CALUDE_greatest_y_value_l289_28942


namespace NUMINAMATH_CALUDE_power_subtraction_l289_28992

theorem power_subtraction (x a b : ℝ) (ha : x^a = 3) (hb : x^b = 5) : x^(a - b) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_power_subtraction_l289_28992


namespace NUMINAMATH_CALUDE_matthew_stocks_solution_l289_28926

def matthew_stocks (expensive_stock_price : ℕ) (cheap_stock_price : ℕ) (cheap_stock_shares : ℕ) (total_assets : ℕ) (expensive_stock_shares : ℕ) : Prop :=
  expensive_stock_price = 2 * cheap_stock_price ∧
  cheap_stock_shares = 26 ∧
  expensive_stock_price = 78 ∧
  total_assets = 2106 ∧
  expensive_stock_shares * expensive_stock_price + cheap_stock_shares * cheap_stock_price = total_assets

theorem matthew_stocks_solution :
  ∃ (expensive_stock_price cheap_stock_price cheap_stock_shares total_assets expensive_stock_shares : ℕ),
    matthew_stocks expensive_stock_price cheap_stock_price cheap_stock_shares total_assets expensive_stock_shares ∧
    expensive_stock_shares = 14 := by
  sorry

end NUMINAMATH_CALUDE_matthew_stocks_solution_l289_28926


namespace NUMINAMATH_CALUDE_nonzero_terms_count_l289_28923

def polynomial (x : ℝ) : ℝ :=
  (x - 3) * (3 * x^3 + 2 * x^2 - 4 * x + 1) + 4 * (x^4 + x^3 - 2 * x^2 + x) - 5 * (x^3 - 3 * x + 1)

theorem nonzero_terms_count :
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
    ∀ x, polynomial x = a * x^4 + b * x^3 + c * x^2 + d * x + e :=
by sorry

end NUMINAMATH_CALUDE_nonzero_terms_count_l289_28923


namespace NUMINAMATH_CALUDE_solve_for_q_l289_28945

theorem solve_for_q (p q : ℝ) 
  (h1 : p > 1)
  (h2 : q > 1)
  (h3 : 1/p + 1/q = 1)
  (h4 : p * q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l289_28945


namespace NUMINAMATH_CALUDE_polynomial_transformation_l289_28910

-- Define the original polynomial
def original_poly (b : ℝ) (x : ℝ) : ℝ := x^4 - b*x - 3

-- Define the transformed polynomial
def transformed_poly (b : ℝ) (x : ℝ) : ℝ := 3*x^4 - b*x^3 - 1

theorem polynomial_transformation (b : ℝ) (a c d : ℝ) :
  (original_poly b a = 0 ∧ original_poly b b = 0 ∧ original_poly b c = 0 ∧ original_poly b d = 0) →
  (transformed_poly b ((a + b + c) / d^2) = 0 ∧
   transformed_poly b ((a + b + d) / c^2) = 0 ∧
   transformed_poly b ((a + c + d) / b^2) = 0 ∧
   transformed_poly b ((b + c + d) / a^2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_transformation_l289_28910


namespace NUMINAMATH_CALUDE_car_repair_cost_john_car_repair_cost_l289_28977

/-- Calculates the amount spent on car repairs given savings information -/
theorem car_repair_cost (monthly_savings : ℕ) (savings_months : ℕ) (remaining_amount : ℕ) : ℕ :=
  let total_savings := monthly_savings * savings_months
  total_savings - remaining_amount

/-- Proves that John spent $400 on car repairs -/
theorem john_car_repair_cost : 
  car_repair_cost 25 24 200 = 400 := by
  sorry

end NUMINAMATH_CALUDE_car_repair_cost_john_car_repair_cost_l289_28977


namespace NUMINAMATH_CALUDE_vector_operations_and_parallelism_l289_28931

/-- Given two 2D vectors a and b, prove properties about their linear combinations and parallelism. -/
theorem vector_operations_and_parallelism 
  (a b : ℝ × ℝ) 
  (ha : a = (2, 0)) 
  (hb : b = (1, 4)) : 
  (2 • a + 3 • b = (7, 12)) ∧ 
  (a - 2 • b = (0, -8)) ∧ 
  (∃ k : ℝ, k • a + b = (2*k + 1, 4) ∧ a + 2 • b = (4, 8) ∧ k = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_vector_operations_and_parallelism_l289_28931


namespace NUMINAMATH_CALUDE_percentage_decrease_of_b_l289_28962

theorem percentage_decrease_of_b (a b x m : ℝ) (p : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- a and b are positive
  a / b = 4 / 5 ∧  -- ratio of a to b is 4 to 5
  x = 1.25 * a ∧  -- x equals a increased by 25 percent of a
  m = b * (1 - p / 100) ∧  -- m equals b decreased by p percent of b
  m / x = 0.4  -- m / x is 0.4
  → p = 60 :=  -- The percentage decrease of b to get m is 60%
by sorry

end NUMINAMATH_CALUDE_percentage_decrease_of_b_l289_28962


namespace NUMINAMATH_CALUDE_mirror_country_transfers_l289_28949

-- Define the type for cities
def City : Type := ℕ

-- Define the type for countries
inductive Country
| Wonderland
| Mirrorland

-- Define a function to represent railroad connections
def connected (country : Country) (city1 city2 : City) : Prop := sorry

-- Define a function to represent the "double" of a city in the other country
def double (city : City) (country : Country) : City := sorry

-- Define the number of transfers needed for a journey
def transfers (country : Country) (start finish : City) : ℕ := sorry

-- State the theorem
theorem mirror_country_transfers 
  (A B : City) 
  (h1 : transfers Country.Wonderland A B ≥ 2) 
  (h2 : ∀ (c1 c2 : City), connected Country.Wonderland c1 c2 ↔ ¬connected Country.Mirrorland (double c1 Country.Mirrorland) (double c2 Country.Mirrorland))
  (h3 : ∀ (c : City), ∃ (d : City), d = double c Country.Mirrorland)
  : ∀ (X Y : City), transfers Country.Mirrorland X Y ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_mirror_country_transfers_l289_28949


namespace NUMINAMATH_CALUDE_intersection_with_complement_l289_28915

open Set

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def P : Set Nat := {1, 2, 3, 4}
def Q : Set Nat := {3, 4, 5}

theorem intersection_with_complement :
  P ∩ (U \ Q) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l289_28915


namespace NUMINAMATH_CALUDE_thirty_percent_of_hundred_l289_28975

theorem thirty_percent_of_hundred : (30 : ℝ) = (30 / 100) * 100 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_of_hundred_l289_28975


namespace NUMINAMATH_CALUDE_black_white_ratio_after_border_l289_28918

/-- Represents a rectangular tile pattern -/
structure TilePattern where
  length : ℕ
  width : ℕ
  blackTiles : ℕ
  whiteTiles : ℕ

/-- Adds a border of black tiles to a tile pattern -/
def addBorder (pattern : TilePattern) (borderWidth : ℕ) : TilePattern :=
  { length := pattern.length + 2 * borderWidth,
    width := pattern.width + 2 * borderWidth,
    blackTiles := pattern.blackTiles + 
      (pattern.length + pattern.width + 2 * borderWidth) * 2 * borderWidth + 4 * borderWidth^2,
    whiteTiles := pattern.whiteTiles }

theorem black_white_ratio_after_border (initialPattern : TilePattern) :
  initialPattern.length = 4 →
  initialPattern.width = 8 →
  initialPattern.blackTiles = 10 →
  initialPattern.whiteTiles = 22 →
  let finalPattern := addBorder initialPattern 2
  (finalPattern.blackTiles : ℚ) / finalPattern.whiteTiles = 19 / 11 := by
  sorry

end NUMINAMATH_CALUDE_black_white_ratio_after_border_l289_28918


namespace NUMINAMATH_CALUDE_angle_properties_l289_28900

theorem angle_properties (a : ℝ) (α : ℝ) (h1 : a < 0) 
  (h2 : Real.tan α = 2) 
  (h3 : (a, 2*a) ∈ Set.range (λ t : ℝ × ℝ => (t.1 * Real.cos α, t.1 * Real.sin α))) :
  Real.cos α = -Real.sqrt 5 / 5 ∧ 
  Real.tan α = 2 ∧ 
  (Real.cos α)^2 / Real.tan α = 1/10 := by
sorry

end NUMINAMATH_CALUDE_angle_properties_l289_28900


namespace NUMINAMATH_CALUDE_marks_garden_flowers_l289_28946

theorem marks_garden_flowers (yellow : ℕ) (purple : ℕ) (green : ℕ) 
  (h1 : yellow = 10)
  (h2 : purple = yellow + yellow * 4 / 5)
  (h3 : green = (yellow + purple) / 4) :
  yellow + purple + green = 35 := by
  sorry

end NUMINAMATH_CALUDE_marks_garden_flowers_l289_28946


namespace NUMINAMATH_CALUDE_triangle_square_side_ratio_l289_28988

theorem triangle_square_side_ratio (perimeter : ℝ) (triangle_side square_side : ℝ) : 
  perimeter > 0 → 
  triangle_side * 3 = perimeter → 
  square_side * 4 = perimeter → 
  triangle_side / square_side = 4 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_square_side_ratio_l289_28988


namespace NUMINAMATH_CALUDE_geometry_propositions_l289_28944

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the main theorem
theorem geometry_propositions
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : subset m β) :
  -- Exactly two of the following propositions are correct
  ∃! (correct : Fin 4 → Prop),
    (∀ i, correct i ↔ i.val < 2) ∧
    correct 0 = (parallel α β → line_perpendicular l m) ∧
    correct 1 = (line_perpendicular l m → parallel α β) ∧
    correct 2 = (plane_perpendicular α β → line_parallel l m) ∧
    correct 3 = (line_parallel l m → plane_perpendicular α β) :=
sorry

end NUMINAMATH_CALUDE_geometry_propositions_l289_28944


namespace NUMINAMATH_CALUDE_piano_lesson_rate_piano_rate_is_28_l289_28966

/-- Calculates the hourly rate for piano lessons given the conditions -/
theorem piano_lesson_rate (clarinet_rate : ℝ) (clarinet_hours : ℝ) (piano_hours : ℝ) 
  (extra_piano_cost : ℝ) (weeks_per_year : ℕ) : ℝ :=
  let annual_clarinet_cost := clarinet_rate * clarinet_hours * weeks_per_year
  let annual_piano_cost := annual_clarinet_cost + extra_piano_cost
  annual_piano_cost / (piano_hours * weeks_per_year)

/-- The hourly rate for piano lessons is $28 -/
theorem piano_rate_is_28 : 
  piano_lesson_rate 40 3 5 1040 52 = 28 := by
sorry

end NUMINAMATH_CALUDE_piano_lesson_rate_piano_rate_is_28_l289_28966


namespace NUMINAMATH_CALUDE_minimum_rice_amount_l289_28958

theorem minimum_rice_amount (o r : ℝ) (ho : o ≥ 8 + r / 3) (ho2 : o ≤ 2 * r) :
  ∃ (min_r : ℕ), min_r = 5 ∧ ∀ (r' : ℕ), r' ≥ min_r → ∃ (o' : ℝ), o' ≥ 8 + r' / 3 ∧ o' ≤ 2 * r' :=
sorry

end NUMINAMATH_CALUDE_minimum_rice_amount_l289_28958


namespace NUMINAMATH_CALUDE_tangent_condition_l289_28936

theorem tangent_condition (a b : ℝ) : 
  (∃ (x y : ℝ), x + y = 0 ∧ (x - a)^2 + (y - b)^2 = 2) → 
  (a + b = 2 → ∃ (x y : ℝ), x + y = 0 ∧ (x - a)^2 + (y - b)^2 = 2) ∧
  (∃ (a b : ℝ), (∃ (x y : ℝ), x + y = 0 ∧ (x - a)^2 + (y - b)^2 = 2) ∧ a + b ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_condition_l289_28936


namespace NUMINAMATH_CALUDE_highest_score_is_179_l289_28964

/-- Represents a batsman's statistics --/
structure BatsmanStats where
  totalInnings : ℕ
  overallAverage : ℚ
  highLowDifference : ℕ
  averageExcludingHighLow : ℚ

/-- Calculates the highest score of a batsman given their statistics --/
def highestScore (stats : BatsmanStats) : ℕ :=
  sorry

/-- Theorem stating that the highest score is 179 for the given conditions --/
theorem highest_score_is_179 (stats : BatsmanStats) 
  (h1 : stats.totalInnings = 46)
  (h2 : stats.overallAverage = 60)
  (h3 : stats.highLowDifference = 150)
  (h4 : stats.averageExcludingHighLow = 58) :
  highestScore stats = 179 := by
  sorry

end NUMINAMATH_CALUDE_highest_score_is_179_l289_28964


namespace NUMINAMATH_CALUDE_parabola_vector_sum_implies_magnitude_sum_l289_28965

noncomputable section

-- Define the parabola
def is_on_parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the vector from focus to a point
def vec_from_focus (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - focus.1, p.2 - focus.2)

-- Define the magnitude of a vector
def vec_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem parabola_vector_sum_implies_magnitude_sum
  (A B C : ℝ × ℝ)
  (hA : is_on_parabola A)
  (hB : is_on_parabola B)
  (hC : is_on_parabola C)
  (h_sum : vec_from_focus A + 2 • vec_from_focus B + 3 • vec_from_focus C = (0, 0)) :
  vec_magnitude (vec_from_focus A) + 2 * vec_magnitude (vec_from_focus B) + 3 * vec_magnitude (vec_from_focus C) = 12 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vector_sum_implies_magnitude_sum_l289_28965


namespace NUMINAMATH_CALUDE_unique_solution_l289_28905

/-- Returns the tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ := n / 10

/-- Returns the ones digit of a two-digit number -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- Returns the product of digits of a two-digit number -/
def digit_product (n : ℕ) : ℕ := tens_digit n * ones_digit n

/-- Returns the reversed number of a two-digit number -/
def reverse_number (n : ℕ) : ℕ := 10 * (ones_digit n) + tens_digit n

/-- Checks if a number satisfies the given conditions -/
def satisfies_conditions (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  (n / digit_product n = 3) ∧ (n % digit_product n = 8) ∧
  (reverse_number n / digit_product n = 2) ∧ (reverse_number n % digit_product n = 5)

theorem unique_solution : ∃! n : ℕ, satisfies_conditions n ∧ n = 53 :=
  sorry


end NUMINAMATH_CALUDE_unique_solution_l289_28905


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l289_28940

theorem cubic_equation_solutions :
  ∀ x : ℝ, (x ^ (1/3) = 15 / (8 - x ^ (1/3))) ↔ (x = 27 ∨ x = 125) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l289_28940


namespace NUMINAMATH_CALUDE_parabola_directrix_parameter_l289_28941

/-- Given a parabola with equation x^2 = ay and directrix y = 1, prove that a = -4 -/
theorem parabola_directrix_parameter (a : ℝ) : 
  (∀ x y : ℝ, x^2 = a*y) →  -- Equation of the parabola
  (1 : ℝ) = -a/4 →          -- Equation of the directrix (y = 1 is equivalent to 1 = -a/4 for a parabola)
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_parameter_l289_28941


namespace NUMINAMATH_CALUDE_pipe_length_proof_l289_28919

theorem pipe_length_proof (shorter_piece longer_piece total_length : ℕ) : 
  shorter_piece = 28 →
  longer_piece = shorter_piece + 12 →
  total_length = shorter_piece + longer_piece →
  total_length = 68 := by
  sorry

end NUMINAMATH_CALUDE_pipe_length_proof_l289_28919


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l289_28997

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g : ℚ → ℚ := λ x => c * x^3 - 8 * x^2 + d * x - 7
  (g 2 = -9) ∧ (g (-1) = -19) → c = 19/3 ∧ d = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l289_28997


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l289_28969

theorem regular_triangular_pyramid_volume 
  (b : ℝ) (β : ℝ) (h : 0 < β ∧ β < π / 2) :
  let volume := (((36 * b^2 * Real.cos β^2) / (1 + 9 * Real.cos β^2))^(3/2) * Real.tan β) / 24
  ∃ (a : ℝ), 
    a > 0 ∧ 
    volume = (a^3 * Real.tan β) / 24 ∧
    a^2 = (36 * b^2 * Real.cos β^2) / (1 + 9 * Real.cos β^2) :=
by sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l289_28969


namespace NUMINAMATH_CALUDE_angle_sum_B_plus_D_l289_28963

-- Define the triangle AFG and external angle BFD
structure Triangle :=
  (A B D F G : Real)

-- State the theorem
theorem angle_sum_B_plus_D (t : Triangle) 
  (h1 : t.A = 30) -- Given: Angle A is 30 degrees
  (h2 : t.F = t.G) -- Given: Angle AFG equals Angle AGF
  : t.B + t.D = 75 := by
  sorry


end NUMINAMATH_CALUDE_angle_sum_B_plus_D_l289_28963


namespace NUMINAMATH_CALUDE_toothbrushes_per_patient_l289_28981

/-- Calculates the number of toothbrushes given to each patient in a dental office -/
theorem toothbrushes_per_patient
  (hours_per_day : ℝ)
  (hours_per_visit : ℝ)
  (days_per_week : ℕ)
  (total_toothbrushes : ℕ)
  (h1 : hours_per_day = 8)
  (h2 : hours_per_visit = 0.5)
  (h3 : days_per_week = 5)
  (h4 : total_toothbrushes = 160) :
  (total_toothbrushes : ℝ) / ((hours_per_day / hours_per_visit) * days_per_week) = 2 := by
  sorry

end NUMINAMATH_CALUDE_toothbrushes_per_patient_l289_28981


namespace NUMINAMATH_CALUDE_basketball_players_l289_28904

theorem basketball_players (C B_and_C B_or_C : ℕ) 
  (h1 : C = 8)
  (h2 : B_and_C = 5)
  (h3 : B_or_C = 10)
  : ∃ B : ℕ, B = 7 ∧ B_or_C = B + C - B_and_C :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_players_l289_28904


namespace NUMINAMATH_CALUDE_removed_term_is_last_l289_28998

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) : ℕ → ℚ :=
  fun n => a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem removed_term_is_last
  (a₁ : ℚ)
  (avg_11 : ℚ)
  (avg_10 : ℚ)
  (h₁ : a₁ = -5)
  (h₂ : avg_11 = 5)
  (h₃ : avg_10 = 4)
  (h₄ : ∃ d : ℚ, sum_arithmetic_sequence a₁ d 11 = 11 * avg_11) :
  arithmetic_sequence a₁ ((sum_arithmetic_sequence a₁ 2 11 - sum_arithmetic_sequence a₁ 2 10) / 1) 11 =
  sum_arithmetic_sequence a₁ 2 11 - 10 * avg_10 :=
by sorry

end NUMINAMATH_CALUDE_removed_term_is_last_l289_28998


namespace NUMINAMATH_CALUDE_last_date_divisible_by_101_in_2011_l289_28952

def is_valid_date (year month day : ℕ) : Prop :=
  year = 2011 ∧ 
  1 ≤ month ∧ month ≤ 12 ∧
  1 ≤ day ∧ day ≤ 31

def date_to_number (year month day : ℕ) : ℕ :=
  year * 10000 + month * 100 + day

theorem last_date_divisible_by_101_in_2011 :
  ∀ (month day : ℕ),
    is_valid_date 2011 month day →
    date_to_number 2011 month day ≤ 20111221 ∨
    ¬(date_to_number 2011 month day % 101 = 0) :=
sorry

end NUMINAMATH_CALUDE_last_date_divisible_by_101_in_2011_l289_28952


namespace NUMINAMATH_CALUDE_grid_configurations_l289_28968

/-- Represents a grid of lightbulbs -/
structure LightbulbGrid where
  rows : Nat
  cols : Nat

/-- Represents the switches for a lightbulb grid -/
structure Switches where
  count : Nat

/-- Calculates the number of distinct configurations for a given lightbulb grid and switches -/
def distinctConfigurations (grid : LightbulbGrid) (switches : Switches) : Nat :=
  2^(switches.count - 1)

/-- Theorem: The number of distinct configurations for a 20x16 grid with 36 switches is 2^35 -/
theorem grid_configurations :
  let grid : LightbulbGrid := ⟨20, 16⟩
  let switches : Switches := ⟨36⟩
  distinctConfigurations grid switches = 2^35 := by
  sorry

#eval distinctConfigurations ⟨20, 16⟩ ⟨36⟩

end NUMINAMATH_CALUDE_grid_configurations_l289_28968


namespace NUMINAMATH_CALUDE_product_equals_difference_of_powers_l289_28920

theorem product_equals_difference_of_powers : 
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) * (3^16 + 5^16) * 
  (3^32 + 5^32) * (3^64 + 5^64) * (3^128 + 5^128) = 3^256 - 5^256 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_difference_of_powers_l289_28920


namespace NUMINAMATH_CALUDE_metal_square_weight_relation_l289_28938

/-- Represents the properties of a square metal slab -/
structure MetalSquare where
  side_length : ℝ
  weight : ℝ

/-- Theorem stating the relationship between two metal squares of the same material and thickness -/
theorem metal_square_weight_relation 
  (uniformDensity : ℝ → ℝ → ℝ) -- Function representing uniform density
  (square1 : MetalSquare) 
  (square2 : MetalSquare) 
  (h1 : square1.side_length = 4) 
  (h2 : square1.weight = 16) 
  (h3 : square2.side_length = 6) 
  (h4 : ∀ s w, uniformDensity s w = w / (s * s)) -- Density is weight divided by area
  : square2.weight = 36 := by
  sorry

end NUMINAMATH_CALUDE_metal_square_weight_relation_l289_28938


namespace NUMINAMATH_CALUDE_x_minus_y_equals_eight_l289_28960

theorem x_minus_y_equals_eight (x y : ℝ) 
  (hx : 3 = 0.15 * x) 
  (hy : 3 = 0.25 * y) : 
  x - y = 8 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_eight_l289_28960


namespace NUMINAMATH_CALUDE_corrected_mean_calculation_l289_28939

/-- Calculates the corrected mean of a set of observations after fixing recording errors -/
theorem corrected_mean_calculation (n : ℕ) (original_mean : ℝ) 
  (error1_recorded error1_actual : ℝ)
  (error2_recorded error2_actual : ℝ)
  (error3_recorded error3_actual : ℝ)
  (h1 : n = 50)
  (h2 : original_mean = 41)
  (h3 : error1_recorded = 23 ∧ error1_actual = 48)
  (h4 : error2_recorded = 42 ∧ error2_actual = 36)
  (h5 : error3_recorded = 28 ∧ error3_actual = 55) :
  let corrected_sum := n * original_mean + 
    (error1_actual - error1_recorded) + 
    (error2_actual - error2_recorded) + 
    (error3_actual - error3_recorded)
  (corrected_sum / n) = 41.92 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_calculation_l289_28939


namespace NUMINAMATH_CALUDE_must_divide_seven_l289_28934

theorem must_divide_seven (a b c d : ℕ+) 
  (h1 : Nat.gcd a.val b.val = 30)
  (h2 : Nat.gcd b.val c.val = 45)
  (h3 : Nat.gcd c.val d.val = 60)
  (h4 : 80 < Nat.gcd d.val a.val)
  (h5 : Nat.gcd d.val a.val < 120) :
  7 ∣ a.val := by
  sorry

end NUMINAMATH_CALUDE_must_divide_seven_l289_28934


namespace NUMINAMATH_CALUDE_calculation_proof_l289_28973

theorem calculation_proof :
  ((-1 : ℚ)^2 + 27/4 * (-4) / (-3)^2 = -4) ∧
  ((-36 : ℚ) * (3/4 - 5/6 + 7/9) = -25) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l289_28973


namespace NUMINAMATH_CALUDE_intersection_and_inequality_l289_28932

/-- Given real numbers a, b, c and functions f and g, prove properties about their intersections and inequalities. -/
theorem intersection_and_inequality 
  (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (hf : f = λ x => a * x + b) 
  (hg : g = λ x => a * x^2 + b * x + c) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = g x₁ ∧ f x₂ = g x₂) ∧ 
  (∃ d : ℝ, 3/2 < d ∧ d < 2 * Real.sqrt 3 ∧ 
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = g x₁ ∧ f x₂ = g x₂ ∧ d = |x₂ - x₁|) ∧
  (∀ x : ℝ, x ≤ -Real.sqrt 3 → f x < g x) := by
sorry

end NUMINAMATH_CALUDE_intersection_and_inequality_l289_28932


namespace NUMINAMATH_CALUDE_james_jello_cost_l289_28994

/-- The cost to fill a bathtub with jello --/
def jello_cost (bathtub_capacity : ℝ) (cubic_foot_to_gallon : ℝ) (gallon_weight : ℝ) 
                (jello_mix_ratio : ℝ) (jello_mix_cost : ℝ) : ℝ :=
  bathtub_capacity * cubic_foot_to_gallon * gallon_weight * jello_mix_ratio * jello_mix_cost

/-- Theorem: The cost to fill James' bathtub with jello is $270 --/
theorem james_jello_cost : 
  jello_cost 6 7.5 8 1.5 0.5 = 270 := by
  sorry

#eval jello_cost 6 7.5 8 1.5 0.5

end NUMINAMATH_CALUDE_james_jello_cost_l289_28994


namespace NUMINAMATH_CALUDE_line_intercept_ratio_l289_28954

theorem line_intercept_ratio (b : ℝ) (u v : ℝ) 
  (h1 : b ≠ 0)
  (h2 : 0 = 8 * u + b)
  (h3 : 0 = 4 * v + b) :
  u / v = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_intercept_ratio_l289_28954


namespace NUMINAMATH_CALUDE_unique_operation_equals_one_l289_28955

theorem unique_operation_equals_one : 
  (-3 + (-3) ≠ 1) ∧ 
  (-3 - (-3) ≠ 1) ∧ 
  (-3 / (-3) = 1) ∧ 
  (-3 * (-3) ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_operation_equals_one_l289_28955


namespace NUMINAMATH_CALUDE_men_earnings_l289_28907

/-- The amount earned by a group of workers given their daily wage and work duration -/
def amount_earned (num_workers : ℕ) (days : ℕ) (daily_wage : ℕ) : ℕ :=
  num_workers * days * daily_wage

theorem men_earnings (woman_daily_wage : ℕ) :
  woman_daily_wage * 40 * 30 = 21600 →
  amount_earned 16 25 (2 * woman_daily_wage) = 14400 := by
  sorry

#check men_earnings

end NUMINAMATH_CALUDE_men_earnings_l289_28907


namespace NUMINAMATH_CALUDE_sum_of_decimals_l289_28928

theorem sum_of_decimals :
  5.256 + 2.89 + 3.75 = 11.96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l289_28928


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l289_28950

theorem not_necessarily_right_triangle (A B C : ℝ) (h : A / B = 3 / 4 ∧ B / C = 4 / 5) :
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l289_28950


namespace NUMINAMATH_CALUDE_john_total_skateboard_distance_l289_28983

/-- The total distance John skateboarded, given his journey to and from the park -/
def total_skateboarded_distance (initial_skate : ℕ) (walk : ℕ) : ℕ :=
  2 * initial_skate

/-- Theorem stating that John skateboarded 20 miles in total -/
theorem john_total_skateboard_distance :
  total_skateboarded_distance 10 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_john_total_skateboard_distance_l289_28983


namespace NUMINAMATH_CALUDE_function_inequality_l289_28902

/-- The function f(x) = x^2 - 2x -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The function g(x) = ax + 2, where a > 0 -/
def g (a : ℝ) (x : ℝ) : ℝ := a*x + 2

/-- The theorem statement -/
theorem function_inequality (a : ℝ) (h_a : a > 0) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, x₂ ≥ -2 ∧ f x₁ > g a x₂) →
  a > 3/2 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l289_28902


namespace NUMINAMATH_CALUDE_two_wheeler_wheels_l289_28901

theorem two_wheeler_wheels (total_wheels : ℕ) (four_wheelers : ℕ) : total_wheels = 46 ∧ four_wheelers = 11 → 
  ∃ (two_wheelers : ℕ), two_wheelers * 2 + four_wheelers * 4 = total_wheels ∧ two_wheelers * 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_wheeler_wheels_l289_28901


namespace NUMINAMATH_CALUDE_unique_integer_solution_implies_a_range_l289_28948

theorem unique_integer_solution_implies_a_range (a : ℝ) :
  (∃! x : ℤ, (2 * x + 3 > 5 ∧ x - a ≤ 0)) →
  (2 ≤ a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_implies_a_range_l289_28948


namespace NUMINAMATH_CALUDE_championship_games_l289_28991

theorem championship_games (n : ℕ) (n_ge_2 : n ≥ 2) : 
  (n * (n - 1)) / 2 = (Finset.sum (Finset.range (n - 1)) (λ i => n - 1 - i)) :=
by sorry

end NUMINAMATH_CALUDE_championship_games_l289_28991


namespace NUMINAMATH_CALUDE_largest_fraction_l289_28924

theorem largest_fraction : 
  (200 : ℚ) / 399 > 5 / 11 ∧
  (200 : ℚ) / 399 > 7 / 15 ∧
  (200 : ℚ) / 399 > 29 / 59 ∧
  (200 : ℚ) / 399 > 251 / 501 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l289_28924


namespace NUMINAMATH_CALUDE_wire_service_coverage_l289_28971

/-- The percentage of reporters covering local politics in country x -/
def local_politics_coverage : ℝ := 12

/-- The percentage of reporters not covering politics -/
def non_politics_coverage : ℝ := 80

/-- The percentage of reporters covering politics but not local politics in country x -/
def politics_not_local : ℝ := 40

/-- Theorem stating that given the conditions, the percentage of reporters
    who cover politics but not local politics in country x is 40% -/
theorem wire_service_coverage :
  local_politics_coverage = 12 →
  non_politics_coverage = 80 →
  politics_not_local = 40 :=
by
  sorry

#check wire_service_coverage

end NUMINAMATH_CALUDE_wire_service_coverage_l289_28971


namespace NUMINAMATH_CALUDE_oliver_candy_boxes_l289_28913

theorem oliver_candy_boxes : ∃ (initial : ℕ), initial + 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_oliver_candy_boxes_l289_28913


namespace NUMINAMATH_CALUDE_infinite_solutions_sum_l289_28914

/-- If the equation ax - 4 = 14x + b has infinitely many solutions, then a + b = 10 -/
theorem infinite_solutions_sum (a b : ℝ) : 
  (∀ x, a * x - 4 = 14 * x + b) → a + b = 10 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_sum_l289_28914


namespace NUMINAMATH_CALUDE_prob_one_black_one_red_is_three_fifths_l289_28922

/-- Represents the color of a ball -/
inductive BallColor
  | Red
  | Black

/-- Represents a ball with a color and number -/
structure Ball where
  color : BallColor
  number : Nat

/-- The bag of balls -/
def bag : Finset Ball := sorry

/-- The number of red balls in the bag -/
def num_red_balls : Nat := sorry

/-- The number of black balls in the bag -/
def num_black_balls : Nat := sorry

/-- The total number of balls in the bag -/
def total_balls : Nat := sorry

/-- The probability of drawing one black ball and one red ball in the first two draws -/
def prob_one_black_one_red : ℚ := sorry

/-- Theorem stating the probability of drawing one black ball and one red ball in the first two draws -/
theorem prob_one_black_one_red_is_three_fifths :
  prob_one_black_one_red = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_prob_one_black_one_red_is_three_fifths_l289_28922


namespace NUMINAMATH_CALUDE_last_two_digits_28_l289_28956

theorem last_two_digits_28 (n : ℕ) (h : Odd n) (h_pos : 0 < n) :
  2^(2*n) * (2^(2*n + 1) - 1) ≡ 28 [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_last_two_digits_28_l289_28956


namespace NUMINAMATH_CALUDE_alexander_paintings_l289_28916

/-- The number of paintings at each new gallery --/
def paintings_per_new_gallery : ℕ := 2

theorem alexander_paintings :
  let first_gallery_paintings : ℕ := 9
  let new_galleries : ℕ := 5
  let pencils_per_painting : ℕ := 4
  let signature_pencils_per_gallery : ℕ := 2
  let total_pencils_used : ℕ := 88
  
  paintings_per_new_gallery = 
    ((total_pencils_used - 
      (signature_pencils_per_gallery * (new_galleries + 1)) - 
      (first_gallery_paintings * pencils_per_painting)) 
     / (new_galleries * pencils_per_painting)) :=
by
  sorry

end NUMINAMATH_CALUDE_alexander_paintings_l289_28916


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l289_28953

/-- The function f(x) = x^4 - 2x^3 -/
def f (x : ℝ) : ℝ := x^4 - 2*x^3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 4*x^3 - 6*x^2

theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -2*x + 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l289_28953
