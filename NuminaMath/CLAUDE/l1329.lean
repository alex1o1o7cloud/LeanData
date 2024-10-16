import Mathlib

namespace NUMINAMATH_CALUDE_rose_more_expensive_l1329_132957

/-- The price of a single rose -/
def rose_price : ℝ := sorry

/-- The price of a single carnation -/
def carnation_price : ℝ := sorry

/-- The total price of 6 roses and 3 carnations is greater than 24 yuan -/
axiom condition1 : 6 * rose_price + 3 * carnation_price > 24

/-- The total price of 4 roses and 5 carnations is less than 22 yuan -/
axiom condition2 : 4 * rose_price + 5 * carnation_price < 22

/-- The price of 2 roses is higher than the price of 3 carnations -/
theorem rose_more_expensive : 2 * rose_price > 3 * carnation_price := by
  sorry

end NUMINAMATH_CALUDE_rose_more_expensive_l1329_132957


namespace NUMINAMATH_CALUDE_cistern_emptying_time_l1329_132990

/-- Represents the cistern problem -/
theorem cistern_emptying_time 
  (volume : ℝ) 
  (time_with_tap : ℝ) 
  (tap_rate : ℝ) 
  (h1 : volume = 480) 
  (h2 : time_with_tap = 24) 
  (h3 : tap_rate = 4) : 
  (volume / (volume / time_with_tap - tap_rate) = 30) := by
  sorry

#check cistern_emptying_time

end NUMINAMATH_CALUDE_cistern_emptying_time_l1329_132990


namespace NUMINAMATH_CALUDE_equation_solutions_l1329_132916

theorem equation_solutions :
  (∀ x : ℝ, (2*x - 1)^2 - 25 = 0 ↔ x = 3 ∨ x = -2) ∧
  (∀ x : ℝ, (1/3) * (x + 3)^3 - 9 = 0 ↔ x = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1329_132916


namespace NUMINAMATH_CALUDE_junior_senior_ratio_l1329_132944

theorem junior_senior_ratio (j s : ℕ) 
  (h1 : j > 0) (h2 : s > 0)
  (h3 : (j / 3 : ℚ) = (2 * s / 3 : ℚ)) : 
  j = 2 * s := by
sorry

end NUMINAMATH_CALUDE_junior_senior_ratio_l1329_132944


namespace NUMINAMATH_CALUDE_old_socks_thrown_away_l1329_132956

def initial_socks : ℕ := 11
def new_socks : ℕ := 26
def final_socks : ℕ := 33

theorem old_socks_thrown_away : 
  initial_socks + new_socks - final_socks = 4 := by
  sorry

end NUMINAMATH_CALUDE_old_socks_thrown_away_l1329_132956


namespace NUMINAMATH_CALUDE_product_112_54_l1329_132964

theorem product_112_54 : 112 * 54 = 6048 := by
  sorry

end NUMINAMATH_CALUDE_product_112_54_l1329_132964


namespace NUMINAMATH_CALUDE_second_polygon_sides_l1329_132979

/-- Given two regular polygons with the same perimeter, where the first has 45 sides
    and a side length three times as long as the second, prove that the second polygon
    has 135 sides. -/
theorem second_polygon_sides (p₁ p₂ : ℕ) (s : ℝ) : 
  p₁ = 45 →  -- First polygon has 45 sides
  p₁ * (3 * s) = p₂ * s →  -- Same perimeter
  p₂ = 135 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l1329_132979


namespace NUMINAMATH_CALUDE_not_center_of_symmetry_l1329_132967

/-- Given that the centers of symmetry for tan(x) are of the form (kπ/2, 0) where k is any integer,
    prove that (-π/18, 0) is not a center of symmetry for the function t = tan(3x + π/3) -/
theorem not_center_of_symmetry :
  ¬ (∃ (k : ℤ), -π/18 = k*π/6 - π/9) := by sorry

end NUMINAMATH_CALUDE_not_center_of_symmetry_l1329_132967


namespace NUMINAMATH_CALUDE_smallest_c_value_l1329_132910

-- Define the polynomial
def polynomial (c d x : ℤ) : ℤ := x^3 - c*x^2 + d*x - 2730

-- Define the property that the polynomial has three positive integer roots
def has_three_positive_integer_roots (c d : ℤ) : Prop :=
  ∃ (r₁ r₂ r₃ : ℤ), r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
    ∀ x, polynomial c d x = (x - r₁) * (x - r₂) * (x - r₃)

-- Theorem statement
theorem smallest_c_value (c d : ℤ) :
  has_three_positive_integer_roots c d → c ≥ 54 :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_value_l1329_132910


namespace NUMINAMATH_CALUDE_souvenir_shop_theorem_l1329_132931

/-- Represents the purchase and profit scenario of a souvenir shop. -/
structure SouvenirShop where
  price_A : ℚ  -- Purchase price of souvenir A
  price_B : ℚ  -- Purchase price of souvenir B
  profit_A : ℚ -- Profit per piece of souvenir A
  profit_B : ℚ -- Profit per piece of souvenir B

/-- Theorem stating the correct purchase prices and total profit -/
theorem souvenir_shop_theorem (shop : SouvenirShop) : 
  (7 * shop.price_A + 8 * shop.price_B = 380) →
  (10 * shop.price_A + 6 * shop.price_B = 380) →
  shop.profit_A = 5 →
  shop.profit_B = 7 →
  (∃ (m n : ℚ), m + n = 40 ∧ shop.price_A * m + shop.price_B * n = 900) →
  (shop.price_A = 20 ∧ shop.price_B = 30 ∧ 
   ∃ (m n : ℚ), m + n = 40 ∧ shop.price_A * m + shop.price_B * n = 900 ∧
                m * shop.profit_A + n * shop.profit_B = 220) := by
  sorry


end NUMINAMATH_CALUDE_souvenir_shop_theorem_l1329_132931


namespace NUMINAMATH_CALUDE_system_solution_unique_l1329_132982

theorem system_solution_unique :
  ∃! (x y : ℚ), 37 * x + 92 * y = 5043 ∧ 92 * x + 37 * y = 2568 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1329_132982


namespace NUMINAMATH_CALUDE_g_100_zeros_l1329_132954

-- Define g₀
def g₀ (x : ℝ) : ℝ := x + |x - 150| - |x + 150|

-- Define gₙ recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 2

-- Theorem statement
theorem g_100_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ g 100 a = 0 ∧ g 100 b = 0 ∧
  ∀ (x : ℝ), g 100 x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_g_100_zeros_l1329_132954


namespace NUMINAMATH_CALUDE_aron_dusting_time_l1329_132905

/-- Represents the cleaning schedule and durations for Aron --/
structure CleaningSchedule where
  vacuum_duration : ℕ  -- Minutes spent vacuuming per day
  vacuum_frequency : ℕ  -- Days per week spent vacuuming
  dust_frequency : ℕ  -- Days per week spent dusting
  total_cleaning_time : ℕ  -- Total minutes spent cleaning per week

/-- Calculates the time spent dusting per day given a cleaning schedule --/
def dusting_time_per_day (schedule : CleaningSchedule) : ℕ :=
  let total_vacuum_time := schedule.vacuum_duration * schedule.vacuum_frequency
  let total_dust_time := schedule.total_cleaning_time - total_vacuum_time
  total_dust_time / schedule.dust_frequency

/-- Theorem stating that Aron spends 20 minutes dusting each day --/
theorem aron_dusting_time (schedule : CleaningSchedule) 
    (h1 : schedule.vacuum_duration = 30)
    (h2 : schedule.vacuum_frequency = 3)
    (h3 : schedule.dust_frequency = 2)
    (h4 : schedule.total_cleaning_time = 130) :
  dusting_time_per_day schedule = 20 := by
  sorry

end NUMINAMATH_CALUDE_aron_dusting_time_l1329_132905


namespace NUMINAMATH_CALUDE_sequence_eventually_constant_l1329_132936

/-- A sequence of non-negative integers satisfying the given conditions -/
def Sequence (m : ℕ+) := { a : ℕ → ℕ // 
  a 0 = m ∧ 
  (∀ n : ℕ, n ≥ 1 → a n ≤ n) ∧
  (∀ n : ℕ+, (n : ℕ) ∣ (Finset.range n).sum (λ i => a i)) }

/-- The main theorem -/
theorem sequence_eventually_constant (m : ℕ+) (a : Sequence m) : 
  ∃ M : ℕ, ∀ n ≥ M, a.val n = a.val M :=
sorry

end NUMINAMATH_CALUDE_sequence_eventually_constant_l1329_132936


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_b_part2_l1329_132995

-- Part 1
def quadratic_inequality (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c ≤ -1

theorem solution_set_part1 (a : ℝ) (h1 : a > 0) :
  let b := -2 * a - 2
  let c := 3
  (∀ x, quadratic_inequality a b c x ↔ 
    (0 < a ∧ a < 1 ∧ 2 ≤ x ∧ x ≤ 2/a) ∨
    (a = 1 ∧ x = 2) ∨
    (a > 1 ∧ 2/a ≤ x ∧ x ≤ 2)) := by sorry

-- Part 2
def quadratic_inequality_part2 (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c ≥ (3/2) * b * x

theorem range_of_b_part2 :
  ∃ b : ℝ, (∀ x, 1 ≤ x ∧ x ≤ 5 → quadratic_inequality_part2 1 b 2 x) ∧
    b ≤ 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_b_part2_l1329_132995


namespace NUMINAMATH_CALUDE_sticker_distribution_l1329_132989

theorem sticker_distribution (total : ℕ) (andrew_kept : ℕ) (daniel_received : ℕ) 
  (h1 : total = 750)
  (h2 : andrew_kept = 130)
  (h3 : daniel_received = 250) :
  total - andrew_kept - daniel_received - daniel_received = 120 :=
by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l1329_132989


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_seven_l1329_132980

theorem sqrt_difference_equals_seven : Real.sqrt (36 + 64) - Real.sqrt (25 - 16) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_seven_l1329_132980


namespace NUMINAMATH_CALUDE_part_one_part_two_l1329_132900

-- Part 1
theorem part_one (x : ℝ) (a b : ℝ × ℝ) :
  a = (Real.sqrt 3 * Real.sin x, -1) →
  b = (Real.cos x, Real.sqrt 3) →
  ∃ (k : ℝ), a = k • b →
  (3 * Real.sin x - Real.cos x) / (Real.sin x + Real.cos x) = -3 :=
sorry

-- Part 2
def f (x m : ℝ) (a b : ℝ × ℝ) : ℝ :=
  2 * ((a.1 + b.1) * b.1 + (a.2 + b.2) * b.2) - 2 * m^2 - 1

theorem part_two (m : ℝ) :
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧
    f x m ((Real.sqrt 3 * Real.sin x, -1)) ((Real.cos x, m)) = 0) →
  m ∈ Set.Icc (-1/2) 1 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1329_132900


namespace NUMINAMATH_CALUDE_remainder_71_cubed_73_fifth_mod_8_l1329_132952

theorem remainder_71_cubed_73_fifth_mod_8 : (71^3 * 73^5) % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_71_cubed_73_fifth_mod_8_l1329_132952


namespace NUMINAMATH_CALUDE_russian_players_pairing_probability_l1329_132913

/-- The probability of all Russian players pairing only with other Russian players in a random pairing -/
theorem russian_players_pairing_probability 
  (total_players : ℕ) 
  (russian_players : ℕ) 
  (h1 : total_players = 10) 
  (h2 : russian_players = 4) 
  (h3 : russian_players ≤ total_players) :
  (russian_players.choose 2 : ℚ) / total_players.choose 2 = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_russian_players_pairing_probability_l1329_132913


namespace NUMINAMATH_CALUDE_gcd_102_238_l1329_132925

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by sorry

end NUMINAMATH_CALUDE_gcd_102_238_l1329_132925


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l1329_132996

theorem smallest_b_for_factorization (b : ℕ) : b = 121 ↔ 
  (b > 0 ∧ 
   ∃ (r s : ℕ), r * s = 2020 ∧ r > s ∧
   ∀ (x : ℤ), x^2 + b*x + 2020 = (x + r) * (x + s) ∧
   ∀ (b' : ℕ), b' > 0 → 
     (∃ (r' s' : ℕ), r' * s' = 2020 ∧ r' > s' ∧
     ∀ (x : ℤ), x^2 + b'*x + 2020 = (x + r') * (x + s')) →
     b ≤ b') := by
sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l1329_132996


namespace NUMINAMATH_CALUDE_min_value_of_function_l1329_132902

theorem min_value_of_function (x : ℝ) (h : x > 2) :
  ∃ (y : ℝ), y = x + 4 / (x - 2) ∧ (∀ (z : ℝ), z = x + 4 / (x - 2) → y ≤ z) ∧ y = 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1329_132902


namespace NUMINAMATH_CALUDE_arithmetic_sequence_prime_divisibility_l1329_132976

theorem arithmetic_sequence_prime_divisibility 
  (n : ℕ) 
  (a : ℕ → ℕ) 
  (h_n : n ≥ 2021) 
  (h_arith : ∀ i j, i < j → j ≤ n → a j - a i = (j - i) * (a 2 - a 1))
  (h_inc : ∀ i j, i < j → j ≤ n → a i < a j)
  (h_first : a 1 > 2021)
  (h_prime : ∀ i, 1 ≤ i → i ≤ n → Nat.Prime (a i)) :
  ∀ p, p < 2021 → Nat.Prime p → (a 2 - a 1) % p = 0 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_prime_divisibility_l1329_132976


namespace NUMINAMATH_CALUDE_max_digit_sum_l1329_132948

/-- Represents a digit (1-9) -/
def Digit := {d : Nat // d > 0 ∧ d ≤ 9}

/-- An is an n-digit integer with all digits equal to a -/
def A (n : Nat) (a : Digit) : Nat :=
  a.val * (10^n - 1) / 9

/-- Bn is an n-digit integer with all digits equal to b -/
def B (n : Nat) (b : Digit) : Nat :=
  b.val * (10^n - 1) / 9

/-- Cn is a 2n-digit integer with all digits equal to c -/
def C (n : Nat) (c : Digit) : Nat :=
  c.val * (10^(2*n) - 1) / 9

/-- The equation Cn - Bn = An^2 holds for at least two values of n -/
def EquationHoldsTwice (a b c : Digit) : Prop :=
  ∃ n1 n2 : Nat, n1 ≠ n2 ∧
    C n1 c - B n1 b = (A n1 a)^2 ∧
    C n2 c - B n2 b = (A n2 a)^2

theorem max_digit_sum :
  ∀ a b c : Digit, EquationHoldsTwice a b c →
    a.val + b.val + c.val ≤ 18 :=
sorry

end NUMINAMATH_CALUDE_max_digit_sum_l1329_132948


namespace NUMINAMATH_CALUDE_semicircles_to_circle_area_ratio_l1329_132992

/-- The ratio of the combined areas of two semicircles with radius r to the area of a circle with radius r is 1 -/
theorem semicircles_to_circle_area_ratio (r : ℝ) (hr : r > 0) : 
  (2 * (1 / 2 * π * r^2)) / (π * r^2) = 1 := by
sorry

end NUMINAMATH_CALUDE_semicircles_to_circle_area_ratio_l1329_132992


namespace NUMINAMATH_CALUDE_problem_statement_l1329_132901

theorem problem_statement :
  (∀ x : ℝ, |x| ≥ 0) ∧
  (1^2 + 1 + 1 ≠ 0) ∧
  ((∀ x : ℝ, |x| ≥ 0) ∧ (1^2 + 1 + 1 ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1329_132901


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1329_132981

theorem imaginary_part_of_z (z : ℂ) : z = (3 - I) / (1 + I) → z.im = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1329_132981


namespace NUMINAMATH_CALUDE_anya_additional_biscuits_l1329_132983

/-- Represents the distribution of biscuits and payments among three sisters. -/
structure BiscuitDistribution where
  total_biscuits : ℕ
  total_payment : ℕ
  anya_payment : ℕ
  berini_payment : ℕ
  carla_payment : ℕ

/-- Calculates the number of additional biscuits Anya would receive if distributed proportionally to payments. -/
def additional_biscuits_for_anya (bd : BiscuitDistribution) : ℕ :=
  let equal_share := bd.total_biscuits / 3
  let proportional_share := (bd.anya_payment * bd.total_biscuits) / bd.total_payment
  proportional_share - equal_share

/-- Theorem stating that Anya would receive 6 more biscuits in a proportional distribution. -/
theorem anya_additional_biscuits :
  ∀ (bd : BiscuitDistribution),
  bd.total_biscuits = 30 ∧
  bd.total_payment = 150 ∧
  bd.anya_payment = 80 ∧
  bd.berini_payment = 50 ∧
  bd.carla_payment = 20 →
  additional_biscuits_for_anya bd = 6 := by
  sorry

end NUMINAMATH_CALUDE_anya_additional_biscuits_l1329_132983


namespace NUMINAMATH_CALUDE_unique_integer_expression_l1329_132946

/-- The function representing the given expression -/
def f (x y : ℕ+) : ℚ := (x^2 + y) / (x * y + 1)

/-- The theorem stating that 1 is the only positive integer expressible
    by the function for at least two distinct pairs of positive integers -/
theorem unique_integer_expression :
  ∀ n : ℕ+, (∃ (x₁ y₁ x₂ y₂ : ℕ+), (x₁, y₁) ≠ (x₂, y₂) ∧ f x₁ y₁ = n ∧ f x₂ y₂ = n) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_expression_l1329_132946


namespace NUMINAMATH_CALUDE_cost_sharing_equalization_l1329_132958

theorem cost_sharing_equalization (A B : ℝ) (h : A < B) : 
  let total_cost := A + B
  let equal_share := total_cost / 2
  let amount_to_pay := equal_share - A
  amount_to_pay = (B - A) / 2 := by
sorry

end NUMINAMATH_CALUDE_cost_sharing_equalization_l1329_132958


namespace NUMINAMATH_CALUDE_arrangement_with_A_middle_or_sides_arrangement_with_males_grouped_arrangement_with_males_not_grouped_arrangement_with_ABC_order_fixed_arrangement_A_not_left_B_not_right_arrangement_with_extra_female_no_adjacent_arrangement_in_two_rows_arrangement_with_person_between_A_and_B_l1329_132945

-- Common definitions
def male_students : ℕ := 3
def female_students : ℕ := 2
def total_students : ℕ := male_students + female_students

-- (1)
theorem arrangement_with_A_middle_or_sides :
  (3 * (total_students - 1).factorial) = 72 := by sorry

-- (2)
theorem arrangement_with_males_grouped :
  (male_students.factorial * (total_students - male_students + 1).factorial) = 36 := by sorry

-- (3)
theorem arrangement_with_males_not_grouped :
  (female_students.factorial * male_students.factorial) = 12 := by sorry

-- (4)
theorem arrangement_with_ABC_order_fixed :
  (total_students.factorial / male_students.factorial) = 20 := by sorry

-- (5)
theorem arrangement_A_not_left_B_not_right :
  ((total_students - 1) * (total_students - 1).factorial - 
   (total_students - 2) * (total_students - 2).factorial) = 78 := by sorry

-- (6)
def extra_female_student : ℕ := 1
def new_total_students : ℕ := total_students + extra_female_student

theorem arrangement_with_extra_female_no_adjacent :
  (male_students.factorial * (new_total_students - male_students + 1).factorial) = 144 := by sorry

-- (7)
theorem arrangement_in_two_rows :
  total_students.factorial = 120 := by sorry

-- (8)
theorem arrangement_with_person_between_A_and_B :
  (3 * 2 * male_students.factorial) = 36 := by sorry

end NUMINAMATH_CALUDE_arrangement_with_A_middle_or_sides_arrangement_with_males_grouped_arrangement_with_males_not_grouped_arrangement_with_ABC_order_fixed_arrangement_A_not_left_B_not_right_arrangement_with_extra_female_no_adjacent_arrangement_in_two_rows_arrangement_with_person_between_A_and_B_l1329_132945


namespace NUMINAMATH_CALUDE_ice_cream_bill_l1329_132972

theorem ice_cream_bill (cost_per_scoop : ℕ) (pierre_scoops : ℕ) (mom_scoops : ℕ) : 
  cost_per_scoop = 2 → pierre_scoops = 3 → mom_scoops = 4 → 
  cost_per_scoop * (pierre_scoops + mom_scoops) = 14 := by
  sorry

#check ice_cream_bill

end NUMINAMATH_CALUDE_ice_cream_bill_l1329_132972


namespace NUMINAMATH_CALUDE_grape_ratio_theorem_l1329_132938

/-- Represents the contents and cost of a fruit basket -/
structure FruitBasket where
  banana_count : ℕ
  apple_count : ℕ
  strawberry_count : ℕ
  avocado_count : ℕ
  banana_price : ℚ
  apple_price : ℚ
  strawberry_price : ℚ
  avocado_price : ℚ
  grape_portion_price : ℚ
  total_cost : ℚ

/-- Calculates the cost of fruits excluding grapes -/
def cost_excluding_grapes (fb : FruitBasket) : ℚ :=
  fb.banana_count * fb.banana_price +
  fb.apple_count * fb.apple_price +
  fb.strawberry_count / 12 * fb.strawberry_price +
  fb.avocado_count * fb.avocado_price

/-- Calculates the cost of grapes in the basket -/
def grape_cost (fb : FruitBasket) : ℚ :=
  fb.total_cost - cost_excluding_grapes fb

/-- Represents the ratio of grapes in the basket to a whole bunch -/
structure GrapeRatio where
  numerator : ℚ
  denominator : ℚ

/-- Theorem stating the ratio of grapes in the basket to a whole bunch -/
theorem grape_ratio_theorem (fb : FruitBasket) (x : ℚ) :
  fb.banana_count = 4 →
  fb.apple_count = 3 →
  fb.strawberry_count = 24 →
  fb.avocado_count = 2 →
  fb.banana_price = 1 →
  fb.apple_price = 2 →
  fb.strawberry_price = 4 →
  fb.avocado_price = 3 →
  fb.grape_portion_price = 2 →
  fb.total_cost = 28 →
  x > 2 →
  ∃ (gr : GrapeRatio), gr.numerator = 2 ∧ gr.denominator = x :=
by sorry

end NUMINAMATH_CALUDE_grape_ratio_theorem_l1329_132938


namespace NUMINAMATH_CALUDE_jacoby_work_hours_l1329_132939

/-- The problem of calculating Jacoby's work hours -/
theorem jacoby_work_hours :
  let trip_cost : ℕ := 5000
  let hourly_wage : ℕ := 20
  let cookies_sold : ℕ := 24
  let cookie_price : ℕ := 4
  let lottery_ticket_cost : ℕ := 10
  let lottery_winnings : ℕ := 500
  let sister_gift : ℕ := 500
  let remaining_needed : ℕ := 3214

  let cookie_earnings := cookies_sold * cookie_price
  let gifts := sister_gift * 2
  let other_income := cookie_earnings + lottery_winnings + gifts - lottery_ticket_cost
  let total_earned := trip_cost - remaining_needed
  let job_earnings := total_earned - other_income
  let hours_worked := job_earnings / hourly_wage

  hours_worked = 10 := by sorry

end NUMINAMATH_CALUDE_jacoby_work_hours_l1329_132939


namespace NUMINAMATH_CALUDE_initial_cells_eq_one_l1329_132978

/-- Represents the doubling time of the bacteria in minutes -/
def doubling_time : ℕ := 20

/-- Represents the growth time in hours -/
def growth_time : ℕ := 4

/-- Represents the final number of bacterial cells -/
def final_cells : ℕ := 4096

/-- Calculates the number of doublings that occurred during the growth period -/
def num_doublings : ℕ := growth_time * 60 / doubling_time

/-- Represents the initial number of bacterial cells -/
def initial_cells : ℕ := final_cells / (2^num_doublings)

/-- Proves that the initial number of cells was 1 -/
theorem initial_cells_eq_one : initial_cells = 1 := by
  sorry

end NUMINAMATH_CALUDE_initial_cells_eq_one_l1329_132978


namespace NUMINAMATH_CALUDE_shifted_parabola_passes_through_point_l1329_132918

/-- The original parabola equation -/
def original_parabola (x : ℝ) : ℝ := -x^2 - 2*x + 3

/-- The shifted parabola equation -/
def shifted_parabola (x : ℝ) : ℝ := -x^2 + 2

/-- Theorem stating that the shifted parabola passes through (-1, 1) -/
theorem shifted_parabola_passes_through_point :
  shifted_parabola (-1) = 1 := by sorry

end NUMINAMATH_CALUDE_shifted_parabola_passes_through_point_l1329_132918


namespace NUMINAMATH_CALUDE_greatest_seven_digit_divisible_by_lcm_l1329_132963

def is_seven_digit (n : ℕ) : Prop := n ≥ 1000000 ∧ n ≤ 9999999

def lcm_primes : ℕ := 41 * 43 * 47 * 53

theorem greatest_seven_digit_divisible_by_lcm :
  ∀ n : ℕ, is_seven_digit n → n % lcm_primes = 0 → n ≤ 8833702 := by sorry

end NUMINAMATH_CALUDE_greatest_seven_digit_divisible_by_lcm_l1329_132963


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l1329_132991

theorem quadratic_root_theorem (p q : ℝ) : 
  let a := p + q
  let b := p - q
  let c := p * q
  (∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = 1) →
  (∃ y : ℝ, a * y^2 + b * y + c = 0 ∧ y = -2*p/(p-2) ∧ p ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l1329_132991


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l1329_132950

/-- Proves that given a car traveling for two hours with an average speed of 45 km/h
    and a speed of 60 km/h in the first hour, the speed in the second hour must be 30 km/h. -/
theorem car_speed_second_hour
  (average_speed : ℝ)
  (first_hour_speed : ℝ)
  (total_time : ℝ)
  (h_average_speed : average_speed = 45)
  (h_first_hour_speed : first_hour_speed = 60)
  (h_total_time : total_time = 2)
  : (2 * average_speed - first_hour_speed = 30) :=
by sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l1329_132950


namespace NUMINAMATH_CALUDE_overall_score_calculation_l1329_132923

/-- Calculate the overall score for a job applicant given their test scores and weights -/
theorem overall_score_calculation
  (written_score : ℝ)
  (interview_score : ℝ)
  (written_weight : ℝ)
  (interview_weight : ℝ)
  (h1 : written_score = 80)
  (h2 : interview_score = 60)
  (h3 : written_weight = 0.6)
  (h4 : interview_weight = 0.4)
  (h5 : written_weight + interview_weight = 1) :
  written_score * written_weight + interview_score * interview_weight = 72 :=
by sorry

end NUMINAMATH_CALUDE_overall_score_calculation_l1329_132923


namespace NUMINAMATH_CALUDE_cafeteria_extra_apples_l1329_132919

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 33

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 23

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 21

/-- Each student takes one apple -/
axiom one_apple_per_student : ℕ

/-- The number of extra apples the cafeteria ended up with -/
def extra_apples : ℕ := (red_apples + green_apples) - students_wanting_fruit

theorem cafeteria_extra_apples : extra_apples = 35 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_extra_apples_l1329_132919


namespace NUMINAMATH_CALUDE_derivative_of_f_derivative_of_f_at_2_l1329_132915

-- Define the function f(x) = x^2 + x
def f (x : ℝ) : ℝ := x^2 + x

-- Theorem 1: The derivative of f(x) is 2x + 1
theorem derivative_of_f (x : ℝ) : deriv f x = 2 * x + 1 := by sorry

-- Theorem 2: The derivative of f(x) at x = 2 is 5
theorem derivative_of_f_at_2 : deriv f 2 = 5 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_derivative_of_f_at_2_l1329_132915


namespace NUMINAMATH_CALUDE_root_sum_transformation_l1329_132968

theorem root_sum_transformation (α β γ : ℂ) : 
  (α^3 - α - 1 = 0) → (β^3 - β - 1 = 0) → (γ^3 - γ - 1 = 0) →
  ((1 - α) / (1 + α)) + ((1 - β) / (1 + β)) + ((1 - γ) / (1 + γ)) = 1 := by
sorry

end NUMINAMATH_CALUDE_root_sum_transformation_l1329_132968


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1329_132929

-- Define the condition for a line to be tangent to the circle
def is_tangent (k : ℝ) : Prop :=
  1 + k^2 = 4

-- Define the main theorem
theorem sufficient_not_necessary :
  (∀ k, k = Real.sqrt 3 → is_tangent k) ∧
  (∃ k, k ≠ Real.sqrt 3 ∧ is_tangent k) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1329_132929


namespace NUMINAMATH_CALUDE_new_hires_all_women_l1329_132977

theorem new_hires_all_women 
  (initial_workers : ℕ) 
  (new_hires : ℕ) 
  (initial_men_fraction : ℚ) 
  (final_women_percentage : ℚ) :
  initial_workers = 90 →
  new_hires = 10 →
  initial_men_fraction = 2/3 →
  final_women_percentage = 40/100 →
  (initial_workers * (1 - initial_men_fraction) + new_hires) / (initial_workers + new_hires) = final_women_percentage →
  new_hires / new_hires = 1 :=
by sorry

end NUMINAMATH_CALUDE_new_hires_all_women_l1329_132977


namespace NUMINAMATH_CALUDE_cones_paths_count_l1329_132965

/-- Represents a position in the diagram --/
structure Position :=
  (row : Fin 5) (col : Fin 5)

/-- Represents a letter in the diagram --/
inductive Letter
  | C | O | N | E | S

/-- The diagram structure --/
def diagram : Position → Option Letter := sorry

/-- Checks if two positions are adjacent --/
def adjacent (p1 p2 : Position) : Prop := sorry

/-- Represents a valid path in the diagram --/
def ValidPath : List Position → Prop := sorry

/-- Checks if a path spells "CONES" --/
def spellsCONES (path : List Position) : Prop := sorry

/-- The main theorem to prove --/
theorem cones_paths_count :
  (∃! (paths : Finset (List Position)),
    (∀ path ∈ paths, ValidPath path ∧ spellsCONES path) ∧
    paths.card = 6) := by sorry

end NUMINAMATH_CALUDE_cones_paths_count_l1329_132965


namespace NUMINAMATH_CALUDE_vote_participation_l1329_132935

theorem vote_participation (veggie_percentage : ℝ) (veggie_votes : ℕ) (total_students : ℕ) : 
  veggie_percentage = 0.28 →
  veggie_votes = 280 →
  (veggie_percentage * total_students : ℝ) = veggie_votes →
  total_students = 1000 := by
sorry

end NUMINAMATH_CALUDE_vote_participation_l1329_132935


namespace NUMINAMATH_CALUDE_cd_cost_calculation_l1329_132904

/-- The cost of the CD that Ibrahim wants to buy -/
def cd_cost : ℝ := 19

/-- The cost of the MP3 player -/
def mp3_cost : ℝ := 120

/-- Ibrahim's savings -/
def savings : ℝ := 55

/-- Money given by Ibrahim's father -/
def father_contribution : ℝ := 20

/-- The amount Ibrahim lacks after his savings and father's contribution -/
def amount_lacking : ℝ := 64

theorem cd_cost_calculation :
  cd_cost = mp3_cost + cd_cost - (savings + father_contribution) - amount_lacking :=
by sorry

end NUMINAMATH_CALUDE_cd_cost_calculation_l1329_132904


namespace NUMINAMATH_CALUDE_trajectory_equation_l1329_132974

theorem trajectory_equation (x y : ℝ) :
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (1, 0)
  let P : ℝ × ℝ := (x, y)
  let PA : ℝ := Real.sqrt ((x + 2)^2 + y^2)
  let PB : ℝ := Real.sqrt ((x - 1)^2 + y^2)
  PA = 2 * PB → x^2 + y^2 - 4*x = 0 := by
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l1329_132974


namespace NUMINAMATH_CALUDE_base3_10201_equals_100_l1329_132984

def base3ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * 3^i) 0

theorem base3_10201_equals_100 :
  base3ToDecimal [1, 0, 2, 0, 1] = 100 := by
  sorry

end NUMINAMATH_CALUDE_base3_10201_equals_100_l1329_132984


namespace NUMINAMATH_CALUDE_even_function_inequality_l1329_132907

/-- An even function satisfying the given condition -/
def EvenFunctionWithCondition (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧
  (∀ x₁ x₂, x₁ ≠ x₂ → x₁ ≤ 0 → x₂ ≤ 0 → (x₂ - x₁) * (f x₂ - f x₁) > 0)

/-- The main theorem -/
theorem even_function_inequality (f : ℝ → ℝ) (n : ℕ) (hn : n > 0) 
  (hf : EvenFunctionWithCondition f) : 
  f (n + 1) < f (-n) ∧ f (-n) < f (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l1329_132907


namespace NUMINAMATH_CALUDE_two_number_problem_l1329_132911

theorem two_number_problem :
  ∃ (x y : ℕ), x > y ∧ x - y = 4 ∧ x * y = 80 ∧ (Even x ∨ Even y) ∧ x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_two_number_problem_l1329_132911


namespace NUMINAMATH_CALUDE_pyramid_volume_transformation_l1329_132921

theorem pyramid_volume_transformation (s h : ℝ) : 
  (1/3 : ℝ) * s^2 * h = 72 → 
  (1/3 : ℝ) * (3*s)^2 * (2*h) = 1296 := by
sorry

end NUMINAMATH_CALUDE_pyramid_volume_transformation_l1329_132921


namespace NUMINAMATH_CALUDE_newspaper_cost_difference_l1329_132973

/-- The amount Grant spends yearly on newspaper delivery -/
def grant_yearly_cost : ℝ := 200

/-- The amount Juanita spends on newspapers Monday through Saturday -/
def juanita_weekday_cost : ℝ := 0.5

/-- The amount Juanita spends on newspapers on Sunday -/
def juanita_sunday_cost : ℝ := 2

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The number of weekdays (Monday through Saturday) -/
def weekdays : ℕ := 6

theorem newspaper_cost_difference : 
  (weekdays * juanita_weekday_cost + juanita_sunday_cost) * weeks_per_year - grant_yearly_cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_cost_difference_l1329_132973


namespace NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_2730_l1329_132988

theorem sum_of_extreme_prime_factors_of_2730 : 
  ∃ (smallest largest : ℕ), 
    smallest.Prime ∧ 
    largest.Prime ∧ 
    smallest ∣ 2730 ∧ 
    largest ∣ 2730 ∧ 
    (∀ p : ℕ, p.Prime → p ∣ 2730 → p ≥ smallest ∧ p ≤ largest) ∧ 
    smallest + largest = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_2730_l1329_132988


namespace NUMINAMATH_CALUDE_unique_real_root_of_cubic_l1329_132986

theorem unique_real_root_of_cubic (α : Real) (h : 0 ≤ α ∧ α ≤ Real.pi / 2) :
  ∃! x : Real, x^3 + x^2 * Real.cos α + x * Real.sin α + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_real_root_of_cubic_l1329_132986


namespace NUMINAMATH_CALUDE_calculate_total_income_person_total_income_l1329_132940

/-- Calculates a person's total income based on given distributions and remaining amount. -/
theorem calculate_total_income (children_percentage : ℝ) (wife_percentage : ℝ) 
  (orphan_donation_percentage : ℝ) (remaining_amount : ℝ) : ℝ :=
  let total_distributed_percentage := 3 * children_percentage + wife_percentage
  let remaining_percentage := 1 - total_distributed_percentage
  let final_remaining_percentage := remaining_percentage * (1 - orphan_donation_percentage)
  remaining_amount / final_remaining_percentage

/-- Proves that the person's total income is $1,000,000 given the conditions. -/
theorem person_total_income : 
  calculate_total_income 0.2 0.3 0.05 50000 = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_total_income_person_total_income_l1329_132940


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l1329_132960

/-- Given a line with equation y - 3 = -3(x - 5), 
    the sum of its x-intercept and y-intercept is 24 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y - 3 = -3 * (x - 5)) → 
  ∃ (x_int y_int : ℝ), 
    (y_int - 3 = -3 * (x_int - 5)) ∧ 
    (0 - 3 = -3 * (x_int - 5)) ∧ 
    (y_int - 3 = -3 * (0 - 5)) ∧ 
    (x_int + y_int = 24) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l1329_132960


namespace NUMINAMATH_CALUDE_min_value_abc_l1329_132951

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 9*a + 4*b = a*b*c) : 
  a + b + c ≥ 10 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    9*a₀ + 4*b₀ = a₀*b₀*c₀ ∧ a₀ + b₀ + c₀ = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abc_l1329_132951


namespace NUMINAMATH_CALUDE_constant_function_property_l1329_132903

theorem constant_function_property (f : ℝ → ℝ) (h : ∀ x, f (4 * x) = 4) :
  ∀ x, f (2 * x) = 4 := by
sorry

end NUMINAMATH_CALUDE_constant_function_property_l1329_132903


namespace NUMINAMATH_CALUDE_factor_divisor_statements_l1329_132924

theorem factor_divisor_statements :
  (∃ k : ℕ, 45 = 5 * k) ∧
  (∃ m : ℕ, 42 = 14 * m) ∧
  (∀ n : ℕ, 63 ≠ 14 * n) ∧
  (∃ p : ℕ, 180 = 9 * p) := by
  sorry

end NUMINAMATH_CALUDE_factor_divisor_statements_l1329_132924


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_36_56_l1329_132955

theorem gcf_lcm_sum_36_56 : Nat.gcd 36 56 + Nat.lcm 36 56 = 508 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_36_56_l1329_132955


namespace NUMINAMATH_CALUDE_percent_relation_l1329_132999

theorem percent_relation (x y z : ℝ) 
  (h1 : 0.45 * z = 0.96 * y) 
  (h2 : y = 0.75 * x) : 
  z = 1.6 * x := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l1329_132999


namespace NUMINAMATH_CALUDE_ice_cream_line_count_l1329_132934

theorem ice_cream_line_count (between : ℕ) (h : between = 5) : 
  between + 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_line_count_l1329_132934


namespace NUMINAMATH_CALUDE_elizabeth_study_time_l1329_132927

/-- Calculates the study time for math test given total study time and science test study time -/
def math_study_time (total_time science_time : ℕ) : ℕ :=
  total_time - science_time

/-- Theorem stating that given the total study time of 60 minutes and science test study time of 25 minutes, 
    the math test study time is 35 minutes -/
theorem elizabeth_study_time : 
  math_study_time 60 25 = 35 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_study_time_l1329_132927


namespace NUMINAMATH_CALUDE_spade_operation_result_l1329_132962

-- Define the spade operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_operation_result : spade 2 (spade 4 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_spade_operation_result_l1329_132962


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l1329_132937

def x : ℕ := 9 * 36 * 54

theorem smallest_y_for_perfect_cube (y : ℕ) : 
  (∀ z < y, ∃ (a b : ℕ), x * z = a^3 → False) ∧
  (∃ (a : ℕ), x * y = a^3) →
  y = 9 := by
sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l1329_132937


namespace NUMINAMATH_CALUDE_product_equality_l1329_132969

def prod (a : ℕ → ℕ) (m n : ℕ) : ℕ :=
  if m > n then 1 else (List.range (n - m + 1)).foldl (fun acc i => acc * a (i + m)) 1

theorem product_equality :
  (prod (fun k => 2 * k - 1) 1 1008) * (prod (fun k => 2 * k) 1 1007) = prod id 1 2015 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1329_132969


namespace NUMINAMATH_CALUDE_four_digit_square_same_digits_l1329_132959

theorem four_digit_square_same_digits : ∃! N : ℕ,
  (1000 ≤ N) ∧ (N ≤ 9999) ∧
  (∃ k : ℕ, N = k^2) ∧
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ N = 1100*a + 11*b) ∧
  N = 7744 := by
sorry

end NUMINAMATH_CALUDE_four_digit_square_same_digits_l1329_132959


namespace NUMINAMATH_CALUDE_range_of_m_l1329_132993

theorem range_of_m (m : ℝ) : 
  (¬((m + 1 ≤ 0) ∧ (∀ x : ℝ, x^2 + m*x + 1 > 0))) → 
  m ∈ Set.Iic (-2) ∪ Set.Ioi (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1329_132993


namespace NUMINAMATH_CALUDE_family_egg_count_l1329_132961

/-- Calculates the final number of eggs a family has after using some and chickens laying new ones. -/
def finalEggCount (initialEggs usedEggs chickens eggsPerChicken : ℕ) : ℕ :=
  initialEggs - usedEggs + chickens * eggsPerChicken

/-- Proves that for the given scenario, the family ends up with 11 eggs. -/
theorem family_egg_count : finalEggCount 10 5 2 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_family_egg_count_l1329_132961


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1329_132994

theorem linear_equation_solution : ∀ (x y : ℝ), x = 3 ∧ y = -2 → 2 * x + 3 * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1329_132994


namespace NUMINAMATH_CALUDE_f_properties_l1329_132942

noncomputable def f (x : ℝ) := 2 * abs (Real.sin x + Real.cos x) - Real.sin (2 * x)

theorem f_properties :
  (∀ x, f (π / 2 - x) = f x) ∧
  (∀ x, f x ≥ 1) ∧
  (∀ x y, π / 4 ≤ x ∧ x ≤ y ∧ y ≤ π / 2 → f x ≤ f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1329_132942


namespace NUMINAMATH_CALUDE_coplanar_points_l1329_132941

/-- The points (0,0,0), (1,a,0), (0,1,a), and (a,0,1) are coplanar if and only if a = -1 -/
theorem coplanar_points (a : ℝ) : 
  (Matrix.det
    ![![1, 0, a],
      ![a, 1, 0],
      ![0, a, 1]] = 0) ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_coplanar_points_l1329_132941


namespace NUMINAMATH_CALUDE_abc_sum_theorem_l1329_132912

theorem abc_sum_theorem (a b c : ℚ) (h : a * b * c > 0) :
  (|a| / a + |b| / b + |c| / c : ℚ) = 3 ∨ (|a| / a + |b| / b + |c| / c : ℚ) = -1 :=
by sorry

end NUMINAMATH_CALUDE_abc_sum_theorem_l1329_132912


namespace NUMINAMATH_CALUDE_train_distance_problem_l1329_132933

theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 20) (h2 : v2 = 25) (h3 : d = 65) :
  let t := d / (v1 + v2)
  let d1 := v1 * t
  let d2 := v2 * t
  d1 + d2 = 585 := by
sorry

end NUMINAMATH_CALUDE_train_distance_problem_l1329_132933


namespace NUMINAMATH_CALUDE_problem_statement_l1329_132966

theorem problem_statement (m n : ℤ) (h : 2*m - 3*n = 7) : 8 - 2*m + 3*n = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1329_132966


namespace NUMINAMATH_CALUDE_triangle_property_l1329_132975

theorem triangle_property (A B C : Real) (a b c R : Real) :
  0 < B → B < π / 2 →
  2 * R - b = 2 * b * Real.sin B →
  a = Real.sqrt 3 →
  c = 3 →
  B = π / 6 ∧ Real.sin C = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l1329_132975


namespace NUMINAMATH_CALUDE_fraction_factorization_l1329_132985

theorem fraction_factorization (a b c : ℝ) : 
  ((a^3 - b^3)^4 + (b^3 - c^3)^4 + (c^3 - a^3)^4) / ((a - b)^4 + (b - c)^4 + (c - a)^4)
  = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_factorization_l1329_132985


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l1329_132928

theorem triangle_perimeter_bound : 
  ∀ s : ℝ, 
  s > 0 → 
  s + 6 > 21 → 
  s + 21 > 6 → 
  6 + 21 > s → 
  54 > 6 + 21 + s ∧ 
  ∀ n : ℕ, n < 54 → ∃ t : ℝ, t > 0 ∧ t + 6 > 21 ∧ t + 21 > 6 ∧ 6 + 21 > t ∧ n ≤ 6 + 21 + t :=
by sorry

#check triangle_perimeter_bound

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l1329_132928


namespace NUMINAMATH_CALUDE_skateboarder_speed_l1329_132906

/-- Proves that a skateboarder traveling 660 feet in 30 seconds is moving at a speed of 15 miles per hour, given that 1 mile equals 5280 feet. -/
theorem skateboarder_speed (distance : ℝ) (time : ℝ) (feet_per_mile : ℝ) 
  (h1 : distance = 660)
  (h2 : time = 30)
  (h3 : feet_per_mile = 5280) : 
  (distance / time) * (3600 / feet_per_mile) = 15 := by
  sorry

#check skateboarder_speed

end NUMINAMATH_CALUDE_skateboarder_speed_l1329_132906


namespace NUMINAMATH_CALUDE_function_properties_imply_specific_form_and_result_l1329_132932

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_properties_imply_specific_form_and_result 
  (ω φ : ℝ) (h_ω : ω > 0) (h_φ : 0 < φ ∧ φ < Real.pi) :
  (∀ x : ℝ, f ω φ (x + Real.pi / 2) = f ω φ (Real.pi / 2 - x)) →
  (∀ x : ℝ, ∃ k : ℤ, f ω φ (x + Real.pi / (2 * ω)) = f ω φ (x + k * Real.pi / ω)) →
  (∃ α : ℝ, 0 < α ∧ α < Real.pi / 2 ∧ f ω φ (α / 2 + Real.pi / 12) = 3 / 5) →
  (∀ x : ℝ, f ω φ x = Real.cos (2 * x)) ∧
  (∀ α : ℝ, 0 < α → α < Real.pi / 2 → f ω φ (α / 2 + Real.pi / 12) = 3 / 5 → 
    Real.sin (2 * α) = (24 + 7 * Real.sqrt 3) / 50) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_imply_specific_form_and_result_l1329_132932


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1329_132920

theorem fixed_point_on_line (t : ℝ) : 
  (t + 1) * (-4) - (2 * t + 5) * (-2) - 6 = 0 := by
  sorry

#check fixed_point_on_line

end NUMINAMATH_CALUDE_fixed_point_on_line_l1329_132920


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1329_132970

/-- Given a line L1 with equation 2x + 3y - 6 = 0 and a point P (0, -3),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation 3x - 2y - 6 = 0 -/
theorem perpendicular_line_equation :
  let L1 : Set (ℝ × ℝ) := {(x, y) | 2 * x + 3 * y - 6 = 0}
  let P : ℝ × ℝ := (0, -3)
  let L2 : Set (ℝ × ℝ) := {(x, y) | 3 * x - 2 * y - 6 = 0}
  (∀ (x y : ℝ), (x, y) ∈ L2 ↔ 3 * x - 2 * y - 6 = 0) ∧
  P ∈ L2 ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ L1 → (x₂, y₂) ∈ L1 → x₁ ≠ x₂ →
    ((x₁ - x₂) * (x - 0) + (y₁ - y₂) * (y + 3) = 0 ↔ (x, y) ∈ L2)) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1329_132970


namespace NUMINAMATH_CALUDE_soccer_team_combinations_l1329_132997

def soccer_team_size : ℕ := 16
def quadruplets_count : ℕ := 4
def starting_lineup_size : ℕ := 7
def max_quadruplets_in_lineup : ℕ := 2

theorem soccer_team_combinations :
  (Nat.choose (soccer_team_size - quadruplets_count) starting_lineup_size) +
  (quadruplets_count * Nat.choose (soccer_team_size - quadruplets_count) (starting_lineup_size - 1)) +
  (Nat.choose quadruplets_count 2 * Nat.choose (soccer_team_size - quadruplets_count) (starting_lineup_size - 2)) = 9240 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_combinations_l1329_132997


namespace NUMINAMATH_CALUDE_ryn_to_nikki_ratio_l1329_132909

/-- The lengths of favorite movies for Joyce, Michael, Nikki, and Ryn -/
structure MovieLengths where
  michael : ℝ
  joyce : ℝ
  nikki : ℝ
  ryn : ℝ

/-- The conditions of the movie lengths problem -/
def movie_conditions (m : MovieLengths) : Prop :=
  m.joyce = m.michael + 2 ∧
  m.nikki = 3 * m.michael ∧
  m.nikki = 30 ∧
  m.michael + m.joyce + m.nikki + m.ryn = 76

/-- The theorem stating the ratio of Ryn's movie length to Nikki's movie length -/
theorem ryn_to_nikki_ratio (m : MovieLengths) :
  movie_conditions m → m.ryn / m.nikki = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ryn_to_nikki_ratio_l1329_132909


namespace NUMINAMATH_CALUDE_flour_calculation_l1329_132949

/-- Calculates the required cups of flour given the original recipe ratio, scaling factor, and amount of butter used. -/
def required_flour (original_butter original_flour scaling_factor butter_used : ℚ) : ℚ :=
  (butter_used / original_butter) * scaling_factor * original_flour

/-- Proves that given the specified conditions, the required amount of flour is 30 cups. -/
theorem flour_calculation (original_butter original_flour scaling_factor butter_used : ℚ) 
  (h1 : original_butter = 2)
  (h2 : original_flour = 5)
  (h3 : scaling_factor = 4)
  (h4 : butter_used = 12) :
  required_flour original_butter original_flour scaling_factor butter_used = 30 := by
sorry

#eval required_flour 2 5 4 12

end NUMINAMATH_CALUDE_flour_calculation_l1329_132949


namespace NUMINAMATH_CALUDE_distributive_analogy_l1329_132908

theorem distributive_analogy (a b c : ℝ) (h : c ≠ 0) :
  (a + b) * c = a * c + b * c ↔ (a + b) / c = a / c + b / c :=
sorry

end NUMINAMATH_CALUDE_distributive_analogy_l1329_132908


namespace NUMINAMATH_CALUDE_refrigerator_price_correct_l1329_132947

/-- The purchase price of the refrigerator that satisfies the given conditions -/
def refrigerator_price : ℝ :=
  let mobile_price : ℝ := 8000
  let refrigerator_loss_rate : ℝ := 0.05
  let mobile_profit_rate : ℝ := 0.10
  let total_profit : ℝ := 50
  15000

/-- Theorem stating that the refrigerator price satisfies the given conditions -/
theorem refrigerator_price_correct : 
  let mobile_price : ℝ := 8000
  let refrigerator_loss_rate : ℝ := 0.05
  let mobile_profit_rate : ℝ := 0.10
  let total_profit : ℝ := 50
  let refrigerator_sell_price := refrigerator_price * (1 - refrigerator_loss_rate)
  let mobile_sell_price := mobile_price * (1 + mobile_profit_rate)
  refrigerator_sell_price + mobile_sell_price = refrigerator_price + mobile_price + total_profit :=
by
  sorry

#eval refrigerator_price

end NUMINAMATH_CALUDE_refrigerator_price_correct_l1329_132947


namespace NUMINAMATH_CALUDE_pentagon_side_length_l1329_132943

/-- Given a triangle with all sides of length 20/9 cm and a pentagon with the same perimeter
    and all sides of equal length, the length of one side of the pentagon is 4/3 cm. -/
theorem pentagon_side_length (triangle_side : ℚ) (pentagon_side : ℚ) :
  triangle_side = 20 / 9 →
  3 * triangle_side = 5 * pentagon_side →
  pentagon_side = 4 / 3 := by
  sorry

#eval (4 : ℚ) / 3  -- Expected output: 4/3

end NUMINAMATH_CALUDE_pentagon_side_length_l1329_132943


namespace NUMINAMATH_CALUDE_cube_vector_sum_divisible_by_11_l1329_132922

/-- The size of the cube. -/
def cubeSize : ℕ := 1000

/-- The sum of squares of integers from 0 to n. -/
def sumOfSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The sum of squares of lengths of vectors from origin to all integer points in the cube. -/
def sumOfVectorLengthSquares : ℕ :=
  3 * (cubeSize + 1)^2 * sumOfSquares cubeSize

theorem cube_vector_sum_divisible_by_11 :
  sumOfVectorLengthSquares % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_vector_sum_divisible_by_11_l1329_132922


namespace NUMINAMATH_CALUDE_docked_amount_is_five_l1329_132914

/-- Calculates the amount docked per late arrival given the hourly rate, weekly hours, 
    number of late arrivals, and actual pay. -/
def amount_docked_per_late_arrival (hourly_rate : ℚ) (weekly_hours : ℚ) 
  (late_arrivals : ℕ) (actual_pay : ℚ) : ℚ :=
  ((hourly_rate * weekly_hours) - actual_pay) / late_arrivals

/-- Proves that the amount docked per late arrival is $5 given the specific conditions. -/
theorem docked_amount_is_five :
  amount_docked_per_late_arrival 30 18 3 525 = 5 := by
  sorry

end NUMINAMATH_CALUDE_docked_amount_is_five_l1329_132914


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l1329_132917

/-- Simple interest calculation -/
theorem simple_interest_calculation (principal interest_rate simple_interest : ℚ) : 
  principal = 8 →
  interest_rate = 5 / 100 →
  simple_interest = 4.8 →
  ∃ (months : ℚ), months = 12 ∧ simple_interest = principal * interest_rate * months :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l1329_132917


namespace NUMINAMATH_CALUDE_shaded_squares_percentage_l1329_132971

/-- Given a 5x5 grid with 9 shaded squares, the percentage of shaded squares is 36%. -/
theorem shaded_squares_percentage :
  ∀ (total_squares shaded_squares : ℕ),
    total_squares = 5 * 5 →
    shaded_squares = 9 →
    (shaded_squares : ℚ) / total_squares * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_shaded_squares_percentage_l1329_132971


namespace NUMINAMATH_CALUDE_remainder_of_repeated_12_l1329_132953

def repeated_digit_number (n : ℕ) : ℕ := 
  -- Function to generate the number with n repetitions of "12"
  -- Implementation details omitted for brevity
  sorry

theorem remainder_of_repeated_12 (n : ℕ) :
  repeated_digit_number 150 % 99 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_repeated_12_l1329_132953


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1329_132998

/-- A polynomial in x and y with a parameter n -/
def polynomial (n : ℤ) (x y : ℤ) : ℤ := x^2 + 4*x*y + 2*x + n*y - n

/-- Predicate to check if a polynomial can be factored into two linear factors with integer coefficients -/
def has_linear_factors (p : ℤ → ℤ → ℤ) : Prop :=
  ∃ (a b c d e f : ℤ), ∀ x y, p x y = (a*x + b*y + c) * (d*x + e*y + f)

theorem polynomial_factorization (n : ℤ) :
  has_linear_factors (polynomial n) ↔ n = 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1329_132998


namespace NUMINAMATH_CALUDE_max_fourth_power_sum_l1329_132930

theorem max_fourth_power_sum (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 4) :
  ∃ (m : ℝ), (∀ x y z w : ℝ, x^3 + y^3 + z^3 + w^3 = 4 → x^4 + y^4 + z^4 + w^4 ≤ m) ∧
             (a^4 + b^4 + c^4 + d^4 = m) ∧
             m = 16 :=
sorry

end NUMINAMATH_CALUDE_max_fourth_power_sum_l1329_132930


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1329_132987

theorem rationalize_denominator :
  ∃ (A B C D E F : ℤ),
    (F > 0) ∧
    (1 / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) =
     (A * Real.sqrt 3 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F) ∧
    A = -13 ∧ B = -9 ∧ C = 3 ∧ D = 2 ∧ E = 165 ∧ F = 51 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1329_132987


namespace NUMINAMATH_CALUDE_parabola_directrix_l1329_132926

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop :=
  y = (x^2 - 4*x + 4) / 8

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop :=
  y = -1/4

/-- Theorem: The directrix of the given parabola is y = -1/4 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_equation x y → ∃ y_d : ℝ, directrix_equation y_d :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1329_132926
