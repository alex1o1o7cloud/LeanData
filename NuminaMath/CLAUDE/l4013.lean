import Mathlib

namespace NUMINAMATH_CALUDE_proposition_equivalence_l4013_401323

theorem proposition_equivalence (p q : Prop) : (p ∨ q) ↔ ¬(¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l4013_401323


namespace NUMINAMATH_CALUDE_weight_of_a_l4013_401318

theorem weight_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 50 →
  (a + b + c + d) / 4 = 53 →
  (b + c + d + e) / 4 = 51 →
  e = d + 3 →
  a = 73 := by
sorry

end NUMINAMATH_CALUDE_weight_of_a_l4013_401318


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l4013_401326

theorem hyperbola_foci_distance : 
  ∀ (x y : ℝ), 
  (y^2 / 75) - (x^2 / 11) = 1 →
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 4 * 86 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l4013_401326


namespace NUMINAMATH_CALUDE_range_of_m_l4013_401305

theorem range_of_m (m : ℝ) : 
  (m + 4)^(-1/2 : ℝ) < (3 - 2*m)^(-1/2 : ℝ) → 
  -1/3 < m ∧ m < 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l4013_401305


namespace NUMINAMATH_CALUDE_quadratic_roots_equality_l4013_401324

theorem quadratic_roots_equality (α β γ p q : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + p*x₁ + q = 0) → 
  (x₂^2 + p*x₂ + q = 0) → 
  (α * x₁^2 + β * x₁ + γ = α * x₂^2 + β * x₂ + γ) ↔ 
  (p^2 = 4*q ∨ p = -β/α) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_equality_l4013_401324


namespace NUMINAMATH_CALUDE_multiply_121_54_l4013_401396

theorem multiply_121_54 : 121 * 54 = 6534 := by
  sorry

end NUMINAMATH_CALUDE_multiply_121_54_l4013_401396


namespace NUMINAMATH_CALUDE_traveler_time_difference_l4013_401360

/-- Proof of the time difference between two travelers meeting at a point -/
theorem traveler_time_difference 
  (speed_A speed_B meeting_distance : ℝ) 
  (h1 : speed_A > 0)
  (h2 : speed_B > speed_A)
  (h3 : meeting_distance > 0) :
  meeting_distance / speed_A - meeting_distance / speed_B = 7 :=
by sorry

end NUMINAMATH_CALUDE_traveler_time_difference_l4013_401360


namespace NUMINAMATH_CALUDE_first_hour_coins_is_20_l4013_401384

/-- The number of coins Tina put in the jar during the first hour -/
def first_hour_coins : ℕ := sorry

/-- The number of coins Tina put in the jar during the second hour -/
def second_hour_coins : ℕ := 30

/-- The number of coins Tina put in the jar during the third hour -/
def third_hour_coins : ℕ := 30

/-- The number of coins Tina put in the jar during the fourth hour -/
def fourth_hour_coins : ℕ := 40

/-- The number of coins Tina took out of the jar during the fifth hour -/
def fifth_hour_coins : ℕ := 20

/-- The total number of coins in the jar after the fifth hour -/
def total_coins : ℕ := 100

/-- Theorem stating that the number of coins Tina put in during the first hour is 20 -/
theorem first_hour_coins_is_20 :
  first_hour_coins = 20 :=
by
  have h : first_hour_coins + second_hour_coins + third_hour_coins + fourth_hour_coins - fifth_hour_coins = total_coins := sorry
  sorry


end NUMINAMATH_CALUDE_first_hour_coins_is_20_l4013_401384


namespace NUMINAMATH_CALUDE_mans_rate_l4013_401325

def with_stream : ℝ := 25
def against_stream : ℝ := 13

theorem mans_rate (with_stream against_stream : ℝ) :
  with_stream = 25 →
  against_stream = 13 →
  (with_stream + against_stream) / 2 = 19 := by
sorry

end NUMINAMATH_CALUDE_mans_rate_l4013_401325


namespace NUMINAMATH_CALUDE_octal_53_to_decimal_l4013_401338

/-- Converts an octal digit to its decimal value -/
def octal_to_decimal (d : ℕ) : ℕ := d

/-- Converts a two-digit octal number to its decimal equivalent -/
def octal_2digit_to_decimal (d1 d0 : ℕ) : ℕ :=
  octal_to_decimal d1 * 8 + octal_to_decimal d0

/-- The decimal representation of the octal number 53 is 43 -/
theorem octal_53_to_decimal :
  octal_2digit_to_decimal 5 3 = 43 := by sorry

end NUMINAMATH_CALUDE_octal_53_to_decimal_l4013_401338


namespace NUMINAMATH_CALUDE_calvins_bug_collection_l4013_401373

theorem calvins_bug_collection (roaches scorpions caterpillars crickets : ℕ) : 
  roaches = 12 →
  scorpions = 3 →
  caterpillars = 2 * scorpions →
  roaches + scorpions + caterpillars + crickets = 27 →
  crickets * 2 = roaches :=
by sorry

end NUMINAMATH_CALUDE_calvins_bug_collection_l4013_401373


namespace NUMINAMATH_CALUDE_digits_of_s_1000_l4013_401362

/-- s(n) is an n-digit number formed by attaching the first n perfect squares in order -/
def s (n : ℕ) : ℕ := sorry

/-- The number of digits in a natural number -/
def num_digits (m : ℕ) : ℕ := sorry

/-- Theorem: The number of digits in s(1000) is 2893 -/
theorem digits_of_s_1000 : num_digits (s 1000) = 2893 := by sorry

end NUMINAMATH_CALUDE_digits_of_s_1000_l4013_401362


namespace NUMINAMATH_CALUDE_xy_value_l4013_401383

theorem xy_value (x y : ℝ) (h : x / 2 + 2 * y - 2 = Real.log x + Real.log y) : 
  x ^ y = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l4013_401383


namespace NUMINAMATH_CALUDE_original_paint_intensity_l4013_401334

-- Define the paint mixing problem
def paint_mixing (original_intensity : ℝ) : Prop :=
  let f : ℝ := 1/3  -- fraction of original paint replaced
  let replacement_intensity : ℝ := 20  -- 20% solution
  let final_intensity : ℝ := 40  -- 40% final intensity
  (1 - f) * original_intensity + f * replacement_intensity = final_intensity

-- Theorem statement
theorem original_paint_intensity :
  ∃ (original_intensity : ℝ), paint_mixing original_intensity ∧ original_intensity = 50 := by
  sorry

end NUMINAMATH_CALUDE_original_paint_intensity_l4013_401334


namespace NUMINAMATH_CALUDE_custom_mult_example_l4013_401352

/-- Custom multiplication operation for rational numbers -/
def custom_mult (a b : ℚ) : ℚ := a * b + b ^ 2

/-- Theorem stating that 4 * (-2) = -4 using the custom multiplication -/
theorem custom_mult_example : custom_mult 4 (-2) = -4 := by sorry

end NUMINAMATH_CALUDE_custom_mult_example_l4013_401352


namespace NUMINAMATH_CALUDE_A_union_B_eq_real_l4013_401399

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x > 0}

-- Define set B
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem A_union_B_eq_real : A ∪ B = Set.univ := by sorry

end NUMINAMATH_CALUDE_A_union_B_eq_real_l4013_401399


namespace NUMINAMATH_CALUDE_sqrt_neg_five_squared_l4013_401372

theorem sqrt_neg_five_squared : Real.sqrt ((-5)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_five_squared_l4013_401372


namespace NUMINAMATH_CALUDE_constant_term_equals_96_l4013_401369

/-- The constant term in the expansion of (2x + a/x)^4 -/
def constantTerm (a : ℝ) : ℝ := a^2 * 2^2 * 6

theorem constant_term_equals_96 (a : ℝ) (h : a > 0) : 
  constantTerm a = 96 → a = 2 := by sorry

end NUMINAMATH_CALUDE_constant_term_equals_96_l4013_401369


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l4013_401344

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem largest_prime_divisor_of_factorial_sum :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (factorial 12 + factorial 13) ∧
    ∀ (q : ℕ), Nat.Prime q → q ∣ (factorial 12 + factorial 13) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l4013_401344


namespace NUMINAMATH_CALUDE_mary_sugar_added_l4013_401337

/-- Given a recipe that requires a certain amount of sugar and the amount of sugar still needed,
    calculate the amount of sugar already added. -/
def sugar_already_added (recipe_required : ℕ) (sugar_needed : ℕ) : ℕ :=
  recipe_required - sugar_needed

/-- Theorem stating that Mary has already added 10 cups of sugar. -/
theorem mary_sugar_added :
  sugar_already_added 11 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_sugar_added_l4013_401337


namespace NUMINAMATH_CALUDE_solve_for_C_l4013_401313

theorem solve_for_C : ∃ C : ℝ, (4 * C + 5 = 37) ∧ (C = 8) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_C_l4013_401313


namespace NUMINAMATH_CALUDE_sequence_properties_l4013_401359

def a : ℕ → ℕ
  | n => if n % 2 = 1 then n else 2 * 3^((n / 2) - 1)

def S (n : ℕ) : ℕ := (List.range n).map a |>.sum

theorem sequence_properties :
  (∀ k : ℕ, a (2 * k + 1) = a 1 + k * (a 3 - a 1)) ∧
  (∀ k : ℕ, k > 0 → a (2 * k) = a 2 * (a 4 / a 2) ^ (k - 1)) ∧
  (S 5 = 2 * a 4 + a 5) ∧
  (a 9 = a 3 + a 4) →
  (∀ n : ℕ, n > 0 → a n = if n % 2 = 1 then n else 2 * 3^((n / 2) - 1)) ∧
  (∀ m : ℕ, m > 0 → (a m * a (m + 1) = a (m + 2)) ↔ m = 2) ∧
  (∀ m : ℕ, m > 0 → (∃ k : ℕ, k > 0 ∧ S (2 * m) / S (2 * m - 1) = a k) ↔ (m = 1 ∨ m = 2)) := by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_sequence_properties_l4013_401359


namespace NUMINAMATH_CALUDE_brass_players_count_l4013_401306

/-- Represents the composition of a marching band -/
structure MarchingBand where
  brass : ℕ
  woodwind : ℕ
  percussion : ℕ

/-- The total number of members in the marching band -/
def MarchingBand.total (band : MarchingBand) : ℕ :=
  band.brass + band.woodwind + band.percussion

/-- Theorem: The number of brass players in the marching band is 10 -/
theorem brass_players_count (band : MarchingBand) :
  band.total = 110 →
  band.woodwind = 2 * band.brass →
  band.percussion = 4 * band.woodwind →
  band.brass = 10 := by
  sorry

end NUMINAMATH_CALUDE_brass_players_count_l4013_401306


namespace NUMINAMATH_CALUDE_polynomial_remainder_l4013_401390

theorem polynomial_remainder (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^4 - 3*x^3 + 2*x^2 + 11*x - 6
  (f x) % (x - 2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l4013_401390


namespace NUMINAMATH_CALUDE_problem_solution_l4013_401330

theorem problem_solution (A B : ℝ) (hB : B ≠ 0) : 
  let f : ℝ → ℝ := λ x ↦ A * x^2 - 3 * B^2
  let g : ℝ → ℝ := λ x ↦ B * x^2
  f (g 1) = 0 → A = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4013_401330


namespace NUMINAMATH_CALUDE_weight_difference_l4013_401339

/-- Given the weights of three people (Ishmael, Ponce, and Jalen), prove that Ishmael is 20 pounds heavier than Ponce. -/
theorem weight_difference (I P J : ℝ) : 
  J = 160 →  -- Jalen's weight
  P = J - 10 →  -- Ponce is 10 pounds lighter than Jalen
  (I + P + J) / 3 = 160 →  -- Average weight is 160 pounds
  I - P = 20 :=  -- Ishmael is 20 pounds heavier than Ponce
by sorry

end NUMINAMATH_CALUDE_weight_difference_l4013_401339


namespace NUMINAMATH_CALUDE_zoe_pop_albums_l4013_401317

/-- Represents the number of songs in each album -/
def songs_per_album : ℕ := 3

/-- Represents the number of country albums bought -/
def country_albums : ℕ := 3

/-- Represents the total number of songs bought -/
def total_songs : ℕ := 24

/-- Calculates the number of pop albums bought -/
def pop_albums : ℕ := (total_songs - country_albums * songs_per_album) / songs_per_album

theorem zoe_pop_albums : pop_albums = 5 := by
  sorry

end NUMINAMATH_CALUDE_zoe_pop_albums_l4013_401317


namespace NUMINAMATH_CALUDE_inequality_proof_l4013_401364

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  (((1 / a + 6 * b) ^ (1/3 : ℝ)) + ((1 / b + 6 * c) ^ (1/3 : ℝ)) + ((1 / c + 6 * a) ^ (1/3 : ℝ))) ≤ 1 / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4013_401364


namespace NUMINAMATH_CALUDE_unique_number_product_sum_digits_l4013_401366

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The theorem stating that 251 is the only number satisfying the condition -/
theorem unique_number_product_sum_digits : 
  ∃! n : ℕ, n * sum_of_digits n = 2008 ∧ n > 0 := by sorry

end NUMINAMATH_CALUDE_unique_number_product_sum_digits_l4013_401366


namespace NUMINAMATH_CALUDE_max_blocks_fit_l4013_401301

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The large box dimensions -/
def largeBox : BoxDimensions := ⟨6, 3, 4⟩

/-- The small block dimensions -/
def smallBlock : BoxDimensions := ⟨3, 1, 2⟩

/-- Calculates the volume of a box given its dimensions -/
def volume (box : BoxDimensions) : ℕ :=
  box.length * box.width * box.height

/-- Theorem: The maximum number of small blocks that can fit in the large box is 12 -/
theorem max_blocks_fit : 
  (volume largeBox) / (volume smallBlock) = 12 ∧ 
  largeBox.length / smallBlock.length * 
  largeBox.width / smallBlock.width * 
  largeBox.height / smallBlock.height = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l4013_401301


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4013_401312

theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (1 - i) / (1 + i) = -i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4013_401312


namespace NUMINAMATH_CALUDE_odd_mult_odd_is_odd_l4013_401361

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def P : Set ℕ := {n : ℕ | is_odd n}

theorem odd_mult_odd_is_odd (a b : ℕ) (ha : a ∈ P) (hb : b ∈ P) : a * b ∈ P := by
  sorry

end NUMINAMATH_CALUDE_odd_mult_odd_is_odd_l4013_401361


namespace NUMINAMATH_CALUDE_second_discount_percentage_l4013_401385

def initial_price : ℝ := 400
def first_discount : ℝ := 20
def final_price : ℝ := 272

theorem second_discount_percentage :
  ∃ (second_discount : ℝ),
    initial_price * (1 - first_discount / 100) * (1 - second_discount / 100) = final_price ∧
    second_discount = 15 := by
  sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l4013_401385


namespace NUMINAMATH_CALUDE_total_heads_calculation_l4013_401389

theorem total_heads_calculation (num_hens : ℕ) (total_feet : ℕ) : 
  num_hens = 24 → total_feet = 136 → ∃ (num_cows : ℕ), num_hens + num_cows = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_heads_calculation_l4013_401389


namespace NUMINAMATH_CALUDE_cookie_problem_l4013_401321

theorem cookie_problem (glenn kenny chris : ℕ) : 
  glenn = 24 →
  glenn = 4 * kenny →
  chris = kenny / 2 →
  glenn + kenny + chris = 33 :=
by sorry

end NUMINAMATH_CALUDE_cookie_problem_l4013_401321


namespace NUMINAMATH_CALUDE_number_of_nickels_l4013_401374

def pennies : ℕ := 123
def dimes : ℕ := 35
def quarters : ℕ := 26
def family_members : ℕ := 5
def ice_cream_cost_per_member : ℚ := 3
def leftover_cents : ℕ := 48

def total_ice_cream_cost : ℚ := family_members * ice_cream_cost_per_member

def total_without_nickels : ℚ := 
  (pennies : ℚ) / 100 + (dimes : ℚ) / 10 + (quarters : ℚ) / 4

theorem number_of_nickels : 
  ∃ (n : ℕ), total_without_nickels + (n : ℚ) / 20 = total_ice_cream_cost + (leftover_cents : ℚ) / 100 ∧ n = 85 := by
  sorry

end NUMINAMATH_CALUDE_number_of_nickels_l4013_401374


namespace NUMINAMATH_CALUDE_decrease_six_l4013_401331

def temperature_change : ℝ → ℝ := id

axiom positive_rise (x : ℝ) : x > 0 → temperature_change x > 0

axiom rise_three : temperature_change 3 = 3

theorem decrease_six : temperature_change (-6) = -6 := by sorry

end NUMINAMATH_CALUDE_decrease_six_l4013_401331


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l4013_401354

theorem consecutive_integers_product (n : ℤ) : 
  n * (n + 1) * (n + 2) * (n + 3) = 2520 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l4013_401354


namespace NUMINAMATH_CALUDE_discounted_shoe_price_l4013_401347

/-- Given a pair of shoes bought at a 20% discount for $480, 
    prove that the original price was $600. -/
theorem discounted_shoe_price (discount_rate : ℝ) (discounted_price : ℝ) :
  discount_rate = 0.20 →
  discounted_price = 480 →
  discounted_price = (1 - discount_rate) * 600 :=
by sorry

end NUMINAMATH_CALUDE_discounted_shoe_price_l4013_401347


namespace NUMINAMATH_CALUDE_power_inequality_l4013_401308

theorem power_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a^5 + b^5 > a^2 * b^3 + a^3 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l4013_401308


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l4013_401302

theorem sum_of_special_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x * y = 50 * (x + y)) (h2 : x * y = 75 * (x - y)) :
  x + y = 360 := by
sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l4013_401302


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l4013_401398

theorem divisible_by_eleven (n : ℤ) : (18888 - n) % 11 = 0 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l4013_401398


namespace NUMINAMATH_CALUDE_least_integer_with_12_factors_l4013_401381

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Theorem: 72 is the least positive integer with exactly 12 positive factors -/
theorem least_integer_with_12_factors :
  (∀ m : ℕ+, m < 72 → num_factors m ≠ 12) ∧ num_factors 72 = 12 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_with_12_factors_l4013_401381


namespace NUMINAMATH_CALUDE_browser_tabs_remaining_l4013_401356

theorem browser_tabs_remaining (initial_tabs : ℕ) : 
  initial_tabs = 400 → 
  (initial_tabs - initial_tabs / 4 - (initial_tabs - initial_tabs / 4) * 2 / 5) / 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_browser_tabs_remaining_l4013_401356


namespace NUMINAMATH_CALUDE_cube_sum_problem_l4013_401349

theorem cube_sum_problem (p q r : ℝ) 
  (h1 : p + q + r = 5)
  (h2 : p * q + p * r + q * r = 7)
  (h3 : p * q * r = -10) :
  p^3 + q^3 + r^3 = -10 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l4013_401349


namespace NUMINAMATH_CALUDE_multiply_three_point_five_by_zero_point_twenty_five_l4013_401397

theorem multiply_three_point_five_by_zero_point_twenty_five : 3.5 * 0.25 = 0.875 := by
  sorry

end NUMINAMATH_CALUDE_multiply_three_point_five_by_zero_point_twenty_five_l4013_401397


namespace NUMINAMATH_CALUDE_line_x_intercept_l4013_401391

/-- Given a line passing through points (2, -2) and (6, 6), its x-intercept is 3 -/
theorem line_x_intercept : 
  ∀ (f : ℝ → ℝ), 
  (f 2 = -2) → 
  (f 6 = 6) → 
  (∀ x y : ℝ, f y - f x = (y - x) * ((6 - (-2)) / (6 - 2))) →
  (∃ x : ℝ, f x = 0 ∧ x = 3) := by
sorry

end NUMINAMATH_CALUDE_line_x_intercept_l4013_401391


namespace NUMINAMATH_CALUDE_maria_friends_money_l4013_401327

/-- The amount of money Maria gave to her three friends -/
def total_given (maria_money : ℝ) (isha_share : ℝ) (florence_share : ℝ) (rene_share : ℝ) : ℝ :=
  isha_share + florence_share + rene_share

/-- Theorem stating the total amount Maria gave to her friends -/
theorem maria_friends_money :
  ∀ (maria_money : ℝ),
  maria_money > 0 →
  let isha_share := (1/3) * maria_money
  let florence_share := (1/2) * isha_share
  let rene_share := 300
  florence_share = 3 * rene_share →
  total_given maria_money isha_share florence_share rene_share = 3000 := by
  sorry

end NUMINAMATH_CALUDE_maria_friends_money_l4013_401327


namespace NUMINAMATH_CALUDE_book_distribution_l4013_401332

theorem book_distribution (x : ℕ) (total_books : ℕ) : 
  (9 * x + 7 ≤ total_books) ∧ (total_books < 11 * x) →
  (9 * x + 7 = total_books) :=
by sorry

end NUMINAMATH_CALUDE_book_distribution_l4013_401332


namespace NUMINAMATH_CALUDE_square_roots_equality_l4013_401371

theorem square_roots_equality (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (2*a + 1)^2 = x ∧ (a + 5)^2 = x) → a = 4 := by
sorry

end NUMINAMATH_CALUDE_square_roots_equality_l4013_401371


namespace NUMINAMATH_CALUDE_intersection_M_N_l4013_401395

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 + x ≤ 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4013_401395


namespace NUMINAMATH_CALUDE_fathers_age_l4013_401394

theorem fathers_age (M F : ℕ) : 
  M = (2 : ℕ) * F / (5 : ℕ) →
  M + 6 = (F + 6) / (2 : ℕ) →
  F = 30 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l4013_401394


namespace NUMINAMATH_CALUDE_balls_sold_count_l4013_401380

def selling_price : ℕ := 720
def cost_price_per_ball : ℕ := 72
def loss : ℕ := 5 * cost_price_per_ball

theorem balls_sold_count :
  ∃ n : ℕ, n * cost_price_per_ball - selling_price = loss ∧ n = 15 :=
by sorry

end NUMINAMATH_CALUDE_balls_sold_count_l4013_401380


namespace NUMINAMATH_CALUDE_cosine_sine_sum_identity_l4013_401348

theorem cosine_sine_sum_identity : 
  Real.cos (43 * π / 180) * Real.cos (77 * π / 180) + 
  Real.sin (43 * π / 180) * Real.cos (167 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_identity_l4013_401348


namespace NUMINAMATH_CALUDE_three_books_purchase_ways_l4013_401322

/-- The number of ways to purchase books given the conditions -/
def purchase_ways (n : ℕ) : ℕ := 2^n - 1

/-- Theorem: There are 7 ways to purchase when there are 3 books -/
theorem three_books_purchase_ways :
  purchase_ways 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_three_books_purchase_ways_l4013_401322


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l4013_401309

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l4013_401309


namespace NUMINAMATH_CALUDE_smallest_square_area_for_rectangles_l4013_401316

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a square given its side length -/
def squareArea (side : ℕ) : ℕ := side * side

/-- Checks if two rectangles can fit side by side within a given width -/
def canFitSideBySide (r1 r2 : Rectangle) (width : ℕ) : Prop :=
  r1.width + r2.width ≤ width

/-- Checks if two rectangles can fit one above the other within a given height -/
def canFitStackedVertically (r1 r2 : Rectangle) (height : ℕ) : Prop :=
  r1.height + r2.height ≤ height

/-- The main theorem stating the smallest possible area of the square -/
theorem smallest_square_area_for_rectangles : 
  let r1 : Rectangle := ⟨2, 4⟩
  let r2 : Rectangle := ⟨3, 5⟩
  let minSideLength : ℕ := max (r1.width + r2.width) (r1.height + r2.height)
  ∃ (side : ℕ), 
    side ≥ minSideLength ∧
    canFitSideBySide r1 r2 side ∧
    canFitStackedVertically r1 r2 side ∧
    squareArea side = 81 ∧
    ∀ (s : ℕ), s < side → ¬(canFitSideBySide r1 r2 s ∧ canFitStackedVertically r1 r2 s) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_area_for_rectangles_l4013_401316


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l4013_401377

theorem quadratic_rewrite (d e f : ℤ) : 
  (∀ x : ℝ, 4 * x^2 + 20 * x - 24 = (d * x + e)^2 + f) → d * e = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l4013_401377


namespace NUMINAMATH_CALUDE_division_with_same_remainder_l4013_401343

theorem division_with_same_remainder (x : ℕ) (h1 : x > 0) (h2 : ∃ k : ℤ, 200 = k * x + 2) :
  ∀ n : ℤ, ∃ k : ℤ, 200 = k * x + 2 ∧ n ≠ k → ∃ m : ℤ, n * x + 2 = m * x + (n * x + 2) % x ∧ (n * x + 2) % x = 2 :=
by sorry

end NUMINAMATH_CALUDE_division_with_same_remainder_l4013_401343


namespace NUMINAMATH_CALUDE_sector_central_angle_l4013_401357

/-- Given a sector with radius R and area 2R^2, 
    the radian measure of its central angle is 4. -/
theorem sector_central_angle (R : ℝ) (h : R > 0) :
  let area := 2 * R^2
  let angle := (2 * area) / R^2
  angle = 4 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l4013_401357


namespace NUMINAMATH_CALUDE_sequence_properties_l4013_401345

def is_arithmetic_progression (s : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, s (n + 1) - s n = d

theorem sequence_properties
  (a b c : ℕ+ → ℝ)
  (h1 : ∀ n : ℕ+, b n = a n - 2 * a (n + 1))
  (h2 : ∀ n : ℕ+, c n = a (n + 1) + 2 * a (n + 2) - 2) :
  (is_arithmetic_progression a → is_arithmetic_progression b) ∧
  (is_arithmetic_progression b ∧ is_arithmetic_progression c →
    ∃ d : ℝ, ∀ n : ℕ+, n ≥ 2 → a (n + 1) - a n = d) ∧
  (is_arithmetic_progression b ∧ b 1 + a 3 = 0 → is_arithmetic_progression a) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l4013_401345


namespace NUMINAMATH_CALUDE_cost_of_hundred_nuggets_l4013_401311

/-- Calculates the total cost of chicken nuggets -/
def chicken_nugget_cost (total_nuggets : ℕ) (nuggets_per_box : ℕ) (cost_per_box : ℕ) : ℕ :=
  (total_nuggets / nuggets_per_box) * cost_per_box

/-- Theorem: The cost of 100 chicken nuggets is $20 -/
theorem cost_of_hundred_nuggets :
  chicken_nugget_cost 100 20 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_hundred_nuggets_l4013_401311


namespace NUMINAMATH_CALUDE_line_equation_l4013_401307

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 5

-- Define the line l
def line_l (x y : ℝ) : Prop := ∃ (m b : ℝ), y = m * x + b

-- Define the center of the circle
def center : ℝ × ℝ := (3, 5)

-- Define that line l passes through the center
def line_through_center (l : ℝ → ℝ → Prop) : Prop :=
  l center.1 center.2

-- Define points A and B on the circle and line
def point_on_circle_and_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  circle_C p.1 p.2 ∧ l p.1 p.2

-- Define point P on y-axis and line
def point_on_y_axis_and_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  p.1 = 0 ∧ l p.1 p.2

-- Define A as midpoint of BP
def A_midpoint_BP (A B P : ℝ × ℝ) : Prop :=
  A.1 = (B.1 + P.1) / 2 ∧ A.2 = (B.2 + P.2) / 2

-- Theorem statement
theorem line_equation :
  ∀ (A B P : ℝ × ℝ) (l : ℝ → ℝ → Prop),
    line_l = l →
    line_through_center l →
    point_on_circle_and_line A l →
    point_on_circle_and_line B l →
    point_on_y_axis_and_line P l →
    A_midpoint_BP A B P →
    (∀ x y, l x y ↔ (2*x - y - 1 = 0 ∨ 2*x + y - 11 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l4013_401307


namespace NUMINAMATH_CALUDE_waiter_customer_count_l4013_401336

/-- Represents the scenario of a waiter serving customers and receiving tips -/
structure WaiterScenario where
  total_customers : ℕ
  non_tipping_customers : ℕ
  tip_amount : ℕ
  total_tips : ℕ

/-- Theorem stating that given the conditions, the waiter had 7 customers in total -/
theorem waiter_customer_count (scenario : WaiterScenario) 
  (h1 : scenario.non_tipping_customers = 5)
  (h2 : scenario.tip_amount = 3)
  (h3 : scenario.total_tips = 6) :
  scenario.total_customers = 7 := by
  sorry


end NUMINAMATH_CALUDE_waiter_customer_count_l4013_401336


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4013_401328

theorem complex_equation_solution (z : ℂ) (i : ℂ) (h1 : i * i = -1) (h2 : z * (1 + i) = 2) : z = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4013_401328


namespace NUMINAMATH_CALUDE_negative_64_to_four_thirds_equals_256_l4013_401333

theorem negative_64_to_four_thirds_equals_256 : (-64 : ℝ) ^ (4/3) = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_64_to_four_thirds_equals_256_l4013_401333


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l4013_401370

theorem inequality_holds_iff (n k : ℕ) : 
  (∀ a b : ℝ, a ≥ 0 → b ≥ 0 → 
    a^k * b^k * (a^2 + b^2)^n ≤ (a + b)^(2*k + 2*n) / 2^(2*k + n)) ↔ 
  k ≥ n := by
sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l4013_401370


namespace NUMINAMATH_CALUDE_division_decimal_l4013_401351

theorem division_decimal : (0.24 : ℚ) / (0.006 : ℚ) = 40 := by sorry

end NUMINAMATH_CALUDE_division_decimal_l4013_401351


namespace NUMINAMATH_CALUDE_square_tiles_count_l4013_401320

theorem square_tiles_count (h s : ℕ) : 
  h + s = 30 →  -- Total number of tiles
  6 * h + 4 * s = 128 →  -- Total number of edges
  s = 26 :=  -- Number of square tiles
by
  sorry

end NUMINAMATH_CALUDE_square_tiles_count_l4013_401320


namespace NUMINAMATH_CALUDE_debby_yoyo_tickets_debby_yoyo_tickets_proof_l4013_401358

/-- Theorem: Debby's yoyo ticket expenditure --/
theorem debby_yoyo_tickets : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun hat_tickets stuffed_animal_tickets total_tickets yoyo_tickets =>
    hat_tickets = 2 ∧ 
    stuffed_animal_tickets = 10 ∧ 
    total_tickets = 14 ∧ 
    yoyo_tickets + hat_tickets + stuffed_animal_tickets = total_tickets →
    yoyo_tickets = 2

/-- Proof of the theorem --/
theorem debby_yoyo_tickets_proof : 
  debby_yoyo_tickets 2 10 14 2 := by
  sorry

end NUMINAMATH_CALUDE_debby_yoyo_tickets_debby_yoyo_tickets_proof_l4013_401358


namespace NUMINAMATH_CALUDE_months_with_average_salary_8900_l4013_401335

def average_salary_jan_to_apr : ℕ := 8000
def average_salary_some_months : ℕ := 8900
def salary_may : ℕ := 6500
def salary_jan : ℕ := 2900

theorem months_with_average_salary_8900 :
  let total_salary_jan_to_apr := average_salary_jan_to_apr * 4
  let total_salary_feb_to_apr := total_salary_jan_to_apr - salary_jan
  let total_salary_feb_to_may := total_salary_feb_to_apr + salary_may
  total_salary_feb_to_may / average_salary_some_months = 4 := by
sorry

end NUMINAMATH_CALUDE_months_with_average_salary_8900_l4013_401335


namespace NUMINAMATH_CALUDE_raj_ravi_age_difference_l4013_401310

/-- Represents the ages of individuals in the problem -/
structure Ages where
  raj : ℕ
  ravi : ℕ
  hema : ℕ
  rahul : ℕ

/-- Conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ∃ (x : ℕ),
    ages.raj = ages.ravi + x ∧
    ages.hema = ages.ravi - 2 ∧
    ages.raj = 3 * ages.rahul ∧
    ages.hema = (3 / 2 : ℚ) * ages.rahul ∧
    20 = ages.hema + (1 / 3 : ℚ) * ages.hema

/-- The theorem to be proved -/
theorem raj_ravi_age_difference (ages : Ages) :
  problem_conditions ages → ages.raj - ages.ravi = 13 :=
by sorry

end NUMINAMATH_CALUDE_raj_ravi_age_difference_l4013_401310


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l4013_401319

theorem quadratic_equation_real_roots (a : ℝ) : 
  ∃ x : ℝ, x^2 + a*x + (a - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l4013_401319


namespace NUMINAMATH_CALUDE_tiles_difference_8th_7th_l4013_401376

/-- The number of tiles in the n-th square of the sequence -/
def tiles_in_square (n : ℕ) : ℕ := n^2

/-- The theorem stating the difference in tiles between the 8th and 7th squares -/
theorem tiles_difference_8th_7th : 
  tiles_in_square 8 - tiles_in_square 7 = 15 := by
  sorry

end NUMINAMATH_CALUDE_tiles_difference_8th_7th_l4013_401376


namespace NUMINAMATH_CALUDE_bottle_cap_count_l4013_401388

/-- Given the total cost of bottle caps and the cost per bottle cap,
    prove that the number of bottle caps is correct. -/
theorem bottle_cap_count 
  (total_cost : ℝ) 
  (cost_per_cap : ℝ) 
  (h1 : total_cost = 25) 
  (h2 : cost_per_cap = 5) : 
  total_cost / cost_per_cap = 5 := by
  sorry

#check bottle_cap_count

end NUMINAMATH_CALUDE_bottle_cap_count_l4013_401388


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l4013_401367

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 :=
sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l4013_401367


namespace NUMINAMATH_CALUDE_math_club_probability_l4013_401303

def club_sizes : List Nat := [6, 9, 10]
def co_presidents_per_club : Nat := 3
def members_selected : Nat := 4

def probability_two_copresidents (n : Nat) : Rat :=
  (Nat.choose co_presidents_per_club 2 * Nat.choose (n - co_presidents_per_club) 2) /
  Nat.choose n members_selected

theorem math_club_probability : 
  (1 / 3 : Rat) * (club_sizes.map probability_two_copresidents).sum = 44 / 105 := by
  sorry

end NUMINAMATH_CALUDE_math_club_probability_l4013_401303


namespace NUMINAMATH_CALUDE_max_k_value_l4013_401304

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 5 = k^2 * (x^2/y^2 + y^2/x^2) + 2*k * (x/y + y/x)) :
  k ≤ (-1 + Real.sqrt 56) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l4013_401304


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l4013_401329

theorem arithmetic_calculation : (((3.242 * (14 + 6)) - (7.234 * 7)) / 20) = 0.7101 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l4013_401329


namespace NUMINAMATH_CALUDE_distance_to_softball_park_l4013_401355

/-- Represents the problem of calculating the distance to the softball park -/
def softball_park_distance (efficiency : ℝ) (initial_gas : ℝ) 
  (to_school : ℝ) (to_restaurant : ℝ) (to_friend : ℝ) (to_home : ℝ) : ℝ :=
  efficiency * initial_gas - (to_school + to_restaurant + to_friend + to_home)

/-- Theorem stating that the distance to the softball park is 6 miles -/
theorem distance_to_softball_park :
  softball_park_distance 19 2 15 2 4 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_softball_park_l4013_401355


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l4013_401368

theorem quadratic_equations_solutions :
  (∃ x : ℝ, 2 * x^2 - 2 * Real.sqrt 2 * x + 1 = 0 ∧ x = Real.sqrt 2 / 2) ∧
  (∃ x₁ x₂ : ℝ, x₁ * (2 * x₁ - 5) = 4 * x₁ - 10 ∧
                x₂ * (2 * x₂ - 5) = 4 * x₂ - 10 ∧
                x₁ = 5 / 2 ∧ x₂ = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l4013_401368


namespace NUMINAMATH_CALUDE_olivia_money_made_l4013_401340

/-- Represents the types of chocolate bars -/
inductive ChocolateType
| A
| B
| C

/-- The cost of each type of chocolate bar -/
def cost (t : ChocolateType) : ℕ :=
  match t with
  | ChocolateType.A => 3
  | ChocolateType.B => 4
  | ChocolateType.C => 5

/-- The total number of bars in the box -/
def total_bars : ℕ := 15

/-- The number of bars of each type in the box -/
def bars_in_box (t : ChocolateType) : ℕ :=
  match t with
  | ChocolateType.A => 7
  | ChocolateType.B => 5
  | ChocolateType.C => 3

/-- The number of bars sold of each type -/
def bars_sold (t : ChocolateType) : ℕ :=
  match t with
  | ChocolateType.A => 4
  | ChocolateType.B => 3
  | ChocolateType.C => 2

/-- The total money made from selling the chocolate bars -/
def total_money : ℕ :=
  (bars_sold ChocolateType.A * cost ChocolateType.A) +
  (bars_sold ChocolateType.B * cost ChocolateType.B) +
  (bars_sold ChocolateType.C * cost ChocolateType.C)

theorem olivia_money_made :
  total_money = 34 :=
by sorry

end NUMINAMATH_CALUDE_olivia_money_made_l4013_401340


namespace NUMINAMATH_CALUDE_middle_share_in_ratio_l4013_401393

/-- Proves that in a 3:5:7 ratio distribution with a 1200 difference between extremes, the middle value is 1500 -/
theorem middle_share_in_ratio (total : ℕ) : 
  let f := 3 * total / 15
  let v := 5 * total / 15
  let r := 7 * total / 15
  r - f = 1200 → v = 1500 := by
  sorry

end NUMINAMATH_CALUDE_middle_share_in_ratio_l4013_401393


namespace NUMINAMATH_CALUDE_quadratic_inequality_l4013_401382

theorem quadratic_inequality (x : ℝ) : 
  10 * x^2 - 2 * x - 3 < 0 ↔ (1 - Real.sqrt 31) / 10 < x ∧ x < (1 + Real.sqrt 31) / 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l4013_401382


namespace NUMINAMATH_CALUDE_hiker_distance_problem_l4013_401365

theorem hiker_distance_problem (v t d : ℝ) :
  v > 0 ∧ t > 0 ∧ d > 0 ∧
  d = v * t ∧
  d = (v + 1) * (3 * t / 4) ∧
  d = (v - 1) * (t + 3) →
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_hiker_distance_problem_l4013_401365


namespace NUMINAMATH_CALUDE_cube_of_4_minus_3i_l4013_401392

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem cube_of_4_minus_3i :
  (4 - 3 * i) ^ 3 = -44 - 117 * i :=
by sorry

end NUMINAMATH_CALUDE_cube_of_4_minus_3i_l4013_401392


namespace NUMINAMATH_CALUDE_fraction_addition_l4013_401387

theorem fraction_addition : (1 : ℚ) / 420 + 19 / 35 = 229 / 420 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l4013_401387


namespace NUMINAMATH_CALUDE_edward_book_spending_l4013_401378

/-- The amount of money Edward spent on books -/
def money_spent (num_books : ℕ) (cost_per_book : ℕ) : ℕ :=
  num_books * cost_per_book

/-- Theorem: Edward spent $6 on books -/
theorem edward_book_spending :
  money_spent 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_edward_book_spending_l4013_401378


namespace NUMINAMATH_CALUDE_white_ball_players_l4013_401363

theorem white_ball_players (total : ℕ) (yellow : ℕ) (both : ℕ) (h1 : total = 35) (h2 : yellow = 28) (h3 : both = 19) :
  total = (yellow - both) + (total - yellow + both) :=
by sorry

#check white_ball_players

end NUMINAMATH_CALUDE_white_ball_players_l4013_401363


namespace NUMINAMATH_CALUDE_two_points_determine_line_l4013_401386

-- Define a Point type
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a Line type
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

-- Two points determine a unique line
theorem two_points_determine_line (P Q : Point) (h : P ≠ Q) :
  ∃! L : Line, (L.a * P.x + L.b * P.y + L.c = 0) ∧ (L.a * Q.x + L.b * Q.y + L.c = 0) :=
sorry

end NUMINAMATH_CALUDE_two_points_determine_line_l4013_401386


namespace NUMINAMATH_CALUDE_quadratic_equation_range_l4013_401300

theorem quadratic_equation_range (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 4*x + m - 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁^2 - 4*x₁ + m - 1 = 0) →
  (x₂^2 - 4*x₂ + m - 1 = 0) →
  (3*x₁*x₂ - x₁ - x₂ > 2) →
  3 < m ∧ m ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_range_l4013_401300


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4013_401346

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : a 1 = 1
  h3 : ∀ n : ℕ, a (n + 1) = a n + d
  h4 : (a 5) ^ 2 = (a 3) * (a 10)

/-- The sum of the first n terms of the arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1) + (n * (n - 1) * seq.d) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) (n : ℕ) :
  sum_n seq n = -3/4 * n^2 + 7/4 * n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4013_401346


namespace NUMINAMATH_CALUDE_inequality_proof_l4013_401375

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 ∧
  ((x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ ∃ (k : ℝ), x = 2*k ∧ y = k ∧ z = k) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4013_401375


namespace NUMINAMATH_CALUDE_remainder_after_adding_1008_l4013_401341

theorem remainder_after_adding_1008 (n : ℤ) : 
  n % 4 = 1 → (n + 1008) % 4 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_after_adding_1008_l4013_401341


namespace NUMINAMATH_CALUDE_intersection_sum_l4013_401379

-- Define the logarithm base 3
noncomputable def log3 (x : ℝ) := Real.log x / Real.log 3

-- Define the condition for the intersection points
def intersection_condition (k : ℝ) : Prop :=
  |log3 k - log3 (k + 5)| = 0.6

-- Define the form of k
def k_form (a b : ℤ) (k : ℝ) : Prop :=
  k = a + Real.sqrt (b : ℝ)

-- Main theorem
theorem intersection_sum (k : ℝ) (a b : ℤ) :
  intersection_condition k → k_form a b k → a + b = 8 :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_l4013_401379


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_bounds_l4013_401342

theorem geometric_sequence_sum_bounds (a : ℕ → ℚ) (S : ℕ → ℚ) (A B : ℚ) :
  (∀ n : ℕ, a n = 4/3 * (-1/3)^n) →
  (∀ n : ℕ, S (n+1) = (4/3 * (1 - (-1/3)^(n+1))) / (1 + 1/3)) →
  (∀ n : ℕ, n > 0 → A ≤ S n - 1 / S n ∧ S n - 1 / S n ≤ B) →
  59/72 ≤ B - A :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_bounds_l4013_401342


namespace NUMINAMATH_CALUDE_square_removal_domino_tiling_l4013_401315

theorem square_removal_domino_tiling (n m : ℕ) (hn : n = 2011) (hm : m = 11) :
  (∃ (k : ℕ), k = (n - m + 1)^2 / 2 + ((n - m + 1)^2 % 2)) ∧
  (∀ (k : ℕ), k = (n - m + 1)^2 / 2 + ((n - m + 1)^2 % 2) → k = 2002001) :=
by sorry

end NUMINAMATH_CALUDE_square_removal_domino_tiling_l4013_401315


namespace NUMINAMATH_CALUDE_garden_perimeter_l4013_401350

theorem garden_perimeter : 
  ∀ (width length : ℝ),
  length = 3 * width + 2 →
  length = 38 →
  2 * length + 2 * width = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l4013_401350


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l4013_401353

theorem regular_polygon_interior_angle_sum 
  (n : ℕ) 
  (h_regular : n ≥ 3) 
  (h_exterior : (360 : ℝ) / n = 40) : 
  (n - 2 : ℝ) * 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l4013_401353


namespace NUMINAMATH_CALUDE_rectangle_area_solution_l4013_401314

/-- A rectangle with dimensions (2x - 3) by (3x + 4) has an area of 14x - 12. -/
theorem rectangle_area_solution (x : ℝ) : 
  (2 * x - 3 > 0) → 
  (3 * x + 4 > 0) → 
  (2 * x - 3) * (3 * x + 4) = 14 * x - 12 → 
  x = 4 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_solution_l4013_401314
