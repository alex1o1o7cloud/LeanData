import Mathlib

namespace NUMINAMATH_CALUDE_pipe_problem_l2644_264480

theorem pipe_problem (fill_time : ℕ → ℝ) (h1 : fill_time 2 = 18) (h2 : ∃ n : ℕ, fill_time n = 12) : 
  ∃ n : ℕ, n = 3 ∧ fill_time n = 12 := by
  sorry

end NUMINAMATH_CALUDE_pipe_problem_l2644_264480


namespace NUMINAMATH_CALUDE_base_for_256_four_digits_l2644_264409

-- Define the property of a number having exactly 4 digits in a given base
def has_four_digits (n : ℕ) (b : ℕ) : Prop :=
  b ^ 3 ≤ n ∧ n < b ^ 4

-- State the theorem
theorem base_for_256_four_digits :
  ∃! b : ℕ, has_four_digits 256 b ∧ b = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_for_256_four_digits_l2644_264409


namespace NUMINAMATH_CALUDE_power_product_equals_ten_thousand_l2644_264485

theorem power_product_equals_ten_thousand : (2 ^ 4) * (5 ^ 4) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_ten_thousand_l2644_264485


namespace NUMINAMATH_CALUDE_perfect_match_production_l2644_264467

theorem perfect_match_production (total_workers : ℕ) 
  (tables_per_worker : ℕ) (chairs_per_worker : ℕ) 
  (table_workers : ℕ) (chair_workers : ℕ) : 
  total_workers = 36 → 
  tables_per_worker = 20 → 
  chairs_per_worker = 50 → 
  table_workers = 20 → 
  chair_workers = 16 → 
  table_workers + chair_workers = total_workers → 
  2 * (table_workers * tables_per_worker) = chair_workers * chairs_per_worker :=
by
  sorry

#check perfect_match_production

end NUMINAMATH_CALUDE_perfect_match_production_l2644_264467


namespace NUMINAMATH_CALUDE_total_is_300_l2644_264422

/-- The number of pennies thrown by Rachelle, Gretchen, and Rocky -/
def penny_throwing (rachelle gretchen rocky : ℕ) : Prop :=
  rachelle = 180 ∧ 
  gretchen = rachelle / 2 ∧ 
  rocky = gretchen / 3

/-- The total number of pennies thrown by all three -/
def total_pennies (rachelle gretchen rocky : ℕ) : ℕ :=
  rachelle + gretchen + rocky

/-- Theorem stating that the total number of pennies thrown is 300 -/
theorem total_is_300 (rachelle gretchen rocky : ℕ) : 
  penny_throwing rachelle gretchen rocky → total_pennies rachelle gretchen rocky = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_total_is_300_l2644_264422


namespace NUMINAMATH_CALUDE_john_coin_collection_value_l2644_264459

/-- Represents the value of John's coin collection -/
def coin_collection_value (total_coins : ℕ) (silver_coins : ℕ) (gold_coins : ℕ) 
  (silver_coin_value : ℚ) (regular_coin_value : ℚ) : ℚ :=
  let gold_coin_value := 2 * silver_coin_value
  let regular_coins := total_coins - (silver_coins + gold_coins)
  silver_coins * silver_coin_value + gold_coins * gold_coin_value + regular_coins * regular_coin_value

theorem john_coin_collection_value : 
  coin_collection_value 20 10 5 (30/4) 1 = 155 := by
  sorry


end NUMINAMATH_CALUDE_john_coin_collection_value_l2644_264459


namespace NUMINAMATH_CALUDE_smallest_initial_value_l2644_264489

theorem smallest_initial_value : 
  ∃ (x : ℕ), x + 42 = 456 ∧ ∀ (y : ℕ), y + 42 = 456 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_initial_value_l2644_264489


namespace NUMINAMATH_CALUDE_total_friends_l2644_264470

/-- The number of friends who attended the movie -/
def M : ℕ := 10

/-- The number of friends who attended the picnic -/
def P : ℕ := 20

/-- The number of friends who attended the games -/
def G : ℕ := 5

/-- The number of friends who attended both movie and picnic -/
def MP : ℕ := 4

/-- The number of friends who attended both movie and games -/
def MG : ℕ := 2

/-- The number of friends who attended both picnic and games -/
def PG : ℕ := 0

/-- The number of friends who attended all three events -/
def MPG : ℕ := 2

/-- The total number of unique friends -/
def N : ℕ := M + P + G - MP - MG - PG + MPG

theorem total_friends : N = 31 := by sorry

end NUMINAMATH_CALUDE_total_friends_l2644_264470


namespace NUMINAMATH_CALUDE_first_valid_year_is_2030_l2644_264451

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2022 ∧ sum_of_digits year = 5

theorem first_valid_year_is_2030 :
  is_valid_year 2030 ∧ ∀ y, is_valid_year y → y ≥ 2030 :=
sorry

end NUMINAMATH_CALUDE_first_valid_year_is_2030_l2644_264451


namespace NUMINAMATH_CALUDE_class_size_calculation_l2644_264407

theorem class_size_calculation (average_age : ℝ) (age_increase : ℝ) (staff_age : ℕ) : ℕ :=
  let n : ℕ := 32
  let T : ℝ := n * average_age
  have h1 : T / n = average_age := by sorry
  have h2 : (T + staff_age) / (n + 1) = average_age + age_increase := by sorry
  have h3 : staff_age = 49 := by sorry
  have h4 : average_age = 16 := by sorry
  have h5 : age_increase = 1 := by sorry
  n

#check class_size_calculation

end NUMINAMATH_CALUDE_class_size_calculation_l2644_264407


namespace NUMINAMATH_CALUDE_dilution_proof_l2644_264494

/-- Proves that adding 7 ounces of water to 12 ounces of a 40% alcohol solution results in a 25% alcohol solution -/
theorem dilution_proof (original_volume : ℝ) (original_concentration : ℝ) 
  (target_concentration : ℝ) (water_added : ℝ) : 
  original_volume = 12 →
  original_concentration = 0.4 →
  target_concentration = 0.25 →
  water_added = 7 →
  (original_volume * original_concentration) / (original_volume + water_added) = target_concentration :=
by
  sorry

end NUMINAMATH_CALUDE_dilution_proof_l2644_264494


namespace NUMINAMATH_CALUDE_characterize_functions_l2644_264404

-- Define the property of the function f
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = f ((x^2 + y^2) / 2) + (x - y)^2

-- State the theorem
theorem characterize_functions (f : ℝ → ℝ) 
  (hf : Continuous f) 
  (hprop : satisfies_property f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c - 2 * x :=
sorry

end NUMINAMATH_CALUDE_characterize_functions_l2644_264404


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2644_264477

theorem no_integer_solutions : ¬∃ (m n : ℤ), m^3 + 4*m^2 + 3*m = 8*n^3 + 12*n^2 + 6*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2644_264477


namespace NUMINAMATH_CALUDE_richard_numbers_l2644_264418

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def all_digits_distinct (n : ℕ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < 5 ∧ 0 ≤ j ∧ j < 5 → i ≠ j →
    (n / (10^i) % 10) ≠ (n / (10^j) % 10)

def all_digits_odd (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 5 → (n / (10^i) % 10) % 2 = 1

def all_digits_even (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 5 → (n / (10^i) % 10) % 2 = 0

def sum_starts_with_11_ends_with_1 (a b : ℕ) : Prop :=
  let sum := a + b
  110000 ≤ sum ∧ sum < 120000 ∧ sum % 10 = 1

def diff_starts_with_2_ends_with_11 (a b : ℕ) : Prop :=
  let diff := a - b
  20000 ≤ diff ∧ diff < 30000 ∧ diff % 100 = 11

theorem richard_numbers :
  ∃ (A B : ℕ),
    is_five_digit A ∧
    is_five_digit B ∧
    all_digits_distinct A ∧
    all_digits_distinct B ∧
    all_digits_odd A ∧
    all_digits_even B ∧
    sum_starts_with_11_ends_with_1 A B ∧
    diff_starts_with_2_ends_with_11 A B ∧
    A = 73591 ∧
    B = 46280 :=
by sorry

end NUMINAMATH_CALUDE_richard_numbers_l2644_264418


namespace NUMINAMATH_CALUDE_function_sum_l2644_264420

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 2)
  (h_def : ∀ x ∈ Set.Ioo 0 1, f x = Real.sin (Real.pi * x)) :
  f (-5/2) + f 1 + f 2 = -1 := by
sorry

end NUMINAMATH_CALUDE_function_sum_l2644_264420


namespace NUMINAMATH_CALUDE_problem_solution_l2644_264491

theorem problem_solution (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 3*x + 3/x + 1/x^2 = 26)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2644_264491


namespace NUMINAMATH_CALUDE_brian_final_cards_l2644_264432

def initial_cards : ℕ := 76
def cards_taken : ℕ := 59
def packs_bought : ℕ := 3
def cards_per_pack : ℕ := 15

theorem brian_final_cards : 
  initial_cards - cards_taken + packs_bought * cards_per_pack = 62 := by
  sorry

end NUMINAMATH_CALUDE_brian_final_cards_l2644_264432


namespace NUMINAMATH_CALUDE_prob_reroll_two_dice_l2644_264497

/-- The number of possible outcomes when rolling three fair six-sided dice -/
def total_outcomes : ℕ := 6^3

/-- The number of ways to get a sum of 8 when rolling three fair six-sided dice -/
def sum_eight_outcomes : ℕ := 20

/-- The probability that the sum of three fair six-sided dice is not equal to 8 -/
def prob_not_eight : ℚ := (total_outcomes - sum_eight_outcomes) / total_outcomes

theorem prob_reroll_two_dice : prob_not_eight = 49 / 54 := by
  sorry

end NUMINAMATH_CALUDE_prob_reroll_two_dice_l2644_264497


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l2644_264479

theorem smallest_lcm_with_gcd_five (k l : ℕ) : 
  k ≥ 10000 ∧ k < 100000 ∧ 
  l ≥ 10000 ∧ l < 100000 ∧ 
  Nat.gcd k l = 5 → 
  Nat.lcm k l ≥ 20010000 := by
sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l2644_264479


namespace NUMINAMATH_CALUDE_beach_trip_time_difference_l2644_264466

theorem beach_trip_time_difference (bus_time car_round_trip : ℕ) : 
  bus_time = 40 → car_round_trip = 70 → bus_time - car_round_trip / 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_beach_trip_time_difference_l2644_264466


namespace NUMINAMATH_CALUDE_quadratic_trinomial_not_factor_l2644_264436

theorem quadratic_trinomial_not_factor (r : ℕ) (p : Polynomial ℤ) :
  (∀ i, |p.coeff i| < r) →
  p ≠ 0 →
  ¬ (X^2 - r • X - 1 : Polynomial ℤ) ∣ p :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_not_factor_l2644_264436


namespace NUMINAMATH_CALUDE_mountain_bike_pricing_l2644_264462

/-- Represents the sales and pricing of mountain bikes over three months -/
structure MountainBikeSales where
  january_sales : ℝ
  february_price_decrease : ℝ
  february_sales : ℝ
  march_price_decrease_percentage : ℝ
  march_profit_percentage : ℝ

/-- Theorem stating the selling price in February and the cost price of each mountain bike -/
theorem mountain_bike_pricing (sales : MountainBikeSales)
  (h1 : sales.january_sales = 27000)
  (h2 : sales.february_price_decrease = 100)
  (h3 : sales.february_sales = 24000)
  (h4 : sales.march_price_decrease_percentage = 0.1)
  (h5 : sales.march_profit_percentage = 0.44) :
  ∃ (february_price cost_price : ℝ),
    february_price = 800 ∧ cost_price = 500 := by
  sorry

end NUMINAMATH_CALUDE_mountain_bike_pricing_l2644_264462


namespace NUMINAMATH_CALUDE_equal_powers_of_negative_one_l2644_264439

theorem equal_powers_of_negative_one : 
  (-7^4 ≠ (-7)^4) ∧ 
  (4^3 ≠ 3^4) ∧ 
  (-(-6) ≠ -|(-6)|) ∧ 
  ((-1)^3 = (-1)^2023) := by
  sorry

end NUMINAMATH_CALUDE_equal_powers_of_negative_one_l2644_264439


namespace NUMINAMATH_CALUDE_weighted_average_constants_l2644_264484

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define points A, B, C, P, Q
variable (A B C P Q : V)

-- Define the conditions
variable (hAPC : ∃ (k : ℝ), P - C = k • (A - C) ∧ k = 4/5)
variable (hBQC : ∃ (k : ℝ), Q - C = k • (B - C) ∧ k = 1/5)

-- Define r and s
variable (r s : ℝ)

-- Define the weighted average conditions
variable (hP : P = r • A + (1 - r) • C)
variable (hQ : Q = s • B + (1 - s) • C)

-- State the theorem
theorem weighted_average_constants : r = 1/5 ∧ s = 4/5 := by sorry

end NUMINAMATH_CALUDE_weighted_average_constants_l2644_264484


namespace NUMINAMATH_CALUDE_find_b_value_l2644_264435

/-- Given the equation a * b * c = ( √ ( a + 2 ) ( b + 3 ) ) / ( c + 1 ),
    when a = 6, c = 3, and the left-hand side of the equation equals 3,
    prove that b = 15. -/
theorem find_b_value (a b c : ℝ) :
  a = 6 →
  c = 3 →
  a * b * c = ( Real.sqrt ((a + 2) * (b + 3)) ) / (c + 1) →
  a * b * c = 3 →
  b = 15 := by
  sorry


end NUMINAMATH_CALUDE_find_b_value_l2644_264435


namespace NUMINAMATH_CALUDE_teacher_student_relationship_l2644_264438

/-- In a school system, prove the relationship between teachers and students -/
theorem teacher_student_relationship (m n k l : ℕ) 
  (h1 : m > 0) -- Ensure there's at least one teacher
  (h2 : n > 0) -- Ensure there's at least one student
  (h3 : k > 0) -- Each teacher has at least one student
  (h4 : l > 0) -- Each student has at least one teacher
  (h5 : ∀ t, t ≤ m → (∃ s, s = k)) -- Each teacher has exactly k students
  (h6 : ∀ s, s ≤ n → (∃ t, t = l)) -- Each student has exactly l teachers
  : m * k = n * l := by
  sorry

end NUMINAMATH_CALUDE_teacher_student_relationship_l2644_264438


namespace NUMINAMATH_CALUDE_quadratic_discriminant_problem_l2644_264486

theorem quadratic_discriminant_problem (m : ℝ) : 
  ((-3)^2 - 4*1*(-m) = 13) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_problem_l2644_264486


namespace NUMINAMATH_CALUDE_max_rectangles_in_square_l2644_264437

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square grid -/
structure Grid where
  size : ℕ

/-- Defines a 4×1 rectangle -/
def fourByOne : Rectangle := { width := 4, height := 1 }

/-- Defines a 6×6 grid -/
def sixBySix : Grid := { size := 6 }

/-- 
  Theorem: The maximum number of 4×1 rectangles that can be placed 
  in a 6×6 square without crossing cell boundaries is 8.
-/
theorem max_rectangles_in_square : 
  ∃ (n : ℕ), n = 8 ∧ 
  (∀ (m : ℕ), m > n → 
    ¬ (∃ (arrangement : List (ℕ × ℕ)), 
      arrangement.length = m ∧
      (∀ (pos : ℕ × ℕ), pos ∈ arrangement → 
        pos.1 + fourByOne.width ≤ sixBySix.size ∧ 
        pos.2 + fourByOne.height ≤ sixBySix.size) ∧
      (∀ (pos1 pos2 : ℕ × ℕ), pos1 ∈ arrangement → pos2 ∈ arrangement → pos1 ≠ pos2 → 
        ¬ (pos1.1 < pos2.1 + fourByOne.width ∧ 
           pos2.1 < pos1.1 + fourByOne.width ∧ 
           pos1.2 < pos2.2 + fourByOne.height ∧ 
           pos2.2 < pos1.2 + fourByOne.height)))) :=
by
  sorry

end NUMINAMATH_CALUDE_max_rectangles_in_square_l2644_264437


namespace NUMINAMATH_CALUDE_linear_function_slope_l2644_264434

theorem linear_function_slope (x₁ y₁ x₂ y₂ k : ℝ) :
  x₁ ≠ x₂ →
  y₁ = 2 * x₁ - k * x₁ + 1 →
  y₂ = 2 * x₂ - k * x₂ + 1 →
  (x₁ - x₂) * (y₁ - y₂) < 0 →
  k > 2 := by
sorry

end NUMINAMATH_CALUDE_linear_function_slope_l2644_264434


namespace NUMINAMATH_CALUDE_angle_abc_measure_l2644_264487

theorem angle_abc_measure (θ : ℝ) : 
  θ > 0 ∧ θ < 180 → -- Angle measure is positive and less than 180°
  (θ / 2) = (1 / 3) * (180 - θ) → -- Condition about angle bisector
  θ = 72 := by
sorry

end NUMINAMATH_CALUDE_angle_abc_measure_l2644_264487


namespace NUMINAMATH_CALUDE_jellybean_probability_l2644_264464

/-- The probability of selecting exactly 2 red and 1 green jellybean when picking 4 jellybeans randomly without replacement from a bowl containing 5 red, 3 blue, 2 green, and 5 white jellybeans (15 total) -/
theorem jellybean_probability : 
  let total := 15
  let red := 5
  let blue := 3
  let green := 2
  let white := 5
  let pick := 4
  Nat.choose total pick ≠ 0 →
  (Nat.choose red 2 * Nat.choose green 1 * Nat.choose (blue + white) 1) / Nat.choose total pick = 32 / 273 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_probability_l2644_264464


namespace NUMINAMATH_CALUDE_complex_number_and_pure_imaginary_l2644_264471

-- Define the complex number z
def z : ℂ := sorry

-- Define the real number m
def m : ℝ := sorry

-- Theorem statement
theorem complex_number_and_pure_imaginary :
  (Complex.abs z = Real.sqrt 2) ∧
  (z.im = 1) ∧
  (z.re < 0) ∧
  (z = -1 + Complex.I) ∧
  (∃ (k : ℝ), m^2 + m + m * z^2 = k * Complex.I) →
  (z = -1 + Complex.I) ∧ (m = -1) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_and_pure_imaginary_l2644_264471


namespace NUMINAMATH_CALUDE_average_book_width_l2644_264416

def book_widths : List ℝ := [7.5, 3, 0.75, 4, 1.25, 12]

theorem average_book_width : 
  (List.sum book_widths) / (List.length book_widths) = 4.75 := by
  sorry

end NUMINAMATH_CALUDE_average_book_width_l2644_264416


namespace NUMINAMATH_CALUDE_molecular_weight_NH4_correct_l2644_264441

/-- The molecular weight of NH4 in grams per mole -/
def molecular_weight_NH4 : ℝ := 18

/-- The number of moles in the given sample -/
def sample_moles : ℝ := 7

/-- The total weight of the sample in grams -/
def sample_weight : ℝ := 126

/-- Theorem stating that the molecular weight of NH4 is correct given the sample information -/
theorem molecular_weight_NH4_correct :
  molecular_weight_NH4 * sample_moles = sample_weight :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_NH4_correct_l2644_264441


namespace NUMINAMATH_CALUDE_pacific_ocean_area_rounded_l2644_264431

/-- Rounds a number to the nearest multiple of 10000 -/
def roundToNearestTenThousand (n : ℕ) : ℕ :=
  ((n + 5000) / 10000) * 10000

theorem pacific_ocean_area_rounded :
  roundToNearestTenThousand 17996800 = 18000000 := by sorry

end NUMINAMATH_CALUDE_pacific_ocean_area_rounded_l2644_264431


namespace NUMINAMATH_CALUDE_lucas_units_digit_l2644_264457

-- Define Lucas numbers
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem lucas_units_digit :
  unitsDigit (lucas (lucas 15)) = 7 := by sorry

end NUMINAMATH_CALUDE_lucas_units_digit_l2644_264457


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2644_264425

theorem tangent_line_to_circle (r : ℝ) : 
  r > 0 → 
  (∃ (x y : ℝ), x + 2*y = r ∧ x^2 + y^2 = 2*r^2) →
  (∀ (x y : ℝ), x + 2*y = r → x^2 + y^2 ≥ 2*r^2) →
  r = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2644_264425


namespace NUMINAMATH_CALUDE_existence_of_counterexample_l2644_264468

theorem existence_of_counterexample : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ |a - b| + (1 / (a - b)) < 2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_counterexample_l2644_264468


namespace NUMINAMATH_CALUDE_unique_number_count_l2644_264498

/-- The number of unique 5-digit numbers that can be formed by rearranging
    the digits 3, 7, 3, 2, 2, 0, where the number doesn't start with 0. -/
def unique_numbers : ℕ := 24

/-- The set of digits available for forming the numbers. -/
def digits : Finset ℕ := {3, 7, 2, 0}

/-- The total number of digits to be used. -/
def total_digits : ℕ := 5

/-- The number of positions where 0 can be placed (not in the first position). -/
def zero_positions : ℕ := 4

/-- The number of times 3 appears in the original number. -/
def count_three : ℕ := 2

/-- The number of times 2 appears in the original number. -/
def count_two : ℕ := 2

theorem unique_number_count :
  unique_numbers = (zero_positions * Nat.factorial (total_digits - 1)) /
                   (Nat.factorial count_three * Nat.factorial count_two) :=
sorry

end NUMINAMATH_CALUDE_unique_number_count_l2644_264498


namespace NUMINAMATH_CALUDE_calorie_difference_l2644_264433

/-- The number of squirrels Brandon can catch in 1 hour -/
def squirrels_per_hour : ℕ := 6

/-- The number of rabbits Brandon can catch in 1 hour -/
def rabbits_per_hour : ℕ := 2

/-- The number of calories in each squirrel -/
def calories_per_squirrel : ℕ := 300

/-- The number of calories in each rabbit -/
def calories_per_rabbit : ℕ := 800

/-- The difference in calories between catching squirrels and rabbits for 1 hour -/
theorem calorie_difference : 
  (squirrels_per_hour * calories_per_squirrel) - (rabbits_per_hour * calories_per_rabbit) = 200 := by
  sorry

end NUMINAMATH_CALUDE_calorie_difference_l2644_264433


namespace NUMINAMATH_CALUDE_gcd_1855_1120_l2644_264492

theorem gcd_1855_1120 : Nat.gcd 1855 1120 = 35 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1855_1120_l2644_264492


namespace NUMINAMATH_CALUDE_john_chips_bought_l2644_264495

-- Define the cost of chips and corn chips
def chip_cost : ℚ := 2
def corn_chip_cost : ℚ := 3/2

-- Define John's budget
def budget : ℚ := 45

-- Define the number of corn chips John can buy with remaining money
def corn_chips_bought : ℚ := 10

-- Define the function to calculate the number of corn chips that can be bought with remaining money
def corn_chips_buyable (x : ℚ) : ℚ := (budget - chip_cost * x) / corn_chip_cost

-- Theorem statement
theorem john_chips_bought : 
  ∃ (x : ℚ), x = 15 ∧ corn_chips_buyable x = corn_chips_bought :=
sorry

end NUMINAMATH_CALUDE_john_chips_bought_l2644_264495


namespace NUMINAMATH_CALUDE_h2o_required_for_reaction_l2644_264428

-- Define the chemical reaction
def chemical_reaction (NaH H2O NaOH H2 : ℕ) : Prop :=
  NaH = H2O ∧ NaH = NaOH ∧ NaH = H2

-- Define the problem statement
theorem h2o_required_for_reaction (NaH : ℕ) (h : NaH = 2) :
  ∃ H2O : ℕ, chemical_reaction NaH H2O NaH NaH ∧ H2O = 2 :=
by sorry

end NUMINAMATH_CALUDE_h2o_required_for_reaction_l2644_264428


namespace NUMINAMATH_CALUDE_second_divisor_problem_l2644_264429

theorem second_divisor_problem : ∃ (D N k m : ℕ+), N = 39 * k + 17 ∧ N = D * m + 4 ∧ D = 13 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_problem_l2644_264429


namespace NUMINAMATH_CALUDE_halloween_candy_problem_l2644_264421

/-- The number of candy pieces Robin's sister gave her -/
def candy_from_sister (initial : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial - eaten)

theorem halloween_candy_problem :
  let initial := 23
  let eaten := 7
  let final := 37
  candy_from_sister initial eaten final = 21 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_problem_l2644_264421


namespace NUMINAMATH_CALUDE_committee_arrangements_eq_1680_l2644_264406

/-- The number of distinct arrangements of letters in "COMMITTEE" -/
def committee_arrangements : ℕ :=
  let total_letters : ℕ := 8
  let c_count : ℕ := 2
  let m_count : ℕ := 2
  let e_count : ℕ := 3
  let i_count : ℕ := 1
  let t_count : ℕ := 1
  Nat.factorial total_letters / (Nat.factorial c_count * Nat.factorial m_count * Nat.factorial e_count * Nat.factorial i_count * Nat.factorial t_count)

theorem committee_arrangements_eq_1680 : committee_arrangements = 1680 := by
  sorry

end NUMINAMATH_CALUDE_committee_arrangements_eq_1680_l2644_264406


namespace NUMINAMATH_CALUDE_credit_card_balance_l2644_264460

theorem credit_card_balance 
  (G : ℝ) 
  (gold_balance : ℝ) 
  (platinum_balance : ℝ) 
  (h1 : gold_balance = G / 3) 
  (h2 : 0.5833333333333334 = 1 - (platinum_balance + gold_balance) / (2 * G)) : 
  platinum_balance = (1 / 4) * (2 * G) := by
sorry

end NUMINAMATH_CALUDE_credit_card_balance_l2644_264460


namespace NUMINAMATH_CALUDE_expand_expression_l2644_264411

theorem expand_expression (x : ℝ) : 20 * (3 * x - 4) = 60 * x - 80 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2644_264411


namespace NUMINAMATH_CALUDE_some_number_value_l2644_264402

theorem some_number_value (some_number : ℝ) : 
  |9 - 8 * (3 - some_number)| - |5 - 11| = 75 → some_number = 12 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2644_264402


namespace NUMINAMATH_CALUDE_ariel_fish_count_l2644_264488

theorem ariel_fish_count (total : ℕ) (male_fraction : ℚ) (female_count : ℕ) : 
  total = 45 → 
  male_fraction = 2/3 → 
  female_count = total - (total * male_fraction).num → 
  female_count = 15 :=
by sorry

end NUMINAMATH_CALUDE_ariel_fish_count_l2644_264488


namespace NUMINAMATH_CALUDE_power_of_nine_mod_fifty_l2644_264447

theorem power_of_nine_mod_fifty : 9^1002 % 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_nine_mod_fifty_l2644_264447


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l2644_264490

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line
  (given_line : Line)
  (given_point : Point)
  (result_line : Line) :
  given_line = Line.mk 1 2 (-1) →
  given_point = Point.mk 1 2 →
  result_line = Line.mk 1 2 (-5) →
  pointOnLine given_point result_line ∧ parallel given_line result_line := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l2644_264490


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2644_264414

theorem cube_root_equation_solution (x p : ℝ) : 
  (Real.rpow (1 - x) (1/3 : ℝ)) + (Real.rpow (1 + x) (1/3 : ℝ)) = p → 
  (x = 0 ∧ p = -1) → True :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2644_264414


namespace NUMINAMATH_CALUDE_anne_distance_l2644_264400

/-- Given a speed and time, calculates the distance traveled -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Anne's distance traveled is 6 miles -/
theorem anne_distance :
  let speed : ℝ := 2  -- miles per hour
  let time : ℝ := 3   -- hours
  distance speed time = 6 := by sorry

end NUMINAMATH_CALUDE_anne_distance_l2644_264400


namespace NUMINAMATH_CALUDE_circle_ellipse_ratio_l2644_264408

/-- A circle with equation x^2 + (y+1)^2 = n -/
structure Circle where
  n : ℝ

/-- An ellipse with equation x^2 + my^2 = 1 -/
structure Ellipse where
  m : ℝ

/-- The theorem stating that for a circle C and an ellipse M satisfying certain conditions,
    the ratio of n/m equals 8 -/
theorem circle_ellipse_ratio (C : Circle) (M : Ellipse) 
  (h1 : C.n > 0) 
  (h2 : M.m > 0) 
  (h3 : ∃ (x y : ℝ), x^2 + (y+1)^2 = C.n ∧ x^2 + M.m * y^2 = 1) 
  (h4 : ∃ (x y : ℝ), x^2 + y^2 = C.n ∧ x^2 + M.m * y^2 = 1) : 
  C.n / M.m = 8 := by
sorry

end NUMINAMATH_CALUDE_circle_ellipse_ratio_l2644_264408


namespace NUMINAMATH_CALUDE_original_class_size_l2644_264448

theorem original_class_size (original_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) :
  original_avg = 40 →
  new_students = 10 →
  new_avg = 32 →
  avg_decrease = 4 →
  ∃ x : ℕ, x * original_avg + new_students * new_avg = (x + new_students) * (original_avg - avg_decrease) ∧ x = 10 :=
by sorry

end NUMINAMATH_CALUDE_original_class_size_l2644_264448


namespace NUMINAMATH_CALUDE_edward_scored_seven_l2644_264483

/-- Given the total points scored and the friend's score, calculate Edward's score. -/
def edward_score (total : ℕ) (friend_score : ℕ) : ℕ :=
  total - friend_score

/-- Theorem: Edward's score is 7 points when the total is 13 and his friend scored 6. -/
theorem edward_scored_seven :
  edward_score 13 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_edward_scored_seven_l2644_264483


namespace NUMINAMATH_CALUDE_lollipop_distribution_l2644_264481

theorem lollipop_distribution (raspberry mint orange lemon : ℕ) 
  (friends : ℕ) (h1 : raspberry = 60) (h2 : mint = 135) (h3 : orange = 10) 
  (h4 : lemon = 300) (h5 : friends = 14) : 
  (raspberry + mint + orange + lemon) % friends = 1 := by
  sorry

end NUMINAMATH_CALUDE_lollipop_distribution_l2644_264481


namespace NUMINAMATH_CALUDE_construction_team_distance_l2644_264473

/-- Calculates the total distance built by a construction team -/
def total_distance_built (days : ℕ) (rate : ℕ) : ℕ :=
  days * rate

/-- Proves that a construction team working for 5 days at 120 meters per day builds 600 meters -/
theorem construction_team_distance : total_distance_built 5 120 = 600 := by
  sorry

end NUMINAMATH_CALUDE_construction_team_distance_l2644_264473


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2644_264482

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2644_264482


namespace NUMINAMATH_CALUDE_partition_infinite_multiples_l2644_264465

-- Define a partition of Natural Numbers
def Partition (A : ℕ → Set ℕ) (k : ℕ) : Prop :=
  (∀ n, ∃! i, i ≤ k ∧ n ∈ A i) ∧
  (∀ i, i ≤ k → Set.Nonempty (A i))

-- Define what it means for a set to contain infinitely many multiples of a number
def InfiniteMultiples (S : Set ℕ) (x : ℕ) : Prop :=
  Set.Infinite {n ∈ S | ∃ k, n = k * x}

-- Main theorem
theorem partition_infinite_multiples 
  {A : ℕ → Set ℕ} {k : ℕ} (h : Partition A k) :
  ∃ i, i ≤ k ∧ ∀ x : ℕ, x > 0 → InfiniteMultiples (A i) x :=
sorry

end NUMINAMATH_CALUDE_partition_infinite_multiples_l2644_264465


namespace NUMINAMATH_CALUDE_max_pieces_in_box_l2644_264453

theorem max_pieces_in_box : 
  ∃ n : ℕ, n < 50 ∧ 
  (∃ k : ℕ, n = 4 * k + 2) ∧ 
  (∃ m : ℕ, n = 6 * m) ∧
  ∀ x : ℕ, x < 50 → 
    ((∃ k : ℕ, x = 4 * k + 2) ∧ (∃ m : ℕ, x = 6 * m)) → 
    x ≤ n :=
by sorry

end NUMINAMATH_CALUDE_max_pieces_in_box_l2644_264453


namespace NUMINAMATH_CALUDE_course_selection_theorem_l2644_264446

theorem course_selection_theorem (total_courses : Nat) (conflicting_courses : Nat) 
  (courses_to_choose : Nat) (h1 : total_courses = 6) (h2 : conflicting_courses = 2) 
  (h3 : courses_to_choose = 2) :
  (Nat.choose (total_courses - conflicting_courses) courses_to_choose + 
   conflicting_courses * Nat.choose (total_courses - conflicting_courses) (courses_to_choose - 1)) = 14 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l2644_264446


namespace NUMINAMATH_CALUDE_candy_division_l2644_264472

theorem candy_division (total_candy : ℕ) (non_chocolate_candy : ℕ) 
  (chocolate_heart_bags : ℕ) (chocolate_kiss_bags : ℕ) :
  total_candy = 63 →
  non_chocolate_candy = 28 →
  chocolate_heart_bags = 2 →
  chocolate_kiss_bags = 3 →
  ∃ (pieces_per_bag : ℕ),
    pieces_per_bag > 0 ∧
    (total_candy - non_chocolate_candy) = 
      (chocolate_heart_bags + chocolate_kiss_bags) * pieces_per_bag ∧
    non_chocolate_candy % pieces_per_bag = 0 ∧
    chocolate_heart_bags + chocolate_kiss_bags + (non_chocolate_candy / pieces_per_bag) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_division_l2644_264472


namespace NUMINAMATH_CALUDE_age_difference_proof_l2644_264419

theorem age_difference_proof (patrick michael monica : ℕ) 
  (h1 : patrick * 5 = michael * 3)  -- Patrick and Michael's age ratio
  (h2 : michael * 4 = monica * 3)   -- Michael and Monica's age ratio
  (h3 : patrick + michael + monica = 88) -- Sum of ages
  : monica - patrick = 22 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2644_264419


namespace NUMINAMATH_CALUDE_probability_not_greater_than_four_l2644_264499

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

theorem probability_not_greater_than_four 
  (a₁ : ℝ) 
  (d : ℝ) 
  (n : ℕ) 
  (h₁ : a₁ = 12) 
  (h₂ : d = -2) 
  (h₃ : n = 16) : 
  (Finset.filter (fun i => arithmetic_sequence a₁ d i ≤ 4) (Finset.range n)).card / n = 3/4 := by
sorry

end NUMINAMATH_CALUDE_probability_not_greater_than_four_l2644_264499


namespace NUMINAMATH_CALUDE_expression_evaluation_l2644_264463

theorem expression_evaluation : 
  let x : ℝ := 2
  let y : ℝ := -1
  (2*x - y)^2 + (x - 2*y) * (x + 2*y) = 25 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2644_264463


namespace NUMINAMATH_CALUDE_jimmy_stair_climbing_l2644_264454

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Jimmy's stair climbing problem -/
theorem jimmy_stair_climbing : arithmetic_sum 30 10 8 = 520 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_stair_climbing_l2644_264454


namespace NUMINAMATH_CALUDE_largest_common_divisor_462_330_l2644_264415

theorem largest_common_divisor_462_330 : Nat.gcd 462 330 = 66 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_462_330_l2644_264415


namespace NUMINAMATH_CALUDE_det_A_eq_48_l2644_264442

def A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 1, -2; 8, 5, -4; 3, 3, 6]

theorem det_A_eq_48 : Matrix.det A = 48 := by sorry

end NUMINAMATH_CALUDE_det_A_eq_48_l2644_264442


namespace NUMINAMATH_CALUDE_marble_difference_l2644_264452

theorem marble_difference (connie_marbles juan_marbles : ℕ) 
  (h1 : connie_marbles = 39)
  (h2 : juan_marbles = 64)
  : juan_marbles - connie_marbles = 25 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l2644_264452


namespace NUMINAMATH_CALUDE_sequence_formulas_correct_l2644_264445

def sequence1 (n : ℕ) : ℚ := 1 / (n * (n + 1))

def sequence2 (n : ℕ) : ℕ := 2^(n - 1)

def sequence3 (n : ℕ) : ℚ := 4 / (3 * n + 2)

theorem sequence_formulas_correct :
  (∀ n : ℕ, n > 0 → sequence1 n = 1 / (n * (n + 1))) ∧
  (∀ n : ℕ, n > 0 → sequence2 n = 2^(n - 1)) ∧
  (∀ n : ℕ, n > 0 → sequence3 n = 4 / (3 * n + 2)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formulas_correct_l2644_264445


namespace NUMINAMATH_CALUDE_expression_defined_iff_l2644_264403

theorem expression_defined_iff (x : ℝ) :
  (∃ y : ℝ, y = (Real.log (3 - x)) / Real.sqrt (x - 1)) ↔ 1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_defined_iff_l2644_264403


namespace NUMINAMATH_CALUDE_newspaper_price_calculation_l2644_264444

/-- The price of each Wednesday, Thursday, and Friday edition of the newspaper -/
def weekday_price : ℚ := 1/2

theorem newspaper_price_calculation :
  let weeks : ℕ := 8
  let weekday_editions_per_week : ℕ := 3
  let sunday_price : ℚ := 2
  let total_spent : ℚ := 28
  weekday_price = (total_spent - (sunday_price * weeks)) / (weekday_editions_per_week * weeks) :=
by
  sorry

#eval weekday_price

end NUMINAMATH_CALUDE_newspaper_price_calculation_l2644_264444


namespace NUMINAMATH_CALUDE_tree_height_proof_l2644_264458

/-- The growth rate of the tree in inches per year -/
def growth_rate : ℝ := 0.5

/-- The number of years it takes for the tree to reach its final height -/
def years_to_grow : ℕ := 240

/-- The final height of the tree in inches -/
def final_height : ℝ := 720

/-- The current height of the tree in inches -/
def current_height : ℝ := final_height - (growth_rate * years_to_grow)

theorem tree_height_proof :
  current_height = 600 := by sorry

end NUMINAMATH_CALUDE_tree_height_proof_l2644_264458


namespace NUMINAMATH_CALUDE_plywood_perimeter_difference_l2644_264461

def plywood_width : ℝ := 3
def plywood_length : ℝ := 9
def num_pieces : ℕ := 6

def is_valid_cut (w h : ℝ) : Prop :=
  w * h * num_pieces = plywood_width * plywood_length ∧
  (w = plywood_width ∨ h = plywood_width ∨ w = plywood_length ∨ h = plywood_length ∨
   w * num_pieces = plywood_width ∨ h * num_pieces = plywood_width ∨
   w * num_pieces = plywood_length ∨ h * num_pieces = plywood_length)

def piece_perimeter (w h : ℝ) : ℝ := 2 * (w + h)

def max_perimeter : ℝ := 20
def min_perimeter : ℝ := 8

theorem plywood_perimeter_difference :
  ∀ w h, is_valid_cut w h →
  ∃ max_w max_h min_w min_h,
    is_valid_cut max_w max_h ∧
    is_valid_cut min_w min_h ∧
    piece_perimeter max_w max_h = max_perimeter ∧
    piece_perimeter min_w min_h = min_perimeter ∧
    max_perimeter - min_perimeter = 12 :=
sorry

end NUMINAMATH_CALUDE_plywood_perimeter_difference_l2644_264461


namespace NUMINAMATH_CALUDE_apple_vendor_problem_l2644_264496

theorem apple_vendor_problem (initial_apples : ℝ) (h_initial_positive : initial_apples > 0) :
  let first_day_sold := 0.6 * initial_apples
  let first_day_remainder := initial_apples - first_day_sold
  let x := (23 * initial_apples - 0.5 * first_day_remainder) / (0.5 * first_day_remainder)
  x = 0.15
  := by sorry

end NUMINAMATH_CALUDE_apple_vendor_problem_l2644_264496


namespace NUMINAMATH_CALUDE_min_value_theorem_l2644_264430

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 2/b = 3) :
  (a + 1) * (b + 2) ≥ 50/9 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2644_264430


namespace NUMINAMATH_CALUDE_means_inequality_l2644_264478

theorem means_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  Real.sqrt (a * b) > Real.rpow (a * b * c) (1/3) ∧
  Real.rpow (a * b * c) (1/3) > (2 * b * c) / (b + c) := by
sorry

end NUMINAMATH_CALUDE_means_inequality_l2644_264478


namespace NUMINAMATH_CALUDE_constant_chord_length_l2644_264440

/-- Definition of the ellipse C -/
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the satellite circle -/
def satellite_circle (x y a b : ℝ) : Prop := x^2 + y^2 = a^2 + b^2

/-- Theorem statement -/
theorem constant_chord_length (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (h_ecc : (a^2 - b^2) / a^2 = 1/2)
  (h_point : ellipse 2 (Real.sqrt 2) a b)
  (h_sat : satellite_circle 2 (Real.sqrt 2) a b) :
  ∃ (M N : ℝ × ℝ),
    ∀ (P : ℝ × ℝ), satellite_circle P.1 P.2 a b →
      ∃ (l₁ l₂ : ℝ → ℝ),
        (∀ x, (l₁ x - P.2) * (l₂ x - P.2) = -(x - P.1)^2) ∧
        (∃! x₁, ellipse x₁ (l₁ x₁) a b) ∧
        (∃! x₂, ellipse x₂ (l₂ x₂) a b) ∧
        satellite_circle M.1 M.2 a b ∧
        satellite_circle N.1 N.2 a b ∧
        (M.1 - N.1)^2 + (M.2 - N.2)^2 = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_constant_chord_length_l2644_264440


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l2644_264456

-- Define an arithmetic sequence of integers
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define an increasing sequence
def is_increasing_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

-- Theorem statement
theorem arithmetic_sequence_product (a : ℕ → ℤ) :
  is_arithmetic_sequence a →
  is_increasing_sequence a →
  a 4 * a 5 = 13 →
  a 3 * a 6 = -275 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l2644_264456


namespace NUMINAMATH_CALUDE_tile_arrangement_count_l2644_264410

/-- The number of distinguishable arrangements of tiles -/
def tile_arrangements (brown green yellow : ℕ) (purple : ℕ) : ℕ :=
  Nat.factorial (brown + green + yellow + purple) /
  (Nat.factorial brown * Nat.factorial green * Nat.factorial yellow * Nat.factorial purple)

/-- Theorem stating that the number of distinguishable arrangements
    of 2 brown, 3 green, 2 yellow, and 1 purple tile is 1680 -/
theorem tile_arrangement_count :
  tile_arrangements 2 3 2 1 = 1680 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangement_count_l2644_264410


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2644_264469

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  (3 * X ^ 3 - 4 * X ^ 2 + 17 * X + 34 : Polynomial ℤ) = 
  (X - 7) * q + 986 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2644_264469


namespace NUMINAMATH_CALUDE_wendy_phone_pictures_l2644_264449

/-- The number of pictures Wendy uploaded from her phone -/
def phone_pictures (total_albums : ℕ) (pictures_per_album : ℕ) (camera_pictures : ℕ) : ℕ :=
  total_albums * pictures_per_album - camera_pictures

/-- Proof that Wendy uploaded 22 pictures from her phone -/
theorem wendy_phone_pictures :
  phone_pictures 4 6 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_wendy_phone_pictures_l2644_264449


namespace NUMINAMATH_CALUDE_triangle_problem_l2644_264412

theorem triangle_problem (A B C a b c : Real) (t : Real) :
  -- Conditions
  (A + B + C = π) →
  (2 * B = A + C) →
  (b = Real.sqrt 7) →
  (a = 3) →
  (t = Real.sin A * Real.sin C) →
  -- Conclusions
  (c = 4 ∧ t ≤ Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2644_264412


namespace NUMINAMATH_CALUDE_base8_157_equals_111_l2644_264405

/-- Converts a base-8 number to base-10 --/
def base8To10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

/-- The base-8 representation of 157 --/
def base8_157 : List Nat := [1, 5, 7]

theorem base8_157_equals_111 :
  base8To10 base8_157 = 111 := by
  sorry

end NUMINAMATH_CALUDE_base8_157_equals_111_l2644_264405


namespace NUMINAMATH_CALUDE_ratio_problem_l2644_264455

theorem ratio_problem (x y : ℝ) : 
  (0.60 / x = 6 / 2) ∧ (x / y = 8 / 12) → x = 0.20 ∧ y = 0.30 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2644_264455


namespace NUMINAMATH_CALUDE_smallest_perfect_square_sum_of_24_consecutive_integers_l2644_264475

theorem smallest_perfect_square_sum_of_24_consecutive_integers :
  ∃ (n : ℕ), 
    (n > 0) ∧ 
    (∃ (k : ℕ), k * k = 12 * (2 * n + 23)) ∧
    (∀ (m : ℕ), m > 0 → m < n → 
      ¬∃ (j : ℕ), j * j = 12 * (2 * m + 23)) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_sum_of_24_consecutive_integers_l2644_264475


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l2644_264424

/-- Given a hyperbola and a parabola with specific properties, 
    prove that the focal length of the hyperbola is 2√5 -/
theorem hyperbola_focal_length 
  (a b p : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hp : p > 0) 
  (h_distance : p/2 + a = 4) 
  (h_intersection : -1 = -2*b/a ∧ -2 = -p/2) : 
  2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l2644_264424


namespace NUMINAMATH_CALUDE_steven_peaches_difference_l2644_264413

theorem steven_peaches_difference (jake steven jill : ℕ) 
  (h1 : jake + 5 = steven)
  (h2 : jill = 87)
  (h3 : jake = jill + 13) :
  steven - jill = 18 := by sorry

end NUMINAMATH_CALUDE_steven_peaches_difference_l2644_264413


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l2644_264417

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 4) :
  a / c = 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l2644_264417


namespace NUMINAMATH_CALUDE_elliot_book_pages_l2644_264426

/-- The number of pages in Elliot's book -/
def total_pages : ℕ := 381

/-- The number of pages Elliot has already read -/
def pages_read : ℕ := 149

/-- The number of pages Elliot reads per day -/
def pages_per_day : ℕ := 20

/-- The number of days Elliot reads -/
def days_reading : ℕ := 7

/-- The number of pages left to be read after reading for 7 days -/
def pages_left : ℕ := 92

theorem elliot_book_pages : 
  total_pages = pages_read + (pages_per_day * days_reading) + pages_left :=
by sorry

end NUMINAMATH_CALUDE_elliot_book_pages_l2644_264426


namespace NUMINAMATH_CALUDE_alice_bushes_l2644_264476

/-- The number of bushes needed to cover three sides of a yard -/
def bushes_needed (side_length : ℕ) (sides : ℕ) (bush_coverage : ℕ) : ℕ :=
  (side_length * sides) / bush_coverage

/-- Theorem: Alice needs 12 bushes for her yard -/
theorem alice_bushes :
  bushes_needed 16 3 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_alice_bushes_l2644_264476


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l2644_264450

/-- A geometric sequence with common ratio 2 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n, a n > 0) →
  a 3 * a 11 = 16 →
  a 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l2644_264450


namespace NUMINAMATH_CALUDE_special_quadratic_roots_nonnegative_l2644_264493

/-- A quadratic polynomial with two distinct roots satisfying f(x^2 + y^2) ≥ f(2xy) for all x and y -/
structure SpecialQuadratic where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  has_two_distinct_roots : ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ f r₁ = 0 ∧ f r₂ = 0
  special_property : ∀ x y : ℝ, f (x^2 + y^2) ≥ f (2*x*y)

/-- The roots of a SpecialQuadratic are non-negative -/
theorem special_quadratic_roots_nonnegative (sq : SpecialQuadratic) :
  ∃ r₁ r₂ : ℝ, r₁ ≥ 0 ∧ r₂ ≥ 0 ∧ sq.f r₁ = 0 ∧ sq.f r₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_quadratic_roots_nonnegative_l2644_264493


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_l2644_264423

theorem sqrt_two_irrational :
  ∃ (x : ℝ), Irrational x ∧ (x = Real.sqrt 2) ∧
  (∀ y : ℝ, (y = 1/3 ∨ y = 3.1415 ∨ y = -5) → ¬Irrational y) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_l2644_264423


namespace NUMINAMATH_CALUDE_unique_solution_3m_plus_4n_eq_5k_l2644_264401

theorem unique_solution_3m_plus_4n_eq_5k :
  ∀ m n k : ℕ+, 3 * m + 4 * n = 5 * k → m = 2 ∧ n = 2 ∧ k = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_3m_plus_4n_eq_5k_l2644_264401


namespace NUMINAMATH_CALUDE_prime_power_plus_144_square_l2644_264427

theorem prime_power_plus_144_square (p : ℕ) (n : ℕ) (m : ℤ) : 
  p.Prime → n > 0 → (p : ℤ)^n + 144 = m^2 → 
  ((p = 2 ∧ n = 9 ∧ m = 36) ∨ (p = 3 ∧ n = 4 ∧ m = 15)) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_plus_144_square_l2644_264427


namespace NUMINAMATH_CALUDE_cube_surface_area_from_volume_l2644_264443

theorem cube_surface_area_from_volume (V : ℝ) (h : V = 64) :
  ∃ (a : ℝ), a > 0 ∧ a^3 = V ∧ 6 * a^2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_volume_l2644_264443


namespace NUMINAMATH_CALUDE_cubic_minus_linear_l2644_264474

theorem cubic_minus_linear (n : ℕ) : ∃ n : ℕ, n^3 - n = 5814 :=
by
  -- We need to prove that there exists a natural number n such that n^3 - n = 5814
  -- given that n^3 - n is even and is the product of three consecutive natural numbers
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_l2644_264474
