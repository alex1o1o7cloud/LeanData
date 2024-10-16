import Mathlib

namespace NUMINAMATH_CALUDE_doll_price_is_five_l3514_351458

/-- Represents the inventory and financial data of Stella's antique shop --/
structure AntiqueShop where
  num_dolls : ℕ
  num_clocks : ℕ
  num_glasses : ℕ
  clock_price : ℕ
  glass_price : ℕ
  total_cost : ℕ
  total_profit : ℕ

/-- Calculates the price of each doll given the shop's data --/
def calculate_doll_price (shop : AntiqueShop) : ℕ :=
  let total_revenue := shop.total_cost + shop.total_profit
  let clock_revenue := shop.num_clocks * shop.clock_price
  let glass_revenue := shop.num_glasses * shop.glass_price
  let doll_revenue := total_revenue - clock_revenue - glass_revenue
  doll_revenue / shop.num_dolls

/-- Theorem stating that the doll price is $5 given Stella's shop data --/
theorem doll_price_is_five (shop : AntiqueShop) 
  (h1 : shop.num_dolls = 3)
  (h2 : shop.num_clocks = 2)
  (h3 : shop.num_glasses = 5)
  (h4 : shop.clock_price = 15)
  (h5 : shop.glass_price = 4)
  (h6 : shop.total_cost = 40)
  (h7 : shop.total_profit = 25) :
  calculate_doll_price shop = 5 := by
  sorry

#eval calculate_doll_price {
  num_dolls := 3,
  num_clocks := 2,
  num_glasses := 5,
  clock_price := 15,
  glass_price := 4,
  total_cost := 40,
  total_profit := 25
}

end NUMINAMATH_CALUDE_doll_price_is_five_l3514_351458


namespace NUMINAMATH_CALUDE_elevation_view_area_bounds_not_possible_area_l3514_351438

/-- The area of the elevation view of a unit cube is between 1 and √2 (inclusive) -/
theorem elevation_view_area_bounds (area : ℝ) : 
  (∃ (angle : ℝ), area = Real.cos angle + Real.sin angle) →
  1 ≤ area ∧ area ≤ Real.sqrt 2 := by
  sorry

/-- (√2 - 1) / 2 is not a possible area for the elevation view of a unit cube -/
theorem not_possible_area : 
  ¬ (∃ (angle : ℝ), (Real.sqrt 2 - 1) / 2 = Real.cos angle + Real.sin angle) := by
  sorry

end NUMINAMATH_CALUDE_elevation_view_area_bounds_not_possible_area_l3514_351438


namespace NUMINAMATH_CALUDE_no_three_subset_partition_of_positive_integers_l3514_351427

theorem no_three_subset_partition_of_positive_integers :
  ¬ ∃ (A B C : Set ℕ),
    (A ∪ B ∪ C = {n : ℕ | n > 0}) ∧
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅) ∧
    (A ≠ ∅) ∧ (B ≠ ∅) ∧ (C ≠ ∅) ∧
    (∀ x y : ℕ, x > 0 → y > 0 →
      ((x ∈ A ∧ y ∈ B) ∨ (x ∈ B ∧ y ∈ A) → x^2 - x*y + y^2 ∈ C) ∧
      ((x ∈ B ∧ y ∈ C) ∨ (x ∈ C ∧ y ∈ B) → x^2 - x*y + y^2 ∈ A) ∧
      ((x ∈ C ∧ y ∈ A) ∨ (x ∈ A ∧ y ∈ C) → x^2 - x*y + y^2 ∈ B)) :=
sorry

end NUMINAMATH_CALUDE_no_three_subset_partition_of_positive_integers_l3514_351427


namespace NUMINAMATH_CALUDE_prime_sum_of_squares_and_divisibility_l3514_351402

theorem prime_sum_of_squares_and_divisibility (p : ℕ) : 
  Prime p → 
  (∃ m n : ℤ, (p : ℤ) = m^2 + n^2 ∧ (m^3 + n^3 - 4) % p = 0) → 
  p = 2 ∨ p = 5 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_of_squares_and_divisibility_l3514_351402


namespace NUMINAMATH_CALUDE_base_conversion_and_operation_l3514_351476

-- Define the base conversions
def base9_to_10 (n : ℕ) : ℕ := n

def base4_to_10 (n : ℕ) : ℕ := n

def base8_to_10 (n : ℕ) : ℕ := n

-- Define the operation
def operation (a b c d : ℕ) : ℕ := a / b - c + d

-- Theorem statement
theorem base_conversion_and_operation :
  operation (base9_to_10 1357) (base4_to_10 100) (base8_to_10 2460) (base9_to_10 5678) = 2938 := by
  sorry

-- Additional lemmas for individual base conversions
lemma base9_1357 : base9_to_10 1357 = 1024 := by sorry
lemma base4_100 : base4_to_10 100 = 16 := by sorry
lemma base8_2460 : base8_to_10 2460 = 1328 := by sorry
lemma base9_5678 : base9_to_10 5678 = 4202 := by sorry

end NUMINAMATH_CALUDE_base_conversion_and_operation_l3514_351476


namespace NUMINAMATH_CALUDE_cubic_properties_l3514_351451

theorem cubic_properties :
  (∀ x : ℝ, x^3 > 0 → x > 0) ∧
  (∀ x : ℝ, x < 1 → x^3 < x) :=
by sorry

end NUMINAMATH_CALUDE_cubic_properties_l3514_351451


namespace NUMINAMATH_CALUDE_katy_made_65_brownies_l3514_351434

/-- The number of brownies Katy made and ate over four days --/
def brownies_problem (total : ℕ) : Prop :=
  ∃ (mon tue wed thu_before thu_after : ℕ),
    -- Monday's consumption
    mon = 5 ∧
    -- Tuesday's consumption
    tue = 2 * mon ∧
    -- Wednesday's consumption
    wed = 3 * tue ∧
    -- Remaining brownies before sharing on Thursday
    thu_before = total - (mon + tue + wed) ∧
    -- Brownies left after sharing on Thursday
    thu_after = thu_before / 2 ∧
    -- Brownies left after sharing equals Tuesday's consumption
    thu_after = tue ∧
    -- All brownies are gone after Thursday
    mon + tue + wed + thu_before = total

/-- The total number of brownies Katy made is 65 --/
theorem katy_made_65_brownies : brownies_problem 65 := by
  sorry

end NUMINAMATH_CALUDE_katy_made_65_brownies_l3514_351434


namespace NUMINAMATH_CALUDE_square_root_of_four_l3514_351449

theorem square_root_of_four : 
  {x : ℝ | x ^ 2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l3514_351449


namespace NUMINAMATH_CALUDE_unique_quadruple_solution_l3514_351422

theorem unique_quadruple_solution :
  ∃! (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
    a^2 + b^2 + c^2 + d^2 = 9 ∧
    (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 81 ∧
    a + b + c + d = 6 :=
by sorry

end NUMINAMATH_CALUDE_unique_quadruple_solution_l3514_351422


namespace NUMINAMATH_CALUDE_goshawk_eurasian_nature_reserve_birds_l3514_351464

theorem goshawk_eurasian_nature_reserve_birds (B : ℝ) (h : B > 0) :
  let hawks := 0.30 * B
  let non_hawks := B - hawks
  let paddyfield_warblers := 0.40 * non_hawks
  let other_birds := 0.35 * B
  let kingfishers := B - hawks - paddyfield_warblers - other_birds
  kingfishers / paddyfield_warblers = 0.25
:= by sorry

end NUMINAMATH_CALUDE_goshawk_eurasian_nature_reserve_birds_l3514_351464


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3514_351467

theorem inequality_solution_set (a b : ℝ) : 
  a > 2 → 
  (∀ x, ax + 3 < 2*x + b ↔ x < 0) → 
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3514_351467


namespace NUMINAMATH_CALUDE_tim_running_hours_l3514_351407

/-- Represents Tim's running schedule --/
structure RunningSchedule where
  initial_days : ℕ  -- Initial number of days Tim ran per week
  added_days : ℕ    -- Number of days Tim added to his schedule
  morning_run : ℕ   -- Hours Tim runs in the morning
  evening_run : ℕ   -- Hours Tim runs in the evening

/-- Calculates the total hours Tim runs per week --/
def total_running_hours (schedule : RunningSchedule) : ℕ :=
  (schedule.initial_days + schedule.added_days) * (schedule.morning_run + schedule.evening_run)

/-- Theorem stating that Tim's total running hours per week is 10 --/
theorem tim_running_hours :
  ∃ (schedule : RunningSchedule),
    schedule.initial_days = 3 ∧
    schedule.added_days = 2 ∧
    schedule.morning_run = 1 ∧
    schedule.evening_run = 1 ∧
    total_running_hours schedule = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_tim_running_hours_l3514_351407


namespace NUMINAMATH_CALUDE_chocolate_bar_ratio_l3514_351456

theorem chocolate_bar_ratio (total pieces : ℕ) (michael paige mandy : ℕ) : 
  total = 60 →
  paige = (total - michael) / 2 →
  mandy = 15 →
  total = michael + paige + mandy →
  michael / total = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_ratio_l3514_351456


namespace NUMINAMATH_CALUDE_amount_spent_on_sweets_l3514_351484

-- Define the initial amount
def initial_amount : ℚ := 5.10

-- Define the amount given to each friend
def amount_per_friend : ℚ := 1.00

-- Define the number of friends
def number_of_friends : ℕ := 2

-- Define the final amount left
def final_amount : ℚ := 2.05

-- Theorem to prove the amount spent on sweets
theorem amount_spent_on_sweets :
  initial_amount - (amount_per_friend * number_of_friends) - final_amount = 1.05 := by
  sorry

end NUMINAMATH_CALUDE_amount_spent_on_sweets_l3514_351484


namespace NUMINAMATH_CALUDE_cubic_quadratic_system_solution_l3514_351432

theorem cubic_quadratic_system_solution :
  ∀ a b c : ℕ,
    a^3 - b^3 - c^3 = 3*a*b*c →
    a^2 = 2*(a + b + c) →
    ((a = 4 ∧ b = 1 ∧ c = 3) ∨ (a = 4 ∧ b = 2 ∧ c = 2) ∨ (a = 4 ∧ b = 3 ∧ c = 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_quadratic_system_solution_l3514_351432


namespace NUMINAMATH_CALUDE_lower_limit_with_two_primes_l3514_351423

theorem lower_limit_with_two_primes (n : ℕ) : 
  (∃ p q : ℕ, p.Prime ∧ q.Prime ∧ p ≠ q ∧ 
   n < p ∧ p < q ∧ q ≤ 87/6 ∧
   ∀ r : ℕ, r.Prime → (n < r ∧ r ≤ 87/6) → (r = p ∨ r = q)) →
  n ≤ 79 :=
by sorry

end NUMINAMATH_CALUDE_lower_limit_with_two_primes_l3514_351423


namespace NUMINAMATH_CALUDE_megan_popsicle_consumption_l3514_351450

/-- The number of minutes between 1:00 PM and 6:20 PM -/
def time_interval : ℕ := 320

/-- The interval in minutes at which Megan eats a Popsicle -/
def popsicle_interval : ℕ := 20

/-- The number of Popsicles Megan consumes -/
def popsicles_consumed : ℕ := time_interval / popsicle_interval

theorem megan_popsicle_consumption :
  popsicles_consumed = 16 :=
by sorry

end NUMINAMATH_CALUDE_megan_popsicle_consumption_l3514_351450


namespace NUMINAMATH_CALUDE_percent_of_y_l3514_351405

theorem percent_of_y (y : ℝ) (h : y > 0) : ((4 * y) / 20 + (3 * y) / 10) / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l3514_351405


namespace NUMINAMATH_CALUDE_min_tan_angle_l3514_351404

/-- The set of complex numbers with nonnegative real and imaginary parts -/
def S : Set ℂ :=
  {z : ℂ | z.re ≥ 0 ∧ z.im ≥ 0}

/-- The condition |z^2 + 2| ≤ |z| -/
def satisfiesCondition (z : ℂ) : Prop :=
  Complex.abs (z^2 + 2) ≤ Complex.abs z

/-- The angle between a complex number and the real axis -/
noncomputable def angle (z : ℂ) : ℝ :=
  Real.arctan (z.im / z.re)

/-- The main theorem -/
theorem min_tan_angle :
  ∃ (min_tan : ℝ), min_tan = Real.sqrt 7 ∧
  ∀ z ∈ S, satisfiesCondition z →
  Real.tan (angle z) ≥ min_tan :=
sorry

end NUMINAMATH_CALUDE_min_tan_angle_l3514_351404


namespace NUMINAMATH_CALUDE_border_area_l3514_351478

/-- Given a rectangular photograph with a frame, calculate the area of the border. -/
theorem border_area (photo_height photo_width border_width : ℝ) 
  (h1 : photo_height = 8)
  (h2 : photo_width = 10)
  (h3 : border_width = 2) : 
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width = 88 := by
  sorry

#check border_area

end NUMINAMATH_CALUDE_border_area_l3514_351478


namespace NUMINAMATH_CALUDE_six_digit_multiple_of_99_l3514_351408

theorem six_digit_multiple_of_99 : ∃ n : ℕ, 
  (n ≥ 978600 ∧ n < 978700) ∧  -- Six-digit number starting with 9786
  (n % 99 = 0) ∧               -- Divisible by 99
  (n / 99 = 6039) :=           -- Quotient is 6039
by sorry

end NUMINAMATH_CALUDE_six_digit_multiple_of_99_l3514_351408


namespace NUMINAMATH_CALUDE_right_triangle_sides_l3514_351428

theorem right_triangle_sides (p Δ : ℝ) (hp : p > 0) (hΔ : Δ > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = p ∧
    a * b = 2 * Δ ∧
    c^2 = a^2 + b^2 ∧
    a = (p - (p^2 - 4*Δ)/(2*p) + ((p - (p^2 - 4*Δ)/(2*p))^2 - 8*Δ).sqrt) / 2 ∧
    b = (p - (p^2 - 4*Δ)/(2*p) - ((p - (p^2 - 4*Δ)/(2*p))^2 - 8*Δ).sqrt) / 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l3514_351428


namespace NUMINAMATH_CALUDE_pauls_money_duration_l3514_351447

/-- 
Given Paul's earnings from mowing lawns and weed eating, and his weekly spending rate,
prove that the money will last for 2 weeks.
-/
theorem pauls_money_duration (lawn_earnings weed_earnings weekly_spending : ℕ) 
  (h1 : lawn_earnings = 3)
  (h2 : weed_earnings = 3)
  (h3 : weekly_spending = 3) :
  (lawn_earnings + weed_earnings) / weekly_spending = 2 := by
  sorry

end NUMINAMATH_CALUDE_pauls_money_duration_l3514_351447


namespace NUMINAMATH_CALUDE_store_a_discount_proof_l3514_351415

/-- The additional discount percentage offered by Store A -/
def store_a_discount : ℝ := 8

/-- The full price of the smartphone at Store A -/
def store_a_full_price : ℝ := 125

/-- The full price of the smartphone at Store B -/
def store_b_full_price : ℝ := 130

/-- The additional discount percentage offered by Store B -/
def store_b_discount : ℝ := 10

/-- The price difference between Store A and Store B after discounts -/
def price_difference : ℝ := 2

theorem store_a_discount_proof :
  store_a_full_price * (1 - store_a_discount / 100) =
  store_b_full_price * (1 - store_b_discount / 100) - price_difference :=
by sorry


end NUMINAMATH_CALUDE_store_a_discount_proof_l3514_351415


namespace NUMINAMATH_CALUDE_kira_breakfast_time_l3514_351420

/-- Calculates the total time Kira spent making breakfast -/
def breakfast_time (num_sausages : ℕ) (num_eggs : ℕ) (time_per_sausage : ℕ) (time_per_egg : ℕ) : ℕ :=
  num_sausages * time_per_sausage + num_eggs * time_per_egg

/-- Proves that Kira's breakfast preparation time is 39 minutes -/
theorem kira_breakfast_time : 
  breakfast_time 3 6 5 4 = 39 := by
  sorry

end NUMINAMATH_CALUDE_kira_breakfast_time_l3514_351420


namespace NUMINAMATH_CALUDE_all_terms_are_squares_l3514_351436

/-- Definition of the n-th term of the sequence -/
def sequence_term (n : ℕ) : ℕ :=
  10^(2*n + 1) + 5 * (10^n - 1) * 10^n + 6

/-- Theorem stating that all terms in the sequence are perfect squares -/
theorem all_terms_are_squares :
  ∀ n : ℕ, ∃ k : ℕ, sequence_term n = k^2 :=
by sorry

end NUMINAMATH_CALUDE_all_terms_are_squares_l3514_351436


namespace NUMINAMATH_CALUDE_range_of_m_l3514_351442

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 3 - |x| ≥ 0}

-- Define set C parameterized by m
def C (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - 2*m - 1) < 0}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (C m ⊆ B) ↔ (m ∈ Set.Icc (-2) 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3514_351442


namespace NUMINAMATH_CALUDE_prob_sector_1_eq_prob_sector_8_prob_consecutive_sectors_correct_l3514_351414

-- Define the number of sectors and the number of played sectors
def total_sectors : ℕ := 13
def played_sectors : ℕ := 6

-- Define a function to calculate the probability of a specific sector being played
def prob_sector_played (sector : ℕ) : ℚ :=
  played_sectors / total_sectors

-- Theorem for part (a)
theorem prob_sector_1_eq_prob_sector_8 :
  prob_sector_played 1 = prob_sector_played 8 :=
sorry

-- Define a function to calculate the probability of sectors 1 to 6 being played consecutively
def prob_consecutive_sectors : ℚ :=
  (7^5 : ℚ) / (13^6 : ℚ)

-- Theorem for part (b)
theorem prob_consecutive_sectors_correct :
  prob_consecutive_sectors = (7^5 : ℚ) / (13^6 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_prob_sector_1_eq_prob_sector_8_prob_consecutive_sectors_correct_l3514_351414


namespace NUMINAMATH_CALUDE_limit_between_exponentials_l3514_351403

theorem limit_between_exponentials (a : ℝ) (ha : a > 0) :
  Real.exp a < (Real.exp (a + 1)) / (Real.exp 1 - 1) ∧
  (Real.exp (a + 1)) / (Real.exp 1 - 1) < Real.exp (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_limit_between_exponentials_l3514_351403


namespace NUMINAMATH_CALUDE_percentage_d_grades_l3514_351475

def scores : List ℕ := [89, 65, 55, 96, 73, 93, 82, 70, 77, 65, 81, 79, 67, 85, 88, 61, 84, 71, 73, 90]

def is_d_grade (score : ℕ) : Bool :=
  65 ≤ score ∧ score ≤ 75

def count_d_grades (scores : List ℕ) : ℕ :=
  scores.filter is_d_grade |>.length

theorem percentage_d_grades :
  (count_d_grades scores : ℚ) / scores.length * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_percentage_d_grades_l3514_351475


namespace NUMINAMATH_CALUDE_integer_between_sqrt2_and_sqrt12_l3514_351481

theorem integer_between_sqrt2_and_sqrt12 (a : ℤ) : 
  (Real.sqrt 2 < a) ∧ (a < Real.sqrt 12) → (a = 2 ∨ a = 3) := by
  sorry

end NUMINAMATH_CALUDE_integer_between_sqrt2_and_sqrt12_l3514_351481


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l3514_351488

theorem triangle_abc_proof (A B C : Real) (a b c : Real) : 
  A = π / 6 →
  (1 + Real.sqrt 3) * c = 2 * b →
  c * b * Real.cos C = 1 + Real.sqrt 3 →
  C = π / 4 ∧ 
  a = Real.sqrt 2 ∧ 
  b = 1 + Real.sqrt 3 ∧ 
  c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l3514_351488


namespace NUMINAMATH_CALUDE_max_value_of_g_l3514_351474

-- Define the function g(x)
def g (x : ℝ) : ℝ := 4 * x - x^3

-- State the theorem
theorem max_value_of_g :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧
  (∀ (y : ℝ), 0 ≤ y ∧ y ≤ 2 → g y ≤ g x) ∧
  g x = 16 * Real.sqrt 3 / 9 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l3514_351474


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3514_351417

theorem max_value_of_expression (x : ℝ) :
  ∃ (max : ℝ), max = (1 / 4 : ℝ) ∧ ∀ y : ℝ, 10^y - 100^y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3514_351417


namespace NUMINAMATH_CALUDE_equation_solution_l3514_351494

theorem equation_solution : ∃ x : ℝ, 0.05 * x + 0.12 * (30 + x) + 0.02 * (50 + 2 * x) = 20 ∧ x = 220 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3514_351494


namespace NUMINAMATH_CALUDE_marbles_selection_theorem_l3514_351437

def total_marbles : ℕ := 9
def marbles_to_choose : ℕ := 4
def blue_marbles : ℕ := 2

theorem marbles_selection_theorem :
  (Nat.choose total_marbles marbles_to_choose) -
  (Nat.choose (total_marbles - blue_marbles) marbles_to_choose) = 91 := by
  sorry

end NUMINAMATH_CALUDE_marbles_selection_theorem_l3514_351437


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3514_351433

theorem cube_volume_problem (a : ℝ) : 
  a > 0 → 
  (a + 2) * (a + 1) * (a - 1) = a^3 - 6 → 
  a^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3514_351433


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l3514_351471

/-- The time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 160)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 215) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 :=
by sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l3514_351471


namespace NUMINAMATH_CALUDE_corrected_mean_l3514_351416

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 ∧ original_mean = 36 ∧ incorrect_value = 23 ∧ correct_value = 34 →
  (n : ℚ) * original_mean + (correct_value - incorrect_value) = n * 36.22 :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_l3514_351416


namespace NUMINAMATH_CALUDE_parabola_point_value_l3514_351452

/-- 
Given a parabola y = -x^2 + bx + c that passes through the point (-2, 3),
prove that 2c - 4b - 9 = 5
-/
theorem parabola_point_value (b c : ℝ) 
  (h : 3 = -(-2)^2 + b*(-2) + c) : 2*c - 4*b - 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_value_l3514_351452


namespace NUMINAMATH_CALUDE_h_equation_l3514_351461

/-- Given the equation 4x^4 + 2x^2 - 5x + 1 + h(x) = x^3 - 3x^2 + 2x - 4,
    prove that h(x) = -4x^4 + x^3 - 5x^2 + 7x - 5 -/
theorem h_equation (x : ℝ) (h : ℝ → ℝ) 
    (eq : 4 * x^4 + 2 * x^2 - 5 * x + 1 + h x = x^3 - 3 * x^2 + 2 * x - 4) : 
  h x = -4 * x^4 + x^3 - 5 * x^2 + 7 * x - 5 := by
  sorry

end NUMINAMATH_CALUDE_h_equation_l3514_351461


namespace NUMINAMATH_CALUDE_server_data_processing_l3514_351440

/-- Represents the data processing rate in megabytes per minute -/
def processing_rate : ℝ := 150

/-- Represents the time period in hours -/
def time_period : ℝ := 12

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℝ := 60

/-- Represents the number of megabytes in a gigabyte -/
def mb_per_gb : ℝ := 1000

/-- Theorem stating that the server processes 108 gigabytes in 12 hours -/
theorem server_data_processing :
  (processing_rate * time_period * minutes_per_hour) / mb_per_gb = 108 := by
  sorry

end NUMINAMATH_CALUDE_server_data_processing_l3514_351440


namespace NUMINAMATH_CALUDE_least_palindrome_addition_l3514_351472

/-- A function that checks if a natural number is a palindrome -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The starting number in our problem -/
def startNumber : ℕ := 250000

/-- The least number to be added to create a palindrome -/
def leastAddition : ℕ := 52

/-- Theorem stating that 52 is the least natural number that,
    when added to 250000, results in a palindrome -/
theorem least_palindrome_addition :
  (∀ k : ℕ, k < leastAddition → ¬isPalindrome (startNumber + k)) ∧
  isPalindrome (startNumber + leastAddition) := by sorry

end NUMINAMATH_CALUDE_least_palindrome_addition_l3514_351472


namespace NUMINAMATH_CALUDE_max_remainder_division_l3514_351430

theorem max_remainder_division (n : ℕ) : 
  (n % 6 < 6) → (n / 6 = 18) → (n % 6 = 5) → n = 113 := by
  sorry

end NUMINAMATH_CALUDE_max_remainder_division_l3514_351430


namespace NUMINAMATH_CALUDE_bruce_initial_eggs_l3514_351495

theorem bruce_initial_eggs (bruce_final : ℕ) (eggs_lost : ℕ) : 
  bruce_final = 5 → eggs_lost = 70 → bruce_final + eggs_lost = 75 := by
  sorry

end NUMINAMATH_CALUDE_bruce_initial_eggs_l3514_351495


namespace NUMINAMATH_CALUDE_birds_on_fence_l3514_351445

theorem birds_on_fence : 
  let initial_sparrows : ℕ := 4
  let initial_storks : ℕ := 46
  let pigeons_joined : ℕ := 6
  let sparrows_left : ℕ := 3
  let storks_left : ℕ := 5
  let swans_came : ℕ := 8
  let ducks_came : ℕ := 2
  
  let total_birds : ℕ := 
    (initial_sparrows + initial_storks + pigeons_joined - sparrows_left - storks_left + swans_came + ducks_came)
  
  total_birds = 58 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3514_351445


namespace NUMINAMATH_CALUDE_multiplication_subtraction_difference_l3514_351409

theorem multiplication_subtraction_difference (x : ℝ) (h : x = 5) : ∃ n : ℝ, 3 * x = (16 - x) + n ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_difference_l3514_351409


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l3514_351413

/-- Proves that a cement mixture with given proportions weighs 48 pounds -/
theorem cement_mixture_weight (sand_fraction : ℚ) (water_fraction : ℚ) (gravel_weight : ℚ) :
  sand_fraction = 1/3 →
  water_fraction = 1/2 →
  gravel_weight = 8 →
  sand_fraction + water_fraction + gravel_weight / (sand_fraction + water_fraction + gravel_weight) = 1 →
  sand_fraction + water_fraction + gravel_weight = 48 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l3514_351413


namespace NUMINAMATH_CALUDE_permutation_remainder_l3514_351469

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid permutations of the 18-character string -/
def N : ℕ := sorry

/-- The sum of valid permutations for different arrangements -/
def permutation_sum : ℕ :=
  (choose 5 0) * (choose 5 0) * (choose 5 1) +
  (choose 5 1) * (choose 5 1) * (choose 5 2) +
  (choose 5 2) * (choose 5 2) * (choose 5 3) +
  (choose 5 3) * (choose 5 3) * (choose 5 4)

theorem permutation_remainder :
  N ≡ 755 [MOD 1000] :=
sorry

end NUMINAMATH_CALUDE_permutation_remainder_l3514_351469


namespace NUMINAMATH_CALUDE_smallest_value_theorem_l3514_351401

theorem smallest_value_theorem (n : ℕ+) : 
  (n : ℝ) / 2 + 18 / (n : ℝ) ≥ 6 ∧ 
  ((6 : ℕ+) : ℝ) / 2 + 18 / ((6 : ℕ+) : ℝ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_theorem_l3514_351401


namespace NUMINAMATH_CALUDE_students_in_same_group_l3514_351489

/-- The number of interest groups -/
def num_groups : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 2

/-- The probability of a student joining any specific group -/
def prob_join_group : ℚ := 1 / num_groups

/-- The probability of both students being in the same group -/
def prob_same_group : ℚ := 1 / num_groups

theorem students_in_same_group : 
  prob_same_group = 1 / num_groups :=
sorry

end NUMINAMATH_CALUDE_students_in_same_group_l3514_351489


namespace NUMINAMATH_CALUDE_machine_selling_price_l3514_351441

/-- Calculates the selling price of a machine given its costs and desired profit percentage --/
def selling_price (purchase_price repair_cost transportation_charges profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transportation_charges
  let profit := total_cost * profit_percentage / 100
  total_cost + profit

/-- Theorem stating that the selling price of the machine is 27000 --/
theorem machine_selling_price :
  selling_price 12000 5000 1000 50 = 27000 := by
  sorry

end NUMINAMATH_CALUDE_machine_selling_price_l3514_351441


namespace NUMINAMATH_CALUDE_sum_of_lg2_and_lg5_power_of_8_two_thirds_l3514_351412

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem 1: lg2 + lg5 = 1
theorem sum_of_lg2_and_lg5 : lg 2 + lg 5 = 1 := by sorry

-- Theorem 2: 8^(2/3) = 4
theorem power_of_8_two_thirds : (8 : ℝ) ^ (2/3) = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_lg2_and_lg5_power_of_8_two_thirds_l3514_351412


namespace NUMINAMATH_CALUDE_negation_equivalence_angle_sine_equivalence_l3514_351424

-- Define the proposition for the first part
def P (x : ℝ) : Prop := x^2 - x > 0

-- Theorem for the first part
theorem negation_equivalence : (¬ ∃ x, P x) ↔ (∀ x, ¬(P x)) := by sorry

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Theorem for the second part
theorem angle_sine_equivalence (t : Triangle) : t.A > t.B ↔ Real.sin t.A > Real.sin t.B := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_angle_sine_equivalence_l3514_351424


namespace NUMINAMATH_CALUDE_problem_solution_l3514_351431

theorem problem_solution : ∃ x : ℝ, 550 - (x / 20.8) = 545 ∧ x = 104 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3514_351431


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3514_351400

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3514_351400


namespace NUMINAMATH_CALUDE_even_number_selection_l3514_351480

theorem even_number_selection (p : ℝ) (n : ℕ) 
  (h_p : p = 0.5) 
  (h_n : n = 4) : 
  1 - p^n ≥ 0.9 := by
sorry

end NUMINAMATH_CALUDE_even_number_selection_l3514_351480


namespace NUMINAMATH_CALUDE_function_properties_l3514_351411

-- Define the function f
def f : ℝ → ℝ := λ x ↦ x^2 + x + 1

-- Define the function g
def g (a : ℝ) : ℝ → ℝ := λ x ↦ |f x - a * x + 3|

-- Theorem statement
theorem function_properties :
  (∀ x : ℝ, f (1 - x) = x^2 - 3*x + 3) ∧
  (∀ a : ℝ, (∀ x y : ℝ, 1 ≤ x ∧ x < y ∧ y ≤ 3 → g a x < g a y) →
    a ∈ Set.Iic 3 ∪ Set.Ici 6) := by
  sorry


end NUMINAMATH_CALUDE_function_properties_l3514_351411


namespace NUMINAMATH_CALUDE_B_power_97_l3514_351477

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_97 : B^97 = B := by sorry

end NUMINAMATH_CALUDE_B_power_97_l3514_351477


namespace NUMINAMATH_CALUDE_probability_three_tails_one_head_probability_three_tails_one_head_proof_l3514_351499

/-- The probability of getting exactly three tails and one head when four fair coins are tossed simultaneously -/
theorem probability_three_tails_one_head : ℚ :=
  1 / 4

/-- Proof that the probability of getting exactly three tails and one head when four fair coins are tossed simultaneously is 1/4 -/
theorem probability_three_tails_one_head_proof :
  probability_three_tails_one_head = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_tails_one_head_probability_three_tails_one_head_proof_l3514_351499


namespace NUMINAMATH_CALUDE_lateral_surface_area_of_rotated_square_l3514_351443

theorem lateral_surface_area_of_rotated_square (Q : ℝ) (h : Q > 0) :
  let side_length := Real.sqrt Q
  let radius := side_length
  let height := side_length
  let lateral_surface_area := 2 * Real.pi * radius * height
  lateral_surface_area = 2 * Real.pi * Q :=
by sorry

end NUMINAMATH_CALUDE_lateral_surface_area_of_rotated_square_l3514_351443


namespace NUMINAMATH_CALUDE_next_term_is_2500x4_l3514_351435

def geometric_sequence (x : ℝ) : ℕ → ℝ
  | 0 => 4
  | 1 => 20 * x
  | 2 => 100 * x^2
  | 3 => 500 * x^3
  | (n + 4) => geometric_sequence x n * 5 * x

theorem next_term_is_2500x4 (x : ℝ) : geometric_sequence x 4 = 2500 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_next_term_is_2500x4_l3514_351435


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3514_351446

-- Define the quadratic equation
def quadratic_equation (x k : ℝ) : Prop :=
  x^2 + 2*x - k = 0

-- Define the condition for real roots
def has_real_roots (k : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation x k

-- Theorem statement
theorem quadratic_real_roots_condition (k : ℝ) :
  has_real_roots k ↔ k ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3514_351446


namespace NUMINAMATH_CALUDE_tank_depth_is_six_l3514_351459

-- Define the tank dimensions and plastering cost
def tankLength : ℝ := 25
def tankWidth : ℝ := 12
def plasteringCostPerSqM : ℝ := 0.45
def totalPlasteringCost : ℝ := 334.8

-- Define the function to calculate the total surface area to be plastered
def surfaceArea (depth : ℝ) : ℝ :=
  tankLength * tankWidth + 2 * (tankLength * depth) + 2 * (tankWidth * depth)

-- Theorem statement
theorem tank_depth_is_six :
  ∃ (depth : ℝ), plasteringCostPerSqM * surfaceArea depth = totalPlasteringCost ∧ depth = 6 :=
sorry

end NUMINAMATH_CALUDE_tank_depth_is_six_l3514_351459


namespace NUMINAMATH_CALUDE_weight_relationships_l3514_351455

/-- Given the weights of Brenda, Mel, and Tom, prove their relationships and specific weights. -/
theorem weight_relationships (B M T : ℕ) : 
  B = 3 * M + 10 →  -- Brenda weighs 10 pounds more than 3 times Mel's weight
  T = 2 * M →       -- Tom weighs twice as much as Mel
  2 * T = B →       -- Tom weighs half as much as Brenda
  B = 220 →         -- Brenda weighs 220 pounds
  M = 70 ∧ T = 140  -- Prove that Mel weighs 70 pounds and Tom weighs 140 pounds
:= by sorry

end NUMINAMATH_CALUDE_weight_relationships_l3514_351455


namespace NUMINAMATH_CALUDE_binomial_sum_cubes_l3514_351454

theorem binomial_sum_cubes (x y : ℤ) :
  (x^4 + 9*x*y^3)^3 + (-3*x^3*y - 9*y^4)^3 = x^12 - 729*y^12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_cubes_l3514_351454


namespace NUMINAMATH_CALUDE_cristobal_beatrix_pages_difference_l3514_351425

theorem cristobal_beatrix_pages_difference (beatrix_pages cristobal_extra_pages : ℕ) 
  (h1 : beatrix_pages = 704)
  (h2 : cristobal_extra_pages = 1423) :
  (beatrix_pages + cristobal_extra_pages) - (3 * beatrix_pages) = 15 := by
  sorry

end NUMINAMATH_CALUDE_cristobal_beatrix_pages_difference_l3514_351425


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3514_351406

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  (g 1 = 1) ∧ 
  (∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y)

/-- The main theorem stating that the function g(x) = 4^x - 3^x satisfies the functional equation -/
theorem functional_equation_solution :
  ∃ g : ℝ → ℝ, FunctionalEquation g ∧ (∀ x : ℝ, g x = 4^x - 3^x) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3514_351406


namespace NUMINAMATH_CALUDE_box_number_problem_l3514_351473

theorem box_number_problem (a b c d e : ℕ) 
  (sum_all : a + b + c + d + e = 35)
  (sum_first_three : a + b + c = 22)
  (sum_last_three : c + d + e = 25)
  (first_box : a = 3)
  (last_box : e = 4) :
  b * d = 63 := by
  sorry

end NUMINAMATH_CALUDE_box_number_problem_l3514_351473


namespace NUMINAMATH_CALUDE_equation_solution_l3514_351486

theorem equation_solution : ∃ x : ℝ, (x^2 + 3*x + 4) / (x^2 - 3*x + 2) = x + 6 := by
  use 1
  -- Proof goes here
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l3514_351486


namespace NUMINAMATH_CALUDE_sequence_property_l3514_351487

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i => a (i + 1))

theorem sequence_property (a : ℕ → ℕ) 
  (h : ∀ n : ℕ, n > 0 → sequence_sum a n = 2 * a n - n) :
  (a 1 = 1 ∧ a 2 = 3 ∧ a 3 = 7) ∧
  (∀ n : ℕ, n > 0 → a n = 2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l3514_351487


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l3514_351470

theorem unique_solution_cube_equation :
  ∃! (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 ∧ x = 27 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l3514_351470


namespace NUMINAMATH_CALUDE_special_function_at_2009_l3514_351410

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧
  (∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 2))

/-- The main theorem -/
theorem special_function_at_2009 (f : ℝ → ℝ) (h : special_function f) : f 2009 = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_2009_l3514_351410


namespace NUMINAMATH_CALUDE_quilt_material_requirement_l3514_351426

theorem quilt_material_requirement (material_per_quilt : ℝ) : 
  (7 * material_per_quilt = 21) ∧ (12 * material_per_quilt = 36) :=
by sorry

end NUMINAMATH_CALUDE_quilt_material_requirement_l3514_351426


namespace NUMINAMATH_CALUDE_mod_difference_of_powers_l3514_351485

theorem mod_difference_of_powers (n : ℕ) : 45^1537 - 25^1537 ≡ 4 [MOD 8] := by
  sorry

end NUMINAMATH_CALUDE_mod_difference_of_powers_l3514_351485


namespace NUMINAMATH_CALUDE_andy_gave_five_to_brother_l3514_351479

/-- The number of cookies Andy had at the start -/
def initial_cookies : ℕ := 72

/-- The number of cookies Andy ate -/
def andy_ate : ℕ := 3

/-- The number of players in Andy's basketball team -/
def team_size : ℕ := 8

/-- The number of cookies taken by the i-th player -/
def player_cookies (i : ℕ) : ℕ := 2 * i - 1

/-- The sum of cookies taken by all team members -/
def team_total : ℕ := (team_size * (player_cookies 1 + player_cookies team_size)) / 2

/-- The number of cookies Andy gave to his little brother -/
def brother_cookies : ℕ := initial_cookies - andy_ate - team_total

theorem andy_gave_five_to_brother : brother_cookies = 5 := by
  sorry

end NUMINAMATH_CALUDE_andy_gave_five_to_brother_l3514_351479


namespace NUMINAMATH_CALUDE_gcd_2183_1947_l3514_351463

theorem gcd_2183_1947 : Nat.gcd 2183 1947 = 59 := by sorry

end NUMINAMATH_CALUDE_gcd_2183_1947_l3514_351463


namespace NUMINAMATH_CALUDE_max_log_sum_l3514_351496

theorem max_log_sum (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 4 * x + y = 40) :
  (Real.log x + Real.log y) ≤ 2 * Real.log 10 :=
sorry

end NUMINAMATH_CALUDE_max_log_sum_l3514_351496


namespace NUMINAMATH_CALUDE_equal_even_odd_probability_l3514_351460

/-- The number of dice being rolled -/
def n : ℕ := 8

/-- The probability of rolling an even number on a single die -/
def p_even : ℚ := 1/2

/-- The probability of rolling an odd number on a single die -/
def p_odd : ℚ := 1/2

/-- The number of ways to choose half of the dice -/
def ways_to_choose : ℕ := n.choose (n/2)

theorem equal_even_odd_probability :
  (ways_to_choose : ℚ) * p_even^(n/2) * p_odd^(n/2) = 35/128 := by sorry

end NUMINAMATH_CALUDE_equal_even_odd_probability_l3514_351460


namespace NUMINAMATH_CALUDE_a_4_equals_20_l3514_351483

def sequence_a (n : ℕ) : ℕ := n^2 + n

theorem a_4_equals_20 : sequence_a 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_20_l3514_351483


namespace NUMINAMATH_CALUDE_expression_evaluation_l3514_351453

theorem expression_evaluation (x y : ℚ) (hx : x = -2) (hy : y = 1) :
  (-2 * x + x + 3 * y) - 2 * (-x^2 - 2 * x + 1/2 * y) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3514_351453


namespace NUMINAMATH_CALUDE_parallel_iff_perpendicular_iff_l3514_351465

-- Define the lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := x + m * y + 6 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * m * y + 2 * m = 0

-- Define parallel lines
def parallel (m : ℝ) : Prop := 
  ∀ x y z w, l1 m x y ∧ l2 m z w → (x - z) * (m - 2) = m * (y - w)

-- Define perpendicular lines
def perpendicular (m : ℝ) : Prop := 
  ∀ x y z w, l1 m x y ∧ l2 m z w → (x - z) * (z - x) + m * (y - w) * (w - y) = 0

-- Theorem for parallel lines
theorem parallel_iff : 
  ∀ m : ℝ, parallel m ↔ m = 0 ∨ m = 5 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_iff : 
  ∀ m : ℝ, perpendicular m ↔ m = -1 ∨ m = 2/3 :=
sorry

end NUMINAMATH_CALUDE_parallel_iff_perpendicular_iff_l3514_351465


namespace NUMINAMATH_CALUDE_triangle_cosine_identities_l3514_351421

theorem triangle_cosine_identities (α β γ : Real) 
  (h : α + β + γ = Real.pi) : 
  (Real.cos (2 * α) + Real.cos (2 * β) + Real.cos (2 * γ) + 4 * Real.cos α * Real.cos β * Real.cos γ + 1 = 0) ∧
  (Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 + 2 * Real.cos α * Real.cos β * Real.cos γ = 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_identities_l3514_351421


namespace NUMINAMATH_CALUDE_sum_even_probability_l3514_351482

/-- Represents a wheel in the game -/
structure Wheel where
  probability_even : ℝ

/-- The game with two wheels -/
structure Game where
  wheel_a : Wheel
  wheel_b : Wheel

/-- A fair wheel has equal probability of landing on even or odd numbers -/
def is_fair (w : Wheel) : Prop := w.probability_even = 1/2

/-- Probability of the sum of two wheels being even -/
def prob_sum_even (g : Game) : ℝ :=
  g.wheel_a.probability_even * g.wheel_b.probability_even +
  (1 - g.wheel_a.probability_even) * (1 - g.wheel_b.probability_even)

theorem sum_even_probability (g : Game) 
  (h1 : is_fair g.wheel_a) 
  (h2 : g.wheel_b.probability_even = 2/3) : 
  prob_sum_even g = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_probability_l3514_351482


namespace NUMINAMATH_CALUDE_last_number_theorem_l3514_351444

theorem last_number_theorem (a b c d : ℝ) 
  (h1 : (a + b + c) / 3 = 6)
  (h2 : (b + c + d) / 3 = 5)
  (h3 : a + d = 11) :
  d = 4 := by
sorry

end NUMINAMATH_CALUDE_last_number_theorem_l3514_351444


namespace NUMINAMATH_CALUDE_coordinate_points_existence_l3514_351468

theorem coordinate_points_existence :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (a₁ - b₁ = 4 ∧ a₁^2 + b₁^2 = 30 ∧ a₁ * b₁ = c₁) ∧
    (a₂ - b₂ = 4 ∧ a₂^2 + b₂^2 = 30 ∧ a₂ * b₂ = c₂) ∧
    a₁ = 2 + Real.sqrt 11 ∧
    b₁ = -2 + Real.sqrt 11 ∧
    c₁ = -15 + 4 * Real.sqrt 11 ∧
    a₂ = 2 - Real.sqrt 11 ∧
    b₂ = -2 - Real.sqrt 11 ∧
    c₂ = -15 - 4 * Real.sqrt 11 :=
by sorry

end NUMINAMATH_CALUDE_coordinate_points_existence_l3514_351468


namespace NUMINAMATH_CALUDE_smallest_p_for_integer_sqrt_l3514_351462

theorem smallest_p_for_integer_sqrt : ∃ (p : ℕ), p > 0 ∧ 
  (∀ (q : ℕ), q > 0 → q < p → ¬ (∃ (n : ℕ), n ^ 2 = 2^3 * 5 * q)) ∧
  (∃ (n : ℕ), n ^ 2 = 2^3 * 5 * p) ∧
  p = 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_p_for_integer_sqrt_l3514_351462


namespace NUMINAMATH_CALUDE_jane_ribbons_theorem_l3514_351492

/-- The number of dresses Jane sews per day in the first week -/
def dresses_per_day_week1 : ℕ := 2

/-- The number of days Jane sews in the first week -/
def days_week1 : ℕ := 7

/-- The number of dresses Jane sews per day in the second period -/
def dresses_per_day_week2 : ℕ := 3

/-- The number of days Jane sews in the second period -/
def days_week2 : ℕ := 2

/-- The number of ribbons Jane adds to each dress -/
def ribbons_per_dress : ℕ := 2

/-- The total number of ribbons Jane uses -/
def total_ribbons : ℕ := 40

theorem jane_ribbons_theorem : 
  (dresses_per_day_week1 * days_week1 + dresses_per_day_week2 * days_week2) * ribbons_per_dress = total_ribbons := by
  sorry

end NUMINAMATH_CALUDE_jane_ribbons_theorem_l3514_351492


namespace NUMINAMATH_CALUDE_reflect_F_coordinates_l3514_351439

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point F -/
def F : ℝ × ℝ := (3, 3)

theorem reflect_F_coordinates :
  (reflect_x (reflect_y F)) = (-3, -3) := by sorry

end NUMINAMATH_CALUDE_reflect_F_coordinates_l3514_351439


namespace NUMINAMATH_CALUDE_weird_calculator_theorem_l3514_351418

/-- Represents the calculator operations -/
inductive Operation
| DSharp : Operation  -- doubles and adds 1
| DFlat  : Operation  -- doubles and subtracts 1

/-- Applies a single operation to a number -/
def apply_operation (op : Operation) (x : ℕ) : ℕ :=
  match op with
  | Operation.DSharp => 2 * x + 1
  | Operation.DFlat  => 2 * x - 1

/-- Applies a sequence of operations to a number -/
def apply_sequence (ops : List Operation) (x : ℕ) : ℕ :=
  match ops with
  | [] => x
  | op :: rest => apply_sequence rest (apply_operation op x)

/-- The set of all possible results after 8 operations starting from 1 -/
def possible_results : Set ℕ :=
  {n | ∃ (ops : List Operation), ops.length = 8 ∧ apply_sequence ops 1 = n}

theorem weird_calculator_theorem :
  possible_results = {n : ℕ | n < 512 ∧ n % 2 = 1} :=
sorry

end NUMINAMATH_CALUDE_weird_calculator_theorem_l3514_351418


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3514_351490

theorem sufficient_but_not_necessary (a : ℝ) :
  (((1 / a) > (1 / 4)) → (∀ x : ℝ, a * x^2 + a * x + 1 > 0)) ∧
  (∃ a : ℝ, (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧ ((1 / a) ≤ (1 / 4))) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3514_351490


namespace NUMINAMATH_CALUDE_power_five_mod_seven_l3514_351491

theorem power_five_mod_seven : 5^207 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_five_mod_seven_l3514_351491


namespace NUMINAMATH_CALUDE_pool_depths_l3514_351457

/-- Pool depths problem -/
theorem pool_depths (john sarah susan mike : ℕ) : 
  john = 2 * sarah + 5 →  -- John's pool is 5 feet deeper than 2 times Sarah's pool
  john = 15 →  -- John's pool is 15 feet deep
  susan = john + sarah - 3 →  -- Susan's pool is 3 feet shallower than the sum of John's and Sarah's pool depths
  mike = john + sarah + susan + 4 →  -- Mike's pool is 4 feet deeper than the combined depth of John's, Sarah's, and Susan's pools
  sarah = 5 ∧ susan = 17 ∧ mike = 41 := by
  sorry

end NUMINAMATH_CALUDE_pool_depths_l3514_351457


namespace NUMINAMATH_CALUDE_no_triple_squares_l3514_351448

theorem no_triple_squares (n : ℕ+) : 
  ¬(∃ (a b c : ℕ), (2 * n.val^2 + 1 = a^2) ∧ (3 * n.val^2 + 1 = b^2) ∧ (6 * n.val^2 + 1 = c^2)) :=
by sorry

end NUMINAMATH_CALUDE_no_triple_squares_l3514_351448


namespace NUMINAMATH_CALUDE_divisor_calculation_l3514_351419

theorem divisor_calculation (quotient dividend : ℚ) (h1 : quotient = -5/16) (h2 : dividend = -5/2) :
  dividend / quotient = 8 := by
  sorry

end NUMINAMATH_CALUDE_divisor_calculation_l3514_351419


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l3514_351429

theorem max_value_of_trig_function :
  ∀ x : ℝ, (π / (1 + Real.tan x ^ 2)) ≤ π :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l3514_351429


namespace NUMINAMATH_CALUDE_eighth_group_selection_l3514_351493

/-- Systematic sampling from a population -/
def systematicSampling (totalPopulation : ℕ) (numGroups : ℕ) (firstGroupSelection : ℕ) (targetGroup : ℕ) : ℕ :=
  (targetGroup - 1) * (totalPopulation / numGroups) + firstGroupSelection

/-- Theorem: In a systematic sampling of 30 groups from 480 students, 
    if the selected number from the first group is 5, 
    then the selected number from the eighth group is 117. -/
theorem eighth_group_selection :
  systematicSampling 480 30 5 8 = 117 := by
  sorry

end NUMINAMATH_CALUDE_eighth_group_selection_l3514_351493


namespace NUMINAMATH_CALUDE_two_integers_sum_and_lcm_l3514_351466

theorem two_integers_sum_and_lcm : ∃ (m n : ℕ), 
  m > 0 ∧ n > 0 ∧ m + n = 60 ∧ Nat.lcm m n = 273 ∧ m = 21 ∧ n = 39 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_sum_and_lcm_l3514_351466


namespace NUMINAMATH_CALUDE_divisors_of_n_squared_less_than_n_not_dividing_n_l3514_351497

def n : ℕ := 2^33 * 5^21

-- Function to count divisors of a number
def count_divisors (m : ℕ) : ℕ := sorry

-- Function to count divisors of m less than n
def count_divisors_less_than (m n : ℕ) : ℕ := sorry

theorem divisors_of_n_squared_less_than_n_not_dividing_n :
  count_divisors_less_than (n^2) n - count_divisors n = 692 := by sorry

end NUMINAMATH_CALUDE_divisors_of_n_squared_less_than_n_not_dividing_n_l3514_351497


namespace NUMINAMATH_CALUDE_prob_at_least_one_box_same_color_exact_l3514_351498

/-- Represents the number of friends -/
def num_friends : ℕ := 4

/-- Represents the number of blocks each friend has -/
def num_blocks : ℕ := 6

/-- Represents the number of boxes -/
def num_boxes : ℕ := 6

/-- Represents the probability of a specific color being placed in a specific box by one friend -/
def prob_color_in_box : ℚ := 1 / num_blocks

/-- Represents the probability of three friends placing the same color in a specific box -/
def prob_three_same_color : ℚ := prob_color_in_box ^ 3

/-- Represents the probability of at least one box having all blocks of the same color -/
def prob_at_least_one_box_same_color : ℚ := 1 - (1 - num_blocks * prob_three_same_color) ^ num_boxes

theorem prob_at_least_one_box_same_color_exact : 
  prob_at_least_one_box_same_color = 517 / 7776 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_box_same_color_exact_l3514_351498
