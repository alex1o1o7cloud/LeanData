import Mathlib

namespace sphere_volume_from_cube_surface_l253_25394

theorem sphere_volume_from_cube_surface (L : ℝ) : 
  let cube_side : ℝ := 3
  let cube_surface_area : ℝ := 6 * cube_side^2
  let sphere_radius : ℝ := (cube_surface_area / (4 * Real.pi))^(1/2)
  let sphere_volume : ℝ := (4/3) * Real.pi * sphere_radius^3
  sphere_volume = L * (15^(1/2)) / (Real.pi^(1/2)) →
  L = 108 * (5^(1/2)) / 5 := by
sorry

end sphere_volume_from_cube_surface_l253_25394


namespace power_of_105_l253_25319

theorem power_of_105 (n : ℕ) : 
  (105 : ℕ) ^ n = 21 * 25 * 45 * 49 ↔ n ≥ 3 :=
sorry

end power_of_105_l253_25319


namespace quadratic_root_negative_reciprocal_l253_25330

/-- For a quadratic equation ax^2 + bx + c = 0, if one root is the negative reciprocal of the other, then c = -a. -/
theorem quadratic_root_negative_reciprocal (a b c : ℝ) (α β : ℝ) : 
  a ≠ 0 →  -- Ensure the equation is quadratic
  a * α^2 + b * α + c = 0 →  -- α is a root
  a * β^2 + b * β + c = 0 →  -- β is a root
  β = -1 / α →  -- One root is the negative reciprocal of the other
  c = -a := by
sorry

end quadratic_root_negative_reciprocal_l253_25330


namespace quadratic_root_product_l253_25379

theorem quadratic_root_product (x₁ x₂ : ℝ) : 
  (x₁^2 - 4*x₁ + 1 = 0) → 
  (x₂^2 - 4*x₂ + 1 = 0) → 
  x₁ * x₂ = 1 := by
sorry

end quadratic_root_product_l253_25379


namespace m_range_l253_25336

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

-- State the theorem
theorem m_range (m : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, f m x > -m + 2) → m > 3 := by
  sorry


end m_range_l253_25336


namespace profit_at_45_price_for_1200_profit_l253_25395

-- Define the parameters of the problem
def cost_price : ℝ := 30
def initial_price : ℝ := 40
def initial_sales : ℝ := 80
def price_change : ℝ := 1
def sales_change : ℝ := 2
def max_price : ℝ := 55

-- Define the sales volume as a function of price
def sales_volume (price : ℝ) : ℝ :=
  initial_sales - sales_change * (price - initial_price)

-- Define the daily profit as a function of price
def daily_profit (price : ℝ) : ℝ :=
  (price - cost_price) * sales_volume price

-- Theorem for part 1
theorem profit_at_45 :
  daily_profit 45 = 1050 := by sorry

-- Theorem for part 2
theorem price_for_1200_profit :
  ∃ (price : ℝ), price ≤ max_price ∧ daily_profit price = 1200 ∧ price = 50 := by sorry

end profit_at_45_price_for_1200_profit_l253_25395


namespace binomial_floor_divisibility_l253_25399

theorem binomial_floor_divisibility (p n : ℕ) (hp : Prime p) (hn : n ≥ p) :
  p ∣ (Nat.choose n p - n / p) := by
  sorry

end binomial_floor_divisibility_l253_25399


namespace percentage_relation_l253_25373

theorem percentage_relation (x a b : ℝ) (h1 : a = 0.07 * x) (h2 : b = 0.14 * x) :
  a = 0.5 * b := by sorry

end percentage_relation_l253_25373


namespace reading_time_difference_l253_25390

/-- Proves that the difference in reading time between Molly and Xanthia is 150 minutes -/
theorem reading_time_difference 
  (xanthia_speed : ℝ) 
  (molly_speed : ℝ) 
  (book_pages : ℝ) 
  (h1 : xanthia_speed = 120)
  (h2 : molly_speed = 60)
  (h3 : book_pages = 300) :
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 150 := by
sorry

end reading_time_difference_l253_25390


namespace right_triangle_groups_l253_25331

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_groups :
  ¬ (is_right_triangle 1.5 2 3) ∧
  (is_right_triangle 7 24 25) ∧
  (is_right_triangle 6 8 10) ∧
  (is_right_triangle 9 12 15) :=
by sorry

end right_triangle_groups_l253_25331


namespace inequality_domain_l253_25323

theorem inequality_domain (x : ℝ) : 
  (4 * x^2 / (1 - Real.sqrt (1 + 2*x))^2 < 2*x + 9) ↔ 
  (x ≥ -1/2 ∧ x < 0) ∨ (x > 0 ∧ x < 45/8) :=
sorry

end inequality_domain_l253_25323


namespace no_real_solutions_l253_25377

theorem no_real_solutions :
  ∀ x : ℝ, x ≠ 2 → (4 * x^3 + 3 * x^2 + x + 2) / (x - 2) ≠ 4 * x^2 + 5 := by
  sorry

end no_real_solutions_l253_25377


namespace complement_of_N_wrt_M_l253_25374

def M : Set Nat := {1, 2, 3, 4, 5}
def N : Set Nat := {2, 4}

theorem complement_of_N_wrt_M :
  (M \ N) = {1, 3, 5} := by sorry

end complement_of_N_wrt_M_l253_25374


namespace leftover_value_is_correct_l253_25306

/-- Represents the number of coins in a roll --/
structure RollSize :=
  (quarters : Nat)
  (dimes : Nat)

/-- Represents the number of coins in a jar --/
structure JarContents :=
  (quarters : Nat)
  (dimes : Nat)

/-- Calculates the total value of leftover coins in dollars --/
def leftoverValue (rollSize : RollSize) (alice : JarContents) (bob : JarContents) : Rat :=
  let totalQuarters := alice.quarters + bob.quarters
  let totalDimes := alice.dimes + bob.dimes
  let leftoverQuarters := totalQuarters % rollSize.quarters
  let leftoverDimes := totalDimes % rollSize.dimes
  (leftoverQuarters * 25 + leftoverDimes * 10) / 100

theorem leftover_value_is_correct (rollSize : RollSize) (alice : JarContents) (bob : JarContents) :
  rollSize.quarters = 50 →
  rollSize.dimes = 60 →
  alice.quarters = 95 →
  alice.dimes = 184 →
  bob.quarters = 145 →
  bob.dimes = 312 →
  leftoverValue rollSize alice bob = 116/10 := by
  sorry

#eval leftoverValue ⟨50, 60⟩ ⟨95, 184⟩ ⟨145, 312⟩

end leftover_value_is_correct_l253_25306


namespace min_bench_sections_for_equal_seating_l253_25396

/-- Represents the capacity of a bench section -/
structure BenchCapacity where
  adults : Nat
  children : Nat

/-- Calculates the minimum number of bench sections needed -/
def minBenchSections (capacity : BenchCapacity) : Nat :=
  Nat.lcm capacity.adults capacity.children / capacity.adults

/-- Theorem stating the minimum number of bench sections needed -/
theorem min_bench_sections_for_equal_seating (capacity : BenchCapacity) 
  (h1 : capacity.adults = 8) 
  (h2 : capacity.children = 12) : 
  minBenchSections capacity = 3 := by
  sorry

#eval minBenchSections ⟨8, 12⟩

end min_bench_sections_for_equal_seating_l253_25396


namespace john_work_hours_john_total_hours_l253_25301

theorem john_work_hours : ℕ → ℕ → ℕ → ℕ
  | hours_per_day, start_day, end_day =>
    (end_day - start_day + 1) * hours_per_day

theorem john_total_hours : john_work_hours 8 3 7 = 40 := by
  sorry

end john_work_hours_john_total_hours_l253_25301


namespace witch_cake_votes_l253_25351

/-- The number of votes for the witch cake -/
def witch_votes : ℕ := sorry

/-- The number of votes for the unicorn cake -/
def unicorn_votes : ℕ := 3 * witch_votes

/-- The number of votes for the dragon cake -/
def dragon_votes : ℕ := witch_votes + 25

/-- The total number of votes cast -/
def total_votes : ℕ := 60

theorem witch_cake_votes :
  witch_votes = 7 ∧
  unicorn_votes = 3 * witch_votes ∧
  dragon_votes = witch_votes + 25 ∧
  witch_votes + unicorn_votes + dragon_votes = total_votes :=
sorry

end witch_cake_votes_l253_25351


namespace ellipse_m_range_l253_25335

/-- The equation of the curve -/
def curve_equation (x y m : ℝ) : Prop :=
  x^2 / (m - 2) + y^2 / (6 - m) = 1

/-- Definition of an ellipse in terms of its equation -/
def is_ellipse (m : ℝ) : Prop :=
  (∀ x y, curve_equation x y m → x^2 / (m - 2) > 0 ∧ y^2 / (6 - m) > 0) ∧
  m - 2 ≠ 6 - m

/-- Theorem: The range of m for which the curve is an ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m ↔ (2 < m ∧ m < 6 ∧ m ≠ 4) :=
sorry

end ellipse_m_range_l253_25335


namespace outfit_combinations_l253_25332

theorem outfit_combinations (shirts : Nat) (pants : Nat) (shoe_types : Nat) (styles_per_type : Nat) :
  shirts = 4 →
  pants = 4 →
  shoe_types = 2 →
  styles_per_type = 2 →
  shirts * pants * (shoe_types * styles_per_type) = 64 := by
sorry

end outfit_combinations_l253_25332


namespace forgotten_angle_measure_l253_25365

/-- The sum of exterior angles of a polygon --/
def sum_exterior_angles : ℝ := 360

/-- The sum of exterior angles calculated by Robert --/
def roberts_sum : ℝ := 345

/-- Theorem: The measure of the forgotten exterior angle is 15° --/
theorem forgotten_angle_measure :
  sum_exterior_angles - roberts_sum = 15 :=
by sorry

end forgotten_angle_measure_l253_25365


namespace odd_prime_square_root_theorem_l253_25302

theorem odd_prime_square_root_theorem (p : ℕ) (k : ℕ) (h_prime : Prime p) (h_odd : Odd p) 
  (h_pos : k > 0) (h_sqrt : ∃ (m : ℕ), m > 0 ∧ m * m = k * k - p * k) : 
  k = (p + 1)^2 / 4 := by
sorry

end odd_prime_square_root_theorem_l253_25302


namespace circle_equation_radius_6_l253_25327

theorem circle_equation_radius_6 (x y k : ℝ) : 
  (∃ h i : ℝ, ∀ x y : ℝ, (x - h)^2 + (y - i)^2 = 6^2 ↔ x^2 + 10*x + y^2 + 6*y - k = 0) ↔ 
  k = 2 := by sorry

end circle_equation_radius_6_l253_25327


namespace odd_function_conditions_l253_25311

noncomputable def f (a b x : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + a)

theorem odd_function_conditions (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →
  (a = 2 ∧ b = 1 ∧
   ∀ x y, x < y → f 2 1 x > f 2 1 y ∧
   ∀ t k, f 2 1 (t^2 - 2*t) + f 2 1 (2*t^2 - k) < 0 → k < -1/3) :=
by sorry

end odd_function_conditions_l253_25311


namespace min_value_of_expression_l253_25337

def expression (a b c : ℕ) : ℚ := ((a + b) / c) / 2

theorem min_value_of_expression :
  ∃ (a b c : ℕ), a ∈ ({2, 3, 5} : Set ℕ) ∧ 
                 b ∈ ({2, 3, 5} : Set ℕ) ∧ 
                 c ∈ ({2, 3, 5} : Set ℕ) ∧ 
                 a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                 (∀ (x y z : ℕ), x ∈ ({2, 3, 5} : Set ℕ) → 
                                 y ∈ ({2, 3, 5} : Set ℕ) → 
                                 z ∈ ({2, 3, 5} : Set ℕ) → 
                                 x ≠ y → y ≠ z → x ≠ z →
                                 expression a b c ≤ expression x y z) ∧
                 expression a b c = 1/2 := by
  sorry

#eval expression 2 3 5

end min_value_of_expression_l253_25337


namespace arrangement_count_l253_25316

def number_of_arrangements (n m k : ℕ) : ℕ :=
  2 * (m * (m - 1) * (m - 2) * (m - 3))

theorem arrangement_count :
  number_of_arrangements 8 6 2 = 720 := by
  sorry

end arrangement_count_l253_25316


namespace average_tape_length_l253_25345

def tape_lengths : List ℝ := [35, 29, 35.5, 36, 30.5]

theorem average_tape_length :
  (tape_lengths.sum / tape_lengths.length : ℝ) = 33.2 := by
  sorry

end average_tape_length_l253_25345


namespace equal_gender_probability_l253_25329

def total_students : ℕ := 8
def men_count : ℕ := 4
def women_count : ℕ := 4
def selection_size : ℕ := 4

theorem equal_gender_probability :
  let total_ways := Nat.choose total_students selection_size
  let ways_to_choose_men := Nat.choose men_count (selection_size / 2)
  let ways_to_choose_women := Nat.choose women_count (selection_size / 2)
  (ways_to_choose_men * ways_to_choose_women : ℚ) / total_ways = 18 / 35 := by
  sorry

end equal_gender_probability_l253_25329


namespace no_functions_satisfying_equation_l253_25308

theorem no_functions_satisfying_equation :
  ¬ ∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), f (x + f y) = y^2 + g x := by
  sorry

end no_functions_satisfying_equation_l253_25308


namespace prime_pairs_sum_58_l253_25380

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- The theorem stating that there are exactly 4 pairs of primes summing to 58 -/
theorem prime_pairs_sum_58 :
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (p q : ℕ), (p, q) ∈ s ↔ isPrime p ∧ isPrime q ∧ p + q = 58) ∧
    s.card = 4 :=
sorry

end prime_pairs_sum_58_l253_25380


namespace tyler_meal_combinations_l253_25387

def meat_options : ℕ := 3
def vegetable_options : ℕ := 5
def dessert_options : ℕ := 4
def drink_options : ℕ := 3
def vegetables_to_choose : ℕ := 3

def meal_combinations : ℕ :=
  meat_options * Nat.choose vegetable_options vegetables_to_choose * dessert_options * drink_options

theorem tyler_meal_combinations :
  meal_combinations = 360 :=
by sorry

end tyler_meal_combinations_l253_25387


namespace mary_nickels_l253_25342

/-- The number of nickels Mary has after receiving some from her dad -/
def total_nickels (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Mary has 12 nickels after receiving 5 from her dad -/
theorem mary_nickels : total_nickels 7 5 = 12 := by
  sorry

end mary_nickels_l253_25342


namespace f_odd_and_monotonic_l253_25391

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2*x else -x^2 + 2*x

theorem f_odd_and_monotonic :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x < y → f x < f y) :=
sorry

end f_odd_and_monotonic_l253_25391


namespace supersonic_pilot_l253_25340

theorem supersonic_pilot (total_distance : ℝ) 
  (dupon_distance dupon_remaining duran_distance duran_remaining : ℝ) : 
  (dupon_distance + dupon_remaining = total_distance) →
  (duran_distance + duran_remaining = total_distance) →
  (2 * dupon_distance + dupon_remaining / 1.5 = total_distance) →
  (duran_distance / 1.5 + 2 * duran_remaining = total_distance) →
  (duran_distance = 3 * duran_remaining) →
  (duran_distance = 3 / 4 * total_distance) :=
by sorry

end supersonic_pilot_l253_25340


namespace consecutive_integer_averages_l253_25325

theorem consecutive_integer_averages (a : ℤ) (h : a + 1 > 0) : 
  let first_set := [a + 1, a + 2, a + 3]
  let first_avg := (a + 1 + a + 2 + a + 3) / 3
  let second_set := [first_avg, first_avg + 1, first_avg + 2]
  (second_set.sum / 3 : ℚ) = a + 3 := by sorry

end consecutive_integer_averages_l253_25325


namespace expression_decrease_l253_25355

theorem expression_decrease (x y : ℝ) (h : x > 0 ∧ y > 0) : 
  let original := 125 * x * y^2
  let new_x := 0.75 * x
  let new_y := 0.75 * y
  let new_value := 125 * new_x * new_y^2
  new_value = (27/64) * original := by
sorry

end expression_decrease_l253_25355


namespace net_population_increase_per_day_l253_25359

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate in people per second -/
def birth_rate : ℚ := 7 / 2

/-- Represents the death rate in people per second -/
def death_rate : ℚ := 2 / 2

/-- Represents the net population increase per second -/
def net_increase_per_second : ℚ := birth_rate - death_rate

/-- Theorem stating the net population increase in one day -/
theorem net_population_increase_per_day :
  ⌊(net_increase_per_second * seconds_per_day : ℚ)⌋ = 216000 := by
  sorry

end net_population_increase_per_day_l253_25359


namespace arithmetic_sequence_property_l253_25328

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 6 + a 10 = 16)
  (h_a4 : a 4 = 1) :
  a 12 = 15 := by
sorry

end arithmetic_sequence_property_l253_25328


namespace total_jelly_beans_l253_25341

/-- The number of vanilla jelly beans -/
def vanilla : ℕ := 120

/-- The number of grape jelly beans -/
def grape : ℕ := 5 * vanilla + 50

/-- The total number of jelly beans -/
def total : ℕ := vanilla + grape

theorem total_jelly_beans : total = 770 := by sorry

end total_jelly_beans_l253_25341


namespace ball_hit_time_l253_25366

/-- The time when a ball hits the ground, given its height equation -/
theorem ball_hit_time : ∃ t : ℝ, t > 0 ∧ -6 * t^2 - 10 * t + 56 = 0 ∧ t = 7/3 := by
  sorry

end ball_hit_time_l253_25366


namespace mailbox_distribution_l253_25392

/-- The number of ways to distribute n distinct objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of mailboxes -/
def num_mailboxes : ℕ := 4

/-- The number of letters -/
def num_letters : ℕ := 3

theorem mailbox_distribution :
  distribute num_letters num_mailboxes = 64 := by
  sorry

end mailbox_distribution_l253_25392


namespace quadratic_one_root_l253_25300

theorem quadratic_one_root (n : ℝ) : 
  (∀ x : ℝ, x^2 + 6*n*x + 2*n = 0 → (∀ y : ℝ, y^2 + 6*n*y + 2*n = 0 → y = x)) →
  n = 2/9 := by
sorry

end quadratic_one_root_l253_25300


namespace solutions_nonempty_and_finite_l253_25370

def solution_set (n : ℕ) : Set (ℕ × ℕ × ℕ) :=
  {(x, y, z) | Real.sqrt ((x^2 : ℝ) + y + n) + Real.sqrt ((y^2 : ℝ) + x + n) = z}

theorem solutions_nonempty_and_finite (n : ℕ) :
  (solution_set n).Nonempty ∧ (solution_set n).Finite :=
sorry

end solutions_nonempty_and_finite_l253_25370


namespace kittens_remaining_l253_25318

def initial_kittens : ℕ := 8
def kittens_given_away : ℕ := 2

theorem kittens_remaining : initial_kittens - kittens_given_away = 6 := by
  sorry

end kittens_remaining_l253_25318


namespace paint_distribution_l253_25339

def paint_problem (total : ℚ) (blue_ratio green_ratio white_ratio : ℕ) : Prop :=
  let total_ratio := blue_ratio + green_ratio + white_ratio
  let blue_amount := (blue_ratio : ℚ) * total / total_ratio
  let green_amount := (green_ratio : ℚ) * total / total_ratio
  let white_amount := (white_ratio : ℚ) * total / total_ratio
  blue_amount = 15 ∧ green_amount = 9 ∧ white_amount = 21

theorem paint_distribution :
  paint_problem 45 5 3 7 := by
  sorry

end paint_distribution_l253_25339


namespace max_sum_of_squares_l253_25368

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 17 →
  a * b + c + d = 94 →
  a * d + b * c = 195 →
  c * d = 120 →
  a^2 + b^2 + c^2 + d^2 ≤ 918 := by
sorry

end max_sum_of_squares_l253_25368


namespace sequence_property_l253_25317

/-- An integer sequence satisfying the given conditions -/
def IntegerSequence (a : ℕ → ℤ) : Prop :=
  a 3 = -1 ∧ 
  a 7 = 4 ∧ 
  (∀ n ≤ 6, ∃ d : ℤ, a (n + 1) - a n = d) ∧ 
  (∀ n ≥ 5, ∃ q : ℚ, a (n + 1) = a n * q)

/-- The property that needs to be satisfied for a given m -/
def SatisfiesProperty (a : ℕ → ℤ) (m : ℕ) : Prop :=
  a m + a (m + 1) + a (m + 2) = a m * a (m + 1) * a (m + 2)

/-- The main theorem statement -/
theorem sequence_property (a : ℕ → ℤ) (h : IntegerSequence a) :
  ∀ m : ℕ, m > 0 → (SatisfiesProperty a m ↔ m = 1 ∨ m = 3) :=
sorry

end sequence_property_l253_25317


namespace price_restoration_l253_25347

theorem price_restoration (original_price : ℝ) (reduced_price : ℝ) : 
  reduced_price = 0.8 * original_price → 
  reduced_price * 1.25 = original_price :=
by sorry

end price_restoration_l253_25347


namespace sally_buttons_theorem_l253_25383

/-- The number of buttons Sally needs to sew all the shirts -/
def total_buttons : ℕ :=
  let monday_shirts := 4
  let tuesday_shirts := 3
  let wednesday_shirts := 2
  let buttons_per_shirt := 5
  let total_shirts := monday_shirts + tuesday_shirts + wednesday_shirts
  total_shirts * buttons_per_shirt

theorem sally_buttons_theorem : total_buttons = 45 := by
  sorry

end sally_buttons_theorem_l253_25383


namespace non_working_video_games_l253_25389

theorem non_working_video_games (total : ℕ) (price : ℕ) (earned : ℕ) :
  total = 15 →
  price = 7 →
  earned = 63 →
  total - (earned / price) = 6 := by
sorry

end non_working_video_games_l253_25389


namespace specific_prism_surface_area_l253_25313

/-- A right triangular prism with given dimensions -/
structure RightTriangularPrism where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  height : ℝ

/-- Calculate the surface area of a right triangular prism -/
def surfaceArea (prism : RightTriangularPrism) : ℝ :=
  prism.leg1 * prism.leg2 + (prism.leg1 + prism.leg2 + prism.hypotenuse) * prism.height

/-- The surface area of the specific right triangular prism is 72 -/
theorem specific_prism_surface_area :
  let prism : RightTriangularPrism := {
    leg1 := 3,
    leg2 := 4,
    hypotenuse := 5,
    height := 5
  }
  surfaceArea prism = 72 := by sorry

end specific_prism_surface_area_l253_25313


namespace dave_won_three_more_than_jerry_l253_25352

/-- Shuffleboard game results -/
structure ShuffleboardResults where
  dave_wins : ℕ
  ken_wins : ℕ
  jerry_wins : ℕ
  total_games : ℕ

/-- Conditions for the shuffleboard game results -/
def valid_results (r : ShuffleboardResults) : Prop :=
  r.ken_wins = r.dave_wins + 5 ∧
  r.dave_wins > r.jerry_wins ∧
  r.jerry_wins = 7 ∧
  r.total_games = r.dave_wins + r.ken_wins + r.jerry_wins ∧
  r.total_games = 32

/-- Theorem: Dave won 3 more games than Jerry -/
theorem dave_won_three_more_than_jerry (r : ShuffleboardResults) 
  (h : valid_results r) : r.dave_wins = r.jerry_wins + 3 := by
  sorry


end dave_won_three_more_than_jerry_l253_25352


namespace mother_twice_lisa_age_l253_25372

/-- Represents a person with their birth year -/
structure Person where
  birth_year : ℕ

/-- The year of Lisa's 6th birthday -/
def lisa_sixth_birthday : ℕ := 2010

/-- Lisa's birth year -/
def lisa : Person :=
  ⟨lisa_sixth_birthday - 6⟩

/-- Lisa's mother's birth year -/
def lisa_mother : Person :=
  ⟨lisa_sixth_birthday - 30⟩

/-- The year when Lisa's mother's age will be twice Lisa's age -/
def target_year : ℕ := 2028

/-- Theorem stating that the target year is correct -/
theorem mother_twice_lisa_age :
  (target_year - lisa_mother.birth_year) = 2 * (target_year - lisa.birth_year) :=
by sorry

end mother_twice_lisa_age_l253_25372


namespace min_value_of_squares_l253_25358

theorem min_value_of_squares (x y : ℝ) (h : x^3 + y^3 + 3*x*y = 1) :
  ∃ (m : ℝ), m = 1/2 ∧ (∀ a b : ℝ, a^3 + b^3 + 3*a*b = 1 → a^2 + b^2 ≥ m) ∧ (x^2 + y^2 = m) := by
  sorry

end min_value_of_squares_l253_25358


namespace numbers_five_units_from_negative_one_l253_25393

theorem numbers_five_units_from_negative_one :
  ∀ x : ℝ, |x - (-1)| = 5 ↔ x = 4 ∨ x = -6 := by sorry

end numbers_five_units_from_negative_one_l253_25393


namespace runners_speed_l253_25385

theorem runners_speed (track_length : ℝ) (time_between_encounters : ℝ) (speed_difference : ℝ)
  (h1 : track_length = 600)
  (h2 : time_between_encounters = 50)
  (h3 : speed_difference = 2)
  : ∃ (faster_speed slower_speed : ℝ),
    faster_speed - slower_speed = speed_difference ∧
    faster_speed + slower_speed = track_length / time_between_encounters ∧
    faster_speed = 7 := by
  sorry

end runners_speed_l253_25385


namespace sunday_cost_theorem_l253_25350

-- Define the constants
def weekday_discount : ℝ := 0.1
def weekend_increase : ℝ := 0.5
def shaving_cost : ℝ := 10
def styling_cost : ℝ := 15
def monday_total : ℝ := 18

-- Define the theorem
theorem sunday_cost_theorem :
  let weekday_haircut_cost := (monday_total - shaving_cost) / (1 - weekday_discount)
  let weekend_haircut_cost := weekday_haircut_cost * (1 + weekend_increase)
  let sunday_total := weekend_haircut_cost + styling_cost
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |sunday_total - 28.34| < ε :=
sorry

end sunday_cost_theorem_l253_25350


namespace eleven_divides_difference_l253_25346

theorem eleven_divides_difference (A B C : ℕ) : 
  A ≠ C →
  A < 10 → B < 10 → C < 10 →
  ∃ k : ℤ, (100 * A + 10 * B + C) - (100 * C + 10 * B + A) = 11 * k :=
by sorry

end eleven_divides_difference_l253_25346


namespace eliana_steps_proof_l253_25386

-- Define the number of steps for each day
def first_day_morning_steps : ℕ := 200
def first_day_additional_steps : ℕ := 300
def third_day_additional_steps : ℕ := 100

-- Define the total steps for the first day
def first_day_total : ℕ := first_day_morning_steps + first_day_additional_steps

-- Define the total steps for the second day
def second_day_total : ℕ := 2 * first_day_total

-- Define the total steps for all three days
def total_steps : ℕ := first_day_total + second_day_total + third_day_additional_steps

-- Theorem statement
theorem eliana_steps_proof : total_steps = 1600 := by
  sorry

end eliana_steps_proof_l253_25386


namespace tile_arrangements_example_l253_25314

/-- The number of distinguishable arrangements of tiles -/
def tileArrangements (brown purple green yellow : ℕ) : ℕ :=
  Nat.factorial (brown + purple + green + yellow) /
  (Nat.factorial brown * Nat.factorial purple * Nat.factorial green * Nat.factorial yellow)

/-- Theorem stating that the number of distinguishable arrangements
    of 2 brown, 2 purple, 3 green, and 2 yellow tiles is 3780 -/
theorem tile_arrangements_example :
  tileArrangements 2 2 3 2 = 3780 := by
  sorry

end tile_arrangements_example_l253_25314


namespace smallest_nine_digit_multiple_of_seven_digit_smallest_nine_digit_is_100_times_smallest_seven_digit_l253_25375

theorem smallest_nine_digit_multiple_of_seven_digit : ℕ → ℕ → Prop :=
  fun smallest_nine_digit smallest_seven_digit =>
    smallest_nine_digit / smallest_seven_digit = 100

/-- The smallest nine-digit number is 100 times the smallest seven-digit number -/
theorem smallest_nine_digit_is_100_times_smallest_seven_digit 
  (h1 : smallest_nine_digit = 100000000)
  (h2 : smallest_seven_digit = 1000000) :
  smallest_nine_digit_multiple_of_seven_digit smallest_nine_digit smallest_seven_digit :=
by
  sorry

end smallest_nine_digit_multiple_of_seven_digit_smallest_nine_digit_is_100_times_smallest_seven_digit_l253_25375


namespace calculation_proof_l253_25353

theorem calculation_proof : -3^2 - (-1)^4 * 5 / (-5/3) = -6 := by
  sorry

end calculation_proof_l253_25353


namespace smallest_divisor_for_perfect_cube_l253_25388

theorem smallest_divisor_for_perfect_cube (n : ℕ) : 
  (n > 0 ∧ ∃ (k : ℕ), 3600 / n = k^3 ∧ ∀ (m : ℕ), m > 0 → m < n → ¬∃ (j : ℕ), 3600 / m = j^3) → n = 15 := by
  sorry

end smallest_divisor_for_perfect_cube_l253_25388


namespace strawberries_picked_l253_25371

theorem strawberries_picked (initial : ℕ) (final : ℕ) (picked : ℕ) : 
  initial = 42 → final = 120 → final = initial + picked → picked = 78 := by
  sorry

end strawberries_picked_l253_25371


namespace total_soccer_games_l253_25309

def soccer_games_this_year : ℕ := 11
def soccer_games_last_year : ℕ := 13
def soccer_games_next_year : ℕ := 15

theorem total_soccer_games :
  soccer_games_this_year + soccer_games_last_year + soccer_games_next_year = 39 := by
  sorry

end total_soccer_games_l253_25309


namespace cylinder_radius_problem_l253_25326

theorem cylinder_radius_problem (r : ℝ) (h : ℝ) : 
  h = 3 → 
  π * (r + 5)^2 * h = π * r^2 * (h + 5) → 
  r = 3 + 2 * Real.sqrt 6 :=
by sorry

end cylinder_radius_problem_l253_25326


namespace square_area_proof_l253_25310

theorem square_area_proof (x : ℝ) : 
  (6 * x - 27 = 30 - 2 * x) → 
  ((6 * x - 27) * (6 * x - 27) = 248.0625) := by
  sorry

end square_area_proof_l253_25310


namespace lcm_gcf_ratio_l253_25343

theorem lcm_gcf_ratio : (Nat.lcm 210 462) / (Nat.gcd 210 462) = 55 := by sorry

end lcm_gcf_ratio_l253_25343


namespace ln_f_greater_than_one_max_a_value_l253_25382

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |x - a|

-- Part I
theorem ln_f_greater_than_one : ∀ x : ℝ, Real.log (f (-1) x) > 1 := by sorry

-- Part II
theorem max_a_value : 
  (∃ a_max : ℝ, 
    (∀ a : ℝ, (∀ x : ℝ, f a x ≥ a) → a ≤ a_max) ∧
    (∀ x : ℝ, f a_max x ≥ a_max) ∧
    a_max = 1) := by sorry

end ln_f_greater_than_one_max_a_value_l253_25382


namespace evaluate_expression_l253_25376

theorem evaluate_expression (a b c : ℚ) 
  (ha : a = 1/2) (hb : b = 1/4) (hc : c = 5) : 
  a^2 * b^3 * c = 5/256 := by
sorry

end evaluate_expression_l253_25376


namespace candy_mixture_cost_l253_25321

/-- Proves the cost of cheaper candy given the conditions of the mixture --/
theorem candy_mixture_cost (expensive_weight : ℝ) (expensive_cost : ℝ) 
  (cheaper_weight : ℝ) (mixture_cost : ℝ) (total_weight : ℝ) :
  expensive_weight = 20 →
  expensive_cost = 8 →
  cheaper_weight = 40 →
  mixture_cost = 6 →
  total_weight = expensive_weight + cheaper_weight →
  ∃ (cheaper_cost : ℝ),
    cheaper_cost = 5 ∧
    expensive_weight * expensive_cost + cheaper_weight * cheaper_cost = 
      total_weight * mixture_cost :=
by
  sorry

end candy_mixture_cost_l253_25321


namespace car_speed_problem_l253_25305

theorem car_speed_problem (first_hour_speed second_hour_speed average_speed : ℝ) :
  second_hour_speed = 60 →
  average_speed = 70 →
  (first_hour_speed + second_hour_speed) / 2 = average_speed →
  first_hour_speed = 80 := by
  sorry

end car_speed_problem_l253_25305


namespace points_collinear_l253_25369

/-- Given four points P, A, B, C in space, if PC = 1/4 PA + 3/4 PB, then A, B, C are collinear -/
theorem points_collinear (P A B C : EuclideanSpace ℝ (Fin 3)) 
  (h : C - P = (1/4 : ℝ) • (A - P) + (3/4 : ℝ) • (B - P)) : 
  ∃ t : ℝ, C - A = t • (B - C) := by
  sorry

end points_collinear_l253_25369


namespace f_difference_l253_25324

theorem f_difference (r : ℝ) : 
  let f : ℝ → ℝ := λ n => (1/4) * n * (n+1) * (n+2) * (n+3)
  f r - f (r-1) = r * (r+1) * (r+2) := by
sorry

end f_difference_l253_25324


namespace strawberry_cartons_correct_l253_25363

/-- Calculates the number of strawberry cartons in the cupboard given the total needed, blueberry cartons, and cartons bought. -/
def strawberry_cartons (total_needed : ℕ) (blueberry_cartons : ℕ) (cartons_bought : ℕ) : ℕ :=
  total_needed - (blueberry_cartons + cartons_bought)

/-- Theorem stating that the number of strawberry cartons is correct given the problem conditions. -/
theorem strawberry_cartons_correct :
  strawberry_cartons 42 7 33 = 2 := by
  sorry

end strawberry_cartons_correct_l253_25363


namespace gcf_of_450_and_210_l253_25378

theorem gcf_of_450_and_210 : Nat.gcd 450 210 = 30 := by
  sorry

end gcf_of_450_and_210_l253_25378


namespace min_planes_for_300_parts_l253_25344

def q (n : ℕ) : ℚ := (n^3 + 5*n + 6) / 6

theorem min_planes_for_300_parts : 
  ∀ n : ℕ, n < 13 → q n < 300 ∧ q 13 ≥ 300 :=
sorry

end min_planes_for_300_parts_l253_25344


namespace exam_score_standard_deviations_l253_25320

/-- Given an exam with mean score 88.8, where 90 is 3 standard deviations above the mean,
    prove that 86 is 7 standard deviations below the mean. -/
theorem exam_score_standard_deviations 
  (mean : ℝ) 
  (above_score : ℝ) 
  (below_score : ℝ) 
  (above_sd : ℝ) 
  (h_mean : mean = 88.8)
  (h_above : above_score = 90)
  (h_below : below_score = 86)
  (h_above_sd : above_score = mean + above_sd * 3)
  (h_below_exists : ∃ (x : ℝ), below_score = mean - x * (above_score - mean) / 3) :
  ∃ (x : ℝ), below_score = mean - x * (above_score - mean) / 3 ∧ x = 7 := by
  sorry

end exam_score_standard_deviations_l253_25320


namespace pizza_toppings_combinations_l253_25349

theorem pizza_toppings_combinations : Nat.choose 7 3 = 35 := by
  sorry

end pizza_toppings_combinations_l253_25349


namespace expansion_terms_count_l253_25364

theorem expansion_terms_count (N : ℕ) : 
  (Nat.choose N 5 = 3003) ↔ (N = 15) := by sorry

end expansion_terms_count_l253_25364


namespace farmers_market_cauliflower_sales_l253_25384

/-- Represents the farmers' market sales problem -/
structure FarmersMarket where
  total_earnings : ℕ
  broccoli_price : ℕ
  carrot_price : ℕ
  spinach_price : ℕ
  cauliflower_price : ℕ
  tomato_price : ℕ
  broccoli_sold : ℕ
  spinach_weight : ℕ

/-- The theorem representing the farmers' market problem -/
theorem farmers_market_cauliflower_sales
  (market : FarmersMarket)
  (h1 : market.total_earnings = 520)
  (h2 : market.broccoli_price = 3)
  (h3 : market.carrot_price = 2)
  (h4 : market.spinach_price = 4)
  (h5 : market.cauliflower_price = 5)
  (h6 : market.tomato_price = 1)
  (h7 : market.broccoli_sold = 19)
  (h8 : market.spinach_weight * 2 * market.carrot_price = market.spinach_weight * market.spinach_price + 16)
  (h9 : market.broccoli_sold * market.broccoli_price + market.spinach_weight * market.spinach_price =
        (market.broccoli_sold * market.broccoli_price + market.spinach_weight * market.spinach_price) * market.tomato_price) :
  market.total_earnings - (market.broccoli_sold * market.broccoli_price +
                           market.spinach_weight * 2 * market.carrot_price +
                           market.spinach_weight * market.spinach_price +
                           (market.broccoli_sold * market.broccoli_price + market.spinach_weight * market.spinach_price)) = 310 := by
  sorry


end farmers_market_cauliflower_sales_l253_25384


namespace union_M_N_equals_real_l253_25361

-- Define the sets M and N
def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | x < 3}

-- State the theorem
theorem union_M_N_equals_real : M ∪ N = Set.univ := by sorry

end union_M_N_equals_real_l253_25361


namespace min_value_expression_l253_25315

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^3 / (y - 1)) + (y^3 / (x - 1)) ≥ 24 := by
  sorry

end min_value_expression_l253_25315


namespace triangle_angle_determination_l253_25381

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that if a = 2√3, b = 6, and A = 30°, then B = 60° or B = 120°. -/
theorem triangle_angle_determination (a b c : ℝ) (A B C : ℝ) :
  a = 2 * Real.sqrt 3 →
  b = 6 →
  A = π / 6 →
  (B = π / 3 ∨ B = 2 * π / 3) := by
  sorry

end triangle_angle_determination_l253_25381


namespace units_digit_of_six_to_sixth_l253_25304

/-- The units digit of 6^6 is 6 -/
theorem units_digit_of_six_to_sixth (n : ℕ) : n = 6^6 → n % 10 = 6 := by
  sorry

end units_digit_of_six_to_sixth_l253_25304


namespace profit_is_42_l253_25338

/-- Calculates the profit from selling face masks given the following conditions:
  * 12 boxes of face masks were bought
  * Each box costs $9
  * Each box contains 50 masks
  * 6 boxes were repacked and sold for $5 per 25 pieces
  * Remaining 300 masks were sold in baggies of 10 pieces for $3 each
-/
def calculate_profit (
  total_boxes : ℕ
  ) (cost_per_box : ℕ
  ) (masks_per_box : ℕ
  ) (repacked_boxes : ℕ
  ) (price_per_repack : ℕ
  ) (masks_per_repack : ℕ
  ) (remaining_masks : ℕ
  ) (price_per_baggy : ℕ
  ) (masks_per_baggy : ℕ
  ) : ℕ :=
  let total_cost := total_boxes * cost_per_box
  let repacked_masks := repacked_boxes * masks_per_box
  let repacked_revenue := (repacked_masks / masks_per_repack) * price_per_repack
  let baggy_revenue := (remaining_masks / masks_per_baggy) * price_per_baggy
  let total_revenue := repacked_revenue + baggy_revenue
  total_revenue - total_cost

theorem profit_is_42 : 
  calculate_profit 12 9 50 6 5 25 300 3 10 = 42 := by
  sorry

end profit_is_42_l253_25338


namespace lcm_of_210_and_605_l253_25360

theorem lcm_of_210_and_605 :
  let a := 210
  let b := 605
  let hcf := 55
  Nat.lcm a b = 2310 := by
  sorry

end lcm_of_210_and_605_l253_25360


namespace sum_of_m_values_l253_25367

-- Define the inequality system
def inequality_system (m : ℤ) : Prop :=
  ∀ x : ℝ, (x > 0) ↔ ((x - m) / 2 > 0 ∧ x - 4 < 2 * (x - 2))

-- Define the fractional equation
def fractional_equation (m : ℤ) : Prop :=
  ∃ y : ℕ, (1 - y) / (2 - y) = 3 - m / (y - 2)

-- Theorem statement
theorem sum_of_m_values :
  (∃ S : Finset ℤ, (∀ m : ℤ, m ∈ S ↔ (inequality_system m ∧ fractional_equation m)) ∧
    S.sum id = -8) := by
  sorry

end sum_of_m_values_l253_25367


namespace corner_sum_is_164_l253_25312

/-- Represents a square on the checkerboard -/
structure Square where
  row : Nat
  col : Nat

/-- The size of the checkerboard -/
def boardSize : Nat := 9

/-- The total number of squares on the board -/
def totalSquares : Nat := boardSize * boardSize

/-- Function to get the number in a given square -/
def getNumber (s : Square) : Nat :=
  s.row * boardSize + s.col + 1

/-- The four corners of the board -/
def corners : List Square := [
  { row := 0, col := 0 },             -- Top left
  { row := 0, col := boardSize - 1 }, -- Top right
  { row := boardSize - 1, col := 0 }, -- Bottom left
  { row := boardSize - 1, col := boardSize - 1 }  -- Bottom right
]

/-- The sum of numbers in the four corners -/
def cornerSum : Nat := (corners.map getNumber).sum

theorem corner_sum_is_164 : cornerSum = 164 := by
  sorry

end corner_sum_is_164_l253_25312


namespace fraction_equivalence_prime_l253_25357

theorem fraction_equivalence_prime (n : ℕ) : 
  Prime n ∧ (4 + n : ℚ) / (7 + n) = 7 / 8 ↔ n = 17 := by
  sorry

end fraction_equivalence_prime_l253_25357


namespace exponential_equation_solution_l253_25307

theorem exponential_equation_solution :
  ∃ x : ℝ, (10 : ℝ) ^ (x + 4) = 100 ^ x ∧ x = 4 :=
by sorry

end exponential_equation_solution_l253_25307


namespace bankers_discount_calculation_l253_25397

/-- Banker's discount calculation -/
theorem bankers_discount_calculation 
  (bankers_gain : ℝ) 
  (time : ℝ) 
  (rate : ℝ) :
  bankers_gain = 180 →
  time = 3 →
  rate = 12 →
  (bankers_gain / (1 - (rate * time) / 100)) = 281.25 :=
by sorry

end bankers_discount_calculation_l253_25397


namespace no_always_positive_f_solution_sets_f_negative_l253_25303

def f (a x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 1

theorem no_always_positive_f :
  ¬∃ a : ℝ, ∀ x : ℝ, f a x > 0 := by sorry

theorem solution_sets_f_negative (a : ℝ) :
  (a = 0 → {x : ℝ | f a x < 0} = {x : ℝ | x > 1}) ∧
  (a < 0 → {x : ℝ | f a x < 0} = {x : ℝ | x < 1/a ∨ x > 1}) ∧
  (a = 1 → {x : ℝ | f a x < 0} = ∅) ∧
  (a > 1 → {x : ℝ | f a x < 0} = {x : ℝ | 1/a < x ∧ x < 1}) ∧
  (0 < a ∧ a < 1 → {x : ℝ | f a x < 0} = {x : ℝ | 1 < x ∧ x < 1/a}) := by sorry

end no_always_positive_f_solution_sets_f_negative_l253_25303


namespace rook_domino_tiling_impossibility_l253_25322

theorem rook_domino_tiling_impossibility :
  ∀ (rook_positions : Finset (Fin 10 × Fin 10)),
    (rook_positions.card = 10) →
    (∀ (r1 r2 : Fin 10 × Fin 10), r1 ∈ rook_positions → r2 ∈ rook_positions → r1 ≠ r2 →
      (r1.1 ≠ r2.1 ∧ r1.2 ≠ r2.2)) →
    ¬∃ (domino_placements : Finset (Fin 10 × Fin 10 × Bool)),
      (domino_placements.card = 45) ∧
      (∀ (d : Fin 10 × Fin 10 × Bool), d ∈ domino_placements →
        (d.1, d.2.1) ∉ rook_positions ∧
        (if d.2.2 then (d.1 + 1, d.2.1) ∉ rook_positions
         else (d.1, d.2.1 + 1) ∉ rook_positions)) ∧
      (∀ (p : Fin 10 × Fin 10), p ∉ rook_positions →
        (∃ (d : Fin 10 × Fin 10 × Bool), d ∈ domino_placements ∧
          (d.1, d.2.1) = p ∨
          (if d.2.2 then (d.1 + 1, d.2.1) = p else (d.1, d.2.1 + 1) = p))) :=
by
  sorry


end rook_domino_tiling_impossibility_l253_25322


namespace olivias_birthday_meals_l253_25362

/-- Given that each meal costs 7 dollars and Olivia's dad spent a total of 21 dollars,
    prove that the number of meals he paid for is 3. -/
theorem olivias_birthday_meals (cost_per_meal : ℕ) (total_spent : ℕ) (num_meals : ℕ) :
  cost_per_meal = 7 →
  total_spent = 21 →
  num_meals * cost_per_meal = total_spent →
  num_meals = 3 := by
  sorry

end olivias_birthday_meals_l253_25362


namespace min_value_3a_plus_1_l253_25333

theorem min_value_3a_plus_1 (a : ℝ) (h : 8 * a^2 + 6 * a + 2 = 2) :
  ∃ (m : ℝ), m = -5/4 ∧ ∀ x, (8 * x^2 + 6 * x + 2 = 2) → (3 * x + 1 ≥ m) :=
sorry

end min_value_3a_plus_1_l253_25333


namespace ellipse_equation_l253_25334

/-- The standard equation of an ellipse with given eccentricity and major axis length -/
theorem ellipse_equation (e : ℝ) (major_axis : ℝ) :
  e = 2/3 →
  major_axis = 6 →
  ∃ (a b : ℝ),
    a = major_axis / 2 ∧
    b^2 = a^2 * (1 - e^2) ∧
    ((∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1) ∨
     (∀ x y : ℝ, x^2/b^2 + y^2/a^2 = 1)) :=
by sorry

end ellipse_equation_l253_25334


namespace quadratic_function_absolute_value_l253_25398

theorem quadratic_function_absolute_value (p q : ℝ) :
  ∃ (x : ℝ), x ∈ ({1, 2, 3} : Set ℝ) ∧ |x^2 + p*x + q| ≥ 1/2 :=
sorry

end quadratic_function_absolute_value_l253_25398


namespace suraya_kayla_difference_l253_25354

/-- The number of apples picked by each person -/
structure ApplePicks where
  suraya : ℕ
  caleb : ℕ
  kayla : ℕ

/-- The conditions of the apple-picking scenario -/
def apple_picking_conditions (a : ApplePicks) : Prop :=
  a.suraya = a.caleb + 12 ∧
  a.caleb + 5 = a.kayla ∧
  a.kayla = 20

/-- The theorem stating that Suraya picked 7 more apples than Kayla -/
theorem suraya_kayla_difference (a : ApplePicks) 
  (h : apple_picking_conditions a) : a.suraya - a.kayla = 7 := by
  sorry


end suraya_kayla_difference_l253_25354


namespace daves_video_games_l253_25356

theorem daves_video_games (total_games : ℕ) (price_per_game : ℕ) (total_earnings : ℕ) :
  total_games = 10 →
  price_per_game = 4 →
  total_earnings = 32 →
  total_games - (total_earnings / price_per_game) = 2 :=
by sorry

end daves_video_games_l253_25356


namespace sin_105_degrees_l253_25348

theorem sin_105_degrees : 
  Real.sin (105 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by sorry

end sin_105_degrees_l253_25348
