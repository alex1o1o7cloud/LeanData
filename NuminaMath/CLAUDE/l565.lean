import Mathlib

namespace NUMINAMATH_CALUDE_ln_product_eq_sum_of_ln_l565_56567

-- Define the formal power series type
def FormalPowerSeries (α : Type*) := ℕ → α

-- Define the logarithm operation for formal power series
noncomputable def Ln (f : FormalPowerSeries ℝ) : FormalPowerSeries ℝ := sorry

-- Define the multiplication operation for formal power series
def mul (f g : FormalPowerSeries ℝ) : FormalPowerSeries ℝ := sorry

-- Theorem statement
theorem ln_product_eq_sum_of_ln 
  (f h : FormalPowerSeries ℝ) 
  (hf : f 0 = 1) 
  (hh : h 0 = 1) : 
  Ln (mul f h) = λ n => (Ln f n) + (Ln h n) := by sorry

end NUMINAMATH_CALUDE_ln_product_eq_sum_of_ln_l565_56567


namespace NUMINAMATH_CALUDE_point_on_line_l565_56557

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line : 
  let A : Point := ⟨1, -5⟩
  let B : Point := ⟨3, -1⟩
  let C : Point := ⟨4.5, 2⟩
  collinear A B C := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l565_56557


namespace NUMINAMATH_CALUDE_least_distinct_values_l565_56570

/-- Given a list of 2023 positive integers with a unique mode occurring exactly 15 times,
    the least number of distinct values in the list is 145. -/
theorem least_distinct_values (l : List ℕ+) 
  (h_length : l.length = 2023)
  (h_unique_mode : ∃! m : ℕ+, l.count m = 15 ∧ ∀ n : ℕ+, l.count n ≤ 15) :
  (l.toFinset.card : ℕ) = 145 ∧ 
  ∀ k : ℕ, k < 145 → ¬∃ l' : List ℕ+, 
    l'.length = 2023 ∧ 
    (∃! m : ℕ+, l'.count m = 15 ∧ ∀ n : ℕ+, l'.count n ≤ 15) ∧
    (l'.toFinset.card : ℕ) = k :=
by sorry

end NUMINAMATH_CALUDE_least_distinct_values_l565_56570


namespace NUMINAMATH_CALUDE_cos_alpha_value_l565_56500

theorem cos_alpha_value (α : Real) (h1 : 0 < α ∧ α < π/2) (h2 : Real.sin (α - π/6) = 1/3) :
  Real.cos α = (2 * Real.sqrt 6 - 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l565_56500


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l565_56512

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x^2 - 2 * (a - 2) * x - 4

-- State the theorem
theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, f a x < 0) ↔ a ∈ Set.Ioc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l565_56512


namespace NUMINAMATH_CALUDE_x15x_divisible_by_18_l565_56534

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := 
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def x15x (x : ℕ) : ℕ := x * 1000 + 100 + 50 + x

theorem x15x_divisible_by_18 :
  ∃! x : ℕ, x < 10 ∧ is_four_digit (x15x x) ∧ (x15x x) % 18 = 0 ∧ x = 6 := by
sorry

end NUMINAMATH_CALUDE_x15x_divisible_by_18_l565_56534


namespace NUMINAMATH_CALUDE_g_of_2_eq_neg_1_l565_56533

/-- The function g defined as g(x) = x^2 - 3x + 1 -/
def g (x : ℝ) : ℝ := x^2 - 3*x + 1

/-- Theorem stating that g(2) = -1 -/
theorem g_of_2_eq_neg_1 : g 2 = -1 := by sorry

end NUMINAMATH_CALUDE_g_of_2_eq_neg_1_l565_56533


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l565_56589

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧
    (∀ n : ℕ, a (n + 1) = a n + d) ∧
    a 3 = 5 ∧
    ∃ r : ℝ, (a 2 = a 1 * r) ∧ (a 5 = a 2 * r)

/-- The main theorem -/
theorem arithmetic_sequence_formula (a : ℕ → ℝ) (h : ArithmeticSequence a) :
    ∀ n : ℕ, a n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l565_56589


namespace NUMINAMATH_CALUDE_train_length_proof_l565_56561

/-- Proves that the length of a train is equal to the total length of the train and bridge,
    given the train's speed, time to cross the bridge, and total length of train and bridge. -/
theorem train_length_proof (train_speed : ℝ) (crossing_time : ℝ) (total_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  total_length = 245 →
  total_length = train_speed * crossing_time - total_length + total_length :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l565_56561


namespace NUMINAMATH_CALUDE_parking_lot_motorcycles_l565_56563

/-- The number of cars in the parking lot -/
def num_cars : ℕ := 19

/-- The number of wheels per car -/
def wheels_per_car : ℕ := 5

/-- The number of wheels per motorcycle -/
def wheels_per_motorcycle : ℕ := 2

/-- The total number of wheels for all vehicles -/
def total_wheels : ℕ := 117

/-- The number of motorcycles in the parking lot -/
def num_motorcycles : ℕ := (total_wheels - num_cars * wheels_per_car) / wheels_per_motorcycle

theorem parking_lot_motorcycles : num_motorcycles = 11 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_motorcycles_l565_56563


namespace NUMINAMATH_CALUDE_log_3_81_sqrt_81_equals_6_l565_56597

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_3_81_sqrt_81_equals_6 :
  log 3 (81 * Real.sqrt 81) = 6 := by
  sorry

end NUMINAMATH_CALUDE_log_3_81_sqrt_81_equals_6_l565_56597


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l565_56511

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + 1

-- Define the condition for the inequality
def condition (b : ℝ) : Prop :=
  ∀ x, f b x < 0 ↔ (x < 2 ∨ x > 6)

-- Theorem statement
theorem quadratic_inequality_condition (b : ℝ) :
  condition b → b = 8 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l565_56511


namespace NUMINAMATH_CALUDE_largest_number_with_equal_quotient_and_remainder_l565_56586

theorem largest_number_with_equal_quotient_and_remainder : ∀ A B C : ℕ,
  A = 8 * B + C →
  B = C →
  C < 8 →
  A ≤ 63 ∧ ∃ A₀ : ℕ, A₀ = 63 ∧ ∃ B₀ C₀ : ℕ, A₀ = 8 * B₀ + C₀ ∧ B₀ = C₀ ∧ C₀ < 8 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_equal_quotient_and_remainder_l565_56586


namespace NUMINAMATH_CALUDE_choose_cooks_l565_56574

theorem choose_cooks (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_cooks_l565_56574


namespace NUMINAMATH_CALUDE_coconut_flavored_jelly_beans_l565_56579

theorem coconut_flavored_jelly_beans (total : ℕ) (red_fraction : ℚ) (coconut_fraction : ℚ) :
  total = 4000 →
  red_fraction = 3 / 4 →
  coconut_fraction = 1 / 4 →
  (total * red_fraction * coconut_fraction : ℚ) = 750 := by
  sorry

end NUMINAMATH_CALUDE_coconut_flavored_jelly_beans_l565_56579


namespace NUMINAMATH_CALUDE_stephens_ant_farm_l565_56529

/-- The number of ants in Stephen's ant farm satisfies the given conditions -/
theorem stephens_ant_farm (total_ants : ℕ) : 
  (total_ants / 2 : ℚ) * (80 / 100 : ℚ) = 44 → total_ants = 110 :=
by
  sorry

#check stephens_ant_farm

end NUMINAMATH_CALUDE_stephens_ant_farm_l565_56529


namespace NUMINAMATH_CALUDE_birds_in_trees_l565_56560

theorem birds_in_trees (stones : ℕ) (trees : ℕ) (birds : ℕ) : 
  stones = 40 →
  trees = 3 * stones →
  birds = 2 * (trees + stones) →
  birds = 400 := by
sorry

end NUMINAMATH_CALUDE_birds_in_trees_l565_56560


namespace NUMINAMATH_CALUDE_decreasing_linear_function_iff_negative_slope_l565_56540

/-- A linear function f(x) = ax + b -/
def linear_function (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

/-- A function is decreasing if f(x1) > f(x2) whenever x1 < x2 -/
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2

theorem decreasing_linear_function_iff_negative_slope (m : ℝ) :
  is_decreasing (linear_function (m + 3) (-2)) ↔ m < -3 :=
sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_iff_negative_slope_l565_56540


namespace NUMINAMATH_CALUDE_leahs_outfits_l565_56591

/-- Calculate the number of possible outfits given the number of options for each clothing item -/
def number_of_outfits (trousers shirts jackets shoes : ℕ) : ℕ :=
  trousers * shirts * jackets * shoes

/-- Theorem: The number of outfits for Leah's wardrobe is 840 -/
theorem leahs_outfits :
  number_of_outfits 5 6 4 7 = 840 := by
  sorry

end NUMINAMATH_CALUDE_leahs_outfits_l565_56591


namespace NUMINAMATH_CALUDE_monotonic_square_exists_l565_56510

/-- A function that returns the number of digits of a positive integer in base 10 -/
def numDigits (x : ℕ+) : ℕ := sorry

/-- A function that checks if a positive integer is monotonic in base 10 -/
def isMonotonic (x : ℕ+) : Prop := sorry

/-- For every positive integer n, there exists an n-digit monotonic number which is a perfect square -/
theorem monotonic_square_exists (n : ℕ+) : ∃ x : ℕ+, 
  (numDigits x = n) ∧ 
  isMonotonic x ∧ 
  ∃ y : ℕ+, x = y * y := by
  sorry

end NUMINAMATH_CALUDE_monotonic_square_exists_l565_56510


namespace NUMINAMATH_CALUDE_line_segment_ratio_l565_56594

theorem line_segment_ratio (a b c d : ℝ) :
  let O := 0
  let A := a
  let B := b
  let C := c
  let D := d
  ∀ P : ℝ, B < P ∧ P < C →
  (P - A) / (D - P) = (P - B) / (C - P) →
  P = (a * c - b * d) / (a - b + c - d) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_ratio_l565_56594


namespace NUMINAMATH_CALUDE_quadratic_polynomial_proof_l565_56590

theorem quadratic_polynomial_proof (b c x₁ x₂ : ℝ) : 
  (∃ (a : ℝ), a ≠ 0 ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂) →
  (b + c + x₁ + x₂ = -3) →
  (b * c * x₁ * x₂ = 36) →
  (b = 4 ∧ c = -3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_proof_l565_56590


namespace NUMINAMATH_CALUDE_confectioner_customers_l565_56508

/-- The number of regular customers for a confectioner -/
def regular_customers : ℕ := 28

/-- The total number of pastries -/
def total_pastries : ℕ := 392

/-- The number of customers in the alternative scenario -/
def alternative_customers : ℕ := 49

/-- The difference in pastries per customer between regular and alternative scenarios -/
def pastry_difference : ℕ := 6

theorem confectioner_customers :
  regular_customers = 28 ∧
  total_pastries = 392 ∧
  alternative_customers = 49 ∧
  pastry_difference = 6 ∧
  (total_pastries / regular_customers : ℚ) = 
    (total_pastries / alternative_customers : ℚ) + pastry_difference := by
  sorry

end NUMINAMATH_CALUDE_confectioner_customers_l565_56508


namespace NUMINAMATH_CALUDE_max_cross_section_area_correct_l565_56599

noncomputable def max_cross_section_area (k : ℝ) (α : ℝ) : ℝ :=
  if Real.tan α < 2 then
    (1/2) * k^2 * (1 + 3 * Real.cos α ^ 2)
  else
    2 * k^2 * Real.sin (2 * α)

theorem max_cross_section_area_correct (k : ℝ) (α : ℝ) (h1 : k > 0) (h2 : 0 < α ∧ α < π/2) :
  ∀ A : ℝ, A ≤ max_cross_section_area k α := by
  sorry

end NUMINAMATH_CALUDE_max_cross_section_area_correct_l565_56599


namespace NUMINAMATH_CALUDE_sin_double_angle_proof_l565_56554

theorem sin_double_angle_proof (α : Real) (h : Real.cos (π/4 - α) = 3/5) : 
  Real.sin (2 * α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_proof_l565_56554


namespace NUMINAMATH_CALUDE_no_solution_iff_k_eq_seven_l565_56502

theorem no_solution_iff_k_eq_seven (k : ℝ) : 
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - k) / (x - 8)) ↔ k = 7 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_eq_seven_l565_56502


namespace NUMINAMATH_CALUDE_jim_journey_distance_l565_56576

/-- The total distance of Jim's journey -/
def total_distance (miles_driven : ℕ) (miles_remaining : ℕ) : ℕ :=
  miles_driven + miles_remaining

/-- Theorem: The total distance of Jim's journey is 1200 miles -/
theorem jim_journey_distance :
  total_distance 768 432 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_jim_journey_distance_l565_56576


namespace NUMINAMATH_CALUDE_joses_swimming_pool_charge_l565_56532

/-- Proves that the daily charge for kids in Jose's swimming pool is $3 -/
theorem joses_swimming_pool_charge (kid_charge : ℚ) (adult_charge : ℚ) 
  (h1 : adult_charge = 2 * kid_charge) 
  (h2 : 8 * kid_charge + 10 * adult_charge = 588 / 7) : 
  kid_charge = 3 := by
  sorry

end NUMINAMATH_CALUDE_joses_swimming_pool_charge_l565_56532


namespace NUMINAMATH_CALUDE_number_of_girls_in_school_l565_56536

/-- Proves the number of girls in a school given specific conditions --/
theorem number_of_girls_in_school (total_students : ℕ) 
  (avg_age_boys avg_age_girls avg_age_school : ℚ) :
  total_students = 604 →
  avg_age_boys = 12 →
  avg_age_girls = 11 →
  avg_age_school = 47/4 →
  ∃ (num_girls : ℕ), num_girls = 151 ∧ 
    (num_girls : ℚ) * avg_age_girls + (total_students - num_girls : ℚ) * avg_age_boys = 
      total_students * avg_age_school :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_school_l565_56536


namespace NUMINAMATH_CALUDE_barbara_wins_2023_barbara_wins_2024_l565_56551

/-- Represents the players in the coin removal game -/
inductive Player
| Barbara
| Jenna

/-- Represents the state of the game -/
structure GameState where
  coins : ℕ
  currentPlayer : Player

/-- Defines a valid move for a player -/
def validMove (player : Player) (coins : ℕ) : Set ℕ :=
  match player with
  | Player.Barbara => {2, 4, 5}
  | Player.Jenna => {1, 3, 5}

/-- Determines if a game state is winning for the current player -/
def isWinningState : GameState → Prop :=
  sorry

/-- Theorem stating that Barbara wins with 2023 coins -/
theorem barbara_wins_2023 :
  isWinningState ⟨2023, Player.Barbara⟩ :=
  sorry

/-- Theorem stating that Barbara wins with 2024 coins -/
theorem barbara_wins_2024 :
  isWinningState ⟨2024, Player.Barbara⟩ :=
  sorry

end NUMINAMATH_CALUDE_barbara_wins_2023_barbara_wins_2024_l565_56551


namespace NUMINAMATH_CALUDE_ninety_squared_l565_56505

theorem ninety_squared : 90 * 90 = 8100 := by
  sorry

end NUMINAMATH_CALUDE_ninety_squared_l565_56505


namespace NUMINAMATH_CALUDE_arithmetic_sequence_average_l565_56525

/-- The average of an arithmetic sequence with 21 terms, 
    starting at -180 and ending at 180, with a common difference of 6, is 0. -/
theorem arithmetic_sequence_average : 
  let first_term : ℤ := -180
  let last_term : ℤ := 180
  let num_terms : ℕ := 21
  let common_diff : ℤ := 6
  let sequence := fun i => first_term + (i : ℤ) * common_diff
  (first_term + last_term) / 2 = 0 ∧ 
  last_term = first_term + (num_terms - 1 : ℕ) * common_diff :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_average_l565_56525


namespace NUMINAMATH_CALUDE_rectangular_solid_width_l565_56542

/-- The surface area of a rectangular solid given its length, width, and height. -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The width of a rectangular solid with length 5, depth 1, and surface area 58 is 4. -/
theorem rectangular_solid_width :
  ∃ w : ℝ, w = 4 ∧ surface_area 5 w 1 = 58 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_width_l565_56542


namespace NUMINAMATH_CALUDE_tv_cost_l565_56575

theorem tv_cost (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  savings = 880 → 
  furniture_fraction = 3/4 → 
  tv_cost = savings * (1 - furniture_fraction) → 
  tv_cost = 220 := by
sorry

end NUMINAMATH_CALUDE_tv_cost_l565_56575


namespace NUMINAMATH_CALUDE_sticker_distribution_l565_56543

/-- The number of ways to distribute n identical objects into k distinct containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- There are 8 identical stickers -/
def num_stickers : ℕ := 8

/-- There are 4 sheets of paper -/
def num_sheets : ℕ := 4

theorem sticker_distribution :
  distribute num_stickers num_sheets = 15 :=
sorry

end NUMINAMATH_CALUDE_sticker_distribution_l565_56543


namespace NUMINAMATH_CALUDE_fraction_power_simplification_l565_56582

theorem fraction_power_simplification :
  (66666 : ℕ) = 3 * 22222 →
  (66666 : ℚ)^4 / (22222 : ℚ)^4 = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_power_simplification_l565_56582


namespace NUMINAMATH_CALUDE_sqrt_3_simplest_l565_56509

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℝ) : Prop := ∃ m : ℤ, n = m^2

-- Define what it means for a quadratic radical to be in its simplest form
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  x > 0 ∧ ¬(is_perfect_square x) ∧ ∀ y z : ℝ, (y > 1 ∧ z > 1 ∧ x = y * z) → ¬(is_perfect_square y)

-- State the theorem
theorem sqrt_3_simplest :
  is_simplest_quadratic_radical 3 ∧
  ¬(is_simplest_quadratic_radical (1/2)) ∧
  ¬(is_simplest_quadratic_radical 8) ∧
  ¬(is_simplest_quadratic_radical 4) :=
sorry

end NUMINAMATH_CALUDE_sqrt_3_simplest_l565_56509


namespace NUMINAMATH_CALUDE_base_conversion_sum_l565_56541

-- Define the base conversion function
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the given numbers in their respective bases
def n1 : Nat := to_base_10 [2, 1, 4] 8
def n2 : Nat := to_base_10 [3, 2] 5
def n3 : Nat := to_base_10 [3, 4, 3] 9
def n4 : Nat := to_base_10 [1, 3, 3] 4

-- State the theorem
theorem base_conversion_sum :
  (n1 : ℚ) / n2 + (n3 : ℚ) / n4 = 9134 / 527 := by sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l565_56541


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l565_56521

/-- Given that the coefficient of x^-3 in the expansion of (2x - a/x)^7 is 84, prove that a = -1 -/
theorem binomial_expansion_coefficient (a : ℝ) : 
  (Nat.choose 7 5 : ℝ) * 2^2 * (-a)^5 = 84 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l565_56521


namespace NUMINAMATH_CALUDE_man_rowing_speed_l565_56559

/-- 
Given a man's rowing speed against the stream and his speed in still water,
calculate his speed with the stream.
-/
theorem man_rowing_speed 
  (speed_against_stream : ℝ) 
  (speed_still_water : ℝ) 
  (h1 : speed_against_stream = 4) 
  (h2 : speed_still_water = 6) : 
  speed_still_water + (speed_still_water - speed_against_stream) = 8 := by
  sorry

#check man_rowing_speed

end NUMINAMATH_CALUDE_man_rowing_speed_l565_56559


namespace NUMINAMATH_CALUDE_specific_female_selection_probability_l565_56544

def total_students : ℕ := 50
def male_students : ℕ := 30
def selected_students : ℕ := 5

theorem specific_female_selection_probability :
  (selected_students : ℚ) / total_students = 1 / 10 :=
by sorry

end NUMINAMATH_CALUDE_specific_female_selection_probability_l565_56544


namespace NUMINAMATH_CALUDE_eight_b_equals_sixteen_l565_56530

theorem eight_b_equals_sixteen (a b : ℝ) 
  (h1 : 6 * a + 3 * b = 0) 
  (h2 : a = b - 3) : 
  8 * b = 16 := by
sorry

end NUMINAMATH_CALUDE_eight_b_equals_sixteen_l565_56530


namespace NUMINAMATH_CALUDE_fraction_subtraction_equality_l565_56517

theorem fraction_subtraction_equality : 
  (4 + 6 + 8 + 10) / (3 + 5 + 7) - (3 + 5 + 7 + 9) / (4 + 6 + 8) = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_equality_l565_56517


namespace NUMINAMATH_CALUDE_inscribed_octahedron_side_length_l565_56553

-- Define the rectangular prism
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

-- Define the octahedron
structure Octahedron where
  sideLength : ℝ

-- Define the function to calculate the side length of the inscribed octahedron
def inscribedOctahedronSideLength (prism : RectangularPrism) : ℝ :=
  sorry

-- Theorem statement
theorem inscribed_octahedron_side_length 
  (prism : RectangularPrism) 
  (h1 : prism.length = 2) 
  (h2 : prism.width = 3) 
  (h3 : prism.height = 1) :
  inscribedOctahedronSideLength prism = Real.sqrt 14 / 2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_octahedron_side_length_l565_56553


namespace NUMINAMATH_CALUDE_digit_222_is_zero_l565_56565

/-- The decimal representation of 41/777 -/
def decimal_rep : ℚ := 41 / 777

/-- The length of the repeating block in the decimal representation of 41/777 -/
def repeating_block_length : ℕ := 42

/-- The position of the 222nd digit within the repeating block -/
def position_in_block : ℕ := 222 % repeating_block_length

/-- The 222nd digit after the decimal point in the decimal representation of 41/777 -/
def digit_222 : ℕ := 0

/-- Theorem stating that the 222nd digit after the decimal point 
    in the decimal representation of 41/777 is 0 -/
theorem digit_222_is_zero : digit_222 = 0 := by sorry

end NUMINAMATH_CALUDE_digit_222_is_zero_l565_56565


namespace NUMINAMATH_CALUDE_arithmetic_mean_x_y_l565_56518

/-- Given two real numbers x and y satisfying certain conditions, 
    prove that their arithmetic mean is 3/4 -/
theorem arithmetic_mean_x_y (x y : ℝ) 
  (h1 : x * y > 0)
  (h2 : 2 * x * (1/2) + 1 * (-1/(2*y)) = 0)  -- Perpendicularity condition
  (h3 : y / x = 2 / y)  -- Geometric sequence condition
  : (x + y) / 2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_x_y_l565_56518


namespace NUMINAMATH_CALUDE_stone_121_is_10_l565_56523

/-- The number of stones in the sequence -/
def n : ℕ := 11

/-- The length of a full cycle (left-to-right and right-to-left) -/
def cycle_length : ℕ := 2 * n - 1

/-- The position of a stone in the original left-to-right count, given its count number -/
def stone_position (count : ℕ) : ℕ :=
  (count - 1) % cycle_length + 1

/-- The theorem stating that the 121st count corresponds to the 10th stone -/
theorem stone_121_is_10 : stone_position 121 = 10 := by
  sorry

end NUMINAMATH_CALUDE_stone_121_is_10_l565_56523


namespace NUMINAMATH_CALUDE_pq_length_l565_56558

/-- Given two lines and a point R that is the midpoint of a line segment PQ, 
    where P is on one line and Q is on the other, prove that the length of PQ 
    is √56512 / 33. -/
theorem pq_length (P Q R : ℝ × ℝ) : 
  R = (10, 8) →
  (∃ x, P = (x, 2*x)) →
  (∃ y, Q = (y, 4*y/11)) →
  R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = Real.sqrt 56512 / 33 := by
  sorry

#check pq_length

end NUMINAMATH_CALUDE_pq_length_l565_56558


namespace NUMINAMATH_CALUDE_sum_of_even_coefficients_l565_56528

theorem sum_of_even_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x - 1) * (x + 1)^9 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + 
                                   a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₂ + a₄ + a₆ + a₈ + a₁₀ = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_coefficients_l565_56528


namespace NUMINAMATH_CALUDE_math_physics_majors_consecutive_probability_l565_56593

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def choose (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem math_physics_majors_consecutive_probability :
  let total_people : ℕ := 12
  let math_majors : ℕ := 5
  let physics_majors : ℕ := 4
  let biology_majors : ℕ := 3
  let favorable_outcomes : ℕ := choose total_people math_majors * factorial (math_majors - 1) * 
                                 choose (total_people - math_majors) physics_majors * 
                                 factorial (physics_majors - 1) * factorial biology_majors
  let total_outcomes : ℕ := factorial (total_people - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_math_physics_majors_consecutive_probability_l565_56593


namespace NUMINAMATH_CALUDE_min_value_quadratic_l565_56598

theorem min_value_quadratic (k : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 6 * y + 9 ≥ 0) ∧
  (∃ x y : ℝ, 3 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 6 * y + 9 = 0) →
  k = 3/2 ∨ k = -3/2 := by
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l565_56598


namespace NUMINAMATH_CALUDE_evaluate_expression_l565_56595

theorem evaluate_expression : -(16 / 4 * 8 - 70 + 4^2 * 7) = -74 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l565_56595


namespace NUMINAMATH_CALUDE_average_of_ABC_l565_56592

theorem average_of_ABC (A B C : ℚ) 
  (eq1 : 2023 * C - 4046 * A = 8092)
  (eq2 : 2023 * B - 6069 * A = 10115) :
  (A + B + C) / 3 = 2 * A + 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_ABC_l565_56592


namespace NUMINAMATH_CALUDE_bobs_hair_length_at_last_cut_l565_56515

/-- The length of Bob's hair at his last haircut, given his current hair length,
    hair growth rate, and time since last haircut. -/
def hair_length_at_last_cut (current_length : ℝ) (growth_rate : ℝ) (years_since_cut : ℝ) : ℝ :=
  current_length - growth_rate * 12 * years_since_cut

/-- Theorem stating that Bob's hair length at his last haircut was 6 inches,
    given the provided conditions. -/
theorem bobs_hair_length_at_last_cut :
  hair_length_at_last_cut 36 0.5 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bobs_hair_length_at_last_cut_l565_56515


namespace NUMINAMATH_CALUDE_garden_vegetable_difference_l565_56537

/-- Represents the number of vegetables in a garden -/
structure GardenVegetables where
  potatoes : ℕ
  cucumbers : ℕ
  peppers : ℕ

/-- Theorem stating the difference between potatoes and cucumbers in the garden -/
theorem garden_vegetable_difference (g : GardenVegetables) :
  g.potatoes = 237 →
  g.peppers = 2 * g.cucumbers →
  g.potatoes + g.cucumbers + g.peppers = 768 →
  g.potatoes - g.cucumbers = 60 := by
  sorry

#check garden_vegetable_difference

end NUMINAMATH_CALUDE_garden_vegetable_difference_l565_56537


namespace NUMINAMATH_CALUDE_certain_number_is_30_l565_56527

theorem certain_number_is_30 (x : ℝ) : 0.5 * x = 0.1667 * x + 10 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_30_l565_56527


namespace NUMINAMATH_CALUDE_painting_time_equation_l565_56522

theorem painting_time_equation (doug_time dave_time lunch_break : ℝ) 
  (h_doug : doug_time = 6)
  (h_dave : dave_time = 8)
  (h_lunch : lunch_break = 2)
  (t : ℝ) :
  (1 / doug_time + 1 / dave_time) * (t - lunch_break) = 1 :=
by sorry

end NUMINAMATH_CALUDE_painting_time_equation_l565_56522


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l565_56546

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define parallelism between two lines
def parallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define when a point lies on a line
def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem statement
theorem parallel_line_through_point :
  ∃ (l : Line2D),
    parallel l (Line2D.mk 2 1 (-1)) ∧
    point_on_line (Point2D.mk 1 2) l ∧
    l = Line2D.mk 2 1 (-4) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l565_56546


namespace NUMINAMATH_CALUDE_smallest_other_integer_l565_56569

theorem smallest_other_integer (x : ℕ) (a b : ℕ) : 
  a = 45 →
  a > 0 →
  b > 0 →
  x > 0 →
  Nat.gcd a b = x + 5 →
  Nat.lcm a b = x * (x + 5) →
  a + b < 100 →
  ∃ (b_min : ℕ), b_min = 12 ∧ ∀ (b' : ℕ), b' ≠ a ∧ 
    Nat.gcd a b' = x + 5 ∧
    Nat.lcm a b' = x * (x + 5) ∧
    a + b' < 100 →
    b' ≥ b_min :=
sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l565_56569


namespace NUMINAMATH_CALUDE_max_value_theorem_l565_56583

theorem max_value_theorem (x y : ℝ) (h : x^2 + 4*y^2 = 4) :
  ∃ (max : ℝ), max = (1 + Real.sqrt 2) / 2 ∧
  ∀ (z : ℝ), z = x*y / (x + 2*y - 2) → z ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l565_56583


namespace NUMINAMATH_CALUDE_dream_team_strategy_l565_56531

/-- Represents the probabilities of correct answers for each team member and category -/
structure TeamProbabilities where
  a_category_a : ℝ
  a_category_b : ℝ
  b_category_a : ℝ
  b_category_b : ℝ

/-- Calculates the probability of entering the final round when answering a specific category first -/
def probability_enter_final (probs : TeamProbabilities) (start_with_a : Bool) : ℝ :=
  if start_with_a then
    let p3 := probs.a_category_a * probs.b_category_a * probs.a_category_b * (1 - probs.b_category_b) +
              probs.a_category_a * probs.b_category_a * (1 - probs.a_category_b) * probs.b_category_b
    let p4 := probs.a_category_a * probs.b_category_a * probs.a_category_b * probs.b_category_b
    p3 + p4
  else
    let p3 := probs.a_category_b * probs.b_category_b * probs.a_category_a * (1 - probs.b_category_a) +
              probs.a_category_b * probs.b_category_b * (1 - probs.a_category_a) * probs.b_category_a
    let p4 := probs.a_category_b * probs.b_category_b * probs.a_category_a * probs.b_category_a
    p3 + p4

/-- The main theorem to be proved -/
theorem dream_team_strategy (probs : TeamProbabilities)
  (h1 : probs.a_category_a = 0.7)
  (h2 : probs.a_category_b = 0.5)
  (h3 : probs.b_category_a = 0.4)
  (h4 : probs.b_category_b = 0.8) :
  probability_enter_final probs false > probability_enter_final probs true :=
by sorry

end NUMINAMATH_CALUDE_dream_team_strategy_l565_56531


namespace NUMINAMATH_CALUDE_vector_sum_in_R2_l565_56539

/-- Given two vectors in R², prove their sum is correct -/
theorem vector_sum_in_R2 (a b : Fin 2 → ℝ) (ha : a = ![5, 2]) (hb : b = ![1, 6]) :
  a + b = ![6, 8] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_in_R2_l565_56539


namespace NUMINAMATH_CALUDE_red_ball_probability_l565_56573

/-- The probability of selecting a red ball from a bag -/
def probability_red_ball (total_balls : ℕ) (red_balls : ℕ) : ℚ :=
  red_balls / total_balls

/-- Theorem: The probability of selecting a red ball from a bag with 15 balls, 
    of which 3 are red, is 1/5 -/
theorem red_ball_probability :
  probability_red_ball 15 3 = 1 / 5 := by
  sorry

#eval probability_red_ball 15 3

end NUMINAMATH_CALUDE_red_ball_probability_l565_56573


namespace NUMINAMATH_CALUDE_find_missing_number_l565_56585

theorem find_missing_number (x : ℕ) : 
  (55 + 48 + x + 2 + 684 + 42) / 6 = 223 → x = 507 := by
  sorry

end NUMINAMATH_CALUDE_find_missing_number_l565_56585


namespace NUMINAMATH_CALUDE_last_three_digits_of_8_to_104_l565_56571

theorem last_three_digits_of_8_to_104 : 8^104 ≡ 984 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_8_to_104_l565_56571


namespace NUMINAMATH_CALUDE_ladder_tournament_rankings_ten_player_tournament_rankings_l565_56516

/-- The number of possible rankings in a ladder-style tournament with n players. -/
def num_rankings (n : ℕ) : ℕ :=
  if n < 2 then 0 else 2^(n-1)

/-- Theorem: The number of possible rankings in a ladder-style tournament with n players (n ≥ 2) is 2^(n-1). -/
theorem ladder_tournament_rankings (n : ℕ) (h : n ≥ 2) :
  num_rankings n = 2^(n-1) := by
  sorry

/-- Corollary: For a tournament with 10 players, there are 512 possible rankings. -/
theorem ten_player_tournament_rankings :
  num_rankings 10 = 512 := by
  sorry

end NUMINAMATH_CALUDE_ladder_tournament_rankings_ten_player_tournament_rankings_l565_56516


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l565_56588

/-- The length of the major axis of an ellipse with given foci and tangent to x-axis -/
theorem ellipse_major_axis_length : 
  let f1 : ℝ × ℝ := (5, 15)
  let f2 : ℝ × ℝ := (40, 45)
  ∀ (E : Set (ℝ × ℝ)), 
    (∀ p ∈ E, dist p f1 + dist p f2 = dist p f1 + dist p f2) →  -- E is an ellipse with foci f1 and f2
    (∃ x, (x, 0) ∈ E) →  -- E is tangent to x-axis
    (∃ a : ℝ, ∀ p ∈ E, dist p f1 + dist p f2 = 2 * a) →  -- Definition of ellipse
    2 * (dist f1 f2) = 10 * Real.sqrt 193 :=
by
  sorry


end NUMINAMATH_CALUDE_ellipse_major_axis_length_l565_56588


namespace NUMINAMATH_CALUDE_money_distribution_l565_56549

theorem money_distribution (P Q R S : ℕ) : 
  P = 2 * Q →  -- P gets twice as that of Q
  S = 4 * R →  -- S gets 4 times as that of R
  Q = R →      -- Q and R are to receive equal amounts
  S - P = 250 →  -- The difference between S and P is 250
  P + Q + R + S = 1000 :=  -- Total amount to be distributed
by sorry

end NUMINAMATH_CALUDE_money_distribution_l565_56549


namespace NUMINAMATH_CALUDE_binomial_6_choose_3_l565_56501

theorem binomial_6_choose_3 : Nat.choose 6 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_6_choose_3_l565_56501


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l565_56514

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + x₁ - 1 = 0) ∧ (x₂^2 + x₂ - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l565_56514


namespace NUMINAMATH_CALUDE_cross_section_area_is_40_div_3_l565_56596

/-- Right prism with isosceles triangle base -/
structure RightPrism where
  -- Base triangle
  AB : ℝ
  BC : ℝ
  angleABC : ℝ
  -- Intersection points
  AD_ratio : ℝ
  EC1_ratio : ℝ
  -- Conditions
  isIsosceles : AB = BC
  baseLength : AB = 5
  angleCondition : angleABC = 2 * Real.arcsin (3/5)
  adIntersection : AD_ratio = 1/3
  ec1Intersection : EC1_ratio = 1/3

/-- The area of the cross-section of the prism -/
def crossSectionArea (p : RightPrism) : ℝ :=
  sorry -- Actual calculation would go here

/-- Theorem stating the area of the cross-section -/
theorem cross_section_area_is_40_div_3 (p : RightPrism) :
  crossSectionArea p = 40/3 := by
  sorry

#check cross_section_area_is_40_div_3

end NUMINAMATH_CALUDE_cross_section_area_is_40_div_3_l565_56596


namespace NUMINAMATH_CALUDE_equation_solutions_l565_56506

def solutions_7 : Set (ℤ × ℤ) := {(6, -42), (-42, 6), (8, 56), (56, 8), (14, 14)}
def solutions_25 : Set (ℤ × ℤ) := {(24, -600), (-600, 24), (26, 650), (650, 26), (50, 50)}

theorem equation_solutions (a b : ℤ) :
  (1 / a + 1 / b = 1 / 7 → (a, b) ∈ solutions_7) ∧
  (1 / a + 1 / b = 1 / 25 → (a, b) ∈ solutions_25) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l565_56506


namespace NUMINAMATH_CALUDE_binomial_product_l565_56564

theorem binomial_product (x : ℝ) : (2*x^2 + 3*x - 4)*(x + 6) = 2*x^3 + 15*x^2 + 14*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l565_56564


namespace NUMINAMATH_CALUDE_approx_cube_root_25_correct_l565_56547

/-- Approximate value of the cube root of 25 -/
def approx_cube_root_25 : ℝ := 2.926

/-- Generalized binomial theorem approximation for small x -/
def binomial_approx (α x : ℝ) : ℝ := 1 + α * x

/-- Cube root of 27 -/
def cube_root_27 : ℝ := 3

theorem approx_cube_root_25_correct :
  let x := -2/27
  let α := 1/3
  approx_cube_root_25 = cube_root_27 * binomial_approx α x := by sorry

end NUMINAMATH_CALUDE_approx_cube_root_25_correct_l565_56547


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l565_56568

/-- A quadrilateral with an inscribed circle -/
structure InscribedCircleQuadrilateral where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- Length of AP -/
  ap : ℝ
  /-- Length of PB -/
  pb : ℝ
  /-- Length of CQ -/
  cq : ℝ
  /-- Length of QD -/
  qd : ℝ
  /-- The circle is tangent to AB at P and to CD at Q -/
  tangent_condition : True

/-- The theorem stating that for the given quadrilateral, the square of the radius is 13325 -/
theorem inscribed_circle_radius_squared
  (quad : InscribedCircleQuadrilateral)
  (h1 : quad.ap = 25)
  (h2 : quad.pb = 35)
  (h3 : quad.cq = 30)
  (h4 : quad.qd = 40) :
  quad.r ^ 2 = 13325 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l565_56568


namespace NUMINAMATH_CALUDE_circles_intersect_l565_56538

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 9

-- Define the centers and radii
def center1 : ℝ × ℝ := (-2, 0)
def center2 : ℝ × ℝ := (2, 1)
def radius1 : ℝ := 2
def radius2 : ℝ := 3

-- Theorem stating that the circles intersect
theorem circles_intersect :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l565_56538


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l565_56545

theorem gcd_of_polynomial_and_multiple (y : ℤ) : 
  (∃ k : ℤ, y = 30492 * k) →
  Int.gcd ((3*y+4)*(8*y+3)*(11*y+5)*(y+11)) y = 660 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l565_56545


namespace NUMINAMATH_CALUDE_american_flag_problem_l565_56552

theorem american_flag_problem (total_stripes : ℕ) (total_red_stripes : ℕ) : 
  total_stripes = 13 →
  total_red_stripes = 70 →
  (total_stripes - 1) / 2 + 1 = 7 →
  total_red_stripes / ((total_stripes - 1) / 2 + 1) = 10 := by
sorry

end NUMINAMATH_CALUDE_american_flag_problem_l565_56552


namespace NUMINAMATH_CALUDE_orange_groups_l565_56548

theorem orange_groups (total_oranges : ℕ) (num_groups : ℕ) 
  (h1 : total_oranges = 384) (h2 : num_groups = 16) :
  total_oranges / num_groups = 24 := by
sorry

end NUMINAMATH_CALUDE_orange_groups_l565_56548


namespace NUMINAMATH_CALUDE_monotonic_increasing_range_always_positive_range_l565_56503

def f (k : ℝ) (x : ℝ) : ℝ := x^2 + 2*k*x + 4

-- Part 1
theorem monotonic_increasing_range (k : ℝ) :
  (∀ x ∈ Set.Icc 1 4, Monotone (f k)) ↔ k ≥ -1 :=
sorry

-- Part 2
theorem always_positive_range (k : ℝ) :
  (∀ x : ℝ, f k x > 0) ↔ -2 < k ∧ k < 2 :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_range_always_positive_range_l565_56503


namespace NUMINAMATH_CALUDE_max_food_per_guest_l565_56581

theorem max_food_per_guest (total_food : ℕ) (min_guests : ℕ) (max_food : ℕ) : 
  total_food = 406 → 
  min_guests = 163 → 
  max_food = 2 → 
  (total_food : ℚ) / min_guests ≤ max_food :=
by sorry

end NUMINAMATH_CALUDE_max_food_per_guest_l565_56581


namespace NUMINAMATH_CALUDE_abs_two_implies_two_or_neg_two_l565_56578

theorem abs_two_implies_two_or_neg_two (x : ℝ) : |x| = 2 → x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_abs_two_implies_two_or_neg_two_l565_56578


namespace NUMINAMATH_CALUDE_eldorado_license_plates_l565_56587

theorem eldorado_license_plates : 
  let letter_choices : ℕ := 26
  let digit_choices : ℕ := 10
  let letter_spots : ℕ := 3
  let digit_spots : ℕ := 4
  letter_choices ^ letter_spots * digit_choices ^ digit_spots = 175760000 :=
by sorry

end NUMINAMATH_CALUDE_eldorado_license_plates_l565_56587


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l565_56519

theorem smallest_angle_solution (x : Real) : 
  (8 * Real.sin x ^ 2 * Real.cos x ^ 4 - 8 * Real.sin x ^ 4 * Real.cos x ^ 2 = 1) →
  (x ≥ 0) →
  (∀ y, y > 0 ∧ y < x → 8 * Real.sin y ^ 2 * Real.cos y ^ 4 - 8 * Real.sin y ^ 4 * Real.cos y ^ 2 ≠ 1) →
  x = 10 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l565_56519


namespace NUMINAMATH_CALUDE_inverse_function_point_l565_56513

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the condition that f(x-1) passes through (1, 2)
def passes_through_point (f : ℝ → ℝ) : Prop :=
  f (1 - 1) = 2

-- Define the inverse function of f
noncomputable def f_inverse (f : ℝ → ℝ) : ℝ → ℝ :=
  Function.invFun f

-- Theorem statement
theorem inverse_function_point (f : ℝ → ℝ) :
  passes_through_point f → f_inverse f 2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_function_point_l565_56513


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l565_56535

-- Define a geometric sequence
def isGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_geo : isGeometric a)
  (h_pos : ∀ n, a n > 0)
  (h_sum : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) :
  a 3 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l565_56535


namespace NUMINAMATH_CALUDE_probability_two_dice_rolls_l565_56584

-- Define the number of sides on each die
def sides : ℕ := 8

-- Define the favorable outcomes for the first die (numbers less than 4)
def favorable_first : ℕ := 3

-- Define the favorable outcomes for the second die (numbers greater than 5)
def favorable_second : ℕ := 3

-- State the theorem
theorem probability_two_dice_rolls : 
  (favorable_first / sides) * (favorable_second / sides) = 9 / 64 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_dice_rolls_l565_56584


namespace NUMINAMATH_CALUDE_fruit_pricing_problem_l565_56562

theorem fruit_pricing_problem (x y : ℚ) : 
  x + y = 1000 →
  (11/9) * x + (4/7) * y = 999 →
  (9 * (11/9) = 11 ∧ 7 * (4/7) = 4) :=
by sorry

end NUMINAMATH_CALUDE_fruit_pricing_problem_l565_56562


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l565_56550

theorem number_puzzle_solution : 
  ∃ x : ℚ, 3 * (2 * x + 7) = 99 ∧ x = 13 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l565_56550


namespace NUMINAMATH_CALUDE_pages_to_read_tonight_l565_56555

/-- The number of pages in Juwella's book -/
def total_pages : ℕ := 500

/-- The number of pages Juwella read three nights ago -/
def pages_three_nights_ago : ℕ := 20

/-- The number of pages Juwella read two nights ago -/
def pages_two_nights_ago : ℕ := pages_three_nights_ago^2 + 5

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The number of pages Juwella read last night -/
def pages_last_night : ℕ := 3 * sum_of_digits pages_two_nights_ago

/-- The total number of pages Juwella has read so far -/
def total_pages_read : ℕ := pages_three_nights_ago + pages_two_nights_ago + pages_last_night

/-- Theorem stating the number of pages Juwella will read tonight -/
theorem pages_to_read_tonight : total_pages - total_pages_read = 48 := by
  sorry

end NUMINAMATH_CALUDE_pages_to_read_tonight_l565_56555


namespace NUMINAMATH_CALUDE_toy_cost_correct_l565_56526

/-- The cost of the assortment box of toys for Julia's new puppy -/
def toy_cost : ℝ := 40

/-- The adoption fee for the puppy -/
def adoption_fee : ℝ := 20

/-- The cost of dog food -/
def dog_food_cost : ℝ := 20

/-- The cost of one bag of treats -/
def treat_cost : ℝ := 2.5

/-- The number of treat bags purchased -/
def treat_bags : ℕ := 2

/-- The cost of the crate -/
def crate_cost : ℝ := 20

/-- The cost of the bed -/
def bed_cost : ℝ := 20

/-- The cost of the collar/leash combo -/
def collar_leash_cost : ℝ := 15

/-- The discount percentage as a decimal -/
def discount_rate : ℝ := 0.2

/-- The total amount Julia spent on the puppy -/
def total_spent : ℝ := 96

theorem toy_cost_correct : 
  (1 - discount_rate) * (adoption_fee + dog_food_cost + treat_cost * treat_bags + 
  crate_cost + bed_cost + collar_leash_cost + toy_cost) = total_spent := by
  sorry

end NUMINAMATH_CALUDE_toy_cost_correct_l565_56526


namespace NUMINAMATH_CALUDE_pizza_slices_l565_56520

theorem pizza_slices (total_pizzas : ℕ) (total_slices : ℕ) (slices_per_pizza : ℕ) : 
  total_pizzas = 21 → total_slices = 168 → slices_per_pizza * total_pizzas = total_slices → slices_per_pizza = 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l565_56520


namespace NUMINAMATH_CALUDE_total_distance_walked_l565_56504

theorem total_distance_walked (first_part second_part : Real) 
  (h1 : first_part = 0.75)
  (h2 : second_part = 0.25) : 
  first_part + second_part = 1 := by
sorry

end NUMINAMATH_CALUDE_total_distance_walked_l565_56504


namespace NUMINAMATH_CALUDE_unique_prime_with_square_divisor_sum_l565_56556

theorem unique_prime_with_square_divisor_sum : 
  ∃! p : ℕ, Prime p ∧ 
  ∃ n : ℕ, (1 + p + p^2 + p^3 + p^4 : ℕ) = n^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_square_divisor_sum_l565_56556


namespace NUMINAMATH_CALUDE_factory_machines_l565_56566

/-- Represents the number of machines in the factory -/
def num_machines : ℕ := 7

/-- Represents the time (in hours) taken by 6 machines to fill the order -/
def time_6_machines : ℕ := 42

/-- Represents the time (in hours) taken by all machines to fill the order -/
def time_all_machines : ℕ := 36

/-- Theorem stating that the number of machines in the factory is 7 -/
theorem factory_machines :
  (6 : ℚ) * time_all_machines * num_machines = time_6_machines * num_machines - 
  6 * time_6_machines := by sorry

end NUMINAMATH_CALUDE_factory_machines_l565_56566


namespace NUMINAMATH_CALUDE_school_bus_seats_l565_56572

/-- Given a school with students and buses, calculate the number of seats per bus. -/
def seats_per_bus (total_students : ℕ) (num_buses : ℕ) : ℕ :=
  total_students / num_buses

/-- Theorem stating that for a school with 11210 students and 95 buses, each bus has 118 seats. -/
theorem school_bus_seats :
  seats_per_bus 11210 95 = 118 := by
  sorry

end NUMINAMATH_CALUDE_school_bus_seats_l565_56572


namespace NUMINAMATH_CALUDE_five_digit_divisibility_l565_56524

/-- A function that removes the middle digit of a five-digit number -/
def removeMidDigit (n : ℕ) : ℕ :=
  (n / 10000) * 1000 + (n % 1000)

/-- A predicate that checks if a number is five-digit -/
def isFiveDigit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem five_digit_divisibility (A : ℕ) :
  isFiveDigit A →
  (∃ k : ℕ, A = k * (removeMidDigit A)) ↔ (∃ m : ℕ, A = m * 1000) :=
by sorry

end NUMINAMATH_CALUDE_five_digit_divisibility_l565_56524


namespace NUMINAMATH_CALUDE_system_solution_correct_l565_56577

theorem system_solution_correct (x y : ℝ) : 
  x = 3 ∧ y = 1 → (2 * x - 3 * y = 3 ∧ x + 2 * y = 5) := by
sorry

end NUMINAMATH_CALUDE_system_solution_correct_l565_56577


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l565_56507

/-- Given two parabolas that intersect at four points, prove that these points lie on a circle with radius squared equal to 5/2 -/
theorem intersection_points_on_circle (x y : ℝ) : 
  (y = (x - 2)^2) ∧ (x - 3 = (y + 1)^2) →
  ∃ (center : ℝ × ℝ), 
    (x - center.1)^2 + (y - center.2)^2 = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l565_56507


namespace NUMINAMATH_CALUDE_octal_to_decimal_1743_l565_56580

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The octal representation of the number -/
def octal_digits : List Nat := [3, 4, 7, 1]

theorem octal_to_decimal_1743 :
  octal_to_decimal octal_digits = 995 := by
  sorry

end NUMINAMATH_CALUDE_octal_to_decimal_1743_l565_56580
