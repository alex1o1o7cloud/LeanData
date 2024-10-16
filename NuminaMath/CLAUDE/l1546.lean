import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l1546_154698

theorem simplify_expression :
  -2^2005 + (-2)^2006 + 3^2007 - 2^2008 = -7 * 2^2005 + 3^2007 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1546_154698


namespace NUMINAMATH_CALUDE_share_ratio_proof_l1546_154658

theorem share_ratio_proof (total : ℝ) (c_share : ℝ) (f : ℝ) :
  total = 700 →
  c_share = 400 →
  0 < f →
  f ≤ 1 →
  total = f^2 * c_share + f * c_share + c_share →
  (f^2 * c_share) / (f * c_share) = 1 / 2 ∧
  (f * c_share) / c_share = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_share_ratio_proof_l1546_154658


namespace NUMINAMATH_CALUDE_sphere_visible_radius_l1546_154648

-- Define the problem parameters
def shadow_length : ℝ := 15
def stick_height : ℝ := 2
def stick_shadow : ℝ := 3

-- Define the theorem
theorem sphere_visible_radius :
  ∀ (r : ℝ), 
    (r / shadow_length = stick_height / stick_shadow) →
    r = 10 := by
  sorry

end NUMINAMATH_CALUDE_sphere_visible_radius_l1546_154648


namespace NUMINAMATH_CALUDE_complex_square_root_l1546_154651

theorem complex_square_root (z : ℂ) : 
  z^2 = -100 - 48*I ↔ z = 2 - 12*I ∨ z = -2 + 12*I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_root_l1546_154651


namespace NUMINAMATH_CALUDE_train_length_l1546_154655

/-- The length of a train given its speed, the speed of a man walking in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_speed = 60.994720422366214 →
  man_speed = 5 →
  passing_time = 6 →
  ∃ (train_length : ℝ), 
    109.98 < train_length ∧ train_length < 110 :=
by sorry


end NUMINAMATH_CALUDE_train_length_l1546_154655


namespace NUMINAMATH_CALUDE_david_money_left_is_275_l1546_154622

/-- Represents the amount of money David has left at the end of his trip -/
def david_money_left (initial_amount accommodations food_euros food_exchange_rate souvenirs_yen souvenirs_exchange_rate loan : ℚ) : ℚ :=
  let total_spent := accommodations + (food_euros * food_exchange_rate) + (souvenirs_yen * souvenirs_exchange_rate)
  initial_amount - total_spent - 500

/-- Theorem stating that David has $275 left at the end of his trip -/
theorem david_money_left_is_275 :
  david_money_left 1500 400 300 1.1 5000 0.009 200 = 275 := by
  sorry

end NUMINAMATH_CALUDE_david_money_left_is_275_l1546_154622


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l1546_154656

/-- The distance between the foci of a hyperbola given by the equation 9x^2 - 27x - 16y^2 - 32y = 72 -/
theorem hyperbola_foci_distance : 
  let equation := fun (x y : ℝ) => 9 * x^2 - 27 * x - 16 * y^2 - 32 * y - 72
  ∃ c : ℝ, c > 0 ∧ 
    (∀ x y : ℝ, equation x y = 0 → 
      ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
        ((x - 3/2)^2 / a^2) - ((y + 1)^2 / b^2) = 1 ∧
        c^2 = a^2 + b^2) ∧
    2 * c = Real.sqrt 41775 / 12 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l1546_154656


namespace NUMINAMATH_CALUDE_cannot_row_against_stream_l1546_154629

theorem cannot_row_against_stream (rate_still : ℝ) (speed_with_stream : ℝ) :
  rate_still = 1 →
  speed_with_stream = 6 →
  let stream_speed := speed_with_stream - rate_still
  stream_speed > rate_still →
  ¬∃ (speed_against_stream : ℝ), speed_against_stream > 0 ∧ speed_against_stream = rate_still - stream_speed :=
by
  sorry

end NUMINAMATH_CALUDE_cannot_row_against_stream_l1546_154629


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_is_attained_l1546_154600

theorem min_value_of_function (x : ℝ) (h : x > 0) : 3 * x + 12 / x^2 ≥ 9 := by
  sorry

theorem min_value_is_attained : ∃ x : ℝ, x > 0 ∧ 3 * x + 12 / x^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_is_attained_l1546_154600


namespace NUMINAMATH_CALUDE_apple_heavier_than_kiwi_l1546_154665

-- Define a type for fruits
inductive Fruit
  | Apple
  | Banana
  | Kiwi

-- Define a weight relation between fruits
def heavier_than (a b : Fruit) : Prop := sorry

-- State the theorem
theorem apple_heavier_than_kiwi 
  (h1 : heavier_than Fruit.Apple Fruit.Banana) 
  (h2 : heavier_than Fruit.Banana Fruit.Kiwi) : 
  heavier_than Fruit.Apple Fruit.Kiwi := by
  sorry

end NUMINAMATH_CALUDE_apple_heavier_than_kiwi_l1546_154665


namespace NUMINAMATH_CALUDE_largest_three_digit_square_base_7_l1546_154670

/-- The largest integer whose square has exactly 3 digits in base 7 -/
def M : ℕ := 18

/-- A number has exactly 3 digits in base 7 if and only if it's in the range [7^2, 7^3) -/
def has_three_digits_base_7 (n : ℕ) : Prop :=
  7^2 ≤ n ∧ n < 7^3

theorem largest_three_digit_square_base_7 :
  M = 18 ∧ has_three_digits_base_7 (M^2) ∧ ∀ k : ℕ, k > M → ¬has_three_digits_base_7 (k^2) := by
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_square_base_7_l1546_154670


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l1546_154636

theorem initial_markup_percentage
  (initial_price : ℝ)
  (price_increase : ℝ)
  (h1 : initial_price = 34)
  (h2 : price_increase = 6)
  (h3 : initial_price + price_increase = 2 * (initial_price - (initial_price + price_increase) / 2)) :
  (initial_price - (initial_price + price_increase) / 2) / ((initial_price + price_increase) / 2) = 0.7 :=
by sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l1546_154636


namespace NUMINAMATH_CALUDE_unique_students_count_unique_students_is_34_l1546_154626

/-- The number of unique students in a mathematics contest at Gauss High School --/
theorem unique_students_count : ℕ :=
  let euclid_class : ℕ := 12
  let raman_class : ℕ := 10
  let pythagoras_class : ℕ := 15
  let euclid_raman_overlap : ℕ := 3
  euclid_class + raman_class + pythagoras_class - euclid_raman_overlap

/-- Proof that the number of unique students is 34 --/
theorem unique_students_is_34 : unique_students_count = 34 := by
  sorry

end NUMINAMATH_CALUDE_unique_students_count_unique_students_is_34_l1546_154626


namespace NUMINAMATH_CALUDE_student_average_greater_than_actual_average_l1546_154603

theorem student_average_greater_than_actual_average (x y z : ℝ) (h : x < y ∧ y < z) :
  (x + y + 2 * z) / 4 > (x + y + z) / 3 := by
  sorry

end NUMINAMATH_CALUDE_student_average_greater_than_actual_average_l1546_154603


namespace NUMINAMATH_CALUDE_expected_BBR_sequences_l1546_154659

/-- Represents a standard deck of 52 cards -/
structure Deck :=
  (cards : Fin 52)

/-- Represents a sequence of three cards -/
structure ThreeCardSequence :=
  (first : Deck)
  (second : Deck)
  (third : Deck)

/-- Checks if a card is black -/
def is_black (card : Deck) : Prop :=
  sorry

/-- Checks if a card is red -/
def is_red (card : Deck) : Prop :=
  sorry

/-- Checks if a sequence is BBR (two black cards followed by a red card) -/
def is_BBR (seq : ThreeCardSequence) : Prop :=
  is_black seq.first ∧ is_black seq.second ∧ is_red seq.third

/-- The probability of a specific BBR sequence -/
def BBR_probability : ℚ :=
  13 / 51

/-- The number of possible starting positions for a BBR sequence -/
def num_starting_positions : ℕ :=
  26

/-- The expected number of BBR sequences in a standard 52-card deck dealt in a circle -/
theorem expected_BBR_sequences :
  (num_starting_positions : ℚ) * BBR_probability = 338 / 51 :=
sorry

end NUMINAMATH_CALUDE_expected_BBR_sequences_l1546_154659


namespace NUMINAMATH_CALUDE_price_reduction_for_1200_profit_no_solution_for_1600_profit_l1546_154644

-- Define the initial conditions
def initial_sales : ℕ := 30
def initial_profit : ℕ := 40
def sales_increase_rate : ℕ := 2

-- Define the profit function
def daily_profit (price_reduction : ℝ) : ℝ :=
  (initial_profit - price_reduction) * (initial_sales + sales_increase_rate * price_reduction)

-- Theorem for part 1
theorem price_reduction_for_1200_profit :
  ∃ (x : ℝ), x > 0 ∧ daily_profit x = 1200 ∧ 
  (∀ (y : ℝ), y > 0 ∧ y ≠ x → daily_profit y ≠ 1200) :=
sorry

-- Theorem for part 2
theorem no_solution_for_1600_profit :
  ¬∃ (x : ℝ), daily_profit x = 1600 :=
sorry

end NUMINAMATH_CALUDE_price_reduction_for_1200_profit_no_solution_for_1600_profit_l1546_154644


namespace NUMINAMATH_CALUDE_juniors_average_score_l1546_154673

theorem juniors_average_score 
  (total_students : ℝ) 
  (junior_ratio : ℝ) 
  (senior_ratio : ℝ) 
  (class_average : ℝ) 
  (senior_average : ℝ) 
  (h1 : junior_ratio = 0.2)
  (h2 : senior_ratio = 0.8)
  (h3 : junior_ratio + senior_ratio = 1)
  (h4 : class_average = 86)
  (h5 : senior_average = 85) :
  (class_average * total_students - senior_average * (senior_ratio * total_students)) / (junior_ratio * total_students) = 90 :=
by sorry

end NUMINAMATH_CALUDE_juniors_average_score_l1546_154673


namespace NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l1546_154652

/-- Given that the line x + y = c is a perpendicular bisector of the line segment from (2,5) to (8,11), prove that c = 13 -/
theorem perpendicular_bisector_c_value :
  ∀ c : ℝ,
  (∀ x y : ℝ, x + y = c ↔ (x - 5)^2 + (y - 8)^2 = (5 - 2)^2 + (8 - 5)^2) →
  c = 13 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l1546_154652


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l1546_154624

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 30 * x + c = 0) → 
  (a + c = 35) →
  (a < c) →
  (a = (35 - 5 * Real.sqrt 13) / 2 ∧ c = (35 + 5 * Real.sqrt 13) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l1546_154624


namespace NUMINAMATH_CALUDE_black_area_after_transformations_l1546_154645

/-- The fraction of black area remaining after one transformation -/
def remaining_fraction : ℚ := 2 / 3

/-- The number of transformations -/
def num_transformations : ℕ := 6

/-- The theorem stating the fraction of black area remaining after six transformations -/
theorem black_area_after_transformations :
  remaining_fraction ^ num_transformations = 64 / 729 := by
  sorry

end NUMINAMATH_CALUDE_black_area_after_transformations_l1546_154645


namespace NUMINAMATH_CALUDE_line_parallel_value_l1546_154699

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + (a^2 - 1) = 0

-- Define the parallel condition for two lines
def parallel (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) : Prop :=
  (A₁ * B₂ - A₂ * B₁ = 0) ∧ ((A₁ * C₂ - A₂ * C₁ ≠ 0) ∨ (B₁ * C₂ - B₂ * C₁ ≠ 0))

-- Define the coincident condition for two lines
def coincident (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) : Prop :=
  (A₁ * B₂ - A₂ * B₁ = 0) ∧ (A₁ * C₂ - A₂ * C₁ = 0) ∧ (B₁ * C₂ - B₂ * C₁ = 0)

-- Theorem statement
theorem line_parallel_value (a : ℝ) : 
  (parallel a 2 6 1 (a-1) (a^2-1)) ∧ 
  ¬(coincident a 2 6 1 (a-1) (a^2-1)) → 
  a = -1 := by sorry

end NUMINAMATH_CALUDE_line_parallel_value_l1546_154699


namespace NUMINAMATH_CALUDE_eight_friends_receive_necklace_l1546_154619

/-- The number of friends receiving a candy necklace -/
def friends_receiving_necklace (pieces_per_necklace : ℕ) (pieces_per_block : ℕ) (blocks_used : ℕ) : ℕ :=
  (blocks_used * pieces_per_block) / pieces_per_necklace - 1

/-- Theorem: Given the conditions, prove that 8 friends receive a candy necklace -/
theorem eight_friends_receive_necklace :
  friends_receiving_necklace 10 30 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_friends_receive_necklace_l1546_154619


namespace NUMINAMATH_CALUDE_waiter_tip_problem_l1546_154614

/-- Calculates the tip amount per customer given the total customers, non-tipping customers, and total tips. -/
def tip_per_customer (total_customers : ℕ) (non_tipping_customers : ℕ) (total_tips : ℚ) : ℚ :=
  total_tips / (total_customers - non_tipping_customers)

/-- Proves that given 10 total customers, 5 non-tipping customers, and $15 total tips, 
    the amount each tipping customer gave is $3. -/
theorem waiter_tip_problem :
  tip_per_customer 10 5 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tip_problem_l1546_154614


namespace NUMINAMATH_CALUDE_women_decrease_l1546_154664

theorem women_decrease (initial_men : ℕ) (initial_women : ℕ) : 
  (initial_men : ℚ) / initial_women = 4 / 5 →
  initial_men + 2 = 14 →
  initial_women - 3 = 24 →
  initial_women - 24 = 3 := by
sorry

end NUMINAMATH_CALUDE_women_decrease_l1546_154664


namespace NUMINAMATH_CALUDE_square_fraction_integers_l1546_154606

theorem square_fraction_integers (n : ℕ) : n > 1 ∧ ∃ (k : ℕ), k > 0 ∧ (n^2 + 7*n + 136) / (n - 1) = k^2 ↔ n = 5 ∨ n = 37 := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_integers_l1546_154606


namespace NUMINAMATH_CALUDE_vacation_cost_division_l1546_154669

theorem vacation_cost_division (total_cost : ℝ) (cost_difference : ℝ) : 
  total_cost = 375 ∧ 
  cost_difference = 50 ∧
  (total_cost / 5 = total_cost / (total_cost / (total_cost / 5 - cost_difference)) - cost_difference) →
  total_cost / (total_cost / 5 - cost_difference) = 15 :=
by sorry

end NUMINAMATH_CALUDE_vacation_cost_division_l1546_154669


namespace NUMINAMATH_CALUDE_hades_can_prevent_sisyphus_l1546_154678

/-- Represents the state of the mountain with stones -/
structure MountainState where
  steps : Nat
  stones : Nat
  stone_positions : Finset Nat

/-- Defines the game rules and initial state -/
def initial_state : MountainState :=
  { steps := 1001
  , stones := 500
  , stone_positions := Finset.range 500 }

/-- Sisyphus's move: Lifts a stone to the nearest free step above -/
def sisyphus_move (state : MountainState) : MountainState :=
  sorry

/-- Hades's move: Lowers a stone to the nearest free step below -/
def hades_move (state : MountainState) : MountainState :=
  sorry

/-- Represents a full round of the game (Sisyphus's move followed by Hades's move) -/
def game_round (state : MountainState) : MountainState :=
  hades_move (sisyphus_move state)

/-- Theorem stating that Hades can prevent Sisyphus from reaching the top step -/
theorem hades_can_prevent_sisyphus (state : MountainState := initial_state) :
  ∀ n : Nat, (game_round^[n] state).stone_positions.max < state.steps :=
  sorry

end NUMINAMATH_CALUDE_hades_can_prevent_sisyphus_l1546_154678


namespace NUMINAMATH_CALUDE_roots_property_l1546_154621

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := 3 * x^2 + 5 * x - 7 = 0

-- Define the theorem
theorem roots_property (p q : ℝ) (hp : quadratic_eq p) (hq : quadratic_eq q) :
  (p - 2) * (q - 2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_roots_property_l1546_154621


namespace NUMINAMATH_CALUDE_complex_modulus_l1546_154694

theorem complex_modulus (z : ℂ) (h : z * Complex.I = 3 - 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1546_154694


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l1546_154653

theorem absolute_value_equation_product (x₁ x₂ : ℝ) : 
  (|3 * x₁ - 5| = 40) ∧ (|3 * x₂ - 5| = 40) ∧ (x₁ ≠ x₂) →
  x₁ * x₂ = -175 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l1546_154653


namespace NUMINAMATH_CALUDE_sum_of_powers_l1546_154643

theorem sum_of_powers (x : ℝ) (h : x + 1/x = 4) : x^6 + 1/x^6 = 2702 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1546_154643


namespace NUMINAMATH_CALUDE_f_is_linear_l1546_154604

/-- Defines a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function representing the equation 3y + 1 = 6 -/
def f (y : ℝ) : ℝ := 3 * y + 1

/-- Theorem stating that f is a linear equation -/
theorem f_is_linear : is_linear_equation f := by
  sorry

#check f_is_linear

end NUMINAMATH_CALUDE_f_is_linear_l1546_154604


namespace NUMINAMATH_CALUDE_exists_point_product_nonnegative_l1546_154693

theorem exists_point_product_nonnegative 
  (f : ℝ → ℝ) 
  (hf : ContDiff ℝ 3 f) : 
  ∃ a : ℝ, f a * (deriv f a) * (deriv^[2] f a) * (deriv^[3] f a) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_point_product_nonnegative_l1546_154693


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1546_154646

theorem trigonometric_equation_solution :
  ∀ x : ℝ, 5.22 * (Real.sin x)^2 - 2 * Real.sin x * Real.cos x = 3 * (Real.cos x)^2 ↔
  (∃ k : ℤ, x = Real.arctan 0.973 + k * Real.pi) ∨
  (∃ k : ℤ, x = Real.arctan (-0.59) + k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1546_154646


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1546_154620

theorem perpendicular_vectors_x_value 
  (x : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (x, 3)) 
  (hb : b = (2, x - 5)) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : 
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1546_154620


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1546_154683

/-- A quadratic function with axis of symmetry at x = 2 -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_inequality (b c : ℝ) :
  (∀ x, f b c (2 - x) = f b c (2 + x)) →  -- axis of symmetry at x = 2
  (∀ x₁ x₂, x₁ < x₂ → f b c x₁ > f b c x₂ → f b c x₂ > f b c (2*x₂ - x₁)) →  -- opens upwards
  f b c 2 < f b c 1 ∧ f b c 1 < f b c 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1546_154683


namespace NUMINAMATH_CALUDE_magazine_purchase_methods_l1546_154611

theorem magazine_purchase_methods (n : ℕ) (m : ℕ) (total : ℕ) : 
  n + m = 11 → 
  n = 8 → 
  m = 3 → 
  total = 10 →
  (Nat.choose n 5 + Nat.choose n 4 * Nat.choose m 2) = 266 := by
  sorry

end NUMINAMATH_CALUDE_magazine_purchase_methods_l1546_154611


namespace NUMINAMATH_CALUDE_exist_positive_integers_satisfying_equation_l1546_154657

theorem exist_positive_integers_satisfying_equation : 
  ∃ (x y z : ℕ+), x^2006 + y^2006 = z^2007 := by
  sorry

end NUMINAMATH_CALUDE_exist_positive_integers_satisfying_equation_l1546_154657


namespace NUMINAMATH_CALUDE_james_excess_calories_james_specific_excess_calories_l1546_154696

/-- Calculates the excess calories eaten by James after eating Cheezits and going for a run. -/
theorem james_excess_calories (bags : Nat) (ounces_per_bag : Nat) (calories_per_ounce : Nat) 
  (run_duration : Nat) (calories_burned_per_minute : Nat) : Nat :=
  let total_calories_consumed := bags * ounces_per_bag * calories_per_ounce
  let total_calories_burned := run_duration * calories_burned_per_minute
  total_calories_consumed - total_calories_burned

/-- Proves that James ate 420 excess calories given the specific conditions. -/
theorem james_specific_excess_calories :
  james_excess_calories 3 2 150 40 12 = 420 := by
  sorry

end NUMINAMATH_CALUDE_james_excess_calories_james_specific_excess_calories_l1546_154696


namespace NUMINAMATH_CALUDE_leisurely_morning_time_l1546_154631

/-- Represents the time taken for each part of Aiden's morning routine -/
structure MorningRoutine where
  prep : ℝ  -- Preparation time
  bus : ℝ   -- Bus ride time
  walk : ℝ  -- Walking time

/-- Calculates the total time for a given morning routine -/
def totalTime (r : MorningRoutine) : ℝ := r.prep + r.bus + r.walk

/-- Represents the conditions given in the problem -/
axiom typical_morning : ∃ r : MorningRoutine, totalTime r = 120

axiom rushed_morning : ∃ r : MorningRoutine, 
  0.5 * r.prep + 1.25 * r.bus + 0.5 * r.walk = 96

/-- Theorem stating the time taken on the leisurely morning -/
theorem leisurely_morning_time : 
  ∀ r : MorningRoutine, 
  totalTime r = 120 → 
  0.5 * r.prep + 1.25 * r.bus + 0.5 * r.walk = 96 → 
  1.25 * r.prep + 0.75 * r.bus + 1.25 * r.walk = 126 := by
  sorry

end NUMINAMATH_CALUDE_leisurely_morning_time_l1546_154631


namespace NUMINAMATH_CALUDE_grape_pickers_l1546_154625

/-- Given information about grape pickers and their work rate, calculate the number of pickers. -/
theorem grape_pickers (total_drums : ℕ) (total_days : ℕ) (drums_per_day : ℕ) :
  total_drums = 90 →
  total_days = 6 →
  drums_per_day = 15 →
  (total_drums / total_days : ℚ) = drums_per_day →
  drums_per_day / drums_per_day = 1 :=
by sorry

end NUMINAMATH_CALUDE_grape_pickers_l1546_154625


namespace NUMINAMATH_CALUDE_f_of_a_plus_one_l1546_154676

/-- The function f(x) = x^2 + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: For the function f(x) = x^2 + 1, f(a+1) = a^2 + 2a + 2 for any real number a -/
theorem f_of_a_plus_one (a : ℝ) : f (a + 1) = a^2 + 2*a + 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_a_plus_one_l1546_154676


namespace NUMINAMATH_CALUDE_investors_in_both_l1546_154672

theorem investors_in_both (total : ℕ) (equities : ℕ) (both : ℕ)
  (h_total : total = 100)
  (h_equities : equities = 80)
  (h_both : both = 40)
  (h_invest : ∀ i, i ∈ Finset.range total → 
    (i ∈ Finset.range equities ∨ i ∈ Finset.range (total - equities + both)))
  : both = 40 := by
  sorry

end NUMINAMATH_CALUDE_investors_in_both_l1546_154672


namespace NUMINAMATH_CALUDE_nick_speed_l1546_154681

/-- Given the speeds of Alan, Maria, and Nick in relation to each other,
    prove that Nick's speed is 6 miles per hour. -/
theorem nick_speed (alan_speed : ℝ) (maria_speed : ℝ) (nick_speed : ℝ)
    (h1 : alan_speed = 6)
    (h2 : maria_speed = 3/4 * alan_speed)
    (h3 : nick_speed = 4/3 * maria_speed) :
    nick_speed = 6 := by
  sorry

end NUMINAMATH_CALUDE_nick_speed_l1546_154681


namespace NUMINAMATH_CALUDE_function_shift_l1546_154667

noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (1/2 * x + φ)

theorem function_shift (φ : ℝ) (h1 : |φ| < Real.pi/2) 
  (h2 : ∀ x, f x φ = f (Real.pi/3 - x) φ) :
  ∀ x, f (x + Real.pi/3) φ = Real.cos (1/2 * x) := by
sorry

end NUMINAMATH_CALUDE_function_shift_l1546_154667


namespace NUMINAMATH_CALUDE_correct_calculation_l1546_154641

theorem correct_calculation (x : ℝ) : 2 * (3 * x + 14) = 946 → 2 * (x / 3 + 14) = 130 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1546_154641


namespace NUMINAMATH_CALUDE_son_father_distance_l1546_154685

/-- 
Given a lamp post, a father, and his son standing on the same straight line,
with their shadows' heads incident at the same point, prove that the distance
between the son and his father is 4.9 meters.
-/
theorem son_father_distance 
  (lamp_height : ℝ) 
  (father_height : ℝ) 
  (son_height : ℝ) 
  (father_lamp_distance : ℝ) 
  (h_lamp : lamp_height = 6)
  (h_father : father_height = 1.8)
  (h_son : son_height = 0.9)
  (h_father_lamp : father_lamp_distance = 2.1)
  (h_shadows : ∀ x : ℝ, father_height / father_lamp_distance = lamp_height / (father_lamp_distance + x) → 
                        son_height / x = father_height / (father_lamp_distance + x)) :
  ∃ x : ℝ, x = 4.9 ∧ 
    father_height / father_lamp_distance = lamp_height / (father_lamp_distance + x) ∧
    son_height / x = father_height / (father_lamp_distance + x) := by
  sorry


end NUMINAMATH_CALUDE_son_father_distance_l1546_154685


namespace NUMINAMATH_CALUDE_determinant_of_2x2_matrix_l1546_154654

theorem determinant_of_2x2_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![9, 5; -3, 4]
  Matrix.det A = 51 := by
sorry

end NUMINAMATH_CALUDE_determinant_of_2x2_matrix_l1546_154654


namespace NUMINAMATH_CALUDE_negation_of_p_l1546_154605

def p (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0

theorem negation_of_p (f : ℝ → ℝ) :
  ¬(p f) ↔ ∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_p_l1546_154605


namespace NUMINAMATH_CALUDE_ali_final_money_l1546_154647

-- Define the initial state of Ali's wallet
def initial_wallet : ℚ :=
  7 * 5 + 1 * 10 + 3 * 20 + 1 * 50 + 8 * 1 + 10 * (1/4)

-- Morning transaction
def morning_transaction (wallet : ℚ) : ℚ :=
  wallet - (50 + 20 + 5) + (3 + 8 * (1/4) + 10 * (1/10))

-- Coffee shop transaction
def coffee_transaction (wallet : ℚ) : ℚ :=
  wallet - (15/4)

-- Afternoon transaction
def afternoon_transaction (wallet : ℚ) : ℚ :=
  wallet + 42

-- Evening transaction
def evening_transaction (wallet : ℚ) : ℚ :=
  wallet - (45/4)

-- Final wallet state after all transactions
def final_wallet : ℚ :=
  evening_transaction (afternoon_transaction (coffee_transaction (morning_transaction initial_wallet)))

-- Theorem statement
theorem ali_final_money :
  final_wallet = 247/2 := by sorry

end NUMINAMATH_CALUDE_ali_final_money_l1546_154647


namespace NUMINAMATH_CALUDE_intersection_line_slope_l1546_154660

-- Define the circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y + 12 = 0

-- Define the intersection points
def intersection_points (C D : ℝ × ℝ) : Prop :=
  circle1 C.1 C.2 ∧ circle1 D.1 D.2 ∧ circle2 C.1 C.2 ∧ circle2 D.1 D.2

-- Theorem statement
theorem intersection_line_slope (C D : ℝ × ℝ) 
  (h : intersection_points C D) : 
  (D.2 - C.2) / (D.1 - C.1) = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l1546_154660


namespace NUMINAMATH_CALUDE_quadratic_equation_standard_form_quadratic_coefficients_l1546_154639

theorem quadratic_equation_standard_form :
  ∀ x : ℝ, (x + 5) * (3 + x) = 2 * x^2 ↔ x^2 - 8 * x - 15 = 0 :=
by sorry

theorem quadratic_coefficients (a b c : ℝ) :
  (∀ x : ℝ, (x + 5) * (3 + x) = 2 * x^2 ↔ a * x^2 + b * x + c = 0) →
  a = 1 ∧ b = -8 ∧ c = -15 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_standard_form_quadratic_coefficients_l1546_154639


namespace NUMINAMATH_CALUDE_fraction_simplification_l1546_154661

theorem fraction_simplification :
  (5 : ℝ) / (2 * Real.sqrt 27 + 3 * Real.sqrt 12 + Real.sqrt 108) = (5 * Real.sqrt 3) / 54 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1546_154661


namespace NUMINAMATH_CALUDE_no_solution_rebus_l1546_154680

theorem no_solution_rebus :
  ¬ ∃ (K U S Y : ℕ),
    K ≠ U ∧ K ≠ S ∧ K ≠ Y ∧ U ≠ S ∧ U ≠ Y ∧ S ≠ Y ∧
    K < 10 ∧ U < 10 ∧ S < 10 ∧ Y < 10 ∧
    1000 ≤ (1000 * K + 100 * U + 10 * S + Y) ∧
    (1000 * K + 100 * U + 10 * S + Y) < 10000 ∧
    1000 ≤ (1000 * U + 100 * K + 10 * S + Y) ∧
    (1000 * U + 100 * K + 10 * S + Y) < 10000 ∧
    10000 ≤ (10000 * U + 1000 * K + 100 * S + 10 * U + S) ∧
    (10000 * U + 1000 * K + 100 * S + 10 * U + S) < 100000 ∧
    (1000 * K + 100 * U + 10 * S + Y) + (1000 * U + 100 * K + 10 * S + Y) =
    (10000 * U + 1000 * K + 100 * S + 10 * U + S) :=
by
  sorry

end NUMINAMATH_CALUDE_no_solution_rebus_l1546_154680


namespace NUMINAMATH_CALUDE_group_photo_arrangements_l1546_154633

def number_of_leaders : ℕ := 21
def front_row : ℕ := 11
def back_row : ℕ := 10

def arrangements (n : ℕ) : ℕ := n.factorial

theorem group_photo_arrangements :
  let remaining_leaders := number_of_leaders - 3
  let us_russia_arrangements := arrangements 2
  let other_arrangements := arrangements remaining_leaders
  us_russia_arrangements * other_arrangements = arrangements 2 * arrangements 18 :=
by sorry

end NUMINAMATH_CALUDE_group_photo_arrangements_l1546_154633


namespace NUMINAMATH_CALUDE_arithmetic_operations_l1546_154682

theorem arithmetic_operations :
  ((-3) + (-1) = -4) ∧
  (0 - 11 = -11) ∧
  (97 - (-3) = 100) ∧
  ((-7) * 5 = -35) ∧
  ((-8) / (-1/4) = 32) ∧
  ((-2/3)^3 = -8/27) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l1546_154682


namespace NUMINAMATH_CALUDE_F_of_2_f_of_3_equals_341_l1546_154608

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 2
def F (a b : ℝ) : ℝ := b^3 - a

-- Theorem statement
theorem F_of_2_f_of_3_equals_341 : F 2 (f 3) = 341 := by
  sorry

end NUMINAMATH_CALUDE_F_of_2_f_of_3_equals_341_l1546_154608


namespace NUMINAMATH_CALUDE_simplify_and_sum_l1546_154684

theorem simplify_and_sum (d : ℝ) (a b c : ℝ) (h : d ≠ 0) :
  (15 * d + 18 + 12 * d^2) + (5 * d + 2) = a * d + b + c * d^2 →
  a + b + c = 52 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_sum_l1546_154684


namespace NUMINAMATH_CALUDE_resultant_of_quadratics_l1546_154634

/-- The resultant of two quadratic polynomials -/
def resultant (a b p q : ℝ) : ℝ :=
  (p - a) * (p * b - a * q) + (q - b)^2

/-- Roots of a quadratic polynomial -/
structure QuadraticRoots (a b : ℝ) where
  x₁ : ℝ
  x₂ : ℝ
  sum : x₁ + x₂ = -a
  product : x₁ * x₂ = b

theorem resultant_of_quadratics (a b p q : ℝ) 
  (f_roots : QuadraticRoots a b) (g_roots : QuadraticRoots p q) :
  (f_roots.x₁ - g_roots.x₁) * (f_roots.x₁ - g_roots.x₂) * 
  (f_roots.x₂ - g_roots.x₁) * (f_roots.x₂ - g_roots.x₂) = 
  resultant a b p q := by
  sorry

end NUMINAMATH_CALUDE_resultant_of_quadratics_l1546_154634


namespace NUMINAMATH_CALUDE_π_approximation_relation_l1546_154618

/-- Approximate value of π obtained with an n-sided inscribed regular polygon -/
noncomputable def π_n (n : ℕ) : ℝ := sorry

/-- Theorem stating the relationship between π_2n and π_n -/
theorem π_approximation_relation (n : ℕ) :
  π_n (2 * n) = π_n n / Real.cos (π / n) := by sorry

end NUMINAMATH_CALUDE_π_approximation_relation_l1546_154618


namespace NUMINAMATH_CALUDE_alice_weight_l1546_154623

theorem alice_weight (alice carol : ℝ) 
  (h1 : alice + carol = 200)
  (h2 : alice - carol = (1 / 3) * alice) : 
  alice = 120 := by
sorry

end NUMINAMATH_CALUDE_alice_weight_l1546_154623


namespace NUMINAMATH_CALUDE_train_speed_l1546_154609

/-- The speed of a train passing a platform -/
theorem train_speed (train_length platform_length time_to_pass : ℝ) 
  (h1 : train_length = 140)
  (h2 : platform_length = 260)
  (h3 : time_to_pass = 23.998080153587715) : 
  ∃ (speed : ℝ), abs (speed - 60.0048) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1546_154609


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1546_154632

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1546_154632


namespace NUMINAMATH_CALUDE_problem_solution_l1546_154635

theorem problem_solution (X Y : ℝ) : 
  (18 / 100 * X = 54 / 100 * 1200) → 
  (X = 4 * Y) → 
  (X = 3600 ∧ Y = 900) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1546_154635


namespace NUMINAMATH_CALUDE_package_weight_l1546_154642

theorem package_weight (total_weight : ℝ) (first_butcher_packages : ℕ) (second_butcher_packages : ℕ) (third_butcher_packages : ℕ) 
  (h1 : total_weight = 100)
  (h2 : first_butcher_packages = 10)
  (h3 : second_butcher_packages = 7)
  (h4 : third_butcher_packages = 8) :
  ∃ (package_weight : ℝ), 
    package_weight * (first_butcher_packages + second_butcher_packages + third_butcher_packages) = total_weight ∧ 
    package_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_package_weight_l1546_154642


namespace NUMINAMATH_CALUDE_alice_bob_sum_l1546_154612

/-- A number is prime if it's greater than 1 and has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- A number is a perfect square if it's the product of an integer with itself. -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem alice_bob_sum : 
  ∀ (A B : ℕ),
  (A ≠ 1) →  -- Alice's number is not the smallest
  (B = 2) →  -- Bob's number is the smallest prime
  (isPrime B) →
  (isPerfectSquare (100 * B + A)) →
  (1 ≤ A ∧ A ≤ 40) →  -- Alice's number is between 1 and 40
  (1 ≤ B ∧ B ≤ 40) →  -- Bob's number is between 1 and 40
  (A + B = 27) := by sorry

end NUMINAMATH_CALUDE_alice_bob_sum_l1546_154612


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_sin_equality_l1546_154677

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_sin_equality : 
  (¬ ∃ x : ℝ, x = Real.sin x) ↔ (∀ x : ℝ, x ≠ Real.sin x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_sin_equality_l1546_154677


namespace NUMINAMATH_CALUDE_sarahs_sweaters_sarahs_sweaters_proof_l1546_154692

theorem sarahs_sweaters (machine_capacity : ℕ) (num_shirts : ℕ) (num_loads : ℕ) : ℕ :=
  let total_pieces := machine_capacity * num_loads
  let num_sweaters := total_pieces - num_shirts
  num_sweaters

theorem sarahs_sweaters_proof 
  (h1 : sarahs_sweaters 5 43 9 = 2) : sarahs_sweaters 5 43 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_sweaters_sarahs_sweaters_proof_l1546_154692


namespace NUMINAMATH_CALUDE_N_swaps_rows_l1546_154668

/-- The matrix that swaps rows of a 2x2 matrix -/
def N : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 1, 0]

/-- Theorem: N swaps the rows of any 2x2 matrix -/
theorem N_swaps_rows (a b c d : ℝ) :
  N • !![a, b; c, d] = !![c, d; a, b] := by
  sorry

end NUMINAMATH_CALUDE_N_swaps_rows_l1546_154668


namespace NUMINAMATH_CALUDE_cost_difference_l1546_154602

-- Define the parameters
def batches : ℕ := 4
def ounces_per_batch : ℕ := 12
def blueberry_carton_size : ℕ := 6
def raspberry_carton_size : ℕ := 8
def blueberry_price : ℚ := 5
def raspberry_price : ℚ := 3

-- Define the total ounces needed
def total_ounces : ℕ := batches * ounces_per_batch

-- Define the number of cartons needed for each fruit
def blueberry_cartons : ℕ := (total_ounces + blueberry_carton_size - 1) / blueberry_carton_size
def raspberry_cartons : ℕ := (total_ounces + raspberry_carton_size - 1) / raspberry_carton_size

-- Define the total cost for each fruit
def blueberry_cost : ℚ := blueberry_price * blueberry_cartons
def raspberry_cost : ℚ := raspberry_price * raspberry_cartons

-- Theorem to prove
theorem cost_difference : blueberry_cost - raspberry_cost = 22 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_l1546_154602


namespace NUMINAMATH_CALUDE_inequality_proof_l1546_154689

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (x^4)/(y*(1-y^2)) + (y^4)/(z*(1-z^2)) + (z^4)/(x*(1-x^2)) ≥ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1546_154689


namespace NUMINAMATH_CALUDE_parallelogram_angle_measure_l1546_154617

theorem parallelogram_angle_measure (α β : ℝ) : 
  (α + β = π) →  -- Adjacent angles in a parallelogram sum to π
  (β = α + π/9) →  -- One angle exceeds the other by π/9
  (α = 4*π/9) :=  -- The smaller angle is 4π/9
by sorry

end NUMINAMATH_CALUDE_parallelogram_angle_measure_l1546_154617


namespace NUMINAMATH_CALUDE_rhombus_diagonals_bisect_l1546_154695

-- Define the property of diagonals bisecting each other
def diagonals_bisect (shape : Type) : Prop := sorry

-- Define the relationship between rhombus and parallelogram
def rhombus_is_parallelogram : Prop := sorry

-- Theorem statement
theorem rhombus_diagonals_bisect :
  diagonals_bisect Parallelogram →
  rhombus_is_parallelogram →
  diagonals_bisect Rhombus := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_bisect_l1546_154695


namespace NUMINAMATH_CALUDE_range_of_a_l1546_154663

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the property of being increasing on (-∞, 0]
def IsIncreasingOnNegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : IsEven f)
  (h2 : IsIncreasingOnNegative f)
  (h3 : f a ≥ f 2) :
  a ∈ Set.Icc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1546_154663


namespace NUMINAMATH_CALUDE_remainder_51_pow_2015_mod_13_l1546_154650

theorem remainder_51_pow_2015_mod_13 : 51^2015 % 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_51_pow_2015_mod_13_l1546_154650


namespace NUMINAMATH_CALUDE_tetrahedron_edges_lengths_l1546_154697

-- Define the tetrahedron and its circumscribed sphere
structure Tetrahedron :=
  (base_edge1 : ℝ)
  (base_edge2 : ℝ)
  (base_edge3 : ℝ)
  (inclined_edge : ℝ)
  (sphere_radius : ℝ)
  (volume : ℝ)

-- Define the conditions
def tetrahedron_conditions (t : Tetrahedron) : Prop :=
  t.base_edge1 = 2 * t.sphere_radius ∧
  t.base_edge2 / t.base_edge3 = 4 / 3 ∧
  t.volume = 40 ∧
  t.base_edge1^2 = t.base_edge2^2 + t.base_edge3^2 ∧
  t.inclined_edge^2 = t.sphere_radius^2 + (t.base_edge2 / 2)^2

-- Theorem statement
theorem tetrahedron_edges_lengths 
  (t : Tetrahedron) 
  (h : tetrahedron_conditions t) : 
  t.base_edge1 = 10 ∧ 
  t.base_edge2 = 8 ∧ 
  t.base_edge3 = 6 ∧ 
  t.inclined_edge = Real.sqrt 50 := 
sorry

end NUMINAMATH_CALUDE_tetrahedron_edges_lengths_l1546_154697


namespace NUMINAMATH_CALUDE_logarithm_identity_l1546_154628

theorem logarithm_identity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha_ne_one : a ≠ 1) (hb_ne_one : b ≠ 1) : 
  Real.log c / Real.log (a * b) = (Real.log c / Real.log a * Real.log c / Real.log b) / 
    (Real.log c / Real.log a + Real.log c / Real.log b) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_identity_l1546_154628


namespace NUMINAMATH_CALUDE_prime_sequence_l1546_154601

theorem prime_sequence (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) := by
sorry

end NUMINAMATH_CALUDE_prime_sequence_l1546_154601


namespace NUMINAMATH_CALUDE_planting_cost_l1546_154687

def flower_cost : ℕ := 9
def clay_pot_cost : ℕ := flower_cost + 20
def soil_cost : ℕ := flower_cost - 2

def total_cost : ℕ := flower_cost + clay_pot_cost + soil_cost

theorem planting_cost : total_cost = 45 := by
  sorry

end NUMINAMATH_CALUDE_planting_cost_l1546_154687


namespace NUMINAMATH_CALUDE_shanna_garden_harvest_l1546_154630

/-- Calculates the total number of vegetables harvested given the initial plant counts and deaths --/
def total_vegetables_harvested (tomato_plants eggplant_plants pepper_plants : ℕ) 
  (tomato_deaths pepper_deaths : ℕ) (vegetables_per_plant : ℕ) : ℕ :=
  let surviving_tomatoes := tomato_plants - tomato_deaths
  let surviving_peppers := pepper_plants - pepper_deaths
  let total_surviving_plants := surviving_tomatoes + surviving_peppers + eggplant_plants
  total_surviving_plants * vegetables_per_plant

/-- Proves that Shanna harvested 56 vegetables given the initial conditions --/
theorem shanna_garden_harvest : 
  total_vegetables_harvested 6 2 4 3 1 7 = 56 := by
  sorry

end NUMINAMATH_CALUDE_shanna_garden_harvest_l1546_154630


namespace NUMINAMATH_CALUDE_intersection_point_correct_l1546_154690

/-- The intersection point of two lines y = -2x and y = x -/
def intersection_point : ℝ × ℝ := (0, 0)

/-- Function representing y = -2x -/
def f (x : ℝ) : ℝ := -2 * x

/-- Function representing y = x -/
def g (x : ℝ) : ℝ := x

/-- Theorem stating that (0, 0) is the unique intersection point of y = -2x and y = x -/
theorem intersection_point_correct :
  (∃! p : ℝ × ℝ, f p.1 = p.2 ∧ g p.1 = p.2) ∧
  (∀ p : ℝ × ℝ, f p.1 = p.2 ∧ g p.1 = p.2 → p = intersection_point) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_correct_l1546_154690


namespace NUMINAMATH_CALUDE_average_speed_palindrome_odometer_l1546_154691

/-- A natural number is a palindrome if it reads the same forwards and backwards. -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The next palindrome after a given natural number. -/
def nextPalindrome (n : ℕ) : ℕ := sorry

theorem average_speed_palindrome_odometer :
  let initial_reading : ℕ := 12321
  let final_reading : ℕ := nextPalindrome initial_reading
  let time_elapsed : ℚ := 3
  let distance_traveled : ℕ := final_reading - initial_reading
  let average_speed : ℚ := distance_traveled / time_elapsed
  isPalindrome initial_reading ∧
  isPalindrome final_reading ∧
  final_reading > initial_reading →
  average_speed = 100 / 3 := by sorry

end NUMINAMATH_CALUDE_average_speed_palindrome_odometer_l1546_154691


namespace NUMINAMATH_CALUDE_chinese_team_gold_medal_probability_l1546_154649

/-- The probability of the Chinese team winning the gold medal in Women's Singles Table Tennis -/
theorem chinese_team_gold_medal_probability :
  let prob_A : ℚ := 3/7  -- Probability of Player A winning
  let prob_B : ℚ := 1/4  -- Probability of Player B winning
  -- Assuming the events are mutually exclusive
  prob_A + prob_B = 19/28 := by sorry

end NUMINAMATH_CALUDE_chinese_team_gold_medal_probability_l1546_154649


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1546_154637

theorem perfect_square_condition (n : ℕ+) : 
  (∃ m : ℕ, 2^n.val + 12^n.val + 2011^n.val = m^2) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1546_154637


namespace NUMINAMATH_CALUDE_zeros_of_h_l1546_154671

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

def g (a : ℝ) (x : ℝ) : ℝ := (a + 1) * x^2 - 1

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := 2 * f a x - g a x

theorem zeros_of_h (a : ℝ) (x₁ x₂ : ℝ) :
  (∃ (x₁ x₂ : ℝ), h a x₁ = 0 ∧ h a x₂ = 0 ∧ x₁ ≠ x₂) →
  -1 < a ∧ a < 0 ∧ x₁ + x₂ > 2 / (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_zeros_of_h_l1546_154671


namespace NUMINAMATH_CALUDE_percent_increase_l1546_154679

theorem percent_increase (P : ℝ) (Q : ℝ) (h : Q = P + (1/3) * P) :
  (Q - P) / P * 100 = 100/3 := by
  sorry

end NUMINAMATH_CALUDE_percent_increase_l1546_154679


namespace NUMINAMATH_CALUDE_shells_remaining_calculation_l1546_154686

/-- The number of shells Lino picked up in the morning -/
def shells_picked_up : ℝ := 324.0

/-- The number of shells Lino put back in the afternoon -/
def shells_put_back : ℝ := 292.00

/-- The number of shells Lino has in all -/
def shells_remaining : ℝ := shells_picked_up - shells_put_back

/-- Theorem stating that the number of shells Lino has in all
    is equal to the difference between shells picked up and shells put back -/
theorem shells_remaining_calculation :
  shells_remaining = 32.0 := by sorry

end NUMINAMATH_CALUDE_shells_remaining_calculation_l1546_154686


namespace NUMINAMATH_CALUDE_triangle_area_l1546_154674

theorem triangle_area (a b c : ℝ) (A : ℝ) (S : ℝ) : 
  b^2 - b*c - 2*c^2 = 0 →
  a = Real.sqrt 6 →
  Real.cos A = 7/8 →
  S = (Real.sqrt 15)/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1546_154674


namespace NUMINAMATH_CALUDE_deleted_pictures_l1546_154627

theorem deleted_pictures (zoo_pics museum_pics remaining_pics : ℕ) 
  (h1 : zoo_pics = 24)
  (h2 : museum_pics = 12)
  (h3 : remaining_pics = 22) :
  zoo_pics + museum_pics - remaining_pics = 14 := by
  sorry

end NUMINAMATH_CALUDE_deleted_pictures_l1546_154627


namespace NUMINAMATH_CALUDE_vector_relations_l1546_154610

def a : ℝ × ℝ := (2, -1)
def c : ℝ × ℝ := (-1, 2)

def b (m : ℝ) : ℝ × ℝ := (-1, m)

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_relations :
  (parallel (a.1 + (b (-1)).1, a.2 + (b (-1)).2) c) ∧
  (perpendicular (a.1 + (b (3/2)).1, a.2 + (b (3/2)).2) c) := by
  sorry

end NUMINAMATH_CALUDE_vector_relations_l1546_154610


namespace NUMINAMATH_CALUDE_nut_storage_impact_l1546_154607

/-- Represents the types of nuts found in Mason's car -/
inductive NutType
  | Almond
  | Walnut
  | Hazelnut

/-- Represents the squirrels and their nut-storing behavior -/
structure Squirrel where
  nutType : NutType
  count : Nat
  nutsPerDay : Nat
  days : Nat

/-- Calculates the total number of nuts stored by a group of squirrels -/
def totalNuts (s : Squirrel) : Nat :=
  s.count * s.nutsPerDay * s.days

/-- Calculates the weight of a single nut in grams -/
def nutWeight (n : NutType) : Rat :=
  match n with
  | NutType.Almond => 1/2
  | NutType.Walnut => 10
  | NutType.Hazelnut => 2

/-- Calculates the total weight of nuts stored by a group of squirrels -/
def totalWeight (s : Squirrel) : Rat :=
  (totalNuts s : Rat) * nutWeight s.nutType

/-- Calculates the efficiency reduction based on the total weight of nuts -/
def efficiencyReduction (totalWeight : Rat) : Rat :=
  min 100 (totalWeight / 100)

/-- The main theorem stating the total weight of nuts and efficiency reduction -/
theorem nut_storage_impact (almondSquirrels walnutSquirrels hazelnutSquirrels : Squirrel) 
    (h1 : almondSquirrels = ⟨NutType.Almond, 2, 30, 35⟩)
    (h2 : walnutSquirrels = ⟨NutType.Walnut, 3, 20, 40⟩)
    (h3 : hazelnutSquirrels = ⟨NutType.Hazelnut, 1, 10, 45⟩) :
    totalWeight almondSquirrels + totalWeight walnutSquirrels + totalWeight hazelnutSquirrels = 25950 ∧
    efficiencyReduction (totalWeight almondSquirrels + totalWeight walnutSquirrels + totalWeight hazelnutSquirrels) = 100 := by
  sorry


end NUMINAMATH_CALUDE_nut_storage_impact_l1546_154607


namespace NUMINAMATH_CALUDE_base_height_ratio_l1546_154615

/-- Represents a triangular field with specific properties -/
structure TriangularField where
  base : ℝ
  height : ℝ
  cultivation_cost : ℝ
  cost_per_hectare : ℝ
  base_multiple_of_height : ∃ k : ℝ, base = k * height
  total_cost : cultivation_cost = 333.18
  cost_rate : cost_per_hectare = 24.68
  base_value : base = 300
  height_value : height = 300

/-- Theorem stating that the ratio of base to height is 1:1 for the given triangular field -/
theorem base_height_ratio (field : TriangularField) : field.base / field.height = 1 := by
  sorry

#check base_height_ratio

end NUMINAMATH_CALUDE_base_height_ratio_l1546_154615


namespace NUMINAMATH_CALUDE_no_existence_of_complex_numbers_l1546_154662

theorem no_existence_of_complex_numbers : ¬∃ (a b c : ℂ) (h : ℕ), 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ 
  (∀ (k l m : ℤ), (abs k + abs l + abs m ≥ 1996) → 
    Complex.abs (1 + k • a + l • b + m • c) > 1 / h) := by
  sorry


end NUMINAMATH_CALUDE_no_existence_of_complex_numbers_l1546_154662


namespace NUMINAMATH_CALUDE_passes_through_point_l1546_154688

/-- A linear function that passes through the point (0, 3) -/
def linearFunction (m : ℝ) (x : ℝ) : ℝ := m * x + 3

/-- Theorem: The linear function passes through the point (0, 3) for any slope m -/
theorem passes_through_point (m : ℝ) : linearFunction m 0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_passes_through_point_l1546_154688


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l1546_154616

theorem sum_of_roots_cubic_equation :
  let f : ℝ → ℝ := λ x => 6 * x^3 - 3 * x^2 - 18 * x + 9
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ + r₂ + r₃ = 0.5 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l1546_154616


namespace NUMINAMATH_CALUDE_equation_solution_l1546_154613

theorem equation_solution : 
  let f (x : ℂ) := (4 * x^3 + 4 * x^2 + 3 * x + 2) / (x - 2)
  let g (x : ℂ) := 4 * x^2 + 5 * x + 4
  let sol₁ : ℂ := (-9 + Complex.I * Real.sqrt 79) / 8
  let sol₂ : ℂ := (-9 - Complex.I * Real.sqrt 79) / 8
  (∀ x : ℂ, x ≠ 2 → f x = g x) → (f sol₁ = g sol₁ ∧ f sol₂ = g sol₂) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1546_154613


namespace NUMINAMATH_CALUDE_kabadi_players_count_l1546_154666

/-- The number of people who play kabadi -/
def kabadi_players : ℕ := 15

/-- The number of people who play kho kho only -/
def kho_kho_only : ℕ := 15

/-- The number of people who play both games -/
def both_games : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := 25

/-- Theorem stating that the number of kabadi players is correct given the conditions -/
theorem kabadi_players_count : 
  kabadi_players = total_players - kho_kho_only + both_games :=
by sorry

end NUMINAMATH_CALUDE_kabadi_players_count_l1546_154666


namespace NUMINAMATH_CALUDE_probability_calculation_l1546_154675

/-- The probability of selecting one qualified and one unqualified product -/
def probability_one_qualified_one_unqualified : ℚ :=
  3 / 5

/-- The total number of products -/
def total_products : ℕ := 5

/-- The number of qualified products -/
def qualified_products : ℕ := 3

/-- The number of unqualified products -/
def unqualified_products : ℕ := 2

/-- The number of products selected for inspection -/
def selected_products : ℕ := 2

theorem probability_calculation :
  probability_one_qualified_one_unqualified = 
    (qualified_products.choose 1 * unqualified_products.choose 1 : ℚ) / 
    (total_products.choose selected_products) :=
by sorry

end NUMINAMATH_CALUDE_probability_calculation_l1546_154675


namespace NUMINAMATH_CALUDE_tree_distance_l1546_154640

theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 100) :
  let distance_between (i j : ℕ) := d * (j - i : ℝ) / 4
  distance_between 1 n = 175 := by
  sorry

end NUMINAMATH_CALUDE_tree_distance_l1546_154640


namespace NUMINAMATH_CALUDE_yadav_yearly_savings_yadav_savings_l1546_154638

/-- Mr. Yadav's monthly salary savings calculation --/
theorem yadav_yearly_savings (monthly_salary : ℝ) 
  (h1 : monthly_salary * 0.2 = 4038) : 
  monthly_salary * 0.2 * 12 = 48456 := by
  sorry

/-- Main theorem: Mr. Yadav's yearly savings --/
theorem yadav_savings : ∃ (monthly_salary : ℝ), 
  monthly_salary * 0.2 = 4038 ∧ 
  monthly_salary * 0.2 * 12 = 48456 := by
  sorry

end NUMINAMATH_CALUDE_yadav_yearly_savings_yadav_savings_l1546_154638
