import Mathlib

namespace NUMINAMATH_CALUDE_fraction_simplification_l2038_203831

theorem fraction_simplification : (1998 - 998) / 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2038_203831


namespace NUMINAMATH_CALUDE_friends_games_count_l2038_203871

/-- The number of games Katie's new friends have -/
def new_friends_games : ℕ := 88

/-- The number of games Katie's old friends have -/
def old_friends_games : ℕ := 53

/-- The total number of games Katie's friends have -/
def total_friends_games : ℕ := new_friends_games + old_friends_games

theorem friends_games_count : total_friends_games = 141 := by
  sorry

end NUMINAMATH_CALUDE_friends_games_count_l2038_203871


namespace NUMINAMATH_CALUDE_quadratic_function_a_range_l2038_203843

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_a_range 
  (a b c : ℝ) 
  (h1 : ∀ x, f a b c x < 0 ↔ x < 1 ∨ x > 3)
  (h2 : ∀ x, f a b c x < 2) :
  -2 < a ∧ a < 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_a_range_l2038_203843


namespace NUMINAMATH_CALUDE_lilly_fish_count_l2038_203876

/-- Given that Rosy has 12 fish and the total number of fish is 22,
    prove that Lilly has 10 fish. -/
theorem lilly_fish_count (rosy_fish : ℕ) (total_fish : ℕ) (h1 : rosy_fish = 12) (h2 : total_fish = 22) :
  total_fish - rosy_fish = 10 := by
  sorry

end NUMINAMATH_CALUDE_lilly_fish_count_l2038_203876


namespace NUMINAMATH_CALUDE_remainder_of_large_number_div_16_l2038_203837

theorem remainder_of_large_number_div_16 :
  65985241545898754582556898522454889 % 16 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_div_16_l2038_203837


namespace NUMINAMATH_CALUDE_stratified_sampling_l2038_203873

theorem stratified_sampling (total_sample : ℕ) (ratio_first : ℕ) (ratio_second : ℕ) (ratio_third : ℕ) : 
  total_sample = 50 → ratio_first = 3 → ratio_second = 4 → ratio_third = 3 →
  (ratio_second : ℚ) / (ratio_first + ratio_second + ratio_third : ℚ) * total_sample = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_l2038_203873


namespace NUMINAMATH_CALUDE_chord_length_is_four_l2038_203897

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a line in polar form ρ(sin θ - cos θ) = k -/
structure PolarLine where
  k : ℝ

/-- Represents a circle in polar form ρ = a sin θ -/
structure PolarCircle where
  a : ℝ

/-- The length of the chord cut by a polar line from a polar circle -/
noncomputable def chordLength (l : PolarLine) (c : PolarCircle) : ℝ := sorry

/-- Theorem: The chord length is 4 for the given line and circle -/
theorem chord_length_is_four :
  let l : PolarLine := { k := 2 }
  let c : PolarCircle := { a := 4 }
  chordLength l c = 4 := by sorry

end NUMINAMATH_CALUDE_chord_length_is_four_l2038_203897


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l2038_203846

-- Define the circles O₁ and O₂
def O₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def O₂ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

-- Define the center of circle C
structure CircleCenter where
  x : ℝ
  y : ℝ

-- Define the property of being externally tangent to O₁ and internally tangent to O₂
def is_tangent_to_O₁_and_O₂ (c : CircleCenter) : Prop :=
  ∃ r : ℝ, r > 0 ∧
    ((c.x - 1)^2 + c.y^2 = (r + 1)^2) ∧
    ((c.x + 1)^2 + c.y^2 = (4 - r)^2)

-- Define the locus of centers of circle C
def locus : Set CircleCenter :=
  {c : CircleCenter | is_tangent_to_O₁_and_O₂ c}

-- Theorem stating that the locus is an ellipse
theorem locus_is_ellipse :
  ∃ a b h k : ℝ, a > 0 ∧ b > 0 ∧
    ∀ c : CircleCenter, c ∈ locus ↔
      (c.x - h)^2 / a^2 + (c.y - k)^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_l2038_203846


namespace NUMINAMATH_CALUDE_platform_length_l2038_203859

/-- Given a train of length 750 m that crosses a platform in 65 seconds
    and a signal pole in 30 seconds, the length of the platform is 875 m. -/
theorem platform_length
  (train_length : ℝ)
  (platform_crossing_time : ℝ)
  (pole_crossing_time : ℝ)
  (h1 : train_length = 750)
  (h2 : platform_crossing_time = 65)
  (h3 : pole_crossing_time = 30) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 875 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l2038_203859


namespace NUMINAMATH_CALUDE_carrot_problem_l2038_203889

theorem carrot_problem (carol_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) :
  carol_carrots = 29 →
  mom_carrots = 16 →
  good_carrots = 38 →
  carol_carrots + mom_carrots - good_carrots = 7 :=
by sorry

end NUMINAMATH_CALUDE_carrot_problem_l2038_203889


namespace NUMINAMATH_CALUDE_smallest_k_for_positive_c_l2038_203806

theorem smallest_k_for_positive_c (a b c k : ℤ) : 
  a < b → b < c → 
  (2 * b = a + c) →  -- arithmetic progression
  (k * c)^2 = a * b →  -- geometric progression
  k > 1 → 
  c > 0 → 
  (∀ m : ℤ, m > 1 → m < k → ¬(∃ a' b' c' : ℤ, 
    a' < b' ∧ b' < c' ∧ 
    (2 * b' = a' + c') ∧ 
    (m * c')^2 = a' * b' ∧ 
    c' > 0)) → 
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_k_for_positive_c_l2038_203806


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l2038_203874

theorem sin_2alpha_value (α : Real) 
  (h1 : 2 * Real.cos (2 * α) = Real.sin (π / 4 - α))
  (h2 : π / 2 < α ∧ α < π) : 
  Real.sin (2 * α) = -7/8 := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l2038_203874


namespace NUMINAMATH_CALUDE_g_is_even_l2038_203850

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define g as the sum of f(x) and f(-x)
def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem stating that g is an even function
theorem g_is_even : ∀ x : ℝ, g f x = g f (-x) := by
  sorry

end NUMINAMATH_CALUDE_g_is_even_l2038_203850


namespace NUMINAMATH_CALUDE_sock_pairs_count_l2038_203851

/-- The number of ways to choose a pair of socks of different colors -/
def differentColorPairs (white brown blue black : ℕ) : ℕ :=
  white * brown + white * blue + white * black +
  brown * blue + brown * black +
  blue * black

/-- Theorem stating the number of ways to choose a pair of socks of different colors -/
theorem sock_pairs_count :
  differentColorPairs 5 5 3 3 = 94 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_count_l2038_203851


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_bound_l2038_203839

theorem quadratic_root_implies_a_bound (a : ℝ) (h1 : a > 0) 
  (h2 : 3^2 - 5/3 * a * 3 - a^2 = 0) : 1 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_bound_l2038_203839


namespace NUMINAMATH_CALUDE_B_pow_15_l2038_203891

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, -1, 0],
    ![1,  0, 0],
    ![0,  0, 1]]

theorem B_pow_15 : B ^ 15 = ![![0,  1, 0],
                              ![-1, 0, 0],
                              ![0,  0, 1]] := by
  sorry

end NUMINAMATH_CALUDE_B_pow_15_l2038_203891


namespace NUMINAMATH_CALUDE_book_count_proof_l2038_203870

/-- Given the number of books each person has, calculate the total number of books. -/
def total_books (darryl lamont loris : ℕ) : ℕ :=
  darryl + lamont + loris

theorem book_count_proof (darryl lamont loris : ℕ) 
  (h1 : darryl = 20)
  (h2 : lamont = 2 * darryl)
  (h3 : loris + 3 = lamont) :
  total_books darryl lamont loris = 97 := by
  sorry

#check book_count_proof

end NUMINAMATH_CALUDE_book_count_proof_l2038_203870


namespace NUMINAMATH_CALUDE_calculate_expression_l2038_203855

theorem calculate_expression : (-1)^2022 + Real.sqrt 9 - 2 * Real.sin (30 * π / 180) = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2038_203855


namespace NUMINAMATH_CALUDE_fixed_point_of_f_l2038_203826

/-- The logarithm function with base a, where a > 0 and a ≠ 1 -/
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

/-- The function f(x) = log_a(x+1) - 2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 1) - 2

/-- Theorem: For any a > 0 and a ≠ 1, f(x) passes through the point (0, -2) -/
theorem fixed_point_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 0 = -2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_f_l2038_203826


namespace NUMINAMATH_CALUDE_jesse_remaining_money_l2038_203854

/-- Represents the currency exchange rates -/
structure ExchangeRates where
  usd_to_gbp : ℝ
  gbp_to_eur : ℝ

/-- Represents Jesse's shopping expenses -/
structure ShoppingExpenses where
  novel_price : ℝ
  novel_count : ℕ
  novel_discount : ℝ
  lunch_multiplier : ℕ
  lunch_tax : ℝ
  lunch_tip : ℝ
  jacket_price : ℝ
  jacket_discount : ℝ

/-- Calculates Jesse's remaining money after shopping -/
def remaining_money (initial_amount : ℝ) (rates : ExchangeRates) (expenses : ShoppingExpenses) : ℝ :=
  sorry

/-- Theorem stating that Jesse's remaining money is $174.66 -/
theorem jesse_remaining_money :
  let rates := ExchangeRates.mk (1/0.7) 1.15
  let expenses := ShoppingExpenses.mk 13 10 0.2 3 0.12 0.18 120 0.3
  remaining_money 500 rates expenses = 174.66 := by sorry

end NUMINAMATH_CALUDE_jesse_remaining_money_l2038_203854


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2038_203848

theorem repeating_decimal_subtraction (x : ℚ) : x = 1/3 → 5 - 7 * x = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2038_203848


namespace NUMINAMATH_CALUDE_square_difference_divided_problem_solution_l2038_203858

theorem square_difference_divided (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
by sorry

theorem problem_solution : (305^2 - 275^2) / 30 = 580 :=
by sorry

end NUMINAMATH_CALUDE_square_difference_divided_problem_solution_l2038_203858


namespace NUMINAMATH_CALUDE_uncles_gift_amount_l2038_203875

def jerseys_cost : ℕ := 5 * 2
def basketball_cost : ℕ := 18
def shorts_cost : ℕ := 8
def money_left : ℕ := 14

theorem uncles_gift_amount : 
  jerseys_cost + basketball_cost + shorts_cost + money_left = 50 := by
  sorry

end NUMINAMATH_CALUDE_uncles_gift_amount_l2038_203875


namespace NUMINAMATH_CALUDE_find_number_to_multiply_l2038_203805

theorem find_number_to_multiply : ∃ x : ℤ, 43 * x - 34 * x = 1251 :=
by sorry

end NUMINAMATH_CALUDE_find_number_to_multiply_l2038_203805


namespace NUMINAMATH_CALUDE_dividing_chord_length_l2038_203863

/-- A hexagon inscribed in a circle with alternating side lengths -/
structure InscribedHexagon where
  side1 : ℝ
  side2 : ℝ

/-- A chord dividing the hexagon into two trapezoids -/
def dividingChord (h : InscribedHexagon) : ℝ := sorry

/-- Theorem stating the length of the dividing chord -/
theorem dividing_chord_length (h : InscribedHexagon) 
  (h_sides : h.side1 = 4 ∧ h.side2 = 7) : 
  dividingChord h = 560 / 81 := by sorry

end NUMINAMATH_CALUDE_dividing_chord_length_l2038_203863


namespace NUMINAMATH_CALUDE_coin_loss_recovery_l2038_203816

theorem coin_loss_recovery (x : ℚ) : 
  x > 0 → 
  let lost := x / 2
  let found := (4 / 5) * lost
  let remaining := x - lost + found
  x - remaining = x / 10 := by
sorry

end NUMINAMATH_CALUDE_coin_loss_recovery_l2038_203816


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l2038_203833

theorem completing_square_quadratic :
  ∀ x : ℝ, x^2 + 4*x - 1 = 0 ↔ (x + 2)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l2038_203833


namespace NUMINAMATH_CALUDE_binary_octal_conversion_l2038_203827

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

/-- The binary representation of the number in question -/
def binary_number : List Bool :=
  [true, false, true, true, false, true, true, true, false]

theorem binary_octal_conversion :
  (binary_to_decimal binary_number = 54) ∧
  (decimal_to_octal 54 = [6, 6]) :=
by sorry

end NUMINAMATH_CALUDE_binary_octal_conversion_l2038_203827


namespace NUMINAMATH_CALUDE_square_difference_equals_324_l2038_203829

theorem square_difference_equals_324 : (422 + 404)^2 - (4 * 422 * 404) = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_324_l2038_203829


namespace NUMINAMATH_CALUDE_trapezoid_angle_sum_l2038_203840

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides -/
structure Trapezoid where
  angles : Fin 4 → ℝ
  sum_angles : (angles 0) + (angles 1) + (angles 2) + (angles 3) = 360
  parallel_sides : ∃ (i j : Fin 4), i ≠ j ∧ (angles i) + (angles j) = 180

/-- Given a trapezoid with two angles of 60° and 120°, the sum of the other two angles is 180° -/
theorem trapezoid_angle_sum (t : Trapezoid) 
  (h1 : ∃ (i : Fin 4), t.angles i = 60)
  (h2 : ∃ (j : Fin 4), t.angles j = 120) :
  ∃ (k l : Fin 4), k ≠ l ∧ t.angles k + t.angles l = 180 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_angle_sum_l2038_203840


namespace NUMINAMATH_CALUDE_number_difference_l2038_203877

theorem number_difference (a b c d : ℝ) : 
  a = 2 * b ∧ 
  a = 3 * c ∧ 
  (a + b + c + d) / 4 = 110 ∧ 
  d = a + b + c 
  → a - c = 80 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2038_203877


namespace NUMINAMATH_CALUDE_candy_distribution_l2038_203872

theorem candy_distribution (total_candy : ℕ) (num_friends : ℕ) (candy_per_friend : ℕ) 
  (h1 : total_candy = 420)
  (h2 : num_friends = 35)
  (h3 : total_candy = num_friends * candy_per_friend) :
  candy_per_friend = 12 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2038_203872


namespace NUMINAMATH_CALUDE_divisibility_floor_factorial_l2038_203828

theorem divisibility_floor_factorial (m n : ℤ) 
  (h1 : 1 < m) (h2 : m < n + 2) (h3 : n > 3) : 
  (m - 1) ∣ ⌊n! / m⌋ := by
  sorry

end NUMINAMATH_CALUDE_divisibility_floor_factorial_l2038_203828


namespace NUMINAMATH_CALUDE_min_max_x_sum_l2038_203868

theorem min_max_x_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 6) 
  (sum_sq_eq : x^2 + y^2 + z^2 = 10) : 
  ∃ (x_min x_max : ℝ), 
    (∀ x', ∃ y' z', x' + y' + z' = 6 ∧ x'^2 + y'^2 + z'^2 = 10 → x_min ≤ x') ∧
    (∀ x', ∃ y' z', x' + y' + z' = 6 ∧ x'^2 + y'^2 + z'^2 = 10 → x' ≤ x_max) ∧
    x_min = 8/3 ∧ 
    x_max = 2 ∧ 
    x_min + x_max = 14/3 := by
  sorry

end NUMINAMATH_CALUDE_min_max_x_sum_l2038_203868


namespace NUMINAMATH_CALUDE_vector_problem_proof_l2038_203819

def vector_problem (a b : ℝ × ℝ) : Prop :=
  a = (1, 1) ∧
  (b.1^2 + b.2^2) = 16 ∧
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2)) = -2 →
  ((3*a.1 - b.1)^2 + (3*a.2 - b.2)^2) = 10

theorem vector_problem_proof : ∀ a b : ℝ × ℝ, vector_problem a b :=
by
  sorry

end NUMINAMATH_CALUDE_vector_problem_proof_l2038_203819


namespace NUMINAMATH_CALUDE_bernoulli_inequalities_l2038_203823

theorem bernoulli_inequalities (α : ℝ) (n : ℕ) :
  (α > 0 ∧ n > 1 → (1 + α)^n > 1 + n * α) ∧
  (0 < α ∧ α ≤ 1 / n → (1 + α)^n < 1 + n * α + n^2 * α^2) := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequalities_l2038_203823


namespace NUMINAMATH_CALUDE_max_value_of_a_l2038_203841

theorem max_value_of_a (a b c : ℕ+) 
  (h : a + b + c = Nat.gcd a b + Nat.gcd b c + Nat.gcd c a + 120) :
  a ≤ 240 ∧ ∃ a₀ b₀ c₀ : ℕ+, a₀ = 240 ∧ 
    a₀ + b₀ + c₀ = Nat.gcd a₀ b₀ + Nat.gcd b₀ c₀ + Nat.gcd c₀ a₀ + 120 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l2038_203841


namespace NUMINAMATH_CALUDE_disk_at_nine_oclock_l2038_203880

/-- Represents a circular clock face with a smaller disk rolling on it. -/
structure ClockWithDisk where
  clock_radius : ℝ
  disk_radius : ℝ
  start_position : ℝ -- in radians, 0 represents 3 o'clock
  rotation_direction : Bool -- true for clockwise

/-- Calculates the position of the disk after one full rotation -/
def position_after_rotation (c : ClockWithDisk) : ℝ :=
  sorry

/-- Theorem stating that the disk will be at 9 o'clock after one full rotation -/
theorem disk_at_nine_oclock (c : ClockWithDisk) 
  (h1 : c.clock_radius = 30)
  (h2 : c.disk_radius = 15)
  (h3 : c.start_position = 0)
  (h4 : c.rotation_direction = true) :
  position_after_rotation c = π := by
  sorry

end NUMINAMATH_CALUDE_disk_at_nine_oclock_l2038_203880


namespace NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l2038_203807

theorem x_range_for_quadratic_inequality (x : ℝ) :
  (∀ a : ℝ, -1 ≤ a ∧ a ≤ 1 → x^2 + (a-4)*x + 4-2*a > 0) ↔
  (x < -3 ∨ x > -2) :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l2038_203807


namespace NUMINAMATH_CALUDE_three_valid_rental_plans_l2038_203865

/-- Represents a rental plan for vehicles --/
structure RentalPlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a rental plan is valid for the given number of people --/
def isValidPlan (plan : RentalPlan) (totalPeople : ℕ) : Prop :=
  plan.typeA * 6 + plan.typeB * 4 = totalPeople

/-- Theorem stating that there are at least three different valid rental plans --/
theorem three_valid_rental_plans :
  ∃ (plan1 plan2 plan3 : RentalPlan),
    isValidPlan plan1 38 ∧
    isValidPlan plan2 38 ∧
    isValidPlan plan3 38 ∧
    plan1 ≠ plan2 ∧
    plan1 ≠ plan3 ∧
    plan2 ≠ plan3 := by
  sorry

end NUMINAMATH_CALUDE_three_valid_rental_plans_l2038_203865


namespace NUMINAMATH_CALUDE_smallest_among_given_rationals_l2038_203885

theorem smallest_among_given_rationals :
  let S : Finset ℚ := {-2, 0, 3, 5}
  ∀ x ∈ S, -2 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_among_given_rationals_l2038_203885


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2038_203802

/-- The line mx - y + 2m + 1 = 0 passes through the point (-2, 1) for any real m -/
theorem fixed_point_on_line (m : ℝ) : m * (-2) - 1 + 2 * m + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2038_203802


namespace NUMINAMATH_CALUDE_jack_and_jill_speed_jack_and_jill_speed_proof_l2038_203838

/-- The common speed of Jack and Jill given their walking conditions -/
theorem jack_and_jill_speed : ℝ → Prop :=
  fun (x : ℝ) ↦ 
    let jack_speed := x^2 - 11*x - 22
    let jill_distance := x^2 - 3*x - 54
    let jill_time := x + 6
    let jill_speed := jill_distance / jill_time
    (jack_speed = jill_speed) → (jack_speed = 4)

/-- Proof of the theorem -/
theorem jack_and_jill_speed_proof : ∃ x : ℝ, jack_and_jill_speed x :=
  sorry

end NUMINAMATH_CALUDE_jack_and_jill_speed_jack_and_jill_speed_proof_l2038_203838


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2038_203895

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x - 3 < 0} = {x : ℝ | -1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2038_203895


namespace NUMINAMATH_CALUDE_two_balls_in_five_boxes_with_adjacent_empty_l2038_203815

/-- Represents the number of ways to arrange two balls in two boxes -/
def A_2_2 : ℕ := 2

/-- Represents the number of distinct boxes -/
def num_boxes : ℕ := 5

/-- Represents the number of balls -/
def num_balls : ℕ := 2

/-- Represents the number of empty boxes -/
def num_empty_boxes : ℕ := num_boxes - num_balls

/-- Represents the number of adjacent empty box pairs -/
def num_adjacent_empty_pairs : ℕ := 4

/-- The main theorem to prove -/
theorem two_balls_in_five_boxes_with_adjacent_empty : 
  (2 * A_2_2 + A_2_2 + A_2_2 + 2 * A_2_2 : ℕ) = 12 := by
  sorry


end NUMINAMATH_CALUDE_two_balls_in_five_boxes_with_adjacent_empty_l2038_203815


namespace NUMINAMATH_CALUDE_flour_needed_for_90_muffins_l2038_203864

-- Define the given ratio of flour to muffins
def flour_per_muffin : ℚ := 1.5 / 15

-- Define the number of muffins Maria wants to bake
def muffins_to_bake : ℕ := 90

-- Theorem to prove
theorem flour_needed_for_90_muffins :
  (flour_per_muffin * muffins_to_bake : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_flour_needed_for_90_muffins_l2038_203864


namespace NUMINAMATH_CALUDE_wheel_revolution_distance_l2038_203813

/-- Proves that given specific wheel sizes and revolution difference, the distance traveled is 315 feet -/
theorem wheel_revolution_distance 
  (back_wheel_perimeter : ℝ) 
  (front_wheel_perimeter : ℝ) 
  (revolution_difference : ℝ) 
  (h1 : back_wheel_perimeter = 9) 
  (h2 : front_wheel_perimeter = 7) 
  (h3 : revolution_difference = 10) :
  (front_wheel_perimeter⁻¹ - back_wheel_perimeter⁻¹)⁻¹ * revolution_difference = 315 :=
by sorry

end NUMINAMATH_CALUDE_wheel_revolution_distance_l2038_203813


namespace NUMINAMATH_CALUDE_count_primes_with_squares_between_5000_and_8000_eq_5_l2038_203887

/-- The count of prime numbers whose squares are between 5000 and 8000 -/
def count_primes_with_squares_between_5000_and_8000 : Nat :=
  (Finset.filter (fun p => 5000 < p * p ∧ p * p < 8000) (Finset.filter Nat.Prime (Finset.range 90))).card

/-- Theorem stating that the count of prime numbers with squares between 5000 and 8000 is 5 -/
theorem count_primes_with_squares_between_5000_and_8000_eq_5 :
  count_primes_with_squares_between_5000_and_8000 = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_primes_with_squares_between_5000_and_8000_eq_5_l2038_203887


namespace NUMINAMATH_CALUDE_equation_solution_l2038_203894

theorem equation_solution (x : ℚ) : (40 / 60 : ℚ) = Real.sqrt (x / 60) → x = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2038_203894


namespace NUMINAMATH_CALUDE_car_replacement_problem_l2038_203888

theorem car_replacement_problem :
  let initial_fleet : ℕ := 20
  let new_cars_per_year : ℕ := 6
  let years : ℕ := 2
  ∃ (x : ℕ),
    x > 0 ∧
    initial_fleet - years * x < initial_fleet / 2 ∧
    ∀ (y : ℕ), y > 0 ∧ initial_fleet - years * y < initial_fleet / 2 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_car_replacement_problem_l2038_203888


namespace NUMINAMATH_CALUDE_annual_interest_rate_proof_l2038_203810

theorem annual_interest_rate_proof (investment1 investment2 interest1 interest2 : ℝ) 
  (h1 : investment1 = 5000)
  (h2 : investment2 = 20000)
  (h3 : interest1 = 250)
  (h4 : interest2 = 1000)
  (h5 : interest1 / investment1 = interest2 / investment2) :
  interest1 / investment1 = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_annual_interest_rate_proof_l2038_203810


namespace NUMINAMATH_CALUDE_least_positive_integer_y_l2038_203852

theorem least_positive_integer_y (y : ℕ) : 
  (∀ k : ℕ, 0 < k ∧ k < 4 → ¬(53 ∣ (3*k)^2 + 3*41*3*k + 41^2)) ∧ 
  (53 ∣ (3*4)^2 + 3*41*3*4 + 41^2) := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_y_l2038_203852


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_33_l2038_203820

theorem smallest_four_digit_divisible_by_33 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 33 = 0 → n ≥ 1023 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_33_l2038_203820


namespace NUMINAMATH_CALUDE_prime_square_plus_twelve_mod_twelve_l2038_203898

theorem prime_square_plus_twelve_mod_twelve (p : ℕ) (h_prime : Nat.Prime p) (h_gt_three : p > 3) :
  (p^2 + 12) % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_plus_twelve_mod_twelve_l2038_203898


namespace NUMINAMATH_CALUDE_angle_range_l2038_203879

theorem angle_range (α : Real) :
  (|Real.sin (4 * Real.pi - α)| = Real.sin (Real.pi + α)) →
  ∃ k : ℤ, 2 * k * Real.pi - Real.pi ≤ α ∧ α ≤ 2 * k * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_angle_range_l2038_203879


namespace NUMINAMATH_CALUDE_farmland_width_l2038_203812

/-- Represents a rectangular plot of farmland -/
structure FarmPlot where
  length : ℝ
  width : ℝ
  area : ℝ

/-- Conversion factor from acres to square feet -/
def acreToSqFt : ℝ := 43560

/-- Theorem stating the width of the farmland plot -/
theorem farmland_width (plot : FarmPlot) 
  (h1 : plot.length = 360)
  (h2 : plot.area = 10 * acreToSqFt)
  (h3 : plot.area = plot.length * plot.width) :
  plot.width = 1210 := by
  sorry

end NUMINAMATH_CALUDE_farmland_width_l2038_203812


namespace NUMINAMATH_CALUDE_students_allowance_l2038_203844

theorem students_allowance (A : ℚ) : 
  (A > 0) →
  (3 / 5 * A + 1 / 3 * (2 / 5 * A) + 0.4 = A) →
  A = 1.5 := by
sorry

end NUMINAMATH_CALUDE_students_allowance_l2038_203844


namespace NUMINAMATH_CALUDE_circle_plus_self_twice_l2038_203803

/-- Definition of the ⊕ operation -/
def circle_plus (x y : ℝ) : ℝ := x^3 + 2*x - y

/-- Theorem stating that k ⊕ (k ⊕ k) = k -/
theorem circle_plus_self_twice (k : ℝ) : circle_plus k (circle_plus k k) = k := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_self_twice_l2038_203803


namespace NUMINAMATH_CALUDE_aluminum_weight_in_compound_l2038_203878

/-- The molecular weight of the aluminum part in Al2(CO3)3 -/
def aluminum_weight : ℝ := 2 * 26.98

/-- Proof that the molecular weight of the aluminum part in Al2(CO3)3 is 53.96 g/mol -/
theorem aluminum_weight_in_compound : aluminum_weight = 53.96 := by
  sorry

end NUMINAMATH_CALUDE_aluminum_weight_in_compound_l2038_203878


namespace NUMINAMATH_CALUDE_least_positive_solution_congruence_l2038_203834

theorem least_positive_solution_congruence :
  ∃! x : ℕ+, x.val + 7813 ≡ 2500 [ZMOD 15] ∧
  ∀ y : ℕ+, y.val + 7813 ≡ 2500 [ZMOD 15] → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_solution_congruence_l2038_203834


namespace NUMINAMATH_CALUDE_star_chain_evaluation_l2038_203809

def star (a b : ℤ) : ℤ := a * b + a + b

theorem star_chain_evaluation :
  ∃ f : ℕ → ℤ, f 1 = star 1 2 ∧ 
  (∀ n : ℕ, n ≥ 2 → f n = star (f (n-1)) (n+1)) ∧
  f 99 = Nat.factorial 101 - 1 := by sorry

end NUMINAMATH_CALUDE_star_chain_evaluation_l2038_203809


namespace NUMINAMATH_CALUDE_largest_mu_inequality_l2038_203801

theorem largest_mu_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  (∀ μ : ℝ, (a^2 + b^2 + c^2 + d^2 ≥ μ * a * b + b * c + 2 * c * d) → μ ≤ 13/2) ∧
  (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a^2 + b^2 + c^2 + d^2 ≥ 13/2 * a * b + b * c + 2 * c * d) :=
by sorry

end NUMINAMATH_CALUDE_largest_mu_inequality_l2038_203801


namespace NUMINAMATH_CALUDE_three_digit_addition_l2038_203849

theorem three_digit_addition (A B : Nat) : A < 10 → B < 10 → 
  600 + 10 * A + 5 + 100 + 10 * B = 748 → B = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_addition_l2038_203849


namespace NUMINAMATH_CALUDE_square_area_ratio_l2038_203804

theorem square_area_ratio (side_c side_d : ℝ) (h1 : side_c = 48) (h2 : side_d = 60) :
  (side_c^2) / (side_d^2) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2038_203804


namespace NUMINAMATH_CALUDE_remaining_cherries_l2038_203860

def initial_cherries : ℕ := 77
def cherries_used : ℕ := 60

theorem remaining_cherries : initial_cherries - cherries_used = 17 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cherries_l2038_203860


namespace NUMINAMATH_CALUDE_list_number_fraction_l2038_203836

theorem list_number_fraction (n : ℕ) (S : ℝ) (h1 : n > 0) (h2 : S ≥ 0) : 
  n = 3 * (S / (n - 1)) → n / (S + n) = 3 / (n + 2) :=
by sorry

end NUMINAMATH_CALUDE_list_number_fraction_l2038_203836


namespace NUMINAMATH_CALUDE_absolute_value_plus_exponent_l2038_203893

theorem absolute_value_plus_exponent : |-8| + 3^0 = 9 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_plus_exponent_l2038_203893


namespace NUMINAMATH_CALUDE_friend_distribution_problem_l2038_203883

/-- The number of friends that satisfies the given conditions --/
def num_friends : ℕ := 16

/-- The total amount distributed in rupees --/
def total_amount : ℕ := 5000

/-- The decrease in amount per person if there were 8 more friends --/
def decrease_amount : ℕ := 125

theorem friend_distribution_problem :
  (total_amount / num_friends : ℚ) - (total_amount / (num_friends + 8) : ℚ) = decrease_amount ∧
  num_friends > 0 := by
  sorry

#check friend_distribution_problem

end NUMINAMATH_CALUDE_friend_distribution_problem_l2038_203883


namespace NUMINAMATH_CALUDE_ratio_equality_l2038_203830

theorem ratio_equality (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2038_203830


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l2038_203857

-- Define the vectors a and b
def a (x : ℝ) : Fin 2 → ℝ := fun i => if i = 0 then x else 1
def b (x : ℝ) : Fin 2 → ℝ := fun i => if i = 0 then 4 else x

-- Define the parallel condition
def are_parallel (x : ℝ) : Prop := ∃ k : ℝ, ∀ i : Fin 2, a x i = k * b x i

-- Statement: x = 2 is sufficient but not necessary for a and b to be parallel
theorem x_eq_2_sufficient_not_necessary :
  (are_parallel 2) ∧ (∃ y : ℝ, y ≠ 2 ∧ are_parallel y) := by sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l2038_203857


namespace NUMINAMATH_CALUDE_unique_solution_when_k_zero_l2038_203800

/-- The equation has exactly one solution when k = 0 -/
theorem unique_solution_when_k_zero :
  ∃! x : ℝ, (x + 2) / (0 * x - 1) = x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_when_k_zero_l2038_203800


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2038_203862

theorem fraction_subtraction : (5 : ℚ) / 12 - (3 : ℚ) / 18 = (1 : ℚ) / 4 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2038_203862


namespace NUMINAMATH_CALUDE_haleys_extra_tickets_l2038_203817

theorem haleys_extra_tickets (ticket_price : ℕ) (initial_tickets : ℕ) (total_spent : ℕ) : 
  ticket_price = 4 →
  initial_tickets = 3 →
  total_spent = 32 →
  total_spent / ticket_price - initial_tickets = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_haleys_extra_tickets_l2038_203817


namespace NUMINAMATH_CALUDE_peach_difference_l2038_203867

theorem peach_difference (red_peaches green_peaches : ℕ) 
  (h1 : red_peaches = 5) 
  (h2 : green_peaches = 11) : 
  green_peaches - red_peaches = 6 := by
sorry

end NUMINAMATH_CALUDE_peach_difference_l2038_203867


namespace NUMINAMATH_CALUDE_problem_solution_l2038_203869

theorem problem_solution :
  ∃ (x y : ℝ),
    (0.3 * x = 0.4 * 150 + 90) ∧
    (0.2 * x = 0.5 * 180 - 60) ∧
    (y = 0.75 * x) ∧
    (y^2 = x + 100) ∧
    (x = 150) ∧
    (y = 112.5) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2038_203869


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l2038_203835

theorem triangle_perimeter_bound (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_angle_B : B = π / 3) (h_side_b : b = 2 * Real.sqrt 3) :
  a + b + c ≤ 6 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l2038_203835


namespace NUMINAMATH_CALUDE_gift_worth_l2038_203856

/-- Calculates the worth of each gift given the company's structure and budget --/
theorem gift_worth (num_blocks : ℕ) (workers_per_block : ℕ) (total_budget : ℕ) : 
  num_blocks = 15 → 
  workers_per_block = 200 → 
  total_budget = 6000 → 
  (total_budget : ℚ) / (num_blocks * workers_per_block : ℚ) = 2 := by
  sorry

#check gift_worth

end NUMINAMATH_CALUDE_gift_worth_l2038_203856


namespace NUMINAMATH_CALUDE_relationship_abc_l2038_203892

theorem relationship_abc :
  let a : ℝ := Real.sqrt 5
  let b : ℝ := 2
  let c : ℝ := Real.sqrt 3
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l2038_203892


namespace NUMINAMATH_CALUDE_robin_extra_gum_l2038_203818

/-- Represents the number of extra pieces of gum Robin has -/
def extra_gum (total_pieces packages pieces_per_package : ℕ) : ℕ :=
  total_pieces - packages * pieces_per_package

/-- Proves that Robin has 6 extra pieces of gum given the conditions -/
theorem robin_extra_gum :
  extra_gum 41 5 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_robin_extra_gum_l2038_203818


namespace NUMINAMATH_CALUDE_sequence_property_l2038_203890

/-- Two sequences satisfying the given conditions -/
def sequences (a b : ℕ+ → ℚ) : Prop :=
  a 1 = 1/2 ∧
  (∀ n : ℕ+, a n + b n = 1) ∧
  (∀ n : ℕ+, b (n + 1) = b n / (1 - (a n)^2))

/-- The theorem to be proved -/
theorem sequence_property (a b : ℕ+ → ℚ) (h : sequences a b) :
  ∀ n : ℕ+, b n = n / (n + 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l2038_203890


namespace NUMINAMATH_CALUDE_theo_cookie_consumption_l2038_203832

/-- The number of cookies Theo eats at a time -/
def cookies_per_time : ℕ := 35

/-- The number of times Theo eats cookies per day -/
def times_per_day : ℕ := 7

/-- The number of days in a month -/
def days_per_month : ℕ := 30

/-- The number of months we are considering -/
def total_months : ℕ := 12

/-- The total number of cookies Theo can eat in the given period -/
def total_cookies : ℕ := cookies_per_time * times_per_day * days_per_month * total_months

theorem theo_cookie_consumption : total_cookies = 88200 := by
  sorry

end NUMINAMATH_CALUDE_theo_cookie_consumption_l2038_203832


namespace NUMINAMATH_CALUDE_ratio_problem_l2038_203811

theorem ratio_problem (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) :
  (2 * a + b) / (b + 2 * c) = 5 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2038_203811


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l2038_203886

def f (x : ℝ) : ℝ := (x - 2) * (x - 1) * (x + 1)

def g (x : ℝ) : ℝ := -f x

def h (x : ℝ) : ℝ := f (-x)

def c : ℕ := 3

def d : ℕ := 2

theorem intersection_points_theorem : 10 * c + d = 32 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l2038_203886


namespace NUMINAMATH_CALUDE_unique_reciprocal_function_l2038_203861

/-- Given a function f(x) = x / (ax + b) where a and b are constants, a ≠ 0,
    f(2) = 1, and f(x) = x has a unique solution, prove that f(x) = 2x / (x + 2) -/
theorem unique_reciprocal_function (a b : ℝ) (ha : a ≠ 0) :
  (∀ x, x ≠ -b/a → (x / (a * x + b) = x → ∀ y, y ≠ -b/a → y / (a * y + b) = y → x = y)) →
  (2 / (2 * a + b) = 1) →
  (∀ x, x ≠ -b/a → x / (a * x + b) = 2 * x / (x + 2)) :=
by sorry

end NUMINAMATH_CALUDE_unique_reciprocal_function_l2038_203861


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l2038_203847

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * (a + b + c) = 200) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) : 
  2 * (a * b + b * c + a * c) = 1875 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l2038_203847


namespace NUMINAMATH_CALUDE_teacher_age_l2038_203853

/-- Given a class of students and their teacher, proves the teacher's age based on average ages. -/
theorem teacher_age (num_students : ℕ) (student_avg_age teacher_age : ℝ) (total_avg_age : ℝ) :
  num_students = 20 →
  student_avg_age = 15 →
  total_avg_age = 16 →
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = total_avg_age →
  teacher_age = 36 := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_l2038_203853


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2038_203822

theorem smallest_n_congruence : 
  ∃ (n : ℕ), n > 0 ∧ 1145 * n ≡ 1717 * n [ZMOD 36] ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n → ¬(1145 * m ≡ 1717 * m [ZMOD 36])) ∧ 
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2038_203822


namespace NUMINAMATH_CALUDE_cosine_symmetry_center_l2038_203814

/-- The symmetry center of the cosine function f(x) = 3cos(2(x - π/6) + π/2) -/
theorem cosine_symmetry_center : 
  let f : ℝ → ℝ := λ x ↦ 3 * Real.cos (2 * (x - π/6) + π/2)
  ∃ (center : ℝ × ℝ), center = (π/6, 0) ∧ 
    ∀ (x : ℝ), f (center.1 + x) = f (center.1 - x) :=
by sorry

end NUMINAMATH_CALUDE_cosine_symmetry_center_l2038_203814


namespace NUMINAMATH_CALUDE_tangent_circles_t_value_l2038_203824

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y t : ℝ) : Prop := (x - t)^2 + y^2 = 1

-- Define the condition of external tangency
def externally_tangent (t : ℝ) : Prop := ∃ x y : ℝ, circle1 x y ∧ circle2 x y t

-- Theorem statement
theorem tangent_circles_t_value :
  ∀ t : ℝ, externally_tangent t → (t = 3 ∨ t = -3) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_t_value_l2038_203824


namespace NUMINAMATH_CALUDE_barry_sotter_magic_l2038_203882

-- Define the length increase factor for day k
def increase_factor (k : ℕ) : ℚ := (2 * k + 2) / (2 * k + 1)

-- Define the total increase factor after n days
def total_increase (n : ℕ) : ℚ := (2 * n + 2) / 2

-- Theorem statement
theorem barry_sotter_magic (n : ℕ) : total_increase n = 50 ↔ n = 49 := by
  sorry

end NUMINAMATH_CALUDE_barry_sotter_magic_l2038_203882


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_twenty_l2038_203884

theorem last_digit_of_one_over_two_to_twenty (n : ℕ) :
  n = 20 →
  ∃ k : ℕ, (1 : ℚ) / (2^n) = k * (1 / 10^n) + 5 * (1 / 10^n) :=
sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_twenty_l2038_203884


namespace NUMINAMATH_CALUDE_min_value_equiv_k_l2038_203821

/-- The polynomial function f(x, y, k) -/
def f (x y k : ℝ) : ℝ := 9*x^2 - 12*k*x*y + (2*k^2 + 3)*y^2 - 6*x - 9*y + 12

/-- The theorem stating the equivalence between the minimum value of f being 0 and k = √(3)/4 -/
theorem min_value_equiv_k (k : ℝ) : 
  (∀ x y : ℝ, f x y k ≥ 0) ∧ (∃ x y : ℝ, f x y k = 0) ↔ k = Real.sqrt 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_equiv_k_l2038_203821


namespace NUMINAMATH_CALUDE_divisibility_property_l2038_203881

theorem divisibility_property (n : ℕ) (k : ℤ) 
  (h1 : n > 0) 
  (h2 : ¬ 2 ∣ n) 
  (h3 : ¬ 3 ∣ n) : 
  (k^2 + k + 1) ∣ ((k + 1)^n - k^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2038_203881


namespace NUMINAMATH_CALUDE_shared_savings_theorem_l2038_203866

/-- Calculates the monthly savings per person for a shared down payment -/
def monthly_savings_per_person (down_payment : ℕ) (years : ℕ) : ℕ :=
  down_payment / (years * 12) / 2

/-- Theorem: Two people saving equally for a $108,000 down payment over 3 years each save $1,500 per month -/
theorem shared_savings_theorem :
  monthly_savings_per_person 108000 3 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_shared_savings_theorem_l2038_203866


namespace NUMINAMATH_CALUDE_xy_positive_necessary_not_sufficient_l2038_203845

theorem xy_positive_necessary_not_sufficient (x y : ℝ) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 → x * y > 0) ∧
  (∃ x y : ℝ, x * y > 0 ∧ ¬(x > 0 ∧ y > 0)) :=
by sorry

end NUMINAMATH_CALUDE_xy_positive_necessary_not_sufficient_l2038_203845


namespace NUMINAMATH_CALUDE_unique_solution_star_l2038_203808

/-- The star operation defined on real numbers -/
def star (x y : ℝ) : ℝ := 3 * x - 2 * y + x^2 * y

/-- Theorem stating that there's exactly one solution to 2 ⋆ y = 9 -/
theorem unique_solution_star :
  ∃! y : ℝ, star 2 y = 9 := by sorry

end NUMINAMATH_CALUDE_unique_solution_star_l2038_203808


namespace NUMINAMATH_CALUDE_alphabet_value_proof_l2038_203825

/-- Given the alphabet values where H = 8, prove that A = 25 when PACK = 50, PECK = 54, and CAKE = 40 -/
theorem alphabet_value_proof (P A C K E : ℤ) (h1 : P + A + C + K = 50) (h2 : P + E + C + K = 54) (h3 : C + A + K + E = 40) : A = 25 := by
  sorry

end NUMINAMATH_CALUDE_alphabet_value_proof_l2038_203825


namespace NUMINAMATH_CALUDE_expression_evaluation_l2038_203899

theorem expression_evaluation : 
  ((-1) ^ 2022) + |1 - Real.sqrt 2| + ((-27) ^ (1/3 : ℝ)) - Real.sqrt (((-2) ^ 2)) = Real.sqrt 2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2038_203899


namespace NUMINAMATH_CALUDE_cannot_cover_naturals_with_disjoint_sets_l2038_203896

def S (α : ℝ) : Set ℕ := {n : ℕ | ∃ m : ℕ, n = ⌊m * α⌋}

theorem cannot_cover_naturals_with_disjoint_sets :
  ∀ (α β γ : ℝ), α > 0 ∧ β > 0 ∧ γ > 0 →
  ¬(Disjoint (S α) (S β) ∧ Disjoint (S α) (S γ) ∧ Disjoint (S β) (S γ) ∧
    (S α ∪ S β ∪ S γ) = Set.univ) :=
sorry

end NUMINAMATH_CALUDE_cannot_cover_naturals_with_disjoint_sets_l2038_203896


namespace NUMINAMATH_CALUDE_not_p_and_p_or_q_implies_q_l2038_203842

theorem not_p_and_p_or_q_implies_q (p q : Prop) : (¬p ∧ (p ∨ q)) → q := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_p_or_q_implies_q_l2038_203842
