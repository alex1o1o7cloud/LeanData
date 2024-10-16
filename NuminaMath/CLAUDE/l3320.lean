import Mathlib

namespace NUMINAMATH_CALUDE_village_foods_monthly_sales_l3320_332041

/-- Represents the monthly sales data for Village Foods --/
structure VillageFoodsSales where
  customers : ℕ
  lettucePerCustomer : ℕ
  lettucePrice : ℚ
  tomatoesPerCustomer : ℕ
  tomatoPrice : ℚ

/-- Calculates the total monthly sales from lettuce and tomatoes --/
def totalMonthlySales (s : VillageFoodsSales) : ℚ :=
  s.customers * (s.lettucePerCustomer * s.lettucePrice + s.tomatoesPerCustomer * s.tomatoPrice)

/-- Theorem stating that the total monthly sales for the given conditions is $2000 --/
theorem village_foods_monthly_sales :
  let sales := VillageFoodsSales.mk 500 2 1 4 (1/2)
  totalMonthlySales sales = 2000 := by sorry

end NUMINAMATH_CALUDE_village_foods_monthly_sales_l3320_332041


namespace NUMINAMATH_CALUDE_region_D_properties_l3320_332017

def region_D (x y : ℝ) : Prop :=
  2 ≤ x ∧ x ≤ 6 ∧
  1 ≤ y ∧ y ≤ 3 ∧
  x^2 / 9 + y^2 / 4 < 1 ∧
  4 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 9 ∧
  0 < y ∧ y < x

theorem region_D_properties (x y : ℝ) :
  region_D x y →
  (2 ≤ x ∧ x ≤ 6) ∧
  (1 ≤ y ∧ y ≤ 3) ∧
  (x^2 / 9 + y^2 / 4 < 1) ∧
  (4 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 9) ∧
  (0 < y ∧ y < x) :=
by sorry

end NUMINAMATH_CALUDE_region_D_properties_l3320_332017


namespace NUMINAMATH_CALUDE_exactly_one_mean_value_point_l3320_332015

-- Define the function f(x) = x³ + 2x
def f (x : ℝ) : ℝ := x^3 + 2*x

-- Define the mean value point condition
def is_mean_value_point (f : ℝ → ℝ) (x₀ : ℝ) (a b : ℝ) : Prop :=
  x₀ ∈ Set.Icc a b ∧ f x₀ = (∫ (x : ℝ) in a..b, f x) / (b - a)

-- Theorem statement
theorem exactly_one_mean_value_point :
  ∃! x₀ : ℝ, is_mean_value_point f x₀ (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_mean_value_point_l3320_332015


namespace NUMINAMATH_CALUDE_fib_product_divisibility_l3320_332059

/-- Mersenne sequence property: for any two positive integers i and j, gcd(aᵢ, aⱼ) = a_{gcd(i,j)} -/
def is_mersenne_sequence (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i > 0 → j > 0 → Nat.gcd (a i) (a j) = a (Nat.gcd i j)

/-- Fibonacci sequence definition -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n+2 => fib (n+1) + fib n

/-- Product of first n terms of a sequence -/
def seq_product (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * a (i+1)) 1

/-- Main theorem: For Fibonacci sequence, product of k consecutive terms 
    is divisible by product of first k terms -/
theorem fib_product_divisibility (k : ℕ) (n : ℕ) : 
  k > 0 → is_mersenne_sequence fib → 
  (seq_product fib k) ∣ (List.range k).foldl (λ acc i => acc * fib (n+i)) 1 :=
sorry

end NUMINAMATH_CALUDE_fib_product_divisibility_l3320_332059


namespace NUMINAMATH_CALUDE_red_balls_count_l3320_332026

/-- The number of red balls in a bag with given conditions -/
theorem red_balls_count (total : ℕ) (white green yellow purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 60 →
  white = 22 →
  green = 18 →
  yellow = 8 →
  purple = 7 →
  prob_not_red_purple = 4/5 →
  (white + green + yellow : ℚ) / total = prob_not_red_purple →
  total - (white + green + yellow + purple) = 5 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l3320_332026


namespace NUMINAMATH_CALUDE_algebraic_simplification_and_evaluation_l3320_332086

theorem algebraic_simplification_and_evaluation (a b : ℝ) :
  2 * (a * b^2 + 3 * a^2 * b) - 3 * (a * b^2 + a^2 * b) = -a * b^2 + 3 * a^2 * b ∧
  2 * ((-1) * 2^2 + 3 * (-1)^2 * 2) - 3 * ((-1) * 2^2 + (-1)^2 * 2) = 10 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_simplification_and_evaluation_l3320_332086


namespace NUMINAMATH_CALUDE_matts_writing_speed_l3320_332007

/-- Matt's writing speed problem -/
theorem matts_writing_speed 
  (x : ℕ) -- x is the number of words Matt can write per minute with his right hand
  (h1 : x > 0) -- Matt can write some words a minute with his right hand
  (h2 : 5 * x = 5 * 7 + 15) -- In 5 minutes, Matt writes 15 more words with his right hand than with his left
  : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_matts_writing_speed_l3320_332007


namespace NUMINAMATH_CALUDE_thirteenth_result_l3320_332010

theorem thirteenth_result (total_count : Nat) (total_average : ℝ) 
  (first_twelve_average : ℝ) (last_twelve_average : ℝ) :
  total_count = 25 →
  total_average = 20 →
  first_twelve_average = 14 →
  last_twelve_average = 17 →
  (12 * first_twelve_average + 12 * last_twelve_average + 
    (total_count * total_average - 12 * first_twelve_average - 12 * last_twelve_average)) / 1 = 128 := by
  sorry

#check thirteenth_result

end NUMINAMATH_CALUDE_thirteenth_result_l3320_332010


namespace NUMINAMATH_CALUDE_zero_in_interval_l3320_332060

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * log x + 2 * x^2 - 4 * x

theorem zero_in_interval :
  ∃ (c : ℝ), c ∈ Set.Ioo 1 (exp 1) ∧ f c = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l3320_332060


namespace NUMINAMATH_CALUDE_scalar_projection_implies_k_l3320_332035

/-- Given vectors a and b in ℝ², prove that if the scalar projection of b onto a is 1, then the first component of b is 3. -/
theorem scalar_projection_implies_k (a b : ℝ × ℝ) :
  a = (3, 4) →
  b.2 = -1 →
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (a.1^2 + a.2^2) = 1 →
  b.1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_scalar_projection_implies_k_l3320_332035


namespace NUMINAMATH_CALUDE_total_money_l3320_332082

-- Define Tim's and Alice's money as fractions of a dollar
def tim_money : ℚ := 5/8
def alice_money : ℚ := 2/5

-- Theorem statement
theorem total_money :
  tim_money + alice_money = 1.025 := by sorry

end NUMINAMATH_CALUDE_total_money_l3320_332082


namespace NUMINAMATH_CALUDE_correct_representation_l3320_332071

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be represented -/
def number : ℕ := 91000

/-- The scientific notation representation of the number -/
def representation : ScientificNotation := {
  coefficient := 9.1
  exponent := 4
  h1 := by sorry
}

/-- Theorem: The given representation is correct for the number -/
theorem correct_representation : 
  (representation.coefficient * (10 : ℝ) ^ representation.exponent) = number := by sorry

end NUMINAMATH_CALUDE_correct_representation_l3320_332071


namespace NUMINAMATH_CALUDE_library_wage_calculation_l3320_332099

/-- Represents the weekly work schedule and earnings of a student with two part-time jobs -/
structure WorkSchedule where
  library_hours : ℝ
  construction_hours : ℝ
  library_wage : ℝ
  construction_wage : ℝ
  total_earnings : ℝ

/-- Theorem stating the library wage given the problem conditions -/
theorem library_wage_calculation (w : WorkSchedule) :
  w.library_hours = 10 ∧
  w.construction_hours = 15 ∧
  w.construction_wage = 15 ∧
  w.library_hours + w.construction_hours = 25 ∧
  w.total_earnings ≥ 300 ∧
  w.total_earnings = w.library_hours * w.library_wage + w.construction_hours * w.construction_wage →
  w.library_wage = 7.5 := by
  sorry

#check library_wage_calculation

end NUMINAMATH_CALUDE_library_wage_calculation_l3320_332099


namespace NUMINAMATH_CALUDE_power_function_not_in_second_quadrant_l3320_332054

def f (x : ℝ) : ℝ := x

theorem power_function_not_in_second_quadrant :
  (∀ x : ℝ, x < 0 → f x ≤ 0) ∧
  (∀ x : ℝ, f x = x) :=
sorry

end NUMINAMATH_CALUDE_power_function_not_in_second_quadrant_l3320_332054


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_squared_times_i_minus_one_l3320_332025

theorem imaginary_part_of_i_squared_times_i_minus_one (i : ℂ) : 
  i^2 = -1 → Complex.im (i^2 * (i - 1)) = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_squared_times_i_minus_one_l3320_332025


namespace NUMINAMATH_CALUDE_difference_calculations_l3320_332030

theorem difference_calculations (d1 d2 d3 : Int) 
  (h1 : d1 = -15)
  (h2 : d2 = 405)
  (h3 : d3 = 1280) :
  let sum := d1 + d2 + d3
  let product := d1 * d2 * d3
  let avg_squares := ((d1^2 + d2^2 + d3^2) : ℚ) / 3
  sum = 1670 ∧ 
  product = -7728000 ∧ 
  avg_squares = 600883 + 1/3 ∧
  (product : ℚ) - avg_squares = -8328883 - 1/3 := by
sorry

#eval (-15 : Int) + 405 + 1280
#eval (-15 : Int) * 405 * 1280
#eval ((-15 : ℚ)^2 + 405^2 + 1280^2) / 3
#eval (-7728000 : ℚ) - (((-15 : ℚ)^2 + 405^2 + 1280^2) / 3)

end NUMINAMATH_CALUDE_difference_calculations_l3320_332030


namespace NUMINAMATH_CALUDE_time_between_periods_l3320_332024

theorem time_between_periods 
  (total_time : ℕ)
  (num_periods : ℕ)
  (period_duration : ℕ)
  (h1 : total_time = 220)
  (h2 : num_periods = 5)
  (h3 : period_duration = 40) :
  (total_time - num_periods * period_duration) / (num_periods - 1) = 5 :=
by sorry

end NUMINAMATH_CALUDE_time_between_periods_l3320_332024


namespace NUMINAMATH_CALUDE_min_probability_bound_l3320_332039

open Real

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- The probability function P(k) -/
noncomputable def P (k : ℕ) : ℝ :=
  let count := Finset.filter (fun n : ℕ => 
    floor (n / k) + floor ((200 - n) / k) = floor (200 / k)) 
    (Finset.range 199)
  (count.card : ℝ) / 199

theorem min_probability_bound :
  ∀ k : ℕ, k % 2 = 1 → 1 ≤ k → k ≤ 199 → P k ≥ 50 / 101 := by sorry

end NUMINAMATH_CALUDE_min_probability_bound_l3320_332039


namespace NUMINAMATH_CALUDE_card_passing_game_theorem_l3320_332047

/-- Represents the state of the card-passing game -/
structure GameState where
  num_students : ℕ
  num_cards : ℕ
  card_distribution : List ℕ

/-- Defines a valid game state -/
def valid_game_state (state : GameState) : Prop :=
  state.num_students = 1994 ∧
  state.card_distribution.length = state.num_students ∧
  state.card_distribution.sum = state.num_cards

/-- Defines the condition for the game to end -/
def game_ends (state : GameState) : Prop :=
  ∀ n, n ∈ state.card_distribution → n ≤ 1

/-- Defines the ability to continue the game -/
def can_continue (state : GameState) : Prop :=
  ∃ n, n ∈ state.card_distribution ∧ n ≥ 2

/-- Main theorem about the card-passing game -/
theorem card_passing_game_theorem (state : GameState) 
  (h_valid : valid_game_state state) :
  (state.num_cards ≥ state.num_students → 
    ∃ (game_sequence : ℕ → GameState), ∀ n, can_continue (game_sequence n)) ∧
  (state.num_cards < state.num_students → 
    ∃ (game_sequence : ℕ → GameState) (end_state : ℕ), game_ends (game_sequence end_state)) :=
sorry

end NUMINAMATH_CALUDE_card_passing_game_theorem_l3320_332047


namespace NUMINAMATH_CALUDE_prime_sum_product_l3320_332045

theorem prime_sum_product (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p + q = 105 → p * q = 206 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_product_l3320_332045


namespace NUMINAMATH_CALUDE_no_solutions_for_inequality_l3320_332087

theorem no_solutions_for_inequality : 
  ¬ ∃ (n : ℕ), n ≥ 1 ∧ n ≤ n! - 4^n ∧ n! - 4^n ≤ 4*n :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_for_inequality_l3320_332087


namespace NUMINAMATH_CALUDE_complex_cube_root_l3320_332061

theorem complex_cube_root (a b : ℕ+) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (↑a + ↑b * Complex.I) ^ 3 = (2 : ℂ) + 11 * Complex.I →
  ↑a + ↑b * Complex.I = (2 : ℂ) + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_l3320_332061


namespace NUMINAMATH_CALUDE_probability_circle_or_square_l3320_332005

def total_figures : ℕ := 10
def num_circles : ℕ := 3
def num_squares : ℕ := 4
def num_triangles : ℕ := 3

theorem probability_circle_or_square :
  (num_circles + num_squares : ℚ) / total_figures = 7 / 10 :=
by sorry

end NUMINAMATH_CALUDE_probability_circle_or_square_l3320_332005


namespace NUMINAMATH_CALUDE_smallest_b_for_divisibility_l3320_332016

def is_single_digit (n : ℕ) : Prop := n < 10

def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem smallest_b_for_divisibility : 
  ∃ (B : ℕ), is_single_digit B ∧ 
             is_divisible_by_13 (200 + 10 * B + 5) ∧ 
             (∀ (k : ℕ), k < B → ¬(is_divisible_by_13 (200 + 10 * k + 5))) ∧
             B = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_b_for_divisibility_l3320_332016


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_300_l3320_332067

theorem distinct_prime_factors_of_300 : Nat.card (Nat.factors 300).toFinset = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_300_l3320_332067


namespace NUMINAMATH_CALUDE_dice_product_nonzero_probability_l3320_332093

/-- The probability of getting a specific outcome when rolling a standard die -/
def roll_probability : ℚ := 1 / 6

/-- The number of faces on a standard die -/
def die_faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The probability that a single die roll is not 1 -/
def prob_not_one : ℚ := (die_faces - 1) / die_faces

theorem dice_product_nonzero_probability :
  (prob_not_one ^ num_dice : ℚ) = 625 / 1296 := by sorry

end NUMINAMATH_CALUDE_dice_product_nonzero_probability_l3320_332093


namespace NUMINAMATH_CALUDE_trig_identity_l3320_332079

theorem trig_identity : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) + 
  Real.sqrt 3 * Real.sin (10 * π / 180) * Real.tan (70 * π / 180) - 
  2 * Real.sin (50 * π / 180) = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3320_332079


namespace NUMINAMATH_CALUDE_triangle_angles_from_height_intersections_l3320_332083

/-- Given an acute-angled triangle ABC with circumscribed circle,
    let p, q, r be positive real numbers representing the ratio of arc lengths
    formed by the intersections of the extended heights with the circle.
    This theorem states the relationship between these ratios and the angles of the triangle. -/
theorem triangle_angles_from_height_intersections
  (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  let α := Real.pi / 2 * ((q + r) / (p + q + r))
  let β := Real.pi / 2 * (q / (p + q + r))
  let γ := Real.pi / 2 * (r / (p + q + r))
  α + β + γ = Real.pi ∧ 0 < α ∧ α < Real.pi/2 ∧ 0 < β ∧ β < Real.pi/2 ∧ 0 < γ ∧ γ < Real.pi/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_from_height_intersections_l3320_332083


namespace NUMINAMATH_CALUDE_smaller_number_problem_l3320_332002

theorem smaller_number_problem (x y : ℝ) : 
  x + y = 14 ∧ y = 3 * x → x = 3.5 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l3320_332002


namespace NUMINAMATH_CALUDE_player_positions_satisfy_distances_l3320_332032

/-- Represents the positions of four soccer players on a number line -/
def PlayerPositions : Fin 4 → ℝ
  | 0 => 0
  | 1 => 1
  | 2 => 4
  | 3 => 6

/-- Calculates the distance between two players -/
def distance (i j : Fin 4) : ℝ :=
  |PlayerPositions i - PlayerPositions j|

/-- The set of required pairwise distances -/
def RequiredDistances : Set ℝ := {1, 2, 3, 4, 5, 6}

/-- Theorem stating that the player positions satisfy the required distances -/
theorem player_positions_satisfy_distances :
  ∀ i j : Fin 4, i ≠ j → distance i j ∈ RequiredDistances :=
sorry

end NUMINAMATH_CALUDE_player_positions_satisfy_distances_l3320_332032


namespace NUMINAMATH_CALUDE_domain_of_f_l3320_332055

noncomputable def f (x : ℝ) := Real.log (2 * (Real.cos x)^2 - 1)

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

theorem domain_of_f : domain f = {x : ℝ | ∃ k : ℤ, k * Real.pi - Real.pi / 4 < x ∧ x < k * Real.pi + Real.pi / 4} := by sorry

end NUMINAMATH_CALUDE_domain_of_f_l3320_332055


namespace NUMINAMATH_CALUDE_cupcake_icing_time_l3320_332052

theorem cupcake_icing_time (total_batches : ℕ) (baking_time_per_batch : ℕ) (total_time : ℕ) :
  total_batches = 4 →
  baking_time_per_batch = 20 →
  total_time = 200 →
  (total_time - total_batches * baking_time_per_batch) / total_batches = 30 :=
by sorry

end NUMINAMATH_CALUDE_cupcake_icing_time_l3320_332052


namespace NUMINAMATH_CALUDE_money_difference_l3320_332014

-- Define the amounts for each day
def tuesday_amount : ℝ := 8.5
def wednesday_amount : ℝ := 5.5 * tuesday_amount
def thursday_amount : ℝ := wednesday_amount * 1.1
def friday_amount : ℝ := thursday_amount * 0.75

-- Define the difference
def difference : ℝ := friday_amount - tuesday_amount

-- Theorem statement
theorem money_difference : difference = 30.06875 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_l3320_332014


namespace NUMINAMATH_CALUDE_largest_positive_integer_for_binary_op_l3320_332044

def binary_op (n : Int) : Int := n - (n * 5)

theorem largest_positive_integer_for_binary_op :
  ∀ n : ℕ+, n > 1 → binary_op n.val ≥ 14 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_positive_integer_for_binary_op_l3320_332044


namespace NUMINAMATH_CALUDE_faye_age_l3320_332075

/-- Represents the ages of the individuals in the problem --/
structure Ages where
  diana : ℕ
  eduardo : ℕ
  chad : ℕ
  faye : ℕ
  george : ℕ

/-- The conditions of the problem --/
def valid_ages (a : Ages) : Prop :=
  a.diana + 2 = a.eduardo ∧
  a.eduardo = a.chad + 6 ∧
  a.faye = a.chad + 4 ∧
  a.george + 5 = a.chad ∧
  a.diana = 16

/-- The theorem to prove --/
theorem faye_age (a : Ages) (h : valid_ages a) : a.faye = 16 := by
  sorry

#check faye_age

end NUMINAMATH_CALUDE_faye_age_l3320_332075


namespace NUMINAMATH_CALUDE_gcd_840_1764_l3320_332089

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l3320_332089


namespace NUMINAMATH_CALUDE_fermats_little_theorem_l3320_332029

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (h : Prime p) :
  a^p ≡ a [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_fermats_little_theorem_l3320_332029


namespace NUMINAMATH_CALUDE_product_of_cosines_equals_one_eighth_l3320_332058

theorem product_of_cosines_equals_one_eighth :
  (1 + Real.cos (π / 12)) * (1 + Real.cos (5 * π / 12)) *
  (1 + Real.cos (7 * π / 12)) * (1 + Real.cos (11 * π / 12)) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cosines_equals_one_eighth_l3320_332058


namespace NUMINAMATH_CALUDE_train_speed_l3320_332042

/-- The speed of a train crossing a platform -/
theorem train_speed (train_length platform_length : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  platform_length = 165 →
  crossing_time = 7.499400047996161 →
  ∃ (speed : ℝ), abs (speed - 132.01) < 0.01 ∧ 
  speed = (train_length + platform_length) / crossing_time * 3.6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3320_332042


namespace NUMINAMATH_CALUDE_ammunition_depot_explosion_probability_l3320_332037

theorem ammunition_depot_explosion_probability 
  (p_first : ℝ) 
  (p_others : ℝ) 
  (h1 : p_first = 0.025) 
  (h2 : p_others = 0.1) : 
  1 - (1 - p_first) * (1 - p_others) * (1 - p_others) = 0.21025 := by
  sorry

end NUMINAMATH_CALUDE_ammunition_depot_explosion_probability_l3320_332037


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3320_332066

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + k = 0 ∧ x = 4) → 
  (∃ y : ℝ, y^2 - 3*y + k = 0 ∧ y = -1) ∧ k = -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3320_332066


namespace NUMINAMATH_CALUDE_p_subset_q_condition_l3320_332091

def P : Set ℝ := {x : ℝ | |x + 2| ≤ 3}
def Q : Set ℝ := {x : ℝ | x ≥ -8}

theorem p_subset_q_condition : P ⊂ Q ∧ 
  (∀ x : ℝ, x ∈ P → x ∈ Q) ∧ 
  (∃ x : ℝ, x ∈ Q ∧ x ∉ P) := by
  sorry

end NUMINAMATH_CALUDE_p_subset_q_condition_l3320_332091


namespace NUMINAMATH_CALUDE_same_gender_probability_same_school_probability_l3320_332056

-- Define the schools and their teacher compositions
def school_A : Nat := 3
def school_A_males : Nat := 2
def school_A_females : Nat := 1

def school_B : Nat := 3
def school_B_males : Nat := 1
def school_B_females : Nat := 2

def total_teachers : Nat := school_A + school_B

-- Theorem for the first question
theorem same_gender_probability :
  (school_A_males * school_B_males + school_A_females * school_B_females) /
  (school_A * school_B) = 4 / 9 :=
by sorry

-- Theorem for the second question
theorem same_school_probability :
  (school_A * (school_A - 1) / 2 + school_B * (school_B - 1) / 2) /
  (total_teachers * (total_teachers - 1) / 2) = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_same_gender_probability_same_school_probability_l3320_332056


namespace NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l3320_332084

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola with foci F₁ and F₂, and real axis length 2a -/
structure Hyperbola where
  F₁ : Point
  F₂ : Point
  a : ℝ

/-- Theorem: Perimeter of triangle ABF₂ in a hyperbola -/
theorem hyperbola_triangle_perimeter 
  (h : Hyperbola) 
  (A B : Point) 
  (m : ℝ) 
  (h_line : A.x = B.x ∧ A.x = h.F₁.x) -- A, B, and F₁ are collinear
  (h_on_hyperbola : 
    |A.x - h.F₂.x| + |A.y - h.F₂.y| - |A.x - h.F₁.x| - |A.y - h.F₁.y| = 2 * h.a ∧
    |B.x - h.F₂.x| + |B.y - h.F₂.y| - |B.x - h.F₁.x| - |B.y - h.F₁.y| = 2 * h.a)
  (h_AB : |A.x - B.x| + |A.y - B.y| = m) :
  |A.x - h.F₂.x| + |A.y - h.F₂.y| + |B.x - h.F₂.x| + |B.y - h.F₂.y| + m = 4 * h.a + 2 * m := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l3320_332084


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_value_of_m_for_intersection_l3320_332051

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x - 5 ≤ 0}

-- Define set B with parameter m
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem 1: Intersection of A and complement of B when m = 3
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 3) = {x | x = -1 ∨ (3 ≤ x ∧ x ≤ 5)} := by sorry

-- Theorem 2: Value of m when A ∩ B = {x | -1 ≤ x < 4}
theorem value_of_m_for_intersection :
  ∃ m : ℝ, A ∩ B m = {x | -1 ≤ x ∧ x < 4} ∧ m = 8 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_value_of_m_for_intersection_l3320_332051


namespace NUMINAMATH_CALUDE_additional_nails_l3320_332028

/-- Calculates the number of additional nails used in a house wall construction. -/
theorem additional_nails (total_nails : ℕ) (nails_per_plank : ℕ) (planks_needed : ℕ) :
  total_nails = 11 →
  nails_per_plank = 3 →
  planks_needed = 1 →
  total_nails - (nails_per_plank * planks_needed) = 8 := by
  sorry

#check additional_nails

end NUMINAMATH_CALUDE_additional_nails_l3320_332028


namespace NUMINAMATH_CALUDE_joe_game_buying_duration_l3320_332001

/-- Calculates the number of months before running out of money given initial amount, monthly spending, and monthly income. -/
def monthsBeforeBroke (initialAmount : ℕ) (monthlySpending : ℕ) (monthlyIncome : ℕ) : ℕ :=
  initialAmount / (monthlySpending - monthlyIncome)

/-- Theorem stating that given the specific conditions, Joe can buy and sell games for 12 months before running out of money. -/
theorem joe_game_buying_duration :
  monthsBeforeBroke 240 50 30 = 12 := by
  sorry

end NUMINAMATH_CALUDE_joe_game_buying_duration_l3320_332001


namespace NUMINAMATH_CALUDE_smallest_number_l3320_332031

-- Define the base conversion function
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the numbers in their respective bases
def A : Nat := to_decimal [1, 1, 1, 1] 2
def B : Nat := to_decimal [0, 1, 2] 6
def C : Nat := to_decimal [0, 0, 0, 1] 4
def D : Nat := to_decimal [1, 0, 1] 8

-- Theorem statement
theorem smallest_number : A < B ∧ A < C ∧ A < D := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3320_332031


namespace NUMINAMATH_CALUDE_chessboard_diagonal_ratio_l3320_332057

/-- Represents a rectangle with chessboard coloring -/
structure ChessboardRectangle where
  a : ℕ  -- length
  b : ℕ  -- width

/-- Calculates the ratio of white to black segment lengths on the diagonal -/
def diagonalSegmentRatio (rect : ChessboardRectangle) : ℚ :=
  if rect.a = 100 ∧ rect.b = 99 then 1
  else if rect.a = 101 ∧ rect.b = 99 then 5000 / 4999
  else 0  -- undefined for other dimensions

theorem chessboard_diagonal_ratio :
  ∀ (rect : ChessboardRectangle),
    (rect.a = 100 ∧ rect.b = 99 → diagonalSegmentRatio rect = 1) ∧
    (rect.a = 101 ∧ rect.b = 99 → diagonalSegmentRatio rect = 5000 / 4999) :=
by sorry

end NUMINAMATH_CALUDE_chessboard_diagonal_ratio_l3320_332057


namespace NUMINAMATH_CALUDE_second_polygon_sides_l3320_332040

/-- Given two regular polygons with equal perimeters, where the first has 50 sides
    and a side length three times as long as the second, prove the second has 150 sides. -/
theorem second_polygon_sides (s : ℝ) (h_s_pos : s > 0) : 
  50 * (3 * s) = 150 * s → 150 = 150 :=
by sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l3320_332040


namespace NUMINAMATH_CALUDE_complex_fraction_equals_seven_plus_i_l3320_332000

theorem complex_fraction_equals_seven_plus_i :
  let i : ℂ := Complex.I
  (1 + i) * (3 + 4*i) / i = 7 + i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_seven_plus_i_l3320_332000


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3320_332019

/-- The area of a square inscribed in a circular segment with an arc of 60° and radius 2√3 + √17 is 1. -/
theorem inscribed_square_area (R : ℝ) (h : R = 2 * Real.sqrt 3 + Real.sqrt 17) : 
  let segment_arc : ℝ := 60 * π / 180
  let square_side : ℝ := (R * (Real.sqrt 17 - 2 * Real.sqrt 3)) / 5
  square_side ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3320_332019


namespace NUMINAMATH_CALUDE_pyramid_volume_theorem_l3320_332080

/-- Represents a pyramid with a square base ABCD and vertex E -/
structure Pyramid where
  baseArea : ℝ
  triangleABEArea : ℝ
  triangleCDEArea : ℝ

/-- Calculates the volume of the pyramid -/
def pyramidVolume (p : Pyramid) : ℝ :=
  sorry

/-- Theorem stating the volume of the pyramid with given conditions -/
theorem pyramid_volume_theorem (p : Pyramid) 
  (h1 : p.baseArea = 256)
  (h2 : p.triangleABEArea = 128)
  (h3 : p.triangleCDEArea = 96) :
  pyramidVolume p = 1194 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_theorem_l3320_332080


namespace NUMINAMATH_CALUDE_spelling_contest_result_l3320_332008

/-- In a spelling contest, given the following conditions:
  * There were 52 total questions
  * Drew got 20 questions correct
  * Drew got 6 questions wrong
  * Carla got twice as many questions wrong as Drew
Prove that Carla got 40 questions correct. -/
theorem spelling_contest_result (total_questions : Nat) (drew_correct : Nat) (drew_wrong : Nat) (carla_wrong_multiplier : Nat) :
  total_questions = 52 →
  drew_correct = 20 →
  drew_wrong = 6 →
  carla_wrong_multiplier = 2 →
  total_questions - (carla_wrong_multiplier * drew_wrong) = 40 := by
  sorry

#check spelling_contest_result

end NUMINAMATH_CALUDE_spelling_contest_result_l3320_332008


namespace NUMINAMATH_CALUDE_common_roots_product_l3320_332078

/-- Given two cubic equations with two common roots, prove that the product of these common roots is 10√[3]{2} -/
theorem common_roots_product (C D : ℝ) : 
  ∃ (u v w t : ℝ),
    (u^3 + C*u + 20 = 0) ∧ 
    (v^3 + C*v + 20 = 0) ∧ 
    (w^3 + C*w + 20 = 0) ∧
    (u^3 + D*u^2 + 100 = 0) ∧ 
    (v^3 + D*v^2 + 100 = 0) ∧ 
    (t^3 + D*t^2 + 100 = 0) ∧
    (u ≠ v) ∧ 
    (u * v = 10 * Real.rpow 2 (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_common_roots_product_l3320_332078


namespace NUMINAMATH_CALUDE_two_digit_square_l3320_332053

/-- Given distinct digits a, b, c, prove that the two-digit number 'ab' is 21 -/
theorem two_digit_square (a b c : ℕ) : 
  a ≠ b → b ≠ c → a ≠ c →
  b = 1 →
  10 * a + b < 100 →
  100 * c + 10 * c + b > 300 →
  (10 * a + b)^2 = 100 * c + 10 * c + b →
  10 * a + b = 21 := by
sorry

end NUMINAMATH_CALUDE_two_digit_square_l3320_332053


namespace NUMINAMATH_CALUDE_day_one_fish_count_l3320_332063

/-- The number of fish counted on day one -/
def day_one_count : ℕ := sorry

/-- The percentage of fish that are sharks -/
def shark_percentage : ℚ := 1/4

/-- The total number of sharks counted over two days -/
def total_sharks : ℕ := 15

theorem day_one_fish_count : 
  day_one_count = 15 :=
by
  have h1 : shark_percentage * (day_one_count + 3 * day_one_count) = total_sharks := sorry
  sorry


end NUMINAMATH_CALUDE_day_one_fish_count_l3320_332063


namespace NUMINAMATH_CALUDE_equal_spacing_theorem_l3320_332095

/-- The width of the wall in millimeters -/
def wall_width : ℕ := 4800

/-- The width of each picture in millimeters -/
def picture_width : ℕ := 420

/-- The number of pictures -/
def num_pictures : ℕ := 4

/-- The distance from the center of each middle picture to the center of the wall -/
def middle_picture_distance : ℕ := 730

/-- Theorem stating that the distance from the center of each middle picture
    to the center of the wall is 730 mm when all pictures are equally spaced -/
theorem equal_spacing_theorem :
  let total_space := wall_width - picture_width
  let spacing := total_space / (num_pictures - 1)
  spacing / 2 = middle_picture_distance := by sorry

end NUMINAMATH_CALUDE_equal_spacing_theorem_l3320_332095


namespace NUMINAMATH_CALUDE_max_k_value_l3320_332043

noncomputable def f (x : ℝ) := x + x * Real.log x

theorem max_k_value (k : ℤ) :
  (∀ x > 2, k * (x - 2) < f x) → k ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l3320_332043


namespace NUMINAMATH_CALUDE_container_volume_ratio_l3320_332090

theorem container_volume_ratio (A B : ℝ) (h1 : A > 0) (h2 : B > 0) 
  (h3 : 2/3 * A = 5/8 * B) : A / B = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l3320_332090


namespace NUMINAMATH_CALUDE_min_sum_of_coefficients_l3320_332069

theorem min_sum_of_coefficients (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a * 2 + b * 3 = a * b) → (∀ x y : ℝ, a * x + b * y = a * b → a + b ≥ 5 + 2 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_coefficients_l3320_332069


namespace NUMINAMATH_CALUDE_distance_to_larger_section_l3320_332088

/-- Given a right hexagonal pyramid with two parallel cross sections -/
structure HexagonalPyramid where
  /-- Ratio of areas of two parallel cross sections -/
  area_ratio : ℝ
  /-- Distance between the two parallel cross sections -/
  distance_between_sections : ℝ

/-- Theorem stating the distance from apex to larger cross section -/
theorem distance_to_larger_section (pyramid : HexagonalPyramid)
  (h_area_ratio : pyramid.area_ratio = 4 / 9)
  (h_distance : pyramid.distance_between_sections = 12) :
  ∃ (d : ℝ), d = 36 ∧ d > 0 ∧ 
  d = (pyramid.distance_between_sections * 3) / (1 - (pyramid.area_ratio)^(1/2)) :=
sorry

end NUMINAMATH_CALUDE_distance_to_larger_section_l3320_332088


namespace NUMINAMATH_CALUDE_isabella_currency_exchange_l3320_332022

theorem isabella_currency_exchange :
  ∃ d : ℕ+, 
    (10 : ℚ) / 7 * d.val - 60 = d.val ∧ 
    (d.val / 100 + (d.val / 10) % 10 + d.val % 10 = 5) := by
  sorry

end NUMINAMATH_CALUDE_isabella_currency_exchange_l3320_332022


namespace NUMINAMATH_CALUDE_count_valid_numbers_l3320_332036

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 10 % 10 : ℚ) = ((n / 100) + (n % 10)) / 2 ∧
  n % 10 = 2 * (n / 100)

theorem count_valid_numbers : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_valid_number n) ∧ S.card = 2 :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l3320_332036


namespace NUMINAMATH_CALUDE_line_minimum_reciprocal_sum_l3320_332062

theorem line_minimum_reciprocal_sum (m n : ℝ) (h1 : m * n > 0) (h2 : m + n = 2) :
  ∀ (x y : ℝ), x * n + y * m = 2 → x = 1 ∧ y = 1 →
  (1 / m + 1 / n) ≥ 2 ∧ ∃ (m₀ n₀ : ℝ), m₀ * n₀ > 0 ∧ m₀ + n₀ = 2 ∧ 1 / m₀ + 1 / n₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_minimum_reciprocal_sum_l3320_332062


namespace NUMINAMATH_CALUDE_five_rows_with_seven_students_l3320_332085

/-- Represents the seating arrangement in a classroom -/
structure Seating :=
  (rows_with_7 : ℕ)
  (rows_with_6 : ℕ)

/-- Checks if a seating arrangement is valid -/
def is_valid_seating (s : Seating) : Prop :=
  s.rows_with_7 * 7 + s.rows_with_6 * 6 = 53

/-- The theorem stating that there are 5 rows with 7 students -/
theorem five_rows_with_seven_students :
  ∃ (s : Seating), is_valid_seating s ∧ s.rows_with_7 = 5 :=
sorry

end NUMINAMATH_CALUDE_five_rows_with_seven_students_l3320_332085


namespace NUMINAMATH_CALUDE_fraction_expression_equality_l3320_332098

theorem fraction_expression_equality : (3/7 + 4/5) / (5/11 + 2/3) = 1419/1295 := by
  sorry

end NUMINAMATH_CALUDE_fraction_expression_equality_l3320_332098


namespace NUMINAMATH_CALUDE_four_digit_number_problem_l3320_332013

theorem four_digit_number_problem :
  ∀ N : ℕ,
  (1000 ≤ N) ∧ (N < 10000) →
  (∃ x y : ℕ,
    (1 ≤ x) ∧ (x ≤ 9) ∧
    (100 ≤ y) ∧ (y < 1000) ∧
    (N = 1000 * x + y) ∧
    (N / y = 3) ∧
    (N % y = 8)) →
  (N = 1496 ∨ N = 2996) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_problem_l3320_332013


namespace NUMINAMATH_CALUDE_tangent_length_to_given_circle_l3320_332038

/-- The circle passing through three given points -/
structure Circle where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- The length of the tangent segment from a point to a circle -/
def tangentLength (p : ℝ × ℝ) (c : Circle) : ℝ := sorry

/-- The origin point (0,0) -/
def origin : ℝ × ℝ := (0, 0)

/-- The circle passing through (4,3), (8,6), and (9,12) -/
def givenCircle : Circle :=
  { point1 := (4, 3)
    point2 := (8, 6)
    point3 := (9, 12) }

theorem tangent_length_to_given_circle :
  tangentLength origin givenCircle = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_length_to_given_circle_l3320_332038


namespace NUMINAMATH_CALUDE_annies_crayons_l3320_332076

/-- Annie's crayon problem -/
theorem annies_crayons (initial : ℕ) (additional : ℕ) : 
  initial = 4 → additional = 36 → initial + additional = 40 := by
  sorry

end NUMINAMATH_CALUDE_annies_crayons_l3320_332076


namespace NUMINAMATH_CALUDE_selection_methods_count_l3320_332020

-- Define the total number of students
def total_students : ℕ := 5

-- Define the number of students needed for each role
def translation_students : ℕ := 2
def transportation_students : ℕ := 1
def protocol_students : ℕ := 1

-- Define the total number of students to be selected
def selected_students : ℕ := translation_students + transportation_students + protocol_students

-- The theorem to be proved
theorem selection_methods_count : 
  (Nat.choose total_students translation_students) * 
  (Nat.choose (total_students - translation_students) transportation_students) * 
  (Nat.choose (total_students - translation_students - transportation_students) protocol_students) = 60 := by
  sorry


end NUMINAMATH_CALUDE_selection_methods_count_l3320_332020


namespace NUMINAMATH_CALUDE_combination_count_l3320_332077

/-- The number of different styles of backpacks -/
def num_backpacks : ℕ := 2

/-- The number of different styles of pencil cases -/
def num_pencil_cases : ℕ := 2

/-- A combination consists of one backpack and one pencil case -/
def combination := ℕ × ℕ

/-- The total number of possible combinations -/
def total_combinations : ℕ := num_backpacks * num_pencil_cases

theorem combination_count : total_combinations = 4 := by sorry

end NUMINAMATH_CALUDE_combination_count_l3320_332077


namespace NUMINAMATH_CALUDE_expression_evaluation_l3320_332064

theorem expression_evaluation : (25 * 5 + 5^2) / (5^2 - 15) = 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3320_332064


namespace NUMINAMATH_CALUDE_complementary_angle_adjustment_l3320_332074

theorem complementary_angle_adjustment (a b : ℝ) (h1 : a + b = 90) (h2 : a / b = 1 / 2) :
  let a' := a * 1.2
  let b' := 90 - a'
  (b - b') / b = 0.1 := by sorry

end NUMINAMATH_CALUDE_complementary_angle_adjustment_l3320_332074


namespace NUMINAMATH_CALUDE_simplify_fraction_l3320_332033

theorem simplify_fraction : (84 : ℚ) / 144 = 7 / 12 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3320_332033


namespace NUMINAMATH_CALUDE_max_value_constrained_product_l3320_332096

theorem max_value_constrained_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5*x + 3*y < 75) :
  x * y * (75 - 5*x - 3*y) ≤ 3125/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_constrained_product_l3320_332096


namespace NUMINAMATH_CALUDE_square_area_not_possible_l3320_332003

-- Define the points
def P : ℝ × ℝ := (1, 0)
def Q : ℝ × ℝ := (2, 0)
def R : ℝ × ℝ := (4, 0)
def S : ℝ × ℝ := (8, 0)

-- Define a predicate for four lines forming a square
def forms_square (l₁ l₂ l₃ l₄ : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (side : ℝ),
    side > 0 ∧
    (∀ p ∈ l₁ ∪ l₂ ∪ l₃ ∪ l₄, ∃ i j : ℤ, (p.1 - center.1)^2 + (p.2 - center.2)^2 = 2 * side^2 * (i^2 + j^2)) ∧
    (P ∈ l₁ ∪ l₂ ∪ l₃ ∪ l₄) ∧
    (Q ∈ l₁ ∪ l₂ ∪ l₃ ∪ l₄) ∧
    (R ∈ l₁ ∪ l₂ ∪ l₃ ∪ l₄) ∧
    (S ∈ l₁ ∪ l₂ ∪ l₃ ∪ l₄)

-- The theorem to prove
theorem square_area_not_possible :
  ∀ l₁ l₂ l₃ l₄ : Set (ℝ × ℝ),
  forms_square l₁ l₂ l₃ l₄ →
  ∀ side : ℝ, side^2 ≠ 26/5 :=
by sorry

end NUMINAMATH_CALUDE_square_area_not_possible_l3320_332003


namespace NUMINAMATH_CALUDE_skew_quadrilateral_angle_sum_less_than_360_l3320_332081

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The angle between three points in 3D space -/
noncomputable def angle (A B C : Point3D) : ℝ := sorry

/-- Four points are non-coplanar if they do not lie in the same plane -/
def nonCoplanar (A B C D : Point3D) : Prop := sorry

/-- A skew quadrilateral is formed by four non-coplanar points -/
structure SkewQuadrilateral where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  nonCoplanar : nonCoplanar A B C D

theorem skew_quadrilateral_angle_sum_less_than_360 (quad : SkewQuadrilateral) :
  angle quad.A quad.B quad.C + angle quad.B quad.C quad.D +
  angle quad.C quad.D quad.A + angle quad.D quad.A quad.B < 2 * π :=
sorry

end NUMINAMATH_CALUDE_skew_quadrilateral_angle_sum_less_than_360_l3320_332081


namespace NUMINAMATH_CALUDE_sequence_periodicity_l3320_332018

def sequence_rule (a : ℕ) (u : ℕ → ℕ) : Prop :=
  ∀ n, (Even (u n) → u (n + 1) = (u n) / 2) ∧
       (Odd (u n) → u (n + 1) = a + u n)

theorem sequence_periodicity (a : ℕ) (u : ℕ → ℕ) 
  (h1 : Odd a) 
  (h2 : a > 0) 
  (h3 : sequence_rule a u) :
  ∃ k : ℕ, ∃ p : ℕ, p > 0 ∧ ∀ n ≥ k, u (n + p) = u n :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l3320_332018


namespace NUMINAMATH_CALUDE_symmetry_and_monotonicity_l3320_332009

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def symmetric_about_one (f : ℝ → ℝ) : Prop := ∀ x, f (2 - x) = f x

def increasing_on_zero_one (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < 1 ∧ 0 < x₂ ∧ x₂ < 1 → (x₁ - x₂) * (f x₁ - f x₂) > 0

theorem symmetry_and_monotonicity
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_sym : symmetric_about_one f)
  (h_inc : increasing_on_zero_one f) :
  (∀ x, f (4 - x) + f x = 0) ∧
  (∀ x, 2 < x ∧ x < 3 → ∀ y, 2 < y ∧ y < 3 ∧ x < y → f y < f x) :=
sorry

end NUMINAMATH_CALUDE_symmetry_and_monotonicity_l3320_332009


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l3320_332049

theorem polynomial_identity_sum_of_squares : 
  ∀ (p q r s t u : ℤ), 
  (∀ x : ℝ, 729 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l3320_332049


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3320_332072

/-- The equation of a potential hyperbola with parameter k -/
def hyperbola_equation (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k - 3) - y^2 / (k + 3) = 1

/-- Predicate to check if an equation represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y, hyperbola_equation k x y ∧ (k - 3) * (k + 3) > 0

/-- Statement: k > 3 is a sufficient but not necessary condition for the equation to represent a hyperbola -/
theorem sufficient_not_necessary_condition :
  (∀ k : ℝ, k > 3 → is_hyperbola k) ∧
  ¬(∀ k : ℝ, is_hyperbola k → k > 3) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3320_332072


namespace NUMINAMATH_CALUDE_intersection_points_form_parallelogram_l3320_332034

-- Define the circle type
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the intersection points
def M : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry
def D : ℝ × ℝ := sorry

-- Define the four circles
def circle1 : Circle := sorry
def circle2 : Circle := sorry
def circle3 : Circle := sorry
def circle4 : Circle := sorry

-- Define the properties of the circles' intersections
def three_circles_intersect (c1 c2 c3 : Circle) : Prop :=
  ∃ p : ℝ × ℝ, (p = M ∨ p = N) ∧ 
    (‖p - c1.center‖ = c1.radius) ∧
    (‖p - c2.center‖ = c2.radius) ∧
    (‖p - c3.center‖ = c3.radius)

def two_circles_intersect (c1 c2 : Circle) : Prop :=
  ∃ p : ℝ × ℝ, (p = A ∨ p = B ∨ p = C ∨ p = D) ∧
    (‖p - c1.center‖ = c1.radius) ∧
    (‖p - c2.center‖ = c2.radius)

-- Theorem statement
theorem intersection_points_form_parallelogram
  (h1 : circle1.radius = circle2.radius ∧ circle2.radius = circle3.radius ∧ circle3.radius = circle4.radius)
  (h2 : three_circles_intersect circle1 circle2 circle3 ∧
        three_circles_intersect circle1 circle2 circle4 ∧
        three_circles_intersect circle1 circle3 circle4 ∧
        three_circles_intersect circle2 circle3 circle4)
  (h3 : two_circles_intersect circle1 circle2 ∧
        two_circles_intersect circle1 circle3 ∧
        two_circles_intersect circle1 circle4 ∧
        two_circles_intersect circle2 circle3 ∧
        two_circles_intersect circle2 circle4 ∧
        two_circles_intersect circle3 circle4) :
  C - D = B - A :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_form_parallelogram_l3320_332034


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3320_332046

theorem intersection_of_lines : ∃! p : ℚ × ℚ, 
  8 * p.1 - 5 * p.2 = 20 ∧ 6 * p.1 + 2 * p.2 = 18 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3320_332046


namespace NUMINAMATH_CALUDE_max_elements_l3320_332073

structure RelationSystem where
  S : Type
  rel : S → S → Prop
  distinct_relation : ∀ a b : S, a ≠ b → (rel a b ∨ rel b a) ∧ ¬(rel a b ∧ rel b a)
  transitivity : ∀ a b c : S, a ≠ b → b ≠ c → a ≠ c → rel a b → rel b c → rel c a

theorem max_elements (R : RelationSystem) : 
  ∃ (n : ℕ), ∀ (m : ℕ), (∃ (f : Fin m → R.S), Function.Injective f) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_elements_l3320_332073


namespace NUMINAMATH_CALUDE_distance_P_to_x_axis_l3320_332048

/-- The distance from a point to the x-axis in a Cartesian coordinate system --/
def distance_to_x_axis (y : ℝ) : ℝ := |y|

/-- Point P in the Cartesian coordinate system --/
def P : ℝ × ℝ := (2, -3)

/-- Theorem: The distance from point P(2, -3) to the x-axis is 3 --/
theorem distance_P_to_x_axis :
  distance_to_x_axis P.2 = 3 := by sorry

end NUMINAMATH_CALUDE_distance_P_to_x_axis_l3320_332048


namespace NUMINAMATH_CALUDE_adam_has_14_apples_l3320_332065

-- Define the number of apples Jackie has
def jackie_apples : ℕ := 9

-- Define Adam's apples in relation to Jackie's
def adam_apples : ℕ := jackie_apples + 5

-- Theorem statement
theorem adam_has_14_apples : adam_apples = 14 := by
  sorry

end NUMINAMATH_CALUDE_adam_has_14_apples_l3320_332065


namespace NUMINAMATH_CALUDE_lewis_money_at_end_of_harvest_l3320_332004

/-- Calculates the money Lewis will have at the end of the harvest season -/
def money_at_end_of_harvest (weekly_earnings : ℕ) (weekly_rent : ℕ) (num_weeks : ℕ) : ℕ :=
  (weekly_earnings - weekly_rent) * num_weeks

/-- Proves that Lewis will have $325175 at the end of the harvest season -/
theorem lewis_money_at_end_of_harvest :
  money_at_end_of_harvest 491 216 1181 = 325175 := by
  sorry

end NUMINAMATH_CALUDE_lewis_money_at_end_of_harvest_l3320_332004


namespace NUMINAMATH_CALUDE_range_of_3x_minus_2y_l3320_332006

theorem range_of_3x_minus_2y (x y : ℝ) 
  (h1 : -1 ≤ x + y ∧ x + y ≤ 1) 
  (h2 : 1 ≤ x - y ∧ x - y ≤ 5) : 
  ∃ (z : ℝ), 2 ≤ z ∧ z ≤ 13 ∧ z = 3*x - 2*y ∧ 
  ∀ (w : ℝ), w = 3*x - 2*y → 2 ≤ w ∧ w ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_range_of_3x_minus_2y_l3320_332006


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l3320_332012

theorem matrix_equation_proof : 
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 0.5, 1]
  M^3 - 3 • M^2 + 4 • M = !![7, 14; 3.5, 7] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l3320_332012


namespace NUMINAMATH_CALUDE_fraction_non_negative_l3320_332011

theorem fraction_non_negative (x : ℝ) : (x + 7) / (x^2 + 2*x + 8) ≥ 0 ↔ x ≥ -7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_non_negative_l3320_332011


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3320_332050

-- Define the quadratic function
def f (x : ℝ) := x^2 + 4*x - 5

-- Define the solution set
def solution_set : Set ℝ := {x | x < -5 ∨ x > 1}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = solution_set :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3320_332050


namespace NUMINAMATH_CALUDE_nested_sqrt_equality_l3320_332021

theorem nested_sqrt_equality : Real.sqrt (64 * Real.sqrt (32 * Real.sqrt 16)) = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_equality_l3320_332021


namespace NUMINAMATH_CALUDE_equation_solution_l3320_332097

theorem equation_solution : ∃! x : ℝ, 2 * x + 1 = x - 1 ∧ x = -2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3320_332097


namespace NUMINAMATH_CALUDE_consecutive_non_prime_powers_l3320_332027

theorem consecutive_non_prime_powers (n : ℕ+) :
  ∃ x : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ¬∃ (p : ℕ) (k : ℕ), Prime p ∧ (x + i = p^k) :=
sorry

end NUMINAMATH_CALUDE_consecutive_non_prime_powers_l3320_332027


namespace NUMINAMATH_CALUDE_exponential_inequality_l3320_332068

theorem exponential_inequality (m n : ℝ) (a b : ℝ) 
  (h1 : a = (0.2 : ℝ) ^ m) 
  (h2 : b = (0.2 : ℝ) ^ n) 
  (h3 : m > n) : 
  a < b := by
sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3320_332068


namespace NUMINAMATH_CALUDE_distribute_5_3_l3320_332023

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    where each box must contain at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 150 ways to distribute 5 distinct objects into 3 distinct boxes,
    where each box must contain at least one object. -/
theorem distribute_5_3 : distribute 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l3320_332023


namespace NUMINAMATH_CALUDE_f_min_max_l3320_332094

def f (x : ℝ) : ℝ := 2 * x^2 - 6 * x + 1

theorem f_min_max :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≥ -3) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = -3) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ 9) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = 9) :=
by sorry

end NUMINAMATH_CALUDE_f_min_max_l3320_332094


namespace NUMINAMATH_CALUDE_rain_received_calculation_l3320_332092

/-- The number of days in a year -/
def daysInYear : ℕ := 365

/-- The normal average daily rainfall in inches -/
def normalDailyRainfall : ℚ := 2

/-- The number of days left in the year -/
def daysLeft : ℕ := 100

/-- The required average daily rainfall for the remaining days, in inches -/
def requiredDailyRainfall : ℚ := 3

/-- The amount of rain received so far this year, in inches -/
def rainReceivedSoFar : ℚ := 430

theorem rain_received_calculation :
  rainReceivedSoFar = 
    normalDailyRainfall * daysInYear - requiredDailyRainfall * daysLeft :=
by sorry

end NUMINAMATH_CALUDE_rain_received_calculation_l3320_332092


namespace NUMINAMATH_CALUDE_prob_no_defective_bulbs_l3320_332070

/-- The probability of selecting 4 non-defective bulbs out of 10 bulbs, where 4 are defective -/
theorem prob_no_defective_bulbs (total : ℕ) (defective : ℕ) (select : ℕ) :
  total = 10 →
  defective = 4 →
  select = 4 →
  (Nat.choose (total - defective) select : ℚ) / (Nat.choose total select : ℚ) = 1 / 14 := by
  sorry

end NUMINAMATH_CALUDE_prob_no_defective_bulbs_l3320_332070
