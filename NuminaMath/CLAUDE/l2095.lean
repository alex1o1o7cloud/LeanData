import Mathlib

namespace point_value_theorem_l2095_209536

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Represents the number line -/
structure NumberLine where
  origin : Point
  pointA : Point
  pointB : Point
  pointC : Point

def NumberLine.sameSide (nl : NumberLine) : Prop :=
  (nl.pointA.value - nl.origin.value) * (nl.pointB.value - nl.origin.value) > 0

theorem point_value_theorem (nl : NumberLine) 
  (h1 : nl.sameSide)
  (h2 : nl.pointB.value = 1)
  (h3 : nl.pointC.value = nl.pointA.value - 3)
  (h4 : nl.pointC.value = -nl.pointB.value) :
  nl.pointA.value = 2 := by
  sorry

end point_value_theorem_l2095_209536


namespace predicted_y_value_l2095_209542

-- Define the linear regression equation
def linear_regression (x : ℝ) (a : ℝ) : ℝ := -0.7 * x + a

-- Define the mean values
def x_mean : ℝ := 1
def y_mean : ℝ := 0.3

-- Theorem statement
theorem predicted_y_value :
  ∃ (a : ℝ), 
    (linear_regression x_mean a = y_mean) ∧ 
    (linear_regression 2 a = -0.4) := by
  sorry

end predicted_y_value_l2095_209542


namespace green_ball_probability_l2095_209537

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The probability of selecting a green ball given the problem conditions -/
theorem green_ball_probability :
  let containerX : Container := ⟨5, 5⟩
  let containerY : Container := ⟨8, 2⟩
  let containerZ : Container := ⟨3, 7⟩
  let totalContainers : ℕ := 3
  (1 : ℚ) / totalContainers * (greenProbability containerX +
                               greenProbability containerY +
                               greenProbability containerZ) = 7 / 15 := by
  sorry


end green_ball_probability_l2095_209537


namespace cat_shelter_ratio_l2095_209510

theorem cat_shelter_ratio : 
  ∀ (initial_cats replacement_cats adopted_cats dogs : ℕ),
    initial_cats = 15 →
    adopted_cats = initial_cats / 3 →
    initial_cats = initial_cats - adopted_cats + replacement_cats →
    dogs = 2 * initial_cats →
    initial_cats + dogs + replacement_cats = 60 →
    replacement_cats / adopted_cats = 3 ∧ adopted_cats ≠ 0 :=
by sorry

end cat_shelter_ratio_l2095_209510


namespace probability_red_ball_l2095_209590

/-- The probability of drawing a red ball from a box with red and black balls -/
theorem probability_red_ball (red_balls black_balls : ℕ) :
  red_balls > 0 →
  black_balls ≥ 0 →
  (red_balls : ℚ) / (red_balls + black_balls : ℚ) = 7 / 10 ↔
  red_balls = 7 ∧ black_balls = 3 :=
by sorry

end probability_red_ball_l2095_209590


namespace range_of_a_for_always_positive_quadratic_l2095_209567

theorem range_of_a_for_always_positive_quadratic :
  (∀ a : ℝ, (∀ x : ℝ, x^2 + 2*a*x + a > 0) ↔ (0 < a ∧ a < 1)) := by sorry

end range_of_a_for_always_positive_quadratic_l2095_209567


namespace ordering_abc_l2095_209555

theorem ordering_abc (a b c : ℝ) (ha : a = 1.01^(1/2 : ℝ)) (hb : b = 1.01^(3/5 : ℝ)) (hc : c = 0.6^(1/2 : ℝ)) : b > a ∧ a > c := by
  sorry

end ordering_abc_l2095_209555


namespace correct_sampling_methods_l2095_209573

-- Define the type for sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic
  | DrawingLots
  | RandomNumber

-- Define the set of correct sampling methods
def correctSamplingMethods : Set SamplingMethod :=
  {SamplingMethod.SimpleRandom, SamplingMethod.Stratified, SamplingMethod.Systematic}

-- Define the property of being a valid sampling method
def isValidSamplingMethod (method : SamplingMethod) : Prop :=
  method ∈ correctSamplingMethods

-- State the conditions
axiom simple_random_valid : isValidSamplingMethod SamplingMethod.SimpleRandom
axiom stratified_valid : isValidSamplingMethod SamplingMethod.Stratified
axiom systematic_valid : isValidSamplingMethod SamplingMethod.Systematic
axiom drawing_lots_is_simple_random : SamplingMethod.DrawingLots = SamplingMethod.SimpleRandom
axiom random_number_is_simple_random : SamplingMethod.RandomNumber = SamplingMethod.SimpleRandom

-- State the theorem
theorem correct_sampling_methods :
  correctSamplingMethods = {SamplingMethod.SimpleRandom, SamplingMethod.Stratified, SamplingMethod.Systematic} :=
by sorry

end correct_sampling_methods_l2095_209573


namespace graduate_ratio_l2095_209578

theorem graduate_ratio (N : ℝ) (G : ℝ) (C : ℝ) 
  (h1 : C = (2/3) * N) 
  (h2 : G / (G + C) = 0.15789473684210525) : 
  G = (1/8) * N := by
sorry

end graduate_ratio_l2095_209578


namespace factorization_equality_l2095_209585

theorem factorization_equality (x y : ℝ) : -2*x^2 + 4*x*y - 2*y^2 = -2*(x-y)^2 := by
  sorry

end factorization_equality_l2095_209585


namespace quadratic_coefficient_l2095_209522

/-- Given a quadratic function y = ax² + bx + c, if the points (2, y₁) and (-2, y₂) lie on the curve
    and y₁ - y₂ = 4, then b = 1 -/
theorem quadratic_coefficient (a b c y₁ y₂ : ℝ) : 
  y₁ = a * 4 + b * 2 + c →
  y₂ = a * 4 - b * 2 + c →
  y₁ - y₂ = 4 →
  b = 1 := by
  sorry


end quadratic_coefficient_l2095_209522


namespace point_on_x_axis_point_in_second_quadrant_equal_distance_l2095_209533

-- Define the point P as a function of a
def P (a : ℝ) : ℝ × ℝ := (2*a - 3, a + 6)

-- Part 1
theorem point_on_x_axis (a : ℝ) : 
  P a = (-15, 0) ↔ (P a).2 = 0 :=
sorry

-- Part 2
theorem point_in_second_quadrant_equal_distance (a : ℝ) :
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ abs (P a).1 = (P a).2 → a^2003 + 2024 = 2023 :=
sorry

end point_on_x_axis_point_in_second_quadrant_equal_distance_l2095_209533


namespace excircle_lengths_sum_gt_semiperimeter_l2095_209523

/-- Given a triangle with sides a, b, and c, and semi-perimeter p,
    BB' and CC' are specific lengths related to the excircles of the triangle. -/
def triangle_excircle_lengths (a b c : ℝ) (p : ℝ) (BB' CC' : ℝ) : Prop :=
  p = (a + b + c) / 2 ∧ BB' = p - a ∧ CC' = p - b

/-- The sum of BB' and CC' is greater than the semi-perimeter p for any triangle. -/
theorem excircle_lengths_sum_gt_semiperimeter 
  {a b c p BB' CC' : ℝ} 
  (h : triangle_excircle_lengths a b c p BB' CC') :
  BB' + CC' > p :=
sorry

end excircle_lengths_sum_gt_semiperimeter_l2095_209523


namespace arithmetic_progression_with_small_prime_factors_l2095_209502

/-- The greatest prime factor of a positive integer n > 1 -/
noncomputable def greatestPrimeFactor (n : ℕ) : ℕ := sorry

/-- Check if three numbers form an arithmetic progression -/
def isArithmeticProgression (x y z : ℕ) : Prop :=
  y - x = z - y

/-- Main theorem -/
theorem arithmetic_progression_with_small_prime_factors
  (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hdistinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (hap : isArithmeticProgression x y z)
  (hprime : greatestPrimeFactor (x * y * z) ≤ 3) :
  ∃ (l a b : ℕ), (a ≥ 0 ∧ b ≥ 0) ∧ l = 2^a * 3^b ∧
    ((x, y, z) = (l, 2*l, 3*l) ∨
     (x, y, z) = (2*l, 3*l, 4*l) ∨
     (x, y, z) = (2*l, 9*l, 16*l)) :=
sorry

end arithmetic_progression_with_small_prime_factors_l2095_209502


namespace seventy_million_scientific_notation_l2095_209596

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem seventy_million_scientific_notation :
  toScientificNotation 70000000 = ScientificNotation.mk 7.0 7 (by sorry) :=
sorry

end seventy_million_scientific_notation_l2095_209596


namespace sqrt_equation_solution_l2095_209576

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (x^2 + 16) = 12 ↔ x = 8 * Real.sqrt 2 ∨ x = -8 * Real.sqrt 2 := by
  sorry

end sqrt_equation_solution_l2095_209576


namespace goat_grazing_area_l2095_209562

/-- The side length of a square plot given a goat tied to one corner -/
theorem goat_grazing_area (rope_length : ℝ) (graze_area : ℝ) (side_length : ℝ) : 
  rope_length = 7 →
  graze_area = 38.48451000647496 →
  side_length = 7 →
  (1 / 4) * Real.pi * rope_length ^ 2 = graze_area →
  side_length = rope_length :=
by sorry

end goat_grazing_area_l2095_209562


namespace ratio_common_value_l2095_209511

theorem ratio_common_value (x y z : ℝ) (h : (x + y) / z = (x + z) / y ∧ (x + z) / y = (y + z) / x) :
  (x + y) / z = -1 ∨ (x + y) / z = 2 :=
sorry

end ratio_common_value_l2095_209511


namespace other_bill_value_l2095_209580

theorem other_bill_value (total_bills : ℕ) (total_value : ℕ) (five_dollar_bills : ℕ) :
  total_bills = 12 →
  total_value = 100 →
  five_dollar_bills = 4 →
  ∃ other_value : ℕ, 
    other_value * (total_bills - five_dollar_bills) + 5 * five_dollar_bills = total_value ∧
    other_value = 10 :=
by sorry

end other_bill_value_l2095_209580


namespace max_y_value_l2095_209538

theorem max_y_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_eq : x * y = (x - y) / (x + 3 * y)) : 
  y ≤ 1 / 3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ = 1 / 3 ∧ x₀ * y₀ = (x₀ - y₀) / (x₀ + 3 * y₀) :=
by sorry

end max_y_value_l2095_209538


namespace factorial_plus_24_equals_square_l2095_209548

theorem factorial_plus_24_equals_square (n m : ℕ) : n.factorial + 24 = m ^ 2 ↔ (n = 1 ∧ m = 5) ∨ (n = 5 ∧ m = 12) := by
  sorry

end factorial_plus_24_equals_square_l2095_209548


namespace triangle_angle_calculation_l2095_209579

theorem triangle_angle_calculation (a b c : ℝ) (A B C : ℝ) :
  a = 3 →
  b = Real.sqrt 6 →
  A = 2 * π / 3 →
  (a / Real.sin A = b / Real.sin B) →
  B = π / 4 :=
sorry

end triangle_angle_calculation_l2095_209579


namespace diophantine_equation_solution_l2095_209531

theorem diophantine_equation_solution :
  ∀ x y z : ℕ+,
    (2 * (x + y + z + 2 * x * y * z)^2 = (2 * x * y + 2 * y * z + 2 * z * x + 1)^2 + 2023) ↔
    ((x = 3 ∧ y = 3 ∧ z = 2) ∨
     (x = 3 ∧ y = 2 ∧ z = 3) ∨
     (x = 2 ∧ y = 3 ∧ z = 3)) :=
by sorry

end diophantine_equation_solution_l2095_209531


namespace harmonic_sum_identity_l2095_209592

def h (n : ℕ) : ℚ :=
  (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

theorem harmonic_sum_identity (n : ℕ) (hn : n ≥ 2) :
  n + (Finset.range (n - 1)).sum h = n * h n := by
  sorry

end harmonic_sum_identity_l2095_209592


namespace richards_walking_ratio_l2095_209561

/-- Proves that the ratio of Richard's second day walking distance to his first day walking distance is 1/5 --/
theorem richards_walking_ratio :
  let total_distance : ℝ := 70
  let first_day : ℝ := 20
  let third_day : ℝ := 10
  let remaining : ℝ := 36
  let second_day := total_distance - remaining - first_day - third_day
  second_day / first_day = 1 / 5 := by
sorry

end richards_walking_ratio_l2095_209561


namespace probability_of_two_specific_stamps_l2095_209527

/-- Represents a set of four distinct stamps -/
def Stamps : Type := Fin 4

/-- The number of ways to choose 2 stamps from 4 stamps -/
def total_combinations : ℕ := Nat.choose 4 2

/-- The number of ways to choose the specific 2 stamps we want -/
def favorable_combinations : ℕ := 1

/-- Theorem: The probability of drawing exactly two specific stamps from a set of four stamps is 1/6 -/
theorem probability_of_two_specific_stamps : 
  (favorable_combinations : ℚ) / total_combinations = 1 / 6 := by
  sorry


end probability_of_two_specific_stamps_l2095_209527


namespace no_digit_move_multiplier_l2095_209543

theorem no_digit_move_multiplier : ¬∃ (N : ℕ), 
  ∃ (d : ℕ) (M : ℕ) (k : ℕ),
    (N = d * 10^k + M) ∧ 
    (d ≥ 1) ∧ (d ≤ 9) ∧ 
    (10 * M + d = 5 * N ∨ 10 * M + d = 6 * N ∨ 10 * M + d = 8 * N) := by
  sorry

end no_digit_move_multiplier_l2095_209543


namespace unique_solution_cubic_system_l2095_209528

theorem unique_solution_cubic_system (x y z : ℝ) :
  x^3 + y = z^2 ∧ y^3 + z = x^2 ∧ z^3 + x = y^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end unique_solution_cubic_system_l2095_209528


namespace expense_difference_zero_l2095_209521

def vacation_expenses (anne_paid beth_paid carlos_paid : ℕ) (a b : ℕ) : Prop :=
  let total := anne_paid + beth_paid + carlos_paid
  let share := total / 3
  (anne_paid + b = share + a) ∧
  (beth_paid = share + b) ∧
  (carlos_paid + a = share)

theorem expense_difference_zero 
  (anne_paid beth_paid carlos_paid : ℕ) 
  (a b : ℕ) 
  (h : vacation_expenses anne_paid beth_paid carlos_paid a b) :
  a - b = 0 :=
sorry

end expense_difference_zero_l2095_209521


namespace inequality_solution_set_l2095_209504

theorem inequality_solution_set : 
  {x : ℝ | (x^2 + x^3 - 3*x^4) / (x + x^2 - 3*x^3) ≥ -1} = 
  {x : ℝ | x ∈ Set.Icc (-1) (((-1 - Real.sqrt 13) / 6 : ℝ)) ∪ 
           Set.Ioo (((-1 - Real.sqrt 13) / 6 : ℝ)) (((-1 + Real.sqrt 13) / 6 : ℝ)) ∪
           Set.Ioo (((-1 + Real.sqrt 13) / 6 : ℝ)) 0 ∪
           Set.Ioi 0} :=
by sorry

end inequality_solution_set_l2095_209504


namespace triangle_equation_l2095_209518

theorem triangle_equation (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let angleA : ℝ := 60 * π / 180
  (a^2 = b^2 + c^2 - 2*b*c*(angleA.cos)) →
  (3 / (a + b + c) = 1 / (a + b) + 1 / (a + c)) :=
by sorry

end triangle_equation_l2095_209518


namespace orange_harvest_difference_l2095_209577

/-- Represents the harvest rates for a type of orange --/
structure HarvestRate where
  ripe : ℕ
  unripe : ℕ

/-- Represents the harvest rates for weekdays and weekends --/
structure WeeklyHarvestRate where
  weekday : HarvestRate
  weekend : HarvestRate

/-- Calculates the total difference between ripe and unripe oranges for a week --/
def weeklyDifference (rate : WeeklyHarvestRate) : ℕ :=
  (rate.weekday.ripe * 5 + rate.weekend.ripe * 2) -
  (rate.weekday.unripe * 5 + rate.weekend.unripe * 2)

theorem orange_harvest_difference :
  let valencia := WeeklyHarvestRate.mk (HarvestRate.mk 90 38) (HarvestRate.mk 75 33)
  let navel := WeeklyHarvestRate.mk (HarvestRate.mk 125 65) (HarvestRate.mk 100 57)
  let blood := WeeklyHarvestRate.mk (HarvestRate.mk 60 42) (HarvestRate.mk 45 36)
  weeklyDifference valencia + weeklyDifference navel + weeklyDifference blood = 838 := by
  sorry

end orange_harvest_difference_l2095_209577


namespace odd_m_triple_g_36_l2095_209508

def g (n : ℤ) : ℤ := 
  if n % 2 = 1 then 2 * n + 3
  else if n % 3 = 0 then n / 3
  else n - 1

theorem odd_m_triple_g_36 (m : ℤ) (h_odd : m % 2 = 1) :
  g (g (g m)) = 36 → m = 54 :=
by
  sorry

end odd_m_triple_g_36_l2095_209508


namespace composite_function_equation_solution_l2095_209560

theorem composite_function_equation_solution :
  let δ : ℝ → ℝ := λ x ↦ 4 * x + 9
  let φ : ℝ → ℝ := λ x ↦ 9 * x + 8
  ∃! x : ℝ, (δ ∘ φ) x = 10 ∧ x = -31 / 36 :=
by
  sorry

end composite_function_equation_solution_l2095_209560


namespace T_is_perfect_square_T_equals_fib_squared_l2095_209595

/-- A tetromino tile is formed by gluing together four unit square tiles, edge to edge. -/
def TetrominoTile : Type := Unit

/-- Tₙ is the number of ways to tile a 2×2n rectangular bathroom floor with tetromino tiles. -/
def T (n : ℕ) : ℕ := sorry

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- The main theorem: Tₙ is always a perfect square, specifically Fₙ₊₁² -/
theorem T_is_perfect_square (n : ℕ) : ∃ k : ℕ, T n = k ^ 2 :=
  sorry

/-- The specific form of Tₙ in terms of Fibonacci numbers -/
theorem T_equals_fib_squared (n : ℕ) : T n = (fib (n + 1)) ^ 2 :=
  sorry

end T_is_perfect_square_T_equals_fib_squared_l2095_209595


namespace brick_width_calculation_l2095_209503

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 11.25

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 25

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 6

/-- The length of the wall in centimeters -/
def wall_length : ℝ := 900

/-- The width of the wall in centimeters -/
def wall_width : ℝ := 600

/-- The height of the wall in centimeters -/
def wall_height : ℝ := 22.5

/-- The number of bricks needed -/
def num_bricks : ℕ := 7200

/-- The volume of the wall in cubic centimeters -/
def wall_volume : ℝ := wall_length * wall_width * wall_height

/-- The volume of a single brick in cubic centimeters -/
def brick_volume : ℝ := brick_length * brick_width * brick_height

theorem brick_width_calculation :
  brick_width = (wall_volume / (num_bricks : ℝ)) / (brick_length * brick_height) :=
by sorry

end brick_width_calculation_l2095_209503


namespace mean_increases_median_may_unchanged_variance_increases_l2095_209597

variable (n : ℕ)
variable (x : Fin n → ℝ)
variable (x_n_plus_1 : ℝ)

-- Assume n ≥ 3
axiom n_ge_3 : n ≥ 3

-- Define the median, mean, and variance of the original dataset
def median : ℝ := sorry
def mean : ℝ := sorry
def variance : ℝ := sorry

-- Assume x_n_plus_1 is much greater than any value in x
axiom x_n_plus_1_much_greater : ∀ i : Fin n, x_n_plus_1 > x i

-- Define the new dataset including x_n_plus_1
def new_dataset : Fin (n + 1) → ℝ :=
  λ i => if h : i.val < n then x ⟨i.val, h⟩ else x_n_plus_1

-- Define the new median, mean, and variance
def new_median : ℝ := sorry
def new_mean : ℝ := sorry
def new_variance : ℝ := sorry

-- Theorem statements
theorem mean_increases : new_mean > mean := sorry
theorem median_may_unchanged : new_median = median ∨ new_median > median := sorry
theorem variance_increases : new_variance > variance := sorry

end mean_increases_median_may_unchanged_variance_increases_l2095_209597


namespace officer_selection_count_correct_l2095_209513

/-- The number of members in the club -/
def club_size : Nat := 12

/-- The number of officer positions to be filled -/
def officer_positions : Nat := 5

/-- The number of ways to choose officers from club members -/
def officer_selection_count : Nat := 95040

/-- Theorem stating that the number of ways to choose officers is correct -/
theorem officer_selection_count_correct :
  (club_size.factorial) / ((club_size - officer_positions).factorial) = officer_selection_count := by
  sorry

end officer_selection_count_correct_l2095_209513


namespace student_class_sizes_l2095_209512

/-- Represents a configuration of students in classes -/
structure StudentConfig where
  total_students : ℕ
  classes : List ℕ
  classes_sum_eq_total : classes.sum = total_students

/-- Checks if any group of n students contains at least k from the same class -/
def satisfies_group_condition (config : StudentConfig) (n k : ℕ) : Prop :=
  ∀ (subset : List ℕ), subset.sum ≤ n → (∃ (c : ℕ), c ∈ config.classes ∧ c ≥ k)

/-- The main theorem to be proved -/
theorem student_class_sizes 
  (config : StudentConfig)
  (h_total : config.total_students = 60)
  (h_condition : satisfies_group_condition config 10 3) :
  (∃ (c : ℕ), c ∈ config.classes ∧ c ≥ 15) ∧
  ¬(∀ (config : StudentConfig), 
    config.total_students = 60 → 
    satisfies_group_condition config 10 3 → 
    ∃ (c : ℕ), c ∈ config.classes ∧ c ≥ 16) :=
by sorry

end student_class_sizes_l2095_209512


namespace arithmetic_mean_sqrt2_l2095_209598

theorem arithmetic_mean_sqrt2 :
  let x := Real.sqrt 2 - 1
  (x + (1 / x)) / 2 = Real.sqrt 2 := by sorry

end arithmetic_mean_sqrt2_l2095_209598


namespace prob_one_pascal_20_l2095_209529

/-- The number of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of ones in the first n rows of Pascal's Triangle -/
def pascal_triangle_ones (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def prob_one_pascal (n : ℕ) : ℚ :=
  (pascal_triangle_ones n) / (pascal_triangle_elements n)

theorem prob_one_pascal_20 :
  prob_one_pascal 20 = 13 / 70 := by
  sorry

end prob_one_pascal_20_l2095_209529


namespace race_distance_l2095_209516

theorem race_distance (a_finish_time : ℝ) (time_diff : ℝ) (distance_diff : ℝ) :
  a_finish_time = 3 →
  time_diff = 7 →
  distance_diff = 56 →
  ∃ (total_distance : ℝ),
    total_distance = 136 ∧
    (total_distance / a_finish_time) * time_diff = distance_diff :=
by sorry

end race_distance_l2095_209516


namespace office_officers_count_l2095_209563

/-- Represents the salary and employee data for an office --/
structure OfficeSalaryData where
  avgSalaryAll : ℚ
  avgSalaryOfficers : ℚ
  avgSalaryNonOfficers : ℚ
  numNonOfficers : ℕ

/-- Calculates the number of officers given the office salary data --/
def calculateOfficers (data : OfficeSalaryData) : ℕ :=
  sorry

/-- Theorem stating that the number of officers is 15 given the specific salary data --/
theorem office_officers_count (data : OfficeSalaryData) 
  (h1 : data.avgSalaryAll = 120)
  (h2 : data.avgSalaryOfficers = 450)
  (h3 : data.avgSalaryNonOfficers = 110)
  (h4 : data.numNonOfficers = 495) :
  calculateOfficers data = 15 := by
  sorry

end office_officers_count_l2095_209563


namespace joes_test_scores_l2095_209587

theorem joes_test_scores (initial_avg : ℝ) (lowest_score : ℝ) (new_avg : ℝ) :
  initial_avg = 70 →
  lowest_score = 55 →
  new_avg = 75 →
  ∃ n : ℕ, n > 1 ∧
    (n : ℝ) * initial_avg - lowest_score = (n - 1 : ℝ) * new_avg ∧
    n = 4 :=
by sorry

end joes_test_scores_l2095_209587


namespace circle_inside_polygon_l2095_209556

/-- A convex polygon in a 2D plane -/
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)
  is_convex : Bool  -- We assume this is true for a convex polygon

/-- The area of a convex polygon -/
def area (p : ConvexPolygon) : ℝ := sorry

/-- The perimeter of a convex polygon -/
def perimeter (p : ConvexPolygon) : ℝ := sorry

/-- The distance from a point to a line segment -/
def distance_to_side (point : ℝ × ℝ) (side : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: In any convex polygon, there exists a point that is at least A/P distance away from all sides -/
theorem circle_inside_polygon (p : ConvexPolygon) :
  ∃ (center : ℝ × ℝ), 
    (∀ (side : (ℝ × ℝ) × (ℝ × ℝ)), 
      side.1 ∈ p.vertices ∧ side.2 ∈ p.vertices →
      distance_to_side center side ≥ area p / perimeter p) :=
sorry

end circle_inside_polygon_l2095_209556


namespace kittens_given_to_jessica_l2095_209594

theorem kittens_given_to_jessica (initial_kittens : ℕ) (received_kittens : ℕ) (final_kittens : ℕ) :
  initial_kittens = 6 →
  received_kittens = 9 →
  final_kittens = 12 →
  initial_kittens + received_kittens - final_kittens = 3 :=
by
  sorry

end kittens_given_to_jessica_l2095_209594


namespace smallest_integer_result_l2095_209530

def expression : List ℕ := [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

def is_valid_bracketing (b : List (List ℕ)) : Prop :=
  b.join = expression ∧ ∀ l ∈ b, l.length > 0

def evaluate_bracketing (b : List (List ℕ)) : ℚ :=
  b.foldl (λ acc l => acc / l.foldl (λ x y => x / y) 1) 1

def is_integer_result (b : List (List ℕ)) : Prop :=
  ∃ n : ℤ, (evaluate_bracketing b).num = n * (evaluate_bracketing b).den

theorem smallest_integer_result :
  ∃ b : List (List ℕ),
    is_valid_bracketing b ∧
    is_integer_result b ∧
    evaluate_bracketing b = 7 ∧
    ∀ b' : List (List ℕ),
      is_valid_bracketing b' →
      is_integer_result b' →
      evaluate_bracketing b' ≥ 7 :=
by sorry

end smallest_integer_result_l2095_209530


namespace quadratic_roots_real_and_unequal_l2095_209568

theorem quadratic_roots_real_and_unequal :
  let a : ℝ := 1
  let b : ℝ := -6
  let c : ℝ := 8
  let discriminant := b^2 - 4*a*c
  discriminant > 0 ∧ ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  sorry

#check quadratic_roots_real_and_unequal

end quadratic_roots_real_and_unequal_l2095_209568


namespace zoo_animal_ratio_l2095_209544

theorem zoo_animal_ratio (parrots : ℕ) (snakes : ℕ) (elephants : ℕ) (zebras : ℕ) (monkeys : ℕ) :
  parrots = 8 →
  snakes = 3 * parrots →
  elephants = (parrots + snakes) / 2 →
  zebras = elephants - 3 →
  monkeys - zebras = 35 →
  monkeys / snakes = 2 :=
by
  sorry

end zoo_animal_ratio_l2095_209544


namespace series_sum_equals_negative_four_l2095_209582

/-- The sum of the infinite series $\sum_{n=1}^\infty \frac{2n^2 - 3n + 2}{n(n+1)(n+2)}$ equals -4. -/
theorem series_sum_equals_negative_four :
  ∑' n : ℕ+, (2 * n^2 - 3 * n + 2 : ℝ) / (n * (n + 1) * (n + 2)) = -4 := by
  sorry

end series_sum_equals_negative_four_l2095_209582


namespace difference_of_squares_153_147_l2095_209520

theorem difference_of_squares_153_147 : 153^2 - 147^2 = 1800 := by
  sorry

end difference_of_squares_153_147_l2095_209520


namespace inequality_proof_l2095_209500

theorem inequality_proof (a : ℝ) (h1 : 0 < a) (h2 : a < 1/2) :
  a^(Real.sqrt a) > a^(a^a) ∧ a^(a^a) > a := by
  sorry

end inequality_proof_l2095_209500


namespace negation_of_existence_quadratic_inequality_negation_l2095_209549

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) := by sorry

theorem quadratic_inequality_negation :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by sorry

end negation_of_existence_quadratic_inequality_negation_l2095_209549


namespace geometric_sequence_product_l2095_209534

def geometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometricSequence a → a 5 = 2 → a 1 * a 2 * a 3 * a 7 * a 8 * a 9 = 64 := by
  sorry

end geometric_sequence_product_l2095_209534


namespace expression_simplification_l2095_209572

theorem expression_simplification (x : ℝ) : 
  ((3 * x - 1) - 5 * x) / 3 = -2/3 * x - 1/3 := by sorry

end expression_simplification_l2095_209572


namespace total_art_pieces_l2095_209565

theorem total_art_pieces (asian : Nat) (egyptian : Nat) (european : Nat)
  (h1 : asian = 465)
  (h2 : egyptian = 527)
  (h3 : european = 320) :
  asian + egyptian + european = 1312 := by
  sorry

end total_art_pieces_l2095_209565


namespace arithmetic_sequence_sum_l2095_209557

theorem arithmetic_sequence_sum (a₁ aₙ : ℤ) (n : ℕ) (h : n > 0) :
  let S := n * (a₁ + aₙ) / 2
  a₁ = -3 ∧ aₙ = 48 ∧ n = 12 → S = 270 := by
  sorry

end arithmetic_sequence_sum_l2095_209557


namespace min_value_quadratic_l2095_209589

theorem min_value_quadratic : 
  ∃ (min : ℝ), min = -39 ∧ ∀ (x : ℝ), x^2 + 14*x + 10 ≥ min := by
  sorry

end min_value_quadratic_l2095_209589


namespace carson_age_carson_age_real_l2095_209583

/-- Given the ages of Aunt Anna, Maria, and Carson, prove Carson's age -/
theorem carson_age (anna_age : ℕ) (maria_age : ℕ) (carson_age : ℕ) : 
  anna_age = 60 →
  maria_age = 2 * anna_age / 3 →
  carson_age = maria_age - 7 →
  carson_age = 33 := by
sorry

/-- Alternative formulation using real numbers for more precise calculations -/
theorem carson_age_real (anna_age : ℝ) (maria_age : ℝ) (carson_age : ℝ) : 
  anna_age = 60 →
  maria_age = 2 / 3 * anna_age →
  carson_age = maria_age - 7 →
  carson_age = 33 := by
sorry

end carson_age_carson_age_real_l2095_209583


namespace jane_average_score_l2095_209584

def jane_scores : List ℝ := [89, 95, 88, 92, 94, 87]

theorem jane_average_score : 
  (jane_scores.sum / jane_scores.length : ℝ) = 90.8333 := by
  sorry

end jane_average_score_l2095_209584


namespace smallest_number_l2095_209541

theorem smallest_number (S : Set ℤ) (h : S = {-2, 0, 1, 2}) : 
  ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = -2 :=
by sorry

end smallest_number_l2095_209541


namespace no_prime_pair_sum_65_l2095_209559

theorem no_prime_pair_sum_65 : ¬∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p + q = 65 ∧ ∃ (k : ℕ), p * q = k := by
  sorry

end no_prime_pair_sum_65_l2095_209559


namespace perp_line_plane_condition_l2095_209575

/-- A straight line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Defines the perpendicular relationship between a line and another line -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Defines the perpendicular relationship between a line and a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when a line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- The main theorem stating that "m ⊥ n" is a necessary but not sufficient condition for "m ⊥ α" -/
theorem perp_line_plane_condition (m n : Line3D) (α : Plane3D) 
  (h : line_in_plane n α) :
  (perpendicular_line_plane m α → perpendicular_lines m n) ∧
  ¬(perpendicular_lines m n → perpendicular_line_plane m α) :=
sorry

end perp_line_plane_condition_l2095_209575


namespace power_of_two_equality_l2095_209599

theorem power_of_two_equality : ∃ y : ℕ, 8^3 + 8^3 + 8^3 + 8^3 = 2^y ∧ y = 11 := by
  sorry

end power_of_two_equality_l2095_209599


namespace tan_4125_degrees_l2095_209540

theorem tan_4125_degrees : Real.tan (4125 * π / 180) = -(2 - Real.sqrt 3) := by sorry

end tan_4125_degrees_l2095_209540


namespace remainder_of_power_divided_by_polynomial_l2095_209581

theorem remainder_of_power_divided_by_polynomial (x : ℤ) :
  (x + 1)^2010 ≡ 1 [ZMOD (x^2 + x + 1)] := by
  sorry

end remainder_of_power_divided_by_polynomial_l2095_209581


namespace negative_sqrt_four_equals_negative_two_l2095_209535

theorem negative_sqrt_four_equals_negative_two : -Real.sqrt 4 = -2 := by
  sorry

end negative_sqrt_four_equals_negative_two_l2095_209535


namespace wechat_payment_balance_l2095_209558

/-- Represents a transaction with a description and an amount -/
structure Transaction where
  description : String
  amount : Int

/-- Calculates the balance from a list of transactions -/
def calculate_balance (transactions : List Transaction) : Int :=
  transactions.foldl (fun acc t => acc + t.amount) 0

/-- Theorem stating that the WeChat change payment balance for the day is an expenditure of $32 -/
theorem wechat_payment_balance : 
  let transactions : List Transaction := [
    { description := "Transfer from LZT", amount := 48 },
    { description := "Blue Wisteria Culture", amount := -30 },
    { description := "Scan QR code payment", amount := -50 }
  ]
  calculate_balance transactions = -32 := by sorry

end wechat_payment_balance_l2095_209558


namespace least_positive_integer_divisible_by_four_primes_l2095_209552

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    p₁ ∣ n ∧ p₂ ∣ n ∧ p₃ ∣ n ∧ p₄ ∣ n) ∧
  (∀ m : ℕ, m > 0 → m < n → 
    ¬(∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      q₁ ∣ m ∧ q₂ ∣ m ∧ q₃ ∣ m ∧ q₄ ∣ m)) ∧
  n = 210 := by
sorry

end least_positive_integer_divisible_by_four_primes_l2095_209552


namespace pen_package_size_l2095_209525

def is_proper_factor (n m : ℕ) : Prop := n ∣ m ∧ n ≠ 1 ∧ n ≠ m

theorem pen_package_size (pen_package_size : ℕ) 
  (h1 : pen_package_size > 0)
  (h2 : ∃ (num_packages : ℕ), num_packages * pen_package_size = 60) :
  is_proper_factor pen_package_size 60 := by
sorry

end pen_package_size_l2095_209525


namespace f_negative_when_x_greater_half_l2095_209570

/-- The linear function f(x) = -2x + 1 -/
def f (x : ℝ) : ℝ := -2 * x + 1

/-- When x > 1/2, f(x) < 0 -/
theorem f_negative_when_x_greater_half : ∀ x : ℝ, x > (1/2) → f x < 0 := by
  sorry

end f_negative_when_x_greater_half_l2095_209570


namespace fraction_simplification_l2095_209506

theorem fraction_simplification : (3/8 + 5/6) / (5/12 + 1/4) = 29/16 := by
  sorry

end fraction_simplification_l2095_209506


namespace pumpkin_total_weight_l2095_209569

/-- The total weight of two pumpkins is 12.7 pounds, given their individual weights -/
theorem pumpkin_total_weight (weight1 weight2 : ℝ) 
  (h1 : weight1 = 4) 
  (h2 : weight2 = 8.7) : 
  weight1 + weight2 = 12.7 := by
  sorry

end pumpkin_total_weight_l2095_209569


namespace sqrt_fraction_equality_l2095_209505

theorem sqrt_fraction_equality : Real.sqrt (25 / 121) = 5 / 11 := by sorry

end sqrt_fraction_equality_l2095_209505


namespace solution_to_equation_l2095_209566

theorem solution_to_equation (x : ℝ) : 2 * x - 8 = 0 ↔ x = 4 := by sorry

end solution_to_equation_l2095_209566


namespace sum_of_roots_l2095_209553

theorem sum_of_roots (h b x₁ x₂ : ℝ) 
  (hx : x₁ ≠ x₂) 
  (eq₁ : 3 * x₁^2 - h * x₁ = b) 
  (eq₂ : 3 * x₂^2 - h * x₂ = b) : 
  x₁ + x₂ = h / 3 := by
sorry

end sum_of_roots_l2095_209553


namespace max_abs_sum_on_circle_l2095_209547

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (max : ℝ), (∀ a b : ℝ, a^2 + b^2 = 4 → |a| + |b| ≤ max) ∧ (|x| + |y| = max) :=
by sorry

end max_abs_sum_on_circle_l2095_209547


namespace intersection_symmetry_l2095_209519

/-- The line y = ax + 1 intersects the curve x^2 + y^2 + bx - y = 1 at two points
    which are symmetric about the line x + y = 0. -/
theorem intersection_symmetry (a b : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    -- Line equation
    (y₁ = a * x₁ + 1) ∧ (y₂ = a * x₂ + 1) ∧
    -- Curve equation
    (x₁^2 + y₁^2 + b * x₁ - y₁ = 1) ∧ (x₂^2 + y₂^2 + b * x₂ - y₂ = 1) ∧
    -- Symmetry condition
    (x₁ + y₁ = -(x₂ + y₂)) ∧
    -- Distinct points
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) →
  a + b = 2 := by
sorry

end intersection_symmetry_l2095_209519


namespace product_equals_seven_l2095_209571

theorem product_equals_seven : 
  (1 + 1/1) * (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) = 7 := by
  sorry

end product_equals_seven_l2095_209571


namespace expand_product_l2095_209591

theorem expand_product (x y : ℝ) : (x + 3) * (x + 2*y + 4) = x^2 + 7*x + 2*x*y + 6*y + 12 := by
  sorry

end expand_product_l2095_209591


namespace greg_initial_amount_l2095_209514

/-- Represents the initial and final monetary states of Earl, Fred, and Greg -/
structure MonetaryState where
  earl_initial : ℕ
  fred_initial : ℕ
  greg_initial : ℕ
  earl_owes_fred : ℕ
  fred_owes_greg : ℕ
  greg_owes_earl : ℕ
  earl_final : ℕ
  fred_final : ℕ
  greg_final : ℕ

/-- The theorem states that given the initial conditions and debt payments,
    Greg's initial amount is 36 dollars -/
theorem greg_initial_amount (state : MonetaryState)
  (h1 : state.earl_initial = 90)
  (h2 : state.fred_initial = 48)
  (h3 : state.earl_owes_fred = 28)
  (h4 : state.fred_owes_greg = 32)
  (h5 : state.greg_owes_earl = 40)
  (h6 : state.earl_final + state.greg_final = 130)
  (h7 : state.earl_final = state.earl_initial - state.earl_owes_fred + state.greg_owes_earl)
  (h8 : state.fred_final = state.fred_initial + state.earl_owes_fred - state.fred_owes_greg)
  (h9 : state.greg_final = state.greg_initial + state.fred_owes_greg - state.greg_owes_earl) :
  state.greg_initial = 36 := by
  sorry


end greg_initial_amount_l2095_209514


namespace inverse_proportion_y_relationship_l2095_209539

/-- Proves the relationship between y-coordinates of points on an inverse proportion function -/
theorem inverse_proportion_y_relationship :
  ∀ (y₁ y₂ y₃ : ℝ),
  y₁ = -6 / (-2) →
  y₂ = -6 / (-1) →
  y₃ = -6 / 3 →
  y₂ > y₁ ∧ y₁ > y₃ :=
by
  sorry

#check inverse_proportion_y_relationship

end inverse_proportion_y_relationship_l2095_209539


namespace baba_yaga_journey_l2095_209501

/-- The problem of Baba Yaga's journey to Bald Mountain -/
theorem baba_yaga_journey 
  (arrival_time : ℕ) 
  (slow_speed : ℕ) 
  (fast_speed : ℕ) 
  (late_hours : ℕ) 
  (early_hours : ℕ) 
  (h : arrival_time = 24) -- Midnight is represented as 24
  (h_slow : slow_speed = 50)
  (h_fast : fast_speed = 150)
  (h_late : late_hours = 2)
  (h_early : early_hours = 2)
  : ∃ (departure_time speed : ℕ),
    departure_time = 20 ∧ 
    speed = 75 ∧
    (arrival_time - departure_time) * speed = 
      (arrival_time - departure_time + late_hours) * slow_speed ∧
    (arrival_time - departure_time) * speed = 
      (arrival_time - departure_time - early_hours) * fast_speed :=
sorry

end baba_yaga_journey_l2095_209501


namespace book_reading_time_l2095_209574

theorem book_reading_time (total_pages : ℕ) (rate1 rate2 : ℕ) (days1 days2 : ℕ) : 
  total_pages = 525 →
  rate1 = 25 →
  rate2 = 21 →
  days1 * rate1 = total_pages →
  days2 * rate2 = total_pages →
  (days1 = 21 ∧ days2 = 25) := by
  sorry

end book_reading_time_l2095_209574


namespace function_composition_problem_l2095_209551

theorem function_composition_problem (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x / 3 + 2) →
  (∀ x, g x = 5 - 2 * x) →
  f (g a) = 6 →
  a = -7/2 := by
sorry

end function_composition_problem_l2095_209551


namespace circle_area_from_circumference_l2095_209509

theorem circle_area_from_circumference (C : ℝ) (r : ℝ) (h : C = 36 * Real.pi) :
  r * r * Real.pi = 324 * Real.pi :=
by
  sorry

end circle_area_from_circumference_l2095_209509


namespace families_left_near_mountain_l2095_209545

/-- The number of bird families initially living near the mountain. -/
def initial_families : ℕ := 41

/-- The number of bird families that flew away for the winter. -/
def families_flew_away : ℕ := 27

/-- Theorem: The number of bird families left near the mountain is 14. -/
theorem families_left_near_mountain :
  initial_families - families_flew_away = 14 := by
  sorry

end families_left_near_mountain_l2095_209545


namespace iron_bar_height_l2095_209564

/-- Proves that the height of an iron bar is 6 cm given specific conditions --/
theorem iron_bar_height : 
  ∀ (length width height : ℝ) (num_bars num_balls ball_volume : ℕ),
  length = 12 →
  width = 8 →
  num_bars = 10 →
  num_balls = 720 →
  ball_volume = 8 →
  (num_bars : ℝ) * length * width * height = (num_balls : ℝ) * (ball_volume : ℝ) →
  height = 6 := by
sorry

end iron_bar_height_l2095_209564


namespace cricketer_average_score_l2095_209546

theorem cricketer_average_score (score1 score2 : ℕ) (matches1 matches2 : ℕ) 
  (h1 : matches1 = 2)
  (h2 : matches2 = 3)
  (h3 : score1 = 60)
  (h4 : score2 = 50) :
  (matches1 * score1 + matches2 * score2) / (matches1 + matches2) = 54 := by
  sorry

end cricketer_average_score_l2095_209546


namespace calculation_proof_l2095_209515

theorem calculation_proof : 0.2 * 63 + 1.9 * 126 + 196 * 9 = 2016 := by
  sorry

end calculation_proof_l2095_209515


namespace number_puzzle_l2095_209554

theorem number_puzzle : ∃! x : ℝ, (x / 12) * 24 = x + 36 := by
  sorry

end number_puzzle_l2095_209554


namespace shoes_theorem_l2095_209524

def shoes_problem (bonny becky bobby cherry diane : ℚ) : Prop :=
  -- Conditions
  bonny = 13 ∧
  bonny = 2 * becky - 5 ∧
  bobby = 3.5 * becky ∧
  cherry = bonny + becky + 4.5 ∧
  diane = 3 * cherry - 2 - 3 ∧
  -- Conclusion
  ⌊bonny + becky + bobby + cherry + diane⌋ = 154

theorem shoes_theorem : ∃ bonny becky bobby cherry diane : ℚ, 
  shoes_problem bonny becky bobby cherry diane := by
  sorry

end shoes_theorem_l2095_209524


namespace bicycle_price_calculation_l2095_209593

theorem bicycle_price_calculation (original_price : ℝ) 
  (initial_discount_rate : ℝ) (additional_discount : ℝ) (sales_tax_rate : ℝ) :
  original_price = 200 →
  initial_discount_rate = 0.25 →
  additional_discount = 10 →
  sales_tax_rate = 0.05 →
  (original_price * (1 - initial_discount_rate) - additional_discount) * (1 + sales_tax_rate) = 147 := by
  sorry

end bicycle_price_calculation_l2095_209593


namespace three_distinct_roots_transformation_l2095_209550

/-- Given an equation a x^5 + b x^4 + c = 0 with three distinct roots,
    prove that c x^5 + b x + a = 0 also has three distinct roots -/
theorem three_distinct_roots_transformation (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    a * x^5 + b * x^4 + c = 0 ∧
    a * y^5 + b * y^4 + c = 0 ∧
    a * z^5 + b * z^4 + c = 0) →
  (∃ u v w : ℝ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    c * u^5 + b * u + a = 0 ∧
    c * v^5 + b * v + a = 0 ∧
    c * w^5 + b * w + a = 0) :=
by sorry

end three_distinct_roots_transformation_l2095_209550


namespace difference_repetition_l2095_209586

theorem difference_repetition (a : Fin 20 → ℕ) 
  (h_order : ∀ i j, i < j → a i < a j) 
  (h_bound : a 19 ≤ 70) : 
  ∃ (j₁ k₁ j₂ k₂ j₃ k₃ j₄ k₄ : Fin 20), 
    k₁ < j₁ ∧ k₂ < j₂ ∧ k₃ < j₃ ∧ k₄ < j₄ ∧
    (a j₁ - a k₁ : ℤ) = (a j₂ - a k₂) ∧
    (a j₁ - a k₁ : ℤ) = (a j₃ - a k₃) ∧
    (a j₁ - a k₁ : ℤ) = (a j₄ - a k₄) :=
by sorry

end difference_repetition_l2095_209586


namespace inequality_abc_l2095_209588

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^a * b^b * c^c ≥ (a*b*c)^((a+b+c)/3) := by
  sorry

end inequality_abc_l2095_209588


namespace triangle_side_a_triangle_angle_C_l2095_209532

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem triangle_side_a (t : Triangle) (hb : t.b = 2) (hB : t.B = π/6) (hC : t.C = 3*π/4) :
  t.a = Real.sqrt 6 - Real.sqrt 2 := by
  sorry

-- Part 2
theorem triangle_angle_C (t : Triangle) (hS : t.a * t.b * Real.sin t.C / 2 = (t.a^2 + t.b^2 - t.c^2) / 4) :
  t.C = π/4 := by
  sorry

end triangle_side_a_triangle_angle_C_l2095_209532


namespace person_A_savings_l2095_209526

/-- The amount of money saved by person A -/
def savings_A : ℕ := sorry

/-- The amount of money saved by person B -/
def savings_B : ℕ := sorry

/-- The amount of money saved by person C -/
def savings_C : ℕ := sorry

/-- Person A and B together have saved 640 yuan -/
axiom AB_savings : savings_A + savings_B = 640

/-- Person B and C together have saved 600 yuan -/
axiom BC_savings : savings_B + savings_C = 600

/-- Person A and C together have saved 440 yuan -/
axiom AC_savings : savings_A + savings_C = 440

/-- Theorem: Given the conditions, person A has saved 240 yuan -/
theorem person_A_savings : savings_A = 240 :=
  sorry

end person_A_savings_l2095_209526


namespace ab_plus_cd_value_l2095_209507

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = -3)
  (eq3 : a + c + d = 10)
  (eq4 : b + c + d = -1) :
  a * b + c * d = -346 / 9 := by
sorry

end ab_plus_cd_value_l2095_209507


namespace boxer_weight_theorem_l2095_209517

/-- Represents a diet with a specific weight loss per month -/
structure Diet where
  weightLossPerMonth : ℝ
  
/-- Calculates the weight after a given number of months on a diet -/
def weightAfterMonths (initialWeight : ℝ) (diet : Diet) (months : ℝ) : ℝ :=
  initialWeight - diet.weightLossPerMonth * months

/-- Theorem about boxer's weight and diets -/
theorem boxer_weight_theorem (x : ℝ) :
  let dietA : Diet := ⟨2⟩
  let dietB : Diet := ⟨3⟩
  let dietC : Diet := ⟨4⟩
  let monthsToFight : ℝ := 4
  
  (weightAfterMonths x dietB monthsToFight = 97) →
  (x = 109) ∧
  (weightAfterMonths x dietA monthsToFight = 101) ∧
  (weightAfterMonths x dietB monthsToFight = 97) ∧
  (weightAfterMonths x dietC monthsToFight = 93) := by
  sorry


end boxer_weight_theorem_l2095_209517
