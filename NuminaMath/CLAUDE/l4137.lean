import Mathlib

namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l4137_413708

/-- A sequence of complex numbers -/
def ComplexSequence := ℕ → ℂ

/-- Predicate to check if a natural number is prime -/
def IsPrime (p : ℕ) : Prop := sorry

/-- Predicate to check if a series converges -/
def Converges (s : ℕ → ℂ) : Prop := sorry

/-- The main theorem -/
theorem existence_of_special_sequence :
  ∃ (a : ComplexSequence), ∀ (p : ℕ), p > 0 →
    (Converges (fun n => (a n)^p) ↔ ¬(IsPrime p)) := by sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l4137_413708


namespace NUMINAMATH_CALUDE_dictionary_chunk_pages_l4137_413740

def is_permutation (a b : ℕ) : Prop := sorry

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem dictionary_chunk_pages (first_page last_page : ℕ) :
  first_page = 213 →
  is_permutation first_page last_page →
  is_even last_page →
  ∀ p, is_permutation first_page p ∧ is_even p → p ≤ last_page →
  last_page - first_page + 1 = 100 :=
by sorry

end NUMINAMATH_CALUDE_dictionary_chunk_pages_l4137_413740


namespace NUMINAMATH_CALUDE_parallel_line_correct_perpendicular_bisector_correct_l4137_413762

-- Define the points
def P : ℝ × ℝ := (-1, 3)
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 1)

-- Define the original line
def original_line (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define the parallel line through P
def parallel_line (x y : ℝ) : Prop := x - 2*y + 7 = 0

-- Define the perpendicular bisector of AB
def perpendicular_bisector (x y : ℝ) : Prop := 4*x - 2*y - 5 = 0

-- Theorem 1: The parallel line passes through P and is parallel to the original line
theorem parallel_line_correct :
  parallel_line P.1 P.2 ∧
  ∀ (x y : ℝ), parallel_line x y ↔ ∃ (k : ℝ), original_line (x + k) (y + k/2) :=
sorry

-- Theorem 2: The perpendicular bisector is correct
theorem perpendicular_bisector_correct :
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  perpendicular_bisector midpoint.1 midpoint.2 ∧
  (B.2 - A.2) * (B.1 - A.1) * 2 = -1 :=
sorry

end NUMINAMATH_CALUDE_parallel_line_correct_perpendicular_bisector_correct_l4137_413762


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt_three_half_l4137_413713

theorem sin_cos_sum_equals_sqrt_three_half : 
  Real.sin (10 * π / 180) * Real.cos (50 * π / 180) + 
  Real.cos (10 * π / 180) * Real.sin (130 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt_three_half_l4137_413713


namespace NUMINAMATH_CALUDE_probability_point_near_origin_l4137_413706

/-- The probability of a point being within 2 units of the origin when randomly selected from a rectangle -/
theorem probability_point_near_origin (rectangle_width : ℝ) (rectangle_height : ℝ) 
  (h_width : rectangle_width = 6) (h_height : rectangle_height = 8) : 
  (π * 2^2) / (rectangle_width * rectangle_height) = π / 12 := by
  sorry

#check probability_point_near_origin

end NUMINAMATH_CALUDE_probability_point_near_origin_l4137_413706


namespace NUMINAMATH_CALUDE_true_discount_calculation_l4137_413737

/-- Given a present worth and banker's gain, calculate the true discount. -/
theorem true_discount_calculation (PW BG : ℚ) (h1 : PW = 576) (h2 : BG = 16) :
  ∃ TD : ℚ, TD^2 = BG * PW ∧ TD = 96 := by
  sorry

end NUMINAMATH_CALUDE_true_discount_calculation_l4137_413737


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l4137_413778

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 1 < 0) → (a < -2 ∨ a > 2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l4137_413778


namespace NUMINAMATH_CALUDE_circle_and_line_equations_l4137_413787

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 1 ∨ y = -x + 1

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y + 2 * Real.sqrt 2 = 0

theorem circle_and_line_equations :
  ∃ (a : ℝ),
    a ≤ 0 ∧
    circle_M 0 (-2) ∧
    (∃ (x y : ℝ), circle_M x y ∧ tangent_line x y) ∧
    line_l 0 1 ∧
    (∃ (A B : ℝ × ℝ),
      circle_M A.1 A.2 ∧
      circle_M B.1 B.2 ∧
      line_l A.1 A.2 ∧
      line_l B.1 B.2 ∧
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = 14) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_and_line_equations_l4137_413787


namespace NUMINAMATH_CALUDE_cars_per_salesperson_per_month_l4137_413772

/-- Proves that given 500 cars for sale, 10 sales professionals, and a 5-month period to sell all cars, each salesperson sells 10 cars per month. -/
theorem cars_per_salesperson_per_month 
  (total_cars : ℕ) 
  (sales_professionals : ℕ) 
  (months_to_sell : ℕ) 
  (h1 : total_cars = 500) 
  (h2 : sales_professionals = 10) 
  (h3 : months_to_sell = 5) :
  total_cars / (sales_professionals * months_to_sell) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_cars_per_salesperson_per_month_l4137_413772


namespace NUMINAMATH_CALUDE_cos_neg_seventeen_pi_fourths_l4137_413716

theorem cos_neg_seventeen_pi_fourths : Real.cos (-17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_neg_seventeen_pi_fourths_l4137_413716


namespace NUMINAMATH_CALUDE_sqrt_15_minus_1_range_l4137_413757

theorem sqrt_15_minus_1_range : 2 < Real.sqrt 15 - 1 ∧ Real.sqrt 15 - 1 < 3 := by
  have h1 : 9 < 15 := by sorry
  have h2 : 15 < 16 := by sorry
  sorry

end NUMINAMATH_CALUDE_sqrt_15_minus_1_range_l4137_413757


namespace NUMINAMATH_CALUDE_fraction_problem_l4137_413761

theorem fraction_problem (f : ℝ) : f * 50.0 - 4 = 6 → f = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l4137_413761


namespace NUMINAMATH_CALUDE_tetrahedron_rotation_common_volume_l4137_413731

theorem tetrahedron_rotation_common_volume
  (V : ℝ) (α : ℝ) (h : 0 < α ∧ α < π) :
  ∃ (common_volume : ℝ),
    common_volume = V * (1 + Real.tan (α/2)^2) / (1 + Real.tan (α/2))^2 :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_rotation_common_volume_l4137_413731


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_simplify_resistance_formula_compare_time_taken_l4137_413775

-- 1. Simplify complex fraction
theorem simplify_complex_fraction (x y : ℝ) (h : y ≠ x) :
  (1 + x / y) / (1 - x / y) = (y + x) / (y - x) := by sorry

-- 2. Simplify resistance formula
theorem simplify_resistance_formula (R R₁ R₂ : ℝ) (h₁ : R₁ > 0) (h₂ : R₂ > 0) :
  (1 / R = 1 / R₁ + 1 / R₂) → R = (R₁ * R₂) / (R₁ + R₂) := by sorry

-- 3. Compare time taken
theorem compare_time_taken (x y z : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) :
  x / (1 / (1 / y + 1 / z)) = (x * y + x * z) / (y * z) := by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_simplify_resistance_formula_compare_time_taken_l4137_413775


namespace NUMINAMATH_CALUDE_three_player_cooperation_strategy_l4137_413714

/-- Represents the dimensions of the game board -/
def boardSize : Nat := 1000

/-- Represents the possible rectangle shapes that can be painted -/
inductive Rectangle
  | twoByOne
  | oneByTwo
  | oneByThree
  | threeByOne

/-- Represents a player in the game -/
inductive Player
  | Andy
  | Bess
  | Charley
  | Dick

/-- Represents a position on the board -/
structure Position where
  x : Fin boardSize
  y : Fin boardSize

/-- Represents a move in the game -/
structure Move where
  player : Player
  rectangle : Rectangle
  position : Position

/-- The game state -/
structure GameState where
  board : Fin boardSize → Fin boardSize → Bool
  currentPlayer : Player

/-- Function to check if a move is valid -/
def isValidMove (state : GameState) (move : Move) : Bool := sorry

/-- Function to apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState := sorry

/-- Function to check if a player has a valid move -/
def hasValidMove (state : GameState) (player : Player) : Bool := sorry

/-- Theorem: There exists a strategy for three players to make the fourth player lose -/
theorem three_player_cooperation_strategy :
  ∃ (strategy : GameState → Move),
    ∀ (initialState : GameState),
      ∃ (losingPlayer : Player),
        ¬(hasValidMove (applyMove initialState (strategy initialState)) losingPlayer) :=
sorry

end NUMINAMATH_CALUDE_three_player_cooperation_strategy_l4137_413714


namespace NUMINAMATH_CALUDE_two_real_roots_condition_l4137_413712

theorem two_real_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x - 2*k + 8 = 0 ∧ y^2 - 4*y - 2*k + 8 = 0) ↔ k ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_two_real_roots_condition_l4137_413712


namespace NUMINAMATH_CALUDE_vector_equation_solution_l4137_413768

theorem vector_equation_solution :
  let c₁ : ℚ := 5/6
  let c₂ : ℚ := -7/18
  let v₁ : Fin 2 → ℚ := ![1, 4]
  let v₂ : Fin 2 → ℚ := ![-3, 6]
  let result : Fin 2 → ℚ := ![2, 1]
  c₁ • v₁ + c₂ • v₂ = result := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l4137_413768


namespace NUMINAMATH_CALUDE_bernardo_wins_with_92_l4137_413719

def game_sequence (M : ℕ) : ℕ → ℕ 
| 0 => M
| 1 => 3 * M
| 2 => 3 * M + 40
| 3 => 9 * M + 120
| 4 => 9 * M + 160
| 5 => 27 * M + 480
| _ => 0

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem bernardo_wins_with_92 :
  ∃ (M : ℕ), 
    M ≥ 1 ∧ 
    M ≤ 1000 ∧ 
    game_sequence M 5 < 3000 ∧ 
    game_sequence M 5 + 40 ≥ 3000 ∧
    sum_of_digits M = 11 ∧
    (∀ (N : ℕ), N < M → 
      (game_sequence N 5 < 3000 → game_sequence N 5 + 40 < 3000) ∨ 
      game_sequence N 5 ≥ 3000) :=
by
  use 92
  sorry

#eval game_sequence 92 5  -- Should output 2964
#eval game_sequence 92 5 + 40  -- Should output 3004
#eval sum_of_digits 92  -- Should output 11

end NUMINAMATH_CALUDE_bernardo_wins_with_92_l4137_413719


namespace NUMINAMATH_CALUDE_inverse_of_i_power_2023_l4137_413725

theorem inverse_of_i_power_2023 : ∃ z : ℂ, z = (Complex.I : ℂ) ^ 2023 ∧ z⁻¹ = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_i_power_2023_l4137_413725


namespace NUMINAMATH_CALUDE_sum_of_numbers_l4137_413773

theorem sum_of_numbers (a b : ℕ) (h : (a + b) * (a - b) = 1996) : a + b = 998 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l4137_413773


namespace NUMINAMATH_CALUDE_britta_winning_strategy_l4137_413736

-- Define the game
def Game (n : ℕ) :=
  n ≥ 5 ∧ Odd n

-- Define Britta's winning condition
def BrittaWins (n x₁ x₂ y₁ y₂ : ℕ) : Prop :=
  (x₁ * x₂ * (x₁ - y₁) * (x₂ - y₂)) ^ ((n - 1) / 2) % n = 1

-- Define Britta's strategy
def BrittaStrategy (n : ℕ) (h : Game n) : Prop :=
  ∀ (x₁ x₂ : ℕ), x₁ < n ∧ x₂ < n ∧ x₁ ≠ x₂ →
  ∃ (y₁ y₂ : ℕ), y₁ < n ∧ y₂ < n ∧ y₁ ≠ y₂ ∧ BrittaWins n x₁ x₂ y₁ y₂

-- Theorem statement
theorem britta_winning_strategy (n : ℕ) (h : Game n) :
  BrittaStrategy n h ↔ Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_britta_winning_strategy_l4137_413736


namespace NUMINAMATH_CALUDE_probability_at_least_one_tenth_grade_l4137_413750

/-- The number of volunteers from the 10th grade -/
def tenth_grade_volunteers : ℕ := 2

/-- The number of volunteers from the 11th grade -/
def eleventh_grade_volunteers : ℕ := 4

/-- The total number of volunteers -/
def total_volunteers : ℕ := tenth_grade_volunteers + eleventh_grade_volunteers

/-- The number of volunteers to be selected -/
def selected_volunteers : ℕ := 2

/-- The probability of selecting at least one volunteer from the 10th grade -/
theorem probability_at_least_one_tenth_grade :
  (1 : ℚ) - (Nat.choose eleventh_grade_volunteers selected_volunteers : ℚ) / 
  (Nat.choose total_volunteers selected_volunteers : ℚ) = 3/5 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_tenth_grade_l4137_413750


namespace NUMINAMATH_CALUDE_diving_class_capacity_is_270_l4137_413786

/-- The number of people who can take diving classes in 3 weeks -/
def diving_class_capacity : ℕ :=
  let weekday_classes_per_day : ℕ := 2
  let weekend_classes_per_day : ℕ := 4
  let weekdays_per_week : ℕ := 5
  let weekend_days_per_week : ℕ := 2
  let people_per_class : ℕ := 5
  let weeks : ℕ := 3

  let weekday_classes_per_week : ℕ := weekday_classes_per_day * weekdays_per_week
  let weekend_classes_per_week : ℕ := weekend_classes_per_day * weekend_days_per_week
  let total_classes_per_week : ℕ := weekday_classes_per_week + weekend_classes_per_week
  let people_per_week : ℕ := total_classes_per_week * people_per_class
  
  people_per_week * weeks

/-- Theorem stating that the diving class capacity for 3 weeks is 270 people -/
theorem diving_class_capacity_is_270 : diving_class_capacity = 270 := by
  sorry

end NUMINAMATH_CALUDE_diving_class_capacity_is_270_l4137_413786


namespace NUMINAMATH_CALUDE_sum_is_five_digits_l4137_413763

/-- A nonzero digit is a natural number between 1 and 9, inclusive. -/
def NonzeroDigit : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- Convert a nonzero digit to a natural number. -/
def to_nat (d : NonzeroDigit) : ℕ := d.val

/-- The first number in the sum. -/
def num1 : ℕ := 59876

/-- The second number in the sum, parameterized by a nonzero digit A. -/
def num2 (A : NonzeroDigit) : ℕ := 1000 + 100 * (to_nat A) + 32

/-- The third number in the sum, parameterized by a nonzero digit B. -/
def num3 (B : NonzeroDigit) : ℕ := 10 * (to_nat B) + 1

/-- The sum of the three numbers. -/
def total_sum (A B : NonzeroDigit) : ℕ := num1 + num2 A + num3 B

/-- A number is a 5-digit number if it's between 10000 and 99999, inclusive. -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem sum_is_five_digits (A B : NonzeroDigit) : is_five_digit (total_sum A B) := by
  sorry

end NUMINAMATH_CALUDE_sum_is_five_digits_l4137_413763


namespace NUMINAMATH_CALUDE_frisbee_price_problem_l4137_413792

/-- Proves that the certain price of frisbees is $4 given the problem conditions -/
theorem frisbee_price_problem (total_frisbees : ℕ) (price_some : ℝ) (price_rest : ℝ) 
  (total_receipts : ℝ) (min_at_price_rest : ℕ) :
  total_frisbees = 60 →
  price_some = 3 →
  total_receipts = 200 →
  min_at_price_rest = 20 →
  price_rest = 4 := by
  sorry

end NUMINAMATH_CALUDE_frisbee_price_problem_l4137_413792


namespace NUMINAMATH_CALUDE_square_perimeter_square_perimeter_holds_l4137_413759

/-- The perimeter of a square with side length 7 meters is 28 meters. -/
theorem square_perimeter : ℝ → Prop :=
  fun side_length =>
    side_length = 7 → 4 * side_length = 28

/-- The theorem holds for the given side length. -/
theorem square_perimeter_holds : square_perimeter 7 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_square_perimeter_holds_l4137_413759


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_not_square_of_radii_ratio_l4137_413784

theorem sphere_volume_ratio_not_square_of_radii_ratio (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) :
  (4 * π * r₁^3 / 3) / (4 * π * r₂^3 / 3) ≠ (r₁ / r₂)^2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_not_square_of_radii_ratio_l4137_413784


namespace NUMINAMATH_CALUDE_equation_root_condition_l4137_413705

theorem equation_root_condition (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ |x| = a * x + 1 ∧ (∀ y : ℝ, y > 0 → |y| ≠ a * y + 1)) → 
  a > -1 := by
sorry

end NUMINAMATH_CALUDE_equation_root_condition_l4137_413705


namespace NUMINAMATH_CALUDE_closest_fraction_is_one_sixth_l4137_413743

def medals_won : ℚ := 17 / 100

def possible_fractions : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]

theorem closest_fraction_is_one_sixth :
  ∀ x ∈ possible_fractions, x ≠ 1/6 → |medals_won - 1/6| < |medals_won - x| :=
sorry

end NUMINAMATH_CALUDE_closest_fraction_is_one_sixth_l4137_413743


namespace NUMINAMATH_CALUDE_color_theorem_l4137_413735

theorem color_theorem :
  ∃ (f : ℕ → ℕ),
    (∀ x, x ∈ Finset.range 2013 → f x ∈ Finset.range 7) ∧
    (∀ y, y ∈ Finset.range 7 → ∃ x ∈ Finset.range 2013, f x = y) ∧
    (∀ a b c, a ∈ Finset.range 2013 → b ∈ Finset.range 2013 → c ∈ Finset.range 2013 →
      a ≠ b → b ≠ c → a ≠ c → f a = f b → f b = f c →
        ¬(2014 ∣ (a * b * c)) ∧
        f ((a * b * c) % 2014) = f a) :=
by sorry

end NUMINAMATH_CALUDE_color_theorem_l4137_413735


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l4137_413753

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 18 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 18 → (x : ℤ) + y ≤ (a : ℤ) + b) →
  (x : ℤ) + y = 75 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l4137_413753


namespace NUMINAMATH_CALUDE_smallest_perfect_square_with_remainders_l4137_413724

theorem smallest_perfect_square_with_remainders : ∃ n : ℕ, 
  n > 1 ∧
  n % 3 = 2 ∧
  n % 7 = 2 ∧
  n % 8 = 2 ∧
  ∃ k : ℕ, n = k^2 ∧
  ∀ m : ℕ, m > 1 → m % 3 = 2 → m % 7 = 2 → m % 8 = 2 → (∃ j : ℕ, m = j^2) → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_with_remainders_l4137_413724


namespace NUMINAMATH_CALUDE_tension_in_rope_l4137_413726

/-- A system of pulleys and masses as described in the problem -/
structure PulleySystem (m : ℝ) where
  /-- The acceleration due to gravity -/
  g : ℝ
  /-- The tension in the rope connecting the bodies m and 2m through the upper pulley -/
  tension : ℝ

/-- The theorem stating the tension in the rope connecting the bodies m and 2m -/
theorem tension_in_rope (m : ℝ) (sys : PulleySystem m) (hm : m > 0) :
  sys.tension = (10 / 3) * m * sys.g := by
  sorry


end NUMINAMATH_CALUDE_tension_in_rope_l4137_413726


namespace NUMINAMATH_CALUDE_chenny_candy_problem_l4137_413723

theorem chenny_candy_problem (initial_candies : ℕ) (num_friends : ℕ) (candies_per_friend : ℕ) : 
  initial_candies = 10 →
  num_friends = 7 →
  candies_per_friend = 2 →
  num_friends * candies_per_friend - initial_candies = 4 := by
  sorry

end NUMINAMATH_CALUDE_chenny_candy_problem_l4137_413723


namespace NUMINAMATH_CALUDE_range_of_g_l4137_413794

def f (x : ℝ) : ℝ := 4 * x + 1

def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → 85 ≤ g x ∧ g x ≤ 853 :=
by sorry

end NUMINAMATH_CALUDE_range_of_g_l4137_413794


namespace NUMINAMATH_CALUDE_episode_length_l4137_413765

def total_days : ℕ := 5
def episodes : ℕ := 20
def daily_hours : ℕ := 2
def minutes_per_hour : ℕ := 60

theorem episode_length :
  (total_days * daily_hours * minutes_per_hour) / episodes = 30 := by
  sorry

end NUMINAMATH_CALUDE_episode_length_l4137_413765


namespace NUMINAMATH_CALUDE_set_equation_solution_l4137_413709

theorem set_equation_solution (A X Y : Set α) 
  (h1 : X ∪ Y = A) 
  (h2 : X ∩ A = Y) : 
  X = A ∧ Y = A := by
  sorry

end NUMINAMATH_CALUDE_set_equation_solution_l4137_413709


namespace NUMINAMATH_CALUDE_f_range_l4137_413701

-- Define the function
def f (x : ℝ) : ℝ := |x + 5| - |x - 3|

-- State the theorem
theorem f_range :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -2 ≤ y ∧ y ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_f_range_l4137_413701


namespace NUMINAMATH_CALUDE_time_spent_on_activities_l4137_413790

theorem time_spent_on_activities (hours_A hours_B : ℕ) : 
  hours_A = 6 → hours_A = hours_B + 3 → hours_B = 3 := by
  sorry

end NUMINAMATH_CALUDE_time_spent_on_activities_l4137_413790


namespace NUMINAMATH_CALUDE_total_books_is_14_l4137_413727

/-- The number of books a librarian takes away. -/
def librarian_books : ℕ := 2

/-- The number of books that can fit on a shelf. -/
def books_per_shelf : ℕ := 3

/-- The number of shelves Roger needs. -/
def shelves_needed : ℕ := 4

/-- The total number of books to put away. -/
def total_books : ℕ := librarian_books + books_per_shelf * shelves_needed

theorem total_books_is_14 : total_books = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_14_l4137_413727


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l4137_413781

theorem inequality_and_equality_condition (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt 2 * (Real.sqrt (a * (a + b)^3) + b * Real.sqrt (a^2 + b^2)) ≤ 3 * (a^2 + b^2) ∧
  (Real.sqrt 2 * (Real.sqrt (a * (a + b)^3) + b * Real.sqrt (a^2 + b^2)) = 3 * (a^2 + b^2) ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l4137_413781


namespace NUMINAMATH_CALUDE_new_ratio_is_one_to_two_l4137_413739

/-- Represents the coin collection --/
structure CoinCollection where
  gold : ℕ
  silver : ℕ

/-- The initial state of the coin collection --/
def initial_collection : CoinCollection :=
  { gold := 0, silver := 0 }

/-- The condition that initially there is one gold coin for every 3 silver coins --/
axiom initial_ratio (c : CoinCollection) : c.gold * 3 = c.silver

/-- The operation of adding 15 gold coins to the collection --/
def add_gold_coins (c : CoinCollection) : CoinCollection :=
  { gold := c.gold + 15, silver := c.silver }

/-- The total number of coins after adding 15 gold coins is 135 --/
axiom total_coins_after_addition (c : CoinCollection) :
  (add_gold_coins c).gold + (add_gold_coins c).silver = 135

/-- Theorem stating that the new ratio of gold to silver coins is 1:2 --/
theorem new_ratio_is_one_to_two (c : CoinCollection) :
  2 * (add_gold_coins c).gold = (add_gold_coins c).silver :=
sorry

end NUMINAMATH_CALUDE_new_ratio_is_one_to_two_l4137_413739


namespace NUMINAMATH_CALUDE_percent_greater_relative_to_sum_l4137_413752

/-- Given two real numbers M and N, this theorem states that the percentage
    by which M is greater than N, relative to their sum, is (100(M-N))/(M+N). -/
theorem percent_greater_relative_to_sum (M N : ℝ) :
  (M - N) / (M + N) * 100 = (100 * (M - N)) / (M + N) := by sorry

end NUMINAMATH_CALUDE_percent_greater_relative_to_sum_l4137_413752


namespace NUMINAMATH_CALUDE_function_inequality_condition_l4137_413711

/-- A function f(x) = ax^2 + b satisfies f(xy) + f(x + y) ≥ f(x) f(y) for all real x and y
    if and only if 0 < b ≤ 1, 0 < a < 1, and 2a + b ≤ 2 -/
theorem function_inequality_condition (a b : ℝ) : 
  (∀ x y : ℝ, a * (x * y)^2 + b + a * (x + y)^2 + b ≥ (a * x^2 + b) * (a * y^2 + b)) ↔ 
  (0 < b ∧ b ≤ 1 ∧ 0 < a ∧ a < 1 ∧ 2 * a + b ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l4137_413711


namespace NUMINAMATH_CALUDE_ratio_to_percentage_increase_l4137_413720

theorem ratio_to_percentage_increase (A B : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : A / B = 1/6 / (1/5)) :
  (B - A) / A * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percentage_increase_l4137_413720


namespace NUMINAMATH_CALUDE_half_sum_of_consecutive_odd_primes_is_composite_l4137_413741

/-- Two natural numbers are consecutive primes if they are both prime and there are no primes between them. -/
def ConsecutivePrimes (p p' : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime p' ∧ p < p' ∧ ∀ q, p < q → q < p' → ¬Nat.Prime q

/-- A natural number is composite if it's greater than 1 and not prime. -/
def Composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬Nat.Prime n

theorem half_sum_of_consecutive_odd_primes_is_composite
  (p p' : ℕ) (h : ConsecutivePrimes p p') (hp_odd : Odd p) (hp'_odd : Odd p') (hp_ge_3 : p ≥ 3) :
  Composite ((p + p') / 2) :=
sorry

end NUMINAMATH_CALUDE_half_sum_of_consecutive_odd_primes_is_composite_l4137_413741


namespace NUMINAMATH_CALUDE_savings_account_interest_rate_l4137_413793

theorem savings_account_interest_rate (initial_deposit : ℝ) (balance_after_first_year : ℝ) (total_increase_percentage : ℝ) : 
  initial_deposit = 5000 →
  balance_after_first_year = 5500 →
  total_increase_percentage = 21 →
  let total_balance := initial_deposit * (1 + total_increase_percentage / 100)
  let increase_second_year := total_balance - balance_after_first_year
  let percentage_increase_second_year := (increase_second_year / balance_after_first_year) * 100
  percentage_increase_second_year = 10 := by
sorry

end NUMINAMATH_CALUDE_savings_account_interest_rate_l4137_413793


namespace NUMINAMATH_CALUDE_greatest_common_divisor_540_462_l4137_413721

theorem greatest_common_divisor_540_462 : Nat.gcd 540 462 = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_540_462_l4137_413721


namespace NUMINAMATH_CALUDE_bernards_blue_notebooks_l4137_413748

/-- Represents the number of notebooks Bernard had -/
structure BernardsNotebooks where
  red : ℕ
  white : ℕ
  blue : ℕ
  given : ℕ
  left : ℕ

/-- Theorem stating the number of blue notebooks Bernard had -/
theorem bernards_blue_notebooks
  (notebooks : BernardsNotebooks)
  (h_red : notebooks.red = 15)
  (h_white : notebooks.white = 19)
  (h_given : notebooks.given = 46)
  (h_left : notebooks.left = 5)
  (h_total : notebooks.red + notebooks.white + notebooks.blue = notebooks.given + notebooks.left) :
  notebooks.blue = 17 := by
  sorry

end NUMINAMATH_CALUDE_bernards_blue_notebooks_l4137_413748


namespace NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l4137_413760

theorem sum_and_ratio_to_difference (x y : ℝ) :
  x + y = 520 → x / y = 0.75 → y - x = 74 := by sorry

end NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l4137_413760


namespace NUMINAMATH_CALUDE_tv_weight_difference_l4137_413754

/-- The difference in weight between two TVs with given dimensions -/
theorem tv_weight_difference : 
  let bill_length : ℕ := 48
  let bill_width : ℕ := 100
  let bob_length : ℕ := 70
  let bob_width : ℕ := 60
  let weight_per_sq_inch : ℚ := 4 / 1
  let oz_per_pound : ℕ := 16
  let bill_area : ℕ := bill_length * bill_width
  let bob_area : ℕ := bob_length * bob_width
  let bill_weight_oz : ℚ := bill_area * weight_per_sq_inch
  let bob_weight_oz : ℚ := bob_area * weight_per_sq_inch
  let bill_weight_lbs : ℚ := bill_weight_oz / oz_per_pound
  let bob_weight_lbs : ℚ := bob_weight_oz / oz_per_pound
  bill_weight_lbs - bob_weight_lbs = 150
  := by sorry

end NUMINAMATH_CALUDE_tv_weight_difference_l4137_413754


namespace NUMINAMATH_CALUDE_win_sector_area_l4137_413710

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 3/7) :
  let total_area := π * r^2
  let win_area := p * total_area
  win_area = 192 * π / 7 := by
sorry

end NUMINAMATH_CALUDE_win_sector_area_l4137_413710


namespace NUMINAMATH_CALUDE_river_boat_journey_time_l4137_413782

theorem river_boat_journey_time 
  (river_speed : ℝ) 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (h1 : river_speed = 2) 
  (h2 : boat_speed = 6) 
  (h3 : distance = 32) : 
  (distance / (boat_speed - river_speed) + distance / (boat_speed + river_speed)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_river_boat_journey_time_l4137_413782


namespace NUMINAMATH_CALUDE_circumcircle_incircle_diameter_implies_equilateral_l4137_413738

-- Define a triangle
structure Triangle where
  -- We don't need to specify the vertices, just that it's a triangle
  is_triangle : Bool

-- Define the circumcircle and incircle of a triangle
def circumcircle (t : Triangle) : ℝ := sorry
def incircle (t : Triangle) : ℝ := sorry

-- Define what it means for a triangle to be equilateral
def is_equilateral (t : Triangle) : Prop := sorry

-- State the theorem
theorem circumcircle_incircle_diameter_implies_equilateral (t : Triangle) :
  circumcircle t = 2 * incircle t → is_equilateral t := by
  sorry


end NUMINAMATH_CALUDE_circumcircle_incircle_diameter_implies_equilateral_l4137_413738


namespace NUMINAMATH_CALUDE_divisors_of_48n5_l4137_413799

/-- Given a positive integer n where 132n^3 has 132 positive integer divisors,
    48n^5 has 105 positive integer divisors -/
theorem divisors_of_48n5 (n : ℕ+) (h : (Nat.divisors (132 * n ^ 3)).card = 132) :
  (Nat.divisors (48 * n ^ 5)).card = 105 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_48n5_l4137_413799


namespace NUMINAMATH_CALUDE_product_of_roots_quadratic_l4137_413767

theorem product_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ - 4 = 0) → (x₂^2 + 2*x₂ - 4 = 0) → x₁ * x₂ = -4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_quadratic_l4137_413767


namespace NUMINAMATH_CALUDE_flour_weight_relation_l4137_413798

/-- Theorem: Given two equations representing the weight of flour bags, 
    prove that the new combined weight is equal to the original weight plus 33 pounds. -/
theorem flour_weight_relation (x y : ℝ) : 
  y = (16 - 4) + (30 - 6) + (x - 3) → 
  y = 12 + 24 + (x - 3) → 
  y = x + 33 := by
  sorry

end NUMINAMATH_CALUDE_flour_weight_relation_l4137_413798


namespace NUMINAMATH_CALUDE_unique_zero_point_l4137_413742

open Real

noncomputable def f (x : ℝ) := exp x + x - 2 * exp 1

theorem unique_zero_point :
  ∃! x, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_zero_point_l4137_413742


namespace NUMINAMATH_CALUDE_intersection_line_equation_l4137_413700

/-- Given two lines l₁ and l₂ that intersect to form a line segment with midpoint P(0, 0),
    prove that the line l passing through their intersection points has equation y = 7/6 * x. -/
theorem intersection_line_equation 
  (l₁ : Set (ℝ × ℝ)) 
  (l₂ : Set (ℝ × ℝ)) 
  (h₁ : l₁ = {(x, y) | 4 * x + y + 6 = 0})
  (h₂ : l₂ = {(x, y) | 3 * x - 5 * y - 6 = 0})
  (h_midpoint : ∃ (a b : ℝ × ℝ), a ∈ l₁ ∧ a ∈ l₂ ∧ b ∈ l₁ ∧ b ∈ l₂ ∧ (0, 0) = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)) :
  ∃ (l : Set (ℝ × ℝ)), l = {(x, y) | y = 7/6 * x} ∧ 
    ∀ (p : ℝ × ℝ), (p ∈ l₁ ∧ p ∈ l₂) → p ∈ l :=
sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l4137_413700


namespace NUMINAMATH_CALUDE_program_result_l4137_413749

theorem program_result : ∀ (x : ℕ), 
  x = 51 → 
  9 < x → 
  x < 100 → 
  let a := x / 10
  let b := x % 10
  10 * b + a = 15 := by
sorry

end NUMINAMATH_CALUDE_program_result_l4137_413749


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l4137_413756

theorem complex_number_quadrant : 
  let z : ℂ := (1 + 2*I) / (1 - I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l4137_413756


namespace NUMINAMATH_CALUDE_quadratic_root_l4137_413744

theorem quadratic_root (k : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - 3 * x - k = 0 ∧ x = 1) → 
  (∃ y : ℝ, 2 * y^2 - 3 * y - k = 0 ∧ y = 1/2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_l4137_413744


namespace NUMINAMATH_CALUDE_expected_heads_is_60_l4137_413788

/-- The number of coins -/
def num_coins : ℕ := 64

/-- The maximum number of flips per coin -/
def max_flips : ℕ := 4

/-- The probability of getting heads on a single flip -/
def p_heads : ℚ := 1/2

/-- The probability of getting heads after up to four flips -/
def p_heads_total : ℚ := 1 - (1 - p_heads)^max_flips

/-- The expected number of coins showing heads after up to four flips -/
def expected_heads : ℚ := num_coins * p_heads_total

theorem expected_heads_is_60 : expected_heads = 60 := by
  sorry

end NUMINAMATH_CALUDE_expected_heads_is_60_l4137_413788


namespace NUMINAMATH_CALUDE_finite_selector_existence_l4137_413789

theorem finite_selector_existence
  (A B C : ℕ → Set ℕ)
  (h_finite : ∀ i, (A i).Finite ∧ (B i).Finite ∧ (C i).Finite)
  (h_disjoint : ∀ i, Disjoint (A i) (B i) ∧ Disjoint (A i) (C i) ∧ Disjoint (B i) (C i))
  (h_cover : ∀ X Y Z : Set ℕ, Disjoint X Y ∧ Disjoint X Z ∧ Disjoint Y Z → X ∪ Y ∪ Z = univ →
    ∃ i, A i ⊆ X ∧ B i ⊆ Y ∧ C i ⊆ Z) :
  ∃ S : Finset ℕ, ∀ X Y Z : Set ℕ, Disjoint X Y ∧ Disjoint X Z ∧ Disjoint Y Z → X ∪ Y ∪ Z = univ →
    ∃ i ∈ S, A i ⊆ X ∧ B i ⊆ Y ∧ C i ⊆ Z :=
by sorry

end NUMINAMATH_CALUDE_finite_selector_existence_l4137_413789


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l4137_413715

theorem triangle_third_side_length (a b c : ℝ) (θ : ℝ) (h1 : a = 9) (h2 : b = 11) (h3 : θ = 150 * π / 180) :
  c^2 = a^2 + b^2 - 2*a*b*Real.cos θ → c = Real.sqrt (202 + 99 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l4137_413715


namespace NUMINAMATH_CALUDE_max_value_is_57_l4137_413728

/-- Represents a type of rock with its weight and value -/
structure Rock where
  weight : Nat
  value : Nat

/-- The problem setup -/
def rockTypes : List Rock := [
  { weight := 6, value := 18 },
  { weight := 3, value := 9 },
  { weight := 2, value := 3 }
]

/-- The maximum weight Carl can carry -/
def maxWeight : Nat := 20

/-- The minimum number of rocks available for each type -/
def minRocksPerType : Nat := 15

/-- A function to calculate the total value of a collection of rocks -/
def totalValue (rocks : List (Rock × Nat)) : Nat :=
  rocks.foldl (fun acc (rock, count) => acc + rock.value * count) 0

/-- A function to calculate the total weight of a collection of rocks -/
def totalWeight (rocks : List (Rock × Nat)) : Nat :=
  rocks.foldl (fun acc (rock, count) => acc + rock.weight * count) 0

/-- The main theorem stating that the maximum value Carl can carry is $57 -/
theorem max_value_is_57 :
  ∃ (rocks : List (Rock × Nat)),
    (∀ r ∈ rocks, r.1 ∈ rockTypes) ∧
    (∀ r ∈ rocks, r.2 ≤ minRocksPerType) ∧
    totalWeight rocks ≤ maxWeight ∧
    totalValue rocks = 57 ∧
    (∀ (other_rocks : List (Rock × Nat)),
      (∀ r ∈ other_rocks, r.1 ∈ rockTypes) →
      (∀ r ∈ other_rocks, r.2 ≤ minRocksPerType) →
      totalWeight other_rocks ≤ maxWeight →
      totalValue other_rocks ≤ 57) :=
by sorry


end NUMINAMATH_CALUDE_max_value_is_57_l4137_413728


namespace NUMINAMATH_CALUDE_terriers_groomed_count_l4137_413771

/-- Represents the time in minutes to groom a poodle -/
def poodle_groom_time : ℕ := 30

/-- Represents the time in minutes to groom a terrier -/
def terrier_groom_time : ℕ := poodle_groom_time / 2

/-- Represents the number of poodles groomed -/
def poodles_groomed : ℕ := 3

/-- Represents the total grooming time in minutes -/
def total_groom_time : ℕ := 210

/-- Proves that the number of terriers groomed is 8 -/
theorem terriers_groomed_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_terriers_groomed_count_l4137_413771


namespace NUMINAMATH_CALUDE_cone_volume_over_pi_l4137_413747

/-- Given a cone formed from a 240-degree sector of a circle with radius 16,
    prove that the volume of the cone divided by π is equal to 8192√10 / 81. -/
theorem cone_volume_over_pi (r : ℝ) (h : ℝ) :
  r = 32 / 3 →
  h = 8 * Real.sqrt 10 / 3 →
  (1 / 3 * π * r^2 * h) / π = 8192 * Real.sqrt 10 / 81 := by sorry

end NUMINAMATH_CALUDE_cone_volume_over_pi_l4137_413747


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l4137_413717

theorem largest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > n → ¬(m + 12 ∣ m^3 + 144)) ∧ 
  (n + 12 ∣ n^3 + 144) ∧ 
  n = 132 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l4137_413717


namespace NUMINAMATH_CALUDE_mike_total_spent_l4137_413766

def trumpet_cost : ℚ := 145.16
def songbook_cost : ℚ := 5.84

theorem mike_total_spent :
  trumpet_cost + songbook_cost = 151 := by sorry

end NUMINAMATH_CALUDE_mike_total_spent_l4137_413766


namespace NUMINAMATH_CALUDE_range_of_m_l4137_413797

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, m * x^2 + 1 ≤ 0) ∧ 
  (¬ ∃ x : ℝ, x^2 + m * x + 1 < 0) → 
  0 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l4137_413797


namespace NUMINAMATH_CALUDE_sum_of_divisors_36_l4137_413785

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_36 : sum_of_divisors 36 = 91 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_36_l4137_413785


namespace NUMINAMATH_CALUDE_counterexample_five_l4137_413718

theorem counterexample_five : 
  ∃ n : ℕ, ¬(3 ∣ n) ∧ ¬(Prime (n^2 - 1)) ∧ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_counterexample_five_l4137_413718


namespace NUMINAMATH_CALUDE_octagon_perimeter_in_cm_l4137_413796

/-- Regular octagon with side length in meters -/
structure RegularOctagon where
  side_length : ℝ

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

/-- Sum of all side lengths of a regular octagon in centimeters -/
def sum_side_lengths (octagon : RegularOctagon) : ℝ :=
  8 * octagon.side_length * meters_to_cm

theorem octagon_perimeter_in_cm (octagon : RegularOctagon) 
    (h : octagon.side_length = 2.3) : 
    sum_side_lengths octagon = 1840 := by
  sorry

end NUMINAMATH_CALUDE_octagon_perimeter_in_cm_l4137_413796


namespace NUMINAMATH_CALUDE_power_division_l4137_413729

theorem power_division (a : ℝ) (h : a ≠ 0) : a^3 / a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l4137_413729


namespace NUMINAMATH_CALUDE_rational_function_sum_l4137_413780

-- Define p(x) and q(x) as functions
variable (p q : ℝ → ℝ)

-- State the conditions
variable (h1 : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c)
variable (h2 : q 2 = 0 ∧ q 4 = 0)
variable (h3 : p 1 = 2)
variable (h4 : q 3 = 3)

-- State the theorem
theorem rational_function_sum :
  ∃ f : ℝ → ℝ, (∀ x, f x = p x + q x) ∧ (∀ x, f x = -3 * x^2 + 18 * x - 22) :=
sorry

end NUMINAMATH_CALUDE_rational_function_sum_l4137_413780


namespace NUMINAMATH_CALUDE_calculate_expression_l4137_413769

theorem calculate_expression : (π - 2) ^ 0 - (-2) ^ (-1 : ℤ) + |Real.sqrt 3 - 2| = 7 / 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l4137_413769


namespace NUMINAMATH_CALUDE_money_saved_monthly_payment_l4137_413733

/-- Calculates the money saved by paying monthly instead of weekly for a hotel stay. -/
theorem money_saved_monthly_payment (weekly_rate : ℕ) (monthly_rate : ℕ) (num_months : ℕ) 
  (h1 : weekly_rate = 280)
  (h2 : monthly_rate = 1000)
  (h3 : num_months = 3) :
  weekly_rate * 4 * num_months - monthly_rate * num_months = 360 := by
  sorry

#check money_saved_monthly_payment

end NUMINAMATH_CALUDE_money_saved_monthly_payment_l4137_413733


namespace NUMINAMATH_CALUDE_number_problem_l4137_413707

theorem number_problem (x : ℝ) : 0.4 * x + 60 = x → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4137_413707


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l4137_413791

theorem p_sufficient_not_necessary_for_q :
  (∃ x, 0 < x ∧ x < 5 ∧ ¬(-1 < x ∧ x < 5)) = False ∧
  (∃ x, -1 < x ∧ x < 5 ∧ ¬(0 < x ∧ x < 5)) = True := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l4137_413791


namespace NUMINAMATH_CALUDE_samantha_overall_percentage_l4137_413770

/-- Represents an exam with its number of questions, weight per question, and percentage correct --/
structure Exam where
  questions : ℕ
  weight : ℕ
  percentCorrect : ℚ

/-- Calculates the total weighted questions for an exam --/
def totalWeightedQuestions (e : Exam) : ℚ :=
  (e.questions * e.weight : ℚ)

/-- Calculates the number of weighted questions answered correctly for an exam --/
def weightedQuestionsCorrect (e : Exam) : ℚ :=
  e.percentCorrect * totalWeightedQuestions e

/-- Calculates the overall percentage of weighted questions answered correctly across multiple exams --/
def overallPercentageCorrect (exams : List Exam) : ℚ :=
  let totalCorrect := (exams.map weightedQuestionsCorrect).sum
  let totalQuestions := (exams.map totalWeightedQuestions).sum
  totalCorrect / totalQuestions

/-- The three exams Samantha took --/
def samanthasExams : List Exam :=
  [{ questions := 30, weight := 1, percentCorrect := 75/100 },
   { questions := 50, weight := 1, percentCorrect := 80/100 },
   { questions := 20, weight := 2, percentCorrect := 65/100 }]

theorem samantha_overall_percentage :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  |overallPercentageCorrect samanthasExams - 74/100| < ε :=
sorry

end NUMINAMATH_CALUDE_samantha_overall_percentage_l4137_413770


namespace NUMINAMATH_CALUDE_power_function_property_l4137_413704

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 3 = Real.sqrt 3) : 
  f 9 = 3 := by sorry

end NUMINAMATH_CALUDE_power_function_property_l4137_413704


namespace NUMINAMATH_CALUDE_bag_volume_proof_l4137_413774

/-- The volume of a cuboid-shaped bag -/
def bag_volume (width length height : ℝ) : ℝ := width * length * height

/-- Theorem: The volume of a cuboid-shaped bag with width 9 cm, length 4 cm, and height 7 cm is 252 cm³ -/
theorem bag_volume_proof : bag_volume 9 4 7 = 252 := by
  sorry

end NUMINAMATH_CALUDE_bag_volume_proof_l4137_413774


namespace NUMINAMATH_CALUDE_unsold_books_percentage_l4137_413758

/-- Calculates the percentage of unsold books in a bookshop -/
theorem unsold_books_percentage 
  (initial_stock : ℕ) 
  (monday_sales tuesday_sales wednesday_sales thursday_sales friday_sales : ℕ) : 
  initial_stock = 700 →
  monday_sales = 50 →
  tuesday_sales = 82 →
  wednesday_sales = 60 →
  thursday_sales = 48 →
  friday_sales = 40 →
  (initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales)) / initial_stock * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_unsold_books_percentage_l4137_413758


namespace NUMINAMATH_CALUDE_no_intersection_l4137_413745

def f (x : ℝ) := |3 * x + 6|
def g (x : ℝ) := -|4 * x - 3|

theorem no_intersection :
  ∀ x : ℝ, f x ≠ g x :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l4137_413745


namespace NUMINAMATH_CALUDE_cricket_bat_price_l4137_413779

theorem cricket_bat_price (profit_A_to_B : ℝ) (profit_B_to_C : ℝ) (price_C : ℝ) : 
  profit_A_to_B = 0.20 →
  profit_B_to_C = 0.25 →
  price_C = 222 →
  ∃ (cost_price_A : ℝ), cost_price_A = 148 ∧ 
    price_C = cost_price_A * (1 + profit_A_to_B) * (1 + profit_B_to_C) :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_price_l4137_413779


namespace NUMINAMATH_CALUDE_point_satisfies_inequalities_l4137_413783

-- Define the system of inequalities
def satisfies_inequalities (x y : ℝ) : Prop :=
  (x - 2*y + 5 > 0) ∧ (x - y + 3 ≤ 0)

-- Theorem statement
theorem point_satisfies_inequalities : 
  satisfies_inequalities (-2 : ℝ) (1 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_point_satisfies_inequalities_l4137_413783


namespace NUMINAMATH_CALUDE_complex_point_on_line_l4137_413734

theorem complex_point_on_line (a : ℝ) : 
  let z : ℂ := (a - Complex.I)⁻¹
  (z.im = 2 * z.re) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_on_line_l4137_413734


namespace NUMINAMATH_CALUDE_x_squared_coefficient_l4137_413795

def expansion (x : ℝ) := (2*x + 1) * (x - 2)^3

theorem x_squared_coefficient : 
  (∃ a b c d : ℝ, expansion x = a*x^3 + b*x^2 + c*x + d) → 
  (∃ a c d : ℝ, expansion x = a*x^3 + 18*x^2 + c*x + d) :=
by sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_l4137_413795


namespace NUMINAMATH_CALUDE_max_sum_of_coefficients_l4137_413722

theorem max_sum_of_coefficients (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ A B : ℝ × ℝ, 
    (a * A.1 + b * A.2 = 1) ∧ 
    (a * B.1 + b * B.2 = 1) ∧ 
    (A.1^2 + A.2^2 = 1) ∧ 
    (B.1^2 + B.2^2 = 1) ∧ 
    (A ≠ B)) →
  (∃ A B : ℝ × ℝ, 
    (a * A.1 + b * A.2 = 1) ∧ 
    (a * B.1 + b * B.2 = 1) ∧ 
    (A.1^2 + A.2^2 = 1) ∧ 
    (B.1^2 + B.2^2 = 1) ∧ 
    (abs (A.1 * B.2 - A.2 * B.1) = 1)) →
  a + b ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_coefficients_l4137_413722


namespace NUMINAMATH_CALUDE_angle_C_measure_l4137_413703

def triangle_ABC (A B C : ℝ) := 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

theorem angle_C_measure (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : triangle_ABC A B C)
  (h_AC : c = Real.sqrt 6)
  (h_BC : b = 2)
  (h_angle_B : B = Real.pi / 3) :
  C = 5 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l4137_413703


namespace NUMINAMATH_CALUDE_expression_value_l4137_413732

theorem expression_value (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = -1) :
  3 * x^2 - 4 * y + 5 * z = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4137_413732


namespace NUMINAMATH_CALUDE_complex_power_result_l4137_413746

theorem complex_power_result (n : ℕ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (1 + 3)^n = 256) : (1 + i : ℂ)^n = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_result_l4137_413746


namespace NUMINAMATH_CALUDE_tamikas_speed_l4137_413702

/-- Tamika's driving problem -/
theorem tamikas_speed (tamika_time logan_time logan_speed extra_distance : ℝ) 
  (h1 : tamika_time = 8)
  (h2 : logan_time = 5)
  (h3 : logan_speed = 55)
  (h4 : extra_distance = 85)
  : (logan_time * logan_speed + extra_distance) / tamika_time = 45 := by
  sorry

#check tamikas_speed

end NUMINAMATH_CALUDE_tamikas_speed_l4137_413702


namespace NUMINAMATH_CALUDE_latia_work_hours_l4137_413764

/-- Proves that Latia works 30 hours per week given the problem conditions -/
theorem latia_work_hours :
  ∀ (tv_price : ℕ) (hourly_rate : ℕ) (additional_hours : ℕ) (weeks_per_month : ℕ),
  tv_price = 1700 →
  hourly_rate = 10 →
  additional_hours = 50 →
  weeks_per_month = 4 →
  ∃ (hours_per_week : ℕ),
    hours_per_week * weeks_per_month * hourly_rate + additional_hours * hourly_rate = tv_price ∧
    hours_per_week = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_latia_work_hours_l4137_413764


namespace NUMINAMATH_CALUDE_vector_equation_holds_l4137_413730

variable {V : Type*} [AddCommGroup V]

/-- Given points A, B, C, M, O in a vector space, 
    prove that AB + MB + BC + OM + CO = AB --/
theorem vector_equation_holds (A B C M O : V) :
  (A - B) + (M - B) + (B - C) + (O - M) + (C - O) = A - B :=
by sorry

end NUMINAMATH_CALUDE_vector_equation_holds_l4137_413730


namespace NUMINAMATH_CALUDE_exam_score_problem_l4137_413776

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) 
  (h1 : total_questions = 80)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -1)
  (h4 : total_score = 120) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * (correct_answers : ℤ) + wrong_score * ((total_questions - correct_answers) : ℤ) = total_score ∧
    correct_answers = 40 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_problem_l4137_413776


namespace NUMINAMATH_CALUDE_square_plus_one_representation_l4137_413777

theorem square_plus_one_representation (x y z : ℕ+) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), x = a^2 + b^2 ∧ y = c^2 + d^2 ∧ z = a * c + b * d :=
sorry

end NUMINAMATH_CALUDE_square_plus_one_representation_l4137_413777


namespace NUMINAMATH_CALUDE_equidistant_implies_d_squared_l4137_413755

/-- A complex function g that scales by a complex number c+di -/
def g (c d : ℝ) (z : ℂ) : ℂ := (c + d * Complex.I) * z

/-- The property that g(z) is equidistant from z and the origin for all z -/
def equidistant (c d : ℝ) : Prop :=
  ∀ z : ℂ, Complex.abs (g c d z - z) = Complex.abs (g c d z)

theorem equidistant_implies_d_squared (c d : ℝ) 
  (h1 : equidistant c d) 
  (h2 : Complex.abs (c + d * Complex.I) = 5) : 
  d^2 = 99/4 := by sorry

end NUMINAMATH_CALUDE_equidistant_implies_d_squared_l4137_413755


namespace NUMINAMATH_CALUDE_max_value_problem_l4137_413751

theorem max_value_problem (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (a / (a + 1)) + (b / (b + 2)) ≤ (5 - 2 * Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_problem_l4137_413751
