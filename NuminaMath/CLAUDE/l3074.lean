import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_pentagon_angles_l3074_307453

theorem sum_of_pentagon_angles : ∀ (A B C D E : ℝ),
  A + B + C + D + E = 180 * 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_pentagon_angles_l3074_307453


namespace NUMINAMATH_CALUDE_potato_rows_count_l3074_307446

/-- Represents the farmer's crop situation -/
structure FarmCrops where
  corn_rows : ℕ
  potato_rows : ℕ
  corn_per_row : ℕ
  potatoes_per_row : ℕ
  intact_crops : ℕ

/-- Theorem stating the number of potato rows given the problem conditions -/
theorem potato_rows_count (farm : FarmCrops)
    (h_corn_rows : farm.corn_rows = 10)
    (h_corn_per_row : farm.corn_per_row = 9)
    (h_potatoes_per_row : farm.potatoes_per_row = 30)
    (h_intact_crops : farm.intact_crops = 120)
    (h_half_destroyed : farm.intact_crops = (farm.corn_rows * farm.corn_per_row + farm.potato_rows * farm.potatoes_per_row) / 2) :
  farm.potato_rows = 2 := by
  sorry


end NUMINAMATH_CALUDE_potato_rows_count_l3074_307446


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l3074_307444

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l3074_307444


namespace NUMINAMATH_CALUDE_problem_solution_l3074_307452

theorem problem_solution (a : ℝ) (h1 : a > 0) 
  (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 + 9)
  (g : ℝ → ℝ) (hg : ∀ x, g x = x^2 - 5)
  (h2 : f (g a) = 25) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3074_307452


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l3074_307451

theorem quadratic_factorization_sum (a b c d : ℤ) : 
  (∀ x : ℚ, 6 * x^2 + x - 12 = (a * x + b) * (c * x + d)) →
  |a| + |b| + |c| + |d| = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l3074_307451


namespace NUMINAMATH_CALUDE_forum_total_posts_per_day_l3074_307473

/-- Represents a question and answer forum --/
structure Forum where
  members : ℕ
  questionsPerHour : ℕ
  answerRatio : ℕ

/-- Calculates the total number of questions and answers posted in a day --/
def totalPostsPerDay (f : Forum) : ℕ :=
  let questionsPerDay := f.members * (f.questionsPerHour * 24)
  let answersPerDay := f.members * (f.questionsPerHour * f.answerRatio * 24)
  questionsPerDay + answersPerDay

/-- Theorem stating the total number of posts per day for the given forum --/
theorem forum_total_posts_per_day :
  ∃ (f : Forum), f.members = 200 ∧ f.questionsPerHour = 3 ∧ f.answerRatio = 3 ∧
  totalPostsPerDay f = 57600 :=
by
  sorry

end NUMINAMATH_CALUDE_forum_total_posts_per_day_l3074_307473


namespace NUMINAMATH_CALUDE_cafeteria_earnings_proof_l3074_307497

/-- Calculates the earnings from selling fruits in a cafeteria -/
def cafeteria_earnings (initial_apples initial_oranges : ℕ) 
                       (apple_price orange_price : ℚ) 
                       (remaining_apples remaining_oranges : ℕ) : ℚ :=
  let sold_apples := initial_apples - remaining_apples
  let sold_oranges := initial_oranges - remaining_oranges
  sold_apples * apple_price + sold_oranges * orange_price

/-- Proves that the cafeteria earns $49.00 for the given conditions -/
theorem cafeteria_earnings_proof :
  cafeteria_earnings 50 40 0.80 0.50 10 6 = 49.00 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_earnings_proof_l3074_307497


namespace NUMINAMATH_CALUDE_triangle_max_area_l3074_307429

theorem triangle_max_area (a b c A B C : ℝ) :
  a = 2 →
  (2 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C →
  (∀ b' c' A' B' C',
    a = 2 →
    (2 + b') * (Real.sin A' - Real.sin B') = (c' - b') * Real.sin C' →
    a * b' * Real.sin C' / 2 ≤ Real.sqrt 3) →
  a * b * Real.sin C / 2 = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3074_307429


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3074_307460

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 4 → x ≥ 4) ∧ (∃ x, x ≥ 4 ∧ ¬(x > 4)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3074_307460


namespace NUMINAMATH_CALUDE_sum_of_squares_l3074_307487

-- Define the triangle FAC
structure Triangle :=
  (F A C : ℝ × ℝ)

-- Define the property of right angle FAC
def isRightAngle (t : Triangle) : Prop :=
  -- This is a placeholder for the right angle condition
  sorry

-- Define the length of CF
def CF_length (t : Triangle) : ℝ := 12

-- Define the area of square ACDE
def area_ACDE (t : Triangle) : ℝ :=
  let (_, A) := t.A
  let (_, C) := t.C
  (A - C) ^ 2

-- Define the area of square AFGH
def area_AFGH (t : Triangle) : ℝ :=
  let (F, _) := t.F
  let (A, _) := t.A
  (F - A) ^ 2

-- The theorem to be proved
theorem sum_of_squares (t : Triangle) 
  (h1 : isRightAngle t) 
  (h2 : CF_length t = 12) : 
  area_ACDE t + area_AFGH t = 144 :=
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3074_307487


namespace NUMINAMATH_CALUDE_equation_solution_l3074_307431

theorem equation_solution : ∃ x : ℝ, 
  (45 * x) / (3/4) = (37.5/100) * 1500 - (62.5/100) * 800 ∧ 
  abs (x - 1.0417) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3074_307431


namespace NUMINAMATH_CALUDE_absolute_value_not_always_greater_than_zero_l3074_307485

theorem absolute_value_not_always_greater_than_zero : 
  ¬ (∀ x : ℝ, |x| > 0) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_not_always_greater_than_zero_l3074_307485


namespace NUMINAMATH_CALUDE_triangle_max_area_l3074_307435

/-- Given a triangle ABC where a^2 + b^2 + 2c^2 = 8, 
    the maximum area of the triangle is 2√5/5 -/
theorem triangle_max_area (a b c : ℝ) (h : a^2 + b^2 + 2*c^2 = 8) :
  ∃ (S : ℝ), S = (2 * Real.sqrt 5) / 5 ∧ 
  (∀ (S' : ℝ), S' = (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) → S' ≤ S) := by
  sorry


end NUMINAMATH_CALUDE_triangle_max_area_l3074_307435


namespace NUMINAMATH_CALUDE_cos_angle_between_vectors_l3074_307430

/-- Given vectors a and b in ℝ², where a = (2, 1) and a + 2b = (4, 5),
    the cosine of the angle between a and b is equal to 4/5. -/
theorem cos_angle_between_vectors (a b : ℝ × ℝ) :
  a = (2, 1) →
  a + 2 • b = (4, 5) →
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_between_vectors_l3074_307430


namespace NUMINAMATH_CALUDE_james_original_weight_l3074_307471

/-- Proves that given the conditions of James's weight gain, his original weight was 120 kg -/
theorem james_original_weight :
  ∀ W : ℝ,
  W > 0 →
  let muscle_gain := 0.20 * W
  let fat_gain := 0.25 * muscle_gain
  let final_weight := W + muscle_gain + fat_gain
  final_weight = 150 →
  W = 120 := by
sorry

end NUMINAMATH_CALUDE_james_original_weight_l3074_307471


namespace NUMINAMATH_CALUDE_square_root_of_25_squared_l3074_307441

theorem square_root_of_25_squared : Real.sqrt 25 ^ 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_25_squared_l3074_307441


namespace NUMINAMATH_CALUDE_division_result_l3074_307456

theorem division_result : (45 : ℝ) / 0.05 = 900 := by sorry

end NUMINAMATH_CALUDE_division_result_l3074_307456


namespace NUMINAMATH_CALUDE_rings_cost_theorem_l3074_307498

/-- The cost of one ring in dollars -/
def cost_per_ring : ℕ := 24

/-- The number of index fingers a person has -/
def index_fingers_per_person : ℕ := 2

/-- The total cost for buying rings for all index fingers of a person -/
def total_cost : ℕ := cost_per_ring * index_fingers_per_person

theorem rings_cost_theorem : total_cost = 48 := by
  sorry

end NUMINAMATH_CALUDE_rings_cost_theorem_l3074_307498


namespace NUMINAMATH_CALUDE_problems_per_page_l3074_307424

theorem problems_per_page
  (total_problems : ℕ)
  (finished_problems : ℕ)
  (remaining_pages : ℕ)
  (h1 : total_problems = 60)
  (h2 : finished_problems = 20)
  (h3 : remaining_pages = 5)
  (h4 : remaining_pages > 0)
  : (total_problems - finished_problems) / remaining_pages = 8 := by
  sorry

end NUMINAMATH_CALUDE_problems_per_page_l3074_307424


namespace NUMINAMATH_CALUDE_cube_root_simplification_l3074_307483

theorem cube_root_simplification :
  (80^3 + 100^3 + 120^3 : ℝ)^(1/3) = 20 * 405^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l3074_307483


namespace NUMINAMATH_CALUDE_grid_selection_count_l3074_307463

theorem grid_selection_count : ℕ := by
  -- Define the size of the grid
  let n : ℕ := 6
  
  -- Define the number of blocks to select
  let k : ℕ := 4
  
  -- Define the function to calculate combinations
  let choose (n m : ℕ) := Nat.choose n m
  
  -- Define the total number of combinations
  let total_combinations := choose n k * choose n k * Nat.factorial k
  
  -- Prove that the total number of combinations is 5400
  sorry

end NUMINAMATH_CALUDE_grid_selection_count_l3074_307463


namespace NUMINAMATH_CALUDE_max_value_cosine_function_l3074_307459

theorem max_value_cosine_function (x : ℝ) :
  ∃ (k : ℤ), 2 * Real.cos x - 1 ≤ 2 * Real.cos (2 * k * Real.pi) - 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_cosine_function_l3074_307459


namespace NUMINAMATH_CALUDE_hat_shop_pricing_l3074_307400

theorem hat_shop_pricing (x : ℝ) : 
  let increased_price := 1.30 * x
  let final_price := 0.75 * increased_price
  final_price = 0.975 * x := by
sorry

end NUMINAMATH_CALUDE_hat_shop_pricing_l3074_307400


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l3074_307465

theorem arithmetic_sequence_product (b : ℕ → ℕ) : 
  (∀ n, b n < b (n + 1)) →  -- increasing sequence
  (∃ d : ℕ, ∀ n, b (n + 1) = b n + d) →  -- arithmetic sequence
  b 3 * b 4 = 72 → 
  b 2 * b 5 = 70 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l3074_307465


namespace NUMINAMATH_CALUDE_balcony_price_is_eight_l3074_307468

/-- Represents the theater ticket sales scenario --/
structure TheaterSales where
  totalTickets : ℕ
  totalCost : ℕ
  orchestraPrice : ℕ
  balconyExcess : ℕ

/-- Calculates the price of a balcony seat given the theater sales data --/
def balconyPrice (sales : TheaterSales) : ℕ :=
  let orchestraTickets := (sales.totalTickets - sales.balconyExcess) / 2
  let balconyTickets := sales.totalTickets - orchestraTickets
  (sales.totalCost - orchestraTickets * sales.orchestraPrice) / balconyTickets

/-- Theorem stating that the balcony price is $8 given the specific sales data --/
theorem balcony_price_is_eight :
  balconyPrice ⟨370, 3320, 12, 190⟩ = 8 := by
  sorry

#eval balconyPrice ⟨370, 3320, 12, 190⟩

end NUMINAMATH_CALUDE_balcony_price_is_eight_l3074_307468


namespace NUMINAMATH_CALUDE_original_number_proof_l3074_307432

theorem original_number_proof (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 30 →
  (a + b + c + 50) / 4 = 40 →
  d = 10 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l3074_307432


namespace NUMINAMATH_CALUDE_unique_n_for_prime_power_difference_l3074_307449

def is_power_of_three (x : ℕ) : Prop :=
  ∃ a : ℕ, x = 3^a ∧ a > 0

theorem unique_n_for_prime_power_difference :
  ∃! n : ℕ, n > 0 ∧ 
    (∃ p : ℕ, Nat.Prime p ∧ is_power_of_three (p^n - (p-1)^n)) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_n_for_prime_power_difference_l3074_307449


namespace NUMINAMATH_CALUDE_power_function_increasing_interval_l3074_307408

/-- Given a power function f(x) = x^a where a is a real number,
    and f(2) = √2, prove that the increasing interval of f is [0, +∞) -/
theorem power_function_increasing_interval
  (f : ℝ → ℝ)
  (a : ℝ)
  (h1 : ∀ x > 0, f x = x ^ a)
  (h2 : f 2 = Real.sqrt 2) :
  ∀ x y, 0 ≤ x ∧ x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_power_function_increasing_interval_l3074_307408


namespace NUMINAMATH_CALUDE_points_form_hyperbola_l3074_307428

/-- The set of points (x,y) defined by x = 2cosh(t) and y = 4sinh(t) for real t forms a hyperbola -/
theorem points_form_hyperbola :
  ∀ (t x y : ℝ), x = 2 * Real.cosh t ∧ y = 4 * Real.sinh t →
  x^2 / 4 - y^2 / 16 = 1 := by
sorry

end NUMINAMATH_CALUDE_points_form_hyperbola_l3074_307428


namespace NUMINAMATH_CALUDE_cosine_sum_theorem_l3074_307469

theorem cosine_sum_theorem (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos y + Real.cos z = 1)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 0) : 
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_theorem_l3074_307469


namespace NUMINAMATH_CALUDE_min_perimeter_rectangle_l3074_307454

/-- The minimum perimeter of a rectangle with area 100 is 40 -/
theorem min_perimeter_rectangle (x y : ℝ) (h : x * y = 100) :
  2 * (x + y) ≥ 40 := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_rectangle_l3074_307454


namespace NUMINAMATH_CALUDE_distance_to_place_l3074_307403

/-- Calculates the distance to a place given rowing speed, current velocity, and round trip time -/
theorem distance_to_place (rowing_speed current_velocity : ℝ) (round_trip_time : ℝ) : 
  rowing_speed = 5 → 
  current_velocity = 1 → 
  round_trip_time = 1 → 
  ∃ (distance : ℝ), distance = 2.4 ∧ 
    round_trip_time = distance / (rowing_speed + current_velocity) + 
                      distance / (rowing_speed - current_velocity) :=
by
  sorry

#check distance_to_place

end NUMINAMATH_CALUDE_distance_to_place_l3074_307403


namespace NUMINAMATH_CALUDE_min_value_of_f_l3074_307433

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + Real.exp (-x)

theorem min_value_of_f (a : ℝ) :
  (∃ k : ℝ, k * f a 0 + 1 = 0 ∧ k * (a - 1) = -1) →
  (∃ x : ℝ, ∀ y : ℝ, f a y ≥ f a x) ∧
  (∃ x : ℝ, f a x = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3074_307433


namespace NUMINAMATH_CALUDE_dana_wins_l3074_307422

/-- Represents a player in the game -/
inductive Player
  | Carl
  | Dana
  | Leah

/-- Represents the state of the game -/
structure GameState where
  chosenNumbers : List ℝ
  currentPlayer : Player

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (move : ℝ) : Prop :=
  0 ≤ move ∧ move ≤ 10 ∧
  ∀ n ∈ state.chosenNumbers, |move - n| ≥ 2

/-- Defines the next player in the turn order -/
def nextPlayer : Player → Player
  | Player.Carl => Player.Dana
  | Player.Dana => Player.Leah
  | Player.Leah => Player.Carl

/-- Represents a winning strategy for a player -/
def hasWinningStrategy (player : Player) : Prop :=
  ∀ initialState : GameState,
    initialState.currentPlayer = player →
    ∃ (strategy : GameState → ℝ),
      ∀ gameSequence : List ℝ,
        (∀ move ∈ gameSequence, isValidMove initialState move) →
        (∃ finalState : GameState,
          finalState.chosenNumbers = initialState.chosenNumbers ++ gameSequence ∧
          finalState.currentPlayer = player ∧
          ¬∃ move, isValidMove finalState move)

/-- The main theorem stating that Dana has a winning strategy -/
theorem dana_wins : hasWinningStrategy Player.Dana := by
  sorry

end NUMINAMATH_CALUDE_dana_wins_l3074_307422


namespace NUMINAMATH_CALUDE_removed_term_is_16th_l3074_307478

/-- Sum of the first n terms of the sequence -/
def S (n : ℕ) : ℕ := 2 * n^2 - n

/-- The k-th term of the sequence -/
def a (k : ℕ) : ℕ := 4 * k - 3

theorem removed_term_is_16th :
  ∀ k : ℕ,
  (S 21 - a k = 40 * 20) →
  k = 16 := by
sorry

end NUMINAMATH_CALUDE_removed_term_is_16th_l3074_307478


namespace NUMINAMATH_CALUDE_tim_sugar_cookies_l3074_307416

/-- Represents the number of sugar cookies Tim baked given the total number of cookies and the ratio of cookie types. -/
def sugar_cookies (total : ℕ) (choc_ratio sugar_ratio pb_ratio : ℕ) : ℕ :=
  (sugar_ratio * total) / (choc_ratio + sugar_ratio + pb_ratio)

/-- Theorem stating that Tim baked 15 sugar cookies given the problem conditions. -/
theorem tim_sugar_cookies :
  sugar_cookies 30 2 5 3 = 15 := by
sorry

end NUMINAMATH_CALUDE_tim_sugar_cookies_l3074_307416


namespace NUMINAMATH_CALUDE_subset_implies_m_values_l3074_307461

def A (m : ℝ) : Set ℝ := {-1, 3, 2*m-1}
def B (m : ℝ) : Set ℝ := {3, m}

theorem subset_implies_m_values (m : ℝ) :
  B m ⊆ A m → m = 1 ∨ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_subset_implies_m_values_l3074_307461


namespace NUMINAMATH_CALUDE_steve_growth_l3074_307440

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

theorem steve_growth :
  let original_height := feet_inches_to_inches 5 6
  let new_height := 72
  new_height - original_height = 6 := by
  sorry

end NUMINAMATH_CALUDE_steve_growth_l3074_307440


namespace NUMINAMATH_CALUDE_digit_product_sum_l3074_307443

/-- A function that converts a pair of digits to a two-digit integer -/
def twoDigitInt (tens ones : Nat) : Nat :=
  10 * tens + ones

/-- A predicate that checks if a number is a positive digit (1-9) -/
def isPositiveDigit (n : Nat) : Prop :=
  0 < n ∧ n ≤ 9

theorem digit_product_sum (p q r : Nat) : 
  isPositiveDigit p ∧ isPositiveDigit q ∧ isPositiveDigit r →
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (twoDigitInt p q) * (twoDigitInt p r) = 221 →
  p + q + r = 11 →
  q = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_product_sum_l3074_307443


namespace NUMINAMATH_CALUDE_multiplication_result_l3074_307455

theorem multiplication_result : 72515 * 10005 = 724787425 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_result_l3074_307455


namespace NUMINAMATH_CALUDE_sqrt_neg_three_l3074_307495

theorem sqrt_neg_three (z : ℂ) : z * z = -3 ↔ z = Complex.I * Real.sqrt 3 ∨ z = -Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_three_l3074_307495


namespace NUMINAMATH_CALUDE_negative_square_cubed_l3074_307401

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l3074_307401


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3074_307413

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ (y : ℝ), m^2 + m - 2 + (m^2 - 1) * Complex.I = y * Complex.I) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3074_307413


namespace NUMINAMATH_CALUDE_opposite_numbers_fifth_power_sum_l3074_307457

theorem opposite_numbers_fifth_power_sum (a b : ℝ) : 
  a + b = 0 → a^5 + b^5 = 0 := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_fifth_power_sum_l3074_307457


namespace NUMINAMATH_CALUDE_sum_of_ratios_bound_l3074_307458

theorem sum_of_ratios_bound (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 1) : 
  a / (b + c^2) + b / (c + a^2) + c / (a + b^2) ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_bound_l3074_307458


namespace NUMINAMATH_CALUDE_worker_payment_schedule_l3074_307405

/-- Represents the worker payment schedule problem -/
theorem worker_payment_schedule 
  (daily_wage : ℕ) 
  (daily_return : ℕ) 
  (days_not_worked : ℕ) : 
  daily_wage = 100 → 
  daily_return = 25 → 
  days_not_worked = 24 → 
  ∃ (days_worked : ℕ), 
    daily_wage * days_worked = daily_return * days_not_worked ∧ 
    days_worked + days_not_worked = 30 := by
  sorry

end NUMINAMATH_CALUDE_worker_payment_schedule_l3074_307405


namespace NUMINAMATH_CALUDE_janet_fertilizer_time_l3074_307407

-- Define the constants from the problem
def gallons_per_horse_per_day : ℕ := 5
def number_of_horses : ℕ := 80
def total_acres : ℕ := 20
def gallons_per_acre : ℕ := 400
def acres_spread_per_day : ℕ := 4

-- Define the theorem
theorem janet_fertilizer_time : 
  (total_acres * gallons_per_acre) / (number_of_horses * gallons_per_horse_per_day) +
  total_acres / acres_spread_per_day = 25 := by
  sorry

end NUMINAMATH_CALUDE_janet_fertilizer_time_l3074_307407


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l3074_307412

/-- The time it takes for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time (jogger_speed train_speed : ℝ) 
  (initial_distance train_length : ℝ) (h1 : jogger_speed > 0) 
  (h2 : train_speed > jogger_speed) (h3 : initial_distance > 0) 
  (h4 : train_length > 0) : 
  (initial_distance + train_length) / (train_speed - jogger_speed) = 40 := by
  sorry

#check train_passing_jogger_time

end NUMINAMATH_CALUDE_train_passing_jogger_time_l3074_307412


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3074_307414

def U : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def A : Set ℤ := {-2, 0, 2, 4}
def B : Set ℤ := {-2, 0, 4, 6, 8}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {6, 8} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3074_307414


namespace NUMINAMATH_CALUDE_wonderful_class_size_l3074_307427

/-- Represents the number of students in Mrs. Wonderful's class -/
def class_size : ℕ := 18

/-- Represents the number of girls in the class -/
def girls : ℕ := class_size / 2 - 2

/-- Represents the number of boys in the class -/
def boys : ℕ := girls + 4

/-- The total number of jelly beans Mrs. Wonderful brought -/
def total_jelly_beans : ℕ := 420

/-- The number of jelly beans left after distribution -/
def remaining_jelly_beans : ℕ := 6

/-- Theorem stating that the given conditions result in 18 students -/
theorem wonderful_class_size : 
  (3 * girls * girls + 2 * boys * boys = total_jelly_beans - remaining_jelly_beans) ∧
  (boys = girls + 4) ∧
  (class_size = girls + boys) := by sorry

end NUMINAMATH_CALUDE_wonderful_class_size_l3074_307427


namespace NUMINAMATH_CALUDE_math_books_probability_math_books_probability_is_one_sixth_l3074_307486

/-- The probability of selecting 2 math books from a shelf with 2 math books and 2 physics books -/
theorem math_books_probability : ℚ :=
  let total_books : ℕ := 4
  let math_books : ℕ := 2
  let books_to_pick : ℕ := 2
  let total_combinations := Nat.choose total_books books_to_pick
  let favorable_combinations := Nat.choose math_books books_to_pick
  (favorable_combinations : ℚ) / total_combinations

/-- The probability of selecting 2 math books from a shelf with 2 math books and 2 physics books is 1/6 -/
theorem math_books_probability_is_one_sixth : math_books_probability = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_math_books_probability_math_books_probability_is_one_sixth_l3074_307486


namespace NUMINAMATH_CALUDE_salary_increase_proof_l3074_307448

def original_salary : ℝ := 60
def percentage_increase : ℝ := 13.333333333333334
def new_salary : ℝ := 68

theorem salary_increase_proof :
  original_salary * (1 + percentage_increase / 100) = new_salary := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_proof_l3074_307448


namespace NUMINAMATH_CALUDE_trivia_team_points_per_member_l3074_307445

theorem trivia_team_points_per_member 
  (total_members : ℝ) 
  (absent_members : ℝ) 
  (total_points : ℝ) 
  (h1 : total_members = 5.0) 
  (h2 : absent_members = 2.0) 
  (h3 : total_points = 6.0) : 
  total_points / (total_members - absent_members) = 2.0 := by
sorry

end NUMINAMATH_CALUDE_trivia_team_points_per_member_l3074_307445


namespace NUMINAMATH_CALUDE_square_and_circle_l3074_307474

theorem square_and_circle (square_area : ℝ) (side_length : ℝ) (circle_radius : ℝ) : 
  square_area = 1 →
  side_length ^ 2 = square_area →
  circle_radius * 2 = side_length →
  side_length = 1 ∧ circle_radius = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_square_and_circle_l3074_307474


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3074_307481

theorem quadratic_equation_solution (b : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x - b = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 - 2*y - b = 0 ∧ y = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3074_307481


namespace NUMINAMATH_CALUDE_equality_proof_l3074_307434

theorem equality_proof (n : ℕ) (h : n ≥ 2) :
  n * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_equality_proof_l3074_307434


namespace NUMINAMATH_CALUDE_list_median_is_106_l3074_307447

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def list_length : ℕ := sequence_sum 150

def median_position : ℕ := (list_length + 1) / 2

theorem list_median_is_106 : ∃ (n : ℕ), 
  n = 106 ∧ 
  sequence_sum (n - 1) < median_position ∧ 
  median_position ≤ sequence_sum n :=
sorry

end NUMINAMATH_CALUDE_list_median_is_106_l3074_307447


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3074_307477

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ j : ℕ, j ≤ 10 ∧ j > 0 ∧ m % j ≠ 0) ∧
  n = 2520 :=
by sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3074_307477


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3074_307482

/-- Given a quadratic equation ax^2 + bx + c = 0 with no real roots,
    if there exist two possible misinterpretations of the equation
    such that one yields roots 2 and 4, and the other yields roots -1 and 4,
    then (2b + 3c) / a = 12 -/
theorem quadratic_equation_roots (a b c : ℝ) : 
  (∀ x : ℝ, a * x^2 + b * x + c ≠ 0) →  -- No real roots
  (∃ a' : ℝ, a' * 4^2 + b * 4 + c = 0 ∧ a' * 2^2 + b * 2 + c = 0) →  -- Misinterpretation 1
  (∃ b' : ℝ, a * 4^2 + b' * 4 + c = 0 ∧ a * (-1)^2 + b' * (-1) + c = 0) →  -- Misinterpretation 2
  (2 * b + 3 * c) / a = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3074_307482


namespace NUMINAMATH_CALUDE_power_sum_fifth_l3074_307406

theorem power_sum_fifth (a b x y : ℝ) 
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_fifth_l3074_307406


namespace NUMINAMATH_CALUDE_steve_earnings_l3074_307484

def total_copies : ℕ := 1000000
def advance_copies : ℕ := 100000
def price_per_copy : ℚ := 2
def agent_percentage : ℚ := 1/10

theorem steve_earnings : 
  (total_copies - advance_copies) * price_per_copy * (1 - agent_percentage) = 1620000 := by
  sorry

end NUMINAMATH_CALUDE_steve_earnings_l3074_307484


namespace NUMINAMATH_CALUDE_frank_remaining_money_l3074_307419

def cheapest_lamp : ℕ := 20
def expensive_multiplier : ℕ := 3
def frank_money : ℕ := 90

theorem frank_remaining_money :
  frank_money - (cheapest_lamp * expensive_multiplier) = 30 := by
  sorry

end NUMINAMATH_CALUDE_frank_remaining_money_l3074_307419


namespace NUMINAMATH_CALUDE_unique_solution_inequality_holds_max_value_l3074_307450

-- Define the functions f, g, and h
def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|
def h (a x : ℝ) : ℝ := |f x| + g a x

-- Theorem for part (1)
theorem unique_solution (a : ℝ) :
  (∃! x, |f x| = g a x) ↔ a < 0 :=
sorry

-- Theorem for part (2)
theorem inequality_holds (a : ℝ) :
  (∀ x, f x ≥ g a x) ↔ a ≤ -2 :=
sorry

-- Theorem for part (3)
theorem max_value (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) 2, h a x ≤ 
    if a ≥ 0 then 3*a + 3
    else if a ≥ -3 then a + 3
    else 0) ∧
  (∃ x ∈ Set.Icc (-2) 2, h a x = 
    if a ≥ 0 then 3*a + 3
    else if a ≥ -3 then a + 3
    else 0) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_holds_max_value_l3074_307450


namespace NUMINAMATH_CALUDE_prism_volume_30_l3074_307418

/-- A right rectangular prism with integer edge lengths -/
structure RightRectangularPrism where
  a : ℕ
  b : ℕ
  h : ℕ

/-- The volume of a right rectangular prism -/
def volume (p : RightRectangularPrism) : ℕ := p.a * p.b * p.h

/-- The areas of the faces of a right rectangular prism -/
def face_areas (p : RightRectangularPrism) : Finset ℕ :=
  {p.a * p.b, p.a * p.h, p.b * p.h}

theorem prism_volume_30 (p : RightRectangularPrism) :
  30 ∈ face_areas p → 13 ∈ face_areas p → volume p = 30 := by
  sorry

#check prism_volume_30

end NUMINAMATH_CALUDE_prism_volume_30_l3074_307418


namespace NUMINAMATH_CALUDE_remainder_proof_l3074_307499

theorem remainder_proof (n : ℕ) (h1 : n = 129) (h2 : 1428 % n = 9) : 2206 % n = 13 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l3074_307499


namespace NUMINAMATH_CALUDE_share_difference_l3074_307437

theorem share_difference (total : ℚ) (p m s : ℚ) : 
  total = 730 →
  p + m + s = total →
  4 * p = 3 * m →
  3 * m = 3.5 * s →
  m - s = 36.5 := by
  sorry

end NUMINAMATH_CALUDE_share_difference_l3074_307437


namespace NUMINAMATH_CALUDE_members_playing_two_sports_l3074_307415

theorem members_playing_two_sports
  (total_members : ℕ)
  (badminton_players : ℕ)
  (tennis_players : ℕ)
  (soccer_players : ℕ)
  (no_sport_players : ℕ)
  (badminton_tennis : ℕ)
  (badminton_soccer : ℕ)
  (tennis_soccer : ℕ)
  (h1 : total_members = 60)
  (h2 : badminton_players = 25)
  (h3 : tennis_players = 32)
  (h4 : soccer_players = 14)
  (h5 : no_sport_players = 5)
  (h6 : badminton_tennis = 10)
  (h7 : badminton_soccer = 8)
  (h8 : tennis_soccer = 6)
  (h9 : badminton_tennis + badminton_soccer + tennis_soccer ≤ badminton_players + tennis_players + soccer_players) :
  badminton_tennis + badminton_soccer + tennis_soccer = 24 :=
by sorry

end NUMINAMATH_CALUDE_members_playing_two_sports_l3074_307415


namespace NUMINAMATH_CALUDE_gcd_g_y_equals_one_l3074_307438

theorem gcd_g_y_equals_one (y : ℤ) (h : ∃ k : ℤ, y = 34567 * k) :
  let g : ℤ → ℤ := λ y => (3*y+4)*(8*y+3)*(14*y+5)*(y+14)
  Int.gcd (g y) y = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_y_equals_one_l3074_307438


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3074_307472

theorem condition_necessary_not_sufficient (a b : ℝ) :
  (a + b ≠ 3 → (a ≠ 1 ∨ b ≠ 2)) ∧
  ¬(a ≠ 1 ∨ b ≠ 2 → a + b ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3074_307472


namespace NUMINAMATH_CALUDE_ab_value_l3074_307417

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l3074_307417


namespace NUMINAMATH_CALUDE_cupcake_milk_calculation_l3074_307436

/-- The number of cupcakes in a full recipe -/
def full_recipe_cupcakes : ℕ := 24

/-- The number of quarts of milk needed for a full recipe -/
def full_recipe_quarts : ℕ := 3

/-- The number of pints in a quart -/
def pints_per_quart : ℕ := 2

/-- The number of cupcakes we want to make -/
def target_cupcakes : ℕ := 6

/-- The amount of milk in pints needed for the target number of cupcakes -/
def milk_needed : ℚ := 1.5

theorem cupcake_milk_calculation :
  (target_cupcakes : ℚ) * (full_recipe_quarts * pints_per_quart : ℚ) / full_recipe_cupcakes = milk_needed :=
sorry

end NUMINAMATH_CALUDE_cupcake_milk_calculation_l3074_307436


namespace NUMINAMATH_CALUDE_vector_sum_not_necessarily_zero_l3074_307470

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given arbitrary points O, A, B, C in a real vector space V, 
    the vector expression OA + OC + BO + CO is not necessarily zero. -/
theorem vector_sum_not_necessarily_zero (O A B C : V) :
  ¬ (∀ (O A B C : V), O + A + C + B + O + C + O = 0) :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_not_necessarily_zero_l3074_307470


namespace NUMINAMATH_CALUDE_percentage_male_students_l3074_307426

theorem percentage_male_students 
  (T : ℝ) -- Total number of students
  (M : ℝ) -- Number of male students
  (F : ℝ) -- Number of female students
  (h1 : M + F = T) -- Total students equation
  (h2 : (2/7) * M + (1/3) * F = 0.3 * T) -- Married students equation
  : M / T = 0.7 := by sorry

end NUMINAMATH_CALUDE_percentage_male_students_l3074_307426


namespace NUMINAMATH_CALUDE_original_male_count_l3074_307467

/-- Represents the number of students of each gender -/
structure StudentCount where
  male : ℕ
  female : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (s : StudentCount) : Prop :=
  (s.male : ℚ) / ((s.female : ℚ) - 15) = 2 ∧
  ((s.male : ℚ) - 45) / ((s.female : ℚ) - 15) = 1/5

/-- The theorem stating that the original number of male students is 50 -/
theorem original_male_count (s : StudentCount) :
  satisfiesConditions s → s.male = 50 := by
  sorry


end NUMINAMATH_CALUDE_original_male_count_l3074_307467


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3074_307462

theorem quadratic_roots_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ 
   x₁^2 - (m+2)*x₁ + 1 = 0 ∧ 
   x₂^2 - (m+2)*x₂ + 1 = 0 ∧ 
   x₁ ≠ x₂) →
  m ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3074_307462


namespace NUMINAMATH_CALUDE_factorization_valid_l3074_307496

theorem factorization_valid (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_valid_l3074_307496


namespace NUMINAMATH_CALUDE_proportional_enlargement_l3074_307494

/-- Proportional enlargement of a rectangle -/
theorem proportional_enlargement (original_width original_height new_width : ℝ) 
  (h1 : original_width > 0)
  (h2 : original_height > 0)
  (h3 : new_width > 0) :
  let scale_factor := new_width / original_width
  let new_height := original_height * scale_factor
  (original_width = 3 ∧ original_height = 2 ∧ new_width = 12) → new_height = 8 := by
sorry

end NUMINAMATH_CALUDE_proportional_enlargement_l3074_307494


namespace NUMINAMATH_CALUDE_train_length_calculation_l3074_307490

/-- The length of a train given its speed, a man's walking speed, and the time it takes to cross the man. -/
theorem train_length_calculation (train_speed man_speed : ℝ) (crossing_time : ℝ) :
  train_speed = 63 →
  man_speed = 3 →
  crossing_time = 41.9966402687785 →
  ∃ (train_length : ℝ), abs (train_length - 700) < 0.1 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3074_307490


namespace NUMINAMATH_CALUDE_min_additional_coins_for_alex_l3074_307492

/-- The minimum number of additional coins needed for distribution -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed -/
theorem min_additional_coins_for_alex : 
  min_additional_coins 15 63 = 57 := by
  sorry

end NUMINAMATH_CALUDE_min_additional_coins_for_alex_l3074_307492


namespace NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l3074_307442

theorem smallest_n_for_exact_tax : ∃ (x : ℕ+), (104 * x : ℚ) / 10000 = 13 ∧
  ∀ (n : ℕ+), n < 13 → ¬∃ (y : ℕ+), (104 * y : ℚ) / 10000 = n := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l3074_307442


namespace NUMINAMATH_CALUDE_no_zeros_of_g_l3074_307464

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (hf_cont : Continuous f)
variable (hf_diff : Differentiable ℝ f)
variable (hf_pos : ∀ x, x * (deriv f x) + f x > 0)

-- Define the function g
def g (x : ℝ) := x * f x + 1

-- State the theorem
theorem no_zeros_of_g :
  ∀ x > 0, g f x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_no_zeros_of_g_l3074_307464


namespace NUMINAMATH_CALUDE_production_today_is_90_l3074_307439

/-- Calculates the production for today given the previous average, new average, and number of previous days. -/
def todayProduction (prevAvg newAvg : ℚ) (prevDays : ℕ) : ℚ :=
  (newAvg * (prevDays + 1) : ℚ) - (prevAvg * prevDays : ℚ)

/-- Proves that the production today is 90 units, given the specified conditions. -/
theorem production_today_is_90 :
  todayProduction 60 62 14 = 90 := by
  sorry

#eval todayProduction 60 62 14

end NUMINAMATH_CALUDE_production_today_is_90_l3074_307439


namespace NUMINAMATH_CALUDE_constant_function_invariant_l3074_307488

/-- Given a function f that is constant 3 for all real inputs, 
    prove that f(x + 5) = 3 for any real x -/
theorem constant_function_invariant (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = 3) :
  ∀ x : ℝ, f (x + 5) = 3 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_invariant_l3074_307488


namespace NUMINAMATH_CALUDE_solve_comic_problem_l3074_307476

def comic_problem (pages_per_comic : ℕ) (found_pages : ℕ) (total_comics : ℕ) : Prop :=
  let repaired_comics := found_pages / pages_per_comic
  let untorn_comics := total_comics - repaired_comics
  untorn_comics = 5

theorem solve_comic_problem :
  comic_problem 25 150 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_comic_problem_l3074_307476


namespace NUMINAMATH_CALUDE_distance_midpoint_endpoint_l3074_307420

theorem distance_midpoint_endpoint (t : ℝ) : 
  let A : ℝ × ℝ := (t - 4, -1)
  let B : ℝ × ℝ := (-2, t + 3)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ((midpoint.1 - A.1)^2 + (midpoint.2 - A.2)^2 = t^2 / 2) →
  t = -5 := by
sorry

end NUMINAMATH_CALUDE_distance_midpoint_endpoint_l3074_307420


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l3074_307423

theorem geometric_progression_first_term
  (S : ℝ)
  (sum_first_two : ℝ)
  (h1 : S = 15)
  (h2 : sum_first_two = 10) :
  ∃ (a : ℝ), (a = (15 * (Real.sqrt 3 - 1)) / Real.sqrt 3 ∨
              a = (15 * (Real.sqrt 3 + 1)) / Real.sqrt 3) ∧
             (∃ (r : ℝ), S = a / (1 - r) ∧ sum_first_two = a + a * r) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l3074_307423


namespace NUMINAMATH_CALUDE_zero_in_interval_l3074_307404

def f (x : ℝ) : ℝ := x^3 + x - 4

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 := by
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l3074_307404


namespace NUMINAMATH_CALUDE_rook_placements_l3074_307466

def chessboard_size : ℕ := 8
def num_rooks : ℕ := 3

theorem rook_placements : 
  (chessboard_size.choose num_rooks) * (chessboard_size * (chessboard_size - 1) * (chessboard_size - 2)) = 18816 :=
by sorry

end NUMINAMATH_CALUDE_rook_placements_l3074_307466


namespace NUMINAMATH_CALUDE_range_of_m_l3074_307489

/-- Given the conditions of P and Q, prove that the range of m is [9, +∞) -/
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (|(4 - x) / 3| ≤ 2 → (x + m - 1) * (x - m - 1) ≤ 0)) ∧
  (∃ x : ℝ, |(4 - x) / 3| > 2 ∧ (x + m - 1) * (x - m - 1) > 0) ∧
  (m > 0) →
  m ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3074_307489


namespace NUMINAMATH_CALUDE_inequality_proof_l3074_307491

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (1/2) * ((a - b)^2 + (b - c)^2 + (c - a)^2) ≤ a^2 + b^2 + c^2 - 3 * (a^2 * b^2 * c^2)^(1/3) ∧
  a^2 + b^2 + c^2 - 3 * (a^2 * b^2 * c^2)^(1/3) ≤ (a - b)^2 + (b - c)^2 + (c - a)^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3074_307491


namespace NUMINAMATH_CALUDE_unique_twin_prime_trio_l3074_307409

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem unique_twin_prime_trio : 
  ∀ p : ℕ, is_prime p → p > 7 → ¬(is_prime p ∧ is_prime (p + 2) ∧ is_prime (p + 4)) :=
sorry

end NUMINAMATH_CALUDE_unique_twin_prime_trio_l3074_307409


namespace NUMINAMATH_CALUDE_eighty_one_power_ten_equals_three_power_q_l3074_307480

theorem eighty_one_power_ten_equals_three_power_q (q : ℕ) : 81^10 = 3^q → q = 40 := by
  sorry

end NUMINAMATH_CALUDE_eighty_one_power_ten_equals_three_power_q_l3074_307480


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3074_307425

/-- A quadratic function f(x) = ax^2 + bx + 6 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 6

theorem quadratic_function_properties (a b : ℝ) :
  f a b 1 = 8 ∧ f a b (-1) = f a b 3 →
  (a + b = 2 ∧ f a b 2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3074_307425


namespace NUMINAMATH_CALUDE_solve_equation_l3074_307479

theorem solve_equation :
  let y := 45 / (8 - 3/7)
  y = 315/53 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l3074_307479


namespace NUMINAMATH_CALUDE_select_medical_team_eq_630_l3074_307475

/-- The number of ways to select a medical team for earthquake relief. -/
def select_medical_team : ℕ :=
  let orthopedic : ℕ := 3
  let neurosurgeon : ℕ := 4
  let internist : ℕ := 5
  let team_size : ℕ := 5
  
  -- Combinations for each possible selection scenario
  let scenario1 := Nat.choose orthopedic 3 * Nat.choose neurosurgeon 1 * Nat.choose internist 1
  let scenario2 := Nat.choose orthopedic 1 * Nat.choose neurosurgeon 3 * Nat.choose internist 1
  let scenario3 := Nat.choose orthopedic 1 * Nat.choose neurosurgeon 1 * Nat.choose internist 3
  let scenario4 := Nat.choose orthopedic 2 * Nat.choose neurosurgeon 2 * Nat.choose internist 1
  let scenario5 := Nat.choose orthopedic 1 * Nat.choose neurosurgeon 2 * Nat.choose internist 2
  let scenario6 := Nat.choose orthopedic 2 * Nat.choose neurosurgeon 1 * Nat.choose internist 2

  -- Sum of all scenarios
  scenario1 + scenario2 + scenario3 + scenario4 + scenario5 + scenario6

/-- Theorem stating that the number of ways to select the medical team is 630. -/
theorem select_medical_team_eq_630 : select_medical_team = 630 := by
  sorry

end NUMINAMATH_CALUDE_select_medical_team_eq_630_l3074_307475


namespace NUMINAMATH_CALUDE_range_of_a_l3074_307410

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x | 2*a - 1 < x ∧ x < 4*a}
def N : Set ℝ := {x | 1 < x ∧ x < 2}

-- State the theorem
theorem range_of_a (h : N ⊆ M a) : 1/2 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3074_307410


namespace NUMINAMATH_CALUDE_smallest_terminating_decimal_l3074_307421

/-- A positive integer n such that n/(n+51) is a terminating decimal -/
def is_terminating_decimal (n : ℕ+) : Prop :=
  ∃ (a b : ℕ), n.val / (n.val + 51) = (a : ℚ) / (10^b : ℚ)

/-- 74 is the smallest positive integer n such that n/(n+51) is a terminating decimal -/
theorem smallest_terminating_decimal :
  (∀ m : ℕ+, m.val < 74 → ¬is_terminating_decimal m) ∧ is_terminating_decimal 74 :=
sorry

end NUMINAMATH_CALUDE_smallest_terminating_decimal_l3074_307421


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3074_307402

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3074_307402


namespace NUMINAMATH_CALUDE_total_eyes_is_92_l3074_307411

/-- Represents a monster family in the portrait --/
structure MonsterFamily where
  totalEyes : ℕ

/-- The main monster family --/
def mainFamily : MonsterFamily :=
  { totalEyes := 1 + 3 + 3 * 4 + 5 + 6 + 2 + 1 + 7 + 8 }

/-- The first neighboring monster family --/
def neighborFamily1 : MonsterFamily :=
  { totalEyes := 9 + 3 + 7 + 3 }

/-- The second neighboring monster family --/
def neighborFamily2 : MonsterFamily :=
  { totalEyes := 4 + 2 * 8 + 5 }

/-- The total number of eyes in the monster family portrait --/
def totalEyesInPortrait : ℕ :=
  mainFamily.totalEyes + neighborFamily1.totalEyes + neighborFamily2.totalEyes

/-- Theorem stating that the total number of eyes in the portrait is 92 --/
theorem total_eyes_is_92 : totalEyesInPortrait = 92 := by
  sorry

end NUMINAMATH_CALUDE_total_eyes_is_92_l3074_307411


namespace NUMINAMATH_CALUDE_rosa_initial_flowers_l3074_307493

theorem rosa_initial_flowers (flowers_from_andre : ℕ) (total_flowers : ℕ) 
  (h1 : flowers_from_andre = 23)
  (h2 : total_flowers = 90) :
  total_flowers - flowers_from_andre = 67 := by
  sorry

end NUMINAMATH_CALUDE_rosa_initial_flowers_l3074_307493
