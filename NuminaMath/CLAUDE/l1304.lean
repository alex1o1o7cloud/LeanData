import Mathlib

namespace NUMINAMATH_CALUDE_ratio_equality_l1304_130474

theorem ratio_equality (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4 ∧ c / 4 ≠ 0) : 
  (a + b) / c = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1304_130474


namespace NUMINAMATH_CALUDE_bobs_smallest_number_l1304_130438

def is_valid_bob_number (alice_num bob_num : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → p ∣ alice_num → p ∣ bob_num

def has_additional_prime_factor (alice_num bob_num : ℕ) : Prop :=
  ∃ q : ℕ, q.Prime ∧ q ∣ bob_num ∧ ¬(q ∣ alice_num)

theorem bobs_smallest_number (alice_num : ℕ) (bob_num : ℕ) :
  alice_num = 36 →
  is_valid_bob_number alice_num bob_num →
  has_additional_prime_factor alice_num bob_num →
  (∀ n : ℕ, n < bob_num →
    ¬(is_valid_bob_number alice_num n ∧ has_additional_prime_factor alice_num n)) →
  bob_num = 30 :=
sorry

end NUMINAMATH_CALUDE_bobs_smallest_number_l1304_130438


namespace NUMINAMATH_CALUDE_line_circle_min_value_l1304_130424

/-- Given a line ax + by + 1 = 0 that divides a circle into two equal areas, 
    prove that the minimum value of 1/(2a) + 2/b is 8 -/
theorem line_circle_min_value (a b : ℝ) : 
  a > 0 → 
  b > 0 → 
  (∀ x y : ℝ, a * x + b * y + 1 = 0 → (x + 4)^2 + (y + 1)^2 = 16 → 
    (∃ k : ℝ, k > 0 ∧ k * ((x + 4)^2 + (y + 1)^2) = 16 ∧ 
    k * (a * x + b * y + 1) = 0)) → 
  (∀ x y : ℝ, (1 / (2 * a) + 2 / b) ≥ 8) ∧ 
  (∃ x y : ℝ, 1 / (2 * a) + 2 / b = 8) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_min_value_l1304_130424


namespace NUMINAMATH_CALUDE_tiles_needed_l1304_130457

/-- Given a rectangular room and tiling specifications, calculate the number of tiles needed --/
theorem tiles_needed (room_length room_width tile_size fraction_to_tile : ℝ) 
  (h1 : room_length = 12)
  (h2 : room_width = 20)
  (h3 : tile_size = 1)
  (h4 : fraction_to_tile = 1/6) :
  (room_length * room_width * fraction_to_tile) / tile_size = 40 := by
  sorry

end NUMINAMATH_CALUDE_tiles_needed_l1304_130457


namespace NUMINAMATH_CALUDE_proposition_implication_l1304_130433

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k ≥ 1 → (P k → P (k + 1)))
  (h2 : ¬ P 10) : 
  ¬ P 9 := by sorry

end NUMINAMATH_CALUDE_proposition_implication_l1304_130433


namespace NUMINAMATH_CALUDE_a_n_bounds_l1304_130420

variable (n : ℕ+)

noncomputable def a : ℕ → ℚ
  | 0 => 1/2
  | k + 1 => a k + (1/n) * (a k)^2

theorem a_n_bounds : 1 - 1/n < a n n ∧ a n n < 1 := by sorry

end NUMINAMATH_CALUDE_a_n_bounds_l1304_130420


namespace NUMINAMATH_CALUDE_delta_cheaper_from_min_posters_gamma_cheaper_or_equal_before_min_posters_l1304_130465

/-- Delta Printing Company's pricing function -/
def delta_price (n : ℕ) : ℝ := 40 + 7 * n

/-- Gamma Printing Company's pricing function -/
def gamma_price (n : ℕ) : ℝ := 11 * n

/-- The minimum number of posters for which Delta is cheaper than Gamma -/
def min_posters_for_delta : ℕ := 11

theorem delta_cheaper_from_min_posters :
  ∀ n : ℕ, n ≥ min_posters_for_delta → delta_price n < gamma_price n :=
sorry

theorem gamma_cheaper_or_equal_before_min_posters :
  ∀ n : ℕ, n < min_posters_for_delta → delta_price n ≥ gamma_price n :=
sorry

end NUMINAMATH_CALUDE_delta_cheaper_from_min_posters_gamma_cheaper_or_equal_before_min_posters_l1304_130465


namespace NUMINAMATH_CALUDE_circle_radius_from_circumference_l1304_130475

/-- The radius of a circle with circumference 100π cm is 50 cm. -/
theorem circle_radius_from_circumference :
  ∀ (r : ℝ), 2 * π * r = 100 * π → r = 50 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_circumference_l1304_130475


namespace NUMINAMATH_CALUDE_oranges_bought_l1304_130403

/-- Represents the fruit shopping scenario over a week -/
structure FruitShopping where
  apples : ℕ
  oranges : ℕ
  total_fruits : apples + oranges = 5
  total_cost : ℕ
  cost_is_whole_dollars : total_cost % 100 = 0
  cost_calculation : total_cost = 30 * apples + 45 * oranges + 20

/-- Theorem stating that the number of oranges bought is 2 -/
theorem oranges_bought (shop : FruitShopping) : shop.oranges = 2 := by
  sorry

end NUMINAMATH_CALUDE_oranges_bought_l1304_130403


namespace NUMINAMATH_CALUDE_unique_n_mod_59_l1304_130464

theorem unique_n_mod_59 : ∃! n : ℤ, 0 ≤ n ∧ n < 59 ∧ 58 * n % 59 = 20 % 59 ∧ n = 39 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_mod_59_l1304_130464


namespace NUMINAMATH_CALUDE_max_visible_cubes_12_10_9_l1304_130470

/-- Represents a rectangular block formed by unit cubes -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point for a given block -/
def max_visible_cubes (b : Block) : ℕ :=
  b.length * b.width + b.width * b.height + b.length * b.height -
  (b.length + b.width + b.height) + 1

/-- The theorem stating that for a 12 × 10 × 9 block, the maximum number of visible unit cubes is 288 -/
theorem max_visible_cubes_12_10_9 :
  max_visible_cubes ⟨12, 10, 9⟩ = 288 := by
  sorry

#eval max_visible_cubes ⟨12, 10, 9⟩

end NUMINAMATH_CALUDE_max_visible_cubes_12_10_9_l1304_130470


namespace NUMINAMATH_CALUDE_problem_statement_l1304_130422

theorem problem_statement (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4*b^2 = 1/(a*b) + 3) :
  (ab ≤ 1) ∧ (b > a → 1/a^3 - 1/b^3 > 3*(1/a - 1/b)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1304_130422


namespace NUMINAMATH_CALUDE_wang_hao_height_l1304_130434

/-- Given Yao Ming's height and the difference between Yao Ming's and Wang Hao's heights,
    prove that Wang Hao's height is 1.58 meters. -/
theorem wang_hao_height (yao_ming_height : ℝ) (height_difference : ℝ) 
  (h1 : yao_ming_height = 2.29)
  (h2 : height_difference = 0.71) :
  yao_ming_height - height_difference = 1.58 := by
  sorry

end NUMINAMATH_CALUDE_wang_hao_height_l1304_130434


namespace NUMINAMATH_CALUDE_inverse_proposition_correct_l1304_130451

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop :=
  a = b → |a| = |b|

-- Define the inverse proposition
def inverse_proposition (a b : ℝ) : Prop :=
  |a| = |b| → a = b

-- Theorem stating that the inverse_proposition is indeed the inverse of the original_proposition
theorem inverse_proposition_correct :
  ∀ a b : ℝ, inverse_proposition a b ↔ ¬(original_proposition a b) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proposition_correct_l1304_130451


namespace NUMINAMATH_CALUDE_smallest_product_is_623_l1304_130481

def Digits : Finset Nat := {7, 8, 9, 0}

def valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def two_digit_number (tens ones : Nat) : Nat :=
  10 * tens + ones

theorem smallest_product_is_623 :
  ∀ a b c d : Nat,
    valid_arrangement a b c d →
    (two_digit_number a b) * (two_digit_number c d) ≥ 623 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_is_623_l1304_130481


namespace NUMINAMATH_CALUDE_total_clothing_items_l1304_130477

theorem total_clothing_items (short_sleeve : ℕ) (long_sleeve : ℕ) (pants : ℕ) (jackets : ℕ) 
  (h1 : short_sleeve = 7)
  (h2 : long_sleeve = 9)
  (h3 : pants = 4)
  (h4 : jackets = 2) :
  short_sleeve + long_sleeve + pants + jackets = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_clothing_items_l1304_130477


namespace NUMINAMATH_CALUDE_min_value_at_four_l1304_130437

/-- The quadratic function we're minimizing -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 15

/-- Theorem stating that f(x) achieves its minimum when x = 4 -/
theorem min_value_at_four :
  ∀ x : ℝ, f x ≥ f 4 := by sorry

end NUMINAMATH_CALUDE_min_value_at_four_l1304_130437


namespace NUMINAMATH_CALUDE_scientific_notation_of_104000_l1304_130408

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation := sorry

/-- The given number from the problem -/
def givenNumber : ℝ := 104000

theorem scientific_notation_of_104000 :
  toScientificNotation givenNumber = ScientificNotation.mk 1.04 5 (by norm_num) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_104000_l1304_130408


namespace NUMINAMATH_CALUDE_quadratic_rewrite_product_l1304_130432

/-- Given a quadratic equation 9x^2 - 30x - 42 that can be rewritten as (ax + b)^2 + c
    where a, b, and c are integers, prove that ab = -15 -/
theorem quadratic_rewrite_product (a b c : ℤ) : 
  (∀ x, 9*x^2 - 30*x - 42 = (a*x + b)^2 + c) → a*b = -15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_product_l1304_130432


namespace NUMINAMATH_CALUDE_odd_periodic_function_value_l1304_130492

-- Define an odd function with period 2
def is_odd_periodic_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2) = f x)

-- Theorem statement
theorem odd_periodic_function_value (f : ℝ → ℝ) 
  (h : is_odd_periodic_function f) : f 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_value_l1304_130492


namespace NUMINAMATH_CALUDE_aquarium_fish_count_l1304_130418

theorem aquarium_fish_count (total : ℕ) : 
  (total : ℚ) / 3 = 60 ∧  -- One third of fish are blue
  (total : ℚ) / 4 ≤ (total : ℚ) / 3 ∧  -- One fourth of fish are yellow
  (total : ℚ) - ((total : ℚ) / 3 + (total : ℚ) / 4) = 45 ∧  -- The rest are red
  (60 : ℚ) / 2 = 30 ∧  -- 50% of blue fish have spots
  30 * (100 : ℚ) / 60 = 50 ∧  -- Verify 50% of blue fish have spots
  9 * (100 : ℚ) / 45 = 20  -- Verify 20% of red fish have spots
  → total = 140 := by
sorry

end NUMINAMATH_CALUDE_aquarium_fish_count_l1304_130418


namespace NUMINAMATH_CALUDE_survey_respondents_l1304_130431

theorem survey_respondents (preferred_x : ℕ) (ratio_x : ℕ) (ratio_y : ℕ) : 
  preferred_x = 150 → ratio_x = 5 → ratio_y = 1 → 
  ∃ (total : ℕ), total = preferred_x + (preferred_x * ratio_y) / ratio_x ∧ total = 180 := by
  sorry

end NUMINAMATH_CALUDE_survey_respondents_l1304_130431


namespace NUMINAMATH_CALUDE_petya_wins_l1304_130466

/-- Represents a 7x7 game board --/
def GameBoard := Fin 7 → Fin 7 → Option (Fin 7)

/-- Checks if a move is valid on the given board --/
def is_valid_move (board : GameBoard) (row col : Fin 7) (digit : Fin 7) : Prop :=
  (∀ i : Fin 7, board i col ≠ some digit) ∧
  (∀ j : Fin 7, board row j ≠ some digit)

/-- Represents a player's strategy --/
def Strategy := GameBoard → Option (Fin 7 × Fin 7 × Fin 7)

/-- Defines a winning strategy for the first player --/
def winning_strategy (s : Strategy) : Prop :=
  ∀ (board : GameBoard),
    (∃ row col digit, is_valid_move board row col digit) →
    ∃ row col digit, s board = some (row, col, digit) ∧ is_valid_move board row col digit

theorem petya_wins : ∃ s : Strategy, winning_strategy s :=
  sorry

end NUMINAMATH_CALUDE_petya_wins_l1304_130466


namespace NUMINAMATH_CALUDE_pears_left_l1304_130485

def initial_pears : ℕ := 35
def given_pears : ℕ := 28

theorem pears_left : initial_pears - given_pears = 7 := by
  sorry

end NUMINAMATH_CALUDE_pears_left_l1304_130485


namespace NUMINAMATH_CALUDE_square_area_ratio_l1304_130452

theorem square_area_ratio (s₂ : ℝ) (h : s₂ > 0) : 
  let s₁ := 4 * s₂
  (s₁ * s₁) / (s₂ * s₂) = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1304_130452


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l1304_130494

/-- Given a cubic polynomial with three distinct real roots, 
    the equation formed by its product with its derivative 
    equals the square of its derivative has exactly two distinct real solutions. -/
theorem cubic_equation_solutions (a b c d : ℝ) (h : ∃ α β γ : ℝ, α < β ∧ β < γ ∧ 
  ∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = α ∨ x = β ∨ x = γ) :
  ∃! (s t : ℝ), s < t ∧ 
    ∀ x, 4 * (a * x^3 + b * x^2 + c * x + d) * (3 * a * x + b) = (3 * a * x^2 + 2 * b * x + c)^2 
    ↔ x = s ∨ x = t :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l1304_130494


namespace NUMINAMATH_CALUDE_total_students_on_ride_l1304_130459

theorem total_students_on_ride (seats_per_ride : ℕ) (empty_seats : ℕ) (num_rides : ℕ) : 
  seats_per_ride = 15 → empty_seats = 3 → num_rides = 18 →
  (seats_per_ride - empty_seats) * num_rides = 216 := by
  sorry

end NUMINAMATH_CALUDE_total_students_on_ride_l1304_130459


namespace NUMINAMATH_CALUDE_work_trip_speed_l1304_130401

/-- Proves that given a round trip of 3 hours, where the return journey takes 1.2 hours at 120 km/h,
    and the journey to work takes 1.8 hours, the average speed to work is 80 km/h. -/
theorem work_trip_speed (total_time : ℝ) (return_time : ℝ) (return_speed : ℝ) (to_work_time : ℝ)
    (h1 : total_time = 3)
    (h2 : return_time = 1.2)
    (h3 : return_speed = 120)
    (h4 : to_work_time = 1.8)
    (h5 : total_time = return_time + to_work_time) :
    (return_speed * return_time) / to_work_time = 80 := by
  sorry

end NUMINAMATH_CALUDE_work_trip_speed_l1304_130401


namespace NUMINAMATH_CALUDE_rice_mixture_price_l1304_130441

/-- Represents the price of rice in Rupees per kilogram -/
@[ext] structure RicePrice where
  price : ℝ

/-- Represents a mixture of two types of rice -/
structure RiceMixture where
  price1 : RicePrice
  price2 : RicePrice
  ratio : ℝ
  mixtureCost : ℝ

/-- The theorem statement -/
theorem rice_mixture_price (mix : RiceMixture) 
  (h1 : mix.price1.price = 16)
  (h2 : mix.ratio = 3)
  (h3 : mix.mixtureCost = 18) :
  mix.price2.price = 24 := by
  sorry

end NUMINAMATH_CALUDE_rice_mixture_price_l1304_130441


namespace NUMINAMATH_CALUDE_brent_initial_lollipops_l1304_130421

/-- The number of lollipops Brent initially received -/
def initial_lollipops : ℕ := sorry

/-- The number of Kit-Kat bars Brent received -/
def kit_kat : ℕ := 5

/-- The number of Hershey kisses Brent received -/
def hershey_kisses : ℕ := 3 * kit_kat

/-- The number of boxes of Nerds Brent received -/
def nerds : ℕ := 8

/-- The number of Baby Ruths Brent had -/
def baby_ruths : ℕ := 10

/-- The number of Reese's Peanut Butter Cups Brent had -/
def reeses_cups : ℕ := baby_ruths / 2

/-- The number of lollipops Brent gave to his sister -/
def lollipops_given : ℕ := 5

/-- The total number of candy pieces Brent had after giving away lollipops -/
def remaining_candy : ℕ := 49

theorem brent_initial_lollipops :
  initial_lollipops = 11 :=
by sorry

end NUMINAMATH_CALUDE_brent_initial_lollipops_l1304_130421


namespace NUMINAMATH_CALUDE_mary_regular_hours_l1304_130460

/-- Represents Mary's work schedule and pay structure --/
structure MaryWork where
  maxHours : Nat
  regularRate : ℝ
  overtimeRateIncrease : ℝ
  maxEarnings : ℝ

/-- Calculates Mary's earnings based on regular hours worked --/
def calculateEarnings (work : MaryWork) (regularHours : ℝ) : ℝ :=
  let overtimeRate := work.regularRate * (1 + work.overtimeRateIncrease)
  let overtimeHours := work.maxHours - regularHours
  regularHours * work.regularRate + overtimeHours * overtimeRate

/-- Theorem stating that Mary works 20 hours at her regular rate to maximize earnings --/
theorem mary_regular_hours (work : MaryWork)
    (h1 : work.maxHours = 50)
    (h2 : work.regularRate = 8)
    (h3 : work.overtimeRateIncrease = 0.25)
    (h4 : work.maxEarnings = 460) :
    ∃ (regularHours : ℝ), regularHours = 20 ∧
    calculateEarnings work regularHours = work.maxEarnings :=
  sorry


end NUMINAMATH_CALUDE_mary_regular_hours_l1304_130460


namespace NUMINAMATH_CALUDE_smallest_y_makes_perfect_cube_no_smaller_y_exists_smallest_y_is_perfect_cube_l1304_130479

def x : ℕ := 7 * 24 * 54

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def smallest_y : ℕ := 1764

theorem smallest_y_makes_perfect_cube :
  (∀ y : ℕ, y < smallest_y → ¬ is_perfect_cube (x * y)) ∧
  is_perfect_cube (x * smallest_y) := by sorry

theorem no_smaller_y_exists (y : ℕ) (h : y < smallest_y) :
  ¬ is_perfect_cube (x * y) := by sorry

theorem smallest_y_is_perfect_cube :
  is_perfect_cube (x * smallest_y) := by sorry

end NUMINAMATH_CALUDE_smallest_y_makes_perfect_cube_no_smaller_y_exists_smallest_y_is_perfect_cube_l1304_130479


namespace NUMINAMATH_CALUDE_function_inequality_l1304_130445

theorem function_inequality (f : ℝ → ℝ) :
  (∀ x ≥ 1, f x ≤ x) →
  (∀ x ≥ 1, f (2 * x) / Real.sqrt 2 ≤ f x) →
  (∀ x ≥ 1, f x < Real.sqrt (2 * x)) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1304_130445


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_3_4_2012_l1304_130447

def arithmetic_sequence_count (a₁ : ℕ) (d : ℕ) (max : ℕ) : ℕ :=
  (max - a₁) / d + 1

theorem arithmetic_sequence_count_3_4_2012 :
  arithmetic_sequence_count 3 4 2012 = 502 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_3_4_2012_l1304_130447


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1304_130471

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) :
  x^3 + y^3 = 1008 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1304_130471


namespace NUMINAMATH_CALUDE_unique_parallel_line_l1304_130486

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (lies_on : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (on_plane : Point → Plane → Prop)
variable (passes_through : Line → Point → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem unique_parallel_line 
  (α β : Plane) (a : Line) (M : Point) :
  parallel α β → 
  lies_on a α → 
  on_plane M β → 
  ∃! l : Line, passes_through l M ∧ line_parallel l a :=
sorry

end NUMINAMATH_CALUDE_unique_parallel_line_l1304_130486


namespace NUMINAMATH_CALUDE_chosen_number_proof_l1304_130415

theorem chosen_number_proof :
  ∀ x : ℝ, (x / 6 - 15 = 5) → x = 120 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l1304_130415


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_l1304_130489

theorem fraction_of_fraction_of_fraction (n : ℚ) : n = 72 → (1/2 : ℚ) * (1/3 : ℚ) * (1/6 : ℚ) * n = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_l1304_130489


namespace NUMINAMATH_CALUDE_quadratic_one_solution_find_m_l1304_130427

/-- A quadratic equation ax² + bx + c = 0 has exactly one solution if and only if its discriminant b² - 4ac = 0 -/
theorem quadratic_one_solution (a b c : ℝ) (h : a ≠ 0) :
  (∃! x, a * x^2 + b * x + c = 0) ↔ b^2 - 4*a*c = 0 := by sorry

theorem find_m : ∃ m : ℚ, (∃! x : ℝ, 3 * x^2 - 7 * x + m = 0) → m = 49/12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_find_m_l1304_130427


namespace NUMINAMATH_CALUDE_investment_partnership_problem_l1304_130419

/-- Investment partnership problem -/
theorem investment_partnership_problem 
  (a b c d : ℝ) -- Investments of partners A, B, C, and D
  (total_profit : ℝ) -- Total profit
  (ha : a = 3 * b) -- A invests 3 times as much as B
  (hb : b = (2/3) * c) -- B invests two-thirds of what C invests
  (hd : d = (1/2) * a) -- D invests half as much as A
  (hp : total_profit = 19900) -- Total profit is Rs.19900
  : b * total_profit / (a + b + c + d) = 2842.86 := by
  sorry

end NUMINAMATH_CALUDE_investment_partnership_problem_l1304_130419


namespace NUMINAMATH_CALUDE_simple_interest_double_l1304_130423

/-- The factor by which a sum of money increases under simple interest -/
def simple_interest_factor (rate : ℝ) (time : ℝ) : ℝ :=
  1 + rate * time

theorem simple_interest_double :
  simple_interest_factor 0.1 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_double_l1304_130423


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1304_130410

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 54 →
  volume = (surface_area / 6) * (surface_area / 6).sqrt →
  volume = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1304_130410


namespace NUMINAMATH_CALUDE_gcd_problem_l1304_130461

theorem gcd_problem (a : ℤ) (h : 2142 ∣ a) : 
  Nat.gcd (Int.natAbs ((a^2 + 11*a + 28) : ℤ)) (Int.natAbs ((a + 6) : ℤ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1304_130461


namespace NUMINAMATH_CALUDE_watermelon_seeds_theorem_l1304_130411

/-- Calculates the total number of seeds in three watermelons -/
def total_seeds (slices1 slices2 slices3 seeds_per_slice1 seeds_per_slice2 seeds_per_slice3 : ℕ) : ℕ :=
  slices1 * seeds_per_slice1 + slices2 * seeds_per_slice2 + slices3 * seeds_per_slice3

/-- Proves that the total number of seeds in the given watermelons is 6800 -/
theorem watermelon_seeds_theorem :
  total_seeds 40 30 50 60 80 40 = 6800 := by
  sorry

#eval total_seeds 40 30 50 60 80 40

end NUMINAMATH_CALUDE_watermelon_seeds_theorem_l1304_130411


namespace NUMINAMATH_CALUDE_fourth_root_unity_sum_l1304_130468

theorem fourth_root_unity_sum (ζ : ℂ) (h : ζ^4 = 1) (h_nonreal : ζ ≠ 1 ∧ ζ ≠ -1) :
  (1 - ζ + ζ^3)^4 + (1 + ζ - ζ^3)^4 = -14 - 48 * I :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_unity_sum_l1304_130468


namespace NUMINAMATH_CALUDE_tetrahedron_subdivision_existence_l1304_130435

theorem tetrahedron_subdivision_existence :
  ∃ (n : ℕ), (1 / 2 : ℝ) ^ n < (1 / 100 : ℝ) := by sorry

end NUMINAMATH_CALUDE_tetrahedron_subdivision_existence_l1304_130435


namespace NUMINAMATH_CALUDE_square_minus_product_identity_l1304_130426

theorem square_minus_product_identity (x y : ℝ) :
  (2*x - 3*y)^2 - (2*x + 3*y)*(2*x - 3*y) = -12*x*y + 18*y^2 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_identity_l1304_130426


namespace NUMINAMATH_CALUDE_carpenter_table_difference_carpenter_table_difference_proof_l1304_130402

theorem carpenter_table_difference : ℕ → ℕ → ℕ → Prop :=
  fun this_month total difference =>
    this_month = 10 →
    total = 17 →
    difference = this_month - (total - this_month) →
    difference = 3

-- The proof is omitted
theorem carpenter_table_difference_proof : carpenter_table_difference 10 17 3 := by
  sorry

end NUMINAMATH_CALUDE_carpenter_table_difference_carpenter_table_difference_proof_l1304_130402


namespace NUMINAMATH_CALUDE_chocolate_chip_per_recipe_l1304_130453

/-- Given that 23 recipes require 46 cups of chocolate chips in total,
    prove that the number of cups of chocolate chips needed for one recipe is 2. -/
theorem chocolate_chip_per_recipe :
  let total_recipes : ℕ := 23
  let total_chips : ℕ := 46
  (total_chips / total_recipes : ℚ) = 2 := by sorry

end NUMINAMATH_CALUDE_chocolate_chip_per_recipe_l1304_130453


namespace NUMINAMATH_CALUDE_shifted_line_not_in_third_quadrant_l1304_130443

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shifts a line horizontally -/
def shift_line (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.slope * shift + l.intercept }

/-- Checks if a line passes through the third quadrant -/
def passes_through_third_quadrant (l : Line) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ y = l.slope * x + l.intercept

/-- The original line y = -2x - 1 -/
def original_line : Line :=
  { slope := -2, intercept := -1 }

/-- The amount of right shift -/
def shift_amount : ℝ := 3

theorem shifted_line_not_in_third_quadrant :
  ¬ passes_through_third_quadrant (shift_line original_line shift_amount) := by
  sorry

end NUMINAMATH_CALUDE_shifted_line_not_in_third_quadrant_l1304_130443


namespace NUMINAMATH_CALUDE_janet_initial_lives_l1304_130488

theorem janet_initial_lives :
  ∀ (initial : ℕ),
  (initial - 16 + 32 = 54) →
  initial = 38 :=
by sorry

end NUMINAMATH_CALUDE_janet_initial_lives_l1304_130488


namespace NUMINAMATH_CALUDE_point_Q_coordinate_l1304_130458

theorem point_Q_coordinate (Q : ℝ) : (|Q - 0| = 3) → (Q = 3 ∨ Q = -3) := by
  sorry

end NUMINAMATH_CALUDE_point_Q_coordinate_l1304_130458


namespace NUMINAMATH_CALUDE_equation_solution_l1304_130482

theorem equation_solution : ∃ x : ℝ, 6*x - 4*x = 380 - 10*(x + 2) ∧ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1304_130482


namespace NUMINAMATH_CALUDE_bottle_caps_difference_l1304_130454

/-- Represents the number of bottle caps in various states --/
structure BottleCaps where
  thrown_away : ℕ
  found : ℕ
  final_collection : ℕ

/-- Theorem stating the difference between found and thrown away bottle caps --/
theorem bottle_caps_difference (caps : BottleCaps)
  (h1 : caps.thrown_away = 6)
  (h2 : caps.found = 50)
  (h3 : caps.final_collection = 60) :
  caps.found - caps.thrown_away = 44 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_difference_l1304_130454


namespace NUMINAMATH_CALUDE_two_point_form_always_valid_two_point_form_works_for_vertical_lines_l1304_130490

/-- A line in a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of a line passing through two points --/
def line_equation (p1 p2 : Point) (x y : ℝ) : Prop :=
  (y - p1.y) * (p2.x - p1.x) = (x - p1.x) * (p2.y - p1.y)

/-- Theorem: The two-point form of a line equation is always valid --/
theorem two_point_form_always_valid (p1 p2 : Point) (h : p1 ≠ p2) :
  ∃ (l : Line), ∀ (x y : ℝ), (y = l.slope * x + l.intercept) ↔ line_equation p1 p2 x y :=
sorry

/-- Corollary: The two-point form works even for vertical lines --/
theorem two_point_form_works_for_vertical_lines (p1 p2 : Point) (h : p1.x = p2.x) (h' : p1 ≠ p2) :
  ∀ (y : ℝ), ∃ (x : ℝ), line_equation p1 p2 x y :=
sorry

end NUMINAMATH_CALUDE_two_point_form_always_valid_two_point_form_works_for_vertical_lines_l1304_130490


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l1304_130429

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x^2 > 4}

-- Define set N
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- The theorem to prove
theorem intersection_complement_theorem :
  N ∩ (Set.compl M) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l1304_130429


namespace NUMINAMATH_CALUDE_divisor_count_not_25323_or_25322_l1304_130469

def sequential_number (n : ℕ) : ℕ :=
  -- Definition of the number formed by writing integers from 1 to n sequentially
  sorry

def count_divisors (n : ℕ) : ℕ :=
  -- Definition to count the number of divisors of n
  sorry

theorem divisor_count_not_25323_or_25322 :
  let N := sequential_number 1975
  (count_divisors N ≠ 25323) ∧ (count_divisors N ≠ 25322) := by
  sorry

end NUMINAMATH_CALUDE_divisor_count_not_25323_or_25322_l1304_130469


namespace NUMINAMATH_CALUDE_peter_and_susan_money_l1304_130409

/-- The total amount of money Peter and Susan have together -/
def total_money (peter_amount susan_amount : ℚ) : ℚ :=
  peter_amount + susan_amount

/-- Theorem stating that Peter and Susan have 0.65 dollars altogether -/
theorem peter_and_susan_money :
  total_money (2/5) (1/4) = 13/20 := by
  sorry

end NUMINAMATH_CALUDE_peter_and_susan_money_l1304_130409


namespace NUMINAMATH_CALUDE_cube_sqrt_16_equals_8_times_8_l1304_130407

theorem cube_sqrt_16_equals_8_times_8 : 
  (8 : ℝ) * 8 = (Real.sqrt 16)^3 := by sorry

end NUMINAMATH_CALUDE_cube_sqrt_16_equals_8_times_8_l1304_130407


namespace NUMINAMATH_CALUDE_salary_reduction_percentage_l1304_130448

theorem salary_reduction_percentage (S : ℝ) (R : ℝ) (h : S > 0) : 
  S = (S - (R/100) * S) * (1 + 25/100) → R = 20 :=
by sorry

end NUMINAMATH_CALUDE_salary_reduction_percentage_l1304_130448


namespace NUMINAMATH_CALUDE_binomial_6_2_l1304_130478

theorem binomial_6_2 : Nat.choose 6 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_binomial_6_2_l1304_130478


namespace NUMINAMATH_CALUDE_zero_not_necessarily_in_2_5_l1304_130439

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f having its only zero in (1,3)
def has_only_zero_in_open_interval (f : ℝ → ℝ) : Prop :=
  ∃ x, 1 < x ∧ x < 3 ∧ f x = 0 ∧ ∀ y, f y = 0 → (1 < y ∧ y < 3)

-- State the theorem
theorem zero_not_necessarily_in_2_5 
  (h : has_only_zero_in_open_interval f) : 
  ¬(∀ f, has_only_zero_in_open_interval f → ∃ x, 2 < x ∧ x < 5 ∧ f x = 0) :=
sorry

end NUMINAMATH_CALUDE_zero_not_necessarily_in_2_5_l1304_130439


namespace NUMINAMATH_CALUDE_smallest_m_is_13_l1304_130476

def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

def has_nth_root_of_unity (n : ℕ) : Prop :=
  ∃ z ∈ T, z^n = 1

theorem smallest_m_is_13 :
  (∃ m : ℕ, m > 0 ∧ ∀ n ≥ m, has_nth_root_of_unity n) ∧
  (∀ m < 13, ∃ n ≥ m, ¬has_nth_root_of_unity n) ∧
  (∀ n ≥ 13, has_nth_root_of_unity n) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_13_l1304_130476


namespace NUMINAMATH_CALUDE_student_marks_average_l1304_130499

/-- Given a student's marks in mathematics, physics, and chemistry, 
    prove that the average of mathematics and chemistry marks is 20. -/
theorem student_marks_average (M P C : ℝ) 
  (h1 : M + P = 20)
  (h2 : C = P + 20) : 
  (M + C) / 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_student_marks_average_l1304_130499


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1304_130498

-- Define the arithmetic sequence
def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := 1 + (n - 1) * d

-- Define the theorem
theorem arithmetic_sequence_difference
  (d : ℝ) (m n : ℕ) 
  (h1 : d ≠ 0)
  (h2 : arithmetic_sequence d 2 * arithmetic_sequence d 6 = (arithmetic_sequence d 4 - 2)^2)
  (h3 : m > n)
  (h4 : m - n = 10) :
  arithmetic_sequence d m - arithmetic_sequence d n = 30 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1304_130498


namespace NUMINAMATH_CALUDE_rebus_solution_l1304_130462

theorem rebus_solution : 
  ∃! (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧
    A = 4 ∧ B = 7 ∧ C = 6 := by
  sorry

end NUMINAMATH_CALUDE_rebus_solution_l1304_130462


namespace NUMINAMATH_CALUDE_partition_ratio_theorem_l1304_130484

theorem partition_ratio_theorem (n : ℕ) : 
  (∃ (A B : Finset ℕ), 
    (A ∪ B = Finset.range (n^2 + 1) \ {0}) ∧ 
    (A ∩ B = ∅) ∧
    (A.card = B.card) ∧
    ((A.sum id) / (B.sum id) = 39 / 64)) ↔ 
  (∃ k : ℕ, n = 206 * k) ∧ 
  Even n :=
sorry

end NUMINAMATH_CALUDE_partition_ratio_theorem_l1304_130484


namespace NUMINAMATH_CALUDE_square_diagonal_l1304_130455

theorem square_diagonal (area : ℝ) (h : area = 800) :
  ∃ (diagonal : ℝ), diagonal = 40 ∧ diagonal^2 = 2 * area := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_l1304_130455


namespace NUMINAMATH_CALUDE_problem_solution_l1304_130436

def A : Set ℝ := {x | |x - 2| < 3}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

theorem problem_solution :
  (∀ x, x ∈ (A ∩ (Set.univ \ B 3)) ↔ 3 ≤ x ∧ x < 5) ∧
  (A ∩ B 8 = {x | -1 < x ∧ x < 4}) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1304_130436


namespace NUMINAMATH_CALUDE_overtime_pay_is_3_20_l1304_130414

/-- Calculates the overtime pay rate given the following conditions:
  * Regular week has 5 working days
  * Regular working hours per day is 8
  * Regular pay rate is 2.40 rupees per hour
  * Total earnings in 4 weeks is 432 rupees
  * Total hours worked in 4 weeks is 175
-/
def overtime_pay_rate (
  regular_days_per_week : ℕ)
  (regular_hours_per_day : ℕ)
  (regular_pay_rate : ℚ)
  (total_earnings : ℚ)
  (total_hours : ℕ) : ℚ :=
by
  sorry

/-- Theorem stating that the overtime pay rate is 3.20 rupees per hour -/
theorem overtime_pay_is_3_20 :
  overtime_pay_rate 5 8 (240/100) 432 175 = 320/100 :=
by
  sorry

end NUMINAMATH_CALUDE_overtime_pay_is_3_20_l1304_130414


namespace NUMINAMATH_CALUDE_product_expansion_l1304_130440

theorem product_expansion (x : ℝ) : 
  (x^2 + 3*x - 4) * (2*x^2 - x + 5) = 2*x^4 + 5*x^3 - 6*x^2 + 19*x - 20 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l1304_130440


namespace NUMINAMATH_CALUDE_division_sum_equals_111_l1304_130412

theorem division_sum_equals_111 : (111 / 3) + (222 / 6) + (333 / 9) = 111 := by
  sorry

end NUMINAMATH_CALUDE_division_sum_equals_111_l1304_130412


namespace NUMINAMATH_CALUDE_arrangement_count_is_2028_l1304_130428

/-- Represents the set of files that can be arranged after lunch -/
def RemainingFiles : Finset ℕ := Finset.range 9 ∪ {12}

/-- The number of ways to arrange a subset of files from {1,2,...,9,12} -/
def ArrangementCount : ℕ := sorry

/-- Theorem stating that the number of different arrangements is 2028 -/
theorem arrangement_count_is_2028 : ArrangementCount = 2028 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_2028_l1304_130428


namespace NUMINAMATH_CALUDE_frood_game_theorem_l1304_130413

/-- Sum of the first n positive integers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Points earned from eating n froods -/
def eating_points (n : ℕ) : ℕ := 12 * n

/-- The least number of froods for which dropping them earns more points than eating them -/
def least_froods : ℕ := 24

theorem frood_game_theorem :
  least_froods = 24 ∧
  (∀ n : ℕ, n < least_froods → triangular_number n ≤ eating_points n) ∧
  triangular_number least_froods > eating_points least_froods :=
by sorry

end NUMINAMATH_CALUDE_frood_game_theorem_l1304_130413


namespace NUMINAMATH_CALUDE_parking_arrangement_l1304_130444

/-- The number of ways to park cars in a row with empty spaces -/
def park_cars (total_spaces : ℕ) (cars : ℕ) (empty_spaces : ℕ) : ℕ :=
  (total_spaces - empty_spaces + 1) * (cars.factorial)

theorem parking_arrangement :
  park_cars 8 4 4 = 120 :=
by sorry

end NUMINAMATH_CALUDE_parking_arrangement_l1304_130444


namespace NUMINAMATH_CALUDE_largest_product_sum_1976_l1304_130487

theorem largest_product_sum_1976 (n : ℕ) (h : n > 0) :
  (∃ (factors : List ℕ), factors.sum = 1976 ∧ factors.prod = n) →
  n ≤ 2 * 3^658 := by
sorry

end NUMINAMATH_CALUDE_largest_product_sum_1976_l1304_130487


namespace NUMINAMATH_CALUDE_michaels_ride_l1304_130480

/-- Calculates the total distance traveled by a cyclist given their speed and time -/
def total_distance (speed : ℚ) (time : ℚ) : ℚ :=
  speed * time

/-- Represents Michael's cycling scenario -/
theorem michaels_ride (total_time : ℚ) (speed : ℚ) 
    (h1 : total_time = 40) 
    (h2 : speed = 2 / 5) : 
  total_distance speed total_time = 16 := by
  sorry

#eval total_distance (2/5) 40

end NUMINAMATH_CALUDE_michaels_ride_l1304_130480


namespace NUMINAMATH_CALUDE_point_line_plane_membership_l1304_130472

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relations for a point being on a line and within a plane
variable (on_line : Point → Line → Prop)
variable (in_plane : Point → Plane → Prop)

-- Define specific points, line, and plane
variable (A E F : Point)
variable (l : Line)
variable (ABC : Plane)

-- State the theorem
theorem point_line_plane_membership :
  (on_line A l) ∧ (in_plane E ABC) ∧ (in_plane F ABC) :=
sorry

end NUMINAMATH_CALUDE_point_line_plane_membership_l1304_130472


namespace NUMINAMATH_CALUDE_alpha_squared_plus_one_times_one_plus_cos_two_alpha_equals_two_l1304_130400

theorem alpha_squared_plus_one_times_one_plus_cos_two_alpha_equals_two
  (α : ℝ) (h1 : α ≠ 0) (h2 : α + Real.tan α = 0) :
  (α^2 + 1) * (1 + Real.cos (2 * α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_squared_plus_one_times_one_plus_cos_two_alpha_equals_two_l1304_130400


namespace NUMINAMATH_CALUDE_abs_sum_lt_abs_diff_when_product_negative_l1304_130406

theorem abs_sum_lt_abs_diff_when_product_negative (a b : ℝ) : 
  a * b < 0 → |a + b| < |a - b| := by
sorry

end NUMINAMATH_CALUDE_abs_sum_lt_abs_diff_when_product_negative_l1304_130406


namespace NUMINAMATH_CALUDE_inequality_range_l1304_130473

theorem inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3/8 < 0) ↔ -3 < k ∧ k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l1304_130473


namespace NUMINAMATH_CALUDE_fraction_addition_l1304_130467

theorem fraction_addition : (1 : ℚ) / 6 + (5 : ℚ) / 12 = (7 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1304_130467


namespace NUMINAMATH_CALUDE_min_value_of_function_l1304_130417

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≥ 3 ∧ ∃ y > 1, y + 1 / (y - 1) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1304_130417


namespace NUMINAMATH_CALUDE_both_selected_l1304_130442

-- Define the probabilities of selection for Ram and Ravi
def prob_ram : ℚ := 1/7
def prob_ravi : ℚ := 1/5

-- Define the probability of both being selected
def prob_both : ℚ := prob_ram * prob_ravi

-- Theorem: The probability of both Ram and Ravi being selected is 1/35
theorem both_selected : prob_both = 1/35 := by
  sorry

end NUMINAMATH_CALUDE_both_selected_l1304_130442


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1304_130449

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 6 ∧ 
  (x₁^2 - 7*x₁ + 6 = 0) ∧ (x₂^2 - 7*x₂ + 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1304_130449


namespace NUMINAMATH_CALUDE_maximum_marks_calculation_l1304_130430

theorem maximum_marks_calculation (passing_threshold : ℝ) (scored_marks : ℕ) (shortfall : ℕ) : 
  passing_threshold = 30 / 100 →
  scored_marks = 212 →
  shortfall = 16 →
  ∃ (total_marks : ℕ), total_marks = 760 ∧ 
    (scored_marks + shortfall : ℝ) / total_marks = passing_threshold :=
by sorry

end NUMINAMATH_CALUDE_maximum_marks_calculation_l1304_130430


namespace NUMINAMATH_CALUDE_brothers_ages_l1304_130456

theorem brothers_ages (x y : ℕ) : 
  x + y = 16 → 
  2 * (x + 4) = y + 4 → 
  ∃ (younger older : ℕ), younger = x ∧ older = y ∧ younger < older :=
by
  sorry

end NUMINAMATH_CALUDE_brothers_ages_l1304_130456


namespace NUMINAMATH_CALUDE_pictures_per_album_l1304_130404

theorem pictures_per_album 
  (total_pictures : ℕ) 
  (phone_pictures camera_pictures : ℕ) 
  (num_albums : ℕ) 
  (h1 : total_pictures = phone_pictures + camera_pictures)
  (h2 : phone_pictures = 5)
  (h3 : camera_pictures = 35)
  (h4 : num_albums = 8)
  (h5 : total_pictures % num_albums = 0) :
  total_pictures / num_albums = 5 := by
sorry

end NUMINAMATH_CALUDE_pictures_per_album_l1304_130404


namespace NUMINAMATH_CALUDE_pancake_fundraiser_l1304_130416

/-- The civic league's pancake breakfast fundraiser --/
theorem pancake_fundraiser 
  (pancake_price : ℝ) 
  (bacon_price : ℝ) 
  (pancake_stacks : ℕ) 
  (bacon_slices : ℕ) 
  (h1 : pancake_price = 4)
  (h2 : bacon_price = 2)
  (h3 : pancake_stacks = 60)
  (h4 : bacon_slices = 90) :
  pancake_price * (pancake_stacks : ℝ) + bacon_price * (bacon_slices : ℝ) = 420 :=
by sorry

end NUMINAMATH_CALUDE_pancake_fundraiser_l1304_130416


namespace NUMINAMATH_CALUDE_perimeter_special_region_l1304_130463

/-- The perimeter of a region bounded by three semicircular arcs and one three-quarter circular arc,
    constructed on the sides of a square with side length 1/π, is equal to 2.25. -/
theorem perimeter_special_region :
  let square_side : ℝ := 1 / Real.pi
  let semicircle_perimeter : ℝ := Real.pi * square_side / 2
  let three_quarter_circle_perimeter : ℝ := 3 * Real.pi * square_side / 4
  let total_perimeter : ℝ := 3 * semicircle_perimeter + three_quarter_circle_perimeter
  total_perimeter = 2.25 := by sorry

end NUMINAMATH_CALUDE_perimeter_special_region_l1304_130463


namespace NUMINAMATH_CALUDE_trigonometric_equation_solutions_l1304_130495

theorem trigonometric_equation_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ)),
    (∀ (a b c : ℝ), (a, b, c) ∈ solutions ↔
      (c ∈ Set.Icc 0 (2 * Real.pi) ∧
       ∀ x : ℝ, 2 * Real.sin (3 * x - Real.pi / 3) = a * Real.sin (b * x + c))) ∧
    Finset.card solutions = 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solutions_l1304_130495


namespace NUMINAMATH_CALUDE_share_of_y_l1304_130405

def total_amount : ℕ := 690
def ratio_x : ℕ := 5
def ratio_y : ℕ := 7
def ratio_z : ℕ := 11

theorem share_of_y : 
  (total_amount * ratio_y) / (ratio_x + ratio_y + ratio_z) = 210 := by
  sorry

end NUMINAMATH_CALUDE_share_of_y_l1304_130405


namespace NUMINAMATH_CALUDE_solution_of_cubic_system_l1304_130496

theorem solution_of_cubic_system :
  ∀ x y : ℝ, x + y = 1 ∧ x^3 + y^3 = 19 →
  (x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3) := by
sorry

end NUMINAMATH_CALUDE_solution_of_cubic_system_l1304_130496


namespace NUMINAMATH_CALUDE_train_speed_is_6_l1304_130491

/-- The speed of a train in km/hr, given its length and time to cross a pole -/
def train_speed (length : Float) (time : Float) : Float :=
  (length / time) * 3.6

/-- Theorem: The speed of the train is 6 km/hr -/
theorem train_speed_is_6 :
  let length : Float := 3.3333333333333335
  let time : Float := 2
  train_speed length time = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_is_6_l1304_130491


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1304_130483

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The terms a_2, (1/2)a_3, a_1 form an arithmetic sequence. -/
def ArithmeticSubsequence (a : ℕ → ℝ) : Prop :=
  a 2 - (1/2 * a 3) = (1/2 * a 3) - a 1

theorem geometric_sequence_property (a : ℕ → ℝ) :
  GeometricSequence a →
  ArithmeticSubsequence a →
  (a 3 + a 4) / (a 4 + a 5) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1304_130483


namespace NUMINAMATH_CALUDE_vector_conclusions_l1304_130425

-- Define the vector space
variable {V : Type*} [AddCommGroup V]

-- Define the vectors
variable (O D E M : V)

-- Define the given equation
axiom given_equation : D - O + (E - O) = M - O

-- Theorem to prove the three correct conclusions
theorem vector_conclusions :
  (M - O + (D - O) = E - O) ∧
  (M - O - (E - O) = D - O) ∧
  ((O - D) + (O - E) = O - M) := by
  sorry

end NUMINAMATH_CALUDE_vector_conclusions_l1304_130425


namespace NUMINAMATH_CALUDE_tv_weight_difference_l1304_130446

def bill_tv_length : ℕ := 48
def bill_tv_width : ℕ := 100
def bob_tv_length : ℕ := 70
def bob_tv_width : ℕ := 60
def weight_per_square_inch : ℚ := 4 / 1
def ounces_per_pound : ℕ := 16

theorem tv_weight_difference :
  let bill_area := bill_tv_length * bill_tv_width
  let bob_area := bob_tv_length * bob_tv_width
  let area_difference := max bill_area bob_area - min bill_area bob_area
  let weight_difference_oz := area_difference * weight_per_square_inch
  let weight_difference_lbs := weight_difference_oz / ounces_per_pound
  weight_difference_lbs = 150 := by sorry

end NUMINAMATH_CALUDE_tv_weight_difference_l1304_130446


namespace NUMINAMATH_CALUDE_apple_distribution_l1304_130497

/-- The number of ways to distribute n apples among k people, with each person receiving at least m apples -/
def distribution_ways (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- The problem statement -/
theorem apple_distribution :
  distribution_ways 30 3 3 = 253 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l1304_130497


namespace NUMINAMATH_CALUDE_sum_of_ratio_terms_l1304_130450

-- Define the points
variable (A B C D O P X Y : ℝ × ℝ)

-- Define the lengths
def length_AD : ℝ := 10
def length_AO : ℝ := 10
def length_OB : ℝ := 10
def length_BC : ℝ := 10
def length_AB : ℝ := 12
def length_DO : ℝ := 12
def length_OC : ℝ := 12

-- Define the conditions
axiom isosceles_DAO : length_AD = length_AO
axiom isosceles_AOB : length_AO = length_OB
axiom isosceles_OBC : length_OB = length_BC
axiom P_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B
axiom OP_perpendicular_AB : (O.1 - P.1) * (B.1 - A.1) + (O.2 - P.2) * (B.2 - A.2) = 0
axiom X_midpoint_AD : X = ((A.1 + D.1) / 2, (A.2 + D.2) / 2)
axiom Y_midpoint_BC : Y = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the areas of trapezoids
def area_ABYX : ℝ := sorry
def area_XYCD : ℝ := sorry

-- Define the ratio of areas
def ratio_areas : ℚ := sorry

-- Theorem to prove
theorem sum_of_ratio_terms : 
  ∃ (p q : ℕ), ratio_areas = p / q ∧ p + q = 12 :=
sorry

end NUMINAMATH_CALUDE_sum_of_ratio_terms_l1304_130450


namespace NUMINAMATH_CALUDE_laura_change_l1304_130493

def change_calculation (pants_cost : ℕ) (pants_count : ℕ) (shirt_cost : ℕ) (shirt_count : ℕ) (amount_given : ℕ) : ℕ :=
  amount_given - (pants_cost * pants_count + shirt_cost * shirt_count)

theorem laura_change : change_calculation 54 2 33 4 250 = 10 := by
  sorry

end NUMINAMATH_CALUDE_laura_change_l1304_130493
