import Mathlib

namespace NUMINAMATH_CALUDE_cricket_bat_selling_price_l173_17326

/-- The selling price of a cricket bat given profit and profit percentage -/
theorem cricket_bat_selling_price 
  (profit : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : profit = 205)
  (h2 : profit_percentage = 31.782945736434108) :
  let cost_price := profit / (profit_percentage / 100)
  let selling_price := cost_price + profit
  selling_price = 850 := by
sorry

end NUMINAMATH_CALUDE_cricket_bat_selling_price_l173_17326


namespace NUMINAMATH_CALUDE_f_properties_l173_17350

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := (x - 2) * Real.log x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := Real.log x - 2 / x + 1

-- Theorem statement
theorem f_properties :
  (∃! x : ℝ, x ∈ Set.Ioo 1 2 ∧ f' x = 0) ∧
  (∀ x : ℝ, x > 0 → f x > 0) := by
  sorry

end

end NUMINAMATH_CALUDE_f_properties_l173_17350


namespace NUMINAMATH_CALUDE_f_max_min_sum_l173_17386

noncomputable def f (x : ℝ) : ℝ :=
  ((Real.sqrt 1008 * x + Real.sqrt 1009)^2 + Real.sin (2018 * x)) / (2016 * x^2 + 2018)

def has_max_min (f : ℝ → ℝ) (M m : ℝ) : Prop :=
  (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ (∀ x, m ≤ f x) ∧ (∃ x, f x = m)

theorem f_max_min_sum :
  ∃ M m : ℝ, has_max_min f M m ∧ M + m = 1 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_sum_l173_17386


namespace NUMINAMATH_CALUDE_room_population_l173_17397

theorem room_population (total : ℕ) (women : ℕ) (married : ℕ) (unmarried_women : ℕ) :
  women = total / 4 →
  married = 3 * total / 4 →
  unmarried_women ≤ 20 →
  unmarried_women = total - married →
  total = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_room_population_l173_17397


namespace NUMINAMATH_CALUDE_remaining_distance_condition_l173_17394

/-- The total distance between points A and B in kilometers -/
def total_distance : ℕ := 500

/-- Alpha's daily cycling distance in kilometers -/
def alpha_daily_distance : ℕ := 30

/-- Beta's cycling distance on active days in kilometers -/
def beta_active_day_distance : ℕ := 50

/-- The number of days after which the condition is met -/
def condition_day : ℕ := 15

/-- The remaining distance for Alpha after n days -/
def alpha_remaining (n : ℕ) : ℕ := total_distance - n * alpha_daily_distance

/-- The remaining distance for Beta after n days -/
def beta_remaining (n : ℕ) : ℕ := total_distance - n * (beta_active_day_distance / 2)

/-- Theorem stating that on the 15th day, Beta's remaining distance is twice Alpha's -/
theorem remaining_distance_condition :
  beta_remaining condition_day = 2 * alpha_remaining condition_day :=
sorry

end NUMINAMATH_CALUDE_remaining_distance_condition_l173_17394


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l173_17378

/-- For a quadratic equation 9x^2 + kx + 49 = 0 to have exactly one real solution,
    the positive value of k must be 42. -/
theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + k * x + 49 = 0) ↔ k = 42 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l173_17378


namespace NUMINAMATH_CALUDE_equation_solution_l173_17368

theorem equation_solution :
  ∃ x : ℚ, (3 * x - 15) / 4 = (x + 7) / 3 ∧ x = 73 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l173_17368


namespace NUMINAMATH_CALUDE_total_spent_equals_20_l173_17371

def bracelet_price : ℕ := 4
def keychain_price : ℕ := 5
def coloring_book_price : ℕ := 3

def paula_bracelets : ℕ := 2
def paula_keychains : ℕ := 1
def olive_coloring_books : ℕ := 1
def olive_bracelets : ℕ := 1

def paula_total : ℕ := paula_bracelets * bracelet_price + paula_keychains * keychain_price
def olive_total : ℕ := olive_coloring_books * coloring_book_price + olive_bracelets * bracelet_price

theorem total_spent_equals_20 : paula_total + olive_total = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_20_l173_17371


namespace NUMINAMATH_CALUDE_ending_number_proof_l173_17396

theorem ending_number_proof (n : ℕ) (h1 : n > 45) (h2 : ∃ (evens : List ℕ), 
  evens.length = 30 ∧ 
  (∀ m ∈ evens, Even m ∧ m > 45 ∧ m ≤ n) ∧
  (∀ m, 45 < m ∧ m ≤ n ∧ Even m → m ∈ evens)) : 
  n = 104 := by
sorry

end NUMINAMATH_CALUDE_ending_number_proof_l173_17396


namespace NUMINAMATH_CALUDE_rain_probability_l173_17334

theorem rain_probability (p_day : ℝ) (p_consecutive : ℝ) 
  (h1 : p_day = 1/3)
  (h2 : p_consecutive = 1/5) :
  p_consecutive / p_day = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l173_17334


namespace NUMINAMATH_CALUDE_average_songs_in_remaining_sets_l173_17353

def bandRepertoire : ℕ := 30
def firstSetSongs : ℕ := 5
def secondSetSongs : ℕ := 7
def encoreSongs : ℕ := 2
def remainingSets : ℕ := 2

theorem average_songs_in_remaining_sets :
  (bandRepertoire - (firstSetSongs + secondSetSongs + encoreSongs)) / remainingSets = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_songs_in_remaining_sets_l173_17353


namespace NUMINAMATH_CALUDE_man_upstream_speed_l173_17340

/-- Given a man's speed in still water and his speed downstream, 
    calculates his speed upstream -/
def speed_upstream (speed_still : ℝ) (speed_downstream : ℝ) : ℝ :=
  2 * speed_still - speed_downstream

/-- Theorem stating that for a man with speed 60 kmph in still water 
    and 65 kmph downstream, his upstream speed is 55 kmph -/
theorem man_upstream_speed :
  speed_upstream 60 65 = 55 := by
  sorry


end NUMINAMATH_CALUDE_man_upstream_speed_l173_17340


namespace NUMINAMATH_CALUDE_problem_solution_l173_17342

theorem problem_solution (a b : ℝ) 
  (sum_eq : a + b = 12) 
  (diff_sq_eq : a^2 - b^2 = 48) : 
  a - b = 4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l173_17342


namespace NUMINAMATH_CALUDE_period_of_tan_transformed_l173_17301

open Real

/-- The period of the tangent function with a transformed argument -/
theorem period_of_tan_transformed (f : ℝ → ℝ) :
  (∀ x, f x = tan (2 * x / 3)) →
  (∃ p > 0, ∀ x, f (x + p) = f x) ∧
  (∀ q > 0, (∀ x, f (x + q) = f x) → q ≥ 3 * π / 2) :=
sorry

end NUMINAMATH_CALUDE_period_of_tan_transformed_l173_17301


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l173_17365

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3, 5}
def N : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {1, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l173_17365


namespace NUMINAMATH_CALUDE_composite_numbers_l173_17318

theorem composite_numbers (n : ℕ) (h : n > 2) : 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^4 + 2*n^2 + 1 = a * b) ∧ 
  (∃ c d : ℕ, c > 1 ∧ d > 1 ∧ n^4 + n^2 + 1 = c * d) := by
  sorry

#check composite_numbers

end NUMINAMATH_CALUDE_composite_numbers_l173_17318


namespace NUMINAMATH_CALUDE_linear_system_k_values_l173_17314

/-- Given a system of linear equations in two variables x and y,
    prove the value of k under certain conditions. -/
theorem linear_system_k_values (x y k : ℝ) : 
  (3 * x + y = k + 1) →
  (x + 3 * y = 3) →
  (
    ((x * y < 0) → (k = -4)) ∧
    ((x + y < 3 ∧ x - y > 1) → (4 < k ∧ k < 8))
  ) := by sorry

end NUMINAMATH_CALUDE_linear_system_k_values_l173_17314


namespace NUMINAMATH_CALUDE_sum_of_fractions_l173_17356

theorem sum_of_fractions (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : ∃ n : ℤ, (a / b + b / c + c / a : ℚ) = n)
  (h2 : ∃ m : ℤ, (b / a + c / b + a / c : ℚ) = m) :
  (a / b + b / c + c / a : ℚ) = 3 ∨ (a / b + b / c + c / a : ℚ) = -3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l173_17356


namespace NUMINAMATH_CALUDE_quadratic_properties_l173_17391

-- Define the quadratic function
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 4 * m * x + m - 2

-- Define the conditions
theorem quadratic_properties (m : ℝ) (h_m_neq_0 : m ≠ 0) 
  (h_distinct_roots : ∃ M N : ℝ, M ≠ N ∧ quadratic_function m M = 0 ∧ quadratic_function m N = 0)
  (h_passes_through_A : quadratic_function m 3 = 0) :
  -- 1. The value of m is -1
  m = -1 ∧
  -- 2. The vertex coordinates are (2, 1)
  (let vertex_x := 2; let vertex_y := 1;
   quadratic_function m vertex_x = vertex_y ∧
   ∀ x : ℝ, quadratic_function m x ≤ vertex_y) ∧
  -- 3. When m < 0 and MN ≤ 4, the range of m is m < 0
  (m < 0 → ∀ M N : ℝ, M ≠ N → quadratic_function m M = 0 → quadratic_function m N = 0 →
    (M - N)^2 ≤ 4^2 → m < 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l173_17391


namespace NUMINAMATH_CALUDE_bedroom_difference_is_sixty_l173_17341

/-- The difference in square footage between two bedrooms -/
def bedroom_size_difference (total_size martha_size : ℝ) : ℝ :=
  (total_size - martha_size) - martha_size

/-- Theorem: Given the total size of two bedrooms and the size of one bedroom,
    prove that the difference between the two bedroom sizes is 60 sq ft -/
theorem bedroom_difference_is_sixty
  (total_size : ℝ)
  (martha_size : ℝ)
  (h1 : total_size = 300)
  (h2 : martha_size = 120) :
  bedroom_size_difference total_size martha_size = 60 := by
  sorry

end NUMINAMATH_CALUDE_bedroom_difference_is_sixty_l173_17341


namespace NUMINAMATH_CALUDE_opposite_of_2023_l173_17325

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l173_17325


namespace NUMINAMATH_CALUDE_min_value_quadratic_l173_17346

theorem min_value_quadratic (x : ℝ) : 
  (∀ x, x^2 - 6*x + 10 ≥ 1) ∧ (∃ x, x^2 - 6*x + 10 = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l173_17346


namespace NUMINAMATH_CALUDE_sum_reciprocal_inequality_root_inequality_l173_17347

-- Problem 1
theorem sum_reciprocal_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by sorry

-- Problem 2
theorem root_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt (x^2 + y^2) > (x^3 + y^3)^(1/3) := by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_inequality_root_inequality_l173_17347


namespace NUMINAMATH_CALUDE_absolute_sum_zero_implies_value_l173_17364

theorem absolute_sum_zero_implies_value (a b : ℝ) :
  |3*a + b + 5| + |2*a - 2*b - 2| = 0 →
  2*a^2 - 3*a*b = -4 := by
sorry

end NUMINAMATH_CALUDE_absolute_sum_zero_implies_value_l173_17364


namespace NUMINAMATH_CALUDE_sum_remainder_mod_9_l173_17354

theorem sum_remainder_mod_9 : (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_9_l173_17354


namespace NUMINAMATH_CALUDE_vector_parallel_perpendicular_l173_17317

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

theorem vector_parallel_perpendicular (x : ℝ) :
  (∃ k : ℝ, a + 2 • b x = k • (2 • a - b x) → x = 1/2) ∧
  ((a + 2 • b x) • (2 • a - b x) = 0 → x = -2 ∨ x = 7/2) :=
sorry

end NUMINAMATH_CALUDE_vector_parallel_perpendicular_l173_17317


namespace NUMINAMATH_CALUDE_train_crossing_time_l173_17316

-- Define the given parameters
def train_speed_kmph : ℝ := 72
def train_speed_ms : ℝ := 20
def platform_length : ℝ := 260
def time_cross_platform : ℝ := 31

-- Define the theorem
theorem train_crossing_time (train_length : ℝ) 
  (h1 : train_length + platform_length = train_speed_ms * time_cross_platform)
  (h2 : train_speed_kmph * (1000 / 3600) = train_speed_ms) :
  train_length / train_speed_ms = 18 := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_train_crossing_time_l173_17316


namespace NUMINAMATH_CALUDE_catch_game_end_state_l173_17383

/-- Represents the state of the game at each throw -/
structure GameState where
  throw_number : ℕ
  distance : ℕ

/-- Calculates the game state for a given throw number -/
def game_state (n : ℕ) : GameState :=
  { throw_number := n,
    distance := (n + 1) / 2 }

/-- Determines if Pat is throwing based on the throw number -/
def is_pat_throwing (n : ℕ) : Prop :=
  n % 2 = 1

theorem catch_game_end_state :
  let final_throw := 29
  let final_state := game_state final_throw
  final_state.distance = 15 ∧ is_pat_throwing final_throw := by
sorry

end NUMINAMATH_CALUDE_catch_game_end_state_l173_17383


namespace NUMINAMATH_CALUDE_volleyball_team_chemistry_l173_17357

theorem volleyball_team_chemistry (total_players : ℕ) (physics_players : ℕ) (both_subjects : ℕ) :
  total_players = 30 →
  physics_players = 15 →
  both_subjects = 10 →
  total_players = (physics_players - both_subjects) + (total_players - (physics_players - both_subjects)) →
  (total_players - (physics_players - both_subjects)) = 25 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_team_chemistry_l173_17357


namespace NUMINAMATH_CALUDE_repeated_root_condition_l173_17376

theorem repeated_root_condition (m : ℝ) : 
  (∃! x : ℝ, (5 * x) / (x - 2) + 1 = m / (x - 2) ∧ x ≠ 2) ↔ m = 10 :=
by sorry

end NUMINAMATH_CALUDE_repeated_root_condition_l173_17376


namespace NUMINAMATH_CALUDE_base_k_subtraction_l173_17390

/-- Represents a digit in base k -/
def Digit (k : ℕ) := {d : ℕ // d < k}

/-- Converts a two-digit number in base k to its decimal representation -/
def toDecimal (k : ℕ) (x y : Digit k) : ℕ := k * x.val + y.val

theorem base_k_subtraction (k : ℕ) (X Y : Digit k) 
  (h_k : k > 8)
  (h_eq : toDecimal k X Y + toDecimal k X X = 2 * k + 1) :
  X.val - Y.val = k - 4 := by sorry

end NUMINAMATH_CALUDE_base_k_subtraction_l173_17390


namespace NUMINAMATH_CALUDE_cube_volume_in_box_l173_17348

theorem cube_volume_in_box (box_length box_width box_height : ℝ)
  (num_cubes : ℕ) (cube_volume : ℝ) :
  box_length = 8 →
  box_width = 9 →
  box_height = 12 →
  num_cubes = 24 →
  cube_volume * num_cubes = box_length * box_width * box_height →
  cube_volume = 36 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_in_box_l173_17348


namespace NUMINAMATH_CALUDE_function_properties_l173_17369

-- Define the function f(x) = -2x + 1
def f (x : ℝ) : ℝ := -2 * x + 1

-- State the theorem
theorem function_properties :
  (f 1 = -1) ∧ 
  (∃ (x y z : ℝ), x > 0 ∧ y < 0 ∧ z > 0 ∧ f x > 0 ∧ f y < 0 ∧ f z < 0) ∧
  (∀ (x y : ℝ), x < y → f x > f y) ∧
  (∃ (x : ℝ), x > 0 ∧ f x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l173_17369


namespace NUMINAMATH_CALUDE_race_results_l173_17366

-- Define the race parameters
def race_distance : ℝ := 200
def time_A : ℝ := 40
def time_B : ℝ := 50
def time_C : ℝ := 45

-- Define the time differences
def time_diff_AB : ℝ := time_B - time_A
def time_diff_AC : ℝ := time_C - time_A
def time_diff_BC : ℝ := time_C - time_B

-- Theorem statement
theorem race_results :
  (time_diff_AB = 10) ∧
  (time_diff_AC = 5) ∧
  (time_diff_BC = -5) := by
  sorry

end NUMINAMATH_CALUDE_race_results_l173_17366


namespace NUMINAMATH_CALUDE_parallelogram_area_14_24_l173_17321

/-- The area of a parallelogram with given base and height -/
def parallelogramArea (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 14 cm and height 24 cm is 336 cm² -/
theorem parallelogram_area_14_24 : parallelogramArea 14 24 = 336 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_14_24_l173_17321


namespace NUMINAMATH_CALUDE_volume_of_rotated_figure_l173_17345

-- Define the figure
def figure (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define the volume of the solid formed by rotation
def volume_of_rotation (f : ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem volume_of_rotated_figure :
  volume_of_rotation figure = 4 * Real.pi^2 := by sorry

end NUMINAMATH_CALUDE_volume_of_rotated_figure_l173_17345


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l173_17351

theorem sqrt_difference_equality : Real.sqrt (49 + 121) - Real.sqrt (64 + 16) = Real.sqrt 170 - 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l173_17351


namespace NUMINAMATH_CALUDE_partner_investment_time_l173_17392

/-- Given two partners p and q with investment ratio 7:5 and profit ratio 7:10,
    where p invested for 20 months, prove that q invested for 40 months. -/
theorem partner_investment_time (x : ℝ) (t : ℝ) : 
  (7 : ℝ) / 5 = 7 * x / (5 * x) →  -- investment ratio
  (7 : ℝ) / 10 = (7 * x * 20) / (5 * x * t) →  -- profit ratio
  t = 40 := by
sorry

end NUMINAMATH_CALUDE_partner_investment_time_l173_17392


namespace NUMINAMATH_CALUDE_three_card_sequence_count_l173_17302

/-- The number of cards in the deck -/
def deck_size : ℕ := 60

/-- The number of suits in the deck -/
def num_suits : ℕ := 5

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := 12

/-- The number of cards to pick -/
def cards_to_pick : ℕ := 3

/-- The number of ways to pick three different cards in sequence from the deck -/
def ways_to_pick : ℕ := deck_size * (deck_size - 1) * (deck_size - 2)

theorem three_card_sequence_count :
  deck_size = num_suits * cards_per_suit →
  ways_to_pick = 205320 := by
  sorry

end NUMINAMATH_CALUDE_three_card_sequence_count_l173_17302


namespace NUMINAMATH_CALUDE_stating_pyramid_base_is_isosceles_l173_17387

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  /-- The length of each lateral edge -/
  edge_length : ℝ
  /-- The area of each lateral face -/
  face_area : ℝ
  /-- Assumption that all lateral edges have the same length -/
  equal_edges : edge_length > 0
  /-- Assumption that all lateral faces have the same area -/
  equal_faces : face_area > 0

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  /-- The length of the two equal sides -/
  equal_side : ℝ
  /-- The length of the base -/
  base : ℝ
  /-- Assumption that the equal sides are positive -/
  positive_equal_side : equal_side > 0
  /-- Assumption that the base is positive -/
  positive_base : base > 0

/-- 
Theorem stating that the base of a triangular pyramid with equal lateral edges 
and equal lateral face areas is an isosceles triangle 
-/
theorem pyramid_base_is_isosceles (p : TriangularPyramid) : 
  ∃ (t : IsoscelesTriangle), True :=
sorry

end NUMINAMATH_CALUDE_stating_pyramid_base_is_isosceles_l173_17387


namespace NUMINAMATH_CALUDE_count_divisible_by_seven_l173_17344

theorem count_divisible_by_seven : 
  (Finset.filter (fun n => n % 7 = 0) (Finset.Icc 200 400)).card = 29 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_by_seven_l173_17344


namespace NUMINAMATH_CALUDE_sum_a_d_equals_six_l173_17330

theorem sum_a_d_equals_six (a b c d : ℝ) 
  (hab : a + b = 12) 
  (hbc : b + c = 9) 
  (hcd : c + d = 3) : 
  a + d = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_equals_six_l173_17330


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l173_17374

theorem min_value_of_function (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 :=
sorry

theorem equality_condition (x : ℝ) (h : x > 1) : 
  x + 1 / (x - 1) = 3 ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l173_17374


namespace NUMINAMATH_CALUDE_car_speed_problem_l173_17363

theorem car_speed_problem (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) (distance : ℝ) :
  speed1 = 45 →
  time = 14/3 →
  distance = 490 →
  (speed1 + speed2) * time = distance →
  speed2 = 60 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l173_17363


namespace NUMINAMATH_CALUDE_unique_tangent_line_l173_17379

/-- The function f(x) = x^4 + 4x^3 - 26x^2 -/
def f (x : ℝ) : ℝ := x^4 + 4*x^3 - 26*x^2

/-- The line L(x) = 60x - 225 -/
def L (x : ℝ) : ℝ := 60*x - 225

theorem unique_tangent_line :
  ∃! (a b : ℝ), 
    (∀ x : ℝ, f x ≥ a*x + b ∨ f x ≤ a*x + b) ∧ 
    (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = a*x₁ + b ∧ f x₂ = a*x₂ + b) ∧
    a = 60 ∧ b = -225 :=
sorry

end NUMINAMATH_CALUDE_unique_tangent_line_l173_17379


namespace NUMINAMATH_CALUDE_one_tangent_line_l173_17373

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y - 26 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0

-- Define a function to count the number of tangent lines
def count_tangent_lines (C1 C2 : ℝ → ℝ → Prop) : ℕ := sorry

-- Theorem stating that there is exactly one tangent line
theorem one_tangent_line : count_tangent_lines C1 C2 = 1 := by sorry

end NUMINAMATH_CALUDE_one_tangent_line_l173_17373


namespace NUMINAMATH_CALUDE_largest_integer_times_eleven_less_than_150_l173_17377

theorem largest_integer_times_eleven_less_than_150 :
  ∀ x : ℤ, x ≤ 13 ↔ 11 * x < 150 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_times_eleven_less_than_150_l173_17377


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l173_17331

theorem negative_sixty_four_to_four_thirds (x : ℝ) : x = (-64)^(4/3) → x = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l173_17331


namespace NUMINAMATH_CALUDE_xyz_sum_sqrt_l173_17333

theorem xyz_sum_sqrt (x y z : ℝ) 
  (h1 : y + z = 15) 
  (h2 : z + x = 18) 
  (h3 : x + y = 17) : 
  Real.sqrt (x * y * z * (x + y + z)) = 10 * Real.sqrt 70 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_sqrt_l173_17333


namespace NUMINAMATH_CALUDE_salmon_population_increase_l173_17329

theorem salmon_population_increase (initial_salmon : ℕ) (increase_factor : ℕ) : 
  initial_salmon = 500 → increase_factor = 10 → initial_salmon * increase_factor = 5000 := by
  sorry

end NUMINAMATH_CALUDE_salmon_population_increase_l173_17329


namespace NUMINAMATH_CALUDE_cube_volume_in_pyramid_l173_17388

/-- Represents a pyramid with a regular hexagonal base and equilateral triangle lateral faces -/
structure Pyramid :=
  (base_side_length : ℝ)

/-- Represents a cube placed inside the pyramid -/
structure Cube :=
  (side_length : ℝ)

/-- Calculate the volume of a cube -/
def cube_volume (c : Cube) : ℝ := c.side_length ^ 3

/-- The configuration of the pyramid and cube as described in the problem -/
def pyramid_cube_configuration (p : Pyramid) (c : Cube) : Prop :=
  p.base_side_length = 2 ∧
  c.side_length = 2 * Real.sqrt 3 / 9

/-- The theorem stating the volume of the cube in the given configuration -/
theorem cube_volume_in_pyramid (p : Pyramid) (c : Cube) :
  pyramid_cube_configuration p c →
  cube_volume c = 8 * Real.sqrt 3 / 243 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_in_pyramid_l173_17388


namespace NUMINAMATH_CALUDE_sequence_sum_equals_33_l173_17398

def arithmetic_sequence (n : ℕ) : ℕ := 3 * n - 2

def geometric_sequence (n : ℕ) : ℕ := 3^(n - 1)

theorem sequence_sum_equals_33 :
  arithmetic_sequence (geometric_sequence 1) +
  arithmetic_sequence (geometric_sequence 2) +
  arithmetic_sequence (geometric_sequence 3) = 33 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_33_l173_17398


namespace NUMINAMATH_CALUDE_min_value_of_f_l173_17319

/-- The quadratic function f(x) = x^2 + 10x + 21 -/
def f (x : ℝ) : ℝ := x^2 + 10*x + 21

/-- Theorem: The minimum value of f(x) = x^2 + 10x + 21 is -4 -/
theorem min_value_of_f : 
  ∀ x : ℝ, f x ≥ -4 ∧ ∃ x₀ : ℝ, f x₀ = -4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l173_17319


namespace NUMINAMATH_CALUDE_cable_theorem_l173_17355

def cable_problem (basic_cost movie_cost sports_cost_diff : ℕ) : Prop :=
  let sports_cost := movie_cost - sports_cost_diff
  let total_cost := basic_cost + movie_cost + sports_cost
  total_cost = 36

theorem cable_theorem : cable_problem 15 12 3 :=
sorry

end NUMINAMATH_CALUDE_cable_theorem_l173_17355


namespace NUMINAMATH_CALUDE_heine_biscuits_l173_17306

/-- The number of biscuits Mrs. Heine needs to buy for her dogs -/
def total_biscuits (num_dogs : ℕ) (biscuits_per_dog : ℕ) : ℕ :=
  num_dogs * biscuits_per_dog

/-- Theorem stating that Mrs. Heine needs to buy 6 biscuits -/
theorem heine_biscuits : total_biscuits 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_heine_biscuits_l173_17306


namespace NUMINAMATH_CALUDE_divided_stick_properties_l173_17362

/-- Represents a stick divided into segments by different colored lines -/
structure DividedStick where
  length : ℝ
  red_segments : ℕ
  blue_segments : ℕ
  black_segments : ℕ

/-- Calculates the total number of segments after cutting -/
def total_segments (stick : DividedStick) : ℕ := sorry

/-- Calculates the length of the shortest segment -/
def shortest_segment (stick : DividedStick) : ℝ := sorry

/-- Theorem stating the properties of a stick divided into 8, 12, and 18 segments -/
theorem divided_stick_properties (L : ℝ) (h : L > 0) :
  let stick := DividedStick.mk L 8 12 18
  total_segments stick = 28 ∧ shortest_segment stick = L / 72 := by sorry

end NUMINAMATH_CALUDE_divided_stick_properties_l173_17362


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l173_17338

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y + 6 = x * y) :
  2 * x + y ≥ 12 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ + 6 = x₀ * y₀ ∧ 2 * x₀ + y₀ = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l173_17338


namespace NUMINAMATH_CALUDE_horner_method_proof_l173_17339

/-- Horner's method for evaluating a polynomial -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^4 + 3x^3 + 5x + 4 -/
def f : ℝ → ℝ := fun x => 2 * x^4 + 3 * x^3 + 5 * x + 4

theorem horner_method_proof :
  f 2 = horner_eval [2, 3, 0, 5, 4] 2 ∧ horner_eval [2, 3, 0, 5, 4] 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_proof_l173_17339


namespace NUMINAMATH_CALUDE_kerosene_cost_in_cents_l173_17313

-- Define the cost of a pound of rice in dollars
def rice_cost : ℚ := 33/100

-- Define the relationship between eggs and rice
def dozen_eggs_cost (rc : ℚ) : ℚ := rc

-- Define the relationship between kerosene and eggs
def half_liter_kerosene_cost (ec : ℚ) : ℚ := (8/12) * ec

-- Define the conversion from dollars to cents
def dollars_to_cents (d : ℚ) : ℚ := 100 * d

-- State the theorem
theorem kerosene_cost_in_cents : 
  dollars_to_cents (2 * half_liter_kerosene_cost (dozen_eggs_cost rice_cost)) = 44 := by
  sorry

end NUMINAMATH_CALUDE_kerosene_cost_in_cents_l173_17313


namespace NUMINAMATH_CALUDE_complex_expression_equals_25_1_l173_17310

theorem complex_expression_equals_25_1 :
  (50 + 5 * (12 / (180 / 3))^2) * Real.sin (30 * π / 180) = 25.1 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_25_1_l173_17310


namespace NUMINAMATH_CALUDE_triangle_perimeter_not_48_l173_17303

theorem triangle_perimeter_not_48 (a b c : ℝ) : 
  a = 25 → b = 12 → a + b + c > a + b → a + c > b → b + c > a → a + b + c ≠ 48 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_not_48_l173_17303


namespace NUMINAMATH_CALUDE_residue_of_negative_1035_mod_37_l173_17324

theorem residue_of_negative_1035_mod_37 :
  ∃ (r : ℤ), r ≥ 0 ∧ r < 37 ∧ -1035 ≡ r [ZMOD 37] ∧ r = 1 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_negative_1035_mod_37_l173_17324


namespace NUMINAMATH_CALUDE_max_sum_distances_l173_17352

/-- Given a real number k, two lines l₁ and l₂, and points P, Q, and M,
    prove that the maximum value of |MP| + |MQ| is 4. -/
theorem max_sum_distances (k : ℝ) :
  let P : ℝ × ℝ := (0, 0)
  let Q : ℝ × ℝ := (2, 2)
  let l₁ := {(x, y) : ℝ × ℝ | k * x + y = 0}
  let l₂ := {(x, y) : ℝ × ℝ | k * x - y - 2 * k + 2 = 0}
  let circle := {M : ℝ × ℝ | (M.1 - 1)^2 + (M.2 - 1)^2 = 2}
  ∀ M ∈ circle, (‖M - P‖ + ‖M - Q‖) ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_distances_l173_17352


namespace NUMINAMATH_CALUDE_can_identify_80_weights_l173_17393

/-- Represents a comparison between two sets of weights -/
def Comparison := List ℕ → List ℕ → Bool

/-- Given a list of weights and a number of comparisons, 
    determines if it's possible to uniquely identify all weights -/
def can_identify (weights : List ℕ) (num_comparisons : ℕ) : Prop :=
  ∃ (comparisons : List Comparison), 
    comparisons.length = num_comparisons ∧ 
    ∀ (w1 w2 : List ℕ), w1 ≠ w2 → w1.length = weights.length → w2.length = weights.length →
      ∃ (c : Comparison), c ∈ comparisons ∧ c w1 ≠ c w2

theorem can_identify_80_weights :
  ∀ (weights : List ℕ), 
    weights.length = 80 → 
    weights.Nodup → 
    (can_identify weights 4 ∧ ¬can_identify weights 3) := by
  sorry

#check can_identify_80_weights

end NUMINAMATH_CALUDE_can_identify_80_weights_l173_17393


namespace NUMINAMATH_CALUDE_cosine_equation_solutions_l173_17322

theorem cosine_equation_solutions (x : Real) :
  (∃ (s : Finset Real), s.card = 14 ∧ 
    (∀ y ∈ s, -π ≤ y ∧ y ≤ π ∧ 
      Real.cos (6 * y) + (Real.cos (3 * y))^4 + (Real.sin (2 * y))^2 + (Real.cos y)^2 = 0) ∧
    (∀ z, -π ≤ z ∧ z ≤ π ∧ 
      Real.cos (6 * z) + (Real.cos (3 * z))^4 + (Real.sin (2 * z))^2 + (Real.cos z)^2 = 0 → 
      z ∈ s)) :=
by sorry

end NUMINAMATH_CALUDE_cosine_equation_solutions_l173_17322


namespace NUMINAMATH_CALUDE_fraction_equality_l173_17359

theorem fraction_equality (m n : ℝ) (h : n ≠ 0) (h1 : m / n = 2 / 3) :
  m / (m + n) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l173_17359


namespace NUMINAMATH_CALUDE_integer_solutions_system_l173_17360

theorem integer_solutions_system : 
  {(x, y, z) : ℤ × ℤ × ℤ | x + y - z = 6 ∧ x^3 + y^3 - z^3 = 414} = 
  {(3, 8, 5), (8, 3, 5), (3, -5, -8), (-5, 8, -3), (-5, 3, -8), (8, -5, -3)} :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_system_l173_17360


namespace NUMINAMATH_CALUDE_chips_purchased_l173_17312

/-- Given that P packets of chips can be purchased for R dimes,
    and 1 dollar is worth 10 dimes, the number of packets that
    can be purchased for M dollars is 10MP/R. -/
theorem chips_purchased (P R M : ℚ) (h1 : P > 0) (h2 : R > 0) (h3 : M > 0) :
  (P / R) * (M * 10) = 10 * M * P / R :=
by sorry

end NUMINAMATH_CALUDE_chips_purchased_l173_17312


namespace NUMINAMATH_CALUDE_tensor_solution_l173_17381

/-- Custom operation ⊗ -/
def tensor (a b : ℝ) : ℝ := a * b + a + b^2

theorem tensor_solution :
  ∀ m : ℝ, m > 0 → tensor 1 m = 3 → m = 1 := by sorry

end NUMINAMATH_CALUDE_tensor_solution_l173_17381


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l173_17311

/-- The eccentricity of a hyperbola with equation x²/4 - y²/5 = 1 is 3/2 -/
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = 3/2 ∧ ∀ (x y : ℝ), x^2/4 - y^2/5 = 1 → 
  ∃ (a b c : ℝ), a^2 = 4 ∧ b^2 = 5 ∧ c^2 = a^2 + b^2 ∧ e = c/a := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l173_17311


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l173_17380

def polynomial (x : ℝ) : ℝ :=
  3 * (x^8 - 2*x^5 + 4*x^3 - 6) + 5 * (x^4 + 3*x^2 - 2*x) - 4 * (2*x^6 - 5)

theorem sum_of_coefficients : 
  (polynomial 1) = -31 :=
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l173_17380


namespace NUMINAMATH_CALUDE_point_on_terminal_side_l173_17309

theorem point_on_terminal_side (x : ℝ) (α : ℝ) :
  (∃ P : ℝ × ℝ, P = (x, 2) ∧ P.2 / Real.sqrt (P.1^2 + P.2^2) = 2/3) →
  x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_terminal_side_l173_17309


namespace NUMINAMATH_CALUDE_x_zero_value_l173_17305

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem x_zero_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 1) → x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_zero_value_l173_17305


namespace NUMINAMATH_CALUDE_kara_water_consumption_l173_17304

/-- Amount of water Kara drinks with each medication dose -/
def water_per_dose (total_water : ℕ) (doses_per_day : ℕ) (total_days : ℕ) (missed_doses : ℕ) : ℚ :=
  total_water / (doses_per_day * total_days - missed_doses)

/-- Theorem stating that Kara drinks 4 ounces of water per medication dose -/
theorem kara_water_consumption :
  water_per_dose 160 3 14 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_kara_water_consumption_l173_17304


namespace NUMINAMATH_CALUDE_prob_even_product_is_four_fifths_l173_17384

/-- Represents a spinner with a given set of numbers -/
structure Spinner where
  numbers : Finset ℕ

/-- The probability of selecting an even number from a spinner -/
def prob_even (s : Spinner) : ℚ :=
  (s.numbers.filter Even).card / s.numbers.card

/-- The probability of selecting an odd number from a spinner -/
def prob_odd (s : Spinner) : ℚ :=
  1 - prob_even s

/-- Spinner A with numbers 1 to 5 -/
def spinner_A : Spinner :=
  ⟨Finset.range 5 ∪ {5}⟩

/-- Spinner B with numbers 1, 2, 4 -/
def spinner_B : Spinner :=
  ⟨{1, 2, 4}⟩

theorem prob_even_product_is_four_fifths :
  1 - (prob_odd spinner_A * prob_odd spinner_B) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_product_is_four_fifths_l173_17384


namespace NUMINAMATH_CALUDE_polynomial_evaluation_gcd_of_three_numbers_l173_17370

-- Problem 1: Polynomial evaluation
def f (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x - 6

theorem polynomial_evaluation : f 1 = 9 := by sorry

-- Problem 2: GCD of three numbers
theorem gcd_of_three_numbers : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_gcd_of_three_numbers_l173_17370


namespace NUMINAMATH_CALUDE_factorization_equality_l173_17335

theorem factorization_equality (x y : ℝ) : x^2 * y - 2 * x * y^2 + y^3 = y * (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l173_17335


namespace NUMINAMATH_CALUDE_second_group_size_l173_17385

/-- Given a gym class with two groups of students, prove that the second group has 37 students. -/
theorem second_group_size (total : ℕ) (group1 : ℕ) (group2 : ℕ) : 
  total = 71 → group1 = 34 → total = group1 + group2 → group2 = 37 := by
  sorry

end NUMINAMATH_CALUDE_second_group_size_l173_17385


namespace NUMINAMATH_CALUDE_cat_walking_rate_l173_17349

/-- Given a cat's walking scenario with total time, resistance time, and distance walked,
    calculate the cat's walking rate in feet per minute. -/
theorem cat_walking_rate 
  (total_time : ℝ) 
  (resistance_time : ℝ) 
  (distance_walked : ℝ) 
  (h1 : total_time = 28) 
  (h2 : resistance_time = 20) 
  (h3 : distance_walked = 64) : 
  (distance_walked / (total_time - resistance_time)) = 8 := by
  sorry

#check cat_walking_rate

end NUMINAMATH_CALUDE_cat_walking_rate_l173_17349


namespace NUMINAMATH_CALUDE_x_in_terms_of_abc_l173_17320

theorem x_in_terms_of_abc (x y z a b c : ℝ) 
  (h1 : x * y / (x + y + 1) = a)
  (h2 : x * z / (x + z + 1) = b)
  (h3 : y * z / (y + z + 1) = c)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hab : a * b + a * c - b * c ≠ 0) :
  x = 2 * a * b * c / (a * b + a * c - b * c) :=
sorry

end NUMINAMATH_CALUDE_x_in_terms_of_abc_l173_17320


namespace NUMINAMATH_CALUDE_flute_players_count_l173_17308

/-- The number of people in an orchestra with specified instrument counts -/
structure Orchestra :=
  (total : ℕ)
  (drums : ℕ)
  (trombone : ℕ)
  (trumpet : ℕ)
  (french_horn : ℕ)
  (violinist : ℕ)
  (cellist : ℕ)
  (contrabassist : ℕ)
  (clarinet : ℕ)
  (conductor : ℕ)

/-- Theorem stating that the number of flute players is 4 -/
theorem flute_players_count (o : Orchestra) 
  (h1 : o.total = 21)
  (h2 : o.drums = 1)
  (h3 : o.trombone = 4)
  (h4 : o.trumpet = 2)
  (h5 : o.french_horn = 1)
  (h6 : o.violinist = 3)
  (h7 : o.cellist = 1)
  (h8 : o.contrabassist = 1)
  (h9 : o.clarinet = 3)
  (h10 : o.conductor = 1) :
  o.total - (o.drums + o.trombone + o.trumpet + o.french_horn + 
             o.violinist + o.cellist + o.contrabassist + 
             o.clarinet + o.conductor) = 4 := by
  sorry


end NUMINAMATH_CALUDE_flute_players_count_l173_17308


namespace NUMINAMATH_CALUDE_provolone_needed_l173_17336

def cheese_blend (m r p : ℝ) : Prop :=
  m / r = 2 ∧ p / r = 2

theorem provolone_needed (m r : ℝ) (hm : m = 20) (hr : r = 10) :
  ∃ p : ℝ, cheese_blend m r p ∧ p = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_provolone_needed_l173_17336


namespace NUMINAMATH_CALUDE_parallel_sum_diff_l173_17372

/-- Two vectors in ℝ² are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Given vectors a and b in ℝ², if a + b is parallel to a - b, then the second component of b is 1 -/
theorem parallel_sum_diff (x : ℝ) : 
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (2, x)
  are_parallel (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_sum_diff_l173_17372


namespace NUMINAMATH_CALUDE_toms_rate_difference_l173_17343

/-- Proves that Tom's rate is 5 steps per minute faster than Matt's, given their relative progress --/
theorem toms_rate_difference (matt_rate : ℕ) (matt_steps : ℕ) (tom_steps : ℕ) :
  matt_rate = 20 →
  matt_steps = 220 →
  tom_steps = 275 →
  ∃ (tom_rate : ℕ), tom_rate = matt_rate + 5 ∧ tom_steps * matt_rate = matt_steps * tom_rate :=
by sorry

end NUMINAMATH_CALUDE_toms_rate_difference_l173_17343


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l173_17375

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → 4/x + 1/y ≥ 9/2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ 4/x + 1/y = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l173_17375


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l173_17300

theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let inner_cube_diagonal := sphere_diameter
  let inner_cube_edge := inner_cube_diagonal / Real.sqrt 3
  inner_cube_edge ^ 3 = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l173_17300


namespace NUMINAMATH_CALUDE_quadratic_inequality_all_reals_l173_17399

/-- The quadratic inequality ax² + bx + c < 0 has a solution set of all real numbers
    if and only if a < 0 and b² - 4ac < 0 -/
theorem quadratic_inequality_all_reals
  (a b c : ℝ) (h_a : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_all_reals_l173_17399


namespace NUMINAMATH_CALUDE_students_per_table_l173_17315

theorem students_per_table (total_tables : ℕ) (total_students : ℕ) 
  (h1 : total_tables = 34) (h2 : total_students = 204) : 
  total_students / total_tables = 6 := by
  sorry

end NUMINAMATH_CALUDE_students_per_table_l173_17315


namespace NUMINAMATH_CALUDE_half_marathon_total_yards_l173_17389

/-- Represents the length of a race in miles and yards -/
structure RaceLength where
  miles : ℕ
  yards : ℚ

def half_marathon : RaceLength := { miles := 13, yards := 192.5 }

def yards_per_mile : ℕ := 1760

def num_races : ℕ := 6

theorem half_marathon_total_yards (m : ℕ) (y : ℚ) 
  (h1 : 0 ≤ y) (h2 : y < yards_per_mile) :
  m * yards_per_mile + y = 
    num_races * (half_marathon.miles * yards_per_mile + half_marathon.yards) → 
  y = 1155 := by
  sorry

end NUMINAMATH_CALUDE_half_marathon_total_yards_l173_17389


namespace NUMINAMATH_CALUDE_expected_value_of_segments_expected_value_is_1037_l173_17337

/-- The number of points in the plane -/
def n : ℕ := 100

/-- The number of pairs connected by line segments -/
def connected_pairs : ℕ := 4026

/-- A function to calculate binomial coefficient -/
def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

/-- The theorem to prove -/
theorem expected_value_of_segments (no_three_collinear : True) 
  (all_points_unique : True) : ℝ :=
  let total_pairs := choose n 2
  let diff_50_pairs := choose 51 2
  let prob_segment := connected_pairs / total_pairs
  prob_segment * diff_50_pairs

/-- The main theorem stating the expected value is 1037 -/
theorem expected_value_is_1037 (no_three_collinear : True) 
  (all_points_unique : True) :
  expected_value_of_segments no_three_collinear all_points_unique = 1037 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_segments_expected_value_is_1037_l173_17337


namespace NUMINAMATH_CALUDE_multiple_of_nine_between_12_and_30_l173_17332

theorem multiple_of_nine_between_12_and_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 9 * k)
  (h2 : x^2 > 144)
  (h3 : x < 30) :
  x = 18 ∨ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_nine_between_12_and_30_l173_17332


namespace NUMINAMATH_CALUDE_triangle_circle_square_sum_l173_17361

/-- Given a system of equations representing triangles, circles, and squares,
    prove that the sum of one triangle, two circles, and one square equals 35. -/
theorem triangle_circle_square_sum : 
  ∀ (x y z : ℝ),
  (2 * x + 3 * y + z = 45) →
  (x + 5 * y + 2 * z = 58) →
  (3 * x + y + 3 * z = 62) →
  (x + 2 * y + z = 35) := by
  sorry

end NUMINAMATH_CALUDE_triangle_circle_square_sum_l173_17361


namespace NUMINAMATH_CALUDE_second_player_winning_strategy_l173_17382

/-- Represents a character in the message -/
inductive Character
| Letter (c : Char)
| ExclamationMark

/-- Represents the state of the game board -/
def Board := List Character

/-- Represents a valid move in the game -/
inductive Move
| EraseSingle (c : Character)
| EraseMultiple (c : Char) (n : Nat)

/-- Applies a move to the board -/
def applyMove (board : Board) (move : Move) : Board :=
  sorry

/-- Checks if the game is over (no more characters to erase) -/
def isGameOver (board : Board) : Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := Board → Move

/-- Checks if a strategy is winning for the current player -/
def isWinningStrategy (strategy : Strategy) (board : Board) : Bool :=
  sorry

theorem second_player_winning_strategy 
  (initialBoard : Board) : 
  ∃ (strategy : Strategy), isWinningStrategy strategy initialBoard :=
sorry

end NUMINAMATH_CALUDE_second_player_winning_strategy_l173_17382


namespace NUMINAMATH_CALUDE_six_hardcover_books_l173_17367

/-- Represents the purchase of a set of books with two price options --/
structure BookPurchase where
  totalVolumes : ℕ
  paperbackPrice : ℕ
  hardcoverPrice : ℕ
  totalCost : ℕ

/-- Calculates the number of hardcover books purchased --/
def hardcoverCount (purchase : BookPurchase) : ℕ :=
  sorry

/-- Theorem stating that for the given purchase scenario, 6 hardcover books were bought --/
theorem six_hardcover_books (purchase : BookPurchase) 
  (h1 : purchase.totalVolumes = 12)
  (h2 : purchase.paperbackPrice = 18)
  (h3 : purchase.hardcoverPrice = 28)
  (h4 : purchase.totalCost = 276) : 
  hardcoverCount purchase = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_hardcover_books_l173_17367


namespace NUMINAMATH_CALUDE_inverse_proportion_ordering_l173_17395

/-- Proves the ordering of y-coordinates for three points on an inverse proportion function -/
theorem inverse_proportion_ordering (k : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h_pos : k > 0)
  (h_A : y₁ = k / (-3))
  (h_B : y₂ = k / (-2))
  (h_C : y₃ = k / 2) :
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ordering_l173_17395


namespace NUMINAMATH_CALUDE_series_sum_l173_17327

theorem series_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hc_def : c = a - b) :
  (∑' n : ℕ, 1 / (n * c * ((n + 1) * c))) = 1 / (b * c) :=
sorry

end NUMINAMATH_CALUDE_series_sum_l173_17327


namespace NUMINAMATH_CALUDE_villager_count_l173_17323

theorem villager_count (milk bottles_of_milk : ℕ) (apples : ℕ) (bread : ℕ) 
  (milk_left : ℕ) (apples_left : ℕ) (bread_short : ℕ) :
  bottles_of_milk = 160 →
  apples = 197 →
  bread = 229 →
  milk_left = 4 →
  apples_left = 2 →
  bread_short = 5 →
  ∃ (villagers : ℕ),
    villagers > 0 ∧
    (bottles_of_milk - milk_left) % villagers = 0 ∧
    (apples - apples_left) % villagers = 0 ∧
    (bread + bread_short) % villagers = 0 ∧
    villagers = 39 := by
  sorry

end NUMINAMATH_CALUDE_villager_count_l173_17323


namespace NUMINAMATH_CALUDE_triangle_inequality_iff_squared_sum_l173_17328

theorem triangle_inequality_iff_squared_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (2 * (a^4 + b^4 + c^4) < (a^2 + b^2 + c^2)^2) ↔ 
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_iff_squared_sum_l173_17328


namespace NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l173_17307

theorem sin_15_cos_15_eq_quarter : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l173_17307


namespace NUMINAMATH_CALUDE_waitress_income_fraction_l173_17358

theorem waitress_income_fraction (salary : ℚ) (tips : ℚ) (income : ℚ) :
  tips = (11 / 4) * salary →
  income = salary + tips →
  tips / income = 11 / 15 :=
by sorry

end NUMINAMATH_CALUDE_waitress_income_fraction_l173_17358
