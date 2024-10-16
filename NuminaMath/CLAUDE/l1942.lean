import Mathlib

namespace NUMINAMATH_CALUDE_stairs_climbed_total_l1942_194275

theorem stairs_climbed_total (jonny_stairs : ℕ) (julia_stairs : ℕ) : 
  jonny_stairs = 1269 →
  julia_stairs = jonny_stairs / 3 - 7 →
  jonny_stairs + julia_stairs = 1685 :=
by sorry

end NUMINAMATH_CALUDE_stairs_climbed_total_l1942_194275


namespace NUMINAMATH_CALUDE_solution_set_of_equation_l1942_194286

theorem solution_set_of_equation (x : ℝ) : 
  (16 * Real.sin (π * x) * Real.cos (π * x) = 16 * x + 1 / x) ↔ (x = 1/4 ∨ x = -1/4) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_equation_l1942_194286


namespace NUMINAMATH_CALUDE_inequality_proof_l1942_194298

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_one : a + b + c + d = 1) : 
  b * c * d / (1 - a)^2 + a * c * d / (1 - b)^2 + 
  a * b * d / (1 - c)^2 + a * b * c / (1 - d)^2 ≤ 1/9 ∧ 
  (b * c * d / (1 - a)^2 + a * c * d / (1 - b)^2 + 
   a * b * d / (1 - c)^2 + a * b * c / (1 - d)^2 = 1/9 ↔ 
   a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1942_194298


namespace NUMINAMATH_CALUDE_school_enrollment_problem_l1942_194241

theorem school_enrollment_problem (total_last_year : ℕ) 
  (xx_increase_rate yy_increase_rate : ℚ)
  (xx_to_yy yy_to_xx : ℕ)
  (xx_dropout_rate yy_dropout_rate : ℚ)
  (net_growth_diff : ℕ) :
  total_last_year = 4000 ∧
  xx_increase_rate = 7/100 ∧
  yy_increase_rate = 3/100 ∧
  xx_to_yy = 10 ∧
  yy_to_xx = 5 ∧
  xx_dropout_rate = 3/100 ∧
  yy_dropout_rate = 1/100 ∧
  net_growth_diff = 40 →
  ∃ (xx_last_year yy_last_year : ℕ),
    xx_last_year + yy_last_year = total_last_year ∧
    (xx_last_year * xx_increase_rate - xx_last_year * xx_dropout_rate - xx_to_yy) -
    (yy_last_year * yy_increase_rate - yy_last_year * yy_dropout_rate + yy_to_xx) = net_growth_diff ∧
    yy_last_year = 1750 :=
by sorry

end NUMINAMATH_CALUDE_school_enrollment_problem_l1942_194241


namespace NUMINAMATH_CALUDE_exchange_impossibility_l1942_194234

theorem exchange_impossibility : ¬ ∃ (N : ℕ), 5 * N = 2001 := by sorry

end NUMINAMATH_CALUDE_exchange_impossibility_l1942_194234


namespace NUMINAMATH_CALUDE_even_function_property_l1942_194272

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_property (f : ℝ → ℝ) 
  (h_even : EvenFunction f) 
  (h_negative : ∀ x < 0, f x = 1 + 2*x) : 
  ∀ x > 0, f x = 1 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_even_function_property_l1942_194272


namespace NUMINAMATH_CALUDE_calculator_correction_l1942_194279

theorem calculator_correction : (0.024 * 3.08) / 0.4 = 0.1848 := by
  sorry

end NUMINAMATH_CALUDE_calculator_correction_l1942_194279


namespace NUMINAMATH_CALUDE_snake_count_l1942_194281

/-- The number of snakes counted at the zoo --/
def snakes : ℕ := sorry

/-- The number of arctic foxes counted at the zoo --/
def arctic_foxes : ℕ := 80

/-- The number of leopards counted at the zoo --/
def leopards : ℕ := 20

/-- The number of bee-eaters counted at the zoo --/
def bee_eaters : ℕ := 10 * leopards

/-- The number of cheetahs counted at the zoo --/
def cheetahs : ℕ := snakes / 2

/-- The number of alligators counted at the zoo --/
def alligators : ℕ := 2 * (arctic_foxes + leopards)

/-- The total number of animals counted at the zoo --/
def total_animals : ℕ := 670

/-- Theorem stating that the number of snakes counted is 113 --/
theorem snake_count : snakes = 113 := by sorry

end NUMINAMATH_CALUDE_snake_count_l1942_194281


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l1942_194253

/-- Given that u and v are roots of 2x^2 + 5x + 3 = 0, prove that x^2 - x + 6 = 0 has roots 2u + 3 and 2v + 3 -/
theorem quadratic_roots_transformation (u v : ℝ) :
  (2 * u^2 + 5 * u + 3 = 0) →
  (2 * v^2 + 5 * v + 3 = 0) →
  ∀ x : ℝ, (x^2 - x + 6 = 0) ↔ (x = 2*u + 3 ∨ x = 2*v + 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l1942_194253


namespace NUMINAMATH_CALUDE_equation_describes_line_l1942_194296

theorem equation_describes_line :
  ∀ (x y : ℝ), (x - y)^2 = 2*(x^2 + y^2) ↔ y = -x := by sorry

end NUMINAMATH_CALUDE_equation_describes_line_l1942_194296


namespace NUMINAMATH_CALUDE_problem_solution_l1942_194288

theorem problem_solution (a b : ℤ) (h : 6 * a * b = 9 * a - 10 * b + 303) : a + b = 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1942_194288


namespace NUMINAMATH_CALUDE_polygon_exists_l1942_194235

-- Define the number of matches and their length
def num_matches : ℕ := 12
def match_length : ℝ := 2

-- Define the target area
def target_area : ℝ := 16

-- Define a polygon as a list of points
def Polygon := List (ℝ × ℝ)

-- Function to calculate the perimeter of a polygon
def perimeter (p : Polygon) : ℝ := sorry

-- Function to calculate the area of a polygon
def area (p : Polygon) : ℝ := sorry

-- Theorem stating the existence of a polygon satisfying the conditions
theorem polygon_exists : 
  ∃ (p : Polygon), 
    perimeter p = num_matches * match_length ∧ 
    area p = target_area :=
sorry

end NUMINAMATH_CALUDE_polygon_exists_l1942_194235


namespace NUMINAMATH_CALUDE_ABD_collinear_l1942_194297

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

variable (m n : V)
variable (A B C D : V)

axiom m_n_not_collinear : ¬ ∃ (k : ℝ), m = k • n

axiom AB_def : B - A = m + 5 • n
axiom BC_def : C - B = -2 • m + 8 • n
axiom CD_def : D - C = 4 • m + 2 • n

theorem ABD_collinear : ∃ (k : ℝ), D - A = k • (B - A) := by sorry

end NUMINAMATH_CALUDE_ABD_collinear_l1942_194297


namespace NUMINAMATH_CALUDE_complex_fraction_equals_two_l1942_194274

theorem complex_fraction_equals_two (c d : ℂ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : c^2 - c*d + d^2 = 0) : 
  (c^6 + d^6) / (c - d)^6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_two_l1942_194274


namespace NUMINAMATH_CALUDE_min_sides_rotatable_polygon_l1942_194237

theorem min_sides_rotatable_polygon (n : ℕ) (angle : ℚ) : 
  n > 0 ∧ 
  angle = 50 ∧ 
  (360 / n : ℚ) ∣ angle →
  n ≥ 36 :=
sorry

end NUMINAMATH_CALUDE_min_sides_rotatable_polygon_l1942_194237


namespace NUMINAMATH_CALUDE_ear_muffs_proof_l1942_194217

/-- The number of ear muffs bought before December -/
def ear_muffs_before_december : ℕ := 7790 - 6444

/-- The total number of ear muffs bought -/
def total_ear_muffs : ℕ := 7790

/-- The number of ear muffs bought during December -/
def ear_muffs_during_december : ℕ := 6444

theorem ear_muffs_proof :
  ear_muffs_before_december = 1346 ∧
  total_ear_muffs = ear_muffs_before_december + ear_muffs_during_december :=
by sorry

end NUMINAMATH_CALUDE_ear_muffs_proof_l1942_194217


namespace NUMINAMATH_CALUDE_a_divisibility_a_specific_cases_l1942_194223

def a (n : ℕ) : ℕ := (10^(3^n) - 1) / 9

theorem a_divisibility (n : ℕ) (h : n > 0) :
  (3^n ∣ a n) ∧ ¬(3^(n+1) ∣ a n) :=
sorry

theorem a_specific_cases :
  (3 ∣ a 1) ∧ ¬(9 ∣ a 1) ∧
  (9 ∣ a 2) ∧ ¬(27 ∣ a 2) ∧
  (27 ∣ a 3) ∧ ¬(81 ∣ a 3) :=
sorry

end NUMINAMATH_CALUDE_a_divisibility_a_specific_cases_l1942_194223


namespace NUMINAMATH_CALUDE_wise_stock_price_l1942_194219

/-- Given the conditions of Mr. Wise's stock purchase, prove the price of the stock he bought 400 shares of. -/
theorem wise_stock_price (total_value : ℝ) (price_known : ℝ) (total_shares : ℕ) (shares_unknown : ℕ) :
  total_value = 1950 →
  price_known = 4.5 →
  total_shares = 450 →
  shares_unknown = 400 →
  ∃ (price_unknown : ℝ),
    price_unknown * shares_unknown + price_known * (total_shares - shares_unknown) = total_value ∧
    price_unknown = 4.3125 :=
by sorry

end NUMINAMATH_CALUDE_wise_stock_price_l1942_194219


namespace NUMINAMATH_CALUDE_uncovered_cells_bound_l1942_194280

/-- Represents a rectangular board with dominoes -/
structure Board where
  m : ℕ  -- width of the board
  n : ℕ  -- height of the board
  uncovered : ℕ  -- number of uncovered cells

/-- Theorem stating that the number of uncovered cells is less than both mn/4 and mn/5 -/
theorem uncovered_cells_bound (b : Board) : 
  b.uncovered < min (b.m * b.n / 4) (b.m * b.n / 5) := by
  sorry

#check uncovered_cells_bound

end NUMINAMATH_CALUDE_uncovered_cells_bound_l1942_194280


namespace NUMINAMATH_CALUDE_brown_shirts_count_l1942_194282

def initial_blue_shirts : ℕ := 26

def remaining_blue_shirts : ℕ := initial_blue_shirts / 2

theorem brown_shirts_count (initial_brown_shirts : ℕ) : 
  remaining_blue_shirts + (initial_brown_shirts - initial_brown_shirts / 3) = 37 →
  initial_brown_shirts = 36 := by
  sorry

end NUMINAMATH_CALUDE_brown_shirts_count_l1942_194282


namespace NUMINAMATH_CALUDE_polynomial_equality_sum_l1942_194212

theorem polynomial_equality_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_sum_l1942_194212


namespace NUMINAMATH_CALUDE_train_speed_l1942_194222

def train_length : Real := 250.00000000000003
def crossing_time : Real := 15

theorem train_speed : 
  (train_length / 1000) / (crossing_time / 3600) = 60 := by sorry

end NUMINAMATH_CALUDE_train_speed_l1942_194222


namespace NUMINAMATH_CALUDE_larry_jogging_time_l1942_194200

/-- Calculates the total jogging time in hours for two weeks given daily jogging time and days jogged each week -/
def total_jogging_time (daily_time : ℕ) (days_week1 : ℕ) (days_week2 : ℕ) : ℚ :=
  ((daily_time * days_week1 + daily_time * days_week2) : ℚ) / 60

/-- Theorem stating that Larry's total jogging time for two weeks is 4 hours -/
theorem larry_jogging_time :
  total_jogging_time 30 3 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_larry_jogging_time_l1942_194200


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_plus_reciprocal_l1942_194268

theorem imaginary_part_of_z_plus_reciprocal (z : ℂ) (h : z = 1 - I) :
  (z + z⁻¹).im = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_plus_reciprocal_l1942_194268


namespace NUMINAMATH_CALUDE_regular_nonagon_angle_l1942_194220

/-- A regular nonagon inscribed in a circle -/
structure RegularNonagon :=
  (vertices : Fin 9 → ℝ × ℝ)
  (is_regular : ∀ i j : Fin 9, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1))
  (is_inscribed : ∃ center : ℝ × ℝ, ∀ i : Fin 9, dist center (vertices i) = dist center (vertices 0))

/-- The angle measure between three consecutive vertices of a regular nonagon -/
def angle_measure (n : RegularNonagon) (i : Fin 9) : ℝ :=
  sorry

/-- Theorem: The angle measure between three consecutive vertices of a regular nonagon is 40 degrees -/
theorem regular_nonagon_angle (n : RegularNonagon) (i : Fin 9) :
  angle_measure n i = 40 := by sorry

end NUMINAMATH_CALUDE_regular_nonagon_angle_l1942_194220


namespace NUMINAMATH_CALUDE_race_finish_orders_l1942_194266

theorem race_finish_orders (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_orders_l1942_194266


namespace NUMINAMATH_CALUDE_remaining_clothing_problem_l1942_194273

/-- The number of remaining pieces of clothing to fold -/
def remaining_clothing (total_shirts : ℕ) (total_shorts : ℕ) (folded_shirts : ℕ) (folded_shorts : ℕ) : ℕ :=
  (total_shirts - folded_shirts) + (total_shorts - folded_shorts)

/-- Theorem stating that given 20 shirts and 8 pairs of shorts, if 12 shirts and 5 shorts are folded,
    the remaining number of pieces of clothing to fold is 11. -/
theorem remaining_clothing_problem :
  remaining_clothing 20 8 12 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remaining_clothing_problem_l1942_194273


namespace NUMINAMATH_CALUDE_clive_change_l1942_194242

/-- The amount of money Clive has to spend -/
def budget : ℚ := 10

/-- The number of olives Clive needs -/
def olives_needed : ℕ := 80

/-- The number of olives in each jar -/
def olives_per_jar : ℕ := 20

/-- The cost of one jar of olives -/
def cost_per_jar : ℚ := 3/2

/-- The change Clive will have after buying the required number of olive jars -/
def change : ℚ := budget - (↑(olives_needed / olives_per_jar) * cost_per_jar)

theorem clive_change :
  change = 4 := by sorry

end NUMINAMATH_CALUDE_clive_change_l1942_194242


namespace NUMINAMATH_CALUDE_cyclic_sum_root_l1942_194287

theorem cyclic_sum_root {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let x := (a * b * c) / (a * b + b * c + c * a + 2 * Real.sqrt (a * b * c * (a + b + c)))
  (Real.sqrt (a * b * x * (a + b + x)) + 
   Real.sqrt (b * c * x * (b + c + x)) + 
   Real.sqrt (c * a * x * (c + a + x))) = 
  Real.sqrt (a * b * c * (a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_sum_root_l1942_194287


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1942_194205

theorem sum_of_three_numbers (S : Finset ℕ) (h1 : S.card = 10) (h2 : S.sum id > 144) :
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a + b + c ≥ 54 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1942_194205


namespace NUMINAMATH_CALUDE_money_redistribution_l1942_194277

/-- Represents the amount of money each person has -/
structure Money where
  amy : ℝ
  jan : ℝ
  toy : ℝ
  kim : ℝ

/-- Represents the redistribution rules -/
def redistribute (m : Money) : Money :=
  let step1 := Money.mk m.amy m.jan m.toy m.kim -- Kim equalizes others
  let step2 := Money.mk m.amy m.jan m.toy m.kim -- Amy doubles Jan and Toy
  let step3 := Money.mk m.amy m.jan m.toy m.kim -- Jan doubles Amy and Toy
  let step4 := Money.mk m.amy m.jan m.toy m.kim -- Toy doubles others
  step4

theorem money_redistribution (initial final : Money) :
  initial.toy = 48 →
  final.toy = 48 →
  final = redistribute initial →
  initial.amy + initial.jan + initial.toy + initial.kim = 192 :=
by
  sorry

#check money_redistribution

end NUMINAMATH_CALUDE_money_redistribution_l1942_194277


namespace NUMINAMATH_CALUDE_johns_car_efficiency_l1942_194257

/-- Calculates the miles per gallon (MPG) of John's car based on his weekly driving habits. -/
def johns_car_mpg (work_miles_one_way : ℕ) (work_days : ℕ) (leisure_miles : ℕ) (gas_used : ℕ) : ℚ :=
  let total_miles := 2 * work_miles_one_way * work_days + leisure_miles
  total_miles / gas_used

/-- Proves that John's car gets 30 miles per gallon based on his weekly driving habits. -/
theorem johns_car_efficiency :
  johns_car_mpg 20 5 40 8 = 30 := by
  sorry

end NUMINAMATH_CALUDE_johns_car_efficiency_l1942_194257


namespace NUMINAMATH_CALUDE_star_equality_implies_x_equals_6_l1942_194252

/-- Binary operation ★ on ordered pairs of integers -/
def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b + d)

/-- Theorem stating that if (3,3) ★ (0,0) = (x,y) ★ (3,2), then x = 6 -/
theorem star_equality_implies_x_equals_6 (x y : ℤ) :
  star 3 3 0 0 = star x y 3 2 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_star_equality_implies_x_equals_6_l1942_194252


namespace NUMINAMATH_CALUDE_log_5_12_equals_fraction_l1942_194251

-- Define the common logarithm (base 10) function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the logarithm with base 5
noncomputable def log_5 (x : ℝ) := Real.log x / Real.log 5

theorem log_5_12_equals_fraction (a b : ℝ) (h1 : lg 2 = a) (h2 : lg 3 = b) :
  log_5 12 = (2 * a + b) / (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_log_5_12_equals_fraction_l1942_194251


namespace NUMINAMATH_CALUDE_keystone_arch_angle_l1942_194278

theorem keystone_arch_angle (n : ℕ) (angle : ℝ) : 
  n = 10 → -- There are 10 trapezoids
  angle = (180 : ℝ) - (360 / (2 * n)) → -- The larger interior angle
  angle = 99 := by
  sorry

end NUMINAMATH_CALUDE_keystone_arch_angle_l1942_194278


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_x_is_smallest_y_is_smallest_x_makes_square_y_makes_cube_l1942_194211

/-- The smallest positive integer x for which 420x is a square -/
def x : ℕ := 735

/-- The smallest positive integer y for which 420y is a cube -/
def y : ℕ := 22050

theorem sum_of_x_and_y : x + y = 22785 := by sorry

theorem x_is_smallest :
  ∀ n : ℕ, n > 0 → (∃ m : ℕ, 420 * n = m ^ 2) → n ≥ x := by sorry

theorem y_is_smallest :
  ∀ n : ℕ, n > 0 → (∃ m : ℕ, 420 * n = m ^ 3) → n ≥ y := by sorry

theorem x_makes_square : ∃ m : ℕ, 420 * x = m ^ 2 := by sorry

theorem y_makes_cube : ∃ m : ℕ, 420 * y = m ^ 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_x_is_smallest_y_is_smallest_x_makes_square_y_makes_cube_l1942_194211


namespace NUMINAMATH_CALUDE_b_and_c_earnings_l1942_194285

/-- Given the daily earnings of three individuals a, b, and c, prove that b and c together earn $300 per day. -/
theorem b_and_c_earnings
  (total : ℝ)
  (a_and_c : ℝ)
  (c_earnings : ℝ)
  (h1 : total = 600)
  (h2 : a_and_c = 400)
  (h3 : c_earnings = 100) :
  total - a_and_c + c_earnings = 300 :=
by sorry

end NUMINAMATH_CALUDE_b_and_c_earnings_l1942_194285


namespace NUMINAMATH_CALUDE_number_satisfying_equations_l1942_194209

theorem number_satisfying_equations (x : ℝ) : 
  16 * x = 3408 ∧ 1.6 * x = 340.8 → x = 213 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_equations_l1942_194209


namespace NUMINAMATH_CALUDE_trick_deck_total_spent_l1942_194254

/-- The total amount spent by Tom and his friend on trick decks -/
theorem trick_deck_total_spent : 
  let deck_price : ℕ := 8
  let tom_decks : ℕ := 3
  let friend_decks : ℕ := 5
  deck_price * (tom_decks + friend_decks) = 64 := by
sorry

end NUMINAMATH_CALUDE_trick_deck_total_spent_l1942_194254


namespace NUMINAMATH_CALUDE_eight_divisors_l1942_194265

theorem eight_divisors (n : ℕ) : (Finset.card (Nat.divisors n) = 8) ↔ 
  (∃ p : ℕ, Nat.Prime p ∧ n = p^7) ∨ 
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p * q^3) ∨ 
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ n = p * q * r) :=
by sorry

end NUMINAMATH_CALUDE_eight_divisors_l1942_194265


namespace NUMINAMATH_CALUDE_concert_songs_theorem_l1942_194233

/-- Represents the number of songs sung by each girl -/
structure SongCounts where
  mary : ℕ
  alina : ℕ
  tina : ℕ
  hanna : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (sc : SongCounts) : Prop :=
  sc.hanna = 4 ∧
  sc.mary = 7 ∧
  sc.alina > sc.hanna ∧
  sc.alina < sc.mary ∧
  sc.tina > sc.hanna ∧
  sc.tina < sc.mary

/-- The total number of songs sung by the trios -/
def total_songs (sc : SongCounts) : ℕ :=
  (sc.mary + sc.alina + sc.tina + sc.hanna) / 3

/-- The main theorem to be proved -/
theorem concert_songs_theorem (sc : SongCounts) :
  satisfies_conditions sc → total_songs sc = 7 := by
  sorry

end NUMINAMATH_CALUDE_concert_songs_theorem_l1942_194233


namespace NUMINAMATH_CALUDE_equal_opposite_angles_imag_prod_zero_l1942_194231

/-- Given complex numbers a, b, c, d where the angles a 0 b and c 0 d are equal and oppositely oriented,
    the imaginary part of their product abcd is zero. -/
theorem equal_opposite_angles_imag_prod_zero
  (a b c d : ℂ)
  (h : ∃ (θ : ℝ), (b / a).arg = θ ∧ (d / c).arg = -θ) :
  (a * b * c * d).im = 0 := by
  sorry

end NUMINAMATH_CALUDE_equal_opposite_angles_imag_prod_zero_l1942_194231


namespace NUMINAMATH_CALUDE_complex_number_location_l1942_194240

theorem complex_number_location (z : ℂ) (h : (1 + 2*I)/z = I) : 
  z = 2/5 + 1/5*I ∧ z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l1942_194240


namespace NUMINAMATH_CALUDE_abc_maximum_l1942_194247

theorem abc_maximum (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a * b + c = (a + c) * (b + c)) (h_sum : a + b + c = 2) :
  a * b * c ≤ 1 / 27 :=
sorry

end NUMINAMATH_CALUDE_abc_maximum_l1942_194247


namespace NUMINAMATH_CALUDE_max_pangs_proof_l1942_194295

/-- The maximum number of pangs that can be purchased given the constraints -/
def max_pangs : ℕ := 9

/-- The price of a pin in dollars -/
def pin_price : ℕ := 3

/-- The price of a pon in dollars -/
def pon_price : ℕ := 4

/-- The price of a pang in dollars -/
def pang_price : ℕ := 9

/-- The total budget in dollars -/
def total_budget : ℕ := 100

/-- The minimum number of pins that must be purchased -/
def min_pins : ℕ := 2

/-- The minimum number of pons that must be purchased -/
def min_pons : ℕ := 3

theorem max_pangs_proof :
  ∃ (pins pons : ℕ),
    pins ≥ min_pins ∧
    pons ≥ min_pons ∧
    pin_price * pins + pon_price * pons + pang_price * max_pangs = total_budget ∧
    ∀ (pangs : ℕ), pangs > max_pangs →
      ∀ (p q : ℕ),
        p ≥ min_pins →
        q ≥ min_pons →
        pin_price * p + pon_price * q + pang_price * pangs ≠ total_budget :=
by sorry

end NUMINAMATH_CALUDE_max_pangs_proof_l1942_194295


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1942_194215

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 →
  (x + 1 - 3 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1)) = (x + 2) / (x - 2) ∧
  (0 + 2) / (0 - 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1942_194215


namespace NUMINAMATH_CALUDE_mancino_garden_length_l1942_194221

theorem mancino_garden_length :
  ∀ (L : ℝ),
  (3 * L * 5 + 2 * 8 * 4 = 304) →
  L = 16 := by
sorry

end NUMINAMATH_CALUDE_mancino_garden_length_l1942_194221


namespace NUMINAMATH_CALUDE_intersection_S_T_l1942_194248

def S : Set ℝ := {x | |x| < 5}
def T : Set ℝ := {x | (x+7)*(x-3) < 0}

theorem intersection_S_T : S ∩ T = {x : ℝ | -5 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_S_T_l1942_194248


namespace NUMINAMATH_CALUDE_abc_maximum_l1942_194256

theorem abc_maximum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * b + c = (a + c) * (b + c)) (h2 : a + b + c = 2) :
  a * b * c ≤ 8 / 27 :=
sorry

end NUMINAMATH_CALUDE_abc_maximum_l1942_194256


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l1942_194250

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The problem statement -/
theorem parallel_lines_k_value :
  (∀ x y : ℝ, y = 5 * x + 3 ↔ y = (3 * k) * x + 7) → k = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l1942_194250


namespace NUMINAMATH_CALUDE_total_amount_paid_l1942_194225

-- Define the structure for an item
structure Item where
  originalPrice : ℝ
  saleDiscount : ℝ
  membershipDiscount : Bool
  taxRate : ℝ

-- Define the function to calculate the final price of an item
def calculateFinalPrice (item : Item) : ℝ :=
  let priceAfterSale := item.originalPrice * (1 - item.saleDiscount)
  let priceAfterMembership := if item.membershipDiscount then priceAfterSale * 0.95 else priceAfterSale
  priceAfterMembership * (1 + item.taxRate)

-- Define the items
def vase : Item := { originalPrice := 250, saleDiscount := 0.25, membershipDiscount := true, taxRate := 0.12 }
def teacups : Item := { originalPrice := 350, saleDiscount := 0.30, membershipDiscount := false, taxRate := 0.08 }
def plate : Item := { originalPrice := 450, saleDiscount := 0, membershipDiscount := true, taxRate := 0.10 }
def ornament : Item := { originalPrice := 150, saleDiscount := 0.20, membershipDiscount := false, taxRate := 0.06 }

-- Theorem statement
theorem total_amount_paid : 
  calculateFinalPrice vase + calculateFinalPrice teacups + calculateFinalPrice plate + calculateFinalPrice ornament = 1061.55 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l1942_194225


namespace NUMINAMATH_CALUDE_volume_circumscribed_sphere_unit_cube_l1942_194244

/-- The volume of a circumscribed sphere of a cube with edge length 1 -/
theorem volume_circumscribed_sphere_unit_cube :
  let edge_length : ℝ := 1
  let radius : ℝ := (Real.sqrt 3) / 2
  let volume : ℝ := (4/3) * Real.pi * radius^3
  volume = (Real.sqrt 3 / 2) * Real.pi := by
sorry

end NUMINAMATH_CALUDE_volume_circumscribed_sphere_unit_cube_l1942_194244


namespace NUMINAMATH_CALUDE_minimum_loads_is_nineteen_l1942_194264

/-- Represents the capacity of the washing machine -/
structure MachineCapacity where
  shirts : ℕ
  sweaters : ℕ
  socks : ℕ

/-- Represents the number of clothes to be washed -/
structure ClothesCount where
  white_shirts : ℕ
  colored_shirts : ℕ
  white_sweaters : ℕ
  colored_sweaters : ℕ
  white_socks : ℕ
  colored_socks : ℕ

/-- Calculates the number of loads required for a given type of clothing -/
def loadsForClothingType (clothes : ℕ) (capacity : ℕ) : ℕ :=
  (clothes + capacity - 1) / capacity

/-- Calculates the total number of loads required -/
def totalLoads (capacity : MachineCapacity) (clothes : ClothesCount) : ℕ :=
  let white_loads := max (loadsForClothingType clothes.white_shirts capacity.shirts)
                         (max (loadsForClothingType clothes.white_sweaters capacity.sweaters)
                              (loadsForClothingType clothes.white_socks capacity.socks))
  let colored_loads := max (loadsForClothingType clothes.colored_shirts capacity.shirts)
                           (max (loadsForClothingType clothes.colored_sweaters capacity.sweaters)
                                (loadsForClothingType clothes.colored_socks capacity.socks))
  white_loads + colored_loads

/-- Theorem: The minimum number of loads required is 19 -/
theorem minimum_loads_is_nineteen (capacity : MachineCapacity) (clothes : ClothesCount) :
  capacity.shirts = 3 ∧ capacity.sweaters = 2 ∧ capacity.socks = 4 ∧
  clothes.white_shirts = 9 ∧ clothes.colored_shirts = 12 ∧
  clothes.white_sweaters = 18 ∧ clothes.colored_sweaters = 20 ∧
  clothes.white_socks = 16 ∧ clothes.colored_socks = 24 →
  totalLoads capacity clothes = 19 := by
  sorry


end NUMINAMATH_CALUDE_minimum_loads_is_nineteen_l1942_194264


namespace NUMINAMATH_CALUDE_complement_of_A_l1942_194245

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}

theorem complement_of_A (x : ℝ) : x ∈ (Set.compl A) ↔ x ∈ Set.Icc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l1942_194245


namespace NUMINAMATH_CALUDE_faster_train_speed_is_72_l1942_194207

/-- The speed of the faster train given the conditions of the problem -/
def faster_train_speed (slower_train_speed : ℝ) (speed_difference : ℝ) 
  (crossing_time : ℝ) (train_length : ℝ) : ℝ :=
  slower_train_speed + speed_difference

/-- Theorem stating the speed of the faster train under the given conditions -/
theorem faster_train_speed_is_72 :
  faster_train_speed 36 36 20 200 = 72 := by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_is_72_l1942_194207


namespace NUMINAMATH_CALUDE_expenditure_difference_l1942_194229

theorem expenditure_difference 
  (original_price : ℝ) 
  (original_amount : ℝ) 
  (price_increase_percent : ℝ) 
  (purchased_amount_percent : ℝ) 
  (h1 : price_increase_percent = 25)
  (h2 : purchased_amount_percent = 70) :
  let new_price := original_price * (1 + price_increase_percent / 100)
  let new_expenditure := new_price * (purchased_amount_percent / 100) * original_amount
  let original_expenditure := original_price * original_amount
  let difference := new_expenditure - original_expenditure
  abs difference / original_expenditure = 0.125 := by
sorry

end NUMINAMATH_CALUDE_expenditure_difference_l1942_194229


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_l1942_194204

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  first_element : ℕ
  h_pop_size : population_size > 0
  h_sample_size : sample_size > 0
  h_sample_size_le_pop : sample_size ≤ population_size
  h_first_element : first_element > 0 ∧ first_element ≤ population_size

/-- The interval between elements in a systematic sample -/
def SystematicSample.interval (s : SystematicSample) : ℕ :=
  s.population_size / s.sample_size

/-- Checks if a number is in the systematic sample -/
def SystematicSample.contains (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.first_element + k * s.interval ∧ n ≤ s.population_size

/-- The theorem to be proved -/
theorem systematic_sample_fourth_element
  (s : SystematicSample)
  (h_pop_size : s.population_size = 36)
  (h_sample_size : s.sample_size = 4)
  (h_contains_5 : s.contains 5)
  (h_contains_23 : s.contains 23)
  (h_contains_32 : s.contains 32) :
  s.contains 14 :=
sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_element_l1942_194204


namespace NUMINAMATH_CALUDE_number_of_pupils_l1942_194292

/-- Represents the number of pupils in the class -/
def n : ℕ := sorry

/-- The correct first mark -/
def correct_first_mark : ℕ := 63

/-- The incorrect first mark -/
def incorrect_first_mark : ℕ := 83

/-- The correct second mark -/
def correct_second_mark : ℕ := 85

/-- The incorrect second mark -/
def incorrect_second_mark : ℕ := 75

/-- The weight for the first mark -/
def weight_first : ℕ := 3

/-- The weight for the second mark -/
def weight_second : ℕ := 2

/-- The increase in average marks due to the errors -/
def average_increase : ℚ := 1/2

theorem number_of_pupils : n = 80 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pupils_l1942_194292


namespace NUMINAMATH_CALUDE_library_shelves_l1942_194201

theorem library_shelves (books : ℕ) (additional_books : ℕ) (shelves : ℕ) : 
  books = 4305 →
  additional_books = 11 →
  (books + additional_books) % shelves = 0 →
  shelves = 11 :=
by sorry

end NUMINAMATH_CALUDE_library_shelves_l1942_194201


namespace NUMINAMATH_CALUDE_complement_M_in_U_l1942_194299

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define the set M
def M : Set ℝ := {x | x > 1}

-- State the theorem
theorem complement_M_in_U : 
  (U \ M) = {x | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_M_in_U_l1942_194299


namespace NUMINAMATH_CALUDE_percent_of_number_zero_point_one_percent_of_12356_l1942_194271

theorem percent_of_number (x : ℝ) : x * 0.001 = 0.001 * x := by sorry

theorem zero_point_one_percent_of_12356 : (12356 : ℝ) * 0.001 = 12.356 := by sorry

end NUMINAMATH_CALUDE_percent_of_number_zero_point_one_percent_of_12356_l1942_194271


namespace NUMINAMATH_CALUDE_games_mike_can_buy_l1942_194208

theorem games_mike_can_buy (initial_amount : ℕ) (spent_amount : ℕ) (game_cost : ℕ) : 
  initial_amount = 69 → spent_amount = 24 → game_cost = 5 →
  (initial_amount - spent_amount) / game_cost = 9 :=
by sorry

end NUMINAMATH_CALUDE_games_mike_can_buy_l1942_194208


namespace NUMINAMATH_CALUDE_toucans_joined_l1942_194263

theorem toucans_joined (initial final joined : ℕ) : 
  initial = 2 → final = 3 → joined = final - initial :=
by sorry

end NUMINAMATH_CALUDE_toucans_joined_l1942_194263


namespace NUMINAMATH_CALUDE_parabola_points_x_coordinate_l1942_194226

/-- The x-coordinate of points on the parabola y^2 = 12x with distance 8 from the focus -/
theorem parabola_points_x_coordinate (x y : ℝ) : 
  y^2 = 12*x →                             -- Point (x,y) is on the parabola
  (x - 3)^2 + y^2 = 64 →                   -- Distance from (x,y) to focus (3,0) is 8
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_parabola_points_x_coordinate_l1942_194226


namespace NUMINAMATH_CALUDE_larger_number_proof_l1942_194206

theorem larger_number_proof (L S : ℕ) (hL : L > S) : 
  L - S = 1365 → L = 6 * S + 15 → L = 1635 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1942_194206


namespace NUMINAMATH_CALUDE_two_discounts_price_l1942_194261

/-- The final price of a product after two consecutive 10% discounts -/
def final_price (a : ℝ) : ℝ := a * (1 - 0.1)^2

/-- Theorem stating that the final price after two 10% discounts is correct -/
theorem two_discounts_price (a : ℝ) :
  final_price a = a * (1 - 0.1)^2 := by
  sorry

end NUMINAMATH_CALUDE_two_discounts_price_l1942_194261


namespace NUMINAMATH_CALUDE_kaleb_toy_purchase_l1942_194202

/-- Represents the problem of calculating how many toys Kaleb can buy -/
theorem kaleb_toy_purchase (saved : ℝ) (new_allowance : ℝ) (allowance_increase : ℝ) 
  (toy_cost : ℝ) : 
  saved = 21 → 
  new_allowance = 15 → 
  allowance_increase = 0.2 →
  toy_cost = 6 →
  (((saved + new_allowance) / 2) / toy_cost : ℝ) = 3 := by
  sorry

#check kaleb_toy_purchase

end NUMINAMATH_CALUDE_kaleb_toy_purchase_l1942_194202


namespace NUMINAMATH_CALUDE_fraction_problem_l1942_194236

theorem fraction_problem (n : ℝ) (h : (1/3) * (1/4) * n = 15) : 
  ∃ f : ℝ, f * n = 54 ∧ f = 3/10 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l1942_194236


namespace NUMINAMATH_CALUDE_frisbee_price_problem_l1942_194213

/-- Represents the price and quantity of frisbees sold at that price -/
structure FrisbeeGroup where
  price : ℝ
  quantity : ℕ

/-- Calculates the total revenue from a group of frisbees -/
def revenue (group : FrisbeeGroup) : ℝ :=
  group.price * group.quantity

theorem frisbee_price_problem (total_frisbees : ℕ) (total_revenue : ℝ) 
    (cheap_frisbees : FrisbeeGroup) (expensive_frisbees : FrisbeeGroup) : 
    total_frisbees = 60 →
    cheap_frisbees.price = 4 →
    cheap_frisbees.quantity ≥ 20 →
    cheap_frisbees.quantity + expensive_frisbees.quantity = total_frisbees →
    revenue cheap_frisbees + revenue expensive_frisbees = total_revenue →
    total_revenue = 200 →
    expensive_frisbees.price = 3 := by
  sorry

end NUMINAMATH_CALUDE_frisbee_price_problem_l1942_194213


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l1942_194270

theorem cara_seating_arrangements (n : ℕ) (h : n = 6) : 
  Nat.choose n 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l1942_194270


namespace NUMINAMATH_CALUDE_complex_power_sum_l1942_194267

theorem complex_power_sum (z : ℂ) (h : z + 1 / z = 2 * Real.cos (5 * π / 180)) :
  z^12 + 1 / z^12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1942_194267


namespace NUMINAMATH_CALUDE_sales_estimate_at_10_l1942_194255

/-- Represents the linear regression equation for sales volume estimation -/
def sales_estimate (x : ℝ) : ℝ := -10 * x + 200

/-- States that when the selling price is 10, the estimated sales volume is approximately 100 -/
theorem sales_estimate_at_10 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |sales_estimate 10 - 100| < ε :=
sorry

end NUMINAMATH_CALUDE_sales_estimate_at_10_l1942_194255


namespace NUMINAMATH_CALUDE_second_derivative_parametric_function_l1942_194269

noncomputable def x (t : ℝ) : ℝ := Real.cosh t

noncomputable def y (t : ℝ) : ℝ := (Real.sinh t) ^ (2/3)

theorem second_derivative_parametric_function (t : ℝ) :
  let x_t' := Real.sinh t
  let y_t' := (2 * Real.cosh t) / (3 * (Real.sinh t)^(1/3))
  let y_x' := y_t' / x_t'
  let y_x'_t' := -2 * (3 + Real.cosh t ^ 2) / (9 * Real.sinh t ^ 3)
  (y_x'_t' / x_t') = -2 * (3 + Real.cosh t ^ 2) / (9 * Real.sinh t ^ 4) :=
by sorry

end NUMINAMATH_CALUDE_second_derivative_parametric_function_l1942_194269


namespace NUMINAMATH_CALUDE_adjacent_angles_l1942_194259

theorem adjacent_angles (α β : ℝ) : 
  α + β = 180 →  -- sum of adjacent angles is 180°
  α = β + 30 →   -- one angle is 30° larger than the other
  (α = 105 ∧ β = 75) ∨ (α = 75 ∧ β = 105) := by
sorry

end NUMINAMATH_CALUDE_adjacent_angles_l1942_194259


namespace NUMINAMATH_CALUDE_ratio_xyz_l1942_194276

theorem ratio_xyz (x y z : ℝ) 
  (h1 : 0.6 * x = 0.3 * y)
  (h2 : 0.8 * z = 0.4 * x)
  (h3 : z = 2 * y) :
  ∃ (k : ℝ), k > 0 ∧ x = 4 * k ∧ y = k ∧ z = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_ratio_xyz_l1942_194276


namespace NUMINAMATH_CALUDE_smallest_number_with_8_divisors_multiple_of_24_l1942_194203

def is_multiple_of_24 (n : ℕ) : Prop := ∃ k : ℕ, n = 24 * k

def count_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_number_with_8_divisors_multiple_of_24 :
  ∀ n : ℕ, is_multiple_of_24 n ∧ count_divisors n = 8 → n ≥ 720 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_8_divisors_multiple_of_24_l1942_194203


namespace NUMINAMATH_CALUDE_positive_A_value_l1942_194232

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l1942_194232


namespace NUMINAMATH_CALUDE_meeting_attendees_l1942_194293

theorem meeting_attendees (total_handshakes : ℕ) (h : total_handshakes = 91) :
  ∃ n : ℕ, n > 0 ∧ n * (n - 1) / 2 = total_handshakes ∧ n = 14 := by
  sorry

end NUMINAMATH_CALUDE_meeting_attendees_l1942_194293


namespace NUMINAMATH_CALUDE_unique_value_2n_plus_m_l1942_194291

theorem unique_value_2n_plus_m :
  ∀ n m : ℤ,
  (3 * n - m < 5) →
  (n + m > 26) →
  (3 * m - 2 * n < 46) →
  (2 * n + m = 36) :=
by sorry

end NUMINAMATH_CALUDE_unique_value_2n_plus_m_l1942_194291


namespace NUMINAMATH_CALUDE_sqrt_k_squared_minus_pk_integer_l1942_194218

theorem sqrt_k_squared_minus_pk_integer (p : ℕ) (hp : Prime p) :
  ∀ k : ℤ, (∃ n : ℕ+, (k^2 - p * k : ℤ) = n^2) ↔ 
    (p ≠ 2 ∧ (k = ((p + 1) / 2)^2 ∨ k = -((p - 1) / 2)^2)) ∨ 
    (p = 2 ∧ False) := by
  sorry

#check sqrt_k_squared_minus_pk_integer

end NUMINAMATH_CALUDE_sqrt_k_squared_minus_pk_integer_l1942_194218


namespace NUMINAMATH_CALUDE_sum_of_47_and_negative_27_l1942_194262

theorem sum_of_47_and_negative_27 : 47 + (-27) = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_47_and_negative_27_l1942_194262


namespace NUMINAMATH_CALUDE_balcony_seat_cost_l1942_194246

/-- Theorem: Cost of a balcony seat in a theater --/
theorem balcony_seat_cost
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (orchestra_price : ℕ)
  (balcony_orchestra_diff : ℕ)
  (h1 : total_tickets = 355)
  (h2 : total_revenue = 3320)
  (h3 : orchestra_price = 12)
  (h4 : balcony_orchestra_diff = 115) :
  ∃ (balcony_price : ℕ),
    balcony_price = 8 ∧
    balcony_price * (total_tickets / 2 + balcony_orchestra_diff / 2) +
    orchestra_price * (total_tickets / 2 - balcony_orchestra_diff / 2) =
    total_revenue :=
by sorry

end NUMINAMATH_CALUDE_balcony_seat_cost_l1942_194246


namespace NUMINAMATH_CALUDE_consecutive_points_length_l1942_194238

/-- Given 5 consecutive points on a straight line, prove that ab = 5 -/
theorem consecutive_points_length (a b c d e : ℝ) : 
  (c - b = 3 * (d - c)) →  -- bc = 3 * cd
  (e - d = 7) →            -- de = 7
  (c - a = 11) →           -- ac = 11
  (e - a = 20) →           -- ae = 20
  (b - a = 5) :=           -- ab = 5
by sorry

end NUMINAMATH_CALUDE_consecutive_points_length_l1942_194238


namespace NUMINAMATH_CALUDE_height_increase_per_decade_l1942_194228

/-- Proves that the height increase per decade is 90 meters, given that the total increase in height over 2 centuries is 1800 meters. -/
theorem height_increase_per_decade : 
  ∀ (increase_per_decade : ℝ),
  (20 * increase_per_decade = 1800) →
  increase_per_decade = 90 := by
sorry

end NUMINAMATH_CALUDE_height_increase_per_decade_l1942_194228


namespace NUMINAMATH_CALUDE_speed_increase_percentage_l1942_194239

/-- Calculates the percentage increase in average speed between two cars -/
theorem speed_increase_percentage
  (distance : ℝ)
  (time_Q time_Y : ℝ)
  (h_distance : distance = 80)
  (h_time_Q : time_Q = 2)
  (h_time_Y : time_Y = 1.3333333333333333)
  (h_time_positive : time_Q > 0 ∧ time_Y > 0)
  (h_Y_faster : time_Y < time_Q) :
  ((distance / time_Y - distance / time_Q) / (distance / time_Q)) * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_speed_increase_percentage_l1942_194239


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1942_194224

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 - 2*x + 3

-- Theorem statement
theorem quadratic_function_properties :
  (∃ (a : ℝ), f x = a * (x + 1)^2 + 4) ∧ -- Vertex form with vertex at (-1, 4)
  f 2 = -5 := by -- Passes through (2, -5)
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1942_194224


namespace NUMINAMATH_CALUDE_pi_approximation_accuracy_l1942_194260

-- Define the approximation of π
def pi_approx : ℚ := 3.14

-- Define the true value of π (we'll use a rational approximation for simplicity)
def pi_true : ℚ := 355 / 113

-- Define the accuracy of the approximation
def accuracy : ℚ := 0.01

-- Theorem statement
theorem pi_approximation_accuracy :
  |pi_approx - pi_true| < accuracy :=
sorry

end NUMINAMATH_CALUDE_pi_approximation_accuracy_l1942_194260


namespace NUMINAMATH_CALUDE_min_quotient_value_l1942_194216

theorem min_quotient_value (a b : ℝ) 
  (ha : 100 ≤ a ∧ a ≤ 300)
  (hb : 400 ≤ b ∧ b ≤ 800)
  (hab : a + b ≤ 950) :
  (∀ a' b', 100 ≤ a' ∧ a' ≤ 300 → 400 ≤ b' ∧ b' ≤ 800 → a' + b' ≤ 950 → a / b ≤ a' / b') →
  a / b = 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_min_quotient_value_l1942_194216


namespace NUMINAMATH_CALUDE_prob_two_diff_numbers_correct_l1942_194227

/-- The number of faces on a standard die -/
def num_faces : ℕ := 6

/-- The number of dice being rolled -/
def num_dice : ℕ := 3

/-- The probability of getting exactly two different numbers when rolling three standard six-sided dice -/
def prob_two_diff_numbers : ℚ := sorry

theorem prob_two_diff_numbers_correct :
  prob_two_diff_numbers = 
    (num_faces.choose 2 * num_dice * (num_faces - 2)) / (num_faces ^ num_dice) :=
by sorry

end NUMINAMATH_CALUDE_prob_two_diff_numbers_correct_l1942_194227


namespace NUMINAMATH_CALUDE_max_equal_quotient_remainder_l1942_194214

theorem max_equal_quotient_remainder (A B C : ℕ) (h1 : A = 7 * B + C) (h2 : B = C) :
  B ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_equal_quotient_remainder_l1942_194214


namespace NUMINAMATH_CALUDE_ellipse_max_angle_ratio_l1942_194284

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x - y + 10 = 0

-- Define the angle F₁PF₂
def angle_F₁PF₂ (P : ℝ × ℝ) : ℝ := sorry

-- Define the ratio PF₁/PF₂
def ratio_PF₁_PF₂ (P : ℝ × ℝ) : ℝ := sorry

theorem ellipse_max_angle_ratio :
  ∀ a b : ℝ, a > 0 → b > 0 →
  ∀ P : ℝ × ℝ,
  ellipse a b P.1 P.2 →
  line_l P.1 P.2 →
  (∀ Q : ℝ × ℝ, ellipse a b Q.1 Q.2 → line_l Q.1 Q.2 → angle_F₁PF₂ P ≥ angle_F₁PF₂ Q) →
  ratio_PF₁_PF₂ P = -1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_max_angle_ratio_l1942_194284


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l1942_194294

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) :
  (2/a + 1/b) ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧
    Real.sqrt 2 = Real.sqrt (4^a₀ * 2^b₀) ∧ 2/a₀ + 1/b₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l1942_194294


namespace NUMINAMATH_CALUDE_staircase_markups_l1942_194210

/-- Represents the number of different markups for a staircase with n cells -/
def L (n : ℕ) : ℕ := n + 1

/-- Theorem stating that the number of different markups for a staircase with n cells is n + 1 -/
theorem staircase_markups (n : ℕ) : L n = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_staircase_markups_l1942_194210


namespace NUMINAMATH_CALUDE_christina_weekly_distance_l1942_194283

/-- The total distance Christina covers in a week -/
def total_distance (school_distance : ℕ) (days : ℕ) (extra_distance : ℕ) : ℕ :=
  2 * school_distance * days + 2 * extra_distance

/-- Theorem stating the total distance Christina covered in a week -/
theorem christina_weekly_distance :
  total_distance 7 5 2 = 74 := by
  sorry

end NUMINAMATH_CALUDE_christina_weekly_distance_l1942_194283


namespace NUMINAMATH_CALUDE_min_value_theorem_l1942_194258

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  (1 / x + 4 / y) ≥ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀ = 1 ∧ 1 / x₀ + 4 / y₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1942_194258


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l1942_194243

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : EvenFunction f) :
  EvenFunction (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l1942_194243


namespace NUMINAMATH_CALUDE_m_range_l1942_194249

/-- A function f parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + 3 * x - m - 2

/-- The property that f has exactly one root in (0, 1) -/
def has_one_root_in_unit_interval (m : ℝ) : Prop :=
  ∃! x, 0 < x ∧ x < 1 ∧ f m x = 0

/-- The main theorem stating the range of m -/
theorem m_range :
  ∀ m : ℝ, has_one_root_in_unit_interval m ↔ m > -2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1942_194249


namespace NUMINAMATH_CALUDE_max_leftover_candy_l1942_194230

theorem max_leftover_candy (y : ℕ) (h : y > 11) : 
  ∃ (q r : ℕ), y = 11 * q + r ∧ r > 0 ∧ r ≤ 10 := by
sorry

end NUMINAMATH_CALUDE_max_leftover_candy_l1942_194230


namespace NUMINAMATH_CALUDE_factorization_equality_l1942_194289

theorem factorization_equality (x y : ℝ) : x^2 * y - 4*y = y*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1942_194289


namespace NUMINAMATH_CALUDE_modulus_of_z_l1942_194290

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def given_equation (z : ℂ) : Prop := (1 + Real.sqrt 3 * i) * z = 4

-- State the theorem
theorem modulus_of_z (z : ℂ) (h : given_equation z) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1942_194290
