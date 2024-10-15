import Mathlib

namespace NUMINAMATH_CALUDE_product_of_symmetric_complex_numbers_l1093_109376

/-- Two complex numbers are symmetric with respect to the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal. -/
def symmetric_wrt_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

theorem product_of_symmetric_complex_numbers :
  ∀ z₁ z₂ : ℂ, symmetric_wrt_imaginary_axis z₁ z₂ → z₁ = 2 + I → z₁ * z₂ = -5 := by
  sorry

#check product_of_symmetric_complex_numbers

end NUMINAMATH_CALUDE_product_of_symmetric_complex_numbers_l1093_109376


namespace NUMINAMATH_CALUDE_binary_1011001_equals_quaternary_1121_l1093_109387

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldr (λ x acc => 2 * acc + if x then 1 else 0) 0

def decimal_to_quaternary (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List (Fin 4) :=
    if m = 0 then [] else (m % 4) :: aux (m / 4)
  aux n |>.reverse

theorem binary_1011001_equals_quaternary_1121 :
  decimal_to_quaternary (binary_to_decimal [true, false, true, true, false, false, true]) = [1, 1, 2, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_1011001_equals_quaternary_1121_l1093_109387


namespace NUMINAMATH_CALUDE_pant_price_before_discount_l1093_109388

/-- The cost of a wardrobe given specific items and prices --/
def wardrobe_cost (skirt_price blouse_price pant_price : ℚ) : ℚ :=
  3 * skirt_price + 5 * blouse_price + (pant_price + pant_price / 2)

/-- Theorem stating the cost of pants before discount --/
theorem pant_price_before_discount :
  ∃ (pant_price : ℚ),
    wardrobe_cost 20 15 pant_price = 180 ∧
    pant_price = 30 :=
by
  sorry

#check pant_price_before_discount

end NUMINAMATH_CALUDE_pant_price_before_discount_l1093_109388


namespace NUMINAMATH_CALUDE_factor_expression_l1093_109373

theorem factor_expression (x : ℝ) : 5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1093_109373


namespace NUMINAMATH_CALUDE_intersection_A_B_l1093_109339

def set_A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1)}
def set_B : Set ℝ := {0, 1, 2, 3}

theorem intersection_A_B : set_A ∩ set_B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1093_109339


namespace NUMINAMATH_CALUDE_cycle_cost_proof_l1093_109399

def cycle_problem (selling_price : ℕ) (gain_percentage : ℕ) : Prop :=
  let original_cost : ℕ := selling_price / 2
  selling_price = original_cost * (100 + gain_percentage) / 100 ∧
  original_cost = 1000

theorem cycle_cost_proof :
  cycle_problem 2000 100 :=
sorry

end NUMINAMATH_CALUDE_cycle_cost_proof_l1093_109399


namespace NUMINAMATH_CALUDE_relay_race_theorem_l1093_109359

def relay_race_length (team_size : ℕ) (standard_distance : ℝ) (long_distance_multiplier : ℝ) : ℝ :=
  (team_size - 1) * standard_distance + long_distance_multiplier * standard_distance

theorem relay_race_theorem :
  relay_race_length 5 3 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_theorem_l1093_109359


namespace NUMINAMATH_CALUDE_solution_set_for_inequality_l1093_109343

theorem solution_set_for_inequality (a b : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + b = 0 ↔ x = 1 ∨ x = 2) →
  {x : ℝ | x ≤ 1} = Set.Iic 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_for_inequality_l1093_109343


namespace NUMINAMATH_CALUDE_max_rooms_needed_l1093_109368

/-- Represents a group of football fans -/
structure FanGroup where
  team : Fin 3
  gender : Bool
  count : Nat

/-- The maximum number of fans that can be accommodated in one room -/
def maxFansPerRoom : Nat := 3

/-- The total number of football fans -/
def totalFans : Nat := 100

/-- Calculates the number of rooms needed for a given fan group -/
def roomsNeeded (group : FanGroup) : Nat :=
  (group.count + maxFansPerRoom - 1) / maxFansPerRoom

/-- Theorem stating the maximum number of rooms needed -/
theorem max_rooms_needed (fans : List FanGroup) 
  (h1 : fans.length = 6)
  (h2 : fans.foldl (λ acc g => acc + g.count) 0 = totalFans) : 
  (fans.foldl (λ acc g => acc + roomsNeeded g) 0) ≤ 37 := by
  sorry

end NUMINAMATH_CALUDE_max_rooms_needed_l1093_109368


namespace NUMINAMATH_CALUDE_find_a_and_b_l1093_109357

theorem find_a_and_b : ∃ a b : ℤ, 
  (a - b = 831) ∧ 
  (a = 21 * b + 11) ∧ 
  (a = 872) ∧ 
  (b = 41) := by
  sorry

end NUMINAMATH_CALUDE_find_a_and_b_l1093_109357


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l1093_109394

-- Define the function f(x) = -x^3
def f (x : ℝ) : ℝ := -x^3

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l1093_109394


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1093_109393

theorem regular_polygon_sides (d : ℕ) : d = 14 → ∃ n : ℕ, n > 2 ∧ d = n * (n - 3) / 2 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1093_109393


namespace NUMINAMATH_CALUDE_team_a_champion_probability_l1093_109350

/-- The probability of a team winning a single game -/
def game_win_prob : ℝ := 0.5

/-- The number of games Team A needs to win to become champion -/
def team_a_games_needed : ℕ := 1

/-- The number of games Team B needs to win to become champion -/
def team_b_games_needed : ℕ := 2

/-- The probability of Team A becoming the champion -/
def team_a_champion_prob : ℝ := 1 - game_win_prob ^ team_b_games_needed

theorem team_a_champion_probability :
  team_a_champion_prob = 0.75 := by sorry

end NUMINAMATH_CALUDE_team_a_champion_probability_l1093_109350


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1093_109378

/-- Given a geometric sequence {a_n} where a_1 and a_10 are the roots of 2x^2 + 5x + 1 = 0,
    prove that a_4 * a_7 = 1/2 -/
theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  (2 * (a 1)^2 + 5 * (a 1) + 1 = 0) →       -- a_1 is a root
  (2 * (a 10)^2 + 5 * (a 10) + 1 = 0) →     -- a_10 is a root
  a 4 * a 7 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1093_109378


namespace NUMINAMATH_CALUDE_total_accessories_count_l1093_109360

def dresses_per_day_first_period : ℕ := 4
def days_first_period : ℕ := 10
def dresses_per_day_second_period : ℕ := 5
def days_second_period : ℕ := 3
def ribbons_per_dress : ℕ := 3
def buttons_per_dress : ℕ := 2
def lace_trims_per_dress : ℕ := 1

theorem total_accessories_count :
  (dresses_per_day_first_period * days_first_period +
   dresses_per_day_second_period * days_second_period) *
  (ribbons_per_dress + buttons_per_dress + lace_trims_per_dress) = 330 := by
sorry

end NUMINAMATH_CALUDE_total_accessories_count_l1093_109360


namespace NUMINAMATH_CALUDE_first_night_rate_is_30_l1093_109340

/-- Represents the pricing structure of a guest house -/
structure GuestHousePricing where
  firstNightRate : ℕ  -- Flat rate for the first night
  additionalNightRate : ℕ  -- Fixed rate for each additional night

/-- Calculates the total cost for a stay -/
def totalCost (p : GuestHousePricing) (nights : ℕ) : ℕ :=
  p.firstNightRate + (nights - 1) * p.additionalNightRate

/-- Theorem stating that the flat rate for the first night is 30 -/
theorem first_night_rate_is_30 :
  ∃ (p : GuestHousePricing),
    totalCost p 4 = 210 ∧
    totalCost p 8 = 450 ∧
    p.firstNightRate = 30 := by
  sorry

end NUMINAMATH_CALUDE_first_night_rate_is_30_l1093_109340


namespace NUMINAMATH_CALUDE_complex_product_equals_43_l1093_109356

theorem complex_product_equals_43 :
  let y : ℂ := Complex.exp (Complex.I * (2 * Real.pi / 9))
  (2 * y + y^2) * (2 * y^2 + y^4) * (2 * y^3 + y^6) * 
  (2 * y^4 + y^8) * (2 * y^5 + y^10) * (2 * y^6 + y^12) = 43 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_43_l1093_109356


namespace NUMINAMATH_CALUDE_gold_coin_value_l1093_109323

theorem gold_coin_value :
  let silver_coin_value : ℕ := 25
  let gold_coins : ℕ := 3
  let silver_coins : ℕ := 5
  let cash : ℕ := 30
  let total_value : ℕ := 305
  ∃ (gold_coin_value : ℕ),
    gold_coin_value * gold_coins + silver_coin_value * silver_coins + cash = total_value ∧
    gold_coin_value = 50 :=
by sorry

end NUMINAMATH_CALUDE_gold_coin_value_l1093_109323


namespace NUMINAMATH_CALUDE_probability_standard_deck_l1093_109380

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (red_cards : Nat)
  (black_cards : Nat)

/-- Probability of drawing two red cards followed by two black cards -/
def probability_two_red_two_black (d : Deck) : Rat :=
  if d.total_cards ≥ 4 ∧ d.red_cards ≥ 2 ∧ d.black_cards ≥ 2 then
    (d.red_cards * (d.red_cards - 1) * d.black_cards * (d.black_cards - 1)) /
    (d.total_cards * (d.total_cards - 1) * (d.total_cards - 2) * (d.total_cards - 3))
  else
    0

theorem probability_standard_deck :
  probability_two_red_two_black ⟨52, 26, 26⟩ = 325 / 4998 := by
  sorry

end NUMINAMATH_CALUDE_probability_standard_deck_l1093_109380


namespace NUMINAMATH_CALUDE_prob_ace_king_correct_l1093_109301

/-- A standard deck of cards. -/
structure Deck :=
  (cards : Fin 52)

/-- The probability of drawing an Ace first and a King second from a standard deck. -/
def prob_ace_then_king (d : Deck) : ℚ :=
  4 / 663

/-- Theorem: The probability of drawing an Ace first and a King second from a standard 52-card deck is 4/663. -/
theorem prob_ace_king_correct (d : Deck) : prob_ace_then_king d = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_king_correct_l1093_109301


namespace NUMINAMATH_CALUDE_cistern_fill_time_l1093_109358

/-- Represents the time taken to fill a cistern given the rates of three pipes -/
def fill_time (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

/-- Theorem stating that given pipes with specific rates will fill the cistern in 7.5 hours -/
theorem cistern_fill_time :
  fill_time (1/10) (1/12) (-1/20) = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l1093_109358


namespace NUMINAMATH_CALUDE_total_pay_calculation_l1093_109397

theorem total_pay_calculation (pay_B : ℕ) (pay_A : ℕ) : 
  pay_B = 224 → 
  pay_A = (150 * pay_B) / 100 → 
  pay_A + pay_B = 560 :=
by
  sorry

end NUMINAMATH_CALUDE_total_pay_calculation_l1093_109397


namespace NUMINAMATH_CALUDE_range_of_k_l1093_109381

-- Define the complex number z
variable (z : ℂ)

-- Define sets A and B
def A (m k : ℝ) : Set ℂ :=
  {z | z = (2*m - Real.log (k+1)/k / Real.log (Real.sqrt 2)) + (m + Real.log (k+1)/k / Real.log (Real.sqrt 2)) * Complex.I}

def B (m : ℝ) : Set ℂ :=
  {z | Complex.abs z ≤ 2*m - 1}

-- Define the theorem
theorem range_of_k (m : ℝ) :
  (∀ k : ℝ, (A m k) ∩ (B m) = ∅) ↔ 
  ((4 * Real.sqrt 2 + 1) / 31 < k ∧ k < Real.sqrt 2 + 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l1093_109381


namespace NUMINAMATH_CALUDE_det_A_equals_two_l1093_109363

theorem det_A_equals_two (a d : ℝ) (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A = !![a, 2; -3, d] →
  A + A⁻¹ = 0 →
  Matrix.det A = 2 := by
sorry

end NUMINAMATH_CALUDE_det_A_equals_two_l1093_109363


namespace NUMINAMATH_CALUDE_smallest_number_is_57_l1093_109392

theorem smallest_number_is_57 (a b c d : ℕ) 
  (sum_abc : a + b + c = 234)
  (sum_abd : a + b + d = 251)
  (sum_acd : a + c + d = 284)
  (sum_bcd : b + c + d = 299) :
  min a (min b (min c d)) = 57 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_is_57_l1093_109392


namespace NUMINAMATH_CALUDE_cube_volume_problem_l1093_109333

theorem cube_volume_problem (reference_cube_volume : ℝ) 
  (surface_area_ratio : ℝ) (target_cube_volume : ℝ) : 
  reference_cube_volume = 8 →
  surface_area_ratio = 3 →
  (6 * (reference_cube_volume ^ (1/3))^2) * surface_area_ratio = 6 * (target_cube_volume ^ (1/3))^2 →
  target_cube_volume = 24 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l1093_109333


namespace NUMINAMATH_CALUDE_infinitely_many_special_numbers_l1093_109361

/-- Sum of digits of a natural number's decimal representation -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for special numbers -/
def is_special (m : ℕ) : Prop :=
  ∀ n : ℕ, m ≠ n + sum_of_digits n

/-- Theorem stating that there are infinitely many special numbers -/
theorem infinitely_many_special_numbers :
  ∀ k : ℕ, ∃ S : Finset ℕ, (∀ m ∈ S, is_special m) ∧ S.card > k :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_special_numbers_l1093_109361


namespace NUMINAMATH_CALUDE_square_sheet_area_decrease_l1093_109329

theorem square_sheet_area_decrease (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (2 * b = 0.1 * 4 * a) → (1 - (a - b)^2 / a^2 = 0.04) := by
  sorry

end NUMINAMATH_CALUDE_square_sheet_area_decrease_l1093_109329


namespace NUMINAMATH_CALUDE_quadratic_equations_same_roots_l1093_109313

/-- Two quadratic equations have the same roots if and only if their coefficients are proportional -/
theorem quadratic_equations_same_roots (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) (ha₁ : a₁ ≠ 0) (ha₂ : a₂ ≠ 0) :
  (∀ x, a₁ * x^2 + b₁ * x + c₁ = 0 ↔ a₂ * x^2 + b₂ * x + c₂ = 0) ↔
  ∃ k : ℝ, k ≠ 0 ∧ a₁ = k * a₂ ∧ b₁ = k * b₂ ∧ c₁ = k * c₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_same_roots_l1093_109313


namespace NUMINAMATH_CALUDE_star_four_three_l1093_109349

def star (a b : ℕ) : ℕ := 3 * a^2 + 5 * b

theorem star_four_three : star 4 3 = 63 := by
  sorry

end NUMINAMATH_CALUDE_star_four_three_l1093_109349


namespace NUMINAMATH_CALUDE_inequality_condition_l1093_109300

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x + a / x ≥ 2) ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_condition_l1093_109300


namespace NUMINAMATH_CALUDE_symmetry_condition_implies_symmetric_about_one_l1093_109320

/-- A function f: ℝ → ℝ is symmetric about x = 1 if f(1 + x) = f(1 - x) for all x ∈ ℝ -/
def SymmetricAboutOne (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 + x) = f (1 - x)

/-- Main theorem: If f(x) - f(2 - x) = 0 for all x, then f is symmetric about x = 1 -/
theorem symmetry_condition_implies_symmetric_about_one (f : ℝ → ℝ) 
    (h : ∀ x, f x - f (2 - x) = 0) : SymmetricAboutOne f := by
  sorry

#check symmetry_condition_implies_symmetric_about_one

end NUMINAMATH_CALUDE_symmetry_condition_implies_symmetric_about_one_l1093_109320


namespace NUMINAMATH_CALUDE_largest_element_complement_A_intersect_B_l1093_109348

def I : Set ℤ := {x | 1 ≤ x ∧ x ≤ 100}
def A : Set ℤ := {m ∈ I | ∃ k : ℤ, m = 2 * k + 1}
def B : Set ℤ := {n ∈ I | ∃ k : ℤ, n = 3 * k}

theorem largest_element_complement_A_intersect_B :
  ∃ x : ℤ, x ∈ (I \ A) ∩ B ∧ x = 96 ∧ ∀ y ∈ (I \ A) ∩ B, y ≤ x :=
sorry

end NUMINAMATH_CALUDE_largest_element_complement_A_intersect_B_l1093_109348


namespace NUMINAMATH_CALUDE_triangle_OAB_and_point_C_l1093_109335

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, -1)
def B : ℝ × ℝ := (1, 2)

-- Define the area of a triangle given three points
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define perpendicularity of two vectors
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := sorry

-- Define parallelism of two vectors
def parallel (v1 v2 : ℝ × ℝ) : Prop := sorry

-- Define the vector between two points
def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem triangle_OAB_and_point_C :
  -- Part 1: Area of triangle OAB
  triangle_area O A B = 5/2 ∧
  -- Part 2: Coordinates of point C
  ∃ C : ℝ × ℝ,
    perpendicular (vector B C) (vector A B) ∧
    parallel (vector A C) (vector O B) ∧
    C = (4, 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_OAB_and_point_C_l1093_109335


namespace NUMINAMATH_CALUDE_cosine_value_proof_l1093_109375

theorem cosine_value_proof (α : Real) 
  (h1 : Real.sin (Real.pi - α) = 1/3)
  (h2 : Real.pi/2 ≤ α)
  (h3 : α ≤ Real.pi) :
  Real.cos α = -2 * Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cosine_value_proof_l1093_109375


namespace NUMINAMATH_CALUDE_inequality_solution_l1093_109367

theorem inequality_solution (x : ℝ) : 
  (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) →
  (2 / (x - 2) - 3 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 20 ↔
   x < -2 ∨ (-1 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 7) ∨ x > 8) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1093_109367


namespace NUMINAMATH_CALUDE_cat_max_distance_l1093_109309

/-- The maximum distance a cat can be from the origin, given it's tied to a post -/
theorem cat_max_distance (post_x post_y rope_length : ℝ) : 
  post_x = 6 → post_y = 8 → rope_length = 15 → 
  ∃ (max_distance : ℝ), max_distance = 25 ∧ 
  ∀ (cat_x cat_y : ℝ), 
    (cat_x - post_x)^2 + (cat_y - post_y)^2 ≤ rope_length^2 → 
    cat_x^2 + cat_y^2 ≤ max_distance^2 :=
by sorry

end NUMINAMATH_CALUDE_cat_max_distance_l1093_109309


namespace NUMINAMATH_CALUDE_base8_532_equals_base7_1006_l1093_109389

/-- Converts a number from base 8 to base 10 --/
def base8ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 --/
def decimalToBase7 (n : ℕ) : ℕ := sorry

/-- Theorem stating that 532 in base 8 is equal to 1006 in base 7 --/
theorem base8_532_equals_base7_1006 : 
  decimalToBase7 (base8ToDecimal 532) = 1006 := by sorry

end NUMINAMATH_CALUDE_base8_532_equals_base7_1006_l1093_109389


namespace NUMINAMATH_CALUDE_power_of_i_sum_l1093_109321

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem power_of_i_sum : i^123 - i^321 + i^432 = -2*i + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_i_sum_l1093_109321


namespace NUMINAMATH_CALUDE_basketball_team_chemistry_count_l1093_109327

theorem basketball_team_chemistry_count :
  ∀ (total_players physics_players both_players chemistry_players : ℕ),
    total_players = 15 →
    physics_players = 8 →
    both_players = 3 →
    physics_players + chemistry_players - both_players = total_players →
    chemistry_players = 10 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_chemistry_count_l1093_109327


namespace NUMINAMATH_CALUDE_parabola_standard_form_l1093_109346

/-- A parabola with vertex at the origin and axis of symmetry x = -2 has the standard form equation y² = 8x -/
theorem parabola_standard_form (p : ℝ) (h : p / 2 = 2) :
  ∀ x y : ℝ, y^2 = 8 * x ↔ y^2 = 2 * p * x :=
by sorry

end NUMINAMATH_CALUDE_parabola_standard_form_l1093_109346


namespace NUMINAMATH_CALUDE_smallest_angle_in_ratio_quadrilateral_l1093_109369

/-- 
Given a quadrilateral where the measures of interior angles are in the ratio 1:2:3:4,
prove that the measure of the smallest interior angle is 36°.
-/
theorem smallest_angle_in_ratio_quadrilateral : 
  ∀ (a b c d : ℝ),
  a > 0 → b > 0 → c > 0 → d > 0 →
  b = 2*a → c = 3*a → d = 4*a →
  a + b + c + d = 360 →
  a = 36 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_ratio_quadrilateral_l1093_109369


namespace NUMINAMATH_CALUDE_f_monotonic_decreasing_iff_a_in_range_l1093_109308

-- Define the function f(x) = ax|x-a|
def f (a : ℝ) (x : ℝ) : ℝ := a * x * abs (x - a)

-- Define the property of being monotonically decreasing on an interval
def monotonically_decreasing (g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → g y < g x

-- State the theorem
theorem f_monotonic_decreasing_iff_a_in_range :
  ∀ a : ℝ, (monotonically_decreasing (f a) 1 (3/2)) ↔ 
    (a < 0 ∨ (3/2 ≤ a ∧ a ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_f_monotonic_decreasing_iff_a_in_range_l1093_109308


namespace NUMINAMATH_CALUDE_inequality_solution_and_a_range_l1093_109370

def f (x : ℝ) := |3*x + 2|

theorem inequality_solution_and_a_range :
  (∀ x : ℝ, f x < 6 - |x - 2| ↔ -3/2 < x ∧ x < 1) ∧
  (∀ m n : ℝ, m > 0 → n > 0 → m + n = 4 →
    (∀ a : ℝ, a > 0 →
      (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) →
        0 < a ∧ a ≤ 1/3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_and_a_range_l1093_109370


namespace NUMINAMATH_CALUDE_toilet_paper_packs_is_14_l1093_109382

/-- The number of packs of toilet paper Stella needs to buy after 4 weeks -/
def toilet_paper_packs : ℕ :=
  let bathrooms : ℕ := 6
  let days_per_week : ℕ := 7
  let rolls_per_pack : ℕ := 12
  let weeks : ℕ := 4
  let rolls_per_day : ℕ := bathrooms
  let rolls_per_week : ℕ := rolls_per_day * days_per_week
  let total_rolls : ℕ := rolls_per_week * weeks
  total_rolls / rolls_per_pack

theorem toilet_paper_packs_is_14 : toilet_paper_packs = 14 := by
  sorry

end NUMINAMATH_CALUDE_toilet_paper_packs_is_14_l1093_109382


namespace NUMINAMATH_CALUDE_walk_distance_l1093_109312

/-- The total distance walked by Erin and Susan -/
def total_distance (susan_distance erin_distance : ℕ) : ℕ :=
  susan_distance + erin_distance

/-- Theorem stating the total distance walked by Erin and Susan -/
theorem walk_distance :
  ∀ (susan_distance erin_distance : ℕ),
    susan_distance = 9 →
    erin_distance = susan_distance - 3 →
    total_distance susan_distance erin_distance = 15 := by
  sorry

end NUMINAMATH_CALUDE_walk_distance_l1093_109312


namespace NUMINAMATH_CALUDE_value_of_y_l1093_109305

theorem value_of_y (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l1093_109305


namespace NUMINAMATH_CALUDE_park_area_is_20000_l1093_109318

/-- Represents a rectangular park -/
structure RectangularPark where
  length : ℝ
  breadth : ℝ
  cyclingSpeed : ℝ
  cyclingTime : ℝ

/-- Calculates the area of a rectangular park -/
def parkArea (park : RectangularPark) : ℝ :=
  park.length * park.breadth

/-- Calculates the perimeter of a rectangular park -/
def parkPerimeter (park : RectangularPark) : ℝ :=
  2 * (park.length + park.breadth)

/-- Theorem: Given the conditions, the area of the park is 20,000 square meters -/
theorem park_area_is_20000 (park : RectangularPark) 
    (h1 : park.length = park.breadth / 2)
    (h2 : park.cyclingSpeed = 6)  -- in km/hr
    (h3 : park.cyclingTime = 1/10)  -- 6 minutes in hours
    (h4 : parkPerimeter park = park.cyclingSpeed * park.cyclingTime * 1000) : 
    parkArea park = 20000 := by
  sorry


end NUMINAMATH_CALUDE_park_area_is_20000_l1093_109318


namespace NUMINAMATH_CALUDE_sum_of_integers_l1093_109324

theorem sum_of_integers (x y : ℕ+) (h1 : x^2 + y^2 = 181) (h2 : x * y = 90) : x + y = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1093_109324


namespace NUMINAMATH_CALUDE_max_z_value_l1093_109353

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x*y + y*z + z*x = 3) :
  z ≤ 13/3 := by
sorry

end NUMINAMATH_CALUDE_max_z_value_l1093_109353


namespace NUMINAMATH_CALUDE_intersection_of_M_and_complement_of_N_l1093_109371

-- Define the universal set U as ℝ
def U := ℝ

-- Define set M
def M : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := {x : ℝ | x ≥ 1}

-- Theorem statement
theorem intersection_of_M_and_complement_of_N :
  M ∩ (Set.univ \ N) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_complement_of_N_l1093_109371


namespace NUMINAMATH_CALUDE_binomial_12_11_l1093_109328

theorem binomial_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_11_l1093_109328


namespace NUMINAMATH_CALUDE_bathroom_length_l1093_109354

theorem bathroom_length (area : ℝ) (width : ℝ) (length : ℝ) : 
  area = 8 → width = 2 → area = length * width → length = 4 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_length_l1093_109354


namespace NUMINAMATH_CALUDE_function_property_l1093_109391

/-- A function satisfying the given condition -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * ((f x)^2 - f x * f y + (f y)^2)

/-- The main theorem -/
theorem function_property (f : ℝ → ℝ) (h : special_function f) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1093_109391


namespace NUMINAMATH_CALUDE_draw_jack_queen_king_of_hearts_probability_l1093_109377

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (jacks : Nat)
  (queens : Nat)
  (king_of_hearts : Nat)

/-- The probability of drawing a specific sequence of cards from a deck -/
def draw_probability (d : Deck) : ℚ :=
  (d.jacks : ℚ) / d.total_cards *
  (d.queens : ℚ) / (d.total_cards - 1) *
  (d.king_of_hearts : ℚ) / (d.total_cards - 2)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  ⟨52, 4, 4, 1⟩

theorem draw_jack_queen_king_of_hearts_probability :
  draw_probability standard_deck = 4 / 33150 := by
  sorry

end NUMINAMATH_CALUDE_draw_jack_queen_king_of_hearts_probability_l1093_109377


namespace NUMINAMATH_CALUDE_complete_square_sum_l1093_109396

theorem complete_square_sum (a b c d : ℤ) : 
  (∀ x : ℝ, 64 * x^2 + 96 * x - 36 = 0 ↔ (a * x + b)^2 + d = c) →
  a > 0 →
  a + b + c + d = -94 :=
by sorry

end NUMINAMATH_CALUDE_complete_square_sum_l1093_109396


namespace NUMINAMATH_CALUDE_sqrt_288_simplification_l1093_109307

theorem sqrt_288_simplification : Real.sqrt 288 = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_288_simplification_l1093_109307


namespace NUMINAMATH_CALUDE_cyclic_equality_l1093_109395

theorem cyclic_equality (a b c x y z : ℝ) 
  (h1 : a = b * z + c * y)
  (h2 : b = c * x + a * z)
  (h3 : c = a * y + b * x)
  (hx : x ≠ 1 ∧ x ≠ -1)
  (hy : y ≠ 1 ∧ y ≠ -1)
  (hz : z ≠ 1 ∧ z ≠ -1) :
  a^2 / (1 - x^2) = b^2 / (1 - y^2) ∧ b^2 / (1 - y^2) = c^2 / (1 - z^2) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_equality_l1093_109395


namespace NUMINAMATH_CALUDE_initial_mat_weavers_l1093_109345

theorem initial_mat_weavers : ℕ :=
  let initial_weavers : ℕ := sorry
  let initial_mats : ℕ := 4
  let initial_days : ℕ := 4
  let second_weavers : ℕ := 14
  let second_mats : ℕ := 49
  let second_days : ℕ := 14

  have h1 : initial_weavers * initial_days * second_mats = second_weavers * second_days * initial_mats := by sorry

  have h2 : initial_weavers = 4 := by sorry

  4


end NUMINAMATH_CALUDE_initial_mat_weavers_l1093_109345


namespace NUMINAMATH_CALUDE_train_passes_jogger_l1093_109344

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passes_jogger (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (initial_distance : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  initial_distance = 260 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 38 := by
  sorry

end NUMINAMATH_CALUDE_train_passes_jogger_l1093_109344


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1093_109310

theorem quadratic_equation_roots (a b c : ℝ) (h1 : a = 1) (h2 : b = -4) (h3 : c = -5) :
  ∃ (x y : ℝ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1093_109310


namespace NUMINAMATH_CALUDE_curve_symmetry_l1093_109384

/-- The curve equation -/
def curve (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0

/-- The line equation -/
def line (m : ℝ) (x y : ℝ) : Prop := x + m*y + 4 = 0

/-- Two points are symmetric with respect to a line -/
def symmetric (P Q : ℝ × ℝ) (m : ℝ) : Prop :=
  ∃ (R : ℝ × ℝ), line m R.1 R.2 ∧ 
    R.1 = (P.1 + Q.1) / 2 ∧ 
    R.2 = (P.2 + Q.2) / 2

theorem curve_symmetry (m : ℝ) :
  (∃ (P Q : ℝ × ℝ), curve P.1 P.2 ∧ curve Q.1 Q.2 ∧ symmetric P Q m) →
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetry_l1093_109384


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l1093_109330

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(2*x) * 50^x = 250^3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l1093_109330


namespace NUMINAMATH_CALUDE_inequalities_hold_l1093_109337

theorem inequalities_hold (a b c x y z : ℝ) 
  (hx : x^2 < a^2) (hy : y^2 < b^2) (hz : z^2 < c^2) :
  (x^2*y^2 + y^2*z^2 + z^2*x^2 < a^2*b^2 + b^2*c^2 + c^2*a^2) ∧
  (x^4 + y^4 + z^4 < a^4 + b^4 + c^4) ∧
  (x^2*y^2*z^2 < a^2*b^2*c^2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l1093_109337


namespace NUMINAMATH_CALUDE_teammates_score_l1093_109332

def volleyball_scores (total_team_score : ℕ) : Prop :=
  ∃ (lizzie nathalie aimee julia ellen other : ℕ),
    lizzie = 4 ∧
    nathalie = 2 * lizzie + 3 ∧
    aimee = 2 * (lizzie + nathalie) + 1 ∧
    julia = nathalie / 2 - 2 ∧
    ellen = Int.sqrt aimee * 3 ∧
    lizzie + nathalie + aimee + julia + ellen + other = total_team_score

theorem teammates_score :
  volleyball_scores 100 → ∃ other : ℕ, other = 36 :=
by sorry

end NUMINAMATH_CALUDE_teammates_score_l1093_109332


namespace NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l1093_109322

theorem greatest_integer_for_all_real_domain : 
  ∃ (b : ℤ), (∀ (x : ℝ), x^2 + b * x + 12 ≠ 0) ∧ 
  (∀ (c : ℤ), c > b → ∃ (x : ℝ), x^2 + c * x + 12 = 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l1093_109322


namespace NUMINAMATH_CALUDE_rectangle_painting_possibilities_l1093_109338

theorem rectangle_painting_possibilities : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun p : ℕ × ℕ => 
      p.2 > p.1 ∧ 
      (p.1 - 4) * (p.2 - 4) = 2 * p.1 * p.2 / 3 ∧ 
      p.1 > 0 ∧ p.2 > 0)
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ 
  n = 3 :=
sorry

end NUMINAMATH_CALUDE_rectangle_painting_possibilities_l1093_109338


namespace NUMINAMATH_CALUDE_smallest_positive_integers_difference_l1093_109341

def m : ℕ := sorry

def n : ℕ := sorry

theorem smallest_positive_integers_difference : 
  (m ≥ 100) ∧ 
  (m < 1000) ∧ 
  (m % 13 = 6) ∧ 
  (∀ k : ℕ, k ≥ 100 ∧ k < 1000 ∧ k % 13 = 6 → m ≤ k) ∧
  (n ≥ 1000) ∧ 
  (n < 10000) ∧ 
  (n % 17 = 7) ∧ 
  (∀ l : ℕ, l ≥ 1000 ∧ l < 10000 ∧ l % 17 = 7 → n ≤ l) →
  n - m = 900 := by sorry

end NUMINAMATH_CALUDE_smallest_positive_integers_difference_l1093_109341


namespace NUMINAMATH_CALUDE_christmas_book_sales_l1093_109303

/-- Given a ratio of books to bookmarks and the number of bookmarks sold,
    calculate the number of books sold. -/
def books_sold (book_ratio : ℕ) (bookmark_ratio : ℕ) (bookmarks_sold : ℕ) : ℕ :=
  (book_ratio * bookmarks_sold) / bookmark_ratio

/-- Theorem stating that given the specific ratio and number of bookmarks sold,
    the number of books sold is 72. -/
theorem christmas_book_sales : books_sold 9 2 16 = 72 := by
  sorry

end NUMINAMATH_CALUDE_christmas_book_sales_l1093_109303


namespace NUMINAMATH_CALUDE_decimal_23_to_binary_binary_to_decimal_23_l1093_109379

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem decimal_23_to_binary :
  toBinary 23 = [true, true, true, false, true] :=
sorry

theorem binary_to_decimal_23 :
  fromBinary [true, true, true, false, true] = 23 :=
sorry

end NUMINAMATH_CALUDE_decimal_23_to_binary_binary_to_decimal_23_l1093_109379


namespace NUMINAMATH_CALUDE_max_rectangle_area_l1093_109342

/-- Given 40 feet of fencing for a rectangular pen, the maximum area enclosed is 100 square feet. -/
theorem max_rectangle_area (fencing : ℝ) (h : fencing = 40) :
  ∃ (length width : ℝ),
    length > 0 ∧ width > 0 ∧
    2 * (length + width) = fencing ∧
    ∀ (l w : ℝ), l > 0 → w > 0 → 2 * (l + w) = fencing → l * w ≤ length * width ∧
    length * width = 100 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l1093_109342


namespace NUMINAMATH_CALUDE_aleesia_weight_loss_weeks_aleesia_weight_loss_weeks_proof_l1093_109334

theorem aleesia_weight_loss_weeks : ℝ → Prop :=
  fun w =>
    let aleesia_weekly_loss : ℝ := 1.5
    let alexei_weekly_loss : ℝ := 2.5
    let alexei_weeks : ℝ := 8
    let total_loss : ℝ := 35
    (aleesia_weekly_loss * w + alexei_weekly_loss * alexei_weeks = total_loss) →
    w = 10

-- The proof would go here
theorem aleesia_weight_loss_weeks_proof : aleesia_weight_loss_weeks 10 := by
  sorry

end NUMINAMATH_CALUDE_aleesia_weight_loss_weeks_aleesia_weight_loss_weeks_proof_l1093_109334


namespace NUMINAMATH_CALUDE_bicycle_problem_l1093_109374

/-- Proves that given the conditions of the bicycle problem, student B's speed is 12 km/h -/
theorem bicycle_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) :
  distance = 12 ∧
  speed_ratio = 1.2 ∧
  time_difference = 1/6 →
  ∃ (speed_B : ℝ),
    speed_B = 12 ∧
    distance / speed_B - distance / (speed_ratio * speed_B) = time_difference :=
by sorry

end NUMINAMATH_CALUDE_bicycle_problem_l1093_109374


namespace NUMINAMATH_CALUDE_promotional_activity_choices_l1093_109304

/-- The number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose volunteers for the promotional activity -/
def chooseVolunteers (totalVolunteers boyCount girlCount chosenCount : ℕ) : ℕ :=
  choose boyCount 3 * choose girlCount 1 + choose boyCount 2 * choose girlCount 2

theorem promotional_activity_choices :
  chooseVolunteers 6 4 2 4 = 14 := by sorry

end NUMINAMATH_CALUDE_promotional_activity_choices_l1093_109304


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1093_109362

/-- Given two points P and Q in a Cartesian coordinate system that are symmetric
    with respect to the x-axis, prove that the sum of their x-coordinate and
    the y-coordinate (before the shift) is 3. -/
theorem symmetric_points_sum (a b : ℝ) : 
  (a - 3 = 2) →  -- x-coordinates are equal
  (1 = -(b + 1)) →  -- y-coordinates are opposites
  a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1093_109362


namespace NUMINAMATH_CALUDE_triangle_property_l1093_109314

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  let m : ℝ × ℝ := (a, Real.sqrt 3 * b)
  let n : ℝ × ℝ := (Real.cos (π / 2 - B), Real.cos (π - A))
  m.1 * n.1 + m.2 * n.2 = 0 →  -- m ⊥ n
  c = 3 →
  (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 →  -- Area formula
  A = π / 3 ∧ a = Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l1093_109314


namespace NUMINAMATH_CALUDE_tom_initial_investment_l1093_109385

/-- Represents the business scenario with Tom and Jose's investments --/
structure BusinessScenario where
  tom_investment : ℕ
  jose_investment : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Tom's initial investment given the business scenario --/
def calculate_tom_investment (scenario : BusinessScenario) : ℕ :=
  (scenario.total_profit - scenario.jose_profit) * 450000 / (scenario.jose_profit * 12)

/-- Theorem stating that Tom's initial investment was 30,000 --/
theorem tom_initial_investment :
  ∀ (scenario : BusinessScenario),
    scenario.jose_investment = 45000 ∧
    scenario.total_profit = 45000 ∧
    scenario.jose_profit = 25000 →
    calculate_tom_investment scenario = 30000 := by
  sorry


end NUMINAMATH_CALUDE_tom_initial_investment_l1093_109385


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_l1093_109372

theorem sunzi_wood_measurement (x y : ℝ) : 
  (x - y = 4.5 ∧ (1/2) * x + 1 = y) ↔ 
  (∃ (rope_length wood_length : ℝ),
    rope_length = x ∧
    wood_length = y ∧
    rope_length - wood_length = 4.5 ∧
    (1/2) * rope_length + 1 = wood_length) :=
by sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_l1093_109372


namespace NUMINAMATH_CALUDE_brown_shoes_count_l1093_109364

theorem brown_shoes_count (brown_shoes black_shoes : ℕ) : 
  black_shoes = 2 * brown_shoes →
  brown_shoes + black_shoes = 66 →
  brown_shoes = 22 := by
sorry

end NUMINAMATH_CALUDE_brown_shoes_count_l1093_109364


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1093_109316

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_prod : a 1 * a 5 = 4)
  (h_a4 : a 4 = 1) :
  ∃ q : ℝ, q = 1/2 ∧ ∀ n : ℕ, a (n + 1) = a n * q := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1093_109316


namespace NUMINAMATH_CALUDE_restaurant_glasses_count_l1093_109352

theorem restaurant_glasses_count :
  -- Define the number of glasses in each box type
  let small_box_glasses : ℕ := 12
  let large_box_glasses : ℕ := 16
  -- Define the difference in number of boxes
  let box_difference : ℕ := 16
  -- Define the average number of glasses per box
  let average_glasses : ℚ := 15
  -- Define variables for the number of each type of box
  ∀ (small_boxes large_boxes : ℕ),
  -- Condition: There are 16 more large boxes than small boxes
  large_boxes = small_boxes + box_difference →
  -- Condition: The average number of glasses per box is 15
  (small_box_glasses * small_boxes + large_box_glasses * large_boxes : ℚ) / 
    (small_boxes + large_boxes : ℚ) = average_glasses →
  -- Conclusion: The total number of glasses is 480
  small_box_glasses * small_boxes + large_box_glasses * large_boxes = 480 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_glasses_count_l1093_109352


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l1093_109306

theorem magnitude_of_complex_power (z : ℂ) :
  z = (4:ℝ)/7 + (3:ℝ)/7 * Complex.I →
  Complex.abs (z^8) = 390625/5764801 := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l1093_109306


namespace NUMINAMATH_CALUDE_tv_show_cost_l1093_109347

/-- Calculates the total cost of producing a TV show with the given parameters -/
def total_cost_tv_show (
  num_seasons : ℕ
  ) (first_season_cost_per_episode : ℕ
  ) (first_season_episodes : ℕ
  ) (last_season_episodes : ℕ
  ) : ℕ :=
  let other_season_cost_per_episode := 2 * first_season_cost_per_episode
  let first_season_cost := first_season_cost_per_episode * first_season_episodes
  let other_seasons_cost := 
    (other_season_cost_per_episode * first_season_episodes * 3 / 2) +
    (other_season_cost_per_episode * first_season_episodes * 9 / 4) +
    (other_season_cost_per_episode * first_season_episodes * 27 / 8) +
    (other_season_cost_per_episode * last_season_episodes)
  first_season_cost + other_seasons_cost

/-- The total cost of producing the TV show is $23,000,000 -/
theorem tv_show_cost :
  total_cost_tv_show 5 100000 12 24 = 23000000 := by
  sorry

end NUMINAMATH_CALUDE_tv_show_cost_l1093_109347


namespace NUMINAMATH_CALUDE_right_triangle_acute_angle_l1093_109398

theorem right_triangle_acute_angle (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
  (a + b = 90) → (a / b = 7 / 2) → min a b = 20 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angle_l1093_109398


namespace NUMINAMATH_CALUDE_angle_A_value_l1093_109325

theorem angle_A_value (A B C : Real) (a b : Real) : 
  0 < A ∧ A < π/2 →  -- A is acute
  0 < B ∧ B < π/2 →  -- B is acute
  0 < C ∧ C < π/2 →  -- C is acute
  A + B + C = π →    -- sum of angles in a triangle
  a > 0 →            -- side length is positive
  b > 0 →            -- side length is positive
  2 * a * Real.sin B = b →  -- given condition
  a / Real.sin A = b / Real.sin B →  -- law of sines
  A = π/6 := by
sorry

end NUMINAMATH_CALUDE_angle_A_value_l1093_109325


namespace NUMINAMATH_CALUDE_min_blocks_removed_for_cube_l1093_109326

/-- Given 59 cubic blocks, the minimum number of blocks that need to be taken away
    to construct a solid cube with none left over is 32. -/
theorem min_blocks_removed_for_cube (total_blocks : ℕ) (h : total_blocks = 59) :
  ∃ (n : ℕ), n^3 ≤ total_blocks ∧
             ∀ (m : ℕ), m^3 ≤ total_blocks → m ≤ n ∧
             total_blocks - n^3 = 32 :=
by sorry

end NUMINAMATH_CALUDE_min_blocks_removed_for_cube_l1093_109326


namespace NUMINAMATH_CALUDE_median_pets_is_three_l1093_109336

/-- Represents the distribution of pet ownership --/
def PetDistribution : List (ℕ × ℕ) :=
  [(2, 5), (3, 6), (4, 1), (5, 4), (6, 3)]

/-- The total number of individuals in the survey --/
def TotalIndividuals : ℕ := 19

/-- Calculates the median position for an odd number of data points --/
def MedianPosition (n : ℕ) : ℕ :=
  (n + 1) / 2

/-- Finds the median number of pets owned given the distribution --/
def MedianPets (dist : List (ℕ × ℕ)) (total : ℕ) : ℕ :=
  sorry -- Proof to be implemented

theorem median_pets_is_three :
  MedianPets PetDistribution TotalIndividuals = 3 :=
by sorry

end NUMINAMATH_CALUDE_median_pets_is_three_l1093_109336


namespace NUMINAMATH_CALUDE_bob_distance_when_met_l1093_109302

/-- The distance between points X and Y in miles -/
def total_distance : ℝ := 17

/-- Yolanda's speed for the first half of the journey in miles per hour -/
def yolanda_speed1 : ℝ := 3

/-- Yolanda's speed for the second half of the journey in miles per hour -/
def yolanda_speed2 : ℝ := 4

/-- Bob's speed for the first half of the journey in miles per hour -/
def bob_speed1 : ℝ := 4

/-- Bob's speed for the second half of the journey in miles per hour -/
def bob_speed2 : ℝ := 3

/-- The time in hours that Yolanda starts walking before Bob -/
def head_start : ℝ := 1

/-- The distance Bob walked when they met -/
def bob_distance : ℝ := 8.5004

theorem bob_distance_when_met :
  ∃ (t : ℝ), 
    t > 0 ∧ 
    t < total_distance / 2 / bob_speed1 ∧
    bob_distance = bob_speed1 * t ∧
    total_distance = 
      yolanda_speed1 * (total_distance / 2 / yolanda_speed1) +
      yolanda_speed2 * (total_distance / 2 / yolanda_speed2) +
      bob_speed1 * t +
      bob_speed2 * ((total_distance / 2 / bob_speed1 + total_distance / 2 / bob_speed2 - head_start) - t) :=
by sorry

end NUMINAMATH_CALUDE_bob_distance_when_met_l1093_109302


namespace NUMINAMATH_CALUDE_f_10_equals_10_l1093_109365

/-- An odd function satisfying certain conditions -/
def f (x : ℝ) : ℝ :=
  sorry

/-- Theorem stating the properties of f and the result to be proved -/
theorem f_10_equals_10 :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  f 1 = 1 →  -- f(1) = 1
  (∀ x : ℝ, f (x + 2) = f x + f 2) →  -- f(x+2) = f(x) + f(2)
  f 10 = 10 :=
by sorry

end NUMINAMATH_CALUDE_f_10_equals_10_l1093_109365


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1093_109311

theorem pure_imaginary_complex_number (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = b * Complex.I) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1093_109311


namespace NUMINAMATH_CALUDE_congruence_solution_l1093_109386

theorem congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 25 ∧ -150 ≡ n [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1093_109386


namespace NUMINAMATH_CALUDE_eight_triangle_positions_l1093_109390

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- The area of a triangle given three grid points -/
def triangleArea (a b c : GridPoint) : ℚ :=
  (1/2) * |a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)|

/-- Theorem: There are exactly 8 points C on the grid such that triangle ABC has area 4.5 -/
theorem eight_triangle_positions (a b : GridPoint) : 
  ∃! (s : Finset GridPoint), s.card = 8 ∧ 
    (∀ c ∈ s, triangleArea a b c = 4.5) ∧
    (∀ c : GridPoint, c ∉ s → triangleArea a b c ≠ 4.5) :=
sorry

end NUMINAMATH_CALUDE_eight_triangle_positions_l1093_109390


namespace NUMINAMATH_CALUDE_daisys_milk_problem_l1093_109355

theorem daisys_milk_problem (total_milk : ℝ) (cooking_percentage : ℝ) (leftover : ℝ) :
  total_milk = 16 ∧ cooking_percentage = 0.5 ∧ leftover = 2 →
  ∃ kids_consumption_percentage : ℝ,
    kids_consumption_percentage = 0.75 ∧
    leftover = (1 - cooking_percentage) * (total_milk - kids_consumption_percentage * total_milk) :=
by sorry

end NUMINAMATH_CALUDE_daisys_milk_problem_l1093_109355


namespace NUMINAMATH_CALUDE_town_population_growth_l1093_109319

/-- Represents the population of a town over time -/
structure TownPopulation where
  pop1991 : Nat
  pop2006 : Nat
  pop2016 : Nat

/-- Conditions for the town population -/
def ValidTownPopulation (t : TownPopulation) : Prop :=
  ∃ (n m k : Nat),
    t.pop1991 = n * n ∧
    t.pop2006 = t.pop1991 + 120 ∧
    t.pop2006 = m * m - 1 ∧
    t.pop2016 = t.pop2006 + 180 ∧
    t.pop2016 = k * k

/-- Calculate percent growth -/
def PercentGrowth (initial : Nat) (final : Nat) : ℚ :=
  (final - initial : ℚ) / initial * 100

/-- Main theorem stating the percent growth is 5% -/
theorem town_population_growth (t : TownPopulation) 
  (h : ValidTownPopulation t) : 
  PercentGrowth t.pop1991 t.pop2016 = 5 := by
  sorry

end NUMINAMATH_CALUDE_town_population_growth_l1093_109319


namespace NUMINAMATH_CALUDE_juice_vitamin_c_content_l1093_109317

/-- Vitamin C content in milligrams for different juice combinations -/
theorem juice_vitamin_c_content 
  (apple orange grapefruit : ℝ) 
  (h1 : apple + orange + grapefruit = 275) 
  (h2 : 2 * apple + 3 * orange + 4 * grapefruit = 683) : 
  orange + 2 * grapefruit = 133 := by
sorry

end NUMINAMATH_CALUDE_juice_vitamin_c_content_l1093_109317


namespace NUMINAMATH_CALUDE_next_monday_birthday_l1093_109331

/-- Represents the day of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Determines if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)

/-- Calculates the day of the week for March 15 in a given year, 
    assuming March 15, 2012 was a Friday -/
def marchFifteenDayOfWeek (year : Nat) : DayOfWeek :=
  sorry

/-- Theorem: The next year after 2012 when March 15 falls on a Monday is 2025 -/
theorem next_monday_birthday (startYear : Nat) (startDay : DayOfWeek) :
  startYear = 2012 →
  startDay = DayOfWeek.Friday →
  (∀ y, startYear < y → y < 2025 → marchFifteenDayOfWeek y ≠ DayOfWeek.Monday) →
  marchFifteenDayOfWeek 2025 = DayOfWeek.Monday :=
by sorry

end NUMINAMATH_CALUDE_next_monday_birthday_l1093_109331


namespace NUMINAMATH_CALUDE_number_of_students_in_B_l1093_109366

/-- The number of students in school B -/
def students_B : ℕ := 30

/-- The number of students in school A -/
def students_A : ℕ := 4 * students_B

/-- The number of students in school C -/
def students_C : ℕ := 3 * students_B

/-- Theorem stating that the number of students in school B is 30 -/
theorem number_of_students_in_B : 
  students_A + students_C = 210 → students_B = 30 := by
  sorry


end NUMINAMATH_CALUDE_number_of_students_in_B_l1093_109366


namespace NUMINAMATH_CALUDE_quadratic_minimum_minimizer_value_l1093_109351

theorem quadratic_minimum (c : ℝ) : 
  2 * c^2 - 7 * c + 4 ≥ 2 * (7/4)^2 - 7 * (7/4) + 4 := by
  sorry

theorem minimizer_value : 
  ∃ (c : ℝ), ∀ (x : ℝ), 2 * x^2 - 7 * x + 4 ≥ 2 * c^2 - 7 * c + 4 ∧ c = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_minimizer_value_l1093_109351


namespace NUMINAMATH_CALUDE_angle_in_specific_pyramid_l1093_109315

/-- A triangular pyramid with specific properties -/
structure TriangularPyramid where
  AB : ℝ
  CD : ℝ
  distance : ℝ
  volume : ℝ

/-- The angle between two lines in a triangular pyramid -/
def angle_between_lines (p : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem stating the angle between AB and CD in the specific triangular pyramid -/
theorem angle_in_specific_pyramid :
  let p : TriangularPyramid := {
    AB := 8,
    CD := 12,
    distance := 6,
    volume := 48
  }
  angle_between_lines p = 30 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_angle_in_specific_pyramid_l1093_109315


namespace NUMINAMATH_CALUDE_f_is_even_l1093_109383

-- Define the function
def f (x : ℝ) : ℝ := |x| + 1

-- State the theorem
theorem f_is_even : ∀ x : ℝ, f x = f (-x) := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l1093_109383
