import Mathlib

namespace NUMINAMATH_CALUDE_rainfall_problem_l536_53697

theorem rainfall_problem (total_rainfall : ℝ) (ratio : ℝ) :
  total_rainfall = 30 →
  ratio = 1.5 →
  ∃ (first_week : ℝ) (second_week : ℝ),
    first_week + second_week = total_rainfall ∧
    second_week = ratio * first_week ∧
    second_week = 18 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_problem_l536_53697


namespace NUMINAMATH_CALUDE_color_selection_problem_l536_53639

/-- The number of ways to select k distinct items from a set of n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of colors available -/
def total_colors : ℕ := 9

/-- The number of colors to be selected -/
def colors_to_select : ℕ := 3

theorem color_selection_problem :
  choose total_colors colors_to_select = 84 := by
  sorry

end NUMINAMATH_CALUDE_color_selection_problem_l536_53639


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l536_53626

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Theorem statement
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (3, 0) ∧ 
    radius = 3 ∧
    ∀ (x y : ℝ), circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l536_53626


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l536_53689

theorem complex_magnitude_proof (z : ℂ) :
  (Complex.arg z = Real.pi / 3) →
  (Complex.abs (z - 1) ^ 2 = Complex.abs z * Complex.abs (z - 2)) →
  (Complex.abs z = Real.sqrt 2 + 1 ∨ Complex.abs z = Real.sqrt 2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l536_53689


namespace NUMINAMATH_CALUDE_mom_tshirt_packages_l536_53673

/-- The number of packages mom will have when buying t-shirts -/
def packages_bought (shirts_per_package : ℕ) (total_shirts : ℕ) : ℕ :=
  total_shirts / shirts_per_package

/-- Theorem: Mom will have 3 packages when buying 39 t-shirts sold in packages of 13 -/
theorem mom_tshirt_packages :
  packages_bought 13 39 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mom_tshirt_packages_l536_53673


namespace NUMINAMATH_CALUDE_girls_share_l536_53661

theorem girls_share (total_amount : ℕ) (total_children : ℕ) (boys_share : ℕ) (num_boys : ℕ)
  (h1 : total_amount = 460)
  (h2 : total_children = 41)
  (h3 : boys_share = 12)
  (h4 : num_boys = 33) :
  (total_amount - num_boys * boys_share) / (total_children - num_boys) = 8 := by
  sorry

end NUMINAMATH_CALUDE_girls_share_l536_53661


namespace NUMINAMATH_CALUDE_selfie_difference_l536_53664

theorem selfie_difference (a b c : ℕ) (h1 : a + b + c = 2430) (h2 : 10 * b = 17 * a) (h3 : 10 * c = 23 * a) : c - a = 637 := by
  sorry

end NUMINAMATH_CALUDE_selfie_difference_l536_53664


namespace NUMINAMATH_CALUDE_trig_inequality_l536_53654

theorem trig_inequality : 
  let a := Real.sin (31 * π / 180)
  let b := Real.cos (58 * π / 180)
  let c := Real.tan (32 * π / 180)
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_trig_inequality_l536_53654


namespace NUMINAMATH_CALUDE_pages_per_day_l536_53674

/-- Given a book with 240 pages read over 12 days with equal pages per day, prove that 20 pages are read daily. -/
theorem pages_per_day (total_pages : ℕ) (days : ℕ) (pages_per_day : ℕ) : 
  total_pages = 240 → days = 12 → total_pages = days * pages_per_day → pages_per_day = 20 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_day_l536_53674


namespace NUMINAMATH_CALUDE_first_cube_weight_l536_53659

/-- Given two cubical blocks of the same metal, where the second cube's sides are twice as long
    as the first cube's and weighs 48 pounds, prove that the first cube weighs 6 pounds. -/
theorem first_cube_weight (s : ℝ) (weight_first : ℝ) (weight_second : ℝ) :
  s > 0 →
  weight_second = 48 →
  weight_second / weight_first = (2 * s)^3 / s^3 →
  weight_first = 6 :=
by sorry

end NUMINAMATH_CALUDE_first_cube_weight_l536_53659


namespace NUMINAMATH_CALUDE_total_spent_after_discount_and_tax_l536_53653

def bracelet_price : ℝ := 4
def keychain_price : ℝ := 5
def coloring_book_price : ℝ := 3
def sticker_pack_price : ℝ := 1
def toy_car_price : ℝ := 6

def bracelet_discount_rate : ℝ := 0.1
def sales_tax_rate : ℝ := 0.05

def paula_bracelets : ℕ := 3
def paula_keychains : ℕ := 2
def paula_coloring_books : ℕ := 1
def paula_sticker_packs : ℕ := 4

def olive_coloring_books : ℕ := 1
def olive_bracelets : ℕ := 2
def olive_toy_cars : ℕ := 1
def olive_sticker_packs : ℕ := 3

def nathan_toy_cars : ℕ := 4
def nathan_sticker_packs : ℕ := 5
def nathan_keychains : ℕ := 1

theorem total_spent_after_discount_and_tax : 
  let paula_total := paula_bracelets * bracelet_price + paula_keychains * keychain_price + 
                     paula_coloring_books * coloring_book_price + paula_sticker_packs * sticker_pack_price
  let olive_total := olive_coloring_books * coloring_book_price + olive_bracelets * bracelet_price + 
                     olive_toy_cars * toy_car_price + olive_sticker_packs * sticker_pack_price
  let nathan_total := nathan_toy_cars * toy_car_price + nathan_sticker_packs * sticker_pack_price + 
                      nathan_keychains * keychain_price
  let paula_discount := paula_bracelets * bracelet_price * bracelet_discount_rate
  let olive_discount := olive_bracelets * bracelet_price * bracelet_discount_rate
  let total_before_tax := paula_total - paula_discount + olive_total - olive_discount + nathan_total
  let total_after_tax := total_before_tax * (1 + sales_tax_rate)
  total_after_tax = 85.05 := by sorry

end NUMINAMATH_CALUDE_total_spent_after_discount_and_tax_l536_53653


namespace NUMINAMATH_CALUDE_ball_drawing_game_l536_53605

def total_balls : ℕ := 10
def red_balls : ℕ := 2
def black_balls : ℕ := 4
def white_balls : ℕ := 4
def win_reward : ℚ := 10
def loss_fine : ℚ := 2
def num_draws : ℕ := 10

def prob_win : ℚ := 1 / 15

theorem ball_drawing_game :
  -- Probability of winning in a single draw
  prob_win = (Nat.choose black_balls 3 + Nat.choose white_balls 3) / Nat.choose total_balls 3 ∧
  -- Probability of more than one win in 10 draws
  1 - (1 - prob_win) ^ num_draws - num_draws * prob_win * (1 - prob_win) ^ (num_draws - 1) = 1 / 6 ∧
  -- Expected total amount won (or lost) by 10 people
  (prob_win * win_reward - (1 - prob_win) * loss_fine) * num_draws = -12
  := by sorry

end NUMINAMATH_CALUDE_ball_drawing_game_l536_53605


namespace NUMINAMATH_CALUDE_probability_both_selected_l536_53601

theorem probability_both_selected (prob_X prob_Y prob_both : ℚ) : 
  prob_X = 1/7 → prob_Y = 2/9 → prob_both = prob_X * prob_Y → prob_both = 2/63 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_selected_l536_53601


namespace NUMINAMATH_CALUDE_volume_ratio_in_cycle_l536_53606

/-- Represents the state of an ideal gas -/
structure GasState where
  volume : ℝ
  pressure : ℝ
  temperature : ℝ

/-- Represents a cycle of an ideal gas -/
structure GasCycle where
  state1 : GasState
  state2 : GasState
  state3 : GasState

/-- Conditions for the gas cycle -/
def cycleConditions (cycle : GasCycle) : Prop :=
  -- 1-2 is isobaric and volume increases by 4 times
  cycle.state1.pressure = cycle.state2.pressure ∧
  cycle.state2.volume = 4 * cycle.state1.volume ∧
  -- 2-3 is isothermal
  cycle.state2.temperature = cycle.state3.temperature ∧
  cycle.state3.pressure > cycle.state2.pressure ∧
  -- 3-1 follows T = γV²
  ∃ γ : ℝ, cycle.state3.temperature = γ * cycle.state1.volume^2

theorem volume_ratio_in_cycle (cycle : GasCycle) 
  (h : cycleConditions cycle) : 
  cycle.state3.volume = 2 * cycle.state1.volume :=
sorry

end NUMINAMATH_CALUDE_volume_ratio_in_cycle_l536_53606


namespace NUMINAMATH_CALUDE_lines_parallel_or_skew_l536_53610

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the subset relation for lines and planes
variable (line_in_plane : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (line_parallel : Line → Line → Prop)

-- Define the skew relation for lines
variable (line_skew : Line → Line → Prop)

-- Theorem statement
theorem lines_parallel_or_skew
  (α β : Plane) (a b : Line)
  (h_parallel : plane_parallel α β)
  (h_a_in_α : line_in_plane a α)
  (h_b_in_β : line_in_plane b β) :
  line_parallel a b ∨ line_skew a b :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_or_skew_l536_53610


namespace NUMINAMATH_CALUDE_taehyung_walk_distance_l536_53630

/-- Proves that given Taehyung's step length of 0.45 meters, and moving 90 steps 13 times, the total distance walked is 526.5 meters. -/
theorem taehyung_walk_distance :
  let step_length : ℝ := 0.45
  let steps_per_set : ℕ := 90
  let num_sets : ℕ := 13
  step_length * (steps_per_set * num_sets : ℝ) = 526.5 := by sorry

end NUMINAMATH_CALUDE_taehyung_walk_distance_l536_53630


namespace NUMINAMATH_CALUDE_solutions_count_l536_53667

def count_solutions (n : ℕ) : ℕ := 4 * n

theorem solutions_count (n : ℕ) : 
  (count_solutions 1 = 4) → 
  (count_solutions 2 = 8) → 
  (count_solutions 3 = 12) → 
  (count_solutions 20 = 80) :=
by sorry

end NUMINAMATH_CALUDE_solutions_count_l536_53667


namespace NUMINAMATH_CALUDE_investment_time_period_l536_53655

theorem investment_time_period (P : ℝ) (rate_diff : ℝ) (interest_diff : ℝ) :
  P = 8400 →
  rate_diff = 0.05 →
  interest_diff = 840 →
  (P * rate_diff * 2 = interest_diff) := by
  sorry

end NUMINAMATH_CALUDE_investment_time_period_l536_53655


namespace NUMINAMATH_CALUDE_art_cost_theorem_l536_53644

def art_cost_problem (cost_A : ℝ) (cost_B : ℝ) (cost_C : ℝ) (cost_D : ℝ) : Prop :=
  let pieces_A := 3
  let pieces_B := 2
  let pieces_C := 3
  let pieces_D := 1
  let total_cost_A := cost_A * pieces_A
  let total_cost_B := cost_B * pieces_B
  let total_cost_C := cost_C * pieces_C
  let total_cost_D := cost_D * pieces_D
  let total_cost := total_cost_A + total_cost_B + total_cost_C + total_cost_D

  (total_cost_A = 45000) ∧
  (cost_B = cost_A * 1.25) ∧
  (cost_C = cost_A * 1.5) ∧
  (cost_D = total_cost_C * 2) ∧
  (total_cost = 285000)

theorem art_cost_theorem : ∃ cost_A cost_B cost_C cost_D, art_cost_problem cost_A cost_B cost_C cost_D :=
  sorry

end NUMINAMATH_CALUDE_art_cost_theorem_l536_53644


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l536_53637

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x^2 - x < 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l536_53637


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l536_53607

theorem arithmetic_sequence_count : 
  let a₁ : ℝ := 2.6
  let aₙ : ℝ := 52.1
  let d : ℝ := 4.5
  let n := (aₙ - a₁) / d + 1
  n = 12 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l536_53607


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l536_53679

theorem arithmetic_expression_equality : 2 - 3*(-4) - 7 + 2*(-5) - 9 + 6*(-2) = -24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l536_53679


namespace NUMINAMATH_CALUDE_liangliang_speed_l536_53624

theorem liangliang_speed (initial_distance : ℝ) (remaining_distance : ℝ) (time : ℝ) (mingming_speed : ℝ) :
  initial_distance = 3000 →
  remaining_distance = 2900 →
  time = 20 →
  mingming_speed = 80 →
  ∃ (liangliang_speed : ℝ), (liangliang_speed = 75 ∨ liangliang_speed = 85) ∧
    (initial_distance - remaining_distance = (mingming_speed - liangliang_speed) * time) :=
by sorry

end NUMINAMATH_CALUDE_liangliang_speed_l536_53624


namespace NUMINAMATH_CALUDE_multiply_by_0_064_l536_53614

theorem multiply_by_0_064 (x : ℝ) (h : 13.26 * x = 132.6) : 0.064 * x = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_0_064_l536_53614


namespace NUMINAMATH_CALUDE_prob_at_least_one_female_is_four_fifths_l536_53647

/-- Represents the number of students in each category -/
structure StudentCounts where
  total : ℕ
  maleHigh : ℕ
  femaleHigh : ℕ
  selected : ℕ
  final : ℕ

/-- Calculates the probability of selecting at least one female student -/
def probAtLeastOneFemale (counts : StudentCounts) : ℚ :=
  1 - (counts.maleHigh.choose counts.final) / (counts.maleHigh + counts.femaleHigh).choose counts.final

/-- The main theorem stating the probability of selecting at least one female student -/
theorem prob_at_least_one_female_is_four_fifths (counts : StudentCounts) 
  (h1 : counts.total = 200)
  (h2 : counts.maleHigh = 100)
  (h3 : counts.femaleHigh = 50)
  (h4 : counts.selected = 6)
  (h5 : counts.final = 3) :
  probAtLeastOneFemale counts = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_prob_at_least_one_female_is_four_fifths_l536_53647


namespace NUMINAMATH_CALUDE_soccer_team_subjects_l536_53643

theorem soccer_team_subjects (total : ℕ) (physics : ℕ) (both : ℕ) (math : ℕ) :
  total = 20 →
  physics = 12 →
  both = 6 →
  total = physics + math - both →
  math = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_team_subjects_l536_53643


namespace NUMINAMATH_CALUDE_smallest_multiple_l536_53621

theorem smallest_multiple (x : ℕ) : x = 54 ↔ 
  (x > 0 ∧ 
   250 * x % 1080 = 0 ∧ 
   ∀ y : ℕ, y > 0 → y < x → 250 * y % 1080 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_l536_53621


namespace NUMINAMATH_CALUDE_circle_equation_l536_53613

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: For a circle with center (4, -6) and radius 3,
    any point (x, y) on the circle satisfies (x - 4)^2 + (y + 6)^2 = 9 -/
theorem circle_equation (c : Circle) (p : Point) :
  c.h = 4 ∧ c.k = -6 ∧ c.r = 3 →
  (p.x - c.h)^2 + (p.y - c.k)^2 = c.r^2 →
  (p.x - 4)^2 + (p.y + 6)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l536_53613


namespace NUMINAMATH_CALUDE_odd_function_symmetry_l536_53641

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the property of being monotonically decreasing on an interval
def is_monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

-- State the theorem
theorem odd_function_symmetry (hf_odd : is_odd f) 
  (hf_decreasing : is_monotone_decreasing_on f 1 2) :
  is_monotone_decreasing_on f (-2) (-1) ∧ 
  (∀ x ∈ Set.Icc (-2) (-1), f x ≤ -f 2) ∧
  f (-2) = -f 2 :=
sorry

end NUMINAMATH_CALUDE_odd_function_symmetry_l536_53641


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l536_53680

theorem coefficient_of_x_squared (z : ℂ) (a₀ a₁ a₂ a₃ a₄ : ℂ) :
  z = 1 + I →
  (∀ x : ℂ, (x + z)^4 = a₀*x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄) →
  a₂ = 12*I :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l536_53680


namespace NUMINAMATH_CALUDE_union_equals_real_l536_53685

open Set Real

def A : Set ℝ := {x : ℝ | x^2 + x - 6 > 0}
def B : Set ℝ := {x : ℝ | -π < x ∧ x < Real.exp 1}

theorem union_equals_real : A ∪ B = univ := by
  sorry

end NUMINAMATH_CALUDE_union_equals_real_l536_53685


namespace NUMINAMATH_CALUDE_smallest_positive_period_of_f_triangle_area_l536_53635

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sqrt 3 * Real.sin x * Real.cos x + 1/2

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem smallest_positive_period_of_f :
  ∃ T > 0, is_periodic f T ∧ ∀ S, 0 < S ∧ S < T → ¬ is_periodic f S :=
sorry

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  f (B + C) = 3/2 →
  a = Real.sqrt 3 →
  b + c = 3 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_period_of_f_triangle_area_l536_53635


namespace NUMINAMATH_CALUDE_train_average_speed_l536_53675

theorem train_average_speed (distance1 distance2 time1 time2 : ℝ) 
  (h1 : distance1 = 250)
  (h2 : distance2 = 350)
  (h3 : time1 = 2)
  (h4 : time2 = 4) :
  (distance1 + distance2) / (time1 + time2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_average_speed_l536_53675


namespace NUMINAMATH_CALUDE_max_k_value_l536_53665

open Real

theorem max_k_value (f : ℝ → ℝ) (k : ℤ) : 
  (∀ x > 2, f x = x + x * log x) →
  (∀ x > 2, ↑k * (x - 2) < f x) →
  k ≤ 4 ∧ ∃ x > 2, 4 * (x - 2) < f x :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l536_53665


namespace NUMINAMATH_CALUDE_final_passenger_count_l536_53617

def bus_passengers (initial : ℕ) (first_stop : ℕ) (off_other : ℕ) (on_other : ℕ) : ℕ :=
  initial + first_stop - off_other + on_other

theorem final_passenger_count :
  bus_passengers 50 16 22 5 = 49 := by
  sorry

end NUMINAMATH_CALUDE_final_passenger_count_l536_53617


namespace NUMINAMATH_CALUDE_greatest_plants_per_row_l536_53604

theorem greatest_plants_per_row (sunflowers corn tomatoes : ℕ) 
  (h1 : sunflowers = 45)
  (h2 : corn = 81)
  (h3 : tomatoes = 63) :
  Nat.gcd sunflowers (Nat.gcd corn tomatoes) = 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_plants_per_row_l536_53604


namespace NUMINAMATH_CALUDE_gate_width_scientific_notation_l536_53602

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem gate_width_scientific_notation :
  toScientificNotation 0.000000007 = ScientificNotation.mk 7 (-9) sorry := by
  sorry

end NUMINAMATH_CALUDE_gate_width_scientific_notation_l536_53602


namespace NUMINAMATH_CALUDE_line_direction_vector_l536_53682

/-- Given a line y = (5x - 7) / 2 parameterized as (x, y) = v + t * d,
    where the distance between (x, y) and (4, 2) is t for x ≥ 4,
    prove that the direction vector d is (2/√29, 5/√29). -/
theorem line_direction_vector (v d : ℝ × ℝ) :
  (∀ x y t : ℝ, x ≥ 4 →
    y = (5 * x - 7) / 2 →
    (x, y) = v + t • d →
    ‖(x, y) - (4, 2)‖ = t) →
  d = (2 / Real.sqrt 29, 5 / Real.sqrt 29) :=
by sorry

end NUMINAMATH_CALUDE_line_direction_vector_l536_53682


namespace NUMINAMATH_CALUDE_supermarket_spend_correct_l536_53669

def supermarket_spend (initial_amount left_amount showroom_spend : ℕ) : ℕ :=
  initial_amount - left_amount - showroom_spend

theorem supermarket_spend_correct (initial_amount left_amount showroom_spend : ℕ) 
  (h1 : initial_amount ≥ left_amount + showroom_spend) :
  supermarket_spend initial_amount left_amount showroom_spend = 
    initial_amount - left_amount - showroom_spend :=
by
  sorry

#eval supermarket_spend 106 26 49

end NUMINAMATH_CALUDE_supermarket_spend_correct_l536_53669


namespace NUMINAMATH_CALUDE_compound_proposition_true_l536_53628

theorem compound_proposition_true (a b : ℝ) :
  (a > 0 ∧ a + b < 0) → b < 0 := by
  sorry

end NUMINAMATH_CALUDE_compound_proposition_true_l536_53628


namespace NUMINAMATH_CALUDE_remaining_distance_to_hotel_l536_53646

def total_distance : ℝ := 1200

def segment1_speed : ℝ := 60
def segment1_time : ℝ := 2

def segment2_speed : ℝ := 70
def segment2_time : ℝ := 3

def segment3_speed : ℝ := 50
def segment3_time : ℝ := 4

def segment4_speed : ℝ := 80
def segment4_time : ℝ := 5

def distance_traveled : ℝ :=
  segment1_speed * segment1_time +
  segment2_speed * segment2_time +
  segment3_speed * segment3_time +
  segment4_speed * segment4_time

theorem remaining_distance_to_hotel :
  total_distance - distance_traveled = 270 := by sorry

end NUMINAMATH_CALUDE_remaining_distance_to_hotel_l536_53646


namespace NUMINAMATH_CALUDE_paper_crane_folding_time_l536_53640

theorem paper_crane_folding_time (time_A time_B : ℝ) (h1 : time_A = 30) (h2 : time_B = 45) :
  (1 / time_A + 1 / time_B)⁻¹ = 18 := by sorry

end NUMINAMATH_CALUDE_paper_crane_folding_time_l536_53640


namespace NUMINAMATH_CALUDE_eggs_leftover_l536_53603

def david_eggs : ℕ := 45
def ella_eggs : ℕ := 58
def fiona_eggs : ℕ := 29
def carton_size : ℕ := 10

theorem eggs_leftover :
  (david_eggs + ella_eggs + fiona_eggs) % carton_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_eggs_leftover_l536_53603


namespace NUMINAMATH_CALUDE_inequality_equivalence_l536_53611

theorem inequality_equivalence (x : ℝ) : 
  (x + 3) / 2 - (5 * x - 1) / 5 ≥ 0 ↔ x ≤ 17 / 5 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l536_53611


namespace NUMINAMATH_CALUDE_concentric_circles_angle_l536_53616

theorem concentric_circles_angle (r₁ r₂ : ℝ) (α : ℝ) :
  r₁ = 1 →
  r₂ = 2 →
  (((360 - α) / 360 * π * r₁^2) + (α / 360 * π * r₂^2) - (α / 360 * π * r₁^2)) = (1/3) * (π * r₂^2) →
  α = 60 := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_angle_l536_53616


namespace NUMINAMATH_CALUDE_a_is_independent_variable_l536_53629

-- Define the perimeter function for a rhombus
def rhombus_perimeter (a : ℝ) : ℝ := 4 * a

-- Statement to prove
theorem a_is_independent_variable :
  ∃ (C : ℝ → ℝ), C = rhombus_perimeter ∧ 
  (∀ (a : ℝ), C a = 4 * a) ∧
  (∀ (a₁ a₂ : ℝ), a₁ ≠ a₂ → C a₁ ≠ C a₂) :=
sorry

end NUMINAMATH_CALUDE_a_is_independent_variable_l536_53629


namespace NUMINAMATH_CALUDE_area_outside_inscribed_angle_l536_53652

theorem area_outside_inscribed_angle (R : ℝ) (h : R = 12) :
  let θ : ℝ := 120 * π / 180
  let sector_area := θ / (2 * π) * π * R^2
  let triangle_area := 1/2 * R^2 * Real.sin θ
  sector_area - triangle_area = 48 * π - 72 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_area_outside_inscribed_angle_l536_53652


namespace NUMINAMATH_CALUDE_median_in_70_79_interval_l536_53622

/-- Represents a score interval with its lower bound and frequency -/
structure ScoreInterval :=
  (lower_bound : ℕ)
  (frequency : ℕ)

/-- The list of score intervals representing the histogram -/
def histogram : List ScoreInterval :=
  [⟨90, 18⟩, ⟨80, 20⟩, ⟨70, 19⟩, ⟨60, 17⟩, ⟨50, 26⟩]

/-- The total number of students -/
def total_students : ℕ := 100

/-- Function to find the interval containing the median score -/
def median_interval (hist : List ScoreInterval) (total : ℕ) : Option ScoreInterval :=
  sorry

/-- Theorem stating that the median score is in the 70-79 interval -/
theorem median_in_70_79_interval :
  median_interval histogram total_students = some ⟨70, 19⟩ := by sorry

end NUMINAMATH_CALUDE_median_in_70_79_interval_l536_53622


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l536_53691

/-- The minimum distance between a point on y = (1/2)e^x and a point on y = ln(2x) -/
theorem min_distance_between_curves : ∃ (d : ℝ),
  (∀ (x₁ x₂ : ℝ), 
    let p := (x₁, (1/2) * Real.exp x₁)
    let q := (x₂, Real.log (2 * x₂))
    d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
  d = Real.sqrt 2 * (1 - Real.log 2) := by
  sorry

#check min_distance_between_curves

end NUMINAMATH_CALUDE_min_distance_between_curves_l536_53691


namespace NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_l536_53633

def repeating_707 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def repeating_909 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

def product : ℕ := repeating_707 * repeating_909

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_tens_and_units_digits :
  tens_digit product + units_digit product = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_l536_53633


namespace NUMINAMATH_CALUDE_jose_wandering_time_l536_53638

/-- Proves that Jose's wandering time is 10 hours given his distance and speed -/
theorem jose_wandering_time : 
  ∀ (distance : ℝ) (speed : ℝ),
  distance = 15 →
  speed = 1.5 →
  distance / speed = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_jose_wandering_time_l536_53638


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l536_53648

theorem complex_fraction_equality (a b : ℝ) : 
  (Complex.I + 1) / (Complex.I - 1) = Complex.mk a b → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l536_53648


namespace NUMINAMATH_CALUDE_min_value_of_expression_l536_53619

theorem min_value_of_expression (a : ℝ) (h : a > 0) :
  a + 4 / a ≥ 4 ∧ (a + 4 / a = 4 ↔ a = 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l536_53619


namespace NUMINAMATH_CALUDE_tangent_segment_difference_l536_53684

/-- Represents a quadrilateral inscribed in a circle with an inscribed circle --/
structure InscribedQuadrilateral where
  /-- Side lengths of the quadrilateral --/
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  /-- Proof that the quadrilateral is inscribed in a circle --/
  inscribed_in_circle : True
  /-- Proof that there's a circle inscribed in the quadrilateral --/
  has_inscribed_circle : True

/-- Theorem about the difference of segments created by the inscribed circle's tangency point --/
theorem tangent_segment_difference (q : InscribedQuadrilateral)
    (h1 : q.side1 = 50)
    (h2 : q.side2 = 80)
    (h3 : q.side3 = 140)
    (h4 : q.side4 = 120) :
    ∃ (x y : ℝ), x + y = 140 ∧ |x - y| = 19 := by
  sorry


end NUMINAMATH_CALUDE_tangent_segment_difference_l536_53684


namespace NUMINAMATH_CALUDE_sin_cos_tan_product_l536_53612

theorem sin_cos_tan_product : 
  Real.sin (4/3 * Real.pi) * Real.cos (5/6 * Real.pi) * Real.tan (-4/3 * Real.pi) = -3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_tan_product_l536_53612


namespace NUMINAMATH_CALUDE_parabola_tangent_properties_l536_53671

/-- Given a parabola and a point, proves properties of its tangent lines -/
theorem parabola_tangent_properties (S : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) :
  S = (-3, 7) →
  (∀ x y, parabola x y ↔ y^2 = 5*x) →
  ∃ (t₁ t₂ : ℝ → ℝ) (P₁ P₂ : ℝ × ℝ) (α : ℝ),
    -- Tangent line equations
    (∀ x, t₁ x = x/6 + 15/2) ∧
    (∀ x, t₂ x = -5/2*x - 1/2) ∧
    -- Points of tangency
    P₁ = (45, 15) ∧
    P₂ = (1/5, -1) ∧
    -- Angle between tangents
    α = Real.arctan (32/7) ∧
    -- Tangent lines pass through S
    t₁ (S.1) = S.2 ∧
    t₂ (S.1) = S.2 ∧
    -- Points of tangency lie on the parabola
    parabola P₁.1 P₁.2 ∧
    parabola P₂.1 P₂.2 ∧
    -- Tangent lines touch the parabola at points of tangency
    t₁ P₁.1 = P₁.2 ∧
    t₂ P₂.1 = P₂.2 :=
by
  sorry


end NUMINAMATH_CALUDE_parabola_tangent_properties_l536_53671


namespace NUMINAMATH_CALUDE_dennis_purchase_cost_l536_53687

/-- The cost of Dennis's purchase after discount --/
def total_cost (pants_price sock_price : ℚ) (pants_quantity sock_quantity : ℕ) (discount : ℚ) : ℚ :=
  let discounted_pants_price := pants_price * (1 - discount)
  let discounted_sock_price := sock_price * (1 - discount)
  (discounted_pants_price * pants_quantity) + (discounted_sock_price * sock_quantity)

/-- Theorem stating the total cost of Dennis's purchase --/
theorem dennis_purchase_cost :
  total_cost 110 60 4 2 (30/100) = 392 := by
  sorry

end NUMINAMATH_CALUDE_dennis_purchase_cost_l536_53687


namespace NUMINAMATH_CALUDE_union_of_positive_and_less_than_one_is_reals_l536_53688

theorem union_of_positive_and_less_than_one_is_reals :
  let A : Set ℝ := {x | x > 0}
  let B : Set ℝ := {x | x < 1}
  A ∪ B = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_of_positive_and_less_than_one_is_reals_l536_53688


namespace NUMINAMATH_CALUDE_magnitude_of_z_l536_53634

open Complex

theorem magnitude_of_z (z : ℂ) (h : (1 + 2*I) / z = 2 - I) : abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l536_53634


namespace NUMINAMATH_CALUDE_correct_assignment_is_correct_l536_53608

-- Define the color type
inductive Color
| Red
| Blue
| Green

-- Define the assignment type
structure Assignment where
  one : Color
  two : Color
  three : Color

-- Define the correct assignment
def correct_assignment : Assignment :=
  { one := Color.Green
  , two := Color.Blue
  , three := Color.Red }

-- Theorem stating that the correct_assignment is indeed correct
theorem correct_assignment_is_correct : 
  correct_assignment.one = Color.Green ∧ 
  correct_assignment.two = Color.Blue ∧ 
  correct_assignment.three = Color.Red :=
by sorry

end NUMINAMATH_CALUDE_correct_assignment_is_correct_l536_53608


namespace NUMINAMATH_CALUDE_square_root_squared_l536_53686

theorem square_root_squared : (Real.sqrt 930249)^2 = 930249 := by
  sorry

end NUMINAMATH_CALUDE_square_root_squared_l536_53686


namespace NUMINAMATH_CALUDE_group_size_calculation_l536_53658

theorem group_size_calculation (children women men : ℕ) : 
  children = 30 →
  women = 3 * children →
  men = 2 * women →
  children + women + men = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l536_53658


namespace NUMINAMATH_CALUDE_inequality_implies_a_geq_two_l536_53609

theorem inequality_implies_a_geq_two (a : ℝ) :
  (∀ x y : ℝ, x^2 + 2*x + a ≥ -y^2 - 2*y) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_a_geq_two_l536_53609


namespace NUMINAMATH_CALUDE_trip_time_calculation_l536_53650

/-- Proves that if a trip takes 4.5 hours at 70 mph, it will take 5.25 hours at 60 mph -/
theorem trip_time_calculation (distance : ℝ) : 
  distance = 70 * 4.5 → distance = 60 * 5.25 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_calculation_l536_53650


namespace NUMINAMATH_CALUDE_gaussian_guardians_score_l536_53632

/-- The total points scored by the Gaussian Guardians basketball team -/
def total_points (daniel curtis sid emily kalyn hyojeong ty winston : ℕ) : ℕ :=
  daniel + curtis + sid + emily + kalyn + hyojeong + ty + winston

/-- Theorem stating that the total points scored by the Gaussian Guardians is 54 -/
theorem gaussian_guardians_score :
  total_points 7 8 2 11 6 12 1 7 = 54 := by
  sorry

end NUMINAMATH_CALUDE_gaussian_guardians_score_l536_53632


namespace NUMINAMATH_CALUDE_range_of_a_l536_53663

/-- The equation |x^2 - a| - x + 2 = 0 has two distinct real roots -/
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ |x^2 - a| - x + 2 = 0 ∧ |y^2 - a| - y + 2 = 0

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : has_two_distinct_roots a) : a > 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l536_53663


namespace NUMINAMATH_CALUDE_football_practice_missed_days_l536_53651

/-- The number of days a football team missed practice due to rain -/
def days_missed (daily_practice_hours : ℕ) (total_practice_hours : ℕ) (days_in_week : ℕ) : ℕ :=
  days_in_week - (total_practice_hours / daily_practice_hours)

/-- Theorem: The football team missed 1 day of practice due to rain -/
theorem football_practice_missed_days :
  days_missed 6 36 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_football_practice_missed_days_l536_53651


namespace NUMINAMATH_CALUDE_negative_reciprocal_equality_l536_53623

theorem negative_reciprocal_equality (a : ℝ) (ha : a ≠ 0) :
  -(1 / a) = (-1) / a := by
  sorry

end NUMINAMATH_CALUDE_negative_reciprocal_equality_l536_53623


namespace NUMINAMATH_CALUDE_bluejay_league_members_l536_53692

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 8

/-- The cost of a cap in dollars -/
def cap_cost : ℕ := sock_cost / 2

/-- The total cost for one member's equipment (home and away sets) -/
def member_cost : ℕ := 2 * (sock_cost + tshirt_cost + cap_cost)

/-- The total expenditure for all members -/
def total_expenditure : ℕ := 3876

/-- The number of members in the Bluejay Basketball League -/
def num_members : ℕ := total_expenditure / member_cost

theorem bluejay_league_members : num_members = 84 := by
  sorry


end NUMINAMATH_CALUDE_bluejay_league_members_l536_53692


namespace NUMINAMATH_CALUDE_valid_f_forms_l536_53696

-- Define the function g
def g (x : ℝ) : ℝ := -x^2 - 3

-- Define the properties of function f
def is_valid_f (f : ℝ → ℝ) : Prop :=
  -- f is a quadratic function
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c ∧ a ≠ 0 ∧
  -- The minimum value of f(x) on [-1,2] is 1
  (∀ x ∈ Set.Icc (-1) 2, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1) 2, f x = 1) ∧
  -- f(x) + g(x) is an odd function
  ∀ x, f (-x) + g (-x) = -(f x + g x)

-- Theorem statement
theorem valid_f_forms :
  ∀ f : ℝ → ℝ, is_valid_f f →
    (∀ x, f x = x^2 - 2 * Real.sqrt 2 * x + 3) ∨
    (∀ x, f x = x^2 + 3 * x + 3) :=
sorry

end NUMINAMATH_CALUDE_valid_f_forms_l536_53696


namespace NUMINAMATH_CALUDE_distance_between_z₁_and_z₂_l536_53695

noncomputable def z₁ : ℂ := (Complex.I * 2 + 1)⁻¹ * (Complex.I * 3 - 1)

noncomputable def z₂ : ℂ := 1 + (1 + Complex.I)^10

theorem distance_between_z₁_and_z₂ : 
  Complex.abs (z₂ - z₁) = Real.sqrt 231.68 := by sorry

end NUMINAMATH_CALUDE_distance_between_z₁_and_z₂_l536_53695


namespace NUMINAMATH_CALUDE_no_extremum_condition_l536_53625

/-- A cubic function f(x) = ax³ + bx² + cx + d with a > 0 -/
def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- The derivative of the cubic function -/
def cubic_derivative (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

/-- Condition for no extremum: the derivative is always non-negative -/
def no_extremum (a b c : ℝ) : Prop :=
  ∀ x, cubic_derivative a b c x ≥ 0

theorem no_extremum_condition (a b c d : ℝ) (ha : a > 0) :
  no_extremum a b c → b^2 - 3*a*c ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_extremum_condition_l536_53625


namespace NUMINAMATH_CALUDE_root_equation_n_value_l536_53656

theorem root_equation_n_value : 
  ∀ n : ℝ, (1 : ℝ)^2 + 3*(1 : ℝ) + n = 0 → n = -4 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_n_value_l536_53656


namespace NUMINAMATH_CALUDE_min_value_of_f_l536_53615

-- Define the function f(x)
def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- State the theorem
theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x a ≥ f y a) ∧ 
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x a = 20) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x a ≤ f y a) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x a = -7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l536_53615


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l536_53627

/-- Proves that given two cyclists on a 45-mile course, one traveling at 16 mph,
    meeting after 1.5 hours, the speed of the other cyclist must be 14 mph. -/
theorem cyclist_speed_problem (course_length : ℝ) (second_cyclist_speed : ℝ) (meeting_time : ℝ)
  (h1 : course_length = 45)
  (h2 : second_cyclist_speed = 16)
  (h3 : meeting_time = 1.5) :
  ∃ (first_cyclist_speed : ℝ),
    first_cyclist_speed * meeting_time + second_cyclist_speed * meeting_time = course_length ∧
    first_cyclist_speed = 14 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_problem_l536_53627


namespace NUMINAMATH_CALUDE_total_distance_traveled_l536_53636

def speed : ℝ := 60
def driving_sessions : List ℝ := [4, 5, 3, 2]

theorem total_distance_traveled :
  (List.sum driving_sessions) * speed = 840 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_traveled_l536_53636


namespace NUMINAMATH_CALUDE_linear_equation_condition_l536_53699

theorem linear_equation_condition (a : ℝ) : 
  (∀ x, ∃ k m, (a - 1) * x^(|a|) + 4 = k * x + m) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l536_53699


namespace NUMINAMATH_CALUDE_logarithmic_algebraic_equivalence_l536_53694

theorem logarithmic_algebraic_equivalence : 
  ¬(∀ x : ℝ, (Real.log (x^2 - 4) = Real.log (4*x - 7)) ↔ (x^2 - 4 = 4*x - 7)) :=
by sorry

end NUMINAMATH_CALUDE_logarithmic_algebraic_equivalence_l536_53694


namespace NUMINAMATH_CALUDE_polynomial_coefficient_B_l536_53666

theorem polynomial_coefficient_B (A C D : ℤ) : 
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (r₁ * r₂ * r₃ * r₄ * r₅ * r₆ : ℤ) = 64 ∧ 
    (r₁ + r₂ + r₃ + r₄ + r₅ + r₆ : ℤ) = 15 ∧ 
    ∀ (z : ℂ), z^6 - 15*z^5 + A*z^4 + (-244)*z^3 + C*z^2 + D*z + 64 = 
      (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) * (z - r₅) * (z - r₆) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_B_l536_53666


namespace NUMINAMATH_CALUDE_factor_expression_l536_53645

theorem factor_expression (x : ℝ) : x * (x - 3) - 5 * (x - 3) = (x - 5) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l536_53645


namespace NUMINAMATH_CALUDE_cubic_polynomial_sum_l536_53681

/-- A cubic polynomial with coefficients in ℝ -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Evaluation of a cubic polynomial at a point -/
def CubicPolynomial.eval (Q : CubicPolynomial) (x : ℝ) : ℝ :=
  Q.a * x^3 + Q.b * x^2 + Q.c * x + Q.d

theorem cubic_polynomial_sum (k : ℝ) (Q : CubicPolynomial) 
    (h0 : Q.eval 0 = k)
    (h1 : Q.eval 1 = 3*k)
    (h2 : Q.eval (-1) = 4*k) :
  Q.eval 2 + Q.eval (-2) = 22*k := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_sum_l536_53681


namespace NUMINAMATH_CALUDE_complex_product_theorem_l536_53631

theorem complex_product_theorem : 
  let z₁ : ℂ := 2 + Complex.I
  let z₂ : ℂ := 1 - Complex.I
  z₁ * z₂ = 3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l536_53631


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l536_53618

def f (x : ℝ) : ℝ := x^2 - 3*x + 1

theorem arithmetic_sequence_formula (a : ℝ) (a_n : ℕ → ℝ) :
  (∀ n, a_n (n + 2) - a_n (n + 1) = a_n (n + 1) - a_n n) →  -- arithmetic sequence
  a_n 1 = f (a + 1) →
  a_n 2 = 0 →
  a_n 3 = f (a - 1) →
  ((a = 1 ∧ ∀ n, a_n n = n - 2) ∨ (a = 2 ∧ ∀ n, a_n n = 2 - n)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l536_53618


namespace NUMINAMATH_CALUDE_min_value_expression_l536_53670

theorem min_value_expression (a b : ℝ) (hb : b ≠ 0) :
  a^2 + b^2 + a/b + 1/b^2 ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l536_53670


namespace NUMINAMATH_CALUDE_taxi_distribution_eq_14_l536_53662

/-- The number of ways to distribute 4 people into 2 taxis with at least one person in each taxi -/
def taxi_distribution : ℕ :=
  2^4 - 2

/-- Theorem stating that the number of ways to distribute 4 people into 2 taxis
    with at least one person in each taxi is equal to 14 -/
theorem taxi_distribution_eq_14 : taxi_distribution = 14 := by
  sorry

end NUMINAMATH_CALUDE_taxi_distribution_eq_14_l536_53662


namespace NUMINAMATH_CALUDE_ball_hit_ground_time_l536_53649

/-- The time when a ball hits the ground, given its height equation -/
theorem ball_hit_ground_time (t : ℝ) : t ≥ 0 → -8*t^2 - 12*t + 72 = 0 → t = 3 := by
  sorry

#check ball_hit_ground_time

end NUMINAMATH_CALUDE_ball_hit_ground_time_l536_53649


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l536_53620

def m : ℕ := 2017^2 + 2^2017

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ := m) : (m^2 + 2^m) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l536_53620


namespace NUMINAMATH_CALUDE_max_x_minus_y_is_half_l536_53698

theorem max_x_minus_y_is_half :
  ∀ x y : ℝ, 2 * (x^2 + y^2 - x*y) = x + y →
  ∀ z : ℝ, z = x - y → z ≤ (1/2 : ℝ) ∧ ∃ x₀ y₀ : ℝ, 2 * (x₀^2 + y₀^2 - x₀*y₀) = x₀ + y₀ ∧ x₀ - y₀ = (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_max_x_minus_y_is_half_l536_53698


namespace NUMINAMATH_CALUDE_circle_equation_l536_53693

/-- The standard equation of a circle with center (2, -2) passing through the origin -/
theorem circle_equation : ∀ (x y : ℝ), 
  (x - 2)^2 + (y + 2)^2 = 8 ↔ 
  (x - 2)^2 + (y + 2)^2 = (2 - 0)^2 + (-2 - 0)^2 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l536_53693


namespace NUMINAMATH_CALUDE_average_b_c_l536_53683

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 80) 
  (h2 : a - c = 200) : 
  (b + c) / 2 = -20 := by
sorry

end NUMINAMATH_CALUDE_average_b_c_l536_53683


namespace NUMINAMATH_CALUDE_sphere_surface_area_l536_53660

theorem sphere_surface_area (v : ℝ) (r : ℝ) (h : v = 72 * Real.pi) :
  4 * Real.pi * r^2 = 36 * Real.pi * (2^(2/3)) :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l536_53660


namespace NUMINAMATH_CALUDE_saturday_balls_count_l536_53600

/-- The number of golf balls Corey wants to find every weekend -/
def weekend_goal : ℕ := 48

/-- The number of golf balls Corey found on Sunday -/
def sunday_balls : ℕ := 18

/-- The number of additional golf balls Corey needs to reach his goal -/
def additional_balls_needed : ℕ := 14

/-- The number of golf balls Corey found on Saturday -/
def saturday_balls : ℕ := weekend_goal - sunday_balls - additional_balls_needed

theorem saturday_balls_count : saturday_balls = 16 := by
  sorry

end NUMINAMATH_CALUDE_saturday_balls_count_l536_53600


namespace NUMINAMATH_CALUDE_sum_of_digits_greatest_prime_divisor_16385_l536_53676

def greatest_prime_divisor (n : ℕ) : ℕ := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_greatest_prime_divisor_16385 :
  sum_of_digits (greatest_prime_divisor 16385) = 19 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_greatest_prime_divisor_16385_l536_53676


namespace NUMINAMATH_CALUDE_not_perfect_square_l536_53677

theorem not_perfect_square (n : ℕ) : ¬∃ (m : ℕ), n^6 + 3*n^5 - 5*n^4 - 15*n^3 + 4*n^2 + 12*n + 3 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l536_53677


namespace NUMINAMATH_CALUDE_original_ratio_l536_53690

theorem original_ratio (x y : ℕ) (h1 : y = 72) (h2 : (x + 6) / y = 1 / 3) : y / x = 4 := by
  sorry

end NUMINAMATH_CALUDE_original_ratio_l536_53690


namespace NUMINAMATH_CALUDE_average_speed_proof_l536_53668

/-- Proves that the average speed of a trip is 32 km/h given the specified conditions -/
theorem average_speed_proof (total_distance : ℝ) (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) :
  total_distance = 60 →
  distance1 = 30 →
  speed1 = 48 →
  distance2 = 30 →
  speed2 = 24 →
  (total_distance / ((distance1 / speed1) + (distance2 / speed2))) = 32 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_proof_l536_53668


namespace NUMINAMATH_CALUDE_student_comprehensive_score_l536_53672

/-- Represents the scores and weights for a science and technology innovation competition. -/
structure CompetitionScores where
  theoretical_knowledge : ℝ
  innovative_design : ℝ
  on_site_presentation : ℝ
  theoretical_weight : ℝ
  innovative_weight : ℝ
  on_site_weight : ℝ

/-- Calculates the comprehensive score for a given set of competition scores. -/
def comprehensive_score (scores : CompetitionScores) : ℝ :=
  scores.theoretical_knowledge * scores.theoretical_weight +
  scores.innovative_design * scores.innovative_weight +
  scores.on_site_presentation * scores.on_site_weight

/-- Theorem stating that the student's comprehensive score is 90 points. -/
theorem student_comprehensive_score :
  let scores : CompetitionScores := {
    theoretical_knowledge := 95,
    innovative_design := 88,
    on_site_presentation := 90,
    theoretical_weight := 0.2,
    innovative_weight := 0.5,
    on_site_weight := 0.3
  }
  comprehensive_score scores = 90 := by
  sorry


end NUMINAMATH_CALUDE_student_comprehensive_score_l536_53672


namespace NUMINAMATH_CALUDE_systematic_sampling_fourth_student_l536_53642

/-- Represents a systematic sampling of students. -/
structure SystematicSample where
  totalStudents : ℕ
  sampleSize : ℕ
  sampleInterval : ℕ
  firstStudent : ℕ

/-- Checks if a student number is in the sample. -/
def isInSample (s : SystematicSample) (studentNumber : ℕ) : Prop :=
  ∃ k : ℕ, studentNumber = s.firstStudent + k * s.sampleInterval ∧ 
           studentNumber ≤ s.totalStudents

theorem systematic_sampling_fourth_student 
  (s : SystematicSample)
  (h1 : s.totalStudents = 60)
  (h2 : s.sampleSize = 4)
  (h3 : s.firstStudent = 3)
  (h4 : isInSample s 33)
  (h5 : isInSample s 48) :
  isInSample s 18 := by
  sorry

#check systematic_sampling_fourth_student

end NUMINAMATH_CALUDE_systematic_sampling_fourth_student_l536_53642


namespace NUMINAMATH_CALUDE_largest_common_divisor_408_330_l536_53678

theorem largest_common_divisor_408_330 : Nat.gcd 408 330 = 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_408_330_l536_53678


namespace NUMINAMATH_CALUDE_chord_existence_l536_53657

-- Define the ellipse and line
def ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 16
def line (x y : ℝ) : Prop := y = x + 1

-- Theorem statement
theorem chord_existence :
  ∃ (length : ℝ), ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    line x₁ y₁ ∧ line x₂ y₂ ∧
    length = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) :=
by sorry

end NUMINAMATH_CALUDE_chord_existence_l536_53657
