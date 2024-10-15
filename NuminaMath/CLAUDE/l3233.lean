import Mathlib

namespace NUMINAMATH_CALUDE_parabola_translation_l3233_323312

/-- Given a parabola y = x^2 + bx + c that is translated 3 units right and 4 units down
    to become y = x^2 - 2x + 2, prove that b = 4 and c = 9 -/
theorem parabola_translation (b c : ℝ) : 
  (∀ x y : ℝ, y = x^2 + b*x + c ↔ y + 4 = (x - 3)^2 - 2*(x - 3) + 2) →
  b = 4 ∧ c = 9 := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3233_323312


namespace NUMINAMATH_CALUDE_justin_total_pages_justin_first_book_pages_justin_second_book_pages_l3233_323393

/-- Represents the reading schedule for a week -/
structure ReadingSchedule where
  firstBookDay1 : ℕ
  secondBookDay1 : ℕ
  firstBookIncrement : ℕ → ℕ
  secondBookIncrement : ℕ
  firstBookBreakDay : ℕ
  secondBookBreakDay : ℕ

/-- Calculates the total pages read for both books in a week -/
def totalPagesRead (schedule : ReadingSchedule) : ℕ := 
  let firstBookPages := schedule.firstBookDay1 + 
    (schedule.firstBookDay1 * 2) + 
    (schedule.firstBookDay1 * 2 + schedule.firstBookIncrement 3) +
    (schedule.firstBookDay1 * 2 + schedule.firstBookIncrement 4) +
    (schedule.firstBookDay1 * 2 + schedule.firstBookIncrement 5) +
    (schedule.firstBookDay1 * 2 + schedule.firstBookIncrement 6)
  let secondBookPages := schedule.secondBookDay1 + 
    (schedule.secondBookDay1 + schedule.secondBookIncrement) +
    (schedule.secondBookDay1 + 2 * schedule.secondBookIncrement) +
    (schedule.secondBookDay1 + 3 * schedule.secondBookIncrement) +
    (schedule.secondBookDay1 + 4 * schedule.secondBookIncrement) +
    (schedule.secondBookDay1 + 5 * schedule.secondBookIncrement)
  firstBookPages + secondBookPages

/-- Justin's reading schedule -/
def justinSchedule : ReadingSchedule := {
  firstBookDay1 := 10,
  secondBookDay1 := 15,
  firstBookIncrement := λ n => 5 * (n - 2),
  secondBookIncrement := 3,
  firstBookBreakDay := 7,
  secondBookBreakDay := 4
}

/-- Theorem stating that Justin reads 295 pages in total -/
theorem justin_total_pages : totalPagesRead justinSchedule = 295 := by
  sorry

/-- Theorem stating that Justin reads 160 pages of the first book -/
theorem justin_first_book_pages : 
  justinSchedule.firstBookDay1 + 
  (justinSchedule.firstBookDay1 * 2) + 
  (justinSchedule.firstBookDay1 * 2 + justinSchedule.firstBookIncrement 3) +
  (justinSchedule.firstBookDay1 * 2 + justinSchedule.firstBookIncrement 4) +
  (justinSchedule.firstBookDay1 * 2 + justinSchedule.firstBookIncrement 5) +
  (justinSchedule.firstBookDay1 * 2 + justinSchedule.firstBookIncrement 6) = 160 := by
  sorry

/-- Theorem stating that Justin reads 135 pages of the second book -/
theorem justin_second_book_pages :
  justinSchedule.secondBookDay1 + 
  (justinSchedule.secondBookDay1 + justinSchedule.secondBookIncrement) +
  (justinSchedule.secondBookDay1 + 2 * justinSchedule.secondBookIncrement) +
  (justinSchedule.secondBookDay1 + 3 * justinSchedule.secondBookIncrement) +
  (justinSchedule.secondBookDay1 + 4 * justinSchedule.secondBookIncrement) +
  (justinSchedule.secondBookDay1 + 5 * justinSchedule.secondBookIncrement) = 135 := by
  sorry

end NUMINAMATH_CALUDE_justin_total_pages_justin_first_book_pages_justin_second_book_pages_l3233_323393


namespace NUMINAMATH_CALUDE_chocolate_division_l3233_323335

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (multiply_factor : ℕ) 
  (h1 : total_chocolate = 60 / 7)
  (h2 : num_piles = 5)
  (h3 : multiply_factor = 3) :
  (total_chocolate / num_piles) * multiply_factor = 36 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l3233_323335


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l3233_323369

theorem intersection_point_of_lines (x y : ℚ) : 
  (5 * x - 2 * y = 4) ∧ (3 * x + 4 * y = 16) ↔ x = 24/13 ∧ y = 34/13 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l3233_323369


namespace NUMINAMATH_CALUDE_sophia_next_test_score_l3233_323300

def current_scores : List ℕ := [95, 85, 75, 65, 95]
def desired_increase : ℕ := 5

def minimum_required_score (scores : List ℕ) (increase : ℕ) : ℕ :=
  let current_sum := scores.sum
  let current_count := scores.length
  let current_average := current_sum / current_count
  let target_average := current_average + increase
  let total_count := current_count + 1
  target_average * total_count - current_sum

theorem sophia_next_test_score :
  minimum_required_score current_scores desired_increase = 113 := by
  sorry

end NUMINAMATH_CALUDE_sophia_next_test_score_l3233_323300


namespace NUMINAMATH_CALUDE_P_initial_investment_l3233_323347

/-- Represents the initial investment of P in rupees -/
def P_investment : ℕ := sorry

/-- Represents Q's investment in rupees -/
def Q_investment : ℕ := 9000

/-- Represents the number of months P's investment was active -/
def P_months : ℕ := 12

/-- Represents the number of months Q's investment was active -/
def Q_months : ℕ := 8

/-- Represents P's share in the profit ratio -/
def P_share : ℕ := 2

/-- Represents Q's share in the profit ratio -/
def Q_share : ℕ := 3

/-- Theorem stating that P's initial investment is 4000 rupees -/
theorem P_initial_investment :
  (P_investment * P_months) * Q_share = (Q_investment * Q_months) * P_share ∧
  P_investment = 4000 := by
  sorry

end NUMINAMATH_CALUDE_P_initial_investment_l3233_323347


namespace NUMINAMATH_CALUDE_middle_number_proof_l3233_323316

theorem middle_number_proof (a b c : ℕ) 
  (h_order : a < b ∧ b < c)
  (h_sum1 : a + b = 15)
  (h_sum2 : a + c = 20)
  (h_sum3 : b + c = 25) :
  b = 10 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l3233_323316


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3233_323346

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_first_term : a 1 = 1) 
  (h_sum : a 3 + a 5 = 14) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3233_323346


namespace NUMINAMATH_CALUDE_distinct_integer_roots_l3233_323308

theorem distinct_integer_roots (a : ℝ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 + 2*a*x = 8*a ∧ y^2 + 2*a*y = 8*a) ↔ 
  a ∈ ({4.5, 1, -12.5, -9} : Set ℝ) :=
by sorry

end NUMINAMATH_CALUDE_distinct_integer_roots_l3233_323308


namespace NUMINAMATH_CALUDE_car_journey_time_l3233_323329

/-- Proves that given a car traveling 210 km in 7 hours for the forward journey,
    and increasing its speed by 12 km/hr for the return journey,
    the time taken for the return journey is 5 hours. -/
theorem car_journey_time (distance : ℝ) (forward_time : ℝ) (speed_increase : ℝ) :
  distance = 210 →
  forward_time = 7 →
  speed_increase = 12 →
  (distance / (distance / forward_time + speed_increase)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_car_journey_time_l3233_323329


namespace NUMINAMATH_CALUDE_min_distance_vectors_l3233_323350

/-- Given planar vectors a and b with an angle of 120° between them and a dot product of -1,
    the minimum value of |a - b| is √6. -/
theorem min_distance_vectors (a b : ℝ × ℝ) : 
  (Real.cos (120 * π / 180) = -1/2) →
  (a.1 * b.1 + a.2 * b.2 = -1) →
  (∀ c d : ℝ × ℝ, c.1 * d.1 + c.2 * d.2 = -1 → 
    Real.sqrt ((c.1 - d.1)^2 + (c.2 - d.2)^2) ≥ Real.sqrt 6) ∧
  (∃ e f : ℝ × ℝ, e.1 * f.1 + e.2 * f.2 = -1 ∧ 
    Real.sqrt ((e.1 - f.1)^2 + (e.2 - f.2)^2) = Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_vectors_l3233_323350


namespace NUMINAMATH_CALUDE_no_real_solutions_l3233_323383

theorem no_real_solutions :
  ∀ x y : ℝ, x^2 + 2*y^2 - 6*x - 8*y + 21 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3233_323383


namespace NUMINAMATH_CALUDE_at_least_three_to_six_colorings_l3233_323342

/-- Represents the colors that can be used to color the hexagons -/
inductive Color
| Red
| Yellow
| Green
| Blue

/-- Represents a hexagon in the figure -/
structure Hexagon where
  color : Color

/-- Represents the central hexagon and its six adjacent hexagons -/
structure CentralHexagonWithAdjacent where
  center : Hexagon
  adjacent : Fin 6 → Hexagon

/-- Two hexagons are considered adjacent if they share a side -/
def areAdjacent (h1 h2 : Hexagon) : Prop := sorry

/-- A coloring is valid if no two adjacent hexagons have the same color -/
def isValidColoring (config : CentralHexagonWithAdjacent) : Prop :=
  config.center.color = Color.Red ∧
  ∀ i j : Fin 6, i ≠ j →
    config.adjacent i ≠ config.adjacent j ∧
    config.adjacent i ≠ config.center ∧
    config.adjacent j ≠ config.center

/-- The number of valid colorings for the central hexagon and its adjacent hexagons -/
def numValidColorings : ℕ := sorry

theorem at_least_three_to_six_colorings :
  numValidColorings ≥ 3^6 := by sorry

end NUMINAMATH_CALUDE_at_least_three_to_six_colorings_l3233_323342


namespace NUMINAMATH_CALUDE_hakimi_age_l3233_323360

/-- Given three friends Hakimi, Jared, and Molly, this theorem proves Hakimi's age
    based on the given conditions. -/
theorem hakimi_age (hakimi_age jared_age molly_age : ℕ) : 
  (hakimi_age + jared_age + molly_age) / 3 = 40 →  -- Average age is 40
  jared_age = hakimi_age + 10 →  -- Jared is 10 years older than Hakimi
  molly_age = 30 →  -- Molly's age is 30
  hakimi_age = 40 :=  -- Hakimi's age is 40
by
  sorry

end NUMINAMATH_CALUDE_hakimi_age_l3233_323360


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l3233_323373

theorem gcd_lcm_product (a b : ℕ) (ha : a = 90) (hb : b = 135) :
  (Nat.gcd a b) * (Nat.lcm a b) = 12150 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l3233_323373


namespace NUMINAMATH_CALUDE_inverse_f_at_negative_seven_sixtyfourth_l3233_323392

noncomputable def f (x : ℝ) : ℝ := (x^7 - 1) / 4

theorem inverse_f_at_negative_seven_sixtyfourth :
  f⁻¹ (-7/64) = (9/16)^(1/7) :=
by sorry

end NUMINAMATH_CALUDE_inverse_f_at_negative_seven_sixtyfourth_l3233_323392


namespace NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l3233_323388

theorem factorization_of_difference_of_squares (a b : ℝ) :
  3 * a^2 - 3 * b^2 = 3 * (a + b) * (a - b) := by sorry

end NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l3233_323388


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l3233_323340

/-- Given vectors a and b in ℝ², if k*a + b is parallel to a + 3*b, then k = 1/3 -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-3, 2))
    (h_parallel : ∃ (t : ℝ), t ≠ 0 ∧ k • a + b = t • (a + 3 • b)) :
  k = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l3233_323340


namespace NUMINAMATH_CALUDE_expression_value_l3233_323319

theorem expression_value : -20 + 8 * (5^2 - 3) = 156 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3233_323319


namespace NUMINAMATH_CALUDE_three_correct_probability_l3233_323358

/-- The probability of exactly 3 out of 5 packages being delivered to their correct houses -/
def probability_three_correct (n : ℕ) : ℚ :=
  if n = 5 then 1 / 12 else 0

/-- Theorem stating the probability of exactly 3 out of 5 packages being delivered correctly -/
theorem three_correct_probability :
  probability_three_correct 5 = 1 / 12 :=
by sorry

end NUMINAMATH_CALUDE_three_correct_probability_l3233_323358


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3233_323367

theorem sufficient_not_necessary :
  (∃ x : ℝ, x < -1 ∧ x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ ¬(x < -1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3233_323367


namespace NUMINAMATH_CALUDE_P_n_formula_S_3_formula_geometric_sequence_condition_l3233_323328

-- Define the sequence and expansion operation
def Sequence := List ℝ

def expand_by_sum (s : Sequence) : Sequence :=
  match s with
  | [] => []
  | [x] => [x]
  | x::y::rest => x :: (x+y) :: expand_by_sum (y::rest)

-- Define P_n and S_n
def P (n : ℕ) (a b c : ℝ) : ℕ := 
  (expand_by_sum^[n] [a, b, c]).length

def S (n : ℕ) (a b c : ℝ) : ℝ := 
  (expand_by_sum^[n] [a, b, c]).sum

-- Theorem statements
theorem P_n_formula (n : ℕ) (a b c : ℝ) : 
  P n a b c = 2^(n+1) + 1 := by sorry

theorem S_3_formula (a b c : ℝ) :
  S 3 a b c = 14*a + 27*b + 14*c := by sorry

theorem geometric_sequence_condition (a b c : ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, S (n+1) a b c = r * S n a b c) ↔ 
  ((a + c = 0 ∧ b ≠ 0) ∨ (2*b + a + c = 0 ∧ b ≠ 0)) := by sorry

end NUMINAMATH_CALUDE_P_n_formula_S_3_formula_geometric_sequence_condition_l3233_323328


namespace NUMINAMATH_CALUDE_expand_expression_l3233_323387

theorem expand_expression (x y : ℝ) :
  -2 * (4 * x^3 - 3 * x * y + 5) = -8 * x^3 + 6 * x * y - 10 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3233_323387


namespace NUMINAMATH_CALUDE_inequality_solution_l3233_323354

theorem inequality_solution (x y : ℝ) :
  (y^2 - 4*x*y + 4*x^2 < x^2) ↔ (x < y ∧ y < 3*x ∧ x > 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3233_323354


namespace NUMINAMATH_CALUDE_injective_implies_different_outputs_injective_implies_at_most_one_preimage_l3233_323376

-- Define the function f from set A to set B
variable {A B : Type*} (f : A → B)

-- Define injectivity
def Injective (f : A → B) : Prop :=
  ∀ x₁ x₂ : A, f x₁ = f x₂ → x₁ = x₂

-- Theorem 1: If f is injective and x₁ ≠ x₂, then f(x₁) ≠ f(x₂)
theorem injective_implies_different_outputs
  (hf : Injective f) :
  ∀ x₁ x₂ : A, x₁ ≠ x₂ → f x₁ ≠ f x₂ := by
sorry

-- Theorem 2: If f is injective, then for any b ∈ B, there is at most one pre-image in A
theorem injective_implies_at_most_one_preimage
  (hf : Injective f) :
  ∀ b : B, ∃! x : A, f x = b := by
sorry

end NUMINAMATH_CALUDE_injective_implies_different_outputs_injective_implies_at_most_one_preimage_l3233_323376


namespace NUMINAMATH_CALUDE_deposit_calculation_l3233_323318

theorem deposit_calculation (total_price : ℝ) (deposit_percentage : ℝ) (remaining_amount : ℝ) 
  (h1 : deposit_percentage = 0.1)
  (h2 : remaining_amount = 720)
  (h3 : total_price * (1 - deposit_percentage) = remaining_amount) :
  total_price * deposit_percentage = 80 := by
  sorry

end NUMINAMATH_CALUDE_deposit_calculation_l3233_323318


namespace NUMINAMATH_CALUDE_total_books_read_l3233_323396

def summer_reading (june july august : ℕ) : Prop :=
  june = 8 ∧ july = 2 * june ∧ august = july - 3

theorem total_books_read (june july august : ℕ) 
  (h : summer_reading june july august) : june + july + august = 37 := by
  sorry

end NUMINAMATH_CALUDE_total_books_read_l3233_323396


namespace NUMINAMATH_CALUDE_divisible_by_six_ratio_l3233_323334

theorem divisible_by_six_ratio (n : ℕ) : n = 120 →
  (Finset.filter (fun x => x % 6 = 0) (Finset.range (n + 1))).card / (n + 1 : ℚ) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_ratio_l3233_323334


namespace NUMINAMATH_CALUDE_distance_between_lines_l3233_323368

/-- Represents a circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  /-- Radius of the circle -/
  radius : ℝ
  /-- Distance between adjacent parallel lines -/
  line_distance : ℝ
  /-- Length of the first chord -/
  chord1_length : ℝ
  /-- Length of the second chord -/
  chord2_length : ℝ
  /-- Length of the third chord -/
  chord3_length : ℝ
  /-- The first and second chords have equal length -/
  chord1_eq_chord2 : chord1_length = chord2_length
  /-- The first chord has length 40 -/
  chord1_is_40 : chord1_length = 40
  /-- The third chord has length 36 -/
  chord3_is_36 : chord3_length = 36

/-- Theorem stating that the distance between adjacent parallel lines is 1.5 -/
theorem distance_between_lines (c : CircleWithParallelLines) : c.line_distance = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_lines_l3233_323368


namespace NUMINAMATH_CALUDE_largest_quantity_l3233_323386

def D : ℚ := 2008/2007 + 2008/2009
def E : ℚ := 2008/2009 + 2010/2009
def F : ℚ := 2009/2008 + 2009/2010 - 1/2009

theorem largest_quantity : D > E ∧ D > F := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l3233_323386


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l3233_323356

theorem gcd_from_lcm_and_ratio (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : X * 5 = Y * 2) : 
  Nat.gcd X Y = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l3233_323356


namespace NUMINAMATH_CALUDE_fraction_relation_l3233_323314

theorem fraction_relation (x y z w : ℚ) 
  (h1 : x / y = 7)
  (h2 : z / y = 5)
  (h3 : z / w = 3 / 4) :
  w / x = 20 / 21 := by
sorry

end NUMINAMATH_CALUDE_fraction_relation_l3233_323314


namespace NUMINAMATH_CALUDE_chocolate_gain_percent_l3233_323397

theorem chocolate_gain_percent (C S : ℝ) (h : 165 * C = 150 * S) : 
  (S - C) / C * 100 = 10 := by sorry

end NUMINAMATH_CALUDE_chocolate_gain_percent_l3233_323397


namespace NUMINAMATH_CALUDE_decreasing_function_odd_product_l3233_323378

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Statement 1
theorem decreasing_function (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0) :
  ∀ x y : ℝ, x < y → f y < f x :=
sorry

-- Define an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Statement 3
theorem odd_product (h : is_odd f) :
  is_odd (λ x => f x * f (|x|)) :=
sorry

end NUMINAMATH_CALUDE_decreasing_function_odd_product_l3233_323378


namespace NUMINAMATH_CALUDE_milk_production_l3233_323348

/-- Milk production calculation -/
theorem milk_production
  (m n p x q r : ℝ)
  (h1 : m > 0)
  (h2 : p > 0)
  (h3 : 0 ≤ x)
  (h4 : x ≤ m)
  : (q * r * (m + 0.2 * x) * n) / (m * p) =
    q * r * ((m - x) * (n / (m * p)) + x * (1.2 * n / (m * p))) :=
by sorry

end NUMINAMATH_CALUDE_milk_production_l3233_323348


namespace NUMINAMATH_CALUDE_gumball_ratio_l3233_323390

/-- Represents the gumball problem scenario -/
structure GumballScenario where
  alicia_gumballs : ℕ
  pedro_multiplier : ℚ
  remaining_gumballs : ℕ

/-- The specific scenario given in the problem -/
def problem_scenario : GumballScenario :=
  { alicia_gumballs := 20
  , pedro_multiplier := 3
  , remaining_gumballs := 60 }

/-- Calculates the total number of gumballs initially in the bowl -/
def total_gumballs (s : GumballScenario) : ℚ :=
  s.alicia_gumballs * (2 + s.pedro_multiplier)

/-- Calculates Pedro's additional gumballs -/
def pedro_additional_gumballs (s : GumballScenario) : ℚ :=
  s.alicia_gumballs * s.pedro_multiplier

/-- The main theorem to prove -/
theorem gumball_ratio (s : GumballScenario) :
  s.alicia_gumballs = 20 →
  s.remaining_gumballs = 60 →
  (total_gumballs s * (3/5) : ℚ) = s.remaining_gumballs →
  (pedro_additional_gumballs s) / s.alicia_gumballs = 3 :=
by sorry

#check gumball_ratio problem_scenario

end NUMINAMATH_CALUDE_gumball_ratio_l3233_323390


namespace NUMINAMATH_CALUDE_mrs_hilt_fountain_trips_l3233_323363

/-- Calculates the total distance walked to a water fountain given the one-way distance and number of trips -/
def total_distance_walked (one_way_distance : ℕ) (num_trips : ℕ) : ℕ :=
  2 * one_way_distance * num_trips

/-- Proves that given a distance of 30 feet from desk to fountain and 4 trips to the fountain, the total distance walked is 240 feet -/
theorem mrs_hilt_fountain_trips :
  total_distance_walked 30 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_fountain_trips_l3233_323363


namespace NUMINAMATH_CALUDE_megan_carrots_l3233_323343

theorem megan_carrots (initial_carrots thrown_out_carrots next_day_carrots : ℕ) :
  initial_carrots ≥ thrown_out_carrots →
  initial_carrots - thrown_out_carrots + next_day_carrots =
    initial_carrots + next_day_carrots - thrown_out_carrots :=
by sorry

end NUMINAMATH_CALUDE_megan_carrots_l3233_323343


namespace NUMINAMATH_CALUDE_p_adic_valuation_factorial_formula_l3233_323384

/-- The sum of digits of n in base p -/
def sum_of_digits (n : ℕ) (p : ℕ) : ℕ :=
  sorry

/-- The p-adic valuation of n! -/
def p_adic_valuation_factorial (p : ℕ) (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The p-adic valuation of n! equals (n - s(n)) / (p - 1) -/
theorem p_adic_valuation_factorial_formula (p : ℕ) (n : ℕ) (hp : Prime p) (hn : n > 0) :
  p_adic_valuation_factorial p n = (n - sum_of_digits n p) / (p - 1) :=
sorry

end NUMINAMATH_CALUDE_p_adic_valuation_factorial_formula_l3233_323384


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3233_323332

theorem trigonometric_identities (α : Real) (h_acute : 0 < α ∧ α < Real.pi / 2) 
  (h_sin : Real.sin α = 3 / 5) : 
  (Real.cos α = 4 / 5) ∧ 
  (Real.cos (α + Real.pi / 6) = (4 * Real.sqrt 3 - 3) / 10) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3233_323332


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3233_323336

theorem arithmetic_geometric_mean_inequality {a b : ℝ} (ha : a ≥ 0) (hb : b ≥ 0) :
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧ ((a + b) / 2 = Real.sqrt (a * b) ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3233_323336


namespace NUMINAMATH_CALUDE_remainder_s_1024_mod_1000_l3233_323357

-- Define the polynomial q(x)
def q (x : ℤ) : ℤ := (x^1025 - 1) / (x - 1)

-- Define the divisor polynomial
def divisor (x : ℤ) : ℤ := x^6 + x^5 + 3*x^4 + x^3 + x^2 + x + 1

-- Define s(x) as the polynomial remainder
noncomputable def s (x : ℤ) : ℤ := q x % divisor x

-- Theorem statement
theorem remainder_s_1024_mod_1000 : |s 1024| % 1000 = 824 := by sorry

end NUMINAMATH_CALUDE_remainder_s_1024_mod_1000_l3233_323357


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3233_323304

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (Real.sin α + Real.cos α) = -1) : 
  Real.tan α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3233_323304


namespace NUMINAMATH_CALUDE_faster_train_speed_l3233_323366

/-- Proves the speed of the faster train given the conditions of the problem -/
theorem faster_train_speed
  (speed_diff : ℝ)
  (faster_train_length : ℝ)
  (crossing_time : ℝ)
  (h1 : speed_diff = 36)
  (h2 : faster_train_length = 120)
  (h3 : crossing_time = 12)
  : ∃ (faster_speed : ℝ), faster_speed = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_l3233_323366


namespace NUMINAMATH_CALUDE_square_difference_theorem_l3233_323371

theorem square_difference_theorem : (13 + 8)^2 - (13 - 8)^2 = 416 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l3233_323371


namespace NUMINAMATH_CALUDE_minimum_value_problems_l3233_323333

theorem minimum_value_problems :
  (∀ x > 0, x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1)) ∧
  (∀ m > 0, (m^2 + 5*m + 12) / m ≥ 4 * Real.sqrt 3 + 5) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_problems_l3233_323333


namespace NUMINAMATH_CALUDE_a_3_equals_1_l3233_323337

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n - 3

theorem a_3_equals_1 (a : ℕ → ℤ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 1 = 7) : 
  a 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_3_equals_1_l3233_323337


namespace NUMINAMATH_CALUDE_equation_solution_exists_l3233_323399

theorem equation_solution_exists : ∃ (x y : ℕ), x^9 = 2013 * y^10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l3233_323399


namespace NUMINAMATH_CALUDE_common_root_of_quadratics_l3233_323349

theorem common_root_of_quadratics (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ b * x^2 + c * x + a = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_common_root_of_quadratics_l3233_323349


namespace NUMINAMATH_CALUDE_money_division_l3233_323309

theorem money_division (a b c : ℝ) : 
  a = (1/3) * (b + c) →
  b = (2/7) * (a + c) →
  a = b + 30 →
  a + b + c = 280 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l3233_323309


namespace NUMINAMATH_CALUDE_marks_ratio_l3233_323317

theorem marks_ratio (P S W : ℚ) 
  (h1 : P / S = 4 / 5) 
  (h2 : S / W = 5 / 2) : 
  P / W = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_marks_ratio_l3233_323317


namespace NUMINAMATH_CALUDE_sally_quarters_l3233_323306

/-- The number of quarters Sally has after her purchases -/
def remaining_quarters (initial : ℕ) (purchase1 : ℕ) (purchase2 : ℕ) : ℕ :=
  initial - purchase1 - purchase2

/-- Theorem stating that Sally has 150 quarters left after her purchases -/
theorem sally_quarters : remaining_quarters 760 418 192 = 150 := by
  sorry

end NUMINAMATH_CALUDE_sally_quarters_l3233_323306


namespace NUMINAMATH_CALUDE_temperature_height_relationship_l3233_323379

/-- The temperature-height relationship function -/
def t (h : ℝ) : ℝ := 20 - 6 * h

/-- The set of given data points -/
def data_points : List (ℝ × ℝ) := [(0, 20), (1, 14), (2, 8), (3, 2), (4, -4)]

/-- Theorem stating that the function t accurately describes the temperature-height relationship -/
theorem temperature_height_relationship :
  ∀ (point : ℝ × ℝ), point ∈ data_points → t point.1 = point.2 := by
  sorry

end NUMINAMATH_CALUDE_temperature_height_relationship_l3233_323379


namespace NUMINAMATH_CALUDE_encoded_xyz_value_l3233_323324

/-- Represents a digit in the base-6 encoding system -/
inductive Digit : Type
| U | V | W | X | Y | Z

/-- Converts a Digit to its corresponding natural number value -/
def digit_to_nat (d : Digit) : ℕ :=
  match d with
  | Digit.U => 0
  | Digit.V => 1
  | Digit.W => 2
  | Digit.X => 3
  | Digit.Y => 4
  | Digit.Z => 5

/-- Represents a three-digit number in the base-6 encoding system -/
structure EncodedNumber :=
  (hundreds : Digit)
  (tens : Digit)
  (ones : Digit)

/-- Converts an EncodedNumber to its base-10 value -/
def to_base_10 (n : EncodedNumber) : ℕ :=
  36 * (digit_to_nat n.hundreds) + 6 * (digit_to_nat n.tens) + (digit_to_nat n.ones)

/-- The theorem to be proved -/
theorem encoded_xyz_value :
  ∀ (v x y z : Digit),
    v ≠ x → v ≠ y → v ≠ z → x ≠ y → x ≠ z → y ≠ z →
    to_base_10 (EncodedNumber.mk v x z) + 1 = to_base_10 (EncodedNumber.mk v x y) →
    to_base_10 (EncodedNumber.mk v x y) + 1 = to_base_10 (EncodedNumber.mk v v y) →
    to_base_10 (EncodedNumber.mk x y z) = 184 :=
sorry

end NUMINAMATH_CALUDE_encoded_xyz_value_l3233_323324


namespace NUMINAMATH_CALUDE_zoey_reading_schedule_l3233_323325

def days_to_read (n : ℕ) : ℕ := 2 * n - 1

def total_days (num_books : ℕ) : ℕ := num_books^2

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed - 1) % 7 + 1

theorem zoey_reading_schedule :
  let num_books := 18
  let start_day := 1  -- Monday
  let total_reading_days := total_days num_books
  day_of_week start_day total_reading_days = 3  -- Wednesday
:= by sorry

end NUMINAMATH_CALUDE_zoey_reading_schedule_l3233_323325


namespace NUMINAMATH_CALUDE_expected_draws_no_ugly_l3233_323372

def bag_total : ℕ := 20
def blue_marbles : ℕ := 9
def ugly_marbles : ℕ := 10
def special_marbles : ℕ := 1

def prob_blue : ℚ := blue_marbles / bag_total
def prob_special : ℚ := special_marbles / bag_total

theorem expected_draws_no_ugly : 
  let p := prob_blue
  let q := prob_special
  (∑' k : ℕ, k * (1 / (1 - p)) * p^(k-1) * q) = 20 / 11 :=
sorry

end NUMINAMATH_CALUDE_expected_draws_no_ugly_l3233_323372


namespace NUMINAMATH_CALUDE_layla_nahima_score_difference_l3233_323359

theorem layla_nahima_score_difference :
  ∀ (total_points layla_score nahima_score : ℕ),
    total_points = 112 →
    layla_score = 70 →
    total_points = layla_score + nahima_score →
    layla_score - nahima_score = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_layla_nahima_score_difference_l3233_323359


namespace NUMINAMATH_CALUDE_bankers_discount_problem_l3233_323351

/-- Proves that given a sum S where the banker's discount is 18 and the true discount is 15, S equals 75 -/
theorem bankers_discount_problem (S : ℝ) 
  (h1 : 18 = 15 + (15^2 / S)) : S = 75 := by
  sorry

end NUMINAMATH_CALUDE_bankers_discount_problem_l3233_323351


namespace NUMINAMATH_CALUDE_factorial_square_root_squared_l3233_323344

theorem factorial_square_root_squared : (Real.sqrt (Nat.factorial 5 * Nat.factorial 4))^2 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_squared_l3233_323344


namespace NUMINAMATH_CALUDE_compound_ratio_example_l3233_323327

def ratio (a b : ℤ) := (a, b)

def compound_ratio (r1 r2 r3 : ℤ × ℤ) : ℤ × ℤ :=
  let (a1, b1) := r1
  let (a2, b2) := r2
  let (a3, b3) := r3
  (a1 * a2 * a3, b1 * b2 * b3)

def simplify_ratio (r : ℤ × ℤ) : ℤ × ℤ :=
  let (a, b) := r
  let gcd := Int.gcd a b
  (a / gcd, b / gcd)

theorem compound_ratio_example : 
  simplify_ratio (compound_ratio (ratio 2 3) (ratio 6 11) (ratio 11 2)) = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_compound_ratio_example_l3233_323327


namespace NUMINAMATH_CALUDE_wise_stock_price_l3233_323391

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

end NUMINAMATH_CALUDE_wise_stock_price_l3233_323391


namespace NUMINAMATH_CALUDE_product_of_repeating_third_and_nine_l3233_323301

/-- The repeating decimal 0.333... -/
def repeating_third : ℚ := 1/3

theorem product_of_repeating_third_and_nine :
  repeating_third * 9 = 3 := by sorry

end NUMINAMATH_CALUDE_product_of_repeating_third_and_nine_l3233_323301


namespace NUMINAMATH_CALUDE_rectangle_difference_l3233_323382

/-- A rectangle with given perimeter and area -/
structure Rectangle where
  length : ℝ
  breadth : ℝ
  perimeter_eq : length + breadth = 93
  area_eq : length * breadth = 2030

/-- The difference between length and breadth of a rectangle with perimeter 186m and area 2030m² is 23m -/
theorem rectangle_difference (rect : Rectangle) : rect.length - rect.breadth = 23 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_difference_l3233_323382


namespace NUMINAMATH_CALUDE_circles_common_points_l3233_323385

/-- Two circles with radii 2 and 3 have common points if and only if 
    the distance between their centers is between 1 and 5 (inclusive). -/
theorem circles_common_points (d : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 4 ∧ (x - d)^2 + y^2 = 9) ↔ 1 ≤ d ∧ d ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_circles_common_points_l3233_323385


namespace NUMINAMATH_CALUDE_train_car_speed_ratio_l3233_323353

/-- Given a bus that travels 320 km in 5 hours, and its speed is 4/5 of the train's speed,
    and a car that travels 525 km in 7 hours, prove that the ratio of the train's speed
    to the car's speed is 16:15 -/
theorem train_car_speed_ratio :
  ∀ (bus_speed train_speed car_speed : ℝ),
    bus_speed = 320 / 5 →
    bus_speed = (4 / 5) * train_speed →
    car_speed = 525 / 7 →
    train_speed / car_speed = 16 / 15 := by
  sorry

end NUMINAMATH_CALUDE_train_car_speed_ratio_l3233_323353


namespace NUMINAMATH_CALUDE_permutation_difference_divisibility_l3233_323395

/-- For any integer n > 2 and any two permutations of {0, 1, ..., n-1},
    there exist distinct indices i and j such that n divides (aᵢ * bᵢ - aⱼ * bⱼ). -/
theorem permutation_difference_divisibility (n : ℕ) (hn : n > 2)
  (a b : Fin n → Fin n) (ha : Function.Bijective a) (hb : Function.Bijective b) :
  ∃ (i j : Fin n), i ≠ j ∧ (n : ℤ) ∣ (a i * b i - a j * b j) :=
sorry

end NUMINAMATH_CALUDE_permutation_difference_divisibility_l3233_323395


namespace NUMINAMATH_CALUDE_sqrt_2_4_3_6_5_2_l3233_323320

theorem sqrt_2_4_3_6_5_2 : Real.sqrt (2^4 * 3^6 * 5^2) = 540 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_4_3_6_5_2_l3233_323320


namespace NUMINAMATH_CALUDE_min_difference_gcd3_lcm135_l3233_323311

def min_difference_with_gcd_lcm : ℕ → ℕ → ℕ → ℕ → ℕ := sorry

theorem min_difference_gcd3_lcm135 :
  min_difference_with_gcd_lcm 3 135 = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_difference_gcd3_lcm135_l3233_323311


namespace NUMINAMATH_CALUDE_cities_under_50k_l3233_323322

/-- City population distribution -/
structure CityDistribution where
  small : ℝ  -- Percentage of cities with fewer than 5,000 residents
  medium : ℝ  -- Percentage of cities with 5,000 to 49,999 residents
  large : ℝ  -- Percentage of cities with 50,000 or more residents

/-- The given city distribution -/
def givenDistribution : CityDistribution where
  small := 20
  medium := 35
  large := 45

/-- Theorem: The percentage of cities with fewer than 50,000 residents is 55% -/
theorem cities_under_50k (d : CityDistribution) 
  (h1 : d.small = 20) 
  (h2 : d.medium = 35) 
  (h3 : d.large = 45) : 
  d.small + d.medium = 55 := by
  sorry

#check cities_under_50k

end NUMINAMATH_CALUDE_cities_under_50k_l3233_323322


namespace NUMINAMATH_CALUDE_existence_of_function_l3233_323310

theorem existence_of_function (a : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x y : ℝ, x + f y = a * (y + f x)) ↔ (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_function_l3233_323310


namespace NUMINAMATH_CALUDE_hyperbola_equation_from_focus_and_midpoint_l3233_323365

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

theorem hyperbola_equation_from_focus_and_midpoint 
  (h : Hyperbola)
  (focus : Point)
  (midpoint : Point)
  (h_focus : focus.x = -2 ∧ focus.y = 0)
  (h_midpoint : midpoint.x = -3 ∧ midpoint.y = -1)
  (h_intersect : ∃ (A B : Point), 
    hyperbola_equation h A ∧ 
    hyperbola_equation h B ∧
    (A.x + B.x) / 2 = midpoint.x ∧
    (A.y + B.y) / 2 = midpoint.y) :
  h.a^2 = 3 ∧ h.b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_from_focus_and_midpoint_l3233_323365


namespace NUMINAMATH_CALUDE_gcd_of_lcm_and_ratio_l3233_323394

theorem gcd_of_lcm_and_ratio (A B : ℕ+) : 
  Nat.lcm A B = 180 → 
  A.val * 6 = B.val * 5 → 
  Nat.gcd A B = 6 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_lcm_and_ratio_l3233_323394


namespace NUMINAMATH_CALUDE_emily_lives_emily_final_lives_l3233_323339

/-- Calculates the final number of lives in Emily's video game. -/
theorem emily_lives (initial : ℕ) (lost : ℕ) (gained : ℕ) :
  initial ≥ lost →
  initial - lost + gained = initial + gained - lost :=
by
  sorry

/-- Proves that Emily ends up with 41 lives. -/
theorem emily_final_lives : 
  let initial : ℕ := 42
  let lost : ℕ := 25
  let gained : ℕ := 24
  initial ≥ lost →
  initial - lost + gained = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_emily_lives_emily_final_lives_l3233_323339


namespace NUMINAMATH_CALUDE_expression_value_at_three_l3233_323307

theorem expression_value_at_three :
  let x : ℕ := 3
  x + x * (x ^ x) + x ^ 3 = 111 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l3233_323307


namespace NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l3233_323303

theorem power_of_eight_sum_equals_power_of_two : ∃ x : ℕ, 8^4 + 8^4 + 8^4 = 2^x ∧ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l3233_323303


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l3233_323321

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively, 
    where a/c = b/d = 3/5, the ratio of the area of Rectangle A to the area 
    of Rectangle B is 9:25. -/
theorem rectangle_area_ratio 
  (a b c d : ℝ) 
  (h1 : a / c = 3 / 5) 
  (h2 : b / d = 3 / 5) :
  (a * b) / (c * d) = 9 / 25 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_ratio_l3233_323321


namespace NUMINAMATH_CALUDE_trig_identity_l3233_323345

theorem trig_identity (θ : Real) (h : Real.tan (θ - Real.pi) = 2) :
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3233_323345


namespace NUMINAMATH_CALUDE_endpoint_sum_coordinates_endpoint_sum_coordinates_proof_l3233_323380

/-- Given a line segment with one endpoint (6, 2) and midpoint (3, 7),
    the sum of coordinates of the other endpoint is 12. -/
theorem endpoint_sum_coordinates : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop :=
  fun endpoint1 midpoint endpoint2 =>
    endpoint1 = (6, 2) ∧
    midpoint = (3, 7) ∧
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) →
    endpoint2.1 + endpoint2.2 = 12
    
#check endpoint_sum_coordinates

theorem endpoint_sum_coordinates_proof : 
  ∃ (endpoint2 : ℝ × ℝ), endpoint_sum_coordinates (6, 2) (3, 7) endpoint2 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_coordinates_endpoint_sum_coordinates_proof_l3233_323380


namespace NUMINAMATH_CALUDE_equation_solution_l3233_323370

theorem equation_solution (x : ℝ) : 
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - y^2 - 4.5 = 0) ↔ x = 3/2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3233_323370


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3233_323364

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (x : ℝ), x^2 + y^2 = 4 ∧ x = -2) →
  (b / a = Real.sqrt 3) →
  (∀ (x y : ℝ), x^2 - y^2 / 3 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3233_323364


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3233_323313

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (∀ a b, (a - b) * a^2 < 0 → a < b) ∧ 
  (∃ a b, a < b ∧ (a - b) * a^2 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3233_323313


namespace NUMINAMATH_CALUDE_circle_area_tripled_l3233_323331

theorem circle_area_tripled (r n : ℝ) : 
  (π * (r + n)^2 = 3 * π * r^2) → r = n/2 * (Real.sqrt 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l3233_323331


namespace NUMINAMATH_CALUDE_sum_of_squares_l3233_323355

theorem sum_of_squares (a b c d e f : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (h1 : b * c * d * e * f / a = 1 / 2)
  (h2 : a * c * d * e * f / b = 1 / 4)
  (h3 : a * b * d * e * f / c = 1 / 8)
  (h4 : a * b * c * e * f / d = 2)
  (h5 : a * b * c * d * f / e = 4)
  (h6 : a * b * c * d * e / f = 8) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 119 / 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3233_323355


namespace NUMINAMATH_CALUDE_ratio_equality_l3233_323377

theorem ratio_equality : ∃ x : ℚ, (x / (2/5)) = ((3/7) / (6/5)) ∧ x = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3233_323377


namespace NUMINAMATH_CALUDE_prob_diff_tens_digits_l3233_323389

/-- The probability of selecting 6 different integers from 10 to 59 with different tens digits -/
theorem prob_diff_tens_digits : ℝ := by
  -- Define the range of integers
  let range : Set ℕ := {n : ℕ | 10 ≤ n ∧ n ≤ 59}

  -- Define the number of integers to be selected
  let k : ℕ := 6

  -- Define the function that returns the tens digit of a number
  let tens_digit (n : ℕ) : ℕ := n / 10

  -- Define the probability
  let prob : ℝ := (5 * 10 * 9 * 10^4 : ℝ) / (Nat.choose 50 6 : ℝ)

  -- State that the probability is equal to 1500000/5296900
  have h : prob = 1500000 / 5296900 := by sorry

  -- Return the probability
  exact prob

end NUMINAMATH_CALUDE_prob_diff_tens_digits_l3233_323389


namespace NUMINAMATH_CALUDE_inequality_proof_l3233_323341

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + a * c ≤ 1 / 3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3233_323341


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3233_323326

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  bridge_length = 140 →
  crossing_time = 23.998080153587715 →
  ∃ (speed : ℝ), abs (speed - 36) < 0.1 ∧ speed = (train_length + bridge_length) / crossing_time * 3.6 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l3233_323326


namespace NUMINAMATH_CALUDE_square_and_sqrt_preserve_geometric_sequence_l3233_323352

-- Define the domain (−∞,0)∪(0,+∞)
def NonZeroReals : Set ℝ := {x : ℝ | x ≠ 0}

-- Define a geometric sequence
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the "preserving geometric sequence" property
def PreservingGeometricSequence (f : ℝ → ℝ) : Prop :=
  ∀ a : ℕ → ℝ, IsGeometricSequence a → IsGeometricSequence (fun n ↦ f (a n))

-- State the theorem
theorem square_and_sqrt_preserve_geometric_sequence :
  (PreservingGeometricSequence (fun x ↦ x^2)) ∧
  (PreservingGeometricSequence (fun x ↦ Real.sqrt (abs x))) :=
sorry

end NUMINAMATH_CALUDE_square_and_sqrt_preserve_geometric_sequence_l3233_323352


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l3233_323330

-- Define the probability of having a boy or girl
def prob_boy_or_girl : ℚ := 1 / 2

-- Define the number of children in the family
def num_children : ℕ := 4

-- The theorem to prove
theorem prob_at_least_one_boy_and_girl : 
  (1 : ℚ) - 2 * (prob_boy_or_girl ^ num_children) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l3233_323330


namespace NUMINAMATH_CALUDE_statement_equivalence_l3233_323361

theorem statement_equivalence (P Q : Prop) : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) := by sorry

end NUMINAMATH_CALUDE_statement_equivalence_l3233_323361


namespace NUMINAMATH_CALUDE_consecutive_composite_sequence_l3233_323362

theorem consecutive_composite_sequence (n : ℕ) : ∃ r : ℕ, ∀ k ∈ Finset.range n, ¬(Nat.Prime (r + k + 1)) :=
sorry

end NUMINAMATH_CALUDE_consecutive_composite_sequence_l3233_323362


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3233_323374

theorem quadratic_root_problem (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x - k = 0 ∧ x = 0) → 
  (∃ y : ℝ, y^2 + 2*y - k = 0 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3233_323374


namespace NUMINAMATH_CALUDE_success_arrangements_l3233_323323

/-- The number of permutations of a multiset -/
def multiset_permutations (n : ℕ) (repeats : List ℕ) : ℕ :=
  Nat.factorial n / (repeats.map Nat.factorial).prod

/-- The number of ways to arrange the letters of SUCCESS -/
theorem success_arrangements : multiset_permutations 7 [3, 2] = 420 := by
  sorry

end NUMINAMATH_CALUDE_success_arrangements_l3233_323323


namespace NUMINAMATH_CALUDE_identity_condition_l3233_323302

theorem identity_condition (a b c : ℝ) : 
  (∀ x y z : ℝ, |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|) ↔ 
  ((a = 0 ∧ b = 0 ∧ c = 1) ∨ 
   (a = 0 ∧ b = 0 ∧ c = -1) ∨ 
   (a = 0 ∧ b = 1 ∧ c = 0) ∨ 
   (a = 0 ∧ b = -1 ∧ c = 0) ∨ 
   (a = 1 ∧ b = 0 ∧ c = 0) ∨ 
   (a = -1 ∧ b = 0 ∧ c = 0)) :=
by sorry

end NUMINAMATH_CALUDE_identity_condition_l3233_323302


namespace NUMINAMATH_CALUDE_total_produce_cost_l3233_323315

/-- Calculates the total cost of produce given specific quantities and pricing conditions -/
theorem total_produce_cost (asparagus_bundles : ℕ) (asparagus_price : ℚ)
                           (grape_boxes : ℕ) (grape_weight : ℚ) (grape_price : ℚ)
                           (apples : ℕ) (apple_price : ℚ)
                           (carrot_bags : ℕ) (carrot_orig_price : ℚ) (carrot_discount : ℚ)
                           (strawberry_pounds : ℕ) (strawberry_orig_price : ℚ) (strawberry_discount : ℚ) :
  asparagus_bundles = 60 ∧ asparagus_price = 3 ∧
  grape_boxes = 40 ∧ grape_weight = 2.2 ∧ grape_price = 2.5 ∧
  apples = 700 ∧ apple_price = 0.5 ∧
  carrot_bags = 100 ∧ carrot_orig_price = 2 ∧ carrot_discount = 0.25 ∧
  strawberry_pounds = 120 ∧ strawberry_orig_price = 3.5 ∧ strawberry_discount = 0.15 →
  (asparagus_bundles : ℚ) * asparagus_price +
  (grape_boxes : ℚ) * grape_weight * grape_price +
  ((apples / 3) * 2 : ℚ) * apple_price +
  (carrot_bags : ℚ) * carrot_orig_price * (1 - carrot_discount) +
  (strawberry_pounds : ℚ) * strawberry_orig_price * (1 - strawberry_discount) = 1140.5 := by
sorry

end NUMINAMATH_CALUDE_total_produce_cost_l3233_323315


namespace NUMINAMATH_CALUDE_abs_sum_iff_positive_l3233_323338

theorem abs_sum_iff_positive (x y : ℝ) : x + y > |x - y| ↔ x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_iff_positive_l3233_323338


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3233_323381

/-- Given a geometric sequence with 8 terms where the first term is 3 and the last term is 39366,
    prove that the 6th term is 23328. -/
theorem sixth_term_of_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : 
  (∀ n, a n = 3 * r^(n-1)) →  -- Geometric sequence definition
  a 8 = 39366 →              -- Last term condition
  a 6 = 23328 :=             -- Theorem to prove
by
  sorry

#check sixth_term_of_geometric_sequence

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3233_323381


namespace NUMINAMATH_CALUDE_base5_division_theorem_l3233_323375

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 5 + d) 0

/-- Converts a base 10 number to base 5 --/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

theorem base5_division_theorem :
  let dividend := [2, 1, 3, 4, 2]  -- 21342₅
  let divisor := [2, 3]            -- 23₅
  let quotient := [4, 0, 4, 3]     -- 4043₅
  (base5ToBase10 dividend) / (base5ToBase10 divisor) = base5ToBase10 quotient := by
  sorry

end NUMINAMATH_CALUDE_base5_division_theorem_l3233_323375


namespace NUMINAMATH_CALUDE_baker_initial_cakes_l3233_323305

theorem baker_initial_cakes (total_cakes : ℕ) (extra_cakes : ℕ) (initial_cakes : ℕ) : 
  total_cakes = 87 → extra_cakes = 9 → initial_cakes = total_cakes - extra_cakes → initial_cakes = 78 := by
  sorry

end NUMINAMATH_CALUDE_baker_initial_cakes_l3233_323305


namespace NUMINAMATH_CALUDE_fourth_number_proof_l3233_323398

theorem fourth_number_proof (n : ℝ) (h1 : n = 27) : 
  let numbers : List ℝ := [3, 16, 33, n + 1]
  (numbers.sum / numbers.length = 20) → (n + 1 = 28) := by
sorry

end NUMINAMATH_CALUDE_fourth_number_proof_l3233_323398
