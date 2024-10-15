import Mathlib

namespace NUMINAMATH_CALUDE_min_value_theorem_l2133_213302

theorem min_value_theorem (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :
  ∃ (min : ℝ), min = -9/8 ∧ ∀ (z : ℝ), z = x + y + x * y → z ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2133_213302


namespace NUMINAMATH_CALUDE_trig_product_equals_one_sixteenth_l2133_213350

theorem trig_product_equals_one_sixteenth : 
  Real.cos (15 * π / 180) * Real.sin (30 * π / 180) * 
  Real.cos (75 * π / 180) * Real.sin (150 * π / 180) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_trig_product_equals_one_sixteenth_l2133_213350


namespace NUMINAMATH_CALUDE_solution_set_l2133_213305

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def satisfies_condition (x : ℕ) : Prop :=
  is_prime (3 * x + 1) ∧ 70 ≤ (3 * x + 1) ∧ (3 * x + 1) ≤ 110

theorem solution_set :
  {x : ℕ | satisfies_condition x} = {24, 26, 32, 34, 36} :=
sorry

end NUMINAMATH_CALUDE_solution_set_l2133_213305


namespace NUMINAMATH_CALUDE_potatoes_for_dinner_l2133_213348

def potatoes_for_lunch : ℕ := 5
def total_potatoes : ℕ := 7

theorem potatoes_for_dinner : total_potatoes - potatoes_for_lunch = 2 := by
  sorry

end NUMINAMATH_CALUDE_potatoes_for_dinner_l2133_213348


namespace NUMINAMATH_CALUDE_restaurant_order_l2133_213314

theorem restaurant_order (b h p s : ℕ) : 
  b = 30 → b = 2 * h → p = h + 5 → s = 3 * p → b + h + p + s = 125 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_order_l2133_213314


namespace NUMINAMATH_CALUDE_calculate_dividend_l2133_213343

/-- Given a division with quotient, divisor, and remainder, calculate the dividend -/
theorem calculate_dividend (quotient divisor remainder : ℝ) :
  quotient = -415.2 →
  divisor = 2735 →
  remainder = 387.3 →
  (quotient * divisor) + remainder = -1135106.7 := by
  sorry

end NUMINAMATH_CALUDE_calculate_dividend_l2133_213343


namespace NUMINAMATH_CALUDE_num_ways_to_select_is_186_l2133_213303

def num_red_balls : ℕ := 4
def num_white_balls : ℕ := 6
def red_ball_score : ℕ := 2
def white_ball_score : ℕ := 1
def total_balls_to_take : ℕ := 5
def min_total_score : ℕ := 7

def score (red white : ℕ) : ℕ :=
  red * red_ball_score + white * white_ball_score

def valid_selection (red white : ℕ) : Prop :=
  red + white = total_balls_to_take ∧ 
  red ≤ num_red_balls ∧ 
  white ≤ num_white_balls ∧ 
  score red white ≥ min_total_score

def num_ways_to_select : ℕ := 
  (Nat.choose num_red_balls 4 * Nat.choose num_white_balls 1) +
  (Nat.choose num_red_balls 3 * Nat.choose num_white_balls 2) +
  (Nat.choose num_red_balls 2 * Nat.choose num_white_balls 3)

theorem num_ways_to_select_is_186 : num_ways_to_select = 186 := by
  sorry

end NUMINAMATH_CALUDE_num_ways_to_select_is_186_l2133_213303


namespace NUMINAMATH_CALUDE_unique_solution_m_l2133_213376

/-- A quadratic equation ax^2 + bx + c = 0 has exactly one solution if and only if its discriminant is zero -/
axiom quadratic_one_solution (a b c : ℝ) : 
  (∃! x, a * x^2 + b * x + c = 0) ↔ b^2 - 4*a*c = 0

/-- The value of m for which 3x^2 - 7x + m = 0 has exactly one solution -/
theorem unique_solution_m : 
  (∃! x, 3 * x^2 - 7 * x + m = 0) ↔ m = 49/12 := by sorry

end NUMINAMATH_CALUDE_unique_solution_m_l2133_213376


namespace NUMINAMATH_CALUDE_replaced_crew_weight_l2133_213357

/-- Proves that the replaced crew member weighs 40 kg given the conditions of the problem -/
theorem replaced_crew_weight (n : ℕ) (old_avg new_avg new_weight : ℝ) :
  n = 20 ∧
  new_avg = old_avg + 2 ∧
  new_weight = 80 →
  n * new_avg - (n - 1) * old_avg = 40 :=
by sorry

end NUMINAMATH_CALUDE_replaced_crew_weight_l2133_213357


namespace NUMINAMATH_CALUDE_third_circle_radius_l2133_213312

/-- Given two externally tangent circles and a third circle tangent to both and their common external tangent, prove the radius of the third circle --/
theorem third_circle_radius (r1 r2 r3 : ℝ) : 
  r1 = 1 →                            -- radius of circle A
  r2 = 4 →                            -- radius of circle B
  (r1 + r2)^2 = r1^2 + r2^2 + 6*r1*r2 → -- circles A and B are externally tangent
  (r1 + r3)^2 = (r1 - r3)^2 + 4*r3 →    -- circle with radius r3 is tangent to circle A
  (r2 + r3)^2 = (r2 - r3)^2 + 16*r3 →   -- circle with radius r3 is tangent to circle B
  r3 = 4/9 :=                           -- radius of the third circle
by sorry

end NUMINAMATH_CALUDE_third_circle_radius_l2133_213312


namespace NUMINAMATH_CALUDE_inequality_solution_l2133_213394

theorem inequality_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 5) (h3 : x ≠ 7) :
  (x - 1) * (x - 4) * (x - 6) / ((x - 2) * (x - 5) * (x - 7)) > 0 ↔
  x < 1 ∨ (2 < x ∧ x < 4) ∨ (5 < x ∧ x < 6) ∨ 7 < x :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2133_213394


namespace NUMINAMATH_CALUDE_science_club_enrollment_l2133_213371

theorem science_club_enrollment (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) :
  total = 150 →
  math = 80 →
  physics = 60 →
  both = 20 →
  total - (math + physics - both) = 30 := by
sorry

end NUMINAMATH_CALUDE_science_club_enrollment_l2133_213371


namespace NUMINAMATH_CALUDE_remaining_area_formula_l2133_213308

/-- The area of a rectangular field with dimensions (x + 8) and (x + 6), 
    excluding a rectangular patch with dimensions (2x - 4) and (x - 3) -/
def remaining_area (x : ℝ) : ℝ :=
  (x + 8) * (x + 6) - (2*x - 4) * (x - 3)

/-- Theorem stating that the remaining area is equal to -x^2 + 24x + 36 -/
theorem remaining_area_formula (x : ℝ) : 
  remaining_area x = -x^2 + 24*x + 36 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_formula_l2133_213308


namespace NUMINAMATH_CALUDE_product_expansion_l2133_213397

theorem product_expansion (y : ℝ) (h : y ≠ 0) :
  (3 / 7) * ((7 / y) + 14 * y^3) = 3 / y + 6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l2133_213397


namespace NUMINAMATH_CALUDE_equation_solution_l2133_213369

theorem equation_solution : ∃ x : ℝ, x = 37/10 ∧ Real.sqrt (3 * Real.sqrt (x - 3)) = (10 - x) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2133_213369


namespace NUMINAMATH_CALUDE_divisibility_by_two_in_odd_base_system_l2133_213330

theorem divisibility_by_two_in_odd_base_system (d : ℕ) (h_odd : Odd d) :
  ∀ (x : ℕ) (digits : List ℕ),
    (x = digits.foldr (λ a acc => a + d * acc) 0) →
    (x % 2 = 0 ↔ digits.sum % 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_two_in_odd_base_system_l2133_213330


namespace NUMINAMATH_CALUDE_total_spectators_l2133_213304

theorem total_spectators (men : ℕ) (children : ℕ) (women : ℕ) 
  (h1 : men = 7000)
  (h2 : children = 2500)
  (h3 : children = 5 * women) :
  men + children + women = 10000 := by
  sorry

end NUMINAMATH_CALUDE_total_spectators_l2133_213304


namespace NUMINAMATH_CALUDE_complex_roots_circle_radius_l2133_213307

theorem complex_roots_circle_radius : 
  ∀ z : ℂ, (z - 2)^6 = 64 * z^6 → Complex.abs (z - (2/3 : ℂ)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_circle_radius_l2133_213307


namespace NUMINAMATH_CALUDE_cube_parallel_edge_pairs_l2133_213313

/-- A cube is a three-dimensional geometric shape with 12 edges. -/
structure Cube where
  edges : Fin 12
  dimensions : Fin 3

/-- A pair of parallel edges in a cube. -/
structure ParallelEdgePair where
  edge1 : Fin 12
  edge2 : Fin 12

/-- The number of parallel edge pairs in a cube. -/
def parallel_edge_pairs (c : Cube) : ℕ := 18

/-- Theorem: A cube has 18 pairs of parallel edges. -/
theorem cube_parallel_edge_pairs (c : Cube) : 
  parallel_edge_pairs c = 18 := by sorry

end NUMINAMATH_CALUDE_cube_parallel_edge_pairs_l2133_213313


namespace NUMINAMATH_CALUDE_shirt_price_proof_l2133_213317

/-- Proves that if a shirt's price after a 15% discount is $68, then its original price was $80. -/
theorem shirt_price_proof (discounted_price : ℝ) (discount_rate : ℝ) : 
  discounted_price = 68 → discount_rate = 0.15 → 
  discounted_price = (1 - discount_rate) * 80 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_proof_l2133_213317


namespace NUMINAMATH_CALUDE_shirt_sales_revenue_function_l2133_213393

/-- The daily net revenue function for shirt sales -/
def daily_net_revenue (x : ℝ) : ℝ :=
  -x^2 + 110*x - 2400

theorem shirt_sales_revenue_function 
  (wholesale_price : ℝ) 
  (initial_price : ℝ) 
  (initial_sales : ℝ) 
  (price_sensitivity : ℝ) 
  (h1 : wholesale_price = 30)
  (h2 : initial_price = 40)
  (h3 : initial_sales = 40)
  (h4 : price_sensitivity = 1)
  (x : ℝ)
  (h5 : x ≥ 40) :
  daily_net_revenue x = (x - wholesale_price) * (initial_sales - (x - initial_price) * price_sensitivity) :=
by
  sorry

#check shirt_sales_revenue_function

end NUMINAMATH_CALUDE_shirt_sales_revenue_function_l2133_213393


namespace NUMINAMATH_CALUDE_unknown_number_proof_l2133_213334

theorem unknown_number_proof (x : ℝ) : 
  (0.15 * 25 + 0.12 * x = 9.15) → x = 45 :=
by sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l2133_213334


namespace NUMINAMATH_CALUDE_shaded_area_of_square_with_removed_triangles_l2133_213384

/-- The area of a shape formed by removing four right triangles from a square -/
theorem shaded_area_of_square_with_removed_triangles 
  (square_side : ℝ) 
  (triangle_leg : ℝ) 
  (h1 : square_side = 6) 
  (h2 : triangle_leg = 2) : 
  square_side ^ 2 - 4 * (1 / 2 * triangle_leg ^ 2) = 28 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_square_with_removed_triangles_l2133_213384


namespace NUMINAMATH_CALUDE_two_unusual_numbers_l2133_213382

/-- A number is unusual if it satisfies the given conditions --/
def IsUnusual (n : ℕ) : Prop :=
  10^99 ≤ n ∧ n < 10^100 ∧ 
  n^3 % 10^100 = n % 10^100 ∧ 
  n^2 % 10^100 ≠ n % 10^100

/-- There exist at least two distinct unusual numbers --/
theorem two_unusual_numbers : ∃ n₁ n₂ : ℕ, IsUnusual n₁ ∧ IsUnusual n₂ ∧ n₁ ≠ n₂ := by
  sorry

end NUMINAMATH_CALUDE_two_unusual_numbers_l2133_213382


namespace NUMINAMATH_CALUDE_fifth_number_in_row_51_l2133_213309

/-- Pascal's triangle binomial coefficient -/
def pascal (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- The number of elements in a row of Pascal's triangle -/
def row_size (row : ℕ) : ℕ :=
  row + 1

theorem fifth_number_in_row_51 :
  pascal 50 4 = 22050 :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_number_in_row_51_l2133_213309


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l2133_213390

theorem smallest_non_factor_product (a b : ℕ) : 
  a ≠ b → 
  a > 0 → 
  b > 0 → 
  48 % a = 0 → 
  48 % b = 0 → 
  48 % (a * b) ≠ 0 → 
  ∀ c d : ℕ, (c ≠ d ∧ c > 0 ∧ d > 0 ∧ 48 % c = 0 ∧ 48 % d = 0 ∧ 48 % (c * d) ≠ 0) → a * b ≤ c * d →
  a * b = 18 :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l2133_213390


namespace NUMINAMATH_CALUDE_mixed_beads_cost_l2133_213300

/-- The cost per box of mixed beads -/
def cost_per_box_mixed (red_cost yellow_cost : ℚ) (total_boxes red_boxes yellow_boxes : ℕ) : ℚ :=
  (red_cost * red_boxes + yellow_cost * yellow_boxes) / total_boxes

/-- Theorem stating the cost per box of mixed beads is $1.32 -/
theorem mixed_beads_cost :
  cost_per_box_mixed (13/10) 2 10 4 4 = 132/100 := by
  sorry

end NUMINAMATH_CALUDE_mixed_beads_cost_l2133_213300


namespace NUMINAMATH_CALUDE_abs_neg_three_halves_l2133_213372

theorem abs_neg_three_halves : |(-3/2 : ℚ)| = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_halves_l2133_213372


namespace NUMINAMATH_CALUDE_last_five_days_avg_l2133_213341

/-- Represents the TV production scenario in a factory --/
structure TVProduction where
  total_days : Nat
  first_period_days : Nat
  first_period_avg : Nat
  monthly_avg : Nat

/-- Calculates the average daily production for the last period --/
def last_period_avg (prod : TVProduction) : Rat :=
  let last_period_days := prod.total_days - prod.first_period_days
  let total_monthly_production := prod.monthly_avg * prod.total_days
  let first_period_production := prod.first_period_avg * prod.first_period_days
  let last_period_production := total_monthly_production - first_period_production
  last_period_production / last_period_days

/-- Theorem stating the average production for the last 5 days --/
theorem last_five_days_avg (prod : TVProduction) 
  (h1 : prod.total_days = 30)
  (h2 : prod.first_period_days = 25)
  (h3 : prod.first_period_avg = 50)
  (h4 : prod.monthly_avg = 45) :
  last_period_avg prod = 20 := by
  sorry

end NUMINAMATH_CALUDE_last_five_days_avg_l2133_213341


namespace NUMINAMATH_CALUDE_average_weight_problem_l2133_213361

theorem average_weight_problem (num_group1 : ℕ) (num_group2 : ℕ) (avg_weight_group2 : ℝ) (avg_weight_total : ℝ) :
  num_group1 = 24 →
  num_group2 = 8 →
  avg_weight_group2 = 45.15 →
  avg_weight_total = 48.975 →
  let total_num := num_group1 + num_group2
  let total_weight := total_num * avg_weight_total
  let weight_group2 := num_group2 * avg_weight_group2
  let weight_group1 := total_weight - weight_group2
  let avg_weight_group1 := weight_group1 / num_group1
  avg_weight_group1 = 50.25 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l2133_213361


namespace NUMINAMATH_CALUDE_citizenship_test_study_time_l2133_213316

/-- Calculates the total study time in hours for a citizenship test -/
theorem citizenship_test_study_time :
  let total_questions : ℕ := 90
  let multiple_choice_questions : ℕ := 30
  let fill_in_blank_questions : ℕ := 30
  let essay_questions : ℕ := 30
  let multiple_choice_time : ℕ := 15  -- minutes per question
  let fill_in_blank_time : ℕ := 25    -- minutes per question
  let essay_time : ℕ := 45            -- minutes per question
  
  let total_time_minutes : ℕ := 
    multiple_choice_questions * multiple_choice_time +
    fill_in_blank_questions * fill_in_blank_time +
    essay_questions * essay_time

  let total_time_hours : ℚ := (total_time_minutes : ℚ) / 60

  total_questions = multiple_choice_questions + fill_in_blank_questions + essay_questions →
  total_time_hours = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_citizenship_test_study_time_l2133_213316


namespace NUMINAMATH_CALUDE_count_special_primes_l2133_213399

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def swap_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  units * 10 + tens

def is_special_prime (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ is_prime (swap_digits n)

theorem count_special_primes : 
  (∃ (s : Finset ℕ), (∀ n ∈ s, is_special_prime n) ∧ s.card = 9 ∧ 
   (∀ m : ℕ, is_special_prime m → m ∈ s)) :=
sorry

end NUMINAMATH_CALUDE_count_special_primes_l2133_213399


namespace NUMINAMATH_CALUDE_semicircle_circumference_from_rectangle_perimeter_l2133_213346

def rectangle_length : ℝ := 16
def rectangle_breadth : ℝ := 14

theorem semicircle_circumference_from_rectangle_perimeter :
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_breadth)
  let square_side := rectangle_perimeter / 4
  let semicircle_circumference := (π * square_side) / 2 + square_side
  ∃ ε > 0, |semicircle_circumference - 38.55| < ε := by
  sorry

end NUMINAMATH_CALUDE_semicircle_circumference_from_rectangle_perimeter_l2133_213346


namespace NUMINAMATH_CALUDE_triangle_theorem_l2133_213386

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the main results -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a^2 = t.b^2 + t.c^2 - t.b * t.c) 
  (h2 : t.a = Real.sqrt 3) : 
  (t.A = π/3) ∧ (Real.sqrt 3 < t.b + t.c ∧ t.b + t.c ≤ 2 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2133_213386


namespace NUMINAMATH_CALUDE_coin_toss_problem_l2133_213321

theorem coin_toss_problem (p : ℝ) (n : ℕ) : 
  p = 1 / 2 →  -- Condition 1: Fair coin
  (1 / 2 : ℝ) ^ n = (1 / 8 : ℝ) →  -- Condition 2: Probability of same side is 0.125 (1/8)
  n = 3 := by  -- Question: Prove n = 3
sorry

end NUMINAMATH_CALUDE_coin_toss_problem_l2133_213321


namespace NUMINAMATH_CALUDE_correct_group_formations_l2133_213323

/-- The number of ways to form n groups of 2 from 2n soldiers -/
def groupFormations (n : ℕ) : ℕ × ℕ :=
  (Nat.factorial (2*n) / Nat.factorial n,
   Nat.factorial (2*n) / (2^n * Nat.factorial n))

/-- Theorem stating the correct number of group formations for both cases -/
theorem correct_group_formations (n : ℕ) :
  groupFormations n = (Nat.factorial (2*n) / Nat.factorial n,
                       Nat.factorial (2*n) / (2^n * Nat.factorial n)) :=
by sorry

end NUMINAMATH_CALUDE_correct_group_formations_l2133_213323


namespace NUMINAMATH_CALUDE_value_of_a_l2133_213354

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2

-- Theorem statement
theorem value_of_a (a : ℝ) : f_derivative a (-1) = 3 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2133_213354


namespace NUMINAMATH_CALUDE_sin_cos_shift_l2133_213359

open Real

theorem sin_cos_shift (x : ℝ) : 
  sin (2 * x) + Real.sqrt 3 * cos (2 * x) = 2 * sin (2 * (x + π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_shift_l2133_213359


namespace NUMINAMATH_CALUDE_system_solution_exists_l2133_213351

theorem system_solution_exists : ∃ (x y z : ℝ),
  (2 * x + 3 * y + z = 13) ∧
  (4 * x^2 + 9 * y^2 + z^2 - 2 * x + 15 * y + 3 * z = 82) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_exists_l2133_213351


namespace NUMINAMATH_CALUDE_triangle_max_area_l2133_213322

/-- Given a triangle ABC with sides a, b, c and area S, where S = a² - (b-c)² 
    and the circumference of its circumcircle is 17π, 
    prove that the maximum value of S is 64. -/
theorem triangle_max_area (a b c S : ℝ) (h1 : S = a^2 - (b - c)^2) 
  (h2 : 2 * Real.pi * (a / (2 * Real.sin (Real.arcsin (8/17)))) = 17 * Real.pi) :
  S ≤ 64 :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2133_213322


namespace NUMINAMATH_CALUDE_blue_fish_with_spots_l2133_213379

theorem blue_fish_with_spots (total_fish : ℕ) (blue_fish : ℕ) (spotted_blue_fish : ℕ) 
  (h1 : total_fish = 60)
  (h2 : blue_fish = total_fish / 3)
  (h3 : spotted_blue_fish = blue_fish / 2) :
  spotted_blue_fish = 10 := by
  sorry

end NUMINAMATH_CALUDE_blue_fish_with_spots_l2133_213379


namespace NUMINAMATH_CALUDE_flag_movement_theorem_l2133_213356

/-- Calculates the total distance a flag moves on a flagpole given the pole height and a sequence of movements. -/
def totalFlagMovement (poleHeight : ℝ) (movements : List ℝ) : ℝ :=
  movements.map (abs) |>.sum

/-- Theorem stating the total distance a flag moves on a 60-foot flagpole when raised to the top, 
    lowered halfway, raised to the top again, and then lowered completely is 180 feet. -/
theorem flag_movement_theorem :
  let poleHeight : ℝ := 60
  let movements : List ℝ := [poleHeight, -poleHeight/2, poleHeight/2, -poleHeight]
  totalFlagMovement poleHeight movements = 180 := by
  sorry

#eval totalFlagMovement 60 [60, -30, 30, -60]

end NUMINAMATH_CALUDE_flag_movement_theorem_l2133_213356


namespace NUMINAMATH_CALUDE_nates_dropped_matches_l2133_213301

/-- Proves that Nate dropped 10 matches in the creek given the initial conditions. -/
theorem nates_dropped_matches (initial_matches : ℕ) (remaining_matches : ℕ) (dropped_matches : ℕ) :
  initial_matches = 70 →
  remaining_matches = 40 →
  initial_matches - remaining_matches = dropped_matches + 2 * dropped_matches →
  dropped_matches = 10 := by
sorry

end NUMINAMATH_CALUDE_nates_dropped_matches_l2133_213301


namespace NUMINAMATH_CALUDE_increase_in_average_marks_l2133_213366

/-- Proves that the increase in average marks is 0.5 when a mark is incorrectly entered as 67 instead of 45 in a class of 44 pupils. -/
theorem increase_in_average_marks 
  (num_pupils : ℕ) 
  (wrong_mark : ℕ) 
  (correct_mark : ℕ) 
  (h1 : num_pupils = 44) 
  (h2 : wrong_mark = 67) 
  (h3 : correct_mark = 45) : 
  (wrong_mark - correct_mark : ℚ) / num_pupils = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_increase_in_average_marks_l2133_213366


namespace NUMINAMATH_CALUDE_min_value_of_z_l2133_213339

theorem min_value_of_z (x y : ℝ) (h : x^2 + 2*x*y - 3*y^2 = 1) : 
  ∃ (z_min : ℝ), z_min = (1 + Real.sqrt 5) / 4 ∧ ∀ z, z = x^2 + y^2 → z ≥ z_min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_z_l2133_213339


namespace NUMINAMATH_CALUDE_modulo_graph_intercepts_sum_l2133_213373

theorem modulo_graph_intercepts_sum (x₀ y₀ : ℕ) : 
  x₀ < 37 → y₀ < 37 →
  (2 * x₀) % 37 = 1 →
  (3 * y₀ + 1) % 37 = 0 →
  x₀ + y₀ = 31 := by
sorry

end NUMINAMATH_CALUDE_modulo_graph_intercepts_sum_l2133_213373


namespace NUMINAMATH_CALUDE_time_subtraction_problem_l2133_213383

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Converts minutes to a Time structure -/
def minutesToTime (m : ℕ) : Time :=
  { hours := m / 60,
    minutes := m % 60,
    valid := by sorry }

/-- Subtracts two Time structures -/
def subtractTime (t1 t2 : Time) : Time :=
  let totalMinutes1 := t1.hours * 60 + t1.minutes
  let totalMinutes2 := t2.hours * 60 + t2.minutes
  minutesToTime (totalMinutes1 - totalMinutes2)

theorem time_subtraction_problem :
  let currentTime : Time := { hours := 18, minutes := 27, valid := by sorry }
  let minutesToSubtract : ℕ := 2880717
  let resultTime : Time := subtractTime currentTime (minutesToTime minutesToSubtract)
  resultTime.hours = 6 ∧ resultTime.minutes = 30 := by sorry

end NUMINAMATH_CALUDE_time_subtraction_problem_l2133_213383


namespace NUMINAMATH_CALUDE_jim_current_age_l2133_213328

/-- Represents the ages of Jim, Fred, and Sam -/
structure Ages where
  jim : ℕ
  fred : ℕ
  sam : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.jim = 2 * ages.fred ∧
  ages.fred = ages.sam + 9 ∧
  ages.jim - 6 = 5 * (ages.sam - 6)

/-- The theorem stating Jim's current age -/
theorem jim_current_age :
  ∃ ages : Ages, satisfiesConditions ages ∧ ages.jim = 46 :=
sorry

end NUMINAMATH_CALUDE_jim_current_age_l2133_213328


namespace NUMINAMATH_CALUDE_ab_positive_iff_hyperbola_l2133_213306

-- Define the condition for a hyperbola
def is_hyperbola (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), a * x^2 - b * y^2 = 1

-- State the theorem
theorem ab_positive_iff_hyperbola (a b : ℝ) :
  a * b > 0 ↔ is_hyperbola a b :=
sorry

end NUMINAMATH_CALUDE_ab_positive_iff_hyperbola_l2133_213306


namespace NUMINAMATH_CALUDE_product_w_z_is_24_l2133_213387

/-- Represents a parallelogram EFGH with given side lengths -/
structure Parallelogram where
  ef : ℝ
  fg : ℝ → ℝ
  gh : ℝ → ℝ
  he : ℝ
  is_parallelogram : ef = gh 0 ∧ fg 0 = he

/-- The product of w and z in the given parallelogram is 24 -/
theorem product_w_z_is_24 (p : Parallelogram)
    (h_ef : p.ef = 42)
    (h_fg : p.fg = fun z => 4 * z^3)
    (h_gh : p.gh = fun w => 3 * w + 6)
    (h_he : p.he = 32) :
    ∃ w z, p.gh w = p.ef ∧ p.fg z = p.he ∧ w * z = 24 := by
  sorry

end NUMINAMATH_CALUDE_product_w_z_is_24_l2133_213387


namespace NUMINAMATH_CALUDE_no_valid_a_l2133_213396

theorem no_valid_a : ¬∃ a : ℕ+, (a ≤ 100) ∧ 
  (∃ x y : ℤ, x ≠ y ∧ 
    2 * x^2 + (3 * a.val + 1) * x + a.val^2 = 0 ∧
    2 * y^2 + (3 * a.val + 1) * y + a.val^2 = 0) :=
by
  sorry

#check no_valid_a

end NUMINAMATH_CALUDE_no_valid_a_l2133_213396


namespace NUMINAMATH_CALUDE_number_greater_than_one_sixth_l2133_213331

theorem number_greater_than_one_sixth (x : ℝ) : x = 1/6 + 0.33333333333333337 → x = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_number_greater_than_one_sixth_l2133_213331


namespace NUMINAMATH_CALUDE_negative_sqrt_of_squared_negative_nine_equals_negative_nine_l2133_213332

theorem negative_sqrt_of_squared_negative_nine_equals_negative_nine :
  -Real.sqrt ((-9)^2) = -9 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_of_squared_negative_nine_equals_negative_nine_l2133_213332


namespace NUMINAMATH_CALUDE_mixed_groups_count_l2133_213315

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ) (group_size : ℕ)
  (boy_boy_photos : ℕ) (girl_girl_photos : ℕ) :
  total_children = 300 →
  total_groups = 100 →
  group_size = 3 →
  total_children = total_groups * group_size →
  boy_boy_photos = 100 →
  girl_girl_photos = 56 →
  (∃ (mixed_groups : ℕ), 
    mixed_groups = 72 ∧
    mixed_groups * 2 = total_groups * group_size - boy_boy_photos - girl_girl_photos) :=
by sorry

end NUMINAMATH_CALUDE_mixed_groups_count_l2133_213315


namespace NUMINAMATH_CALUDE_female_officers_count_l2133_213395

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_duty_percentage : ℚ) :
  total_on_duty = 170 →
  female_on_duty_ratio = 1/2 →
  female_duty_percentage = 17/100 →
  ∃ (total_female : ℕ), total_female = 500 ∧ 
    (↑total_on_duty * female_on_duty_ratio : ℚ) = (↑total_female * female_duty_percentage : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_female_officers_count_l2133_213395


namespace NUMINAMATH_CALUDE_composite_numbers_l2133_213364

theorem composite_numbers (N₁ N₂ : ℕ) : 
  N₁ = 2011 * 2012 * 2013 * 2014 + 1 →
  N₂ = 2012 * 2013 * 2014 * 2015 + 1 →
  ¬(Nat.Prime N₁) ∧ ¬(Nat.Prime N₂) :=
by sorry

end NUMINAMATH_CALUDE_composite_numbers_l2133_213364


namespace NUMINAMATH_CALUDE_nth_roots_of_unity_real_roots_l2133_213358

theorem nth_roots_of_unity_real_roots (n : ℕ) (h : n > 0) :
  ¬ (∀ z : ℂ, z^n = 1 → (z.re = 1 ∧ z.im = 0)) :=
sorry

end NUMINAMATH_CALUDE_nth_roots_of_unity_real_roots_l2133_213358


namespace NUMINAMATH_CALUDE_associated_points_theorem_l2133_213378

/-- Definition of k times associated point -/
def k_times_associated_point (P M : ℝ × ℝ) (k : ℤ) : Prop :=
  let d_PM := Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2)
  let d_PO := Real.sqrt (P.1^2 + P.2^2)
  d_PM = k * d_PO

/-- Main theorem -/
theorem associated_points_theorem :
  let P₁ : ℝ × ℝ := (-1.5, 0)
  let P₂ : ℝ × ℝ := (-1, 0)
  ∀ (b : ℝ),
  (∃ (M : ℝ × ℝ), k_times_associated_point P₁ M 2 ∧ M.2 = 0 →
    (M = (1.5, 0) ∨ M = (-4.5, 0))) ∧
  (∀ (M : ℝ × ℝ) (k : ℤ),
    k_times_associated_point P₁ M k ∧ M.1 = -1.5 ∧ -3 ≤ M.2 ∧ M.2 ≤ 5 →
    k ≤ 3) ∧
  (∃ (A B C : ℝ × ℝ),
    A = (b, 0) ∧ B = (b + 1, 0) ∧
    Real.sqrt ((C.1 - A.1)^2 + C.2^2) = Real.sqrt ((B.1 - A.1)^2 + (C.2 - B.2)^2) ∧
    C.2 / (C.1 - A.1) = Real.sqrt 3 / 3 →
    (∃ (Q : ℝ × ℝ), k_times_associated_point P₂ Q 2 ∧
      (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ Q = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))) →
      (-4 ≤ b ∧ b ≤ -3) ∨ (-1 ≤ b ∧ b ≤ 1))) :=
by
  sorry

end NUMINAMATH_CALUDE_associated_points_theorem_l2133_213378


namespace NUMINAMATH_CALUDE_polynomial_expansion_properties_l2133_213344

theorem polynomial_expansion_properties (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1) ∧
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 243) ∧
  (a₁ + a₃ + a₅ = 122) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_properties_l2133_213344


namespace NUMINAMATH_CALUDE_angle_B_is_pi_over_3_max_area_is_3_sqrt_3_l2133_213318

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def condition (t : Triangle) : Prop :=
  t.b * Real.sin t.A = Real.sqrt 3 * t.a * Real.cos t.B

-- Theorem for part 1
theorem angle_B_is_pi_over_3 (t : Triangle) (h : condition t) : t.B = π / 3 :=
sorry

-- Theorem for part 2
theorem max_area_is_3_sqrt_3 (t : Triangle) (h1 : condition t) (h2 : t.b = 2 * Real.sqrt 3) :
  (∀ s : Triangle, condition s → s.b = 2 * Real.sqrt 3 → 
    1/2 * s.a * s.c * Real.sin s.B ≤ 3 * Real.sqrt 3) ∧ 
  (∃ s : Triangle, condition s ∧ s.b = 2 * Real.sqrt 3 ∧ 
    1/2 * s.a * s.c * Real.sin s.B = 3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_angle_B_is_pi_over_3_max_area_is_3_sqrt_3_l2133_213318


namespace NUMINAMATH_CALUDE_least_n_divisibility_l2133_213389

theorem least_n_divisibility : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), (2 ≤ k) ∧ (k ≤ n) ∧ ((n - 1)^2 % k = 0)) ∧
  (∃ (k : ℕ), (2 ≤ k) ∧ (k ≤ n) ∧ ((n - 1)^2 % k ≠ 0)) ∧
  (∀ (m : ℕ), (m > 0) ∧ (m < n) → 
    (∀ (k : ℕ), (2 ≤ k) ∧ (k ≤ m) → ((m - 1)^2 % k = 0)) ∨
    (∀ (k : ℕ), (2 ≤ k) ∧ (k ≤ m) → ((m - 1)^2 % k ≠ 0))) ∧
  n = 3 :=
by sorry

end NUMINAMATH_CALUDE_least_n_divisibility_l2133_213389


namespace NUMINAMATH_CALUDE_hunting_company_composition_l2133_213349

theorem hunting_company_composition :
  ∃ (foxes wolves bears : ℕ),
    foxes + wolves + bears = 45 ∧
    59 * foxes + 41 * wolves + 40 * bears = 2008 ∧
    foxes = 10 ∧ wolves = 18 ∧ bears = 17 := by
  sorry

end NUMINAMATH_CALUDE_hunting_company_composition_l2133_213349


namespace NUMINAMATH_CALUDE_university_application_options_l2133_213324

theorem university_application_options : 
  let total_universities : ℕ := 6
  let applications_needed : ℕ := 3
  let universities_with_coinciding_exams : ℕ := 2
  
  (Nat.choose (total_universities - universities_with_coinciding_exams) applications_needed) +
  (universities_with_coinciding_exams * Nat.choose (total_universities - universities_with_coinciding_exams) (applications_needed - 1)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_university_application_options_l2133_213324


namespace NUMINAMATH_CALUDE_probability_at_least_one_defective_l2133_213391

/-- The probability of selecting at least one defective bulb when choosing 2 bulbs at random from a box containing 20 bulbs, of which 4 are defective, is 7/19. -/
theorem probability_at_least_one_defective (total_bulbs : Nat) (defective_bulbs : Nat) 
    (h1 : total_bulbs = 20) 
    (h2 : defective_bulbs = 4) : 
  let p := 1 - (total_bulbs - defective_bulbs : ℚ) * (total_bulbs - defective_bulbs - 1) / 
           (total_bulbs * (total_bulbs - 1))
  p = 7 / 19 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_defective_l2133_213391


namespace NUMINAMATH_CALUDE_spinner_probability_l2133_213388

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_C = 1/6 → p_A + p_B + p_C + p_D = 1 → p_D = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l2133_213388


namespace NUMINAMATH_CALUDE_intersection_area_is_three_sqrt_three_half_l2133_213340

/-- Regular tetrahedron with edge length 6 -/
structure RegularTetrahedron where
  edgeLength : ℝ
  edgeLength_eq : edgeLength = 6

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Plane passing through three points -/
structure Plane where
  a : Point3D  -- Vertex A
  m : Point3D  -- Midpoint M
  n : Point3D  -- Point N

/-- The area of intersection between a regular tetrahedron and a plane -/
def intersectionArea (t : RegularTetrahedron) (p : Plane) : ℝ := sorry

/-- Theorem stating that the area of intersection is 3√3/2 -/
theorem intersection_area_is_three_sqrt_three_half (t : RegularTetrahedron) (p : Plane) :
  intersectionArea t p = 3 * Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_intersection_area_is_three_sqrt_three_half_l2133_213340


namespace NUMINAMATH_CALUDE_unique_natural_with_square_neighbors_l2133_213381

theorem unique_natural_with_square_neighbors :
  ∃! (n : ℕ), ∃ (k m : ℕ), n + 15 = k^2 ∧ n - 14 = m^2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_natural_with_square_neighbors_l2133_213381


namespace NUMINAMATH_CALUDE_buratino_coins_impossibility_l2133_213362

theorem buratino_coins_impossibility : ¬ ∃ (n : ℕ), 303 + 6 * n = 456 := by sorry

end NUMINAMATH_CALUDE_buratino_coins_impossibility_l2133_213362


namespace NUMINAMATH_CALUDE_last_three_digits_of_5_to_1999_l2133_213398

theorem last_three_digits_of_5_to_1999 : 5^1999 ≡ 125 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_5_to_1999_l2133_213398


namespace NUMINAMATH_CALUDE_marked_cells_bound_l2133_213347

/-- Represents a cell color on the board -/
inductive CellColor
| Black
| White

/-- Represents a (2n+1) × (2n+1) board -/
def Board (n : ℕ) := Fin (2*n+1) → Fin (2*n+1) → CellColor

/-- Counts the number of cells of a given color in a row -/
def countInRow (board : Board n) (row : Fin (2*n+1)) (color : CellColor) : ℕ := sorry

/-- Counts the number of cells of a given color in a column -/
def countInColumn (board : Board n) (col : Fin (2*n+1)) (color : CellColor) : ℕ := sorry

/-- Determines if a cell should be marked based on its row -/
def isMarkedInRow (board : Board n) (row col : Fin (2*n+1)) : Bool := sorry

/-- Determines if a cell should be marked based on its column -/
def isMarkedInColumn (board : Board n) (row col : Fin (2*n+1)) : Bool := sorry

/-- Counts the total number of marked cells on the board -/
def countMarkedCells (board : Board n) : ℕ := sorry

/-- Counts the total number of black cells on the board -/
def countBlackCells (board : Board n) : ℕ := sorry

/-- Counts the total number of white cells on the board -/
def countWhiteCells (board : Board n) : ℕ := sorry

/-- The main theorem: The number of marked cells is at least half the minimum of black and white cells -/
theorem marked_cells_bound (n : ℕ) (board : Board n) :
  2 * countMarkedCells board ≥ min (countBlackCells board) (countWhiteCells board) := by
  sorry

end NUMINAMATH_CALUDE_marked_cells_bound_l2133_213347


namespace NUMINAMATH_CALUDE_total_bushels_is_65_l2133_213377

/-- The number of bushels needed for all animals for a day on Dany's farm -/
def total_bushels : ℕ :=
  let cow_count : ℕ := 5
  let cow_consumption : ℕ := 3
  let sheep_count : ℕ := 4
  let sheep_consumption : ℕ := 2
  let chicken_count : ℕ := 8
  let chicken_consumption : ℕ := 1
  let pig_count : ℕ := 6
  let pig_consumption : ℕ := 4
  let horse_count : ℕ := 2
  let horse_consumption : ℕ := 5
  cow_count * cow_consumption +
  sheep_count * sheep_consumption +
  chicken_count * chicken_consumption +
  pig_count * pig_consumption +
  horse_count * horse_consumption

theorem total_bushels_is_65 : total_bushels = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_bushels_is_65_l2133_213377


namespace NUMINAMATH_CALUDE_average_selling_price_l2133_213311

def initial_stock : ℝ := 100
def morning_sale_weight : ℝ := 50
def morning_sale_price : ℝ := 1.2
def noon_sale_weight : ℝ := 30
def noon_sale_price : ℝ := 1
def afternoon_sale_weight : ℝ := 20
def afternoon_sale_price : ℝ := 0.8

theorem average_selling_price :
  let total_revenue := morning_sale_weight * morning_sale_price +
                       noon_sale_weight * noon_sale_price +
                       afternoon_sale_weight * afternoon_sale_price
  let total_weight := morning_sale_weight + noon_sale_weight + afternoon_sale_weight
  total_revenue / total_weight = 1.06 := by
  sorry

end NUMINAMATH_CALUDE_average_selling_price_l2133_213311


namespace NUMINAMATH_CALUDE_expression_simplification_l2133_213367

theorem expression_simplification :
  let a : ℚ := 3 / 2015
  let b : ℚ := 11 / 2016
  (6 + a) * (8 + b) - (11 - a) * (3 - b) - 12 * a = 11 / 112 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2133_213367


namespace NUMINAMATH_CALUDE_strongest_teams_in_tournament_l2133_213345

/-- Represents a volleyball team in the tournament -/
structure Team :=
  (name : String)
  (wins : Nat)
  (losses : Nat)

/-- Represents the tournament results -/
structure TournamentResults :=
  (teams : List Team)
  (numTeams : Nat)
  (roundRobin : Bool)
  (bestOfThree : Bool)

/-- Determines if a team is one of the two strongest teams -/
def isStrongestTeam (team : Team) (results : TournamentResults) : Prop :=
  ∃ (otherTeam : Team),
    otherTeam ∈ results.teams ∧
    team ∈ results.teams ∧
    team ≠ otherTeam ∧
    ∀ (t : Team), t ∈ results.teams →
      (t.wins < team.wins ∨ (t.wins = team.wins ∧ t.losses ≥ team.losses)) ∨
      (t.wins < otherTeam.wins ∨ (t.wins = otherTeam.wins ∧ t.losses ≥ otherTeam.losses))

theorem strongest_teams_in_tournament
  (results : TournamentResults)
  (h1 : results.numTeams = 6)
  (h2 : results.roundRobin = true)
  (h3 : results.bestOfThree = true)
  (first : Team)
  (second : Team)
  (fourth : Team)
  (fifth : Team)
  (sixth : Team)
  (h4 : first ∈ results.teams ∧ first.wins = 2 ∧ first.losses = 3)
  (h5 : second ∈ results.teams ∧ second.wins = 4 ∧ second.losses = 1)
  (h6 : fourth ∈ results.teams ∧ fourth.wins = 0 ∧ fourth.losses = 5)
  (h7 : fifth ∈ results.teams ∧ fifth.wins = 4 ∧ fifth.losses = 1)
  (h8 : sixth ∈ results.teams ∧ sixth.wins = 4 ∧ sixth.losses = 1)
  : isStrongestTeam fifth results ∧ isStrongestTeam sixth results :=
sorry

end NUMINAMATH_CALUDE_strongest_teams_in_tournament_l2133_213345


namespace NUMINAMATH_CALUDE_sin_330_degrees_l2133_213365

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l2133_213365


namespace NUMINAMATH_CALUDE_ann_boxes_sold_l2133_213363

theorem ann_boxes_sold (n : ℕ) (mark_sold ann_sold : ℕ) : 
  n = 12 →
  mark_sold = n - 11 →
  ann_sold < n →
  mark_sold ≥ 1 →
  ann_sold ≥ 1 →
  mark_sold + ann_sold < n →
  ann_sold = n - 2 :=
by sorry

end NUMINAMATH_CALUDE_ann_boxes_sold_l2133_213363


namespace NUMINAMATH_CALUDE_oplus_five_two_l2133_213333

-- Define the operation ⊕
def oplus (a b : ℝ) : ℝ := 4 * a + 5 * b

-- Theorem statement
theorem oplus_five_two : oplus 5 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_oplus_five_two_l2133_213333


namespace NUMINAMATH_CALUDE_gcd_204_85_l2133_213336

theorem gcd_204_85 : Nat.gcd 204 85 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_204_85_l2133_213336


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l2133_213338

/-- A convex nonagon is a 9-sided polygon -/
def ConvexNonagon := Nat

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals (n : ConvexNonagon) : Nat :=
  27

theorem nonagon_diagonals :
  ∀ n : ConvexNonagon, num_diagonals n = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l2133_213338


namespace NUMINAMATH_CALUDE_units_digit_of_five_to_ten_l2133_213355

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The units digit of 5^10 is 5 -/
theorem units_digit_of_five_to_ten : unitsDigit (5^10) = 5 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_five_to_ten_l2133_213355


namespace NUMINAMATH_CALUDE_equation_solution_l2133_213326

theorem equation_solution : 
  ∃! x : ℚ, 7 * (2 * x + 3) - 5 = -3 * (2 - 5 * x) + 2 * x ∧ x = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2133_213326


namespace NUMINAMATH_CALUDE_paths_to_n_2_l2133_213385

/-- The number of possible paths from (0,0) to (x, y) -/
def f (x y : ℕ) : ℕ := sorry

/-- The theorem stating that f(n, 2) = (1/2)(n^2 + 3n + 2) for all natural numbers n -/
theorem paths_to_n_2 (n : ℕ) : f n 2 = (n^2 + 3*n + 2) / 2 := by sorry

end NUMINAMATH_CALUDE_paths_to_n_2_l2133_213385


namespace NUMINAMATH_CALUDE_cuboid_volume_is_48_l2133_213392

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cuboid -/
def surfaceArea (d : CuboidDimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Calculates the volume of a cuboid -/
def volume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Theorem stating the volume of a specific cuboid -/
theorem cuboid_volume_is_48 :
  ∃ (d : CuboidDimensions),
    d.length = 2 * d.width ∧
    d.height = 3 * d.width ∧
    surfaceArea d = 88 ∧
    volume d = 48 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_is_48_l2133_213392


namespace NUMINAMATH_CALUDE_goldfish_count_l2133_213325

/-- The number of goldfish in the aquarium -/
def total_goldfish : ℕ := 100

/-- The number of goldfish Maggie was allowed to take home -/
def allowed_goldfish : ℕ := total_goldfish / 2

/-- The number of goldfish Maggie caught -/
def caught_goldfish : ℕ := (3 * allowed_goldfish) / 5

/-- The number of goldfish Maggie still needs to catch -/
def remaining_goldfish : ℕ := 20

theorem goldfish_count : 
  total_goldfish = 100 ∧
  allowed_goldfish = total_goldfish / 2 ∧
  caught_goldfish = (3 * allowed_goldfish) / 5 ∧
  remaining_goldfish = allowed_goldfish - caught_goldfish ∧
  remaining_goldfish = 20 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_count_l2133_213325


namespace NUMINAMATH_CALUDE_dividend_calculation_l2133_213368

/-- Calculates the dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (face_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ) :
  investment = 14400 ∧ 
  face_value = 100 ∧ 
  premium_rate = 0.20 ∧ 
  dividend_rate = 0.05 →
  (investment / (face_value * (1 + premium_rate))) * (face_value * dividend_rate) = 600 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2133_213368


namespace NUMINAMATH_CALUDE_zoo_population_is_90_l2133_213329

/-- Calculates the final number of animals in a zoo after a series of events --/
def final_zoo_population (initial_animals : ℕ) 
                         (gorillas_sent : ℕ) 
                         (hippo_adopted : ℕ) 
                         (rhinos_added : ℕ) 
                         (lion_cubs_born : ℕ) : ℕ :=
  initial_animals - gorillas_sent + hippo_adopted + rhinos_added + lion_cubs_born + 2 * lion_cubs_born

/-- Theorem stating that the final zoo population is 90 given the specific events --/
theorem zoo_population_is_90 : 
  final_zoo_population 68 6 1 3 8 = 90 := by
  sorry


end NUMINAMATH_CALUDE_zoo_population_is_90_l2133_213329


namespace NUMINAMATH_CALUDE_fifteen_balls_four_draws_l2133_213380

/-- The number of ways to draw n balls in order from a bin of m balls,
    where each ball remains outside the bin after it is drawn. -/
def orderedDraw (m n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc * (m - i)) 1

/-- The problem statement -/
theorem fifteen_balls_four_draws :
  orderedDraw 15 4 = 32760 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_balls_four_draws_l2133_213380


namespace NUMINAMATH_CALUDE_original_price_from_discounted_l2133_213335

/-- 
Given a product with an original price, this theorem proves that 
if the price after successive discounts of 15% and 25% is 306, 
then the original price was 480.
-/
theorem original_price_from_discounted (original_price : ℝ) : 
  (1 - 0.25) * (1 - 0.15) * original_price = 306 → original_price = 480 := by
  sorry

end NUMINAMATH_CALUDE_original_price_from_discounted_l2133_213335


namespace NUMINAMATH_CALUDE_max_value_implies_m_eq_one_min_value_of_y_l2133_213327

noncomputable section

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x

-- Define the derivative of f(x)
def f_derivative (m : ℝ) (x : ℝ) : ℝ := 1 / x - m

-- Part 1: Prove that if the maximum value of f(x) is -1, then m = 1
theorem max_value_implies_m_eq_one (m : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > 0, f m x ≤ f m x₀) ∧ f m (1 / m) = -1 → m = 1 :=
sorry

-- Part 2: Prove that the minimum value of y is 2 / (1 + e)
theorem min_value_of_y :
  ∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 →
  f 1 x₁ = 0 ∧ f 1 x₂ = 0 →
  Real.exp x₁ ≤ x₂ →
  (x₁ - x₂) * f_derivative 1 (x₁ + x₂) ≥ 2 / (1 + Real.exp 1) :=
sorry

end

end NUMINAMATH_CALUDE_max_value_implies_m_eq_one_min_value_of_y_l2133_213327


namespace NUMINAMATH_CALUDE_estate_area_calculation_l2133_213370

/-- Represents the scale of the map in miles per inch -/
def scale : ℝ := 350

/-- Represents the length of the rectangle on the map in inches -/
def map_length : ℝ := 9

/-- Represents the width of the rectangle on the map in inches -/
def map_width : ℝ := 6

/-- Calculates the actual length of the estate in miles -/
def actual_length : ℝ := scale * map_length

/-- Calculates the actual width of the estate in miles -/
def actual_width : ℝ := scale * map_width

/-- Calculates the actual area of the estate in square miles -/
def actual_area : ℝ := actual_length * actual_width

theorem estate_area_calculation :
  actual_area = 6615000 := by sorry

end NUMINAMATH_CALUDE_estate_area_calculation_l2133_213370


namespace NUMINAMATH_CALUDE_group_size_problem_l2133_213342

/-- Given a group where each member contributes as many paise as there are members,
    and the total collection is 5929 paise, prove that the number of members is 77. -/
theorem group_size_problem (n : ℕ) (h1 : n * n = 5929) : n = 77 := by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l2133_213342


namespace NUMINAMATH_CALUDE_least_three_digit_9_heavy_l2133_213353

def is_9_heavy (n : ℕ) : Prop := n % 9 > 5

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem least_three_digit_9_heavy : 
  (∀ n : ℕ, is_three_digit n → is_9_heavy n → 105 ≤ n) ∧ 
  is_three_digit 105 ∧ 
  is_9_heavy 105 :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_9_heavy_l2133_213353


namespace NUMINAMATH_CALUDE_factorization_of_a_squared_plus_3a_l2133_213310

theorem factorization_of_a_squared_plus_3a (a : ℝ) : a^2 + 3*a = a*(a+3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_a_squared_plus_3a_l2133_213310


namespace NUMINAMATH_CALUDE_soccer_practice_probability_l2133_213375

theorem soccer_practice_probability (p : ℚ) (h : p = 5/8) :
  1 - p = 3/8 := by sorry

end NUMINAMATH_CALUDE_soccer_practice_probability_l2133_213375


namespace NUMINAMATH_CALUDE_product_testing_theorem_l2133_213360

/-- The number of ways to choose k items from n items, where order matters and repetition is not allowed. -/
def A (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of products -/
def total_products : ℕ := 10

/-- The number of defective products -/
def defective_products : ℕ := 4

/-- The number of ways to find 4 defective products among 10 products, 
    where the first defective is found on the 2nd measurement and the last on the 8th -/
def ways_specific_case : ℕ := A 4 2 * A 5 2 * A 6 4

/-- The number of ways to find 4 defective products among 10 products in at most 6 measurements -/
def ways_at_most_6 : ℕ := A 4 4 + 4 * A 4 3 * A 6 1 + 4 * A 5 3 * A 6 2 + A 6 6

theorem product_testing_theorem :
  (ways_specific_case = A 4 2 * A 5 2 * A 6 4) ∧
  (ways_at_most_6 = A 4 4 + 4 * A 4 3 * A 6 1 + 4 * A 5 3 * A 6 2 + A 6 6) :=
sorry

end NUMINAMATH_CALUDE_product_testing_theorem_l2133_213360


namespace NUMINAMATH_CALUDE_triangle_angle_c_l2133_213320

theorem triangle_angle_c (A B C : ℝ) : 
  A - B = 10 → B = A / 2 → A + B + C = 180 → C = 150 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l2133_213320


namespace NUMINAMATH_CALUDE_first_hole_depth_l2133_213352

/-- Represents the depth of a hole dug by workers. -/
def hole_depth (workers : ℕ) (hours : ℕ) (rate : ℚ) : ℚ :=
  (workers : ℚ) * (hours : ℚ) * rate

/-- The work rate is constant for both holes. -/
def work_rate : ℚ := 1 / 12

theorem first_hole_depth :
  let first_hole := hole_depth 45 8 work_rate
  let second_hole := hole_depth 90 6 work_rate
  second_hole = 45 →
  first_hole = 30 := by sorry

end NUMINAMATH_CALUDE_first_hole_depth_l2133_213352


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2133_213337

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2, 3}
def N : Set Nat := {3, 4, 5}

theorem intersection_complement_equality : M ∩ (U \ N) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2133_213337


namespace NUMINAMATH_CALUDE_container_water_problem_l2133_213374

theorem container_water_problem (x y : ℝ) : 
  x > 0 ∧ y > 0 → -- Containers and total masses are positive
  (4 / 5 * y - x) + (y - x) = 8 * x → -- Pouring water from B to A
  y - x - (4 / 5 * y - x) = 50 → -- B has 50g more water than A
  x = 50 ∧ 4 / 5 * y - x = 150 ∧ y - x = 200 := by
sorry

end NUMINAMATH_CALUDE_container_water_problem_l2133_213374


namespace NUMINAMATH_CALUDE_cone_height_increase_l2133_213319

theorem cone_height_increase (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let V := (1/3) * Real.pi * r^2 * h
  let V' := 2.3 * V
  ∃ x : ℝ, V' = (1/3) * Real.pi * r^2 * (h * (1 + x/100)) → x = 130 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_increase_l2133_213319
