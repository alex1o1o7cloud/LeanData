import Mathlib

namespace NUMINAMATH_CALUDE_total_dress_designs_l716_71624

/-- The number of fabric colors available. -/
def num_colors : ℕ := 5

/-- The number of patterns available. -/
def num_patterns : ℕ := 4

/-- The number of sleeve designs available. -/
def num_sleeve_designs : ℕ := 3

/-- Each dress design requires exactly one color, one pattern, and one sleeve design. -/
theorem total_dress_designs :
  num_colors * num_patterns * num_sleeve_designs = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_dress_designs_l716_71624


namespace NUMINAMATH_CALUDE_candy_bar_cost_l716_71642

def candy_sales (n : Nat) : Nat :=
  10 + 4 * (n - 1)

def total_candy_sales (days : Nat) : Nat :=
  (List.range days).map candy_sales |>.sum

theorem candy_bar_cost (days : Nat) (total_earnings : Rat) :
  days = 6 ∧ total_earnings = 12 →
  total_earnings / total_candy_sales days = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l716_71642


namespace NUMINAMATH_CALUDE_max_intersections_convex_ngon_l716_71644

/-- The maximum number of intersection points of diagonals in a convex n-gon -/
def max_intersections (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3) / 24

/-- Theorem: In a convex n-gon with all diagonals drawn, the maximum number of 
    intersection points of the diagonals is equal to C(n,4) = n(n-1)(n-2)(n-3)/24 -/
theorem max_intersections_convex_ngon (n : ℕ) (h : n ≥ 4) :
  max_intersections n = Nat.choose n 4 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_convex_ngon_l716_71644


namespace NUMINAMATH_CALUDE_curve_description_l716_71647

def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def right_half (x y : ℝ) : Prop := x = Real.sqrt (1 - y^2)

def lower_half (x y : ℝ) : Prop := y = -Real.sqrt (1 - x^2)

def curve_equation (x y : ℝ) : Prop :=
  (x - Real.sqrt (1 - y^2)) * (y + Real.sqrt (1 - x^2)) = 0

theorem curve_description (x y : ℝ) :
  unit_circle x y ∧ (right_half x y ∨ lower_half x y) ↔ curve_equation x y :=
sorry

end NUMINAMATH_CALUDE_curve_description_l716_71647


namespace NUMINAMATH_CALUDE_daniel_purchase_cost_l716_71650

/-- The total cost of items bought by Daniel -/
def total_cost (tax_amount : ℚ) (tax_rate : ℚ) (tax_free_cost : ℚ) : ℚ :=
  (tax_amount / tax_rate) + tax_free_cost

/-- Theorem stating the total cost of items Daniel bought -/
theorem daniel_purchase_cost :
  let tax_amount : ℚ := 30 / 100  -- 30 paise = 0.30 rupees
  let tax_rate : ℚ := 6 / 100     -- 6%
  let tax_free_cost : ℚ := 347 / 10  -- Rs. 34.7
  total_cost tax_amount tax_rate tax_free_cost = 397 / 10 := by
  sorry

#eval total_cost (30/100) (6/100) (347/10)

end NUMINAMATH_CALUDE_daniel_purchase_cost_l716_71650


namespace NUMINAMATH_CALUDE_mixture_problem_l716_71656

theorem mixture_problem (initial_ratio_A B : ℚ) (drawn_off filled_B : ℚ) (final_ratio_A B : ℚ) :
  initial_ratio_A = 7 →
  initial_ratio_B = 5 →
  drawn_off = 9 →
  filled_B = 9 →
  final_ratio_A = 7 →
  final_ratio_B = 9 →
  ∃ x : ℚ,
    let initial_A := initial_ratio_A * x
    let initial_B := initial_ratio_B * x
    let removed_A := (initial_ratio_A / (initial_ratio_A + initial_ratio_B)) * drawn_off
    let removed_B := (initial_ratio_B / (initial_ratio_A + initial_ratio_B)) * drawn_off
    let remaining_A := initial_A - removed_A
    let remaining_B := initial_B - removed_B + filled_B
    remaining_A / remaining_B = final_ratio_A / final_ratio_B ∧
    initial_A = 23.625 :=
by sorry

end NUMINAMATH_CALUDE_mixture_problem_l716_71656


namespace NUMINAMATH_CALUDE_seth_boxes_theorem_l716_71630

/-- The number of boxes Seth bought at the market -/
def market_boxes : ℕ := 3

/-- The number of boxes Seth bought at the farm -/
def farm_boxes : ℕ := 2 * market_boxes

/-- The total number of boxes Seth initially had -/
def initial_boxes : ℕ := market_boxes + farm_boxes

/-- The number of boxes Seth gave to his mother -/
def mother_boxes : ℕ := 1

/-- The number of boxes Seth had after giving to his mother -/
def after_mother_boxes : ℕ := initial_boxes - mother_boxes

/-- The number of boxes Seth donated to charity -/
def charity_boxes : ℕ := after_mother_boxes / 4

/-- The number of boxes Seth had after donating to charity -/
def after_charity_boxes : ℕ := after_mother_boxes - charity_boxes

/-- The number of boxes Seth had left at the end -/
def final_boxes : ℕ := 4

/-- The number of boxes Seth gave to his friends -/
def friend_boxes : ℕ := after_charity_boxes - final_boxes

/-- The total number of boxes Seth bought -/
def total_boxes : ℕ := initial_boxes

theorem seth_boxes_theorem : total_boxes = 14 := by
  sorry

end NUMINAMATH_CALUDE_seth_boxes_theorem_l716_71630


namespace NUMINAMATH_CALUDE_geometric_sum_first_8_terms_l716_71623

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_8_terms :
  let a : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 8
  geometric_sum a r n = 3280/6561 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_first_8_terms_l716_71623


namespace NUMINAMATH_CALUDE_quadratic_residue_prime_power_l716_71664

theorem quadratic_residue_prime_power (p : Nat) (a : Nat) (k : Nat) :
  Nat.Prime p →
  Odd p →
  (∃ y : Nat, y^2 ≡ a [MOD p]) →
  ∃ z : Nat, z^2 ≡ a [MOD p^k] :=
sorry

end NUMINAMATH_CALUDE_quadratic_residue_prime_power_l716_71664


namespace NUMINAMATH_CALUDE_initial_gold_percentage_l716_71626

/-- Given an alloy weighing 48 ounces, adding 12 ounces of pure gold results in a new alloy that is 40% gold.
    This theorem proves that the initial percentage of gold in the alloy is 25%. -/
theorem initial_gold_percentage (initial_weight : ℝ) (added_gold : ℝ) (final_percentage : ℝ) :
  initial_weight = 48 →
  added_gold = 12 →
  final_percentage = 40 →
  (initial_weight * (25 / 100) + added_gold) / (initial_weight + added_gold) = final_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_initial_gold_percentage_l716_71626


namespace NUMINAMATH_CALUDE_oil_press_statement_is_false_l716_71640

-- Define the oil press output function
def oil_press_output (num_presses : ℕ) (output : ℕ) : Prop :=
  num_presses > 0 ∧ output > 0 ∧ (num_presses * (output / num_presses) = output)

-- State the theorem
theorem oil_press_statement_is_false :
  oil_press_output 5 260 →
  ¬ (oil_press_output 20 7200) :=
by
  sorry

end NUMINAMATH_CALUDE_oil_press_statement_is_false_l716_71640


namespace NUMINAMATH_CALUDE_cone_height_is_sqrt_3_l716_71602

-- Define the cone structure
structure Cone where
  base_radius : ℝ
  height : ℝ
  slant_height : ℝ

-- Define the property of the cone's lateral surface
def lateral_surface_is_semicircle (c : Cone) : Prop :=
  c.slant_height = 2

-- Theorem statement
theorem cone_height_is_sqrt_3 (c : Cone) 
  (h_semicircle : lateral_surface_is_semicircle c) : 
  c.height = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_cone_height_is_sqrt_3_l716_71602


namespace NUMINAMATH_CALUDE_z_times_x_plus_y_value_l716_71691

theorem z_times_x_plus_y_value : 
  let x : ℤ := 12
  let y : ℤ := 18
  let z : ℤ := x - y
  z * (x + y) = -180 := by
sorry

end NUMINAMATH_CALUDE_z_times_x_plus_y_value_l716_71691


namespace NUMINAMATH_CALUDE_pelican_shark_ratio_l716_71683

/-- Given that one-third of the Pelicans in Shark Bite Cove moved away, 
    20 Pelicans remain in Shark Bite Cove, and there are 60 sharks in Pelican Bay, 
    prove that the ratio of sharks in Pelican Bay to the original number of 
    Pelicans in Shark Bite Cove is 2:1. -/
theorem pelican_shark_ratio 
  (remaining_pelicans : ℕ) 
  (sharks : ℕ) 
  (h1 : remaining_pelicans = 20)
  (h2 : sharks = 60)
  (h3 : remaining_pelicans = (2/3 : ℚ) * (remaining_pelicans + remaining_pelicans / 2)) :
  (sharks : ℚ) / (remaining_pelicans + remaining_pelicans / 2) = 2 := by
  sorry

#check pelican_shark_ratio

end NUMINAMATH_CALUDE_pelican_shark_ratio_l716_71683


namespace NUMINAMATH_CALUDE_solve_baguette_problem_l716_71659

def baguette_problem (batches_per_day : ℕ) (baguettes_per_batch : ℕ) 
  (sold_after_first : ℕ) (sold_after_second : ℕ) (left_at_end : ℕ) : Prop :=
  let total_baguettes := batches_per_day * baguettes_per_batch
  let sold_first_two := sold_after_first + sold_after_second
  let sold_after_third := total_baguettes - sold_first_two - left_at_end
  sold_after_third = 49

theorem solve_baguette_problem : 
  baguette_problem 3 48 37 52 6 := by sorry

end NUMINAMATH_CALUDE_solve_baguette_problem_l716_71659


namespace NUMINAMATH_CALUDE_f_11_values_l716_71614

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

axiom coprime_property {f : ℕ → ℕ} {a b : ℕ} (h : is_coprime a b) : 
  f (a * b) = f a * f b

axiom prime_property {f : ℕ → ℕ} {m k : ℕ} (hm : is_prime m) (hk : is_prime k) : 
  f (m + k - 3) = f m + f k - f 3

theorem f_11_values (f : ℕ → ℕ) 
  (h1 : ∀ a b : ℕ, is_coprime a b → f (a * b) = f a * f b)
  (h2 : ∀ m k : ℕ, is_prime m → is_prime k → f (m + k - 3) = f m + f k - f 3) :
  f 11 = 1 ∨ f 11 = 11 :=
sorry

end NUMINAMATH_CALUDE_f_11_values_l716_71614


namespace NUMINAMATH_CALUDE_cream_ratio_proof_l716_71638

-- Define the given constants
def servings : ℕ := 4
def fat_per_cup : ℕ := 88
def fat_per_serving : ℕ := 11

-- Define the ratio we want to prove
def cream_ratio : ℚ := 1 / 2

-- Theorem statement
theorem cream_ratio_proof :
  (servings * fat_per_serving : ℚ) / fat_per_cup = cream_ratio := by
  sorry

end NUMINAMATH_CALUDE_cream_ratio_proof_l716_71638


namespace NUMINAMATH_CALUDE_homothety_composition_l716_71628

open Complex

def H_i_squared (i z : ℂ) : ℂ := 2 * (z - i) + i

def T (i z : ℂ) : ℂ := z + i

def H_0_squared (z : ℂ) : ℂ := 2 * z

theorem homothety_composition (i z : ℂ) : H_i_squared i z = (T i ∘ H_0_squared) (z - i) := by sorry

end NUMINAMATH_CALUDE_homothety_composition_l716_71628


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_mean_l716_71670

/-- An arithmetic sequence with common difference 1 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 1

/-- The 5th term is the geometric mean of the 3rd and 11th terms -/
def geometric_mean_condition (a : ℕ → ℝ) : Prop :=
  a 5 ^ 2 = a 3 * a 11

theorem arithmetic_sequence_with_geometric_mean 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : geometric_mean_condition a) : 
  a 1 = -1 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_mean_l716_71670


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_for_solutions_l716_71679

-- Define the function f(x) = |x-a|
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | |x - 1| > (1/2) * (x + 1)} = {x : ℝ | x > 3 ∨ x < 1/3} := by sorry

-- Part 2
theorem range_of_a_for_solutions :
  ∀ a : ℝ, (∃ x : ℝ, f a x + |x - 2| ≤ 3) ↔ -1 ≤ a ∧ a ≤ 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_for_solutions_l716_71679


namespace NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l716_71665

/-- The ages of two people A and B satisfy certain conditions. -/
structure AgeRatio where
  a : ℕ  -- Current age of A
  b : ℕ  -- Current age of B
  past_future_ratio : a - 4 = b + 4  -- Ratio 1:1 for A's past and B's future
  future_past_ratio : a + 4 = 5 * (b - 4)  -- Ratio 5:1 for A's future and B's past

/-- The ratio of current ages of A and B is 2:1 -/
theorem age_ratio_is_two_to_one (ages : AgeRatio) : 
  2 * ages.b = ages.a := by sorry

end NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l716_71665


namespace NUMINAMATH_CALUDE_unique_integral_root_l716_71622

theorem unique_integral_root :
  ∃! (x : ℤ), x - 8 / (x - 4 : ℚ) = 2 - 8 / (x - 4 : ℚ) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_integral_root_l716_71622


namespace NUMINAMATH_CALUDE_apple_water_bottle_difference_l716_71661

theorem apple_water_bottle_difference (total_bottles : ℕ) (water_bottles : ℕ) (apple_bottles : ℕ) : 
  total_bottles = 54 →
  water_bottles = 2 * 12 →
  apple_bottles = total_bottles - water_bottles →
  apple_bottles - water_bottles = 6 := by
  sorry

end NUMINAMATH_CALUDE_apple_water_bottle_difference_l716_71661


namespace NUMINAMATH_CALUDE_sqrt_product_l716_71637

theorem sqrt_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_l716_71637


namespace NUMINAMATH_CALUDE_negation_of_union_membership_l716_71632

theorem negation_of_union_membership {α : Type*} (A B : Set α) (x : α) :
  ¬(x ∈ A ∪ B) ↔ x ∉ A ∧ x ∉ B := by
  sorry

end NUMINAMATH_CALUDE_negation_of_union_membership_l716_71632


namespace NUMINAMATH_CALUDE_tv_production_average_l716_71667

/-- Proves that given the average production of 60 TVs/day for the first 25 days 
    of a 30-day month, and an overall monthly average of 58 TVs/day, 
    the average production for the last 5 days of the month is 48 TVs/day. -/
theorem tv_production_average (first_25_avg : ℕ) (total_days : ℕ) (monthly_avg : ℕ) :
  first_25_avg = 60 →
  total_days = 30 →
  monthly_avg = 58 →
  (monthly_avg * total_days - first_25_avg * 25) / 5 = 48 := by
  sorry

end NUMINAMATH_CALUDE_tv_production_average_l716_71667


namespace NUMINAMATH_CALUDE_dogwood_planting_correct_l716_71666

/-- The number of dogwood trees planted in a park --/
def dogwood_trees_planted (current : ℕ) (total : ℕ) : ℕ :=
  total - current

/-- Theorem stating that the number of dogwood trees planted is correct --/
theorem dogwood_planting_correct (current : ℕ) (total : ℕ) 
  (h : current ≤ total) : 
  dogwood_trees_planted current total = total - current :=
by
  sorry

#eval dogwood_trees_planted 34 83

end NUMINAMATH_CALUDE_dogwood_planting_correct_l716_71666


namespace NUMINAMATH_CALUDE_point_in_region_implies_a_range_l716_71633

theorem point_in_region_implies_a_range (a : ℝ) :
  (1 : ℝ) + (1 : ℝ) + a < 0 → a < -2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_region_implies_a_range_l716_71633


namespace NUMINAMATH_CALUDE_black_stones_count_l716_71625

theorem black_stones_count (total : Nat) (white : Nat) : 
  total = 48 → 
  (4 * white) % 37 = 26 → 
  (4 * white) / 37 = 2 → 
  total - white = 23 := by
sorry

end NUMINAMATH_CALUDE_black_stones_count_l716_71625


namespace NUMINAMATH_CALUDE_sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l716_71635

theorem sum_of_squares_quadratic_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → r₁^2 + r₂^2 = (b^2 - 2*a*c) / a^2 :=
by sorry

theorem sum_of_squares_specific_quadratic :
  let r₁ := (10 + Real.sqrt 36) / 2
  let r₂ := (10 - Real.sqrt 36) / 2
  r₁^2 + r₂^2 = 68 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l716_71635


namespace NUMINAMATH_CALUDE_volume_removed_is_two_l716_71654

/-- Represents a cube with corner cuts -/
structure CutCube where
  side : ℝ
  cut_depth : ℝ
  face_square_side : ℝ

/-- Calculates the volume of material removed from a cut cube -/
def volume_removed (c : CutCube) : ℝ :=
  8 * (c.side - c.face_square_side) * (c.side - c.face_square_side) * c.cut_depth

/-- Theorem stating the volume removed from a 2x2x2 cube with specific cuts is 2 cubic units -/
theorem volume_removed_is_two :
  let c : CutCube := ⟨2, 1, 1⟩
  volume_removed c = 2 := by
  sorry


end NUMINAMATH_CALUDE_volume_removed_is_two_l716_71654


namespace NUMINAMATH_CALUDE_books_in_history_section_l716_71674

/-- Calculates the number of books shelved in the history section. -/
def books_shelved_in_history (initial_books : ℕ) (fiction_books : ℕ) (children_books : ℕ) 
  (misplaced_books : ℕ) (books_left : ℕ) : ℕ :=
  initial_books - fiction_books - children_books + misplaced_books - books_left

/-- Theorem stating the number of books shelved in the history section. -/
theorem books_in_history_section :
  books_shelved_in_history 51 19 8 4 16 = 12 := by
  sorry

#eval books_shelved_in_history 51 19 8 4 16

end NUMINAMATH_CALUDE_books_in_history_section_l716_71674


namespace NUMINAMATH_CALUDE_soda_difference_l716_71609

theorem soda_difference (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 79) (h2 : diet_soda = 53) : 
  regular_soda - diet_soda = 26 := by
  sorry

end NUMINAMATH_CALUDE_soda_difference_l716_71609


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l716_71636

theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples : ℝ) * 0.6 = 420 → initial_apples = 700 :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l716_71636


namespace NUMINAMATH_CALUDE_one_third_vector_AB_l716_71695

/-- Given two vectors OA and OB in 2D space, prove that 1/3 of vector AB equals (-3, -2) -/
theorem one_third_vector_AB (OA OB : ℝ × ℝ) : 
  OA = (2, 8) → OB = (-7, 2) → (1 / 3 : ℝ) • (OB - OA) = (-3, -2) := by sorry

end NUMINAMATH_CALUDE_one_third_vector_AB_l716_71695


namespace NUMINAMATH_CALUDE_apples_per_hour_l716_71677

/-- Proves that eating the same number of apples every hour for 3 hours,
    totaling 15 apples, results in 5 apples per hour. -/
theorem apples_per_hour 
  (total_hours : ℕ) 
  (total_apples : ℕ) 
  (h1 : total_hours = 3) 
  (h2 : total_apples = 15) : 
  total_apples / total_hours = 5 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_hour_l716_71677


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l716_71621

/-- Given two lines l₁ and l₂ in the xy-plane, where l₁ has the equation 2x - y + 1 = 0
    and l₂ is symmetric to l₁ with respect to the line y = -x,
    prove that the equation of l₂ is x - 2y + 1 = 0 -/
theorem symmetric_line_equation :
  ∀ (l₁ l₂ : Set (ℝ × ℝ)),
  (∀ x y, (x, y) ∈ l₁ ↔ 2 * x - y + 1 = 0) →
  (∀ x y, (x, y) ∈ l₂ ↔ ∃ x' y', (x', y') ∈ l₁ ∧ x + x' = y + y') →
  (∀ x y, (x, y) ∈ l₂ ↔ x - 2 * y + 1 = 0) :=
by sorry


end NUMINAMATH_CALUDE_symmetric_line_equation_l716_71621


namespace NUMINAMATH_CALUDE_exactly_two_successes_probability_l716_71668

/-- The probability of success in a single trial -/
def p : ℚ := 3/5

/-- The number of trials -/
def n : ℕ := 5

/-- The number of successes we're interested in -/
def k : ℕ := 2

/-- The binomial probability formula -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1-p)^(n-k)

/-- The main theorem: probability of exactly 2 successes in 5 trials with p = 3/5 is 144/625 -/
theorem exactly_two_successes_probability :
  binomial_probability n k p = 144/625 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_successes_probability_l716_71668


namespace NUMINAMATH_CALUDE_inequality_solution_set_l716_71655

def solution_set (x : ℝ) : Prop := -1 < x ∧ x < 2

theorem inequality_solution_set :
  ∀ x : ℝ, x * (x - 1) < 2 ↔ solution_set x :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l716_71655


namespace NUMINAMATH_CALUDE_inequality_proof_l716_71693

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) / Real.sqrt (x * y) ≤ x / y + y / x :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l716_71693


namespace NUMINAMATH_CALUDE_not_necessarily_p_or_q_l716_71673

theorem not_necessarily_p_or_q (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬(p ∧ q)) : 
  ¬∀ (p q : Prop), (¬p ∧ ¬(p ∧ q)) → (p ∨ q) :=
by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_p_or_q_l716_71673


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l716_71600

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l716_71600


namespace NUMINAMATH_CALUDE_no_cubic_polynomial_satisfies_conditions_l716_71605

theorem no_cubic_polynomial_satisfies_conditions :
  ¬∃ f : ℝ → ℝ, (∃ a b c d : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^3 + b * x^2 + c * x + d) ∧
    (∀ x, f (x^2) = (f x)^2) ∧ (∀ x, f (x^2) = f (f x)) :=
sorry

end NUMINAMATH_CALUDE_no_cubic_polynomial_satisfies_conditions_l716_71605


namespace NUMINAMATH_CALUDE_p_amount_l716_71697

theorem p_amount (p q r : ℝ) : 
  p = (1/8 * p) + (1/8 * p) + 42 → p = 56 := by sorry

end NUMINAMATH_CALUDE_p_amount_l716_71697


namespace NUMINAMATH_CALUDE_range_of_a_l716_71620

/-- The line passing through points on a 2D plane. -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point on a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if two points are on opposite sides of a line. -/
def oppositeSides (p1 p2 : Point2D) (l : Line2D) : Prop :=
  (l.a * p1.x + l.b * p1.y + l.c) * (l.a * p2.x + l.b * p2.y + l.c) < 0

/-- The theorem statement. -/
theorem range_of_a :
  ∀ a : ℝ,
  (oppositeSides (Point2D.mk 0 0) (Point2D.mk 1 1) (Line2D.mk 1 1 (-a))) ↔
  (0 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l716_71620


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l716_71682

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 1 > 0) ↔ (∃ x : ℝ, x^2 - x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l716_71682


namespace NUMINAMATH_CALUDE_remainder_sum_l716_71611

theorem remainder_sum (n : ℤ) (h : n % 21 = 13) : (n % 3 + n % 7 = 7) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l716_71611


namespace NUMINAMATH_CALUDE_peanut_problem_l716_71690

theorem peanut_problem (a b c d : ℕ) : 
  b = a + 6 ∧ 
  c = b + 6 ∧ 
  d = c + 6 ∧ 
  a + b + c + d = 120 → 
  d = 39 := by
sorry

end NUMINAMATH_CALUDE_peanut_problem_l716_71690


namespace NUMINAMATH_CALUDE_distance_to_SFL_is_81_miles_l716_71603

/-- The distance to Super Fun-tastic Land -/
def distance_to_SFL (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: The distance to Super Fun-tastic Land is 81 miles -/
theorem distance_to_SFL_is_81_miles :
  distance_to_SFL 27 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_SFL_is_81_miles_l716_71603


namespace NUMINAMATH_CALUDE_teacher_age_l716_71616

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) :
  num_students = 23 →
  student_avg_age = 22 →
  new_avg_age = student_avg_age + 1 →
  (num_students : ℝ) * student_avg_age + (new_avg_age * (num_students + 1) - student_avg_age * num_students) = 46 * (num_students + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_teacher_age_l716_71616


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l716_71610

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  Real.sqrt 3 * Real.sin B - Real.cos B = 1 →
  b^2 = a * c →
  B = π / 3 ∧ (1/2) * a * c * Real.sin B = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l716_71610


namespace NUMINAMATH_CALUDE_max_value_S_l716_71696

theorem max_value_S (x y z w : Real) 
  (hx : x ∈ Set.Icc 0 1) 
  (hy : y ∈ Set.Icc 0 1) 
  (hz : z ∈ Set.Icc 0 1) 
  (hw : w ∈ Set.Icc 0 1) : 
  (x^2*y + y^2*z + z^2*w + w^2*x - x*y^2 - y*z^2 - z*w^2 - w*x^2) ≤ 8/27 := by
  sorry

end NUMINAMATH_CALUDE_max_value_S_l716_71696


namespace NUMINAMATH_CALUDE_max_candies_in_25_days_l716_71657

/-- Represents the dentist's instructions for candy consumption --/
structure CandyRules :=
  (max_daily : ℕ)
  (threshold : ℕ)
  (reduced_max : ℕ)
  (reduced_days : ℕ)

/-- Calculates the maximum number of candies that can be eaten in a given number of days --/
def max_candies (rules : CandyRules) (days : ℕ) : ℕ :=
  sorry

/-- The dentist's specific instructions --/
def dentist_rules : CandyRules :=
  { max_daily := 10
  , threshold := 7
  , reduced_max := 5
  , reduced_days := 2 }

/-- Theorem stating the maximum number of candies Sonia can eat in 25 days --/
theorem max_candies_in_25_days :
  max_candies dentist_rules 25 = 178 :=
sorry

end NUMINAMATH_CALUDE_max_candies_in_25_days_l716_71657


namespace NUMINAMATH_CALUDE_conference_center_occupancy_l716_71643

def room_capacities : List ℕ := [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320]

def occupancy_rates : List ℚ := [3/4, 5/6, 2/3, 3/5, 4/9, 11/15, 7/10, 1/2, 5/8, 9/14, 8/15, 17/20]

theorem conference_center_occupancy :
  let occupied_rooms := List.zip room_capacities occupancy_rates
  let total_people := occupied_rooms.map (λ (cap, rate) => (cap : ℚ) * rate)
  ⌊total_people.sum⌋ = 1639 := by sorry

end NUMINAMATH_CALUDE_conference_center_occupancy_l716_71643


namespace NUMINAMATH_CALUDE_min_distance_to_point_l716_71686

theorem min_distance_to_point (x y : ℝ) (h : x^2 + y^2 - 4*x + 2 = 0) :
  ∃ (min : ℝ), min = 2 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' + 2 = 0 → x'^2 + (y' - 2)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_point_l716_71686


namespace NUMINAMATH_CALUDE_initial_violet_balloons_count_l716_71646

/-- The number of violet balloons Jason had initially -/
def initial_violet_balloons : ℕ := sorry

/-- The number of violet balloons Jason lost -/
def lost_violet_balloons : ℕ := 3

/-- The number of violet balloons Jason has now -/
def current_violet_balloons : ℕ := 4

/-- Theorem stating that the initial number of violet balloons is 7 -/
theorem initial_violet_balloons_count : initial_violet_balloons = 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_violet_balloons_count_l716_71646


namespace NUMINAMATH_CALUDE_square_difference_nonnegative_l716_71601

theorem square_difference_nonnegative (a b : ℝ) : (a - b)^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_nonnegative_l716_71601


namespace NUMINAMATH_CALUDE_unique_solution_for_reciprocal_squares_sum_l716_71651

theorem unique_solution_for_reciprocal_squares_sum (x y z t : ℕ+) :
  (1 : ℚ) / x^2 + (1 : ℚ) / y^2 + (1 : ℚ) / z^2 + (1 : ℚ) / t^2 = 1 →
  (x = 2 ∧ y = 2 ∧ z = 2 ∧ t = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_reciprocal_squares_sum_l716_71651


namespace NUMINAMATH_CALUDE_rotation_theorem_l716_71680

-- Define a function f with an inverse
variable (f : ℝ → ℝ)
variable (hf : Function.Bijective f)

-- Define the rotation transformation
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)

-- State the theorem
theorem rotation_theorem :
  ∀ x y : ℝ, y = f x ↔ (rotate90 (x, y)).2 = -(Function.invFun f) (rotate90 (x, y)).1 :=
by sorry

end NUMINAMATH_CALUDE_rotation_theorem_l716_71680


namespace NUMINAMATH_CALUDE_athlete_running_time_l716_71645

/-- Represents the calories burned per minute while running -/
def running_rate : ℝ := 10

/-- Represents the calories burned per minute while walking -/
def walking_rate : ℝ := 4

/-- Represents the total calories burned -/
def total_calories : ℝ := 450

/-- Represents the total time spent exercising in minutes -/
def total_time : ℝ := 60

/-- Theorem stating that the athlete spends 35 minutes running -/
theorem athlete_running_time :
  ∃ (r w : ℝ),
    r + w = total_time ∧
    running_rate * r + walking_rate * w = total_calories ∧
    r = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_athlete_running_time_l716_71645


namespace NUMINAMATH_CALUDE_subtraction_of_large_numbers_l716_71649

theorem subtraction_of_large_numbers :
  2222222222222 - 1111111111111 = 1111111111111 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_large_numbers_l716_71649


namespace NUMINAMATH_CALUDE_rob_pennies_l716_71692

/-- The number of pennies Rob has -/
def num_pennies : ℕ := 12

/-- The number of quarters Rob has -/
def num_quarters : ℕ := 7

/-- The number of dimes Rob has -/
def num_dimes : ℕ := 3

/-- The number of nickels Rob has -/
def num_nickels : ℕ := 5

/-- The total amount Rob has in cents -/
def total_amount : ℕ := 242

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

theorem rob_pennies :
  num_quarters * quarter_value + num_dimes * dime_value + num_nickels * nickel_value + num_pennies * penny_value = total_amount :=
by sorry

end NUMINAMATH_CALUDE_rob_pennies_l716_71692


namespace NUMINAMATH_CALUDE_integral_sqrt_minus_sin_l716_71681

theorem integral_sqrt_minus_sin : ∫ x in (-1)..1, (2 * Real.sqrt (1 - x^2) - Real.sin x) = π := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_minus_sin_l716_71681


namespace NUMINAMATH_CALUDE_triangle_inequalities_l716_71660

/-- Properties of a triangle ABC with sides a, b, c, inradius r, area Δ, and circumradius R -/
theorem triangle_inequalities (a b c r Δ R : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 0) (hΔ : Δ > 0) (hR : R > 0) :
  let s := (a + b + c) / 2
  r ≤ (3 / 2) * (a * b * c / (Real.sqrt (a^2 + b^2 + c^2) * (a + b + c))) ∧
  (3 / 2) * (a * b * c / (Real.sqrt (a^2 + b^2 + c^2) * (a + b + c))) ≤ (3 * Real.sqrt 3 / 2) * (a * b * c / (a + b + c)^2) ∧
  (3 * Real.sqrt 3 / 2) * (a * b * c / (a + b + c)^2) ≤ (Real.sqrt 3 / 2) * ((a * b * c)^(2/3) / (a + b + c)) ∧
  (Real.sqrt 3 / 2) * ((a * b * c)^(2/3) / (a + b + c)) ≤ (Real.sqrt 3 / 18) * (a + b + c) ∧
  (Real.sqrt 3 / 18) * (a + b + c) ≤ Real.sqrt (a^2 + b^2 + c^2) / 6 ∧
  Δ ≥ 3 * Real.sqrt 3 * r^2 ∧
  R ≥ 2 * r ∧
  (r = (3 / 2) * (a * b * c / (Real.sqrt (a^2 + b^2 + c^2) * (a + b + c))) ↔ a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l716_71660


namespace NUMINAMATH_CALUDE_prove_distance_l716_71678

def distance_between_cities : ℝ → Prop := λ d =>
  let speed_ab : ℝ := 40
  let speed_ba : ℝ := 49.99999999999999
  let total_time : ℝ := 5 + 24 / 60
  (d / speed_ab + d / speed_ba) = total_time

theorem prove_distance : distance_between_cities 120 := by
  sorry

end NUMINAMATH_CALUDE_prove_distance_l716_71678


namespace NUMINAMATH_CALUDE_not_prime_a_l716_71648

theorem not_prime_a (a b : ℕ+) (h : ∃ k : ℤ, (5 * a^4 + a^2 : ℤ) = k * (b^4 + 3 * b^2 + 4)) : 
  ¬ Nat.Prime a.val := by
  sorry

end NUMINAMATH_CALUDE_not_prime_a_l716_71648


namespace NUMINAMATH_CALUDE_remainder_divisibility_l716_71629

theorem remainder_divisibility (n : ℤ) : 
  ∃ k : ℤ, n = 125 * k + 40 → ∃ m : ℤ, n = 15 * m + 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l716_71629


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l716_71618

theorem units_digit_of_fraction (n : ℕ) : n = 30 * 31 * 32 * 33 * 34 * 35 / 7200 → n % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l716_71618


namespace NUMINAMATH_CALUDE_f_equals_three_l716_71631

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 1
  else if x < 2 then x^2
  else 2*x

theorem f_equals_three (x : ℝ) : f x = 3 ↔ x = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_f_equals_three_l716_71631


namespace NUMINAMATH_CALUDE_min_abs_z_given_constraint_l716_71617

open Complex

theorem min_abs_z_given_constraint (z : ℂ) (h : abs (z - 2*I) = 1) : 
  abs z ≥ 1 ∧ ∃ w : ℂ, abs (w - 2*I) = 1 ∧ abs w = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_given_constraint_l716_71617


namespace NUMINAMATH_CALUDE_max_red_socks_l716_71685

theorem max_red_socks (a b : ℕ) : 
  a + b ≤ 1991 →
  (a.choose 2 + b.choose 2 : ℚ) / ((a + b).choose 2) = 1/2 →
  a ≤ 990 :=
sorry

end NUMINAMATH_CALUDE_max_red_socks_l716_71685


namespace NUMINAMATH_CALUDE_power_two_33_mod_9_l716_71688

theorem power_two_33_mod_9 : 2^33 % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_power_two_33_mod_9_l716_71688


namespace NUMINAMATH_CALUDE_rational_solution_product_l716_71669

theorem rational_solution_product : ∃ (k₁ k₂ : ℕ+), 
  (∃ (x : ℚ), 3 * x^2 + 17 * x + k₁.val = 0) ∧ 
  (∃ (x : ℚ), 3 * x^2 + 17 * x + k₂.val = 0) ∧ 
  (∀ (k : ℕ+), (∃ (x : ℚ), 3 * x^2 + 17 * x + k.val = 0) → k = k₁ ∨ k = k₂) ∧
  k₁.val * k₂.val = 336 := by
sorry

end NUMINAMATH_CALUDE_rational_solution_product_l716_71669


namespace NUMINAMATH_CALUDE_largest_package_size_l716_71606

theorem largest_package_size (ming_pencils catherine_pencils : ℕ) 
  (h1 : ming_pencils = 40)
  (h2 : catherine_pencils = 24) :
  Nat.gcd ming_pencils catherine_pencils = 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l716_71606


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_f_l716_71612

/-- The function f(x) = -x^2 + 2x + 1 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 1

/-- The monotonic decreasing interval of f(x) = -x^2 + 2x + 1 is [1, +∞) -/
theorem monotonic_decreasing_interval_f :
  {x : ℝ | ∀ y, x ≤ y → f x ≥ f y} = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_f_l716_71612


namespace NUMINAMATH_CALUDE_f_negative_range_x_range_for_negative_f_l716_71634

/-- The function f(x) = mx^2 - mx - 6 + m -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 6 + m

theorem f_negative_range (m : ℝ) (x : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f m x < 0) ↔ m < 6/7 :=
sorry

theorem x_range_for_negative_f (m : ℝ) (x : ℝ) :
  (∀ m ∈ Set.Icc (-2) 2, f m x < 0) ↔ -1 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_f_negative_range_x_range_for_negative_f_l716_71634


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l716_71698

/-- Sum of the first n even integers -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of the first n odd integers -/
def sum_odd (n : ℕ) : ℕ := n^2

/-- Count of odd integers divisible by 5 up to 2n-1 -/
def count_odd_div_5 (n : ℕ) : ℕ := (2*n - 1) / 10 + 1

/-- Sum of odd integers divisible by 5 up to 2n-1 -/
def sum_odd_div_5 (n : ℕ) : ℕ := 5 * (count_odd_div_5 n) * (count_odd_div_5 n)

/-- Sum of odd integers not divisible by 5 up to 2n-1 -/
def sum_odd_not_div_5 (n : ℕ) : ℕ := sum_odd n - sum_odd_div_5 n

theorem even_odd_sum_difference (n : ℕ) : 
  sum_even n - sum_odd_not_div_5 n = 51000 := by sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l716_71698


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l716_71672

/-- Given an arithmetic sequence {a_n} with common difference d,
    S_n is the sum of the first n terms. -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d ∧ S n = n * a 1 + n * (n - 1) * d / 2

theorem arithmetic_sequence_ratio
  (a : ℕ → ℚ) (d : ℚ) (S : ℕ → ℚ)
  (h : arithmetic_sequence a d S)
  (h_ratio : S 5 / S 3 = 3) :
  a 5 / a 3 = 17 / 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l716_71672


namespace NUMINAMATH_CALUDE_janice_typing_problem_l716_71639

theorem janice_typing_problem (typing_speed : ℕ) (initial_typing_time : ℕ) 
  (additional_typing_time : ℕ) (erased_sentences : ℕ) (final_typing_time : ℕ) 
  (total_sentences : ℕ) : 
  typing_speed = 6 →
  initial_typing_time = 20 →
  additional_typing_time = 15 →
  erased_sentences = 40 →
  final_typing_time = 18 →
  total_sentences = 536 →
  total_sentences - (typing_speed * (initial_typing_time + additional_typing_time + final_typing_time) - erased_sentences) = 258 := by
  sorry

end NUMINAMATH_CALUDE_janice_typing_problem_l716_71639


namespace NUMINAMATH_CALUDE_no_four_digit_perfect_square_with_condition_l716_71675

theorem no_four_digit_perfect_square_with_condition : ¬ ∃ (abcd : ℕ), 
  (1000 ≤ abcd ∧ abcd ≤ 9999) ∧  -- four-digit number
  (∃ (n : ℕ), abcd = n^2) ∧  -- perfect square
  (∃ (ab cd : ℕ), 
    (10 ≤ ab ∧ ab ≤ 99) ∧  -- ab is two-digit
    (10 ≤ cd ∧ cd ≤ 99) ∧  -- cd is two-digit
    (abcd = 100 * ab + cd) ∧  -- abcd is composed of ab and cd
    (ab = cd / 4)) :=  -- given condition
by sorry


end NUMINAMATH_CALUDE_no_four_digit_perfect_square_with_condition_l716_71675


namespace NUMINAMATH_CALUDE_basketball_free_throw_percentage_l716_71663

theorem basketball_free_throw_percentage 
  (p : ℝ) 
  (h : 0 ≤ p ∧ p ≤ 1) 
  (h_prob : (1 - p)^2 + 2*p*(1 - p) = 16/25) : 
  p = 3/5 := by sorry

end NUMINAMATH_CALUDE_basketball_free_throw_percentage_l716_71663


namespace NUMINAMATH_CALUDE_S_intersect_T_l716_71658

def S : Set ℝ := {x | (x + 5) / (5 - x) > 0}
def T : Set ℝ := {x | x^2 + 4*x - 21 < 0}

theorem S_intersect_T : S ∩ T = {x : ℝ | -5 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_l716_71658


namespace NUMINAMATH_CALUDE_sequence_equality_l716_71607

theorem sequence_equality (C : ℝ) (a : ℕ → ℝ) 
  (hC : C > 1)
  (h1 : a 1 = 1)
  (h2 : a 2 = 2)
  (h3 : ∀ m n : ℕ, m > 0 ∧ n > 0 → a (m * n) = a m * a n)
  (h4 : ∀ m n : ℕ, m > 0 ∧ n > 0 → a (m + n) ≤ C * (a m + a n))
  : ∀ n : ℕ, n > 0 → a n = n := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l716_71607


namespace NUMINAMATH_CALUDE_complex_integer_calculation_l716_71652

theorem complex_integer_calculation : (-7)^7 / 7^4 + 2^6 - 8^2 = -343 := by
  sorry

end NUMINAMATH_CALUDE_complex_integer_calculation_l716_71652


namespace NUMINAMATH_CALUDE_no_additional_omelets_l716_71653

-- Define the number of eggs per omelet type
def eggs_plain : ℕ := 3
def eggs_cheese : ℕ := 4
def eggs_vegetable : ℕ := 5

-- Define the total number of eggs
def total_eggs : ℕ := 36

-- Define the number of omelets already requested
def plain_omelets : ℕ := 4
def cheese_omelets : ℕ := 2
def vegetable_omelets : ℕ := 3

-- Calculate the number of eggs used for requested omelets
def used_eggs : ℕ := plain_omelets * eggs_plain + cheese_omelets * eggs_cheese + vegetable_omelets * eggs_vegetable

-- Define the remaining eggs
def remaining_eggs : ℕ := total_eggs - used_eggs

-- Theorem: No additional omelets can be made
theorem no_additional_omelets :
  remaining_eggs < eggs_plain ∧ remaining_eggs < eggs_cheese ∧ remaining_eggs < eggs_vegetable :=
by sorry

end NUMINAMATH_CALUDE_no_additional_omelets_l716_71653


namespace NUMINAMATH_CALUDE_circle_cutting_theorem_l716_71604

-- Define the circle C1 with center O and radius r
def C1 (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

-- Define a point A on the circumference of C1
def A_on_C1 (O : ℝ × ℝ) (r : ℝ) (A : ℝ × ℝ) : Prop :=
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = r^2

-- Define the existence of a line that cuts C1 into two parts
def cutting_line_exists (O : ℝ × ℝ) (r : ℝ) (A : ℝ × ℝ) : Prop :=
  ∃ (B C : ℝ × ℝ), 
    B ∈ C1 O r ∧ C ∈ C1 O r ∧ 
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = r^2 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = r^2

-- Theorem statement
theorem circle_cutting_theorem (O : ℝ × ℝ) (r : ℝ) (A : ℝ × ℝ) :
  A_on_C1 O r A → cutting_line_exists O r A := by
  sorry

end NUMINAMATH_CALUDE_circle_cutting_theorem_l716_71604


namespace NUMINAMATH_CALUDE_difference_largest_smallest_l716_71689

def digits : List Nat := [6, 2, 5]

def largest_number (digits : List Nat) : Nat :=
  sorry

def smallest_number (digits : List Nat) : Nat :=
  sorry

theorem difference_largest_smallest :
  largest_number digits - smallest_number digits = 396 := by
  sorry

end NUMINAMATH_CALUDE_difference_largest_smallest_l716_71689


namespace NUMINAMATH_CALUDE_horses_meet_after_nine_days_l716_71627

/-- The distance from Chang'an to Qi in li -/
def total_distance : ℝ := 1125

/-- The distance traveled by the good horse on the first day in li -/
def good_horse_initial : ℝ := 103

/-- The daily increase in distance for the good horse in li -/
def good_horse_increase : ℝ := 13

/-- The distance traveled by the mediocre horse on the first day in li -/
def mediocre_horse_initial : ℝ := 97

/-- The daily decrease in distance for the mediocre horse in li -/
def mediocre_horse_decrease : ℝ := 0.5

/-- The number of days it takes for the horses to meet -/
def meeting_days : ℕ := 9

/-- Theorem stating that the horses meet after 9 days -/
theorem horses_meet_after_nine_days :
  (good_horse_initial * meeting_days + (meeting_days * (meeting_days - 1) / 2) * good_horse_increase +
   mediocre_horse_initial * meeting_days - (meeting_days * (meeting_days - 1) / 2) * mediocre_horse_decrease) =
  2 * total_distance := by
  sorry

#check horses_meet_after_nine_days

end NUMINAMATH_CALUDE_horses_meet_after_nine_days_l716_71627


namespace NUMINAMATH_CALUDE_first_marvelous_monday_l716_71662

/-- Represents a date with a year, month, and day. -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents a day of the week. -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns true if the given date is a Monday. -/
def isMonday (d : Date) (startDate : Date) (startDay : DayOfWeek) : Prop :=
  sorry

/-- Returns true if the given date is the fifth Monday of its month. -/
def isFifthMonday (d : Date) (startDate : Date) (startDay : DayOfWeek) : Prop :=
  sorry

/-- Returns true if date d1 is strictly after date d2. -/
def isAfter (d1 d2 : Date) : Prop :=
  sorry

theorem first_marvelous_monday 
  (schoolStartDate : Date)
  (h1 : schoolStartDate.year = 2023)
  (h2 : schoolStartDate.month = 9)
  (h3 : schoolStartDate.day = 11)
  (h4 : isMonday schoolStartDate schoolStartDate DayOfWeek.Monday) :
  ∃ (marvelousMonday : Date), 
    marvelousMonday.year = 2023 ∧ 
    marvelousMonday.month = 10 ∧ 
    marvelousMonday.day = 30 ∧
    isFifthMonday marvelousMonday schoolStartDate DayOfWeek.Monday ∧
    isAfter marvelousMonday schoolStartDate ∧
    ∀ (d : Date), 
      isFifthMonday d schoolStartDate DayOfWeek.Monday → 
      isAfter d schoolStartDate → 
      ¬(isAfter d marvelousMonday) :=
by sorry

end NUMINAMATH_CALUDE_first_marvelous_monday_l716_71662


namespace NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l716_71676

theorem least_positive_integer_with_given_remainders : ∃! N : ℕ,
  (N > 0) ∧
  (N % 6 = 5) ∧
  (N % 7 = 6) ∧
  (N % 8 = 7) ∧
  (N % 9 = 8) ∧
  (N % 10 = 9) ∧
  (N % 11 = 10) ∧
  (∀ M : ℕ, M > 0 ∧ M % 6 = 5 ∧ M % 7 = 6 ∧ M % 8 = 7 ∧ M % 9 = 8 ∧ M % 10 = 9 ∧ M % 11 = 10 → M ≥ N) ∧
  N = 27719 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l716_71676


namespace NUMINAMATH_CALUDE_maria_savings_l716_71671

/-- Calculates the amount left in Maria's savings after buying sweaters and scarves. -/
def amount_left (sweater_price scarf_price : ℕ) (num_sweaters num_scarves : ℕ) (initial_savings : ℕ) : ℕ :=
  initial_savings - (sweater_price * num_sweaters + scarf_price * num_scarves)

/-- Proves that Maria will have $200 left in her savings after buying sweaters and scarves. -/
theorem maria_savings : amount_left 30 20 6 6 500 = 200 := by
  sorry

end NUMINAMATH_CALUDE_maria_savings_l716_71671


namespace NUMINAMATH_CALUDE_six_doctors_three_days_l716_71694

/-- The number of ways for a given number of doctors to each choose one rest day from a given number of days. -/
def restDayChoices (numDoctors : ℕ) (numDays : ℕ) : ℕ :=
  numDays ^ numDoctors

/-- Theorem stating that for 6 doctors choosing from 3 days, the number of choices is 3^6. -/
theorem six_doctors_three_days : 
  restDayChoices 6 3 = 3^6 := by
  sorry

end NUMINAMATH_CALUDE_six_doctors_three_days_l716_71694


namespace NUMINAMATH_CALUDE_green_yards_calculation_l716_71687

/-- The number of yards of silk dyed green in a factory order -/
def green_yards (total_yards pink_yards : ℕ) : ℕ :=
  total_yards - pink_yards

/-- Theorem stating that the number of yards dyed green is 61921 -/
theorem green_yards_calculation :
  green_yards 111421 49500 = 61921 := by
  sorry

end NUMINAMATH_CALUDE_green_yards_calculation_l716_71687


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l716_71619

/-- 
Given that the coefficient of the third term in the binomial expansion 
of (x - 1/(2x))^n is 7, prove that n = 8.
-/
theorem binomial_expansion_coefficient (n : ℕ) : 
  (1/4 : ℚ) * (n.choose 2) = 7 → n = 8 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l716_71619


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l716_71613

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x - 3 = 0 ∧ x = 1) → 
  (∃ y : ℝ, 3 * y^2 - m * y - 3 = 0 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l716_71613


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_value_l716_71699

def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_first_term_value
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (t : ℝ)
  (h1 : a 1 = t)
  (h2 : geometric_sequence a)
  (h3 : ∀ n : ℕ+, a (n + 1) = 2 * S n + 1)
  : t = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_value_l716_71699


namespace NUMINAMATH_CALUDE_even_function_condition_l716_71684

theorem even_function_condition (a : ℝ) :
  (∀ x : ℝ, (x - 1) * (x - a) = (-x - 1) * (-x - a)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_condition_l716_71684


namespace NUMINAMATH_CALUDE_systematic_sample_max_l716_71641

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  totalProducts : Nat
  sampleSize : Nat
  interval : Nat

/-- Creates a systematic sample given total products and sample size -/
def createSystematicSample (totalProducts sampleSize : Nat) : SystematicSample :=
  { totalProducts := totalProducts
  , sampleSize := sampleSize
  , interval := totalProducts / sampleSize }

/-- Checks if a number is in the sample given the first element -/
def isInSample (sample : SystematicSample) (first last : Nat) : Prop :=
  ∃ k, k < sample.sampleSize ∧ first + k * sample.interval = last

/-- Theorem: In a systematic sample of size 5 from 80 products,
    if product 28 is in the sample, then the maximum number in the sample is 76 -/
theorem systematic_sample_max (sample : SystematicSample) 
  (h1 : sample.totalProducts = 80)
  (h2 : sample.sampleSize = 5)
  (h3 : isInSample sample 28 28) :
  isInSample sample 28 76 ∧ ∀ n, isInSample sample 28 n → n ≤ 76 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_max_l716_71641


namespace NUMINAMATH_CALUDE_square_area_problem_l716_71615

theorem square_area_problem (a b c : ℕ) : 
  4 * a < b → c^2 = a^2 + b^2 + 10 → c^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_problem_l716_71615


namespace NUMINAMATH_CALUDE_gcd_of_156_and_195_l716_71608

theorem gcd_of_156_and_195 : Nat.gcd 156 195 = 39 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_156_and_195_l716_71608
