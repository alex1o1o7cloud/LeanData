import Mathlib

namespace NUMINAMATH_CALUDE_cranberry_juice_cost_per_ounce_l2813_281361

/-- The cost per ounce of a can of cranberry juice -/
def cost_per_ounce (total_cost : ℚ) (volume : ℚ) : ℚ :=
  total_cost / volume

/-- Theorem: The cost per ounce of a 12-ounce can of cranberry juice selling for 84 cents is 7 cents -/
theorem cranberry_juice_cost_per_ounce :
  cost_per_ounce 84 12 = 7 := by
  sorry

#eval cost_per_ounce 84 12

end NUMINAMATH_CALUDE_cranberry_juice_cost_per_ounce_l2813_281361


namespace NUMINAMATH_CALUDE_lightning_rod_height_l2813_281352

/-- Given a lightning rod that breaks twice under strong wind conditions, 
    this theorem proves the height of the rod. -/
theorem lightning_rod_height (h : ℝ) (x₁ : ℝ) (x₂ : ℝ) : 
  h > 0 → 
  x₁ > 0 → 
  x₂ > 0 → 
  h^2 - x₁^2 = 400 → 
  h^2 - x₂^2 = 900 → 
  x₂ = x₁ - 5 → 
  h = Real.sqrt 3156.25 := by
sorry

end NUMINAMATH_CALUDE_lightning_rod_height_l2813_281352


namespace NUMINAMATH_CALUDE_table_height_is_130_l2813_281312

/-- The height of the table in cm -/
def table_height : ℝ := 130

/-- The height of the bottle in cm -/
def bottle_height : ℝ := sorry

/-- The height of the can in cm -/
def can_height : ℝ := sorry

/-- The distance from the top of the can on the floor to the top of the bottle on the table is 150 cm -/
axiom condition1 : table_height + bottle_height = can_height + 150

/-- The distance from the top of the bottle on the floor to the top of the can on the table is 110 cm -/
axiom condition2 : table_height + can_height = bottle_height + 110

/-- Theorem: Given the conditions, the height of the table is 130 cm -/
theorem table_height_is_130 : table_height = 130 := by sorry

end NUMINAMATH_CALUDE_table_height_is_130_l2813_281312


namespace NUMINAMATH_CALUDE_unique_prime_in_range_l2813_281368

theorem unique_prime_in_range : 
  ∃! n : ℕ, 30 < n ∧ n ≤ 43 ∧ Prime n ∧ n % 9 = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_in_range_l2813_281368


namespace NUMINAMATH_CALUDE_fraction_sum_minus_eight_l2813_281327

theorem fraction_sum_minus_eight : 
  (4/3 : ℚ) + (7/5 : ℚ) + (12/10 : ℚ) + (23/20 : ℚ) + (45/40 : ℚ) + (89/80 : ℚ) - 8 = -163/240 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_minus_eight_l2813_281327


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2813_281393

theorem min_value_quadratic (s : ℝ) :
  -8 * s^2 + 64 * s + 20 ≥ 148 ∧ ∃ t : ℝ, -8 * t^2 + 64 * t + 20 = 148 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2813_281393


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2813_281397

theorem sum_of_cubes (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a * b + a * c + b * c = 5) 
  (h3 : a * b * c = -6) : 
  a^3 + b^3 + c^3 = -40 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2813_281397


namespace NUMINAMATH_CALUDE_total_paint_used_l2813_281306

-- Define the amount of white paint used
def white_paint : ℕ := 660

-- Define the amount of blue paint used
def blue_paint : ℕ := 6029

-- Theorem stating the total amount of paint used
theorem total_paint_used : white_paint + blue_paint = 6689 := by
  sorry

end NUMINAMATH_CALUDE_total_paint_used_l2813_281306


namespace NUMINAMATH_CALUDE_solution_equation_l2813_281390

theorem solution_equation (m n : ℕ+) (x : ℝ) 
  (h1 : x = m + Real.sqrt n)
  (h2 : x^2 - 10*x + 1 = Real.sqrt x * (x + 1)) : 
  m + n = 55 := by
sorry

end NUMINAMATH_CALUDE_solution_equation_l2813_281390


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2813_281356

theorem polynomial_evaluation :
  let f : ℝ → ℝ := λ x ↦ x^4 + x^3 + x^2 + x + 2
  f 2 = 32 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2813_281356


namespace NUMINAMATH_CALUDE_dividing_line_slope_l2813_281332

/-- Polygon in the xy-plane with given vertices -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Line passing through the origin with a given slope -/
structure Line where
  slope : ℝ

/-- Function to calculate the area of a polygon -/
def calculateArea (p : Polygon) : ℝ := sorry

/-- Function to check if a line divides a polygon into two equal areas -/
def dividesEqualArea (p : Polygon) (l : Line) : Prop := sorry

/-- The polygon with the given vertices -/
def givenPolygon : Polygon := {
  vertices := [(0, 0), (0, 4), (4, 4), (4, 2), (7, 2), (7, 0)]
}

/-- The theorem stating that the line with slope 2/7 divides the given polygon into two equal areas -/
theorem dividing_line_slope : 
  dividesEqualArea givenPolygon { slope := 2/7 } := by sorry

end NUMINAMATH_CALUDE_dividing_line_slope_l2813_281332


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l2813_281338

theorem least_number_with_remainder (n : ℕ) : n = 256 →
  (∃ k : ℕ, n = 18 * k + 4) ∧
  (∀ m : ℕ, m < n → ¬(∃ j : ℕ, m = 18 * j + 4)) := by
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l2813_281338


namespace NUMINAMATH_CALUDE_constant_c_value_l2813_281301

theorem constant_c_value (b c : ℚ) :
  (∀ x : ℚ, (x + 3) * (x + b) = x^2 + c*x + 8) →
  c = 17/3 := by
sorry

end NUMINAMATH_CALUDE_constant_c_value_l2813_281301


namespace NUMINAMATH_CALUDE_lake_pleasant_excursion_l2813_281384

theorem lake_pleasant_excursion (total_kids : ℕ) 
  (h1 : total_kids = 40)
  (h2 : ∃ tubing_kids : ℕ, 4 * tubing_kids = total_kids)
  (h3 : ∃ rafting_kids : ℕ, 2 * rafting_kids = tubing_kids) :
  rafting_kids = 5 := by
  sorry

end NUMINAMATH_CALUDE_lake_pleasant_excursion_l2813_281384


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2813_281381

theorem quadratic_factorization (y : ℝ) (a b : ℤ) 
  (h : ∀ y, 2 * y^2 + 5 * y - 12 = (2 * y + a) * (y + b)) : 
  a - b = -7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2813_281381


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2813_281311

def complex_equation (z : ℂ) : Prop := (1 + Complex.I) * z = 2 * Complex.I

theorem z_in_fourth_quadrant (z : ℂ) (h : complex_equation z) : 
  z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2813_281311


namespace NUMINAMATH_CALUDE_sin_405_degrees_l2813_281373

theorem sin_405_degrees : Real.sin (405 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_405_degrees_l2813_281373


namespace NUMINAMATH_CALUDE_rabbit_carrot_consumption_l2813_281347

theorem rabbit_carrot_consumption :
  ∀ (rabbit_days deer_days : ℕ) (total_food : ℕ),
    rabbit_days = deer_days + 2 →
    5 * rabbit_days = total_food →
    6 * deer_days = total_food →
    5 * rabbit_days = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_rabbit_carrot_consumption_l2813_281347


namespace NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l2813_281360

noncomputable def f (a b x : ℝ) : ℝ := (1/2) * a * x^2 - (a^2 + b) * x + a * Real.log x

theorem tangent_line_and_monotonicity :
  ∀ (a b : ℝ),
  (a = -1 ∧ b = 0 →
    ∃ (m c : ℝ), m = -3 ∧ c = -3/2 ∧
    ∀ x y, y = m * (x - 1) + c ↔ 6 * x + 2 * y - 3 = 0) ∧
  (b = 1 →
    ((a ≤ 0 → ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a b x₁ > f a b x₂) ∧
    (a > 1 →
      (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1/a → f a b x₁ < f a b x₂) ∧
      (∀ x₁ x₂, 1/a < x₁ ∧ x₁ < x₂ ∧ x₂ < a → f a b x₁ > f a b x₂) ∧
      (∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ → f a b x₁ < f a b x₂)))) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l2813_281360


namespace NUMINAMATH_CALUDE_selection_methods_l2813_281388

theorem selection_methods (female_students male_students : ℕ) 
  (h1 : female_students = 3) 
  (h2 : male_students = 2) : 
  female_students + male_students = 5 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_l2813_281388


namespace NUMINAMATH_CALUDE_salary_increase_to_original_l2813_281304

/-- Proves that a 56.25% increase is required to regain the original salary after a 30% reduction and 10% bonus --/
theorem salary_increase_to_original (S : ℝ) (S_pos : S > 0) : 
  let reduced_salary := 0.7 * S
  let bonus := 0.1 * S
  let new_salary := reduced_salary + bonus
  (S - new_salary) / new_salary = 0.5625 := by sorry

end NUMINAMATH_CALUDE_salary_increase_to_original_l2813_281304


namespace NUMINAMATH_CALUDE_coin_toss_probability_l2813_281309

/-- Represents the possible outcomes of a coin toss -/
inductive CoinToss
| Heads
| Tails

/-- The probability of getting heads in a single toss -/
def heads_prob : ℚ := 2/3

/-- The probability of getting tails in a single toss -/
def tails_prob : ℚ := 1/3

/-- The number of coin tosses -/
def num_tosses : ℕ := 10

/-- The target position to reach -/
def target_pos : ℤ := 6

/-- The position to avoid -/
def avoid_pos : ℤ := -3

/-- A function that calculates the probability of reaching the target position
    without hitting the avoid position in the given number of tosses -/
def prob_reach_target (heads_prob : ℚ) (tails_prob : ℚ) (num_tosses : ℕ) 
                      (target_pos : ℤ) (avoid_pos : ℤ) : ℚ :=
  sorry

theorem coin_toss_probability : 
  prob_reach_target heads_prob tails_prob num_tosses target_pos avoid_pos = 5120/59049 :=
sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l2813_281309


namespace NUMINAMATH_CALUDE_complex_modulus_example_l2813_281339

theorem complex_modulus_example : ∃ (z : ℂ), z = 4 + 3*I ∧ Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l2813_281339


namespace NUMINAMATH_CALUDE_birds_landed_on_fence_l2813_281317

/-- Given an initial number of birds and a final total number of birds on a fence,
    calculate the number of birds that landed on the fence. -/
def birds_landed (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that 8 birds landed on the fence given the initial and final counts. -/
theorem birds_landed_on_fence : birds_landed 12 20 = 8 := by
  sorry

end NUMINAMATH_CALUDE_birds_landed_on_fence_l2813_281317


namespace NUMINAMATH_CALUDE_five_items_three_categories_l2813_281383

/-- The number of ways to distribute n distinct items among k distinct categories,
    where each item must be used exactly once. -/
def distributionCount (n k : ℕ) : ℕ :=
  k^n - (k * 1 + k.choose 2 * (2^n - 2))

/-- Theorem stating that there are 150 ways to distribute 5 distinct items
    among 3 distinct categories, where each item must be used exactly once. -/
theorem five_items_three_categories :
  distributionCount 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_five_items_three_categories_l2813_281383


namespace NUMINAMATH_CALUDE_school_gymnastics_ratio_l2813_281374

theorem school_gymnastics_ratio (total_students : ℕ) 
  (h_total : total_students = 120) : 
  ¬ ∃ (boys girls : ℕ), boys + girls = total_students ∧ 9 * girls = 2 * boys := by
  sorry

end NUMINAMATH_CALUDE_school_gymnastics_ratio_l2813_281374


namespace NUMINAMATH_CALUDE_no_prime_generating_pair_l2813_281376

theorem no_prime_generating_pair : 
  ¬ ∃ (a b : ℕ+), ∀ (p q : ℕ), 
    1000 < p ∧ 1000 < q ∧ 
    Nat.Prime p ∧ Nat.Prime q ∧ 
    p ≠ q → 
    Nat.Prime (a * p + b * q) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_generating_pair_l2813_281376


namespace NUMINAMATH_CALUDE_trajectory_equation_l2813_281382

/-- The trajectory of a point P satisfying |PM| - |PN| = 2√2, where M(-2, 0) and N(2, 0) are fixed points -/
def trajectory (x y : ℝ) : Prop :=
  x > 0 ∧ x^2 / 2 - y^2 / 2 = 1

/-- The distance condition for point P -/
def distance_condition (x y : ℝ) : Prop :=
  Real.sqrt ((x + 2)^2 + y^2) - Real.sqrt ((x - 2)^2 + y^2) = 2 * Real.sqrt 2

theorem trajectory_equation :
  ∀ x y : ℝ, distance_condition x y → trajectory x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l2813_281382


namespace NUMINAMATH_CALUDE_old_selling_price_l2813_281302

/-- Given a product with an increased gross profit and new selling price, calculate the old selling price. -/
theorem old_selling_price (cost : ℝ) (new_selling_price : ℝ) : 
  (new_selling_price = cost * 1.15) →  -- New selling price is cost plus 15% profit
  (new_selling_price = 92) →           -- New selling price is $92.00
  (cost * 1.10 = 88) :=                -- Old selling price (cost plus 10% profit) is $88.00
by sorry

end NUMINAMATH_CALUDE_old_selling_price_l2813_281302


namespace NUMINAMATH_CALUDE_jovana_shells_added_l2813_281341

/-- The amount of shells added to a bucket -/
def shells_added (initial_amount final_amount : ℕ) : ℕ :=
  final_amount - initial_amount

/-- Theorem: The amount of shells Jovana added is 23 pounds -/
theorem jovana_shells_added :
  let initial_amount : ℕ := 5
  let final_amount : ℕ := 28
  shells_added initial_amount final_amount = 23 := by
  sorry

end NUMINAMATH_CALUDE_jovana_shells_added_l2813_281341


namespace NUMINAMATH_CALUDE_lauryn_company_men_count_l2813_281315

theorem lauryn_company_men_count :
  ∀ (men women : ℕ),
    men + women = 180 →
    women = men + 20 →
    men = 80 := by
sorry

end NUMINAMATH_CALUDE_lauryn_company_men_count_l2813_281315


namespace NUMINAMATH_CALUDE_range_of_a_l2813_281359

/-- The function g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-1)*x + a - 2*a^2

/-- The function h(x) -/
def h (x : ℝ) : ℝ := (x-1)^2

/-- The set A -/
def A (a : ℝ) : Set ℝ := {x | g a x > 0}

/-- The set B -/
def B : Set ℝ := {x | h x < 1}

/-- The function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x * g a x

/-- The set C -/
def C (a : ℝ) : Set ℝ := {x | f a x > 0}

/-- The theorem stating the range of a -/
theorem range_of_a :
  ∀ a : ℝ, (A a ∩ B).Nonempty ∧ (C a ∩ B).Nonempty ↔ 
    (1/3 < a ∧ a < 2) ∨ (-1/2 < a ∧ a < 1/3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2813_281359


namespace NUMINAMATH_CALUDE_three_digit_mean_rearrangement_l2813_281322

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  7 * a = 3 * b + 4 * c

def solution_set : Set ℕ :=
  {111, 222, 333, 444, 555, 666, 777, 888, 999, 407, 518, 629, 370, 481, 592}

theorem three_digit_mean_rearrangement (n : ℕ) :
  is_valid_number n ↔ n ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_three_digit_mean_rearrangement_l2813_281322


namespace NUMINAMATH_CALUDE_sine_amplitude_negative_a_l2813_281344

theorem sine_amplitude_negative_a (a b : ℝ) (h1 : a < 0) (h2 : b > 0) :
  (∀ x, ∃ y, y = a * Real.sin (b * x)) →
  (∀ x, a * Real.sin (b * x) ≥ -2 ∧ a * Real.sin (b * x) ≤ 0) →
  (∃ x, a * Real.sin (b * x) = -2) →
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_sine_amplitude_negative_a_l2813_281344


namespace NUMINAMATH_CALUDE_elderly_arrangement_count_l2813_281378

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem elderly_arrangement_count :
  let volunteers : ℕ := 5
  let elderly : ℕ := 2
  let total_units : ℕ := volunteers + 1  -- Treating elderly as one unit
  let total_arrangements : ℕ := factorial total_units * factorial elderly
  let end_arrangements : ℕ := 2 * factorial (total_units - 1) * factorial elderly
  total_arrangements - end_arrangements = 960 := by
  sorry

end NUMINAMATH_CALUDE_elderly_arrangement_count_l2813_281378


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_845_l2813_281316

theorem sqrt_expression_equals_sqrt_845 :
  Real.sqrt 80 - 3 * Real.sqrt 5 + Real.sqrt 720 / Real.sqrt 3 = Real.sqrt 845 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_845_l2813_281316


namespace NUMINAMATH_CALUDE_foreign_language_ratio_l2813_281354

theorem foreign_language_ratio (M F : ℕ) (h1 : M > 0) (h2 : F > 0) : 
  (3 * M + 4 * F : ℚ) / (5 * M + 6 * F) = 19 / 30 → M = F :=
by sorry

end NUMINAMATH_CALUDE_foreign_language_ratio_l2813_281354


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_6_with_digit_sum_12_l2813_281326

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := 
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem smallest_four_digit_multiple_of_6_with_digit_sum_12 :
  ∀ n : ℕ, is_four_digit n → n % 6 = 0 → digit_sum n = 12 → n ≥ 1020 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_6_with_digit_sum_12_l2813_281326


namespace NUMINAMATH_CALUDE_boat_return_time_boat_return_time_example_l2813_281370

/-- The time taken for a boat to return upstream along a riverbank, given its downstream travel details and river flow speeds. -/
theorem boat_return_time (downstream_speed : ℝ) (downstream_time : ℝ) (downstream_distance : ℝ)
  (main_flow_speed : ℝ) (bank_flow_speed : ℝ) : ℝ :=
  let boat_speed := downstream_speed - main_flow_speed
  let upstream_speed := boat_speed - bank_flow_speed
  downstream_distance / upstream_speed

/-- The boat's return time is 20 hours given the specified conditions. -/
theorem boat_return_time_example : 
  boat_return_time 36 10 360 10 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_boat_return_time_boat_return_time_example_l2813_281370


namespace NUMINAMATH_CALUDE_trig_expression_equals_sqrt2_over_2_l2813_281366

theorem trig_expression_equals_sqrt2_over_2 :
  (Real.sin (20 * π / 180)) * Real.sqrt (1 + Real.cos (40 * π / 180)) / (Real.cos (50 * π / 180)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_sqrt2_over_2_l2813_281366


namespace NUMINAMATH_CALUDE_vector_dot_product_theorem_l2813_281348

def orthogonal_unit_vectors (i j : ℝ × ℝ) : Prop :=
  i.1 * j.1 + i.2 * j.2 = 0 ∧ i.1^2 + i.2^2 = 1 ∧ j.1^2 + j.2^2 = 1

def vector_a (i j : ℝ × ℝ) : ℝ × ℝ := (2 * i.1 + 3 * j.1, 2 * i.2 + 3 * j.2)

def vector_b (i j : ℝ × ℝ) (m : ℝ) : ℝ × ℝ := (i.1 - m * j.1, i.2 - m * j.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem vector_dot_product_theorem (i j : ℝ × ℝ) (m : ℝ) :
  orthogonal_unit_vectors i j →
  dot_product (vector_a i j) (vector_b i j m) = 1 →
  m = 1/3 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_theorem_l2813_281348


namespace NUMINAMATH_CALUDE_three_four_five_pythagorean_triple_l2813_281323

/-- A Pythagorean triple is a set of three positive integers a, b, and c that satisfy a² + b² = c² -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The set (3, 4, 5) is a Pythagorean triple -/
theorem three_four_five_pythagorean_triple : is_pythagorean_triple 3 4 5 := by
  sorry

end NUMINAMATH_CALUDE_three_four_five_pythagorean_triple_l2813_281323


namespace NUMINAMATH_CALUDE_quadratic_root_l2813_281333

theorem quadratic_root (a b c : ℚ) (r : ℝ) : 
  a ≠ 0 → 
  r = 2 * Real.sqrt 2 - 3 → 
  a * r^2 + b * r + c = 0 → 
  a * (1 : ℝ) = 1 ∧ b = 6 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_l2813_281333


namespace NUMINAMATH_CALUDE_fish_population_estimate_l2813_281379

/-- Estimates the number of fish in a lake using the capture-recapture technique -/
theorem fish_population_estimate (tagged_april : ℕ) (captured_august : ℕ) (tagged_recaptured : ℕ)
  (tagged_survival_rate : ℝ) (original_fish_rate : ℝ) :
  tagged_april = 100 →
  captured_august = 100 →
  tagged_recaptured = 5 →
  tagged_survival_rate = 0.7 →
  original_fish_rate = 0.8 →
  ∃ (estimated_population : ℕ), estimated_population = 1120 :=
by sorry

end NUMINAMATH_CALUDE_fish_population_estimate_l2813_281379


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2813_281346

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (a * b) + 1 / (a * (a - b)) ≥ 4 :=
by sorry

theorem equality_condition (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (a * b) + 1 / (a * (a - b)) = 4 ↔ a = Real.sqrt 2 ∧ b = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2813_281346


namespace NUMINAMATH_CALUDE_box_volume_increase_l2813_281331

/-- Given a rectangular box with length l, width w, and height h, 
    if the volume is 3000, surface area is 1380, and sum of edges is 160,
    then increasing each dimension by 2 results in a volume of 4548 --/
theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 3000)
  (surface_area : 2 * (l * w + w * h + l * h) = 1380)
  (edge_sum : 4 * (l + w + h) = 160) :
  (l + 2) * (w + 2) * (h + 2) = 4548 := by
  sorry


end NUMINAMATH_CALUDE_box_volume_increase_l2813_281331


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_two_l2813_281329

theorem circle_area_with_diameter_two (π : Real) : Real :=
  let diameter : Real := 2
  let radius : Real := diameter / 2
  let area : Real := π * radius^2
  area

#check circle_area_with_diameter_two

end NUMINAMATH_CALUDE_circle_area_with_diameter_two_l2813_281329


namespace NUMINAMATH_CALUDE_direct_variation_problem_l2813_281377

/-- A function representing the relationship between x and y -/
def f (k : ℝ) (y : ℝ) : ℝ := k * y^2

theorem direct_variation_problem (k : ℝ) :
  f k 1 = 6 → f k 4 = 96 := by
  sorry

end NUMINAMATH_CALUDE_direct_variation_problem_l2813_281377


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2813_281353

theorem min_value_of_expression (a : ℝ) (h : a > 1) : 
  a + 1 / (a - 1) ≥ 3 ∧ ∃ a₀ > 1, a₀ + 1 / (a₀ - 1) = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2813_281353


namespace NUMINAMATH_CALUDE_sum_of_squares_equals_two_l2813_281345

/-- Given a 2x2 matrix B with specific properties, prove that the sum of squares of its elements is 2 -/
theorem sum_of_squares_equals_two (x y z w : ℝ) :
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]
  (B.transpose = B⁻¹) →
  (x^2 + y^2 = 1) →
  (z^2 + w^2 = 1) →
  (y + z = 1/2) →
  (x^2 + y^2 + z^2 + w^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_equals_two_l2813_281345


namespace NUMINAMATH_CALUDE_simplify_expression_l2813_281310

theorem simplify_expression : (27 * (10 ^ 9)) / (9 * (10 ^ 5)) = 30000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2813_281310


namespace NUMINAMATH_CALUDE_total_tires_in_parking_lot_l2813_281363

def num_cars : ℕ := 30
def regular_tires_per_car : ℕ := 4
def spare_tires_per_car : ℕ := 1

theorem total_tires_in_parking_lot :
  (num_cars * (regular_tires_per_car + spare_tires_per_car)) = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_tires_in_parking_lot_l2813_281363


namespace NUMINAMATH_CALUDE_juan_stamp_cost_l2813_281389

/-- Represents the cost of stamps for a given country -/
structure StampCost where
  country : String
  cost : Float

/-- Represents the number of stamps for a country in a specific decade -/
structure StampCount where
  country : String
  decade : String
  count : Nat

def brazil_cost : StampCost := ⟨"Brazil", 0.07⟩
def peru_cost : StampCost := ⟨"Peru", 0.05⟩

def brazil_70s : StampCount := ⟨"Brazil", "70s", 12⟩
def brazil_80s : StampCount := ⟨"Brazil", "80s", 15⟩
def peru_70s : StampCount := ⟨"Peru", "70s", 6⟩
def peru_80s : StampCount := ⟨"Peru", "80s", 12⟩

def total_cost (costs : List StampCost) (counts : List StampCount) : Float :=
  sorry

theorem juan_stamp_cost :
  total_cost [brazil_cost, peru_cost] [brazil_70s, brazil_80s, peru_70s, peru_80s] = 2.79 :=
sorry

end NUMINAMATH_CALUDE_juan_stamp_cost_l2813_281389


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2813_281399

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem geometric_sequence_property 
  (b : ℕ → ℝ) (m n p : ℕ) 
  (h_geometric : GeometricSequence b)
  (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p)
  (h_positive : 0 < m ∧ 0 < n ∧ 0 < p) :
  (b p) ^ (m - n) * (b m) ^ (n - p) * (b n) ^ (p - m) = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2813_281399


namespace NUMINAMATH_CALUDE_force_for_18_inch_crowbar_l2813_281324

-- Define the inverse relationship between force and length
def inverse_relationship (force : ℝ) (length : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ force * length = k

-- Define the given condition
def given_condition : Prop :=
  inverse_relationship 200 12

-- Define the theorem to be proved
theorem force_for_18_inch_crowbar :
  given_condition →
  ∃ force : ℝ, inverse_relationship force 18 ∧ 
    (force ≥ 133.33 ∧ force ≤ 133.34) :=
by
  sorry

end NUMINAMATH_CALUDE_force_for_18_inch_crowbar_l2813_281324


namespace NUMINAMATH_CALUDE_president_and_vice_president_choices_l2813_281386

/-- The number of ways to choose a President and a Vice-President from a group of people -/
def choosePresidentAndVicePresident (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: There are 20 ways to choose a President and a Vice-President from a group of 5 people -/
theorem president_and_vice_president_choices :
  choosePresidentAndVicePresident 5 = 20 := by
  sorry

#eval choosePresidentAndVicePresident 5

end NUMINAMATH_CALUDE_president_and_vice_president_choices_l2813_281386


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l2813_281330

/-- Proves that a price reduction resulting in an 80% increase in sales quantity 
    and a 44% increase in total revenue corresponds to a 20% reduction in price. -/
theorem price_reduction_percentage (P : ℝ) (S : ℝ) (P_new : ℝ) 
  (h1 : P > 0) (h2 : S > 0) (h3 : P_new > 0) :
  (P_new * (S * 1.8) = P * S * 1.44) → (P_new = P * 0.8) :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l2813_281330


namespace NUMINAMATH_CALUDE_canning_box_theorem_l2813_281325

/-- Represents the solution to the canning box problem -/
def canning_box_solution (total_sheets : ℕ) (bodies_per_sheet : ℕ) (bottoms_per_sheet : ℕ) 
  (sheets_for_bodies : ℕ) (sheets_for_bottoms : ℕ) : Prop :=
  -- All sheets are used
  sheets_for_bodies + sheets_for_bottoms = total_sheets ∧
  -- Number of bodies matches half the number of bottoms
  bodies_per_sheet * sheets_for_bodies = (bottoms_per_sheet * sheets_for_bottoms) / 2 ∧
  -- Solution is optimal (no other solution exists)
  ∀ (x y : ℕ), 
    x + y = total_sheets ∧ 
    bodies_per_sheet * x = (bottoms_per_sheet * y) / 2 → 
    x ≤ sheets_for_bodies ∧ y ≤ sheets_for_bottoms

/-- The canning box theorem -/
theorem canning_box_theorem : 
  canning_box_solution 33 30 50 15 18 := by
  sorry

end NUMINAMATH_CALUDE_canning_box_theorem_l2813_281325


namespace NUMINAMATH_CALUDE_students_taking_no_subjects_l2813_281351

theorem students_taking_no_subjects (total : ℕ) (music art sports : ℕ) 
  (music_and_art music_and_sports art_and_sports : ℕ) (all_three : ℕ) : 
  total = 1200 →
  music = 60 →
  art = 80 →
  sports = 30 →
  music_and_art = 25 →
  music_and_sports = 15 →
  art_and_sports = 20 →
  all_three = 10 →
  total - (music + art + sports - music_and_art - music_and_sports - art_and_sports + all_three) = 1080 := by
  sorry

#check students_taking_no_subjects

end NUMINAMATH_CALUDE_students_taking_no_subjects_l2813_281351


namespace NUMINAMATH_CALUDE_brodys_calculator_battery_life_l2813_281305

theorem brodys_calculator_battery_life :
  ∀ (total_battery : ℝ) 
    (used_battery : ℝ) 
    (exam_duration : ℝ) 
    (remaining_battery : ℝ),
  used_battery = (3/4) * total_battery →
  exam_duration = 2 →
  remaining_battery = 13 →
  total_battery = 60 := by
sorry

end NUMINAMATH_CALUDE_brodys_calculator_battery_life_l2813_281305


namespace NUMINAMATH_CALUDE_intersection_line_correct_l2813_281320

/-- Two circles in a 2D plane -/
structure TwoCircles where
  c1 : (ℝ × ℝ) → Prop
  c2 : (ℝ × ℝ) → Prop

/-- The equation of a line in 2D -/
structure Line where
  eq : (ℝ × ℝ) → Prop

/-- Given two intersecting circles, returns the line of their intersection -/
def intersectionLine (circles : TwoCircles) : Line :=
  { eq := fun (x, y) => x + 3 * y = 0 }

theorem intersection_line_correct (circles : TwoCircles) :
  circles.c1 = fun (x, y) => x^2 + y^2 = 10 →
  circles.c2 = fun (x, y) => (x - 1)^2 + (y - 3)^2 = 20 →
  ∃ (A B : ℝ × ℝ), circles.c1 A ∧ circles.c1 B ∧ circles.c2 A ∧ circles.c2 B →
  (intersectionLine circles).eq = fun (x, y) => x + 3 * y = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_line_correct_l2813_281320


namespace NUMINAMATH_CALUDE_root_in_interval_l2813_281303

def f (x : ℝ) := x^3 - 2*x - 5

theorem root_in_interval :
  (f 2 < 0) →
  (f 3 > 0) →
  (f 2.5 > 0) →
  ∃ x, x ∈ Set.Ioo 2 2.5 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l2813_281303


namespace NUMINAMATH_CALUDE_square_equation_solution_l2813_281372

theorem square_equation_solution : ∃ (M : ℕ), M > 0 ∧ 12^2 * 30^2 = 15^2 * M^2 ∧ M = 24 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l2813_281372


namespace NUMINAMATH_CALUDE_books_on_cart_l2813_281365

/-- The number of books on a cart -/
theorem books_on_cart 
  (fiction : ℕ) 
  (non_fiction : ℕ) 
  (autobiographies : ℕ) 
  (picture : ℕ) 
  (h1 : fiction = 5)
  (h2 : non_fiction = fiction + 4)
  (h3 : autobiographies = 2 * fiction)
  (h4 : picture = 11) :
  fiction + non_fiction + autobiographies + picture = 35 := by
sorry

end NUMINAMATH_CALUDE_books_on_cart_l2813_281365


namespace NUMINAMATH_CALUDE_angle_value_proof_l2813_281391

theorem angle_value_proof (α β : Real) : 
  0 < α ∧ α < π ∧ 0 < β ∧ β < π →
  Real.tan (α - β) = 1/2 →
  Real.tan β = -1/7 →
  2*α - β = -3*π/4 := by
sorry

end NUMINAMATH_CALUDE_angle_value_proof_l2813_281391


namespace NUMINAMATH_CALUDE_plumber_pipe_cost_l2813_281364

/-- The total cost of pipes bought by a plumber -/
def total_cost (copper_length plastic_length price_per_meter : ℕ) : ℕ :=
  (copper_length + plastic_length) * price_per_meter

/-- Theorem stating the total cost for the plumber's purchase -/
theorem plumber_pipe_cost :
  let copper_length : ℕ := 10
  let plastic_length : ℕ := copper_length + 5
  let price_per_meter : ℕ := 4
  total_cost copper_length plastic_length price_per_meter = 100 := by
  sorry

end NUMINAMATH_CALUDE_plumber_pipe_cost_l2813_281364


namespace NUMINAMATH_CALUDE_consecutive_product_not_perfect_power_l2813_281371

theorem consecutive_product_not_perfect_power (n : ℕ) :
  ∀ m : ℕ, m ≥ 2 → ¬∃ k : ℕ, (n - 1) * n * (n + 1) = k^m :=
by sorry

end NUMINAMATH_CALUDE_consecutive_product_not_perfect_power_l2813_281371


namespace NUMINAMATH_CALUDE_birthday_cookies_l2813_281319

/-- The number of pans of cookies -/
def num_pans : ℕ := 5

/-- The number of cookies per pan -/
def cookies_per_pan : ℕ := 8

/-- The total number of cookies baked -/
def total_cookies : ℕ := num_pans * cookies_per_pan

theorem birthday_cookies : total_cookies = 40 := by
  sorry

end NUMINAMATH_CALUDE_birthday_cookies_l2813_281319


namespace NUMINAMATH_CALUDE_charging_pile_equation_l2813_281308

/-- Represents the growth of smart charging piles over two months -/
def charging_pile_growth (initial : ℕ) (growth_rate : ℝ) : ℝ :=
  initial * (1 + growth_rate)^2

/-- Theorem stating the relationship between the number of charging piles
    in the first and third months, given the monthly average growth rate -/
theorem charging_pile_equation (x : ℝ) : charging_pile_growth 301 x = 500 := by
  sorry

end NUMINAMATH_CALUDE_charging_pile_equation_l2813_281308


namespace NUMINAMATH_CALUDE_tacos_wanted_l2813_281349

/-- Proves the number of tacos given the cheese requirements and constraints -/
theorem tacos_wanted (cheese_per_burrito : ℕ) (cheese_per_taco : ℕ) 
  (burritos_wanted : ℕ) (total_cheese : ℕ) : ℕ :=
by
  sorry

#check tacos_wanted 4 9 7 37 = 1

end NUMINAMATH_CALUDE_tacos_wanted_l2813_281349


namespace NUMINAMATH_CALUDE_characterize_M_and_m_l2813_281340

-- Define the set S
def S : Set ℝ := {1, 2, 3, 6}

-- Define the set M
def M (m : ℝ) : Set ℝ := {x | x^2 - m*x + 6 = 0}

-- State the theorem
theorem characterize_M_and_m :
  ∀ m : ℝ, (M m ∩ S = M m) →
  ((M m = {2, 3} ∧ m = 7) ∨
   (M m = {1, 6} ∧ m = 5) ∨
   (M m = ∅ ∧ m > -2*Real.sqrt 6 ∧ m < 2*Real.sqrt 6)) :=
by sorry

end NUMINAMATH_CALUDE_characterize_M_and_m_l2813_281340


namespace NUMINAMATH_CALUDE_interchanged_digits_theorem_l2813_281343

/-- 
Given a two-digit number n = 10a + b, where n = 3(a + b),
prove that the number formed by interchanging its digits (10b + a) 
is equal to 8 times the sum of its digits (8(a + b)).
-/
theorem interchanged_digits_theorem (a b : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : a ≠ 0)
  (h4 : 10 * a + b = 3 * (a + b)) :
  10 * b + a = 8 * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_interchanged_digits_theorem_l2813_281343


namespace NUMINAMATH_CALUDE_percentage_equivalence_l2813_281394

theorem percentage_equivalence : 
  (75 / 100) * 600 = (50 / 100) * 900 := by sorry

end NUMINAMATH_CALUDE_percentage_equivalence_l2813_281394


namespace NUMINAMATH_CALUDE_olivia_payment_l2813_281336

/-- Represents the number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- Represents the number of quarters Olivia pays for chips -/
def quarters_for_chips : ℕ := 4

/-- Represents the number of quarters Olivia pays for soda -/
def quarters_for_soda : ℕ := 12

/-- Calculates the total amount Olivia pays in dollars -/
def total_payment : ℚ :=
  (quarters_for_chips + quarters_for_soda) / quarters_per_dollar

theorem olivia_payment :
  total_payment = 4 := by sorry

end NUMINAMATH_CALUDE_olivia_payment_l2813_281336


namespace NUMINAMATH_CALUDE_shopkeeper_discount_l2813_281355

theorem shopkeeper_discount (cost_price : ℝ) (h_positive : cost_price > 0) : 
  let labeled_price := cost_price * (1 + 0.4)
  let selling_price := cost_price * (1 + 0.33)
  let discount := labeled_price - selling_price
  let discount_percentage := (discount / labeled_price) * 100
  discount_percentage = 5 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_discount_l2813_281355


namespace NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l2813_281307

theorem sqrt_27_div_sqrt_3_eq_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l2813_281307


namespace NUMINAMATH_CALUDE_ball_distribution_problem_l2813_281357

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes,
    with at least one empty box -/
def distributeWithEmpty (n : ℕ) (k : ℕ) : ℕ := k * (k-1)^n

theorem ball_distribution_problem :
  distribute 6 3 - distributeWithEmpty 6 3 = 537 := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_problem_l2813_281357


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2813_281362

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, x > 2 → x^2 > 4) ∧
  (∃ x : ℝ, x^2 > 4 ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2813_281362


namespace NUMINAMATH_CALUDE_sum_of_extrema_l2813_281395

theorem sum_of_extrema (a b c : ℝ) (h1 : a + b + c = 5) (h2 : a^2 + b^2 + c^2 = 7) : 
  ∃ (n N : ℝ), (∀ x, (∃ y z, x + y + z = 5 ∧ x^2 + y^2 + z^2 = 7) → n ≤ x ∧ x ≤ N) ∧ n + N = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_extrema_l2813_281395


namespace NUMINAMATH_CALUDE_basketball_players_l2813_281358

theorem basketball_players (total : ℕ) (hockey : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 25)
  (h2 : hockey = 15)
  (h3 : neither = 4)
  (h4 : both = 10) :
  ∃ basketball : ℕ, basketball = 16 :=
by sorry

end NUMINAMATH_CALUDE_basketball_players_l2813_281358


namespace NUMINAMATH_CALUDE_mrs_lee_june_earnings_percent_l2813_281321

/-- Represents the Lee family's income situation -/
structure LeeIncome where
  may_total : ℝ
  may_mrs_lee : ℝ
  june_mrs_lee : ℝ

/-- Conditions for the Lee family's income -/
def lee_income_conditions (income : LeeIncome) : Prop :=
  income.may_mrs_lee = 0.5 * income.may_total ∧
  income.june_mrs_lee = 1.2 * income.may_mrs_lee

/-- Theorem: Mrs. Lee's earnings in June were 60% of the family's total income -/
theorem mrs_lee_june_earnings_percent (income : LeeIncome) 
  (h : lee_income_conditions income) : 
  income.june_mrs_lee / income.may_total = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_mrs_lee_june_earnings_percent_l2813_281321


namespace NUMINAMATH_CALUDE_tripled_base_and_exponent_l2813_281385

theorem tripled_base_and_exponent (a b x : ℝ) (h1 : b ≠ 0) :
  let r := (3*a)^(3*b)
  r = a^b * x^b → x = 27 * a^2 := by
sorry

end NUMINAMATH_CALUDE_tripled_base_and_exponent_l2813_281385


namespace NUMINAMATH_CALUDE_razorback_tshirt_revenue_l2813_281380

/-- The total money made by selling a given number of t-shirts at a fixed price -/
def total_money_made (num_shirts : ℕ) (price_per_shirt : ℕ) : ℕ :=
  num_shirts * price_per_shirt

/-- Theorem stating that selling 45 t-shirts at $16 each results in $720 total -/
theorem razorback_tshirt_revenue : total_money_made 45 16 = 720 := by
  sorry

end NUMINAMATH_CALUDE_razorback_tshirt_revenue_l2813_281380


namespace NUMINAMATH_CALUDE_dogwood_trees_in_park_l2813_281369

theorem dogwood_trees_in_park (current : ℕ) (planted : ℕ) (total : ℕ) : 
  planted = 49 → total = 83 → current + planted = total → current = 34 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_in_park_l2813_281369


namespace NUMINAMATH_CALUDE_decagon_diagonals_l2813_281335

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: The number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l2813_281335


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l2813_281350

theorem unique_six_digit_number : ∃! n : ℕ, 
  (100000 ≤ n ∧ n < 1000000) ∧ 
  (n / 100000 = 2) ∧ 
  ((n % 100000) * 10 + 2 = 3 * n) := by
  sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l2813_281350


namespace NUMINAMATH_CALUDE_epipen_cost_l2813_281313

/-- Proves that the cost of each EpiPen is $500, given the specified conditions -/
theorem epipen_cost (epipen_per_year : ℕ) (insurance_coverage : ℚ) (annual_payment : ℚ) :
  epipen_per_year = 2 ∧ insurance_coverage = 3/4 ∧ annual_payment = 250 →
  ∃ (cost : ℚ), cost = 500 ∧ epipen_per_year * (1 - insurance_coverage) * cost = annual_payment :=
by sorry

end NUMINAMATH_CALUDE_epipen_cost_l2813_281313


namespace NUMINAMATH_CALUDE_john_initial_money_l2813_281367

theorem john_initial_money (spent : ℕ) (left : ℕ) : 
  left = 500 → 
  spent = left + 600 → 
  spent + left = 1600 :=
by
  sorry

end NUMINAMATH_CALUDE_john_initial_money_l2813_281367


namespace NUMINAMATH_CALUDE_problem_statement_l2813_281398

theorem problem_statement (a b c : ℝ) 
  (h1 : a^2 - b^2 = 10) 
  (h2 : a * b = 5) 
  (h3 : a^2 + b^2 + c^2 = 20) : 
  a^4 + b^4 + c^4 = 150 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2813_281398


namespace NUMINAMATH_CALUDE_complement_intersection_M_N_l2813_281342

def U : Set Nat := {1, 2, 3, 4}
def M : Set Nat := {1, 2, 3}
def N : Set Nat := {2, 3, 4}

theorem complement_intersection_M_N :
  (U \ (M ∩ N)) = {1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_M_N_l2813_281342


namespace NUMINAMATH_CALUDE_wall_building_time_relation_l2813_281392

/-- Represents the time taken to build a wall given the number of workers -/
def build_time (workers : ℕ) (days : ℚ) : Prop :=
  workers * days = 180

theorem wall_building_time_relation :
  build_time 60 3 → build_time 90 2 := by
  sorry

end NUMINAMATH_CALUDE_wall_building_time_relation_l2813_281392


namespace NUMINAMATH_CALUDE_max_x_placement_l2813_281337

/-- Represents a 5x5 grid --/
def Grid := Fin 5 → Fin 5 → Bool

/-- Checks if there are three X's in a row in any direction --/
def has_three_in_a_row (g : Grid) : Bool :=
  sorry

/-- Counts the number of X's in the grid --/
def count_x (g : Grid) : Nat :=
  sorry

/-- Theorem stating the maximum number of X's that can be placed --/
theorem max_x_placement :
  ∃ (g : Grid), count_x g = 13 ∧ ¬has_three_in_a_row g ∧
  ∀ (h : Grid), count_x h > 13 → has_three_in_a_row h :=
sorry

end NUMINAMATH_CALUDE_max_x_placement_l2813_281337


namespace NUMINAMATH_CALUDE_youngest_child_age_l2813_281396

theorem youngest_child_age (n : ℕ) 
  (h1 : ∃ x : ℕ, x + (x + 2) + (x + 4) = 48)
  (h2 : ∃ y : ℕ, y + (y + 3) + (y + 6) = 60)
  (h3 : ∃ z : ℕ, z + (z + 4) = 30)
  (h4 : n = 8) :
  ∃ w : ℕ, (w = 13 ∧ w ≤ x ∧ w ≤ y ∧ w ≤ z) :=
by
  sorry

end NUMINAMATH_CALUDE_youngest_child_age_l2813_281396


namespace NUMINAMATH_CALUDE_min_value_f_l2813_281387

open Real

noncomputable def f (x : ℝ) : ℝ := (sin x + 1) * (cos x + 1) / (sin x * cos x)

theorem min_value_f :
  ∀ x ∈ Set.Ioo 0 (π/2), f x ≥ 3 + 2 * sqrt 2 ∧
  ∃ x₀ ∈ Set.Ioo 0 (π/2), f x₀ = 3 + 2 * sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_l2813_281387


namespace NUMINAMATH_CALUDE_cubic_integer_roots_l2813_281328

/-- Represents a cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ

/-- Counts the number of integer roots of a cubic polynomial, including multiplicity -/
def count_integer_roots (p : CubicPolynomial) : ℕ := sorry

/-- Theorem stating that the number of integer roots of a cubic polynomial with integer coefficients is 0, 1, 2, or 3 -/
theorem cubic_integer_roots (p : CubicPolynomial) :
  count_integer_roots p = 0 ∨ count_integer_roots p = 1 ∨ count_integer_roots p = 2 ∨ count_integer_roots p = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_integer_roots_l2813_281328


namespace NUMINAMATH_CALUDE_sequence_integer_count_l2813_281375

def sequence_term (n : ℕ) : ℚ :=
  9720 / 3^n

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem sequence_integer_count :
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    ¬is_integer (sequence_term k)) →
  (∃! (k : ℕ), k = 6 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    ¬is_integer (sequence_term k)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_integer_count_l2813_281375


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l2813_281334

theorem pure_imaginary_product (a : ℝ) : 
  (Complex.I * Complex.I = -1) →
  (∃ b : ℝ, (2 - Complex.I) * (a + 2 * Complex.I) = Complex.I * b) →
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l2813_281334


namespace NUMINAMATH_CALUDE_money_distribution_l2813_281300

/-- Given the ratios of money between Ram, Gopal, and Krishan, and Ram's amount,
    prove that Krishan has the same amount as Gopal, which is Rs. 1785. -/
theorem money_distribution (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 735 →
  gopal = 1785 ∧ krishan = 1785 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2813_281300


namespace NUMINAMATH_CALUDE_triangle_sine_relation_l2813_281318

theorem triangle_sine_relation (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_relation : 3 * Real.sin B ^ 2 + 7 * Real.sin C ^ 2 = 
                2 * Real.sin A * Real.sin B * Real.sin C + 2 * Real.sin A ^ 2) : 
  Real.sin (A + π / 4) = -Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_relation_l2813_281318


namespace NUMINAMATH_CALUDE_gcd_768_288_l2813_281314

theorem gcd_768_288 : Int.gcd 768 288 = 96 := by sorry

end NUMINAMATH_CALUDE_gcd_768_288_l2813_281314
