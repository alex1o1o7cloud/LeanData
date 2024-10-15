import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3462_346242

/-- Given an arithmetic sequence {a_n} where a_3 = 20 - a_6, prove that S_8 = 80 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (∀ n, S n = (n * (a 1 + a n)) / 2) →               -- sum formula
  a 3 = 20 - a 6 →                                   -- given condition
  S 8 = 80 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3462_346242


namespace NUMINAMATH_CALUDE_inradius_exradius_inequality_l3462_346226

/-- Given a triangle ABC with inradius r, exradius r' touching side AB, and length c of side AB,
    prove that 4rr' ≤ c^2 -/
theorem inradius_exradius_inequality (r r' c : ℝ) (hr : r > 0) (hr' : r' > 0) (hc : c > 0) :
  4 * r * r' ≤ c^2 := by
  sorry

end NUMINAMATH_CALUDE_inradius_exradius_inequality_l3462_346226


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3462_346280

theorem infinite_series_sum : 
  (∑' n : ℕ, (3 * n - 2) / (n * (n + 1) * (n + 2))) = 1/2 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3462_346280


namespace NUMINAMATH_CALUDE_final_season_premiere_l3462_346255

/-- The number of days needed to watch all episodes of a TV series -/
def days_to_watch (seasons : ℕ) (episodes_per_season : ℕ) (episodes_per_day : ℕ) : ℕ :=
  (seasons * episodes_per_season) / episodes_per_day

/-- Proof that it takes 10 days to watch all episodes -/
theorem final_season_premiere :
  days_to_watch 4 15 6 = 10 := by
  sorry

#eval days_to_watch 4 15 6

end NUMINAMATH_CALUDE_final_season_premiere_l3462_346255


namespace NUMINAMATH_CALUDE_min_value_expression_l3462_346296

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (((a^2 + b^2) * (4*a^2 + b^2)).sqrt) / (a * b) ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3462_346296


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3462_346206

theorem polynomial_expansion (x : ℝ) : (2 - x^4) * (3 + x^5) = -x^9 - 3*x^4 + 2*x^5 + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3462_346206


namespace NUMINAMATH_CALUDE_max_number_bound_l3462_346229

/-- Represents an arc on the circle with two natural numbers -/
structure Arc where
  a : ℕ
  b : ℕ

/-- Represents the circle with 1000 arcs -/
def Circle := Fin 1000 → Arc

/-- The condition that the sum of numbers on each arc is divisible by the product of numbers on the next arc -/
def valid_circle (c : Circle) : Prop :=
  ∀ i : Fin 1000, (c i).a + (c i).b ∣ (c (i + 1)).a * (c (i + 1)).b

/-- The theorem stating that the maximum number on any arc is at most 2001 -/
theorem max_number_bound (c : Circle) (h : valid_circle c) :
  ∀ i : Fin 1000, (c i).a ≤ 2001 ∧ (c i).b ≤ 2001 :=
sorry

end NUMINAMATH_CALUDE_max_number_bound_l3462_346229


namespace NUMINAMATH_CALUDE_terminal_side_quadrant_l3462_346214

-- Define the angle in degrees
def angle : ℤ := -1060

-- Define a function to convert an angle to its equivalent angle between 0° and 360°
def normalizeAngle (θ : ℤ) : ℤ :=
  θ % 360

-- Define a function to determine the quadrant of an angle
def determineQuadrant (θ : ℤ) : ℕ :=
  let normalizedAngle := normalizeAngle θ
  if 0 ≤ normalizedAngle ∧ normalizedAngle < 90 then 1
  else if 90 ≤ normalizedAngle ∧ normalizedAngle < 180 then 2
  else if 180 ≤ normalizedAngle ∧ normalizedAngle < 270 then 3
  else 4

-- Theorem statement
theorem terminal_side_quadrant :
  determineQuadrant angle = 1 := by sorry

end NUMINAMATH_CALUDE_terminal_side_quadrant_l3462_346214


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_of_100_factorial_l3462_346211

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_nonzero_digits (n : ℕ) : ℕ :=
  n % 100

theorem last_two_nonzero_digits_of_100_factorial :
  last_two_nonzero_digits (factorial 100) = 76 := by
  sorry

end NUMINAMATH_CALUDE_last_two_nonzero_digits_of_100_factorial_l3462_346211


namespace NUMINAMATH_CALUDE_value_of_expression_l3462_346290

theorem value_of_expression (x : ℝ) (h : x^2 + 3*x + 5 = 11) : 
  3*x^2 + 9*x + 12 = 30 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l3462_346290


namespace NUMINAMATH_CALUDE_toy_store_revenue_l3462_346258

theorem toy_store_revenue (D : ℝ) (h1 : D > 0) : 
  let nov := (2 / 5 : ℝ) * D
  let jan := (1 / 2 : ℝ) * nov
  let avg := (nov + jan) / 2
  D / avg = 10 / 3 := by sorry

end NUMINAMATH_CALUDE_toy_store_revenue_l3462_346258


namespace NUMINAMATH_CALUDE_parametric_to_standard_equation_l3462_346276

theorem parametric_to_standard_equation :
  ∀ (x y : ℝ), (∃ t : ℝ, x = 4 * t + 1 ∧ y = -2 * t - 5) → x + 2 * y + 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_standard_equation_l3462_346276


namespace NUMINAMATH_CALUDE_inequalities_truth_l3462_346263

theorem inequalities_truth (a b c d : ℝ) : 
  (a^2 + b^2 + c^2 ≥ a*b + b*c + a*c) ∧ 
  (a*(1 - a) ≤ (1/4 : ℝ)) ∧ 
  ((a^2 + b^2)*(c^2 + d^2) ≥ (a*c + b*d)^2) ∧
  ¬(∀ (a b : ℝ), a/b + b/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_truth_l3462_346263


namespace NUMINAMATH_CALUDE_cone_slant_height_l3462_346288

/-- The slant height of a cone with base radius 6 cm and lateral surface sector angle 240° is 9 cm. -/
theorem cone_slant_height (base_radius : ℝ) (sector_angle : ℝ) (slant_height : ℝ) : 
  base_radius = 6 →
  sector_angle = 240 →
  slant_height = (360 / sector_angle) * base_radius →
  slant_height = 9 := by
sorry

end NUMINAMATH_CALUDE_cone_slant_height_l3462_346288


namespace NUMINAMATH_CALUDE_inverse_log_inequality_l3462_346209

theorem inverse_log_inequality (n : ℝ) (h1 : n ≥ 2) :
  (1 / Real.log n) > (1 / (n - 1) - 1 / (n + 1)) :=
by
  -- Proof goes here
  sorry

-- Given condition
axiom log_inequality (x : ℝ) (h : x > 1) : Real.log x < x - 1

end NUMINAMATH_CALUDE_inverse_log_inequality_l3462_346209


namespace NUMINAMATH_CALUDE_smallest_sum_proof_l3462_346233

theorem smallest_sum_proof : 
  let sums : List ℚ := [1/4 + 1/5, 1/4 + 1/6, 1/4 + 1/9, 1/4 + 1/8, 1/4 + 1/7]
  (∀ s ∈ sums, 1/4 + 1/9 ≤ s) ∧ (1/4 + 1/9 = 13/36) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_proof_l3462_346233


namespace NUMINAMATH_CALUDE_beta_value_l3462_346259

theorem beta_value (α β : Real) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2)
  (h_sin_α : Real.sin α = Real.sqrt 5 / 5)
  (h_sin_α_β : Real.sin (α - β) = -(Real.sqrt 10) / 10) :
  β = π/4 := by
sorry

end NUMINAMATH_CALUDE_beta_value_l3462_346259


namespace NUMINAMATH_CALUDE_concentric_circles_chord_theorem_l3462_346274

/-- Represents two concentric circles with chords of the outer circle tangent to the inner circle -/
structure ConcentricCircles where
  outer : ℝ → ℝ → Prop
  inner : ℝ → ℝ → Prop
  is_concentric : Prop
  tangent_chords : Prop

/-- The angle between two adjacent chords -/
def chord_angle (c : ConcentricCircles) : ℝ := 60

/-- The number of chords needed to complete a full circle -/
def num_chords (c : ConcentricCircles) : ℕ := 3

theorem concentric_circles_chord_theorem (c : ConcentricCircles) :
  chord_angle c = 60 → num_chords c = 3 := by sorry

end NUMINAMATH_CALUDE_concentric_circles_chord_theorem_l3462_346274


namespace NUMINAMATH_CALUDE_min_value_expression_l3462_346287

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (m : ℝ), m = Real.sqrt 3 ∧ 
  (∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
    x^2 + x*y + y^2 + 1/(x+y)^2 ≥ m) ∧
  (∃ (u v : ℝ) (hu : u > 0) (hv : v > 0), 
    u^2 + u*v + v^2 + 1/(u+v)^2 = m) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3462_346287


namespace NUMINAMATH_CALUDE_hockey_league_games_l3462_346228

theorem hockey_league_games (n : ℕ) (k : ℕ) (h1 : n = 18) (h2 : k = 10) :
  (n * (n - 1) / 2) * k = 1530 :=
by sorry

end NUMINAMATH_CALUDE_hockey_league_games_l3462_346228


namespace NUMINAMATH_CALUDE_small_planks_count_l3462_346222

/-- Represents the number of planks used in building a house wall. -/
structure Planks where
  total : ℕ
  large : ℕ
  small : ℕ

/-- Theorem stating that given 29 total planks and 12 large planks, the number of small planks is 17. -/
theorem small_planks_count (p : Planks) (h1 : p.total = 29) (h2 : p.large = 12) : p.small = 17 := by
  sorry

end NUMINAMATH_CALUDE_small_planks_count_l3462_346222


namespace NUMINAMATH_CALUDE_total_wax_sticks_is_42_l3462_346286

/-- Calculates the total number of wax sticks used for animal sculptures --/
def total_wax_sticks (large_animal_wax : ℕ) (small_animal_wax : ℕ) (small_animal_total_wax : ℕ) : ℕ :=
  let small_animals := small_animal_total_wax / small_animal_wax
  let large_animals := small_animals / 5
  let large_animal_total_wax := large_animals * large_animal_wax
  small_animal_total_wax + large_animal_total_wax

/-- Theorem stating that the total number of wax sticks used is 42 --/
theorem total_wax_sticks_is_42 :
  total_wax_sticks 6 3 30 = 42 :=
by
  sorry

#eval total_wax_sticks 6 3 30

end NUMINAMATH_CALUDE_total_wax_sticks_is_42_l3462_346286


namespace NUMINAMATH_CALUDE_min_gb_for_y_cheaper_l3462_346218

/-- Cost of Plan X in cents for g gigabytes -/
def cost_x (g : ℕ) : ℕ := 15 * g

/-- Cost of Plan Y in cents for g gigabytes -/
def cost_y (g : ℕ) : ℕ :=
  if g ≤ 500 then
    3000 + 8 * g
  else
    3000 + 8 * 500 + 6 * (g - 500)

/-- Predicate to check if Plan Y is cheaper than Plan X for g gigabytes -/
def y_cheaper_than_x (g : ℕ) : Prop :=
  cost_y g < cost_x g

theorem min_gb_for_y_cheaper :
  ∀ g : ℕ, g < 778 → ¬(y_cheaper_than_x g) ∧
  y_cheaper_than_x 778 :=
sorry

end NUMINAMATH_CALUDE_min_gb_for_y_cheaper_l3462_346218


namespace NUMINAMATH_CALUDE_garden_area_increase_l3462_346299

/-- Given a rectangular garden with length 60 feet and width 20 feet,
    prove that reshaping it into a square garden with the same perimeter
    increases the area by 400 square feet. -/
theorem garden_area_increase :
  let rect_length : ℝ := 60
  let rect_width : ℝ := 20
  let rect_perimeter : ℝ := 2 * (rect_length + rect_width)
  let square_side : ℝ := rect_perimeter / 4
  let rect_area : ℝ := rect_length * rect_width
  let square_area : ℝ := square_side * square_side
  square_area - rect_area = 400 := by
sorry


end NUMINAMATH_CALUDE_garden_area_increase_l3462_346299


namespace NUMINAMATH_CALUDE_maggie_earnings_l3462_346293

def subscriptions_to_parents : ℕ := 4
def subscriptions_to_grandfather : ℕ := 1
def subscriptions_to_next_door : ℕ := 2
def subscriptions_to_another : ℕ := 4

def base_pay : ℚ := 5
def family_bonus : ℚ := 2
def neighbor_bonus : ℚ := 1
def additional_bonus_base : ℚ := 10
def additional_bonus_per_extra : ℚ := 0.5

def total_subscriptions : ℕ := subscriptions_to_parents + subscriptions_to_grandfather + subscriptions_to_next_door + subscriptions_to_another

def family_subscriptions : ℕ := subscriptions_to_parents + subscriptions_to_grandfather
def neighbor_subscriptions : ℕ := subscriptions_to_next_door + subscriptions_to_another

def base_earnings : ℚ := base_pay * total_subscriptions
def family_bonus_earnings : ℚ := family_bonus * family_subscriptions
def neighbor_bonus_earnings : ℚ := neighbor_bonus * neighbor_subscriptions

def additional_bonus : ℚ :=
  if total_subscriptions > 10
  then additional_bonus_base + additional_bonus_per_extra * (total_subscriptions - 10)
  else 0

def total_earnings : ℚ := base_earnings + family_bonus_earnings + neighbor_bonus_earnings + additional_bonus

theorem maggie_earnings : total_earnings = 81.5 := by sorry

end NUMINAMATH_CALUDE_maggie_earnings_l3462_346293


namespace NUMINAMATH_CALUDE_factors_of_50400_l3462_346223

theorem factors_of_50400 : Nat.card (Nat.divisors 50400) = 108 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_50400_l3462_346223


namespace NUMINAMATH_CALUDE_sin_585_degrees_l3462_346285

theorem sin_585_degrees : Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l3462_346285


namespace NUMINAMATH_CALUDE_unique_integer_property_l3462_346225

theorem unique_integer_property : ∃! (n : ℕ), n > 0 ∧ 2000 * n + 1 = 33 * n := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_property_l3462_346225


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_24_l3462_346267

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_8_with_digit_sum_24 :
  ∀ n : ℕ, is_three_digit n → n % 8 = 0 → sum_of_digits n = 24 → n ≤ 888 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_24_l3462_346267


namespace NUMINAMATH_CALUDE_science_book_pages_l3462_346224

/-- Given information about the number of pages in different books -/
structure BookPages where
  history : ℕ
  novel : ℕ
  science : ℕ
  novel_half_of_history : novel = history / 2
  science_four_times_novel : science = 4 * novel
  history_pages : history = 300

/-- Theorem stating that the science book has 600 pages -/
theorem science_book_pages (b : BookPages) : b.science = 600 := by
  sorry

end NUMINAMATH_CALUDE_science_book_pages_l3462_346224


namespace NUMINAMATH_CALUDE_f_satisfies_equation_l3462_346251

/-- The function f satisfying the given functional equation -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ -0.5 then 1 / (x + 0.5) else 0.5

/-- Theorem stating that f satisfies the functional equation for all real x -/
theorem f_satisfies_equation : ∀ x : ℝ, f x - (x - 0.5) * f (-x - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_equation_l3462_346251


namespace NUMINAMATH_CALUDE_rational_number_conditions_l3462_346249

theorem rational_number_conditions (a b : ℚ) : 
  a ≠ 0 → b ≠ 0 → abs a = a → abs b = -b → a + b < 0 → 
  ∃ (a b : ℚ), a = 1 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_rational_number_conditions_l3462_346249


namespace NUMINAMATH_CALUDE_parallelogram_area_l3462_346281

/-- The area of a parallelogram with given side lengths and included angle -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (ha : a = 12) (hb : b = 18) (hθ : θ = 45 * π / 180) :
  abs (a * b * Real.sin θ - 152.73) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3462_346281


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3462_346292

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: If a_6 = S_3 = 12 in an arithmetic sequence, then a_8 = 16 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 6 = 12) (h2 : seq.S 3 = 12) : seq.a 8 = 16 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3462_346292


namespace NUMINAMATH_CALUDE_function_inequality_l3462_346294

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x ∈ Set.Ioo (-π/2) (π/2), (deriv f x) * cos x + f x * sin x > 0) :
  Real.sqrt 2 * f (-π/3) < f (-π/4) := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l3462_346294


namespace NUMINAMATH_CALUDE_triangle_special_area_implies_30_degree_angle_l3462_346232

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area of the triangle is (a² + b² - c²) / (4√3),
    then angle C equals 30°. -/
theorem triangle_special_area_implies_30_degree_angle
  (a b c : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : (a^2 + b^2 - c^2) / (4 * Real.sqrt 3) = 1/2 * a * b * Real.sin (Real.pi / 6)) :
  ∃ A B C : ℝ,
    0 < A ∧ A < Real.pi ∧
    0 < B ∧ B < Real.pi ∧
    0 < C ∧ C < Real.pi ∧
    A + B + C = Real.pi ∧
    C = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_special_area_implies_30_degree_angle_l3462_346232


namespace NUMINAMATH_CALUDE_problem_solution_l3462_346275

theorem problem_solution (A B : ℝ) 
  (h1 : A + 2 * B = 814.8)
  (h2 : A = 10 * B) : 
  A - B = 611.1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3462_346275


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_half_l3462_346245

theorem opposite_of_negative_one_half : 
  -((-1 : ℚ) / 2) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_half_l3462_346245


namespace NUMINAMATH_CALUDE_team_captain_selection_l3462_346250

def total_team_size : ℕ := 15
def shortlisted_size : ℕ := 5
def captains_to_choose : ℕ := 4

theorem team_captain_selection :
  (Nat.choose total_team_size captains_to_choose) -
  (Nat.choose (total_team_size - shortlisted_size) captains_to_choose) = 1155 :=
by sorry

end NUMINAMATH_CALUDE_team_captain_selection_l3462_346250


namespace NUMINAMATH_CALUDE_coin_balance_problem_l3462_346269

theorem coin_balance_problem :
  ∃ (a b c : ℕ),
    a + b + c = 99 ∧
    2 * a + 3 * b + c = 297 ∧
    3 * a + b + 2 * c = 297 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_balance_problem_l3462_346269


namespace NUMINAMATH_CALUDE_range_of_a_l3462_346240

/-- Proposition p: The function y=(a-1)^x is increasing with respect to x -/
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (a - 1) ^ x < (a - 1) ^ y

/-- Proposition q: The inequality -3^x ≤ a is true for all positive real numbers x -/
def q (a : ℝ) : Prop := ∀ x : ℝ, x > 0 → -3 ^ x ≤ a

/-- The range of a given the conditions -/
theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) : -1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3462_346240


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squares_existence_and_uniqueness_l3462_346262

/-- Proves the existence and uniqueness of k that satisfies the given conditions --/
theorem arithmetic_sequence_squares_existence_and_uniqueness :
  ∃! k : ℤ, ∃ n d : ℤ,
    (n - d)^2 = 36 + k ∧
    n^2 = 300 + k ∧
    (n + d)^2 = 596 + k ∧
    k = 925 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squares_existence_and_uniqueness_l3462_346262


namespace NUMINAMATH_CALUDE_calculate_expression_l3462_346277

theorem calculate_expression : (-3 : ℚ) * (1/3) / (-1/3) * 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3462_346277


namespace NUMINAMATH_CALUDE_triangle_theorem_l3462_346273

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b * Real.cos t.C = (2 * t.a - t.c) * Real.cos t.B)
  (h2 : t.b = Real.sqrt 7)
  (h3 : t.a + t.c = 4) :
  t.B = π / 3 ∧ 
  ((t.a = 1 ∧ t.c = 3) ∨ (t.a = 3 ∧ t.c = 1)) := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l3462_346273


namespace NUMINAMATH_CALUDE_range_of_a_l3462_346260

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 8*x - 20 < 0 → x^2 - 2*x + 1 - a^2 ≤ 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x + 1 - a^2 ≤ 0 ∧ x^2 - 8*x - 20 ≥ 0) ∧
  (a > 0) →
  a ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3462_346260


namespace NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_sin_cos_sum_equals_three_fourths_plus_quarter_sin_seventy_l3462_346282

-- Part 1
theorem sin_product_equals_one_sixteenth :
  Real.sin (6 * π / 180) * Real.sin (42 * π / 180) * Real.sin (66 * π / 180) * Real.sin (78 * π / 180) = 1 / 16 := by
  sorry

-- Part 2
theorem sin_cos_sum_equals_three_fourths_plus_quarter_sin_seventy :
  Real.sin (20 * π / 180) ^ 2 + Real.cos (50 * π / 180) ^ 2 + Real.sin (20 * π / 180) * Real.cos (50 * π / 180) =
  3 / 4 + (1 / 4) * Real.sin (70 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_sin_cos_sum_equals_three_fourths_plus_quarter_sin_seventy_l3462_346282


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3462_346208

theorem quadratic_two_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + m = 0 ∧ y^2 + 2*y + m = 0) ↔ m < 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3462_346208


namespace NUMINAMATH_CALUDE_lcm_180_504_l3462_346291

theorem lcm_180_504 : Nat.lcm 180 504 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_lcm_180_504_l3462_346291


namespace NUMINAMATH_CALUDE_no_very_convex_function_l3462_346234

/-- A function is very convex if it satisfies the given inequality for all real x and y -/
def VeryConvex (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y|

/-- Theorem stating that no very convex function exists -/
theorem no_very_convex_function : ¬ ∃ f : ℝ → ℝ, VeryConvex f := by
  sorry


end NUMINAMATH_CALUDE_no_very_convex_function_l3462_346234


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_neg_three_range_of_a_for_interval_condition_l3462_346207

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

-- Part 1
theorem solution_set_for_a_equals_neg_three :
  {x : ℝ | f (-3) x ≥ 3} = {x : ℝ | x ≤ 1 ∨ x ≥ 4} := by sorry

-- Part 2
theorem range_of_a_for_interval_condition :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_neg_three_range_of_a_for_interval_condition_l3462_346207


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3462_346284

theorem x_plus_y_value (x y : ℚ) 
  (eq1 : 5 * x - 3 * y = 17) 
  (eq2 : 3 * x + 5 * y = 1) : 
  x + y = 21 / 17 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3462_346284


namespace NUMINAMATH_CALUDE_line_separate_from_circle_l3462_346227

/-- A point inside a circle that is not the center of the circle -/
structure PointInsideCircle (a : ℝ) where
  x₀ : ℝ
  y₀ : ℝ
  inside : x₀^2 + y₀^2 < a^2
  not_center : (x₀, y₀) ≠ (0, 0)

/-- The line determined by the point inside the circle -/
def line_equation (a : ℝ) (p : PointInsideCircle a) (x y : ℝ) : Prop :=
  p.x₀ * x + p.y₀ * y = a^2

/-- The circle equation -/
def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = a^2

theorem line_separate_from_circle (a : ℝ) (ha : a > 0) (p : PointInsideCircle a) :
  ∀ x y : ℝ, line_equation a p x y → ¬circle_equation a x y :=
sorry

end NUMINAMATH_CALUDE_line_separate_from_circle_l3462_346227


namespace NUMINAMATH_CALUDE_tree_planting_impossibility_l3462_346238

theorem tree_planting_impossibility :
  ∀ (arrangement : List ℕ),
    (arrangement.length = 50) →
    (∀ n : ℕ, n ∈ arrangement → 1 ≤ n ∧ n ≤ 25) →
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 25 → (arrangement.count n = 2)) →
    ¬(∀ n : ℕ, 1 ≤ n ∧ n ≤ 25 →
      ∃ (i j : ℕ), i < j ∧ 
        arrangement.nthLe i sorry = n ∧
        arrangement.nthLe j sorry = n ∧
        (j - i = 2 ∨ j - i = 4)) :=
by sorry

end NUMINAMATH_CALUDE_tree_planting_impossibility_l3462_346238


namespace NUMINAMATH_CALUDE_min_value_of_function_l3462_346200

theorem min_value_of_function (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (1, y - 1)
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- a ⊥ b condition
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 1 / x' + 4 / y' ≥ 1 / x + 4 / y) →
  1 / x + 4 / y = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3462_346200


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3462_346241

/-- A geometric sequence with positive first term and a_2 * a_4 = 25 has a_3 = 5 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h_geom : ∃ q : ℝ, ∀ n, a (n + 1) = a n * q)
  (h_pos : a 1 > 0) (h_prod : a 2 * a 4 = 25) : a 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3462_346241


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l3462_346246

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -1) : 
  x^2 + y^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l3462_346246


namespace NUMINAMATH_CALUDE_ratio_of_a_to_c_l3462_346266

theorem ratio_of_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 8) :
  a / c = 7.5 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_a_to_c_l3462_346266


namespace NUMINAMATH_CALUDE_calculator_transformation_l3462_346204

/-- Transformation function for the calculator -/
def transform (a b : Int) : Int × Int :=
  match (a + b) % 4 with
  | 0 => (a + 1, b)
  | 1 => (a, b + 1)
  | 2 => (a - 1, b)
  | _ => (a, b - 1)

/-- Apply the transformation n times -/
def transformN (n : Nat) (a b : Int) : Int × Int :=
  match n with
  | 0 => (a, b)
  | n + 1 => 
    let (x, y) := transformN n a b
    transform x y

theorem calculator_transformation :
  transformN 6 1 12 = (-2, 15) :=
by sorry

end NUMINAMATH_CALUDE_calculator_transformation_l3462_346204


namespace NUMINAMATH_CALUDE_quadratic_point_theorem_l3462_346219

/-- A quadratic function f(x) = ax^2 + bx + c passing through (2, 6) -/
def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 4

/-- The theorem stating that if f(2) = 6, then 2a - 3b + 4c = 29 -/
theorem quadratic_point_theorem : f 2 = 6 → 2 * 2 - 3 * (-3) + 4 * 4 = 29 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_theorem_l3462_346219


namespace NUMINAMATH_CALUDE_invalid_formula_l3462_346247

def sequence_formula_a (n : ℕ) : ℚ := n
def sequence_formula_b (n : ℕ) : ℚ := n^3 - 6*n^2 + 12*n - 6
def sequence_formula_c (n : ℕ) : ℚ := (1/2)*n^2 - (1/2)*n + 1
def sequence_formula_d (n : ℕ) : ℚ := 6 / (n^2 - 6*n + 11)

theorem invalid_formula :
  (sequence_formula_a 1 = 1 ∧ sequence_formula_a 2 = 2 ∧ sequence_formula_a 3 = 3) ∧
  (sequence_formula_b 1 = 1 ∧ sequence_formula_b 2 = 2 ∧ sequence_formula_b 3 = 3) ∧
  (sequence_formula_d 1 = 1 ∧ sequence_formula_d 2 = 2 ∧ sequence_formula_d 3 = 3) ∧
  ¬(sequence_formula_c 1 = 1 ∧ sequence_formula_c 2 = 2 ∧ sequence_formula_c 3 = 3) :=
by sorry

end NUMINAMATH_CALUDE_invalid_formula_l3462_346247


namespace NUMINAMATH_CALUDE_augmented_matrix_proof_l3462_346212

def system_of_equations : List (List ℝ) := [[1, -2, 5], [3, 1, 8]]

theorem augmented_matrix_proof :
  let eq1 := λ x y : ℝ => x - 2*y = 5
  let eq2 := λ x y : ℝ => 3*x + y = 8
  system_of_equations = 
    (λ (f g : ℝ → ℝ → ℝ) => 
      [[f 1 (-2), f (-2) 1, 5],
       [g 3 1, g 1 3, 8]])
    (λ a b => a)
    (λ a b => b) := by sorry

end NUMINAMATH_CALUDE_augmented_matrix_proof_l3462_346212


namespace NUMINAMATH_CALUDE_license_plate_count_l3462_346268

/-- Represents the number of possible digits (0-9) -/
def digit_choices : ℕ := 10

/-- Represents the number of possible letters (A-Z) -/
def letter_choices : ℕ := 26

/-- Represents the number of digits in a license plate -/
def num_digits : ℕ := 6

/-- Represents the number of adjacent letters in a license plate -/
def num_adjacent_letters : ℕ := 2

/-- Represents the number of positions for the adjacent letter pair -/
def adjacent_letter_positions : ℕ := 7

/-- Represents the number of positions for the optional letter -/
def optional_letter_positions : ℕ := 2

/-- Calculates the total number of distinct license plates -/
def total_license_plates : ℕ :=
  adjacent_letter_positions * 
  optional_letter_positions * 
  digit_choices^num_digits * 
  letter_choices^(num_adjacent_letters + 1)

/-- Theorem stating that the total number of distinct license plates is 936,520,000 -/
theorem license_plate_count : total_license_plates = 936520000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3462_346268


namespace NUMINAMATH_CALUDE_jill_final_llama_count_l3462_346297

/-- Represents the number of llamas Jill has after all operations -/
def final_llama_count (single_calf_llamas twin_calf_llamas traded_calves new_adults : ℕ) : ℕ :=
  let initial_llamas := single_calf_llamas + twin_calf_llamas
  let total_calves := single_calf_llamas + 2 * twin_calf_llamas
  let remaining_calves := total_calves - traded_calves
  let total_before_sale := initial_llamas + remaining_calves + new_adults
  total_before_sale - (total_before_sale / 3)

/-- Theorem stating that Jill ends up with 18 llamas given the initial conditions -/
theorem jill_final_llama_count :
  final_llama_count 9 5 8 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_jill_final_llama_count_l3462_346297


namespace NUMINAMATH_CALUDE_paper_pickup_sum_l3462_346278

theorem paper_pickup_sum : 127.5 + 345.25 + 518.75 = 991.5 := by
  sorry

end NUMINAMATH_CALUDE_paper_pickup_sum_l3462_346278


namespace NUMINAMATH_CALUDE_count_less_than_ten_l3462_346244

def travel_times : List Nat := [10, 12, 15, 6, 3, 8, 9]

def less_than_ten (n : Nat) : Bool := n < 10

theorem count_less_than_ten :
  (travel_times.filter less_than_ten).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_less_than_ten_l3462_346244


namespace NUMINAMATH_CALUDE_swim_meet_transport_theorem_l3462_346230

/-- Represents the transportation setup for the swimming club's trip --/
structure SwimMeetTransport where
  num_cars : Nat
  num_vans : Nat
  people_per_car : Nat
  max_people_per_car : Nat
  max_people_per_van : Nat
  additional_capacity : Nat

/-- Calculates the number of people in each van --/
def people_per_van (t : SwimMeetTransport) : Nat :=
  let total_capacity := t.num_cars * t.max_people_per_car + t.num_vans * t.max_people_per_van
  let actual_people := total_capacity - t.additional_capacity
  let people_in_cars := t.num_cars * t.people_per_car
  let people_in_vans := actual_people - people_in_cars
  people_in_vans / t.num_vans

/-- Theorem stating that the number of people in each van is 3 --/
theorem swim_meet_transport_theorem (t : SwimMeetTransport) 
  (h1 : t.num_cars = 2)
  (h2 : t.num_vans = 3)
  (h3 : t.people_per_car = 5)
  (h4 : t.max_people_per_car = 6)
  (h5 : t.max_people_per_van = 8)
  (h6 : t.additional_capacity = 17) :
  people_per_van t = 3 := by
  sorry

#eval people_per_van { 
  num_cars := 2, 
  num_vans := 3, 
  people_per_car := 5, 
  max_people_per_car := 6, 
  max_people_per_van := 8, 
  additional_capacity := 17 
}

end NUMINAMATH_CALUDE_swim_meet_transport_theorem_l3462_346230


namespace NUMINAMATH_CALUDE_frustum_cut_off_height_l3462_346261

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  originalHeight : ℝ
  frustumHeight : ℝ
  upperRadius : ℝ
  lowerRadius : ℝ

/-- Calculates the height of the smaller cone cut off from the original cone -/
def cutOffHeight (f : Frustum) : ℝ :=
  f.originalHeight - f.frustumHeight

theorem frustum_cut_off_height (f : Frustum) 
  (h1 : f.originalHeight = 30)
  (h2 : f.frustumHeight = 18)
  (h3 : f.upperRadius = 6)
  (h4 : f.lowerRadius = 10) :
  cutOffHeight f = 12 := by
sorry

end NUMINAMATH_CALUDE_frustum_cut_off_height_l3462_346261


namespace NUMINAMATH_CALUDE_certain_number_equation_l3462_346265

theorem certain_number_equation : ∃! x : ℝ, 16 * x + 17 * x + 20 * x + 11 = 170 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l3462_346265


namespace NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l3462_346201

/-- The volume of a cone formed by rolling up a three-quarter sector of a circle -/
theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 4) :
  let circumference := (3/4) * (2 * π * r)
  let base_radius := circumference / (2 * π)
  let height := Real.sqrt (r^2 - base_radius^2)
  (1/3) * π * base_radius^2 * height = 3 * π * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l3462_346201


namespace NUMINAMATH_CALUDE_convenient_logistics_boxes_l3462_346270

/-- Represents the number of large boxes -/
def large_boxes : ℕ := 8

/-- Represents the number of small boxes -/
def small_boxes : ℕ := 21 - large_boxes

/-- The total number of bottles -/
def total_bottles : ℕ := 2000

/-- The capacity of a large box -/
def large_box_capacity : ℕ := 120

/-- The capacity of a small box -/
def small_box_capacity : ℕ := 80

/-- The total number of boxes -/
def total_boxes : ℕ := 21

theorem convenient_logistics_boxes :
  large_boxes * large_box_capacity + small_boxes * small_box_capacity = total_bottles ∧
  large_boxes + small_boxes = total_boxes :=
by sorry

end NUMINAMATH_CALUDE_convenient_logistics_boxes_l3462_346270


namespace NUMINAMATH_CALUDE_log_sum_equality_l3462_346243

theorem log_sum_equality : 
  Real.log 8 / Real.log 2 + 3 * (Real.log 4 / Real.log 2) + 
  2 * (Real.log 16 / Real.log 8) + (Real.log 64 / Real.log 4) = 44 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l3462_346243


namespace NUMINAMATH_CALUDE_ratio_problem_l3462_346213

theorem ratio_problem (a b c d : ℝ) 
  (h1 : b / a = 3)
  (h2 : d / b = 4)
  (h3 : c = (a + b) / 2) :
  (a + b + c) / (b + c + d) = 8 / 17 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3462_346213


namespace NUMINAMATH_CALUDE_specific_cube_structure_surface_area_l3462_346239

/-- A solid structure composed of unit cubes -/
structure CubeStructure :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)
  (total_cubes : ℕ)

/-- Calculate the surface area of a CubeStructure -/
def surface_area (s : CubeStructure) : ℕ :=
  2 * (s.length * s.width + s.length * s.height + s.width * s.height)

/-- Theorem stating that a specific CubeStructure has a surface area of 78 square units -/
theorem specific_cube_structure_surface_area :
  ∃ (s : CubeStructure), s.length = 5 ∧ s.width = 3 ∧ s.height = 3 ∧ s.total_cubes = 15 ∧ surface_area s = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_specific_cube_structure_surface_area_l3462_346239


namespace NUMINAMATH_CALUDE_ratio_problem_l3462_346252

theorem ratio_problem : ∀ x : ℚ, (20 : ℚ) / 1 = x / 10 → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3462_346252


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3462_346210

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3462_346210


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l3462_346279

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a rectangular floor with given dimensions and rate -/
theorem paving_cost_calculation :
  paving_cost 5.5 3.75 800 = 16500 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l3462_346279


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l3462_346205

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a^2 + b^2 = 34) : 
  a * b = 4.5 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l3462_346205


namespace NUMINAMATH_CALUDE_sum_of_roots_l3462_346257

theorem sum_of_roots (k d : ℝ) (x₁ x₂ : ℝ) (h₁ : 4 * x₁^2 - k * x₁ = d)
    (h₂ : 4 * x₂^2 - k * x₂ = d) (h₃ : x₁ ≠ x₂) (h₄ : d ≠ 0) :
  x₁ + x₂ = k / 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3462_346257


namespace NUMINAMATH_CALUDE_inequality_proof_l3462_346203

theorem inequality_proof (a b c d : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (h_sum : a*b + b*c + c*d + d*a = 1) : 
  a^3 / (b+c+d) + b^3 / (c+d+a) + c^3 / (d+a+b) + d^3 / (a+b+c) ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3462_346203


namespace NUMINAMATH_CALUDE_fractional_exponent_simplification_l3462_346253

theorem fractional_exponent_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a ^ (2 * b ^ (1/4))) / (((a * (b ^ (1/2))) ^ (1/2))) = a ^ (1/2) := by
  sorry

end NUMINAMATH_CALUDE_fractional_exponent_simplification_l3462_346253


namespace NUMINAMATH_CALUDE_larger_circle_radius_l3462_346272

-- Define the radii of the three inner circles
def r₁ : ℝ := 2
def r₂ : ℝ := 3
def r₃ : ℝ := 10

-- Define the centers of the three inner circles
variable (A B C : ℝ × ℝ)

-- Define the center and radius of the larger circle
variable (O : ℝ × ℝ)
variable (R : ℝ)

-- Define the condition that all circles are touching one another
def circles_touching (A B C : ℝ × ℝ) (r₁ r₂ r₃ : ℝ) : Prop :=
  (dist A B = r₁ + r₂) ∧ (dist B C = r₂ + r₃) ∧ (dist A C = r₁ + r₃)

-- Define the condition that the larger circle contains the three inner circles
def larger_circle_contains (O : ℝ × ℝ) (R : ℝ) (A B C : ℝ × ℝ) (r₁ r₂ r₃ : ℝ) : Prop :=
  (dist O A = R - r₁) ∧ (dist O B = R - r₂) ∧ (dist O C = R - r₃)

-- The main theorem
theorem larger_circle_radius 
  (h₁ : circles_touching A B C r₁ r₂ r₃)
  (h₂ : larger_circle_contains O R A B C r₁ r₂ r₃) :
  R = 15 := by
  sorry

end NUMINAMATH_CALUDE_larger_circle_radius_l3462_346272


namespace NUMINAMATH_CALUDE_choose_three_from_ten_l3462_346220

theorem choose_three_from_ten : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_ten_l3462_346220


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3462_346271

theorem cubic_equation_roots (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : a * b^2 + 1 = 0) :
  let f := fun x : ℝ => x / a + x^2 / b + x^3 / c - b * c
  (c > 0 → (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0)) ∧
  (c < 0 → (∃! x : ℝ, f x = 0)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3462_346271


namespace NUMINAMATH_CALUDE_area_covered_by_two_squares_l3462_346221

/-- The area covered by two congruent squares with side length 12, where one vertex of one square coincides with a vertex of the other square -/
theorem area_covered_by_two_squares (side_length : ℝ) (h1 : side_length = 12) :
  let square_area := side_length ^ 2
  let total_area := 2 * square_area - square_area
  total_area = 144 := by sorry

end NUMINAMATH_CALUDE_area_covered_by_two_squares_l3462_346221


namespace NUMINAMATH_CALUDE_balloon_difference_l3462_346216

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 6

/-- The number of balloons Jake initially brought to the park -/
def jake_initial_balloons : ℕ := 3

/-- The number of additional balloons Jake bought at the park -/
def jake_additional_balloons : ℕ := 4

/-- The total number of balloons Jake had at the park -/
def jake_total_balloons : ℕ := jake_initial_balloons + jake_additional_balloons

/-- Theorem stating the difference in balloons between Jake and Allan -/
theorem balloon_difference : jake_total_balloons - allan_balloons = 1 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l3462_346216


namespace NUMINAMATH_CALUDE_order_of_abc_l3462_346237

theorem order_of_abc (a b c : ℝ) : 
  a = (Real.exp 1)⁻¹ → 
  b = (Real.log 3) / 3 → 
  c = (Real.log 4) / 4 → 
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l3462_346237


namespace NUMINAMATH_CALUDE_cubic_sum_ge_product_sum_l3462_346217

theorem cubic_sum_ge_product_sum (u v : ℝ) (hu : 0 < u) (hv : 0 < v) :
  u^3 + v^3 ≥ u^2 * v + v^2 * u := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_ge_product_sum_l3462_346217


namespace NUMINAMATH_CALUDE_find_natural_number_A_l3462_346256

theorem find_natural_number_A : ∃ A : ℕ, 
  A > 0 ∧ 
  312 % A = 2 * (270 % A) ∧ 
  270 % A = 2 * (211 % A) ∧ 
  A = 19 := by
  sorry

end NUMINAMATH_CALUDE_find_natural_number_A_l3462_346256


namespace NUMINAMATH_CALUDE_picture_distribution_l3462_346289

theorem picture_distribution (total : ℕ) (transfer : ℕ) 
  (h_total : total = 74) (h_transfer : transfer = 6) : 
  ∃ (wang_original fang_original : ℕ),
    wang_original + fang_original = total ∧
    wang_original - transfer = fang_original + transfer ∧
    wang_original = 43 ∧ 
    fang_original = 31 := by
sorry

end NUMINAMATH_CALUDE_picture_distribution_l3462_346289


namespace NUMINAMATH_CALUDE_online_store_sales_analysis_l3462_346254

/-- Represents the daily sales volume as a function of selling price -/
def daily_sales_volume (x : ℝ) : ℝ := -2 * x + 180

/-- Represents the daily profit as a function of selling price -/
def daily_profit (x : ℝ) : ℝ := (x - 60) * (daily_sales_volume x)

/-- The original selling price -/
def original_price : ℝ := 80

/-- The cost price of each item -/
def cost_price : ℝ := 60

/-- The valid range for the selling price -/
def valid_price_range (x : ℝ) : Prop := 60 ≤ x ∧ x ≤ 80

theorem online_store_sales_analysis 
  (x : ℝ) 
  (h : valid_price_range x) :
  (daily_sales_volume x = -2 * x + 180) ∧
  (∃ x₁, daily_profit x₁ = 432 ∧ x₁ = 72) ∧
  (∃ x₂, ∀ y, valid_price_range y → daily_profit x₂ ≥ daily_profit y ∧ x₂ = 75) := by
  sorry

end NUMINAMATH_CALUDE_online_store_sales_analysis_l3462_346254


namespace NUMINAMATH_CALUDE_jessica_bank_balance_l3462_346264

theorem jessica_bank_balance (B : ℝ) : 
  B - 400 = (3/5) * B → 
  B - 400 + (1/4) * (B - 400) = 750 := by
sorry

end NUMINAMATH_CALUDE_jessica_bank_balance_l3462_346264


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_five_seven_l3462_346231

/-- Represents a repeating decimal with a single digit repeating infinitely after the decimal point. -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

theorem sum_of_repeating_decimals_five_seven :
  RepeatingDecimal 5 + RepeatingDecimal 7 = 4 / 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_five_seven_l3462_346231


namespace NUMINAMATH_CALUDE_supermarket_can_display_l3462_346202

/-- Sum of an arithmetic sequence with given parameters -/
def arithmeticSequenceSum (a₁ aₙ n : ℕ) : ℕ :=
  n * (a₁ + aₙ) / 2

/-- The problem statement -/
theorem supermarket_can_display :
  let a₁ : ℕ := 28  -- first term
  let aₙ : ℕ := 1   -- last term
  let n : ℕ := 10   -- number of terms
  arithmeticSequenceSum a₁ aₙ n = 145 := by
  sorry


end NUMINAMATH_CALUDE_supermarket_can_display_l3462_346202


namespace NUMINAMATH_CALUDE_range_of_a_l3462_346298

open Set

def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - 6*x + 8 > 0

def sufficient_not_necessary (P Q : Set ℝ) : Prop :=
  P ⊂ Q ∧ P ≠ Q

theorem range_of_a (a : ℝ) :
  (a > 0) →
  (sufficient_not_necessary {x | p x a} {x | q x}) →
  (a ≥ 4 ∨ (0 < a ∧ a ≤ 2/3)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3462_346298


namespace NUMINAMATH_CALUDE_money_distribution_solution_l3462_346295

/-- Represents the money distribution problem --/
structure MoneyDistribution where
  ann_initial : ℕ
  bill_initial : ℕ
  charlie_initial : ℕ
  bill_to_ann : ℕ
  charlie_to_bill : ℕ

/-- Checks if the money distribution results in equal amounts --/
def isEqualDistribution (md : MoneyDistribution) : Prop :=
  let ann_final := md.ann_initial + md.bill_to_ann
  let bill_final := md.bill_initial - md.bill_to_ann + md.charlie_to_bill
  let charlie_final := md.charlie_initial - md.charlie_to_bill
  ann_final = bill_final ∧ bill_final = charlie_final

/-- Theorem stating the solution to the money distribution problem --/
theorem money_distribution_solution :
  let md : MoneyDistribution := {
    ann_initial := 777,
    bill_initial := 1111,
    charlie_initial := 1555,
    bill_to_ann := 371,
    charlie_to_bill := 408
  }
  isEqualDistribution md ∧ 
  (md.ann_initial + md.bill_to_ann = 1148) ∧
  (md.bill_initial - md.bill_to_ann + md.charlie_to_bill = 1148) ∧
  (md.charlie_initial - md.charlie_to_bill = 1148) :=
by
  sorry


end NUMINAMATH_CALUDE_money_distribution_solution_l3462_346295


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l3462_346236

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, x^2 + b*x + 9 = 0 → x.im ≠ 0) ↔ -6 < b ∧ b < 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l3462_346236


namespace NUMINAMATH_CALUDE_f_is_odd_l3462_346283

noncomputable def f (x : ℝ) : ℝ := 2 * x - 1 / x

theorem f_is_odd :
  ∀ x ∈ {x : ℝ | x < 0 ∨ x > 0}, f (-x) = -f x :=
by
  sorry

end NUMINAMATH_CALUDE_f_is_odd_l3462_346283


namespace NUMINAMATH_CALUDE_chicken_pot_pie_customers_l3462_346248

/-- The number of pieces in a shepherd's pie -/
def shepherds_pie_pieces : ℕ := 4

/-- The number of pieces in a chicken pot pie -/
def chicken_pot_pie_pieces : ℕ := 5

/-- The number of customers who ordered slices of shepherd's pie -/
def shepherds_pie_customers : ℕ := 52

/-- The total number of pies sold -/
def total_pies_sold : ℕ := 29

/-- Theorem stating the number of customers who ordered slices of chicken pot pie -/
theorem chicken_pot_pie_customers : ℕ := by
  sorry

end NUMINAMATH_CALUDE_chicken_pot_pie_customers_l3462_346248


namespace NUMINAMATH_CALUDE_initial_chicken_wings_chef_initial_wings_l3462_346215

theorem initial_chicken_wings (num_friends : ℕ) (additional_wings : ℕ) (wings_per_friend : ℕ) : ℕ :=
  num_friends * wings_per_friend - additional_wings

theorem chef_initial_wings : initial_chicken_wings 4 7 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_chicken_wings_chef_initial_wings_l3462_346215


namespace NUMINAMATH_CALUDE_salesperson_earnings_theorem_l3462_346235

/-- Represents the earnings of a salesperson based on their sales. -/
structure SalespersonEarnings where
  sales : ℕ
  earnings : ℝ

/-- Represents the direct proportionality between sales and earnings. -/
def directlyProportional (e1 e2 : SalespersonEarnings) : Prop :=
  e1.sales * e2.earnings = e2.sales * e1.earnings

/-- Theorem: If earnings are directly proportional to sales, and a salesperson
    earns $180 for 15 sales, then they will earn $240 for 20 sales. -/
theorem salesperson_earnings_theorem
  (e1 e2 : SalespersonEarnings)
  (h1 : directlyProportional e1 e2)
  (h2 : e1.sales = 15)
  (h3 : e1.earnings = 180)
  (h4 : e2.sales = 20) :
  e2.earnings = 240 := by
  sorry


end NUMINAMATH_CALUDE_salesperson_earnings_theorem_l3462_346235
