import Mathlib

namespace NUMINAMATH_CALUDE_odd_factors_of_360_l1214_121423

/-- The number of odd factors of 360 -/
def num_odd_factors_360 : ℕ := sorry

/-- 360 is the number we're considering -/
def n : ℕ := 360

theorem odd_factors_of_360 : num_odd_factors_360 = 6 := by sorry

end NUMINAMATH_CALUDE_odd_factors_of_360_l1214_121423


namespace NUMINAMATH_CALUDE_papayas_needed_l1214_121403

/-- The number of papayas Jake can eat in a week -/
def jake_weekly : ℕ := 3

/-- The number of papayas Jake's brother can eat in a week -/
def brother_weekly : ℕ := 5

/-- The number of papayas Jake's father can eat in a week -/
def father_weekly : ℕ := 4

/-- The number of weeks to account for -/
def num_weeks : ℕ := 4

/-- The total number of papayas needed for the given number of weeks -/
def total_papayas : ℕ := (jake_weekly + brother_weekly + father_weekly) * num_weeks

theorem papayas_needed : total_papayas = 48 := by
  sorry

end NUMINAMATH_CALUDE_papayas_needed_l1214_121403


namespace NUMINAMATH_CALUDE_triangle_similarity_condition_l1214_121480

/-- Two triangles with side lengths a, b, c and a₁, b₁, c₁ are similar if and only if
    √(a·a₁) + √(b·b₁) + √(c·c₁) = √((a+b+c)·(a₁+b₁+c₁)) -/
theorem triangle_similarity_condition 
  (a b c a₁ b₁ c₁ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha₁ : a₁ > 0) (hb₁ : b₁ > 0) (hc₁ : c₁ > 0) :
  (∃ (k : ℝ), k > 0 ∧ a₁ = k * a ∧ b₁ = k * b ∧ c₁ = k * c) ↔ 
  Real.sqrt (a * a₁) + Real.sqrt (b * b₁) + Real.sqrt (c * c₁) = 
  Real.sqrt ((a + b + c) * (a₁ + b₁ + c₁)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_condition_l1214_121480


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l1214_121433

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  (x^5 / (y^2 + z^2 - y*z)) + (y^5 / (z^2 + x^2 - z*x)) + (z^5 / (x^2 + y^2 - x*y)) ≥ Real.sqrt 3 / 3 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  (x^5 / (y^2 + z^2 - y*z)) + (y^5 / (z^2 + x^2 - z*x)) + (z^5 / (x^2 + y^2 - x*y)) = Real.sqrt 3 / 3 ↔ 
  x = 1 / Real.sqrt 3 ∧ y = 1 / Real.sqrt 3 ∧ z = 1 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l1214_121433


namespace NUMINAMATH_CALUDE_race_distance_l1214_121499

/-- 
Proves that given the conditions of two runners A and B, 
the race distance is 160 meters.
-/
theorem race_distance (t_A t_B : ℝ) (lead : ℝ) : 
  t_A = 28 →  -- A's time
  t_B = 32 →  -- B's time
  lead = 20 → -- A's lead over B at finish
  ∃ d : ℝ, d = 160 ∧ d / t_A = (d - lead) / t_B :=
by sorry

end NUMINAMATH_CALUDE_race_distance_l1214_121499


namespace NUMINAMATH_CALUDE_increasing_function_m_range_l1214_121419

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + m

theorem increasing_function_m_range (m : ℝ) :
  (∀ x₁ x₂ : ℝ, -2 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) →
  m ∈ Set.Ici 4 :=
sorry

end NUMINAMATH_CALUDE_increasing_function_m_range_l1214_121419


namespace NUMINAMATH_CALUDE_expression_evaluation_l1214_121452

theorem expression_evaluation : 15 - 6 / (-2) + |3| * (-5) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1214_121452


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1214_121415

theorem complex_number_in_first_quadrant : 
  let z : ℂ := Complex.I / (1 + Complex.I)
  0 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1214_121415


namespace NUMINAMATH_CALUDE_min_value_of_E_l1214_121464

theorem min_value_of_E :
  ∃ (E : ℝ), (∀ (x : ℝ), |x - 4| + |E| + |x - 5| ≥ 12) ∧
  (∀ (F : ℝ), (∀ (x : ℝ), |x - 4| + |F| + |x - 5| ≥ 12) → |F| ≥ |E|) ∧
  |E| = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_E_l1214_121464


namespace NUMINAMATH_CALUDE_dye_arrangement_count_l1214_121484

/-- The number of ways to arrange 3 organic dyes, 2 inorganic dyes, and 2 additives -/
def total_arrangements : ℕ := sorry

/-- The condition that no two organic dyes are adjacent -/
def organic_not_adjacent (arrangement : List (Fin 7)) : Prop := sorry

/-- The number of valid arrangements where no two organic dyes are adjacent -/
def valid_arrangements : ℕ := sorry

theorem dye_arrangement_count :
  valid_arrangements = 1440 := by sorry

end NUMINAMATH_CALUDE_dye_arrangement_count_l1214_121484


namespace NUMINAMATH_CALUDE_min_filtrations_for_pollution_reduction_l1214_121475

theorem min_filtrations_for_pollution_reduction (initial_conc : ℝ) (final_conc : ℝ) (reduction_rate : ℝ) :
  initial_conc = 1.2 →
  final_conc = 0.2 →
  reduction_rate = 0.2 →
  ∃ n : ℕ, (n = 8 ∧ initial_conc * (1 - reduction_rate)^n ≤ final_conc ∧
    ∀ m : ℕ, m < n → initial_conc * (1 - reduction_rate)^m > final_conc) :=
by sorry

end NUMINAMATH_CALUDE_min_filtrations_for_pollution_reduction_l1214_121475


namespace NUMINAMATH_CALUDE_exponential_characterization_l1214_121426

def is_exponential (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 1 ∧ ∀ x, f x = a^x

theorem exponential_characterization (f : ℝ → ℝ) 
  (h1 : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ * f x₂)
  (h2 : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) :
  is_exponential f :=
sorry

end NUMINAMATH_CALUDE_exponential_characterization_l1214_121426


namespace NUMINAMATH_CALUDE_alcohol_solution_percentage_l1214_121471

theorem alcohol_solution_percentage (initial_volume : ℝ) (initial_percentage : ℝ) (added_alcohol : ℝ) : 
  initial_volume = 6 → 
  initial_percentage = 0.3 → 
  added_alcohol = 2.4 → 
  let final_volume := initial_volume + added_alcohol
  let final_alcohol := initial_volume * initial_percentage + added_alcohol
  final_alcohol / final_volume = 0.5 := by
sorry

end NUMINAMATH_CALUDE_alcohol_solution_percentage_l1214_121471


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l1214_121462

theorem equation_has_real_roots (K : ℝ) : ∃ x : ℝ, x = K^2 * (x - 1) * (x - 3) :=
sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l1214_121462


namespace NUMINAMATH_CALUDE_square_plus_difference_of_squares_l1214_121490

theorem square_plus_difference_of_squares (x y : ℝ) : 
  x^2 + (y - x) * (y + x) = y^2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_difference_of_squares_l1214_121490


namespace NUMINAMATH_CALUDE_absolute_value_inequality_rational_inequality_l1214_121489

-- Problem 1
theorem absolute_value_inequality (x : ℝ) :
  (|x - 2| + |2*x - 3| < 4) ↔ (1/3 < x ∧ x < 3) := by sorry

-- Problem 2
theorem rational_inequality (x : ℝ) :
  ((x^2 - 3*x) / (x^2 - x - 2) ≤ x) ↔ 
  ((-1 < x ∧ x ≤ 0) ∨ x = 1 ∨ (2 < x)) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_rational_inequality_l1214_121489


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1214_121450

/-- A quadratic function of the form (x + m - 3)(x - m) + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := (x + m - 3) * (x - m) + 3

theorem quadratic_inequality (m x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : x₁ + x₂ < 3) :
  f m x₁ > f m x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1214_121450


namespace NUMINAMATH_CALUDE_sallys_shopping_problem_l1214_121401

/-- Sally's shopping problem -/
theorem sallys_shopping_problem 
  (peaches_price_after_coupon : ℝ) 
  (coupon_value : ℝ)
  (total_spent : ℝ)
  (h1 : peaches_price_after_coupon = 12.32)
  (h2 : coupon_value = 3)
  (h3 : total_spent = 23.86) :
  total_spent - (peaches_price_after_coupon + coupon_value) = 8.54 := by
sorry

end NUMINAMATH_CALUDE_sallys_shopping_problem_l1214_121401


namespace NUMINAMATH_CALUDE_min_max_P_values_l1214_121474

/-- Given real numbers x and y satisfying the condition x - 3√(x+1) = 3√(y+2) - y,
    the expression P = x + y has a minimum value of (9 + 3√21) / 2 and a maximum value of 9 + 3√15 -/
theorem min_max_P_values (x y : ℝ) (h : x - 3 * Real.sqrt (x + 1) = 3 * Real.sqrt (y + 2) - y) :
  let P := x + y
  ∃ (P_min P_max : ℝ), P_min ≤ P ∧ P ≤ P_max ∧
    P_min = (9 + 3 * Real.sqrt 21) / 2 ∧
    P_max = 9 + 3 * Real.sqrt 15 ∧
    (∀ P' : ℝ, P_min ≤ P' ∧ P' ≤ P_max → ∃ (x' y' : ℝ), x' - 3 * Real.sqrt (x' + 1) = 3 * Real.sqrt (y' + 2) - y' ∧ P' = x' + y') :=
by
  sorry


end NUMINAMATH_CALUDE_min_max_P_values_l1214_121474


namespace NUMINAMATH_CALUDE_probability_three_two_color_l1214_121478

/-- The probability of drawing 3 balls of one color and 2 of the other from a bin with 10 black and 10 white balls -/
theorem probability_three_two_color (total_balls : ℕ) (black_balls white_balls : ℕ) (drawn_balls : ℕ) : 
  total_balls = black_balls + white_balls →
  black_balls = 10 →
  white_balls = 10 →
  drawn_balls = 5 →
  (Nat.choose total_balls drawn_balls : ℚ) * (30 : ℚ) / (43 : ℚ) = 
    (Nat.choose black_balls 3 * Nat.choose white_balls 2 + 
     Nat.choose black_balls 2 * Nat.choose white_balls 3 : ℚ) :=
by sorry

#check probability_three_two_color

end NUMINAMATH_CALUDE_probability_three_two_color_l1214_121478


namespace NUMINAMATH_CALUDE_max_value_is_eight_l1214_121424

/-- The feasible region defined by the given constraints -/
def FeasibleRegion (x y : ℝ) : Prop :=
  x + y - 7 ≤ 0 ∧ x - 3*y + 1 ≤ 0 ∧ 3*x - y - 5 ≥ 0

/-- The objective function to be maximized -/
def ObjectiveFunction (x y : ℝ) : ℝ :=
  2*x - y

/-- Theorem stating that the maximum value of the objective function is 8 -/
theorem max_value_is_eight :
  ∃ (x y : ℝ), FeasibleRegion x y ∧
    ∀ (x' y' : ℝ), FeasibleRegion x' y' →
      ObjectiveFunction x y ≥ ObjectiveFunction x' y' ∧
      ObjectiveFunction x y = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_is_eight_l1214_121424


namespace NUMINAMATH_CALUDE_cube_painting_l1214_121406

/-- Given a cube of side length n constructed from n³ smaller cubes,
    if (n-2)³ = 343 small cubes remain unpainted after some faces are painted,
    then exactly 3 faces of the large cube must have been painted. -/
theorem cube_painting (n : ℕ) (h : (n - 2)^3 = 343) :
  ∃ (painted_faces : ℕ), painted_faces = 3 ∧ painted_faces < 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_painting_l1214_121406


namespace NUMINAMATH_CALUDE_compound_molar_mass_l1214_121459

/-- Given a compound where 5 moles weigh 1170 grams, prove its molar mass is 234 grams/mole. -/
theorem compound_molar_mass (mass : ℝ) (moles : ℝ) (h1 : mass = 1170) (h2 : moles = 5) :
  mass / moles = 234 := by
sorry

end NUMINAMATH_CALUDE_compound_molar_mass_l1214_121459


namespace NUMINAMATH_CALUDE_banana_count_l1214_121488

/-- Represents the contents and costs of a fruit basket -/
structure FruitBasket where
  num_bananas : ℕ
  num_apples : ℕ
  num_strawberries : ℕ
  num_avocados : ℕ
  num_grape_bunches : ℕ
  banana_cost : ℚ
  apple_cost : ℚ
  strawberry_dozen_cost : ℚ
  avocado_cost : ℚ
  half_grape_bunch_cost : ℚ
  total_cost : ℚ

/-- Theorem stating the number of bananas in the fruit basket -/
theorem banana_count (basket : FruitBasket) 
  (h1 : basket.num_apples = 3)
  (h2 : basket.num_strawberries = 24)
  (h3 : basket.num_avocados = 2)
  (h4 : basket.num_grape_bunches = 1)
  (h5 : basket.banana_cost = 1)
  (h6 : basket.apple_cost = 2)
  (h7 : basket.strawberry_dozen_cost = 4)
  (h8 : basket.avocado_cost = 3)
  (h9 : basket.half_grape_bunch_cost = 2)
  (h10 : basket.total_cost = 28) :
  basket.num_bananas = 4 := by
  sorry

end NUMINAMATH_CALUDE_banana_count_l1214_121488


namespace NUMINAMATH_CALUDE_total_tomatoes_l1214_121457

def number_of_rows : ℕ := 30
def plants_per_row : ℕ := 10
def tomatoes_per_plant : ℕ := 20

theorem total_tomatoes : 
  number_of_rows * plants_per_row * tomatoes_per_plant = 6000 := by
  sorry

end NUMINAMATH_CALUDE_total_tomatoes_l1214_121457


namespace NUMINAMATH_CALUDE_pies_sold_is_fifteen_l1214_121498

/-- Represents the number of slices in an apple pie -/
def apple_slices : ℕ := 8

/-- Represents the number of slices in a peach pie -/
def peach_slices : ℕ := 6

/-- Represents the number of apple pie slices ordered -/
def apple_orders : ℕ := 56

/-- Represents the number of peach pie slices ordered -/
def peach_orders : ℕ := 48

/-- Calculates the total number of pies sold based on the given conditions -/
def total_pies_sold : ℕ := apple_orders / apple_slices + peach_orders / peach_slices

/-- Theorem stating that the total number of pies sold is 15 -/
theorem pies_sold_is_fifteen : total_pies_sold = 15 := by
  sorry

end NUMINAMATH_CALUDE_pies_sold_is_fifteen_l1214_121498


namespace NUMINAMATH_CALUDE_unique_prime_ending_l1214_121456

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def number (A : ℕ) : ℕ := 202100 + A

theorem unique_prime_ending :
  ∃! A : ℕ, A < 10 ∧ is_prime (number A) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_ending_l1214_121456


namespace NUMINAMATH_CALUDE_boy_bike_rest_time_l1214_121439

theorem boy_bike_rest_time 
  (total_distance : ℝ) 
  (outbound_speed inbound_speed : ℝ) 
  (total_time : ℝ) :
  total_distance = 15 →
  outbound_speed = 5 →
  inbound_speed = 3 →
  total_time = 6 →
  (total_distance / 2) / outbound_speed + 
  (total_distance / 2) / inbound_speed + 
  (total_time - (total_distance / 2) / outbound_speed - (total_distance / 2) / inbound_speed) = 2 :=
by sorry

end NUMINAMATH_CALUDE_boy_bike_rest_time_l1214_121439


namespace NUMINAMATH_CALUDE_page_lines_increase_l1214_121467

theorem page_lines_increase (original : ℕ) (increased : ℕ) (percentage : ℚ) : 
  percentage = 100/3 →
  increased = 240 →
  increased = original + (percentage / 100 * original).floor →
  increased - original = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_page_lines_increase_l1214_121467


namespace NUMINAMATH_CALUDE_sum_of_10th_degree_polynomials_l1214_121427

/-- The degree of a polynomial -/
noncomputable def degree (p : Polynomial ℝ) : ℕ := sorry

/-- A polynomial is of 10th degree -/
def is_10th_degree (p : Polynomial ℝ) : Prop := degree p = 10

theorem sum_of_10th_degree_polynomials (p q : Polynomial ℝ) 
  (hp : is_10th_degree p) (hq : is_10th_degree q) : 
  degree (p + q) ≤ 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_10th_degree_polynomials_l1214_121427


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l1214_121482

theorem product_mod_seventeen :
  (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 12 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l1214_121482


namespace NUMINAMATH_CALUDE_john_study_time_for_average_75_l1214_121411

/-- Represents the relationship between study time and test score -/
structure StudyScoreRelation where
  k : ℝ  -- Proportionality constant
  study_time : ℝ → ℝ  -- Function mapping score to study time
  score : ℝ → ℝ  -- Function mapping study time to score

/-- John's hypothesis about study time and test score -/
def john_hypothesis (r : StudyScoreRelation) : Prop :=
  ∀ t, r.score t = r.k * t

theorem john_study_time_for_average_75 
  (r : StudyScoreRelation)
  (h1 : john_hypothesis r)
  (h2 : r.score 3 = 60)  -- First exam result
  (h3 : r.k = 20)  -- Derived from first exam
  : r.study_time 90 = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_john_study_time_for_average_75_l1214_121411


namespace NUMINAMATH_CALUDE_xy_yz_xz_equals_60_l1214_121425

theorem xy_yz_xz_equals_60 
  (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (eq1 : x^2 + x*y + y^2 = 75)
  (eq2 : y^2 + y*z + z^2 = 36)
  (eq3 : z^2 + x*z + x^2 = 111) :
  x*y + y*z + x*z = 60 := by
sorry

end NUMINAMATH_CALUDE_xy_yz_xz_equals_60_l1214_121425


namespace NUMINAMATH_CALUDE_hotdog_cost_l1214_121438

theorem hotdog_cost (h s : ℕ) : 
  3 * h + 2 * s = 360 →
  2 * h + 3 * s = 390 →
  h = 60 := by sorry

end NUMINAMATH_CALUDE_hotdog_cost_l1214_121438


namespace NUMINAMATH_CALUDE_class_size_l1214_121470

theorem class_size (top_scorers : Nat) (zero_scorers : Nat) (top_score : Nat) (rest_avg : Nat) (class_avg : Nat) :
  top_scorers = 3 →
  zero_scorers = 5 →
  top_score = 95 →
  rest_avg = 45 →
  class_avg = 42 →
  ∃ (N : Nat), N = 25 ∧ 
    (N * class_avg = top_scorers * top_score + zero_scorers * 0 + (N - top_scorers - zero_scorers) * rest_avg) :=
by sorry

end NUMINAMATH_CALUDE_class_size_l1214_121470


namespace NUMINAMATH_CALUDE_range_of_a_l1214_121445

-- Define the quadratic equation
def quadratic_eq (a x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2*a + 6

-- Define the property of having one positive and one negative root
def has_pos_neg_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ 
  quadratic_eq a x₁ = 0 ∧ quadratic_eq a x₂ = 0

-- Theorem statement
theorem range_of_a (a : ℝ) :
  has_pos_neg_roots a ↔ a < -3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1214_121445


namespace NUMINAMATH_CALUDE_cades_marbles_l1214_121443

/-- The number of marbles Cade has after receiving marbles from Dylan and Ellie -/
def total_marbles (initial : ℕ) (from_dylan : ℕ) (from_ellie : ℕ) : ℕ :=
  initial + from_dylan + from_ellie

/-- Theorem stating that Cade's total marbles after receiving from Dylan and Ellie is 108 -/
theorem cades_marbles :
  total_marbles 87 8 13 = 108 := by
  sorry

end NUMINAMATH_CALUDE_cades_marbles_l1214_121443


namespace NUMINAMATH_CALUDE_incorrect_induction_proof_l1214_121466

theorem incorrect_induction_proof (n : ℕ+) : 
  ¬(∀ k : ℕ+, (∀ m : ℕ+, m < k → Real.sqrt (m^2 + m) < m + 1) → 
    Real.sqrt ((k+1)^2 + (k+1)) < (k+1) + 1) := by
  sorry

#check incorrect_induction_proof

end NUMINAMATH_CALUDE_incorrect_induction_proof_l1214_121466


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1214_121441

/-- Given a rectangle with area 540 square centimeters, if its length is decreased by 20%
    and its width is increased by 15%, then its new area is 496.8 square centimeters. -/
theorem rectangle_area_change (l w : ℝ) (h1 : l * w = 540) : 
  (0.8 * l) * (1.15 * w) = 496.8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1214_121441


namespace NUMINAMATH_CALUDE_inheritance_tax_problem_l1214_121429

theorem inheritance_tax_problem (x : ℝ) : 
  (0.25 * x + 0.15 * (x - 0.25 * x) = 15000) → x = 41379 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_tax_problem_l1214_121429


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l1214_121468

theorem banana_orange_equivalence : 
  ∀ (banana_value orange_value : ℚ),
  (3/4 : ℚ) * 12 * banana_value = 9 * orange_value →
  (2/3 : ℚ) * 6 * banana_value = 4 * orange_value :=
by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l1214_121468


namespace NUMINAMATH_CALUDE_vector_calculation_l1214_121405

def a : Fin 2 → ℚ := ![1, 1]
def b : Fin 2 → ℚ := ![1, -1]

theorem vector_calculation : (1/2 : ℚ) • a - (3/2 : ℚ) • b = ![-1, 2] := by sorry

end NUMINAMATH_CALUDE_vector_calculation_l1214_121405


namespace NUMINAMATH_CALUDE_election_win_margin_l1214_121497

theorem election_win_margin :
  ∀ (total_votes : ℕ) (winner_votes loser_votes : ℕ),
    winner_votes = 837 →
    winner_votes = (62 * total_votes) / 100 →
    loser_votes = total_votes - winner_votes →
    winner_votes - loser_votes = 324 := by
  sorry

end NUMINAMATH_CALUDE_election_win_margin_l1214_121497


namespace NUMINAMATH_CALUDE_distributive_law_addition_over_multiplication_not_hold_l1214_121473

-- Define the pair type
def Pair := ℝ × ℝ

-- Define addition operation
def add : Pair → Pair → Pair
  | (x₁, y₁), (x₂, y₂) => (x₁ + x₂, y₁ + y₂)

-- Define multiplication operation
def mul : Pair → Pair → Pair
  | (x₁, y₁), (x₂, y₂) => (x₁ * x₂ - y₁ * y₂, x₁ * y₂ + y₁ * x₂)

-- Statement: Distributive law of addition over multiplication does NOT hold
theorem distributive_law_addition_over_multiplication_not_hold :
  ∃ a b c : Pair, add a (mul b c) ≠ mul (add a b) (add a c) := by
  sorry

end NUMINAMATH_CALUDE_distributive_law_addition_over_multiplication_not_hold_l1214_121473


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1214_121463

theorem inequality_equivalence (y : ℝ) : 
  7/36 + |y - 13/72| < 11/24 ↔ -1/12 < y ∧ y < 4/9 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1214_121463


namespace NUMINAMATH_CALUDE_volume_of_cube_with_triple_surface_area_l1214_121413

noncomputable def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

noncomputable def cube_surface_area (side_length : ℝ) : ℝ := 6 * side_length ^ 2

theorem volume_of_cube_with_triple_surface_area (v₁ : ℝ) (h₁ : v₁ = 64) :
  ∃ v₂ : ℝ, 
    (∃ s₁ s₂ : ℝ, 
      cube_volume s₁ = v₁ ∧ 
      cube_surface_area s₂ = 3 * cube_surface_area s₁ ∧ 
      cube_volume s₂ = v₂) ∧ 
    v₂ = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_cube_with_triple_surface_area_l1214_121413


namespace NUMINAMATH_CALUDE_optimal_purchase_minimizes_cost_l1214_121493

/-- Represents the prices and quantities of soccer balls for two brands. -/
structure SoccerBallPurchase where
  priceA : ℝ  -- Price of brand A soccer ball
  priceB : ℝ  -- Price of brand B soccer ball
  quantityA : ℝ  -- Quantity of brand A soccer balls
  quantityB : ℝ  -- Quantity of brand B soccer balls

/-- The optimal purchase strategy for soccer balls. -/
def optimalPurchase : SoccerBallPurchase := {
  priceA := 50,
  priceB := 80,
  quantityA := 60,
  quantityB := 20
}

/-- The total cost of the purchase. -/
def totalCost (p : SoccerBallPurchase) : ℝ :=
  p.priceA * p.quantityA + p.priceB * p.quantityB

/-- Theorem stating the optimal purchase strategy minimizes cost under given conditions. -/
theorem optimal_purchase_minimizes_cost :
  let p := optimalPurchase
  (p.priceB = p.priceA + 30) ∧  -- Condition 1
  (1000 / p.priceA = 1600 / p.priceB) ∧  -- Condition 2
  (p.quantityA + p.quantityB = 80) ∧  -- Condition 3
  (p.quantityA ≥ 30) ∧  -- Condition 4
  (p.quantityA ≤ 3 * p.quantityB) ∧  -- Condition 5
  (∀ q : SoccerBallPurchase,
    (q.priceB = q.priceA + 30) →
    (1000 / q.priceA = 1600 / q.priceB) →
    (q.quantityA + q.quantityB = 80) →
    (q.quantityA ≥ 30) →
    (q.quantityA ≤ 3 * q.quantityB) →
    totalCost p ≤ totalCost q) :=
by
  sorry  -- Proof omitted

#check optimal_purchase_minimizes_cost

end NUMINAMATH_CALUDE_optimal_purchase_minimizes_cost_l1214_121493


namespace NUMINAMATH_CALUDE_yoz_perpendicular_x_xoz_perpendicular_y_xoy_perpendicular_z_l1214_121495

-- Define the three-dimensional Cartesian coordinate system
structure CartesianCoordinate3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the coordinate planes
def yoz_plane : Set CartesianCoordinate3D := {p | p.x = 0}
def xoz_plane : Set CartesianCoordinate3D := {p | p.y = 0}
def xoy_plane : Set CartesianCoordinate3D := {p | p.z = 0}

-- Define the axes
def x_axis : Set CartesianCoordinate3D := {p | p.y = 0 ∧ p.z = 0}
def y_axis : Set CartesianCoordinate3D := {p | p.x = 0 ∧ p.z = 0}
def z_axis : Set CartesianCoordinate3D := {p | p.x = 0 ∧ p.y = 0}

-- Define perpendicularity between a plane and an axis
def perpendicular (plane : Set CartesianCoordinate3D) (axis : Set CartesianCoordinate3D) : Prop :=
  ∀ p ∈ plane, ∀ q ∈ axis, (p.x - q.x) * q.x + (p.y - q.y) * q.y + (p.z - q.z) * q.z = 0

-- Theorem statements
theorem yoz_perpendicular_x : perpendicular yoz_plane x_axis := sorry

theorem xoz_perpendicular_y : perpendicular xoz_plane y_axis := sorry

theorem xoy_perpendicular_z : perpendicular xoy_plane z_axis := sorry

end NUMINAMATH_CALUDE_yoz_perpendicular_x_xoz_perpendicular_y_xoy_perpendicular_z_l1214_121495


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_one_l1214_121447

theorem sum_of_a_and_b_is_one (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : a / (1 - i) = 1 - b * i) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_one_l1214_121447


namespace NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l1214_121416

theorem arithmetic_sequence_remainder (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 3 →
  d = 8 →
  aₙ = 283 →
  n = (aₙ - a₁) / d + 1 →
  (n * (a₁ + aₙ) / 2) % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l1214_121416


namespace NUMINAMATH_CALUDE_complex_multiplication_result_l1214_121455

theorem complex_multiplication_result : 
  let i : ℂ := Complex.I
  (3 - 4*i) * (2 + 6*i) * (-1 + 2*i) = -50 + 50*i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_result_l1214_121455


namespace NUMINAMATH_CALUDE_base_conversion_645_to_base_5_l1214_121409

theorem base_conversion_645_to_base_5 :
  (1 * 5^4 + 0 * 5^3 + 4 * 5^1 + 0 * 5^0 : ℕ) = 645 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_645_to_base_5_l1214_121409


namespace NUMINAMATH_CALUDE_rectangle_area_l1214_121448

/-- Given a rectangle EFGH with vertices E(0, 0), F(0, 6), G(y, 6), and H(y, 0),
    if the area of the rectangle is 42 square units and y > 0, then y = 7. -/
theorem rectangle_area (y : ℝ) : y > 0 → (6 * y = 42) → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1214_121448


namespace NUMINAMATH_CALUDE_points_on_circle_l1214_121451

theorem points_on_circle (t : ℝ) (h : t ≠ 0) :
  ∃ (a : ℝ), (((t^2 + 1) / t)^2 + ((t^2 - 1) / t)^2) = a := by
  sorry

end NUMINAMATH_CALUDE_points_on_circle_l1214_121451


namespace NUMINAMATH_CALUDE_percent_of_percent_l1214_121442

theorem percent_of_percent : (3 / 100) / (5 / 100) * 100 = 60 := by sorry

end NUMINAMATH_CALUDE_percent_of_percent_l1214_121442


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1214_121436

theorem product_of_three_numbers (a b c : ℚ) : 
  a + b + c = 30 →
  a = 3 * (b + c) →
  b = 6 * c →
  a * b * c = 10125 / 14 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1214_121436


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_condition_l1214_121465

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = (n * (a 1 + a n)) / 2

/-- The theorem stating that m = 5 for the given conditions -/
theorem arithmetic_sequence_sum_condition (seq : ArithmeticSequence) (m : ℕ) :
  m > 1 →
  seq.S (m - 1) = -2 →
  seq.S m = 0 →
  seq.S (m + 1) = 3 →
  m = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_condition_l1214_121465


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_4pi_over_3_l1214_121428

theorem cos_2alpha_plus_4pi_over_3 (α : Real) 
  (h : Real.sqrt 3 * Real.sin α + Real.cos α = 1/2) : 
  Real.cos (2 * α + 4 * π / 3) = -7/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_4pi_over_3_l1214_121428


namespace NUMINAMATH_CALUDE_length_of_AC_l1214_121435

/-- Given a quadrilateral ABCD with specific side lengths, prove the length of AC -/
theorem length_of_AC (AB DC AD : ℝ) (h1 : AB = 12) (h2 : DC = 15) (h3 : AD = 9) :
  ∃ (AC : ℝ), AC^2 = 585 := by sorry

end NUMINAMATH_CALUDE_length_of_AC_l1214_121435


namespace NUMINAMATH_CALUDE_algebraic_expression_correct_l1214_121487

/-- The algebraic expression for the number that is 2 less than three times the cube of a and b -/
def algebraic_expression (a b : ℝ) : ℝ := 3 * (a^3 + b^3) - 2

/-- Theorem stating that the algebraic expression is correct -/
theorem algebraic_expression_correct (a b : ℝ) :
  algebraic_expression a b = 3 * (a^3 + b^3) - 2 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_correct_l1214_121487


namespace NUMINAMATH_CALUDE_intersection_nonempty_intersection_equals_B_l1214_121461

-- Define sets A and B
def A : Set ℝ := {x : ℝ | (x + 1) * (4 - x) ≤ 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 2}

-- Theorem 1
theorem intersection_nonempty (a : ℝ) : 
  (A ∩ B a).Nonempty ↔ -1/2 ≤ a ∧ a ≤ 2 :=
sorry

-- Theorem 2
theorem intersection_equals_B (a : ℝ) :
  A ∩ B a = B a ↔ a ≥ 2 ∨ a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_intersection_equals_B_l1214_121461


namespace NUMINAMATH_CALUDE_tangent_and_inequality_conditions_l1214_121400

noncomputable def f (x : ℝ) := Real.exp (2 * x)
def g (k : ℝ) (x : ℝ) := k * x + 1

theorem tangent_and_inequality_conditions (k : ℝ) :
  (∃ t : ℝ, (f t = g k t ∧ (deriv f) t = k)) ↔ k = 2 ∧
  (k > 0 → (∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, 0 < x → x < m → |f x - g k x| > 2 * x) ↔ k > 4) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_inequality_conditions_l1214_121400


namespace NUMINAMATH_CALUDE_sheet_area_difference_l1214_121476

-- Define the dimensions of the sheets in inches
def sheet1_length : ℝ := 15
def sheet1_width : ℝ := 24
def sheet2_length : ℝ := 12
def sheet2_width : ℝ := 18

-- Define the conversion factor from square inches to square feet
def sq_inches_per_sq_foot : ℝ := 144

-- Theorem statement
theorem sheet_area_difference : 
  (2 * sheet1_length * sheet1_width - 2 * sheet2_length * sheet2_width) / sq_inches_per_sq_foot = 2 := by
  sorry


end NUMINAMATH_CALUDE_sheet_area_difference_l1214_121476


namespace NUMINAMATH_CALUDE_range_of_a_l1214_121446

def proposition_p (a x : ℝ) : Prop :=
  x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0

def proposition_q (x : ℝ) : Prop :=
  x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0

def negation_p_necessary_not_sufficient_for_negation_q (a : ℝ) : Prop :=
  ∀ x, ¬(proposition_q x) → ¬(proposition_p a x) ∧
  ∃ x, ¬(proposition_p a x) ∧ proposition_q x

theorem range_of_a :
  ∀ a : ℝ, negation_p_necessary_not_sufficient_for_negation_q a →
  (a < 0 ∧ a > -4) ∨ a ≤ -4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1214_121446


namespace NUMINAMATH_CALUDE_vegetable_options_count_l1214_121472

/-- The number of cheese options available -/
def cheese_options : ℕ := 3

/-- The number of meat options available -/
def meat_options : ℕ := 4

/-- The total number of topping combinations -/
def total_combinations : ℕ := 57

/-- Calculates the number of topping combinations given the number of vegetable options -/
def calculate_combinations (veg_options : ℕ) : ℕ :=
  cheese_options * meat_options * veg_options - 
  cheese_options * (veg_options - 1) + 
  cheese_options

/-- Theorem stating that there are 5 vegetable options -/
theorem vegetable_options_count : 
  ∃ (veg_options : ℕ), veg_options = 5 ∧ calculate_combinations veg_options = total_combinations :=
sorry

end NUMINAMATH_CALUDE_vegetable_options_count_l1214_121472


namespace NUMINAMATH_CALUDE_penny_money_left_l1214_121408

/-- Calculates the amount of money Penny has left after purchasing socks and a hat. -/
def money_left (initial_amount : ℝ) (num_sock_pairs : ℕ) (sock_pair_cost : ℝ) (hat_cost : ℝ) : ℝ :=
  initial_amount - (num_sock_pairs * sock_pair_cost + hat_cost)

/-- Proves that Penny has $5 left after her purchases. -/
theorem penny_money_left :
  money_left 20 4 2 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_penny_money_left_l1214_121408


namespace NUMINAMATH_CALUDE_simplify_radical_product_l1214_121420

theorem simplify_radical_product (x : ℝ) (h : x > 0) :
  2 * Real.sqrt (50 * x^3) * Real.sqrt (45 * x^5) * Real.sqrt (98 * x^7) = 420 * x^7 * Real.sqrt (5 * x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l1214_121420


namespace NUMINAMATH_CALUDE_trig_simplification_l1214_121412

theorem trig_simplification (x y z : ℝ) :
  Real.sin (x - y + z) * Real.cos y - Real.cos (x - y + z) * Real.sin y = Real.sin (x - 2*y + z) := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l1214_121412


namespace NUMINAMATH_CALUDE_least_valid_number_l1214_121479

def is_valid (n : ℕ) : Prop :=
  n > 1 ∧
  n % 4 = 3 ∧
  n % 5 = 3 ∧
  n % 7 = 3 ∧
  n % 10 = 3 ∧
  n % 11 = 3

theorem least_valid_number : 
  is_valid 1543 ∧ ∀ m : ℕ, m < 1543 → ¬(is_valid m) :=
sorry

end NUMINAMATH_CALUDE_least_valid_number_l1214_121479


namespace NUMINAMATH_CALUDE_integral_sin_plus_sqrt_one_minus_x_squared_l1214_121469

open Real MeasureTheory

theorem integral_sin_plus_sqrt_one_minus_x_squared (f g : ℝ → ℝ) :
  (∫ x in (-1)..1, f x) = 0 →
  (∫ x in (-1)..1, g x) = π / 2 →
  (∫ x in (-1)..1, f x + g x) = π / 2 :=
by sorry

end NUMINAMATH_CALUDE_integral_sin_plus_sqrt_one_minus_x_squared_l1214_121469


namespace NUMINAMATH_CALUDE_E_parity_l1214_121404

def E : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => E (n + 1) + E n

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem E_parity : is_odd (E 2021) ∧ is_even (E 2022) ∧ is_odd (E 2023) := by sorry

end NUMINAMATH_CALUDE_E_parity_l1214_121404


namespace NUMINAMATH_CALUDE_f_min_value_l1214_121402

/-- The function f as defined in the problem -/
def f (x y : ℝ) : ℝ := x^3 + y^3 + x^2*y + x*y^2 - 3*(x^2 + y^2 + x*y) + 3*(x + y)

/-- Theorem stating that f(x,y) ≥ 1 for all x,y ≥ 1/2 -/
theorem f_min_value (x y : ℝ) (hx : x ≥ 1/2) (hy : y ≥ 1/2) : f x y ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l1214_121402


namespace NUMINAMATH_CALUDE_exists_valid_8_stamp_combination_min_8_stamps_for_valid_combination_min_stamps_is_8_l1214_121486

/-- Represents the number of stamps of each denomination -/
structure StampCombination :=
  (s06 : ℕ)  -- number of 0.6 yuan stamps
  (s08 : ℕ)  -- number of 0.8 yuan stamps
  (s11 : ℕ)  -- number of 1.1 yuan stamps

/-- The total postage value of a stamp combination -/
def postageValue (sc : StampCombination) : ℚ :=
  0.6 * sc.s06 + 0.8 * sc.s08 + 1.1 * sc.s11

/-- The total number of stamps in a combination -/
def totalStamps (sc : StampCombination) : ℕ :=
  sc.s06 + sc.s08 + sc.s11

/-- A stamp combination is valid if it exactly equals the required postage -/
def isValidCombination (sc : StampCombination) : Prop :=
  postageValue sc = 7.5

/-- There exists a valid stamp combination using 8 stamps -/
theorem exists_valid_8_stamp_combination :
  ∃ (sc : StampCombination), isValidCombination sc ∧ totalStamps sc = 8 :=
sorry

/-- Any valid stamp combination uses at least 8 stamps -/
theorem min_8_stamps_for_valid_combination :
  ∀ (sc : StampCombination), isValidCombination sc → totalStamps sc ≥ 8 :=
sorry

/-- The minimum number of stamps required for a valid combination is 8 -/
theorem min_stamps_is_8 :
  (∃ (sc : StampCombination), isValidCombination sc ∧ totalStamps sc = 8) ∧
  (∀ (sc : StampCombination), isValidCombination sc → totalStamps sc ≥ 8) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_8_stamp_combination_min_8_stamps_for_valid_combination_min_stamps_is_8_l1214_121486


namespace NUMINAMATH_CALUDE_paige_folders_l1214_121422

def initial_files : ℕ := 135
def deleted_files : ℕ := 27
def files_per_folder : ℚ := 8.5

theorem paige_folders : 
  ∃ (folders : ℕ), 
    folders = (initial_files - deleted_files : ℚ) / files_per_folder
    ∧ folders = 13 := by sorry

end NUMINAMATH_CALUDE_paige_folders_l1214_121422


namespace NUMINAMATH_CALUDE_product_negative_implies_one_less_than_one_l1214_121494

theorem product_negative_implies_one_less_than_one (a b c : ℝ) :
  (a - 1) * (b - 1) * (c - 1) < 0 →
  (a < 1) ∨ (b < 1) ∨ (c < 1) :=
by
  sorry

end NUMINAMATH_CALUDE_product_negative_implies_one_less_than_one_l1214_121494


namespace NUMINAMATH_CALUDE_total_deduction_is_111_cents_l1214_121492

-- Define the hourly wage in cents
def hourly_wage : ℚ := 2500

-- Define the tax rate
def tax_rate : ℚ := 15 / 1000

-- Define the retirement contribution rate
def retirement_rate : ℚ := 3 / 100

-- Function to calculate the total deduction
def total_deduction (wage : ℚ) (tax : ℚ) (retirement : ℚ) : ℚ :=
  let tax_amount := wage * tax
  let after_tax := wage - tax_amount
  let retirement_amount := after_tax * retirement
  tax_amount + retirement_amount

-- Theorem stating that the total deduction is 111 cents
theorem total_deduction_is_111_cents :
  ⌊total_deduction hourly_wage tax_rate retirement_rate⌋ = 111 :=
sorry

end NUMINAMATH_CALUDE_total_deduction_is_111_cents_l1214_121492


namespace NUMINAMATH_CALUDE_range_of_a_l1214_121444

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 ≥ a

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1214_121444


namespace NUMINAMATH_CALUDE_lee_cookies_l1214_121432

/-- Given that Lee can make 24 cookies with 4 cups of flour, 
    this theorem proves that he can make 30 cookies with 5 cups of flour. -/
theorem lee_cookies (cookies_per_4_cups : ℕ) (h : cookies_per_4_cups = 24) :
  (cookies_per_4_cups * 5 / 4 : ℚ) = 30 := by
  sorry


end NUMINAMATH_CALUDE_lee_cookies_l1214_121432


namespace NUMINAMATH_CALUDE_yeast_population_after_30_minutes_l1214_121414

/-- The population of yeast cells after a given time period. -/
def yeast_population (initial_population : ℕ) (time_minutes : ℕ) : ℕ :=
  initial_population * (3 ^ (time_minutes / 5))

/-- Theorem: The yeast population after 30 minutes is 36450 cells. -/
theorem yeast_population_after_30_minutes :
  yeast_population 50 30 = 36450 := by
  sorry

end NUMINAMATH_CALUDE_yeast_population_after_30_minutes_l1214_121414


namespace NUMINAMATH_CALUDE_vector_equation_solution_l1214_121458

theorem vector_equation_solution :
  let a : ℚ := -491/342
  let b : ℚ := 233/342
  let c : ℚ := 49/38
  let v1 : Fin 3 → ℚ := ![1, -2, 3]
  let v2 : Fin 3 → ℚ := ![4, 1, -1]
  let v3 : Fin 3 → ℚ := ![-3, 2, 1]
  let result : Fin 3 → ℚ := ![0, 1, 4]
  (a • v1) + (b • v2) + (c • v3) = result :=
by
  sorry


end NUMINAMATH_CALUDE_vector_equation_solution_l1214_121458


namespace NUMINAMATH_CALUDE_arithmetic_operations_l1214_121483

theorem arithmetic_operations :
  (3 - (-2) = 5) ∧
  ((-4) * (-3) = 12) ∧
  (0 / (-3) = 0) ∧
  (|(-12)| + (-4) = 8) ∧
  ((3) - 14 - (-5) + (-16) = -22) ∧
  ((-5) / (-1/5) * (-5) = -125) ∧
  (-24 * ((-5/6) + (3/8) - (1/12)) = 13) ∧
  (3 * (-4) + 18 / (-6) - (-2) = -12) ∧
  ((-99 - 15/16) * 4 = -399 - 3/4) := by
sorry

#eval 3 - (-2)
#eval (-4) * (-3)
#eval 0 / (-3)
#eval |(-12)| + (-4)
#eval 3 - 14 - (-5) + (-16)
#eval (-5) / (-1/5) * (-5)
#eval -24 * ((-5/6) + (3/8) - (1/12))
#eval 3 * (-4) + 18 / (-6) - (-2)
#eval (-99 - 15/16) * 4

end NUMINAMATH_CALUDE_arithmetic_operations_l1214_121483


namespace NUMINAMATH_CALUDE_three_numbers_in_unit_interval_l1214_121431

theorem three_numbers_in_unit_interval (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x < 1) (hy : 0 ≤ y ∧ y < 1) (hz : 0 ≤ z ∧ z < 1) :
  ∃ a b, (a = x ∨ a = y ∨ a = z) ∧ (b = x ∨ b = y ∨ b = z) ∧ a ≠ b ∧ |b - a| < (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_in_unit_interval_l1214_121431


namespace NUMINAMATH_CALUDE_sin_tan_product_l1214_121417

/-- Given an angle α whose terminal side intersects the unit circle at point P(-1/2, y),
    prove that sinα•tanα = -3/2 -/
theorem sin_tan_product (α : Real) (y : Real) 
    (h1 : Real.cos α = -1/2)  -- x-coordinate of P is -1/2
    (h2 : Real.sin α = y)     -- y-coordinate of P is y
    (h3 : (-1/2)^2 + y^2 = 1) -- P is on the unit circle
    : Real.sin α * Real.tan α = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_tan_product_l1214_121417


namespace NUMINAMATH_CALUDE_two_digit_integer_problem_l1214_121454

theorem two_digit_integer_problem :
  ∃ (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧  -- a is a 2-digit positive integer
    10 ≤ b ∧ b < 100 ∧  -- b is a 2-digit positive integer
    a ≠ b ∧             -- a and b are different
    (a + b) / 2 = a + b / 100 ∧  -- average equals the special number
    a < b ∧             -- a is smaller than b
    a = 49 :=           -- a is 49
by sorry

end NUMINAMATH_CALUDE_two_digit_integer_problem_l1214_121454


namespace NUMINAMATH_CALUDE_circumscribed_circle_diameter_l1214_121460

theorem circumscribed_circle_diameter 
  (side : ℝ) (angle : ℝ) (h1 : side = 15) (h2 : angle = π / 4) :
  (side / Real.sin angle) = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_diameter_l1214_121460


namespace NUMINAMATH_CALUDE_intersection_theorem_l1214_121491

-- Define the set A
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Define the set B
def B : Set ℝ := {x | x^2 + x - 2 > 0}

-- State the theorem
theorem intersection_theorem : A ∩ B = {y | y > 1} := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l1214_121491


namespace NUMINAMATH_CALUDE_card_z_value_l1214_121477

/-- Given four cards W, X, Y, Z with specific tagging rules, prove that Z is tagged with 400. -/
theorem card_z_value : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun w x y z =>
    w = 200 ∧
    x = w / 2 ∧
    y = w + x ∧
    w + x + y + z = 1000 →
    z = 400

/-- Proof of the card_z_value theorem -/
lemma prove_card_z_value : card_z_value 200 100 300 400 := by
  sorry

end NUMINAMATH_CALUDE_card_z_value_l1214_121477


namespace NUMINAMATH_CALUDE_geometric_progression_properties_l1214_121410

/-- A geometric progression with given second and fifth terms -/
structure GeometricProgression where
  b₂ : ℝ
  b₅ : ℝ
  h₁ : b₂ = 24.5
  h₂ : b₅ = 196

/-- The third term of the geometric progression -/
def thirdTerm (gp : GeometricProgression) : ℝ := 49

/-- The sum of the first four terms of the geometric progression -/
def sumFirstFour (gp : GeometricProgression) : ℝ := 183.75

theorem geometric_progression_properties (gp : GeometricProgression) :
  thirdTerm gp = 49 ∧ sumFirstFour gp = 183.75 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_properties_l1214_121410


namespace NUMINAMATH_CALUDE_product_of_cubes_l1214_121434

theorem product_of_cubes (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h1 : r^3 + s^3 = 1) (h2 : r^6 + s^6 = 15/16) : r * s = (1/48)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_product_of_cubes_l1214_121434


namespace NUMINAMATH_CALUDE_division_with_maximum_remainder_l1214_121485

theorem division_with_maximum_remainder :
  ∃ (star : ℕ) (triangle : ℕ),
    star / 6 = 102 ∧
    star % 6 = triangle ∧
    triangle ≤ 5 ∧
    (∀ (s t : ℕ), s / 6 = 102 ∧ s % 6 = t → t ≤ triangle) ∧
    triangle = 5 ∧
    star = 617 := by
  sorry

end NUMINAMATH_CALUDE_division_with_maximum_remainder_l1214_121485


namespace NUMINAMATH_CALUDE_minimum_shoeing_time_l1214_121430

theorem minimum_shoeing_time 
  (blacksmiths : ℕ) 
  (horses : ℕ) 
  (time_per_shoe : ℕ) 
  (h1 : blacksmiths = 48) 
  (h2 : horses = 60) 
  (h3 : time_per_shoe = 5) : 
  (horses * 4 * time_per_shoe) / blacksmiths = 25 := by
  sorry

end NUMINAMATH_CALUDE_minimum_shoeing_time_l1214_121430


namespace NUMINAMATH_CALUDE_circle_area_approximation_l1214_121453

/-- The area of a circle with radius 0.6 meters is 1.08 square meters when pi is approximated as 3 -/
theorem circle_area_approximation (r : ℝ) (π : ℝ) (A : ℝ) : 
  r = 0.6 → π = 3 → A = π * r^2 → A = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_approximation_l1214_121453


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l1214_121440

theorem triangle_side_calculation (A B C : Real) (a b c : Real) :
  A = 45 * π / 180 →
  B = 60 * π / 180 →
  a = Real.sqrt 2 →
  b = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l1214_121440


namespace NUMINAMATH_CALUDE_permutation_of_6_choose_2_l1214_121437

def A (n : ℕ) (k : ℕ) : ℕ := n * (n - 1)

theorem permutation_of_6_choose_2 : A 6 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_permutation_of_6_choose_2_l1214_121437


namespace NUMINAMATH_CALUDE_not_pythagorean_triple_8_12_16_l1214_121418

/-- Definition of a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2

/-- Theorem: The set (8, 12, 16) is not a Pythagorean triple -/
theorem not_pythagorean_triple_8_12_16 :
  ¬ is_pythagorean_triple 8 12 16 := by
  sorry

end NUMINAMATH_CALUDE_not_pythagorean_triple_8_12_16_l1214_121418


namespace NUMINAMATH_CALUDE_correct_average_l1214_121421

theorem correct_average (n : ℕ) (initial_avg : ℚ) (increase : ℚ) : 
  n = 10 →
  initial_avg = 5 →
  increase = 10 →
  (n : ℚ) * initial_avg + increase = n * 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l1214_121421


namespace NUMINAMATH_CALUDE_exp_inequality_equivalence_l1214_121481

theorem exp_inequality_equivalence (x : ℝ) : 1 < Real.exp x ∧ Real.exp x < 2 ↔ 0 < x ∧ x < Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_exp_inequality_equivalence_l1214_121481


namespace NUMINAMATH_CALUDE_quadratic_and_slope_l1214_121449

-- Define the quadratic polynomial
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions
def passes_through (a b c : ℝ) : Prop :=
  quadratic a b c 1 = -2 ∧
  quadratic a b c 2 = 4 ∧
  quadratic a b c 3 = 10

-- Define the slope of the tangent line
def tangent_slope (a b c : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

-- Theorem statement
theorem quadratic_and_slope :
  ∃ a b c : ℝ,
    passes_through a b c ∧
    (∀ x : ℝ, quadratic a b c x = 6 * x - 8) ∧
    tangent_slope a b c 2 = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_and_slope_l1214_121449


namespace NUMINAMATH_CALUDE_number_thought_of_l1214_121407

theorem number_thought_of (x : ℝ) : (x / 5 + 6 = 65) → x = 295 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l1214_121407


namespace NUMINAMATH_CALUDE_batsman_average_l1214_121496

/-- Calculates the overall average runs per match for a batsman -/
def overall_average (matches1 : ℕ) (avg1 : ℚ) (matches2 : ℕ) (avg2 : ℚ) : ℚ :=
  (matches1 * avg1 + matches2 * avg2) / (matches1 + matches2)

/-- The batsman's overall average is approximately 21.43 -/
theorem batsman_average : 
  let matches1 := 15
  let avg1 := 30
  let matches2 := 20
  let avg2 := 15
  abs (overall_average matches1 avg1 matches2 avg2 - 21.43) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_l1214_121496
