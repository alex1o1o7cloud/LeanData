import Mathlib

namespace NUMINAMATH_CALUDE_egg_production_increase_proof_l3252_325290

/-- The increase in egg production from last year to this year -/
def egg_production_increase (last_year_production this_year_production : ℕ) : ℕ :=
  this_year_production - last_year_production

/-- Theorem stating the increase in egg production -/
theorem egg_production_increase_proof 
  (last_year_production : ℕ) 
  (this_year_production : ℕ) 
  (h1 : last_year_production = 1416)
  (h2 : this_year_production = 4636) : 
  egg_production_increase last_year_production this_year_production = 3220 := by
  sorry

end NUMINAMATH_CALUDE_egg_production_increase_proof_l3252_325290


namespace NUMINAMATH_CALUDE_employee_salaries_l3252_325278

/-- Proves that the salaries of employees m, n, p, and q sum up to $3000 given the stated conditions --/
theorem employee_salaries (n m p q : ℝ) : 
  (m = 1.4 * n) →
  (p = 0.85 * (m - n)) →
  (q = 1.1 * p) →
  (n + m + p + q = 3000) :=
by
  sorry

end NUMINAMATH_CALUDE_employee_salaries_l3252_325278


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3252_325229

def A : Set ℝ := {x | |x| < 2}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3252_325229


namespace NUMINAMATH_CALUDE_oil_price_reduction_l3252_325261

/-- Proves that given a 40% reduction in oil price, if 8 kg more oil can be bought for Rs. 2400 after the reduction, then the reduced price per kg is Rs. 120. -/
theorem oil_price_reduction (original_price : ℝ) : 
  let reduced_price := original_price * 0.6
  let original_quantity := 2400 / original_price
  let new_quantity := 2400 / reduced_price
  (new_quantity - original_quantity = 8) → reduced_price = 120 := by
  sorry


end NUMINAMATH_CALUDE_oil_price_reduction_l3252_325261


namespace NUMINAMATH_CALUDE_quadratic_equation_with_ratio_roots_l3252_325246

theorem quadratic_equation_with_ratio_roots (k : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ 
    (∀ x : ℝ, x^2 + 8*x + k = 0 ↔ (x = 3*r ∨ x = r)) ∧
    3*r ≠ r) → 
  k = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_ratio_roots_l3252_325246


namespace NUMINAMATH_CALUDE_mans_rate_l3252_325268

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 24)
  (h2 : speed_against_stream = 10) : 
  (speed_with_stream + speed_against_stream) / 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_mans_rate_l3252_325268


namespace NUMINAMATH_CALUDE_two_numbers_with_specific_means_l3252_325223

theorem two_numbers_with_specific_means (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x * y = 600^2) → 
  ((x + y) / 2 = (2 * x * y) / (x + y) + 49) →
  ({x, y} : Set ℝ) = {800, 450} := by
sorry

end NUMINAMATH_CALUDE_two_numbers_with_specific_means_l3252_325223


namespace NUMINAMATH_CALUDE_max_value_of_a_l3252_325212

def f (x a : ℝ) : ℝ := |8 * x^3 - 12 * x - a| + a

theorem max_value_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x a ≤ 0) ∧ (∃ x ∈ Set.Icc 0 1, f x a = 0) →
  a ≤ -2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3252_325212


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3252_325276

/-- Properties of a rectangle and an ellipse -/
structure RectangleEllipseSystem where
  /-- Length of the rectangle -/
  x : ℝ
  /-- Width of the rectangle -/
  y : ℝ
  /-- Semi-major axis of the ellipse -/
  a : ℝ
  /-- Semi-minor axis of the ellipse -/
  b : ℝ
  /-- The area of the rectangle is 3260 -/
  area_rectangle : x * y = 3260
  /-- The area of the ellipse is 3260π -/
  area_ellipse : π * a * b = 3260 * π
  /-- The sum of length and width equals twice the semi-major axis -/
  major_axis : x + y = 2 * a
  /-- The rectangle diagonal equals twice the focal distance -/
  focal_distance : x^2 + y^2 = 4 * (a^2 - b^2)

/-- The perimeter of the rectangle is 8√1630 -/
theorem rectangle_perimeter (s : RectangleEllipseSystem) : 
  2 * (s.x + s.y) = 8 * Real.sqrt 1630 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3252_325276


namespace NUMINAMATH_CALUDE_prob_all_co_captains_l3252_325298

/-- Represents a math team with a certain number of students and co-captains -/
structure MathTeam where
  size : Nat
  coCaptains : Nat

/-- Calculates the probability of selecting all co-captains from a single team -/
def probAllCoCaptains (team : MathTeam) : Rat :=
  1 / (Nat.choose team.size 3)

/-- The set of math teams in the area -/
def mathTeams : List MathTeam := [
  { size := 6, coCaptains := 3 },
  { size := 8, coCaptains := 3 },
  { size := 9, coCaptains := 3 },
  { size := 10, coCaptains := 3 }
]

/-- The main theorem stating the probability of selecting all co-captains -/
theorem prob_all_co_captains : 
  (List.sum (mathTeams.map probAllCoCaptains) / mathTeams.length : Rat) = 53 / 3360 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_co_captains_l3252_325298


namespace NUMINAMATH_CALUDE_blueberry_count_l3252_325282

theorem blueberry_count (total : ℕ) (raspberries : ℕ) (blackberries : ℕ) 
  (h1 : total = 42)
  (h2 : raspberries = total / 2)
  (h3 : blackberries = total / 3) :
  total - raspberries - blackberries = 7 := by
sorry

end NUMINAMATH_CALUDE_blueberry_count_l3252_325282


namespace NUMINAMATH_CALUDE_remainder_3_20_mod_11_l3252_325255

theorem remainder_3_20_mod_11 (h : Prime 11) : 3^20 ≡ 1 [MOD 11] := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_20_mod_11_l3252_325255


namespace NUMINAMATH_CALUDE_total_rope_length_l3252_325275

-- Define the lengths of rope used for each post
def post1_length : ℕ := 24
def post2_length : ℕ := 20
def post3_length : ℕ := 14
def post4_length : ℕ := 12

-- Theorem stating that the total rope length is 70 inches
theorem total_rope_length :
  post1_length + post2_length + post3_length + post4_length = 70 :=
by sorry

end NUMINAMATH_CALUDE_total_rope_length_l3252_325275


namespace NUMINAMATH_CALUDE_union_M_complement_N_l3252_325242

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def M : Set Nat := {1, 3, 5, 6}
def N : Set Nat := {1, 2, 4, 7, 9}

theorem union_M_complement_N : M ∪ (U \ N) = {1, 3, 5, 6, 8} := by sorry

end NUMINAMATH_CALUDE_union_M_complement_N_l3252_325242


namespace NUMINAMATH_CALUDE_train_speed_problem_l3252_325210

theorem train_speed_problem (V₁ V₂ : ℝ) (h₁ : V₁ > 0) (h₂ : V₂ > 0) (h₃ : V₂ > V₁) : 
  (∃ t : ℝ, t > 0 ∧ t * (V₁ + V₂) = 2400) ∧
  (∃ t : ℝ, t > 0 ∧ 2 * V₂ * (t - 3) = 2400) ∧
  (∃ t : ℝ, t > 0 ∧ 2 * V₁ * (t + 5) = 2400) →
  V₁ = 60 ∧ V₂ = 100 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3252_325210


namespace NUMINAMATH_CALUDE_min_value_quadratic_expression_l3252_325215

theorem min_value_quadratic_expression (x y : ℝ) :
  x^2 + y^2 - 4*x + 6*y + 20 ≥ 7 := by
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_expression_l3252_325215


namespace NUMINAMATH_CALUDE_stratified_sampling_female_count_l3252_325262

theorem stratified_sampling_female_count 
  (total_employees : ℕ) 
  (male_employees : ℕ) 
  (female_employees : ℕ) 
  (male_sampled : ℕ) :
  total_employees = 140 →
  male_employees = 80 →
  female_employees = 60 →
  male_sampled = 16 →
  (female_employees : ℚ) * (male_sampled : ℚ) / (male_employees : ℚ) = 12 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_count_l3252_325262


namespace NUMINAMATH_CALUDE_decimal_equivalent_of_half_squared_l3252_325283

theorem decimal_equivalent_of_half_squared : (1 / 2 : ℚ) ^ 2 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalent_of_half_squared_l3252_325283


namespace NUMINAMATH_CALUDE_job_completion_proof_l3252_325258

/-- Given workers P, Q, and R who can complete a job in 3, 9, and 6 hours respectively,
    prove that the combined work of P (1 hour), Q (2 hours), and R (3 hours) completes the job. -/
theorem job_completion_proof (p q r : ℝ) (hp : p = 1/3) (hq : q = 1/9) (hr : r = 1/6) :
  p * 1 + q * 2 + r * 3 ≥ 1 := by
  sorry

#check job_completion_proof

end NUMINAMATH_CALUDE_job_completion_proof_l3252_325258


namespace NUMINAMATH_CALUDE_second_exam_sleep_duration_l3252_325245

/-- Represents the relationship between sleep duration and test score -/
structure SleepScoreRelation where
  sleep : ℝ
  score : ℝ
  constant : ℝ
  inv_relation : sleep * score = constant

/-- Proves the required sleep duration for the second exam -/
theorem second_exam_sleep_duration 
  (first_exam : SleepScoreRelation)
  (h_first_exam : first_exam.sleep = 9 ∧ first_exam.score = 75)
  (target_average : ℝ)
  (h_target_average : target_average = 85) :
  ∃ (second_exam : SleepScoreRelation),
    second_exam.constant = first_exam.constant ∧
    (first_exam.score + second_exam.score) / 2 = target_average ∧
    second_exam.sleep = 135 / 19 := by
  sorry

end NUMINAMATH_CALUDE_second_exam_sleep_duration_l3252_325245


namespace NUMINAMATH_CALUDE_proportional_equation_inequality_l3252_325219

theorem proportional_equation_inequality (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  b / d = c / a → ¬(a * d = b * c) :=
by sorry

end NUMINAMATH_CALUDE_proportional_equation_inequality_l3252_325219


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l3252_325253

-- Define the circles
def circle1_center : ℝ × ℝ := (3, 0)
def circle1_radius : ℝ := 3
def circle2_center : ℝ × ℝ := (8, 0)
def circle2_radius : ℝ := 2

-- Define the tangent line
def tangent_line (y_intercept : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (m : ℝ), p.2 = m * p.1 + y_intercept}

-- Define the condition for the line to be tangent to a circle
def is_tangent_to_circle (line : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ (p : ℝ × ℝ), p ∈ line ∧ 
    (p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2 = radius ^ 2 ∧
    ∀ (q : ℝ × ℝ), q ∈ line → q ≠ p → 
      (q.1 - center.1) ^ 2 + (q.2 - center.2) ^ 2 > radius ^ 2

-- Theorem statement
theorem tangent_line_y_intercept : 
  ∃ (y_intercept : ℝ), 
    y_intercept = 2 * Real.sqrt 104 ∧
    let line := tangent_line y_intercept
    is_tangent_to_circle line circle1_center circle1_radius ∧
    is_tangent_to_circle line circle2_center circle2_radius ∧
    ∀ (p : ℝ × ℝ), p ∈ line → p.1 ≥ 0 ∧ p.2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l3252_325253


namespace NUMINAMATH_CALUDE_sandwich_combinations_l3252_325214

theorem sandwich_combinations (n m k l : ℕ) (hn : n = 7) (hm : m = 3) (hk : k = 2) (hl : l = 1) :
  (n.choose k) * (m.choose l) = 63 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l3252_325214


namespace NUMINAMATH_CALUDE_binomial_11_1_l3252_325221

theorem binomial_11_1 : (11 : ℕ).choose 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_binomial_11_1_l3252_325221


namespace NUMINAMATH_CALUDE_consecutive_even_sequence_unique_l3252_325230

/-- A sequence of four consecutive even integers -/
def ConsecutiveEvenSequence (a b c d : ℤ) : Prop :=
  (b = a + 2) ∧ (c = b + 2) ∧ (d = c + 2) ∧ Even a ∧ Even b ∧ Even c ∧ Even d

theorem consecutive_even_sequence_unique :
  ∀ a b c d : ℤ,
  ConsecutiveEvenSequence a b c d →
  c = 14 →
  a + b + c + d = 52 →
  a = 10 ∧ b = 12 ∧ c = 14 ∧ d = 16 := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_sequence_unique_l3252_325230


namespace NUMINAMATH_CALUDE_AB_vector_l3252_325265

def OA : ℝ × ℝ := (2, 1)
def OB : ℝ × ℝ := (-3, 4)

theorem AB_vector : 
  let AB := (OB.1 - OA.1, OB.2 - OA.2)
  AB = (-5, 3) := by sorry

end NUMINAMATH_CALUDE_AB_vector_l3252_325265


namespace NUMINAMATH_CALUDE_function_properties_l3252_325292

/-- Given a function f(x) = sin(ωx + φ) with the following properties:
    - ω > 0
    - 0 < φ < π
    - The distance between two adjacent zeros of f(x) is π/2
    - g(x) is f(x) shifted left by π/6 units
    - g(x) is an even function
    
    This theorem states that:
    1. f(x) = sin(2x + π/6)
    2. The axis of symmetry is x = kπ/2 + π/6 for k ∈ ℤ
    3. The interval of monotonic increase is [kπ - π/3, kπ + π/6] for k ∈ ℤ -/
theorem function_properties (ω φ : ℝ) (h_ω : ω > 0) (h_φ : 0 < φ ∧ φ < π)
  (f : ℝ → ℝ) (hf : f = fun x ↦ Real.sin (ω * x + φ))
  (h_zeros : ∀ x₁ x₂, f x₁ = 0 → f x₂ = 0 → x₁ ≠ x₂ → |x₁ - x₂| = π / 2)
  (g : ℝ → ℝ) (hg : g = fun x ↦ f (x + π / 6))
  (h_even : ∀ x, g x = g (-x)) :
  (f = fun x ↦ Real.sin (2 * x + π / 6)) ∧
  (∀ k : ℤ, ∃ x, x = k * π / 2 + π / 6 ∧ ∀ y, f (2 * x - y) = f (2 * x + y)) ∧
  (∀ k : ℤ, ∀ x, k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6 → Monotone (f ∘ (fun y ↦ y + x))) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3252_325292


namespace NUMINAMATH_CALUDE_classroom_a_fundraising_l3252_325272

/-- The fundraising goal for each classroom -/
def goal : ℕ := 200

/-- The amount raised from two families at $20 each -/
def amount_20 : ℕ := 2 * 20

/-- The amount raised from eight families at $10 each -/
def amount_10 : ℕ := 8 * 10

/-- The amount raised from ten families at $5 each -/
def amount_5 : ℕ := 10 * 5

/-- The total amount raised by Classroom A -/
def total_raised : ℕ := amount_20 + amount_10 + amount_5

/-- The additional amount needed to reach the goal -/
def additional_amount_needed : ℕ := goal - total_raised

theorem classroom_a_fundraising :
  additional_amount_needed = 30 :=
by sorry

end NUMINAMATH_CALUDE_classroom_a_fundraising_l3252_325272


namespace NUMINAMATH_CALUDE_max_angle_A_l3252_325217

/-- Represents the side lengths of a triangle sequence -/
structure TriangleSequence where
  a : ℕ → ℝ
  b : ℕ → ℝ
  c : ℕ → ℝ

/-- Conditions for the triangle sequence -/
def ValidTriangleSequence (t : TriangleSequence) : Prop :=
  (t.b 1 > t.c 1) ∧
  (t.b 1 + t.c 1 = 2 * t.a 1) ∧
  (∀ n, t.a (n + 1) = t.a n) ∧
  (∀ n, t.b (n + 1) = (t.c n + t.a n) / 2) ∧
  (∀ n, t.c (n + 1) = (t.b n + t.a n) / 2)

/-- The angle A_n in the triangle sequence -/
noncomputable def angleA (t : TriangleSequence) (n : ℕ) : ℝ :=
  Real.arccos ((t.b n ^ 2 + t.c n ^ 2 - t.a n ^ 2) / (2 * t.b n * t.c n))

/-- The theorem stating the maximum value of angle A_n -/
theorem max_angle_A (t : TriangleSequence) (h : ValidTriangleSequence t) :
    (∀ n, angleA t n ≤ π / 3) ∧ (∃ n, angleA t n = π / 3) := by
  sorry


end NUMINAMATH_CALUDE_max_angle_A_l3252_325217


namespace NUMINAMATH_CALUDE_newspaper_pages_l3252_325241

/-- Represents a newspaper with a certain number of pages -/
structure Newspaper where
  num_pages : ℕ

/-- Predicate indicating that two pages are on the same sheet -/
def on_same_sheet (n : Newspaper) (p1 p2 : ℕ) : Prop :=
  p1 ≤ n.num_pages ∧ p2 ≤ n.num_pages ∧ p1 + p2 = n.num_pages + 1

/-- The theorem stating the number of pages in the newspaper -/
theorem newspaper_pages : 
  ∃ (n : Newspaper), n.num_pages = 28 ∧ on_same_sheet n 8 21 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_pages_l3252_325241


namespace NUMINAMATH_CALUDE_college_students_count_l3252_325213

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 400) :
  boys + girls = 1040 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l3252_325213


namespace NUMINAMATH_CALUDE_logarithm_product_theorem_l3252_325202

theorem logarithm_product_theorem (c d : ℕ+) : 
  (d - c = 840) →
  (Real.log d / Real.log c = 3) →
  (c + d : ℕ) = 1010 := by sorry

end NUMINAMATH_CALUDE_logarithm_product_theorem_l3252_325202


namespace NUMINAMATH_CALUDE_composite_form_l3252_325297

theorem composite_form (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (111111 + 9 * 10^n = a * b) := by
  sorry

end NUMINAMATH_CALUDE_composite_form_l3252_325297


namespace NUMINAMATH_CALUDE_game_specific_outcome_l3252_325294

def game_probability (total_rounds : ℕ) 
                     (alex_prob : ℚ) 
                     (mel_chelsea_ratio : ℚ) 
                     (alex_wins : ℕ) 
                     (mel_wins : ℕ) 
                     (chelsea_wins : ℕ) : ℚ :=
  sorry

theorem game_specific_outcome : 
  game_probability 7 (3/5) 2 4 2 1 = 18144/1125 := by sorry

end NUMINAMATH_CALUDE_game_specific_outcome_l3252_325294


namespace NUMINAMATH_CALUDE_parabola_y1_gt_y2_l3252_325248

/-- A parabola with axis of symmetry at x = 1 -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a > 0

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_y1_gt_y2 (p : Parabola) :
  p.y_at (-1) > p.y_at 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_y1_gt_y2_l3252_325248


namespace NUMINAMATH_CALUDE_no_solutions_exist_l3252_325277

theorem no_solutions_exist : ¬∃ (x y : ℕ), 
  x^4 * y^4 - 14 * x^2 * y^2 + 49 = 0 ∧ x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_exist_l3252_325277


namespace NUMINAMATH_CALUDE_undergrad_play_count_l3252_325270

-- Define the total number of students
def total_students : ℕ := 800

-- Define the percentage of undergraduates who play a musical instrument
def undergrad_play_percent : ℚ := 25 / 100

-- Define the percentage of postgraduates who do not play a musical instrument
def postgrad_not_play_percent : ℚ := 20 / 100

-- Define the percentage of all students who do not play a musical instrument
def total_not_play_percent : ℚ := 355 / 1000

-- Theorem stating that the number of undergraduates who play a musical instrument is 57
theorem undergrad_play_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_undergrad_play_count_l3252_325270


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3252_325228

def f (x : ℤ) : ℤ := x^3 - 4*x^2 - 7*x + 10

theorem integer_roots_of_polynomial :
  {x : ℤ | f x = 0} = {1, -2, 5} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3252_325228


namespace NUMINAMATH_CALUDE_weight_gain_difference_l3252_325280

/-- The weight gain problem at the family reunion -/
theorem weight_gain_difference (orlando_gain jose_gain fernando_gain : ℝ) : 
  orlando_gain = 5 →
  jose_gain > 2 * orlando_gain →
  fernando_gain = jose_gain / 2 - 3 →
  orlando_gain + jose_gain + fernando_gain = 20 →
  ∃ ε > 0, |jose_gain - 2 * orlando_gain - 3.67| < ε :=
by sorry

end NUMINAMATH_CALUDE_weight_gain_difference_l3252_325280


namespace NUMINAMATH_CALUDE_lcm_problem_l3252_325267

theorem lcm_problem (a b : ℕ+) (h1 : a + b = 55) (h2 : Nat.gcd a b = 5) 
  (h3 : (1 : ℚ) / a + (1 : ℚ) / b = 0.09166666666666666) : 
  Nat.lcm a b = 120 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3252_325267


namespace NUMINAMATH_CALUDE_max_value_expression_l3252_325206

theorem max_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 + b^2 + c^2 = 2*a*b*c + 1) :
  (a - 2*b*c) * (b - 2*c*a) * (c - 2*a*b) ≤ 1/8 := by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3252_325206


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l3252_325247

/-- A quadratic polynomial that satisfies specific conditions -/
def q (x : ℚ) : ℚ := (4 * x^2 - 6 * x + 5) / 3

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-1) = 5 ∧ q 2 = 3 ∧ q 4 = 15 := by
  sorry


end NUMINAMATH_CALUDE_q_satisfies_conditions_l3252_325247


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3252_325256

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem fifth_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_a1 : a 1 = 3) 
  (h_a3 : a 3 = 12) : 
  a 5 = 48 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3252_325256


namespace NUMINAMATH_CALUDE_victors_friend_bought_two_decks_l3252_325237

/-- The number of decks Victor's friend bought given the conditions of the problem -/
def victors_friend_decks (deck_cost : ℕ) (victors_decks : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent - deck_cost * victors_decks) / deck_cost

/-- Theorem stating that Victor's friend bought 2 decks under the given conditions -/
theorem victors_friend_bought_two_decks :
  victors_friend_decks 8 6 64 = 2 := by
  sorry

end NUMINAMATH_CALUDE_victors_friend_bought_two_decks_l3252_325237


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l3252_325238

theorem complex_arithmetic_equality : (9 - 8 + 7)^2 * 6 + 5 - 4^2 * 3 + 2^3 - 1 = 347 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l3252_325238


namespace NUMINAMATH_CALUDE_age_of_fifteenth_person_l3252_325259

theorem age_of_fifteenth_person (total_persons : ℕ) (avg_all : ℕ) (group1_size : ℕ) (avg_group1 : ℕ) (group2_size : ℕ) (avg_group2 : ℕ) :
  total_persons = 20 →
  avg_all = 15 →
  group1_size = 5 →
  avg_group1 = 14 →
  group2_size = 9 →
  avg_group2 = 16 →
  ∃ (age_15th : ℕ), age_15th = 86 ∧
    total_persons * avg_all = group1_size * avg_group1 + group2_size * avg_group2 + age_15th :=
by sorry

end NUMINAMATH_CALUDE_age_of_fifteenth_person_l3252_325259


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3252_325279

theorem imaginary_part_of_z (z : ℂ) : z = (Complex.I ^ 2017) / (1 - 2 * Complex.I) → z.im = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3252_325279


namespace NUMINAMATH_CALUDE_inequality_proof_l3252_325234

theorem inequality_proof (a b c : ℝ) (k : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k ≥ 2) (habc : a * b * c = 1) :
  (a^k / (a + b)) + (b^k / (b + c)) + (c^k / (c + a)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3252_325234


namespace NUMINAMATH_CALUDE_randy_bats_count_l3252_325232

theorem randy_bats_count :
  ∀ (gloves bats : ℕ),
    gloves = 29 →
    gloves = 7 * bats + 1 →
    bats = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_randy_bats_count_l3252_325232


namespace NUMINAMATH_CALUDE_initial_books_correct_l3252_325235

/-- The number of books initially in the pile to be put away. -/
def initial_books : ℝ := 46.0

/-- The number of books added by the librarian. -/
def added_books : ℝ := 10.0

/-- The number of books that can fit on each shelf. -/
def books_per_shelf : ℝ := 4.0

/-- The number of shelves needed to arrange all books. -/
def shelves_needed : ℕ := 14

/-- Theorem stating that the initial number of books is correct given the conditions. -/
theorem initial_books_correct : 
  initial_books = (books_per_shelf * shelves_needed : ℝ) - added_books :=
by sorry

end NUMINAMATH_CALUDE_initial_books_correct_l3252_325235


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3252_325293

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a x : V) 
  (h : 2 • x - 3 • (x - 2 • a) = 0) : x = 6 • a := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3252_325293


namespace NUMINAMATH_CALUDE_min_people_like_both_l3252_325284

/-- Represents the number of people who like both Vivaldi and Chopin -/
def both_like (v c b : ℕ) : Prop := b = v + c - 150

/-- The minimum number of people who like both Vivaldi and Chopin -/
def min_both_like (v c : ℕ) : ℕ := max 0 (v + c - 150)

theorem min_people_like_both (total v c : ℕ) 
  (h_total : total = 150) 
  (h_v : v = 120) 
  (h_c : c = 90) : 
  min_both_like v c = 60 := by
  sorry

#eval min_both_like 120 90

end NUMINAMATH_CALUDE_min_people_like_both_l3252_325284


namespace NUMINAMATH_CALUDE_joans_kittens_l3252_325264

theorem joans_kittens (initial : ℕ) (additional : ℕ) (total : ℕ) 
  (h1 : initial = 15)
  (h2 : additional = 5)
  (h3 : total = initial + additional) : 
  total = 20 := by
  sorry

end NUMINAMATH_CALUDE_joans_kittens_l3252_325264


namespace NUMINAMATH_CALUDE_supplement_of_angle_with_30_degree_complement_l3252_325269

theorem supplement_of_angle_with_30_degree_complement :
  ∀ (angle : ℝ), 
  (90 - angle = 30) →
  (180 - angle = 120) :=
by
  sorry

end NUMINAMATH_CALUDE_supplement_of_angle_with_30_degree_complement_l3252_325269


namespace NUMINAMATH_CALUDE_parabola_equation_l3252_325225

/-- The equation of a parabola with focus at (2, 1) and the y-axis as its directrix -/
theorem parabola_equation (x y : ℝ) :
  let focus : ℝ × ℝ := (2, 1)
  let directrix : Set (ℝ × ℝ) := {p | p.1 = 0}
  let parabola_equation : ℝ × ℝ → Prop := λ p => (p.2 - 1)^2 = 4 * (p.1 - 1)
  (∀ p, p ∈ directrix → dist p focus = dist p (x, y)) ↔ parabola_equation (x, y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3252_325225


namespace NUMINAMATH_CALUDE_triangle_sides_max_sum_squares_l3252_325227

theorem triangle_sides_max_sum_squares (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (1/2) * c^2 = (1/2) * a * b * Real.sin C →
  a * b = Real.sqrt 2 →
  ∃ (max : ℝ), max = 4 ∧ ∀ (a' b' c' : ℝ),
    a' > 0 → b' > 0 → c' > 0 →
    (1/2) * c'^2 = (1/2) * a' * b' * Real.sin C →
    a' * b' = Real.sqrt 2 →
    a'^2 + b'^2 + c'^2 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_triangle_sides_max_sum_squares_l3252_325227


namespace NUMINAMATH_CALUDE_min_box_value_l3252_325285

/-- Given that (ax+b)(bx+a) = 30x^2 + ⬜x + 30, where a, b, and ⬜ are distinct integers,
    prove that the minimum possible value of ⬜ is 61. -/
theorem min_box_value (a b box : ℤ) : 
  (∀ x, (a*x + b)*(b*x + a) = 30*x^2 + box*x + 30) →
  a ≠ b ∧ b ≠ box ∧ a ≠ box →
  a * b = 30 →
  box = a^2 + b^2 →
  (∀ a' b' box' : ℤ, 
    (∀ x, (a'*x + b')*(b'*x + a') = 30*x^2 + box'*x + 30) →
    a' ≠ b' ∧ b' ≠ box' ∧ a' ≠ box' →
    a' * b' = 30 →
    box' = a'^2 + b'^2 →
    box ≤ box') →
  box = 61 := by
sorry

end NUMINAMATH_CALUDE_min_box_value_l3252_325285


namespace NUMINAMATH_CALUDE_min_value_product_l3252_325274

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : (2/x) + (3/y) + (1/z) = 12) : 
  x^2 * y^3 * z ≥ (1/64) :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l3252_325274


namespace NUMINAMATH_CALUDE_unique_base_system_solution_l3252_325224

/-- Represents a base-b numeral system where 1987 is written as xyz --/
structure BaseSystem where
  b : ℕ
  x : ℕ
  y : ℕ
  z : ℕ
  h1 : b > 1
  h2 : x < b ∧ y < b ∧ z < b
  h3 : x + y + z = 25
  h4 : x * b^2 + y * b + z = 1987

/-- The unique solution to the base system problem --/
theorem unique_base_system_solution :
  ∃! (s : BaseSystem), s.b = 19 ∧ s.x = 5 ∧ s.y = 9 ∧ s.z = 11 :=
sorry

end NUMINAMATH_CALUDE_unique_base_system_solution_l3252_325224


namespace NUMINAMATH_CALUDE_proposition_q_must_be_true_l3252_325208

theorem proposition_q_must_be_true (p q : Prop) 
  (h1 : ¬p) (h2 : p ∨ q) : q := by
  sorry

end NUMINAMATH_CALUDE_proposition_q_must_be_true_l3252_325208


namespace NUMINAMATH_CALUDE_website_visitors_ratio_l3252_325273

/-- Proves that the ratio of visitors on the last day to the total visitors on the first 6 days is 2:1 -/
theorem website_visitors_ratio (daily_visitors : ℕ) (constant_days : ℕ) (revenue_per_visit : ℚ) (total_revenue : ℚ) 
  (h1 : daily_visitors = 100)
  (h2 : constant_days = 6)
  (h3 : revenue_per_visit = 1 / 100)
  (h4 : total_revenue = 18) :
  (total_revenue / revenue_per_visit - daily_visitors * constant_days) / (daily_visitors * constant_days) = 2 := by
sorry

end NUMINAMATH_CALUDE_website_visitors_ratio_l3252_325273


namespace NUMINAMATH_CALUDE_tree_planting_cost_l3252_325281

/-- The cost of planting trees to achieve a specific temperature drop -/
theorem tree_planting_cost (initial_temp final_temp temp_drop_per_tree cost_per_tree : ℝ) : 
  initial_temp - final_temp = 1.8 →
  temp_drop_per_tree = 0.1 →
  cost_per_tree = 6 →
  ((initial_temp - final_temp) / temp_drop_per_tree) * cost_per_tree = 108 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_cost_l3252_325281


namespace NUMINAMATH_CALUDE_student_difference_l3252_325288

theorem student_difference (lower_grades : ℕ) (middle_upper_grades : ℕ) : 
  lower_grades = 325 →
  middle_upper_grades = 4 * lower_grades →
  middle_upper_grades - lower_grades = 975 := by
  sorry

end NUMINAMATH_CALUDE_student_difference_l3252_325288


namespace NUMINAMATH_CALUDE_set_intersection_example_l3252_325201

theorem set_intersection_example :
  let M : Set ℕ := {2, 3, 4, 5}
  let N : Set ℕ := {3, 4, 5}
  M ∩ N = {3, 4, 5} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l3252_325201


namespace NUMINAMATH_CALUDE_shopkeeper_profit_days_l3252_325244

/-- Proves that given the specified mean profits, the total number of days is 30 -/
theorem shopkeeper_profit_days : 
  ∀ (total_days : ℕ) (mean_profit mean_first_15 mean_last_15 : ℚ),
  mean_profit = 350 →
  mean_first_15 = 225 →
  mean_last_15 = 475 →
  mean_profit * total_days = mean_first_15 * 15 + mean_last_15 * 15 →
  total_days = 30 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_days_l3252_325244


namespace NUMINAMATH_CALUDE_problem_solution_l3252_325254

theorem problem_solution (x y z : ℝ) 
  (h1 : 5 = 0.25 * x)
  (h2 : 5 = 0.10 * y)
  (h3 : z = 2 * y) :
  x - z = -80 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3252_325254


namespace NUMINAMATH_CALUDE_robin_sodas_l3252_325243

/-- The number of sodas Robin and her friends drank -/
def sodas_drunk : ℕ := 3

/-- The number of extra sodas Robin had -/
def sodas_extra : ℕ := 8

/-- The total number of sodas Robin bought -/
def total_sodas : ℕ := sodas_drunk + sodas_extra

theorem robin_sodas : total_sodas = 11 := by sorry

end NUMINAMATH_CALUDE_robin_sodas_l3252_325243


namespace NUMINAMATH_CALUDE_volunteer_distribution_l3252_325289

theorem volunteer_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 4) :
  (Nat.choose (n - 1) (k - 1) : ℕ) = 84 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_distribution_l3252_325289


namespace NUMINAMATH_CALUDE_least_period_is_30_l3252_325205

/-- A function satisfying the given condition -/
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least common positive period for all functions satisfying the condition -/
def least_common_positive_period (p : ℝ) : Prop :=
  p > 0 ∧
  (∀ f : ℝ → ℝ, satisfies_condition f → is_period f p) ∧
  ∀ q : ℝ, 0 < q ∧ q < p → ∃ f : ℝ → ℝ, satisfies_condition f ∧ ¬is_period f q

theorem least_period_is_30 :
  least_common_positive_period 30 := by sorry

end NUMINAMATH_CALUDE_least_period_is_30_l3252_325205


namespace NUMINAMATH_CALUDE_probability_non_littermates_correct_l3252_325220

/-- Represents the number of dogs with a specific number of littermates -/
structure DogGroup where
  count : Nat
  littermates : Nat

/-- Represents the total number of dogs and their groupings by littermates -/
structure BreedingKennel where
  totalDogs : Nat
  groups : List DogGroup

/-- Calculates the probability of selecting two non-littermate dogs from a breeding kennel -/
def probabilityNonLittermates (kennel : BreedingKennel) : Rat :=
  sorry

theorem probability_non_littermates_correct (kennel : BreedingKennel) :
  kennel.totalDogs = 20 ∧
  kennel.groups = [
    ⟨8, 1⟩,
    ⟨6, 2⟩,
    ⟨4, 3⟩,
    ⟨2, 4⟩
  ] →
  probabilityNonLittermates kennel = 82 / 95 :=
sorry

end NUMINAMATH_CALUDE_probability_non_littermates_correct_l3252_325220


namespace NUMINAMATH_CALUDE_cat_mouse_positions_after_258_moves_l3252_325257

/-- Represents the positions on the square grid --/
inductive Position
  | TopLeft
  | TopRight
  | BottomRight
  | BottomLeft
  | TopMiddle
  | RightMiddle
  | BottomMiddle
  | LeftMiddle

/-- Represents the movement of the cat --/
def catMove (n : ℕ) : Position :=
  match n % 4 with
  | 0 => Position.TopLeft
  | 1 => Position.TopRight
  | 2 => Position.BottomRight
  | 3 => Position.BottomLeft
  | _ => Position.TopLeft  -- This case is unreachable, but needed for exhaustiveness

/-- Represents the movement of the mouse --/
def mouseMove (n : ℕ) : Position :=
  match n % 8 with
  | 0 => Position.TopMiddle
  | 1 => Position.TopRight
  | 2 => Position.RightMiddle
  | 3 => Position.BottomRight
  | 4 => Position.BottomMiddle
  | 5 => Position.BottomLeft
  | 6 => Position.LeftMiddle
  | 7 => Position.TopLeft
  | _ => Position.TopMiddle  -- This case is unreachable, but needed for exhaustiveness

theorem cat_mouse_positions_after_258_moves :
  catMove 258 = Position.TopRight ∧ mouseMove 258 = Position.TopRight :=
sorry

end NUMINAMATH_CALUDE_cat_mouse_positions_after_258_moves_l3252_325257


namespace NUMINAMATH_CALUDE_downstream_distance_is_35_l3252_325299

-- Define the given constants
def man_speed : ℝ := 5.5
def upstream_distance : ℝ := 20
def swim_time : ℝ := 5

-- Define the theorem
theorem downstream_distance_is_35 :
  let stream_speed := (man_speed - upstream_distance / swim_time) / 2
  let downstream_distance := (man_speed + stream_speed) * swim_time
  downstream_distance = 35 := by sorry

end NUMINAMATH_CALUDE_downstream_distance_is_35_l3252_325299


namespace NUMINAMATH_CALUDE_cos_20_minus_cos_40_l3252_325252

theorem cos_20_minus_cos_40 : Real.cos (20 * π / 180) - Real.cos (40 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_20_minus_cos_40_l3252_325252


namespace NUMINAMATH_CALUDE_prob_one_red_ball_eq_one_third_l3252_325287

/-- The probability of drawing exactly one red ball from a bag -/
def prob_one_red_ball (red_balls black_balls : ℕ) : ℚ :=
  red_balls / (red_balls + black_balls)

/-- Theorem: The probability of drawing exactly one red ball from a bag
    containing 2 red balls and 4 black balls is 1/3 -/
theorem prob_one_red_ball_eq_one_third :
  prob_one_red_ball 2 4 = 1/3 := by sorry

end NUMINAMATH_CALUDE_prob_one_red_ball_eq_one_third_l3252_325287


namespace NUMINAMATH_CALUDE_triangle_problem_l3252_325236

theorem triangle_problem (A B C : Real) (a b c : Real) :
  (1/2 * b * c * Real.sin A = 10 * Real.sqrt 3) →  -- Area condition
  (a = 7) →  -- Given side length
  (Real.sin A)^2 = (Real.sin B)^2 + (Real.sin C)^2 - Real.sin B * Real.sin C →  -- Given equation
  (A = π/3 ∧ ((b = 5 ∧ c = 8) ∨ (b = 8 ∧ c = 5))) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3252_325236


namespace NUMINAMATH_CALUDE_product_of_roots_cubic_l3252_325233

theorem product_of_roots_cubic (x : ℝ) : 
  (∃ p q r : ℝ, x^3 - 9*x^2 + 27*x - 5 = (x - p) * (x - q) * (x - r)) → 
  (∃ p q r : ℝ, x^3 - 9*x^2 + 27*x - 5 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 5) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_cubic_l3252_325233


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l3252_325203

theorem complex_magnitude_product : Complex.abs (5 - 3*I) * Complex.abs (5 + 3*I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l3252_325203


namespace NUMINAMATH_CALUDE_division_problem_l3252_325295

theorem division_problem (N : ℕ) : 
  (N / 3 = 4) ∧ (N % 3 = 3) → N = 15 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3252_325295


namespace NUMINAMATH_CALUDE_total_late_time_l3252_325249

def charlize_late : ℕ := 20

def ana_late : ℕ := charlize_late + charlize_late / 4
def ben_late : ℕ := charlize_late * 3 / 4
def clara_late : ℕ := charlize_late * 2
def daniel_late : ℕ := 30 * 4 / 5

def ana_missed : ℕ := 5
def ben_missed : ℕ := 2
def clara_missed : ℕ := 15
def daniel_missed : ℕ := 10

theorem total_late_time :
  charlize_late +
  (ana_late + ana_missed) +
  (ben_late + ben_missed) +
  (clara_late + clara_missed) +
  (daniel_late + daniel_missed) = 156 := by
  sorry

end NUMINAMATH_CALUDE_total_late_time_l3252_325249


namespace NUMINAMATH_CALUDE_three_and_one_fifth_cubed_l3252_325251

theorem three_and_one_fifth_cubed : (3 + 1/5) ^ 3 = 32.768 := by sorry

end NUMINAMATH_CALUDE_three_and_one_fifth_cubed_l3252_325251


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_l3252_325211

theorem sum_of_solutions_quadratic (a b c d e : ℝ) : 
  (∀ x, x^2 - a*x - b = c*x + d) → 
  (∃ x₁ x₂, x₁^2 - a*x₁ - b = c*x₁ + d ∧ 
            x₂^2 - a*x₂ - b = c*x₂ + d ∧ 
            x₁ ≠ x₂) →
  (x₁ + x₂ = a + c) :=
by sorry

-- Specific instance
theorem sum_of_solutions_specific : 
  (∀ x, x^2 - 6*x - 8 = 4*x + 20) → 
  (∃ x₁ x₂, x₁^2 - 6*x₁ - 8 = 4*x₁ + 20 ∧ 
            x₂^2 - 6*x₂ - 8 = 4*x₂ + 20 ∧ 
            x₁ ≠ x₂) →
  (x₁ + x₂ = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_l3252_325211


namespace NUMINAMATH_CALUDE_equation_solution_l3252_325263

theorem equation_solution : 
  {x : ℝ | (5 + x) / (7 + x) = (2 + x^2) / (4 + x)} = {1, -2, -3} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3252_325263


namespace NUMINAMATH_CALUDE_exterior_angle_parallel_lines_l3252_325209

theorem exterior_angle_parallel_lines (α β γ δ : ℝ) : 
  α = 40 → β = 40 → γ + δ = 180 → α + β + γ = 180 → δ = 80 := by sorry

end NUMINAMATH_CALUDE_exterior_angle_parallel_lines_l3252_325209


namespace NUMINAMATH_CALUDE_f_properties_l3252_325207

noncomputable def f (x : ℝ) : ℝ := (1 / (2^x - 1) + 1/2) * x^3

theorem f_properties :
  (∀ x : ℝ, x ≠ 0 → f x ≠ 0) ∧
  (∀ x : ℝ, x ≠ 0 → f (-x) = f x) ∧
  (∀ x : ℝ, x ≠ 0 → f x > 0) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3252_325207


namespace NUMINAMATH_CALUDE_gcd_of_180_and_270_l3252_325260

theorem gcd_of_180_and_270 : Nat.gcd 180 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_180_and_270_l3252_325260


namespace NUMINAMATH_CALUDE_least_k_correct_l3252_325239

/-- Sum of reciprocal values of non-zero digits of all positive integers up to and including n -/
def S (n : ℕ) : ℚ := sorry

/-- The least positive integer k such that k! * S_2016 is an integer -/
def least_k : ℕ := 7

theorem least_k_correct :
  (∀ m : ℕ, m < least_k → ¬(∃ z : ℤ, z = (m.factorial : ℚ) * S 2016)) ∧
  (∃ z : ℤ, z = (least_k.factorial : ℚ) * S 2016) := by sorry

end NUMINAMATH_CALUDE_least_k_correct_l3252_325239


namespace NUMINAMATH_CALUDE_shopping_mall_sales_l3252_325216

/-- Shopping mall sales problem -/
theorem shopping_mall_sales
  (initial_cost : ℝ)
  (initial_price : ℝ)
  (january_sales : ℝ)
  (march_sales : ℝ)
  (price_decrease : ℝ)
  (sales_increase : ℝ)
  (desired_profit : ℝ)
  (h1 : initial_cost = 60)
  (h2 : initial_price = 80)
  (h3 : january_sales = 64)
  (h4 : march_sales = 100)
  (h5 : price_decrease = 0.5)
  (h6 : sales_increase = 5)
  (h7 : desired_profit = 2160) :
  ∃ (growth_rate : ℝ) (optimal_price : ℝ),
    growth_rate = 0.25 ∧
    optimal_price = 72 ∧
    (1 + growth_rate)^2 * january_sales = march_sales ∧
    (optimal_price - initial_cost) * (march_sales + (sales_increase / price_decrease) * (initial_price - optimal_price)) = desired_profit :=
by sorry

end NUMINAMATH_CALUDE_shopping_mall_sales_l3252_325216


namespace NUMINAMATH_CALUDE_regular_pay_is_2_40_l3252_325296

/-- Calculates the regular pay per hour given the following conditions:
  - Regular week: 5 working days, 8 hours per day
  - Overtime pay: Rs. 3.20 per hour
  - Total earnings in 4 weeks: Rs. 432
  - Total hours worked in 4 weeks: 175 hours
-/
def regularPayPerHour (
  workingDaysPerWeek : ℕ)
  (workingHoursPerDay : ℕ)
  (overtimePay : ℚ)
  (totalEarnings : ℚ)
  (totalHoursWorked : ℕ) : ℚ :=
  let regularHoursPerWeek := workingDaysPerWeek * workingHoursPerDay
  let totalRegularHours := 4 * regularHoursPerWeek
  let overtimeHours := totalHoursWorked - totalRegularHours
  let overtimeEarnings := overtimeHours * overtimePay
  let regularEarnings := totalEarnings - overtimeEarnings
  regularEarnings / totalRegularHours

/-- Proves that the regular pay per hour is Rs. 2.40 given the specified conditions. -/
theorem regular_pay_is_2_40 :
  regularPayPerHour 5 8 (32/10) 432 175 = 24/10 := by
  sorry

end NUMINAMATH_CALUDE_regular_pay_is_2_40_l3252_325296


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l3252_325240

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 3^5 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes :
  ways_to_put_balls_in_boxes 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l3252_325240


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3252_325250

def polynomial (x : ℤ) : ℤ := x^3 - 4*x^2 - 11*x + 24

def is_root (x : ℤ) : Prop := polynomial x = 0

theorem integer_roots_of_polynomial :
  {x : ℤ | is_root x} = {-1, 3, 4} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3252_325250


namespace NUMINAMATH_CALUDE_parabola_properties_incorrect_statement_l3252_325200

-- Define the parabola
def parabola (x : ℝ) : ℝ := -(x - 1)^2 + 4

-- Statements to prove
theorem parabola_properties :
  -- The parabola opens downwards
  (∀ x : ℝ, parabola x ≤ parabola 1) ∧
  -- The shape is the same as y = x^2
  (∃ c : ℝ, ∀ x : ℝ, parabola x = c - x^2) ∧
  -- The vertex is (1,4)
  (parabola 1 = 4 ∧ ∀ x : ℝ, parabola x ≤ 4) ∧
  -- The axis of symmetry is the line x = 1
  (∀ x : ℝ, parabola (1 + x) = parabola (1 - x)) :=
by sorry

-- Statement C is incorrect
theorem incorrect_statement :
  ¬(parabola (-1) = 4 ∧ ∀ x : ℝ, parabola x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_incorrect_statement_l3252_325200


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l3252_325226

theorem quadratic_equation_proof (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, m * x^2 + 2*(m+1)*x + (m-1) = 0 ↔ x = x₁ ∨ x = x₂) →  -- equation has roots x₁ and x₂
  x₁ ≠ x₂ →  -- roots are distinct
  m > -1/3 →  -- condition from part 1
  m ≠ 0 →  -- condition from part 1
  x₁^2 + x₂^2 = 8 →  -- given condition
  m = 2 :=  -- conclusion
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l3252_325226


namespace NUMINAMATH_CALUDE_race_problem_l3252_325218

/-- The race problem -/
theorem race_problem (jack_first_half jack_second_half jill_total : ℕ) 
  (h1 : jack_first_half = 19)
  (h2 : jack_second_half = 6)
  (h3 : jill_total = 32) :
  jill_total - (jack_first_half + jack_second_half) = 7 := by
  sorry

end NUMINAMATH_CALUDE_race_problem_l3252_325218


namespace NUMINAMATH_CALUDE_base_seven_to_ten_63524_l3252_325286

/-- Converts a digit in base 7 to its value in base 10 -/
def baseSevenDigitToBaseTen (d : Nat) : Nat :=
  if d < 7 then d else 0

/-- Converts a list of digits in base 7 to its value in base 10 -/
def baseSevenToBaseTen (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 7 + baseSevenDigitToBaseTen d) 0

/-- The base 7 number 63524 converted to base 10 equals 15698 -/
theorem base_seven_to_ten_63524 :
  baseSevenToBaseTen [6, 3, 5, 2, 4] = 15698 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_to_ten_63524_l3252_325286


namespace NUMINAMATH_CALUDE_weights_not_divisible_by_three_l3252_325204

theorem weights_not_divisible_by_three :
  ¬ (∃ k : ℕ, 3 * k = (67 * 68) / 2) := by
  sorry

end NUMINAMATH_CALUDE_weights_not_divisible_by_three_l3252_325204


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l3252_325266

theorem pure_imaginary_fraction (a : ℝ) : 
  (∀ z : ℂ, z = (Complex.I : ℂ) / (1 + a * Complex.I) → Complex.re z = 0 ∧ Complex.im z ≠ 0) → 
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l3252_325266


namespace NUMINAMATH_CALUDE_maurice_prior_rides_eq_eight_l3252_325271

/-- The number of times Maurice rode during his visit -/
def maurice_visit_rides : ℕ := 8

/-- The number of times Matt rode without Maurice -/
def matt_solo_rides : ℕ := 16

/-- The total number of times Matt rode -/
def matt_total_rides : ℕ := maurice_visit_rides + matt_solo_rides

/-- The number of times Maurice rode before his visit -/
def maurice_prior_rides : ℕ := matt_total_rides / 3

theorem maurice_prior_rides_eq_eight :
  maurice_prior_rides = 8 := by sorry

end NUMINAMATH_CALUDE_maurice_prior_rides_eq_eight_l3252_325271


namespace NUMINAMATH_CALUDE_f_equals_g_l3252_325222

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := (x^4)^(1/4)

-- State the theorem
theorem f_equals_g : ∀ x : ℝ, f x = g x := by
  sorry

end NUMINAMATH_CALUDE_f_equals_g_l3252_325222


namespace NUMINAMATH_CALUDE_bridget_apples_bridget_apples_proof_l3252_325291

theorem bridget_apples : ℕ → Prop :=
  fun x =>
    let remaining_after_ann := x / 3
    let remaining_after_cassie := remaining_after_ann - 5
    let remaining_after_found := remaining_after_cassie + 3
    remaining_after_found = 6 → x = 24

-- Proof
theorem bridget_apples_proof : ∃ x : ℕ, bridget_apples x := by
  sorry

end NUMINAMATH_CALUDE_bridget_apples_bridget_apples_proof_l3252_325291


namespace NUMINAMATH_CALUDE_additional_rook_possible_l3252_325231

/-- Represents a 10x10 chessboard -/
def Board := Fin 10 → Fin 10 → Bool

/-- Checks if a rook at position (x, y) attacks another rook at position (x', y') -/
def attacks (x y x' y' : Fin 10) : Prop :=
  x = x' ∨ y = y'

/-- Represents a valid rook placement on the board -/
def ValidPlacement (b : Board) : Prop :=
  ∃ (n : Nat) (positions : Fin n → Fin 10 × Fin 10),
    n ≤ 8 ∧
    (∀ i j, i ≠ j → ¬attacks (positions i).1 (positions i).2 (positions j).1 (positions j).2) ∧
    (∀ i, b (positions i).1 (positions i).2 = true) ∧
    (∃ (blackCount whiteCount : Nat),
      blackCount = whiteCount ∧
      blackCount + whiteCount = n ∧
      (∀ i, (((positions i).1 + (positions i).2) % 2 = 0) = (i < blackCount)))

/-- The main theorem stating that an additional rook can be placed -/
theorem additional_rook_possible (b : Board) (h : ValidPlacement b) :
  ∃ (x y : Fin 10), b x y = false ∧ ∀ (x' y' : Fin 10), b x' y' = true → ¬attacks x y x' y' := by
  sorry

end NUMINAMATH_CALUDE_additional_rook_possible_l3252_325231
