import Mathlib

namespace solve_quadratic_equation_solve_cubic_equation_l2113_211388

-- Problem 1
theorem solve_quadratic_equation (x : ℝ) :
  4 * x^2 - 81 = 0 ↔ x = 9/2 ∨ x = -9/2 := by
sorry

-- Problem 2
theorem solve_cubic_equation (x : ℝ) :
  8 * (x + 1)^3 = 27 ↔ x = 1/2 := by
sorry

end solve_quadratic_equation_solve_cubic_equation_l2113_211388


namespace parabola_parameter_value_l2113_211393

/-- Proves that for a parabola y^2 = 2px (p > 0) with axis of symmetry at distance 4 from the point (3, 0), the value of p is 2. -/
theorem parabola_parameter_value (p : ℝ) (h1 : p > 0) : 
  (∃ (x y : ℝ), y^2 = 2*p*x) →  -- Parabola equation
  (∃ (a : ℝ), ∀ (x y : ℝ), y^2 = 2*p*x → x = a) →  -- Axis of symmetry exists
  (|3 - (- p/2)| = 4) →  -- Distance from (3, 0) to axis of symmetry is 4
  p = 2 := by
sorry

end parabola_parameter_value_l2113_211393


namespace max_alpha_is_half_l2113_211338

/-- The set of functions satisfying the given condition -/
def F : Set (ℝ → ℝ) :=
  {f | ∀ x > 0, f (3 * x) ≥ f (f (2 * x)) + x}

/-- The theorem stating that 1/2 is the maximum α -/
theorem max_alpha_is_half :
    (∃ α : ℝ, ∀ f ∈ F, ∀ x > 0, f x ≥ α * x) ∧
    (∀ β : ℝ, (∀ f ∈ F, ∀ x > 0, f x ≥ β * x) → β ≤ 1/2) :=
  sorry


end max_alpha_is_half_l2113_211338


namespace algebraic_simplification_l2113_211391

theorem algebraic_simplification (y : ℝ) (h : y ≠ 0) :
  (-24 * y^3) * (5 * y^2) * (1 / (2*y)^3) = -15 * y^2 :=
sorry

end algebraic_simplification_l2113_211391


namespace coin_flip_probability_l2113_211363

/-- Represents the outcome of a coin flip -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the set of five coins -/
structure CoinSet :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)
  (half_dollar : CoinOutcome)

/-- The total number of possible outcomes when flipping five coins -/
def total_outcomes : Nat := 32

/-- Predicate for the desired outcome (penny, nickel, and half dollar are heads) -/
def desired_outcome (cs : CoinSet) : Prop :=
  cs.penny = CoinOutcome.Heads ∧
  cs.nickel = CoinOutcome.Heads ∧
  cs.half_dollar = CoinOutcome.Heads

/-- The number of outcomes satisfying the desired condition -/
def successful_outcomes : Nat := 4

/-- The probability of the desired outcome -/
def probability : ℚ := 1 / 8

theorem coin_flip_probability :
  (successful_outcomes : ℚ) / total_outcomes = probability :=
sorry

end coin_flip_probability_l2113_211363


namespace circles_intersect_l2113_211398

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Theorem statement
theorem circles_intersect : ∃ (x y : ℝ), C₁ x y ∧ C₂ x y :=
sorry

end circles_intersect_l2113_211398


namespace cone_volume_l2113_211307

/-- Given a cone with base area 2π and lateral area 4π, its volume is (2√6/3)π -/
theorem cone_volume (r l h : ℝ) (h_base_area : π * r^2 = 2) (h_lateral_area : π * r * l = 4) 
  (h_height : h^2 = l^2 - r^2) : 
  (1/3) * π * r^2 * h = (2 * Real.sqrt 6 / 3) * π := by
  sorry


end cone_volume_l2113_211307


namespace sticker_distribution_l2113_211322

/-- The number of ways to distribute n identical objects into k distinct containers --/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

theorem sticker_distribution : distribute 10 4 = 251 := by sorry

end sticker_distribution_l2113_211322


namespace tenthDrawnNumber_l2113_211362

/-- Represents the systematic sampling problem -/
def systematicSampling (totalStudents : Nat) (sampleSize : Nat) (firstDrawn : Nat) (nthDraw : Nat) : Nat :=
  let interval := totalStudents / sampleSize
  firstDrawn + interval * (nthDraw - 1)

/-- Theorem stating the 10th drawn number in the given systematic sampling scenario -/
theorem tenthDrawnNumber :
  systematicSampling 1000 50 15 10 = 195 := by
  sorry

end tenthDrawnNumber_l2113_211362


namespace second_question_correct_percentage_l2113_211325

theorem second_question_correct_percentage
  (first_correct : Real)
  (neither_correct : Real)
  (both_correct : Real)
  (h1 : first_correct = 0.63)
  (h2 : neither_correct = 0.20)
  (h3 : both_correct = 0.32) :
  ∃ (second_correct : Real),
    second_correct = 0.49 ∧
    first_correct + second_correct - both_correct = 1 - neither_correct :=
by sorry

end second_question_correct_percentage_l2113_211325


namespace right_triangle_hypotenuse_is_integer_l2113_211356

theorem right_triangle_hypotenuse_is_integer (n : ℤ) :
  let a : ℤ := 2 * n + 1
  let b : ℤ := 2 * n * (n + 1)
  let c : ℤ := 2 * n^2 + 2 * n + 1
  c^2 = a^2 + b^2 := by sorry

end right_triangle_hypotenuse_is_integer_l2113_211356


namespace ball_in_ice_l2113_211305

theorem ball_in_ice (r : ℝ) (h : r = 16.25) :
  let d := 30  -- diameter of the hole
  let depth := 10  -- depth of the hole
  let x := r - depth  -- distance from center of ball to surface
  d^2 / 4 + x^2 = r^2 := by sorry

end ball_in_ice_l2113_211305


namespace even_function_inequality_l2113_211397

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * m * x + 3

theorem even_function_inequality (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →
  f m (Real.sqrt 3) < f m (-Real.sqrt 2) ∧ f m (-Real.sqrt 2) < f m (-1) :=
by sorry

end even_function_inequality_l2113_211397


namespace absolute_value_equation_l2113_211366

theorem absolute_value_equation : ∃! x : ℝ, |x - 30| + |x - 24| = |3*x - 72| := by
  sorry

end absolute_value_equation_l2113_211366


namespace fraction_simplification_l2113_211327

theorem fraction_simplification : (8 : ℚ) / (5 * 42) = 4 / 105 := by
  sorry

end fraction_simplification_l2113_211327


namespace system_solution_l2113_211320

theorem system_solution : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ = 0 ∧ y₁ = 1/3) ∧ 
    (x₂ = 19/2 ∧ y₂ = -6) ∧
    (∀ x y : ℝ, (5*x*(y + 6) = 0 ∧ 2*x + 3*y = 1) → 
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂))) :=
by sorry

end system_solution_l2113_211320


namespace graph_translation_l2113_211326

-- Define the original function
def original_function (x : ℝ) : ℝ := 3 * x - 4

-- Define the transformation (moving up by 2 units)
def transform (f : ℝ → ℝ) : ℝ → ℝ := λ x => f x + 2

-- State the theorem
theorem graph_translation :
  ∀ x : ℝ, transform original_function x = 3 * x - 2 := by
sorry

end graph_translation_l2113_211326


namespace no_rational_multiples_of_pi_l2113_211340

theorem no_rational_multiples_of_pi (x y : ℚ) : 
  (∃ (m n : ℚ), x = m * Real.pi ∧ y = n * Real.pi) →
  0 < x → x < y → y < Real.pi / 2 →
  Real.tan x + Real.tan y = 2 →
  False :=
sorry

end no_rational_multiples_of_pi_l2113_211340


namespace unique_rebus_solution_l2113_211321

/-- Represents a four-digit number ABCD where A, B, C, D are distinct non-zero digits. -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0
  d_nonzero : d ≠ 0
  all_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
  a_digit : a < 10
  b_digit : b < 10
  c_digit : c < 10
  d_digit : d < 10

/-- The rebus equation ABCA = 182 * CD -/
def rebusEquation (n : FourDigitNumber) : Prop :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.a = 182 * (10 * n.c + n.d)

/-- Theorem stating that 2916 is the only solution to the rebus equation -/
theorem unique_rebus_solution :
  ∃! n : FourDigitNumber, rebusEquation n ∧ n.a = 2 ∧ n.b = 9 ∧ n.c = 1 ∧ n.d = 6 :=
sorry

end unique_rebus_solution_l2113_211321


namespace pie_rows_l2113_211302

/-- Given the number of pecan and apple pies, and the number of pies per row,
    calculate the number of complete rows. -/
def calculate_rows (pecan_pies apple_pies pies_per_row : ℕ) : ℕ :=
  (pecan_pies + apple_pies) / pies_per_row

/-- Theorem stating that with 16 pecan pies and 14 apple pies,
    arranged in rows of 5 pies each, there will be 6 complete rows. -/
theorem pie_rows : calculate_rows 16 14 5 = 6 := by
  sorry

end pie_rows_l2113_211302


namespace optimal_strategy_l2113_211357

/-- Represents the profit function for zongzi sales -/
def profit_function (x : ℝ) (a : ℝ) : ℝ := (a - 5) * x + 6000

/-- Represents the constraints on the number of boxes of type A zongzi -/
def valid_quantity (x : ℝ) : Prop := 100 ≤ x ∧ x ≤ 150

/-- Theorem stating the optimal purchasing strategy to maximize profit -/
theorem optimal_strategy (a : ℝ) (h1 : 0 < a) (h2 : a < 10) :
  (0 < a ∧ a < 5 → 
    ∀ x, valid_quantity x → profit_function 100 a ≥ profit_function x a) ∧
  (5 ≤ a ∧ a < 10 → 
    ∀ x, valid_quantity x → profit_function 150 a ≥ profit_function x a) :=
sorry

end optimal_strategy_l2113_211357


namespace smallest_n_below_threshold_l2113_211383

/-- The probability of drawing a red marble on the nth draw -/
def P (n : ℕ) : ℚ := 1 / (n * (n + 1))

/-- The number of boxes -/
def num_boxes : ℕ := 1000

/-- The threshold probability -/
def threshold : ℚ := 1 / 1000

theorem smallest_n_below_threshold :
  (∀ k < 32, P k ≥ threshold) ∧ P 32 < threshold := by sorry

end smallest_n_below_threshold_l2113_211383


namespace divisible_by_eleven_l2113_211387

theorem divisible_by_eleven (n : ℕ) : ∃ k : ℤ, 3^(2*n + 2) + 2^(6*n + 1) = 11 * k := by
  sorry

end divisible_by_eleven_l2113_211387


namespace equation_roots_arithmetic_progression_l2113_211365

theorem equation_roots_arithmetic_progression (a : ℝ) : 
  (∃ r d : ℝ, (∀ x : ℝ, x^8 + a*x^4 + 1 = 0 ↔ 
    x = (r - 3*d)^(1/4) ∨ x = (r - d)^(1/4) ∨ x = (r + d)^(1/4) ∨ x = (r + 3*d)^(1/4))) 
  → a = -82/9 :=
by sorry

end equation_roots_arithmetic_progression_l2113_211365


namespace egg_count_problem_l2113_211301

theorem egg_count_problem (count_sum : ℕ) (error_sum : ℤ) (actual_count : ℕ) : 
  count_sum = 3162 →
  (∃ (e1 e2 e3 : ℤ), (e1 = 1 ∨ e1 = -1) ∧ 
                     (e2 = 10 ∨ e2 = -10) ∧ 
                     (e3 = 100 ∨ e3 = -100) ∧ 
                     error_sum = e1 + e2 + e3) →
  7 * actual_count + error_sum = count_sum →
  actual_count = 439 := by
sorry

end egg_count_problem_l2113_211301


namespace coefficient_expansion_l2113_211368

theorem coefficient_expansion (a : ℝ) : 
  (Nat.choose 5 3) * a^3 = 80 → a = 2 := by
  sorry

end coefficient_expansion_l2113_211368


namespace dorothy_taxes_l2113_211330

/-- Calculates the amount left after taxes given an annual income and tax rate. -/
def amountLeftAfterTaxes (annualIncome : ℝ) (taxRate : ℝ) : ℝ :=
  annualIncome * (1 - taxRate)

/-- Proves that given an annual income of $60,000 and a tax rate of 18%, 
    the amount left after taxes is $49,200. -/
theorem dorothy_taxes : 
  amountLeftAfterTaxes 60000 0.18 = 49200 := by
  sorry

end dorothy_taxes_l2113_211330


namespace product_213_16_l2113_211312

theorem product_213_16 : 213 * 16 = 3408 := by
  sorry

end product_213_16_l2113_211312


namespace max_value_of_f_l2113_211313

theorem max_value_of_f (x : ℝ) : 
  let f := fun (x : ℝ) => 1 / (1 - x * (1 - x))
  f x ≤ 4/3 ∧ ∃ y, f y = 4/3 := by
  sorry

end max_value_of_f_l2113_211313


namespace sin_cos_identity_l2113_211370

theorem sin_cos_identity (α : Real) (h : Real.sin α + 2 * Real.cos α = 0) :
  2 * Real.sin α * Real.cos α - Real.cos α ^ 2 = -1 := by
  sorry

end sin_cos_identity_l2113_211370


namespace graph_not_in_third_quadrant_l2113_211354

def f (x : ℝ) : ℝ := -x + 2

theorem graph_not_in_third_quadrant :
  ∀ x y : ℝ, f x = y → ¬(x < 0 ∧ y < 0) := by
  sorry

end graph_not_in_third_quadrant_l2113_211354


namespace ball_probabilities_l2113_211343

/-- The total number of balls in the bag -/
def total_balls : ℕ := 12

/-- The number of red balls initially in the bag -/
def red_balls : ℕ := 4

/-- The number of black balls in the bag -/
def black_balls : ℕ := 8

/-- The probability of drawing a black ball after removing m red balls -/
def prob_black (m : ℕ) : ℚ :=
  black_balls / (total_balls - m)

/-- The probability of drawing a black ball after removing n red balls -/
def prob_black_n (n : ℕ) : ℚ :=
  black_balls / (total_balls - n)

theorem ball_probabilities :
  (prob_black 4 = 1) ∧
  (prob_black 2 > 0 ∧ prob_black 2 < 1) ∧
  (prob_black 3 > 0 ∧ prob_black 3 < 1) ∧
  (prob_black_n 3 = 8/9) :=
sorry

end ball_probabilities_l2113_211343


namespace total_height_is_24cm_l2113_211331

/-- The number of washers in the stack -/
def num_washers : ℕ := 11

/-- The thickness of each washer in cm -/
def washer_thickness : ℝ := 2

/-- The outer diameter of the top washer in cm -/
def top_diameter : ℝ := 24

/-- The outer diameter of the bottom washer in cm -/
def bottom_diameter : ℝ := 4

/-- The decrease in diameter between consecutive washers in cm -/
def diameter_decrease : ℝ := 2

/-- The extra height for hooks at top and bottom in cm -/
def hook_height : ℝ := 2

theorem total_height_is_24cm : 
  (num_washers : ℝ) * washer_thickness + hook_height = 24 := by
  sorry

end total_height_is_24cm_l2113_211331


namespace valentines_dog_biscuits_l2113_211349

theorem valentines_dog_biscuits (num_dogs : ℕ) (biscuits_per_dog : ℕ) : 
  num_dogs = 2 → biscuits_per_dog = 3 → num_dogs * biscuits_per_dog = 6 :=
by sorry

end valentines_dog_biscuits_l2113_211349


namespace function_value_theorem_l2113_211353

/-- A function f(x) = a(x+2)^2 + 3 passing through points (-2, 3) and (0, 7) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 2)^2 + 3

/-- The theorem stating that given the conditions, a+3a+2 equals 6 -/
theorem function_value_theorem (a : ℝ) :
  f a (-2) = 3 ∧ f a 0 = 7 → a + 3*a + 2 = 6 := by
  sorry


end function_value_theorem_l2113_211353


namespace shelter_dogs_l2113_211395

/-- The number of dogs in an animal shelter given specific ratios -/
theorem shelter_dogs (d c : ℕ) (h1 : d * 7 = c * 15) (h2 : d * 11 = (c + 20) * 15) : d = 175 := by
  sorry

end shelter_dogs_l2113_211395


namespace placemats_length_l2113_211372

theorem placemats_length (R : ℝ) (n : ℕ) (x : ℝ) : 
  R = 5 ∧ n = 8 → x = 2 * R * Real.sin (π / (2 * n)) := by
  sorry

end placemats_length_l2113_211372


namespace r₂_bound_bound_is_tight_l2113_211369

-- Define the function f
def f (r₂ r₃ : ℝ) (x : ℝ) : ℝ := x^2 - r₂ * x + r₃

-- Define the sequence g
def g (r₂ r₃ : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => f r₂ r₃ (g r₂ r₃ n)

-- Define the conditions on the sequence
def sequence_conditions (r₂ r₃ : ℝ) : Prop :=
  (∀ i ≤ 2011, g r₂ r₃ (2*i) < g r₂ r₃ (2*i + 1) ∧ g r₂ r₃ (2*i + 1) > g r₂ r₃ (2*i + 2)) ∧
  (∃ j : ℕ, ∀ i > j, g r₂ r₃ (i + 1) > g r₂ r₃ i) ∧
  (∀ M : ℝ, ∃ N : ℕ, g r₂ r₃ N > M)

theorem r₂_bound (r₂ r₃ : ℝ) (h : sequence_conditions r₂ r₃) : |r₂| > 2 :=
  sorry

theorem bound_is_tight : ∀ ε > 0, ∃ r₂ r₃ : ℝ, sequence_conditions r₂ r₃ ∧ |r₂| < 2 + ε :=
  sorry

end r₂_bound_bound_is_tight_l2113_211369


namespace largest_n_for_unique_k_l2113_211345

theorem largest_n_for_unique_k : ∃ (k : ℤ),
  (8 : ℚ) / 15 < (112 : ℚ) / (112 + k) ∧ (112 : ℚ) / (112 + k) < 7 / 13 ∧
  ∀ (m : ℕ) (k' : ℤ), m > 112 →
    ((8 : ℚ) / 15 < (m : ℚ) / (m + k') ∧ (m : ℚ) / (m + k') < 7 / 13 →
     ∃ (k'' : ℤ), k'' ≠ k' ∧ (8 : ℚ) / 15 < (m : ℚ) / (m + k'') ∧ (m : ℚ) / (m + k'') < 7 / 13) :=
by sorry

#check largest_n_for_unique_k

end largest_n_for_unique_k_l2113_211345


namespace weight_relationship_and_sum_l2113_211318

/-- Given the weights of Haley, Verna, and Sherry, prove their relationship and combined weight -/
theorem weight_relationship_and_sum (haley_weight verna_weight sherry_weight : ℕ) : 
  haley_weight = 103 →
  verna_weight = haley_weight + 17 →
  verna_weight * 2 = sherry_weight →
  verna_weight + sherry_weight = 360 := by
  sorry

end weight_relationship_and_sum_l2113_211318


namespace angle_A_measure_l2113_211373

-- Define the angles A and B
def angle_A : ℝ := sorry
def angle_B : ℝ := sorry

-- State the theorem
theorem angle_A_measure :
  (angle_A = 2 * angle_B - 15) →  -- Condition 1
  (angle_A + angle_B = 180) →     -- Condition 2 (supplementary angles)
  angle_A = 115 := by             -- Conclusion
sorry


end angle_A_measure_l2113_211373


namespace intersecting_circles_common_chord_l2113_211310

/-- Two intersecting circles with given radii and distance between centers have a common chord of length 10 -/
theorem intersecting_circles_common_chord 
  (R : ℝ) 
  (r : ℝ) 
  (d : ℝ) 
  (h1 : R = 13) 
  (h2 : r = 5) 
  (h3 : d = 12) :
  ∃ (chord_length : ℝ), 
    chord_length = 10 ∧ 
    chord_length = 2 * R * Real.sqrt (1 - ((R^2 + d^2 - r^2) / (2 * R * d))^2) := by
  sorry

end intersecting_circles_common_chord_l2113_211310


namespace sum_cube_minus_twice_sum_square_is_zero_l2113_211352

theorem sum_cube_minus_twice_sum_square_is_zero
  (p q r s : ℝ)
  (sum_condition : p + q + r + s = 8)
  (sum_square_condition : p^2 + q^2 + r^2 + s^2 = 16) :
  p^3 + q^3 + r^3 + s^3 - 2*(p^2 + q^2 + r^2 + s^2) = 0 :=
by sorry

end sum_cube_minus_twice_sum_square_is_zero_l2113_211352


namespace correct_calculation_l2113_211344

theorem correct_calculation (x : ℤ) (h : x + 44 - 39 = 63) : x + 39 - 44 = 53 := by
  sorry

end correct_calculation_l2113_211344


namespace bella_steps_to_meet_l2113_211317

/-- The number of steps Bella takes before meeting Ella -/
def steps_to_meet (total_distance : ℕ) (bella_step_length : ℕ) (ella_speed_multiplier : ℕ) : ℕ :=
  let distance_to_meet := total_distance / 2
  let bella_speed := 1
  let ella_speed := ella_speed_multiplier * bella_speed
  let combined_speed := bella_speed + ella_speed
  let distance_bella_walks := (distance_to_meet * bella_speed) / combined_speed
  distance_bella_walks / bella_step_length

/-- Theorem stating that Bella takes 528 steps before meeting Ella -/
theorem bella_steps_to_meet :
  steps_to_meet 15840 3 4 = 528 := by
  sorry

end bella_steps_to_meet_l2113_211317


namespace missing_digit_in_103rd_rising_number_l2113_211390

/-- A rising number is a positive integer each digit of which is larger than each of the digits to its left. -/
def IsRisingNumber (n : ℕ) : Prop := sorry

/-- The set of all five-digit rising numbers using digits from 1 to 9. -/
def FiveDigitRisingNumbers : Set ℕ := {n : ℕ | IsRisingNumber n ∧ n ≥ 10000 ∧ n < 100000}

/-- The 103rd element in the ordered set of five-digit rising numbers. -/
def OneHundredThirdRisingNumber : ℕ := sorry

theorem missing_digit_in_103rd_rising_number :
  ¬ (∃ (d : ℕ), d = 5 ∧ 10 * (OneHundredThirdRisingNumber / 10) + d = OneHundredThirdRisingNumber) :=
sorry

end missing_digit_in_103rd_rising_number_l2113_211390


namespace apple_profit_percentage_l2113_211304

theorem apple_profit_percentage 
  (total_apples : ℝ)
  (first_portion : ℝ)
  (second_portion : ℝ)
  (second_profit : ℝ)
  (overall_profit : ℝ)
  (h1 : total_apples = 280)
  (h2 : first_portion = 0.4)
  (h3 : second_portion = 0.6)
  (h4 : first_portion + second_portion = 1)
  (h5 : second_profit = 0.3)
  (h6 : overall_profit = 0.26)
  : ∃ (first_profit : ℝ),
    first_profit * first_portion * total_apples + 
    second_profit * second_portion * total_apples = 
    overall_profit * total_apples ∧
    first_profit = 0.2 := by
sorry

end apple_profit_percentage_l2113_211304


namespace sin_300_degrees_l2113_211314

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_degrees_l2113_211314


namespace smallest_two_digit_with_digit_product_12_l2113_211371

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → 26 ≤ n :=
sorry

end smallest_two_digit_with_digit_product_12_l2113_211371


namespace mean_score_is_94_5_l2113_211376

structure ScoreData where
  score : ℕ
  count : ℕ

def total_students : ℕ := 120

def score_distribution : List ScoreData := [
  ⟨120, 12⟩,
  ⟨110, 19⟩,
  ⟨100, 33⟩,
  ⟨90, 30⟩,
  ⟨75, 15⟩,
  ⟨65, 9⟩,
  ⟨50, 2⟩
]

def total_score : ℕ := score_distribution.foldl (fun acc data => acc + data.score * data.count) 0

theorem mean_score_is_94_5 :
  (total_score : ℚ) / total_students = 94.5 := by
  sorry

end mean_score_is_94_5_l2113_211376


namespace complex_magnitude_equation_l2113_211375

theorem complex_magnitude_equation (n : ℝ) (h : n > 0) :
  Complex.abs (5 + Complex.I * n) = 5 * Real.sqrt 13 → n = 10 * Real.sqrt 3 := by
  sorry

end complex_magnitude_equation_l2113_211375


namespace unique_n_reaches_16_l2113_211394

def g (n : ℕ) : ℕ :=
  if n % 2 = 1 then n^2 + 1 else (n / 2)^2

theorem unique_n_reaches_16 :
  ∃! n : ℕ, n ∈ Finset.range 100 ∧
  ∃ k : ℕ, (k.iterate g n) = 16 :=
sorry

end unique_n_reaches_16_l2113_211394


namespace exists_increasing_interval_l2113_211386

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := log x + 1 / log x

-- State the theorem
theorem exists_increasing_interval :
  ∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > x₀, deriv f x > 0 :=
sorry

end exists_increasing_interval_l2113_211386


namespace female_managers_count_l2113_211378

/-- Represents a company with employees and managers -/
structure Company where
  total_employees : ℕ
  total_managers : ℕ
  male_employees : ℕ
  male_managers : ℕ
  female_employees : ℕ
  female_managers : ℕ

/-- The conditions of the company as described in the problem -/
def company_conditions (c : Company) : Prop :=
  c.total_managers = (2 * c.total_employees) / 5 ∧
  c.male_managers = (2 * c.male_employees) / 5 ∧
  c.female_employees = 750

/-- The theorem stating that under the given conditions, the number of female managers is 300 -/
theorem female_managers_count (c : Company) 
  (h : company_conditions c) : c.female_managers = 300 := by
  sorry


end female_managers_count_l2113_211378


namespace top_z_conference_teams_l2113_211361

theorem top_z_conference_teams (n : ℕ) : n * (n - 1) / 2 = 45 → n = 10 := by
  sorry

end top_z_conference_teams_l2113_211361


namespace chord_length_l2113_211380

/-- The length of chord AB formed by the intersection of a line and a circle -/
theorem chord_length (A B : ℝ × ℝ) : 
  (∀ (x y : ℝ), x + Real.sqrt 3 * y - 2 = 0 → x^2 + y^2 = 4 → (x, y) = A ∨ (x, y) = B) →
  ((A.1 + Real.sqrt 3 * A.2 - 2 = 0 ∧ A.1^2 + A.2^2 = 4) ∧
   (B.1 + Real.sqrt 3 * B.2 - 2 = 0 ∧ B.1^2 + B.2^2 = 4)) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 := by
  sorry

end chord_length_l2113_211380


namespace double_beavers_half_time_beavers_build_dam_l2113_211347

/-- Represents the time (in hours) it takes a given number of beavers to build a dam -/
def build_time (num_beavers : ℕ) : ℝ := 
  if num_beavers = 18 then 8 else 
  if num_beavers = 36 then 4 else 0

/-- The proposition that doubling the number of beavers halves the build time -/
theorem double_beavers_half_time : 
  build_time 36 = (build_time 18) / 2 := by
sorry

/-- The main theorem stating that 36 beavers can build the dam in 4 hours -/
theorem beavers_build_dam : 
  build_time 36 = 4 := by
sorry

end double_beavers_half_time_beavers_build_dam_l2113_211347


namespace arithmetic_sequence_sum_property_l2113_211335

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ) (d : ℝ) 
  (h_arithmetic : arithmetic_sequence a d)
  (h_sum : a 3 + a 5 + a 7 + a 9 + a 11 = 100) :
  3 * a 9 - a 13 = 2 * (a 1 + 6 * d) :=
sorry

end arithmetic_sequence_sum_property_l2113_211335


namespace stickers_given_correct_l2113_211377

/-- Represents the number of stickers Willie gave to Emily -/
def stickers_given (initial final : ℕ) : ℕ := initial - final

/-- Proves that the number of stickers Willie gave to Emily is correct -/
theorem stickers_given_correct (initial final : ℕ) (h : initial ≥ final) :
  stickers_given initial final = initial - final :=
by
  sorry

end stickers_given_correct_l2113_211377


namespace birthday_money_ratio_l2113_211392

theorem birthday_money_ratio : 
  ∀ (total_money video_game_cost goggles_cost money_left : ℚ),
    total_money = 100 →
    video_game_cost = total_money / 4 →
    money_left = 60 →
    goggles_cost = total_money - video_game_cost - money_left →
    goggles_cost / (total_money - video_game_cost) = 1 / 5 := by
  sorry

end birthday_money_ratio_l2113_211392


namespace number_division_problem_l2113_211381

theorem number_division_problem :
  ∃! n : ℕ, 
    n / (555 + 445) = 2 * (555 - 445) ∧ 
    n % (555 + 445) = 70 := by
  sorry

end number_division_problem_l2113_211381


namespace element_in_union_l2113_211303

theorem element_in_union (M N : Set ℕ) (a : ℕ) 
  (h1 : M ∪ N = {1, 2, 3})
  (h2 : M ∩ N = {a}) : 
  a ∈ M ∪ N := by
  sorry

end element_in_union_l2113_211303


namespace det_A_equals_one_l2113_211350

theorem det_A_equals_one (a b c : ℝ) (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A = ![![2*a, b], ![c, -2*a]] →
  A + A⁻¹ = 0 →
  Matrix.det A = 1 := by sorry

end det_A_equals_one_l2113_211350


namespace cube_sum_greater_than_product_sufficient_l2113_211351

theorem cube_sum_greater_than_product_sufficient (x y z : ℝ) : 
  x + y + z > 0 → x^3 + y^3 + z^3 > 3*x*y*z := by
  sorry

end cube_sum_greater_than_product_sufficient_l2113_211351


namespace smallest_four_digit_in_pascal_triangle_l2113_211339

def pascal_triangle (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def is_in_pascal_triangle (x : ℕ) : Prop :=
  ∃ n k, pascal_triangle n k = x

def is_four_digit (x : ℕ) : Prop :=
  1000 ≤ x ∧ x < 10000

theorem smallest_four_digit_in_pascal_triangle :
  (is_in_pascal_triangle 1000) ∧
  (∀ x, is_in_pascal_triangle x → is_four_digit x → 1000 ≤ x) :=
sorry

end smallest_four_digit_in_pascal_triangle_l2113_211339


namespace larger_number_proof_l2113_211355

theorem larger_number_proof (x y : ℝ) (h1 : y > x) (h2 : 5 * y = 6 * x) (h3 : y - x = 12) : y = 72 := by
  sorry

end larger_number_proof_l2113_211355


namespace parabola_translation_l2113_211379

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := 3 * x^2

/-- The translated parabola function -/
def translated_parabola (x : ℝ) : ℝ := 3 * (x - 1)^2 - 4

/-- Theorem stating that the translated_parabola is the result of
    translating the original_parabola 1 unit right and 4 units down -/
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = original_parabola (x - 1) - 4 :=
by
  sorry

end parabola_translation_l2113_211379


namespace probability_male_student_id_l2113_211389

theorem probability_male_student_id (male_count female_count : ℕ) 
  (h1 : male_count = 6) (h2 : female_count = 4) : 
  (male_count : ℚ) / ((male_count : ℚ) + (female_count : ℚ)) = 3 / 5 := by
  sorry

end probability_male_student_id_l2113_211389


namespace unique_sum_property_l2113_211329

theorem unique_sum_property (A : ℕ) : 
  (0 ≤ A * (A - 1999) / 2 ∧ A * (A - 1999) / 2 ≤ 999) ↔ A = 1999 :=
sorry

end unique_sum_property_l2113_211329


namespace people_who_left_line_l2113_211358

theorem people_who_left_line (initial_people final_people joined_people : ℕ) 
  (h1 : initial_people = 30)
  (h2 : final_people = 25)
  (h3 : joined_people = 5)
  (h4 : final_people = initial_people - (people_who_left : ℕ) + joined_people) :
  people_who_left = 10 := by
sorry

end people_who_left_line_l2113_211358


namespace friends_games_l2113_211309

theorem friends_games (katie_games : ℕ) (difference : ℕ) : 
  katie_games = 81 → difference = 22 → katie_games - difference = 59 := by
  sorry

end friends_games_l2113_211309


namespace functional_equation_properties_l2113_211328

/-- A function satisfying the given functional equation -/
noncomputable def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

theorem functional_equation_properties (f : ℝ → ℝ) 
  (h_eq : FunctionalEquation f) (h_nonzero : f 0 ≠ 0) : 
  (f 0 = 1) ∧ 
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∃ c : ℝ, (∀ x : ℝ, f (x + 2*c) = f x) ∧ f c = -1) := by
  sorry

end functional_equation_properties_l2113_211328


namespace total_cost_after_discount_l2113_211399

def mountain_bike_initial : ℝ := 250
def helmet_initial : ℝ := 60
def gloves_initial : ℝ := 30

def mountain_bike_increase : ℝ := 0.08
def helmet_increase : ℝ := 0.15
def gloves_increase : ℝ := 0.10

def discount : ℝ := 0.05

theorem total_cost_after_discount : 
  let mountain_bike_new := mountain_bike_initial * (1 + mountain_bike_increase)
  let helmet_new := helmet_initial * (1 + helmet_increase)
  let gloves_new := gloves_initial * (1 + gloves_increase)
  let total_before_discount := mountain_bike_new + helmet_new + gloves_new
  let total_after_discount := total_before_discount * (1 - discount)
  total_after_discount = 353.4 := by
  sorry

end total_cost_after_discount_l2113_211399


namespace expression_value_l2113_211367

theorem expression_value (a b : ℤ) (h1 : a = 4) (h2 : b = -5) : 
  -a^2 - b^2 + a*b + b = -66 := by
  sorry

end expression_value_l2113_211367


namespace exists_permutation_distinct_columns_l2113_211348

/-- A table is represented as a function from pairs of indices to integers -/
def Table (n : ℕ) := Fin n → Fin n → ℤ

/-- A predicate stating that no two cells within a row share the same number -/
def DistinctInRows (t : Table n) : Prop :=
  ∀ i j₁ j₂, j₁ ≠ j₂ → t i j₁ ≠ t i j₂

/-- A permutation of a row is a bijection on Fin n -/
def RowPermutation (n : ℕ) := Fin n ≃ Fin n

/-- Apply a row permutation to a table -/
def ApplyRowPermutation (t : Table n) (p : Fin n → RowPermutation n) : Table n :=
  λ i j ↦ t i ((p i).toFun j)

/-- A predicate stating that all columns contain distinct numbers -/
def DistinctInColumns (t : Table n) : Prop :=
  ∀ j i₁ i₂, i₁ ≠ i₂ → t i₁ j ≠ t i₂ j

/-- The main theorem -/
theorem exists_permutation_distinct_columns (n : ℕ) (t : Table n) 
    (h : DistinctInRows t) : 
    ∃ p : Fin n → RowPermutation n, DistinctInColumns (ApplyRowPermutation t p) := by
  sorry

end exists_permutation_distinct_columns_l2113_211348


namespace quadratic_inequality_range_l2113_211308

theorem quadratic_inequality_range (m : ℝ) :
  (¬∃ x : ℝ, x^2 - 2*x + m ≤ 0) ↔ m > 1 := by
  sorry

end quadratic_inequality_range_l2113_211308


namespace average_salary_calculation_l2113_211396

/-- Calculates the average salary of all employees in an office --/
theorem average_salary_calculation (officer_salary : ℕ) (non_officer_salary : ℕ) 
  (officer_count : ℕ) (non_officer_count : ℕ) :
  officer_salary = 470 →
  non_officer_salary = 110 →
  officer_count = 15 →
  non_officer_count = 525 →
  (officer_salary * officer_count + non_officer_salary * non_officer_count) / 
    (officer_count + non_officer_count) = 120 := by
  sorry

#check average_salary_calculation

end average_salary_calculation_l2113_211396


namespace darry_total_steps_l2113_211382

/-- Represents the number of steps climbed on a ladder -/
structure LadderClimb where
  steps : Nat
  times : Nat

/-- Calculates the total number of steps climbed on a ladder -/
def totalStepsOnLadder (climb : LadderClimb) : Nat :=
  climb.steps * climb.times

/-- Represents Darry's ladder climbs for the day -/
structure DarryClimbs where
  largest : LadderClimb
  medium : LadderClimb
  smaller : LadderClimb
  smallest : LadderClimb

/-- Darry's actual climbs for the day -/
def darryActualClimbs : DarryClimbs :=
  { largest := { steps := 20, times := 12 }
  , medium := { steps := 15, times := 8 }
  , smaller := { steps := 10, times := 10 }
  , smallest := { steps := 5, times := 15 }
  }

/-- Calculates the total number of steps Darry climbed -/
def totalStepsClimbed (climbs : DarryClimbs) : Nat :=
  totalStepsOnLadder climbs.largest +
  totalStepsOnLadder climbs.medium +
  totalStepsOnLadder climbs.smaller +
  totalStepsOnLadder climbs.smallest

/-- Theorem stating that Darry climbed 535 steps in total -/
theorem darry_total_steps :
  totalStepsClimbed darryActualClimbs = 535 := by
  sorry

end darry_total_steps_l2113_211382


namespace x_with_three_prime_divisors_including_2_l2113_211332

theorem x_with_three_prime_divisors_including_2 (x n : ℕ) :
  x = 2^n - 32 ∧ 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 2 ∧ q ≠ 2 ∧
    (∀ r : ℕ, Prime r → r ∣ x → r = 2 ∨ r = p ∨ r = q)) →
  x = 2016 ∨ x = 16352 := by
sorry

end x_with_three_prime_divisors_including_2_l2113_211332


namespace triangle_circumradius_l2113_211342

/-- Given a triangle ABC with area S = (1/2) * sin A * sin B * sin C, 
    the radius R of its circumcircle is equal to 1/2. -/
theorem triangle_circumradius (A B C : ℝ) (a b c : ℝ) (S : ℝ) (R : ℝ) : 
  S = (1/2) * Real.sin A * Real.sin B * Real.sin C →
  R = 1/2 := by
  sorry

end triangle_circumradius_l2113_211342


namespace xy_value_l2113_211324

theorem xy_value (x y : ℝ) (h : x * (x + 2*y) = x^2 + 12) : x * y = 6 := by
  sorry

end xy_value_l2113_211324


namespace problem_solution_l2113_211364

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 4 * x + a

def g (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

def t1 (a : ℝ) (x : ℝ) : ℝ := (1/2) * f a x

def t2 (a : ℝ) (x : ℝ) : ℝ := g a x

def t3 (x : ℝ) : ℝ := 2^x

theorem problem_solution (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ m : ℝ, (¬∃ y : ℝ, (∀ x ∈ Set.Icc (-1) (2*m), (f a x ≤ f a y) ∨ (∀ x ∈ Set.Icc (-1) (2*m), f a x ≥ f a y))) ↔ m > 1/2) ∧
  (f a 1 = g a 1 ↔ a = 2) ∧
  (∀ x ∈ Set.Ioo 0 1, t2 a x < t1 a x ∧ t1 a x < t3 x) :=
sorry

end problem_solution_l2113_211364


namespace diamonds_formula_diamonds_G15_l2113_211359

/-- The number of diamonds in figure G_n -/
def diamonds (n : ℕ+) : ℕ :=
  6 * n

/-- The theorem stating that the number of diamonds in G_n is 6n -/
theorem diamonds_formula (n : ℕ+) : diamonds n = 6 * n := by
  sorry

/-- Corollary: The number of diamonds in G_15 is 90 -/
theorem diamonds_G15 : diamonds 15 = 90 := by
  sorry

end diamonds_formula_diamonds_G15_l2113_211359


namespace geometric_sequence_sum_l2113_211323

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_pos : a 1 > 0)
  (h_sum : a 4 + a 7 = 2)
  (h_prod : a 5 * a 6 = -8) :
  a 1 + a 4 + a 7 + a 10 = -5 :=
sorry

end geometric_sequence_sum_l2113_211323


namespace trapezoid_not_axisymmetric_l2113_211311

-- Define the shapes
inductive Shape
  | Angle
  | Rectangle
  | Trapezoid
  | Rhombus

-- Define the property of being axisymmetric
def is_axisymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Angle => True
  | Shape.Rectangle => True
  | Shape.Rhombus => True
  | Shape.Trapezoid => false

-- Theorem stating that trapezoid is the only shape not necessarily axisymmetric
theorem trapezoid_not_axisymmetric :
  ∀ (s : Shape), ¬is_axisymmetric s ↔ s = Shape.Trapezoid :=
by sorry

end trapezoid_not_axisymmetric_l2113_211311


namespace min_teams_for_players_l2113_211337

theorem min_teams_for_players (total_players : ℕ) (max_per_team : ℕ) (min_teams : ℕ) : 
  total_players = 30 → 
  max_per_team = 7 → 
  min_teams = 5 → 
  (∀ t : ℕ, t < min_teams → t * max_per_team < total_players) ∧ 
  (min_teams * (total_players / min_teams) = total_players) := by
  sorry

end min_teams_for_players_l2113_211337


namespace incenter_is_intersection_of_angle_bisectors_l2113_211336

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- The distance from a point to a line segment -/
noncomputable def distanceToSide (P : Point) (side : Point × Point) : ℝ := sorry

/-- The angle bisector of an angle in a triangle -/
noncomputable def angleBisector (vertex : Point) (side1 : Point) (side2 : Point) : Point × Point := sorry

/-- The intersection point of two lines -/
noncomputable def lineIntersection (line1 : Point × Point) (line2 : Point × Point) : Point := sorry

theorem incenter_is_intersection_of_angle_bisectors (T : Triangle) :
  ∃ (P : Point),
    (∀ (side : Point × Point), 
      side ∈ [(T.A, T.B), (T.B, T.C), (T.C, T.A)] → 
      distanceToSide P side = distanceToSide P (T.A, T.B)) ↔
    (P = lineIntersection 
      (angleBisector T.A T.B T.C) 
      (angleBisector T.B T.C T.A)) :=
by sorry

end incenter_is_intersection_of_angle_bisectors_l2113_211336


namespace smallest_n_congruence_l2113_211374

theorem smallest_n_congruence (n : ℕ) : n > 0 → (∀ k < n, (7^k : ℤ) % 5 ≠ k^7 % 5) → (7^n : ℤ) % 5 = n^7 % 5 → n = 2 := by
  sorry

end smallest_n_congruence_l2113_211374


namespace divisibility_condition_l2113_211360

theorem divisibility_condition (a b c : ℝ) :
  ∀ n : ℕ, (∃ k : ℝ, a^n * (b - c) + b^n * (c - a) + c^n * (a - b) = k * (a^2 + b^2 + c^2 + a*b + b*c + c*a)) ↔ n = 4 := by
  sorry

end divisibility_condition_l2113_211360


namespace largest_fraction_l2113_211346

theorem largest_fraction : 
  let fractions := [5/12, 7/15, 29/58, 151/303, 199/400]
  ∀ x ∈ fractions, (29:ℚ)/58 ≥ x := by sorry

end largest_fraction_l2113_211346


namespace cd_player_cost_l2113_211316

/-- The amount spent on the CD player, given the total amount spent and the amounts spent on speakers and tires. -/
theorem cd_player_cost (total spent_on_speakers spent_on_tires : ℚ) 
  (h_total : total = 387.85)
  (h_speakers : spent_on_speakers = 136.01)
  (h_tires : spent_on_tires = 112.46) :
  total - (spent_on_speakers + spent_on_tires) = 139.38 := by
  sorry

end cd_player_cost_l2113_211316


namespace integer_average_sum_l2113_211300

theorem integer_average_sum (a b c d : ℤ) 
  (h1 : (a + b + c) / 3 + d = 29)
  (h2 : (b + c + d) / 3 + a = 23)
  (h3 : (a + c + d) / 3 + b = 21)
  (h4 : (a + b + d) / 3 + c = 17) :
  a = 21 ∨ b = 21 ∨ c = 21 ∨ d = 21 :=
by sorry

end integer_average_sum_l2113_211300


namespace circuit_probability_l2113_211334

/-- The probability that a circuit with two independently controlled switches
    connected in parallel can operate normally. -/
theorem circuit_probability (p1 p2 : ℝ) (h1 : p1 = 0.5) (h2 : p2 = 0.7) :
  p1 * (1 - p2) + (1 - p1) * p2 + p1 * p2 = 0.85 := by
  sorry

end circuit_probability_l2113_211334


namespace tv_watching_time_equivalence_l2113_211315

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of hours Ava watched television -/
def hours_watched : ℕ := 4

/-- The theorem stating that watching TV for 4 hours is equivalent to 240 minutes -/
theorem tv_watching_time_equivalence : 
  hours_watched * minutes_per_hour = 240 := by
  sorry

end tv_watching_time_equivalence_l2113_211315


namespace intersection_of_M_and_N_l2113_211341

def M : Set ℕ := {0, 1, 3}

def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem intersection_of_M_and_N : M ∩ N = {0, 3} := by sorry

end intersection_of_M_and_N_l2113_211341


namespace exponential_dominance_l2113_211385

theorem exponential_dominance (k : ℝ) (hk : k > 0) :
  ∃ x₀ : ℝ, ∀ x ≥ x₀, (2 : ℝ) ^ ((2 : ℝ) ^ x) > ((2 : ℝ) ^ x) ^ k :=
sorry

end exponential_dominance_l2113_211385


namespace function_inequality_implies_a_bound_l2113_211384

open Real

theorem function_inequality_implies_a_bound 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = (x - 1) / (Real.exp x))
  (h2 : ∀ t ∈ Set.Icc (1/2) 2, f t > t) :
  ∃ a, a > Real.exp 2 + 1/2 ∧ ∀ x ∈ Set.Icc (1/2) 2, (a - 1) / (Real.exp x) > x :=
sorry

end function_inequality_implies_a_bound_l2113_211384


namespace log_half_decreasing_l2113_211306

-- Define the function f(x) = log_(1/2)(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- State the theorem
theorem log_half_decreasing :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ > f x₂ := by
  sorry

end log_half_decreasing_l2113_211306


namespace travis_bowls_problem_l2113_211319

/-- Represents the problem of calculating the number of bowls Travis initially had --/
theorem travis_bowls_problem :
  let base_fee : ℕ := 100
  let safe_bowl_pay : ℕ := 3
  let lost_bowl_fee : ℕ := 4
  let lost_bowls : ℕ := 12
  let broken_bowls : ℕ := 15
  let total_payment : ℕ := 1825

  ∃ (total_bowls safe_bowls : ℕ),
    total_bowls = safe_bowls + lost_bowls + broken_bowls ∧
    total_payment = base_fee + safe_bowl_pay * safe_bowls - lost_bowl_fee * (lost_bowls + broken_bowls) ∧
    total_bowls = 638 :=
by
  sorry

end travis_bowls_problem_l2113_211319


namespace little_twelve_games_l2113_211333

/-- Represents a basketball conference with two divisions -/
structure BasketballConference :=
  (teams_per_division : ℕ)
  (intra_division_games : ℕ)
  (inter_division_games : ℕ)

/-- The Little Twelve Basketball Conference setup -/
def little_twelve : BasketballConference :=
  { teams_per_division := 6,
    intra_division_games := 2,
    inter_division_games := 2 }

/-- Calculate the total number of conference games -/
def total_conference_games (conf : BasketballConference) : ℕ :=
  let total_teams := 2 * conf.teams_per_division
  let games_per_team := (conf.teams_per_division - 1) * conf.intra_division_games +
                        conf.teams_per_division * conf.inter_division_games
  (total_teams * games_per_team) / 2

theorem little_twelve_games :
  total_conference_games little_twelve = 132 := by
  sorry

end little_twelve_games_l2113_211333
