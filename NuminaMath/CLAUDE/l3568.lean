import Mathlib

namespace NUMINAMATH_CALUDE_tangent_line_parabola_l3568_356877

/-- The equation of the tangent line to the parabola y = x^2 at the point (-1, 1) -/
theorem tangent_line_parabola :
  let f (x : ℝ) := x^2
  let p : ℝ × ℝ := (-1, 1)
  let tangent_line (x y : ℝ) := 2*x + y + 1 = 0
  (∀ x, (f x, x) ∈ Set.range (λ t => (t, f t))) →
  (p.1, p.2) ∈ Set.range (λ t => (t, f t)) →
  ∃ m b, (∀ x, tangent_line x (m*x + b)) ∧
         (tangent_line p.1 p.2) ∧
         (∀ ε > 0, ∃ δ > 0, ∀ x, |x - p.1| < δ → |f x - (m*x + b)| < ε * |x - p.1|) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_parabola_l3568_356877


namespace NUMINAMATH_CALUDE_questionnaire_responses_l3568_356882

theorem questionnaire_responses (response_rate : ℝ) (questionnaires_mailed : ℕ) 
  (h1 : response_rate = 0.8)
  (h2 : questionnaires_mailed = 375) :
  ⌊response_rate * questionnaires_mailed⌋ = 300 := by
  sorry

end NUMINAMATH_CALUDE_questionnaire_responses_l3568_356882


namespace NUMINAMATH_CALUDE_john_piggy_bank_balance_l3568_356848

/-- The amount John saves monthly in dollars -/
def monthly_savings : ℕ := 25

/-- The number of months John saves -/
def saving_period : ℕ := 2 * 12

/-- The amount John spends on car repairs in dollars -/
def car_repair_cost : ℕ := 400

/-- The amount left in John's piggy bank after savings and car repair -/
def piggy_bank_balance : ℕ := monthly_savings * saving_period - car_repair_cost

theorem john_piggy_bank_balance : piggy_bank_balance = 200 := by
  sorry

end NUMINAMATH_CALUDE_john_piggy_bank_balance_l3568_356848


namespace NUMINAMATH_CALUDE_min_soldiers_to_add_l3568_356860

theorem min_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) :
  ∃ x : ℕ, x = 82 ∧ 
    (∀ y : ℕ, y < x → ¬((N + y) % 7 = 0 ∧ (N + y) % 12 = 0)) ∧
    (N + x) % 7 = 0 ∧ (N + x) % 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_soldiers_to_add_l3568_356860


namespace NUMINAMATH_CALUDE_arithmetic_sequence_range_of_d_l3568_356827

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_range_of_d (d : ℝ) :
  (arithmetic_sequence 24 d 9 ≥ 0 ∧ arithmetic_sequence 24 d 10 < 0) →
  -3 ≤ d ∧ d < -8/3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_range_of_d_l3568_356827


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l3568_356820

/-- Given vectors p and q in ℝ², where p is parallel to q, prove that |p + q| = √13 -/
theorem parallel_vectors_sum_magnitude (p q : ℝ × ℝ) :
  p = (2, -3) →
  q.1 = x ∧ q.2 = 6 →
  (∃ (k : ℝ), q = k • p) →
  ‖p + q‖ = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l3568_356820


namespace NUMINAMATH_CALUDE_exists_n_with_specific_digit_sums_l3568_356828

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number n such that the sum of its digits is 100
    and the sum of the digits of n^3 is 1,000,000 -/
theorem exists_n_with_specific_digit_sums :
  ∃ n : ℕ, sum_of_digits n = 100 ∧ sum_of_digits (n^3) = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_with_specific_digit_sums_l3568_356828


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l3568_356813

/-- Given that i is the imaginary unit and (2+i)/(1+i) = a + bi where a and b are real numbers,
    prove that a + b = 1 -/
theorem complex_fraction_sum (i : ℂ) (a b : ℝ) 
    (h1 : i * i = -1) 
    (h2 : (2 + i) / (1 + i) = a + b * i) : 
  a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l3568_356813


namespace NUMINAMATH_CALUDE_intersection_union_problem_l3568_356886

theorem intersection_union_problem (x : ℝ) : 
  let A : Set ℝ := {1, 3, 5}
  let B : Set ℝ := {1, 2, x^2 - 1}
  (A ∩ B = {1, 3}) → (x = -2 ∧ A ∪ B = {1, 2, 3, 5}) := by
sorry

end NUMINAMATH_CALUDE_intersection_union_problem_l3568_356886


namespace NUMINAMATH_CALUDE_square_diff_equals_four_l3568_356835

theorem square_diff_equals_four (a b : ℝ) (h : a = b + 2) : a^2 - 2*a*b + b^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_equals_four_l3568_356835


namespace NUMINAMATH_CALUDE_sum_inequality_l3568_356867

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/(2*b) + 1/(3*c) = 1) : a + 2*b + 3*c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3568_356867


namespace NUMINAMATH_CALUDE_eight_term_sequence_sum_l3568_356858

def sequence_sum (seq : List ℤ) (i : ℕ) : ℤ :=
  if i + 2 < seq.length then
    seq[i]! + seq[i+1]! + seq[i+2]!
  else
    0

theorem eight_term_sequence_sum (P Q R S T U V W : ℤ) : 
  R = 8 →
  (∀ i, i + 2 < 8 → sequence_sum [P, Q, R, S, T, U, V, W] i = 35) →
  P + W = 27 := by
sorry

end NUMINAMATH_CALUDE_eight_term_sequence_sum_l3568_356858


namespace NUMINAMATH_CALUDE_sqrt_p_div_sqrt_q_l3568_356837

theorem sqrt_p_div_sqrt_q (p q : ℝ) (h : (1/3)^2 + (1/4)^2 = ((25*p)/(61*q)) * ((1/5)^2 + (1/6)^2)) :
  Real.sqrt p / Real.sqrt q = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_p_div_sqrt_q_l3568_356837


namespace NUMINAMATH_CALUDE_postcard_perimeter_l3568_356836

/-- The perimeter of a rectangle with width 6 inches and height 4 inches is 20 inches. -/
theorem postcard_perimeter : 
  let width : ℝ := 6
  let height : ℝ := 4
  let perimeter := 2 * (width + height)
  perimeter = 20 :=
by sorry

end NUMINAMATH_CALUDE_postcard_perimeter_l3568_356836


namespace NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l3568_356849

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l3568_356849


namespace NUMINAMATH_CALUDE_basketball_team_starters_l3568_356881

theorem basketball_team_starters (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 7 →
  (Nat.choose total_players starters) - (Nat.choose (total_players - quadruplets) (starters - quadruplets)) = 11220 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_starters_l3568_356881


namespace NUMINAMATH_CALUDE_find_y_l3568_356870

theorem find_y (x y : ℝ) (h1 : x^(2*y) = 16) (h2 : x = 2) : y = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l3568_356870


namespace NUMINAMATH_CALUDE_average_weight_increase_l3568_356804

theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 5 * initial_average
  let final_total := initial_total - 40 + 90
  let final_average := final_total / 5
  final_average - initial_average = 10 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3568_356804


namespace NUMINAMATH_CALUDE_mildreds_initial_blocks_l3568_356810

/-- Proves that Mildred's initial number of blocks was 2, given that she found 84 blocks and ended up with 86 blocks. -/
theorem mildreds_initial_blocks (found_blocks : ℕ) (final_blocks : ℕ) (h1 : found_blocks = 84) (h2 : final_blocks = 86) :
  final_blocks - found_blocks = 2 := by
  sorry

#check mildreds_initial_blocks

end NUMINAMATH_CALUDE_mildreds_initial_blocks_l3568_356810


namespace NUMINAMATH_CALUDE_water_evaporation_problem_l3568_356874

/-- Proves that given the conditions of water evaporation, the original amount of water is 12 ounces -/
theorem water_evaporation_problem (daily_evaporation : ℝ) (days : ℕ) (evaporation_percentage : ℝ) :
  daily_evaporation = 0.03 →
  days = 22 →
  evaporation_percentage = 0.055 →
  daily_evaporation * (days : ℝ) = evaporation_percentage * 12 :=
by sorry

end NUMINAMATH_CALUDE_water_evaporation_problem_l3568_356874


namespace NUMINAMATH_CALUDE_apple_pie_consumption_l3568_356815

theorem apple_pie_consumption (apples_per_serving : ℝ) (num_guests : ℕ) (num_pies : ℕ) (servings_per_pie : ℕ) :
  apples_per_serving = 1.5 →
  num_guests = 12 →
  num_pies = 3 →
  servings_per_pie = 8 →
  (num_pies * servings_per_pie * apples_per_serving) / num_guests = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_consumption_l3568_356815


namespace NUMINAMATH_CALUDE_cupcakes_left_after_distribution_l3568_356885

/-- Theorem: Cupcakes Left After Distribution

Given:
- Dani brings two and half dozen cupcakes
- There are 27 students (including Dani)
- There is 1 teacher
- There is 1 teacher's aid
- 3 students called in sick

Prove that the number of cupcakes left after Dani gives one to everyone in the class is 4.
-/
theorem cupcakes_left_after_distribution 
  (cupcakes_per_dozen : ℕ)
  (total_students : ℕ)
  (teacher_count : ℕ)
  (teacher_aid_count : ℕ)
  (sick_students : ℕ)
  (h1 : cupcakes_per_dozen = 12)
  (h2 : total_students = 27)
  (h3 : teacher_count = 1)
  (h4 : teacher_aid_count = 1)
  (h5 : sick_students = 3) :
  2 * cupcakes_per_dozen + cupcakes_per_dozen / 2 - 
  (total_students - sick_students + teacher_count + teacher_aid_count) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_left_after_distribution_l3568_356885


namespace NUMINAMATH_CALUDE_cos_seven_expansion_sum_of_squares_l3568_356869

theorem cos_seven_expansion_sum_of_squares : 
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ), 
    (∀ θ : ℝ, Real.cos θ ^ 7 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + 
      b₃ * Real.cos (3 * θ) + b₄ * Real.cos (4 * θ) + b₅ * Real.cos (5 * θ) + 
      b₆ * Real.cos (6 * θ) + b₇ * Real.cos (7 * θ)) →
    b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 + b₆^2 + b₇^2 = 429 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_expansion_sum_of_squares_l3568_356869


namespace NUMINAMATH_CALUDE_correct_additional_money_l3568_356850

/-- Calculates the additional money Jack needs to buy socks and shoes -/
def additional_money_needed (sock_price : ℝ) (shoe_price : ℝ) (jack_has : ℝ) : ℝ :=
  2 * sock_price + shoe_price - jack_has

/-- Proves that the additional money needed is correct -/
theorem correct_additional_money 
  (sock_price : ℝ) (shoe_price : ℝ) (jack_has : ℝ) 
  (h1 : sock_price = 9.5)
  (h2 : shoe_price = 92)
  (h3 : jack_has = 40) :
  additional_money_needed sock_price shoe_price jack_has = 71 :=
by sorry

end NUMINAMATH_CALUDE_correct_additional_money_l3568_356850


namespace NUMINAMATH_CALUDE_power_subtraction_equivalence_l3568_356811

theorem power_subtraction_equivalence :
  2^345 - 3^4 * 9^2 = 2^345 - 6561 :=
by sorry

end NUMINAMATH_CALUDE_power_subtraction_equivalence_l3568_356811


namespace NUMINAMATH_CALUDE_s_iff_q_r_iff_q_p_necessary_for_s_l3568_356818

-- Define the propositions
variable (p q r s : Prop)

-- Define the given conditions
axiom p_necessary_for_r : r → p
axiom q_necessary_for_r : r → q
axiom s_sufficient_for_r : s → r
axiom q_sufficient_for_s : q → s

-- Theorem statements
theorem s_iff_q : s ↔ q := by sorry

theorem r_iff_q : r ↔ q := by sorry

theorem p_necessary_for_s : s → p := by sorry

end NUMINAMATH_CALUDE_s_iff_q_r_iff_q_p_necessary_for_s_l3568_356818


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3568_356871

theorem simplify_trig_expression :
  Real.sqrt (2 - Real.sin 1 ^ 2 + Real.cos 2) = Real.sqrt 3 * Real.cos 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3568_356871


namespace NUMINAMATH_CALUDE_cos_2theta_value_l3568_356891

theorem cos_2theta_value (θ : ℝ) 
  (h1 : 3 * Real.sin (2 * θ) = 4 * Real.tan θ) 
  (h2 : ∀ k : ℤ, θ ≠ k * Real.pi) : 
  Real.cos (2 * θ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_2theta_value_l3568_356891


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3568_356868

theorem polynomial_simplification (p : ℝ) :
  (5 * p^3 - 7 * p^2 + 3 * p + 8) + (-3 * p^3 + 9 * p^2 - 4 * p + 2) =
  2 * p^3 + 2 * p^2 - p + 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3568_356868


namespace NUMINAMATH_CALUDE_alex_shirts_l3568_356845

theorem alex_shirts (alex joe ben : ℕ) 
  (h1 : joe = alex + 3)
  (h2 : ben = joe + 8)
  (h3 : ben = 15) : 
  alex = 4 := by
  sorry

end NUMINAMATH_CALUDE_alex_shirts_l3568_356845


namespace NUMINAMATH_CALUDE_power_of_ten_thousand_zeros_after_one_l3568_356824

theorem power_of_ten_thousand (n : ℕ) : (10000 : ℕ) ^ n = (10 : ℕ) ^ (4 * n) := by sorry

theorem zeros_after_one : (10000 : ℕ) ^ 50 = (10 : ℕ) ^ 200 := by sorry

end NUMINAMATH_CALUDE_power_of_ten_thousand_zeros_after_one_l3568_356824


namespace NUMINAMATH_CALUDE_ladder_problem_l3568_356864

theorem ladder_problem (ladder_length height : Real) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ base : Real, base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l3568_356864


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_l3568_356861

theorem negation_of_existence_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_l3568_356861


namespace NUMINAMATH_CALUDE_base7_perfect_square_last_digit_l3568_356878

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- Represents a number in base 7 as ab2c -/
structure Base7Rep where
  a : ℕ
  b : ℕ
  c : ℕ
  a_nonzero : a ≠ 0

/-- Converts a Base7Rep to its decimal equivalent -/
def toDecimal (rep : Base7Rep) : ℕ :=
  rep.a * 7^3 + rep.b * 7^2 + 2 * 7 + rep.c

theorem base7_perfect_square_last_digit (n : ℕ) (rep : Base7Rep) :
  isPerfectSquare n ∧ n = toDecimal rep → rep.c = 2 ∨ rep.c = 3 ∨ rep.c = 6 := by
  sorry

end NUMINAMATH_CALUDE_base7_perfect_square_last_digit_l3568_356878


namespace NUMINAMATH_CALUDE_height_estimate_theorem_l3568_356872

/-- Represents the regression line for estimating height from foot length -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Represents the sample statistics -/
structure SampleStats where
  mean_x : ℝ
  mean_y : ℝ

/-- Calculates the estimated height given a foot length and regression line -/
def estimate_height (x : ℝ) (line : RegressionLine) : ℝ :=
  line.slope * x + line.intercept

/-- Theorem stating that given the sample statistics and slope, 
    the estimated height for a foot length of 24 cm is 166 cm -/
theorem height_estimate_theorem 
  (stats : SampleStats) 
  (given_slope : ℝ) 
  (h_mean_x : stats.mean_x = 22.5) 
  (h_mean_y : stats.mean_y = 160) 
  (h_slope : given_slope = 4) :
  let line := RegressionLine.mk given_slope (stats.mean_y - given_slope * stats.mean_x)
  estimate_height 24 line = 166 := by
  sorry

#check height_estimate_theorem

end NUMINAMATH_CALUDE_height_estimate_theorem_l3568_356872


namespace NUMINAMATH_CALUDE_evaluate_expression_l3568_356816

theorem evaluate_expression (b : ℕ) (h : b = 4) :
  b^3 * b^6 * 2 = 524288 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3568_356816


namespace NUMINAMATH_CALUDE_expression_undefined_at_twelve_l3568_356898

theorem expression_undefined_at_twelve :
  ∀ x : ℝ, x = 12 → (x^2 - 24*x + 144 = 0) := by sorry

end NUMINAMATH_CALUDE_expression_undefined_at_twelve_l3568_356898


namespace NUMINAMATH_CALUDE_remainder_sum_mod_seven_l3568_356888

theorem remainder_sum_mod_seven (a b c : ℕ) : 
  0 < a ∧ a < 7 ∧ 0 < b ∧ b < 7 ∧ 0 < c ∧ c < 7 →
  (a * b * c) % 7 = 1 →
  (4 * c) % 7 = 5 →
  (5 * b) % 7 = (4 + b) % 7 →
  (a + b + c) % 7 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_seven_l3568_356888


namespace NUMINAMATH_CALUDE_xyz_sum_l3568_356875

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y = 32)
  (h2 : x * z = 64)
  (h3 : y * z = 96) :
  x + y + z = 28 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_l3568_356875


namespace NUMINAMATH_CALUDE_mean_temperature_l3568_356806

def temperatures : List ℤ := [-8, -3, -3, -6, 2, 4, 1]

theorem mean_temperature :
  (temperatures.sum : ℚ) / temperatures.length = -2 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l3568_356806


namespace NUMINAMATH_CALUDE_exactly_one_statement_implies_negation_l3568_356821

def statement1 (p q : Prop) : Prop := p ∨ q
def statement2 (p q : Prop) : Prop := p ∧ ¬q
def statement3 (p q : Prop) : Prop := ¬p ∧ q
def statement4 (p q : Prop) : Prop := ¬p ∧ ¬q

def negation_of_or (p q : Prop) : Prop := ¬(p ∨ q)

theorem exactly_one_statement_implies_negation (p q : Prop) :
  (∃! i : Fin 4, match i with
    | 0 => statement1 p q → negation_of_or p q
    | 1 => statement2 p q → negation_of_or p q
    | 2 => statement3 p q → negation_of_or p q
    | 3 => statement4 p q → negation_of_or p q) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_statement_implies_negation_l3568_356821


namespace NUMINAMATH_CALUDE_gcd_7163_209_l3568_356876

theorem gcd_7163_209 : Nat.gcd 7163 209 = 19 := by
  have h1 : 7163 = 209 * 34 + 57 := by sorry
  have h2 : 209 = 57 * 3 + 38 := by sorry
  have h3 : 57 = 38 * 1 + 19 := by sorry
  have h4 : 38 = 19 * 2 := by sorry
  sorry

end NUMINAMATH_CALUDE_gcd_7163_209_l3568_356876


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3568_356859

-- Define the original expression
def original_expression (x : ℝ) : ℝ := 4 * (x^2 - 2*x + 2) - 7 * (x^3 - 3*x + 1)

-- Define the simplified expression
def simplified_expression (x : ℝ) : ℝ := -7*x^3 + 4*x^2 + 13*x + 1

-- Theorem statement
theorem sum_of_squared_coefficients :
  ((-7)^2 + 4^2 + 13^2 + 1^2 = 235) ∧
  (∀ x : ℝ, original_expression x = simplified_expression x) :=
sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3568_356859


namespace NUMINAMATH_CALUDE_x_value_l3568_356819

theorem x_value : ∃ x : ℚ, (3 * x) / 7 = 12 ∧ x = 28 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3568_356819


namespace NUMINAMATH_CALUDE_intersection_distance_theorem_l3568_356844

/-- A linear function f(x) = ax + b -/
structure LinearFunction where
  a : ℝ
  b : ℝ

/-- The distance between intersection points of two functions -/
def intersectionDistance (f g : ℝ → ℝ) : ℝ := sorry

/-- The theorem statement -/
theorem intersection_distance_theorem (f : LinearFunction) :
  intersectionDistance (fun x => x^2 - 1) (fun x => f.a * x + f.b + 1) = 3 * Real.sqrt 10 →
  intersectionDistance (fun x => x^2) (fun x => f.a * x + f.b + 3) = 3 * Real.sqrt 14 →
  intersectionDistance (fun x => x^2) (fun x => f.a * x + f.b) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_theorem_l3568_356844


namespace NUMINAMATH_CALUDE_meaningful_range_l3568_356853

def is_meaningful (x : ℝ) : Prop :=
  3 - x ≥ 0 ∧ x - 1 > 0 ∧ x ≠ 2

theorem meaningful_range :
  ∀ x : ℝ, is_meaningful x ↔ (1 < x ∧ x ≤ 3 ∧ x ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_meaningful_range_l3568_356853


namespace NUMINAMATH_CALUDE_nonnegative_real_inequality_l3568_356809

theorem nonnegative_real_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x^2 * y^2 + x^2 * y + x * y^2 ≤ x^4 * y + x + y^4 := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_real_inequality_l3568_356809


namespace NUMINAMATH_CALUDE_two_numbers_sum_squares_and_product_l3568_356866

theorem two_numbers_sum_squares_and_product : ∃ u v : ℝ, 
  u^2 + v^2 = 20 ∧ u * v = 8 ∧ ((u = 2 ∧ v = 4) ∨ (u = 4 ∧ v = 2)) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_squares_and_product_l3568_356866


namespace NUMINAMATH_CALUDE_square_side_significant_digits_l3568_356880

/-- Given a square with area 0.12321 m², the number of significant digits in its side length is 5 -/
theorem square_side_significant_digits :
  ∀ (s : ℝ), 
  (s^2 ≥ 0.123205 ∧ s^2 < 0.123215) →  -- Area to the nearest ten-thousandth
  (∃ (n : ℕ), n ≥ 10000 ∧ n < 100000 ∧ s = (n : ℝ) / 100000) := by
  sorry

#check square_side_significant_digits

end NUMINAMATH_CALUDE_square_side_significant_digits_l3568_356880


namespace NUMINAMATH_CALUDE_line_y_intercept_implies_m_l3568_356865

/-- Given a line equation x + my + 3 - 2m = 0 with y-intercept -1, prove that m = 1 -/
theorem line_y_intercept_implies_m (m : ℝ) :
  (∀ x y : ℝ, x + m * y + 3 - 2 * m = 0) →  -- Line equation
  (0 + m * (-1) + 3 - 2 * m = 0) →          -- y-intercept is -1
  m = 1 :=                                  -- Conclusion: m = 1
by
  sorry

end NUMINAMATH_CALUDE_line_y_intercept_implies_m_l3568_356865


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l3568_356894

-- Define the parallelogram properties
def parallelogram_area : ℝ := 360
def parallelogram_height : ℝ := 12

-- Theorem statement
theorem parallelogram_base_length :
  parallelogram_area / parallelogram_height = 30 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l3568_356894


namespace NUMINAMATH_CALUDE_shirt_problem_l3568_356829

/-- Represents the problem of determining the number of shirts and minimum selling price --/
theorem shirt_problem (first_batch_cost second_batch_cost : ℕ) 
  (h1 : first_batch_cost = 13200)
  (h2 : second_batch_cost = 28800)
  (h3 : ∃ x : ℕ, x > 0 ∧ second_batch_cost / (2 * x) = first_batch_cost / x + 10)
  (h4 : ∃ y : ℕ, y > 0 ∧ 350 * y ≥ (first_batch_cost + second_batch_cost) * 125 / 100) :
  (∃ x : ℕ, x = 120 ∧ second_batch_cost / (2 * x) = first_batch_cost / x + 10) ∧
  (∃ y : ℕ, y = 150 ∧ 350 * y ≥ (first_batch_cost + second_batch_cost) * 125 / 100) :=
by sorry


end NUMINAMATH_CALUDE_shirt_problem_l3568_356829


namespace NUMINAMATH_CALUDE_height_edge_relationship_l3568_356812

/-- A triangular pyramid with mutually perpendicular edges -/
structure TriangularPyramid where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  h_pos : 0 < h
  perpendicular : True  -- Represents that SA, SB, and SC are mutually perpendicular

/-- The theorem about the relationship between height and edge lengths in a triangular pyramid -/
theorem height_edge_relationship (p : TriangularPyramid) : 
  1 / p.h^2 = 1 / p.a^2 + 1 / p.b^2 + 1 / p.c^2 := by
  sorry

end NUMINAMATH_CALUDE_height_edge_relationship_l3568_356812


namespace NUMINAMATH_CALUDE_sum_of_differences_correct_l3568_356851

def number : ℕ := 84125398

def place_value (digit : ℕ) (position : ℕ) : ℕ := digit * (10 ^ position)

def sum_of_differences (n : ℕ) : ℕ :=
  let ones_thousands := place_value 1 3
  let ones_tens := place_value 1 1
  let eights_hundred_millions := place_value 8 8
  let eights_tens := place_value 8 1
  (eights_hundred_millions - ones_thousands) + (eights_tens - ones_tens)

theorem sum_of_differences_correct :
  sum_of_differences number = 79999070 := by sorry

end NUMINAMATH_CALUDE_sum_of_differences_correct_l3568_356851


namespace NUMINAMATH_CALUDE_shaded_areas_equality_l3568_356802

theorem shaded_areas_equality (θ : Real) (h1 : 0 < θ) (h2 : θ < π / 4) :
  (∃ (r : Real), r > 0 ∧ θ * r^2 = (r^2 * Real.tan (2 * θ)) / 2) ↔ Real.tan (2 * θ) = 2 * θ := by
  sorry

end NUMINAMATH_CALUDE_shaded_areas_equality_l3568_356802


namespace NUMINAMATH_CALUDE_annie_gives_25_crayons_to_mary_l3568_356800

/-- Calculates the number of crayons Annie gives to Mary -/
def crayons_given_to_mary (new_pack : ℕ) (locker : ℕ) : ℕ :=
  let initial_total := new_pack + locker
  let from_bobby := locker / 2
  let final_total := initial_total + from_bobby
  final_total / 3

/-- Proves that Annie gives 25 crayons to Mary under the given conditions -/
theorem annie_gives_25_crayons_to_mary :
  crayons_given_to_mary 21 36 = 25 := by
  sorry

#eval crayons_given_to_mary 21 36

end NUMINAMATH_CALUDE_annie_gives_25_crayons_to_mary_l3568_356800


namespace NUMINAMATH_CALUDE_line_equation_l3568_356846

/-- A line parameterized by (x, y) = (3t + 6, 5t - 7) where t is a real number has the equation y = (5/3)x - 17 -/
theorem line_equation (t : ℝ) :
  let x : ℝ := 3 * t + 6
  let y : ℝ := 5 * t - 7
  y = (5/3) * x - 17 := by sorry

end NUMINAMATH_CALUDE_line_equation_l3568_356846


namespace NUMINAMATH_CALUDE_least_possible_third_side_length_l3568_356808

theorem least_possible_third_side_length (a b c : ℝ) : 
  a = 8 → b = 15 → c > 0 → 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c ≥ Real.sqrt 161 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_third_side_length_l3568_356808


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l3568_356892

theorem right_triangle_leg_length
  (a c h : ℝ)
  (ha : a = 5)
  (hh : h = 4)
  (right_triangle : a^2 + b^2 = c^2)
  (altitude : (1/2) * a * b = (1/2) * c * h) :
  b = 20/3 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l3568_356892


namespace NUMINAMATH_CALUDE_compound_ratio_proof_l3568_356895

theorem compound_ratio_proof : 
  let r1 : ℚ := 2/3
  let r2 : ℚ := 6/7
  let r3 : ℚ := 1/3
  let r4 : ℚ := 3/8
  (r1 * r2 * r3 * r4 : ℚ) = 0.07142857142857142 :=
by sorry

end NUMINAMATH_CALUDE_compound_ratio_proof_l3568_356895


namespace NUMINAMATH_CALUDE_g_composition_of_three_l3568_356841

def g (x : ℝ) : ℝ := 7 * x + 3

theorem g_composition_of_three : g (g (g 3)) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l3568_356841


namespace NUMINAMATH_CALUDE_f_properties_l3568_356847

-- Define the function f(x) = lg|x|
noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

-- State the theorem
theorem f_properties :
  -- f is defined for all real numbers except 0
  (∀ x : ℝ, x ≠ 0 → f x = Real.log (abs x)) →
  -- f is an even function
  (∀ x : ℝ, x ≠ 0 → f (-x) = f x) ∧
  -- f is monotonically increasing on (0, +∞)
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3568_356847


namespace NUMINAMATH_CALUDE_diagonal_sum_equals_fibonacci_l3568_356863

/-- The sum of binomial coefficients in a diagonal of Pascal's Triangle -/
def diagonalSum (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) (fun k => Nat.choose (n - k) k)

/-- The nth Fibonacci number -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The main theorem: The diagonal sum equals the (n+1)th Fibonacci number -/
theorem diagonal_sum_equals_fibonacci (n : ℕ) : diagonalSum n = fib (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_diagonal_sum_equals_fibonacci_l3568_356863


namespace NUMINAMATH_CALUDE_number_problem_l3568_356801

theorem number_problem (x : ℝ) : x = 456 ↔ 0.5 * x = 0.4 * 120 + 180 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3568_356801


namespace NUMINAMATH_CALUDE_probability_of_at_least_one_of_each_color_l3568_356826

-- Define the number of marbles of each color
def red_marbles : Nat := 3
def blue_marbles : Nat := 3
def green_marbles : Nat := 3

-- Define the total number of marbles
def total_marbles : Nat := red_marbles + blue_marbles + green_marbles

-- Define the number of marbles to be selected
def selected_marbles : Nat := 4

-- Define the probability of selecting at least one marble of each color
def prob_at_least_one_of_each : Rat := 9/14

-- Theorem statement
theorem probability_of_at_least_one_of_each_color :
  prob_at_least_one_of_each = 
    (Nat.choose red_marbles 1 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 2 +
     Nat.choose red_marbles 1 * Nat.choose blue_marbles 2 * Nat.choose green_marbles 1 +
     Nat.choose red_marbles 2 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 1) /
    Nat.choose total_marbles selected_marbles := by
  sorry

end NUMINAMATH_CALUDE_probability_of_at_least_one_of_each_color_l3568_356826


namespace NUMINAMATH_CALUDE_total_difference_of_sequences_l3568_356854

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem total_difference_of_sequences : 
  let n : ℕ := 72
  let d : ℕ := 3
  let a₁ : ℕ := 2001
  let b₁ : ℕ := 501
  arithmetic_sequence_sum a₁ d n - arithmetic_sequence_sum b₁ d n = 108000 := by
sorry

end NUMINAMATH_CALUDE_total_difference_of_sequences_l3568_356854


namespace NUMINAMATH_CALUDE_rene_received_300_l3568_356838

-- Define the amounts given to each person
def rene_amount : ℝ := sorry
def florence_amount : ℝ := sorry
def isha_amount : ℝ := sorry

-- Define the theorem
theorem rene_received_300 
  (h1 : florence_amount = 3 * rene_amount)
  (h2 : isha_amount = florence_amount / 2)
  (h3 : rene_amount + florence_amount + isha_amount = 1650)
  : rene_amount = 300 := by
  sorry

end NUMINAMATH_CALUDE_rene_received_300_l3568_356838


namespace NUMINAMATH_CALUDE_negative_square_cubed_l3568_356856

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l3568_356856


namespace NUMINAMATH_CALUDE_sequence_third_term_l3568_356862

theorem sequence_third_term (a : ℕ → ℕ) (h : ∀ n, a n = n^2 + n) : a 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sequence_third_term_l3568_356862


namespace NUMINAMATH_CALUDE_chosen_number_proof_l3568_356893

theorem chosen_number_proof (x : ℝ) : (x / 12) - 240 = 8 ↔ x = 2976 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l3568_356893


namespace NUMINAMATH_CALUDE_power_equation_solution_l3568_356884

theorem power_equation_solution : ∃! x : ℝ, (5 : ℝ)^3 + (5 : ℝ)^3 + (5 : ℝ)^3 = (15 : ℝ)^x := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3568_356884


namespace NUMINAMATH_CALUDE_series_sum_equals_one_six_hundredth_l3568_356839

/-- The sum of the series Σ(6n + 2) / ((6n - 1)^2 * (6n + 5)^2) from n=1 to infinity equals 1/600. -/
theorem series_sum_equals_one_six_hundredth :
  ∑' n : ℕ, (6 * n + 2) / ((6 * n - 1)^2 * (6 * n + 5)^2) = 1 / 600 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_six_hundredth_l3568_356839


namespace NUMINAMATH_CALUDE_field_length_calculation_l3568_356807

theorem field_length_calculation (width : ℝ) (pond_side : ℝ) : 
  width > 0 →
  pond_side = 4 →
  2 * width * width = 8 * (pond_side * pond_side) →
  2 * width = 16 := by
sorry

end NUMINAMATH_CALUDE_field_length_calculation_l3568_356807


namespace NUMINAMATH_CALUDE_platform_length_calculation_l3568_356817

-- Define the given parameters
def train_length : ℝ := 250
def train_speed_kmph : ℝ := 72
def train_speed_mps : ℝ := 20
def time_to_cross : ℝ := 25

-- Define the theorem
theorem platform_length_calculation :
  let total_distance := train_speed_mps * time_to_cross
  let platform_length := total_distance - train_length
  platform_length = 250 := by sorry

end NUMINAMATH_CALUDE_platform_length_calculation_l3568_356817


namespace NUMINAMATH_CALUDE_expand_product_l3568_356814

theorem expand_product (x : ℝ) : 3 * (x - 3) * (x + 5) = 3 * x^2 + 6 * x - 45 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3568_356814


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l3568_356855

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 61) 
  (h2 : throwers = 37) 
  (h3 : throwers ≤ total_players) 
  (h4 : (total_players - throwers) % 3 = 0) -- Ensures non-throwers can be divided into thirds
  : (throwers + (2 * (total_players - throwers) / 3)) = 53 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l3568_356855


namespace NUMINAMATH_CALUDE_tank_fill_time_l3568_356857

theorem tank_fill_time (r1 r2 r3 : ℚ) 
  (h1 : r1 = 1 / 18) 
  (h2 : r2 = 1 / 30) 
  (h3 : r3 = -1 / 45) : 
  (1 / (r1 + r2 + r3)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l3568_356857


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l3568_356832

theorem smallest_solution_quartic_equation :
  ∃ (x : ℝ), x^4 - 50*x^2 + 625 = 0 ∧ x = -5 ∧ ∀ y : ℝ, y^4 - 50*y^2 + 625 = 0 → y ≥ x := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l3568_356832


namespace NUMINAMATH_CALUDE_exp_ge_linear_l3568_356840

theorem exp_ge_linear (x : ℝ) : x + 1 ≤ Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_exp_ge_linear_l3568_356840


namespace NUMINAMATH_CALUDE_inverse_proportion_l3568_356843

/-- Given that p and q are inversely proportional, prove that if p = 20 when q = 8, then p = 16 when q = 10. -/
theorem inverse_proportion (p q : ℝ) (h : p * q = 20 * 8) : 
  p * 10 = 16 * 10 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l3568_356843


namespace NUMINAMATH_CALUDE_modular_inverse_7_mod_29_l3568_356822

theorem modular_inverse_7_mod_29 :
  ∃ x : ℕ, x < 29 ∧ (7 * x) % 29 = 1 ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_7_mod_29_l3568_356822


namespace NUMINAMATH_CALUDE_vasya_multiplication_error_l3568_356833

-- Define a structure for a two-digit number
structure TwoDigitNumber where
  tens : Fin 10
  ones : Fin 10
  different : tens ≠ ones

-- Define a structure for the result DDEE
structure ResultDDEE where
  d : Fin 10
  e : Fin 10
  different : d ≠ e

-- Define the main theorem
theorem vasya_multiplication_error 
  (ab vg : TwoDigitNumber) 
  (result : ResultDDEE) 
  (h1 : ab.tens ≠ vg.tens)
  (h2 : ab.tens ≠ vg.ones)
  (h3 : ab.ones ≠ vg.tens)
  (h4 : ab.ones ≠ vg.ones)
  (h5 : (ab.tens * 10 + ab.ones) * (vg.tens * 10 + vg.ones) = result.d * 1000 + result.d * 100 + result.e * 10 + result.e) :
  False :=
sorry

end NUMINAMATH_CALUDE_vasya_multiplication_error_l3568_356833


namespace NUMINAMATH_CALUDE_f_neg_three_gt_f_neg_five_l3568_356889

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y → f y < f x

-- State the theorem
theorem f_neg_three_gt_f_neg_five
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_decreasing : is_decreasing_on_nonneg f) :
  f (-3) > f (-5) :=
sorry

end NUMINAMATH_CALUDE_f_neg_three_gt_f_neg_five_l3568_356889


namespace NUMINAMATH_CALUDE_inequality_holds_l3568_356897

theorem inequality_holds (x : ℝ) : 
  -1/2 ≤ x ∧ x < 45/8 → (4 * x^2) / ((1 - Real.sqrt (1 + 2*x))^2) < 2*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3568_356897


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_221_l3568_356896

theorem modular_inverse_of_5_mod_221 : ∃ x : ℕ, x < 221 ∧ (5 * x) % 221 = 1 :=
by
  use 177
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_221_l3568_356896


namespace NUMINAMATH_CALUDE_biology_physics_ratio_l3568_356890

/-- The ratio of students in Biology class to Physics class -/
theorem biology_physics_ratio :
  let girls_biology : ℕ := 3 * 25
  let boys_biology : ℕ := 25
  let students_biology : ℕ := girls_biology + boys_biology
  let students_physics : ℕ := 200
  (students_biology : ℚ) / students_physics = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_biology_physics_ratio_l3568_356890


namespace NUMINAMATH_CALUDE_sunnyvale_fruit_punch_l3568_356873

/-- The total amount of fruit punch at Sunnyvale School's picnic -/
theorem sunnyvale_fruit_punch (orange_punch : ℝ) (cherry_punch : ℝ) (apple_juice : ℝ) 
    (h1 : orange_punch = 4.5)
    (h2 : cherry_punch = 2 * orange_punch)
    (h3 : apple_juice = cherry_punch - 1.5) :
    orange_punch + cherry_punch + apple_juice = 21 := by
  sorry

end NUMINAMATH_CALUDE_sunnyvale_fruit_punch_l3568_356873


namespace NUMINAMATH_CALUDE_ben_joe_shirt_difference_l3568_356823

/-- The number of new shirts Alex has -/
def alex_shirts : ℕ := 4

/-- The number of additional shirts Joe has compared to Alex -/
def joe_extra_shirts : ℕ := 3

/-- The number of new shirts Ben has -/
def ben_shirts : ℕ := 15

/-- The number of new shirts Joe has -/
def joe_shirts : ℕ := alex_shirts + joe_extra_shirts

theorem ben_joe_shirt_difference : ben_shirts - joe_shirts = 8 := by
  sorry

end NUMINAMATH_CALUDE_ben_joe_shirt_difference_l3568_356823


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l3568_356887

/-- Represents the possible stripe configurations on a cube face -/
inductive StripeConfig
| DiagonalA
| DiagonalB
| EdgeToEdgeA
| EdgeToEdgeB

/-- Represents a cube with stripes on each face -/
def StripedCube := Fin 6 → StripeConfig

/-- Checks if a given StripedCube has a continuous stripe encircling it -/
def hasContinuousStripe (cube : StripedCube) : Prop := sorry

/-- The total number of possible stripe configurations for a cube -/
def totalConfigurations : ℕ := 4^6

/-- The number of configurations that result in a continuous stripe -/
def continuousStripeConfigurations : ℕ := 48

theorem continuous_stripe_probability :
  (continuousStripeConfigurations : ℚ) / totalConfigurations = 3 / 256 := by sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l3568_356887


namespace NUMINAMATH_CALUDE_cos_225_degrees_l3568_356805

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l3568_356805


namespace NUMINAMATH_CALUDE_min_value_of_a_l3568_356879

def matrixOp (a b c d : ℝ) : ℝ := a * d - b * c

theorem min_value_of_a (a : ℝ) :
  (∀ x : ℝ, matrixOp (x - 1) (a - 2) (a + 1) x ≥ 1) →
  a ≥ -1/2 ∧ ∀ b, b < -1/2 → ∃ x, matrixOp (x - 1) (b - 2) (b + 1) x < 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l3568_356879


namespace NUMINAMATH_CALUDE_larger_square_construction_l3568_356883

/-- Represents a square in 2D space -/
structure Square where
  side : ℝ
  deriving Inhabited

/-- Represents the construction of a larger square from two smaller squares -/
def construct_larger_square (s1 s2 : Square) : Square :=
  sorry

/-- Theorem stating that it's possible to construct a larger square from two smaller squares
    without cutting the smaller square -/
theorem larger_square_construction (s1 s2 : Square) :
  ∃ (large : Square), 
    large.side^2 = s1.side^2 + s2.side^2 ∧
    construct_larger_square s1 s2 = large :=
  sorry

end NUMINAMATH_CALUDE_larger_square_construction_l3568_356883


namespace NUMINAMATH_CALUDE_pear_peach_weight_equivalence_l3568_356831

/-- If 9 pears weigh the same as 6 peaches, then 36 pears weigh the same as 24 peaches. -/
theorem pear_peach_weight_equivalence :
  ∀ (pear_weight peach_weight : ℝ),
  9 * pear_weight = 6 * peach_weight →
  36 * pear_weight = 24 * peach_weight :=
by
  sorry


end NUMINAMATH_CALUDE_pear_peach_weight_equivalence_l3568_356831


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3568_356803

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (a < b → a < b + 1) ∧ ¬(∀ a b : ℝ, a < b + 1 → a < b) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3568_356803


namespace NUMINAMATH_CALUDE_sqrt_of_sum_of_cubes_l3568_356834

theorem sqrt_of_sum_of_cubes : Real.sqrt (5 * (4^3 + 4^3 + 4^3 + 4^3)) = 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sum_of_cubes_l3568_356834


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l3568_356830

/-- A quadratic polynomial that satisfies specific conditions -/
def q (x : ℚ) : ℚ := (10/9) * x^2 + (4/9) * x + 4/9

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions : 
  q (-2) = 4 ∧ q 1 = 2 ∧ q 3 = 10 := by
  sorry

#eval q (-2)
#eval q 1
#eval q 3

end NUMINAMATH_CALUDE_q_satisfies_conditions_l3568_356830


namespace NUMINAMATH_CALUDE_equation_solutions_l3568_356899

/-- The set of solutions to the equation (a³ + b³)ⁿ = 4(ab)¹⁹⁹⁵ where a, b, n are integers greater than 1 -/
def Solutions : Set (ℕ × ℕ × ℕ) :=
  {(1, 1, 2), (2, 2, 998), (32, 32, 1247), (2^55, 2^55, 1322), (2^221, 2^221, 1328)}

/-- The predicate that checks if a triple (a, b, n) satisfies the equation (a³ + b³)ⁿ = 4(ab)¹⁹⁹⁵ -/
def SatisfiesEquation (a b n : ℕ) : Prop :=
  a > 1 ∧ b > 1 ∧ n > 1 ∧ (a^3 + b^3)^n = 4 * (a * b)^1995

theorem equation_solutions :
  ∀ a b n : ℕ, SatisfiesEquation a b n ↔ (a, b, n) ∈ Solutions := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3568_356899


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l3568_356825

theorem cubic_roots_sum (a b c : ℝ) : 
  a^3 - 15*a^2 + 25*a - 10 = 0 →
  b^3 - 15*b^2 + 25*b - 10 = 0 →
  c^3 - 15*c^2 + 25*c - 10 = 0 →
  a / ((1/a) + b*c) + b / ((1/b) + c*a) + c / ((1/c) + a*b) = 175/11 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l3568_356825


namespace NUMINAMATH_CALUDE_wickets_before_last_match_value_l3568_356842

/-- The number of wickets taken by a bowler before his last match -/
def wickets_before_last_match (initial_average : ℚ) (wickets_last_match : ℕ) 
  (runs_last_match : ℕ) (average_decrease : ℚ) : ℕ :=
  sorry

/-- Theorem stating the number of wickets taken before the last match -/
theorem wickets_before_last_match_value :
  wickets_before_last_match 12.4 3 26 0.4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_wickets_before_last_match_value_l3568_356842


namespace NUMINAMATH_CALUDE_matrix_power_equality_l3568_356852

def A (a : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![1, 3, a; 0, 1, 5; 0, 0, 1]

theorem matrix_power_equality (a : ℝ) (n : ℕ) :
  (A a) ^ n = !![1, 27, 3000; 0, 1, 45; 0, 0, 1] →
  a + n = 278 := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_equality_l3568_356852
