import Mathlib

namespace NUMINAMATH_CALUDE_quadrilateral_problem_l2750_275048

/-- Prove that for a quadrilateral PQRS with specific vertex coordinates,
    consecutive integer side lengths, and an area of 50,
    the product of the odd integer scale factor and the sum of side lengths is 5. -/
theorem quadrilateral_problem (a b k : ℤ) : 
  a > b ∧ b > 0 ∧  -- a and b are consecutive integers with a > b > 0
  ∃ n : ℤ, a = b + 1 ∧  -- a and b are consecutive integers
  ∃ m : ℤ, k = 2 * m + 1 ∧  -- k is an odd integer
  2 * k^2 * (a - b) * (a + b) = 50 →  -- area of PQRS is 50
  k * (a + b) = 5 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_problem_l2750_275048


namespace NUMINAMATH_CALUDE_cube_preserves_order_l2750_275082

theorem cube_preserves_order (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l2750_275082


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2750_275091

/-- Proves that a parabola and a line intersect at two specific points -/
theorem parabola_line_intersection :
  let parabola (x : ℝ) := 2 * x^2 - 8 * x + 10
  let line (x : ℝ) := x + 1
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    parabola x₁ = line x₁ ∧
    parabola x₂ = line x₂ ∧
    ((x₁ = 3 ∧ parabola x₁ = 4) ∨ (x₁ = 3/2 ∧ parabola x₁ = 5/2)) ∧
    ((x₂ = 3 ∧ parabola x₂ = 4) ∨ (x₂ = 3/2 ∧ parabola x₂ = 5/2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2750_275091


namespace NUMINAMATH_CALUDE_shoe_box_problem_l2750_275088

theorem shoe_box_problem (num_pairs : ℕ) (prob_match : ℚ) :
  num_pairs = 6 →
  prob_match = 1 / 11 →
  (num_pairs * 2 : ℕ) = 12 :=
by sorry

end NUMINAMATH_CALUDE_shoe_box_problem_l2750_275088


namespace NUMINAMATH_CALUDE_chipmunk_families_count_l2750_275019

theorem chipmunk_families_count (families_left families_went_away : ℕ) 
  (h1 : families_left = 21)
  (h2 : families_went_away = 65) :
  families_left + families_went_away = 86 := by
  sorry

end NUMINAMATH_CALUDE_chipmunk_families_count_l2750_275019


namespace NUMINAMATH_CALUDE_work_completion_time_l2750_275068

/-- Given that two workers A and B can complete a work together in 16 days,
    and A alone can complete the work in 24 days, prove that B alone will
    complete the work in 48 days. -/
theorem work_completion_time
  (joint_time : ℝ) (a_time : ℝ) (b_time : ℝ)
  (h1 : joint_time = 16)
  (h2 : a_time = 24)
  (h3 : (1 / joint_time) = (1 / a_time) + (1 / b_time)) :
  b_time = 48 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2750_275068


namespace NUMINAMATH_CALUDE_sqrt_seven_fraction_inequality_l2750_275047

theorem sqrt_seven_fraction_inequality (m n : ℤ) 
  (h1 : m ≥ 1) (h2 : n ≥ 1) (h3 : Real.sqrt 7 - (m : ℝ) / n > 0) : 
  Real.sqrt 7 - (m : ℝ) / n > 1 / (m * n) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_fraction_inequality_l2750_275047


namespace NUMINAMATH_CALUDE_hours_to_weeks_l2750_275087

/-- Proves that 2016 hours is equivalent to 12 weeks -/
theorem hours_to_weeks : 
  (∀ (week : ℕ) (day : ℕ) (hour : ℕ), 
    (1 : ℕ) * week = 7 * day ∧ 
    (1 : ℕ) * day = 24 * hour) → 
  2016 = 12 * (7 * 24) :=
by sorry

end NUMINAMATH_CALUDE_hours_to_weeks_l2750_275087


namespace NUMINAMATH_CALUDE_exactly_three_sequences_l2750_275030

/-- Represents a sequence of 10 positive integers -/
def Sequence := Fin 10 → ℕ+

/-- Checks if a sequence satisfies the recurrence relation -/
def satisfies_recurrence (s : Sequence) : Prop :=
  ∀ n : Fin 8, s (n.succ.succ) = s (n.succ) + s n

/-- Checks if a sequence has the required last term -/
def has_correct_last_term (s : Sequence) : Prop :=
  s 9 = 2002

/-- The main theorem stating that there are exactly 3 valid sequences -/
theorem exactly_three_sequences :
  ∃! (sequences : Finset Sequence),
    sequences.card = 3 ∧
    ∀ s ∈ sequences, satisfies_recurrence s ∧ has_correct_last_term s :=
sorry

end NUMINAMATH_CALUDE_exactly_three_sequences_l2750_275030


namespace NUMINAMATH_CALUDE_min_value_theorem_l2750_275054

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2/a + 1/b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2/x + 1/y = 2 → 3*x + y ≥ (7 + 2*Real.sqrt 6)/2) ∧
  ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 2/a₀ + 1/b₀ = 2 ∧ 3*a₀ + b₀ = (7 + 2*Real.sqrt 6)/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2750_275054


namespace NUMINAMATH_CALUDE_long_jump_distance_difference_long_jump_distance_difference_holds_l2750_275092

/-- Proves that Margarita ran and jumped 1 foot farther than Ricciana -/
theorem long_jump_distance_difference : ℕ → Prop :=
  fun margarita_total =>
    let ricciana_run := 20
    let ricciana_jump := 4
    let ricciana_total := ricciana_run + ricciana_jump
    let margarita_run := 18
    let margarita_jump := 2 * ricciana_jump - 1
    margarita_total = margarita_run + margarita_jump ∧
    margarita_total - ricciana_total = 1

/-- The theorem holds for Margarita's total distance of 25 feet -/
theorem long_jump_distance_difference_holds : long_jump_distance_difference 25 := by
  sorry

end NUMINAMATH_CALUDE_long_jump_distance_difference_long_jump_distance_difference_holds_l2750_275092


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2750_275052

theorem sum_of_coefficients (b₆ b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (5*x - 2)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 729 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2750_275052


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2750_275076

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x + 1, -2)
  let b : ℝ × ℝ := (-2*x, 3)
  parallel a b → x = 3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2750_275076


namespace NUMINAMATH_CALUDE_population_approximation_l2750_275010

def initial_population : ℝ := 14999.999999999998
def first_year_change : ℝ := 0.12
def second_year_change : ℝ := 0.12

def population_after_two_years : ℝ :=
  initial_population * (1 + first_year_change) * (1 - second_year_change)

theorem population_approximation :
  ∃ ε > 0, |population_after_two_years - 14784| < ε :=
sorry

end NUMINAMATH_CALUDE_population_approximation_l2750_275010


namespace NUMINAMATH_CALUDE_flag_arrangement_count_remainder_mod_1000_l2750_275055

/-- The number of red flags -/
def red_flags : ℕ := 11

/-- The number of white flags -/
def white_flags : ℕ := 6

/-- The total number of flags -/
def total_flags : ℕ := red_flags + white_flags

/-- The number of distinguishable flagpoles -/
def flagpoles : ℕ := 2

/-- Represents a valid flag arrangement -/
structure FlagArrangement where
  arrangement : List Bool
  red_count : ℕ
  white_count : ℕ
  no_adjacent_white : Bool
  at_least_one_per_pole : Bool

/-- The number of valid distinguishable arrangements -/
def valid_arrangements : ℕ := 10164

theorem flag_arrangement_count :
  (∃ (arrangements : List FlagArrangement),
    (∀ a ∈ arrangements,
      a.red_count = red_flags ∧
      a.white_count = white_flags ∧
      a.no_adjacent_white = true ∧
      a.at_least_one_per_pole = true) ∧
    arrangements.length = valid_arrangements) :=
sorry

theorem remainder_mod_1000 :
  valid_arrangements % 1000 = 164 :=
sorry

end NUMINAMATH_CALUDE_flag_arrangement_count_remainder_mod_1000_l2750_275055


namespace NUMINAMATH_CALUDE_compare_log_and_sqrt_l2750_275034

theorem compare_log_and_sqrt : 2 + Real.log 6 / Real.log 2 > 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_compare_log_and_sqrt_l2750_275034


namespace NUMINAMATH_CALUDE_unique_triplet_solution_l2750_275044

theorem unique_triplet_solution :
  ∀ x y p : ℕ+,
  p.Prime →
  (x.val * y.val^3 : ℚ) / (x.val + y.val) = p.val →
  x = 14 ∧ y = 2 ∧ p = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_triplet_solution_l2750_275044


namespace NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l2750_275021

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 5

def contains_2_and_5 (n : ℕ) : Prop :=
  2 ∈ n.digits 10 ∧ 5 ∈ n.digits 10

theorem smallest_valid_number_last_four_digits :
  ∃ m : ℕ,
    m > 0 ∧
    m % 6 = 0 ∧
    m % 5 = 0 ∧
    is_valid_number m ∧
    contains_2_and_5 m ∧
    (∀ n : ℕ, n > 0 → n % 6 = 0 → n % 5 = 0 → is_valid_number n → contains_2_and_5 n → m ≤ n) ∧
    m % 10000 = 5220 :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l2750_275021


namespace NUMINAMATH_CALUDE_f_zero_points_iff_k_range_l2750_275014

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then k * x + 2 else (1/2) ^ x

def has_three_zero_points (k : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (∀ x, f k (f k x) - 3/2 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)

theorem f_zero_points_iff_k_range :
  ∀ k, has_three_zero_points k ↔ -1/2 < k ∧ k ≤ -1/4 :=
sorry

end NUMINAMATH_CALUDE_f_zero_points_iff_k_range_l2750_275014


namespace NUMINAMATH_CALUDE_max_abs_z3_l2750_275061

theorem max_abs_z3 (z₁ z₂ z₃ : ℂ) 
  (h1 : Complex.abs z₁ ≤ 1)
  (h2 : Complex.abs z₂ ≤ 1)
  (h3 : Complex.abs (2 * z₃ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂)) :
  Complex.abs z₃ ≤ Real.sqrt 2 ∧ 
  ∃ w₁ w₂ w₃ : ℂ, Complex.abs w₁ ≤ 1 ∧ 
               Complex.abs w₂ ≤ 1 ∧ 
               Complex.abs (2 * w₃ - (w₁ + w₂)) ≤ Complex.abs (w₁ - w₂) ∧
               Complex.abs w₃ = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_z3_l2750_275061


namespace NUMINAMATH_CALUDE_distance_is_sqrt_5_l2750_275002

/-- A right triangle with sides of length 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  side_a : a = 6
  side_b : b = 8
  side_c : c = 10

/-- The distance between the centers of the inscribed and circumscribed circles -/
def distance_between_centers (t : RightTriangle) : ℝ := sorry

theorem distance_is_sqrt_5 (t : RightTriangle) :
  distance_between_centers t = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_distance_is_sqrt_5_l2750_275002


namespace NUMINAMATH_CALUDE_minas_numbers_l2750_275099

theorem minas_numbers (x y : ℤ) (h1 : 3 * x + 4 * y = 135) (h2 : x = 15 ∨ y = 15) : x = 25 ∨ y = 25 :=
sorry

end NUMINAMATH_CALUDE_minas_numbers_l2750_275099


namespace NUMINAMATH_CALUDE_patrick_savings_ratio_l2750_275050

theorem patrick_savings_ratio :
  ∀ (bicycle_cost initial_savings current_savings lent_amount : ℕ),
    bicycle_cost = 150 →
    lent_amount = 50 →
    current_savings = 25 →
    initial_savings = current_savings + lent_amount →
    (initial_savings : ℚ) / bicycle_cost = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_patrick_savings_ratio_l2750_275050


namespace NUMINAMATH_CALUDE_tan_105_degrees_l2750_275086

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l2750_275086


namespace NUMINAMATH_CALUDE_factorize_x_squared_minus_one_l2750_275020

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by sorry

end NUMINAMATH_CALUDE_factorize_x_squared_minus_one_l2750_275020


namespace NUMINAMATH_CALUDE_sin_cos_difference_special_angle_l2750_275056

theorem sin_cos_difference_special_angle : 
  Real.sin (80 * π / 180) * Real.cos (20 * π / 180) - 
  Real.cos (80 * π / 180) * Real.sin (20 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_difference_special_angle_l2750_275056


namespace NUMINAMATH_CALUDE_mL_to_L_conversion_l2750_275038

-- Define the conversion rate
def mL_per_L : ℝ := 1000

-- Define the volume in milliliters
def volume_mL : ℝ := 27

-- Theorem to prove the conversion
theorem mL_to_L_conversion :
  volume_mL / mL_per_L = 0.027 := by
  sorry

end NUMINAMATH_CALUDE_mL_to_L_conversion_l2750_275038


namespace NUMINAMATH_CALUDE_julia_tag_tuesday_l2750_275094

/-- The number of kids Julia played tag with on Monday -/
def monday_kids : ℕ := 7

/-- The total number of kids Julia played tag with -/
def total_kids : ℕ := 20

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_kids : ℕ := total_kids - monday_kids

theorem julia_tag_tuesday : tuesday_kids = 13 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_tuesday_l2750_275094


namespace NUMINAMATH_CALUDE_ninth_term_of_sequence_l2750_275045

theorem ninth_term_of_sequence (n : ℕ) : 
  let a : ℕ → ℝ := fun n => Real.sqrt (3 * (2 * n - 1))
  a 14 = 9 := by
sorry

end NUMINAMATH_CALUDE_ninth_term_of_sequence_l2750_275045


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_l2750_275090

theorem geometric_arithmetic_progression (b : ℝ) (q : ℝ) :
  b > 0 ∧ q > 1 →
  (∃ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (b * q ^ i.val + b * q ^ k.val) / 2 = b * q ^ j.val) →
  q = (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_l2750_275090


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2750_275016

theorem trigonometric_identities (α : Real) (h : Real.tan (α / 2) = 3) :
  (Real.tan (α + Real.pi / 3) = (48 - 4 * Real.sqrt 3) / 11) ∧
  ((Real.sin α + 2 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = -5 / 17) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2750_275016


namespace NUMINAMATH_CALUDE_garden_area_calculation_l2750_275033

/-- The area of a rectangular garden plot -/
def garden_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of a rectangular garden plot with length 1.2 meters and width 0.5 meters is 0.6 square meters -/
theorem garden_area_calculation :
  garden_area 1.2 0.5 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_calculation_l2750_275033


namespace NUMINAMATH_CALUDE_oplus_2_4_1_3_l2750_275046

-- Define the # operation
def hash (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- Define the ⊕ operation
def oplus (a b c d : ℝ) : ℝ := hash a (b + d) c - hash a b c

-- Theorem statement
theorem oplus_2_4_1_3 : oplus 2 4 1 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_oplus_2_4_1_3_l2750_275046


namespace NUMINAMATH_CALUDE_cube_sum_rational_l2750_275001

theorem cube_sum_rational (a b c : ℚ) 
  (h1 : a - b + c = 3) 
  (h2 : a^2 + b^2 + c^2 = 3) : 
  a^3 + b^3 + c^3 = 1 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_rational_l2750_275001


namespace NUMINAMATH_CALUDE_average_difference_l2750_275049

theorem average_difference (x : ℝ) : (10 + 60 + x) / 3 = (20 + 40 + 60) / 3 - 5 ↔ x = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l2750_275049


namespace NUMINAMATH_CALUDE_income_calculation_l2750_275084

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 7 = expenditure * 8 →
  income = expenditure + savings →
  savings = 5000 →
  income = 40000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l2750_275084


namespace NUMINAMATH_CALUDE_journey_speed_fraction_l2750_275011

/-- Proves that if a person travels part of a journey at 5 mph and the rest at 15 mph,
    with an average speed of 10 mph for the entire journey,
    then the fraction of time spent traveling at 15 mph is 1/2. -/
theorem journey_speed_fraction (t₅ t₁₅ : ℝ) (h₁ : t₅ > 0) (h₂ : t₁₅ > 0) :
  (5 * t₅ + 15 * t₁₅) / (t₅ + t₁₅) = 10 →
  t₁₅ / (t₅ + t₁₅) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_journey_speed_fraction_l2750_275011


namespace NUMINAMATH_CALUDE_min_a_for_p_half_ge_p_23_value_l2750_275029

def p (a : ℕ) : ℚ :=
  (Nat.choose (41 - a) 2 + Nat.choose (a - 1) 2) / Nat.choose 50 2

theorem min_a_for_p_half_ge :
  ∀ a : ℕ, 1 ≤ a → a ≤ 40 → (∀ b : ℕ, 1 ≤ b → b < a → p b < 1/2) → p a ≥ 1/2 → a = 23 :=
sorry

theorem p_23_value : p 23 = 34/49 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_p_half_ge_p_23_value_l2750_275029


namespace NUMINAMATH_CALUDE_product_of_squares_l2750_275060

theorem product_of_squares (a b : ℝ) (h1 : a + b = 21) (h2 : a^2 - b^2 = 45) :
  a^2 * b^2 = 28606956 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_product_of_squares_l2750_275060


namespace NUMINAMATH_CALUDE_davids_initial_money_l2750_275077

/-- 
Given that David has $800 less than he spent after spending money on a trip,
and he now has $500 left, prove that he had $1800 at the beginning of his trip.
-/
theorem davids_initial_money :
  ∀ (initial_money spent_money remaining_money : ℕ),
  remaining_money = spent_money - 800 →
  remaining_money = 500 →
  initial_money = spent_money + remaining_money →
  initial_money = 1800 :=
by
  sorry

end NUMINAMATH_CALUDE_davids_initial_money_l2750_275077


namespace NUMINAMATH_CALUDE_script_year_proof_l2750_275069

theorem script_year_proof : ∃! (year : ℕ), 
  year < 200 ∧ year^13 = 258145266804692077858261512663 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_script_year_proof_l2750_275069


namespace NUMINAMATH_CALUDE_range_of_abc_l2750_275058

theorem range_of_abc (a b c : ℝ) 
  (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) 
  (h4 : 2 < c) (h5 : c < 3) :
  ∀ x, (∃ (a' b' c' : ℝ), 
    -1 < a' ∧ a' < b' ∧ b' < 1 ∧ 
    2 < c' ∧ c' < 3 ∧ 
    x = (a' - b') * c') ↔ 
  -6 < x ∧ x < 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_abc_l2750_275058


namespace NUMINAMATH_CALUDE_wendy_cupcakes_l2750_275075

/-- Represents the number of pastries in Wendy's bake sale scenario -/
structure BakeSale where
  cupcakes : ℕ
  cookies : ℕ
  pastries_left : ℕ
  pastries_sold : ℕ

/-- The theorem stating the number of cupcakes Wendy baked -/
theorem wendy_cupcakes (b : BakeSale) 
  (h1 : b.cupcakes + b.cookies = b.pastries_left + b.pastries_sold)
  (h2 : b.cookies = 29)
  (h3 : b.pastries_left = 24)
  (h4 : b.pastries_sold = 9) :
  b.cupcakes = 4 := by
  sorry

end NUMINAMATH_CALUDE_wendy_cupcakes_l2750_275075


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_given_l2750_275022

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

def Line.passes_through (l : Line) (p : Point) : Prop :=
  p.y = l.m * p.x + l.b

def Line.parallel_to (l1 l2 : Line) : Prop :=
  l1.m = l2.m

theorem line_through_point_parallel_to_given : 
  let P : Point := ⟨1, 2⟩
  let given_line : Line := ⟨2, 3⟩
  let parallel_line : Line := ⟨2, 0⟩
  parallel_line.passes_through P ∧ parallel_line.parallel_to given_line := by
  sorry


end NUMINAMATH_CALUDE_line_through_point_parallel_to_given_l2750_275022


namespace NUMINAMATH_CALUDE_sequence_inequality_l2750_275095

theorem sequence_inequality (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, a (n + 1) ≥ a n ^ 2 + 1/5) : 
  ∀ n : ℕ, n ≥ 5 → Real.sqrt (a (n + 5)) ≥ a (n - 5) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l2750_275095


namespace NUMINAMATH_CALUDE_max_s_value_l2750_275062

/-- Definition of the lucky number t for a given s -/
def lucky_number (s : ℕ) : ℕ :=
  let x := s / 100 - 1
  let y := (s / 10) % 10
  if y ≤ 6 then
    1000 * (x + 1) + 100 * y + 30 + y + 3
  else
    1000 * (x + 1) + 100 * y + 30 + y - 7

/-- Definition of the function F for a given lucky number N -/
def F (N : ℕ) : ℚ :=
  let ab := N / 100
  let dc := N % 100
  (ab - dc) / 3

/-- Theorem stating the maximum value of s satisfying all conditions -/
theorem max_s_value : 
  ∃ (s : ℕ), s = 913 ∧ 
  (∀ (x y : ℕ), s = 100 * x + 10 * y + 103 → 
    x ≥ y ∧ 
    y ≤ 8 ∧ 
    x ≤ 8 ∧ 
    (lucky_number s) % 17 = 5 ∧ 
    (F (lucky_number s)).den = 1) ∧
  (∀ (s' : ℕ), s' > s → 
    ¬(∃ (x' y' : ℕ), s' = 100 * x' + 10 * y' + 103 ∧ 
      x' ≥ y' ∧ 
      y' ≤ 8 ∧ 
      x' ≤ 8 ∧ 
      (lucky_number s') % 17 = 5 ∧ 
      (F (lucky_number s')).den = 1)) :=
sorry

end NUMINAMATH_CALUDE_max_s_value_l2750_275062


namespace NUMINAMATH_CALUDE_notched_circle_distance_l2750_275072

theorem notched_circle_distance (r AB BC : ℝ) (h_r : r = Real.sqrt 75) 
  (h_AB : AB = 8) (h_BC : BC = 3) : ∃ (x y : ℝ), x^2 + y^2 = 65 ∧ 
  (x + AB)^2 + y^2 = r^2 ∧ x^2 + (y + BC)^2 = r^2 := by
  sorry

end NUMINAMATH_CALUDE_notched_circle_distance_l2750_275072


namespace NUMINAMATH_CALUDE_factory_production_equation_l2750_275008

/-- Given the production data of an agricultural machinery factory,
    this theorem states the equation that the average monthly growth rate satisfies. -/
theorem factory_production_equation (x : ℝ) : 
  (500000 : ℝ) = 500000 ∧ 
  (1820000 : ℝ) = 1820000 → 
  50 + 50*(1+x) + 50*(1+x)^2 = 182 :=
by sorry

end NUMINAMATH_CALUDE_factory_production_equation_l2750_275008


namespace NUMINAMATH_CALUDE_equal_connections_implies_square_l2750_275098

/-- Represents the coloring of vertices in a regular n-gon --/
structure VertexColoring (n : ℕ) where
  red : ℕ
  blue : ℕ
  sum_eq_n : red + blue = n

/-- Condition for equal number of same-colored and different-colored connections --/
def equal_connections (n : ℕ) (c : VertexColoring n) : Prop :=
  (c.red.choose 2) + (c.blue.choose 2) = c.red * c.blue

/-- Theorem stating that if equal_connections holds, then n is a perfect square --/
theorem equal_connections_implies_square (n : ℕ) (c : VertexColoring n) 
  (h : equal_connections n c) : ∃ k : ℕ, n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_equal_connections_implies_square_l2750_275098


namespace NUMINAMATH_CALUDE_last_digit_of_nine_power_l2750_275079

theorem last_digit_of_nine_power (n : ℕ) : 9^(9^8) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_nine_power_l2750_275079


namespace NUMINAMATH_CALUDE_sqrt_two_simplification_l2750_275005

theorem sqrt_two_simplification : 3 * Real.sqrt 2 - Real.sqrt 2 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_simplification_l2750_275005


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2750_275057

theorem floor_equation_solution (x : ℝ) : 
  (⌊⌊3 * x⌋ - 1/2⌋ = ⌊x + 3⌋) ↔ (5/3 ≤ x ∧ x < 7/3) := by
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2750_275057


namespace NUMINAMATH_CALUDE_exactly_two_ultra_squarish_l2750_275083

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A function that extracts the first three digits of a seven-digit base 9 number -/
def first_three_digits (n : ℕ) : ℕ := n / (9^4)

/-- A function that extracts the middle three digits of a seven-digit base 9 number -/
def middle_three_digits (n : ℕ) : ℕ := (n / 9^2) % (9^3)

/-- A function that extracts the last three digits of a seven-digit base 9 number -/
def last_three_digits (n : ℕ) : ℕ := n % (9^3)

/-- A function that checks if a number is ultra-squarish -/
def is_ultra_squarish (n : ℕ) : Prop :=
  n ≥ 9^6 ∧ n < 9^7 ∧  -- seven-digit number in base 9
  (∀ d, d ∈ (List.range 7).map (fun i => (n / (9^i)) % 9) → d ≠ 0) ∧  -- no digit is zero
  is_perfect_square n ∧
  is_perfect_square (first_three_digits n) ∧
  is_perfect_square (middle_three_digits n) ∧
  is_perfect_square (last_three_digits n)

/-- The theorem stating that there are exactly 2 ultra-squarish numbers -/
theorem exactly_two_ultra_squarish : 
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_ultra_squarish n) ∧ s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_ultra_squarish_l2750_275083


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l2750_275081

theorem right_triangle_third_side 
  (x y z : ℝ) 
  (h_right_triangle : x^2 + y^2 = z^2) 
  (h_equation : |x - 3| + Real.sqrt (2 * y - 8) = 0) : 
  z = Real.sqrt 7 ∨ z = 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l2750_275081


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l2750_275036

/-- The system of equations --/
def system (x y : ℝ) : Prop :=
  x^2 * y + x * y^2 - 2*x - 2*y + 10 = 0 ∧
  x^3 * y - x * y^3 - 2*x^2 + 2*y^2 - 30 = 0

/-- The solution to the system of equations --/
def solution : ℝ × ℝ := (-4, -1)

/-- Theorem stating that the solution satisfies the system of equations --/
theorem solution_satisfies_system :
  let (x, y) := solution
  system x y := by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l2750_275036


namespace NUMINAMATH_CALUDE_second_patient_hours_l2750_275040

/-- Represents the psychologist's pricing model and patient charges -/
structure TherapyPricing where
  firstHourCost : ℕ
  additionalHourCost : ℕ
  firstPatientHours : ℕ
  firstPatientCharge : ℕ
  secondPatientCharge : ℕ

/-- 
Given a psychologist's pricing model where:
- The first hour costs $30 more than each additional hour
- A 5-hour therapy session costs $400
- Another therapy session costs $252

This theorem proves that the second therapy session lasted 3 hours.
-/
theorem second_patient_hours (tp : TherapyPricing) 
  (h1 : tp.firstHourCost = tp.additionalHourCost + 30)
  (h2 : tp.firstPatientHours = 5)
  (h3 : tp.firstPatientCharge = 400)
  (h4 : tp.firstPatientCharge = tp.firstHourCost + (tp.firstPatientHours - 1) * tp.additionalHourCost)
  (h5 : tp.secondPatientCharge = 252) : 
  ∃ (h : ℕ), tp.secondPatientCharge = tp.firstHourCost + (h - 1) * tp.additionalHourCost ∧ h = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_patient_hours_l2750_275040


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l2750_275013

-- Define the ellipse equation
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (m - 1) + y^2 / m = 1

-- Theorem statement
theorem ellipse_focal_length (m : ℝ) (h1 : m > m - 1) 
  (h2 : ∀ x y : ℝ, ellipse_equation x y m) : 
  ∃ (a b c : ℝ), a^2 = m ∧ b^2 = m - 1 ∧ c^2 = 1 ∧ 2 * c = 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l2750_275013


namespace NUMINAMATH_CALUDE_shortest_distance_from_start_l2750_275093

-- Define the walker's movements
def north_distance : ℝ := 15
def west_distance : ℝ := 8
def south_distance : ℝ := 10
def east_distance : ℝ := 1

-- Calculate net distances
def net_north : ℝ := north_distance - south_distance
def net_west : ℝ := west_distance - east_distance

-- Theorem statement
theorem shortest_distance_from_start :
  Real.sqrt (net_north ^ 2 + net_west ^ 2) = Real.sqrt 74 := by
  sorry

#check shortest_distance_from_start

end NUMINAMATH_CALUDE_shortest_distance_from_start_l2750_275093


namespace NUMINAMATH_CALUDE_mystic_four_calculator_theorem_l2750_275027

/-- Represents the possible operations on the Mystic Four Calculator --/
inductive Operation
| replace_one
| divide_two
| subtract_three
| multiply_four

/-- Represents the state of the Mystic Four Calculator --/
structure CalculatorState where
  display : ℕ

/-- Applies an operation to the calculator state --/
def apply_operation (state : CalculatorState) (op : Operation) : CalculatorState :=
  match op with
  | Operation.replace_one => CalculatorState.mk 1
  | Operation.divide_two => 
      if state.display % 2 = 0 then CalculatorState.mk (state.display / 2)
      else state
  | Operation.subtract_three => 
      if state.display ≥ 3 then CalculatorState.mk (state.display - 3)
      else state
  | Operation.multiply_four => 
      if state.display * 4 < 10000 then CalculatorState.mk (state.display * 4)
      else state

/-- Applies a sequence of operations to the calculator state --/
def apply_sequence (initial : CalculatorState) (ops : List Operation) : CalculatorState :=
  ops.foldl apply_operation initial

theorem mystic_four_calculator_theorem :
  (¬ ∃ (ops : List Operation), (apply_sequence (CalculatorState.mk 0) ops).display = 2007) ∧
  (∃ (ops : List Operation), (apply_sequence (CalculatorState.mk 0) ops).display = 2008) :=
sorry

end NUMINAMATH_CALUDE_mystic_four_calculator_theorem_l2750_275027


namespace NUMINAMATH_CALUDE_sin_240_degrees_l2750_275039

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l2750_275039


namespace NUMINAMATH_CALUDE_square_side_length_relation_l2750_275071

theorem square_side_length_relation (s L : ℝ) (h1 : s > 0) (h2 : L > 0) : 
  (4 * L) / (4 * s) = 2.5 → (L * Real.sqrt 2) / (s * Real.sqrt 2) = 2.5 → L = 2.5 * s := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_relation_l2750_275071


namespace NUMINAMATH_CALUDE_prime_sum_squares_l2750_275023

theorem prime_sum_squares (p q m : ℕ) : 
  p.Prime → q.Prime → p ≠ q →
  p^2 - 2001*p + m = 0 →
  q^2 - 2001*q + m = 0 →
  p^2 + q^2 = 3996005 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_squares_l2750_275023


namespace NUMINAMATH_CALUDE_apple_production_theorem_l2750_275089

/-- Apple production over three years -/
def AppleProduction : Prop :=
  let first_year : ℕ := 40
  let second_year : ℕ := 8 + 2 * first_year
  let third_year : ℕ := second_year - (second_year / 4)
  let total : ℕ := first_year + second_year + third_year
  total = 194

theorem apple_production_theorem : AppleProduction := by
  sorry

end NUMINAMATH_CALUDE_apple_production_theorem_l2750_275089


namespace NUMINAMATH_CALUDE_craig_apples_l2750_275080

/-- The number of apples Craig has after receiving more from Eugene -/
def total_apples (initial : Real) (received : Real) : Real :=
  initial + received

/-- Proof that Craig will have 27.0 apples -/
theorem craig_apples : total_apples 20.0 7.0 = 27.0 := by
  sorry

end NUMINAMATH_CALUDE_craig_apples_l2750_275080


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_implies_a_less_than_8_l2750_275053

-- Define the sets A, B, C, and U
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 6}
def C (a : ℝ) : Set ℝ := {x | x > a}
def U : Set ℝ := Set.univ

-- Theorem 1: A ∪ B = {x | 1 ≤ x ∧ x ≤ 8}
theorem union_A_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x ≤ 8} := by sorry

-- Theorem 2: (∁ₐA) ∩ B = {x | 1 ≤ x ∧ x < 2}
theorem complement_A_intersect_B : (Aᶜ) ∩ B = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

-- Theorem 3: If A ∩ C ≠ ∅, then a < 8
theorem intersection_A_C_nonempty_implies_a_less_than_8 (a : ℝ) :
  (A ∩ C a).Nonempty → a < 8 := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_implies_a_less_than_8_l2750_275053


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2750_275063

/-- An isosceles triangle with side lengths 3 and 4 has a perimeter of either 10 or 11 -/
theorem isosceles_triangle_perimeter : 
  ∀ (a b c : ℝ), 
  (a = 3 ∨ a = 4) → 
  (b = 3 ∨ b = 4) → 
  (c = 3 ∨ c = 4) →
  (a = b ∨ b = c ∨ a = c) → 
  (a + b > c ∧ b + c > a ∧ a + c > b) →
  (a + b + c = 10 ∨ a + b + c = 11) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2750_275063


namespace NUMINAMATH_CALUDE_max_cars_divided_by_ten_limit_l2750_275067

/-- Represents the safety distance between cars in car lengths per 20 km/h -/
def safety_distance : ℝ := 1

/-- Represents the length of a car in meters -/
def car_length : ℝ := 4

/-- Calculates the maximum number of cars that can pass a counting device in one hour -/
noncomputable def max_cars_per_hour (m : ℝ) : ℝ :=
  (20000 * m) / (car_length * (m + 1))

/-- Theorem stating that the maximum number of cars passing in one hour divided by 10 approaches 500 -/
theorem max_cars_divided_by_ten_limit :
  ∀ ε > 0, ∃ M, ∀ m > M, |max_cars_per_hour m / 10 - 500| < ε :=
sorry

end NUMINAMATH_CALUDE_max_cars_divided_by_ten_limit_l2750_275067


namespace NUMINAMATH_CALUDE_diagonal_cut_result_l2750_275007

/-- Represents a scarf with areas of different colors -/
structure Scarf where
  white : ℚ
  gray : ℚ
  black : ℚ

/-- The original square scarf -/
def original_scarf : Scarf where
  white := 1/2
  gray := 1/3
  black := 1/6

/-- The first triangular scarf after cutting -/
def first_triangular_scarf : Scarf where
  white := 3/4
  gray := 2/9
  black := 1/36

/-- The second triangular scarf after cutting -/
def second_triangular_scarf : Scarf where
  white := 1/4
  gray := 4/9
  black := 11/36

/-- Theorem stating that cutting the original square scarf diagonally 
    results in the two specified triangular scarves -/
theorem diagonal_cut_result : 
  (original_scarf.white + original_scarf.gray + original_scarf.black = 1) →
  (first_triangular_scarf.white + first_triangular_scarf.gray + first_triangular_scarf.black = 1) ∧
  (second_triangular_scarf.white + second_triangular_scarf.gray + second_triangular_scarf.black = 1) ∧
  (first_triangular_scarf.white = 3/4) ∧
  (first_triangular_scarf.gray = 2/9) ∧
  (first_triangular_scarf.black = 1/36) ∧
  (second_triangular_scarf.white = 1/4) ∧
  (second_triangular_scarf.gray = 4/9) ∧
  (second_triangular_scarf.black = 11/36) := by
  sorry

end NUMINAMATH_CALUDE_diagonal_cut_result_l2750_275007


namespace NUMINAMATH_CALUDE_edward_rides_l2750_275042

def max_rides (initial_tickets : ℕ) (spent_tickets : ℕ) (tickets_per_ride : ℕ) : ℕ :=
  (initial_tickets - spent_tickets) / tickets_per_ride

theorem edward_rides : max_rides 325 115 13 = 16 := by
  sorry

end NUMINAMATH_CALUDE_edward_rides_l2750_275042


namespace NUMINAMATH_CALUDE_dinner_bill_contribution_l2750_275043

theorem dinner_bill_contribution (num_friends : ℕ) 
  (num_18_meals num_24_meals num_30_meals : ℕ)
  (cost_18_meal cost_24_meal cost_30_meal : ℚ)
  (num_appetizers : ℕ) (cost_appetizer : ℚ)
  (tip_percentage : ℚ)
  (h1 : num_friends = 8)
  (h2 : num_18_meals = 4)
  (h3 : num_24_meals = 2)
  (h4 : num_30_meals = 2)
  (h5 : cost_18_meal = 18)
  (h6 : cost_24_meal = 24)
  (h7 : cost_30_meal = 30)
  (h8 : num_appetizers = 3)
  (h9 : cost_appetizer = 12)
  (h10 : tip_percentage = 12 / 100) :
  let total_cost := num_18_meals * cost_18_meal + 
                    num_24_meals * cost_24_meal + 
                    num_30_meals * cost_30_meal + 
                    num_appetizers * cost_appetizer
  let total_with_tip := total_cost + total_cost * tip_percentage
  let contribution_per_person := total_with_tip / num_friends
  contribution_per_person = 30.24 := by
sorry

end NUMINAMATH_CALUDE_dinner_bill_contribution_l2750_275043


namespace NUMINAMATH_CALUDE_polynomial_difference_independent_of_x_l2750_275037

theorem polynomial_difference_independent_of_x (m n : ℝ) : 
  (∀ x y : ℝ, ∃ k : ℝ, (x^2 + m*x - 2*y + n) - (n*x^2 - 3*x + 4*y - 7) = k) →
  n - m = 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_difference_independent_of_x_l2750_275037


namespace NUMINAMATH_CALUDE_roots_of_quadratic_sum_l2750_275003

theorem roots_of_quadratic_sum (α β : ℝ) : 
  (α^2 - 3*α - 4 = 0) → (β^2 - 3*β - 4 = 0) → 4*α^3 + 9*β^2 = -72 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_sum_l2750_275003


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2750_275012

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  1/a + 2/b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2750_275012


namespace NUMINAMATH_CALUDE_percentage_relation_l2750_275025

theorem percentage_relation (T S F : ℝ) 
  (h1 : F = 0.06 * T) 
  (h2 : F = (1/3) * S) : 
  S = 0.18 * T := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l2750_275025


namespace NUMINAMATH_CALUDE_exists_solution_infinitely_many_solutions_divisibility_property_no_extended_solution_l2750_275073

-- Define the condition (*) as a predicate
def condition_star (k a b c : ℕ) : Prop :=
  a^2 + k^2 = b^2 + (k+1)^2 ∧ b^2 + (k+1)^2 = c^2 + (k+2)^2

-- Part (a): Existence of a solution
theorem exists_solution : ∃ k a b c : ℕ, condition_star k a b c :=
sorry

-- Part (b): Infinitely many solutions
theorem infinitely_many_solutions : ∀ n : ℕ, ∃ k a b c : ℕ, k > n ∧ condition_star k a b c :=
sorry

-- Part (c): Divisibility property
theorem divisibility_property : ∀ k a b c : ℕ, condition_star k a b c → 144 ∣ (a * b * c) :=
sorry

-- Part (d): Non-existence of extended solution
def extended_condition (k a b c d : ℕ) : Prop :=
  a^2 + k^2 = b^2 + (k+1)^2 ∧ b^2 + (k+1)^2 = c^2 + (k+2)^2 ∧ c^2 + (k+2)^2 = d^2 + (k+3)^2

theorem no_extended_solution : ¬∃ k a b c d : ℕ, extended_condition k a b c d :=
sorry

end NUMINAMATH_CALUDE_exists_solution_infinitely_many_solutions_divisibility_property_no_extended_solution_l2750_275073


namespace NUMINAMATH_CALUDE_book_borrowing_growth_l2750_275096

/-- The number of books borrowed in 2015 -/
def books_2015 : ℕ := 7500

/-- The number of books borrowed in 2017 -/
def books_2017 : ℕ := 10800

/-- The average annual growth rate from 2015 to 2017 -/
def growth_rate : ℝ := 0.2

/-- The expected number of books borrowed in 2018 -/
def books_2018 : ℕ := 12960

/-- Theorem stating the relationship between the given values and the calculated growth rate and expected books for 2018 -/
theorem book_borrowing_growth :
  (books_2017 : ℝ) = books_2015 * (1 + growth_rate)^2 ∧
  books_2018 = Int.floor (books_2017 * (1 + growth_rate)) :=
sorry

end NUMINAMATH_CALUDE_book_borrowing_growth_l2750_275096


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2750_275064

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem stating the relationship between a₄, a₇, and a₁₀ in a geometric sequence -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : geometric_sequence a) :
  a 4 * a 10 = 9 → a 7 = 3 ∨ a 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2750_275064


namespace NUMINAMATH_CALUDE_mountain_loop_trail_length_l2750_275051

/-- The Mountain Loop Trail Theorem -/
theorem mountain_loop_trail_length 
  (x1 x2 x3 x4 x5 : ℝ) 
  (h1 : x1 + x2 = 36)
  (h2 : x2 + x3 = 40)
  (h3 : x3 + x4 + x5 = 45)
  (h4 : x1 + x4 = 38) :
  x1 + x2 + x3 + x4 + x5 = 81 := by
  sorry

#check mountain_loop_trail_length

end NUMINAMATH_CALUDE_mountain_loop_trail_length_l2750_275051


namespace NUMINAMATH_CALUDE_tape_circle_length_l2750_275031

/-- The total length of a circle formed by overlapping tape pieces -/
def circle_length (num_pieces : ℕ) (piece_length : ℝ) (overlap : ℝ) : ℝ :=
  num_pieces * (piece_length - overlap)

/-- Theorem stating the total length of the circle-shaped colored tapes -/
theorem tape_circle_length :
  circle_length 16 10.4 3.5 = 110.4 := by
  sorry

end NUMINAMATH_CALUDE_tape_circle_length_l2750_275031


namespace NUMINAMATH_CALUDE_congruence_properties_l2750_275035

theorem congruence_properties : ∀ n : ℤ,
  (n ≡ 0 [ZMOD 2] → ∃ k : ℤ, n = 2 * k) ∧
  (n ≡ 1 [ZMOD 2] → ∃ k : ℤ, n = 2 * k + 1) ∧
  (n ≡ 2018 [ZMOD 2] → ∃ k : ℤ, n = 2 * k) :=
by sorry

end NUMINAMATH_CALUDE_congruence_properties_l2750_275035


namespace NUMINAMATH_CALUDE_rug_area_proof_l2750_275009

/-- Given three rugs covering a floor area, prove their combined area -/
theorem rug_area_proof (total_covered_area single_layer_area double_layer_area triple_layer_area : ℝ) 
  (h1 : total_covered_area = 140)
  (h2 : double_layer_area = 24)
  (h3 : triple_layer_area = 20)
  (h4 : single_layer_area = total_covered_area - double_layer_area - triple_layer_area) :
  single_layer_area + 2 * double_layer_area + 3 * triple_layer_area = 204 := by
  sorry

end NUMINAMATH_CALUDE_rug_area_proof_l2750_275009


namespace NUMINAMATH_CALUDE_f_is_odd_and_increasing_l2750_275059

-- Define the function
def f (x : ℝ) : ℝ := 3 * x

-- State the theorem
theorem f_is_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ a b, a < b → f a < f b) :=
sorry

end NUMINAMATH_CALUDE_f_is_odd_and_increasing_l2750_275059


namespace NUMINAMATH_CALUDE_least_nonprime_sum_l2750_275074

theorem least_nonprime_sum (p : Nat) (h : Nat.Prime p) : ∃ (n : Nat), 
  (∀ (q : Nat), Nat.Prime q → ¬Nat.Prime (q^2 + n)) ∧ 
  (∀ (m : Nat), m < n → ∃ (r : Nat), Nat.Prime r ∧ Nat.Prime (r^2 + m)) :=
by
  sorry

#check least_nonprime_sum

end NUMINAMATH_CALUDE_least_nonprime_sum_l2750_275074


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2750_275026

def f (b c x : ℝ) : ℝ := -x^2 + b*x + c

theorem quadratic_function_property (b c : ℝ) :
  f b c 2 + f b c 4 = 12138 →
  3*b + c = 6079 →
  f b c 3 = 6070 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2750_275026


namespace NUMINAMATH_CALUDE_kwik_e_tax_center_problem_l2750_275065

/-- The Kwik-e-Tax Center problem -/
theorem kwik_e_tax_center_problem (federal_price state_price quarterly_price : ℕ) 
  (state_count quarterly_count total_revenue : ℕ) 
  (h1 : federal_price = 50)
  (h2 : state_price = 30)
  (h3 : quarterly_price = 80)
  (h4 : state_count = 20)
  (h5 : quarterly_count = 10)
  (h6 : total_revenue = 4400) :
  ∃ (federal_count : ℕ), 
    federal_count = 60 ∧ 
    federal_price * federal_count + state_price * state_count + quarterly_price * quarterly_count = total_revenue :=
by
  sorry


end NUMINAMATH_CALUDE_kwik_e_tax_center_problem_l2750_275065


namespace NUMINAMATH_CALUDE_bedroom_renovation_time_l2750_275028

theorem bedroom_renovation_time :
  ∀ (bedroom_time : ℝ),
    bedroom_time > 0 →
    (3 * bedroom_time) +                                -- Time for 3 bedrooms
    (1.5 * bedroom_time) +                              -- Time for kitchen (50% longer than a bedroom)
    (2 * ((3 * bedroom_time) + (1.5 * bedroom_time))) = -- Time for living room (twice as everything else)
    54 →                                                -- Total renovation time
    bedroom_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_bedroom_renovation_time_l2750_275028


namespace NUMINAMATH_CALUDE_shaded_square_area_ratio_l2750_275017

theorem shaded_square_area_ratio : 
  let shaded_square_side : ℝ := Real.sqrt 2
  let grid_side : ℝ := 6
  (shaded_square_side ^ 2) / (grid_side ^ 2) = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_shaded_square_area_ratio_l2750_275017


namespace NUMINAMATH_CALUDE_seventh_oblong_number_l2750_275024

/-- Definition of an oblong number -/
def oblong_number (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem: The 7th oblong number is 56 -/
theorem seventh_oblong_number : oblong_number 7 = 56 := by
  sorry

end NUMINAMATH_CALUDE_seventh_oblong_number_l2750_275024


namespace NUMINAMATH_CALUDE_point_on_graph_and_sum_l2750_275066

/-- Given a function g such that g(3) = 10, prove that (1, 7.6) is on the graph of 5y = 4g(3x) - 2
    and the sum of its coordinates is 8.6 -/
theorem point_on_graph_and_sum (g : ℝ → ℝ) (h : g 3 = 10) :
  let f := fun x y => 5 * y = 4 * g (3 * x) - 2
  f 1 7.6 ∧ 1 + 7.6 = 8.6 := by
  sorry

end NUMINAMATH_CALUDE_point_on_graph_and_sum_l2750_275066


namespace NUMINAMATH_CALUDE_tysons_swimming_problem_l2750_275041

/-- Tyson's swimming problem -/
theorem tysons_swimming_problem 
  (lake_speed : ℝ) 
  (ocean_speed : ℝ) 
  (total_races : ℕ) 
  (total_time : ℝ) 
  (h1 : lake_speed = 3)
  (h2 : ocean_speed = 2.5)
  (h3 : total_races = 10)
  (h4 : total_time = 11)
  (h5 : total_races % 2 = 0) -- Ensures even number of races for equal distribution
  : ∃ (race_distance : ℝ), 
    race_distance = 3 ∧ 
    (total_races / 2 : ℝ) * (race_distance / lake_speed + race_distance / ocean_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_tysons_swimming_problem_l2750_275041


namespace NUMINAMATH_CALUDE_unique_number_l2750_275018

theorem unique_number : ∃! (n : ℕ), n > 0 ∧ n^2 + n = 217 ∧ 3 ∣ n ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_l2750_275018


namespace NUMINAMATH_CALUDE_sodium_chloride_formation_l2750_275085

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String
  product3 : String

-- Define the molar quantities
def moles_NaHSO3 : ℚ := 2
def moles_HCl : ℚ := 2

-- Define the reaction
def sodium_bisulfite_reaction : Reaction :=
  { reactant1 := "NaHSO3"
  , reactant2 := "HCl"
  , product1 := "NaCl"
  , product2 := "H2O"
  , product3 := "SO2" }

-- Theorem statement
theorem sodium_chloride_formation 
  (r : Reaction) 
  (h1 : r = sodium_bisulfite_reaction) 
  (h2 : moles_NaHSO3 = moles_HCl) :
  moles_NaHSO3 = 2 → 2 = (let moles_NaCl := moles_NaHSO3; moles_NaCl) :=
by
  sorry

end NUMINAMATH_CALUDE_sodium_chloride_formation_l2750_275085


namespace NUMINAMATH_CALUDE_matthew_lollipops_l2750_275000

theorem matthew_lollipops (total_lollipops : ℕ) (friends : ℕ) (h1 : total_lollipops = 500) (h2 : friends = 15) :
  total_lollipops % friends = 5 := by
  sorry

end NUMINAMATH_CALUDE_matthew_lollipops_l2750_275000


namespace NUMINAMATH_CALUDE_product_max_min_two_digit_l2750_275097

def max_two_digit : ℕ := 99
def min_two_digit : ℕ := 10

theorem product_max_min_two_digit : max_two_digit * min_two_digit = 990 := by
  sorry

end NUMINAMATH_CALUDE_product_max_min_two_digit_l2750_275097


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l2750_275070

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_lines 
  (m n : Line) (α β : Plane) :
  parallel m n →
  subset m α →
  perpendicular n β →
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l2750_275070


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2750_275004

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The main theorem -/
theorem perpendicular_line_equation (given_line : Line) (p : Point) :
  given_line.a = 2 ∧ given_line.b = -3 ∧ given_line.c = 4 ∧
  p.x = -1 ∧ p.y = 2 →
  ∃ (l : Line), l.contains p ∧ l.perpendicular given_line ∧
  l.a = 3 ∧ l.b = 2 ∧ l.c = -1 := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l2750_275004


namespace NUMINAMATH_CALUDE_multiply_58_62_l2750_275078

theorem multiply_58_62 : 58 * 62 = 3596 := by
  sorry

end NUMINAMATH_CALUDE_multiply_58_62_l2750_275078


namespace NUMINAMATH_CALUDE_perpendicular_lines_k_value_l2750_275006

/-- Theorem: Given two lines with direction vectors perpendicular to each other, 
    we can determine the value of k in the second line equation. -/
theorem perpendicular_lines_k_value (k : ℝ) 
  (line1 : ℝ × ℝ → Prop) 
  (line2 : ℝ × ℝ → Prop)
  (dir1 : ℝ × ℝ) 
  (dir2 : ℝ × ℝ) :
  (∀ x y, line1 (x, y) ↔ x + 3*y - 7 = 0) →
  (∀ x y, line2 (x, y) ↔ k*x - y - 2 = 0) →
  (dir1 = (1, -3)) →  -- Direction vector of line1
  (dir2 = (k, 1))  →  -- Direction vector of line2
  (dir1.1 * dir2.1 + dir1.2 * dir2.2 = 0) →  -- Dot product = 0
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_k_value_l2750_275006


namespace NUMINAMATH_CALUDE_orange_crayon_boxes_l2750_275015

theorem orange_crayon_boxes (total_crayons : ℕ) 
  (orange_per_box blue_boxes blue_per_box red_boxes red_per_box : ℕ) : 
  total_crayons = 94 →
  orange_per_box = 8 →
  blue_boxes = 7 →
  blue_per_box = 5 →
  red_boxes = 1 →
  red_per_box = 11 →
  (total_crayons - (blue_boxes * blue_per_box + red_boxes * red_per_box)) / orange_per_box = 6 :=
by
  sorry

#check orange_crayon_boxes

end NUMINAMATH_CALUDE_orange_crayon_boxes_l2750_275015


namespace NUMINAMATH_CALUDE_total_height_is_148_inches_l2750_275032

-- Define the heights of sculptures in feet and inches
def sculpture1_feet : ℕ := 2
def sculpture1_inches : ℕ := 10
def sculpture2_feet : ℕ := 3
def sculpture2_inches : ℕ := 5
def sculpture3_feet : ℕ := 4
def sculpture3_inches : ℕ := 7

-- Define the heights of bases in inches
def base1_inches : ℕ := 4
def base2_inches : ℕ := 6
def base3_inches : ℕ := 8

-- Define the number of inches in a foot
def inches_per_foot : ℕ := 12

-- Function to convert feet and inches to total inches
def to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * inches_per_foot + inches

-- Theorem statement
theorem total_height_is_148_inches :
  to_inches sculpture1_feet sculpture1_inches + base1_inches +
  to_inches sculpture2_feet sculpture2_inches + base2_inches +
  to_inches sculpture3_feet sculpture3_inches + base3_inches = 148 := by
  sorry


end NUMINAMATH_CALUDE_total_height_is_148_inches_l2750_275032
