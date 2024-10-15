import Mathlib

namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l3780_378060

/-- Systematic sampling function that returns true if the number is in the sample -/
def in_systematic_sample (total : ℕ) (sample_size : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (total / sample_size) + 1

/-- Theorem stating that in a systematic sample of size 5 from 60 numbered parts,
    if 4, 16, 40, and 52 are in the sample, then 28 must also be in the sample -/
theorem systematic_sample_theorem :
  let total := 60
  let sample_size := 5
  (in_systematic_sample total sample_size 4) →
  (in_systematic_sample total sample_size 16) →
  (in_systematic_sample total sample_size 40) →
  (in_systematic_sample total sample_size 52) →
  (in_systematic_sample total sample_size 28) :=
by
  sorry

#check systematic_sample_theorem

end NUMINAMATH_CALUDE_systematic_sample_theorem_l3780_378060


namespace NUMINAMATH_CALUDE_inscribed_circumscribed_inequality_l3780_378091

/-- A polygon inscribed in one circle and circumscribed around another -/
structure InscribedCircumscribedPolygon where
  /-- Area of the inscribing circle -/
  A : ℝ
  /-- Area of the polygon -/
  B : ℝ
  /-- Area of the circumscribed circle -/
  C : ℝ
  /-- The inscribing circle has positive area -/
  hA : 0 < A
  /-- The polygon has positive area -/
  hB : 0 < B
  /-- The circumscribed circle has positive area -/
  hC : 0 < C
  /-- The polygon's area is less than or equal to the inscribing circle's area -/
  hAB : B ≤ A
  /-- The circumscribed circle's area is less than or equal to the polygon's area -/
  hBC : C ≤ B

/-- The inequality holds for any inscribed-circumscribed polygon configuration -/
theorem inscribed_circumscribed_inequality (p : InscribedCircumscribedPolygon) : 2 * p.B ≤ p.A + p.C := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circumscribed_inequality_l3780_378091


namespace NUMINAMATH_CALUDE_total_puff_pastries_made_l3780_378021

/-- Theorem: Calculating total puff pastries made by volunteers -/
theorem total_puff_pastries_made
  (num_volunteers : ℕ)
  (trays_per_batch : ℕ)
  (pastries_per_tray : ℕ)
  (h1 : num_volunteers = 1000)
  (h2 : trays_per_batch = 8)
  (h3 : pastries_per_tray = 25) :
  num_volunteers * trays_per_batch * pastries_per_tray = 200000 :=
by sorry

end NUMINAMATH_CALUDE_total_puff_pastries_made_l3780_378021


namespace NUMINAMATH_CALUDE_multiply_72516_by_9999_l3780_378095

theorem multiply_72516_by_9999 : 72516 * 9999 = 724787484 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72516_by_9999_l3780_378095


namespace NUMINAMATH_CALUDE_exactly_two_even_dice_probability_l3780_378098

def numDice : ℕ := 5
def numFaces : ℕ := 12

def probEven : ℚ := 1 / 2

def probExactlyTwoEven : ℚ := (numDice.choose 2 : ℚ) * probEven ^ 2 * (1 - probEven) ^ (numDice - 2)

theorem exactly_two_even_dice_probability :
  probExactlyTwoEven = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_even_dice_probability_l3780_378098


namespace NUMINAMATH_CALUDE_D_72_l3780_378039

/-- D(n) is the number of ways to write n as a product of integers greater than 1, 
    considering the order of factors. -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem: D(72) = 97 -/
theorem D_72 : D 72 = 97 := by sorry

end NUMINAMATH_CALUDE_D_72_l3780_378039


namespace NUMINAMATH_CALUDE_class_7_highest_prob_l3780_378013

/-- The number of classes -/
def num_classes : ℕ := 12

/-- The probability of getting a sum of n when throwing two dice -/
def prob_sum (n : ℕ) : ℚ :=
  match n with
  | 2 => 1 / 36
  | 3 => 1 / 18
  | 4 => 1 / 12
  | 5 => 1 / 9
  | 6 => 5 / 36
  | 7 => 1 / 6
  | 8 => 5 / 36
  | 9 => 1 / 9
  | 10 => 1 / 12
  | 11 => 1 / 18
  | 12 => 1 / 36
  | _ => 0

/-- Theorem: Class 7 has the highest probability of being selected -/
theorem class_7_highest_prob :
  ∀ n : ℕ, 2 ≤ n → n ≤ num_classes → prob_sum n ≤ prob_sum 7 :=
by sorry

end NUMINAMATH_CALUDE_class_7_highest_prob_l3780_378013


namespace NUMINAMATH_CALUDE_line_y_intercept_l3780_378043

/-- A line with slope -3 and x-intercept (4,0) has y-intercept (0,12) -/
theorem line_y_intercept (f : ℝ → ℝ) (h1 : ∀ x y, f y - f x = -3 * (y - x)) 
  (h2 : f 4 = 0) : f 0 = 12 := by
  sorry

end NUMINAMATH_CALUDE_line_y_intercept_l3780_378043


namespace NUMINAMATH_CALUDE_tan_beta_value_l3780_378008

theorem tan_beta_value (α β : ℝ) 
  (h1 : Real.tan α = 1 / 2)
  (h2 : Real.tan (α - β / 2) = 1 / 3) :
  Real.tan β = 7 / 24 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_value_l3780_378008


namespace NUMINAMATH_CALUDE_car_cost_sharing_l3780_378088

theorem car_cost_sharing (total_cost : ℕ) (initial_friends : ℕ) (car_wash_earnings : ℕ) (final_friends : ℕ) : 
  total_cost = 1700 →
  initial_friends = 6 →
  car_wash_earnings = 500 →
  final_friends = 5 →
  (total_cost - car_wash_earnings) / final_friends - (total_cost - car_wash_earnings) / initial_friends = 40 := by
  sorry

end NUMINAMATH_CALUDE_car_cost_sharing_l3780_378088


namespace NUMINAMATH_CALUDE_divisibility_condition_l3780_378002

/-- Converts a base-9 number of the form 2d6d4₉ to base 10 --/
def base9_to_base10 (d : ℕ) : ℕ :=
  2 * 9^4 + d * 9^3 + 6 * 9^2 + d * 9 + 4

/-- Checks if a natural number is divisible by 13 --/
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

/-- States that 2d6d4₉ is divisible by 13 if and only if d = 4 --/
theorem divisibility_condition (d : ℕ) (h : d ≤ 8) :
  is_divisible_by_13 (base9_to_base10 d) ↔ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3780_378002


namespace NUMINAMATH_CALUDE_function_difference_implies_m_value_l3780_378027

theorem function_difference_implies_m_value :
  ∀ (f g : ℝ → ℝ) (m : ℝ),
    (∀ x, f x = 4 * x^2 - 3 * x + 5) →
    (∀ x, g x = x^2 - m * x - 8) →
    f 5 - g 5 = 20 →
    m = -13.6 := by
  sorry

end NUMINAMATH_CALUDE_function_difference_implies_m_value_l3780_378027


namespace NUMINAMATH_CALUDE_garden_perimeter_l3780_378051

/-- The perimeter of a rectangular garden with width 12 meters and an area equal to that of a 16m × 12m playground is 56 meters. -/
theorem garden_perimeter : 
  let playground_length : ℝ := 16
  let playground_width : ℝ := 12
  let garden_width : ℝ := 12
  let playground_area : ℝ := playground_length * playground_width
  let garden_length : ℝ := playground_area / garden_width
  let garden_perimeter : ℝ := 2 * (garden_length + garden_width)
  garden_perimeter = 56 := by sorry

end NUMINAMATH_CALUDE_garden_perimeter_l3780_378051


namespace NUMINAMATH_CALUDE_field_length_is_32_l3780_378089

/-- Proves that a rectangular field with specific properties has a length of 32 meters -/
theorem field_length_is_32 (l w : ℝ) (h1 : l = 2 * w) (h2 : (8 * 8 : ℝ) = (1 / 8) * (l * w)) : l = 32 :=
by sorry

end NUMINAMATH_CALUDE_field_length_is_32_l3780_378089


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l3780_378085

theorem logarithm_expression_equality : 
  2 * Real.log 10 / Real.log 5 + Real.log (1/4) / Real.log 5 + 2^(Real.log 3 / Real.log 4) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l3780_378085


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l3780_378014

theorem opposite_of_negative_two (a : ℝ) : a = -(- 2) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l3780_378014


namespace NUMINAMATH_CALUDE_exam_score_calculation_l3780_378093

theorem exam_score_calculation (total_questions : ℕ) (total_marks : ℤ) (correct_answers : ℕ) 
  (h1 : total_questions = 60)
  (h2 : total_marks = 130)
  (h3 : correct_answers = 38)
  (h4 : total_questions = correct_answers + (total_questions - correct_answers)) :
  ∃ (marks_per_correct : ℕ), 
    marks_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧ 
    marks_per_correct = 4 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l3780_378093


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3780_378077

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 1 = 2 ∧
  a 2 + a 3 = 13

/-- The sum of the 4th, 5th, and 6th terms equals 42 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h : ArithmeticSequence a) : a 4 + a 5 + a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3780_378077


namespace NUMINAMATH_CALUDE_sine_function_value_l3780_378073

/-- Given a function f(x) = sin(ωx + π/3) where ω > 0,
    if the distance between adjacent maximum and minimum points is 2√2,
    then f(1) = √3/2 -/
theorem sine_function_value (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x + π / 3)
  (∃ A B : ℝ × ℝ, 
    (A.2 = f A.1 ∧ B.2 = f B.1) ∧ 
    (∀ x ∈ Set.Icc A.1 B.1, f x ≤ A.2 ∧ f x ≥ B.2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2) →
  f 1 = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sine_function_value_l3780_378073


namespace NUMINAMATH_CALUDE_f_2015_value_l3780_378030

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) + f x = 0) ∧
  (∀ x, f (x + 1) = f (1 - x)) ∧
  (f 1 = 5)

theorem f_2015_value (f : ℝ → ℝ) (h : f_properties f) : f 2015 = -5 := by
  sorry

end NUMINAMATH_CALUDE_f_2015_value_l3780_378030


namespace NUMINAMATH_CALUDE_min_value_xyz_min_value_exact_l3780_378045

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) : 
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 1/a + 1/b + 1/c = 9 → x^3 * y^2 * z ≤ a^3 * b^2 * c :=
by sorry

theorem min_value_exact (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) : 
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/a + 1/b + 1/c = 9 ∧ a^3 * b^2 * c = 1/46656 :=
by sorry

end NUMINAMATH_CALUDE_min_value_xyz_min_value_exact_l3780_378045


namespace NUMINAMATH_CALUDE_first_company_visit_charge_is_55_l3780_378064

/-- The visit charge of the first plumbing company -/
def first_company_visit_charge : ℝ := sorry

/-- The hourly rate of the first plumbing company -/
def first_company_hourly_rate : ℝ := 35

/-- The visit charge of Reliable Plumbing -/
def reliable_plumbing_visit_charge : ℝ := 75

/-- The hourly rate of Reliable Plumbing -/
def reliable_plumbing_hourly_rate : ℝ := 30

/-- The number of labor hours -/
def labor_hours : ℝ := 4

theorem first_company_visit_charge_is_55 :
  first_company_visit_charge = 55 :=
by
  have h1 : first_company_visit_charge + labor_hours * first_company_hourly_rate =
            reliable_plumbing_visit_charge + labor_hours * reliable_plumbing_hourly_rate :=
    sorry
  sorry

end NUMINAMATH_CALUDE_first_company_visit_charge_is_55_l3780_378064


namespace NUMINAMATH_CALUDE_division_problem_l3780_378049

theorem division_problem (divisor quotient remainder : ℕ) : 
  divisor = 10 * quotient → 
  divisor = 5 * remainder → 
  remainder = 46 → 
  divisor * quotient + remainder = 5336 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3780_378049


namespace NUMINAMATH_CALUDE_interior_angles_sum_not_270_l3780_378018

theorem interior_angles_sum_not_270 (n : ℕ) (h : 3 ≤ n ∧ n ≤ 5) :
  (n - 2) * 180 ≠ 270 := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_not_270_l3780_378018


namespace NUMINAMATH_CALUDE_range_of_a_inequality_proof_l3780_378054

-- Define the function f
def f (x : ℝ) : ℝ := 3 * |x - 1| + |3 * x + 7|

-- Part 1
theorem range_of_a (a : ℝ) : 
  (∀ x, f x ≥ a^2 - 3*a) → -2 ≤ a ∧ a ≤ 5 := by sorry

-- Part 2
theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  ∀ x, Real.sqrt (a + 1) + Real.sqrt (b + 1) ≤ Real.sqrt (f x) := by sorry

end NUMINAMATH_CALUDE_range_of_a_inequality_proof_l3780_378054


namespace NUMINAMATH_CALUDE_probability_three_blue_six_trials_l3780_378007

/-- The probability of drawing exactly k blue marbles in n trials,
    given b blue marbles and r red marbles in a bag,
    where each draw is independent and the marble is replaced after each draw. -/
def probability_k_blue (n k b r : ℕ) : ℚ :=
  (n.choose k) * ((b : ℚ) / (b + r : ℚ))^k * ((r : ℚ) / (b + r : ℚ))^(n - k)

/-- The main theorem stating the probability of drawing exactly three blue marbles
    in six trials from a bag with 8 blue marbles and 6 red marbles. -/
theorem probability_three_blue_six_trials :
  probability_k_blue 6 3 8 6 = 34560 / 117649 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_blue_six_trials_l3780_378007


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3780_378066

-- Define the ratio of quantities for models A, B, and C
def ratio_A : ℕ := 2
def ratio_B : ℕ := 3
def ratio_C : ℕ := 4

-- Define the number of units of model A in the sample
def units_A : ℕ := 16

-- Define the total sample size
def sample_size : ℕ := units_A + (ratio_B * units_A / ratio_A) + (ratio_C * units_A / ratio_A)

-- Theorem statement
theorem stratified_sample_size :
  sample_size = 72 :=
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l3780_378066


namespace NUMINAMATH_CALUDE_largest_coefficient_7th_8th_term_l3780_378092

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of the r-th term in the expansion of (x^2 + 1/x)^13 -/
def coefficient (r : ℕ) : ℕ := binomial 13 r

theorem largest_coefficient_7th_8th_term :
  ∀ r, r ≠ 6 ∧ r ≠ 7 → coefficient r ≤ coefficient 6 ∧ coefficient r ≤ coefficient 7 :=
sorry

end NUMINAMATH_CALUDE_largest_coefficient_7th_8th_term_l3780_378092


namespace NUMINAMATH_CALUDE_divisibility_by_24_l3780_378075

theorem divisibility_by_24 (n : Nat) : n ≤ 9 → (712 * 10 + n) % 24 = 0 ↔ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_24_l3780_378075


namespace NUMINAMATH_CALUDE_valid_numbers_l3780_378052

def is_valid_number (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : ℕ) (m : ℕ) (p : ℕ),
    n = m + 10^k * a + 10^(k+2) * p ∧
    0 ≤ a ∧ a < 100 ∧
    m < 10^k ∧
    n = 87 * (m + 10^k * p) ∧
    n ≥ 10^99 ∧ n < 10^100

theorem valid_numbers :
  {n : ℕ | is_valid_number n} =
    {435 * 10^97, 1305 * 10^96, 2175 * 10^96, 3045 * 10^96} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l3780_378052


namespace NUMINAMATH_CALUDE_dentist_age_fraction_l3780_378040

theorem dentist_age_fraction (F : ℚ) : 
  let current_age : ℕ := 32
  let age_8_years_ago : ℕ := current_age - 8
  let age_8_years_hence : ℕ := current_age + 8
  F * age_8_years_ago = (1 : ℚ) / 10 * age_8_years_hence →
  F = (1 : ℚ) / 6 := by
  sorry

end NUMINAMATH_CALUDE_dentist_age_fraction_l3780_378040


namespace NUMINAMATH_CALUDE_brenda_stones_count_brenda_bought_36_stones_l3780_378071

theorem brenda_stones_count : ℕ → ℕ → ℕ
  | num_bracelets, stones_per_bracelet => 
    num_bracelets * stones_per_bracelet

theorem brenda_bought_36_stones 
  (num_bracelets : ℕ) 
  (stones_per_bracelet : ℕ) 
  (h1 : num_bracelets = 3) 
  (h2 : stones_per_bracelet = 12) : 
  brenda_stones_count num_bracelets stones_per_bracelet = 36 := by
  sorry

end NUMINAMATH_CALUDE_brenda_stones_count_brenda_bought_36_stones_l3780_378071


namespace NUMINAMATH_CALUDE_morgans_list_count_l3780_378087

theorem morgans_list_count : ∃ n : ℕ, n = 871 ∧ 
  n = (Finset.range (27000 / 30 + 1) \ Finset.range (900 / 30)).card := by
  sorry

end NUMINAMATH_CALUDE_morgans_list_count_l3780_378087


namespace NUMINAMATH_CALUDE_not_divisible_by_49_l3780_378047

theorem not_divisible_by_49 (n : ℤ) : ¬ ∃ k : ℤ, n^2 + 3*n + 4 = 49*k := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_49_l3780_378047


namespace NUMINAMATH_CALUDE_sector_central_angle_l3780_378050

/-- Given a sector with radius 2 cm and area 4 cm², 
    prove that the radian measure of its central angle is 2 radians. -/
theorem sector_central_angle (r : ℝ) (S : ℝ) (α : ℝ) : 
  r = 2 → S = 4 → S = (1/2) * r^2 * α → α = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3780_378050


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3780_378025

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3780_378025


namespace NUMINAMATH_CALUDE_no_real_solution_log_equation_l3780_378080

theorem no_real_solution_log_equation :
  ¬∃ (x : ℝ), (Real.log (x + 4) + Real.log (x - 2) = Real.log (x^2 - 6*x + 8)) ∧ 
  (x + 4 > 0) ∧ (x - 2 > 0) ∧ (x^2 - 6*x + 8 > 0) :=
sorry

end NUMINAMATH_CALUDE_no_real_solution_log_equation_l3780_378080


namespace NUMINAMATH_CALUDE_rod_length_relation_l3780_378082

/-- Two homogeneous rods with equal cross-sectional areas, different densities, and different
    coefficients of expansion are welded together. The system's center of gravity remains
    unchanged despite thermal expansion. -/
theorem rod_length_relation (l₁ l₂ d₁ d₂ α₁ α₂ : ℝ) 
    (h₁ : l₁ > 0) (h₂ : l₂ > 0) (h₃ : d₁ > 0) (h₄ : d₂ > 0) (h₅ : α₁ > 0) (h₆ : α₂ > 0) :
    (l₁ / l₂)^2 = (d₂ * α₂) / (d₁ * α₁) := by
  sorry

end NUMINAMATH_CALUDE_rod_length_relation_l3780_378082


namespace NUMINAMATH_CALUDE_iron_volume_change_l3780_378037

/-- If the volume of iron reduces by 1/34 when solidifying, then the volume increases by 1/33 when melting back to its original state. -/
theorem iron_volume_change (V : ℝ) (V_block : ℝ) (h : V_block = V * (1 - 1/34)) :
  (V - V_block) / V_block = 1/33 := by
sorry

end NUMINAMATH_CALUDE_iron_volume_change_l3780_378037


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3780_378078

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z := (1 + i^3) / (2 - i)
  Complex.im z = -1/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3780_378078


namespace NUMINAMATH_CALUDE_odd_function_inverse_range_l3780_378096

/-- An odd function f defined on ℝ with specific properties -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∃ a b, 0 < a ∧ a < 1 ∧ ∀ x > 0, f x = a^x + b)

/-- Theorem stating the range of b for which f has an inverse -/
theorem odd_function_inverse_range (f : ℝ → ℝ) (h : OddFunction f) 
  (h_inv : Function.Injective f) : 
  ∃ a b, (0 < a ∧ a < 1) ∧ (∀ x > 0, f x = a^x + b) ∧ (b ≤ -1 ∨ b ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_inverse_range_l3780_378096


namespace NUMINAMATH_CALUDE_smallest_possible_b_l3780_378046

theorem smallest_possible_b (a b c : ℝ) : 
  1 < a → a < b → c = 2 → 
  (¬ (c + a > b ∧ c + b > a ∧ a + b > c)) →
  (¬ ((1/b) + (1/a) > c ∧ (1/b) + c > (1/a) ∧ (1/a) + c > (1/b))) →
  b ≥ 2 ∧ ∀ x, (x > 1 ∧ x < b → x ≥ a) → b = 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_b_l3780_378046


namespace NUMINAMATH_CALUDE_large_stores_count_l3780_378074

/-- Represents the total number of stores -/
def total_stores : ℕ := 1500

/-- Represents the sample size -/
def sample_size : ℕ := 90

/-- Represents the ratio of large stores -/
def large_ratio : ℕ := 3

/-- Represents the ratio of medium stores -/
def medium_ratio : ℕ := 5

/-- Represents the ratio of small stores -/
def small_ratio : ℕ := 7

/-- Calculates the number of large stores in the sample -/
def large_stores_in_sample : ℕ :=
  (sample_size * large_ratio) / (large_ratio + medium_ratio + small_ratio)

theorem large_stores_count :
  large_stores_in_sample = 18 := by sorry

end NUMINAMATH_CALUDE_large_stores_count_l3780_378074


namespace NUMINAMATH_CALUDE_circle_area_sum_l3780_378031

/-- The sum of the areas of an infinite sequence of circles with decreasing radii -/
theorem circle_area_sum : 
  let r : ℕ → ℝ := λ n => 2 * (1/3)^(n-1)
  let area : ℕ → ℝ := λ n => π * (r n)^2
  (∑' n, area n) = 9*π/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_sum_l3780_378031


namespace NUMINAMATH_CALUDE_square_area_is_40_l3780_378072

/-- A parabola defined by y = x^2 + 4x + 1 -/
def parabola (x : ℝ) : ℝ := x^2 + 4*x + 1

/-- The y-coordinate of the line that coincides with one side of the square -/
def line_y : ℝ := 7

/-- The theorem stating that the area of the square is 40 -/
theorem square_area_is_40 :
  ∃ (x1 x2 : ℝ),
    parabola x1 = line_y ∧
    parabola x2 = line_y ∧
    x1 ≠ x2 ∧
    (x2 - x1)^2 = 40 :=
by sorry

end NUMINAMATH_CALUDE_square_area_is_40_l3780_378072


namespace NUMINAMATH_CALUDE_inequality_solution_and_function_property_l3780_378015

def f (x : ℝ) := |x - 2|

theorem inequality_solution_and_function_property :
  (∀ x : ℝ, (|x - 2| + |x| ≤ 4) ↔ (x ∈ Set.Icc (-1) 3)) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a ≠ b → a * f b + b * f a ≥ 2 * |a - b|) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_and_function_property_l3780_378015


namespace NUMINAMATH_CALUDE_gift_distribution_ways_l3780_378042

theorem gift_distribution_ways (n : ℕ) (k : ℕ) (h1 : n = 25) (h2 : k = 4) :
  (Nat.factorial n) / (Nat.factorial (n - k)) = 303600 := by
  sorry

end NUMINAMATH_CALUDE_gift_distribution_ways_l3780_378042


namespace NUMINAMATH_CALUDE_complex_fraction_power_eight_l3780_378048

theorem complex_fraction_power_eight :
  ((2 + 2 * Complex.I) / (2 - 2 * Complex.I)) ^ 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_eight_l3780_378048


namespace NUMINAMATH_CALUDE_zeros_product_bound_l3780_378099

/-- Given a > e and f(x) = e^x - a((ln x + x)/x) has two distinct zeros, prove x₁x₂ > e^(2-x₁-x₂) -/
theorem zeros_product_bound (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > Real.exp 1)
  (hf : ∀ x : ℝ, x > 0 → Real.exp x - a * ((Real.log x + x) / x) = 0 ↔ x = x₁ ∨ x = x₂)
  (hx : x₁ ≠ x₂) :
  x₁ * x₂ > Real.exp (2 - x₁ - x₂) :=
by sorry

end NUMINAMATH_CALUDE_zeros_product_bound_l3780_378099


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_opposite_l3780_378094

-- Define the total number of balls and the number of each color
def total_balls : ℕ := 7
def red_balls : ℕ := 5
def black_balls : ℕ := 2

-- Define the number of balls drawn
def drawn_balls : ℕ := 3

-- Define the events
def exactly_one_black (outcome : Finset ℕ) : Prop :=
  outcome.card = drawn_balls ∧ (outcome.filter (λ x => x > red_balls)).card = 1

def exactly_two_black (outcome : Finset ℕ) : Prop :=
  outcome.card = drawn_balls ∧ (outcome.filter (λ x => x > red_balls)).card = 2

-- Theorem statement
theorem mutually_exclusive_not_opposite :
  (∃ outcome, exactly_one_black outcome ∧ exactly_two_black outcome = False) ∧
  (∃ outcome, ¬(exactly_one_black outcome ∨ exactly_two_black outcome)) := by
  sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_opposite_l3780_378094


namespace NUMINAMATH_CALUDE_closest_integer_to_sqrt_35_l3780_378076

theorem closest_integer_to_sqrt_35 :
  ∀ x : ℝ, x = Real.sqrt 35 → (5 < x ∧ x < 6) → ∀ n : ℤ, |x - 6| ≤ |x - n| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_sqrt_35_l3780_378076


namespace NUMINAMATH_CALUDE_min_dimension_sum_for_2310_volume_l3780_378062

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- The volume of a box given its dimensions -/
def volume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- The sum of the dimensions of a box -/
def dimensionSum (d : BoxDimensions) : ℕ := d.length + d.width + d.height

/-- Theorem stating that the minimum sum of dimensions for a box with volume 2310 is 52 -/
theorem min_dimension_sum_for_2310_volume :
  (∃ (d : BoxDimensions), volume d = 2310) →
  (∀ (d : BoxDimensions), volume d = 2310 → dimensionSum d ≥ 52) ∧
  (∃ (d : BoxDimensions), volume d = 2310 ∧ dimensionSum d = 52) :=
sorry

end NUMINAMATH_CALUDE_min_dimension_sum_for_2310_volume_l3780_378062


namespace NUMINAMATH_CALUDE_sugar_solution_replacement_l3780_378086

/-- Represents a sugar solution with a total weight and sugar percentage -/
structure SugarSolution where
  totalWeight : ℝ
  sugarPercentage : ℝ

/-- Represents the mixing of two sugar solutions -/
def mixSolutions (original : SugarSolution) (replacement : SugarSolution) (replacementFraction : ℝ) : SugarSolution :=
  { totalWeight := original.totalWeight,
    sugarPercentage := 
      (1 - replacementFraction) * original.sugarPercentage + 
      replacementFraction * replacement.sugarPercentage }

theorem sugar_solution_replacement (original : SugarSolution) (replacement : SugarSolution) :
  original.sugarPercentage = 12 →
  (mixSolutions original replacement (1/4)).sugarPercentage = 16 →
  replacement.sugarPercentage = 28 := by
  sorry

end NUMINAMATH_CALUDE_sugar_solution_replacement_l3780_378086


namespace NUMINAMATH_CALUDE_felipe_construction_time_l3780_378020

theorem felipe_construction_time :
  ∀ (felipe_time emilio_time : ℝ) (felipe_break emilio_break : ℝ),
    felipe_time + emilio_time = 7.5 * 12 →
    felipe_time = emilio_time / 2 →
    felipe_break = 6 →
    emilio_break = 2 * felipe_break →
    felipe_time + felipe_break = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_felipe_construction_time_l3780_378020


namespace NUMINAMATH_CALUDE_garden_minimum_cost_l3780_378063

/-- Represents the cost of a herb in dollars per square meter -/
structure HerbCost where
  cost : ℝ
  cost_positive : cost > 0

/-- Represents a region in the garden -/
structure Region where
  area : ℝ
  area_positive : area > 0

/-- Calculates the minimum cost for planting a garden given regions and herb costs -/
def minimum_garden_cost (regions : List Region) (herb_costs : List HerbCost) : ℝ :=
  sorry

/-- The main theorem stating the minimum cost for the given garden configuration -/
theorem garden_minimum_cost :
  let regions : List Region := [
    ⟨14, by norm_num⟩,
    ⟨35, by norm_num⟩,
    ⟨10, by norm_num⟩,
    ⟨21, by norm_num⟩,
    ⟨36, by norm_num⟩
  ]
  let herb_costs : List HerbCost := [
    ⟨1.00, by norm_num⟩,
    ⟨1.50, by norm_num⟩,
    ⟨2.00, by norm_num⟩,
    ⟨2.50, by norm_num⟩,
    ⟨3.00, by norm_num⟩
  ]
  minimum_garden_cost regions herb_costs = 195.50 := by
    sorry

end NUMINAMATH_CALUDE_garden_minimum_cost_l3780_378063


namespace NUMINAMATH_CALUDE_perfect_pair_122_14762_l3780_378056

/-- Two natural numbers form a perfect pair if their sum and product are perfect squares. -/
def isPerfectPair (a b : ℕ) : Prop :=
  ∃ (x y : ℕ), a + b = x^2 ∧ a * b = y^2

/-- Theorem stating that 122 and 14762 form a perfect pair. -/
theorem perfect_pair_122_14762 : isPerfectPair 122 14762 := by
  sorry

#check perfect_pair_122_14762

end NUMINAMATH_CALUDE_perfect_pair_122_14762_l3780_378056


namespace NUMINAMATH_CALUDE_nested_bracket_calculation_l3780_378006

-- Define the operation [a,b,c]
def bracket (a b c : ℚ) : ℚ := (a + b) / c

-- Theorem statement
theorem nested_bracket_calculation :
  bracket (bracket 100 20 60) (bracket 7 2 3) (bracket 20 10 10) = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_nested_bracket_calculation_l3780_378006


namespace NUMINAMATH_CALUDE_projection_a_on_b_l3780_378061

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (3, -4)

theorem projection_a_on_b : 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -6/5 := by sorry

end NUMINAMATH_CALUDE_projection_a_on_b_l3780_378061


namespace NUMINAMATH_CALUDE_rectangular_strip_area_l3780_378023

theorem rectangular_strip_area (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b + a * c + a * (b - a) + a * a + a * (c - a) = 43 →
  a = 1 ∧ b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_rectangular_strip_area_l3780_378023


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l3780_378019

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l3780_378019


namespace NUMINAMATH_CALUDE_folding_crease_set_l3780_378016

/-- Given a circle with center O(0,0) and radius R, and a point A(a,0) inside the circle,
    the set of all points P(x,y) that are equidistant from A and any point A' on the circumference
    of the circle satisfies the given inequality. -/
theorem folding_crease_set (R a x y : ℝ) (h1 : R > 0) (h2 : 0 < a ∧ a < R) :
  (x - a/2)^2 / (R/2)^2 + y^2 / ((R/2)^2 - (a/2)^2) ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_folding_crease_set_l3780_378016


namespace NUMINAMATH_CALUDE_inscribed_square_area_specific_inscribed_square_area_l3780_378001

/-- The area of a square inscribed in a right triangle -/
theorem inscribed_square_area (LP SN : ℝ) (h1 : LP > 0) (h2 : SN > 0) :
  let x := Real.sqrt (LP * SN)
  (x : ℝ) ^ 2 = LP * SN := by sorry

/-- The specific case where LP = 30 and SN = 70 -/
theorem specific_inscribed_square_area :
  let LP : ℝ := 30
  let SN : ℝ := 70
  let x := Real.sqrt (LP * SN)
  (x : ℝ) ^ 2 = 2100 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_specific_inscribed_square_area_l3780_378001


namespace NUMINAMATH_CALUDE_chocolate_bar_eating_ways_l3780_378059

/-- Represents a chocolate bar of size m × n -/
structure ChocolateBar (m n : ℕ) where
  size : Fin m × Fin n → Bool

/-- Represents the state of eating a chocolate bar -/
structure EatingState (m n : ℕ) where
  bar : ChocolateBar m n
  eaten : Fin m × Fin n → Bool

/-- Checks if a piece can be eaten (has no more than two shared sides with uneaten pieces) -/
def canEat (state : EatingState m n) (pos : Fin m × Fin n) : Bool :=
  sorry

/-- Counts the number of ways to eat the chocolate bar -/
def countEatingWays (m n : ℕ) : ℕ :=
  sorry

/-- The main theorem: there are 6720 ways to eat a 2 × 4 chocolate bar -/
theorem chocolate_bar_eating_ways :
  countEatingWays 2 4 = 6720 :=
sorry

end NUMINAMATH_CALUDE_chocolate_bar_eating_ways_l3780_378059


namespace NUMINAMATH_CALUDE_charles_earnings_correct_l3780_378004

/-- Calculates Charles' earnings after tax deduction based on his housesitting and dog walking activities. -/
def charles_earnings : ℝ :=
  let housesitting_rate : ℝ := 15
  let labrador_rate : ℝ := 22
  let golden_retriever_rate : ℝ := 25
  let german_shepherd_rate : ℝ := 30
  let housesitting_hours : ℝ := 10
  let labrador_hours : ℝ := 3
  let golden_retriever_hours : ℝ := 2
  let german_shepherd_hours : ℝ := 1.5
  let tax_rate : ℝ := 0.1

  let total_before_tax : ℝ := 
    housesitting_rate * housesitting_hours +
    labrador_rate * labrador_hours * 2 +
    golden_retriever_rate * golden_retriever_hours +
    german_shepherd_rate * german_shepherd_hours

  total_before_tax * (1 - tax_rate)

theorem charles_earnings_correct : charles_earnings = 339.30 := by
  sorry

end NUMINAMATH_CALUDE_charles_earnings_correct_l3780_378004


namespace NUMINAMATH_CALUDE_scientific_notation_of_small_number_l3780_378069

theorem scientific_notation_of_small_number :
  ∃ (a : ℝ) (n : ℤ), 0.00003 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = -5 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_small_number_l3780_378069


namespace NUMINAMATH_CALUDE_fibonacci_periodicity_last_digit_2020th_fibonacci_l3780_378034

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

def last_digit (n : ℕ) : ℕ := n % 10

theorem fibonacci_periodicity (n : ℕ) : last_digit (fibonacci n) = last_digit (fibonacci (n % 60)) := by sorry

theorem last_digit_2020th_fibonacci : last_digit (fibonacci 2020) = 0 := by sorry

end NUMINAMATH_CALUDE_fibonacci_periodicity_last_digit_2020th_fibonacci_l3780_378034


namespace NUMINAMATH_CALUDE_koh_nh4i_reaction_l3780_378079

/-- Represents a chemical reaction with reactants and products -/
structure ChemicalReaction where
  reactants : List String
  products : List String
  ratio : Nat

/-- Represents the state of a chemical system -/
structure ChemicalSystem where
  compounds : List String
  moles : List ℚ

/-- Calculates the moles of products formed and remaining reactants -/
def reactComplete (reaction : ChemicalReaction) (initial : ChemicalSystem) : ChemicalSystem :=
  sorry

theorem koh_nh4i_reaction 
  (reaction : ChemicalReaction)
  (initial : ChemicalSystem)
  (h_reaction : reaction = 
    { reactants := ["KOH", "NH4I"]
    , products := ["KI", "NH3", "H2O"]
    , ratio := 1 })
  (h_initial : initial = 
    { compounds := ["KOH", "NH4I"]
    , moles := [3, 3] })
  : 
  let final := reactComplete reaction initial
  (final.compounds = ["KI", "NH3", "H2O", "KOH", "NH4I"] ∧
   final.moles = [3, 3, 3, 0, 0]) :=
by sorry

end NUMINAMATH_CALUDE_koh_nh4i_reaction_l3780_378079


namespace NUMINAMATH_CALUDE_unique_solution_l3780_378012

/-- Represents a digit in the equation --/
def Digit := Fin 10

/-- The equation is valid if it satisfies all conditions --/
def is_valid_equation (A B C D : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  100 * A.val + 10 * C.val + A.val + 
  100 * B.val + 10 * B.val + D.val = 
  1000 * A.val + 100 * B.val + 10 * C.val + D.val

/-- There exists a unique solution to the equation --/
theorem unique_solution : 
  ∃! (A B C D : Digit), is_valid_equation A B C D ∧ 
    A.val = 9 ∧ B.val = 8 ∧ C.val = 0 ∧ D.val = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3780_378012


namespace NUMINAMATH_CALUDE_hexagon_segment_probability_l3780_378003

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of segments (sides and diagonals) in a regular hexagon -/
def total_segments : ℕ := num_sides + num_diagonals

/-- The number of different diagonal lengths in a regular hexagon -/
def num_diagonal_lengths : ℕ := 3

/-- The number of diagonals of each length in a regular hexagon -/
def diagonals_per_length : ℕ := num_diagonals / num_diagonal_lengths

theorem hexagon_segment_probability : 
  (num_sides * (num_sides - 1) + num_diagonals * (diagonals_per_length - 1)) / 
  (total_segments * (total_segments - 1)) = 11 / 35 := by
sorry

end NUMINAMATH_CALUDE_hexagon_segment_probability_l3780_378003


namespace NUMINAMATH_CALUDE_line_slope_relation_l3780_378028

/-- Theorem: For a straight line y = kx + b passing through points A(-3, y₁) and B(4, y₂),
    if k < 0, then y₁ > y₂. -/
theorem line_slope_relation (k b y₁ y₂ : ℝ) : 
  k < 0 → 
  y₁ = k * (-3) + b →
  y₂ = k * 4 + b →
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_line_slope_relation_l3780_378028


namespace NUMINAMATH_CALUDE_unique_prime_between_squares_l3780_378024

theorem unique_prime_between_squares : ∃! p : ℕ, 
  Prime p ∧ 
  ∃ n : ℕ, p = n^2 + 4 ∧ 
  ∃ m : ℕ, p + 7 = (n + 1)^2 ∧ 
  p = 29 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_between_squares_l3780_378024


namespace NUMINAMATH_CALUDE_percentage_to_decimal_decimal_representation_of_208_percent_l3780_378035

/-- The decimal representation of a percentage is equal to the percentage divided by 100. -/
theorem percentage_to_decimal (p : ℝ) : p / 100 = p * (1 / 100) := by sorry

/-- The decimal representation of 208% is 2.08. -/
theorem decimal_representation_of_208_percent : (208 : ℝ) / 100 = 2.08 := by sorry

end NUMINAMATH_CALUDE_percentage_to_decimal_decimal_representation_of_208_percent_l3780_378035


namespace NUMINAMATH_CALUDE_average_scissors_after_changes_l3780_378044

/-- Represents a drawer with scissors and pencils -/
structure Drawer where
  scissors : ℕ
  pencils : ℕ

/-- Calculates the average number of scissors in the drawers -/
def averageScissors (drawers : List Drawer) : ℚ :=
  (drawers.map (·.scissors)).sum / drawers.length

theorem average_scissors_after_changes : 
  let initialDrawers : List Drawer := [
    { scissors := 39, pencils := 22 },
    { scissors := 27, pencils := 54 },
    { scissors := 45, pencils := 33 }
  ]
  let scissorsAdded : List ℕ := [13, 7, 10]
  let finalDrawers := List.zipWith 
    (fun d a => { scissors := d.scissors + a, pencils := d.pencils }) 
    initialDrawers 
    scissorsAdded
  averageScissors finalDrawers = 47 := by
  sorry

end NUMINAMATH_CALUDE_average_scissors_after_changes_l3780_378044


namespace NUMINAMATH_CALUDE_boxes_needed_to_sell_l3780_378090

def total_chocolate_bars : ℕ := 710
def chocolate_bars_per_box : ℕ := 5

theorem boxes_needed_to_sell (total : ℕ) (per_box : ℕ) :
  total = total_chocolate_bars →
  per_box = chocolate_bars_per_box →
  total / per_box = 142 := by
  sorry

end NUMINAMATH_CALUDE_boxes_needed_to_sell_l3780_378090


namespace NUMINAMATH_CALUDE_imaginary_cube_plus_one_l3780_378022

theorem imaginary_cube_plus_one (i : ℂ) : i^2 = -1 → 1 + i^3 = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_cube_plus_one_l3780_378022


namespace NUMINAMATH_CALUDE_complex_sum_powers_l3780_378083

theorem complex_sum_powers (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^101 + z^102 + z^103 + z^104 + z^105 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_powers_l3780_378083


namespace NUMINAMATH_CALUDE_complex_roots_on_circle_l3780_378000

theorem complex_roots_on_circle : ∃ (r : ℝ), r = 2 / Real.sqrt 3 ∧
  ∀ (z : ℂ), (z + 2)^6 = 64 * z^6 → Complex.abs (z - Complex.ofReal (2/3)) = r :=
sorry

end NUMINAMATH_CALUDE_complex_roots_on_circle_l3780_378000


namespace NUMINAMATH_CALUDE_unique_c_value_l3780_378097

/-- The quadratic equation we're considering -/
def quadratic (b : ℝ) (c : ℝ) (x : ℝ) : Prop :=
  x^2 + (b^2 + 1/b^2) * x + c = 3

/-- The condition for the quadratic to have exactly one solution -/
def has_unique_solution (b : ℝ) (c : ℝ) : Prop :=
  ∃! x, quadratic b c x

/-- The main theorem statement -/
theorem unique_c_value : ∃! c : ℝ, c ≠ 0 ∧ 
  (∃! b : ℕ, b > 0 ∧ has_unique_solution (b : ℝ) c) :=
sorry

end NUMINAMATH_CALUDE_unique_c_value_l3780_378097


namespace NUMINAMATH_CALUDE_martian_right_angle_theorem_l3780_378036

/-- The number of clerts in a full circle in the Martian system -/
def full_circle_clerts : ℕ := 600

/-- The fraction of a full circle that represents a Martian right angle -/
def martian_right_angle_fraction : ℚ := 1/3

/-- The number of clerts in a Martian right angle -/
def martian_right_angle_clerts : ℕ := 200

/-- Theorem stating that the number of clerts in a Martian right angle is 200 -/
theorem martian_right_angle_theorem : 
  (↑full_circle_clerts : ℚ) * martian_right_angle_fraction = martian_right_angle_clerts := by
  sorry

end NUMINAMATH_CALUDE_martian_right_angle_theorem_l3780_378036


namespace NUMINAMATH_CALUDE_count_of_six_from_100_to_999_l3780_378053

/-- Count of digit 6 in a specific place (units, tens, or hundreds) for numbers from 100 to 999 -/
def count_digit_in_place (place : Nat) : Nat :=
  if place = 2 then 100 else 90

/-- Total count of digit 6 in all places for numbers from 100 to 999 -/
def total_count_of_six : Nat :=
  count_digit_in_place 0 + count_digit_in_place 1 + count_digit_in_place 2

/-- Theorem: The digit 6 appears 280 times when writing integers from 100 through 999 inclusive -/
theorem count_of_six_from_100_to_999 : total_count_of_six = 280 := by
  sorry

end NUMINAMATH_CALUDE_count_of_six_from_100_to_999_l3780_378053


namespace NUMINAMATH_CALUDE_tangent_line_points_l3780_378070

/-- The function f(x) = x³ + ax² -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x

theorem tangent_line_points (a : ℝ) :
  ∃ (x₀ : ℝ), (f_deriv a x₀ = -1 ∧ x₀ + f a x₀ = 0) →
  ((x₀ = 1 ∧ f a x₀ = -1) ∨ (x₀ = -1 ∧ f a x₀ = 1)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_points_l3780_378070


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l3780_378029

-- Define the common ratios
variable (p r : ℝ)

-- Define the first term of both sequences
variable (k : ℝ)

-- Define the geometric sequences
def a (n : ℕ) : ℝ := k * p^n
def b (n : ℕ) : ℝ := k * r^n

-- State the theorem
theorem sum_of_common_ratios_is_three 
  (h1 : p ≠ 1) 
  (h2 : r ≠ 1) 
  (h3 : p ≠ r) 
  (h4 : k ≠ 0) 
  (h5 : a 4 - b 4 = 4 * (a 2 - b 2)) : 
  p + r = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l3780_378029


namespace NUMINAMATH_CALUDE_dragon_can_be_defeated_l3780_378009

/-- Represents the possible number of heads a warrior can chop off in one strike -/
inductive Strike
  | thirtythree
  | twentyone
  | seventeen
  | one

/-- Represents the state of the dragon -/
structure DragonState where
  heads : ℕ

/-- Applies a strike to the dragon state -/
def applyStrike (s : Strike) (d : DragonState) : DragonState :=
  match s with
  | Strike.thirtythree => ⟨d.heads + 48 - 33⟩
  | Strike.twentyone => ⟨d.heads - 21⟩
  | Strike.seventeen => ⟨d.heads + 14 - 17⟩
  | Strike.one => ⟨d.heads + 349 - 1⟩

/-- Represents a sequence of strikes -/
def StrikeSequence := List Strike

/-- Applies a sequence of strikes to the dragon state -/
def applySequence (seq : StrikeSequence) (d : DragonState) : DragonState :=
  seq.foldl (fun state strike => applyStrike strike state) d

/-- The theorem stating that the dragon can be defeated -/
theorem dragon_can_be_defeated : 
  ∃ (seq : StrikeSequence), (applySequence seq ⟨2000⟩).heads = 0 := by
  sorry

end NUMINAMATH_CALUDE_dragon_can_be_defeated_l3780_378009


namespace NUMINAMATH_CALUDE_negation_of_false_l3780_378026

theorem negation_of_false (p q : Prop) : p ∧ ¬q → ¬q := by
  sorry

end NUMINAMATH_CALUDE_negation_of_false_l3780_378026


namespace NUMINAMATH_CALUDE_range_of_f_l3780_378055

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ f x = y) ↔ y ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3780_378055


namespace NUMINAMATH_CALUDE_daily_apple_harvest_l3780_378081

/-- The number of sections in the apple orchard -/
def num_sections : ℕ := 8

/-- The number of sacks of apples harvested from each section daily -/
def sacks_per_section : ℕ := 45

/-- The total number of sacks of apples harvested daily -/
def total_sacks : ℕ := num_sections * sacks_per_section

theorem daily_apple_harvest :
  total_sacks = 360 :=
by sorry

end NUMINAMATH_CALUDE_daily_apple_harvest_l3780_378081


namespace NUMINAMATH_CALUDE_plane_equation_satisfies_conditions_l3780_378065

/-- A plane in 3D space --/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- A point in 3D space --/
structure Point where
  x : ℚ
  y : ℚ
  z : ℚ

/-- Check if a point lies on a plane --/
def Point.liesOn (p : Point) (pl : Plane) : Prop :=
  pl.a * p.x + pl.b * p.y + pl.c * p.z + pl.d = 0

/-- Check if two planes are parallel --/
def Plane.isParallelTo (pl1 pl2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ pl1.a = k * pl2.a ∧ pl1.b = k * pl2.b ∧ pl1.c = k * pl2.c

theorem plane_equation_satisfies_conditions
  (given_plane : Plane)
  (given_point : Point)
  (parallel_plane : Plane)
  (h1 : given_plane.a = 3)
  (h2 : given_plane.b = 4)
  (h3 : given_plane.c = -2)
  (h4 : given_plane.d = 16)
  (h5 : given_point.x = 2)
  (h6 : given_point.y = -3)
  (h7 : given_point.z = 5)
  (h8 : parallel_plane.a = 3)
  (h9 : parallel_plane.b = 4)
  (h10 : parallel_plane.c = -2)
  (h11 : parallel_plane.d = 6)
  : given_point.liesOn given_plane ∧
    given_plane.isParallelTo parallel_plane ∧
    given_plane.a > 0 ∧
    Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs given_plane.a) (Int.natAbs given_plane.b)) (Int.natAbs given_plane.c)) (Int.natAbs given_plane.d) = 1 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_satisfies_conditions_l3780_378065


namespace NUMINAMATH_CALUDE_minimum_value_and_range_proof_l3780_378067

theorem minimum_value_and_range_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (min : ℝ), min = 9 ∧ (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' + b' = 1 → 1/a' + 4/b' ≥ min) ∧
    (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 1/a₀ + 4/b₀ = min)) ∧
  (∀ x : ℝ, (1/a + 4/b ≥ |2*x - 1| - |x + 1|) → -7 ≤ x ∧ x ≤ 11) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_and_range_proof_l3780_378067


namespace NUMINAMATH_CALUDE_interest_rate_is_twelve_percent_l3780_378010

/-- Calculate the interest rate given principal, time, and interest amount -/
def calculate_interest_rate (principal : ℕ) (time : ℕ) (interest : ℕ) : ℚ :=
  (interest * 100) / (principal * time)

/-- Theorem: The interest rate is 12% given the problem conditions -/
theorem interest_rate_is_twelve_percent (principal : ℕ) (time : ℕ) (interest : ℕ)
  (h1 : principal = 9200)
  (h2 : time = 3)
  (h3 : interest = principal - 5888) :
  calculate_interest_rate principal time interest = 12 := by
  sorry

#eval calculate_interest_rate 9200 3 (9200 - 5888)

end NUMINAMATH_CALUDE_interest_rate_is_twelve_percent_l3780_378010


namespace NUMINAMATH_CALUDE_initial_tagged_fish_count_l3780_378011

/-- The number of fish initially tagged and returned to the pond -/
def initial_tagged_fish : ℕ := 50

/-- The total number of fish in the pond -/
def total_fish : ℕ := 1250

/-- The number of fish caught in the second catch -/
def second_catch : ℕ := 50

/-- The number of tagged fish in the second catch -/
def tagged_in_second_catch : ℕ := 2

theorem initial_tagged_fish_count :
  initial_tagged_fish = 50 := by sorry

end NUMINAMATH_CALUDE_initial_tagged_fish_count_l3780_378011


namespace NUMINAMATH_CALUDE_prob_three_in_seven_thirteenths_l3780_378058

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : List ℕ := sorry

/-- The probability of a digit occurring in a decimal representation -/
def digitProbability (d : ℕ) (q : ℚ) : ℚ := sorry

/-- Theorem: The probability of selecting 3 from the decimal representation of 7/13 is 1/6 -/
theorem prob_three_in_seven_thirteenths :
  digitProbability 3 (7/13) = 1/6 := by sorry

end NUMINAMATH_CALUDE_prob_three_in_seven_thirteenths_l3780_378058


namespace NUMINAMATH_CALUDE_baseball_card_theorem_l3780_378041

/-- Represents the amount of money each person has and the cost of the baseball card. -/
structure BaseballCardProblem where
  patricia_money : ℝ
  lisa_money : ℝ
  charlotte_money : ℝ
  card_cost : ℝ

/-- Calculates the additional money required to buy the baseball card. -/
def additional_money_required (problem : BaseballCardProblem) : ℝ :=
  problem.card_cost - (problem.patricia_money + problem.lisa_money + problem.charlotte_money)

/-- Theorem stating the additional money required is $49 given the problem conditions. -/
theorem baseball_card_theorem (problem : BaseballCardProblem) 
  (h1 : problem.patricia_money = 6)
  (h2 : problem.lisa_money = 5 * problem.patricia_money)
  (h3 : problem.lisa_money = 2 * problem.charlotte_money)
  (h4 : problem.card_cost = 100) :
  additional_money_required problem = 49 := by
  sorry

#eval additional_money_required { 
  patricia_money := 6, 
  lisa_money := 30, 
  charlotte_money := 15, 
  card_cost := 100 
}

end NUMINAMATH_CALUDE_baseball_card_theorem_l3780_378041


namespace NUMINAMATH_CALUDE_probability_at_least_one_l3780_378017

theorem probability_at_least_one (A B : ℝ) (hA : A = 0.6) (hB : B = 0.7) 
  (h_independent : True) : 1 - (1 - A) * (1 - B) = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_l3780_378017


namespace NUMINAMATH_CALUDE_friends_team_assignment_l3780_378084

theorem friends_team_assignment (n : ℕ) (k : ℕ) :
  (n = 8 ∧ k = 4) →
  (number_of_assignments : ℕ) →
  number_of_assignments = k^n :=
by
  sorry

end NUMINAMATH_CALUDE_friends_team_assignment_l3780_378084


namespace NUMINAMATH_CALUDE_water_evaporation_l3780_378032

/-- Given a bowl with 10 ounces of water, if 2% of the original amount evaporates
    over 50 days, then the amount of water evaporated each day is 0.04 ounces. -/
theorem water_evaporation (initial_water : ℝ) (days : ℕ) (evaporation_rate : ℝ) :
  initial_water = 10 →
  days = 50 →
  evaporation_rate = 0.02 →
  (initial_water * evaporation_rate) / days = 0.04 :=
by sorry

end NUMINAMATH_CALUDE_water_evaporation_l3780_378032


namespace NUMINAMATH_CALUDE_product_of_three_integers_l3780_378068

theorem product_of_three_integers : (-3 : ℤ) * (-4 : ℤ) * (-1 : ℤ) = -12 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_integers_l3780_378068


namespace NUMINAMATH_CALUDE_f_decreasing_f_odd_implies_m_zero_l3780_378057

/-- The function f(x) = -2x + m -/
def f (m : ℝ) : ℝ → ℝ := fun x ↦ -2 * x + m

theorem f_decreasing (m : ℝ) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f m x₁ > f m x₂ := by sorry

theorem f_odd_implies_m_zero (m : ℝ) : 
  (∀ x : ℝ, f m (-x) = -(f m x)) → m = 0 := by sorry

end NUMINAMATH_CALUDE_f_decreasing_f_odd_implies_m_zero_l3780_378057


namespace NUMINAMATH_CALUDE_rectangle_area_error_percentage_l3780_378033

/-- Given a rectangle where one side is measured 16% in excess and the other side is measured 5% in deficit, 
    the error percentage in the calculated area is 10.2%. -/
theorem rectangle_area_error_percentage (L W : ℝ) (L_positive : L > 0) (W_positive : W > 0) : 
  let actual_area := L * W
  let measured_length := L * (1 + 16/100)
  let measured_width := W * (1 - 5/100)
  let calculated_area := measured_length * measured_width
  let error := calculated_area - actual_area
  let error_percentage := (error / actual_area) * 100
  error_percentage = 10.2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percentage_l3780_378033


namespace NUMINAMATH_CALUDE_unique_solution_l3780_378038

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  x^2 * y - x * y^2 - 3*x + 3*y + 1 = 0 ∧
  x^3 * y - x * y^3 - 3*x^2 + 3*y^2 + 3 = 0

/-- The theorem stating that (2, 1) is the only solution to the system -/
theorem unique_solution :
  ∀ x y : ℝ, system x y ↔ x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3780_378038


namespace NUMINAMATH_CALUDE_cube_face_sum_l3780_378005

theorem cube_face_sum (a b c d e f : ℕ+) : 
  (a * b * c + a * e * c + a * b * f + a * e * f + 
   d * b * c + d * e * c + d * b * f + d * e * f = 1089) → 
  (a + b + c + d + e + f = 31) := by
sorry

end NUMINAMATH_CALUDE_cube_face_sum_l3780_378005
