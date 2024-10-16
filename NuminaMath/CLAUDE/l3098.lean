import Mathlib

namespace NUMINAMATH_CALUDE_tv_cost_l3098_309800

theorem tv_cost (mixer_cost tv_cost : ℕ) : 
  (2 * mixer_cost + tv_cost = 7000) → 
  (mixer_cost + 2 * tv_cost = 9800) → 
  tv_cost = 4200 := by
sorry

end NUMINAMATH_CALUDE_tv_cost_l3098_309800


namespace NUMINAMATH_CALUDE_derivative_x_exp_cos_l3098_309873

/-- The derivative of xe^(cos x) is -x sin x * e^(cos x) + e^(cos x) -/
theorem derivative_x_exp_cos (x : ℝ) :
  deriv (fun x => x * Real.exp (Real.cos x)) x =
  -x * Real.sin x * Real.exp (Real.cos x) + Real.exp (Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_exp_cos_l3098_309873


namespace NUMINAMATH_CALUDE_smallest_non_prime_f_l3098_309836

/-- The function f(x) = x^2 + x + 41 -/
def f (x : ℕ) : ℕ := x^2 + x + 41

/-- Predicate to check if a natural number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- Theorem: 40 is the smallest positive integer x for which f(x) is not prime -/
theorem smallest_non_prime_f :
  (∀ x : ℕ, 0 < x → x < 40 → is_prime (f x)) ∧
  ¬ is_prime (f 40) := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_prime_f_l3098_309836


namespace NUMINAMATH_CALUDE_population_exceeds_target_in_2125_l3098_309865

-- Define the initial year and population
def initialYear : ℕ := 1950
def initialPopulation : ℕ := 750

-- Define the doubling period
def doublingPeriod : ℕ := 35

-- Define the target population
def targetPopulation : ℕ := 15000

-- Function to calculate population after n doubling periods
def populationAfterPeriods (n : ℕ) : ℕ :=
  initialPopulation * 2^n

-- Function to calculate the year after n doubling periods
def yearAfterPeriods (n : ℕ) : ℕ :=
  initialYear + n * doublingPeriod

-- Theorem to prove
theorem population_exceeds_target_in_2125 :
  ∃ n : ℕ, yearAfterPeriods n = 2125 ∧ populationAfterPeriods n > targetPopulation ∧
  ∀ m : ℕ, m < n → populationAfterPeriods m ≤ targetPopulation :=
sorry

end NUMINAMATH_CALUDE_population_exceeds_target_in_2125_l3098_309865


namespace NUMINAMATH_CALUDE_sample_size_is_twenty_l3098_309813

/-- Represents the number of brands for each dairy product type -/
structure DairyBrands where
  pureMilk : ℕ
  yogurt : ℕ
  infantFormula : ℕ
  adultFormula : ℕ

/-- Represents the sample sizes for each dairy product type -/
structure SampleSizes where
  pureMilk : ℕ
  yogurt : ℕ
  infantFormula : ℕ
  adultFormula : ℕ

/-- Calculates the total sample size given the sample sizes for each product type -/
def totalSampleSize (s : SampleSizes) : ℕ :=
  s.pureMilk + s.yogurt + s.infantFormula + s.adultFormula

/-- Theorem stating that the total sample size is 20 given the problem conditions -/
theorem sample_size_is_twenty (brands : DairyBrands)
    (h1 : brands.pureMilk = 30)
    (h2 : brands.yogurt = 10)
    (h3 : brands.infantFormula = 35)
    (h4 : brands.adultFormula = 25)
    (sample : SampleSizes)
    (h5 : sample.infantFormula = 7)
    (h6 : sample.pureMilk * brands.infantFormula = brands.pureMilk * sample.infantFormula)
    (h7 : sample.yogurt * brands.infantFormula = brands.yogurt * sample.infantFormula)
    (h8 : sample.adultFormula * brands.infantFormula = brands.adultFormula * sample.infantFormula) :
  totalSampleSize sample = 20 := by
  sorry


end NUMINAMATH_CALUDE_sample_size_is_twenty_l3098_309813


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l3098_309885

theorem y_intercept_of_line (x y : ℝ) :
  2 * x - 3 * y = 6 → x = 0 → y = -2 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l3098_309885


namespace NUMINAMATH_CALUDE_brad_probability_l3098_309887

/-- Represents the outcome of answering a math problem -/
inductive Answer
| correct
| incorrect

/-- Represents a sequence of answers to math problems -/
def AnswerSequence := List Answer

/-- Calculates the probability of a specific answer sequence -/
def probability (seq : AnswerSequence) : Real :=
  sorry

/-- Counts the number of correct answers in a sequence -/
def countCorrect (seq : AnswerSequence) : Nat :=
  sorry

/-- Generates all possible answer sequences for the remaining 8 problems -/
def generateSequences : List AnswerSequence :=
  sorry

theorem brad_probability :
  let allSequences := generateSequences
  let validSequences := allSequences.filter (λ seq => countCorrect (Answer.correct :: Answer.incorrect :: seq) = 5)
  (validSequences.map probability).sum = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_brad_probability_l3098_309887


namespace NUMINAMATH_CALUDE_tan_sum_angle_l3098_309831

theorem tan_sum_angle (α β : Real) : 
  α ∈ Set.Ioo 0 (Real.pi / 2) →
  β ∈ Set.Ioo 0 (Real.pi / 2) →
  2 * Real.tan α = Real.sin (2 * β) / (Real.sin β + Real.sin β ^ 2) →
  Real.tan (2 * α + β + Real.pi / 3) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_angle_l3098_309831


namespace NUMINAMATH_CALUDE_trampoline_jumps_l3098_309896

theorem trampoline_jumps (ronald_jumps rupert_jumps : ℕ) : 
  ronald_jumps = 157 →
  rupert_jumps > ronald_jumps →
  rupert_jumps + ronald_jumps = 400 →
  rupert_jumps - ronald_jumps = 86 := by
sorry

end NUMINAMATH_CALUDE_trampoline_jumps_l3098_309896


namespace NUMINAMATH_CALUDE_jonessas_take_home_pay_l3098_309859

/-- Calculates the take-home pay given the total pay and tax rate -/
def takeHomePay (totalPay : ℝ) (taxRate : ℝ) : ℝ :=
  totalPay * (1 - taxRate)

/-- Proves that Jonessa's take-home pay is $450 -/
theorem jonessas_take_home_pay :
  let totalPay : ℝ := 500
  let taxRate : ℝ := 0.1
  takeHomePay totalPay taxRate = 450 := by
sorry

end NUMINAMATH_CALUDE_jonessas_take_home_pay_l3098_309859


namespace NUMINAMATH_CALUDE_quadratic_y_values_order_l3098_309834

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧
  (∀ x, f x = a * x^2 + b * x + c) ∧
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  f (-2) = 1 ∧ (∀ x, f x ≤ f (-2))

/-- Theorem stating the relationship between y-values of specific points -/
theorem quadratic_y_values_order (f : ℝ → ℝ) (y₁ y₂ y₃ : ℝ)
  (hf : QuadraticFunction f)
  (h1 : f 1 = y₁)
  (h2 : f (-1) = y₂)
  (h3 : f (-4) = y₃) :
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_y_values_order_l3098_309834


namespace NUMINAMATH_CALUDE_skittles_division_l3098_309867

theorem skittles_division (total_skittles : ℕ) (num_students : ℕ) (skittles_per_student : ℕ) :
  total_skittles = 27 →
  num_students = 9 →
  total_skittles = num_students * skittles_per_student →
  skittles_per_student = 3 := by
  sorry

end NUMINAMATH_CALUDE_skittles_division_l3098_309867


namespace NUMINAMATH_CALUDE_min_box_height_l3098_309874

def box_height (x : ℝ) : ℝ := 2 * x + 2

def box_surface_area (x : ℝ) : ℝ := 9 * x^2 + 8 * x

theorem min_box_height :
  ∃ (x : ℝ), 
    x > 0 ∧
    box_surface_area x ≥ 110 ∧
    (∀ y : ℝ, y > 0 → box_surface_area y ≥ 110 → x ≤ y) ∧
    box_height x = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_box_height_l3098_309874


namespace NUMINAMATH_CALUDE_marble_draw_probability_l3098_309882

/-- The probability of drawing one red marble followed by one blue marble without replacement -/
theorem marble_draw_probability (red blue yellow : ℕ) (h_red : red = 4) (h_blue : blue = 3) (h_yellow : yellow = 6) :
  (red : ℚ) / (red + blue + yellow) * blue / (red + blue + yellow - 1) = 1 / 13 := by sorry

end NUMINAMATH_CALUDE_marble_draw_probability_l3098_309882


namespace NUMINAMATH_CALUDE_assistant_productivity_increase_l3098_309815

theorem assistant_productivity_increase 
  (base_output : ℝ) 
  (base_hours : ℝ) 
  (output_increase_factor : ℝ) 
  (hours_decrease_factor : ℝ) 
  (h1 : output_increase_factor = 1.8) 
  (h2 : hours_decrease_factor = 0.9) :
  (output_increase_factor * base_output) / (hours_decrease_factor * base_hours) / 
  (base_output / base_hours) - 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_assistant_productivity_increase_l3098_309815


namespace NUMINAMATH_CALUDE_scent_ratio_equals_candle_ratio_l3098_309810

/-- Represents the number of candles made for each scent -/
structure CandleCounts where
  coconut : ℕ
  lavender : ℕ
  almond : ℕ

/-- Represents the amount of scent used for each type -/
structure ScentAmounts where
  coconut : ℝ
  lavender : ℝ
  almond : ℝ

/-- The conditions of the candle-making scenario -/
class CandleScenario (counts : CandleCounts) (amounts : ScentAmounts) where
  same_scent_amount : amounts.coconut = amounts.lavender ∧ amounts.lavender = amounts.almond
  twice_lavender : counts.lavender = 2 * counts.coconut
  ten_almond : counts.almond = 10

/-- The theorem stating that the ratio of scent amounts equals the ratio of candle counts -/
theorem scent_ratio_equals_candle_ratio 
  (counts : CandleCounts) 
  (amounts : ScentAmounts) 
  [scenario : CandleScenario counts amounts] : 
  amounts.coconut / amounts.almond = counts.coconut / counts.almond :=
sorry

end NUMINAMATH_CALUDE_scent_ratio_equals_candle_ratio_l3098_309810


namespace NUMINAMATH_CALUDE_impossibility_of_transformation_l3098_309805

/-- Represents a triplet of integers -/
structure Triplet where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Represents the allowed operation on a triplet -/
inductive Operation
  | inc_a_dec_bc : Operation
  | inc_b_dec_ac : Operation
  | inc_c_dec_ab : Operation

/-- Applies an operation to a triplet -/
def apply_operation (t : Triplet) (op : Operation) : Triplet :=
  match op with
  | Operation.inc_a_dec_bc => ⟨t.a + 2, t.b - 1, t.c - 1⟩
  | Operation.inc_b_dec_ac => ⟨t.a - 1, t.b + 2, t.c - 1⟩
  | Operation.inc_c_dec_ab => ⟨t.a - 1, t.b - 1, t.c + 2⟩

/-- Checks if a triplet has two zeros -/
def has_two_zeros (t : Triplet) : Prop :=
  (t.a = 0 ∧ t.b = 0) ∨ (t.a = 0 ∧ t.c = 0) ∨ (t.b = 0 ∧ t.c = 0)

/-- Defines a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a triplet -/
def apply_sequence (t : Triplet) : OperationSequence → Triplet
  | [] => t
  | op :: ops => apply_sequence (apply_operation t op) ops

theorem impossibility_of_transformation : 
  ∀ (ops : OperationSequence), ¬(has_two_zeros (apply_sequence ⟨13, 15, 17⟩ ops)) :=
by
  sorry

end NUMINAMATH_CALUDE_impossibility_of_transformation_l3098_309805


namespace NUMINAMATH_CALUDE_eric_marbles_l3098_309819

/-- The number of marbles Eric has -/
def total_marbles (white blue green : ℕ) : ℕ := white + blue + green

/-- Proof that Eric has 20 marbles in total -/
theorem eric_marbles : total_marbles 12 6 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_eric_marbles_l3098_309819


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l3098_309899

/-- Given two points M and N in the plane, this theorem states that
    the equation of the perpendicular bisector of line segment MN
    is x - y + 3 = 0. -/
theorem perpendicular_bisector_equation (M N : ℝ × ℝ) :
  M = (-1, 6) →
  N = (3, 2) →
  ∃ (f : ℝ → ℝ), 
    (∀ x y, f x = y ↔ x - y + 3 = 0) ∧
    (∀ p : ℝ × ℝ, f p.1 = p.2 ↔ 
      (p.1 - M.1)^2 + (p.2 - M.2)^2 = (p.1 - N.1)^2 + (p.2 - N.2)^2 ∧
      (p.1 - M.1) * (N.1 - M.1) + (p.2 - M.2) * (N.2 - M.2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l3098_309899


namespace NUMINAMATH_CALUDE_march_birthdays_march_birthdays_value_l3098_309820

/-- The number of Santana's brothers -/
def total_brothers : ℕ := 7

/-- The number of brothers with birthdays in October -/
def october_birthdays : ℕ := 1

/-- The number of brothers with birthdays in November -/
def november_birthdays : ℕ := 1

/-- The number of brothers with birthdays in December -/
def december_birthdays : ℕ := 2

/-- The number of presents bought in the second half of the year -/
def presents_second_half : ℕ := october_birthdays + november_birthdays + december_birthdays + total_brothers

/-- The difference in presents between the second and first half of the year -/
def present_difference : ℕ := 8

theorem march_birthdays : ℕ :=
  presents_second_half - present_difference
  
theorem march_birthdays_value : march_birthdays = 3 := by
  sorry

end NUMINAMATH_CALUDE_march_birthdays_march_birthdays_value_l3098_309820


namespace NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_75_sin_15_eq_zero_l3098_309877

theorem cos_75_cos_15_minus_sin_75_sin_15_eq_zero :
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) - 
  Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_75_sin_15_eq_zero_l3098_309877


namespace NUMINAMATH_CALUDE_certain_number_is_six_l3098_309811

theorem certain_number_is_six : ∃ x : ℝ, 7 * x - 6 = 4 * x + 12 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_six_l3098_309811


namespace NUMINAMATH_CALUDE_yadav_yearly_savings_l3098_309846

/-- Calculates the yearly savings for Mr. Yadav given his spending habits -/
theorem yadav_yearly_savings (monthly_salary : ℝ) 
  (h1 : 0.5 * (0.4 * monthly_salary) = 3900) : 
  12 * (0.2 * monthly_salary) = 46800 := by
  sorry

#check yadav_yearly_savings

end NUMINAMATH_CALUDE_yadav_yearly_savings_l3098_309846


namespace NUMINAMATH_CALUDE_garage_roof_leak_l3098_309861

/-- The amount of water leaked from three holes in a garage roof over a 2-hour period -/
def water_leaked (largest_hole_rate : ℚ) (time_hours : ℚ) : ℚ :=
  let medium_hole_rate := largest_hole_rate / 2
  let smallest_hole_rate := medium_hole_rate / 3
  let time_minutes := time_hours * 60
  (largest_hole_rate + medium_hole_rate + smallest_hole_rate) * time_minutes

/-- Theorem stating the total amount of water leaked from three holes in a garage roof over a 2-hour period -/
theorem garage_roof_leak : water_leaked 3 2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_garage_roof_leak_l3098_309861


namespace NUMINAMATH_CALUDE_sum_of_differences_l3098_309854

def T : Finset ℕ := Finset.range 9

def M : ℕ := Finset.sum T (λ x => Finset.sum T (λ y => if x > y then 3^x - 3^y else 0))

theorem sum_of_differences (T : Finset ℕ) (M : ℕ) :
  T = Finset.range 9 →
  M = Finset.sum T (λ x => Finset.sum T (λ y => if x > y then 3^x - 3^y else 0)) →
  M = 68896 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_differences_l3098_309854


namespace NUMINAMATH_CALUDE_leifs_oranges_l3098_309875

theorem leifs_oranges (apples : ℕ) (oranges : ℕ) : apples = 14 → oranges = apples + 10 → oranges = 24 := by
  sorry

end NUMINAMATH_CALUDE_leifs_oranges_l3098_309875


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l3098_309881

theorem longest_side_of_triangle (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  Real.tan A = 1/4 ∧
  Real.tan B = 3/5 ∧
  a = min a (min b c) ∧
  a = Real.sqrt 2 ∧
  c = max a (max b c) →
  c = Real.sqrt 17 := by
sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l3098_309881


namespace NUMINAMATH_CALUDE_one_greater_than_digit_squares_l3098_309858

def digit_squares_sum (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d => d^2) |>.sum

theorem one_greater_than_digit_squares : {n : ℕ | n > 0 ∧ n = digit_squares_sum n + 1} = {35, 75} := by
  sorry

end NUMINAMATH_CALUDE_one_greater_than_digit_squares_l3098_309858


namespace NUMINAMATH_CALUDE_linear_function_not_in_second_quadrant_l3098_309825

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear function y = kx + b -/
structure LinearFunction where
  k : ℝ
  b : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: A linear function with k > 0 and b < 0 does not pass through the second quadrant -/
theorem linear_function_not_in_second_quadrant (f : LinearFunction) 
    (h1 : f.k > 0) (h2 : f.b < 0) : 
    ∀ p : Point, p.y = f.k * p.x + f.b → ¬(isInSecondQuadrant p) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_second_quadrant_l3098_309825


namespace NUMINAMATH_CALUDE_tom_free_lessons_l3098_309868

/-- Calculates the number of free dance lessons given the total number of lessons,
    cost per lesson, and total amount paid. -/
def free_lessons (total_lessons : ℕ) (cost_per_lesson : ℕ) (total_paid : ℕ) : ℕ :=
  total_lessons - (total_paid / cost_per_lesson)

/-- Proves that Tom received 2 free dance lessons given the problem conditions. -/
theorem tom_free_lessons :
  let total_lessons : ℕ := 10
  let cost_per_lesson : ℕ := 10
  let total_paid : ℕ := 80
  free_lessons total_lessons cost_per_lesson total_paid = 2 := by
  sorry


end NUMINAMATH_CALUDE_tom_free_lessons_l3098_309868


namespace NUMINAMATH_CALUDE_nonagon_intersection_points_l3098_309806

/-- A regular nonagon is a 9-sided polygon -/
def regular_nonagon : ℕ := 9

/-- The number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of distinct intersection points of diagonals within a regular nonagon -/
def intersection_points (n : ℕ) : ℕ := choose n 4

theorem nonagon_intersection_points :
  intersection_points regular_nonagon = 126 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_intersection_points_l3098_309806


namespace NUMINAMATH_CALUDE_range_of_complex_function_l3098_309826

theorem range_of_complex_function (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (a b : ℝ), a = Real.sqrt 2 - 1 ∧ b = Real.sqrt 2 + 1 ∧
  ∀ θ : ℝ, a ≤ Complex.abs (z^2 + Complex.I * z^2 + 1) ∧
           Complex.abs (z^2 + Complex.I * z^2 + 1) ≤ b :=
by sorry

end NUMINAMATH_CALUDE_range_of_complex_function_l3098_309826


namespace NUMINAMATH_CALUDE_fixed_points_of_square_minus_two_range_of_a_for_two_fixed_points_odd_function_fixed_points_l3098_309807

-- Definition of a fixed point
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Statement 1
theorem fixed_points_of_square_minus_two :
  ∃ (x y : ℝ), x ≠ y ∧ 
    is_fixed_point (fun x => x^2 - 2) x ∧
    is_fixed_point (fun x => x^2 - 2) y ∧
    ∀ z, is_fixed_point (fun x => x^2 - 2) z → (z = x ∨ z = y) :=
sorry

-- Statement 2
theorem range_of_a_for_two_fixed_points (a b : ℝ) :
  (∀ b : ℝ, ∃ (x y : ℝ), x ≠ y ∧
    is_fixed_point (fun x => a*x^2 + b*x - b) x ∧
    is_fixed_point (fun x => a*x^2 + b*x - b) y) →
  (0 < a ∧ a < 1) :=
sorry

-- Statement 3
theorem odd_function_fixed_points (f : ℝ → ℝ) (K : ℕ) :
  (∀ x, f (-x) = -f x) →
  (∃ (S : Finset ℝ), S.card = K ∧ ∀ x ∈ S, is_fixed_point f x) →
  Odd K :=
sorry

end NUMINAMATH_CALUDE_fixed_points_of_square_minus_two_range_of_a_for_two_fixed_points_odd_function_fixed_points_l3098_309807


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l3098_309829

theorem arithmetic_mean_of_first_four_primes_reciprocals :
  let first_four_primes := [2, 3, 5, 7]
  let reciprocals := first_four_primes.map (λ x => 1 / x)
  let arithmetic_mean := (reciprocals.sum) / 4
  arithmetic_mean = 247 / 840 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l3098_309829


namespace NUMINAMATH_CALUDE_num_arrangements_eq_192_l3098_309872

/-- The number of different arrangements for 7 students in a row with specific conditions -/
def num_arrangements : ℕ :=
  let total_students : ℕ := 7
  let middle_student : ℕ := 1
  let together_students : ℕ := 2
  let remaining_students : ℕ := total_students - middle_student - together_students
  let middle_positions : ℕ := 1
  let together_positions : ℕ := 2 * 4
  let remaining_positions : ℕ := remaining_students.factorial
  middle_positions * together_positions * remaining_positions

/-- Theorem stating that the number of arrangements is 192 -/
theorem num_arrangements_eq_192 : num_arrangements = 192 := by
  sorry

end NUMINAMATH_CALUDE_num_arrangements_eq_192_l3098_309872


namespace NUMINAMATH_CALUDE_total_fish_count_l3098_309870

/-- The number of fish owned by Billy, Tony, Sarah, and Bobby -/
def fish_count (billy tony sarah bobby : ℕ) : Prop :=
  (tony = 3 * billy) ∧
  (sarah = tony + 5) ∧
  (bobby = 2 * sarah) ∧
  (billy = 10)

/-- The total number of fish owned by all four people -/
def total_fish (billy tony sarah bobby : ℕ) : ℕ :=
  billy + tony + sarah + bobby

/-- Theorem stating that the total number of fish is 145 -/
theorem total_fish_count :
  ∀ billy tony sarah bobby : ℕ,
  fish_count billy tony sarah bobby →
  total_fish billy tony sarah bobby = 145 :=
by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l3098_309870


namespace NUMINAMATH_CALUDE_rumor_spread_l3098_309832

theorem rumor_spread (n : ℕ) : (∃ m : ℕ, (3^(m+1) - 1) / 2 ≥ 1000 ∧ ∀ k < m, (3^(k+1) - 1) / 2 < 1000) → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_rumor_spread_l3098_309832


namespace NUMINAMATH_CALUDE_sally_peaches_theorem_l3098_309850

/-- The number of peaches Sally picked from the orchard -/
def peaches_picked (initial final : ℕ) : ℕ := final - initial

theorem sally_peaches_theorem (initial final picked : ℕ) 
  (h1 : initial = 13)
  (h2 : final = 55)
  (h3 : picked = peaches_picked initial final) :
  picked = 42 := by sorry

end NUMINAMATH_CALUDE_sally_peaches_theorem_l3098_309850


namespace NUMINAMATH_CALUDE_frequency_converges_to_probability_l3098_309892

-- Define a random event
def RandomEvent : Type := Unit

-- Define the probability of the event
def probability (e : RandomEvent) : ℝ := sorry

-- Define the observed frequency of the event after n experiments
def observedFrequency (e : RandomEvent) (n : ℕ) : ℝ := sorry

-- Statement: As the number of experiments increases, the frequency of the random event
-- will gradually stabilize at the probability of the random event occurring.
theorem frequency_converges_to_probability (e : RandomEvent) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |observedFrequency e n - probability e| < ε :=
sorry

end NUMINAMATH_CALUDE_frequency_converges_to_probability_l3098_309892


namespace NUMINAMATH_CALUDE_regular_polygon_144_degree_interior_angle_l3098_309803

/-- A regular polygon with an interior angle of 144° has 10 sides -/
theorem regular_polygon_144_degree_interior_angle :
  ∀ n : ℕ,
  n > 2 →
  (144 : ℝ) = (n - 2 : ℝ) * 180 / n →
  n = 10 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_144_degree_interior_angle_l3098_309803


namespace NUMINAMATH_CALUDE_find_b_value_l3098_309802

theorem find_b_value (x y b : ℝ) (h1 : y ≠ 0) (h2 : x / (2 * y) = 3 / 2) (h3 : (7 * x + b * y) / (x - 2 * y) = 27) : b = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l3098_309802


namespace NUMINAMATH_CALUDE_eggs_per_unit_is_twelve_l3098_309869

/-- Represents the number of eggs in one unit -/
def eggs_per_unit : ℕ := 12

/-- Represents the number of units supplied to the first store daily -/
def units_to_first_store : ℕ := 5

/-- Represents the number of eggs supplied to the second store daily -/
def eggs_to_second_store : ℕ := 30

/-- Represents the total number of eggs supplied to both stores in a week -/
def total_eggs_per_week : ℕ := 630

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem stating that the number of eggs in one unit is 12 -/
theorem eggs_per_unit_is_twelve :
  eggs_per_unit * units_to_first_store * days_in_week +
  eggs_to_second_store * days_in_week = total_eggs_per_week :=
by sorry

end NUMINAMATH_CALUDE_eggs_per_unit_is_twelve_l3098_309869


namespace NUMINAMATH_CALUDE_stans_paper_words_per_page_l3098_309880

/-- Calculates the number of words per page in Stan's paper. -/
theorem stans_paper_words_per_page 
  (typing_speed : ℕ)        -- Stan's typing speed in words per minute
  (pages : ℕ)               -- Number of pages in the paper
  (water_per_hour : ℕ)      -- Water consumption rate in ounces per hour
  (total_water : ℕ)         -- Total water consumed while writing the paper
  (h1 : typing_speed = 50)  -- Stan types 50 words per minute
  (h2 : pages = 5)          -- The paper is 5 pages long
  (h3 : water_per_hour = 15) -- Stan drinks 15 ounces of water per hour while typing
  (h4 : total_water = 10)   -- Stan drinks 10 ounces of water while writing his paper
  : (typing_speed * (total_water * 60 / water_per_hour)) / pages = 400 := by
  sorry

#check stans_paper_words_per_page

end NUMINAMATH_CALUDE_stans_paper_words_per_page_l3098_309880


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3098_309886

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  2 * X^4 + 9 * X^3 - 38 * X^2 - 50 * X + 35 = 
  (X^2 + 5 * X - 6) * q + (61 * X - 91) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3098_309886


namespace NUMINAMATH_CALUDE_complex_number_conditions_l3098_309818

theorem complex_number_conditions (α : ℂ) :
  α ≠ 1 →
  Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1) →
  Complex.abs (α^3 - 1) = 6 * Complex.abs (α - 1) →
  α = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_number_conditions_l3098_309818


namespace NUMINAMATH_CALUDE_distance_theorem_l3098_309844

/-- The configuration of three squares where the middle one is rotated and lowered -/
structure SquareConfiguration where
  /-- Side length of each square -/
  side_length : ℝ
  /-- Rotation angle of the middle square in radians -/
  rotation_angle : ℝ

/-- Calculate the distance from point B to the original line -/
def distance_to_line (config : SquareConfiguration) : ℝ :=
  sorry

/-- The theorem stating the distance from point B to the original line -/
theorem distance_theorem (config : SquareConfiguration) :
  config.side_length = 1 ∧ config.rotation_angle = π / 4 →
  distance_to_line config = Real.sqrt 2 + 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_theorem_l3098_309844


namespace NUMINAMATH_CALUDE_initial_boarders_count_l3098_309851

theorem initial_boarders_count (initial_boarders day_students new_boarders : ℕ) : 
  initial_boarders > 0 ∧ 
  day_students > 0 ∧
  new_boarders = 66 ∧
  initial_boarders * 12 = day_students * 5 ∧
  (initial_boarders + new_boarders) * 2 = day_students * 1 →
  initial_boarders = 330 := by
sorry

end NUMINAMATH_CALUDE_initial_boarders_count_l3098_309851


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l3098_309821

-- Define a triangle by its side lengths
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

-- State the theorem
theorem triangle_inequality_theorem (t : Triangle) :
  (t.a / (2 * (t.b + t.c))) + (t.b / (2 * (t.c + t.a))) + (t.c / (2 * (t.a + t.b))) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l3098_309821


namespace NUMINAMATH_CALUDE_least_sum_m_n_l3098_309864

theorem least_sum_m_n (m n : ℕ+) (h1 : Nat.gcd (m + n) 330 = 1)
  (h2 : ∃ k : ℕ, m^(m : ℕ) = k * n^(n : ℕ)) (h3 : ¬∃ k : ℕ, m = k * n) :
  m + n ≥ 377 ∧ ∃ m' n' : ℕ+, m' + n' = 377 ∧ 
    Nat.gcd (m' + n') 330 = 1 ∧ 
    (∃ k : ℕ, (m' : ℕ)^(m' : ℕ) = k * (n' : ℕ)^(n' : ℕ)) ∧ 
    ¬∃ k : ℕ, (m' : ℕ) = k * (n' : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l3098_309864


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3098_309879

-- Problem 1
theorem problem_1 (f : ℝ → ℝ) (h : ∀ x ≤ 1, f (1 - Real.sqrt x) = x) :
  ∀ x ≤ 1, f x = x^2 - 2*x + 1 := by sorry

-- Problem 2
theorem problem_2 (f : ℝ → ℝ) 
  (h1 : ∃ a b : ℝ, ∀ x, f x = a * x + b) 
  (h2 : ∀ x, f (f x) = 4 * x + 3) :
  (∀ x, f x = 2 * x + 1) ∨ (∀ x, f x = -2 * x - 3) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3098_309879


namespace NUMINAMATH_CALUDE_least_possible_q_l3098_309845

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_right_triangle (p q : ℕ) : Prop := p + q = 90

theorem least_possible_q (p q : ℕ) :
  is_right_triangle p q →
  is_prime p →
  p > q →
  (∀ q' < q, ¬(is_right_triangle p' q' ∧ is_prime p' ∧ p' > q' ∧ p' < p)) →
  q = 7 :=
sorry

end NUMINAMATH_CALUDE_least_possible_q_l3098_309845


namespace NUMINAMATH_CALUDE_equation_solution_l3098_309804

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  x₁ ≠ 2/3 ∧ x₂ ≠ 2/3 ∧
  (3 * x₁ + 2) / (3 * x₁^2 + 7 * x₁ - 6) = (3 * x₁) / (3 * x₁ - 2) ∧
  (3 * x₂ + 2) / (3 * x₂^2 + 7 * x₂ - 6) = (3 * x₂) / (3 * x₂ - 2) ∧
  x₁ = -2 ∧ x₂ = 1/3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3098_309804


namespace NUMINAMATH_CALUDE_correct_product_after_reversal_error_l3098_309840

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ := 
  (n % 10) * 10 + (n / 10)

theorem correct_product_after_reversal_error (a b : ℕ) : 
  is_two_digit a → 
  is_two_digit b → 
  reverse_digits a * b = 378 → 
  a * b = 504 := by
  sorry

end NUMINAMATH_CALUDE_correct_product_after_reversal_error_l3098_309840


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3098_309855

theorem absolute_value_inequality (x : ℝ) : 
  3 ≤ |x - 3| ∧ |x - 3| ≤ 7 ↔ ((-4 ≤ x ∧ x ≤ 0) ∨ (6 ≤ x ∧ x ≤ 10)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3098_309855


namespace NUMINAMATH_CALUDE_distinct_roots_sum_bound_l3098_309843

theorem distinct_roots_sum_bound (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ → 
  r₁^2 + p*r₁ + 8 = 0 → 
  r₂^2 + p*r₂ + 8 = 0 → 
  |r₁ + r₂| > 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_sum_bound_l3098_309843


namespace NUMINAMATH_CALUDE_only_fourth_prop_true_l3098_309812

-- Define the propositions
def prop1 : Prop := ∀ a b m : ℝ, (a < b → a * m^2 < b * m^2)
def prop2 : Prop := ∀ p q : Prop, (p ∨ q → p ∧ q)
def prop3 : Prop := ∀ x : ℝ, (x > 1 → x > 2) ∧ ¬(x > 2 → x > 1)
def prop4 : Prop := (¬∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0)

-- Theorem statement
theorem only_fourth_prop_true : ¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4 := by
  sorry

end NUMINAMATH_CALUDE_only_fourth_prop_true_l3098_309812


namespace NUMINAMATH_CALUDE_power_product_equality_l3098_309830

theorem power_product_equality : (15 : ℕ)^2 * 8^3 * 256 = 29491200 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l3098_309830


namespace NUMINAMATH_CALUDE_total_spent_is_72_l3098_309838

/-- The cost of a single trick deck in dollars -/
def deck_cost : ℕ := 9

/-- The number of decks Edward bought -/
def edward_decks : ℕ := 4

/-- The number of decks Edward's friend bought -/
def friend_decks : ℕ := 4

/-- The total amount spent by Edward and his friend -/
def total_spent : ℕ := deck_cost * (edward_decks + friend_decks)

theorem total_spent_is_72 : total_spent = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_72_l3098_309838


namespace NUMINAMATH_CALUDE_sin_cos_fourth_power_range_l3098_309824

theorem sin_cos_fourth_power_range (x : ℝ) : 
  0.5 ≤ Real.sin x ^ 4 + Real.cos x ^ 4 ∧ Real.sin x ^ 4 + Real.cos x ^ 4 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_fourth_power_range_l3098_309824


namespace NUMINAMATH_CALUDE_acid_dilution_l3098_309837

/-- Given an initial acid solution of m ounces at m% concentration, 
    prove that adding x ounces of water to reach (m-15)% concentration
    results in x = 15m / (m-15) for m > 30 -/
theorem acid_dilution (m : ℝ) (x : ℝ) (h₁ : m > 30) :
  (m * m / 100 = (m - 15) * (m + x) / 100) → x = 15 * m / (m - 15) := by
  sorry


end NUMINAMATH_CALUDE_acid_dilution_l3098_309837


namespace NUMINAMATH_CALUDE_triangle_existence_condition_l3098_309876

/-- A triangle with given inscribed circle radius, circumscribed circle radius, and one angle. -/
structure Triangle where
  r : ℝ  -- radius of inscribed circle
  R : ℝ  -- radius of circumscribed circle
  α : ℝ  -- one angle of the triangle (in radians)

/-- Theorem stating the conditions for the existence of a triangle with given parameters. -/
theorem triangle_existence_condition (t : Triangle) :
  (∃ (triangle : Triangle), triangle = t) ↔ 
  (0 < t.α ∧ t.α < Real.pi ∧ t.R ≥ 2 * t.r) :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_condition_l3098_309876


namespace NUMINAMATH_CALUDE_vector_equality_holds_l3098_309894

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def vector_equality (e₁ e₂ a b : V) : Prop :=
  (e₁ ≠ 0 ∧ e₂ ≠ 0) ∧  -- non-zero vectors
  (∀ (r : ℝ), r • e₁ ≠ e₂) ∧  -- non-collinear
  (a = 3 • e₁ - 2 • e₂) ∧
  (b = e₂ - 2 • e₁) ∧
  ((1/3) • a + b) + (a - (3/2) • b) + (2 • b - a) = -2 • e₁ + (5/6) • e₂

theorem vector_equality_holds (e₁ e₂ a b : V) :
  vector_equality e₁ e₂ a b := by sorry

end NUMINAMATH_CALUDE_vector_equality_holds_l3098_309894


namespace NUMINAMATH_CALUDE_toms_average_speed_l3098_309883

/-- Prove that Tom's average speed is 45 mph given the race conditions -/
theorem toms_average_speed (karen_speed : ℝ) (karen_delay : ℝ) (karen_win_margin : ℝ) (tom_distance : ℝ) :
  karen_speed = 60 →
  karen_delay = 1 / 15 →
  karen_win_margin = 4 →
  tom_distance = 24 →
  (tom_distance / ((tom_distance + karen_win_margin) / karen_speed + karen_delay)) = 45 :=
by sorry

end NUMINAMATH_CALUDE_toms_average_speed_l3098_309883


namespace NUMINAMATH_CALUDE_negation_equivalence_l3098_309853

theorem negation_equivalence :
  (¬ ∃ (x y : ℝ), 2*x + 3*y + 3 < 0) ↔ (∀ (x y : ℝ), 2*x + 3*y + 3 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3098_309853


namespace NUMINAMATH_CALUDE_vector_coordinates_l3098_309848

/-- Given two vectors a and b in ℝ², prove that if a is parallel to b, 
    a = (2, -1), and the magnitude of b is 2√5, then b is either (-4, 2) or (4, -2) -/
theorem vector_coordinates (a b : ℝ × ℝ) : 
  (∃ (k : ℝ), b = k • a) →  -- a is parallel to b
  a = (2, -1) →
  Real.sqrt ((b.1)^2 + (b.2)^2) = 2 * Real.sqrt 5 →  -- magnitude of b is 2√5
  (b = (-4, 2) ∨ b = (4, -2)) :=
by sorry

end NUMINAMATH_CALUDE_vector_coordinates_l3098_309848


namespace NUMINAMATH_CALUDE_binary_to_base4_conversion_l3098_309839

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldr (λ bit acc => 2 * acc + if bit then 1 else 0) 0

/-- Converts a decimal number to its base-4 representation -/
def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The binary representation of 1101010101₂ -/
def binary_num : List Bool :=
  [true, true, false, true, false, true, false, true, false, true]

theorem binary_to_base4_conversion :
  decimal_to_base4 (binary_to_decimal binary_num) = [3, 1, 1, 1, 1] := by
  sorry

#eval decimal_to_base4 (binary_to_decimal binary_num)

end NUMINAMATH_CALUDE_binary_to_base4_conversion_l3098_309839


namespace NUMINAMATH_CALUDE_arithmetic_sequence_13th_term_l3098_309808

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_13th_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_5 : a 5 = 3) 
  (h_9 : a 9 = 6) : 
  a 13 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_13th_term_l3098_309808


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3098_309862

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 2 = 2 →
  a 3 + a 4 = 10 →
  a 5 + a 6 = 18 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3098_309862


namespace NUMINAMATH_CALUDE_sams_price_per_sheet_l3098_309822

/-- Represents the pricing structure of a photo company -/
structure PhotoCompany where
  price_per_sheet : ℝ
  sitting_fee : ℝ

/-- Calculates the total cost for a given number of sheets -/
def total_cost (company : PhotoCompany) (sheets : ℝ) : ℝ :=
  company.price_per_sheet * sheets + company.sitting_fee

/-- Proves that Sam's Picture Emporium charges $1.50 per sheet -/
theorem sams_price_per_sheet :
  let johns := PhotoCompany.mk 2.75 125
  let sams := PhotoCompany.mk x 140
  total_cost johns 12 = total_cost sams 12 →
  x = 1.50 := by
  sorry

end NUMINAMATH_CALUDE_sams_price_per_sheet_l3098_309822


namespace NUMINAMATH_CALUDE_s₁_less_than_s₂_l3098_309857

/-- Centroid of a triangle -/
structure Centroid (Point : Type*) (Triangle : Type*) where
  center : Point
  triangle : Triangle

/-- Calculate s₁ for a triangle with its centroid -/
def s₁ {Point : Type*} {Triangle : Type*} (c : Centroid Point Triangle) (distance : Point → Point → ℝ) : ℝ :=
  let G := c.center
  let A := sorry
  let B := sorry
  let C := sorry
  2 * (distance G A + distance G B + distance G C)

/-- Calculate s₂ for a triangle -/
def s₂ {Point : Type*} {Triangle : Type*} (t : Triangle) (distance : Point → Point → ℝ) : ℝ :=
  let A := sorry
  let B := sorry
  let C := sorry
  3 * (distance A B + distance B C + distance C A)

/-- The main theorem: s₁ < s₂ for any triangle with its centroid -/
theorem s₁_less_than_s₂ {Point : Type*} {Triangle : Type*} 
  (c : Centroid Point Triangle) (distance : Point → Point → ℝ) :
  s₁ c distance < s₂ c.triangle distance :=
sorry

end NUMINAMATH_CALUDE_s₁_less_than_s₂_l3098_309857


namespace NUMINAMATH_CALUDE_average_rate_for_trip_l3098_309866

/-- Given a trip with the following conditions:
  - Total distance is 640 miles
  - First half is driven at 80 miles per hour
  - Second half takes 200% longer than the first half
  Prove that the average rate for the entire trip is 40 miles per hour -/
theorem average_rate_for_trip (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  (total_distance / (total_distance / (2 * first_half_speed) * (1 + second_half_time_factor))) = 40 :=
by sorry

end NUMINAMATH_CALUDE_average_rate_for_trip_l3098_309866


namespace NUMINAMATH_CALUDE_set_union_problem_l3098_309888

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {1, 2^a}
def B (a b : ℝ) : Set ℝ := {a, b}

-- State the theorem
theorem set_union_problem (a b : ℝ) : 
  A a ∩ B a b = {1/2} → A a ∪ B a b = {-1, 1/2, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l3098_309888


namespace NUMINAMATH_CALUDE_production_system_l3098_309884

/-- Represents the profit functions and properties of a production system with two products. -/
theorem production_system (total_workers : ℕ) 
  (prod_rate_A prod_rate_B : ℕ) 
  (profit_per_A profit_per_B cost_increase_B : ℚ) : 
  total_workers = 65 → 
  prod_rate_A = 2 →
  prod_rate_B = 1 →
  profit_per_A = 15 →
  profit_per_B = 120 →
  cost_increase_B = 2 →
  ∃ (profit_A profit_B : ℚ → ℚ) (x : ℚ),
    (∀ x, profit_A x = 1950 - 30 * x) ∧
    (∀ x, profit_B x = 120 * x - 2 * x^2) ∧
    (profit_A x - profit_B x = 1250 → x = 5) ∧
    (∃ (total_profit : ℚ → ℚ),
      (∀ x, total_profit x = profit_A x + profit_B x) ∧
      (∀ y, total_profit y ≤ 2962) ∧
      (total_profit 22 = 2962 ∨ total_profit 23 = 2962)) :=
by sorry

end NUMINAMATH_CALUDE_production_system_l3098_309884


namespace NUMINAMATH_CALUDE_quadratic_equation_positive_roots_l3098_309863

theorem quadratic_equation_positive_roots (m : ℝ) : 
  (∀ x : ℝ, x^2 + (m+2)*x + m + 5 = 0 → x > 0) ↔ -5 < m ∧ m ≤ -4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_positive_roots_l3098_309863


namespace NUMINAMATH_CALUDE_sum_mod_nine_l3098_309856

theorem sum_mod_nine : (2469 + 2470 + 2471 + 2472 + 2473 + 2474) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l3098_309856


namespace NUMINAMATH_CALUDE_circle_properties_l3098_309828

theorem circle_properties (A : ℝ) (h : A = 64 * Real.pi) : ∃ (r C : ℝ), r = 8 ∧ C = 16 * Real.pi ∧ A = Real.pi * r^2 ∧ C = 2 * Real.pi * r := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l3098_309828


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3098_309833

-- First expression
theorem simplify_expression_1 (a b : ℝ) :
  4 * (a + b) + 2 * (a + b) - (a + b) = 5 * a + 5 * b := by sorry

-- Second expression
theorem simplify_expression_2 (m : ℝ) :
  3 * m / 2 - (5 * m / 2 - 1) + 3 * (4 - m) = -4 * m + 13 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3098_309833


namespace NUMINAMATH_CALUDE_exists_transformation_458_to_14_l3098_309891

-- Define the operations
def double (n : ℕ) : ℕ := 2 * n

def erase_last_digit (n : ℕ) : ℕ :=
  if n < 10 then n else n / 10

-- Define a single step transformation
inductive Step
| Double : Step
| EraseLastDigit : Step

def apply_step (n : ℕ) (s : Step) : ℕ :=
  match s with
  | Step.Double => double n
  | Step.EraseLastDigit => erase_last_digit n

-- Define a sequence of steps
def apply_steps (n : ℕ) (steps : List Step) : ℕ :=
  steps.foldl apply_step n

-- Theorem statement
theorem exists_transformation_458_to_14 :
  ∃ (steps : List Step), apply_steps 458 steps = 14 := by
  sorry

end NUMINAMATH_CALUDE_exists_transformation_458_to_14_l3098_309891


namespace NUMINAMATH_CALUDE_complex_arithmetic_simplification_l3098_309878

theorem complex_arithmetic_simplification :
  (7 - 3 * Complex.I) - 3 * (2 + 4 * Complex.I) = 1 - 15 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_simplification_l3098_309878


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_seven_mod_twelve_l3098_309827

theorem least_five_digit_congruent_to_seven_mod_twelve :
  ∃ n : ℕ, 
    (n ≥ 10000 ∧ n < 100000) ∧  -- n is a five-digit number
    n % 12 = 7 ∧               -- n is congruent to 7 (mod 12)
    (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 12 = 7 → m ≥ n) ∧  -- n is the least such number
    n = 10003 :=               -- n equals 10003
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_seven_mod_twelve_l3098_309827


namespace NUMINAMATH_CALUDE_fraction_exponent_equality_l3098_309890

theorem fraction_exponent_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x / y)^(-3/4 : ℝ) = 4 * (y / x)^3 := by sorry

end NUMINAMATH_CALUDE_fraction_exponent_equality_l3098_309890


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3098_309847

/-- The distance between the foci of the ellipse (x²/36) + (y²/9) = 9 is 2√3 -/
theorem ellipse_foci_distance :
  let ellipse := {(x, y) : ℝ × ℝ | x^2 / 36 + y^2 / 9 = 9}
  ∃ f₁ f₂ : ℝ × ℝ, f₁ ∈ ellipse ∧ f₂ ∈ ellipse ∧ 
    ∀ p ∈ ellipse, dist p f₁ + dist p f₂ = 2 * Real.sqrt 36 ∧
    dist f₁ f₂ = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3098_309847


namespace NUMINAMATH_CALUDE_highway_work_completion_fraction_l3098_309849

theorem highway_work_completion_fraction :
  let total_length : ℝ := 2 -- km
  let initial_workers : ℕ := 100
  let initial_duration : ℕ := 50 -- days
  let initial_daily_hours : ℕ := 8
  let actual_work_days : ℕ := 25
  let additional_workers : ℕ := 60
  let new_daily_hours : ℕ := 10

  let total_man_hours : ℝ := initial_workers * initial_duration * initial_daily_hours
  let remaining_man_hours : ℝ := (initial_workers + additional_workers) * (initial_duration - actual_work_days) * new_daily_hours

  total_man_hours = remaining_man_hours →
  (total_man_hours - remaining_man_hours) / total_man_hours = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_highway_work_completion_fraction_l3098_309849


namespace NUMINAMATH_CALUDE_sports_club_overlap_l3098_309852

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ)
  (h_total : total = 150)
  (h_badminton : badminton = 75)
  (h_tennis : tennis = 60)
  (h_neither : neither = 25) :
  badminton + tennis - (total - neither) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l3098_309852


namespace NUMINAMATH_CALUDE_apples_given_to_teachers_l3098_309823

/-- Given Sarah's apple distribution, prove the number given to teachers. -/
theorem apples_given_to_teachers 
  (initial_apples : Nat) 
  (final_apples : Nat) 
  (friends_given_apples : Nat) 
  (apples_eaten : Nat) 
  (h1 : initial_apples = 25)
  (h2 : final_apples = 3)
  (h3 : friends_given_apples = 5)
  (h4 : apples_eaten = 1) :
  initial_apples - final_apples - friends_given_apples - apples_eaten = 16 := by
  sorry

#check apples_given_to_teachers

end NUMINAMATH_CALUDE_apples_given_to_teachers_l3098_309823


namespace NUMINAMATH_CALUDE_f_properties_l3098_309841

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x + x * Real.log x

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := a + Real.log x + 1

-- State the theorem
theorem f_properties (a : ℝ) :
  (f_deriv a (Real.exp 1) = 3) →
  (∃ (k : ℤ), k = 3 ∧ 
    (∀ x > 1, f 1 x - ↑k * x + ↑k > 0) ∧
    (∀ k' > ↑k, ∃ x > 1, f 1 x - ↑k' * x + ↑k' ≤ 0)) →
  (a = 1) ∧
  (∀ x ∈ Set.Ioo 0 (Real.exp (-2)), ∀ y ∈ Set.Ioo x (Real.exp (-2)), f 1 y < f 1 x) ∧
  (∀ x ∈ Set.Ioi (Real.exp (-2)), ∀ y ∈ Set.Ioi x, f 1 y > f 1 x) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l3098_309841


namespace NUMINAMATH_CALUDE_greatest_3digit_base8_div_by_7_l3098_309809

def base_8_to_10 (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem greatest_3digit_base8_div_by_7 :
  ∃ (n : Nat),
    n ≤ 777 ∧
    base_8_to_10 n = 511 ∧
    511 % 7 = 0 ∧
    ∀ (m : Nat), m ≤ 777 ∧ m ≠ n ∧ (base_8_to_10 m) % 7 = 0 → base_8_to_10 m < 511 :=
by
  sorry

#eval base_8_to_10 777

end NUMINAMATH_CALUDE_greatest_3digit_base8_div_by_7_l3098_309809


namespace NUMINAMATH_CALUDE_alice_walking_time_l3098_309801

/-- Given Bob's walking time and distance, and the relationship between Alice and Bob's walking times and distances, prove that Alice would take 21 minutes to walk 7 miles. -/
theorem alice_walking_time 
  (bob_distance : ℝ) 
  (bob_time : ℝ) 
  (alice_distance : ℝ) 
  (alice_bob_time_ratio : ℝ) 
  (alice_target_distance : ℝ) 
  (h1 : bob_distance = 6) 
  (h2 : bob_time = 36) 
  (h3 : alice_distance = 4) 
  (h4 : alice_bob_time_ratio = 1/3) 
  (h5 : alice_target_distance = 7) : 
  (alice_target_distance / (alice_distance / (alice_bob_time_ratio * bob_time))) = 21 :=
by sorry

end NUMINAMATH_CALUDE_alice_walking_time_l3098_309801


namespace NUMINAMATH_CALUDE_not_perfect_square_l3098_309814

theorem not_perfect_square (n : ℕ) : ¬ ∃ (a : ℕ), 3 * n + 2 = a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l3098_309814


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_and_15_l3098_309871

theorem smallest_divisible_by_1_to_12_and_15 : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 12 → k ∣ n) ∧ 
  (15 ∣ n) ∧ 
  (∀ m : ℕ, m < n → ¬(∀ k : ℕ, k ≤ 12 → k ∣ m) ∨ ¬(15 ∣ m)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_and_15_l3098_309871


namespace NUMINAMATH_CALUDE_complement_of_M_l3098_309898

def U : Set Int := {-1, -2, -3, -4}
def M : Set Int := {-2, -3}

theorem complement_of_M : U \ M = {-1, -4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_l3098_309898


namespace NUMINAMATH_CALUDE_RS_length_value_l3098_309893

/-- Triangle ABC with given side lengths and angle bisectors -/
structure TriangleABC where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- Altitude AD
  AD : ℝ
  -- Points R and S on AD
  AR : ℝ
  AS : ℝ
  -- Conditions
  side_lengths : AB = 11 ∧ BC = 13 ∧ CA = 14
  altitude : AD > 0
  R_on_AD : 0 < AR ∧ AR < AD
  S_on_AD : 0 < AS ∧ AS < AD
  BE_bisector : AR / (AD - AR) = CA / BC
  CF_bisector : AS / (AD - AS) = AB / BC

/-- The length of RS in the given triangle -/
def RS_length (t : TriangleABC) : ℝ := t.AR - t.AS

/-- Theorem stating that RS length is equal to 645√95 / 4551 -/
theorem RS_length_value (t : TriangleABC) : RS_length t = 645 * Real.sqrt 95 / 4551 := by
  sorry

end NUMINAMATH_CALUDE_RS_length_value_l3098_309893


namespace NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l3098_309835

/-- The value of 'a' when the line y = x + a is tangent to the curve y = ln x -/
theorem tangent_line_to_ln_curve (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (x₀ + a = Real.log x₀) ∧  -- The point (x₀, ln x₀) is on the line y = x + a
    (1 = 1 / x₀)) →           -- The slopes of the line and curve are equal at x₀
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l3098_309835


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3098_309842

theorem inequality_and_equality_condition (a b c d : ℝ) :
  (Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) ≥ Real.sqrt ((a + c)^2 + (b + d)^2)) ∧
  (Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) = Real.sqrt ((a + c)^2 + (b + d)^2) ↔ a * d = b * c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3098_309842


namespace NUMINAMATH_CALUDE_mike_toys_count_l3098_309816

/-- Proves that Mike has 6 toys given the conditions of the problem -/
theorem mike_toys_count :
  ∀ (mike annie tom : ℕ),
  annie = 3 * mike →
  tom = annie + 2 →
  mike + annie + tom = 56 →
  mike = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_mike_toys_count_l3098_309816


namespace NUMINAMATH_CALUDE_abc_value_l3098_309895

theorem abc_value (a b c : ℝ) 
  (sum_eq : a + b + c = 4)
  (sum_prod_eq : b * c + c * a + a * b = 5)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 10) : 
  a * b * c = 2 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l3098_309895


namespace NUMINAMATH_CALUDE_cheryl_eggs_count_l3098_309897

/-- The number of eggs found by Kevin -/
def kevin_eggs : ℕ := 5

/-- The number of eggs found by Bonnie -/
def bonnie_eggs : ℕ := 13

/-- The number of eggs found by George -/
def george_eggs : ℕ := 9

/-- The number of additional eggs Cheryl found compared to the others -/
def cheryl_additional_eggs : ℕ := 29

/-- Theorem stating that Cheryl found 56 eggs -/
theorem cheryl_eggs_count : 
  kevin_eggs + bonnie_eggs + george_eggs + cheryl_additional_eggs = 56 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_eggs_count_l3098_309897


namespace NUMINAMATH_CALUDE_fourth_power_of_cube_of_third_smallest_prime_l3098_309860

def third_smallest_prime : ℕ := 5

theorem fourth_power_of_cube_of_third_smallest_prime :
  (third_smallest_prime ^ 3) ^ 4 = 244140625 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_of_cube_of_third_smallest_prime_l3098_309860


namespace NUMINAMATH_CALUDE_lamp_savings_l3098_309817

theorem lamp_savings (num_lamps : ℕ) (original_price : ℚ) (discount_rate : ℚ) (additional_discount : ℚ) :
  num_lamps = 3 →
  original_price = 15 →
  discount_rate = 0.25 →
  additional_discount = 5 →
  num_lamps * original_price - (num_lamps * (original_price * (1 - discount_rate)) - additional_discount) = 16.25 :=
by sorry

end NUMINAMATH_CALUDE_lamp_savings_l3098_309817


namespace NUMINAMATH_CALUDE_max_z_value_l3098_309889

theorem max_z_value (x y z : ℝ) (sum_eq : x + y + z = 3) (prod_eq : x*y + y*z + z*x = 2) :
  z ≤ 5/3 ∧ ∃ (x' y' z' : ℝ), x' + y' + z' = 3 ∧ x'*y' + y'*z' + z'*x' = 2 ∧ z' = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_max_z_value_l3098_309889
