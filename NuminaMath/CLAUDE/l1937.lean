import Mathlib

namespace NUMINAMATH_CALUDE_order_of_numbers_l1937_193756

theorem order_of_numbers : 
  let A := (Nat.factorial 8) ^ (Nat.factorial 8)
  let B := 8 ^ (8 ^ 8)
  let C := 8 ^ 88
  let D := 8 ^ 64
  D < C ∧ C < B ∧ B < A := by sorry

end NUMINAMATH_CALUDE_order_of_numbers_l1937_193756


namespace NUMINAMATH_CALUDE_unique_factorization_l1937_193702

/-- A factorization of 2210 into a two-digit and a three-digit number -/
structure Factorization :=
  (a : ℕ) (b : ℕ)
  (h1 : 10 ≤ a ∧ a ≤ 99)
  (h2 : 100 ≤ b ∧ b ≤ 999)
  (h3 : a * b = 2210)

/-- Two factorizations are considered equal if they have the same factors (regardless of order) -/
def factorization_eq (f1 f2 : Factorization) : Prop :=
  (f1.a = f2.a ∧ f1.b = f2.b) ∨ (f1.a = f2.b ∧ f1.b = f2.a)

/-- The set of all valid factorizations of 2210 -/
def factorizations : Set Factorization :=
  {f : Factorization | true}

theorem unique_factorization : ∃! (f : Factorization), f ∈ factorizations :=
sorry

end NUMINAMATH_CALUDE_unique_factorization_l1937_193702


namespace NUMINAMATH_CALUDE_max_intersection_points_l1937_193769

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The number of intersection points between a circle and a line --/
def intersection_count (circle : Circle) (line : Line) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of intersection points between a circle and a line is 2 --/
theorem max_intersection_points (circle : Circle) (line : Line) :
  intersection_count circle line ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_intersection_points_l1937_193769


namespace NUMINAMATH_CALUDE_cos_two_pi_thirds_plus_two_alpha_l1937_193797

theorem cos_two_pi_thirds_plus_two_alpha (α : Real) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos ((2 * π) / 3 + 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_pi_thirds_plus_two_alpha_l1937_193797


namespace NUMINAMATH_CALUDE_max_profit_at_six_l1937_193705

/-- The profit function for a certain product -/
def profit_function (x : ℝ) : ℝ := -2 * x^3 + 18 * x^2

/-- The derivative of the profit function -/
def profit_derivative (x : ℝ) : ℝ := -6 * x^2 + 36 * x

theorem max_profit_at_six :
  ∃ (x : ℝ), x > 0 ∧
  (∀ (y : ℝ), y > 0 → profit_function y ≤ profit_function x) ∧
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_six_l1937_193705


namespace NUMINAMATH_CALUDE_exam_probabilities_l1937_193727

def prob_above_90 : ℝ := 0.18
def prob_80_to_89 : ℝ := 0.51
def prob_70_to_79 : ℝ := 0.15
def prob_60_to_69 : ℝ := 0.09

theorem exam_probabilities :
  (prob_above_90 + prob_80_to_89 = 0.69) ∧
  (prob_above_90 + prob_80_to_89 + prob_70_to_79 + prob_60_to_69 = 0.93) := by
sorry

end NUMINAMATH_CALUDE_exam_probabilities_l1937_193727


namespace NUMINAMATH_CALUDE_waiter_tip_problem_l1937_193787

theorem waiter_tip_problem (total_customers : ℕ) (tip_amount : ℕ) (total_tips : ℕ) :
  total_customers = 7 →
  tip_amount = 9 →
  total_tips = 27 →
  total_customers - (total_tips / tip_amount) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_waiter_tip_problem_l1937_193787


namespace NUMINAMATH_CALUDE_amy_spent_32_pounds_l1937_193759

/-- Represents the amount spent by Chloe in pounds -/
def chloe_spent : ℝ := 20

/-- Represents the amount spent by Becky as a fraction of Chloe's spending -/
def becky_spent_ratio : ℝ := 0.15

/-- Represents the amount spent by Amy as a fraction above Chloe's spending -/
def amy_spent_ratio : ℝ := 1.6

/-- The total amount spent by all three shoppers in pounds -/
def total_spent : ℝ := 55

theorem amy_spent_32_pounds :
  let becky_spent := becky_spent_ratio * chloe_spent
  let amy_spent := amy_spent_ratio * chloe_spent
  becky_spent + amy_spent + chloe_spent = total_spent ∧
  amy_spent = 32 := by sorry

end NUMINAMATH_CALUDE_amy_spent_32_pounds_l1937_193759


namespace NUMINAMATH_CALUDE_mean_equality_problem_l1937_193729

theorem mean_equality_problem (x : ℝ) : 
  (8 + 16 + 24) / 3 = (10 + x) / 2 → x = 22 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_problem_l1937_193729


namespace NUMINAMATH_CALUDE_tetrahedron_volume_specific_l1937_193733

def tetrahedron_volume (AB AC AD BC BD CD : ℝ) : ℝ := sorry

theorem tetrahedron_volume_specific : 
  tetrahedron_volume 2 4 3 (Real.sqrt 17) (Real.sqrt 13) 5 = 6 * Real.sqrt 247 / 64 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_specific_l1937_193733


namespace NUMINAMATH_CALUDE_max_m_for_monotonic_f_l1937_193722

/-- Given a function f(x) = x^4 - (1/3)mx^3 + (1/2)x^2 + 1, 
    if f is monotonically increasing on (0,1), 
    then the maximum value of m is 4 -/
theorem max_m_for_monotonic_f (m : ℝ) : 
  let f := fun (x : ℝ) ↦ x^4 - (1/3)*m*x^3 + (1/2)*x^2 + 1
  (∀ x ∈ Set.Ioo 0 1, Monotone f) → m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_max_m_for_monotonic_f_l1937_193722


namespace NUMINAMATH_CALUDE_araceli_luana_numbers_l1937_193739

theorem araceli_luana_numbers : ∃ (a b c : ℕ), 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  100 * a + 10 * b + c = (10 * a + b) + (10 * b + c) + (10 * c + a) ∧
  a = 1 ∧ b = 9 ∧ c = 8 := by
sorry

end NUMINAMATH_CALUDE_araceli_luana_numbers_l1937_193739


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l1937_193783

theorem geometric_series_common_ratio :
  let a₁ : ℚ := 8 / 10
  let a₂ : ℚ := -6 / 15
  let a₃ : ℚ := 54 / 225
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 2 → (a₁ * r ^ (n - 1) = if n % 2 = 0 then -a₁ * (1 / 2) ^ (n - 1) else a₁ * (1 / 2) ^ (n - 1))) →
  r = -1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l1937_193783


namespace NUMINAMATH_CALUDE_local_minimum_condition_l1937_193784

/-- The function f(x) = x(x - m)^2 attains a local minimum at x = 1 -/
theorem local_minimum_condition (m : ℝ) :
  let f : ℝ → ℝ := λ x => x * (x - m)^2
  (∃ δ > 0, ∀ x, |x - 1| < δ → f x ≥ f 1) →
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_local_minimum_condition_l1937_193784


namespace NUMINAMATH_CALUDE_two_zeros_implies_a_is_inverse_e_l1937_193747

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x + x else a*x - Real.log x

theorem two_zeros_implies_a_is_inverse_e (a : ℝ) (h_a_pos : a > 0) :
  (∃! x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  a = Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_two_zeros_implies_a_is_inverse_e_l1937_193747


namespace NUMINAMATH_CALUDE_local_minimum_at_two_l1937_193755

/-- The function f(x) defined as x(x-c)² --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- The derivative of f(x) with respect to x --/
def f_derivative (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*c*x + c^2

theorem local_minimum_at_two (c : ℝ) :
  (f_derivative c 2 = 0 ∧ c = 2) → 
  ∃ ε > 0, ∀ x, x ≠ 2 ∧ |x - 2| < ε → f c x > f c 2 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_at_two_l1937_193755


namespace NUMINAMATH_CALUDE_lily_bushes_theorem_l1937_193710

theorem lily_bushes_theorem (bushes : Fin 19 → ℕ) : 
  ∃ i : Fin 19, Even ((bushes i) + (bushes ((i + 1) % 19))) := by
  sorry

end NUMINAMATH_CALUDE_lily_bushes_theorem_l1937_193710


namespace NUMINAMATH_CALUDE_original_paint_intensity_l1937_193704

theorem original_paint_intensity 
  (original_fraction : Real) 
  (replacement_intensity : Real) 
  (new_intensity : Real) 
  (replaced_fraction : Real) :
  original_fraction = 0.5 →
  replacement_intensity = 0.2 →
  new_intensity = 0.15 →
  replaced_fraction = 0.5 →
  (1 - replaced_fraction) * original_fraction + replaced_fraction * replacement_intensity = new_intensity →
  original_fraction = 0.1 := by
sorry

end NUMINAMATH_CALUDE_original_paint_intensity_l1937_193704


namespace NUMINAMATH_CALUDE_no_solution_system_l1937_193762

/-- Proves that the system of equations 3x - 4y = 5 and 6x - 8y = 7 has no solution -/
theorem no_solution_system :
  ¬ ∃ (x y : ℝ), (3 * x - 4 * y = 5) ∧ (6 * x - 8 * y = 7) := by
sorry

end NUMINAMATH_CALUDE_no_solution_system_l1937_193762


namespace NUMINAMATH_CALUDE_count_arrangements_l1937_193766

/-- The number of arrangements of 5 students (2 male and 3 female) in a line formation,
    where one specific male student does not stand at either end and only two of the
    three female students stand next to each other. -/
def num_arrangements : ℕ := 48

/-- Proves that the number of different possible arrangements is 48. -/
theorem count_arrangements :
  let total_students : ℕ := 5
  let male_students : ℕ := 2
  let female_students : ℕ := 3
  let specific_male_not_at_ends : Bool := true
  let two_females_adjacent : Bool := true
  num_arrangements = 48 := by sorry

end NUMINAMATH_CALUDE_count_arrangements_l1937_193766


namespace NUMINAMATH_CALUDE_sphere_surface_area_relation_l1937_193788

theorem sphere_surface_area_relation (R₁ R₂ R₃ S₁ S₂ S₃ : ℝ) 
  (h₁ : R₁ + 2 * R₂ = 3 * R₃)
  (h₂ : S₁ = 4 * Real.pi * R₁^2)
  (h₃ : S₂ = 4 * Real.pi * R₂^2)
  (h₄ : S₃ = 4 * Real.pi * R₃^2) :
  Real.sqrt S₁ + 2 * Real.sqrt S₂ = 3 * Real.sqrt S₃ := by
  sorry

#check sphere_surface_area_relation

end NUMINAMATH_CALUDE_sphere_surface_area_relation_l1937_193788


namespace NUMINAMATH_CALUDE_a_5_value_l1937_193718

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_5_value (a : ℕ → ℝ) (h_geo : geometric_sequence a) 
  (h_3 : a 3 = -1) (h_7 : a 7 = -9) : a 5 = -3 := by
  sorry

end NUMINAMATH_CALUDE_a_5_value_l1937_193718


namespace NUMINAMATH_CALUDE_charity_ticket_revenue_l1937_193757

theorem charity_ticket_revenue :
  ∀ (full_price_tickets half_price_tickets : ℕ) (full_price : ℕ),
    full_price_tickets + half_price_tickets = 180 →
    full_price_tickets * full_price + half_price_tickets * (full_price / 2) = 2750 →
    full_price_tickets * full_price = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_charity_ticket_revenue_l1937_193757


namespace NUMINAMATH_CALUDE_correct_operation_l1937_193721

theorem correct_operation (a b : ℝ) : 3*a + 2*b - 2*(a - b) = a + 4*b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1937_193721


namespace NUMINAMATH_CALUDE_semi_circle_area_l1937_193744

/-- The area of a semi-circle with diameter 10 meters is 12.5π square meters. -/
theorem semi_circle_area (π : ℝ) : 
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let semi_circle_area : ℝ := π * radius^2 / 2
  semi_circle_area = 12.5 * π := by
  sorry

end NUMINAMATH_CALUDE_semi_circle_area_l1937_193744


namespace NUMINAMATH_CALUDE_min_different_numbers_l1937_193750

theorem min_different_numbers (total : ℕ) (max_freq : ℕ) (min_diff : ℕ) : 
  total = 2019 →
  max_freq = 10 →
  min_diff = 225 →
  (∀ k : ℕ, k < min_diff → k * (max_freq - 1) + max_freq < total) ∧
  (min_diff * (max_freq - 1) + max_freq ≥ total) := by
  sorry

end NUMINAMATH_CALUDE_min_different_numbers_l1937_193750


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1937_193782

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ+,
  z = Nat.gcd x y →
  x + y^2 + z^3 = x * y * z →
  ((x = 4 ∧ y = 2) ∨ (x = 4 ∧ y = 6) ∨ (x = 5 ∧ y = 2) ∨ (x = 5 ∧ y = 3)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1937_193782


namespace NUMINAMATH_CALUDE_min_value_perpendicular_vectors_l1937_193780

theorem min_value_perpendicular_vectors (x y : ℝ) :
  (x - 1) * 4 + 2 * y = 0 →
  ∃ (min : ℝ), min = 6 ∧ ∀ (z : ℝ), z = 9^x + 3^y → z ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_perpendicular_vectors_l1937_193780


namespace NUMINAMATH_CALUDE_no_real_roots_implies_nonzero_sum_l1937_193730

theorem no_real_roots_implies_nonzero_sum (a b c : ℝ) : 
  a ≠ 0 → 
  (∀ x : ℝ, a * x^2 + b * x + c ≠ 0) → 
  a^3 + a * b + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_implies_nonzero_sum_l1937_193730


namespace NUMINAMATH_CALUDE_evaluate_expression_at_negative_one_l1937_193728

-- Define the expression as a function of x
def f (x : ℚ) : ℚ := (4 + x * (4 + x) - 4^2) / (x - 4 + x^3)

-- State the theorem
theorem evaluate_expression_at_negative_one :
  f (-1) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_at_negative_one_l1937_193728


namespace NUMINAMATH_CALUDE_complement_of_M_l1937_193724

def U : Set Nat := {1, 2, 3, 4}

def M : Set Nat := {x ∈ U | x^2 - 5*x + 6 = 0}

theorem complement_of_M :
  (U \ M) = {1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l1937_193724


namespace NUMINAMATH_CALUDE_complex_ratio_l1937_193717

theorem complex_ratio (a b : ℝ) (h1 : a * b ≠ 0) :
  let z : ℂ := Complex.mk a b
  (∃ (k : ℝ), z * Complex.mk 1 (-2) = k) → a / b = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_ratio_l1937_193717


namespace NUMINAMATH_CALUDE_certain_number_equation_l1937_193740

theorem certain_number_equation (x : ℝ) : 5 * 1.6 - (2 * 1.4) / x = 4 ↔ x = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l1937_193740


namespace NUMINAMATH_CALUDE_digit_sum_19_or_20_l1937_193711

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def are_different (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def equation_holds (a b c d : ℕ) : Prop :=
  ∃ (x y z : ℕ), is_digit x ∧ is_digit y ∧ is_digit z ∧
  (a * 100 + 50 + b) + (400 + c * 10 + d) = x * 100 + y * 10 + z

theorem digit_sum_19_or_20 (a b c d : ℕ) :
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧
  are_different a b c d ∧
  equation_holds a b c d →
  a + b + c + d = 19 ∨ a + b + c + d = 20 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_19_or_20_l1937_193711


namespace NUMINAMATH_CALUDE_sandrine_dishes_washed_l1937_193758

def number_of_pears_picked : ℕ := 50

def number_of_bananas_cooked (pears : ℕ) : ℕ := 3 * pears

def number_of_dishes_washed (bananas : ℕ) : ℕ := bananas + 10

theorem sandrine_dishes_washed :
  number_of_dishes_washed (number_of_bananas_cooked number_of_pears_picked) = 160 := by
  sorry

end NUMINAMATH_CALUDE_sandrine_dishes_washed_l1937_193758


namespace NUMINAMATH_CALUDE_seventh_rack_dvd_count_l1937_193725

/-- Calculates the number of DVDs on a given rack based on the previous two racks -/
def dvd_count (n : ℕ) : ℕ :=
  match n with
  | 0 => 3  -- First rack
  | 1 => 4  -- Second rack
  | n + 2 => ((dvd_count (n + 1) - dvd_count n) * 2) + dvd_count (n + 1)

/-- The number of DVDs on the seventh rack is 66 -/
theorem seventh_rack_dvd_count :
  dvd_count 6 = 66 := by sorry

end NUMINAMATH_CALUDE_seventh_rack_dvd_count_l1937_193725


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l1937_193731

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^48 + i^96 + i^144 = 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l1937_193731


namespace NUMINAMATH_CALUDE_count_special_numbers_proof_l1937_193767

/-- The number of five-digit numbers with two pairs of adjacent equal digits,
    where digits from different pairs are different, and the remaining digit
    is different from all other digits. -/
def count_special_numbers : ℕ := 1944

/-- The set of valid configurations for the special five-digit numbers. -/
inductive Configuration : Type
  | AABBC : Configuration
  | AACBB : Configuration
  | CAABB : Configuration

/-- The number of possible choices for the first digit of the number. -/
def first_digit_choices : ℕ := 9

/-- The number of possible choices for the second digit of the number. -/
def second_digit_choices : ℕ := 9

/-- The number of possible choices for the third digit of the number. -/
def third_digit_choices : ℕ := 8

/-- The number of valid configurations. -/
def num_configurations : ℕ := 3

theorem count_special_numbers_proof :
  count_special_numbers =
    num_configurations * first_digit_choices * second_digit_choices * third_digit_choices :=
by sorry

end NUMINAMATH_CALUDE_count_special_numbers_proof_l1937_193767


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l1937_193775

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_third : a 3 = 50)
  (h_fifth : a 5 = 30) :
  a 7 = 10 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l1937_193775


namespace NUMINAMATH_CALUDE_solution_in_interval_l1937_193748

open Real

/-- A monotonically increasing function on (0, +∞) satisfying f[f(x) - ln x] = 1 -/
def MonotonicFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 < x ∧ x < y → f x < f y) ∧
  (∀ x, 0 < x → f (f x - log x) = 1)

/-- The solution to f(x) - f'(x) = 1 lies in (1, 2) -/
theorem solution_in_interval (f : ℝ → ℝ) (hf : MonotonicFunction f) :
  ∃ x, 1 < x ∧ x < 2 ∧ f x - (deriv f) x = 1 :=
sorry

end NUMINAMATH_CALUDE_solution_in_interval_l1937_193748


namespace NUMINAMATH_CALUDE_john_volunteer_hours_per_year_l1937_193799

/-- 
Given that John volunteers twice a month for 3 hours each time, 
this theorem proves that he volunteers for 72 hours per year.
-/
theorem john_volunteer_hours_per_year 
  (times_per_month : ℕ) 
  (hours_per_time : ℕ) 
  (h1 : times_per_month = 2) 
  (h2 : hours_per_time = 3) : 
  times_per_month * 12 * hours_per_time = 72 := by
  sorry

end NUMINAMATH_CALUDE_john_volunteer_hours_per_year_l1937_193799


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1937_193708

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo π (3 * π / 2))
  (h2 : Real.tan (2 * α) = -Real.cos α / (2 + Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1937_193708


namespace NUMINAMATH_CALUDE_square_sum_eq_two_l1937_193726

theorem square_sum_eq_two (a b : ℝ) : (a^2 + b^2)^4 - 8*(a^2 + b^2)^2 + 16 = 0 → a^2 + b^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_eq_two_l1937_193726


namespace NUMINAMATH_CALUDE_square_position_after_2023_transformations_l1937_193798

-- Define a square as a list of four vertices
def Square := List Char

-- Define the transformations
def rotate90CW (s : Square) : Square :=
  match s with
  | [a, b, c, d] => [d, a, b, c]
  | _ => s

def reflectVertical (s : Square) : Square :=
  match s with
  | [a, b, c, d] => [c, b, a, d]
  | _ => s

def rotate180 (s : Square) : Square :=
  match s with
  | [a, b, c, d] => [c, d, a, b]
  | _ => s

-- Define the sequence of transformations
def transform (s : Square) (n : Nat) : Square :=
  match n % 3 with
  | 0 => rotate180 s
  | 1 => rotate90CW s
  | _ => reflectVertical s

-- Main theorem
theorem square_position_after_2023_transformations (initial : Square) :
  initial = ['A', 'B', 'C', 'D'] →
  (transform initial 2023) = ['C', 'B', 'A', 'D'] := by
  sorry


end NUMINAMATH_CALUDE_square_position_after_2023_transformations_l1937_193798


namespace NUMINAMATH_CALUDE_symmetry_properties_l1937_193789

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def symmetry_x_axis (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, -p.z⟩

def symmetry_yOz_plane (p : Point3D) : Point3D :=
  ⟨-p.x, p.y, p.z⟩

def symmetry_y_axis (p : Point3D) : Point3D :=
  ⟨-p.x, p.y, -p.z⟩

def symmetry_origin (p : Point3D) : Point3D :=
  ⟨-p.x, -p.y, -p.z⟩

theorem symmetry_properties (p : Point3D) :
  (symmetry_x_axis p = ⟨p.x, -p.y, -p.z⟩) ∧
  (symmetry_yOz_plane p = ⟨-p.x, p.y, p.z⟩) ∧
  (symmetry_y_axis p = ⟨-p.x, p.y, -p.z⟩) ∧
  (symmetry_origin p = ⟨-p.x, -p.y, -p.z⟩) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_properties_l1937_193789


namespace NUMINAMATH_CALUDE_allowance_percentage_increase_l1937_193715

def middle_school_allowance : ℕ := 8 + 2

def senior_year_allowance : ℕ := 2 * middle_school_allowance + 5

def allowance_increase : ℕ := senior_year_allowance - middle_school_allowance

def percentage_increase : ℚ := (allowance_increase : ℚ) / (middle_school_allowance : ℚ) * 100

theorem allowance_percentage_increase :
  percentage_increase = 150 := by sorry

end NUMINAMATH_CALUDE_allowance_percentage_increase_l1937_193715


namespace NUMINAMATH_CALUDE_set_intersection_problem_l1937_193779

theorem set_intersection_problem (M N P : Set Nat) 
  (hM : M = {1})
  (hN : N = {1, 2})
  (hP : P = {1, 2, 3}) :
  (M ∪ N) ∩ P = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l1937_193779


namespace NUMINAMATH_CALUDE_range_of_trigonometric_function_l1937_193720

theorem range_of_trigonometric_function :
  ∀ x : ℝ, -1 ≤ Real.sin x * Real.cos x + Real.sin x + Real.cos x ∧ 
           Real.sin x * Real.cos x + Real.sin x + Real.cos x ≤ 1/2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_trigonometric_function_l1937_193720


namespace NUMINAMATH_CALUDE_square_difference_equality_l1937_193713

theorem square_difference_equality : 1005^2 - 995^2 - 1003^2 + 997^2 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1937_193713


namespace NUMINAMATH_CALUDE_matt_writing_difference_l1937_193707

/-- The number of words Matt can write per minute with his right hand -/
def right_hand_speed : ℕ := 10

/-- The number of words Matt can write per minute with his left hand -/
def left_hand_speed : ℕ := 7

/-- The duration of time in minutes -/
def duration : ℕ := 5

/-- The difference in words written between Matt's right and left hands over the given duration -/
def word_difference : ℕ := (right_hand_speed - left_hand_speed) * duration

theorem matt_writing_difference : word_difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_matt_writing_difference_l1937_193707


namespace NUMINAMATH_CALUDE_rain_probability_both_locations_l1937_193703

theorem rain_probability_both_locations (p_no_rain_A p_no_rain_B : ℝ) 
  (h1 : p_no_rain_A = 0.3)
  (h2 : p_no_rain_B = 0.4)
  (h3 : 0 ≤ p_no_rain_A ∧ p_no_rain_A ≤ 1)
  (h4 : 0 ≤ p_no_rain_B ∧ p_no_rain_B ≤ 1) :
  (1 - p_no_rain_A) * (1 - p_no_rain_B) = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_both_locations_l1937_193703


namespace NUMINAMATH_CALUDE_rectangular_container_volume_l1937_193738

theorem rectangular_container_volume 
  (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : b * c = 20) 
  (h3 : c * a = 12) : 
  a * b * c = 60 := by
sorry

end NUMINAMATH_CALUDE_rectangular_container_volume_l1937_193738


namespace NUMINAMATH_CALUDE_regression_line_equation_l1937_193777

/-- Regression line parameters -/
structure RegressionParams where
  x_bar : ℝ
  y_bar : ℝ
  slope : ℝ

/-- Regression line equation -/
def regression_line (params : RegressionParams) (x : ℝ) : ℝ :=
  params.slope * x + (params.y_bar - params.slope * params.x_bar)

/-- Theorem: Given the slope, x̄, and ȳ, prove the regression line equation -/
theorem regression_line_equation (params : RegressionParams)
  (h1 : params.x_bar = 4)
  (h2 : params.y_bar = 5)
  (h3 : params.slope = 2) :
  ∀ x, regression_line params x = 2 * x - 3 := by
  sorry

#check regression_line_equation

end NUMINAMATH_CALUDE_regression_line_equation_l1937_193777


namespace NUMINAMATH_CALUDE_paper_piles_problem_l1937_193716

theorem paper_piles_problem :
  ∃! N : ℕ,
    1000 < N ∧ N < 2000 ∧
    N % 2 = 1 ∧
    N % 3 = 1 ∧
    N % 4 = 1 ∧
    N % 5 = 1 ∧
    N % 6 = 1 ∧
    N % 7 = 1 ∧
    N % 8 = 1 ∧
    N % 41 = 0 :=
by sorry

end NUMINAMATH_CALUDE_paper_piles_problem_l1937_193716


namespace NUMINAMATH_CALUDE_sphere_ratio_l1937_193781

theorem sphere_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 16 →
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 1 / 64 := by
sorry

end NUMINAMATH_CALUDE_sphere_ratio_l1937_193781


namespace NUMINAMATH_CALUDE_condition_iff_prime_or_prime_square_l1937_193764

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def is_prime_square (n : ℕ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ n = p^2

def satisfies_condition (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ d : ℕ, d ≥ 2 → d ∣ n → (d - 1) ∣ (n - 1)

theorem condition_iff_prime_or_prime_square (n : ℕ) :
  satisfies_condition n ↔ is_prime n ∨ is_prime_square n :=
sorry

end NUMINAMATH_CALUDE_condition_iff_prime_or_prime_square_l1937_193764


namespace NUMINAMATH_CALUDE_mod_equivalence_solution_l1937_193763

theorem mod_equivalence_solution : 
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ -3137 [ZMOD 7] ↔ n = 1 ∨ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_solution_l1937_193763


namespace NUMINAMATH_CALUDE_max_negative_integers_l1937_193778

theorem max_negative_integers
  (a b c d e f : ℤ)
  (h : a * b + c * d * e * f < 0) :
  ∃ (neg_count : ℕ),
    neg_count ≤ 4 ∧
    (∃ (na nb nc nd ne nf : ℕ),
      (na + nb + nc + nd + ne + nf = neg_count) ∧
      (a < 0 ↔ na = 1) ∧
      (b < 0 ↔ nb = 1) ∧
      (c < 0 ↔ nc = 1) ∧
      (d < 0 ↔ nd = 1) ∧
      (e < 0 ↔ ne = 1) ∧
      (f < 0 ↔ nf = 1)) ∧
    ∀ (m : ℕ), m > neg_count →
      ¬∃ (ma mb mc md me mf : ℕ),
        (ma + mb + mc + md + me + mf = m) ∧
        (a < 0 ↔ ma = 1) ∧
        (b < 0 ↔ mb = 1) ∧
        (c < 0 ↔ mc = 1) ∧
        (d < 0 ↔ md = 1) ∧
        (e < 0 ↔ me = 1) ∧
        (f < 0 ↔ mf = 1) := by
  sorry

end NUMINAMATH_CALUDE_max_negative_integers_l1937_193778


namespace NUMINAMATH_CALUDE_system2_solution_l1937_193791

variable (a b : ℝ)

-- Define the first system of equations and its solution
def system1_eq1 (x y : ℝ) : Prop := a * x - b * y = 3
def system1_eq2 (x y : ℝ) : Prop := a * x + b * y = 5
def system1_solution (x y : ℝ) : Prop := x = 2 ∧ y = 1

-- Define the second system of equations
def system2_eq1 (m n : ℝ) : Prop := a * (m + 2 * n) - 2 * b * n = 6
def system2_eq2 (m n : ℝ) : Prop := a * (m + 2 * n) + 2 * b * n = 10

-- State the theorem
theorem system2_solution :
  (∃ x y, system1_eq1 a b x y ∧ system1_eq2 a b x y ∧ system1_solution x y) →
  (∃ m n, system2_eq1 a b m n ∧ system2_eq2 a b m n ∧ m = 2 ∧ n = 1) :=
by sorry

end NUMINAMATH_CALUDE_system2_solution_l1937_193791


namespace NUMINAMATH_CALUDE_infinitely_many_m_with_coprime_binomial_l1937_193776

theorem infinitely_many_m_with_coprime_binomial (k l : ℕ+) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ m ∈ S, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_m_with_coprime_binomial_l1937_193776


namespace NUMINAMATH_CALUDE_mayoral_election_votes_l1937_193746

theorem mayoral_election_votes (Z Y X : ℕ) : 
  Z = 25000 → 
  Y = Z - (2/5 : ℚ) * Z →
  X = Y + (1/2 : ℚ) * Y →
  X = 22500 := by
  sorry

end NUMINAMATH_CALUDE_mayoral_election_votes_l1937_193746


namespace NUMINAMATH_CALUDE_shortest_tree_height_l1937_193734

/-- The heights of four trees satisfying certain conditions -/
structure TreeHeights where
  tallest : ℝ
  second_tallest : ℝ
  third_tallest : ℝ
  shortest : ℝ
  tallest_height : tallest = 108
  second_tallest_height : second_tallest = tallest / 2 - 6
  third_tallest_height : third_tallest = second_tallest / 4
  shortest_height : shortest = second_tallest + third_tallest - 2

/-- The height of the shortest tree is 58 feet -/
theorem shortest_tree_height (t : TreeHeights) : t.shortest = 58 := by
  sorry

end NUMINAMATH_CALUDE_shortest_tree_height_l1937_193734


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1937_193712

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {1, 3, 5, 7}

theorem complement_of_A_in_U :
  (U \ A) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1937_193712


namespace NUMINAMATH_CALUDE_couple_ticket_cost_l1937_193760

theorem couple_ticket_cost (single_ticket_cost : ℚ) (total_sales : ℚ) 
  (total_attendance : ℕ) (couple_tickets_sold : ℕ) :
  single_ticket_cost = 20 →
  total_sales = 2280 →
  total_attendance = 128 →
  couple_tickets_sold = 16 →
  ∃ couple_ticket_cost : ℚ,
    couple_ticket_cost = 22.5 ∧
    total_sales = (total_attendance - 2 * couple_tickets_sold) * single_ticket_cost + 
                  couple_tickets_sold * couple_ticket_cost :=
by
  sorry


end NUMINAMATH_CALUDE_couple_ticket_cost_l1937_193760


namespace NUMINAMATH_CALUDE_condition_2_condition_4_condition_1_not_sufficient_condition_3_not_sufficient_l1937_193795

-- Define the types for planes and lines
variable {Point : Type*}
variable {Line : Type*}
variable {Plane : Type*}

-- Define the necessary relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the planes α and β
variable (α β : Plane)

-- Theorem for condition ②
theorem condition_2 
  (h : ∀ l : Line, contains α l → line_parallel_plane l β) :
  parallel α β :=
sorry

-- Theorem for condition ④
theorem condition_4 
  (a b : Line)
  (h1 : perpendicular a α)
  (h2 : perpendicular b β)
  (h3 : line_parallel a b) :
  parallel α β :=
sorry

-- Theorem for condition ①
theorem condition_1_not_sufficient 
  (h : ∃ S : Set Line, (∀ l ∈ S, contains α l ∧ line_parallel_plane l β) ∧ Set.Infinite S) :
  ¬(parallel α β → True) :=
sorry

-- Theorem for condition ③
theorem condition_3_not_sufficient 
  (a b : Line)
  (h1 : contains α a)
  (h2 : contains β b)
  (h3 : line_parallel_plane a β)
  (h4 : line_parallel_plane b α) :
  ¬(parallel α β → True) :=
sorry

end NUMINAMATH_CALUDE_condition_2_condition_4_condition_1_not_sufficient_condition_3_not_sufficient_l1937_193795


namespace NUMINAMATH_CALUDE_product_of_numbers_l1937_193742

theorem product_of_numbers (x y : ℝ) : x + y = 40 ∧ x - y = 16 → x * y = 336 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1937_193742


namespace NUMINAMATH_CALUDE_ajay_distance_theorem_l1937_193770

/-- Calculates the distance traveled given speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Proves that given Ajay's speed of 50 km/hour and a travel time of 20 hours, 
    the distance traveled is 1000 km -/
theorem ajay_distance_theorem :
  let speed : ℝ := 50
  let time : ℝ := 20
  distance_traveled speed time = 1000 := by
sorry

end NUMINAMATH_CALUDE_ajay_distance_theorem_l1937_193770


namespace NUMINAMATH_CALUDE_max_correct_is_23_l1937_193706

/-- Represents a test score --/
structure TestScore where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correct answers for a given test score --/
def max_correct_answers (ts : TestScore) : ℕ :=
  sorry

/-- Theorem stating that for the given test conditions, the maximum number of correct answers is 23 --/
theorem max_correct_is_23 :
  let ts : TestScore := {
    total_questions := 30,
    correct_points := 4,
    incorrect_points := -1,
    total_score := 85
  }
  max_correct_answers ts = 23 := by
  sorry

end NUMINAMATH_CALUDE_max_correct_is_23_l1937_193706


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1937_193765

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 → 
  (∃ (s : ℝ), 81 * s = b ∧ b * s = 8/27) → 
  b = 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1937_193765


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1937_193772

theorem partial_fraction_decomposition :
  ∃! (A B C : ℚ),
    ∀ (x : ℚ), x ≠ 2 → x ≠ 4 →
      (3 * x + 7) / ((x - 4) * (x - 2)^2) =
      A / (x - 4) + B / (x - 2) + C / (x - 2)^2 ∧
      A = 19 / 4 ∧ B = -19 / 4 ∧ C = -13 / 2 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1937_193772


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1937_193785

theorem sqrt_x_div_sqrt_y (x y : ℝ) :
  (1/3)^2 + (1/4)^2 = (13*x / 53*y) * ((1/5)^2 + (1/6)^2) →
  Real.sqrt x / Real.sqrt y = 1092 / 338 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1937_193785


namespace NUMINAMATH_CALUDE_problem_paths_l1937_193745

/-- Represents the graph of points and their connections -/
structure PointGraph where
  blue_points : Nat
  red_points : Nat
  red_connected_to_blue : Bool
  blue_connected_to_each_other : Bool

/-- Calculates the number of paths between red points -/
def count_paths (g : PointGraph) : Nat :=
  sorry

/-- The specific graph configuration from the problem -/
def problem_graph : PointGraph :=
  { blue_points := 8
  , red_points := 2
  , red_connected_to_blue := true
  , blue_connected_to_each_other := true }

/-- Theorem stating the number of paths in the problem -/
theorem problem_paths :
  count_paths problem_graph = 645120 :=
by sorry

end NUMINAMATH_CALUDE_problem_paths_l1937_193745


namespace NUMINAMATH_CALUDE_range_not_real_l1937_193794

/-- Given real numbers a and b satisfying ab = a + b + 3, 
    the range of (a-1)b is not equal to R. -/
theorem range_not_real : ¬ (∀ (y : ℝ), ∃ (a b : ℝ), a * b = a + b + 3 ∧ (a - 1) * b = y) := by
  sorry

end NUMINAMATH_CALUDE_range_not_real_l1937_193794


namespace NUMINAMATH_CALUDE_badger_walnuts_l1937_193790

theorem badger_walnuts (badger_walnuts_per_hole fox_walnuts_per_hole : ℕ)
  (h_badger_walnuts : badger_walnuts_per_hole = 5)
  (h_fox_walnuts : fox_walnuts_per_hole = 7)
  (h_hole_diff : ℕ)
  (h_hole_diff_value : h_hole_diff = 2)
  (badger_holes fox_holes : ℕ)
  (h_holes_relation : badger_holes = fox_holes + h_hole_diff)
  (total_walnuts : ℕ)
  (h_total_equality : badger_walnuts_per_hole * badger_holes = fox_walnuts_per_hole * fox_holes)
  (h_total_walnuts : total_walnuts = badger_walnuts_per_hole * badger_holes) :
  total_walnuts = 35 :=
by sorry

end NUMINAMATH_CALUDE_badger_walnuts_l1937_193790


namespace NUMINAMATH_CALUDE_space_for_another_circle_l1937_193786

/-- The side length of the large square N -/
def N : ℝ := 6

/-- The side length of the small squares -/
def small_square_side : ℝ := 1

/-- The diameter of the circles -/
def circle_diameter : ℝ := 1

/-- The number of small squares -/
def num_squares : ℕ := 4

/-- The number of circles -/
def num_circles : ℕ := 3

/-- The theorem stating that there is space for another circle -/
theorem space_for_another_circle :
  (N - 1)^2 - (num_squares * (small_square_side^2 + small_square_side * circle_diameter + Real.pi * (circle_diameter / 2)^2) +
   num_circles * Real.pi * (circle_diameter / 2)^2) > 0 := by
  sorry

end NUMINAMATH_CALUDE_space_for_another_circle_l1937_193786


namespace NUMINAMATH_CALUDE_cos_225_degrees_l1937_193773

theorem cos_225_degrees : Real.cos (225 * π / 180) = -1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l1937_193773


namespace NUMINAMATH_CALUDE_largest_coeff_x3_sum_64_l1937_193736

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The condition that the coefficient of x^3 is the largest in (1+x)^n -/
def coeff_x3_largest (n : ℕ) : Prop :=
  ∀ k, k ≠ 3 → binomial n 3 ≥ binomial n k

/-- The sum of all coefficients in the expansion of (1+x)^n -/
def sum_coefficients (n : ℕ) : ℕ := 2^n

theorem largest_coeff_x3_sum_64 :
  ∀ n : ℕ, coeff_x3_largest n → sum_coefficients n = 64 := by sorry

end NUMINAMATH_CALUDE_largest_coeff_x3_sum_64_l1937_193736


namespace NUMINAMATH_CALUDE_product_division_theorem_l1937_193792

theorem product_division_theorem (x y : ℝ) (hx : x = 1.6666666666666667) (hx_nonzero : x ≠ 0) :
  Real.sqrt ((5 * x) / y) = x → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_division_theorem_l1937_193792


namespace NUMINAMATH_CALUDE_max_n_is_eleven_l1937_193793

/-- A coloring of integers from 1 to 14 with two colors -/
def Coloring := Fin 14 → Bool

/-- Check if there exist pairs of numbers with the same color and given difference -/
def hasPairsWithDifference (c : Coloring) (k : Nat) (color : Bool) : Prop :=
  ∃ i j, i < j ∧ j ≤ 14 ∧ j - i = k ∧ c i = color ∧ c j = color

/-- The property that a coloring satisfies the conditions for a given n -/
def validColoring (c : Coloring) (n : Nat) : Prop :=
  ∀ k, k ≤ n → hasPairsWithDifference c k true ∧ hasPairsWithDifference c k false

/-- The main theorem: the maximum possible n is 11 -/
theorem max_n_is_eleven :
  (∃ c : Coloring, validColoring c 11) ∧
  (∀ c : Coloring, ¬validColoring c 12) :=
sorry

end NUMINAMATH_CALUDE_max_n_is_eleven_l1937_193793


namespace NUMINAMATH_CALUDE_total_rats_l1937_193709

theorem total_rats (elodie hunter kenia : ℕ) : 
  elodie = 30 →
  elodie = hunter + 10 →
  kenia = 3 * (elodie + hunter) →
  elodie + hunter + kenia = 200 := by
sorry

end NUMINAMATH_CALUDE_total_rats_l1937_193709


namespace NUMINAMATH_CALUDE_crayons_remaining_l1937_193768

theorem crayons_remaining (initial_crayons : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) : 
  initial_crayons = 48 → 
  kiley_fraction = 1/4 →
  joe_fraction = 1/2 →
  (initial_crayons - (kiley_fraction * initial_crayons).floor - 
   (joe_fraction * (initial_crayons - (kiley_fraction * initial_crayons).floor)).floor) = 18 :=
by sorry

end NUMINAMATH_CALUDE_crayons_remaining_l1937_193768


namespace NUMINAMATH_CALUDE_cube_sum_magnitude_l1937_193741

theorem cube_sum_magnitude (w z : ℂ) (h1 : Complex.abs (w + z) = 2) (h2 : Complex.abs (w^2 + z^2) = 15) :
  Complex.abs (w^3 + z^3) = 41 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_magnitude_l1937_193741


namespace NUMINAMATH_CALUDE_valid_division_l1937_193735

theorem valid_division (divisor quotient remainder dividend : ℕ) : 
  divisor = 3040 →
  quotient = 8 →
  remainder = 7 →
  dividend = 24327 →
  dividend = divisor * quotient + remainder :=
by sorry

end NUMINAMATH_CALUDE_valid_division_l1937_193735


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l1937_193723

theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x > 0 ∧ y > 0 ∧ 2 * x^2 - (m + 1) * x + m = 0 ∧ 2 * y^2 - (m + 1) * y + m = 0) 
  ↔ 
  (0 < m ∧ m < 3 - 2 * Real.sqrt 2) ∨ (m > 3 + 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l1937_193723


namespace NUMINAMATH_CALUDE_range_of_f_l1937_193761

noncomputable def g (x : ℝ) : ℝ := x^2 - 2

noncomputable def f (x : ℝ) : ℝ :=
  if x < g x then g x + x + 4 else g x - x

theorem range_of_f :
  Set.range f = Set.Icc (-2.25) 0 ∪ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1937_193761


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1937_193751

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 8) : y = 26 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1937_193751


namespace NUMINAMATH_CALUDE_z_purely_imaginary_z_in_fourth_quadrant_l1937_193754

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := Complex.mk (m * (m + 2)) (m^2 + m - 2)

-- Part 1: z is purely imaginary iff m = 0
theorem z_purely_imaginary (m : ℝ) : z m = Complex.I * Complex.im (z m) ↔ m = 0 :=
sorry

-- Part 2: z is in the fourth quadrant iff 0 < m < 1
theorem z_in_fourth_quadrant (m : ℝ) : 
  (Complex.re (z m) > 0 ∧ Complex.im (z m) < 0) ↔ (0 < m ∧ m < 1) :=
sorry

end NUMINAMATH_CALUDE_z_purely_imaginary_z_in_fourth_quadrant_l1937_193754


namespace NUMINAMATH_CALUDE_speakers_cost_l1937_193701

def total_spent : ℚ := 387.85
def cd_player_cost : ℚ := 139.38
def new_tires_cost : ℚ := 112.46

theorem speakers_cost (total : ℚ) (cd : ℚ) (tires : ℚ) 
  (h1 : total = total_spent) 
  (h2 : cd = cd_player_cost) 
  (h3 : tires = new_tires_cost) : 
  total - (cd + tires) = 136.01 := by
  sorry

end NUMINAMATH_CALUDE_speakers_cost_l1937_193701


namespace NUMINAMATH_CALUDE_product_difference_difference_of_products_l1937_193749

theorem product_difference (a b c d : ℝ) (h : a * b = c) : 
  a * d - a * b = a * (d - b) :=
by sorry

theorem difference_of_products : 
  (16.47 * 34) - (16.47 * 24) = 164.7 :=
by sorry

end NUMINAMATH_CALUDE_product_difference_difference_of_products_l1937_193749


namespace NUMINAMATH_CALUDE_sum_of_roots_l1937_193753

theorem sum_of_roots (a b : ℝ) : 
  a ≠ b → a * (a - 4) = 12 → b * (b - 4) = 12 → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1937_193753


namespace NUMINAMATH_CALUDE_gcd_lcm_product_75_90_l1937_193719

theorem gcd_lcm_product_75_90 : Nat.gcd 75 90 * Nat.lcm 75 90 = 6750 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_75_90_l1937_193719


namespace NUMINAMATH_CALUDE_soap_duration_l1937_193700

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The cost of one bar of soap in dollars -/
def cost_per_bar : ℚ := 8

/-- The total cost of soap for a year in dollars -/
def yearly_cost : ℚ := 48

/-- The number of months one bar of soap lasts -/
def months_per_bar : ℚ := yearly_cost / cost_per_bar * months_in_year / yearly_cost * cost_per_bar

theorem soap_duration : months_per_bar = 2 := by
  sorry

end NUMINAMATH_CALUDE_soap_duration_l1937_193700


namespace NUMINAMATH_CALUDE_break_even_point_l1937_193737

def parts_cost : ℕ := 3600
def patent_cost : ℕ := 4500
def variable_cost : ℕ := 25
def marketing_cost : ℕ := 2000
def selling_price : ℕ := 180

def total_fixed_cost : ℕ := parts_cost + patent_cost + marketing_cost
def contribution_margin : ℕ := selling_price - variable_cost

def break_even (n : ℕ) : Prop :=
  n * selling_price ≥ total_fixed_cost + n * variable_cost

theorem break_even_point : 
  ∀ m : ℕ, break_even m → m ≥ 66 :=
by sorry

end NUMINAMATH_CALUDE_break_even_point_l1937_193737


namespace NUMINAMATH_CALUDE_Only_Statement3_Is_Correct_l1937_193732

-- Define the basic properties of functions
def Monotonic_Increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def Odd_Function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def Even_Function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def Symmetric_About_Y_Axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define the four statements
def Statement1 : Prop :=
  Monotonic_Increasing (fun x => -1/x)

def Statement2 : Prop :=
  ∀ f : ℝ → ℝ, Odd_Function f → f 0 = 0

def Statement3 : Prop :=
  ∀ f : ℝ → ℝ, Even_Function f → Symmetric_About_Y_Axis f

def Statement4 : Prop :=
  ∀ f : ℝ → ℝ, Odd_Function f → Even_Function f → ∀ x, f x = 0

-- Theorem stating that only Statement3 is correct
theorem Only_Statement3_Is_Correct :
  ¬Statement1 ∧ ¬Statement2 ∧ Statement3 ∧ ¬Statement4 :=
sorry

end NUMINAMATH_CALUDE_Only_Statement3_Is_Correct_l1937_193732


namespace NUMINAMATH_CALUDE_adam_has_more_apples_l1937_193714

/-- The number of apples Adam has -/
def adam_apples : ℕ := 14

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 9

/-- The difference in apples between Adam and Jackie -/
def apple_difference : ℕ := adam_apples - jackie_apples

theorem adam_has_more_apples : apple_difference = 5 := by
  sorry

end NUMINAMATH_CALUDE_adam_has_more_apples_l1937_193714


namespace NUMINAMATH_CALUDE_clothing_factory_production_adjustment_l1937_193743

/-- Represents the scenario of a clothing factory adjusting its production rate -/
theorem clothing_factory_production_adjustment 
  (total_pieces : ℕ) 
  (original_rate : ℕ) 
  (days_earlier : ℕ) 
  (x : ℝ) 
  (h1 : total_pieces = 720)
  (h2 : original_rate = 48)
  (h3 : days_earlier = 5) :
  (total_pieces : ℝ) / original_rate - total_pieces / (x + original_rate) = days_earlier :=
by sorry

end NUMINAMATH_CALUDE_clothing_factory_production_adjustment_l1937_193743


namespace NUMINAMATH_CALUDE_cary_earnings_l1937_193796

/-- Calculates the total net earnings over three years for an employee named Cary --/
def total_net_earnings (initial_wage : ℚ) : ℚ :=
  let year1_base_wage := initial_wage
  let year1_hours := 40 * 50
  let year1_gross := year1_hours * year1_base_wage + 500
  let year1_net := year1_gross * (1 - 0.2)

  let year2_base_wage := year1_base_wage * 1.2 * 0.75
  let year2_regular_hours := 40 * 51
  let year2_overtime_hours := 10 * 51
  let year2_gross := year2_regular_hours * year2_base_wage + 
                     year2_overtime_hours * (year2_base_wage * 1.5) - 300
  let year2_net := year2_gross * (1 - 0.22)

  let year3_base_wage := year2_base_wage * 1.1
  let year3_hours := 40 * 50
  let year3_gross := year3_hours * year3_base_wage + 1000
  let year3_net := year3_gross * (1 - 0.18)

  year1_net + year2_net + year3_net

/-- Theorem stating that Cary's total net earnings over three years equals $52,913.10 --/
theorem cary_earnings : total_net_earnings 10 = 52913.1 := by
  sorry

end NUMINAMATH_CALUDE_cary_earnings_l1937_193796


namespace NUMINAMATH_CALUDE_unique_solutions_l1937_193752

/-- A triple of strictly positive integers (a, b, p) satisfies the equation if a^p = b! + p and p is prime. -/
def SatisfiesEquation (a b p : ℕ+) : Prop :=
  a ^ p.val = Nat.factorial b.val + p.val ∧ Nat.Prime p.val

theorem unique_solutions :
  ∀ a b p : ℕ+, SatisfiesEquation a b p →
    ((a = 2 ∧ b = 2 ∧ p = 2) ∨ (a = 3 ∧ b = 4 ∧ p = 3)) :=
by sorry

end NUMINAMATH_CALUDE_unique_solutions_l1937_193752


namespace NUMINAMATH_CALUDE_alice_bob_meet_l1937_193774

/-- The number of points on the circle -/
def n : ℕ := 24

/-- Alice's starting position -/
def alice_start : ℕ := 1

/-- Bob's starting position -/
def bob_start : ℕ := 12

/-- Alice's movement per turn (clockwise) -/
def alice_move : ℕ := 7

/-- Bob's movement per turn (counterclockwise) -/
def bob_move : ℕ := 17

/-- The number of turns it takes for Alice and Bob to meet -/
def meeting_turns : ℕ := 5

/-- Theorem stating that Alice and Bob meet after the specified number of turns -/
theorem alice_bob_meet :
  (alice_start + meeting_turns * alice_move) % n = 
  (bob_start - meeting_turns * bob_move + n * meeting_turns) % n :=
sorry

end NUMINAMATH_CALUDE_alice_bob_meet_l1937_193774


namespace NUMINAMATH_CALUDE_corys_initial_money_l1937_193771

/-- The problem of determining Cory's initial amount of money -/
theorem corys_initial_money (cost_per_pack : ℝ) (additional_needed : ℝ) : 
  cost_per_pack = 49 → additional_needed = 78 → 
  2 * cost_per_pack - additional_needed = 20 := by
  sorry

end NUMINAMATH_CALUDE_corys_initial_money_l1937_193771
