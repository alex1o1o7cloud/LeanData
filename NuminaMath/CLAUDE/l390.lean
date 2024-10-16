import Mathlib

namespace NUMINAMATH_CALUDE_perfect_cube_prime_factor_addition_l390_39069

theorem perfect_cube_prime_factor_addition (x : ℕ) : ∃ x, 
  (27 = 3^3) ∧ 
  (∃ p : ℕ, Prime p ∧ p = 3 + x) ∧ 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_perfect_cube_prime_factor_addition_l390_39069


namespace NUMINAMATH_CALUDE_sum_of_seventh_powers_l390_39093

theorem sum_of_seventh_powers (α β γ : ℂ) 
  (h1 : α + β + γ = 2)
  (h2 : α^2 + β^2 + γ^2 = 6)
  (h3 : α^3 + β^3 + γ^3 = 14) :
  α^7 + β^7 + γ^7 = -98 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seventh_powers_l390_39093


namespace NUMINAMATH_CALUDE_last_amoeba_is_B_l390_39053

/-- Represents the type of a Martian amoeba -/
inductive AmoebType
  | A
  | B
  | C

/-- Represents the state of the amoeba population -/
structure AmoebState where
  countA : ℕ
  countB : ℕ
  countC : ℕ

/-- Defines the initial state of amoebas -/
def initialState : AmoebState :=
  { countA := 20, countB := 21, countC := 22 }

/-- Defines the merger rule for amoebas -/
def merge (a b : AmoebType) : AmoebType :=
  match a, b with
  | AmoebType.A, AmoebType.B => AmoebType.C
  | AmoebType.B, AmoebType.C => AmoebType.A
  | AmoebType.C, AmoebType.A => AmoebType.B
  | _, _ => a  -- This case should not occur in valid mergers

/-- Theorem: The last remaining amoeba is of type B -/
theorem last_amoeba_is_B (final : AmoebState) 
    (h_final : final.countA + final.countB + final.countC = 1) :
    ∃ (n : ℕ), n > 0 ∧ final = { countA := 0, countB := n, countC := 0 } :=
  sorry

#check last_amoeba_is_B

end NUMINAMATH_CALUDE_last_amoeba_is_B_l390_39053


namespace NUMINAMATH_CALUDE_consecutive_odd_product_l390_39035

theorem consecutive_odd_product (m : ℕ) (N : ℤ) : 
  Odd N → 
  N = (m - 1) * m * (m + 1) - ((m - 1) + m + (m + 1)) → 
  (∃ k : ℕ, m = 2 * k + 1) ∧ 
  N = (m - 2) * m * (m + 2) ∧ 
  Odd (m - 2) ∧ Odd m ∧ Odd (m + 2) :=
sorry

end NUMINAMATH_CALUDE_consecutive_odd_product_l390_39035


namespace NUMINAMATH_CALUDE_square_root_81_l390_39076

theorem square_root_81 : ∀ (x : ℝ), x^2 = 81 ↔ x = 9 ∨ x = -9 := by sorry

end NUMINAMATH_CALUDE_square_root_81_l390_39076


namespace NUMINAMATH_CALUDE_smallest_n_for_divisibility_by_1991_l390_39048

theorem smallest_n_for_divisibility_by_1991 :
  ∃ (n : ℕ), n > 0 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (1991 ∣ (a + b) ∨ 1991 ∣ (a - b))) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℤ), T.card = m ∧
      ∀ (a b : ℤ), a ∈ T → b ∈ T → a ≠ b → ¬(1991 ∣ (a + b)) ∧ ¬(1991 ∣ (a - b))) ∧
  n = 997 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_divisibility_by_1991_l390_39048


namespace NUMINAMATH_CALUDE_permutation_residue_systems_l390_39023

theorem permutation_residue_systems (n : ℕ) : 
  (∃ p : Fin n → Fin n, Function.Bijective p ∧ 
    (∀ (i : Fin n), ∃ (j : Fin n), (p j + j : ℕ) % n = i) ∧
    (∀ (i : Fin n), ∃ (j : Fin n), (p j - j : ℤ) % n = i)) ↔ 
  (n % 6 = 1 ∨ n % 6 = 5) :=
sorry

end NUMINAMATH_CALUDE_permutation_residue_systems_l390_39023


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l390_39096

theorem polar_to_rectangular_conversion :
  let r : ℝ := 4 * Real.sqrt 2
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = 4 ∧ y = 4 := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l390_39096


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_seventeen_fourths_l390_39075

theorem greatest_integer_less_than_negative_seventeen_fourths :
  ⌊-17/4⌋ = -5 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_seventeen_fourths_l390_39075


namespace NUMINAMATH_CALUDE_area_of_triangle_l390_39066

/-- The hyperbola with equation x^2 - y^2/12 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2/12 = 1}

/-- The foci of the hyperbola -/
def Foci : ℝ × ℝ × ℝ × ℝ := sorry

/-- A point on the hyperbola -/
def P : ℝ × ℝ := sorry

/-- The distance ratio condition -/
axiom distance_ratio : 
  let (f1x, f1y, f2x, f2y) := Foci
  let (px, py) := P
  3 * ((f2x - px)^2 + (f2y - py)^2) = 2 * ((f1x - px)^2 + (f1y - py)^2)

/-- P is on the hyperbola -/
axiom P_on_hyperbola : P ∈ Hyperbola

/-- The theorem to be proved -/
theorem area_of_triangle : 
  let (f1x, f1y, f2x, f2y) := Foci
  let (px, py) := P
  (1/2) * |f1x - f2x| * |f1y - f2y| = 12 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_l390_39066


namespace NUMINAMATH_CALUDE_initial_men_count_l390_39094

theorem initial_men_count (initial_days : ℝ) (additional_men : ℕ) (final_days : ℝ) :
  initial_days = 18 →
  additional_men = 450 →
  final_days = 13.090909090909092 →
  ∃ (initial_men : ℕ), 
    initial_men * initial_days = (initial_men + additional_men) * final_days ∧
    initial_men = 1200 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l390_39094


namespace NUMINAMATH_CALUDE_distance_between_foci_l390_39072

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 3)^2) + Real.sqrt ((x + 6)^2 + (y - 5)^2) = 26

-- Define the foci
def focus1 : ℝ × ℝ := (4, 3)
def focus2 : ℝ × ℝ := (-6, 5)

-- Theorem statement
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 2 * Real.sqrt 26 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_foci_l390_39072


namespace NUMINAMATH_CALUDE_y_over_z_equals_negative_five_l390_39010

theorem y_over_z_equals_negative_five (x y z : ℝ) 
  (eq1 : x + y = 2 * x + z)
  (eq2 : x - 2 * y = 4 * z)
  (eq3 : x + y + z = 21) :
  y / z = -5 := by
sorry

end NUMINAMATH_CALUDE_y_over_z_equals_negative_five_l390_39010


namespace NUMINAMATH_CALUDE_sum_of_digits_after_addition_l390_39077

def sum_of_digits (n : ℕ) : ℕ := sorry

def number_of_carries (a b : ℕ) : ℕ := sorry

theorem sum_of_digits_after_addition (A B : ℕ) 
  (hA : A > 0) 
  (hB : B > 0) 
  (hSumA : sum_of_digits A = 19) 
  (hSumB : sum_of_digits B = 20) 
  (hCarries : number_of_carries A B = 2) : 
  sum_of_digits (A + B) = 21 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_after_addition_l390_39077


namespace NUMINAMATH_CALUDE_inequality_proof_l390_39055

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*a*c)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l390_39055


namespace NUMINAMATH_CALUDE_expected_red_balls_l390_39034

/-- The number of red balls in the bag -/
def red_balls : ℕ := 4

/-- The number of white balls in the bag -/
def white_balls : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := red_balls + white_balls

/-- The probability of drawing a red ball in a single draw -/
def p_red : ℚ := red_balls / total_balls

/-- The number of draws -/
def num_draws : ℕ := 6

/-- The random variable representing the number of red balls drawn -/
def ξ : ℕ → ℚ := sorry

/-- The expected value of ξ -/
def E_ξ : ℚ := num_draws * p_red

theorem expected_red_balls : E_ξ = 4 := by sorry

end NUMINAMATH_CALUDE_expected_red_balls_l390_39034


namespace NUMINAMATH_CALUDE_consecutive_integers_base_sum_l390_39024

/-- Represents a number in a given base -/
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem consecutive_integers_base_sum (C D : Nat) : 
  C.succ = D →
  C < D →
  to_base_10 [2, 3, 1] C + to_base_10 [5, 6] D = to_base_10 [1, 0, 5] (C + D) →
  C + D = 7 := by sorry

end NUMINAMATH_CALUDE_consecutive_integers_base_sum_l390_39024


namespace NUMINAMATH_CALUDE_percentage_equality_l390_39036

theorem percentage_equality (x y : ℝ) (P : ℝ) :
  (P / 100) * (x - y) = (20 / 100) * (x + y) →
  y = (50 / 100) * x →
  P = 60 := by
sorry

end NUMINAMATH_CALUDE_percentage_equality_l390_39036


namespace NUMINAMATH_CALUDE_new_average_after_dropout_l390_39043

theorem new_average_after_dropout (initial_students : ℕ) (initial_average : ℚ) 
  (dropout_score : ℕ) (new_students : ℕ) (h1 : initial_students = 16) 
  (h2 : initial_average = 61.5) (h3 : dropout_score = 24) (h4 : new_students = initial_students - 1) :
  let total_score := initial_students * initial_average
  let new_total_score := total_score - dropout_score
  let new_average := new_total_score / new_students
  new_average = 64 := by sorry

end NUMINAMATH_CALUDE_new_average_after_dropout_l390_39043


namespace NUMINAMATH_CALUDE_shifted_line_equation_l390_39064

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  yIntercept : ℝ

/-- Shifts a linear function horizontally and vertically -/
def shiftLinearFunction (f : LinearFunction) (horizontalShift : ℝ) (verticalShift : ℝ) : LinearFunction :=
  { slope := f.slope
    yIntercept := f.slope * (-horizontalShift) + f.yIntercept + verticalShift }

theorem shifted_line_equation (f : LinearFunction) :
  let f' := shiftLinearFunction f 2 3
  f.slope = 2 ∧ f.yIntercept = -3 → f'.slope = 2 ∧ f'.yIntercept = 4 := by
  sorry

#check shifted_line_equation

end NUMINAMATH_CALUDE_shifted_line_equation_l390_39064


namespace NUMINAMATH_CALUDE_first_grade_enrollment_l390_39018

theorem first_grade_enrollment (a : ℕ) : 
  (200 ≤ a ∧ a ≤ 300) →
  (∃ R : ℕ, a = 25 * R + 10) →
  (∃ L : ℕ, a = 30 * L - 15) →
  a = 285 := by
sorry

end NUMINAMATH_CALUDE_first_grade_enrollment_l390_39018


namespace NUMINAMATH_CALUDE_curve_is_ellipse_iff_k_in_range_l390_39003

/-- The curve equation: x^2 / (4 + k) + y^2 / (1 - k) = 1 -/
def curve_equation (x y k : ℝ) : Prop :=
  x^2 / (4 + k) + y^2 / (1 - k) = 1

/-- The range of k values for which the curve represents an ellipse -/
def ellipse_k_range (k : ℝ) : Prop :=
  (k > -4 ∧ k < -3/2) ∨ (k > -3/2 ∧ k < 1)

/-- Theorem stating that the curve represents an ellipse if and only if k is in the specified range -/
theorem curve_is_ellipse_iff_k_in_range :
  ∀ k : ℝ, (∃ x y : ℝ, curve_equation x y k) ↔ ellipse_k_range k :=
sorry

end NUMINAMATH_CALUDE_curve_is_ellipse_iff_k_in_range_l390_39003


namespace NUMINAMATH_CALUDE_mans_rowing_rate_l390_39062

/-- Proves that a man's rowing rate in still water is 11 km/h given his speeds with and against the stream. -/
theorem mans_rowing_rate (with_stream : ℝ) (against_stream : ℝ)
  (h_with : with_stream = 18)
  (h_against : against_stream = 4) :
  (with_stream + against_stream) / 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_mans_rowing_rate_l390_39062


namespace NUMINAMATH_CALUDE_product_of_numbers_l390_39078

theorem product_of_numbers (x y : ℚ) : 
  (- x = 3 / 4) → (y = x - 1 / 2) → (x * y = 15 / 16) := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l390_39078


namespace NUMINAMATH_CALUDE_basketball_court_equation_rewrite_l390_39070

theorem basketball_court_equation_rewrite :
  ∃ (a b c : ℤ), a > 0 ∧
  (∀ x : ℝ, 16 * x^2 + 32 * x - 40 = 0 ↔ (a * x + b)^2 = c) ∧
  a + b + c = 64 := by
  sorry

end NUMINAMATH_CALUDE_basketball_court_equation_rewrite_l390_39070


namespace NUMINAMATH_CALUDE_range_of_a_for_solution_a_value_for_minimum_l390_39050

-- Define the function f
def f (a x : ℝ) : ℝ := |2*x - a| + |x - 1|

-- Part 1
theorem range_of_a_for_solution (a : ℝ) :
  (∃ x, f a x ≤ 2 - |x - 1|) ↔ 0 ≤ a ∧ a ≤ 4 :=
sorry

-- Part 2
theorem a_value_for_minimum (a : ℝ) :
  a < 2 → (∀ x, f a x ≥ 3) → (∃ x, f a x = 3) → a = -4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_solution_a_value_for_minimum_l390_39050


namespace NUMINAMATH_CALUDE_parabola_focus_l390_39058

/-- A parabola is defined by its equation relating x and y coordinates -/
structure Parabola where
  equation : ℝ → ℝ

/-- The focus of a parabola is a point (x, y) -/
structure Focus where
  x : ℝ
  y : ℝ

/-- Predicate to check if a given point is the focus of a parabola -/
def is_focus (p : Parabola) (f : Focus) : Prop :=
  ∀ (y : ℝ), 
    let x := p.equation y
    (x - f.x)^2 + y^2 = (x - (f.x - 3))^2

/-- Theorem stating that (-3, 0) is the focus of the parabola x = -1/12 * y^2 -/
theorem parabola_focus :
  let p : Parabola := ⟨λ y => -1/12 * y^2⟩
  let f : Focus := ⟨-3, 0⟩
  is_focus p f := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l390_39058


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l390_39046

/-- A polynomial with integer coefficients -/
def IntPolynomial : Type := ℕ → ℤ

/-- Evaluate a polynomial at a given integer -/
def evaluate (f : IntPolynomial) (x : ℤ) : ℤ :=
  sorry

/-- A number is divisible by another if their remainder is zero -/
def divisible (a b : ℤ) : Prop := a % b = 0

theorem polynomial_divisibility (f : IntPolynomial) :
  divisible (evaluate f 2) 6 →
  divisible (evaluate f 3) 6 →
  divisible (evaluate f 5) 6 :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l390_39046


namespace NUMINAMATH_CALUDE_sophie_donuts_l390_39033

/-- The number of donuts left for Sophie after giving some away -/
def donuts_left (total_boxes : ℕ) (donuts_per_box : ℕ) (boxes_given : ℕ) (donuts_given : ℕ) : ℕ :=
  (total_boxes - boxes_given) * donuts_per_box - donuts_given

/-- Theorem stating that Sophie is left with 30 donuts -/
theorem sophie_donuts :
  donuts_left 4 12 1 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sophie_donuts_l390_39033


namespace NUMINAMATH_CALUDE_count_students_in_line_l390_39012

/-- The number of students in a line formation -/
def students_in_line (between : ℕ) : ℕ :=
  between + 2

/-- Theorem: Given 14 people between Yoojung and Eunji, there are 16 students in line -/
theorem count_students_in_line :
  students_in_line 14 = 16 := by
  sorry

end NUMINAMATH_CALUDE_count_students_in_line_l390_39012


namespace NUMINAMATH_CALUDE_power_division_equality_l390_39086

theorem power_division_equality : (4 ^ (3^2)) / ((4^3)^2) = 64 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l390_39086


namespace NUMINAMATH_CALUDE_no_nonzero_solutions_l390_39065

theorem no_nonzero_solutions (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (Real.sqrt (a^2 + b^2) = 0 ↔ a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = (a + b) / 2 ↔ a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = Real.sqrt a + Real.sqrt b ↔ a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = a + b - 1 ↔ a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_nonzero_solutions_l390_39065


namespace NUMINAMATH_CALUDE_distinct_digit_count_is_5040_l390_39068

/-- The number of four-digit integers with distinct digits, including those starting with 0 -/
def distinctDigitCount : ℕ := 10 * 9 * 8 * 7

/-- Theorem stating that the count of four-digit integers with distinct digits is 5040 -/
theorem distinct_digit_count_is_5040 : distinctDigitCount = 5040 := by
  sorry

end NUMINAMATH_CALUDE_distinct_digit_count_is_5040_l390_39068


namespace NUMINAMATH_CALUDE_programmers_remote_work_cycle_l390_39040

def alex_cycle : ℕ := 5
def brooke_cycle : ℕ := 3
def charlie_cycle : ℕ := 8
def dana_cycle : ℕ := 9

theorem programmers_remote_work_cycle : 
  Nat.lcm alex_cycle (Nat.lcm brooke_cycle (Nat.lcm charlie_cycle dana_cycle)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_programmers_remote_work_cycle_l390_39040


namespace NUMINAMATH_CALUDE_actual_distance_scientific_notation_l390_39087

/-- The scale of the map -/
def map_scale : ℚ := 1 / 8000000

/-- The distance between A and B on the map in centimeters -/
def map_distance : ℚ := 3.5

/-- The actual distance between A and B in centimeters -/
def actual_distance : ℕ := 28000000

/-- Theorem stating that the actual distance is equal to 2.8 × 10^7 -/
theorem actual_distance_scientific_notation : 
  (actual_distance : ℝ) = 2.8 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_scientific_notation_l390_39087


namespace NUMINAMATH_CALUDE_system_solution_implies_a_minus_b_l390_39007

theorem system_solution_implies_a_minus_b (a b : ℤ) : 
  (a * (-2) + b * 1 = 1) → 
  (b * (-2) + a * 1 = 7) → 
  (a - b = 2) := by
sorry

end NUMINAMATH_CALUDE_system_solution_implies_a_minus_b_l390_39007


namespace NUMINAMATH_CALUDE_solution_to_polynomial_equation_l390_39030

theorem solution_to_polynomial_equation : ∃ x : ℤ, x^5 - 101*x^3 - 999*x^2 + 100900 = 0 :=
by
  use 10
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_solution_to_polynomial_equation_l390_39030


namespace NUMINAMATH_CALUDE_debbys_share_percentage_l390_39091

theorem debbys_share_percentage (total : ℝ) (maggies_share : ℝ) 
  (h1 : total = 6000)
  (h2 : maggies_share = 4500) :
  (total - maggies_share) / total * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_debbys_share_percentage_l390_39091


namespace NUMINAMATH_CALUDE_a_value_l390_39049

def U : Set ℤ := {3, 4, 5}

def M (a : ℤ) : Set ℤ := {|a - 3|, 3}

theorem a_value (a : ℤ) (h : (U \ M a) = {5}) : a = -1 ∨ a = 7 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l390_39049


namespace NUMINAMATH_CALUDE_decreasing_linear_function_negative_slope_l390_39047

/-- A linear function y = kx - 5 where y decreases as x increases -/
def decreasing_linear_function (k : ℝ) : ℝ → ℝ := λ x ↦ k * x - 5

/-- Theorem: If y decreases as x increases in a linear function y = kx - 5, then k < 0 -/
theorem decreasing_linear_function_negative_slope (k : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → decreasing_linear_function k x₁ > decreasing_linear_function k x₂) →
  k < 0 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_negative_slope_l390_39047


namespace NUMINAMATH_CALUDE_chicken_difference_l390_39080

/-- The number of chickens in the coop -/
def coop_chickens : ℕ := 14

/-- The number of chickens in the run -/
def run_chickens : ℕ := 2 * coop_chickens

/-- The number of chickens free ranging -/
def free_ranging_chickens : ℕ := 52

/-- The difference between double the number of chickens in the run and the number of chickens free ranging -/
theorem chicken_difference : 2 * run_chickens - free_ranging_chickens = 4 := by
  sorry

end NUMINAMATH_CALUDE_chicken_difference_l390_39080


namespace NUMINAMATH_CALUDE_largest_prime_factor_is_101_l390_39060

/-- A sequence of four-digit integers with a cyclic digit property -/
def CyclicSequence := List Nat

/-- The sum of all terms in a cyclic sequence -/
def sequenceSum (seq : CyclicSequence) : Nat :=
  seq.sum

/-- Predicate to check if a sequence satisfies the cyclic digit property -/
def hasCyclicDigitProperty (seq : CyclicSequence) : Prop :=
  sorry -- Definition of the cyclic digit property

/-- The largest prime factor that always divides the sum of a cyclic sequence -/
def largestPrimeFactor (seq : CyclicSequence) : Nat :=
  sorry -- Definition to find the largest prime factor

theorem largest_prime_factor_is_101 (seq : CyclicSequence) 
    (h : hasCyclicDigitProperty seq) :
    largestPrimeFactor seq = 101 := by
  sorry

#check largest_prime_factor_is_101

end NUMINAMATH_CALUDE_largest_prime_factor_is_101_l390_39060


namespace NUMINAMATH_CALUDE_complex_product_quadrant_l390_39063

theorem complex_product_quadrant : 
  let z₁ : ℂ := 1 - 2*I
  let z₂ : ℂ := 2 + I
  let product := z₁ * z₂
  (product.re > 0 ∧ product.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_product_quadrant_l390_39063


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l390_39071

theorem smallest_b_in_arithmetic_sequence (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- All terms are positive
  ∃ (d : ℝ), a = b - d ∧ c = b + d →  -- Terms form an arithmetic sequence
  a * b * c = 125 →  -- Product is 125
  ∀ x : ℝ, (x > 0 ∧ 
    (∃ (y z d : ℝ), y > 0 ∧ z > 0 ∧ 
      y = x - d ∧ z = x + d ∧ 
      y * x * z = 125)) → 
    x ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l390_39071


namespace NUMINAMATH_CALUDE_product_of_numbers_l390_39013

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 16) (h2 : x^2 + y^2 = 200) : x * y = 28 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l390_39013


namespace NUMINAMATH_CALUDE_autumn_sales_one_million_l390_39092

/-- Ice cream sales data --/
structure IceCreamSales where
  spring : ℝ
  summer : ℝ
  winter : ℝ
  autumn : ℝ

/-- Calculate total sales --/
def total_sales (sales : IceCreamSales) : ℝ :=
  sales.spring + sales.summer + sales.winter + sales.autumn

/-- Theorem: Autumn ice cream sales are 1 million units --/
theorem autumn_sales_one_million :
  ∀ (sales : IceCreamSales),
  sales.spring = 0.2 * total_sales sales →
  sales.summer = 6 →
  sales.winter = 5 →
  sales.autumn = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_autumn_sales_one_million_l390_39092


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l390_39044

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - 2*a^2 - a + 2 = 0) → 
  (b^3 - 2*b^2 - b + 2 = 0) → 
  (c^3 - 2*c^2 - c + 2 = 0) → 
  (1/(a+2) + 1/(b+2) + 1/(c+2) = 3/2) := by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l390_39044


namespace NUMINAMATH_CALUDE_james_beef_purchase_l390_39041

/-- Proves that James bought 20 pounds of beef given the problem conditions -/
theorem james_beef_purchase :
  ∀ (beef pork : ℝ) (meals : ℕ),
    pork = beef / 2 →
    meals * 1.5 = beef + pork →
    meals * 20 = 400 →
    beef = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_james_beef_purchase_l390_39041


namespace NUMINAMATH_CALUDE_oil_bill_ratio_l390_39074

/-- The oil bill problem -/
theorem oil_bill_ratio (january_bill : ℝ) (february_bill : ℝ) : 
  january_bill = 119.99999999999994 →
  february_bill / january_bill = 3 / 2 →
  (february_bill + 20) / january_bill = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_oil_bill_ratio_l390_39074


namespace NUMINAMATH_CALUDE_batter_distribution_l390_39029

/-- Given two trays of batter where the second tray holds 20 cups less than the first,
    and the total amount is 500 cups, prove that the second tray holds 240 cups. -/
theorem batter_distribution (first_tray second_tray : ℕ) : 
  first_tray = second_tray + 20 →
  first_tray + second_tray = 500 →
  second_tray = 240 := by
sorry

end NUMINAMATH_CALUDE_batter_distribution_l390_39029


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l390_39054

/-- A geometric sequence with sum of first n terms S_n -/
structure GeometricSequence where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms

/-- Given conditions for the geometric sequence -/
def given_sequence : GeometricSequence where
  S := fun n => 
    if n = 2 then 6
    else if n = 4 then 18
    else 0  -- We only know S_2 and S_4, other values are placeholders

theorem geometric_sequence_sum (seq : GeometricSequence) :
  seq.S 2 = 6 → seq.S 4 = 18 → seq.S 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l390_39054


namespace NUMINAMATH_CALUDE_sum_of_four_digit_numbers_l390_39028

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- A four-digit number formed from the given digits -/
structure FourDigitNumber where
  d₁ : Nat
  d₂ : Nat
  d₃ : Nat
  d₄ : Nat
  h₁ : d₁ ∈ digits
  h₂ : d₂ ∈ digits
  h₃ : d₃ ∈ digits
  h₄ : d₄ ∈ digits
  distinct : d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄

/-- The value of a four-digit number -/
def value (n : FourDigitNumber) : Nat :=
  1000 * n.d₁ + 100 * n.d₂ + 10 * n.d₃ + n.d₄

/-- The set of all valid four-digit numbers -/
def allFourDigitNumbers : Finset FourDigitNumber :=
  sorry

theorem sum_of_four_digit_numbers :
  (allFourDigitNumbers.sum value) = 399960 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_digit_numbers_l390_39028


namespace NUMINAMATH_CALUDE_fraction_value_implies_m_l390_39032

theorem fraction_value_implies_m (m : ℚ) : (m - 5) / m = 2 → m = -5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_implies_m_l390_39032


namespace NUMINAMATH_CALUDE_decimal_sum_and_product_l390_39095

theorem decimal_sum_and_product :
  let sum := 0.5 + 0.03 + 0.007
  sum = 0.537 ∧ 3 * sum = 1.611 :=
by sorry

end NUMINAMATH_CALUDE_decimal_sum_and_product_l390_39095


namespace NUMINAMATH_CALUDE_inequality_relationship_l390_39001

theorem inequality_relationship (x : ℝ) : 
  (∀ x, x - 2 > 0 → (x - 2) * (x - 1) > 0) ∧ 
  (∃ x, (x - 2) * (x - 1) > 0 ∧ ¬(x - 2 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_relationship_l390_39001


namespace NUMINAMATH_CALUDE_cross_section_area_is_18_l390_39083

/-- Regular triangular pyramid with given dimensions -/
structure RegularTriangularPyramid where
  base_side : ℝ
  height : ℝ

/-- Cross-section of a regular triangular pyramid -/
structure CrossSection where
  pyramid : RegularTriangularPyramid
  -- Assuming the cross-section passes through the midline and is perpendicular to the base

/-- The area of the cross-section -/
def cross_section_area (cs : CrossSection) : ℝ :=
  sorry

/-- Theorem stating the area of the cross-section for the given dimensions -/
theorem cross_section_area_is_18 (cs : CrossSection) 
  (h1 : cs.pyramid.base_side = 8) 
  (h2 : cs.pyramid.height = 12) : 
  cross_section_area cs = 18 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_area_is_18_l390_39083


namespace NUMINAMATH_CALUDE_triangle_properties_l390_39099

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (4 * a = Real.sqrt 5 * c) →
  (Real.cos C = 3 / 5) →
  (b = 11) →
  (Real.sin A = Real.sqrt 5 / 5) ∧
  (1 / 2 * a * b * Real.sin C = 22) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l390_39099


namespace NUMINAMATH_CALUDE_x_value_for_given_z_and_w_l390_39014

/-- Given that x is directly proportional to y³, and y is directly proportional to √z and w,
    prove that x = 540√3 when z = 36 and w = 2, given that x = 5 when z = 8 and w = 1. -/
theorem x_value_for_given_z_and_w (x y z w c k : ℝ) 
    (h1 : ∃ k, ∀ y, x = k * y^3)
    (h2 : ∃ c, y = c * Real.sqrt z * w)
    (h3 : x = 5 ∧ z = 8 ∧ w = 1) :
    z = 36 ∧ w = 2 → x = 540 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_x_value_for_given_z_and_w_l390_39014


namespace NUMINAMATH_CALUDE_triangle_side_length_l390_39002

theorem triangle_side_length (BC AC : ℝ) (A : ℝ) :
  BC = Real.sqrt 7 →
  AC = 2 * Real.sqrt 3 →
  A = π / 6 →
  ∃ AB : ℝ, (AB = 5 ∨ AB = 1) ∧
    AB^2 + AC^2 - BC^2 = 2 * AB * AC * Real.cos A :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l390_39002


namespace NUMINAMATH_CALUDE_nested_square_roots_equality_l390_39059

theorem nested_square_roots_equality : Real.sqrt (36 * Real.sqrt (27 * Real.sqrt 9)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_roots_equality_l390_39059


namespace NUMINAMATH_CALUDE_infinitely_many_special_triangles_l390_39038

/-- A triangle with integer area formed by square roots of distinct non-square integers -/
structure SpecialTriangle where
  a₁ : ℕ+
  a₂ : ℕ+
  a₃ : ℕ+
  distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₂ ≠ a₃
  not_squares : ¬∃ m : ℕ, a₁ = m^2 ∧ ¬∃ n : ℕ, a₂ = n^2 ∧ ¬∃ k : ℕ, a₃ = k^2
  triangle_inequality : Real.sqrt a₁.val + Real.sqrt a₂.val > Real.sqrt a₃.val ∧
                        Real.sqrt a₁.val + Real.sqrt a₃.val > Real.sqrt a₂.val ∧
                        Real.sqrt a₂.val + Real.sqrt a₃.val > Real.sqrt a₁.val
  integer_area : ∃ S : ℕ, 16 * S^2 = (a₁ + a₂ + a₃)^2 - 2 * (a₁^2 + a₂^2 + a₃^2)

/-- There exist infinitely many SpecialTriangles -/
theorem infinitely_many_special_triangles : 
  ∀ n : ℕ, ∃ (triangles : Fin n → SpecialTriangle), 
    ∀ i j : Fin n, i ≠ j → 
      ¬∃ (k : ℚ), (k * (triangles i).a₁ : ℚ) = (triangles j).a₁ ∧ 
                   (k * (triangles i).a₂ : ℚ) = (triangles j).a₂ ∧ 
                   (k * (triangles i).a₃ : ℚ) = (triangles j).a₃ :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_special_triangles_l390_39038


namespace NUMINAMATH_CALUDE_ab_length_approx_l390_39015

/-- Represents a right triangle with specific properties -/
structure RightTriangle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  ca : ℝ
  -- Angles in radians
  angle_a : ℝ
  angle_b : ℝ
  angle_c : ℝ
  -- Properties
  right_angle : angle_b = π / 2
  angle_sum : angle_a + angle_b + angle_c = π
  bc_length : bc = 12
  angle_a_value : angle_a = π / 6  -- 30 degrees in radians

/-- Theorem stating the approximate length of AB in the specific right triangle -/
theorem ab_length_approx (t : RightTriangle) : 
  ∃ ε > 0, |t.ab - 20.8| < ε ∧ ε < 0.1 :=
sorry

end NUMINAMATH_CALUDE_ab_length_approx_l390_39015


namespace NUMINAMATH_CALUDE_living_space_increase_l390_39061

/-- Proves that the average annual increase in living space needed is approximately 12.05 ten thousand m² --/
theorem living_space_increase (initial_population : ℝ) (initial_space_per_person : ℝ)
  (target_space_per_person : ℝ) (growth_rate : ℝ) (years : ℕ)
  (h1 : initial_population = 20) -- in ten thousands
  (h2 : initial_space_per_person = 8)
  (h3 : target_space_per_person = 10)
  (h4 : growth_rate = 0.01)
  (h5 : years = 4) :
  ∃ x : ℝ, abs (x - 12.05) < 0.01 ∧ 
  x * years = target_space_per_person * (initial_population * (1 + growth_rate) ^ years) - 
              initial_space_per_person * initial_population :=
by sorry


end NUMINAMATH_CALUDE_living_space_increase_l390_39061


namespace NUMINAMATH_CALUDE_total_eyes_in_pond_l390_39005

/-- The number of eyes an animal has -/
def eyes_per_animal : ℕ := 2

/-- The number of frogs in the pond -/
def num_frogs : ℕ := 20

/-- The number of crocodiles in the pond -/
def num_crocodiles : ℕ := 6

/-- The total number of animals in the pond -/
def total_animals : ℕ := num_frogs + num_crocodiles

/-- Theorem: The total number of animal eyes in the pond is 52 -/
theorem total_eyes_in_pond : num_frogs * eyes_per_animal + num_crocodiles * eyes_per_animal = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_eyes_in_pond_l390_39005


namespace NUMINAMATH_CALUDE_inequality_proof_l390_39052

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 1) :
  a / (b^2 * (c + 1)) + b / (c^2 * (a + 1)) + c / (a^2 * (b + 1)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l390_39052


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l390_39090

theorem greatest_power_of_two_factor (n : ℕ) : 
  (∃ k : ℕ, 12^600 - 8^400 = 2^1204 * k ∧ k % 2 ≠ 0) ∧
  (∀ m : ℕ, m > 1204 → ¬(∃ l : ℕ, 12^600 - 8^400 = 2^m * l)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l390_39090


namespace NUMINAMATH_CALUDE_larger_ssr_not_better_fit_l390_39084

/-- Represents a simple linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ
  x : List ℝ
  y : List ℝ

/-- Calculates the sum of squared residuals for a given model -/
def sumSquaredResiduals (model : LinearRegression) : ℝ :=
  sorry

/-- Represents the goodness of fit of a model -/
def goodnessOfFit (model : LinearRegression) : ℝ :=
  sorry

theorem larger_ssr_not_better_fit (model1 model2 : LinearRegression) :
  sumSquaredResiduals model1 > sumSquaredResiduals model2 →
  goodnessOfFit model1 ≤ goodnessOfFit model2 :=
sorry

end NUMINAMATH_CALUDE_larger_ssr_not_better_fit_l390_39084


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l390_39081

/-- The trajectory of point P given the conditions in the problem -/
def trajectory (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

theorem trajectory_is_ellipse (x y : ℝ) :
  let P : ℝ × ℝ := (x, y)
  let M : ℝ × ℝ := (1, 0)
  let d : ℝ := |x - 2|
  (‖P - M‖ : ℝ) / d = Real.sqrt 2 / 2 →
  trajectory x y :=
by sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l390_39081


namespace NUMINAMATH_CALUDE_base_r_is_seven_l390_39022

/-- Represents a number in base r --/
def BaseR (n : ℕ) (r : ℕ) : ℕ → ℕ
| 0 => 0
| (k+1) => (n % r) * r^k + BaseR (n / r) r k

/-- The equation representing the transaction in base r --/
def TransactionEquation (r : ℕ) : Prop :=
  BaseR 210 r 2 + BaseR 260 r 2 = BaseR 500 r 2

theorem base_r_is_seven :
  ∃ r : ℕ, r > 1 ∧ TransactionEquation r ∧ r = 7 := by
  sorry

end NUMINAMATH_CALUDE_base_r_is_seven_l390_39022


namespace NUMINAMATH_CALUDE_third_month_sale_l390_39025

/-- Proves that the sale in the third month is 6855 given the conditions of the problem -/
theorem third_month_sale (sales : Fin 6 → ℕ) : 
  (sales 0 = 6335) → 
  (sales 1 = 6927) → 
  (sales 3 = 7230) → 
  (sales 4 = 6562) → 
  (sales 5 = 5091) → 
  ((sales 0 + sales 1 + sales 2 + sales 3 + sales 4 + sales 5) / 6 = 6500) → 
  sales 2 = 6855 := by
sorry

end NUMINAMATH_CALUDE_third_month_sale_l390_39025


namespace NUMINAMATH_CALUDE_students_in_all_events_l390_39037

theorem students_in_all_events 
  (total_students : ℕ) 
  (event_A_participants : ℕ) 
  (event_B_participants : ℕ) 
  (h1 : total_students = 45)
  (h2 : event_A_participants = 39)
  (h3 : event_B_participants = 28)
  (h4 : event_A_participants + event_B_participants - total_students ≤ event_A_participants)
  (h5 : event_A_participants + event_B_participants - total_students ≤ event_B_participants) :
  event_A_participants + event_B_participants - total_students = 22 := by
  sorry

end NUMINAMATH_CALUDE_students_in_all_events_l390_39037


namespace NUMINAMATH_CALUDE_flag_arrangement_problem_l390_39042

/-- Number of blue flags -/
def blue_flags : ℕ := 10

/-- Number of green flags -/
def green_flags : ℕ := 9

/-- Total number of flags -/
def total_flags : ℕ := blue_flags + green_flags

/-- Number of flagpoles -/
def flagpoles : ℕ := 2

/-- Function to calculate the number of arrangements -/
def calculate_arrangements (a b : ℕ) : ℕ :=
  (a + 1) * Nat.choose (a + 2) b - 2 * Nat.choose (a + 1) b

/-- Theorem stating the result of the flag arrangement problem -/
theorem flag_arrangement_problem :
  calculate_arrangements blue_flags green_flags % 1000 = 310 := by
  sorry

end NUMINAMATH_CALUDE_flag_arrangement_problem_l390_39042


namespace NUMINAMATH_CALUDE_polynomial_factorization_l390_39057

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) =
  (x^2 + 6*x + 7) * (x^2 + 6*x + 8) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l390_39057


namespace NUMINAMATH_CALUDE_stating_safe_zone_condition_l390_39088

/-- Represents the fuse burning speed in cm/s -/
def fuse_speed : ℝ := 0.5

/-- Represents the person's running speed in m/s -/
def person_speed : ℝ := 4

/-- Represents the safe zone distance in meters -/
def safe_distance : ℝ := 150

/-- 
Theorem stating the condition for a person to reach the safe zone before the fuse burns out.
x represents the fuse length in cm.
-/
theorem safe_zone_condition (x : ℝ) :
  (x ≥ 0) →
  (person_speed * (x / fuse_speed) ≥ safe_distance) ↔
  (4 * (x / 0.5) ≥ 150) :=
sorry

end NUMINAMATH_CALUDE_stating_safe_zone_condition_l390_39088


namespace NUMINAMATH_CALUDE_simplify_2A_minus_3B_value_2A_minus_3B_special_case_l390_39017

/-- Given two real numbers a and b, we define A and B as follows: -/
def A (a b : ℝ) : ℝ := 3 * b^2 - 2 * a^2 + 5 * a * b

def B (a b : ℝ) : ℝ := 4 * a * b + 2 * b^2 - a^2

/-- Theorem stating that 2A - 3B simplifies to -a² - 2ab for any real a and b -/
theorem simplify_2A_minus_3B (a b : ℝ) : 2 * A a b - 3 * B a b = -a^2 - 2*a*b := by
  sorry

/-- Theorem stating that when a = -1 and b = 2, the value of 2A - 3B is 3 -/
theorem value_2A_minus_3B_special_case : 2 * A (-1) 2 - 3 * B (-1) 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_2A_minus_3B_value_2A_minus_3B_special_case_l390_39017


namespace NUMINAMATH_CALUDE_function_inequality_l390_39089

-- Define the function f
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem function_inequality (b c : ℝ) 
  (h : ∀ x : ℝ, f b c (-x) = f b c x) : 
  f b c 1 < f b c (-2) ∧ f b c (-2) < f b c 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l390_39089


namespace NUMINAMATH_CALUDE_smallest_num_prime_factors_l390_39008

/-- Given a list of positive integers, returns true if the GCDs of all nonempty subsets are pairwise distinct -/
def has_distinct_gcds (nums : List Nat) : Prop := sorry

/-- Returns the number of prime factors of a natural number -/
def num_prime_factors (n : Nat) : Nat := sorry

theorem smallest_num_prime_factors (N : Nat) (nums : List Nat) 
  (h1 : nums.length = N)
  (h2 : ∀ n ∈ nums, n > 0)
  (h3 : has_distinct_gcds nums) :
  (N = 1 ∧ num_prime_factors (nums.prod) = 0) ∨
  (N ≥ 2 ∧ num_prime_factors (nums.prod) = N) := by sorry

end NUMINAMATH_CALUDE_smallest_num_prime_factors_l390_39008


namespace NUMINAMATH_CALUDE_sequence_a_l390_39011

theorem sequence_a (a : ℕ → ℕ) (h : ∀ n, a (n + 1) = a n + n) :
  a 0 = 19 → a 1 = 20 ∧ a 2 = 22 := by sorry

end NUMINAMATH_CALUDE_sequence_a_l390_39011


namespace NUMINAMATH_CALUDE_johnson_family_seating_l390_39039

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Define the number of sons and daughters
def num_sons : ℕ := 5
def num_daughters : ℕ := 4

-- Define the total number of children
def total_children : ℕ := num_sons + num_daughters

-- Define the function to calculate the number of seating arrangements
def seating_arrangements : ℕ :=
  factorial total_children - (factorial num_sons * factorial num_daughters)

-- Theorem statement
theorem johnson_family_seating :
  seating_arrangements = 360000 :=
sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l390_39039


namespace NUMINAMATH_CALUDE_no_integer_solution_l390_39051

theorem no_integer_solution : ¬ ∃ (m n : ℤ), m^3 = 4*n + 2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l390_39051


namespace NUMINAMATH_CALUDE_sufficient_condition_for_p_l390_39031

-- Define the proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

-- Define what it means for a condition to be sufficient but not necessary
def sufficient_but_not_necessary (condition : ℝ → Prop) (proposition : ℝ → Prop) : Prop :=
  (∃ a : ℝ, condition a ∧ proposition a) ∧
  (∃ a : ℝ, ¬condition a ∧ proposition a)

-- Theorem statement
theorem sufficient_condition_for_p :
  sufficient_but_not_necessary (λ a : ℝ => a = 2) p :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_p_l390_39031


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l390_39021

/-- A trapezoid with given properties -/
structure Trapezoid where
  longer_base : ℝ
  midpoints_distance : ℝ
  shorter_base : ℝ

/-- The theorem stating the relationship between the bases and the midpoints distance -/
theorem trapezoid_shorter_base (t : Trapezoid) 
  (h1 : t.longer_base = 24)
  (h2 : t.midpoints_distance = 4) : 
  t.shorter_base = 16 := by
  sorry

#check trapezoid_shorter_base

end NUMINAMATH_CALUDE_trapezoid_shorter_base_l390_39021


namespace NUMINAMATH_CALUDE_equal_interval_line_segments_l390_39067

/-- Given two line segments with equal interval spacing between points,
    where one segment has 10 points over length a and the other has 100 points over length b,
    prove that b = 11a. -/
theorem equal_interval_line_segments (a b : ℝ) : 
  (∃ (interval : ℝ), 
    a = 9 * interval ∧ 
    b = 99 * interval) → 
  b = 11 * a := by sorry

end NUMINAMATH_CALUDE_equal_interval_line_segments_l390_39067


namespace NUMINAMATH_CALUDE_A_intersect_complement_B_eq_a_l390_39073

-- Define the universal set U
def U : Set Char := {'{', 'a', 'b', 'c', 'd', 'e', '}'}

-- Define set A
def A : Set Char := {'{', 'a', 'b', '}'}

-- Define set B
def B : Set Char := {'{', 'b', 'c', 'd', '}'}

-- Theorem to prove
theorem A_intersect_complement_B_eq_a : A ∩ (U \ B) = {'{', 'a', '}'} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_complement_B_eq_a_l390_39073


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l390_39006

theorem polynomial_remainder_theorem (x : ℝ) :
  let p (x : ℝ) := x^4 - 4*x^2 + 7
  let r := p 3
  r = 52 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l390_39006


namespace NUMINAMATH_CALUDE_train_passing_jogger_l390_39016

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger (jogger_speed train_speed : ℝ) (initial_distance train_length : ℝ) : 
  jogger_speed = 9 →
  train_speed = 45 →
  initial_distance = 360 →
  train_length = 180 →
  (initial_distance + train_length) / (train_speed - jogger_speed) * (3600 / 1000) = 54 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_jogger_l390_39016


namespace NUMINAMATH_CALUDE_melanie_dimes_count_l390_39019

def final_dimes (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

theorem melanie_dimes_count : final_dimes 7 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_count_l390_39019


namespace NUMINAMATH_CALUDE_baseball_team_wins_l390_39027

theorem baseball_team_wins (total_games wins : ℕ) (h1 : total_games = 130) (h2 : wins = 101) :
  let losses := total_games - wins
  wins - 3 * losses = 14 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_wins_l390_39027


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l390_39085

theorem division_multiplication_equality : (0.45 / 0.005) * 0.1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l390_39085


namespace NUMINAMATH_CALUDE_geometric_sum_first_seven_l390_39082

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_seven :
  geometric_sum (1/4) (1/4) 7 = 16383/49152 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_first_seven_l390_39082


namespace NUMINAMATH_CALUDE_sum_of_roots_is_fifteen_l390_39056

/-- A function g: ℝ → ℝ that satisfies g(3+x) = g(3-x) for all real x -/
def SymmetricAboutThree (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (3 + x) = g (3 - x)

/-- The theorem stating that if g is symmetric about 3 and has exactly five distinct real roots,
    then the sum of these roots is 15 -/
theorem sum_of_roots_is_fifteen
    (g : ℝ → ℝ)
    (h_symmetric : SymmetricAboutThree g)
    (h_five_roots : ∃! (s : Finset ℝ), s.card = 5 ∧ ∀ x ∈ s, g x = 0) :
    ∃ (s : Finset ℝ), s.card = 5 ∧ (∀ x ∈ s, g x = 0) ∧ (s.sum id = 15) :=
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_fifteen_l390_39056


namespace NUMINAMATH_CALUDE_inequality_proof_l390_39098

theorem inequality_proof (x y z : ℝ) (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) (hz : z ∈ Set.Icc 0 1) :
  (x / (y + z + 1)) + (y / (z + x + 1)) + (z / (x + y + 1)) ≤ 1 - (1 - x) * (1 - y) * (1 - z) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l390_39098


namespace NUMINAMATH_CALUDE_contradiction_proof_l390_39045

theorem contradiction_proof (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (product_inequality : a * c + b * d > 1) 
  (all_nonnegative : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) : 
  False := by
sorry

end NUMINAMATH_CALUDE_contradiction_proof_l390_39045


namespace NUMINAMATH_CALUDE_students_per_bus_l390_39020

theorem students_per_bus (total_students : ℕ) (num_buses : ℕ) 
  (h1 : total_students = 360) (h2 : num_buses = 8) :
  total_students / num_buses = 45 := by
  sorry

end NUMINAMATH_CALUDE_students_per_bus_l390_39020


namespace NUMINAMATH_CALUDE_frank_defeated_six_enemies_l390_39097

/-- The number of enemies Frank defeated in the game --/
def enemies_defeated : ℕ := sorry

/-- The points earned per enemy defeated --/
def points_per_enemy : ℕ := 9

/-- The bonus points for completing the level --/
def bonus_points : ℕ := 8

/-- The total points Frank earned --/
def total_points : ℕ := 62

/-- Theorem stating that Frank defeated 6 enemies --/
theorem frank_defeated_six_enemies :
  enemies_defeated = 6 ∧
  enemies_defeated * points_per_enemy + bonus_points = total_points :=
sorry

end NUMINAMATH_CALUDE_frank_defeated_six_enemies_l390_39097


namespace NUMINAMATH_CALUDE_teresa_jog_distance_l390_39009

/-- Given a speed of 5 km/h and a time of 5 hours, prove that the distance traveled is 25 km. -/
theorem teresa_jog_distance (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed = 5)
  (h2 : time = 5)
  (h3 : distance = speed * time) : 
  distance = 25 := by
sorry

end NUMINAMATH_CALUDE_teresa_jog_distance_l390_39009


namespace NUMINAMATH_CALUDE_snow_probability_l390_39079

theorem snow_probability (p1 p2 : ℚ) : 
  p1 = 1/4 → p2 = 1/3 → 
  1 - (1 - p1)^4 * (1 - p2)^3 = 68359/100000 := by sorry

end NUMINAMATH_CALUDE_snow_probability_l390_39079


namespace NUMINAMATH_CALUDE_ink_bottle_arrangement_l390_39004

-- Define the type for a row of bottles
def Row := Fin 7 → Bool

-- Define the type for the arrangement of bottles
def Arrangement := Fin 130 → Row

-- Theorem statement
theorem ink_bottle_arrangement (arr : Arrangement) :
  (∃ i j k : Fin 130, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ arr i = arr j ∧ arr j = arr k) ∨
  (∃ i₁ j₁ i₂ j₂ : Fin 130, i₁ ≠ j₁ ∧ i₂ ≠ j₂ ∧ i₁ ≠ i₂ ∧ i₁ ≠ j₂ ∧ j₁ ≠ i₂ ∧ j₁ ≠ j₂ ∧
    arr i₁ = arr j₁ ∧ arr i₂ = arr j₂) :=
by
  sorry

end NUMINAMATH_CALUDE_ink_bottle_arrangement_l390_39004


namespace NUMINAMATH_CALUDE_regular_polygon_with_36_degree_central_angle_l390_39026

theorem regular_polygon_with_36_degree_central_angle (n : ℕ) 
  (h : n > 0) 
  (central_angle : ℝ) 
  (h_central_angle : central_angle = 36) : 
  (360 : ℝ) / central_angle = 10 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_36_degree_central_angle_l390_39026


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l390_39000

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ r ≠ p) →
  (∀ (x : ℝ), x^3 - 20*x^2 + 99*x - 154 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ (t : ℝ), t ≠ p ∧ t ≠ q ∧ t ≠ r → 
    1 / (t^3 - 20*t^2 + 99*t - 154) = A / (t - p) + B / (t - q) + C / (t - r)) →
  1 / A + 1 / B + 1 / C = 245 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l390_39000
