import Mathlib

namespace NUMINAMATH_CALUDE_linear_equation_equivalence_l370_37021

theorem linear_equation_equivalence (x y : ℝ) :
  (3 * x - y + 5 = 0) ↔ (y = 3 * x + 5) := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_equivalence_l370_37021


namespace NUMINAMATH_CALUDE_perfect_square_values_l370_37026

theorem perfect_square_values (p : ℤ) (n : ℚ) : 
  n = 16 * (10 : ℚ)^(-p) →
  -4 < p →
  p < 2 →
  (∃ (a b c : ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (16 * (10 : ℚ)^(-a) = (m : ℚ)^2 ∧
     16 * (10 : ℚ)^(-b) = (k : ℚ)^2 ∧
     16 * (10 : ℚ)^(-c) = (l : ℚ)^2) ∧
    (∀ (x : ℤ), x ≠ a ∧ x ≠ b ∧ x ≠ c →
      ¬∃ (y : ℚ), 16 * (10 : ℚ)^(-x) = y^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_perfect_square_values_l370_37026


namespace NUMINAMATH_CALUDE_binomial_15_3_l370_37080

theorem binomial_15_3 : Nat.choose 15 3 = 455 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_3_l370_37080


namespace NUMINAMATH_CALUDE_solutions_to_z_sixth_eq_neg_64_l370_37084

theorem solutions_to_z_sixth_eq_neg_64 :
  {z : ℂ | z^6 = -64} =
    {2 * (Complex.cos (π / 6) + Complex.I * Complex.sin (π / 6)),
     2 * (Complex.cos (π / 2) + Complex.I * Complex.sin (π / 2)),
     2 * (Complex.cos (5 * π / 6) + Complex.I * Complex.sin (5 * π / 6)),
     2 * (Complex.cos (7 * π / 6) + Complex.I * Complex.sin (7 * π / 6)),
     2 * (Complex.cos (3 * π / 2) + Complex.I * Complex.sin (3 * π / 2)),
     2 * (Complex.cos (11 * π / 6) + Complex.I * Complex.sin (11 * π / 6))} :=
by sorry

end NUMINAMATH_CALUDE_solutions_to_z_sixth_eq_neg_64_l370_37084


namespace NUMINAMATH_CALUDE_proposition_truth_l370_37028

theorem proposition_truth : 
  (¬ (∀ x : ℝ, x + 1/x ≥ 2)) ∧ 
  (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi/2) ∧ Real.sin x + Real.cos x = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l370_37028


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_two_ninths_l370_37079

theorem sum_of_fractions_equals_two_ninths :
  let sum := (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) +
              (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) + (1 / (8 * 9 : ℚ))
  sum = 2 / 9 := by
    sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_two_ninths_l370_37079


namespace NUMINAMATH_CALUDE_sum_of_three_fourth_powers_not_end_2019_l370_37018

theorem sum_of_three_fourth_powers_not_end_2019 :
  ∀ a b c : ℤ, ¬ (∃ k : ℤ, a^4 + b^4 + c^4 = 10000 * k + 2019) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_fourth_powers_not_end_2019_l370_37018


namespace NUMINAMATH_CALUDE_base_conversion_sum_equality_l370_37043

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem base_conversion_sum_equality : 
  let num1 := base_to_decimal [2, 5, 3] 8
  let den1 := base_to_decimal [1, 3] 4
  let num2 := base_to_decimal [1, 4, 4] 5
  let den2 := base_to_decimal [3, 3] 3
  (num1 : ℚ) / den1 + (num2 : ℚ) / den2 = 28.511904 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_equality_l370_37043


namespace NUMINAMATH_CALUDE_max_sum_lcm_165_l370_37017

theorem max_sum_lcm_165 (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  Nat.lcm (Nat.lcm (Nat.lcm a.val b.val) c.val) d.val = 165 →
  a.val + b.val + c.val + d.val ≤ 268 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_lcm_165_l370_37017


namespace NUMINAMATH_CALUDE_floyd_jumps_exist_l370_37004

def sum_of_decimal_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_decimal_digits (n / 10)

def floyd_sequence : ℕ → ℕ
  | 0 => 90
  | n + 1 => 2 * (10^(n + 2)) - 28

theorem floyd_jumps_exist :
  ∃ (a : ℕ → ℕ), (∀ n > 0, a n ≤ 2 * a (n - 1)) ∧
                 (∀ i j, i ≠ j → sum_of_decimal_digits (a i) ≠ sum_of_decimal_digits (a j)) :=
by
  sorry


end NUMINAMATH_CALUDE_floyd_jumps_exist_l370_37004


namespace NUMINAMATH_CALUDE_mary_regular_hours_l370_37030

/-- Represents Mary's work schedule and earnings --/
structure WorkSchedule where
  regularHours : ℕ
  overtimeHours : ℕ
  regularRate : ℕ
  overtimeRate : ℕ
  totalEarnings : ℕ

/-- Calculates the total earnings based on the work schedule --/
def calculateEarnings (schedule : WorkSchedule) : ℕ :=
  schedule.regularHours * schedule.regularRate + schedule.overtimeHours * schedule.overtimeRate

/-- The main theorem stating Mary's work hours at regular rate --/
theorem mary_regular_hours :
  ∃ (schedule : WorkSchedule),
    schedule.regularHours = 40 ∧
    schedule.regularRate = 8 ∧
    schedule.overtimeRate = 10 ∧
    schedule.regularHours + schedule.overtimeHours ≤ 40 ∧
    calculateEarnings schedule = 360 :=
by
  sorry

#check mary_regular_hours

end NUMINAMATH_CALUDE_mary_regular_hours_l370_37030


namespace NUMINAMATH_CALUDE_road_signs_count_l370_37041

/-- The number of road signs at the first intersection -/
def first_intersection : ℕ := 40

/-- The number of road signs at the second intersection -/
def second_intersection : ℕ := first_intersection + (first_intersection / 4)

/-- The number of road signs at the third intersection -/
def third_intersection : ℕ := 2 * second_intersection

/-- The number of road signs at the fourth intersection -/
def fourth_intersection : ℕ := third_intersection - 20

/-- The total number of road signs at all four intersections -/
def total_road_signs : ℕ := first_intersection + second_intersection + third_intersection + fourth_intersection

theorem road_signs_count : total_road_signs = 270 := by
  sorry

end NUMINAMATH_CALUDE_road_signs_count_l370_37041


namespace NUMINAMATH_CALUDE_tan_equality_integer_l370_37062

theorem tan_equality_integer (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (1230 * π / 180) → n = -30 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_integer_l370_37062


namespace NUMINAMATH_CALUDE_p_recurrence_l370_37071

/-- Probability of having a group of length k or more in n tosses of a symmetric coin -/
def p (n k : ℕ) : ℝ :=
  sorry

/-- The recurrence relation for p(n, k) -/
theorem p_recurrence (n k : ℕ) (h : k < n) :
  p n k = p (n - 1) k - (1 / 2^k) * p (n - k) k + (1 / 2^k) :=
sorry

end NUMINAMATH_CALUDE_p_recurrence_l370_37071


namespace NUMINAMATH_CALUDE_waiter_customers_l370_37000

/-- The initial number of customers -/
def initial_customers : ℕ := 47

/-- The number of customers who left -/
def customers_left : ℕ := 41

/-- The number of new customers who arrived -/
def new_customers : ℕ := 20

/-- The final number of customers -/
def final_customers : ℕ := 26

theorem waiter_customers : 
  initial_customers - customers_left + new_customers = final_customers :=
by sorry

end NUMINAMATH_CALUDE_waiter_customers_l370_37000


namespace NUMINAMATH_CALUDE_overbridge_length_l370_37066

/-- Calculates the length of an overbridge given train parameters --/
theorem overbridge_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (crossing_time : ℝ) : 
  train_length = 600 →
  train_speed_kmh = 36 →
  crossing_time = 70 →
  (train_length + (train_speed_kmh * 1000 / 3600 * crossing_time)) - train_length = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_overbridge_length_l370_37066


namespace NUMINAMATH_CALUDE_book_pages_l370_37094

/-- The number of days Lex read the book -/
def days : ℕ := 12

/-- The number of pages Lex read per day -/
def pages_per_day : ℕ := 20

/-- The total number of pages in the book -/
def total_pages : ℕ := days * pages_per_day

theorem book_pages : total_pages = 240 := by sorry

end NUMINAMATH_CALUDE_book_pages_l370_37094


namespace NUMINAMATH_CALUDE_correct_division_l370_37098

theorem correct_division (n : ℕ) : 
  n % 8 = 2 ∧ n / 8 = 156 → n / 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_correct_division_l370_37098


namespace NUMINAMATH_CALUDE_school_average_age_l370_37036

theorem school_average_age (total_students : ℕ) (boys_avg_age girls_avg_age : ℚ) (num_girls : ℕ) :
  total_students = 632 →
  boys_avg_age = 12 →
  girls_avg_age = 11 →
  num_girls = 158 →
  let num_boys := total_students - num_girls
  let total_age := boys_avg_age * num_boys + girls_avg_age * num_girls
  total_age / total_students = 11.75 := by
  sorry

end NUMINAMATH_CALUDE_school_average_age_l370_37036


namespace NUMINAMATH_CALUDE_perfect_square_condition_l370_37033

theorem perfect_square_condition (n : ℕ) : 
  (∃ (a : ℕ), 2^n + 3 = a^2) ↔ n = 0 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l370_37033


namespace NUMINAMATH_CALUDE_power_multiplication_l370_37011

theorem power_multiplication (a : ℝ) : 3 * a^4 * (4 * a) = 12 * a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l370_37011


namespace NUMINAMATH_CALUDE_ratio_problem_l370_37009

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l370_37009


namespace NUMINAMATH_CALUDE_sqrt_inequality_max_abc_positive_l370_37096

-- Problem 1
theorem sqrt_inequality (a : ℝ) (h : a > 0) :
  Real.sqrt (a + 5) - Real.sqrt (a + 3) > Real.sqrt (a + 6) - Real.sqrt (a + 4) := by
  sorry

-- Problem 2
theorem max_abc_positive (x y z : ℝ) :
  let a := x^2 - 2*y + Real.pi/2
  let b := y^2 - 2*z + Real.pi/3
  let c := z^2 - 2*x + Real.pi/6
  max a (max b c) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_max_abc_positive_l370_37096


namespace NUMINAMATH_CALUDE_solutions_of_equation_l370_37006

theorem solutions_of_equation (x : ℝ) : x * (x - 1) = x ↔ x = 0 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_solutions_of_equation_l370_37006


namespace NUMINAMATH_CALUDE_max_vector_sum_on_circle_l370_37002

def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 5 = 0

def point_on_circle (p : ℝ × ℝ) : Prop := circle_C p.1 p.2

theorem max_vector_sum_on_circle (A B : ℝ × ℝ) :
  point_on_circle A →
  point_on_circle B →
  ‖(A.1 - B.1, A.2 - B.2)‖ = 2 * Real.sqrt 3 →
  ∃ (max : ℝ), max = 8 ∧ ∀ (A' B' : ℝ × ℝ),
    point_on_circle A' →
    point_on_circle B' →
    ‖(A'.1 - B'.1, A'.2 - B'.2)‖ = 2 * Real.sqrt 3 →
    ‖(A'.1 + B'.1, A'.2 + B'.2)‖ ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_vector_sum_on_circle_l370_37002


namespace NUMINAMATH_CALUDE_bead_system_eventually_repeats_l370_37090

-- Define the bead system
structure BeadSystem where
  n : ℕ  -- number of beads
  ω : ℝ  -- angular speed
  direction : Fin n → Bool  -- true for clockwise, false for counterclockwise
  initial_position : Fin n → ℝ  -- initial angular position of each bead

-- Define the state of the system at a given time
def system_state (bs : BeadSystem) (t : ℝ) : Fin bs.n → ℝ :=
  sorry

-- Define what it means for the system to repeat its initial configuration
def repeats_initial_config (bs : BeadSystem) (t : ℝ) : Prop :=
  ∃ (perm : Equiv.Perm (Fin bs.n)),
    ∀ i, system_state bs t (perm i) = bs.initial_position i

-- State the theorem
theorem bead_system_eventually_repeats (bs : BeadSystem) :
  ∃ t > 0, repeats_initial_config bs t :=
sorry

end NUMINAMATH_CALUDE_bead_system_eventually_repeats_l370_37090


namespace NUMINAMATH_CALUDE_parabola_symmetry_l370_37015

def C₁ (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 1

def C₂ (x : ℝ) : ℝ := 2 * (x - 3)^2 - 4 * (x - 3) - 1

def is_symmetry_line (f g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a - x) = g (a + x)

theorem parabola_symmetry :
  is_symmetry_line C₁ C₂ (5/2) :=
sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l370_37015


namespace NUMINAMATH_CALUDE_impossible_coin_probabilities_l370_37099

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧ 
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) := by
  sorry

end NUMINAMATH_CALUDE_impossible_coin_probabilities_l370_37099


namespace NUMINAMATH_CALUDE_inequality_proof_l370_37077

theorem inequality_proof (a b c : ℝ) : a^4 + b^4 + c^4 ≥ a*b*c^2 + b*c*a^2 + c*a*b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l370_37077


namespace NUMINAMATH_CALUDE_incorrect_expression_for_repeating_decimal_l370_37013

def repeating_decimal (X Y : ℕ) (a b : ℕ) : ℚ :=
  (X : ℚ) / 10^a + (Y : ℚ) / (10^a * (10^b - 1))

theorem incorrect_expression_for_repeating_decimal (X Y a b : ℕ) :
  ∃ V : ℚ, V = repeating_decimal X Y a b ∧ 10^a * (10^b - 1) * V ≠ X * (Y - 1) :=
sorry

end NUMINAMATH_CALUDE_incorrect_expression_for_repeating_decimal_l370_37013


namespace NUMINAMATH_CALUDE_tangent_slope_angle_l370_37052

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + 5

theorem tangent_slope_angle (x : ℝ) : 
  x = 1 → 
  ∃ θ : ℝ, θ = 3 * Real.pi / 4 ∧ 
    θ = Real.pi + Real.arctan ((deriv f) x) :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_l370_37052


namespace NUMINAMATH_CALUDE_S_formula_no_c_k_exist_l370_37095

def S : ℕ → ℚ
  | 0 => 0
  | n + 1 => 1/2 * S n + 2

theorem S_formula (n : ℕ) : S n = 4 * (1 - 1 / 2^n) := by sorry

theorem no_c_k_exist :
  ¬∃ (c k : ℕ), (S k + 1 - c) / (S k - c) > 2 := by sorry

end NUMINAMATH_CALUDE_S_formula_no_c_k_exist_l370_37095


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l370_37069

theorem quadratic_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x + m = 0 ∧ y^2 - 4*y + m = 0) → m < 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l370_37069


namespace NUMINAMATH_CALUDE_expression_evaluation_l370_37072

theorem expression_evaluation : ∃ (n m k : ℕ),
  (n > 0 ∧ m > 0 ∧ k > 0) ∧
  (2 * n - 1 = 2025) ∧
  (2 * m = 2024) ∧
  (2^k = 1024) →
  (Finset.sum (Finset.range n) (λ i => 2 * i + 5)) -
  (Finset.sum (Finset.range m) (λ i => 2 * i + 4)) +
  2 * (Finset.sum (Finset.range k) (λ i => 2^i)) = 5104 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l370_37072


namespace NUMINAMATH_CALUDE_syllogism_arrangement_correct_l370_37029

-- Define the statements
def statement1 : Prop := 2012 % 2 = 0
def statement2 : Prop := ∀ n : ℕ, Even n → n % 2 = 0
def statement3 : Prop := Even 2012

-- Define the syllogism structure
inductive SyllogismStep
| MajorPremise
| MinorPremise
| Conclusion

-- Define a function to represent the correct arrangement
def correctArrangement : List (SyllogismStep × Prop) :=
  [(SyllogismStep.MajorPremise, statement2),
   (SyllogismStep.MinorPremise, statement3),
   (SyllogismStep.Conclusion, statement1)]

-- Theorem to prove
theorem syllogism_arrangement_correct :
  correctArrangement = 
    [(SyllogismStep.MajorPremise, statement2),
     (SyllogismStep.MinorPremise, statement3),
     (SyllogismStep.Conclusion, statement1)] :=
by sorry

end NUMINAMATH_CALUDE_syllogism_arrangement_correct_l370_37029


namespace NUMINAMATH_CALUDE_part_one_part_two_l370_37051

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x - 1

-- Part (1)
theorem part_one (a : ℝ) :
  (∀ x : ℝ, f a x ≤ -3/4) ↔ a ∈ Set.Icc (-1) (-1/4) :=
sorry

-- Part (2)
theorem part_two (a : ℝ) :
  a ≤ 0 → ((∀ x : ℝ, x > 0 → x * f a x ≤ 1) ↔ a ∈ Set.Icc (-3) 0) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l370_37051


namespace NUMINAMATH_CALUDE_intersection_point_parameter_l370_37014

/-- Given three lines that intersect at a single point but do not form a triangle, 
    prove that the parameter 'a' in one of the lines must equal -1. -/
theorem intersection_point_parameter (a : ℝ) : 
  (∃ (x y : ℝ), ax + 2*y + 8 = 0 ∧ 4*x + 3*y = 10 ∧ 2*x - y = 10) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (ax₁ + 2*y₁ + 8 = 0 ∧ 4*x₁ + 3*y₁ = 10) →
    (ax₂ + 2*y₂ + 8 = 0 ∧ 2*x₂ - y₂ = 10) →
    x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
  (∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (ax₁ + 2*y₁ + 8 = 0 ∧ 4*x₂ + 3*y₂ = 10 ∧ 2*x₃ - y₃ = 10) →
    ¬(Set.ncard {(x₁, y₁), (x₂, y₂), (x₃, y₃)} = 3)) →
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_parameter_l370_37014


namespace NUMINAMATH_CALUDE_square_inequality_condition_l370_37086

theorem square_inequality_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_square_inequality_condition_l370_37086


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l370_37059

/-- For a hyperbola with equation x^2/45 - y^2/5 = 1, the distance between its foci is 10√2 -/
theorem hyperbola_foci_distance :
  ∀ x y : ℝ,
  (x^2 / 45) - (y^2 / 5) = 1 →
  ∃ f₁ f₂ : ℝ × ℝ,
  (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 200 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l370_37059


namespace NUMINAMATH_CALUDE_equation_transformation_l370_37012

theorem equation_transformation (x : ℝ) : (3 * x - 7 = 2 * x) ↔ (3 * x - 2 * x = 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l370_37012


namespace NUMINAMATH_CALUDE_regular_pentagon_diagonal_angle_l370_37038

/-- A regular pentagon is a polygon with 5 equal sides and 5 equal angles -/
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_regular : sorry

/-- The measure of an angle in degrees -/
def angle_measure (a b c : ℝ × ℝ) : ℝ := sorry

theorem regular_pentagon_diagonal_angle 
  (ABCDE : RegularPentagon) 
  (h_interior : ∀ (i : Fin 5), angle_measure (ABCDE.vertices i) (ABCDE.vertices (i + 1)) (ABCDE.vertices (i + 2)) = 108) :
  angle_measure (ABCDE.vertices 0) (ABCDE.vertices 2) (ABCDE.vertices 1) = 36 := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_diagonal_angle_l370_37038


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_from_tangent_circle_l370_37020

/-- Given a circle and a hyperbola, if the circle is tangent to the asymptotes of the hyperbola,
    then the eccentricity of the hyperbola is 5/2. -/
theorem hyperbola_eccentricity_from_tangent_circle
  (a b : ℝ) (h_positive : a > 0 ∧ b > 0) :
  let circle := fun (x y : ℝ) => x^2 + y^2 - 10*y + 21 = 0
  let hyperbola := fun (x y : ℝ) => x^2/a^2 - y^2/b^2 = 1
  let asymptote := fun (x y : ℝ) => b*x - a*y = 0 ∨ b*x + a*y = 0
  let is_tangent := ∃ (x y : ℝ), circle x y ∧ asymptote x y
  let eccentricity := Real.sqrt (1 + b^2/a^2)
  is_tangent → eccentricity = 5/2 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_from_tangent_circle_l370_37020


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l370_37075

-- Problem 1
theorem problem_1 (m n : ℚ) :
  (∀ x, (x - 3) * (x - 4) = x^2 + m*x + n) → m = -7 ∧ n = 12 := by sorry

-- Problem 2
theorem problem_2 (a b : ℚ) :
  (∀ x, (x + a) * (x + b) = x^2 - 3*x + 1/3) →
  (a - 1) * (b - 1) = 13/3 ∧ 1/a^2 + 1/b^2 = 75 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l370_37075


namespace NUMINAMATH_CALUDE_train_problem_solution_l370_37040

/-- Represents the day of the week -/
inductive Day
  | Saturday
  | Monday

/-- Represents a date in a month -/
structure Date where
  day : Day
  number : Nat

/-- Represents a train car -/
structure TrainCar where
  number : Nat
  seat : Nat

/-- The problem setup -/
def TrainProblem (d1 d2 : Date) (car : TrainCar) : Prop :=
  d1.day = Day.Saturday ∧
  d2.day = Day.Monday ∧
  d2.number = car.number ∧
  car.seat < car.number ∧
  d1.number > car.number ∧
  d1.number ≠ d2.number ∧
  car.number < 10

theorem train_problem_solution :
  ∀ (d1 d2 : Date) (car : TrainCar),
    TrainProblem d1 d2 car →
    car.number = 2 ∧ car.seat = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_train_problem_solution_l370_37040


namespace NUMINAMATH_CALUDE_allocate_spots_eq_21_l370_37025

/-- The number of ways to allocate 8 spots among 6 classes, with at least one spot per class -/
def allocate_spots : ℕ :=
  let n_classes : ℕ := 6
  let total_spots : ℕ := 8
  let remaining_spots : ℕ := total_spots - n_classes
  let same_class_outcomes : ℕ := n_classes
  let different_classes_outcomes : ℕ := n_classes.choose 2
  same_class_outcomes + different_classes_outcomes

theorem allocate_spots_eq_21 : allocate_spots = 21 := by
  sorry

end NUMINAMATH_CALUDE_allocate_spots_eq_21_l370_37025


namespace NUMINAMATH_CALUDE_integral_value_l370_37049

theorem integral_value : ∫ x in (2 : ℝ)..4, (x^3 - 3*x^2 + 5) / x^2 = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_integral_value_l370_37049


namespace NUMINAMATH_CALUDE_power_two_greater_than_square_l370_37046

theorem power_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_greater_than_square_l370_37046


namespace NUMINAMATH_CALUDE_quadratic_problem_l370_37001

/-- A quadratic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating the properties of the quadratic function and its value at x = 5 -/
theorem quadratic_problem (a b c : ℝ) :
  (∀ x, f a b c x ≥ 4) →  -- Minimum value is 4
  (f a b c 2 = 4) →  -- Minimum occurs at x = 2
  (f a b c 0 = -8) →  -- Passes through (0, -8)
  (f a b c 5 = 31) :=  -- Passes through (5, 31)
by sorry

end NUMINAMATH_CALUDE_quadratic_problem_l370_37001


namespace NUMINAMATH_CALUDE_second_number_approximation_l370_37057

theorem second_number_approximation (x y z : ℝ) 
  (sum_eq : x + y + z = 120)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 7 / 9)
  (x_pos : x > 0) (y_pos : y > 0) (z_pos : z > 0) : 
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 1 ∧ y = 40 + ε :=
sorry

end NUMINAMATH_CALUDE_second_number_approximation_l370_37057


namespace NUMINAMATH_CALUDE_sin_sum_angles_l370_37074

theorem sin_sum_angles (α β : ℝ) 
  (h1 : Real.sin α + Real.cos β = 1/4)
  (h2 : Real.cos α + Real.sin β = -8/5) : 
  Real.sin (α + β) = 249/800 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_angles_l370_37074


namespace NUMINAMATH_CALUDE_percentage_increase_l370_37061

theorem percentage_increase (original : ℝ) (final : ℝ) (increase : ℝ) :
  original = 90 →
  final = 135 →
  increase = (final - original) / original * 100 →
  increase = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l370_37061


namespace NUMINAMATH_CALUDE_sqrt_sum_of_squares_l370_37050

theorem sqrt_sum_of_squares : 
  Real.sqrt ((43 * 17)^2 + (43 * 26)^2 + (17 * 26)^2) = 1407 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_of_squares_l370_37050


namespace NUMINAMATH_CALUDE_drama_club_theorem_l370_37044

theorem drama_club_theorem (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) 
  (h1 : total = 50)
  (h2 : math = 36)
  (h3 : physics = 27)
  (h4 : both = 20) :
  total - (math - both + physics - both + both) = 7 :=
by sorry

end NUMINAMATH_CALUDE_drama_club_theorem_l370_37044


namespace NUMINAMATH_CALUDE_floral_shop_sale_total_l370_37056

/-- Represents the total number of bouquets sold during a three-day sale at a floral shop. -/
def total_bouquets_sold (monday_sales : ℕ) : ℕ :=
  let tuesday_sales := 3 * monday_sales
  let wednesday_sales := tuesday_sales / 3
  monday_sales + tuesday_sales + wednesday_sales

/-- Theorem stating that given the conditions of the sale, the total number of bouquets sold is 60. -/
theorem floral_shop_sale_total (h : total_bouquets_sold 12 = 60) : 
  total_bouquets_sold 12 = 60 := by
  sorry

end NUMINAMATH_CALUDE_floral_shop_sale_total_l370_37056


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l370_37034

theorem partial_fraction_decomposition :
  let f (x : ℝ) := (2 * x^2 + 5 * x - 3) / (x^2 - x - 42)
  let g (x : ℝ) := (11/13) / (x - 7) + (15/13) / (x + 6)
  ∀ x : ℝ, x ≠ 7 → x ≠ -6 → f x = g x :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l370_37034


namespace NUMINAMATH_CALUDE_red_apples_sold_l370_37063

theorem red_apples_sold (red green : ℕ) : 
  (red : ℚ) / green = 8 / 3 → 
  red + green = 44 → 
  red = 32 := by
sorry

end NUMINAMATH_CALUDE_red_apples_sold_l370_37063


namespace NUMINAMATH_CALUDE_other_asymptote_equation_l370_37005

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One of the asymptotes of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- x-coordinate of the foci -/
  foci_x : ℝ

/-- Theorem: Given a hyperbola with one asymptote y = 4x - 3 and foci with x-coordinate 3,
    the equation of the other asymptote is y = -4x + 21 -/
theorem other_asymptote_equation (h : Hyperbola) 
    (h1 : h.asymptote1 = fun x ↦ 4 * x - 3) 
    (h2 : h.foci_x = 3) : 
    ∃ asymptote2 : ℝ → ℝ, asymptote2 = fun x ↦ -4 * x + 21 := by
  sorry

end NUMINAMATH_CALUDE_other_asymptote_equation_l370_37005


namespace NUMINAMATH_CALUDE_claires_weight_l370_37022

theorem claires_weight (alice_weight claire_weight : ℚ) : 
  alice_weight + claire_weight = 200 →
  claire_weight - alice_weight = claire_weight / 3 →
  claire_weight = 1400 / 9 := by
sorry

end NUMINAMATH_CALUDE_claires_weight_l370_37022


namespace NUMINAMATH_CALUDE_largest_fraction_l370_37019

theorem largest_fraction : 
  let fractions := [2/5, 3/7, 4/9, 5/11, 6/13]
  ∀ x ∈ fractions, (6/13 : ℚ) ≥ x :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_l370_37019


namespace NUMINAMATH_CALUDE_solution_equivalence_l370_37060

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {((1 : ℝ) / Real.rpow 6 (1/6), Real.sqrt 2 / Real.rpow 6 (1/6), Real.sqrt 3 / Real.rpow 6 (1/6)),
   (-(1 : ℝ) / Real.rpow 6 (1/6), -Real.sqrt 2 / Real.rpow 6 (1/6), Real.sqrt 3 / Real.rpow 6 (1/6)),
   (-(1 : ℝ) / Real.rpow 6 (1/6), Real.sqrt 2 / Real.rpow 6 (1/6), -Real.sqrt 3 / Real.rpow 6 (1/6)),
   ((1 : ℝ) / Real.rpow 6 (1/6), -Real.sqrt 2 / Real.rpow 6 (1/6), -Real.sqrt 3 / Real.rpow 6 (1/6))}

def satisfies_equations (x y z : ℝ) : Prop :=
  x^3 * y^3 * z^3 = 1 ∧ x * y^5 * z^3 = 2 ∧ x * y^3 * z^5 = 3

theorem solution_equivalence :
  ∀ x y z : ℝ, satisfies_equations x y z ↔ (x, y, z) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_solution_equivalence_l370_37060


namespace NUMINAMATH_CALUDE_product_of_roots_l370_37078

theorem product_of_roots (x : ℝ) : (x + 2) * (x - 3) = -10 → 
  ∃ (r₁ r₂ : ℝ), r₁ * r₂ = 4 ∧ (x = r₁ ∨ x = r₂) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l370_37078


namespace NUMINAMATH_CALUDE_ellipse_and_line_theorem_l370_37008

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

-- Define the focal length
def focal_length (c : ℝ) : Prop := c = 2

-- Define that C passes through P(2, 5/3)
def passes_through_P (C : ℝ → ℝ → Prop) : Prop :=
  C 2 (5/3)

-- Define line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

-- Define that l passes through M(0, 1)
def passes_through_M (l : ℝ → ℝ → Prop) : Prop :=
  l 0 1

-- Define the condition for A and B
def vector_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 - 0, A.2 - 1) = (-2/3 * (B.1 - 0), -2/3 * (B.2 - 1))

-- Main theorem
theorem ellipse_and_line_theorem :
  ∀ C : ℝ → ℝ → Prop,
  (∀ x y, C x y ↔ x^2 / 9 + y^2 / 5 = 1) →
  focal_length 2 →
  passes_through_P C →
  ∃ k : ℝ, k = 1/3 ∨ k = -1/3 ∧
    ∀ x y, line_l k x y →
    passes_through_M (line_l k) ∧
    ∃ A B : ℝ × ℝ,
      C A.1 A.2 ∧ C B.1 B.2 ∧
      line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧
      vector_condition A B :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_theorem_l370_37008


namespace NUMINAMATH_CALUDE_certain_number_multiplication_l370_37003

theorem certain_number_multiplication (x : ℝ) : 37 - x = 24 → x * 24 = 312 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_multiplication_l370_37003


namespace NUMINAMATH_CALUDE_car_value_reduction_l370_37055

theorem car_value_reduction (original_price current_value : ℝ) : 
  current_value = 0.7 * original_price → 
  current_value = 2800 → 
  original_price = 4000 := by
sorry

end NUMINAMATH_CALUDE_car_value_reduction_l370_37055


namespace NUMINAMATH_CALUDE_monotonically_decreasing_x_ln_x_l370_37067

/-- The function f(x) = x ln x is monotonically decreasing on the interval (0, 1/e) -/
theorem monotonically_decreasing_x_ln_x :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1 / Real.exp 1 →
  x₁ * Real.log x₁ > x₂ * Real.log x₂ := by
sorry

/-- The domain of f(x) = x ln x is (0, +∞) -/
def domain_x_ln_x : Set ℝ := {x : ℝ | x > 0}

end NUMINAMATH_CALUDE_monotonically_decreasing_x_ln_x_l370_37067


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l370_37083

theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 3 = 3 →
  a 6 = 24 →
  a 9 = 45 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l370_37083


namespace NUMINAMATH_CALUDE_gondor_monday_phones_l370_37089

/-- Represents the earnings and repair information for Gondor --/
structure GondorEarnings where
  phone_repair_cost : ℕ
  laptop_repair_cost : ℕ
  tuesday_phones : ℕ
  wednesday_laptops : ℕ
  thursday_laptops : ℕ
  total_earnings : ℕ

/-- Calculates the number of phones repaired on Monday --/
def monday_phones (g : GondorEarnings) : ℕ :=
  (g.total_earnings - (g.phone_repair_cost * g.tuesday_phones + 
   g.laptop_repair_cost * (g.wednesday_laptops + g.thursday_laptops))) / g.phone_repair_cost

/-- Theorem stating that Gondor repaired 3 phones on Monday --/
theorem gondor_monday_phones (g : GondorEarnings) 
  (h1 : g.phone_repair_cost = 10)
  (h2 : g.laptop_repair_cost = 20)
  (h3 : g.tuesday_phones = 5)
  (h4 : g.wednesday_laptops = 2)
  (h5 : g.thursday_laptops = 4)
  (h6 : g.total_earnings = 200) :
  monday_phones g = 3 := by
  sorry

end NUMINAMATH_CALUDE_gondor_monday_phones_l370_37089


namespace NUMINAMATH_CALUDE_larger_box_jellybeans_l370_37093

def jellybeans_in_box (length width height : ℕ) : ℕ := length * width * height * 20

theorem larger_box_jellybeans (l w h : ℕ) :
  jellybeans_in_box l w h = 200 →
  jellybeans_in_box (3 * l) (3 * w) (3 * h) = 5400 :=
by
  sorry

#check larger_box_jellybeans

end NUMINAMATH_CALUDE_larger_box_jellybeans_l370_37093


namespace NUMINAMATH_CALUDE_ab_equals_one_l370_37088

theorem ab_equals_one (a b : ℝ) (ha : a = Real.sqrt 3 / 3) (hb : b = Real.sqrt 3) : a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_one_l370_37088


namespace NUMINAMATH_CALUDE_initial_disappearance_percentage_l370_37065

/-- Proof of the initial percentage of inhabitants that disappeared from a village --/
theorem initial_disappearance_percentage 
  (initial_population : ℕ) 
  (final_population : ℕ) 
  (initial_population_eq : initial_population = 7600)
  (final_population_eq : final_population = 5130) :
  ∃ (p : ℝ), 
    p = 10 ∧ 
    (initial_population : ℝ) * (1 - p / 100) * 0.75 = final_population := by
  sorry

end NUMINAMATH_CALUDE_initial_disappearance_percentage_l370_37065


namespace NUMINAMATH_CALUDE_ellipse_foci_coordinates_l370_37024

/-- The coordinates of the foci of the ellipse x²/16 + y²/25 = 1 are (0, ±3) -/
theorem ellipse_foci_coordinates : 
  ∀ (x y : ℝ), x^2/16 + y^2/25 = 1 → 
  ∃ (c : ℝ), c = 3 ∧ 
  ((x = 0 ∧ y = c) ∨ (x = 0 ∧ y = -c)) := by
sorry

end NUMINAMATH_CALUDE_ellipse_foci_coordinates_l370_37024


namespace NUMINAMATH_CALUDE_complex_magnitude_sum_l370_37045

theorem complex_magnitude_sum (i : ℂ) : i^2 = -1 →
  Complex.abs ((2 + i)^24 + (2 - i)^24) = 488281250 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_sum_l370_37045


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l370_37016

theorem imaginary_power_sum : ∃ (i : ℂ), i^2 = -1 ∧ i^14764 + i^14765 + i^14766 + i^14767 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l370_37016


namespace NUMINAMATH_CALUDE_majority_can_play_and_ride_l370_37073

/-- Represents a person's location and height -/
structure Person where
  location : ℝ × ℝ
  height : ℝ

/-- The population of the country -/
def Population := List Person

/-- Checks if a person is taller than the majority within a given radius -/
def isTallerThanMajority (p : Person) (pop : Population) (radius : ℝ) : Bool :=
  sorry

/-- Checks if a person is shorter than the majority within a given radius -/
def isShorterThanMajority (p : Person) (pop : Population) (radius : ℝ) : Bool :=
  sorry

/-- Checks if a person can play basketball (i.e., can choose a radius to be taller than majority) -/
def canPlayBasketball (p : Person) (pop : Population) : Bool :=
  sorry

/-- Checks if a person is entitled to free transportation (i.e., can choose a radius to be shorter than majority) -/
def hasFreeTrans (p : Person) (pop : Population) : Bool :=
  sorry

/-- Calculates the percentage of people satisfying a given condition -/
def percentageSatisfying (pop : Population) (condition : Person → Population → Bool) : ℝ :=
  sorry

theorem majority_can_play_and_ride (pop : Population) :
  percentageSatisfying pop canPlayBasketball ≥ 90 ∧
  percentageSatisfying pop hasFreeTrans ≥ 90 :=
sorry

end NUMINAMATH_CALUDE_majority_can_play_and_ride_l370_37073


namespace NUMINAMATH_CALUDE_complex_modulus_l370_37092

theorem complex_modulus (z : ℂ) (h : z = (1/2 : ℂ) + (5/2 : ℂ) * Complex.I) : 
  Complex.abs z = Real.sqrt 26 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l370_37092


namespace NUMINAMATH_CALUDE_polynomial_remainder_l370_37042

def p (x : ℝ) : ℝ := 5*x^9 - 3*x^7 + 4*x^6 - 8*x^4 + 3*x^3 - 6*x + 5

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, p = λ x => (3*x - 6) * q x + 2321 :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l370_37042


namespace NUMINAMATH_CALUDE_prop_2_prop_3_prop_4_l370_37087

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the existence of two distinct lines and two distinct planes
variable (a b : Line)
variable (α β : Plane)
variable (h_distinct_lines : a ≠ b)
variable (h_distinct_planes : α ≠ β)

-- Proposition ②
theorem prop_2 : 
  (perpendicular a α ∧ perpendicular a β) → parallel_planes α β :=
sorry

-- Proposition ③
theorem prop_3 :
  perpendicular_planes α β → 
  ∃ γ : Plane, perpendicular_planes γ α ∧ perpendicular_planes γ β :=
sorry

-- Proposition ④
theorem prop_4 :
  perpendicular_planes α β → 
  ∃ l : Line, perpendicular l α ∧ parallel l β :=
sorry

end NUMINAMATH_CALUDE_prop_2_prop_3_prop_4_l370_37087


namespace NUMINAMATH_CALUDE_factorization_proof_l370_37076

theorem factorization_proof (x : ℝ) : 
  (3 * x^2 - 12 = 3 * (x + 2) * (x - 2)) ∧ 
  (x^2 - 2*x - 8 = (x - 4) * (x + 2)) := by
sorry

end NUMINAMATH_CALUDE_factorization_proof_l370_37076


namespace NUMINAMATH_CALUDE_car_distance_in_30_minutes_l370_37048

-- Define the train's speed in miles per hour
def train_speed : ℚ := 100

-- Define the car's speed as a fraction of the train's speed
def car_speed : ℚ := (2/3) * train_speed

-- Define the time in hours (30 minutes = 1/2 hour)
def time : ℚ := 1/2

-- Theorem statement
theorem car_distance_in_30_minutes :
  car_speed * time = 100/3 := by sorry

end NUMINAMATH_CALUDE_car_distance_in_30_minutes_l370_37048


namespace NUMINAMATH_CALUDE_max_edges_no_cycle4_l370_37068

/-- A graph with no cycle of length 4 -/
structure NoCycle4Graph where
  vertexCount : ℕ
  edgeCount : ℕ
  noCycle4 : Bool

/-- The maximum number of edges in a graph with 8 vertices and no 4-cycle -/
def maxEdgesNoCycle4 (g : NoCycle4Graph) : Prop :=
  g.vertexCount = 8 ∧ g.noCycle4 = true → g.edgeCount ≤ 25

/-- Theorem stating the maximum number of edges in a graph with 8 vertices and no 4-cycle -/
theorem max_edges_no_cycle4 (g : NoCycle4Graph) : maxEdgesNoCycle4 g := by
  sorry

#check max_edges_no_cycle4

end NUMINAMATH_CALUDE_max_edges_no_cycle4_l370_37068


namespace NUMINAMATH_CALUDE_max_purple_points_theorem_l370_37097

/-- The maximum number of purple points in a configuration of blue and red lines -/
def max_purple_points (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3) * (n - 4) / 8

/-- Theorem stating the maximum number of purple points given n blue lines -/
theorem max_purple_points_theorem (n : ℕ) (h : n ≥ 5) :
  let blue_lines := n
  let no_parallel := true
  let no_concurrent := true
  max_purple_points n = n * (n - 1) * (n - 2) * (n - 3) * (n - 4) / 8 :=
by
  sorry

#check max_purple_points_theorem

end NUMINAMATH_CALUDE_max_purple_points_theorem_l370_37097


namespace NUMINAMATH_CALUDE_tangent_half_angle_sum_l370_37054

theorem tangent_half_angle_sum (α β γ : Real) (h : α + β + γ = Real.pi) :
  Real.tan (α/2) * Real.tan (β/2) + Real.tan (β/2) * Real.tan (γ/2) + Real.tan (γ/2) * Real.tan (α/2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_half_angle_sum_l370_37054


namespace NUMINAMATH_CALUDE_original_decimal_proof_l370_37007

theorem original_decimal_proof (x : ℝ) : x * 12 = 84.6 ↔ x = 7.05 := by
  sorry

end NUMINAMATH_CALUDE_original_decimal_proof_l370_37007


namespace NUMINAMATH_CALUDE_smallest_three_digit_prime_with_prime_reverse_l370_37023

/-- A function that reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number has three digits -/
def hasThreeDigits (n : ℕ) : Prop := sorry

theorem smallest_three_digit_prime_with_prime_reverse : 
  (∀ n : ℕ, hasThreeDigits n → isPrime n → isPrime (reverseDigits n) → 107 ≤ n) ∧ 
  hasThreeDigits 107 ∧ 
  isPrime 107 ∧ 
  isPrime (reverseDigits 107) := by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_prime_with_prime_reverse_l370_37023


namespace NUMINAMATH_CALUDE_red_peaches_count_l370_37091

theorem red_peaches_count (total_baskets : ℕ) (green_per_basket : ℕ) (total_peaches : ℕ) :
  total_baskets = 11 →
  green_per_basket = 18 →
  total_peaches = 308 →
  ∃ red_per_basket : ℕ,
    red_per_basket * total_baskets + green_per_basket * total_baskets = total_peaches ∧
    red_per_basket = 10 :=
by sorry

end NUMINAMATH_CALUDE_red_peaches_count_l370_37091


namespace NUMINAMATH_CALUDE_min_value_theorem_l370_37070

/-- The line equation ax - by + 3 = 0 --/
def line_equation (a b x y : ℝ) : Prop := a * x - b * y + 3 = 0

/-- The circle equation x^2 + y^2 + 2x - 4y + 1 = 0 --/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line divides the area of the circle in half --/
def line_bisects_circle (a b : ℝ) : Prop := 
  ∃ x y : ℝ, line_equation a b x y ∧ circle_equation x y

/-- The main theorem --/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) 
  (h_bisect : line_bisects_circle a b) : 
  (∀ a' b' : ℝ, a' > 0 → b' > 1 → line_bisects_circle a' b' → 
    2/a + 1/(b-1) ≤ 2/a' + 1/(b'-1)) → 
  2/a + 1/(b-1) = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l370_37070


namespace NUMINAMATH_CALUDE_least_four_digit_square_fourth_power_l370_37082

theorem least_four_digit_square_fourth_power : ∃ n : ℕ, 
  (1000 ≤ n ∧ n < 10000) ∧ 
  (∃ a : ℕ, n = a ^ 2) ∧ 
  (∃ b : ℕ, n = b ^ 4) ∧
  (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) → (∃ c : ℕ, m = c ^ 2) → (∃ d : ℕ, m = d ^ 4) → n ≤ m) ∧
  n = 6561 :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_square_fourth_power_l370_37082


namespace NUMINAMATH_CALUDE_unique_solution_for_inequality_l370_37010

theorem unique_solution_for_inequality : 
  ∃! n : ℕ+, -46 ≤ (2023 : ℝ) / (46 - n.val) ∧ (2023 : ℝ) / (46 - n.val) ≤ 46 - n.val :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_inequality_l370_37010


namespace NUMINAMATH_CALUDE_kanul_cash_percentage_l370_37047

def total_amount : ℝ := 5555.56
def raw_materials_cost : ℝ := 3000
def machinery_cost : ℝ := 2000

theorem kanul_cash_percentage :
  (total_amount - (raw_materials_cost + machinery_cost)) / total_amount * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_kanul_cash_percentage_l370_37047


namespace NUMINAMATH_CALUDE_whitewashing_cost_is_6770_l370_37058

/-- Calculate the cost of white washing a room with given dimensions and openings. -/
def whitewashingCost (roomLength roomWidth roomHeight : ℝ)
                     (doorLength doorWidth : ℝ)
                     (windowLength windowWidth : ℝ)
                     (numDoors numWindows : ℕ)
                     (costPerSqFt : ℝ) : ℝ :=
  let wallArea := 2 * (roomLength * roomHeight + roomWidth * roomHeight)
  let doorArea := numDoors * (doorLength * doorWidth)
  let windowArea := numWindows * (windowLength * windowWidth)
  let paintableArea := wallArea - doorArea - windowArea
  paintableArea * costPerSqFt

/-- Theorem stating that the cost of white washing the room with given specifications is 6770 Rs. -/
theorem whitewashing_cost_is_6770 :
  whitewashingCost 30 20 15 7 4 5 3 2 6 5 = 6770 := by
  sorry

end NUMINAMATH_CALUDE_whitewashing_cost_is_6770_l370_37058


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l370_37031

/-- 
Given a positive integer n, we define two properties:
1. 5n is a perfect square
2. 7n is a perfect cube

This theorem states that 1225 is the smallest positive integer satisfying both properties.
-/
theorem smallest_n_square_and_cube : ∀ n : ℕ+, 
  (∃ k : ℕ+, 5 * n = k^2) ∧ 
  (∃ m : ℕ+, 7 * n = m^3) → 
  n ≥ 1225 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l370_37031


namespace NUMINAMATH_CALUDE_writing_utensils_arrangement_l370_37085

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def circular_permutations (n a b c d : ℕ) : ℕ :=
  factorial (n - 1) / (factorial a * factorial b * factorial c * factorial d)

def adjacent_arrangements (n a b c d : ℕ) : ℕ :=
  circular_permutations (n - 1) a 1 c d

theorem writing_utensils_arrangement :
  let total_items : ℕ := 5 + 3 + 1 + 1
  let black_pencils : ℕ := 5
  let blue_pens : ℕ := 3
  let red_pen : ℕ := 1
  let green_pen : ℕ := 1
  circular_permutations total_items black_pencils blue_pens red_pen green_pen -
  adjacent_arrangements total_items black_pencils blue_pens red_pen green_pen = 168 := by
sorry

end NUMINAMATH_CALUDE_writing_utensils_arrangement_l370_37085


namespace NUMINAMATH_CALUDE_alternating_squares_sum_l370_37037

theorem alternating_squares_sum : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 221 := by
  sorry

end NUMINAMATH_CALUDE_alternating_squares_sum_l370_37037


namespace NUMINAMATH_CALUDE_limit_of_function_at_one_l370_37032

theorem limit_of_function_at_one :
  let f : ℝ → ℝ := λ x ↦ 2 * x - 3 - 1 / x
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → |f x - (-2)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_function_at_one_l370_37032


namespace NUMINAMATH_CALUDE_inequality_system_solution_l370_37039

theorem inequality_system_solution (x : ℝ) :
  (2 * x + 1 > x + 3 ∧ 2 * x - 4 < x) ↔ (2 < x ∧ x < 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l370_37039


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_72_l370_37027

theorem sqrt_18_times_sqrt_72 : Real.sqrt 18 * Real.sqrt 72 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_72_l370_37027


namespace NUMINAMATH_CALUDE_circle_center_condition_l370_37035

-- Define the circle equation
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0

-- Define the condition for the center to be in the third quadrant
def center_in_third_quadrant (k : ℝ) : Prop :=
  k > 0 ∧ -2 < 0

-- Define the range of k
def k_range (k : ℝ) : Prop :=
  k > 4

-- Theorem statement
theorem circle_center_condition (k : ℝ) :
  (∃ x y : ℝ, circle_equation x y k) ∧ 
  center_in_third_quadrant k →
  k_range k :=
by sorry

end NUMINAMATH_CALUDE_circle_center_condition_l370_37035


namespace NUMINAMATH_CALUDE_ken_kept_pencils_l370_37064

def pencil_problem (initial_pencils : ℕ) (given_to_manny : ℕ) (extra_to_nilo : ℕ) : Prop :=
  let given_to_nilo : ℕ := given_to_manny + extra_to_nilo
  let total_given : ℕ := given_to_manny + given_to_nilo
  let kept : ℕ := initial_pencils - total_given
  kept = 20

theorem ken_kept_pencils :
  pencil_problem 50 10 10 :=
sorry

end NUMINAMATH_CALUDE_ken_kept_pencils_l370_37064


namespace NUMINAMATH_CALUDE_heptagon_coloring_l370_37081

-- Define the color type
inductive Color
| Red
| Blue
| Yellow
| Green

-- Define the heptagon type
def Heptagon := Fin 7 → Color

-- Define the coloring conditions
def validColoring (h : Heptagon) : Prop :=
  ∀ i : Fin 7,
    (h i = Color.Red ∨ h i = Color.Blue →
      h ((i + 1) % 7) ≠ Color.Blue ∧ h ((i + 1) % 7) ≠ Color.Green ∧
      h ((i + 4) % 7) ≠ Color.Blue ∧ h ((i + 4) % 7) ≠ Color.Green) ∧
    (h i = Color.Yellow ∨ h i = Color.Green →
      h ((i + 1) % 7) ≠ Color.Red ∧ h ((i + 1) % 7) ≠ Color.Yellow ∧
      h ((i + 4) % 7) ≠ Color.Red ∧ h ((i + 4) % 7) ≠ Color.Yellow)

-- Theorem statement
theorem heptagon_coloring (h : Heptagon) (hvalid : validColoring h) :
  ∃ c : Color, ∀ i : Fin 7, h i = c :=
sorry

end NUMINAMATH_CALUDE_heptagon_coloring_l370_37081


namespace NUMINAMATH_CALUDE_square_difference_fifty_fortynine_l370_37053

theorem square_difference_fifty_fortynine : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fifty_fortynine_l370_37053
