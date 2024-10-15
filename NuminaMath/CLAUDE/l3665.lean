import Mathlib

namespace NUMINAMATH_CALUDE_carl_responsibility_l3665_366506

/-- Calculates the amount a person owes in an accident based on their fault percentage and insurance coverage -/
def calculate_personal_responsibility (total_property_damage : ℝ) (total_medical_bills : ℝ)
  (property_insurance_coverage : ℝ) (medical_insurance_coverage : ℝ) (fault_percentage : ℝ) : ℝ :=
  let remaining_property_damage := total_property_damage * (1 - property_insurance_coverage)
  let remaining_medical_bills := total_medical_bills * (1 - medical_insurance_coverage)
  fault_percentage * (remaining_property_damage + remaining_medical_bills)

/-- Theorem stating Carl's personal responsibility in the accident -/
theorem carl_responsibility :
  let total_property_damage : ℝ := 40000
  let total_medical_bills : ℝ := 70000
  let property_insurance_coverage : ℝ := 0.8
  let medical_insurance_coverage : ℝ := 0.75
  let carl_fault_percentage : ℝ := 0.6
  calculate_personal_responsibility total_property_damage total_medical_bills
    property_insurance_coverage medical_insurance_coverage carl_fault_percentage = 15300 := by
  sorry

end NUMINAMATH_CALUDE_carl_responsibility_l3665_366506


namespace NUMINAMATH_CALUDE_smallest_n_fourth_root_l3665_366593

theorem smallest_n_fourth_root (n : ℕ) : n = 4097 ↔ 
  (n > 0 ∧ 
   ∀ m : ℕ, m > 0 → m < n → 
   ¬(0 < (m : ℝ)^(1/4) - ⌊(m : ℝ)^(1/4)⌋ ∧ (m : ℝ)^(1/4) - ⌊(m : ℝ)^(1/4)⌋ < 1/2015)) ∧
  (0 < (n : ℝ)^(1/4) - ⌊(n : ℝ)^(1/4)⌋ ∧ (n : ℝ)^(1/4) - ⌊(n : ℝ)^(1/4)⌋ < 1/2015) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_fourth_root_l3665_366593


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_two_range_of_a_for_all_real_solution_l3665_366583

def f (a x : ℝ) : ℝ := a * x^2 + a * x - 1

theorem solution_set_when_a_is_two :
  let a := 2
  {x : ℝ | f a x < 0} = {x : ℝ | -(1 + Real.sqrt 3) / 2 < x ∧ x < (-1 + Real.sqrt 3) / 2} := by sorry

theorem range_of_a_for_all_real_solution :
  {a : ℝ | ∀ x, f a x < 0} = {a : ℝ | -4 < a ∧ a ≤ 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_two_range_of_a_for_all_real_solution_l3665_366583


namespace NUMINAMATH_CALUDE_linear_functions_inequality_l3665_366569

theorem linear_functions_inequality (k : ℝ) :
  (∀ x > -1, k * x - 2 < 2 * x + 3) →
  -3 ≤ k ∧ k ≤ 2 ∧ k ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_linear_functions_inequality_l3665_366569


namespace NUMINAMATH_CALUDE_part_one_part_two_l3665_366556

-- Part 1
theorem part_one (a b : ℝ) (h1 : 1 ≤ a - b) (h2 : a - b ≤ 2) (h3 : 2 ≤ a + b) (h4 : a + b ≤ 4) :
  5 ≤ 4*a - 2*b ∧ 4*a - 2*b ≤ 10 := by sorry

-- Part 2
theorem part_two (m : ℝ) (h : ∀ x > (1/2 : ℝ), 2*x^2 - x ≥ 2*m*x - m - 8) :
  m ≤ 9/2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3665_366556


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l3665_366512

theorem complex_magnitude_product : 
  Complex.abs ((5 * Real.sqrt 2 - Complex.I * 3) * (2 * Real.sqrt 3 + Complex.I * 4)) = 2 * Real.sqrt 413 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l3665_366512


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l3665_366525

theorem purely_imaginary_condition (a : ℝ) : 
  (∃ (y : ℝ), Complex.mk (a^2 - 4) (a + 2) = Complex.I * y) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l3665_366525


namespace NUMINAMATH_CALUDE_M_intersect_N_l3665_366590

def M : Set ℕ := {1, 2, 4, 8}

def N : Set ℕ := {x : ℕ | ∃ k : ℕ, x = 2 * k}

theorem M_intersect_N : M ∩ N = {2, 4, 8} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l3665_366590


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3665_366552

theorem diophantine_equation_solutions :
  ∀ m n : ℤ, m ≠ 0 ∧ n ≠ 0 →
  (m^2 + n) * (m + n^2) = (m - n)^3 ↔
  (m = -1 ∧ n = -1) ∨ (m = 8 ∧ n = -10) ∨ (m = 9 ∧ n = -6) ∨ (m = 9 ∧ n = -21) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3665_366552


namespace NUMINAMATH_CALUDE_symmetric_point_l3665_366503

/-- Given a point (a, b) and a line x + y + 1 = 0, the point symmetric to (a, b) with respect to the line is (-b-1, -a-1) -/
theorem symmetric_point (a b : ℝ) : 
  let original_point := (a, b)
  let line_equation (x y : ℝ) := x + y + 1 = 0
  let symmetric_point := (-b - 1, -a - 1)
  ∀ x y, line_equation x y → 
    (x - a) ^ 2 + (y - b) ^ 2 = (x - (-b - 1)) ^ 2 + (y - (-a - 1)) ^ 2 ∧
    line_equation ((a + (-b - 1)) / 2) ((b + (-a - 1)) / 2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_l3665_366503


namespace NUMINAMATH_CALUDE_contrapositive_even_sum_l3665_366508

theorem contrapositive_even_sum (x y : ℤ) :
  (¬(Even (x + y)) → ¬(Even x ∧ Even y)) ↔
  (∀ x y : ℤ, Even x ∧ Even y → Even (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_even_sum_l3665_366508


namespace NUMINAMATH_CALUDE_linear_control_periodic_bound_l3665_366565

/-- A function f: ℝ → ℝ is a linear control function if |f'(x)| ≤ 1 for all x ∈ ℝ -/
def LinearControlFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, DifferentiableAt ℝ f x ∧ |deriv f x| ≤ 1

/-- A function f: ℝ → ℝ is periodic with period T if f(x + T) = f(x) for all x ∈ ℝ -/
def Periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

theorem linear_control_periodic_bound 
    (f : ℝ → ℝ) (T : ℝ) 
    (h1 : LinearControlFunction f)
    (h2 : StrictMono f)
    (h3 : Periodic f T) :
    ∀ x₁ x₂ : ℝ, |f x₁ - f x₂| ≤ T := by
  sorry


end NUMINAMATH_CALUDE_linear_control_periodic_bound_l3665_366565


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3665_366597

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 4 * a * x + 4 * a = a * (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3665_366597


namespace NUMINAMATH_CALUDE_sprinter_probabilities_l3665_366561

/-- Probabilities of three independent events -/
def prob_A : ℚ := 2/5
def prob_B : ℚ := 3/4
def prob_C : ℚ := 1/3

/-- Probability of all three events occurring -/
def prob_all_three : ℚ := prob_A * prob_B * prob_C

/-- Probability of exactly two events occurring -/
def prob_two : ℚ := 
  prob_A * prob_B * (1 - prob_C) + 
  prob_A * (1 - prob_B) * prob_C + 
  (1 - prob_A) * prob_B * prob_C

/-- Probability of at least one event occurring -/
def prob_at_least_one : ℚ := 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

theorem sprinter_probabilities :
  prob_all_three = 1/10 ∧ 
  prob_two = 23/60 ∧ 
  prob_at_least_one = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_sprinter_probabilities_l3665_366561


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3665_366507

/-- Given a geometric sequence {a_n} with first term a₁ = 1 and positive common ratio q,
    if S₄ - 5S₂ = 0, then S₅ = 31. -/
theorem geometric_sequence_sum (q : ℝ) (hq : q > 0) : 
  let a : ℕ → ℝ := λ n => q^(n-1)
  let S : ℕ → ℝ := λ n => (1 - q^n) / (1 - q)
  (S 4 - 5 * S 2 = 0) → S 5 = 31 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3665_366507


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_set_A_forms_triangle_l3665_366559

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_theorem (a b c : ℝ) :
  can_form_triangle a b c ↔ (a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :=
sorry

theorem set_A_forms_triangle :
  can_form_triangle 8 6 5 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_set_A_forms_triangle_l3665_366559


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3665_366553

theorem sum_of_squares_of_roots (a b c : ℂ) : 
  (3 * a^3 - 2 * a^2 + 5 * a + 15 = 0) ∧ 
  (3 * b^3 - 2 * b^2 + 5 * b + 15 = 0) ∧ 
  (3 * c^3 - 2 * c^2 + 5 * c + 15 = 0) →
  a^2 + b^2 + c^2 = -26/9 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3665_366553


namespace NUMINAMATH_CALUDE_remainder_problem_l3665_366574

theorem remainder_problem : ∃ x : ℕ, (71 * x) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3665_366574


namespace NUMINAMATH_CALUDE_equation_solution_l3665_366531

theorem equation_solution : ∃! x : ℚ, 2 * (x - 1) = 2 - (5 * x - 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3665_366531


namespace NUMINAMATH_CALUDE_two_digit_numbers_from_123_l3665_366521

def Digits : Set Nat := {1, 2, 3}

def TwoDigitNumber (n : Nat) : Prop :=
  n ≥ 10 ∧ n ≤ 99

def FormedFromDigits (n : Nat) : Prop :=
  ∃ (tens units : Nat), tens ∈ Digits ∧ units ∈ Digits ∧ n = 10 * tens + units

theorem two_digit_numbers_from_123 :
  {n : Nat | TwoDigitNumber n ∧ FormedFromDigits n} =
  {11, 12, 13, 21, 22, 23, 31, 32, 33} := by sorry

end NUMINAMATH_CALUDE_two_digit_numbers_from_123_l3665_366521


namespace NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_l3665_366546

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1: Two planes perpendicular to the same line are parallel
theorem planes_perpendicular_to_line_are_parallel 
  (l : Line) (p1 p2 : Plane) 
  (h1 : perpendicular_plane_line p1 l) 
  (h2 : perpendicular_plane_line p2 l) : 
  parallel_planes p1 p2 :=
sorry

-- Theorem 2: Two lines perpendicular to the same plane are parallel
theorem lines_perpendicular_to_plane_are_parallel 
  (p : Plane) (l1 l2 : Line) 
  (h1 : perpendicular_line_plane l1 p) 
  (h2 : perpendicular_line_plane l2 p) : 
  parallel_lines l1 l2 :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_l3665_366546


namespace NUMINAMATH_CALUDE_pete_has_enough_money_l3665_366534

/-- Represents the amount of money Pete has and owes -/
structure PetesMoney where
  wallet_twenty : Nat -- number of $20 bills in wallet
  wallet_ten : Nat -- number of $10 bills in wallet
  wallet_pounds : Nat -- number of £5 notes in wallet
  pocket_ten : Nat -- number of $10 bills in pocket
  owed : Nat -- amount owed on the bike in dollars

/-- Calculates the total amount of money Pete has in dollars -/
def total_money (m : PetesMoney) : Nat :=
  m.wallet_twenty * 20 + m.wallet_ten * 10 + m.wallet_pounds * 7 + m.pocket_ten * 10

/-- Proves that Pete has enough money to pay off his bike debt -/
theorem pete_has_enough_money (m : PetesMoney) 
  (h1 : m.wallet_twenty = 2)
  (h2 : m.wallet_ten = 1)
  (h3 : m.wallet_pounds = 1)
  (h4 : m.pocket_ten = 4)
  (h5 : m.owed = 90) :
  total_money m ≥ m.owed :=
by sorry

end NUMINAMATH_CALUDE_pete_has_enough_money_l3665_366534


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l3665_366581

theorem complex_exponential_sum (α β θ : ℝ) :
  Complex.exp (Complex.I * (α + θ)) + Complex.exp (Complex.I * (β + θ)) = (1/3 : ℂ) + (4/9 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * (α + θ)) + Complex.exp (-Complex.I * (β + θ)) = (1/3 : ℂ) - (4/9 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l3665_366581


namespace NUMINAMATH_CALUDE_prob_event_a_is_one_third_l3665_366575

/-- Represents a glove --/
inductive Glove
| Left : Glove
| Right : Glove

/-- Represents a color --/
inductive Color
| Red : Color
| Blue : Color
| Yellow : Color

/-- Represents a pair of gloves --/
def GlovePair := Color × Glove × Glove

/-- The set of all possible glove pairs --/
def allGlovePairs : Finset GlovePair :=
  sorry

/-- The event of selecting two gloves --/
def twoGloveSelection := GlovePair × GlovePair

/-- The event of selecting one left and one right glove of different colors --/
def eventA : Set twoGloveSelection :=
  sorry

/-- The probability of event A --/
def probEventA : ℚ :=
  sorry

theorem prob_event_a_is_one_third :
  probEventA = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_prob_event_a_is_one_third_l3665_366575


namespace NUMINAMATH_CALUDE_divisible_by_five_l3665_366545

theorem divisible_by_five (n : ℕ) : 5 ∣ (2^(4*n+1) + 3^(4*n+1)) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l3665_366545


namespace NUMINAMATH_CALUDE_mary_earnings_per_home_l3665_366540

/-- Given that Mary earned $12696 cleaning 276.0 homes, prove that she earns $46 per home. -/
theorem mary_earnings_per_home :
  let total_earnings : ℚ := 12696
  let homes_cleaned : ℚ := 276
  total_earnings / homes_cleaned = 46 := by
  sorry

end NUMINAMATH_CALUDE_mary_earnings_per_home_l3665_366540


namespace NUMINAMATH_CALUDE_exponential_function_sum_of_extrema_l3665_366595

/-- Given an exponential function y = a^x, if the sum of its maximum and minimum values 
    on the interval [0,1] is 3, then a = 2 -/
theorem exponential_function_sum_of_extrema (a : ℝ) : 
  (a > 0) → 
  (∀ x ∈ Set.Icc 0 1, ∃ y, y = a^x) →
  (Real.exp a + 1 = 3 ∨ a + Real.exp a = 3) →
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_sum_of_extrema_l3665_366595


namespace NUMINAMATH_CALUDE_largest_special_square_proof_l3665_366535

/-- A number is a perfect square if it's the square of an integer -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- Remove the last two digits of a natural number -/
def remove_last_two_digits (n : ℕ) : ℕ := n / 100

/-- The largest perfect square satisfying the given conditions -/
def largest_special_square : ℕ := 1681

theorem largest_special_square_proof :
  (is_perfect_square largest_special_square) ∧ 
  (is_perfect_square (remove_last_two_digits largest_special_square)) ∧ 
  (largest_special_square % 100 ≠ 0) ∧
  (∀ n : ℕ, n > largest_special_square → 
    ¬(is_perfect_square n ∧ 
      is_perfect_square (remove_last_two_digits n) ∧ 
      n % 100 ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_largest_special_square_proof_l3665_366535


namespace NUMINAMATH_CALUDE_beef_weight_loss_percentage_l3665_366527

/-- Calculates the percentage of weight lost during processing of a side of beef. -/
theorem beef_weight_loss_percentage 
  (initial_weight : ℝ) 
  (processed_weight : ℝ) 
  (h1 : initial_weight = 892.31)
  (h2 : processed_weight = 580) : 
  ∃ (percentage : ℝ), abs (percentage - 34.99) < 0.01 ∧ 
  percentage = (initial_weight - processed_weight) / initial_weight * 100 :=
sorry

end NUMINAMATH_CALUDE_beef_weight_loss_percentage_l3665_366527


namespace NUMINAMATH_CALUDE_family_travel_distance_l3665_366578

/-- Proves that the total distance travelled is 448 km given the specified conditions --/
theorem family_travel_distance : 
  ∀ (total_distance : ℝ),
  (total_distance / (2 * 35) + total_distance / (2 * 40) = 12) →
  total_distance = 448 := by
sorry

end NUMINAMATH_CALUDE_family_travel_distance_l3665_366578


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3665_366570

theorem consecutive_integers_sum (n : ℤ) : 
  n * (n + 1) = 20412 → n + (n + 1) = 287 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3665_366570


namespace NUMINAMATH_CALUDE_parabola_point_distance_l3665_366520

/-- Given a point P(a,0) and a parabola y^2 = 4x, if for every point Q on the parabola |PQ| ≥ |a|, then a ≤ 2 -/
theorem parabola_point_distance (a : ℝ) : 
  (∀ x y : ℝ, y^2 = 4*x → ((x - a)^2 + y^2 ≥ a^2)) → 
  a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l3665_366520


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3665_366543

theorem trigonometric_simplification :
  let numerator := Real.sin (15 * π / 180) + Real.sin (25 * π / 180) + Real.sin (35 * π / 180) + 
                   Real.sin (45 * π / 180) + Real.sin (55 * π / 180) + Real.sin (65 * π / 180) + 
                   Real.sin (75 * π / 180) + Real.sin (85 * π / 180)
  let denominator := Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)
  numerator / denominator = (16 * Real.sin (50 * π / 180) * Real.cos (20 * π / 180)) / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3665_366543


namespace NUMINAMATH_CALUDE_missing_number_proof_l3665_366592

theorem missing_number_proof (known_numbers : List ℕ) (mean : ℚ) : 
  known_numbers = [1, 23, 24, 25, 26, 27, 2] ∧ 
  mean = 20 ∧ 
  (List.sum known_numbers + 32) / 8 = mean →
  32 = 8 * mean - List.sum known_numbers :=
by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l3665_366592


namespace NUMINAMATH_CALUDE_assignment_calculations_l3665_366547

/-- Represents the number of volunteers -/
def num_volunteers : ℕ := 5

/-- Represents the number of communities -/
def num_communities : ℕ := 4

/-- Total number of assignment schemes -/
def total_assignments : ℕ := num_communities ^ num_volunteers

/-- Number of assignments with restrictions on community A and minimum volunteers -/
def restricted_assignments : ℕ := 150

/-- Number of assignments with each community having at least one volunteer and two specific volunteers not in the same community -/
def specific_restricted_assignments : ℕ := 216

/-- Theorem stating the correctness of the assignment calculations -/
theorem assignment_calculations :
  (total_assignments = 1024) ∧
  (restricted_assignments = 150) ∧
  (specific_restricted_assignments = 216) := by sorry

end NUMINAMATH_CALUDE_assignment_calculations_l3665_366547


namespace NUMINAMATH_CALUDE_inequality_solution_l3665_366518

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_increasing : ∀ x y, x < y → f x < f y
axiom f_point1 : f 0 = -2
axiom f_point2 : f 3 = 2

-- Define the solution set
def solution_set : Set ℝ := {x | x < -1 ∨ x ≥ 2}

-- Theorem statement
theorem inequality_solution :
  {x : ℝ | |f (x + 1)| ≥ 2} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3665_366518


namespace NUMINAMATH_CALUDE_student_in_first_vehicle_probability_l3665_366524

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of vehicles -/
def num_vehicles : ℕ := 2

/-- The number of seats in each vehicle -/
def seats_per_vehicle : ℕ := 2

/-- The probability that a specific student is in the first vehicle -/
def prob_student_in_first_vehicle : ℚ := 1/2

theorem student_in_first_vehicle_probability :
  prob_student_in_first_vehicle = 1/2 := by sorry

end NUMINAMATH_CALUDE_student_in_first_vehicle_probability_l3665_366524


namespace NUMINAMATH_CALUDE_people_lifting_weights_l3665_366539

/-- The number of people in the gym at the start of Bethany's shift -/
def initial_people : ℕ := sorry

/-- The number of people who arrived during Bethany's shift -/
def arrivals : ℕ := 5

/-- The number of people who left during Bethany's shift -/
def departures : ℕ := 2

/-- The total number of people in the gym after the changes -/
def final_people : ℕ := 19

theorem people_lifting_weights : initial_people = 16 :=
  by sorry

end NUMINAMATH_CALUDE_people_lifting_weights_l3665_366539


namespace NUMINAMATH_CALUDE_clubsuit_equation_solution_l3665_366586

/-- Definition of the clubsuit operation -/
def clubsuit (A B : ℝ) : ℝ := 4 * A + 3 * B + 7

/-- Theorem stating that A clubsuit 6 = 85 when A = 15 -/
theorem clubsuit_equation_solution :
  clubsuit 15 6 = 85 := by sorry

end NUMINAMATH_CALUDE_clubsuit_equation_solution_l3665_366586


namespace NUMINAMATH_CALUDE_semi_circle_radius_equals_rectangle_area_l3665_366591

theorem semi_circle_radius_equals_rectangle_area (length width : ℝ) (h1 : length = 8) (h2 : width = Real.pi) :
  ∃ (r : ℝ), r = 4 ∧ (1/2 * Real.pi * r^2) = (length * width) :=
by sorry

end NUMINAMATH_CALUDE_semi_circle_radius_equals_rectangle_area_l3665_366591


namespace NUMINAMATH_CALUDE_pond_length_l3665_366510

/-- The length of a rectangular pond given its width, depth, and volume. -/
theorem pond_length (width : ℝ) (depth : ℝ) (volume : ℝ) 
  (h_width : width = 12) 
  (h_depth : depth = 5) 
  (h_volume : volume = 1200) : 
  volume / (width * depth) = 20 := by
  sorry

end NUMINAMATH_CALUDE_pond_length_l3665_366510


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l3665_366582

/-- Given a parabola y = -x^2 + (k+1)x - k where (4,0) lies on the parabola,
    prove that the intersection point of the parabola with the y-axis is (0, -4). -/
theorem parabola_y_axis_intersection
  (k : ℝ)
  (h : 0 = -(4^2) + (k+1)*4 - k) :
  ∃ y, y = -4 ∧ 0 = -(0^2) + (k+1)*0 - k ∧ y = -(0^2) + (k+1)*0 - k :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l3665_366582


namespace NUMINAMATH_CALUDE_probability_ratio_l3665_366588

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 10

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The probability of drawing all five slips with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

/-- The probability of drawing three slips with one number and two slips with another number -/
def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 3 * Nat.choose slips_per_number 2 : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

theorem probability_ratio :
  q / p = 450 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_l3665_366588


namespace NUMINAMATH_CALUDE_g_range_l3665_366577

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos (x/3))^2 - 2*Real.pi * Real.arcsin (x/3) + (Real.arcsin (x/3))^2 + 
  (Real.pi^2/8) * (x^2 - 4*x + 12)

theorem g_range :
  ∀ x ∈ Set.Icc (-3 : ℝ) 3,
  g x ∈ Set.Icc (Real.pi^2/4 + 9*Real.pi^2/8) (Real.pi^2/4 + 33*Real.pi^2/8) :=
by sorry

end NUMINAMATH_CALUDE_g_range_l3665_366577


namespace NUMINAMATH_CALUDE_count_symmetric_patterns_l3665_366532

/-- A symmetric digital pattern on an 8x8 grid --/
structure SymmetricPattern :=
  (grid : Fin 8 → Fin 8 → Bool)
  (symmetric : ∀ (i j : Fin 8), grid i j = grid (7 - i) j ∧ grid i j = grid i (7 - j) ∧ grid i j = grid j i)
  (not_monochrome : ∃ (i j k l : Fin 8), grid i j ≠ grid k l)

/-- The number of symmetric regions in an 8x8 grid --/
def num_symmetric_regions : Nat := 12

/-- The total number of possible symmetric digital patterns --/
def total_symmetric_patterns : Nat := 2^num_symmetric_regions - 2

theorem count_symmetric_patterns :
  total_symmetric_patterns = 4094 :=
sorry

end NUMINAMATH_CALUDE_count_symmetric_patterns_l3665_366532


namespace NUMINAMATH_CALUDE_quality_difference_proof_l3665_366573

-- Define the data from the problem
def total_products : ℕ := 400
def machine_a_first_class : ℕ := 150
def machine_a_second_class : ℕ := 50
def machine_b_first_class : ℕ := 120
def machine_b_second_class : ℕ := 80

-- Define the K² formula
def k_squared (n a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the theorem
theorem quality_difference_proof :
  (machine_a_first_class : ℚ) / (machine_a_first_class + machine_a_second_class) = 3/4 ∧
  (machine_b_first_class : ℚ) / (machine_b_first_class + machine_b_second_class) = 3/5 ∧
  k_squared total_products machine_a_first_class machine_a_second_class machine_b_first_class machine_b_second_class > 6635/1000 :=
by sorry

end NUMINAMATH_CALUDE_quality_difference_proof_l3665_366573


namespace NUMINAMATH_CALUDE_silver_medals_count_l3665_366519

theorem silver_medals_count (total_medals gold_medals bronze_medals : ℕ) 
  (h1 : total_medals = 67)
  (h2 : gold_medals = 19)
  (h3 : bronze_medals = 16) :
  total_medals - gold_medals - bronze_medals = 32 := by
sorry

end NUMINAMATH_CALUDE_silver_medals_count_l3665_366519


namespace NUMINAMATH_CALUDE_triangle_properties_l3665_366529

-- Define the lines of triangle ABC
def line_AB (x y : ℝ) : Prop := 3 * x + 4 * y + 12 = 0
def line_BC (x y : ℝ) : Prop := 4 * x - 3 * y + 16 = 0
def line_CA (x y : ℝ) : Prop := 2 * x + y - 2 = 0

-- Define point B as the intersection of AB and BC
def point_B : ℝ × ℝ := (-4, 0)

-- Define the equation of the altitude from A to BC
def altitude_A_to_BC (x y : ℝ) : Prop := x - 2 * y + 4 = 0

theorem triangle_properties :
  (∀ x y : ℝ, line_AB x y ∧ line_BC x y → (x, y) = point_B) ∧
  (∀ x y : ℝ, altitude_A_to_BC x y ↔ 
    (∃ t : ℝ, x = t * (point_B.1 - (2 / 5)) ∧ 
              y = t * (point_B.2 + (1 / 5)) ∧
              2 * x + y - 2 = 0)) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3665_366529


namespace NUMINAMATH_CALUDE_negation_p_false_necessary_not_sufficient_l3665_366584

theorem negation_p_false_necessary_not_sufficient (p q : Prop) :
  (∃ (h : p ∧ q), ¬¬p) ∧ 
  (∃ (h : ¬¬p), ¬(p ∧ q)) := by
sorry

end NUMINAMATH_CALUDE_negation_p_false_necessary_not_sufficient_l3665_366584


namespace NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l3665_366558

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

/-- Proof of the specific bridge length problem -/
theorem specific_bridge_length :
  bridge_length 150 45 30 = 225 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l3665_366558


namespace NUMINAMATH_CALUDE_sqrt_8_is_quadratic_radical_l3665_366599

-- Define what a quadratic radical is
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), y ≥ 0 ∧ x = Real.sqrt y

-- Theorem statement
theorem sqrt_8_is_quadratic_radical :
  is_quadratic_radical (Real.sqrt 8) ∧
  ¬(∀ x : ℝ, is_quadratic_radical (Real.sqrt x)) ∧
  ¬(∀ m n : ℝ, is_quadratic_radical (Real.sqrt (m + n))) :=
sorry

end NUMINAMATH_CALUDE_sqrt_8_is_quadratic_radical_l3665_366599


namespace NUMINAMATH_CALUDE_beef_order_proof_l3665_366589

/-- Calculates the amount of beef ordered given the costs and total amount --/
def beef_ordered (beef_cost chicken_cost total_cost : ℚ) : ℚ :=
  total_cost / (beef_cost + 2 * chicken_cost)

/-- Proves that the amount of beef ordered is 1000 pounds given the problem conditions --/
theorem beef_order_proof :
  beef_ordered 8 3 14000 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_beef_order_proof_l3665_366589


namespace NUMINAMATH_CALUDE_a_less_than_two_l3665_366536

def A (a : ℝ) : Set ℝ := {x | x ≤ a}
def B : Set ℝ := Set.Iio 2

theorem a_less_than_two (a : ℝ) (h : A a ⊆ B) : a < 2 := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_two_l3665_366536


namespace NUMINAMATH_CALUDE_min_value_theorem_l3665_366501

theorem min_value_theorem (a b : ℝ) (h1 : b > 0) (h2 : a + b = 2) :
  (1 / (2 * |a|)) + (|a| / b) ≥ 3/4 ∧ 
  ∃ (a₀ b₀ : ℝ), b₀ > 0 ∧ a₀ + b₀ = 2 ∧ (1 / (2 * |a₀|)) + (|a₀| / b₀) = 3/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3665_366501


namespace NUMINAMATH_CALUDE_equal_amount_after_15_days_l3665_366568

/-- The number of days it takes for Minjeong's and Soohyeok's piggy bank amounts to become equal -/
def days_to_equal_amount (minjeong_initial : ℕ) (soohyeok_initial : ℕ) 
                         (minjeong_daily : ℕ) (soohyeok_daily : ℕ) : ℕ :=
  15

theorem equal_amount_after_15_days 
  (minjeong_initial : ℕ) (soohyeok_initial : ℕ) 
  (minjeong_daily : ℕ) (soohyeok_daily : ℕ)
  (h1 : minjeong_initial = 8000)
  (h2 : soohyeok_initial = 5000)
  (h3 : minjeong_daily = 300)
  (h4 : soohyeok_daily = 500) :
  minjeong_initial + 15 * minjeong_daily = soohyeok_initial + 15 * soohyeok_daily :=
by
  sorry

#eval days_to_equal_amount 8000 5000 300 500

end NUMINAMATH_CALUDE_equal_amount_after_15_days_l3665_366568


namespace NUMINAMATH_CALUDE_grid_line_count_l3665_366511

/-- Represents a point in the grid -/
structure Point where
  x : Fin 50
  y : Fin 50

/-- Represents the color of a point -/
inductive Color
  | Blue
  | Red

/-- Represents the color of a line segment -/
inductive LineColor
  | Blue
  | Red
  | Black

/-- The coloring of the grid -/
def grid_coloring : Point → Color := sorry

/-- The number of blue points in the grid -/
def num_blue_points : Nat := 1510

/-- The number of blue points on the edge of the grid -/
def num_blue_edge_points : Nat := 110

/-- The number of red line segments in the grid -/
def num_red_lines : Nat := 947

/-- Checks if a point is on the edge of the grid -/
def is_edge_point (p : Point) : Bool := 
  p.x = 0 || p.x = 49 || p.y = 0 || p.y = 49

/-- Checks if a point is at a corner of the grid -/
def is_corner_point (p : Point) : Bool :=
  (p.x = 0 && p.y = 0) || (p.x = 0 && p.y = 49) || 
  (p.x = 49 && p.y = 0) || (p.x = 49 && p.y = 49)

/-- The main theorem to prove -/
theorem grid_line_count : 
  (∀ p : Point, is_corner_point p → grid_coloring p = Color.Red) →
  (∃ edge_blue_points : Finset Point, 
    edge_blue_points.card = num_blue_edge_points ∧
    ∀ p ∈ edge_blue_points, is_edge_point p ∧ grid_coloring p = Color.Blue) →
  (∃ black_lines blue_lines : Nat, 
    black_lines = 1972 ∧ 
    blue_lines = 1981 ∧
    black_lines + blue_lines + num_red_lines = 50 * 49 * 2) :=
by sorry

end NUMINAMATH_CALUDE_grid_line_count_l3665_366511


namespace NUMINAMATH_CALUDE_work_payment_theorem_l3665_366504

/-- Represents the time (in days) it takes for a person to complete the work alone -/
structure WorkTime where
  days : ℚ
  days_pos : days > 0

/-- Represents a worker with their work time and share of payment -/
structure Worker where
  work_time : WorkTime
  share : ℚ
  share_nonneg : share ≥ 0

/-- Calculates the total payment for a job given two workers' information -/
def total_payment (worker1 worker2 : Worker) : ℚ :=
  let total_work_rate := 1 / worker1.work_time.days + 1 / worker2.work_time.days
  let total_parts := total_work_rate * worker1.work_time.days * worker2.work_time.days
  let worker1_parts := worker2.work_time.days
  worker1.share * total_parts / worker1_parts

/-- The main theorem stating the total payment for the work -/
theorem work_payment_theorem (rahul rajesh : Worker) 
    (h1 : rahul.work_time.days = 3)
    (h2 : rajesh.work_time.days = 2)
    (h3 : rahul.share = 900) :
    total_payment rahul rajesh = 2250 := by
  sorry


end NUMINAMATH_CALUDE_work_payment_theorem_l3665_366504


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3665_366514

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem stating the property of the geometric sequence -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) 
    (h_prod : a 1 * a 7 * a 13 = 8) : 
    a 3 * a 11 = 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l3665_366514


namespace NUMINAMATH_CALUDE_last_digit_alternating_factorial_sum_2014_l3665_366548

def alternatingFactorialSum (n : ℕ) : ℤ :=
  (List.range n).foldl (fun acc i => acc + (if i % 2 = 0 then 1 else -1) * (i + 1).factorial) 0

theorem last_digit_alternating_factorial_sum_2014 :
  (alternatingFactorialSum 2014) % 10 = 1 := by sorry

end NUMINAMATH_CALUDE_last_digit_alternating_factorial_sum_2014_l3665_366548


namespace NUMINAMATH_CALUDE_green_balls_count_l3665_366522

theorem green_balls_count (total : ℕ) (p : ℚ) (h1 : total = 12) (h2 : p = 1 / 22) : 
  ∃ (green : ℕ), green ≤ total ∧ (green * (green - 1) : ℚ) / (total * (total - 1)) = p :=
by
  sorry

#check green_balls_count

end NUMINAMATH_CALUDE_green_balls_count_l3665_366522


namespace NUMINAMATH_CALUDE_james_seed_planting_l3665_366572

/-- Calculates the percentage of seeds planted -/
def percentage_planted (original_trees : ℕ) (plants_per_tree : ℕ) (seeds_per_plant : ℕ) (new_trees : ℕ) : ℚ :=
  (new_trees : ℚ) / ((original_trees * plants_per_tree * seeds_per_plant) : ℚ) * 100

/-- Proves that the percentage of seeds planted is 60% given the problem conditions -/
theorem james_seed_planting :
  let original_trees : ℕ := 2
  let plants_per_tree : ℕ := 20
  let seeds_per_plant : ℕ := 1
  let new_trees : ℕ := 24
  percentage_planted original_trees plants_per_tree seeds_per_plant new_trees = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_seed_planting_l3665_366572


namespace NUMINAMATH_CALUDE_collatz_3_reaches_421_cycle_l3665_366515

-- Define the operation for a single step
def collatzStep (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

-- Define the sequence of Collatz numbers starting from n
def collatzSequence (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => collatzStep (collatzSequence n k)

-- Theorem stating that the Collatz sequence starting from 3 eventually reaches the cycle 4, 2, 1
theorem collatz_3_reaches_421_cycle :
  ∃ k : ℕ, ∃ m : ℕ, m ≥ k ∧
    (collatzSequence 3 m = 4 ∧
     collatzSequence 3 (m + 1) = 2 ∧
     collatzSequence 3 (m + 2) = 1 ∧
     collatzSequence 3 (m + 3) = 4) :=
sorry

end NUMINAMATH_CALUDE_collatz_3_reaches_421_cycle_l3665_366515


namespace NUMINAMATH_CALUDE_cube_rotation_theorem_l3665_366544

/-- Represents a cube with natural numbers on each face -/
structure Cube where
  front : ℕ
  right : ℕ
  back : ℕ
  left : ℕ
  top : ℕ
  bottom : ℕ

/-- Theorem stating the properties of the cube and the results to be proved -/
theorem cube_rotation_theorem (c : Cube) 
  (h1 : c.front + c.right + c.top = 42)
  (h2 : c.right + c.top + c.back = 34)
  (h3 : c.top + c.back + c.left = 53)
  (h4 : c.bottom = 6) :
  (c.left + c.front + c.top = 61) ∧ 
  (c.front + c.right + c.back + c.left + c.top + c.bottom ≤ 100) := by
  sorry


end NUMINAMATH_CALUDE_cube_rotation_theorem_l3665_366544


namespace NUMINAMATH_CALUDE_B_subset_A_l3665_366554

def A : Set ℝ := {x | x ≥ 1}
def B : Set ℝ := {x | x > 2}

theorem B_subset_A : B ⊆ A := by sorry

end NUMINAMATH_CALUDE_B_subset_A_l3665_366554


namespace NUMINAMATH_CALUDE_distance_between_rectangle_vertices_l3665_366542

/-- Given an acute-angled triangle ABC with AB = √3, AC = 1, and angle BAC = 60°,
    and equal rectangles AMNB and APQC built outward on sides AB and AC respectively,
    the distance between vertices N and Q is 2√(2 + √3). -/
theorem distance_between_rectangle_vertices (A B C M N P Q : ℝ × ℝ) :
  let AB := Real.sqrt 3
  let AC := 1
  let angle_BAC := 60 * π / 180
  -- Triangle ABC properties
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = AB^2 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = AC^2 →
  (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = AB * AC * Real.cos angle_BAC →
  -- Rectangle properties
  (M.1 - A.1)^2 + (M.2 - A.2)^2 = (P.1 - A.1)^2 + (P.2 - A.2)^2 →
  (N.1 - B.1)^2 + (N.2 - B.2)^2 = (Q.1 - C.1)^2 + (Q.2 - C.2)^2 →
  (M.1 - A.1) * (B.1 - A.1) + (M.2 - A.2) * (B.2 - A.2) = 0 →
  (P.1 - A.1) * (C.1 - A.1) + (P.2 - A.2) * (C.2 - A.2) = 0 →
  -- Conclusion
  (N.1 - Q.1)^2 + (N.2 - Q.2)^2 = 4 * (2 + Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_distance_between_rectangle_vertices_l3665_366542


namespace NUMINAMATH_CALUDE_y_equals_five_l3665_366564

/-- Configuration of numbers in a triangular arrangement -/
structure NumberTriangle where
  y : ℝ
  z : ℝ
  second_row : ℝ
  third_row : ℝ
  h1 : second_row = y * 10
  h2 : third_row = second_row * z

/-- The value of y in the given configuration is 5 -/
theorem y_equals_five (t : NumberTriangle) (h3 : t.second_row = 50) (h4 : t.third_row = 300) : t.y = 5 := by
  sorry


end NUMINAMATH_CALUDE_y_equals_five_l3665_366564


namespace NUMINAMATH_CALUDE_square_perimeter_l3665_366585

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 625) (h2 : side * side = area) : 
  4 * side = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3665_366585


namespace NUMINAMATH_CALUDE_rice_pricing_problem_l3665_366566

/-- Represents the linear relationship between price and quantity sold --/
def quantity_sold (x : ℝ) : ℝ := -50 * x + 1200

/-- Represents the profit function --/
def profit (x : ℝ) : ℝ := (x - 4) * (quantity_sold x)

/-- Theorem stating the main results of the problem --/
theorem rice_pricing_problem 
  (x : ℝ) 
  (h1 : 4 ≤ x ∧ x ≤ 7) :
  (∃ x, profit x = 1800 ∧ x = 6) ∧
  (∀ y, 4 ≤ y ∧ y ≤ 7 → profit y ≤ profit 7) ∧
  profit 7 = 2550 := by
  sorry


end NUMINAMATH_CALUDE_rice_pricing_problem_l3665_366566


namespace NUMINAMATH_CALUDE_smallest_ending_nine_div_thirteen_l3665_366557

/-- A function that checks if a number ends with 9 -/
def endsWithNine (n : ℕ) : Prop := n % 10 = 9

/-- The theorem stating that 169 is the smallest positive integer ending in 9 and divisible by 13 -/
theorem smallest_ending_nine_div_thirteen :
  ∀ n : ℕ, n > 0 → endsWithNine n → n % 13 = 0 → n ≥ 169 :=
by sorry

end NUMINAMATH_CALUDE_smallest_ending_nine_div_thirteen_l3665_366557


namespace NUMINAMATH_CALUDE_unique_ticket_number_l3665_366523

def is_valid_ticket (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  let x := n / 100
  let y := (n / 10) % 10
  let z := n % 10
  n = (22 * x + 22 * y + 22 * z) / 2

theorem unique_ticket_number : ∃! n : ℕ, is_valid_ticket n ∧ n = 198 := by
  sorry

end NUMINAMATH_CALUDE_unique_ticket_number_l3665_366523


namespace NUMINAMATH_CALUDE_ancient_chinese_car_problem_l3665_366517

/-- The number of cars in the ancient Chinese problem -/
def num_cars : ℕ := 15

/-- The number of people that can be accommodated when 3 people share a car -/
def people_three_per_car (x : ℕ) : ℕ := 3 * (x - 2)

/-- The number of people that can be accommodated when 2 people share a car -/
def people_two_per_car (x : ℕ) : ℕ := 2 * x

theorem ancient_chinese_car_problem :
  (people_three_per_car num_cars = people_two_per_car num_cars + 9) ∧
  (num_cars > 2) := by
  sorry

end NUMINAMATH_CALUDE_ancient_chinese_car_problem_l3665_366517


namespace NUMINAMATH_CALUDE_completing_square_solution_l3665_366579

theorem completing_square_solution (x : ℝ) : 
  (x^2 + 4*x + 1 = 0) ↔ ((x + 2)^2 = 3) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_solution_l3665_366579


namespace NUMINAMATH_CALUDE_renovation_project_materials_l3665_366598

/-- The total number of truck-loads of material needed for a renovation project -/
theorem renovation_project_materials :
  let sand := 0.16666666666666666 * Real.pi
  let dirt := 0.3333333333333333 * Real.exp 1
  let cement := 0.16666666666666666 * Real.sqrt 2
  let gravel := 0.25 * Real.log 5
  abs ((sand + dirt + cement + gravel) - 1.8401374808985008) < 1e-10 := by
sorry

end NUMINAMATH_CALUDE_renovation_project_materials_l3665_366598


namespace NUMINAMATH_CALUDE_sine_cosine_relation_l3665_366533

theorem sine_cosine_relation (θ : Real) (x : Real) 
  (h1 : 0 < θ ∧ θ < Real.pi / 2) 
  (h2 : x > 1) 
  (h3 : Real.cos (θ / 2) = Real.sqrt ((x + 1) / (2 * x))) : 
  Real.sin θ = Real.sqrt (x^2 - 1) / x := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_relation_l3665_366533


namespace NUMINAMATH_CALUDE_tethered_unicorn_sum_l3665_366509

/-- Represents the configuration of a unicorn tethered to a cylindrical tower. -/
structure TetheredUnicorn where
  towerRadius : ℝ
  ropeLength : ℝ
  unicornHeight : ℝ
  distanceFromTower : ℝ
  ropeTouchLength : ℝ
  a : ℕ
  b : ℕ
  c : ℕ

/-- The theorem stating the sum of a, b, and c for the given configuration. -/
theorem tethered_unicorn_sum (u : TetheredUnicorn)
  (h1 : u.towerRadius = 10)
  (h2 : u.ropeLength = 30)
  (h3 : u.unicornHeight = 6)
  (h4 : u.distanceFromTower = 6)
  (h5 : u.ropeTouchLength = (u.a - Real.sqrt u.b) / u.c)
  (h6 : Nat.Prime u.c) :
  u.a + u.b + u.c = 940 := by
  sorry

end NUMINAMATH_CALUDE_tethered_unicorn_sum_l3665_366509


namespace NUMINAMATH_CALUDE_gear_speed_ratio_l3665_366505

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Represents a system of four gears meshed in sequence -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear
  meshed_AB : A.teeth * A.speed = B.teeth * B.speed
  meshed_BC : B.teeth * B.speed = C.teeth * C.speed
  meshed_CD : C.teeth * C.speed = D.teeth * D.speed

/-- The theorem stating the ratio of angular speeds for the given gear system -/
theorem gear_speed_ratio (sys : GearSystem) 
  (hA : sys.A.teeth = 10)
  (hB : sys.B.teeth = 15)
  (hC : sys.C.teeth = 20)
  (hD : sys.D.teeth = 25) :
  ∃ (k : ℝ), k > 0 ∧ 
    sys.A.speed = 24 * k ∧
    sys.B.speed = 25 * k ∧
    sys.C.speed = 12 * k ∧
    sys.D.speed = 20 * k := by
  sorry

end NUMINAMATH_CALUDE_gear_speed_ratio_l3665_366505


namespace NUMINAMATH_CALUDE_gcd_bound_for_special_numbers_l3665_366530

/-- Given two 2019-digit numbers a and b with specific non-zero digit patterns,
    prove that their greatest common divisor has at most 14 digits. -/
theorem gcd_bound_for_special_numbers (a b : ℕ) : 
  (∃ A B C D : ℕ,
    a = A * 10^2014 + B ∧ 
    b = C * 10^2014 + D ∧
    10^4 < A ∧ A < 10^5 ∧
    10^6 < B ∧ B < 10^7 ∧
    10^4 < C ∧ C < 10^5 ∧
    10^8 < D ∧ D < 10^9) →
  Nat.gcd a b < 10^14 :=
sorry

end NUMINAMATH_CALUDE_gcd_bound_for_special_numbers_l3665_366530


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l3665_366567

/-- A complex number is pure imaginary if its real part is zero. -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0

/-- The problem statement -/
theorem complex_product_pure_imaginary (a : ℝ) :
  IsPureImaginary ((a : ℂ) + Complex.I * (2 - Complex.I)) → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l3665_366567


namespace NUMINAMATH_CALUDE_factorial_difference_l3665_366502

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l3665_366502


namespace NUMINAMATH_CALUDE_carolyn_stitching_rate_l3665_366580

/-- Represents the number of stitches required for a flower -/
def flower_stitches : ℕ := 60

/-- Represents the number of stitches required for a unicorn -/
def unicorn_stitches : ℕ := 180

/-- Represents the number of stitches required for Godzilla -/
def godzilla_stitches : ℕ := 800

/-- Represents the number of unicorns in the embroidery -/
def num_unicorns : ℕ := 3

/-- Represents the number of flowers in the embroidery -/
def num_flowers : ℕ := 50

/-- Represents the total time Carolyn spends embroidering (in minutes) -/
def total_time : ℕ := 1085

/-- Calculates Carolyn's stitching rate -/
def stitching_rate : ℚ :=
  (godzilla_stitches + num_unicorns * unicorn_stitches + num_flowers * flower_stitches) / total_time

theorem carolyn_stitching_rate :
  stitching_rate = 4 := by sorry

end NUMINAMATH_CALUDE_carolyn_stitching_rate_l3665_366580


namespace NUMINAMATH_CALUDE_total_rainbow_nerds_l3665_366555

/-- The number of rainbow nerds in a box with purple, yellow, and green candies. -/
def rainbow_nerds (purple : ℕ) (yellow : ℕ) (green : ℕ) : ℕ := purple + yellow + green

/-- Theorem: The total number of rainbow nerds in the box is 36. -/
theorem total_rainbow_nerds :
  ∃ (purple yellow green : ℕ),
    purple = 10 ∧
    yellow = purple + 4 ∧
    green = yellow - 2 ∧
    rainbow_nerds purple yellow green = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_rainbow_nerds_l3665_366555


namespace NUMINAMATH_CALUDE_arctan_sum_l3665_366537

theorem arctan_sum : Real.arctan (3/4) + Real.arctan (4/3) = π / 2 := by sorry

end NUMINAMATH_CALUDE_arctan_sum_l3665_366537


namespace NUMINAMATH_CALUDE_smallest_aab_value_exists_valid_digit_pair_l3665_366538

/-- Represents a pair of distinct digits from 1 to 9 -/
structure DigitPair where
  a : Nat
  b : Nat
  a_in_range : a ≥ 1 ∧ a ≤ 9
  b_in_range : b ≥ 1 ∧ b ≤ 9
  distinct : a ≠ b

/-- Converts a DigitPair to a two-digit number -/
def to_two_digit (p : DigitPair) : Nat :=
  10 * p.a + p.b

/-- Converts a DigitPair to a three-digit number AAB -/
def to_three_digit (p : DigitPair) : Nat :=
  100 * p.a + 10 * p.a + p.b

/-- The main theorem stating the smallest possible value of AAB -/
theorem smallest_aab_value (p : DigitPair) 
  (h : to_two_digit p = (to_three_digit p) / 8) : 
  to_three_digit p ≥ 773 := by
  sorry

/-- The existence of a DigitPair satisfying the conditions -/
theorem exists_valid_digit_pair : 
  ∃ p : DigitPair, to_two_digit p = (to_three_digit p) / 8 ∧ to_three_digit p = 773 := by
  sorry

end NUMINAMATH_CALUDE_smallest_aab_value_exists_valid_digit_pair_l3665_366538


namespace NUMINAMATH_CALUDE_day_299_is_tuesday_l3665_366571

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the day of the week given a day number -/
def dayOfWeek (dayNumber : Nat) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem day_299_is_tuesday (isLeapYear : Bool) :
  isLeapYear ∧ dayOfWeek 45 = DayOfWeek.Sunday →
  dayOfWeek 299 = DayOfWeek.Tuesday :=
by
  sorry

end NUMINAMATH_CALUDE_day_299_is_tuesday_l3665_366571


namespace NUMINAMATH_CALUDE_car_traveler_speed_ratio_l3665_366563

/-- Represents the bridge in the problem -/
structure Bridge where
  length : ℝ
  mk_pos : length > 0

/-- Represents the traveler in the problem -/
structure Traveler where
  speed : ℝ
  mk_pos : speed > 0

/-- Represents the car in the problem -/
structure Car where
  speed : ℝ
  mk_pos : speed > 0

/-- The main theorem stating the ratio of car speed to traveler speed -/
theorem car_traveler_speed_ratio (b : Bridge) (t : Traveler) (c : Car) :
  (t.speed * (4 / 9) * b.length / t.speed = c.speed * (4 / 9) * b.length / c.speed) →
  (t.speed * (5 / 9) * b.length / t.speed = b.length / c.speed) →
  c.speed / t.speed = 9 := by
  sorry


end NUMINAMATH_CALUDE_car_traveler_speed_ratio_l3665_366563


namespace NUMINAMATH_CALUDE_probability_two_diamonds_one_ace_l3665_366576

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of diamonds in a standard deck -/
def DiamondCount : ℕ := 13

/-- Number of aces in a standard deck -/
def AceCount : ℕ := 4

/-- Probability of drawing two diamonds followed by an ace from a standard deck -/
def probabilityTwoDiamondsOneAce : ℚ :=
  (DiamondCount : ℚ) / StandardDeck *
  (DiamondCount - 1) / (StandardDeck - 1) *
  ((DiamondCount : ℚ) / StandardDeck * (AceCount - 1) / (StandardDeck - 2) +
   (StandardDeck - DiamondCount : ℚ) / StandardDeck * AceCount / (StandardDeck - 2))

theorem probability_two_diamonds_one_ace :
  probabilityTwoDiamondsOneAce = 29 / 11050 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_diamonds_one_ace_l3665_366576


namespace NUMINAMATH_CALUDE_max_angle_at_3_2_l3665_366541

/-- The line l: x + y - 5 = 0 -/
def line (x y : ℝ) : Prop := x + y - 5 = 0

/-- Point A -/
def A : ℝ × ℝ := (1, 0)

/-- Point B -/
def B : ℝ × ℝ := (3, 0)

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Statement: The point (3,2) maximizes the angle APB on the given line -/
theorem max_angle_at_3_2 :
  line 3 2 ∧
  ∀ x y, line x y → angle A (x, y) B ≤ angle A (3, 2) B :=
by sorry

end NUMINAMATH_CALUDE_max_angle_at_3_2_l3665_366541


namespace NUMINAMATH_CALUDE_max_cone_section_area_l3665_366560

/-- The maximum area of a cone section passing through the vertex, given the cone's height and volume --/
theorem max_cone_section_area (h : ℝ) (v : ℝ) : h = 1 → v = π → 
  ∃ (max_area : ℝ), max_area = 2 ∧ 
  ∀ (section_area : ℝ), section_area ≤ max_area := by
  sorry


end NUMINAMATH_CALUDE_max_cone_section_area_l3665_366560


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3665_366551

/-- A point on a cubic curve with a specific tangent slope -/
structure TangentPoint where
  x : ℝ
  y : ℝ
  on_curve : y = x^3 - 10*x + 3
  in_second_quadrant : x < 0 ∧ y > 0
  tangent_slope : 3*x^2 - 10 = 2

/-- The equation of the tangent line -/
def tangent_line (p : TangentPoint) : ℝ → ℝ := λ x => 2*x + 19

theorem tangent_line_equation (p : TangentPoint) :
  tangent_line p p.x = p.y ∧
  (λ x => tangent_line p x - p.y) = (λ x => 2*(x - p.x)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3665_366551


namespace NUMINAMATH_CALUDE_min_value_theorem_l3665_366594

theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ (m : ℝ), m = 4 ∧ (∀ x y, x > y ∧ y > 0 → x^2 + 1 / (y * (x - y)) ≥ m) ∧
  (∃ x y, x > y ∧ y > 0 ∧ x^2 + 1 / (y * (x - y)) = m) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3665_366594


namespace NUMINAMATH_CALUDE_english_score_is_98_l3665_366528

/-- Given the Mathematics score, Korean language score, and average score,
    calculate the English score. -/
def calculate_english_score (math_score : ℕ) (korean_offset : ℕ) (average_score : ℚ) : ℚ :=
  3 * average_score - (math_score : ℚ) - ((math_score : ℚ) + korean_offset)

/-- Theorem stating that under the given conditions, the English score is 98. -/
theorem english_score_is_98 :
  let math_score : ℕ := 82
  let korean_offset : ℕ := 5
  let average_score : ℚ := 89
  calculate_english_score math_score korean_offset average_score = 98 := by
  sorry

#eval calculate_english_score 82 5 89

end NUMINAMATH_CALUDE_english_score_is_98_l3665_366528


namespace NUMINAMATH_CALUDE_championship_outcomes_8_3_l3665_366587

/-- The number of possible outcomes for championships -/
def championship_outcomes (num_students : ℕ) (num_championships : ℕ) : ℕ :=
  num_students ^ num_championships

/-- Theorem: The number of possible outcomes for 3 championships among 8 students is 512 -/
theorem championship_outcomes_8_3 :
  championship_outcomes 8 3 = 512 := by
  sorry

end NUMINAMATH_CALUDE_championship_outcomes_8_3_l3665_366587


namespace NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_nonzero_l3665_366550

theorem x_positive_sufficient_not_necessary_for_x_nonzero :
  (∃ x : ℝ, x ≠ 0 ∧ ¬(x > 0)) ∧
  (∀ x : ℝ, x > 0 → x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_nonzero_l3665_366550


namespace NUMINAMATH_CALUDE_sticker_cost_theorem_l3665_366596

def total_sticker_cost (allowance : ℕ) (card_cost : ℕ) (stickers_per_person : ℕ) : ℕ :=
  2 * allowance - card_cost

theorem sticker_cost_theorem (allowance : ℕ) (card_cost : ℕ) (stickers_per_person : ℕ)
  (h1 : allowance = 9)
  (h2 : card_cost = 10)
  (h3 : stickers_per_person = 2) :
  total_sticker_cost allowance card_cost stickers_per_person = 8 := by
sorry

#eval total_sticker_cost 9 10 2

end NUMINAMATH_CALUDE_sticker_cost_theorem_l3665_366596


namespace NUMINAMATH_CALUDE_vector_sum_zero_parallel_sufficient_not_necessary_l3665_366549

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

theorem vector_sum_zero_parallel_sufficient_not_necessary :
  ∀ (a b : V), a ≠ 0 → b ≠ 0 →
  (a + b = 0 → parallel a b) ∧
  ¬(parallel a b → a + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_zero_parallel_sufficient_not_necessary_l3665_366549


namespace NUMINAMATH_CALUDE_m_range_l3665_366500

-- Define the propositions
def P (t : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (t + 2) + y^2 / (t - 10) = 1

def Q (t m : ℝ) : Prop := 1 - m < t ∧ t < 1 + m ∧ m > 0

-- Define the relationship between P and Q
def relationship (m : ℝ) : Prop :=
  ∀ t, ¬(P t) → ¬(Q t m) ∧ ∃ t, ¬(Q t m) ∧ P t

-- State the theorem
theorem m_range :
  ∀ m, (∀ t, P t ↔ t ∈ Set.Ioo (-2) 10) →
       relationship m →
       m ∈ Set.Ioc 0 3 :=
sorry

end NUMINAMATH_CALUDE_m_range_l3665_366500


namespace NUMINAMATH_CALUDE_gcf_of_36_and_12_l3665_366513

theorem gcf_of_36_and_12 :
  let n : ℕ := 36
  let m : ℕ := 12
  let lcm_nm : ℕ := 54
  lcm n m = lcm_nm →
  Nat.gcd n m = 8 := by
sorry

end NUMINAMATH_CALUDE_gcf_of_36_and_12_l3665_366513


namespace NUMINAMATH_CALUDE_zongzi_production_theorem_l3665_366562

/-- The average daily production of zongzi for Team A -/
def team_a_production : ℝ := 200

/-- The average daily production of zongzi for Team B -/
def team_b_production : ℝ := 150

/-- Theorem stating that given the conditions, the average daily production
    of zongzi for Team A is 200 bags and for Team B is 150 bags -/
theorem zongzi_production_theorem :
  (team_a_production + team_b_production = 350) ∧
  (2 * team_a_production - team_b_production = 250) →
  team_a_production = 200 ∧ team_b_production = 150 := by
  sorry

end NUMINAMATH_CALUDE_zongzi_production_theorem_l3665_366562


namespace NUMINAMATH_CALUDE_triangle_problem_l3665_366516

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  c = 4 * Real.sqrt 2 →
  B = π / 4 →
  (1/2) * a * c * Real.sin B = 2 →
  a = 1 ∧ b = 5 :=
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3665_366516


namespace NUMINAMATH_CALUDE_least_number_to_add_l3665_366526

theorem least_number_to_add (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((28523 + y) % 3 = 0 ∧ (28523 + y) % 5 = 0 ∧ (28523 + y) % 7 = 0 ∧ (28523 + y) % 8 = 0)) ∧
  ((28523 + x) % 3 = 0 ∧ (28523 + x) % 5 = 0 ∧ (28523 + x) % 7 = 0 ∧ (28523 + x) % 8 = 0) →
  x = 137 := by
sorry

end NUMINAMATH_CALUDE_least_number_to_add_l3665_366526
