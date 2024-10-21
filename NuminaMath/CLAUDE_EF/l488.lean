import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_phone_chargers_l488_48830

/-- The number of phone chargers Anna has -/
def P : ℕ := sorry

/-- The number of laptop chargers Anna has -/
def L : ℕ := sorry

/-- The total number of chargers Anna has -/
def total_chargers : ℕ := 24

/-- Laptop chargers are 5 times the number of phone chargers -/
axiom laptop_charger_relation : L = 5 * P

/-- The total number of chargers is the sum of phone and laptop chargers -/
axiom total_chargers_sum : P + L = total_chargers

theorem anna_phone_chargers : P = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_phone_chargers_l488_48830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l488_48846

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ↔ a ∈ Set.Ioc 0 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l488_48846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l488_48890

-- Define the curve C'
noncomputable def curve_C' (x y : ℝ) : Prop := y^2 / 16 + x^2 / 4 = 1

-- Define the line l
noncomputable def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 4 = 0

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x₀ y₀ : ℝ) : ℝ :=
  |Real.sqrt 3 * x₀ + y₀ - 4| / Real.sqrt ((Real.sqrt 3)^2 + 1^2)

-- Theorem statement
theorem max_distance_to_line :
  ∀ x₀ y₀ : ℝ, curve_C' x₀ y₀ →
  (∀ x y : ℝ, curve_C' x y → distance_to_line x y ≤ distance_to_line x₀ y₀) →
  distance_to_line x₀ y₀ = 2 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l488_48890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_implies_m_leq_three_l488_48899

/-- A function f(x) = x^2 + ln(x) + m/x is increasing on [1, +∞) -/
def is_increasing_on_interval (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 1 → Monotone f

/-- The function f(x) = x^2 + ln(x) + m/x -/
noncomputable def f (x m : ℝ) : ℝ := x^2 + Real.log x + m/x

theorem increasing_function_implies_m_leq_three (m : ℝ) :
  is_increasing_on_interval (f · m) m → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_implies_m_leq_three_l488_48899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_open_unit_interval_iso_real_int_not_iso_rat_l488_48891

-- Define order-preserving isomorphism
def OrderPreservingIsomorphic (S T : Type) [PartialOrder S] [PartialOrder T] : Prop :=
  ∃ f : S → T, 
    Function.Surjective f ∧
    (∀ x y : S, x < y → f x < f y)

-- Define the open interval (0, 1)
def OpenUnitInterval : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

-- Theorem statements
theorem open_unit_interval_iso_real : 
  OrderPreservingIsomorphic OpenUnitInterval ℝ :=
sorry

theorem int_not_iso_rat : 
  ¬OrderPreservingIsomorphic ℤ ℚ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_open_unit_interval_iso_real_int_not_iso_rat_l488_48891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_zero_num_zeros_l488_48869

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then 1 - 1 / x else (a - 1) * x + 1

-- Theorem for f(f(0))
theorem f_f_zero (a : ℝ) : f a (f a 0) = 0 := by
  sorry

-- Theorem for the number of zeros
theorem num_zeros (a : ℝ) :
  (∃! x, f a x = 0 ∧ a = 1) ∨
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ a > 1) ∨
  (∃! x, f a x = 0 ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_zero_num_zeros_l488_48869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kate_museum_visits_correct_l488_48810

def kate_museum_visits 
  (initial_fee : ℕ) 
  (increased_fee : ℕ) 
  (first_year_visits : ℕ) 
  (total_years : ℕ) 
  (total_paid : ℕ) : ℕ :=
  let first_year_cost := initial_fee * first_year_visits
  let remaining_years := total_years - 1
  let remaining_cost := total_paid - first_year_cost
  let remaining_visits := remaining_cost / increased_fee
  remaining_visits / remaining_years

#eval kate_museum_visits 5 7 12 3 116

theorem kate_museum_visits_correct 
  (initial_fee : ℕ) 
  (increased_fee : ℕ) 
  (first_year_visits : ℕ) 
  (total_years : ℕ) 
  (total_paid : ℕ) : 
  kate_museum_visits initial_fee increased_fee first_year_visits total_years total_paid = 4 :=
by
  sorry

#check kate_museum_visits_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kate_museum_visits_correct_l488_48810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_chairs_count_l488_48865

theorem total_chairs_count : ℕ := by
  let section1 := 5 * 10
  let section2 := 8 * 12
  let section3 := 4 * 15
  let section4 := 6 * 9
  let late_section1 := 15 * 2
  let late_section2 := 25 * 3
  let late_section3 := 8 * 1
  let late_section4 := 12 * 1
  let total_chairs := section1 + section2 + section3 + section4 + 
                      late_section1 + late_section2 + late_section3 + late_section4
  have h : total_chairs = 385 := by rfl
  exact total_chairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_chairs_count_l488_48865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_heptagon_relation_l488_48892

/-- A regular heptagon -/
structure RegularHeptagon where
  vertices : Fin 7 → ℝ × ℝ

/-- The condition that a point lies on a line segment -/
def lies_on (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop := sorry

/-- The center of a regular heptagon -/
def center (h : RegularHeptagon) : ℝ × ℝ := sorry

theorem regular_heptagon_relation (ABCDEFG APQRSTU : RegularHeptagon) 
  (O : ℝ × ℝ) (h1 : O = center ABCDEFG) 
  (h2 : lies_on (APQRSTU.vertices 1) (ABCDEFG.vertices 0) (ABCDEFG.vertices 1)) 
  (h3 : lies_on (APQRSTU.vertices 6) (ABCDEFG.vertices 6) (ABCDEFG.vertices 0)) 
  (h4 : lies_on (APQRSTU.vertices 2) O (ABCDEFG.vertices 1)) : 
  (ABCDEFG.vertices 1 - ABCDEFG.vertices 0) = 2 * (APQRSTU.vertices 1 - APQRSTU.vertices 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_heptagon_relation_l488_48892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_product_sum_of_digits_l488_48833

def is_single_digit_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ p < 10

def is_valid_triple (d e p : ℕ) : Prop :=
  is_single_digit_prime d ∧ 
  is_single_digit_prime e ∧ 
  Nat.Prime p ∧ 
  p = d^2 + 10*e

def product (d e p : ℕ) : ℕ :=
  d * e * p

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem largest_product_sum_of_digits :
  ∃ (n d e p : ℕ),
    is_valid_triple d e p ∧
    n = product d e p ∧
    (∀ (d' e' p' : ℕ), is_valid_triple d' e' p' → product d' e' p' ≤ n) ∧
    sum_of_digits n = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_product_sum_of_digits_l488_48833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l488_48828

def sequence_a : ℕ → ℚ
  | 0 => 2/3  -- Define for 0 to cover all natural numbers
  | n+1 => 2 * sequence_a n / (sequence_a n + 1)

theorem sequence_a_properties :
  (∀ n : ℕ, (1 / sequence_a (n+1) - 1 = (1/2) * (1 / sequence_a n - 1))) ∧
  (∀ n : ℕ, sequence_a n = (2^n : ℚ) / (1 + 2^n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l488_48828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l488_48893

-- Define the line (noncomputable due to use of real numbers)
noncomputable def line (t : ℝ) : ℝ × ℝ := (2 + t, 4 - t)

-- Define the curve (circle) (noncomputable due to use of Real.sqrt and trigonometric functions)
noncomputable def curve (θ : ℝ) : ℝ × ℝ := (3 + Real.sqrt 2 * Real.cos θ, 5 + Real.sqrt 2 * Real.sin θ)

-- Theorem statement
theorem line_tangent_to_circle :
  ∃! p : ℝ × ℝ, (∃ t : ℝ, line t = p) ∧ (∃ θ : ℝ, curve θ = p) := by
  sorry

#check line_tangent_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l488_48893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_price_comparison_l488_48838

noncomputable def watermelon_price : ℝ := 9.5
noncomputable def watermelon_quantity : ℝ := 5
noncomputable def pineapple_price : ℝ := 15.6
noncomputable def pineapple_quantity : ℝ := 3

noncomputable def watermelon_unit_price : ℝ := watermelon_price / watermelon_quantity
noncomputable def pineapple_unit_price : ℝ := pineapple_price / pineapple_quantity

theorem fruit_price_comparison :
  pineapple_unit_price > watermelon_unit_price ∧
  pineapple_unit_price - watermelon_unit_price = 3.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_price_comparison_l488_48838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_gcd_convergence_l488_48867

def circleGCD (a : List ℕ) : List ℕ :=
  match a with
  | [] => []
  | [x] => [x]
  | x :: xs => 
    let last := xs.getLast?.getD 0
    (Nat.gcd x last) :: circleGCD xs ++ [Nat.gcd (xs.getLast?.getD 0) x]

def iterateUntilEqual (f : List ℕ → List ℕ) (a : List ℕ) : ℕ → List ℕ
  | 0 => a
  | n + 1 => 
    let next := f a
    if next.all (· = next.head!) then next
    else iterateUntilEqual f next n

theorem circle_gcd_convergence (a : List ℕ) (h : a.length > 0) :
  ∃ n : ℕ, ∃ k : ℕ, (iterateUntilEqual circleGCD a n).all (· = k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_gcd_convergence_l488_48867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_inscribed_squares_l488_48804

-- Define the necessary structures and predicates
structure Triangle where
  -- Add necessary fields

structure Square where
  area : ℝ

def IsIsoscelesRight (t : Triangle) : Prop := sorry

def IsInscribed (s : Square) (t : Triangle) : Prop := sorry

def TouchesHypotenuse (s : Square) (t : Triangle) : Prop := sorry

def TouchesLegs (s : Square) (t : Triangle) : Prop := sorry

def TouchesOneLeg (s : Square) (t : Triangle) : Prop := sorry

theorem isosceles_right_triangle_inscribed_squares 
  (triangle : Triangle) 
  (is_isosceles_right : IsIsoscelesRight triangle) 
  (first_square : Square) 
  (first_square_inscribed : IsInscribed first_square triangle)
  (first_square_touches_hypotenuse : TouchesHypotenuse first_square triangle)
  (first_square_touches_legs : TouchesLegs first_square triangle)
  (first_square_area : first_square.area = 484) :
  ∃ (second_square : Square),
    IsInscribed second_square triangle ∧
    TouchesHypotenuse second_square triangle ∧
    TouchesOneLeg second_square triangle ∧
    second_square.area = 3872 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_inscribed_squares_l488_48804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C₁_to_C₂_l488_48805

-- Define the curves C₁ and C₂
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos θ, Real.sin θ)

def C₂ (ρ θ : ℝ) : Prop := ρ * Real.cos θ - ρ * Real.sin θ - 4 = 0

-- Define the distance function from a point to C₂
noncomputable def distance_to_C₂ (P : ℝ × ℝ) : ℝ :=
  (|P.1 - P.2 - 4|) / Real.sqrt 2

-- Theorem statement
theorem max_distance_C₁_to_C₂ :
  ∃ (M : ℝ), M = 3 * Real.sqrt 2 ∧
  ∀ (θ : ℝ), distance_to_C₂ (C₁ θ) ≤ M ∧
  ∃ (θ₀ : ℝ), distance_to_C₂ (C₁ θ₀) = M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C₁_to_C₂_l488_48805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_correct_and_roots_count_l488_48878

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the conditions on f
axiom f_odd (x : ℝ) : f x + f (-x) = 0
axiom f_shift (x : ℝ) : f (x - 1) = f (x + 1)
axiom f_form (x : ℝ) (h : 0 ≤ x ∧ x < 1) : ∃ (a b : ℝ), a > 0 ∧ a ≠ 1 ∧ f x = a^x + b
axiom f_value : f (3/2) = 1 - Real.sqrt 2

-- Define the piecewise function for f on [-1, 1]
noncomputable def f_piecewise (x : ℝ) : ℝ :=
  if -1 < x ∧ x ≤ 0 then 1 - 2^(-x)
  else if x = -1 ∨ x = 1 then 0
  else if 0 ≤ x ∧ x < 1 then 2^x - 1
  else 0  -- This case should never occur for x ∈ [-1, 1]

-- State the theorem
theorem f_correct_and_roots_count :
  (∀ x, -1 ≤ x ∧ x ≤ 1 → f x = f_piecewise x) ∧
  (∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, f x = |Real.log x / Real.log 4|) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_correct_and_roots_count_l488_48878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_minus_b_l488_48820

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + Real.log x

-- Define the tangent line y = kx + b
def tangent_line (k b x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem min_k_minus_b :
  ∀ (k b x₀ : ℝ),
  (∀ x, f x ≥ tangent_line k b x) →  -- The line is tangent to f(x)
  (f x₀ = tangent_line k b x₀) →     -- The line touches f(x) at x₀
  (k - b ≥ 7/2) :=                   -- The minimum value of k - b is 7/2
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_minus_b_l488_48820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l488_48897

theorem system_solution (x y : ℝ) (k : ℤ) : 
  (x - y = -1/3) → 
  (Real.cos (π * x) ^ 2 - Real.sin (π * y) ^ 2 = 1/2) → 
  (x = k - 1/6 ∧ y = k + 1/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l488_48897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_tens_digit_probability_l488_48860

/-- The range of integers to choose from -/
def range : Finset ℕ := Finset.filter (fun n => 20 ≤ n ∧ n ≤ 69) (Finset.range 70)

/-- The set of possible tens digits -/
def tensDigits : Finset ℕ := {2, 3, 4, 5, 6}

/-- The number of integers to choose -/
def k : ℕ := 5

/-- The probability of choosing k different integers from the range
    such that each has a different tens digit -/
def probability : ℚ :=
  (tensDigits.card ^ k : ℚ) / Nat.choose range.card k

theorem different_tens_digit_probability :
  probability = 2500 / 52969 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_tens_digit_probability_l488_48860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_witch_clock_theorem_l488_48821

/-- Represents the movement of the minute hand in one cycle -/
def cycle_movement : ℤ := 5 - 3

/-- Calculates the position of the minute hand after a given number of minutes -/
def minute_hand_position (total_minutes : ℕ) : ℕ :=
  let full_cycles := total_minutes / 8
  let remaining_minutes := total_minutes % 8
  let position_after_cycles := (full_cycles * cycle_movement).toNat % 60
  let additional_movement := if remaining_minutes ≤ 5 then remaining_minutes else 5 - (remaining_minutes - 5)
  (position_after_cycles + additional_movement) % 60

theorem witch_clock_theorem : minute_hand_position 2022 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_witch_clock_theorem_l488_48821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l488_48825

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the even property of f(x+1)
def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = f (1 - x)

-- Define the increasing property of f
def is_increasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 1 ≤ x₁ → x₁ < x₂ → (f x₂ - f x₁) / (x₂ - x₁) > 0

-- Define a, b, and c
noncomputable def a (f : ℝ → ℝ) : ℝ := f (-1/2)
noncomputable def b (f : ℝ → ℝ) : ℝ := f 1
noncomputable def c (f : ℝ → ℝ) : ℝ := f 2

-- State the theorem
theorem f_inequality (f : ℝ → ℝ) (heven : is_even_shifted f) (hincr : is_increasing_on_interval f) :
  b f < c f ∧ c f < a f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l488_48825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_from_equal_angles_l488_48813

/-- AngleRatio △ A B C represents the ratio of angles in triangle ABC -/
def AngleRatio (t : Triangle) : ℕ × ℕ × ℕ := sorry

/-- Equilateral t means that triangle t is equilateral -/
def Equilateral (t : Triangle) : Prop := sorry

/-- Point represents a point in the plane -/
def Point : Type := sorry

/-- Triangle represents a triangle formed by three points -/
def Triangle : Type := Point × Point × Point

theorem equilateral_triangle_from_equal_angles (A B C : Point) 
  (h : AngleRatio (A, B, C) = (1, 1, 1)) : 
  Equilateral (A, B, C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_from_equal_angles_l488_48813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_M_coordinates_l488_48888

theorem point_M_coordinates (θ : Real) (h1 : π/2 < θ ∧ θ < π) (h2 : Real.tan θ = -4/3) :
  let M : ℝ × ℝ := (5 * Real.cos θ, 5 * Real.sin θ)
  M = (-3, 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_M_coordinates_l488_48888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_is_valid_l488_48839

/-- The inclination angle of the line in degrees -/
def inclination_angle : ℝ := 150

/-- The direction vector of the line -/
noncomputable def direction_vector : ℝ × ℝ := (-3, Real.sqrt 3)

/-- Checks if the given direction vector is valid for the given inclination angle -/
def is_valid_direction_vector (angle : ℝ) (vector : ℝ × ℝ) : Prop :=
  let slope := Real.tan (angle * Real.pi / 180)
  vector.2 / vector.1 = -slope

/-- Theorem stating that the direction vector is valid for the given inclination angle -/
theorem direction_vector_is_valid :
  is_valid_direction_vector inclination_angle direction_vector := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_is_valid_l488_48839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_jogger_l488_48823

/-- The time it takes for a train to pass a jogger --/
noncomputable def train_passing_time (jogger_speed : ℝ) (train_speed : ℝ) 
                       (initial_distance : ℝ) (train_length : ℝ) : ℝ :=
  let relative_speed := train_speed - jogger_speed
  let total_distance := initial_distance + train_length
  total_distance / (relative_speed * 1000 / 3600)

/-- Theorem stating the time it takes for the train to pass the jogger --/
theorem train_passes_jogger :
  let jogger_speed := (12 : ℝ)
  let train_speed := (60 : ℝ)
  let initial_distance := (180 : ℝ)
  let train_length := (150 : ℝ)
  ∃ ε > 0, |train_passing_time jogger_speed train_speed initial_distance train_length - 24.75| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_jogger_l488_48823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_heads_probability_l488_48885

/-- The probability of getting heads on a single toss of the unfair coin -/
noncomputable def p : ℝ := 3/5

/-- The number of coin tosses -/
def n : ℕ := 30

/-- The probability of getting an odd number of heads after n tosses -/
noncomputable def P (n : ℕ) : ℝ :=
  1/2 * (1 + (-1/5)^n)

/-- The main theorem: the probability of getting an odd number of heads after 30 tosses -/
theorem odd_heads_probability :
  P n = 1/2 * (1 - 1/5^30) := by
  -- The proof goes here
  sorry

#eval n -- This line is added to use the constant n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_heads_probability_l488_48885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equality_l488_48848

def A : Set ℝ := {-2, 1}

def B (a : ℝ) : Set ℝ := {x | a * x = 2}

theorem intersection_equality : 
  ∀ a : ℝ, (A ∩ B a = B a) ↔ a ∈ ({-1, 0, 2} : Set ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equality_l488_48848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_eighteen_l488_48866

/-- A function g: ℝ → ℝ that satisfies g(3+x) = g(3-x) for all real x -/
def SymmetricAboutThree (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (3 + x) = g (3 - x)

/-- The equation g(x) = 0 has exactly six distinct real roots -/
def HasSixDistinctRoots (g : ℝ → ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ), 
    (∀ x : ℝ, g x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄ ∨ x = r₅ ∨ x = r₆) ∧
    (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₁ ≠ r₅ ∧ r₁ ≠ r₆ ∧
     r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₂ ≠ r₅ ∧ r₂ ≠ r₆ ∧
     r₃ ≠ r₄ ∧ r₃ ≠ r₅ ∧ r₃ ≠ r₆ ∧
     r₄ ≠ r₅ ∧ r₄ ≠ r₆ ∧
     r₅ ≠ r₆)

theorem sum_of_roots_is_eighteen (g : ℝ → ℝ) 
  (h₁ : SymmetricAboutThree g) (h₂ : HasSixDistinctRoots g) : 
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ), 
    (∀ x : ℝ, g x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄ ∨ x = r₅ ∨ x = r₆) ∧
    r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 18 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_eighteen_l488_48866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_zero_l488_48824

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 1

-- Define the derivative of the function
noncomputable def f_derivative (x : ℝ) : ℝ := Real.exp x

-- Theorem statement
theorem tangent_slope_at_zero : 
  (f_derivative 0) = 1 := by
  -- Unfold the definition of f_derivative
  unfold f_derivative
  -- Simplify Real.exp 0
  simp [Real.exp_zero]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_zero_l488_48824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_points_l488_48877

-- Define the concept of an inverse proportion point
def has_inverse_proportion_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀ * f x₀ = 1

-- Define the given functions
noncomputable def f₁ (x : ℝ) : ℝ := -2 * x + 2 * Real.sqrt 2
noncomputable def f₂ (x : ℝ) : ℝ := Real.sin x
noncomputable def f₃ (x : ℝ) : ℝ := x + 1 / x
noncomputable def f₄ (x : ℝ) : ℝ := Real.exp x
noncomputable def f₅ (x : ℝ) : ℝ := -2 * Real.log x

-- State the theorem
theorem inverse_proportion_points :
  has_inverse_proportion_point f₁ ∧
  has_inverse_proportion_point (f₂ ∘ (λ x => x * (2 * Real.pi))) ∧
  has_inverse_proportion_point f₄ ∧
  ¬has_inverse_proportion_point (f₃ ∘ (λ x => x + 1)) ∧
  ¬has_inverse_proportion_point (f₅ ∘ (λ x => x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_points_l488_48877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l488_48882

/-- Calculates the length of a train given its speed, the bridge length, and the time to cross. -/
theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (time_to_cross : ℝ) :
  let speed_ms := train_speed * 1000 / 3600
  let total_distance := speed_ms * time_to_cross
  let train_length := total_distance - bridge_length
  train_speed = 48 ∧ bridge_length = 300 ∧ time_to_cross = 30 →
  (abs (train_length - 99.9) < 0.1) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l488_48882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_with_foci_on_y_axis_l488_48853

theorem ellipse_with_foci_on_y_axis (θ : Real) (h1 : 0 < θ ∧ θ < π) 
  (h2 : Real.sin θ + Real.cos θ = 7/13) : 
  ∃ (a b : Real), a > b ∧ b > 0 ∧
  ∀ (x y : Real), x^2 * Real.sin θ - y^2 * Real.cos θ = 1 ↔ 
    x^2 / b^2 + y^2 / a^2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_with_foci_on_y_axis_l488_48853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_even_function_inequalities_l488_48842

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define a and b
variable (a b : ℝ)

-- State the theorem
theorem odd_even_function_inequalities
  (h_odd : ∀ x, f (-x) = -f x)  -- f is odd
  (h_incr : ∀ x y, x < y → f x < f y)  -- f is increasing
  (h_even : ∀ x, g (-x) = g x)  -- g is even
  (h_coincide : ∀ x, x ≥ 0 → f x = g x)  -- f and g coincide on [0, +∞)
  (h_a_b : a > b ∧ b > 0) :  -- a > b > 0
  (f b - f (-a) > g a - g (-b)) ∧
  (f a - f (-b) > g b - g (-a)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_even_function_inequalities_l488_48842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yard_length_l488_48837

/-- The length of the yard given the father and son's step measurements -/
theorem yard_length (total_footprints : ℕ) (father_step : ℝ) (son_step : ℝ) 
  (h1 : total_footprints = 61)
  (h2 : father_step = 0.72)
  (h3 : son_step = 0.54) :
  ∃ (yard_length : ℝ), yard_length = 21.6 := by
  sorry

#check yard_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yard_length_l488_48837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l488_48858

noncomputable def g (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 - 4*x else -x^2 - 4*x

theorem g_properties :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, MonotoneOn g (Set.Icc (-2 : ℝ) 2)) ∧
  {a : ℝ | g a > a} = {a : ℝ | a > 5 ∨ (-5 < a ∧ a < 0)} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l488_48858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l488_48862

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The upper vertex of an ellipse -/
def upper_vertex (e : Ellipse) : ℝ × ℝ :=
  (0, e.b)

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem statement -/
theorem eccentricity_range (e : Ellipse) :
  (∀ (p : PointOnEllipse e), distance (p.x, p.y) (upper_vertex e) ≤ 2 * e.b) →
  0 < eccentricity e ∧ eccentricity e ≤ Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l488_48862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l488_48874

-- Define the triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides opposite to angles A, B, C respectively

-- Define the conditions
def IsValidTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

def FormsGeometricProgression (t : Triangle) : Prop :=
  ∃ r : Real, r > 0 ∧ (t.b = t.a * r ∧ t.c = t.b * r) ∨ (t.b = t.c * r ∧ t.a = t.b * r)

-- State the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : IsValidTriangle t)
  (h2 : FormsGeometricProgression t)
  (h3 : Real.cos t.B = 3/4)
  (h4 : t.a * t.c * Real.cos t.B = 3/2) :
  (t.a + t.c = 3) ∧ 
  (Real.cos t.A / Real.sin t.A + Real.cos t.C / Real.sin t.C = 4 * Real.sqrt 7 / 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l488_48874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_packing_height_difference_l488_48855

/-- The diameter of each cylindrical pipe in centimeters -/
def pipe_diameter : ℝ := 12

/-- The number of pipes in each crate -/
def num_pipes : ℕ := 200

/-- The height of Crate A in centimeters -/
def height_crate_A : ℝ := pipe_diameter * num_pipes

/-- The vertical distance between consecutive rows in Crate B -/
noncomputable def vertical_distance_B : ℝ := (Real.sqrt 3 / 2) * pipe_diameter

/-- The height of Crate B in centimeters -/
noncomputable def height_crate_B : ℝ := pipe_diameter + (num_pipes - 1 : ℝ) * vertical_distance_B

/-- The difference in heights between Crate A and Crate B -/
noncomputable def height_difference : ℝ := height_crate_A - height_crate_B

theorem packing_height_difference :
  height_difference = 2388 - 1194 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_packing_height_difference_l488_48855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_exist_in_selection_l488_48802

theorem coprime_exist_in_selection :
  ∀ (S : Finset ℕ),
    S ⊆ Finset.range 2023 →
    S.card = 1012 →
    ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ Nat.Coprime a b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_exist_in_selection_l488_48802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_half_l488_48807

open Real

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / (4^x + 2)

-- State the theorem
theorem f_sum_equals_half (x : ℝ) : f x + f (1 - x) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_half_l488_48807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l488_48884

-- Define the function f(x) = 3^(-x^2)
noncomputable def f (x : ℝ) : ℝ := 3^(-x^2)

-- State the theorem about the range of f
theorem range_of_f :
  (∀ x, 0 < f x ∧ f x ≤ 1) ∧
  (∀ y, 0 < y → y ≤ 1 → ∃ x, f x = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l488_48884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l488_48852

/-- The speed of a train in km/hr, given its length in meters and the time it takes to cross a pole -/
noncomputable def train_speed (length_m : ℝ) (time_s : ℝ) : ℝ :=
  (length_m / 1000) / (time_s / 3600)

/-- Theorem: A train with length 50 meters that crosses a pole in 6 seconds has a speed of 30 km/hr -/
theorem train_speed_theorem :
  train_speed 50 6 = 30 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l488_48852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_preimage_characterization_l488_48861

/-- The function f: ℝ → ℝ defined by f(x) = -x^2 + 2x -/
def f (x : ℝ) : ℝ := -x^2 + 2*x

/-- The set of values in the codomain of f without a preimage -/
def no_preimage : Set ℝ := {y | ∀ x, f x ≠ y}

theorem no_preimage_characterization :
  no_preimage = Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_preimage_characterization_l488_48861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_l488_48815

/-- The function for which we are finding the vertical asymptote -/
noncomputable def f (x : ℝ) : ℝ := (3 * x + 1) / (4 * x - 9)

/-- The x-value where the vertical asymptote occurs -/
noncomputable def asymptote_x : ℝ := 9 / 4

/-- Theorem stating the existence of a vertical asymptote -/
theorem vertical_asymptote :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - asymptote_x| ∧ |x - asymptote_x| < δ → |f x| > 1/ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_l488_48815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_theorem_l488_48835

noncomputable def line_equation (x y : ℝ) : Prop := y = (5 * x - 7) / 6

noncomputable def parameterization (v d : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (v.1 + t * d.1, v.2 + t * d.2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem line_parameterization_theorem :
  ∃ (d : ℝ × ℝ),
    (∀ (x y t : ℝ), x ≥ 3 →
      let point := parameterization (3, 1) d t
      line_equation point.1 point.2 ∧
      distance point (3, 1) = t) ∧
    (d = (Real.sqrt (36/61), 5 * Real.sqrt (36/61) / 6) ∨
     d = (-Real.sqrt (36/61), -5 * Real.sqrt (36/61) / 6)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_theorem_l488_48835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_derivative_at_one_l488_48812

-- Define the functions
noncomputable def f1 (x : ℝ) := -x^2
noncomputable def f2 (x : ℝ) := 1/x
noncomputable def f3 (x : ℝ) := 2*x + 1
noncomputable def f4 (x : ℝ) := Real.sqrt x

-- Define the derivatives
noncomputable def f1' (x : ℝ) := -2*x
noncomputable def f2' (x : ℝ) := -1/x^2
def f3' : ℝ → ℝ := λ _ ↦ 2
noncomputable def f4' (x : ℝ) := 1/(2 * Real.sqrt x)

theorem largest_derivative_at_one :
  abs (f3' 1) > abs (f1' 1) ∧
  abs (f3' 1) > abs (f2' 1) ∧
  abs (f3' 1) > abs (f4' 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_derivative_at_one_l488_48812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_library_distance_l488_48843

/-- Calculates the actual distance between two points given the map distance and scale. -/
noncomputable def actual_distance (map_distance : ℝ) (scale_distance : ℝ) (scale_miles : ℝ) : ℝ :=
  (map_distance * scale_miles) / scale_distance

/-- Theorem: The actual distance between the school and library is 160 miles. -/
theorem school_library_distance : 
  let map_distance : ℝ := 10
  let scale_distance : ℝ := 0.5
  let scale_miles : ℝ := 8
  actual_distance map_distance scale_distance scale_miles = 160 := by
  -- Unfold the definition of actual_distance
  unfold actual_distance
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_library_distance_l488_48843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_C_in_region_Y_l488_48895

-- Define the tiles and regions
inductive Tile | A | B | C | D
inductive Region | X | Y | Z | W

-- Define the sides of a tile
inductive Side | Top | Right | Bottom | Left

-- Define the function that gives the number on each side of each tile
def tileNumber (t : Tile) (s : Side) : ℕ :=
  match t, s with
  | Tile.A, Side.Top => 4
  | Tile.A, Side.Right => 2
  | Tile.A, Side.Bottom => 1
  | Tile.A, Side.Left => 5
  | Tile.B, Side.Top => 2
  | Tile.B, Side.Right => 3
  | Tile.B, Side.Bottom => 5
  | Tile.B, Side.Left => 0
  | Tile.C, Side.Top => 3
  | Tile.C, Side.Right => 1
  | Tile.C, Side.Bottom => 4
  | Tile.C, Side.Left => 2
  | Tile.D, Side.Top => 5
  | Tile.D, Side.Right => 0
  | Tile.D, Side.Bottom => 2
  | Tile.D, Side.Left => 3

-- Define the function that assigns tiles to regions
def tilePlacement : Region → Tile :=
  sorry

-- Define the adjacency relation between regions
def adjacent : Region → Region → Prop :=
  sorry

-- Define the matching condition for adjacent tiles
def matchingAdjacent (r1 r2 : Region) (s1 s2 : Side) : Prop :=
  adjacent r1 r2 → tileNumber (tilePlacement r1) s1 = tileNumber (tilePlacement r2) s2

-- State the theorem
theorem tile_C_in_region_Y :
  tilePlacement Region.Y = Tile.C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_C_in_region_Y_l488_48895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_comparison_l488_48800

theorem sqrt_sum_comparison : Real.sqrt 6 + Real.sqrt 7 > Real.sqrt 3 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_comparison_l488_48800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_power_difference_l488_48831

theorem x_power_difference (x : ℝ) (h : x - 1/x = Real.sqrt 2) :
  x^2184 - (1/x)^2184 = 14^546 - (1/14)^546 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_power_difference_l488_48831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_intersection_l488_48876

/-- The function whose asymptotes we're investigating -/
noncomputable def f (x : ℝ) : ℝ := (x^2 - 6*x + 8) / (x^2 - 6*x + 9)

/-- The x-coordinate of the vertical asymptote -/
def vertical_asymptote : ℝ := 3

/-- The y-coordinate of the horizontal asymptote -/
def horizontal_asymptote : ℝ := 1

/-- The point of intersection of the asymptotes -/
def intersection_point : ℝ × ℝ := (vertical_asymptote, horizontal_asymptote)

theorem asymptotes_intersection :
  intersection_point = (3, 1) := by
  -- Proof goes here
  sorry

#eval intersection_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_intersection_l488_48876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_arrangement_theorem_l488_48808

/-- Represents the number of stamps of each denomination --/
structure StampInventory :=
  (one : Nat) (two : Nat) (three : Nat) (four : Nat) (five : Nat)

/-- Represents a valid arrangement of stamps --/
def ValidArrangement (inventory : StampInventory) : Prop :=
  inventory.one ≥ 1 ∧ inventory.two ≥ 1 ∧ inventory.three ≥ 1 ∧ 
  inventory.four ≥ 1 ∧ inventory.five ≥ 1 ∧
  inventory.one * 1 + inventory.two * 2 + inventory.three * 3 + 
  inventory.four * 4 + inventory.five * 5 = 15

/-- The number of valid arrangements --/
def NumValidArrangements (inventory : StampInventory) : Nat :=
  if inventory.one = 1 ∧ inventory.two = 1 ∧ inventory.three = 1 ∧ 
     inventory.four = 1 ∧ inventory.five = 1 
  then 120 
  else 0

theorem stamp_arrangement_theorem (inventory : StampInventory) :
  inventory = ⟨1, 1, 1, 1, 1⟩ →
  NumValidArrangements inventory = 120 := by
  intro h
  simp [NumValidArrangements, h]
  
#check stamp_arrangement_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_arrangement_theorem_l488_48808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operation_count_l488_48879

/-- Represents a polynomial of degree n -/
def MyPolynomial (α : Type*) := List α

/-- Horner's method for evaluating a polynomial -/
def horner_eval {α : Type*} [Ring α] (p : MyPolynomial α) (x : α) : α :=
  p.foldl (fun acc a => acc * x + a) 0

/-- Count of operations in Horner's method -/
structure OperationCount where
  multiplications : Nat
  additions : Nat

/-- Theorem stating the number of operations in Horner's method -/
theorem horner_method_operation_count {α : Type*} [Ring α] (p : MyPolynomial α) :
  ∃ (count : OperationCount), 
    count.multiplications = p.length - 1 ∧ 
    count.additions = p.length - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operation_count_l488_48879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_perpendicular_line_l488_48881

noncomputable section

/-- The line perpendicular to the given line with y-intercept 4 -/
def perpendicular_line (x y : ℝ) : Prop :=
  y = -(3/4) * x + 4

/-- The original line -/
def original_line (x y : ℝ) : Prop :=
  4 * x - 3 * y = 12

/-- The x-intercept of the perpendicular line -/
def x_intercept : ℝ := 16/3

theorem x_intercept_of_perpendicular_line :
  ∀ x y : ℝ, perpendicular_line x y → y = 0 → x = x_intercept :=
by
  intro x y h1 h2
  unfold perpendicular_line at h1
  unfold x_intercept
  rw [h2] at h1
  linarith

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_perpendicular_line_l488_48881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paperboy_deliveries_l488_48822

/-- Represents the number of permissible delivery sequences for n houses -/
def D : ℕ → ℕ
  | 0 => 1  -- Adding the case for 0 to cover all natural numbers
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | n + 4 => D (n + 3) + D (n + 2) + D (n + 1)

/-- Theorem stating that the number of permissible delivery sequences for 12 houses is 778 -/
theorem paperboy_deliveries : D 12 = 778 := by
  -- Compute the value of D 12
  have h1 : D 4 = 6 := by rfl
  have h2 : D 5 = 11 := by rfl
  have h3 : D 6 = 20 := by rfl
  have h4 : D 7 = 37 := by rfl
  have h5 : D 8 = 68 := by rfl
  have h6 : D 9 = 125 := by rfl
  have h7 : D 10 = 230 := by rfl
  have h8 : D 11 = 423 := by rfl
  have h9 : D 12 = 778 := by rfl
  exact h9


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paperboy_deliveries_l488_48822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_volume_range_max_traffic_volume_max_traffic_volume_at_four_l488_48849

-- Define the traffic volume function
noncomputable def trafficVolume (v : ℝ) : ℝ := 25 * v / (v^2 - 5*v + 16)

-- Theorem for the range of v when y ≥ 5
theorem traffic_volume_range (v : ℝ) (h : v > 0) :
  trafficVolume v ≥ 5 ↔ 2 ≤ v ∧ v ≤ 8 := by
  sorry

-- Theorem for the maximum traffic volume and corresponding speed
theorem max_traffic_volume :
  ∃ (v : ℝ), v > 0 ∧ trafficVolume v = 25/3 ∧ 
  ∀ (u : ℝ), u > 0 → trafficVolume u ≤ 25/3 := by
  sorry

-- Theorem stating that the maximum occurs at v = 4
theorem max_traffic_volume_at_four :
  trafficVolume 4 = 25/3 ∧
  ∀ (v : ℝ), v > 0 → trafficVolume v ≤ 25/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_volume_range_max_traffic_volume_max_traffic_volume_at_four_l488_48849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l488_48827

/-- The radius of the inscribed circle in a triangle with sides a, b, and c. -/
noncomputable def inscribedCircleRadius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) / s

/-- Theorem: The radius of the inscribed circle in a triangle with sides 30, 26, and 28 is 8√6. -/
theorem inscribed_circle_radius_specific_triangle :
  inscribedCircleRadius 30 26 28 = 8 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l488_48827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_negative_five_fourths_l488_48854

/-- Represents the probabilities of rolling each number on the biased die -/
def die_probabilities : Fin 6 → ℚ
  | 0 => 1/12
  | 1 => 1/12
  | 2 => 1/12
  | 3 => 1/4
  | 4 => 1/4
  | 5 => 1/4

/-- Represents the payoff for each roll outcome -/
def roll_payoff : Fin 6 → ℚ
  | 0 => 4
  | 1 => 4
  | 2 => 4
  | 3 => -3
  | 4 => -3
  | 5 => -3

/-- The expected value of one roll of the biased die -/
def expected_value : ℚ := Finset.sum Finset.univ (λ i => die_probabilities i * roll_payoff i)

/-- Theorem stating that the expected value of one roll is -5/4 (equivalent to -$1.25) -/
theorem expected_value_is_negative_five_fourths :
  expected_value = -5/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_negative_five_fourths_l488_48854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l488_48816

/-- Represents a university --/
inductive University : Type
| A
| B
| C

/-- Represents a graduate --/
structure Graduate :=
(university : University)

/-- Represents a group of two graduates --/
structure GraduateGroup :=
(grad1 : Graduate)
(grad2 : Graduate)
(different_universities : grad1.university ≠ grad2.university)

/-- Represents an arrangement of graduates to schools --/
structure Arrangement :=
(school1 : GraduateGroup)
(school2 : GraduateGroup)
(school3 : GraduateGroup)
(all_different : school1.grad1.university ≠ school1.grad2.university ∧
                 school2.grad1.university ≠ school2.grad2.university ∧
                 school3.grad1.university ≠ school3.grad2.university)

/-- The number of graduates from each university --/
def graduates_per_university : Nat := 2

/-- The number of universities --/
def num_universities : Nat := 3

/-- The number of schools --/
def num_schools : Nat := 3

/-- The total number of graduates --/
def total_graduates : Nat := graduates_per_university * num_universities

/-- Assume that Arrangement is finite --/
instance : Fintype Arrangement := sorry

theorem arrangement_count : 
  Fintype.card Arrangement = 48 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l488_48816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_third_side_possible_length_l488_48818

-- Define Triangle as a structure
structure Triangle where
  sides : Finset ℝ
  side_count : sides.card = 3
  positive_sides : ∀ s ∈ sides, 0 < s

-- Triangle inequality theorem
theorem triangle_inequality (a b c : ℝ) : 
  (0 < a ∧ 0 < b ∧ 0 < c) → (a < b + c ∧ b < a + c ∧ c < a + b) ↔ 
  ∃ (t : Triangle), t.sides = {a, b, c} := by sorry

-- Theorem for the specific triangle in the problem
theorem third_side_possible_length : 
  ∃ (t : Triangle), t.sides = {5, 6, 8} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_third_side_possible_length_l488_48818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fathers_age_l488_48819

theorem fathers_age (sebastian_age sister_age fathers_age : ℕ) :
  sebastian_age = 40 →
  sister_age = sebastian_age - 10 →
  (sebastian_age - 5 + (sister_age - 5) = 3 * (fathers_age - 5) / 4) →
  fathers_age = 85 :=
by
  sorry

#check fathers_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fathers_age_l488_48819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_log_half_4_l488_48864

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_positive : ∀ x : ℝ, x > 0 → f x = 3^x

-- State the theorem
theorem f_log_half_4 : f (Real.log 4 / Real.log (1/2)) = -9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_log_half_4_l488_48864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sides_denominator_product_l488_48844

/-- Given coprime integers a, b, c representing triangle sides, if the specified
    fractions are integers, then the product of denominators or its double is a perfect square. -/
theorem triangle_sides_denominator_product (a b c : ℤ) : 
  (Int.gcd (Int.gcd a b) c = 1) →
  (∃ k : ℤ, (a^2 + b^2 - c^2) = k * (a + b - c)) →
  (∃ l : ℤ, (b^2 + c^2 - a^2) = l * (b + c - a)) →
  (∃ m : ℤ, (c^2 + a^2 - b^2) = m * (c + a - b)) →
  (∃ n : ℤ, (a + b - c) * (b + c - a) * (c + a - b) = n^2) ∨
  (∃ n : ℤ, 2 * (a + b - c) * (b + c - a) * (c + a - b) = n^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sides_denominator_product_l488_48844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_configurations_count_l488_48894

/-- Represents a configuration of a rectangle and a square -/
structure Configuration where
  rect_width : ℕ
  rect_height : ℕ
  square_side : ℕ

/-- Checks if a configuration is valid according to the problem conditions -/
def is_valid_configuration (c : Configuration) : Bool :=
  c.rect_width < c.rect_height &&
  2 * (c.rect_width + c.rect_height + 2 * c.square_side) = 120

/-- Counts the number of valid configurations -/
def count_valid_configurations : ℕ :=
  (List.range 60).bind (fun w =>
    (List.range 60).bind (fun h =>
      (List.range 60).map (fun s =>
        Configuration.mk w h s
      )
    )
  )
  |>.filter is_valid_configuration
  |>.length

/-- The main theorem stating that there are 196 valid configurations -/
theorem valid_configurations_count :
  count_valid_configurations = 196 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_configurations_count_l488_48894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_women_group_size_l488_48863

/-- The number of women in the first group that takes 1 hour to dig a ditch -/
def W : ℕ := 6

/-- The time taken by 18 women to dig a ditch -/
def time_18_women : ℚ := 1/3

/-- The work done by one woman in one hour in the first group -/
def work_rate_1 : ℚ := 1 / W

/-- The work done by one woman in the second group (18 women) -/
def work_rate_2 : ℚ := 1 / (18 * time_18_women)

theorem women_group_size :
  work_rate_1 = work_rate_2 → W = 6 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_women_group_size_l488_48863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_parabola_l488_48806

/-- The equation x^2 + 4y^2 * sin(θ) = 1 for any real θ -/
def equation (θ : ℝ) (x y : ℝ) : Prop := x^2 + 4 * y^2 * Real.sin θ = 1

/-- Definition of a parabola in general form -/
def is_parabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧
  ∀ x y, f x y ↔ a * y = b * x^2 + c * x + d * y + e

theorem not_parabola :
  ¬ ∃ θ, is_parabola (equation θ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_parabola_l488_48806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_F1PF2_l488_48886

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the foci
noncomputable def F1 : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def F2 : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define a point on the hyperbola
variable (P : ℝ × ℝ)

-- Define the angle F1PF2
noncomputable def angle_F1PF2 : ℝ := 60 * Real.pi / 180

-- Theorem statement
theorem area_triangle_F1PF2 (h1 : hyperbola P.1 P.2) 
  (h2 : Real.cos angle_F1PF2 = 1/2) : 
  let d1 := Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)
  let d2 := Real.sqrt ((P.2 - F2.1)^2 + (P.2 - F2.2)^2)
  let s := (1/2) * d1 * d2 * Real.sin angle_F1PF2
  s = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_F1PF2_l488_48886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_owners_without_motorcycle_l488_48809

theorem car_owners_without_motorcycle (total : ℕ) (car_owners : ℕ) (motorcycle_owners : ℕ)
  (h_total : total = 500)
  (h_car : car_owners = 480)
  (h_motorcycle : motorcycle_owners = 120)
  (h_all_own : ∀ a, a < total → (a < car_owners ∨ a < motorcycle_owners))
  : car_owners - (car_owners + motorcycle_owners - total) = 380 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_owners_without_motorcycle_l488_48809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l488_48889

-- Define the curve
noncomputable def on_curve (x y : ℝ) : Prop := 2 * x = Real.sqrt (4 + y^2)

-- Define points A and B
noncomputable def A : ℝ × ℝ := (-Real.sqrt 5, 0)
noncomputable def B : ℝ × ℝ := (Real.sqrt 5, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem hyperbola_property (P : ℝ × ℝ) :
  on_curve P.1 P.2 → distance P B = 2 → distance P A = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l488_48889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l488_48801

noncomputable section

open Real

def f (x : ℝ) := 2 * sin x * (cos x - cos (x + π / 2)) - 1

def has_maximum_no_minimum_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  (∃ x ∈ S, ∀ y ∈ S, f y ≤ f x) ∧ (¬∃ x ∈ S, ∀ y ∈ S, f y ≥ f x)

theorem f_properties :
  (f (π / 6) = Real.sqrt 3 / 2 - 1 / 2) ∧
  (∀ m : ℝ, (has_maximum_no_minimum_on f (Set.Ioo 0 m)) ↔ m ∈ Set.Ioo (3 * π / 8) (7 * π / 8)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l488_48801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_margo_walked_two_miles_l488_48872

/-- The total distance Margo walked given her trip details -/
noncomputable def total_distance (time_to_friend time_from_friend avg_speed : ℝ) : ℝ :=
  avg_speed * (time_to_friend + time_from_friend) / 60

/-- Theorem stating that Margo walked 2 miles in total -/
theorem margo_walked_two_miles :
  let time_to_friend : ℝ := 15
  let time_from_friend : ℝ := 25
  let avg_speed : ℝ := 3
  total_distance time_to_friend time_from_friend avg_speed = 2 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval total_distance 15 25 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_margo_walked_two_miles_l488_48872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_properties_l488_48847

/-- A function from pairs of nonzero rationals to positive reals satisfying certain properties -/
structure SpecialFunction where
  f : (ℚ × ℚ) → ℝ
  pos : ∀ a b, a ≠ 0 → b ≠ 0 → f (a, b) > 0
  mult_left : ∀ a b c, a ≠ 0 → b ≠ 0 → c ≠ 0 → f (a * b, c) = f (a, c) * f (b, c)
  mult_right : ∀ a b c, a ≠ 0 → b ≠ 0 → c ≠ 0 → f (c, a * b) = f (c, a) * f (c, b)
  special_value : ∀ a, a ≠ 0 → a ≠ 1 → f (a, 1 - a) = 1

/-- Main theorem about properties of the special function -/
theorem special_function_properties (sf : SpecialFunction) :
  (∀ a, a ≠ 0 → sf.f (a, a) = 1) ∧
  (∀ a, a ≠ 0 → sf.f (a, -a) = 1) ∧
  (∀ a b, a ≠ 0 → b ≠ 0 → sf.f (a, b) * sf.f (b, a) = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_properties_l488_48847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l488_48868

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log ((2 - x) / (2 + x))

-- State the theorem about the domain of f
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -2 < x ∧ x < 2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l488_48868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_two_l488_48857

/-- The coefficient of x^2 in the expansion of (a + x)^5 -/
noncomputable def coefficient_x_squared (a : ℝ) : ℝ := (5 * (5 - 1)) / 2 * a^3

/-- Theorem stating that a = 2 given the coefficient of x^2 is 80 -/
theorem a_equals_two (h : coefficient_x_squared a = 80) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_two_l488_48857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l488_48834

/-- The function f(x) = me^x - x - 1 -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.exp x - x - 1

/-- The theorem stating that the tangent line to f(x) at (0, 1) is x - y + 1 = 0 -/
theorem tangent_line_at_origin (m : ℝ) :
  (f m 0 = 1) →
  ∃ (tangent_line : ℝ → ℝ → Prop),
    (tangent_line 0 1) ∧
    (∀ x y, tangent_line x y ↔ x - y + 1 = 0) ∧
    (∀ x, (tangent_line x (f m x)) ↔ x = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l488_48834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_n_terms_l488_48832

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1

/-- Defines the sum of the first n terms of a geometric sequence -/
noncomputable def sum_n (seq : GeometricSequence) (n : ℕ) : ℝ :=
  (seq.a 1) * (1 - (seq.a 2 / seq.a 1)^n) / (1 - seq.a 2 / seq.a 1)

theorem geometric_sequence_n_terms 
  (seq : GeometricSequence) 
  (n : ℕ)
  (h1 : seq.a 1 + seq.a n = 82)
  (h2 : seq.a 3 * seq.a (n - 2) = 81)
  (h3 : sum_n seq n = 121) :
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_n_terms_l488_48832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_pairing_probability_l488_48850

/-- Represents a pairing of shoes -/
def ShoePairing := Fin 8 → Fin 8

/-- Checks if a pairing is valid according to the problem conditions -/
def is_valid_pairing (p : ShoePairing) : Prop :=
  ∀ k < 4, ∀ s : Finset (Fin 8), s.card = k → 
    (∃ i ∈ s, p i ∉ s) ∨ (∃ i ∉ s, p i ∈ s)

/-- The total number of possible pairings -/
def total_pairings : ℕ := Nat.factorial 8

/-- The number of valid pairings -/
def valid_pairings : ℕ := 5670

theorem shoe_pairing_probability :
  (valid_pairings : ℚ) / total_pairings = 9 / 64 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_pairing_probability_l488_48850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_worked_35_hours_l488_48814

/-- Represents the payment structure and hours worked for employees --/
structure EmployeeData where
  regularRate : ℚ
  regularHours : ℚ
  overtimeRate : ℚ
  hoursWorked : ℚ

/-- Calculates the pay for an employee based on their payment structure and hours worked --/
def calculatePay (e : EmployeeData) : ℚ :=
  if e.hoursWorked ≤ e.regularHours then
    e.regularRate * e.hoursWorked
  else
    e.regularRate * e.regularHours + e.overtimeRate * (e.hoursWorked - e.regularHours)

theorem harry_worked_35_hours 
  (x : ℚ)
  (harry : EmployeeData)
  (james : EmployeeData)
  (olivia : EmployeeData)
  (h1 : harry.regularRate = x ∧ harry.regularHours = 21 ∧ harry.overtimeRate = 3/2 * x)
  (h2 : james.regularRate = x ∧ james.regularHours = 40 ∧ james.overtimeRate = 2 * x ∧ james.hoursWorked = 41)
  (h3 : olivia.regularRate = x ∧ olivia.regularHours = 15 ∧ olivia.overtimeRate = 2 * x ∧ olivia.hoursWorked = 26)
  (h4 : calculatePay harry + calculatePay james + calculatePay olivia = 5000)
  (h5 : calculatePay harry = calculatePay james)
  : harry.hoursWorked = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_worked_35_hours_l488_48814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_property_h_at_one_h_at_three_l488_48896

/-- A linear function satisfying specific conditions -/
noncomputable def h (x : ℝ) : ℝ := -Real.sqrt 3 * x + (-3 * Real.sqrt 3 + 3)

/-- The inverse of h -/
noncomputable def h_inv (x : ℝ) : ℝ := (x + 3 * Real.sqrt 3 - 3) / (-Real.sqrt 3)

/-- Property of h: h(x) = 3 * h⁻¹(x) + 9 -/
theorem h_property (x : ℝ) : h x = 3 * h_inv x + 9 := by sorry

/-- h(1) = 5 -/
theorem h_at_one : h 1 = 5 := by sorry

/-- h(3) = -6√3 + 3 -/
theorem h_at_three : h 3 = -6 * Real.sqrt 3 + 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_property_h_at_one_h_at_three_l488_48896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_subset_l488_48851

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 7}
def C (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem set_operations_and_subset :
  (A ∩ B = Set.Icc 3 7) ∧
  (A ∪ B = Set.Ioo 2 10) ∧
  ((Set.univ \ A) ∩ (Set.univ \ B) = Set.Iic 2 ∪ Set.Ici 10) ∧
  (∀ a : ℝ, A ⊆ C a → a < 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_subset_l488_48851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoidalPrismVolume_l488_48845

/-- A right prism with a trapezoidal base -/
structure TrapezoidalPrism where
  S₁ : ℝ  -- Area of one parallel side face
  S₂ : ℝ  -- Area of the other parallel side face
  h : ℝ   -- Height of the trapezoidal base
  H : ℝ   -- Height of the prism

/-- Volume of a trapezoidal prism -/
noncomputable def volume (p : TrapezoidalPrism) : ℝ := ((p.S₁ + p.S₂) * p.h) / 2

/-- Theorem: The volume of a right prism with a trapezoidal base is equal to 
    the product of the arithmetic mean of its parallel side face areas and 
    the height of the trapezoidal base -/
theorem trapezoidalPrismVolume (p : TrapezoidalPrism) : 
  volume p = ((p.S₁ + p.S₂) * p.h) / 2 := by
  -- Unfold the definition of volume
  unfold volume
  -- The rest of the proof is trivial as it's just the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoidalPrismVolume_l488_48845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_tan_sin_l488_48880

theorem arctan_tan_sin (θ : Real) (h : θ = Real.arctan (5/3)) :
  (Real.tan θ = 5/3) ∧ (Real.sin θ = 5 * Real.sqrt 34 / 34) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_tan_sin_l488_48880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_ball_inflation_l488_48829

theorem soccer_ball_inflation (total_balls : ℕ) 
  (hole_percentage : ℚ) (overinflated_percentage : ℚ) : 
  total_balls = 100 → 
  hole_percentage = 40 / 100 → 
  overinflated_percentage = 20 / 100 → 
  (total_balls - total_balls * hole_percentage.num / hole_percentage.den) * (1 - overinflated_percentage) = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_ball_inflation_l488_48829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l488_48859

/-- The distance between two parallel planes -/
noncomputable def distance_between_planes (a b c d : ℝ) (k : ℝ) : ℝ :=
  |k - d| / Real.sqrt (a^2 + b^2 + c^2)

/-- First plane equation: 3x + y - z + 3 = 0 -/
def plane1 (x y z : ℝ) : Prop :=
  3*x + y - z + 3 = 0

/-- Second plane equation: 6x + 2y - 2z + 7 = 0 -/
def plane2 (x y z : ℝ) : Prop :=
  6*x + 2*y - 2*z + 7 = 0

theorem distance_between_specific_planes :
  distance_between_planes 3 1 (-1) 3 (7/2) = 1 / (2 * Real.sqrt 11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l488_48859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_minus_sin_cos_alpha_l488_48871

theorem sin_squared_minus_sin_cos_alpha (α : ℝ) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) : 
  Real.sin α ^ 2 - Real.sin α * Real.cos α = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_minus_sin_cos_alpha_l488_48871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_chests_distribution_eight_chests_distribution_l488_48840

/-- Represents the number of coins in each chest -/
def Coins := ℕ

/-- Represents a collection of chests -/
def Chests := List ℕ

/-- Returns true if coins can be evenly distributed among the given number of chests -/
def canDistribute (chests : Chests) (n : ℕ) : Prop :=
  ∃ (k : ℕ), (chests.sum = n * k) ∧ (n ≤ chests.length)

theorem six_chests_distribution (chests : Chests) :
  (∀ (n : ℕ), 2 ≤ n ∧ n < 6 → canDistribute chests n) →
  canDistribute chests 6 := by
  sorry

theorem eight_chests_distribution (chests : Chests) :
  (∀ (n : ℕ), 2 ≤ n ∧ n < 8 → canDistribute chests n) →
  ¬ (∀ chests : Chests, canDistribute chests 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_chests_distribution_eight_chests_distribution_l488_48840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_condition_implies_60_degree_angle_l488_48873

/-- Given a triangle with sides a, b, c satisfying (a + b + c)(a + b - c) = 2ab + a^2,
    prove that the angle opposite side c is 60 degrees. -/
theorem triangle_special_condition_implies_60_degree_angle
  (a b c : ℝ)
  (h : (a + b + c) * (a + b - c) = 2 * a * b + a^2) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  C = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_condition_implies_60_degree_angle_l488_48873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l488_48836

theorem power_equation (a b : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : b^n = 4) :
  a^(2*m) / b^(2*n) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l488_48836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_odd_half_l488_48870

-- Define a set of positive integers
def PositiveIntegerSet : Type := {s : Set ℕ // s.Nonempty ∧ ∃ (x y : ℕ), x ∈ s ∧ y ∈ s ∧ Even x ∧ Odd y}

-- Define a function to check if a number is odd
def isOdd (n : ℕ) : Prop := ¬(Even n)

-- Define the probability of selecting an odd number from the set
noncomputable def probOdd (s : PositiveIntegerSet) : ℝ := 1 / 2

-- Define the probability of the sum of two randomly chosen numbers being odd
noncomputable def probSumOdd (s : PositiveIntegerSet) : ℝ := 1 / 2

-- State the theorem
theorem prob_sum_odd_half (s : PositiveIntegerSet) : 
  probSumOdd s = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_odd_half_l488_48870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stronger_correlation_as_r_approaches_one_l488_48887

-- Define the linear correlation coefficient
noncomputable def linear_correlation_coefficient (X Y : Type) [NormedAddCommGroup X] [NormedAddCommGroup Y] : ℝ := 
  sorry

-- Define the strength of linear correlation
noncomputable def linear_correlation_strength (r : ℝ) : ℝ := 
  sorry

-- Theorem statement
theorem stronger_correlation_as_r_approaches_one :
  ∀ (ε : ℝ), ε > 0 → ε < 1 →
  ∀ (X Y : Type) [NormedAddCommGroup X] [NormedAddCommGroup Y],
    let r := linear_correlation_coefficient X Y
    (1 - ε < |r| ∧ |r| ≤ 1) →
    linear_correlation_strength r > linear_correlation_strength (1 - ε) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stronger_correlation_as_r_approaches_one_l488_48887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l488_48841

theorem negation_of_sin_inequality :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x₀ : ℝ, Real.sin x₀ ≥ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l488_48841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_is_reals_iff_k_in_zero_one_l488_48898

-- Define the function f as noncomputable
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.sqrt (k * x^2 - 6 * k * x + (k + 8))

-- State the theorem
theorem domain_is_reals_iff_k_in_zero_one :
  ∀ k : ℝ, (∀ x : ℝ, ∃ y : ℝ, f k x = y) ↔ (0 ≤ k ∧ k ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_is_reals_iff_k_in_zero_one_l488_48898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l488_48883

/-- Given a point M with polar coordinates (5, 2π/3), prove that its rectangular coordinates are (-5/2, 5√3/2) -/
theorem polar_to_rectangular_conversion (M : Real × Real) (h : M = (5, 2 * Real.pi / 3)) :
  (M.1 * Real.cos M.2, M.1 * Real.sin M.2) = (-5/2, 5 * Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l488_48883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangles_intersecting_lines_l488_48803

/-- The number of triangles formed by n intersecting lines -/
def num_triangles (n : ℕ) : ℚ := (2 * n - 2) / 3

/-- Theorem: For n ≥ 3 intersecting lines, the number of triangles formed is at least (2n - 2) / 3 -/
theorem min_triangles_intersecting_lines (n : ℕ) (h : n ≥ 3) :
  ∃ (t : ℕ), (t : ℚ) ≥ num_triangles n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangles_intersecting_lines_l488_48803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_value_l488_48856

-- Define the sets A, B, and C
variable (A B C : Finset (Fin 100))

-- Define the conditions
variable (h1 : A.card = 50)
variable (h2 : B.card = 60)
variable (h3 : (A ∩ B).card = 25)
variable (h4 : A.card + B.card + C.card = (A ∪ B ∪ C).card)

-- State the theorem
theorem min_intersection_value :
  25 ≤ (A ∩ B ∩ C).card :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_value_l488_48856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l488_48811

/-- Represents an ellipse with equation x²/4 + y²/m² = 1 and m > 0 -/
structure Ellipse where
  m : ℝ
  m_pos : m > 0

/-- The distance from the center to a focus of the ellipse -/
def focal_distance (e : Ellipse) : ℝ := 1

/-- The semi-major axis length of the ellipse -/
def semi_major_axis (e : Ellipse) : ℝ := 2

/-- The semi-minor axis length of the ellipse -/
def semi_minor_axis (e : Ellipse) : ℝ := e.m

/-- Theorem stating that for an ellipse with equation x²/4 + y²/m² = 1,
    m > 0, and one focal point at (1,0), the value of m is √3 -/
theorem ellipse_m_value (e : Ellipse) :
  focal_distance e = 1 →
  semi_major_axis e = 2 →
  semi_minor_axis e = e.m →
  e.m = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l488_48811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angles_relationship_l488_48817

/-- Two dihedral angles -/
structure DihedralAngle where
  edge : Set ℝ  -- Representing a line as a set of points in ℝ
  face1 : Set ℝ × Set ℝ × Set ℝ  -- Representing a plane as a triple of sets in ℝ
  face2 : Set ℝ × Set ℝ × Set ℝ

/-- The relationship between two angles -/
inductive AngleRelationship
  | Equal
  | Complementary

/-- Represents perpendicularity between planes -/
def Perpendicular (p1 p2 : Set ℝ × Set ℝ × Set ℝ) : Prop := sorry

/-- Represents parallelism between lines -/
def Parallel (l1 l2 : Set ℝ) : Prop := sorry

theorem dihedral_angles_relationship (d1 d2 : DihedralAngle) 
  (h1 : Perpendicular d1.face1 d2.face1)
  (h2 : Perpendicular d1.face2 d2.face2)
  (h3 : Parallel d1.edge d2.edge) :
  ∃ r : AngleRelationship, r = AngleRelationship.Equal ∨ r = AngleRelationship.Complementary :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angles_relationship_l488_48817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_equals_38_l488_48875

/-- The function y in terms of x, a, and b -/
noncomputable def y (x a b : ℝ) : ℝ := a + b / x

/-- The theorem stating that if y(-2) = 2 and y(-3) = 6, then a + b = 38 -/
theorem a_plus_b_equals_38 (a b : ℝ) 
  (h1 : y (-2) a b = 2) 
  (h2 : y (-3) a b = 6) : 
  a + b = 38 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_equals_38_l488_48875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insulation_cost_minimization_l488_48826

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := 800 / (3 * x + 5) + 6 * x

-- State the theorem
theorem insulation_cost_minimization :
  ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 10 ∧
  f x = 70 ∧
  ∀ (y : ℝ), 1 ≤ y ∧ y ≤ 10 → f y ≥ f x :=
by
  -- The proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_insulation_cost_minimization_l488_48826
