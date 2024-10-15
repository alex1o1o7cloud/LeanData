import Mathlib

namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2270_227005

theorem complex_fraction_equality : (1 - I) * (1 + 2*I) / (1 + I) = 2 - I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2270_227005


namespace NUMINAMATH_CALUDE_range_of_a_for_decreasing_f_l2270_227017

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x + 3 else 2 * a / x

-- State the theorem
theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) → 0 < a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_decreasing_f_l2270_227017


namespace NUMINAMATH_CALUDE_max_abs_difference_complex_l2270_227025

theorem max_abs_difference_complex (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 1) 
  (h₂ : z₁ + z₂ = Complex.I * 2) : 
  ∃ (M : ℝ), M = 4 ∧ ∀ (w₁ w₂ : ℂ), Complex.abs w₁ = 1 → w₁ + w₂ = Complex.I * 2 → 
    Complex.abs (w₁ - w₂) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_abs_difference_complex_l2270_227025


namespace NUMINAMATH_CALUDE_optimal_screen_arrangement_l2270_227007

/-- The optimal arrangement of two screens in a corner --/
theorem optimal_screen_arrangement (screen_length : ℝ) (h_length : screen_length = 4) :
  let max_area := 8 * (Real.sqrt 2 + 1)
  let optimal_angle := π / 4
  ∀ angle : ℝ, 0 < angle ∧ angle < π / 2 →
    screen_length * screen_length * Real.sin angle / 2 ≤ max_area ∧
    (screen_length * screen_length * Real.sin angle / 2 = max_area ↔ angle = optimal_angle) :=
by sorry

end NUMINAMATH_CALUDE_optimal_screen_arrangement_l2270_227007


namespace NUMINAMATH_CALUDE_min_value_sum_of_roots_l2270_227074

theorem min_value_sum_of_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt (a^2 + 1/a) + Real.sqrt (b^2 + 1/b) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_roots_l2270_227074


namespace NUMINAMATH_CALUDE_cos_a_minus_pi_fourth_l2270_227039

theorem cos_a_minus_pi_fourth (a : ℝ) (ha : a ∈ Set.Ioo 0 2) (h_tan : Real.tan a = 2) :
  Real.cos (a - π / 4) = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_a_minus_pi_fourth_l2270_227039


namespace NUMINAMATH_CALUDE_middle_digit_zero_l2270_227071

/-- Represents a three-digit number in base 8 -/
structure Base8Number where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds < 8 ∧ tens < 8 ∧ ones < 8

/-- Represents a three-digit number in base 10 -/
structure Base10Number where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- Converts a Base8Number to its decimal representation -/
def toDecimal (n : Base8Number) : Nat :=
  512 * n.hundreds + 64 * n.tens + 8 * n.ones

/-- Converts a Base10Number to its decimal representation -/
def fromBase10 (n : Base10Number) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Checks if the digits of a Base10Number are a right rotation of a Base8Number -/
def isRightRotation (n8 : Base8Number) (n10 : Base10Number) : Prop :=
  n10.hundreds = n8.tens ∧ n10.tens = n8.ones ∧ n10.ones = n8.hundreds

theorem middle_digit_zero (n8 : Base8Number) (n10 : Base10Number) 
  (h : toDecimal n8 = fromBase10 n10) 
  (rot : isRightRotation n8 n10) : 
  n10.tens = 0 := by
  sorry

end NUMINAMATH_CALUDE_middle_digit_zero_l2270_227071


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2270_227080

theorem geometric_series_sum (a r : ℝ) (ha : a = (1 : ℝ) / 2) (hr : r = (1 : ℝ) / 2) :
  (∑' n, a * r ^ n) = 1 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2270_227080


namespace NUMINAMATH_CALUDE_soap_box_width_maximizes_boxes_l2270_227010

/-- The width of a soap box that maximizes the number of boxes in a carton. -/
def soap_box_width : ℝ :=
  let carton_volume : ℝ := 30 * 42 * 60
  let max_boxes : ℕ := 360
  let soap_box_length : ℝ := 7
  let soap_box_height : ℝ := 5
  6

/-- Theorem stating that the calculated width maximizes the number of soap boxes in the carton. -/
theorem soap_box_width_maximizes_boxes (carton_volume : ℝ) (max_boxes : ℕ) 
    (soap_box_length soap_box_height : ℝ) :
  carton_volume = 30 * 42 * 60 →
  max_boxes = 360 →
  soap_box_length = 7 →
  soap_box_height = 5 →
  soap_box_width * soap_box_length * soap_box_height * max_boxes = carton_volume :=
by sorry

end NUMINAMATH_CALUDE_soap_box_width_maximizes_boxes_l2270_227010


namespace NUMINAMATH_CALUDE_a0_value_l2270_227031

theorem a0_value (x : ℝ) (a0 a1 a2 a3 a4 a5 : ℝ) 
  (h : ∀ x, (x + 1)^5 = a0 + a1*(x - 1) + a2*(x - 1)^2 + a3*(x - 1)^3 + a4*(x - 1)^4 + a5*(x - 1)^5) : 
  a0 = 32 := by
sorry

end NUMINAMATH_CALUDE_a0_value_l2270_227031


namespace NUMINAMATH_CALUDE_weight_ratio_proof_l2270_227056

/-- Given the weights of Antoinette and Rupert, prove their weight ratio -/
theorem weight_ratio_proof (A R : ℚ) (k : ℚ) : 
  A = 63 → 
  A + R = 98 → 
  A = k * R - 7 → 
  A / R = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_weight_ratio_proof_l2270_227056


namespace NUMINAMATH_CALUDE_largest_k_inequality_l2270_227020

theorem largest_k_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a + b) * (a * b + 1) * (b + 1) ≥ (27 / 4) * a * b^2 ∧ 
  ∀ k : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → (a + b) * (a * b + 1) * (b + 1) ≥ k * a * b^2) → k ≤ 27 / 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_k_inequality_l2270_227020


namespace NUMINAMATH_CALUDE_min_blue_eyes_and_backpack_l2270_227012

theorem min_blue_eyes_and_backpack (total : Nat) (blue_eyes : Nat) (backpacks : Nat)
  (h1 : total = 35)
  (h2 : blue_eyes = 15)
  (h3 : backpacks = 25)
  (h4 : blue_eyes ≤ total)
  (h5 : backpacks ≤ total) :
  ∃ (both : Nat), both ≥ blue_eyes + backpacks - total ∧ both = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_blue_eyes_and_backpack_l2270_227012


namespace NUMINAMATH_CALUDE_system_and_expression_proof_l2270_227038

theorem system_and_expression_proof :
  -- Part 1: System of equations
  (∃ x y : ℚ, 2 * x - y = -4 ∧ 4 * x - 5 * y = -23 ∧ x = 1/2 ∧ y = 5) ∧
  -- Part 2: Expression evaluation
  (let x : ℚ := 2
   let y : ℚ := -1
   (x - 3 * y)^2 - (2 * x + y) * (y - 2 * x) = 40) := by
sorry

end NUMINAMATH_CALUDE_system_and_expression_proof_l2270_227038


namespace NUMINAMATH_CALUDE_tomato_shipment_ratio_l2270_227067

/-- Calculates the ratio of the second shipment to the first shipment of tomatoes -/
def shipment_ratio (initial_shipment : ℕ) (sold : ℕ) (rotted : ℕ) (final_amount : ℕ) : ℚ :=
  let remaining := initial_shipment - sold - rotted
  let second_shipment := final_amount - remaining
  (second_shipment : ℚ) / initial_shipment

/-- Proves that the ratio of the second shipment to the first shipment is 2:1 -/
theorem tomato_shipment_ratio :
  shipment_ratio 1000 300 200 2500 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tomato_shipment_ratio_l2270_227067


namespace NUMINAMATH_CALUDE_water_force_on_trapezoidal_dam_water_force_on_trapezoidal_dam_proof_l2270_227065

/-- The force exerted by water on a dam with an isosceles trapezoidal cross-section --/
theorem water_force_on_trapezoidal_dam
  (ρ : Real) -- density of water
  (g : Real) -- acceleration due to gravity
  (a : Real) -- top base of trapezoid
  (b : Real) -- bottom base of trapezoid
  (h : Real) -- height of trapezoid
  (hρ : ρ = 1000) -- density of water in kg/m³
  (hg : g = 10) -- acceleration due to gravity in m/s²
  (ha : a = 6.3) -- top base in meters
  (hb : b = 10.2) -- bottom base in meters
  (hh : h = 4.0) -- height in meters
  : Real :=
  -- The force F in Newtons
  608000

/-- Proof of the theorem --/
theorem water_force_on_trapezoidal_dam_proof
  (ρ g a b h : Real)
  (hρ : ρ = 1000)
  (hg : g = 10)
  (ha : a = 6.3)
  (hb : b = 10.2)
  (hh : h = 4.0)
  : water_force_on_trapezoidal_dam ρ g a b h hρ hg ha hb hh = 608000 := by
  sorry

end NUMINAMATH_CALUDE_water_force_on_trapezoidal_dam_water_force_on_trapezoidal_dam_proof_l2270_227065


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2270_227026

def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | (x+4)*(x-2) > 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2270_227026


namespace NUMINAMATH_CALUDE_min_value_sum_l2270_227069

theorem min_value_sum (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_10 : a + b + c + d + e + f = 10) : 
  1/a + 9/b + 16/c + 25/d + 36/e + 49/f ≥ 67.6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l2270_227069


namespace NUMINAMATH_CALUDE_angle_properties_l2270_227024

-- Define the angle θ
def θ : Real := sorry

-- Define the point through which the terminal side of θ passes
def terminal_point : ℝ × ℝ := (4, -3)

-- State the theorem
theorem angle_properties (θ : Real) (terminal_point : ℝ × ℝ) :
  terminal_point = (4, -3) →
  Real.tan θ = -3/4 ∧
  (Real.sin (θ + Real.pi/2) + Real.cos θ) / (Real.sin θ - Real.cos (θ - Real.pi)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_angle_properties_l2270_227024


namespace NUMINAMATH_CALUDE_factoring_quadratic_l2270_227000

theorem factoring_quadratic (a : ℝ) : a^2 - 4*a + 3 = (a - 1) * (a - 3) := by
  sorry

#check factoring_quadratic

end NUMINAMATH_CALUDE_factoring_quadratic_l2270_227000


namespace NUMINAMATH_CALUDE_a6_value_l2270_227059

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem a6_value (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2) ^ 2 - 8 * (a 2) + 4 = 0 →
  (a 10) ^ 2 - 8 * (a 10) + 4 = 0 →
  a 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_a6_value_l2270_227059


namespace NUMINAMATH_CALUDE_lion_death_rate_l2270_227097

/-- Calculates the death rate of lions given initial population, birth rate, and final population after a year. -/
theorem lion_death_rate (initial_population : ℕ) (birth_rate : ℕ) (final_population : ℕ) : 
  initial_population = 100 →
  birth_rate = 5 →
  final_population = 148 →
  ∃ (death_rate : ℕ), 
    initial_population + 12 * birth_rate - 12 * death_rate = final_population ∧
    death_rate = 1 :=
by sorry

end NUMINAMATH_CALUDE_lion_death_rate_l2270_227097


namespace NUMINAMATH_CALUDE_complex_multiplication_l2270_227027

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + 2*i) = -2 + i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2270_227027


namespace NUMINAMATH_CALUDE_consecutive_sum_problem_l2270_227002

theorem consecutive_sum_problem 
  (p q r s t u v : ℕ+) 
  (h1 : p + q + r = 35)
  (h2 : q + r + s = 35)
  (h3 : r + s + t = 35)
  (h4 : s + t + u = 35)
  (h5 : t + u + v = 35)
  (h6 : q + u = 15) :
  p + q + r + s + t + u + v = 90 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_problem_l2270_227002


namespace NUMINAMATH_CALUDE_monomial_count_in_expansion_l2270_227088

theorem monomial_count_in_expansion : 
  let n : ℕ := 2020
  let expression := (fun (x y z : ℝ) => (x + y + z)^n + (x - y - z)^n)
  (∃ (count : ℕ), count = 1022121 ∧ 
    count = (Finset.range (n / 2 + 1)).sum (fun i => 2 * i + 1)) := by
  sorry

end NUMINAMATH_CALUDE_monomial_count_in_expansion_l2270_227088


namespace NUMINAMATH_CALUDE_max_divisors_in_range_20_l2270_227099

def divisor_count (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def max_divisor_count (upper_bound : ℕ) : ℕ :=
  Finset.sup (Finset.range upper_bound.succ) divisor_count

theorem max_divisors_in_range_20 :
  max_divisor_count 20 = 6 ∧
  {12, 18, 20} = {n : ℕ | n ≤ 20 ∧ divisor_count n = max_divisor_count 20} :=
by sorry

end NUMINAMATH_CALUDE_max_divisors_in_range_20_l2270_227099


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l2270_227035

theorem quadratic_factorization_sum (d e f : ℤ) : 
  (∀ x, x^2 + 9*x + 20 = (x + d) * (x + e)) →
  (∀ x, x^2 - x - 56 = (x + e) * (x - f)) →
  d + e + f = 19 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l2270_227035


namespace NUMINAMATH_CALUDE_max_min_values_l2270_227072

def constraint (x y : ℝ) : Prop :=
  Real.sqrt (x - 1) + Real.sqrt (y - 4) = 2

def objective (x y : ℝ) : ℝ :=
  2 * x + y

theorem max_min_values :
  (∃ x y : ℝ, constraint x y ∧ objective x y = 14 ∧ x = 5 ∧ y = 4) ∧
  (∃ x y : ℝ, constraint x y ∧ objective x y = 26/3 ∧ x = 13/9 ∧ y = 52/9) ∧
  (∀ x y : ℝ, constraint x y → objective x y ≤ 14 ∧ objective x y ≥ 26/3) :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_l2270_227072


namespace NUMINAMATH_CALUDE_adams_shelves_capacity_l2270_227015

/-- The number of action figures that can be held on Adam's shelves -/
def total_action_figures (figures_per_shelf : ℕ) (num_shelves : ℕ) : ℕ :=
  figures_per_shelf * num_shelves

/-- Theorem stating that the total number of action figures on Adam's shelves is 44 -/
theorem adams_shelves_capacity :
  total_action_figures 11 4 = 44 := by
  sorry

end NUMINAMATH_CALUDE_adams_shelves_capacity_l2270_227015


namespace NUMINAMATH_CALUDE_ab_length_approx_l2270_227095

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

end NUMINAMATH_CALUDE_ab_length_approx_l2270_227095


namespace NUMINAMATH_CALUDE_smith_children_age_problem_l2270_227086

theorem smith_children_age_problem :
  ∀ (age1 age2 age3 : ℕ),
  age1 = 6 →
  age2 = 8 →
  (age1 + age2 + age3) / 3 = 9 →
  age3 = 13 := by
sorry

end NUMINAMATH_CALUDE_smith_children_age_problem_l2270_227086


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2270_227053

theorem polynomial_factorization (x : ℝ) : x - x^3 = x * (1 - x) * (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2270_227053


namespace NUMINAMATH_CALUDE_number_of_gardens_l2270_227060

theorem number_of_gardens (pots_per_garden : ℕ) (flowers_per_pot : ℕ) (total_flowers : ℕ) :
  pots_per_garden = 544 →
  flowers_per_pot = 32 →
  total_flowers = 174080 →
  total_flowers / (pots_per_garden * flowers_per_pot) = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_of_gardens_l2270_227060


namespace NUMINAMATH_CALUDE_cards_after_home_count_l2270_227073

/-- The number of get well cards Mariela received in the hospital -/
def cards_in_hospital : ℕ := 403

/-- The total number of get well cards Mariela received -/
def total_cards : ℕ := 690

/-- The number of get well cards Mariela received after coming home -/
def cards_after_home : ℕ := total_cards - cards_in_hospital

theorem cards_after_home_count : cards_after_home = 287 := by
  sorry

end NUMINAMATH_CALUDE_cards_after_home_count_l2270_227073


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2270_227075

def solution_set (a : ℝ) : Set ℝ := {x | (a * x - 1) / x > 2 * a ∧ x ≠ 0}

theorem inequality_solution_range (a : ℝ) :
  (2 ∉ solution_set a) ↔ (a ≥ -1/2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2270_227075


namespace NUMINAMATH_CALUDE_x_value_for_given_z_and_w_l2270_227094

/-- Given that x is directly proportional to y³, and y is directly proportional to √z and w,
    prove that x = 540√3 when z = 36 and w = 2, given that x = 5 when z = 8 and w = 1. -/
theorem x_value_for_given_z_and_w (x y z w c k : ℝ) 
    (h1 : ∃ k, ∀ y, x = k * y^3)
    (h2 : ∃ c, y = c * Real.sqrt z * w)
    (h3 : x = 5 ∧ z = 8 ∧ w = 1) :
    z = 36 ∧ w = 2 → x = 540 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_x_value_for_given_z_and_w_l2270_227094


namespace NUMINAMATH_CALUDE_parallel_sufficient_not_necessary_l2270_227077

-- Define the plane α
variable (α : Set Point)

-- Define lines l and m
variable (l m : Line)

-- Define the property of being outside a plane
def OutsidePlane (line : Line) (plane : Set Point) : Prop := sorry

-- Define parallel relation between a line and a plane
def ParallelToPlane (line : Line) (plane : Set Point) : Prop := sorry

-- Define parallel relation between two lines
def ParallelLines (line1 line2 : Line) : Prop := sorry

-- Theorem statement
theorem parallel_sufficient_not_necessary
  (h1 : OutsidePlane l α)
  (h2 : OutsidePlane m α)
  (h3 : l ≠ m)
  (h4 : ParallelToPlane m α) :
  (ParallelLines l m → ParallelToPlane l α) ∧
  ¬(ParallelToPlane l α → ParallelLines l m) := by
  sorry

end NUMINAMATH_CALUDE_parallel_sufficient_not_necessary_l2270_227077


namespace NUMINAMATH_CALUDE_sin_cos_derivative_l2270_227034

theorem sin_cos_derivative (x : ℝ) : 
  deriv (λ x => Real.sin x * Real.cos x) x = Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_derivative_l2270_227034


namespace NUMINAMATH_CALUDE_license_plate_combinations_l2270_227096

def consonants : ℕ := 20
def vowels : ℕ := 6
def digits : ℕ := 10

theorem license_plate_combinations : 
  consonants^2 * vowels^2 * digits = 144000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l2270_227096


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2270_227021

theorem quadratic_equation_roots (a m : ℤ) : 
  (∃ x : ℤ, (a - 1) * x^2 + a * x + 1 = 0 ∧ (m^2 + m) * x^2 + 3 * m * x - 3 = 0) →
  (a = -2 ∧ (m = -1 ∨ m = 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2270_227021


namespace NUMINAMATH_CALUDE_complex_modulus_product_l2270_227008

theorem complex_modulus_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l2270_227008


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l2270_227089

theorem max_product_sum_2000 :
  (∀ a b : ℤ, a + b = 2000 → a * b ≤ 1000000) ∧
  (∃ a b : ℤ, a + b = 2000 ∧ a * b = 1000000) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l2270_227089


namespace NUMINAMATH_CALUDE_polynomial_g_forms_l2270_227016

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the property that g must satisfy
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9*x^2 - 6*x + 1

-- State the theorem
theorem polynomial_g_forms :
  ∀ g : ℝ → ℝ, g_property g →
  (∀ x, g x = 3*x - 1) ∨ (∀ x, g x = -3*x + 1) :=
sorry

end NUMINAMATH_CALUDE_polynomial_g_forms_l2270_227016


namespace NUMINAMATH_CALUDE_bug_path_tiles_l2270_227033

/-- Represents a rectangular floor with integer dimensions -/
structure RectangularFloor where
  width : ℕ
  length : ℕ

/-- Calculates the number of tiles a bug visits when walking diagonally across a rectangular floor -/
def tilesVisited (floor : RectangularFloor) : ℕ :=
  floor.width + floor.length - Nat.gcd floor.width floor.length

theorem bug_path_tiles (floor : RectangularFloor) 
  (h_width : floor.width = 9) 
  (h_length : floor.length = 13) : 
  tilesVisited floor = 21 := by
sorry

#eval tilesVisited ⟨9, 13⟩

end NUMINAMATH_CALUDE_bug_path_tiles_l2270_227033


namespace NUMINAMATH_CALUDE_ellipse_C_equation_constant_ratio_l2270_227049

/-- An ellipse C with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line with slope k passing through point (x₀, y₀) -/
structure Line where
  k : ℝ
  x₀ : ℝ
  y₀ : ℝ

/-- Definition of the ellipse C -/
def ellipse_C : Ellipse := {
  a := sorry,
  b := sorry,
  h_pos := sorry
}

/-- The ellipse C passes through (0, -1) -/
axiom passes_through : 0^2 / ellipse_C.a^2 + (-1)^2 / ellipse_C.b^2 = 1

/-- The eccentricity of C is √2/2 -/
axiom eccentricity : Real.sqrt ((ellipse_C.a^2 - ellipse_C.b^2) / ellipse_C.a^2) = Real.sqrt 2 / 2

/-- The equation of ellipse C -/
def ellipse_equation (p : Point) : Prop :=
  p.x^2 / 2 + p.y^2 = 1

/-- Point F -/
def F : Point := { x := 1, y := 0 }

/-- Theorem: The equation of ellipse C is x²/2 + y² = 1 -/
theorem ellipse_C_equation :
  ∀ p : Point, p.x^2 / ellipse_C.a^2 + p.y^2 / ellipse_C.b^2 = 1 ↔ ellipse_equation p :=
sorry

/-- Theorem: The ratio |MN|/|PF| is constant for any non-zero slope k -/
theorem constant_ratio (k : ℝ) (hk : k ≠ 0) :
  ∃ M N P : Point,
    (∃ l : Line, l.k = k ∧ l.x₀ = F.x ∧ l.y₀ = F.y) ∧
    ellipse_equation M ∧
    ellipse_equation N ∧
    (P.y = 0) ∧
    (N.y - M.y) * (P.x - M.x) = (M.x - N.x) * (P.y - M.y) →
    (N.x - M.x)^2 + (N.y - M.y)^2 = 8 * ((P.x - F.x)^2 + (P.y - F.y)^2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_C_equation_constant_ratio_l2270_227049


namespace NUMINAMATH_CALUDE_fraction_value_l2270_227091

theorem fraction_value (a b : ℝ) (h : Real.sqrt (a + 2) + |b - 3| = 0) : a / b = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2270_227091


namespace NUMINAMATH_CALUDE_special_operation_result_l2270_227090

/-- The "※" operation for integers -/
def star (a b : ℤ) : ℤ := a + b - 1

/-- The "#" operation for integers -/
def hash (a b : ℤ) : ℤ := a * b - 1

/-- Theorem stating that 4#[(6※8)※(3#5)] = 103 -/
theorem special_operation_result : hash 4 (star (star 6 8) (hash 3 5)) = 103 := by
  sorry

end NUMINAMATH_CALUDE_special_operation_result_l2270_227090


namespace NUMINAMATH_CALUDE_total_pencils_l2270_227032

theorem total_pencils (num_boxes : ℕ) (pencils_per_box : ℕ) (h1 : num_boxes = 3) (h2 : pencils_per_box = 9) :
  num_boxes * pencils_per_box = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l2270_227032


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2270_227055

theorem expand_and_simplify (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2270_227055


namespace NUMINAMATH_CALUDE_range_of_m_l2270_227062

/-- The set A defined by a quadratic inequality -/
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

/-- The set B defined by a quadratic inequality with parameter m -/
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 ≤ 0}

/-- The theorem stating the range of m given A is a subset of the complement of B -/
theorem range_of_m (m : ℝ) : A ⊆ (Set.univ \ B m) → m < -3 ∨ m > 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2270_227062


namespace NUMINAMATH_CALUDE_jills_sales_goal_l2270_227061

/-- Represents the number of boxes sold to each customer --/
structure CustomerSales where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

/-- Calculates Jill's sales goal based on customer purchases and boxes left to sell --/
def salesGoal (sales : CustomerSales) (boxesLeft : ℕ) : ℕ :=
  sales.first + sales.second + sales.third + sales.fourth + sales.fifth + boxesLeft

/-- Theorem stating Jill's sales goal --/
theorem jills_sales_goal :
  ∀ (sales : CustomerSales) (boxesLeft : ℕ),
    sales.first = 5 →
    sales.second = 4 * sales.first →
    sales.third = sales.second / 2 →
    sales.fourth = 3 * sales.third →
    sales.fifth = 10 →
    boxesLeft = 75 →
    salesGoal sales boxesLeft = 150 := by
  sorry

end NUMINAMATH_CALUDE_jills_sales_goal_l2270_227061


namespace NUMINAMATH_CALUDE_first_class_students_l2270_227028

/-- Given two classes of students, this theorem proves the number of students in the first class. -/
theorem first_class_students (avg_first : ℝ) (students_second : ℕ) (avg_second : ℝ) (avg_all : ℝ) :
  avg_first = 40 →
  students_second = 50 →
  avg_second = 60 →
  avg_all = 52.5 →
  ∃ students_first : ℕ, 
    students_first = 30 ∧
    (avg_first * students_first + avg_second * students_second) / (students_first + students_second) = avg_all :=
by
  sorry

end NUMINAMATH_CALUDE_first_class_students_l2270_227028


namespace NUMINAMATH_CALUDE_candy_distribution_impossibility_l2270_227037

theorem candy_distribution_impossibility :
  ¬ ∃ (n : ℕ), 7 * n = 3 * 20 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_impossibility_l2270_227037


namespace NUMINAMATH_CALUDE_ship_distance_difference_l2270_227050

/-- The difference in distance traveled between two ships sailing in opposite directions -/
theorem ship_distance_difference (a : ℝ) : 
  let ship_speed := 50
  let time := 2
  let distance_with_current := time * (ship_speed + a)
  let distance_against_current := time * (ship_speed - a)
  distance_with_current - distance_against_current = 4 * a := by
  sorry

end NUMINAMATH_CALUDE_ship_distance_difference_l2270_227050


namespace NUMINAMATH_CALUDE_digit_product_existence_l2270_227078

/-- A single-digit integer is a natural number from 1 to 9. -/
def SingleDigit : Type := { n : ℕ // 1 ≤ n ∧ n ≤ 9 }

/-- The product of a list of single-digit integers -/
def product_of_digits (digits : List SingleDigit) : ℕ :=
  digits.foldl (fun acc d => acc * d.val) 1

/-- Theorem stating the existence or non-existence of integers with specific digit products -/
theorem digit_product_existence :
  (¬ ∃ (digits : List SingleDigit), product_of_digits digits = 1980) ∧
  (¬ ∃ (digits : List SingleDigit), product_of_digits digits = 1990) ∧
  (∃ (digits : List SingleDigit), product_of_digits digits = 2000) :=
sorry

end NUMINAMATH_CALUDE_digit_product_existence_l2270_227078


namespace NUMINAMATH_CALUDE_lemon_fraction_is_one_seventh_l2270_227076

/-- Represents the contents of a cup --/
structure CupContents where
  tea : ℚ
  honey : ℚ
  lemon : ℚ

/-- The initial setup of the cups --/
def initial_setup : CupContents × CupContents :=
  ({tea := 6, honey := 0, lemon := 0}, {tea := 0, honey := 6, lemon := 3})

/-- Pours half of the tea from the first cup to the second --/
def pour_half_tea (cups : CupContents × CupContents) : CupContents × CupContents :=
  let (cup1, cup2) := cups
  let tea_to_pour := cup1.tea / 2
  ({tea := cup1.tea - tea_to_pour, honey := cup1.honey, lemon := cup1.lemon},
   {tea := cup2.tea + tea_to_pour, honey := cup2.honey, lemon := cup2.lemon})

/-- Pours one-third of the second cup's contents back to the first cup --/
def pour_third_back (cups : CupContents × CupContents) : CupContents × CupContents :=
  let (cup1, cup2) := cups
  let total_cup2 := cup2.tea + cup2.honey + cup2.lemon
  let fraction := 1 / 3
  let tea_to_pour := cup2.tea * fraction
  let honey_to_pour := cup2.honey * fraction
  let lemon_to_pour := cup2.lemon * fraction
  ({tea := cup1.tea + tea_to_pour, honey := cup1.honey + honey_to_pour, lemon := cup1.lemon + lemon_to_pour},
   {tea := cup2.tea - tea_to_pour, honey := cup2.honey - honey_to_pour, lemon := cup2.lemon - lemon_to_pour})

/-- Calculates the fraction of lemon juice in a cup --/
def lemon_fraction (cup : CupContents) : ℚ :=
  cup.lemon / (cup.tea + cup.honey + cup.lemon)

theorem lemon_fraction_is_one_seventh :
  let final_state := pour_third_back (pour_half_tea initial_setup)
  lemon_fraction final_state.fst = 1 / 7 := by
  sorry


end NUMINAMATH_CALUDE_lemon_fraction_is_one_seventh_l2270_227076


namespace NUMINAMATH_CALUDE_andre_flowers_l2270_227036

/-- Given Rosa's initial and final number of flowers, prove that the number of flowers
    Andre gave to Rosa is the difference between the final and initial counts. -/
theorem andre_flowers (initial final andre : ℕ) 
  (h1 : initial = 67)
  (h2 : final = 90)
  (h3 : final = initial + andre) : 
  andre = final - initial := by
  sorry

end NUMINAMATH_CALUDE_andre_flowers_l2270_227036


namespace NUMINAMATH_CALUDE_scientific_notation_of_1040000000_l2270_227043

theorem scientific_notation_of_1040000000 :
  (1040000000 : ℝ) = 1.04 * (10 ^ 9) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1040000000_l2270_227043


namespace NUMINAMATH_CALUDE_population_increase_rate_example_l2270_227063

/-- Given a town's initial population and population after one year,
    calculate the population increase rate as a percentage. -/
def populationIncreaseRate (initialPopulation finalPopulation : ℕ) : ℚ :=
  (finalPopulation - initialPopulation : ℚ) / initialPopulation * 100

/-- Theorem stating that for a town with an initial population of 200
    and a population of 220 after 1 year, the population increase rate is 10%. -/
theorem population_increase_rate_example :
  populationIncreaseRate 200 220 = 10 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_rate_example_l2270_227063


namespace NUMINAMATH_CALUDE_sequence_problem_l2270_227084

/-- Given a sequence a_n and a geometric sequence b_n where
    a_1 = 2, b_n = a_{n+1} / a_n for all n, and b_10 * b_11 = 2,
    prove that a_21 = 2^11 -/
theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) :
  a 1 = 2 →
  (∀ n, b n = a (n + 1) / a n) →
  (∃ r, ∀ n, b (n + 1) = r * b n) →
  b 10 * b 11 = 2 →
  a 21 = 2^11 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l2270_227084


namespace NUMINAMATH_CALUDE_nancy_deleted_files_l2270_227040

theorem nancy_deleted_files (initial_files : ℕ) (folders : ℕ) (files_per_folder : ℕ) : 
  initial_files = 43 →
  folders = 2 →
  files_per_folder = 6 →
  initial_files - (folders * files_per_folder) = 31 := by
sorry

end NUMINAMATH_CALUDE_nancy_deleted_files_l2270_227040


namespace NUMINAMATH_CALUDE_product_of_numbers_l2270_227093

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 16) (h2 : x^2 + y^2 = 200) : x * y = 28 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2270_227093


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2270_227019

theorem right_triangle_hypotenuse (shorter_leg longer_leg hypotenuse : ℝ) : 
  shorter_leg > 0 →
  longer_leg = 3 * shorter_leg - 2 →
  (1 / 2) * shorter_leg * longer_leg = 72 →
  hypotenuse ^ 2 = shorter_leg ^ 2 + longer_leg ^ 2 →
  hypotenuse = Real.sqrt 292 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2270_227019


namespace NUMINAMATH_CALUDE_stop_time_is_sixty_l2270_227013

/-- Calculates the stop time in minutes given the total journey time and driving time in hours. -/
def stop_time_minutes (total_journey_hours driving_hours : ℕ) : ℕ :=
  (total_journey_hours - driving_hours) * 60

/-- Theorem stating that the stop time is 60 minutes given the specific journey details. -/
theorem stop_time_is_sixty : stop_time_minutes 13 12 = 60 := by
  sorry

end NUMINAMATH_CALUDE_stop_time_is_sixty_l2270_227013


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l2270_227048

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A * B * C = 2023 →
  A + B + C ≤ 297 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l2270_227048


namespace NUMINAMATH_CALUDE_cornelia_age_l2270_227004

theorem cornelia_age (kilee_age : ℕ) (future_years : ℕ) :
  kilee_age = 20 →
  future_years = 10 →
  ∃ (cornelia_age : ℕ),
    cornelia_age + future_years = 3 * (kilee_age + future_years) ∧
    cornelia_age = 80 :=
by sorry

end NUMINAMATH_CALUDE_cornelia_age_l2270_227004


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2270_227082

/-- Given a geometric sequence {a_n} with sum S_n = 3^n + t, prove a_2 = 6 and t = -1 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) -- geometric sequence condition
  (h_sum : ∀ n : ℕ, S n = 3^n + t) -- sum condition
  : a 2 = 6 ∧ t = -1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2270_227082


namespace NUMINAMATH_CALUDE_largest_angle_in_ratio_triangle_l2270_227029

/-- Theorem: In a triangle where the angles are in the ratio 3:4:5, the largest angle measures 75°. -/
theorem largest_angle_in_ratio_triangle : ∀ (a b c : ℝ),
  -- The angles are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- The angles are in the ratio 3:4:5
  b = (4/3) * a ∧ c = (5/3) * a →
  -- The sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle (c) measures 75°
  c = 75 := by
sorry


end NUMINAMATH_CALUDE_largest_angle_in_ratio_triangle_l2270_227029


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2270_227044

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) * (2 - x) > 0 ↔ 2 < x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2270_227044


namespace NUMINAMATH_CALUDE_expression_factorization_l2270_227009

theorem expression_factorization (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l2270_227009


namespace NUMINAMATH_CALUDE_complex_division_result_l2270_227042

theorem complex_division_result : (1 + 3*Complex.I) / (1 - Complex.I) = -1 + 2*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l2270_227042


namespace NUMINAMATH_CALUDE_orange_purchase_ratio_l2270_227047

/-- Proves the ratio of weekly orange purchases --/
theorem orange_purchase_ratio 
  (initial_purchase : ℕ) 
  (additional_purchase : ℕ) 
  (total_after_three_weeks : ℕ) 
  (h1 : initial_purchase = 10)
  (h2 : additional_purchase = 5)
  (h3 : total_after_three_weeks = 75) :
  (total_after_three_weeks - (initial_purchase + additional_purchase)) / 2 = 
  2 * (initial_purchase + additional_purchase) :=
by
  sorry

#check orange_purchase_ratio

end NUMINAMATH_CALUDE_orange_purchase_ratio_l2270_227047


namespace NUMINAMATH_CALUDE_system_solution_l2270_227041

theorem system_solution : 
  ∀ x y z : ℝ, 
  (x^2 + y^2 + 25*z^2 = 6*x*z + 8*y*z) ∧ 
  (3*x^2 + 2*y^2 + z^2 = 240) → 
  ((x = 6 ∧ y = 8 ∧ z = 2) ∨ (x = -6 ∧ y = -8 ∧ z = -2)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2270_227041


namespace NUMINAMATH_CALUDE_monomial_exponents_l2270_227001

/-- Two monomials are of the same type if they have the same exponents for each variable -/
def SameType (m1 m2 : ℕ → ℕ) : Prop :=
  ∀ i, m1 i = m2 i

theorem monomial_exponents (a b : ℕ) :
  SameType (fun i => if i = 0 then a + 1 else if i = 1 then 3 else 0)
           (fun i => if i = 0 then 3 else if i = 1 then b else 0) →
  a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_monomial_exponents_l2270_227001


namespace NUMINAMATH_CALUDE_barbara_paper_weight_l2270_227083

/-- Calculates the total weight of paper removed from a chest of drawers -/
def total_weight_of_paper (
  colored_bundles : ℕ)
  (white_bunches : ℕ)
  (scrap_heaps : ℕ)
  (sheets_per_bunch : ℕ)
  (sheets_per_bundle : ℕ)
  (sheets_per_heap : ℕ)
  (colored_sheet_weight : ℚ)
  (white_sheet_weight : ℚ)
  (scrap_sheet_weight : ℚ) : ℚ :=
  let colored_sheets := colored_bundles * sheets_per_bundle
  let white_sheets := white_bunches * sheets_per_bunch
  let scrap_sheets := scrap_heaps * sheets_per_heap
  let colored_weight := colored_sheets * colored_sheet_weight
  let white_weight := white_sheets * white_sheet_weight
  let scrap_weight := scrap_sheets * scrap_sheet_weight
  colored_weight + white_weight + scrap_weight

theorem barbara_paper_weight :
  total_weight_of_paper 3 2 5 4 2 20 (3/100) (1/20) (1/25) = 458/100 := by
  sorry

end NUMINAMATH_CALUDE_barbara_paper_weight_l2270_227083


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l2270_227023

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_sqrt : ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1)
  (h_sum : a 6 = a 5 + 2 * a 4) :
  ∃ m n : ℕ, (1 : ℝ) / m + 4 / n ≥ 3 / 2 ∧
    (∀ k l : ℕ, (1 : ℝ) / k + 4 / l ≥ 3 / 2) :=
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l2270_227023


namespace NUMINAMATH_CALUDE_five_houses_built_l2270_227068

/-- The number of houses that can be built given the specified conditions -/
def number_of_houses (builders_per_floor : ℕ) (days_per_floor : ℕ) (daily_wage : ℕ) 
  (total_cost : ℕ) (builders : ℕ) (floors_per_house : ℕ) : ℕ :=
  let daily_cost := builders * daily_wage
  let total_days := total_cost / daily_cost
  let days_per_floor_with_builders := days_per_floor * builders_per_floor / builders
  let total_floors := total_days / days_per_floor_with_builders
  total_floors / floors_per_house

/-- Theorem stating that 5 houses can be built under the given conditions -/
theorem five_houses_built :
  number_of_houses 3 30 100 270000 6 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_houses_built_l2270_227068


namespace NUMINAMATH_CALUDE_knocks_to_knicks_conversion_l2270_227022

/-- Conversion rate between knicks and knacks -/
def knicks_to_knacks : ℚ := 3 / 5

/-- Conversion rate between knacks and knocks -/
def knacks_to_knocks : ℚ := 7 / 4

/-- The number of knocks we want to convert -/
def target_knocks : ℕ := 28

/-- Theorem stating the equivalence between knocks and knicks -/
theorem knocks_to_knicks_conversion :
  (target_knocks : ℚ) * knacks_to_knocks⁻¹ * knicks_to_knacks⁻¹ = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_knocks_to_knicks_conversion_l2270_227022


namespace NUMINAMATH_CALUDE_book_reading_time_l2270_227011

/-- Calculates the remaining reading time for a book -/
def remaining_reading_time (total_pages : ℕ) (pages_per_hour : ℕ) (monday_hours : ℕ) (tuesday_hours : ℚ) : ℚ :=
  let pages_read := (monday_hours : ℚ) * pages_per_hour + tuesday_hours * pages_per_hour
  let pages_left := (total_pages : ℚ) - pages_read
  pages_left / pages_per_hour

theorem book_reading_time :
  remaining_reading_time 248 16 3 (13/2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_time_l2270_227011


namespace NUMINAMATH_CALUDE_dairy_farm_husk_consumption_l2270_227051

theorem dairy_farm_husk_consumption 
  (num_cows : ℕ) 
  (num_bags : ℕ) 
  (num_days : ℕ) 
  (h1 : num_cows = 30) 
  (h2 : num_bags = 30) 
  (h3 : num_days = 30) : 
  (1 : ℕ) * num_days = 30 := by
  sorry

end NUMINAMATH_CALUDE_dairy_farm_husk_consumption_l2270_227051


namespace NUMINAMATH_CALUDE_parallel_vectors_m_equals_6_l2270_227081

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_m_equals_6 :
  ∀ m : ℝ, are_parallel (m, 4) (3, 2) → m = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_equals_6_l2270_227081


namespace NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l2270_227030

/-- The amount of money John spent out of pocket when buying a computer and accessories
    after selling his PlayStation --/
theorem johns_out_of_pocket_expense (computer_cost accessories_cost playstation_value : ℚ)
  (h1 : computer_cost = 700)
  (h2 : accessories_cost = 200)
  (h3 : playstation_value = 400)
  (discount_rate : ℚ)
  (h4 : discount_rate = 1/5) : -- 20% expressed as a fraction
  computer_cost + accessories_cost - playstation_value * (1 - discount_rate) = 580 := by
  sorry

end NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l2270_227030


namespace NUMINAMATH_CALUDE_student_count_l2270_227057

theorem student_count (N : ℕ) 
  (h1 : N / 5 + N / 4 + N / 2 + 5 = N) : N = 100 := by
  sorry

#check student_count

end NUMINAMATH_CALUDE_student_count_l2270_227057


namespace NUMINAMATH_CALUDE_equilateral_triangle_filling_l2270_227018

theorem equilateral_triangle_filling :
  let large_side : ℝ := 15
  let small_side : ℝ := 3
  let area (side : ℝ) := (Real.sqrt 3 / 4) * side^2
  let large_area := area large_side
  let small_area := area small_side
  let num_small_triangles := large_area / small_area
  num_small_triangles = 25 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_filling_l2270_227018


namespace NUMINAMATH_CALUDE_solution_set_inequality_1_solution_set_inequality_2_l2270_227066

-- Problem 1
theorem solution_set_inequality_1 :
  {x : ℝ | (2 - x) / (x + 4) ≤ 0} = {x : ℝ | x ≥ 2 ∨ x < -4} := by sorry

-- Problem 2
theorem solution_set_inequality_2 (a : ℝ) :
  {x : ℝ | x^2 - 3*a*x + 2*a^2 ≥ 0} = 
    if a > 0 then
      {x : ℝ | x ≥ 2*a ∨ x ≤ a}
    else if a < 0 then
      {x : ℝ | x ≥ a ∨ x ≤ 2*a}
    else
      {x : ℝ | True} := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_1_solution_set_inequality_2_l2270_227066


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l2270_227014

/-- Two circles are tangent if they touch at exactly one point. -/
def are_tangent (c1 c2 : Circle) : Prop := sorry

/-- The distance between the centers of two circles. -/
def center_distance (c1 c2 : Circle) : ℝ := sorry

/-- The radius of a circle. -/
def radius (c : Circle) : ℝ := sorry

theorem tangent_circles_radius (c1 c2 : Circle) :
  are_tangent c1 c2 →
  center_distance c1 c2 = 7 →
  radius c1 = 5 →
  radius c2 = 2 ∨ radius c2 = 12 := by sorry

end NUMINAMATH_CALUDE_tangent_circles_radius_l2270_227014


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2270_227003

/-- The common ratio of the infinite geometric series 8/10 - 12/25 + 36/125 - ... is -3/5 -/
theorem geometric_series_common_ratio :
  let a₁ : ℚ := 8 / 10
  let a₂ : ℚ := -12 / 25
  let a₃ : ℚ := 36 / 125
  ∃ r : ℚ, r = a₂ / a₁ ∧ r = a₃ / a₂ ∧ r = -3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2270_227003


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l2270_227058

/-- Represents the distribution of students across four years -/
structure StudentDistribution :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)
  (fourth : ℕ)

/-- Calculates the total number of students -/
def total_students (d : StudentDistribution) : ℕ :=
  d.first + d.second + d.third + d.fourth

/-- Calculates the number of students in a stratified sample for a given year -/
def stratified_sample_size (total : ℕ) (year_count : ℕ) (sample_size : ℕ) : ℕ :=
  (year_count * sample_size) / total

theorem stratified_sample_theorem (d : StudentDistribution) 
  (h1 : d.first = 400)
  (h2 : d.second = 300)
  (h3 : d.third = 200)
  (h4 : d.fourth = 100)
  (h5 : total_students d = 1000)
  (sample_size : ℕ)
  (h6 : sample_size = 200) :
  stratified_sample_size (total_students d) d.third sample_size = 40 := by
  sorry

#check stratified_sample_theorem

end NUMINAMATH_CALUDE_stratified_sample_theorem_l2270_227058


namespace NUMINAMATH_CALUDE_negation_of_universal_quantifier_l2270_227085

theorem negation_of_universal_quantifier :
  (¬ ∀ x : ℝ, x^2 + 1 < 0) ↔ (∃ x : ℝ, x^2 + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantifier_l2270_227085


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l2270_227092

/-- Given the cost price, marked price, and profit percentage, calculate the discount percentage. -/
theorem discount_percentage_calculation (cost_price marked_price : ℝ) (profit_percentage : ℝ) :
  cost_price = 47.50 →
  marked_price = 64.54 →
  profit_percentage = 25 →
  ∃ (discount_percentage : ℝ), 
    (abs (discount_percentage - 8) < 0.1) ∧ 
    (cost_price * (1 + profit_percentage / 100) = marked_price * (1 - discount_percentage / 100)) := by
  sorry


end NUMINAMATH_CALUDE_discount_percentage_calculation_l2270_227092


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_eight_l2270_227052

theorem sqrt_sum_equals_eight : 
  Real.sqrt (24 - 8 * Real.sqrt 3) + Real.sqrt (24 + 8 * Real.sqrt 3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_eight_l2270_227052


namespace NUMINAMATH_CALUDE_triangle_area_l2270_227079

/-- Given a triangle with perimeter 36 cm and inradius 2.5 cm, its area is 45 cm² -/
theorem triangle_area (p : ℝ) (r : ℝ) (A : ℝ) 
    (h1 : p = 36) -- perimeter is 36 cm
    (h2 : r = 2.5) -- inradius is 2.5 cm
    (h3 : A = r * (p / 2)) -- area formula: A = r * s, where s is semiperimeter (p/2)
    : A = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2270_227079


namespace NUMINAMATH_CALUDE_prudence_total_sleep_l2270_227006

/-- Represents Prudence's sleep schedule --/
structure SleepSchedule where
  weekdaySleep : ℕ  -- Hours of sleep on weekdays (Sun-Thu)
  weekendSleep : ℕ  -- Hours of sleep on weekends (Fri-Sat)
  napDuration : ℕ   -- Duration of nap in hours
  napDays : ℕ       -- Number of days with naps
  weekdayNights : ℕ -- Number of weekday nights
  weekendNights : ℕ -- Number of weekend nights

/-- Calculates total sleep in 4 weeks given a sleep schedule --/
def totalSleepInFourWeeks (schedule : SleepSchedule) : ℕ :=
  4 * (schedule.weekdaySleep * schedule.weekdayNights +
       schedule.weekendSleep * schedule.weekendNights +
       schedule.napDuration * schedule.napDays)

/-- Prudence's actual sleep schedule --/
def prudenceSchedule : SleepSchedule :=
  { weekdaySleep := 6
  , weekendSleep := 9
  , napDuration := 1
  , napDays := 2
  , weekdayNights := 5
  , weekendNights := 2 }

/-- Theorem stating that Prudence's total sleep in 4 weeks is 200 hours --/
theorem prudence_total_sleep :
  totalSleepInFourWeeks prudenceSchedule = 200 := by
  sorry


end NUMINAMATH_CALUDE_prudence_total_sleep_l2270_227006


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l2270_227070

theorem quadratic_equation_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ 
   x^2 + 10*x + k = 0 ∧ 
   y^2 + 10*y + k = 0 ∧ 
   x / y = 3 / 2) → 
  k = 24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l2270_227070


namespace NUMINAMATH_CALUDE_expression_evaluation_l2270_227064

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1/2
  3 * x^2 * y - (5 * x * y^2 + 2 * (x^2 * y - 1/2) + x^2 * y) + 6 * x * y^2 = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2270_227064


namespace NUMINAMATH_CALUDE_cuboid_height_calculation_l2270_227087

/-- Represents the dimensions and volume of a cuboid -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ
  volume : ℝ

/-- Theorem: For a cuboid with volume 315 cm³, width 9 cm, and length 7 cm, the height is 5 cm -/
theorem cuboid_height_calculation (c : Cuboid) 
  (h_volume : c.volume = 315)
  (h_width : c.width = 9)
  (h_length : c.length = 7)
  : c.height = 5 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_calculation_l2270_227087


namespace NUMINAMATH_CALUDE_roots_of_equation_l2270_227054

theorem roots_of_equation (x y : ℝ) (h1 : x + y = 10) (h2 : |x - y| = 14) :
  x^2 - 10*x - 24 = 0 ∧ y^2 - 10*y - 24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2270_227054


namespace NUMINAMATH_CALUDE_least_possible_c_l2270_227046

theorem least_possible_c (a b c : ℕ+) : 
  (a + b + c : ℚ) / 3 = 20 →
  a ≤ b →
  b ≤ c →
  b = a + 13 →
  c ≥ 45 ∧ ∃ (a₀ b₀ c₀ : ℕ+), 
    (a₀ + b₀ + c₀ : ℚ) / 3 = 20 ∧
    a₀ ≤ b₀ ∧
    b₀ ≤ c₀ ∧
    b₀ = a₀ + 13 ∧
    c₀ = 45 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_c_l2270_227046


namespace NUMINAMATH_CALUDE_prob_same_length_in_hexagon_l2270_227098

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℕ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of elements in T -/
def total_elements : ℕ := num_sides + num_diagonals

/-- The probability of selecting two elements of the same length from T -/
def prob_same_length : ℚ := 17 / 35

theorem prob_same_length_in_hexagon :
  (num_sides * (num_sides - 1) + num_diagonals * (num_diagonals - 1)) / (total_elements * (total_elements - 1)) = prob_same_length := by
  sorry

end NUMINAMATH_CALUDE_prob_same_length_in_hexagon_l2270_227098


namespace NUMINAMATH_CALUDE_spring_length_correct_l2270_227045

/-- The function describing the total length of a spring -/
def spring_length (x : ℝ) : ℝ := 12 + 3 * x

/-- Theorem stating the correctness of the spring length function -/
theorem spring_length_correct (x : ℝ) : 
  (spring_length 0 = 12) ∧ 
  (∃ k : ℝ, ∀ x : ℝ, spring_length x - 12 = k * x) ∧
  (spring_length 1 - 12 = 3) →
  spring_length x = 12 + 3 * x :=
by sorry

end NUMINAMATH_CALUDE_spring_length_correct_l2270_227045
