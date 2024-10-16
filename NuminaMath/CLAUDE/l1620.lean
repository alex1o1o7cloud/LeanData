import Mathlib

namespace NUMINAMATH_CALUDE_calculation_proof_l1620_162040

theorem calculation_proof : 1.23 * 67 + 8.2 * 12.3 - 90 * 0.123 = 172.2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1620_162040


namespace NUMINAMATH_CALUDE_fraction_of_number_l1620_162069

theorem fraction_of_number : (7 : ℚ) / 25 * 89473 = 25052.44 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_number_l1620_162069


namespace NUMINAMATH_CALUDE_mehki_age_proof_l1620_162045

/-- Proves that Mehki's age is 16 years old given the specified conditions -/
theorem mehki_age_proof (zrinka_age jordyn_age mehki_age : ℕ) : 
  zrinka_age = 6 →
  jordyn_age = zrinka_age - 4 →
  mehki_age = 2 * (jordyn_age + zrinka_age) →
  mehki_age = 16 := by
sorry

end NUMINAMATH_CALUDE_mehki_age_proof_l1620_162045


namespace NUMINAMATH_CALUDE_fabian_marbles_comparison_l1620_162089

theorem fabian_marbles_comparison (fabian_marbles kyle_marbles miles_marbles : ℕ) : 
  fabian_marbles = 15 →
  fabian_marbles = 3 * kyle_marbles →
  kyle_marbles + miles_marbles = 8 →
  fabian_marbles = 5 * miles_marbles :=
by
  sorry

end NUMINAMATH_CALUDE_fabian_marbles_comparison_l1620_162089


namespace NUMINAMATH_CALUDE_abs_z_equals_one_l1620_162062

theorem abs_z_equals_one (z : ℂ) (h : (1 - 2*I)^2 / z = 4 - 3*I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_equals_one_l1620_162062


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1620_162001

theorem trigonometric_identity (x y : ℝ) :
  3 * Real.cos (x + y) * Real.sin x + Real.sin (x + y) * Real.cos x =
  3 * Real.cos x * Real.cos y * Real.sin x - 3 * Real.sin x * Real.sin y * Real.sin x +
  Real.sin x * Real.cos y * Real.cos x + Real.cos x * Real.sin y * Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1620_162001


namespace NUMINAMATH_CALUDE_solution_set_sqrt3_sin_eq_cos_l1620_162095

theorem solution_set_sqrt3_sin_eq_cos :
  {x : ℝ | Real.sqrt 3 * Real.sin x = Real.cos x} =
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.pi / 6} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_sqrt3_sin_eq_cos_l1620_162095


namespace NUMINAMATH_CALUDE_frenchBulldogRatioIsTwo_l1620_162013

/-- The ratio of French Bulldogs Peter wants to Sam's -/
def frenchBulldogRatio (samGermanShepherds samFrenchBulldogs peterTotalDogs : ℕ) : ℚ :=
  let peterGermanShepherds := 3 * samGermanShepherds
  let peterFrenchBulldogs := peterTotalDogs - peterGermanShepherds
  (peterFrenchBulldogs : ℚ) / samFrenchBulldogs

/-- The ratio of French Bulldogs Peter wants to Sam's is 2:1 -/
theorem frenchBulldogRatioIsTwo :
  frenchBulldogRatio 3 4 17 = 2 := by
  sorry

end NUMINAMATH_CALUDE_frenchBulldogRatioIsTwo_l1620_162013


namespace NUMINAMATH_CALUDE_divisor_problem_l1620_162054

theorem divisor_problem (n : ℕ) (h : n = 1101) : 
  ∃ (d : ℕ), d > 1 ∧ (n + 3) % d = 0 ∧ d = 8 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l1620_162054


namespace NUMINAMATH_CALUDE_stratified_sampling_probability_l1620_162036

/-- The probability of selecting an individual in a sampling method -/
def sampling_probability (m : ℕ) (sample_size : ℕ) : ℚ := 1 / sample_size

theorem stratified_sampling_probability (m : ℕ) (h : m ≥ 3) :
  sampling_probability m 3 = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_probability_l1620_162036


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_negation_of_sqrt_proposition_l1620_162030

theorem negation_of_universal_proposition (p : ℝ → Prop) :
  (¬ ∀ x ≤ 0, p x) ↔ (∃ x₀ ≤ 0, ¬ p x₀) := by sorry

-- Define the specific proposition
def sqrt_prop (x : ℝ) : Prop := Real.sqrt (x^2) = -x

-- Main theorem
theorem negation_of_sqrt_proposition :
  (¬ ∀ x ≤ 0, sqrt_prop x) ↔ (∃ x₀ ≤ 0, ¬ sqrt_prop x₀) :=
negation_of_universal_proposition sqrt_prop

end NUMINAMATH_CALUDE_negation_of_universal_proposition_negation_of_sqrt_proposition_l1620_162030


namespace NUMINAMATH_CALUDE_prob_diff_colors_is_11_18_l1620_162021

def num_blue : ℕ := 6
def num_yellow : ℕ := 4
def num_red : ℕ := 2
def total_chips : ℕ := num_blue + num_yellow + num_red

def prob_diff_colors : ℚ :=
  (num_blue : ℚ) / total_chips * ((num_yellow + num_red) : ℚ) / total_chips +
  (num_yellow : ℚ) / total_chips * ((num_blue + num_red) : ℚ) / total_chips +
  (num_red : ℚ) / total_chips * ((num_blue + num_yellow) : ℚ) / total_chips

theorem prob_diff_colors_is_11_18 : prob_diff_colors = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_prob_diff_colors_is_11_18_l1620_162021


namespace NUMINAMATH_CALUDE_sine_equality_solution_l1620_162028

theorem sine_equality_solution (n : ℤ) (h1 : -180 ≤ n) (h2 : n ≤ 180) :
  Real.sin (n * π / 180) = Real.sin (750 * π / 180) → n = 30 := by
sorry

end NUMINAMATH_CALUDE_sine_equality_solution_l1620_162028


namespace NUMINAMATH_CALUDE_product_first_two_terms_of_specific_sequence_l1620_162026

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem product_first_two_terms_of_specific_sequence :
  ∃ (a₁ : ℝ),
    arithmetic_sequence a₁ 1 5 = 11 ∧
    arithmetic_sequence a₁ 1 1 * arithmetic_sequence a₁ 1 2 = 56 :=
by sorry

end NUMINAMATH_CALUDE_product_first_two_terms_of_specific_sequence_l1620_162026


namespace NUMINAMATH_CALUDE_warehouse_repacking_l1620_162053

/-- The number of books left over after repacking in the warehouse scenario -/
theorem warehouse_repacking (initial_boxes : Nat) (books_per_initial_box : Nat) 
  (damaged_books : Nat) (books_per_new_box : Nat) 
  (h1 : initial_boxes = 1200)
  (h2 : books_per_initial_box = 35)
  (h3 : damaged_books = 100)
  (h4 : books_per_new_box = 45) : 
  (initial_boxes * books_per_initial_box - damaged_books) % books_per_new_box = 5 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_repacking_l1620_162053


namespace NUMINAMATH_CALUDE_tamikas_driving_time_l1620_162085

/-- Tamika's driving time given the conditions of the problem -/
theorem tamikas_driving_time :
  ∀ (tamika_speed logan_speed : ℝ) 
    (logan_time extra_distance : ℝ),
  tamika_speed > 0 →
  logan_speed > 0 →
  logan_time > 0 →
  tamika_speed = 45 →
  logan_speed = 55 →
  logan_time = 5 →
  extra_distance = 85 →
  ∃ (tamika_time : ℝ),
    tamika_time * tamika_speed = logan_time * logan_speed + extra_distance ∧
    tamika_time = 8 :=
by sorry

end NUMINAMATH_CALUDE_tamikas_driving_time_l1620_162085


namespace NUMINAMATH_CALUDE_weight_difference_is_correct_l1620_162074

/-- The difference in grams between the total weight of oranges and apples -/
def weight_difference : ℝ :=
  let apple_weight_oz : ℝ := 27.5
  let apple_unit_weight_oz : ℝ := 1.5
  let orange_count_dozen : ℝ := 5.5
  let orange_unit_weight_g : ℝ := 45
  let oz_to_g_conversion : ℝ := 28.35

  let apple_weight_g : ℝ := apple_weight_oz * oz_to_g_conversion
  let orange_count : ℝ := orange_count_dozen * 12
  let orange_weight_g : ℝ := orange_count * orange_unit_weight_g

  orange_weight_g - apple_weight_g

theorem weight_difference_is_correct :
  weight_difference = 2190.375 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_is_correct_l1620_162074


namespace NUMINAMATH_CALUDE_min_fraction_value_l1620_162004

theorem min_fraction_value (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100)
  (h : y^2 - 1 = a^2 * (x^2 - 1)) :
  ∃ (k : ℚ), k = 2 ∧ (∀ (a' x' : ℕ), a' > 100 → x' > 100 → ∃ (y' : ℕ), y' > 100 ∧ 
    y'^2 - 1 = a'^2 * (x'^2 - 1) → (a' : ℚ) / x' ≥ k) ∧
  (∃ (a'' x'' y'' : ℕ), a'' > 100 ∧ x'' > 100 ∧ y'' > 100 ∧
    y''^2 - 1 = a''^2 * (x''^2 - 1) ∧ (a'' : ℚ) / x'' = k) :=
by sorry

end NUMINAMATH_CALUDE_min_fraction_value_l1620_162004


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1620_162022

theorem quadratic_solution_difference_squared : 
  ∀ α β : ℝ, α ≠ β → α^2 = 2*α + 2 → β^2 = 2*β + 2 → (α - β)^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1620_162022


namespace NUMINAMATH_CALUDE_line_segment_length_l1620_162014

/-- Given two points M(-2, a) and N(a, 4) on a line with slope -1/2,
    prove that the distance between M and N is 6√3. -/
theorem line_segment_length (a : ℝ) : 
  (4 - a) / (a + 2) = -1/2 →
  Real.sqrt ((a + 2)^2 + (4 - a)^2) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_length_l1620_162014


namespace NUMINAMATH_CALUDE_ball_catching_circle_l1620_162008

theorem ball_catching_circle (n : ℕ) (skip : ℕ) (h1 : n = 50) (h2 : skip = 6) :
  ∃ (m : ℕ), m = 25 ∧ m = n - (n.lcm skip / skip) :=
sorry

end NUMINAMATH_CALUDE_ball_catching_circle_l1620_162008


namespace NUMINAMATH_CALUDE_sum_inequality_l1620_162047

/-- Given real numbers x₁, x₂, x₃ such that the sum of any two is greater than the third,
    prove that (2/3) * (∑ xᵢ) * (∑ xᵢ²) > ∑ xᵢ³ + x₁x₂x₃ -/
theorem sum_inequality (x₁ x₂ x₃ : ℝ) 
    (h₁ : x₁ + x₂ > x₃) (h₂ : x₂ + x₃ > x₁) (h₃ : x₃ + x₁ > x₂) :
    2/3 * (x₁ + x₂ + x₃) * (x₁^2 + x₂^2 + x₃^2) > x₁^3 + x₂^3 + x₃^3 + x₁*x₂*x₃ := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l1620_162047


namespace NUMINAMATH_CALUDE_valid_arrangement_implies_jack_kelly_beside_nate_l1620_162086

/-- Represents the people sitting around the table -/
inductive Person : Type
  | Jack : Person
  | Kelly : Person
  | Lan : Person
  | Mihai : Person
  | Nate : Person

/-- Represents a circular arrangement of 5 people -/
def Arrangement := Fin 5 → Person

/-- Checks if two positions are adjacent in a circular arrangement -/
def areAdjacent (i j : Fin 5) : Bool :=
  (i = j + 1) ∨ (j = i + 1) ∨ (i = 0 ∧ j = 4) ∨ (j = 0 ∧ i = 4)

/-- Checks if the arrangement satisfies the given conditions -/
def isValidArrangement (arr : Arrangement) : Prop :=
  ∃ (i j : Fin 5),
    (areAdjacent i j) ∧
    (arr i = Person.Lan ∧ arr j = Person.Mihai) ∧
    ∀ (k l : Fin 5),
      (arr k = Person.Jack ∧ arr l = Person.Kelly) → ¬(areAdjacent k l)

/-- Checks if Jack and Kelly are on either side of Nate in the arrangement -/
def jackKellyBesideNate (arr : Arrangement) : Prop :=
  ∃ (i j k : Fin 5),
    areAdjacent i j ∧ areAdjacent j k ∧
    arr i = Person.Jack ∧ arr j = Person.Nate ∧ arr k = Person.Kelly

/-- The main theorem stating that any valid arrangement must have Jack and Kelly beside Nate -/
theorem valid_arrangement_implies_jack_kelly_beside_nate
  (arr : Arrangement) (h : isValidArrangement arr) :
  jackKellyBesideNate arr :=
sorry

end NUMINAMATH_CALUDE_valid_arrangement_implies_jack_kelly_beside_nate_l1620_162086


namespace NUMINAMATH_CALUDE_vector_magnitude_AB_l1620_162003

/-- The magnitude of the vector from point A(1, 0) to point B(0, -1) is √2 -/
theorem vector_magnitude_AB : Real.sqrt 2 = Real.sqrt ((0 - 1)^2 + (-1 - 0)^2) := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_AB_l1620_162003


namespace NUMINAMATH_CALUDE_gcd_problem_l1620_162098

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 997) :
  Int.gcd (3 * b^2 + 34 * b + 102) (b + 21) = 21 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1620_162098


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1620_162010

theorem polynomial_factorization (x : ℤ) :
  5 * (x + 4) * (x + 7) * (x + 9) * (x + 11) - 4 * x^2 =
  (5 * x + 63) * (x + 3) * (x + 5) * (x + 21) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1620_162010


namespace NUMINAMATH_CALUDE_chord_length_l1620_162048

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := 3*x + 4*y - 11 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ line_l A.1 A.2 ∧ line_l B.1 B.2

-- Theorem statement
theorem chord_length (A B : ℝ × ℝ) :
  intersection_points A B → abs (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l1620_162048


namespace NUMINAMATH_CALUDE_job_age_is_five_l1620_162090

def freddy_age : ℕ := 18
def stephanie_age : ℕ := freddy_age + 2

theorem job_age_is_five :
  ∃ (job_age : ℕ), stephanie_age = 4 * job_age ∧ job_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_job_age_is_five_l1620_162090


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1620_162033

/-- Given vectors a = (x, 2) and b = (1, y) where x > 0, y > 0, and a ⋅ b = 1,
    the minimum value of 1/x + 2/y is 35/6 -/
theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_dot_product : x * 1 + 2 * y = 1) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' * 1 + 2 * y' = 1 → 1 / x + 2 / y ≤ 1 / x' + 2 / y') ∧
  1 / x + 2 / y = 35 / 6 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1620_162033


namespace NUMINAMATH_CALUDE_decreasing_interval_minimum_a_l1620_162084

noncomputable section

/-- The function f(x) = (2 - a)(x - 1) - 2ln(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

/-- The function g(x) = f(x) + x -/
def g (a : ℝ) (x : ℝ) : ℝ := f a x + x

/-- The derivative of g(x) -/
def g' (a : ℝ) (x : ℝ) : ℝ := 3 - a - 2 / x

theorem decreasing_interval (a : ℝ) :
  (g' a 1 = -1 ∧ g a 1 = 1) →
  ∀ x, 0 < x → x < 2 → g' a x < 0 :=
sorry

theorem minimum_a :
  (∀ x, 0 < x → x < 1/2 → f a x > 0) →
  a ≥ 2 - 4 * Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_decreasing_interval_minimum_a_l1620_162084


namespace NUMINAMATH_CALUDE_exactly_one_absent_probability_l1620_162077

/-- The probability of a student being absent on a given day -/
def p_absent : ℚ := 1 / 15

/-- The probability of a student being present on a given day -/
def p_present : ℚ := 1 - p_absent

/-- The number of students chosen -/
def n : ℕ := 3

/-- The number of students that should be absent -/
def k : ℕ := 1

theorem exactly_one_absent_probability :
  (n.choose k : ℚ) * p_absent^k * p_present^(n - k) = 588 / 3375 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_absent_probability_l1620_162077


namespace NUMINAMATH_CALUDE_f_properties_l1620_162043

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x) * (Real.sin (ω * x) + Real.cos (ω * x)) - 1/2

theorem f_properties (ω : ℝ) (h_ω : ω > 0) (h_period : (2 * π) / (2 * ω) = 2 * π) :
  let f_max := f ω π
  let f_min := f ω (-π/2)
  let α := π/3
  let β := π/6
  (∀ x ∈ Set.Icc (-π) π, f ω x ≤ f_max) ∧
  (∀ x ∈ Set.Icc (-π) π, f ω x ≥ f_min) ∧
  (f_max = 1/2) ∧
  (f_min = -Real.sqrt 2 / 2) ∧
  (α + 2 * β = 2 * π / 3) ∧
  (f ω (α + π/2) * f ω (2 * β + 3 * π/2) = Real.sqrt 3 / 8) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1620_162043


namespace NUMINAMATH_CALUDE_min_colors_correct_min_colors_is_minimum_l1620_162097

-- Define a function that returns the minimum number of colors needed
def min_colors (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- Edge case: no keys
  | 1 => 1
  | 2 => 2
  | _ => 3

-- Theorem statement
theorem min_colors_correct (n : ℕ) :
  min_colors n = 
    if n = 0 then 0
    else if n = 1 then 1
    else if n = 2 then 2
    else 3 :=
by sorry

-- Theorem stating that this is indeed the minimum
theorem min_colors_is_minimum (n : ℕ) :
  ∀ (m : ℕ), m < min_colors n → ¬(∃ (coloring : Fin n → Fin m), ∀ (i j : Fin n), i ≠ j → coloring i ≠ coloring j) :=
by sorry

end NUMINAMATH_CALUDE_min_colors_correct_min_colors_is_minimum_l1620_162097


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1620_162078

-- Define the inequality
def inequality (x : ℝ) : Prop := -x^2 + 5*x > 6

-- Define the solution set
def solution_set : Set ℝ := {x | 2 < x ∧ x < 3}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1620_162078


namespace NUMINAMATH_CALUDE_remainder_of_3_pow_19_mod_10_l1620_162039

theorem remainder_of_3_pow_19_mod_10 : 3^19 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_pow_19_mod_10_l1620_162039


namespace NUMINAMATH_CALUDE_base12_remainder_theorem_l1620_162024

/-- Converts a base-12 number to decimal --/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 12^(digits.length - 1 - i)) 0

/-- The base-12 representation of 2625₁₂ --/
def base12Number : List Nat := [2, 6, 2, 5]

theorem base12_remainder_theorem :
  (base12ToDecimal base12Number) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_base12_remainder_theorem_l1620_162024


namespace NUMINAMATH_CALUDE_bowling_ball_surface_area_l1620_162076

theorem bowling_ball_surface_area :
  ∀ d r A : ℝ,
  d = 9 →
  r = d / 2 →
  A = 4 * Real.pi * r^2 →
  A = 81 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_surface_area_l1620_162076


namespace NUMINAMATH_CALUDE_digit_difference_in_base_l1620_162025

/-- Represents a digit in base d -/
def Digit (d : ℕ) := {n : ℕ // n < d}

/-- Converts a two-digit number AB in base d to its decimal representation -/
def toDecimal (d : ℕ) (A B : Digit d) : ℕ := d * A.val + B.val

theorem digit_difference_in_base (d : ℕ) (A B : Digit d) 
  (hd : d > 5)
  (h : toDecimal d A B + toDecimal d A A = 150) :
  (A.val : ℚ) - (B.val : ℚ) = (-d + 10) / 3 := by sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_l1620_162025


namespace NUMINAMATH_CALUDE_orange_removal_theorem_l1620_162051

/-- Represents the number of oranges Mary must put back to achieve the desired average price -/
def oranges_to_remove (apple_price orange_price : ℚ) (total_fruits : ℕ) (initial_avg_price desired_avg_price : ℚ) : ℚ :=
  (total_fruits * initial_avg_price - total_fruits * desired_avg_price) / (orange_price - desired_avg_price)

theorem orange_removal_theorem (apple_price orange_price : ℚ) (total_fruits : ℕ) (initial_avg_price desired_avg_price : ℚ) :
  apple_price = 40/100 ∧ 
  orange_price = 60/100 ∧ 
  total_fruits = 10 ∧ 
  initial_avg_price = 54/100 ∧ 
  desired_avg_price = 50/100 → 
  oranges_to_remove apple_price orange_price total_fruits initial_avg_price desired_avg_price = 4 := by
  sorry

#eval oranges_to_remove (40/100) (60/100) 10 (54/100) (50/100)

end NUMINAMATH_CALUDE_orange_removal_theorem_l1620_162051


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l1620_162091

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  totalUnitCubes : Nat
  paintedSquaresPerFace : Nat
  paintedFaces : Nat

/-- Calculates the number of unpainted unit cubes in a painted cube -/
def unpaintedUnitCubes (cube : PaintedCube) : Nat :=
  cube.totalUnitCubes - paintedUnitCubes cube
where
  /-- Calculates the number of painted unit cubes, accounting for overlaps -/
  paintedUnitCubes (cube : PaintedCube) : Nat :=
    let totalPaintedSquares := cube.paintedSquaresPerFace * cube.paintedFaces
    let edgeOverlap := 12 * 2  -- 12 edges, each counted twice
    let cornerOverlap := 8 * 2  -- 8 corners, each counted thrice (so subtract 2)
    totalPaintedSquares - edgeOverlap - cornerOverlap

/-- The theorem to be proved -/
theorem unpainted_cubes_in_6x6x6 :
  let cube : PaintedCube := {
    size := 6,
    totalUnitCubes := 216,
    paintedSquaresPerFace := 13,
    paintedFaces := 6
  }
  unpaintedUnitCubes cube = 210 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l1620_162091


namespace NUMINAMATH_CALUDE_locus_of_points_m_l1620_162015

-- Define the given circle
structure GivenCircle where
  O : ℝ × ℝ  -- Center of the circle
  R : ℝ      -- Radius of the circle
  h : R > 0  -- Radius is positive

-- Define the point A on the given circle
def PointOnCircle (c : GivenCircle) (A : ℝ × ℝ) : Prop :=
  (A.1 - c.O.1)^2 + (A.2 - c.O.2)^2 = c.R^2

-- Define the tangent line at point A
def TangentLine (c : GivenCircle) (A : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ M => (M.1 - A.1) * (A.1 - c.O.1) + (M.2 - A.2) * (A.2 - c.O.2) = 0

-- Define the segment AM with length a
def SegmentAM (A M : ℝ × ℝ) (a : ℝ) : Prop :=
  (M.1 - A.1)^2 + (M.2 - A.2)^2 = a^2

-- Theorem: The locus of points M forms a circle concentric with the given circle
theorem locus_of_points_m (c : GivenCircle) (a : ℝ) (h : a > 0) :
  ∀ A M : ℝ × ℝ,
    PointOnCircle c A →
    TangentLine c A M →
    SegmentAM A M a →
    (M.1 - c.O.1)^2 + (M.2 - c.O.2)^2 = c.R^2 + a^2 :=
  sorry

end NUMINAMATH_CALUDE_locus_of_points_m_l1620_162015


namespace NUMINAMATH_CALUDE_day_crew_load_fraction_l1620_162088

/-- Fraction of boxes loaded by day crew given night crew conditions -/
theorem day_crew_load_fraction (D W : ℚ) : 
  D > 0 → W > 0 →
  (D * W) / ((D * W) + ((3/4 * D) * (4/7 * W))) = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_day_crew_load_fraction_l1620_162088


namespace NUMINAMATH_CALUDE_min_side_c_value_l1620_162059

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the minimum value of c is approximately 2.25 -/
theorem min_side_c_value (a b c : ℝ) (A B C : ℝ) : 
  b = 2 →
  c * Real.cos B + b * Real.cos C = 4 * a * Real.sin B * Real.sin C →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ c ≥ 2.25 - ε :=
sorry

end NUMINAMATH_CALUDE_min_side_c_value_l1620_162059


namespace NUMINAMATH_CALUDE_rachel_steps_up_correct_l1620_162046

/-- The number of steps Rachel climbed going up the Eiffel Tower -/
def steps_up : ℕ := 567

/-- The number of steps Rachel climbed going down the Eiffel Tower -/
def steps_down : ℕ := 325

/-- The total number of steps Rachel climbed -/
def total_steps : ℕ := 892

/-- Theorem: The number of steps Rachel climbed going up is correct -/
theorem rachel_steps_up_correct : steps_up + steps_down = total_steps := by
  sorry

end NUMINAMATH_CALUDE_rachel_steps_up_correct_l1620_162046


namespace NUMINAMATH_CALUDE_fresh_produce_to_soda_ratio_l1620_162005

/-- Proves that the ratio of fresh produce weight to soda weight is 2:1 --/
theorem fresh_produce_to_soda_ratio :
  let empty_truck_weight : ℕ := 12000
  let soda_crates : ℕ := 20
  let soda_crate_weight : ℕ := 50
  let dryers : ℕ := 3
  let dryer_weight : ℕ := 3000
  let loaded_truck_weight : ℕ := 24000
  let soda_weight := soda_crates * soda_crate_weight
  let dryers_weight := dryers * dryer_weight
  let fresh_produce_weight := loaded_truck_weight - (empty_truck_weight + soda_weight + dryers_weight)
  (fresh_produce_weight : ℚ) / soda_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_fresh_produce_to_soda_ratio_l1620_162005


namespace NUMINAMATH_CALUDE_johns_next_birthday_age_l1620_162032

/-- Proves that John's age on his next birthday is 9, given the conditions of the problem -/
theorem johns_next_birthday_age (john carl beth : ℝ) 
  (h1 : john = 0.75 * carl)  -- John is 25% younger than Carl
  (h2 : carl = 1.3 * beth)   -- Carl is 30% older than Beth
  (h3 : john + carl + beth = 30) -- Sum of their ages is 30
  : ⌈john⌉ = 9 := by
  sorry

end NUMINAMATH_CALUDE_johns_next_birthday_age_l1620_162032


namespace NUMINAMATH_CALUDE_name_is_nika_l1620_162050

-- Define a cube face
inductive Face
| Front
| Back
| Left
| Right
| Top
| Bottom

-- Define a letter
inductive Letter
| N
| I
| K
| A
| T

-- Define a cube
structure Cube where
  faces : Face → Letter

-- Define the arrangement of cubes
def CubeArrangement := List Cube

-- Define the function to get the front-facing letters
def getFrontLetters (arrangement : CubeArrangement) : List Letter :=
  arrangement.map (λ cube => cube.faces Face.Front)

-- Theorem statement
theorem name_is_nika (arrangement : CubeArrangement) 
  (h1 : arrangement.length = 4)
  (h2 : getFrontLetters arrangement = [Letter.N, Letter.I, Letter.K, Letter.A]) :
  "Ника" = "Ника" :=
by sorry

end NUMINAMATH_CALUDE_name_is_nika_l1620_162050


namespace NUMINAMATH_CALUDE_closest_fraction_to_japan_medals_l1620_162068

theorem closest_fraction_to_japan_medals :
  let japan_fraction : ℚ := 25 / 120
  let fractions : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]
  (1/5 : ℚ) = fractions.argmin (fun x => |x - japan_fraction|) := by
  sorry

end NUMINAMATH_CALUDE_closest_fraction_to_japan_medals_l1620_162068


namespace NUMINAMATH_CALUDE_skater_race_solution_l1620_162011

/-- Represents the speeds and times of two speed skaters in a race --/
structure SkaterRace where
  v : ℝ  -- Speed of the second skater in m/s
  t1 : ℝ  -- Time for the first skater to complete 10000 m in seconds
  t2 : ℝ  -- Time for the second skater to complete 10000 m in seconds

/-- The speeds and times of the skaters satisfy the race conditions --/
def satisfies_conditions (race : SkaterRace) : Prop :=
  let v1 := race.v + 1/3  -- Speed of the first skater
  (v1 * 600 - race.v * 600 = 200) ∧  -- Overtaking condition
  (400 / race.v - 400 / v1 = 2) ∧  -- Lap time difference
  (10000 / v1 = race.t1) ∧  -- First skater's total time
  (10000 / race.v = race.t2)  -- Second skater's total time

/-- The theorem stating the correct speeds and times for the skaters --/
theorem skater_race_solution :
  ∃ (race : SkaterRace),
    satisfies_conditions race ∧
    race.v = 8 ∧
    race.t1 = 1200 ∧
    race.t2 = 1250 :=
sorry

end NUMINAMATH_CALUDE_skater_race_solution_l1620_162011


namespace NUMINAMATH_CALUDE_line_translation_slope_l1620_162075

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line horizontally and vertically -/
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept + dy - l.slope * dx }

theorem line_translation_slope (l : Line) :
  translate (translate l 3 0) 0 1 = l →
  l.slope = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_line_translation_slope_l1620_162075


namespace NUMINAMATH_CALUDE_other_pencil_length_is_12_l1620_162019

/-- The length of Isha's pencil in cubes -/
def ishas_pencil_length : ℕ := 12

/-- The total length of both pencils in cubes -/
def total_length : ℕ := 24

/-- The length of the other pencil in cubes -/
def other_pencil_length : ℕ := total_length - ishas_pencil_length

theorem other_pencil_length_is_12 : other_pencil_length = 12 := by
  sorry

end NUMINAMATH_CALUDE_other_pencil_length_is_12_l1620_162019


namespace NUMINAMATH_CALUDE_bow_count_l1620_162058

theorem bow_count (red_fraction : ℚ) (blue_fraction : ℚ) (yellow_fraction : ℚ) (green_fraction : ℚ) 
  (white_count : ℕ) :
  red_fraction = 1/6 →
  blue_fraction = 1/3 →
  yellow_fraction = 1/12 →
  green_fraction = 1/8 →
  red_fraction + blue_fraction + yellow_fraction + green_fraction + (white_count : ℚ)/144 = 1 →
  white_count = 42 →
  144 = red_fraction * 144 + blue_fraction * 144 + yellow_fraction * 144 + green_fraction * 144 + white_count :=
by sorry

end NUMINAMATH_CALUDE_bow_count_l1620_162058


namespace NUMINAMATH_CALUDE_bryan_skittles_count_l1620_162099

/-- Given that Ben has 20 M&M's and Bryan has 30 more candies than Ben, 
    prove that Bryan has 50 skittles. -/
theorem bryan_skittles_count : 
  ∀ (ben_candies bryan_candies : ℕ),
  ben_candies = 20 →
  bryan_candies = ben_candies + 30 →
  bryan_candies = 50 := by
sorry

end NUMINAMATH_CALUDE_bryan_skittles_count_l1620_162099


namespace NUMINAMATH_CALUDE_subtract_point_five_from_forty_three_point_two_l1620_162009

theorem subtract_point_five_from_forty_three_point_two :
  43.2 - 0.5 = 42.7 := by sorry

end NUMINAMATH_CALUDE_subtract_point_five_from_forty_three_point_two_l1620_162009


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l1620_162023

theorem camping_trip_percentage (total_students : ℕ) 
  (h1 : (22 : ℝ) / 100 * total_students = (25 : ℝ) / 100 * ((88 : ℝ) / 100 * total_students)) 
  (h2 : (75 : ℝ) / 100 * ((88 : ℝ) / 100 * total_students) + (25 : ℝ) / 100 * ((88 : ℝ) / 100 * total_students) = (88 : ℝ) / 100 * total_students) : 
  (88 : ℝ) / 100 * total_students = (22 : ℝ) / (25 / 100) := by
  sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l1620_162023


namespace NUMINAMATH_CALUDE_boys_average_weight_l1620_162079

/-- Proves that the average weight of boys in a class is 48 kg given the specified conditions -/
theorem boys_average_weight (total_students : Nat) (num_boys : Nat) (num_girls : Nat)
  (class_avg_weight : ℝ) (girls_avg_weight : ℝ) :
  total_students = 25 →
  num_boys = 15 →
  num_girls = 10 →
  class_avg_weight = 45 →
  girls_avg_weight = 40.5 →
  (total_students * class_avg_weight - num_girls * girls_avg_weight) / num_boys = 48 := by
  sorry

end NUMINAMATH_CALUDE_boys_average_weight_l1620_162079


namespace NUMINAMATH_CALUDE_houses_traded_l1620_162073

theorem houses_traded (x y z : ℕ) (h : x + y ≥ z) : ∃ t : ℕ, x - t + y = z :=
sorry

end NUMINAMATH_CALUDE_houses_traded_l1620_162073


namespace NUMINAMATH_CALUDE_some_magical_beings_are_enchanting_creatures_l1620_162071

-- Define the sets
variable (W : Set α) -- Wizards
variable (M : Set α) -- Magical beings
variable (E : Set α) -- Enchanting creatures

-- Define the conditions
variable (h1 : W ⊆ M) -- All wizards are magical beings
variable (h2 : ∃ x, x ∈ E ∩ W) -- Some enchanting creatures are wizards

-- State the theorem
theorem some_magical_beings_are_enchanting_creatures :
  ∃ x, x ∈ M ∩ E := by sorry

end NUMINAMATH_CALUDE_some_magical_beings_are_enchanting_creatures_l1620_162071


namespace NUMINAMATH_CALUDE_min_area_triangle_min_area_is_minimum_l1620_162041

/-- The minimum area of a triangle with vertices (0,0), (30,18), and a third point with integer coordinates -/
theorem min_area_triangle : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (30, 18)
  3

/-- The area of the triangle is indeed the minimum possible -/
theorem min_area_is_minimum (p q : ℤ) : 
  let C : ℝ × ℝ := (p, q)
  let area := (1/2 : ℝ) * |18 * p - 30 * q|
  3 ≤ area := by
  sorry

#check min_area_triangle
#check min_area_is_minimum

end NUMINAMATH_CALUDE_min_area_triangle_min_area_is_minimum_l1620_162041


namespace NUMINAMATH_CALUDE_different_color_chips_probability_l1620_162037

/-- The probability of drawing two chips of different colors from a bag with replacement -/
theorem different_color_chips_probability
  (total_chips : ℕ)
  (blue_chips : ℕ)
  (yellow_chips : ℕ)
  (h_total : total_chips = blue_chips + yellow_chips)
  (h_blue : blue_chips = 5)
  (h_yellow : yellow_chips = 3) :
  (blue_chips : ℚ) / total_chips * (yellow_chips : ℚ) / total_chips +
  (yellow_chips : ℚ) / total_chips * (blue_chips : ℚ) / total_chips =
  15 / 32 :=
sorry

end NUMINAMATH_CALUDE_different_color_chips_probability_l1620_162037


namespace NUMINAMATH_CALUDE_prob_three_heads_l1620_162057

/-- The probability of getting heads for a biased coin -/
def p : ℚ := sorry

/-- Condition: probability of 1 head equals probability of 2 heads in 4 flips -/
axiom condition : 4 * p * (1 - p)^3 = 6 * p^2 * (1 - p)^2

/-- Theorem: Probability of 3 heads in 4 flips is 96/625 -/
theorem prob_three_heads : 4 * p^3 * (1 - p) = 96/625 := by sorry

end NUMINAMATH_CALUDE_prob_three_heads_l1620_162057


namespace NUMINAMATH_CALUDE_emily_furniture_assembly_time_l1620_162012

/-- Calculates the total assembly time for furniture -/
def total_assembly_time (
  num_chairs : ℕ) (chair_time : ℕ)
  (num_tables : ℕ) (table_time : ℕ)
  (num_shelves : ℕ) (shelf_time : ℕ)
  (num_wardrobes : ℕ) (wardrobe_time : ℕ) : ℕ :=
  num_chairs * chair_time +
  num_tables * table_time +
  num_shelves * shelf_time +
  num_wardrobes * wardrobe_time

/-- Proves that the total assembly time for Emily's furniture is 137 minutes -/
theorem emily_furniture_assembly_time :
  total_assembly_time 4 8 2 15 3 10 1 45 = 137 := by
  sorry


end NUMINAMATH_CALUDE_emily_furniture_assembly_time_l1620_162012


namespace NUMINAMATH_CALUDE_smallest_x_divisible_by_3_5_11_l1620_162081

theorem smallest_x_divisible_by_3_5_11 : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → 107 * 151 * y % 3 = 0 ∧ 107 * 151 * y % 5 = 0 ∧ 107 * 151 * y % 11 = 0 → x ≤ y) ∧
  107 * 151 * x % 3 = 0 ∧ 107 * 151 * x % 5 = 0 ∧ 107 * 151 * x % 11 = 0 ∧
  x = 165 := by
  sorry

-- Additional definitions to match the problem conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ (m : ℕ), m > 1 → m < n → n % m ≠ 0

axiom prime_107 : is_prime 107
axiom prime_151 : is_prime 151
axiom prime_3 : is_prime 3
axiom prime_5 : is_prime 5
axiom prime_11 : is_prime 11

end NUMINAMATH_CALUDE_smallest_x_divisible_by_3_5_11_l1620_162081


namespace NUMINAMATH_CALUDE_concert_ticket_cost_haleys_concert_cost_l1620_162016

/-- Calculate the total amount spent on concert tickets --/
theorem concert_ticket_cost (ticket_price : ℝ) (num_tickets : ℕ) 
  (discount_rate : ℝ) (discount_threshold : ℕ) (service_fee : ℝ) : ℝ :=
  let base_cost := ticket_price * num_tickets
  let discount := if num_tickets > discount_threshold then discount_rate * base_cost else 0
  let discounted_cost := base_cost - discount
  let total_service_fee := service_fee * num_tickets
  let total_cost := discounted_cost + total_service_fee
  by
    -- Proof goes here
    sorry

/-- Haley's concert ticket purchase --/
theorem haleys_concert_cost : 
  concert_ticket_cost 4 8 0.1 5 2 = 44.8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_concert_ticket_cost_haleys_concert_cost_l1620_162016


namespace NUMINAMATH_CALUDE_marble_selection_ways_l1620_162064

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of marbles -/
def total_marbles : ℕ := 16

/-- The number of colored marbles -/
def colored_marbles : ℕ := 4

/-- The number of non-colored marbles -/
def non_colored_marbles : ℕ := total_marbles - colored_marbles

/-- The number of marbles to be chosen -/
def chosen_marbles : ℕ := 5

/-- The number of colored marbles to be chosen -/
def chosen_colored : ℕ := 2

/-- The number of non-colored marbles to be chosen -/
def chosen_non_colored : ℕ := chosen_marbles - chosen_colored

theorem marble_selection_ways :
  choose colored_marbles chosen_colored * choose non_colored_marbles chosen_non_colored = 1320 :=
sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l1620_162064


namespace NUMINAMATH_CALUDE_investment_difference_l1620_162092

def initial_investment : ℕ := 10000

def alice_multiplier : ℕ := 3
def bob_multiplier : ℕ := 7

def alice_final : ℕ := initial_investment * alice_multiplier
def bob_final : ℕ := initial_investment * bob_multiplier

theorem investment_difference : bob_final - alice_final = 40000 := by
  sorry

end NUMINAMATH_CALUDE_investment_difference_l1620_162092


namespace NUMINAMATH_CALUDE_facial_tissue_price_decrease_l1620_162006

/-- The percent decrease in price per box of facial tissue during a sale -/
theorem facial_tissue_price_decrease (original_price sale_price : ℚ) : 
  original_price = 5 / 4 →
  sale_price = 4 / 5 →
  abs ((original_price - sale_price) / original_price - 9 / 25) < 1 / 100 := by
  sorry

#eval (5/4 : ℚ) -- Original price per box
#eval (4/5 : ℚ) -- Sale price per box
#eval ((5/4 - 4/5) / (5/4) : ℚ) -- Actual percent decrease

end NUMINAMATH_CALUDE_facial_tissue_price_decrease_l1620_162006


namespace NUMINAMATH_CALUDE_negative_comparison_l1620_162018

theorem negative_comparison : -0.5 > -0.7 := by
  sorry

end NUMINAMATH_CALUDE_negative_comparison_l1620_162018


namespace NUMINAMATH_CALUDE_complement_of_union_l1620_162055

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 4}

theorem complement_of_union :
  (Aᶜ ∩ Bᶜ) ∩ U = {3, 5} :=
sorry

end NUMINAMATH_CALUDE_complement_of_union_l1620_162055


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1620_162027

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (a + 2) * (b + 2) ≥ c * d ∧ 
  (∃ (a₀ b₀ c₀ d₀ : ℝ), a₀^2 + b₀^2 + c₀^2 + d₀^2 = 4 ∧ 
    (a₀ + 2) * (b₀ + 2) = c₀ * d₀ ∧ 
    a₀ = -2 ∧ b₀ = -2 ∧ c₀ = 1 ∧ d₀ = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1620_162027


namespace NUMINAMATH_CALUDE_soldier_difference_l1620_162093

/-- Calculates the difference in the number of soldiers between two sides in a war scenario --/
theorem soldier_difference (
  daily_food : ℕ)  -- Daily food requirement per soldier on the first side
  (food_difference : ℕ)  -- Difference in food given to soldiers on the second side
  (first_side_soldiers : ℕ)  -- Number of soldiers on the first side
  (total_food : ℕ)  -- Total amount of food for both sides
  (h1 : daily_food = 10)  -- Each soldier needs 10 pounds of food per day
  (h2 : food_difference = 2)  -- Soldiers on the second side get 2 pounds less food
  (h3 : first_side_soldiers = 4000)  -- The first side has 4000 soldiers
  (h4 : total_food = 68000)  -- The total amount of food for both sides is 68000 pounds
  : (first_side_soldiers - (total_food - first_side_soldiers * daily_food) / (daily_food - food_difference) = 500) :=
by sorry

end NUMINAMATH_CALUDE_soldier_difference_l1620_162093


namespace NUMINAMATH_CALUDE_divisor_problem_l1620_162070

theorem divisor_problem (d : ℕ) (h_pos : d > 0) :
  1200 % d = 3 ∧ 1640 % d = 2 ∧ 1960 % d = 7 → d = 9 ∨ d = 21 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l1620_162070


namespace NUMINAMATH_CALUDE_no_quadratic_polynomial_satisfies_conditions_l1620_162049

theorem no_quadratic_polynomial_satisfies_conditions :
  ¬∃ (f : ℝ → ℝ), 
    (∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c)) ∧ 
    (∀ x, f (x^2) = x^4) ∧
    (∀ x, f (f x) = (x^2 + 1)^4) :=
by sorry

end NUMINAMATH_CALUDE_no_quadratic_polynomial_satisfies_conditions_l1620_162049


namespace NUMINAMATH_CALUDE_math_team_selection_count_l1620_162017

theorem math_team_selection_count : 
  let total_boys : ℕ := 7
  let total_girls : ℕ := 10
  let team_size : ℕ := 5
  let boys_in_team : ℕ := 2
  let girls_in_team : ℕ := 3
  (Nat.choose total_boys boys_in_team) * (Nat.choose total_girls girls_in_team) = 2520 :=
by sorry

end NUMINAMATH_CALUDE_math_team_selection_count_l1620_162017


namespace NUMINAMATH_CALUDE_eighteen_percent_of_x_is_ninety_l1620_162094

theorem eighteen_percent_of_x_is_ninety (x : ℝ) : (18 / 100) * x = 90 → x = 500 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_percent_of_x_is_ninety_l1620_162094


namespace NUMINAMATH_CALUDE_isosceles_triangle_l1620_162066

theorem isosceles_triangle (a b c : ℝ) (α β γ : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < α ∧ 0 < β ∧ 0 < γ →
  α + β + γ = π →
  a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β) →
  α = β := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l1620_162066


namespace NUMINAMATH_CALUDE_megan_markers_proof_l1620_162000

def final_markers (initial : ℕ) (robert_factor : ℕ) (elizabeth_taken : ℕ) : ℕ :=
  initial + robert_factor * initial - elizabeth_taken

theorem megan_markers_proof :
  final_markers 2475 3 1650 = 8250 := by
  sorry

end NUMINAMATH_CALUDE_megan_markers_proof_l1620_162000


namespace NUMINAMATH_CALUDE_infinite_pairs_geometric_progression_l1620_162060

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricProgression (seq : Fin 4 → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ i : Fin 3, seq (i + 1) = seq i * r

/-- There are infinitely many pairs of real numbers (a,b) such that 12, a, b, ab form a geometric progression. -/
theorem infinite_pairs_geometric_progression :
  {(a, b) : ℝ × ℝ | IsGeometricProgression (λ i => match i with
    | 0 => 12
    | 1 => a
    | 2 => b
    | 3 => a * b)} = Set.univ := by
  sorry


end NUMINAMATH_CALUDE_infinite_pairs_geometric_progression_l1620_162060


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l1620_162034

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 49 →
  3 * girls = 4 * boys →
  total_students = boys + girls →
  girls - boys = 7 := by
sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l1620_162034


namespace NUMINAMATH_CALUDE_escalator_problem_l1620_162087

/-- The number of steps Petya counted while ascending the escalator -/
def steps_ascending : ℕ := 75

/-- The number of steps Petya counted while descending the escalator -/
def steps_descending : ℕ := 150

/-- The ratio of Petya's descending speed to ascending speed -/
def speed_ratio : ℚ := 3

/-- The speed of the escalator in steps per unit time -/
def escalator_speed : ℚ := 3/5

/-- The number of steps on the stopped escalator -/
def escalator_length : ℕ := 120

theorem escalator_problem :
  let ascending_speed : ℚ := 1 + escalator_speed
  let descending_speed : ℚ := speed_ratio - escalator_speed
  steps_ascending * ascending_speed = (steps_descending / speed_ratio) * descending_speed ∧
  escalator_length = steps_ascending * ascending_speed := by
  sorry

#check escalator_problem

end NUMINAMATH_CALUDE_escalator_problem_l1620_162087


namespace NUMINAMATH_CALUDE_matrix_equation_l1620_162082

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]

def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]

theorem matrix_equation (x y z w : ℝ) 
  (h1 : A * B x y z w = B x y z w * A) 
  (h2 : 4 * y ≠ z) : 
  (x - w) / (z - 4 * y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_l1620_162082


namespace NUMINAMATH_CALUDE_number_of_boys_l1620_162096

/-- The number of boys in a school with the given conditions -/
theorem number_of_boys (total : ℕ) (boys : ℕ) : 
  total = 400 → 
  boys + (boys * total) / 100 = total →
  boys = 80 :=
by sorry

end NUMINAMATH_CALUDE_number_of_boys_l1620_162096


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1620_162061

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 11

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties :
  (∀ x, f x ≤ 13) ∧  -- Maximum value is 13
  f 3 = 5 ∧          -- f(3) = 5
  f (-1) = 5 ∧       -- f(-1) = 5
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) -- f is a quadratic function
  :=
by
  sorry

#check quadratic_function_properties

end NUMINAMATH_CALUDE_quadratic_function_properties_l1620_162061


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_l1620_162029

theorem least_four_digit_multiple : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m → n ≤ m) ∧
  n = 1050 :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_l1620_162029


namespace NUMINAMATH_CALUDE_no_increasing_sequence_with_finite_primes_l1620_162080

theorem no_increasing_sequence_with_finite_primes :
  ¬ ∃ (a : ℕ → ℕ),
    (∀ n : ℕ, a n < a (n + 1)) ∧
    (∀ c : ℤ, ∃ N : ℕ, ∀ n ≥ N, ¬ (Prime (c + a n))) :=
by sorry

end NUMINAMATH_CALUDE_no_increasing_sequence_with_finite_primes_l1620_162080


namespace NUMINAMATH_CALUDE_sqrt_a_plus_b_is_three_l1620_162052

theorem sqrt_a_plus_b_is_three (a b : ℝ) 
  (h1 : 2*a - 1 = 9) 
  (h2 : 3*a + 2*b + 4 = 27) : 
  Real.sqrt (a + b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_plus_b_is_three_l1620_162052


namespace NUMINAMATH_CALUDE_promotion_savings_l1620_162056

/-- Represents a shoe promotion strategy -/
inductive Promotion
  | A
  | B

/-- Calculates the total cost for two pairs of shoes given a promotion -/
def calculateCost (originalPrice : ℕ) (promo : Promotion) : ℕ :=
  match promo with
  | Promotion.A => originalPrice + originalPrice / 2
  | Promotion.B => originalPrice + originalPrice - 15

/-- Calculates the savings from using one promotion over another -/
def calculateSavings (originalPrice : ℕ) (promo1 promo2 : Promotion) : ℕ :=
  calculateCost originalPrice promo2 - calculateCost originalPrice promo1

theorem promotion_savings :
  calculateSavings 50 Promotion.A Promotion.B = 10 := by
  sorry

#eval calculateSavings 50 Promotion.A Promotion.B

end NUMINAMATH_CALUDE_promotion_savings_l1620_162056


namespace NUMINAMATH_CALUDE_plane_through_origin_l1620_162035

/-- A plane in 3D Cartesian coordinates represented by the equation Ax + By + Cz = 0 -/
structure Plane3D where
  A : ℝ
  B : ℝ
  C : ℝ
  not_all_zero : A ≠ 0 ∨ B ≠ 0 ∨ C ≠ 0

/-- A point in 3D Cartesian coordinates -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The origin in 3D Cartesian coordinates -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- A point lies on a plane if it satisfies the plane's equation -/
def lies_on (p : Point3D) (plane : Plane3D) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z = 0

/-- A plane passes through the origin if the origin lies on the plane -/
def passes_through_origin (plane : Plane3D) : Prop :=
  lies_on origin plane

theorem plane_through_origin (plane : Plane3D) : 
  passes_through_origin plane :=
sorry

end NUMINAMATH_CALUDE_plane_through_origin_l1620_162035


namespace NUMINAMATH_CALUDE_equations_hold_l1620_162031

-- Define the equations
def equation1 : ℝ := 6.8 + 4.1 + 1.1
def equation2 : ℝ := 6.2 + 6.2 + 7.6
def equation3 : ℝ := 19.9 - 4.3 - 5.6

-- State the theorem
theorem equations_hold :
  equation1 = 12 ∧ equation2 = 20 ∧ equation3 = 10 := by sorry

end NUMINAMATH_CALUDE_equations_hold_l1620_162031


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_achievable_486_largest_k_is_486_l1620_162020

theorem largest_consecutive_sum (k : ℕ) : 
  (∃ a : ℕ, (k * (2 * a + k - 1)) / 2 = 3^11) → k ≤ 486 :=
by
  sorry

theorem achievable_486 : 
  ∃ a : ℕ, (486 * (2 * a + 486 - 1)) / 2 = 3^11 :=
by
  sorry

theorem largest_k_is_486 : 
  (∃ k : ℕ, k > 486 ∧ ∃ a : ℕ, (k * (2 * a + k - 1)) / 2 = 3^11) → False :=
by
  sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_achievable_486_largest_k_is_486_l1620_162020


namespace NUMINAMATH_CALUDE_kramers_packing_rate_l1620_162044

/-- Kramer's cigarette packing rate -/
theorem kramers_packing_rate 
  (boxes_per_case : ℕ) 
  (cases_packed : ℕ) 
  (packing_time_hours : ℕ) 
  (h1 : boxes_per_case = 5)
  (h2 : cases_packed = 240)
  (h3 : packing_time_hours = 2) :
  (boxes_per_case * cases_packed) / (packing_time_hours * 60) = 10 := by
  sorry

#check kramers_packing_rate

end NUMINAMATH_CALUDE_kramers_packing_rate_l1620_162044


namespace NUMINAMATH_CALUDE_total_sleep_week_is_366_l1620_162002

def connor_sleep : ℕ := 6
def luke_sleep : ℕ := connor_sleep + 2
def emma_sleep : ℕ := connor_sleep - 1
def ava_sleep (day : ℕ) : ℕ := 5 + (day - 1) / 2
def puppy_sleep : ℕ := 2 * luke_sleep
def cat_sleep : ℕ := 4 + 7

def total_sleep_week : ℕ :=
  7 * connor_sleep +
  7 * luke_sleep +
  7 * emma_sleep +
  (ava_sleep 1 + ava_sleep 2 + ava_sleep 3 + ava_sleep 4 + ava_sleep 5 + ava_sleep 6 + ava_sleep 7) +
  7 * puppy_sleep +
  7 * cat_sleep

theorem total_sleep_week_is_366 : total_sleep_week = 366 := by
  sorry

end NUMINAMATH_CALUDE_total_sleep_week_is_366_l1620_162002


namespace NUMINAMATH_CALUDE_revenue_not_increased_l1620_162067

/-- The revenue function for the current year -/
def revenue (x : ℝ) : ℝ := 4*x^3 - 20*x^2 + 33*x - 17

/-- The previous year's revenue -/
def previous_revenue : ℝ := 20

theorem revenue_not_increased (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) : 
  revenue x ≤ previous_revenue := by
  sorry

#check revenue_not_increased

end NUMINAMATH_CALUDE_revenue_not_increased_l1620_162067


namespace NUMINAMATH_CALUDE_susan_ate_six_candies_l1620_162065

/-- The number of candies Susan bought on Tuesday -/
def tuesday_candies : ℕ := 3

/-- The number of candies Susan bought on Thursday -/
def thursday_candies : ℕ := 5

/-- The number of candies Susan bought on Friday -/
def friday_candies : ℕ := 2

/-- The number of candies Susan has left -/
def remaining_candies : ℕ := 4

/-- The total number of candies Susan bought -/
def total_candies : ℕ := tuesday_candies + thursday_candies + friday_candies

theorem susan_ate_six_candies : total_candies - remaining_candies = 6 := by
  sorry

end NUMINAMATH_CALUDE_susan_ate_six_candies_l1620_162065


namespace NUMINAMATH_CALUDE_ticket_cost_count_l1620_162038

def ticket_cost_possibilities (total_11th : ℕ) (total_12th : ℕ) : ℕ :=
  (Finset.filter (fun x : ℕ => 
    x > 0 ∧ 
    total_11th % x = 0 ∧ 
    total_12th % x = 0 ∧ 
    total_11th / x < total_12th / x) 
    (Finset.range (min total_11th total_12th + 1))).card

theorem ticket_cost_count : ticket_cost_possibilities 108 90 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ticket_cost_count_l1620_162038


namespace NUMINAMATH_CALUDE_pacos_marble_purchase_l1620_162072

theorem pacos_marble_purchase : 
  0.33 + 0.33 + 0.08 = 0.74 := by sorry

end NUMINAMATH_CALUDE_pacos_marble_purchase_l1620_162072


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l1620_162063

theorem lcm_of_ratio_and_hcf (a b : ℕ+) :
  (a : ℚ) / b = 14 / 21 →
  Nat.gcd a b = 28 →
  Nat.lcm a b = 1176 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l1620_162063


namespace NUMINAMATH_CALUDE_smallest_three_digit_candy_number_l1620_162042

theorem smallest_three_digit_candy_number : ∃ n : ℕ,
  (100 ≤ n ∧ n < 1000) ∧
  (n - 7) % 9 = 0 ∧
  (n + 9) % 7 = 0 ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n) → (m - 7) % 9 ≠ 0 ∨ (m + 9) % 7 ≠ 0) ∧
  n = 124 := by
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_candy_number_l1620_162042


namespace NUMINAMATH_CALUDE_constant_dot_product_l1620_162007

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + 3*y^2/4 = 1

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a tangent line to circle O
def tangent_line (k m : ℝ) (x y : ℝ) : Prop := y = k*x + m ∧ 1 + k^2 = m^2

-- Define the intersection points of the tangent line and ellipse C
def intersection_points (k m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧ 
  tangent_line k m x₁ y₁ ∧ tangent_line k m x₂ y₂

-- Theorem statement
theorem constant_dot_product :
  ∀ (k m x₁ y₁ x₂ y₂ : ℝ),
  intersection_points k m x₁ y₁ x₂ y₂ →
  x₁ * x₂ + y₁ * y₂ = 0 :=
sorry

end NUMINAMATH_CALUDE_constant_dot_product_l1620_162007


namespace NUMINAMATH_CALUDE_warehouse_storage_problem_l1620_162083

/-- Represents the warehouse storage problem -/
theorem warehouse_storage_problem 
  (second_floor_space : ℝ) 
  (h1 : second_floor_space > 0) 
  (h2 : 3 * second_floor_space - (1/4) * second_floor_space = 55000) : 
  (1/4) * second_floor_space = 5000 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_storage_problem_l1620_162083
