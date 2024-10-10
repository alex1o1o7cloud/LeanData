import Mathlib

namespace CO2_yield_calculation_l1411_141196

-- Define the chemical equation
def chemical_equation : String := "HCl + NaHCO3 → NaCl + H2O + CO2"

-- Define the molar quantities of reactants
def moles_HCl : ℝ := 1
def moles_NaHCO3 : ℝ := 1

-- Define the molar mass of CO2
def molar_mass_CO2 : ℝ := 44.01

-- Define the theoretical yield function
def theoretical_yield (moles_reactant : ℝ) (molar_mass_product : ℝ) : ℝ :=
  moles_reactant * molar_mass_product

-- Theorem statement
theorem CO2_yield_calculation :
  theoretical_yield (min moles_HCl moles_NaHCO3) molar_mass_CO2 = 44.01 := by
  sorry


end CO2_yield_calculation_l1411_141196


namespace sharp_composition_l1411_141184

def sharp (N : ℝ) : ℝ := 0.4 * N * 1.5

theorem sharp_composition : sharp (sharp (sharp 80)) = 17.28 := by
  sorry

end sharp_composition_l1411_141184


namespace least_product_of_reciprocal_sum_l1411_141194

theorem least_product_of_reciprocal_sum (a b : ℕ+) 
  (h : (a : ℚ)⁻¹ + (3 * b : ℚ)⁻¹ = (9 : ℚ)⁻¹) : 
  (∀ c d : ℕ+, (c : ℚ)⁻¹ + (3 * d : ℚ)⁻¹ = (9 : ℚ)⁻¹ → (a * b : ℕ) ≤ (c * d : ℕ)) ∧ 
  (a * b : ℕ) = 144 := by
sorry

end least_product_of_reciprocal_sum_l1411_141194


namespace sum_of_coordinates_of_D_l1411_141185

/-- Given that N is the midpoint of CD and C's coordinates, prove the sum of D's coordinates -/
theorem sum_of_coordinates_of_D (N C : ℝ × ℝ) (h1 : N = (2, 6)) (h2 : C = (6, 2)) :
  ∃ D : ℝ × ℝ, N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) ∧ D.1 + D.2 = 8 := by
  sorry

end sum_of_coordinates_of_D_l1411_141185


namespace characterization_of_f_l1411_141120

-- Define the property of being strictly increasing
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the functional equation
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (f x + y) = f (x + y) + f 0

-- Theorem statement
theorem characterization_of_f :
  ∀ f : ℝ → ℝ, StrictlyIncreasing f → SatisfiesEquation f →
  ∃ k : ℝ, ∀ x, f x = x - k :=
sorry

end characterization_of_f_l1411_141120


namespace dana_total_earnings_l1411_141155

/-- Dana's hourly wage in dollars -/
def hourly_wage : ℕ := 13

/-- Hours worked on Friday -/
def friday_hours : ℕ := 9

/-- Hours worked on Saturday -/
def saturday_hours : ℕ := 10

/-- Hours worked on Sunday -/
def sunday_hours : ℕ := 3

/-- Calculate total earnings given hourly wage and hours worked -/
def total_earnings (wage : ℕ) (hours_fri hours_sat hours_sun : ℕ) : ℕ :=
  wage * (hours_fri + hours_sat + hours_sun)

/-- Theorem stating Dana's total earnings -/
theorem dana_total_earnings :
  total_earnings hourly_wage friday_hours saturday_hours sunday_hours = 286 := by
  sorry

end dana_total_earnings_l1411_141155


namespace triangle_properties_l1411_141147

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  (sin A) / a = (sin B) / b ∧ (sin B) / b = (sin C) / c ∧
  cos B * sin (B + π/6) = 1/2 ∧
  c / a + a / c = 4 →
  B = π/3 ∧ 1 / tan A + 1 / tan C = 2 * sqrt 3 :=
by sorry

end triangle_properties_l1411_141147


namespace letters_in_small_envelopes_l1411_141119

/-- Given the total number of letters, the number of large envelopes, and the number of letters
    per large envelope, calculate the number of letters in small envelopes. -/
theorem letters_in_small_envelopes 
  (total_letters : ℕ) 
  (large_envelopes : ℕ) 
  (letters_per_large_envelope : ℕ) 
  (h1 : total_letters = 80)
  (h2 : large_envelopes = 30)
  (h3 : letters_per_large_envelope = 2) : 
  total_letters - large_envelopes * letters_per_large_envelope = 20 :=
by sorry

end letters_in_small_envelopes_l1411_141119


namespace a_range_l1411_141181

-- Define the ⊗ operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem a_range (a : ℝ) : 
  (∀ x : ℝ, otimes (x - a) (x + 1) < 1) → -2 < a ∧ a < 2 := by
  sorry

end a_range_l1411_141181


namespace hundreds_digit_of_factorial_difference_is_zero_l1411_141140

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Theorem statement
theorem hundreds_digit_of_factorial_difference_is_zero :
  ∃ k : ℕ, factorial 25 - factorial 20 = 1000 * k :=
sorry

end hundreds_digit_of_factorial_difference_is_zero_l1411_141140


namespace parabola_equation_correct_l1411_141159

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the equation of a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the equation of a parabola in the form ax^2 + bxy + cy^2 + dx + ey + f = 0 -/
structure ParabolaEq where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- The focus of the parabola -/
def focus : Point := ⟨5, 2⟩

/-- The directrix of the parabola -/
def directrix : Line := ⟨5, 2, 25⟩

/-- The equation of the parabola -/
def parabolaEq : ParabolaEq := ⟨4, -20, 25, -40, -16, -509⟩

/-- Checks if the given parabola equation satisfies the conditions -/
def isValidParabolaEq (eq : ParabolaEq) : Prop :=
  eq.a > 0 ∧ Int.gcd eq.a.natAbs (Int.gcd eq.b.natAbs (Int.gcd eq.c.natAbs (Int.gcd eq.d.natAbs (Int.gcd eq.e.natAbs eq.f.natAbs)))) = 1

/-- Theorem stating that the given parabola equation is correct and satisfies the conditions -/
theorem parabola_equation_correct :
  isValidParabolaEq parabolaEq ∧
  ∀ (p : Point),
    (p.x - focus.x)^2 + (p.y - focus.y)^2 = 
    ((directrix.a * p.x + directrix.b * p.y - directrix.c)^2) / (directrix.a^2 + directrix.b^2) ↔
    parabolaEq.a * p.x^2 + parabolaEq.b * p.x * p.y + parabolaEq.c * p.y^2 + 
    parabolaEq.d * p.x + parabolaEq.e * p.y + parabolaEq.f = 0 := by
  sorry

end parabola_equation_correct_l1411_141159


namespace inequality_proof_l1411_141121

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end inequality_proof_l1411_141121


namespace base8_to_base6_conversion_l1411_141142

-- Define a function to convert from base 8 to base 10
def base8ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

-- Define a function to convert from base 10 to base 6
def base10ToBase6 (n : ℕ) : ℕ :=
  (n / 216) * 1000 + ((n / 36) % 6) * 100 + ((n / 6) % 6) * 10 + (n % 6)

-- Theorem statement
theorem base8_to_base6_conversion :
  base10ToBase6 (base8ToBase10 753) = 2135 := by
  sorry

end base8_to_base6_conversion_l1411_141142


namespace tan_alpha_max_value_l1411_141100

open Real

theorem tan_alpha_max_value (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) 
  (h3 : tan (α + β) = 9 * tan β) : 
  ∃ (max_tan_α : Real), max_tan_α = 4/3 ∧ ∀ (γ : Real), 
    (0 < γ ∧ γ < π/2 ∧ ∃ (δ : Real), (0 < δ ∧ δ < π/2 ∧ tan (γ + δ) = 9 * tan δ)) → 
    tan γ ≤ max_tan_α := by
sorry

end tan_alpha_max_value_l1411_141100


namespace cube_root_ratio_l1411_141123

theorem cube_root_ratio (r_old r_new : ℝ) (a_old a_new : ℝ) : 
  a_old = (2 * r_old)^3 → 
  a_new = (2 * r_new)^3 → 
  a_new = 0.125 * a_old → 
  r_new / r_old = 1/2 := by
sorry

end cube_root_ratio_l1411_141123


namespace carly_dogs_worked_on_l1411_141180

/-- The number of dogs Carly worked on given the number of nails trimmed,
    nails per paw, and number of three-legged dogs. -/
def dogs_worked_on (total_nails : ℕ) (nails_per_paw : ℕ) (three_legged_dogs : ℕ) : ℕ :=
  let total_paws := total_nails / nails_per_paw
  let three_legged_paws := three_legged_dogs * 3
  let four_legged_paws := total_paws - three_legged_paws
  let four_legged_dogs := four_legged_paws / 4
  four_legged_dogs + three_legged_dogs

theorem carly_dogs_worked_on :
  dogs_worked_on 164 4 3 = 11 := by
  sorry

end carly_dogs_worked_on_l1411_141180


namespace flower_bed_area_ratio_l1411_141113

theorem flower_bed_area_ratio :
  ∀ (l w : ℝ), l > 0 → w > 0 →
  (l * w) / ((2 * l) * (3 * w)) = 1 / 6 := by
  sorry

end flower_bed_area_ratio_l1411_141113


namespace train_length_calculation_train_length_proof_l1411_141193

/-- Calculates the length of a train given its speed, the time it takes to pass a bridge, and the length of the bridge. -/
theorem train_length_calculation (train_speed : Real) (bridge_length : Real) (time_to_pass : Real) : Real :=
  let speed_ms := train_speed * 1000 / 3600
  let total_distance := speed_ms * time_to_pass
  total_distance - bridge_length

/-- Proves that a train traveling at 45 km/hour passing a 140-meter bridge in 42 seconds has a length of 385 meters. -/
theorem train_length_proof :
  train_length_calculation 45 140 42 = 385 := by
  sorry

end train_length_calculation_train_length_proof_l1411_141193


namespace paperboy_delivery_count_l1411_141190

def delivery_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | 3 => 2
  | m + 3 => delivery_ways (m + 2) + delivery_ways (m + 1) + delivery_ways m

theorem paperboy_delivery_count :
  delivery_ways 12 = 504 :=
by sorry

end paperboy_delivery_count_l1411_141190


namespace hyperbola_eccentricity_range_l1411_141105

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- A point on the right branch of a hyperbola -/
structure RightBranchPoint (h : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2/a^2 - y^2/b^2 = 1
  right_branch : x ≥ a

/-- Distance between two points -/
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := sorry

/-- Left focus of the hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- Right focus of the hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The ratio |PF₁|²/|PF₂| for a point P on the hyperbola -/
def focal_ratio (h : Hyperbola a b) (p : RightBranchPoint h) : ℝ := sorry

/-- The minimum value of focal_ratio over all points on the right branch -/
def min_focal_ratio (h : Hyperbola a b) : ℝ := sorry

theorem hyperbola_eccentricity_range (a b : ℝ) (h : Hyperbola a b) :
  min_focal_ratio h = 8 * a → 1 < eccentricity h ∧ eccentricity h ≤ 3 := by sorry

end hyperbola_eccentricity_range_l1411_141105


namespace swimming_speed_in_still_water_l1411_141137

/-- Given a person swimming against a current, calculates their swimming speed in still water. -/
theorem swimming_speed_in_still_water 
  (current_speed : ℝ) 
  (distance_against_current : ℝ) 
  (time_against_current : ℝ) 
  (h1 : current_speed = 10)
  (h2 : distance_against_current = 8)
  (h3 : time_against_current = 4) :
  distance_against_current = (swimming_speed - current_speed) * time_against_current →
  swimming_speed = 12 :=
by
  sorry

#check swimming_speed_in_still_water

end swimming_speed_in_still_water_l1411_141137


namespace production_days_calculation_l1411_141138

theorem production_days_calculation (n : ℕ) 
  (h1 : (n * 50 : ℝ) / n = 50)  -- Average of 50 units for n days
  (h2 : ((n * 50 + 105 : ℝ) / (n + 1) = 55)) -- New average of 55 units including today
  : n = 10 := by
  sorry

end production_days_calculation_l1411_141138


namespace determinant_equation_solution_l1411_141144

/-- Definition of a second-order determinant -/
def secondOrderDet (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem stating that if |x+3 x-3; x-3 x+3| = 12, then x = 1 -/
theorem determinant_equation_solution :
  ∀ x : ℝ, secondOrderDet (x + 3) (x - 3) (x - 3) (x + 3) = 12 → x = 1 := by
  sorry

end determinant_equation_solution_l1411_141144


namespace shared_divisors_count_l1411_141145

theorem shared_divisors_count (a b : ℕ) (ha : a = 9240) (hb : b = 8820) :
  (Finset.filter (fun d : ℕ => d ∣ a ∧ d ∣ b) (Finset.range (min a b + 1))).card = 24 := by
  sorry

end shared_divisors_count_l1411_141145


namespace exists_k_for_all_n_l1411_141166

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Statement of the problem -/
theorem exists_k_for_all_n (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, k > 0 ∧ sumOfDigits k = n ∧ sumOfDigits (k^2) = n^2 := by sorry

end exists_k_for_all_n_l1411_141166


namespace correct_operation_l1411_141179

theorem correct_operation (a : ℝ) : 3 * a^3 - 2 * a^3 = a^3 := by
  sorry

end correct_operation_l1411_141179


namespace smaller_number_problem_l1411_141189

theorem smaller_number_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 45) (h4 : y = 4 * x) : x = 9 := by
  sorry

end smaller_number_problem_l1411_141189


namespace sequence_problem_l1411_141152

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d ∧ d ≠ 0

-- Define a geometric sequence
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n, b (n + 1) = r * b n

theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  (2 * a 3 - (a 7)^2 + 2 * a 11 = 0) →
  geometric_sequence b →
  b 7 = a 7 →
  b 6 * b 8 = 16 := by
sorry

end sequence_problem_l1411_141152


namespace sum_of_four_consecutive_integers_divisible_by_two_l1411_141128

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) :
  ∃ (k : ℤ), (n - 1) + n + (n + 1) + (n + 2) = 2 * k := by
  sorry

#check sum_of_four_consecutive_integers_divisible_by_two

end sum_of_four_consecutive_integers_divisible_by_two_l1411_141128


namespace least_k_for_inequality_l1411_141163

theorem least_k_for_inequality (k : ℤ) : 
  (∀ j : ℤ, j < k → (0.00010101 : ℝ) * (10 : ℝ)^j ≤ 1000) ∧ 
  ((0.00010101 : ℝ) * (10 : ℝ)^k > 1000) → 
  k = 8 :=
sorry

end least_k_for_inequality_l1411_141163


namespace binary_subtraction_theorem_l1411_141115

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of 111001₂ -/
def binary_111001 : List Bool := [true, false, false, true, true, true]

theorem binary_subtraction_theorem :
  binary_to_decimal binary_111001 - 3 = 54 := by
  sorry

end binary_subtraction_theorem_l1411_141115


namespace xiao_wang_speed_l1411_141158

/-- Represents the cycling speed of Xiao Wang in km/h -/
def cycling_speed : ℝ := 10

/-- The total distance between City A and City B in km -/
def total_distance : ℝ := 55

/-- The distance Xiao Wang cycled in km -/
def cycling_distance : ℝ := 25

/-- The time difference between cycling and bus ride in hours -/
def time_difference : ℝ := 1

theorem xiao_wang_speed :
  cycling_speed = 10 ∧
  cycling_speed > 0 ∧
  total_distance = 55 ∧
  cycling_distance = 25 ∧
  time_difference = 1 ∧
  (cycling_distance / cycling_speed) = 
    ((total_distance - cycling_distance) / (2 * cycling_speed)) + time_difference :=
by sorry

end xiao_wang_speed_l1411_141158


namespace range_of_x_when_a_is_zero_range_of_a_when_p_implies_q_l1411_141127

-- Define the conditions
def p (x : ℝ) : Prop := 4 * x^2 + 12 * x - 7 ≤ 0
def q (a x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

-- Theorem for the first question
theorem range_of_x_when_a_is_zero :
  ∀ x : ℝ, (p x ∧ ¬(q 0 x)) → (-7/2 ≤ x ∧ x < -3) :=
sorry

-- Theorem for the second question
theorem range_of_a_when_p_implies_q :
  (∀ x : ℝ, p x → q a x) → (-5/2 ≤ a ∧ a ≤ -1/2) :=
sorry

end range_of_x_when_a_is_zero_range_of_a_when_p_implies_q_l1411_141127


namespace frank_can_collection_l1411_141124

/-- Represents the number of cans in each bag for a given day -/
def BagContents := List Nat

/-- Calculates the total number of cans from a list of bag contents -/
def totalCans (bags : BagContents) : Nat :=
  bags.sum

theorem frank_can_collection :
  let saturday : BagContents := [4, 6, 5, 7, 8]
  let sunday : BagContents := [6, 5, 9]
  let monday : BagContents := [8, 8]
  totalCans saturday + totalCans sunday + totalCans monday = 66 := by
  sorry

end frank_can_collection_l1411_141124


namespace triangle_inequality_satisfied_l1411_141132

theorem triangle_inequality_satisfied (a b c : ℝ) (ha : a = 25) (hb : b = 24) (hc : c = 7) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
sorry

end triangle_inequality_satisfied_l1411_141132


namespace count_distinct_terms_l1411_141198

/-- The number of distinct terms in the expansion of (x+y+z)^2026 + (x-y-z)^2026 -/
def num_distinct_terms : ℕ := 1028196

/-- The exponent in the original expression -/
def exponent : ℕ := 2026

-- Theorem stating the number of distinct terms
theorem count_distinct_terms : 
  num_distinct_terms = (exponent / 2 + 1)^2 := by sorry

end count_distinct_terms_l1411_141198


namespace max_x_minus_y_l1411_141169

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), w = a - b ∧ a^2 + b^2 - 4*a - 2*b - 4 = 0) → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l1411_141169


namespace mean_of_xyz_l1411_141199

theorem mean_of_xyz (x y z : ℝ) 
  (eq1 : 9*x + 3*y - 5*z = -4)
  (eq2 : 5*x + 2*y - 2*z = 13) :
  (x + y + z) / 3 = 10 := by
sorry

end mean_of_xyz_l1411_141199


namespace min_presses_to_exceed_200_l1411_141129

def repeated_square (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => (repeated_square x n) ^ 2

def exceed_200 (x : ℕ) : ℕ :=
  match x with
  | 0 => 0
  | n + 1 => if repeated_square 3 n > 200 then n else exceed_200 n

theorem min_presses_to_exceed_200 : exceed_200 0 = 3 := by sorry

end min_presses_to_exceed_200_l1411_141129


namespace equation_solution_l1411_141164

theorem equation_solution (C D : ℝ) : 
  (∀ x : ℝ, x ≠ -2 ∧ x ≠ 5 → (C * x - 20) / (x^2 - 3*x - 10) = D / (x + 2) + 4 / (x - 5)) →
  C + D = 4.7 := by
  sorry

end equation_solution_l1411_141164


namespace cosine_sine_sum_equality_l1411_141154

theorem cosine_sine_sum_equality : 
  Real.cos (43 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (43 * π / 180) * Real.sin (13 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end cosine_sine_sum_equality_l1411_141154


namespace students_wanting_fruit_l1411_141156

theorem students_wanting_fruit (red_apples green_apples extra_apples : ℕ) :
  let total_apples := red_apples + green_apples
  let students := total_apples - extra_apples
  students = 10 :=
by
  sorry

end students_wanting_fruit_l1411_141156


namespace positive_X_value_l1411_141133

-- Define the * operation
def star (X Y : ℝ) : ℝ := X^3 + Y^2

-- Theorem statement
theorem positive_X_value :
  ∃ X : ℝ, X > 0 ∧ star X 4 = 280 ∧ X = 6 :=
sorry

end positive_X_value_l1411_141133


namespace intersection_point_of_g_and_inverse_l1411_141135

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 2*x^2 + 18*x + 36

-- State the theorem
theorem intersection_point_of_g_and_inverse :
  ∃! p : ℝ × ℝ, 
    (∀ x : ℝ, (x, g x) ≠ p → (g x, x) ≠ p) ∧ 
    p.1 = g p.2 ∧ 
    p.2 = g p.1 ∧
    p = (-3, -3) :=
sorry

end intersection_point_of_g_and_inverse_l1411_141135


namespace total_value_of_coins_l1411_141171

/-- Represents the value of a coin in paise -/
inductive CoinType
| OneRupee
| FiftyPaise
| TwentyFivePaise

/-- The number of coins of each type in the bag -/
def coinsPerType : ℕ := 120

/-- Converts a coin type to its value in paise -/
def coinValueInPaise (c : CoinType) : ℕ :=
  match c with
  | CoinType.OneRupee => 100
  | CoinType.FiftyPaise => 50
  | CoinType.TwentyFivePaise => 25

/-- Calculates the total value of all coins of a given type in rupees -/
def totalValueOfCoinType (c : CoinType) : ℚ :=
  (coinsPerType * coinValueInPaise c : ℚ) / 100

/-- Theorem: The total value of all coins in the bag is 210 rupees -/
theorem total_value_of_coins :
  totalValueOfCoinType CoinType.OneRupee +
  totalValueOfCoinType CoinType.FiftyPaise +
  totalValueOfCoinType CoinType.TwentyFivePaise = 210 := by
  sorry

end total_value_of_coins_l1411_141171


namespace inequality_proof_l1411_141116

theorem inequality_proof (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) > 2 := by
  sorry

end inequality_proof_l1411_141116


namespace collinear_points_imply_a_equals_four_l1411_141134

-- Define the points
def A : ℝ × ℝ := (2, 2)
def B : ℝ → ℝ × ℝ := λ a => (a, 0)
def C : ℝ × ℝ := (0, 4)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, q.1 - p.1 = t * (r.1 - p.1) ∧ q.2 - p.2 = t * (r.2 - p.2)

-- Theorem statement
theorem collinear_points_imply_a_equals_four :
  ∀ a : ℝ, collinear A (B a) C → a = 4 := by
  sorry

end collinear_points_imply_a_equals_four_l1411_141134


namespace books_per_box_l1411_141148

theorem books_per_box (total_books : ℕ) (num_boxes : ℕ) (h1 : total_books = 24) (h2 : num_boxes = 8) :
  total_books / num_boxes = 3 := by
  sorry

end books_per_box_l1411_141148


namespace fourth_term_max_implies_n_six_l1411_141108

theorem fourth_term_max_implies_n_six (n : ℕ) : 
  (∀ k : ℕ, k ≠ 3 → (n.choose k) ≤ (n.choose 3)) → n = 6 := by
  sorry

end fourth_term_max_implies_n_six_l1411_141108


namespace simplify_expression_l1411_141107

theorem simplify_expression : 2^2 + 2^2 + 2^2 + 2^2 = 2^4 := by
  sorry

end simplify_expression_l1411_141107


namespace work_completion_time_l1411_141153

/-- Given workers a, b, and c with their work rates, prove the time taken when all work together -/
theorem work_completion_time
  (total_work : ℝ)
  (time_ab : ℝ)
  (time_a : ℝ)
  (time_c : ℝ)
  (h1 : time_ab = 9)
  (h2 : time_a = 18)
  (h3 : time_c = 24)
  : (total_work / (total_work / time_ab + total_work / time_a + total_work / time_c)) = 72 / 11 := by
  sorry

#check work_completion_time

end work_completion_time_l1411_141153


namespace right_triangle_hypotenuse_l1411_141149

theorem right_triangle_hypotenuse (a b h : ℝ) : 
  a = 30 → b = 40 → h^2 = a^2 + b^2 → h = 50 := by sorry

end right_triangle_hypotenuse_l1411_141149


namespace operation_result_is_four_digit_l1411_141143

/-- A nonzero digit is a natural number between 1 and 9, inclusive. -/
def NonzeroDigit : Type :=
  { n : ℕ // 1 ≤ n ∧ n ≤ 9 }

/-- The result of the operation 543C + 721 - DE4 for any nonzero digits C, D, and E. -/
def OperationResult (C D E : NonzeroDigit) : ℕ :=
  5430 + C.val + 721 - (100 * D.val + 10 * E.val + 4)

/-- The theorem stating that the result of the operation is always a 4-digit number. -/
theorem operation_result_is_four_digit (C D E : NonzeroDigit) :
  1000 ≤ OperationResult C D E ∧ OperationResult C D E < 10000 :=
by sorry

end operation_result_is_four_digit_l1411_141143


namespace perpendicular_vectors_m_value_l1411_141103

/-- Given two perpendicular vectors a and b in ℝ², prove that m = 2 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 0) 
  (ha : a = (-2, 3)) (hb : b = (3, m)) : m = 2 := by
  sorry

end perpendicular_vectors_m_value_l1411_141103


namespace intersection_M_N_l1411_141102

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 ≠ x}

theorem intersection_M_N : M ∩ N = {-1, 1} := by
  sorry

end intersection_M_N_l1411_141102


namespace number_of_non_officers_l1411_141109

/-- Proves that the number of non-officers is 525 given the salary conditions --/
theorem number_of_non_officers (avg_salary : ℝ) (officer_salary : ℝ) (non_officer_salary : ℝ) 
  (num_officers : ℕ) (h1 : avg_salary = 120) (h2 : officer_salary = 470) 
  (h3 : non_officer_salary = 110) (h4 : num_officers = 15) : 
  ∃ (num_non_officers : ℕ), 
    (↑num_officers * officer_salary + ↑num_non_officers * non_officer_salary) / 
    (↑num_officers + ↑num_non_officers) = avg_salary ∧ num_non_officers = 525 := by
  sorry

#check number_of_non_officers

end number_of_non_officers_l1411_141109


namespace lcm_factor_proof_l1411_141182

theorem lcm_factor_proof (A B : ℕ+) (X : ℕ) : 
  Nat.gcd A B = 23 →
  Nat.lcm A B = 23 * 15 * X →
  A = 368 →
  X = 16 := by
sorry

end lcm_factor_proof_l1411_141182


namespace max_value_sine_function_l1411_141176

theorem max_value_sine_function (x : Real) (h : -π/2 ≤ x ∧ x ≤ 0) :
  ∃ y_max : Real, y_max = 2 ∧ ∀ y : Real, y = 3 * Real.sin x + 2 → y ≤ y_max :=
by sorry

end max_value_sine_function_l1411_141176


namespace team_selection_with_girls_l1411_141114

theorem team_selection_with_girls (boys girls team_size min_girls : ℕ) 
  (h_boys : boys = 10)
  (h_girls : girls = 12)
  (h_team_size : team_size = 6)
  (h_min_girls : min_girls = 2) : 
  (Finset.range (team_size - min_girls + 1)).sum (λ i => 
    Nat.choose girls (i + min_girls) * Nat.choose boys (team_size - (i + min_girls))) = 71379 := by
  sorry

end team_selection_with_girls_l1411_141114


namespace geometric_sequence_sum_ratio_l1411_141177

/-- Given a geometric sequence with common ratio q ≠ 1, if the ratio of the sum of the first 10 terms
    to the sum of the first 5 terms is 1:2, then the ratio of the sum of the first 15 terms to the
    sum of the first 5 terms is 3:4. -/
theorem geometric_sequence_sum_ratio (q : ℝ) (a : ℕ → ℝ) (h1 : q ≠ 1) 
  (h2 : ∀ n, a (n + 1) = q * a n) 
  (h3 : (1 - q^10) / (1 - q^5) = 1 / 2) :
  (1 - q^15) / (1 - q^5) = 3 / 4 := by
  sorry


end geometric_sequence_sum_ratio_l1411_141177


namespace marble_probability_l1411_141191

/-- Represents a box of marbles -/
structure MarbleBox where
  total : ℕ
  black : ℕ
  white : ℕ
  valid : black + white = total

/-- Represents two boxes of marbles -/
structure TwoBoxes where
  box1 : MarbleBox
  box2 : MarbleBox
  total36 : box1.total + box2.total = 36

/-- The probability of drawing a black marble from a box -/
def probBlack (box : MarbleBox) : ℚ :=
  box.black / box.total

/-- The probability of drawing a white marble from a box -/
def probWhite (box : MarbleBox) : ℚ :=
  box.white / box.total

theorem marble_probability (boxes : TwoBoxes)
    (h : probBlack boxes.box1 * probBlack boxes.box2 = 13/18) :
    probWhite boxes.box1 * probWhite boxes.box2 = 1/9 := by
  sorry

end marble_probability_l1411_141191


namespace candy_bar_profit_l1411_141183

/-- Represents the candy bar sale problem -/
structure CandyBarSale where
  total_bars : ℕ
  bulk_price : ℚ
  bulk_quantity : ℕ
  regular_price : ℚ
  regular_quantity : ℕ
  selling_price : ℚ
  selling_quantity : ℕ

/-- Calculates the profit for the candy bar sale -/
def calculate_profit (sale : CandyBarSale) : ℚ :=
  let cost_per_bar := sale.bulk_price / sale.bulk_quantity
  let total_cost := cost_per_bar * sale.total_bars
  let revenue_per_bar := sale.selling_price / sale.selling_quantity
  let total_revenue := revenue_per_bar * sale.total_bars
  total_revenue - total_cost

/-- The main theorem stating that the profit is $350 -/
theorem candy_bar_profit :
  let sale : CandyBarSale := {
    total_bars := 1200,
    bulk_price := 3,
    bulk_quantity := 8,
    regular_price := 2,
    regular_quantity := 5,
    selling_price := 2,
    selling_quantity := 3
  }
  calculate_profit sale = 350 := by
  sorry


end candy_bar_profit_l1411_141183


namespace measure_when_unit_changed_l1411_141162

-- Define segments a and b as positive real numbers
variable (a b : ℝ) (ha : 0 < a) (hb : 0 < b)

-- Define m as the measure of a when b is the unit length
variable (m : ℝ) (hm : a = m * b)

-- Theorem statement
theorem measure_when_unit_changed : 
  (b / a : ℝ) = 1 / m :=
sorry

end measure_when_unit_changed_l1411_141162


namespace coefficient_x_squared_in_product_l1411_141172

theorem coefficient_x_squared_in_product : 
  let p₁ : Polynomial ℤ := 2 * X^3 - 4 * X^2 + 3 * X + 2
  let p₂ : Polynomial ℤ := -X^2 + 3 * X - 5
  (p₁ * p₂).coeff 2 = 7 := by
  sorry

end coefficient_x_squared_in_product_l1411_141172


namespace rectangle_dimension_change_l1411_141186

theorem rectangle_dimension_change (L B : ℝ) (h₁ : L > 0) (h₂ : B > 0) : 
  let new_B := 1.25 * B
  let new_area := 1.375 * (L * B)
  ∃ x : ℝ, x = 10 ∧ new_area = (L * (1 + x / 100)) * new_B := by
sorry

end rectangle_dimension_change_l1411_141186


namespace john_chore_time_l1411_141106

/-- Given a ratio of cartoon watching time to chore time and the total cartoon watching time,
    calculate the required chore time. -/
def chore_time (cartoon_ratio : ℕ) (chore_ratio : ℕ) (total_cartoon_time : ℕ) : ℕ :=
  (chore_ratio * total_cartoon_time) / cartoon_ratio

theorem john_chore_time :
  let cartoon_ratio : ℕ := 10
  let chore_ratio : ℕ := 8
  let total_cartoon_time : ℕ := 120
  chore_time cartoon_ratio chore_ratio total_cartoon_time = 96 := by
  sorry

#eval chore_time 10 8 120

end john_chore_time_l1411_141106


namespace geometric_progression_existence_l1411_141101

theorem geometric_progression_existence : ∃ (a : ℕ → ℚ), 
  (∀ n, a (n + 1) = a n * (3/2)) ∧ 
  (a 1 = 2^99) ∧
  (∀ n, a (n + 1) > a n) ∧
  (∀ n ≤ 100, ∃ m : ℕ, a n = m) ∧
  (∀ n > 100, ∀ m : ℕ, a n ≠ m) := by
  sorry

#check geometric_progression_existence

end geometric_progression_existence_l1411_141101


namespace cheryl_tournament_cost_l1411_141167

/-- Calculates the total amount Cheryl pays for a golf tournament given her expenses -/
def tournament_cost (electricity_bill : ℕ) (phone_bill_difference : ℕ) (tournament_percentage : ℕ) : ℕ :=
  let phone_bill := electricity_bill + phone_bill_difference
  let tournament_additional_cost := phone_bill * tournament_percentage / 100
  phone_bill + tournament_additional_cost

/-- Proves that Cheryl pays $1440 for the golf tournament given the specified conditions -/
theorem cheryl_tournament_cost :
  tournament_cost 800 400 20 = 1440 := by
  sorry

end cheryl_tournament_cost_l1411_141167


namespace fitness_club_comparison_l1411_141118

/-- Represents a fitness club with a monthly subscription cost -/
structure FitnessClub where
  name : String
  monthlyCost : ℕ

/-- Calculates the yearly cost for a given number of months -/
def yearlyCost (club : FitnessClub) (months : ℕ) : ℕ :=
  club.monthlyCost * months

/-- Calculates the cost per visit given total cost and number of visits -/
def costPerVisit (totalCost : ℕ) (visits : ℕ) : ℚ :=
  totalCost / visits

/-- Represents the two attendance patterns -/
inductive AttendancePattern
  | Regular
  | MoodBased

/-- Calculates the number of visits per year based on the attendance pattern -/
def visitsPerYear (pattern : AttendancePattern) : ℕ :=
  match pattern with
  | .Regular => 96
  | .MoodBased => 56

theorem fitness_club_comparison (alpha beta : FitnessClub) 
    (h_alpha : alpha.monthlyCost = 999)
    (h_beta : beta.monthlyCost = 1299) :
    (∀ (pattern : AttendancePattern), 
      costPerVisit (yearlyCost alpha 12) (visitsPerYear pattern) < 
      costPerVisit (yearlyCost beta 12) (visitsPerYear pattern)) ∧
    (costPerVisit (yearlyCost alpha 12) (visitsPerYear .MoodBased) > 
     costPerVisit (yearlyCost beta 8) (visitsPerYear .MoodBased)) := by
  sorry

#check fitness_club_comparison

end fitness_club_comparison_l1411_141118


namespace single_elimination_games_tournament_with_23_teams_l1411_141157

/-- Represents a single-elimination tournament. -/
structure SingleEliminationTournament where
  num_teams : ℕ
  num_games : ℕ

/-- The number of games in a single-elimination tournament is one less than the number of teams. -/
theorem single_elimination_games (t : SingleEliminationTournament) 
  (h : t.num_teams > 0) : t.num_games = t.num_teams - 1 := by
  sorry

/-- In a single-elimination tournament with 23 teams, 22 games are required to declare a winner. -/
theorem tournament_with_23_teams : 
  ∃ t : SingleEliminationTournament, t.num_teams = 23 ∧ t.num_games = 22 := by
  sorry

end single_elimination_games_tournament_with_23_teams_l1411_141157


namespace variance_invariant_under_translation_regression_line_minimizes_squared_residuals_sum_residuals_zero_l1411_141122

-- Define a data set as a list of real numbers
def DataSet := List ℝ

-- Define the sample variance
def sampleVariance (data : DataSet) : ℝ := sorry

-- Define a function to subtract a constant from each data point
def subtractConstant (data : DataSet) (c : ℝ) : DataSet := sorry

-- Define a type for a regression line
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

-- Define a function to calculate residuals
def residuals (data : DataSet) (line : RegressionLine) : DataSet := sorry

-- Define a function to calculate the sum of squared residuals
def sumSquaredResiduals (data : DataSet) (line : RegressionLine) : ℝ := sorry

-- Define a function to find the least squares regression line
def leastSquaresRegressionLine (data : DataSet) : RegressionLine := sorry

-- Theorem 1: Subtracting a constant doesn't change the sample variance
theorem variance_invariant_under_translation (data : DataSet) (c : ℝ) :
  sampleVariance (subtractConstant data c) = sampleVariance data := by sorry

-- Theorem 2: The regression line minimizes the sum of squared residuals
theorem regression_line_minimizes_squared_residuals (data : DataSet) :
  ∀ line : RegressionLine,
    sumSquaredResiduals data (leastSquaresRegressionLine data) ≤ sumSquaredResiduals data line := by sorry

-- Theorem 3: The sum of residuals for the least squares regression line is zero
theorem sum_residuals_zero (data : DataSet) :
  (residuals data (leastSquaresRegressionLine data)).sum = 0 := by sorry

end variance_invariant_under_translation_regression_line_minimizes_squared_residuals_sum_residuals_zero_l1411_141122


namespace imaginary_part_of_complex_fraction_l1411_141165

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (2 + i) / i
  Complex.im z = -2 := by
  sorry

end imaginary_part_of_complex_fraction_l1411_141165


namespace cost_price_calculation_l1411_141197

theorem cost_price_calculation (cost_price : ℝ) : 
  cost_price * (1 + 0.2) * 0.9 - cost_price = 8 → cost_price = 100 := by
  sorry

end cost_price_calculation_l1411_141197


namespace grape_heap_division_l1411_141161

theorem grape_heap_division (n : ℕ) (h1 : n ≥ 105) 
  (h2 : (n + 1) % 3 = 1) (h3 : (n + 1) % 5 = 1) :
  ∃ x : ℕ, x > 5 ∧ (n + 1) % x = 1 ∧ ∀ y : ℕ, 5 < y ∧ y < x → (n + 1) % y ≠ 1 :=
by
  sorry

end grape_heap_division_l1411_141161


namespace optimal_distance_optimal_distance_with_discount_l1411_141170

/-- Represents the optimal store distance problem --/
structure OptimalStoreDistance where
  s₀ : ℝ  -- Distance from home to city center
  v : ℝ   -- Base utility value

/-- Calculates the price at a given distance --/
def price (s_m : ℝ) : ℝ :=
  1000 * (1 - 0.02 * s_m)

/-- Calculates the transportation cost --/
def transportCost (s₀ s_m : ℝ) : ℝ :=
  0.5 * (s_m - s₀)^2

/-- Calculates the utility without discount --/
def utility (osd : OptimalStoreDistance) (s_m : ℝ) : ℝ :=
  osd.v - price s_m - transportCost osd.s₀ s_m

/-- Calculates the utility with discount --/
def utilityWithDiscount (osd : OptimalStoreDistance) (s_m : ℝ) : ℝ :=
  osd.v - 0.9 * price s_m - transportCost osd.s₀ s_m

/-- Theorem: Optimal store distance without discount --/
theorem optimal_distance (osd : OptimalStoreDistance) :
  ∃ s_m : ℝ, s_m = min 60 (osd.s₀ + 20) ∧
  ∀ s : ℝ, 0 ≤ s ∧ s ≤ 60 → utility osd s_m ≥ utility osd s :=
sorry

/-- Theorem: Optimal store distance with discount --/
theorem optimal_distance_with_discount (osd : OptimalStoreDistance) :
  ∃ s_m : ℝ, s_m = min 60 (osd.s₀ + 9) ∧
  ∀ s : ℝ, 0 ≤ s ∧ s ≤ 60 → utilityWithDiscount osd s_m ≥ utilityWithDiscount osd s :=
sorry

end optimal_distance_optimal_distance_with_discount_l1411_141170


namespace initial_gum_pieces_l1411_141146

theorem initial_gum_pieces (x : ℕ) : x + 16 + 20 = 61 → x = 25 := by
  sorry

end initial_gum_pieces_l1411_141146


namespace will_toy_purchase_l1411_141178

theorem will_toy_purchase (initial_amount : ℕ) (spent_amount : ℕ) (toy_cost : ℕ) : 
  initial_amount = 83 → spent_amount = 47 → toy_cost = 4 →
  (initial_amount - spent_amount) / toy_cost = 9 := by
sorry

end will_toy_purchase_l1411_141178


namespace circle_ratio_l1411_141130

theorem circle_ratio (a b : ℝ) (h : a > 0) (k : b > 0) 
  (h1 : π * b^2 - π * a^2 = 4 * (π * a^2)) : a / b = Real.sqrt 5 / 5 := by
  sorry

end circle_ratio_l1411_141130


namespace fibonacci_like_sequence_a8_l1411_141136

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) + a n

theorem fibonacci_like_sequence_a8 (a : ℕ → ℕ) :
  fibonacci_like_sequence a →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) > a n) →
  (∀ n : ℕ, a n > 0) →
  a 7 = 240 →
  a 8 = 386 := by
  sorry

end fibonacci_like_sequence_a8_l1411_141136


namespace f_derivative_and_tangent_line_l1411_141188

noncomputable def f (x : ℝ) : ℝ := Real.sin x / x

theorem f_derivative_and_tangent_line :
  (∃ (f' : ℝ → ℝ), ∀ x, x ≠ 0 → HasDerivAt f (f' x) x) ∧
  (∀ x, x ≠ 0 → (deriv f) x = (x * Real.cos x - Real.sin x) / x^2) ∧
  (HasDerivAt f (-1/π) π) ∧
  (∀ x, -x/π + 1 = (-1/π) * (x - π)) := by
  sorry

end f_derivative_and_tangent_line_l1411_141188


namespace transport_cost_tripled_bags_reduced_weight_l1411_141195

/-- The cost of transporting cement bags -/
def transport_cost (bags : ℕ) (weight : ℚ) : ℚ :=
  (6000 : ℚ) * bags * weight / (80 * 50)

/-- Theorem: The cost of transporting 240 bags weighing 30 kgs each is $10800 -/
theorem transport_cost_tripled_bags_reduced_weight :
  transport_cost 240 30 = 10800 := by
  sorry

end transport_cost_tripled_bags_reduced_weight_l1411_141195


namespace three_rulers_left_l1411_141112

/-- The number of rulers left in a drawer after some are removed -/
def rulers_left (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

/-- Theorem stating that 3 rulers are left in the drawer -/
theorem three_rulers_left : rulers_left 14 11 = 3 := by
  sorry

end three_rulers_left_l1411_141112


namespace tank_water_calculation_l1411_141125

theorem tank_water_calculation : 
  let tank1_capacity : ℚ := 7000
  let tank2_capacity : ℚ := 5000
  let tank3_capacity : ℚ := 3000
  let tank1_fill_ratio : ℚ := 3/4
  let tank2_fill_ratio : ℚ := 4/5
  let tank3_fill_ratio : ℚ := 1/2
  let total_water : ℚ := tank1_capacity * tank1_fill_ratio + 
                         tank2_capacity * tank2_fill_ratio + 
                         tank3_capacity * tank3_fill_ratio
  total_water = 10750 := by
sorry

end tank_water_calculation_l1411_141125


namespace sarahs_laundry_l1411_141110

theorem sarahs_laundry (machine_capacity : ℕ) (sweaters : ℕ) (loads : ℕ) (shirts : ℕ) : 
  machine_capacity = 5 →
  sweaters = 2 →
  loads = 9 →
  shirts = loads * machine_capacity - sweaters →
  shirts = 43 := by
sorry

end sarahs_laundry_l1411_141110


namespace performances_distribution_l1411_141174

/-- The number of ways to distribute performances among classes -/
def distribute_performances (total : ℕ) (num_classes : ℕ) (min_per_class : ℕ) : ℕ :=
  Nat.choose (total - num_classes * min_per_class + num_classes - 1) (num_classes - 1)

/-- Theorem stating the number of ways to distribute 14 performances among 3 classes -/
theorem performances_distribution :
  distribute_performances 14 3 3 = 21 := by sorry

end performances_distribution_l1411_141174


namespace race_probability_l1411_141126

theorem race_probability (p_x p_y p_z : ℝ) : 
  p_x = 1/7 →
  p_y = 1/3 →
  p_x + p_y + p_z = 0.6761904761904762 →
  p_z = 0.2 := by
sorry

end race_probability_l1411_141126


namespace topsoil_cost_for_seven_cubic_yards_l1411_141192

/-- The cost of topsoil in dollars per cubic foot -/
def topsoil_cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil -/
def cubic_yards_of_topsoil : ℝ := 7

/-- The cost of topsoil for a given number of cubic yards -/
def topsoil_cost (cubic_yards : ℝ) : ℝ :=
  cubic_yards * cubic_feet_per_cubic_yard * topsoil_cost_per_cubic_foot

theorem topsoil_cost_for_seven_cubic_yards :
  topsoil_cost cubic_yards_of_topsoil = 1512 := by
  sorry

end topsoil_cost_for_seven_cubic_yards_l1411_141192


namespace min_sum_a1_a2_l1411_141104

/-- The sequence (aᵢ) is defined by aₙ₊₂ = (aₙ + 3007) / (1 + aₙ₊₁) for n ≥ 1, where all aᵢ are positive integers. -/
def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, a (n + 2) = (a n + 3007) / (1 + a (n + 1))

/-- The minimum possible value of a₁ + a₂ is 114. -/
theorem min_sum_a1_a2 :
  ∀ a : ℕ → ℕ, is_valid_sequence a → a 1 + a 2 ≥ 114 :=
by sorry

end min_sum_a1_a2_l1411_141104


namespace sin_2theta_from_exp_l1411_141141

theorem sin_2theta_from_exp (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 5) : 
  Real.sin (2 * θ) = 12 * Real.sqrt 2 / 25 := by
sorry

end sin_2theta_from_exp_l1411_141141


namespace work_isothermal_expansion_l1411_141111

/-- Work done during isothermal expansion of an ideal gas -/
theorem work_isothermal_expansion 
  (m μ R T V₁ V₂ : ℝ) 
  (hm : m > 0) 
  (hμ : μ > 0) 
  (hR : R > 0) 
  (hT : T > 0) 
  (hV₁ : V₁ > 0) 
  (hV₂ : V₂ > 0) 
  (hexpand : V₂ > V₁) :
  ∃ A : ℝ, A = (m / μ) * R * T * Real.log (V₂ / V₁) ∧
  (∀ V : ℝ, V > 0 → (m / μ) * R * T = V * (m / μ) * R * T / V) :=
sorry

end work_isothermal_expansion_l1411_141111


namespace boots_cost_ratio_l1411_141139

theorem boots_cost_ratio (initial_amount : ℚ) (toilet_paper_cost : ℚ) (additional_money : ℚ) :
  initial_amount = 50 →
  toilet_paper_cost = 12 →
  additional_money = 35 →
  let remaining_after_toilet_paper := initial_amount - toilet_paper_cost
  let groceries_cost := 2 * toilet_paper_cost
  let remaining_after_groceries := remaining_after_toilet_paper - groceries_cost
  let total_boot_cost := remaining_after_groceries + 2 * additional_money
  let single_boot_cost := total_boot_cost / 2
  (single_boot_cost / remaining_after_groceries : ℚ) = 3 := by
sorry

end boots_cost_ratio_l1411_141139


namespace bowTie_equation_solution_l1411_141187

/-- The infinite nested radical operation -/
noncomputable def bowTie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

/-- Theorem stating that if 5 ⋈ z = 12, then z = 42 -/
theorem bowTie_equation_solution :
  ∃ z : ℝ, bowTie 5 z = 12 → z = 42 := by
  sorry

end bowTie_equation_solution_l1411_141187


namespace complex_number_in_first_quadrant_l1411_141173

theorem complex_number_in_first_quadrant (z : ℂ) : z = 2 + I → z.re > 0 ∧ z.im > 0 := by
  sorry

end complex_number_in_first_quadrant_l1411_141173


namespace xiaoyings_journey_equations_correct_l1411_141160

/-- Represents a journey with uphill and downhill sections -/
structure Journey where
  total_distance : ℝ
  total_time : ℝ
  uphill_speed : ℝ
  downhill_speed : ℝ

/-- Xiaoying's journey to school -/
def xiaoyings_journey : Journey where
  total_distance := 1.2  -- 1200 meters converted to kilometers
  total_time := 16
  uphill_speed := 3
  downhill_speed := 5

/-- The system of equations representing Xiaoying's journey -/
def journey_equations (j : Journey) (x y : ℝ) : Prop :=
  (j.uphill_speed / 60 * x + j.downhill_speed / 60 * y = j.total_distance) ∧
  (x + y = j.total_time)

theorem xiaoyings_journey_equations_correct :
  journey_equations xiaoyings_journey = λ x y ↦ 
    (3 / 60 * x + 5 / 60 * y = 1.2) ∧ (x + y = 16) := by sorry

end xiaoyings_journey_equations_correct_l1411_141160


namespace similar_triangles_perimeter_l1411_141131

theorem similar_triangles_perimeter (h_small h_large p_small : ℝ) : 
  h_small / h_large = 3 / 5 →
  p_small = 12 →
  ∃ p_large : ℝ, p_large = 20 ∧ p_small / p_large = h_small / h_large :=
by sorry

end similar_triangles_perimeter_l1411_141131


namespace firework_explosion_velocity_firework_explosion_velocity_is_correct_l1411_141175

/-- The magnitude of the second fragment's velocity after a firework explosion -/
theorem firework_explosion_velocity : ℝ :=
  let initial_velocity : ℝ := 20
  let gravity : ℝ := 10
  let explosion_time : ℝ := 1
  let mass_ratio : ℝ := 2
  let small_fragment_horizontal_velocity : ℝ := 16

  let velocity_at_explosion : ℝ := initial_velocity - gravity * explosion_time
  let small_fragment_mass : ℝ := 1
  let large_fragment_mass : ℝ := mass_ratio * small_fragment_mass

  let small_fragment_vertical_velocity : ℝ := velocity_at_explosion
  let large_fragment_horizontal_velocity : ℝ := 
    -(small_fragment_mass * small_fragment_horizontal_velocity) / large_fragment_mass
  let large_fragment_vertical_velocity : ℝ := velocity_at_explosion

  let large_fragment_velocity_magnitude : ℝ := 
    Real.sqrt (large_fragment_horizontal_velocity^2 + large_fragment_vertical_velocity^2)

  2 * Real.sqrt 41

theorem firework_explosion_velocity_is_correct : 
  firework_explosion_velocity = 2 * Real.sqrt 41 := by
  sorry

end firework_explosion_velocity_firework_explosion_velocity_is_correct_l1411_141175


namespace vector_expression_equality_l1411_141117

theorem vector_expression_equality : 
  let v1 : Fin 2 → ℝ := ![3, -4]
  let v2 : Fin 2 → ℝ := ![2, -3]
  let v3 : Fin 2 → ℝ := ![1, 6]
  v1 + 5 • v2 - v3 = ![12, -25] := by sorry

end vector_expression_equality_l1411_141117


namespace pizza_buffet_l1411_141151

theorem pizza_buffet (A B C : ℕ) (h1 : ∃ x : ℕ, A = x * B) 
  (h2 : B * 8 = C) (h3 : A + B + C = 360) : 
  ∃ x : ℕ, A = 351 * B := by
  sorry

end pizza_buffet_l1411_141151


namespace value_of_x_l1411_141150

theorem value_of_x (x y z : ℚ) : 
  x = (1 / 3) * y → 
  y = (1 / 4) * z → 
  z = 96 → 
  x = 8 := by sorry

end value_of_x_l1411_141150


namespace max_snack_bags_l1411_141168

def granola_bars : ℕ := 24
def dried_fruit : ℕ := 36
def nuts : ℕ := 60

theorem max_snack_bags : 
  ∃ (n : ℕ), n > 0 ∧ 
  granola_bars % n = 0 ∧ 
  dried_fruit % n = 0 ∧ 
  nuts % n = 0 ∧
  ∀ (m : ℕ), m > n → 
    (granola_bars % m = 0 ∧ dried_fruit % m = 0 ∧ nuts % m = 0) → False :=
by sorry

end max_snack_bags_l1411_141168
