import Mathlib

namespace NUMINAMATH_CALUDE_largest_integral_y_l161_16173

theorem largest_integral_y : ∃ y : ℤ, y = 4 ∧ 
  (∀ z : ℤ, (1/4 : ℚ) < (z : ℚ)/7 ∧ (z : ℚ)/7 < 7/11 → z ≤ y) ∧
  (1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 7/11 :=
by sorry

end NUMINAMATH_CALUDE_largest_integral_y_l161_16173


namespace NUMINAMATH_CALUDE_fraction_problem_l161_16155

theorem fraction_problem (x y : ℚ) (h1 : x + y = 14/15) (h2 : x * y = 1/10) :
  min x y = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l161_16155


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_l161_16145

theorem x_squared_plus_y_squared (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + x + y = 99) : 
  x^2 + y^2 = 5745 / 169 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_l161_16145


namespace NUMINAMATH_CALUDE_polar_equations_and_intersection_range_l161_16174

-- Define the line l
def line_l (x : ℝ) : Prop := x = 2

-- Define the curve C
def curve_C (x y α : ℝ) : Prop := x = Real.cos α ∧ y = 1 + Real.sin α

-- Define the polar coordinates
def polar_coords (x y ρ θ : ℝ) : Prop := x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- State the theorem
theorem polar_equations_and_intersection_range :
  ∀ (x y ρ θ α β : ℝ),
  (0 < β ∧ β < Real.pi / 2) →
  (line_l x →
    ∃ (ρ_l : ℝ), polar_coords x y ρ_l θ ∧ ρ_l * Real.cos θ = 2) ∧
  (curve_C x y α →
    ∃ (ρ_c : ℝ), polar_coords x y ρ_c θ ∧ ρ_c = 2 * Real.sin θ) ∧
  (∃ (ρ_p ρ_m : ℝ),
    polar_coords x y ρ_p β ∧
    curve_C x y α ∧
    polar_coords x y ρ_m β ∧
    line_l x ∧
    0 < ρ_p / ρ_m ∧ ρ_p / ρ_m ≤ 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_polar_equations_and_intersection_range_l161_16174


namespace NUMINAMATH_CALUDE_father_walking_time_l161_16127

/-- The time (in minutes) it takes Xiaoming to cycle from the meeting point to B -/
def meeting_to_B : ℝ := 18

/-- Xiaoming's cycling speed is 4 times his father's walking speed -/
def speed_ratio : ℝ := 4

/-- The time (in minutes) it takes Xiaoming's father to walk from the meeting point to A -/
def father_time : ℝ := 288

theorem father_walking_time :
  ∀ (xiaoming_speed father_speed : ℝ),
  xiaoming_speed > 0 ∧ father_speed > 0 →
  xiaoming_speed = speed_ratio * father_speed →
  father_time = 4 * (speed_ratio * meeting_to_B) := by
  sorry

end NUMINAMATH_CALUDE_father_walking_time_l161_16127


namespace NUMINAMATH_CALUDE_quadratic_equation_always_has_real_root_l161_16142

theorem quadratic_equation_always_has_real_root (a : ℝ) : ∃ x : ℝ, a * x^2 - x = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_always_has_real_root_l161_16142


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_z_times_i_real_implies_modulus_l161_16147

-- Define the complex number z as a function of k
def z (k : ℝ) : ℂ := (k^2 - 3*k - 4 : ℝ) + (k - 1 : ℝ) * Complex.I

-- Theorem for the first part of the problem
theorem z_in_second_quadrant (k : ℝ) :
  (z k).re < 0 ∧ (z k).im > 0 ↔ 1 < k ∧ k < 4 := by sorry

-- Theorem for the second part of the problem
theorem z_times_i_real_implies_modulus (k : ℝ) :
  (z k * Complex.I).im = 0 → Complex.abs (z k) = 2 ∨ Complex.abs (z k) = 3 := by sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_z_times_i_real_implies_modulus_l161_16147


namespace NUMINAMATH_CALUDE_max_abs_f_implies_sum_l161_16164

def f (a b x : ℝ) := x^2 + a*x + b

theorem max_abs_f_implies_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |f a b x| ≤ (1/2 : ℝ)) →
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, |f a b x| = (1/2 : ℝ)) →
  4*a + 3*b = -(3/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_max_abs_f_implies_sum_l161_16164


namespace NUMINAMATH_CALUDE_remaining_amount_for_seat_and_tape_l161_16179

def initial_amount : ℕ := 60
def frame_cost : ℕ := 15
def wheel_cost : ℕ := 25

theorem remaining_amount_for_seat_and_tape : 
  initial_amount - (frame_cost + wheel_cost) = 20 := by
sorry

end NUMINAMATH_CALUDE_remaining_amount_for_seat_and_tape_l161_16179


namespace NUMINAMATH_CALUDE_intersection_sum_l161_16166

theorem intersection_sum (n c : ℝ) : 
  (∀ x y : ℝ, y = n * x + 3 → y = 4 * x + c → x = 4 ∧ y = 7) → 
  c + n = -8 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l161_16166


namespace NUMINAMATH_CALUDE_f_composition_equals_one_fourth_l161_16110

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 2^x

theorem f_composition_equals_one_fourth :
  f (f (1/9)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_one_fourth_l161_16110


namespace NUMINAMATH_CALUDE_S_31_composite_bound_l161_16198

def S (k : ℕ+) (n : ℕ) : ℕ :=
  (n.digits k.val).sum

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

theorem S_31_composite_bound :
  ∃ (A : Finset ℕ), A.card ≤ 2 ∧
    ∀ p : ℕ, is_prime p → p < 20000 →
      is_composite (S 31 p) → S 31 p ∈ A :=
sorry

end NUMINAMATH_CALUDE_S_31_composite_bound_l161_16198


namespace NUMINAMATH_CALUDE_max_M_is_8_l161_16160

/-- The number of factors of 2 in the prime factorization of a natural number -/
noncomputable def factorsOfTwo (n : ℕ) : ℕ := sorry

/-- J_k is defined as 10^(k+3) + 256 -/
def J (k : ℕ) : ℕ := 10^(k+3) + 256

/-- M(k) is the number of factors of 2 in the prime factorization of J_k -/
noncomputable def M (k : ℕ) : ℕ := factorsOfTwo (J k)

/-- The maximum value of M(k) for k > 0 is 8 -/
theorem max_M_is_8 : ∀ k : ℕ, k > 0 → M k ≤ 8 := by sorry

end NUMINAMATH_CALUDE_max_M_is_8_l161_16160


namespace NUMINAMATH_CALUDE_finite_divisor_property_l161_16150

/-- A number is a finite decimal if it can be expressed as a/b where b is of the form 2^u * 5^v -/
def IsFiniteDecimal (q : ℚ) : Prop :=
  ∃ (a b : ℕ) (u v : ℕ), q = a / b ∧ b = 2^u * 5^v

/-- A natural number n has the property that all its divisors result in finite decimals -/
def HasFiniteDivisors (n : ℕ) : Prop :=
  ∀ k : ℕ, 0 < k → k < n → IsFiniteDecimal (n / k)

/-- The theorem stating that only 2, 3, and 6 have the finite divisor property -/
theorem finite_divisor_property :
  ∀ n : ℕ, HasFiniteDivisors n ↔ n = 2 ∨ n = 3 ∨ n = 6 :=
sorry

end NUMINAMATH_CALUDE_finite_divisor_property_l161_16150


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cosines_l161_16197

theorem arithmetic_sequence_cosines (a : ℝ) : 
  (0 < a) ∧ (a < 2 * Real.pi) ∧ 
  (∃ d : ℝ, (Real.cos (2 * a) = Real.cos a + d) ∧ 
            (Real.cos (3 * a) = Real.cos (2 * a) + d)) ↔ 
  (a = Real.pi / 4) ∨ (a = 3 * Real.pi / 4) ∨ 
  (a = 5 * Real.pi / 4) ∨ (a = 7 * Real.pi / 4) :=
by sorry

#check arithmetic_sequence_cosines

end NUMINAMATH_CALUDE_arithmetic_sequence_cosines_l161_16197


namespace NUMINAMATH_CALUDE_angle_range_given_sine_l161_16132

theorem angle_range_given_sine (α : Real) (h1 : 0 < α ∧ α < Real.pi / 2) (h2 : Real.sin α = 0.58) :
  Real.pi / 6 < α ∧ α < Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_range_given_sine_l161_16132


namespace NUMINAMATH_CALUDE_total_annual_insurance_cost_l161_16158

def car_insurance_quarterly : ℕ := 378
def home_insurance_monthly : ℕ := 125
def health_insurance_annual : ℕ := 5045

theorem total_annual_insurance_cost :
  car_insurance_quarterly * 4 + home_insurance_monthly * 12 + health_insurance_annual = 8057 := by
  sorry

end NUMINAMATH_CALUDE_total_annual_insurance_cost_l161_16158


namespace NUMINAMATH_CALUDE_denver_temperature_peak_l161_16103

/-- The temperature function modeling a day in Denver, CO -/
def temperature (t : ℝ) : ℝ := -2 * t^2 + 24 * t + 100

/-- Theorem stating that 6 is the smallest non-negative real solution to the temperature equation -/
theorem denver_temperature_peak :
  (∀ t : ℝ, t ≥ 0 → temperature t = 148 → t ≥ 6) ∧
  temperature 6 = 148 := by
  sorry

end NUMINAMATH_CALUDE_denver_temperature_peak_l161_16103


namespace NUMINAMATH_CALUDE_playground_area_l161_16135

/-- Given a rectangular playground with perimeter 90 feet and length three times the width,
    prove that its area is 380.625 square feet. -/
theorem playground_area (w : ℝ) (l : ℝ) :
  (2 * l + 2 * w = 90) →  -- Perimeter is 90 feet
  (l = 3 * w) →           -- Length is three times the width
  (l * w = 380.625) :=    -- Area is 380.625 square feet
by sorry

end NUMINAMATH_CALUDE_playground_area_l161_16135


namespace NUMINAMATH_CALUDE_hyperbola_cosh_sinh_l161_16106

theorem hyperbola_cosh_sinh (t : ℝ) : (Real.cosh t)^2 - (Real.sinh t)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_cosh_sinh_l161_16106


namespace NUMINAMATH_CALUDE_compound_proposition_truth_l161_16165

theorem compound_proposition_truth : 
  (∀ x : ℝ, x < 0 → (2 : ℝ)^x > (3 : ℝ)^x) ∧ 
  (∃ x : ℝ, x > 0 ∧ Real.sqrt x > x^3) := by
sorry

end NUMINAMATH_CALUDE_compound_proposition_truth_l161_16165


namespace NUMINAMATH_CALUDE_broken_beads_count_l161_16119

/-- Calculates the number of necklaces with broken beads -/
def necklaces_with_broken_beads (initial_count : ℕ) (purchased : ℕ) (gifted : ℕ) (final_count : ℕ) : ℕ :=
  initial_count + purchased - gifted - final_count

theorem broken_beads_count :
  necklaces_with_broken_beads 50 5 15 37 = 3 := by
  sorry

end NUMINAMATH_CALUDE_broken_beads_count_l161_16119


namespace NUMINAMATH_CALUDE_distance_driven_l161_16169

/-- Represents the efficiency of a car in kilometers per gallon -/
def car_efficiency : ℝ := 10

/-- Represents the amount of gas available in gallons -/
def gas_available : ℝ := 10

/-- Theorem stating the distance that can be driven given the car's efficiency and available gas -/
theorem distance_driven : car_efficiency * gas_available = 100 := by sorry

end NUMINAMATH_CALUDE_distance_driven_l161_16169


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l161_16191

def fraction : ℚ := 3 / (2^7 * 5^10)

theorem zeros_before_first_nonzero_digit : 
  (∃ (n : ℕ) (d : ℚ), fraction * 10^n = d ∧ d ≥ 1 ∧ d < 10 ∧ n = 8) :=
sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l161_16191


namespace NUMINAMATH_CALUDE_area_triangle_AOB_l161_16137

/-- Given two points A and B in polar coordinates, prove that the area of triangle AOB is 6 -/
theorem area_triangle_AOB (A B : ℝ × ℝ) : 
  A.1 = 3 ∧ A.2 = π/3 ∧ B.1 = 4 ∧ B.2 = 5*π/6 → 
  (1/2) * A.1 * B.1 * Real.sin (B.2 - A.2) = 6 := by
sorry

end NUMINAMATH_CALUDE_area_triangle_AOB_l161_16137


namespace NUMINAMATH_CALUDE_colored_pencils_ratio_l161_16194

/-- Proves that given the conditions in the problem, the ratio of Cheryl's colored pencils to Cyrus's is 3:1 -/
theorem colored_pencils_ratio (madeline_pencils : ℕ) (total_pencils : ℕ) 
  (h1 : madeline_pencils = 63)
  (h2 : total_pencils = 231) : ∃ (cheryl_pencils cyrus_pencils : ℕ),
  cheryl_pencils = 2 * madeline_pencils ∧
  total_pencils = cheryl_pencils + cyrus_pencils + madeline_pencils ∧
  cheryl_pencils / cyrus_pencils = 3 := by
  sorry


end NUMINAMATH_CALUDE_colored_pencils_ratio_l161_16194


namespace NUMINAMATH_CALUDE_sarah_weeds_proof_l161_16121

def tuesday_weeds : ℕ := 25

def wednesday_weeds (t : ℕ) : ℕ := 3 * t

def thursday_weeds (w : ℕ) : ℕ := w / 5

def friday_weeds (th : ℕ) : ℕ := th - 10

def total_weeds (t w th f : ℕ) : ℕ := t + w + th + f

theorem sarah_weeds_proof :
  total_weeds tuesday_weeds 
               (wednesday_weeds tuesday_weeds) 
               (thursday_weeds (wednesday_weeds tuesday_weeds)) 
               (friday_weeds (thursday_weeds (wednesday_weeds tuesday_weeds))) = 120 := by
  sorry

end NUMINAMATH_CALUDE_sarah_weeds_proof_l161_16121


namespace NUMINAMATH_CALUDE_sum_4_inclusive_numbers_eq_1883_l161_16131

/-- Returns true if the number contains the digit 4 -/
def contains4 (n : ℕ) : Bool :=
  n.repr.contains '4'

/-- Returns true if the number is 4-inclusive (multiple of 4 or contains 4) -/
def is4Inclusive (n : ℕ) : Bool :=
  n % 4 = 0 || contains4 n

/-- The sum of all 4-inclusive numbers in the range [0, 100] -/
def sum4InclusiveNumbers : ℕ :=
  (List.range 101).filter is4Inclusive |>.sum

theorem sum_4_inclusive_numbers_eq_1883 : sum4InclusiveNumbers = 1883 := by
  sorry

end NUMINAMATH_CALUDE_sum_4_inclusive_numbers_eq_1883_l161_16131


namespace NUMINAMATH_CALUDE_f_is_linear_l161_16168

/-- A function f: ℝ → ℝ is linear if there exist constants m and b such that f(x) = mx + b for all x ∈ ℝ -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

/-- The function f(x) = -x -/
def f : ℝ → ℝ := fun x ↦ -x

/-- Theorem: The function f(x) = -x is a linear function -/
theorem f_is_linear : IsLinearFunction f := by
  sorry

end NUMINAMATH_CALUDE_f_is_linear_l161_16168


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l161_16159

/-- A rectangle with given diagonal and area has a specific perimeter -/
theorem rectangle_perimeter (x y : ℝ) (h_diagonal : x^2 + y^2 = 17^2) (h_area : x * y = 120) :
  2 * (x + y) = 46 := by
  sorry

#check rectangle_perimeter

end NUMINAMATH_CALUDE_rectangle_perimeter_l161_16159


namespace NUMINAMATH_CALUDE_carol_to_cathy_ratio_ratio_is_one_to_one_l161_16138

-- Define the number of cars each person owns
def cathy_cars : ℕ := 5
def carol_cars : ℕ := cathy_cars
def lindsey_cars : ℕ := cathy_cars + 4
def susan_cars : ℕ := carol_cars - 2

-- Theorem to prove
theorem carol_to_cathy_ratio : 
  carol_cars = cathy_cars := by sorry

-- The ratio is 1:1 if the numbers are equal
theorem ratio_is_one_to_one : 
  carol_cars = cathy_cars → (carol_cars : ℚ) / cathy_cars = 1 := by sorry

end NUMINAMATH_CALUDE_carol_to_cathy_ratio_ratio_is_one_to_one_l161_16138


namespace NUMINAMATH_CALUDE_subset_implies_range_l161_16109

-- Define the sets N and M
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x < 2 * a - 1}
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- State the theorem
theorem subset_implies_range (a : ℝ) : N a ⊆ M → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_range_l161_16109


namespace NUMINAMATH_CALUDE_section_b_average_weight_l161_16118

/-- Proves that the average weight of section B is 40.01 kg given the class composition and weight information -/
theorem section_b_average_weight 
  (num_students_a : ℕ) 
  (num_students_b : ℕ) 
  (avg_weight_a : ℝ) 
  (avg_weight_total : ℝ) : 
  let total_students : ℕ := num_students_a + num_students_b
  let total_weight : ℝ := avg_weight_total * total_students
  let weight_a : ℝ := avg_weight_a * num_students_a
  let weight_b : ℝ := total_weight - weight_a
  let avg_weight_b : ℝ := weight_b / num_students_b
  num_students_a = 40 ∧ 
  num_students_b = 20 ∧ 
  avg_weight_a = 50 ∧ 
  avg_weight_total = 46.67 →
  avg_weight_b = 40.01 := by
sorry

end NUMINAMATH_CALUDE_section_b_average_weight_l161_16118


namespace NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_x_satisfying_inequality_l161_16170

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ 0 ∨ x ≥ 3} := by sorry

-- Theorem for the range of x satisfying the inequality
theorem range_of_x_satisfying_inequality :
  {x : ℝ | ∀ a b : ℝ, a ≠ 0 → |a + b| + |a - b| ≥ a * f x} = 
  {x : ℝ | 1/2 ≤ x ∧ x ≤ 5/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_x_satisfying_inequality_l161_16170


namespace NUMINAMATH_CALUDE_percentage_multiplication_l161_16120

theorem percentage_multiplication : (10 / 100 * 10) * (20 / 100 * 20) = 4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_multiplication_l161_16120


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l161_16143

def M : Set ℝ := {x | x^2 - x - 12 = 0}
def N : Set ℝ := {x | x^2 + 3*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, -3, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l161_16143


namespace NUMINAMATH_CALUDE_purple_ring_weight_l161_16180

/-- The weight of the purple ring given the weights of other rings and the total weight -/
theorem purple_ring_weight 
  (orange_weight : ℝ) 
  (white_weight : ℝ) 
  (total_weight : ℝ) 
  (h_orange : orange_weight = 0.08333333333333333)
  (h_white : white_weight = 0.4166666666666667)
  (h_total : total_weight = 0.8333333333333334) :
  total_weight - (orange_weight + white_weight) = 0.3333333333333334 := by
sorry

end NUMINAMATH_CALUDE_purple_ring_weight_l161_16180


namespace NUMINAMATH_CALUDE_average_weight_of_four_friends_l161_16115

/-- The average weight of four friends given their relative weights -/
theorem average_weight_of_four_friends 
  (jalen_weight : ℝ)
  (ponce_weight : ℝ)
  (ishmael_weight : ℝ)
  (mike_weight : ℝ)
  (h1 : jalen_weight = 160)
  (h2 : ponce_weight = jalen_weight - 10)
  (h3 : ishmael_weight = ponce_weight + 20)
  (h4 : mike_weight = ishmael_weight + ponce_weight + jalen_weight - 15) :
  (jalen_weight + ponce_weight + ishmael_weight + mike_weight) / 4 = 236.25 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_of_four_friends_l161_16115


namespace NUMINAMATH_CALUDE_always_balanced_arrangement_l161_16100

-- Define the cube type
structure Cube :=
  (blue_faces : Nat)
  (red_faces : Nat)

-- Define the set of 8 cubes
def CubeSet := List Cube

-- Define the property of a valid cube set
def ValidCubeSet (cs : CubeSet) : Prop :=
  cs.length = 8 ∧
  (cs.map (·.blue_faces)).sum = 24 ∧
  (cs.map (·.red_faces)).sum = 24

-- Define the property of a balanced surface
def BalancedSurface (surface_blue : Nat) (surface_red : Nat) : Prop :=
  surface_blue = surface_red ∧ surface_blue + surface_red = 24

-- Main theorem
theorem always_balanced_arrangement (cs : CubeSet) 
  (h : ValidCubeSet cs) : 
  ∃ (surface_blue surface_red : Nat), 
    BalancedSurface surface_blue surface_red :=
sorry

end NUMINAMATH_CALUDE_always_balanced_arrangement_l161_16100


namespace NUMINAMATH_CALUDE_divisibility_rule_l161_16129

theorem divisibility_rule (x y : ℕ+) (h : (1000 * y + x : ℕ) > 0) :
  (((x : ℤ) - (y : ℤ)) % 7 = 0 ∨ ((x : ℤ) - (y : ℤ)) % 11 = 0) →
  ((1000 * y + x : ℕ) % 7 = 0 ∨ (1000 * y + x : ℕ) % 11 = 0) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_rule_l161_16129


namespace NUMINAMATH_CALUDE_necessary_condition_inequality_l161_16161

theorem necessary_condition_inequality (a b c : ℝ) (hc : c ≠ 0) :
  (∀ c, c ≠ 0 → a * c^2 > b * c^2) → a > b :=
sorry

end NUMINAMATH_CALUDE_necessary_condition_inequality_l161_16161


namespace NUMINAMATH_CALUDE_light_travel_distance_l161_16192

/-- The distance light travels in one year in kilometers -/
def light_year_distance : ℝ := 9460800000000

/-- The number of years we're calculating for -/
def years : ℕ := 50

/-- The expected distance light travels in 50 years -/
def expected_distance : ℝ := 473.04 * (10 ^ 12)

/-- Theorem stating that the distance light travels in 50 years is equal to the expected distance -/
theorem light_travel_distance : light_year_distance * (years : ℝ) = expected_distance := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l161_16192


namespace NUMINAMATH_CALUDE_silver_knights_enchanted_fraction_l161_16123

structure Kingdom where
  total_knights : ℕ
  silver_knights : ℕ
  gold_knights : ℕ
  enchanted_knights : ℕ
  enchanted_silver : ℕ
  enchanted_gold : ℕ

def is_valid_kingdom (k : Kingdom) : Prop :=
  k.silver_knights + k.gold_knights = k.total_knights ∧
  k.silver_knights = (3 * k.total_knights) / 8 ∧
  k.enchanted_knights = k.total_knights / 8 ∧
  k.enchanted_silver + k.enchanted_gold = k.enchanted_knights ∧
  3 * k.enchanted_gold * k.silver_knights = k.enchanted_silver * k.gold_knights

theorem silver_knights_enchanted_fraction (k : Kingdom) 
  (h : is_valid_kingdom k) : 
  (k.enchanted_silver : ℚ) / k.silver_knights = 1 / 14 := by
  sorry

end NUMINAMATH_CALUDE_silver_knights_enchanted_fraction_l161_16123


namespace NUMINAMATH_CALUDE_sine_inequality_l161_16172

theorem sine_inequality (x y : Real) : 
  x ∈ Set.Icc 0 (Real.pi / 2) → 
  y ∈ Set.Icc 0 Real.pi → 
  Real.sin (x + y) ≥ Real.sin x - Real.sin y := by
sorry

end NUMINAMATH_CALUDE_sine_inequality_l161_16172


namespace NUMINAMATH_CALUDE_apps_added_l161_16133

theorem apps_added (initial_apps final_apps : ℕ) 
  (h1 : initial_apps = 17) 
  (h2 : final_apps = 18) : 
  final_apps - initial_apps = 1 := by
  sorry

end NUMINAMATH_CALUDE_apps_added_l161_16133


namespace NUMINAMATH_CALUDE_point_B_coordinates_l161_16130

-- Define the point A and vector a
def A : ℝ × ℝ := (2, 4)
def a : ℝ × ℝ := (3, 4)

-- Define the relation between AB and a
def AB_relation (B : ℝ × ℝ) : Prop :=
  (B.1 - A.1, B.2 - A.2) = (2 * a.1, 2 * a.2)

-- Theorem stating that B has coordinates (8, 12)
theorem point_B_coordinates :
  ∃ B : ℝ × ℝ, AB_relation B ∧ B = (8, 12) := by sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l161_16130


namespace NUMINAMATH_CALUDE_factorization_of_x4_plus_64_l161_16190

theorem factorization_of_x4_plus_64 (x : ℝ) : x^4 + 64 = (x^2 - 4*x + 8) * (x^2 + 4*x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x4_plus_64_l161_16190


namespace NUMINAMATH_CALUDE_bus_purchase_problem_l161_16171

-- Define the variables
variable (a b : ℝ)
variable (x : ℝ)  -- Number of A model buses

-- Define the conditions
def total_buses : ℝ := 10
def fuel_savings_A : ℝ := 2.4
def fuel_savings_B : ℝ := 2
def price_difference : ℝ := 2
def model_cost_difference : ℝ := 6
def total_fuel_savings : ℝ := 22.4

-- State the theorem
theorem bus_purchase_problem :
  (a - b = price_difference) →
  (3 * b - 2 * a = model_cost_difference) →
  (fuel_savings_A * x + fuel_savings_B * (total_buses - x) = total_fuel_savings) →
  (a = 120 ∧ b = 100 ∧ x = 6 ∧ a * x + b * (total_buses - x) = 1120) := by
  sorry

end NUMINAMATH_CALUDE_bus_purchase_problem_l161_16171


namespace NUMINAMATH_CALUDE_parallel_case_perpendicular_case_l161_16102

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Define parallel condition
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Define perpendicular condition
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Theorem for parallel case
theorem parallel_case :
  ∃ (x : ℝ), parallel (a + 2 • (b x)) (2 • a - b x) ∧ x = 1/2 := by sorry

-- Theorem for perpendicular case
theorem perpendicular_case :
  ∃ (x : ℝ), perpendicular (a + 2 • (b x)) (2 • a - b x) ∧ (x = -2 ∨ x = 7/2) := by sorry

end NUMINAMATH_CALUDE_parallel_case_perpendicular_case_l161_16102


namespace NUMINAMATH_CALUDE_matrix_power_2023_l161_16108

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 0; 2, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1, 0; 4046, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l161_16108


namespace NUMINAMATH_CALUDE_differentials_of_z_l161_16199

noncomputable section

variables (x y : ℝ) (dx dy : ℝ)

def z : ℝ := x^5 * y^3

def dz : ℝ := 5 * x^4 * y^3 * dx + 3 * x^5 * y^2 * dy

def d2z : ℝ := 20 * x^3 * y^3 * dx^2 + 30 * x^4 * y^2 * dx * dy + 6 * x^5 * y * dy^2

def d3z : ℝ := 60 * x^2 * y^3 * dx^3 + 180 * x^3 * y^2 * dx^2 * dy + 90 * x^4 * y * dx * dy^2 + 6 * x^5 * dy^3

theorem differentials_of_z :
  (dz x y dx dy = 5 * x^4 * y^3 * dx + 3 * x^5 * y^2 * dy) ∧
  (d2z x y dx dy = 20 * x^3 * y^3 * dx^2 + 30 * x^4 * y^2 * dx * dy + 6 * x^5 * y * dy^2) ∧
  (d3z x y dx dy = 60 * x^2 * y^3 * dx^3 + 180 * x^3 * y^2 * dx^2 * dy + 90 * x^4 * y * dx * dy^2 + 6 * x^5 * dy^3) :=
by sorry

end NUMINAMATH_CALUDE_differentials_of_z_l161_16199


namespace NUMINAMATH_CALUDE_quadratic_root_and_coefficient_l161_16104

theorem quadratic_root_and_coefficient (m : ℝ) :
  (∃ x, x^2 + m*x + 2 = 0 ∧ x = -2) →
  (∃ y, y^2 + m*y + 2 = 0 ∧ y = -1) ∧ m = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_and_coefficient_l161_16104


namespace NUMINAMATH_CALUDE_principal_amount_proof_l161_16153

-- Define the parameters of the investment
def interest_rate : ℚ := 5 / 100
def investment_duration : ℕ := 5
def final_amount : ℚ := 10210.25

-- Define the compound interest formula
def compound_interest (principal : ℚ) : ℚ :=
  principal * (1 + interest_rate) ^ investment_duration

-- State the theorem
theorem principal_amount_proof :
  ∃ (principal : ℚ), 
    compound_interest principal = final_amount ∧ 
    (principal ≥ 7999.5 ∧ principal ≤ 8000.5) := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l161_16153


namespace NUMINAMATH_CALUDE_suit_price_problem_l161_16156

theorem suit_price_problem (P : ℝ) : 
  (0.7 * (1.3 * P) = 182) → P = 200 := by
  sorry

end NUMINAMATH_CALUDE_suit_price_problem_l161_16156


namespace NUMINAMATH_CALUDE_liza_reading_speed_l161_16136

/-- Given that Suzie reads 15 pages in an hour and Liza reads 15 more pages than Suzie in 3 hours,
    prove that Liza reads 20 pages in an hour. -/
theorem liza_reading_speed (suzie_pages_per_hour : ℕ) (liza_extra_pages : ℕ) :
  suzie_pages_per_hour = 15 →
  liza_extra_pages = 15 →
  ∃ (liza_pages_per_hour : ℕ),
    liza_pages_per_hour * 3 = suzie_pages_per_hour * 3 + liza_extra_pages ∧
    liza_pages_per_hour = 20 :=
by sorry

end NUMINAMATH_CALUDE_liza_reading_speed_l161_16136


namespace NUMINAMATH_CALUDE_perpendicular_points_constant_sum_l161_16113

/-- The curve E in polar coordinates -/
def curve_E (ρ θ : ℝ) : Prop :=
  ρ^2 * (1/3 * Real.cos θ^2 + 1/2 * Real.sin θ^2) = 1

/-- Theorem: For any two perpendicular points on curve E, the sum of reciprocals of their squared distances from the origin is constant -/
theorem perpendicular_points_constant_sum (ρ₁ ρ₂ θ : ℝ) :
  curve_E ρ₁ θ → curve_E ρ₂ (θ + π/2) → 1/ρ₁^2 + 1/ρ₂^2 = 5/6 := by
  sorry

#check perpendicular_points_constant_sum

end NUMINAMATH_CALUDE_perpendicular_points_constant_sum_l161_16113


namespace NUMINAMATH_CALUDE_trajectory_equation_l161_16184

/-- The equation of the trajectory of the center of a circle that passes through point A (2, 0) and is tangent to the circle x^2 + 4x + y^2 - 32 = 0 is x^2/9 + y^2/5 = 1 -/
theorem trajectory_equation : ∃ (f : ℝ × ℝ → ℝ), 
  (∀ (x y : ℝ), f (x, y) = 0 ↔ x^2/9 + y^2/5 = 1) ∧
  (∀ (x y : ℝ), f (x, y) = 0 → 
    ∃ (r : ℝ), r > 0 ∧
    (∀ (u v : ℝ), (u - x)^2 + (v - y)^2 = r^2 → 
      ((u - 2)^2 + v^2 = 0 ∨ u^2 + 4*u + v^2 - 32 = 0))) :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l161_16184


namespace NUMINAMATH_CALUDE_least_nickels_l161_16177

theorem least_nickels (n : ℕ) : 
  (n > 0) → 
  (n % 7 = 2) → 
  (n % 4 = 3) → 
  (∀ m : ℕ, m > 0 → m % 7 = 2 → m % 4 = 3 → n ≤ m) → 
  n = 23 := by
sorry

end NUMINAMATH_CALUDE_least_nickels_l161_16177


namespace NUMINAMATH_CALUDE_f_at_5_eq_neg_13_l161_16188

/-- A polynomial function of degree 7 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

/-- Theorem stating that f(5) = -13 given f(-5) = 17 -/
theorem f_at_5_eq_neg_13 {a b c : ℝ} (h : f a b c (-5) = 17) : f a b c 5 = -13 := by
  sorry

end NUMINAMATH_CALUDE_f_at_5_eq_neg_13_l161_16188


namespace NUMINAMATH_CALUDE_sandwich_count_l161_16152

def num_bread_types : ℕ := 12
def num_spread_types : ℕ := 10

def sandwich_combinations : ℕ := num_bread_types * (num_spread_types.choose 2)

theorem sandwich_count : sandwich_combinations = 540 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_count_l161_16152


namespace NUMINAMATH_CALUDE_smallest_n_divisible_twelve_satisfies_twelve_is_smallest_l161_16193

theorem smallest_n_divisible (n : ℕ) : n > 0 ∧ 24 ∣ n^2 ∧ 864 ∣ n^3 → n ≥ 12 := by
  sorry

theorem twelve_satisfies : 24 ∣ 12^2 ∧ 864 ∣ 12^3 := by
  sorry

theorem twelve_is_smallest : ∃ (n : ℕ), n > 0 ∧ 24 ∣ n^2 ∧ 864 ∣ n^3 ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_twelve_satisfies_twelve_is_smallest_l161_16193


namespace NUMINAMATH_CALUDE_speed_reduction_proof_l161_16122

/-- The speed reduction per passenger in MPH -/
def speed_reduction_per_passenger : ℝ := 2

/-- The speed of an empty plane in MPH -/
def empty_plane_speed : ℝ := 600

/-- The number of passengers on the first plane -/
def passengers_plane1 : ℕ := 50

/-- The number of passengers on the second plane -/
def passengers_plane2 : ℕ := 60

/-- The number of passengers on the third plane -/
def passengers_plane3 : ℕ := 40

/-- The average speed of the three planes in MPH -/
def average_speed : ℝ := 500

theorem speed_reduction_proof :
  (empty_plane_speed - speed_reduction_per_passenger * passengers_plane1 +
   empty_plane_speed - speed_reduction_per_passenger * passengers_plane2 +
   empty_plane_speed - speed_reduction_per_passenger * passengers_plane3) / 3 = average_speed :=
by sorry

end NUMINAMATH_CALUDE_speed_reduction_proof_l161_16122


namespace NUMINAMATH_CALUDE_number_of_people_entered_l161_16163

/-- The number of placards each person takes -/
def placards_per_person : ℕ := 2

/-- The total number of placards the basket can hold -/
def basket_capacity : ℕ := 823

/-- The number of people who entered the stadium -/
def people_entered : ℕ := basket_capacity / placards_per_person

/-- Theorem stating the number of people who entered the stadium -/
theorem number_of_people_entered : people_entered = 411 := by sorry

end NUMINAMATH_CALUDE_number_of_people_entered_l161_16163


namespace NUMINAMATH_CALUDE_system_solution_l161_16187

theorem system_solution (x y : ℝ) (h1 : x + 2*y = 8) (h2 : 2*x + y = 7) : x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l161_16187


namespace NUMINAMATH_CALUDE_triangle_property_l161_16139

/-- Given a triangle ABC with sides a, b, and c satisfying the equation
    a^2 + b^2 + c^2 + 50 = 6a + 8b + 10c, prove that it is a right-angled
    triangle with area 6. -/
theorem triangle_property (a b c : ℝ) (h : a^2 + b^2 + c^2 + 50 = 6*a + 8*b + 10*c) :
  (a = 3 ∧ b = 4 ∧ c = 5) ∧ a^2 + b^2 = c^2 ∧ (1/2) * a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l161_16139


namespace NUMINAMATH_CALUDE_wooden_statue_cost_l161_16105

/-- The cost of a wooden statue given Theodore's production and earnings. -/
theorem wooden_statue_cost :
  let stone_statues : ℕ := 10
  let wooden_statues : ℕ := 20
  let stone_cost : ℚ := 20
  let tax_rate : ℚ := 1/10
  let total_earnings : ℚ := 270
  ∃ (wooden_cost : ℚ),
    (1 - tax_rate) * (stone_statues * stone_cost + wooden_statues * wooden_cost) = total_earnings ∧
    wooden_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_wooden_statue_cost_l161_16105


namespace NUMINAMATH_CALUDE_typing_problem_solution_l161_16175

/-- Represents the typing speed and time for two typists -/
structure TypistData where
  x : ℝ  -- Time taken by first typist to type entire manuscript
  y : ℝ  -- Time taken by second typist to type entire manuscript

/-- Checks if the given typing times satisfy the manuscript typing conditions -/
def satisfiesConditions (d : TypistData) : Prop :=
  let totalPages : ℝ := 80
  let pagesTypedIn5Hours : ℝ := 65
  let timeDiff : ℝ := 3
  (totalPages / d.y - totalPages / d.x = timeDiff) ∧
  (5 * (totalPages / d.x + totalPages / d.y) = pagesTypedIn5Hours)

/-- Theorem stating the solution to the typing problem -/
theorem typing_problem_solution :
  ∃ d : TypistData, satisfiesConditions d ∧ d.x = 10 ∧ d.y = 16 := by
  sorry


end NUMINAMATH_CALUDE_typing_problem_solution_l161_16175


namespace NUMINAMATH_CALUDE_not_center_of_symmetry_l161_16114

/-- The tangent function -/
noncomputable def tan (x : ℝ) : ℝ := sorry

/-- The function y = tan(2x - π/4) -/
noncomputable def f (x : ℝ) : ℝ := tan (2 * x - Real.pi / 4)

/-- A point is a center of symmetry if it has the form (kπ/4 + π/8, 0) for some integer k -/
def is_center_of_symmetry (p : ℝ × ℝ) : Prop :=
  ∃ k : ℤ, p.1 = k * Real.pi / 4 + Real.pi / 8 ∧ p.2 = 0

/-- The statement to be proved -/
theorem not_center_of_symmetry :
  ¬ is_center_of_symmetry (Real.pi / 4, 0) :=
sorry

end NUMINAMATH_CALUDE_not_center_of_symmetry_l161_16114


namespace NUMINAMATH_CALUDE_simplify_product_of_square_roots_l161_16141

theorem simplify_product_of_square_roots (x : ℝ) (hx : x > 0) :
  Real.sqrt (56 * x^3) * Real.sqrt (10 * x^2) * Real.sqrt (63 * x^4) = 84 * x^4 * Real.sqrt (5 * x) :=
by sorry

end NUMINAMATH_CALUDE_simplify_product_of_square_roots_l161_16141


namespace NUMINAMATH_CALUDE_increase_by_percentage_l161_16149

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 200 →
  percentage = 25 →
  final = initial * (1 + percentage / 100) →
  final = 250 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l161_16149


namespace NUMINAMATH_CALUDE_count_squares_l161_16185

/-- The number of groups of squares in the figure -/
def num_groups : ℕ := 5

/-- The number of squares in each group -/
def squares_per_group : ℕ := 5

/-- The total number of squares in the figure -/
def total_squares : ℕ := num_groups * squares_per_group

theorem count_squares : total_squares = 25 := by
  sorry

end NUMINAMATH_CALUDE_count_squares_l161_16185


namespace NUMINAMATH_CALUDE_find_y_l161_16134

theorem find_y : ∃ y : ℝ, (3 * y) / 7 = 12 ∧ y = 28 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l161_16134


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l161_16125

theorem compare_negative_fractions : -3/4 > -4/5 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l161_16125


namespace NUMINAMATH_CALUDE_total_legs_sea_creatures_l161_16196

/-- Calculate the total number of legs for sea creatures --/
theorem total_legs_sea_creatures :
  let num_octopuses : ℕ := 5
  let num_crabs : ℕ := 3
  let num_starfish : ℕ := 2
  let legs_per_octopus : ℕ := 8
  let legs_per_crab : ℕ := 10
  let legs_per_starfish : ℕ := 5
  num_octopuses * legs_per_octopus +
  num_crabs * legs_per_crab +
  num_starfish * legs_per_starfish = 80 :=
by sorry

end NUMINAMATH_CALUDE_total_legs_sea_creatures_l161_16196


namespace NUMINAMATH_CALUDE_no_real_solutions_l161_16186

theorem no_real_solutions (m : ℝ) : 
  (∀ x : ℝ, (x - 1) / (x + 4) ≠ m / (x + 4)) ↔ m = -5 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l161_16186


namespace NUMINAMATH_CALUDE_optimal_sampling_for_populations_l161_16178

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Random
  | Systematic
  | Stratified

/-- Represents a population with its characteristics -/
structure Population where
  total : ℕ
  subgroups : List ℕ
  has_distinct_subgroups : Bool

/-- Determines the optimal sampling method for a given population -/
def optimal_sampling_method (pop : Population) : SamplingMethod :=
  if pop.has_distinct_subgroups then
    SamplingMethod.Stratified
  else
    SamplingMethod.Random

/-- The main theorem stating the optimal sampling methods for given populations -/
theorem optimal_sampling_for_populations 
  (pop1 : Population) 
  (pop2 : Population) 
  (h1 : pop1.has_distinct_subgroups = true) 
  (h2 : pop2.has_distinct_subgroups = false) :
  (optimal_sampling_method pop1 = SamplingMethod.Stratified) ∧
  (optimal_sampling_method pop2 = SamplingMethod.Random) := by
  sorry

#check optimal_sampling_for_populations

end NUMINAMATH_CALUDE_optimal_sampling_for_populations_l161_16178


namespace NUMINAMATH_CALUDE_complex_arithmetic_l161_16157

theorem complex_arithmetic (B N T Q : ℂ) : 
  B = 5 - 2*I ∧ N = -5 + 2*I ∧ T = 3*I ∧ Q = 3 →
  B - N + T - Q = 7 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_l161_16157


namespace NUMINAMATH_CALUDE_some_number_value_l161_16112

theorem some_number_value (some_number : ℝ) :
  (some_number * 14) / 100 = 0.045388 → some_number = 0.3242 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l161_16112


namespace NUMINAMATH_CALUDE_imaginary_power_sum_product_l161_16162

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the periodicity of i
axiom i_period (n : ℕ) : i^(n + 4) = i^n

-- State the theorem
theorem imaginary_power_sum_product : (i^22 + i^222) * i = -2 * i := by sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_product_l161_16162


namespace NUMINAMATH_CALUDE_diagonals_of_25_sided_polygon_convex_polygon_25_sides_diagonals_l161_16116

theorem diagonals_of_25_sided_polygon : ℕ → ℕ
  | n => (n * (n - 1)) / 2 - n

theorem convex_polygon_25_sides_diagonals :
  diagonals_of_25_sided_polygon 25 = 275 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_of_25_sided_polygon_convex_polygon_25_sides_diagonals_l161_16116


namespace NUMINAMATH_CALUDE_all_calculations_incorrect_l161_16167

theorem all_calculations_incorrect : 
  (-|-3| ≠ 3) ∧ 
  (∀ a b : ℝ, (a + b)^2 ≠ a^2 + b^2) ∧ 
  (∀ a : ℝ, a ≠ 0 → a^3 * a^4 ≠ a^12) ∧ 
  (|-3^2| ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_all_calculations_incorrect_l161_16167


namespace NUMINAMATH_CALUDE_competition_configs_l161_16140

/-- Represents a valid competition configuration -/
structure CompetitionConfig where
  n : ℕ
  k : ℕ
  h_n_ge_2 : n ≥ 2
  h_k_ge_1 : k ≥ 1
  h_total_score : k * (n * (n + 1) / 2) = 26 * n

/-- The set of all valid competition configurations -/
def ValidConfigs : Set CompetitionConfig := {c | c.n ≥ 2 ∧ c.k ≥ 1 ∧ c.k * (c.n * (c.n + 1) / 2) = 26 * c.n}

/-- The theorem stating the possible values of (n, k) -/
theorem competition_configs : ValidConfigs = {⟨25, 2, by norm_num, by norm_num, by norm_num⟩, 
                                              ⟨12, 4, by norm_num, by norm_num, by norm_num⟩, 
                                              ⟨3, 13, by norm_num, by norm_num, by norm_num⟩} := by
  sorry

end NUMINAMATH_CALUDE_competition_configs_l161_16140


namespace NUMINAMATH_CALUDE_point_coordinates_l161_16126

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: A point M in the second quadrant, 5 units away from the x-axis
    and 3 units away from the y-axis, has coordinates (-3, 5) -/
theorem point_coordinates (M : Point) 
  (h1 : SecondQuadrant M) 
  (h2 : DistanceToXAxis M = 5) 
  (h3 : DistanceToYAxis M = 3) : 
  M.x = -3 ∧ M.y = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l161_16126


namespace NUMINAMATH_CALUDE_polynomial_equality_l161_16144

theorem polynomial_equality (x : ℝ) : 
  (3*x^3 + 2*x^2 + 5*x + 9)*(x - 2) - (x - 2)*(2*x^3 + 5*x^2 - 74) + (4*x - 17)*(x - 2)*(x + 4) 
  = x^4 + 2*x^3 - 5*x^2 + 9*x - 30 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l161_16144


namespace NUMINAMATH_CALUDE_subcommittee_count_l161_16124

def total_members : ℕ := 12
def officers : ℕ := 5
def subcommittee_size : ℕ := 5

def subcommittees_with_at_least_two_officers : ℕ :=
  Nat.choose total_members subcommittee_size -
  (Nat.choose (total_members - officers) subcommittee_size +
   Nat.choose officers 1 * Nat.choose (total_members - officers) (subcommittee_size - 1))

theorem subcommittee_count :
  subcommittees_with_at_least_two_officers = 596 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_count_l161_16124


namespace NUMINAMATH_CALUDE_open_box_volume_formula_l161_16101

/-- The volume of an open box constructed from a rectangular sheet of metal -/
def openBoxVolume (length width x : ℝ) : ℝ :=
  (length - 2*x) * (width - 2*x) * x

theorem open_box_volume_formula :
  ∀ x : ℝ, openBoxVolume 14 10 x = 140*x - 48*x^2 + 4*x^3 :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_formula_l161_16101


namespace NUMINAMATH_CALUDE_gift_card_balance_l161_16183

/-- Calculates the remaining balance on a gift card after a coffee purchase -/
theorem gift_card_balance 
  (gift_card_amount : ℝ) 
  (coffee_price_per_pound : ℝ) 
  (pounds_purchased : ℝ) 
  (h1 : gift_card_amount = 70) 
  (h2 : coffee_price_per_pound = 8.58) 
  (h3 : pounds_purchased = 4) : 
  gift_card_amount - (coffee_price_per_pound * pounds_purchased) = 35.68 := by
sorry

end NUMINAMATH_CALUDE_gift_card_balance_l161_16183


namespace NUMINAMATH_CALUDE_equation_solutions_l161_16195

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = -1 ∧ 
  (∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l161_16195


namespace NUMINAMATH_CALUDE_attic_junk_items_l161_16181

theorem attic_junk_items (useful_percent : ℝ) (heirloom_percent : ℝ) (junk_percent : ℝ) (useful_count : ℕ) :
  useful_percent = 0.20 →
  heirloom_percent = 0.10 →
  junk_percent = 0.70 →
  useful_percent + heirloom_percent + junk_percent = 1 →
  useful_count = 8 →
  ⌊(useful_count / useful_percent) * junk_percent⌋ = 28 := by
sorry

end NUMINAMATH_CALUDE_attic_junk_items_l161_16181


namespace NUMINAMATH_CALUDE_line_parallel_to_y_axis_l161_16176

/-- A line parallel to the y-axis passing through a point has a constant x-coordinate -/
theorem line_parallel_to_y_axis (x₀ y₀ : ℝ) :
  let L := {p : ℝ × ℝ | p.1 = x₀}
  ((-1, 3) ∈ L) → (∀ p ∈ L, ∀ q ∈ L, p.2 ≠ q.2 → p.1 = q.1) →
  (∀ p ∈ L, p.1 = -1) :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_to_y_axis_l161_16176


namespace NUMINAMATH_CALUDE_unpainted_cubes_4x4x4_l161_16117

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_units : Nat
  painted_per_face : Nat

/-- Calculates the number of unpainted unit cubes in a painted cube -/
def unpainted_cubes (cube : PaintedCube) : Nat :=
  cube.total_units - (cube.painted_per_face * 6)

/-- Theorem: A 4x4x4 cube with 4 unit squares painted on each face has 40 unpainted unit cubes -/
theorem unpainted_cubes_4x4x4 :
  let cube : PaintedCube := {
    size := 4,
    total_units := 64,
    painted_per_face := 4
  }
  unpainted_cubes cube = 40 := by
  sorry


end NUMINAMATH_CALUDE_unpainted_cubes_4x4x4_l161_16117


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l161_16151

theorem quadratic_equation_solution (x : ℝ) : 
  x^2 - 6*x + 8 = 0 → x ≠ 4 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l161_16151


namespace NUMINAMATH_CALUDE_wedge_volume_l161_16154

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d h θ : ℝ) (hd : d = 20) (hh : h = 20) (hθ : θ = 30 * π / 180) :
  let r := d / 2
  let cylinder_volume := π * r^2 * h
  let wedge_volume := (θ / (2 * π)) * cylinder_volume
  wedge_volume = 250 * π := by sorry

end NUMINAMATH_CALUDE_wedge_volume_l161_16154


namespace NUMINAMATH_CALUDE_percentage_problem_l161_16128

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 16 = 40 → P = 250 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l161_16128


namespace NUMINAMATH_CALUDE_trapezium_area_l161_16111

theorem trapezium_area (a b h : ℝ) (ha : a = 24) (hb : b = 14) (hh : h = 18) :
  (a + b) * h / 2 = 342 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_area_l161_16111


namespace NUMINAMATH_CALUDE_min_m_is_one_l161_16107

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem min_m_is_one (f g h : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = Real.exp x) →
  (∀ x, f x = g x - h x) →
  IsEven g →
  IsOdd h →
  (∀ x ∈ Set.Icc (-1) 1, m * g x + h x ≥ 0) →
  (∀ m' : ℝ, (∀ x ∈ Set.Icc (-1) 1, m' * g x + h x ≥ 0) → m' ≥ m) →
  m = 1 := by sorry

end NUMINAMATH_CALUDE_min_m_is_one_l161_16107


namespace NUMINAMATH_CALUDE_percentage_difference_l161_16146

theorem percentage_difference (x y : ℝ) (h : x = 18 * y) :
  (x - y) / x * 100 = 94.44 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l161_16146


namespace NUMINAMATH_CALUDE_football_lineup_count_l161_16189

/-- The number of different lineups that can be created from a football team --/
def number_of_lineups (total_players : ℕ) (skilled_players : ℕ) : ℕ :=
  skilled_players * (total_players - 1) * (total_players - 2) * (total_players - 3) * (total_players - 4)

/-- Theorem stating that the number of lineups for a team of 15 players with 5 skilled players is 109200 --/
theorem football_lineup_count :
  number_of_lineups 15 5 = 109200 := by
  sorry

end NUMINAMATH_CALUDE_football_lineup_count_l161_16189


namespace NUMINAMATH_CALUDE_find_r_l161_16148

theorem find_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_r_l161_16148


namespace NUMINAMATH_CALUDE_subset_P_l161_16182

def P : Set ℝ := {x | x ≤ 3}

theorem subset_P : {-1} ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_subset_P_l161_16182
