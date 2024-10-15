import Mathlib

namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_101011101_base_7_l3413_341394

def base_seven_to_decimal (n : ℕ) : ℕ := 
  7^8 + 7^6 + 7^4 + 7^3 + 7^2 + 1

theorem largest_prime_divisor_of_101011101_base_7 :
  ∃ (p : ℕ), Prime p ∧ p ∣ base_seven_to_decimal 101011101 ∧
  ∀ (q : ℕ), Prime q → q ∣ base_seven_to_decimal 101011101 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_101011101_base_7_l3413_341394


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3413_341371

/-- If 9x^2 + mxy + 16y^2 is a perfect square trinomial, then m = ±24 -/
theorem perfect_square_trinomial_condition (m : ℝ) :
  (∃ (a b : ℝ), ∀ (x y : ℝ), 9*x^2 + m*x*y + 16*y^2 = (a*x + b*y)^2) →
  (m = 24 ∨ m = -24) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3413_341371


namespace NUMINAMATH_CALUDE_interest_less_than_principal_l3413_341318

/-- Calculates the simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Calculates the difference between principal and interest -/
def interest_difference (principal : ℝ) (interest : ℝ) : ℝ :=
  principal - interest

theorem interest_less_than_principal : 
  let principal : ℝ := 400.00000000000006
  let rate : ℝ := 0.04
  let time : ℝ := 8
  let interest := simple_interest principal rate time
  interest_difference principal interest = 272 := by
  sorry

end NUMINAMATH_CALUDE_interest_less_than_principal_l3413_341318


namespace NUMINAMATH_CALUDE_polygon_angle_sum_l3413_341337

theorem polygon_angle_sum (n : ℕ) (A : ℝ) (h1 : n ≥ 3) (h2 : A > 0) :
  (n - 2) * 180 = A + 2460 →
  A = 60 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angle_sum_l3413_341337


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3413_341387

/-- The function f(x) = a^(x-1) - 2 passes through the point (1, -1) for any a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f := fun x : ℝ => a^(x - 1) - 2
  f 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3413_341387


namespace NUMINAMATH_CALUDE_inverse_f_at_8_l3413_341341

def f (x : ℝ) : ℝ := 1 - 3*(x - 1) + 3*(x - 1)^2 - (x - 1)^3

theorem inverse_f_at_8 : f 0 = 8 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_8_l3413_341341


namespace NUMINAMATH_CALUDE_sum_first_50_digits_1001_l3413_341300

/-- The decimal expansion of 1/1001 -/
def decimalExpansion1001 : ℕ → ℕ
| n => match n % 6 with
  | 0 => 0
  | 1 => 0
  | 2 => 0
  | 3 => 9
  | 4 => 9
  | 5 => 9
  | _ => 0  -- This case should never occur

/-- The sum of the first n digits in the decimal expansion of 1/1001 -/
def sumFirstNDigits (n : ℕ) : ℕ :=
  (List.range n).map decimalExpansion1001 |> List.sum

/-- Theorem: The sum of the first 50 digits after the decimal point
    in the decimal expansion of 1/1001 is 216 -/
theorem sum_first_50_digits_1001 : sumFirstNDigits 50 = 216 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_50_digits_1001_l3413_341300


namespace NUMINAMATH_CALUDE_N_is_composite_l3413_341346

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ Nat.Prime N := by
  sorry

end NUMINAMATH_CALUDE_N_is_composite_l3413_341346


namespace NUMINAMATH_CALUDE_two_cos_sixty_degrees_equals_one_l3413_341305

theorem two_cos_sixty_degrees_equals_one : 2 * Real.cos (π / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_two_cos_sixty_degrees_equals_one_l3413_341305


namespace NUMINAMATH_CALUDE_min_packages_correct_min_packages_value_l3413_341377

/-- The number of t-shirts in each package -/
def package_size : ℕ := 6

/-- The number of t-shirts Mom wants to buy -/
def desired_shirts : ℕ := 71

/-- The minimum number of packages needed to buy at least the desired number of shirts -/
def min_packages : ℕ := (desired_shirts + package_size - 1) / package_size

theorem min_packages_correct : 
  min_packages * package_size ≥ desired_shirts ∧ 
  ∀ k : ℕ, k * package_size ≥ desired_shirts → k ≥ min_packages :=
by sorry

theorem min_packages_value : min_packages = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_packages_correct_min_packages_value_l3413_341377


namespace NUMINAMATH_CALUDE_indians_invented_arabic_numerals_l3413_341304

/-- Represents a numerical system -/
structure NumericalSystem where
  digits : Set Nat
  name : String
  isUniversal : Bool

/-- The civilization that invented a numerical system -/
inductive Civilization
  | Indians
  | Chinese
  | Babylonians
  | Arabs

/-- Arabic numerals as defined in the problem -/
def arabicNumerals : NumericalSystem :=
  { digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    name := "Arabic numerals",
    isUniversal := true }

/-- The theorem stating that ancient Indians invented Arabic numerals -/
theorem indians_invented_arabic_numerals :
  ∃ (inventor : Civilization), inventor = Civilization.Indians ∧
  (arabicNumerals.digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
   arabicNumerals.name = "Arabic numerals" ∧
   arabicNumerals.isUniversal = true) :=
by sorry

end NUMINAMATH_CALUDE_indians_invented_arabic_numerals_l3413_341304


namespace NUMINAMATH_CALUDE_function_inequality_l3413_341347

open Real

theorem function_inequality (f : ℝ → ℝ) (f_deriv : Differentiable ℝ f) 
  (h1 : ∀ x, f x + deriv f x > 1) (h2 : f 0 = 4) :
  ∀ x, f x > 3 / exp x + 1 ↔ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3413_341347


namespace NUMINAMATH_CALUDE_coefficient_b_value_l3413_341368

-- Define the polynomial P(x)
def P (a b d c : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + d*x + c

-- Define the sum of zeros
def sum_of_zeros (a : ℝ) : ℝ := -a

-- Define the product of zeros taken three at a time
def product_of_three_zeros (d : ℝ) : ℝ := d

-- Define the sum of coefficients
def sum_of_coefficients (a b d c : ℝ) : ℝ := 1 + a + b + d + c

-- State the theorem
theorem coefficient_b_value (a b d c : ℝ) :
  sum_of_zeros a = product_of_three_zeros d ∧
  sum_of_zeros a = sum_of_coefficients a b d c ∧
  P a b d c 0 = 8 →
  b = -17 := by sorry

end NUMINAMATH_CALUDE_coefficient_b_value_l3413_341368


namespace NUMINAMATH_CALUDE_total_distance_of_trip_l3413_341352

-- Define the triangle XYZ
def Triangle (XY YZ ZX : ℝ) : Prop :=
  XY > 0 ∧ YZ > 0 ∧ ZX > 0 ∧ XY^2 = YZ^2 + ZX^2

-- Theorem statement
theorem total_distance_of_trip (XY YZ ZX : ℝ) 
  (h1 : Triangle XY YZ ZX) (h2 : XY = 5000) (h3 : ZX = 4000) : 
  XY + YZ + ZX = 12000 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_of_trip_l3413_341352


namespace NUMINAMATH_CALUDE_cubic_difference_l3413_341319

theorem cubic_difference (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 20) :
  x^3 - y^3 = -448 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_l3413_341319


namespace NUMINAMATH_CALUDE_soccer_team_selection_l3413_341340

/-- The number of ways to choose an ordered selection of 5 players from a team of 15 players -/
def choose_squad (team_size : Nat) : Nat :=
  team_size * (team_size - 1) * (team_size - 2) * (team_size - 3) * (team_size - 4)

/-- Theorem stating that choosing 5 players from a team of 15 results in 360,360 possibilities -/
theorem soccer_team_selection :
  choose_squad 15 = 360360 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_selection_l3413_341340


namespace NUMINAMATH_CALUDE_cantaloupes_sum_l3413_341344

/-- The number of cantaloupes grown by Fred -/
def fred_cantaloupes : ℕ := 38

/-- The number of cantaloupes grown by Tim -/
def tim_cantaloupes : ℕ := 44

/-- The total number of cantaloupes grown by Fred and Tim -/
def total_cantaloupes : ℕ := fred_cantaloupes + tim_cantaloupes

theorem cantaloupes_sum : total_cantaloupes = 82 := by
  sorry

end NUMINAMATH_CALUDE_cantaloupes_sum_l3413_341344


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l3413_341345

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l3413_341345


namespace NUMINAMATH_CALUDE_base7_5463_equals_1956_l3413_341336

def base7ToBase10 (a b c d : ℕ) : ℕ := a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

theorem base7_5463_equals_1956 : base7ToBase10 5 4 6 3 = 1956 := by
  sorry

end NUMINAMATH_CALUDE_base7_5463_equals_1956_l3413_341336


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l3413_341317

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

-- Define the lines l
def line_l₁ (x y : ℝ) : Prop :=
  4*x + 3*y + 3 = 0

def line_l₂ (x y : ℝ) : Prop :=
  4*x + 3*y - 7 = 0

-- Theorem statement
theorem circle_and_line_properties :
  -- Circle C passes through O(0,0), A(-2,4), and B(1,1)
  circle_C 0 0 ∧ circle_C (-2) 4 ∧ circle_C 1 1 ∧
  -- Line l has slope -4/3
  (∀ x y : ℝ, (line_l₁ x y ∨ line_l₂ x y) → (y - 2) = -4/3 * (x + 1)) ∧
  -- The chord intercepted by circle C on line l has a length of 4
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    ((line_l₁ x₁ y₁ ∧ line_l₁ x₂ y₂) ∨ (line_l₂ x₁ y₁ ∧ line_l₂ x₂ y₂)) ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 16) →
  -- The equation of circle C is correct
  (∀ x y : ℝ, circle_C x y ↔ x^2 + y^2 + 2*x - 4*y = 0) ∧
  -- The equation of line l is one of the two given equations
  (∀ x y : ℝ, (4*x + 3*y + 3 = 0 ∨ 4*x + 3*y - 7 = 0) ↔ (line_l₁ x y ∨ line_l₂ x y)) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l3413_341317


namespace NUMINAMATH_CALUDE_interval_intersection_l3413_341356

theorem interval_intersection (x : ℝ) : (|5 - x| < 5 ∧ x^2 < 25) ↔ (0 < x ∧ x < 5) := by
  sorry

end NUMINAMATH_CALUDE_interval_intersection_l3413_341356


namespace NUMINAMATH_CALUDE_special_square_pt_length_l3413_341331

/-- Square with side length √2 and special folding property -/
structure SpecialSquare where
  -- Square side length
  side : ℝ
  side_eq : side = Real.sqrt 2
  -- Points T and U on sides PQ and RQ
  t : ℝ
  u : ℝ
  t_range : 0 < t ∧ t < side
  u_range : 0 < u ∧ u < side
  -- PT = QU
  pt_eq_qu : t = u
  -- Folding property: PS and RS coincide on diagonal QS
  folding : t * Real.sqrt 2 = side

/-- The length of PT in a SpecialSquare can be expressed as √8 - 2 -/
theorem special_square_pt_length (s : SpecialSquare) : s.t = Real.sqrt 8 - 2 := by
  sorry

#check special_square_pt_length

end NUMINAMATH_CALUDE_special_square_pt_length_l3413_341331


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_is_sqrt3_l3413_341314

noncomputable def hyperbola_eccentricity (a : ℝ) : ℝ := 
  let b : ℝ := Real.sqrt 8
  let c : ℝ := Real.sqrt (a^2 + b^2)
  c / a

theorem hyperbola_eccentricity_is_sqrt3 :
  ∃ (a : ℝ), a > 0 ∧ 
  (∃ (x y : ℝ), x = y ∧ x^2/a^2 - y^2/8 = 1) ∧
  (∃ (x1 x2 : ℝ), x1 * x2 = -8 ∧ 
    x1^2/a^2 - x1^2/8 = 1 ∧ 
    x2^2/a^2 - x2^2/8 = 1) ∧
  hyperbola_eccentricity a = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_is_sqrt3_l3413_341314


namespace NUMINAMATH_CALUDE_natasha_hill_climbing_l3413_341381

/-- Natasha's hill climbing problem -/
theorem natasha_hill_climbing
  (time_up : ℝ)
  (time_down : ℝ)
  (avg_speed_total : ℝ)
  (h_time_up : time_up = 4)
  (h_time_down : time_down = 2)
  (h_avg_speed_total : avg_speed_total = 2) :
  let total_time := time_up + time_down
  let total_distance := avg_speed_total * total_time
  let distance_up := total_distance / 2
  let avg_speed_up := distance_up / time_up
  avg_speed_up = 1.5 := by
sorry

end NUMINAMATH_CALUDE_natasha_hill_climbing_l3413_341381


namespace NUMINAMATH_CALUDE_regular_price_is_0_15_l3413_341390

/-- The regular price per can of soda -/
def regular_price : ℝ := sorry

/-- The discounted price per can when purchased in 24-can cases -/
def discounted_price : ℝ := 0.9 * regular_price

/-- The price of 75 cans purchased in 24-can cases -/
def price_75_cans : ℝ := 10.125

theorem regular_price_is_0_15 : regular_price = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_regular_price_is_0_15_l3413_341390


namespace NUMINAMATH_CALUDE_inequality_proof_l3413_341324

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3413_341324


namespace NUMINAMATH_CALUDE_valid_numbers_characterization_l3413_341306

def is_valid (n : ℕ) : Prop :=
  n ≥ 10 ∧ (100 * (n / 10) + n % 10) % n = 0

def S : Set ℕ := {10, 20, 30, 40, 50, 60, 70, 80, 90, 15, 18, 45}

theorem valid_numbers_characterization :
  ∀ n : ℕ, is_valid n ↔ n ∈ S :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_characterization_l3413_341306


namespace NUMINAMATH_CALUDE_one_in_set_zero_one_l3413_341388

theorem one_in_set_zero_one : 1 ∈ ({0, 1} : Set ℕ) := by sorry

end NUMINAMATH_CALUDE_one_in_set_zero_one_l3413_341388


namespace NUMINAMATH_CALUDE_missing_number_proof_l3413_341325

theorem missing_number_proof (x : ℝ) : x * 240 = 173 * 240 → x = 173 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l3413_341325


namespace NUMINAMATH_CALUDE_complement_intersection_equiv_complement_union_l3413_341382

universe u

theorem complement_intersection_equiv_complement_union {U : Type u} (M N : Set U) :
  ∀ x : U, x ∈ (M ∩ N)ᶜ ↔ x ∈ Mᶜ ∪ Nᶜ := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equiv_complement_union_l3413_341382


namespace NUMINAMATH_CALUDE_find_divisor_l3413_341384

def is_divisor (n : ℕ) (d : ℕ) : Prop :=
  (n / d : ℚ) + 8 = 61

theorem find_divisor :
  ∃ (d : ℕ), is_divisor 265 d ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3413_341384


namespace NUMINAMATH_CALUDE_square_perimeters_sum_l3413_341391

theorem square_perimeters_sum (x y : ℝ) (h1 : x^2 + y^2 = 125) (h2 : x^2 - y^2 = 65) :
  4*x + 4*y = 60 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeters_sum_l3413_341391


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3413_341308

def N : ℕ := 64 * 45 * 91 * 49

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors N : ℚ) / (sum_even_divisors N : ℚ) = 1 / 126 := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3413_341308


namespace NUMINAMATH_CALUDE_lcm_inequality_l3413_341313

theorem lcm_inequality (m n : ℕ) (h1 : 0 < m) (h2 : m < n) :
  Nat.lcm m n + Nat.lcm (m + 1) (n + 1) ≥ 2 * m * Real.sqrt n :=
by sorry

end NUMINAMATH_CALUDE_lcm_inequality_l3413_341313


namespace NUMINAMATH_CALUDE_student_guinea_pig_difference_l3413_341335

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 22

/-- The number of guinea pigs in each classroom -/
def guinea_pigs_per_classroom : ℕ := 3

/-- Theorem stating the difference between total students and total guinea pigs -/
theorem student_guinea_pig_difference :
  num_classrooms * students_per_classroom - num_classrooms * guinea_pigs_per_classroom = 95 :=
by sorry

end NUMINAMATH_CALUDE_student_guinea_pig_difference_l3413_341335


namespace NUMINAMATH_CALUDE_last_two_digits_problem_l3413_341342

theorem last_two_digits_problem (x y : ℕ+) (h1 : x ≠ y) (h2 : (x : ℚ)⁻¹ + (y : ℚ)⁻¹ = 2/13) :
  (x.val ^ y.val + y.val ^ x.val) % 100 = 74 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_problem_l3413_341342


namespace NUMINAMATH_CALUDE_roots_bound_implies_b_bound_l3413_341398

-- Define the function f(x) = x^2 + ax + b
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- Define the theorem
theorem roots_bound_implies_b_bound
  (a b x₁ x₂ : ℝ)
  (h1 : f a b x₁ = 0)  -- x₁ is a root of f
  (h2 : f a b x₂ = 0)  -- x₂ is a root of f
  (h3 : x₁ ≠ x₂)       -- The roots are distinct
  (h4 : |x₁| + |x₂| ≤ 2) :
  b ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_roots_bound_implies_b_bound_l3413_341398


namespace NUMINAMATH_CALUDE_problem_figure_perimeter_l3413_341389

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the figure described in the problem -/
structure Figure where
  bottom_row : Vector Rectangle 3
  middle_square : Rectangle
  side_rectangles : Vector Rectangle 2

/-- The figure described in the problem -/
def problem_figure : Figure := {
  bottom_row := ⟨[{width := 1, height := 1}, {width := 1, height := 1}, {width := 1, height := 1}], rfl⟩
  middle_square := {width := 1, height := 1}
  side_rectangles := ⟨[{width := 1, height := 2}, {width := 1, height := 2}], rfl⟩
}

/-- Calculates the perimeter of the given figure -/
def perimeter (f : Figure) : ℕ :=
  sorry

/-- Theorem stating that the perimeter of the problem figure is 13 -/
theorem problem_figure_perimeter : perimeter problem_figure = 13 :=
  sorry

end NUMINAMATH_CALUDE_problem_figure_perimeter_l3413_341389


namespace NUMINAMATH_CALUDE_students_not_in_biology_l3413_341311

theorem students_not_in_biology (total_students : ℕ) (enrolled_percentage : ℚ) 
  (h1 : total_students = 880)
  (h2 : enrolled_percentage = 40 / 100) :
  (total_students : ℚ) * (1 - enrolled_percentage) = 528 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l3413_341311


namespace NUMINAMATH_CALUDE_square_difference_fourth_power_l3413_341357

theorem square_difference_fourth_power : (7^2 - 5^2)^4 = 331776 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fourth_power_l3413_341357


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_26_l3413_341333

theorem largest_four_digit_congruent_to_17_mod_26 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n ≡ 17 [ZMOD 26] → n ≤ 9972 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_26_l3413_341333


namespace NUMINAMATH_CALUDE_total_puppies_adopted_l3413_341397

def puppies_week1 : ℕ := 20

def puppies_week2 : ℕ := (2 * puppies_week1) / 5

def puppies_week3 : ℕ := (3 * puppies_week2) / 8

def puppies_week4 : ℕ := 2 * puppies_week2

def puppies_week5 : ℕ := puppies_week1 + 10

def puppies_week6 : ℕ := 2 * puppies_week3 - 5

def puppies_week7 : ℕ := 2 * puppies_week6

def puppies_week8 : ℕ := (7 * puppies_week6) / 4

def puppies_week9 : ℕ := (3 * puppies_week8) / 2

def puppies_week10 : ℕ := (9 * puppies_week1) / 4

def puppies_week11 : ℕ := (5 * puppies_week10) / 6

theorem total_puppies_adopted : 
  puppies_week1 + puppies_week2 + puppies_week3 + puppies_week4 + 
  puppies_week5 + puppies_week6 + puppies_week7 + puppies_week8 + 
  puppies_week9 + puppies_week10 + puppies_week11 = 164 := by
  sorry

end NUMINAMATH_CALUDE_total_puppies_adopted_l3413_341397


namespace NUMINAMATH_CALUDE_min_value_expression_l3413_341301

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (4 * z) / (2 * x + y) + (4 * x) / (y + 2 * z) + y / (x + z) ≥ 3 ∧
  ((4 * z) / (2 * x + y) + (4 * x) / (y + 2 * z) + y / (x + z) = 3 ↔ 2 * x = y ∧ y = 2 * z) := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3413_341301


namespace NUMINAMATH_CALUDE_correct_calculation_l3413_341309

theorem correct_calculation (a b : ℝ) : 3 * a * b - 2 * a * b = a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3413_341309


namespace NUMINAMATH_CALUDE_fifteen_choose_three_l3413_341393

theorem fifteen_choose_three : 
  Nat.choose 15 3 = 455 := by sorry

end NUMINAMATH_CALUDE_fifteen_choose_three_l3413_341393


namespace NUMINAMATH_CALUDE_opponent_total_score_l3413_341379

def TeamScores : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def LostGames (scores : List ℕ) : List ℕ := 
  scores.filter (λ x => x % 2 = 1 ∧ x ≤ 13)

def WonGames (scores : List ℕ) (lostGames : List ℕ) : List ℕ :=
  scores.filter (λ x => x ∉ lostGames)

def OpponentScoresInLostGames (lostGames : List ℕ) : List ℕ :=
  lostGames.map (λ x => x + 1)

def OpponentScoresInWonGames (wonGames : List ℕ) : List ℕ :=
  wonGames.map (λ x => x / 2)

theorem opponent_total_score :
  let lostGames := LostGames TeamScores
  let wonGames := WonGames TeamScores lostGames
  let opponentLostScores := OpponentScoresInLostGames lostGames
  let opponentWonScores := OpponentScoresInWonGames wonGames
  (opponentLostScores.sum + opponentWonScores.sum) = 75 :=
sorry

end NUMINAMATH_CALUDE_opponent_total_score_l3413_341379


namespace NUMINAMATH_CALUDE_max_small_boxes_in_large_box_l3413_341378

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the large box -/
def largeBox : BoxDimensions :=
  { length := 12, width := 14, height := 16 }

/-- The dimensions of the small box -/
def smallBox : BoxDimensions :=
  { length := 3, width := 7, height := 2 }

/-- Theorem stating the maximum number of small boxes that fit into the large box -/
theorem max_small_boxes_in_large_box :
  boxVolume largeBox / boxVolume smallBox = 64 := by
  sorry

end NUMINAMATH_CALUDE_max_small_boxes_in_large_box_l3413_341378


namespace NUMINAMATH_CALUDE_theater_ticket_pricing_l3413_341369

/-- Theorem: Theater Ticket Pricing --/
theorem theater_ticket_pricing
  (total_tickets : ℕ)
  (total_cost : ℕ)
  (orchestra_cost : ℕ)
  (balcony_surplus : ℕ)
  (h1 : total_tickets = 370)
  (h2 : total_cost = 3320)
  (h3 : orchestra_cost = 12)
  (h4 : balcony_surplus = 190)
  : ∃ (balcony_cost : ℕ),
    balcony_cost = 8 ∧
    balcony_cost * (total_tickets - (total_tickets - balcony_surplus) / 2) +
    orchestra_cost * ((total_tickets - balcony_surplus) / 2) = total_cost :=
by sorry


end NUMINAMATH_CALUDE_theater_ticket_pricing_l3413_341369


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l3413_341343

/-- The quadratic function -/
def f (x : ℝ) : ℝ := -x^2 + 6*x + 3

/-- The x-coordinate of the vertex -/
def h : ℝ := 3

/-- The y-coordinate of the vertex -/
def k : ℝ := 12

/-- Theorem: The vertex of the quadratic function f(x) = -x^2 + 6x + 3 is at (3, 12) -/
theorem vertex_of_quadratic :
  (∀ x : ℝ, f x = -(x - h)^2 + k) ∧ f h = k :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l3413_341343


namespace NUMINAMATH_CALUDE_sum_of_irrationals_can_be_rational_l3413_341383

theorem sum_of_irrationals_can_be_rational :
  ∃ (x y : ℝ), Irrational x ∧ Irrational y ∧ ∃ (q : ℚ), x + y = q := by
  sorry

end NUMINAMATH_CALUDE_sum_of_irrationals_can_be_rational_l3413_341383


namespace NUMINAMATH_CALUDE_consecutive_odd_squares_same_digit_l3413_341396

theorem consecutive_odd_squares_same_digit : ∃! (n : ℕ), 
  (∃ (d : ℕ), d ∈ Finset.range 10 ∧ 
    (n - 2)^2 + n^2 + (n + 2)^2 = 1111 * d) ∧
  Odd n ∧ n = 43 := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_squares_same_digit_l3413_341396


namespace NUMINAMATH_CALUDE_fo_greater_than_di_l3413_341376

-- Define the points
variable (F I D O : ℝ × ℝ)

-- Define the quadrilateral FIDO
def is_convex_quadrilateral (F I D O : ℝ × ℝ) : Prop := sorry

-- Define the length of a line segment
def length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the angle between two line segments
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem fo_greater_than_di 
  (h_convex : is_convex_quadrilateral F I D O)
  (h_equal_sides : length F I = length D O)
  (h_fi_greater : length F I > length D I)
  (h_equal_angles : angle F I O = angle D I O) :
  length F O > length D I :=
sorry

end NUMINAMATH_CALUDE_fo_greater_than_di_l3413_341376


namespace NUMINAMATH_CALUDE_power_tower_at_three_l3413_341315

theorem power_tower_at_three : (3^3)^(3^(3^3)) = 27^(3^27) := by sorry

end NUMINAMATH_CALUDE_power_tower_at_three_l3413_341315


namespace NUMINAMATH_CALUDE_copper_percentage_second_alloy_l3413_341322

/-- Calculates the percentage of copper in the second alloy -/
theorem copper_percentage_second_alloy 
  (desired_percentage : Real) 
  (first_alloy_percentage : Real)
  (first_alloy_weight : Real)
  (total_weight : Real) :
  let second_alloy_weight := total_weight - first_alloy_weight
  let desired_copper := desired_percentage * total_weight / 100
  let first_alloy_copper := first_alloy_percentage * first_alloy_weight / 100
  let second_alloy_copper := desired_copper - first_alloy_copper
  second_alloy_copper / second_alloy_weight * 100 = 21 :=
by
  sorry

#check copper_percentage_second_alloy 19.75 18 45 108

end NUMINAMATH_CALUDE_copper_percentage_second_alloy_l3413_341322


namespace NUMINAMATH_CALUDE_equation_roots_l3413_341361

theorem equation_roots : 
  let S := {x : ℝ | 0 < x ∧ x < 1 ∧ 8 * x * (2 * x^2 - 1) * (8 * x^4 - 8 * x^2 + 1) = 1}
  S = {Real.cos (π / 9), Real.cos (π / 3), Real.cos (2 * π / 7)} := by sorry

end NUMINAMATH_CALUDE_equation_roots_l3413_341361


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l3413_341316

theorem smallest_constant_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∃ p : ℝ, ∀ a b : ℝ, 0 < a → 0 < b →
    Real.sqrt (a * b) - (2 * a * b) / (a + b) ≤ p * ((a + b) / 2 - Real.sqrt (a * b))) ∧
  (∀ p : ℝ, (∀ a b : ℝ, 0 < a → 0 < b →
    Real.sqrt (a * b) - (2 * a * b) / (a + b) ≤ p * ((a + b) / 2 - Real.sqrt (a * b))) →
    1 ≤ p) ∧
  (∀ a b : ℝ, 0 < a → 0 < b →
    Real.sqrt (a * b) - (2 * a * b) / (a + b) ≤ (a + b) / 2 - Real.sqrt (a * b)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l3413_341316


namespace NUMINAMATH_CALUDE_simplify_fraction_l3413_341307

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3413_341307


namespace NUMINAMATH_CALUDE_unique_odd_pair_divisibility_l3413_341334

theorem unique_odd_pair_divisibility : 
  ∀ (a b : ℤ), 
    Odd a → Odd b →
    (∃ (c : ℕ), ∀ (n : ℕ), ∃ (k : ℤ), (c^n + 1 : ℤ) = k * (2^n * a + b)) →
    a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_odd_pair_divisibility_l3413_341334


namespace NUMINAMATH_CALUDE_function_inequality_solutions_l3413_341348

noncomputable def f (x : ℝ) : ℝ := (Real.log (2 * x)) / x

theorem function_inequality_solutions (a : ℝ) :
  (∃! (s : Finset ℤ), s.card = 2 ∧ ∀ x : ℤ, (x : ℝ) > 0 → (x ∈ s ↔ f x ^ 2 + a * f x > 0)) ↔
  a ∈ Set.Ioo (-Real.log 2) (-1/3 * Real.log 6) ∪ {-1/3 * Real.log 6} :=
sorry

end NUMINAMATH_CALUDE_function_inequality_solutions_l3413_341348


namespace NUMINAMATH_CALUDE_inverse_of_5_mod_31_l3413_341355

theorem inverse_of_5_mod_31 : ∃ x : ℕ, x ≤ 30 ∧ (5 * x) % 31 = 1 :=
by
  use 25
  sorry

end NUMINAMATH_CALUDE_inverse_of_5_mod_31_l3413_341355


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3413_341392

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → 
    ∃ t : ℝ, (b*x + a*y = 0 ∨ b*x - a*y = 0) ∧ 
    x^2 + y^2 - 6*x + 5 = t^2) →
  (3 : ℝ)^2 = a^2 - b^2 →
  a^2 = 5 ∧ b^2 = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3413_341392


namespace NUMINAMATH_CALUDE_tangent_line_length_l3413_341354

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 4*y + 7 = 0

-- Define point P
def P : ℝ × ℝ := (-2, 5)

-- Define the tangent line (abstractly, as we don't know its equation)
def tangent_line (Q : ℝ × ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ (x y : ℝ), y = m*x + b ∧ circle_equation x y → (x, y) = Q

-- Theorem statement
theorem tangent_line_length :
  ∃ (Q : ℝ × ℝ), circle_equation Q.1 Q.2 ∧ tangent_line Q →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_length_l3413_341354


namespace NUMINAMATH_CALUDE_eva_test_probability_l3413_341365

theorem eva_test_probability (p_history : ℝ) (p_geography : ℝ) 
  (h_history : p_history = 5/9)
  (h_geography : p_geography = 1/3)
  (h_independent : True) -- We don't need to define independence formally for this statement
  : (1 - p_history) * (1 - p_geography) = 8/27 := by
  sorry

end NUMINAMATH_CALUDE_eva_test_probability_l3413_341365


namespace NUMINAMATH_CALUDE_land_price_per_acre_l3413_341353

theorem land_price_per_acre (total_acres : ℕ) (num_lots : ℕ) (price_per_lot : ℕ) : 
  total_acres = 4 →
  num_lots = 9 →
  price_per_lot = 828 →
  (num_lots * price_per_lot) / total_acres = 1863 := by
sorry

end NUMINAMATH_CALUDE_land_price_per_acre_l3413_341353


namespace NUMINAMATH_CALUDE_mono_increasing_sufficient_not_necessary_l3413_341372

open Set
open Function

-- Define a monotonically increasing function
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Statement B
def StatementB (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, x₁ < x₂ ∧ f x₁ < f x₂

-- Theorem to prove
theorem mono_increasing_sufficient_not_necessary :
  (∀ f : ℝ → ℝ, MonoIncreasing f → StatementB f) ∧
  (∃ g : ℝ → ℝ, ¬MonoIncreasing g ∧ StatementB g) :=
by sorry

end NUMINAMATH_CALUDE_mono_increasing_sufficient_not_necessary_l3413_341372


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l3413_341332

theorem modulo_eleven_residue : (255 + 6 * 41 + 8 * 154 + 5 * 18) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l3413_341332


namespace NUMINAMATH_CALUDE_parabola_intersection_l3413_341386

-- Define the parabola
def parabola (k x : ℝ) : ℝ := x^2 - (k-1)*x - 3*k - 2

-- Define the intersection points
def α (k : ℝ) : ℝ := sorry
def β (k : ℝ) : ℝ := sorry

-- Theorem statement
theorem parabola_intersection (k : ℝ) : 
  (parabola k (α k) = 0) ∧ 
  (parabola k (β k) = 0) ∧ 
  ((α k)^2 + (β k)^2 = 17) → 
  k = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l3413_341386


namespace NUMINAMATH_CALUDE_sons_age_l3413_341380

/-- Proves that the son's age is 30 given the conditions of the problem -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 35 →
  father_age + 5 = 2 * (son_age + 5) →
  son_age = 30 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l3413_341380


namespace NUMINAMATH_CALUDE_albert_pizza_count_l3413_341310

def pizza_problem (large_pizzas small_pizzas : ℕ) 
  (slices_per_large slices_per_small total_slices : ℕ) : Prop :=
  large_pizzas = 2 ∧ 
  slices_per_large = 16 ∧ 
  slices_per_small = 8 ∧ 
  total_slices = 48 ∧
  small_pizzas * slices_per_small = total_slices - (large_pizzas * slices_per_large)

theorem albert_pizza_count : 
  ∃ (large_pizzas small_pizzas slices_per_large slices_per_small total_slices : ℕ),
    pizza_problem large_pizzas small_pizzas slices_per_large slices_per_small total_slices ∧ 
    small_pizzas = 2 := by
  sorry

end NUMINAMATH_CALUDE_albert_pizza_count_l3413_341310


namespace NUMINAMATH_CALUDE_fourth_side_length_l3413_341323

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The lengths of the four sides of the quadrilateral -/
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- The theorem statement -/
theorem fourth_side_length
  (q : InscribedQuadrilateral)
  (h1 : q.radius = 150)
  (h2 : q.side1 = 200)
  (h3 : q.side2 = 200)
  (h4 : q.side3 = 100) :
  q.side4 = 300 := by
  sorry


end NUMINAMATH_CALUDE_fourth_side_length_l3413_341323


namespace NUMINAMATH_CALUDE_alyssas_turnips_l3413_341360

theorem alyssas_turnips (keith_turnips total_turnips : ℕ) 
  (h1 : keith_turnips = 6)
  (h2 : total_turnips = 15) :
  total_turnips - keith_turnips = 9 :=
by sorry

end NUMINAMATH_CALUDE_alyssas_turnips_l3413_341360


namespace NUMINAMATH_CALUDE_maria_coffee_order_l3413_341312

-- Define the variables
def visits_per_day : ℕ := 2
def total_cups_per_day : ℕ := 6

-- Define the function to calculate cups per visit
def cups_per_visit : ℕ := total_cups_per_day / visits_per_day

-- Theorem statement
theorem maria_coffee_order :
  cups_per_visit = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_maria_coffee_order_l3413_341312


namespace NUMINAMATH_CALUDE_triangle_shape_l3413_341362

theorem triangle_shape (a b : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → 
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a * Real.cos A = b * Real.cos B →
  (A = B ∨ A + B = π / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_l3413_341362


namespace NUMINAMATH_CALUDE_binomial_coeff_not_coprime_l3413_341351

theorem binomial_coeff_not_coprime (k m n : ℕ) (h1 : 0 < k) (h2 : k < m) (h3 : m < n) :
  ¬(Nat.gcd (Nat.choose n k) (Nat.choose n m) = 1) :=
sorry

end NUMINAMATH_CALUDE_binomial_coeff_not_coprime_l3413_341351


namespace NUMINAMATH_CALUDE_turtle_ratio_l3413_341395

def total_turtles : ℕ := 42
def turtles_on_sand : ℕ := 28

theorem turtle_ratio : 
  (total_turtles - turtles_on_sand) / total_turtles = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_turtle_ratio_l3413_341395


namespace NUMINAMATH_CALUDE_can_form_triangle_l3413_341364

/-- Triangle Inequality Theorem: The sum of the lengths of any two sides of a 
    triangle must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Given lengths 5, 3, and 4 can form a triangle. -/
theorem can_form_triangle : triangle_inequality 5 3 4 := by
  sorry

end NUMINAMATH_CALUDE_can_form_triangle_l3413_341364


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3413_341327

theorem polynomial_factorization (x : ℝ) :
  x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x - 1)^4 * (x + 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3413_341327


namespace NUMINAMATH_CALUDE_isabel_bouquets_l3413_341321

def flowers_to_bouquets (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) : ℕ :=
  (initial_flowers - wilted_flowers) / flowers_per_bouquet

theorem isabel_bouquets :
  flowers_to_bouquets 66 8 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_isabel_bouquets_l3413_341321


namespace NUMINAMATH_CALUDE_aunt_marge_candy_distribution_l3413_341302

theorem aunt_marge_candy_distribution (total_candy : ℕ) 
  (kate_candy : ℕ) (robert_candy : ℕ) (mary_candy : ℕ) (bill_candy : ℕ) : 
  total_candy = 20 ∧ 
  robert_candy = kate_candy + 2 ∧
  bill_candy = mary_candy - 6 ∧
  mary_candy = robert_candy + 2 ∧
  kate_candy = bill_candy + 2 ∧
  total_candy = kate_candy + robert_candy + mary_candy + bill_candy →
  kate_candy = 4 := by
sorry

end NUMINAMATH_CALUDE_aunt_marge_candy_distribution_l3413_341302


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_l3413_341375

/-- Given two nonconstant geometric sequences with different common ratios,
    prove that the sum of their common ratios is equal to k. -/
theorem sum_of_common_ratios (k a₂ a₃ b₂ b₃ : ℝ) 
  (hk : k ≠ 0)
  (ha : a₂ ≠ k ∧ a₃ ≠ a₂)  -- Ensures (k, a₂, a₃) is nonconstant
  (hb : b₂ ≠ k ∧ b₃ ≠ b₂)  -- Ensures (k, b₂, b₃) is nonconstant
  (hdiff : a₂ / k ≠ b₂ / k)  -- Ensures different common ratios
  (heq : a₃ - b₃ = k^2 * (a₂ - b₂)) :
  ∃ p q : ℝ, p ≠ q ∧ 
    a₃ = k * p^2 ∧ 
    b₃ = k * q^2 ∧ 
    a₂ = k * p ∧ 
    b₂ = k * q ∧ 
    p + q = k :=
by sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_l3413_341375


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3413_341339

theorem cube_volume_problem (V₁ : ℝ) (A₂ : ℝ) : 
  V₁ = 8 → 
  A₂ = 3 * (6 * (V₁^(1/3))^2) → 
  (A₂ / 6)^(3/2) = 24 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3413_341339


namespace NUMINAMATH_CALUDE_abs_neg_eleven_l3413_341359

theorem abs_neg_eleven : |(-11 : ℤ)| = 11 := by sorry

end NUMINAMATH_CALUDE_abs_neg_eleven_l3413_341359


namespace NUMINAMATH_CALUDE_total_courses_is_200_l3413_341349

/-- The number of college courses attended by Max -/
def max_courses : ℕ := 40

/-- The number of college courses attended by Sid -/
def sid_courses : ℕ := 4 * max_courses

/-- The total number of college courses attended by Max and Sid -/
def total_courses : ℕ := max_courses + sid_courses

/-- Theorem stating that the total number of courses attended by Max and Sid is 200 -/
theorem total_courses_is_200 : total_courses = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_courses_is_200_l3413_341349


namespace NUMINAMATH_CALUDE_time_to_reach_B_after_second_meeting_l3413_341303

-- Define the variables
variable (S : ℝ) -- Total distance between A and B
variable (v_A v_B : ℝ) -- Speeds of A and B
variable (t : ℝ) -- Time taken by B to catch up with A

-- Define the theorem
theorem time_to_reach_B_after_second_meeting : 
  -- A starts 48 minutes (4/5 hours) before B
  v_A * (t + 4/5) = 2/3 * S →
  -- B catches up with A when A has traveled 2/3 of the distance
  v_B * t = 2/3 * S →
  -- They meet again 6 minutes (1/10 hour) after B leaves B
  v_A * (t + 4/5 + 1/2 * t + 1/10) + 1/10 * v_B = S →
  -- The time it takes for A to reach B after meeting B again is 12 minutes (1/5 hour)
  1/5 = S / v_A - (t + 4/5 + 1/2 * t + 1/10) := by
  sorry

end NUMINAMATH_CALUDE_time_to_reach_B_after_second_meeting_l3413_341303


namespace NUMINAMATH_CALUDE_fraction_simplification_l3413_341320

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) : (x - y) / (y - x) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3413_341320


namespace NUMINAMATH_CALUDE_focus_of_given_parabola_l3413_341363

/-- A parabola is defined by the equation y = ax^2 + bx + c where a ≠ 0 -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- The given parabola y = 4x^2 + 8x - 5 -/
def given_parabola : Parabola :=
  { a := 4
    b := 8
    c := -5
    a_nonzero := by norm_num }

theorem focus_of_given_parabola :
  focus given_parabola = (-1, -143/16) := by sorry

end NUMINAMATH_CALUDE_focus_of_given_parabola_l3413_341363


namespace NUMINAMATH_CALUDE_largest_x_abs_value_equation_l3413_341326

theorem largest_x_abs_value_equation : 
  (∃ (x : ℝ), |x - 8| = 15 ∧ ∀ (y : ℝ), |y - 8| = 15 → y ≤ x) → 
  (∃ (x : ℝ), |x - 8| = 15 ∧ ∀ (y : ℝ), |y - 8| = 15 → y ≤ 23) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_abs_value_equation_l3413_341326


namespace NUMINAMATH_CALUDE_blue_cube_faces_l3413_341350

theorem blue_cube_faces (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_cube_faces_l3413_341350


namespace NUMINAMATH_CALUDE_min_squares_to_exceed_500_l3413_341367

def square (n : ℕ) : ℕ := n * n

def repeated_square (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => square (repeated_square n k)

theorem min_squares_to_exceed_500 :
  (∃ k : ℕ, repeated_square 2 k > 500) ∧
  (∀ k : ℕ, k < 4 → repeated_square 2 k ≤ 500) ∧
  (repeated_square 2 4 > 500) :=
by sorry

end NUMINAMATH_CALUDE_min_squares_to_exceed_500_l3413_341367


namespace NUMINAMATH_CALUDE_average_study_time_difference_l3413_341374

/-- The daily differences in study time (in minutes) between Mira and Clara over a week -/
def study_time_differences : List Int := [15, 0, -15, 25, 5, -5, 10]

/-- The number of days in the week -/
def days_in_week : Nat := 7

/-- Theorem stating that the average difference in daily study time is 5 minutes -/
theorem average_study_time_difference :
  (study_time_differences.sum : ℚ) / days_in_week = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l3413_341374


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3413_341399

/-- The repeating decimal 0.567567567... expressed as a rational number -/
def repeating_decimal : ℚ := 567 / 999

theorem repeating_decimal_equals_fraction : repeating_decimal = 21 / 37 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3413_341399


namespace NUMINAMATH_CALUDE_log_sum_approximation_l3413_341330

theorem log_sum_approximation : 
  let x := Real.log 3 / Real.log 10 + 3 * Real.log 4 / Real.log 10 + 
           2 * Real.log 5 / Real.log 10 + 4 * Real.log 2 / Real.log 10 + 
           Real.log 9 / Real.log 10
  ∃ ε > 0, |x - 5.8399| < ε :=
by sorry

end NUMINAMATH_CALUDE_log_sum_approximation_l3413_341330


namespace NUMINAMATH_CALUDE_remainder_problem_l3413_341366

theorem remainder_problem (d : ℕ) (r : ℕ) (h1 : d > 1) 
  (h2 : 1023 % d = r) (h3 : 1386 % d = r) (h4 : 2151 % d = r) : 
  d - r = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3413_341366


namespace NUMINAMATH_CALUDE_part1_part2_l3413_341385

-- Define the given condition
def condition (x y : ℝ) : Prop :=
  |x - 4 - 2 * Real.sqrt 2| + Real.sqrt (y - 4 + 2 * Real.sqrt 2) = 0

-- Define a rhombus
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ

-- Theorem for part 1
theorem part1 {x y : ℝ} (h : condition x y) :
  x * y^2 - x^2 * y = -32 * Real.sqrt 2 := by
  sorry

-- Theorem for part 2
theorem part2 {x y : ℝ} (h : condition x y) :
  let r : Rhombus := ⟨x, y⟩
  (r.diagonal1 * r.diagonal2 / 2 = 4) ∧
  (r.diagonal1 * r.diagonal2 / (4 * Real.sqrt 3) = 2 * Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_part1_part2_l3413_341385


namespace NUMINAMATH_CALUDE_find_divisor_l3413_341329

theorem find_divisor (x : ℕ) (h_x : x = 75) :
  ∃ D : ℕ,
    (∃ Q R : ℕ, x = D * Q + R ∧ R < D ∧ Q = (x % 34) + 8) ∧
    (∀ D' : ℕ, D' < D → ¬(∃ Q' R' : ℕ, x = D' * Q' + R' ∧ R' < D' ∧ Q' = (x % 34) + 8)) ∧
    D = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3413_341329


namespace NUMINAMATH_CALUDE_range_of_x_for_inequality_l3413_341373

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem range_of_x_for_inequality (x : ℝ) :
  (∀ m : ℝ, m ∈ Set.Icc (-2) 2 → f (m*x - 2) + f x < 0) →
  x ∈ Set.Ioo (-2) (2/3) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_for_inequality_l3413_341373


namespace NUMINAMATH_CALUDE_cubic_roots_product_l3413_341328

theorem cubic_roots_product (α₁ α₂ α₃ : ℂ) : 
  (5 * α₁^3 - 6 * α₁^2 + 7 * α₁ + 8 = 0) ∧ 
  (5 * α₂^3 - 6 * α₂^2 + 7 * α₂ + 8 = 0) ∧ 
  (5 * α₃^3 - 6 * α₃^2 + 7 * α₃ + 8 = 0) →
  (α₁^2 + α₁*α₂ + α₂^2) * (α₂^2 + α₂*α₃ + α₃^2) * (α₁^2 + α₁*α₃ + α₃^2) = 764/625 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_product_l3413_341328


namespace NUMINAMATH_CALUDE_min_p_plus_q_l3413_341338

theorem min_p_plus_q (p q : ℕ+) (h : 162 * p = q^3) : 
  ∀ (p' q' : ℕ+), 162 * p' = q'^3 → p + q ≤ p' + q' :=
sorry

end NUMINAMATH_CALUDE_min_p_plus_q_l3413_341338


namespace NUMINAMATH_CALUDE_divisor_sum_condition_l3413_341358

def d (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_sum_condition (n : ℕ) : n ≥ 3 → (d (n - 1) + d n + d (n + 1) ≤ 8 ↔ n = 3 ∨ n = 4 ∨ n = 6) := by
  sorry

end NUMINAMATH_CALUDE_divisor_sum_condition_l3413_341358


namespace NUMINAMATH_CALUDE_pyramid_volume_l3413_341370

-- Define the rectangular parallelepiped
structure Parallelepiped where
  AB : ℝ
  BC : ℝ
  CG : ℝ

-- Define the rectangular pyramid
structure Pyramid where
  base : ℝ -- Area of the base BDFE
  height : ℝ -- Height of the pyramid (XM)

-- Define the problem
theorem pyramid_volume (p : Parallelepiped) (pyr : Pyramid) : 
  p.AB = 4 → 
  p.BC = 2 → 
  p.CG = 5 → 
  pyr.base = p.AB * p.BC → 
  pyr.height = p.CG → 
  (1/3 : ℝ) * pyr.base * pyr.height = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3413_341370
