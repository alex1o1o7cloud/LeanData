import Mathlib

namespace NUMINAMATH_CALUDE_pages_per_day_l942_94258

theorem pages_per_day (chapters : ℕ) (total_pages : ℕ) (days : ℕ) 
  (h1 : chapters = 41) 
  (h2 : total_pages = 450) 
  (h3 : days = 30) :
  total_pages / days = 15 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_day_l942_94258


namespace NUMINAMATH_CALUDE_circle_equation_and_tangent_and_symmetry_l942_94279

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 5

-- Define the line that contains the center of C
def center_line (x y : ℝ) : Prop := 2*x - y - 7 = 0

-- Define the points A and B where C intersects the y-axis
def point_A : ℝ × ℝ := (0, -4)
def point_B : ℝ × ℝ := (0, -2)

-- Define the tangent line l
def line_l (k x y : ℝ) : Prop := k*x - y + k = 0

-- Define the line l₁ for symmetry
def line_l1 (x y : ℝ) : Prop := y = 2*x + 1

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x + 22/5)^2 + (y - 1/5)^2 = 5

theorem circle_equation_and_tangent_and_symmetry :
  ∃ (k : ℝ),
    (∀ x y, circle_C x y ↔ (x - 2)^2 + (y + 3)^2 = 5) ∧
    (k = (-9 + Real.sqrt 65) / 4 ∨ k = (-9 - Real.sqrt 65) / 4) ∧
    (∀ x y, symmetric_circle x y ↔ (x + 22/5)^2 + (y - 1/5)^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_and_tangent_and_symmetry_l942_94279


namespace NUMINAMATH_CALUDE_abc_expression_value_l942_94291

theorem abc_expression_value (a b c : ℚ) 
  (ha : a^2 = 9)
  (hb : abs b = 4)
  (hc : c^3 = 27)
  (hab : a * b < 0)
  (hbc : b * c > 0) :
  a * b - b * c + c * a = -33 := by
sorry

end NUMINAMATH_CALUDE_abc_expression_value_l942_94291


namespace NUMINAMATH_CALUDE_unique_solution_l942_94283

theorem unique_solution : ∃! x : ℝ, x * 3 + 3 * 13 + 3 * 16 + 11 = 134 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l942_94283


namespace NUMINAMATH_CALUDE_sum_of_first_60_digits_l942_94215

/-- The decimal representation of 1/2222 -/
def decimal_rep : ℚ := 1 / 2222

/-- The repeating sequence in the decimal representation -/
def repeating_sequence : List ℕ := [0, 0, 0, 4, 5]

/-- The length of the repeating sequence -/
def sequence_length : ℕ := repeating_sequence.length

/-- The sum of digits in one repetition of the sequence -/
def sequence_sum : ℕ := repeating_sequence.sum

/-- The number of complete repetitions in the first 60 digits -/
def num_repetitions : ℕ := 60 / sequence_length

theorem sum_of_first_60_digits : 
  (num_repetitions * sequence_sum = 108) := by sorry

end NUMINAMATH_CALUDE_sum_of_first_60_digits_l942_94215


namespace NUMINAMATH_CALUDE_intersection_of_logarithmic_curves_l942_94261

theorem intersection_of_logarithmic_curves :
  ∃! x : ℝ, x > 0 ∧ 3 * Real.log x = Real.log (3 * x) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_logarithmic_curves_l942_94261


namespace NUMINAMATH_CALUDE_shift_left_theorem_l942_94287

/-- Represents a quadratic function y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original quadratic function y = x^2 -/
def original : QuadraticFunction := ⟨1, 0, 0⟩

/-- Shifts a quadratic function to the left by h units -/
def shift_left (f : QuadraticFunction) (h : ℝ) : QuadraticFunction :=
  ⟨f.a, f.b + 2 * f.a * h, f.c + f.b * h + f.a * h^2⟩

/-- The shifted quadratic function -/
def shifted : QuadraticFunction := shift_left original 1

theorem shift_left_theorem :
  shifted = ⟨1, 2, 1⟩ := by sorry

end NUMINAMATH_CALUDE_shift_left_theorem_l942_94287


namespace NUMINAMATH_CALUDE_fifth_largest_divisor_l942_94212

def n : ℕ := 1209600000

def is_fifth_largest_divisor (d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (a b c e : ℕ), a > b ∧ b > c ∧ c > d ∧ d > e ∧
    a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ e ∣ n ∧
    ∀ (x : ℕ), x ∣ n → (x ≤ e ∨ x = d ∨ x = c ∨ x = b ∨ x = a ∨ x = n))

theorem fifth_largest_divisor :
  is_fifth_largest_divisor 75600000 :=
sorry

end NUMINAMATH_CALUDE_fifth_largest_divisor_l942_94212


namespace NUMINAMATH_CALUDE_sin_angle_equality_l942_94237

theorem sin_angle_equality (α : Real) (h : Real.sin (π + α) = -1/2) : 
  Real.sin (4*π - α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_angle_equality_l942_94237


namespace NUMINAMATH_CALUDE_joan_socks_total_l942_94284

theorem joan_socks_total (total_socks : ℕ) (white_socks : ℕ) (blue_socks : ℕ) :
  white_socks = (2 : ℚ) / 3 * total_socks →
  blue_socks = total_socks - white_socks →
  blue_socks = 60 →
  total_socks = 180 := by
  sorry

end NUMINAMATH_CALUDE_joan_socks_total_l942_94284


namespace NUMINAMATH_CALUDE_petra_beads_removal_l942_94206

/-- Represents the number of blue beads Petra has initially -/
def initial_blue_beads : ℕ := 49

/-- Represents the number of red beads Petra has initially -/
def initial_red_beads : ℕ := 1

/-- Represents the total number of beads Petra has initially -/
def initial_total_beads : ℕ := initial_blue_beads + initial_red_beads

/-- Represents the number of beads Petra needs to remove -/
def beads_to_remove : ℕ := 40

/-- Represents the desired percentage of blue beads after removal -/
def desired_blue_percentage : ℚ := 90 / 100

theorem petra_beads_removal :
  let remaining_beads := initial_total_beads - beads_to_remove
  let remaining_blue_beads := initial_blue_beads - (beads_to_remove - initial_red_beads)
  (remaining_blue_beads : ℚ) / remaining_beads = desired_blue_percentage :=
sorry

end NUMINAMATH_CALUDE_petra_beads_removal_l942_94206


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l942_94294

-- Define the polynomial
def p (x y : ℝ) : ℝ := 3 * x * y^2 - 2 * y - 1

-- State the theorem
theorem polynomial_coefficients :
  (∃ a b c d : ℝ, ∀ x y : ℝ, p x y = a * x * y^2 + b * y + c * x + d) ∧
  (∀ a b c d : ℝ, (∀ x y : ℝ, p x y = a * x * y^2 + b * y + c * x + d) →
    b = -2 ∧ d = -1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_l942_94294


namespace NUMINAMATH_CALUDE_range_of_b_l942_94286

-- Define the circles and line
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O1 (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 4
def line_P (x y b : ℝ) : Prop := x + Real.sqrt 3 * y - b = 0

-- Define the condition for P satisfying PB = 2PA
def condition_P (x y : ℝ) : Prop := x^2 + y^2 + (8/3) * x - 16/3 = 0

-- Theorem statement
theorem range_of_b :
  ∀ b : ℝ, (∃! (p1 p2 : ℝ × ℝ), 
    p1 ≠ p2 ∧ 
    line_P p1.1 p1.2 b ∧ 
    line_P p2.1 p2.2 b ∧ 
    condition_P p1.1 p1.2 ∧ 
    condition_P p2.1 p2.2) ↔ 
  -20/3 < b ∧ b < 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_b_l942_94286


namespace NUMINAMATH_CALUDE_cricket_bat_cost_price_l942_94205

theorem cricket_bat_cost_price 
  (profit_A_to_B : ℝ) 
  (profit_B_to_C : ℝ) 
  (price_C : ℝ) 
  (h1 : profit_A_to_B = 0.20)
  (h2 : profit_B_to_C = 0.25)
  (h3 : price_C = 234) :
  ∃ (cost_price_A : ℝ), cost_price_A = 156 ∧
    price_C = (1 + profit_B_to_C) * ((1 + profit_A_to_B) * cost_price_A) := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_cost_price_l942_94205


namespace NUMINAMATH_CALUDE_cubic_polynomial_sum_l942_94281

/-- A cubic polynomial with real coefficients -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The roots of the polynomial -/
structure PolynomialRoots (w : ℂ) where
  root1 : ℂ := w - Complex.I
  root2 : ℂ := w - 3 * Complex.I
  root3 : ℂ := 2 * w + 2

/-- Theorem statement -/
theorem cubic_polynomial_sum (P : CubicPolynomial) (w : ℂ) 
  (h : ∀ z : ℂ, (z - (w - Complex.I)) * (z - (w - 3 * Complex.I)) * (z - (2 * w + 2)) = 
       z^3 + P.a * z^2 + P.b * z + P.c) :
  P.a + P.b + P.c = 22 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_sum_l942_94281


namespace NUMINAMATH_CALUDE_ellipse_k_range_l942_94277

/-- An ellipse with equation x^2 / (2-k) + y^2 / (2k-1) = 1 and foci on the y-axis has k in the range (1, 2) -/
theorem ellipse_k_range (k : ℝ) :
  (∀ x y : ℝ, x^2 / (2-k) + y^2 / (2*k-1) = 1) →  -- equation represents an ellipse
  (∃ c : ℝ, c > 0 ∧ ∀ x y : ℝ, x^2 / (2-k) + y^2 / (2*k-1) = 1 → y^2 ≥ c^2) →  -- foci on y-axis
  1 < k ∧ k < 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l942_94277


namespace NUMINAMATH_CALUDE_odot_one_four_odot_comm_l942_94223

-- Define the ⊙ operation for rational numbers
def odot (a b : ℚ) : ℚ := a - a * b + b + 3

-- Theorem: 1 ⊙ 4 = 4
theorem odot_one_four : odot 1 4 = 4 := by sorry

-- Theorem: ⊙ is commutative
theorem odot_comm (a b : ℚ) : odot a b = odot b a := by sorry

end NUMINAMATH_CALUDE_odot_one_four_odot_comm_l942_94223


namespace NUMINAMATH_CALUDE_min_value_of_x_plus_y_l942_94230

theorem min_value_of_x_plus_y (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : 9 / x + 1 / y = 1) : 
  x + y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 9 / x₀ + 1 / y₀ = 1 ∧ x₀ + y₀ = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_x_plus_y_l942_94230


namespace NUMINAMATH_CALUDE_binomial_square_constant_l942_94280

/-- If 9x^2 - 18x + c is the square of a binomial, then c = 9 -/
theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x, 9*x^2 - 18*x + c = (a*x + b)^2) → c = 9 := by
sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l942_94280


namespace NUMINAMATH_CALUDE_probability_of_selection_l942_94274

/-- Given a group of students where each student has an equal chance of being selected as the group leader,
    prove that the probability of a specific student (Xiao Li) being chosen is 1/5. -/
theorem probability_of_selection (total_students : ℕ) (xiao_li : Fin total_students) :
  total_students = 5 →
  (∀ (student : Fin total_students), ℚ) →
  (∃! (prob : Fin total_students → ℚ), ∀ (student : Fin total_students), prob student = 1 / total_students) →
  (∃ (prob : Fin total_students → ℚ), prob xiao_li = 1 / 5) :=
by sorry

end NUMINAMATH_CALUDE_probability_of_selection_l942_94274


namespace NUMINAMATH_CALUDE_complement_union_M_N_l942_94226

open Set

-- Define the universal set as ℝ
universe u
variable {α : Type u}

-- Define sets M and N
def M : Set ℝ := {x | x ≤ 0}
def N : Set ℝ := {x | x > 2}

-- State the theorem
theorem complement_union_M_N :
  (M ∪ N)ᶜ = {x : ℝ | 0 < x ∧ x ≤ 2} :=
by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l942_94226


namespace NUMINAMATH_CALUDE_muffin_division_l942_94282

theorem muffin_division (total_muffins : ℕ) (friends : ℕ) (muffins_per_person : ℕ) :
  total_muffins = 20 →
  friends = 4 →
  muffins_per_person * (friends + 1) = total_muffins →
  muffins_per_person = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_muffin_division_l942_94282


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l942_94214

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2), -(p.1))

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (8, -3)
  reflect_about_y_neg_x original_center = (-3, -8) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l942_94214


namespace NUMINAMATH_CALUDE_tangerine_sales_theorem_l942_94266

/-- Represents the daily sales data for a week -/
def sales_data : List Int := [300, -400, -200, 100, -600, 1200, 500]

/-- The planned daily sales amount in kilograms -/
def planned_daily_sales : Nat := 20000

/-- The selling price per kilogram in yuan -/
def selling_price : Nat := 6

/-- The express delivery cost and other expenses per kilogram in yuan -/
def expenses : Nat := 2

/-- The number of days in a week -/
def days_in_week : Nat := 7

theorem tangerine_sales_theorem :
  (List.maximum? sales_data).isSome ∧ 
  (List.minimum? sales_data).isSome →
  (∃ max min : Int, 
    (List.maximum? sales_data) = some max ∧
    (List.minimum? sales_data) = some min ∧
    max - min = 1800) ∧
  (planned_daily_sales * days_in_week + (List.sum sales_data)) * 
    (selling_price - expenses) = 563600 := by
  sorry

end NUMINAMATH_CALUDE_tangerine_sales_theorem_l942_94266


namespace NUMINAMATH_CALUDE_cubic_monotonic_and_odd_l942_94288

def f (x : ℝ) : ℝ := x^3

theorem cubic_monotonic_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, f (-x) = -f x) := by
sorry

end NUMINAMATH_CALUDE_cubic_monotonic_and_odd_l942_94288


namespace NUMINAMATH_CALUDE_digit_sum_properties_l942_94260

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem digit_sum_properties :
  (∀ N : ℕ, S N ≤ 8 * S (8 * N)) ∧
  (∀ r q : ℕ, ∃ c_k : ℚ, c_k > 0 ∧
    (∀ N : ℕ, S (2^r * 5^q * N) / S N ≥ c_k) ∧
    c_k = 1 / S (2^q * 5^r) ∧
    (∀ c : ℚ, c > c_k → ∃ N : ℕ, S (2^r * 5^q * N) / S N < c)) ∧
  (∀ k : ℕ, (∃ r q : ℕ, k = 2^r * 5^q) ∨
    (∀ c : ℚ, c > 0 → ∃ N : ℕ, S (k * N) / S N < c)) :=
sorry

end NUMINAMATH_CALUDE_digit_sum_properties_l942_94260


namespace NUMINAMATH_CALUDE_car_total_distance_l942_94221

/-- A car driving through a ring in a tunnel -/
structure CarInRing where
  /-- Number of right-hand turns in the ring -/
  turns : ℕ
  /-- Distance traveled after the 1st turn -/
  dist1 : ℝ
  /-- Distance traveled after the 2nd turn -/
  dist2 : ℝ
  /-- Distance traveled after the 3rd turn -/
  dist3 : ℝ

/-- The total distance driven by the car around the ring -/
def totalDistance (car : CarInRing) : ℝ :=
  car.dist1 + car.dist2 + car.dist3

/-- Theorem stating the total distance driven by the car -/
theorem car_total_distance (car : CarInRing) 
  (h1 : car.turns = 4)
  (h2 : car.dist1 = 5)
  (h3 : car.dist2 = 8)
  (h4 : car.dist3 = 10) : 
  totalDistance car = 23 := by
  sorry

end NUMINAMATH_CALUDE_car_total_distance_l942_94221


namespace NUMINAMATH_CALUDE_statue_weight_theorem_l942_94228

/-- Calculates the weight of a marble statue after a series of reductions --/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  let week1 := initial_weight * (1 - 0.35)
  let week2 := week1 * (1 - 0.20)
  let week3 := week2 * (1 - 0.05)^5
  let after_rain := week3 * (1 - 0.02)
  let week4 := after_rain * (1 - 0.08)
  let final := week4 * (1 - 0.25)
  final

/-- The weight of the final statue is approximately 136.04 kg --/
theorem statue_weight_theorem (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ |final_statue_weight 500 - 136.04| < δ ∧ δ < ε :=
sorry

end NUMINAMATH_CALUDE_statue_weight_theorem_l942_94228


namespace NUMINAMATH_CALUDE_evaluate_expression_l942_94224

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 5) : 
  y * (2 * y - 5 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l942_94224


namespace NUMINAMATH_CALUDE_x4_plus_y4_equals_47_l942_94231

theorem x4_plus_y4_equals_47 (x y : ℝ) 
  (h1 : x^2 + 1/x^2 = 7) 
  (h2 : x*y = 1) : 
  x^4 + y^4 = 47 := by
sorry

end NUMINAMATH_CALUDE_x4_plus_y4_equals_47_l942_94231


namespace NUMINAMATH_CALUDE_second_meeting_time_l942_94246

-- Define the pool and swimmers
def Pool : Type := Unit
def Swimmer : Type := Unit

-- Define the time to meet in the center
def time_to_center : ℝ := 1.5

-- Define the function to calculate the time for the second meeting
def time_to_second_meeting (p : Pool) (s1 s2 : Swimmer) : ℝ :=
  2 * time_to_center + time_to_center

-- Theorem statement
theorem second_meeting_time (p : Pool) (s1 s2 : Swimmer) :
  time_to_second_meeting p s1 s2 = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_second_meeting_time_l942_94246


namespace NUMINAMATH_CALUDE_oldest_child_age_l942_94217

theorem oldest_child_age (age1 age2 : ℕ) (avg : ℚ) :
  age1 = 6 →
  age2 = 9 →
  avg = 10 →
  (age1 + age2 + (3 * avg - age1 - age2 : ℚ) : ℚ) / 3 = avg →
  3 * avg - age1 - age2 = 15 :=
by sorry

end NUMINAMATH_CALUDE_oldest_child_age_l942_94217


namespace NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l942_94213

/-- Custom binary operation ⊗ -/
def otimes (a b : ℝ) : ℝ := a^2 - abs b

/-- Theorem stating that (-2) ⊗ (-1) = 3 -/
theorem otimes_neg_two_neg_one : otimes (-2) (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l942_94213


namespace NUMINAMATH_CALUDE_least_prime_factor_of_N_l942_94290

def N : ℕ := 10^2011 + 1

theorem least_prime_factor_of_N :
  (Nat.minFac N = 11) := by sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_N_l942_94290


namespace NUMINAMATH_CALUDE_find_A_l942_94216

theorem find_A : ∀ A : ℝ, 10 + A = 15 → A = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l942_94216


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l942_94273

theorem sphere_volume_ratio (S₁ S₂ S₃ V₁ V₂ V₃ : ℝ) :
  S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 →
  V₁ > 0 ∧ V₂ > 0 ∧ V₃ > 0 →
  S₂ / S₁ = 4 →
  S₃ / S₁ = 9 →
  (4 * π * (V₁ / (4/3 * π))^(2/3) = S₁) →
  (4 * π * (V₂ / (4/3 * π))^(2/3) = S₂) →
  (4 * π * (V₃ / (4/3 * π))^(2/3) = S₃) →
  V₁ + V₂ = (1/3) * V₃ := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l942_94273


namespace NUMINAMATH_CALUDE_range_of_a_l942_94271

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, -1 ≤ x₀ ∧ x₀ ≤ 1 ∧ 2 * a * x₀^2 + 2 * x₀ - 3 - a = 0) → 
  (a ≥ 1 ∨ a ≤ (-3 - Real.sqrt 7) / 2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l942_94271


namespace NUMINAMATH_CALUDE_batting_average_calculation_l942_94243

/-- Calculates the new batting average after a match -/
def newBattingAverage (currentAverage : ℚ) (matchesPlayed : ℕ) (runsScored : ℕ) : ℚ :=
  (currentAverage * matchesPlayed + runsScored) / (matchesPlayed + 1)

/-- Theorem: Given the conditions, the new batting average will be 54 -/
theorem batting_average_calculation (currentAverage : ℚ) (matchesPlayed : ℕ) (runsScored : ℕ)
  (h1 : currentAverage = 51)
  (h2 : matchesPlayed = 5)
  (h3 : runsScored = 69) :
  newBattingAverage currentAverage matchesPlayed runsScored = 54 := by
  sorry

#eval newBattingAverage 51 5 69

end NUMINAMATH_CALUDE_batting_average_calculation_l942_94243


namespace NUMINAMATH_CALUDE_power_multiplication_l942_94248

theorem power_multiplication (a : ℝ) : a^3 * a = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l942_94248


namespace NUMINAMATH_CALUDE_problem_solution_l942_94235

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 9| - |x - 5|

-- Define the function y(x)
def y (x : ℝ) : ℝ := f x + 3*|x - 5|

theorem problem_solution :
  -- Part 1: Solution set of f(x) ≥ 2x-1
  (∀ x : ℝ, f x ≥ 2*x - 1 ↔ x ≤ 5/3) ∧
  -- Part 2: Minimum value of y(x)
  (∀ x : ℝ, y x ≥ 1) ∧
  (∃ x : ℝ, y x = 1) ∧
  -- Part 3: Minimum value of a + 3b given 1/a + 3/b = 1
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 3/b = 1 → a + 3*b ≥ 16) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1/a + 3/b = 1 ∧ a + 3*b = 16) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l942_94235


namespace NUMINAMATH_CALUDE_solution_range_l942_94295

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, x ≤ 1 ∧ 3^x = a^2 + 2*a) → 
  (a ∈ Set.Icc (-3) (-2) ∪ Set.Ioo 0 1) :=
sorry

end NUMINAMATH_CALUDE_solution_range_l942_94295


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l942_94259

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a2 : a 2 = 2)
  (h_sum : a 4 + a 5 = 12) :
  a 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l942_94259


namespace NUMINAMATH_CALUDE_max_profit_at_10_max_profit_value_l942_94249

/-- Profit function for location A -/
def L₁ (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

/-- Profit function for location B -/
def L₂ (x : ℝ) : ℝ := 2 * x

/-- Total profit function -/
def totalProfit (x : ℝ) : ℝ := L₁ x + L₂ (15 - x)

/-- The maximum profit is achieved when selling 10 cars in location A -/
theorem max_profit_at_10 :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 15 → totalProfit x ≤ totalProfit 10 :=
sorry

/-- The maximum profit is 45.6 -/
theorem max_profit_value : totalProfit 10 = 45.6 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_10_max_profit_value_l942_94249


namespace NUMINAMATH_CALUDE_fraction_simplification_l942_94265

theorem fraction_simplification (m : ℝ) (h : m ≠ 3 ∧ m ≠ -3) :
  (m^2 - 3*m) / (9 - m^2) = -m / (m + 3) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l942_94265


namespace NUMINAMATH_CALUDE_H_surjective_l942_94289

def H (x : ℝ) : ℝ := |x^2 + 2*x + 1| - |x^2 - 2*x + 1|

theorem H_surjective : Function.Surjective H := by sorry

end NUMINAMATH_CALUDE_H_surjective_l942_94289


namespace NUMINAMATH_CALUDE_area_of_larger_rectangle_l942_94210

/-- A rectangle with area 2 and length twice its width -/
structure SmallerRectangle where
  width : ℝ
  length : ℝ
  area_eq_two : width * length = 2
  length_eq_twice_width : length = 2 * width

/-- The larger rectangle formed by three smaller rectangles -/
def LargerRectangle (r : SmallerRectangle) : ℝ × ℝ :=
  (3 * r.length, r.width)

/-- The theorem to be proved -/
theorem area_of_larger_rectangle (r : SmallerRectangle) :
  (LargerRectangle r).1 * (LargerRectangle r).2 = 6 := by
  sorry

#check area_of_larger_rectangle

end NUMINAMATH_CALUDE_area_of_larger_rectangle_l942_94210


namespace NUMINAMATH_CALUDE_exists_bound_for_factorial_digit_sum_l942_94209

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Factorial of a natural number -/
def factorial (n : ℕ) : ℕ := sorry

/-- Theorem: Existence of a bound for factorial digit sum -/
theorem exists_bound_for_factorial_digit_sum :
  ∃ b : ℕ, ∀ n : ℕ, n > b → sum_of_digits (factorial n) ≥ 10^100 := by
  sorry

end NUMINAMATH_CALUDE_exists_bound_for_factorial_digit_sum_l942_94209


namespace NUMINAMATH_CALUDE_functional_equation_identity_l942_94251

theorem functional_equation_identity (f : ℕ → ℕ) 
  (h : ∀ m n : ℕ, f (m + f n) = f m + n) : 
  ∀ n : ℕ, f n = n := by
sorry

end NUMINAMATH_CALUDE_functional_equation_identity_l942_94251


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l942_94233

/-- The total surface area of a cylinder with height 12 and radius 4 is 128π. -/
theorem cylinder_surface_area :
  let h : ℝ := 12
  let r : ℝ := 4
  let base_area : ℝ := π * r^2
  let lateral_area : ℝ := 2 * π * r * h
  let total_area : ℝ := 2 * base_area + lateral_area
  total_area = 128 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l942_94233


namespace NUMINAMATH_CALUDE_correct_restroom_count_l942_94255

/-- The number of students in the restroom -/
def students_in_restroom : ℕ := 2

/-- The number of absent students -/
def absent_students : ℕ := 3 * students_in_restroom - 1

/-- The total number of desks -/
def total_desks : ℕ := 4 * 6

/-- The number of occupied desks -/
def occupied_desks : ℕ := (2 * total_desks) / 3

/-- The total number of students Carla teaches -/
def total_students : ℕ := 23

theorem correct_restroom_count :
  students_in_restroom + absent_students + occupied_desks = total_students :=
sorry

end NUMINAMATH_CALUDE_correct_restroom_count_l942_94255


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l942_94263

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 75 → 
  E = 4 * F + 15 → 
  D + E + F = 180 → 
  F = 18 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l942_94263


namespace NUMINAMATH_CALUDE_apple_cost_theorem_l942_94299

/-- The cost of apples given a rate per half dozen -/
def appleCost (halfDozenRate : ℚ) (dozens : ℚ) : ℚ :=
  dozens * (2 * halfDozenRate)

theorem apple_cost_theorem (halfDozenRate : ℚ) :
  halfDozenRate = (4.80 : ℚ) →
  appleCost halfDozenRate 4 = (38.40 : ℚ) :=
by
  sorry

#eval appleCost (4.80 : ℚ) 4

end NUMINAMATH_CALUDE_apple_cost_theorem_l942_94299


namespace NUMINAMATH_CALUDE_q_range_l942_94252

def q (x : ℝ) : ℝ := (3 * x^2 + 1)^2

theorem q_range :
  ∀ y : ℝ, y ∈ Set.range q ↔ y ≥ 1 := by sorry

end NUMINAMATH_CALUDE_q_range_l942_94252


namespace NUMINAMATH_CALUDE_f_at_three_fifths_l942_94208

def f (x : ℝ) : ℝ := 15 * x^5 + 6 * x^4 + x^3 - x^2 - 2*x - 1

theorem f_at_three_fifths :
  f (3/5) = -2/5 := by sorry

end NUMINAMATH_CALUDE_f_at_three_fifths_l942_94208


namespace NUMINAMATH_CALUDE_third_degree_polynomial_specific_value_l942_94236

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- The property that the absolute value of g at certain points equals 10 -/
def HasSpecificValues (g : ThirdDegreePolynomial) : Prop :=
  |g (-1)| = 10 ∧ |g 0| = 10 ∧ |g 2| = 10 ∧ |g 4| = 10 ∧ |g 5| = 10 ∧ |g 8| = 10

/-- The theorem statement -/
theorem third_degree_polynomial_specific_value (g : ThirdDegreePolynomial) 
  (h : HasSpecificValues g) : |g 3| = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_third_degree_polynomial_specific_value_l942_94236


namespace NUMINAMATH_CALUDE_add_preserves_inequality_l942_94238

theorem add_preserves_inequality (a b : ℝ) (h : a > b) : a + 2 > b + 2 := by
  sorry

end NUMINAMATH_CALUDE_add_preserves_inequality_l942_94238


namespace NUMINAMATH_CALUDE_investment_proof_l942_94253

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proof of the investment problem -/
theorem investment_proof (principal : ℝ) (rate : ℝ) (time : ℕ) 
  (h1 : principal = 6000)
  (h2 : rate = 0.1)
  (h3 : time = 2) :
  compound_interest principal rate time = 7260 := by
  sorry

end NUMINAMATH_CALUDE_investment_proof_l942_94253


namespace NUMINAMATH_CALUDE_inverse_cube_theorem_l942_94201

-- Define the relationship between z and x
def inverse_cube_relation (z x : ℝ) : Prop :=
  ∃ k : ℝ, 7 * z = k / (x^3)

-- State the theorem
theorem inverse_cube_theorem :
  ∀ z₁ z₂ : ℝ,
  inverse_cube_relation z₁ 2 ∧ z₁ = 4 →
  inverse_cube_relation z₂ 4 →
  z₂ = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_cube_theorem_l942_94201


namespace NUMINAMATH_CALUDE_deepak_age_l942_94262

theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / (deepak_age : ℚ) = 4 / 3 →
  rahul_age + 6 = 26 →
  deepak_age = 15 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l942_94262


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l942_94285

theorem company_picnic_attendance 
  (total_employees : ℝ) 
  (total_men : ℝ) 
  (total_women : ℝ) 
  (women_picnic_attendance : ℝ) 
  (total_picnic_attendance : ℝ) 
  (h1 : women_picnic_attendance = 0.4 * total_women)
  (h2 : total_men = 0.3 * total_employees)
  (h3 : total_women = total_employees - total_men)
  (h4 : total_picnic_attendance = 0.34 * total_employees)
  : (total_picnic_attendance - women_picnic_attendance) / total_men = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l942_94285


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l942_94222

/-- A function that checks if a positive integer contains the consecutive digit sequence 2048 -/
def contains2048 (n : ℕ+) : Prop := sorry

/-- The set of all positive integers that do not contain the consecutive digit sequence 2048 -/
def S : Set ℕ+ := {n : ℕ+ | ¬contains2048 n}

/-- The theorem to be proved -/
theorem sum_reciprocals_bound (T : Set ℕ+) (h : T ⊆ S) :
  ∑' (n : T), (1 : ℝ) / n ≤ 400000 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l942_94222


namespace NUMINAMATH_CALUDE_youtube_video_length_l942_94297

/-- Represents the duration of a YouTube video session in seconds -/
def YouTubeSession (ad1 ad2 video1 video2 pause totalTime : ℕ) (lastTwoEqual : Bool) : Prop :=
  let firstVideoTotal := ad1 + 120  -- 2 minutes = 120 seconds
  let secondVideoTotal := ad2 + 270  -- 4 minutes 30 seconds = 270 seconds
  let remainingTime := totalTime - (firstVideoTotal + secondVideoTotal)
  let lastTwoVideosTime := remainingTime - pause
  lastTwoEqual ∧ 
  (lastTwoVideosTime / 2 = 495) ∧
  (totalTime = 1500)

theorem youtube_video_length 
  (ad1 ad2 video1 video2 pause totalTime : ℕ) 
  (lastTwoEqual : Bool) 
  (h : YouTubeSession ad1 ad2 video1 video2 pause totalTime lastTwoEqual) :
  ∃ (lastVideoLength : ℕ), lastVideoLength = 495 :=
sorry

end NUMINAMATH_CALUDE_youtube_video_length_l942_94297


namespace NUMINAMATH_CALUDE_sean_patch_selling_price_l942_94268

/-- Proves that the selling price per patch is $12 given the conditions of Sean's patch business. -/
theorem sean_patch_selling_price
  (num_patches : ℕ)
  (cost_per_patch : ℚ)
  (net_profit : ℚ)
  (h_num_patches : num_patches = 100)
  (h_cost_per_patch : cost_per_patch = 1.25)
  (h_net_profit : net_profit = 1075) :
  (cost_per_patch * num_patches + net_profit) / num_patches = 12 := by
  sorry

end NUMINAMATH_CALUDE_sean_patch_selling_price_l942_94268


namespace NUMINAMATH_CALUDE_nonzero_terms_count_l942_94254

def expression (x : ℝ) : ℝ := (2*x + 5)*(3*x^2 + 4*x + 8) - 4*(x^3 - x^2 + 5*x + 2)

theorem nonzero_terms_count : 
  ∃ (a b c d : ℝ), ∀ x : ℝ, 
    expression x = a*x^3 + b*x^2 + c*x + d ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_nonzero_terms_count_l942_94254


namespace NUMINAMATH_CALUDE_inverse_function_point_and_sum_l942_94218

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem inverse_function_point_and_sum :
  (f 2 = 9) →  -- This captures the condition that (2,3) is on y = f(x)/3
  (∃ (x : ℝ), f x = 9 ∧ f⁻¹ 9 = 2) ∧  -- This states that (9, 2/3) is on y = f^(-1)(x)/3
  (9 + 2/3 = 29/3) :=  -- This is the sum of coordinates
by sorry

end NUMINAMATH_CALUDE_inverse_function_point_and_sum_l942_94218


namespace NUMINAMATH_CALUDE_smaller_type_pages_l942_94241

theorem smaller_type_pages 
  (total_words : ℕ) 
  (larger_type_words_per_page : ℕ) 
  (smaller_type_words_per_page : ℕ) 
  (total_pages : ℕ) 
  (h1 : total_words = 48000)
  (h2 : larger_type_words_per_page = 1800)
  (h3 : smaller_type_words_per_page = 2400)
  (h4 : total_pages = 21) :
  ∃ (x y : ℕ), 
    x + y = total_pages ∧ 
    larger_type_words_per_page * x + smaller_type_words_per_page * y = total_words ∧
    y = 17 := by
  sorry

end NUMINAMATH_CALUDE_smaller_type_pages_l942_94241


namespace NUMINAMATH_CALUDE_parabola_points_l942_94202

theorem parabola_points : 
  {p : ℝ × ℝ | p.2 = p.1^2 - 1 ∧ p.2 = 3} = {(-2, 3), (2, 3)} := by
  sorry

end NUMINAMATH_CALUDE_parabola_points_l942_94202


namespace NUMINAMATH_CALUDE_pool_volume_l942_94250

/-- The volume of a cylindrical pool minus a central cylindrical pillar -/
theorem pool_volume (pool_diameter : ℝ) (pool_depth : ℝ) (pillar_diameter : ℝ) (pillar_depth : ℝ)
  (h1 : pool_diameter = 20)
  (h2 : pool_depth = 5)
  (h3 : pillar_diameter = 4)
  (h4 : pillar_depth = 5) :
  (π * (pool_diameter / 2)^2 * pool_depth) - (π * (pillar_diameter / 2)^2 * pillar_depth) = 480 * π := by
  sorry

#check pool_volume

end NUMINAMATH_CALUDE_pool_volume_l942_94250


namespace NUMINAMATH_CALUDE_streaming_service_fee_l942_94298

/-- Given a fixed monthly fee and a charge per hour for extra content,
    if the total for one month is $18.60 and the total for another month
    with triple the extra content usage is $32.40,
    then the fixed monthly fee is $11.70. -/
theorem streaming_service_fee (x y : ℝ)
  (feb_bill : x + y = 18.60)
  (mar_bill : x + 3*y = 32.40) :
  x = 11.70 := by
  sorry

end NUMINAMATH_CALUDE_streaming_service_fee_l942_94298


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l942_94256

theorem existence_of_special_integers :
  ∃ (A : Fin 10 → ℕ+),
    (∀ i j : Fin 10, i ≠ j → ¬(A i ∣ A j)) ∧
    (∀ i j : Fin 10, i ≠ j → (A i)^2 ∣ A j) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l942_94256


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l942_94244

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) :
  x / (x^2 - 1) / (1 + 1 / (x - 1)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l942_94244


namespace NUMINAMATH_CALUDE_hexagram_arrangements_l942_94234

/-- A regular six-pointed star -/
structure HexagramStar :=
  (points : Fin 6 → Type)

/-- The group of symmetries of a regular six-pointed star -/
def hexagramSymmetries : ℕ := 12

/-- The number of ways to arrange 6 distinct objects -/
def totalArrangements : ℕ := 720

/-- The number of distinct arrangements of 6 objects on a hexagram star -/
def distinctArrangements (star : HexagramStar) : ℕ :=
  totalArrangements / hexagramSymmetries

theorem hexagram_arrangements (star : HexagramStar) :
  distinctArrangements star = 60 := by
  sorry

end NUMINAMATH_CALUDE_hexagram_arrangements_l942_94234


namespace NUMINAMATH_CALUDE_conference_handshakes_l942_94229

/-- Represents a group of employees at a conference -/
structure EmployeeGroup where
  size : Nat
  has_closed_loop : Bool

/-- Calculates the number of handshakes in the employee group -/
def count_handshakes (group : EmployeeGroup) : Nat :=
  if group.has_closed_loop && group.size ≥ 3 then
    (group.size * (group.size - 3)) / 2
  else
    0

/-- Theorem: In a group of 10 employees with a closed managerial loop,
    where each person shakes hands with everyone except their direct manager
    and direct subordinate, the total number of handshakes is 35 -/
theorem conference_handshakes :
  let group : EmployeeGroup := { size := 10, has_closed_loop := true }
  count_handshakes group = 35 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l942_94229


namespace NUMINAMATH_CALUDE_max_digit_count_is_24_l942_94239

def apartment_numbers : List Nat := 
  (List.range 46).map (· + 90) ++ (List.range 46).map (· + 190)

def digit_count (d : Nat) (n : Nat) : Nat :=
  if n = 0 then
    if d = 0 then 1 else 0
  else
    digit_count d (n / 10) + if n % 10 = d then 1 else 0

def count_digit (d : Nat) (numbers : List Nat) : Nat :=
  numbers.foldl (fun acc n => acc + digit_count d n) 0

theorem max_digit_count_is_24 :
  (List.range 10).foldl (fun acc d => max acc (count_digit d apartment_numbers)) 0 = 24 := by
  sorry

end NUMINAMATH_CALUDE_max_digit_count_is_24_l942_94239


namespace NUMINAMATH_CALUDE_cosine_inequality_l942_94245

theorem cosine_inequality (y : ℝ) (hy : 0 ≤ y ∧ y ≤ 2 * Real.pi) :
  ∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi → Real.cos (x - y) ≥ Real.cos x - Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_cosine_inequality_l942_94245


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l942_94204

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  (∀ n : ℕ, b (n + 1) > b n) →
  (∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d) →
  b 5 * b 6 = 14 →
  b 4 * b 7 = -324 ∨ b 4 * b 7 = -36 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l942_94204


namespace NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l942_94275

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ, n ≥ 1 → (n ∣ 2^n - 1) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l942_94275


namespace NUMINAMATH_CALUDE_probability_of_specific_draw_l942_94257

def total_balls : ℕ := 18
def red_balls : ℕ := 4
def yellow_balls : ℕ := 5
def green_balls : ℕ := 6
def blue_balls : ℕ := 3
def drawn_balls : ℕ := 4

def favorable_outcomes : ℕ := Nat.choose green_balls 2 * Nat.choose red_balls 1 * Nat.choose blue_balls 1

def total_outcomes : ℕ := Nat.choose total_balls drawn_balls

theorem probability_of_specific_draw :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 17 :=
sorry

end NUMINAMATH_CALUDE_probability_of_specific_draw_l942_94257


namespace NUMINAMATH_CALUDE_set_equality_l942_94267

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set E
def E : Set ℝ := {x : ℝ | x ≤ -3 ∨ x ≥ 2}

-- Define set F
def F : Set ℝ := {x : ℝ | -1 < x ∧ x < 5}

-- Theorem statement
theorem set_equality : {x : ℝ | -1 < x ∧ x < 2} = (Eᶜ ∩ F) := by sorry

end NUMINAMATH_CALUDE_set_equality_l942_94267


namespace NUMINAMATH_CALUDE_chord_equation_l942_94247

/-- The equation of a line that is a chord of the ellipse x^2 + 4y^2 = 36 and is bisected at (4, 2) -/
theorem chord_equation (x y : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    -- Points (x₁, y₁) and (x₂, y₂) lie on the ellipse
    x₁^2 + 4*y₁^2 = 36 ∧ x₂^2 + 4*y₂^2 = 36 ∧
    -- (4, 2) is the midpoint of the chord
    (x₁ + x₂)/2 = 4 ∧ (y₁ + y₂)/2 = 2 ∧
    -- (x, y) is on the line containing the chord
    ∃ (t : ℝ), x = x₁ + t*(x₂ - x₁) ∧ y = y₁ + t*(y₂ - y₁)) →
  x + 2*y - 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_chord_equation_l942_94247


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_l942_94232

/-- The percentage of units that are defective -/
def defective_percentage : ℝ := 6

/-- The percentage of defective units that are shipped for sale -/
def shipped_percentage : ℝ := 4

/-- The result we want to prove -/
def result : ℝ := 0.24

theorem defective_shipped_percentage :
  (defective_percentage / 100) * (shipped_percentage / 100) * 100 = result := by
  sorry

end NUMINAMATH_CALUDE_defective_shipped_percentage_l942_94232


namespace NUMINAMATH_CALUDE_chord_bisected_by_M_l942_94225

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the point M
def M : ℝ × ℝ := (2, 1)

-- Define a chord of the ellipse
def is_chord (A B : ℝ × ℝ) : Prop :=
  is_on_ellipse A.1 A.2 ∧ is_on_ellipse B.1 B.2

-- Define the midpoint of a chord
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define a line by its equation ax + by + c = 0
def line_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- The main theorem
theorem chord_bisected_by_M :
  ∀ A B : ℝ × ℝ,
  is_chord A B →
  is_midpoint M A B →
  line_equation 1 2 (-4) A.1 A.2 ∧ line_equation 1 2 (-4) B.1 B.2 :=
sorry

end NUMINAMATH_CALUDE_chord_bisected_by_M_l942_94225


namespace NUMINAMATH_CALUDE_range_of_x_range_of_a_l942_94278

-- Define the conditions
def p (x : ℝ) := x^2 - x - 2 ≤ 0
def q (x : ℝ) := (x - 3) / x < 0
def r (x a : ℝ) := (x - (a + 1)) * (x + (2 * a - 1)) ≤ 0

-- Question 1
theorem range_of_x (x : ℝ) (h1 : p x) (h2 : q x) : x ∈ Set.Ioc 0 2 := by sorry

-- Question 2
theorem range_of_a (a : ℝ) 
  (h1 : ∀ x, p x → r x a) 
  (h2 : ∃ x, r x a ∧ ¬p x) 
  (h3 : a > 0) : 
  a > 1 := by sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_a_l942_94278


namespace NUMINAMATH_CALUDE_sqrt_meaningful_condition_l942_94207

theorem sqrt_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x + 6) ↔ x ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_condition_l942_94207


namespace NUMINAMATH_CALUDE_toms_barbados_trip_cost_l942_94293

/-- The total cost for Tom's trip to Barbados -/
def total_cost (num_vaccines : ℕ) (vaccine_cost : ℚ) (doctor_visit_cost : ℚ) 
                (insurance_coverage : ℚ) (trip_cost : ℚ) : ℚ :=
  let medical_cost := num_vaccines * vaccine_cost + doctor_visit_cost
  let out_of_pocket_medical := medical_cost * (1 - insurance_coverage)
  out_of_pocket_medical + trip_cost

/-- Theorem stating the total cost for Tom's trip to Barbados -/
theorem toms_barbados_trip_cost :
  total_cost 10 45 250 0.8 1200 = 1340 := by
  sorry

end NUMINAMATH_CALUDE_toms_barbados_trip_cost_l942_94293


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l942_94292

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 3) ^ (1/3 : ℝ) = -4 ∧ x = 207 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l942_94292


namespace NUMINAMATH_CALUDE_remaining_drivable_distance_l942_94272

/-- Proves the remaining drivable distance after a trip --/
theorem remaining_drivable_distance
  (fuel_efficiency : ℝ)
  (tank_capacity : ℝ)
  (trip_distance : ℝ)
  (h1 : fuel_efficiency = 20)
  (h2 : tank_capacity = 16)
  (h3 : trip_distance = 220) :
  fuel_efficiency * tank_capacity - trip_distance = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_drivable_distance_l942_94272


namespace NUMINAMATH_CALUDE_simplify_expression_l942_94264

theorem simplify_expression (a : ℝ) : 2 * (a + 2) - 2 * a = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l942_94264


namespace NUMINAMATH_CALUDE_river_straight_parts_length_l942_94200

theorem river_straight_parts_length 
  (total_length : ℝ) 
  (straight_percentage : ℝ) 
  (h1 : total_length = 80) 
  (h2 : straight_percentage = 0.25) : 
  straight_percentage * total_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_river_straight_parts_length_l942_94200


namespace NUMINAMATH_CALUDE_fraction_sum_l942_94296

theorem fraction_sum (x y : ℝ) (h : y / x = 3 / 4) : (x + y) / x = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l942_94296


namespace NUMINAMATH_CALUDE_smallest_multiple_l942_94211

theorem smallest_multiple (n : ℕ) : n = 481 ↔ 
  n > 0 ∧ 
  (∃ k : ℤ, n = 37 * k) ∧ 
  (∃ m : ℤ, n - 7 = 97 * m) ∧ 
  (∀ x : ℕ, x > 0 ∧ (∃ k : ℤ, x = 37 * k) ∧ (∃ m : ℤ, x - 7 = 97 * m) → x ≥ n) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_l942_94211


namespace NUMINAMATH_CALUDE_prop1_prop4_l942_94240

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (perp_to_plane : Line → Plane → Prop)

-- Proposition 1
theorem prop1 (a b c : Line) :
  parallel a b → perpendicular b c → perpendicular a c :=
sorry

-- Proposition 4
theorem prop4 (a b : Line) (α : Plane) :
  perp_to_plane a α → contained_in b α → perpendicular a b :=
sorry

end NUMINAMATH_CALUDE_prop1_prop4_l942_94240


namespace NUMINAMATH_CALUDE_line_equation_through_points_l942_94276

/-- The line passing through points (-1, 0) and (0, 1) is represented by the equation x - y + 1 = 0 -/
theorem line_equation_through_points : 
  ∀ (x y : ℝ), (x = -1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) → x - y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l942_94276


namespace NUMINAMATH_CALUDE_die_volume_l942_94242

theorem die_volume (side_area : ℝ) (h : side_area = 64) : 
  side_area^(3/2) = 512 := by
  sorry

end NUMINAMATH_CALUDE_die_volume_l942_94242


namespace NUMINAMATH_CALUDE_cindy_same_color_probability_l942_94203

def total_marbles : ℕ := 8
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 3
def green_marbles : ℕ := 1
def yellow_marbles : ℕ := 1

def alice_draw : ℕ := 3
def bob_draw : ℕ := 2
def cindy_draw : ℕ := 2

def probability_cindy_same_color : ℚ := 1 / 35

theorem cindy_same_color_probability :
  probability_cindy_same_color = 1 / 35 :=
by sorry

end NUMINAMATH_CALUDE_cindy_same_color_probability_l942_94203


namespace NUMINAMATH_CALUDE_sequence_periodicity_l942_94269

def is_periodic (a : ℕ → ℕ) : Prop :=
  ∃ (p : ℕ), p > 0 ∧ ∀ (n : ℕ), a (n + p) = a n

theorem sequence_periodicity (a : ℕ → ℕ) 
  (h1 : ∀ n, a n < 1988)
  (h2 : ∀ m n, (a m + a n) % a (m + n) = 0) :
  is_periodic a := by
  sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l942_94269


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l942_94219

/-- The distance between the vertices of a hyperbola with equation x^2/64 - y^2/49 = 1 is 16 -/
theorem hyperbola_vertices_distance : 
  ∀ (x y : ℝ), x^2/64 - y^2/49 = 1 → ∃ (v1 v2 : ℝ × ℝ), 
    (v1.1^2/64 - v1.2^2/49 = 1) ∧ 
    (v2.1^2/64 - v2.2^2/49 = 1) ∧ 
    (v1.2 = 0) ∧ (v2.2 = 0) ∧
    (v2.1 = -v1.1) ∧
    (v2.1 - v1.1 = 16) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l942_94219


namespace NUMINAMATH_CALUDE_repeating_decimal_fractions_l942_94227

def repeating_decimal_3 : ℚ := 0.333333
def repeating_decimal_56 : ℚ := 0.565656

theorem repeating_decimal_fractions :
  (repeating_decimal_3 = 1 / 3) ∧
  (repeating_decimal_56 = 56 / 99) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fractions_l942_94227


namespace NUMINAMATH_CALUDE_lucy_fish_count_lucy_fish_proof_l942_94220

theorem lucy_fish_count : ℕ → Prop :=
  fun current_fish =>
    (current_fish + 68 = 280) → (current_fish = 212)

-- Proof
theorem lucy_fish_proof : lucy_fish_count 212 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_count_lucy_fish_proof_l942_94220


namespace NUMINAMATH_CALUDE_workforce_reduction_l942_94270

theorem workforce_reduction (initial_employees : ℕ) : 
  (initial_employees : ℝ) * 0.85 * 0.75 = 182 → 
  initial_employees = 285 :=
by
  sorry

end NUMINAMATH_CALUDE_workforce_reduction_l942_94270
