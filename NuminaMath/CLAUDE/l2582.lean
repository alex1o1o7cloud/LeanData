import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l2582_258278

-- Define the propositions
def P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
def Q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(5-2*a))^x > (-(5-2*a))^y

-- State the theorem
theorem range_of_a :
  (∃ a : ℝ, (¬(P a) ∧ Q a) ∨ (P a ∧ ¬(Q a))) →
  (∃ a : ℝ, a ≤ -2 ∧ ∀ b : ℝ, b ≤ -2 → (¬(P b) ∧ Q b) ∨ (P b ∧ ¬(Q b))) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2582_258278


namespace NUMINAMATH_CALUDE_right_angle_point_location_l2582_258270

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the plane
def Point := ℝ × ℝ

-- Define the property of being on the circle
def OnCircle (p : Point) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the angle between three points
def Angle (p1 p2 p3 : Point) : ℝ := sorry

-- Define a right angle
def IsRightAngle (angle : ℝ) : Prop :=
  angle = Real.pi / 2

-- Define the property of being diametrically opposite
def DiametricallyOpposite (p1 p2 : Point) (c : Circle) : Prop :=
  (p1.1 + p2.1) / 2 = c.center.1 ∧ (p1.2 + p2.2) / 2 = c.center.2

-- The main theorem
theorem right_angle_point_location
  (c : Circle) (C : Point) (ho : OnCircle C c) :
  ∃! X, OnCircle X c ∧ IsRightAngle (Angle C X c.center) ∧ DiametricallyOpposite C X c :=
sorry

end NUMINAMATH_CALUDE_right_angle_point_location_l2582_258270


namespace NUMINAMATH_CALUDE_unit_circle_sector_angle_l2582_258259

/-- The radian measure of a central angle in a unit circle, given the area of the sector -/
def central_angle (area : ℝ) : ℝ := 2 * area

theorem unit_circle_sector_angle (area : ℝ) (h : area = 1) : 
  central_angle area = 2 := by
  sorry

end NUMINAMATH_CALUDE_unit_circle_sector_angle_l2582_258259


namespace NUMINAMATH_CALUDE_no_consecutive_heads_probability_sum_of_numerator_and_denominator_l2582_258218

/-- Number of valid sequences of length n ending in Tails -/
def a : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n+2) => a (n+1) + a n

/-- Number of valid sequences of length n ending in Heads -/
def b : ℕ → ℕ
| 0 => 0
| (n+1) => a n

/-- Total number of valid sequences of length n -/
def valid_sequences (n : ℕ) : ℕ := a n + b n

/-- Total number of possible sequences of length n -/
def total_sequences (n : ℕ) : ℕ := 2^n

theorem no_consecutive_heads_probability :
  (valid_sequences 10 : ℚ) / (total_sequences 10 : ℚ) = 9 / 64 :=
sorry

theorem sum_of_numerator_and_denominator : 9 + 64 = 73 :=
sorry

end NUMINAMATH_CALUDE_no_consecutive_heads_probability_sum_of_numerator_and_denominator_l2582_258218


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l2582_258290

/-- The range of m for which the quadratic equation (m-3)x^2 + 4x + 1 = 0 has two real roots -/
theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (m - 3) * x₁^2 + 4 * x₁ + 1 = 0 ∧ (m - 3) * x₂^2 + 4 * x₂ + 1 = 0) ↔ 
  (m ≤ 7 ∧ m ≠ 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l2582_258290


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_line_l2582_258286

/-- Given a line l: x - y - 1 = 0 and two points A(-1, 1) and B(2, -2),
    prove that B is symmetric to A with respect to l. -/
theorem symmetric_point_wrt_line :
  let l : ℝ → ℝ → Prop := λ x y => x - y - 1 = 0
  let A : ℝ × ℝ := (-1, 1)
  let B : ℝ × ℝ := (2, -2)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (∀ x y, l x y ↔ x - y - 1 = 0) →
  l midpoint.1 midpoint.2 ∧
  (B.2 - A.2) / (B.1 - A.1) = -((B.1 - A.1) / (B.2 - A.2)) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_line_l2582_258286


namespace NUMINAMATH_CALUDE_subset_P_l2582_258223

def P : Set ℝ := {x | x ≤ 3}

theorem subset_P : {-1} ⊆ P := by sorry

end NUMINAMATH_CALUDE_subset_P_l2582_258223


namespace NUMINAMATH_CALUDE_seagulls_remaining_l2582_258256

theorem seagulls_remaining (initial : ℕ) (scared_fraction : ℚ) (flew_fraction : ℚ) : 
  initial = 36 → scared_fraction = 1/4 → flew_fraction = 1/3 → 
  (initial - initial * scared_fraction - (initial - initial * scared_fraction) * flew_fraction : ℚ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_seagulls_remaining_l2582_258256


namespace NUMINAMATH_CALUDE_ab_bc_ratio_is_two_plus_sqrt_three_l2582_258208

-- Define the quadrilateral ABCD
structure Quadrilateral (A B C D : ℝ × ℝ) : Prop where
  right_angle_B : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0
  right_angle_C : (C.1 - B.1) * (D.1 - C.1) + (C.2 - B.2) * (D.2 - C.2) = 0

-- Define similarity of triangles
def similar_triangles (A B C D E F : ℝ × ℝ) : Prop :=
  ∃ k > 0, (B.1 - A.1)^2 + (B.2 - A.2)^2 = k * ((E.1 - D.1)^2 + (E.2 - D.2)^2) ∧
            (C.1 - B.1)^2 + (C.2 - B.2)^2 = k * ((F.1 - E.1)^2 + (F.2 - E.2)^2) ∧
            (A.1 - C.1)^2 + (A.2 - C.2)^2 = k * ((D.1 - F.1)^2 + (D.2 - F.2)^2)

-- Define the area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Main theorem
theorem ab_bc_ratio_is_two_plus_sqrt_three
  (A B C D E : ℝ × ℝ)
  (h_quad : Quadrilateral A B C D)
  (h_sim_ABC_BCD : similar_triangles A B C B C D)
  (h_AB_gt_BC : (A.1 - B.1)^2 + (A.2 - B.2)^2 > (B.1 - C.1)^2 + (B.2 - C.2)^2)
  (h_E_interior : ∃ t u : ℝ, 0 < t ∧ t < 1 ∧ 0 < u ∧ u < 1 ∧
    E = (t * A.1 + (1 - t) * C.1, u * B.2 + (1 - u) * D.2))
  (h_sim_ABC_CEB : similar_triangles A B C C E B)
  (h_area_ratio : triangle_area A E D = 25 * triangle_area C E B) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) / Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ab_bc_ratio_is_two_plus_sqrt_three_l2582_258208


namespace NUMINAMATH_CALUDE_simplification_order_l2582_258245

-- Define the power operations
inductive PowerOperation
| MultiplicationOfPowers
| PowerOfPower
| PowerOfProduct

-- Define a function to simplify the expression
def simplify (a : ℕ) : ℕ := (a^2 * a^3)^2

-- Define a function to get the sequence of operations
def operationSequence : List PowerOperation :=
  [PowerOperation.PowerOfProduct, PowerOperation.PowerOfPower, PowerOperation.MultiplicationOfPowers]

-- State the theorem
theorem simplification_order :
  simplify a = a^10 ∧ operationSequence = [PowerOperation.PowerOfProduct, PowerOperation.PowerOfPower, PowerOperation.MultiplicationOfPowers] :=
sorry

end NUMINAMATH_CALUDE_simplification_order_l2582_258245


namespace NUMINAMATH_CALUDE_decimal_expansion_222nd_digit_l2582_258282

/-- The decimal expansion of 47/777 -/
def decimal_expansion : ℚ := 47 / 777

/-- The length of the repeating block in the decimal expansion -/
def repeat_length : ℕ := 6

/-- The position we're interested in -/
def position : ℕ := 222

/-- The function that returns the nth digit after the decimal point in the decimal expansion -/
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

theorem decimal_expansion_222nd_digit :
  nth_digit position = 5 := by sorry

end NUMINAMATH_CALUDE_decimal_expansion_222nd_digit_l2582_258282


namespace NUMINAMATH_CALUDE_sqrt_5_minus_2_squared_l2582_258291

theorem sqrt_5_minus_2_squared : (Real.sqrt 5 - 2)^2 = 9 - 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_5_minus_2_squared_l2582_258291


namespace NUMINAMATH_CALUDE_f_2012_equals_2_l2582_258274

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2012_equals_2 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 3) = -f x)
  (h_f_1 : f 1 = 2) : 
  f 2012 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_2012_equals_2_l2582_258274


namespace NUMINAMATH_CALUDE_file_size_proof_l2582_258232

/-- Calculates the file size given upload speed and time -/
def fileSize (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that a file uploaded at 8 MB/min for 20 minutes is 160 MB -/
theorem file_size_proof (speed : ℝ) (time : ℝ) (h1 : speed = 8) (h2 : time = 20) :
  fileSize speed time = 160 := by
  sorry

#check file_size_proof

end NUMINAMATH_CALUDE_file_size_proof_l2582_258232


namespace NUMINAMATH_CALUDE_min_value_w_l2582_258283

theorem min_value_w (x y : ℝ) : 
  3 * x^2 + 5 * y^2 + 12 * x - 10 * y + 45 ≥ 28 ∧ 
  ∃ (a b : ℝ), 3 * a^2 + 5 * b^2 + 12 * a - 10 * b + 45 = 28 := by
sorry

end NUMINAMATH_CALUDE_min_value_w_l2582_258283


namespace NUMINAMATH_CALUDE_largest_number_on_board_l2582_258214

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

def satisfies_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ n % 6 = 0 ∧ ends_in_4 n

theorem largest_number_on_board :
  ∃ (m : ℕ), satisfies_conditions m ∧
  ∀ (n : ℕ), satisfies_conditions n → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_on_board_l2582_258214


namespace NUMINAMATH_CALUDE_y_decreases_as_x_increases_l2582_258268

def tensor (m n : ℝ) : ℝ := -m * n + n

theorem y_decreases_as_x_increases :
  let f : ℝ → ℝ := λ x ↦ tensor x 2
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₂ < f x₁ := by
  sorry

end NUMINAMATH_CALUDE_y_decreases_as_x_increases_l2582_258268


namespace NUMINAMATH_CALUDE_complex_modulus_l2582_258228

theorem complex_modulus (z : ℂ) : (1 + I) * z = 2 * I → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2582_258228


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l2582_258227

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  totalStudents : Nat
  numGroups : Nat
  studentsPerGroup : Nat
  selectedNumber : Nat
  selectedGroup : Nat

/-- Theorem: In a systematic sampling of 50 students into 10 groups of 5,
    if the student numbered 12 is selected from the third group,
    then the student numbered 37 will be selected from the eighth group. -/
theorem systematic_sampling_theorem (s : SystematicSampling)
    (h1 : s.totalStudents = 50)
    (h2 : s.numGroups = 10)
    (h3 : s.studentsPerGroup = 5)
    (h4 : s.selectedNumber = 12)
    (h5 : s.selectedGroup = 3) :
    s.selectedNumber + (8 - s.selectedGroup) * s.studentsPerGroup = 37 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_theorem_l2582_258227


namespace NUMINAMATH_CALUDE_line_mb_equals_two_l2582_258285

/-- Given a line with equation y = mx + b passing through points (0, 1) and (1, 3), prove that mb = 2 -/
theorem line_mb_equals_two (m b : ℝ) : 
  (1 = m * 0 + b) →  -- The line passes through (0, 1)
  (3 = m * 1 + b) →  -- The line passes through (1, 3)
  m * b = 2 := by
sorry

end NUMINAMATH_CALUDE_line_mb_equals_two_l2582_258285


namespace NUMINAMATH_CALUDE_no_club_member_is_fraternity_member_l2582_258211

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (FraternityMember : U → Prop)
variable (ClubMember : U → Prop)
variable (Honest : U → Prop)

-- Define the given conditions
axiom some_students_not_honest : ∃ x, Student x ∧ ¬Honest x
axiom all_fraternity_members_honest : ∀ x, FraternityMember x → Honest x
axiom no_club_members_honest : ∀ x, ClubMember x → ¬Honest x

-- Theorem to prove
theorem no_club_member_is_fraternity_member :
  ∀ x, ClubMember x → ¬FraternityMember x :=
sorry

end NUMINAMATH_CALUDE_no_club_member_is_fraternity_member_l2582_258211


namespace NUMINAMATH_CALUDE_least_exponent_sum_for_500_l2582_258296

/-- Given a natural number n, returns the set of exponents of 2 in its binary representation -/
def binaryExponents (n : ℕ) : Finset ℕ :=
  sorry

/-- The sum of exponents of 2 in the binary representation of n -/
def sumOfExponents (n : ℕ) : ℕ :=
  (binaryExponents n).sum id

/-- Checks if a set of exponents represents a valid sum of powers of 2 for a given number -/
def isValidRepresentation (n : ℕ) (exponents : Finset ℕ) : Prop :=
  (exponents.sum (fun i => 2^i) = n) ∧ (exponents.card ≥ 3)

theorem least_exponent_sum_for_500 :
  ∀ (exponents : Finset ℕ),
    isValidRepresentation 500 exponents →
    sumOfExponents 500 ≤ (exponents.sum id) :=
by sorry

end NUMINAMATH_CALUDE_least_exponent_sum_for_500_l2582_258296


namespace NUMINAMATH_CALUDE_complete_square_sum_l2582_258266

theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 8*x + 8 = 0 ↔ (x + b)^2 = c) → b + c = 4 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l2582_258266


namespace NUMINAMATH_CALUDE_at_least_three_positive_and_negative_l2582_258221

theorem at_least_three_positive_and_negative (a : Fin 12 → ℝ) 
  (h : ∀ i : Fin 11, a (i + 1) * (a i - a (i + 1) + a (i + 2)) < 0) :
  (∃ i j k : Fin 12, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 0 < a i ∧ 0 < a j ∧ 0 < a k) ∧
  (∃ i j k : Fin 12, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a i < 0 ∧ a j < 0 ∧ a k < 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_three_positive_and_negative_l2582_258221


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l2582_258246

theorem divisibility_equivalence (x y : ℤ) : 
  (7 ∣ (2*x + 3*y)) ↔ (7 ∣ (5*x + 4*y)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l2582_258246


namespace NUMINAMATH_CALUDE_square_area_not_tripled_when_side_tripled_l2582_258238

theorem square_area_not_tripled_when_side_tripled (s : ℝ) (h : s > 0) :
  (3 * s)^2 ≠ 3 * s^2 := by sorry

end NUMINAMATH_CALUDE_square_area_not_tripled_when_side_tripled_l2582_258238


namespace NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l2582_258222

/-- The volume of a cylinder formed by rotating a rectangle about its shorter side -/
theorem cylinder_volume_from_rectangle (length width : ℝ) (h_length : length = 30) (h_width : width = 16) :
  let radius : ℝ := width / 2
  let height : ℝ := length
  let volume : ℝ := π * radius^2 * height
  volume = 1920 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l2582_258222


namespace NUMINAMATH_CALUDE_complex_number_relation_l2582_258287

theorem complex_number_relation (x y : ℂ) 
  (h : (x^3 + y^3) / (x^3 - y^3) + (x^3 - y^3) / (x^3 + y^3) = 1) :
  (x^9 + y^9) / (x^9 - y^9) + (x^9 - y^9) / (x^9 + y^9) = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_relation_l2582_258287


namespace NUMINAMATH_CALUDE_power_equality_l2582_258262

theorem power_equality (m : ℝ) : (16 : ℝ) ^ (3/4) = 2^m → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2582_258262


namespace NUMINAMATH_CALUDE_no_zeroes_g_l2582_258217

/-- A function f satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  continuous : Continuous f
  differentiable : Differentiable ℝ f
  condition : ∀ x, x * (deriv f x) + f x > 0

/-- The function g(x) = xf(x) + 1 -/
def g (sf : SpecialFunction) (x : ℝ) : ℝ := x * sf.f x + 1

/-- Theorem stating that g has no zeroes for x > 0 -/
theorem no_zeroes_g (sf : SpecialFunction) : ∀ x > 0, g sf x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_zeroes_g_l2582_258217


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_two_l2582_258272

theorem sum_of_coefficients_is_two 
  (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + 
    a₄*(x - 1)^4 + a₅*(x - 1)^5 + a₆*(x - 1)^6 + a₇*(x - 1)^7 + a₈*(x - 1)^8 + 
    a₉*(x - 1)^9 + a₁₀*(x - 1)^10 + a₁₁*(x - 1)^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_two_l2582_258272


namespace NUMINAMATH_CALUDE_profit_loss_percentage_l2582_258231

/-- 
Given an article with a cost price and two selling prices:
1. An original selling price that yields a 27.5% profit
2. A new selling price that is 2/3 of the original price

This theorem proves that the loss percentage at the new selling price is 15%.
-/
theorem profit_loss_percentage (cost_price : ℝ) (original_price : ℝ) (new_price : ℝ) : 
  original_price = cost_price * (1 + 0.275) →
  new_price = (2/3) * original_price →
  (cost_price - new_price) / cost_price * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_profit_loss_percentage_l2582_258231


namespace NUMINAMATH_CALUDE_specific_tangent_distances_l2582_258226

/-- Two externally tangent circles with radii R and r -/
structure TangentCircles where
  R : ℝ
  r : ℝ
  h_positive_R : R > 0
  h_positive_r : r > 0
  h_external : R > r

/-- The distances from the point of tangency to the common tangents -/
def tangent_distances (c : TangentCircles) : Set ℝ :=
  {0, (c.R + c.r) * c.r / c.R}

/-- Theorem about the distances for specific radii -/
theorem specific_tangent_distances :
  ∃ c : TangentCircles, c.R = 3 ∧ c.r = 1 ∧ tangent_distances c = {0, 7/3} := by
  sorry

end NUMINAMATH_CALUDE_specific_tangent_distances_l2582_258226


namespace NUMINAMATH_CALUDE_sum_of_real_cube_roots_of_64_l2582_258252

theorem sum_of_real_cube_roots_of_64 :
  ∃ (x : ℝ), x^3 = 64 ∧ (∀ y : ℝ, y^3 = 64 → y = x) ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_real_cube_roots_of_64_l2582_258252


namespace NUMINAMATH_CALUDE_taxi_problem_l2582_258267

theorem taxi_problem (fans : ℕ) (company_a company_b : ℕ) : 
  fans = 56 →
  company_b = company_a + 3 →
  5 * company_a < fans →
  6 * company_a > fans →
  4 * company_b < fans →
  5 * company_b > fans →
  company_a = 10 := by
sorry

end NUMINAMATH_CALUDE_taxi_problem_l2582_258267


namespace NUMINAMATH_CALUDE_total_letters_in_names_l2582_258204

/-- Represents the number of letters in a person's name -/
structure NameLength where
  firstName : Nat
  surname : Nat

/-- Calculates the total number of letters in a person's full name -/
def totalLetters (name : NameLength) : Nat :=
  name.firstName + name.surname

/-- Theorem: The total number of letters in Jonathan's and his sister's names is 33 -/
theorem total_letters_in_names : 
  let jonathan : NameLength := { firstName := 8, surname := 10 }
  let sister : NameLength := { firstName := 5, surname := 10 }
  totalLetters jonathan + totalLetters sister = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_letters_in_names_l2582_258204


namespace NUMINAMATH_CALUDE_smallest_valid_sequence_length_l2582_258219

def is_valid_sequence (a : List Int) : Prop :=
  a.sum = 2005 ∧ a.prod = 2005

theorem smallest_valid_sequence_length :
  (∃ (n : Nat) (a : List Int), n > 1 ∧ a.length = n ∧ is_valid_sequence a) ∧
  (∀ (m : Nat) (b : List Int), m > 1 ∧ m < 5 ∧ b.length = m → ¬is_valid_sequence b) ∧
  (∃ (c : List Int), c.length = 5 ∧ is_valid_sequence c) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_sequence_length_l2582_258219


namespace NUMINAMATH_CALUDE_trig_identity_l2582_258257

theorem trig_identity (α : ℝ) : 
  4.62 * (Real.cos (2 * α))^4 - 6 * (Real.cos (2 * α))^2 * (Real.sin (2 * α))^2 + (Real.sin (2 * α))^4 = Real.cos (8 * α) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2582_258257


namespace NUMINAMATH_CALUDE_only_third_set_forms_triangle_l2582_258265

/-- Checks if three lengths can form a triangle according to the triangle inequality theorem -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The sets of line segments given in the problem -/
def segment_sets : List (ℝ × ℝ × ℝ) :=
  [(1, 2, 3), (2, 2, 4), (3, 4, 5), (3, 5, 9)]

theorem only_third_set_forms_triangle :
  ∃! set : ℝ × ℝ × ℝ, set ∈ segment_sets ∧ can_form_triangle set.1 set.2.1 set.2.2 :=
by
  sorry

end NUMINAMATH_CALUDE_only_third_set_forms_triangle_l2582_258265


namespace NUMINAMATH_CALUDE_consecutive_points_segment_length_l2582_258276

/-- Given 5 consecutive points on a straight line, prove the length of a specific segment -/
theorem consecutive_points_segment_length 
  (a b c d e : ℝ) -- Define points as real numbers
  (consecutive : a < b ∧ b < c ∧ c < d ∧ d < e) -- Consecutive points condition
  (bc_eq_3cd : c - b = 3 * (d - c)) -- bc = 3 cd
  (ab_eq_5 : b - a = 5) -- ab = 5
  (ac_eq_11 : c - a = 11) -- ac = 11
  (ae_eq_21 : e - a = 21) -- ae = 21
  : e - d = 8 := by -- de = 8
  sorry

end NUMINAMATH_CALUDE_consecutive_points_segment_length_l2582_258276


namespace NUMINAMATH_CALUDE_ladybug_count_l2582_258206

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := 12170

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The total number of ladybugs -/
def total_ladybugs : ℕ := ladybugs_with_spots + ladybugs_without_spots

theorem ladybug_count : total_ladybugs = 67082 := by
  sorry

end NUMINAMATH_CALUDE_ladybug_count_l2582_258206


namespace NUMINAMATH_CALUDE_largest_valid_n_l2582_258244

/-- A coloring of integers from 1 to 14 using two colors -/
def Coloring := Fin 14 → Bool

/-- Check if a coloring satisfies the condition for a given k -/
def valid_for_k (c : Coloring) (k : Nat) : Prop :=
  ∃ (i j i' j' : Fin 14),
    i < j ∧ j - i = k ∧ c i = c j ∧
    i' < j' ∧ j' - i' = k ∧ c i' ≠ c j'

/-- A coloring is valid up to n if it satisfies the condition for all k from 1 to n -/
def valid_coloring (c : Coloring) (n : Nat) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → valid_for_k c k

/-- The main theorem: 11 is the largest n for which a valid coloring exists -/
theorem largest_valid_n :
  (∃ c : Coloring, valid_coloring c 11) ∧
  (∀ c : Coloring, ¬valid_coloring c 12) := by
  sorry

end NUMINAMATH_CALUDE_largest_valid_n_l2582_258244


namespace NUMINAMATH_CALUDE_discarded_numbers_l2582_258224

-- Define the set of numbers
def numbers : Finset ℕ := Finset.range 11 \ {0}

-- Define the type for a distribution on a rectangular block
structure BlockDistribution where
  vertices : Finset ℕ
  face_sum : ℕ
  is_valid : vertices ⊆ numbers ∧ vertices.card = 8 ∧ face_sum = 18

-- Theorem statement
theorem discarded_numbers (d : BlockDistribution) :
  numbers \ d.vertices = {9, 10} := by
  sorry

end NUMINAMATH_CALUDE_discarded_numbers_l2582_258224


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2582_258229

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- The problem statement -/
theorem simple_interest_problem :
  let principal : ℚ := 26775
  let rate : ℚ := 3
  let time : ℚ := 5
  simple_interest principal rate time = 803.25 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2582_258229


namespace NUMINAMATH_CALUDE_max_daily_sales_revenue_l2582_258281

def P (t : ℕ) : ℝ :=
  if 1 ≤ t ∧ t ≤ 24 then t + 2
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

def Q (t : ℕ) : ℝ :=
  if 1 ≤ t ∧ t ≤ 30 then -t + 40
  else 0

def dailySalesRevenue (t : ℕ) : ℝ := P t * Q t

theorem max_daily_sales_revenue :
  (∃ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ dailySalesRevenue t = 1125) ∧
  (∀ t : ℕ, 1 ≤ t ∧ t ≤ 30 → dailySalesRevenue t ≤ 1125) ∧
  (dailySalesRevenue 25 = 1125) :=
sorry

end NUMINAMATH_CALUDE_max_daily_sales_revenue_l2582_258281


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2582_258250

theorem solution_set_of_inequality (x : ℝ) : 
  (x - 1) / (x + 3) < 0 ↔ -3 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2582_258250


namespace NUMINAMATH_CALUDE_solve_equation_l2582_258251

theorem solve_equation (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2582_258251


namespace NUMINAMATH_CALUDE_largest_share_proof_l2582_258264

def profit_split (partners : ℕ) (ratios : List ℕ) (total_profit : ℕ) : ℕ :=
  let total_parts := ratios.sum
  let part_value := total_profit / total_parts
  (ratios.maximum? |>.getD 0) * part_value

theorem largest_share_proof (partners : ℕ) (ratios : List ℕ) (total_profit : ℕ) 
  (h_partners : partners = 5)
  (h_ratios : ratios = [1, 2, 3, 3, 6])
  (h_profit : total_profit = 36000) :
  profit_split partners ratios total_profit = 14400 :=
by
  sorry

#eval profit_split 5 [1, 2, 3, 3, 6] 36000

end NUMINAMATH_CALUDE_largest_share_proof_l2582_258264


namespace NUMINAMATH_CALUDE_right_triangle_to_square_l2582_258284

theorem right_triangle_to_square (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right triangle condition
  b = 10 →           -- longer leg is 10
  b = 2*a →          -- condition for forming a square
  a = 5 := by sorry

end NUMINAMATH_CALUDE_right_triangle_to_square_l2582_258284


namespace NUMINAMATH_CALUDE_andrew_total_payment_l2582_258213

def grapes_quantity : ℕ := 15
def grapes_rate : ℕ := 98
def mangoes_quantity : ℕ := 8
def mangoes_rate : ℕ := 120
def pineapples_quantity : ℕ := 5
def pineapples_rate : ℕ := 75
def oranges_quantity : ℕ := 10
def oranges_rate : ℕ := 60

def total_cost : ℕ := 
  grapes_quantity * grapes_rate + 
  mangoes_quantity * mangoes_rate + 
  pineapples_quantity * pineapples_rate + 
  oranges_quantity * oranges_rate

theorem andrew_total_payment : total_cost = 3405 := by
  sorry

end NUMINAMATH_CALUDE_andrew_total_payment_l2582_258213


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_3_and_5_l2582_258202

theorem smallest_four_digit_multiple_of_3_and_5 : ∃ n : ℕ,
  (n ≥ 1000 ∧ n < 10000) ∧  -- 4-digit number
  n % 3 = 0 ∧               -- multiple of 3
  n % 5 = 0 ∧               -- multiple of 5
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 3 = 0 ∧ m % 5 = 0) → n ≤ m) ∧
  n = 1005 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_3_and_5_l2582_258202


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2582_258241

theorem absolute_value_inequality_solution_set :
  {x : ℝ | x ≠ 0 ∧ |((x - 2) / x)| > (x - 2) / x} = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2582_258241


namespace NUMINAMATH_CALUDE_number_equality_l2582_258269

theorem number_equality (x : ℝ) : 9^6 = x^12 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l2582_258269


namespace NUMINAMATH_CALUDE_parabola_focus_on_line_l2582_258210

-- Define the line on which the focus lies
def focus_line (x y : ℝ) : Prop := x - 2*y - 4 = 0

-- Define the standard equations of parabolas
def parabola_eq1 (x y : ℝ) : Prop := y^2 = 16*x
def parabola_eq2 (x y : ℝ) : Prop := x^2 = -8*y

-- Theorem statement
theorem parabola_focus_on_line :
  ∀ (x y : ℝ), focus_line x y →
  (∃ (a b : ℝ), parabola_eq1 a b) ∨ (∃ (c d : ℝ), parabola_eq2 c d) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_on_line_l2582_258210


namespace NUMINAMATH_CALUDE_perpendicular_lines_l2582_258215

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, 2*x + y + 2 = 0 ∧ a*x + 4*y - 2 = 0 → 
    ((-1/2) * (-a/4) = -1)) → 
  a = -8 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l2582_258215


namespace NUMINAMATH_CALUDE_joyce_final_egg_count_l2582_258209

/-- Calculates the final number of eggs Joyce has after a series of transactions -/
def final_egg_count (initial_eggs : ℝ) (received_eggs : ℝ) (traded_eggs : ℝ) (given_away_eggs : ℝ) : ℝ :=
  initial_eggs + received_eggs - traded_eggs - given_away_eggs

/-- Proves that Joyce ends up with 9 eggs given the initial conditions and transactions -/
theorem joyce_final_egg_count :
  final_egg_count 8 3.5 0.5 2 = 9 := by sorry

end NUMINAMATH_CALUDE_joyce_final_egg_count_l2582_258209


namespace NUMINAMATH_CALUDE_negation_existential_equivalence_l2582_258237

theorem negation_existential_equivalence (f : ℝ → ℝ) :
  (¬ ∃ x₀ : ℝ, f x₀ < 0) ↔ (∀ x : ℝ, f x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_existential_equivalence_l2582_258237


namespace NUMINAMATH_CALUDE_degree_of_5m2n3_l2582_258263

/-- The degree of a monomial is the sum of the exponents of its variables. -/
def degree_of_monomial (m : ℕ) (n : ℕ) : ℕ := m + n

/-- The monomial 5m^2n^3 has degree 5. -/
theorem degree_of_5m2n3 : degree_of_monomial 2 3 = 5 := by sorry

end NUMINAMATH_CALUDE_degree_of_5m2n3_l2582_258263


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l2582_258236

theorem imaginary_part_of_complex_product (i : ℂ) : i * i = -1 → Complex.im (i * (1 + i) * i) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l2582_258236


namespace NUMINAMATH_CALUDE_bart_survey_earnings_l2582_258205

theorem bart_survey_earnings :
  let questions_per_survey : ℕ := 10
  let earnings_per_question : ℚ := 0.2
  let monday_surveys : ℕ := 3
  let tuesday_surveys : ℕ := 4
  
  let total_questions := questions_per_survey * (monday_surveys + tuesday_surveys)
  let total_earnings := (total_questions : ℚ) * earnings_per_question

  total_earnings = 14 :=
by sorry

end NUMINAMATH_CALUDE_bart_survey_earnings_l2582_258205


namespace NUMINAMATH_CALUDE_pencil_count_problem_l2582_258297

/-- Given an initial number of pencils, a number of lost pencils, and a number of gained pencils,
    calculate the final number of pencils. -/
def finalPencilCount (initial lost gained : ℕ) : ℕ :=
  initial - lost + gained

/-- Theorem stating that given the specific values in the problem,
    the final pencil count is 2060. -/
theorem pencil_count_problem :
  finalPencilCount 2015 5 50 = 2060 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_problem_l2582_258297


namespace NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l2582_258261

/-- Ellipse C with equation x²/4 + y²/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Line l with equation y = kx + m -/
def line_l (k m x y : ℝ) : Prop := y = k*x + m

/-- Point A is the right vertex of the ellipse -/
def point_A : ℝ × ℝ := (2, 0)

/-- Circle with diameter MN passes through point A -/
def circle_passes_through_A (M N : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  (x₁ - 2) * (x₂ - 2) + y₁ * y₂ = 0

theorem ellipse_intersection_fixed_point (k m : ℝ) :
  ∃ (M N : ℝ × ℝ),
    ellipse_C M.1 M.2 ∧
    ellipse_C N.1 N.2 ∧
    line_l k m M.1 M.2 ∧
    line_l k m N.1 N.2 ∧
    circle_passes_through_A M N →
    line_l k m (2/7) 0 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l2582_258261


namespace NUMINAMATH_CALUDE_power_difference_evaluation_l2582_258275

theorem power_difference_evaluation : (3^3)^4 - (4^4)^3 = -16245775 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_evaluation_l2582_258275


namespace NUMINAMATH_CALUDE_fourth_power_nested_square_roots_l2582_258288

theorem fourth_power_nested_square_roots :
  (Real.sqrt (1 + Real.sqrt (2 + Real.sqrt (3 + Real.sqrt 4))))^4 = 3 + Real.sqrt 5 + 2 * Real.sqrt (2 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_fourth_power_nested_square_roots_l2582_258288


namespace NUMINAMATH_CALUDE_quadratic_function_uniqueness_quadratic_function_coefficient_range_l2582_258289

-- Part 1
def is_symmetric_about_negative_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-1 - x) = f (-1 + x)

theorem quadratic_function_uniqueness
  (f : ℝ → ℝ)
  (h1 : ∃ a b c, ∀ x, f x = a * x^2 + b * x + c)
  (h2 : is_symmetric_about_negative_one f)
  (h3 : f 0 = 1)
  (h4 : ∃ x_min, ∀ x, f x ≥ f x_min ∧ f x_min = 0) :
  ∀ x, f x = (x + 1)^2 := by sorry

-- Part 2
theorem quadratic_function_coefficient_range
  (b : ℝ)
  (h : ∀ x ∈ Set.Ioo 0 1, |x^2 + b*x| ≤ 1) :
  b ∈ Set.Icc (-2) 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_uniqueness_quadratic_function_coefficient_range_l2582_258289


namespace NUMINAMATH_CALUDE_largest_four_digit_multiple_of_48_l2582_258280

theorem largest_four_digit_multiple_of_48 : 
  (∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 48 = 0 → n ≤ 9984) ∧ 
  9984 % 48 = 0 ∧ 
  9984 ≤ 9999 ∧ 
  9984 ≥ 1000 := by
sorry

end NUMINAMATH_CALUDE_largest_four_digit_multiple_of_48_l2582_258280


namespace NUMINAMATH_CALUDE_p_true_and_q_false_l2582_258239

-- Define proposition p
def p : Prop := ∀ x y : ℝ, x > y → -x < -y

-- Define proposition q
def q : Prop := ∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 → (1 / x > 1 / y → x < y)

-- Theorem to prove
theorem p_true_and_q_false : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_p_true_and_q_false_l2582_258239


namespace NUMINAMATH_CALUDE_count_integer_length_segments_in_specific_triangle_l2582_258207

-- Define the right triangle DEF
def RightTriangleDEF (DE EF : ℝ) : Prop :=
  DE = 24 ∧ EF = 25

-- Define the function to count integer length segments
def CountIntegerLengthSegments (DE EF : ℝ) : ℕ :=
  -- The actual counting logic would go here
  sorry

-- Theorem statement
theorem count_integer_length_segments_in_specific_triangle :
  ∀ DE EF : ℝ, RightTriangleDEF DE EF →
  CountIntegerLengthSegments DE EF = 14 :=
by
  sorry


end NUMINAMATH_CALUDE_count_integer_length_segments_in_specific_triangle_l2582_258207


namespace NUMINAMATH_CALUDE_solve_equation_solve_system_l2582_258235

-- Problem 1
theorem solve_equation (x : ℝ) : (x + 2) / 3 - 1 = (1 - x) / 2 ↔ x = 1 := by sorry

-- Problem 2
theorem solve_system (x y : ℝ) : x + 2*y = 8 ∧ 3*x - 4*y = 4 ↔ x = 4 ∧ y = 2 := by sorry

end NUMINAMATH_CALUDE_solve_equation_solve_system_l2582_258235


namespace NUMINAMATH_CALUDE_monday_pages_to_reach_average_l2582_258292

def target_average : ℕ := 50
def days_in_week : ℕ := 7
def known_pages : List ℕ := [43, 28, 0, 70, 56, 88]

theorem monday_pages_to_reach_average :
  ∃ (monday_pages : ℕ),
    (monday_pages + known_pages.sum) / days_in_week = target_average ∧
    monday_pages = 65 := by
  sorry

end NUMINAMATH_CALUDE_monday_pages_to_reach_average_l2582_258292


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l2582_258260

/-- Given two nonconstant geometric sequences with different common ratios,
    if a specific condition is met, prove that the sum of their common ratios is 3. -/
theorem sum_of_common_ratios_is_three
  (k a₂ a₃ b₂ b₃ : ℝ)
  (hk : k ≠ 0)
  (ha : ∃ p : ℝ, p ≠ 1 ∧ a₂ = k * p ∧ a₃ = k * p^2)
  (hb : ∃ r : ℝ, r ≠ 1 ∧ b₂ = k * r ∧ b₃ = k * r^2)
  (hdiff : ∀ p r : ℝ, (a₂ = k * p ∧ a₃ = k * p^2 ∧ b₂ = k * r ∧ b₃ = k * r^2) → p ≠ r)
  (hcond : a₃ - b₃ = 3 * (a₂ - b₂)) :
  ∃ p r : ℝ, (a₂ = k * p ∧ a₃ = k * p^2 ∧ b₂ = k * r ∧ b₃ = k * r^2) ∧ p + r = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l2582_258260


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2582_258249

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  (1 / (a + 3*b) + 1 / (b + 3*c) + 1 / (c + 3*a)) ≥ 3/4 ∧ 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 3 ∧ 
    1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x) = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2582_258249


namespace NUMINAMATH_CALUDE_motorboat_travel_time_l2582_258293

/-- Represents the time (in hours) for the motorboat to travel from dock C to dock D -/
def motorboat_time_to_D : ℝ := 5.5

/-- Represents the total journey time in hours -/
def total_journey_time : ℝ := 12

/-- Represents the time (in hours) the motorboat stops at dock E -/
def stop_time_at_E : ℝ := 1

theorem motorboat_travel_time :
  motorboat_time_to_D = (total_journey_time - stop_time_at_E) / 2 :=
sorry

end NUMINAMATH_CALUDE_motorboat_travel_time_l2582_258293


namespace NUMINAMATH_CALUDE_wheel_distance_theorem_l2582_258216

/-- Represents the properties and movement of a wheel -/
structure Wheel where
  rotations_per_minute : ℕ
  cm_per_rotation : ℕ

/-- Calculates the distance in meters that a wheel moves in one hour -/
def distance_in_one_hour (w : Wheel) : ℚ :=
  (w.rotations_per_minute * 60 * w.cm_per_rotation) / 100

/-- Theorem stating that a wheel with given properties moves 420 meters in one hour -/
theorem wheel_distance_theorem (w : Wheel) 
  (h1 : w.rotations_per_minute = 20) 
  (h2 : w.cm_per_rotation = 35) : 
  distance_in_one_hour w = 420 := by
  sorry

#eval distance_in_one_hour ⟨20, 35⟩

end NUMINAMATH_CALUDE_wheel_distance_theorem_l2582_258216


namespace NUMINAMATH_CALUDE_cyclist_speed_proof_l2582_258220

/-- The distance between Town A and Town B in miles -/
def distance_AB : ℝ := 80

/-- The speed difference between Cyclist Y and Cyclist X in mph -/
def speed_difference : ℝ := 6

/-- The distance from Town B where the cyclists meet after Cyclist Y turns back, in miles -/
def meeting_distance : ℝ := 20

/-- The speed of Cyclist X in mph -/
def speed_X : ℝ := 9

/-- The speed of Cyclist Y in mph -/
def speed_Y : ℝ := speed_X + speed_difference

theorem cyclist_speed_proof :
  speed_X * ((distance_AB + meeting_distance) / speed_Y) = distance_AB - meeting_distance :=
sorry

end NUMINAMATH_CALUDE_cyclist_speed_proof_l2582_258220


namespace NUMINAMATH_CALUDE_volume_ratio_of_cubes_l2582_258230

-- Define the edge lengths in inches
def small_cube_edge : ℚ := 4
def large_cube_edge : ℚ := 24  -- 2 feet = 24 inches

-- Define the volumes of the cubes
def small_cube_volume : ℚ := small_cube_edge ^ 3
def large_cube_volume : ℚ := large_cube_edge ^ 3

-- Theorem statement
theorem volume_ratio_of_cubes : 
  small_cube_volume / large_cube_volume = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_of_cubes_l2582_258230


namespace NUMINAMATH_CALUDE_ratio_of_linear_system_l2582_258240

theorem ratio_of_linear_system (x y c d : ℝ) 
  (eq1 : 9 * x - 6 * y = c)
  (eq2 : 15 * x - 10 * y = d)
  (h1 : d ≠ 0)
  (h2 : x ≠ 0)
  (h3 : y ≠ 0) :
  c / d = -2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_linear_system_l2582_258240


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l2582_258271

theorem set_intersection_theorem (m : ℝ) : 
  let A := {x : ℝ | x ≥ 3}
  let B := {x : ℝ | x < m}
  (A ∪ B = Set.univ) ∧ (A ∩ B = ∅) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l2582_258271


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l2582_258277

def P : Set ℝ := {x | x^2 - 2*x ≥ 0}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}

theorem complement_P_intersect_Q : (Set.univ \ P) ∩ Q = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l2582_258277


namespace NUMINAMATH_CALUDE_odd_function_negative_values_l2582_258258

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_values
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_nonneg : ∀ x ≥ 0, f x = x + 1) :
  ∀ x < 0, f x = x - 1 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_negative_values_l2582_258258


namespace NUMINAMATH_CALUDE_total_distance_to_fountain_l2582_258279

/-- The distance from Mrs. Hilt's desk to the water fountain in feet -/
def distance_to_fountain : ℕ := 30

/-- The number of trips Mrs. Hilt makes to the water fountain -/
def number_of_trips : ℕ := 4

/-- Theorem: The total distance Mrs. Hilt walks to the water fountain is 120 feet -/
theorem total_distance_to_fountain :
  distance_to_fountain * number_of_trips = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_to_fountain_l2582_258279


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2582_258225

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, -1, 2*a-2}

theorem subset_implies_a_equals_one (a : ℝ) : A a ⊆ B a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2582_258225


namespace NUMINAMATH_CALUDE_symmetry_condition_range_on_interval_range_positive_l2582_258243

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2 - a) * x - 2 * a

-- Theorem for symmetry condition
theorem symmetry_condition (a : ℝ) :
  (∀ x : ℝ, f a (1 + x) = f a (1 - x)) → a = 4 :=
sorry

-- Theorem for range on [0,4] when a = 4
theorem range_on_interval :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → -9 ≤ f 4 x ∧ f 4 x ≤ -5) ∧
  (∃ x y : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 4 ∧ f 4 x = -9 ∧ f 4 y = -5) :=
sorry

-- Theorem for the range of x when f(x) > 0
theorem range_positive (a : ℝ) (x : ℝ) :
  (a = -2 → (f a x > 0 ↔ x ≠ -2)) ∧
  (a > -2 → (f a x > 0 ↔ x < -2 ∨ x > a)) ∧
  (a < -2 → (f a x > 0 ↔ -2 < x ∧ x < a)) :=
sorry

end NUMINAMATH_CALUDE_symmetry_condition_range_on_interval_range_positive_l2582_258243


namespace NUMINAMATH_CALUDE_rectangle_count_in_5x5_grid_l2582_258253

/-- The number of ways to select a rectangle in a 5x5 grid -/
def rectangleCount : ℕ := 225

/-- The number of horizontal or vertical lines in a 5x5 grid, including boundaries -/
def lineCount : ℕ := 6

theorem rectangle_count_in_5x5_grid :
  rectangleCount = (lineCount.choose 2) * (lineCount.choose 2) :=
sorry

end NUMINAMATH_CALUDE_rectangle_count_in_5x5_grid_l2582_258253


namespace NUMINAMATH_CALUDE_special_function_sum_l2582_258234

/-- A function satisfying f(p+q) = f(p)f(q) for all p and q, and f(1) = 3 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ p q : ℝ, f (p + q) = f p * f q) ∧ (f 1 = 3)

/-- The main theorem to prove -/
theorem special_function_sum (f : ℝ → ℝ) (h : special_function f) :
  (f 1^2 + f 2) / f 1 + (f 2^2 + f 4) / f 3 + (f 3^2 + f 6) / f 5 +
  (f 4^2 + f 8) / f 7 + (f 5^2 + f 10) / f 9 = 30 := by
  sorry

end NUMINAMATH_CALUDE_special_function_sum_l2582_258234


namespace NUMINAMATH_CALUDE_rotate_parabola_180_l2582_258294

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rotates a point 180 degrees around the origin -/
def rotate180 (x y : ℝ) : ℝ × ℝ := (-x, -y)

/-- The original parabola -/
def originalParabola : Parabola := ⟨1, -5, 9⟩

/-- Theorem stating that rotating the original parabola 180 degrees results in the new parabola -/
theorem rotate_parabola_180 :
  let (x, y) := rotate180 x y
  y = -(originalParabola.a * x^2 + originalParabola.b * x + originalParabola.c) :=
by sorry

end NUMINAMATH_CALUDE_rotate_parabola_180_l2582_258294


namespace NUMINAMATH_CALUDE_book_sale_percentage_gain_l2582_258299

/-- Calculates the percentage gain for a book sale given the number of books purchased,
    the number of books whose selling price equals the total cost price,
    and the total number of books purchased. -/
def calculatePercentageGain (booksPurchased : ℕ) (booksSoldForCost : ℕ) : ℚ :=
  ((booksPurchased : ℚ) / booksSoldForCost - 1) * 100

/-- Theorem stating that the percentage gain for the given book sale scenario is (3/7) * 100. -/
theorem book_sale_percentage_gain :
  calculatePercentageGain 50 35 = (3/7) * 100 := by
  sorry

#eval calculatePercentageGain 50 35

end NUMINAMATH_CALUDE_book_sale_percentage_gain_l2582_258299


namespace NUMINAMATH_CALUDE_number_conditions_l2582_258201

theorem number_conditions (x y : ℝ) : 
  (0.65 * x > 26) → 
  (0.4 * y < -3) → 
  ((x - y)^2 ≥ 100) → 
  (x > 40 ∧ y < -7.5) := by
sorry

end NUMINAMATH_CALUDE_number_conditions_l2582_258201


namespace NUMINAMATH_CALUDE_geometric_sum_first_8_terms_l2582_258273

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_8_terms :
  let a : ℚ := 1/3
  let r : ℚ := 1/2
  let n : ℕ := 8
  geometric_sum a r n = 85/128 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_first_8_terms_l2582_258273


namespace NUMINAMATH_CALUDE_earthquake_aid_calculation_l2582_258203

/-- Calculates the total financial aid for a school with high school and junior high students -/
def total_financial_aid (total_students : ℕ) (hs_rate : ℕ) (jhs_rate : ℕ) (hs_exclusion_rate : ℚ) : ℕ :=
  651700

/-- The total financial aid for the given school conditions is 651,700 yuan -/
theorem earthquake_aid_calculation :
  let total_students : ℕ := 1862
  let hs_rate : ℕ := 500
  let jhs_rate : ℕ := 350
  let hs_exclusion_rate : ℚ := 30 / 100
  total_financial_aid total_students hs_rate jhs_rate hs_exclusion_rate = 651700 := by
  sorry

end NUMINAMATH_CALUDE_earthquake_aid_calculation_l2582_258203


namespace NUMINAMATH_CALUDE_cosine_sine_relation_l2582_258254

theorem cosine_sine_relation (x : ℝ) :
  2 * Real.cos x + 3 * Real.sin x = 4 →
  Real.cos x = 8 / 13 ∧ Real.sin x = 12 / 13 →
  3 * Real.cos x - 2 * Real.sin x = 0 := by
sorry

end NUMINAMATH_CALUDE_cosine_sine_relation_l2582_258254


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l2582_258247

/-- Given a quadratic expression x^2 - 16x + 15, when rewritten in the form (x+d)^2 + e,
    the sum of d and e is -57. -/
theorem quadratic_rewrite_sum (d e : ℝ) : 
  (∀ x, x^2 - 16*x + 15 = (x+d)^2 + e) → d + e = -57 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l2582_258247


namespace NUMINAMATH_CALUDE_fraction_ceiling_evaluation_l2582_258242

theorem fraction_ceiling_evaluation : 
  (⌈(23 : ℚ) / 11 - ⌈(31 : ℚ) / 19⌉⌉) / (⌈(35 : ℚ) / 9 + ⌈(9 * 19 : ℚ) / 35⌉⌉) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ceiling_evaluation_l2582_258242


namespace NUMINAMATH_CALUDE_circle_line_intersection_l2582_258233

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y - 15 = 0

/-- The line equation -/
def line_eq (x y m : ℝ) : Prop := (1 + 3*m)*x + (3 - 2*m)*y + 4*m - 17 = 0

/-- The theorem stating that the circle and line always intersect at two points -/
theorem circle_line_intersection :
  ∃ (p q : ℝ × ℝ), p ≠ q ∧
    (∀ m : ℝ, circle_eq p.1 p.2 ∧ line_eq p.1 p.2 m) ∧
    (∀ m : ℝ, circle_eq q.1 q.2 ∧ line_eq q.1 q.2 m) ∧
    (∀ r : ℝ × ℝ, (∀ m : ℝ, circle_eq r.1 r.2 ∧ line_eq r.1 r.2 m) → r = p ∨ r = q) :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l2582_258233


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l2582_258298

theorem trig_expression_equals_one : 
  (Real.sin (20 * π / 180) * Real.cos (15 * π / 180) + 
   Real.cos (160 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (25 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l2582_258298


namespace NUMINAMATH_CALUDE_ellipse_m_range_l2582_258255

/-- An ellipse equation with parameter m -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (m + 2) - y^2 / (m + 1) = 1

/-- Condition for foci on y-axis -/
def foci_on_y_axis (m : ℝ) : Prop :=
  -(m + 1) > m + 2 ∧ m + 2 > 0

/-- Theorem stating the range of m for the given ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, (∃ x y : ℝ, ellipse_equation x y m) ∧ foci_on_y_axis m ↔ -2 < m ∧ m < -3/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l2582_258255


namespace NUMINAMATH_CALUDE_tampa_bay_bucs_players_l2582_258295

/-- The initial number of football players in the Tampa Bay Bucs team. -/
def initial_football_players : ℕ := 13

/-- The initial number of cheerleaders in the Tampa Bay Bucs team. -/
def initial_cheerleaders : ℕ := 16

/-- The number of football players who quit. -/
def quitting_football_players : ℕ := 10

/-- The number of cheerleaders who quit. -/
def quitting_cheerleaders : ℕ := 4

/-- The total number of people left after some quit. -/
def remaining_total : ℕ := 15

theorem tampa_bay_bucs_players :
  initial_football_players = 13 ∧
  (initial_football_players - quitting_football_players) +
  (initial_cheerleaders - quitting_cheerleaders) = remaining_total :=
sorry

end NUMINAMATH_CALUDE_tampa_bay_bucs_players_l2582_258295


namespace NUMINAMATH_CALUDE_fraction_simplification_specific_case_l2582_258212

theorem fraction_simplification (a b c : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

theorem specific_case : 
  let a : ℚ := 12
  let b : ℚ := 16
  let c : ℚ := 9
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = 37 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_specific_case_l2582_258212


namespace NUMINAMATH_CALUDE_samantha_birth_year_l2582_258248

def first_amc_year : ℕ := 1985

def samantha_age_at_seventh_amc : ℕ := 12

theorem samantha_birth_year :
  ∃ (birth_year : ℕ),
    birth_year = first_amc_year + 6 - samantha_age_at_seventh_amc ∧
    birth_year = 1979 :=
by sorry

end NUMINAMATH_CALUDE_samantha_birth_year_l2582_258248


namespace NUMINAMATH_CALUDE_max_quotient_value_l2582_258200

theorem max_quotient_value (a b : ℝ) (ha : 200 ≤ a ∧ a ≤ 400) (hb : 600 ≤ b ∧ b ≤ 1200) :
  (∀ x y, 200 ≤ x ∧ x ≤ 400 → 600 ≤ y ∧ y ≤ 1200 → y / x ≤ b / a) →
  b / a = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l2582_258200
