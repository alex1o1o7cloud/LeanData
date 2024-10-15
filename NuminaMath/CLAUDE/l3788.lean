import Mathlib

namespace NUMINAMATH_CALUDE_polygon_symmetry_l3788_378857

-- Define a convex polygon
def ConvexPolygon : Type := sorry

-- Define a point inside a polygon
def PointInside (P : ConvexPolygon) : Type := sorry

-- Define a line passing through a point
def LineThroughPoint (P : ConvexPolygon) (O : PointInside P) : Type := sorry

-- Define the property of a line dividing the polygon area in half
def DividesAreaInHalf (P : ConvexPolygon) (O : PointInside P) (l : LineThroughPoint P O) : Prop := sorry

-- Define central symmetry of a polygon
def CentrallySymmetric (P : ConvexPolygon) : Prop := sorry

-- Define center of symmetry
def CenterOfSymmetry (P : ConvexPolygon) (O : PointInside P) : Prop := sorry

-- The main theorem
theorem polygon_symmetry (P : ConvexPolygon) (O : PointInside P) 
  (h : ∀ (l : LineThroughPoint P O), DividesAreaInHalf P O l) : 
  CentrallySymmetric P ∧ CenterOfSymmetry P O := by
  sorry

end NUMINAMATH_CALUDE_polygon_symmetry_l3788_378857


namespace NUMINAMATH_CALUDE_distance_A_to_C_l3788_378824

/-- Proves that the distance between city A and C is 300 km given the provided conditions -/
theorem distance_A_to_C (
  eddy_time : ℝ)
  (freddy_time : ℝ)
  (distance_A_to_B : ℝ)
  (speed_ratio : ℝ)
  (h1 : eddy_time = 3)
  (h2 : freddy_time = 4)
  (h3 : distance_A_to_B = 510)
  (h4 : speed_ratio = 2.2666666666666666)
  : ℝ := by
  sorry

#check distance_A_to_C

end NUMINAMATH_CALUDE_distance_A_to_C_l3788_378824


namespace NUMINAMATH_CALUDE_set_operations_l3788_378818

def A : Set ℤ := {x : ℤ | |x| < 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5}

theorem set_operations :
  (B ∩ C = {3}) ∧
  (B ∪ C = {1, 2, 3, 4, 5}) ∧
  (A ∪ (B ∩ C) = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}) ∧
  (A ∩ (A \ (B ∪ C)) = {-5, -4, -3, -2, -1, 0}) := by
sorry

end NUMINAMATH_CALUDE_set_operations_l3788_378818


namespace NUMINAMATH_CALUDE_always_quadratic_l3788_378870

/-- A quadratic equation is of the form ax^2 + bx + c = 0 where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation (a^2 + 1)x^2 + bx + c = 0 is always a quadratic equation -/
theorem always_quadratic (a b c : ℝ) :
  is_quadratic_equation (fun x => (a^2 + 1) * x^2 + b * x + c) :=
sorry

end NUMINAMATH_CALUDE_always_quadratic_l3788_378870


namespace NUMINAMATH_CALUDE_golden_ratio_exponential_monotonicity_l3788_378862

theorem golden_ratio_exponential_monotonicity 
  (a : ℝ) 
  (f : ℝ → ℝ) 
  (m n : ℝ) 
  (h1 : a = (Real.sqrt 5 - 1) / 2) 
  (h2 : ∀ x, f x = a ^ x) 
  (h3 : f m > f n) : 
  m < n := by
sorry

end NUMINAMATH_CALUDE_golden_ratio_exponential_monotonicity_l3788_378862


namespace NUMINAMATH_CALUDE_currency_notes_theorem_l3788_378894

theorem currency_notes_theorem (x y z : ℕ) : 
  x + y + z = 130 →
  95 * x + 45 * y + 20 * z = 7000 →
  75 * x + 25 * y = 4400 := by
sorry

end NUMINAMATH_CALUDE_currency_notes_theorem_l3788_378894


namespace NUMINAMATH_CALUDE_magnitude_of_vector_combination_l3788_378825

/-- Given two vectors a and b in R^2, prove that the magnitude of 2a - b is √17 -/
theorem magnitude_of_vector_combination (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (3, -2) → ‖2 • a - b‖ = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_combination_l3788_378825


namespace NUMINAMATH_CALUDE_four_digit_number_theorem_l3788_378847

/-- Represents a four-digit number ABCD --/
structure FourDigitNumber where
  value : ℕ
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Checks if a four-digit number contains no nines --/
def no_nines (n : FourDigitNumber) : Prop :=
  ∀ d, d ∈ n.value.digits 10 → d ≠ 9

/-- Extracts the first two digits of a four-digit number --/
def first_two_digits (n : FourDigitNumber) : ℕ := n.value / 100

/-- Extracts the last two digits of a four-digit number --/
def last_two_digits (n : FourDigitNumber) : ℕ := n.value % 100

/-- Extracts the first digit of a four-digit number --/
def first_digit (n : FourDigitNumber) : ℕ := n.value / 1000

/-- Extracts the second digit of a four-digit number --/
def second_digit (n : FourDigitNumber) : ℕ := (n.value / 100) % 10

/-- Extracts the third digit of a four-digit number --/
def third_digit (n : FourDigitNumber) : ℕ := (n.value / 10) % 10

/-- Extracts the fourth digit of a four-digit number --/
def fourth_digit (n : FourDigitNumber) : ℕ := n.value % 10

/-- Checks if a quadratic equation ax² + bx + c = 0 has real roots --/
def has_real_roots (a b c : ℝ) : Prop := b^2 - 4*a*c ≥ 0

theorem four_digit_number_theorem (n : FourDigitNumber) 
  (h_no_nines : no_nines n)
  (h_eq1 : has_real_roots (first_digit n : ℝ) (second_digit n : ℝ) (last_two_digits n : ℝ))
  (h_eq2 : has_real_roots (first_digit n : ℝ) ((n.value / 10) % 100 : ℝ) (fourth_digit n : ℝ))
  (h_eq3 : has_real_roots (first_two_digits n : ℝ) (third_digit n : ℝ) (fourth_digit n : ℝ))
  (h_leading : first_digit n ≠ 0 ∧ second_digit n ≠ 0) :
  n.value = 1710 ∨ n.value = 1810 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_theorem_l3788_378847


namespace NUMINAMATH_CALUDE_secant_length_l3788_378877

noncomputable section

def Circle (O : ℝ × ℝ) (R : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = R^2}

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def isTangent (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) := 
  ∃! P, P ∈ l ∩ c

def isSecant (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) :=
  ∃ P Q, P ≠ Q ∧ P ∈ l ∩ c ∧ Q ∈ l ∩ c

def isEquidistant (P : ℝ × ℝ) (Q : ℝ × ℝ) (l : Set (ℝ × ℝ)) :=
  ∀ X ∈ l, distance P X = distance Q X

theorem secant_length (O : ℝ × ℝ) (R : ℝ) (A : ℝ × ℝ) 
  (h1 : distance O A = 2 * R)
  (c : Set (ℝ × ℝ)) (h2 : c = Circle O R)
  (t : Set (ℝ × ℝ)) (h3 : isTangent t c)
  (s : Set (ℝ × ℝ)) (h4 : isSecant s c)
  (B : ℝ × ℝ) (h5 : B ∈ t ∩ c)
  (h6 : isEquidistant O B s) :
  ∃ C G : ℝ × ℝ, C ∈ s ∩ c ∧ G ∈ s ∩ c ∧ distance C G = 2 * R * Real.sqrt (10/13) :=
sorry

end NUMINAMATH_CALUDE_secant_length_l3788_378877


namespace NUMINAMATH_CALUDE_total_wage_proof_l3788_378839

/-- The weekly payment for employee B -/
def wage_B : ℝ := 249.99999999999997

/-- The weekly payment for employee A -/
def wage_A : ℝ := 1.2 * wage_B

/-- The total weekly payment for both employees -/
def total_wage : ℝ := wage_A + wage_B

theorem total_wage_proof : total_wage = 549.9999999999999 := by sorry

end NUMINAMATH_CALUDE_total_wage_proof_l3788_378839


namespace NUMINAMATH_CALUDE_max_value_of_function_l3788_378850

theorem max_value_of_function (x : ℝ) (h : -1 < x ∧ x < 1) : 
  (∀ y : ℝ, -1 < y ∧ y < 1 → x / (x - 1) + x ≥ y / (y - 1) + y) → 
  x / (x - 1) + x = 0 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3788_378850


namespace NUMINAMATH_CALUDE_divisible_by_seven_l3788_378826

theorem divisible_by_seven (n : ℕ) : 7 ∣ (6^(2*n+1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l3788_378826


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3788_378871

theorem ratio_x_to_y (x y : ℝ) (h : y = 0.25 * x) : x / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3788_378871


namespace NUMINAMATH_CALUDE_cube_decomposition_largest_number_l3788_378858

theorem cube_decomposition_largest_number :
  let n : ℕ := 10
  let sum_of_terms : ℕ → ℕ := λ k => k * (k + 1) / 2
  let total_terms : ℕ := sum_of_terms n - sum_of_terms 1
  2 * total_terms + 1 = 109 :=
by sorry

end NUMINAMATH_CALUDE_cube_decomposition_largest_number_l3788_378858


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l3788_378866

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l3788_378866


namespace NUMINAMATH_CALUDE_intersection_A_notB_when_a_is_neg_two_union_A_B_equals_B_implies_a_range_l3788_378830

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Define the complement of B
def notB : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- Theorem for part I
theorem intersection_A_notB_when_a_is_neg_two : 
  A (-2) ∩ notB = {x | -1 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for part II
theorem union_A_B_equals_B_implies_a_range (a : ℝ) : 
  A a ∪ B = B → a < -4 ∨ a > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_A_notB_when_a_is_neg_two_union_A_B_equals_B_implies_a_range_l3788_378830


namespace NUMINAMATH_CALUDE_proportion_solution_l3788_378856

theorem proportion_solution (y : ℝ) : y / 1.35 = 5 / 9 → y = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l3788_378856


namespace NUMINAMATH_CALUDE_course_passing_logic_l3788_378827

variable (Student : Type)
variable (answered_correctly : Student → Prop)
variable (passed_course : Student → Prop)

theorem course_passing_logic :
  (∀ s : Student, answered_correctly s → passed_course s) →
  (∀ s : Student, ¬passed_course s → ¬answered_correctly s) :=
by sorry

end NUMINAMATH_CALUDE_course_passing_logic_l3788_378827


namespace NUMINAMATH_CALUDE_abs_neg_sqrt_six_l3788_378855

theorem abs_neg_sqrt_six : |(-Real.sqrt 6)| = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_sqrt_six_l3788_378855


namespace NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_power_20_l3788_378859

theorem smallest_k_for_64_power_gt_4_power_20 : ∃ k : ℕ, k = 7 ∧ (∀ m : ℕ, 64^m > 4^20 → m ≥ k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_power_20_l3788_378859


namespace NUMINAMATH_CALUDE_first_year_after_2010_with_sum_of_digits_10_l3788_378878

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def isFirstYearAfter2010WithSumOfDigits10 (year : ℕ) : Prop :=
  year > 2010 ∧ 
  sumOfDigits year = 10 ∧
  ∀ y, 2010 < y ∧ y < year → sumOfDigits y ≠ 10

theorem first_year_after_2010_with_sum_of_digits_10 :
  isFirstYearAfter2010WithSumOfDigits10 2017 := by
  sorry

end NUMINAMATH_CALUDE_first_year_after_2010_with_sum_of_digits_10_l3788_378878


namespace NUMINAMATH_CALUDE_greatest_c_for_quadratic_range_l3788_378808

theorem greatest_c_for_quadratic_range (c : ℤ) : 
  (∀ x : ℝ, x^2 + c*x + 18 ≠ -6) ↔ c ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_c_for_quadratic_range_l3788_378808


namespace NUMINAMATH_CALUDE_total_accidents_l3788_378841

/-- Represents the number of vehicles involved in accidents per 100 million vehicles -/
def A (k : ℝ) (x : ℝ) : ℝ := 96 + k * x

/-- The constant k for morning hours -/
def k_morning : ℝ := 1

/-- The constant k for evening hours -/
def k_evening : ℝ := 3

/-- The number of vehicles (in billions) during morning hours -/
def x_morning : ℝ := 2

/-- The number of vehicles (in billions) during evening hours -/
def x_evening : ℝ := 1

/-- Theorem stating the total number of vehicles involved in accidents -/
theorem total_accidents : 
  A k_morning (100 * x_morning) + A k_evening (100 * x_evening) = 5192 := by
  sorry

end NUMINAMATH_CALUDE_total_accidents_l3788_378841


namespace NUMINAMATH_CALUDE_exterior_angle_not_sum_of_adjacent_angles_l3788_378835

-- Define a triangle with interior angles A, B, C and exterior angle A_ext at vertex A
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  A_ext : ℝ

-- State the theorem
theorem exterior_angle_not_sum_of_adjacent_angles (t : Triangle) : 
  t.A_ext ≠ t.B + t.C :=
sorry

end NUMINAMATH_CALUDE_exterior_angle_not_sum_of_adjacent_angles_l3788_378835


namespace NUMINAMATH_CALUDE_library_book_return_days_l3788_378875

theorem library_book_return_days 
  (daily_charge : ℚ)
  (total_books : ℕ)
  (days_for_one_book : ℕ)
  (total_cost : ℚ)
  (h1 : daily_charge = 1/2)
  (h2 : total_books = 3)
  (h3 : days_for_one_book = 20)
  (h4 : total_cost = 41) :
  (total_cost - daily_charge * days_for_one_book) / (daily_charge * 2) = 31 := by
sorry

end NUMINAMATH_CALUDE_library_book_return_days_l3788_378875


namespace NUMINAMATH_CALUDE_unfair_coin_flip_probability_l3788_378809

/-- The probability of flipping exactly k tails in n flips of an unfair coin -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of flipping exactly 3 tails in 8 flips of an unfair coin with 2/3 probability of tails -/
theorem unfair_coin_flip_probability : 
  binomial_probability 8 3 (2/3) = 448/6561 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_flip_probability_l3788_378809


namespace NUMINAMATH_CALUDE_discount_rate_for_profit_margin_l3788_378820

/-- Proves that a 20% discount rate maintains a 20% profit margin for a toy gift box. -/
theorem discount_rate_for_profit_margin :
  let cost_price : ℝ := 160
  let marked_price : ℝ := 240
  let profit_margin : ℝ := 0.2
  let discount_rate : ℝ := 0.2
  let discounted_price : ℝ := marked_price * (1 - discount_rate)
  let profit : ℝ := discounted_price - cost_price
  profit / cost_price = profit_margin :=
by
  sorry

#check discount_rate_for_profit_margin

end NUMINAMATH_CALUDE_discount_rate_for_profit_margin_l3788_378820


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3788_378811

theorem trigonometric_identities :
  -- Part 1
  ¬∃x : ℝ, x = Real.sin (-14 / 3 * π) + Real.cos (20 / 3 * π) + Real.tan (-53 / 6 * π) ∧
  -- Part 2
  Real.tan (675 * π / 180) - Real.sin (-330 * π / 180) - Real.cos (960 * π / 180) = 0 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3788_378811


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3788_378849

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 ∧ x₁^2 - 4*x₁ - 1 = 0 ∧ x₂^2 - 4*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -1/3 ∧ x₂ = 2 ∧ x₁*(3*x₁ + 1) = 2*(3*x₁ + 1) ∧ x₂*(3*x₂ + 1) = 2*(3*x₂ + 1)) ∧
  (∃ x₁ x₂ : ℝ, x₁ = (-1 + Real.sqrt 33) / 4 ∧ x₂ = (-1 - Real.sqrt 33) / 4 ∧ 2*x₁^2 + x₁ - 4 = 0 ∧ 2*x₂^2 + x₂ - 4 = 0) ∧
  (∀ x : ℝ, 4*x^2 - 3*x + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3788_378849


namespace NUMINAMATH_CALUDE_symmetric_quadratic_property_symmetric_quadratic_comparison_l3788_378897

/-- A quadratic function with a positive leading coefficient and symmetric about x = 2 -/
def symmetric_quadratic (a b c : ℝ) (h : a > 0) : ℝ → ℝ :=
  fun x ↦ a * x^2 + b * x + c

theorem symmetric_quadratic_property {a b c : ℝ} (h : a > 0) :
  ∀ x, symmetric_quadratic a b c h (2 + x) = symmetric_quadratic a b c h (2 - x) :=
by sorry

theorem symmetric_quadratic_comparison {a b c : ℝ} (h : a > 0) :
  symmetric_quadratic a b c h 0.5 > symmetric_quadratic a b c h π :=
by sorry

end NUMINAMATH_CALUDE_symmetric_quadratic_property_symmetric_quadratic_comparison_l3788_378897


namespace NUMINAMATH_CALUDE_no_club_member_is_fraternity_member_l3788_378828

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

end NUMINAMATH_CALUDE_no_club_member_is_fraternity_member_l3788_378828


namespace NUMINAMATH_CALUDE_ace_distribution_probability_l3788_378834

def num_players : ℕ := 4
def num_cards : ℕ := 32
def num_aces : ℕ := 4
def cards_per_player : ℕ := num_cards / num_players

theorem ace_distribution_probability :
  let remaining_players := num_players - 1
  let remaining_cards := num_cards - cards_per_player
  let p_no_ace_for_one := 1 / num_players
  let p_two_aces_for_others := 
    (Nat.choose remaining_players 1 * Nat.choose num_aces 2 * Nat.choose (remaining_cards - num_aces) (cards_per_player - 2)) /
    (Nat.choose remaining_cards cards_per_player)
  p_two_aces_for_others = 8 / 11 :=
sorry

end NUMINAMATH_CALUDE_ace_distribution_probability_l3788_378834


namespace NUMINAMATH_CALUDE_fraction_sum_division_simplification_l3788_378853

theorem fraction_sum_division_simplification :
  (3 : ℚ) / 7 + 5 / 8 + 1 / 3 / ((5 : ℚ) / 12 + 2 / 9) = 2097 / 966 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_division_simplification_l3788_378853


namespace NUMINAMATH_CALUDE_homothety_circle_transformation_l3788_378848

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℚ

/-- Applies a homothety transformation to a point -/
def homothety (center : Point) (scale : ℚ) (p : Point) : Point :=
  { x := center.x + scale * (p.x - center.x)
  , y := center.y + scale * (p.y - center.y) }

theorem homothety_circle_transformation :
  let O : Point := { x := 3, y := 4 }
  let originalCircle : Circle := { center := O, radius := 8 }
  let P : Point := { x := 11, y := 12 }
  let scale : ℚ := 2/3
  let newCenter : Point := homothety P scale O
  let newRadius : ℚ := scale * originalCircle.radius
  newCenter.x = 17/3 ∧ newCenter.y = 20/3 ∧ newRadius = 16/3 :=
by sorry

end NUMINAMATH_CALUDE_homothety_circle_transformation_l3788_378848


namespace NUMINAMATH_CALUDE_company_fund_distribution_l3788_378815

/-- Represents the company fund distribution problem -/
theorem company_fund_distribution (n : ℕ) 
  (h1 : 50 * n + 130 = 60 * n - 10) : 
  60 * n - 10 = 830 :=
by
  sorry

#check company_fund_distribution

end NUMINAMATH_CALUDE_company_fund_distribution_l3788_378815


namespace NUMINAMATH_CALUDE_sons_age_l3788_378836

theorem sons_age (father_age : ℕ) (h1 : father_age = 38) : ℕ :=
  let son_age := 14
  let years_ago := 10
  have h2 : father_age - years_ago = 7 * (son_age - years_ago) := by sorry
  son_age

#check sons_age

end NUMINAMATH_CALUDE_sons_age_l3788_378836


namespace NUMINAMATH_CALUDE_quadratic_roots_l3788_378803

theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 1
  ∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3788_378803


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l3788_378819

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_to_scientific_notation :
  toScientificNotation (40.9 * 1000000000) =
    ScientificNotation.mk 4.09 9 (by sorry) := by sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l3788_378819


namespace NUMINAMATH_CALUDE_add_negative_two_l3788_378880

theorem add_negative_two : 1 + (-2) = -1 := by sorry

end NUMINAMATH_CALUDE_add_negative_two_l3788_378880


namespace NUMINAMATH_CALUDE_cube_sum_equality_l3788_378801

theorem cube_sum_equality (a b c : ℕ+) (h : a^3 + b^3 + c^3 = 3*a*b*c) : a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equality_l3788_378801


namespace NUMINAMATH_CALUDE_negation_of_implication_l3788_378840

theorem negation_of_implication (a b c : ℝ) :
  ¬(a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3788_378840


namespace NUMINAMATH_CALUDE_tailoring_cost_james_suits_tailoring_cost_l3788_378896

theorem tailoring_cost (cost_first_suit : ℕ) (total_cost : ℕ) : ℕ :=
  let cost_second_suit := 3 * cost_first_suit
  let tailoring_cost := total_cost - cost_first_suit - cost_second_suit
  tailoring_cost

theorem james_suits_tailoring_cost : tailoring_cost 300 1400 = 200 := by
  sorry

end NUMINAMATH_CALUDE_tailoring_cost_james_suits_tailoring_cost_l3788_378896


namespace NUMINAMATH_CALUDE_neds_video_games_l3788_378814

theorem neds_video_games (non_working : ℕ) (price_per_game : ℕ) (total_earned : ℕ) :
  non_working = 6 →
  price_per_game = 7 →
  total_earned = 63 →
  non_working + (total_earned / price_per_game) = 15 :=
by sorry

end NUMINAMATH_CALUDE_neds_video_games_l3788_378814


namespace NUMINAMATH_CALUDE_min_value_expression_l3788_378823

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 2018) + (y + 1/x) * (y + 1/x - 2018) ≥ -2036162 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3788_378823


namespace NUMINAMATH_CALUDE_inverse_mod_53_l3788_378889

theorem inverse_mod_53 (h : (15⁻¹ : ZMod 53) = 31) : (38⁻¹ : ZMod 53) = 22 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_53_l3788_378889


namespace NUMINAMATH_CALUDE_vacation_cost_share_l3788_378868

/-- Calculates each person's share of vacation costs -/
theorem vacation_cost_share
  (num_people : ℕ)
  (airbnb_cost : ℕ)
  (car_cost : ℕ)
  (h1 : num_people = 8)
  (h2 : airbnb_cost = 3200)
  (h3 : car_cost = 800) :
  (airbnb_cost + car_cost) / num_people = 500 := by
  sorry

#check vacation_cost_share

end NUMINAMATH_CALUDE_vacation_cost_share_l3788_378868


namespace NUMINAMATH_CALUDE_opposite_of_neg_two_l3788_378846

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- Theorem: The opposite of -2 is 2 -/
theorem opposite_of_neg_two : opposite (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_two_l3788_378846


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l3788_378874

def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

theorem f_strictly_increasing :
  (∀ x y, x < y ∧ y < -1 → f x < f y) ∧
  (∀ x y, 11 < x ∧ x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l3788_378874


namespace NUMINAMATH_CALUDE_robie_chocolates_l3788_378869

/-- Calculates the number of chocolate bags left after a series of purchases and giveaways. -/
def chocolates_left (initial_purchase : ℕ) (given_away : ℕ) (additional_purchase : ℕ) : ℕ :=
  initial_purchase - given_away + additional_purchase

/-- Proves that given the specific scenario, 4 bags of chocolates are left. -/
theorem robie_chocolates : chocolates_left 3 2 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_robie_chocolates_l3788_378869


namespace NUMINAMATH_CALUDE_euler_identity_complex_power_cexp_sum_bound_cexp_diff_not_always_bounded_l3788_378890

-- Define the complex exponential function
noncomputable def cexp (x : ℝ) : ℂ := Complex.exp (x * Complex.I)

-- Euler's formula
axiom euler_formula (x : ℝ) : cexp x = Complex.cos x + Complex.I * Complex.sin x

-- Theorem 1
theorem euler_identity : cexp π + 1 = 0 := by sorry

-- Theorem 2
theorem complex_power : (Complex.ofReal (1/2) + Complex.I * Complex.ofReal (Real.sqrt 3 / 2)) ^ 2022 = 1 := by sorry

-- Theorem 3
theorem cexp_sum_bound (x : ℝ) : Complex.abs (cexp x + cexp (-x)) ≤ 2 := by sorry

-- Theorem 4
theorem cexp_diff_not_always_bounded :
  ¬ (∀ x : ℝ, -2 ≤ (cexp x - cexp (-x)).re ∧ (cexp x - cexp (-x)).re ≤ 2 ∧
               -2 ≤ (cexp x - cexp (-x)).im ∧ (cexp x - cexp (-x)).im ≤ 2) := by sorry

end NUMINAMATH_CALUDE_euler_identity_complex_power_cexp_sum_bound_cexp_diff_not_always_bounded_l3788_378890


namespace NUMINAMATH_CALUDE_find_m_value_l3788_378833

theorem find_m_value (a : ℝ) (m : ℝ) : 
  (∀ x, 2*x^2 - 3*x + a < 0 ↔ m < x ∧ x < 1) →
  (2*m^2 - 3*m + a = 0 ∧ 2*1^2 - 3*1 + a = 0) →
  m = 1/2 := by sorry

end NUMINAMATH_CALUDE_find_m_value_l3788_378833


namespace NUMINAMATH_CALUDE_executive_committee_formation_l3788_378821

/-- Represents the number of members in each department -/
def membersPerDepartment : ℕ := 10

/-- Represents the total number of departments -/
def totalDepartments : ℕ := 3

/-- Represents the size of the executive committee -/
def committeeSize : ℕ := 5

/-- Represents the total number of club members -/
def totalMembers : ℕ := membersPerDepartment * totalDepartments

/-- Calculates the number of ways to choose the executive committee -/
def waysToChooseCommittee : ℕ := 
  membersPerDepartment ^ totalDepartments * (Nat.choose (totalMembers - totalDepartments) (committeeSize - totalDepartments))

theorem executive_committee_formation :
  waysToChooseCommittee = 351000 := by sorry

end NUMINAMATH_CALUDE_executive_committee_formation_l3788_378821


namespace NUMINAMATH_CALUDE_class_size_l3788_378876

/-- The number of students in a class with given language course enrollments -/
theorem class_size (french : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : french = 41)
  (h2 : german = 22)
  (h3 : both = 9)
  (h4 : neither = 6) :
  french + german - both + neither = 60 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l3788_378876


namespace NUMINAMATH_CALUDE_two_planes_parallel_to_same_line_are_parallel_two_planes_parallel_to_same_line_not_always_parallel_l3788_378882

-- Define the concept of a plane in 3D space
variable (Plane : Type)

-- Define the concept of a line in 3D space
variable (Line : Type)

-- Define the parallel relation between planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the parallel relation between a plane and a line
variable (parallel_plane_line : Plane → Line → Prop)

-- Theorem to be proven false
theorem two_planes_parallel_to_same_line_are_parallel 
  (p1 p2 : Plane) (l : Line) : 
  parallel_plane_line p1 l → parallel_plane_line p2 l → parallel_planes p1 p2 := by
  sorry

-- The actual theorem should be that the above statement is false
theorem two_planes_parallel_to_same_line_not_always_parallel : 
  ¬∀ (p1 p2 : Plane) (l : Line), 
    parallel_plane_line p1 l → parallel_plane_line p2 l → parallel_planes p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_two_planes_parallel_to_same_line_are_parallel_two_planes_parallel_to_same_line_not_always_parallel_l3788_378882


namespace NUMINAMATH_CALUDE_garage_sale_pricing_l3788_378854

theorem garage_sale_pricing (prices : Finset ℕ) (radio_price : ℕ) (n : ℕ) :
  prices.card = 36 →
  prices.toList.Nodup →
  radio_price ∈ prices →
  (prices.filter (λ x => x > radio_price)).card = n - 1 →
  (prices.filter (λ x => x < radio_price)).card = 21 →
  n = 16 := by
  sorry

end NUMINAMATH_CALUDE_garage_sale_pricing_l3788_378854


namespace NUMINAMATH_CALUDE_slope_angle_l3788_378831

def line_equation (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 2 = 0

theorem slope_angle (x y : ℝ) (h : line_equation x y) : 
  ∃ (θ : ℝ), θ = 120 * Real.pi / 180 ∧ Real.tan θ = -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_slope_angle_l3788_378831


namespace NUMINAMATH_CALUDE_triangulation_count_l3788_378867

/-- A triangulation of a square with marked interior points. -/
structure SquareTriangulation where
  /-- The number of marked points inside the square. -/
  num_points : ℕ
  /-- The number of triangles in the triangulation. -/
  num_triangles : ℕ

/-- Theorem stating the number of triangles in a specific triangulation. -/
theorem triangulation_count (t : SquareTriangulation) 
  (h_points : t.num_points = 100) : 
  t.num_triangles = 202 := by sorry

end NUMINAMATH_CALUDE_triangulation_count_l3788_378867


namespace NUMINAMATH_CALUDE_flags_on_circular_track_l3788_378804

/-- The number of flags needed on a circular track -/
def num_flags (track_length : ℕ) (flag_interval : ℕ) : ℕ :=
  (track_length / flag_interval) + 1

/-- Theorem: 5 flags are needed for a 400m track with 90m intervals -/
theorem flags_on_circular_track :
  num_flags 400 90 = 5 := by
  sorry

end NUMINAMATH_CALUDE_flags_on_circular_track_l3788_378804


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3788_378838

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (k : ℕ), k = 5 ∧ (3830 - k) % 15 = 0 ∧ ∀ (m : ℕ), m < k → (3830 - m) % 15 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3788_378838


namespace NUMINAMATH_CALUDE_smallest_valid_six_digit_number_l3788_378817

def is_valid_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2 ∧ n % 5 = 2 ∧ n % 6 = 2

def append_three_digits (n : ℕ) (m : ℕ) : ℕ :=
  n * 1000 + m

theorem smallest_valid_six_digit_number :
  ∃ (n m : ℕ),
    is_valid_three_digit n ∧
    m < 1000 ∧
    let six_digit := append_three_digits n m
    six_digit = 122040 ∧
    six_digit % 4 = 0 ∧
    six_digit % 5 = 0 ∧
    six_digit % 6 = 0 ∧
    ∀ (n' m' : ℕ),
      is_valid_three_digit n' ∧
      m' < 1000 ∧
      let six_digit' := append_three_digits n' m'
      six_digit' % 4 = 0 ∧
      six_digit' % 5 = 0 ∧
      six_digit' % 6 = 0 →
      six_digit ≤ six_digit' :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_six_digit_number_l3788_378817


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3788_378884

theorem absolute_value_inequality (x : ℝ) : 
  (2 ≤ |x - 1| ∧ |x - 1| ≤ 5) ↔ ((-4 ≤ x ∧ x ≤ -1) ∨ (3 ≤ x ∧ x ≤ 6)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3788_378884


namespace NUMINAMATH_CALUDE_two_visit_days_365_l3788_378888

def alice_visits (d : ℕ) : Bool := d % 4 = 0
def bianca_visits (d : ℕ) : Bool := d % 6 = 0
def carmen_visits (d : ℕ) : Bool := d % 8 = 0

def exactly_two_visit (d : ℕ) : Bool :=
  let visit_count := (alice_visits d).toNat + (bianca_visits d).toNat + (carmen_visits d).toNat
  visit_count = 2

def count_two_visit_days (n : ℕ) : ℕ :=
  (List.range n).filter exactly_two_visit |>.length

theorem two_visit_days_365 :
  count_two_visit_days 365 = 45 := by
  sorry

end NUMINAMATH_CALUDE_two_visit_days_365_l3788_378888


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3788_378887

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  let sum_of_roots := -b / a
  (∀ x, f x = 0 ↔ (x - sum_of_roots / 2)^2 = (sum_of_roots^2 - 4 * (b^2 - 4*a*c) / (4*a)) / 4) →
  sum_of_roots = 5 ↔ a = 1 ∧ b = -5 ∧ c = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3788_378887


namespace NUMINAMATH_CALUDE_smallest_prime_for_divisibility_l3788_378861

theorem smallest_prime_for_divisibility : ∃ (p : ℕ), 
  Nat.Prime p ∧ 
  (11002 + p) % 11 = 0 ∧ 
  (11002 + p) % 7 = 0 ∧
  ∀ (q : ℕ), Nat.Prime q → (11002 + q) % 11 = 0 → (11002 + q) % 7 = 0 → p ≤ q :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_smallest_prime_for_divisibility_l3788_378861


namespace NUMINAMATH_CALUDE_electronic_components_production_ahead_of_schedule_l3788_378898

theorem electronic_components_production_ahead_of_schedule 
  (total_components : ℕ) 
  (planned_days : ℕ) 
  (additional_daily_production : ℕ) : 
  total_components = 15000 → 
  planned_days = 30 → 
  additional_daily_production = 250 → 
  (planned_days - (total_components / ((total_components / planned_days) + additional_daily_production))) = 10 := by
sorry

end NUMINAMATH_CALUDE_electronic_components_production_ahead_of_schedule_l3788_378898


namespace NUMINAMATH_CALUDE_problem_solution_l3788_378851

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem problem_solution :
  (∀ m : ℝ, m = 4 → A ∪ B m = {x | -2 ≤ x ∧ x ≤ 7}) ∧
  (∀ m : ℝ, (B m ∩ A = B m) ↔ m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3788_378851


namespace NUMINAMATH_CALUDE_system_solutions_l3788_378883

def equation1 (x y : ℝ) : Prop := (x + 2*y) * (x + 3*y) = x + y

def equation2 (x y : ℝ) : Prop := (2*x + y) * (3*x + y) = -99 * (x + y)

def solution_set : Set (ℝ × ℝ) :=
  {(0, 0), (-14, 6), (-85/6, 35/6)}

theorem system_solutions :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l3788_378883


namespace NUMINAMATH_CALUDE_unique_four_letter_product_l3788_378852

def letter_value (c : Char) : ℕ :=
  c.toNat - 'A'.toNat + 1

def four_letter_product (s : String) : ℕ :=
  if s.length = 4 then
    s.foldl (fun acc c => acc * letter_value c) 1
  else
    0

theorem unique_four_letter_product : ∀ s : String,
  s.length = 4 ∧ s ≠ "MNOQ" ∧ four_letter_product s = four_letter_product "MNOQ" →
  s = "NOQZ" :=
sorry

end NUMINAMATH_CALUDE_unique_four_letter_product_l3788_378852


namespace NUMINAMATH_CALUDE_projection_bound_implies_coverage_l3788_378865

/-- A figure in a metric space -/
class Figure (α : Type*) [MetricSpace α]

/-- The projection of a figure onto a line -/
def projection (α : Type*) [MetricSpace α] (Φ : Figure α) (l : Set α) : ℝ := sorry

/-- A figure Φ is covered by a circle of diameter d -/
def covered_by_circle (α : Type*) [MetricSpace α] (Φ : Figure α) (d : ℝ) : Prop := sorry

theorem projection_bound_implies_coverage 
  (α : Type*) [MetricSpace α] (Φ : Figure α) :
  (∀ l : Set α, projection α Φ l ≤ 1) →
  (¬ covered_by_circle α Φ 1) ∧ (covered_by_circle α Φ 1.5) := by sorry

end NUMINAMATH_CALUDE_projection_bound_implies_coverage_l3788_378865


namespace NUMINAMATH_CALUDE_dave_initial_apps_l3788_378802

/-- The number of apps Dave initially had on his phone -/
def initial_apps : ℕ := 15

/-- The number of apps Dave added -/
def added_apps : ℕ := 71

/-- The number of apps Dave had left after deleting some -/
def remaining_apps : ℕ := 14

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := added_apps + 1

theorem dave_initial_apps : 
  initial_apps + added_apps - deleted_apps = remaining_apps :=
by sorry

end NUMINAMATH_CALUDE_dave_initial_apps_l3788_378802


namespace NUMINAMATH_CALUDE_dance_partners_l3788_378842

theorem dance_partners (total_participants : ℕ) (n : ℕ) : 
  total_participants = 42 →
  (∀ k : ℕ, k ≥ 1 ∧ k ≤ n → k + 6 ≤ total_participants - n) →
  n + 6 = total_participants - n →
  n = 18 ∧ total_participants - n = 24 :=
by sorry

end NUMINAMATH_CALUDE_dance_partners_l3788_378842


namespace NUMINAMATH_CALUDE_line_through_point_with_given_slope_l3788_378873

/-- Given a line L1: 2x + y - 10 = 0 and a point P(1, 0),
    prove that the line L2 passing through P with the same slope as L1
    has the equation 2x + y - 2 = 0 -/
theorem line_through_point_with_given_slope (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 2 * x + y - 10 = 0
  let P : ℝ × ℝ := (1, 0)
  let L2 : ℝ → ℝ → Prop := λ x y => 2 * x + y - 2 = 0
  (∀ x y, L1 x y ↔ 2 * x + y = 10) →
  (L2 (P.1) (P.2)) →
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → (y₂ - y₁) / (x₂ - x₁) = (y - P.2) / (x - P.1)) →
  ∀ x y, L2 x y ↔ 2 * x + y = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_with_given_slope_l3788_378873


namespace NUMINAMATH_CALUDE_sebastian_grade_size_l3788_378863

/-- The number of students in a grade where a student is ranked both the 70th best and 70th worst -/
def num_students (rank_best : ℕ) (rank_worst : ℕ) : ℕ :=
  (rank_best - 1) + 1 + (rank_worst - 1)

/-- Theorem stating that if a student is ranked both the 70th best and 70th worst, 
    then there are 139 students in total -/
theorem sebastian_grade_size :
  num_students 70 70 = 139 := by
  sorry

end NUMINAMATH_CALUDE_sebastian_grade_size_l3788_378863


namespace NUMINAMATH_CALUDE_angle_330_equivalent_to_negative_30_l3788_378810

/-- Two angles have the same terminal side if they are equivalent modulo 360° -/
def same_terminal_side (a b : ℝ) : Prop := a % 360 = b % 360

/-- The problem statement -/
theorem angle_330_equivalent_to_negative_30 :
  same_terminal_side 330 (-30) := by sorry

end NUMINAMATH_CALUDE_angle_330_equivalent_to_negative_30_l3788_378810


namespace NUMINAMATH_CALUDE_star_equal_is_four_lines_l3788_378813

-- Define the ⋆ operation
def star (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Define the set of points (x, y) where x ⋆ y = y ⋆ x
def star_equal_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

-- Define the union of four lines
def four_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2}

-- Theorem statement
theorem star_equal_is_four_lines : star_equal_set = four_lines := by
  sorry

end NUMINAMATH_CALUDE_star_equal_is_four_lines_l3788_378813


namespace NUMINAMATH_CALUDE_center_distance_of_isosceles_triangle_l3788_378845

/-- An isosceles triangle with two sides of length 6 and one side of length 10 -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  is_isosceles : side1 = side2
  side_lengths : side1 = 6 ∧ base = 10

/-- The distance between the centers of the circumscribed and inscribed circles of the triangle -/
def center_distance (t : IsoscelesTriangle) : ℝ := sorry

/-- Theorem stating the distance between the centers of the circumscribed and inscribed circles -/
theorem center_distance_of_isosceles_triangle (t : IsoscelesTriangle) :
  center_distance t = (5 * Real.sqrt 110) / 11 := by sorry

end NUMINAMATH_CALUDE_center_distance_of_isosceles_triangle_l3788_378845


namespace NUMINAMATH_CALUDE_license_plate_count_l3788_378860

/-- The number of vowels available for the license plate. -/
def num_vowels : ℕ := 6

/-- The number of consonants available for the license plate. -/
def num_consonants : ℕ := 20

/-- The number of digits available for the license plate. -/
def num_digits : ℕ := 10

/-- The number of special characters available for the license plate. -/
def num_special_chars : ℕ := 2

/-- The total number of possible license plates. -/
def total_license_plates : ℕ := num_vowels * num_consonants * num_digits * num_consonants * num_special_chars

theorem license_plate_count : total_license_plates = 48000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3788_378860


namespace NUMINAMATH_CALUDE_lloyd_house_of_cards_solution_l3788_378837

/-- Represents the number of cards in Lloyd's house of cards problem -/
def lloyd_house_of_cards (decks : ℕ) (cards_per_deck : ℕ) (layers : ℕ) : ℕ :=
  (decks * cards_per_deck) / layers

/-- Theorem stating the number of cards per layer in Lloyd's house of cards -/
theorem lloyd_house_of_cards_solution :
  lloyd_house_of_cards 24 78 48 = 39 := by
  sorry

#eval lloyd_house_of_cards 24 78 48

end NUMINAMATH_CALUDE_lloyd_house_of_cards_solution_l3788_378837


namespace NUMINAMATH_CALUDE_rectangle_uncovered_area_l3788_378872

/-- The area of the portion of a rectangle not covered by four circles --/
theorem rectangle_uncovered_area (rectangle_length : ℝ) (rectangle_width : ℝ) (circle_radius : ℝ) :
  rectangle_length = 4 →
  rectangle_width = 8 →
  circle_radius = 1 →
  (rectangle_length * rectangle_width) - (4 * Real.pi * circle_radius ^ 2) = 32 - 4 * Real.pi := by
  sorry

#check rectangle_uncovered_area

end NUMINAMATH_CALUDE_rectangle_uncovered_area_l3788_378872


namespace NUMINAMATH_CALUDE_lowest_possible_price_l3788_378886

def manufacturer_price : ℝ := 45.00
def max_regular_discount : ℝ := 0.30
def additional_sale_discount : ℝ := 0.20

theorem lowest_possible_price :
  let regular_discounted_price := manufacturer_price * (1 - max_regular_discount)
  let final_price := regular_discounted_price * (1 - additional_sale_discount)
  final_price = 25.20 := by
sorry

end NUMINAMATH_CALUDE_lowest_possible_price_l3788_378886


namespace NUMINAMATH_CALUDE_total_components_is_900_l3788_378885

/-- Represents the total number of components --/
def total_components : ℕ := 900

/-- Represents the number of type B components --/
def type_b_components : ℕ := 300

/-- Represents the number of type C components --/
def type_c_components : ℕ := 200

/-- Represents the sample size --/
def sample_size : ℕ := 45

/-- Represents the number of type A components in the sample --/
def sample_type_a : ℕ := 20

/-- Represents the number of type C components in the sample --/
def sample_type_c : ℕ := 10

/-- Theorem stating that the total number of components is 900 --/
theorem total_components_is_900 :
  total_components = 900 ∧
  type_b_components = 300 ∧
  type_c_components = 200 ∧
  sample_size = 45 ∧
  sample_type_a = 20 ∧
  sample_type_c = 10 ∧
  (sample_type_c : ℚ) / (sample_size : ℚ) = (type_c_components : ℚ) / (total_components : ℚ) :=
by sorry

#check total_components_is_900

end NUMINAMATH_CALUDE_total_components_is_900_l3788_378885


namespace NUMINAMATH_CALUDE_box_volume_calculation_l3788_378832

/-- The conversion factor from feet to meters -/
def feet_to_meters : ℝ := 0.3048

/-- The edge length of each box in feet -/
def edge_length_feet : ℝ := 5

/-- The number of boxes -/
def num_boxes : ℕ := 4

/-- The total volume of the boxes in cubic meters -/
def total_volume : ℝ := 14.144

theorem box_volume_calculation :
  (num_boxes : ℝ) * (edge_length_feet * feet_to_meters)^3 = total_volume := by
  sorry

end NUMINAMATH_CALUDE_box_volume_calculation_l3788_378832


namespace NUMINAMATH_CALUDE_cafe_menu_problem_l3788_378895

theorem cafe_menu_problem (total_dishes : ℕ) 
  (vegan_ratio : ℚ) (gluten_ratio : ℚ) (nut_ratio : ℚ) :
  total_dishes = 30 →
  vegan_ratio = 1 / 3 →
  gluten_ratio = 2 / 5 →
  nut_ratio = 1 / 4 →
  (total_dishes : ℚ) * vegan_ratio * (1 - gluten_ratio - nut_ratio) / total_dishes = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cafe_menu_problem_l3788_378895


namespace NUMINAMATH_CALUDE_set_problem_l3788_378881

def A : Set ℝ := {2, 4}
def B (a : ℝ) : Set ℝ := {a, 3*a}

theorem set_problem (a : ℝ) :
  (A ⊆ B a → 4/3 ≤ a ∧ a ≤ 2) ∧
  (A ∩ B a ≠ ∅ → 2/3 < a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_set_problem_l3788_378881


namespace NUMINAMATH_CALUDE_complex_on_line_l3788_378807

theorem complex_on_line (z : ℂ) (a : ℝ) : 
  z = (1 - a * Complex.I) / Complex.I →
  (z.re : ℝ) + 2 * (z.im : ℝ) + 5 = 0 →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_on_line_l3788_378807


namespace NUMINAMATH_CALUDE_certain_number_value_l3788_378806

theorem certain_number_value : ∃ x : ℝ, 15 * x = 165 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l3788_378806


namespace NUMINAMATH_CALUDE_shaded_square_area_l3788_378805

/-- Represents a figure with five squares and two right-angled triangles -/
structure GeometricFigure where
  square1 : ℝ
  square2 : ℝ
  square3 : ℝ
  square4 : ℝ
  square5 : ℝ

/-- The theorem stating the area of the shaded square -/
theorem shaded_square_area (fig : GeometricFigure) 
  (h1 : fig.square1 = 5)
  (h2 : fig.square2 = 8)
  (h3 : fig.square3 = 32) :
  fig.square5 = 45 := by
  sorry


end NUMINAMATH_CALUDE_shaded_square_area_l3788_378805


namespace NUMINAMATH_CALUDE_triangle_properties_l3788_378829

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition for the triangle -/
def satisfies_condition (t : Triangle) : Prop :=
  2 * Real.sqrt 2 * (Real.sin t.A ^ 2 - Real.sin t.C ^ 2) = (t.a - t.b) * Real.sin t.B

/-- The circumradius of the triangle is √2 -/
def has_circumradius_sqrt2 (t : Triangle) : Prop :=
  ∃ (R : ℝ), R = Real.sqrt 2

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : has_circumradius_sqrt2 t) : 
  t.C = Real.pi / 3 ∧ 
  ∃ (S : ℝ), S ≤ 3 * Real.sqrt 3 / 2 ∧ 
  (∀ (S' : ℝ), S' = 1/2 * t.a * t.b * Real.sin t.C → S' ≤ S) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3788_378829


namespace NUMINAMATH_CALUDE_equation_solution_l3788_378892

theorem equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ * (x₁ - 2) + x₁ - 2 = 0) ∧ 
  (x₂ * (x₂ - 2) + x₂ - 2 = 0) ∧ 
  x₁ = 2 ∧ x₂ = -1 ∧ 
  ∀ x : ℝ, x * (x - 2) + x - 2 = 0 → (x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3788_378892


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3788_378844

theorem smallest_integer_with_remainders (k : ℕ) : k = 61 ↔ 
  (k > 1) ∧ 
  (∀ m : ℕ, m < k → 
    (m % 12 ≠ 1 ∨ m % 5 ≠ 1 ∨ m % 3 ≠ 1)) ∧
  (k % 12 = 1 ∧ k % 5 = 1 ∧ k % 3 = 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3788_378844


namespace NUMINAMATH_CALUDE_abcd_inequality_l3788_378879

theorem abcd_inequality (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_eq : a^2/(1+a^2) + b^2/(1+b^2) + c^2/(1+c^2) + d^2/(1+d^2) = 1) : 
  a * b * c * d ≤ 1/9 := by
sorry

end NUMINAMATH_CALUDE_abcd_inequality_l3788_378879


namespace NUMINAMATH_CALUDE_program_list_orders_l3788_378800

/-- Represents the number of items in the program list -/
def n : ℕ := 6

/-- Represents the number of items that must be adjacent -/
def adjacent_items : ℕ := 2

/-- Represents the number of slots available for inserting the item that can't be first -/
def available_slots : ℕ := n - 1

/-- Calculates the number of different orders for the program list -/
def program_orders : ℕ :=
  (Nat.factorial (n - adjacent_items + 1)) *
  (Nat.choose available_slots 1) *
  (Nat.factorial adjacent_items)

theorem program_list_orders :
  program_orders = 192 := by sorry

end NUMINAMATH_CALUDE_program_list_orders_l3788_378800


namespace NUMINAMATH_CALUDE_log_difference_l3788_378816

theorem log_difference (a b c d : ℕ+) 
  (h1 : (Real.log b) / (Real.log a) = 3/2)
  (h2 : (Real.log d) / (Real.log c) = 5/4)
  (h3 : a - c = 9) :
  b - d = 93 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_l3788_378816


namespace NUMINAMATH_CALUDE_companion_pair_expression_l3788_378891

/-- Definition of a companion pair -/
def is_companion_pair (m n : ℝ) : Prop :=
  m / 2 + n / 3 = (m + n) / 5

/-- Theorem: For any companion pair (m, n), the expression 
    m - (22/3)n - [4m - 2(3n - 1)] equals -2 -/
theorem companion_pair_expression (m n : ℝ) 
  (h : is_companion_pair m n) : 
  m - (22/3) * n - (4 * m - 2 * (3 * n - 1)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_companion_pair_expression_l3788_378891


namespace NUMINAMATH_CALUDE_broken_line_length_bound_l3788_378899

/-- A broken line is represented as a list of points -/
def BrokenLine := List ℝ × ℝ

/-- A rectangle is represented by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Predicate to check if a broken line is inside a rectangle -/
def isInside (bl : BrokenLine) (rect : Rectangle) : Prop := sorry

/-- Predicate to check if every line parallel to the sides of the rectangle
    intersects the broken line at most once -/
def intersectsAtMostOnce (bl : BrokenLine) (rect : Rectangle) : Prop := sorry

/-- Function to calculate the length of a broken line -/
def length (bl : BrokenLine) : ℝ := sorry

/-- Theorem: If a broken line is inside a rectangle and every line parallel to the sides
    of the rectangle intersects the broken line at most once, then the length of the
    broken line is less than the sum of the lengths of two adjacent sides of the rectangle -/
theorem broken_line_length_bound (bl : BrokenLine) (rect : Rectangle) :
  isInside bl rect →
  intersectsAtMostOnce bl rect →
  length bl < rect.width + rect.height := by
  sorry

end NUMINAMATH_CALUDE_broken_line_length_bound_l3788_378899


namespace NUMINAMATH_CALUDE_minimal_moves_l3788_378864

/-- Represents a permutation of 2n numbers -/
def Permutation (n : ℕ) := Fin (2 * n) → Fin (2 * n)

/-- Represents a move that can be applied to a permutation -/
inductive Move (n : ℕ)
  | swap : Fin (2 * n) → Fin (2 * n) → Move n
  | cyclic : Fin (2 * n) → Fin (2 * n) → Fin (2 * n) → Move n

/-- Applies a move to a permutation -/
def applyMove (n : ℕ) (p : Permutation n) (m : Move n) : Permutation n :=
  sorry

/-- Checks if a permutation is in increasing order -/
def isIncreasing (n : ℕ) (p : Permutation n) : Prop :=
  sorry

/-- The main theorem: n moves are necessary and sufficient -/
theorem minimal_moves (n : ℕ) :
  (∃ (moves : List (Move n)), moves.length = n ∧
    ∀ (p : Permutation n), ∃ (appliedMoves : List (Move n)),
      appliedMoves.length ≤ n ∧
      isIncreasing n (appliedMoves.foldl (applyMove n) p)) ∧
  (∀ (k : ℕ), k < n →
    ∃ (p : Permutation n), ∀ (moves : List (Move n)),
      moves.length ≤ k → ¬isIncreasing n (moves.foldl (applyMove n) p)) :=
  sorry

end NUMINAMATH_CALUDE_minimal_moves_l3788_378864


namespace NUMINAMATH_CALUDE_matrix_homomorphism_implies_equal_dim_l3788_378812

-- Define the set of valid dimensions
def ValidDim : Set ℕ := {2, 3}

-- Define the property of the bijective function
def IsMatrixHomomorphism {n p : ℕ} (f : Matrix (Fin n) (Fin n) ℂ → Matrix (Fin p) (Fin p) ℂ) : Prop :=
  ∀ X Y : Matrix (Fin n) (Fin n) ℂ, f (X * Y) = f X * f Y

-- The main theorem
theorem matrix_homomorphism_implies_equal_dim (n p : ℕ) 
  (hn : n ∈ ValidDim) (hp : p ∈ ValidDim) :
  (∃ f : Matrix (Fin n) (Fin n) ℂ → Matrix (Fin p) (Fin p) ℂ, 
    Function.Bijective f ∧ IsMatrixHomomorphism f) → n = p := by
  sorry

end NUMINAMATH_CALUDE_matrix_homomorphism_implies_equal_dim_l3788_378812


namespace NUMINAMATH_CALUDE_infinite_solutions_cube_fifth_square_l3788_378822

theorem infinite_solutions_cube_fifth_square (x y z : ℕ+) (k : ℕ+) 
  (h : x^3 + y^5 = z^2) :
  (k^10 * x)^3 + (k^6 * y)^5 = (k^15 * z)^2 := by
  sorry

#check infinite_solutions_cube_fifth_square

end NUMINAMATH_CALUDE_infinite_solutions_cube_fifth_square_l3788_378822


namespace NUMINAMATH_CALUDE_division_problem_l3788_378893

theorem division_problem (a b c d : ℚ) 
  (h1 : a / b = 3)
  (h2 : b / c = 5/3)
  (h3 : c / d = 2) :
  d / a = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3788_378893


namespace NUMINAMATH_CALUDE_total_letters_in_names_l3788_378843

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

end NUMINAMATH_CALUDE_total_letters_in_names_l3788_378843
