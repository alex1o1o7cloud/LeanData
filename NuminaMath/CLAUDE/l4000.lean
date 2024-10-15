import Mathlib

namespace NUMINAMATH_CALUDE_two_dice_outcomes_l4000_400000

/-- The number of possible outcomes for a single die. -/
def outcomes_per_die : ℕ := 6

/-- The total number of possible outcomes when throwing two identical dice simultaneously. -/
def total_outcomes : ℕ := outcomes_per_die * outcomes_per_die

/-- Theorem stating that the total number of possible outcomes when throwing two identical dice simultaneously is 36. -/
theorem two_dice_outcomes : total_outcomes = 36 := by
  sorry

end NUMINAMATH_CALUDE_two_dice_outcomes_l4000_400000


namespace NUMINAMATH_CALUDE_inverse_sum_zero_l4000_400010

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 1; 7, 3]

theorem inverse_sum_zero :
  ∃ (B : Matrix (Fin 2) (Fin 2) ℝ), 
    A * B = 1 ∧ B * A = 1 →
    B 0 0 + B 0 1 + B 1 0 + B 1 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_zero_l4000_400010


namespace NUMINAMATH_CALUDE_triangle_count_l4000_400077

/-- The number of triangles formed by three mutually intersecting line segments
    in a configuration of n points on a circle, where n ≥ 6 and
    any three line segments do not intersect at a single point inside the circle. -/
def num_triangles (n : ℕ) : ℕ :=
  Nat.choose n 3 + 4 * Nat.choose n 4 + 5 * Nat.choose n 5 + Nat.choose n 6

/-- Theorem stating the number of triangles formed under the given conditions. -/
theorem triangle_count (n : ℕ) (h : n ≥ 6) :
  num_triangles n = Nat.choose n 3 + 4 * Nat.choose n 4 + 5 * Nat.choose n 5 + Nat.choose n 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_l4000_400077


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l4000_400078

theorem reciprocal_sum_theorem (a b c : ℝ) (h : 1 / a + 1 / b = 1 / c) : c = a * b / (b + a) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l4000_400078


namespace NUMINAMATH_CALUDE_even_function_extension_l4000_400094

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- State the theorem
theorem even_function_extension
  (f : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_nonneg : ∀ x : ℝ, x ≥ 0 → f x = 2^x + 1) :
  ∀ x : ℝ, x < 0 → f x = 2^(-x) + 1 :=
sorry

end NUMINAMATH_CALUDE_even_function_extension_l4000_400094


namespace NUMINAMATH_CALUDE_fuel_cost_calculation_l4000_400081

/-- Calculates the total cost to fill up both a truck's diesel tank and a car's gasoline tank --/
def total_fuel_cost (truck_capacity : ℝ) (car_capacity : ℝ) (truck_fullness : ℝ) (car_fullness : ℝ) (diesel_price : ℝ) (gasoline_price : ℝ) : ℝ :=
  let truck_to_fill := truck_capacity * (1 - truck_fullness)
  let car_to_fill := car_capacity * (1 - car_fullness)
  truck_to_fill * diesel_price + car_to_fill * gasoline_price

/-- Theorem stating that the total cost to fill up both tanks is $75.75 --/
theorem fuel_cost_calculation :
  total_fuel_cost 25 15 0.5 (1/3) 3.5 3.2 = 75.75 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_calculation_l4000_400081


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4000_400005

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4000_400005


namespace NUMINAMATH_CALUDE_power_of_five_cube_l4000_400083

theorem power_of_five_cube (n : ℤ) : 
  (∃ a : ℕ, n^3 - 3*n^2 + n + 2 = 5^a) ↔ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_power_of_five_cube_l4000_400083


namespace NUMINAMATH_CALUDE_parallel_vectors_l4000_400062

/-- Given two 2D vectors a and b, find the value of k that makes (k*a + b) parallel to (a - 3*b) -/
theorem parallel_vectors (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (-3, 2)) :
  ∃ k : ℝ, k * a.1 + b.1 = (a.1 - 3 * b.1) * ((k * a.2 + b.2) / (a.2 - 3 * b.2)) ∧ k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l4000_400062


namespace NUMINAMATH_CALUDE_student_calculation_l4000_400060

theorem student_calculation (chosen_number : ℕ) (h : chosen_number = 48) : 
  chosen_number * 5 - 138 = 102 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_l4000_400060


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l4000_400049

theorem ratio_sum_problem (a b c : ℕ) : 
  a + b + c = 1000 → 
  5 * b = a → 
  4 * b = c → 
  c = 400 := by sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l4000_400049


namespace NUMINAMATH_CALUDE_no_linear_term_implies_a_value_l4000_400084

/-- 
Given two polynomials (y + 2a) and (5 - y), if their product does not contain 
a linear term of y, then a = 5/2.
-/
theorem no_linear_term_implies_a_value (a : ℚ) : 
  (∀ y : ℚ, ∃ k m : ℚ, (y + 2*a) * (5 - y) = k*y^2 + m) → a = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_a_value_l4000_400084


namespace NUMINAMATH_CALUDE_special_ellipse_eccentricity_l4000_400065

/-- An ellipse with the property that the minimum distance from a point on the ellipse to a directrix is equal to the semi-latus rectum. -/
structure SpecialEllipse where
  /-- The eccentricity of the ellipse -/
  eccentricity : ℝ
  /-- The semi-latus rectum of the ellipse -/
  semiLatusRectum : ℝ
  /-- The minimum distance from a point on the ellipse to a directrix -/
  minDirectrixDistance : ℝ
  /-- The condition that the minimum distance to a directrix equals the semi-latus rectum -/
  distance_eq_semiLatusRectum : minDirectrixDistance = semiLatusRectum

/-- The eccentricity of a special ellipse is √2/2 -/
theorem special_ellipse_eccentricity (e : SpecialEllipse) : e.eccentricity = Real.sqrt 2 / 2 :=
  sorry

end NUMINAMATH_CALUDE_special_ellipse_eccentricity_l4000_400065


namespace NUMINAMATH_CALUDE_license_plate_count_is_9750000_l4000_400055

/-- The number of possible distinct license plates -/
def license_plate_count : ℕ :=
  (Nat.choose 6 2) * 26 * 25 * (10^4)

/-- Theorem stating the number of distinct license plates -/
theorem license_plate_count_is_9750000 :
  license_plate_count = 9750000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_is_9750000_l4000_400055


namespace NUMINAMATH_CALUDE_gcd_50420_35313_l4000_400029

theorem gcd_50420_35313 : Nat.gcd 50420 35313 = 19 := by
  sorry

end NUMINAMATH_CALUDE_gcd_50420_35313_l4000_400029


namespace NUMINAMATH_CALUDE_count_odd_numbers_300_to_600_l4000_400097

theorem count_odd_numbers_300_to_600 : 
  (Finset.filter (fun n => n % 2 = 1) (Finset.Icc 300 600)).card = 150 := by
  sorry

end NUMINAMATH_CALUDE_count_odd_numbers_300_to_600_l4000_400097


namespace NUMINAMATH_CALUDE_keaton_orange_harvest_frequency_l4000_400096

/-- Represents Keaton's farm earnings and harvest information -/
structure FarmData where
  yearly_earnings : ℕ
  apple_harvest_interval : ℕ
  apple_harvest_value : ℕ
  orange_harvest_value : ℕ

/-- Calculates the frequency of orange harvests in months -/
def orange_harvest_frequency (data : FarmData) : ℕ :=
  12 / (data.yearly_earnings - (12 / data.apple_harvest_interval * data.apple_harvest_value)) / data.orange_harvest_value

/-- Theorem stating that Keaton's orange harvest frequency is 2 months -/
theorem keaton_orange_harvest_frequency :
  orange_harvest_frequency ⟨420, 3, 30, 50⟩ = 2 := by
  sorry

end NUMINAMATH_CALUDE_keaton_orange_harvest_frequency_l4000_400096


namespace NUMINAMATH_CALUDE_gcd_lcm_problem_l4000_400079

theorem gcd_lcm_problem (A B : ℕ) (hA : A = 8 * 6) (hB : B = 36 / 3) :
  Nat.gcd A B = 12 ∧ Nat.lcm A B = 48 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_problem_l4000_400079


namespace NUMINAMATH_CALUDE_intersection_M_N_l4000_400088

open Set

-- Define set M
def M : Set ℝ := {x : ℝ | (x + 3) * (x - 2) < 0}

-- Define set N
def N : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem intersection_M_N :
  M ∩ N = Icc 1 2 ∩ Iio 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4000_400088


namespace NUMINAMATH_CALUDE_triangular_grid_properties_l4000_400082

/-- Represents a labeled vertex in the triangular grid -/
structure LabeledVertex where
  x : ℕ
  y : ℕ
  label : ℝ

/-- Represents the triangular grid -/
structure TriangularGrid where
  n : ℕ
  vertices : List LabeledVertex
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for adjacent triangles -/
def adjacent_condition (grid : TriangularGrid) : Prop :=
  ∀ A B C D : LabeledVertex,
    A ∈ grid.vertices → B ∈ grid.vertices → C ∈ grid.vertices → D ∈ grid.vertices →
    (A.x + 1 = B.x ∧ A.y = B.y) →
    (B.x = C.x ∧ B.y + 1 = C.y) →
    (C.x - 1 = D.x ∧ C.y = D.y) →
    A.label + D.label = B.label + C.label

/-- The main theorem -/
theorem triangular_grid_properties (grid : TriangularGrid)
    (h1 : grid.vertices.length = grid.n * (grid.n + 1) / 2)
    (h2 : adjacent_condition grid) :
    (∃ v1 v2 : LabeledVertex,
      v1 ∈ grid.vertices ∧ v2 ∈ grid.vertices ∧
      (∀ v : LabeledVertex, v ∈ grid.vertices → v1.label ≤ v.label ∧ v.label ≤ v2.label) ∧
      ((v1.x - v2.x)^2 + (v1.y - v2.y)^2 : ℝ) = grid.n^2) ∧
    (grid.vertices.map (λ v : LabeledVertex => v.label)).sum =
      (grid.n + 1) * (grid.n + 2) * (grid.a + grid.b + grid.c) / 6 :=
sorry

end NUMINAMATH_CALUDE_triangular_grid_properties_l4000_400082


namespace NUMINAMATH_CALUDE_exists_password_with_twenty_combinations_l4000_400086

/-- Represents a character in the password --/
structure PasswordChar :=
  (value : Char)

/-- Represents a 5-character password --/
structure Password :=
  (chars : Fin 5 → PasswordChar)

/-- Counts the number of unique permutations of a password --/
def countUniqueCombinations (password : Password) : ℕ :=
  sorry

/-- Theorem: There exists a 5-character password with exactly 20 different combinations --/
theorem exists_password_with_twenty_combinations : 
  ∃ (password : Password), countUniqueCombinations password = 20 := by
  sorry

end NUMINAMATH_CALUDE_exists_password_with_twenty_combinations_l4000_400086


namespace NUMINAMATH_CALUDE_book_selection_theorem_l4000_400090

theorem book_selection_theorem (math_books : Nat) (physics_books : Nat) : 
  math_books = 3 → physics_books = 2 → math_books * physics_books = 6 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l4000_400090


namespace NUMINAMATH_CALUDE_fraction_equality_l4000_400044

theorem fraction_equality (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c^2 / d = 16) :
  d / a = 1 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4000_400044


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l4000_400063

/-- A quadratic equation kx^2 - 4x + 1 = 0 has two distinct real roots if and only if k < 4 and k ≠ 0 -/
theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 4 * x + 1 = 0 ∧ k * y^2 - 4 * y + 1 = 0) ↔ 
  (k < 4 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l4000_400063


namespace NUMINAMATH_CALUDE_club_officer_selection_l4000_400013

/-- Represents a club with members and officers -/
structure Club where
  totalMembers : ℕ
  officerPositions : ℕ
  aliceAndBob : ℕ

/-- Calculates the number of ways to choose officers in a club -/
def chooseOfficers (club : Club) : ℕ :=
  let remainingMembers := club.totalMembers - club.aliceAndBob
  let case1 := remainingMembers * (remainingMembers - 1) * (remainingMembers - 2) * (remainingMembers - 3)
  let case2 := (club.officerPositions.choose 2) * remainingMembers * (remainingMembers - 1)
  case1 + case2

/-- Theorem stating the number of ways to choose officers in the specific club scenario -/
theorem club_officer_selection :
  let club : Club := { totalMembers := 30, officerPositions := 4, aliceAndBob := 2 }
  chooseOfficers club = 495936 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_l4000_400013


namespace NUMINAMATH_CALUDE_zoo_animal_ratio_l4000_400048

/-- Prove the ratio of monkeys to camels at the zoo -/
theorem zoo_animal_ratio : 
  ∀ (zebras camels monkeys giraffes : ℕ),
    zebras = 12 →
    camels = zebras / 2 →
    ∃ k : ℕ, monkeys = k * camels →
    giraffes = 2 →
    monkeys = giraffes + 22 →
    monkeys / camels = 4 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_ratio_l4000_400048


namespace NUMINAMATH_CALUDE_g_solutions_l4000_400093

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the property that g must satisfy
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 - 6 * x + 1

-- State the theorem
theorem g_solutions :
  ∀ g : ℝ → ℝ, g_property g →
    (∀ x, g x = 3 * x - 1) ∨ (∀ x, g x = -3 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_g_solutions_l4000_400093


namespace NUMINAMATH_CALUDE_division_of_decimals_l4000_400074

theorem division_of_decimals : (2.4 : ℝ) / 0.06 = 40 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l4000_400074


namespace NUMINAMATH_CALUDE_encryption_correspondence_unique_decryption_l4000_400028

/-- Encryption function that maps a plaintext to a ciphertext -/
def encrypt (p : ℕ × ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let (a, b, c, d) := p
  (a + 2*b, b + c, 2*c + 3*d, 4*d)

/-- Theorem stating that the plaintext (6, 4, 1, 7) corresponds to the ciphertext (14, 9, 23, 28) -/
theorem encryption_correspondence :
  encrypt (6, 4, 1, 7) = (14, 9, 23, 28) := by
  sorry

/-- Theorem stating that the plaintext (6, 4, 1, 7) is the unique solution -/
theorem unique_decryption :
  ∀ p : ℕ × ℕ × ℕ × ℕ, encrypt p = (14, 9, 23, 28) → p = (6, 4, 1, 7) := by
  sorry

end NUMINAMATH_CALUDE_encryption_correspondence_unique_decryption_l4000_400028


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_is_7_or_8_l4000_400047

def isosceles_triangle_perimeter (x y : ℝ) : Prop :=
  (x > 0 ∧ y > 0) ∧  -- positive side lengths
  (x = y ∨ x + y > x)  -- triangle inequality
  ∧ y = Real.sqrt (2 - x) + Real.sqrt (3 * x - 6) + 3

theorem isosceles_triangle_perimeter_is_7_or_8 :
  ∀ x y : ℝ, isosceles_triangle_perimeter x y →
  (x + y + (if x = y then x else y) = 7 ∨ x + y + (if x = y then x else y) = 8) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_is_7_or_8_l4000_400047


namespace NUMINAMATH_CALUDE_figure_side_length_l4000_400014

theorem figure_side_length (total_area : ℝ) (y : ℝ) : 
  total_area = 1300 →
  (3 * y)^2 + (6 * y)^2 + (1/2 * 3 * y * 6 * y) = total_area →
  y = Real.sqrt 1300 / Real.sqrt 54 :=
by sorry

end NUMINAMATH_CALUDE_figure_side_length_l4000_400014


namespace NUMINAMATH_CALUDE_complex_inverse_calculation_l4000_400032

theorem complex_inverse_calculation (i : ℂ) (h : i^2 = -1) : 
  (2*i - 3*i⁻¹)⁻¹ = -i/5 := by sorry

end NUMINAMATH_CALUDE_complex_inverse_calculation_l4000_400032


namespace NUMINAMATH_CALUDE_compute_expression_l4000_400038

theorem compute_expression : 8 * (243 / 3 + 81 / 9 + 25 / 25 + 3) = 752 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l4000_400038


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l4000_400085

theorem arithmetic_sequence_length : 
  ∀ (a₁ : ℝ) (d : ℝ) (aₙ : ℝ),
  a₁ = 2.5 → d = 5 → aₙ = 72.5 →
  ∃ (n : ℕ), n = 15 ∧ aₙ = a₁ + (n - 1) * d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l4000_400085


namespace NUMINAMATH_CALUDE_max_value_a_l4000_400080

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b) 
  (h2 : b < 4 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 80) : 
  a ≤ 4724 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 4724 ∧ 
    b' = 1575 ∧ 
    c' = 394 ∧ 
    d' = 79 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 80 :=
sorry

end NUMINAMATH_CALUDE_max_value_a_l4000_400080


namespace NUMINAMATH_CALUDE_sum_of_max_min_is_4032_l4000_400019

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂ - 2016) ∧
  (∀ x : ℝ, x > 0 → f x > 2016)

/-- The theorem to be proved -/
theorem sum_of_max_min_is_4032 (f : ℝ → ℝ) (h : special_function f) :
  let M := ⨆ (x : ℝ) (hx : x ∈ Set.Icc (-2016) 2016), f x
  let N := ⨅ (x : ℝ) (hx : x ∈ Set.Icc (-2016) 2016), f x
  M + N = 4032 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_is_4032_l4000_400019


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_application_l4000_400070

theorem chinese_remainder_theorem_application (n : ℤ) : 
  n % 158 = 50 → n % 176 = 66 → n % 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_application_l4000_400070


namespace NUMINAMATH_CALUDE_breakfast_dessert_l4000_400018

-- Define the possible breakfast items
inductive BreakfastItem
  | Whiskey
  | Duck
  | Oranges
  | Pie
  | BelleHelenePear
  | StrawberrySherbet
  | Coffee

-- Define the structure of a journalist's statement
structure JournalistStatement where
  items : List BreakfastItem

-- Define the honesty levels of journalists
inductive JournalistHonesty
  | AlwaysTruthful
  | OneFalseStatement
  | AlwaysLies

-- Define the breakfast observation
structure BreakfastObservation where
  jules : JournalistStatement
  jacques : JournalistStatement
  jim : JournalistStatement
  julesHonesty : JournalistHonesty
  jacquesHonesty : JournalistHonesty
  jimHonesty : JournalistHonesty

def breakfast : BreakfastObservation := {
  jules := { items := [BreakfastItem.Whiskey, BreakfastItem.Duck, BreakfastItem.Oranges, BreakfastItem.Coffee] },
  jacques := { items := [BreakfastItem.Pie, BreakfastItem.BelleHelenePear] },
  jim := { items := [BreakfastItem.Whiskey, BreakfastItem.Pie, BreakfastItem.StrawberrySherbet, BreakfastItem.Coffee] },
  julesHonesty := JournalistHonesty.AlwaysTruthful,
  jacquesHonesty := JournalistHonesty.AlwaysLies,
  jimHonesty := JournalistHonesty.OneFalseStatement
}

theorem breakfast_dessert :
  ∃ (dessert : BreakfastItem), dessert = BreakfastItem.StrawberrySherbet :=
by sorry

end NUMINAMATH_CALUDE_breakfast_dessert_l4000_400018


namespace NUMINAMATH_CALUDE_rectangle_perimeter_is_164_l4000_400026

/-- Represents the side lengths of the squares in the rectangle dissection -/
structure SquareSides where
  a₁ : ℕ
  a₂ : ℕ
  a₃ : ℕ
  a₄ : ℕ
  a₅ : ℕ
  a₆ : ℕ
  a₇ : ℕ
  a₈ : ℕ
  a₉ : ℕ

/-- The conditions for the rectangle dissection -/
def RectangleDissectionConditions (s : SquareSides) : Prop :=
  s.a₁ + s.a₂ = s.a₄ ∧
  s.a₁ + s.a₄ = s.a₅ ∧
  s.a₄ + s.a₅ = s.a₇ ∧
  s.a₅ + s.a₇ = s.a₉ ∧
  s.a₂ + s.a₄ + s.a₇ = s.a₈ ∧
  s.a₂ + s.a₈ = s.a₆ ∧
  s.a₁ + s.a₅ + s.a₉ = s.a₃ ∧
  s.a₃ + s.a₆ = s.a₈ + s.a₇

/-- The width of the rectangle -/
def RectangleWidth (s : SquareSides) : ℕ := s.a₄ + s.a₇ + s.a₉

/-- The length of the rectangle -/
def RectangleLength (s : SquareSides) : ℕ := s.a₂ + s.a₈ + s.a₆

/-- The main theorem: Given the conditions, the perimeter of the rectangle is 164 -/
theorem rectangle_perimeter_is_164 (s : SquareSides) 
  (h : RectangleDissectionConditions s) 
  (h_coprime : Nat.Coprime (RectangleWidth s) (RectangleLength s)) :
  2 * (RectangleWidth s + RectangleLength s) = 164 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_is_164_l4000_400026


namespace NUMINAMATH_CALUDE_crayons_given_to_friends_l4000_400031

theorem crayons_given_to_friends (initial : ℕ) (lost : ℕ) (left : ℕ) 
  (h1 : initial = 1453)
  (h2 : lost = 558)
  (h3 : left = 332) :
  initial - left - lost = 563 := by
  sorry

end NUMINAMATH_CALUDE_crayons_given_to_friends_l4000_400031


namespace NUMINAMATH_CALUDE_smallest_digits_to_append_l4000_400002

def is_divisible_by_all_less_than_10 (n : ℕ) : Prop :=
  ∀ k : ℕ, k < 10 → k > 0 → n % k = 0

def append_digits (base n digits : ℕ) : ℕ :=
  base * (10 ^ digits) + n

theorem smallest_digits_to_append : 
  (∀ k < 4, ¬∃ n : ℕ, n < 10^k ∧ is_divisible_by_all_less_than_10 (append_digits 2014 n k)) ∧
  (∃ n : ℕ, n < 10^4 ∧ is_divisible_by_all_less_than_10 (append_digits 2014 n 4)) :=
sorry

end NUMINAMATH_CALUDE_smallest_digits_to_append_l4000_400002


namespace NUMINAMATH_CALUDE_expression_value_l4000_400022

theorem expression_value : (64 + 27)^2 - (27^2 + 64^2) + 3 * Real.rpow 1728 (1/3) = 3492 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4000_400022


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l4000_400024

def is_arithmetic_sequence (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, b - a = r ∧ c - b = r ∧ d - c = r ∧ e - d = r

theorem arithmetic_sequence_difference 
  (a b c : ℝ) (h : is_arithmetic_sequence 2 a b c 9) : 
  c - a = (7 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l4000_400024


namespace NUMINAMATH_CALUDE_angle_A_in_special_triangle_l4000_400092

theorem angle_A_in_special_triangle (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C →  -- Ensuring positive angles
  B = A + 10 →             -- Given condition
  C = B + 10 →             -- Given condition
  A + B + C = 180 →        -- Sum of angles in a triangle
  A = 50 := by sorry

end NUMINAMATH_CALUDE_angle_A_in_special_triangle_l4000_400092


namespace NUMINAMATH_CALUDE_triangle_property_l4000_400064

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : 3 * t.b * Real.cos t.A = t.c * Real.cos t.A + t.a * Real.cos t.C)
  (h2 : t.a = 4 * Real.sqrt 2) : 
  Real.tan t.A = 2 * Real.sqrt 2 ∧ 
  (∃ (S : ℝ), S ≤ 8 * Real.sqrt 2 ∧ 
    ∀ (S' : ℝ), S' = t.a * t.b * Real.sin t.C / 2 → S' ≤ S) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l4000_400064


namespace NUMINAMATH_CALUDE_systematic_sampling_l4000_400017

/-- Systematic sampling problem -/
theorem systematic_sampling 
  (total_students : Nat) 
  (num_parts : Nat) 
  (first_part_end : Nat) 
  (first_drawn : Nat) 
  (nth_draw : Nat) :
  total_students = 1000 →
  num_parts = 50 →
  first_part_end = 20 →
  first_drawn = 15 →
  nth_draw = 40 →
  (nth_draw - 1) * (total_students / num_parts) + first_drawn = 795 :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_l4000_400017


namespace NUMINAMATH_CALUDE_l_shaped_paper_area_l4000_400011

/-- The area of an "L" shaped paper formed by cutting rectangles from a larger rectangle --/
theorem l_shaped_paper_area (original_length original_width cut1_length cut1_width cut2_length cut2_width : ℕ) 
  (h1 : original_length = 10)
  (h2 : original_width = 7)
  (h3 : cut1_length = 3)
  (h4 : cut1_width = 2)
  (h5 : cut2_length = 2)
  (h6 : cut2_width = 4) :
  original_length * original_width - cut1_length * cut1_width - cut2_length * cut2_width = 56 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_paper_area_l4000_400011


namespace NUMINAMATH_CALUDE_initial_population_proof_l4000_400054

def population_change (initial : ℕ) : ℕ := 
  let after_first_year := initial * 125 / 100
  (after_first_year * 70) / 100

theorem initial_population_proof : 
  ∃ (P : ℕ), population_change P = 363650 ∧ P = 415600 := by
  sorry

end NUMINAMATH_CALUDE_initial_population_proof_l4000_400054


namespace NUMINAMATH_CALUDE_carrie_bought_four_shirts_l4000_400067

/-- The number of shirts Carrie bought -/
def num_shirts : ℕ := sorry

/-- The cost of each shirt -/
def shirt_cost : ℕ := 8

/-- The number of pairs of pants Carrie bought -/
def num_pants : ℕ := 2

/-- The cost of each pair of pants -/
def pants_cost : ℕ := 18

/-- The number of jackets Carrie bought -/
def num_jackets : ℕ := 2

/-- The cost of each jacket -/
def jacket_cost : ℕ := 60

/-- The amount Carrie paid for her half of the clothes -/
def carrie_payment : ℕ := 94

/-- Theorem stating that Carrie bought 4 shirts -/
theorem carrie_bought_four_shirts : num_shirts = 4 := by
  sorry

end NUMINAMATH_CALUDE_carrie_bought_four_shirts_l4000_400067


namespace NUMINAMATH_CALUDE_simple_interest_problem_l4000_400066

/-- Given a principal amount P, prove that if the simple interest on P at 4% for 5 years
    is equal to P - 2240, then P = 2800. -/
theorem simple_interest_problem (P : ℝ) : 
  (P * 4 * 5) / 100 = P - 2240 → P = 2800 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l4000_400066


namespace NUMINAMATH_CALUDE_no_primes_in_sequence_l4000_400012

/-- A sequence of natural numbers -/
def Sequence := ℕ → ℕ

/-- Property: The sequence is increasing -/
def IsIncreasing (s : Sequence) : Prop :=
  ∀ n : ℕ, s n < s (n + 1)

/-- Property: Any three consecutive numbers form an arithmetic or geometric progression -/
def IsArithmeticOrGeometric (s : Sequence) : Prop :=
  ∀ n : ℕ, (2 * s (n + 1) = s n + s (n + 2)) ∨ (s (n + 1) ^ 2 = s n * s (n + 2))

/-- Property: The first two numbers are divisible by 4 -/
def FirstTwoDivisibleByFour (s : Sequence) : Prop :=
  4 ∣ s 0 ∧ 4 ∣ s 1

/-- Main theorem -/
theorem no_primes_in_sequence (s : Sequence)
  (h_inc : IsIncreasing s)
  (h_prog : IsArithmeticOrGeometric s)
  (h_div4 : FirstTwoDivisibleByFour s) :
  ∀ n : ℕ, ¬ Nat.Prime (s n) :=
sorry

end NUMINAMATH_CALUDE_no_primes_in_sequence_l4000_400012


namespace NUMINAMATH_CALUDE_ellipse_equation_for_given_properties_l4000_400058

/-- Represents an ellipse with specific properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  minor_axis_length : ℝ
  eccentricity : ℝ

/-- The equation of an ellipse given its properties -/
def ellipse_equation (e : Ellipse) : (ℝ → ℝ → Prop) :=
  fun x y => x^2 / 36 + y^2 / 32 = 1

/-- Theorem stating the equation of the ellipse with given properties -/
theorem ellipse_equation_for_given_properties (e : Ellipse) 
  (h1 : e.center = (0, 0))
  (h2 : e.foci_on_x_axis = true)
  (h3 : e.minor_axis_length = 8 * Real.sqrt 2)
  (h4 : e.eccentricity = 1/3) :
  ellipse_equation e = fun x y => x^2 / 36 + y^2 / 32 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_for_given_properties_l4000_400058


namespace NUMINAMATH_CALUDE_probability_at_least_one_type_b_l4000_400023

def total_questions : ℕ := 5
def type_a_questions : ℕ := 2
def type_b_questions : ℕ := 3
def selected_questions : ℕ := 2

theorem probability_at_least_one_type_b :
  let total_combinations := Nat.choose total_questions selected_questions
  let all_type_a_combinations := Nat.choose type_a_questions selected_questions
  (total_combinations - all_type_a_combinations) / total_combinations = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_type_b_l4000_400023


namespace NUMINAMATH_CALUDE_nested_sqrt_twelve_l4000_400095

theorem nested_sqrt_twelve (x : ℝ) : x > 0 ∧ x = Real.sqrt (12 + x) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_twelve_l4000_400095


namespace NUMINAMATH_CALUDE_no_real_roots_l4000_400052

theorem no_real_roots : ∀ x : ℝ, x^2 - x * Real.sqrt 5 + Real.sqrt 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l4000_400052


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l4000_400050

theorem consecutive_odd_integers_sum (a : ℤ) : 
  (∃ n : ℤ, a = n ∧ 
    (∀ i : Fin 5, Odd (a + 2 * i.val)) ∧
    (a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = -365)) → 
  (a + 8 = -69) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l4000_400050


namespace NUMINAMATH_CALUDE_hezekiah_age_l4000_400003

/-- Given that Ryanne is 7 years older than Hezekiah and their combined age is 15, 
    prove that Hezekiah is 4 years old. -/
theorem hezekiah_age (hezekiah_age ryanne_age : ℕ) 
  (h1 : ryanne_age = hezekiah_age + 7)
  (h2 : hezekiah_age + ryanne_age = 15) : 
  hezekiah_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_hezekiah_age_l4000_400003


namespace NUMINAMATH_CALUDE_alok_mixed_veg_plates_l4000_400027

/-- Represents the order and pricing information for a restaurant bill --/
structure RestaurantBill where
  chapatis : ℕ
  rice : ℕ
  iceCream : ℕ
  chapatiPrice : ℕ
  ricePrice : ℕ
  mixedVegPrice : ℕ
  iceCreamPrice : ℕ
  totalPaid : ℕ

/-- Calculates the number of mixed vegetable plates ordered --/
def mixedVegPlates (bill : RestaurantBill) : ℕ :=
  (bill.totalPaid - (bill.chapatis * bill.chapatiPrice + bill.rice * bill.ricePrice + bill.iceCream * bill.iceCreamPrice)) / bill.mixedVegPrice

/-- Theorem stating that Alok ordered 7 plates of mixed vegetable --/
theorem alok_mixed_veg_plates :
  let bill : RestaurantBill := {
    chapatis := 16,
    rice := 5,
    iceCream := 6,
    chapatiPrice := 6,
    ricePrice := 45,
    mixedVegPrice := 70,
    iceCreamPrice := 40,
    totalPaid := 1051
  }
  mixedVegPlates bill = 7 := by
  sorry

end NUMINAMATH_CALUDE_alok_mixed_veg_plates_l4000_400027


namespace NUMINAMATH_CALUDE_emma_square_calculation_l4000_400040

theorem emma_square_calculation : 37^2 = 38^2 - 75 := by
  sorry

end NUMINAMATH_CALUDE_emma_square_calculation_l4000_400040


namespace NUMINAMATH_CALUDE_min_students_required_l4000_400037

/-- Represents a set of days in which a student participates -/
def ParticipationSet := Finset (Fin 6)

/-- The property that for any 3 days, there's a student participating in all 3 -/
def CoversAllTriples (sets : Finset ParticipationSet) : Prop :=
  ∀ (days : Finset (Fin 6)), days.card = 3 → ∃ s ∈ sets, days ⊆ s

/-- The property that no student participates in all 4 days of any 4-day selection -/
def NoQuadruplesCovered (sets : Finset ParticipationSet) : Prop :=
  ∀ (days : Finset (Fin 6)), days.card = 4 → ∀ s ∈ sets, ¬(days ⊆ s)

/-- The main theorem stating the minimum number of students required -/
theorem min_students_required :
  ∃ (sets : Finset ParticipationSet),
    sets.card = 20 ∧
    (∀ s ∈ sets, s.card = 3) ∧
    CoversAllTriples sets ∧
    NoQuadruplesCovered sets ∧
    (∀ (sets' : Finset ParticipationSet),
      (∀ s' ∈ sets', s'.card = 3) →
      CoversAllTriples sets' →
      NoQuadruplesCovered sets' →
      sets'.card ≥ 20) :=
sorry

end NUMINAMATH_CALUDE_min_students_required_l4000_400037


namespace NUMINAMATH_CALUDE_compute_expression_l4000_400051

theorem compute_expression : (85 * 1515 - 25 * 1515) + (48 * 1515) = 163620 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l4000_400051


namespace NUMINAMATH_CALUDE_log_difference_negative_l4000_400071

theorem log_difference_negative (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  Real.log (b - a) < 0 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_negative_l4000_400071


namespace NUMINAMATH_CALUDE_alpha_para_beta_sufficient_not_necessary_l4000_400007

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Plane → Plane → Prop)
variable (perpLine : Line → Line → Prop)
variable (paraLine : Line → Plane → Prop)

-- State the theorem
theorem alpha_para_beta_sufficient_not_necessary 
  (l m : Line) (α β : Plane) 
  (h1 : perp l α) 
  (h2 : paraLine m β) : 
  (∃ (config : Type), 
    (∀ (α β : Plane), para α β → perpLine l m) ∧ 
    (∃ (α β : Plane), perpLine l m ∧ ¬ para α β)) :=
sorry

end NUMINAMATH_CALUDE_alpha_para_beta_sufficient_not_necessary_l4000_400007


namespace NUMINAMATH_CALUDE_trig_inequality_range_l4000_400034

theorem trig_inequality_range (x : Real) : 
  (x ∈ Set.Icc 0 Real.pi) → 
  (Real.cos x)^2 > (Real.sin x)^2 → 
  x ∈ Set.Ioo 0 (Real.pi / 4) ∪ Set.Ioo (3 * Real.pi / 4) Real.pi :=
by sorry

end NUMINAMATH_CALUDE_trig_inequality_range_l4000_400034


namespace NUMINAMATH_CALUDE_calculation_proofs_l4000_400036

theorem calculation_proofs :
  (1 - 2^3 / 8 - 1/4 * (-2)^2 = -2) ∧
  ((-1/12 - 1/16 + 3/4 - 1/6) * (-48) = -21) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l4000_400036


namespace NUMINAMATH_CALUDE_cab_driver_average_income_l4000_400056

def daily_incomes : List ℝ := [250, 400, 750, 400, 500]

theorem cab_driver_average_income :
  (daily_incomes.sum / daily_incomes.length : ℝ) = 460 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_average_income_l4000_400056


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4000_400015

theorem sufficient_not_necessary_condition :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) ∧
  (∃ a b : ℝ, a * b > 1 ∧ (a ≤ 1 ∨ b ≤ 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4000_400015


namespace NUMINAMATH_CALUDE_eggs_left_after_recovering_capital_l4000_400073

theorem eggs_left_after_recovering_capital 
  (total_eggs : ℕ) 
  (crate_cost_cents : ℕ) 
  (selling_price_cents : ℕ) : ℕ :=
  let eggs_sold := crate_cost_cents / selling_price_cents
  total_eggs - eggs_sold

#check eggs_left_after_recovering_capital 30 500 20 = 5

end NUMINAMATH_CALUDE_eggs_left_after_recovering_capital_l4000_400073


namespace NUMINAMATH_CALUDE_log_property_l4000_400069

theorem log_property (a : ℝ) (f : ℝ → ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x > 0, f x = Real.log x / Real.log a) (h4 : f 9 = 2) : 
  f (a ^ a) = 3 := by
sorry

end NUMINAMATH_CALUDE_log_property_l4000_400069


namespace NUMINAMATH_CALUDE_no_20_digit_square_starting_with_11_ones_l4000_400061

theorem no_20_digit_square_starting_with_11_ones :
  ¬∃ (n : ℕ), 
    (10^19 ≤ n) ∧ 
    (n < 10^20) ∧ 
    (11111111111 * 10^9 ≤ n) ∧ 
    (n < 11111111112 * 10^9) ∧ 
    (∃ (k : ℕ), n = k^2) :=
by sorry

end NUMINAMATH_CALUDE_no_20_digit_square_starting_with_11_ones_l4000_400061


namespace NUMINAMATH_CALUDE_custom_op_neg_four_six_l4000_400004

-- Define the custom operation ﹡
def custom_op (a b : ℝ) : ℝ := 5 * a + 2 * b - 1

-- Theorem statement
theorem custom_op_neg_four_six :
  custom_op (-4) 6 = -9 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_neg_four_six_l4000_400004


namespace NUMINAMATH_CALUDE_firewood_collection_l4000_400087

/-- Firewood collection problem -/
theorem firewood_collection (K H : ℝ) (x E : ℝ) 
  (hK : K = 0.8)
  (hH : H = 1.5)
  (eq1 : 10 * K + x * E + 12 * H = 44)
  (eq2 : 10 + x + 12 = 35) :
  x = 13 ∧ E = 18 / 13 := by
  sorry

end NUMINAMATH_CALUDE_firewood_collection_l4000_400087


namespace NUMINAMATH_CALUDE_circle_diameter_points_exist_l4000_400089

/-- Represents a point on the circumference of a circle -/
structure CirclePoint where
  angle : ℝ
  property : 0 ≤ angle ∧ angle < 2 * Real.pi

/-- Represents an arc on the circumference of a circle -/
structure Arc where
  start : CirclePoint
  length : ℝ
  property : 0 < length ∧ length ≤ 2 * Real.pi

/-- The main theorem statement -/
theorem circle_diameter_points_exist (k : ℕ) (points : Finset CirclePoint) (arcs : Finset Arc) :
  points.card = 3 * k →
  arcs.card = 3 * k →
  (∃ (s₁ : Finset Arc), s₁.card = k ∧ ∀ a ∈ s₁, a.length = 1) →
  (∃ (s₂ : Finset Arc), s₂.card = k ∧ ∀ a ∈ s₂, a.length = 2) →
  (∃ (s₃ : Finset Arc), s₃.card = k ∧ ∀ a ∈ s₃, a.length = 3) →
  ∃ (p₁ p₂ : CirclePoint), p₁ ∈ points ∧ p₂ ∈ points ∧ abs (p₁.angle - p₂.angle) = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_points_exist_l4000_400089


namespace NUMINAMATH_CALUDE_binary_multiplication_division_equality_l4000_400046

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, x) => acc + if x then 2^i else 0) 0

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

theorem binary_multiplication_division_equality :
  let a := [false, true, true, true, false, true] -- 101110₂
  let b := [false, false, true, false, true, false, true] -- 1010100₂
  let c := [false, false, true] -- 100₂
  let result_binary := [true, true, false, false, true, true, false, true, true, false, true, true] -- 101110110011₂
  let result_decimal : ℕ := 2995
  (binary_to_decimal a * binary_to_decimal b) / binary_to_decimal c = binary_to_decimal result_binary ∧
  binary_to_decimal result_binary = result_decimal ∧
  decimal_to_binary result_decimal = result_binary :=
by sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_equality_l4000_400046


namespace NUMINAMATH_CALUDE_co_presidents_selection_l4000_400072

theorem co_presidents_selection (n : ℕ) (k : ℕ) (h1 : n = 18) (h2 : k = 3) :
  Nat.choose n k = 816 := by
  sorry

end NUMINAMATH_CALUDE_co_presidents_selection_l4000_400072


namespace NUMINAMATH_CALUDE_plane_equation_theorem_l4000_400053

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneCoefficients where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- The foot of the perpendicular from the origin to the plane -/
def footOfPerpendicular : Point3D :=
  { x := 10, y := -5, z := 4 }

/-- Check if the given coefficients satisfy the required conditions -/
def validCoefficients (coeff : PlaneCoefficients) : Prop :=
  coeff.A > 0 ∧ Nat.gcd (Int.natAbs coeff.A) (Int.natAbs coeff.B) = 1 ∧
  Nat.gcd (Int.natAbs coeff.A) (Int.natAbs coeff.C) = 1 ∧
  Nat.gcd (Int.natAbs coeff.A) (Int.natAbs coeff.D) = 1

/-- Check if a point satisfies the plane equation -/
def satisfiesPlaneEquation (p : Point3D) (coeff : PlaneCoefficients) : Prop :=
  coeff.A * p.x + coeff.B * p.y + coeff.C * p.z + coeff.D = 0

/-- The main theorem to prove -/
theorem plane_equation_theorem :
  ∃ (coeff : PlaneCoefficients),
    validCoefficients coeff ∧
    satisfiesPlaneEquation footOfPerpendicular coeff ∧
    coeff.A = 10 ∧ coeff.B = -5 ∧ coeff.C = 4 ∧ coeff.D = -141 :=
sorry

end NUMINAMATH_CALUDE_plane_equation_theorem_l4000_400053


namespace NUMINAMATH_CALUDE_puppies_adoption_l4000_400006

theorem puppies_adoption (first_week : ℕ) : 
  first_week + (2/5 : ℚ) * first_week + 2 * ((2/5 : ℚ) * first_week) + (first_week + 10) = 74 → 
  first_week = 20 := by
sorry

end NUMINAMATH_CALUDE_puppies_adoption_l4000_400006


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l4000_400009

theorem real_roots_of_polynomial (x : ℝ) :
  x^4 + 2*x^3 - x - 2 = 0 ↔ x = -2 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l4000_400009


namespace NUMINAMATH_CALUDE_gcd_problems_l4000_400008

theorem gcd_problems :
  (Nat.gcd 72 168 = 24) ∧ (Nat.gcd 98 280 = 14) := by
  sorry

end NUMINAMATH_CALUDE_gcd_problems_l4000_400008


namespace NUMINAMATH_CALUDE_line_circle_intersection_l4000_400041

theorem line_circle_intersection (k : ℝ) : ∃ (x y : ℝ),
  y = k * x - k ∧ (x - 2)^2 + y^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l4000_400041


namespace NUMINAMATH_CALUDE_parabola_vertex_l4000_400057

/-- The vertex of the parabola y = 2x^2 + 16x + 34 is (-4, 2) -/
theorem parabola_vertex :
  let f (x : ℝ) := 2 * x^2 + 16 * x + 34
  ∃! (h k : ℝ), ∀ x, f x = 2 * (x - h)^2 + k ∧ h = -4 ∧ k = 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l4000_400057


namespace NUMINAMATH_CALUDE_fifth_power_inequality_l4000_400068

theorem fifth_power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^5 + b^5 + c^5 ≥ a^3*b*c + b^3*a*c + c^3*a*b := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_inequality_l4000_400068


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l4000_400099

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (x - a) * (x + 1 - a) ≥ 0 → x ≠ 1) → 
  a > 1 ∧ a < 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l4000_400099


namespace NUMINAMATH_CALUDE_coach_b_baseballs_l4000_400076

/-- The number of baseballs Coach B bought -/
def num_baseballs : ℕ := 14

/-- The cost of each basketball -/
def basketball_cost : ℚ := 29

/-- The cost of each baseball -/
def baseball_cost : ℚ := 5/2

/-- The cost of the baseball bat -/
def bat_cost : ℚ := 18

/-- The number of basketballs Coach A bought -/
def num_basketballs : ℕ := 10

/-- The difference in spending between Coach A and Coach B -/
def spending_difference : ℚ := 237

theorem coach_b_baseballs :
  (num_basketballs * basketball_cost) = 
  spending_difference + (num_baseballs * baseball_cost + bat_cost) :=
by sorry

end NUMINAMATH_CALUDE_coach_b_baseballs_l4000_400076


namespace NUMINAMATH_CALUDE_tangent_line_sum_l4000_400001

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_sum (h : ∀ x, x = 1 → f x = (1/2) * x + 2) :
  f 1 + (deriv f) 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l4000_400001


namespace NUMINAMATH_CALUDE_smallest_non_five_divisible_unit_l4000_400043

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def divisible_by_five_units (n : ℕ) : Prop := n % 10 = 0 ∨ n % 10 = 5

theorem smallest_non_five_divisible_unit : 
  (∀ d, is_digit d → (∀ n, divisible_by_five_units n → n % 10 ≠ d) → d ≥ 1) ∧
  (∃ n, divisible_by_five_units n ∧ n % 10 ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_non_five_divisible_unit_l4000_400043


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l4000_400016

theorem unique_two_digit_integer (s : ℕ) : 
  (s ≥ 10 ∧ s < 100) ∧ (13 * s) % 100 = 52 ↔ s = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l4000_400016


namespace NUMINAMATH_CALUDE_same_figure_l4000_400039

noncomputable section

open Complex

/-- Two equations describe the same figure in the complex plane -/
theorem same_figure (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  {z : ℂ | abs (z + n * I) + abs (z - m * I) = n} =
  {z : ℂ | abs (z + n * I) - abs (z - m * I) = -m} :=
sorry

end

end NUMINAMATH_CALUDE_same_figure_l4000_400039


namespace NUMINAMATH_CALUDE_bin_game_expected_value_l4000_400025

theorem bin_game_expected_value (k : ℕ) (h1 : k > 0) : 
  (8 / (8 + k : ℝ)) * 3 + (k / (8 + k : ℝ)) * (-1) = 1 → k = 8 :=
by sorry

end NUMINAMATH_CALUDE_bin_game_expected_value_l4000_400025


namespace NUMINAMATH_CALUDE_log_inequality_l4000_400075

theorem log_inequality : ∃ (a b : ℝ), 
  (a = Real.log 0.8 / Real.log 0.7) ∧ 
  (b = Real.log 0.4 / Real.log 0.5) ∧ 
  (b > a) ∧ (a > 0) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l4000_400075


namespace NUMINAMATH_CALUDE_decreasing_function_l4000_400035

-- Define the four functions
def f1 (x : ℝ) : ℝ := x^2 + 1
def f2 (x : ℝ) : ℝ := -x^2 + 1
def f3 (x : ℝ) : ℝ := 2*x + 1
def f4 (x : ℝ) : ℝ := -2*x + 1

-- Theorem statement
theorem decreasing_function : 
  (∀ x : ℝ, HasDerivAt f4 (-2) x) ∧ 
  (∀ x : ℝ, (HasDerivAt f1 (2*x) x) ∨ (HasDerivAt f2 (-2*x) x) ∨ (HasDerivAt f3 2 x)) :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_l4000_400035


namespace NUMINAMATH_CALUDE_martha_savings_l4000_400098

def daily_allowance : ℚ := 12
def normal_saving_rate : ℚ := 1/2
def exception_saving_rate : ℚ := 1/4
def days_in_week : ℕ := 7
def normal_saving_days : ℕ := 6
def exception_saving_days : ℕ := 1

theorem martha_savings : 
  (normal_saving_days : ℚ) * (daily_allowance * normal_saving_rate) + 
  (exception_saving_days : ℚ) * (daily_allowance * exception_saving_rate) = 39 := by
  sorry

end NUMINAMATH_CALUDE_martha_savings_l4000_400098


namespace NUMINAMATH_CALUDE_no_rain_probability_l4000_400021

theorem no_rain_probability (p_rain_5th p_rain_6th : ℝ) 
  (h1 : p_rain_5th = 0.2) 
  (h2 : p_rain_6th = 0.4) 
  (h3 : 0 ≤ p_rain_5th ∧ p_rain_5th ≤ 1) 
  (h4 : 0 ≤ p_rain_6th ∧ p_rain_6th ≤ 1) :
  (1 - p_rain_5th) * (1 - p_rain_6th) = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l4000_400021


namespace NUMINAMATH_CALUDE_vitamin_a_intake_in_grams_l4000_400091

/-- Conversion factor from grams to milligrams -/
def gram_to_mg : ℝ := 1000

/-- Conversion factor from milligrams to micrograms -/
def mg_to_μg : ℝ := 1000

/-- Daily intake of vitamin A for adult women in micrograms -/
def vitamin_a_intake : ℝ := 750

/-- Theorem stating that 750 micrograms is equal to 7.5 × 10^-4 grams -/
theorem vitamin_a_intake_in_grams :
  (vitamin_a_intake / (gram_to_mg * mg_to_μg)) = 7.5e-4 := by
  sorry

end NUMINAMATH_CALUDE_vitamin_a_intake_in_grams_l4000_400091


namespace NUMINAMATH_CALUDE_opposite_black_is_orange_l4000_400020

-- Define the colors
inductive Color
| Orange | Yellow | Blue | Pink | Violet | Black

-- Define a cube face
structure Face :=
  (color : Color)

-- Define a cube
structure Cube :=
  (top : Face)
  (front : Face)
  (right : Face)
  (bottom : Face)
  (back : Face)
  (left : Face)

-- Define the views
def view1 (c : Cube) : Prop :=
  c.top.color = Color.Orange ∧ c.front.color = Color.Blue ∧ c.right.color = Color.Pink

def view2 (c : Cube) : Prop :=
  c.top.color = Color.Orange ∧ c.front.color = Color.Violet ∧ c.right.color = Color.Pink

def view3 (c : Cube) : Prop :=
  c.top.color = Color.Orange ∧ c.front.color = Color.Yellow ∧ c.right.color = Color.Pink

-- Theorem statement
theorem opposite_black_is_orange (c : Cube) :
  view1 c → view2 c → view3 c → c.bottom.color = Color.Black →
  c.top.color = Color.Orange :=
sorry

end NUMINAMATH_CALUDE_opposite_black_is_orange_l4000_400020


namespace NUMINAMATH_CALUDE_pell_equation_solutions_l4000_400042

theorem pell_equation_solutions :
  let solutions : List (ℤ × ℤ) := [(2, 1), (7, 4), (26, 15), (97, 56)]
  ∀ (x y : ℤ), (x, y) ∈ solutions → x^2 - 3*y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_pell_equation_solutions_l4000_400042


namespace NUMINAMATH_CALUDE_ant_meeting_point_l4000_400059

/-- Triangle with given side lengths --/
structure Triangle where
  xy : ℝ
  yz : ℝ
  xz : ℝ

/-- Point on the perimeter of the triangle --/
structure PerimeterPoint where
  side : Fin 3
  distance : ℝ

/-- Represents the meeting point of two ants --/
def MeetingPoint (t : Triangle) (w : PerimeterPoint) : Prop :=
  w.side = 1 ∧ w.distance ≤ t.yz

/-- The distance YW --/
def YW (t : Triangle) (w : PerimeterPoint) : ℝ :=
  t.yz - w.distance

/-- Main theorem --/
theorem ant_meeting_point (t : Triangle) (w : PerimeterPoint) :
  t.xy = 8 ∧ t.yz = 10 ∧ t.xz = 12 ∧ MeetingPoint t w →
  YW t w = 3 := by
  sorry

end NUMINAMATH_CALUDE_ant_meeting_point_l4000_400059


namespace NUMINAMATH_CALUDE_real_estate_investment_l4000_400033

theorem real_estate_investment
  (total_investment : ℝ)
  (real_estate_ratio : ℝ)
  (h1 : total_investment = 200000)
  (h2 : real_estate_ratio = 6) :
  let mutual_funds := total_investment / (1 + real_estate_ratio)
  let real_estate := real_estate_ratio * mutual_funds
  real_estate = 171428.58 := by sorry

end NUMINAMATH_CALUDE_real_estate_investment_l4000_400033


namespace NUMINAMATH_CALUDE_remaining_document_arrangements_l4000_400030

/-- Represents the number of documents --/
def total_documents : ℕ := 12

/-- Represents the number of the processed document --/
def processed_document : ℕ := 10

/-- Calculates the number of possible arrangements for the remaining documents --/
def possible_arrangements : ℕ :=
  2 * (Nat.factorial 9 + 2 * Nat.factorial 10 + Nat.factorial 11)

/-- Theorem stating the number of possible ways to handle the remaining documents --/
theorem remaining_document_arrangements :
  possible_arrangements = 95116960 := by sorry

end NUMINAMATH_CALUDE_remaining_document_arrangements_l4000_400030


namespace NUMINAMATH_CALUDE_potato_percentage_l4000_400045

/-- Proves that the percentage of cleared land planted with potato is 30% -/
theorem potato_percentage (total_land : ℝ) (cleared_land : ℝ) (grape_land : ℝ) (tomato_land : ℝ) 
  (h1 : total_land = 3999.9999999999995)
  (h2 : cleared_land = 0.9 * total_land)
  (h3 : grape_land = 0.6 * cleared_land)
  (h4 : tomato_land = 360)
  : (cleared_land - grape_land - tomato_land) / cleared_land = 0.3 := by
  sorry

#eval (3999.9999999999995 * 0.9 - 3999.9999999999995 * 0.9 * 0.6 - 360) / (3999.9999999999995 * 0.9)

end NUMINAMATH_CALUDE_potato_percentage_l4000_400045
