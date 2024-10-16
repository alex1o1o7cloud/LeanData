import Mathlib

namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1725_172519

theorem repeating_decimal_to_fraction :
  ∀ (x : ℚ), (∃ (n : ℕ), 100 * x = 56 + x ∧ n * x = x) → x = 56 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1725_172519


namespace NUMINAMATH_CALUDE_function_inequality_l1725_172595

open Real

-- Define the function F
noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f x / (Real.exp x)

-- State the theorem
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (hf' : Differentiable ℝ (deriv f))
  (h : ∀ x, deriv (deriv f) x < f x) :
  f 2 < Real.exp 2 * f 0 ∧ f 2017 < Real.exp 2017 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1725_172595


namespace NUMINAMATH_CALUDE_min_value_sum_of_roots_l1725_172580

theorem min_value_sum_of_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt (a^2 + 1/a) + Real.sqrt (b^2 + 1/b) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_roots_l1725_172580


namespace NUMINAMATH_CALUDE_orderedPartitions_of_five_l1725_172518

/-- The number of ordered partitions of a positive integer n into positive integers -/
def orderedPartitions (n : ℕ+) : ℕ :=
  sorry

/-- Theorem: The number of ordered partitions of 5 is 16 -/
theorem orderedPartitions_of_five :
  orderedPartitions 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_orderedPartitions_of_five_l1725_172518


namespace NUMINAMATH_CALUDE_solution_set_f_geq_4_range_of_a_l1725_172578

-- Define the function f
def f (x : ℝ) : ℝ := |1 - 2*x| - |1 + x|

-- Theorem for the solution set of f(x) ≥ 4
theorem solution_set_f_geq_4 :
  {x : ℝ | f x ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 6} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, a^2 + 2*a + |1 + x| > f x} = {a : ℝ | a < -3 ∨ a > 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_4_range_of_a_l1725_172578


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l1725_172539

theorem quadratic_unique_solution (b c : ℝ) : 
  (∃! x, 3 * x^2 + b * x + c = 0) →
  b + c = 15 →
  3 * c = b^2 →
  b = (-3 + 3 * Real.sqrt 21) / 2 ∧ 
  c = (33 - 3 * Real.sqrt 21) / 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l1725_172539


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1725_172581

def solution_set (a : ℝ) : Set ℝ := {x | (a * x - 1) / x > 2 * a ∧ x ≠ 0}

theorem inequality_solution_range (a : ℝ) :
  (2 ∉ solution_set a) ↔ (a ≥ -1/2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1725_172581


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1725_172529

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1725_172529


namespace NUMINAMATH_CALUDE_min_sum_abc_l1725_172588

theorem min_sum_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 9 * a + 4 * b = a * b * c) : 
  a + b + c ≥ 10 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    9 * a₀ + 4 * b₀ = a₀ * b₀ * c₀ ∧ a₀ + b₀ + c₀ = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_abc_l1725_172588


namespace NUMINAMATH_CALUDE_cube_painting_cost_l1725_172551

/-- The cost to paint a cube with given dimensions and paint properties -/
theorem cube_painting_cost
  (side_length : ℝ)
  (paint_cost_per_kg : ℝ)
  (paint_coverage_per_kg : ℝ)
  (h_side : side_length = 10)
  (h_cost : paint_cost_per_kg = 60)
  (h_coverage : paint_coverage_per_kg = 20) :
  side_length ^ 2 * 6 / paint_coverage_per_kg * paint_cost_per_kg = 1800 :=
by sorry

end NUMINAMATH_CALUDE_cube_painting_cost_l1725_172551


namespace NUMINAMATH_CALUDE_sum_a_b_equals_negative_one_l1725_172570

theorem sum_a_b_equals_negative_one (a b : ℝ) :
  |a - 2| + (b + 3)^2 = 0 → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_negative_one_l1725_172570


namespace NUMINAMATH_CALUDE_registration_methods_count_l1725_172516

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of extracurricular activity groups -/
def num_groups : ℕ := 3

/-- Each student must sign up for exactly one group -/
axiom one_group_per_student : True

/-- The total number of different registration methods -/
def total_registration_methods : ℕ := num_groups ^ num_students

theorem registration_methods_count :
  total_registration_methods = 3^4 :=
by sorry

end NUMINAMATH_CALUDE_registration_methods_count_l1725_172516


namespace NUMINAMATH_CALUDE_tens_digit_of_8_pow_2023_l1725_172506

/-- The length of the cycle of the last two digits of 8^n -/
def cycle_length : ℕ := 20

/-- The last two digits of 8^3 -/
def last_two_digits_8_cubed : ℕ := 12

/-- The exponent we're interested in -/
def target_exponent : ℕ := 2023

theorem tens_digit_of_8_pow_2023 : 
  (target_exponent % cycle_length = 3) → 
  (last_two_digits_8_cubed / 10 = 1) → 
  (8^target_exponent / 10 % 10 = 1) :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_pow_2023_l1725_172506


namespace NUMINAMATH_CALUDE_tricia_age_correct_l1725_172557

-- Define the ages of each person as natural numbers
def Vincent_age : ℕ := 22
def Rupert_age : ℕ := Vincent_age - 2
def Khloe_age : ℕ := Rupert_age - 10
def Eugene_age : ℕ := 3 * Khloe_age
def Yorick_age : ℕ := 2 * Eugene_age
def Amilia_age : ℕ := Yorick_age / 4
def Tricia_age : ℕ := 5

-- State the theorem
theorem tricia_age_correct : 
  Vincent_age = 22 ∧ 
  Rupert_age = Vincent_age - 2 ∧
  Khloe_age = Rupert_age - 10 ∧
  Khloe_age * 3 = Eugene_age ∧
  Eugene_age * 2 = Yorick_age ∧
  Yorick_age / 4 = Amilia_age ∧
  ∃ (n : ℕ), n * Tricia_age = Amilia_age →
  Tricia_age = 5 :=
by sorry

end NUMINAMATH_CALUDE_tricia_age_correct_l1725_172557


namespace NUMINAMATH_CALUDE_cube_sum_digits_equals_square_l1725_172556

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem cube_sum_digits_equals_square (n : ℕ) :
  n > 0 ∧ n < 1000 ∧ (sum_of_digits n)^3 = n^2 ↔ n = 1 ∨ n = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_digits_equals_square_l1725_172556


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l1725_172505

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

/-- Given that vector a = (2, 4) is collinear with vector b = (x, 6), prove that x = 3 -/
theorem collinear_vectors_x_value :
  ∀ x : ℝ, collinear (2, 4) (x, 6) → x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l1725_172505


namespace NUMINAMATH_CALUDE_divisor_is_six_l1725_172574

def original_number : ℕ := 427398
def subtracted_number : ℕ := 6

theorem divisor_is_six : ∃ (d : ℕ), d > 0 ∧ d = subtracted_number ∧ (original_number - subtracted_number) % d = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisor_is_six_l1725_172574


namespace NUMINAMATH_CALUDE_pages_after_break_l1725_172561

theorem pages_after_break (total_pages : ℕ) (break_percentage : ℚ) 
  (h1 : total_pages = 30) 
  (h2 : break_percentage = 7/10) : 
  total_pages - (total_pages * break_percentage).floor = 9 := by
  sorry

end NUMINAMATH_CALUDE_pages_after_break_l1725_172561


namespace NUMINAMATH_CALUDE_quadratic_root_l1725_172550

/- Given a quadratic equation x^2 - (m+n)x + mn - p = 0 with roots α and β -/
theorem quadratic_root (m n p : ℤ) (α β : ℝ) 
  (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p)
  (h_roots : ∀ x : ℝ, x^2 - (m+n)*x + mn - p = 0 ↔ x = α ∨ x = β)
  (h_alpha : α = 3) :
  β = m + n - 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_l1725_172550


namespace NUMINAMATH_CALUDE_zoey_lottery_split_l1725_172547

theorem zoey_lottery_split (lottery_amount : ℕ) (h1 : lottery_amount = 7348340) :
  ∃ (num_friends : ℕ), 
    (lottery_amount + 1) % (num_friends + 1) = 0 ∧ 
    num_friends = 7348340 := by
  sorry

end NUMINAMATH_CALUDE_zoey_lottery_split_l1725_172547


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l1725_172528

theorem purely_imaginary_condition (m : ℝ) : 
  (((2 * m ^ 2 - 3 * m - 2) : ℂ) + (m ^ 2 - 3 * m + 2) * Complex.I).im ≠ 0 ∧ 
  (((2 * m ^ 2 - 3 * m - 2) : ℂ) + (m ^ 2 - 3 * m + 2) * Complex.I).re = 0 ↔ 
  m = -1/2 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l1725_172528


namespace NUMINAMATH_CALUDE_matrix_equation_equivalence_l1725_172569

theorem matrix_equation_equivalence 
  (n : ℕ) 
  (A B C : Matrix (Fin n) (Fin n) ℝ) 
  (h_inv : IsUnit A) 
  (h_eq : (A - B) * C = B * A⁻¹) : 
  C * (A - B) = A⁻¹ * B := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_equivalence_l1725_172569


namespace NUMINAMATH_CALUDE_trail_mix_nuts_l1725_172543

theorem trail_mix_nuts (walnuts almonds : ℚ) 
  (h1 : walnuts = 0.25)
  (h2 : almonds = 0.25) : 
  walnuts + almonds = 0.50 := by
sorry

end NUMINAMATH_CALUDE_trail_mix_nuts_l1725_172543


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l1725_172513

def complex_number (a b : ℝ) : ℂ := Complex.mk a b

theorem z_in_third_quadrant :
  let i : ℂ := complex_number 0 1
  let z : ℂ := (1 - 2 * i) / i
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l1725_172513


namespace NUMINAMATH_CALUDE_father_age_problem_l1725_172520

/-- The age problem -/
theorem father_age_problem (father_age son_age : ℕ) : 
  father_age = 3 * son_age + 3 →
  father_age + 3 = 2 * (son_age + 3) + 10 →
  father_age = 33 := by
sorry

end NUMINAMATH_CALUDE_father_age_problem_l1725_172520


namespace NUMINAMATH_CALUDE_cafe_tables_l1725_172532

def base5_to_decimal (n : Nat) : Nat :=
  3 * 5^2 + 1 * 5^1 + 0 * 5^0

theorem cafe_tables :
  let total_chairs := base5_to_decimal 310
  let people_per_table := 3
  (total_chairs / people_per_table : Nat) = 26 := by
  sorry

end NUMINAMATH_CALUDE_cafe_tables_l1725_172532


namespace NUMINAMATH_CALUDE_triangle_dimensions_l1725_172552

theorem triangle_dimensions (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0)
  (eq1 : 2 * a / 3 = b) (eq2 : 2 * c = a) (eq3 : b - 2 = c) :
  a = 12 ∧ b = 8 ∧ c = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_dimensions_l1725_172552


namespace NUMINAMATH_CALUDE_tangent_square_area_l1725_172565

/-- A square with two vertices on a circle and two on its tangent -/
structure TangentSquare where
  /-- The radius of the circle -/
  R : ℝ
  /-- The side length of the square -/
  x : ℝ
  /-- Two vertices of the square lie on the circle -/
  vertices_on_circle : x ≤ 2 * R
  /-- Two vertices of the square lie on the tangent -/
  vertices_on_tangent : x^2 / 4 = R^2 - (x - R)^2

/-- The area of a TangentSquare with radius 5 is 64 -/
theorem tangent_square_area (s : TangentSquare) (h : s.R = 5) : s.x^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_tangent_square_area_l1725_172565


namespace NUMINAMATH_CALUDE_grandpa_jungmin_age_ratio_l1725_172521

/-- The ratio of grandpa's age to Jung-min's age this year, given their ages last year -/
def age_ratio (grandpa_last_year : ℕ) (jungmin_last_year : ℕ) : ℚ :=
  (grandpa_last_year + 1) / (jungmin_last_year + 1)

/-- Theorem stating that the ratio of grandpa's age to Jung-min's age this year is 8 -/
theorem grandpa_jungmin_age_ratio : age_ratio 71 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_grandpa_jungmin_age_ratio_l1725_172521


namespace NUMINAMATH_CALUDE_inverse_proportion_point_l1725_172586

theorem inverse_proportion_point : 
  let x : ℝ := 2 * Real.sqrt 2
  let y : ℝ := Real.sqrt 2
  y = 4 / x := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_l1725_172586


namespace NUMINAMATH_CALUDE_magnitude_of_z_l1725_172579

theorem magnitude_of_z (z : ℂ) (h : z^2 = 3 - 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l1725_172579


namespace NUMINAMATH_CALUDE_total_pencils_l1725_172590

theorem total_pencils (num_boxes : ℕ) (pencils_per_box : ℕ) (h1 : num_boxes = 3) (h2 : pencils_per_box = 9) :
  num_boxes * pencils_per_box = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l1725_172590


namespace NUMINAMATH_CALUDE_equilateral_ABC_l1725_172587

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The incenter of a triangle -/
def incenter (t : Triangle) : Point :=
  sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  sorry

/-- Main theorem -/
theorem equilateral_ABC (A B C I X Y Z : Point) :
  let ABC := Triangle.mk A B C
  let BIC := Triangle.mk B I C
  let CIA := Triangle.mk C I A
  let AIB := Triangle.mk A I B
  let XYZ := Triangle.mk X Y Z
  (I = incenter ABC) →
  (X = incenter BIC) →
  (Y = incenter CIA) →
  (Z = incenter AIB) →
  isEquilateral XYZ →
  isEquilateral ABC :=
by
  sorry

end NUMINAMATH_CALUDE_equilateral_ABC_l1725_172587


namespace NUMINAMATH_CALUDE_x_value_proof_l1725_172582

theorem x_value_proof (x : ℚ) (h : (1/4 : ℚ) - (1/5 : ℚ) + (1/10 : ℚ) = 4/x) : x = 80/3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1725_172582


namespace NUMINAMATH_CALUDE_books_combination_l1725_172573

def choose (n : ℕ) (r : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

theorem books_combination : choose 15 3 = 455 := by
  sorry

end NUMINAMATH_CALUDE_books_combination_l1725_172573


namespace NUMINAMATH_CALUDE_product_equals_three_l1725_172567

/-- The repeating decimal 0.333... --/
def repeating_third : ℚ := 1 / 3

/-- The product of 0.333... and 9 --/
def product : ℚ := repeating_third * 9

theorem product_equals_three : product = 3 := by sorry

end NUMINAMATH_CALUDE_product_equals_three_l1725_172567


namespace NUMINAMATH_CALUDE_sector_angle_unchanged_l1725_172536

theorem sector_angle_unchanged 
  (r₁ r₂ : ℝ) 
  (s₁ s₂ : ℝ) 
  (θ₁ θ₂ : ℝ) 
  (h_positive : r₁ > 0 ∧ r₂ > 0)
  (h_radius : r₂ = 2 * r₁)
  (h_arc : s₂ = 2 * s₁)
  (h_angle₁ : s₁ = r₁ * θ₁)
  (h_angle₂ : s₂ = r₂ * θ₂) :
  θ₂ = θ₁ := by
sorry

end NUMINAMATH_CALUDE_sector_angle_unchanged_l1725_172536


namespace NUMINAMATH_CALUDE_river_crossing_problem_l1725_172564

/-- The minimum number of trips required to transport a group of people across a river -/
def min_trips (total_people : ℕ) (boat_capacity : ℕ) (boatman_required : Bool) : ℕ :=
  let effective_capacity := if boatman_required then boat_capacity - 1 else boat_capacity
  (total_people + effective_capacity - 1) / effective_capacity

theorem river_crossing_problem :
  min_trips 14 5 true = 4 := by
  sorry

end NUMINAMATH_CALUDE_river_crossing_problem_l1725_172564


namespace NUMINAMATH_CALUDE_parallel_line_k_value_l1725_172577

/-- Given a line passing through points (5, -3) and (k, 20) that is parallel to the line 3x - 2y = 12, 
    prove that k = 61/3 -/
theorem parallel_line_k_value (k : ℚ) : 
  (∃ (m b : ℚ), (∀ x y : ℚ, y = m * x + b → (x = 5 ∧ y = -3) ∨ (x = k ∧ y = 20)) ∧
                (∀ x y : ℚ, 3 * x - 2 * y = 12 → y = m * x - 6)) → 
  k = 61 / 3 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_k_value_l1725_172577


namespace NUMINAMATH_CALUDE_at_least_one_player_same_outcome_l1725_172589

-- Define the type for players
inductive Player : Type
  | A | B | C | D

-- Define the type for match outcomes
inductive Outcome : Type
  | Win | Lose

-- Define a function type for match results
def MatchResult := Player → Outcome

-- Define the three matches
def match1 : MatchResult := sorry
def match2 : MatchResult := sorry
def match3 : MatchResult := sorry

-- Define a function to check if a player has the same outcome in all matches
def sameOutcomeAllMatches (p : Player) : Prop :=
  (match1 p = match2 p) ∧ (match2 p = match3 p)

-- Theorem statement
theorem at_least_one_player_same_outcome :
  ∃ p : Player, sameOutcomeAllMatches p :=
sorry

end NUMINAMATH_CALUDE_at_least_one_player_same_outcome_l1725_172589


namespace NUMINAMATH_CALUDE_greatest_integer_b_for_all_real_domain_l1725_172554

theorem greatest_integer_b_for_all_real_domain : ∃ (b : ℤ), 
  (∀ (x : ℝ), x^2 + (b : ℝ) * x + 7 ≠ 0) ∧ 
  (∀ (b' : ℤ), (∀ (x : ℝ), x^2 + (b' : ℝ) * x + 7 ≠ 0) → b' ≤ b) ∧
  b = 5 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_b_for_all_real_domain_l1725_172554


namespace NUMINAMATH_CALUDE_not_power_of_two_l1725_172541

theorem not_power_of_two (m n : ℕ+) : ¬∃ k : ℕ, (36 * m.val + n.val) * (m.val + 36 * n.val) = 2^k := by
  sorry

end NUMINAMATH_CALUDE_not_power_of_two_l1725_172541


namespace NUMINAMATH_CALUDE_integral_proof_l1725_172562

open Real

theorem integral_proof (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ -2) :
  let f : ℝ → ℝ := λ x => (x^3 + 6*x^2 + 14*x + 10) / ((x+1)*(x+2)^3)
  let F : ℝ → ℝ := λ x => log (abs (x+1)) - 1 / (x+2)^2
  deriv F x = f x := by sorry

end NUMINAMATH_CALUDE_integral_proof_l1725_172562


namespace NUMINAMATH_CALUDE_theorem_1_theorem_2_l1725_172594

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Theorem 1: If a is parallel to α and b is perpendicular to α, then a is perpendicular to b
theorem theorem_1 (a b : Line) (α : Plane) :
  parallel a α → perpendicular b α → perpendicular_lines a b :=
by sorry

-- Theorem 2: If a is perpendicular to α and a is parallel to β, then α is perpendicular to β
theorem theorem_2 (a : Line) (α β : Plane) :
  perpendicular a α → parallel a β → perpendicular_planes α β :=
by sorry

end NUMINAMATH_CALUDE_theorem_1_theorem_2_l1725_172594


namespace NUMINAMATH_CALUDE_non_redundant_password_count_l1725_172508

/-- A password is a string of characters. -/
def Password := String

/-- The set of available characters for passwords. -/
def AvailableChars : Finset Char := sorry

/-- A password is redundant if it contains a block of consecutive characters
    that can be colored red and blue such that the red and blue substrings are identical. -/
def IsRedundant (p : Password) : Prop := sorry

/-- The number of non-redundant passwords of length n. -/
def NonRedundantCount (n : ℕ) : ℕ := sorry

/-- There are at least 18^n non-redundant passwords of length n for any n ≥ 1. -/
theorem non_redundant_password_count (n : ℕ) (h : n ≥ 1) :
  NonRedundantCount n ≥ 18^n := by sorry

end NUMINAMATH_CALUDE_non_redundant_password_count_l1725_172508


namespace NUMINAMATH_CALUDE_initial_salt_concentration_l1725_172596

/-- The initial volume of saltwater solution in gallons -/
def x : ℝ := 120

/-- The initial salt concentration as a percentage -/
def C : ℝ := 18.33333333333333

theorem initial_salt_concentration (C : ℝ) :
  (C / 100 * x + 16) / (3 / 4 * x + 8 + 16) = 1 / 3 → C = 18.33333333333333 :=
by sorry

end NUMINAMATH_CALUDE_initial_salt_concentration_l1725_172596


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l1725_172512

def a (n : ℕ) : ℕ := n.factorial + 3 * n

theorem max_gcd_consecutive_terms :
  ∃ (k : ℕ), ∀ (n : ℕ), Nat.gcd (a n) (a (n + 1)) ≤ 3 ∧
  (∃ (m : ℕ), Nat.gcd (a m) (a (m + 1)) = 3) :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l1725_172512


namespace NUMINAMATH_CALUDE_magic_square_sum_l1725_172560

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a b c d e : ℕ)
  (top_left : ℕ := 30)
  (top_right : ℕ := 27)
  (middle_left : ℕ := 33)
  (bottom_middle : ℕ := 18)
  (sum : ℕ)
  (row_sums : sum = top_left + b + top_right)
  (col_sums : sum = top_left + middle_left + a)
  (diag_sums : sum = top_left + c + e)
  (middle_row : sum = middle_left + c + d)
  (bottom_row : sum = a + bottom_middle + e)

/-- The sum of a and d in the magic square is 38 -/
theorem magic_square_sum (ms : MagicSquare) : ms.a + ms.d = 38 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_l1725_172560


namespace NUMINAMATH_CALUDE_unique_solution_l1725_172538

/-- Represents the letters used in the triangle puzzle -/
inductive Letter
| A | B | C | D | E | F

/-- Represents the mapping of letters to numbers -/
def LetterMapping := Letter → Fin 6

/-- Checks if a mapping is valid according to the puzzle rules -/
def is_valid_mapping (m : LetterMapping) : Prop :=
  m Letter.A ≠ m Letter.B ∧ m Letter.A ≠ m Letter.C ∧ m Letter.A ≠ m Letter.D ∧ m Letter.A ≠ m Letter.E ∧ m Letter.A ≠ m Letter.F ∧
  m Letter.B ≠ m Letter.C ∧ m Letter.B ≠ m Letter.D ∧ m Letter.B ≠ m Letter.E ∧ m Letter.B ≠ m Letter.F ∧
  m Letter.C ≠ m Letter.D ∧ m Letter.C ≠ m Letter.E ∧ m Letter.C ≠ m Letter.F ∧
  m Letter.D ≠ m Letter.E ∧ m Letter.D ≠ m Letter.F ∧
  m Letter.E ≠ m Letter.F ∧
  (m Letter.B).val + (m Letter.D).val + (m Letter.E).val = 14 ∧
  (m Letter.C).val + (m Letter.E).val + (m Letter.F).val = 12

/-- The unique solution to the puzzle -/
def solution : LetterMapping :=
  fun l => match l with
  | Letter.A => 0
  | Letter.B => 2
  | Letter.C => 1
  | Letter.D => 4
  | Letter.E => 5
  | Letter.F => 3

/-- Theorem stating that the solution is the only valid mapping -/
theorem unique_solution :
  is_valid_mapping solution ∧ ∀ m : LetterMapping, is_valid_mapping m → m = solution := by
  sorry


end NUMINAMATH_CALUDE_unique_solution_l1725_172538


namespace NUMINAMATH_CALUDE_existence_equivalence_l1725_172559

theorem existence_equivalence (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ 2^x * (3*x + a) < 1) ↔ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_existence_equivalence_l1725_172559


namespace NUMINAMATH_CALUDE_soccer_team_selection_l1725_172522

theorem soccer_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) (quad_starters : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 6 →
  quad_starters = 2 →
  (quadruplets.choose quad_starters) * ((total_players - quadruplets).choose (starters - quad_starters)) = 2970 :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_selection_l1725_172522


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1725_172576

variable (w z : ℂ)

theorem complex_magnitude_problem (h1 : w * z = 24 - 10 * I) (h2 : Complex.abs w = Real.sqrt 34) :
  Complex.abs z = (13 * Real.sqrt 34) / 17 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1725_172576


namespace NUMINAMATH_CALUDE_unique_solution_of_equation_l1725_172535

theorem unique_solution_of_equation (x : ℝ) :
  x ≥ 0 →
  (2021 * x = 2022 * (x^(2021/2022)) - 1) ↔
  x = 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_of_equation_l1725_172535


namespace NUMINAMATH_CALUDE_A_times_B_equals_result_l1725_172504

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1/2| < 1}
def B : Set ℝ := {x | 1/x ≥ 1}

-- Define the operation ×
def times (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∪ Y ∧ x ∉ X ∩ Y}

-- State the theorem
theorem A_times_B_equals_result : 
  times A B = {x | -1/2 < x ∧ x ≤ 0 ∨ 1 < x ∧ x < 3/2} := by sorry

end NUMINAMATH_CALUDE_A_times_B_equals_result_l1725_172504


namespace NUMINAMATH_CALUDE_sign_sum_theorem_l1725_172597

theorem sign_sum_theorem (a b c : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hsum : a + b + c = 0) :
  (a / |a| + b / |b| + c / |c| - (a * b * c) / |a * b * c|) = 2 ∨
  (a / |a| + b / |b| + c / |c| - (a * b * c) / |a * b * c|) = -2 :=
by sorry

end NUMINAMATH_CALUDE_sign_sum_theorem_l1725_172597


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_sin_2beta_zero_l1725_172503

theorem sin_2alpha_plus_sin_2beta_zero (α β : Real) 
  (h : Real.sin α * Real.sin β + Real.cos α * Real.cos β = 0) : 
  Real.sin (2 * α) + Real.sin (2 * β) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_sin_2beta_zero_l1725_172503


namespace NUMINAMATH_CALUDE_cost_price_is_41_l1725_172533

/-- Calculates the cost price per metre of cloth given the total length,
    total selling price, and loss per metre. -/
def cost_price_per_metre (total_length : ℕ) (total_selling_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  (total_selling_price + total_length * loss_per_metre) / total_length

/-- Proves that the cost price per metre of cloth is 41 rupees given the specified conditions. -/
theorem cost_price_is_41 :
  cost_price_per_metre 500 18000 5 = 41 := by
  sorry

#eval cost_price_per_metre 500 18000 5

end NUMINAMATH_CALUDE_cost_price_is_41_l1725_172533


namespace NUMINAMATH_CALUDE_solve_determinant_equation_l1725_172544

-- Define the determinant operation
def det (a b c d : ℚ) : ℚ := a * d - b * c

-- Theorem statement
theorem solve_determinant_equation :
  ∀ x : ℚ, det 2 4 (1 - x) 5 = 18 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_determinant_equation_l1725_172544


namespace NUMINAMATH_CALUDE_parabola_directrix_l1725_172568

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = (x^2 - 8*x + 12) / 16

/-- The directrix equation -/
def directrix (y : ℝ) : Prop := y = -1/2

/-- Theorem: The directrix of the given parabola is y = -1/2 -/
theorem parabola_directrix :
  ∀ (x y : ℝ), parabola x y → (∃ (d : ℝ), directrix d ∧ 
    d = y - (1/4) ∧ 
    ∀ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y → 
      (p.1 - 4)^2 + (p.2 - 0)^2 = (p.2 - d)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1725_172568


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1725_172509

theorem trigonometric_simplification (α : ℝ) : 
  (1 + Real.tan (2 * α))^2 - 2 * (Real.tan (2 * α))^2 / (1 + (Real.tan (2 * α))^2) - 
  Real.sin (4 * α) - 1 = -2 * (Real.sin (2 * α))^2 := by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1725_172509


namespace NUMINAMATH_CALUDE_power_multiplication_l1725_172546

theorem power_multiplication (a : ℝ) : a * a^3 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1725_172546


namespace NUMINAMATH_CALUDE_vector_subtraction_l1725_172571

/-- Given two plane vectors a and b, prove that a - 2b equals (3, 7) -/
theorem vector_subtraction (a b : ℝ × ℝ) (ha : a = (5, 3)) (hb : b = (1, -2)) :
  a - 2 • b = (3, 7) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1725_172571


namespace NUMINAMATH_CALUDE_probability_theorem_l1725_172583

structure ProfessionalGroup where
  women_percentage : ℝ
  men_percentage : ℝ
  nonbinary_percentage : ℝ
  women_engineer_percentage : ℝ
  women_doctor_percentage : ℝ
  men_engineer_percentage : ℝ
  men_doctor_percentage : ℝ
  nonbinary_engineer_percentage : ℝ
  nonbinary_translator_percentage : ℝ

def probability_selection (group : ProfessionalGroup) : ℝ :=
  group.women_percentage * group.women_engineer_percentage +
  group.men_percentage * group.men_doctor_percentage +
  group.nonbinary_percentage * group.nonbinary_translator_percentage

theorem probability_theorem (group : ProfessionalGroup) 
  (h1 : group.women_percentage = 0.70)
  (h2 : group.men_percentage = 0.20)
  (h3 : group.nonbinary_percentage = 0.10)
  (h4 : group.women_engineer_percentage = 0.20)
  (h5 : group.men_doctor_percentage = 0.20)
  (h6 : group.nonbinary_translator_percentage = 0.20) :
  probability_selection group = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l1725_172583


namespace NUMINAMATH_CALUDE_minimum_nickels_needed_l1725_172572

/-- The price of the book in cents -/
def book_price : ℕ := 4250

/-- The number of $10 bills Jane has -/
def ten_dollar_bills : ℕ := 4

/-- The number of quarters Jane has -/
def quarters : ℕ := 5

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The minimum number of nickels Jane needs to afford the book -/
def min_nickels : ℕ := 25

theorem minimum_nickels_needed :
  ∀ n : ℕ,
  (ten_dollar_bills * 1000 + quarters * 25 + n * nickel_value ≥ book_price) →
  (n ≥ min_nickels) :=
sorry

end NUMINAMATH_CALUDE_minimum_nickels_needed_l1725_172572


namespace NUMINAMATH_CALUDE_probability_for_specific_cube_l1725_172537

/-- Represents a cube with painted faces -/
structure PaintedCube where
  side_length : ℕ
  total_cubes : ℕ
  full_face_painted : ℕ
  half_face_painted : ℕ

/-- Calculates the probability of selecting one cube with exactly one painted face
    and one cube with no painted faces when two cubes are randomly selected -/
def probability_one_painted_one_unpainted (cube : PaintedCube) : ℚ :=
  let one_face_painted := cube.full_face_painted - cube.half_face_painted
  let no_face_painted := cube.total_cubes - cube.full_face_painted - cube.half_face_painted
  let total_combinations := (cube.total_cubes * (cube.total_cubes - 1)) / 2
  let favorable_outcomes := one_face_painted * no_face_painted
  favorable_outcomes / total_combinations

/-- The main theorem stating the probability for the specific cube configuration -/
theorem probability_for_specific_cube : 
  let cube := PaintedCube.mk 5 125 25 12
  probability_one_painted_one_unpainted cube = 44 / 155 := by
  sorry

end NUMINAMATH_CALUDE_probability_for_specific_cube_l1725_172537


namespace NUMINAMATH_CALUDE_fraction_of_powers_equals_five_fourths_l1725_172511

theorem fraction_of_powers_equals_five_fourths :
  (3^1007 + 3^1005) / (3^1007 - 3^1005) = 5/4 := by sorry

end NUMINAMATH_CALUDE_fraction_of_powers_equals_five_fourths_l1725_172511


namespace NUMINAMATH_CALUDE_percentage_of_women_professors_l1725_172593

-- Define the percentage of professors who are women
variable (W : ℝ)

-- Define the percentage of professors who are tenured
def T : ℝ := 70

-- Define the principle of inclusion-exclusion
axiom inclusion_exclusion : W + T - (W * T / 100) = 90

-- Define the percentage of men who are tenured
axiom men_tenured : (100 - W) * 52 / 100 = T - (W * T / 100)

-- Theorem to prove
theorem percentage_of_women_professors : ∃ ε > 0, abs (W - 79.17) < ε := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_women_professors_l1725_172593


namespace NUMINAMATH_CALUDE_fourth_number_value_l1725_172507

theorem fourth_number_value (numbers : List ℝ) 
  (h1 : numbers.length = 6)
  (h2 : numbers.sum / numbers.length = 30)
  (h3 : (numbers.take 4).sum / 4 = 25)
  (h4 : (numbers.drop 3).sum / 3 = 35) :
  numbers[3] = 25 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_value_l1725_172507


namespace NUMINAMATH_CALUDE_consecutive_letters_probability_l1725_172523

/-- The number of cards in the deck -/
def n : ℕ := 5

/-- The number of cards to draw -/
def k : ℕ := 2

/-- The number of ways to choose k cards from n cards -/
def total_outcomes : ℕ := n.choose k

/-- The number of pairs of consecutive letters -/
def favorable_outcomes : ℕ := n - 1

/-- The probability of drawing 2 cards with consecutive letters -/
def probability : ℚ := favorable_outcomes / total_outcomes

/-- Theorem stating that the probability of drawing 2 cards with consecutive letters is 2/5 -/
theorem consecutive_letters_probability :
  probability = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_consecutive_letters_probability_l1725_172523


namespace NUMINAMATH_CALUDE_xyz_value_l1725_172500

theorem xyz_value (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + 1/y = 5)
  (eq2 : y + 1/z = 2)
  (eq3 : z + 1/x = 8/3) :
  x * y * z = 8 + 3 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1725_172500


namespace NUMINAMATH_CALUDE_saplings_in_park_l1725_172526

theorem saplings_in_park (total_trees : ℕ) (ancient_oaks : ℕ) (fir_trees : ℕ) : 
  total_trees = 96 → ancient_oaks = 15 → fir_trees = 23 → 
  total_trees - (ancient_oaks + fir_trees) = 58 := by
  sorry

end NUMINAMATH_CALUDE_saplings_in_park_l1725_172526


namespace NUMINAMATH_CALUDE_haleigh_cats_count_l1725_172553

/-- The number of cats Haleigh has -/
def num_cats : ℕ := 10

/-- The number of dogs Haleigh has -/
def num_dogs : ℕ := 4

/-- The total number of leggings needed -/
def total_leggings : ℕ := 14

/-- Each animal needs one pair of leggings -/
def leggings_per_animal : ℕ := 1

theorem haleigh_cats_count :
  num_cats = total_leggings - num_dogs * leggings_per_animal :=
by sorry

end NUMINAMATH_CALUDE_haleigh_cats_count_l1725_172553


namespace NUMINAMATH_CALUDE_sum_of_max_min_a_l1725_172527

theorem sum_of_max_min_a (a b : ℝ) (h : a - 2*a*b + 2*a*b^2 + 4 = 0) :
  ∃ (a_max a_min : ℝ),
    (∀ x : ℝ, (∃ y : ℝ, x - 2*x*y + 2*x*y^2 + 4 = 0) → x ≤ a_max ∧ x ≥ a_min) ∧
    a_max + a_min = -8 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_min_a_l1725_172527


namespace NUMINAMATH_CALUDE_flour_info_doesnt_determine_sugar_l1725_172545

/-- Represents a cake recipe --/
structure Recipe where
  flour : ℕ
  sugar : ℕ

/-- Represents the state of Jessica's baking process --/
structure BakingProcess where
  flour_added : ℕ
  flour_needed : ℕ

/-- Given information about flour doesn't determine sugar amount --/
theorem flour_info_doesnt_determine_sugar 
  (recipe : Recipe) 
  (baking : BakingProcess) 
  (h1 : recipe.flour = 8)
  (h2 : baking.flour_added = 4)
  (h3 : baking.flour_needed = 4)
  (h4 : baking.flour_added + baking.flour_needed = recipe.flour) :
  ∃ (r1 r2 : Recipe), r1.flour = r2.flour ∧ r1.sugar ≠ r2.sugar :=
sorry

end NUMINAMATH_CALUDE_flour_info_doesnt_determine_sugar_l1725_172545


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1725_172548

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^2 + 3

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 4 * x

-- Define the point of tangency
def x₀ : ℝ := -1

-- Define the slope of the tangent line at x₀
def m : ℝ := f' x₀

-- Define a point on the curve at x₀
def p : ℝ × ℝ := (x₀, f x₀)

-- Theorem statement
theorem tangent_line_equation :
  ∀ x y : ℝ, y - p.2 = m * (x - p.1) ↔ y = -4 * x + 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1725_172548


namespace NUMINAMATH_CALUDE_elephant_count_in_park_final_elephant_count_l1725_172502

/-- Calculates the final number of elephants in Utopia National Park after a specific time period --/
theorem elephant_count_in_park (initial_count : ℕ) (exodus_rate : ℕ) (exodus_duration : ℕ) 
  (entry_rate : ℕ) (entry_duration : ℕ) : ℕ :=
  let elephants_left := exodus_rate * exodus_duration
  let elephants_entered := entry_rate * entry_duration
  initial_count - elephants_left + elephants_entered

/-- Proves that the final number of elephants in Utopia National Park is 28,980 --/
theorem final_elephant_count : 
  elephant_count_in_park 30000 2880 4 1500 7 = 28980 := by
  sorry

end NUMINAMATH_CALUDE_elephant_count_in_park_final_elephant_count_l1725_172502


namespace NUMINAMATH_CALUDE_fruit_purchase_cost_is_correct_l1725_172515

/-- Calculates the total cost in dollars for a fruit purchase with given conditions -/
def fruitPurchaseCost (grapeKg : ℝ) (grapeRate : ℝ) (mangoKg : ℝ) (mangoRate : ℝ)
                      (appleKg : ℝ) (appleRate : ℝ) (orangeKg : ℝ) (orangeRate : ℝ)
                      (grapeMangoeDiscountRate : ℝ) (appleOrangeFixedDiscount : ℝ)
                      (salesTaxRate : ℝ) (fixedTax : ℝ) (exchangeRate : ℝ) : ℝ :=
  let grapeCost := grapeKg * grapeRate
  let mangoCost := mangoKg * mangoRate
  let appleCost := appleKg * appleRate
  let orangeCost := orangeKg * orangeRate
  let grapeMangoeTotal := grapeCost + mangoCost
  let appleOrangeTotal := appleCost + orangeCost
  let grapeMangoeDiscount := grapeMangoeTotal * grapeMangoeDiscountRate
  let discountedGrapeMangoe := grapeMangoeTotal - grapeMangoeDiscount
  let discountedAppleOrange := appleOrangeTotal - appleOrangeFixedDiscount
  let totalDiscountedCost := discountedGrapeMangoe + discountedAppleOrange
  let salesTax := totalDiscountedCost * salesTaxRate
  let totalTax := salesTax + fixedTax
  let totalAmount := totalDiscountedCost + totalTax
  totalAmount * exchangeRate

/-- Theorem stating that the fruit purchase cost under given conditions is $323.79 -/
theorem fruit_purchase_cost_is_correct :
  fruitPurchaseCost 7 68 9 48 5 55 4 38 0.1 25 0.05 15 0.25 = 323.79 := by
  sorry


end NUMINAMATH_CALUDE_fruit_purchase_cost_is_correct_l1725_172515


namespace NUMINAMATH_CALUDE_coin_probability_impossibility_l1725_172542

theorem coin_probability_impossibility : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
sorry

end NUMINAMATH_CALUDE_coin_probability_impossibility_l1725_172542


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1725_172510

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 40)
  (weighted_diff_eq : 2 * y - 3 * x = 10) :
  |y - x| = 12 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1725_172510


namespace NUMINAMATH_CALUDE_chord_cosine_l1725_172531

theorem chord_cosine (r : ℝ) (γ δ : ℝ) : 
  0 < r →
  0 < γ →
  0 < δ →
  γ + δ < π →
  5^2 = 2 * r^2 * (1 - Real.cos γ) →
  12^2 = 2 * r^2 * (1 - Real.cos δ) →
  13^2 = 2 * r^2 * (1 - Real.cos (γ + δ)) →
  Real.cos γ = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_chord_cosine_l1725_172531


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l1725_172599

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a + b + c + d = 200 → 
  a * b + b * c + c * d ≤ 10000 ∧ 
  ∃ a b c d, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
             a + b + c + d = 200 ∧ 
             a * b + b * c + c * d = 10000 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l1725_172599


namespace NUMINAMATH_CALUDE_total_people_is_123_l1725_172534

/-- Calculates the total number of people on a bus given the number of boys and additional information. -/
def total_people_on_bus (num_boys : ℕ) : ℕ :=
  let num_girls := num_boys + (2 * num_boys) / 5
  let num_students := num_boys + num_girls
  let num_adults := 3  -- driver, assistant, and teacher
  num_students + num_adults

/-- Theorem stating that given the conditions, the total number of people on the bus is 123. -/
theorem total_people_is_123 : total_people_on_bus 50 = 123 := by
  sorry

#eval total_people_on_bus 50

end NUMINAMATH_CALUDE_total_people_is_123_l1725_172534


namespace NUMINAMATH_CALUDE_dragons_total_games_dragons_total_games_is_90_l1725_172540

theorem dragons_total_games : ℕ → Prop :=
  fun total_games =>
    ∃ (pre_tournament_games : ℕ) (pre_tournament_wins : ℕ),
      -- Condition 1: 60% win rate before tournament
      pre_tournament_wins = (6 * pre_tournament_games) / 10 ∧
      -- Condition 2: 9 wins and 3 losses in tournament
      total_games = pre_tournament_games + 12 ∧
      -- Condition 3: 62% overall win rate after tournament
      (pre_tournament_wins + 9) = (62 * total_games) / 100 ∧
      -- Prove that total games is 90
      total_games = 90

theorem dragons_total_games_is_90 : dragons_total_games 90 := by
  sorry

end NUMINAMATH_CALUDE_dragons_total_games_dragons_total_games_is_90_l1725_172540


namespace NUMINAMATH_CALUDE_min_value_on_negative_reals_l1725_172591

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^9 + 2

-- State the theorem
theorem min_value_on_negative_reals (a b : ℝ) :
  (∀ x > 0, f a b x ≤ 5) ∧ (∃ x > 0, f a b x = 5) →
  (∀ x < 0, f a b x ≥ -1) ∧ (∃ x < 0, f a b x = -1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_negative_reals_l1725_172591


namespace NUMINAMATH_CALUDE_swordtails_count_l1725_172566

/-- The number of Goldfish Layla has -/
def num_goldfish : ℕ := 2

/-- The amount of food each Goldfish gets (in teaspoons) -/
def goldfish_food : ℚ := 1

/-- The number of Guppies Layla has -/
def num_guppies : ℕ := 8

/-- The amount of food each Guppy gets (in teaspoons) -/
def guppy_food : ℚ := 1/2

/-- The amount of food each Swordtail gets (in teaspoons) -/
def swordtail_food : ℚ := 2

/-- The total amount of food given to all fish (in teaspoons) -/
def total_food : ℚ := 12

/-- The number of Swordtails Layla has -/
def num_swordtails : ℕ := 3

theorem swordtails_count : 
  (num_goldfish : ℚ) * goldfish_food + 
  (num_guppies : ℚ) * guppy_food + 
  (num_swordtails : ℚ) * swordtail_food = total_food :=
sorry

end NUMINAMATH_CALUDE_swordtails_count_l1725_172566


namespace NUMINAMATH_CALUDE_pool_width_is_40_l1725_172575

/-- Represents a rectangular pool with given length and width -/
structure Pool where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangular pool -/
def Pool.perimeter (p : Pool) : ℝ := 2 * (p.length + p.width)

/-- Represents the speeds of Ruth and Sarah -/
structure Speeds where
  ruth : ℝ
  sarah : ℝ

theorem pool_width_is_40 (p : Pool) (s : Speeds) : p.width = 40 :=
  by
  have h1 : p.length = 50 := by sorry
  have h2 : s.ruth = 3 * s.sarah := by sorry
  have h3 : 6 * p.length = 5 * p.perimeter := by sorry
  sorry

end NUMINAMATH_CALUDE_pool_width_is_40_l1725_172575


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l1725_172530

theorem trigonometric_inequality (h : 3 * Real.pi / 8 ∈ Set.Ioo 0 (Real.pi / 2)) :
  Real.sin (Real.cos (3 * Real.pi / 8)) < Real.cos (Real.sin (3 * Real.pi / 8)) ∧
  Real.cos (Real.sin (3 * Real.pi / 8)) < Real.sin (Real.sin (3 * Real.pi / 8)) ∧
  Real.sin (Real.sin (3 * Real.pi / 8)) < Real.cos (Real.cos (3 * Real.pi / 8)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l1725_172530


namespace NUMINAMATH_CALUDE_business_investment_proof_l1725_172555

/-- Praveen's initial investment in the business -/
def praveenInvestment : ℕ := 35280

/-- Hari's investment in the business -/
def hariInvestment : ℕ := 10080

/-- Praveen's investment duration in months -/
def praveenDuration : ℕ := 12

/-- Hari's investment duration in months -/
def hariDuration : ℕ := 7

/-- Praveen's share in the profit ratio -/
def praveenShare : ℕ := 2

/-- Hari's share in the profit ratio -/
def hariShare : ℕ := 3

theorem business_investment_proof :
  praveenInvestment * praveenDuration * hariShare = 
  hariInvestment * hariDuration * praveenShare := by
  sorry

end NUMINAMATH_CALUDE_business_investment_proof_l1725_172555


namespace NUMINAMATH_CALUDE_average_calls_is_40_l1725_172585

/-- Represents the number of calls answered each day for a week --/
structure WeekCalls where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the average number of calls per day --/
def averageCalls (w : WeekCalls) : ℚ :=
  (w.monday + w.tuesday + w.wednesday + w.thursday + w.friday) / 5

/-- Theorem stating that for the given week of calls, the average is 40 --/
theorem average_calls_is_40 (w : WeekCalls) 
    (h1 : w.monday = 35)
    (h2 : w.tuesday = 46)
    (h3 : w.wednesday = 27)
    (h4 : w.thursday = 61)
    (h5 : w.friday = 31) :
    averageCalls w = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_calls_is_40_l1725_172585


namespace NUMINAMATH_CALUDE_min_value_expression_l1725_172563

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1/a + a/(b^2) + b ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1725_172563


namespace NUMINAMATH_CALUDE_total_insects_theorem_l1725_172558

/-- The number of geckos -/
def num_geckos : ℕ := 5

/-- The number of insects eaten by each gecko -/
def insects_per_gecko : ℕ := 6

/-- The number of lizards -/
def num_lizards : ℕ := 3

/-- The number of insects eaten by each lizard -/
def insects_per_lizard : ℕ := 2 * insects_per_gecko

/-- The total number of insects eaten by both geckos and lizards -/
def total_insects_eaten : ℕ := num_geckos * insects_per_gecko + num_lizards * insects_per_lizard

theorem total_insects_theorem : total_insects_eaten = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_insects_theorem_l1725_172558


namespace NUMINAMATH_CALUDE_min_m_plus_n_l1725_172524

theorem min_m_plus_n (m n : ℕ+) (h : 90 * m = n^3) : 
  ∃ (m' n' : ℕ+), 90 * m' = n'^3 ∧ m' + n' ≤ m + n ∧ m' + n' = 120 :=
sorry

end NUMINAMATH_CALUDE_min_m_plus_n_l1725_172524


namespace NUMINAMATH_CALUDE_triangle_type_l1725_172592

theorem triangle_type (A : Real) (hA : 0 < A ∧ A < π) 
  (h : Real.sin A + Real.cos A = 7/12) : 
  ∀ (B C : Real), 0 < B ∧ 0 < C ∧ A + B + C = π → 
  A < π/2 ∧ B < π/2 ∧ C < π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_type_l1725_172592


namespace NUMINAMATH_CALUDE_katies_old_friends_games_l1725_172598

theorem katies_old_friends_games 
  (total_friends_games : ℕ) 
  (new_friends_games : ℕ) 
  (h1 : total_friends_games = 141) 
  (h2 : new_friends_games = 88) : 
  total_friends_games - new_friends_games = 53 := by
sorry

end NUMINAMATH_CALUDE_katies_old_friends_games_l1725_172598


namespace NUMINAMATH_CALUDE_function_inequality_l1725_172549

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x : ℝ, deriv f x > f x) (a : ℝ) (ha : a > 0) : 
  f a > Real.exp a * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1725_172549


namespace NUMINAMATH_CALUDE_female_democrat_ratio_l1725_172584

theorem female_democrat_ratio (total_participants male_participants female_participants : ℕ)
  (female_democrats male_democrats : ℕ) :
  total_participants = 750 →
  total_participants = male_participants + female_participants →
  male_democrats = male_participants / 4 →
  female_democrats = 125 →
  male_democrats + female_democrats = total_participants / 3 →
  2 * female_democrats = female_participants :=
by
  sorry

end NUMINAMATH_CALUDE_female_democrat_ratio_l1725_172584


namespace NUMINAMATH_CALUDE_family_ages_solution_l1725_172525

structure Family where
  father_age : ℕ
  mother_age : ℕ
  john_age : ℕ
  ben_age : ℕ
  mary_age : ℕ

def age_difference (f : Family) : ℕ :=
  f.father_age - f.mother_age

theorem family_ages_solution (f : Family) 
  (h1 : age_difference f = f.john_age - f.ben_age)
  (h2 : age_difference f = f.ben_age - f.mary_age)
  (h3 : f.john_age * f.ben_age = f.father_age)
  (h4 : f.ben_age * f.mary_age = f.mother_age)
  (h5 : f.father_age + f.mother_age + f.john_age + f.ben_age + f.mary_age = 90)
  : f.father_age = 36 ∧ f.mother_age = 36 ∧ f.john_age = 6 ∧ f.ben_age = 6 ∧ f.mary_age = 6 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_solution_l1725_172525


namespace NUMINAMATH_CALUDE_extreme_points_imply_a_l1725_172514

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b * x^2 + x

theorem extreme_points_imply_a (a b : ℝ) :
  (∀ x, x > 0 → (deriv (f a b)) x = 0 ↔ x = 1 ∨ x = 2) →
  a = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_extreme_points_imply_a_l1725_172514


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l1725_172501

theorem quadratic_roots_sum_of_squares (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - m*x₁ + 2*m - 1 = 0 ∧ 
    x₂^2 - m*x₂ + 2*m - 1 = 0 ∧
    x₁^2 + x₂^2 = 7) → 
  m = 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l1725_172501


namespace NUMINAMATH_CALUDE_integer_coordinates_cubic_l1725_172517

/-- A cubic function with integer coordinates for extrema and inflection point -/
structure IntegerCubic where
  n : ℤ
  p : ℤ
  c : ℤ

/-- The cubic function with the given coefficients -/
def cubic_function (f : IntegerCubic) (x : ℝ) : ℝ :=
  x^3 + 3 * f.n * x^2 + 3 * (f.n^2 - f.p^2) * x + f.c

/-- The first derivative of the cubic function -/
def cubic_derivative (f : IntegerCubic) (x : ℝ) : ℝ :=
  3 * x^2 + 6 * f.n * x + 3 * (f.n^2 - f.p^2)

/-- The second derivative of the cubic function -/
def cubic_second_derivative (f : IntegerCubic) (x : ℝ) : ℝ :=
  6 * x + 6 * f.n

/-- Theorem: The cubic function has integer coordinates for extrema and inflection point -/
theorem integer_coordinates_cubic (f : IntegerCubic) :
  ∃ (x1 x2 xi : ℤ),
    (cubic_derivative f x1 = 0 ∧ cubic_derivative f x2 = 0) ∧
    cubic_second_derivative f xi = 0 ∧
    (∀ x : ℤ, cubic_derivative f x = 0 → x = x1 ∨ x = x2) ∧
    (∀ x : ℤ, cubic_second_derivative f x = 0 → x = xi) :=
  sorry

end NUMINAMATH_CALUDE_integer_coordinates_cubic_l1725_172517
