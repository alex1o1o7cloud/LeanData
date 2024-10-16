import Mathlib

namespace NUMINAMATH_CALUDE_f_monotone_decreasing_iff_a_in_range_l2921_292160

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then (2*a - 1)*x + a else Real.log x / Real.log a

-- Define monotonically decreasing
def monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem f_monotone_decreasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → monotone_decreasing (f a)) ↔ (0 < a ∧ a ≤ 1/3) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_iff_a_in_range_l2921_292160


namespace NUMINAMATH_CALUDE_inradius_formula_l2921_292119

theorem inradius_formula (β γ R : Real) (hβ : 0 < β) (hγ : 0 < γ) (hβγ : β + γ < π) (hR : R > 0) :
  ∃ (r : Real), r = 4 * R * Real.sin (β / 2) * Real.sin (γ / 2) * Real.cos ((β + γ) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inradius_formula_l2921_292119


namespace NUMINAMATH_CALUDE_bottle_weight_problem_l2921_292197

/-- Given the weight of 3 glass bottles and the weight difference between glass and plastic bottles,
    calculate the total weight of 4 glass bottles and 5 plastic bottles. -/
theorem bottle_weight_problem (weight_3_glass : ℕ) (weight_diff : ℕ) : 
  weight_3_glass = 600 → weight_diff = 150 → 
  (4 * (weight_3_glass / 3 + weight_diff) + 5 * (weight_3_glass / 3 - weight_diff / 3)) = 1050 := by
  sorry

end NUMINAMATH_CALUDE_bottle_weight_problem_l2921_292197


namespace NUMINAMATH_CALUDE_circle_tangent_and_chord_l2921_292182

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define point P
def P : ℝ × ℝ := (3, 4)

-- Define the tangent line l
def l (x y : ℝ) : Prop := 3*x + 4*y - 25 = 0

-- Define line m
def m (x y : ℝ) : Prop := x = 3 ∨ 7*x - 24*y + 75 = 0

-- Theorem statement
theorem circle_tangent_and_chord :
  (∀ x y, C x y → l x y → (x, y) = P) ∧
  (∀ x y, m x y → 
    (∃ x1 y1 x2 y2, C x1 y1 ∧ C x2 y2 ∧ m x1 y1 ∧ m x2 y2 ∧ 
     (x1 - x2)^2 + (y1 - y2)^2 = 64) ∧
    (x, y) = P) := by sorry

end NUMINAMATH_CALUDE_circle_tangent_and_chord_l2921_292182


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l2921_292149

theorem concentric_circles_radii_difference
  (r R : ℝ)
  (h_positive : r > 0)
  (h_ratio : π * R^2 / (π * r^2) = 4) :
  R - r = r :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l2921_292149


namespace NUMINAMATH_CALUDE_trisection_dot_product_l2921_292135

/-- Given three points A, B, C in 2D space, and E, F as trisection points of BC,
    prove that the dot product of vectors AE and AF is 3. -/
theorem trisection_dot_product (A B C E F : ℝ × ℝ) : 
  A = (1, 2) →
  B = (2, -1) →
  C = (2, 2) →
  E = B + (1/3 : ℝ) • (C - B) →
  F = B + (2/3 : ℝ) • (C - B) →
  (E.1 - A.1) * (F.1 - A.1) + (E.2 - A.2) * (F.2 - A.2) = 3 := by
  sorry

#check trisection_dot_product

end NUMINAMATH_CALUDE_trisection_dot_product_l2921_292135


namespace NUMINAMATH_CALUDE_opposite_sign_fractions_l2921_292183

theorem opposite_sign_fractions (x : ℚ) : 
  x = 7/5 → ((x - 1) / 2) * ((x - 2) / 3) < 0 := by sorry

end NUMINAMATH_CALUDE_opposite_sign_fractions_l2921_292183


namespace NUMINAMATH_CALUDE_ten_dollar_bill_count_l2921_292199

/-- Represents the number of bills of a certain denomination in a wallet. -/
structure BillCount where
  fives : Nat
  tens : Nat
  twenties : Nat

/-- Calculates the total amount in the wallet given the bill counts. -/
def totalAmount (bills : BillCount) : Nat :=
  5 * bills.fives + 10 * bills.tens + 20 * bills.twenties

/-- Theorem stating that given the conditions, there are 2 $10 bills in the wallet. -/
theorem ten_dollar_bill_count : ∃ (bills : BillCount), 
  bills.fives = 4 ∧ 
  bills.twenties = 3 ∧ 
  totalAmount bills = 100 ∧ 
  bills.tens = 2 := by
  sorry

end NUMINAMATH_CALUDE_ten_dollar_bill_count_l2921_292199


namespace NUMINAMATH_CALUDE_cook_sane_cheshire_cat_insane_l2921_292121

/-- Represents the sanity status of an individual -/
inductive Sanity
| Sane
| Insane

/-- Represents the characters in the problem -/
inductive Character
| Cook
| CheshireCat

/-- The cook's assertion about the sanity of the characters -/
def cooksAssertion (sanityStatus : Character → Sanity) : Prop :=
  sanityStatus Character.Cook = Sanity.Insane ∨ sanityStatus Character.CheshireCat = Sanity.Insane

/-- The main theorem to prove -/
theorem cook_sane_cheshire_cat_insane :
  ∃ (sanityStatus : Character → Sanity),
    cooksAssertion sanityStatus ∧
    sanityStatus Character.Cook = Sanity.Sane ∧
    sanityStatus Character.CheshireCat = Sanity.Insane :=
sorry

end NUMINAMATH_CALUDE_cook_sane_cheshire_cat_insane_l2921_292121


namespace NUMINAMATH_CALUDE_seashell_sum_total_seashells_l2921_292177

theorem seashell_sum : Int → Int → Int → Int
  | sam, joan, alex => sam + joan + alex

theorem total_seashells (sam joan alex : Int) 
  (h1 : sam = 35) (h2 : joan = 18) (h3 : alex = 27) : 
  seashell_sum sam joan alex = 80 := by
  sorry

end NUMINAMATH_CALUDE_seashell_sum_total_seashells_l2921_292177


namespace NUMINAMATH_CALUDE_min_correct_answers_for_score_l2921_292169

/-- Given a math test with the following conditions:
  * There are 16 total questions
  * 6 points are awarded for each correct answer
  * 2 points are deducted for each wrong answer
  * No points are deducted for unanswered questions
  * The student did not answer one question
  * The goal is to score more than 60 points

  This theorem proves that the minimum number of correct answers needed is 12. -/
theorem min_correct_answers_for_score (total_questions : ℕ) (correct_points : ℕ) (wrong_points : ℕ) 
  (unanswered : ℕ) (target_score : ℕ) : 
  total_questions = 16 → 
  correct_points = 6 → 
  wrong_points = 2 → 
  unanswered = 1 → 
  target_score = 60 → 
  ∃ (min_correct : ℕ), 
    (∀ (x : ℕ), x ≥ min_correct → 
      x * correct_points - (total_questions - unanswered - x) * wrong_points > target_score) ∧ 
    (∀ (y : ℕ), y < min_correct → 
      y * correct_points - (total_questions - unanswered - y) * wrong_points ≤ target_score) ∧
    min_correct = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_correct_answers_for_score_l2921_292169


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l2921_292186

/-- The number of ways to arrange 15 letters (4 D's, 6 E's, and 5 F's) with specific constraints -/
def letterArrangements : ℕ :=
  Finset.sum (Finset.range 5) (fun j =>
    Nat.choose 4 j * Nat.choose 6 (4 - j) * Nat.choose 5 j)

/-- Theorem stating that the number of valid arrangements is equal to the sum formula -/
theorem valid_arrangements_count :
  letterArrangements =
    Finset.sum (Finset.range 5) (fun j =>
      Nat.choose 4 j * Nat.choose 6 (4 - j) * Nat.choose 5 j) :=
by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l2921_292186


namespace NUMINAMATH_CALUDE_quadratic_behavior_l2921_292175

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 6*x - 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -2*x + 6

-- Theorem statement
theorem quadratic_behavior (x : ℝ) : x > 5 → f x < 0 ∧ f' x < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_behavior_l2921_292175


namespace NUMINAMATH_CALUDE_tetrahedron_inequality_l2921_292109

theorem tetrahedron_inequality 
  (h₁ h₂ h₃ h₄ x₁ x₂ x₃ x₄ : ℝ) 
  (h_nonneg : h₁ ≥ 0 ∧ h₂ ≥ 0 ∧ h₃ ≥ 0 ∧ h₄ ≥ 0)
  (x_nonneg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0)
  (h_tetrahedron : ∃ (S₁ S₂ S₃ S₄ : ℝ), 
    S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0 ∧
    S₁ * h₁ = S₁ * x₁ ∧ 
    S₂ * h₂ = S₂ * x₂ ∧ 
    S₃ * h₃ = S₃ * x₃ ∧ 
    S₄ * h₄ = S₄ * x₄) :
  Real.sqrt (h₁ + h₂ + h₃ + h₄) ≥ Real.sqrt x₁ + Real.sqrt x₂ + Real.sqrt x₃ + Real.sqrt x₄ := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_inequality_l2921_292109


namespace NUMINAMATH_CALUDE_spiral_grid_second_row_sum_l2921_292173

/-- Represents a position in the grid -/
structure Position :=
  (x : Fin 15)
  (y : Fin 15)

/-- Represents the spiral grid -/
def SpiralGrid := Fin 15 → Fin 15 → Nat

/-- Creates a spiral grid according to the problem description -/
def createSpiralGrid : SpiralGrid :=
  sorry

/-- Returns the center position of the grid -/
def centerPosition : Position :=
  ⟨7, 7⟩

/-- Checks if a given position is in the second row from the top -/
def isSecondRow (pos : Position) : Prop :=
  pos.y = 1

/-- Returns the maximum value in the second row -/
def maxSecondRow (grid : SpiralGrid) : Nat :=
  sorry

/-- Returns the minimum value in the second row -/
def minSecondRow (grid : SpiralGrid) : Nat :=
  sorry

theorem spiral_grid_second_row_sum :
  let grid := createSpiralGrid
  maxSecondRow grid + minSecondRow grid = 367 :=
sorry

end NUMINAMATH_CALUDE_spiral_grid_second_row_sum_l2921_292173


namespace NUMINAMATH_CALUDE_prime_pairs_divisibility_l2921_292104

theorem prime_pairs_divisibility (p q : ℕ) : 
  p.Prime ∧ q.Prime ∧ p < 2023 ∧ q < 2023 ∧ 
  (p ∣ q^2 + 8) ∧ (q ∣ p^2 + 8) → 
  ((p = 2 ∧ q = 2) ∨ (p = 17 ∧ q = 3) ∨ (p = 11 ∧ q = 5)) := by
sorry

end NUMINAMATH_CALUDE_prime_pairs_divisibility_l2921_292104


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l2921_292108

/-- The volume of a tetrahedron OABC where:
  - Triangle ABC has sides of length 7, 8, and 9
  - A is on the positive x-axis, B on the positive y-axis, and C on the positive z-axis
  - O is the origin (0, 0, 0)
-/
theorem tetrahedron_volume : ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (a^2 + b^2 : ℝ) = 49 ∧
  (b^2 + c^2 : ℝ) = 64 ∧
  (c^2 + a^2 : ℝ) = 81 ∧
  (1/6 : ℝ) * a * b * c = 8 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l2921_292108


namespace NUMINAMATH_CALUDE_newspaper_profit_bounds_l2921_292152

/-- Profit function for newspaper sales --/
def profit (x : ℝ) : ℝ := 0.95 * x - 90

/-- Domain of the profit function --/
def valid_sales (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 200

theorem newspaper_profit_bounds :
  ∀ x : ℝ, valid_sales x →
    profit x ≤ 100 ∧
    profit x ≥ -90 ∧
    (∃ x₁ x₂ : ℝ, valid_sales x₁ ∧ valid_sales x₂ ∧ profit x₁ = 100 ∧ profit x₂ = -90) :=
by sorry

end NUMINAMATH_CALUDE_newspaper_profit_bounds_l2921_292152


namespace NUMINAMATH_CALUDE_vacuum_time_solution_l2921_292125

def chores_problem (vacuum_time : ℝ) : Prop :=
  let other_chores_time := 3 * vacuum_time
  vacuum_time + other_chores_time = 12

theorem vacuum_time_solution :
  ∃ (t : ℝ), chores_problem t ∧ t = 3 :=
sorry

end NUMINAMATH_CALUDE_vacuum_time_solution_l2921_292125


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2921_292157

theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a = 6 → b = 3 → c = 3 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
  (b = c) →  -- Isosceles condition
  a + b + c = 15 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2921_292157


namespace NUMINAMATH_CALUDE_davids_english_marks_l2921_292120

/-- Represents the marks obtained in each subject --/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculate the average of marks --/
def average (m : Marks) : ℚ :=
  (m.english + m.mathematics + m.physics + m.chemistry + m.biology) / 5

theorem davids_english_marks :
  ∃ m : Marks,
    m.mathematics = 85 ∧
    m.physics = 82 ∧
    m.chemistry = 87 ∧
    m.biology = 85 ∧
    average m = 85 ∧
    m.english = 86 := by
  sorry

#check davids_english_marks

end NUMINAMATH_CALUDE_davids_english_marks_l2921_292120


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2921_292129

theorem regular_polygon_sides (interior_angle : ℝ) (n : ℕ) :
  interior_angle = 144 →
  (n : ℝ) * (180 - interior_angle) = 360 →
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2921_292129


namespace NUMINAMATH_CALUDE_chocolate_distribution_l2921_292159

theorem chocolate_distribution (total_chocolate : ℚ) (piles : ℕ) (friends : ℕ) : 
  total_chocolate = 60 / 7 →
  piles = 5 →
  friends = 3 →
  (total_chocolate / piles) * (piles - 1) / friends = 16 / 7 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l2921_292159


namespace NUMINAMATH_CALUDE_quadratic_has_two_real_roots_quadratic_roots_difference_l2921_292128

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*m*x + 3*m^2

-- Theorem 1: The quadratic equation always has two real roots
theorem quadratic_has_two_real_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 :=
sorry

-- Theorem 2: If m > 0 and the difference between roots is 2, then m = 1
theorem quadratic_roots_difference (m : ℝ) :
  m > 0 →
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ x₁ - x₂ = 2) →
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_two_real_roots_quadratic_roots_difference_l2921_292128


namespace NUMINAMATH_CALUDE_f_properties_l2921_292190

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.sin x)^2 + Real.sin x * Real.cos x - Real.sqrt 3 / 2

theorem f_properties :
  let π := Real.pi
  ∃ (T : ℝ),
    (∀ x, f (x + T) = f x) ∧ 
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧
    T = π ∧
    (∀ x ∈ Set.Icc (5*π/12) (11*π/12), ∀ y ∈ Set.Icc (5*π/12) (11*π/12), x < y → f y < f x) ∧
    (∀ A b c : ℝ, 
      f (A/2 + π/4) = 1 → 
      2 = Real.sqrt (b^2 + c^2 - 2*b*c*Real.cos A) → 
      2 < b + c ∧ b + c ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2921_292190


namespace NUMINAMATH_CALUDE_statues_painted_l2921_292123

theorem statues_painted (total_paint : ℚ) (paint_per_statue : ℚ) :
  total_paint = 7/16 →
  paint_per_statue = 1/16 →
  (total_paint / paint_per_statue : ℚ) = 7 :=
by sorry

end NUMINAMATH_CALUDE_statues_painted_l2921_292123


namespace NUMINAMATH_CALUDE_morio_age_at_michiko_birth_l2921_292101

/-- Proves that Morio's age when Michiko was born is 38 years old -/
theorem morio_age_at_michiko_birth 
  (teresa_current_age : ℕ) 
  (morio_current_age : ℕ) 
  (teresa_age_at_birth : ℕ) 
  (h1 : teresa_current_age = 59) 
  (h2 : morio_current_age = 71) 
  (h3 : teresa_age_at_birth = 26) :
  morio_current_age - (teresa_current_age - teresa_age_at_birth) = 38 :=
by
  sorry


end NUMINAMATH_CALUDE_morio_age_at_michiko_birth_l2921_292101


namespace NUMINAMATH_CALUDE_angle_bisector_product_theorem_l2921_292145

/-- Given a triangle with sides a, b, c, internal angle bisectors fa, fb, fc, and area T,
    this theorem states that the product of the angle bisectors divided by the product of the sides
    is equal to four times the area multiplied by the sum of the sides,
    divided by the product of the pairwise sums of the sides. -/
theorem angle_bisector_product_theorem
  (a b c fa fb fc T : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b)
  (h_bisectors : fa > 0 ∧ fb > 0 ∧ fc > 0)
  (h_area : T > 0) :
  (fa * fb * fc) / (a * b * c) = 4 * T * (a + b + c) / ((a + b) * (b + c) * (a + c)) :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_product_theorem_l2921_292145


namespace NUMINAMATH_CALUDE_function_characterization_l2921_292161

/-- A function from natural numbers to natural numbers. -/
def NatFunction := ℕ → ℕ

/-- The property that f(3x + 2y) = f(x)f(y) for all x, y ∈ ℕ. -/
def SatisfiesProperty (f : NatFunction) : Prop :=
  ∀ x y : ℕ, f (3 * x + 2 * y) = f x * f y

/-- The constant zero function. -/
def ZeroFunction : NatFunction := λ _ => 0

/-- The constant one function. -/
def OneFunction : NatFunction := λ _ => 1

/-- The function that is 1 at 0 and 0 elsewhere. -/
def ZeroOneFunction : NatFunction := λ n => if n = 0 then 1 else 0

/-- The main theorem stating that any function satisfying the property
    must be one of the three specified functions. -/
theorem function_characterization (f : NatFunction) 
  (h : SatisfiesProperty f) : 
  f = ZeroFunction ∨ f = OneFunction ∨ f = ZeroOneFunction :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l2921_292161


namespace NUMINAMATH_CALUDE_preimage_of_one_two_l2921_292158

/-- The mapping f from R² to R² -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

/-- The theorem stating that (3/2, -1/2) is the preimage of (1, 2) under f -/
theorem preimage_of_one_two :
  f (3/2, -1/2) = (1, 2) := by sorry

end NUMINAMATH_CALUDE_preimage_of_one_two_l2921_292158


namespace NUMINAMATH_CALUDE_divisibility_problem_l2921_292117

theorem divisibility_problem (a b : Nat) (n : Nat) : 
  a ≤ 9 → b ≤ 9 → a * b ≤ 15 → (110 * a + b) % n = 0 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2921_292117


namespace NUMINAMATH_CALUDE_max_distance_MN_l2921_292100

-- Define the curves
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

def C₂ (x y : ℝ) : Prop := ∃ φ : ℝ, x = 2 * Real.cos φ ∧ y = Real.sin φ

def C₃ (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 1

-- Define the transformation
def transformation (x y : ℝ) : Prop := x^2 = 2*x ∧ y^2 = y

-- Define the tangent point condition
def is_tangent_point (M N : ℝ × ℝ) : Prop :=
  C₂ M.1 M.2 ∧ C₃ N.1 N.2 ∧
  ∃ t : ℝ, (N.1 - M.1)^2 + (N.2 - M.2)^2 = t^2 ∧
           ∀ P : ℝ × ℝ, C₃ P.1 P.2 → (P.1 - M.1)^2 + (P.2 - M.2)^2 ≥ t^2

-- Theorem statement
theorem max_distance_MN :
  ∀ M N : ℝ × ℝ, is_tangent_point M N →
  (N.1 - M.1)^2 + (N.2 - M.2)^2 ≤ 15 :=
sorry

end NUMINAMATH_CALUDE_max_distance_MN_l2921_292100


namespace NUMINAMATH_CALUDE_number_of_friends_prove_number_of_friends_l2921_292153

theorem number_of_friends (original_bill : ℝ) (discount_percent : ℝ) (individual_payment : ℝ) : ℝ :=
  let discounted_bill := original_bill * (1 - discount_percent / 100)
  discounted_bill / individual_payment

theorem prove_number_of_friends :
  number_of_friends 100 6 18.8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_friends_prove_number_of_friends_l2921_292153


namespace NUMINAMATH_CALUDE_simplify_expression_l2921_292102

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 25) = 152 * x + 45 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2921_292102


namespace NUMINAMATH_CALUDE_tea_store_profit_l2921_292147

theorem tea_store_profit (m n : ℝ) (h : m > n) : 
  let cost := 40 * m + 60 * n
  let revenue := 50 * (m + n)
  revenue - cost > 0 := by
sorry

end NUMINAMATH_CALUDE_tea_store_profit_l2921_292147


namespace NUMINAMATH_CALUDE_folk_song_competition_probability_l2921_292166

theorem folk_song_competition_probability : 
  ∀ (n m k : ℕ),
  n = 6 →  -- number of provinces
  m = 2 →  -- number of singers per province
  k = 4 →  -- number of winners selected
  (Nat.choose n 1 * Nat.choose (n - 1) 2 * Nat.choose m 1 * Nat.choose m 1) / 
  (Nat.choose (n * m) k) = 16 / 33 := by
  sorry

end NUMINAMATH_CALUDE_folk_song_competition_probability_l2921_292166


namespace NUMINAMATH_CALUDE_staircase_steps_l2921_292167

/-- The number of toothpicks used in a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ := 2 * (n * (n + 1) * (2 * n + 1)) / 3

/-- Theorem stating that a staircase with 630 toothpicks has 9 steps -/
theorem staircase_steps : ∃ (n : ℕ), toothpicks n = 630 ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_staircase_steps_l2921_292167


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l2921_292131

theorem camping_trip_percentage 
  (total_students : ℕ) 
  (students_more_than_100 : ℕ) 
  (h1 : students_more_than_100 = (15 * total_students) / 100)
  (h2 : (75 * (students_more_than_100 * 100 / 25)) / 100 + students_more_than_100 = (60 * total_students) / 100) :
  (students_more_than_100 * 100 / 25) * 100 / total_students = 60 :=
sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l2921_292131


namespace NUMINAMATH_CALUDE_multiply_72518_by_9999_l2921_292179

theorem multiply_72518_by_9999 : 72518 * 9999 = 725107482 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72518_by_9999_l2921_292179


namespace NUMINAMATH_CALUDE_combination_permutation_inequality_l2921_292103

theorem combination_permutation_inequality (n : ℕ+) : 
  2 * Nat.choose n 3 ≤ n * (n - 1) ↔ 3 ≤ n ∧ n ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_combination_permutation_inequality_l2921_292103


namespace NUMINAMATH_CALUDE_tourist_survival_l2921_292122

theorem tourist_survival (initial_tourists : ℕ) (eaten : ℕ) (poison_fraction : ℚ) (recovery_fraction : ℚ) : initial_tourists = 30 → eaten = 2 → poison_fraction = 1/2 → recovery_fraction = 1/7 → 
  let remaining_after_eaten := initial_tourists - eaten
  let poisoned := (remaining_after_eaten : ℚ) * poison_fraction
  let recovered := poisoned * recovery_fraction
  (remaining_after_eaten : ℚ) - poisoned + recovered = 16 := by
  sorry

end NUMINAMATH_CALUDE_tourist_survival_l2921_292122


namespace NUMINAMATH_CALUDE_five_digit_cube_root_l2921_292126

theorem five_digit_cube_root (n : ℕ) : 
  (10000 ≤ n ∧ n < 100000) →  -- n is a five-digit number
  (n % 10 = 3) →              -- n ends in 3
  (∃ k : ℕ, k^3 = n) →        -- n has an integer cube root
  (n = 19683 ∨ n = 50653) :=  -- n is either 19683 or 50653
by sorry

end NUMINAMATH_CALUDE_five_digit_cube_root_l2921_292126


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2921_292184

theorem polynomial_expansion (z : ℝ) : 
  (3*z^2 + 4*z - 5) * (4*z^3 - 3*z + 2) = 
  12*z^5 + 16*z^4 - 29*z^3 - 6*z^2 + 23*z - 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2921_292184


namespace NUMINAMATH_CALUDE_sin_cos_15_product_l2921_292170

theorem sin_cos_15_product : 
  (Real.sin (15 * π / 180) + Real.cos (15 * π / 180)) * 
  (Real.sin (15 * π / 180) - Real.cos (15 * π / 180)) = 
  -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_15_product_l2921_292170


namespace NUMINAMATH_CALUDE_circle_radius_order_l2921_292196

theorem circle_radius_order (r_A : ℝ) (c_B : ℝ) (a_C : ℝ) :
  r_A = 3 * Real.pi →
  c_B = 10 * Real.pi →
  a_C = 16 * Real.pi →
  ∃ (r_B r_C : ℝ),
    c_B = 2 * Real.pi * r_B ∧
    a_C = Real.pi * r_C^2 ∧
    r_C < r_B ∧ r_B < r_A :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_order_l2921_292196


namespace NUMINAMATH_CALUDE_three_semi_fixed_points_l2921_292142

/-- A function f has a semi-fixed point at x₀ if f(x₀) = -x₀ -/
def has_semi_fixed_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = -x₀

/-- The function f(x) = ax^3 - 3x^2 - x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^3 - 3 * x^2 - x + 1

/-- The theorem stating the condition for f to have exactly three semi-fixed points -/
theorem three_semi_fixed_points (a : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    has_semi_fixed_point (f a) x₁ ∧
    has_semi_fixed_point (f a) x₂ ∧
    has_semi_fixed_point (f a) x₃ ∧
    (∀ x : ℝ, has_semi_fixed_point (f a) x → x = x₁ ∨ x = x₂ ∨ x = x₃)) ↔
  (a ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioo 0 2) :=
sorry

end NUMINAMATH_CALUDE_three_semi_fixed_points_l2921_292142


namespace NUMINAMATH_CALUDE_batsman_average_increase_l2921_292181

/-- Represents a batsman's cricket statistics -/
structure Batsman where
  innings : ℕ
  totalScore : ℕ
  notOutCount : ℕ

/-- Calculate the batting average -/
def battingAverage (b : Batsman) : ℚ :=
  b.totalScore / (b.innings - b.notOutCount)

/-- The increase in average after a new innings -/
def averageIncrease (before after : Batsman) : ℚ :=
  battingAverage after - battingAverage before

theorem batsman_average_increase :
  ∀ (before : Batsman),
    before.innings = 19 →
    before.notOutCount = 0 →
    let after : Batsman :=
      { innings := 20
      , totalScore := before.totalScore + 90
      , notOutCount := 0
      }
    battingAverage after = 52 →
    averageIncrease before after = 2 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l2921_292181


namespace NUMINAMATH_CALUDE_xiaoli_estimate_l2921_292146

theorem xiaoli_estimate (x y z : ℝ) (hxy : x > y) (hy : y > 0) (hz : z > 0) :
  (x + z) + (y - z) = x + y := by
  sorry

end NUMINAMATH_CALUDE_xiaoli_estimate_l2921_292146


namespace NUMINAMATH_CALUDE_pyramid_volume_in_cube_l2921_292189

structure Cube where
  edge : ℝ
  volume : ℝ
  volume_eq : volume = edge^3

structure Pyramid where
  base_area : ℝ
  height : ℝ
  volume : ℝ
  volume_eq : volume = (1/3) * base_area * height

theorem pyramid_volume_in_cube (c : Cube) (p : Pyramid) :
  c.volume = 8 →
  p.base_area = 2 →
  p.height = c.edge →
  p.volume = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_in_cube_l2921_292189


namespace NUMINAMATH_CALUDE_eggs_per_unit_l2921_292127

/-- Given that Joan bought 6 units of eggs and 72 eggs in total, 
    prove that the number of eggs in one unit is 12. -/
theorem eggs_per_unit (units : ℕ) (total_eggs : ℕ) 
  (h1 : units = 6) (h2 : total_eggs = 72) : 
  total_eggs / units = 12 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_unit_l2921_292127


namespace NUMINAMATH_CALUDE_savings_difference_l2921_292193

def savings_problem : Prop :=
  let dick_1989 := 5000
  let jane_1989 := 5000
  let dick_1990 := dick_1989 * 1.10
  let jane_1990 := jane_1989 * 0.95
  let dick_1991 := dick_1990 * 1.07
  let jane_1991 := jane_1990 * 1.08
  let dick_1992 := dick_1991 * 0.88
  let jane_1992 := jane_1991 * 1.15
  let dick_total := dick_1989 + dick_1990 + dick_1991 + dick_1992
  let jane_total := jane_1989 + jane_1990 + jane_1991 + jane_1992
  dick_total - jane_total = 784.30

theorem savings_difference : savings_problem := by
  sorry

end NUMINAMATH_CALUDE_savings_difference_l2921_292193


namespace NUMINAMATH_CALUDE_roots_and_d_values_l2921_292148

-- Define the polynomial p(x)
def p (c d x : ℝ) : ℝ := x^3 + c*x + d

-- Define the polynomial q(x)
def q (c d x : ℝ) : ℝ := x^3 + c*x + d - 270

-- Theorem statement
theorem roots_and_d_values (u v c d : ℝ) : 
  (p c d u = 0 ∧ p c d v = 0) ∧ 
  (q c d (u+3) = 0 ∧ q c d (v-2) = 0) →
  d = -6 ∨ d = -120 := by
sorry

end NUMINAMATH_CALUDE_roots_and_d_values_l2921_292148


namespace NUMINAMATH_CALUDE_meaningful_expression_l2921_292105

theorem meaningful_expression (x : ℝ) : 
  (x = 4 ∨ x = 8 ∨ x = 12 ∨ x = 16) →
  (10 - x ≥ 0 ∧ x - 4 ≠ 0) ↔ x = 8 := by
sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2921_292105


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l2921_292154

theorem opposite_of_negative_five : -(-5) = 5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l2921_292154


namespace NUMINAMATH_CALUDE_interval_relationship_l2921_292164

theorem interval_relationship : 
  (∀ x, 2 < x ∧ x < 3 → 1 < x ∧ x < 5) ∧ 
  ¬(∀ x, 1 < x ∧ x < 5 → 2 < x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_interval_relationship_l2921_292164


namespace NUMINAMATH_CALUDE_new_job_wage_is_15_l2921_292139

/-- Represents the wage scenario for Maisy's job options -/
structure WageScenario where
  current_hours : ℕ
  current_wage : ℕ
  new_hours : ℕ
  new_bonus : ℕ
  earnings_difference : ℕ

/-- Calculates the wage per hour for the new job -/
def new_job_wage (scenario : WageScenario) : ℕ :=
  (scenario.current_hours * scenario.current_wage + scenario.earnings_difference - scenario.new_bonus) / scenario.new_hours

/-- Theorem stating that given the specified conditions, the new job wage is $15 per hour -/
theorem new_job_wage_is_15 (scenario : WageScenario) 
  (h1 : scenario.current_hours = 8)
  (h2 : scenario.current_wage = 10)
  (h3 : scenario.new_hours = 4)
  (h4 : scenario.new_bonus = 35)
  (h5 : scenario.earnings_difference = 15) :
  new_job_wage scenario = 15 := by
  sorry

#eval new_job_wage { current_hours := 8, current_wage := 10, new_hours := 4, new_bonus := 35, earnings_difference := 15 }

end NUMINAMATH_CALUDE_new_job_wage_is_15_l2921_292139


namespace NUMINAMATH_CALUDE_function_relation_implies_a_half_l2921_292110

/-- Given two functions f and g defined on ℝ satisfying certain conditions, prove that a = 1/2 -/
theorem function_relation_implies_a_half :
  ∀ (f g : ℝ → ℝ) (a : ℝ),
    (∀ x, f x = a^x * g x) →
    (a > 0) →
    (a ≠ 1) →
    (∀ x, g x ≠ 0 → f x * (deriv g x) > (deriv f x) * g x) →
    (f 1 / g 1 + f (-1) / g (-1) = 5/2) →
    a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_relation_implies_a_half_l2921_292110


namespace NUMINAMATH_CALUDE_candies_given_to_stephanie_l2921_292111

theorem candies_given_to_stephanie (initial_candies remaining_candies : ℕ) 
  (h1 : initial_candies = 95)
  (h2 : remaining_candies = 92) :
  initial_candies - remaining_candies = 3 := by
  sorry

end NUMINAMATH_CALUDE_candies_given_to_stephanie_l2921_292111


namespace NUMINAMATH_CALUDE_macy_running_goal_l2921_292163

/-- Calculates the remaining miles to reach a weekly running goal -/
def remaining_miles (weekly_goal : ℕ) (daily_miles : ℕ) (days_run : ℕ) : ℕ :=
  weekly_goal - (daily_miles * days_run)

/-- Proves that given a weekly goal of 24 miles, running 3 miles per day for 6 days,
    the remaining miles to reach the goal is 6 miles -/
theorem macy_running_goal :
  remaining_miles 24 3 6 = 6 := by
sorry

end NUMINAMATH_CALUDE_macy_running_goal_l2921_292163


namespace NUMINAMATH_CALUDE_eighteenth_roots_of_unity_ninth_power_real_l2921_292151

theorem eighteenth_roots_of_unity_ninth_power_real : 
  ∀ z : ℂ, z^18 = 1 → ∃ r : ℝ, z^9 = r :=
by sorry

end NUMINAMATH_CALUDE_eighteenth_roots_of_unity_ninth_power_real_l2921_292151


namespace NUMINAMATH_CALUDE_four_digit_number_with_specific_remainders_l2921_292124

theorem four_digit_number_with_specific_remainders :
  ∃! N : ℕ, 
    N % 131 = 112 ∧
    N % 132 = 98 ∧
    1000 ≤ N ∧ N ≤ 9999 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_with_specific_remainders_l2921_292124


namespace NUMINAMATH_CALUDE_total_books_l2921_292168

-- Define the number of books for each person
def harry_books : ℕ := 50
def flora_books : ℕ := 2 * harry_books
def gary_books : ℕ := harry_books / 2

-- Theorem to prove
theorem total_books : harry_books + flora_books + gary_books = 175 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l2921_292168


namespace NUMINAMATH_CALUDE_third_nonagon_side_length_l2921_292141

/-- Represents a regular nonagon with a given side length -/
structure RegularNonagon where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- The area of a regular nonagon given its side length -/
def nonagonArea (n : RegularNonagon) : ℝ := n.sideLength^2

/-- Theorem: Given three concentric regular nonagons with parallel sides,
    where two have side lengths of 8 and 56, and the third divides the area
    between them in a 1:7 ratio (measured from the smaller nonagon),
    the side length of the third nonagon is 8√7. -/
theorem third_nonagon_side_length
  (n1 n2 n3 : RegularNonagon)
  (h1 : n1.sideLength = 8)
  (h2 : n2.sideLength = 56)
  (h3 : (nonagonArea n3 - nonagonArea n1) / (nonagonArea n2 - nonagonArea n3) = 1 / 7) :
  n3.sideLength = 8 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_third_nonagon_side_length_l2921_292141


namespace NUMINAMATH_CALUDE_number_of_arrangements_l2921_292155

/-- Represents the number of students of each gender -/
def num_students : ℕ := 3

/-- Represents the total number of students -/
def total_students : ℕ := 2 * num_students

/-- Represents the number of positions where male student A can stand -/
def positions_for_A : ℕ := total_students - 2

/-- Represents the number of ways to arrange the two adjacent female students -/
def adjacent_female_arrangements : ℕ := 2

/-- Represents the number of ways to arrange the remaining students -/
def remaining_arrangements : ℕ := 3 * 2

/-- The theorem stating the number of different arrangements -/
theorem number_of_arrangements :
  positions_for_A * adjacent_female_arrangements * remaining_arrangements * Nat.factorial 2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l2921_292155


namespace NUMINAMATH_CALUDE_unique_solution_l2921_292132

theorem unique_solution (x y z : ℝ) : 
  x + 3 * y = 33 ∧ 
  y = 10 ∧ 
  2 * x - y + z = 15 → 
  x = 3 ∧ y = 10 ∧ z = 19 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l2921_292132


namespace NUMINAMATH_CALUDE_system_solution_l2921_292150

theorem system_solution :
  let x : ℝ := (133 - Real.sqrt 73) / 48
  let y : ℝ := (-1 + Real.sqrt 73) / 12
  2 * x - 3 * y^2 = 4 ∧ 4 * x + y = 11 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2921_292150


namespace NUMINAMATH_CALUDE_x_value_l2921_292143

theorem x_value : ∃ x : ℝ, x = 88 * (1 + 0.5) ∧ x = 132 := by sorry

end NUMINAMATH_CALUDE_x_value_l2921_292143


namespace NUMINAMATH_CALUDE_angle_complement_theorem_l2921_292165

theorem angle_complement_theorem (x : ℝ) : 
  (90 - x) = (3 * x + 10) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_theorem_l2921_292165


namespace NUMINAMATH_CALUDE_inscribed_circles_chord_length_l2921_292115

/-- Given two circles, one inscribed in an angle α with radius r and another of radius R 
    touching one side of the angle at the same point as the first circle and intersecting 
    the other side at points A and B, the length of AB can be calculated. -/
theorem inscribed_circles_chord_length (α r R : ℝ) (h_pos_r : r > 0) (h_pos_R : R > 0) :
  ∃ (AB : ℝ), AB = 4 * Real.cos (α / 2) * Real.sqrt ((R - r) * (R * Real.sin (α / 2)^2 + r * Real.cos (α / 2)^2)) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circles_chord_length_l2921_292115


namespace NUMINAMATH_CALUDE_inequality_proof_l2921_292192

theorem inequality_proof (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) ∧
  ((1 + a / b) ^ n + (1 + b / a) ^ n = 2^(n + 1) ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2921_292192


namespace NUMINAMATH_CALUDE_product_sum_multiple_l2921_292198

theorem product_sum_multiple (a b m : ℤ) : 
  b = 7 → 
  b - a = 2 → 
  a * b = m * (a + b) + 11 → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_product_sum_multiple_l2921_292198


namespace NUMINAMATH_CALUDE_equal_numbers_from_different_sequences_l2921_292118

/-- Represents a sequence of consecutive natural numbers -/
def ConsecutiveSequence (start : ℕ) (length : ℕ) : List ℕ :=
  List.range length |>.map (· + start)

/-- Concatenates a list of natural numbers into a single number -/
def concatenateToNumber (list : List ℕ) : ℕ := sorry

theorem equal_numbers_from_different_sequences :
  ∃ (a b : ℕ) (orderA : List ℕ → List ℕ) (orderB : List ℕ → List ℕ),
    let seqA := ConsecutiveSequence a 20
    let seqB := ConsecutiveSequence b 21
    concatenateToNumber (orderA seqA) = concatenateToNumber (orderB seqB) := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_from_different_sequences_l2921_292118


namespace NUMINAMATH_CALUDE_f_neg_one_eq_one_fifteenth_l2921_292172

/-- The function f satisfying the given equation for all x -/
noncomputable def f : ℝ → ℝ := 
  fun x => ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) - 1) / (x^(2^5 - 1) - 1)

/-- Theorem stating that f(-1) = 1/15 -/
theorem f_neg_one_eq_one_fifteenth : f (-1) = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_one_eq_one_fifteenth_l2921_292172


namespace NUMINAMATH_CALUDE_two_ab_value_l2921_292116

theorem two_ab_value (a b : ℝ) 
  (h1 : a^4 + a^2*b^2 + b^4 = 900) 
  (h2 : a^2 + a*b + b^2 = 45) : 
  2*a*b = 25 := by
sorry

end NUMINAMATH_CALUDE_two_ab_value_l2921_292116


namespace NUMINAMATH_CALUDE_base_for_216_four_digits_l2921_292112

def has_exactly_four_digits (b : ℕ) (n : ℕ) : Prop :=
  b ^ 3 ≤ n ∧ n < b ^ 4

theorem base_for_216_four_digits :
  ∃! b : ℕ, b > 1 ∧ has_exactly_four_digits b 216 :=
by
  sorry

end NUMINAMATH_CALUDE_base_for_216_four_digits_l2921_292112


namespace NUMINAMATH_CALUDE_original_triangle_area_l2921_292187

theorem original_triangle_area (original_area new_area : ℝ) : 
  (∀ (side : ℝ), new_area = (5 * side)^2 / 2 → original_area = side^2 / 2) →
  new_area = 200 →
  original_area = 8 :=
by sorry

end NUMINAMATH_CALUDE_original_triangle_area_l2921_292187


namespace NUMINAMATH_CALUDE_unique_solution_for_B_l2921_292106

theorem unique_solution_for_B : ∃! B : ℕ, 
  B < 10 ∧ 
  (∃ A : ℕ, A < 10 ∧ 38 * 10 + A - (10 * B + 1) = 364) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_B_l2921_292106


namespace NUMINAMATH_CALUDE_basketball_free_throws_l2921_292194

/-- Represents the scoring of a basketball team -/
structure BasketballScore where
  two_pointers : ℕ
  three_pointers : ℕ
  free_throws : ℕ

/-- Checks if the given BasketballScore satisfies the problem conditions -/
def is_valid_score (score : BasketballScore) : Prop :=
  3 * score.three_pointers = 2 * 2 * score.two_pointers ∧
  score.free_throws = 2 * score.two_pointers - 3 ∧
  2 * score.two_pointers + 3 * score.three_pointers + score.free_throws = 73

theorem basketball_free_throws (score : BasketballScore) :
  is_valid_score score → score.free_throws = 21 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l2921_292194


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_greater_than_one_l2921_292162

theorem sum_of_reciprocals_greater_than_one 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ > 1) 
  (h₂ : a₂ > 1) 
  (h₃ : a₃ > 1) 
  (hS : a₁ + a₂ + a₃ = a₁ + a₂ + a₃) 
  (hcond₁ : a₁^2 / (a₁ - 1) > a₁ + a₂ + a₃) 
  (hcond₂ : a₂^2 / (a₂ - 1) > a₁ + a₂ + a₃) 
  (hcond₃ : a₃^2 / (a₃ - 1) > a₁ + a₂ + a₃) : 
  1 / (a₁ + a₂) + 1 / (a₂ + a₃) + 1 / (a₃ + a₁) > 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_greater_than_one_l2921_292162


namespace NUMINAMATH_CALUDE_xy_equation_solutions_l2921_292156

theorem xy_equation_solutions (x y : ℤ) : x + y = x * y ↔ (x = 2 ∧ y = 2) ∨ (x = 0 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_xy_equation_solutions_l2921_292156


namespace NUMINAMATH_CALUDE_polynomial_identity_l2921_292171

/-- The polynomial p(x) = x^2 - x + 1 -/
def p (x : ℂ) : ℂ := x^2 - x + 1

/-- α is a root of p(p(p(p(x)))) -/
def α : ℂ := sorry

theorem polynomial_identity :
  (p α - 1) * p α * p (p α) * p (p (p α)) = -1 := by sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2921_292171


namespace NUMINAMATH_CALUDE_distance_between_B_and_D_l2921_292174

theorem distance_between_B_and_D 
  (a b c d : ℝ) 
  (h1 : |2*a - 3*c| = 1) 
  (h2 : |2*b - 3*c| = 1) 
  (h3 : 2/3 * |d - a| = 1) 
  (h4 : a ≠ b) : 
  |d - b| = 1/2 ∨ |d - b| = 5/2 := by
sorry

end NUMINAMATH_CALUDE_distance_between_B_and_D_l2921_292174


namespace NUMINAMATH_CALUDE_relationship_abc_l2921_292114

theorem relationship_abc : 3^(1/5) > 0.3^2 ∧ 0.3^2 > Real.log 0.3 / Real.log 2 := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l2921_292114


namespace NUMINAMATH_CALUDE_fraction_simplification_l2921_292134

theorem fraction_simplification : 
  ((2^12)^2 - (2^10)^2) / ((2^11)^2 - (2^9)^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2921_292134


namespace NUMINAMATH_CALUDE_magic_square_constant_l2921_292113

def MagicSquare (a b c d e f g h i : ℕ) : Prop :=
  a + b + c = d + e + f ∧
  d + e + f = g + h + i ∧
  a + d + g = b + e + h ∧
  b + e + h = c + f + i ∧
  a + e + i = c + e + g

theorem magic_square_constant (a b c d e f g h i : ℕ) :
  MagicSquare a b c d e f g h i →
  a = 12 → c = 4 → d = 7 → h = 1 →
  a + b + c = 15 :=
sorry

end NUMINAMATH_CALUDE_magic_square_constant_l2921_292113


namespace NUMINAMATH_CALUDE_inverse_function_theorem_l2921_292140

noncomputable def f (x : ℝ) : ℝ := Real.log (3 * x + 1)

def f_domain (x : ℝ) : Prop := x > -1

noncomputable def g (x : ℝ) : ℝ := (Real.exp x - 1) ^ 3

theorem inverse_function_theorem (x : ℝ) (hx : f_domain x) :
  g (f x) = x ∧ f (g x) = x :=
sorry

end NUMINAMATH_CALUDE_inverse_function_theorem_l2921_292140


namespace NUMINAMATH_CALUDE_matthew_ate_six_l2921_292191

/-- The number of egg rolls eaten by Matthew, Patrick, and Alvin. -/
structure EggRolls where
  matthew : ℕ
  patrick : ℕ
  alvin : ℕ

/-- The conditions of the egg roll problem. -/
def egg_roll_conditions (e : EggRolls) : Prop :=
  e.matthew = 3 * e.patrick ∧
  e.patrick = e.alvin / 2 ∧
  e.alvin = 4

/-- The theorem stating that Matthew ate 6 egg rolls. -/
theorem matthew_ate_six (e : EggRolls) (h : egg_roll_conditions e) : e.matthew = 6 := by
  sorry

end NUMINAMATH_CALUDE_matthew_ate_six_l2921_292191


namespace NUMINAMATH_CALUDE_smallest_n_for_cube_T_l2921_292136

/-- Function that calculates (n+2)3^n for a positive integer n -/
def T (n : ℕ+) : ℕ := (n + 2) * 3^(n : ℕ)

/-- Predicate to check if a natural number is a perfect cube -/
def is_cube (m : ℕ) : Prop := ∃ k : ℕ, m = k^3

/-- Theorem stating that 1 is the smallest positive integer n for which T(n) is a perfect cube -/
theorem smallest_n_for_cube_T :
  (∃ n : ℕ+, is_cube (T n)) ∧ (∀ n : ℕ+, is_cube (T n) → 1 ≤ n) := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_cube_T_l2921_292136


namespace NUMINAMATH_CALUDE_problem_solution_l2921_292185

theorem problem_solution (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 2)  -- The absolute value of m is 2
  : (a + b) / (4 * m) + 2 * m^2 - 3 * c * d = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2921_292185


namespace NUMINAMATH_CALUDE_cones_from_twelve_cylinders_l2921_292133

/-- The number of cones that can be cast from a given number of cylinders -/
def cones_from_cylinders (num_cylinders : ℕ) : ℕ :=
  3 * num_cylinders

/-- The volume ratio between a cylinder and a cone with the same base and height -/
def cylinder_cone_volume_ratio : ℕ := 3

theorem cones_from_twelve_cylinders :
  cones_from_cylinders 12 = 36 :=
by sorry

end NUMINAMATH_CALUDE_cones_from_twelve_cylinders_l2921_292133


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l2921_292107

/-- Given a hyperbola with asymptotes y = ±(2/3)x and real axis length 12,
    its standard equation is either (x²/36) - (y²/16) = 1 or (y²/36) - (x²/16) = 1 -/
theorem hyperbola_standard_equation
  (asymptote_slope : ℝ)
  (real_axis_length : ℝ)
  (h1 : asymptote_slope = 2/3)
  (h2 : real_axis_length = 12) :
  (∃ (x y : ℝ), x^2/36 - y^2/16 = 1) ∨
  (∃ (x y : ℝ), y^2/36 - x^2/16 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l2921_292107


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l2921_292138

theorem inequality_not_always_true (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (hz : z > 0) :
  ¬ (∀ x y z : ℝ, x > 0 → y > 0 → x > y → z > 0 → |x/z - y/z| = (x-y)/z) :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l2921_292138


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_is_80_l2921_292130

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : Nat
  width : Nat
  depth : Nat

/-- Calculates the smallest number of identical cubes needed to fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : Nat :=
  let cubeSideLength := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSideLength) * (box.width / cubeSideLength) * (box.depth / cubeSideLength)

/-- Theorem stating that the smallest number of cubes to fill the given box is 80 -/
theorem smallest_number_of_cubes_is_80 :
  smallestNumberOfCubes ⟨30, 48, 12⟩ = 80 := by
  sorry

#eval smallestNumberOfCubes ⟨30, 48, 12⟩

end NUMINAMATH_CALUDE_smallest_number_of_cubes_is_80_l2921_292130


namespace NUMINAMATH_CALUDE_divisibility_by_five_l2921_292176

theorem divisibility_by_five (x y : ℕ+) (h1 : 2 * x ^ 2 - 1 = y ^ 15) (h2 : x > 1) :
  5 ∣ x.val :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l2921_292176


namespace NUMINAMATH_CALUDE_binomial_coefficient_recurrence_l2921_292144

theorem binomial_coefficient_recurrence (n r : ℕ) (h1 : n > 0) (h2 : r > 0) (h3 : n > r) :
  Nat.choose n r = Nat.choose (n - 1) r + Nat.choose (n - 1) (r - 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_recurrence_l2921_292144


namespace NUMINAMATH_CALUDE_cube_ratio_equals_27_l2921_292178

theorem cube_ratio_equals_27 : (81000 : ℚ)^3 / (27000 : ℚ)^3 = 27 := by sorry

end NUMINAMATH_CALUDE_cube_ratio_equals_27_l2921_292178


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l2921_292195

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

-- Define the interval (1, 2]
def interval_one_two : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- Theorem statement
theorem intersection_equals_interval : M ∩ N = interval_one_two := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l2921_292195


namespace NUMINAMATH_CALUDE_star_not_commutative_l2921_292188

def star (x y : ℝ) : ℝ := |x - 2*y + 3|

theorem star_not_commutative : ∃ x y : ℝ, star x y ≠ star y x := by sorry

end NUMINAMATH_CALUDE_star_not_commutative_l2921_292188


namespace NUMINAMATH_CALUDE_bryan_mineral_samples_l2921_292180

/-- The number of mineral samples Bryan has left after rearrangement -/
def samples_left (initial_samples_per_shelf : ℕ) (num_shelves : ℕ) (removed_per_shelf : ℕ) : ℕ :=
  (initial_samples_per_shelf - removed_per_shelf) * num_shelves

/-- Theorem stating the number of samples left after Bryan's rearrangement -/
theorem bryan_mineral_samples :
  samples_left 128 13 2 = 1638 := by
  sorry

end NUMINAMATH_CALUDE_bryan_mineral_samples_l2921_292180


namespace NUMINAMATH_CALUDE_tank_capacity_tank_capacity_1440_l2921_292137

/-- Given a tank with a leak and an inlet pipe, prove its capacity. -/
theorem tank_capacity (leak_time : ℝ) (inlet_rate : ℝ) (combined_time : ℝ) : ℝ :=
  let leak_rate := 1 / leak_time
  let inlet_rate_hourly := inlet_rate * 60
  let combined_rate := 1 / combined_time
  let capacity := (inlet_rate_hourly - combined_rate) / (leak_rate - combined_rate)
  by
    -- Assumptions
    have h1 : leak_time = 6 := by sorry
    have h2 : inlet_rate = 6 := by sorry
    have h3 : combined_time = 12 := by sorry
    
    -- Proof
    sorry

/-- The main theorem stating the tank's capacity. -/
theorem tank_capacity_1440 : tank_capacity 6 6 12 = 1440 := by sorry

end NUMINAMATH_CALUDE_tank_capacity_tank_capacity_1440_l2921_292137
