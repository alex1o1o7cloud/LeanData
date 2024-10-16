import Mathlib

namespace NUMINAMATH_CALUDE_nine_digit_prime_square_product_l162_16237

/-- Represents a nine-digit number of the form a₁a₂a₃b₁b₂b₃a₁a₂a₃ --/
def NineDigitNumber (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) : ℕ :=
  a₁ * 100000000 + a₂ * 10000000 + a₃ * 1000000 + 
  b₁ * 100000 + b₂ * 10000 + b₃ * 1000 + 
  a₁ * 100 + a₂ * 10 + a₃

/-- Condition: ⎯⎯⎯⎯⎯b₁b₂b₃ = 2 * ⎯⎯⎯⎯⎯(a₁a₂a₃) --/
def MiddleIsDoubleFirst (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) : Prop :=
  b₁ * 100 + b₂ * 10 + b₃ = 2 * (a₁ * 100 + a₂ * 10 + a₃)

/-- The number is the product of the squares of four different prime numbers --/
def IsProductOfFourPrimeSquares (n : ℕ) : Prop :=
  ∃ p₁ p₂ p₃ p₄ : ℕ, 
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n = p₁^2 * p₂^2 * p₃^2 * p₄^2

theorem nine_digit_prime_square_product :
  ∃ a₁ a₂ a₃ b₁ b₂ b₃ : ℕ,
    a₁ ≠ 0 ∧
    MiddleIsDoubleFirst a₁ a₂ a₃ b₁ b₂ b₃ ∧
    IsProductOfFourPrimeSquares (NineDigitNumber a₁ a₂ a₃ b₁ b₂ b₃) :=
by sorry

end NUMINAMATH_CALUDE_nine_digit_prime_square_product_l162_16237


namespace NUMINAMATH_CALUDE_pants_price_l162_16283

theorem pants_price (total_cost : ℝ) (shirt_price : ℝ → ℝ) (shoes_price : ℝ → ℝ) 
  (h1 : total_cost = 340)
  (h2 : ∀ p, shirt_price p = 3/4 * p)
  (h3 : ∀ p, shoes_price p = p + 10) :
  ∃ p, p = 120 ∧ total_cost = shirt_price p + p + shoes_price p :=
sorry

end NUMINAMATH_CALUDE_pants_price_l162_16283


namespace NUMINAMATH_CALUDE_saree_price_calculation_l162_16272

theorem saree_price_calculation (final_price : ℝ) 
  (h1 : final_price = 227.70) 
  (first_discount : ℝ) (second_discount : ℝ)
  (h2 : first_discount = 0.12)
  (h3 : second_discount = 0.25) : ∃ P : ℝ, 
  P * (1 - first_discount) * (1 - second_discount) = final_price ∧ P = 345 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l162_16272


namespace NUMINAMATH_CALUDE_eight_digit_divisibility_l162_16256

-- Define a four-digit number
def four_digit_number (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + c * 10 + d

-- Define the eight-digit number formed by repeating the four-digit number
def eight_digit_number (a b c d : ℕ) : ℕ := four_digit_number a b c d * 10000 + four_digit_number a b c d

-- Theorem statement
theorem eight_digit_divisibility (a b c d : ℕ) :
  (a < 10) → (b < 10) → (c < 10) → (d < 10) →
  (∃ k₁ k₂ : ℕ, eight_digit_number a b c d = 73 * k₁ ∧ eight_digit_number a b c d = 137 * k₂) := by
  sorry


end NUMINAMATH_CALUDE_eight_digit_divisibility_l162_16256


namespace NUMINAMATH_CALUDE_fraction_of_woodwind_and_brass_players_l162_16254

theorem fraction_of_woodwind_and_brass_players (total_students : ℝ) : 
  let woodwind_last_year := (1 / 2 : ℝ) * total_students
  let brass_last_year := (2 / 5 : ℝ) * total_students
  let percussion_last_year := (1 / 10 : ℝ) * total_students
  let woodwind_this_year := (1 / 2 : ℝ) * woodwind_last_year
  let brass_this_year := (3 / 4 : ℝ) * brass_last_year
  let percussion_this_year := percussion_last_year
  let total_this_year := woodwind_this_year + brass_this_year + percussion_this_year
  (woodwind_this_year + brass_this_year) / total_this_year = (11 / 20 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_woodwind_and_brass_players_l162_16254


namespace NUMINAMATH_CALUDE_angle_measure_in_acute_triangle_l162_16226

-- Define an acute triangle
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

-- State the theorem
theorem angle_measure_in_acute_triangle (t : AcuteTriangle) :
  (t.b^2 + t.c^2 - t.a^2) * Real.tan t.A = t.b * t.c → t.A = π/6 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_acute_triangle_l162_16226


namespace NUMINAMATH_CALUDE_properties_of_negative_three_halves_l162_16228

def x : ℚ := -3/2

theorem properties_of_negative_three_halves :
  (- x = 3/2) ∧ 
  (x⁻¹ = -2/3) ∧ 
  (|x| = 3/2) := by sorry

end NUMINAMATH_CALUDE_properties_of_negative_three_halves_l162_16228


namespace NUMINAMATH_CALUDE_eggs_to_buy_l162_16206

theorem eggs_to_buy (total_needed : ℕ) (given_by_andrew : ℕ) 
  (h1 : total_needed = 222) (h2 : given_by_andrew = 155) : 
  total_needed - given_by_andrew = 67 := by
  sorry

end NUMINAMATH_CALUDE_eggs_to_buy_l162_16206


namespace NUMINAMATH_CALUDE_michaels_boxes_l162_16241

/-- Given that Michael has 16 blocks and each box must contain 2 blocks, 
    prove that the number of boxes Michael has is 8. -/
theorem michaels_boxes (total_blocks : ℕ) (blocks_per_box : ℕ) (h1 : total_blocks = 16) (h2 : blocks_per_box = 2) :
  total_blocks / blocks_per_box = 8 := by
  sorry


end NUMINAMATH_CALUDE_michaels_boxes_l162_16241


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l162_16219

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (8 * x₁^2 + 12 * x₁ - 14 = 0) → 
  (8 * x₂^2 + 12 * x₂ - 14 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 23/4) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l162_16219


namespace NUMINAMATH_CALUDE_marks_remaining_money_l162_16225

def initial_amount : ℕ := 85
def books_seven_dollars : ℕ := 3
def books_five_dollars : ℕ := 4
def books_nine_dollars : ℕ := 2

def cost_seven_dollars : ℕ := 7
def cost_five_dollars : ℕ := 5
def cost_nine_dollars : ℕ := 9

theorem marks_remaining_money :
  initial_amount - 
  (books_seven_dollars * cost_seven_dollars + 
   books_five_dollars * cost_five_dollars + 
   books_nine_dollars * cost_nine_dollars) = 26 := by
  sorry

end NUMINAMATH_CALUDE_marks_remaining_money_l162_16225


namespace NUMINAMATH_CALUDE_intersects_once_impl_a_eq_one_l162_16280

/-- The function f(x) for a given 'a' -/
def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 - 4 * x + 2 * a

/-- Predicate to check if f(x) intersects x-axis at exactly one point -/
def intersects_once (a : ℝ) : Prop :=
  ∃! x, f a x = 0

/-- Theorem: If f(x) intersects x-axis at exactly one point, then a = 1 -/
theorem intersects_once_impl_a_eq_one :
  ∀ a : ℝ, intersects_once a → a = 1 := by
sorry

end NUMINAMATH_CALUDE_intersects_once_impl_a_eq_one_l162_16280


namespace NUMINAMATH_CALUDE_dwarf_ice_cream_problem_l162_16250

theorem dwarf_ice_cream_problem :
  ∀ (n : ℕ) (vanilla chocolate fruit : ℕ),
    n = 10 →
    vanilla = n →
    chocolate = n / 2 →
    fruit = 1 →
    ∃ (truthful : ℕ),
      truthful = 4 ∧
      truthful + (n - truthful) = n ∧
      truthful + 2 * (n - truthful) = vanilla + chocolate + fruit :=
by sorry

end NUMINAMATH_CALUDE_dwarf_ice_cream_problem_l162_16250


namespace NUMINAMATH_CALUDE_real_roots_quadratic_equation_l162_16229

theorem real_roots_quadratic_equation (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) → k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_quadratic_equation_l162_16229


namespace NUMINAMATH_CALUDE_f_less_than_g_for_n_ge_5_l162_16270

theorem f_less_than_g_for_n_ge_5 (n : ℕ) (h : n ≥ 5) : n^2 + n < 2^n := by
  sorry

end NUMINAMATH_CALUDE_f_less_than_g_for_n_ge_5_l162_16270


namespace NUMINAMATH_CALUDE_ratio_of_segments_l162_16286

/-- Given collinear points A, B, C in the Cartesian plane where:
    A = (a, 0) lies on the x-axis
    B lies on the line y = x
    C lies on the line y = 2x
    AB/BC = 2
    D = (a, a)
    E is the second intersection of the circumcircle of triangle ADC with y = x
    F is the intersection of ray AE with y = 2x
    Prove that AE/EF = √2/2 -/
theorem ratio_of_segments (a : ℝ) : ∃ (B C E F : ℝ × ℝ),
  let A := (a, 0)
  let D := (a, a)
  -- B lies on y = x
  B.2 = B.1 ∧
  -- C lies on y = 2x
  C.2 = 2 * C.1 ∧
  -- AB/BC = 2
  (((B.1 - A.1)^2 + (B.2 - A.2)^2) : ℝ) / ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 4 ∧
  -- E lies on y = x
  E.2 = E.1 ∧
  -- E is on the circumcircle of ADC
  (E.1 - A.1)^2 + (E.2 - A.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2 ∧
  (E.1 - C.1)^2 + (E.2 - C.2)^2 = (D.1 - C.1)^2 + (D.2 - C.2)^2 ∧
  -- F lies on y = 2x
  F.2 = 2 * F.1 ∧
  -- F lies on ray AE
  ∃ (t : ℝ), t > 0 ∧ F.1 - A.1 = t * (E.1 - A.1) ∧ F.2 - A.2 = t * (E.2 - A.2) →
  -- Conclusion: AE/EF = √2/2
  (((E.1 - A.1)^2 + (E.2 - A.2)^2) : ℝ) / ((F.1 - E.1)^2 + (F.2 - E.2)^2) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l162_16286


namespace NUMINAMATH_CALUDE_arithmetic_progression_squares_l162_16218

theorem arithmetic_progression_squares (a b c : ℝ) 
  (h : (1 / (a + b) + 1 / (b + c)) / 2 = 1 / (a + c)) : 
  a^2 + c^2 = 2 * b^2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_squares_l162_16218


namespace NUMINAMATH_CALUDE_f_intersects_all_lines_l162_16202

/-- A function that intersects every line in the coordinate plane at least once -/
def f (x : ℝ) : ℝ := x^3

/-- Proposition: The function f intersects every line in the coordinate plane at least once -/
theorem f_intersects_all_lines :
  ∀ (k b : ℝ), ∃ (x : ℝ), f x = k * x + b :=
sorry

end NUMINAMATH_CALUDE_f_intersects_all_lines_l162_16202


namespace NUMINAMATH_CALUDE_eight_S_three_l162_16201

-- Define the operation §
def S (a b : ℤ) : ℤ := 4*a + 7*b

-- Theorem to prove
theorem eight_S_three : S 8 3 = 53 := by
  sorry

end NUMINAMATH_CALUDE_eight_S_three_l162_16201


namespace NUMINAMATH_CALUDE_f_max_value_l162_16279

/-- The function f(x) = 10x - 5x^2 -/
def f (x : ℝ) : ℝ := 10 * x - 5 * x^2

/-- The maximum value of f(x) for any real x is 5 -/
theorem f_max_value : ∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l162_16279


namespace NUMINAMATH_CALUDE_total_students_third_and_fourth_grade_l162_16234

theorem total_students_third_and_fourth_grade 
  (third_grade : ℕ) 
  (difference : ℕ) 
  (h1 : third_grade = 203)
  (h2 : difference = 125) :
  third_grade + (third_grade + difference) = 531 := by
  sorry

end NUMINAMATH_CALUDE_total_students_third_and_fourth_grade_l162_16234


namespace NUMINAMATH_CALUDE_no_one_left_behind_l162_16216

/-- Represents the Ferris wheel problem -/
structure FerrisWheel where
  seats_per_rotation : ℕ
  total_rotations : ℕ
  initial_queue : ℕ
  impatience_rate : ℚ

/-- Calculates the number of people remaining in the queue after a given number of rotations -/
def people_remaining (fw : FerrisWheel) (rotations : ℕ) : ℕ :=
  sorry

/-- The main theorem: proves that no one is left in the queue after three rotations -/
theorem no_one_left_behind (fw : FerrisWheel) 
  (h1 : fw.seats_per_rotation = 56)
  (h2 : fw.total_rotations = 3)
  (h3 : fw.initial_queue = 92)
  (h4 : fw.impatience_rate = 1/10) :
  people_remaining fw 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_one_left_behind_l162_16216


namespace NUMINAMATH_CALUDE_no_arithmetic_progression_with_product_l162_16294

theorem no_arithmetic_progression_with_product : ¬∃ (a b : ℝ), 
  (b - a = a - 5) ∧ (a * b - b = b - a) := by
  sorry

end NUMINAMATH_CALUDE_no_arithmetic_progression_with_product_l162_16294


namespace NUMINAMATH_CALUDE_parallelogram_area_25_15_l162_16222

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 25 cm and height 15 cm is 375 cm² -/
theorem parallelogram_area_25_15 :
  parallelogram_area 25 15 = 375 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_25_15_l162_16222


namespace NUMINAMATH_CALUDE_projection_area_eq_projection_length_l162_16269

/-- A cube with edge length 1 -/
structure UnitCube where
  edge_length : ℝ
  edge_length_eq_one : edge_length = 1

/-- A plane onto which the cube is projected -/
class ProjectionPlane

/-- A line perpendicular to the projection plane -/
class PerpendicularLine (P : ProjectionPlane)

/-- The area of the projection of a cube onto a plane -/
noncomputable def projection_area (cube : UnitCube) (P : ProjectionPlane) : ℝ :=
  sorry

/-- The length of the projection of a cube onto a line perpendicular to the projection plane -/
noncomputable def projection_length (cube : UnitCube) (P : ProjectionPlane) (L : PerpendicularLine P) : ℝ :=
  sorry

/-- Theorem stating that the area of the projection of a unit cube onto a plane
    is equal to the length of its projection onto a perpendicular line -/
theorem projection_area_eq_projection_length
  (cube : UnitCube) (P : ProjectionPlane) (L : PerpendicularLine P) :
  projection_area cube P = projection_length cube P L :=
sorry

end NUMINAMATH_CALUDE_projection_area_eq_projection_length_l162_16269


namespace NUMINAMATH_CALUDE_no_natural_solution_for_equation_l162_16275

theorem no_natural_solution_for_equation : ∀ m n : ℕ, m^2 ≠ n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_for_equation_l162_16275


namespace NUMINAMATH_CALUDE_division_problem_l162_16220

theorem division_problem (total : ℚ) (a_amt b_amt c_amt : ℚ) : 
  total = 544 →
  a_amt = (2/3) * b_amt →
  b_amt = (1/4) * c_amt →
  a_amt + b_amt + c_amt = total →
  b_amt = 96 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l162_16220


namespace NUMINAMATH_CALUDE_last_round_probability_l162_16260

/-- A tournament with the given conditions -/
structure Tournament (n : ℕ) where
  num_players : ℕ := 2^(n+1)
  num_rounds : ℕ := n+1
  pairing : Unit  -- Represents the pairing process
  pushover_game : Unit  -- Represents the Pushover game

/-- The probability of two specific players facing each other in the last round -/
def face_probability (t : Tournament n) : ℚ :=
  (2^n - 1) / 8^n

/-- Theorem stating the probability of players 1 and 2^n facing each other in the last round -/
theorem last_round_probability (n : ℕ) (h : n > 0) :
  ∀ (t : Tournament n), face_probability t = (2^n - 1) / 8^n :=
sorry

end NUMINAMATH_CALUDE_last_round_probability_l162_16260


namespace NUMINAMATH_CALUDE_female_salmon_count_l162_16204

theorem female_salmon_count (male_salmon : ℕ) (total_salmon : ℕ) 
  (h1 : male_salmon = 712261)
  (h2 : total_salmon = 971639) :
  total_salmon - male_salmon = 259378 := by
  sorry

end NUMINAMATH_CALUDE_female_salmon_count_l162_16204


namespace NUMINAMATH_CALUDE_last_locker_opened_l162_16287

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
| Open
| Closed

/-- Represents the direction the student is moving -/
inductive Direction
| Forward
| Backward

/-- Represents the student's action on a locker -/
def StudentAction := Nat → LockerState → Direction → (LockerState × Direction)

/-- The number of lockers in the corridor -/
def numLockers : Nat := 500

/-- The locker opening process -/
def openLockers (action : StudentAction) (n : Nat) : Nat :=
  sorry -- Implementation of the locker opening process

theorem last_locker_opened (action : StudentAction) :
  openLockers action numLockers = 242 := by
  sorry

#check last_locker_opened

end NUMINAMATH_CALUDE_last_locker_opened_l162_16287


namespace NUMINAMATH_CALUDE_point_not_on_transformed_plane_l162_16238

/-- Similarity transformation of a plane with coefficient k and center at the origin -/
def transform_plane (a b c d k : ℝ) : ℝ → ℝ → ℝ → Prop :=
  fun x y z ↦ a * x + b * y + c * z + k * d = 0

/-- The point A -/
def A : ℝ × ℝ × ℝ := (-1, 2, 3)

/-- The original plane equation -/
def plane_a : ℝ → ℝ → ℝ → Prop :=
  fun x y z ↦ x - 3 * y + z + 2 = 0

/-- The similarity transformation coefficient -/
def k : ℝ := 2.5

theorem point_not_on_transformed_plane :
  ¬ transform_plane 1 (-3) 1 2 k A.1 A.2.1 A.2.2 :=
sorry

end NUMINAMATH_CALUDE_point_not_on_transformed_plane_l162_16238


namespace NUMINAMATH_CALUDE_unique_four_digit_solution_l162_16284

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_equation (a b c d : ℕ) : Prop :=
  1000 * a + 100 * b + 10 * c + d - (100 * a + 10 * b + c) - (10 * a + b) - a = 1995

theorem unique_four_digit_solution :
  ∃! (abcd : ℕ), is_four_digit abcd ∧ 
    ∃ (a b c d : ℕ), abcd = 1000 * a + 100 * b + 10 * c + d ∧ digit_equation a b c d ∧
    a ≠ 0 := by sorry

end NUMINAMATH_CALUDE_unique_four_digit_solution_l162_16284


namespace NUMINAMATH_CALUDE_mean_scores_equal_7_l162_16205

def class1_scores : List Nat := [10, 9, 8, 7, 7, 7, 7, 5, 5, 5]
def class2_scores : List Nat := [9, 8, 8, 7, 7, 7, 7, 7, 5, 5]

def mean (scores : List Nat) : Rat :=
  (scores.sum : Rat) / scores.length

theorem mean_scores_equal_7 :
  mean class1_scores = 7 ∧ mean class2_scores = 7 := by
  sorry

end NUMINAMATH_CALUDE_mean_scores_equal_7_l162_16205


namespace NUMINAMATH_CALUDE_triangle_properties_l162_16213

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * Real.sin t.B + t.b * Real.cos t.A = 0) 
  (h2 : 0 < t.A ∧ t.A < Real.pi) 
  (h3 : 0 < t.B ∧ t.B < Real.pi) 
  (h4 : 0 < t.C ∧ t.C < Real.pi) 
  (h5 : t.A + t.B + t.C = Real.pi) :
  t.A = 3 * Real.pi / 4 ∧ 
  (t.a = 2 * Real.sqrt 5 → t.b = 2 → 
    1/2 * t.b * t.c * Real.sin t.A = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l162_16213


namespace NUMINAMATH_CALUDE_inequality_solution_set_l162_16227

theorem inequality_solution_set (x : ℝ) : 
  x^2 - |x - 1| - 1 ≤ 0 ↔ -2 ≤ x ∧ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l162_16227


namespace NUMINAMATH_CALUDE_extension_point_coordinates_l162_16252

/-- Given two points A and B in ℝ², and a point P that lies on the extension of segment AB
    such that |AP| = 4/3 * |PB|, prove that P has specific coordinates. -/
theorem extension_point_coordinates (A B P : ℝ × ℝ) : 
  A = (2, 3) →
  B = (4, -3) →
  (∃ t : ℝ, t > 1 ∧ P = A + t • (B - A)) →
  ‖P - A‖ = (4/3) * ‖P - B‖ →
  P = (10, -21) := by
  sorry


end NUMINAMATH_CALUDE_extension_point_coordinates_l162_16252


namespace NUMINAMATH_CALUDE_range_of_m_l162_16255

theorem range_of_m (P : ∀ x : ℝ, ∃ m : ℝ, 4^x - 2^(x+1) + m = 0) :
  ∀ m : ℝ, (∃ x : ℝ, 4^x - 2^(x+1) + m = 0) → m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l162_16255


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l162_16214

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  a₁ : ℚ  -- First term
  d : ℚ   -- Common difference

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.a₁ + (n - 1 : ℚ) * seq.d

theorem tenth_term_of_specific_sequence :
  ∃ (seq : ArithmeticSequence),
    seq.nthTerm 1 = 5/6 ∧
    seq.nthTerm 16 = 7/8 ∧
    seq.nthTerm 10 = 103/120 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l162_16214


namespace NUMINAMATH_CALUDE_angle_relation_l162_16292

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define an angle between three points
def Angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- State the angle bisector theorem
axiom angle_bisector_theorem (c : Circle) (A X Y C S : ℝ × ℝ) 
  (hA : PointOnCircle c A) (hX : PointOnCircle c X) (hY : PointOnCircle c Y) 
  (hC : PointOnCircle c C) (hS : PointOnCircle c S) :
  Angle A X C - Angle A Y C = Angle A S C

-- State the theorem to be proved
theorem angle_relation (c : Circle) (B X Y D S : ℝ × ℝ) 
  (hB : PointOnCircle c B) (hX : PointOnCircle c X) (hY : PointOnCircle c Y) 
  (hD : PointOnCircle c D) (hS : PointOnCircle c S) :
  Angle B X D - Angle B Y D = Angle B S D := by
  sorry

end NUMINAMATH_CALUDE_angle_relation_l162_16292


namespace NUMINAMATH_CALUDE_student_number_problem_l162_16224

theorem student_number_problem (x : ℝ) : 4 * x - 142 = 110 → x = 63 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l162_16224


namespace NUMINAMATH_CALUDE_qt_length_l162_16277

/-- Square with side length 4 and special points T and U -/
structure SpecialSquare where
  -- Square PQRS with side length 4
  side : ℝ
  side_eq : side = 4

  -- Point T on side PQ
  t : ℝ × ℝ
  t_on_pq : t.1 ≥ 0 ∧ t.1 ≤ side ∧ t.2 = 0

  -- Point U on side PS
  u : ℝ × ℝ
  u_on_ps : u.1 = 0 ∧ u.2 ≥ 0 ∧ u.2 ≤ side

  -- Lines QT and SU divide the square into four equal areas
  equal_areas : (side * t.1) / 2 = (side * side) / 4

/-- The length of QT in a SpecialSquare is 2√3 -/
theorem qt_length (sq : SpecialSquare) : 
  Real.sqrt ((sq.side - sq.t.1)^2 + sq.t.1^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_qt_length_l162_16277


namespace NUMINAMATH_CALUDE_cherry_price_theorem_l162_16242

/-- The price of a bag of cherries satisfies the given conditions -/
theorem cherry_price_theorem (olive_price : ℝ) (bag_count : ℕ) (discount_rate : ℝ) (final_cost : ℝ) :
  olive_price = 7 →
  bag_count = 50 →
  discount_rate = 0.1 →
  final_cost = 540 →
  ∃ (cherry_price : ℝ),
    cherry_price = 5 ∧
    (1 - discount_rate) * (bag_count * cherry_price + bag_count * olive_price) = final_cost :=
by sorry

end NUMINAMATH_CALUDE_cherry_price_theorem_l162_16242


namespace NUMINAMATH_CALUDE_determine_fourth_player_wins_l162_16288

/-- Represents a player in the chess tournament -/
structure Player where
  wins : Nat
  losses : Nat

/-- Represents a chess tournament -/
structure ChessTournament where
  players : Fin 4 → Player
  total_games : Nat

/-- The theorem states that given the wins and losses of three players in a four-player
    round-robin chess tournament, we can determine the number of wins for the fourth player. -/
theorem determine_fourth_player_wins (t : ChessTournament) 
  (h1 : t.players 0 = { wins := 5, losses := 3 })
  (h2 : t.players 1 = { wins := 4, losses := 4 })
  (h3 : t.players 2 = { wins := 2, losses := 6 })
  (h_total : t.total_games = 16)
  (h_balance : ∀ i, (t.players i).wins + (t.players i).losses = 8) :
  (t.players 3).wins = 5 := by
  sorry

end NUMINAMATH_CALUDE_determine_fourth_player_wins_l162_16288


namespace NUMINAMATH_CALUDE_cookie_store_spending_l162_16271

theorem cookie_store_spending : ∀ (ben david : ℝ),
  (david = 0.6 * ben) →  -- For every dollar Ben spent, David spent 40 cents less
  (ben = david + 16) →   -- Ben paid $16.00 more than David
  (ben + david = 64) :=  -- The total amount they spent together
by
  sorry

end NUMINAMATH_CALUDE_cookie_store_spending_l162_16271


namespace NUMINAMATH_CALUDE_bookshelf_picking_l162_16289

theorem bookshelf_picking (english_books math_books : ℕ) 
  (h1 : english_books = 6) 
  (h2 : math_books = 2) : 
  english_books + math_books = 8 := by
  sorry

end NUMINAMATH_CALUDE_bookshelf_picking_l162_16289


namespace NUMINAMATH_CALUDE_remainder_theorem_l162_16285

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 100 * k - 1) : (n^2 - n + 4) % 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l162_16285


namespace NUMINAMATH_CALUDE_expression_value_l162_16236

theorem expression_value (x y : ℝ) (h : (x - y) / (x + y) = 3) :
  2 * (x - y) / (x + y) - (x + y) / (3 * (x - y)) = 53 / 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l162_16236


namespace NUMINAMATH_CALUDE_min_value_product_l162_16259

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : 1/x + 1/y + 1/z = 9) :
  x^4 * y^3 * z^2 ≥ 1/3456 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    1/x₀ + 1/y₀ + 1/z₀ = 9 ∧ 
    x₀^4 * y₀^3 * z₀^2 = 1/3456 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l162_16259


namespace NUMINAMATH_CALUDE_tan_theta_value_l162_16248

open Complex

theorem tan_theta_value (θ : ℝ) :
  (↑(1 : ℂ) + I) * sin θ - (↑(1 : ℂ) + I * cos θ) ∈ {z : ℂ | z.re + z.im + 1 = 0} →
  tan θ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l162_16248


namespace NUMINAMATH_CALUDE_range_of_m_for_p_range_of_m_for_p_and_q_l162_16290

-- Define the equations for p and q
def p (x y m : ℝ) : Prop := x^2 / (m + 1) + y^2 / (4 - m) = 1
def q (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 2*m*y + 5 = 0

-- Define what it means for p to be an ellipse with foci on the x-axis
def p_is_ellipse (m : ℝ) : Prop := m + 1 > 0 ∧ 4 - m > 0 ∧ m + 1 ≠ 4 - m

-- Define what it means for q to be a circle
def q_is_circle (m : ℝ) : Prop := m^2 - 4 > 0

-- Theorem 1
theorem range_of_m_for_p (m : ℝ) :
  p_is_ellipse m → m > 3/2 ∧ m < 4 :=
sorry

-- Theorem 2
theorem range_of_m_for_p_and_q (m : ℝ) :
  p_is_ellipse m ∧ q_is_circle m → m > 2 ∧ m < 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_p_range_of_m_for_p_and_q_l162_16290


namespace NUMINAMATH_CALUDE_mans_swimming_speed_l162_16299

/-- The swimming speed of a man in still water, given that it takes him twice as long to swim upstream
    than downstream in a stream with a speed of 2.5 km/h. -/
theorem mans_swimming_speed (v : ℝ) (s : ℝ) (h1 : s = 2.5) 
    (h2 : ∃ t : ℝ, t > 0 ∧ (v + s) * t = (v - s) * (2 * t)) : v = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_mans_swimming_speed_l162_16299


namespace NUMINAMATH_CALUDE_prime_divides_product_implies_divides_factor_l162_16281

theorem prime_divides_product_implies_divides_factor 
  (p : ℕ) (n : ℕ) (a : Fin n → ℕ) 
  (h_prime : Nat.Prime p) 
  (h_divides_product : p ∣ (Finset.univ.prod a)) : 
  ∃ i, p ∣ a i :=
sorry

end NUMINAMATH_CALUDE_prime_divides_product_implies_divides_factor_l162_16281


namespace NUMINAMATH_CALUDE_function_value_at_negative_one_l162_16208

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_value_at_negative_one
  (h1 : ∀ x : ℝ, f (x + 2009) = -f (x + 2008))
  (h2 : f 2009 = -2009) :
  f (-1) = -2009 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_one_l162_16208


namespace NUMINAMATH_CALUDE_football_practice_hours_l162_16282

/-- The number of hours a football team practices daily, given their weekly schedule and total practice time. -/
def daily_practice_hours (total_hours : ℕ) (practice_days : ℕ) : ℚ :=
  total_hours / practice_days

/-- Theorem stating that the daily practice hours is 6, given the conditions of the problem. -/
theorem football_practice_hours :
  let total_week_hours : ℕ := 36
  let days_in_week : ℕ := 7
  let rain_days : ℕ := 1
  let practice_days : ℕ := days_in_week - rain_days
  daily_practice_hours total_week_hours practice_days = 6 := by
  sorry

end NUMINAMATH_CALUDE_football_practice_hours_l162_16282


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l162_16210

theorem sin_sum_of_complex_exponentials (α β : ℝ) :
  Complex.exp (Complex.I * α) = 3/5 + 4/5 * Complex.I ∧
  Complex.exp (Complex.I * β) = -12/13 + 5/13 * Complex.I →
  Real.sin (α + β) = -33/65 := by
sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l162_16210


namespace NUMINAMATH_CALUDE_total_chairs_is_528_l162_16239

/-- Calculates the total number of chairs carried to the hall by Kingsley and her friends -/
def total_chairs : ℕ :=
  let kingsley_chairs := 7
  let friend_chairs := [6, 8, 5, 9, 7]
  let trips := List.range 6 |>.map (λ i => 10 + i)
  (kingsley_chairs :: friend_chairs).zip trips
  |>.map (λ (chairs, trip) => chairs * trip)
  |>.sum

/-- Theorem stating that the total number of chairs carried is 528 -/
theorem total_chairs_is_528 : total_chairs = 528 := by
  sorry

end NUMINAMATH_CALUDE_total_chairs_is_528_l162_16239


namespace NUMINAMATH_CALUDE_inequality_equivalence_l162_16253

def solution_set : Set ℝ := {x | x ∈ Set.Ioc 0 (3/8) ∪ Set.Icc 3 4}

theorem inequality_equivalence (x : ℝ) : 
  (x / (x - 3) + (x + 4) / (3 * x) ≥ 4) ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l162_16253


namespace NUMINAMATH_CALUDE_OM_range_theorem_l162_16258

-- Define the line equation
def line_eq (m n x y : ℝ) : Prop := 2 * m * x - (4 * m + n) * y + 2 * n = 0

-- Define point P
def point_P : ℝ × ℝ := (2, 6)

-- Define that m and n are not simultaneously zero
def not_zero (m n : ℝ) : Prop := m ≠ 0 ∨ n ≠ 0

-- Define the perpendicular line passing through P
def perp_line (m n : ℝ) (M : ℝ × ℝ) : Prop :=
  line_eq m n M.1 M.2 ∧ 
  (M.1 - point_P.1) * (2 * m) + (M.2 - point_P.2) * (-(4 * m + n)) = 0

-- Define the range of |OM|
def OM_range (x : ℝ) : Prop := 5 - Real.sqrt 5 ≤ x ∧ x ≤ 5 + Real.sqrt 5

-- Theorem statement
theorem OM_range_theorem (m n : ℝ) (M : ℝ × ℝ) :
  not_zero m n →
  perp_line m n M →
  OM_range (Real.sqrt (M.1^2 + M.2^2)) :=
sorry

end NUMINAMATH_CALUDE_OM_range_theorem_l162_16258


namespace NUMINAMATH_CALUDE_r_l162_16291

/-- r'(n) is the sum of distinct primes in the prime factorization of n -/
noncomputable def r' (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

/-- The set of composite positive integers -/
def CompositeSet : Set ℕ :=
  {n : ℕ | n > 1 ∧ ¬Nat.Prime n}

/-- The set of integers that can be expressed as sums of two or more distinct primes -/
def SumOfDistinctPrimesSet : Set ℕ :=
  {n : ℕ | ∃ (s : Finset ℕ), s.card ≥ 2 ∧ (∀ p ∈ s, Nat.Prime p) ∧ s.sum id = n}

/-- The range of r' is equal to the set of integers that can be expressed as sums of two or more distinct primes -/
theorem r'_range_eq_sum_of_distinct_primes :
  (CompositeSet.image r') = SumOfDistinctPrimesSet :=
sorry

end NUMINAMATH_CALUDE_r_l162_16291


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l162_16295

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def num_distributions (balls : ℕ) (boxes : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 8 ways to distribute 6 indistinguishable balls into 4 indistinguishable boxes -/
theorem six_balls_four_boxes : num_distributions 6 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l162_16295


namespace NUMINAMATH_CALUDE_tom_total_distance_l162_16247

/-- Calculates the total distance covered by Tom given his swimming and running times and speeds. -/
theorem tom_total_distance (swim_time swim_speed : ℝ) (h1 : swim_time = 2) (h2 : swim_speed = 2)
  (h3 : swim_time > 0) (h4 : swim_speed > 0) : 
  let run_time := swim_time / 2
  let run_speed := 4 * swim_speed
  swim_time * swim_speed + run_time * run_speed = 12 := by
  sorry

#check tom_total_distance

end NUMINAMATH_CALUDE_tom_total_distance_l162_16247


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_symmetric_points_coordinates_l162_16215

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The symmetric point about the y-axis -/
def symmetricAboutYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

theorem symmetric_point_y_axis (p : Point2D) :
  let p' := symmetricAboutYAxis p
  p'.x = -p.x ∧ p'.y = p.y := by sorry

/-- Given points A, B, and C -/
def A : Point2D := { x := -3, y := 2 }
def B : Point2D := { x := -4, y := -3 }
def C : Point2D := { x := -1, y := -1 }

/-- Symmetric points A', B', and C' -/
def A' : Point2D := symmetricAboutYAxis A
def B' : Point2D := symmetricAboutYAxis B
def C' : Point2D := symmetricAboutYAxis C

theorem symmetric_points_coordinates :
  A'.x = 3 ∧ A'.y = 2 ∧
  B'.x = 4 ∧ B'.y = -3 ∧
  C'.x = 1 ∧ C'.y = -1 := by sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_symmetric_points_coordinates_l162_16215


namespace NUMINAMATH_CALUDE_water_volume_ratio_in_cone_l162_16266

/-- Theorem: Volume ratio of water in a cone filled to 2/3 height -/
theorem water_volume_ratio_in_cone (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let water_height : ℝ := 2 / 3 * h
  let water_radius : ℝ := 2 / 3 * r
  let cone_volume : ℝ := (1 / 3) * π * r^2 * h
  let water_volume : ℝ := (1 / 3) * π * water_radius^2 * water_height
  water_volume / cone_volume = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_water_volume_ratio_in_cone_l162_16266


namespace NUMINAMATH_CALUDE_a_100_value_l162_16257

/-- Sequence S defined recursively -/
def S : ℕ → ℚ
| 0 => 0
| 1 => 3
| (n + 2) => 3 / (3 * n + 1)

/-- Sequence a defined in terms of S -/
def a : ℕ → ℚ
| 0 => 0
| 1 => 3
| (n + 2) => (3 * (S (n + 2))^2) / (3 * S (n + 2) - 2)

/-- Main theorem: a₁₀₀ = -9/84668 -/
theorem a_100_value : a 100 = -9/84668 := by sorry

end NUMINAMATH_CALUDE_a_100_value_l162_16257


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l162_16203

theorem quadratic_equation_solution :
  let x₁ : ℝ := (3 + Real.sqrt 15) / 3
  let x₂ : ℝ := (3 - Real.sqrt 15) / 3
  (3 * x₁^2 - 6 * x₁ - 2 = 0) ∧ (3 * x₂^2 - 6 * x₂ - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l162_16203


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a12_l162_16262

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  (a 7 + a 9 = 16) ∧
  (a 4 = 4)

/-- Theorem: For the given arithmetic sequence, a_12 = 12 -/
theorem arithmetic_sequence_a12 (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  a 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a12_l162_16262


namespace NUMINAMATH_CALUDE_milk_remaining_l162_16298

theorem milk_remaining (initial : ℚ) (given_away : ℚ) (remaining : ℚ) : 
  initial = 5 → given_away = 18/7 → remaining = initial - given_away → remaining = 17/7 := by
  sorry

end NUMINAMATH_CALUDE_milk_remaining_l162_16298


namespace NUMINAMATH_CALUDE_cube_sum_equation_l162_16244

theorem cube_sum_equation (y : ℝ) (h : y^3 + 4 / y^3 = 110) : y + 4 / y = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equation_l162_16244


namespace NUMINAMATH_CALUDE_constant_sum_sequence_2013_l162_16245

/-- A sequence where the sum of any three consecutive terms is constant -/
def ConstantSumSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n + a (n + 1) + a (n + 2) = a (n + 1) + a (n + 2) + a (n + 3)

theorem constant_sum_sequence_2013 (a : ℕ → ℝ) (x : ℝ) 
    (h_constant_sum : ConstantSumSequence a)
    (h_a3 : a 3 = x)
    (h_a999 : a 999 = 3 - 2*x) :
    a 2013 = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_sum_sequence_2013_l162_16245


namespace NUMINAMATH_CALUDE_savings_calculation_l162_16231

theorem savings_calculation (income : ℕ) (ratio_income : ℕ) (ratio_expenditure : ℕ) :
  income = 21000 →
  ratio_income = 7 →
  ratio_expenditure = 6 →
  income - (income * ratio_expenditure / ratio_income) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l162_16231


namespace NUMINAMATH_CALUDE_james_club_expenditure_l162_16265

/-- Calculate the total amount James spent at the club --/
theorem james_club_expenditure :
  let entry_fee : ℕ := 20
  let rounds_for_friends : ℕ := 2
  let num_friends : ℕ := 5
  let drinks_for_self : ℕ := 6
  let drink_cost : ℕ := 6
  let food_cost : ℕ := 14
  let tip_percentage : ℚ := 30 / 100

  let drinks_cost : ℕ := rounds_for_friends * num_friends * drink_cost + drinks_for_self * drink_cost
  let subtotal : ℕ := entry_fee + drinks_cost + food_cost
  let tip : ℕ := (tip_percentage * (drinks_cost + food_cost)).num.toNat
  let total_spent : ℕ := subtotal + tip

  total_spent = 163 := by sorry

end NUMINAMATH_CALUDE_james_club_expenditure_l162_16265


namespace NUMINAMATH_CALUDE_rhombus_area_in_rectangle_l162_16209

/-- The area of a rhombus formed by intersecting equilateral triangles in a rectangle --/
theorem rhombus_area_in_rectangle (a b : ℝ) (h1 : a = 4 * Real.sqrt 3) (h2 : b = 3 * Real.sqrt 3) :
  let triangle_height := (Real.sqrt 3 / 2) * a
  let overlap := 2 * triangle_height - b
  let rhombus_area := (1 / 2) * overlap * a
  rhombus_area = 54 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_in_rectangle_l162_16209


namespace NUMINAMATH_CALUDE_christine_walking_distance_l162_16243

/-- Given Christine's walking speed and time spent walking, calculate the distance she wandered. -/
theorem christine_walking_distance 
  (speed : ℝ) 
  (time : ℝ) 
  (h1 : speed = 4) 
  (h2 : time = 5) : 
  speed * time = 20 := by
sorry

end NUMINAMATH_CALUDE_christine_walking_distance_l162_16243


namespace NUMINAMATH_CALUDE_greatest_common_divisor_420_90_under_50_l162_16207

theorem greatest_common_divisor_420_90_under_50 : 
  ∀ n : ℕ, n ∣ 420 ∧ n < 50 ∧ n ∣ 90 → n ≤ 30 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_420_90_under_50_l162_16207


namespace NUMINAMATH_CALUDE_simplify_expression_l162_16233

theorem simplify_expression :
  1 / ((3 / (Real.sqrt 2 + 2)) + (4 / (Real.sqrt 5 - 2))) =
  1 / (11 + 4 * Real.sqrt 5 - (3 * Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l162_16233


namespace NUMINAMATH_CALUDE_parabola_directrix_l162_16263

/-- Given a parabola with equation 16y^2 = x, its directrix equation is x = -1/64 -/
theorem parabola_directrix (x y : ℝ) : 
  (16 * y^2 = x) → (∃ (k : ℝ), k = -1/64 ∧ k = x) := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l162_16263


namespace NUMINAMATH_CALUDE_roots_transformation_l162_16293

theorem roots_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + 9 = 0) ∧ 
  (r₂^3 - 4*r₂^2 + 9 = 0) ∧ 
  (r₃^3 - 4*r₃^2 + 9 = 0) →
  ((3*r₁)^3 - 12*(3*r₁)^2 + 243 = 0) ∧
  ((3*r₂)^3 - 12*(3*r₂)^2 + 243 = 0) ∧
  ((3*r₃)^3 - 12*(3*r₃)^2 + 243 = 0) := by
sorry

end NUMINAMATH_CALUDE_roots_transformation_l162_16293


namespace NUMINAMATH_CALUDE_solution_count_l162_16246

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem solution_count (a : ℝ) :
  (∀ x > 1, f x ≠ (x - 1) * (a * x - a + 1)) ∨
  (a > 0 ∧ a < 1/2 ∧ (∀ x > 1, f x = (x - 1) * (a * x - a + 1) → 
    ∀ y > 1, y ≠ x → f y ≠ (y - 1) * (a * y - a + 1))) :=
by sorry

end NUMINAMATH_CALUDE_solution_count_l162_16246


namespace NUMINAMATH_CALUDE_equation_solutions_l162_16273

def equation (x : ℝ) : Prop :=
  x ≠ 3 ∧ x ≠ 5 ∧
  (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 3) * (x - 2) * (x - 1) /
  ((x - 3) * (x - 5) * (x - 3)) = 1

theorem equation_solutions :
  {x : ℝ | equation x} = {1, (3 + Real.sqrt 3) / 2, (3 - Real.sqrt 3) / 2} :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l162_16273


namespace NUMINAMATH_CALUDE_percentage_passed_both_l162_16276

theorem percentage_passed_both (failed_hindi : ℝ) (failed_english : ℝ) (failed_both : ℝ)
  (h1 : failed_hindi = 20)
  (h2 : failed_english = 70)
  (h3 : failed_both = 10) :
  100 - (failed_hindi + failed_english - failed_both) = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_passed_both_l162_16276


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_change_l162_16274

theorem rectangular_prism_volume_change (V l w h : ℝ) (h1 : V = l * w * h) :
  2 * l * (3 * w) * (h / 4) = 1.5 * V := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_change_l162_16274


namespace NUMINAMATH_CALUDE_geometric_condition_implies_a_equals_two_l162_16235

/-- The value of a for which the given geometric conditions are satisfied -/
def geometric_a : ℝ := 2

/-- The line equation y = 2x + 2 -/
def line (x : ℝ) : ℝ := 2 * x + 2

/-- The parabola equation y = ax^2 -/
def parabola (a x : ℝ) : ℝ := a * x^2

/-- Theorem stating that under the given geometric conditions, a = 2 -/
theorem geometric_condition_implies_a_equals_two (a : ℝ) 
  (h_pos : a > 0)
  (h_intersect : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ line x₁ = parabola a x₁ ∧ line x₂ = parabola a x₂)
  (h_midpoint : ∃ (x_mid : ℝ), x_mid = (x₁ + x₂) / 2 ∧ 
    parabola a x_mid = a * x_mid^2 ∧ 
    ∀ (y : ℝ), y ≠ a * x_mid^2 → |x_mid - x₁| + |y - line x₁| = |x_mid - x₂| + |y - line x₂|)
  (h_vector_condition : ∀ (A P Q : ℝ × ℝ), 
    P.1 ≠ Q.1 → 
    line P.1 = P.2 → line Q.1 = Q.2 → 
    parabola a A.1 = A.2 → 
    |(A.1 - P.1, A.2 - P.2)| + |(A.1 - Q.1, A.2 - Q.2)| = 
    |(A.1 - P.1, A.2 - P.2)| - |(A.1 - Q.1, A.2 - Q.2)|)
  : a = geometric_a := by sorry

end NUMINAMATH_CALUDE_geometric_condition_implies_a_equals_two_l162_16235


namespace NUMINAMATH_CALUDE_touring_plans_count_l162_16223

def num_destinations : Nat := 3
def num_students : Nat := 4

def total_assignments : Nat := num_destinations ^ num_students

def assignments_without_specific_destination : Nat := (num_destinations - 1) ^ num_students

theorem touring_plans_count : 
  total_assignments - assignments_without_specific_destination = 65 := by
  sorry

end NUMINAMATH_CALUDE_touring_plans_count_l162_16223


namespace NUMINAMATH_CALUDE_correct_number_of_pair_sets_l162_16261

/-- The number of ways to form 6 pairs of balls with different colors -/
def number_of_pair_sets (green red blue : ℕ) : ℕ :=
  if green = 3 ∧ red = 4 ∧ blue = 5 then 1440 else 0

/-- Theorem stating the correct number of pair sets for the given ball counts -/
theorem correct_number_of_pair_sets :
  number_of_pair_sets 3 4 5 = 1440 := by sorry

end NUMINAMATH_CALUDE_correct_number_of_pair_sets_l162_16261


namespace NUMINAMATH_CALUDE_no_simultaneous_integer_fractions_l162_16297

theorem no_simultaneous_integer_fractions :
  ¬ ∃ (n : ℤ), (∃ (a b : ℤ), (n - 6 : ℚ) / 15 = a ∧ (n - 5 : ℚ) / 24 = b) :=
by sorry

end NUMINAMATH_CALUDE_no_simultaneous_integer_fractions_l162_16297


namespace NUMINAMATH_CALUDE_lattice_points_count_l162_16251

/-- The number of lattice points on a line segment with given integer endpoints -/
def countLatticePoints (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem: The number of lattice points on the line segment from (5,13) to (47,275) is 3 -/
theorem lattice_points_count : countLatticePoints 5 13 47 275 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_count_l162_16251


namespace NUMINAMATH_CALUDE_cement_calculation_l162_16278

/-- The renovation project requires materials in truck-loads -/
structure RenovationMaterials where
  total : ℚ
  sand : ℚ
  dirt : ℚ

/-- Calculate the truck-loads of cement required for the renovation project -/
def cement_required (materials : RenovationMaterials) : ℚ :=
  materials.total - (materials.sand + materials.dirt)

theorem cement_calculation (materials : RenovationMaterials) 
  (h1 : materials.total = 0.6666666666666666)
  (h2 : materials.sand = 0.16666666666666666)
  (h3 : materials.dirt = 0.3333333333333333) :
  cement_required materials = 0.1666666666666666 := by
  sorry

#eval cement_required ⟨0.6666666666666666, 0.16666666666666666, 0.3333333333333333⟩

end NUMINAMATH_CALUDE_cement_calculation_l162_16278


namespace NUMINAMATH_CALUDE_distinct_towers_count_l162_16211

/-- Represents the number of cubes of each color -/
structure CubeCount where
  red : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the number of distinct towers -/
def countDistinctTowers (cubes : CubeCount) (towerHeight : Nat) : Nat :=
  sorry

/-- Theorem: The number of distinct towers of height 10 that can be built
    using 3 red cubes, 4 blue cubes, and 5 yellow cubes, with two cubes
    not being used, is equal to 6,812 -/
theorem distinct_towers_count :
  let cubes : CubeCount := { red := 3, blue := 4, yellow := 5 }
  let towerHeight : Nat := 10
  countDistinctTowers cubes towerHeight = 6812 := by
  sorry

end NUMINAMATH_CALUDE_distinct_towers_count_l162_16211


namespace NUMINAMATH_CALUDE_partner_a_investment_l162_16221

/-- Represents the investment and profit distribution scenario described in the problem -/
structure BusinessScenario where
  a_investment : ℚ  -- Investment of partner a
  b_investment : ℚ  -- Investment of partner b
  total_profit : ℚ  -- Total profit
  a_total_received : ℚ  -- Total amount received by partner a
  management_fee_percent : ℚ  -- Percentage of profit for management

/-- The main theorem representing the problem -/
theorem partner_a_investment (scenario : BusinessScenario) : 
  scenario.b_investment = 2500 ∧ 
  scenario.total_profit = 9600 ∧
  scenario.a_total_received = 6000 ∧
  scenario.management_fee_percent = 1/10 →
  scenario.a_investment = 3500 := by
sorry


end NUMINAMATH_CALUDE_partner_a_investment_l162_16221


namespace NUMINAMATH_CALUDE_equation_solution_l162_16249

theorem equation_solution : ∃! (x : ℝ), (81 : ℝ) ^ (x - 2) / (9 : ℝ) ^ (x - 1) = (729 : ℝ) ^ (3 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l162_16249


namespace NUMINAMATH_CALUDE_cone_sphere_volume_l162_16240

/-- Given a cone with lateral surface forming a semicircle of radius 2√3 when unrolled,
    and its vertex and base circumference lying on a sphere O,
    prove that the volume of sphere O is 32π/3 -/
theorem cone_sphere_volume (l : ℝ) (r : ℝ) (h : ℝ) (R : ℝ) :
  l = 2 * Real.sqrt 3 →                  -- lateral surface radius
  r = l / 2 →                            -- base radius
  h^2 + r^2 = l^2 →                      -- Pythagorean theorem
  2 * R * h = l^2 →                      -- sphere diameter relation
  (4 / 3) * π * R^3 = (32 * π) / 3 :=    -- sphere volume
by sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_l162_16240


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l162_16212

theorem fraction_to_decimal : (3 : ℚ) / 80 = 0.0375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l162_16212


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l162_16268

def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-2, -1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l162_16268


namespace NUMINAMATH_CALUDE_cosine_value_l162_16230

theorem cosine_value (α : Real) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : Real.sin (α + π / 6) = 3 / 5) : 
  Real.cos (α - π / 6) = (3 * Real.sqrt 3 - 4) / 10 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_l162_16230


namespace NUMINAMATH_CALUDE_unique_solution_for_2n_plus_1_eq_m2_l162_16217

theorem unique_solution_for_2n_plus_1_eq_m2 :
  ∃! (m n : ℕ), 2^n + 1 = m^2 ∧ m = 3 ∧ n = 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_2n_plus_1_eq_m2_l162_16217


namespace NUMINAMATH_CALUDE_total_wrapping_paper_l162_16200

/-- The amount of wrapping paper needed for three presents -/
def wrapping_paper (first_present second_present third_present : ℝ) : ℝ :=
  first_present + second_present + third_present

/-- Theorem: The total amount of wrapping paper needed is 7 square feet -/
theorem total_wrapping_paper :
  let first_present := 2
  let second_present := 3/4 * first_present
  let third_present := first_present + second_present
  wrapping_paper first_present second_present third_present = 7 :=
by sorry

end NUMINAMATH_CALUDE_total_wrapping_paper_l162_16200


namespace NUMINAMATH_CALUDE_sol_earnings_l162_16232

/-- Calculates the earnings from candy bar sales over a week -/
def candy_bar_earnings (initial_sales : ℕ) (daily_increase : ℕ) (days : ℕ) (price_cents : ℕ) : ℚ :=
  let total_sales := (List.range days).map (fun i => initial_sales + i * daily_increase) |>.sum
  (total_sales * price_cents : ℕ) / 100

/-- Theorem: Sol's earnings from candy bar sales over a week -/
theorem sol_earnings : candy_bar_earnings 10 4 6 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sol_earnings_l162_16232


namespace NUMINAMATH_CALUDE_greg_lunch_payment_l162_16267

/-- Calculates the total amount paid for a meal including tax and tip -/
def total_amount_paid (cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  cost + (cost * tax_rate) + (cost * tip_rate)

/-- Theorem stating that Greg paid $110 for his lunch -/
theorem greg_lunch_payment :
  let cost : ℝ := 100
  let tax_rate : ℝ := 0.04
  let tip_rate : ℝ := 0.06
  total_amount_paid cost tax_rate tip_rate = 110 := by
  sorry

end NUMINAMATH_CALUDE_greg_lunch_payment_l162_16267


namespace NUMINAMATH_CALUDE_calculate_expression_l162_16264

theorem calculate_expression : (8^3 / 8^2) * 2^6 = 512 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l162_16264


namespace NUMINAMATH_CALUDE_walk_bike_time_difference_l162_16296

def blocks : ℕ := 18
def walk_time_per_block : ℚ := 1
def bike_time_per_block : ℚ := 20 / 60

theorem walk_bike_time_difference :
  (blocks * walk_time_per_block) - (blocks * bike_time_per_block) = 12 := by
  sorry

end NUMINAMATH_CALUDE_walk_bike_time_difference_l162_16296
