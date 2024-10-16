import Mathlib

namespace NUMINAMATH_CALUDE_milestone_number_l573_57338

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  ones : ℕ
  h1 : tens ≥ 1 ∧ tens ≤ 9
  h2 : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a TwoDigitNumber to a natural number -/
def TwoDigitNumber.toNat (n : TwoDigitNumber) : ℕ := 10 * n.tens + n.ones

/-- Theorem: Given the conditions of the problem, the initial number must be 16 -/
theorem milestone_number (initial : TwoDigitNumber) 
  (h1 : initial.toNat + initial.toNat = 100 * initial.ones + initial.tens + 100 * initial.tens + initial.ones) :
  initial.tens = 1 ∧ initial.ones = 6 := by
  sorry

#check milestone_number

end NUMINAMATH_CALUDE_milestone_number_l573_57338


namespace NUMINAMATH_CALUDE_chess_tournament_score_l573_57379

theorem chess_tournament_score (total_games wins draws losses : ℕ) 
  (old_score : ℚ) : 
  total_games = wins + draws + losses →
  old_score = wins + (1/2 : ℚ) * draws →
  total_games = 52 →
  old_score = 35 →
  (wins : ℤ) - losses = 18 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_score_l573_57379


namespace NUMINAMATH_CALUDE_angle_expression_equals_half_l573_57366

theorem angle_expression_equals_half (θ : Real) (h : Real.tan θ = 3) :
  (Real.sin θ + Real.cos (π - θ)) / (Real.sin (π / 2 - θ) - Real.sin (π + θ)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_expression_equals_half_l573_57366


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_zero_l573_57333

theorem simplify_and_evaluate (a : ℤ) 
  (h1 : -2 < a) (h2 : a < 3) (h3 : a ≠ 1) (h4 : a ≠ -1) :
  (a^2 / (a + 1) - a + 1) / ((a^2 - 1) / (a^2 + 2*a + 1)) = 1 / (a - 1) :=
by sorry

-- Evaluation at a = 0
theorem evaluate_at_zero : 
  (0^2 / (0 + 1) - 0 + 1) / ((0^2 - 1) / (0^2 + 2*0 + 1)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_zero_l573_57333


namespace NUMINAMATH_CALUDE_prob_allison_between_brian_and_noah_l573_57308

/-- Represents a 6-sided cube with specific face values -/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- Allison's cube with all faces showing 6 -/
def allison_cube : Cube :=
  { faces := λ _ => 6 }

/-- Brian's cube with faces numbered 1 to 6 -/
def brian_cube : Cube :=
  { faces := λ i => i.val + 1 }

/-- Noah's cube with three faces showing 4 and three faces showing 7 -/
def noah_cube : Cube :=
  { faces := λ i => if i.val < 3 then 4 else 7 }

/-- The probability of rolling a specific value or higher on a given cube -/
def prob_roll_ge (c : Cube) (n : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i ≥ n) (Finset.univ : Finset (Fin 6))).card / 6

/-- The probability of rolling a specific value or lower on a given cube -/
def prob_roll_le (c : Cube) (n : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i ≤ n) (Finset.univ : Finset (Fin 6))).card / 6

/-- The main theorem stating the probability of Allison rolling higher than Brian but lower than Noah -/
theorem prob_allison_between_brian_and_noah :
  prob_roll_ge brian_cube 6 * prob_roll_ge noah_cube 7 = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_allison_between_brian_and_noah_l573_57308


namespace NUMINAMATH_CALUDE_equation_solution_l573_57331

theorem equation_solution : ∃ x : ℚ, (4 * x + 5 * x = 350 - 10 * (x - 5)) ∧ (x = 400 / 19) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l573_57331


namespace NUMINAMATH_CALUDE_jellybean_problem_l573_57346

/-- Calculates the number of jellybeans removed after adding some back -/
def jellybeans_removed_after_adding_back (initial : ℕ) (removed : ℕ) (added_back : ℕ) (final : ℕ) : ℕ :=
  initial - removed + added_back - final

theorem jellybean_problem (initial : ℕ) (removed : ℕ) (added_back : ℕ) (final : ℕ)
  (h1 : initial = 37)
  (h2 : removed = 15)
  (h3 : added_back = 5)
  (h4 : final = 23) :
  jellybeans_removed_after_adding_back initial removed added_back final = 4 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l573_57346


namespace NUMINAMATH_CALUDE_gcd_equality_pairs_l573_57300

theorem gcd_equality_pairs :
  ∀ a b : ℕ+, a ≤ b →
  (∀ x : ℕ+, Nat.gcd x a * Nat.gcd x b = Nat.gcd x 20 * Nat.gcd x 22) →
  ((a = 2 ∧ b = 220) ∨ (a = 4 ∧ b = 110) ∨ (a = 10 ∧ b = 44) ∨ (a = 20 ∧ b = 22)) :=
by sorry

end NUMINAMATH_CALUDE_gcd_equality_pairs_l573_57300


namespace NUMINAMATH_CALUDE_crayons_difference_is_seven_l573_57377

/-- The number of crayons Nori gave to Lea more than Mae -/
def crayons_difference : ℕ :=
  let initial_boxes : ℕ := 4
  let crayons_per_box : ℕ := 8
  let crayons_to_mae : ℕ := 5
  let crayons_left : ℕ := 15
  let initial_crayons : ℕ := initial_boxes * crayons_per_box
  let crayons_after_mae : ℕ := initial_crayons - crayons_to_mae
  let crayons_to_lea : ℕ := crayons_after_mae - crayons_left
  crayons_to_lea - crayons_to_mae

theorem crayons_difference_is_seven :
  crayons_difference = 7 := by
  sorry

end NUMINAMATH_CALUDE_crayons_difference_is_seven_l573_57377


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_problem_l573_57302

theorem consecutive_odd_numbers_problem (x : ℤ) : 
  Odd x ∧ 
  (8 * x = 3 * (x + 4) + 2 * (x + 2) + 5) ∧ 
  (∃ p : ℕ, Prime p ∧ (x + (x + 2) + (x + 4)) % p = 0) → 
  x = 7 := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_problem_l573_57302


namespace NUMINAMATH_CALUDE_line_intersects_AB_CD_l573_57319

/-- Given points A, B, C, D, prove that the line x = 8t, y = 2t, z = 11t passes through
    the origin and intersects both lines AB and CD. -/
theorem line_intersects_AB_CD :
  let A : ℝ × ℝ × ℝ := (1, 0, 1)
  let B : ℝ × ℝ × ℝ := (-2, 2, 1)
  let C : ℝ × ℝ × ℝ := (2, 0, 3)
  let D : ℝ × ℝ × ℝ := (0, 4, -2)
  let line (t : ℝ) : ℝ × ℝ × ℝ := (8*t, 2*t, 11*t)
  (∃ t : ℝ, line t = (0, 0, 0)) ∧ 
  (∃ t₁ s₁ : ℝ, line t₁ = (1-3*s₁, 2*s₁, 1)) ∧
  (∃ t₂ s₂ : ℝ, line t₂ = (2-2*s₂, 4*s₂, 3+5*s₂)) :=
by
  sorry


end NUMINAMATH_CALUDE_line_intersects_AB_CD_l573_57319


namespace NUMINAMATH_CALUDE_quadratic_rational_root_even_denominator_l573_57357

theorem quadratic_rational_root_even_denominator
  (a b c : ℤ)  -- Coefficients are integers
  (h_even_sum : Even (a + b))  -- Sum of a and b is even
  (h_odd_c : Odd c)  -- c is odd
  (p q : ℤ)  -- p/q is a rational root in simplest form
  (h_coprime : Nat.Coprime p.natAbs q.natAbs)  -- p and q are coprime
  (h_root : a * p^2 + b * p * q + c * q^2 = 0)  -- p/q is a root
  : Even q  -- q is even
:= by sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_even_denominator_l573_57357


namespace NUMINAMATH_CALUDE_problem_statement_l573_57349

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (parallelPlanes : Plane → Plane → Prop)

-- Given two lines a and b, and two planes α and β
variable (a b : Line) (α β : Plane)

-- Given that b is perpendicular to α
variable (h : perpendicular b α)

theorem problem_statement :
  (parallel a α → perpendicularLines a b) ∧
  (perpendicular b β → parallelPlanes α β) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l573_57349


namespace NUMINAMATH_CALUDE_system_solution_l573_57382

theorem system_solution (a b c x y z : ℝ) : 
  (a * y + b * x = c ∧ c * x + a * z = b ∧ b * z + c * y = a) ↔
  ((a * b * c ≠ 0 ∧ 
    x = (b^2 + c^2 - a^2) / (2*b*c) ∧ 
    y = (a^2 + c^2 - b^2) / (2*a*c) ∧ 
    z = (a^2 + b^2 - c^2) / (2*a*b)) ∨
   (a = 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
    ((x = 1 ∧ y = z) ∨ (x = 1 ∧ y = -z))) ∨
   (b = 0 ∧ a ≠ 0 ∧ c ≠ 0 ∧ 
    ((y = 1 ∧ x = z) ∨ (y = 1 ∧ x = -z))) ∨
   (c = 0 ∧ a ≠ 0 ∧ b ≠ 0 ∧ 
    ((z = 1 ∧ x = y) ∨ (z = 1 ∧ x = -y))) ∨
   (a = 0 ∧ b = 0 ∧ c = 0)) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l573_57382


namespace NUMINAMATH_CALUDE_original_decimal_l573_57390

theorem original_decimal (x : ℝ) : (x - x / 100 = 1.485) → x = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_original_decimal_l573_57390


namespace NUMINAMATH_CALUDE_max_pens_with_ten_dollars_l573_57350

/-- Represents the maximum number of pens that can be bought with a given budget. -/
def maxPens (budget : ℕ) : ℕ :=
  let singlePenCost : ℕ := 1
  let fourPackCost : ℕ := 3
  let sevenPackCost : ℕ := 4
  -- The actual calculation would go here
  16

/-- Theorem stating that with a $10 budget, the maximum number of pens that can be bought is 16. -/
theorem max_pens_with_ten_dollars :
  maxPens 10 = 16 := by
  sorry

#eval maxPens 10

end NUMINAMATH_CALUDE_max_pens_with_ten_dollars_l573_57350


namespace NUMINAMATH_CALUDE_tenth_term_of_a_sum_of_2023rd_terms_l573_57371

/-- Sequence a_n defined as (-2)^n -/
def a (n : ℕ) : ℤ := (-2) ^ n

/-- Sequence b_n defined as (-2)^n + (n+1) -/
def b (n : ℕ) : ℤ := (-2) ^ n + (n + 1)

/-- The 10th term of sequence a_n is (-2)^10 -/
theorem tenth_term_of_a : a 10 = (-2) ^ 10 := by sorry

/-- The sum of the 2023rd terms of sequences a_n and b_n is -2^2024 + 2024 -/
theorem sum_of_2023rd_terms : a 2023 + b 2023 = -2 ^ 2024 + 2024 := by sorry

end NUMINAMATH_CALUDE_tenth_term_of_a_sum_of_2023rd_terms_l573_57371


namespace NUMINAMATH_CALUDE_specific_card_draw_probability_l573_57386

theorem specific_card_draw_probability : 
  let deck_size : ℕ := 52
  let prob_specific_card : ℚ := 1 / deck_size
  let prob_both_specific_cards : ℚ := prob_specific_card * prob_specific_card
  prob_both_specific_cards = 1 / 2704 := by
  sorry

end NUMINAMATH_CALUDE_specific_card_draw_probability_l573_57386


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l573_57343

/-- The coordinates of a point with respect to the origin are the same as its definition. -/
theorem point_coordinates_wrt_origin (x y : ℝ) : 
  let P : ℝ × ℝ := (x, y)
  (P.1, P.2) = (x, y) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l573_57343


namespace NUMINAMATH_CALUDE_max_product_on_line_l573_57353

/-- Given points A(a,b) and B(4,2) on the line y = kx + 3 where k is a non-zero constant,
    the maximum value of the product ab is 9. -/
theorem max_product_on_line (a b : ℝ) (k : ℝ) : 
  k ≠ 0 → 
  b = k * a + 3 → 
  2 = k * 4 + 3 → 
  ∃ (max : ℝ), max = 9 ∧ ∀ (x y : ℝ), y = k * x + 3 → x * y ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_product_on_line_l573_57353


namespace NUMINAMATH_CALUDE_m_minus_n_equals_negative_interval_l573_57324

-- Define the sets M and N
def M : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1}

-- Define the set difference operation
def setDifference (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- State the theorem
theorem m_minus_n_equals_negative_interval :
  setDifference M N = {x | -3 ≤ x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_m_minus_n_equals_negative_interval_l573_57324


namespace NUMINAMATH_CALUDE_box_volume_l573_57307

/-- Given a rectangular box with face areas 30, 18, and 45 square centimeters, 
    its volume is 90√3 cubic centimeters. -/
theorem box_volume (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : b * c = 18) 
  (h3 : c * a = 45) : 
  a * b * c = 90 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l573_57307


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_f_l573_57329

noncomputable def f (x : ℝ) : ℤ :=
  if x > 0 then Int.ceil (1 / (x + 1))
  else if x < 0 then Int.ceil (1 / (x - 1))
  else 0  -- This value doesn't matter as we exclude x = 0

theorem zero_not_in_range_of_f :
  ∀ x : ℝ, x ≠ 0 → f x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_f_l573_57329


namespace NUMINAMATH_CALUDE_orange_purchase_l573_57311

theorem orange_purchase (x : ℝ) : 
  (x + 5) + 2*x + 2*x = 75 → x = 14 := by
  sorry

end NUMINAMATH_CALUDE_orange_purchase_l573_57311


namespace NUMINAMATH_CALUDE_no_solution_implies_a_zero_l573_57361

/-- A system of equations with no solutions implies a = 0 -/
theorem no_solution_implies_a_zero 
  (h : ∀ (x y : ℝ), (y^2 = x^2 + a*x + b ∧ x^2 = y^2 + a*y + b) → False) :
  a = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_zero_l573_57361


namespace NUMINAMATH_CALUDE_difference_squared_l573_57327

theorem difference_squared (x y a b : ℝ) 
  (h1 : x * y = b) 
  (h2 : x / y + y / x = a) : 
  (x - y)^2 = a * b - 2 * b := by
sorry

end NUMINAMATH_CALUDE_difference_squared_l573_57327


namespace NUMINAMATH_CALUDE_right_angled_triangle_obtuse_triangle_l573_57342

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = 180

-- Define types of triangles
def is_right_angled (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

def is_obtuse (t : Triangle) : Prop :=
  t.A > 90 ∨ t.B > 90 ∨ t.C > 90

-- Theorem for the first part
theorem right_angled_triangle (t : Triangle) (h1 : t.A = 30) (h2 : t.B = 60) :
  is_right_angled t :=
by sorry

-- Theorem for the second part
theorem obtuse_triangle (t : Triangle) (h : t.A / t.B = 1 / 3 ∧ t.B / t.C = 3 / 5) :
  is_obtuse t :=
by sorry

end NUMINAMATH_CALUDE_right_angled_triangle_obtuse_triangle_l573_57342


namespace NUMINAMATH_CALUDE_g_sum_equals_two_l573_57323

-- Define the function g
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^6 + b * x^4 - c * x^2 + 5

-- State the theorem
theorem g_sum_equals_two (a b c : ℝ) :
  g a b c 11 = 1 → g a b c 11 + g a b c (-11) = 2 := by
sorry

end NUMINAMATH_CALUDE_g_sum_equals_two_l573_57323


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l573_57391

theorem sum_of_three_numbers (A B C : ℝ) 
  (sum_eq : A + B + C = 2017)
  (A_eq : A = 2 * B - 3)
  (B_eq : B = 3 * C + 20) :
  A = 1213 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l573_57391


namespace NUMINAMATH_CALUDE_icecream_cost_theorem_l573_57374

def chapati_count : ℕ := 16
def rice_count : ℕ := 5
def vegetable_count : ℕ := 7
def icecream_count : ℕ := 6

def chapati_cost : ℕ := 6
def rice_cost : ℕ := 45
def vegetable_cost : ℕ := 70

def total_paid : ℕ := 1015

theorem icecream_cost_theorem : 
  (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) / icecream_count = 34 := by
  sorry

end NUMINAMATH_CALUDE_icecream_cost_theorem_l573_57374


namespace NUMINAMATH_CALUDE_two_digit_divisor_of_2701_l573_57362

/-- Two-digit number type -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n ≤ 99 }

/-- Sum of squares of digits of a two-digit number -/
def sumOfSquaresOfDigits (x : TwoDigitNumber) : ℕ :=
  let tens := x.val / 10
  let ones := x.val % 10
  tens * tens + ones * ones

/-- Main theorem -/
theorem two_digit_divisor_of_2701 (x : TwoDigitNumber) : 
  (2701 % x.val = 0) ↔ (sumOfSquaresOfDigits x = 58) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_divisor_of_2701_l573_57362


namespace NUMINAMATH_CALUDE_existence_of_three_integers_l573_57313

theorem existence_of_three_integers : ∃ (a b c : ℤ), 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b + c = 0 ∧
  ∃ (n : ℕ), a^13 + b^13 + c^13 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_three_integers_l573_57313


namespace NUMINAMATH_CALUDE_ellipse_circle_tangent_l573_57368

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in slope-intercept form -/
structure Line where
  k : ℝ
  m : ℝ

def ellipse_equation (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def line_equation (l : Line) (p : Point) : Prop :=
  p.y = l.k * p.x + l.m

def perpendicular (p1 p2 : Point) : Prop :=
  p1.x * p2.x + p1.y * p2.y = 0

theorem ellipse_circle_tangent (e : Ellipse) (a : Point) (l : Line) :
  ellipse_equation e a ∧ 
  a.x = 2 ∧ a.y = Real.sqrt 2 ∧
  ∃ (p q : Point),
    ellipse_equation e p ∧
    ellipse_equation e q ∧
    line_equation l p ∧
    line_equation l q ∧
    perpendicular p q →
  ∃ (r : ℝ), r = Real.sqrt (8/3) ∧
    ∀ (x y : ℝ), x^2 + y^2 = r^2 →
    ∃ (t : Point), line_equation l t ∧ t.x^2 + t.y^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_circle_tangent_l573_57368


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l573_57380

def U : Set Nat := {1,2,3,4,5,6}
def P : Set Nat := {1,3,5}
def Q : Set Nat := {1,2,4}

theorem complement_P_intersect_Q :
  (U \ P) ∩ Q = {2,4} := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l573_57380


namespace NUMINAMATH_CALUDE_carbon_atoms_in_compound_l573_57385

/-- Represents the number of atoms of each element in a compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (carbonWeight oxygenWeight hydrogenWeight : ℕ) : ℕ :=
  c.carbon * carbonWeight + c.hydrogen * hydrogenWeight + c.oxygen * oxygenWeight

/-- Theorem: A compound with 4 Hydrogen and 2 Oxygen atoms, and molecular weight 60,
    must have 2 Carbon atoms -/
theorem carbon_atoms_in_compound (c : Compound) 
    (h1 : c.hydrogen = 4)
    (h2 : c.oxygen = 2)
    (h3 : molecularWeight c 12 16 1 = 60) :
    c.carbon = 2 := by
  sorry

end NUMINAMATH_CALUDE_carbon_atoms_in_compound_l573_57385


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l573_57397

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l573_57397


namespace NUMINAMATH_CALUDE_absolute_value_equals_sqrt_of_square_l573_57348

theorem absolute_value_equals_sqrt_of_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_sqrt_of_square_l573_57348


namespace NUMINAMATH_CALUDE_correct_product_l573_57341

theorem correct_product (a b c : ℚ) (h1 : a = 0.25) (h2 : b = 3.4) (h3 : c = 0.85) 
  (h4 : (25 : ℤ) * 34 = 850) : a * b = c := by
  sorry

end NUMINAMATH_CALUDE_correct_product_l573_57341


namespace NUMINAMATH_CALUDE_complex_expression_equality_l573_57381

theorem complex_expression_equality : 
  (64 : ℝ) ^ (1/3) - 4 * Real.cos (45 * π / 180) + (1 - Real.sqrt 3) ^ 0 - abs (-Real.sqrt 2) = 5 - 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l573_57381


namespace NUMINAMATH_CALUDE_prove_train_car_capacity_l573_57372

/-- The number of passengers a 747 airplane can carry -/
def airplane_capacity : ℕ := 366

/-- The number of cars in the train -/
def train_cars : ℕ := 16

/-- The additional passengers a train can carry compared to 2 airplanes -/
def additional_passengers : ℕ := 228

/-- The number of passengers a single train car can carry -/
def train_car_capacity : ℕ := 60

theorem prove_train_car_capacity : 
  train_car_capacity * train_cars = 2 * airplane_capacity + additional_passengers :=
sorry

end NUMINAMATH_CALUDE_prove_train_car_capacity_l573_57372


namespace NUMINAMATH_CALUDE_triangle_tangent_ratio_l573_57312

theorem triangle_tangent_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π/2 →
  0 < B → B < π/2 →
  0 < C → C < π/2 →
  (1/a * b + 1/b * a) = 6 * Real.cos C →
  (Real.tan C / Real.tan A) + (Real.tan C / Real.tan B) = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_ratio_l573_57312


namespace NUMINAMATH_CALUDE_other_diagonal_length_l573_57364

/-- A trapezoid with diagonals intersecting at a right angle -/
structure RightAngleDiagonalTrapezoid where
  midline : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  diagonals_perpendicular : diagonal1 * diagonal2 = midline * midline * 4

/-- Theorem: In a trapezoid with diagonals intersecting at a right angle,
    if the midline is 6.5 and one diagonal is 12, then the other diagonal is 5 -/
theorem other_diagonal_length
  (t : RightAngleDiagonalTrapezoid)
  (h1 : t.midline = 6.5)
  (h2 : t.diagonal1 = 12) :
  t.diagonal2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l573_57364


namespace NUMINAMATH_CALUDE_savings_ratio_l573_57318

theorem savings_ratio (initial_savings : ℝ) (may_savings : ℝ) :
  initial_savings = 10 →
  may_savings = 160 →
  ∃ (r : ℝ), r > 0 ∧ may_savings = initial_savings * r^4 →
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_savings_ratio_l573_57318


namespace NUMINAMATH_CALUDE_missing_panels_l573_57395

/-- Calculates the number of missing solar panels in Faith's neighborhood. -/
theorem missing_panels (total_homes : Nat) (panels_per_home : Nat) (homes_with_panels : Nat) :
  total_homes = 20 →
  panels_per_home = 10 →
  homes_with_panels = 15 →
  total_homes * panels_per_home - homes_with_panels * panels_per_home = 50 :=
by
  sorry

#check missing_panels

end NUMINAMATH_CALUDE_missing_panels_l573_57395


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l573_57370

/-- The number of integers satisfying (x - 2)^2 ≤ 4 is 5 -/
theorem count_integers_satisfying_inequality : 
  (Finset.filter (fun x => (x - 2)^2 ≤ 4) (Finset.range 100)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l573_57370


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_400_l573_57347

theorem largest_multiple_of_15_under_400 : 
  ∀ n : ℕ, n * 15 < 400 → n * 15 ≤ 390 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_400_l573_57347


namespace NUMINAMATH_CALUDE_greatest_multiple_under_1000_l573_57394

theorem greatest_multiple_under_1000 : ∃ (n : ℕ), n = 990 ∧ 
  (∀ m : ℕ, m < 1000 → m % 5 = 0 → m % 6 = 0 → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_under_1000_l573_57394


namespace NUMINAMATH_CALUDE_joe_new_average_l573_57393

/-- Calculates the new average score after dropping the lowest score -/
def new_average (num_tests : ℕ) (original_average : ℚ) (lowest_score : ℚ) : ℚ :=
  (num_tests * original_average - lowest_score) / (num_tests - 1)

/-- Theorem: Given Joe's test scores, his new average after dropping the lowest score is 95 -/
theorem joe_new_average :
  let num_tests : ℕ := 4
  let original_average : ℚ := 90
  let lowest_score : ℚ := 75
  new_average num_tests original_average lowest_score = 95 := by
sorry

end NUMINAMATH_CALUDE_joe_new_average_l573_57393


namespace NUMINAMATH_CALUDE_xy_value_when_sum_of_abs_is_zero_l573_57344

theorem xy_value_when_sum_of_abs_is_zero (x y : ℝ) :
  |x - 1| + |y + 2| = 0 → x * y = -2 := by
sorry

end NUMINAMATH_CALUDE_xy_value_when_sum_of_abs_is_zero_l573_57344


namespace NUMINAMATH_CALUDE_galya_number_l573_57392

theorem galya_number (N : ℕ) : (∀ k : ℚ, (k * N + N) / N - N = k - 7729) → N = 7730 := by
  sorry

end NUMINAMATH_CALUDE_galya_number_l573_57392


namespace NUMINAMATH_CALUDE_rectangle_areas_sum_l573_57387

theorem rectangle_areas_sum : 
  let width : ℝ := 2
  let lengths : List ℝ := [1, 8, 27]
  let areas : List ℝ := lengths.map (λ l => width * l)
  areas.sum = 72 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_areas_sum_l573_57387


namespace NUMINAMATH_CALUDE_macaron_difference_l573_57336

/-- The number of macarons made by each person and given to kids --/
structure MacaronProblem where
  mitch : ℕ
  joshua : ℕ
  miles : ℕ
  renz : ℕ
  kids : ℕ
  macarons_per_kid : ℕ

/-- The conditions of the macaron problem --/
def validMacaronProblem (p : MacaronProblem) : Prop :=
  p.mitch = 20 ∧
  p.joshua = p.miles / 2 ∧
  p.joshua > p.mitch ∧
  p.renz = (3 * p.miles) / 4 - 1 ∧
  p.kids = 68 ∧
  p.macarons_per_kid = 2 ∧
  p.mitch + p.joshua + p.miles + p.renz = p.kids * p.macarons_per_kid

/-- The theorem stating the difference between Joshua's and Mitch's macarons --/
theorem macaron_difference (p : MacaronProblem) (h : validMacaronProblem p) :
  p.joshua - p.mitch = 27 := by
  sorry

end NUMINAMATH_CALUDE_macaron_difference_l573_57336


namespace NUMINAMATH_CALUDE_weight_lifting_problem_l573_57378

theorem weight_lifting_problem (total_weight first_lift second_lift : ℕ) : 
  total_weight = 1500 →
  2 * first_lift = second_lift + 300 →
  first_lift + second_lift = total_weight →
  first_lift = 600 := by
sorry

end NUMINAMATH_CALUDE_weight_lifting_problem_l573_57378


namespace NUMINAMATH_CALUDE_equation_solution_l573_57330

theorem equation_solution (x : ℝ) (h : x ≠ 2/3) :
  (6*x + 2) / (3*x^2 + 6*x - 4) = 3*x / (3*x - 2) ↔ x = 1 / Real.sqrt 3 ∨ x = -1 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l573_57330


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l573_57326

theorem similar_triangles_leg_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (a * b / 2 = 18) →  -- Area of smaller triangle
  (c * d / 2 = 288) →  -- Area of larger triangle
  (a^2 + b^2 = 10^2) →  -- Pythagorean theorem for smaller triangle
  (c / a = d / b) →  -- Similar triangles condition
  (c + d = 52) := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l573_57326


namespace NUMINAMATH_CALUDE_floor_inequality_iff_equal_l573_57360

theorem floor_inequality_iff_equal (m n : ℕ+) :
  (∀ α β : ℝ, ⌊(m + n : ℝ) * α⌋ + ⌊(m + n : ℝ) * β⌋ ≥ ⌊(m : ℝ) * α⌋ + ⌊(m : ℝ) * β⌋ + ⌊(n : ℝ) * (α + β)⌋) ↔
  m = n :=
by sorry

end NUMINAMATH_CALUDE_floor_inequality_iff_equal_l573_57360


namespace NUMINAMATH_CALUDE_ab_squared_equals_twelve_l573_57325

theorem ab_squared_equals_twelve (a b : ℝ) : (a + 2)^2 + |b - 3| = 0 → a^2 * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ab_squared_equals_twelve_l573_57325


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l573_57317

theorem arithmetic_mean_problem (x : ℕ) : 
  let numbers := [3, 117, 915, 138, 1917, 2114, x]
  (numbers.sum % 7 = 7) →
  (numbers.sum / numbers.length : ℚ) = 745 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l573_57317


namespace NUMINAMATH_CALUDE_kenneth_distance_past_finish_line_l573_57315

/-- Proves that Kenneth will be 10 yards past the finish line when Biff crosses the finish line in a 500-yard race -/
theorem kenneth_distance_past_finish_line 
  (race_distance : ℝ) 
  (biff_speed : ℝ) 
  (kenneth_speed : ℝ) 
  (h1 : race_distance = 500) 
  (h2 : biff_speed = 50) 
  (h3 : kenneth_speed = 51) : 
  kenneth_speed * (race_distance / biff_speed) - race_distance = 10 := by
  sorry

end NUMINAMATH_CALUDE_kenneth_distance_past_finish_line_l573_57315


namespace NUMINAMATH_CALUDE_square_perimeter_l573_57376

/-- Given a square cut into four equal rectangles, where each rectangle's length is four times
    its width, and these rectangles are arranged to form a shape with perimeter 56,
    prove that the perimeter of the original square is 32. -/
theorem square_perimeter (x : ℝ) : 
  x > 0 →  -- width of each rectangle is positive
  (4 * x) > 0 →  -- length of each rectangle is positive
  (28 * x) = 56 →  -- perimeter of the P shape
  (4 * (4 * x)) = 32  -- perimeter of the original square
  := by sorry

end NUMINAMATH_CALUDE_square_perimeter_l573_57376


namespace NUMINAMATH_CALUDE_product_of_five_integers_l573_57314

theorem product_of_five_integers (E F G H I : ℕ) : 
  E > 0 → F > 0 → G > 0 → H > 0 → I > 0 →
  E + F + G + H + I = 80 →
  E + 2 = F - 2 →
  E + 2 = G * 2 →
  E + 2 = H * 3 →
  E + 2 = I / 2 →
  E * F * G * H * I = 5120000 / 81 := by
sorry

end NUMINAMATH_CALUDE_product_of_five_integers_l573_57314


namespace NUMINAMATH_CALUDE_convex_hull_of_37gons_has_at_least_37_sides_l573_57367

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A set of regular polygons -/
def SetOfPolygons (n : ℕ) := Set (RegularPolygon n)

/-- The convex hull of a set of points in ℝ² -/
def ConvexHull (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- The number of sides in the convex hull of a set of points -/
def NumSides (S : Set (ℝ × ℝ)) : ℕ := sorry

/-- The vertices of all polygons in a set -/
def AllVertices (S : SetOfPolygons n) : Set (ℝ × ℝ) := sorry

/-- Theorem: The convex hull of any set of regular 37-gons has at least 37 sides -/
theorem convex_hull_of_37gons_has_at_least_37_sides (S : SetOfPolygons 37) :
  NumSides (ConvexHull (AllVertices S)) ≥ 37 := by sorry

end NUMINAMATH_CALUDE_convex_hull_of_37gons_has_at_least_37_sides_l573_57367


namespace NUMINAMATH_CALUDE_cube_64_sqrt_is_plus_minus_2_l573_57358

theorem cube_64_sqrt_is_plus_minus_2 (x : ℝ) (h : x^3 = 64) : 
  Real.sqrt x = 2 ∨ Real.sqrt x = -2 := by
sorry

end NUMINAMATH_CALUDE_cube_64_sqrt_is_plus_minus_2_l573_57358


namespace NUMINAMATH_CALUDE_prob_fewer_tails_eight_coins_l573_57354

/-- The number of coins flipped -/
def n : ℕ := 8

/-- The probability of getting fewer tails than heads when flipping n coins -/
def prob_fewer_tails (n : ℕ) : ℚ :=
  (1 - (n.choose (n / 2) : ℚ) / 2^n) / 2

theorem prob_fewer_tails_eight_coins : 
  prob_fewer_tails n = 93 / 256 := by
  sorry

end NUMINAMATH_CALUDE_prob_fewer_tails_eight_coins_l573_57354


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l573_57332

/-- The function f(x) = e^x - ln(x+m) is monotonically increasing on [0,1] iff m ≥ 1 -/
theorem monotone_increasing_condition (m : ℝ) :
  (∀ x ∈ Set.Icc 0 1, MonotoneOn (fun x => Real.exp x - Real.log (x + m)) (Set.Icc 0 1)) ↔ m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l573_57332


namespace NUMINAMATH_CALUDE_outlet_pipe_time_l573_57337

theorem outlet_pipe_time (inlet1 inlet2 outlet : ℚ) 
  (h1 : inlet1 = 1 / 18)
  (h2 : inlet2 = 1 / 20)
  (h3 : inlet1 + inlet2 - outlet = 1 / 12) :
  outlet = 1 / 45 := by
  sorry

end NUMINAMATH_CALUDE_outlet_pipe_time_l573_57337


namespace NUMINAMATH_CALUDE_complement_intersection_l573_57345

def U : Set ℕ := {x | 0 < x ∧ x < 7}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {1, 4}

theorem complement_intersection :
  (U \ A) ∩ (U \ B) = {6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_l573_57345


namespace NUMINAMATH_CALUDE_remaining_distance_l573_57383

/-- Calculates the remaining distance in a bike course -/
theorem remaining_distance (total_course : ℝ) (before_break : ℝ) (after_break : ℝ) :
  total_course = 10.5 ∧ before_break = 1.5 ∧ after_break = 3.73 →
  (total_course - (before_break + after_break)) * 1000 = 5270 := by
  sorry

#check remaining_distance

end NUMINAMATH_CALUDE_remaining_distance_l573_57383


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l573_57340

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- Theorem stating that f is monotonically increasing on ℝ
theorem f_monotone_increasing : Monotone f := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l573_57340


namespace NUMINAMATH_CALUDE_modulus_of_purely_imaginary_complex_l573_57365

/-- If z is a purely imaginary complex number of the form a^2 - 1 + (a + 1)i where a is real,
    then the modulus of z is 2. -/
theorem modulus_of_purely_imaginary_complex (a : ℝ) :
  let z : ℂ := a^2 - 1 + (a + 1) * I
  (z.re = 0) → Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_purely_imaginary_complex_l573_57365


namespace NUMINAMATH_CALUDE_total_distance_walked_and_run_l573_57359

/-- Calculates the total distance traveled when walking and running at given rates and times. -/
theorem total_distance_walked_and_run
  (walking_time : ℝ) (walking_rate : ℝ) (running_time : ℝ) (running_rate : ℝ) :
  walking_time = 30 / 60 →
  walking_rate = 3.5 →
  running_time = 45 / 60 →
  running_rate = 8 →
  walking_time * walking_rate + running_time * running_rate = 7.75 := by
  sorry

#check total_distance_walked_and_run

end NUMINAMATH_CALUDE_total_distance_walked_and_run_l573_57359


namespace NUMINAMATH_CALUDE_missing_number_is_five_l573_57389

/-- Represents the sum of two adjacent children's favorite numbers -/
structure AdjacentSum :=
  (value : ℕ)

/-- Represents a circle of children with their favorite numbers -/
structure ChildrenCircle :=
  (size : ℕ)
  (sums : List AdjacentSum)

/-- Calculates the missing number in the circle -/
def calculateMissingNumber (circle : ChildrenCircle) : ℕ :=
  sorry

/-- Theorem stating that the missing number is 5 -/
theorem missing_number_is_five (circle : ChildrenCircle) 
  (h1 : circle.size = 6)
  (h2 : circle.sums = [⟨8⟩, ⟨14⟩, ⟨12⟩])
  : calculateMissingNumber circle = 5 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_is_five_l573_57389


namespace NUMINAMATH_CALUDE_ram_distances_l573_57398

/-- Represents a mountain on the map -/
structure Mountain where
  name : String
  scale : ℝ  -- km per inch

/-- Represents a location on the map -/
structure Location where
  name : String
  distanceA : ℝ  -- distance from mountain A in inches
  distanceB : ℝ  -- distance from mountain B in inches

def map_distance : ℝ := 312  -- inches
def actual_distance : ℝ := 136  -- km

def mountainA : Mountain := { name := "A", scale := 1 }
def mountainB : Mountain := { name := "B", scale := 2 }

def ram_location : Location := { name := "Ram", distanceA := 25, distanceB := 40 }

/-- Calculates the actual distance from a location to a mountain -/
def actual_distance_to_mountain (loc : Location) (m : Mountain) : ℝ :=
  if m.name = "A" then loc.distanceA * m.scale else loc.distanceB * m.scale

theorem ram_distances :
  actual_distance_to_mountain ram_location mountainA = 25 ∧
  actual_distance_to_mountain ram_location mountainB = 80 := by
  sorry

end NUMINAMATH_CALUDE_ram_distances_l573_57398


namespace NUMINAMATH_CALUDE_power_sum_equality_implies_product_equality_l573_57334

theorem power_sum_equality_implies_product_equality
  (a : ℝ) (m n p q : ℕ)
  (h_a_nonzero : a ≠ 0)
  (h_a_not_one : a ≠ 1)
  (h_a_not_neg_one : a ≠ -1)
  (h_eq1 : a^m + a^n = a^p + a^q)
  (h_eq2 : a^(3*m) + a^(3*n) = a^(3*p) + a^(3*q)) :
  m * n = p * q := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_implies_product_equality_l573_57334


namespace NUMINAMATH_CALUDE_f_pi_8_equals_sqrt_2_l573_57328

noncomputable def f (x : ℝ) : ℝ := 
  1 / (2 * Real.tan x) + (Real.sin (x/2) * Real.cos (x/2)) / (2 * Real.cos (x/2)^2 - 1)

theorem f_pi_8_equals_sqrt_2 : f (π/8) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_f_pi_8_equals_sqrt_2_l573_57328


namespace NUMINAMATH_CALUDE_extra_birds_calculation_l573_57355

structure BirdPopulation where
  totalBirds : Nat
  sparrows : Nat
  robins : Nat
  bluebirds : Nat
  totalNests : Nat
  sparrowNests : Nat
  robinNests : Nat
  bluebirdNests : Nat

def extraBirds (bp : BirdPopulation) : Nat :=
  (bp.sparrows - bp.sparrowNests) + (bp.robins - bp.robinNests) + (bp.bluebirds - bp.bluebirdNests)

theorem extra_birds_calculation (bp : BirdPopulation) 
  (h1 : bp.totalBirds = bp.sparrows + bp.robins + bp.bluebirds)
  (h2 : bp.totalNests = bp.sparrowNests + bp.robinNests + bp.bluebirdNests)
  (h3 : bp.totalBirds = 18) (h4 : bp.sparrows = 10) (h5 : bp.robins = 5) (h6 : bp.bluebirds = 3)
  (h7 : bp.totalNests = 8) (h8 : bp.sparrowNests = 4) (h9 : bp.robinNests = 2) (h10 : bp.bluebirdNests = 2) :
  extraBirds bp = 10 := by
  sorry

end NUMINAMATH_CALUDE_extra_birds_calculation_l573_57355


namespace NUMINAMATH_CALUDE_M_intersect_N_l573_57322

def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

theorem M_intersect_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_l573_57322


namespace NUMINAMATH_CALUDE_min_towns_for_22_routes_l573_57373

/-- A graph representing a country's airline network -/
structure AirlineNetwork where
  towns : Finset ℕ
  connections : towns → towns → Bool
  paid_direction : towns → towns → Bool

/-- The number of free routes between two towns in an airline network -/
def free_routes (g : AirlineNetwork) (a b : g.towns) : ℕ :=
  sorry

/-- The theorem stating that the minimum number of towns for 22 free routes is 7 -/
theorem min_towns_for_22_routes :
  ∃ (g : AirlineNetwork) (a b : g.towns),
    free_routes g a b = 22 ∧
    g.towns.card = 7 ∧
    (∀ (h : AirlineNetwork) (x y : h.towns),
      free_routes h x y = 22 → h.towns.card ≥ 7) :=
  sorry

end NUMINAMATH_CALUDE_min_towns_for_22_routes_l573_57373


namespace NUMINAMATH_CALUDE_jane_crayon_count_l573_57375

/-- The number of crayons Jane ends up with after various events -/
def final_crayon_count (initial_count : ℕ) (eaten : ℕ) (packs_bought : ℕ) (crayons_per_pack : ℕ) (broken : ℕ) : ℕ :=
  initial_count - eaten + packs_bought * crayons_per_pack - broken

/-- Theorem stating that Jane ends up with 127 crayons given the conditions -/
theorem jane_crayon_count :
  final_crayon_count 87 7 5 10 3 = 127 := by
  sorry

end NUMINAMATH_CALUDE_jane_crayon_count_l573_57375


namespace NUMINAMATH_CALUDE_sum_of_cubes_l573_57356

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = -1) (h2 : x * y = -1) : x^3 + y^3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l573_57356


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l573_57335

theorem line_passes_through_fixed_point (a : ℝ) (ha : a ≠ 0) :
  (a + 2) * 1 + (1 - a) * 1 - 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l573_57335


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l573_57351

/-- Given p, q, r are roots of x³ - 3x - 2 = 0, prove p(q - r)² + q(r - p)² + r(p - q)² = -6 -/
theorem cubic_roots_sum (p q r : ℝ) : 
  (p^3 - 3*p - 2 = 0) → 
  (q^3 - 3*q - 2 = 0) → 
  (r^3 - 3*r - 2 = 0) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l573_57351


namespace NUMINAMATH_CALUDE_next_perfect_square_sum_of_digits_l573_57339

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def begins_with_three_twos (n : ℕ) : Prop :=
  n ≥ 222000 ∧ n < 223000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem next_perfect_square_sum_of_digits :
  ∃ n : ℕ, is_perfect_square n ∧ 
           begins_with_three_twos n ∧
           (∀ m : ℕ, is_perfect_square m ∧ begins_with_three_twos m → n ≤ m) ∧
           sum_of_digits n = 18 :=
sorry

end NUMINAMATH_CALUDE_next_perfect_square_sum_of_digits_l573_57339


namespace NUMINAMATH_CALUDE_eds_pets_l573_57301

theorem eds_pets (dogs cats : ℕ) (h1 : dogs = 2) (h2 : cats = 3) : 
  let fish := 2 * (dogs + cats)
  dogs + cats + fish = 15 := by
  sorry

end NUMINAMATH_CALUDE_eds_pets_l573_57301


namespace NUMINAMATH_CALUDE_three_digit_power_ending_theorem_l573_57304

/-- A three-digit number is between 100 and 999, inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A number N satisfies the property if for all k ≥ 1, N^k ≡ N (mod 1000) -/
def SatisfiesProperty (N : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ 1 → N^k ≡ N [MOD 1000]

theorem three_digit_power_ending_theorem :
  ∀ N : ℕ, ThreeDigitNumber N → SatisfiesProperty N ↔ (N = 625 ∨ N = 376) :=
sorry

end NUMINAMATH_CALUDE_three_digit_power_ending_theorem_l573_57304


namespace NUMINAMATH_CALUDE_simplify_expression_l573_57369

theorem simplify_expression (b : ℝ) : (1 : ℝ) * (2 * b) * (3 * b^2) * (4 * b^3) * (6 * b^5) = 144 * b^11 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l573_57369


namespace NUMINAMATH_CALUDE_first_year_after_2000_sum_12_correct_l573_57303

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a year is after 2000 -/
def is_after_2000 (year : ℕ) : Prop := year > 2000

/-- The first year after 2000 with sum of digits 12 -/
def first_year_after_2000_sum_12 : ℕ := 2019

theorem first_year_after_2000_sum_12_correct :
  (is_after_2000 first_year_after_2000_sum_12) ∧
  (sum_of_digits first_year_after_2000_sum_12 = 12) ∧
  (∀ y : ℕ, is_after_2000 y ∧ sum_of_digits y = 12 → y ≥ first_year_after_2000_sum_12) :=
by sorry

end NUMINAMATH_CALUDE_first_year_after_2000_sum_12_correct_l573_57303


namespace NUMINAMATH_CALUDE_ella_gives_one_sixth_l573_57309

-- Define the initial cookie distribution
def initial_distribution (luke_cookies : ℚ) : ℚ × ℚ × ℚ :=
  (2 * luke_cookies, 4 * luke_cookies, luke_cookies)

-- Define the function to calculate the fraction Ella gives to Luke
def fraction_ella_gives (luke_cookies : ℚ) : ℚ :=
  let (ella_cookies, connor_cookies, luke_cookies) := initial_distribution luke_cookies
  let total_cookies := ella_cookies + connor_cookies + luke_cookies
  let equal_share := total_cookies / 3
  (equal_share - luke_cookies) / ella_cookies

-- Theorem statement
theorem ella_gives_one_sixth :
  ∀ (luke_cookies : ℚ), luke_cookies > 0 → fraction_ella_gives luke_cookies = 1/6 := by
  sorry


end NUMINAMATH_CALUDE_ella_gives_one_sixth_l573_57309


namespace NUMINAMATH_CALUDE_height_conversion_l573_57352

/-- Converts a height from inches to centimeters given the conversion factors. -/
def height_in_cm (height_in : ℚ) (in_per_ft : ℚ) (cm_per_ft : ℚ) : ℚ :=
  height_in * (cm_per_ft / in_per_ft)

/-- Theorem stating that 65 inches is equivalent to 162.5 cm given the conversion factors. -/
theorem height_conversion :
  height_in_cm 65 10 25 = 162.5 := by sorry

end NUMINAMATH_CALUDE_height_conversion_l573_57352


namespace NUMINAMATH_CALUDE_basketball_games_played_l573_57399

theorem basketball_games_played (x : ℕ) : 
  (3 : ℚ) / 4 * x + (1 : ℚ) / 4 * x = x ∧ 
  (2 : ℚ) / 3 * (x + 12) = (3 : ℚ) / 4 * x + 6 ∧ 
  (1 : ℚ) / 3 * (x + 12) = (1 : ℚ) / 4 * x + 6 → 
  x = 24 := by sorry

end NUMINAMATH_CALUDE_basketball_games_played_l573_57399


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l573_57396

theorem min_value_quadratic_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  (a + 1)^2 + 4 * b^2 + 9 * c^2 ≥ 144 / 49 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l573_57396


namespace NUMINAMATH_CALUDE_similar_triangle_shortest_side_l573_57305

theorem similar_triangle_shortest_side
  (a b c : ℝ)
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : a^2 + b^2 = c^2)  -- Pythagorean theorem for the first triangle
  (h3 : a = 15)           -- Length of one leg of the first triangle
  (h4 : c = 39)           -- Length of hypotenuse of the first triangle
  (k : ℝ)
  (h5 : k > 0)
  (h6 : k * c = 78)       -- Length of hypotenuse of the second triangle
  : k * a = 30            -- Length of the shortest side of the second triangle
:= by sorry

end NUMINAMATH_CALUDE_similar_triangle_shortest_side_l573_57305


namespace NUMINAMATH_CALUDE_sum_of_digits_of_squared_repeated_ones_l573_57310

/-- The number formed by repeating the digit '1' eight times -/
def repeated_ones : ℕ := 11111111

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_squared_repeated_ones : sum_of_digits (repeated_ones ^ 2) = 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_squared_repeated_ones_l573_57310


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_to_height_ratio_l573_57388

/-- A regular tetrahedron with an inscribed sphere -/
structure RegularTetrahedronWithInscribedSphere where
  /-- The height of the regular tetrahedron -/
  height : ℝ
  /-- The radius of the inscribed sphere -/
  sphereRadius : ℝ
  /-- The area of one face of the regular tetrahedron -/
  faceArea : ℝ
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The sphere radius is positive -/
  sphereRadius_pos : 0 < sphereRadius
  /-- The face area is positive -/
  faceArea_pos : 0 < faceArea
  /-- Volume relation between the tetrahedron and the four pyramids formed by the inscribed sphere -/
  volume_relation : 4 * (1/3 * faceArea * sphereRadius) = 1/3 * faceArea * height

/-- The ratio of the radius of the inscribed sphere to the height of the regular tetrahedron is 1/4 -/
theorem inscribed_sphere_radius_to_height_ratio 
  (t : RegularTetrahedronWithInscribedSphere) : t.sphereRadius = 1/4 * t.height := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_to_height_ratio_l573_57388


namespace NUMINAMATH_CALUDE_bandit_gem_distribution_theorem_l573_57316

/-- Represents the distribution of precious stones for a bandit -/
structure GemDistribution where
  rubies : ℕ
  sapphires : ℕ
  emeralds : ℕ
  sum_is_100 : rubies + sapphires + emeralds = 100

/-- The proposition to be proven -/
theorem bandit_gem_distribution_theorem (bandits : Finset GemDistribution) 
    (h : bandits.card = 102) :
  (∃ b1 b2 : GemDistribution, b1 ∈ bandits ∧ b2 ∈ bandits ∧ b1 ≠ b2 ∧
    b1.rubies = b2.rubies ∧ b1.sapphires = b2.sapphires ∧ b1.emeralds = b2.emeralds) ∨
  (∃ b1 b2 : GemDistribution, b1 ∈ bandits ∧ b2 ∈ bandits ∧ b1 ≠ b2 ∧
    b1.rubies ≠ b2.rubies ∧ b1.sapphires ≠ b2.sapphires ∧ b1.emeralds ≠ b2.emeralds) :=
by
  sorry

end NUMINAMATH_CALUDE_bandit_gem_distribution_theorem_l573_57316


namespace NUMINAMATH_CALUDE_sin_870_degrees_l573_57321

theorem sin_870_degrees : Real.sin (870 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_870_degrees_l573_57321


namespace NUMINAMATH_CALUDE_prob_two_students_same_section_l573_57384

/-- The probability of two specific students being selected and placed in the same section -/
theorem prob_two_students_same_section 
  (total_students : ℕ) 
  (selected_students : ℕ) 
  (num_sections : ℕ) 
  (section_capacity : ℕ) 
  (h1 : total_students = 100)
  (h2 : selected_students = 60)
  (h3 : num_sections = 3)
  (h4 : section_capacity = 20)
  (h5 : selected_students = num_sections * section_capacity) :
  (selected_students : ℚ) / total_students * 
  (selected_students - 1) / (total_students - 1) * 
  (section_capacity - 1) / (selected_students - 1) = 19 / 165 :=
sorry

end NUMINAMATH_CALUDE_prob_two_students_same_section_l573_57384


namespace NUMINAMATH_CALUDE_max_silver_tokens_l573_57363

/-- Represents the state of tokens Alex has -/
structure TokenState where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rules -/
inductive ExchangeRule
  | RedToSilver : ExchangeRule  -- 3 red for 1 silver and 2 blue
  | BlueToSilver : ExchangeRule -- 4 blue for 1 silver and 2 red

/-- Applies an exchange rule to a token state -/
def applyExchange (state : TokenState) (rule : ExchangeRule) : Option TokenState :=
  match rule with
  | ExchangeRule.RedToSilver =>
      if state.red ≥ 3 then
        some ⟨state.red - 3, state.blue + 2, state.silver + 1⟩
      else
        none
  | ExchangeRule.BlueToSilver =>
      if state.blue ≥ 4 then
        some ⟨state.red + 2, state.blue - 4, state.silver + 1⟩
      else
        none

/-- Checks if any exchange is possible -/
def canExchange (state : TokenState) : Bool :=
  state.red ≥ 3 ∨ state.blue ≥ 4

/-- The main theorem to prove -/
theorem max_silver_tokens (initialState : TokenState) 
    (h_initial : initialState = ⟨100, 100, 0⟩) :
    ∃ (finalState : TokenState), 
      (¬canExchange finalState) ∧ 
      (finalState.silver = 88) ∧
      (∃ (exchanges : List ExchangeRule), 
        finalState = exchanges.foldl (λ s r => (applyExchange s r).getD s) initialState) :=
  sorry


end NUMINAMATH_CALUDE_max_silver_tokens_l573_57363


namespace NUMINAMATH_CALUDE_trapezoid_area_l573_57320

-- Define the trapezoid
def trapezoid := {(x, y) : ℝ × ℝ | 0 ≤ x ∧ x ≤ 15 ∧ 10 ≤ y ∧ y ≤ 15 ∧ (y = x ∨ y = 10 ∨ y = 15)}

-- Define the area function
def area (T : Set (ℝ × ℝ)) : ℝ := 62.5

-- Theorem statement
theorem trapezoid_area : area trapezoid = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l573_57320


namespace NUMINAMATH_CALUDE_f_composition_minus_one_l573_57306

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_composition_minus_one : f (f (-1)) = 5 := by sorry

end NUMINAMATH_CALUDE_f_composition_minus_one_l573_57306
