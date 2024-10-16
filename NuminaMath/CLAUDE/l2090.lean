import Mathlib

namespace NUMINAMATH_CALUDE_product_of_numbers_l2090_209062

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 37) (h2 : x - y = 5) : x * y = 336 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2090_209062


namespace NUMINAMATH_CALUDE_factorization_problems_l2090_209040

variable (m x y : ℝ)

theorem factorization_problems :
  (mx^2 - m*y = m*(x^2 - y)) ∧
  (2*x^2 - 8*x + 8 = 2*(x-2)^2) ∧
  (x^2*(2*x-1) + y^2*(1-2*x) = (2*x-1)*(x+y)*(x-y)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l2090_209040


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l2090_209013

theorem incorrect_observation_value 
  (n : ℕ) 
  (original_mean : ℝ) 
  (new_mean : ℝ) 
  (correct_value : ℝ) 
  (h1 : n = 50) 
  (h2 : original_mean = 36) 
  (h3 : new_mean = 36.5) 
  (h4 : correct_value = 45) :
  ∃ (incorrect_value : ℝ), 
    (n : ℝ) * original_mean = (n : ℝ) * new_mean - correct_value + incorrect_value ∧ 
    incorrect_value = 20 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_observation_value_l2090_209013


namespace NUMINAMATH_CALUDE_complex_multiplication_problem_l2090_209021

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Definition of the complex number multiplication -/
def complex_mult (a b c d : ℝ) : ℂ := Complex.mk (a * c - b * d) (a * d + b * c)

/-- The problem statement -/
theorem complex_multiplication_problem :
  complex_mult 4 (-3) 4 3 = 25 := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_problem_l2090_209021


namespace NUMINAMATH_CALUDE_greatest_savings_l2090_209064

def plane_cost : ℚ := 600
def boat_cost : ℚ := 254
def helicopter_cost : ℚ := 850

def savings (cost1 cost2 : ℚ) : ℚ := max cost1 cost2 - min cost1 cost2

theorem greatest_savings :
  max (savings plane_cost boat_cost) (savings helicopter_cost boat_cost) = 596 :=
by sorry

end NUMINAMATH_CALUDE_greatest_savings_l2090_209064


namespace NUMINAMATH_CALUDE_custom_op_three_four_l2090_209030

/-- Custom binary operation * -/
def custom_op (a b : ℝ) : ℝ := 4*a + 5*b - a^2*b

/-- Theorem stating that 3 * 4 = -4 under the custom operation -/
theorem custom_op_three_four : custom_op 3 4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_three_four_l2090_209030


namespace NUMINAMATH_CALUDE_cards_difference_l2090_209051

theorem cards_difference (ann_cards : ℕ) (ann_heike_ratio : ℕ) (anton_heike_ratio : ℕ) :
  ann_cards = 60 →
  ann_heike_ratio = 6 →
  anton_heike_ratio = 3 →
  ann_cards - (anton_heike_ratio * (ann_cards / ann_heike_ratio)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_cards_difference_l2090_209051


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l2090_209074

def f (x : ℝ) := x^3 - 12*x

theorem max_value_of_f_on_interval :
  ∃ (M : ℝ), M = 16 ∧ ∀ x ∈ Set.Icc (-3) 3, f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l2090_209074


namespace NUMINAMATH_CALUDE_negation_equivalence_l2090_209037

theorem negation_equivalence :
  ¬(∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2090_209037


namespace NUMINAMATH_CALUDE_smallest_valid_strategy_l2090_209061

/-- Represents a 9x9 game board -/
def GameBoard := Fin 9 → Fin 9 → Bool

/-- Represents an L-shaped tromino -/
structure Tromino :=
  (x : Fin 9) (y : Fin 9) (orientation : Fin 4)

/-- Checks if a tromino covers a given cell -/
def covers (t : Tromino) (x y : Fin 9) : Bool :=
  sorry

/-- Checks if a marking strategy allows unique determination of tromino placement -/
def is_valid_strategy (board : GameBoard) : Prop :=
  ∀ t1 t2 : Tromino, t1 ≠ t2 →
    ∃ x y : Fin 9, board x y ∧ (covers t1 x y ≠ covers t2 x y)

/-- Counts the number of marked cells on the board -/
def count_marked (board : GameBoard) : Nat :=
  sorry

theorem smallest_valid_strategy :
  ∃ (board : GameBoard),
    is_valid_strategy board ∧
    count_marked board = 68 ∧
    ∀ (other_board : GameBoard),
      is_valid_strategy other_board →
      count_marked other_board ≥ 68 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_strategy_l2090_209061


namespace NUMINAMATH_CALUDE_outfit_count_l2090_209029

/-- Represents the colors available for clothing items -/
inductive Color
| Red | Black | Blue | Gray | Green | Purple | White

/-- Represents a clothing item -/
structure ClothingItem :=
  (color : Color)

/-- Represents an outfit -/
structure Outfit :=
  (shirt : ClothingItem)
  (pants : ClothingItem)
  (hat : ClothingItem)

def is_monochrome (outfit : Outfit) : Prop :=
  outfit.shirt.color = outfit.pants.color ∧ outfit.shirt.color = outfit.hat.color

def num_shirts : Nat := 8
def num_pants : Nat := 5
def num_hats : Nat := 7
def num_pants_colors : Nat := 5
def num_shirt_hat_colors : Nat := 7

theorem outfit_count :
  let total_outfits := num_shirts * num_pants * num_hats
  let monochrome_outfits := num_pants_colors
  (total_outfits - monochrome_outfits : Nat) = 275 := by sorry

end NUMINAMATH_CALUDE_outfit_count_l2090_209029


namespace NUMINAMATH_CALUDE_triangle_area_l2090_209023

/-- The area of a triangle with side lengths 3, 5, and 7 is equal to 15√3/4 -/
theorem triangle_area (a b c : ℝ) (h_a : a = 3) (h_b : b = 5) (h_c : c = 7) :
  (1/2) * b * c * Real.sqrt (1 - ((b^2 + c^2 - a^2) / (2*b*c))^2) = (15 * Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_l2090_209023


namespace NUMINAMATH_CALUDE_floor_neg_seven_fourths_l2090_209076

theorem floor_neg_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_neg_seven_fourths_l2090_209076


namespace NUMINAMATH_CALUDE_handshake_problem_l2090_209015

/-- The number of handshakes in a complete graph with n vertices -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: Given 435 handshakes, there are 30 men -/
theorem handshake_problem :
  ∃ (n : ℕ), n > 0 ∧ handshakes n = 435 ∧ n = 30 := by
  sorry

end NUMINAMATH_CALUDE_handshake_problem_l2090_209015


namespace NUMINAMATH_CALUDE_rectangle_properties_l2090_209087

/-- Rectangle with adjacent sides x and 4, and perimeter y -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- The perimeter of the rectangle is related to x -/
axiom perimeter_relation (rect : Rectangle) : rect.y = 2 * rect.x + 8

theorem rectangle_properties :
  ∀ (rect : Rectangle),
  (rect.x = 10 → rect.y = 28) ∧
  (rect.y = 30 → rect.x = 11) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_properties_l2090_209087


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l2090_209056

theorem quadratic_root_difference (r s : ℝ) (hr : r > 0) : 
  (∃ y1 y2 : ℝ, y1 ≠ y2 ∧ 
    y1^2 - r*y1 - s = 0 ∧ 
    y2^2 - r*y2 - s = 0 ∧ 
    |y1 - y2| = 2) → 
  r = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l2090_209056


namespace NUMINAMATH_CALUDE_compound_interest_problem_l2090_209098

/-- Proves that the principal amount is 20000 given the specified conditions --/
theorem compound_interest_problem (P : ℝ) : 
  P * (1 + 0.2 / 2)^4 - P * (1 + 0.2)^2 = 482 → P = 20000 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l2090_209098


namespace NUMINAMATH_CALUDE_power_of_three_expression_l2090_209073

theorem power_of_three_expression : 3^(1+2+3) - (3^1 + 3^2 + 3^4) = 636 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_expression_l2090_209073


namespace NUMINAMATH_CALUDE_simplify_expression_l2090_209069

theorem simplify_expression (a : ℝ) : 2*a*(3*a^2 - 4*a + 3) - 3*a^2*(2*a - 4) = 4*a^2 + 6*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2090_209069


namespace NUMINAMATH_CALUDE_max_min_f_l2090_209085

-- Define the function f
def f (x : ℝ) : ℝ := 6 - 12 * x + x^3

-- Define the interval
def I : Set ℝ := {x | -1/3 ≤ x ∧ x ≤ 1}

-- Statement of the theorem
theorem max_min_f :
  ∃ (max min : ℝ),
    (∀ x ∈ I, f x ≤ max) ∧
    (∃ x ∈ I, f x = max) ∧
    (∀ x ∈ I, min ≤ f x) ∧
    (∃ x ∈ I, f x = min) ∧
    max = 27 ∧
    min = -5 := by
  sorry

end NUMINAMATH_CALUDE_max_min_f_l2090_209085


namespace NUMINAMATH_CALUDE_exists_dividable_polyhedron_l2090_209099

/-- A face of a polyhedron -/
structure Face where
  -- Add necessary properties of a face

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  faces : Set Face
  -- Add necessary properties to ensure convexity

/-- A function that checks if a set of faces can form a convex polyhedron -/
def can_form_convex_polyhedron (faces : Set Face) : Prop :=
  ∃ (p : ConvexPolyhedron), p.faces = faces

/-- Theorem: There exists a convex polyhedron whose faces can be divided into two sets,
    each of which can form a convex polyhedron -/
theorem exists_dividable_polyhedron :
  ∃ (p : ConvexPolyhedron) (s₁ s₂ : Set Face),
    s₁ ∪ s₂ = p.faces ∧
    s₁ ∩ s₂ = ∅ ∧
    can_form_convex_polyhedron s₁ ∧
    can_form_convex_polyhedron s₂ :=
sorry

end NUMINAMATH_CALUDE_exists_dividable_polyhedron_l2090_209099


namespace NUMINAMATH_CALUDE_triangle_proof_l2090_209068

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define point D
variable (D : ℝ)

-- State the theorem
theorem triangle_proof :
  -- Given conditions
  (0 < A) ∧ (A < π) ∧
  (0 < B) ∧ (B < π) ∧
  (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  ((Real.cos (A + C) + Real.cos (A - C) - 1) / (Real.cos (A - B) + Real.cos C) = c / b) ∧
  (c = 2) ∧
  -- AD = 2DC condition
  (3 * D = A + 2 * C) ∧
  -- BD condition
  (b^2 / 9 + 4 * (D / 3)^2 - 4 * (D / 3) * (b / 3) * Real.cos A = 13 / 9) →
  -- Conclusions
  (B = 2 * π / 3) ∧ (b = 2 * Real.sqrt 7) :=
by sorry


end NUMINAMATH_CALUDE_triangle_proof_l2090_209068


namespace NUMINAMATH_CALUDE_f_min_value_max_b_times_a_plus_one_exists_max_b_times_a_plus_one_l2090_209034

noncomputable section

def f (x : ℝ) : ℝ := Real.exp x - x + (1/2) * x^2

def g (a b x : ℝ) : ℝ := (1/2) * x^2 + a * x + b

theorem f_min_value : ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = 3/2 :=
sorry

theorem max_b_times_a_plus_one (a b : ℝ) :
  (∀ x, f x ≥ g a b x) → b * (a + 1) ≤ Real.exp 1 / 2 :=
sorry

theorem exists_max_b_times_a_plus_one :
  ∃ (a b : ℝ), (∀ x, f x ≥ g a b x) ∧ b * (a + 1) = Real.exp 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_max_b_times_a_plus_one_exists_max_b_times_a_plus_one_l2090_209034


namespace NUMINAMATH_CALUDE_equation_solutions_l2090_209005

theorem equation_solutions : 
  ∃! (s : Set ℝ), s = {x : ℝ | (x + 3)^4 + (x + 1)^4 = 82} ∧ s = {0, -4} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2090_209005


namespace NUMINAMATH_CALUDE_age_gap_ratio_l2090_209027

/-- Given the birth years of family members, prove the ratio of age gaps -/
theorem age_gap_ratio (older_brother_birth : ℕ) (older_sister_birth : ℕ) (grandmother_birth : ℕ)
  (h1 : older_brother_birth = 1932)
  (h2 : older_sister_birth = 1936)
  (h3 : grandmother_birth = 1944) :
  (grandmother_birth - older_sister_birth) / (older_sister_birth - older_brother_birth) = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_gap_ratio_l2090_209027


namespace NUMINAMATH_CALUDE_same_function_absolute_value_and_square_root_l2090_209065

theorem same_function_absolute_value_and_square_root (x t : ℝ) : 
  (fun x => |x|) = (fun t => Real.sqrt (t^2)) :=
sorry

end NUMINAMATH_CALUDE_same_function_absolute_value_and_square_root_l2090_209065


namespace NUMINAMATH_CALUDE_same_color_probability_l2090_209035

def blue_socks : ℕ := 12
def gray_socks : ℕ := 10
def white_socks : ℕ := 8

def total_socks : ℕ := blue_socks + gray_socks + white_socks

def same_color_combinations : ℕ := 
  (blue_socks.choose 2) + (gray_socks.choose 2) + (white_socks.choose 2)

def total_combinations : ℕ := total_socks.choose 2

theorem same_color_probability : 
  (same_color_combinations : ℚ) / total_combinations = 139 / 435 := by sorry

end NUMINAMATH_CALUDE_same_color_probability_l2090_209035


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l2090_209003

theorem units_digit_of_7_power_2023 : (7^2023 : ℕ) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l2090_209003


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_points_l2090_209089

/-- The ellipse on which points A and B lie -/
def Ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The condition that OA is perpendicular to OB -/
def Perpendicular (A B : ℝ × ℝ) : Prop := A.1 * B.1 + A.2 * B.2 = 0

/-- The condition that P is on segment AB -/
def OnSegment (P A B : ℝ × ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

/-- The condition that OP is perpendicular to AB -/
def Perpendicular_OP_AB (O P A B : ℝ × ℝ) : Prop := 
  (P.1 - O.1) * (B.1 - A.1) + (P.2 - O.2) * (B.2 - A.2) = 0

/-- The main theorem -/
theorem ellipse_perpendicular_points 
  (A B : ℝ × ℝ) 
  (hA : Ellipse A.1 A.2) 
  (hB : Ellipse B.1 B.2) 
  (hPerp : Perpendicular A B)
  (P : ℝ × ℝ)
  (hP : OnSegment P A B)
  (hPPerp : Perpendicular_OP_AB (0, 0) P A B) :
  (1 / (A.1^2 + A.2^2) + 1 / (B.1^2 + B.2^2) = 13/36) ∧
  (P.1^2 + P.2^2 = (6 * Real.sqrt 13 / 13)^2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_points_l2090_209089


namespace NUMINAMATH_CALUDE_only_two_and_five_plus_25_square_l2090_209012

theorem only_two_and_five_plus_25_square (N : ℕ+) : 
  (∀ p : ℕ, Nat.Prime p → p ∣ N → (p = 2 ∨ p = 5)) →
  (∃ k : ℕ, N + 25 = k^2) →
  (N = 200 ∨ N = 2000) := by
sorry

end NUMINAMATH_CALUDE_only_two_and_five_plus_25_square_l2090_209012


namespace NUMINAMATH_CALUDE_crayons_per_child_l2090_209010

theorem crayons_per_child (total_crayons : ℕ) (num_children : ℕ) 
  (h1 : total_crayons = 72) (h2 : num_children = 12) :
  total_crayons / num_children = 6 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_child_l2090_209010


namespace NUMINAMATH_CALUDE_no_x4_term_implies_a_zero_l2090_209095

theorem no_x4_term_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, ∃ b c d : ℝ, -5 * x^3 * (x^2 + a * x + 5) = b * x^5 + c * x^3 + d) →
  a = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_x4_term_implies_a_zero_l2090_209095


namespace NUMINAMATH_CALUDE_line_segment_length_l2090_209088

/-- Given points P, Q, R, and S arranged in order on a line segment,
    with PQ = 1, QR = 2PQ, and RS = 3QR, prove that the length of PS is 9. -/
theorem line_segment_length (P Q R S : ℝ) : 
  P < Q ∧ Q < R ∧ R < S →  -- Points are arranged in order
  Q - P = 1 →              -- PQ = 1
  R - Q = 2 * (Q - P) →    -- QR = 2PQ
  S - R = 3 * (R - Q) →    -- RS = 3QR
  S - P = 9 :=             -- PS = 9
by sorry

end NUMINAMATH_CALUDE_line_segment_length_l2090_209088


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2090_209096

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- State the theorem
theorem union_of_M_and_N :
  M ∪ N = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2090_209096


namespace NUMINAMATH_CALUDE_company_demographics_l2090_209020

theorem company_demographics (total : ℕ) (total_pos : 0 < total) :
  let men_percent : ℚ := 48 / 100
  let union_percent : ℚ := 60 / 100
  let union_men_percent : ℚ := 70 / 100
  let men := (men_percent * total).floor
  let union := (union_percent * total).floor
  let union_men := (union_men_percent * union).floor
  let non_union := total - union
  let non_union_men := men - union_men
  let non_union_women := non_union - non_union_men
  (non_union_women : ℚ) / non_union = 85 / 100 :=
by sorry

end NUMINAMATH_CALUDE_company_demographics_l2090_209020


namespace NUMINAMATH_CALUDE_sqrt_49_is_7_l2090_209004

theorem sqrt_49_is_7 : Real.sqrt 49 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49_is_7_l2090_209004


namespace NUMINAMATH_CALUDE_combined_age_is_23_l2090_209072

/-- Represents the ages and relationships in the problem -/
structure AgeRelationship where
  person_age : ℕ
  dog_age : ℕ
  cat_age : ℕ
  sister_age : ℕ

/-- The conditions of the problem -/
def problem_conditions (ar : AgeRelationship) : Prop :=
  ar.person_age = ar.dog_age + 15 ∧
  ar.cat_age = ar.dog_age + 3 ∧
  ar.dog_age + 2 = 4 ∧
  ar.sister_age + 2 = 2 * (ar.dog_age + 2)

/-- The theorem to prove -/
theorem combined_age_is_23 (ar : AgeRelationship) 
  (h : problem_conditions ar) : 
  ar.person_age + ar.sister_age = 23 := by
  sorry

end NUMINAMATH_CALUDE_combined_age_is_23_l2090_209072


namespace NUMINAMATH_CALUDE_special_word_count_l2090_209059

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The length of the words we're considering --/
def word_length : ℕ := 5

/-- 
  Counts the number of five-letter words where:
  - The first and last letters are the same
  - The second and fourth letters are the same
--/
def count_special_words : ℕ := alphabet_size ^ 3

/-- 
  Theorem: The number of five-letter words with the given properties
  is equal to the cube of the alphabet size.
--/
theorem special_word_count :
  count_special_words = alphabet_size ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_special_word_count_l2090_209059


namespace NUMINAMATH_CALUDE_roberts_birth_year_l2090_209070

theorem roberts_birth_year (n : ℕ) : 
  (n + 1)^2 - n^2 = 89 → n^2 = 1936 := by
  sorry

end NUMINAMATH_CALUDE_roberts_birth_year_l2090_209070


namespace NUMINAMATH_CALUDE_pencil_length_l2090_209083

theorem pencil_length (black_fraction : Real) (white_fraction : Real) (blue_length : Real) :
  black_fraction = 1/8 →
  white_fraction = 1/2 →
  blue_length = 3.5 →
  ∃ (total_length : Real),
    total_length * black_fraction +
    (total_length - total_length * black_fraction) * white_fraction +
    blue_length = total_length ∧
    total_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_pencil_length_l2090_209083


namespace NUMINAMATH_CALUDE_largest_even_odd_two_digit_l2090_209055

-- Define the set of two-digit numbers
def TwoDigitNumbers : Set Nat := {n : Nat | 10 ≤ n ∧ n ≤ 99}

-- Define even numbers
def IsEven (n : Nat) : Prop := ∃ k : Nat, n = 2 * k

-- Define odd numbers
def IsOdd (n : Nat) : Prop := ∃ k : Nat, n = 2 * k + 1

-- Theorem statement
theorem largest_even_odd_two_digit :
  (∀ n ∈ TwoDigitNumbers, IsEven n → n ≤ 98) ∧
  (∃ n ∈ TwoDigitNumbers, IsEven n ∧ n = 98) ∧
  (∀ n ∈ TwoDigitNumbers, IsOdd n → n ≤ 99) ∧
  (∃ n ∈ TwoDigitNumbers, IsOdd n ∧ n = 99) :=
sorry

end NUMINAMATH_CALUDE_largest_even_odd_two_digit_l2090_209055


namespace NUMINAMATH_CALUDE_circle_properties_l2090_209042

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y + 3)^2 = 4

-- Define a point being inside a circle
def is_inside_circle (x y : ℝ) : Prop := x^2 + (y + 3)^2 < 4

-- Define the line y = x
def line_y_eq_x (x y : ℝ) : Prop := y = x

-- Theorem statement
theorem circle_properties :
  (is_inside_circle 1 (-2)) ∧
  (∀ x y : ℝ, line_y_eq_x x y → ¬ circle_C x y) := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l2090_209042


namespace NUMINAMATH_CALUDE_max_b_value_l2090_209093

def is_prime (n : ℕ) : Prop := sorry

theorem max_b_value (a b c : ℕ) : 
  (a * b * c = 720) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (c = 3) →
  is_prime a →
  is_prime b →
  is_prime c →
  (∀ x : ℕ, (1 < x) ∧ (x < b) ∧ is_prime x → x ≤ 3) →
  (b ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l2090_209093


namespace NUMINAMATH_CALUDE_complex_simplification_l2090_209054

theorem complex_simplification :
  let z : ℂ := (2 + Complex.I) / Complex.I
  z = 1 - 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l2090_209054


namespace NUMINAMATH_CALUDE_find_a_plus_c_l2090_209014

theorem find_a_plus_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 42)
  (h2 : b + d = 6)
  (h3 : b * d = 5) :
  a + c = 7 := by
sorry

end NUMINAMATH_CALUDE_find_a_plus_c_l2090_209014


namespace NUMINAMATH_CALUDE_tower_height_difference_l2090_209081

theorem tower_height_difference : 
  ∀ (h_clyde h_grace : ℕ), 
  h_grace = 8 * h_clyde → 
  h_grace = 40 → 
  h_grace - h_clyde = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_tower_height_difference_l2090_209081


namespace NUMINAMATH_CALUDE_square_2007_position_l2090_209060

-- Define the possible square positions
inductive SquarePosition
  | ABCD
  | DABC
  | CBAD
  | DCBA

-- Define the transformations
def rotate90Clockwise : SquarePosition → SquarePosition
  | SquarePosition.ABCD => SquarePosition.DABC
  | SquarePosition.DABC => SquarePosition.CBAD
  | SquarePosition.CBAD => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.ABCD

def reflectVertical : SquarePosition → SquarePosition
  | SquarePosition.ABCD => SquarePosition.DCBA
  | SquarePosition.DABC => SquarePosition.CBAD
  | SquarePosition.CBAD => SquarePosition.DABC
  | SquarePosition.DCBA => SquarePosition.ABCD

-- Define the transformation sequence
def transformSquare : Nat → SquarePosition → SquarePosition
  | 0, pos => pos
  | n + 1, pos => 
    if n % 2 == 0 
    then transformSquare n (rotate90Clockwise pos)
    else transformSquare n (reflectVertical pos)

-- Theorem to prove
theorem square_2007_position : 
  transformSquare 2007 SquarePosition.ABCD = SquarePosition.CBAD := by
  sorry

end NUMINAMATH_CALUDE_square_2007_position_l2090_209060


namespace NUMINAMATH_CALUDE_new_average_weight_l2090_209094

def original_team_size : ℕ := 7
def original_average_weight : ℚ := 76
def new_player1_weight : ℚ := 110
def new_player2_weight : ℚ := 60

theorem new_average_weight :
  let original_total_weight := original_team_size * original_average_weight
  let new_total_weight := original_total_weight + new_player1_weight + new_player2_weight
  let new_team_size := original_team_size + 2
  new_total_weight / new_team_size = 78 := by
sorry

end NUMINAMATH_CALUDE_new_average_weight_l2090_209094


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l2090_209086

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the basis vectors
variable (e₁ e₂ : V)

-- Define the points and vectors
variable (A B C D : V)
variable (k : ℝ)

-- State the theorem
theorem collinear_points_k_value
  (hAB : B - A = e₁ - e₂)
  (hBC : C - B = 3 • e₁ + 2 • e₂)
  (hCD : D - C = k • e₁ + 2 • e₂)
  (hCollinear : ∃ (t : ℝ), D - A = t • (C - A)) :
  k = 8 := by sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l2090_209086


namespace NUMINAMATH_CALUDE_square_cutting_existence_l2090_209091

theorem square_cutting_existence : ∃ (a b c S : ℝ), 
  a^2 + 3*b^2 + 5*c^2 = S^2 ∧ 
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ S > 0 := by
  sorry

end NUMINAMATH_CALUDE_square_cutting_existence_l2090_209091


namespace NUMINAMATH_CALUDE_exponent_transform_to_one_l2090_209053

theorem exponent_transform_to_one (a : ℝ) (h : a > 0) :
  ∀ x : ℝ, a^x = 1 ↔ x = 0 := by
sorry

end NUMINAMATH_CALUDE_exponent_transform_to_one_l2090_209053


namespace NUMINAMATH_CALUDE_fraction_denominator_problem_l2090_209052

theorem fraction_denominator_problem (y x : ℝ) (h1 : y > 0) 
  (h2 : (1 * y) / x + (3 * y) / 10 = 0.35 * y) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_problem_l2090_209052


namespace NUMINAMATH_CALUDE_tan_equality_solution_l2090_209024

theorem tan_equality_solution (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1500 * π / 180) →
  n = 60 ∨ n = -120 :=
by sorry

end NUMINAMATH_CALUDE_tan_equality_solution_l2090_209024


namespace NUMINAMATH_CALUDE_angle_B_in_triangle_l2090_209097

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem angle_B_in_triangle (t : Triangle) :
  t.a = 4 →
  t.b = 2 * Real.sqrt 2 →
  t.A = π / 4 →
  t.B = π / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_B_in_triangle_l2090_209097


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l2090_209007

theorem sum_of_coefficients_zero (x y : ℝ) : 
  (fun x y => (3 * x^2 - 5 * x * y + 2 * y^2)^5) 1 1 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l2090_209007


namespace NUMINAMATH_CALUDE_xy_plus_one_is_square_l2090_209045

theorem xy_plus_one_is_square (x y : ℕ) 
  (h : (1 : ℚ) / x + (1 : ℚ) / y = 1 / (x + 2) + 1 / (y - 2)) : 
  ∃ (n : ℤ), (x * y + 1 : ℤ) = n ^ 2 := by
sorry

end NUMINAMATH_CALUDE_xy_plus_one_is_square_l2090_209045


namespace NUMINAMATH_CALUDE_inverse_function_inequality_l2090_209001

open Set
open Function
open Real

noncomputable def f (x : ℝ) := -x * abs x

theorem inverse_function_inequality (h : Bijective f) 
  (h2 : ∀ x ∈ Icc (-2 : ℝ) 2, (invFun f) (x^2 + m) < f x) : 
  m > 12 := by sorry

end NUMINAMATH_CALUDE_inverse_function_inequality_l2090_209001


namespace NUMINAMATH_CALUDE_factorization_proof_l2090_209071

theorem factorization_proof (x y : ℝ) : x * y^2 - x = x * (y + 1) * (y - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2090_209071


namespace NUMINAMATH_CALUDE_correct_option_b_l2090_209019

variable (y : ℝ)

theorem correct_option_b (y : ℝ) :
  (-2 * y^3) * (-y) = 2 * y^4 ∧
  (-y^3) * (-y) ≠ -y ∧
  ((-2*y)^3) * (-y) ≠ -8 * y^4 ∧
  ((-y)^12) * (-y) ≠ -3 * y^13 :=
sorry

end NUMINAMATH_CALUDE_correct_option_b_l2090_209019


namespace NUMINAMATH_CALUDE_ceil_zero_exists_ceil_minus_self_eq_point_two_l2090_209041

-- Define the ceiling function [x)
noncomputable def ceil (x : ℝ) : ℤ :=
  Int.ceil x

-- Theorem 1: [0) = 1
theorem ceil_zero : ceil 0 = 1 := by sorry

-- Theorem 2: There exists an x such that [x) - x = 0.2
theorem exists_ceil_minus_self_eq_point_two :
  ∃ x : ℝ, (ceil x : ℝ) - x = 0.2 := by sorry

end NUMINAMATH_CALUDE_ceil_zero_exists_ceil_minus_self_eq_point_two_l2090_209041


namespace NUMINAMATH_CALUDE_min_broken_line_length_l2090_209028

/-- Given points A and C in the coordinate plane, and point B on the x-axis,
    the minimum length of the broken line ABC is 7.5 -/
theorem min_broken_line_length :
  let A : ℝ × ℝ := (-3, -4)
  let C : ℝ × ℝ := (1.5, -2)
  ∃ B : ℝ × ℝ, B.2 = 0 ∧
    ∀ B' : ℝ × ℝ, B'.2 = 0 →
      Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) +
      Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) ≤
      Real.sqrt ((B'.1 - A.1)^2 + (B'.2 - A.2)^2) +
      Real.sqrt ((C.1 - B'.1)^2 + (C.2 - B'.2)^2) ∧
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) +
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_min_broken_line_length_l2090_209028


namespace NUMINAMATH_CALUDE_product_of_tans_equals_two_l2090_209044

theorem product_of_tans_equals_two : (1 + Real.tan (1 * π / 180)) * (1 + Real.tan (44 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_tans_equals_two_l2090_209044


namespace NUMINAMATH_CALUDE_weight_selection_theorem_l2090_209048

theorem weight_selection_theorem (N : ℕ) :
  (∃ (S : Finset ℕ) (k : ℕ), 
    1 < k ∧ 
    k ≤ N ∧
    (∀ i ∈ S, 1 ≤ i ∧ i ≤ N) ∧
    S.card = k ∧
    (S.sum id) * (N - k + 1) = (N * (N + 1)) / 2) ↔ 
  (∃ m : ℕ, N + 1 = m^2) :=
by sorry

end NUMINAMATH_CALUDE_weight_selection_theorem_l2090_209048


namespace NUMINAMATH_CALUDE_product_purely_imaginary_l2090_209079

theorem product_purely_imaginary (x : ℝ) : 
  (Complex.I : ℂ).im * ((x + Complex.I) * ((x^2 + 1 : ℝ) + Complex.I) * ((x^2 + 2 : ℝ) + Complex.I)).re = 0 ↔ 
  x^4 + x^3 + x^2 + 2*x - 2 = 0 :=
by sorry

#check product_purely_imaginary

end NUMINAMATH_CALUDE_product_purely_imaginary_l2090_209079


namespace NUMINAMATH_CALUDE_optimal_distribution_maximizes_sum_l2090_209092

/-- Represents the distribution of blue balls between two boxes -/
structure Distribution where
  first_box : ℕ
  second_box : ℕ

/-- Calculates the sum of percentages of blue balls in each box -/
def sum_of_percentages (d : Distribution) : ℚ :=
  d.first_box / 24 + d.second_box / 23

/-- Checks if a distribution is valid given the total number of blue balls -/
def is_valid_distribution (d : Distribution) (total_blue : ℕ) : Prop :=
  d.first_box + d.second_box = total_blue ∧ d.first_box ≤ 24 ∧ d.second_box ≤ 23

theorem optimal_distribution_maximizes_sum :
  ∀ d : Distribution,
  is_valid_distribution d 25 →
  sum_of_percentages d ≤ sum_of_percentages { first_box := 2, second_box := 23 } :=
by sorry

end NUMINAMATH_CALUDE_optimal_distribution_maximizes_sum_l2090_209092


namespace NUMINAMATH_CALUDE_reggie_long_shots_l2090_209080

/-- Represents the number of points for each type of shot --/
inductive ShotType
  | layup : ShotType
  | freeThrow : ShotType
  | longShot : ShotType

def shotValue : ShotType → ℕ
  | ShotType.layup => 1
  | ShotType.freeThrow => 2
  | ShotType.longShot => 3

/-- Represents the number of shots made by each player --/
structure ShotsMade where
  layups : ℕ
  freeThrows : ℕ
  longShots : ℕ

def totalPoints (shots : ShotsMade) : ℕ :=
  shots.layups * shotValue ShotType.layup +
  shots.freeThrows * shotValue ShotType.freeThrow +
  shots.longShots * shotValue ShotType.longShot

theorem reggie_long_shots
  (reggie : ShotsMade)
  (reggie_brother : ShotsMade)
  (h1 : reggie.layups = 3)
  (h2 : reggie.freeThrows = 2)
  (h3 : reggie_brother.layups = 0)
  (h4 : reggie_brother.freeThrows = 0)
  (h5 : reggie_brother.longShots = 4)
  (h6 : totalPoints reggie + 2 = totalPoints reggie_brother) :
  reggie.longShots = 1 := by
  sorry

end NUMINAMATH_CALUDE_reggie_long_shots_l2090_209080


namespace NUMINAMATH_CALUDE_bike_sharing_growth_model_l2090_209043

/-- Represents the bike-sharing company's growth model -/
theorem bike_sharing_growth_model (x : ℝ) :
  let initial_bikes : ℕ := 1000
  let additional_bikes : ℕ := 440
  let growth_factor : ℝ := (1 + x)
  let months : ℕ := 2
  (initial_bikes : ℝ) * growth_factor ^ months = (initial_bikes : ℝ) + additional_bikes :=
by
  sorry

end NUMINAMATH_CALUDE_bike_sharing_growth_model_l2090_209043


namespace NUMINAMATH_CALUDE_sara_movie_purchase_cost_l2090_209000

/-- The amount Sara spent on buying a movie, given her other movie-related expenses --/
theorem sara_movie_purchase_cost (ticket_price : ℝ) (ticket_count : ℕ) 
  (rental_cost : ℝ) (total_spent : ℝ) (h1 : ticket_price = 10.62) 
  (h2 : ticket_count = 2) (h3 : rental_cost = 1.59) (h4 : total_spent = 36.78) : 
  total_spent - (ticket_price * ↑ticket_count + rental_cost) = 13.95 := by
  sorry

end NUMINAMATH_CALUDE_sara_movie_purchase_cost_l2090_209000


namespace NUMINAMATH_CALUDE_tshirt_packages_l2090_209032

theorem tshirt_packages (total_tshirts : ℕ) (tshirts_per_package : ℕ) (h1 : total_tshirts = 426) (h2 : tshirts_per_package = 6) :
  total_tshirts / tshirts_per_package = 71 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_packages_l2090_209032


namespace NUMINAMATH_CALUDE_adjusted_target_heart_rate_for_30_year_old_l2090_209008

/-- Calculates the adjusted target heart rate for a runner --/
def adjustedTargetHeartRate (age : ℕ) : ℕ :=
  let maxHeartRate : ℕ := 220 - age
  let initialTargetRate : ℚ := 0.7 * maxHeartRate
  let adjustment : ℚ := 0.1 * initialTargetRate
  let adjustedRate : ℚ := initialTargetRate + adjustment
  (adjustedRate + 0.5).floor.toNat

/-- Theorem stating that for a 30-year-old runner, the adjusted target heart rate is 146 bpm --/
theorem adjusted_target_heart_rate_for_30_year_old :
  adjustedTargetHeartRate 30 = 146 := by
  sorry

#eval adjustedTargetHeartRate 30

end NUMINAMATH_CALUDE_adjusted_target_heart_rate_for_30_year_old_l2090_209008


namespace NUMINAMATH_CALUDE_complement_A_in_U_l2090_209058

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define set A
def A : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem complement_A_in_U : 
  {x ∈ U | x ∉ A} = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l2090_209058


namespace NUMINAMATH_CALUDE_grape_rate_calculation_l2090_209066

/-- The rate per kg of grapes -/
def grape_rate : ℝ := 70

/-- The amount of grapes purchased in kg -/
def grape_amount : ℝ := 9

/-- The rate per kg of mangoes -/
def mango_rate : ℝ := 55

/-- The amount of mangoes purchased in kg -/
def mango_amount : ℝ := 9

/-- The total amount paid -/
def total_paid : ℝ := 1125

theorem grape_rate_calculation :
  grape_rate * grape_amount + mango_rate * mango_amount = total_paid :=
by sorry

end NUMINAMATH_CALUDE_grape_rate_calculation_l2090_209066


namespace NUMINAMATH_CALUDE_f_negative_a_l2090_209067

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 - x^2) - x) + 1

theorem f_negative_a (a : ℝ) (h : f a = 4) : f (-a) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_l2090_209067


namespace NUMINAMATH_CALUDE_stating_carlas_counting_problem_l2090_209002

/-- 
Theorem stating that there exists a positive integer solution for the number of tiles and books
that satisfies the equation from Carla's counting problem.
-/
theorem carlas_counting_problem :
  ∃ (T B : ℕ), T > 0 ∧ B > 0 ∧ 2 * T + 3 * B = 301 := by
  sorry

end NUMINAMATH_CALUDE_stating_carlas_counting_problem_l2090_209002


namespace NUMINAMATH_CALUDE_red_marble_probability_l2090_209050

/-- The probability of drawing exactly k red marbles out of n draws with replacement
    from a bag containing r red marbles and b blue marbles. -/
def probability (r b k n : ℕ) : ℚ :=
  (n.choose k) * ((r : ℚ) / (r + b : ℚ)) ^ k * ((b : ℚ) / (r + b : ℚ)) ^ (n - k)

/-- The probability of drawing exactly 4 red marbles out of 8 draws with replacement
    from a bag containing 8 red marbles and 4 blue marbles is equal to 1120/6561. -/
theorem red_marble_probability : probability 8 4 4 8 = 1120 / 6561 := by
  sorry

end NUMINAMATH_CALUDE_red_marble_probability_l2090_209050


namespace NUMINAMATH_CALUDE_line_properties_l2090_209075

-- Define the line l₁
def l₁ (m : ℝ) (x y : ℝ) : Prop :=
  (m + 1) * x - (m - 3) * y - 8 = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (2, 2)

-- Define the line l₂
def l₂ (m : ℝ) (x y : ℝ) : Prop :=
  (m + 1) * x - (m - 3) * y = 0

-- Define the maximized distance line
def max_distance_line (x y : ℝ) : Prop :=
  x + y = 0

theorem line_properties :
  (∀ m : ℝ, l₁ m (fixed_point.1) (fixed_point.2)) ∧
  (∃ m : ℝ, ∀ x y : ℝ, l₂ m x y ↔ max_distance_line x y) :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l2090_209075


namespace NUMINAMATH_CALUDE_annie_walk_distance_l2090_209016

/-- The number of blocks Annie walked from her house to the bus stop -/
def annie_walk : ℕ := sorry

/-- The number of blocks Annie rode on the bus each way -/
def bus_ride : ℕ := 7

/-- The total number of blocks Annie traveled -/
def total_distance : ℕ := 24

theorem annie_walk_distance : annie_walk = 5 := by
  have h1 : 2 * annie_walk + 2 * bus_ride = total_distance := sorry
  sorry

end NUMINAMATH_CALUDE_annie_walk_distance_l2090_209016


namespace NUMINAMATH_CALUDE_existence_of_special_real_l2090_209078

theorem existence_of_special_real : ∃ A : ℝ, ∀ n : ℕ, ∃ m : ℕ, (⌊A^n⌋ : ℤ) + 2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_real_l2090_209078


namespace NUMINAMATH_CALUDE_unfenced_side_length_is_ten_l2090_209017

/-- Represents a rectangular yard with fencing on three sides -/
structure FencedYard where
  length : ℝ
  width : ℝ
  area : ℝ
  fenceLength : ℝ

/-- The unfenced side length of a rectangular yard -/
def unfencedSideLength (yard : FencedYard) : ℝ := yard.length

/-- Theorem stating the conditions and the result to be proved -/
theorem unfenced_side_length_is_ten
  (yard : FencedYard)
  (area_constraint : yard.area = 200)
  (fence_constraint : yard.fenceLength = 50)
  (rectangle_constraint : yard.area = yard.length * yard.width)
  (fence_sides_constraint : yard.fenceLength = 2 * yard.width + yard.length) :
  unfencedSideLength yard = 10 := by sorry

end NUMINAMATH_CALUDE_unfenced_side_length_is_ten_l2090_209017


namespace NUMINAMATH_CALUDE_total_books_theorem_melanie_books_l2090_209009

/-- Calculates the total number of books after a purchase -/
def total_books_after_purchase (initial_books : ℕ) (books_bought : ℕ) : ℕ :=
  initial_books + books_bought

/-- Theorem: The total number of books after a purchase is the sum of initial books and books bought -/
theorem total_books_theorem (initial_books books_bought : ℕ) :
  total_books_after_purchase initial_books books_bought = initial_books + books_bought :=
by
  sorry

/-- Melanie's book collection problem -/
theorem melanie_books :
  total_books_after_purchase 41 46 = 87 :=
by
  sorry

end NUMINAMATH_CALUDE_total_books_theorem_melanie_books_l2090_209009


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l2090_209082

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A heptagon has 7 sides -/
def heptagon_sides : ℕ := 7

/-- Theorem: A convex heptagon has 14 diagonals -/
theorem heptagon_diagonals : num_diagonals heptagon_sides = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l2090_209082


namespace NUMINAMATH_CALUDE_f_geq_two_range_of_x_l2090_209063

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem 1: f(x) ≥ 2 for all real x
theorem f_geq_two (x : ℝ) : f x ≥ 2 := by
  sorry

-- Theorem 2: If f(x) ≥ (|2b+1| - |1-b|) / |b| for all non-zero real b,
-- then x ≤ -1.5 or x ≥ 1.5
theorem range_of_x (x : ℝ) 
  (h : ∀ b : ℝ, b ≠ 0 → f x ≥ (|2*b + 1| - |1 - b|) / |b|) : 
  x ≤ -1.5 ∨ x ≥ 1.5 := by
  sorry

end NUMINAMATH_CALUDE_f_geq_two_range_of_x_l2090_209063


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2090_209047

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 3 + a 4 + a 5 = 12 → a 1 + a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2090_209047


namespace NUMINAMATH_CALUDE_leila_earnings_proof_l2090_209046

-- Define the given conditions
def voltaire_daily_viewers : ℕ := 50
def leila_daily_viewers : ℕ := 2 * voltaire_daily_viewers
def earnings_per_view : ℚ := 1/2

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define Leila's weekly earnings
def leila_weekly_earnings : ℚ := leila_daily_viewers * earnings_per_view * days_in_week

-- Theorem statement
theorem leila_earnings_proof : leila_weekly_earnings = 350 := by
  sorry

end NUMINAMATH_CALUDE_leila_earnings_proof_l2090_209046


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2090_209031

theorem polynomial_evaluation (y : ℝ) (h : y = 2) : y^4 + y^3 + y^2 + y + 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2090_209031


namespace NUMINAMATH_CALUDE_inequalities_in_quadrants_I_and_II_l2090_209011

-- Define the conditions
def satisfies_inequalities (x y : ℝ) : Prop :=
  y > x^2 ∧ y > 4 - x

-- Define the quadrants
def in_quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0
def in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0
def in_quadrant_III (x y : ℝ) : Prop := x < 0 ∧ y < 0
def in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Theorem statement
theorem inequalities_in_quadrants_I_and_II :
  ∀ x y : ℝ, satisfies_inequalities x y →
    (in_quadrant_I x y ∨ in_quadrant_II x y) ∧
    ¬(in_quadrant_III x y) ∧ ¬(in_quadrant_IV x y) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_in_quadrants_I_and_II_l2090_209011


namespace NUMINAMATH_CALUDE_expression_evaluation_l2090_209038

theorem expression_evaluation : 6^3 - 4 * 6^2 + 4 * 6 + 2 = 98 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2090_209038


namespace NUMINAMATH_CALUDE_total_candy_pieces_l2090_209022

def chocolate_boxes : ℕ := 2
def caramel_boxes : ℕ := 5
def pieces_per_box : ℕ := 4

theorem total_candy_pieces : 
  (chocolate_boxes + caramel_boxes) * pieces_per_box = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_candy_pieces_l2090_209022


namespace NUMINAMATH_CALUDE_pages_left_after_three_weeks_l2090_209026

-- Define the structure for a book
structure Book where
  totalPages : ℕ
  pagesRead : ℕ
  pagesPerDay : ℕ

-- Define Elliot's books
def book1 : Book := ⟨512, 194, 30⟩
def book2 : Book := ⟨298, 0, 20⟩
def book3 : Book := ⟨365, 50, 25⟩
def book4 : Book := ⟨421, 0, 15⟩

-- Define the number of days
def days : ℕ := 21

-- Function to calculate pages left after reading
def pagesLeftAfterReading (b : Book) (days : ℕ) : ℕ :=
  max 0 (b.totalPages - b.pagesRead - b.pagesPerDay * days)

-- Theorem statement
theorem pages_left_after_three_weeks :
  pagesLeftAfterReading book1 days + 
  pagesLeftAfterReading book2 days + 
  pagesLeftAfterReading book3 days + 
  pagesLeftAfterReading book4 days = 106 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_after_three_weeks_l2090_209026


namespace NUMINAMATH_CALUDE_lollipop_consumption_days_l2090_209018

/-- The number of days it takes to finish all lollipops -/
def days_to_finish_lollipops (alison_lollipops henry_extra diane_ratio daily_consumption : ℕ) : ℕ :=
  let henry_lollipops := alison_lollipops + henry_extra
  let diane_lollipops := alison_lollipops * diane_ratio
  let total_lollipops := alison_lollipops + henry_lollipops + diane_lollipops
  total_lollipops / daily_consumption

/-- Theorem stating that it takes 6 days to finish all lollipops under given conditions -/
theorem lollipop_consumption_days :
  days_to_finish_lollipops 60 30 2 45 = 6 := by
  sorry

#eval days_to_finish_lollipops 60 30 2 45

end NUMINAMATH_CALUDE_lollipop_consumption_days_l2090_209018


namespace NUMINAMATH_CALUDE_series_sum_l2090_209077

-- Define the series
def series_term (n : ℕ) : ℚ := n / 5^n

-- State the theorem
theorem series_sum :
  (∑' n, series_term n) = 5/16 := by sorry

end NUMINAMATH_CALUDE_series_sum_l2090_209077


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2090_209090

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  (1 / x + 1 / y) ≥ 1 := by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2090_209090


namespace NUMINAMATH_CALUDE_first_discount_rate_l2090_209036

/-- Proves that given a shirt with an original price of 400, which after two
    consecutive discounts (the second being 5%) results in a final price of 340,
    the first discount rate is equal to (200/19)%. -/
theorem first_discount_rate (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 400 →
  final_price = 340 →
  second_discount = 0.05 →
  ∃ (first_discount : ℝ),
    final_price = original_price * (1 - first_discount) * (1 - second_discount) ∧
    first_discount = 200 / 19 / 100 := by
  sorry

end NUMINAMATH_CALUDE_first_discount_rate_l2090_209036


namespace NUMINAMATH_CALUDE_distance_AP_equals_one_l2090_209049

-- Define the triangle and circle
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (2, 0)
def center : ℝ × ℝ := (1, 1)

-- Define the inscribed circle
def ω (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Define point M where ω touches BC
def M : ℝ × ℝ := (0, 0)

-- Define point P where AM intersects ω
def P : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem distance_AP_equals_one :
  let d := Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)
  d = 1 := by sorry

end NUMINAMATH_CALUDE_distance_AP_equals_one_l2090_209049


namespace NUMINAMATH_CALUDE_quadratic_equation_single_solution_l2090_209025

theorem quadratic_equation_single_solution 
  (b : ℝ) 
  (h1 : b ≠ 0)
  (h2 : ∃! x : ℝ, 3 * x^2 + b * x + 10 = 0) :
  ∃ x : ℝ, x = -Real.sqrt 30 / 3 ∧ 3 * x^2 + b * x + 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_single_solution_l2090_209025


namespace NUMINAMATH_CALUDE_fraction_sum_equals_negative_one_l2090_209039

theorem fraction_sum_equals_negative_one 
  (a b : ℝ) 
  (h_distinct : a ≠ b) 
  (h_equation : a / b + a = b / a + b) : 
  1 / a + 1 / b = -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_negative_one_l2090_209039


namespace NUMINAMATH_CALUDE_stating_minimum_red_cubes_correct_l2090_209057

/-- 
Given a positive integer n, we construct a cube of side length 3n using smaller 3x3x3 cubes.
Each 3x3x3 cube is made of 26 white unit cubes and 1 black unit cube.
This function returns the minimum number of white unit cubes that need to be painted red
so that every remaining white unit cube has at least one common point with at least one red unit cube.
-/
def minimum_red_cubes (n : ℕ+) : ℕ :=
  (n + 1) * n^2

/-- 
Theorem stating that the minimum number of white unit cubes that need to be painted red
is indeed (n+1)n^2, where n is the number of 3x3x3 cubes along each edge of the larger cube.
-/
theorem minimum_red_cubes_correct (n : ℕ+) : 
  minimum_red_cubes n = (n + 1) * n^2 := by sorry

end NUMINAMATH_CALUDE_stating_minimum_red_cubes_correct_l2090_209057


namespace NUMINAMATH_CALUDE_extra_digit_sum_l2090_209084

theorem extra_digit_sum (x y : ℕ) (a : Fin 10) :
  x + y = 23456 →
  (10 * x + a.val) + y = 55555 →
  a.val = 5 :=
by sorry

end NUMINAMATH_CALUDE_extra_digit_sum_l2090_209084


namespace NUMINAMATH_CALUDE_decimal_digits_of_fraction_l2090_209006

/-- The number of digits to the right of the decimal point when 5^7 / (10^5 * 125) is expressed as a decimal is 5. -/
theorem decimal_digits_of_fraction : ∃ (n : ℕ) (d : ℕ+) (k : ℕ),
  5^7 / (10^5 * 125) = n / d ∧
  10^k ≤ d ∧ d < 10^(k+1) ∧
  k = 5 := by sorry

end NUMINAMATH_CALUDE_decimal_digits_of_fraction_l2090_209006


namespace NUMINAMATH_CALUDE_list_price_correct_l2090_209033

/-- Given a book's cost price, calculates the list price that results in a 40% profit
    after an 18% deduction from the list price -/
def listPrice (costPrice : ℝ) : ℝ :=
  costPrice * 1.7073

theorem list_price_correct (costPrice : ℝ) :
  let listPrice := listPrice costPrice
  let sellingPrice := listPrice * (1 - 0.18)
  sellingPrice = costPrice * 1.4 := by
  sorry

end NUMINAMATH_CALUDE_list_price_correct_l2090_209033
