import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_iff_positive_l361_36134

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2^x

-- State the theorem
theorem two_solutions_iff_positive (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |f x₁ - 2| = m ∧ |f x₂ - 2| = m) ↔ m > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_iff_positive_l361_36134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_trapezoid_equal_area_l361_36149

/-- Represents a trapezoid with given dimensions -/
structure Trapezoid where
  upper_side : ℝ
  lower_side : ℝ
  height : ℝ

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoid_area (t : Trapezoid) : ℝ :=
  (t.upper_side + t.lower_side) / 2 * t.height

/-- Calculates the area of a square -/
noncomputable def square_area (side : ℝ) : ℝ :=
  side * side

/-- Theorem stating that a square with side length equal to the trapezoid's height
    has the same area as the trapezoid with given dimensions -/
theorem square_trapezoid_equal_area (t : Trapezoid)
    (h1 : t.upper_side = 15)
    (h2 : t.lower_side = 9)
    (h3 : t.height = 12) :
    square_area t.height = trapezoid_area t := by
  sorry

#check square_trapezoid_equal_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_trapezoid_equal_area_l361_36149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_neg_one_l361_36170

def sequence_a : ℕ → ℚ
  | 0 => 1 / 2  -- Add case for 0
  | 1 => 1 / 2
  | n + 1 => 1 - 1 / sequence_a n

theorem a_5_equals_neg_one : sequence_a 5 = -1 := by
  -- Evaluate the sequence step by step
  have h1 : sequence_a 1 = 1 / 2 := rfl
  have h2 : sequence_a 2 = -1 := by
    simp [sequence_a]
    norm_num
  have h3 : sequence_a 3 = 2 := by
    simp [sequence_a, h2]
    norm_num
  have h4 : sequence_a 4 = 1 / 2 := by
    simp [sequence_a, h3]
    norm_num
  -- Final step
  calc
    sequence_a 5 = 1 - 1 / sequence_a 4 := by rfl
    _ = 1 - 1 / (1 / 2) := by rw [h4]
    _ = 1 - 2 := by norm_num
    _ = -1 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_neg_one_l361_36170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_nine_equals_78_l361_36124

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

noncomputable def g (x : ℝ) : ℝ := 3 * ((x - 3) / 2)^2 + 4 * ((x - 3) / 2) - 6

-- Theorem statement
theorem g_of_negative_nine_equals_78 : g (-9) = 78 := by
  -- Expand the definition of g
  unfold g
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_nine_equals_78_l361_36124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l361_36162

theorem problem_statement : 4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l361_36162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l361_36115

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

-- State the theorem
theorem g_monotone_increasing :
  StrictMonoOn g (Set.Icc (-Real.pi/12) (5*Real.pi/12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l361_36115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l361_36110

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi/6) + Real.sin (x - Real.pi/6)
noncomputable def g (x : ℝ) : ℝ := Real.cos x

noncomputable def M (a : ℝ) : ℝ × ℝ := (a, f a)
noncomputable def N (a : ℝ) : ℝ × ℝ := (a, g a)

noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem max_distance_MN :
  ∀ a : ℝ, distance (M a) (N a) ≤ 2 ∧ ∃ a : ℝ, distance (M a) (N a) = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l361_36110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_bound_l361_36128

/-- A cubic polynomial with three distinct real roots -/
def P (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c

/-- The quadratic polynomial Q(x) = x² + x + 2001 -/
def Q : ℝ → ℝ := fun x ↦ x^2 + x + 2001

/-- The theorem statement -/
theorem cubic_polynomial_bound
  (a b c : ℝ)
  (h1 : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ P a b c x = 0 ∧ P a b c y = 0 ∧ P a b c z = 0)
  (h2 : ∀ x : ℝ, P a b c (Q x) ≠ 0) :
  P a b c 2001 > 1/64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_bound_l361_36128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_segment_mark_correct_l361_36112

def max_segment_mark (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2*n - 4 else 2*n - 3

theorem max_segment_mark_correct (n : ℕ) (h : n > 4) :
  ∀ k : ℕ, (∃ (points : Fin n → ℝ × ℝ) 
    (segments : List (Fin n × Fin n))
    (marks : List ℕ),
    (∀ i j l : Fin n, i ≠ j → j ≠ l → i ≠ l → 
      ¬ (∃ (a b c : ℝ), a * (points i).1 + b * (points i).2 = c ∧
                         a * (points j).1 + b * (points j).2 = c ∧
                         a * (points l).1 + b * (points l).2 = c)) ∧
    (∀ s ∈ segments, 
      ∀ t ∈ segments, 
      (s.1 = t.1 ∨ s.1 = t.2 ∨ s.2 = t.1 ∨ s.2 = t.2) → 
      segments.indexOf s < segments.indexOf t → 
      marks[segments.indexOf s]? < marks[segments.indexOf t]?) ∧
    k ∈ marks) →
  k ≤ max_segment_mark n :=
by
  sorry

#check max_segment_mark_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_segment_mark_correct_l361_36112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l361_36177

noncomputable def f (k : ℤ) (x : ℝ) : ℝ := 2^x - (k - 1) * 2^(-x)

noncomputable def g (x : ℝ) : ℝ := f 2 x / f 0 x

theorem problem_solution :
  (∃ x : ℝ, f 2 x = 2) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂) ∧
  (∀ m : ℝ, m > -17/12 → ∀ x : ℝ, x ≥ 1 → f 0 (2*x) + 2*m * f 2 x ≠ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l361_36177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_heart_face_then_club_face_l361_36131

/-- A standard deck of cards. -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size : cards.card = 52)

/-- Face cards are Jack, Queen, and King. -/
def isFaceCard (card : Nat × Nat) : Bool :=
  card.1 ≥ 11 ∧ card.1 ≤ 13

/-- Hearts suit. -/
def isHeart (card : Nat × Nat) : Bool :=
  card.2 = 1

/-- Clubs suit. -/
def isClub (card : Nat × Nat) : Bool :=
  card.2 = 2

/-- Number of heart face cards in a standard deck. -/
def numHeartFaceCards (d : Deck) : Nat :=
  (d.cards.filter (λ c => isFaceCard c ∧ isHeart c)).card

/-- Number of club face cards in a standard deck. -/
def numClubFaceCards (d : Deck) : Nat :=
  (d.cards.filter (λ c => isFaceCard c ∧ isClub c)).card

theorem prob_heart_face_then_club_face (d : Deck) :
  (numHeartFaceCards d : ℚ) / d.cards.card * (numClubFaceCards d : ℚ) / (d.cards.card - 1) = 1 / 294 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_heart_face_then_club_face_l361_36131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_double_sum_l361_36142

theorem triangle_tan_double_sum (A B : Real) (h1 : 0 < A) (h2 : A < π / 2) (h3 : 0 < B) (h4 : B < π / 2) 
  (h5 : A + B < π) (h6 : Real.sin A = 3 / 5) (h7 : Real.tan B = 2) : 
  Real.tan (2 * (A + B)) = 44 / 117 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_double_sum_l361_36142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l361_36107

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (-3, m)

theorem vector_properties :
  (∀ m : ℝ, (a.1 * (b m).1 + a.2 * (b m).2 = 0) → m = 3/2) ∧
  (Real.arccos ((a.1 * (b (-1)).1 + a.2 * (b (-1)).2) / 
    (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt ((b (-1)).1^2 + (b (-1)).2^2))) = 3 * Real.pi / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l361_36107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b2f_hex_to_decimal_l361_36132

/-- Converts a base 16 digit to its decimal value -/
def hexToDecimal (c : Char) : ℕ :=
  match c with
  | 'B' => 11
  | 'F' => 15
  | d => d.toString.toNat!

/-- Converts a base 16 number (as a string) to its decimal value -/
def hexStringToDecimal (s : String) : ℕ :=
  s.data.reverse.enum.foldl (fun acc (i, c) => acc + 16^i * hexToDecimal c) 0

theorem b2f_hex_to_decimal :
  hexStringToDecimal "B2F" = 2863 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b2f_hex_to_decimal_l361_36132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_half_l361_36121

/-- A rectangle in 2D space --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The probability that a randomly chosen point (x,y) in the given rectangle satisfies x < 2y --/
noncomputable def probability_x_less_than_2y (r : Rectangle) : ℝ :=
  let area_triangle := (r.y_max - r.y_min) * (r.x_max - r.x_min) / 2
  let area_rectangle := (r.x_max - r.x_min) * (r.y_max - r.y_min)
  area_triangle / area_rectangle

/-- The theorem stating that the probability is 1/2 for the specific rectangle --/
theorem probability_is_half :
  let r : Rectangle := { x_min := 0, x_max := 4, y_min := 0, y_max := 3,
                         h_x := by norm_num, h_y := by norm_num }
  probability_x_less_than_2y r = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_half_l361_36121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_from_medians_l361_36187

/-- A triangle can be constructed from three given lengths if and only if these lengths satisfy the triangle inequality -/
theorem triangle_construction_from_medians (s_a s_b s_c : ℝ) :
  (∃ (A B C : ℝ × ℝ), 
    let d := λ (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
    let midpoint := λ (p q : ℝ × ℝ) => ((p.1 + q.1) / 2, (p.2 + q.2) / 2)
    d (midpoint B C) A = s_a ∧
    d (midpoint A C) B = s_b ∧
    d (midpoint A B) C = s_c) ↔
  (s_a + s_b > s_c ∧ s_a + s_c > s_b ∧ s_b + s_c > s_a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_from_medians_l361_36187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_tan_inequality_l361_36189

/-- Given a > 0, a ≠ 1, and log_a x > tan x for any x ∈ (0, π/4), prove that a ∈ [π/4, 1) -/
theorem log_tan_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x ∈ Set.Ioo 0 (π/4), Real.log x / Real.log a > Real.tan x) :
  a ∈ Set.Icc (π/4) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_tan_inequality_l361_36189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_ticket_price_problem_l361_36130

/-- Calculates the plane ticket price given luggage weight, free allowance, overweight charge rate, and paid luggage fee -/
noncomputable def plane_ticket_price (luggage_weight : ℝ) (free_allowance : ℝ) (overweight_charge_rate : ℝ) (paid_luggage_fee : ℝ) : ℝ :=
  paid_luggage_fee / ((luggage_weight - free_allowance) * overweight_charge_rate)

theorem plane_ticket_price_problem : 
  plane_ticket_price 30 20 0.015 180 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_ticket_price_problem_l361_36130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l361_36137

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a₁^2 + b₁^2)

/-- The first line: 3x + 4y - 3 = 0 -/
def line1 : ℝ × ℝ → Prop :=
  fun (x, y) ↦ 3 * x + 4 * y - 3 = 0

/-- The second line: 6x + 8y + 14 = 0 -/
def line2 : ℝ × ℝ → Prop :=
  fun (x, y) ↦ 6 * x + 8 * y + 14 = 0

theorem parallel_lines_distance :
  distance_between_parallel_lines 3 4 (-3) 6 8 14 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l361_36137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_a_value_l361_36100

/-- Three points are collinear if they lie on the same straight line. -/
def AreCollinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  ∃ k b : ℝ, (p1.2 = k * p1.1 + b) ∧ (p2.2 = k * p2.1 + b) ∧ (p3.2 = k * p3.1 + b)

/-- Given three points A(a,2), B(5,1), and C(-4,2a) are collinear, prove that a = 2 or a = 7/2. -/
theorem collinear_points_a_value (a : ℝ) :
  AreCollinear (a, 2) (5, 1) (-4, 2*a) → a = 2 ∨ a = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_a_value_l361_36100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l361_36168

/-- Given a triangle ABC and a point M on its plane, if AC + AB = 2AM, then MC + MB = 0 -/
theorem triangle_vector_relation {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n] 
  (A B C M : n) : 
  (C - A) + (B - A) = (2 : ℝ) • (M - A) → (C - M) + (B - M) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l361_36168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l361_36120

open Set Real

theorem problem_solution :
  ∀ (a : ℝ), a > 0 →
  let p (x : ℝ) := x^2 - 4*a*x + 3*a^2 < 0
  let q (x : ℝ) := (x - 3)^2 < 1
  (∀ x, a = 1 → (p x ∧ q x ↔ 2 < x ∧ x < 3)) ∧
  ((∀ x, ¬(p x) → ¬(q x)) ∧ (∃ x, ¬(p x) ∧ q x) ↔ 4/3 ≤ a ∧ a ≤ 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l361_36120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l361_36156

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x ^ 2 - Real.cos (2 * x + Real.pi / 3)

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ T = Real.pi ∧ ∀ x, f (x + T) = f x ∧
    ∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -1 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l361_36156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_specific_parabola_focus_l361_36129

/-- The focus of a parabola given by y = a * x^2 is at (0, -1/(4*a)) when a < 0 -/
theorem parabola_focus (a : ℝ) (h : a < 0) :
  let parabola := λ x : ℝ ↦ a * x^2
  let focus := (0, -1 / (4 * a))
  ∀ x : ℝ, parabola x = a * x^2 ∧ focus.1 = 0 ∧ focus.2 = -1 / (4 * a) :=
by sorry

/-- The focus of the parabola y = -1/16 * x^2 is at (0, -4) -/
theorem specific_parabola_focus :
  let a : ℝ := -1/16
  let parabola := λ x : ℝ ↦ a * x^2
  let focus := (0, -4)
  ∀ x : ℝ, parabola x = a * x^2 ∧ focus.1 = 0 ∧ focus.2 = -4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_specific_parabola_focus_l361_36129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_and_max_function_l361_36181

-- Define the geometric sequence property
def is_geometric_sequence (a b c d : ℝ) : Prop := b^2 = a*c ∧ c*d = b^2

-- Define the function
noncomputable def f (x : ℝ) := Real.log (x + 2) - x

-- State the theorem
theorem geometric_sequence_and_max_function
  (a b c d : ℝ)
  (h_geo : is_geometric_sequence a b c d)
  (h_max : ∃ (c : ℝ), f b = c ∧ ∀ x, f x ≤ c)
  : a * d = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_and_max_function_l361_36181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nehas_mother_age_l361_36105

/-- Neha's age -/
def N : ℕ := sorry

/-- Neha's mother's age -/
def M : ℕ := sorry

/-- Neha's mother was 4 times her age 12 years ago -/
axiom mother_age_12_years_ago : M - 12 = 4 * (N - 12)

/-- Neha's mother will be twice as old as Neha 12 years from now -/
axiom mother_age_12_years_future : M + 12 = 2 * (N + 12)

/-- The present age of Neha's mother is 60 years old -/
theorem nehas_mother_age : M = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nehas_mother_age_l361_36105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_11_l361_36176

/-- An arithmetic sequence where a_3 and a_9 are roots of x^2 - 16x + c = 0 (c < 64) -/
def arithmetic_sequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (a 3)^2 - 16 * (a 3) + c = 0 ∧
  (a 9)^2 - 16 * (a 9) + c = 0 ∧
  c < 64

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (a 1 + a n)

theorem arithmetic_sequence_sum_11 (a : ℕ → ℝ) (c : ℝ) :
  arithmetic_sequence a c → arithmetic_sum a 11 = 88 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_11_l361_36176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_is_5_or_6_l361_36123

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def sum_of_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem max_sum_is_5_or_6 (a₁ d : ℝ) :
  d < 0 →
  |arithmetic_sequence a₁ d 3| = |arithmetic_sequence a₁ d 9| →
  ∃ n : ℕ, (n = 5 ∨ n = 6) ∧
    ∀ k : ℕ, k > 0 → sum_of_first_n_terms a₁ d n ≥ sum_of_first_n_terms a₁ d k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_is_5_or_6_l361_36123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l361_36178

/-- Given a perimeter, an angle, and an altitude, there exists a triangle satisfying these conditions. -/
theorem triangle_existence (p α m : ℝ) (h_p : p > 0) (h_α : 0 < α ∧ α < Real.pi) (h_m : m > 0) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- sides are positive
    a + b + c = p ∧  -- perimeter condition
    ∃ (θ : ℝ), θ = α ∧ Real.cos θ = (b^2 + c^2 - a^2) / (2*b*c) ∧  -- angle condition
    m = b * c * Real.sin α / p  -- altitude condition
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l361_36178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_derivative_equals_surface_area_l361_36179

-- Define the volume function for a sphere
noncomputable def sphereVolume (R : ℝ) : ℝ := (4/3) * Real.pi * R^3

-- Define the surface area function for a sphere
noncomputable def sphereSurfaceArea (R : ℝ) : ℝ := 4 * Real.pi * R^2

-- State the theorem
theorem sphere_volume_derivative_equals_surface_area
  (R : ℝ) (h : R > 0) :
  deriv sphereVolume R = sphereSurfaceArea R := by
  sorry

-- You can add more theorems or lemmas here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_derivative_equals_surface_area_l361_36179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_order_determinant_not_all_positive_l361_36165

theorem third_order_determinant_not_all_positive :
  ¬ ∃ (A : Matrix (Fin 3) (Fin 3) ℝ), 
    (Matrix.det A > 0) ∧
    (∀ σ : Equiv.Perm (Fin 3), (A 0 (σ 0)) * (A 1 (σ 1)) * (A 2 (σ 2)) > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_order_determinant_not_all_positive_l361_36165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_A_equal_functions_C_equal_functions_D_equal_l361_36199

-- Function definitions
noncomputable def f_A (x : ℝ) : ℝ := x
noncomputable def g_A (x : ℝ) : ℝ := Real.rpow x (1/3)

noncomputable def f_C (x : ℝ) : ℝ := abs x / x
noncomputable def g_C (x : ℝ) : ℝ := if x > 0 then 1 else -1

noncomputable def f_D (t : ℝ) : ℝ := abs (t - 1)
noncomputable def g_D (x : ℝ) : ℝ := abs (x - 1)

-- Theorems to prove
theorem functions_A_equal : ∀ x : ℝ, f_A x = g_A x := by sorry

theorem functions_C_equal : ∀ x : ℝ, x ≠ 0 → f_C x = g_C x := by sorry

theorem functions_D_equal : ∀ x : ℝ, f_D x = g_D x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_A_equal_functions_C_equal_functions_D_equal_l361_36199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l361_36122

-- Define an arithmetic sequence
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

-- Theorem statement
theorem arithmetic_sequence_sum_ratio 
  (a₁ d : ℝ) (h : d ≠ 0) (h_ratio : S a₁ d 3 / S a₁ d 6 = 1 / 3) :
  S a₁ d 6 / S a₁ d 12 = 3 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l361_36122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_seven_triangles_l361_36145

/-- A triangle is a set of points in ℝ² -/
def IsTriangle (T : Set (ℝ × ℝ)) : Prop := sorry

/-- A quadrilateral is a set of points in ℝ² -/
def IsQuadrilateral (Q : Set (ℝ × ℝ)) : Prop := sorry

/-- The area of a set of points in ℝ² -/
noncomputable def Area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- A triangle can be divided into 4 equal triangles -/
axiom triangle_division (T : Set (ℝ × ℝ)) : 
  IsTriangle T → ∃ (T1 T2 T3 T4 : Set (ℝ × ℝ)), 
    IsTriangle T1 ∧ IsTriangle T2 ∧ IsTriangle T3 ∧ IsTriangle T4 ∧
    T = T1 ∪ T2 ∪ T3 ∪ T4 ∧
    Area T1 = Area T2 ∧ Area T2 = Area T3 ∧ Area T3 = Area T4

/-- Theorem: There exists a quadrilateral that can be divided into 7 equal triangles -/
theorem quadrilateral_seven_triangles : 
  ∃ (Q : Set (ℝ × ℝ)) (T1 T2 T3 T4 T5 T6 T7 : Set (ℝ × ℝ)), 
    IsQuadrilateral Q ∧
    IsTriangle T1 ∧ IsTriangle T2 ∧ IsTriangle T3 ∧ IsTriangle T4 ∧
    IsTriangle T5 ∧ IsTriangle T6 ∧ IsTriangle T7 ∧
    Q = T1 ∪ T2 ∪ T3 ∪ T4 ∪ T5 ∪ T6 ∪ T7 ∧
    Area T1 = Area T2 ∧ Area T2 = Area T3 ∧ Area T3 = Area T4 ∧
    Area T4 = Area T5 ∧ Area T5 = Area T6 ∧ Area T6 = Area T7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_seven_triangles_l361_36145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_18_hours_l361_36108

/-- Represents the journey of Tom, Dick, and Harry --/
structure Journey where
  totalDistance : ℝ
  carSpeed : ℝ
  walkingSpeed : ℝ
  harryCarDistance : ℝ
  tomBacktrackDistance : ℝ

/-- The total time of the journey --/
noncomputable def journeyTime (j : Journey) : ℝ :=
  (j.harryCarDistance / j.carSpeed) + ((j.totalDistance - j.harryCarDistance) / j.walkingSpeed)

/-- Theorem stating that the journey time is 18 hours --/
theorem journey_time_is_18_hours (j : Journey) 
  (h1 : j.totalDistance = 150)
  (h2 : j.carSpeed = 30)
  (h3 : j.walkingSpeed = 4)
  (h4 : j.harryCarDistance = 90)
  (h5 : j.tomBacktrackDistance = 60)
  (h6 : (j.harryCarDistance / j.carSpeed) + ((j.totalDistance - j.harryCarDistance) / j.walkingSpeed) = 
        (j.harryCarDistance / j.carSpeed) + (j.tomBacktrackDistance / j.carSpeed) + 
        ((j.totalDistance - (j.harryCarDistance - j.tomBacktrackDistance)) / j.carSpeed))
  (h7 : ((j.harryCarDistance - j.tomBacktrackDistance) / j.walkingSpeed) + 
        ((j.totalDistance - (j.harryCarDistance - j.tomBacktrackDistance)) / j.carSpeed) = 
        (j.harryCarDistance / j.carSpeed) + ((j.totalDistance - j.harryCarDistance) / j.walkingSpeed)) :
  journeyTime j = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_18_hours_l361_36108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_max_min_in_13_comparisons_l361_36111

/-- A type representing a collection of comparable elements -/
def Coins (α : Type*) [LinearOrder α] (n : ℕ) := Fin n → α

/-- A function that performs a comparison between two elements -/
def Comparison (α : Type*) [LinearOrder α] := α → α → Bool

/-- The result of identifying the maximum and minimum elements -/
structure IdentificationResult (α : Type*) where
  max : α
  min : α

/-- A function that identifies the maximum and minimum elements -/
def identifyMaxMin (α : Type*) [LinearOrder α] (coins : Coins α 10) (compare : Comparison α) : ℕ → Option (IdentificationResult α) :=
  sorry

/-- Theorem stating that it's possible to identify the max and min in at most 13 comparisons -/
theorem identify_max_min_in_13_comparisons 
  (α : Type*) [LinearOrder α] (coins : Coins α 10) (compare : Comparison α) :
  ∃ (result : IdentificationResult α), identifyMaxMin α coins compare 13 = some result := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_max_min_in_13_comparisons_l361_36111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l361_36186

theorem trigonometric_identity (α β : Real) 
  (h1 : α ∈ Set.Ioo (3 * Real.pi / 4) Real.pi) 
  (h2 : β ∈ Set.Ioo (3 * Real.pi / 4) Real.pi) 
  (h3 : Real.cos (α + β) = 4 / 5) 
  (h4 : Real.cos (β - Real.pi / 4) = -5 / 13) : 
  Real.sin (α + Real.pi / 4) = -33 / 65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l361_36186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_journey_time_l361_36183

/-- The time taken for Martin to reach Lawrence's house -/
noncomputable def time_to_Lawrence (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Theorem stating that Martin takes 6 hours to reach Lawrence's house -/
theorem martin_journey_time :
  let distance : ℝ := 12 -- miles
  let speed : ℝ := 2 -- miles per hour
  time_to_Lawrence distance speed = 6 := by
  -- Unfold the definition of time_to_Lawrence
  unfold time_to_Lawrence
  -- Simplify the expression
  simp
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_journey_time_l361_36183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_comparison_l361_36184

theorem sin_comparison :
  (∀ x y, x ∈ Set.Icc (-Real.pi/2) 0 → y ∈ Set.Icc (-Real.pi/2) 0 → x < y → Real.sin x < Real.sin y) →
  -Real.pi/18 ∈ Set.Icc (-Real.pi/2) 0 →
  -Real.pi/10 ∈ Set.Icc (-Real.pi/2) 0 →
  -Real.pi/18 > -Real.pi/10 →
  Real.sin (-Real.pi/18) > Real.sin (-Real.pi/10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_comparison_l361_36184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixtieth_point_coordinates_fifty_fifth_point_points_after_55th_l361_36173

/-- Represents a point in 2D space -/
structure Point where
  x : Nat
  y : Nat

/-- Defines the sequence of points -/
def pointSequence : Nat → Point
  | 0 => ⟨1, 1⟩  -- Add case for 0
  | n + 1 =>
    let prev := pointSequence n
    if prev.x + prev.y < n + 2 then
      ⟨prev.x + 1, prev.y⟩
    else
      ⟨1, prev.y + 1⟩

/-- The theorem to prove -/
theorem sixtieth_point_coordinates :
  pointSequence 59 = Point.mk 5 7 := by
  sorry

/-- Helper function to compute the sum of natural numbers from 1 to n -/
def sum_to_n (n : Nat) : Nat :=
  n * (n + 1) / 2

/-- Helper theorem: The 55th point is (10, 1) -/
theorem fifty_fifth_point :
  pointSequence 54 = Point.mk 10 1 := by
  sorry

/-- Helper theorem: Points after 55th follow the pattern -/
theorem points_after_55th (n : Nat) (h : n > 54 ∧ n ≤ 59) :
  let p := pointSequence n
  p.x + p.y = 12 ∧ p.x = n - 54 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixtieth_point_coordinates_fifty_fifth_point_points_after_55th_l361_36173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_O_to_M_is_sqrt3_div_2_l361_36175

noncomputable section

-- Define the points A and B
def A : ℝ × ℝ := (Real.cos (110 * Real.pi / 180), Real.sin (110 * Real.pi / 180))
def B : ℝ × ℝ := (Real.cos (50 * Real.pi / 180), Real.sin (50 * Real.pi / 180))

-- Define the midpoint M of segment AB
def M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem distance_O_to_M_is_sqrt3_div_2 :
  distance O M = Real.sqrt 3 / 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_O_to_M_is_sqrt3_div_2_l361_36175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_fraction_given_to_son_l361_36153

/-- Proves that the fraction of apples given to the son is 1/5 --/
theorem apple_fraction_given_to_son 
  (blue_apples : ℕ) 
  (yellow_apples : ℕ) 
  (remaining_apples : ℕ) : 
  blue_apples = 5 → 
  yellow_apples = 2 * blue_apples → 
  remaining_apples = 12 → 
  (blue_apples + yellow_apples - remaining_apples : ℚ) / (blue_apples + yellow_apples) = 1 / 5 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_fraction_given_to_son_l361_36153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l361_36119

theorem power_equality (x : ℝ) (h : (3 : ℝ)^(4*x) = 16) : (27 : ℝ)^(x+1) = 432 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l361_36119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l361_36125

-- Define the hyperbola parameters
def center : ℝ × ℝ := (-3, 1)
def focus : ℝ × ℝ := (2, 1)
def vertex : ℝ × ℝ := (-1, 1)

-- Define h and k
def h : ℝ := center.1
def k : ℝ := center.2

-- Define a as the distance from center to vertex
noncomputable def a : ℝ := Real.sqrt ((vertex.1 - center.1)^2 + (vertex.2 - center.2)^2)

-- Define c as the distance from center to focus
noncomputable def c : ℝ := Real.sqrt ((focus.1 - center.1)^2 + (focus.2 - center.2)^2)

-- Define b using the relationship c^2 = a^2 + b^2
noncomputable def b : ℝ := Real.sqrt (c^2 - a^2)

-- Theorem to prove
theorem hyperbola_sum : h + k + a + b = 0 + Real.sqrt 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l361_36125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l361_36169

/-- The area of a triangle with side lengths a, b, and c, using Heron's formula -/
noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The circumradius of a triangle with side lengths a, b, and c -/
noncomputable def circumradius (a b c : ℝ) : ℝ :=
  (a * b * c) / (4 * area_triangle a b c)

/-- A triangle with side lengths 7½, 10, and 12½ has a circumcircle with radius 25/4 -/
theorem triangle_circumradius : ∀ (a b c : ℝ),
  a = 15/2 ∧ b = 10 ∧ c = 25/2 →
  ∃ r : ℝ, r = 25/4 ∧ r = circumradius a b c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l361_36169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_expression_l361_36160

theorem equivalent_expression : -8 - 4 - 5 + 6 = (-8) - (4) + (-5) - (-6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_expression_l361_36160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_for_integer_S_l361_36133

noncomputable def S (n : ℕ+) : ℝ := Real.sqrt (17^2 + n.val^4)

theorem unique_n_for_integer_S :
  ∃! (n : ℕ+), ∃ (m : ℕ), S n = m :=
by
  -- We know the unique solution is n = 12
  use 12
  constructor
  · use 145
    -- Here we would prove that S 12 = 145
    sorry
  · intro n' ⟨m', hm'⟩
    -- Here we would prove that if S n' is an integer, then n' must be 12
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_for_integer_S_l361_36133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_a_speed_le_car_b_speed_l361_36106

/-- Represents the motion of a car --/
structure CarMotion where
  distance : ℝ
  acceleration : ℝ
  average_speed : ℝ

/-- Car A's motion --/
noncomputable def car_a (D a : ℝ) : CarMotion :=
  { distance := D,
    acceleration := a,
    average_speed := D / (D / (3 * Real.sqrt (2 * a * D / 3)) + Real.sqrt (2 * a * D / 3) / a) }

/-- Car B's motion --/
noncomputable def car_b (D a : ℝ) : CarMotion :=
  { distance := D,
    acceleration := a,
    average_speed := D / ((18 * D / a)^(1/3)) }

/-- Theorem stating that Car A's average speed is less than or equal to Car B's --/
theorem car_a_speed_le_car_b_speed (D a : ℝ) (hD : D > 0) (ha : a > 0) :
  (car_a D a).average_speed ≤ (car_b D a).average_speed := by
  sorry

#check car_a_speed_le_car_b_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_a_speed_le_car_b_speed_l361_36106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l361_36197

-- Define the two curves
def curve1 (x : ℝ) : ℝ := 2 * (x - 1)
noncomputable def curve2 (x : ℝ) : ℝ := x + Real.exp x

-- Define the distance function between two x-coordinates
def distance (x1 x2 : ℝ) : ℝ := |x2 - x1|

-- Theorem statement
theorem min_distance_between_curves :
  ∃ (min_dist : ℝ), 
    (∀ (x1 x2 : ℝ), curve1 x1 = curve2 x2 → distance x1 x2 ≥ min_dist) ∧
    (∃ (x1' x2' : ℝ), curve1 x1' = curve2 x2' ∧ distance x1' x2' = min_dist) ∧
    min_dist = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l361_36197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_l361_36148

noncomputable def rotateClockwise (x y h k θ : ℝ) : ℝ × ℝ :=
  (h + (x - h) * Real.cos θ + (y - k) * Real.sin θ,
   k - (x - h) * Real.sin θ + (y - k) * Real.cos θ)

def reflectAboutYEqualX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  let p := (a, b)
  let rotated := rotateClockwise a b 2 3 (π/4)
  let reflected := reflectAboutYEqualX rotated.1 rotated.2
  reflected = (5, -1) →
  ‖b - a - 3.1‖ < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_l361_36148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_equals_zero_trig_fraction_equals_cos_alpha_l361_36126

open Real

-- Part 1
theorem trig_sum_equals_zero :
  Real.sin (25 * π / 6) + Real.cos (25 * π / 3) + Real.tan (-(25 * π / 4)) = 0 := by sorry

-- Part 2
theorem trig_fraction_equals_cos_alpha (α : ℝ) :
  (Real.sin (5 * π - α) * Real.cos (α + 3 * π / 2) * Real.cos (π + α)) /
  (Real.sin (α - 3 * π / 2) * Real.cos (α + π / 2) * Real.tan (α - 3 * π)) = Real.cos α := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_equals_zero_trig_fraction_equals_cos_alpha_l361_36126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_order_is_consistent_correct_order_is_unique_l361_36141

/-- Represents a coin in the configuration. -/
inductive Coin
  | A | B | C | D | E | F
  deriving BEq, Repr

/-- Represents the relationship between two coins. -/
inductive Relation
  | Above (top bottom : Coin)
  deriving BEq, Repr

/-- The configuration of coins on the table. -/
def Configuration : List Relation :=
  [ Relation.Above Coin.F Coin.B
  , Relation.Above Coin.F Coin.C
  , Relation.Above Coin.F Coin.D
  , Relation.Above Coin.F Coin.E
  , Relation.Above Coin.F Coin.A
  , Relation.Above Coin.B Coin.C
  , Relation.Above Coin.B Coin.D
  , Relation.Above Coin.B Coin.E
  , Relation.Above Coin.B Coin.A
  , Relation.Above Coin.C Coin.E
  , Relation.Above Coin.C Coin.A
  , Relation.Above Coin.D Coin.A
  , Relation.Above Coin.E Coin.A
  ]

/-- The correct order of coins from top to bottom. -/
def CorrectOrder : List Coin :=
  [Coin.F, Coin.B, Coin.C, Coin.D, Coin.E, Coin.A]

/-- Checks if the given order is consistent with the configuration. -/
def isConsistentOrder (order : List Coin) (config : List Relation) : Prop :=
  ∀ (r : Relation), r ∈ config →
    match r with
    | Relation.Above top bottom =>
        (order.indexOf top) < (order.indexOf bottom)

/-- Theorem stating that the CorrectOrder is consistent with the Configuration. -/
theorem correct_order_is_consistent :
  isConsistentOrder CorrectOrder Configuration := by
  sorry

/-- Theorem stating that CorrectOrder is the only consistent order. -/
theorem correct_order_is_unique :
  ∀ (order : List Coin),
    isConsistentOrder order Configuration →
    order = CorrectOrder := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_order_is_consistent_correct_order_is_unique_l361_36141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_min_max_element_l361_36135

def matrix : Matrix (Fin 5) (Fin 5) ℕ := fun i j =>
  match i, j with
  | 0, 0 => 5  | 0, 1 => 12 | 0, 2 => 7  | 0, 3 => 4  | 0, 4 => 10
  | 1, 0 => 6  | 1, 1 => 3  | 1, 2 => 9  | 1, 3 => 13 | 1, 4 => 11
  | 2, 0 => 14 | 2, 1 => 8  | 2, 2 => 2  | 2, 3 => 15 | 2, 4 => 5
  | 3, 0 => 1  | 3, 1 => 7  | 3, 2 => 12 | 3, 3 => 6  | 3, 4 => 8
  | 4, 0 => 9  | 4, 1 => 11 | 4, 2 => 4  | 4, 3 => 2  | 4, 4 => 3
  | _, _ => 0  -- This case should never be reached due to Fin 5

theorem no_min_max_element : 
  ¬ ∃ (i j : Fin 5), 
    (∀ k : Fin 5, matrix i j ≤ matrix i k) ∧ 
    (∀ k : Fin 5, matrix k j ≤ matrix i j) := by
  sorry

#eval matrix 0 0  -- Should output 5
#eval matrix 4 4  -- Should output 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_min_max_element_l361_36135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_root_approximation_l361_36185

-- Define the constant for the given decimal
noncomputable def x : ℝ := 32 / 10^5

-- Theorem statement
theorem nested_root_approximation :
  ∀ (ε : ℝ), ε > 0 → |Real.sqrt (Real.sqrt (Real.sqrt (Real.sqrt (Real.sqrt x)))) - 0.669| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_root_approximation_l361_36185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_implies_n_odd_and_not_div_3_l361_36157

theorem sequence_property_implies_n_odd_and_not_div_3 (n : ℕ) 
  (a b c : Fin n → ℤ) 
  (h : ∀ i j : Fin n, i ≠ j → 
    (¬ (n : ℤ) ∣ (a i - a j)) ∧
    (¬ (n : ℤ) ∣ ((b i + c i) - (b j + c j))) ∧
    (¬ (n : ℤ) ∣ (b i - b j)) ∧
    (¬ (n : ℤ) ∣ ((c i + a i) - (c j + a j))) ∧
    (¬ (n : ℤ) ∣ (c i - c j)) ∧
    (¬ (n : ℤ) ∣ ((a i + b i) - (a j + b j))) ∧
    (¬ (n : ℤ) ∣ ((a i + b i + c i) - (a j + b j + c j)))) :
  Odd n ∧ ¬(3 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_implies_n_odd_and_not_div_3_l361_36157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_radii_relation_l361_36190

/-- Helper function to calculate the area of a triangle given its side lengths -/
noncomputable def area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Given a triangle ABC with sides a, b, and c, this theorem proves the relationship
    between the radius of the inscribed circle and the radii of circles touching
    each side and extending along the prolongations of the other two sides. -/
theorem triangle_radii_relation (a b c : ℝ) (r r_a r_b r_c : ℝ) 
    (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
    (h_r : r = 2 * (area a b c) / (a + b + c))
    (h_r_a : r_a = 2 * (area a b c) / (-a + b + c))
    (h_r_b : r_b = 2 * (area a b c) / (a - b + c))
    (h_r_c : r_c = 2 * (area a b c) / (a + b - c))
    (h_area : area a b c > 0) :
    1 / r = 1 / r_a + 1 / r_b + 1 / r_c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_radii_relation_l361_36190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_triangle_area_l361_36174

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def isAcute (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def satisfiesCondition1 (t : Triangle) : Prop :=
  2 * t.a * Real.sin t.B = Real.sqrt 3 * t.b

def satisfiesCondition2 (t : Triangle) : Prop :=
  t.a = 6 ∧ t.b + t.c = 8

-- State the theorems
theorem angle_A_measure (t : Triangle) 
  (h1 : isAcute t) (h2 : satisfiesCondition1 t) : 
  t.A = Real.pi/3 := by sorry

theorem triangle_area (t : Triangle) 
  (h1 : isAcute t) (h2 : satisfiesCondition2 t) (h3 : t.A = Real.pi/3) :
  (1/2) * t.b * t.c * Real.sin t.A = 7 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_triangle_area_l361_36174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_equality_l361_36147

theorem special_number_equality (n : ℕ) :
  (4 * (10 : ℝ)^(2*n) + 4 * (10 : ℝ)^n + 1) / 9 = ((6 * (10 : ℝ)^n - 6 + 63) / 9)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_equality_l361_36147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l361_36164

/-- Definition of the ellipse -/
def is_ellipse (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (P.1^2 / a^2) + (P.2^2 / b^2) = 1

/-- Definition of focal distance -/
def focal_distance (c : ℝ) : Prop := c = 2 * Real.sqrt 3

/-- Definition of right angle at P with foci -/
def right_angle_at_P (F₁ F₂ P : ℝ × ℝ) : Prop :=
  (F₂.1 - P.1) * (F₁.1 - P.1) + (F₂.2 - P.2) * (F₁.2 - P.2) = 0

/-- Definition of triangle area -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  |((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))| / 2

/-- Definition of inner point -/
def is_inner_point (M : ℝ × ℝ) (a b : ℝ) : Prop :=
  (M.1^2 / a^2) + (M.2^2 / b^2) < 1

/-- Definition of area ratio -/
def area_ratio (A B C D : ℝ × ℝ) (r : ℝ) : Prop :=
  triangle_area A B C = r * triangle_area A B D

/-- Main theorem -/
theorem ellipse_theorem (a b c : ℝ) (F₁ F₂ P : ℝ × ℝ) (m : ℝ) :
  is_ellipse a b P →
  focal_distance c →
  right_angle_at_P F₁ F₂ P →
  triangle_area F₁ F₂ P = 1 →
  is_inner_point (0, m) a b →
  (∃ C D : ℝ × ℝ, is_ellipse a b C ∧ is_ellipse a b D ∧
    area_ratio (0, b) (0, m) C D 2) →
  (a = 2 ∧ b = 1) ∧ (1/3 < |m| ∧ |m| < 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l361_36164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_league_female_fraction_l361_36114

theorem soccer_league_female_fraction :
  ∀ (males_last_year females_last_year : ℕ)
    (males_this_year females_this_year total_this_year : ℕ),
  males_last_year = 20 →
  males_this_year = (105 * males_last_year) / 100 →
  females_this_year = (120 * females_last_year) / 100 →
  total_this_year = (110 * (males_last_year + females_last_year)) / 100 →
  total_this_year = males_this_year + females_this_year →
  (females_this_year : ℚ) / total_this_year = 4 / 11 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_league_female_fraction_l361_36114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_pyramid_equal_volume_l361_36139

/-- The volume of a cube with edge length s -/
noncomputable def cube_volume (s : ℝ) : ℝ := s^3

/-- The volume of a square-based pyramid with base edge length b and height h -/
noncomputable def pyramid_volume (b h : ℝ) : ℝ := (1/3) * b^2 * h

/-- Theorem stating that if a cube with edge length 6 has the same volume as a square-based pyramid
    with base edge length 12, then the height of the pyramid is 4.5 -/
theorem cube_pyramid_equal_volume :
  ∀ h : ℝ, cube_volume 6 = pyramid_volume 12 h → h = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_pyramid_equal_volume_l361_36139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_nested_expression_l361_36161

theorem absolute_value_nested_expression : 
  |(|-|(-2 + 2)| - 2| * 2)| = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_nested_expression_l361_36161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_die_roll_arithmetic_prob_l361_36155

/-- A standard die has 6 faces -/
def die_faces : Nat := 6

/-- The number of times the die is rolled -/
def num_rolls : Nat := 3

/-- The total number of possible outcomes when rolling a die three times -/
def total_outcomes : Nat := die_faces ^ num_rolls

/-- A sequence of three numbers from a die roll -/
def die_sequence := Fin 3 → Fin die_faces

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def is_arithmetic (s : die_sequence) : Prop :=
  s 1 - s 0 = s 2 - s 1

/-- The number of arithmetic sequences possible from rolling a die three times -/
def num_arithmetic_sequences : Nat := 18

/-- The probability of rolling an arithmetic sequence -/
noncomputable def prob_arithmetic : ℚ := num_arithmetic_sequences / total_outcomes

theorem die_roll_arithmetic_prob : 
  prob_arithmetic = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_die_roll_arithmetic_prob_l361_36155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l361_36158

noncomputable def problem (a b n d : ℝ) (x : ℝ → ℝ) : Prop :=
  a^2 - 1 = 123 * 125 ∧ 
  a > 0 ∧
  (x 0)^3 - 16*(x 0)^2 - 9*(x 0) + a = b * ((x 0) - 2) + b ∧
  (n * (n - 3)) / 2 = b + 4 ∧
  (1 - n) / 2 = (d - 1) / 2

theorem problem_solution : 
  ∀ a b n d x, problem a b n d x → a = 124 ∧ b = 50 ∧ n = 12 ∧ d = -10 :=
by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l361_36158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eulers_theorem_l361_36140

theorem eulers_theorem (k : ℕ+) (a : ℕ) (h : Nat.Coprime a k) :
  (a : ZMod k) ^ (Nat.totient k) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eulers_theorem_l361_36140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l361_36152

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions for the triangle
def validTriangle (t : Triangle) : Prop :=
  9 ≥ t.a ∧ t.a ≥ 8 ∧ 8 ≥ t.b ∧ t.b ≥ 4 ∧ 4 ≥ t.c ∧ t.c ≥ 3

-- Define the area of a triangle
noncomputable def triangleArea (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

-- Theorem statement
theorem max_triangle_area (t : Triangle) (h : validTriangle t) :
  triangleArea t ≤ 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l361_36152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_composition_l361_36182

theorem committee_composition (S : ℕ) : 
  let total_members := 7 + S
  let probability_two_english := (3 : ℚ) / ((total_members.choose 2) : ℚ)
  probability_two_english = 1/12 → S = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_composition_l361_36182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_twice_volume_l361_36144

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

def sphere_diameter (r : ℝ) : ℝ := 2 * r

theorem sphere_diameter_twice_volume (r : ℝ) (h : r = 9) :
  ∃ (a b : ℕ), 
    (a > 0 ∧ b > 0) ∧ 
    (∀ k : ℕ, k > 1 → ¬(k^3 ∣ b)) ∧
    sphere_diameter (((2 * sphere_volume r)/(4/3 * Real.pi))^(1/3)) = a * (b : ℝ)^(1/3) ∧
    a + b = 20 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

#check sphere_diameter_twice_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_twice_volume_l361_36144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l361_36166

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a three-digit number ABB -/
structure ThreeDigitNumber where
  a : Digit
  b : Digit

/-- Checks if the given ThreeDigitNumber satisfies all conditions -/
def satisfiesConditions (n : ThreeDigitNumber) : Prop :=
  let a := n.a.val
  let b := n.b.val
  ∃ c : Digit,
    -- The product of digits is a two-digit number AC
    10 ≤ a * b * b ∧ a * b * b < 100 ∧
    -- AC = a * b * b
    (10 : ℕ) * a + c.val = a * b * b ∧
    -- The product of digits of AC is equal to C
    a * c.val = c.val ∧
    -- Different letters represent different digits
    n.a ≠ n.b ∧ n.a ≠ c ∧ n.b ≠ c

theorem unique_solution :
  ∃! n : ThreeDigitNumber, satisfiesConditions n ∧ n.a = ⟨1, by norm_num⟩ ∧ n.b = ⟨4, by norm_num⟩ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l361_36166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_red_squares_4x4_min_red_squares_nxn_l361_36117

/-- Represents a grid with red squares -/
structure ColoredGrid (n : ℕ) where
  red_squares : Finset (Fin n × Fin n)

/-- Checks if a square is not crossed out by given rows and columns -/
def not_crossed_out (n : ℕ) (square : Fin n × Fin n) (rows columns : Finset (Fin n)) : Prop :=
  square.1 ∉ rows ∧ square.2 ∉ columns

/-- Checks if at least one red square is not crossed out -/
def has_uncrossed_red (n : ℕ) (grid : ColoredGrid n) (rows columns : Finset (Fin n)) : Prop :=
  ∃ square ∈ grid.red_squares, not_crossed_out n square rows columns

/-- The minimum number of red squares needed for a 4x4 grid -/
theorem min_red_squares_4x4 :
  ∀ (grid : ColoredGrid 4),
    (∀ (rows columns : Finset (Fin 4)), rows.card = 2 → columns.card = 2 →
      has_uncrossed_red 4 grid rows columns) →
    grid.red_squares.card ≥ 7 :=
sorry

/-- The minimum number of red squares needed for an nxn grid (n≥5) -/
theorem min_red_squares_nxn (n : ℕ) (h : n ≥ 5) :
  ∀ (grid : ColoredGrid n),
    (∀ (rows columns : Finset (Fin n)), rows.card = 2 → columns.card = 2 →
      has_uncrossed_red n grid rows columns) →
    grid.red_squares.card ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_red_squares_4x4_min_red_squares_nxn_l361_36117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_not_opposite_l361_36103

-- Define the sample space
def SampleSpace : Type := Fin 2 × Fin 2

-- Define the events
def ExactlyOneWhite : Set SampleSpace := {p | p.2 = 1}
def ExactlyTwoWhite : Set SampleSpace := {p | p.2 = 0}

-- Theorem statement
theorem mutually_exclusive_not_opposite :
  (ExactlyOneWhite ∩ ExactlyTwoWhite = ∅) ∧
  (ExactlyOneWhite ∪ ExactlyTwoWhite ≠ Set.univ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_not_opposite_l361_36103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l361_36127

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 - x) / 3

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + (a + 2) * x + 1 = 0}

-- Define the propositions p and q
def p (a : ℝ) : Prop := |f a| < 2
def q (a : ℝ) : Prop := A a ≠ ∅

-- Define the range of a
def range_a : Set ℝ := Set.Iic (-5) ∪ Set.Ioo (-4) 0 ∪ Set.Ici 7

theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ range_a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l361_36127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_differing_in_two_ways_l361_36172

/-- Represents the characteristics of a block -/
structure BlockCharacteristics where
  material : Fin 3
  size : Fin 3
  color : Fin 4
  shape : Fin 4
deriving Fintype

/-- The total number of blocks in the collection -/
def totalBlocks : ℕ := 144

/-- The reference block (metal medium red circle) -/
def referenceBlock : BlockCharacteristics := {
  material := 2,  -- Assuming 2 represents metal
  size := 1,      -- Assuming 1 represents medium
  color := 2,     -- Assuming 2 represents red
  shape := 0      -- Assuming 0 represents circle
}

/-- Counts the number of differences between two BlockCharacteristics -/
def countDifferences (b1 b2 : BlockCharacteristics) : ℕ :=
  (if b1.material ≠ b2.material then 1 else 0) +
  (if b1.size ≠ b2.size then 1 else 0) +
  (if b1.color ≠ b2.color then 1 else 0) +
  (if b1.shape ≠ b2.shape then 1 else 0)

/-- The main theorem stating that 37 blocks differ from the reference block in exactly two ways -/
theorem blocks_differing_in_two_ways :
  (Finset.filter (fun b : BlockCharacteristics => countDifferences b referenceBlock = 2) Finset.univ).card = 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_differing_in_two_ways_l361_36172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l361_36104

noncomputable def f (x : ℝ) : ℝ := 2 / Real.sqrt (x - 3) + (x - 4) ^ 0

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = Set.Ioo 3 4 ∪ Set.Ioi 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l361_36104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_boys_given_at_least_one_l361_36196

-- Define the sample space
def Ω : Type := List Bool

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define the event of having at least one boy
def at_least_one_boy : Set Ω := {ω : Ω | ω.any id}

-- Define the event of having two boys
def two_boys : Set Ω := {ω : Ω | ω = [true, true]}

-- Axioms for probability measure
axiom prob_nonneg : ∀ A : Set Ω, P A ≥ 0
axiom prob_total : P (Set.univ : Set Ω) = 1

-- Axiom for the probability of having at least one boy
axiom prob_at_least_one_boy : P at_least_one_boy = 3/4

-- Axiom for the probability of having two boys
axiom prob_two_boys : P two_boys = 1/4

-- Theorem to prove
theorem prob_two_boys_given_at_least_one :
  P (two_boys ∩ at_least_one_boy) / P at_least_one_boy = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_boys_given_at_least_one_l361_36196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l361_36195

noncomputable section

def g (x : ℝ) : ℝ :=
  (Real.arcsin (x / 3))^2 + (Real.pi / 2) * Real.arccos (x / 3) - (Real.arccos (x / 3))^2 + (Real.pi^2 / 18) * (x^2 - 3*x + 9)

theorem g_range :
  ∀ y ∈ Set.range g, Real.pi^2 / 4 ≤ y ∧ y ≤ 5 * Real.pi^2 / 4 ∧
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, g x = Real.pi^2 / 4 ∧
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, g x = 5 * Real.pi^2 / 4 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l361_36195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percent_l361_36101

/-- Calculates the profit percent from selling a car -/
theorem car_profit_percent
  (purchase_price repair_costs insurance registration_fees selling_price : ℕ)
  (h1 : purchase_price = 42000)
  (h2 : repair_costs = 13000)
  (h3 : insurance = 5000)
  (h4 : registration_fees = 3000)
  (h5 : selling_price = 76000) :
  let total_cost := purchase_price + repair_costs + insurance + registration_fees
  let profit := selling_price - total_cost
  let profit_percent := (profit : ℚ) / total_cost * 100
  abs (profit_percent - 20.63) < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percent_l361_36101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_equation_l361_36109

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 1) / x

-- State the theorem
theorem root_of_equation : ∀ x : ℝ, f (4 * x) = x ↔ x = 1/2 := by
  intro x
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_equation_l361_36109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sinα_sinαβ_sinβ_l361_36151

theorem max_value_sinα_sinαβ_sinβ (α β : Real) (hα : α ∈ Set.Icc 0 π) (hβ : β ∈ Set.Icc 0 π) :
  (Real.sin α + Real.sin (α + β)) * Real.sin β ≤ 8 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sinα_sinαβ_sinβ_l361_36151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventeen_factorial_digits_sum_l361_36198

def factorial (n : ℕ) : ℕ := Nat.factorial n

def digits_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem seventeen_factorial_digits_sum (T H : ℕ) :
  T < 10 →
  H < 10 →
  factorial 17 = 3556 * 10^9 + T * 10^8 + 7841280 * 10^2 + H * 10^2 →
  T + H = 1 := by
  sorry

#eval factorial 17

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventeen_factorial_digits_sum_l361_36198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_twos_in_six_rolls_l361_36146

noncomputable def roll_die : Fin 6 → ℝ
  | _ => 1 / 6

def is_two : Fin 6 → Bool
  | 1 => True
  | _ => False

noncomputable def probability_exactly_k_successes (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

theorem probability_four_twos_in_six_rolls :
  probability_exactly_k_successes 6 4 (roll_die 1) = 375 / 46656 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_twos_in_six_rolls_l361_36146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l361_36167

theorem trig_inequality (θ : ℝ) (h : 0 < θ ∧ θ < Real.pi / 2) :
  0 < Real.sin θ + Real.cos θ + Real.tan θ + (Real.cos θ / Real.sin θ) - (1 / Real.cos θ) - (1 / Real.sin θ) ∧
  Real.sin θ + Real.cos θ + Real.tan θ + (Real.cos θ / Real.sin θ) - (1 / Real.cos θ) - (1 / Real.sin θ) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l361_36167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l361_36113

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

-- Theorem statement
theorem f_properties :
  -- 1. Domain is (-1, 1)
  (∀ x, f x ≠ Real.log 0 ↔ -1 < x ∧ x < 1) ∧
  -- 2. Function is decreasing on its domain
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f y < f x) ∧
  -- 3. Solution to f(2x - 1) < 0 is 1/2 < x < 1
  (∀ x, f (2*x - 1) < 0 ↔ 1/2 < x ∧ x < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l361_36113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l361_36136

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 - x + 2)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l361_36136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_two_equals_26_l361_36143

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the theorem
theorem g_of_two_equals_26 :
  (∀ x : ℝ, g (3 * x - 7) = 5 * x + 11) → g 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_two_equals_26_l361_36143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exponential_positivity_l361_36138

theorem negation_of_exponential_positivity :
  (¬ ∀ x : ℝ, (2 : ℝ)^x > 0) ↔ (∃ x₀ : ℝ, (2 : ℝ)^x₀ ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exponential_positivity_l361_36138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l361_36154

/-- Triangle PQR with base PQ and height PR -/
structure TrianglePQR where
  PQ : ℝ  -- base length
  PR : ℝ  -- height

/-- Calculate the area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

/-- The area of triangle PQR is 4 square miles -/
theorem triangle_PQR_area (t : TrianglePQR) (h1 : t.PQ = 4) (h2 : t.PR = 2) :
  triangleArea t.PQ t.PR = 4 := by
  sorry

#check triangle_PQR_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l361_36154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l361_36191

theorem triangle_area (a b c : ℝ) (h1 : a = 20) (h2 : b = 48) (h3 : c = 52) : 
  Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)) = 480 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l361_36191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l361_36163

/-- Given a train of length 500 meters that takes 75 seconds to cross a platform
    and 30 seconds to cross a signal pole, the length of the platform is 750.25 meters. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ)
  (h1 : train_length = 500)
  (h2 : time_platform = 75)
  (h3 : time_pole = 30) :
  let train_speed := train_length / time_pole
  let platform_length := train_speed * time_platform - train_length
  platform_length = 750.25 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l361_36163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_condition_l361_36188

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x - x^2

-- State the theorem
theorem function_inequality_condition (a : ℝ) :
  (∀ p q : ℝ, 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q →
    (f a (p + 1) - f a (q + 1)) / (p - q) > 1) ↔
  a ≥ 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_condition_l361_36188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_hit_correct_l361_36194

/-- The probability that at least one hunter hits the target given n hunters with individual hit probability of 1/3 -/
noncomputable def prob_at_least_one_hit (n : ℕ) : ℝ :=
  1 - (2/3)^n

/-- Theorem stating that the probability of at least one hunter hitting the target
    is equal to 1 minus the probability of all hunters missing -/
theorem prob_at_least_one_hit_correct (n : ℕ) :
  prob_at_least_one_hit n = 1 - (1 - 1/3)^n :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_hit_correct_l361_36194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l361_36102

/-- Represents the distribution of scores in a physics test -/
structure ScoreDistribution where
  score60 : ℝ
  score75 : ℝ
  score85 : ℝ
  score95 : ℝ
  sum_to_one : score60 + score75 + score85 + score95 = 1
  non_negative : score60 ≥ 0 ∧ score75 ≥ 0 ∧ score85 ≥ 0 ∧ score95 ≥ 0

/-- Calculates the mean score given a score distribution -/
def mean (d : ScoreDistribution) : ℝ :=
  60 * d.score60 + 75 * d.score75 + 85 * d.score85 + 95 * d.score95

/-- Determines the median score given a score distribution -/
noncomputable def median (d : ScoreDistribution) : ℝ :=
  if d.score60 + d.score75 > (1/2 : ℝ) then 75 else 85

/-- The main theorem stating the difference between mean and median -/
theorem mean_median_difference (d : ScoreDistribution) 
  (h1 : d.score60 = 0.2) 
  (h2 : d.score75 = 0.5) 
  (h3 : d.score85 = 0.15) : 
  |mean d - median d| = 1.5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l361_36102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l361_36180

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.log (4 * x - 3) / Real.log 2)

theorem domain_of_f : Set ℝ = Set.Ioc (3/4) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l361_36180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fraction_with_20_percent_increase_l361_36171

theorem no_fraction_with_20_percent_increase : 
  ¬∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧
    (Nat.gcd x y = 1) ∧ 
    ((x + 1 : ℚ) / (y + 1) = 6/5 * (x / y)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fraction_with_20_percent_increase_l361_36171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_specific_l361_36116

/-- The eccentricity of an ellipse defined by parametric equations x = a * cos(φ) and y = b * sin(φ) -/
noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (min a b / max a b) ^ 2)

/-- Theorem: The eccentricity of the ellipse defined by x = 4cos(φ) and y = 5sin(φ) is 3/5 -/
theorem ellipse_eccentricity_specific : ellipse_eccentricity 4 5 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_specific_l361_36116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_exists_unique_l361_36192

/-- Definition of a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Definition of collinearity for three points -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- Theorem: Existence and uniqueness of circumcenter for non-collinear points -/
theorem circumcenter_exists_unique (A B C : Point) 
  (h : ¬ collinear A B C) : 
  ∃! O : Point, distance O A = distance O B ∧ distance O B = distance O C := by
  sorry

#check circumcenter_exists_unique

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_exists_unique_l361_36192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eh_length_l361_36118

/-- Represents a trapezoid EFGH with specific properties -/
structure Trapezoid where
  -- Points E, F, G, H
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  -- Q is the intersection of diagonals
  Q : ℝ × ℝ
  -- R is the midpoint of FH
  R : ℝ × ℝ
  -- EF is parallel to GH
  ef_parallel_gh : (F.1 - E.1) * (H.2 - G.2) = (F.2 - E.2) * (H.1 - G.1)
  -- EF = 2 * GH
  ef_twice_gh : dist E F = 2 * dist G H
  -- FG = GH = 37
  fg_eq_gh : dist F G = dist G H
  fg_eq_37 : dist F G = 37
  -- EH is perpendicular to FH
  eh_perp_fh : (H.1 - E.1) * (H.1 - F.1) + (H.2 - E.2) * (H.2 - F.2) = 0
  -- QR = 17
  qr_eq_17 : dist Q R = 17
  -- R is midpoint of FH
  r_midpoint_fh : R = ((F.1 + H.1) / 2, (F.2 + H.2) / 2)

/-- The main theorem stating the length of EH in the trapezoid -/
theorem eh_length (t : Trapezoid) : dist t.E t.H = 16 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eh_length_l361_36118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_and_max_function_l361_36159

noncomputable def f (x : ℝ) := Real.log x - x

theorem geometric_sequence_and_max_function 
  (a b c d : ℝ) 
  (h1 : ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) 
  (h2 : ∀ x : ℝ, x > 0 → f x ≤ f b) : 
  a * d = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_and_max_function_l361_36159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_real_iff_m_in_range_l361_36150

/-- A function f(x) with parameter m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (x - 4) / (m * x^2 + 4 * m * x + 3)

/-- The domain of f is ℝ if and only if m is in [0, 3/4) -/
theorem f_domain_real_iff_m_in_range (m : ℝ) : 
  (∀ x, f m x ≠ 0 → f m x = f m x) ↔ 0 ≤ m ∧ m < 3/4 := by
  sorry

#check f_domain_real_iff_m_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_real_iff_m_in_range_l361_36150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l361_36193

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the area function
noncomputable def area (t : Triangle) : ℝ := 
  1 / 2 * t.a * t.b * Real.sin t.C

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * t.b * Real.cos t.C = t.a * Real.cos t.C + t.c * Real.cos t.A)
  (h2 : t.b = 2)
  (h3 : t.c = Real.sqrt 7) :
  t.C = π / 3 ∧ t.a = 3 ∧ area t = 3 * Real.sqrt 3 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l361_36193
