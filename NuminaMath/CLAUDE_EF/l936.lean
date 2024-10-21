import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_side_length_l936_93687

/-- Given a triangle ABC with angles A, B, C and two sides of lengths 10 and 13,
    prove that the maximum length of the third side is √399 when cos 3A + cos 3B + cos 3C = 1 -/
theorem max_third_side_length (A B C : ℝ) (a b c : ℝ) :
  Real.cos (3 * A) + Real.cos (3 * B) + Real.cos (3 * C) = 1 →
  ((a = 10 ∧ b = 13) ∨ (a = 13 ∧ b = 10)) →
  c ≤ Real.sqrt 399 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_side_length_l936_93687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_498_to_500_l936_93611

def digit_sequence : ℕ → ℕ
  | 0 => 2
  | n + 1 => 
    let prev := digit_sequence n
    if prev % 10 < 9 then prev + 1
    else if prev < 29 then prev + 1
    else if prev < 299 then prev + 1
    else prev + 1

def digits_at (n : ℕ) : ℕ := 
  let num := digit_sequence (n / 3)
  let str := toString num
  let last_three := str.takeRight 3
  last_three.toNat!

theorem digits_498_to_500 : digits_at 498 = 205 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_498_to_500_l936_93611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_test_ratio_l936_93635

theorem math_test_ratio (grant_score john_score hunter_score : ℝ) : 
  (grant_score = john_score + 10) →
  (hunter_score = 45) →
  (grant_score = 100) →
  (john_score / hunter_score = 2 / 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_test_ratio_l936_93635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l936_93615

/-- Given a hyperbola with the following properties:
    - Standard form equation: x²/m² - y²/n² = 1
    - m > 0, n > 0
    - Focal length = √7
    - Asymptote equation: y = √6 * x
    Prove that the equation of the hyperbola is 4x² - (2/3)y² = 1 -/
theorem hyperbola_equation (m n : ℝ) (hm : m > 0) (hn : n > 0)
  (h_focal : m^2 + n^2 = (Real.sqrt 7 / 2)^2)
  (h_asymptote : n / m = Real.sqrt 6) :
  ∃ (x y : ℝ), 4 * x^2 - (2/3) * y^2 = 1 ∧ x^2 / m^2 - y^2 / n^2 = 1 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l936_93615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l936_93638

noncomputable def ellipse (k : ℝ) := {(x, y) : ℝ × ℝ | x^2 / 4 + y^2 / k = 1}

noncomputable def eccentricity (k : ℝ) : ℝ :=
  if k < 4 then Real.sqrt ((4 - k) / 4) else Real.sqrt ((k - 4) / k)

theorem ellipse_k_range :
  ∀ k : ℝ, (∃ e : ℝ, e ∈ Set.Ioo (1/2) 1 ∧ eccentricity k = e) ↔ 
    k ∈ Set.union (Set.Ioo 0 3) (Set.Ioi (16/3)) :=
by
  sorry

#check ellipse_k_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l936_93638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l936_93698

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 6 = 1

-- Define the left focus F₁ and right focus F₂
noncomputable def F₁ : ℝ × ℝ := ⟨-Real.sqrt 15, 0⟩
noncomputable def F₂ : ℝ × ℝ := ⟨Real.sqrt 15, 0⟩

-- Define a line passing through F₁
def line_through_F₁ (m : ℝ) (x y : ℝ) : Prop :=
  y - F₁.2 = m * (x - F₁.1)

-- Define points A and B on the left branch of the hyperbola
noncomputable def A : ℝ × ℝ := ⟨-3, 0⟩
noncomputable def B : ℝ × ℝ := ⟨-3, 0⟩

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- State the theorem
theorem min_sum_distances :
  ∀ m : ℝ,
  hyperbola A.1 A.2 →
  hyperbola B.1 B.2 →
  line_through_F₁ m A.1 A.2 →
  line_through_F₁ m B.1 B.2 →
  A.1 < F₁.1 →
  B.1 < F₁.1 →
  ∀ C D : ℝ × ℝ,
  hyperbola C.1 C.2 →
  hyperbola D.1 D.2 →
  line_through_F₁ m C.1 C.2 →
  line_through_F₁ m D.1 D.2 →
  C.1 < F₁.1 →
  D.1 < F₁.1 →
  distance A F₂ + distance B F₂ ≤ distance C F₂ + distance D F₂ →
  distance A F₂ + distance B F₂ = 16 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l936_93698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l936_93601

theorem problem_solution : ∀ a b c : ℤ,
  a ≥ b ∧ b ≥ c ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 - b^2 - c^2 + a*b = 2035 ∧
  a^2 + 3*b^2 + 3*c^2 - 3*a*b - 2*a*c - 2*b*c = -2067 →
  a = 255 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l936_93601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_is_54_l936_93633

/-- Represents a cube with labeled faces -/
structure LabeledCube where
  labels : Fin 6 → ℕ
  consecutive_even : ∀ i j : Fin 6, i.val < j.val → labels i % 2 = 0 ∧ labels j - labels i = 2 * (j.val - i.val)
  opposite_faces_sum : ∃ k : ℕ, ∀ i : Fin 3, labels i + labels (Fin.add i 3) = k
  smallest_is_four : labels 0 = 4

/-- The sum of all numbers on a labeled cube -/
def cube_sum (c : LabeledCube) : ℕ := Finset.sum (Finset.univ : Finset (Fin 6)) c.labels

theorem cube_sum_is_54 (c : LabeledCube) : cube_sum c = 54 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_is_54_l936_93633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_minus_a_l936_93672

theorem sin_pi_minus_a (a : ℝ) (h1 : Real.cos a = Real.sqrt 5 / 3) (h2 : a ∈ Set.Ioo (-π/2) 0) : 
  Real.sin (π - a) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_minus_a_l936_93672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_sequence_l936_93640

def my_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = (-1)^n * n - a n

theorem max_product_of_sequence (a : ℕ → ℚ) :
  my_sequence a →
  a 10 = a 1 →
  ∃ m : ℚ, m = 33/4 ∧ ∀ n : ℕ, n ≥ 1 → a n * a (n + 1) ≤ m := by
  sorry

#check max_product_of_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_sequence_l936_93640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonius_circle_equation_l936_93651

/-- The Apollonius Circle is the locus of points P such that the ratio of the distances
    from P to two fixed points A and B is constant. -/
def ApolloniusCircle (A B : ℝ × ℝ) (lambda : ℝ) : Set (ℝ × ℝ) :=
  {P | dist P A = lambda * dist P B}

/-- The equation of an Apollonius Circle given two fixed points and a ratio. -/
def ApolloniusCircleEquation (A B : ℝ × ℝ) (lambda : ℝ) : ℝ × ℝ → Prop :=
  fun P => (P.1^2 + P.2^2) + (20/3) * P.1 + 4 = 0

theorem apollonius_circle_equation :
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (2, 0)
  let lambda : ℝ := 1/2
  ApolloniusCircle A B lambda = {P | ApolloniusCircleEquation A B lambda P} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonius_circle_equation_l936_93651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_selling_price_approx_l936_93636

/-- Represents a bond with its face value, interest rate, and selling price interest rate. -/
structure Bond where
  faceValue : ℝ
  interestRate : ℝ
  sellingPriceInterestRate : ℝ

/-- Calculates the selling price of a bond. -/
noncomputable def sellingPrice (bond : Bond) : ℝ :=
  (bond.faceValue * bond.interestRate) / bond.sellingPriceInterestRate

/-- The three bonds from the problem. -/
def bondA : Bond := ⟨5000, 0.06, 0.065⟩
def bondB : Bond := ⟨7000, 0.08, 0.075⟩
def bondC : Bond := ⟨10000, 0.05, 0.045⟩

/-- The combined selling price of the three bonds is approximately $23,193.16. -/
theorem combined_selling_price_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs ((sellingPrice bondA + sellingPrice bondB + sellingPrice bondC) - 23193.16) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_selling_price_approx_l936_93636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_and_square_sum_zero_l936_93642

theorem cube_root_and_square_sum_zero : 
  ((-27 : ℝ) ^ (1/3 : ℝ)) + (-Real.sqrt 3)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_and_square_sum_zero_l936_93642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_plus_one_derivative_l936_93660

open Real

theorem sin_cos_plus_one_derivative (x : ℝ) :
  deriv (λ x ↦ sin x * (cos x + 1)) x = cos (2 * x) + cos x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_plus_one_derivative_l936_93660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_even_g_l936_93688

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x - Real.sin x

/-- The function g(x) which is f(x) shifted left by n units -/
noncomputable def g (n : ℝ) (x : ℝ) : ℝ := f (x + n)

/-- A function is even if f(-x) = f(x) for all x -/
def is_even (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x

theorem smallest_n_for_even_g :
  ∃ n : ℝ, n > 0 ∧ is_even (g n) ∧ ∀ m, 0 < m ∧ m < n → ¬is_even (g m) := by
  sorry

#check smallest_n_for_even_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_even_g_l936_93688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sequence_sum_l936_93697

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sequence_term (n : ℕ) : ℕ := factorial n + n

def sequence_sum : ℕ := (List.range 10).map (λ n => sequence_term (n + 1)) |>.sum

theorem units_digit_of_sequence_sum :
  sequence_sum % 10 = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sequence_sum_l936_93697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l936_93661

theorem expression_evaluation : 
  (0.064 : ℝ)^(-(1/3 : ℝ)) - (-1/8 : ℝ)^(0 : ℝ) + 16^(3/4 : ℝ) + (0.25 : ℝ)^(1/2 : ℝ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l936_93661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l936_93644

theorem unique_solution : ∃! x : ℝ, (8 : ℝ)^(x^2 - 6*x + 9) = 1 ∧ x > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l936_93644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_S_l936_93620

noncomputable def i : ℂ := Complex.I

noncomputable def S (n : ℤ) : ℂ := i^n + i^(-n)

theorem distinct_values_of_S :
  ∃ (A : Finset ℂ), (∀ n : ℤ, S n ∈ A) ∧ (Finset.card A = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_S_l936_93620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_le_chord_l936_93648

/-- A circular sector with radius R and central angle less than 180° -/
structure CircularSector where
  R : ℝ
  angle : ℝ
  angle_pos : 0 < angle
  angle_lt_pi : angle < π

/-- A line segment inside a circular sector -/
structure LineSegmentInSector (S : CircularSector) where
  M : ℝ × ℝ
  N : ℝ × ℝ
  inside : Prop  -- Changed from IsInside to Prop

/-- The chord connecting the endpoints of the sector -/
noncomputable def chord_length (S : CircularSector) : ℝ :=
  2 * S.R * Real.sin (S.angle / 2)

/-- Theorem: Any line segment inside a circular sector is no longer than the chord -/
theorem line_segment_le_chord (S : CircularSector) (L : LineSegmentInSector S) :
    Real.sqrt ((L.M.1 - L.N.1)^2 + (L.M.2 - L.N.2)^2) ≤ chord_length S := by
  sorry

#check line_segment_le_chord

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_le_chord_l936_93648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l936_93671

-- Define the points A, B, and D
def A : ℝ × ℝ := (7, 1)
def B : ℝ × ℝ := (5, -3)
def D : ℝ × ℝ := (5, 1)

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the altitude property
def is_altitude (A D : ℝ × ℝ) (line : Set (ℝ × ℝ)) : Prop :=
  D ∈ line ∧ ∀ p ∈ line, (A.1 - D.1) * (p.1 - D.1) + (A.2 - D.2) * (p.2 - D.2) = 0

-- Define the vertical line BC
def line_BC : Set (ℝ × ℝ) := {p | p.1 = 5}

-- Theorem statement
theorem point_C_coordinates :
  ∀ C : ℝ × ℝ,
  Triangle A B C →
  is_altitude A D line_BC →
  C = (5, 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l936_93671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_from_asymptote_angle_l936_93607

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The theorem stating the relation between the asymptote angle and eccentricity -/
theorem hyperbola_eccentricity_from_asymptote_angle (h : Hyperbola) 
  (h_angle : Real.tan (130 * π / 180) = - h.b / h.a) :
  eccentricity h = 1 / Real.cos (50 * π / 180) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_from_asymptote_angle_l936_93607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_ratio_l936_93622

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := dist A B
  let AC := dist A C
  let BC := dist B C
  AB = 4 ∧ AC = 5 ∧ BC = 6

-- Define the angle bisector property
def AngleBisector (A B C M : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ M = (1 - t) • B + t • C ∧
  (dist A B / dist A C) = (dist B M / dist C M)

-- Theorem statement
theorem angle_bisector_ratio (A B C M : ℝ × ℝ) :
  Triangle A B C →
  AngleBisector A B C M →
  dist A M / dist C M = 4 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_ratio_l936_93622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_inequality_l936_93683

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => if x ≥ 0 then (2 : ℝ) ^ x else (2 : ℝ) ^ (-x)

-- State the theorem
theorem range_of_inequality (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = f (-x)) →  -- f is even
  (∀ x : ℝ, x ≥ 0 → f x = (2 : ℝ) ^ x) →  -- f(x) = 2^x for x ≥ 0
  {x : ℝ | f (1 - 2*x) < f 3} = Set.Ioo (-1 : ℝ) 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_inequality_l936_93683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_coloring_iff_odd_l936_93645

/-- A coloring of edges and diagonals of a convex n-gon --/
def Coloring (n : ℕ) := Fin n → Fin n → Fin n

/-- Predicate to check if a coloring satisfies the triangle condition --/
def SatisfiesTriangleCondition (n : ℕ) (c : Coloring n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ∃ (a b d : Fin n), a ≠ b ∧ b ≠ d ∧ a ≠ d ∧
      ({c a b, c b d, c a d} : Finset (Fin n)) = {i, j, k}

/-- Theorem stating that a valid coloring exists if and only if n is odd --/
theorem valid_coloring_iff_odd (n : ℕ) :
  (∃ c : Coloring n, SatisfiesTriangleCondition n c) ↔ Odd n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_coloring_iff_odd_l936_93645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_powers_of_two_l936_93684

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the sequence a_n
noncomputable def a (n : ℕ) : ℤ := floor (n * Real.sqrt 2)

-- Define the set B
def B : Set ℕ := {b : ℕ | ∃ k : ℕ, a b = 2^k}

-- Statement to prove
theorem infinitely_many_powers_of_two : Set.Infinite B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_powers_of_two_l936_93684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l936_93627

noncomputable def data : List ℝ := [10, 6, 8, 5, 6]

noncomputable def mean (xs : List ℝ) : ℝ :=
  (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (λ x => (x - mean xs) ^ 2)).sum / xs.length

theorem variance_of_data : variance data = 16 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l936_93627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_b_sum_bound_l936_93682

def a (n : ℕ) : ℚ := (2^n - 1) / 2^(n-1)

def S (n : ℕ) : ℚ := 2 * n - a n

def b (n : ℕ) : ℚ := 2^(n-1) * a n

theorem a_formula (n : ℕ) : a n = (2^n - 1) / 2^(n-1) := by sorry

theorem b_sum_bound (n : ℕ) : 
  (Finset.range n).sum (λ i => 1 / b (i+1)) < 5/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_b_sum_bound_l936_93682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_men_for_road_project_l936_93624

/-- Represents the road construction project -/
structure RoadProject where
  total_length : ℚ
  total_days : ℚ
  initial_men : ℚ
  days_passed : ℚ
  length_completed : ℚ

/-- Calculates the number of extra men needed to complete the project on time -/
noncomputable def extra_men_needed (project : RoadProject) : ℚ :=
  let remaining_length := project.total_length - project.length_completed
  let remaining_days := project.total_days - project.days_passed
  let initial_rate := project.length_completed / project.days_passed
  let required_rate := remaining_length / remaining_days
  let total_men_needed := (required_rate / initial_rate) * project.initial_men
  total_men_needed - project.initial_men

/-- Theorem stating that 30 extra men are needed for the given project -/
theorem extra_men_for_road_project :
  let project : RoadProject := {
    total_length := 10,
    total_days := 150,
    initial_men := 30,
    days_passed := 50,
    length_completed := 2
  }
  extra_men_needed project = 30 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_men_for_road_project_l936_93624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coefficient_term_l936_93639

/-- Given a binomial (1/2 + 2x)^n where the sum of the binomial coefficients 
    of the first three terms equals 79, prove that the term with the maximum 
    coefficient is the 11th term and has the form 16896x^10. -/
theorem max_coefficient_term (n : ℕ) (x : ℝ) : 
  (Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2 = 79) →
  ∃ k, k = 10 ∧ 
    ∀ j, 0 ≤ j ∧ j ≤ n → 
      (Nat.choose n k : ℝ) * (1/2)^(n-k) * (2*x)^k ≥ (Nat.choose n j : ℝ) * (1/2)^(n-j) * (2*x)^j ∧
    (Nat.choose n k : ℝ) * (1/2)^(n-k) * (2*x)^k = 16896 * x^10 := by
  sorry

#check max_coefficient_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coefficient_term_l936_93639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l936_93630

-- Define the line equation
def line (x y a : ℝ) : Prop := 2 * x - (a + 1) * y + 2 * a = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 8 * y = 0

-- Theorem statement
theorem shortest_chord_length (a : ℝ) :
  ∃ (l : ℝ), l = 2 * Real.sqrt 11 ∧
  ∀ (x y : ℝ), line x y a → circle_eq x y →
  ∀ (x' y' : ℝ), line x' y' a → circle_eq x' y' →
  Real.sqrt ((x - x')^2 + (y - y')^2) ≥ l :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l936_93630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_x_coords_on_log3_graph_l936_93618

noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem product_of_x_coords_on_log3_graph (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = log3 x₁) 
  (h2 : y₂ = log3 x₂) 
  (h3 : (y₁ + y₂) / 2 = 0) : 
  x₁ * x₂ = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_x_coords_on_log3_graph_l936_93618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_plus_two_at_most_two_tangent_lines_l936_93641

/-- The base of the natural logarithm -/
noncomputable def e : ℝ := Real.exp 1

/-- The natural logarithm function -/
noncomputable def ln (x : ℝ) : ℝ := Real.log x

/-- The function f(x) = e^(x-1) -/
noncomputable def f (x : ℝ) : ℝ := e^(x-1)

/-- The function g(x) = ln x - 1 -/
noncomputable def g (x : ℝ) : ℝ := ln x - 1

theorem f_geq_g_plus_two (x : ℝ) (h : x > 0) : f x ≥ g x + 2 := by
  sorry

theorem at_most_two_tangent_lines :
  ∃ (n : ℕ), n ≤ 2 ∧
  ∃ (S : Finset ℝ), Finset.card S = n ∧
  ∀ t ∈ S, ∃ (m b : ℝ),
    (∀ x, m * x + b = f x ↔ x = t) ∧
    (∃ y, m * y + b = g y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_plus_two_at_most_two_tangent_lines_l936_93641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_passing_through_point_l936_93685

theorem quadratic_function_passing_through_point (m : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = x^2 + x + m) ∧ f 1 = -2) → m = -4 := by
  intro h
  rcases h with ⟨f, hf, hf1⟩
  have h1 : f 1 = 1^2 + 1 + m := hf 1
  rw [hf1] at h1
  linarith

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_passing_through_point_l936_93685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_islander_group_theorem_l936_93643

/-- Represents the types of islanders -/
inductive IslanderType
| Knight
| Liar

/-- Represents the possible statements an islander can make -/
inductive Statement
| MoreLiars
| MoreKnights
| Equal
deriving DecidableEq

/-- A group of islanders -/
structure IslanderGroup where
  size : Nat
  knights : Nat
  liars : Nat
  knight_statements : Finset Statement
  liar_statements : Finset Statement

/-- The statement made by an islander based on their type and the actual group composition -/
def makeStatement (t : IslanderType) (g : IslanderGroup) : Statement :=
  if g.knights > g.liars then
    match t with
    | IslanderType.Knight => Statement.MoreKnights
    | IslanderType.Liar => Statement.MoreLiars
  else if g.liars > g.knights then
    match t with
    | IslanderType.Knight => Statement.MoreLiars
    | IslanderType.Liar => Statement.MoreKnights
  else
    match t with
    | IslanderType.Knight => Statement.Equal
    | IslanderType.Liar => Statement.MoreLiars -- Default to MoreLiars for Liars when equal

theorem islander_group_theorem (g : IslanderGroup) :
  g.size = 10 ∧
  g.knights + g.liars = g.size ∧
  g.knight_statements.card + g.liar_statements.card = g.size ∧
  (g.knight_statements.filter (· = Statement.MoreLiars)).card +
    (g.liar_statements.filter (· = Statement.MoreLiars)).card = 5 →
  (g.knight_statements.filter (· = Statement.Equal)).card +
    (g.liar_statements.filter (· = Statement.Equal)).card = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_islander_group_theorem_l936_93643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_yen_received_l936_93612

/-- The exchange rate from AUD to JPY -/
noncomputable def exchange_rate : ℚ := 8800 / 100

/-- The transaction fee as a rational number -/
def transaction_fee : ℚ := 5 / 100

/-- The amount of AUD to exchange -/
def exchange_amount : ℚ := 250

/-- The amount of JPY received after exchange and fee deduction -/
noncomputable def yen_received : ℚ := exchange_amount * exchange_rate * (1 - transaction_fee)

theorem correct_yen_received : yen_received = 20900 := by
  -- Expand the definition of yen_received
  unfold yen_received
  -- Perform the calculation
  norm_num
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_yen_received_l936_93612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_discount_rate_for_given_conditions_l936_93603

/-- Represents the maximum discount rate that can be offered on an item -/
noncomputable def max_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) : ℝ :=
  100 * (1 - (cost_price * (1 + min_profit_margin) / selling_price))

/-- Theorem stating the maximum discount rate for the given conditions -/
theorem max_discount_rate_for_given_conditions :
  max_discount_rate 4 5 0.1 = 12 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_discount_rate_for_given_conditions_l936_93603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_evaluation_l936_93662

theorem complex_expression_evaluation : 
  let x := (Real.sqrt (561^2 - 459^2)) / (4 * (2/7) * 0.15 + 4 * (2/7) / (20/3)) + 4 * Real.sqrt 10
  (x / ((1/3) * Real.sqrt 40)) = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_evaluation_l936_93662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocal_squares_l936_93602

/-- The curve C is defined by the equation x²/4 + y²/3 = 1 -/
def C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Convert polar coordinates (r, θ) to Cartesian coordinates (x, y) -/
noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ := (r * Real.cos θ, r * Real.sin θ)

/-- The points A, B, and C on curve C -/
noncomputable def A (ρ₁ θ : ℝ) : ℝ × ℝ := polar_to_cartesian ρ₁ θ
noncomputable def B (ρ₂ θ : ℝ) : ℝ × ℝ := polar_to_cartesian ρ₂ (θ + 2*Real.pi/3)
noncomputable def D (ρ₃ θ : ℝ) : ℝ × ℝ := polar_to_cartesian ρ₃ (θ + 4*Real.pi/3)

/-- The theorem to be proved -/
theorem sum_of_reciprocal_squares (ρ₁ ρ₂ ρ₃ θ : ℝ) 
  (hA : C (A ρ₁ θ).1 (A ρ₁ θ).2)
  (hB : C (B ρ₂ θ).1 (B ρ₂ θ).2)
  (hD : C (D ρ₃ θ).1 (D ρ₃ θ).2) :
  1 / ρ₁^2 + 1 / ρ₂^2 + 1 / ρ₃^2 = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocal_squares_l936_93602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_inequality_in_range_closer_to_f_l936_93657

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Theorem 1: Tangent line equation
theorem tangent_line_at_e :
  ∃ (m b : ℝ), m * Real.exp 1 + b = 1 ∧ 
  ∀ x, m * x + b = (1 / Real.exp 1) * x := by sorry

-- Theorem 2: Inequality for 1 < x < e^2
theorem inequality_in_range (x : ℝ) (h1 : 1 < x) (h2 : x < Real.exp 2) :
  x < (2 + f x) / (2 - f x) := by sorry

-- Theorem 3: Comparison of distances
theorem closer_to_f (x a : ℝ) (ha : a ≥ 2) (hx : x ≥ 1) :
  |Real.exp 1 / x - f x| < |Real.exp (x - 1) + a - f x| := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_inequality_in_range_closer_to_f_l936_93657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_perpendicular_distance_perpendicular_distance_problem_l936_93696

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the centroid of a triangle -/
noncomputable def centroid (t : Triangle) : Point :=
  { x := (t.A.x + t.B.x + t.C.x) / 3,
    y := (t.A.y + t.B.y + t.C.y) / 3 }

/-- The theorem stating that the perpendicular distance from the centroid to a line
    is the average of the perpendicular distances from the vertices to the line -/
theorem centroid_perpendicular_distance (t : Triangle) (line_y : ℝ) :
  let G := centroid t
  (G.y - line_y) = ((t.A.y - line_y) + (t.B.y - line_y) + (t.C.y - line_y)) / 3 := by
  sorry

/-- The main theorem for the problem -/
theorem perpendicular_distance_problem (A B C : Point) (line_y : ℝ) 
    (h1 : A.y - line_y = 14)
    (h2 : B.y - line_y = 8)
    (h3 : C.y - line_y = 26) :
  let t : Triangle := { A := A, B := B, C := C }
  let G := centroid t
  G.y - line_y = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_perpendicular_distance_perpendicular_distance_problem_l936_93696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_equality_trig_ratio_given_condition_l936_93670

-- Part 1
theorem trig_sum_equality : 
  Real.cos (9 * Real.pi / 4) + Real.tan (- Real.pi / 4) + Real.sin (21 * Real.pi) = Real.sqrt 2 / 2 - 1 := by sorry

-- Part 2
theorem trig_ratio_given_condition (θ : Real) (h : Real.sin θ = 2 * Real.cos θ) : 
  (Real.sin θ)^2 + 2 * Real.sin θ * Real.cos θ / (2 * (Real.sin θ)^2 - (Real.cos θ)^2) = 8 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_equality_trig_ratio_given_condition_l936_93670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l936_93649

theorem sufficient_not_necessary : 
  (∀ x : ℝ, (abs x < 1 → x^2 - 2*x - 3 < 0)) ∧ 
  (∃ x : ℝ, (x^2 - 2*x - 3 < 0 ∧ ¬(abs x < 1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l936_93649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grouping_arrangements_l936_93608

def number_of_drivers : ℕ := 4
def number_of_collectors : ℕ := 4
def group_size : ℕ := 2

def permutations (n : ℕ) (r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

def number_of_different_groupings (drivers : ℕ) (collectors : ℕ) (group_size : ℕ) : ℕ :=
  permutations collectors collectors

theorem grouping_arrangements :
  permutations number_of_collectors number_of_collectors = 
  number_of_different_groupings number_of_drivers number_of_collectors group_size :=
by
  unfold number_of_different_groupings
  rfl

#eval number_of_different_groupings number_of_drivers number_of_collectors group_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grouping_arrangements_l936_93608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_in_specific_cylinder_l936_93689

/-- The longest segment that can fit inside a cylinder -/
noncomputable def longest_segment (radius : ℝ) (height : ℝ) : ℝ :=
  Real.sqrt (height^2 + (2 * radius)^2)

/-- Theorem stating the longest segment in a cylinder with radius 5 cm and height 12 cm -/
theorem longest_segment_in_specific_cylinder :
  longest_segment 5 12 = Real.sqrt 244 := by
  sorry

/-- Approximate value of the longest segment -/
def approx_longest_segment : ℚ :=
  3.95284707521 -- This is an approximation of √244

#eval approx_longest_segment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_in_specific_cylinder_l936_93689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_collinearity_condition_l936_93694

-- Part 1: Trigonometric expression
theorem trig_expression_simplification (x : ℝ) :
  (Real.sin (-x) * Real.cos (Real.pi - x)) / (Real.sin (Real.pi + x) * Real.cos (2*Real.pi - x)) -
  (Real.sin (Real.pi - x) * Real.cos (Real.pi + x)) / (Real.cos (Real.pi/2 - x) * Real.cos (-x)) = 0 := by
  sorry

-- Part 2: Collinearity of points
def point := ℝ × ℝ

def OA (k : ℝ) : point := (k, 12)
def OB : point := (4, 5)
def OC (k : ℝ) : point := (10, k)

def collinear (A B C : point) : Prop :=
  ∃ t : ℝ, B.1 - A.1 = t * (C.1 - B.1) ∧ B.2 - A.2 = t * (C.2 - B.2)

theorem collinearity_condition (k : ℝ) :
  collinear (OA k) OB (OC k) ↔ k = -2 ∨ k = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_collinearity_condition_l936_93694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_functions_sum_l936_93695

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (10^x + 1) / Real.log 10 + a * x
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := (4^x - b) / 2^x

-- State the theorem
theorem even_odd_functions_sum (a b : ℝ) : 
  (∀ x, f a x = f a (-x)) → -- f is even
  (∀ x, g b x = -g b (-x)) → -- g is odd
  a + b = 1/2 := by
  sorry

#check even_odd_functions_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_functions_sum_l936_93695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l936_93681

/-- An ellipse with center at origin, right vertex at (2,0), intersected by the line y = x - 1 at points M and N, where the x-coordinate of the midpoint of MN is 2/3, has the standard equation x^2/4 + y^2/2 = 1. -/
theorem ellipse_equation (M N : ℝ × ℝ) : 
  (∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | p.1^2/4 + p.2^2/2 = 1} ↔ 
    (∃ t : ℝ, (x, y) = (2 * Real.cos t, Real.sqrt 2 * Real.sin t))) ∧ 
  (2, 0) ∈ {p : ℝ × ℝ | p.1^2/4 + p.2^2/2 = 1} ∧
  M.2 = M.1 - 1 ∧ N.2 = N.1 - 1 ∧
  M ∈ {p : ℝ × ℝ | p.1^2/4 + p.2^2/2 = 1} ∧
  N ∈ {p : ℝ × ℝ | p.1^2/4 + p.2^2/2 = 1} ∧
  (M.1 + N.1) / 2 = 2/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l936_93681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_theorem_l936_93617

noncomputable def train_passing_time (length_A length_B speed_A speed_B : ℝ) : ℝ :=
  (length_A + length_B) / (speed_A + speed_B)

theorem train_passing_theorem :
  let length_A : ℝ := 235
  let length_B : ℝ := 260
  let speed_A : ℝ := 108 * 1000 / 3600  -- Convert km/h to m/s
  let speed_B : ℝ := 90 * 1000 / 3600   -- Convert km/h to m/s
  train_passing_time length_A length_B speed_A speed_B = 9 := by
  -- Unfold the definition of train_passing_time
  unfold train_passing_time
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_theorem_l936_93617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_at_neg_two_l936_93621

/-- A polynomial of degree 4 satisfying specific conditions -/
noncomputable def P : ℝ → ℝ := sorry

/-- P is a polynomial of degree 4 -/
axiom P_degree : ∃ (a b c d e : ℝ), ∀ x, P x = a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- P satisfies the given conditions -/
axiom P_conditions : P 0 = 1 ∧ P 1 = 1 ∧ P 2 = 4 ∧ P 3 = 9 ∧ P 4 = 16

/-- Theorem: P(-2) = 19 -/
theorem P_at_neg_two : P (-2) = 19 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_at_neg_two_l936_93621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_25_l936_93664

/-- The acute angle between the minute hand and the hour hand on a standard clock -/
noncomputable def clock_angle (hours : ℕ) (minutes : ℕ) : ℝ :=
  let hour_angle : ℝ := (hours % 12 + minutes / 60 : ℝ) * 30
  let minute_angle : ℝ := minutes * 6
  min (|hour_angle - minute_angle|) (360 - |hour_angle - minute_angle|)

theorem clock_angle_at_7_25 : 
  clock_angle 7 25 = 72.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_25_l936_93664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_implies_b_equals_3_l936_93647

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 1 / (3 * x + b)

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

-- Theorem statement
theorem inverse_function_implies_b_equals_3 (b : ℝ) :
  (∀ x, f b (f_inv x) = x) → b = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_implies_b_equals_3_l936_93647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_argument_range_l936_93604

theorem complex_argument_range (z : ℂ) (h : Complex.abs (2 * z + 1 / z) = 1) :
  ∃ (k : ℕ) (θ : ℝ), k ∈ ({0, 1} : Set ℕ) ∧
  Complex.arg z = θ ∧
  k * Real.pi + Real.pi / 2 - Real.arccos (3 / 4) / 2 ≤ θ ∧
  θ ≤ k * Real.pi + Real.pi / 2 + Real.arccos (3 / 4) / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_argument_range_l936_93604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_validColorings_recurrence_validColorings_initial_conditions_validColorings_formula_l936_93626

/-- The number of valid colorings for n points -/
def validColorings (n : ℕ) : ℚ := (3^n + (-1)^n : ℚ) / 2

/-- Proposition: The number of valid colorings for n points satisfies the recurrence relation -/
theorem validColorings_recurrence (n : ℕ) (h : n ≥ 2) :
  validColorings n = 2 * validColorings (n-1) + 3 * validColorings (n-2) :=
sorry

/-- Proposition: The initial conditions for the recurrence are correct -/
theorem validColorings_initial_conditions :
  validColorings 1 = 5 ∧ validColorings 2 = 13 :=
sorry

/-- Main theorem: The number of valid colorings for n points is (3^n + (-1)^n) / 2 -/
theorem validColorings_formula (n : ℕ) :
  validColorings n = (3^n + (-1)^n : ℚ) / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_validColorings_recurrence_validColorings_initial_conditions_validColorings_formula_l936_93626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l936_93613

open Real

theorem triangle_theorem (A B C a b c : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  -- Given condition
  (Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) →
  (A = 2 * B) →
  -- Conclusion
  (C = 5 * π / 8 ∧ 2 * a^2 = b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l936_93613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_x_minus_2_positive_l936_93653

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^3 - 8 else -(-x)^3 - 8

-- State the theorem
theorem f_x_minus_2_positive :
  (∀ x, f x = f (-x)) →  -- f is even
  (∀ x ≥ 0, f x = x^3 - 8) →  -- f(x) = x^3 - 8 for x ≥ 0
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_x_minus_2_positive_l936_93653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_of_three_l936_93679

def card_numbers : Finset ℕ := {6, 7, 8, 9}

def is_multiple_of_three (n : ℕ) : Bool :=
  n % 3 = 0

theorem probability_multiple_of_three :
  (card_numbers.filter (fun n => is_multiple_of_three n)).card / card_numbers.card = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_of_three_l936_93679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_point_coordinates_l936_93699

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line segment defined by two points -/
structure LineSegment where
  start : Point
  end_ : Point

/-- Translation of a point by a vector -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

/-- Theorem: Given the conditions, prove that Q has coordinates (2, 2) -/
theorem translated_point_coordinates :
  let A : Point := { x := -3, y := 2 }
  let B : Point := { x := 1, y := 1 }
  let AB : LineSegment := { start := A, end_ := B }
  let P : Point := { x := -2, y := 3 }
  let dx : ℝ := P.x - A.x
  let dy : ℝ := P.y - A.y
  let Q : Point := translate B dx dy
  Q.x = 2 ∧ Q.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_point_coordinates_l936_93699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_area_l936_93691

/-- A circle ω with points P and Q on its circumference, and tangent lines at P and Q intersecting on the x-axis -/
structure SpecialCircle where
  ω : Set (ℝ × ℝ)
  center : ℝ × ℝ
  radius : ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  is_circle : ∀ (x y : ℝ), (x, y) ∈ ω ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2
  P_on_circle : P ∈ ω
  Q_on_circle : Q ∈ ω
  P_coords : P = (7, 15)
  Q_coords : Q = (11, 9)
  tangent_intersection : ∃ (x : ℝ), 
    let slope_PQ := (Q.2 - P.2) / (Q.1 - P.1)
    let perp_slope := -1 / slope_PQ
    let y_intercept := P.2 - perp_slope * P.1
    x = -y_intercept / perp_slope ∧ (x, 0) ∈ ω

/-- The area of the special circle is 709π -/
theorem special_circle_area (c : SpecialCircle) : Real.pi * c.radius^2 = 709 * Real.pi := by
  sorry

#check special_circle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_area_l936_93691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_to_cos_transformations_l936_93634

/-- Theorem: Transformations of sin(x) to cos(2x + π/6) --/
theorem sin_to_cos_transformations (x : ℝ) :
  (∀ (y : ℝ), y = Real.sin x → 
    (y = Real.cos (2 * (x + π/3) - π/2) ∧
     y = Real.cos (2 * (x - 2*π/3) - π/2) ∧
     y = Real.cos (2 * (x + 2*π/3) - π/2))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_to_cos_transformations_l936_93634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triples_eq_134_l936_93673

def valid_triple (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 60 ∧ a * b = c

def count_valid_triples : ℕ :=
  (Finset.range 60).sum (λ a => 
    (Finset.range 60).sum (λ b => 
      (Finset.range 60).sum (λ c => 
        if (1 ≤ a + 1 && a + 1 ≤ b + 1 && b + 1 ≤ c + 1 && c + 1 ≤ 60 && (a + 1) * (b + 1) = c + 1)
        then 1 else 0)))

theorem count_valid_triples_eq_134 : count_valid_triples = 134 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triples_eq_134_l936_93673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l936_93686

-- Define the triangle ABC
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = Real.pi

-- Theorem statement
theorem triangle_properties 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_order : a < b ∧ b < c)
  (h_sin_A : Real.sin A = (Real.sqrt 3 * a) / (2 * b)) :
  B = Real.pi/3 ∧ 
  (a = 2 ∧ b = Real.sqrt 7 → c = 3 ∧ (1/2 * a * c * Real.sin B = (3 * Real.sqrt 3) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l936_93686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_comparison_l936_93692

open Real

-- Define the periodic property of cosine
axiom cos_periodic (θ : ℝ) (n : ℤ) : cos (θ + 2 * π * n) = cos θ

-- Define the symmetry property of cosine
axiom cos_symmetry (θ : ℝ) : cos (-θ) = cos θ

-- Define that cosine is decreasing in [0, π]
axiom cos_decreasing {x y : ℝ} (hx : 0 ≤ x) (hy : x ≤ y) (hypi : y ≤ π) : cos y ≤ cos x

-- Convert degrees to radians
noncomputable def deg_to_rad (x : ℝ) : ℝ := x * π / 180

-- State the theorem
theorem cos_comparison : cos (deg_to_rad (-508)) < cos (deg_to_rad (-144)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_comparison_l936_93692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_circle_radius_l936_93668

open Real

-- Define the angle in radians
noncomputable def angle : ℝ := π / 3

-- Define the radius of the smaller circle
variable (r : ℝ)

-- Define the radius of the larger circle
noncomputable def R (r : ℝ) : ℝ := 3 * r

-- Assume these definitions exist (we'll need to define or import them properly in a real implementation)
axiom Circle : Type
axiom Point : Type
axiom inscribed_in_angle : Circle → ℝ → Prop
axiom externally_tangent : Circle → Circle → Prop
axiom circle_radius : Circle → ℝ
axiom point_on_angle_bisector : Point → ℝ → Prop

-- Theorem statement
theorem larger_circle_radius (r : ℝ) (h : r > 0) :
  let angle := π / 3
  let R := 3 * r
  (∃ (c1 c2 : Circle) (pt : Point),
    inscribed_in_angle c1 angle ∧
    inscribed_in_angle c2 angle ∧
    externally_tangent c1 c2 ∧
    circle_radius c1 = r ∧
    circle_radius c2 = R ∧
    point_on_angle_bisector pt angle) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_circle_radius_l936_93668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l936_93606

/-- Represents a parabola with equation y² = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- The point (40, 30) lies on the parabola -/
def point_on_parabola (par : Parabola) : Prop :=
  30^2 = 2 * par.p * 40

/-- The distance from the focus to the vertex of the parabola -/
noncomputable def focus_to_vertex (par : Parabola) : ℝ :=
  par.p / 2

/-- Theorem stating that if (40, 30) is on the parabola, then the focus-to-vertex distance is 45/8 -/
theorem parabola_focus_distance (par : Parabola) 
    (h : point_on_parabola par) : focus_to_vertex par = 45 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l936_93606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l936_93629

theorem smallest_number_divisible (n : ℕ) : n = 65517 ↔ 
  (∀ d : ℕ, d ∈ [12, 16, 18, 21, 28, 35, 39] → (n - 3) % d = 0) ∧
  (∀ m : ℕ, m < n → ∃ d : ℕ, d ∈ [12, 16, 18, 21, 28, 35, 39] ∧ (m - 3) % d ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l936_93629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camp_food_consumption_l936_93631

/-- Calculates the total food consumption for dogs and puppies in a day -/
noncomputable def total_food_consumption (num_puppies num_dogs : ℕ) 
                           (dog_meal_frequency : ℕ) 
                           (dog_meal_amount : ℚ) : ℚ :=
  let puppy_meal_frequency := 3 * dog_meal_frequency
  let puppy_meal_amount := dog_meal_amount / 2
  let dog_daily_consumption := dog_meal_frequency * dog_meal_amount
  let puppy_daily_consumption := puppy_meal_frequency * puppy_meal_amount
  (num_dogs : ℚ) * dog_daily_consumption + (num_puppies : ℚ) * puppy_daily_consumption

/-- Theorem stating the total food consumption for the given conditions -/
theorem camp_food_consumption :
  total_food_consumption 4 3 3 4 = 108 := by
  -- Unfold the definition of total_food_consumption
  unfold total_food_consumption
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_camp_food_consumption_l936_93631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l936_93666

-- Define the sequence S_n
def S (n : ℕ) : ℚ := n * (n + 1) / 2

-- Define the sequence a_n
def a (n : ℕ) : ℚ := n

-- Define the sequence b_n
def b (n : ℕ) : ℚ := a (n + 1) / (2 * S n * S (n + 1))

-- Define the sequence T_n
def T (n : ℕ) : ℚ := (n^2 + 3*n) / (2 * (n + 1) * (n + 2))

theorem sequence_properties :
  (∀ n : ℕ, a n = n) ∧
  (∀ n : ℕ, T n = (n^2 + 3*n) / (2 * (n + 1) * (n + 2))) ∧
  (∀ l : ℚ, (∃ n : ℕ+, T n - l * a n ≥ 3 * l) ↔ l ≤ 1/12) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l936_93666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_sum_l936_93667

/-- A parabola in the xy-plane with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The vertex of a parabola -/
noncomputable def vertex (p : Parabola) : Point :=
  { x := -p.b^2 / (4 * p.a) + p.c,
    y := -p.b / (2 * p.a) }

theorem parabola_sum (p : Parabola) : 
  p.x_coord (-9) = 0 ∧ vertex p = Point.mk (-1) (-10) → p.a + p.b + p.c = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_sum_l936_93667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_area_proof_l936_93674

-- Define the curve f(x) = ax^2 + 2
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2 * a * x

theorem curve_and_area_proof (a : ℝ) :
  (∀ x, f_derivative a x = 2 * a * x) →  -- Derivative condition
  (f_derivative a 1 = 2) →  -- Tangent parallel to 2x - y + 1 = 0 at x = 1
  (∀ x, f a x = x^2 + 2) ∧  -- Prove f(x) = x^2 + 2
  (∫ x in (0 : ℝ)..(2 : ℝ), max (f a x) (3 * x) - min (f a x) (3 * x) = 1) :=  -- Prove area = 1
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_area_proof_l936_93674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_theorem_l936_93637

theorem power_sum_theorem (x : ℝ) (h : (2 : ℝ)^x + (2 : ℝ)^(-x) = 5) : 
  (4 : ℝ)^x + (4 : ℝ)^(-x) = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_theorem_l936_93637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_sum_fifty_l936_93654

theorem existence_of_sum_fifty (x : Fin 100 → ℝ) 
  (h : ∀ i : Fin 100, 0 ≤ x i ∧ x i ≤ 1) :
  ∃ y : ℝ, 0 ≤ y ∧ y ≤ 1 ∧ 
    (Finset.univ : Finset (Fin 100)).sum (λ i => |y - x i|) = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_sum_fifty_l936_93654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_smallest_region_l936_93614

-- Define the curves
def abs_curve (x : ℝ) : ℝ := abs x
def circle_curve (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the region
def enclosed_region (x y : ℝ) : Prop :=
  y ≥ abs_curve x ∧ circle_curve x y

-- Define the area of the region
noncomputable def area_of_region : ℝ := 
  sorry -- Placeholder for the actual area calculation

-- Theorem statement
theorem area_of_smallest_region :
  area_of_region = 9 * Real.pi / 4 := by
  sorry -- Placeholder for the proof

#check area_of_smallest_region

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_smallest_region_l936_93614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminating_decimal_expansion_of_7_over_250_l936_93693

theorem terminating_decimal_expansion_of_7_over_250 :
  ∃ (n : ℕ) (k : ℕ), (7 : ℚ) / 250 = (n : ℚ) / (10 ^ k) ∧ (n : ℚ) / (10 ^ k) = 0.028 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminating_decimal_expansion_of_7_over_250_l936_93693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_difference_quotient_at_two_l936_93656

noncomputable def f (x : ℝ) : ℝ := (x + 1) / x

theorem limit_difference_quotient_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 0 < |h| ∧ |h| < δ →
    |(f (2 + h) - f 2) / h + 1/4| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_difference_quotient_at_two_l936_93656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_sum_l936_93678

/-- Given two vectors a and b in a real inner product space with specified magnitudes and angle between them, prove that the magnitude of their sum is √37. -/
theorem magnitude_of_vector_sum 
  (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) 
  (ha : ‖a‖ = 4) 
  (hb : ‖b‖ = 3) 
  (hab : inner a b = ‖a‖ * ‖b‖ * (1 / 2)) : 
  ‖a + b‖ = Real.sqrt 37 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_sum_l936_93678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_closed_form_l936_93623

/-- The sum of the series for non-negative integer n -/
noncomputable def seriesSum (n : ℕ) : ℝ :=
  ∑' k, (-1)^k * (1/3)^k * (n.choose (2*k + 1))

/-- The closed form expression for the sum -/
noncomputable def closedForm (n : ℕ) : ℝ :=
  2^n * 3^((1 - n) / 2 : ℝ) * Real.sin (n * Real.pi / 6)

/-- Theorem stating the equality of the series sum and the closed form -/
theorem series_sum_equals_closed_form (n : ℕ) :
  seriesSum n = closedForm n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_closed_form_l936_93623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_t4_minus_t2sin2t_l936_93609

theorem sqrt_t4_minus_t2sin2t (t : ℝ) : 
  Real.sqrt (t^4 - t^2 * Real.sin t^2) = |t| * Real.sqrt (t^2 - Real.sin t^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_t4_minus_t2sin2t_l936_93609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_extremum_implies_a_equals_three_l936_93677

/-- The function f(x) defined as (x^2 + a) / (x + 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a) / (x + 1)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := (x^2 + 2*x - a) / ((x + 1)^2)

/-- Theorem stating that if f(x) has an extremum at x = 1, then f'(1) = 0 implies a = 3 -/
theorem extremum_condition (a : ℝ) : 
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x - 1| ∧ |x - 1| < ε → f a x ≤ f a 1) →
  f_derivative a 1 = 0 →
  a = 3 := by
  sorry

/-- Main theorem: If f(x) has an extremum at x = 1, then a = 3 -/
theorem extremum_implies_a_equals_three (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x - 1| ∧ |x - 1| < ε → f a x ≤ f a 1) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_extremum_implies_a_equals_three_l936_93677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l936_93669

theorem angle_in_third_quadrant (α : ℝ) 
  (h1 : Real.sin α < 0) (h2 : Real.cos α < 0) : 
  α ∈ Set.Icc π (3 * π / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l936_93669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_at_eight_l936_93680

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  4 * x^2 + 4 * y^2 = -a^2 + 16 * a - 32 ∧ 2 * x * y = a

-- Define the magnitude function
noncomputable def magnitude (x y : ℝ) : ℝ :=
  Real.sqrt ((x + y)^2)

-- Theorem statement
theorem max_magnitude_at_eight :
  ∃ (x y : ℝ), system x y 8 ∧
  ∀ (a x' y' : ℝ), system x' y' a → magnitude x y ≥ magnitude x' y' := by
  sorry

#check max_magnitude_at_eight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_at_eight_l936_93680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_spheres_l936_93605

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r ^ 3

theorem volume_between_spheres (r₁ r₂ : ℝ) (h₁ : r₁ = 5) (h₂ : r₂ = 8) :
  sphere_volume r₂ - sphere_volume r₁ = 516 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_spheres_l936_93605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l936_93690

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
noncomputable def area_triangle (A B C : Point) : ℝ :=
  abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y)) / 2

/-- Given an ellipse C and a point P on it, proves properties about the ellipse and a triangle formed by intersecting lines -/
theorem ellipse_properties (C : Ellipse) (P : Point)
  (h_on_ellipse : P.x^2 / C.a^2 + P.y^2 / C.b^2 = 1)
  (h_P : P.x = 1 ∧ P.y = Real.sqrt 2 / 2)
  (h_foci_sum : ∃ F₁ F₂ : Point, |P.x - F₁.x| + |P.x - F₂.x| = 2 * Real.sqrt 2) :
  (∃ a b : ℝ, C.a = a ∧ C.b = b ∧ a^2 = 2 ∧ b^2 = 1) ∧
  (∃ S : ℝ, S = Real.sqrt 2 / 2 ∧
    ∀ A B : Point, (∃ m : ℝ, A.x - 1 = m * A.y ∧ B.x - 1 = m * B.y ∧
      A.x^2 / C.a^2 + A.y^2 / C.b^2 = 1 ∧ B.x^2 / C.a^2 + B.y^2 / C.b^2 = 1) →
    area_triangle A ⟨0, 0⟩ B ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l936_93690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_13_7743_to_hundredth_l936_93610

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The statement that rounding 13.7743 to the nearest hundredth equals 13.77 -/
theorem round_13_7743_to_hundredth :
  roundToHundredth 13.7743 = 13.77 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_13_7743_to_hundredth_l936_93610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_pear_equivalence_l936_93632

/-- Represents the cost of a fruit in an arbitrary unit -/
structure Cost where
  value : ℚ

instance : HMul ℕ Cost Cost where
  hMul n c := ⟨n * c.value⟩

/-- The cost ratio between bananas and apples -/
axiom banana_apple_ratio : ∀ (c : Cost), (6 : ℕ) * c = (4 : ℕ) * c

/-- The cost ratio between apples and oranges -/
axiom apple_orange_ratio : ∀ (c : Cost), (5 : ℕ) * c = (3 : ℕ) * c

/-- The cost ratio between oranges and pears -/
axiom orange_pear_ratio : ∀ (c : Cost), (4 : ℕ) * c = (7 : ℕ) * c

/-- Theorem stating that 36 bananas cost as much as 28 pears -/
theorem banana_pear_equivalence : 
  ∀ (c : Cost), (36 : ℕ) * c = (28 : ℕ) * c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_pear_equivalence_l936_93632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l936_93665

/-- The ellipse defined by x²/4 + y²/9 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) + (p.2^2 / 9) = 1}

/-- The line defined by 2x + y - 6 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p | 2 * p.1 + p.2 - 6 = 0}

/-- The distance from a point to the line -/
noncomputable def distToLine (p : ℝ × ℝ) : ℝ :=
  |2 * p.1 + p.2 - 6| / Real.sqrt 5

/-- The theorem stating the minimum distance from any point on the ellipse to the line,
    where the connecting line makes a 45° angle with the given line -/
theorem min_distance_ellipse_to_line :
  ∀ p ∈ Ellipse, (distToLine p / Real.sin (π/4) : ℝ) ≥ Real.sqrt 10 / 5 := by
  sorry

#check min_distance_ellipse_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l936_93665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadratic_b_l936_93652

/-- A quadratic function f(x) = (1/12)x^2 + ax + b that intersects the x-axis at two points
    and the y-axis at one point, with a special point T(3, 3) equidistant from these intersections. -/
structure SpecialQuadratic where
  a : ℝ
  b : ℝ
  /-- x-coordinates of the intersection points with the x-axis -/
  x₁ : ℝ
  x₂ : ℝ
  /-- The function f(x) -/
  f : ℝ → ℝ
  f_def : ∀ x, f x = (1/12) * x^2 + a * x + b
  /-- f(x) intersects x-axis at x₁ and x₂ -/
  root_x₁ : f x₁ = 0
  root_x₂ : f x₂ = 0
  /-- Point T(3, 3) is equidistant from (x₁, 0), (x₂, 0), and (0, b) -/
  equidistant : (3 - x₁)^2 + 3^2 = (3 - x₂)^2 + 3^2 ∧ (3 - x₁)^2 + 3^2 = 3^2 + (3 - b)^2

/-- The value of b in a SpecialQuadratic is always -6 -/
theorem special_quadratic_b (sq : SpecialQuadratic) : sq.b = -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadratic_b_l936_93652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crystal_run_distance_l936_93663

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the Euclidean distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The distance of the final segment in Crystal's run is √2 miles -/
theorem crystal_run_distance :
  let start := Point.mk 0 0
  let end_point := Point.mk 1 1
  distance start end_point = Real.sqrt 2 := by
  sorry

#eval "Crystal's run distance theorem is defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crystal_run_distance_l936_93663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_not_sum_of_few_odd_squares_l936_93675

theorem infinitely_many_not_sum_of_few_odd_squares : 
  ∃ f : ℕ → ℕ, 
    (∀ n, ∃ p q : ℕ, 
      Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ 
      (∃ k₁, p = 8 * k₁ + 3) ∧ 
      (∃ k₂, q = 8 * k₂ + 3) ∧ 
      f n = 2 * p * q) ∧
    (∀ n, ∀ (squares : Finset ℕ), 
      (∀ s ∈ squares, ∃ k, s = (2 * k + 1)^2) →
      squares.card < 10 →
      f n ≠ (squares.sum id)) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_not_sum_of_few_odd_squares_l936_93675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_cube_root_16000_l936_93616

theorem simplify_cube_root_16000 : ∃ (a b : ℕ+), 
  (a = 4 ∧ b = 25) ∧ 
  (∀ (c d : ℕ+), c * (d : ℝ)^(1/3) = 16000^(1/3) → d ≥ b) ∧
  (4 : ℝ) * 25^(1/3) = 16000^(1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_cube_root_16000_l936_93616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_cost_calculation_l936_93625

/-- Represents a city in the trip plan -/
structure City where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two cities -/
noncomputable def distance (a b : City) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

/-- Calculates the cost of flying between two cities -/
noncomputable def flyCost (a b : City) : ℝ :=
  150 + 0.12 * distance a b

theorem trip_cost_calculation (a b c : City) :
  a.x = c.x ∧ b.y = c.y ∧  -- C is directly south of B and directly west of A
  distance a b = 4000 ∧
  distance b c = 3000 ∧
  (a.x - b.x)^2 + (a.y - b.y)^2 = 4000^2 →  -- Pythagorean theorem
  flyCost a b = 630 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_cost_calculation_l936_93625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_females_in_town_l936_93646

/-- Represents the population of a town with a given male-to-female ratio --/
structure TownPopulation where
  total : ℕ
  male_ratio : ℕ
  female_ratio : ℕ

/-- Calculates the number of females in the town --/
def females (town : TownPopulation) : ℕ :=
  town.total * town.female_ratio / (town.male_ratio + town.female_ratio)

/-- Theorem stating that in a town of 480 people with a 3:5 male-to-female ratio, there are 300 females --/
theorem females_in_town (town : TownPopulation) 
    (h1 : town.total = 480) 
    (h2 : town.male_ratio = 3) 
    (h3 : town.female_ratio = 5) : 
  females town = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_females_in_town_l936_93646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_t_l936_93676

noncomputable def f (t k x : ℝ) : ℝ := (1/3) * x^3 - (t/2) * x^2 + k * x

noncomputable def f' (t k x : ℝ) : ℝ := x^2 - t * x + k

theorem find_t (t k a b : ℝ) : 
  t > 0 → 
  k > 0 → 
  a < b → 
  f' t k a = 0 → 
  f' t k b = 0 → 
  ((a + b) / 2 = -2 ∨ (b + (-2)) / 2 = a) → 
  (a * (-2) = b^2 ∨ b * (-2) = a^2) → 
  t = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_t_l936_93676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_regular_triangular_prism_l936_93628

/-- A regular triangular prism where a lateral edge is equal to the side of the base -/
structure RegularTriangularPrism where
  /-- The side length of the base -/
  side_length : ℝ
  /-- The side length is positive -/
  side_length_pos : 0 < side_length

/-- The angle between a side of the base and a diagonal of a lateral face that does not intersect it -/
noncomputable def angle_between_side_and_diagonal (prism : RegularTriangularPrism) : ℝ :=
  Real.arccos (Real.sqrt 2 / 4)

/-- The theorem stating that the angle between a side of the base and a diagonal of a lateral face 
    that does not intersect it in a regular triangular prism (where a lateral edge is equal to 
    the side of the base) is equal to arccos(√2/4) -/
theorem angle_in_regular_triangular_prism (prism : RegularTriangularPrism) : 
  angle_between_side_and_diagonal prism = Real.arccos (Real.sqrt 2 / 4) := by
  -- Unfold the definition of angle_between_side_and_diagonal
  unfold angle_between_side_and_diagonal
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_regular_triangular_prism_l936_93628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l936_93655

noncomputable section

/-- The line y = -2x + 3 --/
def line (x : ℝ) : ℝ := -2 * x + 3

/-- The point we're finding the closest point to --/
def point : ℝ × ℝ := (3, -1)

/-- The proposed closest point on the line --/
def closest_point : ℝ × ℝ := (11/5, -7/5)

/-- Theorem stating that closest_point is on the line and is the closest point to point --/
theorem closest_point_is_closest :
  (line closest_point.fst = closest_point.snd) ∧
  ∀ p : ℝ × ℝ, line p.fst = p.snd →
    (closest_point.fst - point.fst)^2 + (closest_point.snd - point.snd)^2 ≤
    (p.fst - point.fst)^2 + (p.snd - point.snd)^2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l936_93655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_example_l936_93619

/-- Calculates the time (in seconds) for a train to cross an electric pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  train_length / train_speed_ms

theorem train_crossing_time_example :
  train_crossing_time 320 144 = 8 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof is completed numerically
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_example_l936_93619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_PQR_STU_l936_93659

/-- Triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Area of a triangle given two sides and the angle between them -/
noncomputable def area (t : Triangle) : ℝ := 
  1/2 * t.a * t.b

/-- Triangle PQR with sides 7, 24, and 25 -/
def PQR : Triangle := ⟨7, 24, 25⟩

/-- Triangle STU with sides 9, 40, and 41 -/
def STU : Triangle := ⟨9, 40, 41⟩

/-- The ratio of the areas of triangles PQR and STU is 7/15 -/
theorem area_ratio_PQR_STU : 
  (area PQR) / (area STU) = 7/15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_PQR_STU_l936_93659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l936_93650

/-- A parabola with equation x^2 = 4y -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4 * p.2}

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (0, 1)

/-- A line with slope k passing through the focus -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + 1}

/-- Distance from a point to a line -/
noncomputable def distanceToLine (p : ℝ × ℝ) (k : ℝ) : ℝ :=
  |k * p.1 - p.2 + 1| / Real.sqrt (k^2 + 1)

/-- Theorem stating the range of k -/
theorem parabola_line_intersection (k : ℝ) :
  (∃ (p1 p2 p3 p4 : ℝ × ℝ),
    p1 ∈ Parabola ∧ p2 ∈ Parabola ∧ p3 ∈ Parabola ∧ p4 ∈ Parabola ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    distanceToLine p1 k = 2 ∧ distanceToLine p2 k = 2 ∧
    distanceToLine p3 k = 2 ∧ distanceToLine p4 k = 2) →
  k < -Real.sqrt 3 ∨ k > Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l936_93650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l936_93658

-- Define the circles
def circle_M (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Define the centers of the circles
def center_M : ℝ × ℝ := (0, 2)
def center_N : ℝ × ℝ := (1, 1)

-- Define the radii of the circles
def radius_M : ℝ := 2
def radius_N : ℝ := 1

-- Define the distance between the centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 2

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  distance_between_centers > |radius_M - radius_N| ∧
  distance_between_centers < radius_M + radius_N :=
by
  sorry

#check circles_intersect

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l936_93658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l936_93600

/-- The function f(x) defined for x > 0 -/
noncomputable def f (x : ℝ) : ℝ := x^2 + 3*x + 6/x + 4/x^2 - 1

/-- The theorem stating the minimum value of f(x) for x > 0 -/
theorem min_value_of_f :
  ∀ x > 0, f x ≥ 3 - 6 * Real.sqrt 2 ∧
  ∃ x > 0, f x = 3 - 6 * Real.sqrt 2 :=
by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l936_93600
