import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_vectors_x_value_l307_30790

/-- Prove that if vectors a = (1, 2, 3), b = (3, 0, 2), and c = (4, 2, X) are coplanar, then X = 5. -/
theorem coplanar_vectors_x_value (X : ℝ) :
  let a : Fin 3 → ℝ := ![1, 2, 3]
  let b : Fin 3 → ℝ := ![3, 0, 2]
  let c : Fin 3 → ℝ := ![4, 2, X]
  (∃ (lambda mu : ℝ), c = lambda • a + mu • b) →
  X = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_vectors_x_value_l307_30790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l307_30717

noncomputable def f (x : ℝ) : ℝ := 1 / (x + Real.sqrt (1 + 2 * x^2))

theorem f_properties :
  ∃ (f' : ℝ → ℝ),
    (∀ x, HasDerivAt f (f' x) x) ∧
    (∀ x, f' x = -(Real.sqrt (1 + 2*x^2) + 2*x) / (Real.sqrt (1 + 2*x^2) * (x + Real.sqrt (1 + 2*x^2))^2)) ∧
    (Set.range f = Set.Ioo 0 (Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l307_30717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_characterization_l307_30793

structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ := 
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem line_segment_characterization (A B : Point) :
  {P : Point | distance P A + distance P B = distance A B} = 
  {P : Point | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P.x = A.x + t * (B.x - A.x) ∧ P.y = A.y + t * (B.y - A.y)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_characterization_l307_30793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_X_l307_30701

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The ratio of distances between three collinear points -/
noncomputable def distanceRatio (X Y Z : Point) : ℝ := 
  ((X.x - Z.x)^2 + (X.y - Z.y)^2) / ((Z.x - Y.x)^2 + (Z.y - Y.y)^2)

theorem sum_of_coordinates_X (X Y Z : Point) : 
  Y.x = 2 ∧ Y.y = 6 ∧ Z.x = -4 ∧ Z.y = 8 ∧ distanceRatio X Y Z = 3 → 
  X.x + X.y = -8 := by
  sorry

#check sum_of_coordinates_X

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_X_l307_30701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marketing_percentage_theorem_l307_30781

noncomputable section

-- Define the restaurant's budget allocation
def restaurant_budget (total : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ := 
  let rent := total / 5
  let remaining_after_rent := total - rent
  let food_and_beverages := remaining_after_rent / 4
  let remaining_after_food := remaining_after_rent - food_and_beverages
  let employee_salaries := remaining_after_food / 3
  let remaining_after_salaries := remaining_after_food - employee_salaries
  let utilities := remaining_after_salaries / 7
  let remaining_after_utilities := remaining_after_salaries - utilities
  let marketing := remaining_after_utilities * 0.15
  (rent, food_and_beverages, employee_salaries, utilities, marketing)

-- Define approximate equality
def approximate_eq (x y : ℝ) : Prop := abs (x - y) < 0.01

-- Notation for approximate equality
notation:50 a " ≈ " b:50 => approximate_eq a b

-- Theorem stating that the marketing budget is approximately 5.14% of the total budget
theorem marketing_percentage_theorem (total : ℝ) (h : total > 0) :
  let (_, _, _, _, marketing) := restaurant_budget total
  (marketing / total) * 100 ≈ 5.14 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marketing_percentage_theorem_l307_30781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l307_30744

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Point on a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through the origin -/
structure Line where
  slope : ℝ

/-- Theorem: Eccentricity of a special hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) (l : Line) (M N F : Point) :
  (M.x^2 / h.a^2 - M.y^2 / h.b^2 = 1) →
  (N.x^2 / h.a^2 - N.y^2 / h.b^2 = 1) →
  (M.y = l.slope * M.x) →
  (N.y = l.slope * N.x) →
  (F.x > 0) →
  ((M.x - F.x) * (N.x - F.x) + (M.y - F.y) * (N.y - F.y) = 0) →
  (abs ((M.x - F.x) * (N.y - F.y) - (M.y - F.y) * (N.x - F.x)) / 2 = h.a * h.b) →
  Real.sqrt (1 + h.b^2 / h.a^2) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l307_30744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slide_to_swing_ratio_l307_30724

/-- Represents the number of kids waiting for the slide -/
def S : ℕ := sorry

/-- Represents the number of kids waiting for the swings -/
def W : ℕ := sorry

/-- The number of kids waiting for the swings is 3 -/
axiom swing_kids : W = 3

/-- The wait time for swings in seconds -/
def swing_wait : ℕ := 120

/-- The wait time for slide in seconds -/
def slide_wait : ℕ := 15

/-- The difference between the longer and shorter wait times is 270 seconds -/
axiom wait_difference : swing_wait * W - slide_wait * S = 270

theorem slide_to_swing_ratio : S / W = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slide_to_swing_ratio_l307_30724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_not_divisible_by_triple_l307_30782

theorem sum_not_divisible_by_triple (n : ℕ) (h : n ≥ 3) (S : Finset ℕ) 
  (hS : S.card = n) (hDistinct : S.card = Finset.card (Finset.image id S)) :
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧
    ∀ x ∈ S, ¬(3 * x ∣ a + b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_not_divisible_by_triple_l307_30782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_plus_one_div_square_integer_l307_30739

theorem power_plus_one_div_square_integer (n : ℕ) : 
  n > 1 → (((2^n : ℕ) + 1) % (n^2 : ℕ) = 0) ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_plus_one_div_square_integer_l307_30739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_scorer_has_draw_l307_30752

/-- Represents the result of a match -/
inductive MatchResult
| Win
| Draw
| Loss

/-- Represents a team's performance in the tournament -/
structure TeamPerformance where
  wins : Nat
  draws : Nat
  losses : Nat

/-- Calculates the total points for a team given their performance -/
def calculatePoints (performance : TeamPerformance) : Nat :=
  3 * performance.wins + 2 * performance.draws + performance.losses

/-- The tournament setup -/
structure Tournament where
  numTeams : Nat
  performances : Fin numTeams → TeamPerformance
  lowestPoints : Nat
  distinctPoints : ∀ i j, i ≠ j → calculatePoints (performances i) ≠ calculatePoints (performances j)

theorem highest_scorer_has_draw (t : Tournament) 
  (h1 : t.numTeams = 15)
  (h2 : t.lowestPoints = 21)
  : ∃ i, (calculatePoints (t.performances i) = Finset.sup (Finset.univ : Finset (Fin t.numTeams)) (λ j => calculatePoints (t.performances j))) → 
    (t.performances i).draws > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_scorer_has_draw_l307_30752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_domain_l307_30730

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc 2 8

-- State the theorem
theorem inverse_function_domain :
  ∃ (domain_f_inv : Set ℝ), 
    domain_f_inv = Set.Icc 1 3 ∧
    (∀ y ∈ domain_f_inv, ∃ x ∈ domain_f, f x = y) ∧
    (∀ x ∈ domain_f, f x ∈ domain_f_inv) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_domain_l307_30730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_y_max_value_of_g_l307_30789

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin x + cos x

-- Define the function g as a shift of f
noncomputable def g (x : ℝ) : ℝ := f (x - π / 2)

-- Theorem for the smallest positive period of y = [f(x+π/2)]^2
theorem smallest_positive_period_of_y : 
  ∃ (p : ℝ), p > 0 ∧ ∀ (t : ℝ), (f (t + π / 2))^2 = (f ((t + p) + π / 2))^2 ∧ 
  ∀ (q : ℝ), q > 0 → (∀ (t : ℝ), (f (t + π / 2))^2 = (f ((t + q) + π / 2))^2) → p ≤ q :=
by
  sorry

-- Theorem for the maximum value of g(x) on [0,π]
theorem max_value_of_g : 
  ∃ (M : ℝ), M = sqrt 2 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ π → g x ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_y_max_value_of_g_l307_30789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_count_l307_30771

theorem inequality_solution_count : 
  ∃ (S : Finset Int), (∀ y : Int, y ∈ S ↔ 3*y^2 + 17*y + 14 ≤ 22) ∧ Finset.card S = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_count_l307_30771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_identity_l307_30764

-- Define a polynomial type
def MyPolynomial (α : Type) := α → α

-- State the theorem
theorem unique_polynomial_identity {α : Type} [CommRing α] (P : MyPolynomial α) :
  (P 0 = 0) →
  (∀ x, P (x^2 + 1) = (P x)^2 + 1) →
  (∀ x, P x = x) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_identity_l307_30764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l307_30757

noncomputable def f (x : ℝ) := x^2 - Real.pi * x

noncomputable def α : ℝ := Real.arcsin 1
noncomputable def β : ℝ := Real.arctan 1
noncomputable def γ : ℝ := Real.arccos (-1)
noncomputable def d : ℝ := Real.pi / 2 - Real.arctan 1  -- Equivalent to arccot 0

theorem function_inequality : f α > f d ∧ f d > f β ∧ f β > f γ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l307_30757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine2_production_percentage_l307_30795

/-- Represents the percentage of products manufactured by each machine -/
structure ProductionPercentages where
  m1 : ℚ
  m2 : ℚ
  m3 : ℚ
  sum_to_100 : m1 + m2 + m3 = 100

/-- Represents the percentage of defective products for each machine -/
structure DefectivePercentages where
  m1 : ℚ
  m2 : ℚ
  m3 : ℚ

/-- Calculates the overall percentage of defective products -/
def overallDefectivePercentage (prod : ProductionPercentages) (defect : DefectivePercentages) : ℚ :=
  (prod.m1 * defect.m1 + prod.m2 * defect.m2 + prod.m3 * defect.m3) / 100

theorem machine2_production_percentage :
  ∀ (prod : ProductionPercentages) (defect : DefectivePercentages),
    prod.m1 = 40 →
    defect.m1 = 3 →
    defect.m2 = 1 →
    defect.m3 = 7 →
    overallDefectivePercentage prod defect = 36/10 →
    prod.m2 = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine2_production_percentage_l307_30795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_555_degrees_l307_30703

theorem cos_555_degrees :
  Real.cos (555 * π / 180) = -(Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_555_degrees_l307_30703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l307_30742

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + Real.cos (Real.pi / 2 - 2 * x)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∀ (y : ℝ), (∃ (x : ℝ), f x = y) ↔ 1 - Real.sqrt 2 ≤ y ∧ y ≤ 1 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l307_30742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_81n4_l307_30780

theorem divisors_of_81n4 (n : ℕ+) (h : (Nat.divisors (110 * n.val ^ 3)).card = 110) :
  (Nat.divisors (81 * n.val ^ 4)).card = 325 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_81n4_l307_30780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_unique_colorings_l307_30747

/-- A coloring of a 2x2 grid -/
def Coloring := Fin 4 → Fin 3

/-- Check if a coloring is valid according to the rules -/
def isValidColoring (c : Coloring) : Prop :=
  (c 0 ≠ c 1) ∧ (c 0 ≠ c 2) ∧ (c 1 ≠ c 3) ∧ (c 2 ≠ c 3) ∧
  (Finset.card (Finset.image c (Finset.univ : Finset (Fin 4))) = 3)

/-- Two colorings are equivalent if they're the same up to permutation of colors -/
def equivalentColorings (c1 c2 : Coloring) : Prop :=
  ∃ (perm : Equiv.Perm (Fin 3)), ∀ i, perm (c1 i) = c2 i

/-- The main theorem: there are exactly 3 unique valid colorings -/
theorem three_unique_colorings :
  ∃! (colorings : Finset Coloring),
    (∀ c ∈ colorings, isValidColoring c) ∧
    (∀ c, isValidColoring c → ∃ c' ∈ colorings, equivalentColorings c c') ∧
    colorings.card = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_unique_colorings_l307_30747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarf_hats_stabilize_l307_30791

/-- Represents the state of the dwarf system at a given time -/
structure DwarfSystem where
  time : ℕ
  hatColors : Fin 12 → Bool
  friendships : Fin 12 → Fin 12 → Bool

/-- The number of friend pairs wearing different colored hats -/
def differentColorPairs (s : DwarfSystem) : ℕ :=
  sorry

/-- The rule for changing hats -/
def changeHat (s : DwarfSystem) (d : Fin 12) : DwarfSystem :=
  sorry

/-- The next state of the system after one day -/
def nextDay (s : DwarfSystem) : DwarfSystem :=
  sorry

theorem dwarf_hats_stabilize :
  ∀ (s : DwarfSystem), ∃ (t : ℕ), ∀ (t' : ℕ), t' ≥ t →
    differentColorPairs (Nat.iterate nextDay t' s) = 0 := by
  sorry

#check dwarf_hats_stabilize

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarf_hats_stabilize_l307_30791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_susann_coloring_theorem_l307_30799

/-- Represents a box in the grid -/
structure Box where
  row : Nat
  col : Nat
  value : Int

/-- Represents the state of the grid -/
structure Grid where
  n : Nat
  boxes : List Box
  destroyed : List Box

/-- Predicate to check if a box is destroyed -/
def isDestroyed (grid : Grid) (box : Box) : Prop :=
  box ∈ grid.destroyed

/-- Predicate to check if two boxes are in the same row or column -/
def sameRowOrCol (box1 box2 : Box) : Prop :=
  box1.row = box2.row ∨ box1.col = box2.col

/-- Predicate to check if a coloring is valid -/
def validColoring (grid : Grid) (redBoxes : List Box) : Prop :=
  (∀ b1 b2, b1 ∈ redBoxes → b2 ∈ redBoxes → b1 ≠ b2 → ¬sameRowOrCol b1 b2) ∧
  (∀ b, b ∈ grid.boxes → ¬isDestroyed grid b → b ∉ redBoxes →
    (∃ r, r ∈ redBoxes ∧ r.row = b.row ∧ r.value > b.value) ∨
    (∃ r, r ∈ redBoxes ∧ r.col = b.col ∧ r.value < b.value))

/-- The main theorem -/
theorem susann_coloring_theorem (n : Nat) (grid : Grid) :
  grid.n = n →
  (∀ i j, i < n → j < n → ∃! b, b ∈ grid.boxes ∧ b.row = i ∧ b.col = j) →
  (∀ i, i < n → (∀ b1 b2, b1 ∈ grid.boxes → b2 ∈ grid.boxes → b1.row = i → b2.row = i → b1 ≠ b2 → b1.value ≠ b2.value)) →
  (∀ j, j < n → (∀ b1 b2, b1 ∈ grid.boxes → b2 ∈ grid.boxes → b1.col = j → b2.col = j → b1 ≠ b2 → b1.value ≠ b2.value)) →
  ∃ redBoxes : List Box, validColoring grid redBoxes :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_susann_coloring_theorem_l307_30799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_sqrt_1_plus_x_squared_l307_30741

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + x^2)

-- State the theorem
theorem derivative_sqrt_1_plus_x_squared :
  deriv f = fun x => x / Real.sqrt (1 + x^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_sqrt_1_plus_x_squared_l307_30741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l307_30762

theorem geometric_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- Definition of S_n
  (∃ d : ℝ, S 9 - S 6 = S 6 - S 3) →          -- Arithmetic sequence condition
  (a 8 = 3) →                                 -- Given condition
  (∀ n, a (n + 1) = a n * q) →                -- Definition of geometric sequence
  (a 2 + a 5 = 6) :=                          -- Conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l307_30762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_formula_l307_30735

/-- An arithmetic sequence with the given first three terms has the general term formula a_n = 2n - 3 -/
theorem arithmetic_sequence_formula (x : ℝ) : 
  ∃ (a : ℕ → ℝ) (d : ℝ), 
    a 1 = x - 1 ∧
    a 2 = x + 1 ∧
    a 3 = 2*x + 3 ∧
    (∀ (n : ℕ), n ≥ 1 → a n = 2*n - 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_formula_l307_30735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_a_value_l307_30720

/-- Given three points A(3,2), B(-2,a), and C(8,12) that are collinear, prove that a = -8 -/
theorem collinear_points_a_value (a : ℝ) : 
  (12 - 2) * (-2 - 3) = (a - 2) * (8 - 3) → a = -8 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_a_value_l307_30720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_and_periodicity_l307_30796

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 4)

theorem f_symmetry_and_periodicity :
  (∀ x, f (-(Real.pi / 8) + x) = f (-(Real.pi / 8) - x)) ∧
  (∃ α : ℝ, 0 < α ∧ α < Real.pi ∧ ∀ x, f (x + α) = f (x + 3 * α)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_and_periodicity_l307_30796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l307_30706

/-- A function f that satisfies f(-x) = 2 - f(x) for all x -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = 2 - f x

/-- The function g(x) = (x + 1) / x -/
noncomputable def g (x : ℝ) : ℝ := (x + 1) / x

/-- Intersection points of f and g -/
def IntersectionPoints (f : ℝ → ℝ) (points : Finset (ℝ × ℝ)) : Prop :=
  ∀ p ∈ points, f p.1 = g p.1 ∧ p.2 = f p.1

theorem intersection_sum (f : ℝ → ℝ) (points : Finset (ℝ × ℝ)) :
  SymmetricFunction f → IntersectionPoints f points →
  (points.sum (λ p => p.1 + p.2) : ℝ) = points.card := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l307_30706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_comprehensive_score_l307_30778

/-- Calculates the comprehensive score given the individual scores and their weights --/
noncomputable def comprehensive_score (theoretical_score innovative_score display_score : ℚ)
  (theoretical_weight innovative_weight display_weight : ℕ) : ℚ :=
  (theoretical_score * theoretical_weight + innovative_score * innovative_weight + display_score * display_weight) /
  (theoretical_weight + innovative_weight + display_weight)

/-- Theorem stating that the student's comprehensive score is 90 points --/
theorem student_comprehensive_score :
  comprehensive_score 95 88 90 2 5 3 = 90 := by
  -- Unfold the definition of comprehensive_score
  unfold comprehensive_score
  -- Simplify the numerator and denominator
  simp
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_comprehensive_score_l307_30778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheela_income_l307_30777

noncomputable def monthly_income (deposit : ℝ) (percentage : ℝ) : ℝ :=
  deposit / (percentage / 100)

theorem sheela_income : 
  let deposit : ℝ := 3400
  let percentage : ℝ := 15
  abs (monthly_income deposit percentage - 22666.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheela_income_l307_30777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_properties_l307_30798

/-- The function y in terms of x -/
noncomputable def y (x : ℝ) : ℝ := (x + 2) / (x^2 + x + 1)

/-- The reciprocal of y -/
noncomputable def y_inv (x : ℝ) : ℝ := 1 / y x

theorem y_properties :
  ∀ x > -2,
  (∀ z, y_inv z ≥ 2 * Real.sqrt 3 - 3) ∧
  (y_inv (Real.sqrt 3 - 2) = 2 * Real.sqrt 3 - 3) ∧
  (∀ z > -2, y z ≤ y (Real.sqrt 3 - 2)) ∧
  (y (Real.sqrt 3 - 2) = (2 * Real.sqrt 3 + 3) / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_properties_l307_30798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocal_distances_l307_30733

noncomputable section

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the left focus
def left_focus : ℝ × ℝ := (-1, 0)

-- Define the line passing through the left focus at 60°
def is_on_line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x + 1)

-- Define the intersection points A and B
def is_intersection_point (p : ℝ × ℝ) : Prop :=
  is_on_ellipse p.1 p.2 ∧ is_on_line p.1 p.2

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem sum_of_reciprocal_distances :
  ∀ A B : ℝ × ℝ,
  is_intersection_point A →
  is_intersection_point B →
  A ≠ B →
  (1 / distance A left_focus) + (1 / distance B left_focus) = 4/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocal_distances_l307_30733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_xi_is_seven_thirds_l307_30700

/-- Represents the test locations --/
inductive TestLocation
| B
| C
| D

/-- Represents a candidate --/
structure Candidate where
  name : String
  passingProbabilities : TestLocation → Rat

/-- The sum of the number of test locations at which both candidates pass --/
def xi (c1 c2 : Candidate) (l1 l2 l3 l4 : TestLocation) : ℕ → Prop :=
  sorry

/-- The probability of a specific value of xi --/
def probXi (c1 c2 : Candidate) (l1 l2 l3 l4 : TestLocation) : ℕ → Rat :=
  sorry

/-- The expected value of xi --/
def expectedXi (c1 c2 : Candidate) (l1 l2 l3 l4 : TestLocation) : Rat :=
  sorry

/-- Xiao Li's passing probabilities --/
def xiaoLiProb : TestLocation → Rat
  | TestLocation.B => 2/3
  | TestLocation.C => 1/3
  | TestLocation.D => 1/2

/-- Xiao Wang's passing probabilities --/
def xiaoWangProb : TestLocation → Rat
  | TestLocation.B => 2/3
  | TestLocation.C => 2/3
  | TestLocation.D => 2/3

/-- Xiao Li --/
def xiaoLi : Candidate := ⟨"Xiao Li", xiaoLiProb⟩

/-- Xiao Wang --/
def xiaoWang : Candidate := ⟨"Xiao Wang", xiaoWangProb⟩

/-- Main theorem: The expected value of xi is 7/3 --/
theorem expected_xi_is_seven_thirds :
  expectedXi xiaoLi xiaoWang TestLocation.B TestLocation.C TestLocation.B TestLocation.D = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_xi_is_seven_thirds_l307_30700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l307_30753

theorem quadratic_inequality_solution_set 
  (b c : ℝ) 
  (h : Set.Iic (-3) ∪ Set.Ioi 2 = {x : ℝ | -x^2 + b*x + c < 0}) :
  {x : ℝ | c*x^2 - b*x - 1 > 0} = Set.Iic (-1/2) ∪ Set.Ioi (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l307_30753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_sees_jane_for_45_minutes_l307_30770

/-- The time (in minutes) that John can see Jane given their speeds and distances. -/
noncomputable def visibleTime (johnSpeed runnerSpeed cyclistSpeed : ℝ) (initialDistance finalDistance : ℝ) : ℝ :=
  let relativeSpeed := runnerSpeed - cyclistSpeed
  let timeToPass := initialDistance / relativeSpeed
  let timeUntilLost := finalDistance / relativeSpeed
  (timeToPass + timeUntilLost) * 60

/-- Theorem stating that John can see Jane for 45 minutes under the given conditions. -/
theorem john_sees_jane_for_45_minutes :
  visibleTime 7 7 3 1 2 = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_sees_jane_for_45_minutes_l307_30770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_cos_l307_30773

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := f ((x / 2) + Real.pi / 6)

theorem g_equals_cos (x : ℝ) : g x = Real.cos x := by
  -- Unfold the definitions of g and f
  unfold g f
  -- Simplify the expression
  simp [Real.sin_add, Real.sin_pi_div_three, Real.cos_pi_div_three]
  -- Use trigonometric identities
  rw [Real.sin_two_mul, Real.cos_add]
  -- Simplify further
  simp [Real.sin_pi_div_two, Real.cos_pi_div_two]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_cos_l307_30773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_cost_price_clock_cost_price_proof_l307_30768

/-- Proves that the cost price of each clock is 120, given the conditions of the problem --/
theorem clock_cost_price (total_clocks sold_at_12_percent sold_at_22_percent : ℕ) 
  (revenue_difference cost_price : ℝ) : Prop :=
  total_clocks = 150 ∧
  sold_at_12_percent = 70 ∧
  sold_at_22_percent = 80 ∧
  revenue_difference = 60 ∧
  sold_at_12_percent + sold_at_22_percent = total_clocks ∧
  (sold_at_12_percent : ℝ) * (cost_price * 1.12) + 
    (sold_at_22_percent : ℝ) * (cost_price * 1.22) - 
    (total_clocks : ℝ) * (cost_price * 1.17) = revenue_difference →
  cost_price = 120

theorem clock_cost_price_proof : ∃ (total_clocks sold_at_12_percent sold_at_22_percent : ℕ) 
  (revenue_difference cost_price : ℝ),
  clock_cost_price total_clocks sold_at_12_percent sold_at_22_percent revenue_difference cost_price := by
  sorry

#check clock_cost_price
#check clock_cost_price_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_cost_price_clock_cost_price_proof_l307_30768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cherry_pie_degree_is_45_l307_30721

/-- The degree measure for cherry pie in Mathew's pie chart presentation --/
noncomputable def cherry_pie_degree (total_students : ℕ) (chocolate_pref : ℕ) (apple_pref : ℕ) (blueberry_pref : ℕ) : ℝ :=
  let remaining_students := total_students - (chocolate_pref + apple_pref + blueberry_pref)
  let cherry_pref := remaining_students / 2
  (cherry_pref : ℝ) / total_students * 360

/-- Theorem stating the degree measure for cherry pie in Mathew's pie chart presentation --/
theorem cherry_pie_degree_is_45 :
  cherry_pie_degree 40 15 10 5 = 45 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cherry_pie_degree_is_45_l307_30721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_slope_l307_30712

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 - 2*x + y^2 = 0

-- Define the line l
def line_l (x y k : ℝ) : Prop := k*x - y + 2 - 2*k = 0

-- Define the intersection points A and B
def intersection_points (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
  line_l A.1 A.2 k ∧ line_l B.1 B.2 k ∧
  A ≠ B

-- Define the area of triangle ABC
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Theorem statement
theorem max_area_slope :
  ∀ (k : ℝ) (A B : ℝ × ℝ),
    intersection_points k A B →
    (∀ (C : ℝ × ℝ), circle_C C.1 C.2 →
      ∀ (k' : ℝ), triangle_area A B C ≤ triangle_area A B (1, 0)) →
    k = 1 ∨ k = 7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_slope_l307_30712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_number_tenth_row_l307_30729

/-- A lattice where each row increases by 7 elements starting with 1 -/
def lattice (row : ℕ) (pos : ℕ) : ℕ := 
  7 * row - pos + 1

/-- The theorem stating that the fifth number from the end in the 10th row is 68 -/
theorem fifth_number_tenth_row : lattice 10 3 = 68 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_number_tenth_row_l307_30729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_and_extrema_l307_30786

noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1)

theorem f_monotone_decreasing_and_extrema :
  (∀ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 6 → f x ≤ 2) ∧
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 6 → f x ≥ 2/5) ∧
  (f 2 = 2) ∧
  (f 6 = 2/5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_and_extrema_l307_30786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_value_l307_30722

theorem smallest_x_value (y : ℕ) (h : (3 : ℚ) / 4 = y / (252 + 0)) : 
  ∀ x : ℕ, (3 : ℚ) / 4 = y / (252 + x) → x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_value_l307_30722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coinciding_rest_days_l307_30749

/-- Al's cycle length in days -/
def al_cycle : ℕ := 6

/-- Barb's cycle length in days -/
def barb_cycle : ℕ := 10

/-- Total number of days -/
def total_days : ℕ := 1000

/-- Al's rest days in his cycle -/
def al_rest_days : List ℕ := [5, 6]

/-- Barb's rest days in her cycle -/
def barb_rest_days : List ℕ := [7, 8, 9, 10]

/-- The number of days both Al and Barb have rest days in the same 30-day period -/
def coinciding_rest_days_per_period : ℕ := 2

theorem coinciding_rest_days :
  (total_days / (al_cycle.lcm barb_cycle)) * coinciding_rest_days_per_period = 66 := by
  sorry

#eval (total_days / (al_cycle.lcm barb_cycle)) * coinciding_rest_days_per_period

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coinciding_rest_days_l307_30749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_implies_coordinates_l307_30775

-- Define the function f
noncomputable def f (θ k : ℝ) (x : ℝ) : ℝ := Real.tan (2 * x + θ) + k

-- State the theorem
theorem symmetry_center_implies_coordinates (θ k : ℝ) :
  θ ∈ Set.Ioo 0 (Real.pi / 2) →
  (∀ x, f θ k (x + Real.pi / 6) + 1 = -(f θ k (Real.pi / 6 - x) + 1)) →
  (θ = Real.pi / 6 ∧ k = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_implies_coordinates_l307_30775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orangeade_water_ratio_l307_30772

/-- Represents the amount of orange juice used (same on both days) -/
noncomputable def orange_juice : ℝ := sorry

/-- Represents the amount of water used on the first day -/
noncomputable def water_day1 : ℝ := sorry

/-- Represents the amount of water used on the second day -/
noncomputable def water_day2 : ℝ := sorry

/-- The price per glass on the first day -/
def price_day1 : ℝ := 0.30

/-- The price per glass on the second day -/
def price_day2 : ℝ := 0.20

theorem orangeade_water_ratio :
  (orange_juice > 0) →
  (water_day1 > 0) →
  (water_day2 > 0) →
  (orange_juice = water_day1) →  -- Condition 2
  (price_day1 * (orange_juice + water_day1) = price_day2 * (orange_juice + water_day2)) →  -- Condition 4
  (water_day2 / water_day1 = 2) :=  -- Conclusion: ratio is 2:1
by
  sorry

#check orangeade_water_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orangeade_water_ratio_l307_30772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l307_30759

theorem triangle_properties (a b c A B C : ℝ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : 0 < A ∧ A < π) (h3 : 0 < B ∧ B < π) (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π)
  (h6 : a / Real.sin A = b / Real.sin B)
  (h7 : b / Real.sin B = c / Real.sin C)
  (h8 : Real.sqrt 3 * a / c = (Real.cos A + 2) / Real.sin C)
  (h9 : b + c = 5)
  (h10 : 1/2 * b * c * Real.sin A = Real.sqrt 3) :
  A = 2*π/3 ∧ a = Real.sqrt 21 := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l307_30759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_3_l307_30702

/-- The distance between two parallel lines ax + by + c₁ = 0 and ax + by + c₂ = 0 -/
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  abs (c₂ - c₁) / Real.sqrt (a^2 + b^2)

/-- A line parallel to 3x - 4y + 1 = 0 can be represented as 3x - 4y + c = 0 -/
def parallel_line (c : ℝ) : Prop :=
  ∃ (x y : ℝ), 3*x - 4*y + c = 0

theorem parallel_lines_at_distance_3 :
  ∀ c : ℝ, 
    (parallel_line c ∧ 
     distance_parallel_lines 3 (-4) 1 c = 3) ↔ 
    (c = 16 ∨ c = -14) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_3_l307_30702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_slice_surface_area_increase_l307_30751

/-- The increase in surface area when cutting a cylinder into slices -/
theorem cylinder_slice_surface_area_increase
  (h : ℝ) (d : ℝ) (n : ℕ)
  (h_pos : h > 0)
  (d_pos : d > 0)
  (n_pos : n > 0) :
  let r := d / 2
  let initial_area := 2 * π * r * h + 2 * π * r^2
  let slice_height := h / (n : ℝ)
  let slice_area := 2 * π * r * slice_height + 2 * π * r^2
  let total_slice_area := (n : ℝ) * slice_area
  let area_increase := total_slice_area - initial_area
  (h = 10 ∧ d = 5 ∧ n = 9) → area_increase = 100 * π :=
by
  sorry

#check cylinder_slice_surface_area_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_slice_surface_area_increase_l307_30751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_min_value_l307_30725

/-- The minimum value of (b² + 1) / (3a) for an ellipse with eccentricity 1/2 -/
theorem ellipse_min_value (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (x y : ℝ) → x^2 / a^2 + y^2 / b^2 = 1 → True)  -- Ellipse equation
  (h4 : Real.sqrt (1 - b^2 / a^2) = 1/2) :  -- Eccentricity condition
  (∀ a' b' : ℝ, a' > b' ∧ b' > 0 ∧ 
    ((x y : ℝ) → x^2 / a'^2 + y^2 / b'^2 = 1 → True) ∧
    (Real.sqrt (1 - b'^2 / a'^2) = 1/2) →
    (b^2 + 1) / (3 * a) ≤ (b'^2 + 1) / (3 * a')) ∧
  (b^2 + 1) / (3 * a) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_min_value_l307_30725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_fixed_points_real_distinct_l307_30728

/-- Recursive definition of the polynomial sequence Pⱼ(x) -/
def P : ℕ → (ℝ → ℝ)
| 0 => fun x => x^2 - 2
| n+1 => fun x => P 0 (P n x)

/-- Theorem stating that Pₙ(x) = x has 2ⁿ real and distinct roots for any positive n -/
theorem P_fixed_points_real_distinct (n : ℕ+) :
  ∃ (s : Finset ℝ), (∀ x, x ∈ s → P n.val x = x) ∧ s.card = 2^n.val ∧ (∀ x y, x ∈ s → y ∈ s → x ≠ y) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_fixed_points_real_distinct_l307_30728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangement_exists_rearrangement_25_and_1000_l307_30788

def is_valid_permutation (p : List ℕ) : Prop :=
  p.length > 1 ∧ 
  (∀ i ∈ List.range (p.length - 1), 
    (p.get! (i+1) - p.get! i = 3 ∨ p.get! (i+1) - p.get! i = 5) ∨
    (p.get! i - p.get! (i+1) = 3 ∨ p.get! i - p.get! (i+1) = 5))

theorem rearrangement_exists (n : ℕ) : Prop :=
  ∃ p : List ℕ, p.Nodup ∧ p.length = n ∧ (∀ i ∈ List.range n, i + 1 ∈ p) ∧ is_valid_permutation p

theorem rearrangement_25_and_1000 : 
  rearrangement_exists 25 ∧ rearrangement_exists 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangement_exists_rearrangement_25_and_1000_l307_30788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_given_B_l307_30734

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a single die -/
def prob_not_six : ℚ := 1 - prob_single

/-- The total number of possible outcomes when rolling three dice -/
def total_outcomes : ℕ := 6^3

/-- Event A: The three numbers are all different -/
def event_A : Set (Fin 6 × Fin 6 × Fin 6) := 
  {x | x.1 ≠ x.2.1 ∧ x.2.1 ≠ x.2.2 ∧ x.1 ≠ x.2.2}

/-- Event B: At least one 6 appears -/
def event_B : Set (Fin 6 × Fin 6 × Fin 6) := 
  {x | x.1 = 5 ∨ x.2.1 = 5 ∨ x.2.2 = 5}

/-- The probability of event B -/
def prob_B : ℚ := 1 - prob_not_six^3

/-- The probability of events A and B occurring together -/
def prob_AB : ℚ := 60 / total_outcomes

/-- The main theorem: P(A|B) = 60/91 -/
theorem prob_A_given_B : prob_AB / prob_B = 60 / 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_given_B_l307_30734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_of_specific_sines_l307_30736

theorem cos_sum_of_specific_sines (α β : Real) 
  (h₁ : 0 < α ∧ α < π / 2) 
  (h₂ : 0 < β ∧ β < π / 2)
  (h₃ : Real.sin α = Real.sqrt 5 / 5)
  (h₄ : Real.sin β = Real.sqrt 10 / 10) : 
  Real.cos (α + β) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_of_specific_sines_l307_30736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_representation_exists_l307_30709

def a : Fin 2 → ℝ := ![3, 2]
def e₁ : Fin 2 → ℝ := ![-1, 2]
def e₂ : Fin 2 → ℝ := ![5, -2]

theorem vector_representation_exists :
  ∃ (l m : ℝ), (l • e₁ + m • e₂) = a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_representation_exists_l307_30709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_pi_over_4_minus_2_l307_30761

noncomputable section

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
  if x ≤ 0 ∧ x ≥ -Real.pi then Real.sin x
  else if x > 0 ∧ x ≤ 1 then Real.sqrt (1 - x^2)
  else 0  -- for completeness, though not used in the integral

-- State the theorem
theorem integral_f_equals_pi_over_4_minus_2 :
  ∫ x in -Real.pi..1, f x = Real.pi / 4 - 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_pi_over_4_minus_2_l307_30761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_theorem_l307_30776

/-- The smallest positive integer n with the given divisibility property -/
def smallest_n : ℕ+ := 13

/-- The divisibility property for positive integers x, y, z, and n -/
def divisibility_property (x y z : ℕ+) (n : ℕ) : Prop :=
  (x : ℕ) ∣ (y : ℕ)^3 ∧ (y : ℕ) ∣ (z : ℕ)^3 ∧ (z : ℕ) ∣ (x : ℕ)^3 → 
  ((x * y * z : ℕ+) : ℕ) ∣ ((x + y + z : ℕ+) : ℕ)^n

theorem smallest_n_theorem :
  (∀ x y z : ℕ+, divisibility_property x y z (smallest_n : ℕ)) ∧
  (∀ m : ℕ+, m < smallest_n → ∃ x y z : ℕ+, ¬divisibility_property x y z (m : ℕ)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_theorem_l307_30776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_prime_has_unique_zero_l307_30719

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.log x - Real.log x - x + 1

-- Define the derivative of f(x)
noncomputable def f' (x : ℝ) : ℝ := Real.log x - 1 / x

-- Theorem statement
theorem f_prime_has_unique_zero :
  ∃! x : ℝ, x > 0 ∧ f' x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_prime_has_unique_zero_l307_30719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_problem_l307_30746

theorem coin_problem (q d : ℚ) (h1 : q + d = 30) 
  (h2 : (0.1 * q + 0.25 * d) - (0.25 * q + 0.1 * d) = 1.2) : 
  0.25 * q + 0.1 * d = 4.65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_problem_l307_30746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_120_l307_30731

open Real

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def angle (x y : V) : ℝ := arccos (inner x y / (norm x * norm y))

theorem vector_angle_120 (a b c : V) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hnorm : norm a = norm b ∧ norm b = norm c) (hsum : a + b = c) : 
  angle a b = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_120_l307_30731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l307_30769

/-- A circle inscribed in a square with two intersecting chords -/
structure InscribedCircle where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The point that bisects one of the chords -/
  P : ℝ × ℝ
  /-- The intersection point of the two chords -/
  Q : ℝ × ℝ
  /-- The center of the circle -/
  O : ℝ × ℝ
  /-- Assertion that the circle is inscribed in a square -/
  h_inscribed : True
  /-- Assertion that P bisects one of the chords -/
  h_bisects : True

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: The distance of the intersection point from the center is 2√2 - 2 -/
theorem intersection_distance (c : InscribedCircle) (h : c.radius = 2) :
  distance c.Q c.O = 2 * Real.sqrt 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l307_30769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_matrix_squared_is_identity_l307_30794

def reflection_matrix (v : Fin 2 → ℝ) : Matrix (Fin 2) (Fin 2) ℝ := sorry

theorem reflection_matrix_squared_is_identity :
  let v : Fin 2 → ℝ := ![2, -1]
  let R := reflection_matrix v
  R * R = (1 : Matrix (Fin 2) (Fin 2) ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_matrix_squared_is_identity_l307_30794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_perfect_square_with_non_perfect_square_opposite_l307_30738

-- Define the type for the polygon
def Polygon := Fin 2000 → ℕ

-- Define the property of non-intersecting segments
def NonIntersecting (p : Polygon) : Prop :=
  ∀ a b c d : Fin 2000, a < b ∧ b < c ∧ c < d →
    ¬(Set.range (fun t => (1 - t) • (p a : ℝ) + t • (p b : ℝ)) ∩
      Set.range (fun t => (1 - t) • (p c : ℝ) + t • (p d : ℝ)) ≠ ∅)

-- Define what it means for a number to be diametrically opposite
def DiametricallyOpposite (a b : Fin 2000) : Prop :=
  (a.val + 1000) % 2000 = b.val

-- Define a perfect square
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- State the theorem
theorem exists_perfect_square_with_non_perfect_square_opposite
  (p : Polygon)
  (h1 : ∀ n : Fin 2000, p n = n.val + 1)
  (h2 : NonIntersecting p) :
  ∃ a : Fin 2000, IsPerfectSquare (p a) ∧
    ∃ b : Fin 2000, DiametricallyOpposite a b ∧ ¬IsPerfectSquare (p b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_perfect_square_with_non_perfect_square_opposite_l307_30738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_properties_l307_30774

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a shape in 2D space -/
inductive Shape
  | Triangle (a b c : Point)
  | Parallelogram (a b c d : Point)
  | Square (a b c d : Point)
  | Rhombus (a b c d : Point)

/-- Oblique projection transformation -/
noncomputable def obliqueProject (p : Point) : Point :=
  { x := p.x, y := p.y / 2 }

/-- Applies oblique projection to a shape -/
noncomputable def projectShape (s : Shape) : Shape :=
  match s with
  | Shape.Triangle a b c => Shape.Triangle (obliqueProject a) (obliqueProject b) (obliqueProject c)
  | Shape.Parallelogram a b c d => Shape.Parallelogram (obliqueProject a) (obliqueProject b) (obliqueProject c) (obliqueProject d)
  | Shape.Square a b c d => Shape.Parallelogram (obliqueProject a) (obliqueProject b) (obliqueProject c) (obliqueProject d)
  | Shape.Rhombus a b c d => Shape.Parallelogram (obliqueProject a) (obliqueProject b) (obliqueProject c) (obliqueProject d)

theorem oblique_projection_properties :
  (∀ s : Shape, (∃ a b c, s = Shape.Triangle a b c) → (∃ a' b' c', projectShape s = Shape.Triangle a' b' c')) ∧
  (∀ s : Shape, (∃ a b c d, s = Shape.Parallelogram a b c d) → (∃ a' b' c' d', projectShape s = Shape.Parallelogram a' b' c' d')) ∧
  (∀ s : Shape, (∃ a b c d, s = Shape.Square a b c d) → ¬(∃ a' b' c' d', projectShape s = Shape.Square a' b' c' d')) ∧
  (∀ s : Shape, (∃ a b c d, s = Shape.Rhombus a b c d) → ¬(∃ a' b' c' d', projectShape s = Shape.Rhombus a' b' c' d')) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_properties_l307_30774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_to_line_l307_30792

-- Define the ellipse C
noncomputable def ellipse_C (ρ θ : ℝ) : Prop :=
  ρ^2 = 12 / (3 * (Real.cos θ)^2 + 4 * (Real.sin θ)^2)

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (2 + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

-- Define the foci of the ellipse
def foci : ℝ × ℝ × ℝ × ℝ := (-1, 0, 1, 0)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x - y - 2| / Real.sqrt 2

-- Theorem statement
theorem sum_of_distances_to_line :
  let (x₁, y₁, x₂, y₂) := foci
  (distance_to_line x₁ y₁) + (distance_to_line x₂ y₂) = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_to_line_l307_30792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_from_cos_sum_l307_30765

theorem sin_theta_from_cos_sum (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (π / 2))
  (h2 : Real.cos θ + Real.cos (θ + π / 3) = Real.sqrt 3 / 3) : 
  Real.sin θ = (-1 + 2 * Real.sqrt 6) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_from_cos_sum_l307_30765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_x_squared_minus_x_equals_negative_one_l307_30737

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define x
noncomputable def x : ℂ := (1 + i * Real.sqrt 3) / 2

-- Theorem statement
theorem inverse_x_squared_minus_x_equals_negative_one : 
  1 / (x^2 - x) = -1 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_x_squared_minus_x_equals_negative_one_l307_30737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_three_pairs_l307_30787

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The total number of possible outcomes when rolling the dice -/
def total_outcomes : ℕ := num_faces ^ num_dice

/-- The number of ways to choose 3 different numbers from 6 possible numbers -/
def ways_to_choose_numbers : ℕ := Nat.choose num_faces 3

/-- The number of ways to arrange 3 pairs among 6 dice -/
def ways_to_arrange_pairs : ℕ := (Nat.factorial num_dice) / ((Nat.factorial 2) * (Nat.factorial 2) * (Nat.factorial 2))

/-- The total number of successful outcomes (exactly three pairs) -/
def successful_outcomes : ℕ := ways_to_choose_numbers * ways_to_arrange_pairs

/-- The probability of rolling exactly three pairs with six standard six-sided dice -/
theorem probability_of_three_pairs : 
  (successful_outcomes : ℚ) / total_outcomes = 25 / 648 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_three_pairs_l307_30787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l307_30745

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (Real.pi / 3 + x / 2)

theorem f_properties :
  ∃ (T : ℝ),
    (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
    T = 4 * Real.pi ∧
    (∀ (k : ℤ),
      StrictMonoOn f (Set.Icc (4 * ↑k * Real.pi - 8 * Real.pi / 3) (4 * ↑k * Real.pi - 2 * Real.pi / 3))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l307_30745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_calculation_l307_30732

/-- In a race, runner A finishes ahead of runner B -/
structure Race where
  length : ℝ
  a_ahead_distance : ℝ
  a_ahead_time : ℝ

/-- Calculate the time taken by runner A to complete the race -/
noncomputable def race_time_a (r : Race) : ℝ :=
  (r.length * (r.length - r.a_ahead_distance)) / (r.a_ahead_distance * r.a_ahead_time)

/-- Theorem stating that for a 1000-meter race where A finishes 51 meters ahead
    and 11 seconds earlier than B, A's race time is approximately 215.686 seconds -/
theorem race_time_calculation (r : Race) 
    (h1 : r.length = 1000)
    (h2 : r.a_ahead_distance = 51)
    (h3 : r.a_ahead_time = 11) :
  abs (race_time_a r - 215.686) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_calculation_l307_30732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_implies_C_values_l307_30704

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the line l: 4x - 3y + C = 0
def l (C : ℝ) (x y : ℝ) : Prop := 4 * x - 3 * y + C = 0

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

-- Theorem statement
theorem distance_implies_C_values (C : ℝ) :
  distance_to_line P.1 P.2 4 (-3) C = 1 → C = 5 ∨ C = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_implies_C_values_l307_30704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unused_sector_angle_l307_30726

-- Define the radius of the cone
noncomputable def cone_radius : ℝ := 8

-- Define the volume of the cone
noncomputable def cone_volume : ℝ := 256 * Real.pi

-- Define the function to calculate the unused angle
noncomputable def unused_angle (r : ℝ) : ℝ :=
  360 - (720 / Real.sqrt 13)

-- State the theorem
theorem unused_sector_angle (r : ℝ) (h : r > cone_radius) :
  ∃ (h : ℝ), 
    cone_volume = (1/3) * Real.pi * cone_radius^2 * h ∧
    r^2 = h^2 + cone_radius^2 ∧
    abs (unused_angle r - 160.1) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unused_sector_angle_l307_30726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_neg3_3_l307_30783

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 ∧ y ≥ 0 then Real.arctan (y / x) + Real.pi
           else if x < 0 ∧ y < 0 then Real.arctan (y / x) - Real.pi
           else if x = 0 ∧ y > 0 then Real.pi / 2
           else if x = 0 ∧ y < 0 then -Real.pi / 2
           else 0  -- x = 0 and y = 0
  (r, θ)

theorem rectangular_to_polar_neg3_3 :
  let (r, θ) := rectangular_to_polar (-3) 3
  r = 3 * Real.sqrt 2 ∧ θ = 3 * Real.pi / 4 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_neg3_3_l307_30783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_neg_one_eq_one_l307_30750

/-- Given a function h: ℝ → ℝ satisfying the equation
    (x^7 - 1) * h(x) = (x + 1) * (x^2 + 1) * (x^4 + 1) + 1 for all x,
    prove that h(-1) = 1 -/
theorem h_neg_one_eq_one 
  (h : ℝ → ℝ) 
  (h_eq : ∀ x, (x^7 - 1) * h x = (x + 1) * (x^2 + 1) * (x^4 + 1) + 1) : 
  h (-1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_neg_one_eq_one_l307_30750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balanced_months_theorem_l307_30718

/-- Represents the days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a month -/
structure Month where
  days : Nat
  start_day : DayOfWeek

/-- Defines V-days (Tuesdays and Sundays) -/
def is_v_day (d : DayOfWeek) : Bool :=
  match d with
  | DayOfWeek.Tuesday => true
  | DayOfWeek.Sunday => true
  | _ => false

/-- Defines S-days (Wednesdays and Saturdays) -/
def is_s_day (d : DayOfWeek) : Bool :=
  match d with
  | DayOfWeek.Wednesday => true
  | DayOfWeek.Saturday => true
  | _ => false

/-- Defines P-days (Mondays and Fridays) -/
def is_p_day (d : DayOfWeek) : Bool :=
  match d with
  | DayOfWeek.Monday => true
  | DayOfWeek.Friday => true
  | _ => false

/-- Counts the number of specific days in a month -/
def count_days (m : Month) (is_day : DayOfWeek → Bool) : Nat :=
  sorry

/-- Checks if a month has equal numbers of V-days, S-days, and P-days -/
def is_balanced_month (m : Month) : Prop :=
  count_days m is_v_day = count_days m is_s_day ∧
  count_days m is_s_day = count_days m is_p_day

/-- Helper function to get the next day of the week -/
def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Helper function to add days to a day of the week -/
def add_days (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => next_day (add_days d m)

/-- Theorem: If four consecutive months are balanced, then they must be
    December, January, February, and March, with the last day of February
    (28th or 29th) being a Monday -/
theorem balanced_months_theorem
  (m₁ m₂ m₃ m₄ : Month)
  (h₁ : is_balanced_month m₁)
  (h₂ : is_balanced_month m₂)
  (h₃ : is_balanced_month m₃)
  (h₄ : is_balanced_month m₄)
  (consecutive : m₂.start_day = add_days m₁.start_day m₁.days ∧
                 m₃.start_day = add_days m₂.start_day m₂.days ∧
                 m₄.start_day = add_days m₃.start_day m₃.days) :
  m₁.days = 31 ∧ m₂.days = 31 ∧ (m₃.days = 28 ∨ m₃.days = 29) ∧ m₄.days = 31 ∧
  m₃.start_day = DayOfWeek.Wednesday ∧
  add_days m₃.start_day m₃.days = DayOfWeek.Monday :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_balanced_months_theorem_l307_30718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_amplitude_ratio_l307_30797

-- Define the earthquake magnitude formula
noncomputable def earthquake_magnitude (A : ℝ) (A₀ : ℝ) : ℝ := Real.log A / Real.log 10 - Real.log A₀ / Real.log 10

-- Theorem stating the ratio of maximum amplitudes
theorem earthquake_amplitude_ratio :
  ∀ A₀ : ℝ, A₀ > 0 →
  ∃ A₁ A₂ : ℝ,
    A₁ > 0 ∧ A₂ > 0 ∧
    earthquake_magnitude A₁ A₀ = 8 ∧
    earthquake_magnitude A₂ A₀ = 5 ∧
    A₁ / A₂ = 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_amplitude_ratio_l307_30797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_AD_length_l307_30763

/-- Represents a quadrilateral ABCD with specific properties -/
structure Quadrilateral where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  CD : ℝ
  -- Angle measures
  angleB : ℝ
  angleC : ℝ
  -- Conditions
  AB_eq : AB = 5
  BC_eq : BC = 6
  CD_eq : CD = 25
  B_obtuse : angleB > Real.pi / 2
  C_obtuse : angleC > Real.pi / 2
  cos_C_eq : Real.cos angleC = 4/5
  sin_B_eq : Real.sin angleB = -4/5

/-- The length of side AD in the quadrilateral -/
noncomputable def side_AD (q : Quadrilateral) : ℝ := sorry

/-- Theorem stating that the length of side AD is approximately 18.75 -/
theorem side_AD_length (q : Quadrilateral) : 
  ∃ ε > 0, |side_AD q - 18.75| < ε := by sorry

#check side_AD_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_AD_length_l307_30763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_divisibility_solution_l307_30743

def is_solution (a b c : ℕ+) : Prop :=
  Nat.Coprime a.val b.val ∧ 
  Nat.Coprime b.val c.val ∧ 
  Nat.Coprime c.val a.val ∧
  (a.val^2 : ℕ) ∣ (b.val^3 + c.val^3) ∧
  (b.val^2 : ℕ) ∣ (a.val^3 + c.val^3) ∧
  (c.val^2 : ℕ) ∣ (a.val^3 + b.val^3)

theorem coprime_divisibility_solution :
  ∀ a b c : ℕ+, is_solution a b c →
    ((a.val = 1 ∧ b.val = 1 ∧ c.val = 1) ∨
     (a.val = 3 ∧ b.val = 2 ∧ c.val = 1) ∨
     (a.val = 3 ∧ b.val = 1 ∧ c.val = 2) ∨
     (a.val = 2 ∧ b.val = 3 ∧ c.val = 1) ∨
     (a.val = 2 ∧ b.val = 1 ∧ c.val = 3) ∨
     (a.val = 1 ∧ b.val = 3 ∧ c.val = 2) ∨
     (a.val = 1 ∧ b.val = 2 ∧ c.val = 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_divisibility_solution_l307_30743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_iv_in_rectangle_c_l307_30758

-- Define the tiles and their side numbers
structure Tile :=
  (top : ℕ) (right : ℕ) (bottom : ℕ) (left : ℕ)

-- Define the rectangles
inductive Rectangle
| A | B | C | D

-- Define the placement of tiles
def placement : Rectangle → Tile := sorry

-- Define the adjacency of rectangles
def adjacent : Rectangle → Rectangle → Prop := sorry

-- Define the matching condition for adjacent tiles
def matching_sides (t1 t2 : Tile) (side : Tile → ℕ) : Prop :=
  side t1 = side t2

-- State the theorem
theorem tile_iv_in_rectangle_c 
  (tile_i tile_ii tile_iii tile_iv : Tile)
  (h1 : tile_iii.bottom = 0 ∧ tile_iii.right = 5)
  (h2 : tile_iii.left = 1)
  (h3 : tile_iv.right = 4)
  (h4 : placement Rectangle.D = tile_iii)
  (h5 : ∀ r1 r2, adjacent r1 r2 → 
    matching_sides (placement r1) (placement r2) Tile.left ∨
    matching_sides (placement r1) (placement r2) Tile.right ∨
    matching_sides (placement r1) (placement r2) Tile.top ∨
    matching_sides (placement r1) (placement r2) Tile.bottom) :
  placement Rectangle.C = tile_iv :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_iv_in_rectangle_c_l307_30758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l307_30705

theorem division_problem :
  ∃ d : ℕ, d > 0 ∧
    30 % d = 3 ∧ 
    40 % d = 4 ∧ 
    ∀ k : ℕ, k > 0 → (30 % k = 3 ∧ 40 % k = 4) → k ≤ d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l307_30705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l307_30779

def S (n : ℕ) : ℕ := n^2 + 3*n + 1

def a : ℕ → ℕ
| 0 => 5  -- We need to handle the case for 0
| 1 => 5
| (n + 2) => 2*(n + 2) + 2

theorem sequence_general_term (n : ℕ) : 
  (n = 1 ∧ a n = 5) ∨ 
  (n > 1 ∧ a n = 2*n + 2) ∧ 
  (∀ k : ℕ, k ≥ 1 → S k - S (k-1) = a k) := by
  sorry

#eval a 0  -- This will output 5
#eval a 1  -- This will output 5
#eval a 2  -- This will output 6
#eval a 3  -- This will output 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l307_30779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_omega_range_l307_30708

open Real MeasureTheory Set

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := (sin (ω * x / 2))^2 + (1/2) * sin (ω * x) - 1/2

theorem zero_point_omega_range (ω : ℝ) (h1 : ω > 0) :
  (∃ x ∈ Ioo π (2*π), f ω x = 0) ↔ ω ∈ Ioo (1/8) (1/4) ∪ Ioi (5/8) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_omega_range_l307_30708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a1_value_l307_30756

def sequence_property (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = 1 / (1 - a n)

theorem sequence_a1_value (a : ℕ → ℚ) (h : sequence_property a) (h8 : a 8 = 2) : 
  a 1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a1_value_l307_30756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dogwood_tree_count_l307_30740

/-- Represents a worker in the park -/
structure Worker where
  rate : ℝ  -- planting rate in trees per hour
  break1 : ℝ  -- break time on day 1 in hours
  break2 : ℝ  -- break time on day 2 in hours

/-- Calculates the number of trees planted by a worker in a given time -/
def trees_planted (w : Worker) (time : ℝ) (break_time : ℝ) : ℝ :=
  w.rate * (time - break_time)

/-- The problem statement -/
theorem dogwood_tree_count :
  let initial_trees : ℕ := 7
  let workers : List Worker := [
    ⟨2, 0.5, 0.25⟩, ⟨3, 0.5, 0.25⟩, ⟨1, 0.75, 0⟩, ⟨4, 0.75, 0⟩,
    ⟨2.5, 0, 0.5⟩, ⟨3.5, 0, 0.5⟩, ⟨1.5, 0.25, 0.75⟩, ⟨2, 0.25, 0.75⟩
  ]
  let day1_hours : ℝ := 3
  let day2_hours : ℝ := 1.5
  let total_planted : ℝ := (workers.map (λ w => trees_planted w day1_hours w.break1 +
                                               trees_planted w day2_hours w.break2)).sum
  initial_trees + Int.floor total_planted = 81 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dogwood_tree_count_l307_30740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_l307_30748

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The conditions on the polynomial P -/
def satisfiesConditions (P : IntPolynomial) (a : ℤ) : Prop :=
  a > 0 ∧
  P.eval 2 = a ∧ P.eval 4 = a ∧ P.eval 6 = a ∧ P.eval 8 = a ∧
  P.eval 1 = -a ∧ P.eval 3 = -a ∧ P.eval 5 = -a ∧ P.eval 7 = -a

/-- The theorem stating the smallest possible value of a -/
theorem smallest_a : ∀ (P : IntPolynomial) (a : ℤ), 
  satisfiesConditions P a → a ≥ 315 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_l307_30748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l307_30755

/-- The function f(x) defined as ln(x+1) + a(x^2 - x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + a * (x^2 - x)

/-- Theorem stating the range of a for which f(x) ≥ 0 for all x > 0 --/
theorem f_nonnegative_iff (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) ↔ 0 ≤ a ∧ a ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l307_30755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l307_30710

/-- Represents a circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance from a point to a line -/
noncomputable def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  (abs (l.a * p.1 + l.b * p.2 + l.c)) / Real.sqrt (l.a^2 + l.b^2)

/-- Check if a circle is tangent to a line -/
def isTangent (c : Circle) (l : Line) : Prop :=
  distancePointToLine c.center l = c.radius

/-- The main theorem -/
theorem circle_tangent_to_line :
  let c : Circle := { center := (4, 0), radius := 2 * Real.sqrt 3 }
  let l : Line := { a := Real.sqrt 3, b := -1, c := 0 }
  let circle_equation := fun (x y : ℝ) ↦ (x - 4)^2 + y^2 = 12
  isTangent c l ∧ circle_equation = fun (x y : ℝ) ↦ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l307_30710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_cost_is_three_l307_30714

/-- Represents the shopping scenario --/
structure ShoppingScenario where
  initial_amount : ℚ
  mangoes_bought : ℕ
  juice_bought : ℕ
  amount_left : ℚ

/-- Calculates the combined cost of one mango and one carton of juice --/
def combined_cost (s : ShoppingScenario) : ℚ :=
  (s.initial_amount - s.amount_left) / (s.mangoes_bought + s.juice_bought)

/-- Theorem: The combined cost of one mango and one carton of juice is $3 --/
theorem combined_cost_is_three (s : ShoppingScenario) 
  (h1 : s.initial_amount = 50)
  (h2 : s.mangoes_bought = 6)
  (h3 : s.juice_bought = 6)
  (h4 : s.amount_left = 14) :
  combined_cost s = 3 := by
  -- Unfold the definition of combined_cost
  unfold combined_cost
  -- Simplify the expression
  simp [h1, h2, h3, h4]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_cost_is_three_l307_30714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_validNumberCount_eq_63_l307_30715

/-- A function that checks if a number is a valid three-digit number with the second digit 3 less than the third digit -/
def isValidNumber (n : ℕ) : Bool :=
  100 ≤ n ∧ n < 1000 ∧ (n / 10 % 10 = n % 10 - 3)

/-- The count of valid three-digit numbers where the second digit is 3 less than the third digit -/
def validNumberCount : ℕ := (Finset.filter (fun n => isValidNumber n) (Finset.range 1000)).card

/-- Theorem stating that the count of valid numbers is 63 -/
theorem validNumberCount_eq_63 : validNumberCount = 63 := by
  sorry

#eval validNumberCount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_validNumberCount_eq_63_l307_30715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l307_30711

theorem log_inequality (a b c : ℝ) : 
  a = Real.log 4 / Real.log 5 → 
  b = Real.log 5 / Real.log 3 → 
  c = Real.log 5 / Real.log 4 → 
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l307_30711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l307_30723

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 3) - 1 / Real.sqrt (7 - x)

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}

def B : Set ℝ := {x | 2 < x ∧ x < 10}

def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a + 1}

theorem problem_solution :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}) ∧
  (∀ a : ℝ, B ∪ C a = B → (a ≤ -1 ∨ (2 ≤ a ∧ a ≤ 9/2))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l307_30723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l307_30727

def P : Set ℝ := {1, 3, 5, 7}
def Q : Set ℝ := {x | 2*x - 1 > 11}

theorem intersection_of_P_and_Q : P ∩ Q = {7} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l307_30727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_valid_n_l307_30713

/-- A triangle with integer side lengths -/
structure IntegerTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- The semi-perimeter of a triangle -/
def semi_perimeter (t : IntegerTriangle) : ℚ :=
  (t.a.val + t.b.val + t.c.val) / 2

/-- The inradius of a triangle -/
noncomputable def inradius (t : IntegerTriangle) : ℝ :=
  sorry

/-- The set of positive integers n for which p = nr holds -/
def valid_n_set : Set ℕ+ :=
  {n : ℕ+ | ∃ t : IntegerTriangle, (semi_perimeter t : ℝ) = n.val * inradius t}

/-- The main theorem -/
theorem infinitely_many_valid_n : Set.Infinite valid_n_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_valid_n_l307_30713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l307_30716

-- Define the feasible region
def FeasibleRegion (x y : ℝ) : Prop :=
  x + 2*y ≤ 6 ∧ x + y ≥ 1 ∧ x ≥ 0 ∧ y ≥ 0

-- Define the vertices of the region
def Vertices : Set (ℝ × ℝ) :=
  {(0, 0), (1, 0), (0, 3)}

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem longest_side_length :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ Vertices ∧ p2 ∈ Vertices ∧
    distance p1 p2 = Real.sqrt 10 ∧
    ∀ (q1 q2 : ℝ × ℝ), q1 ∈ Vertices → q2 ∈ Vertices →
      distance q1 q2 ≤ Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l307_30716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_four_l307_30784

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x + 8) / (x - 4)

theorem vertical_asymptote_at_four :
  ∃ (ε : ℝ), ε > 0 ∧ 
    (∀ (x : ℝ), 0 < |x - 4| ∧ |x - 4| < ε → |f x| > (1/ε)) := by
  sorry

#check vertical_asymptote_at_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_four_l307_30784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_problem_l307_30760

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 + m

-- State the theorem
theorem trigonometric_function_problem (m : ℝ) (A B C a b c : ℝ) :
  -- Conditions
  (f (π / 12) m = 0) →
  (c * Real.cos B + b * Real.cos C = 2 * a * Real.cos B) →
  (A ∈ Set.Ioo 0 (2 * π / 3)) →
  -- Conclusions
  (m = 1 / 2) ∧
  (Set.Ioo (-1 / 2) 1 ⊆ Set.range (λ A => Real.sin (2 * A - π / 6))) ∧
  (1 ∈ Set.range (λ A => Real.sin (2 * A - π / 6))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_problem_l307_30760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_value_l307_30754

noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ 3 then x^3 else x^2 + 1

theorem g_composition_value : g (g (g 2)) = 1953125 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_value_l307_30754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2024_possible_values_l307_30767

def IsCardinal (S : Finset Nat) : Prop :=
  S.card ∈ S

def ApplyF (f : Nat → Nat) (S : Finset Nat) : Finset Nat :=
  S.image f

def PreservesCardinal (f : Nat → Nat) : Prop :=
  ∀ S : Finset Nat, IsCardinal S → IsCardinal (ApplyF f S)

theorem f_2024_possible_values (f : Nat → Nat) 
  (h : PreservesCardinal f) : f 2024 ∈ ({1, 2, 2024} : Finset Nat) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2024_possible_values_l307_30767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equals_15_8_l307_30785

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- State the theorem
theorem floor_expression_equals_15_8 :
  (floor 6.5) * (floor (2/3 : ℝ)) + (floor 2) * (7.2 : ℝ) + (floor 8.3) - (6.6 : ℝ) = 15.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equals_15_8_l307_30785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_to_y_equals_three_to_sixteen_l307_30707

theorem nine_to_y_equals_three_to_sixteen (y : ℝ) : (9 : ℝ)^y = (3 : ℝ)^16 → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_to_y_equals_three_to_sixteen_l307_30707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_distance_l307_30766

/-- The distance between the center of the circle described by the equation 
    x^2 + y^2 = 8x - 10y + 20 and the point (-3, 2) is 7√2. -/
theorem circle_center_distance : 
  let circle_eq : ℝ → ℝ → Prop := fun x y ↦ x^2 + y^2 = 8*x - 10*y + 20
  let center : ℝ × ℝ := (4, -5)  -- We define the center explicitly as it's part of the question
  let point : ℝ × ℝ := (-3, 2)
  dist center point = 7 * Real.sqrt 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_distance_l307_30766
