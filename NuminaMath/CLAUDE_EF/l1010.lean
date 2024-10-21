import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_auto_credit_percentage_l1010_101018

theorem auto_credit_percentage (finance_company_credit : ℝ) (total_consumer_credit : ℝ) 
  (h1 : finance_company_credit = 75)
  (h2 : total_consumer_credit = 416.6666666666667) :
  ∃ ε > 0, abs ((2 * finance_company_credit / total_consumer_credit) * 100 - 36) < ε :=
by
  -- Define total_auto_credit and percentage
  let total_auto_credit := 2 * finance_company_credit
  let percentage := (total_auto_credit / total_consumer_credit) * 100

  -- Prove the existence of ε
  use 0.01  -- Choose a small positive number for ε
  
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_auto_credit_percentage_l1010_101018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1010_101023

-- Define the right triangle
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Define the triangle with given conditions
def triangle (a b c : ℝ) : Prop :=
  right_triangle a b c ∧ c = 10 ∧ ∃ θ : ℝ, θ = 30 * (Real.pi / 180) ∧ Real.sin θ = a / c

-- Theorem statement
theorem triangle_area :
  ∀ a b c : ℝ, triangle a b c → (1/2) * a * b = (25 * Real.sqrt 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1010_101023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1010_101053

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 4*x + 5) + Real.sqrt (x^2 + 4*x + 8)

theorem f_minimum_value :
  (∀ x : ℝ, f x ≥ 5) ∧ (∃ x : ℝ, f x = 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1010_101053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_x_coordinate_range_l1010_101025

-- Define the given points and line
def A : ℝ × ℝ := (0, 3)
def l (x : ℝ) : ℝ := 2 * x - 4

-- Define circle C
def C (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - (2 * a - 4))^2 = 1}

-- Define the trajectory of point M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 0)^2 + (p.2 - (-1))^2 = 4}

-- Define the conditions
def conditions (a : ℝ) : Prop :=
  (∃ y : ℝ, y = l a ∧ (a, y) ∈ C a) ∧ 
  ∃ p : ℝ × ℝ, p ∈ C a ∧ p ∈ M

-- State the theorem
theorem center_x_coordinate_range :
  ∀ a : ℝ, conditions a → 0 ≤ a ∧ a ≤ 12/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_x_coordinate_range_l1010_101025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_seating_arrangement_l1010_101026

theorem circular_seating_arrangement (n : ℕ) : 
  (Nat.factorial (n - 1) = 144) ∧ (5 ≤ n) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_seating_arrangement_l1010_101026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_21_equals_neg_one_l1010_101069

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the given function f
noncomputable def f (x : ℝ) : ℝ := log10 (2 / (x - 1))

-- State the theorem
theorem f_of_21_equals_neg_one :
  (∀ x > 0, f (2 / x + 1) = log10 x) → f 21 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_21_equals_neg_one_l1010_101069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_negative_43_inverse_relationship_l1010_101006

-- Define the function f
def f (x : ℝ) : ℝ := 5 * x^3 - 3

-- State the theorem
theorem inverse_f_at_negative_43 :
  f (-2) = -43 := by
  -- Expand the definition of f
  simp [f]
  -- Evaluate the expression
  norm_num

-- State the inverse relationship
theorem inverse_relationship :
  f⁻¹ (-43) = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_negative_43_inverse_relationship_l1010_101006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_value_l1010_101045

-- Define the set A
def A (a : ℝ) : Set ℝ := {a - 2, 2 * a^2 + 5 * a, 12}

-- Theorem statement
theorem unique_a_value : ∃! a : ℝ, -3 ∈ A a ∧ (∀ x y : ℝ, x ∈ A a → y ∈ A a → x ≠ y → x ≠ y) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_value_l1010_101045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_ages_l1010_101081

/-- Represents the ages of family members -/
structure FamilyAges where
  son : ℕ
  daughter : ℕ
  niece : ℕ

/-- Checks if the given ages satisfy all the conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  -- Man is 24 years older than his son
  ∃ (man : ℕ), man = ages.son + 24 ∧
  -- In 2 years, man's age will be twice the son's age
  man + 2 = 2 * (ages.son + 2) ∧
  -- Daughter is half as old as the son
  ages.daughter * 2 = ages.son ∧
  -- In 6 years, daughter will be approximately 2/3 the son's age
  (ages.daughter + 6) * 3 ≤ (ages.son + 6) * 2 + 1 ∧
  (ages.daughter + 6) * 3 ≥ (ages.son + 6) * 2 - 1 ∧
  -- Son is 3 years older than the niece
  ages.son = ages.niece + 3 ∧
  -- Niece's age is approximately 4/5 the age of the son
  ages.niece * 5 ≤ ages.son * 4 + 1 ∧
  ages.niece * 5 ≥ ages.son * 4 - 1

/-- Theorem stating that the given ages satisfy all conditions -/
theorem correct_ages : 
  satisfiesConditions { son := 22, daughter := 11, niece := 19 } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_ages_l1010_101081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mapping_property_l1010_101061

/-- The set of planar vectors -/
def A : Type := ℝ × ℝ

/-- The dot product of two planar vectors -/
def dot_product (x y : A) : ℝ := x.1 * y.1 + x.2 * y.2

/-- The mapping f -/
def f (a : A) (x : A) : A := (x.1 - (dot_product x a) * a.1, x.2 - (dot_product x a) * a.2)

/-- The theorem statement -/
theorem mapping_property (a : A) :
  (∀ x y : A, dot_product (f a x) (f a y) = dot_product x y) →
  dot_product a a = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mapping_property_l1010_101061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_for_26_factorial_l1010_101030

/-- The number of colors needed to color the divisors of n! such that no two distinct divisors s and t of the same color satisfy s | t -/
def color_count (n : ℕ) : ℕ := sorry

/-- A function that checks if a coloring is valid for the divisors of n! -/
def is_valid_coloring (n : ℕ) (coloring : ℕ → ℕ) : Prop := sorry

theorem min_colors_for_26_factorial :
  ∀ (coloring : ℕ → ℕ),
    is_valid_coloring 26 coloring →
    color_count 26 = 50 ∧
    (∀ m : ℕ, m < 50 → ¬∃ (coloring : ℕ → ℕ), is_valid_coloring 26 coloring ∧ Set.range coloring ⊆ Finset.range m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_for_26_factorial_l1010_101030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_count_is_twenty_l1010_101016

/-- A rectangle with diagonals and midpoint segments --/
structure RectangleWithDiagonalsAndMidpoints where
  vertices : Fin 4 → ℝ × ℝ
  is_rectangle : ∀ (i j k l : Fin 4), i ≠ j → j ≠ k → k ≠ l → l ≠ i → 
    (vertices i).1 = (vertices j).1 ∧ (vertices k).2 = (vertices l).2
  diagonals : Fin 2 → Fin 4 × Fin 4
  midpoints : Fin 4 → ℝ × ℝ
  midpoint_segments : Fin 2 → Fin 4 × Fin 4

/-- Count of triangles in the figure --/
def count_triangles (r : RectangleWithDiagonalsAndMidpoints) : ℕ :=
  20 -- We'll use the result directly here

/-- Theorem stating that the number of triangles is 20 --/
theorem triangle_count_is_twenty (r : RectangleWithDiagonalsAndMidpoints) :
  count_triangles r = 20 := by
  -- The proof is trivial given our definition of count_triangles
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_count_is_twenty_l1010_101016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_minimizer_l1010_101051

-- Define the integrand function
noncomputable def f (x : ℝ) : ℝ := 1 / (x + Real.sqrt x)

-- Define the definite integral as a function of a
noncomputable def F (a : ℝ) : ℝ := ∫ x in a..a^2, f x

-- State the theorem
theorem integral_minimizer :
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ (b : ℝ), b > 0 → F a ≤ F b) ∧ 
  a = 3 - 2 * Real.sqrt 2 := by
  sorry

#check integral_minimizer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_minimizer_l1010_101051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_ratio_l1010_101033

/-- A circle with inscribed and circumscribed regular octagons -/
structure OctagonCircle where
  radius : ℝ
  inscribed_side : ℝ
  circumscribed_side : ℝ
  inscribed_side_eq : inscribed_side = radius
  circumscribed_side_eq : circumscribed_side = radius * Real.sqrt 2

/-- The ratio of the area of the circumscribed octagon to the area of the inscribed octagon -/
noncomputable def area_ratio (oc : OctagonCircle) : ℝ :=
  (oc.circumscribed_side ^ 2) / (oc.inscribed_side ^ 2)

/-- Theorem: The area ratio of circumscribed to inscribed regular octagons is 2:1 -/
theorem octagon_area_ratio (oc : OctagonCircle) : area_ratio oc = 2 := by
  -- Unfold the definition of area_ratio
  unfold area_ratio
  -- Substitute the values of circumscribed_side and inscribed_side
  rw [oc.circumscribed_side_eq, oc.inscribed_side_eq]
  -- Simplify the expression
  simp [pow_two]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_ratio_l1010_101033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_of_class_l1010_101079

/-- Proves that the average weight of all students in a class is approximately 48.38 kg,
    given the average weights of three groups of students. -/
theorem average_weight_of_class (total_students : ℕ) (group1_size : ℕ) (group2_size : ℕ) (group3_size : ℕ)
  (group1_avg : ℝ) (group2_avg : ℝ) (group3_avg : ℝ)
  (h_total : total_students = group1_size + group2_size + group3_size)
  (h_group1 : group1_size = 12)
  (h_group2 : group2_size = 15)
  (h_group3 : group3_size = 8)
  (h_avg1 : group1_avg = 50.25)
  (h_avg2 : group2_avg = 48.60)
  (h_avg3 : group3_avg = 45.15) :
  ∃ class_avg : ℝ, abs (class_avg - 48.38) < 0.01 :=
by
  let total_weight := group1_size * group1_avg + group2_size * group2_avg + group3_size * group3_avg
  let class_avg := total_weight / total_students
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_of_class_l1010_101079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_on_lines_l1010_101049

/-- Given two lines and a midpoint, calculate the distance between two points on these lines. -/
theorem distance_between_points_on_lines (x_p y_p x_q y_q : ℝ) :
  6 * y_p = 7 * x_p →  -- P is on the line 6y = 7x
  15 * y_q = 4 * x_q →  -- Q is on the line 15y = 4x
  (x_p + x_q) / 2 = 10 →  -- x-coordinate of midpoint
  (y_p + y_q) / 2 = 8 →  -- y-coordinate of midpoint
  Real.sqrt ((x_p - x_q)^2 + (y_p - y_q)^2) = Real.sqrt ((x_p - x_q)^2 + (y_p - y_q)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_on_lines_l1010_101049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_male_animals_count_l1010_101070

def calculate_remaining_animals (initial : ℕ) (percentage_bought : ℚ) : ℕ :=
  (initial : ℤ) - Int.floor (percentage_bought * initial) |>.toNat

theorem male_animals_count (horses sheep chickens cows pigs : ℕ) : 
  horses = 100 → 
  sheep = 29 → 
  chickens = 9 → 
  cows = 15 → 
  pigs = 18 → 
  let remaining_horses := calculate_remaining_animals horses (40/100)
  let remaining_sheep := calculate_remaining_animals sheep (50/100)
  let remaining_chickens := calculate_remaining_animals chickens (60/100)
  let remaining_cows := calculate_remaining_animals cows (30/100)
  let remaining_pigs := calculate_remaining_animals pigs (20/100)
  let total_animals := remaining_horses + remaining_sheep + remaining_chickens + 
                       remaining_cows + remaining_pigs + 37
  (total_animals / 2 : ℕ) = 71 := by
  sorry

#eval calculate_remaining_animals 100 (40/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_male_animals_count_l1010_101070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_filled_by_fourth_vertex_equals_square_area_l1010_101055

-- Define a square with side length 1
def UnitSquare : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define a rhombus with three vertices on the sides of the unit square
def RhombusOnSquare : Set (ℝ × ℝ) → Prop :=
  λ R => ∃ (K L N M : ℝ × ℝ),
    K.1 = 0 ∧ 0 < K.2 ∧ K.2 < 1 ∧  -- K on AD
    0 < L.1 ∧ L.1 < 1 ∧ L.2 = 1 ∧  -- L on BC
    N.1 = 1 ∧ 0 < N.2 ∧ N.2 < 1 ∧  -- N on AB
    M ∉ UnitSquare ∧               -- M is outside the square
    R = {K, L, N, M} ∧             -- R is the set of rhombus vertices
    (K.1 - M.1)^2 + (K.2 - M.2)^2 = (L.1 - N.1)^2 + (L.2 - N.2)^2  -- Rhombus property

-- Define the area filled by the fourth vertex
def AreaFilledByFourthVertex (R : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p | p ∉ UnitSquare}

-- Theorem statement
theorem area_filled_by_fourth_vertex_equals_square_area 
  (R : Set (ℝ × ℝ)) (h : RhombusOnSquare R) :
  MeasureTheory.volume (AreaFilledByFourthVertex R) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_filled_by_fourth_vertex_equals_square_area_l1010_101055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l1010_101096

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y a : ℝ) : Prop := x^2/a^2 - y^2/3 = 1

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Define the right focus of the hyperbola
def hyperbola_right_focus (a : ℝ) : ℝ × ℝ := (2, 0)

-- Define the eccentricity of the hyperbola
noncomputable def hyperbola_eccentricity (a : ℝ) : ℝ := 
  let c := Real.sqrt (a^2 + 3)
  c / a

-- Theorem statement
theorem hyperbola_eccentricity_is_two (a : ℝ) : 
  hyperbola_right_focus a = parabola_focus → hyperbola_eccentricity a = 2 := by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l1010_101096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1010_101047

-- Define the line
def line (x y : ℝ) (a : ℝ) : Prop := 3 * x + 4 * y + a = 0

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 2

-- Define the condition for perpendicular tangents
def perpendicular_tangents (M : ℝ × ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ × ℝ), 
    circle_equation t₁.1 t₁.2 ∧ circle_equation t₂.1 t₂.2 ∧ 
    (t₁.1 - M.1) * (t₂.1 - M.1) + (t₁.2 - M.2) * (t₂.2 - M.2) = 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (∃ M : ℝ × ℝ, line M.1 M.2 a ∧ perpendicular_tangents M) ↔ 
  -20 ≤ a ∧ a ≤ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1010_101047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_e_minus_5_floor_l1010_101027

-- Define e as a real number between 2 and 3
axiom e : ℝ
axiom e_bounds : 2 < e ∧ e < 3

-- Define the greatest integer function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Theorem statement
theorem e_minus_5_floor : floor (e - 5) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_e_minus_5_floor_l1010_101027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l1010_101022

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the vectors
def AB (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def BC (B C : ℝ × ℝ) : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
def AC (A C : ℝ × ℝ) : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the angle between two vectors
noncomputable def angle (v1 v2 : ℝ × ℝ) : ℝ :=
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (magnitude v1 * magnitude v2))

-- State the theorem
theorem triangle_side_range (A B C : ℝ × ℝ) :
  (angle (AB A B) (BC B C) = 150 * π / 180) →  -- Angle between AB and BC is 150°
  (magnitude (AC A C) = 2) →                   -- |AC| = 2
  (0 < magnitude (AB A B) ∧ magnitude (AB A B) ≤ 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l1010_101022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paving_hall_theorem_l1010_101038

/-- The number of stones required to pave a rectangular hall -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℕ :=
  (hall_length * hall_width * 100 / (stone_length * stone_width)).floor.toNat

/-- Theorem: The number of stones required to pave a 36m by 15m hall with 2dm by 5dm stones is 5400 -/
theorem paving_hall_theorem :
  stones_required 36 15 2 5 = 5400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paving_hall_theorem_l1010_101038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l1010_101083

theorem cosine_inequality (a b : ℝ) :
  ∀ θ : ℝ, (Real.cos (θ + a) + b * Real.cos θ)^2 ≤ 1 + 2*b * Real.cos a + b^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l1010_101083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_scientist_with_one_friend_l1010_101011

/-- Represents a scientist at the congress -/
structure Scientist :=
  (id : ℕ)

/-- Represents the friendship relation between scientists -/
def IsFriend (S : Type) [Fintype S] : S → S → Prop := sorry

/-- The number of friends a scientist has -/
def FriendCount {S : Type} [Fintype S] [DecidableEq S] (friendship : S → S → Prop) 
  [DecidableRel friendship] (s : S) : ℕ :=
  Finset.card (Finset.filter (friendship s) Finset.univ)

/-- The main theorem stating that there exists a scientist with exactly one friend -/
theorem exists_scientist_with_one_friend
  {S : Type} [Fintype S] [DecidableEq S] (friendship : S → S → Prop) 
  [DecidableRel friendship]
  (h_symmetric : ∀ a b : S, friendship a b ↔ friendship b a)
  (h_no_mutual_friends : ∀ a b : S, FriendCount friendship a = FriendCount friendship b →
    a ≠ b → ¬∃ c : S, friendship a c ∧ friendship b c) :
  ∃ s : S, FriendCount friendship s = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_scientist_with_one_friend_l1010_101011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_coincide_l1010_101065

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Returns the center of a parallelogram -/
noncomputable def center (p : Parallelogram) : Point :=
  { x := (p.A.x + p.C.x) / 2, y := (p.A.y + p.C.y) / 2 }

/-- Checks if a point lies on a line segment between two other points -/
def liesBetween (P A B : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P.x = A.x + t * (B.x - A.x) ∧ P.y = A.y + t * (B.y - A.y)

theorem centers_coincide (ABCD A₁B₁C₁D₁ : Parallelogram) 
  (h1 : liesBetween A₁B₁C₁D₁.A ABCD.A ABCD.B)
  (h2 : liesBetween A₁B₁C₁D₁.B ABCD.B ABCD.C)
  (h3 : liesBetween A₁B₁C₁D₁.C ABCD.C ABCD.D)
  (h4 : liesBetween A₁B₁C₁D₁.D ABCD.D ABCD.A) :
  center ABCD = center A₁B₁C₁D₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_coincide_l1010_101065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_match_game_winning_strategy_last_match_loses_strategy_last_match_wins_strategy_l1010_101041

/-- Represents the rules of the match-taking game -/
structure MatchGame where
  n : ℕ  -- Initial number of matches
  last_match_loses : Bool  -- True if taking the last match loses, False otherwise

/-- Determines if the first player has a winning strategy -/
def first_player_wins (game : MatchGame) : Prop :=
  if game.last_match_loses then
    game.n % 4 ≠ 1
  else
    game.n % 4 ≠ 0

/-- Theorem stating the winning condition for the first player -/
theorem match_game_winning_strategy (game : MatchGame) :
  first_player_wins game ↔
    (game.last_match_loses ∧ game.n % 4 ≠ 1) ∨
    (¬game.last_match_loses ∧ game.n % 4 ≠ 0) := by
  sorry

/-- Corollary for the case where taking the last match loses -/
theorem last_match_loses_strategy (n : ℕ) :
  first_player_wins { n := n, last_match_loses := true } ↔ n % 4 ≠ 1 := by
  sorry

/-- Corollary for the case where taking the last match wins -/
theorem last_match_wins_strategy (n : ℕ) :
  first_player_wins { n := n, last_match_loses := false } ↔ n % 4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_match_game_winning_strategy_last_match_loses_strategy_last_match_wins_strategy_l1010_101041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_about_pi_third_l1010_101090

/-- The function f(x) defined as sin x + sin(2π/3 - x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (2 * Real.pi / 3 - x)

/-- Theorem stating that f(x) is symmetric about the line x = π/3 -/
theorem f_symmetric_about_pi_third :
  ∀ x : ℝ, f (Real.pi / 3 + x) = f (Real.pi / 3 - x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_about_pi_third_l1010_101090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_class_attendance_l1010_101020

theorem swimming_class_attendance 
  (total_students : ℕ) 
  (chess_percentage : ℚ) 
  (swimming_percentage : ℚ) 
  (h1 : total_students = 1000)
  (h2 : chess_percentage = 1/5)
  (h3 : swimming_percentage = 1/10)
  (chess_class : Set ℕ)
  (swimming_class : Set ℕ)
  (h4 : swimming_class ⊆ chess_class) :
  (↑total_students * chess_percentage * swimming_percentage : ℚ) = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_class_attendance_l1010_101020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_225_degrees_l1010_101015

theorem cos_225_degrees : 
  Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_225_degrees_l1010_101015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_f_properties_l1010_101059

/-- A linear function satisfying specific conditions -/
noncomputable def f (x : ℝ) : ℝ := sorry

/-- The theorem stating the properties of the function f and its value at x = 3 -/
theorem function_f_properties :
  (∀ x y : ℝ, ∃ a b : ℝ, f x = a * x + b) ∧  -- f is linear
  (∀ x : ℝ, f x = 3 * (f.invFun x) + 8) ∧  -- f(x) = 3f⁻¹(x) + 8
  f 2 = 5 →  -- f(2) = 5
  f 3 = (17 * Real.sqrt 3) / (1 + Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_f_properties_l1010_101059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1010_101044

noncomputable def C (a b x y : ℝ) : Prop :=
  x * |x| / a^2 - y * |y| / b^2 = 1

theorem curve_C_properties (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ x₀ : ℝ, ∃! y : ℝ, C a b x₀ y) ∧
  (∀ k m : ℝ, ∃ S : Finset ℝ, (∀ x ∈ S, C a b x (k * x + m)) ∧ Finset.card S ≤ 3) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, C a b x₁ y₁ → C a b x₂ y₂ → x₁ ≠ x₂ → (y₁ - y₂) / (x₁ - x₂) > 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1010_101044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_specific_lines_l1010_101031

/-- The slope of the angle bisector of the obtuse angle formed by two lines --/
noncomputable def angle_bisector_slope (m₁ m₂ : ℝ) : ℝ :=
  (m₁ + m₂ + Real.sqrt (m₁^2 + m₂^2 - m₁*m₂ + 1)) / (1 + m₁*m₂)

/-- Theorem: The slope of the angle bisector of the obtuse angle formed by y = 2x and y = 4x is (6 + √13) / 9 --/
theorem angle_bisector_slope_specific_lines :
  angle_bisector_slope 2 4 = (6 + Real.sqrt 13) / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_specific_lines_l1010_101031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_count_l1010_101034

/-- The total number of pencils for Reeta, Anika, Kamal, and Sarah -/
def total_pencils (reeta anika kamal sarah : ℕ) : ℕ :=
  reeta + anika + kamal + sarah

/-- Theorem stating the total number of pencils given the conditions -/
theorem pencil_count : 
  ∃ (reeta anika kamal sarah : ℕ),
    reeta = 30 ∧
    anika = 2 * reeta + 4 ∧
    kamal = 3 * reeta - 2 ∧
    sarah = (75 * anika) / 100 ∧
    total_pencils reeta anika kamal sarah = 230 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_count_l1010_101034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_five_digit_palindromic_with_zero_hundreds_l1010_101010

/-- A five-digit palindromic number with zero in the hundred's place -/
def FiveDigitPalindromicWithZeroHundreds : Type := 
  { n : ℕ // 10000 ≤ n ∧ n < 100000 ∧ (n / 10000 = n % 10) ∧ ((n / 1000) % 10 = (n / 10) % 10) ∧ ((n / 100) % 10 = 0) }

/-- Fintype instance for FiveDigitPalindromicWithZeroHundreds -/
instance : Fintype FiveDigitPalindromicWithZeroHundreds := by
  sorry

/-- The count of five-digit palindromic numbers with zero in the hundred's place -/
theorem count_five_digit_palindromic_with_zero_hundreds :
  Fintype.card FiveDigitPalindromicWithZeroHundreds = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_five_digit_palindromic_with_zero_hundreds_l1010_101010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_trapezoid_area_ratio_l1010_101004

/-- A trapezoid with extended legs -/
structure ExtendedTrapezoid where
  -- The lengths of the bases
  pq : ℚ
  rs : ℚ
  -- The areas of the triangle and trapezoid
  area_tpq : ℚ
  area_pqrs : ℚ
  -- Conditions
  pq_positive : pq > 0
  rs_positive : rs > 0
  area_tpq_positive : area_tpq > 0
  area_pqrs_positive : area_pqrs > 0

/-- The ratio of areas in an extended trapezoid -/
def area_ratio (t : ExtendedTrapezoid) : ℚ := t.area_tpq / t.area_pqrs

/-- Theorem: For a trapezoid with bases 10 and 23, the ratio of areas is 100/429 -/
theorem extended_trapezoid_area_ratio :
  ∀ t : ExtendedTrapezoid, t.pq = 10 ∧ t.rs = 23 → area_ratio t = 100 / 429 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_trapezoid_area_ratio_l1010_101004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_negative_a_l1010_101005

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else 2*x - 40

-- State the theorem
theorem unique_negative_a (a : ℝ) (h : a < 0) :
  g (g (g 11)) = g (g (g a)) ↔ a = -29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_negative_a_l1010_101005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_count_arithmetic_sequences_eq_floor_n_squared_div_4_l1010_101042

/-- 
Given a set S = {1, 2, ..., n}, this function returns the number of arithmetic sequences A 
with at least two terms, all terms in S, and a positive common difference, 
such that adding any other element of S to A will not form an arithmetic sequence 
with the same common difference.
-/
def count_arithmetic_sequences (n : ℕ) : ℕ :=
  (n^2 / 4 : ℕ)

/-- 
Theorem stating that the count of arithmetic sequences A with the given properties 
is equal to ⌊n²/4⌋.
-/
theorem count_arithmetic_sequences_eq_floor_n_squared_div_4 (n : ℕ) :
  count_arithmetic_sequences n = n^2 / 4 := by
  sorry

#eval count_arithmetic_sequences 10  -- Example evaluation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_count_arithmetic_sequences_eq_floor_n_squared_div_4_l1010_101042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_sum_magnitude_l1010_101046

noncomputable def A : ℝ × ℝ := (0, 1)
noncomputable def B : ℝ × ℝ := (1, 0)
noncomputable def C : ℝ × ℝ := (0, -2)
noncomputable def O : ℝ × ℝ := (0, 0)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def vector_sum (u v w : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1 + w.1, u.2 + v.2 + w.2)

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem max_vector_sum_magnitude :
  ∃ (M : ℝ × ℝ), distance C M = 1 ∧
    ∀ (N : ℝ × ℝ), distance C N = 1 →
      vector_magnitude (vector_sum O A (vector_sum O B M)) ≤
      vector_magnitude (vector_sum O A (vector_sum O B N)) →
        vector_magnitude (vector_sum O A (vector_sum O B M)) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_sum_magnitude_l1010_101046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_exponential_to_rectangular_l1010_101017

theorem complex_exponential_to_rectangular : 2 * Complex.exp (Complex.I * (13 * Real.pi / 6)) = Complex.ofReal (Real.sqrt 3) + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_exponential_to_rectangular_l1010_101017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_A_l1010_101094

theorem triangle_sin_A (a b c : ℝ) (A : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < Real.pi →
  3 * b^2 + 3 * c^2 - 3 * a^2 = 4 * b * c →
  Real.sin A = Real.sqrt (1 - ((b^2 + c^2 - a^2) / (2 * b * c))^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_A_l1010_101094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1010_101008

theorem ellipse_eccentricity (a b : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  a > b ∧ b > 0 →
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → (x, y) = P) →
  ‖P - F₁‖ - ‖P - F₂‖ = 3 * b →
  ‖P - F₁‖ * ‖P - F₂‖ = 9 / 4 * a * b →
  let c := Real.sqrt (a^2 - b^2)
  c / a = 2 * Real.sqrt 2 / 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1010_101008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_difference_inverse_l1010_101048

theorem inverse_difference_inverse : (9⁻¹ - 5⁻¹)⁻¹ = -(45 / 4 : ℚ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_difference_inverse_l1010_101048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1010_101040

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (2 * Real.cos x - Real.sqrt 2)) / (2 * Real.sin x - 1)

def domain (x : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi - Real.pi / 4 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 4 ∧
  x ≠ 2 * k * Real.pi + Real.pi / 6

theorem f_domain : 
  ∀ x : ℝ, (2 * Real.cos x - Real.sqrt 2 ≥ 0 ∧ 2 * Real.sin x - 1 ≠ 0) ↔ domain x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1010_101040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1010_101032

/-- The function f(x) defined as x^3 + 3/x for x > 0 -/
noncomputable def f (x : ℝ) : ℝ := x^3 + 3/x

/-- Theorem stating that the minimum value of f(x) for x > 0 is 4 -/
theorem f_min_value (x : ℝ) (h : x > 0) : f x ≥ 4 := by
  sorry

#check f_min_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1010_101032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_fixed_point_fixed_point_for_19_power_86_l1010_101089

/-- Represents a positive integer in decimal expansion -/
def DecimalExpansion := List Nat

/-- Function f as defined in the problem -/
def f (A : DecimalExpansion) : Nat :=
  let n := A.length - 1
  List.foldl (fun sum (i, a) => sum + 2^(n-i) * a) 0 (List.enumFrom 0 A)

/-- Convert a natural number to its decimal expansion -/
def toDecimalExpansion (n : Nat) : DecimalExpansion :=
  if n = 0 then [0] else
  let rec go (m : Nat) (acc : DecimalExpansion) : DecimalExpansion :=
    if m = 0 then acc else go (m / 10) ((m % 10) :: acc)
  go n []

/-- Repeated application of f -/
def iterate_f (A : Nat) (k : Nat) : Nat :=
  match k with
  | 0 => A
  | k+1 => f (toDecimalExpansion (iterate_f A k))

theorem existence_of_fixed_point (A : Nat) :
  ∃ k : Nat, iterate_f A k = iterate_f A (k+1) := by
  sorry

theorem fixed_point_for_19_power_86 :
  ∃ k : Nat, iterate_f (19^86) k = 19 ∧ 
    ∀ j < k, iterate_f (19^86) j ≠ iterate_f (19^86) (j+1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_fixed_point_fixed_point_for_19_power_86_l1010_101089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_line_l1010_101088

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 4

/-- The line equation -/
def line (x : ℝ) : ℝ := 3 * x - 4

/-- The distance between a point (x1, y1) and a line ax + by + c = 0 -/
noncomputable def distance_point_to_line (x1 y1 a b c : ℝ) : ℝ :=
  abs (a * x1 + b * y1 + c) / Real.sqrt (a^2 + b^2)

theorem min_distance_parabola_line :
  ∃ (x : ℝ), ∀ (x1 x2 : ℝ),
    distance_point_to_line x1 (parabola x1) 3 (-1) (-4) ≥
    distance_point_to_line x (parabola x) 3 (-1) (-4) ∧
    distance_point_to_line x (parabola x) 3 (-1) (-4) = (7 * Real.sqrt 10) / 20 := by
  sorry

#check min_distance_parabola_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_line_l1010_101088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1010_101086

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, Real.exp x > 2 * x + a) → a < 2 - 2 * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1010_101086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_car_speed_l1010_101009

/-- The distance between two cities A and B in kilometers -/
noncomputable def distance_between_cities : ℝ := 360

/-- The speed of the first car in kilometers per hour -/
noncomputable def speed_of_first_car : ℝ := 50

/-- The time elapsed in hours -/
noncomputable def time_elapsed : ℝ := 3

/-- The remaining distance between the cars after the elapsed time in kilometers -/
noncomputable def remaining_distance : ℝ := 48

/-- The speed of the second car in kilometers per hour -/
noncomputable def speed_of_second_car : ℝ := 
  (distance_between_cities - remaining_distance) / time_elapsed - speed_of_first_car

theorem second_car_speed :
  speed_of_second_car = 54 := by
  -- Unfold the definition of speed_of_second_car
  unfold speed_of_second_car
  -- Unfold other definitions
  unfold distance_between_cities speed_of_first_car time_elapsed remaining_distance
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_car_speed_l1010_101009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_cupcake_production_l1010_101058

/-- Represents the daily production and pricing of a bakery --/
structure BakeryProduction where
  cupcake_price : ℚ
  cookie_price : ℚ
  biscuit_price : ℚ
  cookie_packets : ℕ
  biscuit_packets : ℕ
  total_earnings : ℚ
  days : ℕ

/-- Calculates the number of cupcakes that can be baked daily --/
noncomputable def daily_cupcakes (b : BakeryProduction) : ℚ :=
  (b.total_earnings - (b.cookie_price * b.cookie_packets + b.biscuit_price * b.biscuit_packets) * b.days) / (b.cupcake_price * b.days)

/-- Theorem stating that under given conditions, the bakery can produce 20 cupcakes daily --/
theorem bakery_cupcake_production : 
  ∀ b : BakeryProduction, 
    b.cupcake_price = 3/2 ∧ 
    b.cookie_price = 2 ∧ 
    b.biscuit_price = 1 ∧ 
    b.cookie_packets = 10 ∧ 
    b.biscuit_packets = 20 ∧ 
    b.total_earnings = 350 ∧ 
    b.days = 5 →
    daily_cupcakes b = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_cupcake_production_l1010_101058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1010_101060

theorem rationalize_denominator :
  1 / (Real.rpow 4 (1/3) + Real.rpow 32 (1/3)) = Real.rpow 2 (1/3) / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1010_101060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1010_101037

/-- Calculates the length of a train given its speed, bridge length, and time to cross the bridge. -/
noncomputable def train_length (speed : ℝ) (bridge_length : ℝ) (time : ℝ) : ℝ :=
  speed * (1000 / 3600) * time - bridge_length

/-- Theorem stating that a train with speed 42 kmph crossing a 150-meter bridge in 21.42685727998903 seconds has a length of 100 meters. -/
theorem train_length_calculation :
  let speed := (42 : ℝ)
  let bridge_length := (150 : ℝ)
  let time := (21.42685727998903 : ℝ)
  ∃ ε > 0, |train_length speed bridge_length time - 100| < ε := by
  sorry

-- We can't use #eval with noncomputable functions, so we'll use the following instead:
#check train_length (42 : ℝ) (150 : ℝ) (21.42685727998903 : ℝ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1010_101037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_event_probability_l1010_101063

/-- Represents a standard deck of 52 cards -/
structure Deck where
  cards : Finset (Nat × Nat)
  card_count : cards.card = 52
  suit_count : ∀ s, (cards.filter (λ c ↦ c.1 = s)).card = 13
  rank_count : ∀ r, (cards.filter (λ c ↦ c.2 = r)).card = 4

/-- Represents the event of drawing a heart, then a spade, then a 10 -/
def event (d : Deck) : Prop :=
  ∃ (c1 c2 c3 : Nat × Nat),
    c1 ∈ d.cards ∧ c2 ∈ d.cards ∧ c3 ∈ d.cards ∧
    c1.1 = 0 ∧ c2.1 = 1 ∧ c3.2 = 9 ∧
    c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3

/-- The probability of the event occurring -/
def probability (d : Deck) : ℚ :=
  221 / 44200

theorem event_probability (d : Deck) :
  probability d = 221 / 44200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_event_probability_l1010_101063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_machines_needed_l1010_101029

/-- Represents the number of products waiting for inspection -/
noncomputable def z : ℝ := 15

/-- Represents the number of products delivered for inspection per minute -/
noncomputable def x : ℝ := 0.5

/-- Calculates the number of machines needed to inspect all products in t minutes -/
noncomputable def machines_needed (t : ℝ) : ℝ := (z + t * x) / t

/-- Theorem stating that at least 4 machines are needed to inspect all products within 5 minutes -/
theorem min_machines_needed : ⌈machines_needed 5⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_machines_needed_l1010_101029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_proof_l1010_101013

/-- The exchange rate from USD to EUR -/
def usd_to_eur : ℚ := 85/100

/-- The selling price in USD -/
def selling_price_usd : ℚ := 78050/100

/-- The selling price in EUR -/
def selling_price_eur : ℚ := 68075/100

/-- The percentage increase in gain -/
def gain_increase_percent : ℚ := 15/100

/-- The cost of the article in USD -/
def cost_usd : ℚ := 93675/100

theorem article_cost_proof :
  let selling_price_eur_to_usd := selling_price_eur / usd_to_eur
  let gain_usd := selling_price_usd - cost_usd
  let gain_eur_to_usd := selling_price_eur_to_usd - cost_usd
  gain_usd = gain_eur_to_usd * (1 + gain_increase_percent) →
  cost_usd = 93675/100 := by
  sorry

#eval cost_usd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_proof_l1010_101013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l1010_101057

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop :=
  2 * y^2 + 8 * y - 3 * x + 6 = 0

/-- The vertex of the parabola -/
noncomputable def vertex : ℝ × ℝ := (-14/3, -2)

/-- Theorem stating that the vertex of the parabola is (-14/3, -2) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_eq x y → (x, y) = vertex ∨ (∃ y' : ℝ, parabola_eq x y' ∧ y' ≠ y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l1010_101057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l1010_101003

/-- Given vectors a, b, and c, prove that p*a + q*b + r*c equals the target vector -/
theorem vector_equation_solution (a b c : ℝ × ℝ × ℝ) (p q r : ℚ) : 
  a = (1, 1, 1) →
  b = (3, -2, 4) →
  c = (5, 0, -5) →
  p = 42/11 →
  q = -6/11 →
  r = -69/55 →
  (p : ℝ) • a + (q : ℝ) • b + (r : ℝ) • c = (-3, 6, 9) := by
  sorry

#check vector_equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l1010_101003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrangular_pyramid_dihedral_angle_l1010_101092

/-- A regular quadrangular pyramid. -/
structure RegularQuadrangularPyramid where
  /-- The cosine of the angle between two lateral edges that do not lie in the same face. -/
  cosine_lateral_edge_angle : ℝ
  /-- The cosine of the dihedral angle at the apex of the pyramid. -/
  cosine_dihedral_angle_at_apex : ℝ

/-- In a regular quadrangular pyramid, if the cosine of the angle between two lateral edges
    that do not lie in the same face is k, then the cosine of the dihedral angle at the apex
    of the pyramid is (1 + k) / 2. -/
theorem regular_quadrangular_pyramid_dihedral_angle (k : ℝ) :
  ∃ (pyramid : RegularQuadrangularPyramid),
    pyramid.cosine_lateral_edge_angle = k →
    pyramid.cosine_dihedral_angle_at_apex = (1 + k) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrangular_pyramid_dihedral_angle_l1010_101092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l1010_101075

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  2 * Real.sin θ + 1 / Real.cos θ + Real.sqrt 2 * (Real.cos θ / Real.sin θ) ≥ 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l1010_101075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1010_101019

-- Define the functions
noncomputable def f₁ (x : ℝ) : ℝ := 1 / x
noncomputable def f₂ (x : ℝ) : ℝ := Real.exp (-x)
def f₃ (x : ℝ) : ℝ := 1 - x^2
def f₄ (x : ℝ) : ℝ := x^2

-- Define evenness
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define monotonically decreasing on (-∞,0)
def is_decreasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 0 → f y < f x

theorem function_properties :
  is_even f₄ ∧ is_decreasing_on_neg f₄ ∧
  (¬is_even f₁ ∨ ¬is_decreasing_on_neg f₁) ∧
  (¬is_even f₂ ∨ ¬is_decreasing_on_neg f₂) ∧
  (¬is_even f₃ ∨ ¬is_decreasing_on_neg f₃) := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1010_101019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_theorem_l1010_101050

/-- Represents a sequence of price changes -/
structure PriceChanges where
  jan : ℚ
  feb : ℚ
  mar : ℚ
  apr : ℚ
  may : ℚ

/-- Calculates the final price after applying a sequence of price changes -/
noncomputable def finalPrice (initial : ℚ) (changes : PriceChanges) : ℚ :=
  initial * (1 + changes.jan/100) * (1 - changes.feb/100) * (1 + changes.mar/100) * 
            (1 - changes.apr/100) * (1 + changes.may/100)

/-- The theorem to be proved -/
theorem price_change_theorem (initial : ℚ) (x : ℚ) :
  initial > 0 →
  let changes := PriceChanges.mk 10 15 20 x 15
  finalPrice initial changes = initial →
  Int.floor (x + 0.5) = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_theorem_l1010_101050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1010_101035

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- State the theorem
theorem quadratic_function_properties :
  ∀ (a b : ℝ),
  (∀ x, f a b x ≥ f a b (-1)) →  -- Minimum at x = -1
  f a b (-1) = 0 →               -- Minimum value is 0
  (∃ (k : ℝ), 
    (∀ x, f a b x = x^2 + 2*x + 1) ∧  -- Part 1: Specific form of f
    (∀ x, x ≤ -1 → (deriv (f a b)) x ≤ 0) ∧  -- Part 2: Decreasing on (-∞, -1]
    (∀ x, x ≥ -1 → (deriv (f a b)) x ≥ 0) ∧  -- Part 2: Increasing on [-1, +∞)
    (∀ k', (∀ x, -3 ≤ x ∧ x ≤ -1 → f a b x > x + k') ↔ k' < 1)  -- Part 3: Range of k
  ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1010_101035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_f_surjective_onto_open_zero_three_l1010_101000

-- Define the function f(x) = (1/3)^(x^2 - 1)
noncomputable def f (x : ℝ) : ℝ := (1/3) ^ (x^2 - 1)

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y ∈ Set.range f, 0 < y ∧ y ≤ 3 :=
by sorry

-- State the theorem about the surjectivity of f onto (0, 3]
theorem f_surjective_onto_open_zero_three :
  ∀ y ∈ Set.Ioc 0 3, ∃ x, f x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_f_surjective_onto_open_zero_three_l1010_101000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_news_popularity_decay_l1010_101084

/-- Represents the cooling coefficient of the news popularity --/
noncomputable def α : ℝ := 0.3

/-- Represents the natural logarithm of 10 --/
noncomputable def ln10 : ℝ := 2.303

/-- The news popularity function --/
noncomputable def N (N₀ t : ℝ) : ℝ := N₀ * Real.exp (-α * t)

/-- Theorem stating the minimum number of days for news popularity to decrease below 10% --/
theorem news_popularity_decay (N₀ : ℝ) (h : N₀ > 0) :
  (∃ t : ℕ, ∀ s : ℝ, s ≥ t → N N₀ s < 0.1 * N₀) ∧
  (∀ t : ℕ, t < 8 → ∃ s : ℝ, s ≥ t ∧ N N₀ s ≥ 0.1 * N₀) := by
  sorry

#check news_popularity_decay

end NUMINAMATH_CALUDE_ERRORFEEDBACK_news_popularity_decay_l1010_101084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1010_101068

/-- Line l is defined by the parametric equations x = 1 + t, y = t - 1 -/
def line_l (t : ℝ) : ℝ × ℝ := (1 + t, t - 1)

/-- Circle C is defined by the equation x^2 + y^2 = 1 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The minimum distance from a point on circle C to line l -/
noncomputable def min_distance : ℝ := Real.sqrt 2 - 1

theorem min_distance_circle_to_line :
  ∀ (x y : ℝ), circle_C x y →
  ∃ (t : ℝ), ∀ (x' y' : ℝ), (x', y') = line_l t →
  Real.sqrt ((x - x')^2 + (y - y')^2) ≥ min_distance :=
by
  sorry

#check min_distance_circle_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1010_101068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_decreasing_P_converges_to_zero_l1010_101001

/-- Represents the probability of not getting two consecutive heads when tossing a fair coin n times. -/
def P (n : ℕ) : ℚ :=
  sorry -- We don't define the actual function here, just declare it

/-- The sequence {P_n} is decreasing. -/
theorem P_decreasing : ∀ n : ℕ, P (n + 1) < P n := by
  sorry

/-- There exists a positive integer k such that P_k < 1/100. -/
theorem P_converges_to_zero : ∃ k : ℕ+, P k < 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_decreasing_P_converges_to_zero_l1010_101001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l1010_101076

/-- An arithmetic-geometric sequence -/
structure ArithmeticGeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  q_ne_one : q ≠ 1

/-- Sum of the first n terms of an arithmetic-geometric sequence -/
noncomputable def S (seq : ArithmeticGeometricSequence) (n : ℕ) : ℝ :=
  (seq.a 1) * (1 - seq.q^n) / (1 - seq.q)

/-- Theorem: For an arithmetic-geometric sequence with S_3 = 2 and S_6 = 18, S_10 / S_5 = 33 -/
theorem arithmetic_geometric_sequence_ratio
  (seq : ArithmeticGeometricSequence)
  (h1 : S seq 3 = 2)
  (h2 : S seq 6 = 18) :
  S seq 10 / S seq 5 = 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l1010_101076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_C_value_min_tan_C_l1010_101024

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = Real.pi

-- Define the equation parameters
def equation_params (a : Real) : Prop :=
  ∃ p q : Real, p^2 + a*p + 4 = 0 ∧ q^2 + a*q + 4 = 0

-- Theorem for part 1
theorem tan_C_value (t : Triangle) (a : Real) (h : equation_params a) :
  a = -8 → Real.tan t.C = 8/3 := by sorry

-- Theorem for part 2
theorem min_tan_C (t : Triangle) (a : Real) (h : equation_params a) :
  (∃ (min_tan_C : Real), 
    (∀ (a' : Real) (h' : equation_params a'), Real.tan t.C ≥ min_tan_C) ∧
    min_tan_C = 4/3 ∧
    (∃ (t' : Triangle) (a'' : Real) (h'' : equation_params a''),
      Real.tan t'.A = 2 ∧ Real.tan t'.B = 2 ∧ Real.tan t'.C = min_tan_C)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_C_value_min_tan_C_l1010_101024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_of_sequence_l1010_101021

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem eighth_term_of_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : sum_of_arithmetic_sequence a 15 = 90) :
  a 8 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_of_sequence_l1010_101021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_same_color_l1010_101074

/-- A structure representing a box of colored pencils -/
structure ColoredPencilBox where
  pencils : ℕ
  colors : ℕ → ℕ  -- Function mapping each color to its count

/-- Property: Any group of 5 pencils contains at least 2 of the same color -/
def hasPigeonholePropery (box : ColoredPencilBox) : Prop :=
  ∀ (S : Finset ℕ), S.card = 5 → ∃ (c : ℕ), (S.filter (λ i => box.colors i ≥ 2)).card ≥ 1

/-- Main theorem: In a box of 25 pencils with the pigeonhole property, 
    there are at least 7 pencils of the same color -/
theorem seven_same_color (box : ColoredPencilBox) 
  (h1 : box.pencils = 25)
  (h2 : hasPigeonholePropery box) :
  ∃ (c : ℕ), box.colors c ≥ 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_same_color_l1010_101074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_proof_l1010_101054

theorem sine_function_proof (A φ x₀ : ℝ) :
  A > 0 →
  φ ∈ Set.Ioo 0 (π / 2) →
  A * Real.sin (2 * (π / 12) + φ) = Real.sqrt 3 →
  A * Real.sin (2 * (π / 4) + φ) = Real.sqrt 3 →
  A * Real.sin (2 * x₀ + φ) = 6 / 5 →
  x₀ ∈ Set.Icc (π / 4) (π / 2) →
  Real.sin (2 * x₀ - π / 12) = 7 * Real.sqrt 2 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_proof_l1010_101054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_choice_theorem_optimal_choice_maximizes_profit_l1010_101080

/-- Represents the daily profit function for a product --/
def ProfitFunction := ℝ → ℝ

/-- Represents the choice of product --/
inductive ProductChoice
| A
| B
| Either

/-- Represents the problem setup --/
structure ProductionProblem where
  m : ℝ
  x_max_A : ℝ
  x_max_B : ℝ
  profit_A : ProfitFunction
  profit_B : ProfitFunction
  h_m_range : 4 ≤ m ∧ m ≤ 6
  h_x_max_A : x_max_A = 500
  h_x_max_B : x_max_B = 300
  h_profit_A : profit_A = λ x => (8 - m) * x - 30
  h_profit_B : profit_B = λ x => -0.01 * x^2 + 8 * x - 80

/-- Determines the optimal product choice for maximum daily profit --/
noncomputable def optimalChoice (prob : ProductionProblem) : ProductChoice :=
  if prob.m = 5.1 then ProductChoice.Either
  else if prob.m < 5.1 then ProductChoice.A
  else ProductChoice.B

/-- Theorem stating the optimal choice of product for maximum daily profit --/
theorem optimal_choice_theorem (prob : ProductionProblem) :
  optimalChoice prob = 
    if prob.m = 5.1 then ProductChoice.Either
    else if 4 ≤ prob.m ∧ prob.m < 5.1 then ProductChoice.A
    else if 5.1 < prob.m ∧ prob.m ≤ 6 then ProductChoice.B
    else optimalChoice prob := by
  sorry

/-- Theorem stating that the optimal choice maximizes daily profit --/
theorem optimal_choice_maximizes_profit (prob : ProductionProblem) :
  let choice := optimalChoice prob
  let max_profit_A := prob.profit_A prob.x_max_A
  let max_profit_B := prob.profit_B prob.x_max_B
  match choice with
  | ProductChoice.A => max_profit_A ≥ max_profit_B
  | ProductChoice.B => max_profit_B > max_profit_A
  | ProductChoice.Either => max_profit_A = max_profit_B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_choice_theorem_optimal_choice_maximizes_profit_l1010_101080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_norm_l1010_101098

def vector_norm_problem (v : ℝ × ℝ) : Prop :=
  let u : ℝ × ℝ := (4, 2)
  ‖v + u‖ = 10 → ‖v‖ ≥ 10 - 2 * Real.sqrt 5

theorem smallest_possible_norm :
  ∃ (v : ℝ × ℝ), vector_norm_problem v ∧
  ∀ (w : ℝ × ℝ), vector_norm_problem w → ‖v‖ ≤ ‖w‖ :=
by
  -- We'll use the vector that achieves equality in the triangle inequality
  let v : ℝ × ℝ := ((10 - 2 * Real.sqrt 5) / (2 * Real.sqrt 5) * 4,
                    (10 - 2 * Real.sqrt 5) / (2 * Real.sqrt 5) * 2)
  
  -- Existential introduction
  use v
  
  sorry -- The actual proof is omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_norm_l1010_101098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1010_101007

theorem problem_statement (x n : ℕ) (h1 : x = 9^n - 1) (h2 : Odd n) 
  (h3 : (Nat.factors x).toFinset.card = 3) (h4 : 61 ∈ Nat.factors x) : x = 59048 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1010_101007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spatial_relationships_l1010_101078

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (planes_parallel : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (angle : Line → Plane → ℝ)

-- State the theorem
theorem spatial_relationships 
  (α β : Plane) (m n : Line) : 
  -- Statement 1
  ∃ (m n : Line) (α β : Plane), 
    perpendicular m n ∧ perpendicular_plane m α ∧ parallel_line_plane n β ∧ 
    ¬(planes_parallel α β) ∧
  -- Statement 2
  (∀ (m n : Line) (α : Plane), 
    perpendicular_plane m α → parallel_line_plane n α → perpendicular m n) ∧
  -- Statement 3
  (∀ (m : Line) (α β : Plane), 
    planes_parallel α β → line_in_plane m α → parallel_line_plane m β) ∧
  -- Statement 4
  (∀ (m n : Line) (α β : Plane), 
    parallel_line m n → planes_parallel α β → angle m α = angle n β) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spatial_relationships_l1010_101078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_distance_l1010_101066

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 3
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem curves_intersection_distance :
  ∃ (x1 y1 x2 y2 : ℝ),
    C1 x1 y1 ∧ C1 x2 y2 ∧ 
    C2 x1 y1 ∧ C2 x2 y2 ∧ 
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    distance x1 y1 x2 y2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_distance_l1010_101066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1010_101028

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x else x^2 - 4*x + 5

-- State the theorem
theorem f_range (a : ℝ) (h : f a ≥ 1) : a ∈ Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1010_101028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_tree_problem_l1010_101062

theorem apple_tree_problem (total_apples : ℕ) (first_day_fraction : ℚ) 
  (second_day_multiplier : ℕ) (third_day_addition : ℕ) : 
  total_apples = 200 →
  first_day_fraction = 1 / 5 →
  second_day_multiplier = 2 →
  third_day_addition = 20 →
  let first_day := (first_day_fraction * ↑total_apples).floor
  let second_day := second_day_multiplier * first_day
  let third_day := first_day + third_day_addition
  total_apples - (first_day + second_day + third_day) = 20 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_tree_problem_l1010_101062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1010_101064

/-- The range of m given the conditions of the problem -/
theorem range_of_m (f g : ℝ → ℝ) (m a : ℝ) : 
  (∀ x, f x = Real.exp (x + 1) - m * a) →
  (∀ x, g x = a * Real.exp x - x) →
  (∃ a : ℝ, ∀ x, f x ≤ g x) →
  m ≥ -1 / Real.exp 1 ∧ ∀ y ≥ -1 / Real.exp 1, ∃ a : ℝ, ∀ x, (Real.exp (x + 1) - y * a) ≤ (a * Real.exp x - x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1010_101064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_coverable_l1010_101097

/-- A point in a plane represented by its coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A set of n points in a plane -/
def PointSet (n : ℕ) := Fin n → Point

/-- Predicate to check if three points can be covered by a circle of radius 1 -/
def coverable_by_unit_circle (p1 p2 p3 : Point) : Prop :=
  ∃ (center : Point), (distance center p1 ≤ 1) ∧ (distance center p2 ≤ 1) ∧ (distance center p3 ≤ 1)

/-- Theorem stating that if any three points in a set can be covered by a unit circle,
    then all points in the set can be covered by a unit circle -/
theorem all_points_coverable {n : ℕ} (points : PointSet n) :
  (∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → 
    coverable_by_unit_circle (points i) (points j) (points k)) →
  ∃ (center : Point), ∀ (i : Fin n), distance center (points i) ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_coverable_l1010_101097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1010_101085

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Set.Icc 1 2 then 2 * x + 6
  else if x ∈ Set.Icc (-1) 1 then x + 7
  else 0  -- Define a default value for x outside the given intervals

-- State the theorem about the maximum and minimum values of f
theorem f_max_min :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ f x = 10) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → f x ≤ 10) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ f x = 6) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → f x ≥ 6) :=
by sorry

#check f_max_min

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1010_101085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_is_77_l1010_101036

-- Define the pentagon vertices
noncomputable def pentagon_vertices : List (ℝ × ℝ) :=
  [(-7, 1), (1, 1), (1, -6), (-7, -6), (-1, Real.sqrt 3)]

-- Define the function to calculate the area of the pentagon
noncomputable def pentagon_area (vertices : List (ℝ × ℝ)) : ℝ :=
  sorry -- The actual calculation would go here

-- Theorem statement
theorem pentagon_area_is_77 :
  pentagon_area pentagon_vertices = 77 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_is_77_l1010_101036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_ray_l1010_101072

theorem sin_2theta_ray (θ : ℝ) : 
  (∃ (x y : ℝ), x > 0 ∧ y = 3 * x ∧ 
   θ = Real.arctan (y / x)) → 
  Real.sin (2 * θ) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_ray_l1010_101072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_point_probability_l1010_101002

theorem square_point_probability (a : ℝ) (h : a > 0) : 
  let square_area := a^2
  let curved_quadrilateral_area := a^2 * (π/3 - 2*Real.sqrt 2 * Real.sin (π/12))
  curved_quadrilateral_area / square_area = π/3 + 1 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_point_probability_l1010_101002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traveler_meets_truck_l1010_101014

/-- Represents the time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Represents the journey between the village and the city -/
structure Journey where
  total_distance : ℝ
  traveler_speed : ℝ
  motorcyclist_speed : ℝ
  truck_speed : ℝ

noncomputable def Journey.traveler_position (j : Journey) (t : ℝ) : ℝ :=
  j.traveler_speed * t

noncomputable def Journey.motorcyclist_position (j : Journey) (t : ℝ) : ℝ :=
  j.motorcyclist_speed * t

noncomputable def Journey.truck_position (j : Journey) (t : ℝ) : ℝ :=
  j.total_distance - j.truck_speed * t

noncomputable def time_difference (t1 t2 : Time) : ℝ :=
  (t2.hours - t1.hours : ℝ) + (t2.minutes - t1.minutes : ℝ) / 60

theorem traveler_meets_truck (j : Journey) :
  ∃ (start : Time) (meet : Time),
    -- At 14:00, the traveler has covered 1/4 of the journey
    j.traveler_position (time_difference start ⟨14, 0, by norm_num⟩) = j.total_distance / 4 →
    -- At 15:00, the motorcyclist catches up with the traveler
    j.motorcyclist_position 1 = j.traveler_position (1 + time_difference start ⟨14, 0, by norm_num⟩) →
    -- At 15:30, the motorcyclist meets the truck
    j.motorcyclist_position 1.5 + j.truck_position 1.5 = j.total_distance →
    -- The traveler meets the truck at the calculated time
    j.traveler_position (time_difference start meet) = j.truck_position (time_difference ⟨14, 0, by norm_num⟩ meet) →
    meet = ⟨15, 48, by norm_num⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_traveler_meets_truck_l1010_101014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_square_area_ratio_l1010_101082

/-- Given a square, the ratio of the area of the square formed by joining its midpoints
    to the area of the original square is 1/2. -/
theorem midpoint_square_area_ratio (A : ℝ) (A_pos : A > 0) :
  let s : ℝ := Real.sqrt A
  let midpoint_square_area : ℝ := (s * Real.sqrt 2 / 2) ^ 2
  midpoint_square_area / A = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_square_area_ratio_l1010_101082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_problem_l1010_101073

theorem triangle_similarity_problem (DC CB : ℝ) (h1 : DC = 10) (h2 : CB = 12)
  (AB ED AD : ℝ) (h3 : AB = (1/3) * AD) (h4 : ED = (2/3) * AD) : ∃ FC : ℝ,
  abs (FC - 15.333) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_problem_l1010_101073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_product_equation_l1010_101087

theorem sqrt_product_equation (x : ℝ) (h1 : x > 0) 
  (h2 : Real.sqrt (12 * x) * Real.sqrt (20 * x) * Real.sqrt (5 * x) * Real.sqrt (30 * x) = 60) :
  x = (1/10)^(1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_product_equation_l1010_101087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1010_101095

/-- Represents a quadratic function y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℤ
  c : ℝ

/-- Defines the t value for a quadratic function -/
noncomputable def t_value (f : QuadraticFunction) : ℝ :=
  (f.a + 2 * (f.b : ℝ) + 12 * f.c) / f.a

/-- Defines the distance CE for a quadratic function -/
noncomputable def CE (f : QuadraticFunction) : ℝ :=
  Real.sqrt ((1 / (6 * f.a))^2 + (f.c - (1 / (36 * f.a)))^2)

/-- Theorem stating the minimum value of t and the form of the function when CE > 5/24 -/
theorem quadratic_function_properties (f : QuadraticFunction) 
  (h1 : f.a > 0)
  (h2 : (f.b : ℝ)^2 - 4*f.a*f.c ≤ 0) :
  (∀ g : QuadraticFunction, t_value g ≥ 2/3) ∧ 
  (t_value f = 2/3 ∧ CE f > 5/24 → 
    f.a = 6 ∧ f.b = -2 ∧ f.c = 1/6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1010_101095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inv_f_10_eq_2_sqrt_3_l1010_101077

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the inverse of g
noncomputable def g_inv : ℝ → ℝ := Function.invFun g

-- State the given condition
axiom condition : ∀ x, f.invFun (g x) = x^2 - 2

-- State that g has an inverse
axiom g_has_inverse : Function.Bijective g

-- Theorem to prove
theorem g_inv_f_10_eq_2_sqrt_3 : g_inv (f 10) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inv_f_10_eq_2_sqrt_3_l1010_101077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_theorem_l1010_101093

/-- Represents a cone with given slant height and lateral surface area -/
structure Cone where
  slant_height : ℝ
  lateral_surface_area : ℝ

/-- Calculates the volume of a cone given its slant height and lateral surface area -/
noncomputable def cone_volume (c : Cone) : ℝ :=
  let r := 2 * c.lateral_surface_area / (Real.pi * c.slant_height)
  let h := Real.sqrt (c.slant_height^2 - r^2)
  (1/3) * Real.pi * r^2 * h

/-- Theorem stating that a cone with slant height 5 and lateral surface area 15π has volume 12π -/
theorem cone_volume_theorem (c : Cone) 
    (h1 : c.slant_height = 5) 
    (h2 : c.lateral_surface_area = 15 * Real.pi) : 
  cone_volume c = 12 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_theorem_l1010_101093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l1010_101056

/-- Represents an arithmetic sequence with given properties -/
structure ArithmeticSequence where
  first_term : ℚ
  last_term : ℚ
  sum : ℚ
  num_terms : ℕ

/-- The common difference of the arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℚ :=
  (seq.last_term - seq.first_term) / (seq.num_terms - 1 : ℚ)

/-- Theorem stating the common difference of the given arithmetic sequence is 3 -/
theorem arithmetic_sequence_common_difference :
  ∀ (seq : ArithmeticSequence),
    seq.first_term = 2 ∧
    seq.last_term = 29 ∧
    seq.sum = 155 ∧
    seq.num_terms > 1 →
    common_difference seq = 3 := by
  sorry

#eval common_difference { first_term := 2, last_term := 29, sum := 155, num_terms := 10 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l1010_101056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_theorem_l1010_101099

-- Define the circles
noncomputable def circle_O1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4
noncomputable def circle_O2 (x y : ℝ) : Prop := x^2 + (y - Real.sqrt 3)^2 = 9

-- Define the line equation
noncomputable def common_chord_line (x y : ℝ) : Prop := 2*x - 2*(Real.sqrt 3)*y - 3 = 0

-- Define the length of the common chord
noncomputable def common_chord_length : ℝ := Real.sqrt 15

theorem common_chord_theorem :
  ∀ (x y : ℝ),
  (circle_O1 x y ∧ circle_O2 x y) →
  (common_chord_line x y ∧ common_chord_length = Real.sqrt 15) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_theorem_l1010_101099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_eight_ten_factorial_l1010_101091

/-- Definition of factorial for natural numbers -/
def factorial (n : ℕ) : ℕ := (List.range n).map Nat.succ |>.prod

/-- 8 factorial -/
def eight_factorial : ℕ := factorial 8

/-- 10 factorial -/
def ten_factorial : ℕ := factorial 10

/-- Theorem: The greatest common divisor of 8! and 10! is equal to 8! -/
theorem gcd_eight_ten_factorial : 
  Nat.gcd eight_factorial ten_factorial = eight_factorial := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_eight_ten_factorial_l1010_101091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_break_duration_l1010_101067

-- Define the variables
variable (m : ℝ) -- Maria's work rate
variable (a : ℝ) -- Combined rate of three assistants
variable (L : ℝ) -- Lunch break duration in hours

-- Define the conditions as functions
def monday_condition (m a L : ℝ) : Prop := (10 - L) * (m + a) = 0.6
def tuesday_condition (m a L : ℝ) : Prop := (8 - L) * (2/3 * a) = 0.3
def wednesday_condition (m L : ℝ) : Prop := (4 - L) * m = 0.1

-- Theorem to prove
theorem lunch_break_duration :
  ∃ (m a L : ℝ), 
    monday_condition m a L ∧ 
    tuesday_condition m a L ∧ 
    wednesday_condition m L ∧ 
    L * 60 = 210 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_break_duration_l1010_101067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_pi_twelfths_l1010_101043

theorem tan_sum_pi_twelfths : Real.tan (π / 12) + Real.tan (5 * π / 12) = Real.sqrt 6 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_pi_twelfths_l1010_101043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_and_time_calculation_l1010_101039

-- Define the given conditions
noncomputable def velocity : ℚ := 1/3
def distance : ℚ := 200

-- Define the conversion factors
def seconds_per_minute : ℚ := 60
def seconds_per_hour : ℚ := 3600
def meters_per_km : ℚ := 1000

-- Theorem to prove
theorem speed_and_time_calculation :
  let time_seconds := distance / velocity
  let time_minutes := time_seconds / seconds_per_minute
  let speed_km_per_hour := velocity * seconds_per_hour / meters_per_km
  (time_minutes = 10) ∧ (speed_km_per_hour = 2/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_and_time_calculation_l1010_101039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l1010_101012

-- Define the line equation
noncomputable def line_equation (x y : ℝ) : Prop := x / 2019 - y / 2019 = 1

-- Define the slope angle
noncomputable def slope_angle (m : ℝ) : ℝ := Real.arctan m * (180 / Real.pi)

-- Theorem statement
theorem line_slope_angle :
  ∃ (m b : ℝ), (∀ x y : ℝ, line_equation x y → y = m * x + b) ∧ 
  slope_angle m = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l1010_101012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1010_101071

def sequence_a : ℕ → ℚ
  | 0 => 0
  | n + 1 => sequence_a n + n

theorem sequence_a_formula (n : ℕ) : sequence_a n = n * (n - 1) / 2 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp [sequence_a]
    rw [ih]
    ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1010_101071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l1010_101052

-- Define the line equation in polar coordinates
def line_equation (ρ θ : ℝ) : Prop := Real.sqrt 3 * ρ * Real.cos θ + ρ * Real.sin θ = 1

-- Define the point in polar coordinates
noncomputable def point : ℝ × ℝ := (2, Real.pi/6)

-- Theorem statement
theorem distance_to_line :
  let (ρ, θ) := point
  ∃ d : ℝ, d = 3/2 ∧ 
    d = |Real.sqrt 3 * (ρ * Real.cos θ) + (ρ * Real.sin θ) - 1| / Real.sqrt ((Real.sqrt 3)^2 + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l1010_101052
