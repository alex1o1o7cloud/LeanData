import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_value_l1164_116489

-- Define the logarithm function as noncomputable
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_base_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  log a 9 = 2 → a = 3 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_value_l1164_116489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1164_116466

noncomputable section

open Real

-- Define the function f
def f (ω : ℝ) (x : ℝ) : ℝ := 3 * sin (ω * x + π / 6)

-- State the theorem
theorem function_properties (ω : ℝ) (h1 : ω > 0) (h2 : π / 2 = 2 * π / ω) :
  -- 1. f(0) = 3/2
  f ω 0 = 3 / 2 ∧
  -- 2. f(x) = 3sin(4x + π/6)
  (∀ x, f ω x = 3 * sin (4 * x + π / 6)) ∧
  -- 3. If f(α/4 + π/12) = 9/5, then sinα = ±4/5
  (∀ α, f ω (α / 4 + π / 12) = 9 / 5 → sin α = 4 / 5 ∨ sin α = -4 / 5) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1164_116466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_17_over_70_l1164_116467

theorem digit_150_of_17_over_70 : 
  let decimal_expansion := (17 : ℚ) / 70
  let digit_150 := (decimal_expansion * 10^150).floor % 10
  digit_150 = 7 := by 
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_17_over_70_l1164_116467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solutions_l1164_116407

noncomputable def log_equation (x : ℝ) : ℝ :=
  (Real.log x^2) / (Real.log (x/2)) - 14 * (Real.log x^3) / (Real.log (16*x)) + 40 * (Real.log (Real.sqrt x)) / (Real.log (4*x))

theorem log_equation_solutions :
  ∀ x : ℝ, x > 0 ∧ x ≠ 1/16 ∧ x ≠ 1/4 ∧ x ≠ 2 →
    log_equation x = 0 ↔ x = 1 ∨ x = 1/Real.sqrt 2 ∨ x = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solutions_l1164_116407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l1164_116446

open Set Real

theorem intersection_A_complement_B : 
  let A : Set ℝ := {x | (2 : ℝ)^x > 1}
  let B : Set ℝ := {x | log x > 1}
  A ∩ (univ \ B) = {x | 0 < x ∧ x ≤ (exp 1)} := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l1164_116446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1164_116452

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2*x + 1) / (x - 1)

-- Theorem statement
theorem f_properties :
  -- There exists exactly one x such that f(x) = 0
  (∃! x, f x = 0) ∧
  -- The range of f is all real numbers except 2
  (∀ y : ℝ, y ≠ 2 → ∃ x, f x = y) ∧
  (∀ x, f x ≠ 2) ∧
  -- Symmetry property about the point (1,2)
  (∀ a b : ℝ, f a = b → f (2 - a) = 4 - b) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1164_116452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_special_case_l1164_116406

theorem sin_half_angle_special_case (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_special_case_l1164_116406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_with_inscribed_ellipse_and_rectangle_l1164_116457

/-- A rectangle is inscribed in an ellipse -/
def Rectangle_inscribed_in_ellipse (width height : ℝ) : Prop :=
  sorry

/-- An ellipse is inscribed in a circle -/
def Ellipse_inscribed_in_circle : Prop :=
  sorry

/-- The diagonal of the rectangle is the major axis of the ellipse -/
def Diagonal_is_major_axis (width height : ℝ) : Prop :=
  sorry

/-- The major axis of the ellipse is the diameter of the circle -/
def Major_axis_is_circle_diameter : Prop :=
  sorry

/-- The circumference of the circle -/
noncomputable def circle_circumference : ℝ :=
  sorry

/-- The circumference of a circle containing an inscribed ellipse with an inscribed rectangle -/
theorem circle_circumference_with_inscribed_ellipse_and_rectangle 
  (rectangle_width : ℝ) 
  (rectangle_height : ℝ) 
  (h_rectangle_width : rectangle_width = 10)
  (h_rectangle_height : rectangle_height = 24)
  (h_inscribed_in_ellipse : Rectangle_inscribed_in_ellipse rectangle_width rectangle_height)
  (h_ellipse_inscribed_in_circle : Ellipse_inscribed_in_circle)
  (h_diagonal_is_major_axis : Diagonal_is_major_axis rectangle_width rectangle_height)
  (h_major_axis_is_diameter : Major_axis_is_circle_diameter) :
  circle_circumference = 26 * π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_with_inscribed_ellipse_and_rectangle_l1164_116457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_fraction_l1164_116479

theorem last_digit_of_fraction : ∃ (n : ℕ), n < 10 ∧ 
  (∃ (m : ℚ), (1 : ℚ) / (2^12 * 3) = m + (n : ℚ) / 10) ∧ 
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_fraction_l1164_116479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l1164_116461

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmeticSum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem no_solution_exists : ¬ ∃ (n : ℕ), n > 0 ∧
  (arithmeticSum 5 6 n) * (arithmeticSum 12 4 n) = 24 * (n : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l1164_116461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1164_116426

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the distance function between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define the theorem
theorem min_distance_sum :
  ∀ x' y' : ℝ,
  parabola x' y' →
  12 ≤ distance 2 0 x' y' + distance 8 6 x' y' :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1164_116426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_three_l1164_116443

-- Define the positive real numbers
def PositiveReals := {x : ℝ | x > 0}

-- Define the function type
def FunctionType := ℝ → ℝ

-- Define the functional equation
def SatisfiesFunctionalEquation (f : FunctionType) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f x + f y = f x * f y + 1 - 1 / (x * y)

theorem function_value_at_three
  (f : FunctionType)
  (h1 : SatisfiesFunctionalEquation f)
  (h2 : f 2 < 1) :
  f 3 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_three_l1164_116443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_square_two_terms_no_perfect_square_three_terms_l1164_116422

-- Part 1
theorem no_perfect_square_two_terms :
  ¬ ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ ∀ (n : ℕ), ∃ (x : ℕ), x^2 = 2^n * a + 5^n * b := by
  sorry

-- Part 2
theorem no_perfect_square_three_terms :
  ¬ ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ ∀ (n : ℕ), ∃ (x : ℕ), x^2 = 2^n * a + 5^n * b + c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_square_two_terms_no_perfect_square_three_terms_l1164_116422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_l1164_116414

/-- The time it takes for two people to complete a job together, given their individual completion times -/
noncomputable def combined_completion_time (time_a time_b : ℝ) : ℝ :=
  1 / (1 / time_a + 1 / time_b)

/-- Theorem: If A can complete a job in 6 days and B can complete it in 12 days, 
    then A and B working together will complete the job in 4 days -/
theorem combined_work_time (time_a time_b : ℝ) 
  (ha : time_a = 6) 
  (hb : time_b = 12) : 
  combined_completion_time time_a time_b = 4 := by
  sorry

#check combined_work_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_l1164_116414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_primitive_lattice_triangles_l1164_116448

/-- A lattice polygon is a simple polygon with vertices at lattice points. -/
def LatticePolygon (P : Set (ℝ × ℝ)) : Prop := sorry

/-- A primitive lattice triangle is a lattice triangle with no lattice points inside or on its sides except for the three vertices. -/
def PrimitiveLatticeTri (T : Set (ℝ × ℝ)) : Prop := sorry

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

theorem hexagon_primitive_lattice_triangles :
  ∀ H : Set (ℝ × ℝ),
  LatticePolygon H →
  (∃ (v1 v2 v3 v4 v5 v6 : ℝ × ℝ), H = {v1, v2, v3, v4, v5, v6}) →
  area H = 18 →
  ∃ (n : ℕ) (partition : Fin n → Set (ℝ × ℝ)),
    (∀ i : Fin n, PrimitiveLatticeTri (partition i)) ∧
    (∀ x : ℝ × ℝ, x ∈ H ↔ ∃ i : Fin n, x ∈ partition i) ∧
    n = 36 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_primitive_lattice_triangles_l1164_116448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l1164_116437

theorem evaluate_expression : (64 : ℝ) ^ (1/2 : ℝ) * (27 : ℝ) ^ (-1/3 : ℝ) * (16 : ℝ) ^ (1/4 : ℝ) = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l1164_116437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equidistant_and_passes_through_intersection_l1164_116432

-- Define the points A and B
def A : ℝ × ℝ := (3, 3)
def B : ℝ × ℝ := (5, 2)

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := 3 * x - y - 1 = 0
def l2 (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the intersection point of l1 and l2
def intersection : ℝ × ℝ := (1, 2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the line l
def l (x y : ℝ) : Prop := x + 2 * y - 5 = 0 ∨ x - 6 * y + 11 = 0

-- State the theorem
theorem line_l_equidistant_and_passes_through_intersection :
  (∀ (x y : ℝ), l x y → distance (x, y) A = distance (x, y) B) ∧
  l intersection.1 intersection.2 ∧
  l1 intersection.1 intersection.2 ∧
  l2 intersection.1 intersection.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equidistant_and_passes_through_intersection_l1164_116432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_symmetric_l1164_116483

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 4 * x^2) - 2 * x) + 1

-- State the theorem
theorem f_sum_symmetric : f 3 + f (-3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_symmetric_l1164_116483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_line_l1164_116412

/-- The distance from the origin to a line ax + by + c = 0 is |c| / √(a² + b²) -/
noncomputable def distanceOriginToLine (a b c : ℝ) : ℝ := |c| / Real.sqrt (a^2 + b^2)

/-- The line 4x + 3y - 15 = 0 -/
def line : ℝ → ℝ → Prop :=
  fun x y => 4 * x + 3 * y - 15 = 0

theorem distance_origin_to_line :
  distanceOriginToLine 4 3 (-15) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_line_l1164_116412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_digit_integers_is_4536_count_is_correct_l1164_116418

/-- The number of four-digit integers with all digits different -/
def count_distinct_digit_integers : ℕ :=
  9 * 9 * 8 * 7

/-- A four-digit integer -/
def four_digit_integer : Type := {n : ℕ // 1000 ≤ n ∧ n ≤ 9999}

/-- Predicate for a four-digit integer with all digits different -/
def has_distinct_digits (n : four_digit_integer) : Prop :=
  ∀ i j, i ≠ j → (n.val / 10^i) % 10 ≠ (n.val / 10^j) % 10

/-- The main theorem -/
theorem count_distinct_digit_integers_is_4536 :
  ∃ (n : four_digit_integer), n.val = count_distinct_digit_integers ∧
  (∀ m : four_digit_integer, has_distinct_digits m → m.val ≤ n.val) :=
by sorry

/-- Auxiliary theorem: The count is correct -/
theorem count_is_correct :
  (∃ (n : four_digit_integer), n.val = count_distinct_digit_integers) ∧
  (∀ n : four_digit_integer, has_distinct_digits n → n.val ≤ count_distinct_digit_integers) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_digit_integers_is_4536_count_is_correct_l1164_116418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1164_116491

noncomputable def f (x : ℝ) := (Real.log (x + 1)) / Real.sqrt (-x^2 - 3*x + 4)

theorem domain_of_f :
  {x : ℝ | x + 1 > 0 ∧ -x^2 - 3*x + 4 ≥ 0} = Set.Ioo (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1164_116491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1164_116462

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define the side lengths as functions of angles
noncomputable def BC (A B C AB AC : ℝ) : ℝ :=
  Real.sqrt (AB^2 + AC^2 - 2 * AB * AC * Real.cos C)

noncomputable def AC (A B C AB BC : ℝ) : ℝ :=
  Real.sqrt (AB^2 + BC^2 - 2 * AB * BC * Real.cos A)

noncomputable def AB (A B C BC AC : ℝ) : ℝ :=
  Real.sqrt (BC^2 + AC^2 - 2 * BC * AC * Real.cos B)

-- Define the theorem
theorem triangle_ABC_properties
  (A B C : ℝ)
  (h_triangle : triangle_ABC A B C)
  (h_BC : Real.sqrt 5 = BC A B C (AB A B C (Real.sqrt 5) 3) 3)
  (h_AC : 3 = AC A B C (AB A B C (Real.sqrt 5) 3) (Real.sqrt 5))
  (h_sin_C : Real.sin C = 2 * Real.sin A)
  : AB A B C (Real.sqrt 5) 3 = 2 * Real.sqrt 5 ∧
    Real.sin (2 * A - Real.pi / 4) = Real.sqrt 2 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1164_116462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_normal_line_equation_l1164_116477

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := 6 * x^(1/3) - (16/3) * x^(1/4)

-- Define the point of interest
def a : ℝ := 1

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∀ x y : ℝ, y = f a + (deriv f a) * (x - a) ↔ 2*x - 3*y = 0 := by sorry

-- Theorem for the normal line equation
theorem normal_line_equation :
  ∀ x y : ℝ, y = f a - (1 / deriv f a) * (x - a) ↔ 9*x + 6*y - 13 = 0 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_normal_line_equation_l1164_116477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1164_116469

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_properties :
  ∃ (period : ℝ),
    (∀ (x : ℝ), f (x + period) = f x) ∧
    (∀ (p : ℝ), (∀ (x : ℝ), f (x + p) = f x) → p ≥ period) ∧
    period = Real.pi ∧
    (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ 2) ∧
    (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -1) ∧
    (∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc 0 (Real.pi / 2) ∧ x₂ ∈ Set.Icc 0 (Real.pi / 2) ∧ f x₁ = 2 ∧ f x₂ = -1) ∧
    (∀ (x₀ : ℝ), x₀ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → f x₀ = 6/5 → Real.cos (2 * x₀) = (3 - 4 * Real.sqrt 3) / 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1164_116469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_range_l1164_116499

/-- The function f(x) defined in terms of a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x * (a^x - 3*a^2 - 1)

/-- Theorem stating the range of a for which f is increasing on [0, +∞) -/
theorem f_increasing_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x < y → f a x < f a y) →
  Real.sqrt 3 / 3 ≤ a ∧ a < 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_range_l1164_116499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worst_competitor_is_sister_l1164_116463

-- Define the competitors
inductive Competitor
| Man
| Wife
| Son
| Sister

-- Define the gender
inductive Gender
| Male
| Female

-- Define the generation
inductive Generation
| Older
| Younger

-- Define the functions
def gender : Competitor → Gender := sorry
def generation : Competitor → Generation := sorry
def sibling : Competitor → Option Competitor := sorry
def is_worst : Competitor → Prop := sorry
def is_best : Competitor → Prop := sorry

-- Define the axioms
axiom different_genders : 
  ∀ (c : Competitor), is_worst c → 
    ∀ (s : Competitor), sibling c = some s → 
      ∀ (b : Competitor), is_best b → gender s ≠ gender b

axiom different_generations : 
  ∀ (w b : Competitor), is_worst w → is_best b → generation w ≠ generation b

axiom man_sister_siblings : sibling Competitor.Sister = some Competitor.Man

axiom wife_no_sibling : sibling Competitor.Wife = none

axiom son_no_sibling : sibling Competitor.Son = none

axiom man_wife_same_generation : 
  generation Competitor.Man = generation Competitor.Wife

axiom sister_different_generation : 
  generation Competitor.Sister ≠ generation Competitor.Wife

-- Theorem to prove
theorem worst_competitor_is_sister : is_worst Competitor.Sister := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worst_competitor_is_sister_l1164_116463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_painted_cells_l1164_116420

/-- Represents a rectangle in a grid --/
structure Rectangle where
  top_left : Nat × Nat
  bottom_right : Nat × Nat

/-- Checks if a rectangle is within the bounds of the grid --/
def Rectangle.isValid (r : Rectangle) (n : Nat) : Prop :=
  r.top_left.1 ≤ r.bottom_right.1 ∧ r.top_left.2 ≤ r.bottom_right.2 ∧
  r.bottom_right.1 ≤ n ∧ r.bottom_right.2 ≤ n

/-- Represents the grid of natural numbers --/
def Grid (n : Nat) := Fin n → Fin n → Nat

/-- Checks if a rectangle is "good" (sum of numbers inside is divisible by 17) --/
def Rectangle.isGood (r : Rectangle) (grid : Grid 100) : Prop :=
  ∃ k : Nat, 17 * k = (Finset.sum (Finset.range (r.bottom_right.1 - r.top_left.1 + 1)) (λ i =>
    Finset.sum (Finset.range (r.bottom_right.2 - r.top_left.2 + 1)) (λ j =>
      grid ⟨i + r.top_left.1, sorry⟩ ⟨j + r.top_left.2, sorry⟩)))

/-- The main theorem --/
theorem max_painted_cells :
  ∀ (grid : Grid 100),
  ∃ (rectangles : List Rectangle),
    (∀ r ∈ rectangles, r.isValid 100 ∧ r.isGood grid) ∧
    (∀ i j, (∃ r ∈ rectangles, i ∈ Finset.range (r.bottom_right.1 - r.top_left.1 + 1) ∧
                               j ∈ Finset.range (r.bottom_right.2 - r.top_left.2 + 1)) →
            (∀ r' ∈ rectangles, r' ≠ r →
              i ∉ Finset.range (r'.bottom_right.1 - r'.top_left.1 + 1) ∨
              j ∉ Finset.range (r'.bottom_right.2 - r'.top_left.2 + 1))) ∧
    (Finset.card (Finset.biUnion (Finset.range rectangles.length) (λ i =>
      let r := rectangles.get ⟨i, sorry⟩
      Finset.product
        (Finset.range (r.bottom_right.1 - r.top_left.1 + 1))
        (Finset.range (r.bottom_right.2 - r.top_left.2 + 1)))) ≥ 10000 - 256) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_painted_cells_l1164_116420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_m_value_l1164_116435

/-- A power function f(x) = (2m^2 + m)x^m that is monotonically increasing on [0, +∞) -/
noncomputable def power_function (m : ℝ) : ℝ → ℝ := fun x ↦ (2 * m^2 + m) * x^m

/-- The power function is monotonically increasing on [0, +∞) -/
def is_monotone_increasing (m : ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x < y → power_function m x < power_function m y

theorem power_function_m_value :
  ∃ m : ℝ, is_monotone_increasing m ∧ m = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_m_value_l1164_116435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_icosahedron_colorings_l1164_116444

/-- The number of faces in a regular icosahedron -/
def num_faces : ℕ := 20

/-- The number of different colors available -/
def num_colors : ℕ := 10

/-- The order of rotational symmetry around a vertex of a regular icosahedron -/
def rotational_symmetry : ℕ := 5

/-- The number of distinguishable colorings of a regular icosahedron -/
def distinguishable_colorings : ℕ := (num_colors - 1).factorial / rotational_symmetry

theorem icosahedron_colorings :
  distinguishable_colorings = 72576 :=
sorry

#eval distinguishable_colorings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_icosahedron_colorings_l1164_116444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_positive_real_solutions_l1164_116402

theorem product_of_positive_real_solutions : ∃ (S : Finset ℂ), 
  (∀ z ∈ S, z^8 = -256 ∧ z.re > 0) ∧ 
  (∀ z : ℂ, z^8 = -256 ∧ z.re > 0 → z ∈ S) ∧
  (S.prod id = 16) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_positive_real_solutions_l1164_116402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l1164_116400

theorem circular_table_seating (n m : ℕ) (hn : n = 8) (hm : m = 6) :
  (n.choose m) * (Nat.factorial (m - 1)) = 3360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l1164_116400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1164_116459

theorem trigonometric_identities (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.cos (α + π / 6) = 1 / 3) : 
  Real.sin α = (2 * Real.sqrt 6 + 1) / 6 ∧ 
  Real.sin (2 * α + 5 * π / 6) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1164_116459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1164_116408

def a : ℝ × ℝ := (1, -3)
def b : ℝ × ℝ := (4, -2)

theorem perpendicular_vectors (l : ℝ) :
  (l • a.1 + b.1, l • a.2 + b.2) • a = 0 → l = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1164_116408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1164_116474

theorem trigonometric_identities (α : Real) 
  (h1 : Real.sin α = 1/3) 
  (h2 : 0 < α ∧ α < Real.pi/2) : 
  Real.cos (2*α) = 7/9 ∧ 
  Real.sin (2*α + Real.pi/3) = (4*Real.sqrt 2 + 7*Real.sqrt 3)/18 ∧ 
  Real.tan (2*α) = 4*Real.sqrt 2/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1164_116474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_l1164_116441

/-- Predicate to check if a given real number is the slope of a line --/
def is_slope (m : ℝ) (line : Set (ℝ × ℝ)) : Prop :=
  ∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ line → (x₂, y₂) ∈ line → x₁ ≠ x₂ → 
    m = (y₂ - y₁) / (x₂ - x₁)

/-- Predicate to check if a given angle is the angle of inclination of a line --/
def is_angle_of_inclination (α : ℝ) (line : Set (ℝ × ℝ)) : Prop :=
  ∃ m : ℝ, is_slope m line ∧ α = Real.arctan m

/-- The angle of inclination of the line x + √3y + a = 0 is 5π/6 --/
theorem angle_of_inclination (a : ℝ) : 
  let line := {(x, y) : ℝ × ℝ | x + Real.sqrt 3 * y + a = 0}
  ∃ α : ℝ, α = 5 * Real.pi / 6 ∧ is_angle_of_inclination α line :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_l1164_116441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_intersection_exists_l1164_116415

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Determines if a point lies on a circle -/
def pointOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  distance p c.center = c.radius

/-- Determines if a point lies on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  (p.2 - l.point1.2) * (l.point2.1 - l.point1.1) = 
  (p.1 - l.point1.1) * (l.point2.2 - l.point1.2)

theorem equal_distance_intersection_exists 
  (c1 c2 : Circle) (p : ℝ × ℝ) : 
  ∃ (l : Line) (a1 a2 : ℝ × ℝ), 
    pointOnLine p l ∧ 
    pointOnLine a1 l ∧ 
    pointOnLine a2 l ∧ 
    pointOnCircle a1 c1 ∧ 
    pointOnCircle a2 c2 ∧ 
    distance p a1 = distance p a2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_intersection_exists_l1164_116415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_real_l1164_116465

/-- The function f(x) parameterized by a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt ((1 - a^2) * x^2 + 3 * (1 - a) * x + 6)

/-- The theorem stating the condition for f to have domain ℝ -/
theorem f_domain_real (a : ℝ) : 
  (∀ x, ∃ y, f a x = y) ↔ a ∈ Set.Ioc (-5/11) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_real_l1164_116465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_with_digit_sum_property_l1164_116421

/-- Sum of digits of a natural number in decimal representation -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem arithmetic_progression_with_digit_sum_property (M : ℝ) :
  ∃ (a₀ d : ℕ+), 
    ¬(10 ∣ d) ∧ 
    ∀ (n : ℕ), (sum_of_digits (a₀ + n * d) : ℝ) > M :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_with_digit_sum_property_l1164_116421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_greater_cos_range_l1164_116468

theorem sin_greater_cos_range (x : ℝ) :
  x ∈ Set.Ioo 0 (2 * Real.pi) →
  (Real.sin x > Real.cos x) ↔ x ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_greater_cos_range_l1164_116468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_postage_count_l1164_116401

/-- Represents an envelope with length and height in inches -/
structure Envelope where
  length : ℚ
  height : ℚ

/-- Determines if an envelope requires extra postage -/
def requiresExtraPostage (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 1.5 || ratio > 2.8

/-- The set of envelopes -/
def envelopes : List Envelope := [
  ⟨7, 5⟩,  -- Envelope A
  ⟨10, 3⟩, -- Envelope B
  ⟨7, 7⟩,  -- Envelope C
  ⟨12, 4⟩  -- Envelope D
]

/-- Theorem: The number of envelopes requiring extra postage is 4 -/
theorem extra_postage_count : 
  (envelopes.filter requiresExtraPostage).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_postage_count_l1164_116401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l1164_116449

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  F₁.1 = -Real.sqrt 2 ∧ F₁.2 = 0 ∧ F₂.1 = Real.sqrt 2 ∧ F₂.2 = 0

-- Define a point on the ellipse
def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

-- Define the right angle condition
def right_angle (F₁ P F₂ : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1)*(P.1 - F₂.1) + (P.2 - F₁.2)*(P.2 - F₂.2) = 0

-- Define Q as the intersection of PF₂ extended with the ellipse
def Q_intersection (P F₂ Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ Q.1 = P.1 + t*(F₂.1 - P.1) ∧ Q.2 = P.2 + t*(F₂.2 - P.2) ∧ on_ellipse Q

-- The main theorem
theorem ellipse_focal_distance (F₁ F₂ P Q : ℝ × ℝ) :
  foci F₁ F₂ →
  on_ellipse P →
  right_angle F₁ P F₂ →
  Q_intersection P F₂ Q →
  Real.sqrt ((F₁.1 - Q.1)^2 + (F₁.2 - Q.2)^2) = 10/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l1164_116449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_transformations_l1164_116403

open Real

/-- An angle is in the fourth quadrant if it's between 3π/2 and 2π radians -/
def is_fourth_quadrant (α : ℝ) : Prop := 3 * π / 2 < α ∧ α < 2 * π

/-- An angle is in the third quadrant if it's between π and 3π/2 radians -/
def is_third_quadrant (α : ℝ) : Prop := π < α ∧ α < 3 * π / 2

/-- An angle is in the second quadrant if it's between π/2 and π radians -/
def is_second_quadrant (α : ℝ) : Prop := π / 2 < α ∧ α < π

theorem angle_transformations (α : ℝ) (h : is_fourth_quadrant α) :
  is_third_quadrant (π - α) ∧ is_second_quadrant (π / 2 - α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_transformations_l1164_116403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_triangle_area_l1164_116413

/-- Regular octagon with side length 3 -/
structure RegularOctagon where
  side_length : ℝ
  is_regular : side_length = 3

/-- Triangle formed by three vertices of a regular octagon -/
noncomputable def OctagonTriangle (octagon : RegularOctagon) : ℝ := 
  13.5 * Real.sqrt 3 + 6.75 * Real.sqrt 6

/-- The area of a triangle formed by three vertices of a regular octagon
    with side length 3 is equal to 13.5√3 + 6.75√6 -/
theorem octagon_triangle_area (octagon : RegularOctagon) : 
  OctagonTriangle octagon = 13.5 * Real.sqrt 3 + 6.75 * Real.sqrt 6 := by
  sorry

#check octagon_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_triangle_area_l1164_116413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1164_116473

def inequality (x : ℝ) := 
  (Real.sin x) ^ 2018 + (Real.cos x) ^ (-2019 : ℤ) ≥ (Real.cos x) ^ 2018 + (Real.sin x) ^ (-2019 : ℤ)

def solution_set (x : ℝ) :=
  (x ∈ Set.Icc (-Real.pi / 4) 0) ∨
  (x ∈ Set.Ico (Real.pi / 4) (Real.pi / 2)) ∨
  (x ∈ Set.Ioc Real.pi (5 * Real.pi / 4)) ∨
  (x ∈ Set.Ioo (3 * Real.pi / 2) (7 * Real.pi / 4))

theorem inequality_solution :
  ∀ x ∈ Set.Icc (-Real.pi / 4) (7 * Real.pi / 4), 
    inequality x ↔ solution_set x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1164_116473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_two_sevenths_l1164_116442

/-- The area of the triangle bounded by the y-axis and two lines -/
noncomputable def triangleArea : ℝ :=
  let line1 : ℝ → ℝ := fun x => 2 * x + 4
  let line2 : ℝ → ℝ := fun x => (1 / 4) * x + 5
  let xIntersect : ℝ := 4 / 7
  let yIntersect : ℝ := line1 xIntersect
  let base : ℝ := 5 - 4
  let height : ℝ := xIntersect
  (1 / 2) * base * height

/-- Theorem stating that the area of the triangle is 2/7 -/
theorem triangle_area_is_two_sevenths : triangleArea = 2 / 7 := by
  -- Unfold the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_two_sevenths_l1164_116442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unobserved_planet_exists_l1164_116425

/-- A type representing a planet -/
def Planet : Type := ℕ

/-- A function representing the distance between two planets -/
def distance : Planet → Planet → ℝ := sorry

/-- A function representing the closest planet to a given planet -/
def closest_planet (p : Planet) : Planet := sorry

theorem unobserved_planet_exists 
  (total_planets : Finset Planet)
  (h1 : total_planets.card = 15)
  (h2 : ∀ p1 p2 : Planet, p1 ≠ p2 → distance p1 p2 ≠ distance p1 (closest_planet p1))
  (h3 : ∀ p : Planet, p ∈ total_planets → closest_planet p ∈ total_planets)
  (h4 : ∀ p : Planet, p ∈ total_planets → closest_planet p ≠ p) :
  ∃ p : Planet, p ∈ total_planets ∧ ∀ q : Planet, q ∈ total_planets → closest_planet q ≠ p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unobserved_planet_exists_l1164_116425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_equality_l1164_116471

theorem factorial_sum_equality (w x y z : ℕ) :
  w > 0 → x > 0 → y > 0 → z > 0 →
  Nat.factorial w = Nat.factorial x + Nat.factorial y + Nat.factorial z →
  (w, x, y, z) = (3, 2, 2, 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_equality_l1164_116471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_l1164_116495

noncomputable def scooter_A_purchase : ℚ := 4700
noncomputable def scooter_A_repair : ℚ := 600
noncomputable def scooter_A_sell : ℚ := 5800

noncomputable def scooter_B_purchase : ℚ := 3500
noncomputable def scooter_B_repair : ℚ := 800
noncomputable def scooter_B_sell : ℚ := 4800

noncomputable def scooter_C_purchase : ℚ := 5400
noncomputable def scooter_C_repair : ℚ := 1000
noncomputable def scooter_C_sell : ℚ := 7000

noncomputable def total_cost : ℚ := scooter_A_purchase + scooter_A_repair + 
                      scooter_B_purchase + scooter_B_repair + 
                      scooter_C_purchase + scooter_C_repair

noncomputable def total_sell : ℚ := scooter_A_sell + scooter_B_sell + scooter_C_sell

noncomputable def total_gain : ℚ := total_sell - total_cost

noncomputable def gain_percentage : ℚ := (total_gain / total_cost) * 100

theorem overall_gain_percentage : gain_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_l1164_116495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_valid_x_l1164_116496

/-- Definition of the sequence relation -/
def sequence_relation (a b c : ℝ) : Prop :=
  b = a * c - 1

/-- Definition of a valid sequence starting with x and 3000 -/
def valid_sequence (x : ℝ) : Prop :=
  x > 0 ∧ ∃ (y : ℝ), y > 0 ∧ sequence_relation x 3000 y

/-- Definition of a sequence containing 3001 -/
def contains_3001 (x : ℝ) : Prop :=
  ∃ (seq : ℕ → ℝ),
    seq 0 = x ∧ seq 1 = 3000 ∧
    (∀ i, sequence_relation (seq i) (seq (i+1)) (seq (i+2))) ∧
    (∃ k, seq k = 3001)

/-- The main theorem -/
theorem four_valid_x :
  ∃ (S : Finset ℝ), (∀ x ∈ S, valid_sequence x ∧ contains_3001 x) ∧ S.card = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_valid_x_l1164_116496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_undefined_sum_l1164_116480

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line2D where
  slope : Option ℝ
  yIntercept : Option ℝ

/-- Create a line from two points -/
noncomputable def lineFromPoints (p1 p2 : Point2D) : Line2D :=
  if p1.x = p2.x then
    { slope := none, yIntercept := none }
  else
    { slope := some ((p2.y - p1.y) / (p2.x - p1.x)),
      yIntercept := some (p1.y - (p2.y - p1.y) / (p2.x - p1.x) * p1.x) }

/-- Sum of slope and y-intercept of a line -/
def slopeYInterceptSum (l : Line2D) : Option ℝ :=
  match l.slope, l.yIntercept with
  | some s, some y => some (s + y)
  | _, _ => none

theorem vertical_line_undefined_sum (p1 p2 : Point2D) (h1 : p1.x = p2.x) (h2 : p1.y ≠ p2.y) :
  slopeYInterceptSum (lineFromPoints p1 p2) = none := by
  sorry

#check vertical_line_undefined_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_undefined_sum_l1164_116480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_count_l1164_116487

/-- A quadrilateral type -/
inductive QuadrilateralType
  | Square
  | RectangleNotSquare
  | KiteNotRhombus
  | GeneralQuadrilateral
  | EquilateralTrapezoidNotParallelogram

/-- Function indicating whether a quadrilateral type has a point equidistant from all corners -/
def has_equidistant_point (q : QuadrilateralType) : Bool :=
  match q with
  | .Square => true
  | .RectangleNotSquare => true
  | .KiteNotRhombus => false
  | .GeneralQuadrilateral => false
  | .EquilateralTrapezoidNotParallelogram => true

/-- The list of all quadrilateral types -/
def all_quadrilaterals : List QuadrilateralType :=
  [QuadrilateralType.Square, QuadrilateralType.RectangleNotSquare, 
   QuadrilateralType.KiteNotRhombus, QuadrilateralType.GeneralQuadrilateral, 
   QuadrilateralType.EquilateralTrapezoidNotParallelogram]

theorem equidistant_point_count :
  (all_quadrilaterals.filter has_equidistant_point).length = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_count_l1164_116487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_average_l1164_116440

/-- Calculates the overall average marks per student given the number of students and mean marks for each section. -/
def overallAverage (students : List Nat) (marks : List Rat) : Rat :=
  if students.length = marks.length ∧ students.length > 0 then
    (List.sum (List.zipWith (fun s m => (s : Rat) * m) students marks)) / (List.sum students : Rat)
  else
    0

/-- The theorem states that for the given data, the overall average is approximately 59.55 -/
theorem chemistry_average : 
  let students := [55, 48, 62, 39, 50, 45, 58, 53]
  let marks := [52, 68, 47, 75, 63, 58, 71, 49]
  abs (overallAverage students marks - 59.55) < 0.01 := by
  sorry

#eval overallAverage [55, 48, 62, 39, 50, 45, 58, 53] [52, 68, 47, 75, 63, 58, 71, 49]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_average_l1164_116440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_l1164_116470

-- Define the original circle
noncomputable def original_circle : Set (ℝ × ℝ) :=
  {p | (p.1^2 + p.2^2) = 36}

-- Define point P
def P : ℝ × ℝ := (8, 0)

-- Define the homothety factor
def homothety_factor : ℚ := 1/3

-- Define the locus of midpoint M
noncomputable def locus_M : Set (ℝ × ℝ) :=
  {m | ∃ q ∈ original_circle,
       m.1 = (homothety_factor : ℝ) * (2 * P.1 + q.1) ∧
       m.2 = (homothety_factor : ℝ) * (2 * P.2 + q.2)}

-- Theorem statement
theorem locus_is_circle :
  locus_M = {m | (m.1 - 16/3)^2 + m.2^2 = 4} := by
  sorry

#check locus_is_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_l1164_116470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_animal_difference_l1164_116498

def zoo_problem (zebras camels monkeys parrots giraffes : ℤ) : Prop :=
  zebras = 12 ∧
  camels = zebras / 2 ∧
  monkeys = 4 * camels ∧
  parrots = 2 * monkeys - 5 ∧
  giraffes = 3 * parrots + 1

theorem zoo_animal_difference :
  ∀ zebras camels monkeys parrots giraffes : ℤ,
  zoo_problem zebras camels monkeys parrots giraffes →
  monkeys - giraffes = -106 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_animal_difference_l1164_116498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_negative_five_l1164_116454

theorem complex_expression_equals_negative_five :
  (-1/3)⁻¹ - Real.sqrt 12 + 3 * Real.tan (30 * π / 180) - (π - Real.sqrt 3)^0 + abs (1 - Real.sqrt 3) = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_negative_five_l1164_116454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_in_triangle_l1164_116450

theorem cosine_inequality_in_triangle (α β γ : ℝ) (h : α + β + γ = Real.pi) :
  Real.cos (2 * α) + Real.cos (2 * β) - Real.cos (2 * γ) ≤ (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_in_triangle_l1164_116450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_special_angles_l1164_116482

theorem tan_sum_special_angles (α β : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.1^2 + A.2^2 = 1) ∧ 
    (B.1^2 + B.2^2 = 1) ∧
    (A.2 = Real.sqrt 2 * A.1) ∧ 
    (B.2 = Real.sqrt 2 * B.1) ∧
    (α = Real.arctan (A.2 / A.1)) ∧
    (β = Real.arctan (B.2 / B.1)) ∧
    (0 ≤ α) ∧ (α < β) ∧ (β < π)) →
  Real.tan (α + β) = -2 * Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_special_angles_l1164_116482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tissue_diameter_calculation_l1164_116494

/-- Given a magnification factor and the diameter of a magnified image,
    calculate the actual diameter of the object. -/
noncomputable def actual_diameter (magnification : ℝ) (magnified_diameter : ℝ) : ℝ :=
  magnified_diameter / magnification

/-- Theorem stating that for a magnification of 1000 and a magnified diameter of 5 cm,
    the actual diameter is 0.005 cm. -/
theorem tissue_diameter_calculation :
  let magnification : ℝ := 1000
  let magnified_diameter : ℝ := 5
  actual_diameter magnification magnified_diameter = 0.005 := by
  -- Unfold the definition of actual_diameter
  unfold actual_diameter
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tissue_diameter_calculation_l1164_116494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_domain_range_sum_l1164_116492

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * (x - 1)^2 + a

theorem quadratic_function_domain_range_sum (a b : ℝ) :
  b > 1 →
  (∀ x, x ∈ Set.Icc 1 b ↔ f a x ∈ Set.Icc 1 b) →
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_domain_range_sum_l1164_116492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_pairs_properties_l1164_116428

def Digits : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def DifferentPairs : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 ∈ Digits ∧ p.2 ∈ Digits ∧ p.1 ≠ p.2) (Finset.product Digits Digits)

theorem digit_pairs_properties :
  (∀ p ∈ DifferentPairs, p.1 * p.2 ≤ 72) ∧
  (∃ p ∈ DifferentPairs, p.1 * p.2 = 72) ∧
  (∀ p ∈ DifferentPairs, p.1 + p.2 ≥ 1) ∧
  (∃ p ∈ DifferentPairs, p.1 + p.2 = 1) ∧
  (Finset.filter (fun p => p.1 + p.2 = 10) DifferentPairs).card = 4 :=
by sorry

#check digit_pairs_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_pairs_properties_l1164_116428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_random_events_l1164_116423

-- Define the type for events
inductive Event
| DiceRoll
| Rain
| Lottery
| SumGreaterThanTwo
| WaterBoil

-- Define the property of being a random event
def is_random (e : Event) : Bool :=
  match e with
  | Event.DiceRoll => true
  | Event.Rain => true
  | Event.Lottery => true
  | Event.SumGreaterThanTwo => false
  | Event.WaterBoil => false

-- Define the list of all events
def all_events : List Event :=
  [Event.DiceRoll, Event.Rain, Event.Lottery, Event.SumGreaterThanTwo, Event.WaterBoil]

-- State the theorem
theorem count_random_events :
  (all_events.filter is_random).length = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_random_events_l1164_116423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_at_chord_l1164_116409

noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  (|a * x₀ + b * y₀ + c|) / Real.sqrt (a^2 + b^2)

noncomputable def chord_length (r d : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - d^2)

theorem line_intersects_circle_at_chord (m : ℝ) : 
  (distance_point_to_line 1 0 1 (-1) m = (|1 + m|) / Real.sqrt 2) →
  (chord_length (Real.sqrt 5) ((|1 + m|) / Real.sqrt 2) = 2 * Real.sqrt 3) →
  m = 1 ∨ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_at_chord_l1164_116409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_square_rectangle_ratio_l1164_116445

/-- Represents a checkerboard with n x n squares --/
structure Checkerboard where
  size : ℕ
  horizontal_lines : ℕ
  vertical_lines : ℕ

/-- Calculates the number of rectangles in a checkerboard --/
def num_rectangles (board : Checkerboard) : ℕ :=
  (board.horizontal_lines.choose 2) * (board.vertical_lines.choose 2)

/-- Calculates the number of squares in a checkerboard --/
def num_squares (board : Checkerboard) : ℕ :=
  (board.size * (board.size + 1) * (2 * board.size + 1)) / 6

/-- The main theorem to prove --/
theorem checkerboard_square_rectangle_ratio 
  (board : Checkerboard) 
  (h1 : board.size = 6)
  (h2 : board.horizontal_lines = 7) 
  (h3 : board.vertical_lines = 7) : 
  (num_squares board : ℚ) / (num_rectangles board) = 1 / 7 := by
  sorry

#eval num_squares (Checkerboard.mk 6 7 7)
#eval num_rectangles (Checkerboard.mk 6 7 7)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_square_rectangle_ratio_l1164_116445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solutions_l1164_116439

theorem cosine_equation_solutions :
  ∀ x : ℝ, (Real.cos x)^2 + (Real.cos (2*x))^2 + (Real.cos (3*x))^2 = 1 ↔ 
  (∃ k : ℤ, x = Real.pi/2 + 2*k*Real.pi ∨ x = -Real.pi/2 + 2*k*Real.pi) ∨
  (∃ l : ℤ, x = Real.pi/4 + l*Real.pi/2 ∨ x = -Real.pi/4 + l*Real.pi/2) ∨
  (∃ m : ℤ, x = Real.pi/6 + m*Real.pi ∨ x = -Real.pi/6 + m*Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solutions_l1164_116439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_unplowed_cells_l1164_116453

/-- 
A plowing strategy is considered valid if:
1. It uses exactly k tractors
2. Each tractor starts at the lower-left corner (0, 0)
3. Each tractor ends at the upper-right corner (n-1, n-1)
4. Each tractor only moves up or right
-/
structure PlowingStrategy (n k : ℕ) where
  paths : Fin k → List (ℕ × ℕ)
  valid : Bool
  unplowed_cells : ℕ

theorem min_unplowed_cells (n k : ℕ) (h1 : n > k) (h2 : k > 0) :
  let grid_size := n * n
  let tractors := k
  let unplowed := (n - k) * (n - k)
  (∀ (plowing_strategy : PlowingStrategy n k), 
    plowing_strategy.valid → 
    plowing_strategy.unplowed_cells ≥ unplowed) ∧
  (∃ (optimal_strategy : PlowingStrategy n k), 
    optimal_strategy.valid ∧ 
    optimal_strategy.unplowed_cells = unplowed) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_unplowed_cells_l1164_116453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_geq_b_l1164_116493

open Real

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem a_geq_b (m n : ℕ) (x : ℝ) 
  (h_m_gt_n : m > n) 
  (h_n_pos : n > 0) 
  (h_x_gt_1 : x > 1) :
  (lg x)^(m : ℝ) + (lg x)^(-(m : ℝ)) ≥ (lg x)^(n : ℝ) + (lg x)^(-(n : ℝ)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_geq_b_l1164_116493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_plane_l1164_116490

/-- Given a line l with direction vector (-2, 1, t) and a plane α with normal vector (4, -2, -2),
    if l is perpendicular to α, then t = 1. -/
theorem perpendicular_line_plane (t : ℝ) : 
  let a : Fin 3 → ℝ := ![(-2 : ℝ), 1, t]
  let m : Fin 3 → ℝ := ![4, -2, -2]
  (∀ (x : Fin 3 → ℝ), x • a = 0 → x • m = 0) → t = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_plane_l1164_116490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_eval_neg_five_l1164_116481

/-- A polynomial that satisfies the given remainder conditions -/
noncomputable def p : Polynomial ℚ := sorry

/-- The remainder when p is divided by (X + 1) -/
axiom p_rem_x_plus_1 : p %ₘ (Polynomial.X + 1 : Polynomial ℚ) = 2

/-- The remainder when p is divided by (X - 3) -/
axiom p_rem_x_minus_3 : p %ₘ (Polynomial.X - 3 : Polynomial ℚ) = -4

/-- The remainder when p is divided by (X + 4) -/
axiom p_rem_x_plus_4 : p %ₘ (Polynomial.X + 4 : Polynomial ℚ) = 5

/-- The remainder when p is divided by (X + 1)(X - 3)(X + 4) -/
noncomputable def r : Polynomial ℚ := 
  p %ₘ ((Polynomial.X + 1) * (Polynomial.X - 3) * (Polynomial.X + 4))

theorem r_eval_neg_five : 
  r.eval (-5) = 154 / 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_eval_neg_five_l1164_116481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_cannot_cover_ground_l1164_116460

noncomputable def interior_angle (n : ℕ) : ℝ := 180 - (360 / n)

def divides_360 (angle : ℝ) : Prop := ∃ k : ℕ, 360 = k * angle

theorem pentagon_cannot_cover_ground :
  ¬(divides_360 (interior_angle 5)) ∧
  (divides_360 (interior_angle 4)) ∧
  (divides_360 (interior_angle 6)) ∧
  (divides_360 (interior_angle 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_cannot_cover_ground_l1164_116460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_triangle_side_range_l1164_116451

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.sin (2 * x - Real.pi / 6)

theorem function_properties_and_triangle_side_range 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : f A = 3/2) 
  (h2 : b + c = 2) 
  (h3 : 0 < A ∧ A < Real.pi) 
  (h4 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h5 : a = ((b^2 + c^2 - 2*b*c*Real.cos A)^(1/2))) :
  (∀ x, f x ≤ 2) ∧ (1 ≤ a ∧ a < 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_triangle_side_range_l1164_116451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l1164_116434

noncomputable section

-- Define the functions for Set 1
def f1 (x : ℝ) : ℝ := (Real.sqrt (1 - x^2)) / |x + 2|
def g1 (x : ℝ) : ℝ := (Real.sqrt (1 - x^2)) / (x + 2)

-- Define the functions for Set 2
def f2 (x : ℝ) : ℝ := Real.sqrt (x - 1) * Real.sqrt (x - 2)
def g2 (x : ℝ) : ℝ := Real.sqrt (x^2 - 3*x + 2)

end noncomputable

-- Define the domains
def domain1 : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def domain2f : Set ℝ := {x | x ≥ 2}
def domain2g : Set ℝ := {x | x ≤ 1 ∨ x ≥ 2}

theorem function_equivalence :
  (∀ x ∈ domain1, f1 x = g1 x) ∧
  (∃ x ∈ domain2f, f2 x ≠ g2 x ∨ x ∉ domain2g) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l1164_116434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longer_train_length_is_500m_l1164_116484

/-- Calculates the length of the longer train given the speeds of two trains,
    the time they take to cross each other, and the length of the shorter train. -/
noncomputable def longer_train_length (v1 v2 : ℝ) (t : ℝ) (l1 : ℝ) : ℝ :=
  (t * (v1 + v2) * (5/18)) - l1

/-- Theorem stating that under the given conditions, the length of the longer train
    is approximately 500 meters. -/
theorem longer_train_length_is_500m :
  let v1 : ℝ := 60  -- speed of first train in km/hr
  let v2 : ℝ := 40  -- speed of second train in km/hr
  let t : ℝ := 26.99784017278618  -- time to cross in seconds
  let l1 : ℝ := 250  -- length of shorter train in meters
  abs (longer_train_length v1 v2 t l1 - 500) < 1e-6 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longer_train_length_is_500m_l1164_116484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_of_g_7_l1164_116464

-- Define the functions u and g
noncomputable def u (x : ℝ) : ℝ := Real.sqrt (5 * x + 2)
noncomputable def g (x : ℝ) : ℝ := 7 - u x

-- State the theorem
theorem u_of_g_7 : u (g 7) = Real.sqrt (37 - 5 * Real.sqrt 37) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_of_g_7_l1164_116464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grade11PaperCuttingSurveyCorrect_l1164_116486

/-- The number of students to be selected from the Grade 11 "Paper Cutting" club for the survey -/
def grade11PaperCuttingSurvey (totalStudents : ℕ) (claySculptureRatio : ℚ) 
  (paperCuttingRatio : List ℚ) (sampleSize : ℕ) : ℕ :=
  let paperCuttingStudents := totalStudents - (claySculptureRatio * ↑totalStudents).floor
  let grade11Ratio := paperCuttingRatio[1]! / (paperCuttingRatio.sum)
  (grade11Ratio * (↑sampleSize * ↑paperCuttingStudents / ↑totalStudents)).floor.toNat

/-- Main theorem -/
theorem grade11PaperCuttingSurveyCorrect : 
  grade11PaperCuttingSurvey 800 (3/5) [5, 3, 2] 50 = 6 := by
  sorry

#eval grade11PaperCuttingSurvey 800 (3/5) [5, 3, 2] 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grade11PaperCuttingSurveyCorrect_l1164_116486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_inequality_l1164_116431

theorem fermat_inequality (a b c n : ℕ) (h1 : n > 1) (h2 : a^n + b^n = c^n) : 
  a > n ∧ b > n ∧ c > n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_inequality_l1164_116431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_8_l1164_116436

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_8 :
  ∀ (d : ℝ),
  (arithmetic_sequence 8 d 4 + arithmetic_sequence 8 d 6 = 0) →
  (arithmetic_sum 8 d 8 = 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_8_l1164_116436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1164_116458

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  (Real.cos B = 4/5) →
  (b = 2) →
  -- Part I
  (A = Real.pi/6) →
  (a = 5/3) ∧
  -- Part II
  (1/2 * a * c * Real.sin B = 3) →
  (a + c = 2 * Real.sqrt 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1164_116458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_order_l1164_116472

-- Define the constants
noncomputable def a : ℝ := 2^(1/10 : ℝ)
noncomputable def b : ℝ := (1/2)^(-(2/5 : ℝ))
noncomputable def c : ℝ := 2 * (Real.log 2 / Real.log 7)

-- State the theorem
theorem a_b_c_order : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_order_l1164_116472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1164_116475

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.A - t.B/4 = t.B/4 - t.C ∧
  t.b = Real.sqrt 13 ∧
  t.a = 3

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.c = 1 ∧ 1/4 < Real.sin t.A * Real.sin t.C ∧ Real.sin t.A * Real.sin t.C ≤ 1/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1164_116475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_theorem_satisfying_function_multiplicative_satisfying_function_nonnegative_satisfying_function_division_property_l1164_116417

/-- A totally multiplicative function from ℤ to ℤ. -/
def TotallyMultiplicative (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (a * b) = f a * f b

/-- The division algorithm property for function f. -/
def DivisionAlgorithmProperty (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, b ≠ 0 → ∃ q r : ℤ, a = q * b + r ∧ f r < f b

/-- Main theorem: Characterization of functions satisfying given properties. -/
theorem characterization_theorem (f : ℤ → ℤ) 
  (h_mult : TotallyMultiplicative f)
  (h_nonneg : ∀ n : ℤ, 0 ≤ f n)
  (h_div : DivisionAlgorithmProperty f) :
  ∃ c : ℕ+, ∀ n : ℤ, f n = |n| ^ (c : ℕ) :=
sorry

/-- The function satisfying the properties. -/
def satisfying_function (c : ℕ+) (n : ℤ) : ℤ :=
  |n| ^ (c : ℕ)

/-- Proof that the satisfying_function is totally multiplicative. -/
theorem satisfying_function_multiplicative (c : ℕ+) :
  TotallyMultiplicative (satisfying_function c) :=
sorry

/-- Proof that the satisfying_function is non-negative. -/
theorem satisfying_function_nonnegative (c : ℕ+) :
  ∀ n : ℤ, 0 ≤ satisfying_function c n :=
sorry

/-- Proof that the satisfying_function satisfies the division algorithm property. -/
theorem satisfying_function_division_property (c : ℕ+) :
  DivisionAlgorithmProperty (satisfying_function c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_theorem_satisfying_function_multiplicative_satisfying_function_nonnegative_satisfying_function_division_property_l1164_116417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1164_116416

theorem problem_solution (a b : ℝ) 
  (h1 : (100 : ℝ)^a = 4) 
  (h2 : (100 : ℝ)^b = 10) : 
  (25 : ℝ)^((2 - 2*a - b)/(3*(1 - b))) = 3.968 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1164_116416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_difference_l1164_116427

noncomputable def a (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

noncomputable def b : ℝ × ℝ := (Real.sqrt 3, 1)

theorem max_vector_difference :
  ∃ (θ : ℝ), ∀ (φ : ℝ), ‖a θ - b‖ ≥ ‖a φ - b‖ ∧ ‖a θ - b‖ = 3 := by
  sorry

#check max_vector_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_difference_l1164_116427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_2x_leq_1_l1164_116419

theorem negation_of_sin_2x_leq_1 :
  (¬ ∀ x : ℝ, Real.sin (2 * x) ≤ 1) ↔ (∃ x₀ : ℝ, Real.sin (2 * x₀) > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_2x_leq_1_l1164_116419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sf_win_probability_correct_l1164_116455

/-- Represents the probability of winning a game at home --/
def HomeWinProb : Type := ℝ

/-- Represents the probability of an earthquake occurring after a game in San Francisco --/
def EarthquakeProb : Type := ℝ

/-- Represents the state of the series --/
inductive SeriesState
| SFLead (n : ℤ)  -- San Francisco leads by n games
| OaklandLead (n : ℤ)  -- Oakland leads by n games

/-- Calculates the probability of San Francisco winning the series --/
noncomputable def sfWinProbability (sfHomeWinProb oaklandHomeWinProb earthquakeProb : ℝ) : ℝ :=
  sorry

/-- Theorem stating that given the conditions, the probability of San Francisco winning is 34/73 --/
theorem sf_win_probability_correct 
  (sfHomeWinProb : ℝ) 
  (oaklandHomeWinProb : ℝ) 
  (earthquakeProb : ℝ) 
  (h1 : sfHomeWinProb = (1 : ℝ) / 2) 
  (h2 : oaklandHomeWinProb = (3 : ℝ) / 5) 
  (h3 : earthquakeProb = (1 : ℝ) / 2) : 
  sfWinProbability sfHomeWinProb oaklandHomeWinProb earthquakeProb = (34 : ℝ) / 73 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sf_win_probability_correct_l1164_116455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_annual_profit_l1164_116447

-- Define the annual profit function
noncomputable def L (x a : ℝ) : ℝ := 500 * (x - 30 - a) * Real.exp (40 - x)

-- State the theorem
theorem max_annual_profit (a : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ 5) :
  (∀ x, 35 ≤ x → x ≤ 41 → L x a ≤ 
    if a ≤ 4 
    then 500 * (5 - a) * Real.exp 5
    else 500 * Real.exp (9 - a)) ∧
  (if a ≤ 4
   then L 35 a = 500 * (5 - a) * Real.exp 5
   else L (31 + a) a = 500 * Real.exp (9 - a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_annual_profit_l1164_116447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_distance_to_carol_l1164_116433

-- Define the coordinates of Alice, Bob, and Carol
def alice : ℝ × ℝ := (10, -30)
def bob : ℝ × ℝ := (2, 22)
def carol : ℝ × ℝ := (7, 12)

-- Define the meeting point as the midpoint between Alice and Bob
noncomputable def meeting_point : ℝ × ℝ := (
  (alice.fst + bob.fst) / 2,
  (alice.snd + bob.snd) / 2
)

-- Define the eastward distance as the difference in x-coordinates
noncomputable def eastward_distance : ℝ := carol.fst - meeting_point.fst

-- Theorem to prove
theorem walk_distance_to_carol : eastward_distance = 1 := by
  -- Unfold definitions
  unfold eastward_distance
  unfold meeting_point
  unfold carol
  unfold alice
  unfold bob
  
  -- Simplify the expression
  simp
  
  -- The proof itself
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_distance_to_carol_l1164_116433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1164_116456

noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 1) + 1/3

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1164_116456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1164_116497

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  side_angle_relation : a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B

-- Part 1
theorem part_one (tri : AcuteTriangle) (h1 : tri.a = 2) (h2 : tri.b = Real.sqrt 7) : 
  tri.c = 3 := by
  sorry

-- Part 2
theorem part_two (tri : AcuteTriangle) 
  (h : Real.sqrt 3 * Real.sin (2 * tri.A - π/6) - 2 * (Real.sin (tri.C - π/12))^2 = 0) : 
  tri.A = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1164_116497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_C₃_l1164_116476

-- Define the curves and line
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (-4 + Real.cos α, 3 + Real.sin α)
noncomputable def C₂ (φ : ℝ) : ℝ × ℝ := (8 * Real.cos φ, 3 * Real.sin φ)
def C₃ (t : ℝ) : ℝ × ℝ := (3 + 2*t, -2 + t)

-- Define point P on C₁
noncomputable def P : ℝ × ℝ := C₁ (Real.pi / 2)

-- Define the midpoint M of PQ
noncomputable def M (φ : ℝ) : ℝ × ℝ := 
  let Q := C₂ φ
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the distance function from a point to C₃
noncomputable def distToC₃ (x y : ℝ) : ℝ := 
  (|4*x - y - 13|) / Real.sqrt 5

-- State the theorem
theorem min_distance_to_C₃ : 
  ∃ φ, distToC₃ (M φ).1 (M φ).2 = (8 * Real.sqrt 5) / 5 ∧ 
  ∀ ψ, distToC₃ (M ψ).1 (M ψ).2 ≥ (8 * Real.sqrt 5) / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_C₃_l1164_116476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_properties_l1164_116438

-- Define the vertices of the tetrahedron
def A₁ : ℝ × ℝ × ℝ := (-3, 4, -7)
def A₂ : ℝ × ℝ × ℝ := (1, 5, -4)
def A₃ : ℝ × ℝ × ℝ := (-5, -2, 0)
def A₄ : ℝ × ℝ × ℝ := (2, 5, 4)

-- Define the volume of the tetrahedron
noncomputable def tetrahedron_volume : ℝ := 25 + 1/6

-- Define the height from A₄ to the face A₁A₂A₃
noncomputable def tetrahedron_height : ℝ := Real.sqrt (151/15)

-- Theorem statement
theorem tetrahedron_properties :
  let v := tetrahedron_volume
  let h := tetrahedron_height
  (v = 25 + 1/6) ∧ (h^2 = 151/15) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_properties_l1164_116438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1164_116430

noncomputable def f (x : ℝ) : ℝ := x + 4/x
noncomputable def g (x a : ℝ) : ℝ := 2^x + a

theorem function_inequality (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 1, ∃ x₂ ∈ Set.Icc 2 3, f x₁ ≥ g x₂ a) → 
  a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1164_116430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_betty_hair_color_count_l1164_116405

/-- Represents Betty's order -/
structure Order where
  total_items : ℕ
  slipper_count : ℕ
  slipper_price : ℚ
  lipstick_count : ℕ
  lipstick_price : ℚ
  hair_color_price : ℚ
  total_paid : ℚ

/-- Calculates the number of hair color pieces in Betty's order -/
def hair_color_count (o : Order) : ℕ :=
  let slipper_cost := o.slipper_count * o.slipper_price
  let lipstick_cost := o.lipstick_count * o.lipstick_price
  let hair_color_total := o.total_paid - slipper_cost - lipstick_cost
  (hair_color_total / o.hair_color_price).floor.toNat

/-- Betty's actual order -/
def betty_order : Order :=
  { total_items := 18
  , slipper_count := 6
  , slipper_price := 5/2
  , lipstick_count := 4
  , lipstick_price := 5/4
  , hair_color_price := 3
  , total_paid := 44 }

theorem betty_hair_color_count :
  hair_color_count betty_order = 8 := by
  sorry

#eval hair_color_count betty_order

end NUMINAMATH_CALUDE_ERRORFEEDBACK_betty_hair_color_count_l1164_116405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_p_quadruple_application_eq_neg_four_l1164_116488

noncomputable def f (p : ℝ) : ℝ := 2 * p^2 + 20 * Real.sin p

theorem exists_p_quadruple_application_eq_neg_four :
  ∃ p : ℝ, f (f (f (f p))) = -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_p_quadruple_application_eq_neg_four_l1164_116488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1164_116478

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x - 2 * Real.pi / 3)

theorem f_monotone_increasing : 
  MonotoneOn f (Set.Icc (Real.pi / 12) (7 * Real.pi / 12)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1164_116478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_work_days_l1164_116485

/-- The number of days A needs to complete the work alone -/
noncomputable def a_days : ℝ := 4

/-- The number of days B needs to complete the work alone -/
noncomputable def b_days : ℝ := 8

/-- The number of days A, B, and C need to complete the work together -/
noncomputable def abc_days : ℝ := 2

/-- The work rate of A per day -/
noncomputable def a_rate : ℝ := 1 / a_days

/-- The work rate of B per day -/
noncomputable def b_rate : ℝ := 1 / b_days

/-- The combined work rate of A and B per day -/
noncomputable def ab_rate : ℝ := a_rate + b_rate

/-- The theorem stating that C can complete the work alone in 8 days -/
theorem c_work_days : ∃ (c_days : ℝ), c_days = 8 ∧ 1 / c_days = 1 / abc_days - ab_rate := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_work_days_l1164_116485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1164_116429

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.sin (x + Real.pi/3) - 1/2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (α : ℝ), Real.tan α = Real.sqrt 3/2 → f α = 11/14) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1164_116429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_remainder_l1164_116410

/-- The sequence defined by f(n) = 6n - 5 for positive integers n -/
def f (n : ℕ) : ℕ := 6 * n - 5

/-- The last term of the sequence -/
def last_term : ℕ := 259

/-- The number of terms in the sequence -/
def num_terms : ℕ := (last_term + 5) / 6

/-- The sum of the sequence -/
def sequence_sum : ℕ := (f 1 + f num_terms) * num_terms / 2

theorem sequence_sum_remainder (h : ∀ n, 1 ≤ n → n ≤ num_terms → f n % 6 = 1) :
  sequence_sum % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_remainder_l1164_116410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1164_116411

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the left focus F₁
def left_focus : ℝ × ℝ := (-1, 0)

-- Define the line l
def line (x y : ℝ) : Prop := y = x + 1

-- Define the circle (renamed to avoid conflict)
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem statement
theorem chord_length :
  ∃ (A B : ℝ × ℝ),
    (line A.1 A.2 ∧ circle_equation A.1 A.2) ∧
    (line B.1 B.2 ∧ circle_equation B.1 B.2) ∧
    (A ≠ B) ∧
    (line left_focus.1 left_focus.2) ∧
    (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 14) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1164_116411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l1164_116424

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given triangle satisfies the conditions -/
def special_triangle (t : Triangle) : Prop :=
  (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C ∧
  t.b = Real.sqrt 7 ∧
  t.a + t.c = 4

/-- Area of a triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.c * Real.sin t.B

theorem special_triangle_properties (t : Triangle) (h : special_triangle t) :
  t.B = 60 * π / 180 ∧ area t = 3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l1164_116424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_119_between_10_and_11_l1164_116404

theorem sqrt_119_between_10_and_11 : ∃ (a b : ℕ), a = 10 ∧ b = 11 ∧ (a : ℝ) < Real.sqrt 119 ∧ Real.sqrt 119 < (b : ℝ) ∧ a * b = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_119_between_10_and_11_l1164_116404
