import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_121_l883_88391

theorem square_root_121 : (Real.sqrt 121 = 11) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_121_l883_88391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_four_non_diverse_cells_l883_88317

/-- A color is represented as a natural number. -/
def Color := Fin 101

/-- A grid is represented as a function from pairs of natural numbers to colors. -/
def Grid := Fin 100 → Fin 100 → Color

/-- A cell is diverse if every color appears at least once in its row or column. -/
def is_diverse (g : Grid) (i j : Fin 100) : Prop :=
  ∀ c : Color, 
    (∃ k : Fin 100, g i k = c ∨ g k j = c)

/-- The main theorem: In a 100x100 grid with 101 colors, there are at least 4 non-diverse cells. -/
theorem at_least_four_non_diverse_cells (g : Grid) : 
  ∃ (a b c d : Fin 100),
    (¬ is_diverse g a b) ∧ (¬ is_diverse g a c) ∧ 
    (¬ is_diverse g d b) ∧ (¬ is_diverse g d c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_four_non_diverse_cells_l883_88317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_island_inhabitants_even_l883_88337

/-- Represents the two types of inhabitants on the Island of Contrasts -/
inductive Inhabitant
| Knight
| Liar
deriving BEq, Repr

/-- The Island of Contrasts -/
structure Island where
  inhabitants : List Inhabitant

/-- A statement made by an inhabitant -/
inductive Statement
| EvenKnights
| OddLiars

/-- Function to determine if a statement is true -/
def isStatementTrue (s : Statement) (island : Island) : Prop :=
  match s with
  | Statement.EvenKnights => Even (island.inhabitants.filter (· == Inhabitant.Knight) |>.length)
  | Statement.OddLiars => Odd (island.inhabitants.filter (· == Inhabitant.Liar) |>.length)

/-- Function to determine if an inhabitant tells the truth -/
def tellsTruth (i : Inhabitant) (s : Statement) (island : Island) : Prop :=
  match i with
  | Inhabitant.Knight => isStatementTrue s island
  | Inhabitant.Liar => ¬(isStatementTrue s island)

/-- Theorem: The number of inhabitants on the Island of Contrasts must be even -/
theorem island_inhabitants_even (island : Island) : Even island.inhabitants.length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_island_inhabitants_even_l883_88337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_plane_tetrahedron_volume_constant_l883_88375

/-- The volume of the tetrahedron formed by a tangent plane to x*y*z = m^3 and coordinate planes is constant -/
theorem tangent_plane_tetrahedron_volume_constant (m : ℝ) (x₀ y₀ z₀ : ℝ) :
  x₀ * y₀ * z₀ = m^3 →
  (1/6) * (3 * x₀) * (3 * y₀) * (3 * z₀) = (9/2) * m^3 := by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_plane_tetrahedron_volume_constant_l883_88375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rating_decrease_l883_88305

/-- Represents a movie rating system where viewers rate with integer scores from 0 to 10 --/
structure MovieRating where
  totalScore : ℕ
  voterCount : ℕ
  rating : ℚ

/-- Calculates the new rating after adding a vote --/
def addVote (currentRating : MovieRating) (newVote : ℕ) : MovieRating :=
  { totalScore := currentRating.totalScore + newVote,
    voterCount := currentRating.voterCount + 1,
    rating := (currentRating.totalScore + newVote : ℚ) / (currentRating.voterCount + 1) }

/-- Theorem: The maximum number of consecutive votes that can decrease an integer rating by exactly one unit each time is 5 --/
theorem max_rating_decrease (initialRating : MovieRating) 
  (h1 : initialRating.rating ∈ Set.range (Int.cast : ℤ → ℚ)) 
  (h2 : 0 < initialRating.rating ∧ initialRating.rating ≤ 10) :
  (∃ (n : ℕ), n ≤ 5 ∧ 
    (∀ (votes : List ℕ), votes.length = n → 
      (votes.foldl addVote initialRating).rating = initialRating.rating - n)) ∧
  (∀ (n : ℕ), n > 5 → 
    (∃ (votes : List ℕ), votes.length = n ∧ 
      (votes.foldl addVote initialRating).rating ≠ initialRating.rating - n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rating_decrease_l883_88305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_constant_l883_88373

/-- Given that y = -3x + b is a tangent line to y = x³ - 3x², prove that b = 1 -/
theorem tangent_line_constant (b : ℝ) : 
  (∃ x₀ : ℝ, (x₀^3 - 3*x₀^2 = -3*x₀ + b) ∧ 
             (∀ x : ℝ, x^3 - 3*x^2 ≤ -3*x + b) ∧ 
             (∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |x^3 - 3*x^2 - (-3*x + b)| < ε * |x - x₀|)) →
  b = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_constant_l883_88373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l883_88327

theorem power_function_through_point (α : ℝ) : (4 : ℝ)^α = 2 → α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l883_88327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_return_l883_88383

theorem stock_price_return (original_price : ℝ) (h : original_price > 0) :
  let price_after_two_years := original_price * 1.3 * 1.1
  let decrease_percentage := 1 - (original_price / price_after_two_years)
  ∃ ε > 0, |decrease_percentage - 0.3007| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_return_l883_88383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l883_88344

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (5 - Real.sqrt (9 - Real.sqrt x))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc 0 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l883_88344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_rectangle_l883_88358

-- Define the rectangle ABCD
def Rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 = D.1 ∧ A.2 = B.2 ∧ B.1 = C.1 ∧ C.2 = D.2

-- Define that E is on BC and F is on CD
def OnSide (P Q R : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • Q + t • R

-- Define the distance between two points
noncomputable def Distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the dot product of two vectors
def DotProduct (V W : ℝ × ℝ) : ℝ :=
  V.1 * W.1 + V.2 * W.2

theorem min_dot_product_rectangle
  (A B C D : ℝ × ℝ)
  (E : ℝ × ℝ)
  (F : ℝ × ℝ)
  (h_rect : Rectangle A B C D)
  (h_AB : Distance A B = 1)
  (h_BC : Distance B C = 2)
  (h_E_on_BC : OnSide E B C)
  (h_F_on_CD : OnSide F C D)
  (h_EF : Distance E F = 1) :
  ∃ (min : ℝ), min = 8 - 2 * Real.sqrt 5 ∧
    ∀ (E' F' : ℝ × ℝ), OnSide E' B C → OnSide F' C D → Distance E' F' = 1 →
      DotProduct (A.1 - E'.1, A.2 - E'.2) (A.1 - F'.1, A.2 - F'.2) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_rectangle_l883_88358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l883_88386

theorem trigonometric_identity (α : ℝ) 
  (h1 : Real.cos α = -4/5) 
  (h2 : 2 * Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  (1 + Real.tan (α/2)) / (1 - Real.tan (α/2)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l883_88386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_linear_has_mean_value_two_l883_88334

-- Define the type for our functions
def RealFunction := ℝ → ℝ

-- Define the mean value property
def has_mean_value (f : RealFunction) (c : ℝ) : Prop :=
  ∀ x₁ : ℝ, ∃! x₂ : ℝ, (f x₁ + f x₂) / 2 = c

-- Define our list of functions
def f₁ : RealFunction := λ x ↦ x
def f₂ : RealFunction := λ x ↦ |x|
def f₃ : RealFunction := λ x ↦ x^2
noncomputable def f₄ : RealFunction := λ x ↦ 1/x
noncomputable def f₅ : RealFunction := λ x ↦ x + 1/x

-- State the theorem
theorem only_linear_has_mean_value_two :
  has_mean_value f₁ 2 ∧
  ¬(has_mean_value f₂ 2) ∧
  ¬(has_mean_value f₃ 2) ∧
  ¬(has_mean_value f₄ 2) ∧
  ¬(has_mean_value f₅ 2) := by
  sorry

#check only_linear_has_mean_value_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_linear_has_mean_value_two_l883_88334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_vector_coordinates_l883_88338

/-- Given a plane α with normal vector a and two vectors b and c lying within the plane,
    prove that the normal vector a has specific coordinates. -/
theorem normal_vector_coordinates (x y : ℝ) :
  let a : ℝ × ℝ × ℝ := (x, 2*y - 1, -1/4)
  let b : ℝ × ℝ × ℝ := (-1, 2, 1)
  let c : ℝ × ℝ × ℝ := (3, 1/2, -2)
  (x * (-1) + (2*y - 1) * 2 + (-1/4) * 1 = 0) →
  (x * 3 + (2*y - 1) * (1/2) + (-1/4) * (-2) = 0) →
  a = (-9/52, 1/26, -1/4) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_vector_coordinates_l883_88338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l883_88323

-- Define the parabola
noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the line
noncomputable def line (x y : ℝ) : Prop := y = -Real.sqrt 3 * (x - 1)

-- Define the focus of the parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define the directrix of the parabola
def directrix (p : ℝ) (x : ℝ) : Prop := x = -p/2

-- Define the intersection points M and N
noncomputable def intersection_points (p : ℝ) : Set (ℝ × ℝ) :=
  {point | parabola p point.1 point.2 ∧ line point.1 point.2}

-- Define the circle with MN as diameter
noncomputable def circle_MN (M N : ℝ × ℝ) (x y : ℝ) : Prop :=
  (x - (M.1 + N.1)/2)^2 + (y - (M.2 + N.2)/2)^2 = ((M.1 - N.1)^2 + (M.2 - N.2)^2)/4

-- Theorem statement
theorem parabola_line_intersection 
  (p : ℝ) 
  (h_focus : line (focus p).1 (focus p).2) 
  (M N : ℝ × ℝ) 
  (h_MN : M ∈ intersection_points p ∧ N ∈ intersection_points p) :
  p = 2 ∧ 
  ∃ (x : ℝ), directrix p x ∧ circle_MN M N x ((M.2 + N.2)/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l883_88323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_side_lengths_exist_l883_88345

/-- Represents a square divided into rectangles by parallel lines -/
structure DividedSquare where
  /-- Total number of rectangles -/
  total_rectangles : ℕ
  /-- Number of lines parallel to each side -/
  lines_per_side : ℕ
  /-- Number of squares among the rectangles -/
  num_squares : ℕ
  /-- Ensure the total number of rectangles is correct -/
  total_rectangles_eq : total_rectangles = (lines_per_side + 1) ^ 2
  /-- Ensure the number of squares is less than or equal to the total rectangles -/
  num_squares_le : num_squares ≤ total_rectangles

/-- Helper function to represent the side length of a square -/
def side_length (ds : DividedSquare) (s : Fin ds.num_squares) : ℝ := sorry

/-- Main theorem: In a divided square with 9 squares, at least two have equal side lengths -/
theorem equal_side_lengths_exist (ds : DividedSquare) 
  (h_total : ds.total_rectangles = 100)
  (h_lines : ds.lines_per_side = 9)
  (h_squares : ds.num_squares = 9) :
  ∃ (s1 s2 : Fin ds.num_squares), s1 ≠ s2 ∧ side_length ds s1 = side_length ds s2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_side_lengths_exist_l883_88345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_30_l883_88314

noncomputable def total_distance : ℝ := 120
noncomputable def walking_distance : ℝ := 5
noncomputable def walking_speed : ℝ := 5
noncomputable def biking_distance : ℝ := 35
noncomputable def biking_speed : ℝ := 15
noncomputable def driving_speed : ℝ := 120

noncomputable def driving_distance : ℝ := total_distance - (walking_distance + biking_distance)

noncomputable def walking_time : ℝ := walking_distance / walking_speed
noncomputable def biking_time : ℝ := biking_distance / biking_speed
noncomputable def driving_time : ℝ := driving_distance / driving_speed

noncomputable def total_time : ℝ := walking_time + biking_time + driving_time

theorem average_speed_is_30 : 
  total_distance / total_time = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_30_l883_88314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_theorem_l883_88348

noncomputable def P (b : ℝ) (x : ℝ) : ℝ := x^2 + x/2 + b
noncomputable def Q (c d : ℝ) (x : ℝ) : ℝ := x^2 + c*x + d

theorem polynomial_roots_theorem :
  ∃ b c d : ℝ, 
    (∀ x : ℝ, P b x * Q c d x = Q c d (P b x)) ∧
    c = 1/2 ∧ d = 0 ∧ b = -1/2 ∧
    (∀ x : ℝ, P b (Q c d x) = 0 ↔ x = -1 ∨ x = 1/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_theorem_l883_88348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_range_l883_88376

theorem trig_inequality_range (a : ℝ) : 
  (∀ x : ℝ, Real.sin x^6 + Real.cos x^6 + 2*a*Real.sin x*Real.cos x ≥ 0) ↔ 
  (-1/4 : ℝ) ≤ a ∧ a ≤ (1/4 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_range_l883_88376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_is_integer_l883_88398

def p (x : ℤ) : ℚ :=
  (1/630) * x^9 - (1/21) * x^7 + (13/20) * x^5 - (82/63) * x^3 + (32/35) * x

theorem p_is_integer (x : ℤ) : ∃ n : ℤ, p x = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_is_integer_l883_88398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l883_88333

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x)

def is_symmetry_axis (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem f_increasing_on_interval (ω : ℝ) (h1 : ω > 0) 
  (h2 : is_symmetry_axis (f ω) (π / 6))
  (h3 : ∀ ω' > 0, is_symmetry_axis (f ω') (π / 6) → ω ≤ ω') :
  is_increasing_on (f ω) 0 (π / 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l883_88333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l883_88397

/-- A power function that passes through the point (2, √2) -/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

/-- The theorem stating that f(16) = 4 for a power function passing through (2, √2) -/
theorem power_function_through_point (a : ℝ) (h : f a 2 = Real.sqrt 2) : f a 16 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l883_88397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l883_88310

noncomputable def f (x : ℝ) := Real.exp x + 1 / Real.exp x

theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∃ x₀ : ℝ, x₀ ≥ 1 ∧ f x₀ < a * (-x₀^3 + 3*x₀)) →
  a > (1/2) * (Real.exp 1 + 1 / Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l883_88310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombusParallelepipedSurfaceArea_l883_88325

/-- Represents a right parallelepiped with a rhombus base -/
structure RhombusParallelepiped where
  α : Real  -- One of the angles of the rhombus base
  d : Real  -- Length of the shorter diagonal of the rhombus base
  h : Real  -- Height of the parallelepiped

/-- Calculates the total surface area of a RhombusParallelepiped -/
noncomputable def totalSurfaceArea (rp : RhombusParallelepiped) : Real :=
  (rp.d^2 * (Real.cos (Real.pi/4 - rp.α/2))^2) / (Real.sin (rp.α/2))^2

/-- Theorem: The total surface area of a RhombusParallelepiped is correctly calculated -/
theorem rhombusParallelepipedSurfaceArea (rp : RhombusParallelepiped) 
  (h_angle : 0 < rp.α ∧ rp.α < Real.pi)  -- Ensure angle is valid
  (h_diagonal : rp.d > 0)  -- Ensure diagonal is positive
  (h_height : rp.h > 0)  -- Ensure height is positive
  (h_height_prop : rp.h = (rp.d / (2 * Real.sqrt (1 - Real.cos rp.α))))  -- Height is half the side length
  : totalSurfaceArea rp = 
    (rp.d^2 * (Real.cos (Real.pi/4 - rp.α/2))^2) / (Real.sin (rp.α/2))^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombusParallelepipedSurfaceArea_l883_88325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_squared_l883_88372

/-- An ellipse with the given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- The theorem statement -/
theorem ellipse_eccentricity_squared (e : Ellipse) 
  (P : ℝ × ℝ) 
  (F₁ F₂ A : ℝ × ℝ)
  (h_P_on_ellipse : (P.1 / e.a)^2 + (P.2 / e.b)^2 = 1)
  (h_PF₂_perp_F₁F₂ : (P.1 - F₂.1) * (F₂.1 - F₁.1) + (P.2 - F₂.2) * (F₂.2 - F₁.2) = 0)
  (h_A_on_x_axis : A.2 = 0)
  (h_PA_perp_PF₁ : (P.1 - A.1) * (P.1 - F₁.1) + (P.2 - A.2) * (P.2 - F₁.2) = 0)
  (h_AF₂ : Real.sqrt ((A.1 - F₂.1)^2 + (A.2 - F₂.2)^2) = e.c / 2) :
  (eccentricity e)^2 = (3 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_squared_l883_88372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dormitory_students_majors_l883_88396

theorem dormitory_students_majors (T : ℝ) (h1 : T > 0) : 
  (T / 2 - 4 * (T / 2 - 4 * T / 10)) / T = 1 / 10 := by
  -- Let's break down the calculation step by step
  have first_year : ℝ := T / 2
  have second_year : ℝ := T / 2
  have first_year_undeclared : ℝ := (4 / 5) * first_year
  have first_year_declared : ℝ := first_year - first_year_undeclared
  have second_year_declared : ℝ := 4 * first_year_declared
  have second_year_undeclared : ℝ := second_year - second_year_declared

  -- Now we can proceed with the proof
  sorry

#eval (10 : ℚ)⁻¹ -- This should output (1 : ℚ) / 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dormitory_students_majors_l883_88396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_numbers_sum_problem_l883_88392

theorem ordered_numbers_sum_problem (a b c d e : ℝ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℝ) = 
    ({32, 36, 37, 38, 39, 40, 41, 44, 48, 51} : Finset ℝ) →
  e = 27.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_numbers_sum_problem_l883_88392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_seats_taken_l883_88399

-- Define the groups
structure CafeteriaGroup where
  tables : ℕ
  seats_per_table : ℕ
  unseated_ratio : ℚ

-- Define the cafeteria
def cafeteria : List CafeteriaGroup := [
  ⟨10, 8, 1/4⟩,  -- Group A
  ⟨7, 12, 1/3⟩,  -- Group B
  ⟨8, 10, 1/5⟩   -- Group C
]

-- Function to calculate seats taken for a group
def seats_taken (g : CafeteriaGroup) : ℚ :=
  g.tables * g.seats_per_table * (1 - g.unseated_ratio)

-- Theorem to prove
theorem total_seats_taken :
  (cafeteria.map seats_taken).sum = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_seats_taken_l883_88399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l883_88350

theorem log_inequality_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : Real.log 2 / Real.log a < 1) :
  (0 < a ∧ a < 1) ∨ (a > 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l883_88350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scalene_triangle_l883_88388

/-- A triangle is scalene if all its sides have different lengths. -/
def IsScalene (a b c : ℝ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem scalene_triangle (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  IsScalene a b c :=
by
  unfold IsScalene
  exact ⟨hab, hbc, hca⟩

#check scalene_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scalene_triangle_l883_88388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_xiao_ming_choose_any_movie_prob_all_choose_c_given_xiao_ming_chose_c_l883_88390

-- Define the set of movies
inductive Movie : Type
| A : Movie  -- "Life is Unfamiliar"
| B : Movie  -- "King of the Sky"
| C : Movie  -- "Prosecution Storm"

-- Define the set of people
inductive Person : Type
| XiaoMing : Person
| XiaoLi : Person
| XiaoHong : Person

-- Define a function to represent a person's movie choice
def movieChoice : Person → Movie → Prop := sorry

-- Define the probability measure
noncomputable def P : (Set (Person → Movie)) → ℝ := sorry

-- Theorem 1: Probability of Xiao Ming choosing any specific movie is 1/3
theorem prob_xiao_ming_choose_any_movie (m : Movie) :
  P {c | c Person.XiaoMing = m} = 1/3 := by sorry

-- Theorem 2: Given Xiao Ming chose C, probability all three choose C is 1/9
theorem prob_all_choose_c_given_xiao_ming_chose_c :
  P {c | c Person.XiaoMing = Movie.C ∧ c Person.XiaoLi = Movie.C ∧ c Person.XiaoHong = Movie.C} /
  P {c | c Person.XiaoMing = Movie.C} = 1/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_xiao_ming_choose_any_movie_prob_all_choose_c_given_xiao_ming_chose_c_l883_88390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decaf_percentage_is_44_percent_l883_88378

/-- Calculates the percentage of decaffeinated coffee in the total stock -/
noncomputable def decaf_percentage (initial_stock : ℝ) (initial_decaf_percent : ℝ) 
  (additional_stock : ℝ) (additional_decaf_percent : ℝ) : ℝ :=
  let initial_decaf := initial_stock * initial_decaf_percent / 100
  let additional_decaf := additional_stock * additional_decaf_percent / 100
  let total_decaf := initial_decaf + additional_decaf
  let total_stock := initial_stock + additional_stock
  (total_decaf / total_stock) * 100

/-- Theorem stating that the percentage of decaffeinated coffee in the total stock is 44% -/
theorem decaf_percentage_is_44_percent :
  decaf_percentage 400 40 100 60 = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decaf_percentage_is_44_percent_l883_88378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_shifts_exist_l883_88387

theorem coprime_shifts_exist (a b c : ℤ) (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S,
    Int.gcd (a + n) (b + n) = 1 ∧
    Int.gcd (a + n) (c + n) = 1 ∧
    Int.gcd (b + n) (c + n) = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_shifts_exist_l883_88387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_d_l883_88321

-- Define the function f(x) = -2x + c
noncomputable def f (c : ℤ) : ℝ → ℝ := λ x ↦ -2 * x + c

-- Define the inverse function of f
noncomputable def f_inv (c : ℤ) : ℝ → ℝ := λ x ↦ (c - x) / 2

theorem intersection_point_d (c : ℤ) :
  (∃ d : ℤ, (f c 4 : ℝ) = d ∧ f_inv c 4 = d) → (f c 4 : ℝ) = 4 := by
  sorry

#check intersection_point_d

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_d_l883_88321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_plane_distance_l883_88308

theorem sphere_plane_distance (V : ℝ) (d : ℝ) (h1 : V = 36 * Real.pi) (h2 : d = 2 * Real.sqrt 5) :
  let R := (3 * V / (4 * Real.pi)) ^ (1/3)
  let r := d / 2
  Real.sqrt (R^2 - r^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_plane_distance_l883_88308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_density_function_transformation_l883_88374

open Real Matrix

variable {n : ℕ}
variable (A : Matrix (Fin n) (Fin n) ℝ)
variable (X Y : Fin n → ℝ)
variable (b : Fin n → ℝ)
variable (f_X f_Y : (Fin n → ℝ) → ℝ)

def transformation (A : Matrix (Fin n) (Fin n) ℝ) (X : Fin n → ℝ) (b : Fin n → ℝ) : Fin n → ℝ :=
  fun i => (A.mulVec X + b) i

theorem density_function_transformation
  (h_det : |det A| > 0)
  (h_transform : Y = transformation A X b) :
  f_Y Y = (1 / |det A|) * f_X (A⁻¹.mulVec (Y - b)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_density_function_transformation_l883_88374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tiling_5x7_rectangle_l883_88362

/-- Represents an L-tromino -/
structure LTromino where
  cells : Finset (ℕ × ℕ)
  size_eq_3 : cells.card = 3

/-- Represents a tiling of a rectangle -/
structure Tiling where
  width : ℕ
  height : ℕ
  trominos : List LTromino
  covers_all : ∀ x y, x < width → y < height → ∃ t ∈ trominos, (x, y) ∈ t.cells
  equal_coverage : ∃ k : ℕ, ∀ x y, x < width → y < height → 
    (trominos.filter (fun t ↦ (x, y) ∈ t.cells)).length = k

/-- Theorem: It's not possible to tile a 5x7 rectangle with L-trominos such that each cell is covered by the same number of trominos -/
theorem no_tiling_5x7_rectangle :
  ¬ ∃ (t : Tiling), t.width = 5 ∧ t.height = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tiling_5x7_rectangle_l883_88362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shelf_books_problem_l883_88393

theorem shelf_books_problem (total : ℕ) (percentage : ℚ) (result : ℕ) : 
  total = 280 → 
  percentage = 1/8 → 
  result = 160 → 
  (1 - percentage) * (result : ℚ) = ((total - result : ℕ) : ℚ) + percentage * (result : ℚ) → 
  result = total / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shelf_books_problem_l883_88393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_squares_l883_88394

/-- A square in the coordinate plane -/
structure Square where
  center : ℕ × ℕ
  vertices : Fin 4 → ℕ × ℕ

/-- Predicate to check if a square has natural number coordinates for vertices and center (55, 40) -/
def is_valid_square (s : Square) : Prop :=
  s.center = (55, 40) ∧ 
  ∀ i : Fin 4, s.vertices i ∈ Set.prod (Set.range id) (Set.range id)

/-- The set of all valid squares -/
def valid_squares : Set Square :=
  {s : Square | is_valid_square s}

/-- The theorem to be proved -/
theorem count_valid_squares : 
  ∃ (S : Finset Square), (∀ s ∈ S, is_valid_square s) ∧ S.card = 1560 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_squares_l883_88394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_bisector_set_l883_88340

/-- The set of angles with terminal side equal to the second quadrant bisector -/
def SecondQuadrantBisectorAngles : Set ℝ :=
  {x | ∃ k : ℤ, x = 3 * Real.pi / 4 + 2 * k * Real.pi}

/-- Predicate for an angle having its terminal side as the second quadrant bisector -/
def IsSecondQuadrantBisector (α : ℝ) : Prop :=
  -- This is a placeholder definition. In reality, this would involve a more complex
  -- geometric condition, but we simplify it for the purpose of this problem.
  α % (2 * Real.pi) = 3 * Real.pi / 4

theorem angle_in_second_quadrant_bisector_set (α : ℝ) :
  IsSecondQuadrantBisector α → α ∈ SecondQuadrantBisectorAngles := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_bisector_set_l883_88340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l883_88349

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = √6, b = √3 + 1, and C = 45°, then A = 60° -/
theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 6 →
  b = Real.sqrt 3 + 1 →
  C = 45 * Real.pi / 180 →
  A = 60 * Real.pi / 180 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l883_88349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_focal_lengths_l883_88315

/-- Focal length of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def focal_length (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

theorem equal_focal_lengths :
  ∀ k : ℝ, 0 < k → k < 9 →
  focal_length 5 3 = focal_length (Real.sqrt (25 - k)) (Real.sqrt (9 - k)) := by
  intro k hk1 hk2
  unfold focal_length
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_focal_lengths_l883_88315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f3_decreases_f3_only_decreasing_function_l883_88316

-- Define the slope of a linear function
def linear_slope (a b : ℝ) : ℝ := a

-- Define the four functions
def f1 (x : ℝ) : ℝ := 2 * x + 8
def f2 (x : ℝ) : ℝ := 4 * x - 2
def f3 (x : ℝ) : ℝ := -2 * x + 8
def f4 (x : ℝ) : ℝ := 4 * x

-- Theorem stating that only f3 has a negative slope
theorem only_f3_decreases :
  linear_slope 2 8 > 0 ∧ 
  linear_slope 4 (-2) > 0 ∧ 
  linear_slope (-2) 8 < 0 ∧ 
  linear_slope 4 0 > 0 :=
by
  apply And.intro
  · exact (by norm_num : (2 : ℝ) > 0)
  · apply And.intro
    · exact (by norm_num : (4 : ℝ) > 0)
    · apply And.intro
      · exact (by norm_num : (-2 : ℝ) < 0)
      · exact (by norm_num : (4 : ℝ) > 0)

-- Theorem stating that f3 is the only function where y decreases as x increases
theorem f3_only_decreasing_function :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f3 x₂ < f3 x₁ :=
by
  intros x₁ x₂ h
  unfold f3
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f3_decreases_f3_only_decreasing_function_l883_88316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milo_eggs_needed_l883_88380

/-- The number of dozens of eggs needed given the total weight, weight per egg, and eggs per dozen -/
def dozens_needed (total_weight : ℚ) (weight_per_egg : ℚ) (eggs_per_dozen : ℕ) : ℕ :=
  (((total_weight / weight_per_egg) / eggs_per_dozen).ceil).toNat

/-- Theorem: Given the conditions, Milo needs 8 dozen eggs -/
theorem milo_eggs_needed :
  dozens_needed 6 (1/16) 12 = 8 := by
  sorry

#eval dozens_needed 6 (1/16) 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milo_eggs_needed_l883_88380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C₂_and_intersection_property_l883_88381

-- Define the unit circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the stretched curve C₂
def C₂ (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 2

-- Define point P
def P : ℝ × ℝ := (2, 0)

theorem curve_C₂_and_intersection_property :
  -- The equation of C₂ is correct
  (∀ x y : ℝ, C₂ x y ↔ (x / (2 * Real.sqrt 2))^2 + (y / 2)^2 = 1) ∧
  -- The sum of reciprocals of distances from P to intersection points is √2
  (∃ A B : ℝ × ℝ, 
    C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧ 
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    1 / ((A.1 - P.1)^2 + (A.2 - P.2)^2).sqrt + 
    1 / ((B.1 - P.1)^2 + (B.2 - P.2)^2).sqrt = Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C₂_and_intersection_property_l883_88381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_property_l883_88331

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 8x -/
def Parabola := {p : Point | p.y^2 = 8 * p.x}

/-- The focus of the parabola y^2 = 8x -/
def focus : Point := ⟨2, 0⟩

/-- The projection of a point onto the y-axis -/
def proj_y_axis (p : Point) : Point := ⟨0, p.y⟩

/-- Distance between two points -/
noncomputable def dist (p1 p2 : Point) : ℝ := Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_distance_property (p : Point) (h : p ∈ Parabola) :
  dist p focus - dist p (proj_y_axis p) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_property_l883_88331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l883_88343

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y + 6 = 0

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 20

-- Define the center of the circle
def center : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem circle_tangent_to_line : 
  ∃ (x₀ y₀ : ℝ), line x₀ y₀ ∧ circle_equation x₀ y₀ ∧
  ∀ (x y : ℝ), line x y → ((x - 2)^2 + (y - 1)^2 ≥ 20) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l883_88343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calories_l883_88360

/-- Represents the composition of lemonade in grams -/
structure LemonadeComposition where
  lemon_juice : ℚ
  sugar : ℚ
  honey : ℚ

/-- Represents the calorie content per 100g of each ingredient -/
structure CalorieContent where
  lemon_juice : ℚ
  sugar : ℚ
  honey : ℚ

/-- Calculates the total calories in the lemonade -/
def total_calories (comp : LemonadeComposition) (cal : CalorieContent) : ℚ :=
  (comp.lemon_juice * cal.lemon_juice + 
   comp.sugar * cal.sugar + 
   comp.honey * cal.honey) / 100

/-- Calculates the total weight of the lemonade -/
def total_weight (comp : LemonadeComposition) : ℚ :=
  comp.lemon_juice + comp.sugar + comp.honey

/-- Theorem stating that 250g of lemonade contains 665 calories -/
theorem lemonade_calories 
  (comp : LemonadeComposition) 
  (cal : CalorieContent) 
  (h1 : comp = ⟨150, 200, 300⟩) 
  (h2 : cal = ⟨30, 386, 304⟩) : 
  (250 / total_weight comp) * total_calories comp cal = 665 := by
  sorry

#eval (250 / (150 + 200 + 300 : ℚ)) * ((150 * 30 + 200 * 386 + 300 * 304) / 100 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calories_l883_88360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_score_is_74_l883_88379

/-- Represents the mean score of a class of students -/
structure ClassMean where
  score : ℚ
  students : ℚ

/-- Calculates the weighted average of class means -/
noncomputable def weightedAverage (classes : List ClassMean) : ℚ :=
  (classes.map (λ c => c.score * c.students)).sum / (classes.map (λ c => c.students)).sum

/-- Theorem stating that the mean score of all students is 74 -/
theorem mean_score_is_74 (morning midday afternoon : ClassMean)
  (h1 : morning.score = 85)
  (h2 : midday.score = 75)
  (h3 : afternoon.score = 65)
  (h4 : morning.students / midday.students = 2/3)
  (h5 : midday.students / afternoon.students = 3/2)
  : weightedAverage [morning, midday, afternoon] = 74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_score_is_74_l883_88379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_fourth_quadrant_l883_88301

theorem tan_value_fourth_quadrant (α : ℝ) 
  (h1 : Real.sin α = -4/5) 
  (h2 : α ∈ Set.Icc (3*Real.pi/2) (2*Real.pi)) : 
  Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_fourth_quadrant_l883_88301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l883_88354

def mixed_number_to_fraction (whole : ℤ) (numerator : ℕ) (denominator : ℕ) : ℚ :=
  (whole : ℚ) + (numerator : ℚ) / (denominator : ℚ)

def fraction_to_mixed_number (q : ℚ) : ℤ × ℕ × ℕ :=
  let whole := q.floor
  let remainder := q - (whole : ℚ)
  let numerator := (remainder.num.natAbs)
  let denominator := (remainder.den)
  (whole, numerator, denominator)

theorem calculation_proof :
  let a := mixed_number_to_fraction 5 2 7
  let b := mixed_number_to_fraction 3 3 4
  let c := mixed_number_to_fraction 4 1 6
  let d := mixed_number_to_fraction 2 1 5
  let result := 47 * (a - b) / (c + d)
  fraction_to_mixed_number result = (11, 13, 99) := by
  sorry

#eval fraction_to_mixed_number (47 * (mixed_number_to_fraction 5 2 7 - mixed_number_to_fraction 3 3 4) / (mixed_number_to_fraction 4 1 6 + mixed_number_to_fraction 2 1 5))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l883_88354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l883_88318

/-- Converts kilometers per hour to meters per second -/
noncomputable def km_per_hr_to_m_per_s (v : ℝ) : ℝ := v * (1000 / 3600)

/-- Calculates the length of the second train given the parameters of the problem -/
noncomputable def second_train_length (v1 v2 t l1 : ℝ) : ℝ :=
  (km_per_hr_to_m_per_s v1 + km_per_hr_to_m_per_s v2) * t - l1

theorem train_length_problem :
  let v1 : ℝ := 60  -- speed of first train in km/hr
  let v2 : ℝ := 40  -- speed of second train in km/hr
  let t : ℝ := 12.239020878329734  -- time to cross in seconds
  let l1 : ℝ := 140  -- length of first train in meters
  second_train_length v1 v2 t l1 = 200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l883_88318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamonds_in_F_10_l883_88347

/-- Triangular number -/
def T (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Number of diamonds in figure F_n -/
def F : ℕ → ℕ
  | 0 => 1  -- Add this base case
  | 1 => 1
  | n + 1 => F n + 4 * T (n + 1)

/-- The main theorem: F_10 has 877 diamonds -/
theorem diamonds_in_F_10 : F 10 = 877 := by
  -- Proof steps would go here
  sorry

#eval F 10  -- This will evaluate F 10 and display the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamonds_in_F_10_l883_88347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_lambda_range_l883_88320

theorem hyperbola_lambda_range (lambda : ℝ) (e : ℝ) :
  (0 < lambda) →
  (lambda < 1) →
  (1 < e) →
  (e < 2) →
  (∀ x y : ℝ, x^2 / lambda - y^2 / (1 - lambda) = 1) →
  (e = 1 / Real.sqrt lambda) →
  (1 / 4 < lambda ∧ lambda < 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_lambda_range_l883_88320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_length_is_75_l883_88342

-- Define the route and trains
structure Route where
  length : ℝ

structure Train where
  travel_time : ℝ

-- Define the meeting distance
def meeting_distance : ℝ := 30

-- State the theorem
theorem route_length_is_75 (route : Route) (train_A train_B : Train) :
  train_A.travel_time = 3 →
  train_B.travel_time = 2 →
  (∃ (t : ℝ), t > 0 ∧ t * (route.length / train_A.travel_time) = meeting_distance ∧
              t * (route.length / train_B.travel_time) + meeting_distance = route.length) →
  route.length = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_length_is_75_l883_88342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_probability_l883_88339

/-- Represents a point on the lattice --/
structure LatticePoint where
  x : Int
  y : Int

/-- Represents the possible directions of movement --/
inductive Direction where
  | Up
  | Down
  | Left
  | Right

/-- A function to determine if a point is reachable in n moves --/
def isReachable (start : LatticePoint) (end_ : LatticePoint) (n : Nat) : Prop :=
  (abs (end_.x - start.x) + abs (end_.y - start.y)) ≤ n

/-- A function to determine if a point has even parity (sum of coordinates is even) --/
def hasEvenParity (point : LatticePoint) : Prop :=
  (point.x + point.y) % 2 = 0

/-- The theorem stating the probability of reaching (0,2) from (0,0) in 7 moves --/
theorem ant_probability :
  let start : LatticePoint := ⟨0, 0⟩
  let end_ : LatticePoint := ⟨0, 2⟩
  let numMoves : Nat := 7
  isReachable start end_ numMoves ∧
  hasEvenParity start ∧
  hasEvenParity end_ →
  (1 : ℚ) / 8 = sorry := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_probability_l883_88339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l883_88395

def S (n : ℕ) : ℕ := 3^n + 1

def a : ℕ → ℕ
  | 0 => 4  -- Added case for 0
  | 1 => 4
  | (n+2) => 2 * 3^(n+1)

theorem sequence_formula (n : ℕ) : 
  Finset.sum (Finset.range n) (fun i => a (i+1)) = S n := by
  sorry

#check sequence_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l883_88395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_two_identical_digits_l883_88357

def has_two_identical_digits (n : Nat) : Bool :=
  let digits := n.digits 10
  digits.any (fun d => digits.count d = 2)

def count_numbers_with_two_identical_digits (start finish : Nat) : Nat :=
  (List.range (finish - start + 1)).map (· + start)
    |>.filter has_two_identical_digits
    |>.length

theorem unique_number_with_two_identical_digits :
  count_numbers_with_two_identical_digits 10 40 = 1 ∧
  has_two_identical_digits 33 = true := by
  sorry

#eval count_numbers_with_two_identical_digits 10 40
#eval has_two_identical_digits 33

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_two_identical_digits_l883_88357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_power_and_log_equivalence_l883_88326

theorem half_power_and_log_equivalence (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1/2 : ℝ)^a < (1/2 : ℝ)^b ↔ Real.log a > Real.log b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_power_and_log_equivalence_l883_88326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sequence_and_sum_l883_88366

theorem prime_sequence_and_sum : 
  (∃ p : ℕ → ℕ, p 7 = 17 ∧ (∀ n, n ≥ 7 → Nat.Prime (p n)) ∧ 
   (∀ n m, n < m → p n < p m)) → 
  (∃ q : ℕ → ℕ, q 12 = 37 ∧ 
   (Finset.sum (Finset.range 6) (fun i => q (i + 7))) = 156) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sequence_and_sum_l883_88366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_not_divisible_by_seven_l883_88346

theorem max_subset_size_not_divisible_by_seven :
  ∃ (S : Finset ℕ),
    (∀ x, x ∈ S → x ≤ 50) ∧
    (∀ x y, x ∈ S → y ∈ S → x ≠ y → ¬(7 ∣ (x + y))) ∧
    S.card = 23 ∧
    (∀ T : Finset ℕ,
      (∀ x, x ∈ T → x ≤ 50) →
      (∀ x y, x ∈ T → y ∈ T → x ≠ y → ¬(7 ∣ (x + y))) →
      T.card ≤ 23) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_not_divisible_by_seven_l883_88346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_four_integers_l883_88367

theorem product_of_four_integers (A B C D : ℕ) 
  (A_pos : A > 0) (B_pos : B > 0) (C_pos : C > 0) (D_pos : D > 0)
  (sum_eq : A + B + C + D = 100)
  (relation : A + 5 = B - 2 ∧ B - 2 = C * 2 ∧ C * 2 = D / 2) :
  A * B * C * D = (161 * 224 * 103 * 412) / 6561 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_four_integers_l883_88367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_EI_is_24_l883_88313

/-- Two circles with radii 2 and 3, centers 10 units apart -/
structure TwoCircles where
  center_distance : ℝ := 10
  radius_small : ℝ := 2
  radius_large : ℝ := 3

/-- Point E: intersection of common external tangents -/
def external_tangent_intersection (c : TwoCircles) : ℝ → Prop :=
  λ e => True  -- placeholder definition

/-- Point I: intersection of common internal tangents -/
def internal_tangent_intersection (c : TwoCircles) : ℝ → Prop :=
  λ i => True  -- placeholder definition

/-- Distance between points E and I -/
def distance_EI (c : TwoCircles) (e i : ℝ) : ℝ :=
  abs (e - i)

/-- Theorem: The distance between E and I is 24 -/
theorem distance_EI_is_24 (c : TwoCircles) 
  (e : ℝ) (he : external_tangent_intersection c e)
  (i : ℝ) (hi : internal_tangent_intersection c i) :
  distance_EI c e i = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_EI_is_24_l883_88313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_two_l883_88319

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1 else 2^x

-- State the theorem
theorem f_composition_equals_two : f (f (2/3)) = 2 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_two_l883_88319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l883_88353

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def sequence_prop (a : ℕ → ℝ) : Prop :=
  a 1 = 1/2 ∧ 
  (∀ n : ℕ, a (n + 2) = f (a n)) ∧
  (∀ n : ℕ, a n > 0) ∧
  a 20 = a 18

theorem sequence_sum (a : ℕ → ℝ) (h : sequence_prop a) : 
  a 2016 + a 2017 = Real.sqrt 2 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l883_88353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_non_overlapping_crosses_l883_88306

/-- A cross-shaped object with two perpendicular branches -/
structure Cross where
  center : ℝ × ℝ
  branch_length : ℝ
  branch_length_ge_1 : branch_length ≥ 1

/-- A circular field -/
structure CircularField where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two crosses overlap -/
def crosses_overlap (c1 c2 : Cross) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 < (c1.branch_length / Real.sqrt 2)^2 + (c2.branch_length / Real.sqrt 2)^2

/-- Checks if a cross is within the field -/
def cross_in_field (c : Cross) (f : CircularField) : Prop :=
  let (x, y) := c.center
  let (cx, cy) := f.center
  (x - cx)^2 + (y - cy)^2 ≤ f.radius^2

/-- The main theorem -/
theorem impossible_non_overlapping_crosses :
  ¬ ∃ (crosses : Fin (10^9) → Cross) (f : CircularField),
    (f.radius = 1000) ∧ 
    (∀ i, cross_in_field (crosses i) f) ∧
    (∀ i j, i ≠ j → ¬ crosses_overlap (crosses i) (crosses j)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_non_overlapping_crosses_l883_88306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_function_b_value_l883_88385

-- Define the piecewise function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 3 then 3 * x^2 - 8 else b * x + 7

-- State the theorem
theorem continuous_function_b_value :
  ∃ b : ℝ, Continuous (f b) ∧ b = 4 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_function_b_value_l883_88385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_equals_two_a_equals_two_implies_local_maximum_three_tangent_lines_l883_88389

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (a * x^2 + x - 1)

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (a * x^2 + x + 2 * a * x)

theorem extremum_implies_a_equals_two :
  ∀ a : ℝ, (∃ x : ℝ, x = -5/2 ∧ f_derivative a x = 0) → a = 2 := by
  sorry

theorem a_equals_two_implies_local_maximum :
  (∀ x : ℝ, x < -5/2 → f_derivative 2 x > 0) ∧
  (∀ x : ℝ, -5/2 < x ∧ x < 0 → f_derivative 2 x < 0) ∧
  (∀ x : ℝ, x > 0 → f_derivative 2 x > 0) := by
  sorry

-- Define the equation for finding tangent lines
def tangent_line_equation (x : ℝ) : ℝ := 4 * x^3 + 5 * x^2 - 13 * x + 4

theorem three_tangent_lines :
  ∃ x₁ x₂ x₃ : ℝ, 
    x₁ < x₂ ∧ x₂ < x₃ ∧
    tangent_line_equation x₁ = 0 ∧
    tangent_line_equation x₂ = 0 ∧
    tangent_line_equation x₃ = 0 ∧
    (∀ x : ℝ, tangent_line_equation x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_equals_two_a_equals_two_implies_local_maximum_three_tangent_lines_l883_88389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_positive_l883_88311

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x * (2 * x - 1) else x * (2 * x + 1)

-- State the theorem
theorem f_even_and_positive (x : ℝ) : 
  (∀ y : ℝ, f y = f (-y)) →  -- f is even
  (∀ z : ℝ, z < 0 → f z = z * (2 * z - 1)) →  -- definition for x < 0
  (x > 0 → f x = x * (2 * x + 1)) :=  -- conclusion for x > 0
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_positive_l883_88311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_number_with_digit_sum_l883_88302

theorem existence_of_number_with_digit_sum (n k : ℕ)
  (hn : n > 0)
  (h1 : ¬ 3 ∣ n)
  (h2 : k ≥ n) :
  ∃ m : ℕ, m > 0 ∧ n ∣ m ∧ (Nat.digits 10 m).sum = k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_number_with_digit_sum_l883_88302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l883_88322

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + 1 / x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 0 ∧ x ≠ -1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l883_88322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_symmetry_l883_88328

/-- Recursive definition of the sequence R_n -/
def R : ℕ → List ℕ
  | 0 => []  -- Add this case to handle Nat.zero
  | 1 => [1]
  | n + 1 => 
    let prev := R n
    List.join (prev.map (λ x => List.range x)) ++ [n + 1]

/-- The k-th element from the left in R_n is 1 iff the k-th element from the right is not 1 -/
theorem R_symmetry (n : ℕ) (h : n > 1) (k : ℕ) (hk : k > 0 ∧ k ≤ (R n).length) :
  (R n).get? (k - 1) = some 1 ↔ (R n).get? ((R n).length - k) ≠ some 1 := by
  sorry

#check R_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_symmetry_l883_88328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_locations_l883_88371

/-- The distance between two locations given the conditions of two cars meeting --/
theorem distance_between_locations (v_b : ℝ) (t : ℝ) (h1 : v_b = 60) (h2 : t = 2.4) : 
  (1.5 * v_b + v_b) * t = 360 := by
  -- Let's define v_a and d for clarity
  let v_a := 1.5 * v_b
  let d := (v_a + v_b) * t

  -- Now we'll prove that d equals 360
  calc
    d = (v_a + v_b) * t := rfl
    _ = (1.5 * v_b + v_b) * t := rfl
    _ = (1.5 * 60 + 60) * 2.4 := by rw [h1, h2]
    _ = 150 * 2.4 := by norm_num
    _ = 360 := by norm_num

-- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_locations_l883_88371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_contains_point_and_line_l883_88355

/-- The line described by the given equations -/
def line (t : ℝ) : ℝ × ℝ × ℝ := (4*t + 2, -t + 2, 3*t - 1)

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop := 3*x - 4*z - 10 = 0

theorem plane_contains_point_and_line :
  let point : ℝ × ℝ × ℝ := (2, 1, -1)
  (plane_equation point.1 point.2.1 point.2.2) ∧
  (∀ t, plane_equation (line t).1 (line t).2.1 (line t).2.2) ∧
  (3 > 0) ∧
  (Nat.gcd (Nat.gcd 3 0) (Nat.gcd 4 10) = 1) := by
  sorry

#check plane_contains_point_and_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_contains_point_and_line_l883_88355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l883_88335

/-- Given functions f and g, prove that the interval of monotonic increase for their product is (-∞, 3] -/
theorem monotonic_increase_interval
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = 2 * x - 4)
  (hg : ∀ x, g x = -x + 4) :
  ∃ (S : Set ℝ), S = Set.Iic 3 ∧ StrictMonoOn (fun x ↦ f x * g x) S :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l883_88335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_2_l883_88309

def sequence_a : ℕ → ℚ
  | 0 => 3
  | (n + 1) => (5 * sequence_a n - 13) / (3 * sequence_a n - 7)

theorem a_2016_equals_2 : sequence_a 2016 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_2_l883_88309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_A_measures_l883_88329

/-- The number of possible measures for angle A -/
def num_possible_measures : ℕ := 11

/-- Predicate to check if a pair of angles satisfies the given conditions -/
def valid_angle_pair (A B : ℕ) : Prop :=
  A > 0 ∧ B > 0 ∧ (∃ k : ℕ+, A = k * B) ∧ A + B = 90

/-- The set of all valid measures for angle A -/
def valid_A_measures : Set ℕ :=
  {A : ℕ | ∃ B : ℕ, valid_angle_pair A B}

/-- Axiom stating that valid_A_measures is finite -/
axiom valid_A_measures_finite : Finite valid_A_measures

/-- Instance of Fintype for valid_A_measures -/
noncomputable instance : Fintype valid_A_measures :=
  Set.Finite.fintype valid_A_measures_finite

/-- Theorem stating the number of valid measures for angle A -/
theorem count_valid_A_measures :
  Fintype.card valid_A_measures = num_possible_measures := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_A_measures_l883_88329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_path_difference_l883_88370

theorem park_path_difference :
  let length : ℝ := 200
  let width : ℝ := 150
  let jerry_path := length + width
  let silvia_path := Real.sqrt (length^2 + width^2)
  let difference_percentage := (jerry_path - silvia_path) / jerry_path * 100
  ⌊difference_percentage⌋₊ = 29 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_path_difference_l883_88370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exp_function_l883_88368

theorem min_value_exp_function (x : ℝ) : 
  Real.exp x + 4 * Real.exp (-x) ≥ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exp_function_l883_88368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l883_88303

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : a 1 = 1) (h2 : a 3 = 4) :
  (arithmetic_sequence a → a 11 = 16) ∧
  (arithmetic_sequence (λ n ↦ 1 / (1 + a n)) → 
    ∀ n, a n = (7 + 3 * ↑n) / (13 - 3 * ↑n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l883_88303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_result_is_four_l883_88361

def S : Finset Nat := {2, 4, 6, 8, 10, 12}

def process (a b c : Nat) : Nat :=
  (max a (max b c) - min a (min b c)) * (a + b + c - max a (max b c) - min a (min b c))

theorem smallest_result_is_four :
  (∀ a b c, a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → process a b c ≥ 4) ∧
  (∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ process a b c = 4) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_result_is_four_l883_88361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_transformation_l883_88369

theorem determinant_transformation (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = 5 →
  Matrix.det !![p, 5*p + 4*q; r, 5*r + 4*s] = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_transformation_l883_88369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_trip_fraction_l883_88382

theorem field_trip_fraction (b g : ℕ) (h1 : b = 2 * g) : 
  (4 : ℚ) / 5 * b / ((4 : ℚ) / 5 * b + (3 : ℚ) / 5 * g) = 8 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_trip_fraction_l883_88382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squared_radius_in_cones_l883_88312

/-- Represents a right circular cone --/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones and a sphere --/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ
  sphereRadius : ℝ

/-- The maximum squared radius of a sphere fitting in the cone configuration --/
noncomputable def maxSquaredRadius (config : ConeConfiguration) : ℝ :=
  (144 : ℝ) / 29

/-- Theorem stating the maximum squared radius of the sphere in the given configuration --/
theorem max_squared_radius_in_cones 
  (config : ConeConfiguration)
  (h1 : config.cone1 = config.cone2)
  (h2 : config.cone1.baseRadius = 4)
  (h3 : config.cone1.height = 10)
  (h4 : config.intersectionDistance = 4)
  : maxSquaredRadius config = (144 : ℝ) / 29 :=
by
  sorry

#check max_squared_radius_in_cones

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squared_radius_in_cones_l883_88312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l883_88359

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem g_properties :
  (∀ x : ℝ, g (π/6 + x) = -g (π/6 - x)) ∧ 
  (∀ x y : ℝ, π/12 ≤ x ∧ x < y ∧ y ≤ 5*π/12 → g x < g y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l883_88359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_factorial_problem_l883_88351

theorem gcd_factorial_problem (b : ℕ) : 
  Nat.gcd (Nat.factorial (b - 2)) (Nat.gcd (Nat.factorial (b + 1)) (Nat.factorial (b + 4))) = 5040 → b = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_factorial_problem_l883_88351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_1_part_2_part_3_l883_88356

-- Define the inequality system
def inequality_system (x k : ℝ) : Prop :=
  x > -1 ∧ x ≤ 1 - k

-- Part 1
theorem part_1 :
  ∀ x : ℝ, inequality_system x (-2) ↔ -1 < x ∧ x ≤ 3 :=
by sorry

-- Part 2
theorem part_2 :
  (∀ x : ℝ, inequality_system x k ↔ -1 < x ∧ x ≤ 4) → k = -3 :=
by sorry

-- Part 3
theorem part_3 :
  (∃! x₁ x₂ x₃ : ℤ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    inequality_system (↑x₁) k ∧
    inequality_system (↑x₂) k ∧
    inequality_system (↑x₃) k) →
  -2 < k ∧ k ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_1_part_2_part_3_l883_88356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_difference_l883_88377

def is_arithmetic_progression (s : List ℝ) : Prop :=
  s.length > 1 ∧ ∀ i : Fin (s.length - 2), s[i.val + 1] - s[i.val] = s[i.val + 2] - s[i.val + 1]

theorem arithmetic_sequence_difference (a b c : ℝ) :
  is_arithmetic_progression [1, a, b, c, 9] →
  c - a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_difference_l883_88377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_relation_l883_88352

/-- Geometric sequence with first term 1 and fourth term 8 -/
noncomputable def geometric_sequence (n : ℕ) : ℝ :=
  let q := (8 : ℝ) ^ (1/3)
  q ^ (n - 1)

/-- Sum of the first 3n terms of the geometric sequence -/
noncomputable def S (n : ℕ) : ℝ :=
  let a₁ := 1
  let q := (8 : ℝ) ^ (1/3)
  a₁ * (1 - q^(3*n)) / (1 - q)

/-- Sum of the first n terms of the sequence {a_n^3} -/
noncomputable def T (n : ℕ) : ℝ :=
  let a₁ := 1
  let q := 8
  a₁ * (1 - q^n) / (1 - q)

/-- The theorem to be proved -/
theorem geometric_sequence_sum_relation :
  ∃ t : ℝ, ∀ n : ℕ, S n = t * T n ∧ t = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_relation_l883_88352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_and_sign_properties_l883_88336

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2)^x + 1/x

-- State the theorem
theorem root_and_sign_properties (x₀ x₁ x₂ : ℝ) 
  (h_root : f x₀ = 0)
  (h_order : x₁ < x₀ ∧ x₀ < x₂ ∧ x₂ < 0) :
  f x₁ > 0 ∧ f x₂ < 0 := by
  sorry

#check root_and_sign_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_and_sign_properties_l883_88336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nina_rainy_drive_time_l883_88332

/-- Represents Nina's driving scenario -/
structure DrivingScenario where
  sunny_speed : ℚ  -- Speed in sunny conditions (miles per hour)
  rainy_speed : ℚ  -- Speed in rainy conditions (miles per hour)
  total_distance : ℚ  -- Total distance driven (miles)
  total_time : ℚ  -- Total time driven (minutes)

/-- Calculates the time driven in rainy conditions -/
def rainy_time (scenario : DrivingScenario) : ℚ :=
  let sunny_speed_per_minute := scenario.sunny_speed / 60
  let rainy_speed_per_minute := scenario.rainy_speed / 60
  ((sunny_speed_per_minute * scenario.total_time - scenario.total_distance) /
   (sunny_speed_per_minute - rainy_speed_per_minute))

/-- Theorem stating that Nina drove approximately 27 minutes in the rain -/
theorem nina_rainy_drive_time :
  let scenario := DrivingScenario.mk 40 25 20 40
  Int.floor (rainy_time scenario) = 26 ∧ Int.ceil (rainy_time scenario) = 27 := by
  sorry

#eval Int.floor (rainy_time (DrivingScenario.mk 40 25 20 40))
#eval Int.ceil (rainy_time (DrivingScenario.mk 40 25 20 40))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nina_rainy_drive_time_l883_88332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_difference_is_2824_l883_88300

noncomputable def initial_loan : ℝ := 20000

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  principal * (1 + rate / n) ^ (n * t)

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (t : ℝ) : ℝ :=
  principal * (1 + rate * t)

noncomputable def compound_scheme (loan : ℝ) : ℝ :=
  let half_payment := compound_interest loan 0.08 2 6 / 2
  half_payment + compound_interest half_payment 0.08 2 6

noncomputable def simple_scheme (loan : ℝ) : ℝ :=
  simple_interest loan 0.1 12

theorem loan_difference_is_2824 :
  ⌊(simple_scheme initial_loan - compound_scheme initial_loan)⌋ = 2824 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_difference_is_2824_l883_88300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_G_has_inverse_l883_88363

-- Define the types for our functions
def Function : Type := ℝ → ℝ

-- Define the property of having an inverse
def HasInverse (f : Function) : Prop := ∃ g : Function, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Define our functions (we don't need to implement them, just declare them)
noncomputable def F : Function := sorry
noncomputable def G : Function := sorry
noncomputable def H : Function := sorry
noncomputable def I : Function := sorry
noncomputable def J : Function := sorry

-- State the theorem
theorem only_G_has_inverse :
  HasInverse G ∧ 
  ¬HasInverse F ∧ 
  ¬HasInverse H ∧ 
  ¬HasInverse I ∧ 
  ¬HasInverse J := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_G_has_inverse_l883_88363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounds_l883_88365

/-- An inscribed quadrilateral with a diagonal of length a forming angles α and β with two sides. -/
structure InscribedQuadrilateral where
  a : ℝ
  α : ℝ
  β : ℝ

/-- The area of an inscribed quadrilateral. -/
noncomputable def area (q : InscribedQuadrilateral) : ℝ :=
  sorry

/-- Theorem: The area of an inscribed quadrilateral is bounded. -/
theorem area_bounds (q : InscribedQuadrilateral) :
  (q.a^2 * Real.sin (q.α + q.β) * Real.sin q.β) / (2 * Real.sin q.α) ≤ 
  area q ∧
  area q ≤ 
  (q.a^2 * Real.sin (q.α + q.β) * Real.sin q.α) / (2 * Real.sin q.β) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounds_l883_88365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_covers_three_points_l883_88364

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A disk in 2D space -/
structure Disk where
  center : Point
  radius : ℝ

/-- A square in 2D space -/
structure Square where
  bottomLeft : Point
  sideLength : ℝ

noncomputable def pointsInDisk (d : Disk) (points : Finset Point) : Finset Point :=
  points.filter (fun p => (p.x - d.center.x)^2 + (p.y - d.center.y)^2 ≤ d.radius^2)

def pointInSquare (s : Square) (p : Point) : Prop :=
  s.bottomLeft.x ≤ p.x ∧ p.x ≤ s.bottomLeft.x + s.sideLength ∧
  s.bottomLeft.y ≤ p.y ∧ p.y ≤ s.bottomLeft.y + s.sideLength

theorem disk_covers_three_points (s : Square) (points : Finset Point) :
  s.sideLength = 7 →
  points.card = 51 →
  (∀ p ∈ points, pointInSquare s p) →
  ∃ d : Disk, d.radius = 1 ∧ (pointsInDisk d points).card ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_covers_three_points_l883_88364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_right_triangle_PZ_l883_88324

/-- A right triangle with a special interior point -/
structure SpecialRightTriangle where
  -- X, Y, Z are points in the real plane
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  -- P is a point in the real plane
  P : ℝ × ℝ
  -- Y is the right angle
  right_angle_at_Y : (X.1 - Y.1) * (Z.1 - Y.1) + (X.2 - Y.2) * (Z.2 - Y.2) = 0
  -- P is inside the triangle
  P_inside : sorry
  -- PX = 8
  PX_length : Real.sqrt ((P.1 - X.1)^2 + (P.2 - X.2)^2) = 8
  -- PY = 4
  PY_length : Real.sqrt ((P.1 - Y.1)^2 + (P.2 - Y.2)^2) = 4
  -- ∠XPY = ∠YPZ = ∠ZPX
  equal_angles : sorry

/-- The main theorem -/
theorem special_right_triangle_PZ (t : SpecialRightTriangle) :
  Real.sqrt ((t.P.1 - t.Z.1)^2 + (t.P.2 - t.Z.2)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_right_triangle_PZ_l883_88324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l883_88330

noncomputable def a (α : Real) : Real × Real := (4 * Real.cos α, Real.sin α)
noncomputable def b (β : Real) : Real × Real := (Real.sin β, 4 * Real.cos β)
noncomputable def c (β : Real) : Real × Real := (Real.cos β, -4 * Real.sin β)

def perpendicular (v w : Real × Real) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

noncomputable def magnitude (v : Real × Real) : Real :=
  Real.sqrt (v.1^2 + v.2^2)

theorem vector_problem (α β : Real) 
    (h : perpendicular (a α) (b β - 2 • c β)) :
  Real.tan (α + β) = 2 ∧ 
  0 ≤ magnitude (b β + c β) ∧ 
  magnitude (b β + c β) ≤ 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l883_88330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nuts_per_meter_l883_88307

/-- The number of squirrels in the tree -/
def num_squirrels : ℕ := 4

/-- The height of the tree in meters -/
def tree_height : ℕ := 10

/-- The number of acorns collected by each squirrel -/
def acorns_per_squirrel : ℕ := 2

/-- The number of walnuts collected by each squirrel -/
def walnuts_per_squirrel : ℕ := 3

/-- The number of walnuts stolen by the bird -/
def stolen_walnuts : ℕ := 5

/-- Theorem stating that the sum of nuts per meter of the tree's height is 1.5 -/
theorem nuts_per_meter :
  (num_squirrels * acorns_per_squirrel + 
   num_squirrels * walnuts_per_squirrel - stolen_walnuts : ℚ) / tree_height = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nuts_per_meter_l883_88307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l883_88384

/-- The profit percentage of revenues in the previous year -/
noncomputable def previous_profit_percentage : ℚ := 10

/-- The revenue reduction percentage in 2009 -/
noncomputable def revenue_reduction : ℚ := 20

/-- The profit percentage of revenues in 2009 -/
noncomputable def current_profit_percentage : ℚ := 12

/-- The ratio of profits in 2009 to profits in the previous year -/
noncomputable def profit_ratio : ℚ := 9600000000000001 / 10000000000000000

theorem profit_percentage_calculation :
  let previous_revenue : ℚ := 100 -- Assuming previous year's revenue is 100 for simplicity
  let current_revenue : ℚ := previous_revenue * (1 - revenue_reduction / 100)
  let previous_profit : ℚ := previous_revenue * (previous_profit_percentage / 100)
  let current_profit : ℚ := current_revenue * (current_profit_percentage / 100)
  current_profit = profit_ratio * previous_profit :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l883_88384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_single_value_l883_88341

/-- A polynomial that takes an infinite set of values at least twice at integer points -/
structure SpecialPolynomial (α : Type*) [CommRing α] where
  P : Polynomial α
  infinite_double_values : ∃ (S : Set α), Set.Infinite S ∧ ∀ y ∈ S, (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ P.eval (↑x₁) = y ∧ P.eval (↑x₂) = y)

/-- The main theorem statement -/
theorem at_most_one_single_value {α : Type*} [CommRing α] (P : SpecialPolynomial α) :
  ∃! y : α, ∃! x : ℤ, P.P.eval (↑x) = y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_single_value_l883_88341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_205_property_l883_88304

def binary_representation (n : ℕ) : List Bool := sorry

theorem binary_205_property :
  let bin_205 := binary_representation 205
  let u := (bin_205.filter (· = false)).length
  let v := (bin_205.filter (· = true)).length
  2 * (v - u) = 8 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_205_property_l883_88304
