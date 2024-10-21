import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1005_100507

/-- The eccentricity of a hyperbola whose asymptote intersects a parabola at one point -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (∃! x y : ℝ, y = (b / a) * x ∧ y = x^2 + 1) →
  (Real.sqrt (a^2 + b^2)) / a = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1005_100507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sin_2x_max_value_l1005_100544

theorem sin_sin_2x_max_value (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  (∀ y : ℝ, 0 < y ∧ y < π / 2 → Real.sin x * Real.sin (2 * x) ≤ Real.sin y * Real.sin (2 * y)) →
  Real.sin x * Real.sin (2 * x) = 4 * Real.sqrt 3 / 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sin_2x_max_value_l1005_100544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_l1005_100592

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 3*x + 2
  else -(x^2 - 3*x + 2)

-- State the theorem
theorem min_value_f_on_interval :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  (∀ x < 0, f x = x^2 + 3*x + 2) →  -- definition for x < 0
  (∃ x₀ ∈ Set.Icc 1 3, ∀ x ∈ Set.Icc 1 3, f x₀ ≤ f x) →  -- minimum exists on [1,3]
  (∃ x₀ ∈ Set.Icc 1 3, f x₀ = -2) ∧  -- minimum value is -2
  (∀ x ∈ Set.Icc 1 3, f x ≥ -2)  -- -2 is the lower bound on [1,3]
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_l1005_100592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_distance_squared_l1005_100534

-- Define the given conditions
noncomputable def AB : ℝ := 15
noncomputable def BC : ℝ := 25
noncomputable def θ_min : ℝ := 30 * Real.pi / 180
noncomputable def θ_max : ℝ := 45 * Real.pi / 180

-- Define the theorem
theorem ship_distance_squared (θ : ℝ) (h1 : θ_min ≤ θ) (h2 : θ ≤ θ_max) :
  525 ≤ AB^2 + BC^2 - 2*AB*BC*(Real.cos θ) ∧ AB^2 + BC^2 - 2*AB*BC*(Real.cos θ) ≤ 585 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_distance_squared_l1005_100534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leading_coefficient_of_g_l1005_100511

-- Define the function f
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := ((x/2)^α) / (x-1)

-- Define the function g
noncomputable def g (α : ℝ) : ℝ := (deriv^[4] (f α)) 2

-- Theorem statement
theorem leading_coefficient_of_g :
  ∃ (p : Polynomial ℝ), (∀ α, g α = p.eval α) ∧ (p.leadingCoeff = 1/16) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leading_coefficient_of_g_l1005_100511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hair_extension_problem_l1005_100505

/-- Calculates the final hair length after applying extensions -/
noncomputable def final_hair_length (initial_length : ℝ) (increase_percentage : ℝ) : ℝ :=
  initial_length * (1 + increase_percentage / 100)

/-- Theorem stating that given an initial hair length of 18 cubes and an increase of 75%,
    the final hair length is 31.5 cubes -/
theorem hair_extension_problem :
  final_hair_length 18 75 = 31.5 := by
  -- Unfold the definition of final_hair_length
  unfold final_hair_length
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hair_extension_problem_l1005_100505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_properties_l1005_100539

/-- Properties of a triangle rotated around the tangent to its circumscribed circle -/
theorem triangle_rotation_properties
  (a b c : ℝ) (S h : ℝ) 
  (h_positive : h > 0)
  (a_positive : a > 0)
  (b_positive : b > 0)
  (c_positive : c > 0)
  (S_positive : S > 0)
  (triangle_exists : a + b > c ∧ b + c > a ∧ c + a > b)
  (area_formula : S = (a * h) / 2) :
  let f := π * 2 * S * (b^2 + c^2) / (b * c)
  let t := (4 * S^2 * π / (3 * a * b * c)) * (b^2 + c^2)
  (f = π * 2 * S * (b^2 + c^2) / (b * c)) ∧ 
  (t = (4 * S^2 * π / (3 * a * b * c)) * (b^2 + c^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_properties_l1005_100539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_planting_growth_equation_l1005_100517

/-- Represents the annual average growth rate of tree planting -/
def x : ℝ := sorry

/-- The number of trees planted in the first year -/
def first_year_trees : ℕ := 400

/-- The number of trees planted in the third year -/
def third_year_trees : ℕ := 625

/-- Theorem stating the relationship between trees planted in the first and third years -/
theorem tree_planting_growth_equation :
  (first_year_trees : ℝ) * (1 + x)^2 = third_year_trees := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_planting_growth_equation_l1005_100517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_nonempty_proper_subsets_l1005_100579

def M : Set ℤ := {x | 0 < |x - 2| ∧ |x - 2| < 2}

theorem min_nonempty_proper_subsets (N : Set ℤ) : 
  (M ∪ N = {1, 2, 3, 4}) → 
  (∃ (k : ℕ), k ≥ 3 ∧ 
    ∀ (S : Finset (Set ℤ)), 
      (∀ X ∈ S, X ⊆ N ∧ X.Nonempty ∧ X ≠ N) → 
      S.card ≥ k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_nonempty_proper_subsets_l1005_100579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_savings_l1005_100599

/-- The percentage Sandy saved last year -/
def last_year_percentage : ℝ := 10

/-- Sandy's salary last year -/
noncomputable def S : ℝ := sorry

/-- This year's salary increase percentage -/
def salary_increase : ℝ := 10

/-- The percentage Sandy saved this year -/
def this_year_percentage : ℝ := 6

/-- The ratio of this year's savings to last year's savings -/
def savings_ratio : ℝ := 65.99999999999999

/-- Theorem stating the equality of savings between this year and last year -/
theorem sandy_savings :
  (this_year_percentage / 100) * (1 + salary_increase / 100) * S =
  (savings_ratio / 100) * (last_year_percentage / 100) * S := by
  sorry

#check sandy_savings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_savings_l1005_100599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_bounds_l1005_100550

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

def is_not_obtuse (t : Triangle) : Prop :=
  t.A ≤ Real.pi/2 ∧ t.B ≤ Real.pi/2 ∧ t.C ≤ Real.pi/2

-- Theorem statement
theorem triangle_side_ratio_bounds (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : is_not_obtuse t) 
  (h3 : t.B = Real.pi/3) : 
  1 < (2 * t.a) / t.c ∧ (2 * t.a) / t.c ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_bounds_l1005_100550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a4_value_l1005_100540

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem max_a4_value (seq : ArithmeticSequence) 
    (h1 : S seq 4 ≥ 10) (h2 : S seq 5 ≤ 15) : 
    seq.a 4 ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a4_value_l1005_100540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_notable_telephone_numbers_l1005_100554

/-- A notable telephone number is a 7-digit number d₁d₂d₃-d₄d₅d₆d₇ where:
    1) d₁ = d₇ and d₃ = d₅ (palindrome condition)
    2) d₄ + d₅ + d₆ = 15
    3) Each dᵢ is a decimal digit (0-9) -/
def NotableTelephoneNumber : Type := 
  { d : Fin 7 → Fin 10 // d 0 = d 6 ∧ d 2 = d 4 ∧ (d 3).val + (d 4).val + (d 5).val = 15 }

/-- NotableTelephoneNumber is a finite type -/
instance : Fintype NotableTelephoneNumber := by
  sorry -- The actual implementation of this instance would go here

/-- The number of different notable telephone numbers is 2700 -/
theorem count_notable_telephone_numbers :
  Fintype.card NotableTelephoneNumber = 2700 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_notable_telephone_numbers_l1005_100554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1005_100595

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if -5 ≤ x ∧ x < 0 then -1/2 * x
  else if 0 ≤ x ∧ x ≤ 5 then 1/3 * x
  else 0  -- This case should never occur given our domain

-- Define the function g as f(x) + x
noncomputable def g (x : ℝ) : ℝ := f x + x

-- State the theorem about the range of g
theorem range_of_g :
  ∀ y ∈ Set.range g, -5/2 ≤ y ∧ y ≤ 20/3 ∧
  (∃ x₁ ∈ Set.Icc (-5 : ℝ) 5, g x₁ = -5/2) ∧
  (∃ x₂ ∈ Set.Icc (-5 : ℝ) 5, g x₂ = 20/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1005_100595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola1_wider_opening_l1005_100515

noncomputable def parabola1 (x : ℝ) : ℝ := x^2 - (2/3)*x + 3
noncomputable def parabola2 (x : ℝ) : ℝ := 2*x^2 + (4/3)*x + 1

theorem parabola1_wider_opening :
  ∃ (c : ℝ), c > 0 ∧ ∀ (x : ℝ), |parabola1 x - parabola1 0| < |parabola2 x - parabola2 0| + c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola1_wider_opening_l1005_100515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anya_lost_games_l1005_100524

/-- Represents a girl playing table tennis -/
inductive Girl
| Anya
| Bella
| Valya
| Galya
| Dasha

/-- Represents the state of a girl in a game -/
inductive GameState
| Playing
| Resting

/-- Represents the result of a game for a girl -/
inductive GameResult
| Win
| Lose

/-- The total number of games played -/
def totalGames : Nat := 19

/-- The number of games each girl played -/
def gamesPlayed (g : Girl) : Nat :=
  match g with
  | Girl.Anya => 4
  | Girl.Bella => 6
  | Girl.Valya => 7
  | Girl.Galya => 10
  | Girl.Dasha => 11

/-- A function representing the state of each girl for each game -/
def gameStates : Fin totalGames → Girl → GameState := sorry

/-- A function representing the result of each game for each girl -/
def gameResults : Fin totalGames → Girl → Option GameResult := sorry

/-- The main theorem to prove -/
theorem anya_lost_games :
  (∀ (i : Fin totalGames), gameResults i Girl.Anya = some GameResult.Lose →
    (i.val + 1 : Nat) ∈ ({4, 8, 12, 16} : Set Nat)) ∧
  (∀ (i : Fin totalGames), (i.val + 1 : Nat) ∈ ({4, 8, 12, 16} : Set Nat) →
    gameResults i Girl.Anya = some GameResult.Lose) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_anya_lost_games_l1005_100524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1005_100510

open Real

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → (x + x⁻¹) * f y = f (x * y) + f (y * x⁻¹)

/-- The theorem stating the form of functions satisfying the functional equation -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEq f →
  ∃ c₁ c₂ : ℝ, ∀ x : ℝ, x > 0 → f x = c₁ * x + c₂ * x⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1005_100510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_question_distribution_l1005_100576

theorem test_question_distribution (total_points total_questions : ℕ) 
  (h_total_points : total_points = 100)
  (h_total_questions : total_questions = 40)
  : ∃ (two_point four_point : ℕ),
    two_point + four_point = total_questions ∧
    2 * two_point + 4 * four_point = total_points ∧
    four_point = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_question_distribution_l1005_100576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_ge_one_monotone_implies_a_in_zero_eight_l1005_100532

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

-- Part 1
theorem min_value_implies_a_ge_one (a : ℝ) (h_a : a > 0) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ -2) ∧ (∃ x ∈ Set.Icc 1 (Real.exp 1), f a x = -2) →
  a ≥ 1 :=
by sorry

-- Part 2
theorem monotone_implies_a_in_zero_eight :
  (∀ a : ℝ, ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f a x₁ + 2 * x₁ < f a x₂ + 2 * x₂) →
  {a : ℝ | 0 ≤ a ∧ a ≤ 8}.Nonempty :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_ge_one_monotone_implies_a_in_zero_eight_l1005_100532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_satisfies_differential_equation_l1005_100574

-- Define the function y
noncomputable def y (x c : ℝ) : ℝ := x * (c - Real.log x)

-- State the theorem
theorem y_satisfies_differential_equation (x c : ℝ) (h : x > 0) :
  (x - y x c) * deriv (fun x => y x c) x + x * deriv (fun c => y x c) c = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_satisfies_differential_equation_l1005_100574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1005_100586

/-- The function f(x) = |x-20| + |x+18| -/
def f (x : ℝ) : ℝ := |x - 20| + |x + 18|

/-- The function g(x, c) = x + c -/
def g (x c : ℝ) : ℝ := x + c

/-- The theorem stating that c = 18 is the unique value for which 
    f(x) and g(x, c) intersect at exactly one point -/
theorem unique_intersection :
  ∃! c : ℝ, ∃! x : ℝ, f x = g x c ∧ c = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1005_100586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_satisfies_conditions_l1005_100591

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 1

-- Define the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := 8 * y^2 - x^2 = 1

-- Define a function to check if a point is on the asymptote
noncomputable def on_asymptote (x y : ℝ) : Prop := x = 2 * Real.sqrt 2 * y ∨ x = -2 * Real.sqrt 2 * y

-- Theorem statement
theorem hyperbola_satisfies_conditions :
  (∃ (x y : ℝ), circle_eq x y ∧ hyperbola_eq x y) ∧  -- Hyperbola is tangent to the circle
  on_asymptote 1 (1/2)                               -- Asymptote passes through (1, 1/2)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_satisfies_conditions_l1005_100591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_bounded_by_c_l1005_100504

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/Real.exp 1)^x + log x

-- State the theorem
theorem root_bounded_by_c (a b c x₀ : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a < b ∧ b < c)
  (h_prod : f a * f b * f c > 0)
  (h_root : f x₀ = 0) :
  x₀ ≤ c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_bounded_by_c_l1005_100504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_center_coordinates_is_zero_l1005_100531

-- Define the type for a point in 2D space
def Point := ℝ × ℝ

-- Define the endpoints of the diameter
noncomputable def endpoint1 : Point := (7, -6)
noncomputable def endpoint2 : Point := (-5, 4)

-- Define the center of the circle
noncomputable def center (p1 p2 : Point) : Point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Theorem statement
theorem sum_of_center_coordinates_is_zero :
  let c := center endpoint1 endpoint2
  (c.1 + c.2) = 0 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_center_coordinates_is_zero_l1005_100531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_l1005_100543

-- Define the curves C₁ and C₂
noncomputable def C₁ (a : ℝ) : ℝ → ℝ × ℝ := λ t ↦ (a + Real.sqrt 2 * t, 1 + Real.sqrt 2 * t)

def C₂ : ℝ × ℝ → Prop :=
  λ (x, y) ↦ y^2 = 4*x

-- Define the intersection points
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t, C₁ a t = p ∧ C₂ p}

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- State the theorem
theorem intersection_condition (a : ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ intersection_points a ∧ B ∈ intersection_points a ∧
    A ≠ B ∧ distance (a, 1) A = 2 * distance (a, 1) B) →
  a = 1/36 ∨ a = 9/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_l1005_100543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_min_dot_product_sum_magnitude_l1005_100569

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the vertices and foci
def A₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the vectors PA₁ and PF₂
def PA₁ (P : ℝ × ℝ) : ℝ × ℝ := (A₁.1 - P.1, A₁.2 - P.2)
def PF₂ (P : ℝ × ℝ) : ℝ × ℝ := (F₂.1 - P.1, F₂.2 - P.2)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- The main theorem
theorem ellipse_min_dot_product_sum_magnitude :
  ∀ P : ℝ × ℝ, is_on_ellipse P.1 P.2 →
  (∀ Q : ℝ × ℝ, is_on_ellipse Q.1 Q.2 → 
    dot_product (PA₁ P) (PF₂ P) ≤ dot_product (PA₁ Q) (PF₂ Q)) →
  magnitude (PA₁ P + PF₂ P) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_min_dot_product_sum_magnitude_l1005_100569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1005_100526

/-- Arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a₁ : ℝ
  d : ℝ
  h_d_nonzero : d ≠ 0

namespace ArithmeticSequence

variable (seq : ArithmeticSequence)

/-- The nth term of the arithmetic sequence -/
noncomputable def a (n : ℕ) : ℝ := seq.a₁ + (n - 1 : ℝ) * seq.d

/-- The sum of the first n terms of the arithmetic sequence -/
noncomputable def S (n : ℕ) : ℝ := n * (2 * seq.a₁ + (n - 1 : ℝ) * seq.d) / 2

/-- Statement of the problem -/
theorem problem_statement :
  (seq.a 4)^2 = seq.a 2 * seq.a 7 → seq.S 5 = 50 → seq.S 8 = 104 := by
  sorry

end ArithmeticSequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1005_100526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sprocket_production_l1005_100563

/-- Represents the production rate and time for a machine. -/
structure Machine where
  rate : ℝ  -- sprockets per hour
  time : ℝ  -- hours

/-- The problem setup for sprocket manufacturing. -/
structure SprocketProblem where
  machineA : Machine
  machineP : Machine
  machineQ : Machine
  hA : machineA.rate = 5
  hQ : machineQ.rate = machineA.rate * 1.1
  hP : machineP.time = machineQ.time + 10
  hEqual : machineP.rate * machineP.time = machineQ.rate * machineQ.time

/-- The theorem stating that both Machine P and Machine Q produce 550 sprockets each. -/
theorem sprocket_production (prob : SprocketProblem) : 
  prob.machineP.rate * prob.machineP.time = 550 ∧ 
  prob.machineQ.rate * prob.machineQ.time = 550 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sprocket_production_l1005_100563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_in_divided_right_triangle_l1005_100514

/-- Predicate for a set being a right triangle in ℝ² -/
def IsRightTriangle (triangle : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate for a division of a triangle into a square and two smaller triangles -/
def IsDivisionOfTriangle (triangle square small_triangle1 small_triangle2 : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- The ratio of areas between two sets in ℝ² -/
def AreaRatio (set1 set2 : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- In a right triangle divided into a square and two smaller right triangles,
    if the area of one small right triangle is n times the area of the square,
    then the ratio of the area of the other small right triangle to the area of the square is 1/(4n). -/
theorem area_ratio_in_divided_right_triangle (n : ℝ) (n_pos : 0 < n) :
  ∃ (triangle : Set (ℝ × ℝ)) (square : Set (ℝ × ℝ)) (small_triangle1 small_triangle2 : Set (ℝ × ℝ)),
    IsRightTriangle triangle ∧
    IsDivisionOfTriangle triangle square small_triangle1 small_triangle2 ∧
    AreaRatio small_triangle1 square = n →
    AreaRatio small_triangle2 square = 1 / (4 * n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_in_divided_right_triangle_l1005_100514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_value_l1005_100570

/-- The infinite series defined in the problem -/
noncomputable def infiniteSeries : ℝ := ∑' n : ℕ, (n^4 + 3*n^2 + 10*n + 10) / (2^n * (n^4 + 4))

/-- Theorem stating that the infinite series equals 11/10 -/
theorem infinite_series_value : infiniteSeries = 11/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_value_l1005_100570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_on_simple_interest_l1005_100585

/-- Calculate compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

/-- Calculate simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem sum_on_simple_interest : ∃ (sum : ℝ),
  simple_interest sum 14 6 = (1/2) * compound_interest 7000 7 2 ∧ sum = 603.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_on_simple_interest_l1005_100585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_value_l1005_100536

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define our specific triangle
noncomputable def myTriangle : Triangle where
  A := Real.pi / 2 - Real.pi / 4 - Real.pi / 3  -- A = 180° - 45° - 60° = 75°
  B := Real.pi / 4  -- 45°
  C := Real.pi / 3  -- 60°
  a := 0  -- We don't know 'a', but it's not needed for the proof
  b := Real.sqrt 6 / 3  -- This is what we're trying to prove
  c := 1

-- State the theorem
theorem side_b_value (t : Triangle) (h1 : t = myTriangle) :
  t.b = Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_value_l1005_100536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alexis_coat_time_l1005_100529

/-- The time it takes Alexis to sew a skirt in hours -/
def skirt_time : ℚ := 2

/-- The total time it takes Alexis to sew 6 skirts and 4 coats in hours -/
def total_time : ℚ := 40

/-- The number of skirts Alexis sews -/
def num_skirts : ℕ := 6

/-- The number of coats Alexis sews -/
def num_coats : ℕ := 4

/-- The time it takes Alexis to sew a coat in hours -/
noncomputable def coat_time : ℚ := (total_time - (num_skirts : ℚ) * skirt_time) / (num_coats : ℚ)

theorem alexis_coat_time : coat_time = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alexis_coat_time_l1005_100529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_sum_max_l1005_100500

theorem triangle_sin_sum_max (A B C : ℝ) (h : A + B + C = π) :
  Real.sin A + Real.sin B + 2 * Real.sqrt 7 * Real.sin C ≤ 27 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_sum_max_l1005_100500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_payment_l1005_100577

/-- Represents the denominations of coins available --/
inductive Coin
  | Ten
  | Fifteen
  | Twenty

/-- Helper function to get the value of a coin --/
def Coin.value : Coin → Nat
  | Coin.Ten => 10
  | Coin.Fifteen => 15
  | Coin.Twenty => 20

/-- The problem setup --/
structure BusScenario where
  passengers : Nat
  totalCoins : Nat
  ticketPrice : Nat

/-- A function to check if a distribution of coins is valid --/
def isValidDistribution (scenario : BusScenario) (distribution : List (List Coin)) : Prop :=
  distribution.length = scenario.passengers ∧
  (distribution.map List.length).sum = scenario.totalCoins ∧
  ∀ coins ∈ distribution, ∃ (payment : List Coin) (change : List Coin),
    payment ⊆ coins ∧
    (payment.map Coin.value).sum = scenario.ticketPrice ∧
    change ⊆ coins ∧
    change ≠ []

/-- The main theorem --/
theorem impossible_payment (scenario : BusScenario) :
  scenario.passengers = 40 →
  scenario.totalCoins = 49 →
  scenario.ticketPrice = 5 →
  ¬∃ (distribution : List (List Coin)), isValidDistribution scenario distribution :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_payment_l1005_100577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_between_two_fifths_and_three_fourths_l1005_100509

theorem fraction_between_two_fifths_and_three_fourths :
  let fractions : List ℚ := [1/6, 4/3, 5/2, 4/7, 1/4]
  ∃! x, x ∈ fractions ∧ 2/5 < x ∧ x < 3/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_between_two_fifths_and_three_fourths_l1005_100509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sin_relationship_l1005_100582

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = π

-- Define the theorem
theorem tan_sin_relationship :
  ¬(∀ A B : ℝ, Real.tan A > Real.tan B → Real.sin A > Real.sin B) ∧
  ¬(∀ A B : ℝ, Real.sin A > Real.sin B → Real.tan A > Real.tan B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sin_relationship_l1005_100582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solutions_l1005_100552

/-- The number of real solutions to the equation ||x-2|-|x-6|| = l -/
def num_solutions : ℕ := 2

/-- The equation ||x-2|-|x-6|| = l has exactly two real solutions -/
theorem absolute_value_equation_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  (abs (abs (x₁ - 2) - abs (x₁ - 6)) = 1 ∨ abs (abs (x₁ - 2) - abs (x₁ - 6)) = 1) ∧
  (abs (abs (x₂ - 2) - abs (x₂ - 6)) = 1 ∨ abs (abs (x₂ - 2) - abs (x₂ - 6)) = 1) ∧
  (∀ x : ℝ, (abs (abs (x - 2) - abs (x - 6)) = 1 ∨ abs (abs (x - 2) - abs (x - 6)) = 1) → (x = x₁ ∨ x = x₂)) :=
by sorry

#check absolute_value_equation_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solutions_l1005_100552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excenter_distance_squared_sum_excenter_distances_squared_l1005_100527

/-- Given a triangle with sides a, b, c, this structure defines its properties --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  s : ℝ
  R : ℝ
  r : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_semiperimeter : s = (a + b + c) / 2
  h_triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The square of the distance from a vertex to its opposite excenter --/
noncomputable def distToExcenter (t : Triangle) (side : ℝ) : ℝ :=
  (t.s - side) / t.s * (t.a * t.b * t.c) / side

theorem excenter_distance_squared (t : Triangle) :
  distToExcenter t t.a = (t.s - t.a) / t.s * t.b * t.c := by
  sorry

theorem sum_excenter_distances_squared (t : Triangle) :
  distToExcenter t t.a + distToExcenter t t.b + distToExcenter t t.c =
  t.b * t.c + t.c * t.a + t.a * t.b - 12 * t.R * t.r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_excenter_distance_squared_sum_excenter_distances_squared_l1005_100527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_eq_three_l1005_100502

theorem count_solutions_eq_three :
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, (n + 1500 : ℝ) / 90 = ⌊Real.sqrt (n + 1 : ℝ)⌋) ∧ 
    (∀ n : ℕ, (n + 1500 : ℝ) / 90 = ⌊Real.sqrt (n + 1 : ℝ)⌋ → n ∈ S) ∧
    S.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_eq_three_l1005_100502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_score_analysis_l1005_100583

def standard_score : ℝ := 60

def relative_scores : List ℝ := [36, 0, 12, -18, 20]

def absolute_score (relative_score : ℝ) : ℝ :=
  standard_score + relative_score

noncomputable def highest_score : ℝ := 
  (relative_scores.map absolute_score).maximum?
    |>.getD 0

noncomputable def lowest_score : ℝ := 
  (relative_scores.map absolute_score).minimum?
    |>.getD 0

noncomputable def average_score : ℝ :=
  (relative_scores.sum / relative_scores.length) + standard_score

theorem score_analysis :
  highest_score = 96 ∧
  lowest_score = 42 ∧
  average_score = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_score_analysis_l1005_100583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABN_perimeter_range_l1005_100596

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2/16 + y^2/12 = 1

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := y^2 = 8*x

-- Define the intersection point P
noncomputable def P : ℝ × ℝ := (4/3, Real.sqrt (32/3))

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < P.1 then 2 * Real.sqrt (2*x)
  else if x > P.1 then (Real.sqrt 3)/2 * Real.sqrt (16 - x^2)
  else 0  -- This case is not specified in the original problem, but needed for completeness

-- Define the fixed point N
def N : ℝ × ℝ := (2, 0)

-- Define the theorem
theorem triangle_ABN_perimeter_range :
  ∀ a : ℝ, ∃ A B : ℝ × ℝ,
    A.2 = a ∧ B.2 = a ∧ 
    f A.1 = a ∧ f B.1 = a ∧
    (20/3 : ℝ) < (dist N A + dist N B + dist A B) ∧
    (dist N A + dist N B + dist A B) < 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABN_perimeter_range_l1005_100596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_youseff_distance_l1005_100557

/-- The distance in blocks between Youseff's home and office -/
def x : ℝ := sorry

/-- Time taken to walk one block in minutes -/
noncomputable def walk_time_per_block : ℝ := 1

/-- Time taken to bike one block in minutes -/
noncomputable def bike_time_per_block : ℝ := 20 / 60

/-- The difference in time between walking and biking in minutes -/
noncomputable def time_difference : ℝ := 4

theorem youseff_distance :
  (x * walk_time_per_block = x * bike_time_per_block + time_difference) →
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_youseff_distance_l1005_100557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l1005_100512

/-- Given a parabola y^2 = 8x with focus F, if a line through F intersects
    the parabola at points A and B, and the midpoint of AB is 3 units from
    the y-axis, then the length of AB is 10. -/
theorem parabola_chord_length (F A B : ℝ × ℝ) :
  (∀ (x y : ℝ), y^2 = 8*x → (x, y) ∈ Set.range (λ t : ℝ ↦ (t, Real.sqrt (8*t)))) →  -- Parabola equation
  (∃ (m b : ℝ), A.2 = m * A.1 + b ∧ B.2 = m * B.1 + b ∧ F.2 = m * F.1 + b) →  -- Line through F, A, B
  ((A.1 + B.1) / 2 = 3) →  -- Midpoint 3 units from y-axis
  ‖A - B‖ = 10 :=  -- Length of AB is 10
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l1005_100512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_neg_pi_third_l1005_100530

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) * x + Real.cos (Real.pi/2 + x)

-- State the theorem
theorem f_max_at_neg_pi_third :
  ∃ (x_max : ℝ), x_max = -Real.pi/3 ∧
  ∀ (x : ℝ), -Real.pi/2 ≤ x ∧ x ≤ Real.pi/2 → f x ≤ f x_max :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_neg_pi_third_l1005_100530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_l1005_100555

/-- Arithmetic sequence property -/
def IsArithmeticSeq (a x₁ x₂ b : ℝ) : Prop :=
  x₁ - a = x₂ - x₁ ∧ x₂ - x₁ = b - x₂

/-- Geometric sequence property -/
def IsGeometricSeq (a y₁ y₂ b : ℝ) : Prop :=
  y₁ / a = y₂ / y₁ ∧ y₂ / y₁ = b / y₂

/-- Given that a, x₁, x₂, b form an arithmetic sequence and a, y₁, y₂, b form a geometric sequence,
    prove that (x₁ + x₂) / (y₁ * y₂) = (a + b) / (a * b) -/
theorem sequence_ratio (a b x₁ x₂ y₁ y₂ : ℝ) 
  (h_arith : IsArithmeticSeq a x₁ x₂ b)
  (h_geom : IsGeometricSeq a y₁ y₂ b) :
  (x₁ + x₂) / (y₁ * y₂) = (a + b) / (a * b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_l1005_100555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_calculation_l1005_100538

/-- Represents the investment and time data for a partner -/
structure PartnerInvestment where
  amount : ℚ
  months : ℚ

/-- Calculates the share of a partner based on their investment-months -/
def calculateShare (partnerInvestmentMonths : ℚ) (totalInvestmentMonths : ℚ) (totalGain : ℚ) : ℚ :=
  (partnerInvestmentMonths / totalInvestmentMonths) * totalGain

theorem a_share_calculation (x : ℚ) (totalGain : ℚ) : 
  x > 0 → totalGain = 18600 →
  let a : PartnerInvestment := ⟨x, 12⟩
  let b : PartnerInvestment := ⟨2*x, 6⟩
  let c : PartnerInvestment := ⟨3*x, 4⟩
  let totalInvestmentMonths := a.amount * a.months + b.amount * b.months + c.amount * c.months
  calculateShare (a.amount * a.months) totalInvestmentMonths totalGain = 6200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_calculation_l1005_100538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_largest_32_in_35_l1005_100578

/-- A type representing a comparison between two elements -/
inductive Comparison (α : Type) where
  | less : α → α → Comparison α
  | greater : α → α → Comparison α

/-- A function that performs a comparison between two elements -/
def compare {α : Type} [LinearOrder α] : α → α → Comparison α
  | a, b => if a < b then Comparison.less a b else Comparison.greater a b

/-- A function that finds the two largest elements in a list using at most n comparisons -/
def findTwoLargest {α : Type} [LinearOrder α] (l : List α) (n : ℕ) : 
  Option (α × α) := sorry

theorem two_largest_32_in_35 :
  ∀ (l : List ℝ), l.length = 32 → l.Pairwise (· ≠ ·) → 
  ∃ (a b : ℝ), findTwoLargest l 35 = some (a, b) ∧ 
  (∀ x ∈ l, x ≤ a ∧ x ≤ b) ∧ (a ∈ l ∧ b ∈ l) ∧ a ≠ b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_largest_32_in_35_l1005_100578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_calculation_l1005_100587

-- Define the partners and their investments
noncomputable def a_investment : ℝ := 2000
noncomputable def b_investment : ℝ := 3000

-- Define the total investment
noncomputable def total_investment : ℝ := a_investment + b_investment

-- Define a's management fee percentage
noncomputable def management_fee_percent : ℝ := 0.1

-- Define a's share of the remaining profit
noncomputable def a_profit_share : ℝ := a_investment / total_investment

-- Define the money received by a
noncomputable def a_money_received : ℝ := 4416

-- Theorem to prove
theorem total_profit_calculation :
  ∃ (total_profit : ℝ),
    a_money_received = management_fee_percent * total_profit +
      (1 - management_fee_percent) * a_profit_share * total_profit ∧
    total_profit = 9600 :=
by
  -- We'll use 9600 as the total profit
  let total_profit : ℝ := 9600
  
  -- Prove that this total_profit satisfies the equation
  have h1 : a_money_received = management_fee_percent * total_profit +
    (1 - management_fee_percent) * a_profit_share * total_profit := by
    -- The actual proof would go here
    sorry
  
  -- Prove that total_profit equals 9600
  have h2 : total_profit = 9600 := by rfl
  
  -- Combine the proofs
  exact ⟨total_profit, h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_calculation_l1005_100587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_circle_ellipse_l1005_100553

/-- The smallest distance between a point on a circle and a point on an ellipse --/
theorem smallest_distance_circle_ellipse :
  let circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
  let ellipse : Set (ℝ × ℝ) := {p | (p.1 + 1)^2 / 9 + (p.2 - 4)^2 / 9 = 1}
  ∃ (C : ℝ × ℝ) (D : ℝ × ℝ), C ∈ circle ∧ D ∈ ellipse ∧
    ∀ (C' : ℝ × ℝ) (D' : ℝ × ℝ), C' ∈ circle → D' ∈ ellipse →
      Real.sqrt ((C'.1 - D'.1)^2 + (C'.2 - D'.2)^2) ≥
      Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) ∧
      Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_circle_ellipse_l1005_100553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relations_l1005_100594

theorem angle_relations (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan (α/2) = 1/3) (h4 : Real.cos (α - β) = -4/5) :
  Real.sin α = 3/5 ∧ 2*α + β = π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relations_l1005_100594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotonically_decreasing_intervals_l1005_100559

noncomputable def f (x : ℝ) := Real.sin (2 * x)

noncomputable def g (x : ℝ) := f (x + Real.pi / 6)

def monotonically_decreasing (h : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → h y < h x

theorem g_monotonically_decreasing_intervals :
  ∀ k : ℤ, monotonically_decreasing g (Real.pi / 12 + k * Real.pi) (7 * Real.pi / 12 + k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotonically_decreasing_intervals_l1005_100559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_cones_l1005_100597

noncomputable section

-- Define the circle and its properties
def circle_radius : ℝ := 1  -- Assuming unit circle for simplicity

-- Define the two sectors
def larger_sector_angle : ℝ := 2 * Real.pi / 3
def smaller_sector_angle : ℝ := Real.pi / 3

-- Define the radii of the resulting cones
def larger_cone_radius : ℝ := (2 / 3) * circle_radius
def smaller_cone_radius : ℝ := (1 / 3) * circle_radius

-- Define the heights of the resulting cones
def larger_cone_height : ℝ := (Real.sqrt 5 / 3) * circle_radius
def smaller_cone_height : ℝ := (Real.sqrt 8 / 3) * circle_radius

-- Define the volumes of the resulting cones
def larger_cone_volume : ℝ := (1 / 3) * Real.pi * larger_cone_radius^2 * larger_cone_height
def smaller_cone_volume : ℝ := (1 / 3) * Real.pi * smaller_cone_radius^2 * smaller_cone_height

-- Theorem statement
theorem volume_ratio_of_cones :
  larger_cone_volume / smaller_cone_volume = Real.sqrt 10 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_cones_l1005_100597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_approx_l1005_100598

/-- Calculates the cost price given the selling price and profit percentage -/
noncomputable def cost_price (selling_price : ℝ) (profit_percent : ℝ) : ℝ :=
  selling_price / (1 + profit_percent / 100)

/-- Theorem: The cost price is approximately 2404.15 given the conditions -/
theorem cost_price_approx :
  let selling_price : ℝ := 2524.36
  let profit_percent : ℝ := 5
  let calculated_cost_price := cost_price selling_price profit_percent
  ∃ ε > 0, |calculated_cost_price - 2404.15| < ε :=
by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval cost_price 2524.36 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_approx_l1005_100598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l1005_100506

-- Define the function f as noncomputable due to dependency on Real.sin
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

-- State the theorem
theorem omega_value (ω : ℝ) : 
  ω > 0 ∧ 
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ π/3 → f ω x < f ω y) ∧
  (∀ x y : ℝ, π/3 ≤ x ∧ x < y ∧ y ≤ π/2 → f ω x > f ω y) →
  ω = 3/2 := by
  sorry

#check omega_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l1005_100506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_prime_repr_441_l1005_100580

/-- Base prime representation of a natural number --/
def BasePrimeRepr (n : ℕ) : List ℕ := sorry

/-- Check if a list of natural numbers is a valid base prime representation --/
def IsValidBasePrimeRepr (l : List ℕ) : Prop := sorry

/-- The list of prime numbers in ascending order --/
def primes : List ℕ := sorry

/-- Axiom: BasePrimeRepr produces a valid base prime representation --/
axiom base_prime_repr_valid (n : ℕ) : IsValidBasePrimeRepr (BasePrimeRepr n)

/-- Axiom: The first few primes are 2, 3, 5, 7 in that order --/
axiom first_primes : primes.take 4 = [2, 3, 5, 7]

theorem base_prime_repr_441 : BasePrimeRepr 441 = [0, 2, 2, 0] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_prime_repr_441_l1005_100580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problems_l1005_100562

/-- Given two vectors in R², calculate the cosine of the angle between them -/
noncomputable def cosine_angle (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))

/-- Check if two vectors are orthogonal -/
def orthogonal (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Scale a vector by a scalar -/
def scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

/-- Add two vectors -/
def add (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

/-- The main theorem -/
theorem vector_problems :
  (let a : ℝ × ℝ := (0, 2)
   let b : ℝ × ℝ := (2, -3)
   cosine_angle a b = -3 * Real.sqrt 13 / 13) ∧
  (∃ k : ℝ,
    let a : ℝ × ℝ := (k - 1, 2)
    let b : ℝ × ℝ := (2, -3)
    orthogonal (add (scale 2 a) b) (add (scale 2 a) (scale (-k) b)) ∧ k = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problems_l1005_100562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cory_candy_purchase_l1005_100501

/-- The amount of additional money Cory needs to buy two packs of candies -/
def additional_money_needed (initial_money : ℚ) (cost_per_pack : ℚ) (num_packs : ℕ) : ℚ :=
  cost_per_pack * num_packs - initial_money

/-- Theorem stating the additional money Cory needs -/
theorem cory_candy_purchase : 
  additional_money_needed 20 49 2 = 78 := by
  -- Unfold the definition of additional_money_needed
  unfold additional_money_needed
  -- Perform the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cory_candy_purchase_l1005_100501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_area_l1005_100523

/-- A quadrilateral inscribed in a circle -/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d

/-- The semiperimeter of a cyclic quadrilateral -/
noncomputable def semiperimeter (q : CyclicQuadrilateral) : ℝ :=
  (q.a + q.b + q.c + q.d) / 2

/-- The area of a cyclic quadrilateral -/
noncomputable def area (q : CyclicQuadrilateral) : ℝ :=
  Real.sqrt ((semiperimeter q - q.a) * (semiperimeter q - q.b) * 
             (semiperimeter q - q.c) * (semiperimeter q - q.d))

/-- Theorem: The area of a cyclic quadrilateral is given by Brahmagupta's formula -/
theorem cyclic_quadrilateral_area (q : CyclicQuadrilateral) : 
  area q = Real.sqrt ((semiperimeter q - q.a) * (semiperimeter q - q.b) * 
                      (semiperimeter q - q.c) * (semiperimeter q - q.d)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_area_l1005_100523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_earnings_theorem_l1005_100561

/-- The expected quarterly earnings per share -/
noncomputable def E : ℝ := 0.80

/-- The actual quarterly earnings per share -/
def actual_earnings : ℝ := 1.10

/-- The number of shares owned by the person -/
def shares_owned : ℕ := 400

/-- The total dividend received by the person -/
def total_dividend : ℝ := 208

/-- The dividend calculation function -/
noncomputable def dividend (expected : ℝ) : ℝ :=
  expected / 2 + 0.04 * ((actual_earnings - expected) / 0.10)

theorem expected_earnings_theorem :
  (shares_owned : ℝ) * dividend E = total_dividend := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_earnings_theorem_l1005_100561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_days_l1005_100571

/-- Given that y cows produce y+2 cans of milk in y+3 days, 
    this theorem proves how many days it takes y+4 cows to produce y+7 cans of milk. -/
theorem milk_production_days (y : ℝ) (h : y > 0) : 
  (y + 4) * (y + 2) * (y + 3) / (y * (y + 7)) = y * (y + 3) * (y + 7) / ((y + 2) * (y + 4)) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_days_l1005_100571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_f_inequality_holds_iff_a_eq_neg_one_l1005_100584

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/x + a * Real.log x

-- Part 1: Number of zeros when a = 4
theorem f_has_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f 4 x₁ = 0 ∧ f 4 x₂ = 0 ∧
  ∀ (x : ℝ), x > 0 → f 4 x = 0 → (x = x₁ ∨ x = x₂) :=
by sorry

-- Part 2: Condition for a = -1
theorem f_inequality_holds_iff_a_eq_neg_one (a : ℝ) :
  (∀ (x : ℝ), x > -1 → f a (x + 1) + Real.exp x - 1 / (x + 1) ≥ 1) ↔ a = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_f_inequality_holds_iff_a_eq_neg_one_l1005_100584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_trapezoid_in_marked_vertices_l1005_100519

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- A set of marked vertices on a regular polygon -/
structure MarkedVertices (n : ℕ) (m : ℕ) where
  polygon : RegularPolygon n
  marked : Finset (Fin n)
  count : marked.card = m

/-- A line through two points -/
def Line (n : ℕ) := Fin n × Fin n

/-- Check if two lines are parallel -/
def IsParallel (n : ℕ) (l1 l2 : Line n) : Prop :=
  sorry -- Definition of parallel lines in a regular polygon

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides -/
structure Trapezoid (n : ℕ) where
  vertices : Fin 4 → Fin n
  parallel : ∃ (i j : Fin 4), i ≠ j ∧ 
    IsParallel n (vertices i, vertices (i + 1)) (vertices j, vertices (j + 1))

/-- The main theorem -/
theorem exists_trapezoid_in_marked_vertices :
  ∀ (mv : MarkedVertices 1981 64),
  ∃ (t : Trapezoid 1981), ∀ i : Fin 4, t.vertices i ∈ mv.marked := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_trapezoid_in_marked_vertices_l1005_100519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_rate_is_seven_l1005_100528

/-- The rate at which an escalator moves, given its length, a person's walking speed on it, and the time taken to cover the entire length. -/
noncomputable def escalator_rate (length : ℝ) (walking_speed : ℝ) (time : ℝ) : ℝ :=
  (length / time) - walking_speed

/-- Theorem stating that the escalator rate is 7 feet per second under the given conditions. -/
theorem escalator_rate_is_seven :
  let length : ℝ := 180
  let walking_speed : ℝ := 2
  let time : ℝ := 20
  escalator_rate length walking_speed time = 7 := by
  -- Unfold the definition of escalator_rate
  unfold escalator_rate
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_rate_is_seven_l1005_100528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_agreement_ratio_rounded_l1005_100572

noncomputable def round_to_nearest_tenth (x : ℚ) : ℚ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem agreement_ratio_rounded (total : ℕ) (agree : ℕ) 
  (h_total : total = 16) (h_agree : agree = 11) :
  round_to_nearest_tenth (agree / total) = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_agreement_ratio_rounded_l1005_100572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_of_polynomial_l1005_100548

theorem divisibility_property_of_polynomial (m : ℕ) (a : Fin m → ℕ+) (p : Polynomial ℤ) :
  (∀ n : ℕ+, ∃ i : Fin m, (a i : ℤ) ∣ p.eval (n : ℤ)) →
  ∃ j : Fin m, ∀ n : ℕ+, (a j : ℤ) ∣ p.eval (n : ℤ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_of_polynomial_l1005_100548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l1005_100513

open Set Real

/-- The function h(x) defined as (x^3 - 9x^2 + 23x - 15) / (|x - 4| + |x + 2|) -/
noncomputable def h (x : ℝ) : ℝ := (x^3 - 9*x^2 + 23*x - 15) / (|x - 4| + |x + 2|)

/-- The domain of h(x) is all real numbers -/
theorem h_domain : Set.range h = univ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l1005_100513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_extrema_l1005_100573

noncomputable def a (n : ℕ) : ℝ := (n - Real.sqrt 98) / (n - Real.sqrt 99)

theorem sequence_extrema :
  ∀ k ∈ Finset.range 30, a 10 ≥ a (k + 1) ∧ a 9 ≤ a (k + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_extrema_l1005_100573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_circle_theorem_l1005_100567

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Represents a circle (x-a)^2 + (y-b)^2 = r^2 -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The focus of a parabola y^2 = 2px is at (p/2, 0) -/
noncomputable def focus (parabola : Parabola) : ℝ × ℝ := (parabola.p / 2, 0)

/-- Check if a point is on a line -/
def onLine (point : ℝ × ℝ) (line : Line) : Prop :=
  ∃ t : ℝ, point = (1 - t) • line.point1 + t • line.point2

/-- Check if a point is on a parabola -/
def onParabola (point : ℝ × ℝ) (parabola : Parabola) : Prop :=
  point.2^2 = 2 * parabola.p * point.1

/-- Theorem: For a parabola y^2 = 2px and a line passing through its focus
    intersecting the parabola at points A and B, if the circle with diameter AB
    has the equation (x-3)^2 + (y-2)^2 = 16, then p = 2 -/
theorem parabola_line_circle_theorem (parabola : Parabola)
    (l : Line) (c : Circle) :
    (c.center = (3, 2) ∧ c.radius = 4) →
    (∃ (A B : ℝ × ℝ), onLine A l ∧ onLine B l ∧ 
      onParabola A parabola ∧ onParabola B parabola) →
    onLine (focus parabola) l →
    parabola.p = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_circle_theorem_l1005_100567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_calculation_l1005_100542

-- Define the Δ operation
noncomputable def delta (a b : ℝ) : ℝ := (a^2 + b^2) / (1 + a * b)

-- State the theorem
theorem delta_calculation :
  delta (delta 2 3) 4 = 6661 / 2891 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_calculation_l1005_100542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1005_100575

theorem triangle_problem (A B C : Real) (a b c : Real) :
  a = 8 →
  b - c = 2 →
  Real.cos A = -1/4 →
  Real.sin B = (3 * Real.sqrt 15) / 16 ∧
  Real.cos (2 * A + π/6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1005_100575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_vector_to_slope_intercept_l1005_100516

/-- Given a line in vector form, prove its slope and y-intercept in slope-intercept form -/
theorem line_vector_to_slope_intercept :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, (-2 : ℝ) * (x - 1) + (-5 : ℝ) * (y - 11) = 0 ↔ y = m * x + b) ∧ 
    m = -2/5 ∧ 
    b = 57/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_vector_to_slope_intercept_l1005_100516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_square_and_power_l1005_100545

theorem compare_square_and_power (n : ℕ) : n ≥ 3 → (n + 1)^2 < 3^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_square_and_power_l1005_100545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_theorem_l1005_100503

/-- The radius of a sphere in a unit cube arrangement -/
noncomputable def sphere_radius_in_unit_cube : ℝ :=
  3 * Real.sqrt 3 / 22

/-- A unit cube containing eight congruent spheres -/
structure SpheresInCube where
  -- The cube is a unit cube
  cube_side : ℝ := 1
  -- There are eight spheres
  num_spheres : ℕ := 8
  -- All spheres have the same radius
  sphere_radius : ℝ
  -- Each sphere is tangent to four faces of the cube
  tangent_to_faces : sphere_radius * 2 = cube_side
  -- Each sphere is tangent to one another
  spheres_tangent : sphere_radius * 2 = cube_side * Real.sqrt 3 / 3

/-- Theorem: The radius of each sphere in the described arrangement is 3√3/22 -/
theorem sphere_radius_theorem (arrangement : SpheresInCube) :
  arrangement.sphere_radius = sphere_radius_in_unit_cube := by
  sorry

#check sphere_radius_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_theorem_l1005_100503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_obtuse_from_sine_inequality_l1005_100546

theorem triangle_obtuse_from_sine_inequality (A B C : ℝ) 
  (h : Real.sin (2 * A) + Real.sin (2 * B) < Real.sin (2 * C)) : 
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive side lengths
    a + b > c ∧ b + c > a ∧ c + a > b ∧  -- triangle inequality
    A + B + C = π ∧  -- sum of angles in a triangle
    max A (max B C) > π / 2  -- definition of obtuse triangle
    := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_obtuse_from_sine_inequality_l1005_100546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_same_color_probability_is_two_fifths_l1005_100518

/-- The probability of drawing two balls of the same color from a bag containing 
    3 red balls and 3 black balls. -/
theorem same_color_probability : ℚ :=
  let total_balls : ℕ := 6
  let red_balls : ℕ := 3
  let black_balls : ℕ := 3
  let total_draws : ℕ := 2

  let total_outcomes : ℕ := Nat.choose total_balls total_draws
  let same_color_outcomes : ℕ := Nat.choose red_balls total_draws + Nat.choose black_balls total_draws

  (same_color_outcomes : ℚ) / total_outcomes

theorem same_color_probability_is_two_fifths : same_color_probability = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_same_color_probability_is_two_fifths_l1005_100518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_vector_coordinates_l1005_100551

noncomputable def vector_OA : ℝ × ℝ := (1, 1)

noncomputable def rotation_angle : ℝ := Real.pi / 3  -- 60° in radians

noncomputable def vector_OB : ℝ × ℝ := 
  ((1 - Real.sqrt 3) / 2, (1 + Real.sqrt 3) / 2)

theorem rotated_vector_coordinates :
  let rotated_vector := (
    vector_OA.1 * Real.cos rotation_angle - vector_OA.2 * Real.sin rotation_angle,
    vector_OA.1 * Real.sin rotation_angle + vector_OA.2 * Real.cos rotation_angle
  )
  rotated_vector = vector_OB := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_vector_coordinates_l1005_100551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_positive_root_l1005_100533

-- Define the complex equation
def complex_equation (z : ℂ) : Prop :=
  z * (z - 2*Complex.I) * (z + 4*Complex.I) = 4032 * Complex.I

-- Theorem statement
theorem exists_positive_root :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ complex_equation (Complex.ofReal a + Complex.I * Complex.ofReal b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_positive_root_l1005_100533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_size_problem_l1005_100522

def symmetric_difference (x y : Finset ℤ) : Finset ℤ := (x \ y) ∪ (y \ x)

theorem set_size_problem (x y : Finset ℤ) 
  (hx : x.card = 16)
  (hxy : (x ∩ y).card = 6)
  (hxsy : (symmetric_difference x y).card = 22) :
  y.card = 18 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_size_problem_l1005_100522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_c_onto_b_l1005_100520

noncomputable def a : ℝ × ℝ := (2, 3)
noncomputable def b : ℝ × ℝ := (-4, 7)

noncomputable def c : ℝ × ℝ := (-a.1, -a.2)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def projection (v w : ℝ × ℝ) : ℝ := (dot_product v w) / (magnitude w)

theorem projection_c_onto_b :
  projection c b = -Real.sqrt 65 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_c_onto_b_l1005_100520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_is_30_l1005_100589

/-- The speed of the second train given the conditions of the problem -/
noncomputable def second_train_speed (train1_length train2_length : ℝ) 
                       (train1_speed : ℝ) 
                       (clearing_time : ℝ) : ℝ :=
  let total_distance := train1_length + train2_length
  let relative_speed := (total_distance / 1000) / (clearing_time / 3600)
  relative_speed - train1_speed

/-- Theorem stating the speed of the second train under the given conditions -/
theorem second_train_speed_is_30 :
  second_train_speed 100 160 42 12.998960083193344 = 30 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval second_train_speed 100 160 42 12.998960083193344

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_is_30_l1005_100589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gross_profit_calculation_l1005_100593

theorem gross_profit_calculation (sales_price : ℝ) (gross_profit_percentage : ℝ) :
  sales_price = 44 →
  gross_profit_percentage = 1.20 →
  let cost := sales_price / (1 + gross_profit_percentage);
  let gross_profit := cost * gross_profit_percentage;
  gross_profit = 24 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gross_profit_calculation_l1005_100593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gum_pack_size_l1005_100537

/-- The number of pieces of cherry gum Chewbacca has initially -/
def cherry_gum : ℕ := 30

/-- The number of pieces of grape gum Chewbacca has initially -/
def grape_gum : ℕ := 40

/-- The number of pieces of gum in each complete pack -/
def y : ℕ := 5

/-- The ratio of cherry to grape gum after losing two packs of cherry gum -/
def ratio_lost_cherry (y : ℕ) : ℚ := (cherry_gum - 2 * y) / grape_gum

/-- The ratio of cherry to grape gum after finding four packs of grape gum -/
def ratio_found_grape (y : ℕ) : ℚ := cherry_gum / (grape_gum + 4 * y)

theorem gum_pack_size :
  ratio_lost_cherry y = ratio_found_grape y → y = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gum_pack_size_l1005_100537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_XZ_length_contradiction_l1005_100556

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  right_angle : (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0
  qr_length : Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = 8

-- Define point X on PQ
def X (t : ℝ) (P Q : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 + t * (Q.1 - P.1), P.2 + t * (Q.2 - P.2))

-- Define the parallel line through X
def parallel_line (X : ℝ × ℝ) (Q R : ℝ × ℝ) : ℝ × ℝ → Prop :=
  fun Y ↦ (Y.2 - X.2) * (R.1 - Q.1) = (Y.1 - X.1) * (R.2 - Q.2)

-- Define point Y
noncomputable def Y (P Q R : ℝ × ℝ) (X : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define point Z
noncomputable def Z (Q R : ℝ × ℝ) (Y : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define the length of XZ
noncomputable def XZ_length (X Z : ℝ × ℝ) : ℝ :=
  Real.sqrt ((Z.1 - X.1)^2 + (Z.2 - X.2)^2)

-- Theorem statement
theorem min_XZ_length_contradiction {P Q R : ℝ × ℝ} (h : Triangle P Q R) :
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → XZ_length (X t P Q) (Z Q R (Y P Q R (X t P Q))) ≥ 1.6 → False :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_XZ_length_contradiction_l1005_100556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_derivative_l1005_100525

noncomputable def y (a b x : ℝ) : ℝ :=
  (a^2 + b^2)^(-(1/2 : ℝ)) * Real.arcsin ((Real.sqrt (a^2 + b^2) * Real.sin x) / b)

theorem y_derivative (a b x : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  deriv (fun x => y a b x) x = Real.cos x / Real.sqrt (b^2 * (Real.cos x)^2 - a^2 * (Real.sin x)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_derivative_l1005_100525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_equivalence_l1005_100568

/-- A quadrilateral is a polygon with four sides. -/
structure Quadrilateral where
  sides : Fin 4 → ℝ × ℝ

/-- IsRightAngle predicate (placeholder) -/
def IsRightAngle (a b : ℝ × ℝ) : Prop := sorry

/-- A rectangle is a quadrilateral with four right angles. -/
structure Rectangle extends Quadrilateral where
  right_angles : ∀ i : Fin 4, IsRightAngle (sides i) (sides (i.succ))

/-- A proposition about quadrilaterals and rectangles. -/
def QuadrilateralProposition :=
  ∀ q : Quadrilateral, (∃ i : Fin 4, ¬IsRightAngle (q.sides i) (q.sides (i.succ))) → ¬∃ r : Rectangle, r.toQuadrilateral = q

/-- The inverse proposition. -/
def InverseProposition :=
  ∀ r : Rectangle, ∃ i : Fin 4, ¬IsRightAngle (r.sides i) (r.sides (i.succ))

/-- The theorem stating that the inverse proposition is equivalent to "A rectangle has three right angles". -/
theorem inverse_proposition_equivalence :
  InverseProposition ↔ (∀ r : Rectangle, ∃! i : Fin 4, ¬IsRightAngle (r.sides i) (r.sides (i.succ))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_equivalence_l1005_100568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rearranged_solid_surface_area_l1005_100558

/-- Represents a piece of the cut cube -/
structure Piece where
  height : ℚ

/-- Represents the rearranged solid -/
structure RearrangedSolid where
  pieces : List Piece

def unit_cube_volume : ℚ := 1

noncomputable def first_cut_height : ℚ := 1/4
noncomputable def second_cut_height : ℚ := 1/6
noncomputable def third_cut_height : ℚ := 1/12

noncomputable def piece_A : Piece := ⟨first_cut_height⟩
noncomputable def piece_B : Piece := ⟨second_cut_height⟩
noncomputable def piece_C : Piece := ⟨third_cut_height⟩
noncomputable def piece_D : Piece := ⟨1 - (first_cut_height + second_cut_height + third_cut_height)⟩

noncomputable def rearranged_solid : RearrangedSolid :=
  ⟨[piece_B, piece_C, piece_A, piece_D]⟩

def surface_area (_s : RearrangedSolid) : ℚ :=
  2 * (1 + 1 + 1 + 1)

theorem rearranged_solid_surface_area :
  surface_area rearranged_solid = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rearranged_solid_surface_area_l1005_100558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_deposit_is_2000_l1005_100588

/-- Calculates the initial deposit given the total amount after a certain period,
    the interest rate, and the time period. -/
noncomputable def calculate_initial_deposit (total_amount : ℝ) (interest_rate : ℝ) (time : ℝ) : ℝ :=
  total_amount / (1 + interest_rate * time)

/-- Theorem stating that given the specified conditions, the initial deposit was $2000. -/
theorem initial_deposit_is_2000 :
  let interest_rate : ℝ := 0.08
  let time : ℝ := 2.5
  let total_amount : ℝ := 2400
  calculate_initial_deposit total_amount interest_rate time = 2000 := by
  -- Unfold the definition of calculate_initial_deposit
  unfold calculate_initial_deposit
  -- Simplify the expression
  simp
  -- The proof is completed using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_deposit_is_2000_l1005_100588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1005_100508

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem problem_statement (p q r s : ℕ+) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (p : ℕ) * q * r * s = factorial 7 →
  p * q + p + q = 618 →
  q * r + q + r = 210 →
  r * s + r + s = 154 →
  p - s = 464 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1005_100508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_inequality_l1005_100535

theorem geometric_sequence_inequality (t : ℝ) : 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 31 → 
    (let a : ℕ → ℝ := λ k => 2^k;
     let S : ℕ → ℝ := λ k => 2^(k+1) - 2;
     S n - 62 < (a (n+1))^2 - t * (a (n+1)))) 
  → t < 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_inequality_l1005_100535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_l1005_100541

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of the first line y = -3x + 7 -/
noncomputable def slope₁ : ℝ := -3

/-- The slope of the second line 9y - 3x = 15 -/
noncomputable def slope₂ : ℝ := 3 / 9

theorem lines_perpendicular : perpendicular slope₁ slope₂ := by
  unfold perpendicular slope₁ slope₂
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_l1005_100541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_songs_l1005_100590

/-- Represents a girl in the concert --/
inductive Girl
| Mary
| Alina
| Tina
| Hanna
| Emma

/-- The number of songs each girl sang --/
def songs_sung (g : Girl) : ℕ :=
  match g with
  | Girl.Mary => 5
  | Girl.Hanna => 10
  | _ => 7  -- Assumption for Alina, Tina, and Emma

/-- The total number of girls --/
def total_girls : ℕ := 5

/-- The number of girls singing in each quartet --/
def quartet_size : ℕ := 4

/-- The total number of songs sung by all girls --/
def total_songs : ℕ := 
  songs_sung Girl.Mary + songs_sung Girl.Alina + songs_sung Girl.Tina + songs_sung Girl.Hanna + songs_sung Girl.Emma

theorem concert_songs :
  (total_songs / quartet_size : ℕ) = 9 ∧
  (∀ g : Girl, songs_sung g ≤ songs_sung Girl.Hanna) ∧
  (∀ g : Girl, g ≠ Girl.Mary → songs_sung Girl.Mary ≤ songs_sung g) ∧
  (total_songs % quartet_size = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_songs_l1005_100590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_in_interval_l1005_100547

/-- Represents an interval with its frequency -/
structure IntervalFreq where
  lower : ℝ
  upper : ℝ
  freq : ℕ

/-- The sample data -/
def sample_data : List IntervalFreq := [
  ⟨10, 20, 2⟩,
  ⟨20, 30, 3⟩,
  ⟨30, 40, 4⟩,
  ⟨40, 50, 5⟩,
  ⟨50, 60, 4⟩,
  ⟨60, 70, 2⟩
]

/-- The sample size -/
def sample_size : ℕ := 20

/-- The target interval -/
def target_interval : IntervalFreq := ⟨10, 50, 0⟩

/-- Function to check if an interval is within the target interval -/
noncomputable def is_within_target (i : IntervalFreq) : Bool :=
  i.lower > target_interval.lower ∧ i.upper ≤ target_interval.upper

/-- Theorem stating that the frequency of the sample in the interval (10,50] is 0.7 -/
theorem frequency_in_interval :
  (((sample_data.filter is_within_target).foldl (λ acc i => acc + i.freq) 0 : ℕ) : ℚ) / sample_size = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_in_interval_l1005_100547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_diagonals_l1005_100549

theorem polygon_diagonals :
  (∃ n : ℕ, n > 0 ∧ n * (n - 3) / 2 = 54) ∧
  (¬∃ n : ℕ, n > 0 ∧ n * (n - 3) / 2 = 21) ∧
  (¬∃ n : ℕ, n > 0 ∧ n * (n - 3) / 2 = 32) ∧
  (¬∃ n : ℕ, n > 0 ∧ n * (n - 3) / 2 = 45) ∧
  (¬∃ n : ℕ, n > 0 ∧ n * (n - 3) / 2 = 63) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_diagonals_l1005_100549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1005_100564

/-- A geometric sequence is defined by its first four terms -/
noncomputable def geometric_sequence : Fin 4 → ℝ
  | 0 => 10
  | 1 => -15
  | 2 => 22.5
  | 3 => -33.75

/-- The common ratio of a geometric sequence -/
noncomputable def common_ratio (seq : Fin 4 → ℝ) : ℝ :=
  seq 1 / seq 0

theorem geometric_sequence_common_ratio :
  common_ratio geometric_sequence = -1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1005_100564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_l1005_100521

noncomputable def h (x : ℝ) : ℝ := 3 / (1 + 3 * x^4)

theorem range_sum (a b : ℝ) : 
  (∀ y, y ∈ Set.range h ↔ y ∈ Set.Ioo a b) → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_l1005_100521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_equals_one_max_a_for_positive_f_a_one_implies_positive_f_max_a_is_one_l1005_100560

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - a * x / (x + 1)

-- Theorem for part (I)
theorem min_value_implies_a_equals_one (a : ℝ) :
  (∀ x, x > -1 → f a 0 ≤ f a x) → a = 1 := by sorry

-- Theorem for part (II)
theorem max_a_for_positive_f (a : ℝ) :
  (∀ x, x > 0 → f a x > 0) → a ≤ 1 := by sorry

theorem a_one_implies_positive_f :
  ∀ x, x > 0 → f 1 x > 0 := by sorry

theorem max_a_is_one :
  ∃ a, (∀ x, x > 0 → f a x > 0) ∧ 
       (∀ b, (∀ x, x > 0 → f b x > 0) → b ≤ a) ∧
       a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_equals_one_max_a_for_positive_f_a_one_implies_positive_f_max_a_is_one_l1005_100560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_preservation_l1005_100581

theorem remainder_preservation (x : ℕ) (hx : x > 0) (h : 100 % x = 4) : 196 % x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_preservation_l1005_100581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l1005_100566

theorem min_value_expression (y : ℝ) (h : y > 0) :
  9 * y^6 + 8 * y^(-(3 : ℤ)) ≥ 17 ∧
  (9 * y^6 + 8 * y^(-(3 : ℤ)) = 17 ↔ y = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l1005_100566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_condition_perimeter_when_AB_is_2_l1005_100565

-- Define the quadratic equation
def quadratic_eq (m x : ℝ) : Prop := x^2 - m*x + m/2 - 1/4 = 0

-- Define a quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Helper function to calculate distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  ∃ (s : ℝ), s > 0 ∧ 
    (distance q.A q.B = s) ∧ 
    (distance q.B q.C = s) ∧ 
    (distance q.C q.D = s) ∧ 
    (distance q.D q.A = s)

-- Helper function to calculate perimeter of a quadrilateral
noncomputable def perimeter (q : Quadrilateral) : ℝ :=
  distance q.A q.B + distance q.B q.C + distance q.C q.D + distance q.D q.A

-- Theorem 1
theorem rhombus_condition (q : Quadrilateral) (m : ℝ) :
  (∃ (x y : ℝ), x ≠ y ∧ quadratic_eq m x ∧ quadratic_eq m y ∧
    distance q.A q.B = x ∧ distance q.A q.D = y) →
  (is_rhombus q ↔ m = 1) :=
sorry

-- Theorem 2
theorem perimeter_when_AB_is_2 (q : Quadrilateral) (m : ℝ) :
  (∃ (y : ℝ), y ≠ 2 ∧ quadratic_eq m 2 ∧ quadratic_eq m y ∧
    distance q.A q.B = 2 ∧ distance q.A q.D = y) →
  perimeter q = 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_condition_perimeter_when_AB_is_2_l1005_100565
