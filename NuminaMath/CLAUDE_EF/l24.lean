import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_symmetry_axis_parabola_y_comparison_l24_2415

noncomputable def parabola (b : ℝ) (x y : ℝ) : Prop := y = -x^2 + b*x

noncomputable def axis_of_symmetry (b : ℝ) : ℝ := b / 2

theorem parabola_symmetry_axis (b : ℝ) (m n : ℝ) :
  (parabola b 1 0) →
  (axis_of_symmetry b = 1/2) ∧
  ((parabola b 1 m ∧ parabola b 2 n ∧ m*n < 0) →
   (1/2 < axis_of_symmetry b ∧ axis_of_symmetry b < 1)) :=
by sorry

-- Additional theorem for the comparison of y-values
theorem parabola_y_comparison (b t y₁ y₂ y₃ : ℝ) :
  (1/2 < t ∧ t < 1) →
  parabola b (-1) y₁ →
  parabola b (3/2) y₂ →
  parabola b 3 y₃ →
  y₃ < y₁ ∧ y₁ < y₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_symmetry_axis_parabola_y_comparison_l24_2415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_sqrt_radii_equals_147_14_l24_2444

/-- Represents a circle in the upper half-plane -/
structure Circle where
  radius : ℝ
  x_coord : ℝ

/-- Represents a layer of circles -/
def Layer := List Circle

/-- Constructs a new circle tangent to two given circles -/
noncomputable def new_circle (c1 c2 : Circle) : Circle :=
  { radius := (c1.radius * c2.radius) / ((Real.sqrt c1.radius + Real.sqrt c2.radius) ^ 2),
    x_coord := 0 }  -- x-coordinate is not used in this problem, so we set it to 0

/-- Constructs the next layer given the previous layers -/
noncomputable def next_layer (prev_layers : List Layer) : Layer :=
  sorry  -- Implementation details omitted

/-- Constructs all layers up to the 7th layer -/
noncomputable def construct_layers : List Layer :=
  let l0 : Layer := [{ radius := 100^2, x_coord := 0 }, { radius := 105^2, x_coord := 0 }]
  sorry  -- Implementation details omitted

/-- Calculates the sum of 1/sqrt(r(C)) for all circles in the given layers -/
noncomputable def sum_inverse_sqrt_radii (layers : List Layer) : ℝ :=
  sorry  -- Implementation details omitted

/-- The main theorem to be proved -/
theorem sum_inverse_sqrt_radii_equals_147_14 :
  sum_inverse_sqrt_radii construct_layers = 147 / 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_sqrt_radii_equals_147_14_l24_2444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_value_l24_2497

theorem square_value (p : ℝ) (square : ℝ) (h1 : square + p = 75) (h2 : (square + p) + p = 142) : square = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_value_l24_2497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_7_l24_2434

def T : ℕ → ℕ
  | 0 => 3  -- Add this case to handle Nat.zero
  | 1 => 3
  | n + 2 => 3^(T (n + 1))

theorem t_50_mod_7 : T 50 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_7_l24_2434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_fraction_sum_l24_2443

/-- Definition of collinearity for three points in ℝ² -/
def collinear (points : List (ℝ × ℝ)) : Prop :=
  match points with
  | [p1, p2, p3] => 
      let (x1, y1) := p1
      let (x2, y2) := p2
      let (x3, y3) := p3
      (x2 - x1) * (y3 - y1) = (x3 - x1) * (y2 - y1)
  | _ => False

/-- Given three collinear points A(2,2), B(a,0), and C(0,b) where ab ≠ 0, 
    prove that 1/a + 1/b = 1/2 -/
theorem collinear_points_fraction_sum (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : collinear [(2, 2), (a, 0), (0, b)]) : 
  1 / a + 1 / b = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_fraction_sum_l24_2443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_not_parabola_l24_2479

/-- A curve represented by the equation x^2 + ky^2 = 1, where k is any real number -/
def Curve (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + k * p.2^2 = 1}

/-- Predicate stating that a set of points in ℝ² forms a parabola -/
def IsParabola (S : Set (ℝ × ℝ)) : Prop := sorry

/-- The statement that the curve cannot be a parabola for any real k -/
theorem curve_not_parabola : ∀ k : ℝ, ¬ IsParabola (Curve k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_not_parabola_l24_2479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l24_2458

noncomputable def f (x : ℝ) : ℝ := x^2 / (x^2 + 1)

theorem range_of_f : 
  let domain : Set ℝ := {0, 1}
  let range := Set.range (f ∘ (coe : domain → ℝ))
  range = {0, 1/2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l24_2458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l24_2440

/-- The probability of drawing two chips of different colors from a bag with replacement -/
theorem different_color_probability (total_chips : ℕ) (red_chips : ℕ) (green_chips : ℕ) 
  (h1 : total_chips = red_chips + green_chips)
  (h2 : red_chips = 6)
  (h3 : green_chips = 4) :
  (red_chips : ℚ) / total_chips * (green_chips : ℚ) / total_chips +
  (green_chips : ℚ) / total_chips * (red_chips : ℚ) / total_chips = 12 / 25 := by
  sorry

#check different_color_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l24_2440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l24_2494

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 6*x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 6

-- Theorem statement
theorem tangent_line_equation 
  (x₀ : ℝ) 
  (h1 : f' x₀ = 3) : 
  ∃ (y₀ : ℝ), y₀ = f x₀ ∧ 
  (λ (x y : ℝ) => 3*x - y - 1 = 0) = 
  (λ (x y : ℝ) => y - y₀ = 3*(x - x₀)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l24_2494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_dot_product_sum_l24_2492

-- Part 1
theorem vector_magnitude (a b c : ℝ × ℝ) : 
  a = (1, 0) → b = (-1, 1) → c = (2 * a.1 + b.1, 2 * a.2 + b.2) → 
  Real.sqrt ((c.1)^2 + (c.2)^2) = Real.sqrt 2 := by sorry

-- Part 2
theorem dot_product_sum (a b : ℝ × ℝ) :
  Real.sqrt ((a.1)^2 + (a.2)^2) = 2 →
  Real.sqrt ((b.1)^2 + (b.2)^2) = 1 →
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt ((a.1)^2 + (a.2)^2) * Real.sqrt ((b.1)^2 + (b.2)^2)) = 1/2 →
  a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_dot_product_sum_l24_2492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_negative_implies_obtuse_l24_2402

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = Real.pi
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define an obtuse triangle
def isObtuse (t : Triangle) : Prop :=
  t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2

-- Theorem statement
theorem cosine_product_negative_implies_obtuse (t : Triangle) :
  (Real.cos t.A) * (Real.cos t.B) * (Real.cos t.C) < 0 → isObtuse t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_negative_implies_obtuse_l24_2402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eaten_black_squares_count_l24_2493

-- Define a chessboard as an 8x8 grid
def Chessboard := Fin 8 → Fin 8 → Bool

-- Define a function to determine if a square is black
def isBlack (row col : Fin 8) : Bool :=
  (row.val + col.val) % 2 == 0

-- Define the pattern of eaten squares
def isEaten (row col : Fin 8) : Bool :=
  -- This is a simplified representation of the eaten pattern
  -- You may need to adjust this based on the exact diagram
  (row.val ≤ 3 && col.val ≤ 5) || (row.val ≤ 2 && col.val ≤ 6)

-- Count the number of eaten black squares
def countEatenBlackSquares (board : Chessboard) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 8)) fun row =>
    Finset.sum (Finset.univ : Finset (Fin 8)) fun col =>
      if isEaten row col && isBlack row col then 1 else 0)

-- The theorem to prove
theorem eaten_black_squares_count :
  ∀ (board : Chessboard), countEatenBlackSquares board = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eaten_black_squares_count_l24_2493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l24_2407

theorem trigonometric_problem (x : ℝ) 
  (h1 : -π/2 < x) (h2 : x < 0) (h3 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x - Real.cos x = -7/5) ∧ (4 * Real.sin x * Real.cos x - Real.cos x^2 = -64/25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l24_2407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_problem_solution_l24_2456

/-- Represents the bicycle problem scenario -/
structure BicycleProblem where
  totalDistance : ℝ
  speedA : ℝ
  speedB : ℝ
  delayB : ℝ

/-- Calculates the times when the two cyclists are 200 meters apart -/
def timesSeparatedBy200Meters (problem : BicycleProblem) : Set ℝ :=
  { t | (t ≥ problem.delayB) ∧
        ((problem.speedA * t - problem.speedB * (t - problem.delayB) = 200) ∨
         (problem.speedB * (t - problem.delayB) - problem.speedA * t = 200)) }

/-- The main theorem stating the solution to the bicycle problem -/
theorem bicycle_problem_solution (problem : BicycleProblem) 
  (h1 : problem.totalDistance = 3000)
  (h2 : problem.speedA = 120)
  (h3 : problem.speedB = 200)
  (h4 : problem.delayB = 5) :
  timesSeparatedBy200Meters problem = {10, 15} := by
  sorry

-- Remove the #eval statement as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_problem_solution_l24_2456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fraction_is_three_fourths_l24_2476

/-- Represents the composition of a cement mixture -/
structure CementMixture where
  totalWeight : ℝ
  sandFraction : ℝ
  gravelWeight : ℝ

/-- Calculates the fraction of water in the cement mixture -/
noncomputable def waterFraction (mixture : CementMixture) : ℝ :=
  1 - mixture.sandFraction - mixture.gravelWeight / mixture.totalWeight

/-- Theorem stating that the water fraction is 3/4 for the given mixture -/
theorem water_fraction_is_three_fourths :
  let mixture : CementMixture := {
    totalWeight := 120,
    sandFraction := 1/5,
    gravelWeight := 6
  }
  waterFraction mixture = 3/4 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fraction_is_three_fourths_l24_2476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_under_8000_l24_2488

structure City where
  name : String
  deriving Repr, DecidableEq

def distance : City → City → ℕ := sorry

def cities : Finset City := sorry

axiom city_count : cities.card = 5

axiom valid_distances : ∀ c1 c2 : City, c1 ∈ cities → c2 ∈ cities → c1 ≠ c2 → distance c1 c2 > 0

def city_pairs : Finset (City × City) :=
  (cities.product cities).filter (λ p => p.1 ≠ p.2)

def pairs_under_8000 : Finset (City × City) :=
  city_pairs.filter (λ p => distance p.1 p.2 < 8000)

theorem probability_under_8000 :
  (pairs_under_8000.card : ℚ) / city_pairs.card = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_under_8000_l24_2488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_l24_2467

/-- The matrix A as defined in the problem -/
def A (n : ℕ) : Matrix (Fin n) (Fin n) ℤ :=
  λ i j => if i = j then 2 else (-1) ^ (Int.natAbs (i.val - j.val))

/-- The theorem stating that the determinant of A is n + 1 -/
theorem det_A (n : ℕ) : Matrix.det (A n) = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_l24_2467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_sixty_l24_2487

theorem factors_of_sixty : 
  let n : ℕ := 60
  let prime_factorization := (2^2 * 3 * 5 : ℕ)
  (Finset.filter (λ x : ℕ ↦ x ∣ n) (Finset.range (n + 1))).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_sixty_l24_2487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l24_2426

def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | 1 < (2:ℝ)^x ∧ (2:ℝ)^x < 2}

theorem union_of_M_and_N : M ∪ N = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l24_2426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_picnic_men_count_l24_2457

theorem picnic_men_count (total : ℕ) (men_women_diff : ℕ) (adults_children_diff : ℕ) : ℕ := by
  let men := total / 2 - men_women_diff / 2
  have h1 : total = 240 := by sorry
  have h2 : men_women_diff = 40 := by sorry
  have h3 : adults_children_diff = 40 := by sorry
  have h4 : men = total / 2 - men_women_diff / 2 := by sorry
  exact 90

#check picnic_men_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_picnic_men_count_l24_2457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_sum_l24_2425

-- Define the complex equations
def equation1 (z : ℂ) : Prop := z^2 = 5 + 5 * Complex.I * Real.sqrt 21
def equation2 (z : ℂ) : Prop := z^2 = 3 + 3 * Complex.I * Real.sqrt 7

-- Define the solutions to the equations
def solutions : Set ℂ := {z : ℂ | equation1 z ∨ equation2 z}

-- Define the parallelogram formed by the solutions
def parallelogram : Set ℂ := {z : ℂ | ∃ (a b : ℂ), a ∈ solutions ∧ b ∈ solutions ∧ z = a + b}

-- Define the area of the parallelogram
noncomputable def area : ℝ := Complex.abs (2 * (Real.sqrt 26 - Real.sqrt 22))

-- Define the property of q and s not being divisible by the square of any prime
def not_divisible_by_square_prime (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ n)

-- Main theorem
theorem parallelogram_area_sum : 
  ∃ (p q r s : ℕ), 
    area = p * Real.sqrt q - r * Real.sqrt s ∧ 
    not_divisible_by_square_prime q ∧
    not_divisible_by_square_prime s ∧
    p + q + r + s = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_sum_l24_2425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_revenue_formula_min_extraction_formula_l24_2499

/-- Annual extraction from company profits (in million yuan) -/
def annual_extraction : ℝ → ℝ := id

/-- Annual return rate on extracted amount -/
noncomputable def return_rate : ℝ := 0.1

/-- Time period in years -/
def time_period : ℕ := 3

/-- Total revenue after 3 years given annual extraction -/
noncomputable def total_revenue (A : ℝ) : ℝ :=
  A * (1 + (1 + return_rate) + (1 + return_rate)^2)

/-- Minimum annual extraction needed for a target total revenue -/
noncomputable def min_extraction (target : ℝ) : ℝ :=
  target / (1 + (1 + return_rate) + (1 + return_rate)^2)

theorem total_revenue_formula (A : ℝ) :
  total_revenue A = A * (1 + (1 + return_rate) + (1 + return_rate)^2) :=
by sorry

theorem min_extraction_formula (target : ℝ) :
  min_extraction target = target / (1 + (1 + return_rate) + (1 + return_rate)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_revenue_formula_min_extraction_formula_l24_2499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_plane_implies_planes_perp_l24_2435

/-- A plane in 3D space -/
structure Plane where

/-- A line in 3D space -/
structure Line where

/-- Subset relation between a line and a plane -/
def Line.subset_of (l : Line) (p : Plane) : Prop := sorry

/-- Perpendicular relation between a line and a plane -/
def Line.perp (l : Line) (p : Plane) : Prop := sorry

/-- Perpendicular relation between two planes -/
def Plane.perp (p1 p2 : Plane) : Prop := sorry

/-- Theorem: If a line is perpendicular to a plane and is a subset of another plane, 
    then the two planes are perpendicular -/
theorem line_perp_plane_implies_planes_perp 
  (α β : Plane) (l : Line) 
  (h1 : l.subset_of α) 
  (h2 : l.perp β) : 
  α.perp β := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_plane_implies_planes_perp_l24_2435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_product_shift_sine_shift_main_theorem_l24_2412

open Real

theorem sine_cosine_product_shift (x : ℝ) :
  2 * sin (x + π/6) * cos (x + π/6) = sin (2 * (x + π/6)) :=
by sorry

theorem sine_shift (x : ℝ) :
  sin (2 * (x + π/6)) = sin (2*x + π/3) :=
by sorry

theorem main_theorem (x : ℝ) :
  2 * sin (x + π/6) * cos (x + π/6) = sin (2*x + π/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_product_shift_sine_shift_main_theorem_l24_2412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_symmetric_about_point_l24_2431

noncomputable def f (x : ℝ) : ℝ := Real.sin (5 * x + Real.pi / 4) + Real.sqrt 3 * Real.cos (5 * x + Real.pi / 4)

theorem not_symmetric_about_point : 
  ∃ y, f (7 * Real.pi / 60 - y) ≠ f (7 * Real.pi / 60 + y) - 2 * f (7 * Real.pi / 60) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_symmetric_about_point_l24_2431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pell_tan_series_sum_l24_2405

/-- Pell numbers sequence -/
def PellNumber : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2 * PellNumber (n + 1) + PellNumber n

/-- The infinite series involving Pell numbers and inverse tangent -/
noncomputable def PellTanSeries : ℝ := ∑' n, (Real.arctan (1 / (PellNumber (2 * n) : ℝ)) + Real.arctan (1 / (PellNumber (2 * n + 2) : ℝ))) * Real.arctan (2 / (PellNumber (2 * n + 1) : ℝ))

/-- Theorem stating that the infinite series equals (arctan(1/2))^2 -/
theorem pell_tan_series_sum : PellTanSeries = (Real.arctan (1 / 2)) ^ 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pell_tan_series_sum_l24_2405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_room_carpet_area_l24_2477

/-- Calculates the area of an L-shaped room in square yards -/
noncomputable def l_shaped_room_area (length1 length2 width feet_per_yard : ℝ) : ℝ :=
  let yard_length1 := length1 / feet_per_yard
  let yard_length2 := length2 / feet_per_yard
  let yard_width := width / feet_per_yard
  yard_length1 * yard_width + yard_length2 * yard_width

theorem l_shaped_room_carpet_area :
  l_shaped_room_area 12 6 9 3 = 18 := by
  unfold l_shaped_room_area
  norm_num
  -- The proof is completed by normalization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_room_carpet_area_l24_2477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_proof_l24_2410

def a (n : ℕ) : ℚ := (2 - 2*n) / (3 + 4*n)

noncomputable def N (ε : ℚ) : ℕ := ⌈(7 + 2*ε) / (8*ε)⌉.toNat

theorem limit_proof (ε : ℚ) (hε : ε > 0) :
  ∀ n ≥ N ε, |a n - (-1/2)| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_proof_l24_2410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_from_cosine_and_point_l24_2489

noncomputable section

open Real

theorem sine_value_from_cosine_and_point 
  (α : ℝ) 
  (m : ℝ) 
  (h1 : m ≠ 0) 
  (h2 : Real.cos α = m / 13) 
  (h3 : ∃ (P : ℝ × ℝ), P.1 = m ∧ P.2 = 5 ∧ P ∈ Set.Iio 0) : 
  Real.sin α = 5 / 13 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_from_cosine_and_point_l24_2489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_l24_2485

noncomputable def sequence_a (a₁ : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => (3 * sequence_a a₁ n) / (2 + sequence_a a₁ n)

theorem sequence_limit (a₁ : ℝ) (h1 : 0 < a₁) (h2 : a₁ ≠ 1) :
  ∃ (l : ℝ), l = 1 ∧ Filter.Tendsto (sequence_a a₁) Filter.atTop (nhds l) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_l24_2485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_when_perpendicular_to_perpendicular_plane_l24_2428

/-- A plane in 3D space -/
structure Plane where

/-- A line in 3D space -/
structure Line where

/-- Perpendicularity between planes -/
def perpendicular_planes (α β : Plane) : Prop := sorry

/-- Perpendicularity between a line and a plane -/
def perpendicular_line_plane (m : Line) (α : Plane) : Prop := sorry

/-- Line not contained in a plane -/
def line_not_in_plane (m : Line) (α : Plane) : Prop := sorry

/-- Parallelism between a line and a plane -/
def parallel_line_plane (m : Line) (α : Plane) : Prop := sorry

/-- Theorem: If α ⊥ β, m ⊥ β, and m ⊈ α, then m ∥ α -/
theorem line_parallel_to_plane_when_perpendicular_to_perpendicular_plane 
  (α β : Plane) (m : Line) 
  (h1 : perpendicular_planes α β) 
  (h2 : perpendicular_line_plane m β) 
  (h3 : line_not_in_plane m α) : 
  parallel_line_plane m α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_when_perpendicular_to_perpendicular_plane_l24_2428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_remaining_credit_l24_2463

noncomputable def credit_limit : ℝ := 100
noncomputable def bakery_purchase : ℝ := 20
noncomputable def tuesday_payment : ℝ := 15
noncomputable def thursday_payment : ℝ := 23

noncomputable def discount_rate (total_spent : ℝ) : ℝ :=
  if total_spent ≥ 100 then 0.08
  else if total_spent ≥ 75 then 0.05
  else if total_spent ≥ 50 then 0.03
  else 0

noncomputable def calculate_discount (amount : ℝ) (rate : ℝ) : ℝ :=
  amount * rate

noncomputable def remaining_credit : ℝ :=
  let total_spent := credit_limit + bakery_purchase
  let discount_rate := discount_rate total_spent
  let grocery_discount := calculate_discount credit_limit discount_rate
  let bakery_discount := calculate_discount bakery_purchase discount_rate
  let total_after_discount := credit_limit + bakery_purchase - grocery_discount - bakery_discount
  let total_paid := tuesday_payment + thursday_payment
  total_after_discount - total_paid

theorem mary_remaining_credit :
  remaining_credit = 72.40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_remaining_credit_l24_2463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_theta_l24_2449

theorem tan_pi_minus_theta (θ : Real) (h1 : Real.sin θ = -3/4) (h2 : π < θ ∧ θ < 2*π) : 
  Real.tan (π - θ) = 3 * Real.sqrt 7 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_theta_l24_2449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_collinearity_l24_2418

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points on the plane
def Point := ℝ × ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- ax + by + c = 0

-- Function to check if a point lies on a circle
def lies_on_circle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Function to check if a line is tangent to a circle
noncomputable def is_tangent (l : Line) (c : Circle) : Prop :=
  sorry  -- Definition of tangency

-- Function to find the intersection of two lines
noncomputable def intersection (l1 l2 : Line) : Point :=
  sorry  -- Intersection calculation

-- Function to check if three points are collinear
def collinear (p1 p2 p3 : Point) : Prop :=
  sorry  -- Collinearity check

-- Main theorem
theorem circle_tangent_collinearity 
  (c : Circle) 
  (A B C D S : Point) 
  (SA SD : Line) 
  (h1 : lies_on_circle A c)
  (h2 : lies_on_circle B c)
  (h3 : lies_on_circle C c)
  (h4 : lies_on_circle D c)
  (h5 : is_tangent SA c)
  (h6 : is_tangent SD c)
  : let AB := Line.mk 1 1 0  -- Placeholder for line AB
    let CD := Line.mk 1 1 0  -- Placeholder for line CD
    let AC := Line.mk 1 1 0  -- Placeholder for line AC
    let BD := Line.mk 1 1 0  -- Placeholder for line BD
    let P := intersection AB CD
    let Q := intersection AC BD
    collinear P Q S :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_collinearity_l24_2418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l24_2484

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2*x else -((-x)^2 + 2*(-x))

-- State the theorem
theorem range_of_a (h_odd : ∀ x, f (-x) = -f x) 
  (h_nonneg : ∀ x ≥ 0, f x = x^2 + 2*x) :
  Set.Ioo (-2 : ℝ) 1 = {a : ℝ | f (2 - a^2) > f a} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l24_2484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_perpendicular_vectors_l24_2406

-- Define the points and vectors
def A : ℝ × ℝ × ℝ := (-2, 0, 2)
def B (t : ℝ) : ℝ × ℝ × ℝ := (t - 2, 4, 3)
def C (s : ℝ) : ℝ × ℝ × ℝ := (-4, s, 1)

def vec_a (t : ℝ) : ℝ × ℝ × ℝ := (t, 4, 1)
def vec_b (s : ℝ) : ℝ × ℝ × ℝ := (-2, s, -1)

-- Part 1
theorem collinear_points (t s : ℝ) :
  t = 2 →
  (∃ (k : ℝ), vec_a t = k • vec_b s) →
  s = -4 :=
by sorry

-- Part 2
theorem perpendicular_vectors (t s : ℝ) :
  s = 2 →
  (vec_a t + vec_b s) • (vec_b s) = 0 →
  t = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_perpendicular_vectors_l24_2406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_diameter_ratio_for_equal_area_circles_l24_2437

/-- The ratio of radius to diameter for circles with equal area -/
theorem radius_diameter_ratio_for_equal_area_circles (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h_area : π * x^2 = π * y^2) : 
  y / (2 * y) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_diameter_ratio_for_equal_area_circles_l24_2437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_45_l24_2498

/-- Represents a fish with upstream and downstream speeds -/
structure Fish where
  upstreamSpeed : ℚ
  downstreamSpeed : ℚ

/-- Calculates the still water speed of a fish -/
def stillWaterSpeed (f : Fish) : ℚ := (f.upstreamSpeed + f.downstreamSpeed) / 2

/-- The set of all fish -/
def allFish : List Fish := 
  [ Fish.mk 40 60,  -- Fish A
    Fish.mk 30 50,  -- Fish B
    Fish.mk 45 65,  -- Fish C
    Fish.mk 35 55,  -- Fish D
    Fish.mk 25 45 ] -- Fish E

/-- The average speed of all fish in still water -/
def averageStillWaterSpeed : ℚ := 
  (allFish.map stillWaterSpeed).sum / allFish.length

theorem average_speed_is_45 : averageStillWaterSpeed = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_45_l24_2498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_three_sum_l24_2401

theorem multiples_of_three_sum (seq : List ℕ) : 
  (seq.sum = 72) →
  (seq.maximum? = some 27) →
  (∀ n ∈ seq, n % 3 = 0) →
  (seq.length = 3) →
  (List.iota 3).all (λ i => i + 1 < seq.length → seq[i + 1]! = seq[i]! + 3) →
  seq = [21, 24, 27] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_three_sum_l24_2401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_perfect_square_l24_2480

def divisors_of_2010_squared : ℕ := 81

def perfect_square_divisors : ℕ := 16

theorem probability_one_perfect_square : 
  (perfect_square_divisors * (divisors_of_2010_squared - perfect_square_divisors) : ℚ) / 
  (divisors_of_2010_squared.choose 2) = 26 / 81 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_perfect_square_l24_2480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_black_is_81_512_l24_2470

/-- Represents a 3x3 grid where each cell can be either black or white -/
def Grid := Fin 3 → Fin 3 → Bool

/-- Probability of a single cell being black initially -/
noncomputable def initial_black_prob : ℝ := 1 / 2

/-- Rotates the grid 180 degrees -/
def rotate (g : Grid) : Grid :=
  fun i j => g (2 - i) (2 - j)

/-- Applies the blackening rule to the grid after rotation -/
def blacken (g : Grid) : Grid :=
  fun i j => sorry  -- Implementation details omitted for brevity

/-- The probability of the grid being entirely black after rotation and blackening -/
noncomputable def prob_all_black (g : Grid) : ℝ :=
  sorry  -- Implementation details omitted for brevity

theorem prob_all_black_is_81_512 :
  ∀ g : Grid, prob_all_black g = 81 / 512 := by
  sorry

#check prob_all_black_is_81_512

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_black_is_81_512_l24_2470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l24_2421

theorem equation_solution : ∃ x : ℝ, (2 : ℝ) ^ ((16 : ℝ) ^ x) = (16 : ℝ) ^ ((2 : ℝ) ^ x) ∧ x = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l24_2421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l24_2414

/-- The focus of a parabola y = ax² + bx + c is at (h, k + 1/(4a)) where (h, k) is the vertex -/
noncomputable def focus_of_parabola (a b c : ℝ) : ℝ × ℝ :=
  let h := -b / (2 * a)
  let k := c - b^2 / (4 * a)
  (h, k + 1 / (4 * a))

/-- Theorem: The focus of the parabola y = 2x² - 8x + 1 is at (2, -55/8) -/
theorem focus_of_specific_parabola :
  focus_of_parabola 2 (-8) 1 = (2, -55/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l24_2414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l24_2452

noncomputable def f (x m : ℝ) : ℝ := (1/3) * x^3 - x + m

theorem min_value_of_f (m : ℝ) :
  (∃ x₀ : ℝ, ∀ x : ℝ, f x m ≤ f x₀ m) ∧ (∃ x₁ : ℝ, f x₁ m = 1) →
  ∃ x₂ : ℝ, f x₂ m = -1/3 ∧ ∀ x : ℝ, f x m ≥ f x₂ m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l24_2452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_hemisphere_radius_largest_hemisphere_fits_l24_2473

/-- An oblique circular cone with height and base radius both equal to 1 -/
structure ObliqueCone where
  height : ℝ := 1
  baseRadius : ℝ := 1
  isOblique : height = baseRadius

/-- A hemisphere with its center on the base plane of the cone -/
structure Hemisphere where
  radius : ℝ
  centerOnBase : ℝ × ℝ

/-- The hemisphere fits inside the cone -/
def fitsInside (h : Hemisphere) (c : ObliqueCone) : Prop :=
  let (x, _) := h.centerOnBase
  h.radius ≤ min x ((c.height - x) / Real.sqrt 5)

/-- The largest hemisphere that fits inside the cone -/
noncomputable def largestHemisphere (c : ObliqueCone) : Hemisphere :=
  { radius := (Real.sqrt 5 - 1) / 2,
    centerOnBase := ((Real.sqrt 5 - 1) / 2, 0) }

/-- The theorem stating that the largest hemisphere has radius (√5 - 1) / 2 -/
theorem largest_hemisphere_radius (c : ObliqueCone) :
  (largestHemisphere c).radius = (Real.sqrt 5 - 1) / 2 :=
by sorry

/-- The theorem stating that the largest hemisphere fits inside the cone -/
theorem largest_hemisphere_fits (c : ObliqueCone) :
  fitsInside (largestHemisphere c) c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_hemisphere_radius_largest_hemisphere_fits_l24_2473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_on_interval_l24_2413

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := x / (x + 2)

-- State the theorem
theorem f_range_on_interval :
  ∀ x ∈ Set.Icc 2 4, 1/2 ≤ f x ∧ f x ≤ 2/3 := by
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_on_interval_l24_2413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_function_l24_2461

-- Define the regions M and N
noncomputable def M (x y : ℝ) : Prop := y ≥ 0 ∧ y ≤ x ∧ y ≤ 2 - x

noncomputable def N (t x : ℝ) : Prop := t ≤ x ∧ x ≤ t + 1 ∧ 0 ≤ t ∧ t ≤ 1

-- Define the common area function f(t)
noncomputable def f (t : ℝ) : ℝ := -t^2 + t + 1/2

-- Theorem statement
theorem common_area_function (t : ℝ) : 
  (∀ x y, M x y → N t x → (∃ a, a = f t)) := by
  sorry

#check common_area_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_function_l24_2461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_2pi_l24_2417

theorem cos_alpha_minus_2pi (α : ℝ) (h1 : Real.sin (π + α) = 4/5) (h2 : π < α ∧ α < 3*π/2) :
  Real.cos (α - 2*π) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_2pi_l24_2417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_floor_value_l24_2424

theorem fraction_floor_value : ⌊((3012 : ℝ) - 2933)^2 / 196⌋ = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_floor_value_l24_2424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_l24_2411

/-- Represents the round trip between A-ville and B-town -/
structure RoundTrip where
  speed_to_b : ℚ
  speed_from_b : ℚ
  time_to_b_minutes : ℚ

/-- Calculates the total time for the round trip in hours -/
noncomputable def total_time (trip : RoundTrip) : ℚ :=
  let time_to_b := trip.time_to_b_minutes / 60
  let distance := trip.speed_to_b * time_to_b
  let time_from_b := distance / trip.speed_from_b
  time_to_b + time_from_b

/-- The theorem stating that the total time for the given trip is 5 hours -/
theorem round_trip_time (trip : RoundTrip) 
    (h1 : trip.speed_to_b = 90)
    (h2 : trip.speed_from_b = 160)
    (h3 : trip.time_to_b_minutes = 192) :
  total_time trip = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_l24_2411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_4752_l24_2423

theorem largest_prime_factor_of_4752 :
  (Nat.factors 4752).maximum? = some 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_4752_l24_2423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colors_for_connected_l24_2420

/-- A chess board -/
structure Board :=
  (size : ℕ)

/-- A coloring of a chess board -/
def Coloring (b : Board) := Fin b.size → Fin b.size → ℕ

/-- Whether a color is connected in a given coloring -/
def is_connected (b : Board) (c : Coloring b) (color : ℕ) : Prop :=
  ∃ x y, c x y = color ∧
  (∀ x' y', c x' y' = color →
    ∃ path : List (Fin b.size × Fin b.size),
      path.head? = some (x, y) ∧
      path.getLast? = some (x', y') ∧
      ∀ (i j : Fin b.size), (i, j) ∈ path → c i j = color)

/-- The main theorem -/
theorem max_colors_for_connected (b : Board) (h : b.size = 2009) :
  ∃ n : ℕ, n = 4017 ∧
  (∀ m : ℕ, m ≤ n →
    ∃ c : Coloring b, ∀ color, color < m → ¬is_connected b c color) ∧
  (∀ m : ℕ, m > n →
    ∀ c : Coloring b, ∃ color, color < m ∧ is_connected b c color) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colors_for_connected_l24_2420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l24_2466

/-- Given two vectors in 2D space, calculate the area of the triangle formed by these vectors and the origin. -/
noncomputable def triangle_area (v₁ v₂ : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((v₁.1 * v₂.2) - (v₁.2 * v₂.1))

/-- Prove that the area of triangle ABC is 5, given the vectors AB and AC. -/
theorem area_of_triangle_ABC :
  let AB : ℝ × ℝ := (4, 2)
  let AC : ℝ × ℝ := (3, 4)
  triangle_area AB AC = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l24_2466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_8_equals_301_l24_2455

def sequence_a : ℕ → ℚ
| 0 => 428
| n + 1 => (1 / 2) * sequence_a n + 150

theorem a_8_equals_301 : sequence_a 7 = 301 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_8_equals_301_l24_2455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l24_2400

/-- The area of a trapezoid formed between two equilateral triangles -/
theorem trapezoid_area (large_triangle_area small_triangle_area : ℝ) 
  (h1 : large_triangle_area = 36)
  (h2 : small_triangle_area = 4)
  (h3 : large_triangle_area > small_triangle_area) :
  let num_trapezoids : ℕ := 6
  let area_between_triangles := large_triangle_area - small_triangle_area
  let trapezoid_area := area_between_triangles / num_trapezoids
  abs (trapezoid_area - 5.33) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l24_2400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l24_2471

noncomputable def polynomial_terms (n : ℕ) (x : ℝ) : ℝ := sorry

theorem binomial_expansion_coefficient (n : ℕ) (a b c : ℝ) (h1 : n ≥ 3) 
  (h2 : ∀ x, (x + 2)^n = x^n + a*x^(n-1) + polynomial_terms n x + b*x + c) 
  (h3 : b = 4*c) : a = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l24_2471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shuffle_to_right_majority_l24_2438

/-- Represents a card with its names -/
structure Card where
  names : Finset String

/-- Represents the state of the card stacks -/
structure CardStacks where
  left : Finset Card
  right : Finset Card

/-- A shuffle operation moves cards with a specific name between stacks -/
def shuffle (stacks : CardStacks) (name : String) : CardStacks :=
  sorry

/-- Theorem: It's always possible to end up with more cards in the right stack -/
theorem shuffle_to_right_majority (initial : CardStacks) 
    (h : initial.left.card > initial.right.card) :
    ∃ (names : List String), 
      (let final := names.foldl shuffle initial
       final.right.card > final.left.card) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shuffle_to_right_majority_l24_2438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_l24_2450

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x - 2 / x

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := 1 + 2 / (x^2)

-- Theorem statement
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ 3 * x - 2 * y - 4 = 0) ∧
    (m = f' 2) ∧
    (f 2 = m * 2 + b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_l24_2450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_three_coins_feasible_l24_2478

def coins_per_edge : List Nat := [2, 3, 4, 5, 6, 7]

def total_coins : Nat := 12

def is_feasible (n : Nat) : Prop := n * 4 = total_coins

theorem only_three_coins_feasible :
  ∃! n, n ∈ coins_per_edge ∧ is_feasible n ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_three_coins_feasible_l24_2478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l24_2495

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_specific_points :
  distance (3, 6) (-5, 2) = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l24_2495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_max_profit_l24_2427

/-- Investment problem parameters -/
structure InvestmentParams where
  totalFunds : ℚ
  minRatio : ℚ
  minInvestment : ℚ
  profitRateA : ℚ
  profitRateB : ℚ

/-- Calculate the maximum profit for the investment problem -/
noncomputable def maxProfit (params : InvestmentParams) : ℚ :=
  let investmentB := min (params.totalFunds / (1 + params.minRatio)) (params.totalFunds - params.minInvestment)
  let investmentA := params.totalFunds - investmentB
  investmentA * params.profitRateA + investmentB * params.profitRateB

/-- The main theorem stating the maximum profit for the given parameters -/
theorem investment_max_profit :
  let params : InvestmentParams := {
    totalFunds := 600000,
    minRatio := 2/3,
    minInvestment := 50000,
    profitRateA := 4/10,
    profitRateB := 6/10
  }
  maxProfit params = 312000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_max_profit_l24_2427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_is_85_over_7_l24_2408

def b : ℕ → ℚ
  | 0 => 2
  | 1 => 3
  | n + 2 => (1 / 3) * b (n + 1) + (1 / 5) * b n

theorem sum_of_sequence_is_85_over_7 :
  ∑' n, b n = 85 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_is_85_over_7_l24_2408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l24_2462

def sequence_property (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 2 ∧
  (∀ n : ℕ, n > 0 → a n * a (n + 1) * a (n + 2) ≠ 1) ∧
  (∀ n : ℕ, n > 0 → a n * a (n + 1) * a (n + 2) * a (n + 3) = a n + a (n + 1) + a (n + 2) + a (n + 3))

theorem sequence_sum (a : ℕ → ℕ) (h : sequence_property a) :
  (Finset.range 100).sum a = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l24_2462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_on_curve_l24_2448

open Real

-- Define the curve C
def curve_C (θ : ℝ) : Prop := sin θ = abs (cos θ)

-- Define the line l
def line_l (ρ θ : ℝ) : Prop := ρ * cos θ - 2 * ρ * sin θ = 2

-- Define the distance from a point to the line l
noncomputable def distance_to_line_l (x y : ℝ) : ℝ := 
  abs (x - 2*y - 2) / sqrt 5

-- Define the theorem
theorem distance_between_points_on_curve (M N : ℝ × ℝ) :
  (∃ θ₁ θ₂ : ℝ, 
    curve_C θ₁ ∧ curve_C θ₂ ∧ 
    M = (cos θ₁, sin θ₁) ∧ N = (cos θ₂, sin θ₂) ∧
    M ≠ N ∧
    distance_to_line_l M.1 M.2 = sqrt 5 ∧
    distance_to_line_l N.1 N.2 = sqrt 5) →
  sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 2 * sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_on_curve_l24_2448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_2_tangent_through_A_l24_2433

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 4

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 8*x + 5

-- Define the tangent line at a point
def tangent_line (a : ℝ) (x : ℝ) : ℝ := f a + f' a * (x - a)

-- Theorem for the first question
theorem tangent_at_2 :
  ∀ x y : ℝ, y = tangent_line 2 x ↔ x - y - 4 = 0 :=
sorry

-- Theorem for the second question
theorem tangent_through_A :
  ∀ a x y : ℝ, y = tangent_line a x ∧ tangent_line a 2 = -2 ↔ 
    (x - y - 4 = 0 ∨ y + 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_2_tangent_through_A_l24_2433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_for_pure_imaginary_product_l24_2436

theorem no_real_solutions_for_pure_imaginary_product : 
  ¬∃ (x : ℝ), (x + Complex.I) * ((x + 1) + Complex.I) * ((x + 2) + Complex.I) * ((x + 3) + Complex.I) = 
  Complex.I * Complex.abs ((x + Complex.I) * ((x + 1) + Complex.I) * ((x + 2) + Complex.I) * ((x + 3) + Complex.I)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_for_pure_imaginary_product_l24_2436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l24_2481

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

-- State the theorem
theorem odd_function_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f a b x = -f a b (-x)) →  -- f is odd on [-1,1]
  f a b 1 = 1 →                                        -- f(1) = 1
  (∃ c : ℝ, ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f a b x = 2 * x / (1 + x^2)) ∧  -- f(x) = 2x/(1+x^2)
  (∀ x y, x ∈ Set.Icc (-1 : ℝ) 1 → y ∈ Set.Icc (-1 : ℝ) 1 → x < y → f a b x < f a b y) ∧  -- f is increasing on [-1,1]
  {m : ℝ | f a b (2*m+1) + f a b (m^2-1) < 0} = Set.Ioc (-1 : ℝ) 0 := by  -- Range of m
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l24_2481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_product_constant_l24_2490

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_slope_product_constant (e : Ellipse) 
  (h_focal : e.a^2 - e.b^2 = 1)
  (h_point : PointOnEllipse e) 
  (h_point_coords : h_point.x = 1 ∧ h_point.y = 3/2) :
  ∀ (A P : PointOnEllipse e), A ≠ P →
  let B : PointOnEllipse e := ⟨-A.x, -A.y, by sorry⟩
  (B.y - A.y) / (B.x - A.x) * (P.y - A.y) / (P.x - A.x) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_product_constant_l24_2490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l24_2441

theorem power_equation_solution (x : ℝ) : (4 : ℝ)^(2*x + 2) = (16 : ℝ)^(3*x - 1) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l24_2441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_l24_2447

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

-- State the theorem
theorem odd_function_values (a b : ℝ) :
  (∀ x ≠ 1, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_l24_2447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l24_2416

noncomputable def f (x y : ℝ) : ℝ := (2015 * (x + y)) / Real.sqrt (2015 * x^2 + 2015 * y^2)

theorem f_minimum_value :
  ∀ x y : ℝ, f x y ≥ -Real.sqrt 4030 ∧
  (f x y = -Real.sqrt 4030 ↔ x = y ∧ x < 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l24_2416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l24_2419

/-- The speed of a train in kilometers per hour -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * (3600 / 1000)

/-- Theorem: The speed of a 280 m long train that passes a tree in 14 seconds 
    is approximately 5.56 km/hr (rounded to two decimal places) -/
theorem train_speed_calculation :
  (round (train_speed 280 14 * 100) : ℝ) / 100 = 5.56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l24_2419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_per_square_foot_is_four_l24_2422

/-- Represents the tile problem with given conditions -/
structure TileProblem where
  length : ℝ
  width : ℝ
  green_tile_percentage : ℝ
  green_tile_cost : ℝ
  red_tile_cost : ℝ
  total_cost : ℝ

/-- Calculates the number of tiles per square foot -/
noncomputable def tiles_per_square_foot (p : TileProblem) : ℝ :=
  let area := p.length * p.width
  let green_tile_ratio := p.green_tile_percentage / 100
  let red_tile_ratio := 1 - green_tile_ratio
  let total_tiles := p.total_cost / (green_tile_ratio * p.green_tile_cost + red_tile_ratio * p.red_tile_cost)
  total_tiles / area

/-- Theorem stating that the tiles per square foot is 4 for the given problem -/
theorem tiles_per_square_foot_is_four :
  let problem := TileProblem.mk 10 25 40 3 1.5 2100
  tiles_per_square_foot problem = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_per_square_foot_is_four_l24_2422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l24_2439

/-- Represents a hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- The angle between the asymptote and the y-axis in radians -/
noncomputable def asymptote_angle (h : Hyperbola) : ℝ :=
  Real.arctan (h.b / h.a)

/-- The area of the quadrilateral formed by the vertices of the hyperbola -/
def quadrilateral_area (h : Hyperbola) : ℝ :=
  2 * h.a * h.b

theorem hyperbola_equation (h : Hyperbola) 
  (angle_condition : asymptote_angle h = π / 6)
  (area_condition : quadrilateral_area h = 8 * Real.sqrt 3) :
  h.a = 2 * Real.sqrt 3 ∧ h.b = 2 := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l24_2439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_l24_2442

/-- A circle with center O and radius r -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Determine if a point is outside a circle -/
def is_outside (c : Circle) (p : Point) : Prop :=
  distance (Point.mk c.center.1 c.center.2) p > c.radius

theorem point_outside_circle (O : Circle) (A : Point) 
  (h1 : O.radius = 3)
  (h2 : distance (Point.mk O.center.1 O.center.2) A = 5) :
  is_outside O A := by
  sorry

#check point_outside_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_l24_2442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_negative_product_l24_2468

def sequenceA (n : ℕ) : ℚ :=
  if n = 0 then 3 else (11 - 2 * (n + 1)) / 3

theorem first_negative_product :
  (∀ k < 5, sequenceA k * sequenceA (k + 1) ≥ 0) ∧
  sequenceA 5 * sequenceA 6 < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_negative_product_l24_2468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_floor_is_one_l24_2486

noncomputable def x_sequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 1/2
  | n + 1 => let x := x_sequence n; x^2 + x

noncomputable def series_sum : ℚ :=
  (Finset.range 100).sum (λ k => 1 / (x_sequence k + 1))

theorem series_sum_floor_is_one : ⌊series_sum⌋ = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_floor_is_one_l24_2486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_volume_l24_2496

/-- Represents the initial volume of water in the mixture -/
def initial_water : ℝ := sorry

/-- Represents the initial volume of milk in the mixture -/
def initial_milk : ℝ := sorry

/-- Represents the initial ratio of milk to water -/
def initial_ratio : ℝ := 4

/-- Represents the volume of water added to the mixture -/
def added_water : ℝ := 21

/-- Represents the new ratio of milk to water after adding water -/
def new_ratio : ℝ := 1.2

/-- Represents the initial volume of the mixture -/
def initial_volume : ℝ := initial_milk + initial_water

theorem mixture_volume :
  initial_milk = initial_ratio * initial_water ∧
  initial_milk / (initial_water + added_water) = new_ratio →
  initial_volume = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_volume_l24_2496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_and_inequality_l24_2446

-- Define the sector
def sector_area : ℝ := 4
def sector_perimeter : ℝ := 8

-- Define the point on the exponential function
def point_on_exp (a : ℝ) : Prop := (3 : ℝ)^a = 9

-- Define the central angle
noncomputable def central_angle (r : ℝ) : ℝ := (sector_perimeter - 2*r) / r

-- Define the solution set
noncomputable def solution_set (a : ℝ) : Set ℝ := 
  {x | ∃ k : ℤ, Real.pi/6 + k*Real.pi ≤ x ∧ x ≤ Real.pi/3 + k*Real.pi}

-- Theorem statement
theorem sector_and_inequality :
  ∃ r : ℝ, 
    r > 0 ∧
    (1/2) * (sector_perimeter - 2*r) * r = sector_area ∧
    central_angle r = 2 ∧
    ∃ a : ℝ, point_on_exp a ∧
      {x : ℝ | Real.sin (a*x) ≥ Real.sqrt 3/2} = solution_set a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_and_inequality_l24_2446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_l24_2474

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2

noncomputable def g (x : ℝ) : ℝ := -1/2 * Real.sin (2*x - 2*Real.pi/3) + 1/2

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem g_increasing (k : ℤ) :
  is_increasing g (-5*Real.pi/12 + k*Real.pi) (Real.pi/12 + k*Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_l24_2474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_line_equation_min_area_line_minimizes_l24_2472

/-- A line passing through (2,1) and intersecting the positive x and y axes -/
structure IntersectingLine where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : 2/a + 1/b = 1

/-- The area of the triangle formed by the line and the origin -/
noncomputable def triangleArea (l : IntersectingLine) : ℝ := (1/2) * l.a * l.b

/-- The line that minimizes the area of the triangle -/
noncomputable def minAreaLine : IntersectingLine :=
  { a := 4
  , b := 2
  , h1 := by norm_num
  , h2 := by norm_num
  , h3 := by
      field_simp
      ring }

theorem min_area_line_equation :
  let l := minAreaLine
  l.a = 4 ∧ l.b = 2 := by
  simp [minAreaLine]

-- The following theorem states that minAreaLine indeed minimizes the triangle area
theorem min_area_line_minimizes :
  ∀ l : IntersectingLine, triangleArea minAreaLine ≤ triangleArea l := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_line_equation_min_area_line_minimizes_l24_2472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_quadratic_m_l24_2459

/-- A function f(x) is quadratic if it can be written in the form f(x) = ax^2 + bx + c, where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function y=(m+1)x^(m^2-3m-2)+3x-2 -/
noncomputable def F (m : ℝ) (x : ℝ) : ℝ :=
  (m + 1) * x^(m^2 - 3*m - 2) + 3 * x - 2

/-- Theorem stating that m = 4 is the only value that makes F(m)(x) a quadratic function -/
theorem unique_quadratic_m : 
  ∃! m : ℝ, IsQuadratic (F m) ∧ m = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_quadratic_m_l24_2459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_4_5_6_parallelepiped_extended_l24_2403

/-- The volume of the set of points that are inside or within one unit of a rectangular parallelepiped -/
noncomputable def parallelepiped_extended_volume (l w h : ℝ) : ℝ :=
  let box_volume := l * w * h
  let external_volume := 2 * (l * w + l * h + w * h)
  let spherical_caps_volume := 4 / 3 * Real.pi
  let cylindrical_wedges_volume := (l + w + h) * Real.pi
  box_volume + external_volume + spherical_caps_volume + cylindrical_wedges_volume

/-- Theorem stating the volume of the set of points that are inside or within one unit of a 4x5x6 rectangular parallelepiped -/
theorem volume_4_5_6_parallelepiped_extended :
  parallelepiped_extended_volume 4 5 6 = 268 + 49 / 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_4_5_6_parallelepiped_extended_l24_2403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_sum_in_triangle_l24_2453

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if B is obtuse and 4bS = a(b² + c² - a²), where S is the area of the triangle,
    then the maximum value of sin A + sin C is 9/8 -/
theorem max_sin_sum_in_triangle (a b c : ℝ) (A B C : ℝ) (S : ℝ) 
    (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c)
    (h_angles : 0 < A ∧ 0 < B ∧ 0 < C)
    (h_sum_angles : A + B + C = Real.pi)
    (h_obtuse : Real.pi / 2 < B)
    (h_area : 4 * b * S = a * (b^2 + c^2 - a^2)) :
    (∀ (A' B' C' : ℝ), A' + B' + C' = Real.pi → Real.pi / 2 < B' → 
      Real.sin A + Real.sin C ≤ Real.sin A' + Real.sin C') → 
    Real.sin A + Real.sin C = 9 / 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_sum_in_triangle_l24_2453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_iff_l24_2451

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - 4 * a
  else a * x^2 - 3 * x

def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem f_strictly_increasing_iff (a : ℝ) :
  strictly_increasing (f a) ↔ a ∈ Set.Icc (3/2) 3 := by
  sorry

#check f_strictly_increasing_iff

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_iff_l24_2451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l24_2460

-- Define the function
noncomputable def f (x : ℝ) := Real.log (5 - x) / Real.sqrt (2 * x - 2)

-- State the theorem
theorem f_defined_iff (x : ℝ) : 
  (∃ y, f x = y) ↔ (1 ≤ x ∧ x < 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l24_2460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_324_and_648_are_36_times_digit_sum_l24_2482

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Predicate for numbers that are 36 times the sum of their digits -/
def is36TimesDigitSum (n : ℕ) : Prop :=
  n = 36 * sumOfDigits n

/-- Theorem stating that 324 and 648 are the only natural numbers
    that are 36 times the sum of their digits -/
theorem only_324_and_648_are_36_times_digit_sum :
  ∀ n : ℕ, is36TimesDigitSum n ↔ n = 324 ∨ n = 648 := by
  sorry

-- Explicitly define instances for decidability
instance : Decidable (is36TimesDigitSum 324) :=
  if h : 324 = 36 * sumOfDigits 324 then
    isTrue h
  else
    isFalse h

instance : Decidable (is36TimesDigitSum 648) :=
  if h : 648 = 36 * sumOfDigits 648 then
    isTrue h
  else
    isFalse h

#eval sumOfDigits 324  -- Should output 9
#eval sumOfDigits 648  -- Should output 18
#eval is36TimesDigitSum 324  -- Should output true
#eval is36TimesDigitSum 648  -- Should output true

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_324_and_648_are_36_times_digit_sum_l24_2482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_divisible_sum_l24_2475

open BigOperators

def f (n : ℕ) : ℕ := ∑ k in Finset.range 2011, n^k

theorem no_divisible_sum (m : ℕ) (hm : 2 ≤ m ∧ m ≤ 2010) : 
  ¬ ∃ n : ℕ, m ∣ f n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_divisible_sum_l24_2475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_at_distance_l24_2483

-- Define the circle D
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem max_points_at_distance (D : Set (ℝ × ℝ)) (center_D : ℝ × ℝ) (radius_D : ℝ) (Q : ℝ × ℝ) :
  D = Circle center_D radius_D →
  Q ∉ D →
  (∃ (n : ℕ), ∀ (S : Finset (ℝ × ℝ)), 
    (↑S ⊆ D ∧ (∀ p ∈ S, distance p Q = 5)) → 
    S.card ≤ n) ∧
  (∃ (S : Finset (ℝ × ℝ)), ↑S ⊆ D ∧ (∀ p ∈ S, distance p Q = 5) ∧ S.card = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_at_distance_l24_2483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l24_2429

theorem vector_equation_solution :
  ∃! (a b : ℝ), a • (![1, 4] : Fin 2 → ℝ) + b • (![(-1), 6] : Fin 2 → ℝ) = ![0, 5] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l24_2429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_negative_21_l24_2469

def sequenceA : ℕ → ℤ
  | 0 => 1
  | 1 => -5
  | 2 => 9
  | 3 => -13
  | 4 => 17
  | n + 5 => sequenceA n

theorem sixth_term_is_negative_21 : sequenceA 5 = -21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_negative_21_l24_2469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_b_completion_time_l24_2454

/-- Represents the time in minutes for a cyclist to complete one round of a circular road. -/
def CompletionTime := ℝ

/-- Represents the meeting time in minutes after the cyclists start. -/
def MeetingTime := ℝ

/-- 
Given two cyclists A and B starting from the same point on a circular road at the same time 
but in opposite directions, where A takes 70 minutes to complete one round and they meet 
after 45 minutes, this function calculates the time it takes for B to complete one round.
-/
def calculateBCompletionTime (a_time : CompletionTime) (meet_time : MeetingTime) : CompletionTime :=
  (126 : ℝ)

/-- 
Theorem stating that under the given conditions, cyclist B takes 126 minutes to complete one round.
-/
theorem b_completion_time : 
  let a_time : CompletionTime := (70 : ℝ)
  let meet_time : MeetingTime := (45 : ℝ)
  calculateBCompletionTime a_time meet_time = (126 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_b_completion_time_l24_2454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_q_l24_2409

theorem polynomial_coefficient_q (p q r s : ℝ) (i : ℂ) : 
  i^2 = -1 →
  (∃ (u v w z : ℂ), 
    (∀ x : ℂ, x^4 + p*x^3 + q*x^2 + r*x + s = 0 ↔ x = u ∨ x = v ∨ x = w ∨ x = z) ∧
    (u * v = 17 + 2*i) ∧
    (w + z = 2 + 5*i) ∧
    (u ∉ Set.range (Complex.ofReal : ℝ → ℂ)) ∧ 
    (v ∉ Set.range (Complex.ofReal : ℝ → ℂ)) ∧ 
    (w ∉ Set.range (Complex.ofReal : ℝ → ℂ)) ∧ 
    (z ∉ Set.range (Complex.ofReal : ℝ → ℂ))) →
  q = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_q_l24_2409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_l24_2465

theorem cosine_of_angle (α : ℝ) (P : ℝ × ℝ) : 
  P = (-3, -4) → Real.cos α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_l24_2465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_sum_bounds_l24_2464

def b (n : ℕ) : ℚ :=
  if n ≤ 4 then (2 * n + 1 : ℚ)
  else 1 / (n * (n + 2))

def T (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => b (i + 1))

theorem b_sum_bounds : ∀ n : ℕ, n > 0 → 3 ≤ T n ∧ T n < 24 + 11/60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_sum_bounds_l24_2464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_f_of_g_solution_f_gt_g_l24_2430

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x^2 - 1

noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ 0 then 2 * x - 1 else 2 - x

-- Theorem for the range of g
theorem range_of_g : Set.range g = Set.Ici (-1) := by sorry

-- Theorem for the expression of f(g(x))
theorem f_of_g (x : ℝ) : f (g x) =
  if x ≥ 0 then 8 * x^2 - 8 * x + 1 else 2 * x^2 - 8 * x + 7 := by sorry

-- Theorem for the solution of f(x) > g(x)
theorem solution_f_gt_g : {x : ℝ | f x > g x} =
  Set.Iio (-3/2) ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_f_of_g_solution_f_gt_g_l24_2430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l24_2491

-- Define the slope of a line given its equation in the form ax + by = c
noncomputable def line_slope (a b : ℝ) : ℝ := -a / b

-- Define what it means for two lines to be parallel
def parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  line_slope a₁ b₁ = line_slope a₂ b₂

-- Theorem statement
theorem parallel_line_slope :
  ∀ (a b c : ℝ), b ≠ 0 → parallel 3 6 (-24) a b c → line_slope a b = -1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l24_2491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_increase_l24_2404

/-- The annual percentage increase in expenditure -/
def annual_increase : ℝ → ℝ := λ r ↦
  (20000 * (1 + r)^2) - 24200.000000000004

theorem expenditure_increase : ∃ r : ℝ, annual_increase r = 0 ∧ r = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_increase_l24_2404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_jogging_speed_l24_2432

/-- Represents the problem of Jack jogging to the beach with an ice cream cone -/
def beach_ice_cream_problem (normal_melt_time : ℝ) (wind_speed : ℝ) (temperature : ℝ) 
  (melt_factor : ℝ) (distance_blocks : ℕ) (block_length : ℝ) : Prop :=
  let adjusted_melt_time : ℝ := normal_melt_time * (1 - melt_factor)
  let adjusted_melt_hours : ℝ := adjusted_melt_time / 60
  let distance_miles : ℝ := (distance_blocks : ℝ) * block_length
  let required_speed : ℝ := distance_miles / adjusted_melt_hours
  required_speed = 16

/-- Theorem stating the minimum speed Jack needs to jog -/
theorem jack_jogging_speed : 
  beach_ice_cream_problem 10 15 85 0.25 16 (1/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_jogging_speed_l24_2432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_dozen_cost_l24_2445

/-- The cost of lemons in rubles -/
def lemon_cost : ℝ → ℝ := sorry

/-- The number of lemons you can buy for a given amount of rubles -/
def lemons_for_rubles : ℝ → ℝ := sorry

theorem lemon_dozen_cost :
  (∀ x : ℝ, lemon_cost 24 = x ↔ lemons_for_rubles 500 = x) →
  lemon_cost 12 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_dozen_cost_l24_2445
