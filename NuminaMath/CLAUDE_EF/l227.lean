import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_property_l227_22740

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter of a triangle
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the angle BAC
noncomputable def angle_BAC (t : Triangle) : ℝ := sorry

-- Define vector operations
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vector_scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

theorem circumcenter_property (t : Triangle) :
  let P := circumcenter t
  vector_add (vector_scale 5 (vector_add P (vector_scale (-1) t.A)))
             (vector_scale (-2) (vector_add t.B t.C)) = (0, 0) →
  Real.cos (angle_BAC t) = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_property_l227_22740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l227_22772

/-- Curve C with parametric equations x = 4t^2 and y = 4t -/
noncomputable def C : ℝ → ℝ × ℝ := λ t ↦ (4 * t^2, 4 * t)

/-- Line l passing through (2,0) with inclination angle π/4 -/
noncomputable def l : ℝ → ℝ × ℝ := λ s ↦ (2 + s * Real.sqrt 2 / 2, s * Real.sqrt 2 / 2)

/-- The length of segment AB formed by the intersection of C and l -/
noncomputable def length_AB : ℝ := 4 * Real.sqrt 6

theorem intersection_length :
  ∃ (s₁ s₂ : ℝ), s₁ ≠ s₂ ∧ C (Real.sqrt (s₁ / 4)) = l s₁ ∧ C (Real.sqrt (s₂ / 4)) = l s₂ ∧
  Real.sqrt ((s₁ - s₂)^2) = length_AB := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l227_22772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_radius_of_circle_l227_22700

/-- Given a circle C passing through points (4, 0) and (-4, 0) in a rectangular coordinate system, 
    the maximum possible radius of C is 4 units. -/
theorem max_radius_of_circle (C : Set (ℝ × ℝ)) : 
  ((4 : ℝ), 0) ∈ C → ((-4 : ℝ), 0) ∈ C → ∃ (center : ℝ × ℝ) (r : ℝ), 
    C = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2} ∧ 
    r ≤ 4 ∧ 
    ∃ (C' : Set (ℝ × ℝ)) (center' : ℝ × ℝ), 
      ((4 : ℝ), 0) ∈ C' ∧ ((-4 : ℝ), 0) ∈ C' ∧
      C' = {p : ℝ × ℝ | (p.1 - center'.1)^2 + (p.2 - center'.2)^2 = 4^2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_radius_of_circle_l227_22700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l227_22771

noncomputable def g (x : ℝ) : ℝ := |⌊x + 2⌋| - |⌊3 - x⌋|

theorem g_symmetry (x : ℝ) : g x = g (3 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l227_22771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_interval_l227_22782

theorem count_integers_in_interval : 
  (Finset.filter (fun n : ℕ => 24 ≤ n ∧ n ≤ 42) (Finset.range 43)).card = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_interval_l227_22782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_marbles_in_basket_A_l227_22712

/-- Represents the number of marbles of a specific color in a basket -/
structure MarbleCount where
  red : ℕ := 0
  yellow : ℕ := 0
  green : ℕ := 0
  white : ℕ := 0

/-- Represents a basket of marbles -/
structure Basket where
  marbles : MarbleCount

/-- The greatest difference between the number of marbles of different colors in a basket -/
def greatestDifference (b : Basket) : ℕ :=
  max (b.marbles.red) (max b.marbles.yellow (max b.marbles.green b.marbles.white)) -
  min (b.marbles.red) (min b.marbles.yellow (min b.marbles.green b.marbles.white))

/-- The maximum difference among all baskets -/
def maxDifference : ℕ := 6

def basketA (R : ℕ) : Basket :=
  { marbles := { red := R, yellow := 2 } }

def basketB : Basket :=
  { marbles := { green := 6, yellow := 1 } }

def basketC : Basket :=
  { marbles := { white := 3, yellow := 9 } }

theorem red_marbles_in_basket_A :
  ∃ R : ℕ, basketA R = { marbles := { red := R, yellow := 2 } } ∧
  greatestDifference (basketA R) ≤ maxDifference ∧
  greatestDifference basketB ≤ maxDifference ∧
  greatestDifference basketC = maxDifference ∧
  R = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_marbles_in_basket_A_l227_22712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limsup_complement_lim_complement_liminf_subset_limsup_limsup_union_liminf_inter_limsup_liminf_inclusion_monotone_convergence_l227_22743

variable {α : Type*} [MeasurableSpace α]
variable (A B : ℕ → Set α)

-- Define limsup and liminf
def limsup (A : ℕ → Set α) : Set α :=
  ⋂ n, ⋃ m ≥ n, A m

def liminf (A : ℕ → Set α) : Set α :=
  ⋃ n, ⋂ m ≥ n, A m

-- Define limit of sets (renamed to avoid conflict)
def set_lim (A : ℕ → Set α) : Set α :=
  {x | ∃ N, ∀ n ≥ N, x ∈ A n}

-- Define non-decreasing and non-increasing sequences
def non_decreasing (A : ℕ → Set α) : Prop :=
  ∀ n m, n ≤ m → A n ⊆ A m

def non_increasing (A : ℕ → Set α) : Prop :=
  ∀ n m, n ≤ m → A m ⊆ A n

-- Define convergence for non-decreasing and non-increasing sequences
def converges_to_up (A : ℕ → Set α) (limit : Set α) : Prop :=
  non_decreasing A ∧ (⋃ n, A n) = limit

def converges_to_down (A : ℕ → Set α) (limit : Set α) : Prop :=
  non_increasing A ∧ (⋂ n, A n) = limit

-- State the theorems
theorem limsup_complement (A : ℕ → Set α) :
  (limsup A)ᶜ = liminf (fun n => (A n)ᶜ) := by sorry

theorem lim_complement (A : ℕ → Set α) :
  (set_lim A)ᶜ = limsup (fun n => (A n)ᶜ) := by sorry

theorem liminf_subset_limsup (A : ℕ → Set α) :
  liminf A ⊆ limsup A := by sorry

theorem limsup_union (A B : ℕ → Set α) :
  limsup (fun n => A n ∪ B n) = limsup A ∪ set_lim B := by sorry

theorem liminf_inter (A B : ℕ → Set α) :
  liminf (fun n => A n ∩ B n) = liminf A ∩ liminf B := by sorry

theorem limsup_liminf_inclusion (A B : ℕ → Set α) :
  (limsup A ∩ liminf B ⊆ limsup (fun n => A n ∩ B n)) ∧
  (limsup (fun n => A n ∩ B n) ⊆ limsup A ∩ limsup B) := by sorry

theorem monotone_convergence (A : ℕ → Set α) (limit : Set α) :
  (converges_to_up A limit ∨ converges_to_down A limit) →
  liminf A = limsup A := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limsup_complement_lim_complement_liminf_subset_limsup_limsup_union_liminf_inter_limsup_liminf_inclusion_monotone_convergence_l227_22743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yeast_count_relationship_l227_22724

/-- Represents the dimensions of a hemocytometer square in millimeters -/
structure HemocytometerSquare where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the volume of a hemocytometer square in cubic millimeters -/
def square_volume (square : HemocytometerSquare) : ℝ :=
  square.length * square.width * square.depth

/-- Calculates the total number of yeast in 1mL of culture medium -/
noncomputable def total_yeast_count (square : HemocytometerSquare) (yeast_in_square : ℕ) : ℝ :=
  (1000 / square_volume square) * (yeast_in_square : ℝ)

/-- Theorem stating the relationship between yeast count in a square and total count in 1mL -/
theorem yeast_count_relationship (square : HemocytometerSquare) (yeast_in_square : ℕ) :
  square.length = 2 ∧ square.width = 2 ∧ square.depth = 0.1 →
  total_yeast_count square yeast_in_square = 2.5e3 * (yeast_in_square : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_yeast_count_relationship_l227_22724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_cells_bound_l227_22786

/-- Represents a grid with marked cells -/
structure MarkedGrid :=
  (size : Nat)
  (marked : Finset (Nat × Nat))

/-- Checks if a 3x3 square in the grid contains exactly one marked cell -/
def validSquare (g : MarkedGrid) (i j : Nat) : Prop :=
  ∃! (x y : Nat), x ∈ Finset.range 3 ∧ y ∈ Finset.range 3 ∧ (i + x, j + y) ∈ g.marked

/-- A 10x10 grid where each 3x3 square contains exactly one marked cell -/
def validGrid (g : MarkedGrid) : Prop :=
  g.size = 10 ∧ 
  ∀ i j, i < 8 ∧ j < 8 → validSquare g i j

theorem marked_cells_bound (g : MarkedGrid) (h : validGrid g) : 
  9 ≤ g.marked.card ∧ g.marked.card ≤ 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_cells_bound_l227_22786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_iter_ratio_l227_22735

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (2*x)

noncomputable def f_iter : ℕ → ℝ → ℝ
| 0, x => x
| n + 1, x => f (f_iter n x)

theorem f_iter_ratio (n : ℕ) (x : ℝ) (h : x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1) :
  f_iter n x / f_iter (n + 1) x = 1 + 1 / f ((((x + 1) / (x - 1)) : ℝ)^(2^n)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_iter_ratio_l227_22735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l227_22745

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (x^2 + m*x) * Real.exp x

theorem f_monotonicity (m : ℝ) :
  (m = -2 → StrictMono (f m)) ∧
  (∀ x ∈ Set.Icc 1 3, StrictMonoOn (f m) (Set.Icc 1 3) → m ≤ -15/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l227_22745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_difference_l227_22778

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Define the range of a function
def range (h : ℝ → ℝ) (S : Set ℝ) : Prop := ∀ y, y ∈ S ↔ ∃ x, h x = y

-- State the theorem
theorem range_of_difference (hf : is_odd f) (hg : is_even g) 
  (h_range : range (fun x ↦ f x + g x) (Set.Ici 1 ∩ Set.Iio 3)) :
  range (fun x ↦ f x - g x) (Set.Ioo (-3) (-1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_difference_l227_22778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_question_determines_brothers_l227_22744

-- Define the brothers
inductive Brother
| Vanya
| Vasya

-- Define the possible responses
inductive Response
| Yes
| No

-- Define the truthfulness of a brother
def isTruthful (b : Brother) : Prop :=
  match b with
  | Brother.Vanya => true
  | Brother.Vasya => false

-- Define the question asked
def askQuestion (b : Brother) : Response :=
  match b with
  | Brother.Vanya => Response.No
  | Brother.Vasya => Response.Yes

-- Define the property that the question can determine the brothers' identities
def canDetermineBrothers : Prop :=
  ∀ (b1 b2 : Brother), b1 ≠ b2 →
    (isTruthful b1 ≠ isTruthful b2) →
    (askQuestion b1 = Response.Yes ↔ b1 = Brother.Vanya) ∧
    (askQuestion b2 = Response.Yes ↔ b2 = Brother.Vanya)

-- Theorem statement
theorem question_determines_brothers :
  canDetermineBrothers := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_question_determines_brothers_l227_22744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_l227_22703

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (x + φ) + Real.sqrt 3 * Real.cos (x + φ)

noncomputable def g (x φ : ℝ) : ℝ := Real.cos (x + φ)

theorem min_value_g (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) :
  ∃ (x : ℝ), x ∈ Set.Icc (-π) (π/6) ∧
  g x φ = -1/2 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-π) (π/6) → g y φ ≥ -1/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_l227_22703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trisection_intersection_l227_22722

/-- The x-coordinate of point E when a horizontal line through C intersects the natural logarithm curve -/
noncomputable def x₃ (x₁ x₂ : ℝ) : ℝ :=
  let y₁ := Real.log x₁
  let y₂ := Real.log x₂
  let yC := (2 * y₁ + y₂) / 3
  Real.exp yC

theorem trisection_intersection (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) 
    (h₃ : x₁ = 2) (h₄ : x₂ = 500) : 
  x₃ x₁ x₂ = (200000 : ℝ) ^ (1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trisection_intersection_l227_22722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hike_time_is_130_minutes_l227_22749

/-- Calculates the time taken to complete a hike with uphill and downhill sections -/
noncomputable def hike_time (trail_length : ℝ) (uphill_percentage : ℝ) (uphill_speed : ℝ) (downhill_speed : ℝ) : ℝ :=
  let uphill_distance := trail_length * uphill_percentage
  let downhill_distance := trail_length * (1 - uphill_percentage)
  let uphill_time := uphill_distance / uphill_speed
  let downhill_time := downhill_distance / downhill_speed
  (uphill_time + downhill_time) * 60

/-- Theorem: The time taken to complete the specific hike is 130 minutes -/
theorem hike_time_is_130_minutes :
  hike_time 5 0.6 2 3 = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hike_time_is_130_minutes_l227_22749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_l227_22767

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 - Real.tan x) * (Real.cos x)^2

-- State the theorem
theorem f_strictly_decreasing :
  ∀ x y, x ∈ Set.Icc (11 * Real.pi / 12) Real.pi →
         y ∈ Set.Icc (11 * Real.pi / 12) Real.pi →
         x < y → f x > f y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_l227_22767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_new_percentage_l227_22733

/-- A cricket team's performance statistics --/
structure CricketTeam where
  initial_matches : ℕ
  initial_win_percentage : ℚ
  additional_wins : ℕ

/-- Calculate the new winning percentage of a cricket team --/
def new_winning_percentage (team : CricketTeam) : ℚ :=
  let initial_wins := (team.initial_win_percentage * team.initial_matches) / 100
  let total_wins := initial_wins + team.additional_wins
  let total_matches := team.initial_matches + team.additional_wins
  (total_wins / total_matches) * 100

/-- Theorem: The cricket team's new winning percentage is 52% --/
theorem cricket_team_new_percentage :
  let team := CricketTeam.mk 120 30 55
  new_winning_percentage team = 52 := by
  sorry

#eval new_winning_percentage (CricketTeam.mk 120 30 55)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_new_percentage_l227_22733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_value_approx_l227_22725

/-- Given that 7a = 2b and 42ab = 674.9999999999999, prove that the common value of 7a and 2b is approximately 15. -/
theorem common_value_approx (a b : ℝ) 
  (h1 : 7 * a = 2 * b) 
  (h2 : 42 * a * b = 674.9999999999999) : 
  abs ((7 * a) - 15) < 0.0000001 ∧ abs ((2 * b) - 15) < 0.0000001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_value_approx_l227_22725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_ellipse_eccentricity_range_problem_l227_22752

open Real

theorem ellipse_eccentricity_range (a b c : ℝ) (h_ellipse : a > b ∧ b > 0) 
  (h_directrices : 2 * (a / (c / a)) ≤ 3 * (2 * a)) : 
  1 / 3 ≤ c / a ∧ c / a < 1 := by
  -- Define eccentricity
  let e := c / a

  -- Prove lower bound
  have h_lower : 1 / 3 ≤ e := by
    -- Simplify the directrices condition
    have h1 : a / e ≤ 3 * a := by
      -- Proof steps here
      sorry
    -- Deduce lower bound
    -- Proof steps here
    sorry

  -- Prove upper bound
  have h_upper : e < 1 := by
    -- Use the definition of ellipse (c < a)
    -- Proof steps here
    sorry

  -- Combine the bounds
  exact ⟨h_lower, h_upper⟩

-- Main theorem connecting to the problem statement
theorem ellipse_eccentricity_range_problem : 
  ∃ (e : ℝ), 1 / 3 ≤ e ∧ e < 1 ∧ 
  ∀ (e' : ℝ), (1 / 3 ≤ e' ∧ e' < 1) → 
    ∃ (a b c : ℝ), a > b ∧ b > 0 ∧ 
    2 * (a / (c / a)) ≤ 3 * (2 * a) ∧ 
    e' = c / a := by
  -- Proof of existence and bounds
  -- Proof steps here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_ellipse_eccentricity_range_problem_l227_22752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l227_22737

open Real

theorem trigonometric_identities 
  (α β : ℝ) 
  (h_acute_α : 0 < α ∧ α < π/2)
  (h_acute_β : 0 < β ∧ β < π/2)
  (h_cos_α : Real.cos α = 4/5)
  (h_cos_αβ : Real.cos (α+β) = -16/65)
  (h_angle_order : 0 < β ∧ β < π/4 ∧ π/4 < α ∧ α < 3*π/4)
  (h_cos_π4_α : Real.cos (π/4 - α) = 3/5)
  (h_sin_3π4_β : Real.sin (3*π/4 + β) = 5/13) : 
  Real.cos β = 5/13 ∧ Real.sin (α+β) = 56/65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l227_22737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_probability_correct_probability_l227_22705

-- Define the circle equation
noncomputable def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 + k*x - 2*y - (5/4)*k = 0

-- Define the condition for two tangents
def two_tangents_condition (k : ℝ) : Prop :=
  k > -1 ∧ k < 0

-- Define the probability calculation
noncomputable def probability_calculation (lower upper : ℝ) : ℝ :=
  (upper - lower) / 4

-- Main theorem
theorem tangent_probability : 
  probability_calculation (-1) 0 = 1/4 := by
  sorry

-- Proof that the calculated probability is correct
theorem correct_probability : 
  ∀ k : ℝ, k ∈ Set.Icc (-2 : ℝ) 2 →
  (∃ x y : ℝ, circle_equation x y k ∧ 
   two_tangents_condition k) ↔ 
  k ∈ Set.Ioo (-1 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_probability_correct_probability_l227_22705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orientation_changes_odd_times_after_25_moves_l227_22739

-- Define a type for points on a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a type for triangles
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the orientation of a triangle
noncomputable def orientation (t : Triangle) : Int :=
  let det := (t.B.x - t.A.x) * (t.C.y - t.A.y) - (t.C.x - t.A.x) * (t.B.y - t.A.y)
  if det > 0 then 1 else if det < 0 then -1 else 0

-- Define a single move
def move (t : Triangle) : Triangle :=
  sorry

-- Define the result of n moves
def n_moves (t : Triangle) (n : ℕ) : Triangle :=
  match n with
  | 0 => t
  | n + 1 => move (n_moves t n)

-- Theorem statement
theorem orientation_changes_odd_times_after_25_moves (t : Triangle) :
  orientation (n_moves t 25) ≠ orientation t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orientation_changes_odd_times_after_25_moves_l227_22739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l227_22787

theorem matrix_transformation (N : Matrix (Fin 3) (Fin 3) ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; 1, 0, 0; 0, 0, -2]
  M * N = !![N 1 0, N 1 1, N 1 2;
             N 0 0, N 0 1, N 0 2;
             -2 * N 2 0, -2 * N 2 1, -2 * N 2 2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l227_22787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l227_22759

-- Define the propositions
def prop1 (x : ℝ) : Prop := x^2 = 1
def prop2 (x : ℝ) : Prop := x = 1
def prop3 (x : ℝ) : Prop := x = -1
def prop4 (x : ℝ) : Prop := x^2 - 3*x + 2 = 0

-- State the theorem
theorem problem_statement :
  (¬(∀ x : ℝ, (prop1 x → prop2 x) ∧ ¬(prop2 x → prop1 x))) ∧
  (¬(∀ x : ℝ, (prop4 x → prop3 x) ∧ ¬(prop3 x → prop4 x))) ∧
  ((¬∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0)) ∧
  (∀ x y : ℝ, Real.sin x ≠ Real.sin y → x ≠ y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l227_22759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_D_cannot_form_triangle_l227_22791

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the sets of line segments
def set_A : List ℝ := [4, 5, 6]
def set_B : List ℝ := [3, 4, 5]
def set_C : List ℝ := [2, 3, 4]
def set_D : List ℝ := [1, 2, 3]

-- Theorem statement
theorem only_set_D_cannot_form_triangle :
  (can_form_triangle set_A[0]! set_A[1]! set_A[2]!) ∧
  (can_form_triangle set_B[0]! set_B[1]! set_B[2]!) ∧
  (can_form_triangle set_C[0]! set_C[1]! set_C[2]!) ∧
  ¬(can_form_triangle set_D[0]! set_D[1]! set_D[2]!) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_D_cannot_form_triangle_l227_22791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_speed_approximation_l227_22736

/-- The speed of person A in km/h -/
noncomputable def speed_A : ℝ := 5

/-- The time in hours that A walks before B starts -/
noncomputable def time_before_B : ℝ := 0.5

/-- The time in hours that B walks to overtake A -/
noncomputable def time_B_overtake : ℝ := 1.8

/-- The speed of person B in km/h -/
noncomputable def speed_B : ℝ := (speed_A * time_before_B + speed_A * time_B_overtake) / time_B_overtake

theorem B_speed_approximation :
  abs (speed_B - 6.39) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_speed_approximation_l227_22736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_expression_l227_22756

theorem simplify_sqrt_expression : Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_expression_l227_22756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_two_digit_prime_factor_of_binom_300_150_l227_22763

/-- The largest 2-digit prime factor of (300 choose 150) -/
theorem largest_two_digit_prime_factor_of_binom_300_150 : ℕ :=
  let n := Nat.choose 300 150
  let p := 89
  have h1 : 10 ≤ p ∧ p < 100 := by sorry
  have h2 : Nat.Prime p := by sorry
  have h3 : p^3 ∣ n := by sorry
  have h4 : ∀ q : ℕ, 10 ≤ q ∧ q < 100 ∧ Nat.Prime q ∧ q^3 ∣ n → q ≤ p := by sorry
  p

-- Add this line to make the result computable
def largest_two_digit_prime_factor_of_binom_300_150_computable : ℕ := 89

#eval largest_two_digit_prime_factor_of_binom_300_150_computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_two_digit_prime_factor_of_binom_300_150_l227_22763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_circle_square_area_l227_22701

/-- The area of a square formed by a wire that can also form a circle with radius 56 cm -/
theorem wire_circle_square_area : 
  ∀ (r : ℝ) (circle_circumference square_perimeter : ℝ) (square_side square_area : ℝ),
  r = 56 →
  circle_circumference = 2 * Real.pi * r →
  square_perimeter = circle_circumference →
  square_side = square_perimeter / 4 →
  square_area = square_side ^ 2 →
  square_area = 784 * Real.pi := by
  sorry

#check wire_circle_square_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_circle_square_area_l227_22701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l227_22796

theorem sin_cos_difference (θ : ℝ) (h1 : θ ∈ Set.Ioo 0 Real.pi) (h2 : Real.sin θ + Real.cos θ = 1/5) :
  Real.sin θ - Real.cos θ = 7/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l227_22796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l227_22762

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

/-- The line function -/
def g (x : ℝ) : ℝ := x - 1

/-- The distance function from a point (x, f(x)) to the line y = x - 1 -/
noncomputable def distance (x : ℝ) : ℝ := 
  |f x - g x| / Real.sqrt 2

theorem min_distance_theorem :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → distance x ≤ distance y ∧ distance x = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l227_22762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_is_81_l227_22726

/-- The sum of an arithmetic sequence with n terms, first term a, and last term l -/
noncomputable def arithmetic_sum (n : ℕ) (a l : ℝ) : ℝ := (n / 2) * (a + l)

/-- The sum of the sequence (x + 1) + (x + 2) + ... + (x + 20) -/
noncomputable def left_sum (x : ℝ) : ℝ := arithmetic_sum 20 (x + 1) (x + 20)

/-- The sum of the sequence 174 + 176 + 178 + ... + 192 -/
noncomputable def right_sum : ℝ := arithmetic_sum 10 174 192

theorem x_value_is_81 (x : ℝ) : left_sum x = right_sum → x = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_is_81_l227_22726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_hyperbola_l227_22751

/-- The hyperbola function -/
noncomputable def hyperbola (k : ℝ) (x : ℝ) : ℝ := k / x

/-- The distance from the origin to a point on the hyperbola -/
noncomputable def distance_to_origin (k : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 + (hyperbola k x)^2)

/-- The theorem statement -/
theorem min_distance_to_hyperbola (k : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b = 4 ∧ hyperbola k (a/2) = b/2) →
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → distance_to_origin k x ≤ distance_to_origin k y) →
  (∃ (x : ℝ), x > 0 ∧ distance_to_origin k x = Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_hyperbola_l227_22751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_8_value_l227_22793

def sequence_a : ℕ → ℤ
  | 0 => -1
  | n + 1 => sequence_a n - 3

theorem a_8_value : sequence_a 8 = -22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_8_value_l227_22793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_value_area_triangle_ABC_l227_22718

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
def triangle_conditions (A B C a b c : ℝ) : Prop :=
  A = Real.pi/4 ∧ Real.cos B = 3/5 ∧ a = 5

-- Theorem for sin C
theorem sin_C_value (h : triangle_conditions A B C a b c) :
  Real.sin C = (7 * Real.sqrt 2) / 10 := by
  sorry

-- Theorem for area of triangle ABC
theorem area_triangle_ABC (h : triangle_conditions A B C a b c) :
  (1/2) * a * b * Real.sin C = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_value_area_triangle_ABC_l227_22718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_satisfies_conditions_l227_22741

-- Define p as a predicate on x and a
def p (x a : ℝ) : Prop := 2 * x^2 - 9 * x + a < 0

-- Define q as a proposition (the exact definition is not provided, so we leave it abstract)
axiom q : Prop

-- Define the range of a
def range_of_a : Set ℝ := sorry

-- Theorem statement
theorem range_of_a_satisfies_conditions :
  (∀ x a, ¬(p x a) → ¬q) →
  ∃ S, range_of_a = S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_satisfies_conditions_l227_22741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barium_iodide_molecular_weight_l227_22794

/-- The atomic weight of barium in g/mol -/
def barium_weight : ℝ := 137.33

/-- The atomic weight of iodine in g/mol -/
def iodine_weight : ℝ := 126.90

/-- The number of iodine atoms in barium iodide -/
def iodine_count : ℕ := 2

/-- The molecular weight of barium iodide in g/mol -/
def barium_iodide_weight : ℝ := barium_weight + iodine_count * iodine_weight

/-- Theorem stating that the molecular weight of barium iodide is approximately 391.13 g/mol -/
theorem barium_iodide_molecular_weight :
  abs (barium_iodide_weight - 391.13) < 0.01 := by
  -- Unfold the definition of barium_iodide_weight
  unfold barium_iodide_weight
  -- Perform the calculation
  simp [barium_weight, iodine_weight, iodine_count]
  -- The result is true by computation
  norm_num
  -- Complete the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barium_iodide_molecular_weight_l227_22794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotone_increasing_iff_l227_22764

open Real

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + b * x^2 + (b + 2) * x + 3

noncomputable def f' (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*b*x + (b + 2)

theorem not_monotone_increasing_iff (b : ℝ) :
  (∃ x y : ℝ, x < y ∧ f b x > f b y) ↔ b < -1 ∨ b > 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotone_increasing_iff_l227_22764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_time_l227_22784

/-- Represents the speed and time characteristics of a bus journey -/
structure BusJourney where
  speed_without_stops : ℝ  -- Average speed without stops in km/hr
  speed_with_stops : ℝ     -- Average speed with stops in km/hr
  total_time : ℝ           -- Total journey time in hours

/-- Calculates the time spent stopped during a bus journey -/
noncomputable def time_stopped (journey : BusJourney) : ℝ :=
  journey.total_time * (1 - journey.speed_with_stops / journey.speed_without_stops)

/-- Theorem: Given a bus with average speeds of 60 km/hr without stops and 30 km/hr with stops,
    the time spent stopped in a one-hour journey is 30 minutes (0.5 hours) -/
theorem bus_stop_time (journey : BusJourney) 
  (h1 : journey.speed_without_stops = 60)
  (h2 : journey.speed_with_stops = 30)
  (h3 : journey.total_time = 1) : 
  time_stopped journey = 0.5 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_time_l227_22784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_double_angle_l227_22728

/-- The slope of a line given its inclination angle -/
noncomputable def slope_from_angle (θ : Real) : Real := Real.tan θ

/-- The inclination angle of a line given its slope -/
noncomputable def angle_from_slope (m : Real) : Real := Real.arctan m

theorem slope_of_double_angle :
  let ref_slope := Real.sqrt 3 / 3
  let ref_angle := angle_from_slope ref_slope
  let l_angle := 2 * ref_angle
  slope_from_angle l_angle = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_double_angle_l227_22728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_savings_percentage_l227_22715

/-- Calculates the percentage saved in a sale transaction -/
noncomputable def percentage_saved (amount_saved : ℝ) (amount_spent : ℝ) : ℝ :=
  (amount_saved / (amount_spent + amount_saved)) * 100

/-- Theorem stating that the percentage saved is approximately 12.09% -/
theorem sale_savings_percentage : 
  let amount_saved : ℝ := 2.75
  let amount_spent : ℝ := 20
  abs (percentage_saved amount_saved amount_spent - 12.09) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_savings_percentage_l227_22715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_theorem_l227_22704

noncomputable def f (x : ℝ) : ℝ := Real.sin (2019 * x + Real.pi / 6) + Real.cos (2019 * x - Real.pi / 3)

theorem min_difference_theorem (A : ℝ) (x₁ x₂ : ℝ) 
  (h_max : ∀ x, f x ≤ A)
  (h_bounds : ∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂) :
  ∃ (x₁' x₂' : ℝ), A * |x₁' - x₂'| = Real.pi / 1009 ∧ ∀ x₃ x₄, A * |x₃ - x₄| ≥ Real.pi / 1009 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_theorem_l227_22704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_altitudes_theorem_l227_22742

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : ℂ
  B : ℂ
  C : ℂ
  isAcute : Prop

-- Define the extended points A', B', C'
def extendedPoints (t : AcuteTriangle) (k : ℝ) : ℂ × ℂ × ℂ :=
  let A' := t.A + 2 * k * Complex.I * (t.C - t.B)
  let B' := t.B + 2 * k * Complex.I * (t.A - t.C)
  let C' := t.C + 2 * k * Complex.I * (t.B - t.A)
  (A', B', C')

-- Define the condition for equilateral triangle
def isEquilateral (A B C : ℂ) : Prop :=
  (C - A) = (Complex.exp (Complex.I * Real.pi / 3)) * (B - A)

-- Theorem statement
theorem extended_altitudes_theorem (t : AcuteTriangle) :
  ∃ k : ℝ, k > 0 ∧
  let (A', B', C') := extendedPoints t k
  isEquilateral A' B' C' →
  k = 1 / (2 * Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_altitudes_theorem_l227_22742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_example_is_equation_l227_22717

-- Definition of an equation
def is_equation (e : ℝ → Prop) : Prop :=
  ∃ (x : ℝ), e x

-- The statement to prove
theorem example_is_equation :
  is_equation (λ x : ℝ ↦ (1/2 * x - 5 * x) = 18) :=
by
  -- We need to prove that there exists an x that satisfies the equation
  use 0  -- We can use any real number here, as we're just proving existence
  -- The proof is left as 'sorry' as requested
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_example_is_equation_l227_22717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_implies_sin_double_l227_22731

theorem sin_sum_implies_sin_double (x : ℝ) :
  Real.sin (π + x) + Real.sin ((3 * π) / 2 + x) = 1 / 2 →
  Real.sin (2 * x) = -3 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_implies_sin_double_l227_22731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l227_22757

/-- Represents a pile of stones in the game -/
structure Pile where
  stones : ℕ

/-- Represents the game state -/
structure GameState where
  piles : List Pile

/-- A move in the game is splitting a pile into two smaller piles -/
def Move := Pile → Pile × Pile

/-- The initial game state -/
def initialState : GameState :=
  { piles := [⟨10⟩, ⟨15⟩, ⟨20⟩] }

/-- A player has a move if there exists a pile with more than one stone -/
def hasMove (state : GameState) : Prop :=
  ∃ p ∈ state.piles, p.stones > 1

/-- The game ends when no moves are possible -/
def gameOver (state : GameState) : Prop :=
  ¬(hasMove state)

/-- Apply a single move to a game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Apply a sequence of moves to a game state -/
def applyMoves : GameState → List Move → GameState
  | state, [] => state
  | state, move :: moves => applyMoves (applyMove state move) moves

/-- A winning strategy for a player is a sequence of moves that leads to victory -/
def winningStrategy (player : ℕ) (state : GameState) : Prop :=
  ∃ (moves : List Move), 
    (moves.length % 2 = player % 2) ∧ 
    (gameOver (applyMoves state moves)) ∧
    (∀ i < moves.length, hasMove (applyMoves state (moves.take i)))

/-- The main theorem: the second player has a winning strategy -/
theorem second_player_wins :
  winningStrategy 1 initialState :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l227_22757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_regular_pentagon_l227_22711

/-- The area of a circle circumscribed about a regular pentagon with side length 10 units -/
theorem circle_area_regular_pentagon (π : ℝ) : ∃ A : ℝ, A = 2000 * (5 + 2 * Real.sqrt 5) * π := by
  -- Define the side length of the regular pentagon
  let s : ℝ := 10

  -- Define the number of sides in a pentagon
  let n : ℕ := 5

  -- Define the radius of the circumscribed circle
  let R : ℝ := s / (2 * Real.sin (π / n))

  -- Define the area of the circle
  let A : ℝ := π * R^2

  -- The theorem statement
  use A
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_regular_pentagon_l227_22711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_after_120_moves_l227_22798

noncomputable def π : ℝ := Real.pi

structure Particle where
  x : ℝ
  y : ℝ

def initialPosition : Particle := ⟨6, 0⟩

noncomputable def rotateAndTranslate (p : Particle) : Particle :=
  let rotated_x := p.x * (Real.cos (π/3)) - p.y * (Real.sin (π/3))
  let rotated_y := p.x * (Real.sin (π/3)) + p.y * (Real.cos (π/3))
  ⟨rotated_x + 12, rotated_y⟩

noncomputable def moveNTimes : ℕ → Particle → Particle
  | 0, p => p
  | n+1, p => rotateAndTranslate (moveNTimes n p)

theorem final_position_after_120_moves :
  moveNTimes 120 initialPosition = ⟨1446, 0⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_after_120_moves_l227_22798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_coordinates_l227_22773

/-- A circle in the 2D plane -/
structure Circle where
  a : ℝ
  b : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a circle -/
def Point.onCircle (p : Point) (c : Circle) : Prop :=
  p.x^2 + p.y^2 + c.a * p.x - 2 * p.y + c.b = 0

/-- The symmetric point of P with respect to the line x + y - 1 = 0 -/
def symmetricPoint (p : Point) : Point :=
  ⟨2 - p.y, 2 - p.x⟩

/-- The center of a circle given its equation parameters -/
noncomputable def Circle.center (c : Circle) : Point :=
  ⟨-c.a/2, 1⟩

theorem circle_center_coordinates (c : Circle) :
  let p : Point := ⟨2, 1⟩
  p.onCircle c ∧ (symmetricPoint p).onCircle c →
  c.center = ⟨0, 1⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_coordinates_l227_22773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_percent_decrease_approx_l227_22747

noncomputable section

-- Define the original prices
def trouser_price : ℝ := 100
def shirt_price : ℝ := 80

-- Define the discount rates
def trouser_discount : ℝ := 0.35
def shirt_discount : ℝ := 0.25

-- Define the sales tax rate
def sales_tax_rate : ℝ := 0.10

-- Define the function to calculate the final price after discounts and tax
noncomputable def final_price (trouser_price shirt_price trouser_discount shirt_discount sales_tax_rate : ℝ) : ℝ :=
  let discounted_trouser := trouser_price * (1 - trouser_discount)
  let discounted_shirt := shirt_price * (1 - shirt_discount)
  let total_discounted := discounted_trouser + discounted_shirt
  total_discounted * (1 + sales_tax_rate)

-- Define the function to calculate the percent decrease
noncomputable def percent_decrease (original_price final_price : ℝ) : ℝ :=
  (original_price - final_price) / original_price * 100

-- Theorem statement
theorem overall_percent_decrease_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧
  abs (percent_decrease (trouser_price + shirt_price) 
    (final_price trouser_price shirt_price trouser_discount shirt_discount sales_tax_rate) - 23.61) < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_percent_decrease_approx_l227_22747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_large_union_l227_22755

theorem existence_of_large_union (S : Finset ℕ) (A : Finset ℕ) : 
  S = Finset.range 100 →
  A ⊆ S →
  Finset.card A = 10 →
  ∃ (I : Finset ℕ), 
    Finset.card I = 10 ∧
    I ⊆ Finset.range 100 ∧
    Finset.card (I.biUnion (fun i ↦ Finset.image (fun a ↦ (a + i) % 100) A)) ≥ 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_large_union_l227_22755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_totient_prime_power_euler_totient_product_l227_22746

-- Define Euler's totient function
def φ : ℕ → ℕ := sorry

-- Define primality
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 0 → m < p → p % m ≠ 0

-- Define coprimality
def coprime (m n : ℕ) : Prop :=
  Nat.gcd m n = 1

-- Theorem for part a
theorem euler_totient_prime_power (p n : ℕ) (h : is_prime p) :
  φ (p^n) = p^n - p^(n-1) := by
  sorry

-- Theorem for part b
theorem euler_totient_product (m n : ℕ) (h : coprime m n) :
  φ (m * n) = φ m * φ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_totient_prime_power_euler_totient_product_l227_22746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_max_value_achieved_l227_22777

theorem max_value_theorem (x y z : ℝ) (h : 9*x^2 + 4*y^2 + 25*z^2 = 1) :
  5*x + 3*y + 10*z ≤ 5*Real.sqrt 13/6 :=
by sorry

theorem max_value_achieved :
  ∀ ε > 0, ∃ x y z : ℝ, 9*x^2 + 4*y^2 + 25*z^2 = 1 ∧ 
  5*x + 3*y + 10*z > 5*Real.sqrt 13/6 - ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_max_value_achieved_l227_22777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_consecutive_integers_l227_22768

def is_consecutive (s : List Int) : Prop :=
  s.length = 7 ∧ ∀ i : Fin 6, s[i.val + 1]! = s[i.val]! + 1

def rearranged (s : List Int) : List Int :=
  [s[3]!, s[1]!, s[2]!, s[6]!, s[4]!, s[5]!, s[0]!]

theorem seven_consecutive_integers :
  ∃! s : List Int,
    is_consecutive s ∧
    s = [-3, -2, -1, 0, 1, 2, 3] ∧
    (rearranged s)[3]! = (rearranged s).maximum? ∧
    |((rearranged s)[3]!)| = |((rearranged s)[2]!)| := by
  sorry

#check seven_consecutive_integers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_consecutive_integers_l227_22768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cytosine_molecules_required_l227_22780

/-- The number of base pairs in the DNA fragment -/
def base_pairs : ℕ := 500

/-- The percentage of A+T bases in the DNA fragment -/
def at_percentage : ℚ := 34/100

/-- The number of replications -/
def replications : ℕ := 2

/-- Theorem: The number of free cytosine deoxyribonucleotide molecules required -/
theorem cytosine_molecules_required : 
  (2 * base_pairs * (1 - at_percentage) * replications).floor = 1320 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cytosine_molecules_required_l227_22780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equality_l227_22732

/-- Helper function to calculate the area of a triangle given three points -/
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Helper function to calculate the area of a square given four points -/
def area_square (A B C D : ℝ × ℝ) : ℝ := sorry

/-- Predicate to check if four points form a square -/
def is_square (A B C D : ℝ × ℝ) : Prop := sorry

/-- Given a square OPQR with O at (0,0) and Q at (3,3), prove that T(3,6) creates a triangle PQT
    with area equal to the area of square OPQR -/
theorem area_equality (O P Q R T : ℝ × ℝ) : 
  O = (0, 0) →
  Q = (3, 3) →
  T = (3, 6) →
  is_square O P Q R →
  area_triangle P Q T = area_square O P Q R :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equality_l227_22732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_l227_22720

def T : ℝ × ℝ := (3, 3)
def O : ℝ × ℝ := (0, 0)

theorem circle_triangle_area :
  ∀ A : ℝ × ℝ,
  (A.1 - T.1)^2 + (A.2 - T.2)^2 = (O.1 - T.1)^2 + (O.2 - T.2)^2 →
  Real.arccos ((A.1 * O.1 + A.2 * O.2) / (Real.sqrt (A.1^2 + A.2^2) * Real.sqrt (O.1^2 + O.2^2))) = π / 4 →
  (1 / 2) * Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) * Real.sqrt ((T.1 - O.1)^2 + (T.2 - O.2)^2) = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_l227_22720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bella_hits_ten_l227_22754

/-- Represents a player in the dart game -/
inductive Player
| Adam
| Bella
| Carlos
| Diana
| Evan
| Fiona
| Grace

/-- Represents a dart throw score -/
def DartScore := Fin 12

/-- Represents a pair of dart throws -/
structure ThrowPair where
  first : DartScore
  second : DartScore
  different : first ≠ second

/-- The score of a player is the sum of their two throws -/
def score (throws : ThrowPair) : Nat := throws.first.val + throws.second.val + 2

/-- All throws in the game are distinct -/
def distinct_throws (throws : Player → ThrowPair) : Prop :=
  ∀ p1 p2, p1 ≠ p2 →
    throws p1 ≠ throws p2 ∧
    (throws p1).first ≠ (throws p2).first ∧
    (throws p1).first ≠ (throws p2).second ∧
    (throws p1).second ≠ (throws p2).first ∧
    (throws p1).second ≠ (throws p2).second

theorem bella_hits_ten (throws : Player → ThrowPair)
  (h_distinct : distinct_throws throws)
  (h_adam : score (throws Player.Adam) = 18)
  (h_bella : score (throws Player.Bella) = 15)
  (h_carlos : score (throws Player.Carlos) = 12)
  (h_diana : score (throws Player.Diana) = 9)
  (h_evan : score (throws Player.Evan) = 20)
  (h_fiona : score (throws Player.Fiona) = 13)
  (h_grace : score (throws Player.Grace) = 17) :
  (throws Player.Bella).first.val = 9 ∨ (throws Player.Bella).second.val = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bella_hits_ten_l227_22754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_is_six_l227_22713

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3 * Real.exp (abs (x - 1)) - Real.sin (x - 1)) / Real.exp (abs (x - 1))

-- Define the interval
def interval : Set ℝ := Set.Icc (-3) 5

-- Theorem statement
theorem sum_of_max_min_is_six :
  ∃ (p q : ℝ), (∀ x, x ∈ interval → f x ≤ p) ∧
               (∀ x, x ∈ interval → q ≤ f x) ∧
               (∃ x₁ x₂, x₁ ∈ interval ∧ x₂ ∈ interval ∧ f x₁ = p ∧ f x₂ = q) ∧
               p + q = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_is_six_l227_22713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_one_extremum_in_interval_sequence_inequality_l227_22761

open Real

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + 1 / x + a * x

-- Theorem 1
theorem extremum_at_one (a : ℝ) :
  (∀ x > 0, HasDerivAt (f a) ((1 / x - 1 / (x^2) + a) : ℝ) x) →
  (HasDerivAt (f a) 0 1) →
  f a 1 = 1 :=
by sorry

-- Theorem 2
theorem extremum_in_interval (a : ℝ) :
  (∃ c ∈ Set.Ioo 2 3, HasDerivAt (f a) 0 c) →
  a ∈ Set.Ioo (-1/4) (-2/9) :=
by sorry

-- Theorem 3
theorem sequence_inequality (x : ℕ → ℝ) :
  (∀ n : ℕ, x n > 0) →
  (∀ n : ℕ, Real.log (x n) + 1 / (x (n + 1)) < 1) →
  x 0 ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_one_extremum_in_interval_sequence_inequality_l227_22761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_factor_of_60n_l227_22775

theorem other_factor_of_60n (x : ℕ) (h1 : ∀ n : ℕ, n ≥ 8 → x ∣ (60 * n))
  (h2 : ∀ n : ℕ, n ≥ 8 → 8 ∣ (60 * n)) (h3 : 8 ∣ (60 * 8)) : x = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_factor_of_60n_l227_22775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_theorem_l227_22706

theorem sphere_diameter_theorem (r₁ : ℝ) (r₂ : ℝ) :
  r₁ = 6 →
  (4 / 3) * Real.pi * r₂^3 = 3 * ((4 / 3) * Real.pi * r₁^3) →
  2 * r₂ = 12 * (3 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_theorem_l227_22706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_scenario_days_l227_22783

/-- Represents the walking scenario from place A to place B -/
structure WalkingScenario where
  first_day_distance : ℚ
  additional_distance : ℚ
  total_distance : ℚ

/-- The number of days in the first scenario -/
def first_scenario_days : ℕ := 10

/-- The number of days in the second scenario -/
def second_scenario_days : ℕ := 15

/-- Calculates the total distance walked in the first scenario -/
def first_scenario_distance (w : WalkingScenario) : ℚ :=
  w.first_day_distance * first_scenario_days + w.additional_distance * (first_scenario_days * (first_scenario_days - 1) / 2)

/-- Calculates the total distance walked in the second scenario -/
def second_scenario_distance (w : WalkingScenario) : ℚ :=
  w.first_day_distance * second_scenario_days

/-- Theorem stating the number of days required in the third scenario -/
theorem third_scenario_days (w : WalkingScenario) :
  first_scenario_distance w = second_scenario_distance w →
  (w.first_day_distance + w.additional_distance * (first_scenario_days - 1)) * (15/2) = w.total_distance := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_scenario_days_l227_22783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class1_qualified_l227_22748

noncomputable def is_qualified (data : List ℝ) : Prop :=
  data.length = 5 ∧ data.all (λ x => x ≤ 5)

noncomputable def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  (data.map (λ x => (x - mean data) ^ 2)).sum / data.length

theorem class1_qualified (data : List ℝ) :
  data.length = 5 → mean data = 2 → variance data = 2 → is_qualified data :=
by
  intro h_length h_mean h_variance
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class1_qualified_l227_22748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_two_l227_22790

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem tangent_line_at_two :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ x + 4 * y - 4 = 0) ∧
    (m = deriv f 2) ∧
    (f 2 = m * 2 + b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_two_l227_22790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_630_l227_22770

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (w : ℂ) : Prop := w^4 = 81 * i

-- Define the polar form of a complex number
noncomputable def polar_form (r : ℝ) (θ : ℝ) : ℂ := r * (Complex.exp (Complex.I * θ))

-- Define the condition for the angles
def angle_condition (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < 2 * Real.pi

-- Theorem statement
theorem sum_of_angles_630 :
  ∃ (r₁ r₂ r₃ r₄ θ₁ θ₂ θ₃ θ₄ : ℝ),
    (r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧ r₄ > 0) ∧
    (angle_condition θ₁ ∧ angle_condition θ₂ ∧ angle_condition θ₃ ∧ angle_condition θ₄) ∧
    equation (polar_form r₁ θ₁) ∧
    equation (polar_form r₂ θ₂) ∧
    equation (polar_form r₃ θ₃) ∧
    equation (polar_form r₄ θ₄) ∧
    θ₁ + θ₂ + θ₃ + θ₄ = 11 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_630_l227_22770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_common_terms_l227_22785

def a : ℕ → ℤ
  | 0 => 2
  | 1 => 14
  | (n + 2) => 14 * a (n + 1) + a n

def b : ℕ → ℤ
  | 0 => 2
  | 1 => 14
  | (n + 2) => 6 * b (n + 1) - b n

theorem infinite_common_terms : 
  ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ k, a (2 * f k + 1) = b (3 * f k + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_common_terms_l227_22785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_theorem_l227_22738

noncomputable def f (x : ℝ) : ℝ := Real.sin x

noncomputable def translate (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x => f (x + a)

noncomputable def stretch_horizontal (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f (x / k)

noncomputable def g (x : ℝ) : ℝ := Real.sin (x / 2 + Real.pi / 3)

theorem transformation_theorem :
  stretch_horizontal (translate f (Real.pi / 3)) 2 = g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_theorem_l227_22738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_n_a_n_l227_22792

noncomputable def L (x : ℝ) : ℝ := x - x^2 / 2

noncomputable def a (n : ℕ+) : ℝ :=
  let rec iterate (k : ℕ) (x : ℝ) : ℝ :=
    if k = 0 then x else iterate (k - 1) (L x)
  iterate n (17 / n.val)

theorem limit_n_a_n :
  ∀ ε > 0, ∃ N : ℕ+, ∀ n : ℕ+, n ≥ N → |↑n * a n - 34 / 19| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_n_a_n_l227_22792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calories_theorem_l227_22734

/-- Represents the recipe and calorie content of lemonade --/
structure LemonadeRecipe where
  lemon_juice_weight : ℚ
  honey_weight : ℚ
  water_weight : ℚ
  lemon_juice_calories : ℚ
  honey_calories_per_100g : ℚ

/-- Calculates the calories in a given weight of lemonade --/
def calories_in_lemonade (recipe : LemonadeRecipe) (weight : ℚ) : ℚ :=
  let total_weight := recipe.lemon_juice_weight + recipe.honey_weight + recipe.water_weight
  let total_calories := recipe.lemon_juice_calories + (recipe.honey_calories_per_100g * recipe.honey_weight / 100)
  (total_calories / total_weight) * weight

/-- The main theorem stating that 300g of the given lemonade recipe contains 200.4 calories --/
theorem lemonade_calories_theorem (recipe : LemonadeRecipe) 
    (h1 : recipe.lemon_juice_weight = 150)
    (h2 : recipe.honey_weight = 100)
    (h3 : recipe.water_weight = 250)
    (h4 : recipe.lemon_juice_calories = 30)
    (h5 : recipe.honey_calories_per_100g = 304) :
    calories_in_lemonade recipe 300 = 2004 / 10 := by
  sorry

def main : IO Unit := do
  let result := calories_in_lemonade 
    { lemon_juice_weight := 150
      honey_weight := 100
      water_weight := 250
      lemon_juice_calories := 30
      honey_calories_per_100g := 304 } 300
  IO.println s!"Calories in 300g of lemonade: {result}"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calories_theorem_l227_22734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_flight_time_l227_22799

/-- The time taken for four birds to fly a given distance -/
noncomputable def flight_time (eagle_speed falcon_speed pelican_speed hummingbird_speed total_distance : ℝ) : ℝ :=
  total_distance / (eagle_speed + falcon_speed + pelican_speed + hummingbird_speed)

/-- Theorem: The flight time for the given conditions is 2 hours -/
theorem bird_flight_time :
  flight_time 15 46 33 30 248 = 2 := by
  -- Unfold the definition of flight_time
  unfold flight_time
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_flight_time_l227_22799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_proof_l227_22721

theorem car_distance_proof (speed1 speed2 avg_speed total_distance : ℝ) 
  (h1 : speed1 = 90)
  (h2 : speed2 = 80)
  (h3 : avg_speed = 84.70588235294117)
  (h4 : total_distance = 320) :
  let d := total_distance / 2
  let t1 := d / speed1
  let t2 := d / speed2
  avg_speed = total_distance / (t1 + t2) → d = 160 := by
  intro h5
  -- Proof steps would go here
  sorry

#check car_distance_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_proof_l227_22721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_asymptote_l227_22719

-- Define the denominator polynomial
noncomputable def d (x : ℝ) : ℝ := 3*x^6 - 2*x^5 + 4*x^3 - x + 2

-- Define a general polynomial of degree n with leading coefficient a
noncomputable def p (n : ℕ) (a : ℝ) (x : ℝ) : ℝ := a * x^n

-- Define the rational function
noncomputable def f (n : ℕ) (a : ℝ) (x : ℝ) : ℝ := (p n a x) / (d x)

-- Theorem statement
theorem rational_function_asymptote :
  (∃ (n : ℕ) (a : ℝ), ∀ (m : ℕ) (b : ℝ), (m > n ∨ (m = n ∧ |b| > |a|)) → 
    ¬∃ (L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x, |x| > δ → |f m b x - L| < ε) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x| > δ → |f 6 3 x - 1| < ε) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_asymptote_l227_22719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_ABM_l227_22708

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola x^2 = 4y -/
def Parabola := {p : Point | p.x^2 = 4 * p.y}

/-- Represents a circle x^2 + y^2 = 5 -/
def Circle := {p : Point | p.x^2 + p.y^2 = 5}

/-- The focus of the parabola -/
def Focus : Point := ⟨0, 1⟩

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  (1/2) * abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))

/-- Theorem stating the minimum area of triangle ABM -/
theorem min_area_triangle_ABM :
  ∃ (A B M : Point),
    A ∈ Parabola ∧ B ∈ Parabola ∧
    (∃ (k : ℝ), A.y = k * A.x + Focus.y ∧ B.y = k * B.x + Focus.y) ∧
    M.y = 0 ∧
    (∃ (m : ℝ), M.y - A.y = m * (M.x - A.x) ∧ m = A.x / 2) ∧
    (∀ (A' B' M' : Point),
      A' ∈ Parabola → B' ∈ Parabola →
      (∃ (k' : ℝ), A'.y = k' * A'.x + Focus.y ∧ B'.y = k' * B'.x + Focus.y) →
      M'.y = 0 →
      (∃ (m' : ℝ), M'.y - A'.y = m' * (M'.x - A'.x) ∧ m' = A'.x / 2) →
      triangleArea A B M ≤ triangleArea A' B' M') ∧
    triangleArea A B M = 8 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_ABM_l227_22708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l227_22730

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of line l₁: ax + 2y + 6 = 0 -/
noncomputable def slope_l₁ (a : ℝ) : ℝ := -a / 2

/-- The slope of line l₂: x + (a-1)y + a² - 1 = 0 -/
noncomputable def slope_l₂ (a : ℝ) : ℝ := -1 / (a - 1)

/-- Theorem: If lines l₁ and l₂ are perpendicular, then a = 2/3 -/
theorem perpendicular_lines_a_value (a : ℝ) :
  perpendicular (slope_l₁ a) (slope_l₂ a) → a = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l227_22730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_one_sixth_l227_22779

-- Define the function f
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 4 * Real.cos (ω * x + φ)

-- State the theorem
theorem function_value_at_one_sixth 
  (ω φ : ℝ) 
  (h_ω : ω > 0)
  (h_φ : 0 < φ ∧ φ < Real.pi)
  (h_odd : ∀ x, f ω φ (-x) = -(f ω φ x))
  (a b : ℝ)
  (h_points : f ω φ a = 0 ∧ f ω φ b = 0)
  (h_min_dist : ∀ x y, f ω φ x = 0 → f ω φ y = 0 → |x - y| ≥ 1)
  (h_exists_min : ∃ x y, f ω φ x = 0 ∧ f ω φ y = 0 ∧ |x - y| = 1) :
  f ω φ (1/6) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_one_sixth_l227_22779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_one_l227_22760

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 3 then 1 / (x^2 - 1)
  else 2 * x^(-1/2:ℝ)

-- State the theorem
theorem f_composition_equals_one :
  f (f (Real.sqrt 5 / 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_one_l227_22760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_necessary_not_sufficient_l227_22753

noncomputable def distance_to_line (a b : ℝ) : ℝ := |a + b| / Real.sqrt 2

def is_tangent (a b : ℝ) : Prop := distance_to_line a b = Real.sqrt 2

theorem tangent_line_necessary_not_sufficient :
  (∀ a b : ℝ, a + b = 2 → is_tangent a b) ∧
  ¬(∀ a b : ℝ, is_tangent a b → a + b = 2) := by
  sorry

#check tangent_line_necessary_not_sufficient

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_necessary_not_sufficient_l227_22753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_inequality_l227_22788

/-- Predicate to represent that a, b, c, d are sides and S is the area of a quadrilateral -/
def is_quadrilateral (a b c d S : ℝ) : Prop := sorry

/-- The area of a quadrilateral is less than or equal to half the sum of the products of opposite sides. -/
theorem quadrilateral_area_inequality (a b c d S : ℝ) 
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : d > 0) 
  (h₅ : S > 0) (h₆ : is_quadrilateral a b c d S) : 
  S ≤ (1/2) * (a*b + c*d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_inequality_l227_22788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l227_22765

-- Define the function f as noncomputable
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 
  if x > 0 then (4 : ℝ)^(m - x) else (4 : ℝ)^(m + x)

-- State the theorem
theorem find_m : ∃ m : ℝ, 
  (∀ x, f m x = f m (-x)) ∧  -- f is even
  (f m (-2) = 1/8) ∧         -- f(-2) = 1/8
  (m = 1/2) := by            -- m = 1/2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l227_22765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l227_22714

theorem sin_2alpha_value (α β : ℝ) 
  (h1 : π / 2 < β) 
  (h2 : β < α) 
  (h3 : α < 3 * π / 4)
  (h4 : Real.cos (α - β) = 12 / 13)
  (h5 : Real.sin (α + β) = -3 / 5) : 
  Real.sin (2 * α) = -56 / 65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l227_22714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_l227_22723

/-- The function f(x) = cos(ω*x + φ) -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + φ)

/-- The period of the cosine function -/
noncomputable def period (ω : ℝ) : ℝ := 2 * Real.pi / ω

theorem min_omega (ω φ : ℝ) (h1 : ω > 0) (h2 : 0 < φ) (h3 : φ < Real.pi)
  (h4 : f ω φ (period ω) = Real.sqrt 3 / 2)
  (h5 : f ω φ (Real.pi / 9) = 0) :
  ω ≥ 3 ∧ ∃ ω₀, ω₀ = 3 ∧ f ω₀ φ (period ω₀) = Real.sqrt 3 / 2 ∧ f ω₀ φ (Real.pi / 9) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_l227_22723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_and_largest_interesting_numbers_l227_22766

/-- A function to check if a natural number is a palindrome -/
def is_palindrome (n : ℕ) : Prop :=
  ∃ (d : ℕ), n.digits d = (n.digits d).reverse

/-- Definition of an interesting number -/
def is_interesting (n : ℕ) : Prop :=
  is_palindrome n ∧ is_palindrome (n + 2023)

/-- Theorem stating the smallest and largest interesting numbers -/
theorem smallest_and_largest_interesting_numbers :
  (∀ n : ℕ, is_interesting n → n ≥ 969) ∧
  (∀ n : ℕ, is_interesting n → n ≤ 8778) ∧
  is_interesting 969 ∧
  is_interesting 8778 := by
  sorry

#check smallest_and_largest_interesting_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_and_largest_interesting_numbers_l227_22766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_reciprocal_l227_22774

open Real

-- Define the function f(x) = -1/x
noncomputable def f (x : ℝ) : ℝ := -1/x

-- State the theorem
theorem derivative_of_reciprocal (x : ℝ) (h : x ≠ 0) : 
  deriv f x = 1 / x^2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_reciprocal_l227_22774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_ratio_is_one_fourth_l227_22776

/-- Represents a square in the arrangement -/
structure Square where
  sideLength : ℝ
  overlapLength : ℝ

/-- Represents the arrangement of four squares -/
structure SquareArrangement where
  squares : Fin 4 → Square
  overlapArea : ℝ
  intersectionArea : ℝ

/-- The ratio of overlapping area to total area in the square arrangement -/
noncomputable def overlapRatio (arr : SquareArrangement) : ℝ :=
  (arr.overlapArea + arr.intersectionArea) / (4 * (arr.squares 0).sideLength ^ 2)

/-- Theorem stating the overlap ratio is 1/4 for the given arrangement -/
theorem overlap_ratio_is_one_fourth (arr : SquareArrangement) 
  (h1 : ∀ i, (arr.squares i).sideLength = 1)
  (h2 : ∀ i, (arr.squares i).overlapLength = 1/4)
  (h3 : arr.overlapArea = 3/16)
  (h4 : arr.intersectionArea = 1/16) :
  overlapRatio arr = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_ratio_is_one_fourth_l227_22776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_two_thirds_l227_22702

noncomputable section

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the tangent line at x = 2
def tangent_line (x : ℝ) : ℝ := 4*x - 4

-- Define the curvilinear area
noncomputable def curvilinear_area : ℝ :=
  ∫ x in (0)..(2), parabola x - ∫ x in (1)..(2), tangent_line x

-- Theorem statement
theorem area_equals_two_thirds : curvilinear_area = 2/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_two_thirds_l227_22702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_fraction_implies_prime_exponent_l227_22710

theorem prime_power_fraction_implies_prime_exponent (n : ℕ) (h1 : n > 2) :
  (∃ b : ℕ+, ∃ p : ℕ, ∃ k : ℕ+, (b.val^n - 1) / (b.val - 1) = p^k.val) → Prime n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_fraction_implies_prime_exponent_l227_22710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_correct_largest_rectangle_correct_l227_22795

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

-- Define the side length of the largest square
noncomputable def largest_square_side (t : RightTriangle) : ℝ :=
  (t.a * t.b) / (t.a + t.b)

-- Define the dimensions of the largest rectangle
noncomputable def largest_rectangle_dims (t : RightTriangle) : ℝ × ℝ :=
  (t.a / 2, t.b / 2)

-- Theorem for the largest square
theorem largest_square_correct (t : RightTriangle) :
  largest_square_side t = (t.a * t.b) / (t.a + t.b) :=
by
  -- Unfold the definition of largest_square_side
  unfold largest_square_side
  -- The equality holds by definition
  rfl

-- Theorem for the largest rectangle
theorem largest_rectangle_correct (t : RightTriangle) :
  largest_rectangle_dims t = (t.a / 2, t.b / 2) :=
by
  -- Unfold the definition of largest_rectangle_dims
  unfold largest_rectangle_dims
  -- The equality holds by definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_correct_largest_rectangle_correct_l227_22795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_l227_22797

/-- The function f(x) defined as √3 sin(2x) + 3sin²(x) + cos²(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 3 * (Real.sin x)^2 + (Real.cos x)^2

/-- Theorem stating that if f(x) is increasing on [-a, a], then a is in the range (0, π/6] -/
theorem f_increasing_implies_a_range (a : ℝ) :
  (∀ x y, x ∈ Set.Icc (-a) a → y ∈ Set.Icc (-a) a → x < y → f x < f y) →
  a ∈ Set.Ioo 0 (Real.pi / 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_l227_22797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_polynomial_characterization_l227_22716

theorem constant_polynomial_characterization (P : Polynomial ℂ) :
  (∀ X : ℂ, P.eval (2 * X) = P.eval (X - 1) * P.eval 1) →
  ∃ α : ℂ, P = Polynomial.C α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_polynomial_characterization_l227_22716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l227_22789

theorem trig_problem (α : ℝ) 
  (h : (Real.tan α - 3) * (Real.sin α + Real.cos α + 3) = 0) : 
  ((4 * Real.sin α + 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 1) ∧ 
  (2 + (2/3) * (Real.sin α)^2 + (1/4) * (Real.cos α)^2 = 21/8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l227_22789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_above_line_l227_22709

-- Define the circle equation
def circle_eq (p : ℝ × ℝ) : Prop := 
  let (x, y) := p
  x^2 - 4*x + y^2 - 10*y + 21 = 0

-- Define the line equation
def line_eq (y : ℝ) : Prop := y = 3

-- Define the area of the circle above the line
noncomputable def area_above_line (circle : (ℝ × ℝ) → Prop) (line : ℝ → Prop) : ℝ := 
  6 * Real.pi + 4 * Real.sqrt 2

-- Theorem statement
theorem circle_area_above_line :
  area_above_line circle_eq line_eq = 6 * Real.pi + 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_above_line_l227_22709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_range_l227_22769

-- Define a scalene triangle
structure ScaleneTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  a_ne_b : a ≠ b
  b_ne_c : b ≠ c
  a_ne_c : a ≠ c
  a_longest : a ≥ b ∧ a ≥ c

-- Define the angle A in radians
noncomputable def angle_A (t : ScaleneTriangle) : ℝ := Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))

-- The main theorem
theorem angle_A_range (t : ScaleneTriangle) (h : t.a^2 < t.b^2 + t.c^2) :
  Real.pi / 3 < angle_A t ∧ angle_A t < Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_range_l227_22769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_14_power_100_l227_22758

theorem unit_digit_14_power_100 : ∃ n : ℕ, 14^100 ≡ 6 [MOD 10] :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_14_power_100_l227_22758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_per_tree_is_40_l227_22707

/-- Represents the forest with given properties --/
structure Forest where
  plank_yield_per_hectare : ℚ
  plank_yield_per_tree : ℚ
  hectare_to_square_meters : ℚ

/-- Calculates the area occupied by one tree in square meters --/
def area_per_tree (f : Forest) : ℚ :=
  (f.hectare_to_square_meters * f.plank_yield_per_tree) / f.plank_yield_per_hectare

/-- The main theorem stating that the integer part of the area occupied by one tree is 40 --/
theorem area_per_tree_is_40 (f : Forest) 
  (h1 : f.plank_yield_per_hectare = 100)
  (h2 : f.plank_yield_per_tree = 2/5)
  (h3 : f.hectare_to_square_meters = 10000) :
  Int.floor (area_per_tree f) = 40 := by
  sorry

def example_forest : Forest := {
  plank_yield_per_hectare := 100,
  plank_yield_per_tree := 2/5,
  hectare_to_square_meters := 10000
}

#eval Int.floor (area_per_tree example_forest)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_per_tree_is_40_l227_22707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_elevation_locus_l227_22750

/-- Given two flagpoles with heights 5 meters and 3 meters at coordinates (-5,0) and (5,0) respectively,
    the locus of points on the ground where the angles of elevation to the tops of the poles are equal
    is a circle with equation (x - 85/8)² + y² = (75/8)² -/
theorem flagpole_elevation_locus (x y : ℝ) : 
  (∀ (θ : ℝ), Real.tan θ = (5 - y) / (x + 5) ↔ Real.tan θ = (3 - y) / (x - 5)) ↔ 
  (x - 85/8)^2 + y^2 = (75/8)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_elevation_locus_l227_22750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_travel_distance_l227_22729

theorem circle_center_travel_distance 
  (a b c : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_side_lengths : a = 9 ∧ b = 12 ∧ c = 15) 
  (r : ℝ) 
  (h_radius : r = 2) :
  (a - 2*r) + (b - 2*r) + (c - 2*r) = 24 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_travel_distance_l227_22729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_second_quadrant_l227_22781

theorem sin_minus_cos_second_quadrant (θ : ℝ) (m : ℝ) :
  (π / 2 < θ ∧ θ < π) →
  (2 * (Real.sin θ)^2 + (Real.sqrt 3 - 1) * Real.sin θ + m = 0) →
  (2 * (Real.cos θ)^2 + (Real.sqrt 3 - 1) * Real.cos θ + m = 0) →
  Real.sin θ - Real.cos θ = (1 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_second_quadrant_l227_22781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_sequence_values_l227_22727

def sequenceN (n : ℕ+) : ℚ := (-1)^n.val / n.val

theorem sequence_formula (n : ℕ+) : sequenceN n = (-1)^n.val / n.val := by
  rfl

theorem sequence_values :
  sequenceN 1 = -1 ∧
  sequenceN 2 = 1/2 ∧
  sequenceN 3 = -1/3 ∧
  sequenceN 4 = 1/4 ∧
  sequenceN 5 = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_sequence_values_l227_22727
