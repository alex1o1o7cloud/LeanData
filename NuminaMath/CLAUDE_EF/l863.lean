import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_modulus_l863_86317

theorem complex_number_modulus : Complex.abs ((1 - Complex.I)^2 / (1 + Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_modulus_l863_86317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l863_86344

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Ioo (-1 : ℝ) 1 → x₂ ∈ Set.Ioo (-1 : ℝ) 1 →
    f x₁ + f x₂ = f ((x₁ + x₂) / (1 + x₁ * x₂))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l863_86344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_coplanar_iff_k_eq_neg_one_or_neg_five_l863_86309

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Check if three vectors are linearly dependent -/
def linearlyDependent (v1 v2 v3 : ℝ × ℝ × ℝ) : Prop :=
  let (x1, y1, z1) := v1
  let (x2, y2, z2) := v2
  let (x3, y3, z3) := v3
  x1 * (y2 * z3 - y3 * z2) - 
  y1 * (x2 * z3 - x3 * z2) + 
  z1 * (x2 * y3 - x3 * y2) = 0

/-- Check if two lines are coplanar -/
def coplanar (l1 l2 : Line3D) : Prop :=
  let v := (l2.point.1 - l1.point.1, l2.point.2.1 - l1.point.2.1, l2.point.2.2 - l1.point.2.2)
  linearlyDependent l1.direction l2.direction v

/-- The main theorem -/
theorem lines_coplanar_iff_k_eq_neg_one_or_neg_five (k : ℝ) :
  let l1 := Line3D.mk (3, 2, 5) (2, 1, -k)
  let l2 := Line3D.mk (4, 5, 6) (k, 3, 2)
  coplanar l1 l2 ↔ k = -1 ∨ k = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_coplanar_iff_k_eq_neg_one_or_neg_five_l863_86309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_cover_iff_condition_l863_86308

/-- Represents a parallelogram ABCD with the given properties -/
structure Parallelogram where
  a : ℝ
  α : ℝ
  is_acute : 0 < α ∧ α < π/2
  is_parallelogram : True

/-- Predicate to check if four unit circles cover the parallelogram -/
def circles_cover (p : Parallelogram) : Prop :=
  ∃ (kA kB kC kD : Set (ℝ × ℝ)),
    (∀ x y, (x, y) ∈ kA ↔ (x - 0)^2 + (y - 0)^2 ≤ 1) ∧
    (∀ x y, (x, y) ∈ kB ↔ (x - p.a)^2 + y^2 ≤ 1) ∧
    (∀ x y, (x, y) ∈ kC ↔ (x - p.a)^2 + (y - 1)^2 ≤ 1) ∧
    (∀ x y, (x, y) ∈ kD ↔ x^2 + (y - 1)^2 ≤ 1) ∧
    (∀ x y, 0 ≤ x ∧ x ≤ p.a ∧ 0 ≤ y ∧ y ≤ 1 → (x, y) ∈ kA ∪ kB ∪ kC ∪ kD)

/-- The main theorem to be proved -/
theorem circles_cover_iff_condition (p : Parallelogram) :
  circles_cover p ↔ p.a ≤ Real.cos p.α + Real.sqrt 3 * Real.sin p.α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_cover_iff_condition_l863_86308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_max_lambda_l863_86332

theorem inequality_max_lambda : 
  (∃ lambda_max : ℝ, 
    (∀ a b : ℝ, a > 0 → b > 0 → lambda_max / (a + b) ≤ 1/a + 2/b) ∧ 
    (∀ lambda : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → lambda / (a + b) ≤ 1/a + 2/b) → lambda ≤ lambda_max) ∧
    lambda_max = 3 + 2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_max_lambda_l863_86332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l863_86318

/-- The slope angle of a line given by the equation x + (a² + 1)y + 1 = 0 -/
noncomputable def slope_angle (a : ℝ) : ℝ :=
  Real.arctan (-1 / (a^2 + 1))

theorem slope_angle_range :
  ∀ a : ℝ, slope_angle a ∈ Set.Icc (3 * Real.pi / 4) Real.pi :=
by
  sorry

#check slope_angle_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l863_86318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_proof_l863_86304

theorem starting_number_proof : 
  ∃ x : ℕ, x < 114 ∧ 
    (∀ y : ℕ, y < x → y + 1 < 114) ∧
    (Finset.filter (λ n : ℕ ↦ 19 ∣ n ∧ n > x ∧ n ≤ 500) (Finset.range 501)).card = 21 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_proof_l863_86304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_area_equality_circle_circumference_area_close_l863_86300

-- Define π as a constant (approximation)
noncomputable def π : ℝ := Real.pi

-- Define the diameter of the circle
def diameter : ℝ := 4

-- Define the radius of the circle
noncomputable def radius : ℝ := diameter / 2

-- Define the circumference of the circle
noncomputable def circumference : ℝ := π * diameter

-- Define the area of the circle
noncomputable def area : ℝ := π * radius^2

-- Theorem statement
theorem circle_circumference_area_equality : 
  circumference ≠ area :=
sorry

-- Theorem to show that circumference and area are close in value
theorem circle_circumference_area_close (ε : ℝ) (h : ε > 0) : 
  |circumference - area| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_area_equality_circle_circumference_area_close_l863_86300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l863_86355

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (1 - x^2) + x^0

-- Define the domain of f
def domain_f : Set ℝ := Set.Ioo (-1) 0 ∪ Set.Ioo 0 1

-- Theorem statement
theorem domain_of_f : 
  {x : ℝ | (1 - x^2 > 0) ∧ (x ≠ 0)} = domain_f :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l863_86355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_staircase_perimeter_is_63_5_l863_86362

/-- Represents the properties of a staircase-shaped region -/
structure StaircaseRegion where
  rectangle_width : ℝ
  region_area : ℝ
  congruent_side_length : ℝ
  congruent_side_count : ℕ

/-- Calculates the perimeter of the staircase-shaped region -/
noncomputable def calculate_perimeter (s : StaircaseRegion) : ℝ :=
  let rectangle_height := (s.region_area + s.congruent_side_count * s.congruent_side_length ^ 2) / s.rectangle_width
  rectangle_height + s.rectangle_width + s.congruent_side_count * s.congruent_side_length

/-- Theorem stating that for the given staircase region, the perimeter is 63.5 feet -/
theorem staircase_perimeter_is_63_5 :
  let s : StaircaseRegion := {
    rectangle_width := 12,
    region_area := 162,
    congruent_side_length := 2,
    congruent_side_count := 6
  }
  calculate_perimeter s = 63.5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_staircase_perimeter_is_63_5_l863_86362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_is_conic_or_frustum_l863_86313

/-- Represents a point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Describes the shape in spherical coordinates -/
def Shape (a b : ℝ) : Set SphericalPoint :=
  {p : SphericalPoint | p.ρ > 0 ∧ 0 ≤ p.θ ∧ p.θ < 2 * Real.pi ∧ a ≤ p.φ ∧ p.φ ≤ b}

/-- Represents a cone or conic frustum -/
inductive ConicShape
  | Cone
  | ConicFrustum

/-- The theorem stating that the shape is either a cone or a conic frustum -/
theorem shape_is_conic_or_frustum (a b : ℝ) (h1 : 0 ≤ a) (h2 : a < b) (h3 : b ≤ Real.pi) :
  ∃ (s : ConicShape), Shape a b ≠ ∅ ∧
    ((s = ConicShape.Cone ∧ a = 0 ∧ b = Real.pi) ∨
     (s = ConicShape.ConicFrustum ∧ (a > 0 ∨ b < Real.pi))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_is_conic_or_frustum_l863_86313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_math_problems_not_set_l863_86353

/-- Represents a group of objects that may or may not form a set -/
inductive ObjectGroup
  | TableTennisPlayers
  | PositiveIntegersLessThan5
  | MathProblems2023
  | IrrationalNumbers

/-- Predicate to determine if a group can form a well-defined set -/
def canFormSet (g : ObjectGroup) : Prop :=
  match g with
  | ObjectGroup.TableTennisPlayers => True
  | ObjectGroup.PositiveIntegersLessThan5 => True
  | ObjectGroup.MathProblems2023 => False
  | ObjectGroup.IrrationalNumbers => True

theorem only_math_problems_not_set :
  ∀ g : ObjectGroup, ¬(canFormSet g) ↔ g = ObjectGroup.MathProblems2023 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_math_problems_not_set_l863_86353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_sqrt_radii_S_eq_answer_l863_86379

/-- Represents a circle in the construction -/
structure Circle where
  radius : ℝ

/-- Represents a layer of circles in the construction -/
def Layer := List Circle

/-- The initial two circles in the upper half-plane -/
def initial_circles : Layer := [
  { radius := 60^2 },
  { radius := 65^2 }
]

/-- Generates the next layer of circles based on the current layer -/
def next_layer (current : Layer) : Layer :=
  sorry

/-- Generates layers up to the given index -/
def generate_layers (n : ℕ) : List Layer :=
  sorry

/-- The set S of all circles from L₀ to L₅ -/
def S : List Circle :=
  sorry

/-- The sum of inverse square roots of radii for circles in S -/
noncomputable def sum_inverse_sqrt_radii (circles : List Circle) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem sum_inverse_sqrt_radii_S_eq_answer : 
  ∃ (answer : ℝ), sum_inverse_sqrt_radii S = answer ∧ answer ∈ ({1.25, 2.50, 3.75, 5.00} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_sqrt_radii_S_eq_answer_l863_86379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_lcms_l863_86328

theorem lcm_of_lcms : Nat.lcm (Nat.lcm 12 16) (Nat.lcm 18 24) = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_lcms_l863_86328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_perfect_square_product_l863_86369

theorem exists_non_perfect_square_product (d : ℕ) 
  (h1 : d > 0) 
  (h2 : d ≠ 2) 
  (h3 : d ≠ 5) 
  (h4 : d ≠ 13) : 
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ a ≠ b ∧ ¬∃ (k : ℕ), a * b - 1 = k^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_perfect_square_product_l863_86369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l863_86376

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if -1 < x ∧ x < 2 then x^2
  else 0  -- This case should never occur based on the domain, but Lean requires a complete function definition

-- State the theorem
theorem f_properties :
  (∀ y ∈ Set.range f, y < 4) ∧
  (∀ y ∈ Set.range f, ∃ x, f x = y) ∧
  (∀ x, f x = 3 → x = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l863_86376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l863_86384

noncomputable def f (x : ℝ) : ℝ := -2/3 * x^3 + 3/2 * x^2 - x

theorem f_increasing_interval :
  ∃ (a b : ℝ), a = 1/2 ∧ b = 1 ∧
  (∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, x ≤ y → f x ≤ f y) ∧
  (∀ c d : ℝ, c < a ∨ b < d →
    ¬(∀ x ∈ Set.Icc c d, ∀ y ∈ Set.Icc c d, x ≤ y → f x ≤ f y)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l863_86384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_theorem_l863_86367

/-- Represents a chord on a circle --/
structure Chord (n : ℕ) where
  start : Fin n
  finish : Fin n
  ne : start ≠ finish

/-- Represents a configuration of chords on a circle --/
def ChordConfiguration (n k : ℕ) := { chords : Finset (Chord n) // chords.card = 2 * k }

/-- Predicate for non-intersecting chords --/
def NonIntersecting (n : ℕ) (chords : Finset (Chord n)) : Prop :=
  ∀ c1 c2, c1 ∈ chords → c2 ∈ chords → c1 ≠ c2 →
    c1.start ≠ c2.start ∧ c1.start ≠ c2.finish ∧ c1.finish ≠ c2.start ∧ c1.finish ≠ c2.finish

/-- Predicate for chords with endpoints differing by at most m --/
def DifferByAtMost (n m : ℕ) (c : Chord n) : Prop :=
  (c.finish.val - c.start.val) % n ≤ m ∨ (c.start.val - c.finish.val) % n ≤ m

theorem circle_chord_theorem (k : ℕ) :
  ∃ (config : ChordConfiguration (4 * k) k),
    NonIntersecting (4 * k) config.val ∧
    (∀ c ∈ config.val, DifferByAtMost (4 * k) (3 * k - 1) c) ∧
    ¬∃ (config' : ChordConfiguration (4 * k) k),
      NonIntersecting (4 * k) config'.val ∧
      (∀ c ∈ config'.val, DifferByAtMost (4 * k) ((3 * k - 1) - 1) c) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_theorem_l863_86367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_area_l863_86399

theorem square_diagonal_area (a b : ℝ) :
  let diagonal := a - b
  let area := (1 / 2) * (a - b)^2
  diagonal^2 = 2 * area := by
  intro diagonal area
  have h1 : diagonal^2 = (a - b)^2 := by rfl
  have h2 : area = (1 / 2) * (a - b)^2 := by rfl
  calc
    diagonal^2 = (a - b)^2 := h1
    _ = 2 * ((1 / 2) * (a - b)^2) := by ring
    _ = 2 * area := by rw [h2]

#check square_diagonal_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_area_l863_86399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_mappings_l863_86350

-- Define the sets and functions for each relationship
def A1 : Set ℕ := {1, 2, 3}
def B1 : Set ℕ := {0, 1, 4, 5, 9, 10}
def f1 : ℕ → ℕ := λ x => x^2

noncomputable def A2 : Set ℝ := Set.univ
noncomputable def B2 : Set ℝ := Set.univ
noncomputable def f2 : ℝ → ℝ := λ x => 1 / x

def A3 : Set ℕ := Set.univ
def B3 : Set ℕ := {x : ℕ | x ≠ 0}
def f3 : ℕ → ℕ := λ x => x^2

def A4 : Set ℤ := Set.univ
def B4 : Set ℤ := Set.univ
def f4 : ℤ → ℤ := λ x => 2*x - 1

-- Define what it means for a function to be a valid mapping
def is_valid_mapping {A B : Type} (f : A → B) (domain : Set A) (codomain : Set B) : Prop :=
  ∀ x ∈ domain, f x ∈ codomain

-- State the theorem
theorem valid_mappings :
  (is_valid_mapping f1 A1 B1) ∧
  (¬ is_valid_mapping f2 A2 B2) ∧
  (¬ is_valid_mapping f3 A3 B3) ∧
  (is_valid_mapping f4 A4 B4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_mappings_l863_86350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_functional_classification_l863_86382

def is_bounded_functional (f : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ x : ℝ, |f x| ≤ M * |x|

def f₁ : ℝ → ℝ := λ x ↦ -3 * x
def f₂ : ℝ → ℝ := λ x ↦ x^2
noncomputable def f₃ : ℝ → ℝ := λ x ↦ Real.sin x ^ 2
noncomputable def f₄ : ℝ → ℝ := λ x ↦ 2^x
noncomputable def f₅ : ℝ → ℝ := λ x ↦ x * Real.cos x

theorem bounded_functional_classification :
  is_bounded_functional f₁ ∧
  ¬is_bounded_functional f₂ ∧
  is_bounded_functional f₃ ∧
  ¬is_bounded_functional f₄ ∧
  is_bounded_functional f₅ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_functional_classification_l863_86382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_capacity_value_l863_86336

def initial_machines : ℕ := sorry
def initial_capacity : ℕ := sorry
def modernized_capacity : ℕ := sorry

axiom total_initial : initial_machines * initial_capacity = 38880
axiom total_modernized : (initial_machines + 3) * modernized_capacity = 44800
axiom capacity_increased : initial_capacity < modernized_capacity

theorem initial_capacity_value : initial_capacity = 1215 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_capacity_value_l863_86336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_plus_one_l863_86398

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_plus_one (x : ℝ) : floor (x + 1) = floor x + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_plus_one_l863_86398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_even_2012_l863_86396

/-- f(n) represents the last digit of the sum 1 + 2 + 3 + ... + n -/
def f (n : Nat) : Nat := (n * (n + 1) / 2) % 10

/-- The sum of f(2n) for n from 1 to 1006 -/
def sum_f_even (m : Nat) : Nat := Finset.sum (Finset.range m) (fun n => f (2 * (n + 1)))

theorem sum_f_even_2012 : sum_f_even 1006 = 3523 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_even_2012_l863_86396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tau_eq_n_div_3_iff_in_S_l863_86314

/-- τ(n) is the number of positive factors of n -/
def tau (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The set of numbers that satisfy τ(n) = n/3 -/
def S : Set ℕ := {9, 18, 24}

/-- Theorem: n satisfies τ(n) = n/3 if and only if n is in the set S -/
theorem tau_eq_n_div_3_iff_in_S (n : ℕ) : tau n = n / 3 ↔ n ∈ S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tau_eq_n_div_3_iff_in_S_l863_86314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_points_l863_86372

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

-- Define a set of 6 points inside the rectangle
def points : Finset (ℝ × ℝ) :=
  sorry

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem exists_close_points :
  ∃ p q, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ distance p q ≤ Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_points_l863_86372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l863_86311

/-- The curve C₁ in polar coordinates -/
noncomputable def C₁ (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

/-- The line l in polar coordinates -/
noncomputable def line_l (ρ θ : ℝ) (a : ℝ) : Prop := ρ * Real.cos (θ - Real.pi/4) = a

/-- Point A in polar coordinates -/
noncomputable def point_A : ℝ × ℝ := (3 * Real.sqrt 2, Real.pi/4)

/-- The line l' in polar coordinates -/
noncomputable def line_l' (ρ θ : ℝ) : Prop := θ = 3*Real.pi/4

/-- The theorem statement -/
theorem intersection_length :
  ∃ (a : ℝ),
    line_l point_A.1 point_A.2 a ∧
    (∀ ρ θ, line_l' ρ θ → C₁ ρ θ →
      ∃ M N : ℝ × ℝ,
        (M.1 - N.1)^2 + (M.2 - N.2)^2 = 32) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l863_86311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_range_m_l863_86343

-- Define the functions f and g
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x + 1/4
noncomputable def g (x : ℝ) : ℝ := -Real.log x

-- Define the function h as the minimum of f and g
noncomputable def h (m : ℝ) (x : ℝ) : ℝ := min (f m x) (g x)

-- State the theorem
theorem three_zeros_range_m :
  ∀ m : ℝ, (∃! (s : Set ℝ), s.Finite ∧ s.ncard = 3 ∧ ∀ x ∈ s, x > 0 ∧ h m x = 0) →
  m > -5/4 ∧ m < -3/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_range_m_l863_86343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_pq_length_l863_86341

noncomputable section

open Real EuclideanGeometry

theorem right_triangle_pq_length 
  (P Q R : EuclideanSpace ℝ (Fin 2))
  (h_right_angle : ∠ Q P R = π / 2)
  (h_qpr_angle : ∠ Q P R = π / 4)
  (h_rp_length : dist R P = 12) :
  dist P Q = 12 * Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_pq_length_l863_86341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_M_N_l863_86307

-- Define the sets M and N
def M : Set ℝ := {x | x ≥ 1}
def N : Set ℝ := {x | x < 2}

-- State the theorem
theorem complement_intersection_M_N :
  (M ∩ N)ᶜ = Set.Iic 1 ∪ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_M_N_l863_86307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_phi_l863_86316

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The given function f(x) = 2sin(2x + 3φ) -/
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x + 3 * φ)

/-- Theorem stating that if f is odd, then φ is of the form π/3 + kπ/3 for some integer k -/
theorem odd_function_phi (φ : ℝ) (h : IsOdd (f φ)) :
  ∃ k : ℤ, φ = Real.pi / 3 + k * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_phi_l863_86316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_infinite_l863_86375

theorem ramsey_infinite (k c : ℕ+) (X : Set α) (h_infinite : Set.Infinite X) 
  (coloring : Finset α → Fin c) :
  ∃ (Y : Set α), Set.Infinite Y ∧ Y ⊆ X ∧ 
  ∀ (S T : Finset α), S.card = k → T.card = k → S.toSet ⊆ Y → T.toSet ⊆ Y → coloring S = coloring T :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_infinite_l863_86375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equations_l863_86393

noncomputable def x (t : ℝ) : ℝ := 2 - Real.exp (-t)

noncomputable def y (t : ℝ) : ℝ := -4 + Real.exp (-t) + 2 * Real.exp t

def integral_equation_1 (x : ℝ → ℝ) (t : ℝ) : Prop :=
  x t = 1 + ∫ s in Set.Icc 0 t, Real.exp (-2 * (t - s)) * x s

def integral_equation_2 (x y : ℝ → ℝ) (t : ℝ) : Prop :=
  y t = ∫ s in Set.Icc 0 t, Real.exp (-2 * (t - s)) * (2 * x s + 3 * y s)

theorem solution_satisfies_equations :
  ∀ t : ℝ, integral_equation_1 x t ∧ integral_equation_2 x y t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equations_l863_86393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l863_86356

def M : ℕ := 57^4 + 4*57^3 + 6*57^2 + 4*57 + 1

theorem number_of_factors_of_M : (Finset.filter (λ x : ℕ => x ∣ M) (Finset.range (M + 1))).card = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l863_86356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_value_l863_86334

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin (2 * x) + (1 / 2) * Real.cos (2 * x)

theorem min_shift_value (φ : ℝ) (h1 : φ > 0) 
  (h2 : ∀ x, f x = Real.sin (2 * x + 2 * φ)) : 
  ∃ k : ℤ, φ = π / 12 + k * π ∧ ∀ m : ℤ, φ ≤ π / 12 + m * π := by
  sorry

#check min_shift_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_value_l863_86334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_equality_l863_86302

/-- Given a triangle ABC with points D and E on sides AB and AC respectively,
    and F as the intersection point of BE and CD, if AE + EF = AD + DF,
    then AC + CF = AB + BF -/
theorem triangle_segment_equality (A B C D E F : EuclideanSpace ℝ (Fin 2)) :
  (∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ 0 ≤ t₂ ∧ t₂ ≤ 1 ∧
    D = (1 - t₁) • A + t₁ • B ∧
    E = (1 - t₂) • A + t₂ • C) →
  (∃ s₁ s₂ : ℝ, F = (1 - s₁) • B + s₁ • E ∧
               F = (1 - s₂) • C + s₂ • D) →
  dist A E + dist E F = dist A D + dist D F →
  dist A C + dist C F = dist A B + dist B F := by
  sorry

#check triangle_segment_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_equality_l863_86302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_equality_l863_86319

theorem trig_expression_equality :
  1 / Real.sin (10 * π / 180) - Real.sqrt 3 / Real.cos (10 * π / 180) =
  2 * Real.sqrt 3 * (Real.cos (10 * π / 180) / Real.sin (10 * π / 180)) - 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_equality_l863_86319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l863_86390

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The sine rule for a triangle -/
axiom sine_rule (t : Triangle) : t.a / (Real.sin t.A) = t.b / (Real.sin t.B)

/-- The area formula for a triangle -/
noncomputable def triangle_area (t : Triangle) : Real := (1/2) * t.a * t.c * Real.sin t.B

theorem triangle_problem (t : Triangle) :
  (t.a + t.c) / t.b = 2 * Real.sin (t.C + π/6) →
  (t.B = π/3 ∧
   (t.b = 2 * Real.sqrt 7 → triangle_area t = 3 * Real.sqrt 3 → t.a + t.c = 8)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l863_86390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_1_5_power_200_l863_86370

def power_of_1_5 (n : ℕ) : ℚ := (3/2)^n

theorem units_digit_of_1_5_power_200 :
  (power_of_1_5 200).num % 10 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_1_5_power_200_l863_86370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_power_series_expansion_and_convergence_l863_86312

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 3*x + 2)

-- Define the power series
noncomputable def power_series (x : ℝ) : ℝ := Real.log 2 + ∑' n, (-1)^(n+1) * ((2^n + 1) / (2^n * n)) * x^n

-- State the theorem
theorem f_power_series_expansion_and_convergence :
  ∃ (R : ℝ), R > 0 ∧
  (∀ x : ℝ, -R < x ∧ x ≤ R → f x = power_series x) ∧
  R = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_power_series_expansion_and_convergence_l863_86312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cos_leq_one_l863_86392

theorem negation_of_cos_leq_one :
  (¬ ∀ x : ℝ, Real.cos x ≤ 1) ↔ (∃ x : ℝ, Real.cos x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cos_leq_one_l863_86392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_eight_thirty_l863_86342

/-- Represents a time on an analog clock -/
structure ClockTime where
  hours : ℕ
  minutes : ℕ
  hh_valid : hours < 12
  mm_valid : minutes < 60

/-- Calculates the angle of the hour hand from 12 o'clock position -/
noncomputable def hour_hand_angle (t : ClockTime) : ℝ :=
  (t.hours % 12 : ℝ) * 30 + (t.minutes : ℝ) * 0.5

/-- Calculates the angle of the minute hand from 12 o'clock position -/
noncomputable def minute_hand_angle (t : ClockTime) : ℝ :=
  (t.minutes : ℝ) * 6

/-- Calculates the angle between hour and minute hands -/
noncomputable def angle_between_hands (t : ClockTime) : ℝ :=
  abs (hour_hand_angle t - minute_hand_angle t)

/-- The theorem stating that at 8:30, the angle between the clock hands is 75° -/
theorem angle_at_eight_thirty :
  let t : ClockTime := ⟨8, 30, by norm_num, by norm_num⟩
  angle_between_hands t = 75 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_eight_thirty_l863_86342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_l863_86330

noncomputable def f (a x : ℝ) := 2*x - 2/x - a

theorem zero_point_interval (a : ℝ) :
  (∃ x ∈ Set.Ioo 1 2, f a x = 0) ↔ a ∈ Set.Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_l863_86330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_l863_86389

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the given line
noncomputable def given_line (x y : ℝ) : Prop := x + Real.sqrt 3 * y = 0

-- Define the tangent point Q
noncomputable def point_Q : ℝ × ℝ := (3, -Real.sqrt 3)

-- Define a general circle with center (a, b) and radius r
def circle_C (a b r : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

-- State the theorem
theorem circle_C_equation :
  ∃ (a b r : ℝ),
    (∀ (x y : ℝ), circle_C a b r x y ↔ 
      ((x - 4)^2 + y^2 = 4 ∨ x^2 + (y + 4*Real.sqrt 3)^2 = 36)) ∧
    (∀ (x y : ℝ), circle_C a b r x y → given_circle x y → False) ∧
    (circle_C a b r point_Q.1 point_Q.2) ∧
    (∀ (x y : ℝ), circle_C a b r x y → given_line x y → 
      x = point_Q.1 ∧ y = point_Q.2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_l863_86389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_increase_l863_86374

/-- Represents the production rate of a small enterprise before and after equipment update -/
structure ProductionRate where
  before : ℚ
  after : ℚ
  efficiency_increase : after = 1.25 * before

/-- Represents the time taken to produce a certain number of products -/
def production_time (rate : ProductionRate) (products : ℚ) : ℚ :=
  products / rate.before

theorem production_increase (rate : ProductionRate) 
  (h : production_time rate 5000 = production_time rate 6000 + 2) : 
  rate.after = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_increase_l863_86374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_is_7x4_l863_86388

/-- Represents a term in the sequence -/
structure Term where
  coefficient : ℕ
  exponent : ℕ

/-- The sequence of terms -/
def sequenceTerms : List Term := [
  ⟨1, 1⟩,
  ⟨3, 2⟩,
  ⟨5, 3⟩,
  ⟨7, 4⟩,
  ⟨9, 5⟩
]

/-- The rule for generating the next term in the sequence -/
def nextTerm (t : Term) : Term :=
  ⟨t.coefficient + 2, t.exponent + 1⟩

/-- Theorem: The fourth term in the sequence is 7x^4 -/
theorem fourth_term_is_7x4 :
  sequenceTerms.get? 3 = some ⟨7, 4⟩ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_is_7x4_l863_86388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meeting_time_l863_86361

-- Define the train lengths and initial distance
noncomputable def train_A_length : ℝ := 100
noncomputable def train_B_length : ℝ := 200
noncomputable def initial_distance : ℝ := 70

-- Define the train speeds in km/h
noncomputable def train_A_speed_kmh : ℝ := 54
noncomputable def train_B_speed_kmh : ℝ := 72

-- Convert speeds to m/s
noncomputable def train_A_speed_ms : ℝ := train_A_speed_kmh * 1000 / 3600
noncomputable def train_B_speed_ms : ℝ := train_B_speed_kmh * 1000 / 3600

-- Calculate relative speed
noncomputable def relative_speed : ℝ := train_A_speed_ms + train_B_speed_ms

-- Calculate total distance to be covered
noncomputable def total_distance : ℝ := train_A_length + train_B_length + initial_distance

-- Theorem: The time for the trains to meet is approximately 10.57 seconds
theorem trains_meeting_time :
  abs (total_distance / relative_speed - 10.57) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meeting_time_l863_86361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l863_86345

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2

theorem tangent_line_at_one :
  let tangent_slope : ℝ := (1 : ℝ) / 1 + 2 * 1
  let tangent_intercept : ℝ := f 1 - tangent_slope * 1
  (λ x : ℝ => tangent_slope * x + tangent_intercept) = (λ x : ℝ => 3 * x - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l863_86345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_diagonal_theorem_l863_86320

/-- A trapezoid with sides a, b, c, d and diagonals e and f -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  a_parallel_c : Prop  -- We change this to a proposition

/-- The theorem stating the relationship between sides and diagonals of a trapezoid -/
theorem trapezoid_diagonal_theorem (t : Trapezoid) :
  t.e^2 + t.f^2 = t.b^2 + t.d^2 + 2*t.a*t.c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_diagonal_theorem_l863_86320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_players_count_l863_86329

/-- Represents a tournament with the given conditions -/
structure Tournament where
  n : ℕ  -- Number of players excluding the 8 lowest-scoring players
  total_players : ℕ := n + 8
  points_distribution : ℚ := 2/3

/-- The total number of games played in the tournament -/
def Tournament.total_games (t : Tournament) : ℕ :=
  (t.total_players * (t.total_players - 1)) / 2

/-- The number of games played among the top n players -/
def Tournament.top_games (t : Tournament) : ℕ :=
  (t.n * (t.n - 1)) / 2

/-- The number of games played among the 8 lowest-scoring players -/
def Tournament.bottom_games : ℕ := 28

/-- The condition that two-thirds of points come from games against lowest 8 players -/
def Tournament.point_condition (t : Tournament) : Prop :=
  (t.points_distribution * (t.total_games - Tournament.bottom_games) : ℚ) =
    (t.total_games - t.top_games - Tournament.bottom_games : ℚ)

/-- The main theorem: if the point condition holds, there are 20 players -/
theorem tournament_players_count (t : Tournament) :
    t.point_condition → t.total_players = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_players_count_l863_86329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_fourth_power_sum_l863_86306

theorem sin_cos_fourth_power_sum (x : ℝ) (h : Real.sin (2 * x) = 1 / 7) :
  Real.sin x ^ 4 + Real.cos x ^ 4 = 97 / 98 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_fourth_power_sum_l863_86306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_present_value_l863_86326

/-- The depreciation rate of the machine per annum -/
noncomputable def depreciation_rate : ℝ := 0.25

/-- The number of years after which the machine's value is known -/
def years : ℕ := 3

/-- The value of the machine after 3 years -/
noncomputable def future_value : ℝ := 54000

/-- The present value of the machine -/
noncomputable def present_value : ℝ := future_value / ((1 - depreciation_rate) ^ years)

/-- Theorem stating that the present value of the machine is 128000 -/
theorem machine_present_value : ⌊present_value⌋ = 128000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_present_value_l863_86326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_with_point_properties_l863_86359

/-- An equilateral triangle with a point on its circumcircle -/
structure EquilateralTriangleWithPoint where
  -- The vertices of the equilateral triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- The point on the circumcircle
  M : ℝ × ℝ
  -- The intersection point of BM and AC
  N : ℝ × ℝ
  -- Condition that ABC is equilateral
  equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
                (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2
  -- Condition that M is on the circumcircle
  on_circumcircle : (M.1 - A.1)^2 + (M.2 - A.2)^2 = (M.1 - B.1)^2 + (M.2 - B.2)^2 ∧
                    (M.1 - B.1)^2 + (M.2 - B.2)^2 = (M.1 - C.1)^2 + (M.2 - C.2)^2
  -- Condition that MA = 4√5 and MC = 4√5
  distances : (M.1 - A.1)^2 + (M.2 - A.2)^2 = 80 ∧ (M.1 - C.1)^2 + (M.2 - C.2)^2 = 80
  -- Condition that N is on AC and BM
  intersection : (N.1 - A.1) * (C.2 - A.2) = (N.2 - A.2) * (C.1 - A.1) ∧
                 (N.1 - B.1) * (M.2 - B.2) = (N.2 - B.2) * (M.1 - B.1)

/-- The main theorem -/
theorem equilateral_triangle_with_point_properties (t : EquilateralTriangleWithPoint) :
  -- The length of MN is 20/9
  ((t.M.1 - t.N.1)^2 + (t.M.2 - t.N.2)^2) = (20/9)^2 ∧
  -- The side length of ABC is √61
  ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2) = 61 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_with_point_properties_l863_86359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stand_distance_l863_86387

/-- The distance to the bus stand in kilometers -/
noncomputable def distance_to_bus_stand : ℝ := 5

/-- The time difference in hours between the two scenarios -/
noncomputable def time_difference : ℝ := 15 / 60

theorem bus_stand_distance :
  (distance_to_bus_stand / 4 - distance_to_bus_stand / 5 = time_difference) →
  distance_to_bus_stand = 5 := by
  intro h
  -- The proof goes here
  sorry

#check bus_stand_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stand_distance_l863_86387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l863_86368

-- Define the train's properties and the time taken to pass the bridge
noncomputable def train_length : Real := 327
noncomputable def train_speed_kmh : Real := 40
noncomputable def time_to_pass : Real := 40.41

-- Define the conversion factor from km/h to m/s
noncomputable def km_per_hour_to_m_per_second : Real := 1000 / 3600

-- Calculate the train's speed in m/s
noncomputable def train_speed_ms : Real := train_speed_kmh * km_per_hour_to_m_per_second

-- Calculate the total distance traveled
noncomputable def total_distance : Real := train_speed_ms * time_to_pass

-- Define the bridge length
noncomputable def bridge_length : Real := total_distance - train_length

-- Theorem to prove
theorem bridge_length_calculation :
  ∃ ε > 0, abs (bridge_length - 122.15) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l863_86368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_ABO_line_l_min_area_equation_MO_min_PC_PD_l863_86351

noncomputable section

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0

-- Define points A and B
noncomputable def point_A (k : ℝ) : ℝ × ℝ := (-2 - 1/k, 0)
noncomputable def point_B (k : ℝ) : ℝ × ℝ := (0, 1 + 2*k)

-- Define the area of triangle ABO
noncomputable def area_ABO (k : ℝ) : ℝ := (1/2) * (1 + 2*k) * (2 + 1/k)

-- Define the fixed point M
def point_M : ℝ × ℝ := (-2, 1)

-- Define points C and D when k = -1
def point_C : ℝ × ℝ := (-1, 0)
def point_D : ℝ × ℝ := (0, -1)

-- Theorem statements
theorem min_area_ABO :
  ∃ (k : ℝ), k > 0 ∧ ∀ (k' : ℝ), k' > 0 → area_ABO k ≤ area_ABO k' ∧ area_ABO k = 4 :=
by sorry

theorem line_l_min_area :
  ∃ (k : ℝ), k > 0 ∧ area_ABO k = 4 ∧ ∀ (x y : ℝ), line_l k x y ↔ x - 2*y + 4 = 0 :=
by sorry

theorem equation_MO :
  ∀ (x y : ℝ), (x = point_M.1 ∧ y = point_M.2) ∨ (x = 0 ∧ y = 0) → x + 2*y = 0 :=
by sorry

theorem min_PC_PD :
  ∃ (P : ℝ × ℝ), (P.1 + 2*P.2 = 0) ∧
    ∀ (Q : ℝ × ℝ), (Q.1 + 2*Q.2 = 0) →
      Real.sqrt ((P.1 - point_C.1)^2 + (P.2 - point_C.2)^2) +
      Real.sqrt ((P.1 - point_D.1)^2 + (P.2 - point_D.2)^2) ≤
      Real.sqrt ((Q.1 - point_C.1)^2 + (Q.2 - point_C.2)^2) +
      Real.sqrt ((Q.1 - point_D.1)^2 + (Q.2 - point_D.2)^2) ∧
    Real.sqrt ((P.1 - point_C.1)^2 + (P.2 - point_C.2)^2) +
    Real.sqrt ((P.1 - point_D.1)^2 + (P.2 - point_D.2)^2) = 3 * Real.sqrt 10 / 5 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_ABO_line_l_min_area_equation_MO_min_PC_PD_l863_86351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_intersection_ratio_l863_86352

open Real

theorem cosine_intersection_ratio :
  ∃ (p q : ℕ), 
    (p < q) ∧ 
    (Nat.Coprime p q) ∧
    (∀ x : ℝ, cos x = cos (50 * π / 180) → 
      (∃ n : ℤ, x = (50 + 360 * n) * π / 180 ∨ x = (310 + 360 * n) * π / 180)) ∧
    (p : ℝ) / q = 50 / 260 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_intersection_ratio_l863_86352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_vertex_product_value_l863_86386

/-- Regular octagon in the complex plane -/
structure RegularOctagon where
  /-- Center of the octagon -/
  center : ℂ
  /-- Radius of the octagon -/
  radius : ℝ
  /-- Vertices of the octagon -/
  vertices : Fin 8 → ℂ
  /-- The first vertex is at center - radius -/
  first_vertex : vertices 0 = center - radius
  /-- The fifth vertex is at center + radius -/
  fifth_vertex : vertices 4 = center + radius
  /-- All vertices are equidistant from the center -/
  equidistant : ∀ k, Complex.abs (vertices k - center) = radius

/-- The product of complex numbers representing the vertices of a regular octagon -/
def octagonVertexProduct (oct : RegularOctagon) : ℂ :=
  Finset.prod Finset.univ (λ k => oct.vertices k)

/-- Theorem stating the product of vertices of a specific regular octagon -/
theorem octagon_vertex_product_value :
  ∃ oct : RegularOctagon, oct.center = 3 ∧ oct.radius = 1 ∧ octagonVertexProduct oct = 6559 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_vertex_product_value_l863_86386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l863_86391

-- Define the line l in rectangular coordinates
def line_l (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the curve C in rectangular coordinates
def curve_C (x y : ℝ) : Prop := x^2 + y^2/3 = 1

-- Define the distance function from a point (x, y) to the line l
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x + y - 3| / Real.sqrt 2

-- Theorem statement
theorem min_distance_curve_to_line :
  ∃ (min_d : ℝ), min_d = Real.sqrt 2 / 2 ∧
  ∀ (x y : ℝ), curve_C x y →
  distance_to_line x y ≥ min_d :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l863_86391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_b_value_l863_86337

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the hyperbola
def hyperbola (x y b : ℝ) : Prop := x^2/8 - y^2/b = 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 2)

-- Define the left vertex of the hyperbola
noncomputable def left_vertex (b : ℝ) : ℝ × ℝ := (-2*Real.sqrt 2, 0)

-- Define the slope of the asymptote
noncomputable def asymptote_slope (b : ℝ) : ℝ := Real.sqrt b / (2 * Real.sqrt 2)

-- Define the slope of line AF
noncomputable def af_slope : ℝ := 1 / Real.sqrt 2

-- Theorem statement
theorem hyperbola_b_value :
  ∀ b : ℝ,
  (∀ x y : ℝ, parabola x y → hyperbola x y b) →
  (asymptote_slope b = af_slope) →
  b = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_b_value_l863_86337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l863_86381

noncomputable def f (x : ℝ) : ℝ := 2^(-abs x)

theorem range_of_f :
  (∀ y ∈ Set.range f, 0 < y ∧ y ≤ 1) ∧
  (∀ y ∈ Set.Ioo 0 1, ∃ x, f x = y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l863_86381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_parabola_l863_86377

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a point in Cartesian coordinates -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Converts a point from polar to Cartesian coordinates -/
noncomputable def polar_to_cartesian (p : PolarPoint) : CartesianPoint :=
  { x := p.r * Real.cos p.θ
    y := p.r * Real.sin p.θ }

/-- The polar equation of the curve -/
def polar_equation (p : PolarPoint) : Prop :=
  p.r = 1 / (1 - Real.cos p.θ)

/-- The Cartesian equation of a parabola -/
def parabola_equation (p : CartesianPoint) : Prop :=
  p.y^2 = 2 * p.x + 1

/-- Theorem stating that the polar equation represents a parabola -/
theorem polar_equation_is_parabola :
  ∀ p : PolarPoint, polar_equation p →
  ∃ q : CartesianPoint, q = polar_to_cartesian p ∧ parabola_equation q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_parabola_l863_86377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_for_2km_l863_86373

/-- The number of revolutions a wheel makes to cover a given distance -/
noncomputable def wheelRevolutions (wheelDiameter : ℝ) (distance : ℝ) : ℝ :=
  distance / (wheelDiameter * Real.pi)

/-- Conversion factor from kilometers to feet -/
def kmToFeet : ℝ := 3280.84

theorem wheel_revolutions_for_2km (ε : ℝ) (ε_pos : ε > 0) :
  ∃ (n : ℝ), abs (n - wheelRevolutions 8 (2 * kmToFeet)) < ε ∧ 
             abs (n - 820.21 / Real.pi) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_for_2km_l863_86373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_quadratic_l863_86357

theorem min_value_quadratic (a c : ℝ) (h1 : a > c) 
  (h2 : ∀ x : ℝ, a * x^2 + 4 * x + c ≥ 0) 
  (h3 : ∃ x : ℝ, a * x^2 + 4 * x + c = 0) :
  (∀ y : ℝ, (4 * a^2 + c^2) / (2 * a - c) ≥ 8) ∧ 
  (∃ z : ℝ, (4 * a^2 + c^2) / (2 * a - c) = 8) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_quadratic_l863_86357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_seven_l863_86385

theorem three_digit_numbers_divisible_by_seven : 
  ∃ S : Finset ℕ, S = {n : ℕ | 900 ≤ n ∧ n < 1000 ∧ n % 7 = 0} ∧ S.card = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_seven_l863_86385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_result_expressions_l863_86397

theorem negative_result_expressions : 
  (-(abs (-3 : ℝ)) < 0) ∧ 
  (-(-3) ≥ 0) ∧ 
  (-(-3^2) ≥ 0) ∧ 
  ((-3)^2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_result_expressions_l863_86397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_constant_cube_inequality_l863_86335

theorem max_constant_cube_inequality {x y z : ℝ} (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  ∃ (c : ℝ), c = (Real.sqrt 6 + 3 * Real.sqrt 2) / 2 * Real.rpow 3 (1/4) ∧
  x^3 + y^3 + z^3 - 3*x*y*z ≥ c * abs ((x-y)*(y-z)*(z-x)) ∧
  ∀ (c' : ℝ), (∀ (a b d : ℝ), a ≥ 0 → b ≥ 0 → d ≥ 0 →
    a^3 + b^3 + d^3 - 3*a*b*d ≥ c' * abs ((a-b)*(b-d)*(d-a))) →
  c' ≤ c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_constant_cube_inequality_l863_86335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_line_l863_86380

/-- The function representing the curve -x^2 + 3ln(x) --/
noncomputable def f (x : ℝ) : ℝ := -x^2 + 3 * Real.log x

/-- The function representing the line x + 2 --/
def g (x : ℝ) : ℝ := x + 2

/-- The distance between two points (x₁, y₁) and (x₂, y₂) --/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem min_distance_curve_line :
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧
  ∀ (x₁ x₂ : ℝ), x₁ > 0 →
    distance x₁ (f x₁) x₂ (g x₂) ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_line_l863_86380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_properties_l863_86324

/-- An arithmetic sequence with common difference 1 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 1

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a 1 + (n - 1)) / 2

theorem arithmetic_seq_properties (a : ℕ → ℝ) (h_arith : arithmetic_seq a) :
  (((1 : ℝ) * a 3 = (a 1) ^ 2) → (a 1 = 2 ∨ a 1 = -1)) ∧
  ((S a 5 > a 1 * a 9) → (-5 < a 1 ∧ a 1 < 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_properties_l863_86324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_l863_86360

theorem cosine_value (α β : ℝ) : 
  0 < α → α < π/2 →
  0 < β → β < π/2 →
  Real.cos α = Real.sqrt 5 / 5 →
  Real.sin (α - β) = 3 * Real.sqrt 10 / 10 →
  Real.cos β = 7 * Real.sqrt 2 / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_l863_86360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_functions_l863_86358

-- Define the set of natural numbers starting from 1
def NatPos : Type := {n : Nat // n > 0}

-- Define the property that f(a) + f(b) divides 2(a + b - 1)
def divides_property (f : NatPos → NatPos) : Prop :=
  ∀ a b : NatPos, ∃ k : NatPos, (k.val * (f a).val + (f b).val) = 2 * (a.val + b.val - 1)

-- Theorem statement
theorem characterize_functions :
  ∀ f : NatPos → NatPos, divides_property f →
    (∀ x : NatPos, (f x).val = 1) ∨ (∀ x : NatPos, (f x).val = 2 * x.val - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_functions_l863_86358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_event_wheel_increase_l863_86305

/-- Represents the parameters of a cycling event with wheel change --/
structure CyclingEvent where
  original_distance : ℝ
  new_distance : ℝ
  original_diameter : ℝ

/-- Calculates the increase in wheel diameter for a given cycling event --/
noncomputable def wheel_diameter_increase (event : CyclingEvent) : ℝ :=
  let inches_per_mile := 63360
  let pi := Real.pi
  let original_circumference := pi * event.original_diameter
  let new_circumference := (event.original_distance / event.new_distance) * original_circumference
  (new_circumference / pi) - event.original_diameter

/-- Theorem stating the increase in wheel diameter for the given cycling event --/
theorem cycling_event_wheel_increase :
  let event : CyclingEvent := {
    original_distance := 120,
    new_distance := 118,
    original_diameter := 26
  }
  ∃ ε > 0, |wheel_diameter_increase event - 0.67| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_event_wheel_increase_l863_86305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_f_inequality_solution_l863_86364

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 2 / (x - 1)

-- Theorem for the minimum value and corresponding x
theorem f_minimum (x : ℝ) (h : x > 1) :
  f x ≥ 2 * Real.sqrt 2 + 1 ∧
  (f x = 2 * Real.sqrt 2 + 1 ↔ x = Real.sqrt 2 + 1) := by
  sorry

-- Theorem for the inequality solution
theorem f_inequality_solution (x : ℝ) :
  f x ≥ -2 ↔ (-1 ≤ x ∧ x ≤ 0) ∨ x > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_f_inequality_solution_l863_86364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piston_acceleration_theorem_l863_86363

/-- The acceleration of a piston in a cylindrical container filled with gas -/
noncomputable def piston_acceleration (Q M τ c R : ℝ) : ℝ :=
  Real.sqrt (2 * Q / (M * τ^2 * (1 + c/R)))

/-- Theorem stating the acceleration of the piston -/
theorem piston_acceleration_theorem
  (Q M τ c R : ℝ)
  (hQ : Q > 0)
  (hM : M > 0)
  (hτ : τ > 0)
  (hc : c > 0)
  (hR : R > 0) :
  ∃ (a : ℝ), a = piston_acceleration Q M τ c R ∧ a > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_piston_acceleration_theorem_l863_86363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_three_l863_86325

/-- The parabola y = x^2 -/
def parabola (x y : ℝ) : Prop := y = x^2

/-- The line x + y = 2 -/
def line (x y : ℝ) : Prop := x + y = 2

/-- The circle x^2 + y^2 = 1 -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The area of a triangle given three points -/
noncomputable def triangle_area (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  (1/2) * abs (x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂))

/-- The theorem stating that the area of the triangle is 3 -/
theorem triangle_area_is_three :
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
    parabola x₁ y₁ ∧ line x₁ y₁ ∧
    parabola x₂ y₂ ∧ line x₂ y₂ ∧
    unit_circle x₃ y₃ ∧
    triangle_area x₁ y₁ x₂ y₂ x₃ y₃ = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_three_l863_86325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_chart_characteristics_l863_86322

/-- Characteristics of line charts --/
theorem line_chart_characteristics : True :=
  sorry

/-- The answer to the line chart question --/
def line_chart_answer : String :=
  "Line charts can not only represent the amount, but also clearly show the situation of the increase or decrease in the amount."

#eval line_chart_answer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_chart_characteristics_l863_86322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_sqrt_two_l863_86349

theorem sin_minus_cos_sqrt_two (x : ℝ) :
  0 ≤ x → x < 2 * Real.pi → Real.sin x - Real.cos x = Real.sqrt 2 → x = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_sqrt_two_l863_86349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_effective_speed_against_current_l863_86348

/-- Calculate the effective speed of a man against a river current, considering headwind and obstacles --/
theorem effective_speed_against_current 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (headwind_speed : ℝ) 
  (obstacle_slowdown_percentage : ℝ) 
  (h1 : speed_with_current = 25)
  (h2 : current_speed = 4)
  (h3 : headwind_speed = 2)
  (h4 : obstacle_slowdown_percentage = 0.15) :
  let speed_in_still_water := speed_with_current - current_speed
  let speed_against_current_and_headwind := speed_in_still_water - current_speed - headwind_speed
  let reduction_due_to_obstacles := obstacle_slowdown_percentage * speed_against_current_and_headwind
  let effective_speed := speed_against_current_and_headwind - reduction_due_to_obstacles
  effective_speed = 12.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_effective_speed_against_current_l863_86348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_course_selection_theorem_l863_86315

def total_courses : ℕ := 8
def program_size : ℕ := 5
def math_courses : ℕ := 3
def humanities_courses : ℕ := 3

theorem course_selection_theorem :
  (Nat.choose (total_courses - 1) (program_size - 1)) -
  (Nat.choose (total_courses - 1 - math_courses) (program_size - 1)) -
  (Nat.choose (total_courses - 1 - humanities_courses) (program_size - 1)) = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_course_selection_theorem_l863_86315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l863_86365

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x + 3)^2 + (y - 1)^2 = 2

-- Define the line
def my_line (x y : ℝ) : Prop := y = x

-- Define the distance between two points
noncomputable def my_distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Statement of the theorem
theorem min_distance_circle_line :
  ∃ (min_dist : ℝ),
    (∀ (x1 y1 x2 y2 : ℝ),
      my_circle x1 y1 → my_line x2 y2 → my_distance x1 y1 x2 y2 ≥ min_dist) ∧
    (∃ (x1 y1 x2 y2 : ℝ),
      my_circle x1 y1 ∧ my_line x2 y2 ∧ my_distance x1 y1 x2 y2 = min_dist) ∧
    min_dist = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l863_86365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l863_86347

/-- A cubic function with two distinct extreme points -/
structure CubicFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  f : ℝ → ℝ
  hf : f = λ x => x^3 + 2*a*x^2 + 2*b*x + 3*c
  x₁ : ℝ
  x₂ : ℝ
  h_distinct : x₁ ≠ x₂
  h_extreme : (deriv f x₁ = 0) ∧ (deriv f x₂ = 0)
  h_fx₁ : f x₁ = x₁

/-- The theorem stating that the equation has exactly 3 different real roots -/
theorem cubic_equation_roots (cf : CubicFunction) :
  ∃ (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    (∀ x : ℝ, 3*(cf.f x)^2 + 4*cf.a*(cf.f x) + 2*cf.b = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l863_86347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_on_line_l_min_distance_curve_C_to_line_l_l863_86395

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 4 = 0

-- Define the curve C
def curve_C (x y : ℝ) : Prop := ∃ α : ℝ, x = Real.sqrt 3 * Real.cos α ∧ y = Real.sin α

-- Define point P
def point_P : ℝ × ℝ := (0, 4)

-- Statement 1: Point P lies on line l
theorem point_P_on_line_l : line_l point_P.1 point_P.2 := by sorry

-- Define the distance function from a point to line l
noncomputable def distance_to_line_l (x y : ℝ) : ℝ :=
  abs (x - y + 4) / Real.sqrt 2

-- Statement 2: Minimum distance from curve C to line l is √2
theorem min_distance_curve_C_to_line_l :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 ∧
  ∀ (x y : ℝ), curve_C x y → distance_to_line_l x y ≥ min_dist := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_on_line_l_min_distance_curve_C_to_line_l_l863_86395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l863_86366

/-- Curve C in the xy-plane -/
def C (x y : ℝ) : Prop := (1/5) * x^2 + y^2 = 1

/-- Line l in the xy-plane -/
def l (x y : ℝ) : Prop := y = x - 2

/-- Point P in the xy-plane -/
def P : ℝ × ℝ := (0, -2)

/-- Distance between two points in the plane -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ),
    C A.1 A.2 ∧ C B.1 B.2 ∧
    l A.1 A.2 ∧ l B.1 B.2 ∧
    distance P A + distance P B = (10 * Real.sqrt 2) / 3 := by
  sorry

#check intersection_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l863_86366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycles_count_l863_86378

/-- Represents the number of motorcycles in the parking lot -/
def motorcycles : ℕ := sorry

/-- Represents the number of cars in the parking lot -/
def cars : ℕ := sorry

/-- The total number of wheels in the parking lot -/
def total_wheels : ℕ := 84

/-- Axiom: The number of motorcycles is greater than the number of cars -/
axiom more_motorcycles : motorcycles > cars

/-- Axiom: The total number of wheels is the sum of motorcycle wheels and car wheels -/
axiom wheel_count : 2 * motorcycles + 4 * cars = total_wheels

/-- Axiom: After removing 3 vehicles, the remaining wheels are 3 times the remaining vehicles -/
axiom after_removal : 2 * (motorcycles - 3) + 4 * cars = 3 * (motorcycles + cars - 3)

/-- Theorem: The number of motorcycles originally parked is 16 -/
theorem motorcycles_count : motorcycles = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycles_count_l863_86378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_amount_for_bubble_mix_l863_86338

-- Define the constants from the problem
def soap_per_cup : ℚ := 3
def container_capacity : ℚ := 40
def ounces_per_cup : ℚ := 8

-- Define the theorem
theorem soap_amount_for_bubble_mix :
  (container_capacity / ounces_per_cup) * soap_per_cup = 15 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_amount_for_bubble_mix_l863_86338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_l863_86383

theorem sum_of_a_and_b (a b : ℝ) (h : Real.sqrt (a - 8 / b) = a * Real.sqrt (8 / b)) 
  (ha : a > 0) (hb : b > 0) : a + b = 73 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_l863_86383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_ratio_theorem_l863_86321

/-- The first five even composite numbers -/
def first_five_even_composite : List Nat := [4, 6, 8, 10, 12]

/-- The first five odd composite numbers -/
def first_five_odd_composite : List Nat := [9, 15, 21, 25, 27]

/-- The product of a list of natural numbers -/
def list_product (l : List Nat) : Nat := l.foldl Nat.mul 1

/-- The ratio of the products of even and odd composite numbers -/
def composite_ratio : Rat :=
  (list_product first_five_even_composite : Rat) /
  (list_product first_five_odd_composite : Rat)

theorem composite_ratio_theorem :
  composite_ratio = (2^10 : Rat) / ((3^6 * 5^2 * 7) : Rat) := by
  sorry

#eval composite_ratio.num
#eval composite_ratio.den

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_ratio_theorem_l863_86321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_S_bounds_l863_86301

/-- The function T as defined in the problem -/
noncomputable def T (t : ℝ) : ℝ := t / (1 + ⌊t⌋)

/-- The set S defined by the given conditions -/
def S (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1 - T t| ≤ T t ∧ |p.2| ≤ T t}

/-- The area of set S -/
noncomputable def area_S (t : ℝ) : ℝ := 4 * (T t)^2

/-- The theorem stating that the area of S is between 0 and 4 inclusive for all real t -/
theorem area_S_bounds : ∀ t : ℝ, 0 ≤ area_S t ∧ area_S t ≤ 4 := by
  intro t
  have h1 : 0 ≤ T t := by
    -- Proof that T(t) is non-negative
    sorry
  have h2 : T t < 1 := by
    -- Proof that T(t) is less than 1
    sorry
  -- Using these facts to prove the bounds on area_S
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_S_bounds_l863_86301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_recorded_as_negative_l863_86340

/-- Represents the recording of financial transactions. -/
def RecordedAmount := Int

/-- Represents the actual amount of money involved in a transaction. -/
def ActualAmount := Nat

/-- Records an income amount. -/
def recordIncome (amount : ActualAmount) : RecordedAmount :=
  Int.ofNat amount

/-- Records an expenditure amount. -/
def recordExpenditure (amount : ActualAmount) : RecordedAmount :=
  -Int.ofNat amount

/-- Theorem stating that if income is recorded positively, expenditure should be recorded negatively. -/
theorem expenditure_recorded_as_negative 
  (income_amount expenditure_amount : ActualAmount) 
  (h : recordIncome income_amount = Int.ofNat income_amount) :
  recordExpenditure expenditure_amount = -Int.ofNat expenditure_amount := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_recorded_as_negative_l863_86340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_parallel_lines_l863_86339

/-- Definition of the ellipse E -/
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Set of lines at distance d from the origin -/
def A_d (d : ℝ) : Set (Set (ℝ × ℝ)) := {l | ∃ k m : ℝ, l = {(x, y) | y = k * x + m ∧ |m| / Real.sqrt (k^2 + 1) = d}}

/-- Two lines are parallel -/
def parallel (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  ∃ k m1 m2 : ℝ, l1 = {(x, y) | y = k * x + m1} ∧ l2 = {(x, y) | y = k * x + m2}

/-- Main theorem -/
theorem ellipse_intersection_parallel_lines (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  ∃! d : ℝ, 0 < d ∧ d < b ∧
  ∀ l ∈ A_d d, ∃ l1 l2 : Set (ℝ × ℝ), l1 ∈ A_d d ∧ l2 ∈ A_d d ∧
    (∀ p : ℝ × ℝ, p ∈ l ∩ {p | ellipse a b p.1 p.2} → p ∈ l1 ∨ p ∈ l2) ∧
    parallel l1 l2 ∧
    d = a * b / Real.sqrt (a^2 + b^2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_parallel_lines_l863_86339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_riders_count_l863_86371

/-- The number of people who rode bicycles -/
def b : ℕ := sorry

/-- The number of people who rode tricycles -/
def t : ℕ := sorry

/-- The total number of people who rode in the race -/
def total_riders : ℕ := b + t

theorem race_riders_count :
  (b = t + 15) →  -- 15 more people rode bicycles than tricycles
  (3 * t = 2 * b + 15) →  -- 15 more tan wheels than blue wheels
  total_riders = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_riders_count_l863_86371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yura_reads_entire_book_l863_86303

theorem yura_reads_entire_book (x : ℝ) (hx : x > 0) : 
  let day1 := x / 2
  let day2 := (x - day1) / 3
  let day3 := (day1 + day2) / 2
  day1 + day2 + day3 = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yura_reads_entire_book_l863_86303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_leq_4_max_negative_x_when_f_geq_bound_l863_86333

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Part I
theorem solution_set_of_f_leq_4 :
  {x : ℝ | f x ≤ 4} = Set.Icc (-2) 2 :=
sorry

-- Part II
theorem max_negative_x_when_f_geq_bound (b : ℝ) (hb : b ≠ 0) :
  (∀ x, f x ≥ (|2*b + 1| + |1 - b|) / |b|) →
  ∃ x₀, x₀ = -1.5 ∧ ∀ x < 0, f x ≥ (|2*b + 1| + |1 - b|) / |b| → x ≤ x₀ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_leq_4_max_negative_x_when_f_geq_bound_l863_86333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_72_l863_86346

theorem probability_factor_of_72 : 
  let n : ℕ := 36
  let m : ℕ := 72
  (Finset.filter (λ x => m % x = 0) (Finset.range n.succ)).card / n = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_72_l863_86346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lloyd_worked_ten_and_half_hours_l863_86327

/-- Calculates the total hours worked given regular hours, regular rate, overtime multiplier, and total earnings -/
noncomputable def total_hours_worked (regular_hours : ℝ) (regular_rate : ℝ) (overtime_multiplier : ℝ) (total_earnings : ℝ) : ℝ :=
  let regular_pay := regular_hours * regular_rate
  let overtime_rate := regular_rate * overtime_multiplier
  let overtime_pay := total_earnings - regular_pay
  let overtime_hours := overtime_pay / overtime_rate
  regular_hours + overtime_hours

/-- Theorem stating that Lloyd worked 10.5 hours given the problem conditions -/
theorem lloyd_worked_ten_and_half_hours :
  let regular_hours : ℝ := 7.5
  let regular_rate : ℝ := 3.5
  let overtime_multiplier : ℝ := 1.5
  let total_earnings : ℝ := 42
  total_hours_worked regular_hours regular_rate overtime_multiplier total_earnings = 10.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lloyd_worked_ten_and_half_hours_l863_86327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_geometric_sequence_sum_ratio_l863_86331

noncomputable def geometric_sequence (a₁ : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => -1/2 * geometric_sequence a₁ n

theorem geometric_sequence_ratio (a₁ : ℝ) :
  ∀ n : ℕ, geometric_sequence a₁ (n + 1) = -1/2 * geometric_sequence a₁ n :=
by sorry

theorem geometric_sequence_sum_ratio :
  ∀ a₁ : ℝ, 
  (geometric_sequence a₁ 0 + geometric_sequence a₁ 2 + geometric_sequence a₁ 4) / 
  (geometric_sequence a₁ 1 + geometric_sequence a₁ 3 + geometric_sequence a₁ 5) = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_geometric_sequence_sum_ratio_l863_86331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_l863_86310

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f is not constantly zero if there exists an x such that f(x) ≠ 0 -/
def NotConstantlyZero (f : ℝ → ℝ) : Prop := ∃ x, f x ≠ 0

theorem problem (f : ℝ → ℝ) 
  (h1 : IsEven f)
  (h2 : NotConstantlyZero f)
  (h3 : ∀ x, x * f (x + 1) = (x + 1) * f x) :
  f (f (5/2)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_l863_86310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_equal_real_l863_86394

theorem quadratic_roots_equal_real (a c : ℝ) (h_disc_zero : 32 - 4*a*c = 0) :
  ∃ r : ℝ, (∀ t : ℝ, a*t^2 - 4*t*Real.sqrt 2 + c = 0 ↔ t = r) ∧ (a*r^2 - 4*r*Real.sqrt 2 + c = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_equal_real_l863_86394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_f_l863_86354

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 3)

theorem increasing_interval_of_f (ω : ℝ) (h1 : 0 < ω) (h2 : ω < 2 * Real.pi) 
  (h3 : ∀ x : ℝ, f ω x = f ω (-1/3 - x)) :
  ∀ k : ℤ, StrictMonoOn (f ω) (Set.Icc (-1/6 + 2 * (k : ℝ)) (5/6 + 2 * (k : ℝ))) :=
by
  sorry

#check increasing_interval_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_f_l863_86354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_offset_length_l863_86323

/-- Represents a quadrilateral with a diagonal and two offsets -/
structure Quadrilateral where
  diagonal : ℝ
  offset1 : ℝ
  offset2 : ℝ

/-- Calculates the area of a quadrilateral given its diagonal and offsets -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  (1/2) * q.diagonal * q.offset1 + (1/2) * q.diagonal * q.offset2

theorem second_offset_length (q : Quadrilateral) (h1 : q.diagonal = 28) 
    (h2 : q.offset1 = 9) (h3 : area q = 210) : q.offset2 = 6 := by
  sorry

#check second_offset_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_offset_length_l863_86323
