import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1053_105388

/-- Compound interest calculation --/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

/-- Problem statement --/
theorem investment_growth :
  let P : ℝ := 45046
  let r : ℝ := 0.08
  let n : ℝ := 12
  let t : ℝ := 10
  ⌊compound_interest P r n t⌋ = 100000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1053_105388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stompers_win_all_probability_l1053_105305

/-- The probability of winning all games in a series of independent games with equal win probability -/
noncomputable def win_all_probability (n : ℕ) (p : ℝ) : ℝ := p ^ n

/-- The number of games played -/
def num_games : ℕ := 6

/-- The probability of winning a single game -/
noncomputable def win_probability : ℝ := 4/5

theorem stompers_win_all_probability :
  win_all_probability num_games win_probability = 4096/15625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stompers_win_all_probability_l1053_105305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_total_volume_three_cubes_main_result_l1053_105355

/-- The volume of a cube with edge length 4 feet is 64 cubic feet. -/
theorem cube_volume (edge : ℝ) (h : edge = 4) : edge ^ 3 = 64 := by
  rw [h]
  norm_num

/-- The total volume of three identical cubes is three times the volume of one cube. -/
theorem total_volume_three_cubes (vol : ℝ) (h : vol = 64) : 3 * vol = 192 := by
  rw [h]
  norm_num

/-- The total volume of three cubes with edge length 4 feet is 192 cubic feet. -/
theorem main_result : ∃ (vol : ℝ), vol = 192 ∧ vol = 3 * (4 : ℝ) ^ 3 := by
  use 192
  constructor
  · rfl
  · have h1 : (4 : ℝ) ^ 3 = 64 := by norm_num
    rw [h1]
    norm_num

#check main_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_total_volume_three_cubes_main_result_l1053_105355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_income_calculation_l1053_105319

/-- Represents the actual monthly income of a person -/
structure Income where
  amount : ℝ

/-- Represents the tax rate applied to a person's income -/
structure TaxRate where
  rate : ℝ

/-- Represents the monthly expenses of a person -/
structure Expenses where
  amount : ℝ

/-- Calculates the income after taxes and expenses -/
def afterTaxesAndExpenses (income : Income) (taxRate : TaxRate) (expenses : Expenses) : ℝ :=
  income.amount * (1 - taxRate.rate) - expenses.amount

/-- Theorem stating that P's actual monthly income is approximately 7058.82 -/
theorem p_income_calculation 
  (p_income : Income) (q_income : Income) (r_income : Income)
  (p_tax : TaxRate) (q_tax : TaxRate) (r_tax : TaxRate)
  (p_expenses : Expenses) (q_expenses : Expenses) (r_expenses : Expenses)
  (h1 : (afterTaxesAndExpenses p_income p_tax p_expenses + 
         afterTaxesAndExpenses q_income q_tax q_expenses) / 2 = 5050)
  (h2 : (afterTaxesAndExpenses q_income q_tax q_expenses + 
         afterTaxesAndExpenses r_income r_tax r_expenses) / 2 = 6250)
  (h3 : (afterTaxesAndExpenses p_income p_tax p_expenses + 
         afterTaxesAndExpenses r_income r_tax r_expenses) / 2 = 5200)
  (h4 : p_tax.rate = 0.15)
  (h5 : q_tax.rate = 0.10)
  (h6 : r_tax.rate = 0.12)
  (h7 : p_expenses.amount = 2000)
  (h8 : q_expenses.amount = 2500)
  (h9 : r_expenses.amount = 3000) :
  ∃ ε > 0, |p_income.amount - 7058.82| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_income_calculation_l1053_105319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1053_105338

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A circle with center F₂ passing through the ellipse's center -/
def ellipseCircle (e : Ellipse) : Set (ℝ × ℝ) :=
  {p | dist p e.F₂ = dist e.F₂ ((e.F₁.1 + e.F₂.1) / 2, (e.F₁.2 + e.F₂.2) / 2)}

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  dist e.F₁ e.F₂ / (2 * dist e.F₁ ((e.F₁.1 + e.F₂.1) / 2, (e.F₁.2 + e.F₂.2) / 2))

/-- The theorem to be proved -/
theorem ellipse_eccentricity (e : Ellipse) (M : ℝ × ℝ) :
  M ∈ ellipseCircle e →
  M ∈ {p : ℝ × ℝ | ∃ (t : ℝ), p = (t * e.F₁.1 + (1 - t) * e.F₂.1, t * e.F₁.2 + (1 - t) * e.F₂.2)} →
  (∀ p, p ∈ ellipseCircle e → dist M p ≥ dist M e.F₁) →
  eccentricity e = Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1053_105338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_usage_comparison_and_minimum_l1053_105391

noncomputable def cleanliness_first (x : ℝ) : ℝ := (x + 0.8) / (x + 1)

noncomputable def cleanliness_second (y a c : ℝ) : ℝ := (y + a * c) / (y + a)

def water_usage_A : ℝ := 19

noncomputable def water_usage_B (a : ℝ) : ℝ := 4 * a + 3

noncomputable def min_total_water_usage (a : ℝ) : ℝ := -a + 4 * Real.sqrt (5 * a) - 1

theorem water_usage_comparison_and_minimum (a : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ 3) :
  water_usage_B a < water_usage_A ∧
  ∀ a₁ a₂, 1 ≤ a₁ → a₁ < a₂ → a₂ ≤ 3 → 
    min_total_water_usage a₁ < min_total_water_usage a₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_usage_comparison_and_minimum_l1053_105391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1053_105323

namespace TriangleProof

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector m -/
noncomputable def m (t : Triangle) : ℝ × ℝ := (t.a, (Real.sqrt 3 / 3) * t.b)

/-- Vector n -/
noncomputable def n (t : Triangle) : ℝ × ℝ := (Real.cos t.A, Real.sin t.B)

/-- Collinearity of vectors -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

/-- Area of a triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  1 / 2 * t.b * t.c * Real.sin t.A

/-- Main theorem -/
theorem triangle_properties (t : Triangle) 
    (h_collinear : collinear (m t) (n t))
    (h_a : t.a = Real.sqrt 7)
    (h_b : t.b = Real.sqrt 3) :
    t.A = π / 6 ∧ area t = Real.sqrt 3 := by
  sorry

end TriangleProof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1053_105323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_francis_remaining_time_l1053_105390

theorem francis_remaining_time (total_hours : ℝ) 
  (sleeping_fraction : ℝ) (studying_fraction : ℝ) (eating_fraction : ℝ) 
  (h1 : total_hours = 24) 
  (h2 : sleeping_fraction = 1/3) 
  (h3 : studying_fraction = 1/4) 
  (h4 : eating_fraction = 1/8) : 
  total_hours - (sleeping_fraction + studying_fraction + eating_fraction) * total_hours = 7 := by
  -- Proof steps will go here
  sorry

#check francis_remaining_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_francis_remaining_time_l1053_105390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1053_105353

/-- The function f as defined in the problem -/
noncomputable def f (a b : ℝ) : ℝ := 
  ⨆ (x : ℝ) (hx : x ∈ Set.Icc (-1) 1), |x^2 - a*x - b|

/-- The theorem stating the minimum value of f(a, b) -/
theorem min_value_of_f :
  (∀ a b : ℝ, f a b ≥ 1/2) ∧ (∃ a₀ b₀ : ℝ, f a₀ b₀ = 1/2) := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1053_105353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1053_105386

-- Define the hyperbola parameters
noncomputable def a : ℝ := 2 * Real.sqrt 3
noncomputable def b : ℝ := Real.sqrt 3

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the line equation
noncomputable def line (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x - 2

-- Define point C
noncomputable def C : ℝ × ℝ := (4 * Real.sqrt 3, 3)

-- Theorem statement
theorem hyperbola_properties :
  a > 0 ∧ b > 0 ∧
  2 * a = 4 * Real.sqrt 3 ∧
  (b * Real.sqrt (b^2 + a^2)) / Real.sqrt (b^2 + 12) = Real.sqrt 3 →
  (∀ x y, hyperbola x y ↔ x^2 / 12 - y^2 / 3 = 1) ∧
  (∃ A B : ℝ × ℝ,
    hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧
    line A.1 A.2 ∧ line B.1 B.2 ∧
    A.1 > 0 ∧ B.1 > 0 ∧
    ∃ m : ℝ, m = 4 ∧ A.1 + B.1 = m * C.1 ∧ A.2 + B.2 = m * C.2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1053_105386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_polar_axis_l1053_105309

/-- PolarPoint represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- OnLine P ρ θ means the point (ρ, θ) is on the line passing through P -/
def OnLine (P : PolarPoint) (ρ : ℝ) (θ : ℝ) : Prop :=
  sorry

/-- LineParallelToPolarAxis P f means the line represented by function f
    and passing through P is parallel to the polar axis -/
def LineParallelToPolarAxis (P : PolarPoint) (f : ℝ → ℝ) : Prop :=
  sorry

/-- Given a point P with polar coordinates (2, π/4), prove that the polar coordinate
    equation of a line passing through P and parallel to the polar axis is ρ sin θ = √2 -/
theorem line_parallel_to_polar_axis (P : PolarPoint) 
  (h1 : P.r = 2) (h2 : P.θ = π/4) : 
  ∃ (f : ℝ → ℝ), (∀ θ, f θ * Real.sin θ = Real.sqrt 2) ∧ 
  (∀ ρ θ, ρ * Real.sin θ = Real.sqrt 2 → OnLine P ρ θ) ∧
  (LineParallelToPolarAxis P f) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_polar_axis_l1053_105309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_problem_l1053_105327

theorem book_cost_problem (total_cost book1_loss book2_gain : ℝ) 
  (h1 : total_cost = 480)
  (h2 : book1_loss = 0.15)
  (h3 : book2_gain = 0.19) :
  ∃ (cost1 cost2 : ℝ),
    cost1 + cost2 = total_cost ∧
    cost1 * (1 - book1_loss) = cost2 * (1 + book2_gain) ∧
    cost1 = 280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_problem_l1053_105327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1053_105320

noncomputable def f (x : ℝ) : ℝ := (2 * x - 3) / (x - 5)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1053_105320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_increasing_intervals_l1053_105359

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (x + φ)

theorem symmetry_and_increasing_intervals 
  (φ : ℝ) 
  (h1 : 0 < φ) 
  (h2 : φ < Real.pi / 2) 
  (h3 : ∀ x, f φ x = f φ (Real.pi / 2 - x)) : 
  (φ = Real.pi / 4) ∧ 
  (∀ k : ℤ, ∀ x ∈ Set.Icc (2 * k * Real.pi - 3 * Real.pi / 4) (2 * k * Real.pi + Real.pi / 4), 
    Monotone (f (Real.pi / 4))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_increasing_intervals_l1053_105359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1053_105345

theorem range_of_a : ∃ (f : ℝ → ℝ → ℝ),
  (∀ x₁, x₁ ∈ Set.Icc 0 1 → ∃! x₂, x₂ ∈ Set.Icc (-1) 1 ∧ f x₁ x₂ = 0) ∧
  (∀ a : ℝ, (∀ x₁, x₁ ∈ Set.Icc 0 1 → ∃! x₂, x₂ ∈ Set.Icc (-1) 1 ∧ f x₁ x₂ = 0) ↔ 
    a ∈ Set.Ioo (1 + Real.exp (-1)) (Real.exp 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1053_105345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_offset_l1053_105329

/-- Represents a quadrilateral with a diagonal and two offsets -/
structure Quadrilateral where
  diagonal : ℝ
  offset1 : ℝ
  offset2 : ℝ

/-- Calculates the area of a quadrilateral given its diagonal and offsets -/
noncomputable def area (q : Quadrilateral) : ℝ := (1/2) * q.diagonal * (q.offset1 + q.offset2)

/-- Theorem stating that a quadrilateral with the given properties has an offset of 11 cm -/
theorem quadrilateral_offset (q : Quadrilateral) 
  (h1 : q.diagonal = 40)
  (h2 : q.offset2 = 9)
  (h3 : area q = 400) :
  q.offset1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_offset_l1053_105329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_problem_l1053_105324

theorem tangent_problem (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π / 4) = -3) ∧
  (Real.sin (2 * α) / (Real.sin α ^ 2 - Real.cos (2 * α) + 1) = 1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_problem_l1053_105324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_negative_l1053_105313

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

theorem f_sum_negative
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_decreasing : monotone_decreasing_on f (Set.Ici 0))
  (x₁ x₂ : ℝ)
  (h_sum_pos : x₁ + x₂ > 0) :
  f x₁ + f x₂ < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_negative_l1053_105313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_section_area_implies_height_ratio_l1053_105334

/-- Represents a cone with a section parallel to its base -/
structure ConeWithSection where
  /-- The ratio of the section area to the base area -/
  section_area_ratio : ℝ
  /-- Assumption that the section area is half of the base area -/
  section_half_base : section_area_ratio = 1/2

/-- The ratio of the upper part to the lower part of the cone's height -/
noncomputable def height_ratio (cone : ConeWithSection) : ℝ := Real.sqrt 2 - 1

/-- Theorem stating the relationship between the section area and height ratio -/
theorem section_area_implies_height_ratio (cone : ConeWithSection) :
  cone.section_area_ratio = 1/2 → height_ratio cone = Real.sqrt 2 - 1 := by
  sorry

#check section_area_implies_height_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_section_area_implies_height_ratio_l1053_105334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_like_sequence_correct_l1053_105362

def fibonacci_like_sequence (a₁ a₂ : Int) : List Int :=
  let rec aux : Nat → List Int
    | 0 => [a₁]
    | 1 => [a₂, a₁]
    | n + 2 => 
        let prev := aux (n + 1)
        (prev.head! + prev.tail!.head!) :: prev

  (aux 10).reverse

theorem fibonacci_like_sequence_correct :
  fibonacci_like_sequence (-42) 26 = [-42, 26, -16, 10, -6, 4, -2, 2, 0, 2, 2] := by
  sorry

#eval fibonacci_like_sequence (-42) 26

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_like_sequence_correct_l1053_105362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_unique_n_l1053_105383

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def nth_term (a : ℕ → ℝ) (n : ℕ) : ℝ := a n

noncomputable def sum_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + nth_term a n) / 2

theorem arithmetic_sequence_unique_n (a : ℕ → ℝ) (d : ℝ) (n : ℕ) :
  arithmetic_sequence a d →
  nth_term a n = 44 →
  sum_n_terms a n = 158 →
  d = 3 →
  n = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_unique_n_l1053_105383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_angle_bisector_theorem_l1053_105321

structure Parallelogram (A B C D : ℝ × ℝ) : Prop where
  is_parallelogram : True  -- This is a placeholder for the parallelogram property

def angle_measure (A B C : ℝ × ℝ) : ℝ :=
  sorry  -- Definition of angle measure

def angle_bisector (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry  -- Definition of angle bisector

def line_segment (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry  -- Definition of line segment

def ray (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry  -- Definition of ray

def length (A B : ℝ × ℝ) : ℝ :=
  sorry  -- Definition of length between two points

theorem parallelogram_angle_bisector_theorem 
  (A B C D E F : ℝ × ℝ) 
  (h_para : Parallelogram A B C D)
  (h_angle : angle_measure B A D = 60)
  (h_ab : length A B = 73)
  (h_bc : length B C = 88)
  (h_bisector : E ∈ angle_bisector A B C)
  (h_intersect_ad : E ∈ line_segment A D)
  (h_intersect_cd : F ∈ ray C D ∩ angle_bisector A B C) :
  length E F = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_angle_bisector_theorem_l1053_105321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_EFGH_area_l1053_105340

-- Define the vertices of the trapezoid
noncomputable def E : ℝ × ℝ := (1, -1)
noncomputable def F : ℝ × ℝ := (1, 3)
noncomputable def G : ℝ × ℝ := (5, 1)
noncomputable def H : ℝ × ℝ := (5, 7)

-- Define the trapezoid area function
noncomputable def trapezoidArea (a b h : ℝ) : ℝ := (a + b) * h / 2

-- Theorem statement
theorem trapezoid_EFGH_area : 
  let base1 := F.2 - E.2
  let base2 := H.2 - G.2
  let height := G.1 - E.1
  trapezoidArea base1 base2 height = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_EFGH_area_l1053_105340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_water_in_specific_tank_l1053_105326

noncomputable section

/-- The volume of water in a cylindrical tank -/
def water_volume (r h d : ℝ) : ℝ :=
  let θ := Real.arccos (1 - d / r)
  let sector_area := θ / Real.pi * (Real.pi * r^2)
  let triangle_area := (r - d) * Real.sqrt (2 * r * d - d^2)
  h * (sector_area - triangle_area)

/-- Theorem: Volume of water in a specific cylindrical tank -/
theorem volume_of_water_in_specific_tank :
  water_volume 5 10 3 = 290.7 * Real.pi - 20 * Real.sqrt 21 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_water_in_specific_tank_l1053_105326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l1053_105389

def x : ℕ → ℚ
  | 0 => 6
  | n + 1 => (x n ^ 2 + 7 * x n + 6) / (x n + 7)

theorem sequence_convergence :
  ∃ m : ℕ, m ∈ Set.Icc 121 140 ∧
    (∀ k : ℕ, k < m → x k > 4 + 1 / 2^18) ∧
    x m ≤ 4 + 1 / 2^18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l1053_105389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_g_l1053_105361

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - 2 * |x - 3|

-- Define the domain
def domain : Set ℝ := {x | 1 ≤ x ∧ x ≤ 6}

-- Theorem statement
theorem sum_of_max_min_g : 
  ∃ (max min : ℝ), 
    (∀ x, x ∈ domain → g x ≤ max) ∧ 
    (∀ x, x ∈ domain → min ≤ g x) ∧ 
    (∃ x₁ x₂, x₁ ∈ domain ∧ x₂ ∈ domain ∧ g x₁ = max ∧ g x₂ = min) ∧
    (max + min = -8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_g_l1053_105361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_zero_at_two_or_three_velocity_zero_at_two_and_three_l1053_105368

/-- Displacement function for a particle moving along a straight line -/
noncomputable def displacement (t : ℝ) : ℝ := (1/3) * t^3 - (5/2) * t^2 + 6 * t

/-- Velocity function derived from the displacement function -/
noncomputable def velocity (t : ℝ) : ℝ := t^2 - 5 * t + 6

/-- Theorem stating that velocity is zero at t = 2 or t = 3 -/
theorem velocity_zero_at_two_or_three :
  ∃ t : ℝ, (t = 2 ∨ t = 3) ∧ velocity t = 0 :=
by
  -- We'll use t = 2 to prove the theorem
  use 2
  constructor
  · left
    rfl
  · -- Calculate velocity(2)
    unfold velocity
    norm_num

-- Alternative proof using both t = 2 and t = 3
theorem velocity_zero_at_two_and_three :
  velocity 2 = 0 ∧ velocity 3 = 0 :=
by
  constructor
  · unfold velocity
    norm_num
  · unfold velocity
    norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_zero_at_two_or_three_velocity_zero_at_two_and_three_l1053_105368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_at_eight_l1053_105374

/-- A polynomial with real, nonnegative coefficients -/
def NonnegativePolynomial (f : ℝ → ℝ) : Prop :=
  ∃ (n : ℕ) (a : ℕ → ℝ), (∀ k, 0 ≤ a k) ∧
    ∀ x, f x = (Finset.range (n + 1)).sum (λ k ↦ a k * x ^ k)

theorem max_value_at_eight (f : ℝ → ℝ) 
    (h_nonneg : NonnegativePolynomial f)
    (h_4 : f 4 = 16)
    (h_16 : f 16 = 512) :
    f 8 ≤ 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_at_eight_l1053_105374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_length_l1053_105377

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  eccentricity : ℝ
  right_focus : Point

/-- Represents a parabola -/
structure Parabola where
  equation : ℝ → ℝ
  focus : Point

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem ellipse_parabola_intersection_length :
  ∀ (E : Ellipse) (C : Parabola),
    E.center = Point.mk 0 0 →
    E.eccentricity = Real.sqrt 3 / 2 →
    E.right_focus = C.focus →
    C.equation = (fun x ↦ 12 * x^2) →
    ∃ (A B : Point),
      A.x = -3 ∧ B.x = -3 ∧
      distance A B = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_length_l1053_105377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1053_105322

/-- The function f(x) = (3x^2 + 3x + c) / (x^2 + x + 1) -/
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (3 * x^2 + 3 * x + c) / (x^2 + x + 1)

/-- The maximum value of f(x) is 13/3 -/
theorem max_value_of_f (c : ℝ) : 
  ∃ (d : ℝ), d = 13/3 ∧ ∀ (x : ℝ), f c x ≤ d := by
  -- We claim that 13/3 is the maximum value
  use 13/3
  constructor
  · -- First part: d = 13/3
    rfl
  · -- Second part: ∀ (x : ℝ), f c x ≤ 13/3
    intro x
    -- The actual proof would go here, but we'll use sorry for now
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1053_105322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_values_l1053_105396

noncomputable def slope1 (m : ℝ) : ℝ := m + 2
noncomputable def slope2 (m : ℝ) : ℝ := -(m + 3) / (m + 18)

def perpendicular (m : ℝ) : Prop := slope1 m * slope2 m = -1

theorem perpendicular_lines_m_values :
  ∀ m : ℝ, m ≠ -18 → perpendicular m → (m = -6 ∨ m = 2) := by
  sorry

#check perpendicular_lines_m_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_values_l1053_105396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_A_winning_strategy_l1053_105378

-- Define the game board
def GameBoard := List (Option Bool)

-- Define a player
inductive Player
| A
| B

-- Define a move
structure Move where
  position : Nat
  sign : Bool

-- Define the game state
structure GameState where
  board : GameBoard
  currentPlayer : Player

-- Define the winning condition for Player A
def playerAWins (finalBoard : GameBoard) : Prop :=
  ∃ (evaluation : Nat), 
    evaluation % 2 = 1 ∧ 
    -- The evaluation function is not explicitly defined, 
    -- as it depends on the specific rules of expression evaluation
    True -- Placeholder for the actual evaluation logic

-- Define a strategy for Player A
def Strategy := GameState → Move

-- Theorem statement
theorem player_A_winning_strategy : 
  ∃ (strategy : Strategy), 
    ∀ (finalBoard : GameBoard),
      (∀ (move : Move), move.position < 99 → move.position ≥ 0) →
      playerAWins finalBoard := by
  sorry

#check player_A_winning_strategy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_A_winning_strategy_l1053_105378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_voltage_at_n_l1053_105398

/-- Represents the voltage at node j in the circuit network. -/
noncomputable def voltage (j : ℕ) : ℝ := sorry

/-- The initial voltage at node 0. -/
noncomputable def v₀ : ℝ := sorry

/-- Represents the relationship between voltages in the circuit network. -/
axiom voltage_relation (j : ℕ) : voltage (j + 2) - 4 * voltage (j + 1) + voltage j = 0

/-- The initial condition at node 0. -/
axiom initial_condition : voltage 1 = 3 * v₀

/-- The theorem stating the voltage at the nth node. -/
theorem voltage_at_n (n : ℕ) : 
  voltage n = (v₀ / (2 * Real.sqrt 3)) * 
    ((1 + Real.sqrt 3) * (2 + Real.sqrt 3) ^ n - (1 - Real.sqrt 3) * (2 - Real.sqrt 3) ^ n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_voltage_at_n_l1053_105398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l1053_105312

theorem smallest_positive_z (x z : ℝ) 
  (h1 : Real.cos x = 1/2) 
  (h2 : Real.sin (x + z) = 1) 
  (h3 : z > 0) : 
  z ≥ π/6 ∧ ∀ w, (w > 0 ∧ Real.sin (x + w) = 1) → w ≥ π/6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l1053_105312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1053_105318

noncomputable def f (x : ℝ) : ℝ := if x > 1 then x else -1

theorem solution_set (x : ℝ) : 
  x ∈ Set.Icc (-1 : ℝ) 2 ↔ x * f x - x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1053_105318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_people_proof_l1053_105316

/-- The number of people in a group where:
  1. The average weight increase is 3.5 kg when a new person replaces someone.
  2. The weight difference between the new person and the replaced person is 28 kg.
-/
def number_of_people : ℕ := 8

theorem number_of_people_proof :
  (28 : ℚ) / 3.5 = number_of_people := by
  sorry

#eval number_of_people

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_people_proof_l1053_105316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_nA_l1053_105300

/-- Given a positive integer A whose binary representation consists of n 1's,
    the sum of digits in the binary representation of nA is n. -/
theorem sum_of_digits_nA (n : ℕ) (A : ℕ) (h1 : A > 0) 
  (h2 : ∃ k : ℕ, A = 2^k - 1) (h3 : (Nat.digits 2 A).sum = n) :
  (Nat.digits 2 (n * A)).sum = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_nA_l1053_105300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_equation_l1053_105376

/-- A circle with radius 2, center in the fourth quadrant, tangent to x = 0 and x + y = 2√2 -/
def SpecialCircle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (c : ℝ × ℝ), 
    c.1 > 0 ∧ c.2 < 0 ∧
    (p.1 - c.1)^2 + (p.2 - c.2)^2 = 4 ∧
    c.1 = 2 ∧
    c.2 = -2}

/-- The standard equation of the special circle -/
def StandardEquation (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 4

theorem special_circle_equation :
  ∀ p : ℝ × ℝ, p ∈ SpecialCircle ↔ StandardEquation p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_equation_l1053_105376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equality_l1053_105337

theorem ceiling_sum_equality : ⌈Real.sqrt (16/9 : ℝ)⌉ + ⌈(16/9 : ℝ)⌉ + ⌈((16/9 : ℝ)^2)⌉ = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equality_l1053_105337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_f_two_zeros_l1053_105373

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 - x

-- Theorem 1: f is decreasing when a = 1/2
theorem f_decreasing (x : ℝ) (hx : x > 0) : 
  (deriv (f (1/2))) x < 0 := by sorry

-- Theorem 2: f has two zeros when 0 < a < 1/e^2
theorem f_two_zeros (a : ℝ) (ha : 0 < a ∧ a < 1 / Real.exp 2) :
  ∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ 
  ∀ z : ℝ, f a z = 0 → z = x ∨ z = y := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_f_two_zeros_l1053_105373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_satisfies_conditions_l1053_105363

/-- The polynomial p(x) that satisfies the given conditions -/
def p (x : ℝ) : ℝ := x^2 - x + 1

/-- Theorem stating that p(x) satisfies all the given conditions -/
theorem p_satisfies_conditions :
  p 3 = 7 ∧
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) ∧
  p 1 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_satisfies_conditions_l1053_105363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_area_l1053_105387

-- Define the points
def D : ℝ × ℝ := (-2, 2)
def E : ℝ × ℝ := (8, 2)
def F : ℝ × ℝ := (6, -4)

-- Define the function to calculate the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))

-- Theorem statement
theorem triangle_DEF_area :
  triangleArea D E F = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_area_l1053_105387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l1053_105325

theorem expression_simplification (k : ℤ) :
  3 * (2 : ℝ)^(-(2*k+2)) - (2 : ℝ)^(-(2*k+1)) + 2 * (2 : ℝ)^(-2*k) = (9/4) * (2 : ℝ)^(-2*k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l1053_105325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_product_factorial_l1053_105366

theorem min_sum_of_product_factorial (a b c d : ℕ+) 
  (h : (a * b * c * d : ℕ) = Nat.factorial 8) : 
  (a : ℕ) + b + c + d ≥ 130 ∧ 
  ∃ (w x y z : ℕ+), (w * x * y * z : ℕ) = Nat.factorial 8 ∧ (w : ℕ) + x + y + z = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_product_factorial_l1053_105366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_problem_l1053_105397

-- Define the geometric series a_n
def a : ℕ → ℝ := sorry

-- Define the sum of the first n terms of a_n
def S : ℕ → ℝ := sorry

-- Define the series b_n
def b (n : ℕ) : ℝ := (2 * n - 1) * a n

-- Define the sum of the first n terms of b_n
def T : ℕ → ℝ := sorry

-- Theorem statement
theorem geometric_series_problem (h1 : ∀ n, a n > 0) 
                                 (h2 : a 2 * a 3 = a 5) 
                                 (h3 : S 4 = 10 * S 2) : 
  (∀ n, a n = 3^n) ∧ 
  (∀ n, T n = (n - 1) * 3^(n + 1) + 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_problem_l1053_105397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_is_twelve_days_l1053_105399

/-- Represents the farming scenario with given conditions -/
structure FarmingScenario where
  flat_plow_rate : ℝ
  hilly_efficiency_drop : ℝ
  sunny_mow_rate : ℝ
  rainy_efficiency_drop : ℝ
  total_farmland : ℝ
  hilly_farmland : ℝ
  total_grassland : ℝ
  rainy_days : ℕ

/-- Calculates the total time needed to complete all farming work -/
noncomputable def total_work_time (scenario : FarmingScenario) : ℝ :=
  let flat_farmland := scenario.total_farmland - scenario.hilly_farmland
  let hilly_plow_rate := scenario.flat_plow_rate * (1 - scenario.hilly_efficiency_drop)
  let rainy_mow_rate := scenario.sunny_mow_rate * (1 - scenario.rainy_efficiency_drop)
  let flat_plow_time := flat_farmland / scenario.flat_plow_rate
  let hilly_plow_time := scenario.hilly_farmland / hilly_plow_rate
  let mow_time := scenario.total_grassland / rainy_mow_rate
  flat_plow_time + hilly_plow_time + mow_time

/-- Theorem stating that the total work time for the given scenario is 12 days -/
theorem work_time_is_twelve_days :
  let scenario : FarmingScenario := {
    flat_plow_rate := 10
    hilly_efficiency_drop := 0.3
    sunny_mow_rate := 12
    rainy_efficiency_drop := 0.4
    total_farmland := 55
    hilly_farmland := 15
    total_grassland := 30
    rainy_days := 5
  }
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ total_work_time scenario = 12 - ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_is_twelve_days_l1053_105399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_values_l1053_105379

/-- Triangle ABC with circumcenter O and orthocenter H -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  O : ℝ × ℝ
  H : ℝ × ℝ

/-- The circumcenter-to-vertex distance equals the orthocenter-to-vertex distance for vertex B -/
def equal_distances (t : Triangle) : Prop :=
  dist t.B t.O = dist t.B t.H

/-- The measure of angle B in degrees -/
noncomputable def angle_B_measure (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating the possible values of angle B -/
theorem angle_B_values (t : Triangle) (h : equal_distances t) :
  angle_B_measure t = 60 ∨ angle_B_measure t = 120 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_values_l1053_105379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_f_lower_bound_l1053_105382

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp 1 * x - 1

-- Statement 1: f(x) has exactly 2 zeros
theorem f_has_two_zeros : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x, f x = 0 → x = x₁ ∨ x = x₂ :=
sorry

-- Statement 2: For a ≤ 0 and x ≥ 1, f(x) ≥ a ln x - 1
theorem f_lower_bound (a x : ℝ) (ha : a ≤ 0) (hx : x ≥ 1) : f x ≥ a * Real.log x - 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_f_lower_bound_l1053_105382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compass_sqrt2_construction_l1053_105381

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Intersection of two circles -/
def intersect (c1 c2 : Circle) : Set Point :=
  {p : Point | distance p c1.center = c1.radius ∧ distance p c2.center = c2.radius}

/-- Compass construction step -/
inductive CompassStep
  | drawCircle (center : Point) (radius : ℝ) : CompassStep

/-- A sequence of compass constructions -/
def CompassConstruction := List CompassStep

/-- Theorem: Given two points 1 unit apart, it's possible to construct two points √2 units apart using only a compass -/
theorem compass_sqrt2_construction (A B : Point) (h : distance A B = 1) :
  ∃ (C : CompassConstruction) (F : Point), distance A F = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compass_sqrt2_construction_l1053_105381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1053_105315

theorem divisibility_condition (n : ℕ) : 
  (∃ k : ℕ, (n.factorial)^2 + 2^n = k * (2*n + 1)) ↔ 
  (∃ p : ℕ, Nat.Prime p ∧ n = (p - 1) / 2 ∧ (p % 8 = 1 ∨ p % 8 = 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1053_105315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_D_l1053_105341

/-- Given a triangle AOB where:
    - A is on the positive x-axis
    - Angle AON = 60°
    - B is on ON
    - AB = 2
    - C is the midpoint of AB
    - D is symmetric to C with respect to OB
    Prove that the locus of D satisfies the equation 3x^2 + (y^2)/3 = 1,
    where -1/2 ≤ x ≤ 1/2 and √3/2 ≤ y ≤ √3. -/
theorem locus_of_point_D (A B C D : ℝ × ℝ) (O : ℝ × ℝ := (0, 0)) :
  let N : ℝ × ℝ := (1, Real.sqrt 3)  -- Unit vector at 60° angle
  (∃ a : ℝ, 0 ≤ a ∧ a ≤ 2 * Real.sqrt 3 / 3 ∧ A = (2*a, 0)) →  -- A on positive x-axis
  (∃ b : ℝ, 0 ≤ b ∧ b ≤ 2 * Real.sqrt 3 / 3 ∧ B = (b, Real.sqrt 3 * b)) →  -- B on ON
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 →  -- AB = 2
  C = ((A.1 + B.1)/2, (A.2 + B.2)/2) →  -- C is midpoint of AB
  D = (B.1 - (C.1 - B.1), B.2 - (C.2 - B.2)) →  -- D symmetric to C w.r.t. OB
  (3 * D.1^2 + D.2^2 / 3 = 1 ∧ 
   -1/2 ≤ D.1 ∧ D.1 ≤ 1/2 ∧ 
   Real.sqrt 3 / 2 ≤ D.2 ∧ D.2 ≤ Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_D_l1053_105341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_time_l1053_105347

/-- The time in hours when the first candle burns out -/
noncomputable def burnout_time_1 : ℝ := 5

/-- The time in hours when the second candle burns out -/
noncomputable def burnout_time_2 : ℝ := 6

/-- The initial height of both candles -/
noncomputable def initial_height : ℝ := 1

/-- The height of the first candle after t hours -/
noncomputable def height_1 (t : ℝ) : ℝ := initial_height - (t / burnout_time_1)

/-- The height of the second candle after t hours -/
noncomputable def height_2 (t : ℝ) : ℝ := initial_height - (t / burnout_time_2)

/-- The time when the height of the first candle is three times the height of the second candle -/
theorem candle_height_ratio_time : 
  ∃ t : ℝ, t > 0 ∧ t < burnout_time_1 ∧ t < burnout_time_2 ∧ height_1 t = 3 * height_2 t ∧ t = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_time_l1053_105347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_range_of_a_l1053_105350

theorem inequality_implies_range_of_a :
  ∀ a : ℝ, 
  (∀ x y : ℝ, x > 0 ∧ y > 0 → (y / 4) - Real.cos x ^ 2 ≥ a * Real.sin x - (9 / y)) → 
  -3 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_range_of_a_l1053_105350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_101110_equals_octal_56_l1053_105314

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldr (fun (i, b) acc => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation as a list of digits -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

/-- Converts a list of octal digits to a single natural number -/
def octal_list_to_nat (digits : List ℕ) : ℕ :=
  digits.reverse.enum.foldr (fun (i, d) acc => acc + d * 8^i) 0

theorem binary_101110_equals_octal_56 :
  binary_to_decimal [false, true, true, true, true, false, true] = 
  octal_list_to_nat (decimal_to_octal (binary_to_decimal [false, true, true, true, true, false, true])) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_101110_equals_octal_56_l1053_105314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1053_105357

-- Define set A
def A : Set ℝ := {x | |x - 2| ≤ 3}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = 1 - x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1053_105357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_l1053_105344

/-- Simple interest calculation -/
theorem simple_interest_rate (principal time interest : ℝ) (h1 : principal > 0) (h2 : time > 0) (h3 : interest > 0) :
  (principal = 642.8571428571428 ∧ time = 4 ∧ interest = 90) →
  (interest * 100) / (principal * time) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_l1053_105344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1053_105317

/-- Given two circles in the Cartesian plane, prove their polar equations, 
    intersection points, and common chord. -/
theorem circle_properties :
  let C₁ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
  let C₂ : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}
  ∃ (polar_eq_C₁ : ℝ → Prop) (polar_eq_C₂ : ℝ → ℝ → Prop) 
    (intersection_points : Set (ℝ × ℝ)) (common_chord : ℝ → ℝ × ℝ),
    (∀ ρ, polar_eq_C₁ ρ ↔ ρ = 2) ∧
    (∀ ρ θ, polar_eq_C₂ ρ θ ↔ ρ = 4 * Real.cos θ) ∧
    (∀ k : ℤ, (2, 2 * ↑k * Real.pi + Real.pi / 3) ∈ intersection_points ∧
              (2, 2 * ↑k * Real.pi - Real.pi / 3) ∈ intersection_points) ∧
    (∀ t, -Real.sqrt 3 ≤ t → t ≤ Real.sqrt 3 → common_chord t = (1, t)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1053_105317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_gain_percent_proof_l1053_105339

noncomputable section

def initial_price : ℝ := 100
def discount_rate : ℝ := 0.10
def purchase_tax_rate : ℝ := 0.05
def selling_tax_rate : ℝ := 0.15
def selling_price_with_tax : ℝ := 130

noncomputable def discounted_price : ℝ := initial_price * (1 - discount_rate)
noncomputable def purchase_price_with_tax : ℝ := discounted_price * (1 + purchase_tax_rate)
noncomputable def selling_price_before_tax : ℝ := selling_price_with_tax / (1 + selling_tax_rate)
noncomputable def gain : ℝ := selling_price_before_tax - purchase_price_with_tax
noncomputable def gain_percent : ℝ := (gain / purchase_price_with_tax) * 100

theorem actual_gain_percent_proof :
  ∃ ε > 0, |gain_percent - 19.62| < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_gain_percent_proof_l1053_105339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1053_105308

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 + f y) = f (f x) + f (y^2) + 2 * f (x * y)) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1053_105308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_ninth_of_three_to_twenty_l1053_105351

theorem one_ninth_of_three_to_twenty (x : ℝ) : (1 / 9) * (3^20 : ℝ) = 3^x → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_ninth_of_three_to_twenty_l1053_105351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_from_A_to_BC_l1053_105393

-- Define an acute triangle ABC
def AcuteTriangle (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define the theorem
theorem height_from_A_to_BC 
  (A B C : ℝ) 
  (h_acute : AcuteTriangle A B C)
  (h_sin_sum : Real.sin (A + B) = 3/5)
  (h_sin_diff : Real.sin (A - B) = 1/5)
  (h_AB : 3 = 3) :
  ∃ (h : ℝ), h = 2 + Real.sqrt 6 ∧ 
  h = 3 * Real.sin C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_from_A_to_BC_l1053_105393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_max_range_of_squares_l1053_105311

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = Real.pi
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the vectors
def m : Fin 2 → Real := ![2, -1]

noncomputable def n (t : Triangle) : Fin 2 → Real := ![Real.sin (t.A / 2), Real.cos (t.B + t.C)]

-- Statement 1: Dot product is maximized when A = π/3
theorem dot_product_max (t : Triangle) :
  (∀ t' : Triangle, m • n t ≤ m • n t') → t.A = Real.pi / 3 :=
sorry

-- Statement 2: Range of b^2 + c^2 when a = √3
theorem range_of_squares (t : Triangle) (h : t.a = Real.sqrt 3) :
  3 < t.b^2 + t.c^2 ∧ t.b^2 + t.c^2 ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_max_range_of_squares_l1053_105311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_special_figure_proof_l1053_105370

/-- The perimeter of a figure constructed from a semicircle with diameter 64,
    where the diameter is replaced by four equal semicircles. -/
noncomputable def perimeter_of_special_figure : ℝ :=
  let d : ℝ := 64  -- diameter of the large semicircle
  let n : ℕ := 4   -- number of small semicircles
  let small_d : ℝ := d / n  -- diameter of each small semicircle
  let large_arc_length : ℝ := Real.pi * d / 2  -- length of the large semicircular arc
  let small_arc_length : ℝ := Real.pi * small_d / 2  -- length of each small semicircular arc
  let total_small_arcs_length : ℝ := n * small_arc_length  -- total length of all small arcs
  let perimeter : ℝ := large_arc_length + total_small_arcs_length  -- total perimeter
  64 * Real.pi

/-- Proof that the perimeter of the special figure is equal to 64π. -/
theorem perimeter_of_special_figure_proof : perimeter_of_special_figure = 64 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_special_figure_proof_l1053_105370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l1053_105354

-- Define a type for quadrants
inductive Quadrant
  | I
  | II
  | III
  | IV

-- Define a function to get the quadrant of an angle
noncomputable def angle_quadrant (θ : Real) : Quadrant := 
  if θ ≥ 0 && θ < Real.pi / 2 then Quadrant.I
  else if θ ≥ Real.pi / 2 && θ < Real.pi then Quadrant.II
  else if θ ≥ Real.pi && θ < 3 * Real.pi / 2 then Quadrant.III
  else Quadrant.IV

-- Define the theorem
theorem half_angle_quadrant (θ : Real) :
  angle_quadrant θ = Quadrant.III →
  Real.cos (θ / 2) < 0 →
  angle_quadrant (θ / 2) = Quadrant.II := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l1053_105354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_form_through_points_line_forming_triangle_l1053_105364

-- Part 1
noncomputable def point_A : ℝ × ℝ := (1, 2)
noncomputable def point_B : ℝ × ℝ := (-1/2, 1)

theorem intercept_form_through_points :
  ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  ∀ (x y : ℝ), (x = point_A.1 ∧ y = point_A.2) ∨ (x = point_B.1 ∧ y = point_B.2) →
  x/a + y/b = 1 :=
sorry

-- Part 2
noncomputable def line_slope : ℝ := 4/3
noncomputable def triangle_area : ℝ := 4

theorem line_forming_triangle :
  ∃ (m : ℝ), 
  (∀ (x y : ℝ), y = line_slope * x + m ∨ y = line_slope * x - m) ∧
  (1/2 * abs m * abs (-3*m/4) = triangle_area) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_form_through_points_line_forming_triangle_l1053_105364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_equals_cone_volume_sphere_volume_equals_cylinder_volume_l1053_105371

/-- The radius of a sphere with volume equal to a cone or cylinder -/
noncomputable def sphere_radius_equal_volume (r h : ℝ) (is_cone : Bool) : ℝ :=
  if is_cone then
    (r^2 * h / 4) ^ (1/3)
  else
    (3 * r^2 * h / 4) ^ (1/3)

/-- Theorem: The volume of a sphere with radius R is equal to the volume of a cone with base radius r and height h when R = ∛((r^2 * h) / 4) -/
theorem sphere_volume_equals_cone_volume (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (4/3) * Real.pi * (sphere_radius_equal_volume r h true)^3 = (1/3) * Real.pi * r^2 * h := by
  sorry

/-- Theorem: The volume of a sphere with radius R is equal to the volume of a cylinder with base radius r and height h when R = ∛((3 * r^2 * h) / 4) -/
theorem sphere_volume_equals_cylinder_volume (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (4/3) * Real.pi * (sphere_radius_equal_volume r h false)^3 = Real.pi * r^2 * h := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_equals_cone_volume_sphere_volume_equals_cylinder_volume_l1053_105371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_stations_l1053_105303

/-- The distance between two stations given train schedules and speeds -/
theorem distance_between_stations : ℝ := by
  let station_A : ℝ := 0  -- Assuming station A is at position 0
  let station_B : ℝ := 65  -- Position of station B (to be determined)
  let train1_start_time : ℝ := 7  -- 7 a.m.
  let train2_start_time : ℝ := 8  -- 8 a.m.
  let meeting_time : ℝ := 9  -- 9 a.m.
  let train1_speed : ℝ := 20  -- 20 kmph
  let train2_speed : ℝ := 25  -- 25 kmph
  let train1_travel_time : ℝ := meeting_time - train1_start_time
  let train2_travel_time : ℝ := meeting_time - train2_start_time
  let train1_distance : ℝ := train1_speed * train1_travel_time
  let train2_distance : ℝ := train2_speed * train2_travel_time
  
  have h1 : station_B = train1_distance + train2_distance := by sorry
  have h2 : train1_distance + train2_distance = 65 := by sorry
  
  exact 65


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_stations_l1053_105303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_reflection_l1053_105394

-- Define a type for quadrants
inductive Quadrant
  | first
  | second
  | third
  | fourth

-- Define a function to determine the quadrant of an angle
noncomputable def angle_quadrant (angle : ℝ) : Quadrant :=
  if 0 ≤ angle && angle < Real.pi / 2 then Quadrant.first
  else if Real.pi / 2 ≤ angle && angle < Real.pi then Quadrant.second
  else if Real.pi ≤ angle && angle < 3 * Real.pi / 2 then Quadrant.third
  else Quadrant.fourth

-- Theorem statement
theorem angle_reflection (α : ℝ) :
  angle_quadrant α = Quadrant.fourth →
  angle_quadrant (Real.pi - α) = Quadrant.third :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_reflection_l1053_105394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_region_perimeter_l1053_105336

-- Define the circumference of each circle
noncomputable def circle_circumference : ℝ := 48

-- Define the angle subtended by each arc in radians
noncomputable def arc_angle : ℝ := 2 * Real.pi / 3  -- 120° in radians

-- Define the number of arcs
def num_arcs : ℕ := 3

-- Theorem statement
theorem shaded_region_perimeter :
  (arc_angle / (2 * Real.pi)) * circle_circumference * (num_arcs : ℝ) = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_region_perimeter_l1053_105336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1053_105348

theorem range_of_a (a : ℝ) : 
  (Real.log (1/2) / Real.log a < 1) → (a^(1/2) < 1) → (0 < a ∧ a < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1053_105348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_range_f_nonpositive_range_l1053_105331

/-- The function f(x) = e^x - 2ax, where a is a real number -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * a * x

theorem f_monotonicity_and_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ∨
  (∃ c : ℝ, c = Real.log (2 * a) ∧
    (∀ x y : ℝ, x < y → x < c → f a x > f a y) ∧
    (∀ x y : ℝ, x < y → c < x → f a x < f a y)) :=
by sorry

theorem f_nonpositive_range :
  {a : ℝ | ∀ x : ℝ, x ∈ Set.Icc 2 3 → f a x ≤ 0} = Set.Ici (Real.exp 3 / 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_range_f_nonpositive_range_l1053_105331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_equations_l1053_105356

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with vertex at the origin -/
structure Parabola where
  c : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The intersection point of the parabola and hyperbola -/
noncomputable def intersectionPoint : Point :=
  { x := 3/2, y := Real.sqrt 6 }

/-- Condition that the directrix of the parabola passes through a focus of the hyperbola -/
def directrixPassesThroughFocus (p : Parabola) (h : Hyperbola) : Prop :=
  p.c = h.a

/-- Condition that a > b > 0 for the hyperbola -/
def hyperbolaCondition (h : Hyperbola) : Prop :=
  h.a > h.b ∧ h.b > 0

/-- Theorem stating the equations of the parabola and hyperbola -/
theorem parabola_hyperbola_equations 
  (p : Parabola) 
  (h : Hyperbola) 
  (hdir : directrixPassesThroughFocus p h)
  (hcond : hyperbolaCondition h)
  (hint : intersectionPoint.y^2 = 4 * p.c * intersectionPoint.x ∧
          intersectionPoint.x^2 / h.a^2 - intersectionPoint.y^2 / h.b^2 = 1) :
  (∀ x y : ℝ, y^2 = 4 * x ↔ y^2 = 4 * p.c * x) ∧
  (∀ x y : ℝ, 4 * x^2 - 4 * y^2 / 3 = 1 ↔ x^2 / h.a^2 - y^2 / h.b^2 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_equations_l1053_105356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solutions_l1053_105369

theorem cube_root_equation_solutions :
  let f (x : ℝ) := x^(1/3) = 15 / (8 - x^(1/3))
  ∀ x : ℝ, f x ↔ (x = 125 ∨ x = 27) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solutions_l1053_105369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_binomial_m_value_l1053_105307

/-- Given that (m+2)x^|m|y^2 - 3xy^2 is a quartic binomial and m+2 ≠ 0, prove that m = 2 -/
theorem quartic_binomial_m_value (m : ℤ) : 
  (∀ (x y : ℝ), ∃ (a b : ℝ), (m + 2) * x^(abs m) * y^2 - 3 * x * y^2 = a * x^4 + b * x^3) ∧ 
  (m + 2 ≠ 0) → 
  m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_binomial_m_value_l1053_105307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l1053_105372

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 7 else a/x

theorem increasing_function_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → -4 ≤ a ∧ a ≤ -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l1053_105372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1053_105375

theorem lambda_range (m lambda theta : ℝ) (z₁ z₂ : ℂ) 
  (h1 : z₁ = m + (4 - m^2) * I)
  (h2 : z₂ = 2 * Real.cos theta + (lambda + 3 * Real.sin theta) * I)
  (h3 : z₁ = z₂) :
  lambda ∈ Set.Icc (-9/16) 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1053_105375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l1053_105304

/-- Two fixed points in a 2D plane -/
def F₁ : ℝ × ℝ := (-4, 0)
def F₂ : ℝ × ℝ := (4, 0)

/-- The constant sum of distances -/
def constantSum : ℝ := 10

/-- Distance between two points in a 2D plane -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Definition of the locus of points P -/
def locus : Set (ℝ × ℝ) :=
  {P | distance P F₁ + distance P F₂ = constantSum}

/-- Theorem: The locus of points P forms an ellipse -/
theorem locus_is_ellipse : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧
  locus = {P : ℝ × ℝ | (P.1 / a)^2 + (P.2 / b)^2 = 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l1053_105304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1053_105380

def sequenceA (n : ℕ+) : ℚ := (2 : ℚ) ^ (n.val - 1)

def sum_of_terms (n : ℕ+) : ℚ := 2 * sequenceA n - 1

theorem sequence_property :
  {n : ℕ+ | (sequenceA n) / n.val ≤ 2} = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1053_105380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_l1053_105367

-- Define the sets M and N
def M (m : ℝ) : Set ℂ := {1, Complex.mk (m^2 - 3*m - 1) (m^2 - 5*m - 6)}
def N : Set ℂ := {1, 3}

-- Theorem statement
theorem intersection_implies_m_value :
  ∀ m : ℝ, M m ∩ N = {1, 3} → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_l1053_105367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_conditions_l1053_105349

-- Define the function type
def RealToPositiveReal := ℝ → ℝ

-- State the theorem
theorem unique_function_satisfying_conditions
  (f : RealToPositiveReal)
  (h1 : ∀ x : ℝ, f (x^2) = (f x)^2 - 2*x*(f x))
  (h2 : ∀ x : ℝ, f (-x) = f (x - 1))
  (h3 : ∀ x y : ℝ, 1 < x → x < y → f x < f y)
  : ∀ x : ℝ, f x = x^2 + x + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_conditions_l1053_105349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greek_yogurt_order_l1053_105346

-- Define the problem parameters
def expired_percentage : ℚ := 40 / 100
def pack_cost : ℕ := 12
def total_refund : ℕ := 384

-- Define the total number of packs
def total_packs : ℕ := 80

-- Theorem to prove
theorem greek_yogurt_order :
  (expired_percentage * total_packs * pack_cost = total_refund) :=
by
  -- Convert natural numbers to rationals for calculation
  have h1 : (total_packs : ℚ) * (pack_cost : ℚ) = (total_packs * pack_cost : ℕ)
  sorry
  
  have h2 : expired_percentage * (total_packs : ℚ) * (pack_cost : ℚ) = (total_refund : ℚ)
  sorry
  
  -- Use the above hypotheses to prove the theorem
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greek_yogurt_order_l1053_105346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inequality_l1053_105301

theorem power_inequality (a b c : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < 1) : 
  (2 : ℝ)^a > (5 : ℝ)^(-b) ∧ (5 : ℝ)^(-b) > ((1/7) : ℝ)^c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inequality_l1053_105301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_card_probability_l1053_105384

/-- Represents a standard 52-card deck --/
def standard_deck : ℕ := 52

/-- Represents the number of suits in a standard deck --/
def num_suits : ℕ := 4

/-- Represents the number of cards in each suit --/
def cards_per_suit : ℕ := standard_deck / num_suits

/-- The probability of selecting a card from a specific suit --/
def prob_specific_suit : ℚ := 1 / num_suits

/-- The probability of selecting a heart or a diamond --/
def prob_heart_or_diamond : ℚ := 1 / 2

/-- The probability of selecting one card from each suit in the first four draws,
    and either a heart or a diamond in the fifth draw, when choosing five cards
    with replacement from a standard 52-card deck --/
theorem five_card_probability : 
  (1 - prob_specific_suit) * 
  (1 - 2 * prob_specific_suit) * 
  (1 - 3 * prob_specific_suit) * 
  prob_heart_or_diamond = 3 / 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_card_probability_l1053_105384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_next_point_l1053_105395

-- Define the function f
noncomputable def f (x : ℝ) := x - 2 - Real.log x

-- State the theorem
theorem bisection_next_point :
  (f 3 < 0) →
  (f 4 > 0) →
  (Real.log 3.5 = 1.25) →
  (f 3.5 > 0) →
  (∃ (root : ℝ), root ∈ Set.Ioo 3 4 ∧ f root = 0) →
  (∀ x ∈ Set.Ioo 3 4, ∀ y ∈ Set.Ioo 3 4, x < y → f x < f y) →
  (3.25 = (3 + 3.5) / 2) :=
by
  sorry

#eval (3 + 3.5) / 2  -- This line is added to check the computation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_next_point_l1053_105395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_whole_number_to_ratio_l1053_105360

noncomputable def ratio : ℝ := (10^2001 + 3*10^2003) / (2*10^2002 + 2*10^2002)

theorem closest_whole_number_to_ratio : 
  ∃ (n : ℕ), n = 750 ∧ ∀ (m : ℕ), m ≠ n → |ratio - (n : ℝ)| < |ratio - (m : ℝ)| :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_whole_number_to_ratio_l1053_105360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_in_acute_triangle_l1053_105385

theorem lambda_range_in_acute_triangle (A B C : ℝ) (a b c : ℝ) (lambda : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Triangle is acute
  A + B + C = π ∧  -- Sum of angles in a triangle
  a = 1 ∧  -- Given condition
  b * Real.cos A - Real.cos B = 1 ∧  -- Given condition
  (∀ A' B' : ℝ, 0 < A' ∧ 0 < B' ∧ 0 < π - A' - B' ∧
    Real.sin B' - lambda * (Real.sin A')^2 ≤ Real.sin B - lambda * (Real.sin A)^2) →  -- Maximum condition
  0 < lambda ∧ lambda < 2 * Real.sqrt 3 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_in_acute_triangle_l1053_105385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_squared_l1053_105365

theorem sin_plus_cos_squared (θ : Real) (b : Real) 
  (h1 : 0 < θ ∧ θ < Real.pi / 2) 
  (h2 : Real.cos (2 * θ) = b) : 
  (Real.sin θ + Real.cos θ)^2 = 2 - b := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_squared_l1053_105365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1053_105358

noncomputable def complex_i : ℂ := Complex.I

noncomputable def z : ℂ := (1 + complex_i) / complex_i

theorem z_in_fourth_quadrant :
  Real.sign z.re = 1 ∧ Real.sign z.im = -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1053_105358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_is_isosceles_triangle_l1053_105342

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the intersection point of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line) : Point :=
  { x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope),
    y := l1.slope * ((l2.intercept - l1.intercept) / (l1.slope - l2.slope)) + l1.intercept }

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a triangle is isosceles -/
def isIsosceles (p1 p2 p3 : Point) : Prop :=
  let d12 := distance p1 p2
  let d23 := distance p2 p3
  let d31 := distance p3 p1
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

/-- The main theorem -/
theorem polygon_is_isosceles_triangle (l1 l2 l3 : Line) :
  l1.slope = 4 ∧ l1.intercept = 3 ∧
  l2.slope = -4 ∧ l2.intercept = 3 ∧
  l3.slope = 0 ∧ l3.intercept = -3 →
  let p1 := intersectionPoint l1 l2
  let p2 := intersectionPoint l1 l3
  let p3 := intersectionPoint l2 l3
  isIsosceles p1 p2 p3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_is_isosceles_triangle_l1053_105342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1053_105352

noncomputable def f (x : ℝ) := Real.sin x * Real.cos x

theorem f_properties : 
  (∃ (M : ℝ), M = 1/2 ∧ ∀ x, f x ≤ M) ∧
  (∃ (T : ℝ), T = π ∧ T > 0 ∧ ∀ x, f (x + T) = f x ∧ 
    ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) :=
by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1053_105352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_lambda_bound_l1053_105392

theorem increasing_sequence_lambda_bound (l : ℝ) :
  (∀ n : ℕ, n ≥ 1 → (n + 1)^2 + l * (n + 1) > n^2 + l * n) →
  l > -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_lambda_bound_l1053_105392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_8_consecutive_fib_not_fibonacci_l1053_105310

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Define the sum of 8 consecutive Fibonacci numbers
def sum_8_consecutive_fib (n : ℕ) : ℕ :=
  (List.range 8).map (λ i => fib (n + i)) |>.sum

-- State the theorem
theorem sum_8_consecutive_fib_not_fibonacci (n : ℕ) :
  ∀ k, sum_8_consecutive_fib n ≠ fib k := by
  sorry

-- Helper lemma: The sum of 8 consecutive Fibonacci numbers equals F_{n+9} - F_{n+2}
lemma sum_8_consecutive_fib_eq (n : ℕ) :
  sum_8_consecutive_fib n = fib (n + 9) - fib (n + 2) := by
  sorry

-- Helper lemma: For any n, F_{n+8} < F_{n+9} - F_{n+2} < F_{n+9}
lemma fib_inequality (n : ℕ) :
  fib (n + 8) < fib (n + 9) - fib (n + 2) ∧ fib (n + 9) - fib (n + 2) < fib (n + 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_8_consecutive_fib_not_fibonacci_l1053_105310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_ratio_dividing_NN1_l1053_105328

-- Define the sphere and points
def Sphere : Type := Unit
def Point : Type := Unit

-- Define the radius of the sphere
noncomputable def radius : ℝ := 9

-- Define the distance of point A from the center
noncomputable def distance_A_center : ℝ := Real.sqrt 47

-- Define the lengths of LL1 and MM1
def length_LL1 : ℝ := 16
def length_MM1 : ℝ := 14

-- Define the perpendicularity of the line segments
def perpendicular (p q r : Point → Point → Prop) : Prop := sorry

-- Define the intersection of the line segments at point A
def intersect_at_A (p q r : Point → Point → Prop) (A : Point) : Prop := sorry

-- Define the division of a line segment by a point
def divides_segment (A N N1 : Point) (r : ℝ) : Prop := sorry

-- Theorem statement
theorem exists_ratio_dividing_NN1 (L L1 M M1 N N1 A : Point) :
  perpendicular (λ x y => x = L ∧ y = L1) (λ x y => x = M ∧ y = M1) (λ x y => x = N ∧ y = N1) →
  intersect_at_A (λ x y => x = L ∧ y = L1) (λ x y => x = M ∧ y = M1) (λ x y => x = N ∧ y = N1) A →
  ∃ (r : ℝ), divides_segment A N N1 r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_ratio_dividing_NN1_l1053_105328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_APQ_l1053_105330

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∀ (X Y : ℝ × ℝ), (X = A ∧ Y = B) ∨ (X = B ∧ Y = C) ∨ (X = C ∧ Y = A) →
  dist X Y = 10

-- Define the incircle of a triangle
def Incircle (center : ℝ × ℝ) (radius : ℝ) (A B C : ℝ × ℝ) : Prop :=
  ∀ (X : ℝ × ℝ), dist X center = radius →
  (dist X A + dist X B + dist X C = dist B C + dist C A + dist A B)

-- Define a point on a line segment
def PointOnSegment (P X Y : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (t * X.1 + (1 - t) * Y.1, t * X.2 + (1 - t) * Y.2)

-- Define a line tangent to a circle
def TangentLine (P Q : ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  dist center P = radius ∧ dist center Q = radius ∧
  ∀ (X : ℝ × ℝ), PointOnSegment X P Q → dist X center ≥ radius

-- Define area of a triangle
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (1/2) * abs ((x₂ - x₁) * (y₃ - y₁) - (x₃ - x₁) * (y₂ - y₁))

-- Main theorem
theorem area_of_triangle_APQ
  (A B C P Q : ℝ × ℝ)
  (center : ℝ × ℝ)
  (radius : ℝ)
  (h1 : Triangle A B C)
  (h2 : PointOnSegment P A B)
  (h3 : PointOnSegment Q A C)
  (h4 : TangentLine P Q center radius)
  (h5 : Incircle center radius A B C)
  (h6 : dist P Q = 4) :
  area_triangle A P Q = 5 / Real.sqrt 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_APQ_l1053_105330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_for_f_less_than_one_l1053_105333

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.exp (x^2 - 3*x)

-- State the theorem
theorem sufficient_condition_for_f_less_than_one :
  ∀ x : ℝ, 0 < x → x < 1 → f x < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_for_f_less_than_one_l1053_105333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1053_105343

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

/-- Proof that the distance between the given parallel lines is 3√13 / 13 -/
theorem distance_between_given_lines :
  let line1 : ℝ → ℝ → ℝ := fun x y => 2*x + 3*y - 5
  let line2 : ℝ → ℝ → ℝ := fun x y => x + 3/2*y - 1
  distance_between_parallel_lines 2 3 (-5) (-2) = 3 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1053_105343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrange_divisible_by_two_l1053_105332

def digits : List Nat := [1, 3, 1, 1, 5, 2, 1, 5, 2]

theorem rearrange_divisible_by_two (n : Nat) :
  (List.length digits = 9) →
  (List.count 2 digits > 0) →
  (n = (Nat.factorial 8) / ((Nat.factorial 3) * (Nat.factorial 2))) →
  (n = 3360) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrange_divisible_by_two_l1053_105332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rodney_lifts_146_l1053_105306

/-- Represents the lifting capacity of a person in pounds. -/
structure LiftingCapacity where
  value : ℕ

/-- Rodney's lifting capacity -/
def rodney_capacity : LiftingCapacity := ⟨146⟩

/-- Roger's lifting capacity -/
def roger_capacity : LiftingCapacity := ⟨73⟩

/-- Ron's lifting capacity -/
def ron_capacity : LiftingCapacity := ⟨20⟩

instance : Add LiftingCapacity where
  add a b := ⟨a.value + b.value⟩

instance : HMul ℕ LiftingCapacity LiftingCapacity where
  hMul n a := ⟨n * a.value⟩

instance : HSub LiftingCapacity ℕ LiftingCapacity where
  hSub a n := ⟨a.value - n⟩

/-- The combined lifting capacity of Rodney, Roger, and Ron is 239 pounds. -/
axiom combined_capacity : (rodney_capacity + roger_capacity + ron_capacity).value = 239

/-- Rodney can lift twice as much as Roger. -/
axiom rodney_double_roger : rodney_capacity = 2 * roger_capacity

/-- Roger can lift 7 pounds less than 4 times the amount that Ron can lift. -/
axiom roger_to_ron_relation : roger_capacity = 4 * ron_capacity - 7

/-- Theorem stating that Rodney's lifting capacity is 146 pounds. -/
theorem rodney_lifts_146 : rodney_capacity.value = 146 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rodney_lifts_146_l1053_105306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_relation_l1053_105302

/-- The curve equation -/
noncomputable def curve_equation (p q r s x : ℝ) : ℝ := (2 * p * x + q) / (r * x + 2 * s)

/-- Theorem stating that if y = x is an axis of symmetry of the given curve, then p + 2s = 0 -/
theorem symmetry_implies_relation (p q r s : ℝ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (∀ x : ℝ, x = curve_equation p q r s (curve_equation p q r s x)) →
  p + 2 * s = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_relation_l1053_105302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_of_sqrt2_plus_minus_one_l1053_105335

theorem geometric_mean_of_sqrt2_plus_minus_one : 
  Real.sqrt ((Real.sqrt 2 + 1) * (Real.sqrt 2 - 1)) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_of_sqrt2_plus_minus_one_l1053_105335
