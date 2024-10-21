import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_coefficient_l597_59701

/-- Given a curve y = x^3 passing through (2, 8) with tangent line 12x - ay - 16 = 0, prove a = 1 -/
theorem tangent_line_coefficient (a : ℝ) : 
  (∃ f : ℝ → ℝ, f = λ x ↦ x^3) ∧  -- The curve is y = x^3
  (2^3 = 8) ∧                    -- The curve passes through (2, 8)
  (∀ x y : ℝ, 12*x - a*y - 16 = 0 → y = 2^3) -- The tangent line equation at (2, 8)
  → a = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_coefficient_l597_59701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_constant_term_problem_l597_59787

/-- Given polynomials p, q, and r such that r(x) = p(x) * q(x),
    with p having constant term 5 and leading coefficient 2,
    and r having constant term -15, then q(0) = -3. -/
theorem polynomial_constant_term_problem
  (p q r : Polynomial ℝ)
  (h₁ : r = p * q)
  (h₂ : p.coeff 0 = 5)
  (h₃ : p.leadingCoeff = 2)
  (h₄ : p.degree = 2)
  (h₅ : r.coeff 0 = -15) :
  q.eval 0 = -3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_constant_term_problem_l597_59787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l597_59786

/-- An ellipse C with center at the origin, right focus at (1,0), and the symmetric point of (1,0) about the line y = (1/2)x lying on C, has the equation (5x²/9) + (5y²/4) = 1 -/
theorem ellipse_equation (C : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ C ↔ (5 * x^2 / 9 + 5 * y^2 / 4 = 1)) ↔
  (-- Center at origin
   (0, 0) ∈ C ∧
   -- Right focus at (1,0)
   (1, 0) ∈ C ∧
   -- Symmetric point of (1,0) about y = (1/2)x lies on C
   (∃ (x y : ℝ), (x, y) ∈ C ∧ 
    y / (x - 1) = -2 ∧ 
    y = (1/2) * ((1 + x) / 2))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l597_59786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_progression_sin_cos_range_l597_59784

noncomputable section

open Real

theorem triangle_geometric_progression_sin_cos_range (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  b^2 = a * c →
  a^2 + b^2 + c^2 = 2 * (a * b * (cos C) + b * c * (cos A) + c * a * (cos B)) →
  1 < sin B + cos B ∧ sin B + cos B ≤ sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_progression_sin_cos_range_l597_59784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l597_59771

def A : Set ℚ := {-2, -1, 0, 1, 2}
def B : Set ℚ := {x : ℚ | 0 ≤ x ∧ x < 5/2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l597_59771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_omega_for_trig_function_l597_59772

theorem exists_omega_for_trig_function : ∃ (ω a : ℝ), 
  ω > 0 ∧ a > 0 ∧ 
  (∀ x, Real.sin (ω * x) + a * Real.cos (ω * x) ≥ -2) ∧
  Real.sin (ω * (π / 6)) + a * Real.cos (ω * (π / 6)) = -2 ∧
  ω = 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_omega_for_trig_function_l597_59772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_sum_l597_59706

theorem simplify_fraction_sum (a b : ℕ) (h : a = 63 ∧ b = 126) :
  ∃ (n d : ℕ), (n : ℚ) / d = (a : ℚ) / b ∧ Nat.gcd n d = 1 ∧ n + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_sum_l597_59706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l597_59759

/-- The eccentricity of an ellipse with parametric equations x = a * cos(θ) and y = b * sin(θ) -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (min a b)^2 / (max a b)^2)

/-- Theorem: The eccentricity of the ellipse x = 3cos(θ), y = 4sin(θ) is √7/4 -/
theorem ellipse_eccentricity : eccentricity 3 4 = Real.sqrt 7 / 4 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l597_59759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_defective_chips_calculation_l597_59788

/-- The ratio of defective chips to total chips from combined shipments S1, S2, S3, and S4 -/
def defective_ratio : ℝ := sorry

/-- The total number of chips in the new shipment -/
def total_chips : ℕ := 60000

/-- The expected number of defective chips in a shipment of 60,000 chips -/
def expected_defective_chips : ℝ := defective_ratio * (total_chips : ℝ)

theorem expected_defective_chips_calculation :
  expected_defective_chips = defective_ratio * (total_chips : ℝ) :=
by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_defective_chips_calculation_l597_59788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_of_books_l597_59765

/-- Represents the cost and selling price of a book -/
structure Book where
  cost : ℝ
  sellPrice : ℝ

/-- The problem setup for the book sale -/
structure BookSaleProblem where
  book1 : Book
  book2 : Book
  book1_loss_percent : ℝ
  book2_gain_percent : ℝ
  same_sell_price : Prop := book1.sellPrice = book2.sellPrice
  book1_loss : Prop := book1.sellPrice = book1.cost * (1 - book1_loss_percent)
  book2_gain : Prop := book2.sellPrice = book2.cost * (1 + book2_gain_percent)
  book1_cost : Prop := book1.cost = 291.67

/-- The theorem stating the total cost of the two books -/
theorem total_cost_of_books (p : BookSaleProblem) 
  (h1 : p.book1_loss_percent = 0.15)
  (h2 : p.book2_gain_percent = 0.19) :
  p.book1.cost + p.book2.cost = 499.586 := by
  sorry

#eval "BookSaleProblem theorem defined successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_of_books_l597_59765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_in_circle_l597_59711

/-- A broken line on a plane -/
structure BrokenLine (n : ℕ) where
  points : Fin (n + 1) → ℝ × ℝ

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The angle between three points -/
noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

theorem broken_line_in_circle (n : ℕ) (line : BrokenLine n) : 
  (∀ i j : Fin n, distance (line.points i) (line.points (i.succ)) = 1) →
  (∀ i : Fin (n - 1), 
    π/3 ≤ angle (line.points i) (line.points i.succ) (line.points i.succ.succ) ∧
    angle (line.points i) (line.points i.succ) (line.points i.succ.succ) ≤ 2*π/3) →
  (∀ i : Fin (n + 1), distance (line.points 0) (line.points i) ≤ 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_in_circle_l597_59711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_value_l597_59744

-- Define the curves C₁ and C₂
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.cos t, Real.sin t)

noncomputable def C₂ (r : ℝ) (θ : ℝ) : ℝ × ℝ := (r * Real.cos θ, r * Real.sin θ)

-- Define the conditions
axiom r_positive : ∃ r : ℝ, r > 0

-- A and B are on C₁ and not at the origin
axiom A_on_C₁ : ∃ t₁ : ℝ, C₁ t₁ ≠ (0, 0)
axiom B_on_C₁ : ∃ t₂ : ℝ, C₁ t₂ ≠ (0, 0)

-- ∠AOB = 90°
axiom angle_AOB_90 : ∃ (t₁ t₂ : ℝ), t₂ - t₁ = Real.pi / 2

-- C₂ has exactly one point in common with line AB
axiom C₂_intersects_AB_once (r : ℝ) (t₁ t₂ : ℝ) : 
  ∃! p : ℝ × ℝ, (∃ θ : ℝ, C₂ r θ = p) ∧ (∃ s : ℝ, p = s • C₁ t₁ + (1 - s) • C₁ t₂)

-- The theorem to prove
theorem r_value : ∃ r : ℝ, r = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_value_l597_59744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l597_59760

/-- Predicate to check if a point is on the directrix of a parabola -/
def IsOnDirectrix (p : ℝ) (point : ℝ × ℝ) : Prop :=
  point.1 = -p

/-- For a parabola with equation y^2 = 8x, its directrix has the equation x = -2 -/
theorem parabola_directrix (x y : ℝ) :
  y^2 = 8*x → (∃ (k : ℝ), k = -2 ∧ (x = k ↔ IsOnDirectrix 2 (x, y))) :=
by
  intro h
  use -2
  constructor
  · rfl
  · constructor
    · intro hx
      rw [hx]
      rfl
    · intro h_on_directrix
      exact h_on_directrix


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l597_59760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_convergence_l597_59720

/-- Represents the position of the bug on a 2D plane -/
structure BugPosition where
  x : ℝ
  y : ℝ

/-- Represents a move of the bug -/
inductive Move
  | Right
  | Up
  | Left
  | Down

/-- The sequence of moves the bug makes -/
def moveSequence : ℕ → Move
  | 0 => Move.Right
  | 1 => Move.Up
  | 2 => Move.Left
  | 3 => Move.Down
  | n + 4 => moveSequence n

/-- The distance the bug moves in each step -/
noncomputable def moveDistance (n : ℕ) : ℝ := (1/2) ^ n

/-- The position of the bug after n moves -/
noncomputable def bugPosition : ℕ → BugPosition
  | 0 => ⟨0, 0⟩
  | n + 1 =>
    let prev := bugPosition n
    match moveSequence n with
    | Move.Right => ⟨prev.x + moveDistance n, prev.y⟩
    | Move.Up => ⟨prev.x, prev.y + moveDistance n⟩
    | Move.Left => ⟨prev.x - moveDistance n, prev.y⟩
    | Move.Down => ⟨prev.x, prev.y - moveDistance n⟩

/-- The theorem to be proved -/
theorem bug_convergence :
  ∃ (L : BugPosition), L = ⟨4/5, 2/5⟩ ∧ ∀ ε > 0, ∃ N, ∀ n ≥ N,
    Real.sqrt ((bugPosition n).x - L.x)^2 + ((bugPosition n).y - L.y)^2 < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_convergence_l597_59720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_with_tangent_relation_l597_59739

theorem sum_of_angles_with_tangent_relation (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.tan α + Real.tan β = Real.sqrt 3 - Real.sqrt 3 * Real.tan α * Real.tan β →
  α + β = π/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_with_tangent_relation_l597_59739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l597_59752

-- Define the quadrilateral region
def QuadrilateralRegion (x y : ℝ) : Prop :=
  x + 2*y ≤ 4 ∧ 3*x + 2*y ≥ 3 ∧ x ≥ 0 ∧ y ≥ 0

-- Define the vertices of the quadrilateral
def Vertices : Set (ℝ × ℝ) :=
  {(0, 2), (0, 1.5), (4, 0), (1, 0)}

-- Define the function to calculate the length of a side
noncomputable def SideLength (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem: The length of the longest side is √5
theorem longest_side_length :
  ∃ (p q : ℝ × ℝ), p ∈ Vertices ∧ q ∈ Vertices ∧
  SideLength p q = Real.sqrt 5 ∧
  ∀ (r s : ℝ × ℝ), r ∈ Vertices → s ∈ Vertices →
  SideLength r s ≤ Real.sqrt 5 := by
  sorry

#check longest_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l597_59752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_irrational_pairs_l597_59746

theorem infinitely_many_irrational_pairs : 
  ∃ f : ℕ → ℝ × ℝ, 
    (∀ n : ℕ, 
      let (x, y) := f n
      Irrational x ∧ Irrational y ∧
      x + y = x * y ∧
      ∃ k : ℕ, x + y = k) ∧
    Function.Injective f :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_irrational_pairs_l597_59746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sasha_or_max_wrong_l597_59715

/-- A 100 x 100 table of nonzero digits -/
def Table := Fin 100 → Fin 100 → Fin 9

/-- The sum of digits in a row -/
def rowSum (t : Table) (i : Fin 100) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin 100)) fun j => (t i j).val.succ)

/-- The sum of digits in a column -/
def colSum (t : Table) (j : Fin 100) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin 100)) fun i => (t i j).val.succ)

/-- Sasha's claim: all row sums are divisible by 9 -/
def sashasClaim (t : Table) : Prop :=
  ∀ i : Fin 100, (rowSum t i) % 9 = 0

/-- Max's claim: exactly one column sum is not divisible by 9 -/
def maxsClaim (t : Table) : Prop :=
  ∃! j : Fin 100, (colSum t j) % 9 ≠ 0

theorem sasha_or_max_wrong (t : Table) : ¬(sashasClaim t ∧ maxsClaim t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sasha_or_max_wrong_l597_59715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_exists_l597_59717

theorem difference_exists (S : Finset ℕ) : 
  S.card = 700 → (∀ n, n ∈ S → n ≤ 2017) → 
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a - b = 3 ∨ a - b = 4 ∨ a - b = 7 ∨ b - a = 3 ∨ b - a = 4 ∨ b - a = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_exists_l597_59717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_50_equals_52_l597_59756

def G : ℕ → ℚ
  | 0 => 3  -- Adding the base case for 0
  | 1 => 3
  | (n + 1) => (3 * G n + 3) / 3

theorem G_50_equals_52 : G 50 = 52 := by
  -- The proof goes here
  sorry

#eval G 50  -- This will evaluate G(50) and display the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_50_equals_52_l597_59756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equality_l597_59758

def A : ℂ := 5 - 2*Complex.I
def M : ℂ := -3 + 3*Complex.I
def S : ℂ := -Complex.I
def P : ℝ := 3

theorem complex_expression_equality : 2*A - M + 3*S - (P : ℂ) = 10 - 10*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equality_l597_59758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gardener_care_area_l597_59781

/-- Represents a position in the 2D grid -/
structure Position where
  x : ℤ
  y : ℤ

/-- Represents a gardener -/
structure Gardener where
  position : Position

/-- Represents a flower -/
structure Flower where
  position : Position

/-- Predicate to check if three gardeners are the nearest to a flower -/
def areThreeNearest (g₁ g₂ g₃ : Gardener) (f : Flower) : Prop := sorry

/-- The field of flowers -/
def FlowerField : Type :=
  {flowers : Set Flower // ∀ f ∈ flowers, ∃ g₁ g₂ g₃ : Gardener, areThreeNearest g₁ g₂ g₃ f}

/-- Predicate to check if a position is slightly left below or right below another -/
def isSlightlyBelowLeftOrRight (p₁ p₂ : Position) : Prop := sorry

/-- Theorem: A gardener cares for flowers slightly left below or right below their position -/
theorem gardener_care_area (field : FlowerField) (g : Gardener) :
  ∀ f : Flower, f ∈ field.val → isSlightlyBelowLeftOrRight f.position g.position →
  ∃ g₁ g₂ : Gardener, areThreeNearest g g₁ g₂ f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gardener_care_area_l597_59781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machines_needed_for_reduced_time_l597_59742

/-- The number of additional machines needed to complete a job in reduced time -/
def additional_machines (original_machines : ℕ) (original_days : ℕ) (time_reduction_fraction : ℚ) : ℕ :=
  let new_days : ℚ := original_days * (1 - time_reduction_fraction)
  let work_rate : ℚ := 1 / (original_machines * original_days)
  let new_machines : ℚ := 1 / (work_rate * new_days)
  (Int.ceil new_machines).toNat - original_machines

/-- Given 15 machines can finish a job in 36 days, 3 more machines are needed to finish the job in one-eighth less time -/
theorem machines_needed_for_reduced_time : additional_machines 15 36 (1/8) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machines_needed_for_reduced_time_l597_59742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_sequence_l597_59725

/-- A geometric sequence is defined by its first term and common ratio -/
def geometric_sequence (a r : ℝ) : ℕ → ℝ :=
  fun n ↦ a * r^(n - 1)

theorem first_term_of_geometric_sequence
  (a r : ℝ) 
  (h1 : geometric_sequence a r 3 = 18)
  (h2 : geometric_sequence a r 5 = 162) :
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_sequence_l597_59725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_beverage_price_is_1_50_l597_59728

/-- Represents the price of a beverage bottle -/
def beverage_price : ℝ → Prop := sorry

/-- The number of movie tickets in the estimation unit -/
def tickets : ℕ := 6

/-- The number of grain cracker packs sold per estimation unit -/
def crackers : ℕ := 3

/-- The price of each grain cracker pack -/
def cracker_price : ℝ := 2.25

/-- The number of beverage bottles sold per estimation unit -/
def beverages : ℕ := 4

/-- The number of chocolate bars sold per estimation unit -/
def chocolates : ℕ := 4

/-- The price of each chocolate bar -/
def chocolate_price : ℝ := 1.00

/-- The average snack sales per movie ticket -/
def avg_sales : ℝ := 2.79

/-- 
Theorem stating that the beverage price that satisfies the given conditions
is approximately $1.50 (rounded to the nearest cent)
-/
theorem beverage_price_is_1_50 : 
  ∀ p : ℝ, beverage_price p → 
  (crackers * cracker_price + beverages * p + chocolates * chocolate_price) / tickets = avg_sales →
  ∃ ε > 0, |p - 1.50| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_beverage_price_is_1_50_l597_59728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l597_59721

/-- A parabola with focus on the x-axis passing through (1,1) -/
structure Parabola where
  /-- The equation of the parabola in the form y^2 = mx -/
  equation : ℝ → ℝ → Prop

/-- The exponential function passing through (1,1) -/
noncomputable def exp_func (a : ℝ) (x : ℝ) : ℝ := a^(x-1)

theorem parabola_equation (p : Parabola) :
  (∀ a : ℝ, a > 0 → a ≠ 1 → exp_func a 1 = 1) →
  p.equation 1 1 →
  ∀ x y : ℝ, p.equation x y ↔ y^2 = x := by
  sorry

#check parabola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l597_59721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weaving_rate_approx_l597_59792

/-- The rate of cloth weaving by an industrial loom -/
noncomputable def weaving_rate (cloth_length : ℝ) (time : ℝ) : ℝ :=
  cloth_length / time

/-- Theorem: The weaving rate is approximately 0.126 meters per second -/
theorem weaving_rate_approx :
  let cloth_length : ℝ := 15
  let time : ℝ := 119.04761904761905
  abs (weaving_rate cloth_length time - 0.126) < 0.001 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weaving_rate_approx_l597_59792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_is_pi_third_l597_59763

/-- The central angle corresponding to the minor arc intercepted by a line on a circle -/
noncomputable def central_angle (a b c : ℝ) (r : ℝ) : ℝ :=
  2 * Real.arccos ((c / Real.sqrt (a^2 + b^2)) / r)

/-- The line equation is √3x + y - 2√3 = 0 -/
def line_equation (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y - 2 * Real.sqrt 3 = 0

/-- The circle equation is x^2 + y^2 = 4 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

/-- The theorem stating that the central angle is π/3 -/
theorem central_angle_is_pi_third :
  central_angle (Real.sqrt 3) 1 (2 * Real.sqrt 3) 2 = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_is_pi_third_l597_59763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_parabola_l597_59795

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 10*x + 25 = 0

-- Define the parabola
def parabola_eq (x y : ℝ) : Prop := y^2 = 16*x

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem max_distance_circle_parabola :
  ∃ (x1 y1 x2 y2 : ℝ),
    circle_eq x1 y1 ∧ parabola_eq x2 y2 ∧
    ∀ (a1 b1 a2 b2 : ℝ),
      circle_eq a1 b1 → parabola_eq a2 b2 →
      distance x1 y1 x2 y2 ≥ distance a1 b1 a2 b2 ∧
      distance x1 y1 x2 y2 = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_parabola_l597_59795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_implies_m_value_l597_59764

/-- The function f(x) with parameter m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sin x ^ 4 + m * Real.sin x * Real.cos x + Real.cos x ^ 4

/-- The theorem stating that if the range of f is [0, 9/8], then m = ±1 -/
theorem f_range_implies_m_value (m : ℝ) :
  (∀ x, 0 ≤ f m x ∧ f m x ≤ 9/8) ∧ 
  (∃ x, f m x = 0) ∧ 
  (∃ x, f m x = 9/8) →
  m = 1 ∨ m = -1 := by
  sorry

#check f_range_implies_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_implies_m_value_l597_59764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_ways_l597_59780

-- Define the function that calculates the number of ways to choose one person
def number_of_ways_to_choose_one_person (method1_people method2_people : ℕ) : ℕ :=
  method1_people + method2_people

theorem task_completion_ways (method1_people method2_people : ℕ) :
  let total_ways := method1_people + method2_people
  total_ways = number_of_ways_to_choose_one_person method1_people method2_people :=
by
  -- Unfold the definition of number_of_ways_to_choose_one_person
  unfold number_of_ways_to_choose_one_person
  -- The rest of the proof is trivial, as both sides are now identical
  rfl

-- Example usage with the given problem values
def example_task_ways : ℕ :=
  number_of_ways_to_choose_one_person 3 5

#eval example_task_ways

end NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_ways_l597_59780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_k_range_l597_59713

-- Define the circle and points
noncomputable def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 8
def C : ℝ × ℝ := (-1, 0)
def A : ℝ × ℝ := (1, 0)

-- Define the conditions
def on_circle (P : ℝ × ℝ) : Prop := circle_eq P.1 P.2
def Q_on_CP (P Q : ℝ × ℝ) : Prop := ∃ t : ℝ, Q = C + t • (P - C)
def M_on_AP (A P M : ℝ × ℝ) : Prop := ∃ t : ℝ, M = A + t • (P - A)
def MQ_perp_AP (M Q A P : ℝ × ℝ) : Prop := (Q - M) • (P - A) = 0
def AP_twice_AM (A P M : ℝ × ℝ) : Prop := P - A = 2 • (M - A)

-- Define the tangent line and its properties
def tangent_line (k b : ℝ) : Prop := b^2 = k^2 + 1
def line_eq (k b x y : ℝ) : Prop := y = k * x + b
def intersect_trajectory (F H : ℝ × ℝ) : Prop :=
  F.1^2 / 2 + F.2^2 = 1 ∧ H.1^2 / 2 + H.2^2 = 1 ∧ F ≠ H
def dot_product_condition (O F H : ℝ × ℝ) : Prop :=
  3/4 ≤ (F - O) • (H - O) ∧ (F - O) • (H - O) ≤ 4/5

-- Main theorem
theorem trajectory_and_k_range :
  ∀ (P Q M F H : ℝ × ℝ) (k b : ℝ),
    on_circle P →
    Q_on_CP P Q →
    M_on_AP A P M →
    MQ_perp_AP M Q A P →
    AP_twice_AM A P M →
    tangent_line k b →
    intersect_trajectory F H →
    line_eq k b F.1 F.2 →
    line_eq k b H.1 H.2 →
    dot_product_condition (0, 0) F H →
    (∀ (x y : ℝ), Q.1 = x ∧ Q.2 = y → x^2 / 2 + y^2 = 1) ∧
    (-Real.sqrt 2 / 2 ≤ k ∧ k ≤ -Real.sqrt 3 / 3 ∨
     Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_k_range_l597_59713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_180_degrees_l597_59709

theorem triangle_angle_180_degrees 
  (a b c : ℝ) 
  (h : (a + b + c) * (a + b - c) = 4 * a * b) 
  (triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  C = π := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_180_degrees_l597_59709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l597_59793

-- Define the domain for both functions
def Domain : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- Define the two functions
noncomputable def f (μ : ℝ) : ℝ := Real.sqrt ((1 + μ) / (1 - μ))
noncomputable def g (v : ℝ) : ℝ := Real.sqrt ((1 + v) / (1 - v))

-- Theorem stating that the functions are identical on the given domain
theorem f_equals_g : ∀ x ∈ Domain, f x = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l597_59793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clearance_theorem_l597_59722

noncomputable def train_clearance_time (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let total_distance := length1 + length2
  let relative_speed := (speed1 + speed2) * (5 / 18)
  total_distance / relative_speed

theorem train_clearance_theorem :
  train_clearance_time 137 163 42 48 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clearance_theorem_l597_59722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_array_arrangement_impossible_l597_59700

theorem array_arrangement_impossible : ¬ ∃ (arr : Matrix (Fin 10) (Fin 10) ℕ),
  (∀ i j, 51 ≤ arr i j ∧ arr i j ≤ 150) ∧
  (∀ i j, i.val + 1 < 10 → 
    (∃ x y : ℤ, x^2 - (arr i j : ℤ) * x + (arr (i + 1) j : ℤ) = 0 ∧ 
                y^2 - (arr i j : ℤ) * y + (arr (i + 1) j : ℤ) = 0) ∨
    (∃ x y : ℤ, x^2 - (arr (i + 1) j : ℤ) * x + (arr i j : ℤ) = 0 ∧ 
                y^2 - (arr (i + 1) j : ℤ) * x + (arr i j : ℤ) = 0)) ∧
  (∀ i j, j.val + 1 < 10 → 
    (∃ x y : ℤ, x^2 - (arr i j : ℤ) * x + (arr i (j + 1) : ℤ) = 0 ∧ 
                y^2 - (arr i j : ℤ) * y + (arr i (j + 1) : ℤ) = 0) ∨
    (∃ x y : ℤ, x^2 - (arr i (j + 1) : ℤ) * x + (arr i j : ℤ) = 0 ∧ 
                y^2 - (arr i (j + 1) : ℤ) * x + (arr i j : ℤ) = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_array_arrangement_impossible_l597_59700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_symmetry_l597_59719

theorem parabola_intersection_symmetry (a b c d : ℝ) :
  let f (x : ℝ) := x^2 + a*x + b
  let g (x : ℝ) := x^2 + c*x + d
  (f 2019 = 0 ∧ g 2019 = 0) →
  (∃ r : ℝ, f r = 0 ∧ g (-r) = 0 ∧ r ≠ 2019) →
  (∃ y : ℝ, f 0 = y ∧ g 0 = -y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_symmetry_l597_59719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_position_after_1000_turns_l597_59757

/-- Represents the direction the ant is facing -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents the ant's position and state -/
structure AntState where
  x : Int
  y : Int
  direction : Direction
  stepLength : Nat

/-- Calculates the new position after a single move -/
def move (state : AntState) : AntState :=
  match state.direction with
  | Direction.North => { state with y := state.y + state.stepLength }
  | Direction.East => { state with x := state.x + state.stepLength }
  | Direction.South => { state with y := state.y - state.stepLength }
  | Direction.West => { state with x := state.x - state.stepLength }

/-- Turns the ant 90 degrees right and increases step length -/
def turn (state : AntState) : AntState :=
  { state with
    direction :=
      match state.direction with
      | Direction.North => Direction.East
      | Direction.East => Direction.South
      | Direction.South => Direction.West
      | Direction.West => Direction.North,
    stepLength := state.stepLength + 1
  }

/-- Performs a full cycle of move and turn -/
def cycle (state : AntState) : AntState :=
  turn (move state)

/-- The initial state of the ant -/
def initialState : AntState :=
  { x := 30, y := -30, direction := Direction.North, stepLength := 2 }

/-- Applies the cycle function n times -/
def applyNCycles (state : AntState) (n : Nat) : AntState :=
  match n with
  | 0 => state
  | n+1 => cycle (applyNCycles state n)

/-- The theorem to prove -/
theorem ant_position_after_1000_turns :
  (applyNCycles initialState 1000).x = 30 ∧
  (applyNCycles initialState 1000).y = 124720 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_position_after_1000_turns_l597_59757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l597_59751

/-- The line y = x -/
def line (x : ℝ) : ℝ := x

/-- The circle x^2 + y^2 = 1 -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- A and B are intersection points of the line and the circle -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 = A.2 ∧ unit_circle A.1 A.2 ∧
  line B.1 = B.2 ∧ unit_circle B.1 B.2

theorem length_of_AB (A B : ℝ × ℝ) (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l597_59751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_trip_buses_l597_59745

theorem field_trip_buses (students_per_bus : Float) (total_seats : Float) : 
  students_per_bus = 14.0 → total_seats = 28 → total_seats / students_per_bus = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_trip_buses_l597_59745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_lines_l597_59790

-- Define the triangle ABC
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (-1, 4)
def C : ℝ × ℝ := (4, 5)

-- Define the altitude line l₁
def l₁ (x y : ℝ) : Prop := 5 * x + y - 7 = 0

-- Define the equidistant line l₂
def l₂ (x y : ℝ) : Prop := (x + y - 9 = 0) ∨ (x - 2 * y + 6 = 0)

-- Theorem statement
theorem triangle_lines :
  -- l₁ is the altitude to BC
  (∀ x y : ℝ, l₁ x y ↔ (y - A.2 = -(B.2 - C.2)/(B.1 - C.1) * (x - A.1))) ∧
  -- l₂ passes through C and is equidistant from A and B
  (∀ x y : ℝ, l₂ x y →
    (x = C.1 ∧ y = C.2) ∨
    (abs ((y - A.2) - (A.1 - x) * (B.2 - A.2)/(B.1 - A.1)) /
      Real.sqrt (1 + ((B.2 - A.2)/(B.1 - A.1))^2) =
     abs ((y - B.2) - (B.1 - x) * (B.2 - A.2)/(B.1 - A.1)) /
      Real.sqrt (1 + ((B.2 - A.2)/(B.1 - A.1))^2))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_lines_l597_59790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_bisectors_l597_59731

/-- Definition: A triangle in 2D space -/
def Triangle (A B C : ℝ × ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

/-- Definition: Interior angle of a triangle -/
noncomputable def InteriorAngle (A B C : ℝ × ℝ) : ℝ :=
  sorry

/-- Definition: Angle between bisectors of two angles in a triangle -/
noncomputable def AngleBetweenBisectors (B A C : ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem: In a triangle where one interior angle measures 50°, 
    the angle between the bisectors of the other two interior angles is 65°. -/
theorem angle_between_bisectors (A B C : ℝ × ℝ) (h_triangle : Triangle A B C) 
  (h_angle : InteriorAngle A B C = 50) : 
  AngleBetweenBisectors B A C = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_bisectors_l597_59731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l597_59782

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 4 * x - 3 * y = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 10

-- State the theorem
theorem chord_length : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line_eq x₁ y₁ ∧ line_eq x₂ y₂ ∧ 
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 36 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l597_59782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_percentage_change_l597_59777

/-- Two positive numbers are inversely proportional if their product is constant. -/
def InverselyProportional (x y : ℝ) : Prop := ∃ k : ℝ, k > 0 ∧ x * y = k

/-- The percentage increase of a number. -/
noncomputable def PercentageIncrease (x : ℝ) (p : ℝ) : ℝ := x * (1 + p / 100)

/-- The percentage decrease of a number. -/
noncomputable def PercentageDecrease (y : ℝ) (q : ℝ) : ℝ := y * (1 - q / 100)

theorem inverse_proportion_percentage_change 
  (x y p : ℝ) (hx : x > 0) (hy : y > 0) (hp : p > 0) 
  (h_inverse : InverselyProportional x y) :
  ∃ y' : ℝ, 
    y' = PercentageDecrease y ((100 * p) / (100 + p)) ∧ 
    InverselyProportional (PercentageIncrease x p) y' :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_percentage_change_l597_59777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodic_problem_solution_l597_59710

/-- Sum of squares of digits of a natural number -/
def sumSquareDigits (n : ℕ) : ℕ := sorry

/-- The sequence defined in the problem -/
def sequenceTerms : ℕ → ℕ
  | 0 => 3243
  | n + 1 => sumSquareDigits (sequenceTerms n) + 2

/-- The sequence is periodic with period 3 starting from the 6th term -/
theorem sequence_periodic (n : ℕ) (h : n ≥ 6) : sequenceTerms n = sequenceTerms (n + 3) := by sorry

theorem problem_solution : sequenceTerms 1000 = 51 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodic_problem_solution_l597_59710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l597_59702

-- Define the function v(x)
noncomputable def v (x : ℝ) : ℝ := 1 / Real.sqrt (x - 4)

-- State the theorem about the domain of v(x)
theorem domain_of_v :
  {x : ℝ | IsRegular (v x)} = Set.Ioi 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l597_59702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_zero_solution_l597_59761

theorem sin_zero_solution (α : ℝ) (h1 : Real.sin α = 0) (h2 : α ∈ Set.Icc 0 (2 * Real.pi)) :
  α = 0 ∨ α = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_zero_solution_l597_59761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_sum_implies_c_l597_59768

-- Define the line equation
def line_equation (x y c : ℝ) : Prop := 3 * x + 5 * y + c = 0

-- Define the x-intercept
noncomputable def x_intercept (c : ℝ) : ℝ := -c / 3

-- Define the y-intercept
noncomputable def y_intercept (c : ℝ) : ℝ := -c / 5

-- Theorem statement
theorem intercept_sum_implies_c (c : ℝ) : 
  (x_intercept c + y_intercept c = 16) → c = -30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_sum_implies_c_l597_59768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_companion_equations_part1_companion_equation_part2_companion_equations_part3_l597_59798

-- Definition of companion equation
def is_companion_equation (eq : ℝ → Prop) (ineq : ℝ → Prop) : Prop :=
  ∀ x, eq x → ineq x

-- Part 1
theorem companion_equations_part1 :
  is_companion_equation (λ x ↦ x - 1 = 0) (λ x ↦ x + 1 > 0 ∧ x < 2) ∧
  is_companion_equation (λ x ↦ 2*x + 1 = 0) (λ x ↦ x + 1 > 0 ∧ x < 2) :=
sorry

-- Part 2
theorem companion_equation_part2 :
  ∀ k, is_companion_equation (λ x ↦ 2*x - k = 2) (λ x ↦ 3*x - 6 > 4 - x ∧ x - 1 ≥ 4*x - 10) ↔
    3 < k ∧ k ≤ 4 :=
sorry

-- Part 3
theorem companion_equations_part3 :
  ∀ m, m > 2 →
    (is_companion_equation (λ x ↦ 2*x + 4 = 0) (λ x ↦ (m - 2)*x < m - 2 ∧ x + 5 ≥ m) ∧
     is_companion_equation (λ x ↦ (2*x - 1)/3 = -1) (λ x ↦ (m - 2)*x < m - 2 ∧ x + 5 ≥ m)) ↔
    2 < m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_companion_equations_part1_companion_equation_part2_companion_equations_part3_l597_59798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_range_l597_59730

-- Define the function f and its derivative
def f : ℝ → ℝ := sorry
def f' (x a : ℝ) : ℝ := a * (x + 1) * (x - a)

-- State the theorem
theorem max_value_range (a : ℝ) (h1 : a < 0) 
  (h2 : ∀ x, HasDerivAt f (f' x a) x)
  (h3 : ∃ (M : ℝ), ∀ x, f x ≤ M ∧ f a = M) :
  -1 < a ∧ a < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_range_l597_59730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_pi_plus_alpha_l597_59729

theorem cos_negative_pi_plus_alpha (α : ℝ) :
  (∃ P : ℝ × ℝ, P = (-3, 4) ∧ P.1 = -3 * Real.cos α ∧ P.2 = 3 * Real.sin α) →
  Real.cos (-π - α) = 3/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_pi_plus_alpha_l597_59729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_intersections_intersection_count_l597_59753

/-- Given that a, b, and c form an arithmetic sequence, 
    prove that the discriminant of ax^2 + 2bx + c is non-negative -/
theorem quadratic_intersections (a b c : ℝ) 
  (h : 2 * b = a + c) : -- arithmetic sequence condition
  (2 * b)^2 - 4 * a * c ≥ 0 := by
  sorry

/-- The number of intersection points is either 1 or 2 -/
theorem intersection_count (a b c : ℝ) 
  (h : 2 * b = a + c) : -- arithmetic sequence condition
  let discriminant := (2 * b)^2 - 4 * a * c
  (discriminant > 0 ∧ (2 : ℕ) = 2) ∨ (discriminant = 0 ∧ (1 : ℕ) = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_intersections_intersection_count_l597_59753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_range_l597_59727

theorem log_base_range (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x y : ℝ, x < y → (Real.log (3/4) / Real.log a)^x < (Real.log (3/4) / Real.log a)^y) →
  3/4 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_range_l597_59727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_sets_satisfying_condition_l597_59750

open Set
open Finset

def set_135 : Finset ℕ := {1, 3, 5}
def set_13 : Finset ℕ := {1, 3}

theorem count_sets_satisfying_condition :
  (Finset.filter (fun A => set_13 ∪ A = set_135) (Finset.powerset set_135)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_sets_satisfying_condition_l597_59750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_integer_is_133_l597_59754

/-- Given three consecutive odd integers where the sum of the first and third is 131 less than 3 times the second, and the third integer is 133, prove that the third integer is 133. -/
theorem third_integer_is_133 
  (x : ℤ) -- First integer
  (h1 : ∃ k : ℤ, x = 2 * k + 1) -- x is odd
  (h2 : x + (x + 4) = 3 * (x + 2) - 131) -- Sum of first and third is 131 less than 3 times the second
  (h3 : x + 4 = 133) -- Third integer is 133
  : x + 4 = 133 := by
  exact h3

#check third_integer_is_133

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_integer_is_133_l597_59754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sides_of_folded_paper_l597_59738

/-- Represents a square piece of paper with white top and red bottom sides -/
structure Paper where
  side : ℝ
  is_square : side > 0

/-- A point within the square paper -/
structure Point (p : Paper) where
  x : ℝ
  y : ℝ
  within_square : 0 ≤ x ∧ x ≤ p.side ∧ 0 ≤ y ∧ y ≤ p.side

/-- The expected number of sides of the resulting red polygon -/
noncomputable def expected_sides (p : Paper) : ℝ := 5 - Real.pi / 2

/-- Theorem stating the expected number of sides of the red polygon -/
theorem expected_sides_of_folded_paper (p : Paper) :
  ∀ (F : Point p), expected_sides p = 5 - Real.pi / 2 := by
  intro F
  -- The proof is omitted for now
  sorry

#check expected_sides_of_folded_paper

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sides_of_folded_paper_l597_59738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_merger_workforce_proportion_l597_59783

theorem company_merger_workforce_proportion :
  ∀ (A B : ℝ), 
    (A > 0) → (B > 0) →
    (0.1 * A + 0.3 * B = 0.25 * (A + B)) →
    (A / (A + B) = 0.25) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_merger_workforce_proportion_l597_59783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lava_lamp_probability_l597_59712

def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 4
def total_lamps : ℕ := num_red_lamps + num_blue_lamps
def num_lamps_on : ℕ := 4

def probability_leftmost_blue_off_rightmost_red_on : ℚ := 4 / 49

theorem lava_lamp_probability :
  probability_leftmost_blue_off_rightmost_red_on = 
    (Nat.choose (total_lamps - 2) (num_red_lamps - 1))^2 / 
    ((Nat.choose total_lamps num_red_lamps) * (Nat.choose total_lamps num_lamps_on)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lava_lamp_probability_l597_59712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_implies_a_neg_b_pos_l597_59749

/-- A function f(x) with parameters a and b -/
noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - (1/2) * x^2 + b * x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a b x : ℝ) : ℝ := a / x - x + b

theorem function_minimum_implies_a_neg_b_pos (a b : ℝ) :
  (∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → f a b x ≤ f a b y) →
  a < 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_implies_a_neg_b_pos_l597_59749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_not_necessarily_parallelogram_l597_59797

/-- A trapezoid with bases a and c, and legs b and d -/
structure Trapezoid :=
  (a c b d : ℝ)
  (a_pos : 0 < a)
  (c_pos : 0 < c)
  (b_pos : 0 < b)
  (d_pos : 0 < d)
  (a_neq_c : a ≠ c)

/-- A line segment parallel to the bases of a trapezoid -/
noncomputable def parallelLine (t : Trapezoid) : ℝ := Real.sqrt ((t.a^2 + t.c^2) / 2)

/-- The condition for the line to divide the area in half -/
def dividesArea (t : Trapezoid) : Prop :=
  (parallelLine t + t.c) * (parallelLine t - t.c) / (t.a - t.c) = (t.a + t.c) / 2

/-- The condition for the line to divide the perimeter in half -/
def dividesPerimeter (t : Trapezoid) : Prop :=
  t.b + t.d = Real.sqrt (2 * t.a^2 + 2 * t.c^2) + t.a + t.c

/-- The theorem stating that such a trapezoid exists but is not necessarily a parallelogram -/
theorem trapezoid_not_necessarily_parallelogram :
  ∃ (t : Trapezoid), dividesArea t ∧ dividesPerimeter t ∧ t.a ≠ t.c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_not_necessarily_parallelogram_l597_59797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_binomial_expansion_l597_59743

theorem largest_coefficient_binomial_expansion :
  let n : ℕ := 9
  let expansion := fun (k : ℕ) => (n.choose k) * ((-1) ^ k : ℤ)
  (∃ m : ℕ, m ≤ n ∧ ∀ k : ℕ, k ≤ n → |expansion m| ≥ |expansion k|) ∧
  (∀ m : ℕ, m ≤ n → |expansion m| ≤ 126) ∧
  (∃ m : ℕ, m ≤ n ∧ |expansion m| = 126) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_binomial_expansion_l597_59743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l597_59778

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Define the line
def line_eq (k x y : ℝ) : Prop := (k + 1) * x - k * y - 1 = 0

-- Statement to prove
theorem chord_length (k : ℝ) : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ 
    line_eq k x₁ y₁ ∧ line_eq k x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l597_59778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elena_recipe_flour_l597_59794

/-- Represents Elena's bread recipe -/
structure Recipe where
  butter : ℚ  -- amount of butter in ounces
  flour : ℚ   -- amount of flour in cups

/-- The original recipe -/
noncomputable def original_recipe : Recipe := sorry

/-- The scaled up recipe (4 times the original) -/
noncomputable def scaled_recipe : Recipe := sorry

/-- The ratio of butter to flour in the recipe -/
noncomputable def butter_flour_ratio (r : Recipe) : ℚ := r.butter / r.flour

theorem elena_recipe_flour :
  -- The original recipe uses 2 ounces of butter for some cups of flour
  butter_flour_ratio original_recipe = 2 / original_recipe.flour →
  -- The scaled recipe is 4 times the original
  scaled_recipe.butter = 4 * original_recipe.butter →
  scaled_recipe.flour = 4 * original_recipe.flour →
  -- The scaled recipe uses 12 ounces of butter and 20 cups of flour
  scaled_recipe.butter = 12 →
  scaled_recipe.flour = 20 →
  -- The original recipe requires approximately 13 cups of flour
  ⌊original_recipe.flour⌋ = 13 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elena_recipe_flour_l597_59794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_many_representations_l597_59705

open Nat

theorem existence_of_many_representations : 
  ∃ (n : ℕ), n < 10^9 ∧ 
  (∃ (S : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ S → a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = n) ∧
    S.card > 1000) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_many_representations_l597_59705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l597_59714

theorem binomial_expansion_coefficient (x : ℝ) : 
  ∃ (c : ℤ), c = (Nat.choose 9 4) ∧ 
  c * x^3 = (Finset.range 10).sum (λ k ↦ 
    Nat.choose 9 k * (-1)^k * x^(9 - 3*k/2) * 
    if 9 - 3*k/2 = 3 then 1 else 0)
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l597_59714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gerald_toy_cars_l597_59755

theorem gerald_toy_cars (initial_cars : ℕ) (donated_fraction : ℚ) (remaining_cars : ℕ) : 
  initial_cars = 20 → 
  donated_fraction = 1 / 4 →
  remaining_cars = initial_cars - (initial_cars * donated_fraction).floor →
  remaining_cars = 15 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gerald_toy_cars_l597_59755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_rectangle_l597_59736

noncomputable def perimeter : ℝ := 16

-- Define the area function for a rectangle given one side length
noncomputable def area (x : ℝ) : ℝ := x * (perimeter / 2 - x)

theorem max_area_rectangle :
  ∃ (max_area : ℝ), ∀ (x : ℝ), 0 < x → x < perimeter / 2 → area x ≤ max_area ∧
  ∃ (optimal_x : ℝ), 0 < optimal_x ∧ optimal_x < perimeter / 2 ∧ area optimal_x = max_area ∧
  max_area = (perimeter / 4) ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_rectangle_l597_59736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_arithmetic_sequence_l597_59724

/-- An arithmetic sequence with common difference d -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a₁  -- Add case for 0
  | n + 1 => arithmeticSequence a₁ d n + d

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def sumArithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a₁ + (n - 1) * d) / 2

theorem min_sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) :
  d > 0 →
  arithmeticSequence a₁ d 1 + arithmeticSequence a₁ d 11 = 0 →
  ∃ n : ℕ, n = 5 ∨ n = 6 ∧
    ∀ k : ℕ, sumArithmeticSequence a₁ d n ≤ sumArithmeticSequence a₁ d k :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_arithmetic_sequence_l597_59724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_investment_l597_59703

-- Define the profit functions
def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1) + 2

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := 6 * Real.log (x + b)

-- Define the total profit function
noncomputable def s (x : ℝ) : ℝ := f 2 (5 - x) + g 1 x

-- State the theorem
theorem optimal_investment :
  ∃ (x : ℝ), 0 < x ∧ x ≤ 5 ∧
  (∀ (y : ℝ), 0 < y ∧ y ≤ 5 → s x ≥ s y) ∧
  x = 2 ∧ s x > 12.5 ∧ s x < 12.7 := by
  sorry

#eval Float.sin 1  -- This is just to test if Mathlib is properly imported

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_investment_l597_59703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trisecting_chords_theorem_l597_59723

/-- Given a circle with center O and radius r, and an interior point P,
    this function returns the number of chords that P trisects. -/
noncomputable def num_trisecting_chords (O P : EuclideanSpace ℝ (Fin 2)) (r : ℝ) : ℕ :=
  let d := ‖P - O‖
  if d > r / 3 then 2
  else if d = r / 3 then 1
  else 0

/-- Theorem stating the relationship between the position of P and the number of chords it trisects -/
theorem trisecting_chords_theorem (O P : EuclideanSpace ℝ (Fin 2)) (r : ℝ) 
    (h_circle : ‖P - O‖ < r) : 
  num_trisecting_chords O P r = 
    if ‖P - O‖ > r / 3 then 2
    else if ‖P - O‖ = r / 3 then 1
    else 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trisecting_chords_theorem_l597_59723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l597_59704

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- Theorem: The area of a triangle with vertices at (0,0), (2,3), and (6,8) is 17 square units -/
theorem triangle_area_specific : triangleArea 0 0 2 3 6 8 = 17 := by
  -- Unfold the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l597_59704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l597_59774

-- Define the hyperbola
noncomputable def hyperbola (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4

-- Define the positive slope of an asymptote
noncomputable def positive_asymptote_slope : ℝ := Real.sqrt 5 / 2

-- Theorem statement
theorem hyperbola_asymptote_slope :
  ∃ (x y : ℝ), hyperbola x y ∧ positive_asymptote_slope > 0 ∧
  (∀ (ε : ℝ), ε > 0 → ∃ (x' y' : ℝ), hyperbola x' y' ∧
    |((y' - y) / (x' - x) - positive_asymptote_slope)| < ε) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l597_59774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_l597_59779

-- Define the points
variable (C D E F : ℝ × ℝ)

-- Define the distances
def CD : ℝ := 12
def DE : ℝ := 15
def CF : ℝ := 15
def EC : ℝ := 20
def FD : ℝ := 20

-- Define the triangles
def triangle_CDE (C D E : ℝ × ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (p = (1 - t) • C + t • D ∨ p = (1 - t) • D + t • E ∨ p = (1 - t) • E + t • C)}
def triangle_CFD (C F D : ℝ × ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (p = (1 - t) • C + t • F ∨ p = (1 - t) • F + t • D ∨ p = (1 - t) • D + t • C)}

-- Define the congruence of triangles
def congruent_triangles (t1 t2 : Set (ℝ × ℝ)) : Prop := ∃ (f : ℝ × ℝ → ℝ × ℝ), Isometry f ∧ f '' t1 = t2

-- Define the intersection of triangles
def intersection (C D E F : ℝ × ℝ) : Set (ℝ × ℝ) := triangle_CDE C D E ∩ triangle_CFD C F D

-- Define the area function
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem intersection_area (C D E F : ℝ × ℝ) :
  congruent_triangles (triangle_CDE C D E) (triangle_CFD C F D) →
  area (intersection C D E F) = 807.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_l597_59779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_zero_l597_59799

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def not_collinear (x y : V) : Prop := ∀ (k : ℝ), x ≠ k • y

theorem vector_sum_zero 
  (a b c : V) 
  (h1 : not_collinear a b ∧ not_collinear b c ∧ not_collinear a c)
  (h2 : ∃ (k : ℝ), a + b = k • c)
  (h3 : ∃ (m : ℝ), b + c = m • a) :
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_zero_l597_59799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_iff_K_bounded_l597_59718

/-- The function f parameterized by K -/
noncomputable def f (K : ℝ) (x : ℝ) : ℝ := (x^4 + K*x^2 + 1) / (x^4 + x^2 + 1)

/-- The triangle inequality holds for f(K) if and only if -1/2 < K < 4 -/
theorem triangle_inequality_iff_K_bounded (K : ℝ) :
  (∀ a b c : ℝ, 2 * min (f K a) (min (f K b) (f K c)) > max (f K a) (max (f K b) (f K c))) ↔
  -1/2 < K ∧ K < 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_iff_K_bounded_l597_59718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_tangent_intersections_l597_59708

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  (a_pos : 0 < a)
  (b_pos : 0 < b)

/-- A point on the x-axis -/
def PointOnXAxis (x : ℝ) : ℝ × ℝ := (x, 0)

/-- A point on the y-axis -/
def PointOnYAxis (y : ℝ) : ℝ × ℝ := (0, y)

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: The minimum distance between intersections of a tangent line 
    to an ellipse with the x and y axes is equal to the sum of the semi-axes -/
theorem min_distance_tangent_intersections (a b : ℝ) (e : Ellipse a b) :
  ∃ (A : ℝ × ℝ) (B : ℝ × ℝ),
    (∃ x, A = PointOnXAxis x) ∧ 
    (∃ y, B = PointOnYAxis y) ∧
    (∀ C D, (∃ x', C = PointOnXAxis x') → (∃ y', D = PointOnYAxis y') →
      distance A B ≤ distance C D) ∧
    distance A B = a + b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_tangent_intersections_l597_59708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_is_rectangle_is_minor_premise_square_is_rectangle_is_minor_premise_actual_l597_59741

-- Define the basic shapes as classes
class Rectangle
class Square
class Parallelogram

-- Define the relationships between shapes
axiom rectangle_is_parallelogram : Rectangle → Parallelogram
axiom square_is_rectangle : Square → Rectangle

-- Define the syllogism
def syllogism (major minor conclusion : Prop) : Prop :=
  (major ∧ minor) → conclusion

-- Define our specific syllogism
def square_parallelogram_syllogism : Prop :=
  syllogism 
    (∀ r : Rectangle, ∃ p : Parallelogram, True)
    (∀ s : Square, ∃ r : Rectangle, True)
    (∀ s : Square, ∃ p : Parallelogram, True)

-- Theorem: The statement "A square is a rectangle" is the minor premise
theorem square_is_rectangle_is_minor_premise :
  (∀ s : Square, ∃ r : Rectangle, True) = 
  (∀ s : Square, ∃ r : Rectangle, True) := by
  rfl

-- The actual proof is omitted
theorem square_is_rectangle_is_minor_premise_actual :
  (∀ s : Square, ∃ r : Rectangle, True) = 
  (∀ s : Square, ∃ r : Rectangle, True) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_is_rectangle_is_minor_premise_square_is_rectangle_is_minor_premise_actual_l597_59741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_decomposition_product_l597_59767

noncomputable section

-- Define the polynomial in the numerator
def numerator (x : ℝ) : ℝ := x^2 + 25

-- Define the polynomial in the denominator
def denominator (x : ℝ) : ℝ := x^3 + 2*x^2 - 13*x - 14

-- Define the partial fraction decomposition
def partial_fraction (A B C x : ℝ) : ℝ := A / (x + 2) + B / (x - 1) + C / (x + 7)

-- State the theorem
theorem partial_fraction_decomposition_product :
  ∃ (A B C : ℝ), 
    (∀ x, x ≠ -2 ∧ x ≠ 1 ∧ x ≠ -7 → 
      numerator x / denominator x = partial_fraction A B C x) ∧
    A * B * C = 13949 / 2700 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_decomposition_product_l597_59767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l597_59733

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 4 * x ≤ 6 then (4 - a / 2) * x else a^(x - 5)

-- Define the sequence a_n
noncomputable def a_n (a : ℝ) : ℕ → ℝ
| 0 => 1  -- Arbitrary initial value
| n + 1 => f a (a_n a n)

-- State the theorem
theorem a_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ n : ℕ, (n : ℝ) = f a n) →
  (∀ n : ℕ, a_n a (n + 1) > a_n a n) →
  a ∈ Set.Ioo 1 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l597_59733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_paving_l597_59766

/-- Calculates the minimum number of whole bricks required to pave a courtyard -/
def min_bricks_required (courtyard_length courtyard_width brick_length brick_width : ℚ) : ℕ :=
  let courtyard_area := courtyard_length * courtyard_width * 10000
  let brick_area := brick_length * brick_width
  (courtyard_area / brick_area).ceil.toNat

/-- The minimum number of whole bricks required to pave the courtyard is 107143 -/
theorem courtyard_paving :
  min_bricks_required 45 25 15 7 = 107143 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_paving_l597_59766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_21_l597_59748

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence with S_7 = 48 and S_14 = 60, S_21 = 63 -/
theorem geometric_sequence_sum_21 (a q : ℝ) (h1 : geometric_sum a q 7 = 48) 
  (h2 : geometric_sum a q 14 = 60) : geometric_sum a q 21 = 63 := by
  sorry

#check geometric_sequence_sum_21

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_21_l597_59748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sophia_sees_sam_for_twenty_minutes_l597_59737

/-- The time (in minutes) Sophia can see Sam given their speeds and the distance covered --/
noncomputable def time_sophia_sees_sam (sophia_speed sam_speed : ℝ) (initial_distance final_distance : ℝ) : ℝ :=
  ((initial_distance + final_distance) / (sophia_speed - sam_speed)) * 60

theorem sophia_sees_sam_for_twenty_minutes :
  let sophia_speed : ℝ := 20
  let sam_speed : ℝ := 14
  let initial_distance : ℝ := 1
  let final_distance : ℝ := 1
  time_sophia_sees_sam sophia_speed sam_speed initial_distance final_distance = 20 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_sophia_sees_sam 20 14 1 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sophia_sees_sam_for_twenty_minutes_l597_59737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercept_circle_l597_59735

/-- The circle equation x^2 + y^2 - 2x - 2y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- A line passing through (1,0) -/
def line_through_point (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

/-- The length of a chord intercepted by a line on the circle -/
noncomputable def chord_length (k : ℝ) : ℝ := 2 * Real.sqrt (k^2 / (k^2 + 1))

/-- The theorem stating that y = x - 1 and y = -x + 1 are the only lines
    passing through (1,0) and intercepted by a chord of length √2 on the given circle -/
theorem line_intercept_circle :
  ∀ k : ℝ, (line_through_point k 1 0 ∧ chord_length k = Real.sqrt 2) ↔ (k = 1 ∨ k = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercept_circle_l597_59735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_of_52200625_l597_59747

theorem fourth_root_of_52200625 : (52200625 : ℝ).sqrt.sqrt = 51 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_of_52200625_l597_59747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_root_l597_59775

theorem quadratic_polynomial_root : 
  let p : ℂ → ℂ := λ z => 3 * z^2 - 30 * z + 87
  (p (5 + 2*Complex.I) = 0) ∧ 
  (∃ a b c : ℝ, ∀ z : ℂ, p z = a * z^2 + b * z + c) ∧
  (∀ z : ℂ, p z = 3 * z^2 + (p z - 3 * z^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_root_l597_59775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_theorem_l597_59791

/-- Represents the shortest distance between max and min numbers -/
noncomputable def shortest_distance (n : ℕ) : ℝ :=
  if n % 2 = 0 then
    Real.sqrt 3 / 2
  else
    Real.sqrt (3 + 1 / (n^2 : ℝ)) / 2

/-- Represents an equilateral triangle with the given properties -/
structure EquilateralTriangle (n : ℕ) where
  sideLength : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem statement -/
theorem equilateral_triangle_theorem (n : ℕ) (triangle : EquilateralTriangle n) :
  triangle.sideLength = 1 →
  (∀ (rhombus : Fin 4 → ℝ), rhombus 0 + rhombus 2 = rhombus 1 + rhombus 3) →
  (∃ (S : ℝ), S = (1 / 6 : ℝ) * (n + 1 : ℝ) * (n + 2 : ℝ) * (triangle.a + triangle.b + triangle.c)) ∧
  (∃ (r : ℝ), r = shortest_distance n) := by
  sorry

#check equilateral_triangle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_theorem_l597_59791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l597_59769

/-- The area of a triangle given the coordinates of its vertices -/
noncomputable def triangleArea (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  (1/2) * |x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)|

/-- Theorem: The area of triangle PQR with vertices P(-3, 4), Q(4, 9), and R(5, -3) is 44.5 square units -/
theorem triangle_PQR_area :
  triangleArea (-3) 4 4 9 5 (-3) = 44.5 := by
  -- Expand the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp [abs_of_nonneg]
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l597_59769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_properties_l597_59734

/-- Represents a theater with a given number of rows. -/
structure Theater where
  rows : ℕ

/-- Represents a class of students. -/
structure StudentClass where
  students : ℕ

/-- Predicts whether at least two students will be in the same row when seated. -/
def atLeastTwoInSameRow (t : Theater) (c : StudentClass) : Prop :=
  c.students > t.rows

/-- Calculates the number of empty rows after seating students. -/
def emptyRows (t : Theater) (c : StudentClass) : ℕ :=
  if t.rows ≥ c.students then t.rows - c.students else 0

/-- The main theorem stating the properties of a 29-row theater. -/
theorem theater_properties :
  ∃ (t : Theater),
    t.rows = 29 ∧
    (∀ (c : StudentClass), c.students = 30 → atLeastTwoInSameRow t c) ∧
    (∀ (c : StudentClass), c.students = 26 → emptyRows t c ≥ 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_properties_l597_59734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_income_maximizes_take_home_pay_l597_59726

/-- Calculates the tax rate percentage based on income in thousands of dollars -/
noncomputable def tax_rate (x : ℝ) : ℝ := x + 10

/-- Calculates the take-home pay in dollars given income in thousands of dollars -/
noncomputable def take_home_pay (x : ℝ) : ℝ := 1000 * x - (tax_rate x / 100) * (1000 * x)

/-- The income in dollars that maximizes take-home pay -/
def optimal_income : ℝ := 45000

theorem optimal_income_maximizes_take_home_pay :
  ∀ x : ℝ, x > 0 → take_home_pay (optimal_income / 1000) ≥ take_home_pay (x / 1000) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_income_maximizes_take_home_pay_l597_59726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_interval_l597_59740

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 2 ≤ (2:ℝ)^x ∧ (2:ℝ)^x ≤ 8}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the open-closed interval (2, 3]
def interval_2_3 : Set ℝ := {x | 2 < x ∧ x ≤ 3}

-- Theorem statement
theorem intersection_equals_interval : A_intersect_B = interval_2_3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_interval_l597_59740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_asymptote_distance_l597_59707

/-- The distance from the focus of a hyperbola to its asymptote -/
noncomputable def distance_focus_to_asymptote (b : ℝ) : ℝ :=
  let focus := (3 : ℝ)
  let asymptote_slope := b / 2
  |focus * asymptote_slope - 0| / Real.sqrt (asymptote_slope^2 + 1)

/-- The theorem stating the distance from the focus of the hyperbola to its asymptote -/
theorem hyperbola_focus_asymptote_distance :
  ∃ b : ℝ,
    (∀ x y : ℝ, x^2 / 4 - y^2 / b^2 = 1 → y^2 = 12 * x) →
    distance_focus_to_asymptote b = Real.sqrt 5 := by
  -- We'll use b = √5
  use Real.sqrt 5
  intro h
  -- Calculate the distance
  simp [distance_focus_to_asymptote]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_asymptote_distance_l597_59707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cube_product_exists_l597_59776

/-- A set of integers whose prime factors are limited to 2 and 3 -/
def LimitedFactorSet (S : Set Int) : Prop :=
  ∀ n ∈ S, ∃ a b : ℕ, n = 2^a * 3^b ∨ n = -(2^a * 3^b)

/-- The existence of three distinct elements in S whose product is a perfect cube -/
def ExistsPerfectCubeProduct (S : Set Int) : Prop :=
  ∃ x y z : Int, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ ∃ k : Int, x * y * z = k^3

theorem perfect_cube_product_exists (S : Set Int) 
  (h1 : Fintype S) 
  (h2 : Fintype.card S = 9) 
  (h3 : LimitedFactorSet S) : 
  ExistsPerfectCubeProduct S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cube_product_exists_l597_59776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l597_59785

/-- A cubic function with specific properties -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (a/2) * x^2 + 2*x + b

/-- The theorem stating the properties of the function f -/
theorem cubic_function_properties :
  ∃ (a b : ℝ),
    (∀ x, f a b x = (1/3) * x^3 + (a/2) * x^2 + 2*x + b) ∧
    (f a b 1 = 11/6) ∧
    (∀ x, x ≠ 1 → f a b x ≤ 11/6) ∧
    (f a b 3 = 5/2) ∧
    (f a b 2 = 5/3) ∧
    (∀ x, x ∈ Set.Icc 1 3 → f a b x ≤ 5/2) ∧
    (∀ x, x ∈ Set.Icc 1 3 → f a b x ≥ 5/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l597_59785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l597_59789

theorem cube_root_simplification : 
  (54880000 : ℝ) ^ (1/3) = 20 * (6850 : ℝ) ^ (1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l597_59789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l597_59762

/-- The function f(x) = sin(2x - π/2) for x ∈ R -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

theorem f_properties :
  (∀ x, f (-x) = f x) ∧  -- f is even
  (∀ p, p > 0 ∧ (∀ x, f (x + p) = f x) → p ≥ Real.pi) ∧  -- π is the least positive period
  (∀ x, f (x + Real.pi) = f x)  -- π is a period
  := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l597_59762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_51_equals_3_l597_59796

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3

-- State the theorem
theorem inverse_f_51_equals_3 : 
  ∃ (f_inv : ℝ → ℝ), Function.RightInverse f_inv f ∧ f_inv 51 = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_51_equals_3_l597_59796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_returns_to_one_l597_59770

def sequenceA (d : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => let a := sequenceA d n
              if a % 2 = 0 then a / 2 else a + d

theorem sequence_returns_to_one (d : ℕ) :
  (∃ n : ℕ, n > 0 ∧ sequenceA d n = 1) ↔ d % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_returns_to_one_l597_59770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_diophantine_equation_l597_59732

theorem unique_solution_diophantine_equation :
  ∃! (x y z n : ℕ), 
    (n ≥ 2) ∧
    (z ≤ 5 * 2^(2*n)) ∧
    (x > 0) ∧ (y > 0) ∧ (z > 0) ∧
    ((x : ℤ)^(2*n + 1) - (y : ℤ)^(2*n + 1) = (x : ℤ)*(y : ℤ)*(z : ℤ) + 2^(2*n + 1)) ∧
    (x = 3 ∧ y = 1 ∧ z = 70 ∧ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_diophantine_equation_l597_59732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_circular_ellipse_l597_59716

/-- The eccentricity of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

/-- The equation of an ellipse parameterized by m -/
def ellipse_equation (m : ℝ) : (ℝ × ℝ) → Prop :=
  fun (x, y) ↦ x^2 / m + y^2 / (m^2 + 1) = 1

/-- Theorem stating the conditions for the most circular ellipse -/
theorem most_circular_ellipse (m : ℝ) (hm : m > 0) :
  (∀ k > 0, eccentricity (Real.sqrt (m^2 + 1)) (Real.sqrt m) ≤ eccentricity (Real.sqrt (k^2 + 1)) (Real.sqrt k)) →
  m = 1 ∧ ellipse_equation m = fun (x, y) ↦ x^2 + y^2 / 2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_circular_ellipse_l597_59716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_coin_collection_l597_59773

theorem smallest_coin_collection (n : ℕ) : 
  (∃ (d₁ d₂ : ℕ), 1 < d₁ ∧ d₁ < n ∧ 1 < d₂ ∧ d₂ < n ∧ d₁ ≠ d₂ ∧ d₁ ∣ n ∧ d₂ ∣ n) →
  (Finset.filter (λ d : ℕ ↦ 1 < d ∧ d < n ∧ d ∣ n) (Finset.range (n + 1))).card = 17 →
  3 ∣ n →
  n ≥ 262144 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_coin_collection_l597_59773
