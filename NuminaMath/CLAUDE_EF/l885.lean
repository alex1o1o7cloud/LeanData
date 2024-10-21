import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_properties_l885_88521

noncomputable def f (x : ℝ) := Real.cos (2 * x + Real.pi / 3)

theorem cosine_properties :
  ∃ (k : ℤ),
    (∀ (x : ℝ), f x = f (k * Real.pi + Real.pi / 6 - x)) ∧
    (∀ (x y : ℝ), x ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3) →
      y ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3) →
      x ≤ y → f x ≥ f y) ∧
    (∀ (β : ℝ), β ∈ Set.Ioo 0 Real.pi → f (β / 2) = -1/2 → β = Real.pi / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_properties_l885_88521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_asymptotes_and_holes_l885_88576

/-- The function f(x) = (x^2 - 4x + 4) / (x^3 - 2x^2 - x + 2) -/
noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x + 4) / (x^3 - 2*x^2 - x + 2)

/-- Number of holes in the graph of f -/
def num_holes : ℕ := 1

/-- Number of vertical asymptotes in the graph of f -/
def num_vertical_asymptotes : ℕ := 2

/-- Number of horizontal asymptotes in the graph of f -/
def num_horizontal_asymptotes : ℕ := 1

/-- Number of oblique asymptotes in the graph of f -/
def num_oblique_asymptotes : ℕ := 0

/-- Theorem stating the sum of asymptotes and holes -/
theorem sum_of_asymptotes_and_holes :
  num_holes + 2 * num_vertical_asymptotes + 3 * num_horizontal_asymptotes + 4 * num_oblique_asymptotes = 8 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_asymptotes_and_holes_l885_88576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l885_88530

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + Real.cos x ^ 2 - Real.sin x ^ 2

-- Theorem statement
theorem f_properties :
  -- The smallest positive period is π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- The minimum value in [0, π/2] is -√2
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ -Real.sqrt 2) ∧
  -- The minimum occurs at x = π/2
  (f (Real.pi / 2) = -Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l885_88530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l885_88572

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y + y * f x) = f x + f y + x * f y) →
  ((∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l885_88572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_difference_l885_88520

/-- Simple interest calculation -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_difference
  (principal : ℝ)
  (time : ℝ)
  (original_rate : ℝ)
  (higher_rate : ℝ)
  (interest_difference : ℝ)
  (h1 : principal = 3000)
  (h2 : time = 9)
  (h3 : simple_interest principal higher_rate time -
        simple_interest principal original_rate time = interest_difference)
  (h4 : interest_difference = 1350) :
  higher_rate - original_rate = 5 := by
  sorry

#eval "Theorem defined successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_difference_l885_88520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l885_88528

/-- A domino covers two adjacent cells sharing a side -/
structure Domino where
  cell1 : Nat × Nat
  cell2 : Nat × Nat
  adjacent : (cell1.1 = cell2.1 ∧ cell1.2.succ = cell2.2) ∨
             (cell1.1 = cell2.1 ∧ cell1.2 = cell2.2.succ) ∨
             (cell1.1.succ = cell2.1 ∧ cell1.2 = cell2.2) ∨
             (cell1.1 = cell2.1.succ ∧ cell1.2 = cell2.2)

/-- Two dominoes do not overlap if they don't share any points -/
def non_overlapping (d1 d2 : Domino) : Prop :=
  d1.cell1 ≠ d2.cell1 ∧ d1.cell1 ≠ d2.cell2 ∧
  d1.cell2 ≠ d2.cell1 ∧ d1.cell2 ≠ d2.cell2

/-- A valid domino placement on an n × n grid -/
def valid_placement (n : Nat) (dominoes : Finset Domino) : Prop :=
  dominoes.card = 1014 ∧
  (∀ d, d ∈ dominoes → d.cell1.1 < n ∧ d.cell1.2 < n ∧ d.cell2.1 < n ∧ d.cell2.2 < n) ∧
  (∀ d1 d2, d1 ∈ dominoes → d2 ∈ dominoes → d1 ≠ d2 → non_overlapping d1 d2)

/-- The main theorem: 77 is the smallest n for which a valid placement exists -/
theorem smallest_valid_n :
  (∃ dominoes, valid_placement 77 dominoes) ∧
  (∀ m, m < 77 → ¬∃ dominoes, valid_placement m dominoes) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l885_88528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_benson_ticket_purchase_l885_88552

/-- Calculates the number of tickets bought given the ticket cost, discount rate, discount threshold, and total payment -/
def calculate_tickets (ticket_cost : ℚ) (discount_rate : ℚ) (discount_threshold : ℕ) (total_payment : ℚ) : ℕ :=
  let discounted_price := ticket_cost * (1 - discount_rate)
  let full_price_cost := ticket_cost * discount_threshold
  let remaining_cost := total_payment - full_price_cost
  let discounted_tickets := remaining_cost / discounted_price
  discount_threshold + (Int.floor discounted_tickets).toNat

/-- Theorem stating that given the specific conditions, the number of tickets bought is 12 -/
theorem benson_ticket_purchase :
  calculate_tickets 40 (5/100) 10 476 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_benson_ticket_purchase_l885_88552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_neg_five_sixths_pi_l885_88569

theorem sin_neg_five_sixths_pi : Real.sin (-5/6 * Real.pi) = -(1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_neg_five_sixths_pi_l885_88569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_of_profit_l885_88578

/-- Calculates the share of profit for partner A in a business partnership --/
def calculate_share_of_profit (a_initial : ℕ) (b_initial : ℕ) (a_change : ℤ) (b_change : ℕ) (total_profit : ℕ) : ℕ :=
  let a_investment_months : ℤ := a_initial * 8 + (a_initial + a_change) * 4
  let b_investment_months : ℕ := b_initial * 8 + (b_initial + b_change) * 4
  let total_parts : ℤ := a_investment_months + b_investment_months
  ((a_investment_months * total_profit) / total_parts).toNat

theorem a_share_of_profit :
  calculate_share_of_profit 3000 4000 (-1000) 1000 756 = 288 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_of_profit_l885_88578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l885_88553

noncomputable def proj_b_on_a (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (a.1^2 + a.2^2)

theorem projection_theorem (a b : ℝ × ℝ) :
  (Real.sqrt (a.1^2 + a.2^2) = 1) →
  (b = (Real.sqrt 3 / 3, Real.sqrt 3 / 3)) →
  (Real.sqrt ((a.1 + 3 * b.1)^2 + (a.2 + 3 * b.2)^2) = 2) →
  (proj_b_on_a a b = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l885_88553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_opposite_intercepts_l885_88506

/-- A line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) is on a line --/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line passes through the origin --/
def Line.throughOrigin (l : Line) : Prop :=
  l.c = 0

/-- Get the x-intercept of a line --/
noncomputable def Line.xIntercept (l : Line) : ℝ :=
  -l.c / l.a

/-- Get the y-intercept of a line --/
noncomputable def Line.yIntercept (l : Line) : ℝ :=
  -l.c / l.b

/-- Check if the intercepts of a line are opposite in sign --/
noncomputable def Line.oppositeIntercepts (l : Line) : Prop :=
  l.xIntercept * l.yIntercept < 0

theorem line_through_point_with_opposite_intercepts :
  ∃ (l1 l2 : Line),
    (l1.contains 2 3 ∧ l1.oppositeIntercepts) ∧
    (l2.contains 2 3 ∧ l2.oppositeIntercepts) ∧
    ((l1.a = 1 ∧ l1.b = -1 ∧ l1.c = -1) ∨ (l1.a = 3 ∧ l1.b = -2 ∧ l1.c = 0)) ∧
    ((l2.a = 1 ∧ l2.b = -1 ∧ l2.c = -1) ∨ (l2.a = 3 ∧ l2.b = -2 ∧ l2.c = 0)) ∧
    l1 ≠ l2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_opposite_intercepts_l885_88506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bishop_placement_count_l885_88568

/-- Represents a chessboard with dark and light squares -/
structure Chessboard :=
  (dark_squares : Set (Fin 8 × Fin 8))
  (light_squares : Set (Fin 8 × Fin 8))

/-- Represents a bishop on the chessboard -/
structure Bishop :=
  (position : Fin 8 × Fin 8)

/-- Represents a diagonal on the chessboard -/
structure Diagonal :=
  (squares : Set (Fin 8 × Fin 8))

/-- The number of diagonals on each color of the chessboard -/
def num_diagonals : ℕ := 7

/-- A function that checks if two bishops threaten each other -/
def threaten (b1 b2 : Bishop) : Prop := sorry

/-- A function that returns all valid placements of 14 bishops on the chessboard -/
def valid_placements (cb : Chessboard) : Finset (Finset Bishop) := sorry

/-- The main theorem stating that there are 256 ways to place 14 bishops on a chessboard
    so that no two bishops threaten each other -/
theorem bishop_placement_count (cb : Chessboard) :
  Finset.card (valid_placements cb) = 256 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bishop_placement_count_l885_88568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_m_bounds_l885_88503

-- Define the complex numbers z₁, z₂, and m
variable (z₁ z₂ m : ℂ)

-- Define the roots α and β
variable (α β : ℂ)

-- State the conditions
def quadratic_equation (z₁ z₂ m : ℂ) (x : ℂ) : Prop :=
  x^2 + z₁*x + z₂ + m = 0

def root_condition (z₁ z₂ m α β : ℂ) : Prop :=
  quadratic_equation z₁ z₂ m α ∧ quadratic_equation z₁ z₂ m β

def z_condition (z₂ : ℂ) : Prop :=
  ∃ z : ℂ, z - 4*z₂ = 16 + 20*Complex.I

def root_distance (α β : ℂ) : Prop :=
  Complex.abs (α - β) = 2

-- State the theorem
theorem quadratic_m_bounds (z₁ z₂ m α β : ℂ) :
  z_condition z₂ →
  root_condition z₁ z₂ m α β →
  root_distance α β →
  (Complex.abs m ≤ 7 + Real.sqrt 37) ∧
  (Complex.abs m ≥ 7 - Real.sqrt 37) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_m_bounds_l885_88503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_AOB_l885_88575

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 12 = 1

/-- Point A on the ellipse -/
noncomputable def point_A : ℝ × ℝ := (2 * Real.sqrt 6, 2)

/-- Left focus of the ellipse -/
noncomputable def focus_left : ℝ × ℝ := (-2 * Real.sqrt 6, 0)

/-- Right focus of the ellipse -/
noncomputable def focus_right : ℝ × ℝ := (2 * Real.sqrt 6, 0)

/-- Length of line segment AB -/
def AB_length : ℝ := 6

/-- Theorem stating the maximum area of triangle AOB -/
theorem max_area_triangle_AOB :
  ∀ A B : ℝ × ℝ,
  ellipse_C A.1 A.2 → ellipse_C B.1 B.2 →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = AB_length^2 →
  ∀ area : ℝ,
  area = (1/2) * abs (A.1 * B.2 - A.2 * B.1) →
  area ≤ 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_AOB_l885_88575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l885_88538

-- Define set A
def A : Set ℝ := {x | x > 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = 2^x + 1}

-- Theorem statement
theorem set_intersection_theorem :
  A ∩ B = Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l885_88538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_liquid_poured_l885_88551

/-- Represents a vessel with a given capacity and alcohol percentage -/
structure Vessel where
  capacity : ℚ
  alcoholPercentage : ℚ

/-- Calculates the amount of alcohol in a vessel -/
def alcoholAmount (v : Vessel) : ℚ :=
  v.capacity * v.alcoholPercentage / 100

theorem total_liquid_poured (vessel1 vessel2 newVessel : Vessel)
    (h1 : vessel1.capacity = 2)
    (h2 : vessel1.alcoholPercentage = 30)
    (h3 : vessel2.capacity = 6)
    (h4 : vessel2.alcoholPercentage = 40)
    (h5 : newVessel.capacity = 10)
    (h6 : newVessel.alcoholPercentage = 30)
    (h7 : alcoholAmount vessel1 + alcoholAmount vessel2 = alcoholAmount newVessel) :
    vessel1.capacity + vessel2.capacity = 8 := by
  sorry

#eval toString (2 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_liquid_poured_l885_88551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l885_88542

/-- The coefficient of x^2 in the parabola equation -/
noncomputable def a : ℝ := 12

/-- The vertical shift of the parabola -/
noncomputable def v : ℝ := 5

/-- The equation of the parabola -/
def parabola_eq (x y : ℝ) : Prop := y = a * x^2 + v

/-- The y-coordinate of the directrix -/
noncomputable def directrix_y : ℝ := 239 / 48

/-- Theorem stating the relationship between the parabola and its directrix -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_eq x y → 
  ∃ d : ℝ, d = directrix_y ∧ 
  (∀ p : ℝ × ℝ, p.2 = a * p.1^2 + v → 
   (p.1 - x)^2 + (p.2 - y)^2 = (y - d)^2) :=
by
  sorry

#check parabola_directrix

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l885_88542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangles_equidecomposable_triangles_equidecomposable_l885_88510

-- Define a structure for rectangles
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define a structure for triangles
structure Triangle where
  base : ℝ
  height : ℝ

-- Define area for rectangles
noncomputable def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

-- Define area for triangles
noncomputable def Triangle.area (t : Triangle) : ℝ := (1/2) * t.base * t.height

-- Define equidecomposability
def Equidecomposable {α : Type} (X Y : Set α) : Prop := sorry

-- Theorem for rectangles
theorem rectangles_equidecomposable (r1 r2 : Rectangle) :
  r1.area = r2.area → Equidecomposable (Set.Icc 0 r1.width) (Set.Icc 0 r2.width) := by sorry

-- Theorem for triangles
theorem triangles_equidecomposable (t1 t2 : Triangle) :
  t1.area = t2.area → Equidecomposable (Set.Icc 0 t1.base) (Set.Icc 0 t2.base) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangles_equidecomposable_triangles_equidecomposable_l885_88510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_40_l885_88594

-- Define the circle
def circleRadius : ℝ := 10

-- Define the right triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop := sorry
def hypotenuse_is_chord (t : Triangle) (r : ℝ) : Prop := sorry
def C_on_diameter (t : Triangle) : Prop := sorry
def angle_A_is_75 (t : Triangle) : Prop := sorry

-- Define the area of the triangle
noncomputable def area (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_area_is_40 (t : Triangle) :
  is_right_triangle t →
  hypotenuse_is_chord t circleRadius →
  C_on_diameter t →
  angle_A_is_75 t →
  area t = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_40_l885_88594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l885_88598

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem max_omega_value (ω : ℝ) (φ : ℝ) :
  ω > 0 →
  |φ| ≤ π / 2 →
  (∀ x, f ω φ (x - π/4) = -f ω φ (-x - π/4)) →
  (∃ k : ℤ, ω * π/4 + φ = k * π + π/2) →
  (∀ x ∈ Set.Ioo (5*π/18) (2*π/5), 
    (∀ y ∈ Set.Ioo (5*π/18) (2*π/5), x < y → f ω φ x < f ω φ y) ∨
    (∀ y ∈ Set.Ioo (5*π/18) (2*π/5), x < y → f ω φ x > f ω φ y)) →
  ω ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l885_88598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_2019th_term_l885_88588

/-- Represents the sequence where even numbers 2n appear once and odd numbers 2n+1 appear 2n+1 times -/
def special_sequence : ℕ → ℕ := sorry

/-- Returns the number of times a given natural number appears in the sequence -/
def appearance_count (n : ℕ) : ℕ :=
  if n % 2 = 0 then 1 else n

/-- Returns the cumulative sum of appearances up to a given index -/
def cumulative_sum (n : ℕ) : ℕ :=
  (n / 2) * (n / 2 + 2)

/-- The 2019th term of the special sequence is 87 -/
theorem special_sequence_2019th_term :
  ∃ (k : ℕ), cumulative_sum k ≤ 2019 ∧
             cumulative_sum (k + 1) > 2019 ∧
             special_sequence 2019 = 2 * k + 1 := by
  sorry

#check special_sequence_2019th_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_2019th_term_l885_88588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_players_count_l885_88512

theorem tournament_players_count :
  ∃ (n : ℕ),
    n > 10 ∧
    (∀ i j : Fin n, i ≠ j → ∃ (result : ℚ), result ∈ ({0, 1/2, 1} : Set ℚ)) ∧
    (∀ i : Fin n, ∃ (points : ℚ), 2 * (points / 2) = points) ∧
    (∃ (low_ten : Finset (Fin n)), low_ten.card = 10 ∧ 
      (∀ i ∈ low_ten, ∃ (points : ℚ), 2 * (points / 2) = points ∧
      points / 2 = (low_ten.card - 1) / 2)) ∧
    n = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_players_count_l885_88512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_classification_l885_88547

/-- Definition of the complex number z in terms of m -/
noncomputable def z (m : ℝ) : ℂ := (m^2 - m - 6) / (m + 3) + (m^2 - 2*m - 15) * Complex.I

/-- Theorem stating the conditions for z to be real, imaginary, or purely imaginary -/
theorem z_classification (m : ℝ) :
  (z m ∈ Set.range (Complex.ofReal) ↔ m = 5) ∧
  (z m ∈ {w : ℂ | w.re = 0 ∧ w ≠ 0} ↔ m ≠ 5 ∧ m ≠ -3) ∧
  (z m ∈ {w : ℂ | w.re = 0 ∧ w.im ≠ 0} ↔ m = 3 ∨ m = -2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_classification_l885_88547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_pi_over_two_l885_88518

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  3 * x^2 + 3 * y^2 - 15 * x + 9 * y + 27 = 0

/-- The area of the circle -/
noncomputable def circle_area : ℝ := Real.pi / 2

theorem circle_area_is_pi_over_two :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y : ℝ, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    circle_area = Real.pi * radius^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_pi_over_two_l885_88518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_expressions_l885_88573

theorem order_of_expressions : ∀ (x : ℝ), x = 1/8 → (Real.exp x > Real.sin x ∧ Real.sin x > Real.log (1/8)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_expressions_l885_88573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_range_l885_88548

open Real

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  domain : Set ℝ := {x | x > 0}
  pos : ∀ x ∈ domain, f x > 0
  diff : DifferentiableOn ℝ f domain
  ineq : ∀ x ∈ domain, 2 * f x < x * (deriv f x) ∧ x * (deriv f x) < 3 * f x

theorem special_function_range (F : SpecialFunction) :
  27/64 < F.f 3 / F.f 4 ∧ F.f 3 / F.f 4 < 9/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_range_l885_88548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l885_88560

-- Define the functions f and g
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (2 * x + φ)
noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f φ (x - Real.pi/6)

-- State the theorem
theorem phi_value (φ : ℝ) (h1 : 0 < φ ∧ φ < Real.pi) 
  (h2 : ∀ x, g φ (|x|) = g φ x) : φ = 5*Real.pi/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l885_88560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l885_88531

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given triangle satisfies the problem conditions -/
def problem_triangle : Triangle where
  A := Real.arcsin (1/2)
  B := 2 * Real.pi / 3
  C := Real.pi / 6
  a := 2
  b := 2 * Real.sqrt 3
  c := 2

theorem problem_solution (t : Triangle) 
  (h1 : t.B = 2 * Real.pi / 3)
  (h2 : t.a = 2)
  (h3 : t.b = 2 * Real.sqrt 3) :
  t.C = Real.pi / 6 ∧ 
  0 < Real.sin t.A * Real.sin t.C ∧ 
  Real.sin t.A * Real.sin t.C ≤ 1/4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l885_88531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_circle_area_l885_88511

theorem quarter_circle_area (f : ℝ → ℝ) (h : ∀ x, f x = Real.sqrt (1 - x^2)) :
  ∫ x in Set.Icc 0 1, f x = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_circle_area_l885_88511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l885_88539

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := -6/x - 5 * Real.log x

-- State the theorem
theorem f_monotone_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 6/5 → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l885_88539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_and_perpendicular_line_parallel_to_x_axis_distance_l885_88561

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (l1 l2 : Line) : ℝ :=
  abs (l1.c - l2.c) / Real.sqrt (l1.a^2 + l1.b^2)

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

theorem min_distance_and_perpendicular_line 
  (l1 : Line) 
  (l2 : Line) 
  (P : Point) :
  l1.a = 3 ∧ l1.b = 4 ∧ l1.c = -7 →
  l2.a = 3 ∧ l2.b = 4 ∧ l2.c = 8 →
  P.x = 2 ∧ P.y = 3 →
  (∃ (l : Line),
    l.a = 3 ∧ l.b = 4 ∧ l.c = -18 ∧
    distance_between_parallel_lines l1 l2 = 3 ∧
    (∀ (l' : Line),
      l'.a * P.x + l'.b * P.y + l'.c = 0 →
      distance_between_parallel_lines l1 l2 ≤ 
        abs (l'.a * P.x + l'.b * P.y + l'.c) / Real.sqrt (l'.a^2 + l'.b^2))) :=
by sorry

theorem parallel_to_x_axis_distance
  (l1 : Line)
  (l2 : Line) :
  l1.a = 3 ∧ l1.b = 4 ∧ l1.c = -7 →
  l2.a = 3 ∧ l2.b = 4 ∧ l2.c = 8 →
  (∃ (x1 x2 : ℝ),
    3 * x1 + 4 * 3 - 7 = 0 ∧
    3 * x2 + 4 * 3 + 8 = 0 ∧
    abs (x2 - x1) = 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_and_perpendicular_line_parallel_to_x_axis_distance_l885_88561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l885_88505

theorem problem_statement (a b : ℕ) (h1 : a > b) (h2 : b > 0) 
  (h3 : Nat.Coprime a b) 
  (h4 : (a^3 - b^3 : ℚ) / ((a - b)^3 : ℚ) = 50/3) : 
  a - b = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l885_88505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_third_l885_88527

theorem cos_alpha_minus_pi_third (α : Real) 
  (h1 : Real.cos α = -3/5) 
  (h2 : α ∈ Set.Ioo (Real.pi/2) Real.pi) : 
  Real.cos (α - Real.pi/3) = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_third_l885_88527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_ac_not_five_l885_88529

-- Define the Triangle structure
structure Triangle where
  sides : Finset ℝ
  valid : sides.card = 3

-- Triangle inequality theorem
theorem triangle_inequality {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b > c ∧ b + c > a ∧ a + c > b ↔ ∃ (t : Triangle), t.sides = {a, b, c} := by
  sorry

-- Main theorem
theorem ac_not_five (ABC : Triangle) 
  (h1 : 3 ∈ ABC.sides) (h2 : 2 ∈ ABC.sides) : 5 ∉ ABC.sides := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_ac_not_five_l885_88529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_proof_l885_88504

/-- Curve C1 in polar coordinates -/
def C1 (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 1

/-- Curve C2 in polar coordinates -/
def C2 (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

/-- The distance between intersection points of C1 and C2 -/
noncomputable def intersection_distance : ℝ := 2 * Real.sqrt 3

theorem intersection_distance_proof :
  ∃ (A B : ℝ × ℝ),
    (∃ (ρ₁ θ₁ : ℝ), C1 ρ₁ θ₁ ∧ A = (ρ₁ * Real.cos θ₁, ρ₁ * Real.sin θ₁)) ∧
    (∃ (ρ₂ θ₂ : ℝ), C2 ρ₂ θ₂ ∧ A = (ρ₂ * Real.cos θ₂, ρ₂ * Real.sin θ₂)) ∧
    (∃ (ρ₃ θ₃ : ℝ), C1 ρ₃ θ₃ ∧ B = (ρ₃ * Real.cos θ₃, ρ₃ * Real.sin θ₃)) ∧
    (∃ (ρ₄ θ₄ : ℝ), C2 ρ₄ θ₄ ∧ B = (ρ₄ * Real.cos θ₄, ρ₄ * Real.sin θ₄)) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = intersection_distance :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_proof_l885_88504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_a_equals_one_l885_88508

theorem intersection_implies_a_equals_one :
  let A : Set ℝ := {-1, 1, 2}
  let B : Set ℝ := {x | ∃ a : ℝ, x = a + 1 ∨ x = a^2 + 3}
  ∀ a : ℝ, (A ∩ B = {2}) → a = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_a_equals_one_l885_88508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_eight_four_l885_88563

/-- Definition of the diamond operation -/
noncomputable def diamond (a b c : ℝ) : ℝ := a + a / b - c

/-- Theorem stating that 8 ⋄ 4 = 7 when c = 3 -/
theorem diamond_eight_four : diamond 8 4 3 = 7 := by
  -- Unfold the definition of diamond
  unfold diamond
  -- Simplify the expression
  simp [add_sub_assoc]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_eight_four_l885_88563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_marked_points_with_distance_l885_88522

/-- Represents a marked point on the line segment -/
structure MarkedPoint where
  position : ℚ
  deriving Repr

/-- Represents the division process on a line segment -/
def divide_segment (start : ℚ) (length : ℚ) : List MarkedPoint :=
  sorry

/-- Generates all marked points on a line segment of length 3^n -/
def generate_marked_points (n : ℕ) : List MarkedPoint :=
  sorry

/-- Theorem: For any integer k between 1 and 3^n, there exists a pair of marked points with distance k -/
theorem exists_marked_points_with_distance (n : ℕ) (k : ℕ) 
    (h1 : 1 ≤ k) (h2 : k ≤ 3^n) :
  ∃ (p q : MarkedPoint), p ∈ generate_marked_points n ∧ 
                         q ∈ generate_marked_points n ∧ 
                         |p.position - q.position| = k := by
  sorry

#check exists_marked_points_with_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_marked_points_with_distance_l885_88522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_l885_88587

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ
  area : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def trapezium_area (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

/-- Theorem: Given a trapezium with one side 22 cm, height 15 cm, 
    and area 300 sq cm, the other side is 18 cm -/
theorem trapezium_other_side (t : Trapezium) 
    (h1 : t.side1 = 22)
    (h2 : t.height = 15)
    (h3 : t.area = 300)
    (h4 : t.area = trapezium_area t) :
    t.side2 = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_l885_88587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l885_88585

/-- Given a parabola y² = 2px (p > 0) and a line with slope √3 passing through its focus,
    if the distance from the midpoint of the line's intersection points with the parabola
    to the parabola's axis of symmetry is 4, then p = 3. -/
theorem parabola_line_intersection (p : ℝ) (h₁ : p > 0) : 
  let parabola := fun (x y : ℝ) ↦ y^2 = 2*p*x
  let focus : ℝ × ℝ := (p/2, 0)
  let line_slope := Real.sqrt 3
  let line := fun (x y : ℝ) ↦ y = line_slope * (x - p/2)
  let axis_of_symmetry := fun x ↦ x = -p/2
  ∃ A B : ℝ × ℝ,
    (parabola A.1 A.2 ∧ line A.1 A.2) ∧
    (parabola B.1 B.2 ∧ line B.1 B.2) ∧
    A ≠ B ∧
    (A.1 + B.1)/2 + p/2 = 4 →
    p = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l885_88585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_combinations_l885_88591

theorem library_combinations (n : ℕ) (h : n = 10) : 
  Fintype.card (Fin n → Fin 2) = 2^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_combinations_l885_88591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_range_condition_no_monotonic_increasing_l885_88549

-- Define the function f(x) = log₁/₂(x² - 2ax + 3)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - 2*a*x + 3) / Real.log (1/2)

-- Theorem 1: Domain of f is ℝ iff -√3 < a < √3
theorem domain_condition (a : ℝ) :
  (∀ x, ∃ y, f a x = y) ↔ -Real.sqrt 3 < a ∧ a < Real.sqrt 3 := by
  sorry

-- Theorem 2: Range of f is ℝ iff a ≤ -√3 or a ≥ √3
theorem range_condition (a : ℝ) :
  (∀ y, ∃ x, f a x = y) ↔ a ≤ -Real.sqrt 3 ∨ a ≥ Real.sqrt 3 := by
  sorry

-- Theorem 3: No a exists for f to be monotonically increasing on (-∞, 2)
theorem no_monotonic_increasing (a : ℝ) :
  ¬(∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 2 → f a x₁ < f a x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_range_condition_no_monotonic_increasing_l885_88549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_sum_bounds_l885_88535

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def all_digits_different (abcd efgh : ℕ) : Prop :=
  let digits := [abcd / 1000, (abcd / 100) % 10, (abcd / 10) % 10, abcd % 10,
                 efgh / 1000, (efgh / 100) % 10, (efgh / 10) % 10, efgh % 10]
  (digits.toFinset).card = 8

theorem four_digit_sum_bounds (abcd efgh : ℕ) :
  is_four_digit abcd ∧
  is_four_digit efgh ∧
  all_digits_different abcd efgh ∧
  abcd / 1000 ≠ 0 ∧
  efgh / 1000 ≠ 0 ∧
  abcd - efgh = 1994 →
  (abcd + efgh ≤ 15000 ∧ 4998 ≤ abcd + efgh) :=
by sorry

#check four_digit_sum_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_sum_bounds_l885_88535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_example_l885_88543

/-- The time (in seconds) it takes for a train to cross an electric pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (5 / 18)
  train_length / train_speed_ms

/-- Theorem stating that a train of length 120 meters traveling at 180 km/h takes 2.4 seconds to cross an electric pole -/
theorem train_crossing_time_example :
  train_crossing_time 120 180 = 2.4 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_example_l885_88543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pump_fill_time_l885_88586

/-- The time it takes for the pump to fill the tank without the leak -/
noncomputable def T : ℝ := 2

/-- The time it takes to fill the tank with both pump and leak working -/
noncomputable def fillTimeWithLeak : ℝ := 17/8

/-- The time it takes for the leak to empty the full tank -/
noncomputable def leakEmptyTime : ℝ := 34

/-- Theorem stating the relationship between pump fill time, leak time, and combined fill time -/
theorem pump_fill_time :
  (1 / T - 1 / leakEmptyTime) * fillTimeWithLeak = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pump_fill_time_l885_88586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_problem_l885_88554

theorem log_base_problem (b : ℝ) (h : b > 0) :
  Real.log 729 / Real.log b = -2/3 → b = 1/19683 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_problem_l885_88554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_lower_bound_l885_88595

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Adding case for 0
  | 1 => 2
  | (n + 2) => (sequence_a (n + 1))^2 - sequence_a (n + 1) + 1

theorem sequence_a_lower_bound (n : ℕ) (h : n ≥ 2) :
  sequence_a n ≥ 2^(n-1) + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_lower_bound_l885_88595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l885_88525

/-- The surface area of a sphere circumscribing a right hexagonal prism -/
theorem circumscribed_sphere_surface_area (base_side_length lateral_edge_length : ℝ) 
  (h1 : base_side_length = 2)
  (h2 : lateral_edge_length = 3) :
  4 * π * (5 / 2)^2 = 25 * π := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l885_88525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_l885_88545

/-- Represents the division of money among five individuals -/
structure MoneyDivision where
  x : ℝ
  y : ℝ
  z : ℝ
  w : ℝ
  v : ℝ

/-- The conditions of the money division problem -/
def division_conditions (d : MoneyDivision) : Prop :=
  d.y = 0.45 * d.x ∧
  d.z = 0.50 * d.x ∧
  d.w = 0.70 * d.x ∧
  d.v = 0.25 * d.x ∧
  d.y = 63 ∧
  d.w = 33 * 1.10

/-- The theorem stating the total amount of money -/
theorem total_amount (d : MoneyDivision) 
  (h : division_conditions d) : 
  ∃ ε > 0, |d.x + d.y + d.z + d.w + d.v - 190.05| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_l885_88545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_and_b_for_real_roots_l885_88540

/-- The smallest positive real number such that there exists a positive real number b
    for which all roots of x^4 - ax^3 + bx^2 - ax + 2 are real -/
noncomputable def a : ℝ := 4 * (2 : ℝ)^(1/4)

/-- The positive real number b such that all roots of x^4 - ax^3 + bx^2 - ax + 2 are real -/
noncomputable def b : ℝ := 6 * (2 : ℝ)^(1/2)

/-- The polynomial x^4 - ax^3 + bx^2 - ax + 2 -/
noncomputable def p (x : ℝ) : ℝ := x^4 - a*x^3 + b*x^2 - a*x + 2

theorem smallest_a_and_b_for_real_roots :
  (∀ x : ℝ, (p x = 0) → x ∈ Set.univ) ∧
  (∀ a' : ℝ, 0 < a' ∧ a' < a →
    ¬∃ b' : ℝ, b' > 0 ∧ ∀ x : ℝ, (x^4 - a'*x^3 + b'*x^2 - a'*x + 2 = 0) → x ∈ Set.univ) ∧
  b = 6 * (2 : ℝ)^(1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_and_b_for_real_roots_l885_88540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l885_88536

noncomputable def f (x : ℝ) := (4 * x^2 + 8 * x + 13) / (6 * (1 + Real.exp (-x)))

theorem min_value_of_f (x : ℝ) (h : x ≥ 0) : f x ≥ 13/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l885_88536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l885_88507

-- Define the geometric sequence a_n
noncomputable def a (n : ℕ) (q : ℝ) : ℝ := 2 * q^(n - 1)

-- Define the sum S_n
noncomputable def S (n : ℕ) (q : ℝ) : ℝ := (a 1 q) * (1 - q^n) / (1 - q)

-- State the theorem
theorem geometric_sequence_properties :
  ∃ (q : ℝ), q > 0 ∧ 
  (a 2 q + a 3 q = 12) ∧
  (q = 2) ∧
  (∃ (r : ℝ), ∀ (n : ℕ), S (n + 1) q + 2 = r * (S n q + 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l885_88507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_visits_count_l885_88550

def visit_period (friend : Fin 4) : ℕ :=
  match friend with
  | 0 => 4  -- Alice
  | 1 => 6  -- Beatrix
  | 2 => 8  -- Claire
  | 3 => 10 -- Diana

def is_visiting (friend : Fin 4) (day : ℕ) : Bool :=
  (day + 1) % visit_period friend = 0

def exactly_two_visiting (day : ℕ) : Bool :=
  (Finset.filter (λ f => is_visiting f day) Finset.univ).card = 2

theorem exactly_two_visits_count :
  (Finset.filter (λ d => exactly_two_visiting d) (Finset.range 400)).card = 142 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_visits_count_l885_88550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_diamond_equals_one_l885_88570

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- Define a function to represent the nested diamond operation
noncomputable def nestedDiamond : ℕ → ℝ
  | 0 => 500  -- Base case
  | n + 1 => diamond (501 - n) (nestedDiamond n)

-- Theorem statement
theorem nested_diamond_equals_one : diamond 1 (nestedDiamond 499) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_diamond_equals_one_l885_88570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_minus_2alpha_l885_88555

theorem cos_pi_third_minus_2alpha (α : ℝ) 
  (h : Real.cos (α + π/6) + Real.sin α = 3/5) : 
  Real.cos (π/3 - 2*α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_minus_2alpha_l885_88555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_when_prob_zero_or_one_l885_88524

open MeasureTheory

/-- Two events are independent if the probability of their intersection
    equals the product of their individual probabilities -/
def IndependentEvents {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) (A B : Set Ω) : Prop :=
  μ (A ∩ B) = μ A * μ B

/-- Main theorem: If P(A) is either 0 or 1, then A and B are independent for any event B -/
theorem independence_when_prob_zero_or_one {Ω : Type*} [MeasurableSpace Ω] 
  (μ : Measure Ω) [IsProbabilityMeasure μ] (A B : Set Ω) 
  (h : μ A = 0 ∨ μ A = 1) : IndependentEvents μ A B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_when_prob_zero_or_one_l885_88524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l885_88582

open Real

noncomputable def lg (x : ℝ) : ℝ := log x / log 10

theorem simplify_expression :
  exp (Real.log 2) + (lg 2)^2 + lg 2 * lg 5 + lg 5 = 3 + (lg 2)^2 + lg 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l885_88582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_option_B_is_comprehensive_l885_88514

/-- Represents a survey option -/
inductive SurveyOption
  | A  -- Survey on the quality of food in our city
  | B  -- Survey on the height of classmates in your class
  | C  -- Survey on the viewership of a TV program
  | D  -- Survey on the lifespan of light bulbs produced by a light bulb factory

/-- Defines what makes a survey comprehensive -/
def isComprehensive (s : SurveyOption) : Prop :=
  ∃ (population : Set ℕ), 
    (Set.Finite population) ∧ 
    (∀ x ∈ population, ∃ (measurement : ℕ → ℝ), measurement x ≠ 0)

/-- Theorem stating that option B is suitable for a comprehensive survey -/
theorem option_B_is_comprehensive : isComprehensive SurveyOption.B := by
  sorry

#check option_B_is_comprehensive

end NUMINAMATH_CALUDE_ERRORFEEDBACK_option_B_is_comprehensive_l885_88514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_approx_l885_88597

/-- The angle at the center of a circular sector, given its radius and area -/
noncomputable def sectorAngle (radius : ℝ) (area : ℝ) : ℝ :=
  (area * 360) / (Real.pi * radius^2)

/-- Theorem: The angle at the center of a circular sector with radius 12 meters
    and area 67.88571428571429 square meters is approximately 54 degrees -/
theorem sector_angle_approx :
  let r : ℝ := 12
  let a : ℝ := 67.88571428571429
  abs (sectorAngle r a - 54) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_approx_l885_88597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l885_88502

noncomputable def f (x : ℝ) : ℝ := (1 : ℝ) / (4 ^ x) - (1 : ℝ) / (2 ^ x) + 1

theorem f_min_max :
  ∀ x ∈ Set.Icc (-3 : ℝ) 2,
    (∀ y ∈ Set.Icc (-3 : ℝ) 2, f y ≥ (3 : ℝ) / 4) ∧
    (∃ y ∈ Set.Icc (-3 : ℝ) 2, f y = (3 : ℝ) / 4) ∧
    (∀ y ∈ Set.Icc (-3 : ℝ) 2, f y ≤ 57) ∧
    (∃ y ∈ Set.Icc (-3 : ℝ) 2, f y = 57) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l885_88502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_geq_two_l885_88533

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x ∈ ({a + 1/b, b + 1/c, c + 1/a} : Set ℝ), x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_geq_two_l885_88533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_set_when_m_neg_one_f_range_of_m_when_interval_in_solution_set_l885_88577

/-- The function f(x) = |x+m| + |2x-1| -/
def f (m : ℝ) (x : ℝ) : ℝ := |x + m| + |2 * x - 1|

theorem f_solution_set_when_m_neg_one :
  {x : ℝ | f (-1) x ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by
  sorry

theorem f_range_of_m_when_interval_in_solution_set :
  {m : ℝ | ∀ x ∈ Set.Icc 1 2, f m x ≤ |2 * x + 1|} = Set.Icc (-3) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_set_when_m_neg_one_f_range_of_m_when_interval_in_solution_set_l885_88577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l885_88558

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 + 3/x

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), m = 4 ∧ ∀ (x : ℝ), x > 0 → f x ≥ m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l885_88558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_coordinate_product_l885_88537

/-- The product of all coordinates of all intersection points of two specific circles is 16 -/
theorem intersection_coordinate_product : ∃ (p : ℝ × ℝ), 
  (p.1^2 - 4*p.1 + p.2^2 - 8*p.2 + 16 = 0) ∧ 
  (p.1^2 - 10*p.1 + p.2^2 - 8*p.2 + 52 = 0) ∧
  p.1 * p.2 = 16 := by
  -- Define the circles
  let circle1 := fun (x y : ℝ) ↦ x^2 - 4*x + y^2 - 8*y + 16 = 0
  let circle2 := fun (x y : ℝ) ↦ x^2 - 10*x + y^2 - 8*y + 52 = 0
  
  -- Assert the existence of the intersection point
  use (4, 4)
  
  -- Prove that this point satisfies both circle equations and has the correct product
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_coordinate_product_l885_88537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_P_in_B_l885_88580

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | |p.1| + |p.2| ≤ 2}
def B : Set (ℝ × ℝ) := {p ∈ A | p.2 ≤ p.1^2}

-- Define the measure (area) of a set
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem probability_P_in_B : 
  area B / area A = 17 / 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_P_in_B_l885_88580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_product_of_special_lines_l885_88567

/-- The angle a line makes with the horizontal axis -/
noncomputable def angle_with_horizontal (slope : ℝ) : ℝ := Real.arctan slope

theorem slope_product_of_special_lines (m n : ℝ) :
  m ≠ 0 →  -- L1 is not horizontal
  angle_with_horizontal m = 3 * angle_with_horizontal n →  -- L1 makes three times the angle with the horizontal as L2
  m = 9 * n →  -- The slope of L1 is 9 times the slope of L2
  m * n = 9 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_product_of_special_lines_l885_88567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_handshakes_l885_88501

theorem max_handshakes (N : ℕ) (h1 : N > 5) : 
  ∃ (S : Finset (Fin N)) (h2 : S.card = N - 2),
    (∀ i j : Fin N, i ∈ S → j ∈ S → i ≠ j → ∃ (H : Fin N → Fin N → Prop), H i j) ∧
    (∃ a b : Fin N, a ∉ S ∧ b ∉ S ∧ a ≠ b ∧ 
      (∃ x : Fin N, ¬(∃ (H : Fin N → Fin N → Prop), H a x)) ∧
      (∃ y : Fin N, ¬(∃ (H : Fin N → Fin N → Prop), H b y))) ∧
    (∀ T : Finset (Fin N), T.card > N - 2 → 
      ∃ i j : Fin N, i ∈ T ∧ j ∈ T ∧ i ≠ j ∧ ¬(∃ (H : Fin N → Fin N → Prop), H i j)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_handshakes_l885_88501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_plus_3sin_l885_88584

theorem max_cos_plus_3sin : 
  ∃ (M : ℝ), M = Real.sqrt 10 ∧ ∀ y : ℝ, Real.cos y + 3 * Real.sin y ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_plus_3sin_l885_88584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l885_88599

def M : Set ℤ := {-2, -1, 0, 1, 2}

def N : Set ℤ := {x : ℤ | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l885_88599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_angles_l885_88500

theorem cos_sum_angles (α β : ℝ) 
  (h1 : α - β = π / 6)
  (h2 : Real.tan α - Real.tan β = 3) : 
  Real.cos (α + β) = 1/3 - Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_angles_l885_88500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_and_proper_subset_l885_88544

def A (m : ℝ) : Set ℝ := {x | x^2 - (m+1)*x + m = 0}
def B (m : ℝ) : Set ℝ := {x | x*m - 1 = 0}

theorem subset_and_proper_subset (m : ℝ) :
  (A m ⊆ B m ↔ m = 1) ∧
  (B m ⊂ A m ↔ m = 0 ∨ m = -1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_and_proper_subset_l885_88544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_is_12_l885_88565

/-- A rectangular park with given properties --/
structure Park where
  length : ℝ
  breadth : ℝ
  area : ℝ
  cyclingTime : ℝ
  lengthBreathRatio : length / breadth = 1 / 4
  areaConstraint : length * breadth = area
  areaValue : area = 102400
  timeInMinutes : cyclingTime = 8

/-- The speed of a cyclist around the park --/
noncomputable def cyclistSpeed (p : Park) : ℝ :=
  let perimeter := 2 * (p.length + p.breadth)
  let distanceKm := perimeter / 1000
  let timeHours := p.cyclingTime / 60
  distanceKm / timeHours

/-- Theorem stating the cyclist's speed is approximately 12 km/hr --/
theorem cyclist_speed_is_12 (p : Park) : 
  ∃ ε > 0, |cyclistSpeed p - 12| < ε := by
  sorry

#eval "Lean code compiled successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_is_12_l885_88565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_z_l885_88562

/-- Given a triangle XYZ with sin X = 4/5 and cos Y = 3/5, prove that cos Z = 7/25 -/
theorem triangle_cosine_z (X Y Z : ℝ) : 
  X + Y + Z = Real.pi →
  Real.sin X = 4/5 →
  Real.cos Y = 3/5 →
  Real.cos Z = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_z_l885_88562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_locus_l885_88596

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a given line -/
def Point2D.lies_on (P : Point2D) (l : Line2D) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

/-- Creates a line passing through two points -/
noncomputable def line_through (P Q : Point2D) : Line2D :=
  sorry

/-- Creates a perpendicular line to a given line passing through a point -/
noncomputable def perpendicular_line (l : Line2D) (P : Point2D) : Line2D :=
  sorry

/-- Given a point O and a line l in a 2D plane, this theorem describes the set of points
    traced by the perpendiculars to the line XO erected from the point X, where X moves along line l. -/
theorem perpendicular_locus (O : Point2D) (l : Line2D) :
  ∃ (S : Set Point2D), ∀ P : Point2D, P ∈ S ↔
    (∃ X : Point2D, X.lies_on l ∧ P.lies_on (perpendicular_line (line_through X O) X)) ∧
    P.y^2 ≥ 4 * P.x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_locus_l885_88596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_transfer_proof_l885_88515

noncomputable def milk_transfer (initial_quantity : ℝ) (percentage_less : ℝ) : ℝ :=
  let container_a := initial_quantity
  let container_b := container_a * (1 - percentage_less)
  let container_c := container_a - container_b
  (container_c - container_b) / 2

theorem milk_transfer_proof (initial_quantity : ℝ) (percentage_less : ℝ) 
  (h1 : initial_quantity = 1200)
  (h2 : percentage_less = 0.625) :
  milk_transfer initial_quantity percentage_less = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_transfer_proof_l885_88515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_a_l885_88559

-- Define the function f
noncomputable def f (x : Real) : Real :=
  2 * (Real.sin x * (Real.sqrt 3 * Real.cos x) + Real.cos x * Real.cos x)

-- Define the triangle ABC
structure Triangle :=
  (a b c : Real)
  (A B C : Real)

theorem triangle_side_a (ABC : Triangle) :
  f ABC.A = 2 →
  ABC.b = 1 →
  (1/2) * ABC.b * ABC.c * Real.sin ABC.A = Real.sqrt 3 / 2 →
  ABC.a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_a_l885_88559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_when_b_zero_axis_of_symmetry_f_analytical_expression_l885_88589

-- Define the function g
noncomputable def g (a b c x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x + c

-- Theorem 1: Range of g when b = 0
theorem g_range_when_b_zero (a c : ℝ) :
  (∀ x, ∃ y, g a 0 c x = y) →
  (a = 0 → Set.range (g a 0 c) = {c}) ∧
  (a ≠ 0 → Set.range (g a 0 c) = Set.Icc (c - |a|) (c + |a|)) := by
  sorry

-- Theorem 2: Axis of symmetry
theorem axis_of_symmetry (b : ℝ) :
  (∀ x, g 1 b 0 (x + 5 * Real.pi / 3) = g 1 b 0 (5 * Real.pi / 3 - x)) →
  (∃ k : ℤ, ∀ x, b * Real.sin x + Real.cos x = b * Real.sin (2 * (k * Real.pi - Real.pi / 6) - x) + Real.cos (2 * (k * Real.pi - Real.pi / 6) - x)) := by
  sorry

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi / 3 * x) + 3

-- Theorem 3: Analytical expression of f
theorem f_analytical_expression :
  (∃ a b c : ℝ, ∀ x, g a b c ((Real.pi / 3) * x + Real.pi / 6) = f (x + 1)) ∧
  (∀ n : ℕ, n ≥ 2 → ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ = 3 ∧ f x₂ = 3 ∧ x₂ - x₁ = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_when_b_zero_axis_of_symmetry_f_analytical_expression_l885_88589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C₁_to_C₂_l885_88513

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

def C₂ (x y : ℝ) : Prop := Real.sqrt 3 * x - y + Real.sqrt 3 = 0

-- Define the distance function between a point and a line
noncomputable def dist_point_line (px py : ℝ) (a b c : ℝ) : ℝ :=
  (abs (a * px + b * py + c)) / Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem max_distance_C₁_to_C₂ :
  ∃ (max_dist : ℝ), max_dist = (Real.sqrt 3 + 1) / 2 ∧
  ∀ (px py : ℝ), C₁ px py →
    dist_point_line px py (Real.sqrt 3) (-1) (Real.sqrt 3) ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C₁_to_C₂_l885_88513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_length_l885_88574

/-- Given an arithmetic sequence starting at 3.5 and ending at 63.5, 
    prove that it contains exactly 13 terms. -/
theorem arithmetic_sequence_length : 
  ∀ (a : List ℝ), 
    (a.length > 0) →
    (a.head? = some 3.5) →
    (a.getLast? = some 63.5) →
    (∀ i j, 0 ≤ i ∧ i < j ∧ j < a.length → a[j]! - a[i]! = (j - i) * 5) →
    a.length = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_length_l885_88574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_is_3_l885_88592

/-- The speed of a man rowing in still water (in km/h) -/
def rowing_speed : ℚ := 15

/-- The time taken to cover the downstream distance (in seconds) -/
def downstream_time : ℚ := 23998080153587715 / 1000000000000000

/-- The downstream distance covered (in meters) -/
def downstream_distance : ℚ := 120

/-- Calculates the speed of the current given the rowing speed in still water,
    the time taken to cover a certain distance downstream, and that distance -/
noncomputable def current_speed (rowing_speed : ℚ) (downstream_time : ℚ) (downstream_distance : ℚ) : ℚ :=
  (downstream_distance / downstream_time - rowing_speed * 1000 / 3600) * 3600 / 1000

/-- Theorem stating that under the given conditions, the speed of the current is 3 km/h -/
theorem current_speed_is_3 : 
  current_speed rowing_speed downstream_time downstream_distance = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_is_3_l885_88592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotonicity_tangent_line_equation_l885_88523

noncomputable section

-- Define the function f(x) = x ln(x)
def f (x : ℝ) : ℝ := x * Real.log x

-- Define the function g(x) = f(x) - a(x - 1)
def g (a : ℝ) (x : ℝ) : ℝ := f x - a * (x - 1)

-- Statement for the monotonicity of g(x)
theorem g_monotonicity (a : ℝ) :
  (∀ x ∈ Set.Ioo (0 : ℝ) (Real.exp (a - 1)), StrictMonoOn (fun x => -g a x) (Set.Ioo (0 : ℝ) (Real.exp (a - 1)))) ∧
  (∀ x ∈ Set.Ioi (Real.exp (a - 1)), StrictMonoOn (g a) (Set.Ioi (Real.exp (a - 1)))) :=
sorry

-- Statement for the tangent line equation
theorem tangent_line_equation :
  ∃ x₀ : ℝ, x₀ > 0 ∧ 
    (HasDerivAt f (Real.log x₀ + 1) x₀) ∧
    (Real.log x₀ + 1) * (-x₀) = -1 - f x₀ ∧
    (∀ x : ℝ, x - 1 = (Real.log x₀ + 1) * (x - x₀) + f x₀) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotonicity_tangent_line_equation_l885_88523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_satisfies_conditions_l885_88546

def parabola_vertex : ℝ × ℝ := (0, -2)
def parabola_focus : ℝ × ℝ := (0, -1)
def point_P : ℝ × ℝ := (14, 47)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def on_parabola (p v f : ℝ × ℝ) : Prop :=
  distance p f = |p.2 - (v.2 - (f.2 - v.2))|

theorem point_P_satisfies_conditions :
  on_parabola point_P parabola_vertex parabola_focus ∧
  point_P.1 > 0 ∧ point_P.2 > 0 ∧
  distance point_P parabola_focus = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_satisfies_conditions_l885_88546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_expression_l885_88532

-- Define the variables
variable (a b c d : ℤ)
variable (w x y z : ℕ)

-- Define the conditions
axiom int_cond : True  -- This is already implied by the type declaration
axiom prime_cond : Nat.Prime w ∧ Nat.Prime x ∧ Nat.Prime y ∧ Nat.Prime z
axiom order_cond : w < x ∧ x < y ∧ y < z
axiom product_cond : (w^(a.toNat)) * (x^(b.toNat)) * (y^(c.toNat)) * (z^(d.toNat)) = 660

-- Theorem to prove
theorem value_of_expression :
  (a + b) - (c + d) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_expression_l885_88532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_intersection_l885_88556

/-- Two non-congruent isosceles right triangles sharing a vertex always have a point on the line segment connecting their non-shared vertices that forms an isosceles right triangle with the other two vertices -/
theorem isosceles_right_triangle_intersection (A B C D E : ℂ) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ D ≠ E →  -- Distinct points
  (B - A).re * (C - A).re + (B - A).im * (C - A).im = 0 →  -- ABC is right-angled at A
  Complex.abs (B - A) = Complex.abs (C - A) →  -- ABC is isosceles
  (D - A).re * (E - A).re + (D - A).im * (E - A).im = 0 →  -- ADE is right-angled at A
  Complex.abs (D - A) = Complex.abs (E - A) →  -- ADE is isosceles
  Complex.abs (B - A) ≠ Complex.abs (D - A) →  -- ABC and ADE are non-congruent
  ∃ M : ℂ, ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = C + t • (E - C) ∧  -- M is on segment CE
         (B - M).re * (D - M).re + (B - M).im * (D - M).im = 0 ∧  -- BMD is right-angled at M
         Complex.abs (B - M) = Complex.abs (D - M) ∧  -- BMD is isosceles
         M = (C + E) / 2  -- M is the midpoint of CE
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_intersection_l885_88556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weighted_average_problem_l885_88516

/-- Weighted average of a set of numbers -/
noncomputable def weightedAverage (average : ℝ) (weight : ℝ) : ℝ × ℝ := (average, weight)

/-- Calculate the overall weighted average -/
noncomputable def overallWeightedAverage (sets : List (ℝ × ℝ)) : ℝ :=
  (sets.map (fun s => s.1 * s.2)).sum / (sets.map (fun s => s.2)).sum

/-- The problem setup -/
theorem weighted_average_problem :
  let setA := weightedAverage 8.1 5
  let setB := weightedAverage 8.7 4
  let setC := weightedAverage 7.9 2
  let setD := weightedAverage 9.1 1
  let allSets := [setA, setB, setC, setD]
  let overallAvg := overallWeightedAverage allSets
  overallAvg = 8.35 →
  let combinedABD := overallWeightedAverage [setA, setB, setD]
  combinedABD = 8.44 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weighted_average_problem_l885_88516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l885_88534

noncomputable section

/-- The function f(x) = sin(ωx - π/6) + 1/2 --/
def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 6) + 1 / 2

/-- The theorem stating that ω = 2/3 given the conditions --/
theorem omega_value (ω α β : ℝ) (h_ω_pos : ω > 0)
  (h_f_α : f ω α = -1/2)
  (h_f_β : f ω β = 1/2)
  (h_min_diff : ∀ (a b : ℝ), f ω a = -1/2 → f ω b = 1/2 → |a - b| ≥ 3*Real.pi/4)
  (h_exists_min : ∃ (a b : ℝ), f ω a = -1/2 ∧ f ω b = 1/2 ∧ |a - b| = 3*Real.pi/4) :
  ω = 2/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l885_88534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l885_88583

-- Define the function p(x)
noncomputable def p (a : ℝ) (x : ℝ) : ℝ := 2 * x * Real.log x + x^2 - a * x + 3

-- State the theorem
theorem max_a_value (a : ℝ) :
  (∀ x > 0, p a x ≥ 0) →
  a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l885_88583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_a_l885_88566

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a/b = (b + √3c)/a and sin C = 2√3 sin B, then tan A = √3/3 -/
theorem triangle_tan_a (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  a / b = (b + Real.sqrt 3 * c) / a →
  Real.sin C = 2 * Real.sqrt 3 * Real.sin B →
  Real.tan A = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_a_l885_88566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_factorization_l885_88593

-- Define variables
variable (x y : ℝ)

-- Define the factorizations
def factorization_A (x : ℝ) : ℝ := (2 - x) * (2 + x) + 3 * x
def factorization_B (x : ℝ) : ℝ := -(x - 4) * (x - 1)
def factorization_C (x : ℝ) : ℝ := (1 - 2 * x)^2
def factorization_D (x y : ℝ) : ℝ := x * y * (x - 1 + x^2)

-- Theorem to prove
theorem correct_factorization (x y : ℝ) :
  (factorization_A x ≠ 4 - x^2 + 3 * x) ∧
  (factorization_B x ≠ -x^2 + 3 * x + 4) ∧
  (factorization_C x ≠ 1 - 4 * x + x^2) ∧
  (factorization_D x y = x^2 * y - x * y + x^3 * y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_factorization_l885_88593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_COB_area_l885_88557

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

theorem triangle_COB_area (x p : ℝ) (hx : x > 0) (hp : 0 < p) (hp_upper : p < 15) :
  triangleArea x p = (1/2) * x * p := by
  -- Unfold the definition of triangleArea
  unfold triangleArea
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_COB_area_l885_88557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_gamma_value_l885_88564

-- Define the point Q
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the angles
noncomputable def alpha' : ℝ := Real.arccos (1/4)
noncomputable def beta' : ℝ := Real.arccos (1/2)
noncomputable def gamma' : ℝ := Real.arccos (Real.sqrt 11 / 4)

-- State the theorem
theorem cos_gamma_value (Q : Point3D) 
  (h1 : Q.x > 0 ∧ Q.y > 0 ∧ Q.z > 0) 
  (h2 : Real.cos alpha' = 1/4)
  (h3 : Real.cos beta' = 1/2) :
  Real.cos gamma' = Real.sqrt 11 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_gamma_value_l885_88564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_sum_of_sums_l885_88517

/-- A type representing the faces of an octahedron -/
inductive OctahedronFace : Type
  | Face1 | Face2 | Face3 | Face4 | Face5 | Face6 | Face7 | Face8
deriving Fintype, Repr

/-- A function representing the numbering of the octahedron faces -/
def octahedronNumbering : OctahedronFace → Fin 8 :=
  sorry

/-- A function that returns the three neighboring faces for a given face -/
def neighboringFaces : OctahedronFace → Finset OctahedronFace :=
  sorry

/-- The theorem stating that the sum of sums is always 144 -/
theorem octahedron_sum_of_sums (numbering : OctahedronFace → Fin 8) :
  (Finset.univ : Finset OctahedronFace).sum (fun face =>
    (numbering face).val + (neighboringFaces face).sum (fun neighbor => (numbering neighbor).val)
  ) = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_sum_of_sums_l885_88517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_ratio_is_one_to_one_l885_88579

/-- Represents the admission data for an exhibition -/
structure ExhibitionData where
  adult_fee : ℕ
  child_fee : ℕ
  total_collected : ℕ
  adult_count : ℕ
  child_count : ℕ

/-- Checks if the given ExhibitionData satisfies the problem conditions -/
def valid_data (data : ExhibitionData) : Prop :=
  data.adult_fee = 30 ∧
  data.child_fee = 15 ∧
  data.total_collected = 2925 ∧
  data.adult_count % 5 = 0 ∧
  data.child_count % 5 = 0 ∧
  data.adult_fee * data.adult_count + data.child_fee * data.child_count = data.total_collected

/-- Calculates the absolute difference between two ratios -/
noncomputable def ratio_difference (a b c d : ℕ) : ℚ :=
  |(a : ℚ) / (b : ℚ) - (c : ℚ) / (d : ℚ)|

/-- Theorem: The ratio of adults to children closest to 1 is 1:1 -/
theorem closest_ratio_is_one_to_one (data : ExhibitionData) 
  (h : valid_data data) : 
  ∀ (other_data : ExhibitionData), 
    valid_data other_data → 
    ratio_difference data.adult_count data.child_count 1 1 ≤ 
    ratio_difference other_data.adult_count other_data.child_count 1 1 := by
  sorry

#check closest_ratio_is_one_to_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_ratio_is_one_to_one_l885_88579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_angle_satisfying_conditions_l885_88571

theorem unique_angle_satisfying_conditions :
  ∃! x : ℝ, 0 ≤ x ∧ x < 2 * Real.pi ∧ Real.sin x = -0.5 ∧ Real.cos x = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_angle_satisfying_conditions_l885_88571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_even_function_l885_88581

open Real

theorem shifted_sine_even_function (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π/2) :
  let f := λ x ↦ 2 * sin (x + φ)
  let g := λ x ↦ f (x + π/3)
  (∀ x, g x = g (-x)) → φ = π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_even_function_l885_88581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_circle_center_l885_88519

noncomputable def circle_center (a b : ℝ) : ℝ × ℝ := (-a/2, -b/2)

def point_on_line (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

theorem line_through_circle_center (m : ℝ) : 
  let center := circle_center (-2) 4
  point_on_line 2 1 m center.1 center.2 → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_circle_center_l885_88519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_y_l885_88590

noncomputable def y (x : ℝ) : ℝ := (x * Real.sin x) ^ (8 * Real.log (x * Real.sin x))

theorem derivative_y (x : ℝ) (h : x ≠ 0) (h' : Real.sin x ≠ 0) :
  deriv y x = (16 * (x * Real.sin x) ^ (8 * Real.log (x * Real.sin x)) * Real.log (x * Real.sin x) * (1 + x * (Real.cos x / Real.sin x))) / x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_y_l885_88590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_inequality_l885_88541

-- Define the function f as noncomputable
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := c^x

-- State the theorem
theorem function_properties_and_inequality (c : ℝ) (h1 : c > 1) :
  (∀ x y : ℝ, f c (x + y) = f c x * f c y) ∧
  (Monotone (f c)) →
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → f c x + 2*x + a ≥ 0) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_inequality_l885_88541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l885_88509

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.arcsin (x^2 - x)

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Icc (- Real.arcsin (1/4)) (π/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l885_88509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_equal_l885_88526

/-- Time taken for a train to pass a platform -/
noncomputable def platform_passing_time : ℝ := 16

/-- Speed of the train in km/hr -/
noncomputable def train_speed : ℝ := 54

/-- Length of the platform in meters -/
noncomputable def platform_length : ℝ := 90.0072

/-- Conversion factor from km/hr to m/s -/
noncomputable def km_hr_to_m_s : ℝ := 1000 / 3600

theorem train_passing_time_equal (man_passing_time : ℝ) :
  man_passing_time = platform_passing_time :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_equal_l885_88526
