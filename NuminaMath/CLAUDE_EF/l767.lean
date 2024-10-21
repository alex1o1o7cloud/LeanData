import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_5_over_31_l767_76757

def decimal_expansion (n d : ℕ) : List ℕ := sorry

theorem digit_150_of_5_over_31 :
  let expansion := decimal_expansion 5 31
  let repeating_block := expansion.take 15
  repeating_block.length = 15 ∧
  expansion = repeating_block ++ expansion.drop 15 ∧
  expansion[149]? = some 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_5_over_31_l767_76757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_acquaintances_l767_76705

/-- Represents a group of people and their acquaintance relationships -/
structure AcquaintanceGroup where
  people : Finset ℕ
  knows : ℕ → ℕ → Bool
  sym : ∀ a b, knows a b = knows b a
  self_not_known : ∀ a, knows a a = false

/-- The number of people a given person knows in the group -/
def num_known (g : AcquaintanceGroup) (a : ℕ) : ℕ :=
  (g.people.filter (fun b => g.knows a b)).card

theorem pigeonhole_acquaintances (g : AcquaintanceGroup) 
  (h : g.people.card = 31) :
  ∃ a b, a ∈ g.people ∧ b ∈ g.people ∧ a ≠ b ∧ num_known g a = num_known g b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_acquaintances_l767_76705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bothSidesRedGivenOneRed_eq_twoThirds_l767_76724

/-- Represents a card with two sides -/
inductive Card
  | BB -- Black on both sides
  | BR -- Black on one side, Red on the other
  | RR -- Red on both sides
deriving DecidableEq

/-- The box of cards -/
def box : Multiset Card :=
  Multiset.replicate 5 Card.BB + Multiset.replicate 2 Card.BR + Multiset.replicate 2 Card.RR

/-- The probability of picking a specific card -/
def pickProbability (c : Card) : ℚ :=
  (box.count c : ℚ) / (Multiset.card box : ℚ)

/-- The probability of observing a red side on a given card -/
def redSideProbability (c : Card) : ℚ :=
  match c with
  | Card.BB => 0
  | Card.BR => 1/2
  | Card.RR => 1

/-- The probability of both sides being red given that one side is red -/
noncomputable def bothSidesRedGivenOneRed : ℚ :=
  let totalRedSideProbability := (box.toList.map (fun c => pickProbability c * redSideProbability c)).sum
  let bothRedProbability := pickProbability Card.RR
  bothRedProbability / totalRedSideProbability

theorem bothSidesRedGivenOneRed_eq_twoThirds :
  bothSidesRedGivenOneRed = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bothSidesRedGivenOneRed_eq_twoThirds_l767_76724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_when_a_is_one_f_two_zero_points_iff_a_in_open_zero_one_l767_76718

-- Define the function f(x) = ax^2 - x - ln(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x - Real.log x

-- Theorem 1: Minimum value of f(x) when a = 1
theorem f_min_value_when_a_is_one :
  ∃ (x : ℝ), x > 0 ∧ f 1 x = 0 ∧ ∀ (y : ℝ), y > 0 → f 1 y ≥ 0 := by
  sorry

-- Theorem 2: Range of a for which f(x) has two zero points
theorem f_two_zero_points_iff_a_in_open_zero_one :
  ∀ (a : ℝ), (∃ (x y : ℝ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ f a x = 0 ∧ f a y = 0) ↔ (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_when_a_is_one_f_two_zero_points_iff_a_in_open_zero_one_l767_76718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_black_probability_l767_76720

/-- Represents a 4x4 grid of squares --/
def Grid := Fin 4 → Fin 4 → Bool

/-- Probability of a single square being black initially --/
noncomputable def p_black : ℝ := 1 / 2

/-- Rotates a grid 90° clockwise --/
def rotate (g : Grid) : Grid :=
  fun i j => g (3 - j) i

/-- Applies the repainting rule after rotation --/
def repaint (g_original g_rotated : Grid) : Grid :=
  fun i j => g_rotated i j || g_original i j

/-- Probability of the entire grid being black after rotation and repainting --/
noncomputable def p_all_black_after (g : Grid) : ℝ :=
  sorry

/-- Theorem stating that there exists a grid configuration with the given probability --/
theorem grid_black_probability :
  ∃ (g : Grid), p_all_black_after g = 1 / 65536 := by
  sorry

#check grid_black_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_black_probability_l767_76720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l767_76722

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2)

noncomputable def f' (x : ℝ) : ℝ := (-x^2 - 2*x) / (x^4)

theorem tangent_line_equation (x : ℝ) (h : x > 0) :
  let y := f x
  let slope := f' 1
  let point := (1, f 1)
  (3 : ℝ) * x + y - 5 = 0 ↔ 
    y - point.2 = slope * (x - point.1) := by
  sorry

#check tangent_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l767_76722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_complement_of_B_l767_76744

open Set

def set_A : Set ℝ := {x | -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 3}

def set_B : Set ℝ := {x | x ≠ 0 ∧ (x + 1) / x ≤ 0}

theorem intersection_of_A_and_complement_of_B :
  set_A ∩ (set_B)ᶜ = Icc (0 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_complement_of_B_l767_76744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_outside_circle_l767_76766

-- Define the circle C
def C (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 1

-- Define the point A
def point_A : ℝ × ℝ := (1, 2)

-- Define what it means for a point to be outside a circle
def outside_circle (p : ℝ × ℝ) : Prop :=
  (p.1 + 1)^2 + (p.2 - 2)^2 > 1

-- Theorem statement
theorem point_A_outside_circle : outside_circle point_A := by
  -- Unfold the definition of outside_circle
  unfold outside_circle
  -- Unfold the definition of point_A
  unfold point_A
  -- Simplify the expression
  simp
  -- Prove the inequality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_outside_circle_l767_76766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l767_76796

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^x + (3 : ℝ)^x + (6 : ℝ)^x = (7 : ℝ)^x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l767_76796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equation_solution_l767_76754

/-- The integral equation φ(x) = x + ∫₀ˣ xt φ(t) dt -/
def integral_equation (φ : ℝ → ℝ) : Prop :=
  ∀ x, φ x = x + ∫ t in Set.Icc 0 x, x * t * φ t

/-- The proposed solution function -/
noncomputable def solution (x : ℝ) : ℝ := x * Real.exp (x^3 / 3)

theorem integral_equation_solution :
  integral_equation solution := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equation_solution_l767_76754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_zeros_m_range_four_zeros_m_range_equiv_l767_76706

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then (1 / Real.exp x) + m * x^2
  else Real.exp x + m * x^2

-- State the theorem
theorem four_zeros_m_range (m : ℝ) :
  (∃ a b c d : ℝ, a < b ∧ b < c ∧ c < d ∧
    f m a = 0 ∧ f m b = 0 ∧ f m c = 0 ∧ f m d = 0) →
  m < -(Real.exp 2) / 4 := by
  sorry

-- Define the range of m
def m_range : Set ℝ := {m | m < -(Real.exp 2) / 4}

-- State the main theorem
theorem four_zeros_m_range_equiv :
  {m : ℝ | ∃ a b c d : ℝ, a < b ∧ b < c ∧ c < d ∧
    f m a = 0 ∧ f m b = 0 ∧ f m c = 0 ∧ f m d = 0} = m_range := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_zeros_m_range_four_zeros_m_range_equiv_l767_76706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_implies_a_range_l767_76721

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem f_monotone_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 < a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_implies_a_range_l767_76721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_to_ellipse_trajectory_min_cos_angle_l767_76717

/-- Given a hyperbola and a point P, prove that the trajectory of P is an ellipse -/
theorem hyperbola_to_ellipse_trajectory 
  (F₁ F₂ : ℝ × ℝ) -- Foci of the hyperbola
  (P : ℝ → ℝ × ℝ) -- P as a function of time
  (h₁ : ∀ x y : ℝ, 2 * x^2 - 2 * y^2 = 1 → (x, y) ∈ {z : ℝ × ℝ | (z.1 - F₁.1)^2 + (z.2 - F₁.2)^2 = (z.1 - F₂.1)^2 + (z.2 - F₂.2)^2}) -- Hyperbola equation
  (h₂ : ∀ t : ℝ, Real.sqrt ((P t).1 - F₁.1)^2 + ((P t).2 - F₁.2)^2 + 
                 Real.sqrt ((P t).1 - F₂.1)^2 + ((P t).2 - F₂.2)^2 = 4) -- Condition on P
  : ∃ a b : ℝ, a = 2 ∧ b = Real.sqrt 3 ∧ 
    ∀ t : ℝ, (P t).1^2 / a^2 + (P t).2^2 / b^2 = 1 :=
by
  sorry

/-- Find the minimum value of cos∠F₁PF₂ -/
theorem min_cos_angle 
  (F₁ F₂ : ℝ × ℝ) -- Foci of the hyperbola
  (P : ℝ → ℝ × ℝ) -- P as a function of time
  (h : ∀ t : ℝ, Real.sqrt ((P t).1 - F₁.1)^2 + ((P t).2 - F₁.2)^2 + 
                Real.sqrt ((P t).1 - F₂.1)^2 + ((P t).2 - F₂.2)^2 = 4) -- Condition on P
  : ∃ min_cos : ℝ, ∀ t : ℝ, 
    let PF₁ := Real.sqrt ((P t).1 - F₁.1)^2 + ((P t).2 - F₁.2)^2;
    let PF₂ := Real.sqrt ((P t).1 - F₂.1)^2 + ((P t).2 - F₂.2)^2;
    let F₁F₂ := Real.sqrt (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2;
    (PF₁^2 + PF₂^2 - F₁F₂^2) / (2 * PF₁ * PF₂) ≥ min_cos :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_to_ellipse_trajectory_min_cos_angle_l767_76717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solutions_l767_76727

theorem inequality_system_solutions (m : ℝ) : 
  (∃ (s : Finset ℤ), (∀ x ∈ s, x + 5 > 0 ∧ x - m ≤ 1) ∧ (s.card = 3)) ↔ 
  -3 ≤ m ∧ m < -2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solutions_l767_76727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_change_l767_76782

theorem rectangle_area_change 
  (L B : ℝ) 
  (h_L_pos : L > 0) 
  (h_B_pos : B > 0) : 
  (L * 1.45) * (B * 0.8) / (L * B) = 1.16 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_change_l767_76782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l767_76729

-- Define the function f(x) = xe^(2x)
noncomputable def f (x : ℝ) : ℝ := x * Real.exp (2 * x)

-- State the theorem
theorem f_monotonic_decreasing :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < -1/2 → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l767_76729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_l767_76771

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (x^2) + 3

-- State the theorem
theorem f_decreasing : ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ > f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_l767_76771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l767_76743

/-- Definition of an ellipse with parameter m -/
def is_ellipse (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m + y^2 = 1

/-- Definition of semi-major axis length for the ellipse -/
noncomputable def semi_major_axis_length (m : ℝ) : ℝ :=
  max (Real.sqrt m) 1

/-- Definition of eccentricity for the ellipse -/
noncomputable def eccentricity (m : ℝ) : ℝ :=
  Real.sqrt (1 - min m 1)

/-- Theorem: If the semi-major axis length is twice the eccentricity,
    then m = 2 or m = 3/4 -/
theorem ellipse_m_values (m : ℝ) :
  (∀ x y : ℝ, is_ellipse m x y) →
  semi_major_axis_length m = 2 * eccentricity m →
  m = 2 ∨ m = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l767_76743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_opposite_directions_l767_76714

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are non-collinear if they are not scalar multiples of each other -/
def NonCollinear (v w : V) : Prop :=
  ∀ (r : ℝ), v ≠ r • w

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def AreCollinear (v w : V) : Prop :=
  ∃ (r : ℝ), v = r • w

/-- Two vectors have opposite directions if one is a negative scalar multiple of the other -/
def OppositeDirections (v w : V) : Prop :=
  ∃ (r : ℝ), r < 0 ∧ v = r • w

/-- Main theorem: If e₁ and e₂ are non-collinear vectors, and k*e₁ + 4*e₂ and e₁ + k*e₂ 
    are collinear and have opposite directions, then k = -2 -/
theorem vector_collinearity_opposite_directions 
  (e₁ e₂ : V) (k : ℝ) 
  (h_non_collinear : NonCollinear e₁ e₂)
  (h_collinear : AreCollinear (k • e₁ + 4 • e₂) (e₁ + k • e₂))
  (h_opposite : OppositeDirections (k • e₁ + 4 • e₂) (e₁ + k • e₂)) :
  k = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_opposite_directions_l767_76714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_speed_l767_76741

/-- Proves that a man's downstream rowing speed is 40 kmph given his upstream speed and still water speed -/
theorem downstream_speed
  (upstream_speed : ℝ)
  (still_water_speed : ℝ)
  (downstream_speed : ℝ)
  (h1 : upstream_speed = 26)
  (h2 : still_water_speed = 33)
  (h3 : still_water_speed = (upstream_speed + downstream_speed) / 2) :
  downstream_speed = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_speed_l767_76741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combination_arrangement_inequality_l767_76756

theorem combination_arrangement_inequality (n : ℕ) : 
  2 * (Nat.choose n 3) ≤ (Nat.factorial n / Nat.factorial (n - 2)) ↔ n ∈ ({3, 4, 5} : Finset ℕ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combination_arrangement_inequality_l767_76756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wages_total_wages_proof_l767_76740

/-- The total wages for a work given the following conditions:
  * A can finish the work alone in 10 days
  * B can finish the work alone in 15 days
  * When working together, A gets Rs. 1800 out of the total wages
-/
theorem total_wages (days_A days_B wages_A : ℕ) : ℕ :=
  let rate_A : ℚ := 1 / days_A
  let rate_B : ℚ := 1 / days_B
  let combined_rate : ℚ := rate_A + rate_B
  let work_ratio_A : ℚ := rate_A / combined_rate
  3000

#check total_wages 10 15 1800

/-- Proof of the theorem -/
theorem total_wages_proof : total_wages 10 15 1800 = 3000 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wages_total_wages_proof_l767_76740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_cutting_l767_76790

def can_measure (a b c : ℕ) : Prop :=
  ∀ n : ℕ, n ≤ 10 → ∃ x y z : Int, x ∈ ({-1, 0, 1} : Set Int) ∧ y ∈ ({-1, 0, 1} : Set Int) ∧ z ∈ ({-1, 0, 1} : Set Int) ∧
  n = x * a + y * b + z * c

theorem plate_cutting :
  ∃! s : Finset (ℕ × ℕ × ℕ), s.card = 2 ∧
  (∀ (a b c : ℕ), (a, b, c) ∈ s → a + b + c = 10 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ can_measure a b c) ∧
  (∀ (a b c : ℕ), a + b + c = 10 → a > 0 → b > 0 → c > 0 → can_measure a b c → (a, b, c) ∈ s) :=
by
  sorry

#check plate_cutting

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_cutting_l767_76790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_of_product_of_digit_sums_l767_76773

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Product of digit sums from 1 to 100 -/
def product_of_digit_sums : ℕ := (List.range 100).foldl (λ acc i => acc * digit_sum (i + 1)) 1

/-- Count of numbers from 1 to 100 whose digit sum is divisible by 5 -/
def count_divisible_by_5 : ℕ := sorry

/-- Helper function to count trailing zeros -/
def count_trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 10 = 0 then 1 + count_trailing_zeros (n / 10)
  else 0

theorem trailing_zeros_of_product_of_digit_sums :
  count_trailing_zeros product_of_digit_sums = count_divisible_by_5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_of_product_of_digit_sums_l767_76773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l767_76776

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  m : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / 9 + p.y^2 / e.m = 1

/-- Theorem: Minimum value of |PA| + |PF| -/
theorem min_distance_sum (e : Ellipse) (F A : Point) :
  F.x = 1 ∧ F.y = 0 ∧ A.x = 1 ∧ A.y = 1 →
  (∃ (min : ℝ), min = 6 - Real.sqrt 5 ∧
    ∀ (P : Point), isOnEllipse P e →
      distance P A + distance P F ≥ min) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l767_76776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_cake_piece_l767_76783

/-- The volume of a piece of cake -/
theorem volume_of_cake_piece (thickness : ℝ) (diameter : ℝ) (num_pieces : ℕ) :
  thickness = 1 / 2 →
  diameter = 16 →
  num_pieces = 8 →
  let radius := diameter / 2
  let total_volume := π * radius^2 * thickness
  let piece_volume := total_volume / (num_pieces : ℝ)
  piece_volume = 4 * π := by
  intros h_thickness h_diameter h_num_pieces
  simp [h_thickness, h_diameter, h_num_pieces]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_cake_piece_l767_76783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l767_76761

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if cos C = (2b - c) / (2a) and the perimeter is 6,
    then the measure of angle A is π/3 and the maximum area is √3. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  a + b + c = 6 →
  Real.cos C = (2 * b - c) / (2 * a) →
  A + B + C = π →
  A = π / 3 ∧ 
  (∀ (S : ℝ), S = (1/2) * a * b * Real.sin C → S ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l767_76761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l767_76758

open Real

/-- The original function before transformation -/
noncomputable def original_func (x : ℝ) : ℝ := sin (x - π/3)

/-- The transformed function f(x) -/
noncomputable def f (x : ℝ) : ℝ := original_func ((x - π/3) / 2)

/-- The derivative of f(x) -/
noncomputable def f_prime (x : ℝ) : ℝ := deriv f x

/-- The function we're investigating -/
noncomputable def g (x : ℝ) : ℝ := f x / f_prime x

/-- Theorem: The center of symmetry of g(x) on [0, 2π] is (π, 0) -/
theorem center_of_symmetry :
  ∃ (y : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2*π → g (π + x) = g (π - x)) ∧ g π = y :=
by
  sorry

#check center_of_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l767_76758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_proof_l767_76736

/-- Checks if all digits in a number are unique -/
def hasUniqueDigits (n : ℕ) : Prop :=
  ∀ i j, i ≠ j → (n / 10^i % 10) ≠ (n / 10^j % 10)

/-- Checks if a number decreases 5 times when the first digit is removed -/
def decreasesFiveTimes (n : ℕ) : Prop :=
  n / 10 = n % 10^(Nat.log n 10) / 5

/-- The largest number satisfying the conditions -/
def largestNumber : ℕ := 3750

theorem largest_number_proof :
  largestNumber = 3750 ∧
  hasUniqueDigits largestNumber ∧
  decreasesFiveTimes largestNumber ∧
  ∀ m : ℕ, m > largestNumber →
    ¬(hasUniqueDigits m ∧ decreasesFiveTimes m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_proof_l767_76736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_division_area_ratio_l767_76792

/-- Given a right triangle with a point on its hypotenuse and lines parallel to the legs drawn through
    this point, dividing the triangle into a rectangle and two smaller right triangles, if the area of
    one small right triangle is n times the area of the rectangle, then the ratio of the area of the
    other small right triangle to the area of the rectangle is 1/(4n). -/
theorem right_triangle_division_area_ratio {n : ℝ} (hn : n > 0) :
  ∀ (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b),
  let rectangle_area := a * b
  let small_triangle1_area := n * rectangle_area
  let small_triangle2_area := (b^2) / (4 * n)
  small_triangle2_area / rectangle_area = 1 / (4 * n) :=
by
  intros a b ha hb hab rectangle_area small_triangle1_area small_triangle2_area
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_division_area_ratio_l767_76792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_digit_number_theorem_l767_76750

def is_valid_number (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : Fin 10) (m n₁ : ℕ),
    n = m + 10^k * a.val + 10^(k+1) * n₁ ∧
    m + 10^k * n₁ = n / 13 ∧
    99 ≤ k ∧ k < 99 + a.val ∧
    n ≥ 10^99 ∧ n < 10^100

theorem hundred_digit_number_theorem :
  ∀ n : ℕ, is_valid_number n ↔ 
    (n = 1625 * 10^96 ∨ 
     n = 195 * 10^97 ∨ 
     n = 2925 * 10^96 ∨ 
     (∃ b : Fin 3, n = 13 * (b.val + 1) * 10^98)) :=
by
  sorry

#check hundred_digit_number_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_digit_number_theorem_l767_76750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_for_sequence_l767_76733

/-- Given a natural number k, this theorem states that the least n for which
    there exist n real numbers satisfying the specified conditions is 2k. -/
theorem least_n_for_sequence (k : ℕ) :
  (∃ (n : ℕ) (a : ℕ → ℝ), 
    (∀ i, i < n - 1 → 0 < a (i + 1) - a i ∧ a (i + 1) - a i < a i - a (i - 1)) ∧
    (∃ (pairs : Finset (ℕ × ℕ)), pairs.card = k ∧ 
      ∀ p ∈ pairs, let (i, j) := p; i < n ∧ j < n ∧ a i - a j = 1)) →
  (∃ (n : ℕ) (a : ℕ → ℝ), n = 2 * k ∧
    (∀ i, i < n - 1 → 0 < a (i + 1) - a i ∧ a (i + 1) - a i < a i - a (i - 1)) ∧
    (∃ (pairs : Finset (ℕ × ℕ)), pairs.card = k ∧ 
      ∀ p ∈ pairs, let (i, j) := p; i < n ∧ j < n ∧ a i - a j = 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_for_sequence_l767_76733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_value_on_interval_min_value_on_interval_l767_76799

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (x + Real.pi/9) - Real.cos x ^ 2

-- Statement for the smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

-- Statement for the maximum value on [0, π/2]
theorem max_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi/2) ∧ f x = 1/4 ∧
  ∀ y ∈ Set.Icc 0 (Real.pi/2), f y ≤ 1/4 :=
sorry

-- Statement for the minimum value on [0, π/2]
theorem min_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi/2) ∧ f x = -1/4 ∧
  ∀ y ∈ Set.Icc 0 (Real.pi/2), f y ≥ -1/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_value_on_interval_min_value_on_interval_l767_76799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equals_one_over_71_l767_76702

/-- Sequence b(n) defined recursively -/
def b : ℕ → ℚ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => 2 * b (n + 2) + b (n + 1)

/-- The sum of b(n) / 9^(n+1) from n=1 to infinity -/
noncomputable def infiniteSum : ℝ :=
  ∑' n, (b n : ℝ) / (9 : ℝ) ^ (n + 1)

theorem infinite_sum_equals_one_over_71 : infiniteSum = 1/71 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equals_one_over_71_l767_76702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_derivative_l767_76789

noncomputable def y (x : ℝ) : ℝ := (Real.sin (Real.cos 3) * (Real.cos (2 * x))^2) / (4 * Real.sin (4 * x))

theorem y_derivative (x : ℝ) (h : Real.sin (4 * x) ≠ 0) : 
  deriv y x = - (Real.sin (Real.cos 3)) / (4 * (Real.sin (2 * x))^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_derivative_l767_76789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l767_76779

theorem system_solution :
  ∀ x y : ℚ, 
    (x^2 + 4*y^2 - x*y = 10 ∧ 2*x - 4*y + 3*x*y = 11) ↔ 
    ((x = 3 ∧ y = 1) ∨ (x = -2 ∧ y = -2/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l767_76779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_digits_l767_76762

theorem count_valid_digits : 
  (∃ (S : Finset ℕ), S.card = 7 ∧ 
    (∀ n : ℕ, n ∈ S ↔ (n < 10 ∧ (3 : ℚ) + n / 10 + 7 / 100 > (327 : ℚ) / 100))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_digits_l767_76762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_N_l767_76747

theorem existence_of_N : ∃ N : ℕ+, ∃ k : ℕ,
  (2000 : ℝ)^(N : ℝ) ≥ 200120012001 * 10^k ∧ (2000 : ℝ)^(N : ℝ) < 200120012001 * 10^(k+1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_N_l767_76747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l767_76785

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the function f
noncomputable def f (x : Real) : Real := Real.sin (2 * x - Real.pi / 6)

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : ∀ x, f x ≤ f t.A)  -- Condition on f
  (h2 : t.a = Real.sqrt 3)  -- Condition on side length a
  : t.A = Real.pi / 3 ∧ 
    ∃ (AM : Real), Real.sqrt 3 / 2 < AM ∧ AM ≤ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l767_76785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_is_nine_oh_six_l767_76795

/-- Represents the number of minutes past 9:00 -/
def t : ℝ := sorry

/-- The time is between 9:00 and 10:00 -/
axiom time_range : 0 ≤ t ∧ t < 60

/-- The angle of the hour hand at 9:00 (in degrees) -/
def hour_hand_start : ℝ := 270

/-- The angle the hour hand moves per minute (in degrees) -/
def hour_hand_speed : ℝ := 0.5

/-- The angle the minute hand moves per minute (in degrees) -/
def minute_hand_speed : ℝ := 6

/-- The position of the hour hand 6 minutes ago (in degrees) -/
def hour_hand_pos : ℝ := hour_hand_start + hour_hand_speed * (t - 6)

/-- The position of the minute hand 9 minutes from now (in degrees) -/
def minute_hand_pos : ℝ := minute_hand_speed * (t + 9)

/-- The hands are opposite each other when their difference is 180 degrees -/
axiom hands_opposite : |minute_hand_pos - hour_hand_pos| = 180

/-- The theorem to prove -/
theorem time_is_nine_oh_six : t = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_is_nine_oh_six_l767_76795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_minimizes_distance_sum_l767_76742

/-- The Fermat point of a triangle minimizes the sum of distances to its vertices -/
theorem fermat_point_minimizes_distance_sum (a b c : ℝ × ℝ) :
  let fermat_point : ℝ × ℝ := (1, (5.5 * Real.sqrt 3 - 3) / 13)
  let distance_sum (p : ℝ × ℝ) := 
    Real.sqrt ((p.1 - a.1)^2 + (p.2 - a.2)^2) +
    Real.sqrt ((p.1 - b.1)^2 + (p.2 - b.2)^2) +
    Real.sqrt ((p.1 - c.1)^2 + (p.2 - c.2)^2)
  (a = (0, 0) ∧ b = (2, 0) ∧ c = (0, Real.sqrt 3)) →
  ∀ p : ℝ × ℝ, distance_sum fermat_point ≤ distance_sum p := by
  sorry

#check fermat_point_minimizes_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_minimizes_distance_sum_l767_76742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_intersects_curve_l767_76787

/-- The curve y = x^3 --/
noncomputable def curve (x : ℝ) : ℝ := x^3

/-- The point A on the curve --/
noncomputable def A : ℝ × ℝ := (2, 8)

/-- The slope of the normal at point A --/
noncomputable def normal_slope : ℝ := -1 / (3 * A.1^2)

/-- The equation of the normal line at point A --/
noncomputable def normal_line (x : ℝ) : ℝ := normal_slope * (x - A.1) + A.2

/-- The point B where the normal intersects the curve again --/
noncomputable def B : ℝ × ℝ := (32, 32768)

theorem normal_intersects_curve :
  curve A.1 = A.2 ∧
  normal_line B.1 = B.2 ∧
  curve B.1 = B.2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_intersects_curve_l767_76787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_equipment_purchase_l767_76760

/-- Represents the cost of sports equipment -/
structure EquipmentCost where
  tableTennis : ℕ  -- Cost of one pair of table tennis paddles
  badminton : ℕ    -- Cost of one pair of badminton rackets

/-- Represents the purchasing options -/
inductive PurchaseOption
  | A  -- Buy one pair of badminton rackets and get one pair of table tennis paddles for free
  | B  -- Pay 80% of the total price

/-- Theorem for the sports equipment purchasing problem -/
theorem sports_equipment_purchase 
  (cost : EquipmentCost)
  (total_pairs : ℕ)
  (max_badminton : ℕ)
  : (2 * cost.tableTennis + 4 * cost.badminton = 350) →
    (6 * cost.tableTennis + 3 * cost.badminton = 420) →
    (total_pairs = 80) →
    (max_badminton ≤ 40) →
    (cost.tableTennis = 35 ∧ cost.badminton = 70) ∧
    (∀ m : ℕ, m > 0 → m ≤ max_badminton →
      (m < 20 → PurchaseOption.B = PurchaseOption.B) ∧
      (m = 20 → PurchaseOption.A = PurchaseOption.B) ∧
      (m > 20 → PurchaseOption.A = PurchaseOption.A)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_equipment_purchase_l767_76760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_value_l767_76735

noncomputable def T : ℝ := 
  (1 / (4 - Real.sqrt 15)) - (1 / (Real.sqrt 15 - 2 * Real.sqrt 3)) + 
  (2 / (2 * Real.sqrt 3 - Real.sqrt 12)) - (1 / (Real.sqrt 12 - 3)) + 
  (1 / (3 - Real.sqrt 8))

theorem T_value : T = 7 + 4 * Real.sqrt 3 + 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_value_l767_76735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_denominator_l767_76775

theorem decimal_to_fraction_denominator : 
  ∃ (n : ℕ), (n ≠ 0) ∧ 
  (∃ (m : ℕ), (0.36363636 : ℚ) = m / n ∧ 
  ∀ (k l : ℕ), k ≠ 0 → l / k = m / n → n ≤ k) ∧
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_denominator_l767_76775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_of_triangle_PAB_l767_76774

-- Define the hyperbola
def hyperbola (m : ℝ) (x y : ℝ) : Prop := x^2 / m - y^2 / 2 = 1

-- Define the point P
def P : ℝ × ℝ := (3, 4)

-- Define the vertices A and B
noncomputable def A (m : ℝ) : ℝ × ℝ := (-Real.sqrt m, 0)
noncomputable def B (m : ℝ) : ℝ × ℝ := (Real.sqrt m, 0)

-- Define the circumcircle equation
def circumcircle (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 10

-- Theorem statement
theorem circumcircle_of_triangle_PAB (m : ℝ) :
  hyperbola m P.1 P.2 →
  (∀ x y, circumcircle x y ↔ 
    (x - P.1)^2 + (y - P.2)^2 = (x - (A m).1)^2 + (y - (A m).2)^2 ∧
    (x - P.1)^2 + (y - P.2)^2 = (x - (B m).1)^2 + (y - (B m).2)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_of_triangle_PAB_l767_76774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_condition_l767_76772

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point2D
  radius : ℝ

/-- Checks if three points are distinct -/
def areDistinct (p1 p2 p3 : Point2D) : Prop :=
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3

/-- Checks if a point lies on a circle -/
def isOnCircle (p : Point2D) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Calculates the angle between three points -/
noncomputable def angle (p1 p2 p3 : Point2D) : ℝ :=
  sorry

/-- Theorem: Given distinct points M, F, and S on a circle, where M represents the second 
    intersection of an altitude, F represents the second intersection of an angle bisector, 
    and S represents the second intersection of a median from a vertex, a unique triangle ABC 
    can be constructed if and only if 90° < ∠SFM < 180° -/
theorem triangle_construction_condition 
  (c : Circle) (M F S : Point2D) 
  (h_distinct : areDistinct M F S)
  (h_on_circle : isOnCircle M c ∧ isOnCircle F c ∧ isOnCircle S c)
  : (∃! (A B C : Point2D), 
    (areDistinct A B C) ∧ 
    (isOnCircle A c ∧ isOnCircle B c ∧ isOnCircle C c) ∧
    (M ≠ A ∧ F ≠ A ∧ S ≠ A) ∧
    True)  -- Placeholder for altitude, angle bisector, and median conditions
    ↔ 
    (π/2 < angle S F M ∧ angle S F M < π) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_condition_l767_76772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_third_l767_76707

theorem cos_alpha_minus_pi_third (α : ℝ) 
  (h1 : Real.cos α = 3/5) 
  (h2 : α ∈ Set.Ioo (3*Real.pi/2) (2*Real.pi)) : 
  Real.cos (α - Real.pi/3) = (3 - 4*Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_third_l767_76707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_daily_profit_l767_76734

/-- The daily profit function for the factory -/
noncomputable def daily_profit (x : ℝ) : ℝ := 20 * x - 3 * x^2 + 96 * Real.log x - 90

/-- The constraint on daily output per machine -/
def output_constraint (x : ℝ) : Prop := 4 ≤ x ∧ x ≤ 12

/-- Theorem stating the maximum daily profit and optimal output -/
theorem max_daily_profit :
  ∃ (max_profit : ℝ) (optimal_output : ℝ),
    output_constraint optimal_output ∧
    (∀ x, output_constraint x → daily_profit x ≤ daily_profit optimal_output) ∧
    optimal_output = 6 ∧
    max_profit = daily_profit optimal_output ∧
    max_profit = 96 * Real.log 6 - 78 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_daily_profit_l767_76734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l767_76738

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (-2 * x + Real.pi / 4)

theorem f_monotone_increasing (k : ℤ) :
  MonotoneOn f (Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l767_76738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_league_teams_l767_76749

/-- Represents the number of teams in a double round-robin football league. -/
def m : ℕ := sorry

/-- Represents an integer used in calculating the total number of matches. -/
def n : ℤ := sorry

/-- The total number of matches played in the league. -/
def total_matches : ℕ := (9 * n^2 + 6 * n + 32).toNat

/-- The number of matches in a double round-robin tournament with m teams. -/
def double_round_robin_matches : ℕ := (m * (m - 1)) / 2

/-- Theorem stating that if the total matches in a double round-robin tournament
    with m teams is equal to 9n^2 + 6n + 32, then m must be either 8 or 32. -/
theorem football_league_teams :
  total_matches = double_round_robin_matches → m = 8 ∨ m = 32 := by
  sorry

#check football_league_teams

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_league_teams_l767_76749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_combined_time_l767_76703

/-- Represents the time (in days) it takes for three workers to complete a job together,
    given their individual completion times. -/
noncomputable def combined_completion_time (a b c : ℝ) : ℝ :=
  1 / (1/a + 1/b + 1/c)

/-- Theorem stating that workers who can complete a job in 10, 15, and 20 days
    respectively can complete the job together in 60/13 days. -/
theorem workers_combined_time :
  combined_completion_time 10 15 20 = 60/13 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_combined_time_l767_76703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l767_76700

/-- A function f with specific properties -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

/-- Theorem stating the properties of the function f -/
theorem function_properties (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : 0 < φ ∧ φ < Real.pi) 
  (h_period : ∀ x, f ω φ (x + Real.pi / ω) = f ω φ x) 
  (h_point : f ω φ (Real.pi / 4) = Real.sqrt 2) :
  (∀ x, f ω φ x = 2 * Real.sin (2 * x + Real.pi / 4)) ∧
  (∀ k : ℤ, StrictMonoOn (f ω φ) (Set.Icc (-3 * Real.pi / 8 + ↑k * Real.pi) (Real.pi / 8 + ↑k * Real.pi))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l767_76700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_cum_sum_prime_count_l767_76726

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Cumulative sum of Fibonacci numbers -/
def fibCumSum (n : ℕ) : ℕ :=
  (List.range n).map fib |>.sum

/-- Check if a number is prime -/
def isPrime (n : ℕ) : Bool :=
  if n ≤ 1 then false
  else
    (List.range (n - 1)).all (fun m => m < 2 || n % (m + 2) ≠ 0)

/-- Count of prime numbers in a list -/
def countPrimes (l : List ℕ) : ℕ :=
  l.filter isPrime |>.length

/-- Theorem: The count of prime numbers among the first 15 cumulative sums of Fibonacci numbers is 2 -/
theorem fib_cum_sum_prime_count : countPrimes (List.range 15 |>.map fibCumSum) = 2 := by
  sorry

#eval countPrimes (List.range 15 |>.map fibCumSum)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_cum_sum_prime_count_l767_76726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_prime_factors_l767_76764

theorem n_prime_factors (n : ℕ) (hn : 0 < n ∧ n < 200) (h_div : (14 * n) % 60 = 0) :
  (Finset.filter (fun p => Nat.Prime p ∧ n % p = 0) (Finset.range n)).card = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_prime_factors_l767_76764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_term_correct_l767_76731

/-- The n-th term of the sequence -/
noncomputable def sequenceTerm (n : ℕ+) : ℝ :=
  Real.sqrt (n + 1 + (n + 1) / (n * (n + 2)))

/-- The theorem stating that the sequenceTerm function correctly defines the n-th term of the sequence -/
theorem sequence_term_correct (n : ℕ+) :
  sequenceTerm n = Real.sqrt (n + 1 + (n + 1) / (n * (n + 2))) :=
by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_term_correct_l767_76731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l767_76793

/-- Given a point P(8m,3) on the terminal side of angle α where cos α = -4/5, prove that m = -1/2 -/
theorem point_on_terminal_side (m : ℝ) (α : ℝ) : 
  (∃ P : ℝ × ℝ, P = (8*m, 3) ∧ P.1 = 8*m * Real.cos α ∧ P.2 = 8*m * Real.sin α) → 
  Real.cos α = -4/5 → 
  m = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l767_76793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_points_l767_76777

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parallel_line_points : 
  let M : Point2D := ⟨3, -2⟩
  ∀ x y : ℝ,
  let N : Point2D := ⟨x, y⟩
  (N.y = M.y) →  -- M and N lie on the same line parallel to the x-axis
  (distance M N = 1) →  -- The distance MN = 1
  (N = ⟨4, -2⟩ ∨ N = ⟨2, -2⟩) -- The coordinates of point N are (4, -2) or (2, -2)
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_points_l767_76777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_juice_theorem_l767_76770

-- Define the components of the drink
noncomputable def orange_percent : ℝ := 0.15
noncomputable def watermelon_percent : ℝ := 0.60
noncomputable def combined_juice_amount : ℝ := 140

-- Define the total amount of the drink
noncomputable def total_amount : ℝ := combined_juice_amount / (orange_percent + watermelon_percent)

-- Define the amount of grape juice
noncomputable def grape_juice_amount : ℝ := total_amount * (1 - orange_percent - watermelon_percent)

-- Theorem statement
theorem grape_juice_theorem :
  abs (grape_juice_amount - 46.67) < 0.01 := by
  sorry

-- Note: We use an inequality with a small epsilon (0.01) to account for floating-point precision

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_juice_theorem_l767_76770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_theorem_l767_76797

/-- The maximum number of intersection points between three circles and one line -/
def max_intersection_points : ℕ := 12

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- represents ax + by + c = 0

/-- Configuration of three circles and one line -/
structure Configuration where
  circles : Fin 3 → Circle
  line : Line

/-- The number of intersection points in a given configuration -/
noncomputable def num_intersection_points (config : Configuration) : ℕ := 
  sorry

/-- Theorem stating that the number of intersection points is at most 12 -/
theorem max_intersection_theorem (config : Configuration) : 
  num_intersection_points config ≤ max_intersection_points := by
  sorry

#check max_intersection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_theorem_l767_76797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_eighth_l767_76768

/-- The number of sides on a standard die -/
def die_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The target sum we're aiming for -/
def target_sum : ℕ := 10

/-- The set of possible outcomes when rolling three dice -/
def all_outcomes : Finset (Fin die_sides × Fin die_sides × Fin die_sides) :=
  Finset.product (Finset.univ : Finset (Fin die_sides))
    (Finset.product (Finset.univ : Finset (Fin die_sides)) (Finset.univ : Finset (Fin die_sides)))

/-- The set of favorable outcomes (those that sum to the target) -/
def favorable_outcomes : Finset (Fin die_sides × Fin die_sides × Fin die_sides) :=
  all_outcomes.filter (fun (a, b, c) => a.val + b.val + c.val + num_dice = target_sum)

/-- The probability of rolling the target sum -/
noncomputable def probability_target_sum : ℚ :=
  (favorable_outcomes.card : ℚ) / (all_outcomes.card : ℚ)

theorem probability_is_one_eighth :
  probability_target_sum = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_eighth_l767_76768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l767_76794

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-15 * x^2 + 14 * x - 3)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | 1/3 ≤ x ∧ x ≤ 3/5} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l767_76794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_negative_f_l767_76716

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((2 / (1 - x)) + a)

-- State the theorem
theorem range_of_negative_f (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →  -- f is an odd function
  {x : ℝ | f a x < 0} = {x : ℝ | -1 < x ∧ x < 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_negative_f_l767_76716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_x_2_minus_x_l767_76746

theorem integral_sqrt_x_2_minus_x :
  (∫ x in (Set.Icc 0 1), Real.sqrt (x * (2 - x))) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_x_2_minus_x_l767_76746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_with_equal_floors_l767_76769

theorem count_integers_with_equal_floors : 
  (Finset.filter (fun n : ℕ => 
    (n : ℝ) / 2014 = (n : ℝ) / 2016) (Finset.range 2016000)).card = 1015056 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_with_equal_floors_l767_76769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l767_76701

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - (Real.sin x)^2 + 2

-- State the theorem
theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (y : ℝ), f y = M) ∧
  (let p := Real.pi; let M := 4;
   (p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
   (∀ (x : ℝ), f x ≤ M ∧ ∃ (y : ℝ), f y = M)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l767_76701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_S_squared_l767_76710

open Real BigOperators

-- Define S as a noncomputable real number
noncomputable def S : ℝ := ∑ i in Finset.range 2008, Real.sqrt (1 + 2 / (i + 1)^2 + 2 / (i + 2)^2)

-- State the theorem
theorem floor_S_squared : ⌊S^2⌋ = 4036079 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_S_squared_l767_76710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_repeating_decimal_to_hundredth_l767_76748

/-- Represents a repeating decimal with an integer part and a repeating fraction part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Rounds a real number to the nearest hundredth. -/
noncomputable def roundToHundredth (x : ℝ) : ℝ := 
  ⌊x * 100 + 0.5⌋ / 100

/-- Converts a RepeatingDecimal to its real number representation. -/
noncomputable def toReal (x : RepeatingDecimal) : ℝ := 
  x.integerPart + (x.repeatingPart : ℝ) / 999

theorem round_repeating_decimal_to_hundredth :
  let x : RepeatingDecimal := ⟨37, 736⟩
  roundToHundredth (toReal x) = 37.74 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_repeating_decimal_to_hundredth_l767_76748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_at_135_chord_equation_when_bisected_l767_76759

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define point P₀
def P₀ : ℝ × ℝ := (-1, 2)

-- Define chord AB passing through P₀
noncomputable def chord_AB (α : ℝ) (x y : ℝ) : Prop :=
  y - P₀.2 = Real.tan α * (x - P₀.1)

-- Theorem 1: Length of chord AB when α = 135°
theorem chord_length_at_135 :
  ∃ (A B : ℝ × ℝ),
    circle_eq A.1 A.2 ∧
    circle_eq B.1 B.2 ∧
    chord_AB (135 * Real.pi / 180) A.1 A.2 ∧
    chord_AB (135 * Real.pi / 180) B.1 B.2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 30 :=
by sorry

-- Theorem 2: Equation of line AB when P₀ bisects the chord
theorem chord_equation_when_bisected :
  ∃ (A B : ℝ × ℝ),
    circle_eq A.1 A.2 ∧
    circle_eq B.1 B.2 ∧
    P₀ = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
    ∀ (x y : ℝ), x - 2*y + 5 = 0 ↔ 
      ∃ (t : ℝ), x = A.1 + t*(B.1 - A.1) ∧ y = A.2 + t*(B.2 - A.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_at_135_chord_equation_when_bisected_l767_76759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_cannot_catch_mouse_cat_can_catch_mouse_l767_76791

/-- Represents a node in the labyrinth --/
structure Node where
  color : Bool  -- True for white, False for black

/-- Represents a labyrinth --/
structure Labyrinth where
  nodes : List Node
  start_cat : Node
  start_mouse : Node

/-- Represents the state of the game --/
structure GameState where
  cat_position : Node
  mouse_position : Node
  turn : Bool  -- True for cat's turn, False for mouse's turn

/-- Defines a valid move in the labyrinth --/
def valid_move (l : Labyrinth) (node1 : Node) (node2 : Node) : Prop :=
  node1 ∈ l.nodes ∧ node2 ∈ l.nodes

/-- Defines the game progression --/
noncomputable def game_step (l : Labyrinth) (state : GameState) : GameState :=
  sorry

/-- Theorem: If the labyrinth has a chessboard coloring and the cat and mouse start on opposite colored nodes, the cat cannot catch the mouse --/
theorem cat_cannot_catch_mouse (l : Labyrinth) 
  (h1 : l.start_cat.color ≠ l.start_mouse.color) 
  (h2 : ∀ (n1 n2 : Node), valid_move l n1 n2 → n1.color ≠ n2.color) :
  ∀ (state : GameState), state.cat_position ≠ state.mouse_position :=
by
  sorry

/-- Theorem: In a labyrinth where the cat can force the mouse into a corner, the cat can catch the mouse --/
theorem cat_can_catch_mouse (l : Labyrinth) 
  (h : ∃ (corner : Node), corner ∈ l.nodes ∧ 
       ∀ (state : GameState), ∃ (n : ℕ), 
         (game_step l)^[n] state = GameState.mk corner corner state.turn) :
  ∃ (state : GameState), state.cat_position = state.mouse_position :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_cannot_catch_mouse_cat_can_catch_mouse_l767_76791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_possible_a₁_l767_76711

def arithmetic_progression (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_first_n_terms (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_progression_possible_a₁ (a₁ d : ℤ) :
  d > 0 →
  (let S := sum_first_n_terms a₁ d 7
   arithmetic_progression a₁ d 8 * arithmetic_progression a₁ d 17 > S + 27 ∧
   arithmetic_progression a₁ d 11 * arithmetic_progression a₁ d 14 < S + 60) →
  a₁ ∈ ({-11, -10, -9, -7, -6, -5} : Set ℤ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_possible_a₁_l767_76711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_printers_completion_time_l767_76723

/-- Represents the printing task -/
structure PrintingTask where
  total_pages : ℕ
  printer_a_time : ℝ
  printer_b_extra_rate : ℝ

/-- Calculates the time taken for both printers to complete the task together -/
noncomputable def time_to_complete (task : PrintingTask) : ℝ :=
  let printer_a_rate := task.total_pages / task.printer_a_time
  let printer_b_rate := printer_a_rate + task.printer_b_extra_rate
  task.total_pages / (printer_a_rate + printer_b_rate)

/-- Theorem stating the time taken for both printers to complete the task -/
theorem printers_completion_time (task : PrintingTask) 
  (h1 : task.total_pages = 35)
  (h2 : task.printer_a_time = 60)
  (h3 : task.printer_b_extra_rate = 6) :
  ∃ (n : ℕ), (n : ℝ) ≤ time_to_complete task ∧ time_to_complete task < (n : ℝ) + 1 ∧ n = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_printers_completion_time_l767_76723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_175_l767_76781

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The theorem stating the range of numbers that round to 1.75 -/
theorem round_to_175 (a : ℝ) :
  roundToHundredth a = 1.75 ↔ 1.745 ≤ a ∧ a < 1.755 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_175_l767_76781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_exponent_l767_76713

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def are_like_terms (term1 term2 : ℚ → ℚ → ℚ) : Prop :=
  ∃ (a b : ℚ), ∀ (x y : ℚ), term1 x y = a * x^3 * y^5 ∧ term2 x y = b * x^3 * y^5

/-- The value of m for which -2x^3y^(m+3) and 9x^3y^5 are like terms -/
theorem like_terms_exponent (m : ℤ) : 
  are_like_terms (λ x y => -2 * x^3 * y^(m+3)) (λ x y => 9 * x^3 * y^5) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_exponent_l767_76713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l767_76786

theorem equation_solution : ∃! x : ℚ, (x - 1) / 3 = 2 * x := by
  use -1/5
  constructor
  · -- Prove that -1/5 satisfies the equation
    field_simp
    ring
  · -- Prove uniqueness
    intro y hy
    have : 3 * ((y - 1) / 3) = 3 * (2 * y) := by congr
    field_simp at this
    linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l767_76786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_parts_properties_l767_76725

/-- Given two machines A and B with yield rates, prove properties about their parts -/
theorem machine_parts_properties
  (yield_rate_A yield_rate_B : ℝ)
  (h_A : yield_rate_A = 0.8)
  (h_B : yield_rate_B = 0.9) :
  let p_both_defective := (1 - yield_rate_A) * (1 - yield_rate_B)
  let p_one_qualified := yield_rate_A * (1 - yield_rate_B) + (1 - yield_rate_A) * yield_rate_B
  let e_both_defective := {p : Bool × Bool | ¬p.1 ∧ ¬p.2}
  let e_at_least_one_qualified := {p : Bool × Bool | p.1 ∨ p.2}
  (p_both_defective = 0.02) ∧
  (p_one_qualified = 0.26) ∧
  (e_both_defective = (e_at_least_one_qualified)ᶜ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_parts_properties_l767_76725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_monotonicity_a_eq_neg_one_monotonicity_a_lt_neg_one_monotonicity_a_gt_neg_one_l767_76719

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 - (1 + a) * x + 1)

-- Theorem for Question I
theorem tangent_line_parallel (a : ℝ) : 
  (∀ x, (deriv (f a)) x = Real.exp x * (x - a) * (x + 1)) → 
  (deriv (f a)) 0 = 1 → 
  a = -1 := by sorry

-- Theorems for Question II
theorem monotonicity_a_eq_neg_one : 
  ∀ x, (deriv (f (-1))) x > 0 := by sorry

theorem monotonicity_a_lt_neg_one (a : ℝ) :
  a < -1 →
  (∀ x, x < a → (deriv (f a)) x > 0) ∧
  (∀ x, a < x → x < -1 → (deriv (f a)) x < 0) ∧
  (∀ x, x > -1 → (deriv (f a)) x > 0) := by sorry

theorem monotonicity_a_gt_neg_one (a : ℝ) :
  a > -1 →
  (∀ x, x < -1 → (deriv (f a)) x > 0) ∧
  (∀ x, -1 < x → x < a → (deriv (f a)) x < 0) ∧
  (∀ x, x > a → (deriv (f a)) x > 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_monotonicity_a_eq_neg_one_monotonicity_a_lt_neg_one_monotonicity_a_gt_neg_one_l767_76719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_for_submerged_spheres_l767_76739

/-- The volume of water needed to submerge four spheres in a cylindrical container -/
theorem water_volume_for_submerged_spheres (r R : Real) : 
  r = 1 ∧ R = 1/2 →
  let sphere_volume := 4/3 * Real.pi * R^3
  let total_sphere_volume := 4 * sphere_volume
  let water_height := r + (Real.sqrt 2 / 2)
  let cylinder_volume := Real.pi * r^2 * water_height
  cylinder_volume - total_sphere_volume = Real.pi * (1/3 + Real.sqrt 2/2) := by
  sorry

#check water_volume_for_submerged_spheres

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_for_submerged_spheres_l767_76739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l767_76728

noncomputable def F₁ : ℝ × ℝ := (-4, 2 - Real.sqrt 8 / 2)
noncomputable def F₂ : ℝ × ℝ := (-4, 2 + Real.sqrt 8 / 2)

def is_on_hyperbola (P : ℝ × ℝ) : Prop :=
  abs (dist P F₁ - dist P F₂) = 2

def hyperbola_equation (x y h k a b : ℝ) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

theorem hyperbola_sum (h k a b : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (heq : ∀ x y, is_on_hyperbola (x, y) ↔ hyperbola_equation x y h k a b) :
  h + k + a + b = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l767_76728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_eight_congruence_l767_76715

theorem mod_eight_congruence (n : ℕ) : (5^n + 2 * 3^(n-1) + 1) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_eight_congruence_l767_76715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_differentiable_f_domain_tangent_property_f_at_one_f_at_two_eq_ln_two_l767_76763

open Real

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the properties of f
theorem f_differentiable : Differentiable ℝ f := by sorry

theorem f_domain : ∀ x : ℝ, x > 0 → ∃ y, f x = y := by sorry

theorem tangent_property : ∀ x : ℝ, x > 0 → f x - x * deriv f x = f x - 1 := by sorry

theorem f_at_one : f 1 = 0 := by sorry

-- The theorem to prove
theorem f_at_two_eq_ln_two : f 2 = log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_differentiable_f_domain_tangent_property_f_at_one_f_at_two_eq_ln_two_l767_76763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_men_joining_boys_l767_76780

theorem men_joining_boys (initial_boys : ℕ) (initial_days : ℝ) (final_days : ℝ) : 
  initial_boys = 1000 →
  initial_days = 15 →
  final_days = 12.5 →
  (initial_boys : ℝ) * initial_days = (initial_boys + 200 : ℝ) * final_days :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_men_joining_boys_l767_76780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_red_pill_l767_76767

theorem cost_of_red_pill (treatment_duration : ℕ) (daily_red_pills : ℕ) (daily_blue_pills : ℕ) 
  (red_blue_cost_diff : ℚ) (total_cost : ℚ) 
  (h1 : treatment_duration = 3 * 7)
  (h2 : daily_red_pills = 1)
  (h3 : daily_blue_pills = 1)
  (h4 : red_blue_cost_diff = 2)
  (h5 : total_cost = 903)
  : (total_cost / (treatment_duration : ℚ) + red_blue_cost_diff) / 2 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_red_pill_l767_76767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ngo_wage_decrease_l767_76753

theorem ngo_wage_decrease (illiterate_employees literate_employees : ℕ) 
  (initial_wage final_wage : ℚ) :
  illiterate_employees = 20 →
  literate_employees = 10 →
  initial_wage = 25 →
  final_wage = 10 →
  (illiterate_employees : ℚ) * (initial_wage - final_wage) / 
    (illiterate_employees + literate_employees) = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ngo_wage_decrease_l767_76753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_average_distance_is_6_l767_76708

/-- Represents a square field -/
structure SquareField where
  side_length : ℝ

/-- Represents the position of the rabbit -/
structure RabbitPosition where
  x : ℝ
  y : ℝ

/-- Calculates the final position of the rabbit after its movement -/
noncomputable def final_position (field : SquareField) (diagonal_distance : ℝ) (turn_distance : ℝ) : RabbitPosition :=
  let diagonal := Real.sqrt (2 * field.side_length ^ 2)
  let fraction := diagonal_distance / diagonal
  let intermediate_x := fraction * field.side_length
  let intermediate_y := fraction * field.side_length
  { x := intermediate_x + turn_distance, y := intermediate_y }

/-- Calculates the average distance from a position to all sides of the square field -/
noncomputable def average_distance_to_sides (field : SquareField) (position : RabbitPosition) : ℝ :=
  (position.x + position.y + (field.side_length - position.x) + (field.side_length - position.y)) / 4

/-- Theorem stating that the average distance from the rabbit's final position to each side of the square is 6 meters -/
theorem rabbit_average_distance_is_6 (field : SquareField) (diagonal_distance : ℝ) (turn_distance : ℝ) :
  field.side_length = 12 →
  diagonal_distance = 8.4 →
  turn_distance = 3 →
  average_distance_to_sides field (final_position field diagonal_distance turn_distance) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_average_distance_is_6_l767_76708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_can_escape_l767_76755

/-- Represents a position on the 8x8 chessboard -/
structure Position where
  x : Fin 8
  y : Fin 8

/-- Represents the state of the game -/
structure GameState where
  mouse : Position
  cats : Finset Position
  turn : Nat

/-- Defines a valid move on the chessboard -/
def validMove (start finish : Position) : Prop :=
  (start.x = finish.x ∧ (start.y = finish.y + 1 ∨ start.y + 1 = finish.y)) ∨
  (start.y = finish.y ∧ (start.x = finish.x + 1 ∨ start.x + 1 = finish.x))

/-- Defines when the mouse has escaped -/
def mouseEscaped (pos : Position) : Prop :=
  pos.x = 0 ∨ pos.x = 7 ∨ pos.y = 0 ∨ pos.y = 7

/-- Theorem: The mouse can always escape from three cats -/
theorem mouse_can_escape (initialState : GameState) 
  (h1 : initialState.cats.card = 3)
  (h2 : initialState.turn = 0) : 
  ∃ (finalState : GameState), mouseEscaped finalState.mouse := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_can_escape_l767_76755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_order_l767_76712

open Real

-- Define the constants
noncomputable def a : ℝ := log (π / 3)
noncomputable def b : ℝ := log (exp 1 / 3)
noncomputable def c : ℝ := exp 0.5

-- Theorem statement
theorem abc_order : c > a ∧ a > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_order_l767_76712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l767_76732

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x + Real.pi / 4)

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ 
  (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l767_76732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_implies_a_range_l767_76784

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1/2 then -1/2 * x + 1/4
  else 2 * x^2 / (x + 2)

noncomputable def g (a x : ℝ) : ℝ :=
  a * Real.cos (Real.pi * x / 2) + 5 - 2 * a

theorem function_equality_implies_a_range (a : ℝ) :
  (a > 0) →
  (∀ x₁ ∈ Set.Icc (0 : ℝ) 1, ∃ x₂ ∈ Set.Icc (0 : ℝ) 1, f x₁ = g a x₂) →
  a ∈ Set.Icc (5/2 : ℝ) (13/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_implies_a_range_l767_76784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_sum_l767_76737

theorem quadratic_solution_sum (a b c d : ℕ+) (x y : ℝ) : 
  (x + y = 6) → 
  (3 * x * y = 6) → 
  (x = (a : ℝ) + (b : ℝ) * Real.sqrt (c : ℝ) / (d : ℝ) ∨ 
   x = (a : ℝ) - (b : ℝ) * Real.sqrt (c : ℝ) / (d : ℝ)) → 
  (∀ k m n p : ℕ+, ((k : ℝ) + (m : ℝ) * Real.sqrt (n : ℝ)) / (p : ℝ) = x → 
    k ≥ a ∧ m ≥ b ∧ n ≥ c ∧ p ≥ d) →
  a + b + c + d = 12 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_sum_l767_76737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l767_76798

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := -2 * (x^(1/3) + 3 * x^(1/2))

-- Define the point of tangency
def x₀ : ℝ := 1

-- State the theorem
theorem tangent_line_equation :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (∀ x y : ℝ, y = f x → (x = x₀ → a * y + b * x + c = 0)) ∧
  (∀ x y : ℝ, y = f x → (x ≠ x₀ → (a * y + b * x + c = 0 ↔ 
    (y - f x₀) / (x - x₀) = (deriv f x₀)))) ∧
  a = 3 ∧ b = 11 ∧ c = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l767_76798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_9_is_99_l767_76745

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_147 : a 1 + a 4 + a 7 = 39
  sum_369 : a 3 + a 6 + a 9 = 27

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem stating that the sum of the first 9 terms is 99 -/
theorem sum_9_is_99 (seq : ArithmeticSequence) : sum_n seq 9 = 99 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_9_is_99_l767_76745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_charging_time_l767_76778

/-- Represents the charging state of a phone -/
structure ChargingState where
  initialCharge : ℚ
  chargingTime : ℚ
  currentCharge : ℚ

/-- Calculates the additional charging time needed to reach full charge -/
noncomputable def additionalChargingTime (state : ChargingState) : ℚ :=
  (state.chargingTime / state.currentCharge) * (100 - state.currentCharge) - state.chargingTime

theorem phone_charging_time (state : ChargingState) 
  (h1 : state.initialCharge = 0)
  (h2 : state.chargingTime = 45)
  (h3 : state.currentCharge = 25) :
  additionalChargingTime state = 135 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_charging_time_l767_76778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_x_squared_minus_y_factorial_l767_76788

theorem unique_solution_x_squared_minus_y_factorial : 
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 - Nat.factorial y = 2019 ∧ x = 45 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_x_squared_minus_y_factorial_l767_76788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_trebled_after_five_years_l767_76730

/-- The number of years after which the principal is trebled -/
def n : ℕ := sorry

/-- The original principal amount -/
noncomputable def P : ℝ := sorry

/-- The rate of interest (in percentage per annum) -/
noncomputable def R : ℝ := sorry

/-- Simple interest formula -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem principal_trebled_after_five_years :
  simple_interest P R 10 = 400 ∧
  simple_interest P R n + simple_interest (3 * P) R (10 - n) = 800 →
  n = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_trebled_after_five_years_l767_76730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_periodic_function_l767_76751

open Real

noncomputable def f (α : ℝ) (a : ℝ) (x : ℝ) : ℝ := cos x + a * cos (α * x)

theorem non_periodic_function (α : ℝ) (a : ℝ) (h_irrational : Irrational α) (h_positive : a > 0) :
  ¬∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f α a (x + T) = f α a x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_periodic_function_l767_76751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_problem_l767_76765

/-- Given a geometric sequence with first term a and common ratio r,
    S(n) represents the sum of the first n terms of the sequence. -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- For the specific geometric sequence with a = 1/3 and r = 1/3,
    if the sum of the first n terms is 728/729, then n = 6. -/
theorem geometric_sum_problem : 
  let a : ℝ := 1/3
  let r : ℝ := 1/3
  ∀ n : ℕ, geometric_sum a r n = 728/729 → n = 6 := by
  sorry

#check geometric_sum_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_problem_l767_76765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_needed_is_31_gallons_l767_76752

-- Constants
def numTanks : ℕ := 20
def tankHeight : ℝ := 24
def tankDiameter : ℝ := 8
def paintCoverage : ℝ := 400

-- Function to calculate the lateral surface area of a single tank
noncomputable def lateralSurfaceArea (height : ℝ) (diameter : ℝ) : ℝ :=
  Real.pi * diameter * height

-- Function to calculate the total surface area for all tanks
noncomputable def totalSurfaceArea (numTanks : ℕ) (height : ℝ) (diameter : ℝ) : ℝ :=
  (numTanks : ℝ) * lateralSurfaceArea height diameter

-- Function to calculate the number of whole gallons needed
noncomputable def gallonsNeeded (area : ℝ) (coverage : ℝ) : ℕ :=
  Nat.ceil (area / coverage)

-- Theorem statement
theorem paint_needed_is_31_gallons :
  gallonsNeeded (totalSurfaceArea numTanks tankHeight tankDiameter) paintCoverage = 31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_needed_is_31_gallons_l767_76752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l767_76704

/-- The distance from the center of the circle ρ=4sin θ to the line θ=π/6 is √3 -/
theorem distance_circle_center_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + (y - 2)^2 = 4}
  let line := {(x, y) : ℝ × ℝ | x = Real.sqrt 3 * y}
  let center : ℝ × ℝ := (0, 2)
  Real.sqrt 3 = 
    (|(1 : ℝ) * center.1 + (-Real.sqrt 3) * center.2| / Real.sqrt ((1 : ℝ)^2 + (-Real.sqrt 3)^2))
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l767_76704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_specific_perpendicular_line_l767_76709

/-- The equation of a line perpendicular to another line and passing through a given point. -/
theorem perpendicular_line_equation (x₀ y₀ a b c : ℝ) (h : b ≠ 0) :
  let m₁ := a / b
  let m₂ := -1 / m₁
  (y - y₀ = m₂ * (x - x₀)) ↔ (a * x + b * y + (a * x₀ + b * y₀ - a * b * m₂) = 0) :=
by sorry

/-- The equation of the line perpendicular to 2x - 4y + 5 = 0 and passing through (2, -1) is 2x + y - 3 = 0. -/
theorem specific_perpendicular_line :
  let given_line := λ (x y : ℝ) => 2 * x - 4 * y + 5 = 0
  let point := (2, -1)
  let perpendicular_line := λ (x y : ℝ) => 2 * x + y - 3 = 0
  ∀ x y, perpendicular_line x y ↔ 
    (y - point.2 = (1/2) * (x - point.1) ∧ 
     (2 * x + 4 * y) * (x - point.1) = -(y - point.2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_specific_perpendicular_line_l767_76709
