import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eraser_sale_revenue_l741_74154

/-- Calculate the total revenue from selling erasers with bulk discount and sales tax -/
theorem eraser_sale_revenue :
  let num_boxes : ℕ := 48
  let erasers_per_box : ℕ := 24
  let price_per_eraser : ℚ := 3/4  -- $0.75 as a rational number
  let bulk_discount_rate : ℚ := 1/10  -- 10% discount
  let sales_tax_rate : ℚ := 6/100  -- 6% sales tax
  let total_erasers := num_boxes * erasers_per_box
  let discounted_price := price_per_eraser * (1 - bulk_discount_rate)
  let subtotal := (total_erasers : ℚ) * discounted_price
  let total_with_tax := subtotal * (1 + sales_tax_rate)
  ∃ (revenue : ℚ), (revenue * 100).floor / 100 = 82426 / 100 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eraser_sale_revenue_l741_74154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_locus_l741_74124

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line segment
structure Segment where
  start : Point2D
  finish : Point2D

-- Helper functions (not implemented, just signatures)
noncomputable def area_triangle (A B C : Point2D) : ℝ := sorry
noncomputable def corresponding_point (M : Point2D) (AB CD : Segment) : Point2D := sorry
def is_line (s : Set Point2D) : Prop := sorry

-- Define the problem statement
theorem equal_area_locus (AB CD : Segment) 
  (h : ¬ (AB.start.x - AB.finish.x) * (CD.finish.y - CD.start.y) = (AB.start.y - AB.finish.y) * (CD.finish.x - CD.start.x)) :
  ∃ (l₁ l₂ : Set Point2D), 
    (∀ M : Point2D, (area_triangle AB.start AB.finish M = area_triangle CD.start CD.finish (corresponding_point M AB CD)) ↔ 
      (M ∈ l₁ ∨ M ∈ l₂)) ∧ 
    is_line l₁ ∧ 
    is_line l₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_locus_l741_74124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basis_properties_l741_74171

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def is_basis (S : Set V) : Prop :=
  LinearIndependent ℝ (fun i => i : S → V) ∧ Submodule.span ℝ S = ⊤

theorem basis_properties
  {a b c : V} (h : is_basis {a, b, c}) :
  (∀ x y z : ℝ, x • a + y • b + z • c = 0 → x = 0 ∧ y = 0 ∧ z = 0) ∧
  (is_basis {a + b, b - c, c + 2 • a}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basis_properties_l741_74171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_properties_l741_74184

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the related circle E
def circle_E (x y : ℝ) : Prop := x^2 + y^2 = 2/3

-- Define a point on the circle E
def point_on_E (P : ℝ × ℝ) : Prop := circle_E P.1 P.2

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the angle AOB
noncomputable def angle_AOB (A B : ℝ × ℝ) : ℝ :=
  Real.arccos ((A.1 * B.1 + A.2 * B.2) / (Real.sqrt (A.1^2 + A.2^2) * Real.sqrt (B.1^2 + B.2^2)))

-- Define the area of triangle ABQ
noncomputable def area_ABQ (A B Q : ℝ × ℝ) : ℝ :=
  abs ((A.1 - Q.1) * (B.2 - Q.2) - (B.1 - Q.1) * (A.2 - Q.2)) / 2

theorem ellipse_circle_properties :
  ∀ P : ℝ × ℝ, point_on_E P →
  ∃ A B Q : ℝ × ℝ,
    ellipse_C A.1 A.2 ∧
    ellipse_C B.1 B.2 ∧
    circle_E Q.1 Q.2 ∧
    angle_AOB A B = π / 2 ∧
    4/3 ≤ area_ABQ A B Q ∧ area_ABQ A B Q ≤ Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_properties_l741_74184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_above_x_axis_l741_74170

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (9 : ℝ)^x - m * (3 : ℝ)^x + m + 1

-- State the theorem
theorem function_above_x_axis (m : ℝ) : 
  (∀ x > 0, f m x > 0) ↔ m < 2 + 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_above_x_axis_l741_74170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2014_l741_74159

def my_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ a 2 = 6 ∧ ∀ n : ℕ, a (n + 2) = a (n + 1) - a n

theorem sequence_2014 (a : ℕ → ℤ) (h : my_sequence a) : a 2014 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2014_l741_74159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_a_value_l741_74132

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (Real.exp (2 * x) + 1) + a * x

-- State the theorem
theorem even_function_a_value (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = -Real.log (Real.exp 2 + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_a_value_l741_74132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l741_74103

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 4)

-- State the theorem
theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Ioo (-Real.pi / 4) 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l741_74103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_savings_percentage_l741_74100

theorem salary_savings_percentage
  (last_year_salary : ℝ)
  (last_year_savings_rate : ℝ)
  (salary_increase_rate : ℝ)
  (h1 : last_year_savings_rate = 0.06)
  (h2 : salary_increase_rate = 0.20)
  (h3 : last_year_salary > 0) :
  let this_year_salary := last_year_salary * (1 + salary_increase_rate)
  let this_year_savings := last_year_salary * last_year_savings_rate
  this_year_savings / this_year_salary = 0.05 := by
  -- Unfold the let bindings
  simp only [h1, h2]
  -- Perform algebraic manipulations
  calc
    (last_year_salary * 0.06) / (last_year_salary * (1 + 0.20))
      = 0.06 / 1.20 := by
        field_simp [h3]
        ring
    _ = 0.05 := by norm_num

-- Check that the theorem is recognized
#check salary_savings_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_savings_percentage_l741_74100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l741_74138

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x + 1

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := Real.exp x + 2

-- State the theorem
theorem tangent_line_at_zero :
  let point := (0, f 0)
  let slope := f' 0
  (fun x => slope * x + f 0) = (fun x => 3 * x + 2) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l741_74138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_satisfies_conditions_l741_74109

noncomputable def mySequence (n : ℕ+) : ℚ :=
  3 - 2 / n.val

theorem mySequence_satisfies_conditions :
  (mySequence 1 = 1) ∧
  (mySequence 2 = 2) ∧
  (∀ n : ℕ+, n > 1 → mySequence n = ((n - 1) * mySequence (n - 1) + (n + 1) * mySequence (n + 1)) / (2 * n)) :=
by
  sorry

#check mySequence_satisfies_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_satisfies_conditions_l741_74109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_l741_74183

/-- The area of a regular octagon inscribed in a circle with radius 3 units -/
noncomputable def octagon_area : ℝ := 18 * Real.sqrt 2

/-- The radius of the circle in which the regular octagon is inscribed -/
def circle_radius : ℝ := 3

/-- Theorem stating that the area of a regular octagon inscribed in a circle 
    with radius 3 units is equal to 18√2 square units -/
theorem regular_octagon_area :
  let r := circle_radius
  octagon_area = 8 * (1/2 * r * r * Real.sin (π/4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_l741_74183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_6x8_first_player_wins_even_dimensions_second_player_wins_odd_dimensions_l741_74166

/-- Represents the state of the Chocolate game -/
structure ChocolateGame where
  rows : Nat
  cols : Nat
  markedRow : Nat
  markedCol : Nat

/-- Determines if a player has a winning strategy in the Chocolate game -/
def hasWinningStrategy (game : ChocolateGame) : Bool :=
  sorry

/-- Theorem: In a 6 × 8 Chocolate game, the first player has a winning strategy -/
theorem first_player_wins_6x8 :
  ∀ (mr mc : Nat), mr < 6 → mc < 8 →
  hasWinningStrategy ⟨6, 8, mr, mc⟩ = true := by
  sorry

/-- Theorem: If both dimensions are even, the first player always has a winning strategy -/
theorem first_player_wins_even_dimensions :
  ∀ (r c mr mc : Nat), Even r → Even c → mr < r → mc < c →
  hasWinningStrategy ⟨r, c, mr, mc⟩ = true := by
  sorry

/-- Theorem: If both dimensions are odd, the second player always has a winning strategy -/
theorem second_player_wins_odd_dimensions :
  ∀ (r c mr mc : Nat), Odd r → Odd c → mr < r → mc < c →
  hasWinningStrategy ⟨r, c, mr, mc⟩ = false := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_6x8_first_player_wins_even_dimensions_second_player_wins_odd_dimensions_l741_74166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l741_74196

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2 - x else 2 - x^2

-- State the theorem
theorem solution_set_equivalence :
  ∀ a : ℝ, f (2*a + 1) > f (3*a - 4) ↔ a > 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l741_74196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_135_degrees_proof_l741_74143

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def angle_between_vectors (a b : V) : ℝ :=
  Real.arccos ((inner a b) / (norm a * norm b))

theorem angle_135_degrees_proof (a b c : V) 
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h2 : norm a = norm b ∧ norm b = norm c ∧ norm c = norm (a + b + c)) : 
  angle_between_vectors a (b + c) = π * (3 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_135_degrees_proof_l741_74143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l741_74137

def set_A : Set ℤ := {1, 3, 5, 7}

def set_B : Set ℤ := {x : ℤ | -x^2 + 4*x ≥ 0}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l741_74137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_m_range_l741_74178

-- Define the curve C
def on_curve_C (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-1)^2) - abs y = 1

-- Define the line that intersects C
def line_intersects_C (k m : ℝ) (A B : ℝ × ℝ) : Prop :=
  m > 0 ∧ 
  on_curve_C A.1 A.2 ∧
  on_curve_C B.1 B.2 ∧
  A.2 = k * A.1 + m ∧
  B.2 = k * B.1 + m

-- Define the dot product condition
def dot_product_condition (k m : ℝ) (A B : ℝ × ℝ) : Prop :=
  let F := (0, 1)
  (A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) < 0

-- Main theorem
theorem curve_C_and_m_range :
  (∀ x y : ℝ, on_curve_C x y ↔ (y ≥ 0 ∧ x^2 = 4*y) ∨ (y < 0 ∧ x = 0)) ∧
  (∀ k m : ℝ, ∀ A B : ℝ × ℝ,
    line_intersects_C k m A B →
    (∀ k' : ℝ, dot_product_condition k' m A B) →
    3 - 2 * Real.sqrt 2 < m ∧ m < 3 + 2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_m_range_l741_74178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_negative_21_l741_74152

def mySequence (n : ℕ) : ℤ :=
  match n with
  | 0 => 1
  | n + 1 => mySequence n + ((-1) ^ (n + 1) : ℤ) * (4 * (n + 2) - 2)

theorem sixth_term_is_negative_21 : mySequence 5 = -21 := by
  -- The proof goes here
  sorry

#eval mySequence 5  -- This will evaluate the 6th term (index 5)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_negative_21_l741_74152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l741_74105

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4) - 1

theorem f_properties :
  -- Smallest positive period
  (∃ T : ℝ, T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧
  -- Period value
  (let T := Real.pi; ∀ x, f (x + T) = f x) ∧
  -- Monotonically increasing intervals
  (∀ k : ℤ, ∀ x y : ℝ, k * Real.pi - 3 * Real.pi / 8 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + Real.pi / 8 →
    f x < f y) ∧
  -- Maximum value
  (∀ x : ℝ, f x ≤ Real.sqrt 2 - 1) ∧
  (∃ x : ℝ, f x = Real.sqrt 2 - 1) ∧
  -- Minimum value
  (∀ x : ℝ, f x ≥ -Real.sqrt 2 - 1) ∧
  (∃ x : ℝ, f x = -Real.sqrt 2 - 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l741_74105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_N_l741_74180

theorem smallest_b_N (N : ℕ) (hN : N > 0) :
  (∃ (b_N : ℝ), ∀ (x : ℝ), ((x^(2*N) + 1)/2)^(1/N : ℝ) ≤ b_N*(x-1)^2 + x) ∧ 
  (∀ (b : ℝ), (∀ (x : ℝ), ((x^(2*N) + 1)/2)^(1/N : ℝ) ≤ b*(x-1)^2 + x) → b ≥ N/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_N_l741_74180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l741_74192

/-- Represents a train with its length and speed -/
structure Train where
  length : ℝ  -- length in meters
  speed : ℝ   -- speed in km/hr

/-- Calculates the time taken for two trains to cross each other -/
noncomputable def timeToCross (train1 : Train) (train2 : Train) : ℝ :=
  let totalLength := train1.length + train2.length
  let relativeSpeed := (train1.speed + train2.speed) * (5/18)  -- Convert km/hr to m/s
  totalLength / relativeSpeed

/-- Theorem stating the time taken for the trains to cross each other -/
theorem trains_crossing_time :
  let train1 : Train := { length := 300, speed := 60 }
  let train2 : Train := { length := 450, speed := 40 }
  ∃ ε > 0, |timeToCross train1 train2 - 27| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l741_74192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_center_of_transformed_function_l741_74160

noncomputable section

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - 2 * (Real.cos x) ^ 2

-- Define the transformed function
noncomputable def h (x : ℝ) : ℝ := 2 * Real.sin (2/3 * x - Real.pi/4) - 1

-- State the theorem
theorem symmetric_center_of_transformed_function :
  ∃ (center_x center_y : ℝ), 
    center_x = 3 * Real.pi / 8 ∧ 
    center_y = -1 ∧
    (∀ (x : ℝ), h (center_x + x) = h (center_x - x)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_center_of_transformed_function_l741_74160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nephews_problem_l741_74153

/-- The number of nephews problem -/
theorem nephews_problem (alden_past vihaan_diff nikhil_diff : ℕ) :
  alden_past = 70 →
  vihaan_diff = 120 →
  nikhil_diff = 40 →
  (let alden_current := 3 * alden_past
   let vihaan := alden_current + vihaan_diff
   let shruti := 2 * vihaan
   let nikhil := alden_current + shruti - nikhil_diff
   alden_current + vihaan + shruti + nikhil) = 2030 := by
  intros h1 h2 h3
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nephews_problem_l741_74153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_true_proposition_l741_74167

-- Define a power function
def is_power_function (f : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ b

-- Define the property of not passing through the fourth quadrant
def not_in_fourth_quadrant (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = y → ¬(x > 0 ∧ y < 0)

-- The main theorem
theorem one_true_proposition :
  ∃! p : Prop, p ∈ 
    ({ (∀ f : ℝ → ℝ, not_in_fourth_quadrant f → is_power_function f),  -- Converse
      (∃ f : ℝ → ℝ, ¬is_power_function f ∧ ¬not_in_fourth_quadrant f),  -- Inverse
      (∀ f : ℝ → ℝ, ¬not_in_fourth_quadrant f → ¬is_power_function f) } : Set Prop)  -- Contrapositive
  ∧ p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_true_proposition_l741_74167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merry_go_round_revolutions_l741_74176

/-- Calculates the number of revolutions needed for a horse at a different radius to travel the same distance on a merry-go-round -/
theorem merry_go_round_revolutions (r₁ r₂ n₁ : ℝ) (hr₁ : r₁ > 0) (hr₂ : r₂ > 0) (hn₁ : n₁ > 0) :
  2 * π * r₁ * n₁ = 2 * π * r₂ * (n₁ * (r₁ / r₂)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merry_go_round_revolutions_l741_74176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_five_n_values_l741_74149

/-- The number of positive integer values of n satisfying the condition -/
def num_valid_n : ℕ := 5

/-- The function f(x) = cos(2x) - a*sin(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.cos (2*x) - a * Real.sin x

/-- The number of zeros of f in the interval (0, n*π) -/
noncomputable def num_zeros (a : ℝ) (n : ℕ) : ℕ := 2022

/-- Theorem stating that exactly five n values satisfy the condition -/
theorem exactly_five_n_values :
  ∃ (S : Finset ℕ), S.card = num_valid_n ∧
  (∀ n ∈ S, ∃ a : ℝ, num_zeros a n = 2022) ∧
  (∀ n : ℕ, n ∉ S → ∀ a : ℝ, num_zeros a n ≠ 2022) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_five_n_values_l741_74149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_30_degrees_l741_74101

theorem cot_30_degrees : Real.tan (π / 6)⁻¹ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_30_degrees_l741_74101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_left_focus_l741_74142

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 7 = 1

-- Define the left focus
def left_focus : ℝ × ℝ := (-3, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem max_distance_to_left_focus :
  ∃ (max_dist : ℝ), max_dist = 7 ∧
  (∀ (p : ℝ × ℝ), ellipse p.1 p.2 → distance p left_focus ≤ max_dist) ∧
  (∃ (p : ℝ × ℝ), ellipse p.1 p.2 ∧ distance p left_focus = max_dist) := by
  sorry

#check max_distance_to_left_focus

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_left_focus_l741_74142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_l741_74190

theorem floor_inequality (α β : ℝ) (h1 : α ≥ 1) (h2 : β ≥ 1) :
  ⌊Real.sqrt α⌋ + ⌊Real.sqrt (α + β)⌋ + ⌊Real.sqrt β⌋ ≥ ⌊Real.sqrt (2 * α)⌋ + ⌊Real.sqrt (2 * β)⌋ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_l741_74190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_silver_solution_percentage_l741_74198

/-- Proves that the percentage of silver in the first solution is 4% given the specified conditions --/
theorem silver_solution_percentage : 
  ∀ (x : ℝ),
  x > 0 →  -- Ensuring x is positive
  5 * (x / 100) + 2.5 * (10 / 100) = 7.5 * (6 / 100) →
  x = 4 := by
  intro x hx_pos heq
  -- The proof steps would go here
  sorry

#check silver_solution_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_silver_solution_percentage_l741_74198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_switch_game_winnable_l741_74150

/-- Represents the position of a switch -/
inductive SwitchPosition
| Up
| Down

/-- Represents the configuration of the four switches -/
def SwitchConfig := Fin 4 → SwitchPosition

/-- Represents a move in the game -/
structure Move where
  switch1 : Fin 4
  switch2 : Fin 4
  flip1 : Bool
  flip2 : Bool

/-- Applies a move to a configuration -/
def applyMove (config : SwitchConfig) (move : Move) : SwitchConfig :=
  fun i =>
    if i = move.switch1 && move.flip1 then
      match config i with
      | SwitchPosition.Up => SwitchPosition.Down
      | SwitchPosition.Down => SwitchPosition.Up
    else if i = move.switch2 && move.flip2 then
      match config i with
      | SwitchPosition.Up => SwitchPosition.Down
      | SwitchPosition.Down => SwitchPosition.Up
    else
      config i

/-- Checks if all switches are in the same position -/
def allSame (config : SwitchConfig) : Prop :=
  (config 0 = config 1) ∧ (config 1 = config 2) ∧ (config 2 = config 3)

/-- The main theorem: there exists a sequence of moves to win from any starting configuration -/
theorem switch_game_winnable :
  ∀ (start : SwitchConfig), ∃ (moves : List Move), allSame (moves.foldl applyMove start) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_switch_game_winnable_l741_74150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_negative_sum_l741_74146

/-- The smallest non-negative result of adding "+" and "-" symbols to 1..1998 -/
theorem smallest_non_negative_sum : ℕ := by
  -- Define a function that represents all possible ways to add "+" and "-" symbols
  -- to a list of natural numbers and compute the result
  let compute_sum (signs : List Bool) (numbers : List ℕ) : ℤ := sorry

  -- Define the list of numbers from 1 to 1998
  let numbers : List ℕ := List.range 1998

  -- Define the type of all possible sign combinations
  let SignCombinations := List Bool

  -- State that 1 is achievable
  have h1 : ∃ (signs : SignCombinations), compute_sum signs numbers = 1 := by sorry

  -- State that no combination results in 0
  have h2 : ∀ (signs : SignCombinations), compute_sum signs numbers ≠ 0 := by sorry

  -- Conclude that 1 is the smallest non-negative result
  exact 1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_negative_sum_l741_74146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l741_74115

-- Define the ellipse G
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define a point on the ellipse
structure PointOnEllipse (G : Ellipse) where
  P : ℝ × ℝ
  h : (P.1 / G.a)^2 + (P.2 / G.b)^2 = 1

-- Define the conditions
def conditions (G : Ellipse) (P : PointOnEllipse G) : Prop :=
  let d₁ := Real.sqrt ((P.P.1 - F₁.1)^2 + (P.P.2 - F₁.2)^2)
  let d₂ := Real.sqrt ((P.P.1 - F₂.1)^2 + (P.P.2 - F₂.2)^2)
  (P.P.1 - F₂.1) * (F₂.1 - F₁.1) + (P.P.2 - F₂.2) * (F₂.2 - F₁.2) = 0 ∧
  d₁ - d₂ = G.a / 2

-- Define the theorem
theorem ellipse_theorem (G : Ellipse) (P : PointOnEllipse G) 
  (h : conditions G P) :
  G.a^2 = 4 ∧ G.b^2 = 3 ∧
  ∃ (m : ℝ), (∀ (x y : ℝ), y = m * (x - 1) ↔ x = m * y + 1) ∧
             (m = Real.sqrt 5 / 2 ∨ m = -Real.sqrt 5 / 2) := by
  sorry

-- Example of how to use the theorem
example (G : Ellipse) (P : PointOnEllipse G) (h : conditions G P) :
  G.a^2 = 4 ∧ G.b^2 = 3 := by
  have result := ellipse_theorem G P h
  exact ⟨result.1, result.2.1⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l741_74115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_packing_cylinder_height_l741_74140

/-- The height of a cylinder containing 8 spheres of radius 2, arranged in two layers
    such that each sphere is tangent to its four neighboring spheres and to one base
    and the side surface of the cylinder. -/
def cylinderHeight : ℝ := 4 + 2 * (8 : ℝ) ^ (1/4)

/-- Theorem stating the height of the cylinder under the given conditions. -/
theorem sphere_packing_cylinder_height :
  let n : ℕ := 8  -- number of spheres
  let r : ℝ := 2  -- radius of each sphere
  let layers : ℕ := 2  -- number of layers
  cylinderHeight = r * 2 * (layers : ℝ) + 2 * (8 : ℝ) ^ (1/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_packing_cylinder_height_l741_74140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_range_l741_74122

/-- The ellipse with equation x²/9 + y²/4 = 1 -/
def Ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

/-- The foci of the ellipse -/
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 5, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 5, 0)

/-- The condition for an obtuse angle F₁PF₂ -/
def IsObtuse (x y : ℝ) : Prop :=
  (x - F₁.1)^2 + (y - F₁.2)^2 + (x - F₂.1)^2 + (y - F₂.2)^2 < 20

/-- The theorem stating the range of x for points on the ellipse with obtuse F₁PF₂ -/
theorem ellipse_x_range (x y : ℝ) :
  Ellipse x y → IsObtuse x y → -3 * Real.sqrt 5 / 5 < x ∧ x < 3 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_range_l741_74122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_subsequence_iff_arithmetic_indices_l741_74164

/-- Given an infinite geometric sequence with common ratio q -/
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

/-- The sequence formed by taking elements at indices b_n -/
def subsequence (a : ℕ → ℝ) (b : ℕ → ℕ) : ℕ → ℝ :=
  fun n ↦ a (b n)

/-- An arithmetic sequence -/
def is_arithmetic_sequence (b : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, b (n + 1) = b n + d

theorem geometric_subsequence_iff_arithmetic_indices
  (a : ℕ → ℝ) (q : ℝ) (b : ℕ → ℕ) :
  is_geometric_sequence a q →
  q^2 ≠ 1 →
  (∀ n, b n > 0) →
  (is_geometric_sequence (subsequence a b) (q^(b 2 - b 1)) ↔
   is_arithmetic_sequence b) :=
by
  sorry

#check geometric_subsequence_iff_arithmetic_indices

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_subsequence_iff_arithmetic_indices_l741_74164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_face_angle_in_regular_tetrahedron_l741_74148

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- The angle between an edge and the plane of a face that does not contain this edge in a regular tetrahedron -/
noncomputable def edge_face_angle (t : RegularTetrahedron) : ℝ :=
  Real.arccos (1 / Real.sqrt 3)

/-- Theorem: In a regular tetrahedron, the angle between an edge and the plane of a face 
    that does not contain this edge is arccos(1/√3) -/
theorem edge_face_angle_in_regular_tetrahedron (t : RegularTetrahedron) :
  edge_face_angle t = Real.arccos (1 / Real.sqrt 3) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_face_angle_in_regular_tetrahedron_l741_74148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equals_twenty_l741_74134

def sequenceTerm (n : ℕ) : ℤ := n^2 - 14*n + 65

theorem sequence_equals_twenty (n : ℕ) : sequenceTerm n = 20 ↔ n = 5 ∨ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equals_twenty_l741_74134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_is_three_l741_74199

/-- The sum of the infinite geometric series 5 - 10/3 + 20/9 - 40/27 + ... -/
noncomputable def geometricSeriesSum : ℝ := 3

/-- The first term of the geometric series -/
noncomputable def a : ℝ := 5

/-- The common ratio of the geometric series -/
noncomputable def r : ℝ := -2/3

/-- Theorem stating that the sum of the infinite geometric series is 3 -/
theorem geometric_series_sum_is_three : 
  geometricSeriesSum = a / (1 - r) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_is_three_l741_74199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_cube_root_l741_74157

theorem rationalize_denominator_cube_root : 
  1 / (Real.rpow 3 (1/3) - 2) = -(Real.rpow 3 (2/3) + 2 * Real.rpow 3 (1/3) + 4) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_cube_root_l741_74157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_is_derivative_of_motion_l741_74139

/-- The motion equation of an object -/
noncomputable def s (t : ℝ) : ℝ := 2 * t * Real.sin t + t

/-- The velocity equation of the object -/
noncomputable def v (t : ℝ) : ℝ := 2 * Real.sin t + 2 * t * Real.cos t + 1

/-- Theorem: The velocity is the derivative of the motion equation -/
theorem velocity_is_derivative_of_motion : ∀ t, deriv s t = v t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_is_derivative_of_motion_l741_74139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l741_74113

def my_sequence (a : ℕ → ℤ) : Prop :=
  (∀ n, a (n + 1) = a (n + 2) - a n) ∧
  a 1 = 2 ∧
  a 2 = 5

theorem fifth_term_value (a : ℕ → ℤ) (h : my_sequence a) : a 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l741_74113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l741_74168

noncomputable def f (x : ℝ) : ℝ := (Real.sin x - 1) / Real.sqrt (3 - 2 * Real.cos x - 2 * Real.sin x)

theorem f_range :
  ∀ y : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f x = y) ↔ -1 ≤ y ∧ y ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l741_74168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_distribution_l741_74128

/-- Given n tourists and n cinemas, returns the number of ways to distribute 
    the tourists such that each cinema has exactly one tourist -/
def number_of_distributions (tourists : ℕ) (cinemas : ℕ) : ℕ :=
  sorry

/-- The number of ways to distribute n tourists among n cinemas, 
    such that each cinema has exactly one tourist, is equal to n! -/
theorem tourist_distribution (n : ℕ) : 
  (number_of_distributions n n) = Nat.factorial n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_distribution_l741_74128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l741_74145

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b = 1) :
  (a + Real.sqrt b ≤ Real.sqrt 2) ∧
  ((1/2 : ℝ) < (2 : ℝ)^(a - Real.sqrt b) ∧ (2 : ℝ)^(a - Real.sqrt b) < 2) ∧
  (a^2 - b > -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l741_74145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_in_70_74_interval_l741_74141

-- Define the score intervals
inductive ScoreInterval
| interval_60_64 : ScoreInterval
| interval_65_69 : ScoreInterval
| interval_70_74 : ScoreInterval
| interval_75_79 : ScoreInterval
| interval_80_84 : ScoreInterval
| interval_85_89 : ScoreInterval

-- Define the function that gives the number of students for each interval
def studentsInInterval (interval : ScoreInterval) : Nat :=
  match interval with
  | .interval_60_64 => 15
  | .interval_65_69 => 18
  | .interval_70_74 => 21
  | .interval_75_79 => 16
  | .interval_80_84 => 10
  | .interval_85_89 => 21

-- Define the total number of students
def totalStudents : Nat := 101

-- Define a function to calculate the cumulative sum up to a given interval
def cumulativeSum (interval : ScoreInterval) : Nat :=
  match interval with
  | .interval_60_64 => 0
  | .interval_65_69 => studentsInInterval .interval_60_64
  | .interval_70_74 => studentsInInterval .interval_60_64 + studentsInInterval .interval_65_69
  | .interval_75_79 => studentsInInterval .interval_60_64 + studentsInInterval .interval_65_69 + studentsInInterval .interval_70_74
  | .interval_80_84 => studentsInInterval .interval_60_64 + studentsInInterval .interval_65_69 + studentsInInterval .interval_70_74 + studentsInInterval .interval_75_79
  | .interval_85_89 => studentsInInterval .interval_60_64 + studentsInInterval .interval_65_69 + studentsInInterval .interval_70_74 + studentsInInterval .interval_75_79 + studentsInInterval .interval_80_84

-- Define a function to check if an interval contains the median
def containsMedian (interval : ScoreInterval) : Prop :=
  let cumSum := cumulativeSum interval
  cumSum < (totalStudents + 1) / 2 ∧
  (totalStudents + 1) / 2 ≤ cumSum + studentsInInterval interval

-- Theorem statement
theorem median_in_70_74_interval :
  containsMedian ScoreInterval.interval_70_74 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_in_70_74_interval_l741_74141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_integers_digits_sum_l741_74193

def count_digits (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else 4

def sum_digits_even (upper_bound : ℕ) : ℕ :=
  (List.range (upper_bound / 2)).map (fun i => count_digits ((i + 1) * 2)) |>.sum

theorem even_integers_digits_sum :
  sum_digits_even 6006 = 11460 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_integers_digits_sum_l741_74193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l741_74151

theorem solution_difference : 
  ∃ x₁ x₂ : ℝ, 
    ((7 - x₁^2 / 4)^(1/3) = 1) ∧ 
    ((7 - x₂^2 / 4)^(1/3) = 1) ∧ 
    x₁ ≠ x₂ ∧ 
    |x₁ - x₂| = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l741_74151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_side_lengths_l741_74194

/-- Definition of a triangle -/
structure Triangle where
  sides : Finset ℝ
  h_card : sides.card = 3
  h_positive : ∀ s ∈ sides, s > 0
  h_triangle_inequality : ∀ a b c, {a, b, c} = sides → a < b + c ∧ b < a + c ∧ c < a + b

/-- Definition of an isosceles triangle -/
def Triangle.isIsosceles (T : Triangle) : Prop :=
  ∃ a b c, {a, b, c} = T.sides ∧ a = b ∧ c ≠ a

/-- Definition of triangle perimeter -/
def Triangle.perimeter (T : Triangle) : ℝ :=
  T.sides.sum id

/-- An isosceles triangle with perimeter 12 and one side 3 has the other two sides equal to 4.5 each -/
theorem isosceles_triangle_side_lengths
  (T : Triangle)
  (h_isosceles : T.isIsosceles)
  (h_perimeter : T.perimeter = 12)
  (h_one_side : ∃ s, s ∈ T.sides ∧ s = 3) :
  ∃ a b, a ∈ T.sides ∧ b ∈ T.sides ∧ a = 4.5 ∧ b = 4.5 ∧ a ≠ 3 ∧ b ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_side_lengths_l741_74194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l741_74116

-- Define the function f(x) = x + a/x + b
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x + a / x + b

-- State the theorem
theorem function_properties (a b : ℝ) 
  (h1 : f a b 1 = 2) 
  (h2 : f a b 2 = 5/2) :
  -- 1. Explicit formula
  (∀ x : ℝ, x ≠ 0 → f a b x = x + 1/x) ∧
  -- 2. Odd function
  (∀ x : ℝ, x ≠ 0 → f a b (-x) = -(f a b x)) ∧
  -- 3. Increasing on (1, +∞)
  (∀ x y : ℝ, 1 < x → x < y → f a b x < f a b y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l741_74116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l741_74163

/-- Two lines are perpendicular if their slopes are negative reciprocals of each other -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of the first line (m+2)x + my + 1 = 0 -/
noncomputable def slope1 (m : ℝ) : ℝ := -((m + 2) / m)

/-- The slope of the second line (m-2)x + (m+2)y - 3 = 0 -/
noncomputable def slope2 (m : ℝ) : ℝ := -(m - 2) / (m + 2)

/-- The condition that m = -2 is sufficient but not necessary for the lines to be perpendicular -/
theorem perpendicular_condition (m : ℝ) : 
  (m = -2 → perpendicular (slope1 m) (slope2 m)) ∧ 
  ¬(perpendicular (slope1 m) (slope2 m) → m = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l741_74163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pawns_is_36_l741_74102

/-- A configuration of pawns on a 12x12 checkerboard -/
def PawnConfiguration := Fin 12 → Fin 12 → Bool

/-- Two positions are adjacent if they share a side or corner -/
def adjacent (x1 y1 x2 y2 : Fin 12) : Prop :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y2 = y1 + 1)) ∨
  (y1 = y2 ∧ (x1 = x2 + 1 ∨ x2 = x1 + 1)) ∨
  (x1 = x2 + 1 ∧ y1 = y2 + 1) ∨
  (x2 = x1 + 1 ∧ y1 = y2 + 1) ∨
  (x1 = x2 + 1 ∧ y2 = y1 + 1) ∨
  (x2 = x1 + 1 ∧ y2 = y1 + 1)

/-- A valid pawn configuration has no adjacent pawns -/
def valid_configuration (config : PawnConfiguration) : Prop :=
  ∀ x1 y1 x2 y2, config x1 y1 ∧ config x2 y2 → ¬adjacent x1 y1 x2 y2

/-- Count the number of pawns in a configuration -/
def pawn_count (config : PawnConfiguration) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 12)) fun x =>
    Finset.sum (Finset.univ : Finset (Fin 12)) fun y =>
      if config x y then 1 else 0)

/-- The maximum number of pawns that can be placed on the board -/
def max_pawns : Nat := 36

/-- Theorem: The maximum number of pawns that can be placed on a 12x12 checkerboard,
    such that no two pawns are on adjacent squares, is 36 -/
theorem max_pawns_is_36 :
  (∃ config : PawnConfiguration, valid_configuration config ∧ pawn_count config = max_pawns) ∧
  (∀ config : PawnConfiguration, valid_configuration config → pawn_count config ≤ max_pawns) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pawns_is_36_l741_74102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_dot_product_l741_74155

-- Define the triangle and points
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A B = 2 ∧ d B C = 2 ∧ d C A = 2

def Midpoint (D B C : ℝ × ℝ) : Prop :=
  D.1 = (B.1 + C.1) / 2 ∧ D.2 = (B.2 + C.2) / 2

def VectorScale (E C A : ℝ × ℝ) : Prop :=
  E.1 - C.1 = (1/3) * (A.1 - C.1) ∧ E.2 - C.2 = (1/3) * (A.2 - C.2)

def DotProduct (p q r s : ℝ × ℝ) : ℝ :=
  (r.1 - p.1) * (s.1 - q.1) + (r.2 - p.2) * (s.2 - q.2)

theorem triangle_dot_product 
  (A B C D E : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : Midpoint D B C) 
  (h3 : VectorScale E C A) : 
  DotProduct D E C B = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_dot_product_l741_74155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l741_74186

/-- Given a hyperbola with equation (x-4)²/3² - (y-15)²/10² = 1, 
    the coordinates of the focus with the larger x-coordinate are (4 + √109, 15) -/
theorem hyperbola_focus (x y : ℝ) :
  (x - 4)^2 / 3^2 - (y - 15)^2 / 10^2 = 1 →
  ∃ (f_x f_y : ℝ), f_x > 4 ∧ f_y = 15 ∧ 
  f_x = 4 + Real.sqrt 109 ∧
  ∀ (p : ℝ × ℝ), p.1 > 4 ∧ p.2 = 15 ∧ 
    ((p.1 - 4)^2 / 3^2 - (p.2 - 15)^2 / 10^2 = 1) → 
    f_x ≥ p.1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l741_74186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_iff_a_pm_one_l741_74126

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x - 1 else -x^2 - 2*x

-- State the theorem
theorem f_equals_one_iff_a_pm_one :
  ∀ a : ℝ, f a = 1 ↔ (a = 1 ∨ a = -1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_iff_a_pm_one_l741_74126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l741_74162

/-- The length of a train in meters, given its speed in kmph and the time it takes to cross a pole -/
noncomputable def train_length (speed_kmph : ℝ) (crossing_time : ℝ) : ℝ :=
  (speed_kmph / 3.6) * crossing_time

/-- Theorem: A train traveling at 270 kmph that crosses a pole in 5 seconds has a length of 375 meters -/
theorem train_length_calculation :
  train_length 270 5 = 375 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check train_length 270 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l741_74162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_a_l741_74110

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a / Real.exp x

-- State the theorem
theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≥ 3/2) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = 3/2) →
  a = -Real.sqrt (Real.exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_a_l741_74110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mitzi_remaining_money_l741_74136

/-- Calculates the remaining amount in dollars after expenses, given an initial amount in yen, total expenses in yen, and an exchange rate. The result is rounded to two decimal places. -/
def remaining_amount (initial : ℕ) (expenses : ℕ) (exchange_rate : ℚ) : ℚ :=
  let remaining_yen := initial - expenses
  let dollars := (remaining_yen : ℚ) / exchange_rate
  (dollars * 100).floor / 100

/-- Proves that given the specified conditions, the remaining amount is $7.27 -/
theorem mitzi_remaining_money :
  remaining_amount 10000 9200 110 = 727/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mitzi_remaining_money_l741_74136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l741_74121

-- Define the sequences a_n, b_n, and c_n
noncomputable def a : ℕ → ℝ := sorry
noncomputable def b : ℕ → ℝ := sorry
noncomputable def c (n : ℕ) : ℝ := a n * b n

-- Define S_n (sum of first n terms of b_n)
noncomputable def S : ℕ → ℝ := sorry

-- Define T_n (sum of first n terms of c_n)
noncomputable def T : ℕ → ℝ := sorry

-- State the theorem
theorem sequence_sum_theorem (n : ℕ) :
  (∀ k, a (k + 1) > a k) →  -- a_n is increasing
  a 5^2 = a 10 →
  (∀ k, 2 * (a k + a (k + 2)) = 5 * a (k + 1)) →
  b 1 = 1 →
  (∀ k, b k ≠ 0) →
  (∀ k, b k * b (k + 1) = 4 * S k - 1) →
  T n = (2 * n - 3) * 2^(n + 1) + 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l741_74121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_relation_l741_74175

-- Define the isosceles triangle
def isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

-- Define the right triangle
def right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

-- Define the area of a triangle using Heron's formula
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem isosceles_right_triangle_area_relation :
  ∀ (a b c d e f : ℝ),
    isosceles_triangle a b c →
    right_triangle d e f →
    a = 13 ∧ b = 13 ∧ c = 10 →
    d = 5 ∧ e = 12 ∧ f = 13 →
    triangle_area a b c = 2 * triangle_area d e f :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_relation_l741_74175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_z_equals_100_l741_74147

-- Define the function f
def f (t : ℝ) : ℝ := t^2 + 49

-- Define the variables
def x : ℝ := 200
def y : ℝ := 2 * 100
def z : ℝ := 100
noncomputable def w : ℝ := ∫ t in (0:ℝ)..(1:ℝ), f t

-- State the theorem
theorem prove_z_equals_100 :
  x + y + z + w = 500 ∧
  x = 200 ∧
  y = 2 * z ∧
  x - z = 0.5 * y ∧
  w = ∫ t in (0:ℝ)..(1:ℝ), f t ∧
  f 0 = 2 ∧
  f 1 = 50 ∧
  (∀ t, deriv f t = 2 * t) →
  z = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_z_equals_100_l741_74147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_invariant_l741_74111

noncomputable section

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_a_ge_b : a ≥ b

-- Define a point on the ellipse
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1

-- Define the center and foci
def center : ℝ × ℝ := (0, 0)
def focusA (E : Ellipse) : ℝ × ℝ := (E.a * Real.sqrt (1 - E.b^2 / E.a^2), 0)
def focusB (E : Ellipse) : ℝ × ℝ := (-E.a * Real.sqrt (1 - E.b^2 / E.a^2), 0)

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the distance from center to tangent
def distanceToTangent (E : Ellipse) (P : PointOnEllipse E) : ℝ :=
  E.a * E.b / Real.sqrt (E.b^2 * P.x^2 + E.a^2 * P.y^2)

-- State the theorem
theorem ellipse_invariant (E : Ellipse) (P : PointOnEllipse E) :
  distance (P.x, P.y) (focusA E) * distance (P.x, P.y) (focusB E) *
  (distanceToTangent E P)^2 = E.a^2 * E.b^2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_invariant_l741_74111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_discount_percentage_l741_74133

theorem clothing_discount_percentage (p : ℝ) (hp : p > 0) : 
  let first_sale_price := (4/5 : ℝ) * p
  let second_sale_price := (1/2 : ℝ) * p
  (first_sale_price - second_sale_price) / first_sale_price * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_discount_percentage_l741_74133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_circle_l741_74197

-- Define the circle
def is_on_circle (x y : ℤ) : Prop := x^2 + y^2 = 16

-- Define a point on the circle
structure Point :=
  (x : ℤ)
  (y : ℤ)
  (on_circle : is_on_circle x y)

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Theorem statement
theorem max_ratio_on_circle (A B C D : Point) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h_unequal : distance A B ≠ distance C D) :
  (∃ (A' B' C' D' : Point), 
    distance A' B' / distance C' D' ≤ Real.sqrt 10 / 3) ∧
  (∃ (A'' B'' C'' D'' : Point), 
    distance A'' B'' / distance C'' D'' = Real.sqrt 10 / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_circle_l741_74197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_meeting_time_l741_74174

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ  -- Speed in meters per second
  headStart : ℝ  -- Head start in seconds

/-- Calculates the time when two runners meet in a race -/
noncomputable def meetingTime (runnerA runnerB : Runner) (raceLength : ℝ) : ℝ :=
  (raceLength + runnerB.speed * runnerB.headStart - runnerA.speed * runnerB.headStart) / (runnerA.speed - runnerB.speed)

/-- Theorem: In a 100-meter race, with given conditions, the slower runner runs for 42 seconds before being caught -/
theorem race_meeting_time :
  let cristina : Runner := { speed := 5, headStart := 0 }
  let nicky : Runner := { speed := 3, headStart := 12 }
  let raceLength : ℝ := 100
  meetingTime cristina nicky raceLength + nicky.headStart = 42 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_meeting_time_l741_74174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_plus_cos_x_gt_1_l741_74129

theorem sin_2x_plus_cos_x_gt_1 (x : ℝ) (h : 0 < x ∧ x < π/3) : 
  Real.sin (2*x) + Real.cos x > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_plus_cos_x_gt_1_l741_74129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l741_74114

-- Define points A and B
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (-3, 1)

-- Define the set of slopes k for which y = kx intersects AB
def intersecting_slopes : Set ℝ :=
  {k : ℝ | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2) ∈ {(x, y) : ℝ × ℝ | y = k * x}}

-- Theorem statement
theorem slope_range :
  intersecting_slopes = Set.Iic (-1/3) ∪ Set.Ici 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l741_74114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_thirteen_problem_l741_74156

theorem mod_thirteen_problem (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 13) (h3 : 4 * n ≡ 1 [ZMOD 13]) :
  (3^n)^4 - 3 ≡ 6 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_thirteen_problem_l741_74156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculations_l741_74104

theorem arithmetic_calculations : 
  ((1 : ℤ) * (-7) - 5 + (-4) - (-10) = -6) ∧ 
  ((-10 : ℤ)^3 + ((-4)^2 - (1 - 3^2) * 2) = -968) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculations_l741_74104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_relationship_l741_74131

-- Define a type for lines
structure Line where
  -- Add necessary fields here
  mk ::

-- Define a type for planes
structure Plane where
  -- Add necessary fields here
  mk ::

-- Define what it means for a line to be parallel to a plane
def line_parallel_to_plane (l : Line) (π : Plane) : Prop := sorry

-- Define what it means for two planes to be parallel
def planes_parallel (π₁ π₂ : Plane) : Prop := sorry

-- Define what it means for two planes to intersect
def planes_intersect (π₁ π₂ : Plane) : Prop := sorry

-- Define what it means for there to be countless lines in a plane parallel to another plane
noncomputable def countless_parallel_lines (π₁ π₂ : Plane) : Prop :=
  ∃ (S : Set Line), (∀ l ∈ S, line_parallel_to_plane l π₂) ∧ Set.Infinite S

-- State the theorem
theorem plane_relationship (π₁ π₂ : Plane) :
  countless_parallel_lines π₁ π₂ → planes_parallel π₁ π₂ ∨ planes_intersect π₁ π₂ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_relationship_l741_74131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l741_74173

noncomputable section

open Real

def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0

def area (a b c : ℝ) : ℝ := 
  let s := (a + b + c) / 2
  sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  triangle A B C a b c →
  sqrt 3 * b * cos A - a * sin B = 0 →
  c = 4 →
  area a b c = 6 * sqrt 3 →
  A = Real.pi / 3 ∧ a = 2 * sqrt 7 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l741_74173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log4_80_not_calculable_l741_74106

-- Define the given logarithmic values
noncomputable def log4_16 : ℝ := Real.log 16 / Real.log 4
noncomputable def log4_32 : ℝ := Real.log 32 / Real.log 4

-- Define a function that represents our ability to calculate logarithms
def can_calculate (x : ℝ) : Prop :=
  ∃ (f : ℝ → ℝ → ℝ), f log4_16 log4_32 = x

-- State the theorem
theorem log4_80_not_calculable : ¬(can_calculate (Real.log 80 / Real.log 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log4_80_not_calculable_l741_74106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_contains_point_l741_74125

/-- The line equation is 2kx + 1 = -7y -/
def line_equation (k x y : ℚ) : Prop := 2 * k * x + 1 = -7 * y

/-- The point coordinates -/
def point : ℚ × ℚ := (-1/2, 3)

/-- Theorem stating that k = 22 is the unique value for which the line contains the given point -/
theorem line_contains_point :
  ∃! k : ℚ, line_equation k point.fst point.snd := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_contains_point_l741_74125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_monotonic_decreasing_phi_l741_74135

-- Define the function f
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.cos (2 * x + φ)

-- Define the monotonically decreasing property
def monotonically_decreasing (g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → g y < g x

-- Theorem statement
theorem cos_monotonic_decreasing_phi (φ : ℝ) :
  monotonically_decreasing (f φ) 0 (Real.pi / 2) →
  φ = 2 * Real.pi ∨ ∃ k : ℤ, φ = 2 * k * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_monotonic_decreasing_phi_l741_74135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_b_values_l741_74191

theorem count_b_values : 
  let count := Finset.filter (fun b : ℕ => 
    b ≤ 100 ∧ 
    b % 2 = 1
  ) (Finset.range 101)
  Finset.card count = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_b_values_l741_74191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_line_l741_74165

noncomputable section

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Definition of the line l -/
def line_l (x y m : ℝ) : Prop := y = x + m

/-- Definition of a point being on the ellipse C -/
def point_on_ellipse (x y : ℝ) : Prop := ellipse_C x y

/-- Definition of a point being on the line l -/
def point_on_line (x y m : ℝ) : Prop := line_l x y m

/-- Definition of the area of a triangle formed by two points on the ellipse and the origin -/
def triangle_area (x₁ y₁ x₂ y₂ : ℝ) : ℝ := abs (x₁ * y₂ - x₂ * y₁) / 2

theorem ellipse_intersection_line :
  ∀ m : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    point_on_ellipse x₁ y₁ ∧
    point_on_ellipse x₂ y₂ ∧
    point_on_line x₁ y₁ m ∧
    point_on_line x₂ y₂ m ∧
    x₁ ≠ x₂ ∧
    triangle_area x₁ y₁ x₂ y₂ = 1) →
  m = Real.sqrt 10 / 2 ∨ m = -Real.sqrt 10 / 2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_line_l741_74165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_156_l741_74172

theorem greatest_prime_factor_of_156 : 
  (Nat.factors 156).maximum? = some 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_156_l741_74172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_approx_l741_74108

/-- The radius of a circle inscribed within three mutually externally tangent circles --/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  1 / (1/a + 1/b + 1/c + 2 * Real.sqrt (1/(a*b) + 1/(a*c) + 1/(b*c)))

/-- The radius of the inscribed circle is approximately 1.381 --/
theorem inscribed_circle_radius_approx :
  let a : ℝ := 5
  let b : ℝ := 10
  let c : ℝ := 20
  ∃ ε > 0, |inscribed_circle_radius a b c - 1.381| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_approx_l741_74108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_finding_same_state_l741_74130

/-- Represents the state of an Ameikafu (either has an electron or doesn't) -/
inductive AmeikafulState
| hasElectron
| noElectron

/-- Represents a diode connection between two Ameikafus -/
structure DiodeConnection where
  source : Fin 2015
  target : Fin 2015

/-- Represents the state of all Ameikafus -/
def AmeikafulConfiguration := Fin 2015 → AmeikafulState

/-- Applies a diode connection to a configuration -/
def applyDiodeConnection (config : AmeikafulConfiguration) (connection : DiodeConnection) : AmeikafulConfiguration :=
  sorry

/-- Represents a sequence of diode connections -/
def DiodeSequence := List DiodeConnection

/-- Applies a sequence of diode connections to a configuration -/
def applyDiodeSequence (config : AmeikafulConfiguration) (sequence : DiodeSequence) : AmeikafulConfiguration :=
  sorry

/-- Determines if two Ameikafus are in the same state -/
def sameState (config : AmeikafulConfiguration) (a b : Fin 2015) : Prop :=
  config a = config b

/-- The main theorem stating that it's impossible to guarantee finding two Ameikafus in the same state -/
theorem impossibility_of_finding_same_state :
  ∀ (sequence : DiodeSequence),
    ∃ (initial_config : AmeikafulConfiguration),
      ∀ (a b : Fin 2015),
        a ≠ b →
          ¬(sameState (applyDiodeSequence initial_config sequence) a b) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_finding_same_state_l741_74130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_strategy_exists_l741_74158

/-- Represents the possible numbers that can be chosen -/
inductive Number
| one : Number
| two : Number
| three : Number

/-- Represents a pair of distinct numbers from which a sage can choose -/
structure Choice where
  first : Number
  second : Number
  distinct : first ≠ second

/-- Represents the information about the previous two sages' choices -/
structure PreviousChoices where
  penultimate : Number
  last : Number

/-- A strategy for a sage to choose a number -/
def Strategy := Choice → PreviousChoices → Number

/-- The result of applying a strategy to all 100 sages -/
def ApplyStrategy (strategy : Strategy) : List Number :=
  sorry -- Implementation details omitted for brevity

/-- The sum of the numbers chosen by all sages -/
def SumOfChoices (choices : List Number) : Nat :=
  sorry -- Implementation details omitted for brevity

/-- Theorem stating that there exists a winning strategy -/
theorem winning_strategy_exists :
  ∃ (strategy : Strategy),
    ∀ (choices : List Number),
      choices = ApplyStrategy strategy →
        SumOfChoices choices ≠ 200 :=
by
  sorry -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_strategy_exists_l741_74158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_truthtellers_l741_74144

structure Person where
  number : ℝ
  truthful : Bool

def makeClaim (p : Person) (n : ℕ) : Prop :=
  (p.number > n) ∧ (p.number < n + 1)

def validClaims (people : Fin 10 → Person) : Prop :=
  ∀ i : Fin 10, (people i).truthful → makeClaim (people i) i.val

theorem max_truthtellers (people : Fin 10 → Person) 
  (h : validClaims people) : 
  (Finset.filter (fun i => (people i).truthful) (Finset.univ : Finset (Fin 10))).card ≤ 9 := by
  sorry

#check max_truthtellers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_truthtellers_l741_74144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_satisfying_equation_l741_74120

theorem largest_x_satisfying_equation :
  ∃ (x : ℝ), x ≥ 0 ∧ Real.sqrt (3 * x) = 6 * x^2 ∧
  ∀ (y : ℝ), y ≥ 0 ∧ Real.sqrt (3 * y) = 6 * y^2 → y ≤ x ∧
  x = (12 : ℝ)^(-(1/3 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_satisfying_equation_l741_74120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_function_from_M_to_N_l741_74118

def M : Set Int := {-2, 1, 2, 4}
def N : Set Int := {1, 2, 4, 16}

def f (x : Int) : Int := 2^(Int.natAbs x)

theorem f_is_function_from_M_to_N :
  (∀ x ∈ M, f x ∈ N) ∧ (∀ y ∈ N, ∃ x ∈ M, f x = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_function_from_M_to_N_l741_74118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_inverse_l741_74117

-- Define the functions f, g, and k
noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := 3 * x - 4
noncomputable def k (x : ℝ) : ℝ := f (g x)

-- Define the inverse function of k
noncomputable def k_inv (x : ℝ) : ℝ := (x + 11) / 12

-- Theorem stating that k_inv is the inverse of k
theorem k_inverse : 
  (∀ x : ℝ, k (k_inv x) = x) ∧ (∀ x : ℝ, k_inv (k x) = x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_inverse_l741_74117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_rotation_cylinder_volume_l741_74195

/-- Represents the volume of a cylinder formed by rotating a rectangle. -/
noncomputable def cylinderVolume (shortSide length : ℝ) : ℝ :=
  Real.pi * (shortSide / 2)^2 * length

/-- Theorem stating that the volume of a cylinder formed by rotating a rectangle
    with sides 12 cm and 18 cm about its shorter side is 648π cm³. -/
theorem rectangle_rotation_cylinder_volume :
  cylinderVolume 12 18 = 648 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_rotation_cylinder_volume_l741_74195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l741_74182

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  S : Real

-- State the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0)
  (h2 : t.A + t.B + t.C = Real.pi)
  (h3 : Real.sin (t.B + t.C) = (2 * t.S) / (t.a^2 - t.c^2))
  (h4 : t.b = 2)
  (h5 : t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2) : 
  t.A = 2 * t.C ∧ 
  (Real.sqrt 3/2 < t.S ∧ t.S < 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l741_74182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_property_l741_74161

/-- Given a cubic function g(x) = px³ + qx² + rx + s where g(-1) = 1,
    prove that 9p - 5q + 3r - s = -9 -/
theorem cubic_function_property (p q r s : ℝ) : 
  (fun x ↦ p * x^3 + q * x^2 + r * x + s) (-1) = 1 →
  9 * p - 5 * q + 3 * r - s = -9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_property_l741_74161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadrilateral_smallest_angle_not_opposite_longest_side_l741_74185

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the length of a side
noncomputable def side_length (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define an angle in a quadrilateral
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  sorry -- Actual implementation would calculate the angle

-- Theorem statement
theorem exists_quadrilateral_smallest_angle_not_opposite_longest_side :
  ∃ (q : Quadrilateral),
    let sides := [
      side_length q.A q.B,
      side_length q.B q.C,
      side_length q.C q.D,
      side_length q.D q.A
    ]
    let angles := [
      angle q.B q.A q.D,
      angle q.C q.B q.A,
      angle q.D q.C q.B,
      angle q.A q.D q.C
    ]
    ∃ (smallest_angle longest_side : ℝ),
    ∃ (i j : Fin 4),
      angles[i] = smallest_angle ∧
      sides[j] = longest_side ∧
      smallest_angle = angles.minimum ∧
      longest_side = sides.maximum ∧
      i ≠ j :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadrilateral_smallest_angle_not_opposite_longest_side_l741_74185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_burgers_count_l741_74119

/-- Calculates the number of double burgers bought given the total spend, total number of burgers, and prices of single and double burgers. -/
def double_burgers_bought (total_spend : ℚ) (total_burgers : ℕ) (single_price : ℚ) (double_price : ℚ) : ℕ :=
  let double_burgers := (total_spend - single_price * total_burgers) / (double_price - single_price)
  (Int.floor double_burgers).toNat

/-- Proves that given the specified conditions, the number of double burgers bought is 41. -/
theorem double_burgers_count : 
  double_burgers_bought 70.5 50 1 1.5 = 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_burgers_count_l741_74119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_problem_l741_74189

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define a point on a parabola
def point_on_parabola (p : ℝ) (A : ℝ × ℝ) : Prop :=
  parabola p A.1 A.2

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the point M
def M : ℝ × ℝ := (0, 10)

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define a circle
def on_circle (center : ℝ × ℝ) (radius : ℝ) (P : ℝ × ℝ) : Prop :=
  distance center P = radius

-- Define an equilateral triangle
def equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  distance A B = distance B C ∧ distance B C = distance C A

-- The main theorem
theorem parabola_problem (p : ℝ) (A B : ℝ × ℝ) :
  point_on_parabola p A →
  on_circle M (distance origin A) A →
  on_circle M (distance origin A) B →
  equilateral_triangle A B origin →
  p = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_problem_l741_74189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_with_two_equalizers_l741_74181

/-- A triangle represented by its side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- An equalizer is a line that divides a triangle into two regions of equal area and perimeter -/
def is_equalizer (t : Triangle) (l : ℝ × ℝ → Prop) : Prop :=
  ∃ (area_left area_right perimeter_left perimeter_right : ℝ),
    area_left = area_right ∧
    perimeter_left = perimeter_right

/-- A triangle has exactly two distinct equalizers -/
def has_two_distinct_equalizers (t : Triangle) : Prop :=
  ∃ (l₁ l₂ : ℝ × ℝ → Prop), is_equalizer t l₁ ∧ is_equalizer t l₂ ∧ l₁ ≠ l₂ ∧
    ∀ (l : ℝ × ℝ → Prop), is_equalizer t l → (l = l₁ ∨ l = l₂)

/-- The theorem stating the existence of a triangle with side lengths 9, 8, 7 
    having exactly two distinct equalizers, and that this is the smallest possible set -/
theorem smallest_triangle_with_two_equalizers :
  ∃ (t : Triangle),
    t.a = 9 ∧ t.b = 8 ∧ t.c = 7 ∧ 
    has_two_distinct_equalizers t ∧
    ∀ (t' : Triangle), has_two_distinct_equalizers t' →
      (t'.a > 9 ∨ 
       (t'.a = 9 ∧ t'.b > 8) ∨
       (t'.a = 9 ∧ t'.b = 8 ∧ t'.c ≥ 7)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_with_two_equalizers_l741_74181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_approx_l741_74179

/-- The number of sides in a regular hexadecagon -/
def n : ℕ := 16

/-- The central angle of each triangle in radians -/
noncomputable def θ : ℝ := 2 * Real.pi / n

/-- The area of a regular hexadecagon inscribed in a circle with radius r -/
noncomputable def hexadecagon_area (r : ℝ) : ℝ :=
  n * (1/2 * r^2 * Real.sin θ)

/-- Theorem stating that the area of a regular hexadecagon inscribed in a circle
    with radius r is approximately equal to 3.0616 r^2 -/
theorem hexadecagon_area_approx (r : ℝ) (h : r > 0) :
  ∃ ε > 0, |hexadecagon_area r - 3.0616 * r^2| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_approx_l741_74179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_or_diff_divisible_by_100_l741_74107

theorem sum_or_diff_divisible_by_100 (S : Finset ℤ) (h : S.card = 52) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (100 ∣ (a + b) ∨ 100 ∣ (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_or_diff_divisible_by_100_l741_74107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_f_set_characterization_l741_74127

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^3 - 8 else -x^3 - 8

theorem f_even : ∀ x, f x = f (-x) := by sorry

theorem f_set_characterization :
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_f_set_characterization_l741_74127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l741_74112

open Real

noncomputable def f (x : ℝ) := 7 * sin (x - π/6)

theorem f_increasing_on_interval :
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < π/2 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l741_74112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_center_l741_74123

-- Define the basic structures
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Define the given conditions
def touch_internally (Ω ω : Circle) (A : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def is_chord (C D : EuclideanSpace ℝ (Fin 2)) (Ω : Circle) : Prop := sorry

def is_tangent_point (B : EuclideanSpace ℝ (Fin 2)) (CD : Set (EuclideanSpace ℝ (Fin 2))) (ω : Circle) : Prop := sorry

def not_diameter (A B : EuclideanSpace ℝ (Fin 2)) (ω : Circle) : Prop := sorry

def is_midpoint (M A B : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def on_circumcircle (P : EuclideanSpace ℝ (Fin 2)) (triangle : Fin 3 → EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Theorem statement
theorem circle_through_center 
  (Ω ω : Circle) (A B C D M : EuclideanSpace ℝ (Fin 2)) (O : EuclideanSpace ℝ (Fin 2)) :
  touch_internally Ω ω A →
  is_chord C D Ω →
  is_tangent_point B {C, D} ω →
  not_diameter A B ω →
  is_midpoint M A B →
  O = ω.center →
  on_circumcircle O (fun i => [C, M, D].get i) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_center_l741_74123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_α_in_third_quadrant_sin_α_tan_half_α_l741_74177

-- Define the angle α
noncomputable def α : ℝ := Real.pi + Real.arcsin (-24/25)

-- Define that α is in the third quadrant
theorem α_in_third_quadrant : 2 * Real.pi < α ∧ α < 2 * Real.pi + Real.pi / 2 := by
  sorry

-- Define sin(α)
theorem sin_α : Real.sin α = -24/25 := by
  sorry

-- Theorem to prove
theorem tan_half_α : Real.tan (α/2) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_α_in_third_quadrant_sin_α_tan_half_α_l741_74177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l741_74169

open Real

-- Define the function f
noncomputable def f (x : ℝ) := 2 * (sin (π / 4 + x))^2 + sqrt 3 * cos (2 * x) - 1

-- Part 1
theorem part_one : ∃ x₀ ∈ Set.Ioo 0 (π / 3), f x₀ = 1 → x₀ = π / 4 := by sorry

-- Part 2
def condition_p (x : ℝ) := x ∈ Set.Icc (π / 6) (5 * π / 6)

def condition_q (x m : ℝ) := -3 < f x - m ∧ f x - m < sqrt 3

theorem part_two : 
  (∃ m : ℝ, ∀ x, condition_p x → condition_q x m) → 
  (∃ m : ℝ, m ∈ Set.Ioo 0 1 ∧ ∀ x, condition_p x → condition_q x m) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l741_74169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_on_domain_l741_74187

/-- The function f(x) defined in terms of k -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 
  ((k + 1) * x^2 + (k + 3) * x + (2 * k - 8)) / 
  ((2 * k - 1) * x^2 + (k + 1) * x + (k - 4))

/-- The domain D of f(x) -/
def D (k : ℝ) : Set ℝ :=
  {x : ℝ | (2 * k - 1) * x^2 + (k + 1) * x + (k - 4) ≠ 0}

/-- The theorem stating the conditions for f(x) > 0 for all x in D -/
theorem f_positive_on_domain (k : ℝ) : 
  (∀ x ∈ D k, f k x > 0) ↔ 
  (k = 1 ∨ k > (15 + 16 * Real.sqrt 2) / 7 ∨ k < (15 - 16 * Real.sqrt 2) / 7) := by
  sorry

#check f_positive_on_domain

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_on_domain_l741_74187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_arrangements_count_l741_74188

def balloon_arrangements : ℕ := 1260

theorem balloon_arrangements_count :
  let total_letters : ℕ := 7
  let repeating_L : ℕ := 2
  let repeating_O : ℕ := 2
  balloon_arrangements = (Nat.factorial total_letters) / (Nat.factorial repeating_L * Nat.factorial repeating_O) ∧
  balloon_arrangements = 1260 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_arrangements_count_l741_74188
