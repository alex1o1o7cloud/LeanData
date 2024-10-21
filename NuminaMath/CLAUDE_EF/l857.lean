import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_eq_eight_l857_85770

def count_pairs : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let (a, b) := p
    a + b = 915 ∧ Nat.gcd a b = 61 ∧ a > 0 ∧ b > 0
  ) (Finset.product (Finset.range 916) (Finset.range 916))).card

theorem count_pairs_eq_eight : count_pairs = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_eq_eight_l857_85770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_proof_l857_85705

-- Define the cost price and loss percent
def cost_price : ℚ := 560
def loss_percent : ℚ := 39285714285714285 / 1000000000000000

-- Define the selling price function
noncomputable def selling_price (cp : ℚ) (lp : ℚ) : ℚ :=
  cp * (1 - lp / 100)

-- Theorem statement
theorem selling_price_proof :
  ⌊selling_price cost_price loss_percent⌋ = 340 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_proof_l857_85705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_reduction_l857_85756

/-- Given a total of 25 cars initially, if 18 cars go out and 12 cars come in, 
    the reduction in the number of cars is 6. -/
theorem car_reduction (total : ℕ) (out : ℕ) (in_cars : ℕ) : 
  total = 25 → out = 18 → in_cars = 12 → out - in_cars = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_reduction_l857_85756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_has_most_symmetry_lines_l857_85755

/-- Represents the number of lines of symmetry for a shape. -/
inductive LinesOfSymmetry
  | Finite : ℕ → LinesOfSymmetry
  | Infinite : LinesOfSymmetry

/-- A shape with its number of lines of symmetry. -/
structure Shape where
  name : String
  symmetry_lines : LinesOfSymmetry

/-- The set of shapes we're comparing. -/
def shapes : List Shape :=
  [⟨"Regular Pentagon", LinesOfSymmetry.Finite 5⟩,
   ⟨"Isosceles Triangle", LinesOfSymmetry.Finite 1⟩,
   ⟨"Circle", LinesOfSymmetry.Infinite⟩,
   ⟨"Rectangle", LinesOfSymmetry.Finite 2⟩,
   ⟨"Parallelogram", LinesOfSymmetry.Finite 0⟩]

/-- Compares two LinesOfSymmetry values. -/
def linesOfSymmetryLt : LinesOfSymmetry → LinesOfSymmetry → Prop
  | LinesOfSymmetry.Finite n, LinesOfSymmetry.Finite m => n < m
  | LinesOfSymmetry.Finite _, LinesOfSymmetry.Infinite => True
  | LinesOfSymmetry.Infinite, _ => False

/-- States that the circle has more lines of symmetry than any other shape in the list. -/
theorem circle_has_most_symmetry_lines : 
  ∀ s ∈ shapes, s.name ≠ "Circle" → 
    linesOfSymmetryLt s.symmetry_lines LinesOfSymmetry.Infinite := by
  sorry

#check circle_has_most_symmetry_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_has_most_symmetry_lines_l857_85755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_existence_l857_85790

def Point := ℝ × ℝ

structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

def circle_equation (c : Circle) (x y : ℝ) : ℝ :=
  x^2 + y^2 + c.D * x + c.E * y + c.F

def passes_through (c : Circle) (p : Point) : Prop :=
  circle_equation c p.1 p.2 = 0

def cuts_chord_on_x_axis (c : Circle) (length : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, circle_equation c x₁ 0 = 0 ∧ circle_equation c x₂ 0 = 0 ∧ |x₁ - x₂| = length

theorem circle_existence :
  ∃ c₁ c₂ : Circle,
    passes_through c₁ (1, 2) ∧
    passes_through c₁ (3, 4) ∧
    cuts_chord_on_x_axis c₁ 6 ∧
    passes_through c₂ (1, 2) ∧
    passes_through c₂ (3, 4) ∧
    cuts_chord_on_x_axis c₂ 6 ∧
    c₁ ≠ c₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_existence_l857_85790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_or_right_angled_l857_85703

/-- 
If in triangle ABC, the ratio of the tangents of two angles is equal to the ratio 
of the squares of the sines of the same angles, then the triangle is either 
isosceles or right-angled.
-/
theorem triangle_isosceles_or_right_angled 
  (A B C : ℝ) 
  (triangle_sum : A + B + C = π) 
  (angle_ratio : Real.tan A / Real.tan B = (Real.sin A) ^ 2 / (Real.sin B) ^ 2) : 
  A = B ∨ A + B = π / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_or_right_angled_l857_85703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_exterior_angle_l857_85782

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- All sides are equal in length
  sides_equal : True
  -- All angles are equal in size
  angles_equal : True

/-- The measure of an exterior angle of a regular polygon -/
noncomputable def exterior_angle_measure (n : ℕ) : ℝ := 360 / n

/-- Theorem: The measure of an exterior angle of a regular polygon with n sides is 360/n -/
theorem regular_polygon_exterior_angle (n : ℕ) (p : RegularPolygon n) :
  exterior_angle_measure n = 360 / n := by
  -- Proof is omitted
  sorry

#check regular_polygon_exterior_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_exterior_angle_l857_85782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_operation_solution_l857_85753

noncomputable def star_operation (v : ℝ) : ℝ := v - v / 3

theorem star_operation_solution :
  ∀ v : ℝ, star_operation (star_operation v) = 12 → v = 27 := by
  intro v
  intro h
  -- The proof steps would go here
  sorry

#check star_operation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_operation_solution_l857_85753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_probability_l857_85723

/-- Represents a player in the coin game -/
structure Player where
  coins : ℕ

/-- Represents the state of the game -/
structure GameState where
  players : List Player
  rounds : ℕ

/-- Represents the possible ball colors -/
inductive BallColor where
  | Green
  | Red
  | Yellow
  | White

/-- Defines the initial game state -/
def initialGameState : GameState :=
  { players := List.replicate 5 { coins := 5 },
    rounds := 0 }

/-- Calculates the probability of a specific ball draw configuration -/
noncomputable def drawProbability : ℝ := 1 / 30

/-- Calculates the probability of no net change in coins after one round -/
noncomputable def noChangeProb : ℝ := 1 / 3

/-- The main theorem to prove -/
theorem coin_game_probability :
  ∀ (finalState : GameState),
    finalState.rounds = 4 →
    (∀ player ∈ finalState.players, player.coins = 5) →
    (noChangeProb ^ 4 : ℝ) = 1 / 81 := by
  sorry

#eval "Coin game probability theorem is defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_probability_l857_85723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_cosine_value_l857_85766

-- Define the function f
noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (2 * x) * Real.cos φ + Real.cos (2 * x) * Real.sin φ

-- State the theorem
theorem function_and_cosine_value 
  (φ : ℝ) 
  (h1 : 0 < φ) 
  (h2 : φ < π) 
  (h3 : f (π / 4) φ = -Real.sqrt 3 / 2) 
  (α : ℝ) 
  (h4 : π / 2 < α) 
  (h5 : α < π) 
  (h6 : f (α / 2 - π / 3) φ = 5 / 13) : 
  (∀ x : ℝ, f x φ = Real.sin (2 * x + 5 * π / 6)) ∧ 
  Real.cos α = (5 - 12 * Real.sqrt 3) / 26 := by
  sorry

#check function_and_cosine_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_cosine_value_l857_85766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_star_four_l857_85787

-- Define the # operation
def hash (r s : ℝ) : ℝ :=
  sorry

-- Define the * operation
def star (r s : ℝ) : ℝ := hash r s - s

-- Properties of the # operation
axiom hash_zero (r : ℝ) : hash r 0 = r + 1
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + s + 2

-- Theorem to prove
theorem eight_star_four : star 8 4 = 85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_star_four_l857_85787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_2022_fixed_point_l857_85722

noncomputable def g₁ (x : ℝ) : ℝ := (3 * x - 8) / (5 * x + 2)

noncomputable def g : ℕ → (ℝ → ℝ)
  | 0 => id
  | 1 => g₁
  | n + 2 => g₁ ∘ g (n + 1)

theorem g_2022_fixed_point (x : ℝ) (h : x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6) : 
  g 2022 x = x - 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_2022_fixed_point_l857_85722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_power_product_l857_85768

theorem fraction_power_product : 
  (4 / 5 : ℚ) ^ 10 * (2 / 3 : ℚ) ^ (-4 : ℤ) = 84934656 / 156250000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_power_product_l857_85768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_three_element_set_l857_85763

theorem subsets_of_three_element_set :
  let S : Finset Int := {-1, 0, 1}
  Finset.powerset S |>.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_three_element_set_l857_85763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_simplified_polynomial_l857_85761

theorem sum_of_coefficients_simplified_polynomial (d : ℝ) (h : d ≠ 0) :
  let p : Polynomial ℝ := (16 * X + 15 + 18 * X^2 + 3 * X^3) + (4 * X + 2 + X^2 + 2 * X^3)
  let simplified : Polynomial ℝ := 5 * X^3 + 19 * X^2 + 20 * X + 17
  p = simplified ∧ (5 : ℝ) + 19 + 20 + 17 = 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_simplified_polynomial_l857_85761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l857_85762

noncomputable def f (x : ℝ) : ℝ := (x - 1) * (x - 3) * (x - 4) / ((x - 2) * (x - 5) * (x - 6))

theorem inequality_solution :
  ∀ x : ℝ, f x > 0 ↔ x ∈ Set.Iio 1 ∪ Set.Ioo 3 4 ∪ Set.Ioo 4 5 ∪ Set.Ioo 5 6 ∪ Set.Ioi 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l857_85762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_from_circle_to_line_l857_85772

noncomputable def line_l (x : ℝ) : ℝ := Real.sqrt 3 * x

def on_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

noncomputable def min_distance : ℝ := Real.sqrt 3 - 1

theorem min_distance_from_circle_to_line :
  ∀ (x y : ℝ), on_circle x y →
  (∃ (d : ℝ), d ≥ 0 ∧ d = min_distance ∧
    ∀ (x' y' : ℝ), on_circle x' y' →
      Real.sqrt ((x' - x)^2 + (y' - line_l x')^2) ≥ d) :=
by
  sorry

#check min_distance_from_circle_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_from_circle_to_line_l857_85772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_120_90_l857_85777

/-- The number of common divisors of 120 and 90 is 16 -/
theorem common_divisors_120_90 : 
  (Finset.filter (fun x => x ∣ 120 ∧ x ∣ 90) (Finset.range 121)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_120_90_l857_85777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_bulbs_possible_l857_85769

/-- Represents the state of all light bulbs on the number line -/
def LightState := ℤ → Bool

/-- The template set S -/
def S : Finset ℤ := sorry

/-- Applies the template S at a given position -/
def applyTemplate (state : LightState) (pos : ℤ) : LightState :=
  fun i => if (i - pos) ∈ S then !state i else state i

/-- Checks if exactly two bulbs are lit in the given state -/
def exactlyTwoLit (state : LightState) : Prop :=
  ∃ a b : ℤ, a ≠ b ∧ state a ∧ state b ∧ ∀ c : ℤ, c ≠ a ∧ c ≠ b → ¬state c

/-- The initial state where all bulbs are off -/
def initialState : LightState := fun _ => false

/-- The main theorem stating that it's possible to achieve a state with exactly two bulbs lit -/
theorem two_bulbs_possible : 
  ∃ (sequence : List ℤ), exactlyTwoLit (sequence.foldl applyTemplate initialState) := by
  sorry

#check two_bulbs_possible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_bulbs_possible_l857_85769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_satisfies_simple_interest_l857_85700

/-- Calculates the principal amount given the final amount, interest rate, and time period. -/
noncomputable def calculate_principal (A : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  A / (1 + r * t)

/-- Theorem stating that the calculated principal satisfies the simple interest formula. -/
theorem principal_satisfies_simple_interest 
  (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) 
  (h_A : A = 1120) 
  (h_r : r = 0.06) 
  (h_t : t = 2.4) 
  (h_P : P = calculate_principal A r t) :
  A = P * (1 + r * t) := by
  sorry

/-- Compute the principal for the given problem -/
def problem_principal : ℚ :=
  (1120 : ℚ) / (1 + (6 : ℚ) / 100 * (12 : ℚ) / 5)

#eval problem_principal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_satisfies_simple_interest_l857_85700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l857_85736

-- Define the work rates for A and B
noncomputable def work_rate_A (x : ℝ) : ℝ := 1 / x
noncomputable def work_rate_B : ℝ := 1 / 60

-- Define the combined work rate
noncomputable def combined_work_rate (x : ℝ) : ℝ := work_rate_A x + work_rate_B

-- State the theorem
theorem work_completion_time (x : ℝ) (hx : x > 0) :
  combined_work_rate x = 1 / 24 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l857_85736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_2_14_in_terms_of_a_and_b_l857_85796

theorem log_2_14_in_terms_of_a_and_b (a b : ℝ) 
  (ha : Real.log 3 / Real.log 2 = a) 
  (hb : Real.log 7 / Real.log 3 = b) : 
  Real.log 14 / Real.log 2 = 1 + a * b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_2_14_in_terms_of_a_and_b_l857_85796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tangent_circle_unique_tangent_circle_proof_l857_85746

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are tangent to each other -/
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Theorem: There exists exactly one circle of radius 2 tangent to three unit circles that are mutually tangent -/
theorem unique_tangent_circle : Prop :=
  ∀ C1 C2 C3 : Circle,
    C1.radius = 1 ∧ C2.radius = 1 ∧ C3.radius = 1 ∧
    are_tangent C1 C2 ∧ are_tangent C2 C3 ∧ are_tangent C3 C1 →
    ∃! C : Circle, C.radius = 2 ∧ are_tangent C C1 ∧ are_tangent C C2 ∧ are_tangent C C3

/-- Proof of the theorem -/
theorem unique_tangent_circle_proof : unique_tangent_circle := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tangent_circle_unique_tangent_circle_proof_l857_85746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jelly_cost_l857_85754

theorem jelly_cost (N B J : ℕ) (h1 : N > 1) (h2 : 3 * N * B + 6 * N * J = 336) : 
  6 * N * J = 210 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jelly_cost_l857_85754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_arrangements_eq_72_l857_85757

/-- Represents the number of boys in the arrangement -/
def num_boys : ℕ := 3

/-- Represents the number of girls in the arrangement -/
def num_girls : ℕ := 4

/-- Represents the total number of people in the arrangement -/
def total_people : ℕ := num_boys + num_girls

/-- Represents the condition that one specific boy must be adjacent to one specific girl -/
def adjacent_pair : ℕ := 1

/-- Represents the number of remaining boys after considering the adjacent pair -/
def remaining_boys : ℕ := num_boys - adjacent_pair

/-- Represents the number of remaining girls after considering the adjacent pair -/
def remaining_girls : ℕ := num_girls - adjacent_pair

/-- Represents the number of positions for the adjacent pair -/
def adjacent_pair_positions : ℕ := total_people - 1

/-- Represents the number of ways to arrange the remaining boys and girls -/
def remaining_arrangements : ℕ := (remaining_girls + 1) * remaining_boys

/-- The main theorem stating that the total number of arrangements is 72 -/
theorem total_arrangements_eq_72 : 
  adjacent_pair_positions * remaining_arrangements = 72 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_arrangements_eq_72_l857_85757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_below_line_iff_a_in_range_l857_85785

/-- The function f(x) = (a - 1/2)x^2 + ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 + Real.log x

/-- The theorem stating the equivalence between the function being below 2ax and the range of a -/
theorem function_below_line_iff_a_in_range :
  ∀ (a : ℝ), (∀ (x : ℝ), x > 1 → f a x < 2 * a * x) ↔ a ∈ Set.Icc (-1/2) (1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_below_line_iff_a_in_range_l857_85785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_proof_l857_85798

-- Define the circle circumference
noncomputable def circle_circumference : ℝ := 100

-- Define the central angle in degrees
noncomputable def central_angle : ℝ := 45

-- Define the arc length
noncomputable def arc_length : ℝ := circle_circumference * (central_angle / 360)

-- Theorem statement
theorem arc_length_proof : arc_length = 12.5 := by
  -- Unfold the definitions
  unfold arc_length circle_circumference central_angle
  -- Simplify the expression
  simp [mul_div_assoc]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_proof_l857_85798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_two_isosceles_triangles_l857_85714

/-- A set of 5 positive real numbers where no three can form a triangle -/
def NoTriangleSet : Set (Set ℝ) := {S | ∃ a b c d e : ℝ, S = {a, b, c, d, e} ∧
  (∀ x ∈ S, x > 0) ∧
  (∀ p q r, p ∈ S → q ∈ S → r ∈ S → p + q ≤ r ∨ p + r ≤ q ∨ q + r ≤ p)}

/-- Predicate to check if three numbers form an isosceles triangle -/
def IsIsoscelesTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ ((a = b ∧ a + b > c) ∨ (a = c ∧ a + c > b) ∨ (b = c ∧ b + c > a))

/-- Main theorem statement -/
theorem exists_two_isosceles_triangles (S : Set ℝ) (hS : S ∈ NoTriangleSet) :
  ∃ (x : ℝ) (y z : ℝ), x ∈ S ∧ y > 0 ∧ z > 0 ∧ y + z = x ∧
    ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ), 
      {a₁, b₁, c₁, a₂, b₂, c₂} = (S \ {x}) ∪ {y, z} ∧
      IsIsoscelesTriangle a₁ b₁ c₁ ∧ IsIsoscelesTriangle a₂ b₂ c₂ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_two_isosceles_triangles_l857_85714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_is_negative_88_l857_85780

def sequence_with_unknown (x : ℤ) : List ℤ := [8, 86, 2, x, -12]

theorem fourth_term_is_negative_88 :
  ∃ (x : ℤ), sequence_with_unknown x = [8, 86, 2, x, -12] ∧ x = -88 :=
by
  use -88
  apply And.intro
  · rfl
  · rfl

#check fourth_term_is_negative_88

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_is_negative_88_l857_85780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_range_l857_85717

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2)*x + 2

theorem monotonic_increasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc 4 8 ∧ a ≠ 8 :=
by
  sorry

#check monotonic_increasing_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_range_l857_85717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_and_inverse_are_false_l857_85794

-- Define the properties of triangles
def IsEquilateral (t : Triangle) : Prop := sorry
def IsIsosceles (t : Triangle) : Prop := sorry

-- Define a triangle type
structure Triangle : Type where
  -- You can add fields here if needed, e.g.:
  -- sides : Fin 3 → ℝ

-- State the original implication
axiom original_statement : ∀ t : Triangle, IsEquilateral t → IsIsosceles t

-- State the theorem to be proved
theorem converse_and_inverse_are_false :
  (¬ ∀ t : Triangle, IsIsosceles t → IsEquilateral t) ∧
  (¬ ∀ t : Triangle, ¬IsEquilateral t → ¬IsIsosceles t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_and_inverse_are_false_l857_85794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_boys_in_weight_range_l857_85742

/-- Estimates the number of boys in a population within a specific weight range based on a sample. -/
theorem estimate_boys_in_weight_range 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (sample_proportion : ℚ) : 
  total_population > 0 → 
  sample_size > 0 → 
  sample_size ≤ total_population → 
  0 ≤ sample_proportion → 
  sample_proportion ≤ 1 → 
  (sample_proportion * ↑total_population : ℚ) = 240 → 
  ⌊sample_proportion * ↑total_population⌋ = 240 := by 
  sorry

#check estimate_boys_in_weight_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_boys_in_weight_range_l857_85742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_external_angle_bisector_length_l857_85741

-- Define the triangle ABC and point D
variable (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h1 : A ≠ B ∧ B ≠ C ∧ C ≠ A)
variable (h2 : IsExternalAngleBisector A B C D)
variable (h3 : SegmentBetween B A D)
variable (m n : ℝ)
variable (h4 : dist B D - dist B C = m)
variable (h5 : dist A C + dist A D = n)

-- The theorem to prove
theorem triangle_external_angle_bisector_length :
  dist C D = Real.sqrt (m * n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_external_angle_bisector_length_l857_85741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_sum_dice_rolls_l857_85734

-- Define the number of coins
def num_coins : ℕ := 3

-- Define the probability of getting heads on a fair coin
def prob_heads : ℚ := 1/2

-- Define the number of sides on the dice
def sides_head_die : ℕ := 6
def sides_tail_die : ℕ := 4

-- Define a function to represent the probability of rolling an even number on a die
def prob_even_roll (sides : ℕ) : ℚ := (sides / 2) / sides

-- Define a function to calculate the probability of getting k heads in n coin tosses
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * prob_heads^k * (1 - prob_heads)^(n - k)

-- State the theorem
theorem prob_even_sum_dice_rolls : 
  ∃ P : ℚ, P = (Finset.sum (Finset.range (num_coins + 1)) (λ k => 
    prob_k_heads num_coins k * 
    (prob_even_roll sides_head_die)^k * 
    (prob_even_roll sides_tail_die)^(num_coins - k))) := 
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_sum_dice_rolls_l857_85734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_seven_exists_l857_85713

theorem divisible_by_seven_exists (S : Finset ℤ) (h : Finset.card S = 7) :
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 7 ∣ (a^2 + b^2 + c^2 - a*b - b*c - a*c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_seven_exists_l857_85713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_zero_is_local_min_when_a_eq_one_number_of_zeros_zeros_conditions_l857_85715

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x - a * x

-- Statement 1: f(x) is increasing on ℝ iff a ∈ (-∞, -1/e²]
theorem f_increasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≤ -Real.exp (-2) := by sorry

-- Statement 2: When a = 1, x = 0 is a local minimum point of f(x)
theorem zero_is_local_min_when_a_eq_one :
  ∃ δ > 0, ∀ x : ℝ, |x| < δ → f 1 0 ≤ f 1 x := by sorry

-- Statement 3: Number of zeros of f(x)
theorem number_of_zeros (a : ℝ) :
  (∃! x : ℝ, f a x = 0) ∨
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z : ℝ, f a z = 0 → z = x ∨ z = y) := by sorry

-- Additional theorem to specify the conditions for each case
theorem zeros_conditions (a : ℝ) :
  ((a ≤ 0 ∨ a = 1) → ∃! x : ℝ, f a x = 0) ∧
  ((0 < a ∧ a < 1) ∨ a > 1 → ∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z : ℝ, f a z = 0 → z = x ∨ z = y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_zero_is_local_min_when_a_eq_one_number_of_zeros_zeros_conditions_l857_85715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l857_85758

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi/6) + Real.cos x

theorem function_properties (α : ℝ) 
  (h1 : 0 < α ∧ α < Real.pi/2) 
  (h2 : f (α + Real.pi/3) = 4/5) : 
  (∃ T : ℝ, T > 0 ∧ T = 2*Real.pi ∧ ∀ x : ℝ, f (x + T) = f x) ∧ 
  Real.tan (α - Real.pi/4) = -1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l857_85758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_probability_l857_85740

/-- Represents a cell on the plane --/
structure Cell where
  m : ℤ
  n : ℤ

/-- The probability of a coin falling completely within a cell (m, n) where m + n is divisible by 3 --/
noncomputable def probability_coin_in_cell (a : ℝ) (r : ℝ) : ℝ :=
  (a - 2*r)^2 / (3*a^2)

theorem coin_probability 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a > 0) 
  (h2 : r > 0) 
  (h3 : r < a/2) :
  probability_coin_in_cell a r = 
    (a - 2*r)^2 / (3*a^2) :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_probability_l857_85740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_l857_85793

/-- A regular pentagon. -/
structure RegularPentagon where
  vertices : Finset (ℕ × ℕ)
  is_regular : vertices.card = 5

/-- A line segment in the pentagon. -/
inductive Segment
  | Side
  | Diagonal

/-- The color of a line segment. -/
inductive Color
  | Blue
  | Red

/-- The state of the game. -/
structure GameState where
  pentagon : RegularPentagon
  colored_segments : List (Segment × Color)

/-- A player in the game. -/
inductive Player
  | First
  | Second

/-- Convert a player to their corresponding color. -/
def Player.toColor : Player → Color
  | First => Color.Blue
  | Second => Color.Red

/-- A strategy for a player. -/
def Strategy := GameState → Segment

/-- Determines if a player has won the game. -/
def has_won (state : GameState) (player : Player) : Prop :=
  ∃ (v1 v2 v3 : ℕ × ℕ),
    v1 ∈ state.pentagon.vertices ∧
    v2 ∈ state.pentagon.vertices ∧
    v3 ∈ state.pentagon.vertices ∧
    v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
    (∀ (s : Segment), (s, player.toColor) ∈ state.colored_segments)

/-- The main theorem stating that the first player has a winning strategy. -/
theorem first_player_winning_strategy :
  ∃ (s : Strategy), ∀ (game : GameState),
    has_won game Player.First := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_l857_85793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drews_leftover_coverage_l857_85739

/-- Represents the lawn dimensions and seed coverage --/
structure LawnData where
  rect_length : ℝ
  rect_width : ℝ
  tri_base : ℝ
  tri_height : ℝ
  seed_coverage : ℝ
  tri_coverage_factor : ℝ
  bags_bought : ℕ

/-- Calculates the extra square feet that can be covered by leftover grass seed --/
noncomputable def leftover_coverage (data : LawnData) : ℝ :=
  let rect_area := data.rect_length * data.rect_width
  let tri_area := data.tri_base * data.tri_height / 2
  let adjusted_tri_area := tri_area * data.tri_coverage_factor
  let total_area := rect_area + adjusted_tri_area
  let total_coverage := data.seed_coverage * (data.bags_bought : ℝ)
  total_coverage - total_area

/-- Theorem stating the leftover coverage for Drew's lawn --/
theorem drews_leftover_coverage :
  let data : LawnData := {
    rect_length := 32,
    rect_width := 45,
    tri_base := 25,
    tri_height := 20,
    seed_coverage := 420,
    tri_coverage_factor := 1.5,
    bags_bought := 7
  }
  leftover_coverage data = 1125 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_drews_leftover_coverage_l857_85739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_art_gallery_theorem_l857_85767

/-- A non-convex polygon with n vertices -/
structure NonConvexPolygon (n : ℕ) where
  nonconvex : Bool
  vertices : Fin n → ℝ × ℝ

/-- A guard position in the polygon -/
def Guard := ℝ × ℝ

/-- Predicate to check if a set of guards can monitor the entire polygon -/
def monitors (p : NonConvexPolygon n) (guards : Finset Guard) : Prop :=
  sorry

/-- The Art Gallery Theorem for non-convex polygons -/
theorem art_gallery_theorem (n : ℕ) (p : NonConvexPolygon n) :
  ∃ (guards : Finset Guard), guards.card = ⌊(n : ℝ) / 3⌋ ∧ monitors p guards := by
  sorry

#check art_gallery_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_art_gallery_theorem_l857_85767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_midpoint_relation_l857_85764

/-- Represents a tetrahedron with edge lengths and distances between midpoints of opposite edges -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ  -- edge lengths
  x : ℝ
  y : ℝ
  z : ℝ  -- distances between midpoints of opposite edges

/-- 
Theorem: The sum of squares of all edges of a tetrahedron equals four times 
the sum of squares of distances between midpoints of opposite edges 
-/
theorem tetrahedron_edge_midpoint_relation (t : Tetrahedron) : 
  t.a^2 + t.b^2 + t.c^2 + t.d^2 + t.e^2 + t.f^2 = 4 * (t.x^2 + t.y^2 + t.z^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_midpoint_relation_l857_85764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_fraction_in_jarX_l857_85743

-- Define the capacity of Jar X
variable (C : ℝ)

-- Define the capacities and initial water levels of the jars
noncomputable def jarX_initial (C : ℝ) : ℝ := C / 2
noncomputable def jarY_capacity (C : ℝ) : ℝ := C / 2
noncomputable def jarY_initial (C : ℝ) : ℝ := jarY_capacity C / 2
noncomputable def jarZ_capacity (C : ℝ) : ℝ := C / 4
noncomputable def jarZ_initial (C : ℝ) : ℝ := jarZ_capacity C * 3 / 4

-- Define the water transfer process
noncomputable def water_in_jarZ_after_Y (C : ℝ) : ℝ := jarZ_initial C + jarY_initial C
noncomputable def excess_water (C : ℝ) : ℝ := max (water_in_jarZ_after_Y C - jarZ_capacity C) 0
noncomputable def final_water_in_jarX (C : ℝ) : ℝ := jarX_initial C + excess_water C

-- Theorem statement
theorem final_fraction_in_jarX (C : ℝ) :
  final_water_in_jarX C = 11 * C / 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_fraction_in_jarX_l857_85743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_value_l857_85709

/-- The perpendicular bisector of a line segment passes through its midpoint -/
axiom perpendicular_bisector_passes_through_midpoint {A B M : ℝ × ℝ} {c : ℝ} :
  (∀ (x y : ℝ), x + y = c ↔ ((x, y) ∈ Set.Ioo A B ∧ dist (x, y) A = dist (x, y) B)) →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  M.1 + M.2 = c

/-- The value of c for the perpendicular bisector of the line segment from (2,5) to (8,11) -/
theorem perpendicular_bisector_value (c : ℝ) :
  (∀ (x y : ℝ), x + y = c ↔ ((x, y) ∈ Set.Ioo (2, 5) (8, 11) ∧ 
    dist (x, y) (2, 5) = dist (x, y) (8, 11))) →
  c = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_value_l857_85709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_x_correct_l857_85711

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The x-coordinate of the point where a common tangent line to two circles intersects the x-axis -/
noncomputable def tangentIntersectionX (c1 c2 : Circle) : ℝ :=
  54 / 11

theorem tangent_intersection_x_correct (c1 c2 : Circle) :
  c1.center = (0, 0) →
  c1.radius = 3 →
  c2.center = (18, 0) →
  c2.radius = 8 →
  tangentIntersectionX c1 c2 = 54 / 11 := by
  sorry

#check tangent_intersection_x_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_x_correct_l857_85711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l857_85708

theorem complex_number_problem (z : ℂ) :
  (z.im ≠ 0) →
  (z.re = 0) →
  ((z + 2)^2 - 8*Complex.I).re = 0 →
  z = -2*Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l857_85708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_twenty_consecutive_integers_l857_85791

/-- The smallest possible sum of 20 consecutive positive integers 
    that is twice a perfect square is 450. -/
theorem smallest_sum_of_twenty_consecutive_integers : ℕ := by
  -- Define the property for the sum of 20 consecutive integers
  let is_valid_sum (n : ℕ) : Prop :=
    ∃ (k : ℕ), n = 10 * (2 * k + 19) ∧ ∃ (m : ℕ), n = 2 * m^2

  -- State that 450 satisfies the property
  have property_450 : is_valid_sum 450 := by sorry

  -- State that 450 is the smallest such number
  have smallest : ∀ (n : ℕ), n < 450 → ¬(is_valid_sum n) := by sorry

  -- Conclude that 450 is the answer
  exact 450


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_twenty_consecutive_integers_l857_85791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l857_85771

theorem gcd_problem (b : ℤ) (h1 : ∃ k : ℤ, b = 9 * (2 * k + 1)) :
  Int.gcd (8 * b^2 + 81 * b + 289) (4 * b + 17) = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l857_85771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunscreen_calculation_l857_85707

/-- Calculates the amount of sunscreen needed per application given the total duration,
    reapplication interval, and amount of sunscreen in a bottle. -/
noncomputable def sunscreenPerApplication (totalDuration reapplyInterval bottleAmount : ℝ) : ℝ :=
  bottleAmount / (totalDuration / reapplyInterval)

/-- Theorem stating that for a 16-hour beach visit, reapplying every 2 hours,
    and a 12-ounce bottle, 1.5 ounces of sunscreen are needed per application. -/
theorem sunscreen_calculation :
  sunscreenPerApplication 16 2 12 = 1.5 := by
  -- Unfold the definition of sunscreenPerApplication
  unfold sunscreenPerApplication
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunscreen_calculation_l857_85707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_l857_85730

/-- The quadratic function f(x) = -x^2 + 2x + m^2 -/
def f (x m : ℝ) : ℝ := -x^2 + 2*x + m^2

/-- The maximum value of f(x) -/
def max_value (m : ℝ) : ℝ := m^2 + 1

theorem quadratic_max_value (m : ℝ) : 
  (∀ x, f x m ≤ 3) ∧ (∃ x, f x m = 3) → m = Real.sqrt 2 ∨ m = -Real.sqrt 2 := by
  sorry

#check quadratic_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_l857_85730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approx_l857_85776

/-- Represents a segment of the cyclist's trip -/
structure Segment where
  distance : ℝ
  baseSpeed : ℝ
  elevationChange : ℝ
  windSpeed : ℝ
  windType : String

/-- Calculates the effective speed for a segment -/
noncomputable def effectiveSpeed (s : Segment) : ℝ :=
  match s.windType with
  | "headwind" => s.baseSpeed - s.windSpeed
  | "tailwind" => s.baseSpeed + s.windSpeed
  | _ => s.baseSpeed

/-- Calculates the time taken for a segment -/
noncomputable def segmentTime (s : Segment) : ℝ :=
  s.distance / effectiveSpeed s

/-- The cyclist's trip -/
def trip : List Segment := [
  { distance := 12, baseSpeed := 13, elevationChange := 150, windSpeed := 0, windType := "none" },
  { distance := 18, baseSpeed := 16, elevationChange := -200, windSpeed := 5, windType := "headwind" },
  { distance := 25, baseSpeed := 20, elevationChange := 0, windSpeed := 8, windType := "crosswind" },
  { distance := 35, baseSpeed := 25, elevationChange := 300, windSpeed := 10, windType := "tailwind" },
  { distance := 50, baseSpeed := 22, elevationChange := 0, windSpeed := 10, windType := "headwind" }
]

/-- Theorem: The average speed for the entire trip is approximately 15.59 km/hr -/
theorem average_speed_approx (ε : ℝ) (hε : ε > 0) :
  ∃ (avg_speed : ℝ), 
    |avg_speed - 15.59| < ε ∧
    avg_speed = (List.sum (List.map Segment.distance trip)) / (List.sum (List.map segmentTime trip)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approx_l857_85776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l857_85752

-- Helper definitions
def is_median (a b c x : ℝ) : Prop := sorry

def triangle_area (a b c : ℝ) : ℝ := sorry

-- Main theorem
theorem third_median_length (a b c : ℝ) (m₁ m₂ : ℝ) (area : ℝ) 
  (h₁ : m₁ = 5) (h₂ : m₂ = 7) (h₃ : area = 6 * Real.sqrt 10) :
  ∃ (m₃ : ℝ), m₃ = 9 ∧ 
    is_median a b c m₃ ∧ 
    is_median b c a m₁ ∧ 
    is_median c a b m₂ ∧
    triangle_area a b c = area :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l857_85752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_ln_to_line_l857_85735

/-- The shortest distance from a point on the curve y = ln(x) to the line y = x + 1 is √2 -/
theorem shortest_distance_ln_to_line : 
  ∃ (d : ℝ), d = Real.sqrt 2 ∧ 
  ∀ (x y : ℝ), y = Real.log x → 
  ∀ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y → 
  d ≤ Real.sqrt ((p.1 - x)^2 + (p.2 - (x + 1))^2) ∧
  ∃ (x₀ y₀ : ℝ), y₀ = Real.log x₀ ∧ 
  d = Real.sqrt ((x₀ - x)^2 + (y₀ - (x + 1))^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_ln_to_line_l857_85735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l857_85759

theorem remainder_theorem (x y z : ℕ) 
  (hx : x % 23 = 9)
  (hy : y % 29 = 15)
  (hz : z % 37 = 12) :
  ∃ (k m n : ℤ),
    (x : ℤ) = 23 * k + 9 ∧
    (y : ℤ) = 29 * m + 15 ∧
    (z : ℤ) = 37 * n + 12 ∧
    (3 * x + 7 * y - 5 * z) % 31517 = (69 * k + 203 * m - 185 * n + 72) % 31517 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l857_85759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l857_85792

-- Define the problem statement
theorem power_equation (y : ℝ) : (100 : ℝ)^y = 16 → (100 : ℝ)^(-y) = 1/6.31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l857_85792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l857_85774

/-- Given an arithmetic sequence {a_n} with first term a_1 and common difference d,
    S_n represents the sum of the first n terms. -/
noncomputable def S (n : ℕ) (a_1 d : ℝ) : ℝ := n * a_1 + (n * (n - 1) / 2) * d

/-- Theorem stating that for the given arithmetic sequence, S_2011 = 0 -/
theorem arithmetic_sequence_sum (a_1 d : ℝ) 
  (h1 : a_1 = -2010)
  (h2 : S 2009 a_1 d / 2009 - S 2007 a_1 d / 2007 = 2) :
  S 2011 a_1 d = 0 := by
  sorry

#check arithmetic_sequence_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l857_85774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_trigonometry_l857_85779

theorem acute_angle_trigonometry (α : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 3 / 5) :
  (Real.cos α = 4 / 5) ∧ (Real.cos (α + π / 6) = (4 * Real.sqrt 3 - 3) / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_trigonometry_l857_85779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l857_85702

theorem find_n : ∃ n : ℝ, (15 : ℝ)^(3*n) = (1/15 : ℝ)^(n-30) → n = 7.5 := by
  use 7.5
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l857_85702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_theorem_optimal_allocation_theorem_l857_85727

/-- The profit function for good A -/
noncomputable def profit_A (x : ℝ) : ℝ := (1/5) * x

/-- The profit function for good B -/
noncomputable def profit_B (x : ℝ) : ℝ := (3/5) * Real.sqrt x

/-- The total profit function -/
noncomputable def total_profit (x : ℝ) : ℝ := profit_A (3 - x) + profit_B x

/-- Theorem stating the maximum profit -/
theorem max_profit_theorem :
  ∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ 
  ∀ y : ℝ, y ∈ Set.Icc 0 3 → total_profit x ≥ total_profit y ∧
  total_profit x = 21/20 := by
  sorry

/-- Theorem stating the optimal allocation -/
theorem optimal_allocation_theorem :
  ∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ 
  total_profit x = 21/20 ∧
  x = 9/4 ∧ (3 - x) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_theorem_optimal_allocation_theorem_l857_85727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l857_85706

-- Define the equations
def line1 (x : ℝ) : ℝ := x + 2
def line2 (x : ℝ) : ℝ := -3*x + 4
def line3 (x : ℝ) : ℝ := -x + 2
def parabola (x : ℝ) : ℝ := -x^2 + 2

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- Function to check if a point satisfies an equation
def satisfies (p : Point) (f : ℝ → ℝ) : Prop :=
  p.2 = f p.1

-- Define the set of all intersection points
def intersectionPoints : Set Point :=
  {p : Point | (satisfies p line1 ∧ satisfies p line3) ∨
               (satisfies p line1 ∧ satisfies p parabola) ∨
               (satisfies p line2 ∧ satisfies p line3) ∨
               (satisfies p line2 ∧ satisfies p parabola)}

-- Theorem statement
theorem intersection_count :
  ∃ (s : Finset Point), s.toSet = intersectionPoints ∧ s.card = 4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l857_85706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_growth_time_l857_85701

/-- The time interval in minutes between bacterial divisions -/
def division_interval : ℕ := 15

/-- The initial number of bacteria -/
def initial_bacteria : ℕ := 1

/-- The final number of bacteria -/
def final_bacteria : ℕ := 4096

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculates the time in hours required for bacterial growth -/
noncomputable def growth_time : ℝ :=
  (Real.log (final_bacteria : ℝ) - Real.log (initial_bacteria : ℝ)) / 
  (Real.log 2 * minutes_per_hour) * division_interval

theorem bacteria_growth_time :
  growth_time = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_growth_time_l857_85701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l857_85728

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane represented by the equation y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Check if a point is on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  p.2 = l.m * p.1 + l.b

/-- Calculate the area of a circle divided by a line -/
noncomputable def dividedArea (c : Circle) (l : Line) : ℝ := sorry

/-- The main theorem -/
theorem equal_area_division :
  ∃ (l : Line), 
    pointOnLine (15, 130) l ∧ 
    |l.m| = 1 ∧
    let circles := [
      Circle.mk (10, 150) 5,
      Circle.mk (15, 130) 5,
      Circle.mk (20, 145) 5,
      Circle.mk (25, 120) 5
    ]
    (circles.map (dividedArea · l)).sum = 
    (circles.map (λ c => π * c.radius^2 - dividedArea c l)).sum := by
  sorry

#check equal_area_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l857_85728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_sum_l857_85788

noncomputable def f (a b c : ℕ) (x : ℝ) : ℝ :=
  if x > 0 then a * x + 4
  else if x = 0 then a * b
  else b * x + c

theorem piecewise_function_sum (a b c : ℕ) :
  f a b c 3 = 7 ∧ f a b c 0 = 6 ∧ f a b c (-3) = -15 →
  a + b + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_sum_l857_85788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_problem_l857_85720

/-- The chord length cut by a line from a circle -/
theorem chord_length_problem : 
  let line := {(x, y) : ℝ × ℝ | x - y + Real.sqrt 10 = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 4*x - 4*y - 1 = 0}
  let chord_length := 2 * Real.sqrt (9 - (Real.sqrt 10 / Real.sqrt 2)^2)
  chord_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_problem_l857_85720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_length_to_d_ratio_l857_85773

/-- Represents a thin, uniform rod -/
structure Rod where
  mass : ℝ
  length : ℝ

/-- The rotational inertia of a rod about its center -/
noncomputable def rotationalInertia (rod : Rod) : ℝ := (1 / 12) * rod.mass * rod.length^2

/-- Theorem stating the ratio of rod length to d -/
theorem rod_length_to_d_ratio (rod : Rod) (d : ℝ) 
    (h : rotationalInertia rod = rod.mass * d^2) :
  rod.length / d = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_length_to_d_ratio_l857_85773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_comparison_l857_85783

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_comparison : 
  triangle_area 15 15 20 < triangle_area 15 15 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_comparison_l857_85783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l857_85781

/-- The y-intercept of the line x + y + 1 = 0 is -1 -/
theorem y_intercept_of_line (x y : ℝ) : x + y + 1 = 0 → y = -1 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l857_85781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_condition_l857_85744

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the scalars
variable (m n : ℝ)

-- Define the points
variable (A B C : V)

-- Theorem statement
theorem collinear_points_condition 
  (h_not_collinear : ¬ ∃ (k : ℝ), e₁ = k • e₂)
  (h_AB : B - A = e₁ + m • e₂)
  (h_AC : C - A = n • e₁ + e₂)
  (h_collinear : ∃ (t : ℝ), B - A = t • (C - A)) :
  m * n = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_condition_l857_85744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_revolution_l857_85716

/-- The volume of the solid of revolution around the Ox axis -/
noncomputable def revolution_volume (f g : ℝ → ℝ) (a b : ℝ) : ℝ :=
  Real.pi * ∫ x in a..b, (f x)^2 - (g x)^2

/-- The upper bounding function: y = (3 - 2x) / 2 -/
noncomputable def f (x : ℝ) : ℝ := (3 - 2*x) / 2

/-- The lower bounding function: y = x^2 / 2 -/
noncomputable def g (x : ℝ) : ℝ := x^2 / 2

/-- The theorem stating the volume of the solid of revolution -/
theorem volume_of_revolution :
  revolution_volume f g (-3) 1 = 272 / 15 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_revolution_l857_85716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_needed_for_room_l857_85726

/-- The number of square floor tiles needed to cover a rectangular room -/
def num_tiles (room_length room_width tile_side_length : ℚ) : ℕ :=
  (room_length * room_width * 100 / (tile_side_length * tile_side_length)).ceil.toNat

/-- Theorem stating that 600 square floor tiles are needed to cover the given room -/
theorem tiles_needed_for_room : num_tiles 9 6 3 = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_needed_for_room_l857_85726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_benson_ticket_cost_l857_85745

/-- Calculates the total cost of concert tickets with a discount applied to tickets exceeding a threshold. -/
noncomputable def calculate_ticket_cost (ticket_price : ℝ) (num_tickets : ℕ) (discount_percent : ℝ) (discount_threshold : ℕ) : ℝ :=
  let full_price_tickets := min num_tickets discount_threshold
  let discounted_tickets := num_tickets - full_price_tickets
  let discounted_price := ticket_price * (1 - discount_percent / 100)
  ticket_price * (full_price_tickets : ℝ) + discounted_price * ((num_tickets - full_price_tickets) : ℝ)

/-- Proves that Mr. Benson paid $476 for the concert tickets. -/
theorem benson_ticket_cost : 
  calculate_ticket_cost 40 12 5 10 = 476 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_benson_ticket_cost_l857_85745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l857_85731

noncomputable section

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle conditions
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  -- Given conditions
  (cos A * cos B = sin A * sin B - sqrt 2 / 2) ∧
  (b = 4) ∧
  (1/2 * a * b * sin C = 6) →
  -- Conclusions
  (C = π/4) ∧
  (c^2 = 10) ∧
  (cos (2*B - C) = sqrt 2 / 10) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l857_85731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_daily_profit_at_14_yuan_l857_85799

/-- Represents the daily profit function for a product sale -/
noncomputable def daily_profit (initial_cost initial_price initial_sales : ℝ) 
  (price_increase sales_decrease : ℝ) (price : ℝ) : ℝ :=
  let units_sold := initial_sales - (price - initial_price) / price_increase * sales_decrease
  (price - initial_cost) * units_sold

/-- Theorem stating the maximum daily profit and optimal price -/
theorem max_daily_profit_at_14_yuan 
  (initial_cost : ℝ) (initial_price : ℝ) (initial_sales : ℝ) 
  (price_increase : ℝ) (sales_decrease : ℝ) :
  initial_cost = 8 ∧ 
  initial_price = 10 ∧ 
  initial_sales = 200 ∧ 
  price_increase = 0.5 ∧
  sales_decrease = 10 →
  ∃ (optimal_price : ℝ), 
    optimal_price = 14 ∧ 
    daily_profit initial_cost initial_price initial_sales price_increase sales_decrease optimal_price = 720 ∧
    ∀ (price : ℝ), 
      daily_profit initial_cost initial_price initial_sales price_increase sales_decrease price ≤ 
      daily_profit initial_cost initial_price initial_sales price_increase sales_decrease optimal_price :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_daily_profit_at_14_yuan_l857_85799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l857_85712

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (5 + 4*x - x^2) / Real.log 0.5

-- Define the interval of increase
def interval_of_increase : Set ℝ := Set.Ioo 2 5

-- Theorem statement
theorem f_increasing_interval :
  ∀ x ∈ interval_of_increase, ∀ y ∈ interval_of_increase,
    x < y → f x < f y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l857_85712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l857_85775

/-- An ellipse with semi-major axis 2 and semi-minor axis b -/
structure Ellipse where
  b : ℝ
  h_b_pos : 0 < b
  h_b_lt_2 : b < 2

/-- Points on the ellipse -/
def EllipsePoint (e : Ellipse) :=
  {p : ℝ × ℝ // p.1^2 / 4 + p.2^2 / e.b^2 = 1}

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Left focus of the ellipse -/
noncomputable def leftFocus : ℝ × ℝ := (-Real.sqrt 4, 0)

/-- Right focus of the ellipse -/
noncomputable def rightFocus : ℝ × ℝ := (Real.sqrt 4, 0)

/-- Theorem about properties of the ellipse -/
theorem ellipse_properties (e : Ellipse) 
  (h_max : ∀ (A B : EllipsePoint e), 
    distance A.val rightFocus + distance B.val rightFocus ≤ 5) :
  (∃ (A B : EllipsePoint e), 
    distance A.val rightFocus + distance B.val rightFocus = 5 ∧ 
    distance A.val rightFocus = distance B.val rightFocus) ∧
  e.b = Real.sqrt 3 ∧
  (∀ (A B : EllipsePoint e), distance A.val B.val ≥ 3) ∧
  (∃ (A B : EllipsePoint e), distance A.val B.val = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l857_85775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_has_max_area_l857_85725

/-- Given a perimeter p and a side length a of a rectangle, 
    the area function of the rectangle is defined as a * (p/2 - a) -/
noncomputable def rectangle_area (p a : ℝ) : ℝ := a * (p/2 - a)

/-- The theorem states that for any positive real perimeter p,
    the maximum area of a rectangle with that perimeter occurs when
    one side length is p/4 (i.e., when the rectangle is a square) -/
theorem square_has_max_area (p : ℝ) (h : p > 0) :
  ∀ a : ℝ, 0 < a ∧ a < p/2 → rectangle_area p (p/4) ≥ rectangle_area p a := by
  sorry

#check square_has_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_has_max_area_l857_85725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_difference_divisibility_l857_85750

/-- Two polynomials have the same set of integer coefficients -/
def same_coefficients (P Q : Polynomial ℤ) : Prop :=
  ∃ (σ : Equiv ℕ ℕ), ∀ n, P.coeff n = Q.coeff (σ n)

theorem polynomial_difference_divisibility 
  (P Q : Polynomial ℤ) (h : same_coefficients P Q) : 
  (1007 : ℤ) ∣ (P.eval 2015 - Q.eval 2015) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_difference_divisibility_l857_85750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l857_85718

noncomputable def passing_speed (v_a v_b : ℝ) : ℝ := v_a + v_b

noncomputable def meters_to_km (m : ℝ) : ℝ := m / 1000

noncomputable def seconds_to_hours (s : ℝ) : ℝ := s / 3600

theorem goods_train_speed 
  (v_a : ℝ) 
  (length_b : ℝ) 
  (time : ℝ) 
  (h_v_a : v_a = 64) 
  (h_length_b : length_b = 420) 
  (h_time : time = 18) :
  ∃ v_b : ℝ, 
    v_b = 20 ∧ 
    passing_speed v_a v_b = (meters_to_km length_b) / (seconds_to_hours time) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l857_85718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_specific_triangle_l857_85797

/-- Represents a right triangle with sides a, b, and c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : a^2 + b^2 = c^2

/-- Calculates the length of the crease when folding a right triangle so that two vertices coincide -/
noncomputable def creaseLength (t : RightTriangle) : ℝ :=
  (3 * t.c) / (2 * t.b)

theorem crease_length_specific_triangle :
  ∃ (t : RightTriangle), t.a = 3 ∧ t.b = 4 ∧ t.c = 5 ∧ creaseLength t = 15/8 := by
  -- Construct the specific right triangle
  let t : RightTriangle := {
    a := 3
    b := 4
    c := 5
    is_right := by
      -- Prove that it's a right triangle (3^2 + 4^2 = 5^2)
      ring
  }
  
  -- Prove that this triangle satisfies all conditions
  use t
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  
  -- Calculate the crease length and show it equals 15/8
  calc
    creaseLength t = (3 * 5) / (2 * 4) := rfl
    _              = 15 / 8            := by ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_specific_triangle_l857_85797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_tan_l857_85738

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 3 * Real.tan (2 * x)

-- Define what it means to be a symmetry center
def IsSymmetryCenter (f : ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∀ x, f (c.1 + x) - c.2 = -(f (c.1 - x) - c.2)

-- State the theorem
theorem symmetry_center_of_tan (k : ℤ) :
  ∃ (c : ℝ × ℝ), c = (↑k * Real.pi / 4, 0) ∧ IsSymmetryCenter f c :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_tan_l857_85738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_perpendicular_lines_l857_85724

/-- Given two points P and Q and a point A in a plane, if PA is perpendicular to QA,
    then the line PQ passes through a fixed point. -/
theorem fixed_point_on_perpendicular_lines 
  (A P Q : ℝ × ℝ) 
  (a b θ : ℝ) 
  (h_a_neq_p : A ≠ P) 
  (h_a_neq_q : A ≠ Q) 
  (h_perp : (P.1 - A.1) * (Q.1 - A.1) + (P.2 - A.2) * (Q.2 - A.2) = 0) :
  ∃ (F : ℝ × ℝ), F.1 = a * Real.cosh θ * (a^2 + b^2) / (b^2 - a^2) ∧ 
                 F.2 = b * Real.sinh θ * (a^2 + b^2) / (a^2 - b^2) ∧
                 (F.2 - P.2) * (Q.1 - P.1) = (F.1 - P.1) * (Q.2 - P.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_perpendicular_lines_l857_85724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_food_stall_pricing_l857_85721

/-- Represents the price of food items at Ramon's food stall -/
structure FoodPrices where
  enchilada : ℚ
  taco : ℚ
  burrito : ℚ

/-- Calculates the total price for a given number of each food item -/
def totalPrice (prices : FoodPrices) (e t b : ℕ) : ℚ :=
  prices.enchilada * e + prices.taco * t + prices.burrito * b

/-- The main theorem stating the price of two enchiladas, three tacos, and one burrito -/
theorem food_stall_pricing (prices : FoodPrices) :
  totalPrice prices 1 2 1 = 7/2 →
  totalPrice prices 2 1 2 = 26/5 →
  totalPrice prices 2 3 1 = 11/2 := by
  sorry

-- Remove the #eval line as it's not necessary for this theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_food_stall_pricing_l857_85721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_l857_85737

noncomputable def w : Fin 3 → ℝ := ![3, -3, 3]

noncomputable def proj_w (v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  let dot_product := (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)
  let norm_squared := (w 0)^2 + (w 1)^2 + (w 2)^2
  fun i => (dot_product / norm_squared) * (w i)

theorem plane_equation (v : Fin 3 → ℝ) 
  (h : proj_w v = fun i => ![6, -6, 6] i) :
  (v 0) - (v 1) + (v 2) - 6 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_l857_85737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l857_85704

-- Define the curve
noncomputable def curve (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, 2 * Real.sin θ)

-- Define the point
def point : ℝ × ℝ := (3, -2)

-- Define the ellipse equation
def is_ellipse (p : ℝ × ℝ) : Prop := p.1^2 / 15 + p.2^2 / 10 = 1

-- Theorem statement
theorem ellipse_equation :
  ∃ (f : ℝ × ℝ) (d : ℝ), 
    (∀ θ, dist (curve θ) f = d) ∧ 
    dist point f = d ∧
    (∀ p : ℝ × ℝ, is_ellipse p ↔ (∃ θ, p = curve θ) ∨ p = point) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l857_85704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l857_85795

/-- The function f(x) = k(x-1) - 2ln(x) where k > 0 -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * (x - 1) - 2 * Real.log x

/-- The function g(x) = xe^(1-x) -/
noncomputable def g (x : ℝ) : ℝ := x * Real.exp (1 - x)

/-- Theorem stating the conditions and conclusions of the problem -/
theorem problem_statement (k : ℝ) : 
  (k > 0) → 
  ((∃! x, f k x = 0) → k = 2) ∧ 
  ((∀ s, s ∈ Set.Ioo 0 (Real.exp 1) → 
    ∃ t₁ t₂, t₁ ≠ t₂ ∧ t₁ ∈ Set.Ioo (Real.exp (-2)) (Real.exp 1) ∧ 
    t₂ ∈ Set.Ioo (Real.exp (-2)) (Real.exp 1) ∧ f k t₁ = g s ∧ f k t₂ = g s) → 
  k ∈ Set.Icc (3 / (Real.exp 1 - 1)) (3 * (Real.exp 1)^2 / ((Real.exp 1)^2 - 1))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l857_85795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_A_squared_minus_B_squared_l857_85748

theorem min_value_A_squared_minus_B_squared 
  (x y z : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) : 
  let A := Real.sqrt (x + 4) + Real.sqrt (y + 7) + Real.sqrt (z + 12)
  let B := Real.sqrt (x + 1) + Real.sqrt (y + 1) + Real.sqrt (z + 1)
  (A^2 - B^2) ≥ 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_A_squared_minus_B_squared_l857_85748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_y_intersection_l857_85729

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) (constantSum : ℝ) : Prop :=
  distance p e.focus1 + distance p e.focus2 = constantSum

/-- The main theorem -/
theorem ellipse_y_intersection
  (e : Ellipse)
  (h1 : e.focus1 = ⟨1, -1⟩)
  (h2 : e.focus2 = ⟨-2, 2⟩)
  (h3 : isOnEllipse e ⟨0, 0⟩ (3 * Real.sqrt 2)) :
  ∃ y : ℝ, y = Real.sqrt ((9 * Real.sqrt 2 - 4) / 2) ∧
    isOnEllipse e ⟨0, y⟩ (3 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_y_intersection_l857_85729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_violates_all_conditions_l857_85733

-- Define the function f(x) = 2^x
noncomputable def f (x : ℝ) : ℝ := 2^x

-- State the theorem
theorem function_violates_all_conditions :
  (∃ a b : ℝ, f (a + b) ≠ f a + f b) ∧
  (∃ a b : ℝ, f (a * b) ≠ f a + f b) ∧
  (∃ a b : ℝ, f (a * b) ≠ f a * f b) := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_violates_all_conditions_l857_85733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l857_85747

/-- Given real numbers m and n, if the line (m+1)x + (n+1)y - 2 = 0 is tangent to the circle (x-1)² + (y-1)² = 1,
    then m+n belongs to the set (-∞, 2-2√2] ∪ [2+2√2, +∞) -/
theorem line_tangent_to_circle (m n : ℝ) 
  (h : ∀ x y : ℝ, (m+1)*x + (n+1)*y - 2 = 0 → (x-1)^2 + (y-1)^2 = 1 → 
     ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, (x'-x)^2 + (y'-y)^2 < δ^2 → 
     ((m+1)*x' + (n+1)*y' - 2) * ((m+1)*x + (n+1)*y - 2) ≥ 0) : 
  m + n ∈ Set.Iic (2 - 2*Real.sqrt 2) ∪ Set.Ici (2 + 2*Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l857_85747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_max_value_l857_85732

/-- The expression to be maximized -/
noncomputable def f (a : ℝ) : ℝ :=
  (7 * Real.sqrt ((4 * a) ^ 2 + 4) - 2 * a ^ 2 - 2) / (Real.sqrt (4 + 16 * a ^ 2) + 6)

/-- The constraint on a -/
def constraint (a : ℝ) : Prop := a ^ 2 ≤ 4

/-- Theorem stating the existence of a maximum value -/
theorem exists_max_value :
  ∃ t_max : ℝ, ∃ a : ℝ, constraint a ∧ f a = t_max ∧ ∀ b : ℝ, constraint b → f b ≤ t_max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_max_value_l857_85732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_integers_count_l857_85778

def sequence_a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => sequence_a (n + 1) + sequence_a n

def sequence_b : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => sequence_b (n + 1) + sequence_b n

theorem common_integers_count :
  ∃ S : Finset ℕ, (∀ n ∈ S, ∃ i j, sequence_a i = sequence_b j ∧ sequence_a i = n) ∧
                   S.card = 3 ∧
                   ∀ n, (∃ i j, sequence_a i = sequence_b j ∧ sequence_a i = n) → n ∈ S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_integers_count_l857_85778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_divisors_iff_perfect_square_l857_85749

/-- A natural number has an odd number of divisors if and only if it is a perfect square. -/
theorem odd_divisors_iff_perfect_square (n : ℕ) : 
  Odd (Finset.card (Finset.filter (· ∣ n) (Finset.range (n + 1)))) ↔ ∃ k : ℕ, n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_divisors_iff_perfect_square_l857_85749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kola_solution_problem_l857_85760

/-- Represents the solution composition --/
structure SolutionComposition where
  total_volume : ℝ
  water_volume : ℝ
  kola_volume : ℝ
  sugar_volume : ℝ

/-- Calculates the sugar percentage in the solution --/
noncomputable def sugar_percentage (s : SolutionComposition) : ℝ :=
  s.sugar_volume / s.total_volume * 100

/-- The problem statement --/
theorem kola_solution_problem 
  (initial : SolutionComposition)
  (added_water : ℝ)
  (added_kola : ℝ)
  (added_sugar : ℝ)
  (final_sugar_percentage : ℝ)
  (h1 : initial.total_volume = 340)
  (h2 : initial.water_volume = 0.75 * initial.total_volume)
  (h3 : initial.kola_volume = 0.05 * initial.total_volume)
  (h4 : initial.sugar_volume = 0.20 * initial.total_volume)
  (h5 : added_water = 12)
  (h6 : added_kola = 6.8)
  (h7 : final_sugar_percentage = 19.66850828729282) :
  abs (added_sugar - 3.23) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kola_solution_problem_l857_85760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_six_equals_64I_l857_85719

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, 1; 4, -1]

theorem B_power_six_equals_64I :
  B^6 = 0 • B + 64 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_six_equals_64I_l857_85719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_decreasing_exponential_range_l857_85751

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * a - 1) ^ x

theorem strictly_decreasing_exponential_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y < f a x) → a > 1/2 ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_decreasing_exponential_range_l857_85751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_piece_volume_l857_85789

-- Define the circumference of the sphere
noncomputable def sphere_circumference : ℝ := 18 * Real.pi

-- Define the number of congruent pieces
def num_pieces : ℕ := 6

-- Theorem statement
theorem sphere_piece_volume :
  let radius : ℝ := sphere_circumference / (2 * Real.pi)
  let sphere_volume : ℝ := (4 / 3) * Real.pi * radius^3
  let piece_volume : ℝ := sphere_volume / (num_pieces : ℝ)
  piece_volume = 162 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_piece_volume_l857_85789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l857_85784

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 - x + 1)

-- State the theorem
theorem f_properties :
  -- Part 1: Monotonicity intervals
  (∀ x < -1, StrictMono (f ∘ (fun y ↦ min y x))) ∧
  (StrictAntiOn f (Set.Ioc (-1) 0)) ∧
  (∀ x > 0, StrictMono (f ∘ (fun y ↦ max y x))) ∧
  -- Part 2: Minimum value on [-1, 1]
  (∀ x ∈ Set.Icc (-1) 1, f 0 ≤ f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l857_85784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_six_average_l857_85786

theorem last_six_average (numbers : List ℝ) : 
  numbers.length = 13 ∧ 
  numbers.sum / 13 = 9 ∧ 
  (numbers.take 6).sum / 6 = 5 ∧ 
  numbers.get? 6 = some 45 →
  (numbers.drop 7).sum / 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_six_average_l857_85786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l857_85765

/-- The area of a rectangle inscribed in a triangle -/
theorem inscribed_rectangle_area (b h y : ℝ) (hb : b > 0) (hh : h > 0) (hy : 0 < y ∧ y < h) :
  b * y * (h - y) / h = b * y * (h - y) / h := by
  -- The proof is trivial as we're asserting equality to itself
  rfl

#check inscribed_rectangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l857_85765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l857_85710

/-- The central angle of a cone's lateral surface development diagram --/
noncomputable def central_angle (r h : ℝ) : ℝ :=
  2 * Real.pi * r / Real.sqrt (r^2 + h^2)

/-- Theorem: For a cone with base radius 1 and volume (√3 * π) / 3, 
    the central angle of its lateral surface development diagram is π. --/
theorem cone_central_angle :
  let r : ℝ := 1
  let v : ℝ := Real.sqrt 3 * Real.pi / 3
  let h : ℝ := 3 * v / (Real.pi * r^2)
  central_angle r h = Real.pi :=
by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l857_85710
