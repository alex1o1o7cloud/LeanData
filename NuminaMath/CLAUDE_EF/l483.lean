import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l483_48379

noncomputable def original_function (x : ℝ) : ℝ := 2 * Real.sin (2 * x)

noncomputable def shift_right (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f (x - shift)

noncomputable def shorten_x (f : ℝ → ℝ) (factor : ℝ) : ℝ → ℝ := λ x => f (x / factor)

noncomputable def transformed_function (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 3)

theorem function_transformation :
  shorten_x (shift_right original_function (Real.pi / 6)) 2 = transformed_function :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l483_48379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_head_start_l483_48385

/-- Calculates the head start of a rabbit given the speeds of a dog and rabbit and the time it takes for the dog to catch up. -/
theorem rabbit_head_start 
  (dog_speed : ℝ) 
  (rabbit_speed : ℝ) 
  (catch_up_time : ℝ) 
  (h1 : dog_speed = 24) 
  (h2 : rabbit_speed = 15) 
  (h3 : catch_up_time = 4 / 60) : 
  (dog_speed * catch_up_time) - (rabbit_speed * catch_up_time) = 0.6 := by
  -- Convert speeds from miles per hour to miles per minute
  have dog_speed_per_minute : dog_speed / 60 = 0.4 := by
    rw [h1]
    norm_num
  have rabbit_speed_per_minute : rabbit_speed / 60 = 0.25 := by
    rw [h2]
    norm_num
  
  -- Calculate distances traveled
  have dog_distance : dog_speed * catch_up_time = 1.6 := by
    rw [h1, h3]
    norm_num
  have rabbit_distance : rabbit_speed * catch_up_time = 1 := by
    rw [h2, h3]
    norm_num
  
  -- Prove the final result
  calc
    (dog_speed * catch_up_time) - (rabbit_speed * catch_up_time) 
      = 1.6 - 1 := by rw [dog_distance, rabbit_distance]
    _ = 0.6 := by norm_num

#check rabbit_head_start

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_head_start_l483_48385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_functions_l483_48355

-- Define the expressions
def expr1 : ℝ → ℝ := λ _ => 1
def expr2 : ℝ → ℝ := λ x => x^2
def expr3 : ℝ → ℝ := λ x => 1 - x
noncomputable def expr4 : ℝ → ℝ := λ x => Real.sqrt (x - 2) + Real.sqrt (1 - x)

-- Define what it means for an expression to be a function
def is_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, ∃ y : ℝ, f x = y

-- State the theorem
theorem three_functions :
  (is_function expr1 ∧ is_function expr2 ∧ is_function expr3) ∧
  ¬(is_function expr4) := by
  sorry

#check three_functions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_functions_l483_48355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l483_48309

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the circle (renamed to avoid conflict)
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 12/7

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 4*x

-- Define the line passing through upper and right vertices of C
def vertex_line (a b x y : ℝ) : Prop :=
  b*x + a*y = a*b

theorem ellipse_properties (a b : ℝ) :
  (∃ x y : ℝ, ellipse_C a b x y ∧
              (∃ x' y' : ℝ, vertex_line a b x' y' ∧ circle_eq x' y') ∧
              (∃ x_focus : ℝ, x_focus = 1 ∧ ellipse_C a b x_focus 0)) →
  (a^2 = 4 ∧ b^2 = 3) ∧
  (∀ A B : ℝ × ℝ, 
    ellipse_C a b A.1 A.2 → 
    ellipse_C a b B.1 B.2 → 
    A.1 * B.1 + A.2 * B.2 = 0 → 
    A.1^2 + A.2^2 + B.1^2 + B.2^2 ≥ 24/7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l483_48309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wang_water_usage_l483_48356

/-- Water pricing structure and usage calculation -/
noncomputable def water_fee (usage : ℝ) : ℝ :=
  if usage ≤ 10 then 2 * usage
  else 2 * 10 + 3 * (usage - 10)

/-- Mr. Wang's water usage in March -/
theorem wang_water_usage :
  ∃ (usage : ℝ), usage > 0 ∧ water_fee usage = 2.5 * usage ∧ usage = 20 := by
  use 20
  constructor
  · -- Prove usage > 0
    norm_num
  constructor
  · -- Prove water_fee usage = 2.5 * usage
    unfold water_fee
    simp [if_neg]
    norm_num
  · -- Prove usage = 20
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wang_water_usage_l483_48356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_is_131_over_11_l483_48386

noncomputable def floor_square_minus_floor_product (x : ℝ) : ℤ :=
  ⌊x^2⌋ - ⌊x * ⌊x⌋⌋

theorem smallest_solution_is_131_over_11 :
  ∀ x : ℝ, x > 0 → floor_square_minus_floor_product x = 10 → x ≥ 131/11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_is_131_over_11_l483_48386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_problem_rectangle_l483_48348

/-- Rectangle ABCD with given vertices -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  D : ℝ × ℝ
  is_rectangle : Bool

/-- The specific rectangle from the problem -/
def problem_rectangle (y : ℤ) : Rectangle where
  A := (1, -1)
  B := (101, 19)
  D := (3, y)
  is_rectangle := true

/-- Calculate the area of a rectangle given its vertices -/
noncomputable def calculate_area (r : Rectangle) : ℝ :=
  let AB := Real.sqrt ((r.B.1 - r.A.1)^2 + (r.B.2 - r.A.2)^2)
  let AD := Real.sqrt ((r.D.1 - r.A.1)^2 + (r.D.2 - r.A.2)^2)
  AB * AD

/-- Theorem stating that the area of the problem rectangle is 1040 -/
theorem area_of_problem_rectangle :
  ∃ y : ℤ, calculate_area (problem_rectangle y) = 1040 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_problem_rectangle_l483_48348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_a_value_l483_48338

def A : Set ℝ := {x : ℝ | |x - 1| > 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - (a+1)*x + a < 0}

theorem intersection_implies_a_value (a : ℝ) :
  A ∩ B a = Set.Ioo 3 5 → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_a_value_l483_48338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_of_two_equals_277_div_16_l483_48333

-- Define the functions r and u
def r (x : ℝ) : ℝ := 4 * x - 9

noncomputable def u (x : ℝ) : ℝ := 
  let y := (x + 9) / 4
  y^2 + 5 * y - 4

-- State the theorem
theorem u_of_two_equals_277_div_16 : u 2 = 277 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_of_two_equals_277_div_16_l483_48333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l483_48340

theorem trig_inequality (θ : Real) (h : -π/8 < θ ∧ θ < 0) : 
  Real.tan θ < Real.sin θ ∧ Real.sin θ < Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l483_48340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skill_test_probabilities_l483_48378

/-- Probability of a worker passing a skill test -/
def PassProbability := ℝ

/-- A worker passes the skill test with probability p -/
def worker_pass_probability (p : PassProbability) : ℝ := p

/-- Probability that a worker fails at least once in three consecutive tests -/
def fail_at_least_once_in_three (p : PassProbability) : ℝ :=
  1 - (worker_pass_probability p) ^ 3

/-- Probability that a worker is disqualified after exactly four tests -/
def disqualified_after_four (q : PassProbability) : ℝ :=
  let pass_prob := worker_pass_probability q
  let fail_prob := 1 - pass_prob
  (pass_prob ^ 2) * (fail_prob ^ 2) + pass_prob * fail_prob * (pass_prob ^ 2)

theorem skill_test_probabilities 
  (p q : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) 
  (hq : 0 ≤ q ∧ q ≤ 1) : 
  (fail_at_least_once_in_three p = 1 - p^3) ∧ 
  (disqualified_after_four q = q^2*(1-q)^2 + q*(1-q)*q^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_skill_test_probabilities_l483_48378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_theorem_l483_48354

-- Define the basic structures and types
structure Point where
  x : ℝ
  y : ℝ

structure Rectangle (A B C D : Point) where
  is_rectangle : Prop  -- We'll define this property later

-- Define the necessary functions and predicates
def angle_bisector (A B C E : Point) : Prop :=
  sorry  -- We'll define this later

def on_segment (P Q R : Point) : Prop :=
  sorry  -- We'll define this later

def length (P Q : Point) : ℝ :=
  sorry  -- We'll define this later

def area (A B C D : Point) : ℝ :=
  sorry  -- We'll define this later

-- State the theorem
theorem rectangle_area_theorem (A B C D E F : Point) 
  (h_rect : Rectangle A B C D)
  (h_bisect : angle_bisector A C D E)
  (h_E_on_AB : on_segment A E B)
  (h_F_on_AD : on_segment A F D)
  (h_BE : length B E = 10)
  (h_AF : length A F = 5) :
  area A B C D = 200 := by
  sorry  -- We'll prove this later

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_theorem_l483_48354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projections_on_circle_distance_product_constant_l483_48383

structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h : c^2 = a^2 - b^2

def tangent_line (E : Ellipse) (x₀ y₀ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | x₀ * p.1 / E.a^2 + y₀ * p.2 / E.b^2 = 1}

noncomputable def distance_to_line (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem projections_on_circle (E : Ellipse) :
  ∀ (x₀ y₀ : ℝ), (x₀^2 / E.a^2 + y₀^2 / E.b^2 = 1) →
  ∃ (center : ℝ × ℝ) (r : ℝ),
    (∀ (p : ℝ × ℝ), p ∈ tangent_line E x₀ y₀ →
      distance_to_line (-E.c, 0) (tangent_line E x₀ y₀) = r ∧
      distance_to_line (E.c, 0) (tangent_line E x₀ y₀) = r) :=
by
  sorry

theorem distance_product_constant (E : Ellipse) :
  ∃ (k : ℝ), ∀ (x₀ y₀ : ℝ), (x₀^2 / E.a^2 + y₀^2 / E.b^2 = 1) →
    distance_to_line (-E.c, 0) (tangent_line E x₀ y₀) *
    distance_to_line (E.c, 0) (tangent_line E x₀ y₀) = k :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projections_on_circle_distance_product_constant_l483_48383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base3_sum_equality_l483_48319

/-- Represents a number in base 3 --/
structure Base3 where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 3

/-- Converts a base 3 number to a natural number --/
def Base3.toNat (b : Base3) : Nat :=
  b.digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- Addition of two base 3 numbers --/
def addBase3 (a b : Base3) : Base3 :=
  sorry

/-- Converts a list of digits to a Base3 number, assuming validity --/
def listToBase3 (l : List Nat) : Base3 :=
  ⟨l, sorry⟩

theorem base3_sum_equality : 
  let a := listToBase3 [2]
  let b := listToBase3 [0, 2, 1]
  let c := listToBase3 [1, 0, 2]
  let d := listToBase3 [2, 0, 2, 1]
  let sum := listToBase3 [2, 0, 1, 1]
  (addBase3 (addBase3 (addBase3 a b) c) d).toNat = sum.toNat := by
    sorry

#check base3_sum_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base3_sum_equality_l483_48319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_hat_changes_l483_48369

/-- Represents the color of a dwarf's hat -/
inductive HatColor
| Red
| Blue

/-- Represents a dwarf -/
structure Dwarf where
  hat : HatColor

/-- Represents the village of dwarfs -/
def Village := Fin 2011 → Dwarf

/-- A dwarf with a red hat always tells the truth -/
def alwaysTruthful (d : Dwarf) : Prop :=
  d.hat = HatColor.Red → ∀ claim, claim = true

/-- A dwarf with a blue hat can lie -/
def canLie (d : Dwarf) : Prop :=
  d.hat = HatColor.Blue

/-- Each dwarf claims every other dwarf is wearing a blue hat -/
def allClaimBlue (v : Village) : Prop :=
  ∀ i j : Fin 2011, i ≠ j → (v i).hat = HatColor.Blue

/-- The number of hat color changes -/
noncomputable def hatColorChanges (v : Village) : ℕ := sorry

/-- The theorem to prove -/
theorem min_hat_changes (v : Village) 
  (h1 : ∀ i : Fin 2011, alwaysTruthful (v i) ∨ canLie (v i))
  (h2 : allClaimBlue v) :
  ∃ (v' : Village), hatColorChanges v' = 2009 ∧ 
    ∀ v'' : Village, hatColorChanges v'' ≥ 2009 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_hat_changes_l483_48369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l483_48362

-- Define the parabola C: x^2 = 2py (p > 0)
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define points
def point_O : ℝ × ℝ := (0, 0)
def point_A : ℝ × ℝ := (1, 1)
def point_B : ℝ × ℝ := (0, -1)

-- Define the line through two points
def line_through (p1 p2 : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p1.2) * (p2.1 - p1.1) = (x - p1.1) * (p2.2 - p1.2)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_properties 
  (p : ℝ) 
  (h_parabola : parabola p point_A.1 point_A.2) :
  -- 1. Line AB is tangent to C
  (∃ (x y : ℝ), parabola p x y ∧ line_through point_A point_B x y → x = point_A.1 ∧ y = point_A.2) ∧
  -- 2. For any line through B intersecting C at P and Q, |OP| · |OQ| > |OA|^2
  (∀ (P Q : ℝ × ℝ), parabola p P.1 P.2 → parabola p Q.1 Q.2 → 
    line_through point_B P P.1 P.2 → line_through point_B Q Q.1 Q.2 → 
    P ≠ Q → distance point_O P * distance point_O Q > distance point_O point_A ^ 2) ∧
  -- 3. For any line through B intersecting C at P and Q, |BP| · |BQ| > |BA|^2
  (∀ (P Q : ℝ × ℝ), parabola p P.1 P.2 → parabola p Q.1 Q.2 → 
    line_through point_B P P.1 P.2 → line_through point_B Q Q.1 Q.2 → 
    P ≠ Q → distance point_B P * distance point_B Q > distance point_B point_A ^ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l483_48362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_four_terms_l483_48316

def sequence_term (n : ℕ) : ℚ := n + 1 / (2^n : ℚ)

theorem sequence_first_four_terms :
  (sequence_term 1 = 3/2) ∧
  (sequence_term 2 = 9/4) ∧
  (sequence_term 3 = 25/8) ∧
  (sequence_term 4 = 65/16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_four_terms_l483_48316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l483_48387

-- Define the function f
def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then x^2 else x^2 -- We extend x^2 beyond [-1,1] for simplicity

-- Define the properties of f
axiom f_period (x : ℝ) : f (x + 2) = f x

-- Define the absolute value of logarithm
noncomputable def abs_log (x : ℝ) : ℝ := |Real.log x|

-- Define the domain of abs_log
def abs_log_domain (x : ℝ) : Prop := x > 0

-- Define an intersection point
def is_intersection (x : ℝ) : Prop :=
  abs_log_domain x ∧ f x = abs_log x

-- State the theorem
theorem intersection_count :
  ∃ (S : Finset ℝ), (∀ x ∈ S, is_intersection x) ∧ Finset.card S = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l483_48387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_nonnegative_iff_a_geq_neg_two_l483_48346

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 3*x + a
noncomputable def g (x : ℝ) : ℝ := Real.exp (x * Real.log 2) - x^2

-- State the theorem
theorem f_g_nonnegative_iff_a_geq_neg_two (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, f a (g x) ≥ 0) ↔ a ≥ -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_nonnegative_iff_a_geq_neg_two_l483_48346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_width_approx_l483_48366

/-- Calculates the width of a river given its depth, flow rate, and volume of water per minute. -/
noncomputable def river_width (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) : ℝ :=
  let flow_rate_mpm := flow_rate_kmph * 1000 / 60
  volume_per_minute / (depth * flow_rate_mpm)

/-- Theorem stating that for a river with depth 2 meters, flow rate 2 kmph, and volume 3000 cubic meters per minute, the width is approximately 45 meters. -/
theorem river_width_approx :
  let depth := (2 : ℝ)
  let flow_rate := (2 : ℝ)
  let volume := (3000 : ℝ)
  ∃ ε > 0, abs (river_width depth flow_rate volume - 45) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_width_approx_l483_48366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_points_l483_48321

noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

theorem inverse_proportion_points (m : ℝ) :
  let A : ℝ × ℝ := (-1, 3)
  let B : ℝ × ℝ := (2, m)
  ∀ (k : ℝ), (inverse_proportion k A.fst = A.snd ∧ inverse_proportion k B.fst = B.snd) →
  m = -3/2 := by
  sorry

#check inverse_proportion_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_points_l483_48321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_intersections_l483_48377

/-- Calculates the number of intersections between two regular polygons inscribed in a circle -/
def intersections (n m : ℕ) : ℕ := 2 * n * m

/-- The set of side counts for the inscribed regular polygons -/
def polygon_sides : Finset ℕ := {7, 8, 9, 10}

/-- Theorem: The total number of intersection points between the sides of regular polygons
    with 7, 8, 9, and 10 sides inscribed in the same circle, where no two polygons share
    a vertex and no three sides intersect at a common point, is equal to 862. -/
theorem total_intersections :
  (Finset.sum (Finset.filter (fun p => p.1 < p.2) (polygon_sides.product polygon_sides))
    (fun p => intersections p.1 p.2)) = 862 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_intersections_l483_48377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l483_48392

-- Define the system of equations
def system (t x y z : ℝ) : Prop :=
  x + y + z = t ∧ 
  x + (t + 1) * y + z = 0 ∧ 
  x + y - (t + 1) * z = 2 * t

-- Define the solution
noncomputable def solution (t : ℝ) : ℝ × ℝ × ℝ :=
  ((t^2 + 4*t + 2) / (t + 2), -1, -t / (t + 2))

-- Theorem statement
theorem system_solution (t : ℝ) (h1 : t ≠ 0) (h2 : t ≠ -2) :
  let (x, y, z) := solution t
  system t x y z := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l483_48392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_side_length_in_cone_l483_48325

/-- The side-length of a cube inscribed in a right circular cone -/
noncomputable def inscribedCubeSideLength (baseRadius height : ℝ) : ℝ :=
  6 / (2 + 3 * Real.sqrt 2)

/-- Theorem: The side-length of a cube inscribed in a right circular cone with base radius 1
    and height 3, such that one face of the cube is contained in the base of the cone,
    is equal to 6 / (2 + 3√2) -/
theorem inscribed_cube_side_length_in_cone :
  inscribedCubeSideLength 1 3 = 6 / (2 + 3 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_side_length_in_cone_l483_48325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_tube_volume_difference_l483_48351

/-- The volume of a cylinder with radius r and height h -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The radius of a cylinder with circumference c -/
noncomputable def radiusFromCircumference (c : ℝ) : ℝ := c / (2 * Real.pi)

theorem paper_tube_volume_difference : 
  let paperWidth : ℝ := 10
  let paperLength : ℝ := 12
  let charlieRadius := radiusFromCircumference paperWidth
  let charlieHeight := paperLength
  let danaRadius := radiusFromCircumference paperLength
  let danaHeight := paperWidth
  let charlieVolume := cylinderVolume charlieRadius charlieHeight
  let danaVolume := cylinderVolume danaRadius danaHeight
  Real.pi * |danaVolume - charlieVolume| = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_tube_volume_difference_l483_48351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_diameter_bounds_l483_48305

/-- Represents the diameter of a part with tolerances -/
structure Diameter where
  nominal : ℝ
  upper_tolerance : ℝ
  lower_tolerance : ℝ

/-- Calculates the maximum diameter of a part -/
def max_diameter (d : Diameter) : ℝ :=
  d.nominal + d.upper_tolerance

/-- Calculates the minimum diameter of a part -/
def min_diameter (d : Diameter) : ℝ :=
  d.nominal - d.lower_tolerance

/-- Theorem stating the maximum and minimum diameters of the part -/
theorem part_diameter_bounds (d : Diameter)
  (h1 : d.nominal = 30)
  (h2 : d.upper_tolerance = 0.03)
  (h3 : d.lower_tolerance = 0.04) :
  max_diameter d = 30.03 ∧ min_diameter d = 29.96 := by
  constructor
  · -- Proof for maximum diameter
    unfold max_diameter
    rw [h1, h2]
    norm_num
  · -- Proof for minimum diameter
    unfold min_diameter
    rw [h1, h3]
    norm_num

-- Example usage
def example_diameter : Diameter := ⟨30, 0.03, 0.04⟩

#eval max_diameter example_diameter
#eval min_diameter example_diameter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_diameter_bounds_l483_48305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_axis_ratio_l483_48393

/-- A hyperbola is defined by the equation x^2 - my^2 = 1, where m is a real number. -/
def Hyperbola (m : ℝ) : Prop := ∃ x y : ℝ, x^2 - m*y^2 = 1

/-- The length of the real axis of a hyperbola x^2 - my^2 = 1 -/
noncomputable def RealAxisLength (m : ℝ) : ℝ := 2

/-- The length of the imaginary axis of a hyperbola x^2 - my^2 = 1 -/
noncomputable def ImaginaryAxisLength (m : ℝ) : ℝ := 2 / Real.sqrt m

/-- Theorem: If the length of the imaginary axis is twice the length of the real axis
    for a hyperbola x^2 - my^2 = 1, then m = 1/4 -/
theorem hyperbola_axis_ratio (m : ℝ) (h1 : Hyperbola m) 
    (h2 : ImaginaryAxisLength m = 2 * RealAxisLength m) : m = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_axis_ratio_l483_48393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_course_angle_determination_l483_48397

theorem course_angle_determination (k : ℝ) (h : 0 < k ∧ k < 1) :
  ∃ α : ℝ, 0 ≤ α ∧ α ≤ π / 2 ∧ Real.sin α = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_course_angle_determination_l483_48397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_and_angle_relation_l483_48347

theorem orthogonal_vectors_and_angle_relation :
  ∀ (θ φ : Real),
  0 < θ ∧ θ < Real.pi / 2 →
  0 < φ ∧ φ < Real.pi / 2 →
  (Real.sin θ * 1 + (-2) * Real.cos θ) = 0 →
  5 * Real.cos (θ - φ) = 3 * Real.sqrt 5 * Real.cos φ →
  Real.sin θ = 2 * Real.sqrt 5 / 5 ∧
  Real.cos θ = Real.sqrt 5 / 5 ∧
  Real.cos φ = Real.sqrt 2 / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_and_angle_relation_l483_48347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_100_rings_l483_48349

/-- The number of unit squares in the nth ring of a square grid -/
def squares_in_ring (n : ℕ) : ℕ := 8 * n

/-- The sum of unit squares in the first n rings of a square grid -/
def sum_squares (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun i => squares_in_ring (i + 1))

/-- Theorem: The sum of unit squares in the first 100 rings is 40400 -/
theorem sum_squares_100_rings :
  sum_squares 100 = 40400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_100_rings_l483_48349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_hydrogen_atoms_in_compound_l483_48330

/-- Atomic weight of Calcium (Ca) in g/mol -/
def Ca_weight : ℝ := 40.08

/-- Atomic weight of Oxygen (O) in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Hydrogen (H) in g/mol -/
def H_weight : ℝ := 1.008

/-- The molecular weight of the compound in g/mol -/
def compound_weight : ℝ := 74

/-- 
Theorem stating that the number of hydrogen atoms in the compound is 2,
given the atomic weights and the total molecular weight of the compound.
-/
theorem hydrogen_atoms_in_compound : 
  ∃ (x : ℕ), (Ca_weight + 2 * O_weight + (x : ℝ) * H_weight = compound_weight) ∧ x = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_hydrogen_atoms_in_compound_l483_48330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_distance_l483_48315

/-- Represents a car rental scenario with two options and gasoline information -/
structure CarRental where
  option1_cost : ℚ  -- Cost of first option (excluding gasoline)
  option2_cost : ℚ  -- Cost of second option (including gasoline)
  gas_efficiency : ℚ  -- Kilometers per liter of gasoline
  gas_cost : ℚ  -- Cost per liter of gasoline
  savings : ℚ  -- Amount saved by choosing first option

/-- Calculates the one-way distance of the trip in kilometers -/
def calculate_distance (rental : CarRental) : ℚ :=
  (rental.option2_cost - rental.option1_cost - rental.savings) * rental.gas_efficiency / (2 * rental.gas_cost)

theorem trip_distance (rental : CarRental) 
  (h1 : rental.option1_cost = 50)
  (h2 : rental.option2_cost = 90)
  (h3 : rental.gas_efficiency = 15)
  (h4 : rental.gas_cost = 9/10)
  (h5 : rental.savings = 22) :
  calculate_distance rental = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_distance_l483_48315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_dodecagon_angles_l483_48375

/-- A regular dodecagon is a polygon with 12 sides and all sides and angles equal. -/
structure RegularDodecagon where
  sides : Nat
  sides_eq : sides = 12

/-- The measure of an interior angle of a regular polygon with n sides. -/
noncomputable def interior_angle_measure (n : Nat) : ℝ :=
  (n - 2 : ℝ) * 180 / n

/-- The measure of an exterior angle of a regular polygon. -/
noncomputable def exterior_angle_measure (interior : ℝ) : ℝ :=
  180 - interior

theorem regular_dodecagon_angles (d : RegularDodecagon) :
  interior_angle_measure d.sides = 150 ∧ 
  exterior_angle_measure (interior_angle_measure d.sides) = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_dodecagon_angles_l483_48375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_scheme_b_is_200_l483_48389

/-- Represents the investment schemes and their properties -/
structure InvestmentScheme where
  initialInvestmentA : ℚ
  yieldRateA : ℚ
  yieldRateB : ℚ
  differenceBetweenSchemes : ℚ

/-- Calculates the amount invested in scheme B given the investment conditions -/
def calculateInvestmentB (scheme : InvestmentScheme) : ℚ :=
  let totalA := scheme.initialInvestmentA * (1 + scheme.yieldRateA)
  (totalA - scheme.differenceBetweenSchemes) / (1 + scheme.yieldRateB)

/-- Theorem stating that the investment in scheme B is $200 given the problem conditions -/
theorem investment_scheme_b_is_200 :
  let scheme : InvestmentScheme := {
    initialInvestmentA := 300,
    yieldRateA := 3/10,
    yieldRateB := 1/2,
    differenceBetweenSchemes := 90
  }
  calculateInvestmentB scheme = 200 := by
  -- Unfold the definition and simplify
  unfold calculateInvestmentB
  simp [InvestmentScheme]
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_scheme_b_is_200_l483_48389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_is_one_l483_48322

-- Define the given constants
def distance_BC : ℝ := 12
def time_AC : ℝ := 4
def time_CA : ℝ := 6
def time_AB_motor : ℝ := 0.75  -- 45 minutes in hours

-- Define variables
def speed_current : ℝ → ℝ := λ v_T ↦ v_T
def speed_boat : ℝ → ℝ := λ v_L ↦ v_L
def distance_AB : ℝ → ℝ := λ d ↦ d

-- State the theorem
theorem current_speed_is_one :
  ∀ v_T v_L d,
  (speed_boat v_L + speed_current v_T) * time_AC = distance_AB d + distance_BC →
  (speed_boat v_L - speed_current v_T) * time_CA = distance_AB d + distance_BC →
  3 * speed_boat v_L * time_AB_motor = distance_AB d →
  speed_current v_T = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_is_one_l483_48322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_area_theorem_l483_48391

/-- Represents the area traced by highest points of projectile trajectories -/
noncomputable def projectileArea (u : ℝ) (g : ℝ) : ℝ := 3 * u^4 / (4 * g^2)

/-- Represents the x-coordinate of the highest point for a given initial velocity -/
noncomputable def highestPointX (v : ℝ) (g : ℝ) : ℝ := v^2 / (2 * g)

/-- Represents the y-coordinate of the highest point for a given initial velocity -/
noncomputable def highestPointY (v : ℝ) (g : ℝ) : ℝ := v^2 / (4 * g)

theorem projectile_area_theorem (u : ℝ) (g : ℝ) 
    (h_u : u > 0) (h_g : g > 0) : 
  ∫ x in Set.Icc (highestPointX u g) (highestPointX (2*u) g), 
    (1/2) * x = projectileArea u g := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_area_theorem_l483_48391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_in_expansion_l483_48399

theorem constant_term_in_expansion (a : ℝ) : 
  (∀ x : ℝ, (x + a / x) * (2 * x - 1)^5 = 4) →
  ∃ t : ℝ → ℝ, (∀ x : ℝ, (x + a / x) * (2 * x - 1)^5 = t x) ∧ 
    (∃ c : ℝ, c = 30 ∧ (∀ ε > (0 : ℝ), ∃ δ > (0 : ℝ), ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |t x - c| < ε)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_in_expansion_l483_48399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_after_rotation_l483_48335

/-- Represents a line in 2D space --/
structure Line2D where
  slope : ℝ
  yIntercept : ℝ

/-- Rotates a line around a point by a given angle --/
noncomputable def rotateLine (l : Line2D) (center : ℝ × ℝ) (angle : ℝ) : Line2D :=
  sorry

/-- Calculates the x-intercept of a line --/
noncomputable def xIntercept (l : Line2D) : ℝ :=
  -l.yIntercept / l.slope

theorem x_intercept_after_rotation (l : Line2D) :
  let originalLine : Line2D := { slope := -2/3, yIntercept := 2 }
  let rotationCenter : ℝ × ℝ := (3, -1)
  let rotationAngle : ℝ := π/6  -- 30 degrees in radians
  let rotatedLine := rotateLine originalLine rotationCenter rotationAngle
  xIntercept rotatedLine = -rotatedLine.yIntercept / rotatedLine.slope :=
by sorry

#check x_intercept_after_rotation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_after_rotation_l483_48335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_EF_is_one_l483_48395

open Real

/-- Curve C1 in polar coordinates -/
noncomputable def C1 (θ : ℝ) : ℝ := cos θ + sin θ

/-- Curve C2 in polar coordinates -/
noncomputable def C2 (θ : ℝ) : ℝ := 1 / (2 * (cos θ + sin θ))

/-- The angle at which we calculate the intersection points -/
noncomputable def θ_intersection : ℝ := 2 * π / 3

theorem length_EF_is_one :
  C2 θ_intersection - C1 θ_intersection = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_EF_is_one_l483_48395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_proposition_correct_l483_48336

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relations
def perpendicular : Line → Line → Prop := sorry
def perpendicular_plane : Plane → Plane → Prop := sorry
def perpendicular_line_plane : Line → Plane → Prop := sorry
def parallel : Line → Line → Prop := sorry
def parallel_plane : Plane → Plane → Prop := sorry
def contains : Plane → Line → Prop := sorry

-- Define the lines and planes
variable (l m n : Line)
variable (α β : Plane)

-- Define the propositions
def proposition1 (l : Line) (α β : Plane) : Prop :=
  perpendicular_plane α β → perpendicular_line_plane l α → parallel l β

def proposition2 (l : Line) (α β : Plane) : Prop :=
  perpendicular_plane α β → contains α l → perpendicular_line_plane l β

def proposition3 (l m n : Line) : Prop :=
  perpendicular l m → perpendicular m n → parallel l n

def proposition4 (m n : Line) (α β : Plane) : Prop :=
  perpendicular_line_plane m α → parallel n β → parallel_plane α β → perpendicular m n

-- The main theorem
theorem only_fourth_proposition_correct
  (h_diff_lines : l ≠ m ∧ m ≠ n ∧ l ≠ n)
  (h_diff_planes : α ≠ β) :
  ¬proposition1 l α β ∧ ¬proposition2 l α β ∧ ¬proposition3 l m n ∧ proposition4 m n α β :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_proposition_correct_l483_48336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l483_48398

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def fractional_part (x : ℝ) : ℝ := x - floor x

def circle_equation (x y : ℝ) : Prop :=
  (fractional_part x)^2 + y^2 = fractional_part x

def line_equation (x y : ℝ) : Prop :=
  y = (1/4) * x

def intersection_point (x y : ℝ) : Prop :=
  circle_equation x y ∧ line_equation x y ∧ x < 5

theorem intersection_count :
  ∃ (S : Finset (ℝ × ℝ)), (∀ p ∈ S, intersection_point p.1 p.2) ∧ S.card = 10 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l483_48398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_OLA_area_product_l483_48341

/-- Regular polygons DIAL, FOR, and FRIEND in the plane -/
structure RegularPolygons where
  DIAL : Set (ℝ × ℝ)
  FOR : Set (ℝ × ℝ)
  FRIEND : Set (ℝ × ℝ)

/-- The length of ID is 1 -/
def ID_length : ℝ := 1

/-- The possible positions of point O -/
inductive OPosition
  | Midpoint
  | EquilateralUp
  | EquilateralDown

/-- The area of triangle OLA given the position of O -/
noncomputable def OLA_area (pos : OPosition) : ℝ :=
  match pos with
  | OPosition.Midpoint => 1 / 2
  | OPosition.EquilateralUp => (1 + Real.sqrt 3 / 2) / 2
  | OPosition.EquilateralDown => (1 - Real.sqrt 3 / 2) / 2

/-- The theorem stating that the product of all possible areas of OLA is 1/32 -/
theorem OLA_area_product (polygons : RegularPolygons) :
  (OLA_area OPosition.Midpoint) *
  (OLA_area OPosition.EquilateralUp) *
  (OLA_area OPosition.EquilateralDown) = 1 / 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_OLA_area_product_l483_48341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_source_height_light_source_calculation_l483_48396

/-- Given a cube with edge length 2 cm and a light source above a vertex creating a shadow,
    calculate the floor of 1000 times the height of the light source. -/
theorem light_source_height (shadow_area : ℝ) (x : ℝ) :
  shadow_area = 200 → ⌊1000 * x⌋ = 12280 := by
  intro h
  sorry

/-- The edge length of the cube in cm -/
def cube_edge : ℝ := 2

/-- The shadow forms a square whose area includes the area beneath the cube -/
noncomputable def total_shadow_area (shadow_area : ℝ) : ℝ :=
  shadow_area + cube_edge ^ 2

/-- The side length of the total shadow square -/
noncomputable def shadow_side (shadow_area : ℝ) : ℝ :=
  Real.sqrt (total_shadow_area shadow_area)

/-- Relation between light source height and shadow expansion -/
axiom similar_triangles (x : ℝ) (shadow_area : ℝ) :
  x / cube_edge = (shadow_side shadow_area - cube_edge) / cube_edge

/-- The calculation of the light source height -/
theorem light_source_calculation (shadow_area : ℝ) (x : ℝ) :
  shadow_area = 200 → ⌊1000 * x⌋ = 12280 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_source_height_light_source_calculation_l483_48396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_conditions_l483_48350

theorem no_function_satisfies_conditions : ¬∃ f : ℝ → ℝ, 
  (∀ x : ℝ, f (x^2) - (f x)^2 ≥ (1/4 : ℝ)) ∧ 
  (∀ x y : ℝ, f x = f y → x = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_conditions_l483_48350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_row_perfect_squares_exist_l483_48334

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def valid_arrangement (arr : List ℕ) : Prop :=
  arr.length = 9 ∧ 
  (∀ n, n ∈ arr → 1 ≤ n ∧ n ≤ 9) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 9 → n ∈ arr)

theorem second_row_perfect_squares_exist : 
  ∃ (second_row : List ℕ), 
    valid_arrangement second_row ∧ 
    (∀ i, i ∈ List.range 9 → 
      is_perfect_square ((i + 1) + (second_row.get ⟨i, by sorry⟩))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_row_perfect_squares_exist_l483_48334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l483_48339

noncomputable def f (x : ℝ) := 2 / (x - 1)

theorem min_value_of_f :
  ∀ x ∈ Set.Icc 3 6, f x ≥ 2/5 ∧ ∃ y ∈ Set.Icc 3 6, f y = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l483_48339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_binary_38_l483_48373

def decimalToBinary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec loop (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else loop (m / 2) ((m % 2) :: acc)
    loop n []

theorem decimal_to_binary_38 : 
  decimalToBinary 38 = [1, 0, 0, 1, 1, 0] := by
  rfl

#eval decimalToBinary 38

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_binary_38_l483_48373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_radius_theorem_l483_48364

/-- The radius of an inscribed cylinder in a cone -/
noncomputable def inscribed_cylinder_radius (cone_diameter : ℝ) (cone_altitude : ℝ) (cylinder_height_ratio : ℝ) : ℝ :=
  140 / 41

/-- Theorem stating the radius of an inscribed cylinder in a cone -/
theorem inscribed_cylinder_radius_theorem 
  (cone_diameter : ℝ) 
  (cone_altitude : ℝ) 
  (cylinder_height_ratio : ℝ) 
  (h1 : cone_diameter = 14)
  (h2 : cone_altitude = 20)
  (h3 : cylinder_height_ratio = 3) : 
  inscribed_cylinder_radius cone_diameter cone_altitude cylinder_height_ratio = 140 / 41 :=
by
  sorry

#check inscribed_cylinder_radius_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_radius_theorem_l483_48364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_floor_power_existence_l483_48352

/-- A number is squarefree if it is not divisible by any perfect square other than 1 -/
def IsSquarefree (n : ℕ) : Prop :=
  ∀ (p : ℕ), Nat.Prime p → (p * p ∣ n) → p = 1

theorem coprime_floor_power_existence
  (k M : ℕ)
  (hk : k > 0)
  (hM : M > 0)
  (hksq : ¬ IsSquarefree (k - 1)) :
  ∃ (α : ℝ), α > 0 ∧ ∀ (n : ℕ), n > 0 →
    Nat.Coprime (Int.toNat (Int.floor (α * (k ^ n : ℝ)))) M :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_floor_power_existence_l483_48352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_preservation_time_at_33_l483_48384

-- Define the functional relationship
noncomputable def preservation_time (x k b : ℝ) : ℝ := Real.exp (k * x + b)

-- State the theorem
theorem preservation_time_at_33 (k b : ℝ) :
  preservation_time 0 k b = 192 →
  preservation_time 22 k b = 48 →
  preservation_time 33 k b = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_preservation_time_at_33_l483_48384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_sqrt_2_quadratic_equation_solutions_l483_48331

-- Part 1
theorem complex_expression_equals_sqrt_2 :
  ((-1)^2 : ℝ)^(1/3) + (-8 : ℝ)^(1/3) + Real.sqrt 3 - abs (1 - Real.sqrt 3) + Real.sqrt 2 = Real.sqrt 2 := by
sorry

-- Part 2
theorem quadratic_equation_solutions (x : ℝ) :
  25 * (x + 2)^2 - 36 = 0 ↔ x = -16/5 ∨ x = -4/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_sqrt_2_quadratic_equation_solutions_l483_48331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ratio_l483_48361

noncomputable section

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the condition that a > b
def a_greater_b (a b : ℝ) : Prop := a > b

-- Define the angle between asymptotes
noncomputable def asymptote_angle (a b : ℝ) : ℝ := 2 * Real.arctan (b / a)

theorem hyperbola_ratio (a b : ℝ) :
  a_greater_b a b →
  asymptote_angle a b = Real.pi / 4 →
  a / b = 3 + 2 * Real.sqrt 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ratio_l483_48361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_change_l483_48359

noncomputable def initial_area : ℝ := 100 * Real.sqrt 3

def side_decrease : ℝ := 8

theorem equilateral_triangle_area_change :
  ∀ s : ℝ,
  s > 0 →
  s^2 * Real.sqrt 3 / 4 = initial_area →
  let new_side := s - side_decrease
  let new_area := new_side^2 * Real.sqrt 3 / 4
  new_area = 36 * Real.sqrt 3 ∧
  initial_area - new_area = 64 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_change_l483_48359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_with_abc_property_l483_48382

def S (m : ℕ) : Set ℕ := {n : ℕ | 5 ≤ n ∧ n ≤ m}

def valid_partition (m : ℕ) (A B : Set ℕ) : Prop :=
  A ∪ B = S m ∧ A ∩ B = ∅

def contains_abc (X : Set ℕ) : Prop :=
  ∃ a b c, a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ a * b = c

theorem smallest_m_with_abc_property :
  ∀ m : ℕ, m ≥ 5 →
    (∀ A B : Set ℕ, valid_partition m A B → (contains_abc A ∨ contains_abc B)) ↔
    m ≥ 3125 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_with_abc_property_l483_48382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extrema_of_f_l483_48308

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x / 8 + 2 / x

-- Define the interval
def I : Set ℝ := Set.Ioo (-5) 10

-- Theorem statement
theorem extrema_of_f :
  ∃ (max_x min_x : ℝ),
    max_x ∈ I ∧ min_x ∈ I ∧
    (∀ x ∈ I, f x ≤ f max_x) ∧
    (∀ x ∈ I, f min_x ≤ f x) ∧
    max_x = -4 ∧ f max_x = -1 ∧
    min_x = 4 ∧ f min_x = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extrema_of_f_l483_48308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_fourth_l483_48365

theorem tan_theta_plus_pi_fourth (θ : Real) : 
  (∃ (x y : Real), x = 2 ∧ y = 4 ∧ Real.tan θ = y / x) →
  Real.tan (θ + π/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_fourth_l483_48365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l483_48307

def f (x : ℝ) : ℝ := x^2
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 2^x - m

theorem m_range_theorem (m : ℝ) : 
  (∀ x₁ ∈ Set.Icc (-1) 3, ∃ x₂ ∈ Set.Icc 0 2, f x₁ ≥ g m x₂) ↔ m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l483_48307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_collinearity_l483_48337

/-- Ellipse C with equation x²/4 + y² = 1 -/
noncomputable def C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Point P on ellipse C -/
noncomputable def P (x₀ y₀ : ℝ) : Prop := C x₀ y₀

/-- Point Q on ellipse C, symmetric to P about y-axis -/
noncomputable def Q (x₀ y₀ : ℝ) : Prop := C (-x₀) y₀

/-- Point M, midpoint of OP -/
noncomputable def M (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀/2, y₀/2)

/-- Point N, midpoint of BP -/
noncomputable def N (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀/2, (y₀-1)/2)

/-- Point D, intersection of AM and C -/
noncomputable def D (x₀ y₀ : ℝ) : ℝ × ℝ := 
  (2*x₀*(2-y₀)/(5-4*y₀), (-2*y₀^2 + 4*y₀ - 3)/(5-4*y₀))

/-- Collinearity of three points -/
def AreCollinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - q.1) = (r.2 - q.2) * (q.1 - p.1)

theorem ellipse_collinearity (x₀ y₀ : ℝ) 
  (hP : P x₀ y₀) (hQ : Q x₀ y₀) :
  AreCollinear (D x₀ y₀) (N x₀ y₀) (-x₀, y₀) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_collinearity_l483_48337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_wears_white_hat_l483_48300

structure Person :=
  (name : String)
  (canSeeAhead : Bool)
  (canSeeBehind : Bool)
  (hatColor : Option HatColor)
  (knowsOwnHatColor : Bool)

inductive HatColor
  | White
  | Black

def totalHats : Nat := 5
def whiteHats : Nat := 3
def blackHats : Nat := 2

def A : Person := ⟨"A", false, false, none, false⟩
def B : Person := ⟨"B", true, false, none, false⟩
def C : Person := ⟨"C", true, true, none, false⟩

theorem a_wears_white_hat 
  (h1 : C.canSeeAhead ∧ C.canSeeBehind)
  (h2 : B.canSeeAhead ∧ ¬B.canSeeBehind)
  (h3 : ¬A.canSeeAhead ∧ ¬A.canSeeBehind)
  (h4 : totalHats = whiteHats + blackHats)
  (h5 : ¬C.knowsOwnHatColor)
  (h6 : ¬B.knowsOwnHatColor)
  (h7 : A.knowsOwnHatColor) :
  A.hatColor = some HatColor.White :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_wears_white_hat_l483_48300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pace_improved_by_two_minutes_l483_48318

-- Define the initial condition
noncomputable def initial_distance : ℚ := 8
noncomputable def initial_time : ℚ := 96

-- Define the current condition
noncomputable def current_distance : ℚ := 10
noncomputable def current_time : ℚ := 100

-- Define the pace improvement
noncomputable def pace_improvement : ℚ := 
  (initial_time / initial_distance) - (current_time / current_distance)

-- Theorem statement
theorem pace_improved_by_two_minutes : 
  pace_improvement = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pace_improved_by_two_minutes_l483_48318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l483_48328

theorem unique_solution : ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^4 - 6*(x^2) + 1 = 7 * 2^y :=
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l483_48328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninety_third_term_l483_48342

/-- Represents a term in the sequence as a pair of natural numbers -/
def Term := ℕ × ℕ

/-- The nth group in the sequence -/
def group (n : ℕ) : List Term :=
  List.range n |>.map (λ i => (i + 1, n - i))

/-- The sequence up to the nth group -/
def sequenceUpTo (n : ℕ) : List Term :=
  List.join (List.range n |>.map (λ i => group (i + 1)))

/-- The number of terms in the first n groups -/
def num_terms (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem ninety_third_term :
  (sequenceUpTo 14).get? 92 = some (2, 13) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninety_third_term_l483_48342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_not_sum_of_three_cubes_l483_48302

/-- A perfect cube is a number that is the cube of an integer. -/
def PerfectCube (n : ℤ) : Prop := ∃ k : ℤ, n = k^3

/-- A number is expressible as the sum of three perfect cubes if it can be written as the sum of three perfect cubes. -/
def ExpressibleAsSumOfThreeCubes (n : ℤ) : Prop :=
  ∃ a b c : ℤ, PerfectCube a ∧ PerfectCube b ∧ PerfectCube c ∧ n = a + b + c

/-- The set of numbers that cannot be expressed as the sum of three perfect cubes is infinite. -/
theorem infinitely_many_not_sum_of_three_cubes :
  ∃ S : Set ℤ, (∀ n ∈ S, ¬ExpressibleAsSumOfThreeCubes n) ∧ Set.Infinite S :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_not_sum_of_three_cubes_l483_48302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_plane_and_normal_line_l483_48360

noncomputable section

variable (a b c : ℝ)
variable (x y z : ℝ)

def surface (a b c x y z : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) + (z^2 / c^2) = 1

def point_M (a b c : ℝ) : ℝ × ℝ × ℝ := (a / Real.sqrt 2, b / 2, c / 2)

def tangent_plane (a b c x y z : ℝ) : Prop := (Real.sqrt 2 * x / a) + (y / b) + (z / c) = 2

def normal_line (a b c x y z : ℝ) : Prop := 
  (a / Real.sqrt 2) * (x - a / Real.sqrt 2) = b * (y - b / 2) ∧
  b * (y - b / 2) = c * (z - c / 2)

theorem tangent_plane_and_normal_line (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  let (x₀, y₀, z₀) := point_M a b c
  surface a b c x₀ y₀ z₀ →
  tangent_plane a b c x₀ y₀ z₀ ∧
  normal_line a b c x₀ y₀ z₀ :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_plane_and_normal_line_l483_48360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_T_is_quadrilateral_l483_48301

/-- The set T of points (x, y) satisfying the given conditions forms a quadrilateral -/
theorem set_T_is_quadrilateral (a : ℝ) (ha : a > 0) :
  let T := {p : ℝ × ℝ | 
    a ≤ p.1 ∧ p.1 ≤ 3 * a ∧
    a ≤ p.2 ∧ p.2 ≤ 3 * a ∧
    p.1 + p.2 ≥ 2 * a ∧
    p.1 + 2 * a ≥ 2 * p.2 ∧
    p.2 + 2 * a ≥ 2 * p.1}
  ∃ (v1 v2 v3 v4 : ℝ × ℝ),
    v1 ∈ T ∧ v2 ∈ T ∧ v3 ∈ T ∧ v4 ∈ T ∧
    (∀ p ∈ T, p ∈ convexHull ℝ {v1, v2, v3, v4}) ∧
    (∀ (S : Finset (ℝ × ℝ)), (∀ p ∈ T, p ∈ convexHull ℝ S) → S.card ≥ 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_T_is_quadrilateral_l483_48301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_number_problem_l483_48388

theorem middle_number_problem (numbers : Finset ℝ) 
  (h_count : numbers.card = 8)
  (h_distinct : numbers.card = Finset.card (Finset.image id numbers))
  (h_avg_all : (Finset.sum numbers id) / 8 = 7)
  (first_five last_five : Finset ℝ)
  (h_first_five : first_five ⊆ numbers ∧ first_five.card = 5)
  (h_last_five : last_five ⊆ numbers ∧ last_five.card = 5)
  (h_avg_first : (Finset.sum first_five id) / 5 = 6)
  (h_avg_last : (Finset.sum last_five id) / 5 = 9)
  (middle : ℝ)
  (h_middle_in_first : middle ∈ first_five)
  (h_middle_in_last : middle ∈ last_five) :
  middle = 9.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_number_problem_l483_48388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_slope_AB_max_k_l483_48358

noncomputable section

-- Define the ellipse M
def ellipse_M (x y : ℝ) : Prop := x^2 / 8 + y^2 / 2 = 1

-- Define point P
def point_P : ℝ × ℝ := (2, 1)

-- Define the slope of line AB
def slope_AB (k₁ k₂ : ℝ) : ℝ := 1 / 2

-- Define the ratio k
noncomputable def k (t : ℝ) : ℝ := ((t^2 + 4) * (t^2 + 36)) / ((t^2 + 12) * (t^2 + 12))

-- Theorem 1: Constant slope of line AB
theorem constant_slope_AB (k₁ k₂ : ℝ) (h : k₁ + k₂ = 0) :
  slope_AB k₁ k₂ = 1 / 2 := by sorry

-- Theorem 2: Maximum value of k
theorem max_k : 
  ∃ t : ℝ, ∀ s : ℝ, k s ≤ k t ∧ k t = 4 / 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_slope_AB_max_k_l483_48358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_monochromatic_rectangle_l483_48381

-- Define a color type
inductive Color where
  | Red
  | Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point
axiom colorAssignment : Point → Color

-- Define a rectangle
structure Rectangle where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

-- Theorem statement
theorem exists_monochromatic_rectangle :
  ∃ (r : Rectangle), 
    colorAssignment r.p1 = colorAssignment r.p2 ∧
    colorAssignment r.p2 = colorAssignment r.p3 ∧
    colorAssignment r.p3 = colorAssignment r.p4 := by
  sorry

#check exists_monochromatic_rectangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_monochromatic_rectangle_l483_48381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_tangent_l483_48327

/-- The equation of circle C -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8 = 0

/-- The center of circle C -/
def center : ℝ × ℝ := (3, 0)

/-- The slope of the tangent line -/
noncomputable def k : ℝ := -Real.sqrt 2 / 4

/-- The tangent line equation -/
def tangent_line (x y : ℝ) : Prop := y = k * x

theorem circle_center_and_tangent :
  (∀ x y, circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = 1) ∧
  (∃ x y, x > 0 ∧ y < 0 ∧ circle_eq x y ∧ tangent_line x y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_tangent_l483_48327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_ratio_sum_l483_48326

theorem sin_cos_ratio_sum (x y : ℝ) 
  (h1 : Real.sin x / Real.sin y = 3) 
  (h2 : Real.cos x / Real.cos y = 1/2) : 
  Real.sin (2*x) / Real.sin (2*y) + Real.cos (2*x) / Real.cos (2*y) = 49/58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_ratio_sum_l483_48326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_coin_weights_l483_48310

/-- Represents a coin with an unknown weight -/
structure Coin where
  weight : ℕ
  unknown : Bool

/-- Represents a weighing operation on a two-pan balance -/
def Weighing := List Coin → List Coin → Ordering

/-- Represents the state of knowledge about the coins -/
structure CoinState where
  coins : List Coin
  weighings : List Weighing

/-- The main theorem statement -/
theorem determine_coin_weights 
  (coins : List Coin) 
  (h_count : coins.length = 6) 
  (h_weights : ∀ c ∈ coins, c.weight ∈ [9, 10, 11]) 
  (h_unknown : ∀ c ∈ coins, c.unknown) :
  ∃ (weighings : List Weighing), 
    weighings.length ≤ 4 ∧ 
    ∀ c ∈ coins, ∃ w, w ∈ [9, 10, 11] ∧ 
      (CoinState.mk coins weighings).coins.find? (fun c' ↦ c'.weight = w ∧ ¬c'.unknown) = some c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_coin_weights_l483_48310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factor_implies_c_value_l483_48374

theorem polynomial_factor_implies_c_value : 
  ∀ (c p : ℝ), (∃ (q : ℝ → ℝ), ∀ x, 3*x^3 + c*x + 12 = (x^2 + p*x + 4) * q x) → c = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factor_implies_c_value_l483_48374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_imply_neg_a_f_not_even_and_odd_range_shift_not_equal_no_single_intersection_l483_48371

-- Proposition 1
theorem quadratic_roots_imply_neg_a (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + (a-3)*x + a = 0 ∧ y^2 + (a-3)*y + a = 0) → a < 0 := by
  sorry

-- Proposition 2
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 1) + Real.sqrt (1 - x^2)

theorem f_not_even_and_odd :
  ¬(∀ x : ℝ, f x = f (-x)) ∨ ¬(∀ x : ℝ, f (-x) = -f x) := by
  sorry

-- Proposition 3
theorem range_shift_not_equal (f : ℝ → ℝ) :
  (∀ y : ℝ, y ∈ Set.range f ↔ -2 ≤ y ∧ y ≤ 2) →
  ¬(∀ y : ℝ, y ∈ Set.range (fun x ↦ f (x + 1)) ↔ -3 ≤ y ∧ y ≤ 1) := by
  sorry

-- Proposition 4
def g (x : ℝ) : ℝ := |3 - x^2|

theorem no_single_intersection :
  ¬(∃ a : ℝ, ∃! x : ℝ, g x = a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_imply_neg_a_f_not_even_and_odd_range_shift_not_equal_no_single_intersection_l483_48371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_necklaces_bijection_l483_48357

/-- A necklace is represented as a list of colors. -/
def Necklace (colors : Type*) := List colors

/-- A necklace is prime if it cannot be decomposed into equal substrings. -/
def isPrime {colors : Type*} (necklace : Necklace colors) : Prop :=
  ∀ k : ℕ, 1 < k → k ∣ necklace.length → ¬ (∃ s : List colors, necklace = List.join (List.replicate k s))

/-- Two necklaces are equivalent if they can be rotated to match each other. -/
def areEquivalent {colors : Type*} (a b : Necklace colors) : Prop :=
  a.length = b.length ∧ ∃ k : ℕ, k < a.length ∧ a = b.rotateLeft k

/-- The set of prime necklaces with n beads and c colors. -/
def PrimeNecklaces (n : ℕ) (c : Type*) : Set (Necklace c) :=
  {necklace | necklace.length = n ∧ isPrime necklace}

/-- There exists a bijection between prime necklaces with n beads and q^n colors,
    and n copies of prime necklaces with n^2 beads and q colors. -/
theorem prime_necklaces_bijection (n q : ℕ) :
  ∃ f : PrimeNecklaces n (Fin (q^n)) → Fin n × PrimeNecklaces (n^2) (Fin q),
    Function.Bijective f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_necklaces_bijection_l483_48357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_angle_tangent_l483_48344

/-- Given a line y = x + 1 intersecting an ellipse mx^2 + ny^2 = 1 (where m > n > 0) at points A and B,
    if the x-coordinate of the midpoint of AB is -1/3, then the tangent value of the angle between
    the two asymptotes of the hyperbola x^2/m^2 - y^2/n^2 = 1 is 4/3. -/
theorem asymptote_angle_tangent (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m > n) :
  let line := λ x : ℝ ↦ x + 1
  let ellipse := λ x y : ℝ ↦ m * x^2 + n * y^2 = 1
  let hyperbola := λ x y : ℝ ↦ x^2 / m^2 - y^2 / n^2 = 1
  let midpoint_x := -1/3
  (∃ A B : ℝ × ℝ, ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
    A.2 = line A.1 ∧ B.2 = line B.1 ∧
    (A.1 + B.1) / 2 = midpoint_x) →
  (let asymptote_angle_tan := 2 * (n / m) / (1 - (n / m)^2)
   asymptote_angle_tan = 4/3)
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_angle_tangent_l483_48344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_perimeter_ratio_l483_48380

/-- The ratio of area to perimeter for an equilateral triangle with side length 8 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 8
  let area : ℝ := (Real.sqrt 3 / 4) * side_length^2
  let perimeter : ℝ := 3 * side_length
  area / perimeter = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_perimeter_ratio_l483_48380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_average_speed_l483_48367

/-- Calculates the average speed given distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem jane_average_speed :
  let distance : ℝ := 160
  let time : ℝ := 6
  average_speed distance time = 80 / 3 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- This should now be a simple numerical equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_average_speed_l483_48367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_books_sold_wednesday_correct_l483_48324

def books_sold_wednesday (initial_stock : ℕ) (monday_sales tuesday_sales thursday_sales friday_sales : ℕ) 
  (unsold_percentage : ℚ) : ℕ :=
  let total_sales_excl_wednesday := monday_sales + tuesday_sales + thursday_sales + friday_sales
  let sold_percentage : ℚ := 1 - unsold_percentage
  let total_sales : ℕ := (sold_percentage * initial_stock).floor.toNat
  total_sales - total_sales_excl_wednesday

#eval books_sold_wednesday 800 62 62 48 40 (66/100)

theorem books_sold_wednesday_correct : 
  books_sold_wednesday 800 62 62 48 40 (66/100) = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_books_sold_wednesday_correct_l483_48324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_simplification_l483_48345

variable {V : Type*} [AddCommGroup V]

variable (AD MB BM BC CM AB CD OC OA : V)

theorem vector_simplification :
  ((AD + MB) + (BC + CM) = AD) ∧
  ((AB + CD) + BC = AD) ∧
  (OC - OA + CD = AD) ∧
  (MB + AD - BM ≠ AD) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_simplification_l483_48345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_item_value_theorem_l483_48306

/-- The initial value of the item in dollars -/
def C : ℝ := sorry

/-- The percentage of loss and profit -/
def x : ℝ := sorry

/-- The final selling price after profit -/
def S : ℝ := sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem item_value_theorem (h1 : 100 = C * (1 - x / 100))
                           (h2 : S = 100 * (1 + x / 100))
                           (h3 : S - C = 10 / 9) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_item_value_theorem_l483_48306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_QAMB_l483_48353

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define a point on the x-axis
def point_on_x_axis (x : ℝ) : ℝ × ℝ := (x, 0)

-- Define the area of quadrilateral QAMB
noncomputable def area_QAMB (Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - 0)^2 + (Q.2 - 2)^2 - 1)

theorem min_area_QAMB :
  ∃ (Q : ℝ × ℝ), 
    (∃ x, Q = point_on_x_axis x) ∧ 
    (∀ P : ℝ × ℝ, (∃ x, P = point_on_x_axis x) → area_QAMB Q ≤ area_QAMB P) ∧
    area_QAMB Q = Real.sqrt 3 := by
  sorry

#check min_area_QAMB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_QAMB_l483_48353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l483_48313

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) → 
  a ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l483_48313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_l483_48363

/-- The width of a rectangle given specific conditions -/
theorem rectangle_width (w : ℝ) (h1 : 6 * w = 8 * Real.pi + 10) (h2 : 0 < w) : w = (4 * Real.pi + 5) / 3 := by
  sorry

/-- Approximate numerical evaluation of the width -/
def approximate_width : ℚ :=
  (4 * 3.14159 + 5) / 3

#eval approximate_width

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_l483_48363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_two_composites_not_prime_probability_prime_product_is_zero_l483_48317

def first_eight_composites : List Nat := [4, 6, 8, 9, 10, 12, 14, 15]

theorem product_of_two_composites_not_prime (a b : Nat) 
  (ha : a ∈ first_eight_composites) (hb : b ∈ first_eight_composites) (hab : a ≠ b) : 
  ¬ Nat.Prime (a * b) :=
sorry

theorem probability_prime_product_is_zero : 
  (Finset.card (Finset.filter (fun p : Nat × Nat => 
    p.1 ∈ first_eight_composites ∧ 
    p.2 ∈ first_eight_composites ∧ 
    p.1 ≠ p.2 ∧ 
    Nat.Prime (p.1 * p.2)) (Finset.product (Finset.range 16) (Finset.range 16)))) /
  (Finset.card (Finset.filter (fun p : Nat × Nat => 
    p.1 ∈ first_eight_composites ∧ 
    p.2 ∈ first_eight_composites ∧ 
    p.1 ≠ p.2) (Finset.product (Finset.range 16) (Finset.range 16)))) = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_two_composites_not_prime_probability_prime_product_is_zero_l483_48317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_edge_angle_approx_l483_48304

/-- The angle between any two edges meeting at a vertex of a regular tetrahedron -/
noncomputable def regular_tetrahedron_edge_angle : ℝ :=
  Real.arccos (-1 / 3)

/-- Theorem: The angle between any two edges meeting at a vertex of a regular tetrahedron
    is approximately 109.47 degrees -/
theorem regular_tetrahedron_edge_angle_approx :
  ⌊regular_tetrahedron_edge_angle * 180 / Real.pi⌋ = 109 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_edge_angle_approx_l483_48304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_two_integers_with_property_l483_48343

/-- The number of divisors function -/
noncomputable def tau (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The property we're looking for -/
def has_property (n : ℕ) : Prop := tau n + tau (n + 2) = 8

theorem sum_of_two_integers_with_property :
  ∃ (a b : ℕ), a < b ∧ has_property a ∧ has_property b ∧ a + b = 36 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_two_integers_with_property_l483_48343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inscribed_quadrilateral_l483_48376

-- Define the rhombus and circle
def Point : Type := ℝ × ℝ

structure Circle where
  center : Point
  radius : ℝ

def Rhombus (H M T M₂ : Point) : Prop := sorry

-- Define the tangent points
def TangentPoints (ω : Circle) (H M₁ M₂ T A I M E : Point) : Prop := sorry

-- Define the areas
noncomputable def AreaRhombus (H M T M₂ : Point) : ℝ := sorry
noncomputable def AreaTriangle (E M T : Point) : ℝ := sorry
noncomputable def AreaQuadrilateral (A I M E : Point) : ℝ := sorry

theorem area_of_inscribed_quadrilateral 
  (H M T M₂ M₁ A I M E : Point) (ω : Circle) :
  Rhombus H M T M₂ →
  TangentPoints ω H M₁ M₂ T A I M E →
  AreaRhombus H M T M₂ = 1440 →
  AreaTriangle E M T = 405 →
  AreaQuadrilateral A I M E = 540 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inscribed_quadrilateral_l483_48376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_safes_theorem_l483_48303

/-- Represents the state of a safe (locked or unlocked) -/
inductive SafeState
| Locked
| Unlocked

/-- Toggles the state of a safe -/
def toggleState (s : SafeState) : SafeState :=
  match s with
  | SafeState.Locked => SafeState.Unlocked
  | SafeState.Unlocked => SafeState.Locked

/-- Represents a sequence of n safes -/
def SafeSequence (n : ℕ) := Fin n → SafeState

/-- Performs operation T_k on a sequence of safes -/
def applyOperation (k : ℕ) (s : SafeSequence n) : SafeSequence n :=
  λ i ↦ if (i.val + 1) % k = 0 then toggleState (s i) else s i

/-- Applies all operations T_1 to T_n on a sequence of safes -/
def applyAllOperations (n : ℕ) (s : SafeSequence n) : SafeSequence n :=
  (List.range n).foldl (λ acc k ↦ applyOperation (k + 1) acc) s

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The main theorem to be proved -/
theorem safes_theorem (n : ℕ) :
  let initialSafes : SafeSequence n := λ _ ↦ SafeState.Locked
  let finalSafes := applyAllOperations n initialSafes
  ∀ i : Fin n, finalSafes i = SafeState.Unlocked ↔ isPerfectSquare (i.val + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_safes_theorem_l483_48303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounded_quotient_in_result_set_l483_48372

/-- Round down a real number to two decimal places -/
noncomputable def roundDownTwoDecimals (x : ℝ) : ℝ :=
  ⌊x * 100⌋ / 100

/-- The set of possible resulting values -/
def resultSet : Set ℝ :=
  {x | x ∈ Set.Icc (50/100) 1 ∧ ∃ (n : ℕ), x = n/100}

theorem rounded_quotient_in_result_set (α : ℝ) (hα : α > 0) :
  roundDownTwoDecimals (roundDownTwoDecimals α / α) ∈ resultSet := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounded_quotient_in_result_set_l483_48372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_g_inv_at_three_point_five_l483_48332

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 3 * x - 7

-- Define the inverse function g_inv
noncomputable def g_inv (x : ℝ) : ℝ := (x + 7) / 3

-- Theorem statement
theorem g_equals_g_inv_at_three_point_five :
  g (3.5) = g_inv (3.5) := by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_g_inv_at_three_point_five_l483_48332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l483_48314

/-- A trapezoid with right angles -/
structure RightTrapezoid where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  right_angle_BAD : True
  right_angle_ADC : True

/-- The area of a right trapezoid -/
noncomputable def area (t : RightTrapezoid) : ℝ :=
  (t.AB + t.CD) * (Real.sqrt (t.BC^2 - (t.CD - t.AB)^2)) / 2

/-- Theorem: The area of the specific trapezoid is 15 -/
theorem specific_trapezoid_area :
  let t : RightTrapezoid := {
    AB := 3,
    BC := 5,
    CD := 7,
    right_angle_BAD := True.intro,
    right_angle_ADC := True.intro
  }
  area t = 15 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l483_48314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_sequence_fourth_term_l483_48368

def sequence_a (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- Define a₀ to complete the sequence
  | 1 => 3  -- a₁ = 3 as calculated in the solution
  | n + 1 => 2 * sequence_a n - n + 1

theorem sequence_formula (n : ℕ) (h : n > 0) : sequence_a n = 2^n + n := by
  sorry

theorem sequence_fourth_term : sequence_a 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_sequence_fourth_term_l483_48368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_l483_48329

theorem sum_of_factors (m n p q : ℕ) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0 →
  (8 - m) * (8 - n) * (8 - p) * (8 - q) = 9 →
  m + n + p + q = 32 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_l483_48329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_difference_approximately_10142_l483_48311

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  principal * (1 + rate / n) ^ (n * t)

noncomputable def plan1_payment (principal : ℝ) (rate : ℝ) : ℝ :=
  let half_time := 7.5
  let full_time := 15.0
  let half_balance := compound_interest principal rate 4 half_time
  let half_payment := half_balance / 2
  let remaining_balance := half_balance - half_payment
  half_payment + compound_interest remaining_balance rate 4 half_time

noncomputable def plan2_payment (principal : ℝ) (rate : ℝ) (fee : ℝ) : ℝ :=
  compound_interest principal rate 1 15 + fee

theorem payment_difference_approximately_10142 :
  let principal := 12000
  let rate := 0.08
  let fee := 500
  ⌊plan2_payment principal rate fee - plan1_payment principal rate⌋ = 10142 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_difference_approximately_10142_l483_48311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l483_48390

/-- Limacon curve in polar coordinates -/
def limacon (ρ θ : ℝ) : Prop := 2 * ρ * Real.cos θ = 1

/-- Circle in polar coordinates -/
def circle_curve (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- The length of the chord of intersection between the limacon and circle -/
noncomputable def chord_length : ℝ := Real.sqrt 3

theorem intersection_chord_length :
  ∀ (ρ₁ θ₁ ρ₂ θ₂ : ℝ),
  limacon ρ₁ θ₁ ∧ circle_curve ρ₁ θ₁ ∧
  limacon ρ₂ θ₂ ∧ circle_curve ρ₂ θ₂ ∧
  (ρ₁ * Real.cos θ₁ ≠ ρ₂ * Real.cos θ₂ ∨ ρ₁ * Real.sin θ₁ ≠ ρ₂ * Real.sin θ₂) →
  Real.sqrt ((ρ₁ * Real.cos θ₁ - ρ₂ * Real.cos θ₂)^2 + (ρ₁ * Real.sin θ₁ - ρ₂ * Real.sin θ₂)^2) = chord_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l483_48390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l483_48370

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (2 - x)

-- State the theorem
theorem solution_set_of_inequality :
  ∀ x : ℝ, f (2 * x + 1) < f x ↔ -1 < x ∧ x < 1/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l483_48370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l483_48320

theorem power_equation_solution (x : ℝ) : (27 : ℝ)^x * (27 : ℝ)^x * (27 : ℝ)^x = (243 : ℝ)^3 → x = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l483_48320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangles_in_ellipse_l483_48323

/-- Represents the number of isosceles right-angled triangles that can be constructed inside an ellipse -/
noncomputable def num_triangles (a : ℝ) : ℕ :=
  if a > Real.sqrt 3 then 3 else 1

/-- Theorem stating the number of isosceles right-angled triangles in an ellipse -/
theorem isosceles_right_triangles_in_ellipse (a : ℝ) (h1 : a > 1) :
  num_triangles a = (if a > Real.sqrt 3 then 3 else 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangles_in_ellipse_l483_48323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_addition_problem_solvable_l483_48312

/-- A type representing digits from 0 to 9 -/
def Digit := Fin 10

/-- A structure representing the letter-digit assignment -/
structure LetterAssignment where
  R : Digit
  A : Digit
  T : Digit
  M : Digit
  D : Digit
  U : Digit
  L : Digit
  O : Digit
  H : Digit
  Y : Digit

/-- The main theorem statement -/
theorem letter_addition_problem_solvable : 
  ∃ (assignment : LetterAssignment),
    -- All digits are different
    (assignment.R ≠ assignment.A) ∧ (assignment.R ≠ assignment.T) ∧ (assignment.R ≠ assignment.M) ∧
    (assignment.R ≠ assignment.D) ∧ (assignment.R ≠ assignment.U) ∧ (assignment.R ≠ assignment.L) ∧
    (assignment.R ≠ assignment.O) ∧ (assignment.R ≠ assignment.H) ∧ (assignment.R ≠ assignment.Y) ∧
    (assignment.A ≠ assignment.T) ∧ (assignment.A ≠ assignment.M) ∧ (assignment.A ≠ assignment.D) ∧
    (assignment.A ≠ assignment.U) ∧ (assignment.A ≠ assignment.L) ∧ (assignment.A ≠ assignment.O) ∧
    (assignment.A ≠ assignment.H) ∧ (assignment.A ≠ assignment.Y) ∧ (assignment.T ≠ assignment.M) ∧
    (assignment.T ≠ assignment.D) ∧ (assignment.T ≠ assignment.U) ∧ (assignment.T ≠ assignment.L) ∧
    (assignment.T ≠ assignment.O) ∧ (assignment.T ≠ assignment.H) ∧ (assignment.T ≠ assignment.Y) ∧
    (assignment.M ≠ assignment.D) ∧ (assignment.M ≠ assignment.U) ∧ (assignment.M ≠ assignment.L) ∧
    (assignment.M ≠ assignment.O) ∧ (assignment.M ≠ assignment.H) ∧ (assignment.M ≠ assignment.Y) ∧
    (assignment.D ≠ assignment.U) ∧ (assignment.D ≠ assignment.L) ∧ (assignment.D ≠ assignment.O) ∧
    (assignment.D ≠ assignment.H) ∧ (assignment.D ≠ assignment.Y) ∧ (assignment.U ≠ assignment.L) ∧
    (assignment.U ≠ assignment.O) ∧ (assignment.U ≠ assignment.H) ∧ (assignment.U ≠ assignment.Y) ∧
    (assignment.L ≠ assignment.O) ∧ (assignment.L ≠ assignment.H) ∧ (assignment.L ≠ assignment.Y) ∧
    (assignment.O ≠ assignment.H) ∧ (assignment.O ≠ assignment.Y) ∧ (assignment.H ≠ assignment.Y) ∧
    -- Specific conditions
    (assignment.U.val = assignment.R.val + 1) ∧
    (assignment.A = ⟨9, by norm_num⟩) ∧
    (assignment.H = ⟨8, by norm_num⟩) ∧
    (assignment.O.val = assignment.T.val + assignment.R.val - 9) ∧
    (assignment.Y.val = assignment.M.val + assignment.D.val) ∧
    -- The addition is correct
    (10000 * assignment.R.val + 1000 * assignment.A.val + 100 * assignment.T.val + 
     10 * assignment.A.val + assignment.M.val + 
     100 * assignment.R.val + 10 * assignment.A.val + assignment.D.val =
     10000 * assignment.U.val + 1000 * assignment.L.val + 100 * assignment.O.val + 
     10 * assignment.H.val + assignment.Y.val) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_addition_problem_solvable_l483_48312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_over_two_l483_48394

theorem angle_sum_is_pi_over_two (α β : ℝ)
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin (α + β) = (Real.sin α) ^ 2 + (Real.sin β) ^ 2) :
  α + β = π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_over_two_l483_48394
