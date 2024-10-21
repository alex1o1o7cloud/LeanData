import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_votes_is_120_brenda_votes_fraction_l493_49333

-- Define the total number of votes as a natural number
def total_votes : ℕ := 120

-- Define Brenda's votes
def brenda_votes : ℕ := 36

-- Define the fraction of votes Brenda received
def brenda_fraction : ℚ := 3 / 10

-- Theorem stating that the total number of votes is 120
theorem total_votes_is_120 : total_votes = 120 := by
  rfl

-- Proof that Brenda's votes are 3/10 of the total votes
theorem brenda_votes_fraction :
  (brenda_votes : ℚ) / total_votes = brenda_fraction := by
  norm_num
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_votes_is_120_brenda_votes_fraction_l493_49333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tv_more_expensive_per_square_inch_l493_49399

/-- Represents a TV with its dimensions and cost -/
structure TV where
  width : ℚ
  height : ℚ
  cost : ℚ

/-- Calculates the area of a TV -/
def TV.area (tv : TV) : ℚ := tv.width * tv.height

/-- Calculates the cost per square inch of a TV -/
def TV.costPerSquareInch (tv : TV) : ℚ := tv.cost / tv.area

/-- The first TV -/
def firstTV : TV := { width := 24, height := 16, cost := 672 }

/-- The new TV -/
def newTV : TV := { width := 48, height := 32, cost := 1152 }

/-- Theorem: The first TV is $1 more expensive per square inch than the new TV -/
theorem first_tv_more_expensive_per_square_inch :
  firstTV.costPerSquareInch - newTV.costPerSquareInch = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tv_more_expensive_per_square_inch_l493_49399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l493_49351

/-- The equation of a circle in the xy-plane -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 3

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of a line in the xy-plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- A line is tangent to a circle if it touches the circle at exactly one point -/
def IsTangent (l : Line) (c : ℝ → ℝ → Prop) : Prop := sorry

theorem tangent_line_to_circle (p : Point) (h1 : p.x = 1 ∧ p.y = 2) 
  (h2 : Circle p.x p.y) : 
  ∃ l : Line, IsTangent l Circle ∧ l.slope = 0 ∧ l.y_intercept = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l493_49351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l493_49367

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + 2^x - x / Real.log 2

-- State the theorem
theorem tangent_slope_at_one :
  HasDerivAt f (2 * Real.log 2) 1 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l493_49367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_germanMeaslesCases1995_l493_49372

/-- Linear decrease of German measles cases from 1960 to 2000 -/
noncomputable def germanMeaslesCases (year : ℕ) : ℝ :=
  450000 - (450000 - 50) * ((year - 1960) / 40 : ℝ)

/-- The number of German measles cases in 1995 -/
theorem germanMeaslesCases1995 :
  ⌊germanMeaslesCases 1995⌋ = 56041 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_germanMeaslesCases1995_l493_49372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_colorings_eq_667_l493_49393

-- Define the colors for squares and triangles
inductive SquareColor : Type
| Blue : SquareColor
| Red : SquareColor
| Green : SquareColor

inductive TriangleColor : Type
| Blue : TriangleColor
| Red : TriangleColor
| Yellow : TriangleColor

-- Define the figure
structure Figure :=
  (squares : Fin 5 → SquareColor)
  (triangles : Fin 4 → TriangleColor)

-- Define a predicate for valid colorings
def is_valid_coloring (f : Figure) : Prop :=
  -- Add conditions to ensure adjacent polygons have different colors
  -- This is a simplified version and doesn't capture all adjacency rules
  ∀ i j, i ≠ j → f.squares i ≠ f.squares j ∧ f.triangles i ≠ f.triangles j

-- Assume Fintype and DecidablePred instances
instance : Fintype Figure := sorry
instance : DecidablePred is_valid_coloring := sorry

-- Define the number of valid colorings
def num_valid_colorings : ℕ :=
  (Finset.filter is_valid_coloring (Finset.univ : Finset Figure)).card

-- State the theorem
theorem num_valid_colorings_eq_667 : num_valid_colorings = 667 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_colorings_eq_667_l493_49393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_candidate_percentage_l493_49306

def election_votes : List Nat := [2500, 5000, 15000]

def total_votes : Nat := election_votes.sum

noncomputable def winning_votes : Nat := 
  match election_votes.maximum? with
  | some n => n
  | none => 0

theorem winning_candidate_percentage :
  (winning_votes : ℚ) / (total_votes : ℚ) = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_candidate_percentage_l493_49306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_fraction_to_add_l493_49354

open BigOperators

def series_sum (n : ℕ) : ℚ :=
  ∑ i in Finset.range (n - 1), 1 / ((i + 2 : ℚ) * (i + 3 : ℚ))

theorem least_fraction_to_add (n : ℕ) (h : n = 22) : 
  ∃ (x : ℚ), x = 15 / 22 ∧ series_sum n + x = 1 ∧ 
  ∀ (y : ℚ), y > 0 ∧ y < x → series_sum n + y < 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_fraction_to_add_l493_49354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_curve_points_theorem_l493_49384

/-- Circle O -/
def circleO (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Curve C -/
def curveC (x y t : ℝ) : Prop := y = 3 * |x - t|

/-- Distance ratio -/
noncomputable def distance_ratio (x₀ y₀ m n s p : ℝ) : ℝ :=
  ((x₀ - m)^2 + (y₀ - n)^2) / ((x₀ - s)^2 + (y₀ - p)^2)

theorem circle_curve_points_theorem (m n s p : ℕ+) (k : ℝ) 
  (hk : k > 1)
  (h_circle : ∀ x₀ y₀, circleO x₀ y₀ → ∃ t, curveC x₀ y₀ t)
  (h_curve_A : ∃ t, curveC (m : ℝ) (n : ℝ) t)
  (h_curve_B : ∃ t, curveC (s : ℝ) (p : ℝ) t)
  (h_ratio : ∀ x₀ y₀, circleO x₀ y₀ → distance_ratio x₀ y₀ (m : ℝ) (n : ℝ) (s : ℝ) (p : ℝ) = k^2) :
  (m : ℕ) ^ (s : ℕ) - (n : ℕ) ^ (p : ℕ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_curve_points_theorem_l493_49384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elyse_had_100_pieces_l493_49375

-- Define the initial number of gum pieces for each person
def elyse_initial : ℕ := sorry
def rick : ℕ := sorry
def shane_initial : ℕ := sorry
def shane_final : ℕ := 14

-- Define the relationships between the numbers
axiom rick_from_elyse : rick = elyse_initial / 2
axiom shane_from_rick : shane_initial = rick / 2
axiom shane_chewed : shane_initial = shane_final + 11

-- Theorem to prove
theorem elyse_had_100_pieces : elyse_initial = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elyse_had_100_pieces_l493_49375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_largest_shapes_l493_49366

structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

noncomputable def largest_cube_side (t : Tetrahedron) : ℝ :=
  (t.a * t.b * t.c) / (t.a * t.b + t.b * t.c + t.a * t.c)

noncomputable def largest_parallelepiped_dimensions (t : Tetrahedron) : ℝ × ℝ × ℝ :=
  (t.a / 3, t.b / 3, t.c / 3)

theorem tetrahedron_largest_shapes (t : Tetrahedron) :
  (largest_cube_side t = (t.a * t.b * t.c) / (t.a * t.b + t.b * t.c + t.a * t.c)) ∧
  (largest_parallelepiped_dimensions t = (t.a / 3, t.b / 3, t.c / 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_largest_shapes_l493_49366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_grade_students_l493_49318

/-- The number of students in the fifth grade at Parkway Elementary School -/
def total_students : ℕ := 420

/-- The number of boys in the fifth grade -/
def boys : ℕ := 320

/-- The number of students playing soccer -/
def soccer_players : ℕ := 250

/-- The percentage of soccer players who are boys -/
def boys_soccer_percentage : ℚ := 86 / 100

/-- The number of girls not playing soccer -/
def girls_not_soccer : ℕ := 65

/-- Theorem stating the total number of students in the fifth grade -/
theorem fifth_grade_students : 
  total_students = boys + (soccer_players - (boys_soccer_percentage * ↑soccer_players).floor) + girls_not_soccer :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_grade_students_l493_49318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l493_49326

/-- The distance from a point (x₀, y₀) to a line ax + by + c = 0 is |ax₀ + by₀ + c| / √(a² + b²) -/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

theorem distance_circle_center_to_line :
  let circle_center_x : ℝ := 1
  let circle_center_y : ℝ := 6
  let line_a : ℝ := 1
  let line_b : ℝ := -1
  let line_c : ℝ := -1
  distance_point_to_line circle_center_x circle_center_y line_a line_b line_c = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l493_49326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_l493_49365

/-- Represents the price of oil in Rupees per kg -/
structure OilPrice where
  price : ℝ
  price_positive : price > 0

/-- Calculates the reduced price after a percentage reduction -/
noncomputable def reduced_price (original : OilPrice) (reduction_percent : ℝ) : ℝ :=
  original.price * (1 - reduction_percent / 100)

/-- Calculates the amount of oil that can be purchased with a given amount of money -/
noncomputable def oil_amount (price : ℝ) (money : ℝ) : ℝ :=
  money / price

theorem oil_price_reduction (original : OilPrice) :
  let reduction_percent : ℝ := 40
  let budget : ℝ := 2000
  let extra_amount : ℝ := 10
  let reduced : ℝ := reduced_price original reduction_percent
  oil_amount reduced budget = oil_amount original.price budget + extra_amount →
  ∃ ε > 0, |reduced - 80| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_l493_49365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l493_49370

/-- The function f(x) = 4x / (x - 1) -/
noncomputable def f (x : ℝ) : ℝ := 4 * x / (x - 1)

/-- The function g(x) satisfying g(2-x) + g(x) = 8 -/
noncomputable def g : ℝ → ℝ := sorry

theorem f_properties :
  (∃ x₁ x₂ : ℝ, x₁ ≠ 1 ∧ x₂ ≠ 1 ∧ x₁ < x₂ ∧ f x₁ < f x₂) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, (∀ x : ℝ, g (2 - x) + g x = 8) → f x₁ = y₁ → f x₂ = y₂ → g x₁ = y₁ → g x₂ = y₂ → x₁ + y₁ + x₂ + y₂ = 10) ∧
  (∀ x₁ x₂ : ℝ, x₁ < 1 → x₂ < 1 → (f x₁ + f x₂) / 2 ≤ f ((x₁ + x₂) / 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l493_49370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_george_school_speed_l493_49361

/-- The speed required for George to arrive at school on time -/
noncomputable def required_speed (total_distance : ℝ) (usual_speed : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) (remaining_distance : ℝ) : ℝ :=
  let total_time := total_distance / usual_speed
  let first_part_time := first_part_distance / first_part_speed
  let remaining_time := total_time - first_part_time
  remaining_distance / remaining_time

theorem george_school_speed :
  required_speed 1.5 3 1 2.5 0.5 = 5 := by
  -- Unfold the definition of required_speed
  unfold required_speed
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_george_school_speed_l493_49361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_web_design_competition_arrangements_l493_49330

/-- The number of possible arrangements for 5 students in a ranking where
    one student is not in 1st place and another student is in 3rd place. -/
theorem web_design_competition_arrangements :
  (Nat.factorial 4 - Nat.factorial 3) = 18 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_web_design_competition_arrangements_l493_49330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l493_49328

theorem problem_solution (a b c d : ℝ) 
  (h1 : a^2 + b^2 + c^2 + 4 = 2*d + Real.sqrt (2*a + 2*b + 2*c - 3*d))
  (h2 : a + b + c = 3) :
  d = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l493_49328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carnival_time_calculation_l493_49303

/-- The time Jimmy was given to sell pizzas at the carnival --/
noncomputable def carnival_time (total_flour : ℚ) (flour_per_pizza : ℚ) (time_per_pizza : ℚ) (leftover_pizzas : ℕ) : ℚ :=
  let flour_used := total_flour - (leftover_pizzas : ℚ) * flour_per_pizza
  let pizzas_made := flour_used / flour_per_pizza
  let total_minutes := pizzas_made * time_per_pizza
  total_minutes / 60

/-- Theorem stating the carnival time given the problem conditions --/
theorem carnival_time_calculation :
  carnival_time 22 (1/2) 10 2 = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carnival_time_calculation_l493_49303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_race_total_races_l493_49374

/-- Represents a participant in the relay race -/
inductive Participant
| Alex
| Betty
| Clara
| Dave

/-- The number of races each participant ran -/
def races_run (p : Participant) : ℕ :=
  match p with
  | Participant.Alex => 5
  | Participant.Betty => 3
  | Participant.Clara => 4
  | Participant.Dave => 8

/-- The set of all participants -/
def all_participants : List Participant :=
  [Participant.Alex, Participant.Betty, Participant.Clara, Participant.Dave]

theorem relay_race_total_races :
  (∃ p₁ p₂, p₁ ∈ all_participants ∧ p₂ ∈ all_participants ∧
    races_run p₁ = 8 ∧ races_run p₂ = 3 ∧
    ∀ p ∈ all_participants, races_run p ≤ races_run p₁ ∧ races_run p₂ ≤ races_run p) →
  (all_participants.map races_run).sum / 2 = 10 := by
  sorry

#eval (all_participants.map races_run).sum / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_race_total_races_l493_49374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_and_bisection_l493_49388

noncomputable section

-- Define the line l
def line_l (x y : ℝ) : Prop := 5 * x - 7 * y - 70 = 0

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define point Q
def point_Q : ℝ × ℝ := (25/14, -9/10)

-- Define a point P on line l
def point_on_line_l (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  line_l x y

-- Define tangent points M and N
def tangent_points (P M N : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xm, ym) := M
  let (xn, yn) := N
  ellipse xm ym ∧ ellipse xn yn ∧
  (xm * xp / 25 + ym * yp / 9 = 1) ∧
  (xn * xp / 25 + yn * yp / 9 = 1)

-- Define line MN passing through Q
def line_MN_through_Q (M N : ℝ × ℝ) : Prop :=
  let (xm, ym) := M
  let (xn, yn) := N
  let (xq, yq) := point_Q
  (yn - ym) * (xq - xm) = (xn - xm) * (yq - ym)

-- Define MN parallel to l
def MN_parallel_l (M N : ℝ × ℝ) : Prop :=
  let (xm, ym) := M
  let (xn, yn) := N
  5 * (xn - xm) = 7 * (yn - ym)

-- Define Q bisecting MN
def Q_bisects_MN (M N : ℝ × ℝ) : Prop :=
  let (xm, ym) := M
  let (xn, yn) := N
  let (xq, yq) := point_Q
  2 * xq = xm + xn ∧ 2 * yq = ym + yn

-- Main theorem
theorem fixed_point_and_bisection 
  (P M N : ℝ × ℝ) 
  (h_P : point_on_line_l P) 
  (h_MN : tangent_points P M N) :
  line_MN_through_Q M N ∧ 
  (MN_parallel_l M N → Q_bisects_MN M N) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_and_bisection_l493_49388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_solution_l493_49344

theorem factorial_equation_solution :
  ∀ n m : ℕ, (Nat.factorial (n + 1)) * (Nat.factorial (m + 1)) = Nat.factorial (n + m) ↔ 
    (n = 2 ∧ m = 4) ∨ (n = 4 ∧ m = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_solution_l493_49344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_equidistant_l493_49357

/-- Given a point P on the line 2x - 3y + 6 = 0, with O as the origin and A(-1, 1),
    if |PO| = |PA|, then P has coordinates (3, 4) -/
theorem point_on_line_equidistant (P : ℝ × ℝ) :
  (2 * P.1 - 3 * P.2 + 6 = 0) →
  (P.1^2 + P.2^2 = (P.1 + 1)^2 + (P.2 - 1)^2) →
  P = (3, 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_equidistant_l493_49357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fidos_yard_l493_49363

theorem fidos_yard (s : ℝ) (h : s > 0) : 
  let hexagon_area := 3 * Real.sqrt 3 / 2 * s^2
  let circle_area := Real.pi * s^2
  let ratio := circle_area / hexagon_area
  ∃ (a b : ℕ), (a > 0 ∧ b > 0) ∧ 
    ratio = Real.sqrt a / b * Real.pi ∧
    a * b = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fidos_yard_l493_49363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l493_49376

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + x) + x / (1 - x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ∈ Set.Icc (-1) 1 ∨ x > 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l493_49376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l493_49353

-- Define the simple interest calculation
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

-- Theorem statement
theorem principal_calculation (interest rate time : ℝ) 
  (h1 : interest = 4016.25)
  (h2 : rate = 9)
  (h3 : time = 5) :
  ∃ (principal : ℝ), simple_interest principal rate time = interest ∧ principal = 8925 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l493_49353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowTie_solution_l493_49317

/-- Custom operator definition -/
noncomputable def bowTie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

/-- Theorem statement -/
theorem bowTie_solution :
  ∃ z : ℝ, bowTie 7 z = 15 ∧ z = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowTie_solution_l493_49317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_0_values_l493_49313

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 3^(-x) - 2 else Real.sqrt x

-- State the theorem
theorem x_0_values (x_0 : ℝ) (h : f x_0 = 1) : x_0 = 1 ∨ x_0 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_0_values_l493_49313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l493_49349

/-- The speed of a train given its length and time to cross a fixed point. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

/-- Theorem: A train with length 480 meters that crosses a point in 16 seconds has a speed of 30 meters per second. -/
theorem train_speed_calculation :
  train_speed 480 16 = 30 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l493_49349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l493_49389

/-- Custom operation ⊕ for non-zero real numbers -/
noncomputable def circplus (a b : ℝ) : ℝ := 1 / b - 1 / a

/-- Theorem stating that if 2 ⊕ (2x-1) = 1, then x = 5/6 -/
theorem solve_equation (x : ℝ) (h1 : 2 ≠ 0) (h2 : 2*x - 1 ≠ 0) :
  circplus 2 (2*x - 1) = 1 → x = 5/6 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l493_49389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equally_distant_x_coordinate_l493_49362

/-- The distance from a point (x, y) to a line ax + by + c = 0 --/
noncomputable def distanceToLine (x y a b c : ℝ) : ℝ :=
  (abs (a * x + b * y + c)) / (Real.sqrt (a^2 + b^2))

/-- The point (x, y) is equally distant from x-axis, y-axis, and line x + y = 5 --/
def equallyDistant (x y : ℝ) : Prop :=
  abs y = abs x ∧ abs y = distanceToLine x y 1 1 (-5)

theorem point_equally_distant_x_coordinate :
  ∀ x y : ℝ, equallyDistant x y → x = 5/2 := by
  sorry

#check point_equally_distant_x_coordinate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equally_distant_x_coordinate_l493_49362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_for_point_three_neg_four_l493_49341

/-- If the terminal side of angle α passes through the point (3, -4), then cos(α) = 3/5 -/
theorem cos_alpha_for_point_three_neg_four (α : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ r * Real.cos α = 3 ∧ r * Real.sin α = -4) →
  Real.cos α = 3/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_for_point_three_neg_four_l493_49341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_participants_l493_49381

/-- Represents a round-robin chess tournament. -/
structure Tournament where
  participants : Finset Nat
  draws : Nat → Nat → Bool
  round_robin : ∀ i j, i ∈ participants → j ∈ participants → i ≠ j → draws i j = draws j i

/-- The number of drawn games a participant has played with a given subset. -/
def num_draws (t : Tournament) (i : Nat) (s : Finset Nat) : Nat :=
  (s.filter (λ j => t.draws i j)).card

/-- The condition that for any subset, there's a participant with odd draws. -/
def odd_draws_condition (t : Tournament) : Prop :=
  ∀ s : Finset Nat, s ⊆ t.participants →
    ∃ i, i ∈ t.participants ∧ Odd (num_draws t i s)

theorem even_participants (t : Tournament) (h : odd_draws_condition t) :
  Even t.participants.card :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_participants_l493_49381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l493_49334

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The point coordinates -/
def point : ℝ × ℝ := (3, 0)

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- Theorem stating the existence of asymptotes and the distance to one of them -/
theorem distance_to_asymptote :
  ∃ (A B C : ℝ), 
    (∀ x y, hyperbola x y → (A * x + B * y + C = 0 ∨ A * x - B * y + C = 0)) ∧ 
    distance_point_to_line point.1 point.2 A B C = 9/5 := by
  sorry

#check distance_to_asymptote

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l493_49334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l493_49360

theorem constant_term_expansion (a : ℝ) : 
  (∃ (x : ℝ), (Real.sqrt x + a / x^2)^5 = 10 + (Real.sqrt x + a / x^2)^5 - 10) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l493_49360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_longer_leg_l493_49304

/-- A triangle with properties relevant to our problem -/
structure Triangle where
  hypotenuse : ℝ
  longer_leg : ℝ
  shorter_leg : ℝ
  is_30_60_90 : hypotenuse^2 = longer_leg^2 + shorter_leg^2 ∧ 
                longer_leg = shorter_leg * Real.sqrt 3 ∧
                hypotenuse = 2 * shorter_leg

/-- A sequence of four connected 30-60-90 triangles -/
structure TriangleSequence where
  largest_hypotenuse : ℝ
  triangles : Fin 4 → Triangle
  connected : ∀ i : Fin 3, (triangles i).hypotenuse = (triangles i.succ).longer_leg
  largest_hypotenuse_length : (triangles 0).hypotenuse = largest_hypotenuse
  largest_hypotenuse_is_16 : largest_hypotenuse = 16

theorem smallest_triangle_longer_leg 
  (ts : TriangleSequence) : (ts.triangles 3).longer_leg = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_longer_leg_l493_49304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_third_l493_49380

/-- Given a triangle ABC where the ratio of sines of angles is 5:7:8, prove that angle B is π/3 -/
theorem angle_B_is_pi_third (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = Real.pi)
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C)
  (h_sine_ratio : ∃ (k : ℝ), Real.sin A = 5*k ∧ Real.sin B = 7*k ∧ Real.sin C = 8*k) :
  B = Real.pi/3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_third_l493_49380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numerator_and_denominator_l493_49373

def repeating_decimal : ℚ := 45 / 99

theorem sum_of_numerator_and_denominator : ∃ (a b : ℕ), 
  repeating_decimal = a / b ∧ 
  Nat.Coprime a b ∧ 
  a + b = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numerator_and_denominator_l493_49373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_face_area_l493_49359

/-- The total area of the four triangular faces of a right, square-based pyramid -/
noncomputable def triangular_faces_area (base_edge : ℝ) (lateral_edge : ℝ) : ℝ :=
  4 * (1/2 * base_edge * Real.sqrt (lateral_edge^2 - (base_edge/2)^2))

/-- Theorem: The total area of the four triangular faces of a right, square-based pyramid
    with base edges of 8 units and lateral edges of 7 units is equal to 16√33 square units. -/
theorem pyramid_face_area :
  triangular_faces_area 8 7 = 16 * Real.sqrt 33 := by
  -- Unfold the definition of triangular_faces_area
  unfold triangular_faces_area
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_face_area_l493_49359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_two_l493_49323

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  1/2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_is_two :
  let P : ℝ × ℝ := (2, f 2)
  let tangent_line (x : ℝ) : ℝ := f 2  -- y = f(2) is the equation of the tangent line at x = 2
  let line_x_eq_1 (x : ℝ) : Prop := x = 1
  let line_y_eq_x (x y : ℝ) : Prop := y = x
  ∃ A B C : ℝ × ℝ,
    line_x_eq_1 A.1 ∧
    line_y_eq_x B.1 B.2 ∧
    tangent_line C.1 = C.2 ∧
    (triangle_area A B C = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_two_l493_49323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_order_l493_49386

-- Define the angles in radians
noncomputable def angle25 : ℝ := 25 * Real.pi / 180
noncomputable def angle41 : ℝ := 41 * Real.pi / 180
noncomputable def angle58 : ℝ := 58 * Real.pi / 180

-- Define the right triangle
structure RightTriangle where
  hypotenuse : ℝ
  adjacentLeg : ℝ
  cosine : ℝ
  angle : ℝ
  cosine_def : cosine = adjacentLeg / hypotenuse
  leg_length : adjacentLeg = 1

-- Define the three specific right triangles
noncomputable def triangle25 : RightTriangle := {
  hypotenuse := 1 / Real.cos angle25
  adjacentLeg := 1
  cosine := Real.cos angle25
  angle := angle25
  cosine_def := by simp
  leg_length := rfl
}

noncomputable def triangle41 : RightTriangle := {
  hypotenuse := 1 / Real.cos angle41
  adjacentLeg := 1
  cosine := Real.cos angle41
  angle := angle41
  cosine_def := by simp
  leg_length := rfl
}

noncomputable def triangle58 : RightTriangle := {
  hypotenuse := 1 / Real.cos angle58
  adjacentLeg := 1
  cosine := Real.cos angle58
  angle := angle58
  cosine_def := by simp
  leg_length := rfl
}

-- State the theorem
theorem cosine_order :
  triangle58.cosine < triangle41.cosine ∧ triangle41.cosine < triangle25.cosine := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_order_l493_49386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scrap_cookie_radius_proof_l493_49387

/-- The radius of a small cookie in inches -/
def small_cookie_radius : ℝ := 0.5

/-- The number of small cookies cut from the large cookie dough -/
def num_small_cookies : ℕ := 9

/-- The diameter of the large cookie dough in terms of small cookie widths -/
def large_cookie_diameter_ratio : ℝ := 6

/-- The radius of the cookie formed from leftover scrap -/
noncomputable def scrap_cookie_radius : ℝ := Real.sqrt 6.75

theorem scrap_cookie_radius_proof :
  let large_cookie_radius : ℝ := large_cookie_diameter_ratio * small_cookie_radius / 2
  let large_cookie_area : ℝ := π * large_cookie_radius ^ 2
  let small_cookie_area : ℝ := π * small_cookie_radius ^ 2
  let total_small_cookie_area : ℝ := (num_small_cookies : ℝ) * small_cookie_area
  let scrap_area : ℝ := large_cookie_area - total_small_cookie_area
  scrap_cookie_radius = Real.sqrt (scrap_area / π) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scrap_cookie_radius_proof_l493_49387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_over_four_plus_two_alpha_l493_49314

theorem tan_pi_over_four_plus_two_alpha (α : ℝ) 
  (h1 : π < α ∧ α < 3*π/2) -- α is in the third quadrant
  (h2 : Real.cos (2*α) = -3/5) :
  Real.tan (π/4 + 2*α) = -1/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_over_four_plus_two_alpha_l493_49314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_theorem_l493_49312

def tetrahedron_edge_conditions (k : Nat) (a : ℝ) : Prop :=
  match k with
  | 1 => 0 < a ∧ a < Real.sqrt 3
  | 2 => 0 < a ∧ a < Real.sqrt (2 + Real.sqrt 3)
  | 3 => 0 < a
  | 4 => a > Real.sqrt (2 - Real.sqrt 3)
  | 5 => a > 1 / Real.sqrt 3
  | _ => False

theorem tetrahedron_edge_theorem (k : Nat) (a : ℝ) :
  (∃ (tetrahedron : Type) (edges : Fin 6 → ℝ),
    (∀ i : Fin 6, edges i = a ∨ edges i = 1) ∧
    ((Finset.filter (fun i => edges i = a) (Finset.univ : Finset (Fin 6))).card = k)) ↔
  tetrahedron_edge_conditions k a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_theorem_l493_49312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_proof_num_arcs_is_12_octagon_side_is_3_arc_length_is_pi_l493_49329

/-- The area enclosed by a curve consisting of 12 congruent circular arcs of length π, 
    centered at the vertices of a regular octagon with side length 3 -/
noncomputable def enclosed_area : ℝ := 54 + 54 * Real.sqrt 2 + 1.5 * Real.pi

/-- The number of circular arcs -/
def num_arcs : ℕ := 12

/-- The length of each circular arc -/
noncomputable def arc_length : ℝ := Real.pi

/-- The side length of the regular octagon -/
def octagon_side : ℝ := 3

/-- Proof that the enclosed area is equal to the given expression -/
theorem enclosed_area_proof :
  enclosed_area = 54 + 54 * Real.sqrt 2 + 1.5 * Real.pi :=
by
  -- Unfold the definition of enclosed_area
  unfold enclosed_area
  -- The equality is now trivial
  rfl

/-- Verify that the number of arcs is 12 -/
theorem num_arcs_is_12 : num_arcs = 12 :=
by rfl

/-- Verify that the octagon side length is 3 -/
theorem octagon_side_is_3 : octagon_side = 3 :=
by rfl

/-- Verify that the arc length is π -/
theorem arc_length_is_pi : arc_length = Real.pi :=
by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_proof_num_arcs_is_12_octagon_side_is_3_arc_length_is_pi_l493_49329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_cars_theorem_l493_49340

/-- Represents the distance between two locations in kilometers. -/
def Distance := ℝ

/-- Represents the speed of a car in km/h. -/
def Speed := ℝ

/-- Represents the time taken for a journey in hours. -/
def Time := ℝ

/-- The problem setup for two cars traveling towards each other. -/
structure TwoCarsProblem where
  initial_speed_ratio : ℚ
  speed_change_A : ℚ
  speed_change_B : ℚ
  final_distance_B_to_A : ℝ

/-- The solution to the two cars problem. -/
noncomputable def solve_two_cars_problem (p : TwoCarsProblem) : ℝ :=
  sorry

/-- Theorem stating that the solution to the given problem is 75 km. -/
theorem two_cars_theorem (p : TwoCarsProblem)
  (h1 : p.initial_speed_ratio = 5 / 4)
  (h2 : p.speed_change_A = -1 / 5)
  (h3 : p.speed_change_B = 1 / 5)
  (h4 : p.final_distance_B_to_A = 10) :
  solve_two_cars_problem p = 75 := by
  sorry

#check two_cars_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_cars_theorem_l493_49340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_roots_iff_k_in_range_l493_49364

open Real

/-- The function f(x, k) represents the left side of the equation kx^2 - 2ln(x) - k = 0 --/
noncomputable def f (x k : ℝ) : ℝ := k * x^2 - 2 * Real.log x - k

/-- The theorem states that the equation kx^2 - 2ln(x) - k = 0 has two distinct real roots
    if and only if k is in the open interval (0, 1) or (1, +∞) --/
theorem two_distinct_roots_iff_k_in_range :
  ∀ k : ℝ, (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f x₁ k = 0 ∧ f x₂ k = 0) ↔ 
  (k > 0 ∧ k ≠ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_roots_iff_k_in_range_l493_49364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evenFourDigitNumbersCount_l493_49377

/-- The set of digits used to compose the numbers -/
def digits : Finset Nat := {1, 2, 3, 4}

/-- A four-digit number is even if its last digit is even -/
def isEven (n : Nat) : Bool := n % 2 = 0

/-- A four-digit number composed of the given digits -/
structure FourDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  d1_in_digits : d1 ∈ digits
  d2_in_digits : d2 ∈ digits
  d3_in_digits : d3 ∈ digits
  d4_in_digits : d4 ∈ digits

/-- The set of all valid four-digit numbers composed of the given digits -/
def allFourDigitNumbers : Finset FourDigitNumber := sorry

/-- The set of even four-digit numbers composed of the given digits -/
def evenFourDigitNumbers : Finset FourDigitNumber :=
  allFourDigitNumbers.filter (fun n => isEven n.d4)

/-- The main theorem stating that there are 12 distinct even four-digit numbers -/
theorem evenFourDigitNumbersCount :
  Finset.card evenFourDigitNumbers = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evenFourDigitNumbersCount_l493_49377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_constant_l493_49350

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the circle centered at M(x₀, y₀)
def circle_at_m (x y x₀ y₀ r : ℝ) : Prop := (x - x₀)^2 + (y - y₀)^2 = r^2

-- Define the point M on the ellipse
def point_on_ellipse (x₀ y₀ : ℝ) : Prop := ellipse x₀ y₀

-- Define the tangent lines from O to the circle
def tangent_lines (x₀ y₀ r : ℝ) (k₁ k₂ : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_at_m x₁ y₁ x₀ y₀ r ∧
    circle_at_m x₂ y₂ x₀ y₀ r ∧
    ellipse x₁ y₁ ∧
    ellipse x₂ y₂ ∧
    y₁ = k₁ * x₁ ∧
    y₂ = k₂ * x₂

-- Theorem statement
theorem max_product_constant
  (x₀ y₀ r k₁ k₂ : ℝ)
  (h_ellipse : point_on_ellipse x₀ y₀)
  (h_circle : 0 < r ∧ r < 1)
  (h_tangent : tangent_lines x₀ y₀ r k₁ k₂)
  (h_constant : ∃ (c : ℝ), k₁ * k₂ = c) :
  ∃ (C : ℝ), ∀ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ → ellipse x₂ y₂ → y₁ = k₁ * x₁ → y₂ = k₂ * x₂ →
    (x₁^2 + y₁^2) * (x₂^2 + y₂^2) ≤ C :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_constant_l493_49350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_third_l493_49316

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sqrt 3 * Real.sin x

theorem derivative_f_at_pi_third : 
  deriv f (π / 3) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_third_l493_49316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_six_factors_l493_49343

theorem least_integer_with_six_factors : 
  ∃ n : ℕ+, (n = 12) ∧ 
  (∀ m : ℕ+, m < n → (Finset.card (Nat.divisors m.val) ≠ 6)) ∧
  (Finset.card (Nat.divisors 12) = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_six_factors_l493_49343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_term_is_S8_div_a8_l493_49356

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem largest_term_is_S8_div_a8 (seq : ArithmeticSequence) 
  (h15 : sum_n seq 15 > 0)
  (h16 : sum_n seq 16 < 0) :
  ∀ k ∈ Finset.range 15, (sum_n seq 8) / (seq.a 8) ≥ (sum_n seq (k + 1)) / (seq.a (k + 1)) :=
by
  sorry

#check largest_term_is_S8_div_a8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_term_is_S8_div_a8_l493_49356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_is_correct_l493_49395

def a : ℝ × ℝ × ℝ := (-1, 2, 1)
def b : ℝ × ℝ × ℝ := (-2, -2, 4)

noncomputable def projection_vector (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2.1 * w.2.1 + v.2.2 * w.2.2
  let magnitude_squared := v.1 * v.1 + v.2.1 * v.2.1 + v.2.2 * v.2.2
  let scalar := dot_product / magnitude_squared
  (scalar * v.1, scalar * v.2.1, scalar * v.2.2)

theorem projection_vector_is_correct :
  projection_vector b a = (-1/3, 2/3, 1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_is_correct_l493_49395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_construction_l493_49324

/-- Two lines in a plane -/
structure PlaneLine where
  -- We'll use a placeholder for now
  dummy : Unit

/-- A point in a plane -/
structure PlanePoint where
  -- We'll use a placeholder for now
  dummy : Unit

/-- An isosceles triangle in a plane -/
structure IsoscelesTriangle where
  vertex : PlanePoint
  base1 : PlanePoint
  base2 : PlanePoint
  -- We'll add a property to ensure it's isosceles
  isIsosceles : Unit -- This is a placeholder for the actual isosceles property

/-- Define membership for PlanePoint in PlaneLine -/
instance : Membership PlanePoint PlaneLine where
  mem := λ _ _ => True -- For now, we'll assume all points are on all lines

/-- Define a function to create a line through two points -/
def line_through (p1 p2 : PlanePoint) : PlaneLine :=
  { dummy := () }

/-- The main theorem -/
theorem isosceles_triangle_construction
  (g a : PlaneLine) (B C : PlanePoint) :
  ∃! (T1 T2 : IsoscelesTriangle),
    (T1.vertex ∈ g) ∧
    (T2.vertex ∈ g) ∧
    (T1.base1 ∈ a) ∧
    (T1.base2 ∈ a) ∧
    (T2.base1 ∈ a) ∧
    (T2.base2 ∈ a) ∧
    (B ∈ line_through T1.vertex T1.base1 ∨ B ∈ line_through T1.vertex T1.base2) ∧
    (C ∈ line_through T1.vertex T1.base1 ∨ C ∈ line_through T1.vertex T1.base2) ∧
    (B ∈ line_through T2.vertex T2.base1 ∨ B ∈ line_through T2.vertex T2.base2) ∧
    (C ∈ line_through T2.vertex T2.base1 ∨ C ∈ line_through T2.vertex T2.base2) ∧
    (T1 ≠ T2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_construction_l493_49324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_square_in_right_triangle_l493_49308

theorem largest_inscribed_square_in_right_triangle
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let q := (a * b) / (a + b)
  ∃ (s : Set (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ s → 0 ≤ p.1 ∧ p.1 ≤ a ∧ 0 ≤ p.2 ∧ p.2 ≤ b ∧ p.1 + p.2 ≤ a + b) ∧
    (∀ (p : ℝ × ℝ), p ∈ s → (p.1 + q ≤ a ∧ p.2 + q ≤ b)) ∧
    (∃ (p : ℝ × ℝ), p ∈ s ∧ (p.1 = 0 ∨ p.2 = 0)) ∧
    (∀ (s' : Set (ℝ × ℝ)),
      (∀ (p : ℝ × ℝ), p ∈ s' → 0 ≤ p.1 ∧ p.1 ≤ a ∧ 0 ≤ p.2 ∧ p.2 ≤ b ∧ p.1 + p.2 ≤ a + b) →
      (∃ (q' : ℝ), ∀ (p : ℝ × ℝ), p ∈ s' → (p.1 + q' ≤ a ∧ p.2 + q' ≤ b)) →
      (∃ (p : ℝ × ℝ), p ∈ s' ∧ (p.1 = 0 ∨ p.2 = 0)) →
      q' ≤ q) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_square_in_right_triangle_l493_49308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_degree_with_horizontal_asymptote_largest_possible_degree_l493_49305

/-- A rational function with a specific denominator -/
noncomputable def rationalFunction (p : ℝ → ℝ) : ℝ → ℝ := fun x ↦ p x / (3 * x^7 - x^3 + 5)

/-- Predicate to check if a function has a horizontal asymptote -/
def hasHorizontalAsymptote (f : ℝ → ℝ) : Prop :=
  ∃ L : ℝ, ∀ ε > 0, ∃ M : ℝ, ∀ x > M, |f x - L| < ε

/-- The degree of a polynomial -/
noncomputable def degree (p : ℝ → ℝ) : ℕ := sorry

/-- Theorem stating the maximum degree of p(x) for a rational function with horizontal asymptote -/
theorem max_degree_with_horizontal_asymptote (p : ℝ → ℝ) : 
  hasHorizontalAsymptote (rationalFunction p) → degree p ≤ 7 := by sorry

/-- Theorem stating that 7 is the largest possible degree for p(x) -/
theorem largest_possible_degree : 
  ∃ p : ℝ → ℝ, degree p = 7 ∧ hasHorizontalAsymptote (rationalFunction p) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_degree_with_horizontal_asymptote_largest_possible_degree_l493_49305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l493_49327

/-- Given two plane vectors a and b, prove that if they are parallel and have the given coordinates, then x = -2/3 --/
theorem parallel_vectors_x_value (x : ℝ) : 
  let a : Fin 2 → ℝ := ![x, 1]
  let b : Fin 2 → ℝ := ![2, -3]
  (∃ (k : ℝ), a = k • b) → x = -2/3 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l493_49327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_multiples_l493_49300

theorem count_special_multiples : 
  (Finset.filter (fun n : ℕ => 
    n < 200 ∧ 
    n % 5 = 0 ∧ 
    n % 10 ≠ 0 ∧ 
    n % 6 ≠ 0) 
  (Finset.range 200)).card = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_multiples_l493_49300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l493_49337

theorem divisibility_condition (m n : ℕ) : 
  m > 0 → n > 0 → (m^2019 + n) % (m * n) = 0 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 2^2019) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l493_49337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_value_l493_49355

-- Define f and g as variables instead of functions
variable (f g : ℝ → ℝ)

-- Define the range constraints as axioms
axiom f_range : ∀ y, y ∈ Set.range f → -6 ≤ y ∧ y ≤ 4
axiom g_range : ∀ y, y ∈ Set.range g → -3 ≤ y ∧ y ≤ 2

-- Theorem stating the maximum sum value
theorem max_sum_value :
  ∃ x : ℝ, f x + g x = 6 ∧ ∀ z : ℝ, f z + g z ≤ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_value_l493_49355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_in_region_l493_49311

/-- A point with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- A square with integer coordinates -/
structure IntSquare where
  bottomLeft : IntPoint
  sideLength : Nat

/-- Check if a point is within the specified region -/
def isInRegion (p : IntPoint) : Bool :=
  p.x ≥ 1 ∧ p.x ≤ 7 ∧ p.y ≥ -2 ∧ p.y ≤ 2 * p.x

/-- Check if a square is entirely within the region -/
def isSquareInRegion (s : IntSquare) : Bool :=
  isInRegion s.bottomLeft ∧
  isInRegion { x := s.bottomLeft.x + s.sideLength, y := s.bottomLeft.y } ∧
  isInRegion { x := s.bottomLeft.x, y := s.bottomLeft.y + s.sideLength } ∧
  isInRegion { x := s.bottomLeft.x + s.sideLength, y := s.bottomLeft.y + s.sideLength }

/-- Count valid squares in the region -/
def countValidSquares : Nat :=
  (List.range 7).foldl (fun count x =>
    (List.range 17).foldl (fun innerCount y =>
      innerCount +
      (if isSquareInRegion { bottomLeft := { x := x + 1, y := y - 2 }, sideLength := 1 } then 1 else 0) +
      (if isSquareInRegion { bottomLeft := { x := x + 1, y := y - 2 }, sideLength := 2 } then 1 else 0)
    ) count
  ) 0

theorem count_squares_in_region : countValidSquares = 130 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_in_region_l493_49311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_product_matrix_l493_49322

def is_magic_product (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  let product_of := λ (xs : List ℕ) => xs.foldl (· * ·) 1
  let rows := [0, 1, 2].map (λ i => [m i 0, m i 1, m i 2])
  let cols := [0, 1, 2].map (λ j => [m 0 j, m 1 j, m 2 j])
  let diags := [[m 0 0, m 1 1, m 2 2], [m 0 2, m 1 1, m 2 0]]
  let all_lines := rows ++ cols ++ diags
  ∀ x y, x ∈ all_lines → y ∈ all_lines → product_of x = product_of y

theorem magic_product_matrix :
  ∀ (m : Matrix (Fin 3) (Fin 3) ℕ),
    is_magic_product m →
    m 0 1 = 1 →
    m 1 0 = 4 →
    m 2 0 = 5 →
    m 2 2 = 2 →
    m 0 0 = 18 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_product_matrix_l493_49322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l493_49396

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1 else Real.exp (Real.log 2 * x)

-- Define the set of a that satisfies the equation
def S : Set ℝ := {a | f (f a) = Real.exp (Real.log 2 * f a)}

-- Theorem statement
theorem range_of_a : S = Set.Ici (2/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l493_49396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_approx_l493_49369

/-- The volume of a right rectangular prism with face areas a, b, and c -/
noncomputable def prism_volume (a b c : ℝ) : ℝ :=
  Real.sqrt (a * b * c)

/-- Theorem: The volume of a right rectangular prism with face areas 72, 75, and 80 
    square units is approximately 657 cubic units -/
theorem prism_volume_approx :
  ⌊prism_volume 72 75 80⌋ = 657 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_approx_l493_49369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_density_of_intersection_l493_49347

open Real

-- Define the probability density function
noncomputable def g (y : ℝ) : ℝ := 4 / (Real.pi * (16 + y^2))

-- Define the uniform distribution of angle t
noncomputable def f (t : ℝ) : ℝ := if -Real.pi/2 < t ∧ t < Real.pi/2 then 1/Real.pi else 0

-- Theorem statement
theorem probability_density_of_intersection (y : ℝ) :
  let t := arctan (y/4)
  (∀ t', -Real.pi/2 < t' ∧ t' < Real.pi/2 → f t' = 1/Real.pi) →
  y = 4 * tan t →
  g y = f t * |((4 : ℝ)/(16 + y^2))| := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_density_of_intersection_l493_49347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l493_49397

-- Define the necessary types and structures
structure Plane where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Plane
  B : Plane
  C : Plane

-- Define the necessary functions and predicates
def RightTriangle (t : Triangle) : Prop := sorry

def SegmentLength (A B : Plane) : ℝ := sorry

def Rectangle (A B C D : Plane) : Prop := sorry

def RectangleWidth (A B C D : Plane) : ℝ := sorry

def CyclicQuadrilateral (A B C D : Plane) : Prop := sorry

def Perimeter (t : Triangle) : ℝ := sorry

/-- Given a right triangle DEF with DE = 10 and EF = 6, and rectangles DEUV (10 × 5) and 
    EFWX (6 × 2) constructed outside the triangle such that U, V, W, and X lie on a circle, 
    the perimeter of triangle DEF is 10 + 6 + √(100 - 36). -/
theorem triangle_perimeter (D E F U V W X : Plane) (t : Triangle) : 
  RightTriangle t → 
  t.A = D → t.B = E → t.C = F →
  SegmentLength D E = 10 → 
  SegmentLength E F = 6 → 
  Rectangle D E U V → 
  RectangleWidth D E U V = 5 → 
  Rectangle E F W X → 
  RectangleWidth E F W X = 2 → 
  CyclicQuadrilateral U V W X → 
  Perimeter t = 10 + 6 + Real.sqrt (100 - 36) := 
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l493_49397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l493_49335

/-- The radius of the inscribed sphere in a regular hexagonal pyramid -/
noncomputable def inscribed_sphere_radius (a b : ℝ) : ℝ :=
  (a * Real.sqrt (3 * (4 * b ^ 2 - 3 * a ^ 2))) / (2 * a * Real.sqrt 3 + 2 * Real.sqrt (4 * b ^ 2 - a ^ 2))

/-- Theorem: The radius of the inscribed sphere in a regular hexagonal pyramid -/
theorem inscribed_sphere_radius_formula {a b : ℝ} (ha : a > 0) (hb : b > a * Real.sqrt 3 / 2) :
  ∃ r : ℝ, r = inscribed_sphere_radius a b ∧ 
    r = (a * Real.sqrt (3 * (4 * b ^ 2 - 3 * a ^ 2))) / (2 * a * Real.sqrt 3 + 2 * Real.sqrt (4 * b ^ 2 - a ^ 2)) :=
by
  use inscribed_sphere_radius a b
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l493_49335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l493_49352

/-- The area of the triangle formed by the intersection of two lines and the y-axis -/
theorem triangle_area (line1 line2 : ℝ → ℝ) (h1 : line1 = fun x ↦ 2 * x + 1) 
  (h2 : line2 = fun x ↦ -3 * x + 16) : 
  (1/2) * (16 - 1) * ((16 - 1) / 5) = 22.5 := by
  -- Calculate the x-coordinate of the intersection point
  let x_intersect := (16 - 1) / 5
  -- Calculate the y-coordinate of the intersection point
  let y_intersect := line1 x_intersect
  -- Calculate the base of the triangle
  let base := line2 0 - line1 0
  -- The height is the x-coordinate of the intersection point
  let height := x_intersect
  -- The area formula
  have area_formula : (1/2) * base * height = (1/2) * (16 - 1) * ((16 - 1) / 5) := by sorry
  -- Prove that this equals 22.5
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l493_49352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_and_x_axis_l493_49301

-- Define the function
def f (x : ℝ) : ℝ := -x^3 + x^2 + 2*x

-- State the theorem
theorem area_enclosed_by_curve_and_x_axis :
  ∫ x in (-1)..(0), -f x + ∫ x in (0)..(2), f x = 37/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_and_x_axis_l493_49301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_negative_thirteen_thirds_l493_49392

/-- An arithmetic sequence with specific first four terms -/
def ArithmeticSequence (x y : ℚ) : ℕ → ℚ
  | 0 => x + 2*y
  | 1 => x - 2*y
  | 2 => 2*x*y
  | 3 => x/y
  | n+4 => ArithmeticSequence x y 3 + (n + 1 : ℚ) * (ArithmeticSequence x y 1 - ArithmeticSequence x y 0)

/-- The theorem stating that the fifth term of the specific arithmetic sequence is -13/3 -/
theorem fifth_term_is_negative_thirteen_thirds (x y : ℚ) 
  (h1 : y = 1) 
  (h2 : x = -1/3) 
  (h3 : ArithmeticSequence x y 1 - ArithmeticSequence x y 0 = ArithmeticSequence x y 2 - ArithmeticSequence x y 1) 
  (h4 : ArithmeticSequence x y 2 - ArithmeticSequence x y 1 = ArithmeticSequence x y 3 - ArithmeticSequence x y 2) : 
  ArithmeticSequence x y 4 = -13/3 := by
  sorry

#eval ArithmeticSequence (-1/3) 1 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_negative_thirteen_thirds_l493_49392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_pays_more_than_jane_l493_49358

-- Define the constants
def original_price : ℝ := 84.00000000000009
def discount_rate : ℝ := 0.10
def tip_rate : ℝ := 0.15

-- Define the functions for calculations
def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def tip_amount (price : ℝ) (tip_rate : ℝ) : ℝ :=
  price * tip_rate

def total_payment (base_price : ℝ) (tip : ℝ) : ℝ :=
  base_price + tip

-- Theorem statement
theorem john_pays_more_than_jane : 
  let johns_tip := tip_amount original_price tip_rate
  let janes_tip := tip_amount (discounted_price original_price discount_rate) tip_rate
  let john_total := total_payment original_price johns_tip
  let jane_total := total_payment (discounted_price original_price discount_rate) janes_tip
  abs (john_total - jane_total - 9.66) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_pays_more_than_jane_l493_49358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l493_49378

/-- Calculates simple interest given principal, rate, and time -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates compound interest given principal, rate, and time -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

theorem interest_calculation (P : ℝ) (h1 : P > 0) :
  simple_interest P 20 2 = 400 →
  compound_interest P 20 2 = 440 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l493_49378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_painting_progress_mark_paints_fifth_in_twelve_minutes_l493_49331

/-- Given that Mark can paint a wall in 60 minutes, this theorem proves
    that he can paint 1/5 of the wall in 12 minutes. -/
theorem marks_painting_progress (total_time : ℚ) (part_time : ℚ) 
    (h1 : total_time = 60) (h2 : part_time = 12) :
    (part_time / total_time) = (1 : ℚ) / 5 := by
  sorry

/-- This function calculates the fraction of the wall painted given the time spent painting. -/
noncomputable def fraction_painted (time_spent : ℚ) (total_time : ℚ) : ℚ :=
  time_spent / total_time

/-- This theorem proves that Mark can paint 1/5 of the wall in 12 minutes. -/
theorem mark_paints_fifth_in_twelve_minutes :
    fraction_painted 12 60 = (1 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_painting_progress_mark_paints_fifth_in_twelve_minutes_l493_49331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_of_special_sequence_l493_49385

/-- A sequence defined by its first four terms and a recurrence relation. -/
noncomputable def special_sequence (x y : ℝ) : ℕ → ℝ
  | 0 => x + y
  | 1 => x - y
  | 2 => x * y
  | 3 => x / y
  | n+4 => special_sequence x y (n+3) + special_sequence x y (n+2) - special_sequence x y (n+1)

/-- The fifth term of the special sequence is (x/y) + xy - x + y. -/
theorem fifth_term_of_special_sequence (x y : ℝ) (hy : y ≠ 0) :
  special_sequence x y 4 = (x / y) + x * y - x + y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_of_special_sequence_l493_49385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_equal_iff_a_eq_two_A_proper_subset_B_iff_a_range_l493_49336

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < a * x + 1 ∧ a * x + 1 ≤ 5}
def B : Set ℝ := {x | -1/2 < x ∧ x ≤ 2}

-- Theorem 1: A = B if and only if a = 2
theorem sets_equal_iff_a_eq_two (a : ℝ) : A a = B ↔ a = 2 := by sorry

-- Theorem 2: A is a proper subset of B if and only if a > 2 or a < -8
theorem A_proper_subset_B_iff_a_range (a : ℝ) : A a ⊂ B ↔ a > 2 ∨ a < -8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_equal_iff_a_eq_two_A_proper_subset_B_iff_a_range_l493_49336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_reciprocals_l493_49398

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1) + x + Real.sin x

-- State the theorem
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : f (4 * a) + f (b - 9) = 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → f (4 * x) + f (y - 9) = 0 → 1/a + 1/b ≤ 1/x + 1/y) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ f (4 * x) + f (y - 9) = 0 ∧ 1/x + 1/y = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_reciprocals_l493_49398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_length_l493_49382

noncomputable section

open Real

theorem circle_chord_length (r : ℝ) (O A B C : EuclideanSpace ℝ (Fin 2)) 
  (h1 : dist O A = r) (h2 : dist O B = r) (h3 : dist O C = r)
  (h4 : dist A B = dist A C) (h5 : dist A B > r) 
  (h6 : (2 * r * sin ((dist B C) / (2 * r))) = π * r / 3) : 
  dist A B = r * sqrt (2 + sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_length_l493_49382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meters_examined_wed_thu_l493_49394

theorem meters_examined_wed_thu (rejection_rate : ℚ) (tuesday_meters : ℕ) (total_rejected : ℕ) :
  rejection_rate = 25 / 10000 →
  tuesday_meters = 800 →
  total_rejected = 10 →
  ∃ (wed_thu_meters : ℕ),
    (⌊(rejection_rate * tuesday_meters)⌋ + ⌊(rejection_rate * wed_thu_meters)⌋ = total_rejected) ∧
    wed_thu_meters = 3200 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meters_examined_wed_thu_l493_49394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_c_l493_49332

/-- Given a triangle ABC where angle B = π/4, b = √2 * a, and a < b, prove that angle C = 7π/12 -/
theorem triangle_angle_c (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  A + B + C = π →  -- angle sum theorem
  a < b →  -- given condition
  B = π/4 →  -- given condition
  b = Real.sqrt 2 * a →  -- given condition
  C = 7*π/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_c_l493_49332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_to_equation_l493_49368

theorem four_solutions_to_equation :
  ∃ (S : Finset ℝ), (∀ x ∈ S, (x^2 - 10)^2 = 81) ∧ (S.card = 4) ∧
  (∀ y : ℝ, (y^2 - 10)^2 = 81 → y ∈ S) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_to_equation_l493_49368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_calculation_correct_l493_49348

/-- Represents the tax calculation function based on taxable income -/
noncomputable def tax_function (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 500 then
    0.05 * x
  else if 500 < x ∧ x ≤ 2000 then
    0.1 * (x - 500) + 0.05 * 500
  else if 2000 < x ∧ x ≤ 5000 then
    0.15 * (x - 2000) + 0.1 * 1500 + 0.05 * 500
  else
    0  -- For completeness, though not specified in the original problem

/-- The tax-free threshold -/
def tax_free_threshold : ℝ := 800

/-- Calculates the taxable income -/
noncomputable def taxable_income (total_income : ℝ) : ℝ :=
  max (total_income - tax_free_threshold) 0

/-- Theorem stating that for a monthly income of 3000 yuan, the tax is 205 yuan -/
theorem tax_calculation_correct :
  tax_function (taxable_income 3000) = 205 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_calculation_correct_l493_49348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l493_49315

theorem min_value_theorem (b : ℝ) (m n : ℝ) : 
  b > 0 → 
  m > 0 → 
  n > 0 → 
  (∃ (x : ℝ), Real.sin x + Real.sqrt 3 * Real.cos x = b) →
  (6 * m + b * n = 2) →
  (∀ (k : ℝ), k > 0 → 1 / m + 4 / n ≥ k) →
  (∃ (k : ℝ), k > 0 ∧ 1 / m + 4 / n = k) →
  (1 / m + 4 / n = 7 + 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l493_49315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_properties_l493_49345

/-- Represents a trapezoid WXYZ with specific properties -/
structure Trapezoid (a : ℝ) where
  /-- WX is parallel to ZY -/
  wx_parallel_zy : True
  /-- WY is perpendicular to ZY -/
  wy_perp_zy : True
  /-- Length of YZ -/
  yz_length : a > 0
  /-- Tangent of angle Z -/
  tan_z : Real.sqrt 3 > 0
  /-- Tangent of angle X -/
  tan_x : (1 : ℝ) / 2 > 0

/-- Main theorem about the trapezoid -/
theorem trapezoid_properties (a : ℝ) (t : Trapezoid a) :
  ∃ (xy wy wx : ℝ),
    xy = a * Real.sqrt 15 ∧
    wy + wx + xy = 3 * a * Real.sqrt 3 + a * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_properties_l493_49345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l493_49325

noncomputable def floor (x : ℝ) := ⌊x⌋

noncomputable def f (x : ℝ) : ℝ :=
  (x + 1/x) / (floor x * floor (1/x) + floor x + floor (1/x) + 1)

theorem range_of_f :
  ∀ x > 0, f x ∈ ({1/2} : Set ℝ) ∪ Set.Icc (5/6) (5/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l493_49325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_difference_l493_49342

/-- Calculates the simple interest given principal, rate, and time -/
noncomputable def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Theorem: The difference between the principal and simple interest -/
theorem simple_interest_difference (principal rate time : ℚ) 
  (h_principal : principal = 2400)
  (h_rate : rate = 4)
  (h_time : time = 5) :
  principal - simple_interest principal rate time = 1920 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_difference_l493_49342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l493_49302

theorem empty_subset_singleton_zero : ∅ ⊆ ({0} : Set ℕ) := by
  apply Set.empty_subset


end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l493_49302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l493_49390

/-- An ellipse with center at the origin, foci on the x-axis, and focal distance 4 -/
structure Ellipse where
  focal_distance : ℝ
  focal_distance_eq : focal_distance = 4

/-- A point on the ellipse -/
structure PointOnEllipse (C : Ellipse) where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / 8 + y^2 / 4 = 1

/-- The foci of the ellipse -/
noncomputable def foci (C : Ellipse) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((C.focal_distance / 2, 0), (-C.focal_distance / 2, 0))

/-- The internal angle of the triangle formed by a point on the ellipse and the foci -/
noncomputable def internal_angle (C : Ellipse) (P : PointOnEllipse C) : ℝ :=
  sorry

/-- Theorem about the ellipse properties -/
theorem ellipse_properties (C : Ellipse) :
  (∀ P : PointOnEllipse C, internal_angle C P ≤ π / 2) →
  (∃ k m : ℝ, ∃ A B : PointOnEllipse C,
    A.y = k * A.x + m ∧
    B.y = k * B.x + m ∧
    (A.x - B.x)^2 + (A.y - B.y)^2 = (A.x + B.x)^2 + (A.y + B.y)^2 ∧
    (m > Real.sqrt 2 ∨ m < -Real.sqrt 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l493_49390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l493_49338

/-- Circle C in rectangular coordinates -/
def circleC (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

/-- Line l in general form -/
def lineL (x y : ℝ) : Prop := 2 * Real.sqrt 2 * x - y - 1 = 0

/-- Point on the circle -/
def pointOnCircle (P : ℝ × ℝ) : Prop := circleC P.1 P.2

/-- Intersection points of the circle and line -/
def intersectionPoints (M N : ℝ × ℝ) : Prop :=
  pointOnCircle M ∧ pointOnCircle N ∧ lineL M.1 M.2 ∧ lineL N.1 N.2 ∧ M ≠ N

/-- Theorem stating the maximum area of triangle PMN -/
theorem max_area_triangle (M N P : ℝ × ℝ) :
  intersectionPoints M N → pointOnCircle P → P ≠ M → P ≠ N →
  ∃ (S : ℝ), S = (10 * Real.sqrt 5) / 9 ∧
  ∀ (A : ℝ), A = abs ((P.1 - M.1) * (N.2 - M.2) - (P.2 - M.2) * (N.1 - M.1)) / 2 →
  A ≤ S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l493_49338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_acute_angles_l493_49320

-- Define the triangle PQR
variable (P Q R : EuclideanSpace ℝ (Fin 2))

-- Define the incircle
variable (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ)

-- Define the points of tangency
variable (X Y Z : EuclideanSpace ℝ (Fin 2))

-- Define the property of being an inscribed circle
def is_inscribed_circle (P Q R center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ) : Prop := sorry

-- Define the property of a point being on a line segment
def on_segment (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define an acute angle
def is_acute_angle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

theorem inscribed_circle_acute_angles 
  (h_inscribed : is_inscribed_circle P Q R center radius)
  (h_X : on_segment P Q X)
  (h_Y : on_segment Q R Y)
  (h_Z : on_segment R P Z) :
  is_acute_angle X Y Z ∧ is_acute_angle Y Z X ∧ is_acute_angle Z X Y := by
  sorry

#check inscribed_circle_acute_angles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_acute_angles_l493_49320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_direction_l493_49383

-- Define a non-zero vector a
variable (a : ℝ → ℝ)
variable (h_a : a ≠ 0)

-- Define a non-zero real number λ
variable (l : ℝ)
variable (h_l : l ≠ 0)

-- Theorem stating that a and λ²a have the same direction
theorem same_direction : ∃ (k : ℝ), k > 0 ∧ (fun x => l^2 * (a x)) = (fun x => k * (a x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_direction_l493_49383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_value_exists_l493_49321

theorem exact_value_exists : ∃ (x : ℝ), 
  x^2 = (2 - Real.sin (π/9)^2) * (2 - Real.sin (2*π/9)^2) * (2 - Real.sin (4*π/9)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_value_exists_l493_49321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l493_49391

/-- Given a natural number n, this function represents the binomial coefficient C(n,k) -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- This function checks if three numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℕ) : Prop := 2 * b = a + c

/-- This function represents the general term of the expansion (√x + 2/√x)^n -/
noncomputable def generalTerm (n r : ℕ) (x : ℝ) : ℝ := 
  (2^r : ℝ) * (binomial n r : ℝ) * x^((n - 2*r : ℤ) / 2)

theorem expansion_properties (n : ℕ) : 
  (isArithmeticSequence (binomial n 1) (binomial n 2) (binomial n 3)) → 
  (n = 7 ∧ ¬ ∃ (r : ℕ), (n : ℤ) - 2*r = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l493_49391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_power_function_l493_49379

/-- The equation of the tangent line to y = x^n at the point (2, 8) is 12x - y - 16 = 0 -/
theorem tangent_line_power_function :
  ∃ (n : ℝ), (2 : ℝ)^n = 8 →
  let f : ℝ → ℝ := λ x ↦ x^n
  let A : ℝ × ℝ := (2, 8)
  let slope : ℝ := (deriv f) A.fst
  ∀ x y : ℝ, (y - A.snd = slope * (x - A.fst)) ↔ (12*x - y - 16 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_power_function_l493_49379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l493_49339

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)
  (a b c : Real)
  (h_angles : A + B + C = Real.pi)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)

-- Define the theorem
theorem triangle_inequalities (t : Triangle) : 
  (Real.sin t.B * Real.sin t.C) / Real.sin t.A ≤ (1/2) * Real.tan (Real.pi/2 - t.A/2) ∧
  (1/2) * t.b * t.c * Real.sin t.A ≤ (t.a^2 / 4) * Real.tan (Real.pi/2 - t.A/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l493_49339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_black_probability_l493_49310

/-- Represents a 2x2 grid where each cell can be either black or white -/
def Grid := Fin 2 → Fin 2 → Bool

/-- The probability of a cell being initially black -/
noncomputable def initial_black_prob : ℝ := 1 / 2

/-- Rotates the grid 180 degrees -/
def rotate (g : Grid) : Grid :=
  fun i j => g (1 - i) (1 - j)

/-- Applies the repainting rule after rotation -/
def repaint (g : Grid) : Grid :=
  fun i j => g i j || (rotate g) i j

/-- The probability of the grid being entirely black after rotation and repainting -/
noncomputable def prob_all_black (g : Grid) : ℝ :=
  (initial_black_prob ^ 2) * (initial_black_prob ^ 2)

theorem grid_black_probability :
  ∀ g : Grid, prob_all_black g = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_black_probability_l493_49310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_prism_W_l493_49319

-- Define a unit cube
def unit_cube : Set (Fin 3 → ℝ) := {v | ∀ i, 0 ≤ v i ∧ v i ≤ 1}

-- Define the volume of a set in ℝ³
noncomputable def volume (S : Set (Fin 3 → ℝ)) : ℝ := sorry

-- Define the cuts on the cube
def initial_cuts (c : Set (Fin 3 → ℝ)) : List (Set (Fin 3 → ℝ)) := sorry

-- Define the second cut on one of the prisms
def second_cut (p : Set (Fin 3 → ℝ)) : List (Set (Fin 3 → ℝ)) := sorry

-- Define the vertex W
def vertex_W : Fin 3 → ℝ := λ i ↦ 0

-- Define the prism containing vertex W after all cuts
def prism_W (c : Set (Fin 3 → ℝ)) : Set (Fin 3 → ℝ) := sorry

-- Theorem statement
theorem volume_prism_W :
  volume (prism_W unit_cube) = 1/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_prism_W_l493_49319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_PQ_l493_49309

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the altitude AH
noncomputable def AH (A B C : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define the variable point P
variable (P : ℝ × ℝ)

-- Define angle bisectors k and l
noncomputable def k (P B C : ℝ × ℝ) : Set (ℝ × ℝ) := sorry
noncomputable def l (P B C : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define points E, F, and Q
noncomputable def E (A B C P : ℝ × ℝ) : ℝ × ℝ := sorry
noncomputable def F (A B C P : ℝ × ℝ) : ℝ × ℝ := sorry
noncomputable def Q (A B C P : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the fixed point T
noncomputable def T (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define necessary geometric concepts
def IsAcuteTriangle : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop := sorry
def IsAltitude : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop := sorry
def Line : (ℝ × ℝ) → (ℝ × ℝ) → Set (ℝ × ℝ) := sorry

-- State the theorem
theorem fixed_point_on_PQ (A B C P : ℝ × ℝ) (H : ℝ × ℝ)
  (hAcute : IsAcuteTriangle A B C) 
  (hAltitude : IsAltitude A H C) 
  (hBisectorsIntersect : (k P B C) ∩ (l P B C) ⊆ AH A B C) 
  (hE : E A B C P ∈ (k P B C) ∩ (Line A C)) 
  (hF : F A B C P ∈ (l P B C) ∩ (Line A B)) 
  (hQ : Q A B C P ∈ (Line (E A B C P) (F A B C P)) ∩ (AH A B C)) : 
  T A B C ∈ Line P (Q A B C P) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_PQ_l493_49309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delivery_charges_theorem_l493_49371

/-- Charging standards and actual charges for express delivery --/
structure DeliveryCharges where
  flatRateGZ : ℚ
  excessChargeGZ : ℚ
  flatRateSH : ℚ
  excessChargeSH : ℚ
  costGZ : ℚ
  weightGZ : ℚ
  costSH : ℚ
  weightSH : ℚ

/-- Calculate the cost for a given weight and charging standard --/
def calculateCost (flatRate : ℚ) (excessCharge : ℚ) (weight : ℚ) : ℚ :=
  flatRate + max 0 (weight - 1) * excessCharge

/-- The main theorem that proves the flat rate, excess charge, and cost for 5 kg to Guangzhou --/
theorem delivery_charges_theorem (charges : DeliveryCharges) :
  charges.flatRateSH = charges.flatRateGZ + 2 ∧
  charges.excessChargeSH = charges.excessChargeGZ + 3 ∧
  calculateCost charges.flatRateGZ charges.excessChargeGZ charges.weightGZ = charges.costGZ ∧
  calculateCost charges.flatRateSH charges.excessChargeSH charges.weightSH = charges.costSH ∧
  charges.weightGZ = 3 ∧
  charges.weightSH = 4 ∧
  charges.costGZ = 10 ∧
  charges.costSH = 23 →
  charges.flatRateGZ = 6 ∧
  charges.excessChargeGZ = 2 ∧
  calculateCost charges.flatRateGZ charges.excessChargeGZ 5 = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_delivery_charges_theorem_l493_49371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_minimum_points_l493_49307

-- Define a line
def Line : Type := ℝ → ℝ → Prop

-- Define a point on a line
def Point (l : Line) : Type := { p : ℝ × ℝ // l p.1 p.2 }

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the sum of distances from a point to four other points
noncomputable def sumOfDistances (p A B C D : ℝ × ℝ) : ℝ :=
  distance p A + distance p B + distance p C + distance p D

-- Function to convert Point to ℝ × ℝ
def pointToReal (l : Line) (p : Point l) : ℝ × ℝ := p.val

-- State the theorem
theorem infinitely_many_minimum_points (l : Line) (A B C D : Point l) 
  (h_distinct : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ C ∧ A ≠ D ∧ B ≠ D) :
  ∃ (S : Set (ℝ × ℝ)), Infinite S ∧ 
    ∀ (p : ℝ × ℝ), p ∈ S → 
      ∀ (q : ℝ × ℝ), sumOfDistances p (pointToReal l A) (pointToReal l B) (pointToReal l C) (pointToReal l D) 
                    ≤ sumOfDistances q (pointToReal l A) (pointToReal l B) (pointToReal l C) (pointToReal l D) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_minimum_points_l493_49307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_verify_seven_gram_coins_determine_coin_weight_l493_49346

-- Define the set of possible coin weights
def CoinWeights : Finset ℕ := {7, 8, 9, 10, 11, 12, 13}

-- Define a type for bags of coins
structure CoinBag where
  weight : ℕ
  count : ℕ
deriving DecidableEq

-- Define the set of bags
def Bags : Finset CoinBag := 
  Finset.image (λ w => ⟨w, 100⟩) CoinWeights

-- Define a balance scale function
def balance (left right : Finset ℕ) : Ordering :=
  compare (Finset.sum left id) (Finset.sum right id)

-- Theorem 1: Verifying 7-gram coins in one weighing
theorem verify_seven_gram_coins (bags : Finset CoinBag) 
  (h1 : bags = Bags) 
  (h2 : ∀ b ∈ bags, b.weight ∈ CoinWeights ∧ b.count = 100) :
  ∃ (left right : Finset ℕ),
    balance left right = Ordering.lt ↔ 
    ∃ b ∈ bags, b.weight = 7 := by
  sorry

-- Theorem 2: Determining coin weight in at most two weighings
theorem determine_coin_weight (bags : Finset CoinBag) 
  (h1 : bags = Bags)
  (h2 : ∀ b ∈ bags, b.weight ∈ CoinWeights ∧ b.count = 100) :
  ∀ b ∈ bags, ∃ (weighings : Fin 2 → Finset ℕ × Finset ℕ),
    ∃ (results : Fin 2 → Ordering),
      (∀ i, balance (weighings i).1 (weighings i).2 = results i) →
      b.weight = (CoinWeights.filter (λ w => 
        ∀ i, balance (weighings i).1 (weighings i).2 = 
          balance ((weighings i).1.filter (λ x => x ≠ w)) 
                  ((weighings i).2.filter (λ x => x ≠ w))
      )).min' (by sorry) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_verify_seven_gram_coins_determine_coin_weight_l493_49346
