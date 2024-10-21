import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_combination_l234_23421

noncomputable def average (x y z : ℝ) : ℝ := (x + y + z) / 3

theorem average_combination (x₁ x₂ x₃ y₁ y₂ y₃ a b : ℝ) 
  (hx : average x₁ x₂ x₃ = a) 
  (hy : average y₁ y₂ y₃ = b) : 
  average (2*x₁ + 3*y₁) (2*x₂ + 3*y₂) (2*x₃ + 3*y₃) = 2*a + 3*b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_combination_l234_23421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l234_23480

theorem rationalize_denominator :
  ∃ (a b : ℝ), b ≠ 0 ∧ (7 / (2 * Real.sqrt 50) = a / b) ∧ ∃ (q : ℚ), b * Real.sqrt 2 = q ∧ a = (7 * Real.sqrt 2) / 20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l234_23480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candles_before_birthday_l234_23474

/-- Represents the number of candles on Molly's birthday cake -/
def candles : ℕ → ℕ := sorry

/-- Molly's current age -/
def current_age : ℕ := 20

/-- Number of additional candles Molly received -/
def additional_candles : ℕ := 6

/-- Theorem stating the number of candles Molly had before -/
theorem candles_before_birthday (n : ℕ) : 
  candles n = current_age - additional_candles ↔ n = current_age - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candles_before_birthday_l234_23474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_sum_equals_two_l234_23454

-- Define lg as the base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_two : lg 4 + 2 * lg 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_sum_equals_two_l234_23454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_sum_abs_l234_23488

theorem circle_max_sum_abs (x y : ℝ) : x^2 + y^2 = 4 → |x| + |y| ≤ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_sum_abs_l234_23488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_number_a_equals_nine_l234_23453

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a telephone number in the format ABC-DEF-GHIJ -/
structure PhoneNumber where
  A : Digit
  B : Digit
  C : Digit
  D : Digit
  E : Digit
  F : Digit
  G : Digit
  H : Digit
  I : Digit
  J : Digit

/-- Check if three digits are consecutive odd numbers -/
def consecutiveOdd (d e f : Digit) : Prop :=
  d.val % 2 = 1 ∧ e.val % 2 = 1 ∧ f.val % 2 = 1 ∧
  e.val = d.val + 2 ∧ f.val = e.val + 2

/-- Check if four digits are consecutive even numbers -/
def consecutiveEven (g h i j : Digit) : Prop :=
  g.val % 2 = 0 ∧ h.val % 2 = 0 ∧ i.val % 2 = 0 ∧ j.val % 2 = 0 ∧
  h.val = g.val + 2 ∧ i.val = h.val + 2 ∧ j.val = i.val + 2

theorem phone_number_a_equals_nine (pn : PhoneNumber)
  (h1 : pn.A.val > pn.B.val ∧ pn.B.val > pn.C.val)
  (h2 : pn.D.val > pn.E.val ∧ pn.E.val > pn.F.val)
  (h3 : pn.G.val > pn.H.val ∧ pn.H.val > pn.I.val ∧ pn.I.val > pn.J.val)
  (h4 : consecutiveOdd pn.D pn.E pn.F)
  (h5 : consecutiveEven pn.G pn.H pn.I pn.J)
  (h6 : pn.A.val + pn.B.val + pn.C.val = 15)
  (h7 : pn.A ≠ pn.B ∧ pn.A ≠ pn.C ∧ pn.A ≠ pn.D ∧ pn.A ≠ pn.E ∧ pn.A ≠ pn.F ∧ pn.A ≠ pn.G ∧ pn.A ≠ pn.H ∧ pn.A ≠ pn.I ∧ pn.A ≠ pn.J ∧
      pn.B ≠ pn.C ∧ pn.B ≠ pn.D ∧ pn.B ≠ pn.E ∧ pn.B ≠ pn.F ∧ pn.B ≠ pn.G ∧ pn.B ≠ pn.H ∧ pn.B ≠ pn.I ∧ pn.B ≠ pn.J ∧
      pn.C ≠ pn.D ∧ pn.C ≠ pn.E ∧ pn.C ≠ pn.F ∧ pn.C ≠ pn.G ∧ pn.C ≠ pn.H ∧ pn.C ≠ pn.I ∧ pn.C ≠ pn.J ∧
      pn.D ≠ pn.E ∧ pn.D ≠ pn.F ∧ pn.D ≠ pn.G ∧ pn.D ≠ pn.H ∧ pn.D ≠ pn.I ∧ pn.D ≠ pn.J ∧
      pn.E ≠ pn.F ∧ pn.E ≠ pn.G ∧ pn.E ≠ pn.H ∧ pn.E ≠ pn.I ∧ pn.E ≠ pn.J ∧
      pn.F ≠ pn.G ∧ pn.F ≠ pn.H ∧ pn.F ≠ pn.I ∧ pn.F ≠ pn.J ∧
      pn.G ≠ pn.H ∧ pn.G ≠ pn.I ∧ pn.G ≠ pn.J ∧
      pn.H ≠ pn.I ∧ pn.H ≠ pn.J ∧
      pn.I ≠ pn.J) :
  pn.A = ⟨9, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_number_a_equals_nine_l234_23453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_distance_sqrt_74_l234_23486

/-- The vertex of a quadratic function f(x) = ax² + bx + c is at (-b/(2a), f(-b/(2a))) -/
noncomputable def quadratic_vertex (a b c : ℝ) : ℝ × ℝ :=
  let x := -b / (2 * a)
  (x, a * x^2 + b * x + c)

/-- The distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem vertex_distance_sqrt_74 : 
  let f1 := quadratic_vertex 1 (-4) 8
  let f2 := quadratic_vertex 1 6 20
  distance f1 f2 = Real.sqrt 74 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_distance_sqrt_74_l234_23486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_octagon_area_ratio_l234_23401

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
noncomputable def midpointOctagon (o : RegularOctagon) : RegularOctagon :=
  { vertices := λ i => 
      let a := o.vertices i
      let b := o.vertices ((i + 1) % 8)
      ((a.1 + b.1) / 2, (a.2 + b.2) / 2) }

/-- The area of a regular octagon -/
noncomputable def area (o : RegularOctagon) : ℝ := sorry

/-- The ratio of the area of the midpoint octagon to the original octagon is 3/4 -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpointOctagon o) = (3 / 4) * area o := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_octagon_area_ratio_l234_23401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_tile_checkerboard_l234_23456

def color (ℓ c : ℕ) : ℕ := (c + ℓ) % 4

def checkerboard := Fin 10 × Fin 10

def rectangle := Fin 4 × Fin 1

theorem cannot_tile_checkerboard : 
  ¬ ∃ (tiling : Set (checkerboard → rectangle)), 
    (∀ (x : checkerboard), ∃! (r : rectangle), ∃ (t : checkerboard → rectangle), t ∈ tiling ∧ t x = r) ∧
    (∀ (t₁ t₂ : checkerboard → rectangle), t₁ ∈ tiling → t₂ ∈ tiling →
      ∀ (x y : checkerboard), t₁ x = t₂ y → (t₁ = t₂ ∧ x = y) ∨ (t₁ ≠ t₂ ∧ x ≠ y)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_tile_checkerboard_l234_23456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_conversion_l234_23426

/-- Converts speed from kilometers per hour to meters per second -/
noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

theorem train_speed_conversion :
  let train_speed_kmph : ℝ := 189
  kmph_to_mps train_speed_kmph = 52.5 := by
  -- Unfold the definition of kmph_to_mps
  unfold kmph_to_mps
  -- Simplify the expression
  simp
  -- The proof is completed numerically
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_conversion_l234_23426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_counterexample_l234_23491

/-- The inverse proportion function f(x) = -6/x -/
noncomputable def f (x : ℝ) : ℝ := -6 / x

theorem inverse_proportion_counterexample :
  ∃ x : ℝ, x ≥ -1 ∧ f x < 6 :=
by
  -- We can use x = 0 as a counterexample
  use 0
  constructor
  · -- Prove 0 ≥ -1
    linarith
  · -- Prove f 0 < 6
    -- Note: f 0 is undefined, so we need to approach it from the right
    sorry -- This part requires more advanced techniques to prove rigorously


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_counterexample_l234_23491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_game_probability_l234_23459

noncomputable def num_players : ℕ := 10
noncomputable def die_faces : ℕ := 6

-- Probability that one player wins a prize
noncomputable def prob_one_wins : ℝ := (5/6)^9

-- Probability that at least one player wins a prize
noncomputable def prob_at_least_one_wins : ℝ := 
  10 * (5^9/6^9) - 45 * (5 * 4^8/6^9) + 120 * (5 * 4 * 3^7/6^9) - 
  210 * (5 * 4 * 3 * 2^6/6^9) + 252 * (5 * 4 * 3 * 2 * 1/6^9)

theorem dice_game_probability : 
  ∃ ε > 0, abs (prob_at_least_one_wins - 0.919) < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_game_probability_l234_23459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_samantha_laundry_cost_l234_23435

/-- Represents the laundromat scenario --/
structure LaundryScenario where
  washer_cost : ℚ
  dryer_cost_per_10_min : ℚ
  loads_washed : ℕ
  dryers_used : ℕ
  dryer_time : ℕ

/-- Calculates the total cost for a given laundry scenario --/
def total_cost (scenario : LaundryScenario) : ℚ :=
  let washing_cost := scenario.washer_cost * scenario.loads_washed
  let drying_cost := scenario.dryer_cost_per_10_min * (scenario.dryer_time / 10) * scenario.dryers_used
  washing_cost + drying_cost

/-- Theorem: The total cost for Samantha's laundry scenario is $11 --/
theorem samantha_laundry_cost :
  let scenario : LaundryScenario := {
    washer_cost := 4,
    dryer_cost_per_10_min := 1/4,
    loads_washed := 2,
    dryers_used := 3,
    dryer_time := 40
  }
  total_cost scenario = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_samantha_laundry_cost_l234_23435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l234_23433

theorem sin_alpha_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < π)
  (h2 : 0 < β ∧ β < π)
  (h3 : Real.tan β = 4/3)
  (h4 : Real.sin (α + β) = 5/13) : 
  Real.sin α = 63/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l234_23433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_girls_in_class_l234_23408

/-- Proves the number of girls in a class given specific conditions about class composition and average scores. -/
theorem number_of_girls_in_class : ∃ (num_girls : ℕ), 
  let num_boys : ℕ := 12
  let avg_score_boys : ℚ := 84
  let avg_score_girls : ℚ := 92
  let avg_score_class : ℚ := 86
  (num_boys * avg_score_boys + num_girls * avg_score_girls) / (num_boys + num_girls : ℚ) = avg_score_class ∧
  num_girls = 4
:= by
  -- We claim that 4 is the solution
  use 4
  -- Split the goal into two parts
  apply And.intro
  · -- Prove the equation
    simp [Nat.cast_add, Nat.cast_mul]
    norm_num
  · -- Prove that the number of girls is indeed 4
    rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_girls_in_class_l234_23408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_satisfy_condition_l234_23436

-- Define the domain (0, +∞)
def PositiveReals := {x : ℝ | x > 0}

-- Define the functions
noncomputable def f₁ : ℝ → ℝ := fun x => -(1/x)
noncomputable def f₂ : ℝ → ℝ := fun x => (Real.exp x - Real.exp (-x)) / 2
noncomputable def f₃ : ℝ → ℝ := fun x => Real.log (x + Real.sqrt (x^2 + 1))

-- State the theorem
theorem functions_satisfy_condition :
  ∀ (f : ℝ → ℝ) (x₁ x₂ : ℝ),
  x₁ > 0 → x₂ > 0 →
  (f = f₁ ∨ f = f₂ ∨ f = f₃) →
  (x₁ - x₂) * (f x₂ - f x₁) < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_satisfy_condition_l234_23436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_two_l234_23465

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 4 * x^2) - 2 * x) + 1

-- State the theorem
theorem f_sum_equals_two : f (Real.log 2) + f (Real.log (1/2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_two_l234_23465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l234_23428

/-- The rational function h(x) -/
noncomputable def h (x : ℝ) : ℝ := (x^3 - 2*x^2 + 4*x + 3) / (x^2 - 5*x + 6)

/-- The domain of h(x) -/
def domain_h : Set ℝ := {x | x ≠ 2 ∧ x ≠ 3}

theorem domain_of_h :
  domain_h = Set.Iio 2 ∪ Set.Ioo 2 3 ∪ Set.Ioi 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l234_23428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liam_reading_finish_day_l234_23455

def daysOfWeek := Fin 7

def wednesday : Fin 7 := 3

def nextDay (d : Fin 7) : Fin 7 := 
  (d + 1 : Fin 7)

def addDays (start : Fin 7) (n : ℕ) : Fin 7 :=
  (start + n : Fin 7)

def sumFirstN (n : ℕ) : ℕ := 
  n * (n + 1) / 2

theorem liam_reading_finish_day :
  addDays wednesday (sumFirstN 20) = wednesday := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liam_reading_finish_day_l234_23455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencils_bought_l234_23490

def total_pencils : ℕ := 6
def defective_pencils : ℕ := 2
def probability_no_defective_target : ℚ := 1/5

def combinations (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def probability_no_defective (n : ℕ) : ℚ :=
  (combinations (total_pencils - defective_pencils) n : ℚ) / 
  (combinations total_pencils n : ℚ)

theorem pencils_bought : 
  ∃ (n : ℕ), n > 0 ∧ n ≤ total_pencils ∧ 
  probability_no_defective n = probability_no_defective_target :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencils_bought_l234_23490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l234_23449

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line mx + (2m-1)y - 2 = 0 -/
noncomputable def slope1 (m : ℝ) : ℝ := -m / (2*m - 1)

/-- The slope of the line 3x + my + 3 = 0 -/
noncomputable def slope2 (m : ℝ) : ℝ := -3 / m

theorem perpendicular_condition (m : ℝ) :
  (m = -1 → perpendicular (slope1 m) (slope2 m)) ∧
  ¬(perpendicular (slope1 m) (slope2 m) → m = -1) := by
  sorry

#check perpendicular_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l234_23449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_of_triangle_MAF_l234_23445

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 4*x

/-- Point type representing a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The focus of the parabola -/
def F : Point := ⟨1, 0⟩

/-- Point A -/
def A : Point := ⟨5, 3⟩

/-- Theorem stating that the minimum perimeter of triangle MAF is 11 -/
theorem min_perimeter_of_triangle_MAF :
  ∀ (M : Parabola), M.x ≠ 5 ∨ M.y ≠ 3 →
  (∀ (ε : ℝ), ε > 0 →
    ∃ (M' : Parabola), distance A ⟨M'.x, M'.y⟩ + distance F ⟨M'.x, M'.y⟩ + distance A F < 11 + ε) ∧
  (∀ (M' : Parabola), distance A ⟨M'.x, M'.y⟩ + distance F ⟨M'.x, M'.y⟩ + distance A F ≥ 11) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_of_triangle_MAF_l234_23445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paving_rate_example_l234_23447

/-- The rate per square meter for paving a room -/
noncomputable def paving_rate (length width total_cost : ℝ) : ℝ :=
  total_cost / (length * width)

/-- Theorem: The paving rate for a room with given dimensions and cost is $300 per square meter -/
theorem paving_rate_example : paving_rate 5.5 3.75 6187.5 = 300 := by
  -- Unfold the definition of paving_rate
  unfold paving_rate
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paving_rate_example_l234_23447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_identity_l234_23417

theorem triangle_identity (A B C : ℝ) 
  (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi)
  (h_sides : Real.sin A / Real.sin C = 2)
  (h_trig : Real.sin B - Real.sin (A + B) = 2 * Real.sin C * Real.cos A) :
  (Real.cos B + Real.sin B)^2 + Real.sin (2 * C) = 1 + Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_identity_l234_23417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_person_is_knight_l234_23414

/-- Represents a participant in the game, either a Knight or a Liar -/
inductive Participant
| Knight
| Liar

/-- Represents the chain of participants -/
def Chain := List Participant

/-- Function to calculate the change in number based on participant type -/
def change (p : Participant) : Int :=
  match p with
  | Participant.Knight => 0
  | Participant.Liar => 1

/-- Function to calculate the total change in number for a chain -/
def totalChange (c : Chain) : Int :=
  c.foldl (fun acc p => acc + change p) 0

theorem last_person_is_knight
  (chain : Chain)
  (h1 : chain.head? = some Participant.Liar)
  (h2 : chain.getLast? ≠ chain.head?)
  (h3 : totalChange chain % 2 = 0)
  (h4 : totalChange chain.reverse % 2 = 1) :
  chain.getLast? = some Participant.Knight :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_person_is_knight_l234_23414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perpendiculars_constant_l234_23475

/-- A regular pentagon inscribed in a circle -/
structure RegularPentagon where
  r : ℝ  -- radius of the circumscribed circle
  center : ℝ × ℝ  -- center of the circle
  vertices : Fin 5 → ℝ × ℝ  -- vertices of the pentagon

/-- A point inside the pentagon -/
def InsidePoint (p : RegularPentagon) := { point : ℝ × ℝ // point ∈ Set.range p.vertices }

/-- The perpendicular distance from a point to a line segment -/
noncomputable def perpendicularDistance (point : ℝ × ℝ) (segment : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ :=
  sorry  -- Definition of perpendicular distance

/-- The sum of perpendicular distances from a point to all sides of the pentagon -/
noncomputable def sumOfPerpendiculars (p : RegularPentagon) (point : InsidePoint p) : ℝ :=
  (Finset.sum (Finset.range 5) fun i => 
    perpendicularDistance point.val (p.vertices i, p.vertices ((i + 1) % 5)))

/-- The main theorem -/
theorem sum_of_perpendiculars_constant (p : RegularPentagon) :
    ∀ point : InsidePoint p, sumOfPerpendiculars p point = 5 * p.r * Real.cos (36 * π / 180) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perpendiculars_constant_l234_23475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petyas_friends_l234_23402

/-- The number of stickers Petya has -/
def total_stickers : ℕ := sorry

/-- The number of friends Petya has -/
def num_friends : ℕ := sorry

/-- Condition 1: When giving 5 stickers to each friend, 8 stickers are left -/
axiom condition1 : total_stickers = 5 * num_friends + 8

/-- Condition 2: When giving 6 stickers to each friend, 11 more stickers are needed -/
axiom condition2 : total_stickers + 11 = 6 * num_friends

/-- Theorem: Petya has 19 friends -/
theorem petyas_friends : num_friends = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petyas_friends_l234_23402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l234_23441

/-- The function f(x) = |ax - 1| -/
def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 1|

/-- The theorem stating the value of a and the minimum value of u -/
theorem problem_solution :
  (∀ x : ℝ, f 1 x ≤ 2 ↔ x ∈ Set.Icc (-1) 3) ∧
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 1 →
    1 / (x + y) + (x + y) / z ≥ 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l234_23441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l234_23466

/-- A parallelogram with one angle of 150 degrees and two consecutive sides of lengths 10 and 18 --/
structure Parallelogram where
  angle : ℝ
  side1 : ℝ
  side2 : ℝ
  angle_eq : angle = 150
  side1_eq : side1 = 10
  side2_eq : side2 = 18

/-- The area of the parallelogram --/
noncomputable def area (p : Parallelogram) : ℝ := 60 * Real.sqrt 3

/-- Theorem stating that the area of the parallelogram is 60√3 --/
theorem parallelogram_area (p : Parallelogram) : area p = 60 * Real.sqrt 3 := by
  -- Unfold the definition of area
  unfold area
  -- The equality is trivial by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l234_23466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_planting_theorem_l234_23424

def boys_trees : ℕ := 130
def girls_trees : ℕ := 80

theorem tree_planting_theorem :
  -- Percentage by which boys planted more trees than girls
  (boys_trees - girls_trees : ℚ) / girls_trees * 100 = 62.5 ∧
  -- Fraction of total trees planted by girls
  (girls_trees : ℚ) / (boys_trees + girls_trees) = 4 / 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_planting_theorem_l234_23424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C1_C2_l234_23458

noncomputable section

/-- Curve C1 in parametric form -/
def C1 (α : Real) : Real × Real :=
  (Real.cos α, Real.sqrt 3 * Real.sin α)

/-- Curve C2 in polar form -/
def C2 (θ : Real) : Real × Real :=
  let ρ := 2 * Real.sqrt 2 / Real.sin (θ - Real.pi/4)
  (ρ * Real.cos θ, ρ * Real.sin θ)

/-- Distance between two points -/
def distance (p q : Real × Real) : Real :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_C1_C2 :
  (∃ (α θ : Real),
    let p := C1 α
    let q := C2 θ
    (∀ (β γ : Real), distance (C1 β) (C2 γ) ≥ distance p q) ∧
    distance p q = Real.sqrt 2 ∧
    p = (-1/2, 3/2)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C1_C2_l234_23458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l234_23468

/-- Represents a rhombus with given side length and shorter diagonal length -/
structure Rhombus where
  side_length : ℝ
  shorter_diagonal : ℝ

/-- Calculates the length of the longer diagonal of a rhombus -/
noncomputable def longer_diagonal (r : Rhombus) : ℝ :=
  2 * Real.sqrt (r.side_length^2 - (r.shorter_diagonal / 2)^2)

/-- Theorem: In a rhombus with side length 65 and shorter diagonal 72, the longer diagonal is 108 -/
theorem rhombus_longer_diagonal :
  let r : Rhombus := { side_length := 65, shorter_diagonal := 72 }
  longer_diagonal r = 108 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l234_23468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_exactly_three_red_prob_red_specific_draws_prob_red_B_l234_23496

-- Define the probabilities and ratios
def prob_red_A : ℚ := 1 / 3
def prob_red_combined : ℚ := 2 / 5
def ratio_A_to_B : ℚ := 1 / 2

-- Define the number of draws and successes for part 1
def num_draws : ℕ := 5
def num_successes : ℕ := 3

-- Theorem for part 1(i)
theorem prob_exactly_three_red :
  (Nat.choose num_draws num_successes : ℚ) * prob_red_A ^ num_successes * (1 - prob_red_A) ^ (num_draws - num_successes) = 40 / 243 := by sorry

-- Theorem for part 1(ii)
theorem prob_red_specific_draws :
  prob_red_A ^ 3 = 1 / 27 := by sorry

-- Theorem for part 2
theorem prob_red_B (m : ℚ) :
  let total_balls := m + 2 * m
  let red_balls := prob_red_A * m + 2 * m * (13 / 30)
  red_balls / total_balls = prob_red_combined := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_exactly_three_red_prob_red_specific_draws_prob_red_B_l234_23496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_six_l234_23415

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in slope-intercept form -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Creates a line passing through a given point with a given slope -/
noncomputable def lineFromPointAndSlope (p : Point) (m : ℝ) : Line :=
  { slope := m
  , yIntercept := p.y - m * p.x }

/-- Finds the intersection point of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line) : Point :=
  { x := (l2.yIntercept - l1.yIntercept) / (l1.slope - l2.slope)
  , y := l1.slope * ((l2.yIntercept - l1.yIntercept) / (l1.slope - l2.slope)) + l1.yIntercept }

/-- Calculates the area of a triangle given its three vertices -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem triangle_area_is_six :
  let p : Point := { x := 2, y := 2 }
  let l1 : Line := lineFromPointAndSlope p (1/2)
  let l2 : Line := lineFromPointAndSlope p 2
  let l3 : Line := { slope := -1, yIntercept := 10 }
  let p1 : Point := p
  let p2 : Point := intersectionPoint l1 l3
  let p3 : Point := intersectionPoint l2 l3
  triangleArea p1 p2 p3 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_six_l234_23415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_install_one_window_l234_23404

theorem time_to_install_one_window 
  (total_windows : ℕ) 
  (installed_windows : ℕ) 
  (time_for_remaining : ℕ) 
  (h1 : total_windows = 14)
  (h2 : installed_windows = 8)
  (h3 : time_for_remaining = 48)
  : time_for_remaining / (total_windows - installed_windows) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_install_one_window_l234_23404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_2_statement_4_l234_23492

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (subset : Line → Plane → Prop)
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Statement ②
theorem statement_2 
  (l m : Line) (α β : Plane) 
  (h1 : subset l α) 
  (h2 : parallel_line_plane l β) 
  (h3 : intersect α β m) : 
  parallel_line l m := 
sorry

-- Statement ④
theorem statement_4 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicular l α) 
  (h2 : parallel_line m l) 
  (h3 : parallel_plane α β) : 
  perpendicular m β := 
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_2_statement_4_l234_23492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_B_is_quadratic_one_var_l234_23425

/-- Represents a polynomial equation -/
structure PolynomialEquation where
  lhs : ℝ → ℝ
  rhs : ℝ → ℝ

/-- Checks if a polynomial equation is quadratic with one variable -/
def is_quadratic_one_var (eq : PolynomialEquation) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧
    ∀ x, eq.lhs x = a * x^2 + b * x + c ∧ eq.rhs x = 0

/-- The equation x^2 + x + 3 = 0 -/
def equation_B : PolynomialEquation :=
  { lhs := λ x ↦ x^2 + x + 3
    rhs := λ _ ↦ 0 }

/-- Theorem stating that equation_B is a quadratic equation with one variable -/
theorem equation_B_is_quadratic_one_var : is_quadratic_one_var equation_B := by
  sorry

#check equation_B_is_quadratic_one_var

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_B_is_quadratic_one_var_l234_23425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_factory_average_production_l234_23483

/-- Given production data for a TV factory, prove the average production for the entire month. -/
theorem tv_factory_average_production
  (first_period_avg : ℝ) (first_period_days : ℕ)
  (second_period_avg : ℝ) (second_period_days : ℕ)
  (total_days : ℕ)
  (h1 : first_period_avg = 65)
  (h2 : first_period_days = 25)
  (h3 : second_period_avg = 35)
  (h4 : second_period_days = 5)
  (h5 : total_days = first_period_days + second_period_days) :
  (first_period_avg * (first_period_days : ℝ) + second_period_avg * (second_period_days : ℝ)) / (total_days : ℝ) = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_factory_average_production_l234_23483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_segment_region_area_l234_23477

/-- The area of the region consisting of all line segments of length 4 that are tangent to a circle of radius 3 at their midpoints -/
theorem tangent_segment_region_area
  (circle_radius : Real)
  (segment_length : Real)
  : circle_radius = 3 → segment_length = 4 →
    (π * ((circle_radius^2 + (segment_length/2)^2) - circle_radius^2)) = 4*π :=
by
  intros h_radius h_segment
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_segment_region_area_l234_23477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l234_23422

/-- Given a hyperbola with the standard equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    asymptotic equations y = ±(√3/3)x, and distance from vertex to asymptote equal to 1,
    prove that its equation is x²/4 - 3y²/4 = 1. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x y, y = Real.sqrt 3 / 3 * x ∨ y = -Real.sqrt 3 / 3 * x) →
  (∃ x₀ y₀, x₀^2 / a^2 - y₀^2 / b^2 = 1 ∧
    |Real.sqrt 3 * x₀ - 3 * y₀| / Real.sqrt (3^2 + 3^2) = 1) →
  (∀ x y, x^2 / 4 - 3 * y^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l234_23422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_width_calculation_l234_23464

/-- Calculates the width of a river given the distance of a tower from the river bank,
    the height of the observation point in the tower, and the angle of sight to the opposite bank. -/
noncomputable def river_width (tower_distance : ℝ) (observation_height : ℝ) (sight_angle : ℝ) : ℝ :=
  tower_distance * Real.tan (sight_angle * Real.pi / 180)

/-- Theorem stating that the width of the river is equal to 45 * tan(20°) meters
    given the specified conditions. -/
theorem river_width_calculation :
  let tower_distance : ℝ := 45
  let observation_height : ℝ := 18
  let sight_angle : ℝ := 20
  river_width tower_distance observation_height sight_angle = 45 * Real.tan (20 * Real.pi / 180) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_width_calculation_l234_23464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_asymptotes_sum_l234_23409

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x^3 - 3*x^2 - 4*x)

noncomputable def num_holes (f : ℝ → ℝ) : ℕ := sorry
noncomputable def num_vertical_asymptotes (f : ℝ → ℝ) : ℕ := sorry
noncomputable def num_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := sorry
noncomputable def num_oblique_asymptotes (f : ℝ → ℝ) : ℕ := sorry

theorem rational_function_asymptotes_sum :
  let a := num_holes f
  let b := num_vertical_asymptotes f
  let c := num_horizontal_asymptotes f
  let d := num_oblique_asymptotes f
  a + 2*b + 3*c + 4*d = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_asymptotes_sum_l234_23409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_l234_23450

def puzzle_result (a : ℤ) : ℤ :=
  3 * a - 6 * Int.ceil (a / 7 : ℚ) + 6 * Int.floor (a / 7 : ℚ)

theorem puzzle_solution :
  (puzzle_result 1 = 9) ∧
  (puzzle_result 4 = 12) ∧
  (puzzle_result 7 = 15) ∧
  (puzzle_result 8 = 18) := by
  sorry

#eval puzzle_result 1
#eval puzzle_result 4
#eval puzzle_result 7
#eval puzzle_result 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_l234_23450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_probability_is_one_l234_23451

/-- Represents the players in the game -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Carol : Player

/-- Represents the probability of passing the ball from one player to another -/
def pass_probability : Player → Player → ℚ
| Player.Alice, Player.Bob => 2/3
| Player.Alice, Player.Alice => 1/3
| Player.Bob, Player.Carol => 1/4
| Player.Bob, Player.Alice => 3/4
| Player.Carol, Player.Alice => 1/2
| Player.Carol, Player.Bob => 1/2
| _, _ => 0

/-- The probability of Alice having the ball after three turns -/
def probability_alice_after_three_turns : ℚ :=
  (pass_probability Player.Alice Player.Bob) * (pass_probability Player.Bob Player.Carol) * (pass_probability Player.Carol Player.Alice) +
  (pass_probability Player.Alice Player.Bob) * (pass_probability Player.Bob Player.Alice) +
  (pass_probability Player.Alice Player.Alice) * (pass_probability Player.Alice Player.Alice) * (pass_probability Player.Alice Player.Alice)

/-- Theorem stating that the probability of Alice having the ball after three turns is 1 -/
theorem alice_probability_is_one : probability_alice_after_three_turns = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_probability_is_one_l234_23451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l234_23479

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi / 4 + x) * Real.cos (Real.pi / 4 + x)

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l234_23479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_is_odd_l234_23463

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x)

def isDomainPoint (x : ℝ) : Prop :=
  ∀ k : ℤ, x ≠ (1/2 : ℝ) * (k : ℝ) * Real.pi + Real.pi/4

def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, isDomainPoint x → isDomainPoint (-x) → f (-x) = -f x

theorem tan_2x_is_odd : isOddFunction f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_is_odd_l234_23463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l234_23469

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Define the base case for n = 0
  | n + 1 => 1 + 2 / sequence_a n

theorem sequence_a_formula (n : ℕ) : 
  sequence_a n = 2 + 3 / ((-2)^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l234_23469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l234_23413

theorem trig_identity (α : ℝ) (h : Real.sin α = 3 * Real.cos α) :
  Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α - 3 * Real.cos α ^ 2 = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l234_23413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_intersection_l234_23410

/-- The parabola y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

/-- Point A on the parabola -/
def A : ℝ × ℝ := (2, 4)

/-- Slope of the normal line at A -/
noncomputable def normal_slope : ℝ := -1 / (2 * A.1)

/-- The normal line to the parabola at A -/
noncomputable def normal_line (x : ℝ) : ℝ := normal_slope * (x - A.1) + A.2

/-- Point B, the other intersection of the normal line and the parabola -/
def B : ℝ × ℝ := (-2.25, 5.0625)

theorem normal_intersection :
  A.2 = parabola A.1 ∧
  B.2 = parabola B.1 ∧
  B.2 = normal_line B.1 ∧
  B ≠ A := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_intersection_l234_23410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l234_23457

def is_valid_arrangement (arr : List Char) : Bool :=
  let n := arr.length
  List.all (List.range n) (fun i =>
    (i + 1 < n →
      ((arr.get! i = 'A' → arr.get! (i+1) ≠ 'B' ∧ arr.get! (i+1) ≠ 'C') ∧
       (arr.get! (i+1) = 'A' → arr.get! i ≠ 'B' ∧ arr.get! i ≠ 'C') ∧
       (arr.get! i = 'D' → arr.get! (i+1) ≠ 'E') ∧
       (arr.get! (i+1) = 'D' → arr.get! i ≠ 'E'))))

def count_valid_arrangements (people : List Char) : Nat :=
  (people.permutations.filter is_valid_arrangement).length

theorem valid_arrangements_count :
  count_valid_arrangements ['A', 'B', 'C', 'D', 'E'] = 28 := by
  sorry

#eval count_valid_arrangements ['A', 'B', 'C', 'D', 'E']

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l234_23457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_ticket_sales_l234_23438

/-- Calculates the total money made from ticket sales at a zoo --/
theorem zoo_ticket_sales (total_people : ℕ) (num_adults : ℕ) (adult_price : ℕ) (kid_price : ℕ) : 
  total_people = 254 → num_adults = 51 → adult_price = 28 → kid_price = 12 →
  num_adults * adult_price + (total_people - num_adults) * kid_price = 3864 := by
  intros h1 h2 h3 h4
  -- Here we would normally write the proof
  sorry

#check zoo_ticket_sales

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_ticket_sales_l234_23438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l234_23419

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2^x - 2^(-x)

-- Theorem stating that f is odd and monotonically increasing
theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l234_23419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l234_23439

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := (1 - Real.log x) / (x^2)

-- Theorem statement
theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let k : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = k * (x - x₀) ↔ y = x - 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l234_23439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_special_polynomial_l234_23406

theorem gcd_special_polynomial (x : ℤ) : 
  (∃ k : ℤ, x = 18711 * k) → 
  Int.gcd ((4*x+5)*(5*x+3)*(6*x+7)*(3*x+11)) x = 1155 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_special_polynomial_l234_23406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_gt_one_and_pos_l234_23403

theorem abs_gt_one_and_pos : ¬(∀ a : ℝ, (|a| > 1 → a > 0) ∧ (a > 0 → |a| > 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_gt_one_and_pos_l234_23403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_floor_frac_l234_23499

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

/-- Theorem: If x is a nonzero real number such that frac(x), ⌊x⌋, and x form a geometric sequence
    in that order, then x = √5/2 -/
theorem geometric_sequence_floor_frac (x : ℝ) (hx : x ≠ 0) :
  (frac x * frac x = ⌊x⌋ * x) → x = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_floor_frac_l234_23499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_term_of_sequence_l234_23498

noncomputable def sequenceMonomial (n : ℕ) (a : ℝ) : ℝ := Real.sqrt (n : ℝ) * a^(n+1)

theorem nth_term_of_sequence (n : ℕ) (a : ℝ) (h : n > 0) :
  sequenceMonomial n a = Real.sqrt (n : ℝ) * a^(n+1) :=
by
  -- The proof is trivial as it's just the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_term_of_sequence_l234_23498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_40pi_l234_23481

/-- A shape composed of two rectangles -/
structure CompositeShape where
  rect1_length : ℝ
  rect1_width : ℝ
  rect2_length : ℝ
  rect2_width : ℝ

/-- The volume of a solid formed by rotating a rectangle about an axis -/
noncomputable def rotationVolume (outerRadius : ℝ) (height : ℝ) (thickness : ℝ) : ℝ :=
  2 * Real.pi * outerRadius * height * thickness

/-- The total volume of the solid formed by rotating the composite shape about the y-axis -/
noncomputable def totalVolume (shape : CompositeShape) : ℝ :=
  rotationVolume shape.rect1_length shape.rect1_width shape.rect1_width +
  rotationVolume (shape.rect1_length + shape.rect2_length) shape.rect2_width shape.rect2_length

/-- Theorem stating that the volume of the solid is 40π cubic units -/
theorem volume_is_40pi (shape : CompositeShape)
  (h1 : shape.rect1_length = 4)
  (h2 : shape.rect1_width = 1)
  (h3 : shape.rect2_length = 2)
  (h4 : shape.rect2_width = 4)
  : totalVolume shape = 40 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_40pi_l234_23481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_area_theorem_l234_23478

/-- Projectile motion in 2D space -/
structure ProjectileMotion where
  u : ℝ  -- Initial velocity
  g : ℝ  -- Acceleration due to gravity
  φ : ℝ  -- Launch angle

/-- Trajectory of the projectile -/
noncomputable def trajectory (p : ProjectileMotion) (t : ℝ) : ℝ × ℝ :=
  (p.u * t * Real.cos p.φ, p.u * t * Real.sin p.φ - (1/2) * p.g * t^2)

/-- Time at which the projectile reaches its highest point -/
noncomputable def timeAtHighestPoint (p : ProjectileMotion) : ℝ :=
  p.u * Real.sin p.φ / p.g

/-- Coordinates of the highest point in the trajectory -/
noncomputable def highestPoint (p : ProjectileMotion) : ℝ × ℝ :=
  let t := timeAtHighestPoint p
  (p.u^2 / (2 * p.g) * Real.sin (2 * p.φ), p.u^2 / (2 * p.g) * Real.sin p.φ^2)

/-- Area enclosed by the curve traced by highest points -/
noncomputable def enclosedArea (p : ProjectileMotion) : ℝ :=
  Real.pi / 8 * p.u^4 / p.g^2

theorem projectile_area_theorem (p : ProjectileMotion) :
  enclosedArea p = Real.pi / 8 * p.u^4 / p.g^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_area_theorem_l234_23478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_sine_function_l234_23472

-- Define the function f as noncomputable
noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

-- State the theorem
theorem parity_of_sine_function (ω φ : ℝ) (h : ω > 0) :
  (∀ x, f ω φ x = f ω φ (-x)) ↔ ∃ k : ℤ, φ = k * Real.pi + Real.pi / 2 := by
  sorry

#check parity_of_sine_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_sine_function_l234_23472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_apartment_size_l234_23489

/-- The rental rate in dollars per square foot -/
noncomputable def rental_rate : ℝ := 1.20

/-- Jillian's monthly budget for rent in dollars -/
noncomputable def monthly_budget : ℝ := 720

/-- The largest apartment size Jillian should consider in square feet -/
noncomputable def max_apartment_size : ℝ := monthly_budget / rental_rate

theorem largest_apartment_size :
  max_apartment_size = 600 :=
by
  -- Unfold the definitions
  unfold max_apartment_size monthly_budget rental_rate
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_apartment_size_l234_23489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l234_23430

theorem circle_area_ratio : 
  ∃ (large_circle_diameter small_circle_diameter square_side : ℝ),
  large_circle_diameter = 4 ∧
  small_circle_diameter = 2 ∧
  square_side = 1 ∧
  let large_circle_area := π * (large_circle_diameter / 2)^2
  let small_circle_area := π * (small_circle_diameter / 2)^2
  let square_area := square_side^2
  let blue_area := large_circle_area - small_circle_area
  let red_area := small_circle_area - square_area
  blue_area / red_area = 3 * π / (π - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l234_23430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_proof_l234_23493

-- Define the hyperbola
structure Hyperbola where
  center : ℝ × ℝ
  foci_on_axes : Bool
  eccentricity : ℝ
  point_on_curve : ℝ × ℝ

-- Define the equation of a hyperbola
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 10 - x^2 / 5 = 1

-- Theorem statement
theorem hyperbola_equation_proof (h : Hyperbola) :
  h.center = (0, 0) →
  h.foci_on_axes = true →
  h.eccentricity = Real.sqrt 6 / 2 →
  h.point_on_curve = (2, 3 * Real.sqrt 2) →
  ∀ (x y : ℝ), hyperbola_equation x y ↔ (x, y) ∈ Set.range (λ t : ℝ × ℝ ↦ t) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_proof_l234_23493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_equals_six_l234_23473

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := 3 + (a^x - 1)/(a^x + 1) + x * Real.cos x

-- State the theorem
theorem sum_of_max_min_equals_six (a : ℝ) (ha : 0 < a ∧ a ≠ 1) :
  ∃ (M N : ℝ), (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f a x ≤ M) ∧
                (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → N ≤ f a x) ∧
                (M + N = 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_equals_six_l234_23473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_points_difference_l234_23471

/-- Given an angle α with vertex at the origin, initial side along the positive x-axis,
    points A(1,a) and B(2,b) on its terminal side, and cos(2α) = 2/3,
    prove that |a - b| = √5/5 -/
theorem angle_points_difference (α : ℝ) (a b : ℝ) 
  (h1 : Real.cos (2 * α) = 2/3)
  (h2 : ∃ (t : ℝ), t > 0 ∧ Real.cos α * t = 1 ∧ Real.sin α * t = a)
  (h3 : ∃ (s : ℝ), s > 0 ∧ Real.cos α * s = 2 ∧ Real.sin α * s = b) :
  |a - b| = Real.sqrt 5 / 5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_points_difference_l234_23471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_P_P_2013_l234_23405

-- Define the set [n]
def range (n : ℕ) : Set ℕ := {i | 1 ≤ i ∧ i ≤ n}

-- Define P(S) as the set of non-empty subsets of S
def P {α : Type*} (S : Set α) : Set (Set α) := {T | T ⊆ S ∧ T.Nonempty}

-- State the theorem
theorem last_digit_of_P_P_2013 : 
  (Finset.card (Finset.powerset (Finset.powerset (Finset.range 2013)) \ {∅})) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_P_P_2013_l234_23405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_knitting_theorem_l234_23412

/-- Represents the time in hours to knit each item -/
structure KnittingTime where
  hat : ℚ
  scarf : ℚ
  mitten : ℚ
  sock : ℚ
  sweater : ℚ

/-- Calculates the total time to knit one complete set of clothes -/
def timePerSet (kt : KnittingTime) : ℚ :=
  kt.hat + kt.scarf + 2 * kt.mitten + 2 * kt.sock + kt.sweater

/-- Theorem: Martha can knit for 3 grandchildren in 48 hours -/
theorem martha_knitting_theorem (kt : KnittingTime) 
  (h_hat : kt.hat = 2)
  (h_scarf : kt.scarf = 3)
  (h_mitten : kt.mitten = 1)
  (h_sock : kt.sock = 3/2)
  (h_sweater : kt.sweater = 6)
  (h_total_time : 48 = 3 * timePerSet kt) : 
  3 = 48 / timePerSet kt := by
  sorry

/-- Verify the theorem with the given values -/
example : 3 = 48 / timePerSet ⟨2, 3, 1, 3/2, 6⟩ := by
  have h : 48 = 3 * timePerSet ⟨2, 3, 1, 3/2, 6⟩ := by
    rfl
  exact martha_knitting_theorem ⟨2, 3, 1, 3/2, 6⟩ rfl rfl rfl rfl rfl h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_knitting_theorem_l234_23412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_bob_sum_l234_23495

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem alice_bob_sum : 
  ∀ (A B : ℕ),
  (A ∈ Finset.range 60 \ Finset.range 1) →
  (B ∈ Finset.range 60 \ Finset.range 1) →
  (¬ Nat.Prime A) →
  (is_perfect_square (60 * B + A)) →
  A + B = 45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_bob_sum_l234_23495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_pi_range_l234_23484

theorem count_integers_in_pi_range : 
  (Finset.range (Int.toNat (Int.floor (15 * Real.pi) + 1)) ∩ 
   Finset.Icc (Int.toNat (Int.ceil (-5 * Real.pi))) 
              (Int.toNat (Int.floor (15 * Real.pi)))).card = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_pi_range_l234_23484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_increase_rate_l234_23460

/-- The rate of increase in the straight-line distance between two cars -/
noncomputable def rate_of_increase (v1 v2 : ℝ) : ℝ :=
  Real.sqrt (v1^2 + v2^2)

/-- Theorem: The rate of increase in the straight-line distance between two cars
    moving perpendicular to each other, with one car traveling at 30 km/h and
    the other at 40 km/h, is equal to 50 km/h. -/
theorem car_distance_increase_rate :
  rate_of_increase 30 40 = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_increase_rate_l234_23460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_angle_relation_l234_23444

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Theorem 1
theorem area_of_triangle (t : Triangle) 
  (h1 : triangle_conditions t)
  (h2 : t.a = 10)
  (h3 : t.c = 3)
  (h4 : Real.tan t.B = 3/4) :
  (1/2) * t.a * t.c * Real.sin t.B = 9 := by
  sorry

-- Theorem 2
theorem angle_relation (t : Triangle)
  (h1 : triangle_conditions t)
  (h2 : t.a^2 = t.b * (t.b + t.c)) :
  t.A = 2 * t.B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_angle_relation_l234_23444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l234_23476

theorem min_value_expression (x : ℝ) (hx : x > 0) : 3 * x^4 + 8 / x^3 ≥ 11 ∧ 
  (3 * x^4 + 8 / x^3 = 11 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l234_23476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l234_23442

def S (n : ℕ) (a : ℕ → ℕ) : ℕ := 2 * a n - 2^n + 1

theorem sequence_formula (a : ℕ → ℕ) (h : ∀ n : ℕ, n > 0 → S n a = 2 * a n - 2^n + 1) :
  ∀ n : ℕ, n > 0 → a n = n * 2^(n - 1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l234_23442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l234_23437

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_problem (a b : V) 
  (h1 : ‖a‖^2 = (3 : ℝ) / 2)
  (h2 : inner a b = (3 : ℝ) / 2)
  (h3 : ‖a + b‖ = 2 * Real.sqrt 2) :
  ‖b‖ = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l234_23437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l234_23416

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + y^2 + 2 * f (x * y)) = (f (x + y))^2

/-- The set of x where f(x) = -1 for the third type of solution. -/
def X : Set ℝ := {x : ℝ | x < -2/3}

/-- The third type of solution function. -/
noncomputable def f₃ (x : ℝ) : ℝ :=
  if x < -2/3 then -1 else 1

/-- Theorem stating the possible forms of functions satisfying the equation. -/
theorem functional_equation_solutions (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) :
  (f = id) ∨ (f = λ _ ↦ 0) ∨ (∃ X : Set ℝ, X ⊆ Set.Iio (-2/3) ∧ f = f₃) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l234_23416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graduation_ceremony_chairs_correct_l234_23452

def graduation_ceremony_chairs 
  (graduates : ℕ) 
  (parents_per_graduate : ℕ) 
  (additional_family_percentage : ℚ) 
  (teachers : ℕ) 
  (admin_teacher_ratio : ℚ) : ℕ :=
  let parents := graduates * parents_per_graduate
  let additional_family := (graduates : ℚ) * additional_family_percentage
  let administrators := (teachers : ℚ) * admin_teacher_ratio
  ((graduates : ℚ) + (parents : ℚ) + additional_family + (teachers : ℚ) + administrators).floor.toNat

theorem graduation_ceremony_chairs_correct : 
  graduation_ceremony_chairs 150 2 (2/5) 35 (4/3) = 589 := by
  sorry

#eval graduation_ceremony_chairs 150 2 (2/5) 35 (4/3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graduation_ceremony_chairs_correct_l234_23452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l234_23482

/-- Definition of the sequence a_n -/
def a : ℕ → ℝ := sorry

/-- Definition of S_n, the sum of first n terms of a_n -/
def S : ℕ → ℝ := sorry

/-- Definition of T_n, the sum of first n terms of 2 * 3^n / (a_n * a_{n+1}) -/
def T : ℕ → ℝ := sorry

/-- The main theorem stating the properties of sequences a_n and T_n -/
theorem sequence_properties :
  (∀ n, S n = (a (n + 1) / 2) - n - 1) →
  a 2 = 8 →
  (∀ n, n ≥ 1 → a n = 3^(n - 1)) ∧
  (∀ n, T n = 1/2 - 1/(3^(n + 1) - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l234_23482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_is_nine_l234_23427

/-- The repeating decimal representation of 7/19 -/
def repeating_decimal : ℚ := 7 / 19

/-- The two-digit repeating part of the decimal representation -/
def repeating_part : ℕ := 
  (((100 * repeating_decimal - repeating_decimal) * 19).floor.toNat)

/-- The first digit of the repeating part -/
def c : ℕ := repeating_part / 10

/-- The second digit of the repeating part -/
def d : ℕ := repeating_part % 10

theorem sum_of_digits_is_nine : c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_is_nine_l234_23427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_side_lengths_l234_23497

/-- A trapezoid with the given properties --/
structure Trapezoid where
  area : ℝ
  long_base : ℝ
  short_base : ℝ
  non_parallel_side : ℝ
  area_eq : area = 3600
  diagonal_divides : ∃ (a b : ℝ), a / b = 5 / 4 ∧ 
    a + b = area ∧ 
    a = (Real.sqrt 3 / 4) * long_base ^ 2 ∧
    b = (1 / 2) * short_base * non_parallel_side

/-- The theorem stating the side lengths of the trapezoid --/
theorem trapezoid_side_lengths (t : Trapezoid) : 
  t.long_base = 40 * Real.sqrt (5 / Real.sqrt 3) ∧
  t.short_base = 32 * Real.sqrt (5 / Real.sqrt 3) ∧
  t.non_parallel_side = 8 * Real.sqrt (105 / Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_side_lengths_l234_23497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_rectangle_l234_23470

/-- Function to calculate the area of a triangle given three points -/
def area_triangle (A B C : ℝ × ℝ) : ℝ := 
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  0.5 * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Given a 6x7 rectangle with triangle ABC where:
    A is on the left side, 3 units from the bottom
    B is on the bottom side, 5 units from the left
    C is on the top side, 2 units from the right
    Prove that the area of triangle ABC is 17.5 square units -/
theorem triangle_area_in_rectangle (A B C : ℝ × ℝ) : 
  A.1 = 0 ∧ A.2 = 3 ∧
  B.1 = 5 ∧ B.2 = 0 ∧
  C.1 = 4 ∧ C.2 = 7 →
  area_triangle A B C = 17.5 :=
by
  intro h
  simp [area_triangle]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_rectangle_l234_23470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l234_23446

-- Define the function
noncomputable def f (x : Real) : Real := (3:Real)^x * (3:Real)^x - 2 * (3:Real)^x + 2

-- State the theorem
theorem min_value_of_f :
  ∃ (min_val : Real), min_val = 1 ∧
  ∀ x : Real, -1 ≤ x ∧ x ≤ 1 → f x ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l234_23446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_unformable_amount_correct_l234_23434

/-- Represents the coin denominations in Limonia -/
def coin_denominations (n : ℕ) : Finset ℕ :=
  {3*n - 2, 6*n - 1, 6*n + 2, 6*n + 5}

/-- Predicate to check if an amount can be formed using the given coin denominations -/
def can_form_amount (n : ℕ) (amount : ℕ) : Prop :=
  ∃ (coeffs : ℕ → ℕ), amount = (coin_denominations n).sum (λ d ↦ d * coeffs d)

/-- The largest amount that cannot be formed using the given coin denominations -/
def largest_unformable_amount (n : ℕ) : ℕ := 6*n^2 - 4*n - 3

/-- Main theorem: The largest amount that cannot be formed is 6n^2 - 4n - 3 -/
theorem largest_unformable_amount_correct (n : ℕ) :
  (∀ m : ℕ, m > largest_unformable_amount n → can_form_amount n m) ∧
  ¬(can_form_amount n (largest_unformable_amount n)) := by
  sorry

#check largest_unformable_amount_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_unformable_amount_correct_l234_23434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_dot_products_l234_23411

/-- An equilateral triangle with side length 6 -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  side_length : ℝ
  equilateral : side_length = 6 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = side_length^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = side_length^2 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = side_length^2

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem equilateral_triangle_dot_products (t : EquilateralTriangle) :
  let AB := (t.B.1 - t.A.1, t.B.2 - t.A.2)
  let AC := (t.C.1 - t.A.1, t.C.2 - t.A.2)
  let BC := (t.C.1 - t.B.1, t.C.2 - t.B.2)
  dot_product AB AC = 18 ∧ dot_product AB BC = -18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_dot_products_l234_23411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_cost_proof_l234_23487

/-- Proves that the total cost of gas for the trip is $200 given the initial conditions --/
theorem gas_cost_proof (initial_friends : ℚ) (final_friends : ℚ) (cost_decrease : ℚ) :
  initial_friends = 5 →
  final_friends = 8 →
  cost_decrease = 15 →
  ∃ (total_cost : ℚ), total_cost = 200 ∧
    (total_cost / initial_friends) - (total_cost / final_friends) = cost_decrease :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_cost_proof_l234_23487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_imply_a_range_l234_23485

/-- The function f(x) = (1/2)x^2 + a*ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a * Real.log x

/-- The derivative of f(x) -/
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := x + a / x

/-- Theorem: If f(x) has two perpendicular tangent lines on (1, 2), then a ∈ (-3, -2) -/
theorem perpendicular_tangents_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 ∧ 
    f_prime a x₁ * f_prime a x₂ = -1) →
  -3 < a ∧ a < -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_imply_a_range_l234_23485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_leq_9_min_a_for_B_subset_A_l234_23461

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 4| + |x + 1|

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | f x < 2*x + a}

-- Define the set B
def B : Set ℝ := {x | x^2 - 3*x < 0}

-- Theorem for part I
theorem solution_set_f_leq_9 :
  {x : ℝ | f x ≤ 9} = Set.Icc (-2) 4 := by sorry

-- Theorem for part II
theorem min_a_for_B_subset_A :
  (∃ a : ℝ, B ⊆ A a) → (∀ a : ℝ, B ⊆ A a → a ≥ 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_leq_9_min_a_for_B_subset_A_l234_23461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_movement_time_l234_23448

-- Define the circle and ant movements
def LargeCircle : Type := Unit
def SmallCircle : Type := Unit
def AntPath : Type := LargeCircle ⊕ (SmallCircle × SmallCircle)

-- Define the speeds and time variables
def initialSpeedA : ℚ := 3
def initialSpeedB : ℚ := 2
def newSpeedA : ℚ := initialSpeedA * (4/3)
def T₁ : ℚ := 0 -- Placeholder value
def T₂ : ℚ := 0 -- Placeholder value

-- Define the sum of cubes
def sumOfCubes : ℕ → ℚ
  | 0 => 0
  | n + 1 => (n + 1)^3 - n^3 + sumOfCubes n

-- State the theorem
theorem ant_movement_time : 
  -- Given conditions
  (∀ n : ℕ, n ≤ 100 → sumOfCubes n = n * (n + 1) * (2*n + 1) / 2) →
  (T₁ + T₂ = sumOfCubes 100) →
  -- Conclusion
  (1015000 : ℚ) / 9 = (T₁ + T₂) * (2/3) / 3 := by
  sorry

#check ant_movement_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_movement_time_l234_23448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_miquels_theorem_l234_23432

/-- Representation of a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Representation of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to check if a point lies on a circle -/
def lies_on_circle (p : Point) (center : Point) (radius : ℝ) : Prop :=
  (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2

/-- Function to check if a point lies on a line -/
def lies_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Function to check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Miquel's theorem -/
theorem miquels_theorem (l1 l2 l3 l4 : Line) (O M : Point) :
  -- Four lines intersect pairwise
  (∃ A B C D P Q : Point,
    (lies_on_line A l1 ∧ lies_on_line A l2) ∧
    (lies_on_line B l1 ∧ lies_on_line B l3) ∧
    (lies_on_line C l2 ∧ lies_on_line C l4) ∧
    (lies_on_line D l3 ∧ lies_on_line D l4) ∧
    (lies_on_line P l1 ∧ lies_on_line P l4) ∧
    (lies_on_line Q l2 ∧ lies_on_line Q l3)) →
  -- M is the Miquel point
  (∃ r : ℝ, lies_on_circle M O r) →
  -- Four out of six intersection points lie on a circle centered at O
  (∃ r : ℝ, 
    (lies_on_circle A O r ∧ lies_on_circle B O r ∧ 
     lies_on_circle C O r ∧ lies_on_circle D O r) ∨
    (lies_on_circle A O r ∧ lies_on_circle B O r ∧ 
     lies_on_circle C O r ∧ lies_on_circle P O r) ∨
    (lies_on_circle A O r ∧ lies_on_circle B O r ∧ 
     lies_on_circle C O r ∧ lies_on_circle Q O r) ∨
    (lies_on_circle A O r ∧ lies_on_circle B O r ∧ 
     lies_on_circle D O r ∧ lies_on_circle P O r) ∨
    (lies_on_circle A O r ∧ lies_on_circle B O r ∧ 
     lies_on_circle D O r ∧ lies_on_circle Q O r) ∨
    (lies_on_circle A O r ∧ lies_on_circle C O r ∧ 
     lies_on_circle D O r ∧ lies_on_circle P O r) ∨
    (lies_on_circle A O r ∧ lies_on_circle C O r ∧ 
     lies_on_circle D O r ∧ lies_on_circle Q O r) ∨
    (lies_on_circle B O r ∧ lies_on_circle C O r ∧ 
     lies_on_circle D O r ∧ lies_on_circle P O r) ∨
    (lies_on_circle B O r ∧ lies_on_circle C O r ∧ 
     lies_on_circle D O r ∧ lies_on_circle Q O r)) →
  -- The line through the remaining two points contains M and is perpendicular to OM
  ∃ l : Line, 
    (∀ X Y : Point, 
      (lies_on_line X l ∧ lies_on_line Y l) ∧ 
      (¬lies_on_circle X O r ∧ ¬lies_on_circle Y O r)) →
    lies_on_line M l ∧ perpendicular l (Line.mk (M.x - O.x) (M.y - O.y) 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_miquels_theorem_l234_23432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_person_Y_share_l234_23462

/-- Represents the ratio of each person's share --/
structure Ratio where
  A : ℝ
  B : ℝ
  X : ℝ
  Y : ℝ
  Z : ℝ

/-- Calculates the total ratio --/
noncomputable def totalRatio (r : Ratio) : ℝ := r.A + r.B + r.X + r.Y + r.Z

/-- Calculates a person's share based on their ratio and the total amount --/
noncomputable def calculateShare (personalRatio : ℝ) (totalRatio : ℝ) (totalAmount : ℝ) : ℝ :=
  (personalRatio / totalRatio) * totalAmount

theorem person_Y_share (totalAmount : ℝ) (r : Ratio) :
  totalAmount = 9870 ∧
  r.A = 3.5 ∧
  r.B = 7/8 ∧
  r.X = 47 ∧
  r.Y = 21.5 ∧
  r.Z = 35.75 ∧
  r.A + r.B = (1/2) * r.X →
  abs (calculateShare r.Y (totalRatio r) totalAmount - 1954.41) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_person_Y_share_l234_23462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_distance_is_12_l234_23423

/-- A rectangular parallelepiped with dimensions 5×5×4 -/
structure Parallelepiped where
  length : ℝ := 5
  width : ℝ := 5
  height : ℝ := 4

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The vertices of the parallelepiped -/
def A : Point3D := ⟨5, 0, 0⟩
def B : Point3D := ⟨0, 5, 0⟩
def C : Point3D := ⟨0, 0, 4⟩
def D : Point3D := ⟨5, 5, 4⟩

/-- The perpendicular distance from a point to a plane -/
noncomputable def perpendicular_distance (p : Point3D) (a b c : Point3D) : ℝ := sorry

/-- The theorem stating the perpendicular distance from D to plane ABC is 12 -/
theorem perpendicular_distance_is_12 (p : Parallelepiped) :
  perpendicular_distance D A B C = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_distance_is_12_l234_23423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axes_imply_phi_l234_23407

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem symmetry_axes_imply_phi (ω φ : ℝ) : 
  (∀ x, f ω φ (π/4 - x) = f ω φ (π/4 + x)) ∧ 
  (∀ x, f ω φ (5*π/4 - x) = f ω φ (5*π/4 + x)) →
  φ = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axes_imply_phi_l234_23407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_l234_23418

theorem problem_1 (α : Real) (x y : Real) :
  x = 4 ∧ y = -3 →
  2 * Real.sin α + Real.cos α = -2/5 := by
  intro h
  sorry

theorem problem_2 (α : Real) (x y a : Real) :
  a ≠ 0 ∧ x = 4*a ∧ y = -3*a →
  (2 * Real.sin α + Real.cos α = 2/5 ∨ 2 * Real.sin α + Real.cos α = -2/5) := by
  intro h
  sorry

theorem problem_3 (α : Real) (x y : Real) :
  |x| / |y| = 4/3 →
  (2 * Real.sin α + Real.cos α = 2 ∨
   2 * Real.sin α + Real.cos α = 2/5 ∨
   2 * Real.sin α + Real.cos α = -2 ∨
   2 * Real.sin α + Real.cos α = -2/5) := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_l234_23418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ara_height_calculation_l234_23443

-- Define the initial heights and growth rates
noncomputable def initial_height : ℝ := 75 / 1.25
def shea_growth_rate : ℝ := 0.25
def ara_growth_ratio : ℝ := 0.75

-- Define Shea's current height
def shea_current_height : ℝ := 75

-- Define the function to calculate Ara's current height
noncomputable def ara_current_height : ℝ :=
  initial_height + ara_growth_ratio * (shea_current_height - initial_height)

-- Theorem statement
theorem ara_height_calculation :
  ara_current_height = 71.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ara_height_calculation_l234_23443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_proof_l234_23467

theorem problem_proof : (1/2)^(-2 : ℤ) - Real.sqrt 12 - (-2023 : ℤ)^(0 : ℕ) + 6 * Real.tan (π/6) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_proof_l234_23467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l234_23440

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A line with slope 1 and y-intercept 3 -/
def line (x : ℝ) : ℝ := x + 3

theorem ellipse_equation (e : Ellipse) 
  (h_single_point : ∃! p : ℝ × ℝ, p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1 ∧ p.2 = line p.1)
  (h_eccentricity : eccentricity e = Real.sqrt 5 / 5) :
  e.a^2 = 5 ∧ e.b^2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l234_23440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_rate_example_l234_23429

/-- Calculate the discount rate given the marked price and selling price -/
noncomputable def discount_rate (marked_price selling_price : ℝ) : ℝ :=
  (marked_price - selling_price) / marked_price * 100

theorem discount_rate_example :
  let marked_price : ℝ := 240
  let selling_price : ℝ := 120
  discount_rate marked_price selling_price = 50 := by
  -- Unfold the definition of discount_rate
  unfold discount_rate
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_rate_example_l234_23429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_license_plates_l234_23400

/-- The number of states in the US -/
def total_states : ℕ := 50

/-- The amount Paul earns per percentage point of states he has plates from -/
def earnings_per_percent : ℚ := 2

/-- The total amount Paul earns -/
def total_earnings : ℚ := 160

/-- The number of states Paul has plates from -/
def states_collected : ℕ := 40

theorem paul_license_plates :
  states_collected = (total_earnings / earnings_per_percent * total_states : ℚ).floor := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_license_plates_l234_23400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_semicircle_l234_23420

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A ray in 2D space -/
structure Ray2D where
  origin : Point2D
  direction : Point2D

/-- A circle in 2D space -/
structure Circle2D where
  center : Point2D
  radius : ℝ

/-- The locus problem setup -/
structure LocusProblem where
  O : Point2D  -- vertex of right angle
  a : Ray2D    -- first leg of right angle
  b : Ray2D    -- second leg of right angle
  A : Point2D  -- point on ray a
  B : Point2D  -- point on ray b
  C : Point2D  -- point on circle with diameter AB
  A₀ : Point2D -- endpoint on ray a
  B₀ : Point2D -- endpoint on ray b
  constant_sum : ℝ -- constant value of AO + OB

/-- Define membership for Point2D in Circle2D -/
def Point2D.mem (p : Point2D) (c : Circle2D) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

instance : Membership Point2D Circle2D where
  mem := Point2D.mem

/-- The locus of points C is a semicircle -/
theorem locus_is_semicircle (problem : LocusProblem) : 
  ∃ (semicircle : Circle2D), 
    semicircle.center = Point2D.mk ((problem.A₀.x + problem.B₀.x) / 2) ((problem.A₀.y + problem.B₀.y) / 2) ∧
    semicircle.radius = (((problem.A₀.x - problem.B₀.x)^2 + (problem.A₀.y - problem.B₀.y)^2)^(1/2)) / 2 ∧
    problem.C ∈ semicircle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_semicircle_l234_23420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_representation_l234_23494

theorem vector_representation (a e₁ e₂ : ℝ × ℝ) : 
  a = (3, 2) → e₁ = (-1, 2) → e₂ = (5, -2) →
  ∃ (lambda mu : ℝ), a = lambda • e₁ + mu • e₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_representation_l234_23494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_when_a_neg_one_line_passes_through_point_l234_23431

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop :=
  (a^2 + a + 1) * x - y + 1 = 0

-- Statement A
theorem line_perpendicular_when_a_neg_one :
  ∀ x y x' y', line_l (-1) x y → line_l (-1) x' y' → 
  (x + y = 0 → (x' - x) * (y' - y) = -(x - x') * (y' - y)) :=
sorry

-- Statement C
theorem line_passes_through_point :
  ∀ a : ℝ, line_l a 0 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_when_a_neg_one_line_passes_through_point_l234_23431
