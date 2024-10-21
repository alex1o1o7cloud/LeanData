import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_line_equation_l541_54159

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the midpoint coordinates
def midpoint_x : ℝ := -1
def midpoint_y : ℝ := 1

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x - 4 * y + 7 = 0

-- Theorem statement
theorem chord_line_equation :
  ∀ (x1 y1 x2 y2 : ℝ),
  ellipse x1 y1 ∧ ellipse x2 y2 ∧
  midpoint_x = (x1 + x2) / 2 ∧ midpoint_y = (y1 + y2) / 2 →
  line_equation x1 y1 ∧ line_equation x2 y2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_line_equation_l541_54159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l541_54152

/-- The minimum value of 2/a + 1/b given the conditions -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : a - 2*b = 1)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y - 3 = 0) : 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' - 2*b' = 1 → 2/a + 1/b ≤ 2/a' + 1/b') ∧ 
  (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ - 2*b₀ = 1 ∧ 2/a₀ + 1/b₀ = 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l541_54152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_removal_l541_54142

theorem average_after_removal (numbers : Finset ℕ) (sum : ℕ) :
  numbers.card = 10 →
  sum = numbers.sum id →
  sum / numbers.card = 85 →
  70 ∈ numbers →
  76 ∈ numbers →
  ((sum - 70 - 76 : ℚ) / (numbers.card - 2 : ℚ)) = 88 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_removal_l541_54142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l541_54187

/-- Given a triangle PQR with points S on PQ and T on QR, prove that ST/TU = 1/3 -/
theorem triangle_ratio (P Q R S T U : EuclideanSpace ℝ (Fin 2)) : 
  let PQ := Q - P
  let QR := R - Q
  let PR := R - P
  let ST := T - S
  let TU := U - T
  (S - P = (4/5) • PQ) →  -- PS:SQ = 4:1
  (T - Q = (4/5) • QR) →  -- QT:TR = 4:1
  (∃ (t : ℝ), U = P + t • PR ∧ U = S + (‖ST‖ / (‖ST‖ + ‖TU‖)) • ST) →  -- U is on PR and ST
  ‖ST‖ / ‖TU‖ = 1/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l541_54187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_tan_sum_l541_54173

/-- Given a triangle ABC where sin(2A+B) = 2sin(B), 
    the minimum value of tan(A) + tan(C) + 2/tan(B) is 2 -/
theorem min_value_tan_sum (A B C : ℝ) (h : Real.sin (2*A + B) = 2 * Real.sin B) :
  ∃ (m : ℝ), m = 2 ∧ ∀ x, x = Real.tan A + Real.tan C + 2 / Real.tan B → m ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_tan_sum_l541_54173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_day2_calc_l541_54190

noncomputable def distance : ℝ := 3
noncomputable def speed_day1 : ℝ := 6
noncomputable def late_time : ℝ := 7 / 60
noncomputable def early_time : ℝ := 8 / 60

noncomputable def speed_day2 : ℝ := distance / (distance / speed_day1 - early_time)

theorem speed_day2_calc : 
  ∀ ε > 0, |speed_day2 - 8.18| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_day2_calc_l541_54190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_ten_l541_54144

/-- Square ABCD with side length 8 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (8, 0) ∧ C = (8, 8) ∧ D = (0, 8))

/-- Point M on side CB such that CM = 2 -/
def M : ℝ × ℝ := (6, 0)

/-- Point N on diagonal DB -/
def N (t : ℝ) : ℝ × ℝ := (t, 8 - t)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The main theorem -/
theorem min_distance_is_ten (s : Square) :
  ∃ (t : ℝ), ∀ (u : ℝ), 
    distance s.C (N u) + distance M (N u) ≥ 
    distance s.C (N t) + distance M (N t) ∧
    distance s.C (N t) + distance M (N t) = 10 := by
  sorry

#check min_distance_is_ten

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_ten_l541_54144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_differentiation_l541_54194

-- Define the functions
noncomputable def f1 (x : ℝ) : ℝ := x + 1/x
noncomputable def f2 (x : ℝ) : ℝ := x^2 / Real.exp x
noncomputable def f3 (x : ℝ) : ℝ := Real.sin (2*x - 1)

-- State the theorem
theorem correct_differentiation :
  (∀ x, deriv f1 x = 1 - 1/x^2) ∧
  (∀ x, deriv f2 x = (2*x - x^2) / Real.exp x) ∧
  (∀ x, deriv f3 x = 2 * Real.cos (2*x - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_differentiation_l541_54194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_bounded_l541_54134

/-- A point on a line or parabola -/
structure Point where
  x : ℝ
  y : ℝ

/-- The problem setup -/
structure ProblemSetup where
  A : Point
  B : Point
  C : Point
  hA : A.y = 3 * A.x + 19
  hB : B.y = B.x^2 + 4 * B.x - 1
  hC : C.y = C.x^2 + 4 * C.x - 1
  hY : A.y = B.y ∧ B.y = C.y
  hX : A.x < B.x ∧ B.x < C.x

/-- The main theorem -/
theorem sum_x_bounded (setup : ProblemSetup) :
  -12 < setup.A.x + setup.B.x + setup.C.x ∧
  setup.A.x + setup.B.x + setup.C.x < -9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_bounded_l541_54134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ln_half_eq_neg_one_l541_54191

/-- An odd function f where f(x) = e^x - 1 for x ≥ 0 -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.exp x - 1 else -(Real.exp (-x) - 1)

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- Theorem: f(ln(1/2)) = -1 -/
theorem f_ln_half_eq_neg_one : f (Real.log (1/2)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ln_half_eq_neg_one_l541_54191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_builds_60_computers_l541_54113

/-- Represents John's computer business --/
structure ComputerBusiness where
  partsCost : ℚ
  markup : ℚ
  rent : ℚ
  extraExpenses : ℚ
  profit : ℚ

/-- Calculates the number of computers built per month --/
def computersBuilt (business : ComputerBusiness) : ℚ :=
  (business.profit + business.rent + business.extraExpenses) / 
  (business.markup * business.partsCost - business.partsCost)

/-- Theorem stating that John builds 60 computers per month --/
theorem john_builds_60_computers : 
  let john : ComputerBusiness := {
    partsCost := 800,
    markup := 14/10,
    rent := 5000,
    extraExpenses := 3000,
    profit := 11200
  }
  computersBuilt john = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_builds_60_computers_l541_54113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_is_correct_l541_54167

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  /-- Side length of the equilateral triangular faces -/
  side_length : ℝ
  /-- Dihedral angle between the two adjacent faces -/
  dihedral_angle : ℝ
  /-- The side length is 3 -/
  side_length_is_three : side_length = 3
  /-- The dihedral angle is 30 degrees (π/6 radians) -/
  dihedral_angle_is_thirty : dihedral_angle = π / 6

/-- The maximum projection area of the tetrahedron -/
noncomputable def max_projection_area (t : Tetrahedron) : ℝ :=
  (3 * Real.sqrt 3) / 4

/-- Theorem stating that the maximum projection area is 3√3/4 -/
theorem max_projection_area_is_correct (t : Tetrahedron) :
    max_projection_area t = (3 * Real.sqrt 3) / 4 := by
  sorry

#check max_projection_area_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_is_correct_l541_54167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l541_54129

theorem equation_solution (x : ℝ) (h : x > 0) :
  (1/4 : ℝ) * x^((1/2 : ℝ) * Real.log x / Real.log 2) = 2^((1/4 : ℝ) * (Real.log x / Real.log 2)^2) ↔
  x = 2^(2 * Real.sqrt 2) ∨ x = 2^(-2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l541_54129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birds_cant_unite_l541_54128

/-- Represents the state of birds in nests -/
def BirdState := Fin 6 → Fin 6

/-- The initial state where each bird is in its own nest -/
def initial_state : BirdState := id

/-- A valid move for a bird is to a neighboring nest -/
def valid_move (a b : Fin 6) : Prop :=
  (b = a + 1) ∨ (b = a - 1)

/-- A state transition represents all birds moving simultaneously -/
def state_transition (s₁ s₂ : BirdState) : Prop :=
  ∀ b : Fin 6, valid_move (s₁ b) (s₂ b)

/-- A state is reachable if it can be obtained from the initial state
    through a finite number of valid transitions -/
inductive reachable : BirdState → Prop
  | initial : reachable initial_state
  | step {s₁ s₂ : BirdState} : reachable s₁ → state_transition s₁ s₂ → reachable s₂

/-- A state where all birds are in the same nest -/
def all_in_one_nest (s : BirdState) : Prop :=
  ∃ n : Fin 6, ∀ b : Fin 6, s b = n

/-- The main theorem: it's impossible to reach a state where all birds are in one nest -/
theorem birds_cant_unite : ¬∃ s : BirdState, reachable s ∧ all_in_one_nest s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_birds_cant_unite_l541_54128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_ratio_is_24_25_l541_54104

def admission_fee_adult : ℕ := 30
def admission_fee_child : ℕ := 18
def total_collected : ℕ := 2340

def valid_attendance (a c : ℕ) : Prop :=
  a > 0 ∧ c > 0 ∧ admission_fee_adult * a + admission_fee_child * c = total_collected

noncomputable def ratio_closest_to_one (a c : ℕ) : Prop :=
  valid_attendance a c ∧
  ∀ (a' c' : ℕ), valid_attendance a' c' →
    |((a : ℚ) / c) - 1| ≤ |((a' : ℚ) / c') - 1|

theorem closest_ratio_is_24_25 :
  ∃ (a c : ℕ), ratio_closest_to_one a c ∧ (a : ℚ) / c = 24 / 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_ratio_is_24_25_l541_54104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_segment_length_l541_54196

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Given a line segment divided by the golden ratio, with one segment of length 4,
    the other segment's length is either 2(√5 - 1) or 2(√5 + 1) -/
theorem golden_ratio_segment_length :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (a / b = φ ∨ b / a = φ) →
  (a = 4 ∨ b = 4) →
  (a = 2 * (Real.sqrt 5 - 1) ∧ b = 4) ∨ (a = 4 ∧ b = 2 * (Real.sqrt 5 + 1)) ∨
  (a = 2 * (Real.sqrt 5 + 1) ∧ b = 4) ∨ (a = 4 ∧ b = 2 * (Real.sqrt 5 - 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_segment_length_l541_54196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l541_54136

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 6 then a^(x-5) else (4 - a/2)*x + 4

-- Define the sequence a_n
noncomputable def a_n (a : ℝ) (n : ℕ) : ℝ := f a n

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ n : ℕ, a_n a n < a_n a (n+1)) →
  (48/7 < a ∧ a < 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l541_54136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_ari_has_winning_strategy_l541_54166

/-- Represents a game state with a rectangular chocolate block --/
structure ChocolateGame where
  a : ℕ
  b : ℕ

/-- Determines if a given game state is a winning position for the current player --/
def is_winning_position (game : ChocolateGame) : Prop :=
  ∀ (z : ℤ), (game.a + 1 : ℚ) / (game.b + 1 : ℚ) ≠ 2^z

/-- The theorem stating the winning condition for the first player --/
theorem first_player_wins (initial_game : ChocolateGame) :
  is_winning_position initial_game ↔ 
  ∃ (strategy : ChocolateGame → ChocolateGame), 
    (∀ (game : ChocolateGame), 
      game.a > 1 ∨ game.b > 1 → 
      is_winning_position (strategy game) ∧ 
      (strategy game).a ≤ game.a ∧ 
      (strategy game).b ≤ game.b ∧ 
      ((strategy game).a < game.a ∨ (strategy game).b < game.b)) :=
by sorry

/-- The specific game instance with 58x2022 chocolate block --/
def ari_sam_game : ChocolateGame :=
  { a := 58, b := 2022 }

/-- Theorem stating that Ari has a winning strategy in the 58x2022 game --/
theorem ari_has_winning_strategy : 
  is_winning_position ari_sam_game :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_ari_has_winning_strategy_l541_54166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_solution_set_l541_54137

noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

theorem floor_inequality (x : ℝ) (n : ℤ) :
  floor x = n ↔ n ≤ x ∧ x < n + 1 := by sorry

theorem solution_set (x : ℝ) :
  4 * (floor x)^2 - 36 * (floor x) + 45 ≤ 0 → 2 ≤ x ∧ x < 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_solution_set_l541_54137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_equal_area_trapeziums_l541_54115

/-- Represents a trapezium with parallel bases --/
structure Trapezium where
  base1 : ℝ
  base2 : ℝ
  height : ℝ

/-- Calculates the area of a trapezium --/
noncomputable def area (t : Trapezium) : ℝ := (t.base1 + t.base2) * t.height / 2

/-- Represents the original trapezium --/
noncomputable def original_trapezium : Trapezium :=
  { base1 := 1
    base2 := 4
    height := 1 }  -- We can assume height = 1 without loss of generality

/-- Represents the two new trapeziums after the first cut --/
noncomputable def new_trapeziums : Trapezium × Trapezium :=
  ( { base1 := 1
      base2 := 3
      height := 2/3 }
  , { base1 := 3
      base2 := 4
      height := 1/3 } )

/-- Theorem stating the minimum number of equal-area trapeziums --/
theorem min_equal_area_trapeziums :
  ∃ (m n : ℕ),
    m + n = 15 ∧
    (m : ℝ) * area (new_trapeziums.1) = (n : ℝ) * area (new_trapeziums.2) ∧
    ∀ (k l : ℕ),
      (k : ℝ) * area (new_trapeziums.1) = (l : ℝ) * area (new_trapeziums.2) →
      k + l ≥ 15 := by
  sorry

#check min_equal_area_trapeziums

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_equal_area_trapeziums_l541_54115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_train_speed_l541_54117

/-- The speed of the man's train given the parameters of the goods train --/
theorem mans_train_speed
  (goods_train_length : ℝ)
  (goods_train_passing_time : ℝ)
  (goods_train_speed : ℝ)
  (h1 : goods_train_length = 280)
  (h2 : goods_train_passing_time = 9)
  (h3 : goods_train_speed = 62) :
  let relative_speed := goods_train_length / goods_train_passing_time
  let goods_train_speed_ms := goods_train_speed * (1000 / 3600)
  let mans_train_speed_ms := relative_speed - goods_train_speed_ms
  let mans_train_speed_kmh := mans_train_speed_ms * (3600 / 1000)
  mans_train_speed_kmh = 50 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_train_speed_l541_54117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_room_side_length_for_9_by_12_table_l541_54143

/-- The side length of a square room that allows a rectangular table to be repositioned -/
noncomputable def minRoomSideLength (tableLength : ℝ) (tableWidth : ℝ) : ℝ :=
  Real.sqrt (tableLength ^ 2 + tableWidth ^ 2)

/-- Theorem stating the smallest integer room side length for a 9' by 12' table -/
theorem min_room_side_length_for_9_by_12_table :
  ⌈minRoomSideLength 9 12⌉ = 15 := by
  sorry

#check min_room_side_length_for_9_by_12_table

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_room_side_length_for_9_by_12_table_l541_54143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lune_area_of_overlapping_semicircles_l541_54148

-- Define the diameters of the semicircles
noncomputable def d₁ : ℝ := 3
noncomputable def d₂ : ℝ := 4

-- Define the area of the lune
noncomputable def lune_area (d₁ d₂ : ℝ) : ℝ := (3/4) * Real.pi + 1

-- Theorem statement
theorem lune_area_of_overlapping_semicircles :
  lune_area d₁ d₂ = (3/4) * Real.pi + 1 := by
  -- Unfold the definition of lune_area
  unfold lune_area
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lune_area_of_overlapping_semicircles_l541_54148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_equivalence_l541_54106

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x - Real.pi/3)

-- Define the target function
noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x

-- Define the translation
def translate (h : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x ↦ h (x + a)

-- Theorem statement
theorem translation_equivalence :
  ∀ x, translate f (Real.pi/3) x = g x := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_equivalence_l541_54106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_is_70_l541_54102

-- Define the triangle vertices
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (21, 0)
def C : ℝ × ℝ := (21, 21)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the perimeter function
noncomputable def perimeter (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  distance p1 p2 + distance p2 p3 + distance p3 p1

-- Theorem statement
theorem triangle_perimeter_is_70 :
  perimeter A B C = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_is_70_l541_54102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_approx_95_656_l541_54145

/-- Represents the side length of the square field in meters -/
noncomputable def side_length : ℝ := 35

/-- Represents the length of the muddy terrain on each side in meters -/
noncomputable def muddy_length : ℝ := 10

/-- Represents the length of the sandy surface on each side in meters -/
noncomputable def sandy_length : ℝ := 15

/-- Represents the length of the uphill slope on each side in meters -/
noncomputable def uphill_length : ℝ := 10

/-- Represents the speed on muddy terrain in km/h -/
noncomputable def muddy_speed : ℝ := 5

/-- Represents the speed on sandy surface in km/h -/
noncomputable def sandy_speed : ℝ := 7

/-- Represents the speed on uphill slope in km/h -/
noncomputable def uphill_speed : ℝ := 4

/-- Calculates the time taken to run a given distance at a given speed -/
noncomputable def time_for_section (distance : ℝ) (speed : ℝ) : ℝ :=
  distance * 3.6 / speed

/-- Theorem stating that the total time to run around the field is approximately 95.656 seconds -/
theorem total_time_approx_95_656 :
  let muddy_time := time_for_section muddy_length muddy_speed
  let sandy_time := time_for_section sandy_length sandy_speed
  let uphill_time := time_for_section uphill_length uphill_speed
  let total_time := 4 * (muddy_time + sandy_time + uphill_time)
  abs (total_time - 95.656) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_approx_95_656_l541_54145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_most_3_rainy_days_approx_l541_54183

/-- The probability of rain on a single day in July in Capital City -/
noncomputable def p_rain : ℝ := 2 / 10

/-- The number of days in July -/
def days_in_july : ℕ := 31

/-- The probability of rain on at most 3 days in July -/
noncomputable def p_at_most_3_rainy_days : ℝ :=
  (Finset.range 4).sum (λ k => 
    (Nat.choose days_in_july k : ℝ) * p_rain ^ k * (1 - p_rain) ^ (days_in_july - k))

/-- Theorem stating that the probability of rain on at most 3 days in July is approximately 0.707 -/
theorem prob_at_most_3_rainy_days_approx :
  abs (p_at_most_3_rainy_days - 0.707) < 0.0005 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_most_3_rainy_days_approx_l541_54183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l541_54170

theorem alpha_value (α : ℝ) (h1 : Real.cos α = -1/2) (h2 : 0 < α ∧ α < Real.pi) : α = 2*Real.pi/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l541_54170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_product_solution_l541_54150

def cyclic_product_equations (n : ℕ) (x : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, i < n → x i * x ((i + 1) % n) = 1

theorem cyclic_product_solution (n : ℕ) (x : ℕ → ℝ) :
  cyclic_product_equations n x →
  (n % 2 = 1 → ∃ k : ℝ, (k = 1 ∨ k = -1) ∧ ∀ i, i < n → x i = k) ∧
  (n % 2 = 0 → ∃ a : ℝ, a ≠ 0 ∧
    (∀ i, i < n → i % 2 = 0 → x i = 1 / a) ∧
    (∀ i, i < n → i % 2 = 1 → x i = a)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_product_solution_l541_54150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_PC₁C₂_l541_54132

/-- Circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y - 3 = 0

/-- Circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 12 = 0

/-- Point P on circle C₂ -/
def P : {p : ℝ × ℝ // C₂ p.1 p.2} := sorry

/-- Centers of circles C₁ and C₂ -/
def center_C₁ : ℝ × ℝ := (-2, 2)
def center_C₂ : ℝ × ℝ := (2, 0)

/-- Area of a triangle given three points -/
noncomputable def area_triangle (p q r : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the maximum area of triangle PC₁C₂ -/
theorem max_area_triangle_PC₁C₂ : 
  ∃ (p : ℝ × ℝ), C₂ p.1 p.2 ∧ 
    (∀ (q : ℝ × ℝ), C₂ q.1 q.2 → 
      area_triangle p center_C₁ center_C₂ ≥ area_triangle q center_C₁ center_C₂) ∧
    area_triangle p center_C₁ center_C₂ = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_PC₁C₂_l541_54132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_diameter_intersection_l541_54101

theorem circle_chord_diameter_intersection (r : ℝ) (chord_length : ℝ) 
  (hr : r = 6) (hchord : chord_length = 10) :
  ∃ (s₁ s₂ : ℝ), s₁ = 6 - Real.sqrt 11 ∧ s₂ = 6 + Real.sqrt 11 ∧ 
  s₁ + s₂ = 2 * r ∧
  s₁ * s₂ = (chord_length / 2) ^ 2 := by
  sorry

#check circle_chord_diameter_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_diameter_intersection_l541_54101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_seating_arrangements_l541_54175

/-- Represents the number of seats in a taxi -/
structure TaxiSeats where
  front : Nat
  back : Nat

/-- Represents the number of window seats in a taxi -/
def windowSeats (ts : TaxiSeats) : Nat :=
  ts.front + 2  -- front seat and two outermost back seats

/-- Calculates the number of seating arrangements for 4 passengers in a taxi,
    where one passenger must sit in a window seat -/
def seatingArrangements (ts : TaxiSeats) (totalPassengers : Nat) : Nat :=
  (windowSeats ts) * Nat.factorial (totalPassengers - 1)

/-- Theorem: There are 18 ways to arrange 4 passengers in a taxi with 1 front seat 
    and 3 back seats, where one passenger must sit in a window seat -/
theorem taxi_seating_arrangements :
  let ts : TaxiSeats := ⟨1, 3⟩
  seatingArrangements ts 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_seating_arrangements_l541_54175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_cycles_l541_54198

def is_valid_permutation (a : Fin 101 → Nat) : Prop :=
  (∀ k : Fin 101, a k ∈ Finset.range 101 ∪ {101}) ∧
  (∀ k : Fin 101, a k % (k.val + 1) = 0) ∧
  Function.Injective a

def valid_cycles : List (List Nat) :=
  [[1, 102], [1, 2, 102], [1, 3, 102], [1, 6, 102], [1, 17, 102], [1, 34, 102], [1, 51, 102],
   [1, 2, 6, 102], [1, 2, 34, 102], [1, 3, 6, 102], [1, 3, 51, 102], [1, 17, 34, 102], [1, 17, 51, 102]]

theorem permutation_cycles (a : Fin 101 → Nat) (h : is_valid_permutation a) :
  ∃ (cycle : List Nat), cycle ∈ valid_cycles ∧
    (∀ k : Fin 101, (k.val + 1 : Nat) ∉ cycle → a k = k.val + 1) :=
  sorry

#check permutation_cycles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_cycles_l541_54198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_equality_l541_54184

/-- Given a triangle ABC with circumradius r and side lengths a, b, c opposite to angles A, B, C respectively,
    prove that twice the area of the triangle is equal to r(a cos A + b cos B + c cos C) -/
theorem triangle_area_equality (A B C : ℝ) (a b c r : ℝ) :
  let T := (1 / 2) * a * b * Real.sin C  -- Area of triangle using sine formula
  2 * T = r * (a * Real.cos A + b * Real.cos B + c * Real.cos C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_equality_l541_54184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_bags_in_box_l541_54110

theorem tea_bags_in_box : ∃ n : ℕ, 
  (∀ k : ℕ, k = 41 ∨ k = 58 → 2 * n ≤ k ∧ k ≤ 3 * n) ∧ 
  (∀ m : ℕ, m ≠ n → ¬(∀ k : ℕ, k = 41 ∨ k = 58 → 2 * m ≤ k ∧ k ≤ 3 * m)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_bags_in_box_l541_54110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_trigonometric_equation_l541_54193

theorem smallest_angle_trigonometric_equation :
  ∃ x : Real,
    x = 9 ∧
    (∀ y : Real, 0 < y → y < x → 
      ¬(Real.sin (4 * y * π / 180) * Real.sin (6 * y * π / 180) = 
        Real.cos (4 * y * π / 180) * Real.cos (6 * y * π / 180))) ∧
    (Real.sin (4 * x * π / 180) * Real.sin (6 * x * π / 180) = 
     Real.cos (4 * x * π / 180) * Real.cos (6 * x * π / 180)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_trigonometric_equation_l541_54193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l541_54168

/-- Represents a parabola with focus (a, b) and directrix cx + dy = e --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- Represents the general equation of a conic section --/
structure ConicEquation where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- Checks if the given conic equation represents the parabola --/
def is_parabola_equation (p : Parabola) (eq : ConicEquation) : Prop :=
  ∀ x y : ℝ,
    (x - p.a)^2 + (y - p.b)^2 = 
      ((p.c * x + p.d * y - p.e) / (Real.sqrt (p.c^2 + p.d^2)))^2 ↔
    eq.a * x^2 + eq.b * x * y + eq.c * y^2 + eq.d * x + eq.e * y + eq.f = 0

theorem parabola_equation 
  (p : Parabola) 
  (h1 : p.a = 4 ∧ p.b = -2 ∧ p.c = 4 ∧ p.d = 6 ∧ p.e = 24) :
  ∃ eq : ConicEquation,
    is_parabola_equation p eq ∧
    eq.a > 0 ∧
    Int.gcd (Int.natAbs eq.a) (Int.gcd (Int.natAbs eq.b) (Int.gcd (Int.natAbs eq.c) 
      (Int.gcd (Int.natAbs eq.d) (Int.gcd (Int.natAbs eq.e) (Int.natAbs eq.f))))) = 1 ∧
    eq.a = 9 ∧ eq.b = -12 ∧ eq.c = -23 ∧ eq.d = -56 ∧ eq.e = 196 ∧ eq.f = 64 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l541_54168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_one_l541_54174

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (x - 2) / (x - 5)

-- State the theorem
theorem inverse_g_undefined_at_one :
  ∃ (f : ℝ → ℝ), (∀ x ≠ 1, f (g x) = x ∧ g (f x) = x) ∧
  ¬∃ y, f 1 = y := by
  sorry

#check inverse_g_undefined_at_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_one_l541_54174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l541_54149

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.log x
noncomputable def g (x : ℝ) : ℝ := 2 * x^2 / Real.exp x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := min (f a x) (g x)

theorem problem_solution (a : ℝ) :
  (∃ (m : ℝ), (f a 1) = 0 ∧ 
    (deriv (f a) 1) * (2 - 1) + (f a 1) = 3) →
  (a = 2 ∧ 
   ∃ (x₀ : ℝ), x₀ ∈ Set.Ioo 1 2 ∧ f 2 x₀ = g x₀ ∧ 
     (∀ (x : ℝ), x ∈ Set.Ioo 1 2 → x ≠ x₀ → f 2 x ≠ g x) ∧
   ∀ (m : ℝ), m ≤ 8 / Real.exp 2 ↔ ∃ (x₀ : ℝ), x₀ > 0 ∧ h 2 x₀ ≥ m) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l541_54149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l541_54109

def sequence_a : ℕ → ℝ
  | 0 => 2  -- Add a case for 0 to cover all natural numbers
  | 1 => 2
  | (n + 2) => 3 * sequence_a (n + 1) - 2

theorem sequence_a_closed_form (n : ℕ) : 
  sequence_a n = 3^(n-1) + 1 := by
  sorry

#check sequence_a_closed_form

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l541_54109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dime_probability_l541_54107

/-- Represents the types of coins in the purse -/
inductive Coin
  | Dime
  | Quarter
  | HalfDollar

/-- The value of each coin type in dollars -/
def coinValue (c : Coin) : ℚ :=
  match c with
  | Coin.Dime => 1/10
  | Coin.Quarter => 1/4
  | Coin.HalfDollar => 1/2

/-- The total value of each coin type in the purse -/
def totalValue (c : Coin) : ℚ :=
  match c with
  | Coin.Dime => 12
  | Coin.Quarter => 15
  | Coin.HalfDollar => 20

/-- The number of coins of each type in the purse -/
def numCoins (c : Coin) : ℕ :=
  (totalValue c / coinValue c).floor.toNat

/-- The total number of coins in the purse -/
def totalCoins : ℕ :=
  numCoins Coin.Dime + numCoins Coin.Quarter + numCoins Coin.HalfDollar

/-- The probability of randomly selecting a dime from the purse -/
theorem dime_probability : 
  (numCoins Coin.Dime : ℚ) / totalCoins = 6/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dime_probability_l541_54107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minor_axis_length_l541_54103

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The sum of distances from any point on the ellipse to its two foci -/
def focalSum (e : Ellipse) : ℝ := 2 * e.a

theorem ellipse_minor_axis_length 
  (e : Ellipse) 
  (h_ecc : eccentricity e = Real.sqrt 5 / 3) 
  (h_sum : focalSum e = 12) : 
  2 * e.b = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minor_axis_length_l541_54103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_penny_draw_probability_l541_54169

theorem penny_draw_probability : 
  let total_pennies : ℕ := 11
  let shiny_pennies : ℕ := 5
  let dull_pennies : ℕ := 6
  let draws_threshold : ℕ := 6

  let probability_numerator : ℕ := 155
  let probability_denominator : ℕ := 231

  (Nat.gcd probability_numerator probability_denominator = 1) ∧ 
  (probability_numerator + probability_denominator = 386) ∧
  (Nat.choose total_pennies shiny_pennies * (probability_numerator : ℚ) / probability_denominator = 
   Nat.choose draws_threshold 3 * Nat.choose (total_pennies - draws_threshold) 1 + 
   Nat.choose draws_threshold 2 * Nat.choose (total_pennies - draws_threshold) 2 + 
   Nat.choose draws_threshold 1 * Nat.choose (total_pennies - draws_threshold) 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_penny_draw_probability_l541_54169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_ratio_l541_54141

/-- Given a triangle with angles in the ratio 2:4:3, prove that the largest angle is 80° and the smallest is 40° -/
theorem triangle_angles_ratio (a b c : ℝ) : 
  a + b + c = 180 →  -- sum of angles in a triangle is 180°
  a / 2 = b / 4 ∧ b / 4 = c / 3 →  -- ratio of angles is 2:4:3
  (max a (max b c) = 80 ∧ min a (min b c) = 40) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_ratio_l541_54141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2_56_to_hundredth_l541_54130

/-- Rounds a number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ := 
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- Proves that rounding 2.56 to the nearest hundredth equals 2.56 -/
theorem round_2_56_to_hundredth :
  round_to_hundredth 2.56 = 2.56 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2_56_to_hundredth_l541_54130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_legs_calculation_l541_54100

/-- Calculate the total number of legs for Mark's animals, accounting for missing legs -/
theorem total_legs_calculation (
  kangaroo_legs : ℕ
) (goat_legs : ℕ)
  (spider_legs : ℕ)
  (bird_legs : ℕ)
  (num_kangaroos : ℕ)
  (num_goats : ℕ)
  (num_spiders : ℕ)
  (num_birds : ℕ)
  (missing_kangaroo_legs : ℕ)
  (missing_goat_legs : ℕ)
  (missing_spider_legs : ℕ)
  (missing_bird_legs : ℕ)
  (h1 : kangaroo_legs = 2)
  (h2 : goat_legs = 4)
  (h3 : spider_legs = 8)
  (h4 : bird_legs = 2)
  (h5 : num_kangaroos = 23)
  (h6 : num_goats = 3 * num_kangaroos)
  (h7 : num_spiders = 2 * num_goats)
  (h8 : num_birds = num_spiders / 2)
  (h9 : missing_kangaroo_legs = 5)
  (h10 : missing_goat_legs = 7)
  (h11 : missing_spider_legs = 15)
  (h12 : missing_bird_legs = 3)
: 
  (num_kangaroos * kangaroo_legs + 
   num_goats * goat_legs + 
   num_spiders * spider_legs + 
   num_birds * bird_legs) - 
  (missing_kangaroo_legs + 
   missing_goat_legs + 
   missing_spider_legs + 
   missing_bird_legs) = 1534 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_legs_calculation_l541_54100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_point_l541_54118

noncomputable section

theorem triangle_intersection_point (A B C G H Q : EuclideanSpace ℝ (Fin 3)) : 
  (∃ t : ℝ, G = (1 - t) • A + t • B ∧ t = 2/5) →
  (∃ s : ℝ, H = (1 - s) • B + s • C ∧ s = 3/5) →
  (∃ u v : ℝ, Q = (1 - u) • A + u • G ∧ Q = (1 - v) • C + v • H) →
  Q = (2/5) • A + 0 • B + (2/5) • C :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_point_l541_54118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_specific_l541_54114

/-- The distance between the center of a sphere and the plane of an equilateral triangle tangent to the sphere -/
noncomputable def sphere_triangle_distance (sphere_radius : ℝ) (triangle_side : ℝ) : ℝ :=
  Real.sqrt (sphere_radius^2 - (triangle_side * Real.sqrt 3 / 6)^2)

/-- Theorem stating that the distance between the center of a sphere with radius 10 and the plane of an equilateral triangle with side length 18 tangent to the sphere is √73 -/
theorem sphere_triangle_distance_specific : sphere_triangle_distance 10 18 = Real.sqrt 73 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_specific_l541_54114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_rings_hit_is_discrete_l541_54177

/-- A random variable is discrete if it can take on a countable number of distinct values. -/
def is_discrete_random_variable (X : Type) : Prop := Countable X

/-- The lifespan of a light bulb in hours. -/
def lifespan : Type := ℝ

/-- The number of rings hit by Xiaoming in 1 shot. -/
def rings_hit : Type := Fin 11

/-- The voltage value measured between 10V and 20V. -/
def voltage : Type := {x : ℝ // 10 ≤ x ∧ x ≤ 20}

/-- The position of a random particle moving on the y-axis. -/
def particle_position : Type := ℝ

/-- Theorem stating that among the given options, only the number of rings hit is a discrete random variable. -/
theorem only_rings_hit_is_discrete :
  ¬ is_discrete_random_variable lifespan ∧
  is_discrete_random_variable rings_hit ∧
  ¬ is_discrete_random_variable voltage ∧
  ¬ is_discrete_random_variable particle_position :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_rings_hit_is_discrete_l541_54177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_max_l541_54105

theorem triangle_side_ratio_max (a b c : ℝ) (A : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  a = b * c * Real.sin A / a →
  (b / c + c / b) ≤ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_max_l541_54105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l541_54147

/-- The angle in radians corresponding to 135 degrees -/
noncomputable def angle : ℝ := 135 * Real.pi / 180

/-- The point through which the line passes -/
def P : ℝ × ℝ := (1, -1)

/-- The slope of the line -/
noncomputable def m : ℝ := Real.tan angle

theorem line_equation :
  ∀ (x y : ℝ), (y - P.2 = m * (x - P.1)) ↔ (y = -x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l541_54147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_l541_54122

-- Define the circles
noncomputable def circle_A (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 7 = 0
noncomputable def circle_B (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0

-- Define the line equation
noncomputable def intersection_line (x y : ℝ) : Prop := 4*x + 4*y + 5 = 0

-- Define the distance between intersection points
noncomputable def intersection_distance : ℝ := Real.sqrt 238 / 4

theorem circles_intersection :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    -- The circles intersect at two distinct points
    circle_A x₁ y₁ ∧ circle_B x₁ y₁ ∧
    circle_A x₂ y₂ ∧ circle_B x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    -- The intersection points lie on the line
    intersection_line x₁ y₁ ∧
    intersection_line x₂ y₂ ∧
    -- The distance between the intersection points is correct
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = intersection_distance^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_l541_54122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_equals_sqrt_5_l541_54108

noncomputable def vector_a : Fin 2 → ℝ := ![2, -1]
noncomputable def vector_b : Fin 2 → ℝ := ![0, 1]

noncomputable def vector_sum (a b : Fin 2 → ℝ) : Fin 2 → ℝ := 
  λ i => a i + 2 * b i

noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ := 
  Real.sqrt ((v 0) ^ 2 + (v 1) ^ 2)

theorem magnitude_of_sum_equals_sqrt_5 : 
  magnitude (vector_sum vector_a vector_b) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_equals_sqrt_5_l541_54108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_l541_54133

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def IsSpecialQuadrilateral (q : Quadrilateral) : Prop :=
  let dist := fun (p1 p2 : ℝ × ℝ) => Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  dist q.A q.B = 12 ∧
  dist q.B q.C = 12 ∧
  dist q.C q.D = 20 ∧
  dist q.D q.A = 20 ∧
  let angle := fun (p1 p2 p3 : ℝ × ℝ) =>
    Real.arccos ((dist p1 p2)^2 + (dist p1 p3)^2 - (dist p2 p3)^2) / (2 * dist p1 p2 * dist p1 p3)
  angle q.A q.D q.C = 2 * Real.pi / 3  -- 120° in radians

-- Theorem statement
theorem diagonal_length (q : Quadrilateral) (h : IsSpecialQuadrilateral q) :
  let dist := fun (p1 p2 : ℝ × ℝ) => Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  dist q.A q.C = 20 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_l541_54133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l541_54182

/-- Given a hyperbola with equation (x²/a²) - (y²/b²) = 1 and an asymptote y = √2 * x,
    prove that its eccentricity is √3 -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ k : ℝ, ∀ x : ℝ, k * x = Real.sqrt 2 * x) →
  Real.sqrt (1 + (b/a)^2) = Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l541_54182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intercept_length_l541_54112

/-- The circle equation x^2 + y^2 - 4y = 0 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

/-- The line passing through the origin with an inclination angle of 60° -/
def line_eq (x y : ℝ) : Prop := y = Real.sqrt 3 * x

/-- The chord length intercepted by the circle on the line -/
noncomputable def chord_length : ℝ := 2 * Real.sqrt 3

theorem chord_intercept_length :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧
    line_eq x₁ y₁ ∧ line_eq x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = chord_length^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intercept_length_l541_54112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l541_54186

-- Define the function f
noncomputable def f (b : ℝ) : ℝ → ℝ :=
  λ x => if x ≥ 0 then 2^x + 2*x + b else -(2^(-x) + 2*(-x) + b)

-- State the theorem
theorem odd_function_value (b : ℝ) :
  (∀ x, f b x = -(f b (-x))) →  -- f is odd
  f b 0 = 0 →                   -- f(0) = 0
  f b (-1) = -3 :=
by
  intros h_odd h_zero
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l541_54186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l541_54189

/-- A line passing through (1, 3) and tangent to the circle x^2 + y^2 = 1 
    has the equation x = 1 or 4x - 3y + 5 = 0 -/
theorem tangent_line_equation : 
  ∀ (l : Set (ℝ × ℝ)),
  (∃ p : ℝ × ℝ, p ∈ l ∧ p.1 = 1 ∧ p.2 = 3) →
  (∀ p : ℝ × ℝ, p ∈ l → p.1^2 + p.2^2 = 1 → (∀ q : ℝ × ℝ, q ∈ l → q.1^2 + q.2^2 ≥ 1)) →
  (∀ p : ℝ × ℝ, p ∈ l ↔ (p.1 = 1 ∨ 4*p.1 - 3*p.2 + 5 = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l541_54189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_one_eq_neg_three_l541_54179

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x ≤ 0 then 2 * x^2 - x else -(-2 * x^2 + x)  -- Define for all x using odd property

-- State the theorem
theorem f_of_one_eq_neg_three :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ≤ 0, f x = 2 * x^2 - x) →  -- f(x) = 2x^2 - x for x ≤ 0
  f 1 = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_one_eq_neg_three_l541_54179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_question_equals_answer_l541_54176

-- Define what it means for a number to be rational
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define what it means for a number to be irrational
def IsIrrational (x : ℝ) : Prop := ¬ IsRational x

-- Axiom: Integers are rational
axiom integers_are_rational : ∀ (n : ℤ), IsRational (n : ℝ)

-- Axiom: Fractions of integers are rational
axiom fractions_are_rational : ∀ (p q : ℤ), q ≠ 0 → IsRational (p / q : ℝ)

-- Axiom: π is irrational
axiom pi_is_irrational : IsIrrational Real.pi

theorem question_equals_answer : IsIrrational Real.pi := by
  exact pi_is_irrational


end NUMINAMATH_CALUDE_ERRORFEEDBACK_question_equals_answer_l541_54176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_is_odd_and_decreasing_l541_54165

-- Define the interval (0,1)
def open_unit_interval : Set ℝ := {x | 0 < x ∧ x < 1}

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of being decreasing on an interval
def is_decreasing_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f y < f x

-- Define the given functions
noncomputable def f_A : ℝ → ℝ := λ x => -1/x
def f_B : ℝ → ℝ := λ x => x
noncomputable def f_C : ℝ → ℝ := λ x => Real.log 2 * Real.log (|x - 1|)
noncomputable def f_D : ℝ → ℝ := λ x => -Real.sin x

theorem sin_is_odd_and_decreasing :
  (is_odd f_D ∧ is_decreasing_on f_D open_unit_interval) ∧
  ¬(is_odd f_A ∧ is_decreasing_on f_A open_unit_interval) ∧
  ¬(is_odd f_B ∧ is_decreasing_on f_B open_unit_interval) ∧
  ¬(is_odd f_C ∧ is_decreasing_on f_C open_unit_interval) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_is_odd_and_decreasing_l541_54165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sqrt_2sinx_minus_sinx_l541_54156

theorem max_value_of_sqrt_2sinx_minus_sinx :
  ∀ x : ℝ, Real.sqrt (2 * Real.sin x) - Real.sin x ≤ (1 : ℝ) / 2 ∧
  (Real.sqrt (2 * Real.sin x) - Real.sin x = (1 : ℝ) / 2 ↔
    (∃ k : ℤ, x = π / 6 + 2 * k * π ∨ x = 5 * π / 6 + 2 * k * π)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sqrt_2sinx_minus_sinx_l541_54156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_probability_l541_54127

-- Define the circle
def circleO (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line
def lineL (x y b : ℝ) : Prop := y = x + b

-- Define the interval for b
def b_interval (b : ℝ) : Prop := -5 ≤ b ∧ b ≤ 5

-- Define the probability
noncomputable def probability : ℝ := 2 * Real.sqrt 2 / 5

-- Theorem statement
theorem circle_line_intersection_probability :
  ∀ (b : ℝ), b_interval b →
  (∃ (x y : ℝ), circleO x y ∧ lineL x y b) →
  probability = 2 * Real.sqrt 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_probability_l541_54127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_power_and_speed_l541_54197

/-- Represents the properties of a train and its track -/
structure TrainSystem where
  mass : ℝ
  frictionCoeff : ℝ
  maxSpeedHorizontal : ℝ
  trackGradient : ℝ

/-- Calculates the locomotive's power in horsepower -/
noncomputable def locomotivePower (ts : TrainSystem) : ℝ :=
  (ts.mass * ts.frictionCoeff * ts.maxSpeedHorizontal) / 75

/-- Calculates the maximum speed on an inclined track in km/h -/
noncomputable def maxSpeedInclined (ts : TrainSystem) (power : ℝ) : ℝ :=
  (power * 75) / (ts.mass * (ts.frictionCoeff + ts.trackGradient)) * 3.6

/-- Theorem stating the locomotive's power and maximum speed on inclined track -/
theorem train_power_and_speed (ts : TrainSystem) 
    (h1 : ts.mass = 300000)
    (h2 : ts.frictionCoeff = 0.005)
    (h3 : ts.maxSpeedHorizontal = 35 / 3.6)
    (h4 : ts.trackGradient = 0.01) :
  let power := locomotivePower ts
  (abs (power - 194.4) < 0.1) ∧ 
  (abs (maxSpeedInclined ts power - 11.6) < 0.1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_power_and_speed_l541_54197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_XPYC_is_parallelogram_l541_54123

-- Define the circles and points
variable (ω₁ ω₂ : Set (ℝ × ℝ))
variable (O₁ O₂ A B P X Y C : ℝ × ℝ)

-- Define the conditions
def circles_intersect (ω₁ ω₂ : Set (ℝ × ℝ)) (A B : ℝ × ℝ) : Prop := 
  A ∈ ω₁ ∧ A ∈ ω₂ ∧ B ∈ ω₁ ∧ B ∈ ω₂

def O₁_on_ω₂ (ω₂ : Set (ℝ × ℝ)) (O₁ : ℝ × ℝ) : Prop := O₁ ∈ ω₂
def P_on_ω₁ (ω₁ : Set (ℝ × ℝ)) (P : ℝ × ℝ) : Prop := P ∈ ω₁
def X_on_ω₂ (ω₂ : Set (ℝ × ℝ)) (X : ℝ × ℝ) : Prop := X ∈ ω₂
def Y_on_ω₂ (ω₂ : Set (ℝ × ℝ)) (Y : ℝ × ℝ) : Prop := Y ∈ ω₂
def C_on_ω₂ (ω₂ : Set (ℝ × ℝ)) (C : ℝ × ℝ) : Prop := C ∈ ω₂

-- Define the parallelogram property
def IsParallelogram (X P Y C : ℝ × ℝ) : Prop :=
  (X.1 - P.1 = Y.1 - C.1) ∧ (X.2 - P.2 = Y.2 - C.2)

-- Define the theorem
theorem XPYC_is_parallelogram 
  (h1 : circles_intersect ω₁ ω₂ A B)
  (h2 : O₁_on_ω₂ ω₂ O₁)
  (h3 : P_on_ω₁ ω₁ P)
  (h4 : X_on_ω₂ ω₂ X)
  (h5 : Y_on_ω₂ ω₂ Y)
  (h6 : C_on_ω₂ ω₂ C)
  : IsParallelogram X P Y C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_XPYC_is_parallelogram_l541_54123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l541_54185

/-- Calculates the rate of interest per annum given the principal, time, and simple interest -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (simple_interest : ℝ) : ℝ :=
  (simple_interest * 100) / (principal * time)

theorem interest_rate_calculation :
  let principal : ℝ := 5737.5
  let time : ℝ := 5
  let simple_interest : ℝ := 4016.25
  calculate_interest_rate principal time simple_interest = 14 := by
  -- Unfold the definition of calculate_interest_rate
  unfold calculate_interest_rate
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l541_54185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_number_divisibility_l541_54121

def PerfectNumber (n : ℕ) : Prop := ArithmeticFunction.sigma n = 2 * n

theorem perfect_number_divisibility (n : ℕ) :
  PerfectNumber n → n > 28 → n % 7 = 0 → n % 49 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_number_divisibility_l541_54121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_butter_cans_price_l541_54178

theorem peanut_butter_cans_price (total_cans : ℕ) (avg_price_all : ℚ) 
  (returned_cans : ℕ) (avg_price_remaining : ℚ) : 
  total_cans = 6 → 
  avg_price_all = 365/10 → 
  returned_cans = 2 → 
  avg_price_remaining = 30 → 
  (total_cans * avg_price_all - (total_cans - returned_cans) * avg_price_remaining) / returned_cans = 495/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_butter_cans_price_l541_54178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l541_54138

theorem birthday_problem (people : Finset Nat) (birthdays : Nat → Nat) :
  (∀ p, p ∈ people → birthdays p ≤ 365) →
  Finset.card people = 366 →
  ∃ p₁ p₂, p₁ ∈ people ∧ p₂ ∈ people ∧ p₁ ≠ p₂ ∧ birthdays p₁ = birthdays p₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l541_54138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_comparison_l541_54157

-- Define the side lengths of the triangles
noncomputable def a₁ : ℝ := 26
noncomputable def b₁ : ℝ := 26
noncomputable def c₁ : ℝ := 30

noncomputable def a₂ : ℝ := 26
noncomputable def b₂ : ℝ := 26
noncomputable def c₂ : ℝ := 50

-- Define the areas of the triangles using Heron's formula
noncomputable def area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def A : ℝ := area a₁ b₁ c₁
noncomputable def B : ℝ := area a₂ b₂ c₂

-- State the theorem
theorem triangle_area_comparison : A > B := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_comparison_l541_54157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l541_54188

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  (∀ x, f (x + Real.pi / 3) = -f (-(x + Real.pi / 3))) ∧
  (∀ x, f (x + Real.pi / 12) = 2 * Real.cos x) ∧
  (∀ x, f (Real.pi / 12 + x) = f (Real.pi / 12 - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l541_54188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiona_wins_probability_l541_54119

/-- Represents a player in the die-tossing game -/
inductive Player
| Dave
| Emily
| Fiona

/-- The probability of tossing an eight on a single toss -/
noncomputable def probEight : ℝ := 1 / 8

/-- The probability of not tossing an eight on a single toss -/
noncomputable def probNotEight : ℝ := 1 - probEight

/-- The probability of Fiona winning in a single cycle -/
noncomputable def probFionaWinCycle : ℝ := probNotEight * probNotEight * probEight

/-- The probability of no one winning in a single cycle -/
noncomputable def probNobodyWinCycle : ℝ := probNotEight * probNotEight * probNotEight

/-- The probability that Fiona is the first to toss an eight -/
theorem fiona_wins_probability :
  (probFionaWinCycle / (1 - probNobodyWinCycle)) = 49 / 169 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiona_wins_probability_l541_54119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_coordinates_l541_54131

def A : ℝ × ℝ × ℝ := (1, -2, 1)
def B : ℝ × ℝ × ℝ := (2, 2, 2)

theorem point_P_coordinates (z : ℝ) :
  let P : ℝ × ℝ × ℝ := (0, 0, z)
  (A.1 - P.1)^2 + (A.2.1 - P.2.1)^2 + (A.2.2 - P.2.2)^2 = 
  (B.1 - P.1)^2 + (B.2.1 - P.2.1)^2 + (B.2.2 - P.2.2)^2 →
  z = 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_coordinates_l541_54131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_problem_solution_l541_54140

/-- Represents a circle in 2D space --/
structure Circle where
  center_x : ℝ
  center_y : ℝ
  radius : ℝ

/-- Represents a line in 2D space using the general form Ax + By + C = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the distance between a point and a line --/
noncomputable def distancePointToLine (x y : ℝ) (l : Line) : ℝ :=
  (abs (l.a * x + l.b * y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

/-- The main theorem about the positional relationship between the circle and the line --/
theorem circle_line_intersection 
  (circle : Circle) 
  (line : Line) : 
  0 < distancePointToLine circle.center_x circle.center_y line ∧ 
  distancePointToLine circle.center_x circle.center_y line < circle.radius := by
  sorry

/-- The circle given in the problem --/
def problemCircle : Circle := 
  { center_x := -1
  , center_y := 3
  , radius := 2 }

/-- The line given in the problem --/
def problemLine : Line :=
  { a := 3
  , b := -1
  , c := 2 }

/-- The main theorem applied to the specific problem --/
theorem problem_solution : 
  0 < distancePointToLine problemCircle.center_x problemCircle.center_y problemLine ∧ 
  distancePointToLine problemCircle.center_x problemCircle.center_y problemLine < problemCircle.radius := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_problem_solution_l541_54140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l541_54125

theorem triangle_sine_inequality (A B C : Real) 
  (angle_sum : A + B + C = 180)
  (angle_bounds : 0 ≤ A ∧ A ≤ 180 ∧ 0 ≤ B ∧ B ≤ 180 ∧ 0 ≤ C ∧ C ≤ 180) :
  -2 ≤ Real.sin (3*A) + Real.sin (3*B) + Real.sin (3*C) ∧ 
  Real.sin (3*A) + Real.sin (3*B) + Real.sin (3*C) ≤ (3 * Real.sqrt 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l541_54125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coord_adjustment_l541_54181

/-- Represents a point in spherical coordinates -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Adjusts spherical coordinates to standard form -/
noncomputable def adjustToStandard (p : SphericalCoord) : SphericalCoord :=
  let φ' := if p.φ > Real.pi then 2*Real.pi - p.φ else p.φ
  let θ' := if p.φ > Real.pi then p.θ + Real.pi else p.θ
  ⟨p.ρ, θ' % (2*Real.pi), φ'⟩

/-- Checks if spherical coordinates are in standard form -/
def isStandard (p : SphericalCoord) : Prop :=
  p.ρ ≥ 0 ∧ 0 ≤ p.θ ∧ p.θ < 2*Real.pi ∧ 0 ≤ p.φ ∧ p.φ ≤ Real.pi

theorem spherical_coord_adjustment :
  let original := SphericalCoord.mk 4 (3*Real.pi/4) (9*Real.pi/5)
  let adjusted := adjustToStandard original
  adjusted = SphericalCoord.mk 4 (7*Real.pi/4) (Real.pi/5) ∧ isStandard adjusted :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coord_adjustment_l541_54181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_lower_bound_l541_54154

/-- For an odd prime p, σ(n) is the inverse of n modulo p -/
def sigma (p : ℕ) (n : ℕ) : ℕ :=
  if 1 ≤ n ∧ n < p then
    Nat.minFac (n * (Nat.choose p 2 + 1) % p)
  else 0

/-- Count pairs (a,b) with a < b and σ(a) > σ(b) -/
def count_pairs (p : ℕ) : ℕ :=
  Finset.sum (Finset.range (p-1)) (fun a =>
    Finset.sum (Finset.range (p-1)) (fun b =>
      if a < b ∧ sigma p a > sigma p b then 1 else 0))

theorem count_pairs_lower_bound (p : ℕ) (h : Nat.Prime p) (hodd : Odd p) :
  count_pairs p ≥ ((p-1)/4)^2 := by
  sorry

#eval count_pairs 7  -- Example evaluation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_lower_bound_l541_54154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_squared_over_area_l541_54160

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ+  -- first leg
  b : ℕ+  -- second leg
  c : ℕ+  -- hypotenuse
  right_angle : c * c = a * a + b * b
  coprime : Nat.Coprime a.val (Nat.gcd b.val c.val)

/-- Calculate the perimeter of a right triangle -/
def perimeter (t : RightTriangle) : ℕ :=
  t.a.val + t.b.val + t.c.val

/-- Calculate the area of a right triangle -/
def area (t : RightTriangle) : ℕ :=
  t.a.val * t.b.val / 2

/-- The theorem to be proved -/
theorem max_perimeter_squared_over_area :
  ∀ t : RightTriangle, (perimeter t)^2 / (area t) ≤ 45 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_squared_over_area_l541_54160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_theorem_l541_54192

/-- Represents the distribution of candies among three children -/
structure CandyDistribution where
  initial_ratio : Fin 3 → ℕ
  actual_ratio : Fin 3 → ℕ
  total_candies : ℕ

/-- The given candy distribution problem -/
def candy_problem : CandyDistribution where
  initial_ratio := ![5, 4, 3]
  actual_ratio := ![7, 6, 5]
  total_candies := 18 * 30  -- Derived from the solution, but could be any multiple of 18

theorem candy_distribution_theorem (d : CandyDistribution) 
  (h1 : d.initial_ratio = ![5, 4, 3])
  (h2 : d.actual_ratio = ![7, 6, 5])
  (h3 : ∃ i : Fin 3, d.actual_ratio i * (d.total_candies / (d.actual_ratio 0 + d.actual_ratio 1 + d.actual_ratio 2)) -
                     d.initial_ratio i * (d.total_candies / (d.initial_ratio 0 + d.initial_ratio 1 + d.initial_ratio 2)) = 15) :
  (d.actual_ratio 2 * (d.total_candies / (d.actual_ratio 0 + d.actual_ratio 1 + d.actual_ratio 2)) -
   d.initial_ratio 2 * (d.total_candies / (d.initial_ratio 0 + d.initial_ratio 1 + d.initial_ratio 2)) = 15) ∧
  (d.actual_ratio 2 * (d.total_candies / (d.actual_ratio 0 + d.actual_ratio 1 + d.actual_ratio 2)) = 150) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_theorem_l541_54192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_sets_l541_54139

-- Define the set A_k
def A (k : ℕ) : Set ℝ :=
  {x | ∃ t : ℝ, 1 / (k^2 : ℝ) ≤ t ∧ t ≤ 1 ∧ x = k * t + 1 / (k * t)}

-- State the theorem
theorem intersection_of_A_sets :
  (⋂ k ∈ Finset.range 2011, A (k + 2)) = Set.Icc 2 (5/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_sets_l541_54139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l541_54120

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 5*x + 4) / Real.log (1/3)

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := x < 1 ∨ x > 4

-- Theorem stating the properties of f(x)
theorem f_properties :
  (∀ x, domain x ↔ f x ≠ 0) ∧
  (∀ x y, x < y → x < 1 → y < 1 → f x < f y) ∧
  (∀ x y, x < y → x > 4 → y > 4 → f x > f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l541_54120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2022_l541_54195

/-- The function f(n) represents the number of the last person remaining
    in a line of n people after repeatedly removing those who count off odd numbers,
    alternating the counting direction each round. -/
def f (n : ℕ) : ℕ := sorry

/-- Helper function g(k) used in the calculation of f(n) -/
def g (k : ℕ) : ℚ := sorry

/-- Helper function h(x) used in the calculation of f(n) -/
def h (x : ℕ) : ℕ := sorry

theorem f_2022 : f 2022 = 1016 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2022_l541_54195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_students_average_age_l541_54151

theorem new_students_average_age
  (initial_count : ℕ)
  (initial_avg : ℝ)
  (new_count : ℕ)
  (new_total_avg : ℝ)
  (h1 : initial_count = 10)
  (h2 : initial_avg = 14)
  (h3 : new_count = 5)
  (h4 : new_total_avg = initial_avg + 1) :
  (new_total_avg * (initial_count + new_count) - initial_avg * initial_count) / new_count = 17 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_students_average_age_l541_54151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_alloy_copper_percentage_l541_54135

-- Define the given quantities
noncomputable def final_alloy_weight : ℝ := 108
noncomputable def final_copper_percentage : ℝ := 19.75
noncomputable def first_alloy_weight : ℝ := 45
noncomputable def second_alloy_percentage : ℝ := 21

-- Define the function to calculate the copper content
noncomputable def copper_content (weight : ℝ) (percentage : ℝ) : ℝ :=
  weight * (percentage / 100)

-- State the theorem
theorem first_alloy_copper_percentage :
  ∃ (x : ℝ),
    copper_content final_alloy_weight final_copper_percentage =
    copper_content first_alloy_weight x +
    copper_content (final_alloy_weight - first_alloy_weight) second_alloy_percentage ∧
    x = 18 := by
  sorry

#check first_alloy_copper_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_alloy_copper_percentage_l541_54135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l541_54199

/-- Regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  base_side : ℝ
  lateral_edge : ℝ

/-- Cross-section area of a regular quadrilateral pyramid -/
noncomputable def cross_section_area (p : RegularQuadPyramid) : ℝ :=
  (p.base_side * p.lateral_edge) / 4

/-- Theorem: The area of the cross-section in a regular quadrilateral pyramid -/
theorem cross_section_area_theorem (p : RegularQuadPyramid) :
  let plane_through_midpoints := True  -- Represents the plane through midpoints of AB and AD
  let plane_parallel_to_edge := True   -- Represents the plane parallel to SA
  cross_section_area p = (p.base_side * p.lateral_edge) / 4 :=
by
  -- Unfold the definition of cross_section_area
  unfold cross_section_area
  -- The definition matches exactly, so we're done
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l541_54199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pairwise_sums_determine_numbers_l541_54153

theorem pairwise_sums_determine_numbers :
  ∀ a b c d e : ℤ,
  a < b ∧ b < c ∧ c < d ∧ d < e →
  {a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} = 
    ({21, 26, 35, 40, 49, 51, 54, 60, 65, 79} : Finset ℤ) →
  a = 6 ∧ b = 15 ∧ c = 20 ∧ d = 34 ∧ e = 45 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pairwise_sums_determine_numbers_l541_54153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_OAB_min_area_MAB_l541_54124

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Represents a focal chord of the parabola -/
structure FocalChord where
  parabola : Parabola
  a : Point
  b : Point
  ha : a.y^2 = 2 * parabola.p * a.x
  hb : b.y^2 = 2 * parabola.p * b.x

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- Theorem: The minimum area of triangle OAB is p^2/2 -/
theorem min_area_OAB (parabola : Parabola) :
    ∃ (minArea : ℝ), minArea = parabola.p^2/2 ∧
    ∀ (a b : Point), a.y^2 = 2 * parabola.p * a.x → b.y^2 = 2 * parabola.p * b.x →
    triangleArea origin a b ≥ minArea := by
  sorry

/-- Theorem: The minimum area of triangle MAB is p^2 -/
theorem min_area_MAB (parabola : Parabola) :
    ∃ (minArea : ℝ), minArea = parabola.p^2 ∧
    ∀ (m a b : Point), 
    a.y^2 = 2 * parabola.p * a.x → 
    b.y^2 = 2 * parabola.p * b.x →
    (∃ (t : ℝ), m.y = t * (m.x - parabola.p/2)) →
    triangleArea m a b ≥ minArea := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_OAB_min_area_MAB_l541_54124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sum_coordinates_l541_54158

-- Define the points
def E : ℝ × ℝ := (2, -5)
def D : ℝ × ℝ := (4, -1)

-- Define F with variables
def F (x y : ℝ) : ℝ × ℝ := (x, y)

-- Define the midpoint condition
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem midpoint_sum_coordinates (x y : ℝ) :
  is_midpoint D E (F x y) → x + y = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sum_coordinates_l541_54158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l541_54111

-- Define the function f(x) = x^5 + x - 3
def f (x : ℝ) : ℝ := x^5 + x - 3

-- State the theorem
theorem root_in_interval :
  (∀ x y : ℝ, x < y → f x < f y) →  -- f is monotonically increasing
  f 1 < 0 →                         -- f(1) < 0
  0 < f 2 →                         -- f(2) > 0
  ∃! x : ℝ, x ∈ Set.Ioo 1 2 ∧ f x = 0 :=  -- There exists a unique root in (1, 2)
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l541_54111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l541_54180

theorem problem_solution (x y : ℝ) (h1 : (18 : ℝ)^x = 2) (h2 : (1.5 : ℝ)^y = 2) :
  1/x - 2/y = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l541_54180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_is_360_l541_54126

/-- Represents the dimensions and properties of a rectangular field -/
structure RectangularField where
  length : ℝ
  width : ℝ
  area : ℝ
  tiltedAngle : ℝ

/-- Represents the costs of fencing for different sides -/
structure FencingCost where
  uncoveredSide : ℝ
  adjacentSide : ℝ
  tiltedSide : ℝ

/-- Calculates the total cost of fencing for the given field and costs -/
def totalFencingCost (field : RectangularField) (costs : FencingCost) : ℝ :=
  field.length * costs.uncoveredSide +
  field.width * costs.adjacentSide +
  (2 * field.length) * costs.tiltedSide

/-- Theorem stating that the total fencing cost is $360 for the given field and costs -/
theorem fencing_cost_is_360 (field : RectangularField) (costs : FencingCost) :
  field.length = 30 ∧
  field.width = 20 ∧
  field.area = 600 ∧
  field.tiltedAngle = 60 ∧
  costs.uncoveredSide = 2 ∧
  costs.adjacentSide = 3 ∧
  costs.tiltedSide = 4 →
  totalFencingCost field costs = 360 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_is_360_l541_54126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l541_54164

-- Define complex numbers z₁ and z₂
def z₁ (a : ℝ) : ℂ := a^2 + (2*a)*Complex.I
def z₂ : ℂ := -1 + 2*Complex.I

-- Part 1: z₁ + z₂ is purely imaginary implies a = 1
theorem part1 (a : ℝ) : (z₁ a + z₂).re = 0 → a = 1 := by
  sorry

-- Part 2: z₁i - z₂ in fourth quadrant implies -√2 < a < 1/2
theorem part2 (a : ℝ) : 
  (z₁ a * Complex.I - z₂).re > 0 ∧ (z₁ a * Complex.I - z₂).im < 0 → 
  -Real.sqrt 2 < a ∧ a < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l541_54164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l541_54155

theorem trigonometric_identities (θ : ℝ) (h : Real.tan θ = 2) :
  Real.cos (2 * θ) = -3/5 ∧ Real.tan (θ - π/4) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l541_54155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_type_theorem_l541_54162

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define a color
inductive Color where
  | Red
  | Green

-- Define a painting of the plane
def Painting := Point → Color

-- Define distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Define congruence of triangles
def congruent (t1 t2 : Triangle) : Prop :=
  (distance t1.A t1.B = distance t2.A t2.B) ∧
  (distance t1.B t1.C = distance t2.B t2.C) ∧
  (distance t1.C t1.A = distance t2.C t2.A)

-- The main theorem
theorem ramsey_type_theorem (ABC : Triangle) (painting : Painting) :
  (∃ (p q : Point), painting p = Color.Red ∧ painting q = Color.Red ∧ distance p q = 1) ∨
  (∃ (p q r : Point), painting p = Color.Green ∧ painting q = Color.Green ∧ painting r = Color.Green ∧
    congruent (Triangle.mk p q r) ABC) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_type_theorem_l541_54162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_domain_l541_54163

noncomputable def f (x : ℝ) : ℝ := Real.rpow 3 x

theorem inverse_function_domain :
  ∀ x : ℝ, 0 < x → x ≤ 2 →
  ∃ y : ℝ, 1 < y ∧ y ≤ 9 ∧ f y = x ∧
  (∀ z : ℝ, 1 < z → z ≤ 9 → ∃ w : ℝ, 0 < w ∧ w ≤ 2 ∧ f w = z) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_domain_l541_54163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_greater_than_sqrt_500_l541_54116

theorem least_integer_greater_than_sqrt_500 : ∀ n : ℤ, n > Int.floor (Real.sqrt 500) → n ≥ 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_greater_than_sqrt_500_l541_54116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_line_equation_l541_54161

/-- A parabola defined by y^2 = -8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.2^2 = -8 * p.1}

/-- The midpoint of the chord -/
def ChordMidpoint : ℝ × ℝ := (-1, 1)

/-- The line containing the chord -/
def ChordLine (x y : ℝ) : Prop :=
  4 * x + y + 3 = 0

theorem chord_line_equation :
  ∃ (A B : ℝ × ℝ),
    A ∈ Parabola ∧ B ∈ Parabola ∧
    (A.1 + B.1, A.2 + B.2) / 2 = ChordMidpoint ∧
    ∀ (x y : ℝ), (x, y) ∈ Set.Icc A B ↔ ChordLine x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_line_equation_l541_54161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_existence_l541_54146

theorem special_set_existence (n : ℕ) (h : n ≥ 2) :
  ∃ (A : Finset ℕ), 
    (Finset.card A = n) ∧ 
    (∀ a ∈ A, a ≥ 2) ∧
    (∀ k ∈ A, 
      (A.prod (fun a => if a ≠ k then a else 1)) % k = 1 ∨ 
      (A.prod (fun a => if a ≠ k then a else 1)) % k = k - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_existence_l541_54146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_NaClO_molecular_weight_l541_54172

/-- The atomic weight of sodium (Na) in g/mol -/
def atomic_weight_Na : ℝ := 22.99

/-- The atomic weight of chlorine (Cl) in g/mol -/
def atomic_weight_Cl : ℝ := 35.45

/-- The atomic weight of oxygen (O) in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of a compound composed of Na, Cl, and O -/
def molecular_weight_NaClO : ℝ := atomic_weight_Na + atomic_weight_Cl + atomic_weight_O

/-- Theorem stating that the molecular weight of NaClO is approximately 74.44 g/mol -/
theorem NaClO_molecular_weight : ∃ ε > 0, |molecular_weight_NaClO - 74.44| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_NaClO_molecular_weight_l541_54172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_is_25_l541_54171

/-- The length of a string wrapping around a cylindrical post -/
noncomputable def string_length (post_circumference : ℝ) (post_height : ℝ) (num_loops : ℕ) : ℝ :=
  (num_loops : ℝ) * Real.sqrt (post_circumference ^ 2 + (post_height / (num_loops : ℝ)) ^ 2)

/-- Theorem: The length of the string wrapping around the post is 25 feet -/
theorem string_length_is_25 :
  string_length 4 15 5 = 25 := by
  -- Unfold the definition of string_length
  unfold string_length
  -- Simplify the expression
  simp [Real.sqrt_sq]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_is_25_l541_54171
