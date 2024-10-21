import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_DAB_is_pi_over_8_l1154_115484

/-- A right triangle with a specific point satisfying certain conditions -/
structure SpecialTriangle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  D : ℝ × ℝ
  -- Point C on AD
  C : ℝ × ℝ
  -- BAD is a right triangle with right angle at B
  right_angle_at_B : (A.1 - B.1) * (D.1 - B.1) + (A.2 - B.2) * (D.2 - B.2) = 0
  -- C is on line segment AD
  C_on_AD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (t * A.1 + (1 - t) * D.1, t * A.2 + (1 - t) * D.2)
  -- AC = CD
  AC_eq_CD : (A.1 - C.1)^2 + (A.2 - C.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2
  -- BC = 2AB
  BC_eq_2AB : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 4 * ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- The main theorem stating that angle DAB is π/8 radians -/
theorem angle_DAB_is_pi_over_8 (t : SpecialTriangle) : 
  Real.arccos ((t.D.1 - t.A.1) * (t.B.1 - t.A.1) + (t.D.2 - t.A.2) * (t.B.2 - t.A.2)) / 
    (Real.sqrt ((t.D.1 - t.A.1)^2 + (t.D.2 - t.A.2)^2) * Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)) = π / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_DAB_is_pi_over_8_l1154_115484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l1154_115457

theorem min_value_of_expression (x y : ℝ) (h : Real.log x + Real.log y = Real.log 10) (hx : x > 0) (hy : y > 0) :
  ∃ (min : ℝ), min = 20 ∧ ∀ a b, Real.log a + Real.log b = Real.log 10 → a > 0 → b > 0 → 2*a + 5*b ≥ min :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l1154_115457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_garden_theorem_l1154_115429

/-- Represents the tea garden areas and yields -/
structure TeaGarden where
  total_area : ℝ
  total_yield : ℝ
  wild_yield_per_mu : ℝ
  pruned_yield_per_mu : ℝ

/-- Calculates the areas of wild and pruned tea gardens -/
noncomputable def calculate_areas (g : TeaGarden) : ℝ × ℝ :=
  let wild_area := (g.pruned_yield_per_mu * g.total_area - g.total_yield) / (g.pruned_yield_per_mu - g.wild_yield_per_mu)
  let pruned_area := g.total_area - wild_area
  (wild_area, pruned_area)

/-- Calculates the minimum area to convert from pruned to wild -/
noncomputable def min_area_to_convert (g : TeaGarden) (wild_area pruned_area : ℝ) : ℝ :=
  (g.pruned_yield_per_mu * pruned_area - g.wild_yield_per_mu * wild_area) / (g.wild_yield_per_mu + g.pruned_yield_per_mu)

/-- Main theorem statement -/
theorem tea_garden_theorem (g : TeaGarden) 
  (h1 : g.total_area = 16)
  (h2 : g.total_yield = 660)
  (h3 : g.wild_yield_per_mu = 30)
  (h4 : g.pruned_yield_per_mu = 50) :
  let (wild_area, pruned_area) := calculate_areas g
  wild_area = 7 ∧ 
  pruned_area = 9 ∧ 
  min_area_to_convert g wild_area pruned_area = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_garden_theorem_l1154_115429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ellipse_d_value_l1154_115469

/-- An ellipse in the first quadrant tangent to both axes with foci at (4,8) and (d,8) -/
structure TangentEllipse where
  d : ℝ
  is_in_first_quadrant : Prop
  is_tangent_to_axes : Prop
  focus1 : ℝ × ℝ := (4, 8)
  focus2 : ℝ × ℝ := (d, 8)

/-- The value of d for the given ellipse is 6 -/
theorem tangent_ellipse_d_value (e : TangentEllipse) : e.d = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ellipse_d_value_l1154_115469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1154_115481

theorem log_inequality (a : ℝ) (h : a > 1) :
  (Real.log 16 / Real.log a) + 2 * (Real.log a / Real.log 4) ≥ 4 ∧
  (Real.log 16 / Real.log a) + 2 * (Real.log a / Real.log 4) = 4 ↔ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1154_115481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_f_2007_l1154_115458

/-- A function satisfying f(xy) = f(x) + f(y) for positive x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), 0 < x → 0 < y → f (x * y) = f x + f y

theorem find_f_2007 (f : ℝ → ℝ) (h1 : special_function f) (h2 : f (1007 / 2007) = 1) :
  f 2007 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_f_2007_l1154_115458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_increase_l1154_115459

/-- Theorem: Percentage increases when a square's side length is increased by 50% -/
theorem square_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.5 * s
  let area_increase := (new_side^2 - s^2) / s^2 * 100
  let perimeter_increase := (4 * new_side - 4 * s) / (4 * s) * 100
  let diagonal_increase := (new_side * Real.sqrt 2 - s * Real.sqrt 2) / (s * Real.sqrt 2) * 100
  area_increase = 125 ∧ perimeter_increase = 50 ∧ diagonal_increase = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_increase_l1154_115459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_area_is_maximum_l1154_115472

/-- A rectangle with given constraints --/
structure Rectangle where
  length : ℝ
  width : ℝ
  perimeter_constraint : length + width = 150
  length_constraint : length ≥ 70
  width_constraint : width ≥ 50

/-- The area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The optimal rectangle satisfying the constraints --/
def optimal_rectangle : Rectangle where
  length := 75
  width := 75
  perimeter_constraint := by norm_num
  length_constraint := by norm_num
  width_constraint := by norm_num

theorem optimal_area_is_maximum :
  ∀ r : Rectangle, area r ≤ area optimal_rectangle :=
by
  sorry

#eval area optimal_rectangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_area_is_maximum_l1154_115472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_MN_l1154_115471

/-- Theorem: The slope of line MN is 3, given M(2,3) and N(4,9) -/
theorem slope_of_line_MN : (9 - 3) / (4 - 2) = 3 := by
  norm_num

#check slope_of_line_MN

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_MN_l1154_115471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_monotonic_decreasing_function_a_range_l1154_115482

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ
| x => if x > 0 then -x^2 + a*x - 1 - a else -(if -x > 0 then -(-x)^2 + a*(-x) - 1 - a else 0)

-- State the theorem
theorem odd_monotonic_decreasing_function_a_range :
  ∀ a : ℝ,
  (∀ x : ℝ, f a x = -(f a (-x))) →  -- f is odd
  (∀ x y : ℝ, x < y → f a x > f a y) →  -- f is monotonically decreasing
  -1 ≤ a ∧ a ≤ 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_monotonic_decreasing_function_a_range_l1154_115482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1154_115415

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x - Real.cos x

theorem f_range :
  ∀ y ∈ Set.Icc (-2 : ℝ) (Real.sqrt 3),
  ∃ x ∈ Set.Icc (-Real.pi/2) (Real.pi/2),
  f x = y ∧
  ∀ x ∈ Set.Icc (-Real.pi/2) (Real.pi/2),
  -2 ≤ f x ∧ f x ≤ Real.sqrt 3 := by
  sorry

#check f_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1154_115415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_of_powers_l1154_115496

theorem units_digit_of_sum_of_powers : 
  ∃ (n : ℤ), (17 + Real.sqrt 198)^21 + (17 - Real.sqrt 198)^21 = n ∧ n % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_of_powers_l1154_115496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_arrangement_l1154_115412

def isValidArrangement (digits : Finset Nat) : Prop :=
  digits.card = 6 ∧
  ∀ d ∈ digits, d ∈ Finset.range 10 ∧ d ≠ 0 ∧
  ∃ (a b c d e f : Nat),
    {a, b, c, d, e, f} = digits ∧
    a + b + c = 25 ∧
    b + d + e + f = 14

theorem sum_of_valid_arrangement (digits : Finset Nat) :
  isValidArrangement digits → Finset.sum digits id = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_arrangement_l1154_115412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_triples_equation_l1154_115443

theorem positive_integer_triples_equation :
  ∀ x y z : ℕ,
  x > 0 → y > 0 → z > 0 →
  x^2 + 4^y = 5^z →
  ((x = 3 ∧ y = 2 ∧ z = 2) ∨
   (x = 1 ∧ y = 1 ∧ z = 1) ∨
   (x = 11 ∧ y = 1 ∧ z = 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_triples_equation_l1154_115443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_dividing_special_sum_has_three_prime_factors_l1154_115406

/-- Euler's totient function -/
def φ : ℕ → ℕ := sorry

/-- Sum of positive divisors function -/
def σ : ℕ → ℕ := sorry

/-- Count of distinct prime factors -/
def ω : ℕ → ℕ := sorry

theorem composite_dividing_special_sum_has_three_prime_factors (n : ℕ) 
  (h1 : n > 4)
  (h2 : ¬ Nat.Prime n)
  (h3 : (φ n * σ n + 1) % n = 0) :
  ω n ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_dividing_special_sum_has_three_prime_factors_l1154_115406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_skateboard_mph_l1154_115466

-- Define constants and variables
noncomputable def inch_per_mile : ℝ := 63360
noncomputable def minutes_per_hour : ℝ := 60

-- Define speeds relative to Susan's forward walking speed
noncomputable def pete_backward_speed : ℝ := 3
noncomputable def tracy_cartwheel_speed : ℝ := 2
noncomputable def mike_swim_speed : ℝ := 8 * tracy_cartwheel_speed
noncomputable def pete_hand_walk_speed : ℝ := tracy_cartwheel_speed / 4
noncomputable def pete_bike_speed : ℝ := 5 * mike_swim_speed

-- Define Pete's hand walking speed in inches per hour
noncomputable def pete_hand_walk_inch_per_hour : ℝ := 2.0

-- Define Patty's rowing speed relative to Pete's backward walking speed
noncomputable def patty_row_speed : ℝ := 3 * pete_backward_speed

-- Define Sarah's skateboarding speed relative to Patty's rowing speed
noncomputable def sarah_skateboard_speed : ℝ := 6 * patty_row_speed

-- Theorem to prove
theorem sarah_skateboard_mph : 
  sarah_skateboard_speed * minutes_per_hour = 2160.0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_skateboard_mph_l1154_115466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_APB_acute_rhombus_dot_product_l1154_115421

-- Define the points
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (0, 1)

-- Define the line y = x - 1
def on_line (P : ℝ × ℝ) : Prop := P.2 = P.1 - 1

-- Define the angle APB
noncomputable def angle_APB (P : ℝ × ℝ) : ℝ :=
  let PA := (A.1 - P.1, A.2 - P.2)
  let PB := (B.1 - P.1, B.2 - P.2)
  Real.arccos ((PA.1 * PB.1 + PA.2 * PB.2) / (Real.sqrt (PA.1^2 + PA.2^2) * Real.sqrt (PB.1^2 + PB.2^2)))

-- Define a rhombus
def is_rhombus (A B P Q : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BP := (P.1 - B.1, P.2 - B.2)
  let PQ := (Q.1 - P.1, Q.2 - P.2)
  let QA := (A.1 - Q.1, A.2 - Q.2)
  AB.1^2 + AB.2^2 = BP.1^2 + BP.2^2 ∧
  BP.1^2 + BP.2^2 = PQ.1^2 + PQ.2^2 ∧
  PQ.1^2 + PQ.2^2 = QA.1^2 + QA.2^2

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statements
theorem angle_APB_acute (P : ℝ × ℝ) (h : on_line P) :
  0 < angle_APB P ∧ angle_APB P < Real.pi / 2 := by sorry

theorem rhombus_dot_product (P Q : ℝ × ℝ) (h1 : on_line P) (h2 : is_rhombus A B P Q) :
  dot_product (B.1 - Q.1, B.2 - Q.2) (A.1 - Q.1, A.2 - Q.2) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_APB_acute_rhombus_dot_product_l1154_115421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_plus_s_eq_l1154_115435

-- Define the polynomials r(x) and s(x)
noncomputable def r : ℝ → ℝ := sorry
noncomputable def s : ℝ → ℝ := sorry

-- Define the conditions
axiom r_quadratic : ∃ a b c : ℝ, ∀ x, r x = a * x^2 + b * x + c
axiom s_cubic : ∃ a b c d : ℝ, ∀ x, s x = a * x^3 + b * x^2 + c * x + d
axiom r_at_2 : r 2 = 2
axiom s_at_neg_1 : s (-1) = 3
axiom vertical_asymptote : ∃ k : ℝ, ∀ x, s x = (x - 3)^2 * (x - k)
axiom hole_at_1 : ∃ m n : ℝ, ∀ x, r x = (x - 1) * (m * x + n)

-- The theorem to prove
theorem r_plus_s_eq (x : ℝ) : r x + s x = x^3 + (49 * x) / 3 - 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_plus_s_eq_l1154_115435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_theorem_l1154_115431

-- Define a type for colors
inductive Color
| Blue
| Red

-- Define a graph type
def Graph := Fin 9 → Fin 9 → Color

-- Define a complete graph
def isComplete (G : Graph) : Prop :=
  ∀ i j : Fin 9, i ≠ j → (G i j = Color.Blue ∨ G i j = Color.Red)

-- Define a monochromatic complete subgraph
def hasMonochromaticSubgraph (G : Graph) (c : Color) (n : Nat) : Prop :=
  ∃ (S : Finset (Fin 9)), S.card = n ∧ 
    ∀ (i j : Fin 9), i ∈ S → j ∈ S → i ≠ j → G i j = c

-- The main theorem
theorem ramsey_theorem (G : Graph) (h : isComplete G) : 
  (hasMonochromaticSubgraph G Color.Blue 4) ∨ 
  (hasMonochromaticSubgraph G Color.Red 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_theorem_l1154_115431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_five_l1154_115420

noncomputable def f (x : ℝ) : ℝ := (x^3 + 3*x^2 + 2*x + 10) / (x - 5)

theorem vertical_asymptote_at_five :
  ∃ (M : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 →
    ∀ (x : ℝ), 0 < |x - 5| ∧ |x - 5| < δ → |f x| > M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_five_l1154_115420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marcus_painting_multiple_l1154_115413

/-- The number of days Marcus paints -/
def days : ℕ := 5

/-- The number of paintings Marcus paints on the first day -/
def first_day_paintings : ℕ := 2

/-- The total number of paintings Marcus paints after 5 days -/
def total_paintings : ℕ := 62

/-- The multiple by which Marcus increases his painting output each day -/
def painting_multiple : ℕ → ℕ → ℕ := λ m n ↦ m^n

/-- The sum of paintings for all days given a multiple m -/
def sum_paintings (m : ℕ) : ℕ :=
  List.sum (List.map (λ n ↦ first_day_paintings * painting_multiple m n) (List.range days))

theorem marcus_painting_multiple :
  ∃ m : ℕ, sum_paintings m = total_paintings ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marcus_painting_multiple_l1154_115413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l1154_115473

theorem log_equation_solution (b x : ℝ) 
  (hb_pos : b > 0) 
  (hb_neq_one : b ≠ 1) 
  (hx_pos : x > 0) 
  (hx_neq_one : x ≠ 1) 
  (h_eq : (Real.log x) / (3 * Real.log b) + (Real.log b) / (3 * Real.log x) = 1) : 
  x = b := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l1154_115473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_zero_times_l1154_115438

/-- The distance function representing the position of a point moving along a straight line -/
noncomputable def distance (t : ℝ) : ℝ := (1/4) * t^4 - (5/3) * t^3 + 2 * t^2

/-- The velocity function, which is the derivative of the distance function -/
noncomputable def velocity (t : ℝ) : ℝ := t^3 - 5 * t^2 + 4 * t

theorem velocity_zero_times :
  ∀ t : ℝ, velocity t = 0 ↔ t = 0 ∨ t = 1 ∨ t = 4 := by
  sorry

#check velocity_zero_times

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_zero_times_l1154_115438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_value_l1154_115492

theorem tan_2x_value (x : ℝ) (h1 : x ∈ Set.Ioo (-π/2) (π/2)) (h2 : Real.sin x + Real.cos x = 1/5) : 
  Real.tan (2*x) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_value_l1154_115492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2016th_term_l1154_115477

def my_sequence (a : ℕ → ℚ) : Prop :=
  (∀ n ≥ 3, a n = (a (n-1)) / (a (n-2))) ∧
  a 1 = 2 ∧
  a 5 = 1/3

theorem sequence_2016th_term (a : ℕ → ℚ) (h : my_sequence a) : a 2016 = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2016th_term_l1154_115477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l1154_115404

/-- Represents a point in space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a segment connecting two points -/
structure Segment where
  start : Point
  endPoint : Point

/-- The game state -/
structure GameState where
  points : Finset Point
  segments : Finset Segment
  colors : Finset ℕ

/-- Winning strategy for the first player -/
def has_winning_strategy (g : GameState) : Prop :=
  ∃ (coloring : Segment → ℕ), 
    ∀ (point_coloring : Point → ℕ),
      ∃ (s : Segment), s ∈ g.segments ∧ 
        coloring s = point_coloring s.start ∧
        coloring s = point_coloring s.endPoint

/-- Main theorem statement -/
theorem first_player_wins (g : GameState) 
  (h1 : g.points.card = 200)
  (h2 : ∀ p1 p2, p1 ∈ g.points → p2 ∈ g.points → p1 ≠ p2 → 
    ∃ s, s ∈ g.segments ∧ ((s.start = p1 ∧ s.endPoint = p2) ∨ (s.start = p2 ∧ s.endPoint = p1)))
  (h3 : ∀ s1 s2, s1 ∈ g.segments → s2 ∈ g.segments → s1 ≠ s2 → 
    (s1.start ≠ s2.start ∧ s1.start ≠ s2.endPoint ∧ s1.endPoint ≠ s2.start ∧ s1.endPoint ≠ s2.endPoint))
  (h4 : g.colors.card ≥ 7) :
  has_winning_strategy g :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l1154_115404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_AMN_ABC_l1154_115456

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 15 ∧ BC = 30 ∧ AC = 22

-- Define the incenter of a triangle
def Incenter (A B C O : ℝ × ℝ) : Prop :=
  -- The incenter is equidistant from all sides of the triangle
  ∃ r : ℝ, ∀ P : ℝ × ℝ, P ∈ Set.range (fun t => (1-t) • A + t • B) ∪ 
                         Set.range (fun t => (1-t) • B + t • C) ∪ 
                         Set.range (fun t => (1-t) • C + t • A) → 
    Real.sqrt ((O.1 - P.1)^2 + (O.2 - P.2)^2) = r

-- Define a line parallel to BC through the incenter
def ParallelLineToBC (B C O M N : ℝ × ℝ) : Prop :=
  (M.2 - O.2) / (M.1 - O.1) = (N.2 - O.2) / (N.1 - O.1) ∧
  (M.2 - O.2) / (M.1 - O.1) = (C.2 - B.2) / (C.1 - B.1)

-- Define M and N as intersection points
def IntersectionPoints (A B C O M N : ℝ × ℝ) : Prop :=
  M ∈ Set.range (fun t => (1-t) • A + t • B) ∧ 
  N ∈ Set.range (fun t => (1-t) • A + t • C) ∧ 
  ParallelLineToBC B C O M N

-- Define the area of a triangle
noncomputable def AreaTriangle (A B C : ℝ × ℝ) : ℝ :=
  let s := ((Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) + 
            (Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)) + 
            (Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))) / 2
  Real.sqrt (s * (s - Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) * 
                 (s - Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)) * 
                 (s - Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)))

-- Theorem statement
theorem area_ratio_AMN_ABC 
  (A B C O M N : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : Incenter A B C O) 
  (h3 : IntersectionPoints A B C O M N) : 
  AreaTriangle A M N = (1/4) * AreaTriangle A B C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_AMN_ABC_l1154_115456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_income_calculation_l1154_115491

/-- Calculates the total income for a clothing store with specific pricing and discount rules -/
def calculate_total_income (
  tshirt_price : ℝ)
  (pants_price : ℝ)
  (skirt_price : ℝ)
  (refurbished_tshirt_discount : ℝ)
  (skirt_discount : ℝ)
  (tshirt_discount : ℝ)
  (pants_promotion : ℕ → ℕ)
  (sales_tax : ℝ)
  (tshirts_sold : ℕ)
  (refurbished_tshirts_sold : ℕ)
  (pants_sold : ℕ)
  (skirts_sold : ℕ) : ℝ :=
  sorry

theorem total_income_calculation :
  calculate_total_income
    5 4 6 0.5
    0.1 0.2 (λ n : ℕ => n + n / 3) 0.08
    15 7 6 12 = 141.804 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_income_calculation_l1154_115491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1154_115417

noncomputable def f (x : ℝ) := Real.log (1/x - 1)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1154_115417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_six_three_even_one_odd_l1154_115494

def is_even (n : ℕ) : Bool := n % 2 = 0
def is_odd (n : ℕ) : Bool := n % 2 ≠ 0

def has_three_even_one_odd (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.length = 4) ∧
  (digits.filter is_even).length = 3 ∧
  (digits.filter is_odd).length = 1

theorem smallest_four_digit_divisible_by_six_three_even_one_odd :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 6 = 0 ∧ has_three_even_one_odd n → 1002 ≤ n) ∧
  1000 ≤ 1002 ∧ 1002 < 10000 ∧ 1002 % 6 = 0 ∧ has_three_even_one_odd 1002 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_six_three_even_one_odd_l1154_115494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_shifted_cos_l1154_115434

open Real

noncomputable def f (x : ℝ) : ℝ := cos (2 * x + π / 4)

noncomputable def g (x : ℝ) : ℝ := cos (2 * x + 7 * π / 12)

axiom shift : ∀ x, g x = f (x + π / 6)

def is_symmetry_center (h : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, h (c + x) = h (c - x)

theorem symmetry_center_of_shifted_cos :
  is_symmetry_center g (11 * π / 24) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_shifted_cos_l1154_115434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_perimeter_l1154_115416

/-- A rectangular garden with given area and length -/
structure RectangularGarden where
  area : ℝ
  length : ℝ

/-- The perimeter of a rectangular garden -/
noncomputable def perimeter (g : RectangularGarden) : ℝ :=
  2 * (g.length + g.area / g.length)

/-- Theorem: A rectangular garden with area 28 and length 7 has perimeter 22 -/
theorem garden_perimeter :
  let g : RectangularGarden := { area := 28, length := 7 }
  perimeter g = 22 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_perimeter_l1154_115416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_meeting_time_l1154_115407

/-- The time taken for two cyclists to meet for the sixth time on an elliptical track -/
theorem cyclists_meeting_time (track_perimeter : ℝ) (initial_distance : ℝ) 
  (speed_A : ℝ) (speed_B : ℝ) (h1 : track_perimeter = 400) 
  (h2 : initial_distance = 300) (h3 : speed_A = 4) (h4 : speed_B = 6) : 
  ∃ (total_time : ℝ), total_time = 230 := by
  let relative_speed := speed_A + speed_B
  let first_meeting_time := initial_distance / relative_speed
  let subsequent_meeting_time := track_perimeter / relative_speed
  let total_time := first_meeting_time + 5 * subsequent_meeting_time
  use total_time
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_meeting_time_l1154_115407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1154_115490

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := x^2 = 8*y

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the directrix l
def directrix_l (y : ℝ) : Prop := y = -2

-- Define the point (0, 1)
def fixed_point : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem parabola_properties :
  -- The directrix is tangent to the circle
  (∃ x y : ℝ, my_circle x y ∧ directrix_l y) →
  -- The line passing through (0, 1) intersects the parabola at two points
  (∃ k : ℝ, ∀ x : ℝ, parabola_C x (k*x + 1)) →
  -- The directrix intersects the parabola at two points
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    parabola_C A.1 A.2 ∧ parabola_C B.1 B.2 ∧
    directrix_l A.2 ∧ directrix_l B.2) →
  -- Prove the equation of the parabola and the dot product property
  (∀ x y : ℝ, parabola_C x y ↔ x^2 = 8*y) ∧
  (∀ A B : ℝ × ℝ, 
    parabola_C A.1 A.2 → parabola_C B.1 B.2 → 
    (∃ k : ℝ, A.2 = k * A.1 + 1 ∧ B.2 = k * B.1 + 1) →
    A.1 * B.1 + A.2 * B.2 = -7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1154_115490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l1154_115451

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℤ, (3 * X^2 - 20 * X + 60 : Polynomial ℤ) = (X - 3) * q + 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l1154_115451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_l1154_115495

-- Define the hyperbolas
def hyperbola1 (x y : ℝ) : Prop := x^2 - y^2 = 1
def hyperbola2 (x y : ℝ) : Prop := x^2/4 - y^2/9 = 1

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Define the square
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the conditions
def square_conditions (s : Square) : Prop :=
  hyperbola1 s.A.x s.A.y ∧
  hyperbola1 s.B.x s.B.y ∧
  hyperbola2 s.C.x s.C.y ∧
  hyperbola2 s.D.x s.D.y ∧
  s.A.x > 0 ∧ s.B.x > 0 ∧ s.C.x > 0 ∧ s.D.x > 0 ∧
  s.A.y = s.D.y ∧ s.B.y = s.C.y ∧
  s.A.x = s.B.x ∧ s.C.x = s.D.x ∧
  (s.B.y - s.A.y) = (s.D.x - s.A.x)

-- Theorem statement
theorem square_area (s : Square) (h : square_conditions s) :
  abs ((s.B.y - s.A.y) * (s.D.x - s.A.x) - 0.8506) < 0.0001 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_l1154_115495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_fraction_inequality_l1154_115455

-- Statement 1
theorem sqrt_inequality : Real.sqrt 5 + Real.sqrt 7 > 1 + Real.sqrt 13 := by sorry

-- Statement 2
theorem fraction_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_fraction_inequality_l1154_115455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_focus_to_line_l1154_115445

-- Define the hyperbola
noncomputable def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus : ℝ × ℝ := (3, 0)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x₀ y₀ : ℝ) (A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

-- The theorem to prove
theorem distance_from_focus_to_line :
  distance_to_line right_focus.1 right_focus.2 1 2 (-8) = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_focus_to_line_l1154_115445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1154_115475

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point lies on the hyperbola -/
def onHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if two points are symmetric with respect to the origin -/
def symmetricOrigin (p1 p2 : Point) : Prop :=
  p1.x = -p2.x ∧ p1.y = -p2.y

/-- Check if two line segments are perpendicular -/
def perpendicular (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- Calculate the eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

theorem hyperbola_eccentricity (h : Hyperbola) (F P Q : Point) :
  F.x = Real.sqrt 5 ∧ F.y = 0 →
  onHyperbola h P ∧ onHyperbola h Q →
  symmetricOrigin P Q →
  perpendicular P F Q →
  triangleArea P Q F = 4 →
  eccentricity h = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1154_115475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_proof_l1154_115470

noncomputable def f (x : ℝ) : ℝ := 2^x

theorem function_range_proof :
  (∀ x ∈ Set.Icc 0 2, f x ∈ Set.Icc 1 4) ∧
  (∀ y ∈ Set.Icc 1 4, ∃ x ∈ Set.Icc 0 2, f x = y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_proof_l1154_115470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_mass_in_pounds_l1154_115499

-- Define the mass of the car in kilograms
def car_mass_kg : ℚ := 1500

-- Define the conversion factor from kilograms to pounds
def kg_to_pound : ℚ := 4536 / 10000

-- Function to convert kilograms to pounds
def kg_to_pounds (mass_kg : ℚ) : ℚ := mass_kg / kg_to_pound

-- Function to round to the nearest whole number
def round_to_nearest_whole (x : ℚ) : ℤ := 
  if x - Int.floor x < 1/2 then Int.floor x else Int.ceil x

-- Theorem statement
theorem car_mass_in_pounds : 
  round_to_nearest_whole (kg_to_pounds car_mass_kg) = 3307 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_mass_in_pounds_l1154_115499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platforms_l1154_115446

/-- Calculates the time taken for a train to cross a platform -/
noncomputable def time_to_cross (train_length : ℝ) (platform_length : ℝ) (speed : ℝ) : ℝ :=
  (train_length + platform_length) / speed

/-- Proves that a train of given length crossing two platforms of different lengths takes the calculated times -/
theorem train_crossing_platforms 
  (train_length : ℝ) 
  (platform1_length : ℝ) 
  (platform2_length : ℝ) 
  (time1 : ℝ) 
  (h1 : train_length = 310)
  (h2 : platform1_length = 110)
  (h3 : platform2_length = 250)
  (h4 : time1 = 15)
  (h5 : time_to_cross train_length platform1_length ((train_length + platform1_length) / time1) = time1) :
  time_to_cross train_length platform2_length ((train_length + platform1_length) / time1) = 20 := by
  sorry

-- Remove the #eval line as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platforms_l1154_115446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_remaining_volume_l1154_115488

noncomputable section

open Real

def sphere_volume (diameter : ℝ) : ℝ := (4 / 3) * Real.pi * (diameter / 2) ^ 3

def cylinder_volume (diameter : ℝ) (depth : ℝ) : ℝ := Real.pi * (diameter / 2) ^ 2 * depth

def bowling_ball_volume : ℝ := sphere_volume 40

def hole1_volume : ℝ := cylinder_volume 3 10
def hole2_volume : ℝ := cylinder_volume 3 10
def hole3_volume : ℝ := cylinder_volume 4 12
def hole4_volume : ℝ := cylinder_volume 5 8

def remaining_volume : ℝ := bowling_ball_volume - (hole1_volume + hole2_volume + hole3_volume + hole4_volume)

theorem bowling_ball_remaining_volume :
  remaining_volume = 10523.67 * Real.pi := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_remaining_volume_l1154_115488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_example_l1154_115487

/-- The speed of a car given its distance traveled and time taken -/
noncomputable def car_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Theorem: A car traveling 69 meters in 3 seconds has a speed of 23 meters per second -/
theorem car_speed_example : car_speed 69 3 = 23 := by
  -- Unfold the definition of car_speed
  unfold car_speed
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_example_l1154_115487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_money_days_l1154_115425

/-- Proves that it takes 40 days for Jeongyeon and Juho to have the same amount of money -/
theorem equal_money_days : 
  let jeongyeon_allowance : ℕ := 300
  let juho_allowance : ℕ := 500
  let jeongyeon_current : ℕ := 12000
  let juho_current : ℕ := 4000
  let days : ℕ := 40
  let jeongyeon_final := jeongyeon_current + jeongyeon_allowance * days
  let juho_final := juho_current + juho_allowance * days
  jeongyeon_final = juho_final := by
    -- Proof goes here
    sorry

/- The theorem statement defines:
   - The daily allowances for Jeongyeon and Juho
   - Their current money
   - The number of days (40)
   - The final amounts for both after 40 days
   - An assertion that these final amounts are equal
-/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_money_days_l1154_115425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_correct_l1154_115408

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The original number to be rounded -/
def original_number : ℝ := 52.63847

/-- The result of rounding to the nearest hundredth -/
def rounded_result : ℝ := 52.64

theorem round_to_hundredth_correct :
  round_to_hundredth original_number = rounded_result := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_correct_l1154_115408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_count_l1154_115478

theorem frog_count (b c : ℕ) : 
  (6 * b + 10 * c = 122) →  -- Total toes equation
  (4 * b + 8 * c = 92) →    -- Total fingers equation
  b + c = 15                -- Total number of frogs
:= by
  -- The proof is omitted
  sorry

#check frog_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_count_l1154_115478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1154_115441

/-- Proves that for given vectors a and b, if (a - λb) is perpendicular to b, then λ = 3/5 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (lambda : ℝ) :
  a = (1, 3) →
  b = (3, 4) →
  (a.1 - lambda * b.1, a.2 - lambda * b.2) • b = 0 →
  lambda = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1154_115441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_face_value_calculation_l1154_115476

/-- Calculates the face value of a bill given the true discount, interest rate, and time period. -/
noncomputable def calculate_face_value (true_discount : ℝ) (interest_rate : ℝ) (time_months : ℝ) : ℝ :=
  let time_years := time_months / 12
  let numerator := true_discount * (100 + interest_rate * time_years)
  let denominator := interest_rate * time_years
  numerator / denominator

/-- Theorem stating that the face value of a bill with the given conditions is approximately 200002.4 -/
theorem bill_face_value_calculation :
  let true_discount := (240 : ℝ)
  let interest_rate := (16 : ℝ)
  let time_months := (9 : ℝ)
  let calculated_value := calculate_face_value true_discount interest_rate time_months
  abs (calculated_value - 200002.4) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_face_value_calculation_l1154_115476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_part_three_l1154_115433

noncomputable section

-- Define the function f(x) = a^(x-1)
def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1)

-- Theorem 1
theorem part_one (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 3 = 4) : a = 2 := by
  sorry

-- Theorem 2
theorem part_two (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 → f a (Real.log (1/100) / Real.log 10) > f a (-2.1)) ∧
  (a < 1 → f a (Real.log (1/100) / Real.log 10) < f a (-2.1)) := by
  sorry

-- Theorem 3
theorem part_three (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a (Real.log a / Real.log 10) = 100) :
  a = 1/10 ∨ a = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_part_three_l1154_115433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1154_115493

/-- The function f(x) representing the middle term of the expansion -/
noncomputable def f (x : ℝ) : ℝ := (5/2) * x^3

/-- The lower bound of the interval -/
noncomputable def a : ℝ := Real.sqrt 2 / 2

/-- The upper bound of the interval -/
noncomputable def b : ℝ := Real.sqrt 2

/-- The theorem stating the range of m -/
theorem range_of_m (m : ℝ) : 
  (∀ x ∈ Set.Icc a b, f x ≤ m * x) → m ≥ 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1154_115493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octahedron_volume_l1154_115460

/-- The volume of a regular octahedron with edge length a -/
noncomputable def octahedron_volume (a : ℝ) : ℝ := (a^3 * Real.sqrt 2) / 3

/-- Theorem: The volume of a regular octahedron with edge length a is (a^3 * sqrt(2)) / 3 -/
theorem regular_octahedron_volume (a : ℝ) (h : a > 0) :
  octahedron_volume a = (a^3 * Real.sqrt 2) / 3 := by
  -- Unfold the definition of octahedron_volume
  unfold octahedron_volume
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octahedron_volume_l1154_115460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_is_pi_l1154_115405

/-- The smallest positive real number t such that the set of points 
    {(sin θ · cos θ, sin θ · sin θ) | 0 ≤ θ ≤ t} forms a complete circle in the xy-plane. -/
noncomputable def smallest_period : ℝ := Real.pi

/-- The set of points generated by r = sin θ for 0 ≤ θ ≤ t -/
def circle_points (t : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ p = (Real.sin θ * Real.cos θ, Real.sin θ * Real.sin θ)}

/-- Predicate to check if a set of points forms a complete circle -/
def is_complete_circle (s : Set (ℝ × ℝ)) : Prop :=
  ∀ x y : ℝ, x^2 + y^2 = 1 → (x, y) ∈ s

/-- Theorem stating that π is the smallest positive real number t such that 
    circle_points t forms a complete circle -/
theorem smallest_period_is_pi :
  (is_complete_circle (circle_points smallest_period)) ∧
  (∀ t : ℝ, 0 < t → t < smallest_period → ¬(is_complete_circle (circle_points t))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_is_pi_l1154_115405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_one_asymptote_iff_l1154_115497

/-- The function g(x) parametrized by b -/
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x + b) / (x^2 - 3*x + 2)

/-- A proposition stating that g(x) has exactly one vertical asymptote -/
def has_exactly_one_vertical_asymptote (b : ℝ) : Prop :=
  (∃! x : ℝ, x^2 - 3*x + 2 = 0 ∧ x^2 - 2*x + b = 0) ∧
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x^2 - 2*x + b ≠ 0)

/-- Theorem stating that g(x) has exactly one vertical asymptote iff b = 0 or b = 1 -/
theorem g_has_one_asymptote_iff (b : ℝ) :
  has_exactly_one_vertical_asymptote b ↔ b = 0 ∨ b = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_one_asymptote_iff_l1154_115497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l1154_115452

theorem triangle_shape (A B C : ℝ) (a b c : ℝ) : 
  B = 2 * A → 
  a = 1 → 
  b = 4 / 3 → 
  0 < A ∧ A < π → 
  0 < B ∧ B < π → 
  0 < C ∧ C < π → 
  A + B + C = π → 
  a * Real.sin B = b * Real.sin A → 
  B > π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l1154_115452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_S_n_l1154_115489

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  d : ℚ      -- Common difference
  seq_def : ∀ n, a (n + 1) = a n + d  -- Definition of arithmetic sequence

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1 : ℚ) * seq.d)

/-- The main theorem -/
theorem minimize_S_n (seq : ArithmeticSequence) 
  (h1 : seq.a 1 + seq.a 5 = -14)
  (h2 : S seq 9 = -27) :
  ∃ (n : ℕ), (∀ (m : ℕ), S seq n ≤ S seq m) ∧ n = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_S_n_l1154_115489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_greater_than_400_l1154_115440

def sequenceTerms : ℕ → ℕ
  | 0 => 1
  | n => 
    let groupNum := (Nat.sqrt (8 * n + 1) + 1) / 2
    let posInGroup := n - groupNum * (groupNum - 1) / 2
    if posInGroup = 1 then 1 else 2 * posInGroup - 1

def S (n : ℕ) : ℕ := (List.range n).map sequenceTerms |>.sum

theorem smallest_n_greater_than_400 : ∀ k : ℕ, k < 59 → S k ≤ 400 ∧ S 59 > 400 := by
  sorry

#eval S 58
#eval S 59

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_greater_than_400_l1154_115440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_group1_collinear_group2_not_collinear_group3_l1154_115467

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define collinearity
def collinear (a b : V) : Prop := ∃ (k : ℝ), a = k • b ∨ b = k • a

-- Group 1
theorem collinear_group1 (e : V) (he : e ≠ 0) :
  let a := (-3/2 : ℝ) • e
  let b := (2 : ℝ) • e
  collinear a b :=
by
  sorry

-- Group 2
theorem collinear_group2 (e₁ e₂ : V) (h₁ : e₁ ≠ 0) (h₂ : e₂ ≠ 0) (h₃ : ¬ collinear e₁ e₂) :
  let a := e₁ - e₂
  let b := (-3 : ℝ) • e₁ + (3 : ℝ) • e₂
  collinear a b :=
by
  sorry

-- Group 3
theorem not_collinear_group3 (e₁ e₂ : V) (h₁ : e₁ ≠ 0) (h₂ : e₂ ≠ 0) (h₃ : ¬ collinear e₁ e₂) :
  let a := e₁ - e₂
  let b := e₁ + (2 : ℝ) • e₂
  ¬ collinear a b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_group1_collinear_group2_not_collinear_group3_l1154_115467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_root_exists_l1154_115430

/-- The equation we're solving -/
def equation (x : ℝ) : Prop := x^2 = Real.sqrt (Real.log x + 100)

/-- The smallest root of the equation -/
noncomputable def smallest_root : ℝ := 10^(-100 : ℝ)

theorem equation_root_exists :
  ∃ (x₀ : ℝ), equation x₀ ∧
  smallest_root < x₀ ∧
  x₀ < smallest_root + 10^(-396 : ℝ) ∧
  (x₀ - smallest_root) / smallest_root < 10^(-390 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_root_exists_l1154_115430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_maximized_at_40_l1154_115427

/-- The volume function of the box -/
noncomputable def V (x : ℝ) : ℝ := x^2 * ((60 + x) / 2)

/-- Theorem stating that the volume is maximized when the base edge length is 40 -/
theorem volume_maximized_at_40 :
  ∃ (x : ℝ), x > 0 ∧ x < 60 ∧ 
  (∀ (y : ℝ), y > 0 → y < 60 → V y ≤ V x) ∧
  x = 40 := by
  sorry

#check volume_maximized_at_40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_maximized_at_40_l1154_115427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_mixture_percentage_l1154_115428

/-- Represents the composition of an acid-water mixture -/
structure Mixture where
  acid : ℝ
  water : ℝ

/-- The percentage of acid in a mixture -/
noncomputable def acid_percentage (m : Mixture) : ℝ :=
  m.acid / (m.acid + m.water) * 100

theorem acid_mixture_percentage (original : Mixture) :
  let mixture1 := Mixture.mk original.acid (original.water + 2)
  let mixture2 := Mixture.mk (original.acid + 2) (original.water + 4)
  acid_percentage mixture1 = 25 ∧
  acid_percentage mixture2 = 40 →
  acid_percentage original = 100 / 3 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_mixture_percentage_l1154_115428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monitor_width_21_l1154_115449

/-- Represents a monitor with given specifications -/
structure Monitor where
  height : ℕ
  dotsPerInch : ℕ
  totalPixels : ℕ

/-- Calculates the width of a monitor in inches -/
def monitorWidth (m : Monitor) : ℚ :=
  m.totalPixels / (m.height * m.dotsPerInch : ℚ)

/-- Theorem stating that a monitor with the given specifications has a width of 21 inches -/
theorem monitor_width_21 (m : Monitor)
  (h1 : m.height = 12)
  (h2 : m.dotsPerInch = 100)
  (h3 : m.totalPixels = 2520000) :
  monitorWidth m = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monitor_width_21_l1154_115449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_min_a_value_l1154_115442

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + x

-- Part I
theorem g_decreasing (a : ℝ) :
  (∃ k : ℝ, k * (1 - 0) + g a 1 = 2 ∧ (deriv (g a)) 1 = k) →
  ∀ x ∈ Set.Ioo 0 2, (deriv (g a)) x < 0 :=
by sorry

-- Part II
theorem min_a_value (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 (1/2), f a x ≠ 0) →
  a ≥ 2 - 4 * Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_min_a_value_l1154_115442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_5_eq_9_l1154_115444

noncomputable def f (x : ℝ) : ℝ := 5 / (3 - x)

noncomputable def g (x : ℝ) : ℝ := 2 / (f.invFun x) + 8

theorem g_of_5_eq_9 : g 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_5_eq_9_l1154_115444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_division_l1154_115426

/-- The number of regions formed by n lines in a plane -/
def num_regions (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- A line in the plane -/
structure Line where
  mk :: -- Add a constructor

/-- A point in the plane -/
structure Point where
  mk :: -- Add a constructor

/-- Two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- Three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop := sorry

/-- A point on a given line -/
def point_on_line (i : ℕ) : Point := sorry

/-- A line given by an index -/
def line (i : ℕ) : Line := sorry

/-- 
Theorem: n lines in a plane, where no two lines are parallel and 
no three lines intersect at a single point, divide the plane into 
1 + n(n+1)/2 parts.
-/
theorem plane_division (n : ℕ) :
  (∀ (i j : ℕ), i < j → j ≤ n → ¬ parallel (line i) (line j)) →
  (∀ (i j k : ℕ), i < j → j < k → k ≤ n → ¬ collinear (point_on_line i) (point_on_line j) (point_on_line k)) →
  num_regions n = 1 + n * (n + 1) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_division_l1154_115426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_2_eq_18_l1154_115436

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 15 / Real.sqrt x
def g (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 3

-- State the theorem
theorem f_of_g_of_2_eq_18 : f (g 2) = 18 := by
  -- Evaluate g(2)
  have h1 : g 2 = 1 := by
    simp [g]
    norm_num
  
  -- Calculate f(g(2)) = f(1)
  have h2 : f (g 2) = f 1 := by
    rw [h1]
  
  -- Simplify f(1)
  have h3 : f 1 = 18 := by
    simp [f]
    norm_num
  
  -- Conclude the proof
  rw [h2, h3]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_2_eq_18_l1154_115436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_successes_correct_l1154_115409

/-- The probability of exactly two successes in three independent attempts -/
def prob_two_successes (p : ℝ) : ℝ := 3 * p^2 * (1 - p)

/-- Theorem stating that the probability of exactly two successes in three independent 
    attempts with constant success probability p is equal to 3p^2(1-p) -/
theorem prob_two_successes_correct (p : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) : 
  prob_two_successes p = 3 * p^2 * (1 - p) := by
  -- Unfold the definition of prob_two_successes
  unfold prob_two_successes
  -- The result follows directly from the definition
  rfl

/-- Lemma: The probability of exactly two successes in three independent attempts
    is always between 0 and 1 (inclusive) -/
lemma prob_two_successes_in_range (p : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) : 
  0 ≤ prob_two_successes p ∧ prob_two_successes p ≤ 1 := by
  sorry  -- Proof omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_successes_correct_l1154_115409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_representable_number_l1154_115479

theorem largest_non_representable_number
  (a b c : ℕ+) 
  (h_coprime_ab : Nat.Coprime a b)
  (h_coprime_bc : Nat.Coprime b c)
  (h_coprime_ac : Nat.Coprime a c) :
  ∀ n : ℤ, n > 2*a*b*c - a*b - b*c - a*c →
    ∃ x y z : ℕ, n = x*b*c + y*c*a + z*a*b ∧
  ¬∃ x y z : ℕ, 2*a*b*c - a*b - b*c - a*c = x*b*c + y*c*a + z*a*b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_representable_number_l1154_115479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_product_l1154_115448

theorem complex_equality_product (a b : ℝ) : 
  (1 - 2*Complex.I) * Complex.I = Complex.ofReal a + Complex.I * b → a * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_product_l1154_115448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l1154_115402

theorem find_x (y z x : ℝ) (h1 : y * z ≠ 0)
  (h2 : ({2 * x, 3 * z, x * y} : Set ℝ) = {y, 2 * x^2, 3 * x * z}) : x = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l1154_115402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_l1154_115437

/-- The quadratic equation in x with parameter a -/
def quadratic_equation (a : ℝ) (x : ℝ) : Prop :=
  a^2 * x^2 + a * x + 1 - 7 * a^2 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (a : ℝ) : ℝ :=
  a^2 - 4 * a^2 * (1 - 7 * a^2)

/-- Condition for the equation to have two distinct integer roots -/
def has_two_distinct_integer_roots (a : ℝ) : Prop :=
  ∃ (p q : ℤ), p ≠ q ∧ quadratic_equation a (↑p) ∧ quadratic_equation a (↑q)

/-- The main theorem stating that there are infinitely many positive real numbers a
    for which the quadratic equation has two distinct integer roots -/
theorem infinitely_many_solutions :
  ∃ (S : Set ℝ), (∀ (a : ℝ), a ∈ S → a > 0) ∧
                 (∀ (a : ℝ), a ∈ S → has_two_distinct_integer_roots a) ∧
                 Set.Infinite S :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_l1154_115437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonically_increasing_l1154_115483

open Real

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := sin (ω * x + φ)

theorem function_monotonically_increasing
  (ω φ : ℝ)
  (h_ω : ω > 0)
  (h_φ : -π < φ ∧ φ < 0)
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x)
  (h_point : f ω φ (π/3) = 1)
  : StrictMonoOn (f ω φ) (Set.Icc (-π/6) (π/3)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonically_increasing_l1154_115483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_geometric_sequence_l1154_115462

theorem roots_geometric_sequence (k : ℝ) : 
  (∃ α β : ℝ, α^2 - 2*α + k^2 = 0 ∧ β^2 - 2*β + k^2 = 0) →
  (∃ r : ℝ, r ≠ 0 ∧ (α + β)^2 = α * β * r^2) →
  k = 2 ∨ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_geometric_sequence_l1154_115462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_g_inequality_l1154_115411

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x - 4 / x

noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ 1 then f x else -2 * x - 1

-- Theorem 1: f is monotonically increasing on [1, +∞)
theorem f_monotone_increasing :
  ∀ (x₁ x₂ : ℝ), 1 ≤ x₁ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

-- Theorem 2: g(a) < g(a+1) for a > 1/3
theorem g_inequality :
  ∀ (a : ℝ), a > 1/3 → g a < g (a + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_g_inequality_l1154_115411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bar_height_represents_frequency_l1154_115414

/-- Represents a 3D bar chart --/
structure ThreeDBarChart where
  -- Add necessary fields here

/-- Represents a categorical variable in the chart --/
structure CategoricalVariable where
  -- Add necessary fields here

/-- Represents the height of a bar in the chart --/
def barHeight (chart : ThreeDBarChart) (v : CategoricalVariable) : ℝ := sorry

/-- Represents the frequency of a categorical variable --/
def frequency (chart : ThreeDBarChart) (v : CategoricalVariable) : ℝ := sorry

/-- Theorem stating that the height of bars represents the frequency of categorical variables --/
theorem bar_height_represents_frequency (chart : ThreeDBarChart) (v : CategoricalVariable) :
  barHeight chart v = frequency chart v := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bar_height_represents_frequency_l1154_115414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l1154_115450

theorem cos_beta_value (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos α = 1/7) (h4 : Real.cos (α + β) = -11/14) : Real.cos β = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l1154_115450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l1154_115422

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + Real.sqrt 3 * Real.cos (2 * x)

theorem period_of_f :
  ∃ (T : ℝ), T > 0 ∧ 
  (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l1154_115422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_existence_l1154_115464

theorem divisor_existence (S : Finset ℕ) : 
  S ⊆ Finset.range 2015 → S.card = 1008 → 
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_existence_l1154_115464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lakers_win_in_seven_games_l1154_115468

def probability_celtics_win : ℚ := 3/4

def games_to_win : ℕ := 4

def total_games : ℕ := 7

def probability_lakers_win_series (p : ℚ) (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k) * (1 - p)^k * p^(n - k) * (1 - p)

theorem lakers_win_in_seven_games :
  probability_lakers_win_series probability_celtics_win (total_games - 1) (games_to_win - 1) = 135/4096 := by
  sorry

#eval probability_lakers_win_series probability_celtics_win (total_games - 1) (games_to_win - 1)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lakers_win_in_seven_games_l1154_115468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eraser_price_is_one_l1154_115432

/-- Represents the store's pricing and sales policy -/
structure StoreSales where
  pencil_price : ℝ
  eraser_price : ℝ
  pencils_sold : ℕ
  total_earnings : ℝ
  eraser_price_ratio : eraser_price = pencil_price / 2
  eraser_quantity_ratio : ℕ := pencils_sold * 2

/-- Theorem stating that under the given conditions, the eraser price is $1 -/
theorem eraser_price_is_one (s : StoreSales)
  (h1 : s.pencils_sold = 20)
  (h2 : s.total_earnings = 80)
  : s.eraser_price = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eraser_price_is_one_l1154_115432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_approx_6m_l1154_115439

/-- The volume of a frustum of a cone --/
noncomputable def frustumVolume (r1 r2 h : ℝ) : ℝ := 
  (1/3) * Real.pi * h * (r1^2 + r2^2 + r1 * r2)

/-- The length of a wire made from a given volume of silver --/
noncomputable def wireLengthInMeters (volume : ℝ) (d1 d2 : ℝ) : ℝ :=
  let r1 := d1 / 2 / 10  -- Convert mm to cm and diameter to radius
  let r2 := d2 / 2 / 10
  let h := volume / (frustumVolume r1 r2 1)
  h / 100  -- Convert cm to m

theorem wire_length_approx_6m (volume : ℝ) (d1 d2 : ℝ) 
    (h_volume : volume = 33)
    (h_d1 : d1 = 1)
    (h_d2 : d2 = 2) :
  ∃ ε > 0, |wireLengthInMeters volume d1 d2 - 6| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_approx_6m_l1154_115439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1154_115465

/-- The line equation mx - y + 1 - m = 0 -/
def line_eq (m : ℝ) (x y : ℝ) : Prop :=
  m * x - y + 1 - m = 0

/-- The circle equation x^2 + (y-1)^2 = 5 -/
def circle_eq (x y : ℝ) : Prop :=
  x^2 + (y-1)^2 = 5

/-- Theorem stating that the line intersects the circle for any real m -/
theorem line_intersects_circle (m : ℝ) :
  ∃ x y : ℝ, line_eq m x y ∧ circle_eq x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1154_115465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1154_115498

/-- Represents the train's journey with an accident -/
structure TrainJourney where
  originalSpeed : ℝ
  distanceBeforeAccident : ℝ
  reducedSpeedFactor : ℝ
  totalDistance : ℝ
  normalTime : ℝ
  delayTime : ℝ

/-- Calculates the total journey time given a TrainJourney -/
noncomputable def totalJourneyTime (tj : TrainJourney) : ℝ :=
  tj.distanceBeforeAccident / tj.originalSpeed +
  (tj.totalDistance - tj.distanceBeforeAccident) / (tj.reducedSpeedFactor * tj.originalSpeed)

theorem train_speed_theorem (tj1 tj2 : TrainJourney) :
  tj1.originalSpeed = 48 ∧
  tj1.distanceBeforeAccident = 50 ∧
  tj1.reducedSpeedFactor = 3/4 ∧
  tj1.delayTime = 35/60 ∧
  tj2.distanceBeforeAccident = 74 ∧
  tj2.delayTime = 25/60 ∧
  tj1.totalDistance = tj2.totalDistance ∧
  tj1.normalTime = tj2.normalTime ∧
  totalJourneyTime tj1 = tj1.normalTime + tj1.delayTime ∧
  totalJourneyTime tj2 = tj2.normalTime + tj2.delayTime →
  tj1.originalSpeed = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1154_115498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_min_distance_is_correct_l1154_115403

/-- Circle C1 defined by the equation x^2 + y^2 - 8x - 4y + 11 = 0 -/
def C1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x - 4*y + 11 = 0

/-- Circle C2 defined by the equation x^2 + y^2 + 4x + 2y + 1 = 0 -/
def C2 (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x + 2*y + 1 = 0

/-- The distance between two points (x1, y1) and (x2, y2) -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem min_distance_between_circles :
  ∀ (x1 y1 x2 y2 : ℝ), C1 x1 y1 → C2 x2 y2 →
  distance x1 y1 x2 y2 ≥ 3 * Real.sqrt 5 - 3 - Real.sqrt 6 :=
by
  sorry

/-- The minimum distance between any point on C1 and any point on C2 -/
noncomputable def min_distance : ℝ :=
  3 * Real.sqrt 5 - 3 - Real.sqrt 6

theorem min_distance_is_correct :
  ∀ (x1 y1 x2 y2 : ℝ), C1 x1 y1 → C2 x2 y2 →
  distance x1 y1 x2 y2 ≥ min_distance :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_min_distance_is_correct_l1154_115403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_price_before_increase_l1154_115418

/-- The price of a candy box after a 25% increase -/
def candy_new_price : ℚ := 10

/-- The price of a can of soda after a 50% increase -/
def soda_new_price : ℚ := 9

/-- The percentage increase for the candy box price -/
def candy_increase_rate : ℚ := 1/4

/-- The percentage increase for the soda can price -/
def soda_increase_rate : ℚ := 1/2

/-- The original price of a candy box -/
noncomputable def candy_original_price : ℚ := candy_new_price / (1 + candy_increase_rate)

/-- The original price of a can of soda -/
noncomputable def soda_original_price : ℚ := soda_new_price / (1 + soda_increase_rate)

/-- The combined original price of a candy box and a can of soda -/
noncomputable def combined_original_price : ℚ := candy_original_price + soda_original_price

theorem combined_price_before_increase : combined_original_price = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_price_before_increase_l1154_115418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_linear_l1154_115453

/-- A function f: ℝ → ℝ is linear if there exists a constant m such that f(x) = mx for all x ∈ ℝ -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, f x = m * x

/-- The function f(x) = (1/3)x -/
noncomputable def f : ℝ → ℝ := fun x ↦ (1/3) * x

/-- Theorem: f is a linear function -/
theorem f_is_linear : IsLinearFunction f := by
  use (1/3)
  intro x
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_linear_l1154_115453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_dependence_iff_k_eq_8_l1154_115463

/-- Two vectors in ℝ³ -/
def v1 : Fin 3 → ℝ := ![1, 2, 3]
def v2 (k : ℝ) : Fin 3 → ℝ := ![4, k, 6]

/-- The vectors are linearly dependent -/
def are_linearly_dependent (k : ℝ) : Prop :=
  ∃ (a b : ℝ) (h : (a, b) ≠ (0, 0)), 
    (Finset.sum Finset.univ (λ i => a * v1 i + b * v2 k i)) = 0

/-- The main theorem -/
theorem linear_dependence_iff_k_eq_8 :
  ∀ k, are_linearly_dependent k ↔ k = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_dependence_iff_k_eq_8_l1154_115463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_set_size_l1154_115424

def p : ℕ := 2^16 + 1

theorem max_set_size (h_prime : Nat.Prime p) :
  (∃ (S : Finset ℕ), 
    (∀ x, x ∈ S → x < p) ∧ 
    (∀ a b, a ∈ S → b ∈ S → a ≠ b → (a^2 % p ≠ b % p)) ∧
    (S.card = 43691)) ∧
  (∀ (T : Finset ℕ), 
    (∀ x, x ∈ T → x < p) → 
    (∀ a b, a ∈ T → b ∈ T → a ≠ b → (a^2 % p ≠ b % p)) →
    T.card ≤ 43691) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_set_size_l1154_115424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_line_equation_l1154_115400

/-- The equation of an ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The midpoint of a chord -/
def chord_midpoint : Point := ⟨2, -1⟩

/-- The equation of a line -/
def line_equation (x y : ℝ) : Prop := y = x - 3

/-- Theorem: The equation of the line containing the chord -/
theorem chord_line_equation :
  ∀ (A B : Point),
  ellipse A.x A.y ∧ ellipse B.x B.y →
  chord_midpoint = Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2) →
  line_equation A.x A.y ∧ line_equation B.x B.y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_line_equation_l1154_115400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_height_of_volleyball_team_l1154_115401

/-- Represents a stem and leaf plot entry -/
structure StemLeafEntry where
  stem : ℕ
  leaves : List ℕ

/-- Calculate the mean height of players given their heights in stem and leaf format -/
def meanHeight (heights : List StemLeafEntry) : ℚ :=
  let totalSum : ℚ := heights.foldl (fun acc entry => 
    acc + (entry.stem * 100 + entry.leaves.sum : ℕ)) 0
  let totalCount : ℕ := heights.foldl (fun acc entry => acc + entry.leaves.length) 0
  totalSum / totalCount

/-- The heights of the players on the West End High School boys' volleyball team -/
def volleyballTeamHeights : List StemLeafEntry := [
  ⟨15, [0, 2, 5, 8]⟩,
  ⟨16, [4, 8, 9, 1, 3, 5, 7]⟩,
  ⟨17, [0, 1, 3, 5, 6, 9]⟩,
  ⟨18, [2, 4, 8]⟩
]

theorem mean_height_of_volleyball_team : 
  meanHeight volleyballTeamHeights = 3389 / 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_height_of_volleyball_team_l1154_115401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_q_value_l1154_115485

/-- Given p and q are real numbers, if there are exactly three distinct values of x that satisfy
    the equation |x^2 + px + q| = 3, then the minimum value of q is -3. -/
theorem min_q_value (p q : ℝ) : 
  (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, |x^2 + p*x + q| = 3) → 
  (∀ r : ℝ, (∃! (t : Finset ℝ), t.card = 3 ∧ ∀ x ∈ t, |x^2 + p*x + r| = 3) → q ≤ r) ∧
  q = -3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_q_value_l1154_115485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l1154_115454

theorem cube_root_simplification :
  (Real.rpow (8 + 1/9) (1/3) : ℝ) = Real.rpow 73 (1/3) / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l1154_115454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l1154_115480

/-- A hyperbola with parameter b > 0 -/
structure Hyperbola (b : ℝ) where
  eq : ∀ x y : ℝ, x^2 - y^2 / b^2 = 1
  b_pos : b > 0

/-- The distance from a focus of the hyperbola to one of its asymptotes -/
noncomputable def focus_to_asymptote_distance (b : ℝ) (h : Hyperbola b) : ℝ := 
  b * Real.sqrt (1 + b^2) / Real.sqrt (1 + b^2)

/-- If the distance from a focus to an asymptote is 1, then b = 1 -/
theorem hyperbola_focus_distance (b : ℝ) (h : Hyperbola b) :
  focus_to_asymptote_distance b h = 1 → b = 1 :=
by sorry

/-- A possible equation for hyperbola C1 with the same asymptotes as C -/
def hyperbola_C1_eq (k : ℝ) (h : k ≠ 1) (x y : ℝ) : Prop :=
  x^2 - y^2 = k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l1154_115480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_pentagon_area_is_745_l1154_115474

/-- The area of a pentagon formed by cutting a right-angled triangle from a rectangle -/
theorem pentagon_area : ℝ := by
  let pentagon_sides : List ℝ := [13, 19, 20, 25, 31]
  let triangle_hypotenuse : ℝ := 13
  let triangle_leg1 : ℝ := 12
  let triangle_leg2 : ℝ := 5
  let rectangle_length : ℝ := 31
  let rectangle_width : ℝ := 25

  let rectangle_area : ℝ := rectangle_length * rectangle_width
  let triangle_area : ℝ := (1 / 2) * triangle_leg1 * triangle_leg2
  let pentagon_area : ℝ := rectangle_area - triangle_area

  have h : pentagon_area = 745 := by sorry
  exact pentagon_area

/-- Proof that the calculated pentagon area equals 745 -/
theorem pentagon_area_is_745 : pentagon_area = 745 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_pentagon_area_is_745_l1154_115474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_survivors_l1154_115410

/-- Approximation relation for natural numbers -/
def approx (a b : ℕ) : Prop :=
  (a : ℚ) - (b : ℚ) < 1 ∧ (b : ℚ) - (a : ℚ) < 1

notation:50 a " ≈ " b => approx a b

/-- The expected number of survivors after three months in an animal population -/
theorem expected_survivors (initial_population : ℕ) 
  (survival_prob_month1 : ℚ) (survival_prob_month2 : ℚ) (survival_prob_month3 : ℚ) :
  initial_population = 700 →
  survival_prob_month1 = 9/10 →
  survival_prob_month2 = 8/10 →
  survival_prob_month3 = 7/10 →
  ∃ (survivors : ℕ), approx survivors 353 ∧ 
    (survivors : ℚ) = initial_population * survival_prob_month1 * survival_prob_month2 * survival_prob_month3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_survivors_l1154_115410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1154_115461

/-- Given a point A(2,5), its reflection B over the y-axis, and the reflection C of B over the line y=-x, 
    prove that the area of triangle ABC is 14 square units. -/
theorem area_of_triangle_ABC : ∃ (A B C : ℝ × ℝ), 
  A = (2, 5) ∧ 
  B = (-A.1, A.2) ∧ 
  C = (-B.2, -B.1) ∧ 
  abs ((A.1 - C.1) * (B.2 - A.2) - (A.1 - B.1) * (C.2 - A.2)) / 2 = 14 := by
  -- Define points A, B, and C
  let A : ℝ × ℝ := (2, 5)
  let B : ℝ × ℝ := (-A.1, A.2)
  let C : ℝ × ℝ := (-B.2, -B.1)

  -- Calculate the area
  have area : ℝ := abs ((A.1 - C.1) * (B.2 - A.2) - (A.1 - B.1) * (C.2 - A.2)) / 2

  -- Prove that A, B, and C satisfy the conditions and the area is 14
  use A, B, C
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  -- Prove that the area equals 14
  sorry  -- This step requires actual computation, which we skip for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1154_115461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_example_l1154_115486

/-- Definition of the diamond operation -/
noncomputable def diamond (A B : ℝ) : ℝ := (A^2 + B^2) / 5

/-- Theorem stating the result of (3 ◇ 7) ◇ 4 -/
theorem diamond_example : diamond (diamond 3 7) 4 = 30.112 := by
  -- Expand the definition of diamond
  unfold diamond
  -- Simplify the expression
  simp [pow_two]
  -- Perform numerical calculations
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_example_l1154_115486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mice_elimination_theorem_l1154_115419

/-- Represents the number of days it takes to eliminate all mice -/
def eliminate_mice_days (total_work : ℚ) (initial_cats : ℕ) (initial_days : ℕ) (initial_work : ℚ) (added_cats : ℕ) : ℚ :=
  initial_days + (total_work - initial_work) / (initial_work / initial_days * (initial_cats + added_cats) / initial_cats)

/-- Theorem stating that it takes 7 days to eliminate all mice under given conditions -/
theorem mice_elimination_theorem :
  eliminate_mice_days 1 2 5 (1/2) 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mice_elimination_theorem_l1154_115419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_and_vertical_displacement_l1154_115447

noncomputable def f (x : ℝ) := Real.sin (5 * x - Real.pi / 2) + 2

theorem phase_shift_and_vertical_displacement :
  ∃ (phase_shift vertical_displacement : ℝ),
    phase_shift = -Real.pi / 10 ∧
    vertical_displacement = 2 ∧
    ∀ x, f x = Real.sin (5 * (x + phase_shift)) + vertical_displacement :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_and_vertical_displacement_l1154_115447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l1154_115423

-- Define the sequences
def a : ℕ → ℕ := sorry
def b : ℕ → ℕ := sorry
def c : ℕ → ℕ := sorry

-- Define the sum function
def S : ℕ → ℕ := sorry

-- State the theorem
theorem sequence_sum (n : ℕ) :
  (∀ k, b k = 2^(a k - 1)) →  -- b_n = 2^(a_n - 1)
  (a 1 = 1) →                 -- a_1 = 1
  (a 3 = 3) →                 -- a_3 = 3
  (∀ k, c k = a k * b k) →    -- c_n = a_n * b_n
  (∀ k, k > 1 → b k / b (k-1) = b (k+1) / b k) →  -- {b_n} is geometric
  S n = n * 2^n :=            -- S_n = n * 2^n
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l1154_115423
