import Mathlib

namespace NUMINAMATH_CALUDE_unique_m_value_l1444_144428

/-- Given function f(x) = |x-a| + m|x+a| -/
def f (x a m : ℝ) : ℝ := |x - a| + m * |x + a|

theorem unique_m_value (m a : ℝ) 
  (h1 : 0 < m) (h2 : m < 1)
  (h3 : ∀ x : ℝ, f x a m ≥ 2)
  (h4 : a ≤ -5 ∨ a ≥ 5) :
  m = 1/5 := by sorry

end NUMINAMATH_CALUDE_unique_m_value_l1444_144428


namespace NUMINAMATH_CALUDE_convex_curve_sum_containment_l1444_144458

/-- A convex curve in a 2D plane -/
structure ConvexCurve where
  points : Set (ℝ × ℝ)
  convex : sorry -- Add appropriate convexity condition

/-- The Minkowski sum of two convex curves -/
def minkowski_sum (K L : ConvexCurve) : ConvexCurve :=
  sorry

/-- One curve does not go beyond another -/
def not_beyond (K L : ConvexCurve) : Prop :=
  K.points ⊆ L.points

theorem convex_curve_sum_containment
  (K₁ K₂ L₁ L₂ : ConvexCurve)
  (h₁ : not_beyond K₁ L₁)
  (h₂ : not_beyond K₂ L₂) :
  not_beyond (minkowski_sum K₁ K₂) (minkowski_sum L₁ L₂) :=
sorry

end NUMINAMATH_CALUDE_convex_curve_sum_containment_l1444_144458


namespace NUMINAMATH_CALUDE_exhibition_planes_l1444_144402

/-- The number of wings on a commercial plane -/
def wings_per_plane : ℕ := 2

/-- The total number of wings counted -/
def total_wings : ℕ := 50

/-- The number of planes in the exhibition -/
def num_planes : ℕ := total_wings / wings_per_plane

theorem exhibition_planes : num_planes = 25 := by
  sorry

end NUMINAMATH_CALUDE_exhibition_planes_l1444_144402


namespace NUMINAMATH_CALUDE_age_ratio_problem_l1444_144465

theorem age_ratio_problem (p q : ℕ) 
  (h1 : p - 12 = (q - 12) / 2)
  (h2 : p + q = 42) :
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ a * q = b * p ∧ a = 3 ∧ b = 4 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l1444_144465


namespace NUMINAMATH_CALUDE_max_sum_nonnegative_l1444_144413

theorem max_sum_nonnegative (a b c d : ℝ) (h : a + b + c + d = 0) :
  max a b + max a c + max a d + max b c + max b d + max c d ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_nonnegative_l1444_144413


namespace NUMINAMATH_CALUDE_resultant_calculation_l1444_144441

theorem resultant_calculation : 
  let original : ℕ := 13
  let doubled := 2 * original
  let added_seven := doubled + 7
  let trebled := 3 * added_seven
  trebled = 99 := by sorry

end NUMINAMATH_CALUDE_resultant_calculation_l1444_144441


namespace NUMINAMATH_CALUDE_max_sum_reciprocals_l1444_144482

theorem max_sum_reciprocals (p q r x y z : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hpqr : p + q + r = 2) (hxyz : x + y + z = 1) :
  1/(p+q) + 1/(p+r) + 1/(q+r) + 1/(x+y) + 1/(x+z) + 1/(y+z) ≤ 27/4 := by
sorry

end NUMINAMATH_CALUDE_max_sum_reciprocals_l1444_144482


namespace NUMINAMATH_CALUDE_fraction_reducibility_l1444_144486

theorem fraction_reducibility (l : ℤ) :
  ∃ (d : ℤ), d > 1 ∧ d ∣ (5 * l + 6) ∧ d ∣ (8 * l + 7) ↔ ∃ (k : ℤ), l = 13 * k + 4 :=
sorry

end NUMINAMATH_CALUDE_fraction_reducibility_l1444_144486


namespace NUMINAMATH_CALUDE_inequality_implies_range_l1444_144459

theorem inequality_implies_range (a : ℝ) : 
  (∀ x ∈ Set.Icc (0 : ℝ) (1/2), 4^x + x - a ≤ 3/2) → a ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_range_l1444_144459


namespace NUMINAMATH_CALUDE_remaining_days_temperature_l1444_144481

/-- Calculates the total temperature of the remaining days in a week given specific temperature conditions. -/
theorem remaining_days_temperature
  (avg_temp : ℝ)
  (days_in_week : ℕ)
  (first_three_temp : ℝ)
  (thursday_friday_temp : ℝ)
  (h1 : avg_temp = 60)
  (h2 : days_in_week = 7)
  (h3 : first_three_temp = 40)
  (h4 : thursday_friday_temp = 80) :
  (days_in_week : ℝ) * avg_temp - (3 * first_three_temp + 2 * thursday_friday_temp) = 140 := by
  sorry

#check remaining_days_temperature

end NUMINAMATH_CALUDE_remaining_days_temperature_l1444_144481


namespace NUMINAMATH_CALUDE_volume_maximized_at_one_meter_l1444_144477

/-- Represents the dimensions of a rectangular box --/
structure BoxDimensions where
  x : Real  -- Length of the shorter side of the base
  h : Real  -- Height of the box

/-- Calculates the volume of the box given its dimensions --/
def boxVolume (d : BoxDimensions) : Real :=
  2 * d.x^2 * d.h

/-- Calculates the total wire length used for the box frame --/
def wireLengthUsed (d : BoxDimensions) : Real :=
  12 * d.x + 4 * d.h

/-- Theorem stating that the volume is maximized when the shorter side is 1m --/
theorem volume_maximized_at_one_meter :
  ∃ (d : BoxDimensions),
    wireLengthUsed d = 18 ∧
    (∀ (d' : BoxDimensions), wireLengthUsed d' = 18 → boxVolume d' ≤ boxVolume d) ∧
    d.x = 1 :=
  sorry

end NUMINAMATH_CALUDE_volume_maximized_at_one_meter_l1444_144477


namespace NUMINAMATH_CALUDE_kenny_must_do_at_least_three_on_thursday_l1444_144468

/-- Represents the number of jumping jacks done on each day of the week -/
structure WeeklyJumpingJacks where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the total number of jumping jacks for a week -/
def weekTotal (w : WeeklyJumpingJacks) : ℕ :=
  w.sunday + w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday

theorem kenny_must_do_at_least_three_on_thursday 
  (lastWeek : ℕ) 
  (thisWeek : WeeklyJumpingJacks) 
  (someDay : ℕ) :
  lastWeek = 324 →
  thisWeek.sunday = 34 →
  thisWeek.monday = 20 →
  thisWeek.tuesday = 0 →
  thisWeek.wednesday = 123 →
  thisWeek.saturday = 61 →
  (thisWeek.thursday = someDay ∨ thisWeek.friday = someDay) →
  someDay = 23 →
  weekTotal thisWeek > lastWeek →
  thisWeek.thursday ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_kenny_must_do_at_least_three_on_thursday_l1444_144468


namespace NUMINAMATH_CALUDE_cosine_sine_equality_l1444_144434

theorem cosine_sine_equality (α : ℝ) : 
  3.3998 * (Real.cos α)^4 - 4 * (Real.cos α)^3 - 8 * (Real.cos α)^2 + 3 * Real.cos α + 1 = 
  -2 * Real.sin (7 * α / 2) * Real.sin (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_equality_l1444_144434


namespace NUMINAMATH_CALUDE_inheritance_problem_l1444_144451

/-- The inheritance problem -/
theorem inheritance_problem (x : ℝ) 
  (h1 : 0.25 * x + 0.15 * x = 15000) : x = 37500 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_problem_l1444_144451


namespace NUMINAMATH_CALUDE_smallest_lpm_l1444_144410

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- Represents a two-digit number with repeating digits -/
def TwoDigitRepeating (d : Digit) := 10 * d.val + d.val

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Digit
  tens : Digit
  ones : Digit

/-- Converts a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds.val + 10 * n.tens.val + n.ones.val

/-- Checks if a natural number is a valid result of the multiplication -/
def isValidResult (l : Digit) (result : ThreeDigitNumber) : Prop :=
  (TwoDigitRepeating l) * l.val = result.toNat ∧
  result.hundreds = l

theorem smallest_lpm :
  ∃ (result : ThreeDigitNumber),
    (∃ (l : Digit), isValidResult l result) ∧
    (∀ (other : ThreeDigitNumber),
      (∃ (l : Digit), isValidResult l other) →
      result.toNat ≤ other.toNat) ∧
    result.toNat = 275 := by
  sorry

end NUMINAMATH_CALUDE_smallest_lpm_l1444_144410


namespace NUMINAMATH_CALUDE_expression_evaluation_l1444_144461

theorem expression_evaluation (a b : ℚ) (h1 : a = 5) (h2 : b = 6) : 
  (3 * b) / (a + b) = 18 / 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1444_144461


namespace NUMINAMATH_CALUDE_thirty_divides_p_squared_minus_one_l1444_144444

theorem thirty_divides_p_squared_minus_one (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  30 ∣ (p^2 - 1) ↔ p = 5 := by
  sorry

end NUMINAMATH_CALUDE_thirty_divides_p_squared_minus_one_l1444_144444


namespace NUMINAMATH_CALUDE_intersection_implies_a_gt_three_l1444_144437

/-- A function f(x) = x³ - ax² + 4 that intersects the positive x-axis at two different points -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4

/-- The property that f intersects the positive x-axis at two different points -/
def intersects_positive_x_axis_twice (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0

/-- If f(x) = x³ - ax² + 4 intersects the positive x-axis at two different points, then a > 3 -/
theorem intersection_implies_a_gt_three :
  ∀ a : ℝ, intersects_positive_x_axis_twice a → a > 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_gt_three_l1444_144437


namespace NUMINAMATH_CALUDE_binomial_coefficient_1500_2_l1444_144480

theorem binomial_coefficient_1500_2 : Nat.choose 1500 2 = 1124250 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1500_2_l1444_144480


namespace NUMINAMATH_CALUDE_stream_speed_l1444_144485

theorem stream_speed (upstream_distance : ℝ) (downstream_distance : ℝ) (time : ℝ) 
  (h1 : upstream_distance = 16)
  (h2 : downstream_distance = 24)
  (h3 : time = 4)
  (h4 : upstream_distance / time + downstream_distance / time = 10) :
  let stream_speed := (downstream_distance - upstream_distance) / (2 * time)
  stream_speed = 1 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1444_144485


namespace NUMINAMATH_CALUDE_parallel_tangents_l1444_144419

/-- A homogeneous differential equation y' = φ(y/x) -/
noncomputable def homogeneous_de (φ : ℝ → ℝ) (x y : ℝ) : ℝ := φ (y / x)

/-- The slope of the tangent line at a point (x, y) -/
noncomputable def tangent_slope (φ : ℝ → ℝ) (x y : ℝ) : ℝ := homogeneous_de φ x y

theorem parallel_tangents (φ : ℝ → ℝ) (x y x₁ y₁ : ℝ) (hx : x ≠ 0) (hx₁ : x₁ ≠ 0) 
  (h_corresp : y / x = y₁ / x₁) :
  tangent_slope φ x y = tangent_slope φ x₁ y₁ := by
  sorry

end NUMINAMATH_CALUDE_parallel_tangents_l1444_144419


namespace NUMINAMATH_CALUDE_tangent_line_inclination_l1444_144427

/-- The angle of inclination of the tangent line to y = x^3 - 2x + 4 at (1, 3) is 45°. -/
theorem tangent_line_inclination (f : ℝ → ℝ) (x₀ y₀ : ℝ) :
  f x = x^3 - 2*x + 4 →
  x₀ = 1 →
  y₀ = 3 →
  f x₀ = y₀ →
  HasDerivAt f (3*x₀^2 - 2) x₀ →
  (Real.arctan (3*x₀^2 - 2)) * (180 / Real.pi) = 45 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_inclination_l1444_144427


namespace NUMINAMATH_CALUDE_largest_integer_in_special_set_l1444_144478

theorem largest_integer_in_special_set (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d →                   -- Four different integers
  (a + b + c + d) / 4 = 70 →                -- Average is 70
  a ≥ 13 →                                  -- Smallest integer is at least 13
  d ≤ 238 :=                                -- Largest integer is at most 238
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_special_set_l1444_144478


namespace NUMINAMATH_CALUDE_xiaoqiang_father_annual_income_l1444_144439

def monthly_salary : ℕ := 4380
def months_in_year : ℕ := 12

theorem xiaoqiang_father_annual_income :
  monthly_salary * months_in_year = 52560 := by sorry

end NUMINAMATH_CALUDE_xiaoqiang_father_annual_income_l1444_144439


namespace NUMINAMATH_CALUDE_no_nontrivial_solution_x2_plus_y2_eq_3z2_l1444_144496

theorem no_nontrivial_solution_x2_plus_y2_eq_3z2 :
  ∀ (x y z : ℤ), x^2 + y^2 = 3 * z^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nontrivial_solution_x2_plus_y2_eq_3z2_l1444_144496


namespace NUMINAMATH_CALUDE_billy_ice_cubes_l1444_144479

/-- Calculates the total number of ice cubes that can be made given the tray capacity and number of trays. -/
def total_ice_cubes (tray_capacity : ℕ) (num_trays : ℕ) : ℕ :=
  tray_capacity * num_trays

/-- Proves that with a tray capacity of 48 ice cubes and 24 trays, the total number of ice cubes is 1152. -/
theorem billy_ice_cubes : total_ice_cubes 48 24 = 1152 := by
  sorry

end NUMINAMATH_CALUDE_billy_ice_cubes_l1444_144479


namespace NUMINAMATH_CALUDE_expression_equality_l1444_144470

theorem expression_equality (y a : ℝ) (h1 : y > 0) 
  (h2 : (a * y) / 20 + (3 * y) / 10 = 0.5 * y) : a = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1444_144470


namespace NUMINAMATH_CALUDE_parabola_line_intersection_midpoint_line_equation_min_distance_product_parabola_line_intersection_properties_l1444_144473

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line passing through P(-2, 2)
def line (k : ℝ) (x y : ℝ) : Prop := y = k*(x + 2) + 2

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define the distance from a point to the focus
def distToFocus (x y : ℝ) : ℝ := y + 1

theorem parabola_line_intersection (k : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    x₁ ≠ x₂ := by sorry

theorem midpoint_line_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line (-1) x₁ y₁ ∧ line (-1) x₂ y₂ ∧
    x₁ + x₂ = -4 ∧ y₁ + y₂ = 4 := by sorry

theorem min_distance_product :
  ∃ (k : ℝ),
    ∀ (x₁ y₁ x₂ y₂ : ℝ),
      parabola x₁ y₁ → parabola x₂ y₂ →
      line k x₁ y₁ → line k x₂ y₂ →
      distToFocus x₁ y₁ * distToFocus x₂ y₂ ≥ 9/2 := by sorry

-- Main theorems to prove
theorem parabola_line_intersection_properties :
  -- 1) When P(-2, 2) is the midpoint of AB, the equation of line AB is x + y = 0
  (∀ (x y : ℝ), line (-1) x y ↔ x + y = 0) ∧
  -- 2) The minimum value of |AF|•|BF| is 9/2
  (∃ (k : ℝ),
    ∀ (x₁ y₁ x₂ y₂ : ℝ),
      parabola x₁ y₁ → parabola x₂ y₂ →
      line k x₁ y₁ → line k x₂ y₂ →
      distToFocus x₁ y₁ * distToFocus x₂ y₂ = 9/2) := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_midpoint_line_equation_min_distance_product_parabola_line_intersection_properties_l1444_144473


namespace NUMINAMATH_CALUDE_contractor_labor_problem_l1444_144447

theorem contractor_labor_problem (planned_days : ℕ) (absent_workers : ℕ) (actual_days : ℕ) 
  (h1 : planned_days = 9)
  (h2 : absent_workers = 6)
  (h3 : actual_days = 15) :
  ∃ (original_workers : ℕ), 
    original_workers * planned_days = (original_workers - absent_workers) * actual_days ∧ 
    original_workers = 15 := by
  sorry


end NUMINAMATH_CALUDE_contractor_labor_problem_l1444_144447


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1444_144432

-- Define the arithmetic sequence
def arithmetic_sequence (n : ℕ+) : ℚ := 2 * n - 1

-- Define the sum of the first n terms
def S (n : ℕ+) : ℚ := n * (arithmetic_sequence 1 + arithmetic_sequence n) / 2

-- Define b_n
def b (n : ℕ+) : ℚ := 1 / (arithmetic_sequence (n + 1) * arithmetic_sequence (n + 2))

-- Define T_n
def T (n : ℕ+) : ℚ := (Finset.range n).sum (λ i => b ⟨i + 1, Nat.succ_pos i⟩)

-- Theorem statement
theorem arithmetic_sequence_properties :
  (arithmetic_sequence 1 + arithmetic_sequence 13 = 26) ∧
  (S 9 = 81) →
  (∀ n : ℕ+, arithmetic_sequence n = 2 * n - 1) ∧
  (∀ n : ℕ+, T n = n / (3 * (2 * n + 3))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1444_144432


namespace NUMINAMATH_CALUDE_percentage_of_c_grades_l1444_144405

def grading_scale : List (String × (Int × Int)) :=
  [("A", (95, 100)), ("B", (88, 94)), ("C", (78, 87)), ("D", (65, 77)), ("F", (0, 64))]

def scores : List Int :=
  [94, 65, 59, 99, 82, 89, 90, 68, 79, 62, 85, 81, 64, 83, 91]

def is_grade_c (score : Int) : Bool :=
  78 ≤ score ∧ score ≤ 87

def count_grade_c (scores : List Int) : Nat :=
  (scores.filter is_grade_c).length

theorem percentage_of_c_grades :
  (count_grade_c scores : Rat) / (scores.length : Rat) * 100 = 100/3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_c_grades_l1444_144405


namespace NUMINAMATH_CALUDE_percentage_calculation_correct_l1444_144492

/-- The total number of students in the class -/
def total_students : ℕ := 30

/-- The number of students scoring in the 70%-79% range -/
def students_in_range : ℕ := 8

/-- The percentage of students scoring in the 70%-79% range -/
def percentage_in_range : ℚ := 26.67

/-- Theorem stating that the percentage of students scoring in the 70%-79% range is correct -/
theorem percentage_calculation_correct : 
  (students_in_range : ℚ) / (total_students : ℚ) * 100 = percentage_in_range := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_correct_l1444_144492


namespace NUMINAMATH_CALUDE_jason_football_games_l1444_144443

/-- Given the number of football games Jason attended this month and last month,
    and the total number of games he plans to attend,
    prove that the number of games he plans to attend next month is 16. -/
theorem jason_football_games (this_month last_month total : ℕ) 
    (h1 : this_month = 11)
    (h2 : last_month = 17)
    (h3 : total = 44) :
    total - (this_month + last_month) = 16 := by
  sorry

end NUMINAMATH_CALUDE_jason_football_games_l1444_144443


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1444_144498

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) ∧ 
  (∃ x : ℝ, |x - 1| < 2 ∧ x * (x - 3) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1444_144498


namespace NUMINAMATH_CALUDE_impossible_to_gather_all_stones_l1444_144455

/-- Represents the number of stones in each pile -/
structure PileState :=
  (pile1 : Nat) (pile2 : Nat) (pile3 : Nat)

/-- Represents a valid move -/
inductive Move
  | move12 : Move  -- Move from pile 1 and 2 to 3
  | move13 : Move  -- Move from pile 1 and 3 to 2
  | move23 : Move  -- Move from pile 2 and 3 to 1

/-- Apply a move to a PileState -/
def applyMove (state : PileState) (move : Move) : PileState :=
  match move with
  | Move.move12 => PileState.mk (state.pile1 - 1) (state.pile2 - 1) (state.pile3 + 2)
  | Move.move13 => PileState.mk (state.pile1 - 1) (state.pile2 + 2) (state.pile3 - 1)
  | Move.move23 => PileState.mk (state.pile1 + 2) (state.pile2 - 1) (state.pile3 - 1)

/-- Check if all stones are in one pile -/
def isAllInOnePile (state : PileState) : Prop :=
  (state.pile1 = 0 ∧ state.pile2 = 0) ∨
  (state.pile1 = 0 ∧ state.pile3 = 0) ∨
  (state.pile2 = 0 ∧ state.pile3 = 0)

/-- Initial state of the piles -/
def initialState : PileState := PileState.mk 20 1 9

/-- Theorem stating it's impossible to gather all stones in one pile -/
theorem impossible_to_gather_all_stones :
  ¬∃ (moves : List Move), isAllInOnePile (moves.foldl applyMove initialState) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_gather_all_stones_l1444_144455


namespace NUMINAMATH_CALUDE_joan_lost_balloons_l1444_144426

/-- Given that Joan initially had 8 orange balloons and now has 6,
    prove that she lost 2 balloons. -/
theorem joan_lost_balloons (initial : ℕ) (current : ℕ) (h1 : initial = 8) (h2 : current = 6) :
  initial - current = 2 := by
  sorry

end NUMINAMATH_CALUDE_joan_lost_balloons_l1444_144426


namespace NUMINAMATH_CALUDE_savings_duration_l1444_144424

/-- Proves that saving $34 daily for a total of $12,410 results in 365 days of savings -/
theorem savings_duration (daily_savings : ℕ) (total_savings : ℕ) (days : ℕ) :
  daily_savings = 34 →
  total_savings = 12410 →
  total_savings = daily_savings * days →
  days = 365 := by
sorry

end NUMINAMATH_CALUDE_savings_duration_l1444_144424


namespace NUMINAMATH_CALUDE_edith_books_count_l1444_144487

theorem edith_books_count : ∀ (novels : ℕ) (writing_books : ℕ),
  novels = 80 →
  writing_books = 2 * novels →
  novels + writing_books = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_edith_books_count_l1444_144487


namespace NUMINAMATH_CALUDE_max_area_rectangular_prism_volume_l1444_144491

/-- The volume of a rectangular prism with maximum base area -/
theorem max_area_rectangular_prism_volume
  (base_perimeter : ℝ)
  (height : ℝ)
  (h_base_perimeter : base_perimeter = 32)
  (h_height : height = 9)
  (h_max_area : ∀ (l w : ℝ), l + w = base_perimeter / 2 → l * w ≤ (base_perimeter / 4) ^ 2) :
  (base_perimeter / 4) ^ 2 * height = 576 :=
sorry

end NUMINAMATH_CALUDE_max_area_rectangular_prism_volume_l1444_144491


namespace NUMINAMATH_CALUDE_sara_onions_l1444_144466

theorem sara_onions (sally_onions fred_onions total_onions : ℕ) 
  (h1 : sally_onions = 5)
  (h2 : fred_onions = 9)
  (h3 : total_onions = 18)
  : total_onions - (sally_onions + fred_onions) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sara_onions_l1444_144466


namespace NUMINAMATH_CALUDE_power_of_power_three_l1444_144421

theorem power_of_power_three : (3^3)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l1444_144421


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1444_144460

theorem fraction_equivalence (x : ℝ) (h : x ≠ 5) :
  ¬(∀ x : ℝ, x ≠ 5 → (x + 3) / (x - 5) = 3 / (-5)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1444_144460


namespace NUMINAMATH_CALUDE_chris_age_l1444_144471

theorem chris_age (a b c : ℕ) : 
  (a + b + c) / 3 = 12 →
  b - 5 = 2 * (c + 2) →
  b + 3 = a + 3 →
  c = 4 :=
by sorry

end NUMINAMATH_CALUDE_chris_age_l1444_144471


namespace NUMINAMATH_CALUDE_eds_initial_money_l1444_144474

def night_rate : ℚ := 1.5
def morning_rate : ℚ := 2
def night_hours : ℕ := 6
def morning_hours : ℕ := 4
def remaining_money : ℚ := 63

theorem eds_initial_money :
  night_rate * night_hours + morning_rate * morning_hours + remaining_money = 80 := by
  sorry

end NUMINAMATH_CALUDE_eds_initial_money_l1444_144474


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_one_l1444_144469

def h (t : ℝ) : ℝ := -4.9 * t^2 + 10 * t

theorem instantaneous_velocity_at_one :
  (deriv h) 1 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_one_l1444_144469


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1444_144417

theorem right_triangle_third_side 
  (a b : ℝ) 
  (h1 : Real.sqrt (a - 3) + |b - 4| = 0) : 
  ∃ c : ℝ, (c = 5 ∨ c = Real.sqrt 7) ∧ 
    ((a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2) ∨ (b^2 + c^2 = a^2)) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1444_144417


namespace NUMINAMATH_CALUDE_intersection_implies_a_less_than_two_l1444_144464

def A : Set ℝ := {1}
def B (a : ℝ) : Set ℝ := {x | a - 2*x < 0}

theorem intersection_implies_a_less_than_two (a : ℝ) : 
  (A ∩ B a).Nonempty → a < 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_less_than_two_l1444_144464


namespace NUMINAMATH_CALUDE_blanch_breakfast_slices_l1444_144440

/-- The number of pizza slices Blanch ate for breakfast -/
def breakfast_slices : ℕ := sorry

/-- The total number of pizza slices Blanch started with -/
def total_slices : ℕ := 15

/-- The number of pizza slices Blanch ate for lunch -/
def lunch_slices : ℕ := 2

/-- The number of pizza slices Blanch ate as a snack -/
def snack_slices : ℕ := 2

/-- The number of pizza slices Blanch ate for dinner -/
def dinner_slices : ℕ := 5

/-- The number of pizza slices left at the end -/
def leftover_slices : ℕ := 2

/-- Theorem stating that Blanch ate 4 slices for breakfast -/
theorem blanch_breakfast_slices : 
  breakfast_slices = total_slices - (lunch_slices + snack_slices + dinner_slices + leftover_slices) :=
by sorry

end NUMINAMATH_CALUDE_blanch_breakfast_slices_l1444_144440


namespace NUMINAMATH_CALUDE_interview_problem_l1444_144484

/-- The number of people to be hired -/
def people_hired : ℕ := 3

/-- The probability of two specific individuals being hired together -/
def prob_two_hired : ℚ := 1 / 70

/-- The total number of people interviewed -/
def total_interviewed : ℕ := 21

theorem interview_problem :
  (people_hired = 3) →
  (prob_two_hired = 1 / 70) →
  (total_interviewed = 21) := by
  sorry

end NUMINAMATH_CALUDE_interview_problem_l1444_144484


namespace NUMINAMATH_CALUDE_gear_rotation_l1444_144409

theorem gear_rotation (teeth_A teeth_B turns_A : ℕ) (h1 : teeth_A = 6) (h2 : teeth_B = 8) (h3 : turns_A = 12) :
  teeth_A * turns_A = teeth_B * (teeth_A * turns_A / teeth_B) :=
sorry

end NUMINAMATH_CALUDE_gear_rotation_l1444_144409


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1444_144429

theorem arithmetic_equality : 3 * 9 + 4 * 10 + 11 * 3 + 3 * 8 = 124 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1444_144429


namespace NUMINAMATH_CALUDE_largest_constant_divisor_inequality_l1444_144438

/-- The number of divisors function -/
def tau (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The statement of the theorem -/
theorem largest_constant_divisor_inequality :
  (∃ (c : ℝ), c > 0 ∧
    (∀ (n : ℕ), n ≥ 2 →
      (∃ (d : ℕ), d > 0 ∧ d ∣ n ∧ (d : ℝ) ≤ Real.sqrt n ∧
        (tau d : ℝ) ≥ c * Real.sqrt (tau n : ℝ)))) ∧
  (∀ (c : ℝ), c > Real.sqrt (1 / 2) →
    (∃ (n : ℕ), n ≥ 2 ∧
      (∀ (d : ℕ), d > 0 → d ∣ n → (d : ℝ) ≤ Real.sqrt n →
        (tau d : ℝ) < c * Real.sqrt (tau n : ℝ)))) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_divisor_inequality_l1444_144438


namespace NUMINAMATH_CALUDE_projection_parallel_condition_l1444_144462

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a 3D line
  -- (simplified for this example)

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane
  -- (simplified for this example)

/-- Projection of a line onto a plane -/
def project (l : Line3D) (p : Plane3D) : Line3D :=
  sorry -- Definition of projection

/-- Parallel lines -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry -- Definition of parallel lines

theorem projection_parallel_condition 
  (a b m n : Line3D) (α : Plane3D) 
  (h1 : a ≠ b)
  (h2 : m = project a α)
  (h3 : n = project b α)
  (h4 : m ≠ n) :
  (∀ (a b : Line3D), parallel a b → parallel (project a α) (project b α)) ∧
  (∃ (a b : Line3D), parallel (project a α) (project b α) ∧ ¬parallel a b) :=
sorry

end NUMINAMATH_CALUDE_projection_parallel_condition_l1444_144462


namespace NUMINAMATH_CALUDE_least_number_with_remainder_five_forty_five_satisfies_least_number_is_545_l1444_144416

theorem least_number_with_remainder (n : ℕ) : 
  (n % 12 = 5 ∧ n % 15 = 5 ∧ n % 20 = 5 ∧ n % 54 = 5) → n ≥ 545 := by
  sorry

theorem five_forty_five_satisfies :
  545 % 12 = 5 ∧ 545 % 15 = 5 ∧ 545 % 20 = 5 ∧ 545 % 54 = 5 := by
  sorry

theorem least_number_is_545 : 
  ∃! n : ℕ, (n % 12 = 5 ∧ n % 15 = 5 ∧ n % 20 = 5 ∧ n % 54 = 5) ∧
  ∀ m : ℕ, (m % 12 = 5 ∧ m % 15 = 5 ∧ m % 20 = 5 ∧ m % 54 = 5) → m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_five_forty_five_satisfies_least_number_is_545_l1444_144416


namespace NUMINAMATH_CALUDE_ava_remaining_distance_l1444_144411

/-- The remaining distance for Ava to finish the race -/
def remaining_distance (race_length : ℕ) (distance_covered : ℕ) : ℕ :=
  race_length - distance_covered

/-- Proof that Ava's remaining distance is 167 meters -/
theorem ava_remaining_distance :
  remaining_distance 1000 833 = 167 := by
  sorry

end NUMINAMATH_CALUDE_ava_remaining_distance_l1444_144411


namespace NUMINAMATH_CALUDE_tax_calculation_l1444_144448

/-- Calculates the annual income before tax given tax rates and differential savings -/
def annual_income_before_tax (original_rate new_rate : ℚ) (differential_savings : ℚ) : ℚ :=
  differential_savings / (original_rate - new_rate)

/-- Theorem stating that given the specified tax rates and differential savings, 
    the annual income before tax is $34,500 -/
theorem tax_calculation (original_rate new_rate differential_savings : ℚ) 
  (h1 : original_rate = 42 / 100)
  (h2 : new_rate = 28 / 100)
  (h3 : differential_savings = 4830) :
  annual_income_before_tax original_rate new_rate differential_savings = 34500 := by
  sorry

#eval annual_income_before_tax (42/100) (28/100) 4830

end NUMINAMATH_CALUDE_tax_calculation_l1444_144448


namespace NUMINAMATH_CALUDE_sum_of_diagonals_is_190_l1444_144400

/-- A hexagon inscribed in a circle -/
structure InscribedHexagon where
  -- Sides of the hexagon
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  side6 : ℝ
  -- Conditions on the sides
  h1 : side1 = 20
  h2 : side3 = 30
  h3 : side2 = 50
  h4 : side4 = 50
  h5 : side5 = 50
  h6 : side6 = 50

/-- The sum of diagonals from one vertex in the inscribed hexagon -/
def sumOfDiagonals (h : InscribedHexagon) : ℝ := sorry

/-- Theorem: The sum of diagonals from one vertex in the specified hexagon is 190 -/
theorem sum_of_diagonals_is_190 (h : InscribedHexagon) : sumOfDiagonals h = 190 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_diagonals_is_190_l1444_144400


namespace NUMINAMATH_CALUDE_probability_three_white_two_black_l1444_144457

/-- The number of white balls in the box -/
def white_balls : ℕ := 8

/-- The number of black balls in the box -/
def black_balls : ℕ := 9

/-- The total number of balls drawn -/
def drawn_balls : ℕ := 5

/-- The number of white balls drawn -/
def white_drawn : ℕ := 3

/-- The number of black balls drawn -/
def black_drawn : ℕ := 2

/-- The probability of drawing 3 white balls and 2 black balls -/
theorem probability_three_white_two_black :
  (Nat.choose white_balls white_drawn * Nat.choose black_balls black_drawn : ℚ) /
  (Nat.choose (white_balls + black_balls) drawn_balls : ℚ) = 672 / 2063 :=
sorry

end NUMINAMATH_CALUDE_probability_three_white_two_black_l1444_144457


namespace NUMINAMATH_CALUDE_logarithm_and_exponential_equalities_l1444_144408

theorem logarithm_and_exponential_equalities :
  (Real.log 9 / Real.log 6 + 2 * Real.log 2 / Real.log 6 = 2) ∧
  (Real.exp 0 + Real.sqrt ((1 - Real.sqrt 2)^2) - 8^(1/6) = 1 + Real.sqrt 5 - Real.sqrt 2 - 2^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_and_exponential_equalities_l1444_144408


namespace NUMINAMATH_CALUDE_zack_classroom_count_l1444_144403

/-- The number of students in each classroom -/
structure ClassroomCounts where
  tina : ℕ
  maura : ℕ
  zack : ℕ

/-- The conditions of the problem -/
def classroom_problem (c : ClassroomCounts) : Prop :=
  c.tina = c.maura ∧
  c.zack = (c.tina + c.maura) / 2 ∧
  c.tina + c.maura + c.zack = 69

/-- The theorem to prove -/
theorem zack_classroom_count (c : ClassroomCounts) : 
  classroom_problem c → c.zack = 23 := by
  sorry

end NUMINAMATH_CALUDE_zack_classroom_count_l1444_144403


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_inequality_l1444_144475

theorem sum_of_fourth_powers_inequality (x y z : ℝ) 
  (h : x^2 + y^2 + z^2 + 9 = 4*(x + y + z)) :
  x^4 + y^4 + z^4 + 16*(x^2 + y^2 + z^2) ≥ 8*(x^3 + y^3 + z^3) + 27 ∧
  (x^4 + y^4 + z^4 + 16*(x^2 + y^2 + z^2) = 8*(x^3 + y^3 + z^3) + 27 ↔ 
   (x = 1 ∨ x = 3) ∧ (y = 1 ∨ y = 3) ∧ (z = 1 ∨ z = 3)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_inequality_l1444_144475


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l1444_144454

theorem unique_solution_quadratic_inequality (a : ℝ) : 
  (∃! x : ℝ, |x^2 + 2*a*x + 4*a| ≤ 4) ↔ a = 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l1444_144454


namespace NUMINAMATH_CALUDE_arccos_cos_eight_l1444_144476

theorem arccos_cos_eight : Real.arccos (Real.cos 8) = 8 - 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_arccos_cos_eight_l1444_144476


namespace NUMINAMATH_CALUDE_integer_representation_l1444_144472

theorem integer_representation (k : ℤ) (h : -1985 ≤ k ∧ k ≤ 1985) :
  ∃ (a : Fin 8 → ℤ), (∀ i, a i ∈ ({-1, 0, 1} : Set ℤ)) ∧
    k = (a 0) * 1 + (a 1) * 3 + (a 2) * 9 + (a 3) * 27 +
        (a 4) * 81 + (a 5) * 243 + (a 6) * 729 + (a 7) * 2187 :=
by sorry

end NUMINAMATH_CALUDE_integer_representation_l1444_144472


namespace NUMINAMATH_CALUDE_circle_symmetry_line_coefficient_product_l1444_144418

/-- Given a circle and a line, prove that the product of the line's coefficients is non-positive -/
theorem circle_symmetry_line_coefficient_product (a b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0 →
    ∃ x' y' : ℝ, x'^2 + y'^2 + 2*x' - 4*y' + 1 = 0 ∧
      2*a*x - b*y + 2 = 2*a*x' - b*y' + 2) →
  a * b ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_coefficient_product_l1444_144418


namespace NUMINAMATH_CALUDE_distance_table_1_to_3_l1444_144406

/-- Calculates the distance between the first and third table in a relay race. -/
def distance_between_tables_1_and_3 (race_length : ℕ) (num_tables : ℕ) : ℕ :=
  2 * (race_length / num_tables)

/-- Proves that in a 1200-meter race with 6 equally spaced tables, 
    the distance between the first and third table is 400 meters. -/
theorem distance_table_1_to_3 : 
  distance_between_tables_1_and_3 1200 6 = 400 := by
  sorry

end NUMINAMATH_CALUDE_distance_table_1_to_3_l1444_144406


namespace NUMINAMATH_CALUDE_power_function_value_l1444_144401

theorem power_function_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x = x ^ a) →
  f 2 = Real.sqrt 2 / 2 →
  f 4 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l1444_144401


namespace NUMINAMATH_CALUDE_circumradius_of_specific_trapezoid_l1444_144452

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  longBase : ℝ
  shortBase : ℝ
  lateralSide : ℝ

/-- The radius of the circumscribed circle of an isosceles trapezoid -/
def circumradius (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The radius of the circumscribed circle of the given isosceles trapezoid is 5√2 -/
theorem circumradius_of_specific_trapezoid :
  let t : IsoscelesTrapezoid := ⟨14, 2, 10⟩
  circumradius t = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circumradius_of_specific_trapezoid_l1444_144452


namespace NUMINAMATH_CALUDE_min_value_sum_l1444_144453

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2 / x) + (8 / y) = 1) : x + y ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l1444_144453


namespace NUMINAMATH_CALUDE_cube_square_third_prime_times_fourth_prime_l1444_144407

def third_smallest_prime : Nat := 5

def fourth_smallest_prime : Nat := 7

theorem cube_square_third_prime_times_fourth_prime :
  (third_smallest_prime ^ 2) ^ 3 * fourth_smallest_prime = 109375 := by
  sorry

end NUMINAMATH_CALUDE_cube_square_third_prime_times_fourth_prime_l1444_144407


namespace NUMINAMATH_CALUDE_ten_times_a_l1444_144499

theorem ten_times_a (a : ℝ) (h : a = 6) : 10 * a = 60 := by
  sorry

end NUMINAMATH_CALUDE_ten_times_a_l1444_144499


namespace NUMINAMATH_CALUDE_initial_water_calculation_l1444_144446

/-- The amount of water initially poured into the pool -/
def initial_amount : ℝ := 1

/-- The amount of water added later -/
def added_amount : ℝ := 8.8

/-- The total amount of water in the pool -/
def total_amount : ℝ := 9.8

/-- Theorem stating that the initial amount plus the added amount equals the total amount -/
theorem initial_water_calculation :
  initial_amount + added_amount = total_amount := by
  sorry

end NUMINAMATH_CALUDE_initial_water_calculation_l1444_144446


namespace NUMINAMATH_CALUDE_range_of_m_l1444_144420

-- Define the conditions
def p (x : ℝ) : Prop := (x + 2) * (x - 10) ≤ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, q x m → p x) →
  (∃ x, p x ∧ ¬q x m) →
  (0 < m ∧ m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1444_144420


namespace NUMINAMATH_CALUDE_earnings_calculation_l1444_144494

/-- Calculates the discounted price for a given quantity and unit price with a discount rate and minimum quantity for discount --/
def discountedPrice (quantity : ℕ) (unitPrice : ℚ) (discountRate : ℚ) (minQuantity : ℕ) : ℚ :=
  if quantity ≥ minQuantity then
    (1 - discountRate) * (quantity : ℚ) * unitPrice
  else
    (quantity : ℚ) * unitPrice

/-- Calculates the total earnings after all discounts --/
def totalEarnings (smallQuantity mediumQuantity largeQuantity extraLargeQuantity : ℕ) : ℚ :=
  let smallPrice := discountedPrice smallQuantity (30 : ℚ) (1/10 : ℚ) 4
  let mediumPrice := discountedPrice mediumQuantity (45 : ℚ) (3/20 : ℚ) 3
  let largePrice := discountedPrice largeQuantity (60 : ℚ) (1/20 : ℚ) 6
  let extraLargePrice := discountedPrice extraLargeQuantity (85 : ℚ) (2/25 : ℚ) 2
  let subtotal := smallPrice + mediumPrice + largePrice + extraLargePrice
  if smallQuantity + mediumQuantity ≥ 10 then
    (97/100 : ℚ) * subtotal
  else
    subtotal

theorem earnings_calculation (smallQuantity mediumQuantity largeQuantity extraLargeQuantity : ℕ) :
  smallQuantity = 8 ∧ mediumQuantity = 11 ∧ largeQuantity = 4 ∧ extraLargeQuantity = 3 →
  totalEarnings smallQuantity mediumQuantity largeQuantity extraLargeQuantity = (1078.01 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_earnings_calculation_l1444_144494


namespace NUMINAMATH_CALUDE_prime_residue_theorem_l1444_144489

/-- Definition of suitable triple -/
def suitable (p : ℕ) (a b c : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  a % p ≠ b % p ∧ b % p ≠ c % p ∧ a % p ≠ c % p

/-- Definition of f_k function -/
def f_k (p k a b c : ℕ) : ℤ :=
  a * (b - c)^(p - k) + b * (c - a)^(p - k) + c * (a - b)^(p - k)

theorem prime_residue_theorem (p : ℕ) (hp : p.Prime) (hp11 : p ≥ 11) :
  (∃ a b c : ℕ, suitable p a b c ∧ (p : ℤ) ∣ f_k p 2 a b c) ∧
  (∀ a b c : ℕ, suitable p a b c → (p : ℤ) ∣ f_k p 2 a b c →
    (∃ k : ℕ, k ≥ 3 ∧ ¬((p : ℤ) ∣ f_k p k a b c))) ∧
  (∀ a b c : ℕ, suitable p a b c → (p : ℤ) ∣ f_k p 2 a b c →
    (∀ k : ℕ, k ≥ 3 → k < 4 → (p : ℤ) ∣ f_k p k a b c)) :=
sorry

end NUMINAMATH_CALUDE_prime_residue_theorem_l1444_144489


namespace NUMINAMATH_CALUDE_scientific_notation_14800_l1444_144422

theorem scientific_notation_14800 :
  14800 = 1.48 * (10 : ℝ)^4 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_14800_l1444_144422


namespace NUMINAMATH_CALUDE_problem_solution_l1444_144463

theorem problem_solution (x y : ℝ) 
  (h1 : Real.sqrt (3 + Real.sqrt x) = 4) 
  (h2 : x + y = 58) : 
  y = -111 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1444_144463


namespace NUMINAMATH_CALUDE_fraction_simplification_l1444_144467

theorem fraction_simplification (x : ℝ) (h : x ≠ 1) : 
  (2 / (1 - x)) - ((2 * x) / (1 - x)) = 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1444_144467


namespace NUMINAMATH_CALUDE_simplify_expression_l1444_144495

theorem simplify_expression (x : ℝ) (h : x^2 ≥ 49) :
  (7 - Real.sqrt (x^2 - 49))^2 = x^2 - 14 * Real.sqrt (x^2 - 49) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1444_144495


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l1444_144412

-- Define points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the condition for point P
def P_condition (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  2 * Real.sqrt ((x - 1)^2 + y^2) = 2 * (x + 1)

-- Define the trajectory equation
def trajectory_equation (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  y^2 = 4 * x

-- Define the intersection points M and N
def intersection_points (m : ℝ) (M N : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  y₁ = x₁ + m ∧ y₂ = x₂ + m ∧
  trajectory_equation M ∧ trajectory_equation N ∧
  m ≠ 0

-- Define the perpendicularity condition
def perpendicular (O M N : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem trajectory_and_intersection :
  (∀ P, P_condition P → trajectory_equation P) ∧
  (∀ m M N, intersection_points m M N → perpendicular (0, 0) M N → m = -4) := by
  sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l1444_144412


namespace NUMINAMATH_CALUDE_prob_b_is_three_fourths_l1444_144449

/-- The probability that either A or B solves a problem, given their individual probabilities -/
def prob_either_solves (prob_a prob_b : ℝ) : ℝ :=
  prob_a + prob_b - prob_a * prob_b

/-- Theorem stating that if A's probability is 2/3 and the probability of either A or B solving
    is 0.9166666666666666, then B's probability is 3/4 -/
theorem prob_b_is_three_fourths (prob_a prob_b : ℝ) 
    (h1 : prob_a = 2/3)
    (h2 : prob_either_solves prob_a prob_b = 0.9166666666666666) :
    prob_b = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_prob_b_is_three_fourths_l1444_144449


namespace NUMINAMATH_CALUDE_total_players_count_l1444_144483

theorem total_players_count (kabadi : ℕ) (kho_kho_only : ℕ) (both : ℕ) : 
  kabadi = 10 → kho_kho_only = 30 → both = 5 → 
  kabadi + kho_kho_only - both = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_players_count_l1444_144483


namespace NUMINAMATH_CALUDE_evie_shell_collection_l1444_144425

/-- The number of shells Evie collects per day -/
def shells_per_day : ℕ := 10

/-- The number of shells Evie gives to her brother -/
def shells_given : ℕ := 2

/-- The number of shells Evie has left after giving some to her brother -/
def shells_left : ℕ := 58

/-- The number of days Evie collected shells -/
def collection_days : ℕ := 6

theorem evie_shell_collection :
  shells_per_day * collection_days - shells_given = shells_left :=
by sorry

end NUMINAMATH_CALUDE_evie_shell_collection_l1444_144425


namespace NUMINAMATH_CALUDE_bicycle_problem_solution_l1444_144436

/-- Represents the bicycle sales and inventory problem over three days -/
def bicycle_problem (S1 S2 S3 B1 B2 B3 P1 P2 P3 C1 C2 C3 : ℕ) : Prop :=
  let sale_profit1 := S1 * P1
  let sale_profit2 := S2 * P2
  let sale_profit3 := S3 * P3
  let repair_cost1 := B1 * C1
  let repair_cost2 := B2 * C2
  let repair_cost3 := B3 * C3
  let net_profit1 := sale_profit1 - repair_cost1
  let net_profit2 := sale_profit2 - repair_cost2
  let net_profit3 := sale_profit3 - repair_cost3
  let total_net_profit := net_profit1 + net_profit2 + net_profit3
  let net_increase := (B1 - S1) + (B2 - S2) + (B3 - S3)
  (S1 = 10 ∧ S2 = 12 ∧ S3 = 9 ∧
   B1 = 15 ∧ B2 = 8 ∧ B3 = 11 ∧
   P1 = 250 ∧ P2 = 275 ∧ P3 = 260 ∧
   C1 = 100 ∧ C2 = 110 ∧ C3 = 120) →
  (total_net_profit = 4440 ∧ net_increase = 3)

theorem bicycle_problem_solution :
  ∀ S1 S2 S3 B1 B2 B3 P1 P2 P3 C1 C2 C3 : ℕ,
  bicycle_problem S1 S2 S3 B1 B2 B3 P1 P2 P3 C1 C2 C3 :=
sorry

end NUMINAMATH_CALUDE_bicycle_problem_solution_l1444_144436


namespace NUMINAMATH_CALUDE_team_combinations_l1444_144435

-- Define the number of people and team size
def total_people : ℕ := 7
def team_size : ℕ := 4

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem team_combinations : combination total_people team_size = 35 := by
  sorry

end NUMINAMATH_CALUDE_team_combinations_l1444_144435


namespace NUMINAMATH_CALUDE_problem_statement_l1444_144433

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1444_144433


namespace NUMINAMATH_CALUDE_max_kitchen_towel_sets_is_13_l1444_144488

-- Define the given parameters
def budget : ℚ := 600
def guest_bathroom_sets : ℕ := 2
def master_bathroom_sets : ℕ := 4
def hand_towel_sets : ℕ := 3
def guest_bathroom_price : ℚ := 40
def master_bathroom_price : ℚ := 50
def hand_towel_price : ℚ := 30
def kitchen_towel_price : ℚ := 20
def guest_bathroom_discount : ℚ := 0.15
def master_bathroom_discount : ℚ := 0.20
def hand_towel_discount : ℚ := 0.15
def kitchen_towel_discount : ℚ := 0.10
def sales_tax : ℚ := 0.08

-- Define the function to calculate the maximum number of kitchen towel sets
def max_kitchen_towel_sets : ℕ :=
  let guest_bathroom_cost := guest_bathroom_sets * guest_bathroom_price * (1 - guest_bathroom_discount)
  let master_bathroom_cost := master_bathroom_sets * master_bathroom_price * (1 - master_bathroom_discount)
  let hand_towel_cost := hand_towel_sets * hand_towel_price * (1 - hand_towel_discount)
  let total_cost_before_tax := guest_bathroom_cost + master_bathroom_cost + hand_towel_cost
  let total_cost_after_tax := total_cost_before_tax * (1 + sales_tax)
  let remaining_budget := budget - total_cost_after_tax
  let kitchen_towel_cost_after_tax := kitchen_towel_price * (1 - kitchen_towel_discount) * (1 + sales_tax)
  (remaining_budget / kitchen_towel_cost_after_tax).floor.toNat

-- Theorem statement
theorem max_kitchen_towel_sets_is_13 : max_kitchen_towel_sets = 13 := by
  sorry


end NUMINAMATH_CALUDE_max_kitchen_towel_sets_is_13_l1444_144488


namespace NUMINAMATH_CALUDE_walk_distance_l1444_144456

theorem walk_distance (x y : ℝ) : 
  x > 0 → y > 0 → 
  (x^2 + y^2 - x*y = 9) → 
  x = 3 :=
by sorry

end NUMINAMATH_CALUDE_walk_distance_l1444_144456


namespace NUMINAMATH_CALUDE_sophomore_sample_count_l1444_144430

/-- Given a school with 1000 students, including 320 sophomores,
    prove that a random sample of 200 students will contain 64 sophomores. -/
theorem sophomore_sample_count (total_students : ℕ) (sophomores : ℕ) (sample_size : ℕ) :
  total_students = 1000 →
  sophomores = 320 →
  sample_size = 200 →
  (sophomores : ℚ) / total_students * sample_size = 64 := by
  sorry

end NUMINAMATH_CALUDE_sophomore_sample_count_l1444_144430


namespace NUMINAMATH_CALUDE_banner_coverage_count_l1444_144414

/-- A banner is a 2x5 grid with one 1x1 square removed from one of the four corners -/
def Banner : Type := Unit

/-- The grid table dimensions -/
def grid_width : Nat := 18
def grid_height : Nat := 9

/-- The number of banners used to cover the grid -/
def num_banners : Nat := 18

/-- The number of squares in each banner -/
def squares_per_banner : Nat := 9

/-- The number of pairs of banners -/
def num_pairs : Nat := 9

theorem banner_coverage_count : 
  (2 ^ num_pairs : Nat) + (2 ^ num_pairs : Nat) = 1024 := by sorry

end NUMINAMATH_CALUDE_banner_coverage_count_l1444_144414


namespace NUMINAMATH_CALUDE_angle_difference_l1444_144445

theorem angle_difference (α β : Real) 
  (h1 : 3 * Real.sin α - Real.cos α = 0)
  (h2 : 7 * Real.sin β + Real.cos β = 0)
  (h3 : 0 < α) (h4 : α < Real.pi / 2)
  (h5 : Real.pi / 2 < β) (h6 : β < Real.pi) :
  2 * α - β = -3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_difference_l1444_144445


namespace NUMINAMATH_CALUDE_fish_weight_sum_l1444_144404

/-- The total weight of fish caught by Ali, Peter, and Joey -/
def total_fish_weight (ali_weight peter_weight joey_weight : ℝ) : ℝ :=
  ali_weight + peter_weight + joey_weight

/-- Theorem: The total weight of fish caught by Ali, Peter, and Joey is 25 kg -/
theorem fish_weight_sum :
  ∀ (peter_weight : ℝ),
  peter_weight > 0 →
  let ali_weight := 2 * peter_weight
  let joey_weight := peter_weight + 1
  ali_weight = 12 →
  total_fish_weight ali_weight peter_weight joey_weight = 25 := by
sorry

end NUMINAMATH_CALUDE_fish_weight_sum_l1444_144404


namespace NUMINAMATH_CALUDE_optimal_chair_removal_l1444_144490

theorem optimal_chair_removal (chairs_per_row : ℕ) (total_chairs : ℕ) (expected_participants : ℕ) 
  (h1 : chairs_per_row = 15)
  (h2 : total_chairs = 225)
  (h3 : expected_participants = 140) :
  let chairs_to_remove := 75
  let remaining_chairs := total_chairs - chairs_to_remove
  (remaining_chairs % chairs_per_row = 0) ∧ 
  (remaining_chairs ≥ expected_participants) ∧
  (∀ n : ℕ, n < chairs_to_remove → 
    (total_chairs - n) % chairs_per_row ≠ 0 ∨ 
    (total_chairs - n < expected_participants)) :=
by sorry

end NUMINAMATH_CALUDE_optimal_chair_removal_l1444_144490


namespace NUMINAMATH_CALUDE_saline_drip_rate_l1444_144423

/-- Proves that the saline drip makes 20 drops per minute given the treatment conditions -/
theorem saline_drip_rate (treatment_duration : ℕ) (drops_per_ml : ℚ) (total_volume : ℚ) :
  treatment_duration = 2 * 60 →  -- 2 hours in minutes
  drops_per_ml = 100 / 5 →       -- 100 drops per 5 ml
  total_volume = 120 →           -- 120 ml total volume
  (total_volume * drops_per_ml) / treatment_duration = 20 := by
  sorry

#check saline_drip_rate

end NUMINAMATH_CALUDE_saline_drip_rate_l1444_144423


namespace NUMINAMATH_CALUDE_combination_20_choose_6_l1444_144497

theorem combination_20_choose_6 : Nat.choose 20 6 = 19380 := by
  sorry

end NUMINAMATH_CALUDE_combination_20_choose_6_l1444_144497


namespace NUMINAMATH_CALUDE_product_and_reciprocal_sum_l1444_144442

theorem product_and_reciprocal_sum (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ x * y = 16 ∧ 1 / x = 3 * (1 / y) → x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_sum_l1444_144442


namespace NUMINAMATH_CALUDE_vessel_volume_ratio_l1444_144450

/-- Represents a vessel containing a mixture of milk and water -/
structure Vessel where
  milk : ℚ
  water : ℚ

/-- The ratio of milk to water in a vessel -/
def milkWaterRatio (v : Vessel) : ℚ := v.milk / v.water

/-- The total volume of a vessel -/
def volume (v : Vessel) : ℚ := v.milk + v.water

/-- Combines the contents of two vessels -/
def combineVessels (v1 v2 : Vessel) : Vessel :=
  { milk := v1.milk + v2.milk, water := v1.water + v2.water }

theorem vessel_volume_ratio (v1 v2 : Vessel) :
  milkWaterRatio v1 = 1/2 →
  milkWaterRatio v2 = 6/4 →
  milkWaterRatio (combineVessels v1 v2) = 1 →
  volume v1 / volume v2 = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_vessel_volume_ratio_l1444_144450


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1444_144431

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, 8 - 5*x > 25 → x ≤ -4 ∧ 8 - 5*(-4) > 25 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1444_144431


namespace NUMINAMATH_CALUDE_total_flooring_cost_l1444_144493

/-- Represents the dimensions and costs associated with a room's flooring replacement. -/
structure Room where
  length : ℝ
  width : ℝ
  removal_cost : ℝ
  new_flooring_cost_per_sqft : ℝ

/-- Calculates the total cost of replacing flooring in a room. -/
def room_cost (r : Room) : ℝ :=
  r.removal_cost + r.length * r.width * r.new_flooring_cost_per_sqft

/-- Theorem stating that the total cost of replacing flooring in all rooms is $264. -/
theorem total_flooring_cost (living_room bedroom kitchen : Room)
    (h1 : living_room = { length := 8, width := 7, removal_cost := 50, new_flooring_cost_per_sqft := 1.25 })
    (h2 : bedroom = { length := 6, width := 6, removal_cost := 35, new_flooring_cost_per_sqft := 1.50 })
    (h3 : kitchen = { length := 5, width := 4, removal_cost := 20, new_flooring_cost_per_sqft := 1.75 }) :
    room_cost living_room + room_cost bedroom + room_cost kitchen = 264 := by
  sorry

end NUMINAMATH_CALUDE_total_flooring_cost_l1444_144493


namespace NUMINAMATH_CALUDE_probability_of_zero_in_one_over_99999_l1444_144415

def decimal_expansion (n : ℕ) : List ℕ := 
  if n = 99999 then [0, 0, 0, 0, 1] else sorry

theorem probability_of_zero_in_one_over_99999 : 
  let expansion := decimal_expansion 99999
  let total_digits := expansion.length
  let zero_count := (expansion.filter (· = 0)).length
  (zero_count : ℚ) / total_digits = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_zero_in_one_over_99999_l1444_144415
