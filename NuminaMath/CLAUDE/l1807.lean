import Mathlib

namespace mildreds_father_oranges_l1807_180742

/-- The number of oranges Mildred's father ate -/
def oranges_eaten (initial : ℝ) (remaining : ℝ) : ℝ :=
  initial - remaining

/-- Proof that Mildred's father ate 2.0 oranges -/
theorem mildreds_father_oranges : oranges_eaten 77.0 75 = 2.0 := by
  sorry

end mildreds_father_oranges_l1807_180742


namespace altitude_equation_tangent_lines_equal_intercepts_l1807_180798

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y = 0

-- Define the center of circle C
def center_C : ℝ × ℝ := (-1, 1)

-- Define points A and B
def point_A : ℝ × ℝ := (4, 0)
def point_B : ℝ × ℝ := (0, -2)

-- Theorem for the altitude equation
theorem altitude_equation :
  ∃ (x y : ℝ), 2*x + y + 1 = 0 ∧
  (∀ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y →
    (p.2 - center_C.2) = -2 * (p.1 - center_C.1) ∧
    (p.2 - point_A.2) * (point_B.1 - point_A.1) = -(p.1 - point_A.1) * (point_B.2 - point_A.2)) :=
sorry

-- Theorem for tangent lines with equal intercepts
theorem tangent_lines_equal_intercepts :
  (∀ (x y : ℝ), (x - y = 0 ∨ x + y - 2 = 0 ∨ x + y + 2 = 0) →
    (∃ (t : ℝ), x = t ∧ y = t) ∨
    (∃ (t : ℝ), x = t ∧ y = 2 - t) ∨
    (∃ (t : ℝ), x = t ∧ y = -2 - t)) ∧
  (∀ (x y : ℝ), ((∃ (t : ℝ), x = t ∧ y = t) ∨
                 (∃ (t : ℝ), x = t ∧ y = 2 - t) ∨
                 (∃ (t : ℝ), x = t ∧ y = -2 - t)) →
    (x - center_C.1)^2 + (y - center_C.2)^2 = 2) :=
sorry

end altitude_equation_tangent_lines_equal_intercepts_l1807_180798


namespace total_distance_is_20_l1807_180766

/-- Represents the walking scenario with given speeds and total time -/
structure WalkingScenario where
  flat_speed : ℝ
  uphill_speed : ℝ
  downhill_speed : ℝ
  total_time : ℝ

/-- Calculates the total distance walked given a WalkingScenario -/
def total_distance (s : WalkingScenario) : ℝ :=
  sorry

/-- Theorem stating that the total distance walked is 20 km -/
theorem total_distance_is_20 (s : WalkingScenario) 
  (h1 : s.flat_speed = 4)
  (h2 : s.uphill_speed = 3)
  (h3 : s.downhill_speed = 6)
  (h4 : s.total_time = 5) :
  total_distance s = 20 := by
  sorry

end total_distance_is_20_l1807_180766


namespace range_of_a_l1807_180747

theorem range_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ x : ℝ, |x - 4| + |x - 3| < a) : a > 1 := by
  sorry

end range_of_a_l1807_180747


namespace negation_of_universal_proposition_l1807_180784

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by sorry

end negation_of_universal_proposition_l1807_180784


namespace candy_bar_cost_l1807_180741

/-- The cost of the candy bar given the total spent and the cost of cookies -/
theorem candy_bar_cost (total_spent : ℕ) (cookie_cost : ℕ) (h1 : total_spent = 53) (h2 : cookie_cost = 39) :
  total_spent - cookie_cost = 14 := by
  sorry

end candy_bar_cost_l1807_180741


namespace min_value_quadratic_l1807_180759

theorem min_value_quadratic (k : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 - 4 * k * x * y + (2 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9 ≥ 0) ∧ 
  (∃ x y : ℝ, 3 * x^2 - 4 * k * x * y + (2 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9 = 0) ↔ 
  k = 2 :=
sorry

end min_value_quadratic_l1807_180759


namespace parabola_chord_length_l1807_180727

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Represents a parabola x^2 = 2py -/
def Parabola (p : ℝ) : Type :=
  {point : ParabolaPoint // point.x^2 = 2 * p * point.y}

theorem parabola_chord_length (p : ℝ) (h_p : p > 0) :
  ∀ (A B : Parabola p),
    (A.val.y + B.val.y) / 2 = 3 →
    (∀ C D : Parabola p, |C.val.y - D.val.y + p| ≤ 8) →
    (∃ E F : Parabola p, |E.val.y - F.val.y + p| = 8) →
    p = 2 := by sorry

end parabola_chord_length_l1807_180727


namespace thursday_monday_difference_l1807_180749

/-- Represents the number of bonnets made on each day of the week --/
structure BonnetProduction where
  monday : ℕ
  tuesday_wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Theorem stating the difference between Thursday and Monday bonnet production --/
theorem thursday_monday_difference (bp : BonnetProduction) : 
  bp.monday = 10 →
  bp.tuesday_wednesday = 2 * bp.monday →
  bp.friday = bp.thursday - 5 →
  (bp.monday + bp.tuesday_wednesday + bp.thursday + bp.friday) / 5 = 11 →
  bp.thursday - bp.monday = 5 := by
  sorry

end thursday_monday_difference_l1807_180749


namespace min_value_theorem_l1807_180786

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  second_term : a 2 = 4
  tenth_sum : S 10 = 110

/-- The theorem statement -/
theorem min_value_theorem (seq : ArithmeticSequence) :
  ∃ (n : ℕ), ∀ (m : ℕ), (seq.S m + 64) / seq.a m ≥ 17 / 2 :=
sorry

end min_value_theorem_l1807_180786


namespace paper_stack_height_l1807_180756

/-- Given a package of paper with known thickness and number of sheets,
    calculate the height of a stack with a different number of sheets. -/
theorem paper_stack_height
  (package_sheets : ℕ)
  (package_thickness : ℝ)
  (stack_sheets : ℕ)
  (h_package_sheets : package_sheets = 400)
  (h_package_thickness : package_thickness = 4)
  (h_stack_sheets : stack_sheets = 1000) :
  (stack_sheets : ℝ) * package_thickness / package_sheets = 10 :=
sorry

end paper_stack_height_l1807_180756


namespace geometric_sequence_property_l1807_180701

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_prod : a 4 * a 5 * a 6 = 27) :
  a 1 * a 9 = 9 := by
sorry

end geometric_sequence_property_l1807_180701


namespace teresas_class_size_l1807_180761

theorem teresas_class_size :
  ∃! n : ℕ, 50 < n ∧ n < 100 ∧ n % 3 = 2 ∧ n % 4 = 2 ∧ n % 5 = 2 ∧ n = 62 := by
  sorry

end teresas_class_size_l1807_180761


namespace pairing_ways_eq_5040_l1807_180783

/-- Represents the number of students with each grade -/
structure GradeDistribution where
  grade5 : Nat
  grade4 : Nat
  grade3 : Nat

/-- Calculates the number of ways to form pairs of students with different grades -/
def pairingWays (dist : GradeDistribution) : Nat :=
  Nat.choose dist.grade4 dist.grade5 * Nat.factorial dist.grade5

/-- The given grade distribution in the problem -/
def problemDistribution : GradeDistribution :=
  { grade5 := 6, grade4 := 7, grade3 := 1 }

/-- Theorem stating that the number of pairing ways for the given distribution is 5040 -/
theorem pairing_ways_eq_5040 :
  pairingWays problemDistribution = 5040 := by
  sorry

end pairing_ways_eq_5040_l1807_180783


namespace special_polyhedron_interior_segments_l1807_180705

/-- Represents a convex polyhedron with specific face types -/
structure SpecialPolyhedron where
  square_faces : ℕ
  hexagon_faces : ℕ
  octagon_faces : ℕ
  vertex_configuration : Bool  -- True if each vertex meets one square, one hexagon, and one octagon

/-- Calculates the number of interior segments in the special polyhedron -/
def interior_segments (p : SpecialPolyhedron) : ℕ :=
  sorry

/-- Theorem stating the number of interior segments in the specific polyhedron -/
theorem special_polyhedron_interior_segments :
  let p : SpecialPolyhedron := {
    square_faces := 12,
    hexagon_faces := 8,
    octagon_faces := 6,
    vertex_configuration := true
  }
  interior_segments p = 840 := by
  sorry

end special_polyhedron_interior_segments_l1807_180705


namespace arc_length_radius_l1807_180722

/-- Given an arc length and central angle, calculate the radius of the circle. -/
theorem arc_length_radius (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = 2) :
  s / θ = 2 := by sorry

end arc_length_radius_l1807_180722


namespace set_equality_problem_l1807_180733

theorem set_equality_problem (A B : Set ℕ) (a : ℕ) :
  A = {1, 4} →
  B = {0, 1, a} →
  A ∪ B = {0, 1, 4} →
  a = 4 := by
sorry

end set_equality_problem_l1807_180733


namespace perfect_square_property_l1807_180739

theorem perfect_square_property (n : ℤ) : 
  (∃ k : ℤ, 2 + 2 * Real.sqrt (1 + 12 * n^2) = k) → 
  ∃ m : ℤ, (2 + 2 * Real.sqrt (1 + 12 * n^2))^2 = m^2 := by
sorry

end perfect_square_property_l1807_180739


namespace product_last_two_digits_not_consecutive_l1807_180794

theorem product_last_two_digits_not_consecutive (a b c : ℕ) : 
  ¬ (∃ n : ℕ, 
    (10 ≤ n ∧ n < 100) ∧
    (ab % 100 = n ∧ ac % 100 = n + 1 ∧ bc % 100 = n + 2) ∨
    (ab % 100 = n ∧ bc % 100 = n + 1 ∧ ac % 100 = n + 2) ∨
    (ac % 100 = n ∧ ab % 100 = n + 1 ∧ bc % 100 = n + 2) ∨
    (ac % 100 = n ∧ bc % 100 = n + 1 ∧ ab % 100 = n + 2) ∨
    (bc % 100 = n ∧ ab % 100 = n + 1 ∧ ac % 100 = n + 2) ∨
    (bc % 100 = n ∧ ac % 100 = n + 1 ∧ ab % 100 = n + 2)) :=
by
  sorry

end product_last_two_digits_not_consecutive_l1807_180794


namespace hostel_provisions_l1807_180709

/-- Proves that given the initial conditions of a hostel's food provisions,
    the initial number of days the provisions were planned for is 28. -/
theorem hostel_provisions (initial_men : ℕ) (left_men : ℕ) (days_after_leaving : ℕ) 
  (h1 : initial_men = 250)
  (h2 : left_men = 50)
  (h3 : days_after_leaving = 35) :
  (initial_men * ((initial_men - left_men) * days_after_leaving / initial_men) : ℚ) = 
  (initial_men * 28 : ℚ) := by
sorry

end hostel_provisions_l1807_180709


namespace rearrange_pegs_l1807_180767

/-- Represents a position on the board --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents the board state --/
def BoardState := List Position

/-- Checks if a given arrangement of pegs satisfies the condition of 5 rows with 4 pegs each --/
def isValidArrangement (arrangement : BoardState) : Bool :=
  sorry

/-- Counts the number of pegs that need to be moved to transform one arrangement into another --/
def pegsMoved (initial : BoardState) (final : BoardState) : Nat :=
  sorry

/-- The main theorem stating that it's possible to achieve the desired arrangement by moving exactly 3 pegs --/
theorem rearrange_pegs (initial : BoardState) :
  (initial.length = 10) →
  ∃ (final : BoardState), 
    isValidArrangement final ∧ 
    pegsMoved initial final = 3 :=
  sorry

end rearrange_pegs_l1807_180767


namespace a_power_m_plus_2n_l1807_180700

theorem a_power_m_plus_2n (a : ℝ) (m n : ℤ) (h1 : a^m = 2) (h2 : a^n = 3) :
  a^(m + 2*n) = 18 := by
  sorry

end a_power_m_plus_2n_l1807_180700


namespace remaining_slices_eq_ten_l1807_180751

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := 8

/-- The number of slices in an extra-large pizza -/
def extra_large_pizza_slices : ℕ := 12

/-- The number of slices Mary eats from the large pizza -/
def slices_eaten_from_large : ℕ := 7

/-- The number of slices Mary eats from the extra-large pizza -/
def slices_eaten_from_extra_large : ℕ := 3

/-- The total number of remaining slices after Mary eats from both pizzas -/
def total_remaining_slices : ℕ := 
  (large_pizza_slices - slices_eaten_from_large) + 
  (extra_large_pizza_slices - slices_eaten_from_extra_large)

theorem remaining_slices_eq_ten : total_remaining_slices = 10 := by
  sorry

end remaining_slices_eq_ten_l1807_180751


namespace xOzSymmetry_of_A_l1807_180763

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Performs symmetry transformation with respect to the xOz plane -/
def xOzSymmetry (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

/-- The original point A -/
def A : Point3D :=
  { x := 2, y := -3, z := 1 }

/-- The expected result after symmetry -/
def expectedResult : Point3D :=
  { x := 2, y := 3, z := 1 }

theorem xOzSymmetry_of_A : xOzSymmetry A = expectedResult := by
  sorry

end xOzSymmetry_of_A_l1807_180763


namespace rhombus_side_length_l1807_180730

/-- A rhombus with one diagonal of length 20 and area 480 has sides of length 26 -/
theorem rhombus_side_length (d1 d2 area side : ℝ) : 
  d1 = 20 →
  area = 480 →
  area = d1 * d2 / 2 →
  side * side = (d1/2)^2 + (d2/2)^2 →
  side = 26 := by
sorry


end rhombus_side_length_l1807_180730


namespace only_rectangle_area_certain_l1807_180731

-- Define the events
inductive Event
  | WaterFreeze : Event
  | ExamScore : Event
  | CoinToss : Event
  | RectangleArea : Event

-- Define a function to check if an event is certain
def isCertainEvent : Event → Prop
  | Event.WaterFreeze => False
  | Event.ExamScore => False
  | Event.CoinToss => False
  | Event.RectangleArea => True

-- Theorem statement
theorem only_rectangle_area_certain :
  ∀ e : Event, isCertainEvent e ↔ e = Event.RectangleArea :=
by sorry

end only_rectangle_area_certain_l1807_180731


namespace not_necessary_not_sufficient_l1807_180711

-- Define the function f(x) = x^3 - x + a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x + a

-- Define the property of being an increasing function
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem not_necessary_not_sufficient :
  ¬(∀ a : ℝ, (a^2 - a = 0 ↔ is_increasing (f a))) :=
sorry

end not_necessary_not_sufficient_l1807_180711


namespace f_property_l1807_180771

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Real.sqrt x
  else if x ≥ 1 then 2 * (x - 1)
  else 0  -- We define f as 0 for x ≤ 0 to make it total

-- State the theorem
theorem f_property : ∃ a : ℝ, f a = f (a + 1) → f (1 / a) = 6 := by
  sorry

end f_property_l1807_180771


namespace total_bread_is_370_l1807_180764

/-- The amount of bread Cara ate for dinner, in grams -/
def dinner_bread : ℕ := 240

/-- The amount of bread Cara ate for lunch, in grams -/
def lunch_bread : ℕ := dinner_bread / 8

/-- The amount of bread Cara ate for breakfast, in grams -/
def breakfast_bread : ℕ := dinner_bread / 6

/-- The amount of bread Cara ate for snack, in grams -/
def snack_bread : ℕ := dinner_bread / 4

/-- The total amount of bread Cara ate, in grams -/
def total_bread : ℕ := dinner_bread + lunch_bread + breakfast_bread + snack_bread

theorem total_bread_is_370 : total_bread = 370 := by
  sorry

end total_bread_is_370_l1807_180764


namespace geometric_series_sum_l1807_180758

/-- Sum of a geometric series with n terms -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_series_sum :
  let a : ℚ := 1/6
  let r : ℚ := -1/2
  let n : ℕ := 7
  geometric_sum a r n = 129/1152 := by
sorry

end geometric_series_sum_l1807_180758


namespace custom_mult_square_identity_l1807_180778

-- Define the custom multiplication operation
def custom_mult (a b : ℝ) : ℝ := (a - b)^2

-- Theorem statement
theorem custom_mult_square_identity (x y : ℝ) :
  custom_mult (x^2) (y^2) = (x + y)^2 * (x - y)^2 := by
  sorry

end custom_mult_square_identity_l1807_180778


namespace min_y_value_l1807_180770

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 20*x + 72*y) : 
  ∀ y' : ℝ, (∃ x' : ℝ, x'^2 + y'^2 = 20*x' + 72*y') → y ≥ 36 - Real.sqrt 1396 := by
sorry

end min_y_value_l1807_180770


namespace rectangle_puzzle_l1807_180744

-- Define the lengths of the segments
def top_segment1 : ℝ := 2
def top_segment2 : ℝ := 3
def top_segment4 : ℝ := 4
def bottom_segment1 : ℝ := 3
def bottom_segment2 : ℝ := 5

-- Define X as a real number
def X : ℝ := sorry

-- State the theorem
theorem rectangle_puzzle :
  top_segment1 + top_segment2 + X + top_segment4 = bottom_segment1 + bottom_segment2 + (X + 1) →
  X = 1 := by
  sorry

end rectangle_puzzle_l1807_180744


namespace river_current_calculation_l1807_180799

/-- Represents the speed of a boat in still water -/
def boat_speed : ℝ := 20

/-- Represents the distance traveled up the river -/
def distance : ℝ := 91

/-- Represents the total time for the round trip -/
def total_time : ℝ := 10

/-- Calculates the speed of the river's current -/
def river_current_speed : ℝ := 6

theorem river_current_calculation :
  ∃ (c : ℝ), c = river_current_speed ∧
  distance / (boat_speed - c) + distance / (boat_speed + c) = total_time :=
by sorry

end river_current_calculation_l1807_180799


namespace water_tank_capacity_l1807_180720

theorem water_tank_capacity (initial_fraction : Rat) (final_fraction : Rat) (removed_liters : ℕ) 
  (h1 : initial_fraction = 2/3)
  (h2 : final_fraction = 1/3)
  (h3 : removed_liters = 20)
  (h4 : initial_fraction * tank_capacity - removed_liters = final_fraction * tank_capacity) :
  tank_capacity = 60 := by
  sorry

end water_tank_capacity_l1807_180720


namespace curve_and_lines_distance_properties_l1807_180797

/-- Given a curve C and lines l and l1 in a 2D plane, prove properties about distances -/
theorem curve_and_lines_distance_properties
  (B : ℝ × ℝ)
  (C : ℝ → ℝ × ℝ)
  (A : ℝ × ℝ)
  (l l1 : ℝ × ℝ → Prop)
  (h_B : B = (1, 1))
  (h_C : ∀ θ, C θ = (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ))
  (h_A : A = (4 * Real.sqrt 2 * Real.cos (π/4), 4 * Real.sqrt 2 * Real.sin (π/4)))
  (h_l : ∃ a, ∀ ρ θ, l (ρ * Real.cos θ, ρ * Real.sin θ) ↔ ρ * Real.cos (θ - π/4) = a)
  (h_l_A : l A)
  (h_l1_parallel : ∃ k, ∀ p, l1 p ↔ l (p.1 - k, p.2 - k))
  (h_l1_B : l1 B)
  (h_l1_intersect : ∃ M N, M ≠ N ∧ l1 (C M) ∧ l1 (C N)) :
  (∃ d, d = (8 * Real.sqrt 2 - Real.sqrt 14) / 2 ∧
    ∀ p, (∃ θ, C θ = p) → 
      ∀ q, l q → Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d) ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 12 * Real.sqrt 2 / 7 :=
sorry

end curve_and_lines_distance_properties_l1807_180797


namespace fib_150_mod_9_l1807_180757

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- Define the property that Fibonacci sequence mod 9 repeats every 24 terms
axiom fib_mod_9_period : ∀ n : ℕ, fib n % 9 = fib (n % 24) % 9

-- Theorem statement
theorem fib_150_mod_9 : fib 149 % 9 = 8 := by
  sorry

end fib_150_mod_9_l1807_180757


namespace triangle_area_l1807_180772

theorem triangle_area (base height : ℝ) (h1 : base = 4.5) (h2 : height = 6) :
  (base * height) / 2 = 13.5 := by
  sorry

end triangle_area_l1807_180772


namespace series_convergence_l1807_180723

noncomputable def x : ℕ → ℝ
  | 0 => 1
  | n + 1 => Real.log (Real.exp (x n) - x n)

theorem series_convergence :
  (∑' n, x n) = Real.exp 1 - 1 := by sorry

end series_convergence_l1807_180723


namespace decimal_point_error_l1807_180704

theorem decimal_point_error (actual_amount : ℚ) : 
  (actual_amount * 10 - actual_amount = 153) → actual_amount = 17 := by
  sorry

end decimal_point_error_l1807_180704


namespace pqr_value_l1807_180712

theorem pqr_value (p q r : ℂ) 
  (eq1 : p * q + 5 * q = -20)
  (eq2 : q * r + 5 * r = -20)
  (eq3 : r * p + 5 * p = -20) :
  p * q * r = 80 := by
sorry

end pqr_value_l1807_180712


namespace exponential_function_property_l1807_180703

theorem exponential_function_property (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x - 2
  f 0 = -1 := by sorry

end exponential_function_property_l1807_180703


namespace range_of_2alpha_minus_beta_l1807_180702

theorem range_of_2alpha_minus_beta (α β : ℝ) 
  (h : -π/2 < α ∧ α < β ∧ β < π/2) : 
  -3*π/2 < 2*α - β ∧ 2*α - β < π/2 := by
  sorry

end range_of_2alpha_minus_beta_l1807_180702


namespace total_distance_ran_l1807_180717

/-- The length of a football field in meters -/
def football_field_length : ℕ := 168

/-- The distance Nate ran in the first part, in terms of football field lengths -/
def initial_distance_in_fields : ℕ := 4

/-- The additional distance Nate ran in meters -/
def additional_distance : ℕ := 500

/-- Theorem: The total distance Nate ran is 1172 meters -/
theorem total_distance_ran : 
  football_field_length * initial_distance_in_fields + additional_distance = 1172 := by
  sorry

end total_distance_ran_l1807_180717


namespace unique_divisible_number_l1807_180740

theorem unique_divisible_number : ∃! n : ℕ, 
  45400 ≤ n ∧ n < 45500 ∧ 
  n % 2 = 0 ∧ 
  n % 7 = 0 ∧ 
  n % 9 = 0 :=
by sorry

end unique_divisible_number_l1807_180740


namespace light_ray_reflection_l1807_180776

/-- Represents a direction vector in 3D space -/
structure DirectionVector where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a trirectangular corner -/
structure TrirectangularCorner where

/-- Reflects a direction vector off a plane perpendicular to the x-axis -/
def reflectX (v : DirectionVector) : DirectionVector :=
  { x := -v.x, y := v.y, z := v.z }

/-- Reflects a direction vector off a plane perpendicular to the y-axis -/
def reflectY (v : DirectionVector) : DirectionVector :=
  { x := v.x, y := -v.y, z := v.z }

/-- Reflects a direction vector off a plane perpendicular to the z-axis -/
def reflectZ (v : DirectionVector) : DirectionVector :=
  { x := v.x, y := v.y, z := -v.z }

/-- 
  Theorem: A light ray reflecting off all three faces of a trirectangular corner
  will change its direction to the opposite of its initial direction.
-/
theorem light_ray_reflection 
  (corner : TrirectangularCorner) 
  (initial_direction : DirectionVector) :
  reflectX (reflectY (reflectZ initial_direction)) = 
  { x := -initial_direction.x, 
    y := -initial_direction.y, 
    z := -initial_direction.z } := by
  sorry


end light_ray_reflection_l1807_180776


namespace mo_hot_chocolate_consumption_l1807_180737

/-- The number of cups of hot chocolate Mo drinks on rainy mornings -/
def cups_of_hot_chocolate : ℚ := 1.75

/-- The number of cups of tea Mo drinks on non-rainy mornings -/
def cups_of_tea_non_rainy : ℕ := 5

/-- The total number of cups of tea and hot chocolate Mo drank last week -/
def total_cups_last_week : ℕ := 22

/-- The difference between tea cups and hot chocolate cups Mo drank last week -/
def tea_minus_chocolate : ℕ := 8

/-- The number of rainy days last week -/
def rainy_days : ℕ := 4

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem mo_hot_chocolate_consumption :
  cups_of_hot_chocolate * rainy_days = 
    total_cups_last_week - (cups_of_tea_non_rainy * (days_in_week - rainy_days)) - tea_minus_chocolate := by
  sorry

end mo_hot_chocolate_consumption_l1807_180737


namespace A_in_B_l1807_180706

-- Define the set A
def A : Set ℕ := {0, 1}

-- Define the set B
def B : Set (Set ℕ) := {x | x ⊆ A}

-- Theorem statement
theorem A_in_B : A ∈ B := by sorry

end A_in_B_l1807_180706


namespace min_distance_line_parabola_l1807_180721

open Real

/-- The minimum distance between a point on the line y = (12/5)x - 3 and a point on the parabola y = x^2 is 3/5. -/
theorem min_distance_line_parabola :
  let line := fun x => (12/5) * x - 3
  let parabola := fun x => x^2
  ∃ (a b : ℝ),
    (∀ x y : ℝ, 
      (y = line x ∨ y = parabola x) → 
      (a - x)^2 + (line a - y)^2 ≥ (3/5)^2) ∧
    line a = parabola b ∧
    (a - b)^2 + (line a - parabola b)^2 = (3/5)^2 :=
sorry

end min_distance_line_parabola_l1807_180721


namespace game_score_l1807_180718

/-- Calculates the total score of a father and son in a game where the son scores three times more than the father. -/
def totalScore (fatherScore : ℕ) : ℕ :=
  fatherScore + 3 * fatherScore

/-- Theorem stating that when the father scores 7 points, the total score is 28 points. -/
theorem game_score : totalScore 7 = 28 := by
  sorry

end game_score_l1807_180718


namespace expression_simplification_l1807_180796

theorem expression_simplification (b : ℝ) : ((3 * b + 6) - 5 * b) / 3 = -2/3 * b + 2 := by
  sorry

end expression_simplification_l1807_180796


namespace age_problem_l1807_180724

theorem age_problem (a₁ a₂ a₃ a₄ a₅ : ℕ) : 
  a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅ →
  (a₁ + a₂ + a₃) / 3 = 18 →
  a₅ - a₄ = 5 →
  (a₃ + a₄ + a₅) / 3 = 26 →
  a₂ - a₁ = 7 →
  (a₁ + a₅) / 2 = 22 →
  a₁ = 13 ∧ a₂ = 20 ∧ a₃ = 21 ∧ a₄ = 26 ∧ a₅ = 31 :=
by sorry

#check age_problem

end age_problem_l1807_180724


namespace day_of_week_N_minus_1_l1807_180715

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a year -/
structure Year where
  value : ℕ
  is_leap : Bool

/-- Function to get the day of the week for a given day in a year -/
def day_of_week (year : Year) (day : ℕ) : DayOfWeek := sorry

/-- Function to get the next day of the week -/
def next_day (d : DayOfWeek) : DayOfWeek := sorry

/-- Function to get the previous day of the week -/
def prev_day (d : DayOfWeek) : DayOfWeek := sorry

theorem day_of_week_N_minus_1 
  (N : Year)
  (h1 : N.is_leap = true)
  (h2 : day_of_week N 250 = DayOfWeek.Friday)
  (h3 : (Year.mk (N.value + 1) true).is_leap = true)
  (h4 : day_of_week (Year.mk (N.value + 1) true) 150 = DayOfWeek.Friday) :
  day_of_week (Year.mk (N.value - 1) false) 50 = DayOfWeek.Wednesday :=
sorry

end day_of_week_N_minus_1_l1807_180715


namespace same_answers_l1807_180743

-- Define a type for questions
variable (Question : Type)

-- Define predicates for each witness's "yes" answers
variable (A B C : Question → Prop)

-- State the conditions
variable (h1 : ∀ q, B q ∧ C q → A q)
variable (h2 : ∀ q, A q → B q)
variable (h3 : ∀ q, B q → A q ∨ C q)

-- Theorem statement
theorem same_answers : ∀ q, A q ↔ B q := by sorry

end same_answers_l1807_180743


namespace double_money_l1807_180789

theorem double_money (initial_amount : ℕ) : 
  initial_amount + initial_amount = 2 * initial_amount := by
  sorry

#check double_money

end double_money_l1807_180789


namespace maria_stationery_cost_l1807_180746

/-- The cost of Maria's stationery purchase -/
def stationery_cost (pencil_cost : ℝ) (pen_cost : ℝ) : Prop :=
  pencil_cost = 8 ∧ 
  pen_cost = pencil_cost / 2 ∧
  pencil_cost + pen_cost = 12

/-- Theorem: Maria paid $12 for both the pen and the pencil -/
theorem maria_stationery_cost : 
  ∃ (pencil_cost pen_cost : ℝ), stationery_cost pencil_cost pen_cost :=
by
  sorry

end maria_stationery_cost_l1807_180746


namespace least_number_with_remainder_four_ninety_four_satisfies_conditions_ninety_four_is_least_number_l1807_180792

theorem least_number_with_remainder_four (n : ℕ) : 
  (n % 5 = 4 ∧ n % 6 = 4 ∧ n % 9 = 4 ∧ n % 18 = 4) → n ≥ 94 :=
by sorry

theorem ninety_four_satisfies_conditions : 
  94 % 5 = 4 ∧ 94 % 6 = 4 ∧ 94 % 9 = 4 ∧ 94 % 18 = 4 :=
by sorry

theorem ninety_four_is_least_number : 
  ∀ n : ℕ, (n % 5 = 4 ∧ n % 6 = 4 ∧ n % 9 = 4 ∧ n % 18 = 4) → n ≥ 94 :=
by sorry

end least_number_with_remainder_four_ninety_four_satisfies_conditions_ninety_four_is_least_number_l1807_180792


namespace max_value_function_l1807_180729

theorem max_value_function (a b : ℝ) (h1 : a > b) (h2 : b ≥ 0) :
  ∃ M : ℝ, M = Real.sqrt ((a - b)^2 + a^2) ∧
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 →
    (a - b) * Real.sqrt (1 - x^2) + a * x ≤ M :=
by sorry

end max_value_function_l1807_180729


namespace sum_of_max_min_g_l1807_180738

/-- The function g(x) as defined in the problem -/
def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2*x - 8|

/-- The theorem stating that the sum of the maximum and minimum values of g(x) over [1, 10] is 2 -/
theorem sum_of_max_min_g :
  (⨆ (x : ℝ) (h : x ∈ Set.Icc 1 10), g x) + (⨅ (x : ℝ) (h : x ∈ Set.Icc 1 10), g x) = 2 :=
by sorry

end sum_of_max_min_g_l1807_180738


namespace original_profit_percentage_l1807_180752

theorem original_profit_percentage 
  (original_selling_price : ℝ) 
  (additional_profit : ℝ) :
  original_selling_price = 1100 →
  additional_profit = 70 →
  ∃ (original_purchase_price : ℝ),
    (1.3 * (0.9 * original_purchase_price) = original_selling_price + additional_profit) ∧
    ((original_selling_price - original_purchase_price) / original_purchase_price * 100 = 10) :=
by sorry

end original_profit_percentage_l1807_180752


namespace chosen_number_proof_l1807_180791

theorem chosen_number_proof (x : ℝ) : (x / 2) - 100 = 4 → x = 208 := by
  sorry

end chosen_number_proof_l1807_180791


namespace response_rate_is_sixty_percent_l1807_180782

/-- The response rate percentage for a questionnaire mailing --/
def response_rate_percentage (responses_needed : ℕ) (questionnaires_mailed : ℕ) : ℚ :=
  (responses_needed : ℚ) / (questionnaires_mailed : ℚ) * 100

/-- Theorem stating that the response rate percentage is 60% given the specified conditions --/
theorem response_rate_is_sixty_percent 
  (responses_needed : ℕ) 
  (questionnaires_mailed : ℕ) 
  (h1 : responses_needed = 300) 
  (h2 : questionnaires_mailed = 500) : 
  response_rate_percentage responses_needed questionnaires_mailed = 60 := by
  sorry

#eval response_rate_percentage 300 500

end response_rate_is_sixty_percent_l1807_180782


namespace fiftiethTerm_l1807_180753

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

/-- The 50th term of the specific arithmetic sequence -/
theorem fiftiethTerm : arithmeticSequenceTerm 2 5 50 = 247 := by
  sorry

end fiftiethTerm_l1807_180753


namespace divisibility_implication_l1807_180719

theorem divisibility_implication (a b : ℕ) (h : a < 1000) :
  (a^21 ∣ b^10) → (a^2 ∣ b) :=
by
  sorry

end divisibility_implication_l1807_180719


namespace cube_edge_ratio_l1807_180745

theorem cube_edge_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (h_vol : a^3 / b^3 = 27 / 8) : 
  a / b = 3 / 2 := by
sorry

end cube_edge_ratio_l1807_180745


namespace nine_to_150_mod_50_l1807_180735

theorem nine_to_150_mod_50 : 9^150 % 50 = 1 := by
  sorry

end nine_to_150_mod_50_l1807_180735


namespace circle_equation_l1807_180732

theorem circle_equation (x y : ℝ) : 
  (∃ c : ℝ × ℝ, (x - c.1)^2 + (y - c.2)^2 = 8^2) ↔ 
  x^2 + 14*x + y^2 + 8*y + 1 = 0 :=
sorry

end circle_equation_l1807_180732


namespace solutions_of_quadratic_equation_l1807_180750

theorem solutions_of_quadratic_equation :
  ∀ x : ℝ, (x - 1) * (x + 2) = 0 ↔ x = 1 ∨ x = -2 := by sorry

end solutions_of_quadratic_equation_l1807_180750


namespace cyclist_distance_l1807_180713

/-- The distance traveled by a cyclist moving between two people walking towards each other -/
theorem cyclist_distance (distance : ℝ) (speed_vasya : ℝ) (speed_roma : ℝ) 
  (h1 : distance > 0)
  (h2 : speed_vasya > 0)
  (h3 : speed_roma > 0) :
  let speed_dima := speed_vasya + speed_roma
  let time := distance / (speed_vasya + speed_roma)
  speed_dima * time = distance :=
by sorry

end cyclist_distance_l1807_180713


namespace negative_root_range_l1807_180773

theorem negative_root_range (x a : ℝ) : 
  x < 0 → 
  (2/3)^x = (1+a)/(1-a) → 
  0 < a ∧ a < 1 :=
by sorry

end negative_root_range_l1807_180773


namespace five_twos_equal_twentyfour_l1807_180785

theorem five_twos_equal_twentyfour : ∃ (f : ℕ → ℕ → ℕ) (g : ℕ → ℕ → ℕ), 
  f (g 2 2 + 2) (g 2 2) = 24 :=
by sorry

end five_twos_equal_twentyfour_l1807_180785


namespace sequence_problem_l1807_180708

/-- Given a sequence a and a geometric sequence b, prove that a_2016 = 1 -/
theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) : 
  a 1 = 1 →
  (∀ n, b n = a (n + 1) / a n) →
  (∀ n, b (n + 1) / b n = b 2 / b 1) →
  b 1008 = 1 →
  a 2016 = 1 := by
sorry

end sequence_problem_l1807_180708


namespace shaded_area_calculation_l1807_180736

/-- The area of the shaded region in a geometric figure with the following properties:
    - A large square with side length 20 cm
    - Four quarter circles with radius 10 cm centered at the corners of the large square
    - A smaller square with side length 10 cm centered inside the larger square
    is equal to 100π - 100 cm². -/
theorem shaded_area_calculation (π : ℝ) : ℝ := by
  -- Define the side lengths and radius
  let large_square_side : ℝ := 20
  let small_square_side : ℝ := 10
  let quarter_circle_radius : ℝ := 10

  -- Define the areas
  let large_square_area : ℝ := large_square_side ^ 2
  let small_square_area : ℝ := small_square_side ^ 2
  let quarter_circles_area : ℝ := π * quarter_circle_radius ^ 2

  -- Calculate the shaded area
  let shaded_area : ℝ := quarter_circles_area - small_square_area

  -- Prove that the shaded area equals 100π - 100
  sorry

end shaded_area_calculation_l1807_180736


namespace expected_vote_for_a_l1807_180793

/-- Percentage of registered voters who are Democrats -/
def democrat_percentage : ℝ := 0.60

/-- Percentage of registered voters who are Republicans -/
def republican_percentage : ℝ := 1 - democrat_percentage

/-- Percentage of Democrats expected to vote for candidate A -/
def democrat_vote_a : ℝ := 0.70

/-- Percentage of Republicans expected to vote for candidate A -/
def republican_vote_a : ℝ := 0.20

/-- Theorem: The percentage of registered voters expected to vote for candidate A is 50% -/
theorem expected_vote_for_a :
  democrat_percentage * democrat_vote_a + republican_percentage * republican_vote_a = 0.50 := by
  sorry

end expected_vote_for_a_l1807_180793


namespace quadratic_roots_properties_l1807_180707

theorem quadratic_roots_properties (p q : ℕ) (hp : Prime p) (hq : Prime q) 
  (h_roots : ∃ (r₁ r₂ : ℕ), r₁ ≠ r₂ ∧ r₁ > 0 ∧ r₂ > 0 ∧ r₁^2 - p*r₁ + q = 0 ∧ r₂^2 - p*r₂ + q = 0) :
  (∃ (r₁ r₂ : ℕ), r₁ ≠ r₂ ∧ r₁ > 0 ∧ r₂ > 0 ∧ r₁^2 - p*r₁ + q = 0 ∧ r₂^2 - p*r₂ + q = 0 ∧ 
    (∃ (k : ℕ), r₁ - r₂ = 2*k + 1 ∨ r₂ - r₁ = 2*k + 1)) ∧ 
  (∃ (r : ℕ), (r^2 - p*r + q = 0) ∧ Prime r) ∧
  Prime (p^2 - q) ∧
  Prime (p + q) := by
  sorry

#check quadratic_roots_properties

end quadratic_roots_properties_l1807_180707


namespace fibonacci_factorial_last_two_digits_sum_l1807_180728

def fibonacci_factorial_series : List Nat := [1, 1, 2, 3, 5, 8, 13, 21]

def last_two_digits (n : Nat) : Nat :=
  n % 100

def sum_last_two_digits (series : List Nat) : Nat :=
  (series.map (λ x => last_two_digits (Nat.factorial x))).sum % 100

theorem fibonacci_factorial_last_two_digits_sum :
  sum_last_two_digits fibonacci_factorial_series = 5 := by
  sorry

end fibonacci_factorial_last_two_digits_sum_l1807_180728


namespace fahrenheit_to_celsius_l1807_180765

theorem fahrenheit_to_celsius (C F : ℚ) : 
  C = 35 → C = (7/12) * (F - 40) → F = 100 := by
  sorry

end fahrenheit_to_celsius_l1807_180765


namespace birds_to_asia_count_l1807_180726

/-- The number of bird families that flew away to Asia -/
def birds_to_asia : ℕ := sorry

/-- The number of bird families living near the mountain -/
def birds_near_mountain : ℕ := 38

/-- The number of bird families that flew away to Africa -/
def birds_to_africa : ℕ := 47

theorem birds_to_asia_count : birds_to_asia = 94 := by
  sorry

end birds_to_asia_count_l1807_180726


namespace area_is_54_height_is_7_2_l1807_180716

/-- A triangle with side lengths 9, 12, and 15 units -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : a = 9
  h_b : b = 12
  h_c : c = 15
  h_right : a ^ 2 + b ^ 2 = c ^ 2

/-- The area of the triangle is 54 square units -/
theorem area_is_54 (t : RightTriangle) : (1 / 2) * t.a * t.b = 54 := by sorry

/-- The height from the right angle vertex to the hypotenuse is 7.2 units -/
theorem height_is_7_2 (t : RightTriangle) : (t.a * t.b) / t.c = 7.2 := by sorry

end area_is_54_height_is_7_2_l1807_180716


namespace equation_b_not_symmetric_l1807_180777

def is_symmetric_to_x_axis (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = f x (-y)

theorem equation_b_not_symmetric :
  ¬(is_symmetric_to_x_axis (fun x y => x^2*y + x*y^2 - 1)) ∧
  (is_symmetric_to_x_axis (fun x y => x^2 - x + y^2 - 1)) ∧
  (is_symmetric_to_x_axis (fun x y => 2*x^2 - y^2 - 1)) ∧
  (is_symmetric_to_x_axis (fun x y => x + y^2 + 1)) :=
by sorry

end equation_b_not_symmetric_l1807_180777


namespace student_count_l1807_180760

theorem student_count (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 16) 
  (h2 : rank_from_left = 6) : 
  rank_from_right + rank_from_left - 1 = 21 := by
  sorry

end student_count_l1807_180760


namespace triangle_area_determines_p_l1807_180775

/-- Given a triangle ABC with vertices A(3, 15), B(15, 0), and C(0, p),
    prove that if the area of the triangle is 36, then p = 12.75 -/
theorem triangle_area_determines_p :
  let A : ℝ × ℝ := (3, 15)
  let B : ℝ × ℝ := (15, 0)
  let C : ℝ × ℝ := (0, p)
  let triangle_area (A B C : ℝ × ℝ) : ℝ :=
    (1/2) * abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2))
  ∀ p : ℝ, triangle_area A B C = 36 → p = 12.75 := by
  sorry

#check triangle_area_determines_p

end triangle_area_determines_p_l1807_180775


namespace factorization_difference_of_squares_l1807_180769

theorem factorization_difference_of_squares (m n : ℝ) : (m + n)^2 - (m - n)^2 = 4 * m * n := by
  sorry

end factorization_difference_of_squares_l1807_180769


namespace phone_prob_theorem_l1807_180795

def phone_prob (p1 p2 p3 : ℝ) : Prop :=
  p1 = 0.5 ∧ p2 = 0.3 ∧ p3 = 0.2 →
  p1 + p2 = 0.8

theorem phone_prob_theorem :
  ∀ p1 p2 p3 : ℝ, phone_prob p1 p2 p3 :=
by
  sorry

end phone_prob_theorem_l1807_180795


namespace system_equations_range_l1807_180725

theorem system_equations_range (a b x y : ℝ) : 
  3 * x - y = 2 * a - 5 →
  x + 2 * y = 3 * a + 3 →
  x > 0 →
  y > 0 →
  a - b = 4 →
  b < 2 →
  a > 1 ∧ -2 < a + b ∧ a + b < 8 := by
sorry

end system_equations_range_l1807_180725


namespace contrapositive_product_nonzero_l1807_180734

theorem contrapositive_product_nonzero (a b : ℝ) :
  (¬(a * b ≠ 0) → ¬(a ≠ 0 ∧ b ≠ 0)) ↔ ((a = 0 ∨ b = 0) → a * b = 0) := by
  sorry

end contrapositive_product_nonzero_l1807_180734


namespace max_point_difference_is_n_l1807_180754

/-- A soccer tournament with n teams -/
structure SoccerTournament where
  n : ℕ  -- number of teams
  n_pos : 0 < n  -- number of teams is positive

/-- The result of a match between two teams -/
inductive MatchResult
  | Win
  | Loss
  | Draw

/-- Points awarded for each match result -/
def pointsForResult (result : MatchResult) : ℕ :=
  match result with
  | MatchResult.Win => 2
  | MatchResult.Loss => 0
  | MatchResult.Draw => 1

/-- The maximum possible difference in points between adjacent teams -/
def maxPointDifference (tournament : SoccerTournament) : ℕ :=
  tournament.n

theorem max_point_difference_is_n (tournament : SoccerTournament) :
  ∃ (team1 team2 : ℕ),
    team1 < tournament.n ∧
    team2 < tournament.n ∧
    team1 + 1 = team2 ∧
    ∃ (points1 points2 : ℕ),
      points1 - points2 = maxPointDifference tournament :=
by
  sorry

#check max_point_difference_is_n

end max_point_difference_is_n_l1807_180754


namespace perfect_square_polynomial_l1807_180755

/-- Given a polynomial x^4 - x^3 + x^2 + ax + b that is a perfect square,
    prove that b = 9/64 -/
theorem perfect_square_polynomial (a b : ℚ) : 
  (∃ p q r : ℚ, ∀ x, x^4 - x^3 + x^2 + a*x + b = (p*x^2 + q*x + r)^2) →
  b = 9/64 := by
sorry

end perfect_square_polynomial_l1807_180755


namespace f_composition_result_l1807_180710

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3 else 2^x

theorem f_composition_result : f (f (1/9)) = 1/4 := by
  sorry

end f_composition_result_l1807_180710


namespace janice_stairs_problem_l1807_180788

/-- The number of times Janice goes down the stairs in a day -/
def times_down (flights_per_floor : ℕ) (times_up : ℕ) (total_flights : ℕ) : ℕ :=
  (total_flights - flights_per_floor * times_up) / flights_per_floor

theorem janice_stairs_problem (flights_per_floor : ℕ) (times_up : ℕ) (total_flights : ℕ) 
    (h1 : flights_per_floor = 3)
    (h2 : times_up = 5)
    (h3 : total_flights = 24) :
  times_down flights_per_floor times_up total_flights = 3 := by
  sorry

#eval times_down 3 5 24

end janice_stairs_problem_l1807_180788


namespace complex_power_sum_l1807_180779

theorem complex_power_sum (z : ℂ) (h : z^2 + z + 1 = 0) : z^2010 + z^2009 + 1 = 0 := by
  sorry

end complex_power_sum_l1807_180779


namespace seedling_survival_probability_l1807_180787

/-- Represents the data for a single sample of transplanted ginkgo seedlings -/
structure SeedlingData where
  transplanted : ℕ
  survived : ℕ
  survival_rate : ℚ
  transplanted_positive : transplanted > 0
  survived_le_transplanted : survived ≤ transplanted
  rate_calculation : survival_rate = survived / transplanted

/-- The data set of ginkgo seedling transplantation experiments -/
def seedling_samples : List SeedlingData := [
  ⟨100, 84, 84/100, by norm_num, by norm_num, by norm_num⟩,
  ⟨300, 279, 279/300, by norm_num, by norm_num, by norm_num⟩,
  ⟨600, 505, 505/600, by norm_num, by norm_num, by norm_num⟩,
  ⟨1000, 847, 847/1000, by norm_num, by norm_num, by norm_num⟩,
  ⟨7000, 6337, 6337/7000, by norm_num, by norm_num, by norm_num⟩,
  ⟨15000, 13581, 13581/15000, by norm_num, by norm_num, by norm_num⟩
]

/-- The estimated probability of ginkgo seedling survival -/
def estimated_probability : ℚ := 9/10

/-- Theorem stating that the estimated probability approaches 0.9 as sample size increases -/
theorem seedling_survival_probability :
  ∀ ε > 0, ∃ N, ∀ sample ∈ seedling_samples,
    sample.transplanted ≥ N →
    |sample.survival_rate - estimated_probability| < ε :=
sorry

end seedling_survival_probability_l1807_180787


namespace marie_profit_l1807_180790

def total_loaves : ℕ := 60
def cost_per_loaf : ℚ := 1
def morning_price : ℚ := 3
def afternoon_discount : ℚ := 0.25
def donated_loaves : ℕ := 5

def morning_sales : ℕ := total_loaves / 3
def remaining_after_morning : ℕ := total_loaves - morning_sales
def afternoon_sales : ℕ := remaining_after_morning / 2
def remaining_after_afternoon : ℕ := remaining_after_morning - afternoon_sales
def unsold_loaves : ℕ := remaining_after_afternoon - donated_loaves

def afternoon_price : ℚ := morning_price * (1 - afternoon_discount)

def total_revenue : ℚ := morning_sales * morning_price + afternoon_sales * afternoon_price
def total_cost : ℚ := total_loaves * cost_per_loaf
def profit : ℚ := total_revenue - total_cost

theorem marie_profit : profit = 45 := by
  sorry

end marie_profit_l1807_180790


namespace brennan_pepper_amount_l1807_180780

def initial_pepper : ℝ := 0.25
def used_pepper : ℝ := 0.16
def remaining_pepper : ℝ := 0.09

theorem brennan_pepper_amount :
  initial_pepper = used_pepper + remaining_pepper :=
by sorry

end brennan_pepper_amount_l1807_180780


namespace eldest_child_age_l1807_180781

/-- Represents the ages of three grandchildren -/
structure GrandchildrenAges where
  youngest : ℕ
  middle : ℕ
  eldest : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : GrandchildrenAges) : Prop :=
  ages.middle = ages.youngest + 3 ∧
  ages.eldest = 3 * ages.youngest ∧
  ages.eldest = ages.youngest + ages.middle + 2

/-- The theorem stating that the eldest child's age is 15 years -/
theorem eldest_child_age (ages : GrandchildrenAges) :
  satisfiesConditions ages → ages.eldest = 15 := by
  sorry


end eldest_child_age_l1807_180781


namespace company_blocks_l1807_180768

/-- Given a company with the following properties:
  - The total amount for gifts is $4000
  - Each gift costs $4
  - There are approximately 100 workers per block
  Prove that the number of blocks in the company is 10 -/
theorem company_blocks (total_amount : ℕ) (gift_cost : ℕ) (workers_per_block : ℕ) : 
  total_amount = 4000 →
  gift_cost = 4 →
  workers_per_block = 100 →
  (total_amount / gift_cost) / workers_per_block = 10 := by
  sorry

end company_blocks_l1807_180768


namespace same_gender_probability_theorem_l1807_180774

/-- Represents a school with a certain number of male and female teachers -/
structure School where
  male_teachers : ℕ
  female_teachers : ℕ

/-- The probability of selecting two teachers of the same gender from two schools -/
def same_gender_probability (school_a school_b : School) : ℚ :=
  let total_combinations := (school_a.male_teachers + school_a.female_teachers) * (school_b.male_teachers + school_b.female_teachers)
  let same_gender_combinations := school_a.male_teachers * school_b.male_teachers + school_a.female_teachers * school_b.female_teachers
  same_gender_combinations / total_combinations

/-- Theorem stating that the probability of selecting two teachers of the same gender
    from the given schools is 4/9 -/
theorem same_gender_probability_theorem :
  let school_a := School.mk 2 1
  let school_b := School.mk 1 2
  same_gender_probability school_a school_b = 4 / 9 := by
  sorry

end same_gender_probability_theorem_l1807_180774


namespace tournament_participants_l1807_180748

theorem tournament_participants :
  ∃ n : ℕ,
    n > 0 ∧
    (n - 2) * (n - 3) / 2 + 7 = 62 ∧
    n = 13 :=
by sorry

end tournament_participants_l1807_180748


namespace koby_sparklers_count_l1807_180762

/-- The number of boxes Koby has -/
def koby_boxes : ℕ := 2

/-- The number of boxes Cherie has -/
def cherie_boxes : ℕ := 1

/-- The number of whistlers in each of Koby's boxes -/
def koby_whistlers_per_box : ℕ := 5

/-- The number of sparklers in Cherie's box -/
def cherie_sparklers : ℕ := 8

/-- The number of whistlers in Cherie's box -/
def cherie_whistlers : ℕ := 9

/-- The total number of fireworks Koby and Cherie have -/
def total_fireworks : ℕ := 33

/-- The number of sparklers in each of Koby's boxes -/
def koby_sparklers_per_box : ℕ := 3

theorem koby_sparklers_count :
  koby_sparklers_per_box * koby_boxes +
  cherie_sparklers +
  koby_whistlers_per_box * koby_boxes +
  cherie_whistlers = total_fireworks :=
by sorry

end koby_sparklers_count_l1807_180762


namespace a_1998_value_l1807_180714

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n m, n < m → a n < a m) ∧ 
  (∀ k : ℕ, ∃! (i j k : ℕ), k = a i + 2 * a j + 4 * a k)

theorem a_1998_value (a : ℕ → ℕ) (h : is_valid_sequence a) : a 1998 = 1227096648 := by
  sorry

end a_1998_value_l1807_180714
