import Mathlib

namespace NUMINAMATH_CALUDE_gcd_lcm_sum_36_2310_l1905_190555

theorem gcd_lcm_sum_36_2310 : Nat.gcd 36 2310 + Nat.lcm 36 2310 = 13866 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_36_2310_l1905_190555


namespace NUMINAMATH_CALUDE_subtract_negative_add_l1905_190569

theorem subtract_negative_add : 3 - (-5) + 7 = 15 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_add_l1905_190569


namespace NUMINAMATH_CALUDE_total_pears_is_five_l1905_190500

/-- The number of pears Keith picked -/
def keith_pears : ℕ := 3

/-- The number of pears Jason picked -/
def jason_pears : ℕ := 2

/-- The total number of pears picked -/
def total_pears : ℕ := keith_pears + jason_pears

/-- Theorem: The total number of pears picked is 5 -/
theorem total_pears_is_five : total_pears = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_is_five_l1905_190500


namespace NUMINAMATH_CALUDE_right_triangle_3_4_5_l1905_190595

/-- A triangle with side lengths 3, 4, and 5 is a right triangle. -/
theorem right_triangle_3_4_5 :
  ∀ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 →
  a^2 + b^2 = c^2 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_3_4_5_l1905_190595


namespace NUMINAMATH_CALUDE_investment_return_calculation_l1905_190505

theorem investment_return_calculation (total_investment small_investment large_investment : ℝ)
  (combined_return_rate small_return_rate : ℝ) :
  total_investment = small_investment + large_investment →
  small_investment = 500 →
  large_investment = 1500 →
  combined_return_rate = 0.085 →
  small_return_rate = 0.07 →
  (combined_return_rate * total_investment = small_return_rate * small_investment + 
    (large_investment * 0.09)) :=
by sorry

end NUMINAMATH_CALUDE_investment_return_calculation_l1905_190505


namespace NUMINAMATH_CALUDE_half_plus_five_equals_eleven_l1905_190528

theorem half_plus_five_equals_eleven (n : ℝ) : (1/2 : ℝ) * n + 5 = 11 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_five_equals_eleven_l1905_190528


namespace NUMINAMATH_CALUDE_chord_length_circle_line_intersection_chord_length_proof_l1905_190594

/-- The chord length cut by the line y = x from the circle x^2 + (y+2)^2 = 4 is 2√2 -/
theorem chord_length_circle_line_intersection : Real → Prop :=
  λ chord_length =>
    let circle := λ x y => x^2 + (y+2)^2 = 4
    let line := λ x y => y = x
    ∃ x₁ y₁ x₂ y₂,
      circle x₁ y₁ ∧ circle x₂ y₂ ∧
      line x₁ y₁ ∧ line x₂ y₂ ∧
      chord_length = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ∧
      chord_length = 2 * Real.sqrt 2

/-- Proof of the chord length theorem -/
theorem chord_length_proof : chord_length_circle_line_intersection (2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_circle_line_intersection_chord_length_proof_l1905_190594


namespace NUMINAMATH_CALUDE_square_ratio_side_lengths_l1905_190540

theorem square_ratio_side_lengths :
  let area_ratio : ℚ := 8 / 125
  let side_ratio : ℝ := Real.sqrt (area_ratio)
  let rationalized_ratio : ℝ := side_ratio * Real.sqrt 5 / Real.sqrt 5
  rationalized_ratio = 2 * Real.sqrt 10 / 25 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_side_lengths_l1905_190540


namespace NUMINAMATH_CALUDE_technicians_count_l1905_190578

/-- Represents the workshop scenario with workers and salaries -/
structure Workshop where
  total_workers : ℕ
  avg_salary : ℚ
  technician_salary : ℚ
  other_salary : ℚ

/-- Calculates the number of technicians in the workshop -/
def num_technicians (w : Workshop) : ℚ :=
  ((w.avg_salary - w.other_salary) * w.total_workers) / (w.technician_salary - w.other_salary)

/-- The given workshop scenario -/
def given_workshop : Workshop :=
  { total_workers := 22
    avg_salary := 850
    technician_salary := 1000
    other_salary := 780 }

/-- Theorem stating that the number of technicians in the given workshop is 7 -/
theorem technicians_count :
  num_technicians given_workshop = 7 := by
  sorry


end NUMINAMATH_CALUDE_technicians_count_l1905_190578


namespace NUMINAMATH_CALUDE_water_balloon_count_l1905_190513

/-- The total number of filled water balloons Max and Zach have -/
def total_filled_balloons (max_time max_rate zach_time zach_rate popped : ℕ) : ℕ :=
  max_time * max_rate + zach_time * zach_rate - popped

/-- Theorem: The total number of filled water balloons Max and Zach have is 170 -/
theorem water_balloon_count : total_filled_balloons 30 2 40 3 10 = 170 := by
  sorry

end NUMINAMATH_CALUDE_water_balloon_count_l1905_190513


namespace NUMINAMATH_CALUDE_students_left_l1905_190568

theorem students_left (initial_students new_students final_students : ℕ) :
  initial_students = 8 →
  new_students = 8 →
  final_students = 11 →
  initial_students + new_students - final_students = 5 := by
sorry

end NUMINAMATH_CALUDE_students_left_l1905_190568


namespace NUMINAMATH_CALUDE_largest_divisor_of_m_l1905_190565

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 39 ∣ m^2) :
  39 = Nat.gcd 39 m := by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_m_l1905_190565


namespace NUMINAMATH_CALUDE_inequality_solution_l1905_190503

theorem inequality_solution : 
  ∃! (n : ℕ), n ≥ 3 ∧ 
  (∀ (x : ℝ), x ≥ 3 → 
    (Real.sqrt (5 * x - 11) - Real.sqrt (5 * x^2 - 21 * x + 21) ≥ 5 * x^2 - 26 * x + 32) →
    x = n) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1905_190503


namespace NUMINAMATH_CALUDE_simplify_power_l1905_190554

theorem simplify_power (y : ℝ) : (3 * y^4)^4 = 81 * y^16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_l1905_190554


namespace NUMINAMATH_CALUDE_polynomial_descending_order_x_l1905_190592

theorem polynomial_descending_order_x (x y : ℝ) :
  3 * x * y^2 - 2 * x^2 * y - x^3 * y^3 - 4 =
  -x^3 * y^3 - 2 * x^2 * y + 3 * x * y^2 - 4 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_descending_order_x_l1905_190592


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l1905_190566

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l1905_190566


namespace NUMINAMATH_CALUDE_cartesian_coordinates_l1905_190534

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define planes and axes
def yOz_plane (p : Point3D) : Prop := p.x = 0
def z_axis (p : Point3D) : Prop := p.x = 0 ∧ p.y = 0
def xOz_plane (p : Point3D) : Prop := p.y = 0

-- Theorem statement
theorem cartesian_coordinates :
  (∃ (p : Point3D), yOz_plane p ∧ ∃ (b c : ℝ), p.y = b ∧ p.z = c) ∧
  (∃ (p : Point3D), z_axis p ∧ ∃ (c : ℝ), p.z = c) ∧
  (∃ (p : Point3D), xOz_plane p ∧ ∃ (a c : ℝ), p.x = a ∧ p.z = c) :=
by sorry

end NUMINAMATH_CALUDE_cartesian_coordinates_l1905_190534


namespace NUMINAMATH_CALUDE_equation_is_linear_l1905_190529

/-- A linear equation with two variables is of the form ax + by = c, where a, b, and c are constants, and a and b are not both zero. -/
def IsLinearEquationWithTwoVariables (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0), ∀ x y, f x y ↔ a * x + b * y = c

/-- The equation 2x = 3y + 1 -/
def Equation (x y : ℝ) : Prop := 2 * x = 3 * y + 1

theorem equation_is_linear : IsLinearEquationWithTwoVariables Equation := by
  sorry

end NUMINAMATH_CALUDE_equation_is_linear_l1905_190529


namespace NUMINAMATH_CALUDE_problem_statement_l1905_190524

/-- Floor function: greatest integer less than or equal to x -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The problem statement -/
theorem problem_statement :
  floor ((2015^2 : ℝ) / (2013 * 2014) - (2013^2 : ℝ) / (2014 * 2015)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1905_190524


namespace NUMINAMATH_CALUDE_sam_travel_distance_l1905_190579

/-- Given that Marguerite drove 150 miles in 3 hours, and Sam increased his speed by 20% and drove for 4 hours, prove that Sam traveled 240 miles. -/
theorem sam_travel_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) 
  (sam_speed_increase : ℝ) (sam_time : ℝ) :
  marguerite_distance = 150 →
  marguerite_time = 3 →
  sam_speed_increase = 0.2 →
  sam_time = 4 →
  (marguerite_distance / marguerite_time) * (1 + sam_speed_increase) * sam_time = 240 := by
  sorry

end NUMINAMATH_CALUDE_sam_travel_distance_l1905_190579


namespace NUMINAMATH_CALUDE_push_up_sets_l1905_190545

/-- Represents the number of push-ups done by each person -/
structure PushUps where
  zachary : ℕ
  david : ℕ
  emily : ℕ

/-- Calculates the number of complete sets of push-ups done together -/
def completeSets (p : PushUps) : ℕ :=
  1

theorem push_up_sets (p : PushUps) 
  (h1 : p.zachary = 47)
  (h2 : p.david = p.zachary + 15)
  (h3 : p.emily = 2 * p.david) :
  completeSets p = 1 := by
  sorry

#check push_up_sets

end NUMINAMATH_CALUDE_push_up_sets_l1905_190545


namespace NUMINAMATH_CALUDE_ellipse_equation_l1905_190516

/-- Given an ellipse with the specified properties, prove its equation -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (a^2 - b^2) / a^2 = 1/9) -- eccentricity = 1/3
  (F1 F2 P : ℝ × ℝ) -- foci and a point on the ellipse
  (h4 : (F1.1 - F2.1)^2 + (F1.2 - F2.2)^2 = 4 * (a^2 - b^2)) -- distance between foci
  (h5 : (P.1 - F1.1)^2 + (P.2 - F1.2)^2 + 
        (P.1 - F2.1)^2 + (P.2 - F2.2)^2 = 4 * a^2) -- sum of distances from foci
  (h6 : ((P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2)) / 
        (((P.1 - F1.1)^2 + (P.2 - F1.2)^2) * ((P.1 - F2.1)^2 + (P.2 - F2.2)^2))^(1/2) = 3/5) -- cos∠F1PF2
  (h7 : 1/2 * ((P.1 - F1.1) * (P.2 - F2.2) - (P.2 - F1.2) * (P.1 - F2.1)) = 4) -- area of triangle
  : a^2 = 9 ∧ b^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1905_190516


namespace NUMINAMATH_CALUDE_altitudes_5_12_13_impossible_l1905_190507

-- Define a function to check if three numbers can be altitudes of a triangle
def canBeAltitudes (a b c : ℝ) : Prop :=
  ∀ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 →
    x * a = y * b ∧ y * b = z * c →
    x + y > z ∧ y + z > x ∧ z + x > y

-- Theorem statement
theorem altitudes_5_12_13_impossible :
  ¬ canBeAltitudes 5 12 13 :=
sorry

end NUMINAMATH_CALUDE_altitudes_5_12_13_impossible_l1905_190507


namespace NUMINAMATH_CALUDE_unique_grid_solution_l1905_190535

-- Define the grid type
def Grid := List (List Nat)

-- Define the visibility type
def Visibility := List Nat

-- Function to check if a grid is valid
def is_valid_grid (g : Grid) : Prop := sorry

-- Function to check if visibility conditions are met
def meets_visibility (g : Grid) (v : Visibility) : Prop := sorry

-- Function to extract the four-digit number from the grid
def extract_number (g : Grid) : Nat := sorry

-- Theorem statement
theorem unique_grid_solution :
  ∀ (g : Grid) (v : Visibility),
    is_valid_grid g ∧ meets_visibility g v →
    extract_number g = 2213 := by sorry

end NUMINAMATH_CALUDE_unique_grid_solution_l1905_190535


namespace NUMINAMATH_CALUDE_triangle_circumradius_l1905_190512

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = √3 and c - 2b + 2√3 cos C = 0, then the radius of the circumcircle is 1. -/
theorem triangle_circumradius (a b c A B C : ℝ) : 
  a = Real.sqrt 3 →
  c - 2*b + 2*(Real.sqrt 3)*(Real.cos C) = 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  b / (Real.sin B) = c / (Real.sin C) →
  (a / (2 * Real.sin A)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumradius_l1905_190512


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l1905_190519

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  sides_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an excircle of a triangle -/
structure Excircle where
  center : ℝ × ℝ
  radius : ℝ

/-- The incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The incircle of a triangle -/
def incircle (t : Triangle) : Circle := sorry

/-- The excircles of a triangle -/
def excircles (t : Triangle) : Fin 3 → Excircle := sorry

/-- Checks if two circles are internally tangent -/
def is_internally_tangent (c1 c2 : Circle) : Prop := sorry

/-- The main theorem -/
theorem min_perimeter_triangle (t : Triangle) 
  (h1 : is_internally_tangent (incircle t) { center := (excircles t 0).center, radius := (excircles t 0).radius })
  (h2 : ¬ is_internally_tangent (incircle t) { center := (excircles t 1).center, radius := (excircles t 1).radius })
  (h3 : ¬ is_internally_tangent (incircle t) { center := (excircles t 2).center, radius := (excircles t 2).radius }) :
  t.a + t.b + t.c ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l1905_190519


namespace NUMINAMATH_CALUDE_shaded_shapes_area_l1905_190520

/-- Represents a point on a grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a shape on the grid -/
structure GridShape where
  vertices : List GridPoint

/-- The grid size -/
def gridSize : ℕ := 7

/-- Function to calculate the area of a shape on the grid -/
def calculateArea (shape : GridShape) : ℚ :=
  sorry

/-- The newly designed shaded shapes on the grid -/
def shadedShapes : List GridShape :=
  sorry

/-- Theorem stating that the total area of the shaded shapes is 3 -/
theorem shaded_shapes_area :
  (shadedShapes.map calculateArea).sum = 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_shapes_area_l1905_190520


namespace NUMINAMATH_CALUDE_triangle_construction_uniqueness_l1905_190571

/-- A point in the Euclidean plane -/
structure Point :=
  (x y : ℝ)

/-- A triangle in the Euclidean plane -/
structure Triangle :=
  (A B C : Point)

/-- The centroid of a triangle -/
def centroid (t : Triangle) : Point :=
  sorry

/-- The incenter of a triangle -/
def incenter (t : Triangle) : Point :=
  sorry

/-- The touch point of the incircle on a side of a triangle -/
def touchPoint (t : Triangle) : Point :=
  sorry

/-- Theorem: Given the centroid, incenter, and touch point of the incircle on a side,
    a unique triangle can be constructed -/
theorem triangle_construction_uniqueness 
  (M I Q_a : Point) : 
  ∃! t : Triangle, 
    centroid t = M ∧ 
    incenter t = I ∧ 
    touchPoint t = Q_a :=
  sorry

end NUMINAMATH_CALUDE_triangle_construction_uniqueness_l1905_190571


namespace NUMINAMATH_CALUDE_absolute_value_zero_l1905_190560

theorem absolute_value_zero (x : ℚ) : |4*x + 6| = 0 ↔ x = -3/2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_zero_l1905_190560


namespace NUMINAMATH_CALUDE_megan_initial_markers_l1905_190543

/-- The number of markers Megan initially had -/
def initial_markers : ℕ := sorry

/-- The number of markers Robert gave to Megan -/
def roberts_markers : ℕ := 109

/-- The total number of markers Megan has after receiving markers from Robert -/
def total_markers : ℕ := 326

/-- Theorem stating that the initial number of markers Megan had is 217 -/
theorem megan_initial_markers : initial_markers = 217 := by
  sorry

end NUMINAMATH_CALUDE_megan_initial_markers_l1905_190543


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_36_degrees_l1905_190589

def angle_measure : ℝ := 36

def complement (x : ℝ) : ℝ := 90 - x

def supplement (x : ℝ) : ℝ := 180 - x

theorem supplement_of_complement_of_36_degrees : 
  supplement (complement angle_measure) = 126 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_36_degrees_l1905_190589


namespace NUMINAMATH_CALUDE_M_when_a_is_one_M_union_N_equals_N_l1905_190596

-- Define the set M as a function of a
def M (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}

-- Define the set N
def N : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Theorem 1: When a = 1, M is the open interval (0, 2)
theorem M_when_a_is_one : M 1 = {x | 0 < x ∧ x < 2} := by sorry

-- Theorem 2: M ∪ N = N if and only if a ∈ [-1, 2]
theorem M_union_N_equals_N (a : ℝ) : M a ∪ N = N ↔ -1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_M_when_a_is_one_M_union_N_equals_N_l1905_190596


namespace NUMINAMATH_CALUDE_fan_daily_usage_l1905_190573

/-- Calculates the daily usage of an electric fan given its power, monthly energy consumption, and days in a month -/
theorem fan_daily_usage 
  (fan_power : ℝ) 
  (monthly_energy : ℝ) 
  (days_in_month : ℕ) 
  (h1 : fan_power = 75) 
  (h2 : monthly_energy = 18) 
  (h3 : days_in_month = 30) : 
  (monthly_energy * 1000) / (fan_power * days_in_month) = 8 := by
  sorry

end NUMINAMATH_CALUDE_fan_daily_usage_l1905_190573


namespace NUMINAMATH_CALUDE_player_A_wins_l1905_190550

/-- Represents a game state with three piles of matches -/
structure GameState where
  pile1 : Nat
  pile2 : Nat
  pile3 : Nat

/-- Represents a player in the game -/
inductive Player
  | A
  | B

/-- Defines a valid move in the game -/
def ValidMove (state : GameState) (newState : GameState) : Prop :=
  ∃ (i j : Fin 3) (k : Nat),
    i ≠ j ∧
    k > 0 ∧
    k < state.pile1 + state.pile2 + state.pile3 ∧
    newState.pile1 + newState.pile2 + newState.pile3 = state.pile1 + state.pile2 + state.pile3 - k

/-- Defines the winning condition for a player -/
def Wins (player : Player) (initialState : GameState) : Prop :=
  ∀ (state : GameState),
    state = initialState →
    ∃ (strategy : GameState → GameState),
      (∀ (s : GameState), ValidMove s (strategy s)) ∧
      (∀ (opponent : Player → GameState → GameState),
        (∀ (s : GameState), ValidMove s (opponent player s)) →
        ∃ (n : Nat), ¬ValidMove (Nat.iterate (λ s => opponent player (strategy s)) n initialState) (opponent player (Nat.iterate (λ s => opponent player (strategy s)) n initialState)))

/-- The main theorem stating that Player A has a winning strategy -/
theorem player_A_wins :
  Wins Player.A ⟨100, 200, 300⟩ := by
  sorry


end NUMINAMATH_CALUDE_player_A_wins_l1905_190550


namespace NUMINAMATH_CALUDE_sarah_toad_count_l1905_190576

/-- The number of toads each person has -/
structure ToadCount where
  tim : ℕ
  jim : ℕ
  sarah : ℕ

/-- Given conditions about toad counts -/
def toad_conditions (tc : ToadCount) : Prop :=
  tc.tim = 30 ∧ 
  tc.jim = tc.tim + 20 ∧ 
  tc.sarah = 2 * tc.jim

/-- Theorem stating Sarah has 100 toads under given conditions -/
theorem sarah_toad_count (tc : ToadCount) (h : toad_conditions tc) : tc.sarah = 100 := by
  sorry

end NUMINAMATH_CALUDE_sarah_toad_count_l1905_190576


namespace NUMINAMATH_CALUDE_investment_ratio_problem_l1905_190581

theorem investment_ratio_problem (profit_ratio_p profit_ratio_q : ℚ) 
  (investment_time_p investment_time_q : ℚ) 
  (investment_ratio_p investment_ratio_q : ℚ) : 
  profit_ratio_p / profit_ratio_q = 7 / 11 →
  investment_time_p = 5 →
  investment_time_q = 10.999999999999998 →
  (investment_ratio_p * investment_time_p) / (investment_ratio_q * investment_time_q) = profit_ratio_p / profit_ratio_q →
  investment_ratio_p / investment_ratio_q = 7 / 5 := by
sorry

end NUMINAMATH_CALUDE_investment_ratio_problem_l1905_190581


namespace NUMINAMATH_CALUDE_ryan_owns_eleven_twentyfourths_l1905_190548

/-- The fraction of the total amount that Ryan owns -/
def ryan_fraction (total : ℚ) (leo_final : ℚ) (ryan_debt : ℚ) (leo_debt : ℚ) : ℚ :=
  1 - (leo_final + leo_debt - ryan_debt) / total

theorem ryan_owns_eleven_twentyfourths :
  let total : ℚ := 48
  let leo_final : ℚ := 19
  let ryan_debt : ℚ := 10
  let leo_debt : ℚ := 7
  ryan_fraction total leo_final ryan_debt leo_debt = 11 / 24 := by
sorry

end NUMINAMATH_CALUDE_ryan_owns_eleven_twentyfourths_l1905_190548


namespace NUMINAMATH_CALUDE_dartboard_probability_l1905_190585

/-- The probability of hitting a specific region on a square dartboard -/
theorem dartboard_probability : 
  ∃ (square_side_length : ℝ) (region_area : ℝ → ℝ),
    square_side_length = 2 ∧
    (∀ x, x > 0 → region_area x = (π * x^2) / 4 - x^2 / 2) ∧
    region_area square_side_length / square_side_length^2 = (π - 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_probability_l1905_190585


namespace NUMINAMATH_CALUDE_max_product_sum_300_l1905_190501

theorem max_product_sum_300 :
  (∃ (a b : ℤ), a + b = 300 ∧ ∀ (x y : ℤ), x + y = 300 → x * y ≤ a * b) ∧
  (∀ (a b : ℤ), a + b = 300 → a * b ≤ 22500) :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l1905_190501


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l1905_190515

theorem least_number_with_remainder (n : ℕ) : n = 266 ↔ 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 33 ≠ 2 ∨ m % 8 ≠ 2)) ∧ 
  n % 33 = 2 ∧ n % 8 = 2 :=
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l1905_190515


namespace NUMINAMATH_CALUDE_plane_to_center_distance_l1905_190556

/-- Represents a point on the surface of a sphere -/
structure SpherePoint where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The distance between two points on the sphere -/
def distance (p q : SpherePoint) : ℝ := sorry

/-- The radius of the sphere -/
def sphereRadius : ℝ := 13

/-- Theorem: The distance from the plane passing through A, B, C to the sphere center -/
theorem plane_to_center_distance 
  (A B C : SpherePoint) 
  (h1 : distance A B = 6)
  (h2 : distance B C = 8)
  (h3 : distance C A = 10) :
  ∃ (d : ℝ), d = 12 ∧ d^2 + sphereRadius^2 = (distance A B)^2 + (distance B C)^2 + (distance C A)^2 := by
  sorry

end NUMINAMATH_CALUDE_plane_to_center_distance_l1905_190556


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l1905_190593

/-- Represents a rectangular piece of plywood -/
structure Plywood where
  length : ℝ
  width : ℝ

/-- Represents a cut of the plywood into congruent rectangles -/
structure Cut where
  num_pieces : ℕ
  piece_length : ℝ
  piece_width : ℝ

/-- Calculate the perimeter of a rectangular piece -/
def perimeter (l w : ℝ) : ℝ := 2 * (l + w)

/-- Check if a cut is valid for a given plywood -/
def is_valid_cut (p : Plywood) (c : Cut) : Prop :=
  c.num_pieces * c.piece_length = p.length ∧ 
  c.num_pieces * c.piece_width = p.width

/-- The main theorem -/
theorem plywood_cut_perimeter_difference 
  (p : Plywood) 
  (h1 : p.length = 10 ∧ p.width = 5) 
  (h2 : ∃ c : Cut, is_valid_cut p c ∧ c.num_pieces = 5) :
  ∃ (max_perim min_perim : ℝ),
    (∀ c : Cut, is_valid_cut p c ∧ c.num_pieces = 5 → 
      perimeter c.piece_length c.piece_width ≤ max_perim) ∧
    (∀ c : Cut, is_valid_cut p c ∧ c.num_pieces = 5 → 
      perimeter c.piece_length c.piece_width ≥ min_perim) ∧
    max_perim - min_perim = 8 := by
  sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l1905_190593


namespace NUMINAMATH_CALUDE_sum_inequality_l1905_190557

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a^2 + b^2 + c^2 = 1) :
  (1/a^2 + 1/b^2 + 1/c^2) ≥ ((4*b*c/(a^2 + 1) + 4*a*c/(b^2 + 1) + 4*a*b/(c^2 + 1)))^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l1905_190557


namespace NUMINAMATH_CALUDE_farmer_land_calculation_l1905_190552

theorem farmer_land_calculation (total_land : ℝ) : 
  0.2 * 0.5 * 0.9 * total_land = 252 → total_land = 2800 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_calculation_l1905_190552


namespace NUMINAMATH_CALUDE_quadratic_roots_and_exponential_inequality_l1905_190538

theorem quadratic_roots_and_exponential_inequality (a : ℝ) :
  (∃ x : ℝ, x^2 + 8*x + a^2 = 0) ∧ 
  (∀ x : ℝ, Real.exp x + 1 / Real.exp x > a) →
  -4 ≤ a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_exponential_inequality_l1905_190538


namespace NUMINAMATH_CALUDE_triangle_parallelogram_area_relation_l1905_190584

theorem triangle_parallelogram_area_relation (x : ℝ) :
  let triangle_base := x - 2
  let triangle_height := x - 2
  let parallelogram_base := x - 3
  let parallelogram_height := x + 4
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let parallelogram_area := parallelogram_base * parallelogram_height
  parallelogram_area = 3 * triangle_area →
  (∀ y : ℝ, (y - 8) * (y - 3) = 0 ↔ y = x) →
  8 + 3 = 11 :=
by sorry

end NUMINAMATH_CALUDE_triangle_parallelogram_area_relation_l1905_190584


namespace NUMINAMATH_CALUDE_cylinder_sphere_surface_area_l1905_190564

theorem cylinder_sphere_surface_area (r : ℝ) (h : ℝ) :
  h = 2 * r →
  (4 / 3) * Real.pi * r^3 = 4 * Real.sqrt 3 * Real.pi →
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 18 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_sphere_surface_area_l1905_190564


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1905_190502

theorem polynomial_evaluation : 7^4 - 4 * 7^3 + 6 * 7^2 - 4 * 7 + 1 = 1296 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1905_190502


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1905_190574

/-- A quadratic equation with roots -1 and 3 -/
theorem quadratic_equation_roots (x : ℝ) : 
  (x^2 - 2*x - 3 = 0) ↔ (x = -1 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1905_190574


namespace NUMINAMATH_CALUDE_kelly_games_left_l1905_190542

/-- Given that Kelly has 106 Nintendo games initially and gives away 64 games,
    prove that she will have 42 games left. -/
theorem kelly_games_left (initial_games : ℕ) (games_given_away : ℕ) 
    (h1 : initial_games = 106) (h2 : games_given_away = 64) : 
    initial_games - games_given_away = 42 := by
  sorry

end NUMINAMATH_CALUDE_kelly_games_left_l1905_190542


namespace NUMINAMATH_CALUDE_bird_nest_twigs_l1905_190533

theorem bird_nest_twigs (circle_twigs : ℕ) (found_fraction : ℚ) (remaining_twigs : ℕ) :
  circle_twigs = 12 →
  found_fraction = 1 / 3 →
  remaining_twigs = 48 →
  (circle_twigs : ℚ) * (1 - found_fraction) * (circle_twigs : ℚ) = (remaining_twigs : ℚ) →
  circle_twigs * found_fraction * (circle_twigs : ℚ) + (remaining_twigs : ℚ) = 18 * (circle_twigs : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_bird_nest_twigs_l1905_190533


namespace NUMINAMATH_CALUDE_star_symmetry_l1905_190531

/-- The star operation for real numbers -/
def star (a b : ℝ) : ℝ := (a^2 - b^2)^2

/-- Theorem: For all real x and y, (x² - y²) ⋆ (y² - x²) = 0 -/
theorem star_symmetry (x y : ℝ) : star (x^2 - y^2) (y^2 - x^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_symmetry_l1905_190531


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_product_l1905_190553

theorem quadratic_equation_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x ≠ y) →
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9) →
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x * y = 20) →
  p + q = 87 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_product_l1905_190553


namespace NUMINAMATH_CALUDE_clock_hands_angle_at_8_30_clock_hands_angle_at_8_30_is_75_l1905_190518

/-- The angle between clock hands at 8:30 -/
theorem clock_hands_angle_at_8_30 : ℝ :=
  let hours : ℝ := 8
  let minutes : ℝ := 30
  let degrees_per_hour : ℝ := 360 / 12
  let degrees_per_minute : ℝ := 360 / 60
  let hour_hand_angle : ℝ := hours * degrees_per_hour + (minutes / 60) * degrees_per_hour
  let minute_hand_angle : ℝ := minutes * degrees_per_minute
  let angle_diff : ℝ := |hour_hand_angle - minute_hand_angle|
  75

/-- The theorem stating that the angle between clock hands at 8:30 is 75 degrees -/
theorem clock_hands_angle_at_8_30_is_75 : clock_hands_angle_at_8_30 = 75 := by
  sorry

end NUMINAMATH_CALUDE_clock_hands_angle_at_8_30_clock_hands_angle_at_8_30_is_75_l1905_190518


namespace NUMINAMATH_CALUDE_divisibility_condition_l1905_190551

theorem divisibility_condition (m n : ℕ) :
  (1 + (m + n) * m) ∣ ((n + 1) * (m + n) - 1) ↔ (m = 0 ∨ m = 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1905_190551


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l1905_190514

theorem incorrect_observation_value 
  (n : ℕ) 
  (original_mean : ℝ) 
  (new_mean : ℝ) 
  (correct_value : ℝ) 
  (h_n : n = 50) 
  (h_original_mean : original_mean = 36) 
  (h_new_mean : new_mean = 36.02) 
  (h_correct_value : correct_value = 48) :
  ∃ (incorrect_value : ℝ), 
    n * new_mean = n * original_mean - incorrect_value + correct_value ∧ 
    incorrect_value = 47 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_observation_value_l1905_190514


namespace NUMINAMATH_CALUDE_roots_problem_l1905_190562

theorem roots_problem :
  (∀ x : ℝ, x ^ 2 = 0 → x = 0) ∧
  (∃ x : ℝ, x ≥ 0 ∧ x ^ 2 = 9 ∧ ∀ y : ℝ, y ≥ 0 ∧ y ^ 2 = 9 → x = y) ∧
  (∃ x : ℝ, x ^ 3 = (64 : ℝ).sqrt ∧ ∀ y : ℝ, y ^ 3 = (64 : ℝ).sqrt → x = y) :=
by sorry

end NUMINAMATH_CALUDE_roots_problem_l1905_190562


namespace NUMINAMATH_CALUDE_total_uniform_cost_is_355_l1905_190539

/-- Calculates the total cost of school uniforms for a student --/
def uniform_cost (num_uniforms : ℕ) (pants_cost : ℚ) (sock_cost : ℚ) : ℚ :=
  let shirt_cost := 2 * pants_cost
  let tie_cost := (1 / 5) * shirt_cost
  let single_uniform_cost := pants_cost + shirt_cost + tie_cost + sock_cost
  num_uniforms * single_uniform_cost

/-- Proves that the total cost of school uniforms for a student is $355 --/
theorem total_uniform_cost_is_355 :
  uniform_cost 5 20 3 = 355 := by
  sorry

#eval uniform_cost 5 20 3

end NUMINAMATH_CALUDE_total_uniform_cost_is_355_l1905_190539


namespace NUMINAMATH_CALUDE_system_solution_l1905_190547

theorem system_solution : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 → 
  (Real.log x / Real.log 4 - Real.log y / Real.log 2 = 0) ∧ 
  (x^2 - 5*y^2 + 4 = 0) → 
  ((x = 1 ∧ y = 1) ∨ (x = 4 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1905_190547


namespace NUMINAMATH_CALUDE_area_enclosed_by_midpoints_l1905_190575

/-- The area enclosed by midpoints of line segments with length 3 and endpoints on adjacent sides of a square with side length 3 -/
theorem area_enclosed_by_midpoints (square_side : ℝ) (segment_length : ℝ) : square_side = 3 → segment_length = 3 → 
  ∃ (area : ℝ), area = 9 - (9 * Real.pi / 16) := by
  sorry

end NUMINAMATH_CALUDE_area_enclosed_by_midpoints_l1905_190575


namespace NUMINAMATH_CALUDE_pirate_catch_time_l1905_190541

/-- Represents the pursuit problem between a pirate ship and a trading vessel -/
structure PursuitProblem where
  initial_distance : ℝ
  pirate_speed_initial : ℝ
  trader_speed : ℝ
  pursuit_start_time : ℝ
  speed_change_time : ℝ
  new_speed_ratio : ℝ

/-- Calculates the time at which the pirate ship catches the trading vessel -/
def catch_time (p : PursuitProblem) : ℝ :=
  sorry

/-- Theorem stating that the catch time for the given problem is 4:40 p.m. (16.67 hours) -/
theorem pirate_catch_time :
  let problem := PursuitProblem.mk 12 12 9 12 3 1.2
  catch_time problem = 16 + 2/3 :=
sorry

end NUMINAMATH_CALUDE_pirate_catch_time_l1905_190541


namespace NUMINAMATH_CALUDE_inner_triangle_area_l1905_190559

/-- Given a triangle ABC with sides a, b, c, and lines parallel to the sides drawn at a distance d from them,
    the area of the resulting inner triangle is (t - ds)^2 / t, where t is the area of ABC and s is its semi-perimeter. -/
theorem inner_triangle_area (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  let s := (a + b + c) / 2
  let t := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let inner_area := (t - d * s)^2 / t
  ∃ (inner_triangle_area : ℝ), inner_triangle_area = inner_area :=
by sorry

end NUMINAMATH_CALUDE_inner_triangle_area_l1905_190559


namespace NUMINAMATH_CALUDE_gcd_1343_816_l1905_190598

theorem gcd_1343_816 : Nat.gcd 1343 816 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1343_816_l1905_190598


namespace NUMINAMATH_CALUDE_broker_commission_rate_change_l1905_190526

/-- Proves that the new commission rate is 5% given the conditions of the problem -/
theorem broker_commission_rate_change
  (original_rate : ℝ)
  (business_slump : ℝ)
  (new_rate : ℝ)
  (h1 : original_rate = 0.04)
  (h2 : business_slump = 0.20000000000000007)
  (h3 : original_rate * (1 - business_slump) = new_rate) :
  new_rate = 0.05 := by
  sorry

#eval (0.04 / 0.7999999999999999 : Float)

end NUMINAMATH_CALUDE_broker_commission_rate_change_l1905_190526


namespace NUMINAMATH_CALUDE_condition_for_reciprocal_inequality_l1905_190523

theorem condition_for_reciprocal_inequality (a : ℝ) :
  (∀ a, (1 / a > 1 → a < 1)) ∧ (∃ a, a < 1 ∧ 1 / a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_condition_for_reciprocal_inequality_l1905_190523


namespace NUMINAMATH_CALUDE_second_mechanic_rate_calculation_l1905_190546

/-- Represents the hourly rate of the second mechanic -/
def second_mechanic_rate : ℝ := sorry

/-- The first mechanic's hourly rate -/
def first_mechanic_rate : ℝ := 45

/-- Total combined work hours -/
def total_hours : ℝ := 20

/-- Total charge for both mechanics -/
def total_charge : ℝ := 1100

/-- Hours worked by the second mechanic -/
def second_mechanic_hours : ℝ := 5

theorem second_mechanic_rate_calculation : 
  second_mechanic_rate = 85 :=
by
  sorry

#check second_mechanic_rate_calculation

end NUMINAMATH_CALUDE_second_mechanic_rate_calculation_l1905_190546


namespace NUMINAMATH_CALUDE_frac_less_one_necessary_not_sufficient_l1905_190577

theorem frac_less_one_necessary_not_sufficient (a : ℝ) :
  (∀ a, a > 1 → 1/a < 1) ∧ 
  (∃ a, 1/a < 1 ∧ ¬(a > 1)) :=
sorry

end NUMINAMATH_CALUDE_frac_less_one_necessary_not_sufficient_l1905_190577


namespace NUMINAMATH_CALUDE_triangle_rectangle_equal_area_l1905_190572

theorem triangle_rectangle_equal_area (h : ℝ) (h_pos : h > 0) :
  let triangle_base : ℝ := 24
  let triangle_area : ℝ := (1 / 2) * triangle_base * h
  let rectangle_base : ℝ := (1 / 2) * triangle_base
  let rectangle_area : ℝ := rectangle_base * h
  triangle_area = rectangle_area →
  rectangle_base = 12 := by
sorry


end NUMINAMATH_CALUDE_triangle_rectangle_equal_area_l1905_190572


namespace NUMINAMATH_CALUDE_article_cost_price_l1905_190517

/-- Given an article marked 15% above its cost price, sold at Rs. 462 with a discount of 25.603864734299517%, prove that the cost price of the article is Rs. 540. -/
theorem article_cost_price (cost_price : ℝ) : 
  let markup_percentage : ℝ := 0.15
  let selling_price : ℝ := 462
  let discount_percentage : ℝ := 25.603864734299517
  let marked_price : ℝ := cost_price * (1 + markup_percentage)
  let discounted_price : ℝ := marked_price * (1 - discount_percentage / 100)
  discounted_price = selling_price → cost_price = 540 := by
sorry

#eval (462 : ℚ) / (1 - 25.603864734299517 / 100) / 1.15

end NUMINAMATH_CALUDE_article_cost_price_l1905_190517


namespace NUMINAMATH_CALUDE_odd_sum_15_to_55_l1905_190508

theorem odd_sum_15_to_55 : 
  let a₁ : ℕ := 15  -- First term
  let d : ℕ := 4    -- Common difference
  let n : ℕ := (55 - 15) / d + 1  -- Number of terms
  let aₙ : ℕ := a₁ + (n - 1) * d  -- Last term
  (n : ℝ) / 2 * (a₁ + aₙ) = 385 :=
by sorry

end NUMINAMATH_CALUDE_odd_sum_15_to_55_l1905_190508


namespace NUMINAMATH_CALUDE_score_statistics_l1905_190563

def scores : List ℝ := [80, 85, 90, 95]
def frequencies : List ℕ := [4, 6, 8, 2]

def total_students : ℕ := frequencies.sum

def median (s : List ℝ) (f : List ℕ) : ℝ := sorry

def mode (s : List ℝ) (f : List ℕ) : ℝ := sorry

theorem score_statistics :
  median scores frequencies = 87.5 ∧ mode scores frequencies = 90 := by sorry

end NUMINAMATH_CALUDE_score_statistics_l1905_190563


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l1905_190583

theorem largest_n_with_unique_k : ∀ n : ℕ, n > 24 →
  ¬(∃! k : ℤ, (3 : ℚ)/7 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 8/19) ∧
  (∃! k : ℤ, (3 : ℚ)/7 < (24 : ℚ)/(24 + k) ∧ (24 : ℚ)/(24 + k) < 8/19) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l1905_190583


namespace NUMINAMATH_CALUDE_batter_distribution_l1905_190558

/-- Given two trays of batter where the second tray holds 20 cups less than the first,
    and the total amount is 500 cups, prove that the second tray holds 240 cups. -/
theorem batter_distribution (first_tray second_tray : ℕ) : 
  first_tray = second_tray + 20 →
  first_tray + second_tray = 500 →
  second_tray = 240 := by
sorry

end NUMINAMATH_CALUDE_batter_distribution_l1905_190558


namespace NUMINAMATH_CALUDE_martyrs_cemetery_distance_l1905_190586

/-- The distance from the school to the Martyrs' Cemetery in kilometers -/
def distance : ℝ := 216

/-- The scheduled time for the journey in minutes -/
def scheduledTime : ℝ := 180

/-- The time saved in minutes when increasing speed by one-fifth after 1 hour -/
def timeSaved1 : ℝ := 20

/-- The time saved in minutes when increasing speed by one-third after 72km -/
def timeSaved2 : ℝ := 30

/-- The distance traveled at original speed before increasing by one-third -/
def initialDistance : ℝ := 72

theorem martyrs_cemetery_distance :
  (distance = 216) ∧
  (scheduledTime * (1 - 5/6) = timeSaved1) ∧
  (scheduledTime * (1 - 3/4) > timeSaved2) ∧
  (initialDistance / (1 - 2/3) = distance) :=
sorry

end NUMINAMATH_CALUDE_martyrs_cemetery_distance_l1905_190586


namespace NUMINAMATH_CALUDE_addition_puzzle_l1905_190510

theorem addition_puzzle (P Q R : ℕ) : 
  P < 10 ∧ Q < 10 ∧ R < 10 →
  1000 * P + 100 * Q + 10 * P + R * 1000 + Q * 100 + Q * 10 + Q = 2009 →
  P + Q + R = 10 := by
  sorry

end NUMINAMATH_CALUDE_addition_puzzle_l1905_190510


namespace NUMINAMATH_CALUDE_negative_numbers_roots_l1905_190561

theorem negative_numbers_roots :
  (∀ x : ℝ, x < 0 → ¬∃ y : ℝ, y ^ 2 = x) ∧
  (∀ x : ℝ, x < 0 → ∃ y : ℝ, y ^ 3 = x) :=
by sorry

end NUMINAMATH_CALUDE_negative_numbers_roots_l1905_190561


namespace NUMINAMATH_CALUDE_one_more_tile_possible_exists_blocking_configuration_l1905_190511

/-- Represents a 4 × 6 grid -/
def Grid := Fin 4 → Fin 6 → Bool

/-- Represents an L-shaped tile -/
structure LTile :=
  (pos : Fin 4 × Fin 6)

/-- Checks if a tile placement is valid -/
def is_valid_placement (g : Grid) (t : LTile) : Prop :=
  sorry

/-- Places a tile on the grid -/
def place_tile (g : Grid) (t : LTile) : Grid :=
  sorry

/-- Theorem: After placing two tiles, one more can always be placed -/
theorem one_more_tile_possible (g : Grid) (t1 t2 : LTile) 
  (h1 : is_valid_placement g t1)
  (h2 : is_valid_placement (place_tile g t1) t2) :
  ∃ t3 : LTile, is_valid_placement (place_tile (place_tile g t1) t2) t3 :=
sorry

/-- Theorem: There exists a configuration of three tiles that blocks further placement -/
theorem exists_blocking_configuration :
  ∃ g : Grid, ∃ t1 t2 t3 : LTile,
    is_valid_placement g t1 ∧
    is_valid_placement (place_tile g t1) t2 ∧
    is_valid_placement (place_tile (place_tile g t1) t2) t3 ∧
    ∀ t4 : LTile, ¬is_valid_placement (place_tile (place_tile (place_tile g t1) t2) t3) t4 :=
sorry

end NUMINAMATH_CALUDE_one_more_tile_possible_exists_blocking_configuration_l1905_190511


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l1905_190504

/-- Represents a batsman's cricket performance -/
structure BatsmanStats where
  totalInnings : ℕ
  lastInningsScore : ℕ
  averageIncrease : ℚ
  notOutInnings : ℕ

/-- Calculates the average score of a batsman after their latest innings,
    considering 'not out' innings -/
def calculateAdjustedAverage (stats : BatsmanStats) : ℚ :=
  let totalRuns := stats.totalInnings * (stats.averageIncrease + 
    (stats.lastInningsScore / stats.totalInnings : ℚ))
  totalRuns / (stats.totalInnings - stats.notOutInnings : ℚ)

/-- Theorem stating that for a batsman with given statistics, 
    their adjusted average is approximately 88.64 -/
theorem batsman_average_theorem (stats : BatsmanStats) 
  (h1 : stats.totalInnings = 25)
  (h2 : stats.lastInningsScore = 150)
  (h3 : stats.averageIncrease = 3)
  (h4 : stats.notOutInnings = 3) :
  abs (calculateAdjustedAverage stats - 88.64) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_theorem_l1905_190504


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1905_190537

theorem sum_of_fractions : (3 : ℚ) / 10 + (3 : ℚ) / 1000 = 303 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1905_190537


namespace NUMINAMATH_CALUDE_no_negative_log_base_exists_positive_fraction_log_base_l1905_190525

-- Define the property of being a valid logarithm base
def IsValidLogBase (b : ℝ) : Prop := b > 0 ∧ b ≠ 1

-- Theorem 1: No negative number can be a valid logarithm base
theorem no_negative_log_base :
  ∀ b : ℝ, b < 0 → ¬(IsValidLogBase b) :=
sorry

-- Theorem 2: There exists a positive fraction that is a valid logarithm base
theorem exists_positive_fraction_log_base :
  ∃ b : ℝ, 0 < b ∧ b < 1 ∧ IsValidLogBase b :=
sorry

end NUMINAMATH_CALUDE_no_negative_log_base_exists_positive_fraction_log_base_l1905_190525


namespace NUMINAMATH_CALUDE_correct_number_of_pair_sets_l1905_190509

/-- The number of ways to form 6 pairs of balls with different colors -/
def number_of_pair_sets (green red blue : ℕ) : ℕ :=
  if green = 3 ∧ red = 4 ∧ blue = 5 then 1440 else 0

/-- Theorem stating the correct number of pair sets for the given ball counts -/
theorem correct_number_of_pair_sets :
  number_of_pair_sets 3 4 5 = 1440 := by sorry

end NUMINAMATH_CALUDE_correct_number_of_pair_sets_l1905_190509


namespace NUMINAMATH_CALUDE_student_age_problem_l1905_190580

theorem student_age_problem (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ) 
  (total_avg_age : ℕ) (group1_avg_age : ℕ) (group2_avg_age : ℕ) :
  total_students = 20 →
  group1_students = 9 →
  group2_students = 10 →
  total_avg_age = 20 →
  group1_avg_age = 11 →
  group2_avg_age = 24 →
  (total_students * total_avg_age) - (group1_students * group1_avg_age + group2_students * group2_avg_age) = 61 :=
by sorry

end NUMINAMATH_CALUDE_student_age_problem_l1905_190580


namespace NUMINAMATH_CALUDE_sin_3x_plus_pi_4_eq_cos_3x_minus_pi_12_l1905_190597

theorem sin_3x_plus_pi_4_eq_cos_3x_minus_pi_12 :
  ∀ x : ℝ, Real.sin (3 * x + π / 4) = Real.cos (3 * (x - π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_sin_3x_plus_pi_4_eq_cos_3x_minus_pi_12_l1905_190597


namespace NUMINAMATH_CALUDE_tan_theta_value_l1905_190587

theorem tan_theta_value (θ : Real) 
  (h : 2 * Real.cos (θ - π/3) = 3 * Real.cos θ) : 
  Real.tan θ = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l1905_190587


namespace NUMINAMATH_CALUDE_f_pi_sixth_l1905_190588

/-- The function f(x) = sin x + a cos x, where a < 0 and max f(x) = 2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x + a * Real.cos x

/-- Theorem stating that f(π/6) = -1 under given conditions -/
theorem f_pi_sixth (a : ℝ) (h1 : a < 0) (h2 : ∀ x, f a x ≤ 2) (h3 : ∃ x, f a x = 2) :
  f a (Real.pi / 6) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_pi_sixth_l1905_190588


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l1905_190544

theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples : ℝ) * (1 - 0.4) = 420 → initial_apples = 700 :=
by sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l1905_190544


namespace NUMINAMATH_CALUDE_not_square_p_cubed_plus_p_plus_one_l1905_190521

theorem not_square_p_cubed_plus_p_plus_one (p : ℕ) (hp : Prime p) :
  ¬ ∃ (n : ℕ), n^2 = p^3 + p + 1 := by
  sorry

end NUMINAMATH_CALUDE_not_square_p_cubed_plus_p_plus_one_l1905_190521


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1905_190506

/-- An arithmetic sequence with the given properties has a common difference of 2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) -- The arithmetic sequence
  (h1 : a 1 + a 5 = 10) -- First condition: a_1 + a_5 = 10
  (h2 : a 4 = 7) -- Second condition: a_4 = 7
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) -- Definition of arithmetic sequence
  : a 2 - a 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1905_190506


namespace NUMINAMATH_CALUDE_fishing_trip_total_l1905_190599

/-- The total number of fish caught by Brendan and his dad -/
def total_fish (morning_catch : ℕ) (thrown_back : ℕ) (afternoon_catch : ℕ) (dad_catch : ℕ) : ℕ :=
  (morning_catch + afternoon_catch - thrown_back) + dad_catch

/-- Theorem stating that the total number of fish caught is 23 -/
theorem fishing_trip_total : 
  total_fish 8 3 5 13 = 23 := by sorry

end NUMINAMATH_CALUDE_fishing_trip_total_l1905_190599


namespace NUMINAMATH_CALUDE_find_k_l1905_190582

theorem find_k : ∃ k : ℕ, 3 * 10 * 4 * k = Nat.factorial 9 ∧ k = 15120 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l1905_190582


namespace NUMINAMATH_CALUDE_mink_skins_per_coat_l1905_190549

theorem mink_skins_per_coat 
  (initial_minks : ℕ) 
  (babies_per_mink : ℕ) 
  (fraction_set_free : ℚ) 
  (coats_made : ℕ) :
  initial_minks = 30 →
  babies_per_mink = 6 →
  fraction_set_free = 1/2 →
  coats_made = 7 →
  (initial_minks * (1 + babies_per_mink) * (1 - fraction_set_free)) / coats_made = 15 := by
sorry

end NUMINAMATH_CALUDE_mink_skins_per_coat_l1905_190549


namespace NUMINAMATH_CALUDE_sarah_initial_trucks_l1905_190532

/-- The number of trucks Sarah gave to Jeff -/
def trucks_given_to_jeff : ℕ := 13

/-- The number of trucks Sarah has left -/
def trucks_left : ℕ := 38

/-- The initial number of trucks Sarah had -/
def initial_trucks : ℕ := trucks_given_to_jeff + trucks_left

theorem sarah_initial_trucks : initial_trucks = 51 := by sorry

end NUMINAMATH_CALUDE_sarah_initial_trucks_l1905_190532


namespace NUMINAMATH_CALUDE_intersection_condition_l1905_190590

def A (a : ℝ) : Set ℝ := {3, Real.sqrt a}
def B (a : ℝ) : Set ℝ := {1, a}

theorem intersection_condition (a : ℝ) : A a ∩ B a = {a} → a = 0 ∨ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l1905_190590


namespace NUMINAMATH_CALUDE_rhombus_side_length_l1905_190527

/-- For a rhombus with area K and one diagonal three times the length of the other,
    the side length s is equal to √(5K/3). -/
theorem rhombus_side_length (K : ℝ) (h : K > 0) :
  ∃ (d : ℝ), d > 0 ∧ 
  let s := Real.sqrt ((5 * K) / 3)
  let area := (1/2) * d * (3*d)
  area = K ∧ 
  s^2 = (d/2)^2 + (3*d/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l1905_190527


namespace NUMINAMATH_CALUDE_maryville_population_increase_l1905_190522

/-- The average number of people added each year in Maryville between 2000 and 2005 -/
def average_population_increase (pop_2000 pop_2005 : ℕ) : ℚ :=
  (pop_2005 - pop_2000 : ℚ) / 5

/-- Theorem stating that the average population increase in Maryville between 2000 and 2005 is 3400 -/
theorem maryville_population_increase :
  average_population_increase 450000 467000 = 3400 := by
  sorry

end NUMINAMATH_CALUDE_maryville_population_increase_l1905_190522


namespace NUMINAMATH_CALUDE_prob_three_red_cards_standard_deck_l1905_190570

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)

/-- A standard deck of cards -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    red_suits := 2,
    black_suits := 2 }

/-- The probability of drawing three red cards in succession from a standard deck -/
def prob_three_red_cards (d : Deck) : ℚ :=
  (d.red_suits * d.cards_per_suit : ℚ) / d.total_cards *
  ((d.red_suits * d.cards_per_suit - 1) : ℚ) / (d.total_cards - 1) *
  ((d.red_suits * d.cards_per_suit - 2) : ℚ) / (d.total_cards - 2)

/-- Theorem: The probability of drawing three red cards in succession from a standard deck is 2/17 -/
theorem prob_three_red_cards_standard_deck :
  prob_three_red_cards standard_deck = 2 / 17 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_red_cards_standard_deck_l1905_190570


namespace NUMINAMATH_CALUDE_math_test_difference_l1905_190567

theorem math_test_difference (total_questions word_problems addition_subtraction_problems steve_can_answer : ℕ) :
  total_questions = 45 →
  word_problems = 17 →
  addition_subtraction_problems = 28 →
  steve_can_answer = 38 →
  total_questions - steve_can_answer = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_math_test_difference_l1905_190567


namespace NUMINAMATH_CALUDE_future_age_difference_l1905_190530

/-- Proves that the number of years in the future when the father's age will be 20 years more than twice the son's age is 4, given the conditions stated in the problem. -/
theorem future_age_difference (father_age son_age x : ℕ) : 
  father_age = 44 →
  father_age = 4 * son_age + 4 →
  father_age + x = 2 * (son_age + x) + 20 →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_future_age_difference_l1905_190530


namespace NUMINAMATH_CALUDE_machine_production_time_l1905_190536

/-- Given a machine that produces 150 items in 2 hours, 
    prove that it takes 0.8 minutes to produce one item. -/
theorem machine_production_time : 
  let total_items : ℕ := 150
  let total_hours : ℝ := 2
  let minutes_per_hour : ℝ := 60
  let total_minutes : ℝ := total_hours * minutes_per_hour
  total_minutes / total_items = 0.8 := by sorry

end NUMINAMATH_CALUDE_machine_production_time_l1905_190536


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1905_190591

/-- Given a boat that travels 15 km/hr along a stream and 5 km/hr against the same stream,
    its speed in still water is 10 km/hr. -/
theorem boat_speed_in_still_water
  (along_stream : ℝ)
  (against_stream : ℝ)
  (h_along : along_stream = 15)
  (h_against : against_stream = 5) :
  (along_stream + against_stream) / 2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1905_190591
