import Mathlib

namespace NUMINAMATH_CALUDE_basis_iff_not_parallel_l3215_321540

def is_basis (e₁ e₂ : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ (v : ℝ × ℝ), v = (a * e₁.1 + b * e₂.1, a * e₁.2 + b * e₂.2)

def are_parallel (v₁ v₂ : ℝ × ℝ) : Prop :=
  v₁.1 * v₂.2 = v₁.2 * v₂.1

theorem basis_iff_not_parallel (e₁ e₂ : ℝ × ℝ) :
  is_basis e₁ e₂ ↔ ¬ are_parallel e₁ e₂ :=
sorry

end NUMINAMATH_CALUDE_basis_iff_not_parallel_l3215_321540


namespace NUMINAMATH_CALUDE_polynomial_problem_l3215_321524

theorem polynomial_problem (f : ℝ → ℝ) :
  (∃ (a b c d e : ℤ), ∀ x, f x = a*x^4 + b*x^3 + c*x^2 + d*x + e) →
  f (1 + Real.rpow 3 (1/3)) = 1 + Real.rpow 3 (1/3) →
  f (1 + Real.sqrt 3) = 7 + Real.sqrt 3 →
  ∀ x, f x = x^4 - 3*x^3 + 3*x^2 - 3*x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_problem_l3215_321524


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3215_321506

theorem rectangular_to_polar_conversion :
  ∀ (x y : ℝ),
    x = 3 ∧ y = -3 →
    ∃ (r θ : ℝ),
      r > 0 ∧
      0 ≤ θ ∧ θ < 2 * Real.pi ∧
      r = 3 * Real.sqrt 2 ∧
      θ = 7 * Real.pi / 4 ∧
      x = r * Real.cos θ ∧
      y = r * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3215_321506


namespace NUMINAMATH_CALUDE_boatworks_production_l3215_321577

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem boatworks_production : geometric_sum 5 3 6 = 1820 := by
  sorry

end NUMINAMATH_CALUDE_boatworks_production_l3215_321577


namespace NUMINAMATH_CALUDE_nested_fraction_equation_l3215_321501

theorem nested_fraction_equation (x : ℚ) : 
  3 + 1 / (2 + 1 / (3 + 3 / (4 + x))) = 53/16 → x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equation_l3215_321501


namespace NUMINAMATH_CALUDE_xy_min_max_l3215_321530

theorem xy_min_max (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) (h3 : -2 ≤ a ∧ a ≤ 2) :
  (∃ (x y : ℝ), x * y = -1 ∧ 
    ∀ (x' y' : ℝ), x' + y' = a → x'^2 + y'^2 = -a^2 + 2 → x' * y' ≥ -1) ∧
  (∃ (x y : ℝ), x * y = 1/3 ∧ 
    ∀ (x' y' : ℝ), x' + y' = a → x'^2 + y'^2 = -a^2 + 2 → x' * y' ≤ 1/3) :=
by sorry

end NUMINAMATH_CALUDE_xy_min_max_l3215_321530


namespace NUMINAMATH_CALUDE_max_square_pen_area_l3215_321522

/-- Given 36 feet of fencing, the maximum area of a square pen is 81 square feet. -/
theorem max_square_pen_area (fencing : ℝ) (h : fencing = 36) : 
  (fencing / 4) ^ 2 = 81 :=
by sorry

end NUMINAMATH_CALUDE_max_square_pen_area_l3215_321522


namespace NUMINAMATH_CALUDE_child_tickets_sold_l3215_321511

theorem child_tickets_sold (adult_price child_price total_tickets total_receipts : ℕ) 
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_tickets = 130)
  (h4 : total_receipts = 840) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_receipts ∧
    child_tickets = 90 := by
  sorry

end NUMINAMATH_CALUDE_child_tickets_sold_l3215_321511


namespace NUMINAMATH_CALUDE_cherry_picking_time_l3215_321590

/-- The time spent picking cherries by 王芳 and 李丽 -/
def picking_time : ℝ := 0.25

/-- 王芳's picking rate in kg/hour -/
def wang_rate : ℝ := 8

/-- 李丽's picking rate in kg/hour -/
def li_rate : ℝ := 7

/-- Amount of cherries 王芳 gives to 李丽 after picking -/
def transfer_amount : ℝ := 0.25

theorem cherry_picking_time :
  wang_rate * picking_time - transfer_amount = li_rate * picking_time :=
by sorry

end NUMINAMATH_CALUDE_cherry_picking_time_l3215_321590


namespace NUMINAMATH_CALUDE_bent_polygon_total_angle_l3215_321507

/-- For a regular polygon with n sides (n > 4), if each side is bent inward at an angle θ = 360°/(2n),
    then the total angle formed by all the bends is 180°. -/
theorem bent_polygon_total_angle (n : ℕ) (h : n > 4) :
  let θ : ℝ := 360 / (2 * n)
  n * θ = 180 := by sorry

end NUMINAMATH_CALUDE_bent_polygon_total_angle_l3215_321507


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3215_321566

/-- 
Given a triangle XYZ:
- ext_angle_x is the exterior angle at vertex X
- angle_y is the angle at vertex Y
- angle_z is the angle at vertex Z

This theorem states that if the exterior angle at X is 150° and the angle at Y is 140°, 
then the angle at Z must be 110°.
-/
theorem triangle_angle_calculation 
  (ext_angle_x angle_y angle_z : ℝ) 
  (h1 : ext_angle_x = 150)
  (h2 : angle_y = 140) :
  angle_z = 110 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3215_321566


namespace NUMINAMATH_CALUDE_average_weight_increase_l3215_321588

/-- Proves that replacing a person weighing 47 kg with a person weighing 68 kg in a group of 6 people increases the average weight by 3.5 kg -/
theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 6 →
  old_weight = 47 →
  new_weight = 68 →
  (new_weight - old_weight) / initial_count = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3215_321588


namespace NUMINAMATH_CALUDE_leading_coefficient_is_negative_six_l3215_321565

def polynomial (x : ℝ) : ℝ :=
  -5 * (x^5 - 2*x^3 + x) + 8 * (x^5 + x^3 - 3) - 3 * (3*x^5 + x^3 + 2)

theorem leading_coefficient_is_negative_six :
  ∃ (p : ℝ → ℝ), ∀ x, ∃ (r : ℝ), polynomial x = -6 * x^5 + r ∧ (∀ y, |y| ≥ 1 → |r| ≤ |y|^5 * |-6 * x^5|) :=
sorry

end NUMINAMATH_CALUDE_leading_coefficient_is_negative_six_l3215_321565


namespace NUMINAMATH_CALUDE_supermarket_bread_count_l3215_321513

/-- Calculates the final number of loaves in a supermarket after sales and delivery -/
def final_loaves (initial : ℕ) (sold : ℕ) (delivered : ℕ) : ℕ :=
  initial - sold + delivered

/-- Theorem stating that given the specific numbers from the problem, 
    the final number of loaves is 2215 -/
theorem supermarket_bread_count : 
  final_loaves 2355 629 489 = 2215 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_bread_count_l3215_321513


namespace NUMINAMATH_CALUDE_three_number_ratio_sum_l3215_321584

theorem three_number_ratio_sum (a b c : ℝ) : 
  (a : ℝ) > 0 ∧ b = 2 * a ∧ c = 4 * a ∧ a^2 + b^2 + c^2 = 1701 →
  a + b + c = 63 := by
sorry

end NUMINAMATH_CALUDE_three_number_ratio_sum_l3215_321584


namespace NUMINAMATH_CALUDE_f_extrema_l3215_321574

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (cos (3*x/2), sin (3*x/2))
noncomputable def b (x : ℝ) : ℝ × ℝ := (cos (x/2), -sin (x/2))

noncomputable def f (x : ℝ) : ℝ := 
  (a x).1 * (b x).1 + (a x).2 * (b x).2 - 
  Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2

theorem f_extrema :
  ∀ x ∈ Set.Icc (-π/3) (π/4),
    (∀ y ∈ Set.Icc (-π/3) (π/4), f y ≤ -1) ∧
    (∃ y ∈ Set.Icc (-π/3) (π/4), f y = -1) ∧
    (∀ y ∈ Set.Icc (-π/3) (π/4), f y ≥ -Real.sqrt 2) ∧
    (∃ y ∈ Set.Icc (-π/3) (π/4), f y = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l3215_321574


namespace NUMINAMATH_CALUDE_solve_transportation_problem_l3215_321560

/-- Represents the daily transportation problem for building materials -/
structure TransportationProblem where
  daily_requirement : ℕ
  max_supply_A : ℕ
  max_supply_B : ℕ
  cost_scenario1 : ℕ
  cost_scenario2 : ℕ

/-- Represents the solution to the transportation problem -/
structure TransportationSolution where
  cost_per_ton_A : ℕ
  cost_per_ton_B : ℕ
  min_total_cost : ℕ
  optimal_tons_A : ℕ
  optimal_tons_B : ℕ

/-- Theorem stating the solution to the transportation problem -/
theorem solve_transportation_problem (p : TransportationProblem) 
  (h1 : p.daily_requirement = 120)
  (h2 : p.max_supply_A = 80)
  (h3 : p.max_supply_B = 90)
  (h4 : p.cost_scenario1 = 26000)
  (h5 : p.cost_scenario2 = 27000) :
  ∃ (s : TransportationSolution),
    s.cost_per_ton_A = 240 ∧
    s.cost_per_ton_B = 200 ∧
    s.min_total_cost = 25200 ∧
    s.optimal_tons_A = 30 ∧
    s.optimal_tons_B = 90 ∧
    s.optimal_tons_A + s.optimal_tons_B = p.daily_requirement ∧
    s.optimal_tons_A ≤ p.max_supply_A ∧
    s.optimal_tons_B ≤ p.max_supply_B ∧
    s.min_total_cost = s.cost_per_ton_A * s.optimal_tons_A + s.cost_per_ton_B * s.optimal_tons_B :=
by
  sorry


end NUMINAMATH_CALUDE_solve_transportation_problem_l3215_321560


namespace NUMINAMATH_CALUDE_jason_arm_tattoos_count_l3215_321589

-- Define the number of tattoos Jason has on each arm
def jason_arm_tattoos : ℕ := sorry

-- Define the number of tattoos Jason has on each leg
def jason_leg_tattoos : ℕ := 3

-- Define the total number of tattoos Jason has
def jason_total_tattoos : ℕ := 2 * jason_arm_tattoos + 2 * jason_leg_tattoos

-- Define the number of tattoos Adam has
def adam_tattoos : ℕ := 23

-- Theorem to prove
theorem jason_arm_tattoos_count :
  jason_arm_tattoos = 2 ∧
  adam_tattoos = 2 * jason_total_tattoos + 3 :=
by sorry

end NUMINAMATH_CALUDE_jason_arm_tattoos_count_l3215_321589


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_l3215_321556

-- Define the equation
def equation (x y : ℝ) : Prop := x^2 - y^2 = 0

-- Define what it means to be a straight line
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

-- Theorem statement
theorem equation_represents_two_lines :
  ∃ f g : ℝ → ℝ, 
    (is_straight_line f ∧ is_straight_line g) ∧
    (∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_l3215_321556


namespace NUMINAMATH_CALUDE_divisors_of_8820_multiple_of_3_and_5_l3215_321563

def number_of_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_8820_multiple_of_3_and_5 : 
  number_of_divisors 8820 = 18 := by sorry

end NUMINAMATH_CALUDE_divisors_of_8820_multiple_of_3_and_5_l3215_321563


namespace NUMINAMATH_CALUDE_farmer_problem_solution_l3215_321582

/-- A farmer sells ducks and chickens and buys a wheelbarrow -/
def FarmerProblem (duck_price chicken_price : ℕ) (duck_sold chicken_sold : ℕ) (wheelbarrow_profit : ℕ) :=
  let total_earnings := duck_price * duck_sold + chicken_price * chicken_sold
  let wheelbarrow_cost := wheelbarrow_profit / 2
  (wheelbarrow_cost : ℚ) / total_earnings = 1 / 2

theorem farmer_problem_solution :
  FarmerProblem 10 8 2 5 60 := by sorry

end NUMINAMATH_CALUDE_farmer_problem_solution_l3215_321582


namespace NUMINAMATH_CALUDE_jack_john_vote_difference_l3215_321583

/-- Calculates the number of votes Jack received more than John in an election with given conditions. -/
theorem jack_john_vote_difference :
  let total_votes : ℕ := 1150
  let john_votes : ℕ := 150
  let remaining_votes : ℕ := total_votes - john_votes
  let james_votes : ℕ := (7 * remaining_votes) / 10
  let jacob_votes : ℕ := (3 * (john_votes + james_votes)) / 10
  let joey_votes : ℕ := ((125 * jacob_votes) + 50) / 100
  let jack_votes : ℕ := (95 * joey_votes) / 100
  jack_votes - john_votes = 153 := by sorry

end NUMINAMATH_CALUDE_jack_john_vote_difference_l3215_321583


namespace NUMINAMATH_CALUDE_point_in_quadrants_I_and_II_l3215_321509

-- Define the quadrants
def QuadrantI (x y : ℝ) : Prop := x > 0 ∧ y > 0
def QuadrantII (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Define the inequalities
def Inequality1 (x y : ℝ) : Prop := y > -3 * x
def Inequality2 (x y : ℝ) : Prop := y > x + 2

-- Theorem statement
theorem point_in_quadrants_I_and_II (x y : ℝ) :
  Inequality1 x y ∧ Inequality2 x y → QuadrantI x y ∨ QuadrantII x y :=
by sorry

end NUMINAMATH_CALUDE_point_in_quadrants_I_and_II_l3215_321509


namespace NUMINAMATH_CALUDE_stick_cutting_l3215_321533

theorem stick_cutting (short_length long_length : ℝ) : 
  short_length > 0 →
  long_length = short_length + 12 →
  short_length + long_length = 20 →
  (long_length / short_length : ℝ) = 4 := by
sorry

end NUMINAMATH_CALUDE_stick_cutting_l3215_321533


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l3215_321558

theorem solution_satisfies_system :
  let x : ℚ := 7/2
  let y : ℚ := 1/2
  (2 * x + 4 * y = 9) ∧ (3 * x - 5 * y = 8) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l3215_321558


namespace NUMINAMATH_CALUDE_parabola_passes_origin_l3215_321545

/-- A parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

/-- The origin point (0, 0) -/
def origin : Point :=
  { x := 0, y := 0 }

/-- Check if a point lies on a parabola -/
def lies_on (p : Point) (para : Parabola) : Prop :=
  p.y = para.f p.x

/-- The given parabola y = (x+2)^2 -/
def given_parabola : Parabola :=
  { f := λ x => (x + 2)^2 }

/-- Theorem: Rightward translation by 2 units makes the parabola pass through the origin -/
theorem parabola_passes_origin :
  ∃ (p : Point), lies_on (translate p 2 0) given_parabola ∧ p = origin := by
  sorry

end NUMINAMATH_CALUDE_parabola_passes_origin_l3215_321545


namespace NUMINAMATH_CALUDE_division_problem_l3215_321586

theorem division_problem (number : ℕ) : 
  (number / 20 = 6) ∧ (number % 20 = 2) → number = 122 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3215_321586


namespace NUMINAMATH_CALUDE_final_value_l3215_321526

/-- The value of A based on bundles --/
def A : ℕ := 6 * 1000 + 36 * 100

/-- The value of B based on jumping twice --/
def B : ℕ := 876 - 197 - 197

/-- Theorem stating the final result --/
theorem final_value : A - B = 9118 := by
  sorry

end NUMINAMATH_CALUDE_final_value_l3215_321526


namespace NUMINAMATH_CALUDE_paths_on_4x10_grid_with_forbidden_segments_l3215_321561

/-- Represents a grid with forbidden segments -/
structure Grid where
  height : ℕ
  width : ℕ
  forbidden_segments : ℕ

/-- Calculates the number of paths on a grid with forbidden segments -/
def count_paths (g : Grid) : ℕ :=
  let total_paths := Nat.choose (g.height + g.width) g.height
  let forbidden_paths := g.forbidden_segments * (Nat.choose (g.height + g.width / 2 - 2) (g.height - 2) * Nat.choose (g.width / 2 + 2) 2)
  total_paths - forbidden_paths

/-- Theorem stating the number of paths on a 4x10 grid with two forbidden segments -/
theorem paths_on_4x10_grid_with_forbidden_segments :
  count_paths { height := 4, width := 10, forbidden_segments := 2 } = 161 := by
  sorry

end NUMINAMATH_CALUDE_paths_on_4x10_grid_with_forbidden_segments_l3215_321561


namespace NUMINAMATH_CALUDE_cube_root_scaling_l3215_321548

theorem cube_root_scaling (a b c d : ℝ) (ha : a > 0) (hc : c > 0) :
  (a^(1/3) = b) → (c^(1/3) = d) →
  ((1000 * a)^(1/3) = 10 * b) ∧ ((-0.001 * c)^(1/3) = -0.1 * d) := by
  sorry

/- The theorem above captures the essence of the problem without directly using the specific numbers.
   It shows the scaling properties of cube roots that are used to solve the original problem. -/

end NUMINAMATH_CALUDE_cube_root_scaling_l3215_321548


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l3215_321576

theorem polynomial_multiplication (x : ℝ) :
  (x^4 + 50*x^2 + 625) * (x^2 - 25) = x^6 - 75*x^4 + 1875*x^2 - 15625 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l3215_321576


namespace NUMINAMATH_CALUDE_gold_quarter_value_ratio_is_80_l3215_321594

/-- Represents the ratio of melted gold value to face value for gold quarters -/
def gold_quarter_value_ratio : ℚ :=
  let quarter_weight : ℚ := 1 / 5
  let melted_gold_value_per_ounce : ℚ := 100
  let quarter_face_value : ℚ := 1 / 4
  let quarters_per_ounce : ℚ := 1 / quarter_weight
  let melted_value_per_quarter : ℚ := melted_gold_value_per_ounce * quarter_weight
  melted_value_per_quarter / quarter_face_value

/-- Theorem stating that the gold quarter value ratio is 80 -/
theorem gold_quarter_value_ratio_is_80 : gold_quarter_value_ratio = 80 := by
  sorry

end NUMINAMATH_CALUDE_gold_quarter_value_ratio_is_80_l3215_321594


namespace NUMINAMATH_CALUDE_min_value_theorem_l3215_321504

theorem min_value_theorem (x : ℝ) (h : x > 1) :
  (x + 8) / Real.sqrt (x - 1) ≥ 6 ∧
  ((x + 8) / Real.sqrt (x - 1) = 6 ↔ x = 10) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3215_321504


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_10_and_6_l3215_321547

theorem smallest_common_multiple_of_10_and_6 : ∃ n : ℕ+, (∀ m : ℕ+, (10 ∣ m) ∧ (6 ∣ m) → n ≤ m) ∧ (10 ∣ n) ∧ (6 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_10_and_6_l3215_321547


namespace NUMINAMATH_CALUDE_angle_between_vectors_l3215_321592

def vector1 : Fin 2 → ℝ := ![2, 5]
def vector2 : Fin 2 → ℝ := ![-3, 7]

theorem angle_between_vectors (v1 v2 : Fin 2 → ℝ) :
  v1 = vector1 → v2 = vector2 →
  Real.arccos ((v1 0 * v2 0 + v1 1 * v2 1) /
    (Real.sqrt (v1 0 ^ 2 + v1 1 ^ 2) * Real.sqrt (v2 0 ^ 2 + v2 1 ^ 2))) =
  45 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l3215_321592


namespace NUMINAMATH_CALUDE_rational_point_coloring_l3215_321529

/-- A coloring function for rational points in the plane -/
def coloringFunction (n : ℕ) (p : ℚ × ℚ) : Fin n :=
  sorry

/-- A predicate to check if a point is on a line segment -/
def isOnLineSegment (p q r : ℚ × ℚ) : Prop :=
  sorry

theorem rational_point_coloring (n : ℕ) (hn : n > 0) :
  ∃ (f : ℚ × ℚ → Fin n),
    ∀ (p q : ℚ × ℚ) (c : Fin n),
      ∃ (r : ℚ × ℚ), isOnLineSegment p q r ∧ f r = c :=
sorry

end NUMINAMATH_CALUDE_rational_point_coloring_l3215_321529


namespace NUMINAMATH_CALUDE_abs_neg_eight_l3215_321520

theorem abs_neg_eight : |(-8 : ℤ)| = 8 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_eight_l3215_321520


namespace NUMINAMATH_CALUDE_sum_of_consecutive_integers_l3215_321546

theorem sum_of_consecutive_integers :
  let start : Int := -9
  let count : Nat := 20
  let sequence := List.range count |>.map (λ i => start + i)
  sequence.sum = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_integers_l3215_321546


namespace NUMINAMATH_CALUDE_balloon_distribution_difference_l3215_321562

/-- Represents the number of balloons of each color brought by a person -/
structure Balloons :=
  (red : ℕ)
  (blue : ℕ)
  (green : ℕ)

/-- Calculates the total number of balloons -/
def totalBalloons (b : Balloons) : ℕ := b.red + b.blue + b.green

theorem balloon_distribution_difference :
  let allan_brought := Balloons.mk 150 75 30
  let jake_brought := Balloons.mk 100 50 45
  let allan_forgot := 25
  let allan_distributed := totalBalloons { red := allan_brought.red,
                                           blue := allan_brought.blue - allan_forgot,
                                           green := allan_brought.green }
  let jake_distributed := totalBalloons jake_brought
  allan_distributed - jake_distributed = 35 := by sorry

end NUMINAMATH_CALUDE_balloon_distribution_difference_l3215_321562


namespace NUMINAMATH_CALUDE_sin_120_degrees_l3215_321536

theorem sin_120_degrees : Real.sin (2 * π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l3215_321536


namespace NUMINAMATH_CALUDE_ratio_value_l3215_321502

theorem ratio_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  a * c / (b * d) = 20 := by
sorry

end NUMINAMATH_CALUDE_ratio_value_l3215_321502


namespace NUMINAMATH_CALUDE_roger_spent_calculation_l3215_321542

/-- Calculates the amount of money Roger spent given his initial amount,
    the amount received from his mom, and his current amount. -/
def money_spent (initial : ℕ) (received : ℕ) (current : ℕ) : ℕ :=
  initial + received - current

theorem roger_spent_calculation :
  money_spent 45 46 71 = 20 := by
  sorry

end NUMINAMATH_CALUDE_roger_spent_calculation_l3215_321542


namespace NUMINAMATH_CALUDE_cylindrical_eight_queens_impossible_l3215_321593

/-- Represents a position on the cylindrical chessboard -/
structure Position :=
  (x : Fin 8) -- column
  (y : Fin 8) -- row

/-- Checks if two positions are attacking each other on the cylindrical chessboard -/
def isAttacking (p1 p2 : Position) : Prop :=
  p1.x = p2.x ∨ 
  p1.y = p2.y ∨ 
  (p1.x.val - p2.x.val) % 8 = (p1.y.val - p2.y.val) % 8 ∨
  (p1.x.val - p2.x.val) % 8 = (p2.y.val - p1.y.val) % 8

/-- A configuration of 8 queens on the cylindrical chessboard -/
def QueenConfiguration := Fin 8 → Position

/-- Theorem: It's impossible to place 8 queens on a cylindrical chessboard without attacks -/
theorem cylindrical_eight_queens_impossible :
  ∀ (config : QueenConfiguration), 
    ∃ (i j : Fin 8), i ≠ j ∧ isAttacking (config i) (config j) := by
  sorry


end NUMINAMATH_CALUDE_cylindrical_eight_queens_impossible_l3215_321593


namespace NUMINAMATH_CALUDE_sin_cos_shift_l3215_321587

theorem sin_cos_shift (x : ℝ) : 
  Real.sin (x + π/2) + Real.cos (x + π/2) = Real.sin x - Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_shift_l3215_321587


namespace NUMINAMATH_CALUDE_equation_solutions_l3215_321503

theorem equation_solutions :
  (∀ x : ℝ, (x + 1)^2 = 4 ↔ x = 1 ∨ x = -3) ∧
  (∀ x : ℝ, 3*x^3 + 4 = -20 ↔ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3215_321503


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_twenty_eight_l3215_321579

theorem reciprocal_of_negative_twenty_eight :
  (1 : ℚ) / (-28 : ℚ) = -1 / 28 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_twenty_eight_l3215_321579


namespace NUMINAMATH_CALUDE_angle_measure_proof_l3215_321531

theorem angle_measure_proof (x : ℝ) : 
  (90 - x = 3 * x - 7) → x = 24.25 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l3215_321531


namespace NUMINAMATH_CALUDE_vacation_miles_driven_l3215_321585

theorem vacation_miles_driven (days : ℕ) (miles_per_day : ℕ) (h1 : days = 5) (h2 : miles_per_day = 250) :
  days * miles_per_day = 1250 := by
  sorry

end NUMINAMATH_CALUDE_vacation_miles_driven_l3215_321585


namespace NUMINAMATH_CALUDE_four_three_seating_chart_l3215_321567

/-- Represents a seating chart configuration -/
structure SeatingChart where
  columns : ℕ
  rows : ℕ

/-- Interprets a pair of natural numbers as a seating chart -/
def interpret (pair : ℕ × ℕ) : SeatingChart :=
  { columns := pair.1, rows := pair.2 }

/-- States that (4,3) represents 4 columns and 3 rows -/
theorem four_three_seating_chart :
  let chart := interpret (4, 3)
  chart.columns = 4 ∧ chart.rows = 3 := by
  sorry

end NUMINAMATH_CALUDE_four_three_seating_chart_l3215_321567


namespace NUMINAMATH_CALUDE_james_field_goal_value_l3215_321516

/-- Represents the score of a basketball game -/
structure BasketballScore where
  fieldGoals : ℕ
  fieldGoalValue : ℕ
  twoPointers : ℕ
  totalScore : ℕ

/-- Theorem stating that given the conditions of James' game, each field goal is worth 3 points -/
theorem james_field_goal_value (score : BasketballScore) 
  (h1 : score.fieldGoals = 13)
  (h2 : score.twoPointers = 20)
  (h3 : score.totalScore = 79)
  (h4 : score.totalScore = score.fieldGoals * score.fieldGoalValue + score.twoPointers * 2) :
  score.fieldGoalValue = 3 := by
  sorry


end NUMINAMATH_CALUDE_james_field_goal_value_l3215_321516


namespace NUMINAMATH_CALUDE_inverse_function_problem_l3215_321538

theorem inverse_function_problem (f : ℝ → ℝ) (f_inv : ℝ → ℝ) :
  (∀ x, f_inv x = 2^(x + 1)) → f 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_problem_l3215_321538


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_equals_three_l3215_321559

theorem cubic_sum_over_product_equals_three
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c = 0) :
  (a^3 + b^3 + c^3) / (a * b * c) = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_equals_three_l3215_321559


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l3215_321510

def recipe_flour (total_sugar : ℕ) (added_sugar : ℕ) (flour_sugar_diff : ℕ) : ℕ :=
  (total_sugar - added_sugar) + flour_sugar_diff

theorem recipe_flour_amount : recipe_flour 6 4 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l3215_321510


namespace NUMINAMATH_CALUDE_total_wheels_is_25_l3215_321523

/-- Calculates the total number of wheels in Jordan's driveway -/
def total_wheels : ℕ :=
  let num_cars : ℕ := 2
  let wheels_per_car : ℕ := 4
  let num_bikes : ℕ := 2
  let wheels_per_bike : ℕ := 2
  let num_trash_cans : ℕ := 1
  let wheels_per_trash_can : ℕ := 2
  let num_tricycles : ℕ := 1
  let wheels_per_tricycle : ℕ := 3
  let num_roller_skate_pairs : ℕ := 1
  let wheels_per_roller_skate : ℕ := 4
  let wheels_per_roller_skate_pair : ℕ := 2 * wheels_per_roller_skate

  num_cars * wheels_per_car +
  num_bikes * wheels_per_bike +
  num_trash_cans * wheels_per_trash_can +
  num_tricycles * wheels_per_tricycle +
  num_roller_skate_pairs * wheels_per_roller_skate_pair

theorem total_wheels_is_25 : total_wheels = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_25_l3215_321523


namespace NUMINAMATH_CALUDE_equation_2x_minus_y_eq_2_is_linear_l3215_321575

/-- A linear equation in two variables is of the form ax + by + c = 0, where a and b are not both zero -/
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y = a * x + b * y + c

/-- The function representing the equation 2x - y = 2 -/
def f (x y : ℝ) : ℝ := 2 * x - y - 2

theorem equation_2x_minus_y_eq_2_is_linear : is_linear_equation f :=
sorry

end NUMINAMATH_CALUDE_equation_2x_minus_y_eq_2_is_linear_l3215_321575


namespace NUMINAMATH_CALUDE_rainfall_difference_l3215_321519

theorem rainfall_difference (sunday_rain monday_rain tuesday_rain : ℝ) : 
  sunday_rain = 4 ∧ 
  tuesday_rain = 2 * monday_rain ∧ 
  monday_rain > sunday_rain ∧ 
  sunday_rain + monday_rain + tuesday_rain = 25 →
  monday_rain - sunday_rain = 3 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_difference_l3215_321519


namespace NUMINAMATH_CALUDE_dianes_trip_length_l3215_321572

theorem dianes_trip_length :
  ∀ (total_length : ℝ),
  (1/4 : ℝ) * total_length + 24 + (1/3 : ℝ) * total_length = total_length →
  total_length = 57.6 := by
sorry

end NUMINAMATH_CALUDE_dianes_trip_length_l3215_321572


namespace NUMINAMATH_CALUDE_sin_thirty_degrees_l3215_321591

theorem sin_thirty_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirty_degrees_l3215_321591


namespace NUMINAMATH_CALUDE_common_tangents_of_circles_l3215_321537

/-- Circle C1 with equation x² + y² - 2x - 4y - 4 = 0 -/
def C1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y - 4 = 0

/-- Circle C2 with equation x² + y² - 6x - 10y - 2 = 0 -/
def C2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 10*y - 2 = 0

/-- The number of common tangents between C1 and C2 -/
def num_common_tangents : ℕ := 2

theorem common_tangents_of_circles :
  num_common_tangents = 2 :=
sorry

end NUMINAMATH_CALUDE_common_tangents_of_circles_l3215_321537


namespace NUMINAMATH_CALUDE_triangle_area_l3215_321568

/-- The area of a triangle with base 3 meters and height 4 meters is 6 square meters. -/
theorem triangle_area : 
  let base : ℝ := 3
  let height : ℝ := 4
  let area : ℝ := (base * height) / 2
  area = 6 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3215_321568


namespace NUMINAMATH_CALUDE_largest_binomial_equality_l3215_321553

theorem largest_binomial_equality : ∃ n : ℕ, (n ≤ 11 ∧ Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n ∧ ∀ m : ℕ, m ≤ 11 → Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_largest_binomial_equality_l3215_321553


namespace NUMINAMATH_CALUDE_subcommittee_count_l3215_321552

theorem subcommittee_count (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 3) :
  n * (Nat.choose (n - 1) (k - 1)) = 12180 :=
by sorry

end NUMINAMATH_CALUDE_subcommittee_count_l3215_321552


namespace NUMINAMATH_CALUDE_line_equation_equiv_l3215_321581

/-- The line equation in vector form -/
def line_equation (x y : ℝ) : Prop :=
  (3 : ℝ) * (x - 2) + (-4 : ℝ) * (y - 8) = 0

/-- The line equation in slope-intercept form -/
def slope_intercept_form (x y : ℝ) : Prop :=
  y = (3/4) * x + (13/2)

/-- Theorem stating the equivalence of the two forms -/
theorem line_equation_equiv :
  ∀ x y : ℝ, line_equation x y ↔ slope_intercept_form x y :=
sorry

end NUMINAMATH_CALUDE_line_equation_equiv_l3215_321581


namespace NUMINAMATH_CALUDE_enemies_left_undefeated_l3215_321599

theorem enemies_left_undefeated 
  (points_per_enemy : ℕ) 
  (total_enemies : ℕ) 
  (points_earned : ℕ) : ℕ :=
by
  have h1 : points_per_enemy = 5 := by sorry
  have h2 : total_enemies = 8 := by sorry
  have h3 : points_earned = 10 := by sorry
  
  -- Define the number of enemies defeated
  let enemies_defeated := points_earned / points_per_enemy
  
  -- Calculate enemies left undefeated
  let enemies_left := total_enemies - enemies_defeated
  
  exact enemies_left

end NUMINAMATH_CALUDE_enemies_left_undefeated_l3215_321599


namespace NUMINAMATH_CALUDE_jorkins_christmas_spending_l3215_321541

-- Define the type for British currency
structure BritishCurrency where
  pounds : ℕ
  shillings : ℕ

def BritishCurrency.toShillings (bc : BritishCurrency) : ℕ :=
  20 * bc.pounds + bc.shillings

def BritishCurrency.halfValue (bc : BritishCurrency) : ℕ :=
  bc.toShillings / 2

theorem jorkins_christmas_spending (initial : BritishCurrency) 
  (h1 : initial.halfValue = 20 * (initial.shillings / 2) + initial.pounds)
  (h2 : initial.shillings / 2 = initial.pounds)
  (h3 : initial.pounds = initial.shillings / 2) :
  initial = BritishCurrency.mk 19 18 := by
  sorry

#check jorkins_christmas_spending

end NUMINAMATH_CALUDE_jorkins_christmas_spending_l3215_321541


namespace NUMINAMATH_CALUDE_pairball_playing_time_l3215_321595

theorem pairball_playing_time (num_children : ℕ) (total_time : ℕ) : 
  num_children = 6 →
  total_time = 90 →
  ∃ (time_per_child : ℕ), 
    time_per_child * num_children = 2 * total_time ∧
    time_per_child = 30 :=
by sorry

end NUMINAMATH_CALUDE_pairball_playing_time_l3215_321595


namespace NUMINAMATH_CALUDE_parabola_sum_l3215_321573

/-- Represents a parabola of the form y = ax^2 + c -/
structure Parabola where
  a : ℝ
  c : ℝ

/-- The area of the kite formed by the intersections of the parabolas with the axes -/
def kite_area (p1 p2 : Parabola) : ℝ := 12

/-- The parabolas intersect the coordinate axes at exactly four points -/
def intersect_at_four_points (p1 p2 : Parabola) : Prop := sorry

theorem parabola_sum (p1 p2 : Parabola) 
  (h1 : p1.c = -2 ∧ p2.c = 4) 
  (h2 : intersect_at_four_points p1 p2) 
  (h3 : kite_area p1 p2 = 12) : 
  p1.a + p2.a = 1.5 := by sorry

end NUMINAMATH_CALUDE_parabola_sum_l3215_321573


namespace NUMINAMATH_CALUDE_ratio_equation_solution_product_l3215_321532

theorem ratio_equation_solution_product (x : ℝ) :
  (3 * x + 5) / (4 * x + 4) = (5 * x + 4) / (10 * x + 5) →
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (3 * x₁ + 5) / (4 * x₁ + 4) = (5 * x₁ + 4) / (10 * x₁ + 5) ∧
    (3 * x₂ + 5) / (4 * x₂ + 4) = (5 * x₂ + 4) / (10 * x₂ + 5) ∧
    x₁ * x₂ = 9 / 10 :=
by sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_product_l3215_321532


namespace NUMINAMATH_CALUDE_unique_two_digit_reverse_pair_l3215_321570

theorem unique_two_digit_reverse_pair (z : ℕ) (h : z ≥ 3) :
  ∃! (A B : ℕ),
    (A < z^2 ∧ A ≥ z) ∧
    (B < z^2 ∧ B ≥ z) ∧
    (∃ (p q : ℕ), A = p * z + q ∧ B = q * z + p) ∧
    (∀ x : ℝ, (x^2 - A*x + B = 0) → (∃! r : ℝ, x = r)) ∧
    A = (z - 1)^2 ∧
    B = 2*(z - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_reverse_pair_l3215_321570


namespace NUMINAMATH_CALUDE_common_factor_proof_l3215_321508

def p (x : ℝ) := x^2 - 2*x - 3
def q (x : ℝ) := x^2 - 6*x + 9
def common_factor (x : ℝ) := x - 3

theorem common_factor_proof :
  ∀ x : ℝ, (∃ k₁ k₂ : ℝ, p x = common_factor x * k₁ ∧ q x = common_factor x * k₂) :=
sorry

end NUMINAMATH_CALUDE_common_factor_proof_l3215_321508


namespace NUMINAMATH_CALUDE_cat_litter_weight_l3215_321505

/-- Calculates the weight of cat litter in each container given the problem conditions. -/
theorem cat_litter_weight 
  (container_price : ℝ) 
  (litter_box_capacity : ℝ) 
  (total_cost : ℝ) 
  (total_days : ℝ) 
  (h1 : container_price = 21)
  (h2 : litter_box_capacity = 15)
  (h3 : total_cost = 210)
  (h4 : total_days = 210) :
  (total_cost * litter_box_capacity) / (container_price * total_days / 7) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cat_litter_weight_l3215_321505


namespace NUMINAMATH_CALUDE_speeding_ticket_theorem_l3215_321514

/-- Represents the percentage of motorists who receive speeding tickets -/
def ticket_percentage : ℝ := 10

/-- Represents the percentage of motorists who exceed the speed limit -/
def exceed_limit_percentage : ℝ := 16.666666666666664

/-- Theorem stating that 40% of motorists who exceed the speed limit do not receive speeding tickets -/
theorem speeding_ticket_theorem :
  (exceed_limit_percentage - ticket_percentage) / exceed_limit_percentage * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_speeding_ticket_theorem_l3215_321514


namespace NUMINAMATH_CALUDE_sum_of_products_l3215_321521

theorem sum_of_products : 5 * 12 + 7 * 15 + 13 * 4 + 6 * 9 = 271 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_l3215_321521


namespace NUMINAMATH_CALUDE_correct_addition_with_digit_change_l3215_321528

theorem correct_addition_with_digit_change :
  ∃ (d e : ℕ), d ≠ e ∧ d < 10 ∧ e < 10 ∧
  ((853697 + 930541 = 1383238 ∧ d = 8 ∧ e = 4) ∨
   (453697 + 930541 = 1383238 ∧ d = 8 ∧ e = 4)) ∧
  d + e = 12 := by
sorry

end NUMINAMATH_CALUDE_correct_addition_with_digit_change_l3215_321528


namespace NUMINAMATH_CALUDE_lagrange_interpolation_identity_l3215_321557

theorem lagrange_interpolation_identity 
  (a b c x : ℝ) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  c^2 * ((x-a)*(x-b))/((c-a)*(c-b)) + 
  b^2 * ((x-a)*(x-c))/((b-a)*(b-c)) + 
  a^2 * ((x-b)*(x-c))/((a-b)*(a-c)) = x^2 := by
  sorry

end NUMINAMATH_CALUDE_lagrange_interpolation_identity_l3215_321557


namespace NUMINAMATH_CALUDE_fraction_simplification_l3215_321571

theorem fraction_simplification (x y : ℚ) (hx : x = 5) (hy : y = 8) :
  (1 / x - 1 / y) / (1 / x) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3215_321571


namespace NUMINAMATH_CALUDE_student_pairs_l3215_321580

theorem student_pairs (n : ℕ) (h : n = 12) : Nat.choose n 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_student_pairs_l3215_321580


namespace NUMINAMATH_CALUDE_park_visit_cost_family_park_visit_cost_l3215_321512

/-- Calculates the total cost for a family visiting a park with given conditions -/
theorem park_visit_cost (entrance_fee : ℝ) (kid_attraction_fee : ℝ) (adult_attraction_fee : ℝ)
  (entrance_discount_rate : ℝ) (senior_attraction_discount_rate : ℝ)
  (num_children num_parents num_grandparents : ℕ) : ℝ :=
  let total_people := num_children + num_parents + num_grandparents
  let entrance_discount := if total_people ≥ 6 then entrance_discount_rate else 0
  let entrance_cost := (1 - entrance_discount) * entrance_fee * total_people
  let attraction_cost_children := kid_attraction_fee * num_children
  let attraction_cost_parents := adult_attraction_fee * num_parents
  let attraction_cost_grandparents := 
    adult_attraction_fee * (1 - senior_attraction_discount_rate) * num_grandparents
  let total_cost := entrance_cost + attraction_cost_children + 
                    attraction_cost_parents + attraction_cost_grandparents
  total_cost

/-- The total cost for the family visit is $49.50 -/
theorem family_park_visit_cost : 
  park_visit_cost 5 2 4 0.1 0.5 4 2 1 = 49.5 := by
  sorry

end NUMINAMATH_CALUDE_park_visit_cost_family_park_visit_cost_l3215_321512


namespace NUMINAMATH_CALUDE_stock_change_is_negative_4_375_percent_l3215_321550

/-- The overall percent change in a stock value after three days of fluctuations -/
def stock_percent_change : ℝ := by
  -- Define the daily changes
  let day1_change : ℝ := 0.85  -- 15% decrease
  let day2_change : ℝ := 1.25  -- 25% increase
  let day3_change : ℝ := 0.90  -- 10% decrease

  -- Calculate the overall change
  let overall_change : ℝ := day1_change * day2_change * day3_change

  -- Calculate the percent change
  exact (overall_change - 1) * 100

/-- Theorem stating that the overall percent change in the stock is -4.375% -/
theorem stock_change_is_negative_4_375_percent : 
  stock_percent_change = -4.375 := by
  sorry

end NUMINAMATH_CALUDE_stock_change_is_negative_4_375_percent_l3215_321550


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_137_l3215_321569

theorem first_nonzero_digit_of_one_over_137 :
  ∃ (n : ℕ) (k : ℕ), 
    (1000 : ℚ) / 137 = 7 + (n : ℚ) / (10 ^ k) ∧ 
    0 < n ∧ 
    n < 10 ^ k ∧ 
    n % 10 = 7 :=
by sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_137_l3215_321569


namespace NUMINAMATH_CALUDE_three_intersection_points_l3215_321551

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y = 6
def line3 (x y : ℝ) : Prop := 6 * x - 9 * y = 8

-- Define an intersection point
def is_intersection (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line2 x y ∧ line3 x y)

-- Theorem statement
theorem three_intersection_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    is_intersection p3.1 p3.2 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    ∀ (x y : ℝ), is_intersection x y → (x, y) = p1 ∨ (x, y) = p2 ∨ (x, y) = p3 :=
by sorry

end NUMINAMATH_CALUDE_three_intersection_points_l3215_321551


namespace NUMINAMATH_CALUDE_wiper_line_to_surface_l3215_321549

/-- A car wiper blade modeled as a line -/
structure WiperBlade :=
  (length : ℝ)

/-- A windshield modeled as a surface -/
structure Windshield :=
  (width : ℝ)
  (height : ℝ)

/-- The area swept by a wiper blade on a windshield -/
def swept_area (blade : WiperBlade) (shield : Windshield) : ℝ :=
  blade.length * shield.width

/-- Theorem stating that a car wiper on a windshield represents a line moving into a surface -/
theorem wiper_line_to_surface (blade : WiperBlade) (shield : Windshield) :
  ∃ (area : ℝ), area = swept_area blade shield ∧ area > 0 :=
sorry

end NUMINAMATH_CALUDE_wiper_line_to_surface_l3215_321549


namespace NUMINAMATH_CALUDE_gaeun_taller_than_nana_l3215_321544

/-- Nana's height in meters -/
def nana_height_m : ℝ := 1.618

/-- Gaeun's height in centimeters -/
def gaeun_height_cm : ℝ := 162.3

/-- Conversion factor from meters to centimeters -/
def m_to_cm : ℝ := 100

theorem gaeun_taller_than_nana : 
  gaeun_height_cm > nana_height_m * m_to_cm := by sorry

end NUMINAMATH_CALUDE_gaeun_taller_than_nana_l3215_321544


namespace NUMINAMATH_CALUDE_cream_strawberry_prices_l3215_321539

/-- Represents the price of a box of cream strawberries in yuan -/
@[ext] structure StrawberryPrice where
  price : ℚ
  price_positive : price > 0

/-- The problem of finding cream strawberry prices -/
theorem cream_strawberry_prices 
  (price_A price_B : StrawberryPrice)
  (price_difference : price_A.price = price_B.price + 10)
  (quantity_equality : 800 / price_A.price = 600 / price_B.price) :
  price_A.price = 40 ∧ price_B.price = 30 := by
  sorry

end NUMINAMATH_CALUDE_cream_strawberry_prices_l3215_321539


namespace NUMINAMATH_CALUDE_tangerine_boxes_count_l3215_321554

/-- Given information about apples and tangerines, prove the number of tangerine boxes --/
theorem tangerine_boxes_count
  (apple_boxes : ℕ)
  (apples_per_box : ℕ)
  (tangerines_per_box : ℕ)
  (total_fruits : ℕ)
  (h1 : apple_boxes = 19)
  (h2 : apples_per_box = 46)
  (h3 : tangerines_per_box = 170)
  (h4 : total_fruits = 1894)
  : ∃ (tangerine_boxes : ℕ), tangerine_boxes = 6 ∧ 
    apple_boxes * apples_per_box + tangerine_boxes * tangerines_per_box = total_fruits :=
by
  sorry


end NUMINAMATH_CALUDE_tangerine_boxes_count_l3215_321554


namespace NUMINAMATH_CALUDE_constant_term_is_integer_coefficients_not_necessarily_integer_l3215_321527

/-- A real quadratic polynomial that takes integer values for all integer inputs -/
structure IntegerValuedQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  integer_valued : ∀ (x : ℤ), ∃ (y : ℤ), a * x^2 + b * x + c = y

theorem constant_term_is_integer (p : IntegerValuedQuadratic) : ∃ (n : ℤ), p.c = n := by
  sorry

theorem coefficients_not_necessarily_integer : 
  ∃ (p : IntegerValuedQuadratic), ¬(∃ (m n : ℤ), p.a = m ∧ p.b = n) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_is_integer_coefficients_not_necessarily_integer_l3215_321527


namespace NUMINAMATH_CALUDE_expression_equals_one_l3215_321500

theorem expression_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h : a + b - c = 0) :
  (a^2 * b^2) / ((a^2 + b*c) * (b^2 + a*c)) +
  (a^2 * c^2) / ((a^2 + b*c) * (c^2 + a*b)) +
  (b^2 * c^2) / ((b^2 + a*c) * (c^2 + a*b)) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3215_321500


namespace NUMINAMATH_CALUDE_cone_volume_l3215_321578

/-- The volume of a cone with base radius 1 and slant height 2 is (√3/3)π -/
theorem cone_volume (r h l : ℝ) : 
  r = 1 → l = 2 → h^2 + r^2 = l^2 → (1/3 * π * r^2 * h) = (Real.sqrt 3 / 3) * π :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l3215_321578


namespace NUMINAMATH_CALUDE_chessboard_coverage_l3215_321518

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a domino -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Checks if a chessboard can be covered by dominoes -/
def can_cover (board : Chessboard) (domino : Domino) : Prop :=
  sorry

/-- Checks if a chessboard with one corner removed can be covered by dominoes -/
def can_cover_one_corner_removed (board : Chessboard) (domino : Domino) : Prop :=
  sorry

/-- Checks if a chessboard with two opposite corners removed can be covered by dominoes -/
def can_cover_two_corners_removed (board : Chessboard) (domino : Domino) : Prop :=
  sorry

theorem chessboard_coverage (board : Chessboard) (domino : Domino) :
  board.rows = 8 ∧ board.cols = 8 ∧ domino.length = 2 ∧ domino.width = 1 →
  (can_cover board domino) ∧
  ¬(can_cover_one_corner_removed board domino) ∧
  ¬(can_cover_two_corners_removed board domino) :=
sorry

end NUMINAMATH_CALUDE_chessboard_coverage_l3215_321518


namespace NUMINAMATH_CALUDE_second_car_speed_theorem_l3215_321517

/-- Two cars traveling on perpendicular roads towards an intersection -/
structure TwoCars where
  s₁ : ℝ  -- Initial distance of first car from intersection
  s₂ : ℝ  -- Initial distance of second car from intersection
  v₁ : ℝ  -- Speed of first car
  s  : ℝ  -- Distance between cars when first car reaches intersection

/-- The speed of the second car in the TwoCars scenario -/
def second_car_speed (cars : TwoCars) : Set ℝ :=
  {v₂ | v₂ = 12 ∨ v₂ = 16}

/-- Theorem stating the possible speeds of the second car -/
theorem second_car_speed_theorem (cars : TwoCars) 
    (h₁ : cars.s₁ = 500)
    (h₂ : cars.s₂ = 700)
    (h₃ : cars.v₁ = 10)  -- 36 km/h converted to m/s
    (h₄ : cars.s = 100) :
  second_car_speed cars = {12, 16} := by
  sorry

end NUMINAMATH_CALUDE_second_car_speed_theorem_l3215_321517


namespace NUMINAMATH_CALUDE_equal_expressions_l3215_321525

theorem equal_expressions (x : ℝ) (hx : x > 0) :
  (x^(x+1) + x^(x+1) = 2*x^(x+1)) ∧
  (x^(x+1) + x^(x+1) ≠ x^(2*x+2)) ∧
  (x^(x+1) + x^(x+1) ≠ (x+1)^(x+1)) ∧
  (x^(x+1) + x^(x+1) ≠ (2*x)^(x+1)) :=
by sorry

end NUMINAMATH_CALUDE_equal_expressions_l3215_321525


namespace NUMINAMATH_CALUDE_subset_intersection_condition_l3215_321598

theorem subset_intersection_condition (M N : Set α) (h_nonempty : M.Nonempty) (h_subset : M ⊆ N) :
  (∀ a, a ∈ M ∩ N → (a ∈ M ∨ a ∈ N)) ∧
  ¬(∀ a, (a ∈ M ∨ a ∈ N) → a ∈ M ∩ N) :=
by sorry

end NUMINAMATH_CALUDE_subset_intersection_condition_l3215_321598


namespace NUMINAMATH_CALUDE_f_is_convex_f_range_a_l3215_321534

/-- Definition of a convex function -/
def IsConvex (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f ((x₁ + x₂) / 2) ≤ (f x₁ + f x₂) / 2

/-- The quadratic function f(x) = ax^2 + x -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

/-- Theorem: f is convex when a > 0 -/
theorem f_is_convex (a : ℝ) (ha : a > 0) : IsConvex (f a) := by sorry

/-- Theorem: Range of a when |f(x)| ≤ 1 for x ∈ [0,1] -/
theorem f_range_a (a : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |f a x| ≤ 1) ↔ a ∈ Set.Icc (-2) 0 := by sorry

end NUMINAMATH_CALUDE_f_is_convex_f_range_a_l3215_321534


namespace NUMINAMATH_CALUDE_triple_base_exponent_l3215_321535

theorem triple_base_exponent (a b x : ℝ) (h1 : b ≠ 0) : 
  (3 * a) ^ (3 * b) = a ^ b * x ^ b → x = 27 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triple_base_exponent_l3215_321535


namespace NUMINAMATH_CALUDE_pencils_left_l3215_321515

/-- Given two boxes of pencils with fourteen pencils each, prove that after giving away six pencils, the number of pencils left is 22. -/
theorem pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (pencils_given_away : ℕ) : 
  boxes = 2 → pencils_per_box = 14 → pencils_given_away = 6 →
  boxes * pencils_per_box - pencils_given_away = 22 := by
sorry

end NUMINAMATH_CALUDE_pencils_left_l3215_321515


namespace NUMINAMATH_CALUDE_alice_paid_48_percent_of_srp_l3215_321597

-- Define the suggested retail price (SRP)
def suggested_retail_price : ℝ := 100

-- Define the marked price (MP) as 80% of SRP
def marked_price : ℝ := 0.8 * suggested_retail_price

-- Define Alice's purchase price as 60% of MP
def alice_price : ℝ := 0.6 * marked_price

-- Theorem to prove
theorem alice_paid_48_percent_of_srp :
  alice_price / suggested_retail_price = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_alice_paid_48_percent_of_srp_l3215_321597


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_fraction_l3215_321596

theorem first_nonzero_digit_of_fraction (n : ℕ) (h : n = 1029) : 
  ∃ (k : ℕ) (d : ℕ), 
    0 < d ∧ d < 10 ∧
    (↑k : ℚ) < (1 : ℚ) / n ∧
    (1 : ℚ) / n < ((↑k + 1) : ℚ) / 10 ∧
    d = 9 :=
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_fraction_l3215_321596


namespace NUMINAMATH_CALUDE_paper_length_is_correct_l3215_321555

/-- The length of a rectangular sheet of paper satisfying given conditions -/
def paper_length : ℚ :=
  let width : ℚ := 9
  let second_sheet_length : ℚ := 11
  let second_sheet_width : ℚ := 9/2
  let area_difference : ℚ := 100
  (2 * second_sheet_length * second_sheet_width + area_difference) / (2 * width)

theorem paper_length_is_correct :
  let width : ℚ := 9
  let second_sheet_length : ℚ := 11
  let second_sheet_width : ℚ := 9/2
  let area_difference : ℚ := 100
  2 * paper_length * width = 2 * second_sheet_length * second_sheet_width + area_difference :=
by
  sorry

#eval paper_length

end NUMINAMATH_CALUDE_paper_length_is_correct_l3215_321555


namespace NUMINAMATH_CALUDE_coin_flip_probability_is_two_elevenths_l3215_321564

/-- The probability of getting 4 consecutive heads before 3 consecutive tails
    when repeatedly flipping a fair coin -/
def coin_flip_probability : ℚ :=
  2/11

/-- Theorem stating that the probability of getting 4 consecutive heads
    before 3 consecutive tails when repeatedly flipping a fair coin is 2/11 -/
theorem coin_flip_probability_is_two_elevenths :
  coin_flip_probability = 2/11 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_is_two_elevenths_l3215_321564


namespace NUMINAMATH_CALUDE_root_sum_theorem_l3215_321543

theorem root_sum_theorem (a b c d r : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (r - a) * (r - b) * (r - c) * (r - d) = 4 →
  4 * r = a + b + c + d := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l3215_321543
