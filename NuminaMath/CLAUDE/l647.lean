import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l647_64779

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1) :
  (1 / x) + (4 / y) + (9 / z) ≥ 36 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l647_64779


namespace NUMINAMATH_CALUDE_min_sum_of_product_72_l647_64738

theorem min_sum_of_product_72 (a b : ℤ) (h : a * b = 72) :
  ∀ x y : ℤ, x * y = 72 → a + b ≤ x + y :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_72_l647_64738


namespace NUMINAMATH_CALUDE_shells_found_l647_64754

def initial_shells : ℕ := 68
def final_shells : ℕ := 89

theorem shells_found (initial : ℕ) (final : ℕ) (h1 : initial = initial_shells) (h2 : final = final_shells) :
  final - initial = 21 := by
  sorry

end NUMINAMATH_CALUDE_shells_found_l647_64754


namespace NUMINAMATH_CALUDE_rotated_point_x_coordinate_l647_64769

theorem rotated_point_x_coordinate 
  (P : ℝ × ℝ) 
  (h_unit_circle : P.1^2 + P.2^2 = 1) 
  (h_P : P = (4/5, -3/5)) : 
  let Q := (
    P.1 * Real.cos (π/3) - P.2 * Real.sin (π/3),
    P.1 * Real.sin (π/3) + P.2 * Real.cos (π/3)
  )
  Q.1 = (4 + 3 * Real.sqrt 3) / 10 := by
sorry

end NUMINAMATH_CALUDE_rotated_point_x_coordinate_l647_64769


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l647_64784

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x - 6 * x + 18 = 0) ∧ 
  (∀ x y : ℝ, 3 * x^2 - m * x - 6 * x + 18 = 0 ∧ 3 * y^2 - m * y - 6 * y + 18 = 0 → x = y) ∧
  ((m + 6) / 3 < -2) →
  m = -6 - 6 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l647_64784


namespace NUMINAMATH_CALUDE_patio_layout_change_l647_64724

/-- Represents a rectangular patio layout --/
structure PatioLayout where
  rows : ℕ
  columns : ℕ
  total_tiles : ℕ
  is_rectangular : rows * columns = total_tiles

/-- The change in patio layout --/
def change_layout (initial : PatioLayout) (row_increase : ℕ) : PatioLayout :=
  { rows := initial.rows + row_increase,
    columns := initial.total_tiles / (initial.rows + row_increase),
    total_tiles := initial.total_tiles,
    is_rectangular := sorry }

theorem patio_layout_change (initial : PatioLayout) 
  (h1 : initial.total_tiles = 30)
  (h2 : initial.rows = 5) :
  let final := change_layout initial 4
  initial.columns - final.columns = 3 := by sorry

end NUMINAMATH_CALUDE_patio_layout_change_l647_64724


namespace NUMINAMATH_CALUDE_accidental_division_correction_l647_64790

theorem accidental_division_correction (x : ℝ) : 
  x / 15 = 6 → x * 15 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_accidental_division_correction_l647_64790


namespace NUMINAMATH_CALUDE_sabrina_video_votes_l647_64782

theorem sabrina_video_votes (total_votes : ℕ) (upvotes downvotes : ℕ) (score : ℤ) : 
  upvotes = (3 * total_votes) / 4 →
  downvotes = total_votes / 4 →
  score = 150 →
  (upvotes : ℤ) - (downvotes : ℤ) = score →
  total_votes = 300 := by
sorry

end NUMINAMATH_CALUDE_sabrina_video_votes_l647_64782


namespace NUMINAMATH_CALUDE_custom_product_of_A_and_B_l647_64758

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x ≥ 0}

-- Define the custom cartesian product operation
def custom_product (X Y : Set ℝ) : Set ℝ := {x : ℝ | x ∈ X ∪ Y ∧ x ∉ X ∩ Y}

-- Theorem statement
theorem custom_product_of_A_and_B :
  custom_product A B = {x : ℝ | x > 2} := by
  sorry

end NUMINAMATH_CALUDE_custom_product_of_A_and_B_l647_64758


namespace NUMINAMATH_CALUDE_scientific_notation_million_l647_64711

theorem scientific_notation_million (x : ℝ) (h : x = 1464.3) :
  x * (10 : ℝ)^6 = 1.4643 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_million_l647_64711


namespace NUMINAMATH_CALUDE_box_volume_l647_64726

/-- A rectangular box with specific proportions -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  front_half_top : length * height = 0.5 * (length * width)
  top_1_5_side : length * width = 1.5 * (width * height)
  side_area : width * height = 72

/-- The volume of a box is equal to 648 cubic units -/
theorem box_volume (b : Box) : b.length * b.width * b.height = 648 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l647_64726


namespace NUMINAMATH_CALUDE_irreducible_fraction_to_mersenne_form_l647_64721

theorem irreducible_fraction_to_mersenne_form 
  (p q : ℕ+) 
  (h_q_odd : q.val % 2 = 1) : 
  ∃ (n k : ℕ+), (p : ℚ) / q = (n : ℚ) / (2^k.val - 1) :=
sorry

end NUMINAMATH_CALUDE_irreducible_fraction_to_mersenne_form_l647_64721


namespace NUMINAMATH_CALUDE_journalist_selection_theorem_l647_64765

-- Define the number of domestic and foreign journalists
def domestic_journalists : ℕ := 5
def foreign_journalists : ℕ := 4

-- Define the total number of journalists to be selected
def selected_journalists : ℕ := 3

-- Function to calculate the number of ways to select and arrange journalists
def select_and_arrange_journalists : ℕ := sorry

-- Theorem stating the correct number of ways
theorem journalist_selection_theorem : 
  select_and_arrange_journalists = 260 := by sorry

end NUMINAMATH_CALUDE_journalist_selection_theorem_l647_64765


namespace NUMINAMATH_CALUDE_chess_and_go_problem_l647_64713

theorem chess_and_go_problem (chess_price go_price : ℝ) 
  (h1 : 6 * chess_price + 5 * go_price = 190)
  (h2 : 8 * chess_price + 10 * go_price = 320)
  (budget : ℝ) (total_sets : ℕ)
  (h3 : budget ≤ 1800)
  (h4 : total_sets = 100) :
  chess_price = 15 ∧ 
  go_price = 20 ∧ 
  ∃ (min_chess : ℕ), min_chess ≥ 40 ∧ 
    chess_price * min_chess + go_price * (total_sets - min_chess) ≤ budget :=
by sorry

end NUMINAMATH_CALUDE_chess_and_go_problem_l647_64713


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_nine_implies_product_l647_64703

theorem sqrt_sum_equals_nine_implies_product (x : ℝ) :
  (Real.sqrt (7 + x) + Real.sqrt (28 - x) = 9) →
  ((7 + x) * (28 - x) = 529) := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_nine_implies_product_l647_64703


namespace NUMINAMATH_CALUDE_nineteen_power_nineteen_not_sum_of_cube_and_fourth_power_l647_64701

theorem nineteen_power_nineteen_not_sum_of_cube_and_fourth_power :
  ¬ ∃ (x y : ℤ), 19^19 = x^3 + y^4 := by
  sorry

end NUMINAMATH_CALUDE_nineteen_power_nineteen_not_sum_of_cube_and_fourth_power_l647_64701


namespace NUMINAMATH_CALUDE_total_packages_l647_64737

theorem total_packages (num_trucks : ℕ) (packages_per_truck : ℕ) 
  (h1 : num_trucks = 7) 
  (h2 : packages_per_truck = 70) : 
  num_trucks * packages_per_truck = 490 := by
  sorry

end NUMINAMATH_CALUDE_total_packages_l647_64737


namespace NUMINAMATH_CALUDE_volleyball_betting_strategy_exists_l647_64786

theorem volleyball_betting_strategy_exists : ∃ (x₁ x₂ x₃ x₄ : ℝ),
  x₁ + x₂ + x₃ + x₄ = 1 ∧
  x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧
  6 * x₁ ≥ 1 ∧ 2 * x₂ ≥ 1 ∧ 6 * x₃ ≥ 1 ∧ 7 * x₄ ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_betting_strategy_exists_l647_64786


namespace NUMINAMATH_CALUDE_quadrilateral_interior_angles_sum_l647_64723

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A quadrilateral is a polygon with 4 sides -/
def is_quadrilateral (n : ℕ) : Prop := n = 4

theorem quadrilateral_interior_angles_sum :
  ∀ n : ℕ, is_quadrilateral n → sum_interior_angles n = 360 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_interior_angles_sum_l647_64723


namespace NUMINAMATH_CALUDE_percentage_of_50_to_125_l647_64709

theorem percentage_of_50_to_125 : 
  (50 : ℝ) / 125 * 100 = 40 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_50_to_125_l647_64709


namespace NUMINAMATH_CALUDE_warehouse_problem_l647_64712

/-- Represents the time (in hours) it takes a team to move all goods in a warehouse -/
structure TeamSpeed :=
  (hours : ℝ)
  (positive : hours > 0)

/-- Represents the division of Team C's time between helping Team A and Team B -/
structure TeamCHelp :=
  (helpA : ℝ)
  (helpB : ℝ)
  (positive_helpA : helpA > 0)
  (positive_helpB : helpB > 0)

/-- The main theorem stating the solution to the warehouse problem -/
theorem warehouse_problem 
  (speedA : TeamSpeed) 
  (speedB : TeamSpeed) 
  (speedC : TeamSpeed)
  (h_speedA : speedA.hours = 6)
  (h_speedB : speedB.hours = 7)
  (h_speedC : speedC.hours = 14) :
  ∃ (help : TeamCHelp),
    help.helpA = 7/4 ∧ 
    help.helpB = 7/2 ∧
    help.helpA + help.helpB = speedA.hours * speedB.hours / (speedA.hours + speedB.hours) ∧
    1 / speedA.hours + 1 / speedC.hours * help.helpA = 1 ∧
    1 / speedB.hours + 1 / speedC.hours * help.helpB = 1 :=
sorry

end NUMINAMATH_CALUDE_warehouse_problem_l647_64712


namespace NUMINAMATH_CALUDE_toy_store_fraction_l647_64748

-- Define John's weekly allowance
def weekly_allowance : ℚ := 4.80

-- Define the fraction spent at the arcade
def arcade_fraction : ℚ := 3/5

-- Define the amount spent at the candy store
def candy_store_spending : ℚ := 1.28

-- Theorem statement
theorem toy_store_fraction :
  let remaining_after_arcade := weekly_allowance * (1 - arcade_fraction)
  let toy_store_spending := remaining_after_arcade - candy_store_spending
  toy_store_spending / remaining_after_arcade = 1/3 := by
sorry

end NUMINAMATH_CALUDE_toy_store_fraction_l647_64748


namespace NUMINAMATH_CALUDE_firstDigitOfPowerOfTwoNotPeriodic_l647_64752

-- Define the sequence of first digits of powers of 2
def firstDigitOfPowerOfTwo (n : ℕ) : ℕ :=
  (2^n : ℕ).repr.front.toNat

-- Theorem statement
theorem firstDigitOfPowerOfTwoNotPeriodic :
  ¬ ∃ (d : ℕ), d > 0 ∧ ∀ (n : ℕ), firstDigitOfPowerOfTwo (n + d) = firstDigitOfPowerOfTwo n :=
sorry

end NUMINAMATH_CALUDE_firstDigitOfPowerOfTwoNotPeriodic_l647_64752


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l647_64785

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y^2 = 4) :
  x + 2*y ≥ 3 * Real.rpow 4 (1/3) := by
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l647_64785


namespace NUMINAMATH_CALUDE_only_traffic_light_is_random_l647_64747

/-- Represents a phenomenon that can be observed --/
inductive Phenomenon
  | WaterBoiling : Phenomenon
  | TrafficLight : Phenomenon
  | RectangleArea : Phenomenon
  | LinearEquation : Phenomenon

/-- Determines if a phenomenon is random --/
def isRandom (p : Phenomenon) : Prop :=
  match p with
  | Phenomenon.TrafficLight => True
  | _ => False

/-- Theorem stating that only the traffic light phenomenon is random --/
theorem only_traffic_light_is_random :
  ∀ (p : Phenomenon), isRandom p ↔ p = Phenomenon.TrafficLight := by
  sorry


end NUMINAMATH_CALUDE_only_traffic_light_is_random_l647_64747


namespace NUMINAMATH_CALUDE_pairings_equal_25_l647_64729

/-- The number of bowls and glasses -/
def n : ℕ := 5

/-- The total number of possible pairings of bowls and glasses -/
def total_pairings : ℕ := n * n

/-- Theorem stating that the total number of pairings is 25 -/
theorem pairings_equal_25 : total_pairings = 25 := by
  sorry

end NUMINAMATH_CALUDE_pairings_equal_25_l647_64729


namespace NUMINAMATH_CALUDE_point_coordinate_sum_l647_64776

/-- Given points A and B, where A is at (0, 0) and B is on the line y = 3,
    if the slope of segment AB is 3/4, then the sum of the x- and y-coordinates of B is 7. -/
theorem point_coordinate_sum (x : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 3)
  (3 - 0) / (x - 0) = 3 / 4 → x + 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinate_sum_l647_64776


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l647_64795

/-- The center of a circle tangent to two parallel lines and lying on a third line -/
theorem circle_center_coordinates (x y : ℚ) : 
  (∃ (r : ℚ), r > 0 ∧ 
    (∀ (x' y' : ℚ), (x' - x)^2 + (y' - y)^2 = r^2 → 
      (3*x' + 4*y' = 24 ∨ 3*x' + 4*y' = -16))) → 
  x - 3*y = 0 → 
  (x, y) = (12/13, 4/13) := by
sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l647_64795


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l647_64717

theorem fraction_sum_equality : 
  let a : ℕ := 1
  let b : ℕ := 6
  let c : ℕ := 7
  let d : ℕ := 3
  let e : ℕ := 5
  let f : ℕ := 2
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
   d ≠ e ∧ d ≠ f ∧
   e ≠ f) →
  (Nat.gcd a b = 1 ∧ Nat.gcd c d = 1 ∧ Nat.gcd e f = 1) →
  (a : ℚ) / b + (c : ℚ) / d = (e : ℚ) / f :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l647_64717


namespace NUMINAMATH_CALUDE_problem_solution_l647_64751

def f (x : ℝ) : ℝ := |x| - |2*x - 1|

def M : Set ℝ := {x | f x > -1}

theorem problem_solution :
  (M = {x : ℝ | 0 < x ∧ x < 2}) ∧
  (∀ a ∈ M,
    (0 < a ∧ a < 1 → a^2 - a + 1 < 1/a) ∧
    (a = 1 → a^2 - a + 1 = 1/a) ∧
    (1 < a ∧ a < 2 → a^2 - a + 1 > 1/a)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l647_64751


namespace NUMINAMATH_CALUDE_shaded_area_is_four_point_five_l647_64716

/-- The area of a shape composed of a large isosceles right triangle and a crescent (lune) -/
theorem shaded_area_is_four_point_five 
  (large_triangle_leg : ℝ) 
  (semicircle_diameter : ℝ) 
  (π : ℝ) 
  (h1 : large_triangle_leg = 2)
  (h2 : semicircle_diameter = 2)
  (h3 : π = 3) : 
  (1/2 * large_triangle_leg * large_triangle_leg) + 
  ((1/2 * π * (semicircle_diameter/2)^2) - (1/2 * (semicircle_diameter/2) * (semicircle_diameter/2))) = 4.5 := by
  sorry

#check shaded_area_is_four_point_five

end NUMINAMATH_CALUDE_shaded_area_is_four_point_five_l647_64716


namespace NUMINAMATH_CALUDE_young_inequality_l647_64783

theorem young_inequality (a b p q : ℝ) 
  (ha : a > 0) (hb : b > 0) (hp : p > 1) (hq : q > 1) (hpq : 1/p + 1/q = 1) :
  a^(1/p) * b^(1/q) ≤ a/p + b/q := by
  sorry

end NUMINAMATH_CALUDE_young_inequality_l647_64783


namespace NUMINAMATH_CALUDE_fraction_multiplication_l647_64727

theorem fraction_multiplication : (1 : ℚ) / 3 * (3 : ℚ) / 4 * (4 : ℚ) / 5 = (1 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l647_64727


namespace NUMINAMATH_CALUDE_square_intersection_dot_product_l647_64732

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 3) ∧ B = (3, 3) ∧ C = (3, 0) ∧ D = (0, 0)

-- Define point E as the midpoint of DC
def Midpoint (E D C : ℝ × ℝ) : Prop :=
  E = ((D.1 + C.1) / 2, (D.2 + C.2) / 2)

-- Define the intersection point F
def Intersection (F : ℝ × ℝ) (A E B D : ℝ × ℝ) : Prop :=
  (F.2 = -2 * F.1 + 3) ∧ (F.2 = F.1)

-- Define the dot product of two 2D vectors
def DotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Main theorem
theorem square_intersection_dot_product 
  (A B C D E F : ℝ × ℝ) : 
  Square A B C D → 
  Midpoint E D C → 
  Intersection F A E B D → 
  DotProduct (F.1 - D.1, F.2 - D.2) (E.1 - D.1, E.2 - D.2) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_square_intersection_dot_product_l647_64732


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l647_64705

/-- The breadth of a rectangular plot given its area and length-breadth relationship -/
theorem rectangular_plot_breadth (area : ℝ) (length_ratio : ℝ) : 
  area = 360 → length_ratio = 0.75 → ∃ breadth : ℝ, 
    area = (length_ratio * breadth) * breadth ∧ breadth = 4 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l647_64705


namespace NUMINAMATH_CALUDE_junyoung_remaining_pencils_l647_64743

/-- Calculates the number of remaining pencils after giving some away -/
def remaining_pencils (initial_dozens : ℕ) (given_dozens : ℕ) (given_individual : ℕ) : ℕ :=
  initial_dozens * 12 - (given_dozens * 12 + given_individual)

/-- Theorem stating that given the initial conditions, 75 pencils remain -/
theorem junyoung_remaining_pencils :
  remaining_pencils 11 4 9 = 75 := by
  sorry

end NUMINAMATH_CALUDE_junyoung_remaining_pencils_l647_64743


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l647_64736

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l647_64736


namespace NUMINAMATH_CALUDE_bridge_length_is_80_l647_64796

/-- The length of a bridge given train parameters and crossing time -/
def bridge_length (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) : ℝ :=
  train_speed * crossing_time - train_length

/-- Theorem: The bridge length is 80 meters given the specified conditions -/
theorem bridge_length_is_80 :
  bridge_length 280 18 20 = 80 := by
  sorry

#eval bridge_length 280 18 20

end NUMINAMATH_CALUDE_bridge_length_is_80_l647_64796


namespace NUMINAMATH_CALUDE_erased_value_determinable_l647_64756

-- Define the type for our circle system
structure CircleSystem where
  -- The values in each circle (we'll use Option to represent the erased circle)
  circle_values : Fin 6 → Option ℝ
  -- The values on each segment
  segment_values : Fin 6 → ℝ

-- Define the property that circle values are sums of incoming segment values
def valid_circle_system (cs : CircleSystem) : Prop :=
  ∀ i : Fin 6, 
    cs.circle_values i = some (cs.segment_values i + cs.segment_values ((i + 5) % 6))

-- Define the property that exactly one circle value is erased (None)
def one_erased (cs : CircleSystem) : Prop :=
  ∃! i : Fin 6, cs.circle_values i = none

-- Theorem stating that the erased value can be determined
theorem erased_value_determinable (cs : CircleSystem) 
  (h1 : valid_circle_system cs) (h2 : one_erased cs) : 
  ∃ (x : ℝ), ∀ (cs' : CircleSystem), 
    valid_circle_system cs' → 
    (∀ i : Fin 6, cs.circle_values i ≠ none → cs'.circle_values i = cs.circle_values i) →
    (∀ i : Fin 6, cs'.segment_values i = cs.segment_values i) →
    (∃ i : Fin 6, cs'.circle_values i = some x ∧ cs.circle_values i = none) :=
sorry

end NUMINAMATH_CALUDE_erased_value_determinable_l647_64756


namespace NUMINAMATH_CALUDE_train_passing_time_l647_64725

theorem train_passing_time (fast_train_length slow_train_length : ℝ)
  (fast_train_passing_time : ℝ) (h1 : fast_train_length = 315)
  (h2 : slow_train_length = 300) (h3 : fast_train_passing_time = 21) :
  slow_train_length / (fast_train_length / fast_train_passing_time) = 20 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l647_64725


namespace NUMINAMATH_CALUDE_log_inequality_implies_order_l647_64700

theorem log_inequality_implies_order (x y : ℝ) :
  (Real.log x / Real.log (1/2)) < (Real.log y / Real.log (1/2)) ∧
  (Real.log y / Real.log (1/2)) < 0 →
  1 < y ∧ y < x :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_implies_order_l647_64700


namespace NUMINAMATH_CALUDE_complementary_angle_proof_l647_64772

-- Define complementary angles
def complementary (angle1 angle2 : ℝ) : Prop :=
  angle1 + angle2 = 90

-- Theorem statement
theorem complementary_angle_proof (angle1 angle2 : ℝ) 
  (h1 : complementary angle1 angle2) (h2 : angle1 = 25) : 
  angle2 = 65 := by
  sorry


end NUMINAMATH_CALUDE_complementary_angle_proof_l647_64772


namespace NUMINAMATH_CALUDE_luncheon_no_shows_l647_64731

/-- Theorem: Number of no-shows at a luncheon --/
theorem luncheon_no_shows 
  (invited : ℕ) 
  (tables : ℕ) 
  (people_per_table : ℕ) 
  (h1 : invited = 47) 
  (h2 : tables = 8) 
  (h3 : people_per_table = 5) : 
  invited - (tables * people_per_table) = 7 := by
  sorry

#check luncheon_no_shows

end NUMINAMATH_CALUDE_luncheon_no_shows_l647_64731


namespace NUMINAMATH_CALUDE_min_value_expression_l647_64759

theorem min_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : b ≠ 0) :
  ((a + b)^2 + (b - c)^2 + (c - a)^2) / b^2 ≥ 4/3 ∧
  ∃ (a' b' c' : ℝ), b' > c' ∧ c' > a' ∧ b' ≠ 0 ∧
    ((a' + b')^2 + (b' - c')^2 + (c' - a')^2) / b'^2 = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l647_64759


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l647_64740

/-- The parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2}

/-- The point Q -/
def Q : ℝ × ℝ := (20, 14)

/-- The line through Q with slope m -/
def line_through_Q (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - Q.2 = m * (p.1 - Q.1)}

/-- The condition for non-intersection -/
def no_intersection (m : ℝ) : Prop :=
  line_through_Q m ∩ P = ∅

theorem parabola_line_intersection :
  ∃ (r s : ℝ), (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 80 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l647_64740


namespace NUMINAMATH_CALUDE_smallest_cube_ending_368_l647_64766

theorem smallest_cube_ending_368 :
  ∀ n : ℕ+, n.val^3 ≡ 368 [MOD 1000] → n.val ≥ 14 :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_368_l647_64766


namespace NUMINAMATH_CALUDE_arrangements_theorem_l647_64763

def number_of_arrangements (n : ℕ) (red yellow blue : ℕ) : ℕ :=
  sorry

theorem arrangements_theorem :
  number_of_arrangements 5 2 2 1 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_theorem_l647_64763


namespace NUMINAMATH_CALUDE_solutions_of_f_eq_quarter_solution_set_of_f_leq_two_l647_64704

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else Real.log x / Real.log 4

-- Theorem for the solutions of f(x) = 1/4
theorem solutions_of_f_eq_quarter :
  {x : ℝ | f x = 1/4} = {2, Real.sqrt 2} :=
sorry

-- Theorem for the solution set of f(x) ≤ 2
theorem solution_set_of_f_leq_two :
  {x : ℝ | f x ≤ 2} = Set.Icc (-1) 16 :=
sorry

end NUMINAMATH_CALUDE_solutions_of_f_eq_quarter_solution_set_of_f_leq_two_l647_64704


namespace NUMINAMATH_CALUDE_cool_drink_solution_volume_l647_64744

/-- Represents the cool-drink solution problem --/
theorem cool_drink_solution_volume 
  (initial_jasmine_percent : Real)
  (added_jasmine : Real)
  (added_water : Real)
  (final_jasmine_percent : Real)
  (h1 : initial_jasmine_percent = 0.05)
  (h2 : added_jasmine = 8)
  (h3 : added_water = 2)
  (h4 : final_jasmine_percent = 0.125)
  : ∃ (initial_volume : Real),
    initial_volume * initial_jasmine_percent + added_jasmine = 
    (initial_volume + added_jasmine + added_water) * final_jasmine_percent ∧
    initial_volume = 90 := by
  sorry

end NUMINAMATH_CALUDE_cool_drink_solution_volume_l647_64744


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_negative_l647_64767

theorem consecutive_integers_sum_negative : ∃ n : ℤ, 
  (n^2 - 13*n + 36) + ((n+1)^2 - 13*(n+1) + 36) < 0 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_negative_l647_64767


namespace NUMINAMATH_CALUDE_evaluate_expression_l647_64770

theorem evaluate_expression : -((18 / 3)^2 * 4 - 80 + 5 * 7) = -99 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l647_64770


namespace NUMINAMATH_CALUDE_proof_by_contradiction_principle_l647_64797

theorem proof_by_contradiction_principle :
  ∀ (P : Prop), (¬P → False) → P :=
by
  sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_principle_l647_64797


namespace NUMINAMATH_CALUDE_jerrys_painting_time_l647_64741

theorem jerrys_painting_time (fixing_time painting_time mowing_time hourly_rate total_payment : ℝ) :
  fixing_time = 3 * painting_time →
  mowing_time = 6 →
  hourly_rate = 15 →
  total_payment = 570 →
  hourly_rate * (painting_time + fixing_time + mowing_time) = total_payment →
  painting_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_painting_time_l647_64741


namespace NUMINAMATH_CALUDE_initial_cats_l647_64788

theorem initial_cats (initial_cats final_cats bought_cats : ℕ) : 
  final_cats = initial_cats + bought_cats ∧ 
  final_cats = 54 ∧ 
  bought_cats = 43 →
  initial_cats = 11 := by sorry

end NUMINAMATH_CALUDE_initial_cats_l647_64788


namespace NUMINAMATH_CALUDE_smallest_nonfactor_product_of_48_l647_64749

theorem smallest_nonfactor_product_of_48 :
  ∃ (u v : ℕ), 
    u ≠ v ∧ 
    u > 0 ∧ 
    v > 0 ∧ 
    48 % u = 0 ∧ 
    48 % v = 0 ∧ 
    48 % (u * v) ≠ 0 ∧
    u * v = 18 ∧
    (∀ (x y : ℕ), x ≠ y → x > 0 → y > 0 → 48 % x = 0 → 48 % y = 0 → 48 % (x * y) ≠ 0 → x * y ≥ 18) :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonfactor_product_of_48_l647_64749


namespace NUMINAMATH_CALUDE_joan_apples_l647_64764

/-- The number of apples Joan has after picking and giving some away -/
def apples_remaining (picked : ℕ) (given_away : ℕ) : ℕ :=
  picked - given_away

/-- Theorem: Joan has 16 apples after picking 43 and giving away 27 -/
theorem joan_apples : apples_remaining 43 27 = 16 := by
  sorry

end NUMINAMATH_CALUDE_joan_apples_l647_64764


namespace NUMINAMATH_CALUDE_largest_square_area_l647_64794

/-- Represents a right triangle with squares on each side -/
structure RightTriangleWithSquares where
  -- Side lengths
  xz : ℝ
  yz : ℝ
  xy : ℝ
  -- Right angle condition
  right_angle : xy^2 = xz^2 + yz^2
  -- Non-negativity of side lengths
  xz_nonneg : xz ≥ 0
  yz_nonneg : yz ≥ 0
  xy_nonneg : xy ≥ 0

/-- The theorem to be proved -/
theorem largest_square_area
  (t : RightTriangleWithSquares)
  (sum_area : t.xz^2 + t.yz^2 + t.xy^2 = 450) :
  t.xy^2 = 225 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_area_l647_64794


namespace NUMINAMATH_CALUDE_log_inequality_l647_64735

theorem log_inequality (x : ℝ) (h : 0 < x) (h' : x < 1) : Real.log (1 + x) > x^3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l647_64735


namespace NUMINAMATH_CALUDE_car_wash_contribution_l647_64771

def goal : ℕ := 150
def families_with_known_contribution : ℕ := 15
def known_contribution_per_family : ℕ := 5
def remaining_families : ℕ := 3
def amount_needed : ℕ := 45

theorem car_wash_contribution :
  ∀ (contribution_per_remaining_family : ℕ),
    (families_with_known_contribution * known_contribution_per_family) +
    (remaining_families * contribution_per_remaining_family) =
    goal - amount_needed →
    contribution_per_remaining_family = 10 := by
  sorry

end NUMINAMATH_CALUDE_car_wash_contribution_l647_64771


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l647_64753

theorem solution_set_of_inequality (x : ℝ) :
  Set.Icc (-1/2 : ℝ) 3 \ {3} = {x | (2*x + 1) / (3 - x) ≥ 0 ∧ x ≠ 3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l647_64753


namespace NUMINAMATH_CALUDE_eraser_cost_proof_l647_64761

/-- Represents the price of a single pencil -/
def pencil_price : ℝ := 2

/-- Represents the price of a single eraser -/
def eraser_price : ℝ := 1

/-- The number of pencils sold -/
def pencils_sold : ℕ := 20

/-- The number of erasers sold -/
def erasers_sold : ℕ := pencils_sold * 2

/-- The total revenue from sales -/
def total_revenue : ℝ := 80

theorem eraser_cost_proof :
  (pencils_sold : ℝ) * pencil_price + (erasers_sold : ℝ) * eraser_price = total_revenue ∧
  2 * eraser_price = pencil_price ∧
  eraser_price = 1 :=
by sorry

end NUMINAMATH_CALUDE_eraser_cost_proof_l647_64761


namespace NUMINAMATH_CALUDE_solution_set_inequality_l647_64702

theorem solution_set_inequality (x : ℝ) : 
  (2*x - 1) / x < 0 ↔ 0 < x ∧ x < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l647_64702


namespace NUMINAMATH_CALUDE_apple_cost_price_l647_64728

theorem apple_cost_price (SP : ℝ) (CP : ℝ) : SP = 16 ∧ SP = (5/6) * CP → CP = 19.2 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_price_l647_64728


namespace NUMINAMATH_CALUDE_circle_radius_nine_iff_k_94_l647_64792

/-- The equation of a circle in general form --/
def circle_equation (x y k : ℝ) : Prop :=
  2 * x^2 + 20 * x + 3 * y^2 + 18 * y - k = 0

/-- The equation of a circle in standard form with center (h, k) and radius r --/
def standard_circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that the given equation represents a circle with radius 9 iff k = 94 --/
theorem circle_radius_nine_iff_k_94 :
  (∃ h k : ℝ, ∀ x y : ℝ, circle_equation x y 94 ↔ standard_circle_equation x y h k 9) ↔
  (∀ k : ℝ, (∃ h k : ℝ, ∀ x y : ℝ, circle_equation x y k ↔ standard_circle_equation x y h k 9) → k = 94) :=
sorry

end NUMINAMATH_CALUDE_circle_radius_nine_iff_k_94_l647_64792


namespace NUMINAMATH_CALUDE_f_monotone_and_roots_sum_l647_64719

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2*x - 2*(a+1)*Real.exp x + a*Real.exp (2*x)

theorem f_monotone_and_roots_sum (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a = 1 ∧
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ = a * Real.exp (2*x₁) → f a x₂ = a * Real.exp (2*x₂) → x₁ + x₂ > 2 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_and_roots_sum_l647_64719


namespace NUMINAMATH_CALUDE_total_count_is_1500_l647_64789

/-- The number of people counted on the second day -/
def second_day_count : ℕ := 500

/-- The number of people counted on the first day -/
def first_day_count : ℕ := 2 * second_day_count

/-- The total number of people counted over two days -/
def total_count : ℕ := first_day_count + second_day_count

theorem total_count_is_1500 : total_count = 1500 := by
  sorry

end NUMINAMATH_CALUDE_total_count_is_1500_l647_64789


namespace NUMINAMATH_CALUDE_solution_l647_64750

def problem (B : ℕ) (A : ℕ) (X : ℕ) : Prop :=
  B = 38 ∧ 
  A = B + 8 ∧ 
  A + 10 = 2 * (B - X)

theorem solution : ∃ X, problem 38 (38 + 8) X ∧ X = 10 := by
  sorry

end NUMINAMATH_CALUDE_solution_l647_64750


namespace NUMINAMATH_CALUDE_m_is_positive_l647_64793

-- Define the sets M and N
def M (m : ℝ) : Set ℝ := {x | x ≤ m}
def N : Set ℝ := {y | ∃ x : ℝ, y = 2^(-x)}

-- State the theorem
theorem m_is_positive (m : ℝ) (h : (M m) ∩ N ≠ ∅) : m > 0 := by
  sorry

end NUMINAMATH_CALUDE_m_is_positive_l647_64793


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l647_64774

/-- Given a cube with surface area 24 square centimeters, its volume is 8 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ (side_length : ℝ),
  (6 * side_length^2 = 24) →
  side_length^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l647_64774


namespace NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l647_64746

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem third_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : arithmeticSequence a)
  (h_first : a 1 = 2)
  (h_second : a 2 = 8) :
  a 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l647_64746


namespace NUMINAMATH_CALUDE_part_one_part_two_l647_64780

-- Define propositions p and q
def p (k : ℝ) : Prop := k^2 - 2*k - 24 ≤ 0

def q (k : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b < 0 ∧ a = 3 - k ∧ b = 3 + k

-- Part 1
theorem part_one (k : ℝ) : q k → k ∈ Set.Iio (-3) := by
  sorry

-- Part 2
theorem part_two (k : ℝ) : (p k ∨ q k) ∧ ¬(p k ∧ q k) → k ∈ Set.Iio (-4) ∪ Set.Icc (-3) 6 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l647_64780


namespace NUMINAMATH_CALUDE_min_value_trig_expression_equality_condition_l647_64720

theorem min_value_trig_expression (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 20)^2 ≥ 236.137 := by
  sorry

theorem equality_condition (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 20)^2 = 236.137 ↔ 
  (Real.cos α = 10 / Real.sqrt 500 ∧ Real.sin α = 20 / Real.sqrt 500 ∧ β = Real.pi/2 - α) := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_equality_condition_l647_64720


namespace NUMINAMATH_CALUDE_negative_real_inequality_l647_64718

theorem negative_real_inequality (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  a > b ↔ a - 1 / a > b - 1 / b := by sorry

end NUMINAMATH_CALUDE_negative_real_inequality_l647_64718


namespace NUMINAMATH_CALUDE_pet_store_cats_l647_64773

/-- The number of house cats initially in the pet store -/
def initial_house_cats : ℝ := 5.0

/-- The number of Siamese cats initially in the pet store -/
def initial_siamese_cats : ℝ := 13.0

/-- The number of cats added during the purchase -/
def added_cats : ℝ := 10.0

/-- The total number of cats after the addition -/
def total_cats_after : ℝ := 28.0

theorem pet_store_cats :
  initial_house_cats + initial_siamese_cats + added_cats = total_cats_after :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cats_l647_64773


namespace NUMINAMATH_CALUDE_line_problem_l647_64781

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3 * x + 2 * y - 1 = 0
def l₂ (x y : ℝ) : Prop := 5 * x + 2 * y + 1 = 0
def l₃ (a x y : ℝ) : Prop := (a^2 - 1) * x + a * y - 1 = 0

-- Define the intersection point A
def A : ℝ × ℝ := (-1, 2)

-- Define parallelism
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, f x y ↔ g (k * x) (k * y)

-- Define a line with equal intercepts
def equal_intercepts (m b : ℝ) : Prop :=
  b / m + b = 0

theorem line_problem :
  (∃ a : ℝ, parallel (l₃ a) l₁ ∧ a = -1/2) ∧
  (∃ m b : ℝ, (m = -1 ∧ b = 1) ∨ (m = -2 ∧ b = 0) ∧
    l₁ A.1 A.2 ∧ l₂ A.1 A.2 ∧ equal_intercepts m b) :=
sorry

end NUMINAMATH_CALUDE_line_problem_l647_64781


namespace NUMINAMATH_CALUDE_quadratic_radicals_theorem_l647_64745

-- Define the condition that the radicals can be combined
def radicals_can_combine (a : ℝ) : Prop := 3 * a - 8 = 17 - 2 * a

-- Define the range of x that makes √(4a-2x) meaningful
def valid_x_range (a x : ℝ) : Prop := 4 * a - 2 * x ≥ 0

-- Theorem statement
theorem quadratic_radicals_theorem (a x : ℝ) :
  radicals_can_combine a → (∃ a, radicals_can_combine a ∧ a = 5) →
  (valid_x_range a x ↔ x ≤ 10) :=
sorry

end NUMINAMATH_CALUDE_quadratic_radicals_theorem_l647_64745


namespace NUMINAMATH_CALUDE_range_of_m_l647_64757

-- Define the sets A and B
def A : Set ℝ := {x | x < 1}
def B (m : ℝ) : Set ℝ := {x | x > m}

-- State the theorem
theorem range_of_m (m : ℝ) : (Set.compl A) ⊆ B m → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l647_64757


namespace NUMINAMATH_CALUDE_necessary_condition_for_inequality_l647_64742

theorem necessary_condition_for_inequality (a b c : ℝ) (h1 : a * c^2 > b * c^2) (h2 : c ≠ 0) : a > b := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_for_inequality_l647_64742


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l647_64707

theorem unique_function_satisfying_conditions :
  ∃! f : ℝ → ℝ, 
    (∀ x : ℝ, f (x + 1) ≥ f x + 1) ∧ 
    (∀ x y : ℝ, f (x * y) ≥ f x * f y) ∧
    (∀ x : ℝ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l647_64707


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l647_64762

theorem polynomial_identity_sum (a₀ a₁ a₂ a₃ : ℝ) 
  (h : ∀ x : ℝ, 1 + x + x^2 + x^3 = a₀ + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3) : 
  a₁ + a₂ + a₃ = -3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l647_64762


namespace NUMINAMATH_CALUDE_parallelogram_area_12_8_l647_64768

/-- The area of a parallelogram with given base and height -/
def parallelogramArea (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 12 cm and height 8 cm is 96 square centimeters -/
theorem parallelogram_area_12_8 : parallelogramArea 12 8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_12_8_l647_64768


namespace NUMINAMATH_CALUDE_class_size_l647_64739

theorem class_size :
  ∀ (m d : ℕ),
  (m + d > 30) →
  (m + d < 40) →
  (3 * m = 5 * d) →
  (m + d = 32) :=
by
  sorry

end NUMINAMATH_CALUDE_class_size_l647_64739


namespace NUMINAMATH_CALUDE_ace_king_probability_l647_64710

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Aces in a standard deck -/
def num_aces : ℕ := 4

/-- The number of Kings in a standard deck -/
def num_kings : ℕ := 4

/-- The probability of drawing an Ace followed by a King from a standard deck -/
theorem ace_king_probability : 
  (num_aces : ℚ) / deck_size * num_kings / (deck_size - 1) = 4 / 663 := by
sorry

end NUMINAMATH_CALUDE_ace_king_probability_l647_64710


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l647_64755

theorem polynomial_root_sum (m : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2500*x + m = 0 ↔ x = p ∨ x = q ∨ x = r) →
  p + q + r = 0 →
  p * q + q * r + r * p = -2500 →
  p * q * r = -m →
  |p| + |q| + |r| = 100 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l647_64755


namespace NUMINAMATH_CALUDE_probability_three_even_dice_l647_64714

def num_dice : ℕ := 5
def faces_per_die : ℕ := 20
def target_even : ℕ := 3

theorem probability_three_even_dice :
  let p_even : ℚ := 1 / 2  -- Probability of rolling an even number on a single die
  let p_arrangement : ℚ := p_even ^ target_even * (1 - p_even) ^ (num_dice - target_even)
  let num_arrangements : ℕ := Nat.choose num_dice target_even
  num_arrangements * p_arrangement = 5 / 16 := by
sorry

end NUMINAMATH_CALUDE_probability_three_even_dice_l647_64714


namespace NUMINAMATH_CALUDE_tournament_participants_l647_64730

/-- The number of games played in a chess tournament -/
def num_games : ℕ := 136

/-- Calculates the number of games played in a tournament given the number of participants -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Proves that the number of participants in the tournament is 17 -/
theorem tournament_participants : ∃ n : ℕ, n > 0 ∧ games_played n = num_games ∧ n = 17 := by
  sorry

end NUMINAMATH_CALUDE_tournament_participants_l647_64730


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l647_64734

/-- An arithmetic sequence with the given properties has 13 terms -/
theorem arithmetic_sequence_terms (a d : ℝ) (n : ℕ) 
  (h1 : 3 * a + 3 * d = 34)
  (h2 : 3 * a + 3 * (n - 1) * d = 146)
  (h3 : n * (2 * a + (n - 1) * d) / 2 = 390)
  : n = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l647_64734


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l647_64708

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ - 3 = 0 ∧ x₂^2 + x₂ - 3 = 0 → x₁^3 - 4*x₂^2 + 19 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l647_64708


namespace NUMINAMATH_CALUDE_abc_inequality_l647_64777

noncomputable def a : ℝ := 2 / Real.log 2
noncomputable def b : ℝ := Real.exp 2 / (4 - Real.log 4)
noncomputable def c : ℝ := 2 * Real.sqrt (Real.exp 1)

theorem abc_inequality : c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l647_64777


namespace NUMINAMATH_CALUDE_modular_inverse_of_2_mod_191_l647_64787

theorem modular_inverse_of_2_mod_191 : ∃ x : ℕ, x < 191 ∧ (2 * x) % 191 = 1 :=
  ⟨96, by
    constructor
    · simp
    · norm_num
  ⟩

#eval (2 * 96) % 191  -- This should output 1

end NUMINAMATH_CALUDE_modular_inverse_of_2_mod_191_l647_64787


namespace NUMINAMATH_CALUDE_flower_bed_area_l647_64722

theorem flower_bed_area (a b c : ℝ) (ha : a = 6) (hb : b = 8) (hc : c = 10) :
  (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_area_l647_64722


namespace NUMINAMATH_CALUDE_grade_distribution_l647_64760

theorem grade_distribution (total_students : ℕ) 
  (below_b_percent : ℚ) (b_or_bplus_percent : ℚ) (a_or_aminus_percent : ℚ) (aplus_percent : ℚ) :
  total_students = 60 →
  below_b_percent = 40 / 100 →
  b_or_bplus_percent = 30 / 100 →
  a_or_aminus_percent = 20 / 100 →
  aplus_percent = 10 / 100 →
  below_b_percent + b_or_bplus_percent + a_or_aminus_percent + aplus_percent = 1 →
  (b_or_bplus_percent + a_or_aminus_percent) * total_students = 30 := by
  sorry

end NUMINAMATH_CALUDE_grade_distribution_l647_64760


namespace NUMINAMATH_CALUDE_problem_solution_l647_64715

theorem problem_solution (a b : ℚ) 
  (eq1 : 8*a + 3*b = -1)
  (eq2 : a = b - 3) : 
  5*b = 115/11 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l647_64715


namespace NUMINAMATH_CALUDE_min_value_constrained_l647_64778

/-- Given that x + 2y + 3z = 1, the minimum value of x^2 + y^2 + z^2 is 1/14 -/
theorem min_value_constrained (x y z : ℝ) (h : x + 2*y + 3*z = 1) :
  ∃ (min : ℝ), min = (1 : ℝ) / 14 ∧ 
  ∀ (a b c : ℝ), a + 2*b + 3*c = 1 → x^2 + y^2 + z^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_constrained_l647_64778


namespace NUMINAMATH_CALUDE_pastry_count_consistency_l647_64799

/-- Represents the number of pastries in different states --/
structure Pastries where
  initial : ℕ
  sold : ℕ
  remaining : ℕ

/-- The problem statement --/
theorem pastry_count_consistency (p : Pastries) 
  (h1 : p.initial = 148)
  (h2 : p.sold = 103)
  (h3 : p.remaining = 45) :
  p.initial = p.sold + p.remaining := by
  sorry

end NUMINAMATH_CALUDE_pastry_count_consistency_l647_64799


namespace NUMINAMATH_CALUDE_unique_intersection_condition_l647_64775

/-- A function f(x) = kx^2 + 2(k+1)x + k-1 has only one intersection point with the x-axis if and only if k = 0 or k = -1/3 -/
theorem unique_intersection_condition (k : ℝ) : 
  (∃! x, k * x^2 + 2*(k+1)*x + (k-1) = 0) ↔ (k = 0 ∨ k = -1/3) := by
sorry

end NUMINAMATH_CALUDE_unique_intersection_condition_l647_64775


namespace NUMINAMATH_CALUDE_shekars_weighted_average_sum_of_weightages_is_one_l647_64706

/-- Represents the subjects and their corresponding scores and weightages -/
structure Subject where
  name : String
  score : ℝ
  weightage : ℝ

/-- Calculates the weighted average of a list of subjects -/
def weightedAverage (subjects : List Subject) : ℝ :=
  (subjects.map (fun s => s.score * s.weightage)).sum

/-- Shekar's subjects with their scores and weightages -/
def shekarsSubjects : List Subject := [
  ⟨"Mathematics", 76, 0.15⟩,
  ⟨"Science", 65, 0.15⟩,
  ⟨"Social Studies", 82, 0.20⟩,
  ⟨"English", 67, 0.20⟩,
  ⟨"Biology", 75, 0.10⟩,
  ⟨"Computer Science", 89, 0.10⟩,
  ⟨"History", 71, 0.10⟩
]

/-- Theorem stating that Shekar's weighted average marks is 74.45 -/
theorem shekars_weighted_average :
  weightedAverage shekarsSubjects = 74.45 := by
  sorry

/-- Proof that the sum of weightages is 1 -/
theorem sum_of_weightages_is_one :
  (shekarsSubjects.map (fun s => s.weightage)).sum = 1 := by
  sorry

end NUMINAMATH_CALUDE_shekars_weighted_average_sum_of_weightages_is_one_l647_64706


namespace NUMINAMATH_CALUDE_petya_running_time_l647_64791

theorem petya_running_time (a V : ℝ) (h1 : a > 0) (h2 : V > 0) :
  (a / (2.5 * V) + a / (1.6 * V)) > (a / V) := by
  sorry

end NUMINAMATH_CALUDE_petya_running_time_l647_64791


namespace NUMINAMATH_CALUDE_polynomial_roots_l647_64798

theorem polynomial_roots : ∃ (a b : ℝ), 
  (∀ x : ℝ, 6*x^4 + 25*x^3 - 59*x^2 + 28*x = 0 ↔ 
    x = 0 ∨ x = 1 ∨ x = (-31 + Real.sqrt 1633) / 12 ∨ x = (-31 - Real.sqrt 1633) / 12) ∧
  a = (-31 + Real.sqrt 1633) / 12 ∧
  b = (-31 - Real.sqrt 1633) / 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l647_64798


namespace NUMINAMATH_CALUDE_unique_prime_with_remainder_l647_64733

theorem unique_prime_with_remainder : ∃! n : ℕ,
  20 < n ∧ n < 30 ∧
  Prime n ∧
  n % 8 = 5 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_remainder_l647_64733
