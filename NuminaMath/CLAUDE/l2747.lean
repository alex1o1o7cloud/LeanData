import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_composition_l2747_274783

/-- Given f(x) = x² and f(h(x)) = 9x² + 6x + 1, prove that h(x) = 3x + 1 or h(x) = -3x - 1 -/
theorem polynomial_composition (f h : ℝ → ℝ) : 
  (∀ x, f x = x^2) → 
  (∀ x, f (h x) = 9*x^2 + 6*x + 1) → 
  (∀ x, h x = 3*x + 1 ∨ h x = -3*x - 1) := by
sorry

end NUMINAMATH_CALUDE_polynomial_composition_l2747_274783


namespace NUMINAMATH_CALUDE_probability_green_given_no_red_l2747_274731

/-- The set of all possible colors for memories -/
inductive Color
| Red
| Green
| Blue
| Yellow
| Purple

/-- A memory coloring is a set of at most two distinct colors -/
def MemoryColoring := Finset Color

/-- The set of all valid memory colorings -/
def AllColorings : Finset MemoryColoring :=
  sorry

/-- The set of memory colorings without red -/
def ColoringsWithoutRed : Finset MemoryColoring :=
  sorry

/-- The set of memory colorings that are at least partly green and have no red -/
def GreenColoringsWithoutRed : Finset MemoryColoring :=
  sorry

/-- The probability of a memory being at least partly green given that it has no red -/
theorem probability_green_given_no_red :
  (Finset.card GreenColoringsWithoutRed) / (Finset.card ColoringsWithoutRed) = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_probability_green_given_no_red_l2747_274731


namespace NUMINAMATH_CALUDE_bananas_left_l2747_274701

def dozen : Nat := 12

theorem bananas_left (initial : Nat) (eaten : Nat) : 
  initial = dozen → eaten = 1 → initial - eaten = 11 := by
  sorry

end NUMINAMATH_CALUDE_bananas_left_l2747_274701


namespace NUMINAMATH_CALUDE_field_area_is_500_l2747_274765

/-- Represents the area of a field divided into two parts -/
structure FieldArea where
  small : ℝ
  large : ℝ

/-- Calculates the total area of the field -/
def total_area (f : FieldArea) : ℝ := f.small + f.large

/-- Theorem: The total area of the field is 500 hectares -/
theorem field_area_is_500 (f : FieldArea) 
  (h1 : f.small = 225)
  (h2 : f.large - f.small = (1/5) * ((f.small + f.large) / 2)) :
  total_area f = 500 := by
  sorry

end NUMINAMATH_CALUDE_field_area_is_500_l2747_274765


namespace NUMINAMATH_CALUDE_trapezoid_area_theorem_l2747_274714

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Represents a trapezoid formed by the arrangement of squares -/
structure Trapezoid where
  squares : List Square
  connector : ℝ × ℝ  -- Represents the connecting line segment

/-- Calculates the area of the trapezoid formed by the arrangement of squares -/
noncomputable def calculateTrapezoidArea (t : Trapezoid) : ℝ :=
  sorry

/-- The main theorem stating the area of the trapezoid -/
theorem trapezoid_area_theorem (s1 s2 s3 s4 : Square) 
  (h1 : s1.sideLength = 3)
  (h2 : s2.sideLength = 5)
  (h3 : s3.sideLength = 7)
  (h4 : s4.sideLength = 7)
  (t : Trapezoid)
  (ht : t.squares = [s1, s2, s3, s4]) :
  abs (calculateTrapezoidArea t - 12.83325) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_theorem_l2747_274714


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2747_274775

def U : Set Nat := {0,1,2,3,4,5,6,7,8,9}
def A : Set Nat := {0,1,3,5,8}
def B : Set Nat := {2,4,5,6,8}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {7,9} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2747_274775


namespace NUMINAMATH_CALUDE_quadratic_value_at_5_l2747_274711

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_value_at_5 
  (a b c : ℝ) 
  (max_at_2 : ∀ x, quadratic a b c x ≤ quadratic a b c 2)
  (max_value : quadratic a b c 2 = 6)
  (passes_origin : quadratic a b c 0 = -10) :
  quadratic a b c 5 = -30 := by
sorry

end NUMINAMATH_CALUDE_quadratic_value_at_5_l2747_274711


namespace NUMINAMATH_CALUDE_combined_weight_theorem_l2747_274794

/-- Represents the elevator scenario with people and their weights -/
structure ElevatorScenario where
  initial_people : ℕ
  initial_avg_weight : ℝ
  new_avg_weights : List ℝ

/-- Calculates the combined weight of new people entering the elevator -/
def combined_weight_of_new_people (scenario : ElevatorScenario) : ℝ :=
  sorry

/-- Theorem stating the combined weight of new people in the given scenario -/
theorem combined_weight_theorem (scenario : ElevatorScenario) :
  scenario.initial_people = 6 →
  scenario.initial_avg_weight = 152 →
  scenario.new_avg_weights = [154, 153, 151] →
  combined_weight_of_new_people scenario = 447 := by
  sorry

#check combined_weight_theorem

end NUMINAMATH_CALUDE_combined_weight_theorem_l2747_274794


namespace NUMINAMATH_CALUDE_girls_in_class_l2747_274747

/-- Proves that in a class with a boy-to-girl ratio of 5:8 and 260 total students, there are 160 girls -/
theorem girls_in_class (total : ℕ) (boys_ratio girls_ratio : ℕ) (h1 : total = 260) (h2 : boys_ratio = 5) (h3 : girls_ratio = 8) : 
  (girls_ratio : ℚ) / (boys_ratio + girls_ratio : ℚ) * total = 160 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l2747_274747


namespace NUMINAMATH_CALUDE_sandy_earnings_l2747_274727

/-- Calculates Sandy's earnings for a given day -/
def daily_earnings (hours : ℝ) (hourly_rate : ℝ) (with_best_friend : Bool) (longer_than_12_hours : Bool) : ℝ :=
  let base_wage := hours * hourly_rate
  let commission := if with_best_friend then base_wage * 0.1 else 0
  let bonus := if longer_than_12_hours then base_wage * 0.05 else 0
  let total_before_tax := base_wage + commission + bonus
  total_before_tax * 0.93  -- Apply 7% tax deduction

/-- Sandy's total earnings for Friday, Saturday, and Sunday -/
def total_earnings : ℝ :=
  let hourly_rate := 15
  let friday := daily_earnings 10 hourly_rate true false
  let saturday := daily_earnings 6 hourly_rate false false
  let sunday := daily_earnings 14 hourly_rate false true
  friday + saturday + sunday

/-- Theorem stating Sandy's total earnings -/
theorem sandy_earnings : total_earnings = 442.215 := by
  sorry

end NUMINAMATH_CALUDE_sandy_earnings_l2747_274727


namespace NUMINAMATH_CALUDE_blue_section_probability_l2747_274715

/-- The number of Bernoulli trials -/
def n : ℕ := 7

/-- The probability of success in each trial -/
def p : ℚ := 2/7

/-- The number of successes we're interested in -/
def k : ℕ := 7

/-- The probability of achieving exactly k successes in n Bernoulli trials 
    with success probability p -/
def bernoulli_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem blue_section_probability : 
  bernoulli_probability n k p = 128/823543 := by
  sorry

end NUMINAMATH_CALUDE_blue_section_probability_l2747_274715


namespace NUMINAMATH_CALUDE_four_propositions_l2747_274779

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + x - m

-- Define what it means for a function to have zero points
def has_zero_points (f : ℝ → ℝ) : Prop := ∃ x, f x = 0

-- Define what it means for four points to be coplanar
def coplanar (E F G H : ℝ × ℝ × ℝ) : Prop := sorry

-- Define what it means for two lines to intersect
def lines_intersect (E F G H : ℝ × ℝ × ℝ) : Prop := sorry

-- Define what it means for an equation to represent a hyperbola
def is_hyperbola (m : ℝ) : Prop := sorry

theorem four_propositions :
  (∀ m > 0, has_zero_points (f m)) ∧ 
  (∀ E F G H, ¬coplanar E F G H → ¬lines_intersect E F G H) ∧
  (∃ E F G H, ¬lines_intersect E F G H ∧ coplanar E F G H) ∧
  (∀ a : ℝ, (∀ x : ℝ, |x+1| + |x-1| ≥ a) ↔ a < 2) ∧
  (∀ m : ℝ, (0 < m ∧ m < 1) ↔ is_hyperbola m) :=
by
  sorry

end NUMINAMATH_CALUDE_four_propositions_l2747_274779


namespace NUMINAMATH_CALUDE_outdoor_section_length_l2747_274760

/-- The length of a rectangular outdoor section, given its width and area -/
theorem outdoor_section_length (width : ℝ) (area : ℝ) (h1 : width = 4) (h2 : area = 24) :
  area / width = 6 := by
  sorry

end NUMINAMATH_CALUDE_outdoor_section_length_l2747_274760


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2747_274753

/-- The perimeter of a rhombus with diagonals of 12 inches and 16 inches is 40 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 40 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2747_274753


namespace NUMINAMATH_CALUDE_triangle_area_in_circle_l2747_274743

-- Define the triangle and circle
def triangle_ratio : Vector ℝ 3 := ⟨[6, 8, 10], by simp⟩
def circle_radius : ℝ := 5

-- Theorem statement
theorem triangle_area_in_circle (sides : Vector ℝ 3) (r : ℝ) 
  (h1 : sides = triangle_ratio) 
  (h2 : r = circle_radius) : 
  ∃ (a b c : ℝ), 
    a * sides[0] = b * sides[1] ∧ 
    a * sides[0] = c * sides[2] ∧
    b * sides[1] = c * sides[2] ∧
    (a * sides[0])^2 + (b * sides[1])^2 = (c * sides[2])^2 ∧
    c * sides[2] = 2 * r ∧
    (1/2) * (a * sides[0]) * (b * sides[1]) = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_in_circle_l2747_274743


namespace NUMINAMATH_CALUDE_chocolates_gain_percent_l2747_274754

/-- Calculates the gain percent given the number of chocolates at cost price and selling price that are equal in value -/
def gain_percent (cost_chocolates : ℕ) (sell_chocolates : ℕ) : ℚ :=
  ((cost_chocolates : ℚ) / sell_chocolates - 1) * 100

/-- Theorem stating that if the cost price of 81 chocolates equals the selling price of 45 chocolates, the gain percent is 80% -/
theorem chocolates_gain_percent :
  gain_percent 81 45 = 80 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_gain_percent_l2747_274754


namespace NUMINAMATH_CALUDE_one_plus_three_squared_l2747_274752

theorem one_plus_three_squared : 1 + 3^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_one_plus_three_squared_l2747_274752


namespace NUMINAMATH_CALUDE_sum_of_integers_ending_in_3_l2747_274793

def sequence_first_term : ℕ := 103
def sequence_last_term : ℕ := 443
def sequence_common_difference : ℕ := 10

def sequence_length : ℕ := (sequence_last_term - sequence_first_term) / sequence_common_difference + 1

theorem sum_of_integers_ending_in_3 :
  (sequence_length : ℕ) * (sequence_first_term + sequence_last_term) / 2 = 9555 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_integers_ending_in_3_l2747_274793


namespace NUMINAMATH_CALUDE_shell_ratio_l2747_274704

/-- Prove that the ratio of Kyle's shells to Mimi's shells is 2:1 -/
theorem shell_ratio : 
  ∀ (mimi_shells kyle_shells leigh_shells : ℕ),
    mimi_shells = 2 * 12 →
    leigh_shells = 16 →
    3 * leigh_shells = kyle_shells →
    kyle_shells / mimi_shells = 2 := by
  sorry

end NUMINAMATH_CALUDE_shell_ratio_l2747_274704


namespace NUMINAMATH_CALUDE_any_nonzero_rational_to_zero_power_is_one_l2747_274734

theorem any_nonzero_rational_to_zero_power_is_one (r : ℚ) (h : r ≠ 0) : r^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_any_nonzero_rational_to_zero_power_is_one_l2747_274734


namespace NUMINAMATH_CALUDE_rectangle_area_error_percent_l2747_274730

/-- Given a rectangle with actual length L and width W, where one side is measured
    as 1.05L and the other as 0.96W, the error percent in the calculated area is 0.8%. -/
theorem rectangle_area_error_percent (L W : ℝ) (L_pos : L > 0) (W_pos : W > 0) :
  let actual_area := L * W
  let measured_area := (1.05 * L) * (0.96 * W)
  let error := measured_area - actual_area
  let error_percent := (error / actual_area) * 100
  error_percent = 0.8 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percent_l2747_274730


namespace NUMINAMATH_CALUDE_a_range_l2747_274742

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_decreasing : ∀ x y, x < y → f x > f y
axiom f_domain : ∀ x, f x ≠ 0 → -7 < x ∧ x < 7
axiom f_condition : ∀ a, f (1 - a) + f (2*a - 5) < 0

-- Theorem statement
theorem a_range : 
  ∃ a₁ a₂, a₁ = 4 ∧ a₂ = 6 ∧ 
  (∀ a, (f (1 - a) + f (2*a - 5) < 0) → a₁ < a ∧ a < a₂) :=
sorry

end NUMINAMATH_CALUDE_a_range_l2747_274742


namespace NUMINAMATH_CALUDE_base_b_problem_l2747_274797

theorem base_b_problem (b : ℕ) : 
  (6 * b^2 + 5 * b + 5 = (2 * b + 5)^2) → b = 9 := by
  sorry

end NUMINAMATH_CALUDE_base_b_problem_l2747_274797


namespace NUMINAMATH_CALUDE_train_length_l2747_274751

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 21 → ∃ length : ℝ, abs (length - 350.07) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2747_274751


namespace NUMINAMATH_CALUDE_farmer_wheat_harvest_l2747_274713

/-- The farmer's wheat harvest problem -/
theorem farmer_wheat_harvest (estimated : ℕ) (actual : ℕ) 
  (h1 : estimated = 48097) 
  (h2 : actual = 48781) : 
  actual - estimated = 684 := by
  sorry

end NUMINAMATH_CALUDE_farmer_wheat_harvest_l2747_274713


namespace NUMINAMATH_CALUDE_gravel_path_rate_l2747_274780

/-- Given a rectangular plot with an inner gravel path, calculate the rate per square meter for gravelling. -/
theorem gravel_path_rate (length width path_width total_cost : ℝ) 
  (h1 : length = 100)
  (h2 : width = 70)
  (h3 : path_width = 2.5)
  (h4 : total_cost = 742.5) : 
  total_cost / ((length * width) - ((length - 2 * path_width) * (width - 2 * path_width))) = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_gravel_path_rate_l2747_274780


namespace NUMINAMATH_CALUDE_add_1873_minutes_to_noon_l2747_274721

def minutes_in_hour : ℕ := 60
def hours_in_day : ℕ := 24

def add_minutes (start_hour : ℕ) (start_minute : ℕ) (minutes_to_add : ℕ) : (ℕ × ℕ) :=
  let total_minutes := start_hour * minutes_in_hour + start_minute + minutes_to_add
  let final_hour := (total_minutes / minutes_in_hour) % hours_in_day
  let final_minute := total_minutes % minutes_in_hour
  (final_hour, final_minute)

theorem add_1873_minutes_to_noon :
  add_minutes 12 0 1873 = (19, 13) :=
sorry

end NUMINAMATH_CALUDE_add_1873_minutes_to_noon_l2747_274721


namespace NUMINAMATH_CALUDE_trig_expression_equals_sqrt_three_l2747_274784

/-- Proves that the given trigonometric expression evaluates to √3 --/
theorem trig_expression_equals_sqrt_three :
  (Real.cos (350 * π / 180) - 2 * Real.sin (160 * π / 180)) / Real.sin (-190 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_sqrt_three_l2747_274784


namespace NUMINAMATH_CALUDE_polygon_sides_l2747_274705

theorem polygon_sides (n : ℕ) : (n - 2) * 180 = 1800 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2747_274705


namespace NUMINAMATH_CALUDE_sams_basketball_score_l2747_274735

theorem sams_basketball_score (total : ℕ) (friend_score : ℕ) (sam_score : ℕ) :
  total = 87 →
  friend_score = 12 →
  total = sam_score + friend_score →
  sam_score = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_sams_basketball_score_l2747_274735


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2747_274761

theorem regular_polygon_sides (internal_angle : ℝ) (h : internal_angle = 150) :
  (360 : ℝ) / (180 - internal_angle) = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2747_274761


namespace NUMINAMATH_CALUDE_unique_f_2_l2747_274710

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x + y) = f x + f y - x * y

theorem unique_f_2 (f : ℝ → ℝ) (hf : special_function f) : 
  f 2 = 3 ∧ ∀ y : ℝ, f 2 = y → y = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_f_2_l2747_274710


namespace NUMINAMATH_CALUDE_kia_vehicles_count_l2747_274756

def total_vehicles : ℕ := 400

def dodge_vehicles : ℕ := total_vehicles / 2

def hyundai_vehicles : ℕ := dodge_vehicles / 2

def kia_vehicles : ℕ := total_vehicles - dodge_vehicles - hyundai_vehicles

theorem kia_vehicles_count : kia_vehicles = 100 := by
  sorry

end NUMINAMATH_CALUDE_kia_vehicles_count_l2747_274756


namespace NUMINAMATH_CALUDE_special_square_PT_l2747_274728

/-- A square with side length 2 and special points T and U -/
structure SpecialSquare where
  -- Point P is at (0, 0), Q at (2, 0), R at (2, 2), and S at (0, 2)
  T : ℝ × ℝ  -- Point on PQ
  U : ℝ × ℝ  -- Point on SQ
  h_T_on_PQ : T.1 ∈ Set.Icc 0 2 ∧ T.2 = 0
  h_U_on_SQ : U.1 = 2 ∧ U.2 ∈ Set.Icc 0 2
  h_PT_eq_QU : T.1 = 2 - U.2  -- PT = QU
  h_fold : (2 - T.1)^2 + T.1^2 = 8  -- Condition for PR and SR to coincide with RQ when folded

theorem special_square_PT (s : SpecialSquare) : s.T.1 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_special_square_PT_l2747_274728


namespace NUMINAMATH_CALUDE_absolute_value_greater_than_x_l2747_274799

theorem absolute_value_greater_than_x (x : ℝ) : (x < 0) ↔ (abs x > x) := by sorry

end NUMINAMATH_CALUDE_absolute_value_greater_than_x_l2747_274799


namespace NUMINAMATH_CALUDE_expression_value_l2747_274791

theorem expression_value (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2747_274791


namespace NUMINAMATH_CALUDE_cylinder_different_views_l2747_274707

/-- Represents a geometric body --/
inductive GeometricBody
  | Sphere
  | TriangularPyramid
  | Cube
  | Cylinder

/-- Represents the dimensions of a view --/
structure ViewDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Returns true if all three views have the same dimensions --/
def sameViewDimensions (top front left : ViewDimensions) : Prop :=
  top.length = front.length ∧
  front.height = left.height ∧
  left.width = top.width

/-- Returns the three orthogonal views of a geometric body --/
def getViews (body : GeometricBody) : (ViewDimensions × ViewDimensions × ViewDimensions) :=
  sorry

theorem cylinder_different_views :
  ∀ (body : GeometricBody),
    (∃ (top front left : ViewDimensions),
      getViews body = (top, front, left) ∧
      ¬(sameViewDimensions top front left)) ↔
    body = GeometricBody.Cylinder :=
  sorry

end NUMINAMATH_CALUDE_cylinder_different_views_l2747_274707


namespace NUMINAMATH_CALUDE_binomial_sum_36_implies_n_8_l2747_274720

theorem binomial_sum_36_implies_n_8 (n : ℕ+) :
  (Nat.choose n 1 + Nat.choose n 2 = 36) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_36_implies_n_8_l2747_274720


namespace NUMINAMATH_CALUDE_eighth_term_value_l2747_274709

/-- An arithmetic sequence with 30 terms, first term 3, and last term 87 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  let d := (87 - 3) / (30 - 1)
  3 + (n - 1) * d

/-- The 8th term of the arithmetic sequence -/
def eighth_term : ℚ := arithmetic_sequence 8

theorem eighth_term_value : eighth_term = 675 / 29 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l2747_274709


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l2747_274771

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define external tangency
def externally_tangent (x y : ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧
    ((x + 3)^2 + y^2 = (r + 1)^2) ∧
    ((x - 3)^2 + y^2 = (r + 3)^2)

-- Define the trajectory of the center of M
def trajectory (x y : ℝ) : Prop :=
  x < 0 ∧ x^2 - y^2/8 = 1

-- Theorem statement
theorem moving_circle_trajectory :
  ∀ x y : ℝ, externally_tangent x y → trajectory x y :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l2747_274771


namespace NUMINAMATH_CALUDE_total_discount_is_65_percent_l2747_274787

/-- Represents the discount percentage as a real number between 0 and 1 -/
def Discount := { d : ℝ // 0 ≤ d ∧ d ≤ 1 }

/-- The half-price sale discount -/
def half_price_discount : Discount := ⟨0.5, by norm_num⟩

/-- The additional coupon discount -/
def coupon_discount : Discount := ⟨0.3, by norm_num⟩

/-- Calculates the final price after applying two successive discounts -/
def apply_discounts (d1 d2 : Discount) : ℝ := (1 - d1.val) * (1 - d2.val)

/-- The theorem to be proved -/
theorem total_discount_is_65_percent :
  apply_discounts half_price_discount coupon_discount = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_total_discount_is_65_percent_l2747_274787


namespace NUMINAMATH_CALUDE_triangle_properties_l2747_274724

-- Define the points A and B
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 3)

-- Define the properties of triangle ABC
def is_isosceles (A B C : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

def is_perpendicular (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0

-- Define the possible coordinates of point C
def C1 : ℝ × ℝ := (3, -1)
def C2 : ℝ × ℝ := (-3, 7)

-- Define the equations of the median lines
def median_eq1 (x y : ℝ) : Prop := 7 * x - y + 3 = 0
def median_eq2 (x y : ℝ) : Prop := x + 7 * y - 21 = 0

-- Theorem statement
theorem triangle_properties :
  (is_isosceles A B C1 ∧ is_perpendicular A B C1 ∧
   median_eq1 ((A.1 + C1.1) / 2) ((A.2 + C1.2) / 2)) ∨
  (is_isosceles A B C2 ∧ is_perpendicular A B C2 ∧
   median_eq2 ((A.1 + C2.1) / 2) ((A.2 + C2.2) / 2)) := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2747_274724


namespace NUMINAMATH_CALUDE_cantor_set_segments_l2747_274746

/-- The number of segments after n iterations of the process -/
def num_segments (n : ℕ) : ℕ := 2^n

/-- The length of each segment after n iterations of the process -/
def segment_length (n : ℕ) : ℚ := (1 : ℚ) / 3^n

theorem cantor_set_segments :
  num_segments 16 = 2^16 ∧ segment_length 16 = (1 : ℚ) / 3^16 := by
  sorry

#eval num_segments 16  -- To check the result

end NUMINAMATH_CALUDE_cantor_set_segments_l2747_274746


namespace NUMINAMATH_CALUDE_train_crossing_time_l2747_274782

/-- The time taken for two trains to cross each other -/
theorem train_crossing_time (length1 length2 speed1 speed2 : ℝ) : 
  length1 = 180 ∧ 
  length2 = 360 ∧ 
  speed1 = 60 * (1000 / 3600) ∧ 
  speed2 = 30 * (1000 / 3600) →
  (length1 + length2) / (speed1 + speed2) = 21.6 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2747_274782


namespace NUMINAMATH_CALUDE_ratio_first_to_last_l2747_274778

/-- An arithmetic sequence with five terms -/
structure ArithmeticSequence :=
  (a x c d b : ℚ)
  (is_arithmetic : ∃ (diff : ℚ), x = a + diff ∧ c = x + diff ∧ d = c + diff ∧ b = d + diff)
  (fourth_term : d = 3 * x)
  (fifth_term : b = 4 * x)

/-- The ratio of the first term to the last term is -1/4 -/
theorem ratio_first_to_last (seq : ArithmeticSequence) : seq.a / seq.b = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_first_to_last_l2747_274778


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2747_274790

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  1 / x + 2 / y ≥ 8 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ 1 / x + 2 / y = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2747_274790


namespace NUMINAMATH_CALUDE_equation_has_two_solutions_l2747_274706

-- Define the equation
def equation (x : ℝ) : Prop := Real.sqrt (9 - x) = x * Real.sqrt (9 - x)

-- Theorem statement
theorem equation_has_two_solutions :
  ∃ (a b : ℝ), a ≠ b ∧ equation a ∧ equation b ∧ 
  ∀ (x : ℝ), equation x → (x = a ∨ x = b) :=
sorry

end NUMINAMATH_CALUDE_equation_has_two_solutions_l2747_274706


namespace NUMINAMATH_CALUDE_power_equation_solution_l2747_274748

theorem power_equation_solution : ∃ y : ℝ, (12 : ℝ) ^ y * 6 ^ 3 / 432 = 72 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2747_274748


namespace NUMINAMATH_CALUDE_negative_five_meters_decrease_l2747_274702

-- Define a type for distance changes
inductive DistanceChange
| Increase (amount : ℤ)
| Decrease (amount : ℤ)

-- Define a function to interpret integers as distance changes
def interpretDistance (d : ℤ) : DistanceChange :=
  if d > 0 then DistanceChange.Increase d
  else DistanceChange.Decrease (-d)

-- Theorem statement
theorem negative_five_meters_decrease :
  interpretDistance (-5) = DistanceChange.Decrease 5 :=
by sorry

end NUMINAMATH_CALUDE_negative_five_meters_decrease_l2747_274702


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_y_axis_l2747_274785

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line parallel to y-axis passing through a given point
def LineParallelToYAxis (p : Point2D) := {q : Point2D | q.x = p.x}

theorem line_through_point_parallel_to_y_axis 
  (A : Point2D) 
  (h : A.x = -3 ∧ A.y = 1) 
  (P : Point2D) 
  (h_on_line : P ∈ LineParallelToYAxis A) : 
  P.x = -3 := by
sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_y_axis_l2747_274785


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_l2747_274750

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallelToPlane : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane_and_line (a b : Line) (α : Plane) :
  (perpendicularToPlane a α ∧ perpendicular a b) → parallelToPlane b α := by
  sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_l2747_274750


namespace NUMINAMATH_CALUDE_hyperbola_s_squared_l2747_274737

/-- A hyperbola centered at the origin passing through specific points -/
structure Hyperbola where
  -- The hyperbola passes through (5, -3)
  point1 : (5 : ℝ)^2 - (-3 : ℝ)^2 * a = b
  -- The hyperbola passes through (3, 0)
  point2 : (3 : ℝ)^2 = b
  -- The hyperbola passes through (s, -1)
  point3 : s^2 - (-1 : ℝ)^2 * a = b
  -- Ensure a and b are positive
  a_pos : a > 0
  b_pos : b > 0
  -- s is a real number
  s : ℝ

/-- The theorem stating the value of s^2 for the given hyperbola -/
theorem hyperbola_s_squared (h : Hyperbola) : h.s^2 = 873/81 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_s_squared_l2747_274737


namespace NUMINAMATH_CALUDE_smaller_rectangle_area_l2747_274767

theorem smaller_rectangle_area (length width : ℝ) (h1 : length = 40) (h2 : width = 20) :
  (length / 2) * (width / 2) = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_smaller_rectangle_area_l2747_274767


namespace NUMINAMATH_CALUDE_not_power_of_two_l2747_274758

theorem not_power_of_two (a b : ℕ+) : ¬ ∃ k : ℕ, (36 * a + b) * (a + 36 * b) = 2^k := by
  sorry

end NUMINAMATH_CALUDE_not_power_of_two_l2747_274758


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2747_274776

theorem geometric_sequence_sum (u v : ℝ) : 
  (∃ (a r : ℝ), a ≠ 0 ∧ r ≠ 0 ∧
    u = a * r^3 ∧
    v = a * r^4 ∧
    4 = a * r^5 ∧
    1 = a * r^6) →
  u + v = 80 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2747_274776


namespace NUMINAMATH_CALUDE_problem_statement_l2747_274738

theorem problem_statement (a b : ℝ) : 
  |a + b - 1| + Real.sqrt (2 * a + b - 2) = 0 → (b - a)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2747_274738


namespace NUMINAMATH_CALUDE_zigzag_angle_theorem_l2747_274781

/-- In a rectangle with a zigzag line, given specific angles, prove that ∠CDE is 11° --/
theorem zigzag_angle_theorem (ABC BCD DEF EFG : ℝ) (h1 : ABC = 10) (h2 : BCD = 14) 
  (h3 : DEF = 26) (h4 : EFG = 33) : ∃ (CDE : ℝ), CDE = 11 := by
  sorry

end NUMINAMATH_CALUDE_zigzag_angle_theorem_l2747_274781


namespace NUMINAMATH_CALUDE_stratified_sampling_under_35_l2747_274745

/-- Calculates the number of people to be drawn from a stratum in stratified sampling -/
def stratifiedSampleSize (totalPopulation : ℕ) (stratumSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (stratumSize * sampleSize) / totalPopulation

/-- The problem statement -/
theorem stratified_sampling_under_35 :
  let totalPopulation : ℕ := 500
  let under35 : ℕ := 125
  let between35and49 : ℕ := 280
  let over50 : ℕ := 95
  let sampleSize : ℕ := 100
  stratifiedSampleSize totalPopulation under35 sampleSize = 25 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_under_35_l2747_274745


namespace NUMINAMATH_CALUDE_square_even_implies_even_l2747_274726

theorem square_even_implies_even (a : ℤ) (h : Even (a ^ 2)) : Even a := by
  sorry

end NUMINAMATH_CALUDE_square_even_implies_even_l2747_274726


namespace NUMINAMATH_CALUDE_isosceles_triangle_altitude_midpoint_l2747_274732

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Check if a triangle is isosceles with AB = AC -/
def isIsosceles (t : Triangle) : Prop :=
  (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 = (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2

/-- Check if a point is on the altitude from A to BC -/
def isOnAltitude (t : Triangle) (d : Point) : Prop :=
  (t.B.y - t.C.y) * (d.x - t.A.x) = (t.C.x - t.B.x) * (d.y - t.A.y)

theorem isosceles_triangle_altitude_midpoint (t : Triangle) (d : Point) :
  t.A = Point.mk 5 7 →
  t.B = Point.mk (-1) 3 →
  d = Point.mk 1 5 →
  isIsosceles t →
  isOnAltitude t d →
  t.C = Point.mk 3 7 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_altitude_midpoint_l2747_274732


namespace NUMINAMATH_CALUDE_inequality_solution_l2747_274736

def solution_set (a : ℝ) : Set ℝ :=
  if a < 1 then {x | x < a ∨ x > 1}
  else if a = 1 then {x | x ≠ 1}
  else {x | x < 1 ∨ x > a}

theorem inequality_solution (a : ℝ) :
  {x : ℝ | x^2 - (a + 1) * x + a > 0} = solution_set a :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2747_274736


namespace NUMINAMATH_CALUDE_polynomial_multiple_divisible_by_three_l2747_274708

theorem polynomial_multiple_divisible_by_three 
  {R : Type*} [CommRing R] [Nontrivial R] :
  ∀ (P : Polynomial R), P ≠ 0 → 
  ∃ (Q : Polynomial R), Q ≠ 0 ∧ 
  ∀ (i : ℕ), (P * Q).coeff i ≠ 0 → i % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiple_divisible_by_three_l2747_274708


namespace NUMINAMATH_CALUDE_f_properties_l2747_274777

noncomputable def f (x : ℝ) := 4 * (Real.cos x)^2 + 4 * Real.sqrt 3 * Real.sin x * Real.cos x - 2

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ is_periodic f T ∧ ∀ T' > 0, is_periodic f T' → T ≤ T'

def has_max_at (f : ℝ → ℝ) (M : ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = M ∧ ∀ x, f x ≤ M

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem f_properties :
  (is_smallest_positive_period f Real.pi) ∧
  (∀ k : ℤ, has_max_at f 4 (k * Real.pi + Real.pi / 6)) ∧
  (∀ k : ℤ, is_increasing_on f (-Real.pi / 3 + k * Real.pi) (Real.pi / 6 + k * Real.pi)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2747_274777


namespace NUMINAMATH_CALUDE_max_value_of_f_l2747_274798

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc (-1 : ℝ) 1 ∧
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f x ≤ f c) ∧
  f c = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2747_274798


namespace NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_1540_l2747_274729

theorem sum_largest_smallest_prime_factors_1540 : ∃ (p q : Nat), 
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  p ∣ 1540 ∧ 
  q ∣ 1540 ∧ 
  (∀ r : Nat, Nat.Prime r → r ∣ 1540 → p ≤ r ∧ r ≤ q) ∧ 
  p + q = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_1540_l2747_274729


namespace NUMINAMATH_CALUDE_graph_quadrants_l2747_274788

/-- Given a > 1 and b < -1, the graph of f(x) = a^x + b intersects Quadrants I, III, and IV, but not Quadrant II -/
theorem graph_quadrants (a b : ℝ) (ha : a > 1) (hb : b < -1) :
  let f : ℝ → ℝ := λ x ↦ a^x + b
  (∃ x y, x > 0 ∧ y > 0 ∧ f x = y) ∧  -- Quadrant I
  (∃ x y, x < 0 ∧ y < 0 ∧ f x = y) ∧  -- Quadrant III
  (∃ x y, x > 0 ∧ y < 0 ∧ f x = y) ∧  -- Quadrant IV
  (∀ x y, ¬(x < 0 ∧ y > 0 ∧ f x = y))  -- Not in Quadrant II
  := by sorry

end NUMINAMATH_CALUDE_graph_quadrants_l2747_274788


namespace NUMINAMATH_CALUDE_cubic_equation_three_distinct_roots_l2747_274733

theorem cubic_equation_three_distinct_roots (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 12*x + a = 0 ∧
    y^3 - 12*y + a = 0 ∧
    z^3 - 12*z + a = 0) ↔
  -16 < a ∧ a < 16 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_three_distinct_roots_l2747_274733


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2747_274744

theorem trigonometric_identity (x : ℝ) : 
  8.435 * (Real.sin (3 * x))^10 + (Real.cos (3 * x))^10 = 
  4 * ((Real.sin (3 * x))^6 + (Real.cos (3 * x))^6) / 
  (4 * (Real.cos (6 * x))^2 + (Real.sin (6 * x))^2) ↔ 
  ∃ k : ℤ, x = k * Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2747_274744


namespace NUMINAMATH_CALUDE_triangle_condition_line_through_intersection_l2747_274770

-- Define the lines
def l1 (x y : ℝ) : Prop := x + y - 4 = 0
def l2 (x y : ℝ) : Prop := x - y + 2 = 0
def l3 (a x y : ℝ) : Prop := a * x - y + 1 - 4 * a = 0

-- Define point M
def M : ℝ × ℝ := (-1, 2)

-- Theorem for the range of a
theorem triangle_condition (a : ℝ) :
  (∃ x y z : ℝ, l1 x y ∧ l2 y z ∧ l3 a z x) ↔ 
  (a ≠ -2/3 ∧ a ≠ 1 ∧ a ≠ -1) :=
sorry

-- Theorem for the equation of line l
theorem line_through_intersection (x y : ℝ) :
  (∃ p q : ℝ, l1 p q ∧ l2 p q) ∧ 
  (abs (3*x + 4*y - 15) / Real.sqrt (3^2 + 4^2) = 2) ↔
  3*x + 4*y - 15 = 0 :=
sorry

end NUMINAMATH_CALUDE_triangle_condition_line_through_intersection_l2747_274770


namespace NUMINAMATH_CALUDE_company_kw_price_percentage_l2747_274792

/-- The price of Company KW as a percentage of the combined assets of Companies A and B -/
theorem company_kw_price_percentage (P A B : ℝ) 
  (h1 : P = 1.30 * A) 
  (h2 : P = 2.00 * B) : 
  ∃ (ε : ℝ), abs (P / (A + B) - 0.7879) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_company_kw_price_percentage_l2747_274792


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2747_274769

/-- Given a boat that travels 8 km along a stream and 2 km against the stream in one hour,
    prove that its speed in still water is 5 km/hr. -/
theorem boat_speed_in_still_water :
  ∀ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = 8 →
    boat_speed - stream_speed = 2 →
    boat_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2747_274769


namespace NUMINAMATH_CALUDE_alison_money_l2747_274766

def money_problem (kent_original brittany brooke kent alison : ℚ) : Prop :=
  let kent_after_lending := kent_original - 200
  alison = brittany / 2 ∧
  brittany = 4 * brooke ∧
  brooke = 2 * kent ∧
  kent = kent_after_lending ∧
  kent_original = 1000

theorem alison_money :
  ∀ kent_original brittany brooke kent alison,
    money_problem kent_original brittany brooke kent alison →
    alison = 3200 := by
  sorry

end NUMINAMATH_CALUDE_alison_money_l2747_274766


namespace NUMINAMATH_CALUDE_lucas_age_probability_l2747_274772

def coin_sides : Finset ℕ := {5, 15}
def die_sides : Finset ℕ := {1, 2, 3, 4, 5, 6}

def sum_probability (target : ℕ) : ℚ :=
  (coin_sides.filter (λ c => ∃ d ∈ die_sides, c + d = target)).card /
    (coin_sides.card * die_sides.card)

theorem lucas_age_probability :
  sum_probability 15 = 0 := by sorry

end NUMINAMATH_CALUDE_lucas_age_probability_l2747_274772


namespace NUMINAMATH_CALUDE_weight_after_first_week_l2747_274762

/-- Given Jessie's initial weight and weight loss in the first week, 
    calculate her weight after the first week of jogging. -/
theorem weight_after_first_week 
  (initial_weight : ℕ) 
  (weight_loss_first_week : ℕ) 
  (h1 : initial_weight = 92) 
  (h2 : weight_loss_first_week = 56) : 
  initial_weight - weight_loss_first_week = 36 := by
  sorry

end NUMINAMATH_CALUDE_weight_after_first_week_l2747_274762


namespace NUMINAMATH_CALUDE_caterer_order_total_price_l2747_274717

theorem caterer_order_total_price :
  let ice_cream_bars := 125
  let sundaes := 125
  let ice_cream_bar_price := 0.60
  let sundae_price := 1.2
  let total_price := ice_cream_bars * ice_cream_bar_price + sundaes * sundae_price
  total_price = 225 := by sorry

end NUMINAMATH_CALUDE_caterer_order_total_price_l2747_274717


namespace NUMINAMATH_CALUDE_correct_weight_calculation_l2747_274773

theorem correct_weight_calculation (class_size : ℕ) 
  (incorrect_avg : ℚ) (misread_weight : ℚ) (correct_avg : ℚ) :
  class_size = 20 →
  incorrect_avg = 58.4 →
  misread_weight = 56 →
  correct_avg = 58.6 →
  (class_size : ℚ) * correct_avg - (class_size : ℚ) * incorrect_avg + misread_weight = 60 :=
by sorry

end NUMINAMATH_CALUDE_correct_weight_calculation_l2747_274773


namespace NUMINAMATH_CALUDE_quadratic_trinomial_condition_l2747_274700

/-- Given a constant m, if x^|m| + (m-2)x - 10 is a quadratic trinomial, then m = -2 -/
theorem quadratic_trinomial_condition (m : ℝ) : 
  (∃ (a b c : ℝ), ∀ x, x^(|m|) + (m-2)*x - 10 = a*x^2 + b*x + c) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_condition_l2747_274700


namespace NUMINAMATH_CALUDE_marsh_birds_total_l2747_274703

theorem marsh_birds_total (initial_geese ducks swans herons : ℕ) 
  (h1 : initial_geese = 58)
  (h2 : ducks = 37)
  (h3 : swans = 15)
  (h4 : herons = 22) :
  initial_geese * 2 + ducks + swans + herons = 190 := by
  sorry

end NUMINAMATH_CALUDE_marsh_birds_total_l2747_274703


namespace NUMINAMATH_CALUDE_nickel_probability_l2747_274723

/-- Represents the types of coins in the box -/
inductive Coin
  | Dime
  | Nickel
  | Penny

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- The total value of each coin type in the box in cents -/
def total_value : Coin → ℕ
  | Coin.Dime => 500
  | Coin.Nickel => 250
  | Coin.Penny => 100

/-- The number of coins of each type in the box -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the box -/
def total_coins : ℕ := coin_count Coin.Dime + coin_count Coin.Nickel + coin_count Coin.Penny

/-- The probability of randomly selecting a nickel from the box -/
theorem nickel_probability : 
  (coin_count Coin.Nickel : ℚ) / total_coins = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_nickel_probability_l2747_274723


namespace NUMINAMATH_CALUDE_smallest_prime_scalene_perimeter_l2747_274722

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that checks if three numbers form a scalene triangle -/
def isScaleneTriangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ a + c > b ∧ b + c > a

/-- The theorem stating the smallest possible perimeter of a scalene triangle with prime side lengths -/
theorem smallest_prime_scalene_perimeter :
  ∀ a b c : ℕ,
    isPrime a → isPrime b → isPrime c →
    isScaleneTriangle a b c →
    a + b + c ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_scalene_perimeter_l2747_274722


namespace NUMINAMATH_CALUDE_work_completion_time_l2747_274718

theorem work_completion_time (a b c : ℝ) 
  (h1 : a + b + c = 1 / 4)  -- a, b, and c together finish in 4 days
  (h2 : b = 1 / 18)         -- b alone finishes in 18 days
  (h3 : c = 1 / 9)          -- c alone finishes in 9 days
  : a = 1 / 12 :=           -- a alone finishes in 12 days
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2747_274718


namespace NUMINAMATH_CALUDE_isabel_afternoon_runs_l2747_274755

/-- Calculates the number of afternoon runs given circuit length, morning runs, and total weekly distance -/
def afternoon_runs (circuit_length : ℕ) (morning_runs : ℕ) (total_weekly_distance : ℕ) : ℕ :=
  (total_weekly_distance - 7 * morning_runs * circuit_length) / circuit_length

/-- Proves that Isabel runs the circuit 21 times in the afternoon during a week -/
theorem isabel_afternoon_runs : 
  afternoon_runs 365 7 25550 = 21 := by
  sorry

end NUMINAMATH_CALUDE_isabel_afternoon_runs_l2747_274755


namespace NUMINAMATH_CALUDE_calculation_proof_l2747_274795

theorem calculation_proof : 
  41 * ((2 + 2/7) - (3 + 3/5)) / ((3 + 1/5) + (2 + 1/4)) = -10 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2747_274795


namespace NUMINAMATH_CALUDE_binary_110101_equals_53_l2747_274796

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enum b).foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110101_equals_53 :
  binary_to_decimal [true, false, true, false, true, true] = 53 := by
  sorry

end NUMINAMATH_CALUDE_binary_110101_equals_53_l2747_274796


namespace NUMINAMATH_CALUDE_trebled_result_of_doubled_plus_nine_l2747_274712

theorem trebled_result_of_doubled_plus_nine (x : ℕ) : x = 4 → 3 * (2 * x + 9) = 51 := by
  sorry

end NUMINAMATH_CALUDE_trebled_result_of_doubled_plus_nine_l2747_274712


namespace NUMINAMATH_CALUDE_rain_on_tuesday_l2747_274739

theorem rain_on_tuesday (rain_monday : ℝ) (no_rain : ℝ) (rain_both : ℝ)
  (h1 : rain_monday = 0.62)
  (h2 : no_rain = 0.28)
  (h3 : rain_both = 0.44) :
  rain_monday + (1 - no_rain) - rain_both = 0.54 := by
sorry

end NUMINAMATH_CALUDE_rain_on_tuesday_l2747_274739


namespace NUMINAMATH_CALUDE_quadratic_roots_equal_irrational_l2747_274740

theorem quadratic_roots_equal_irrational (d : ℝ) :
  let a : ℝ := 3
  let b : ℝ := -4 * Real.pi
  let c : ℝ := d
  let discriminant := b^2 - 4*a*c
  discriminant = 16 →
  ∃ (x : ℝ), (a*x^2 + b*x + c = 0 ∧ 
              ∀ (y : ℝ), a*y^2 + b*y + c = 0 → y = x) ∧
             (¬ ∃ (p q : ℤ), x = p / q) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_equal_irrational_l2747_274740


namespace NUMINAMATH_CALUDE_denominator_one_root_l2747_274725

theorem denominator_one_root (k : ℝ) : 
  (∃! x : ℝ, -2 * x^2 + 8 * x + k = 0) ↔ k = -8 := by sorry

end NUMINAMATH_CALUDE_denominator_one_root_l2747_274725


namespace NUMINAMATH_CALUDE_max_fleas_l2747_274749

/-- Represents a flea's direction of movement --/
inductive Direction
| Up
| Down
| Left
| Right

/-- Represents a position on the board --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents a flea on the board --/
structure Flea :=
  (position : Position)
  (direction : Direction)

/-- Represents the board state --/
def BoardState := List Flea

/-- The size of the board --/
def boardSize : Nat := 10

/-- The duration of observation in minutes --/
def observationTime : Nat := 60

/-- Function to update a flea's position based on its direction --/
def updateFleaPosition (f : Flea) : Flea :=
  sorry

/-- Function to check if two fleas occupy the same position --/
def fleaCollision (f1 f2 : Flea) : Bool :=
  sorry

/-- Function to simulate one minute of flea movement --/
def simulateMinute (state : BoardState) : BoardState :=
  sorry

/-- Function to simulate the entire observation period --/
def simulateObservation (initialState : BoardState) : Bool :=
  sorry

/-- Theorem stating the maximum number of fleas --/
theorem max_fleas : 
  ∀ (initialState : BoardState),
    simulateObservation initialState → List.length initialState ≤ 40 :=
  sorry

end NUMINAMATH_CALUDE_max_fleas_l2747_274749


namespace NUMINAMATH_CALUDE_bisector_triangle_area_l2747_274789

/-- Given a tetrahedron ABCD with face areas P (ABC) and Q (ADC), and dihedral angle α between these faces,
    the area S of the triangle formed by the plane bisecting α is (2PQ cos(α/2)) / (P + Q) -/
theorem bisector_triangle_area (P Q α : ℝ) (hP : P > 0) (hQ : Q > 0) (hα : 0 < α ∧ α < π) :
  ∃ S : ℝ, S = (2 * P * Q * Real.cos (α / 2)) / (P + Q) ∧ S > 0 :=
sorry

end NUMINAMATH_CALUDE_bisector_triangle_area_l2747_274789


namespace NUMINAMATH_CALUDE_sports_club_probability_l2747_274764

/-- The probability of selecting two girls when randomly choosing two members from a group. -/
def probability_two_girls (total : ℕ) (girls : ℕ) : ℚ :=
  (girls.choose 2 : ℚ) / (total.choose 2 : ℚ)

/-- The theorem stating the probability of selecting two girls from the sports club. -/
theorem sports_club_probability :
  let total := 15
  let girls := 8
  probability_two_girls total girls = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_probability_l2747_274764


namespace NUMINAMATH_CALUDE_shifted_parabola_equation_l2747_274716

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := 2 * p.a * h + p.b
  , c := p.a * h^2 + p.b * h + p.c + v }

theorem shifted_parabola_equation (p : Parabola) :
  let original := Parabola.mk (-2) 0 0
  let shifted := shift_parabola (shift_parabola original 1 0) 0 (-3)
  shifted = Parabola.mk (-2) (-4) (-5) := by sorry

end NUMINAMATH_CALUDE_shifted_parabola_equation_l2747_274716


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2747_274763

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    if 2a_7 - a_8 = 5, then S_11 = 55 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
  (∀ n, S n = n * (a 1 + a n) / 2) →    -- sum formula
  (2 * a 7 - a 8 = 5) →                 -- given condition
  S 11 = 55 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2747_274763


namespace NUMINAMATH_CALUDE_last_triangle_perimeter_l2747_274757

/-- Represents a triangle with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Generates the next triangle in the sequence based on the incircle tangent points -/
def nextTriangle (t : Triangle) : Triangle :=
  sorry

/-- Checks if a triangle is valid (satisfies the triangle inequality) -/
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- The sequence of triangles starting from the initial triangle -/
def triangleSequence : ℕ → Triangle
  | 0 => { a := 1015, b := 1016, c := 1017 }
  | n + 1 => nextTriangle (triangleSequence n)

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Finds the index of the last valid triangle in the sequence -/
def lastValidTriangleIndex : ℕ :=
  sorry

theorem last_triangle_perimeter :
  perimeter (triangleSequence lastValidTriangleIndex) = 762 / 128 :=
sorry

end NUMINAMATH_CALUDE_last_triangle_perimeter_l2747_274757


namespace NUMINAMATH_CALUDE_total_weight_moved_l2747_274719

/-- Represents an exercise with weight, reps, and sets -/
structure Exercise where
  weight : Nat
  reps : Nat
  sets : Nat

/-- Calculates the total weight moved for a single exercise -/
def totalWeightForExercise (e : Exercise) : Nat :=
  e.weight * e.reps * e.sets

/-- John's workout routine -/
def workoutRoutine : List Exercise := [
  { weight := 15, reps := 10, sets := 3 },  -- Bench press
  { weight := 12, reps := 8,  sets := 4 },  -- Bicep curls
  { weight := 50, reps := 12, sets := 3 },  -- Squats
  { weight := 80, reps := 6,  sets := 2 }   -- Deadlift
]

/-- Theorem stating the total weight John moves during his workout -/
theorem total_weight_moved : 
  (workoutRoutine.map totalWeightForExercise).sum = 3594 := by
  sorry


end NUMINAMATH_CALUDE_total_weight_moved_l2747_274719


namespace NUMINAMATH_CALUDE_base_conversion_problem_l2747_274741

theorem base_conversion_problem (n C D : ℕ) : 
  n > 0 ∧ 
  C < 8 ∧ 
  D < 6 ∧ 
  n = 8 * C + D ∧ 
  n = 6 * D + C → 
  n = 43 := by sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l2747_274741


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_a_in_range_l2747_274759

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - (1/2) * (x - a)^2 + 4

theorem f_nonnegative_iff_a_in_range (a : ℝ) :
  (∀ x ≥ 0, f a x ≥ 0) ↔ a ∈ Set.Icc (Real.log 4 - 4) (Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_a_in_range_l2747_274759


namespace NUMINAMATH_CALUDE_solve_equation_l2747_274768

theorem solve_equation (m n : ℝ) : 
  |m - 2| + n^2 - 8*n + 16 = 0 → m = 2 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2747_274768


namespace NUMINAMATH_CALUDE_binary_subtraction_l2747_274774

def binary_to_decimal (b : ℕ) : ℕ := 
  if b = 0 then 0
  else if b % 10 = 1 then 1 + 2 * (binary_to_decimal (b / 10))
  else 2 * (binary_to_decimal (b / 10))

def binary_1111111111 : ℕ := 1111111111
def binary_11111 : ℕ := 11111

theorem binary_subtraction :
  binary_to_decimal binary_1111111111 - binary_to_decimal binary_11111 = 992 := by
  sorry

end NUMINAMATH_CALUDE_binary_subtraction_l2747_274774


namespace NUMINAMATH_CALUDE_three_digit_congruence_count_l2747_274786

theorem three_digit_congruence_count : 
  let count := Finset.filter (fun y => 100 ≤ y ∧ y ≤ 999 ∧ (4325 * y + 692) % 17 = 1403 % 17) (Finset.range 1000)
  ↑count.card = 53 := by sorry

end NUMINAMATH_CALUDE_three_digit_congruence_count_l2747_274786
