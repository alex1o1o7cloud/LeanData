import Mathlib

namespace NUMINAMATH_CALUDE_zoo_animals_l1679_167968

theorem zoo_animals (M P L : ℕ) : 
  (26 ≤ M + P + L ∧ M + P + L ≤ 32) →
  M + L > P →
  P + L = 2 * M →
  M + P > 3 * L →
  P < 2 * L →
  P = 12 := by
sorry

end NUMINAMATH_CALUDE_zoo_animals_l1679_167968


namespace NUMINAMATH_CALUDE_hike_distance_l1679_167965

/-- Represents a 5-day hike with given conditions -/
structure FiveDayHike where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ
  first_two_days : day1 + day2 = 28
  second_fourth_avg : (day2 + day4) / 2 = 15
  last_three_days : day3 + day4 + day5 = 42
  first_third_days : day1 + day3 = 30

/-- The total distance of the hike is 70 miles -/
theorem hike_distance (h : FiveDayHike) : h.day1 + h.day2 + h.day3 + h.day4 + h.day5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_hike_distance_l1679_167965


namespace NUMINAMATH_CALUDE_billy_tickets_l1679_167932

theorem billy_tickets (tickets_won : ℕ) (tickets_left : ℕ) (difference : ℕ) : 
  tickets_left = 32 →
  difference = 16 →
  tickets_won - tickets_left = difference →
  tickets_won = 48 := by
sorry

end NUMINAMATH_CALUDE_billy_tickets_l1679_167932


namespace NUMINAMATH_CALUDE_necklace_beads_l1679_167953

theorem necklace_beads (total : ℕ) (blue : ℕ) (red : ℕ) (white : ℕ) (silver : ℕ) :
  total = 40 →
  red = 2 * blue →
  white = blue + red →
  silver = 10 →
  blue + red + white + silver = total →
  blue = 5 := by
sorry

end NUMINAMATH_CALUDE_necklace_beads_l1679_167953


namespace NUMINAMATH_CALUDE_no_solution_double_inequality_l1679_167951

theorem no_solution_double_inequality :
  ¬ ∃ y : ℝ, (3 * y^2 - 4 * y - 5 < (y + 1)^2) ∧ ((y + 1)^2 < 4 * y^2 - y - 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_double_inequality_l1679_167951


namespace NUMINAMATH_CALUDE_cross_in_square_l1679_167952

theorem cross_in_square (s : ℝ) : 
  (2 * (s/2)^2 + 2 * (s/4)^2 = 810) → s = 36 := by
  sorry

end NUMINAMATH_CALUDE_cross_in_square_l1679_167952


namespace NUMINAMATH_CALUDE_smallest_multiple_of_45_and_60_not_25_l1679_167904

theorem smallest_multiple_of_45_and_60_not_25 :
  ∃ n : ℕ, n > 0 ∧ 45 ∣ n ∧ 60 ∣ n ∧ ¬(25 ∣ n) ∧
  ∀ m : ℕ, m > 0 ∧ 45 ∣ m ∧ 60 ∣ m ∧ ¬(25 ∣ m) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_45_and_60_not_25_l1679_167904


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1679_167981

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 4*a^2 + 50*a - 7 = 0) →
  (b^3 - 4*b^2 + 50*b - 7 = 0) →
  (c^3 - 4*c^2 + 50*c - 7 = 0) →
  (a + b + c = 4) →
  (a*b + b*c + c*a = 50) →
  (a*b*c = 7) →
  (a + b + 1)^3 + (b + c + 1)^3 + (c + a + 1)^3 = 991 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1679_167981


namespace NUMINAMATH_CALUDE_area_of_ring_area_of_specific_ring_l1679_167935

/-- The area of a ring-shaped region formed by two concentric circles -/
theorem area_of_ring (r₁ r₂ : ℝ) (h : r₁ > r₂) :
  (π * r₁^2 - π * r₂^2) = π * (r₁^2 - r₂^2) :=
by sorry

/-- The area of a ring-shaped region formed by two concentric circles with radii 12 and 5 -/
theorem area_of_specific_ring :
  π * (12^2 - 5^2) = 119 * π :=
by sorry

end NUMINAMATH_CALUDE_area_of_ring_area_of_specific_ring_l1679_167935


namespace NUMINAMATH_CALUDE_javier_first_throw_l1679_167929

/-- Represents the distances of three javelin throws -/
structure JavelinThrows where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Conditions for Javier's javelin throws -/
def javierThrows (t : JavelinThrows) : Prop :=
  t.first = 2 * t.second ∧
  t.first = 1/2 * t.third ∧
  t.first + t.second + t.third = 1050

/-- Theorem stating that Javier's first throw was 300 meters -/
theorem javier_first_throw :
  ∀ t : JavelinThrows, javierThrows t → t.first = 300 := by
  sorry

end NUMINAMATH_CALUDE_javier_first_throw_l1679_167929


namespace NUMINAMATH_CALUDE_barge_power_increase_l1679_167959

/-- Given a barge pushed by tugboats in water, this theorem proves that
    doubling the force results in a power increase by a factor of 2√2,
    when water resistance is proportional to the square of speed. -/
theorem barge_power_increase
  (F : ℝ) -- Initial force
  (v : ℝ) -- Initial velocity
  (k : ℝ) -- Constant of proportionality for water resistance
  (h1 : F = k * v^2) -- Initial force equals water resistance
  (h2 : 2 * F = k * v_new^2) -- New force equals new water resistance
  : (2 * F * v_new) / (F * v) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_barge_power_increase_l1679_167959


namespace NUMINAMATH_CALUDE_sector_radius_l1679_167901

/-- Given a sector with a central angle of 90° and an arc length of 3π, its radius is 6. -/
theorem sector_radius (θ : Real) (l : Real) (r : Real) : 
  θ = 90 → l = 3 * Real.pi → l = (θ * Real.pi * r) / 180 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l1679_167901


namespace NUMINAMATH_CALUDE_power_calculation_l1679_167946

theorem power_calculation : 16^16 * 8^8 / 4^40 = 256 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l1679_167946


namespace NUMINAMATH_CALUDE_unknown_number_proof_l1679_167955

theorem unknown_number_proof (a b : ℝ) : 
  (a - 3 = b - a) →  -- arithmetic sequence condition
  ((a - 6) / 3 = b / (a - 6)) →  -- geometric sequence condition
  b = 27 := by
sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l1679_167955


namespace NUMINAMATH_CALUDE_ellipse_angle_ratio_l1679_167971

noncomputable section

variables (a b : ℝ) (x y : ℝ)

-- Define the ellipse
def is_on_ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define eccentricity
def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Define the angles α and β
def alpha (x y a : ℝ) : ℝ := Real.arctan (y / (x + a))
def beta (x y a : ℝ) : ℝ := Real.arctan (y / (x - a))

theorem ellipse_angle_ratio 
  (h1 : a > b) (h2 : b > 0)
  (h3 : is_on_ellipse x y a b)
  (h4 : eccentricity a b = Real.sqrt 3 / 2)
  (h5 : x ≠ a ∧ x ≠ -a) :
  (Real.cos (alpha x y a - beta x y a)) / 
  (Real.cos (alpha x y a + beta x y a)) = 3/5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_angle_ratio_l1679_167971


namespace NUMINAMATH_CALUDE_product_divisible_by_49_l1679_167984

theorem product_divisible_by_49 (a b : ℕ) (h : 7 ∣ (a^2 + b^2)) : 49 ∣ (a * b) := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_49_l1679_167984


namespace NUMINAMATH_CALUDE_ants_meet_at_66cm_l1679_167933

/-- Represents a point on the tile grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents a movement on the grid -/
inductive GridMove
  | Right
  | Up
  | Left
  | Down

/-- The path of an ant on the grid -/
def AntPath := List GridMove

/-- Calculate the distance traveled given a path -/
def pathDistance (path : AntPath) (tileWidth tileLength : ℕ) : ℕ :=
  path.foldl (fun acc move =>
    acc + match move with
      | GridMove.Right => tileLength
      | GridMove.Up => tileWidth
      | GridMove.Left => tileLength
      | GridMove.Down => tileWidth) 0

/-- Check if two paths meet at the same point -/
def pathsMeet (path1 path2 : AntPath) (start1 start2 : GridPoint) : Prop :=
  sorry

theorem ants_meet_at_66cm (tileWidth tileLength : ℕ) (startM startN : GridPoint) 
    (pathM pathN : AntPath) : 
  tileWidth = 4 →
  tileLength = 6 →
  startM = ⟨0, 0⟩ →
  startN = ⟨14, 12⟩ →
  pathsMeet pathM pathN startM startN →
  pathDistance pathM tileWidth tileLength = 66 ∧
  pathDistance pathN tileWidth tileLength = 66 :=
by
  sorry

#check ants_meet_at_66cm

end NUMINAMATH_CALUDE_ants_meet_at_66cm_l1679_167933


namespace NUMINAMATH_CALUDE_parabola_through_point_l1679_167911

/-- A parabola passing through the point (4, -2) has either the equation y² = x or x² = -8y -/
theorem parabola_through_point (P : ℝ × ℝ) (h : P = (4, -2)) :
  (∃ (x y : ℝ), y^2 = x ∧ P = (x, y)) ∨ (∃ (x y : ℝ), x^2 = -8*y ∧ P = (x, y)) :=
sorry

end NUMINAMATH_CALUDE_parabola_through_point_l1679_167911


namespace NUMINAMATH_CALUDE_intersection_points_count_l1679_167987

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

def opposite_signs_at_1_2 (f : ℝ → ℝ) : Prop := f 1 * f 2 < 0

-- Define the number of intersections with the x-axis
def num_intersections (f : ℝ → ℝ) : ℕ :=
  -- This is a placeholder definition
  -- In practice, this would be defined more rigorously
  2

-- State the theorem
theorem intersection_points_count
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_mono : monotone_increasing_pos f)
  (h_signs : opposite_signs_at_1_2 f) :
  num_intersections f = 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_count_l1679_167987


namespace NUMINAMATH_CALUDE_chameleon_theorem_l1679_167992

/-- Represents the resting period before catching the m-th fly -/
def resting_period (m : ℕ) : ℕ :=
  sorry

/-- Represents the total time before catching the m-th fly -/
def total_time (m : ℕ) : ℕ :=
  sorry

/-- Represents the number of flies caught after t minutes -/
def flies_caught (t : ℕ) : ℕ :=
  sorry

/-- The chameleon's resting and catching behavior -/
axiom resting_rule_1 : resting_period 1 = 1
axiom resting_rule_2 : ∀ m : ℕ, resting_period (2 * m) = resting_period m
axiom resting_rule_3 : ∀ m : ℕ, resting_period (2 * m + 1) = resting_period m + 1
axiom catch_instantly : ∀ m : ℕ, total_time (m + 1) = total_time m + resting_period (m + 1) + 1

theorem chameleon_theorem :
  (∃ m : ℕ, m = 510 ∧ resting_period (m + 1) = 9 ∧ ∀ k < m, resting_period (k + 1) < 9) ∧
  (total_time 98 = 312) ∧
  (flies_caught 1999 = 462) :=
sorry

end NUMINAMATH_CALUDE_chameleon_theorem_l1679_167992


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l1679_167920

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l1679_167920


namespace NUMINAMATH_CALUDE_graph_transform_properties_l1679_167986

/-- A graph in a 2D plane -/
structure Graph where
  -- We don't need to define the internal structure of the graph
  -- as we're only concerned with its properties under transformations

/-- Properties of a graph that may or may not change under transformations -/
structure GraphProperties where
  shape : Bool  -- True if shape is preserved
  size : Bool   -- True if size is preserved
  direction : Bool  -- True if direction is preserved

/-- Rotation of a graph -/
def rotate (g : Graph) : Graph :=
  sorry

/-- Translation of a graph -/
def translate (g : Graph) : Graph :=
  sorry

/-- Properties preserved under rotation and translation -/
def properties_after_transform (g : Graph) : GraphProperties :=
  sorry

theorem graph_transform_properties :
  ∀ g : Graph,
    let props := properties_after_transform g
    props.shape = true ∧ props.size = true ∧ props.direction = false :=
by sorry

end NUMINAMATH_CALUDE_graph_transform_properties_l1679_167986


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1679_167900

-- Define a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific function
def f (x : ℝ) : ℝ := -4 * x^2 + 5

-- Theorem statement
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1679_167900


namespace NUMINAMATH_CALUDE_task_selection_ways_l1679_167943

/-- The number of ways to select individuals for tasks with specific requirements -/
def select_for_tasks (total_people : ℕ) (task_a_people : ℕ) (task_b_people : ℕ) (task_c_people : ℕ) : ℕ :=
  Nat.choose total_people task_a_people *
  (Nat.choose (total_people - task_a_people) (task_b_people + task_c_people) * Nat.factorial (task_b_people + task_c_people))

/-- Theorem stating the number of ways to select 4 individuals from 10 for the given tasks -/
theorem task_selection_ways :
  select_for_tasks 10 2 1 1 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_task_selection_ways_l1679_167943


namespace NUMINAMATH_CALUDE_tims_lunch_cost_l1679_167924

/-- The total amount Tim spent on lunch, including taxes, surcharge, and tips -/
def total_lunch_cost (meal_cost : ℝ) (tip_rate state_tax_rate city_tax_rate surcharge_rate : ℝ) : ℝ :=
  let tip := meal_cost * tip_rate
  let state_tax := meal_cost * state_tax_rate
  let city_tax := meal_cost * city_tax_rate
  let subtotal := meal_cost + state_tax + city_tax
  let surcharge := subtotal * surcharge_rate
  meal_cost + tip + state_tax + city_tax + surcharge

/-- Theorem stating that Tim's total lunch cost is $78.43 -/
theorem tims_lunch_cost :
  total_lunch_cost 60.50 0.20 0.05 0.03 0.015 = 78.43 := by
  sorry


end NUMINAMATH_CALUDE_tims_lunch_cost_l1679_167924


namespace NUMINAMATH_CALUDE_trig_identity_l1679_167936

theorem trig_identity (θ a b : ℝ) (h : 0 < a) (h' : 0 < b) :
  (Real.sin θ)^6 / a^2 + (Real.cos θ)^6 / b^2 = 1 / (a + b) →
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 = 1 / (a + b)^5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1679_167936


namespace NUMINAMATH_CALUDE_investment_doubling_time_l1679_167944

/-- The minimum number of years required for an investment to at least double -/
theorem investment_doubling_time (A r : ℝ) (h1 : A > 0) (h2 : r > 0) :
  let t := Real.log 2 / Real.log (1 + r)
  ∀ s : ℝ, s ≥ t → A * (1 + r) ^ s ≥ 2 * A :=
by sorry

end NUMINAMATH_CALUDE_investment_doubling_time_l1679_167944


namespace NUMINAMATH_CALUDE_ellipse_tangent_inequality_l1679_167923

/-- Represents an ellipse with foci A and B, and semi-major and semi-minor axes a and b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_ge_b : a ≥ b

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Checks if a point is outside an ellipse -/
def is_outside (ε : Ellipse) (T : Point) : Prop := sorry

/-- Represents a tangent line from a point to an ellipse -/
def tangent_line (ε : Ellipse) (T : Point) : Type := sorry

/-- The length of a tangent line -/
def tangent_length (l : tangent_line ε T) : ℝ := sorry

theorem ellipse_tangent_inequality (ε : Ellipse) (T : Point) 
  (h_outside : is_outside ε T) 
  (TP TQ : tangent_line ε T) : 
  (tangent_length TP) / (tangent_length TQ) ≥ ε.b / ε.a := by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_inequality_l1679_167923


namespace NUMINAMATH_CALUDE_equator_scientific_notation_l1679_167947

/-- The circumference of the equator in meters -/
def equator_circumference : ℕ := 40210000

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Convert a natural number to scientific notation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem equator_scientific_notation :
  to_scientific_notation equator_circumference = ScientificNotation.mk 4.021 7 := by
  sorry

end NUMINAMATH_CALUDE_equator_scientific_notation_l1679_167947


namespace NUMINAMATH_CALUDE_x_cubed_minus_2x_plus_1_l1679_167967

theorem x_cubed_minus_2x_plus_1 (x : ℝ) (h : x^2 - x - 1 = 0) : x^3 - 2*x + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_cubed_minus_2x_plus_1_l1679_167967


namespace NUMINAMATH_CALUDE_smallest_root_property_l1679_167910

theorem smallest_root_property : ∃ a : ℝ, 
  (∀ x : ℝ, x^2 - 9*x - 10 = 0 → a ≤ x) ∧ 
  (a^2 - 9*a - 10 = 0) ∧
  (a^4 - 909*a = 910) := by
  sorry

end NUMINAMATH_CALUDE_smallest_root_property_l1679_167910


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l1679_167962

-- Define the probability of having a boy or a girl
def p_boy_or_girl : ℚ := 1 / 2

-- Define the number of children in the family
def num_children : ℕ := 4

-- Theorem statement
theorem prob_at_least_one_boy_one_girl :
  1 - (p_boy_or_girl ^ num_children + p_boy_or_girl ^ num_children) = 7 / 8 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l1679_167962


namespace NUMINAMATH_CALUDE_wendy_small_glasses_l1679_167928

/-- The number of small glasses polished by Wendy -/
def small_glasses : ℕ := 50

/-- The number of large glasses polished by Wendy -/
def large_glasses : ℕ := small_glasses + 10

/-- The total number of glasses polished by Wendy -/
def total_glasses : ℕ := 110

/-- Proof that Wendy polished 50 small glasses -/
theorem wendy_small_glasses :
  small_glasses = 50 ∧
  large_glasses = small_glasses + 10 ∧
  small_glasses + large_glasses = total_glasses :=
by sorry

end NUMINAMATH_CALUDE_wendy_small_glasses_l1679_167928


namespace NUMINAMATH_CALUDE_percent_relation_l1679_167914

theorem percent_relation (x y z : ℝ) 
  (h1 : x = 1.20 * y) 
  (h2 : y = 0.30 * z) : 
  x = 0.36 * z := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l1679_167914


namespace NUMINAMATH_CALUDE_sum_f_half_integers_l1679_167954

def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem sum_f_half_integers (f : ℝ → ℝ) 
  (h1 : is_even (λ x ↦ f (2*x + 2)))
  (h2 : is_odd (λ x ↦ f (x + 1)))
  (h3 : ∃ a b : ℝ, ∀ x ∈ Set.Icc 0 1, f x = a * x + b)
  (h4 : f 4 = 1) :
  (f (3/2) + f (5/2) + f (7/2)) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sum_f_half_integers_l1679_167954


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1679_167985

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) + 2
  f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1679_167985


namespace NUMINAMATH_CALUDE_symmetric_line_fixed_point_l1679_167993

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the symmetry relation
def symmetric_about (l1 l2 : Line) (p : ℝ × ℝ) : Prop :=
  ∀ (x y : ℝ), (y = l1.slope * (x - 4)) → 
  ∃ (x' y' : ℝ), (y' = l2.slope * x' + l2.intercept) ∧ 
  ((x + x') / 2 = p.1) ∧ ((y + y') / 2 = p.2)

-- Define when a line passes through a point
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

-- Theorem statement
theorem symmetric_line_fixed_point (k : ℝ) :
  ∀ (l2 : Line),
  symmetric_about (Line.mk k (-4*k)) l2 (2, 1) →
  passes_through l2 (0, 2) := by sorry

end NUMINAMATH_CALUDE_symmetric_line_fixed_point_l1679_167993


namespace NUMINAMATH_CALUDE_toy_car_production_l1679_167942

theorem toy_car_production (yesterday : ℕ) (today : ℕ) (total : ℕ) : 
  today = 2 * yesterday → 
  total = yesterday + today → 
  total = 180 → 
  yesterday = 60 :=
by sorry

end NUMINAMATH_CALUDE_toy_car_production_l1679_167942


namespace NUMINAMATH_CALUDE_min_value_of_squared_differences_l1679_167916

theorem min_value_of_squared_differences (a α β : ℝ) : 
  (α^2 - 2*a*α + a + 6 = 0) →
  (β^2 - 2*a*β + a + 6 = 0) →
  α ≠ β →
  ∃ (m : ℝ), m = 8 ∧ ∀ (x y : ℝ), 
    (x^2 - 2*a*x + a + 6 = 0) → 
    (y^2 - 2*a*y + a + 6 = 0) → 
    x ≠ y → 
    (x - 1)^2 + (y - 1)^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_squared_differences_l1679_167916


namespace NUMINAMATH_CALUDE_least_integer_in_ratio_l1679_167931

theorem least_integer_in_ratio (a b c : ℕ+) : 
  a.val + b.val + c.val = 90 →
  2 * a = 3 * a →
  5 * a = 3 * b →
  a ≤ b ∧ a ≤ c →
  a.val = 9 := by
sorry

end NUMINAMATH_CALUDE_least_integer_in_ratio_l1679_167931


namespace NUMINAMATH_CALUDE_characterize_function_l1679_167949

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem characterize_function (f : RealFunction) :
  (∀ x y : ℝ, f (f x ^ 2 + f y) = x * f x + y) →
  (∀ x : ℝ, f x = x ∨ f x = -x) :=
by sorry

end NUMINAMATH_CALUDE_characterize_function_l1679_167949


namespace NUMINAMATH_CALUDE_fractional_method_optimization_l1679_167902

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The theorem for the fractional method optimization -/
theorem fractional_method_optimization (range : ℕ) (division_points : ℕ) (n : ℕ) :
  range = 21 →
  division_points = 20 →
  fib (n + 1) - 1 = division_points →
  n = 6 :=
by sorry

end NUMINAMATH_CALUDE_fractional_method_optimization_l1679_167902


namespace NUMINAMATH_CALUDE_meeting_day_is_wednesday_l1679_167937

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

-- Define the brothers
inductive Brother
| Tralalala
| Trulala

def lies (b : Brother) (d : Day) : Prop :=
  match b with
  | Brother.Tralalala => d = Day.Monday ∨ d = Day.Tuesday ∨ d = Day.Wednesday
  | Brother.Trulala => d = Day.Thursday ∨ d = Day.Friday ∨ d = Day.Saturday

def next_day (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

theorem meeting_day_is_wednesday :
  ∃ (b1 b2 : Brother) (d : Day),
    b1 ≠ b2 ∧
    (lies b1 Day.Saturday ↔ lies b1 d) ∧
    (lies b2 (next_day d) ↔ ¬(lies b2 d)) ∧
    (lies b1 Day.Sunday ↔ lies b1 d) ∧
    d = Day.Wednesday :=
  sorry


end NUMINAMATH_CALUDE_meeting_day_is_wednesday_l1679_167937


namespace NUMINAMATH_CALUDE_square_divisibility_l1679_167922

theorem square_divisibility (n : ℤ) : (∃ k : ℤ, n^2 = 9*k) ∨ (∃ m : ℤ, n^2 = 3*m + 1) := by
  sorry

end NUMINAMATH_CALUDE_square_divisibility_l1679_167922


namespace NUMINAMATH_CALUDE_jerry_birthday_money_weighted_mean_l1679_167948

-- Define the exchange rates
def euro_to_usd : ℝ := 1.20
def gbp_to_usd : ℝ := 1.38

-- Define the weighted percentages
def family_weight : ℝ := 0.40
def friends_weight : ℝ := 0.60

-- Define the money received from family members in USD
def aunt_gift : ℝ := 9
def uncle_gift : ℝ := 9 * euro_to_usd
def sister_gift : ℝ := 7

-- Define the money received from friends in USD
def friends_gifts : List ℝ := [22, 23, 18 * euro_to_usd, 15 * gbp_to_usd, 22]

-- Calculate total family gifts
def family_total : ℝ := aunt_gift + uncle_gift + sister_gift

-- Calculate total friends gifts
def friends_total : ℝ := friends_gifts.sum

-- Define the weighted mean calculation
def weighted_mean : ℝ := family_total * family_weight + friends_total * friends_weight

-- Theorem to prove
theorem jerry_birthday_money_weighted_mean :
  weighted_mean = 76.30 := by sorry

end NUMINAMATH_CALUDE_jerry_birthday_money_weighted_mean_l1679_167948


namespace NUMINAMATH_CALUDE_s_eight_value_l1679_167917

theorem s_eight_value (x : ℝ) (h : x + 1/x = 4) : 
  let S : ℕ → ℝ := λ m => x^m + 1/(x^m)
  S 8 = 37634 := by
  sorry

end NUMINAMATH_CALUDE_s_eight_value_l1679_167917


namespace NUMINAMATH_CALUDE_exists_common_language_l1679_167982

/-- Represents a scientist at the conference -/
structure Scientist where
  id : Nat
  languages : Finset String
  lang_count : languages.card ≤ 4

/-- The set of all scientists at the conference -/
def Scientists : Finset Scientist :=
  sorry

/-- The number of scientists at the conference -/
axiom scientist_count : Scientists.card = 200

/-- For any three scientists, at least two share a common language -/
axiom common_language (s1 s2 s3 : Scientist) :
  s1 ∈ Scientists → s2 ∈ Scientists → s3 ∈ Scientists →
  ∃ (l : String), (l ∈ s1.languages ∧ l ∈ s2.languages) ∨
                  (l ∈ s1.languages ∧ l ∈ s3.languages) ∨
                  (l ∈ s2.languages ∧ l ∈ s3.languages)

/-- Main theorem: There exists a language spoken by at least 26 scientists -/
theorem exists_common_language :
  ∃ (l : String), (Scientists.filter (fun s => l ∈ s.languages)).card ≥ 26 :=
sorry

end NUMINAMATH_CALUDE_exists_common_language_l1679_167982


namespace NUMINAMATH_CALUDE_square_difference_l1679_167934

theorem square_difference (a b : ℕ) (h1 : b - a = 3) (h2 : b = 8) : b^2 - a^2 = 39 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1679_167934


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l1679_167999

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l1679_167999


namespace NUMINAMATH_CALUDE_linear_system_ratio_l1679_167977

theorem linear_system_ratio (x y a b : ℝ) 
  (eq1 : 4 * x - 6 * y = a)
  (eq2 : 9 * x - 6 * y = b)
  (x_nonzero : x ≠ 0)
  (y_nonzero : y ≠ 0)
  (b_nonzero : b ≠ 0) :
  a / b = 2 := by
sorry

end NUMINAMATH_CALUDE_linear_system_ratio_l1679_167977


namespace NUMINAMATH_CALUDE_fraction_calculation_l1679_167995

theorem fraction_calculation : (1/4 + 1/6 - 1/2) / (-1/24) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1679_167995


namespace NUMINAMATH_CALUDE_pomelo_price_at_6kg_l1679_167908

-- Define the relationship between weight and price
def price_function (x : ℝ) : ℝ := 1.4 * x

-- Theorem statement
theorem pomelo_price_at_6kg : price_function 6 = 8.4 := by
  sorry

end NUMINAMATH_CALUDE_pomelo_price_at_6kg_l1679_167908


namespace NUMINAMATH_CALUDE_point_coordinates_on_directed_segment_l1679_167997

/-- Given points M and N, and point P on the directed line segment MN such that MP = 3PN,
    prove that the coordinates of point P are (11/4, -1/4). -/
theorem point_coordinates_on_directed_segment (M N P : ℝ × ℝ) :
  M = (2, 5) →
  N = (3, -2) →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • M + t • N) →
  3 • (N - P) = P - M →
  P = (11/4, -1/4) := by
sorry

end NUMINAMATH_CALUDE_point_coordinates_on_directed_segment_l1679_167997


namespace NUMINAMATH_CALUDE_tip_percentage_calculation_l1679_167963

theorem tip_percentage_calculation (total_bill : ℝ) (num_people : ℕ) (individual_payment : ℝ) :
  total_bill = 139 ∧ num_people = 5 ∧ individual_payment = 30.58 →
  (individual_payment * num_people - total_bill) / total_bill * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_calculation_l1679_167963


namespace NUMINAMATH_CALUDE_water_volume_in_cone_l1679_167913

/-- The volume of water remaining in a conical container after pouring from a cylindrical container -/
theorem water_volume_in_cone (base_radius : ℝ) (height : ℝ) (overflow_volume : ℝ) :
  base_radius > 0 ∧ height > 0 ∧ overflow_volume = 36.2 →
  let cone_volume := (1 / 3) * Real.pi * base_radius^2 * height
  let cylinder_volume := Real.pi * base_radius^2 * height
  overflow_volume = 2 / 3 * cylinder_volume →
  cone_volume = 18.1 := by
sorry

end NUMINAMATH_CALUDE_water_volume_in_cone_l1679_167913


namespace NUMINAMATH_CALUDE_tangent_point_and_inequality_condition_l1679_167969

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem tangent_point_and_inequality_condition (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (∀ x : ℝ, f a x₀ + (a - 1 / x₀) * (x - x₀) = 0 → x = 0) ∧ 
    x₀ = Real.exp 1) ∧
  (∀ x : ℝ, x ≥ 1 → f a x ≥ a * (2 * x - x^2) → a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_point_and_inequality_condition_l1679_167969


namespace NUMINAMATH_CALUDE_ratio_problem_l1679_167983

theorem ratio_problem (a b : ℝ) : 
  (a / b = 3 / 8) → 
  ((a - 24) / (b - 24) = 4 / 9) → 
  max a b = 192 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1679_167983


namespace NUMINAMATH_CALUDE_quarters_borrowed_l1679_167996

/-- Represents the number of quarters Jessica had initially -/
def initial_quarters : ℕ := 8

/-- Represents the number of quarters Jessica has now -/
def current_quarters : ℕ := 5

/-- Represents the number of quarters Jessica's sister borrowed -/
def borrowed_quarters : ℕ := initial_quarters - current_quarters

theorem quarters_borrowed :
  borrowed_quarters = initial_quarters - current_quarters :=
by sorry

end NUMINAMATH_CALUDE_quarters_borrowed_l1679_167996


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1679_167940

def quadratic_equation (x : ℝ) : Prop := 2 * x^2 - 11 * x + 5 = 0

def is_square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p^2 ∣ n) → p = 1

theorem quadratic_root_difference (p q : ℕ) : 
  (∃ x₁ x₂ : ℝ, 
    quadratic_equation x₁ ∧ 
    quadratic_equation x₂ ∧ 
    x₁ ≠ x₂ ∧
    |x₁ - x₂| = Real.sqrt p / q) →
  q > 0 →
  is_square_free p →
  p + q = 83 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1679_167940


namespace NUMINAMATH_CALUDE_intersected_half_of_non_intersected_for_three_l1679_167974

/-- The number of unit cubes intersected by space diagonals in a cube of edge length n -/
def intersected_cubes (n : ℕ) : ℕ :=
  if n % 2 = 0 then 4 * n else 4 * n - 3

/-- The total number of unit cubes in a cube of edge length n -/
def total_cubes (n : ℕ) : ℕ := n^3

/-- The number of unit cubes not intersected by space diagonals in a cube of edge length n -/
def non_intersected_cubes (n : ℕ) : ℕ := total_cubes n - intersected_cubes n

/-- Theorem stating that for a cube with edge length 3, the number of intersected cubes
    is exactly half the number of non-intersected cubes -/
theorem intersected_half_of_non_intersected_for_three :
  2 * intersected_cubes 3 = non_intersected_cubes 3 := by
  sorry

end NUMINAMATH_CALUDE_intersected_half_of_non_intersected_for_three_l1679_167974


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1679_167964

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + m = 0) ↔ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1679_167964


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1679_167979

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 6

theorem quadratic_function_properties :
  (f (-1) = 0) ∧ 
  (f 3 = 0) ∧ 
  (f 1 = -8) ∧
  (∀ x ∈ Set.Icc 0 3, f x ≥ -8) ∧
  (∀ x ∈ Set.Icc 0 3, f x ≤ 0) ∧
  (f 1 = -8) ∧
  (f 3 = 0) ∧
  (∀ x, f x ≥ 0 ↔ x ≤ -1 ∨ x ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1679_167979


namespace NUMINAMATH_CALUDE_inequality_proof_l1679_167926

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1679_167926


namespace NUMINAMATH_CALUDE_grocer_sales_theorem_l1679_167972

def sales : List ℕ := [5420, 5660, 6200, 6350, 6500, 6780, 7000, 7200]
def target_average : ℕ := 6600
def num_months : ℕ := 10

theorem grocer_sales_theorem : 
  let total_target := target_average * num_months
  let current_total := sales.sum
  let remaining_months := num_months - sales.length
  let remaining_sales := total_target - current_total
  remaining_sales / remaining_months = 9445 := by sorry

end NUMINAMATH_CALUDE_grocer_sales_theorem_l1679_167972


namespace NUMINAMATH_CALUDE_percentage_problem_l1679_167906

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 100) : 
  (600 / x) * 100 = 120 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l1679_167906


namespace NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_198_l1679_167990

theorem first_nonzero_digit_after_decimal_1_198 : ∃ (n : ℕ) (d : ℕ), 
  1 ≤ d ∧ d ≤ 9 ∧ 
  (∃ (m : ℕ), 1/198 = (n : ℚ)/10^m + d/(10^(m+1) : ℚ) + (1/198 - (n : ℚ)/10^m - d/(10^(m+1) : ℚ)) ∧ 
   0 ≤ 1/198 - (n : ℚ)/10^m - d/(10^(m+1) : ℚ) ∧ 
   1/198 - (n : ℚ)/10^m - d/(10^(m+1) : ℚ) < 1/(10^(m+1) : ℚ)) ∧
  d = 5 :=
by sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_198_l1679_167990


namespace NUMINAMATH_CALUDE_a_must_be_positive_l1679_167930

theorem a_must_be_positive
  (a b c d : ℝ)
  (h1 : b ≠ 0)
  (h2 : d ≠ 0)
  (h3 : d > 0)
  (h4 : a / b > -(3 / (2 * d))) :
  a > 0 :=
by sorry

end NUMINAMATH_CALUDE_a_must_be_positive_l1679_167930


namespace NUMINAMATH_CALUDE_degree_of_example_monomial_not_six_l1679_167919

/-- The degree of a monomial is the sum of the exponents of its variables -/
def monomial_degree (m : Polynomial ℤ) : ℕ := sorry

/-- A function to represent the monomial -2^2xab^2 -/
def example_monomial : Polynomial ℤ := sorry

theorem degree_of_example_monomial_not_six :
  monomial_degree example_monomial ≠ 6 := by sorry

end NUMINAMATH_CALUDE_degree_of_example_monomial_not_six_l1679_167919


namespace NUMINAMATH_CALUDE_xiao_hua_at_13_l1679_167998

/-- The floor Xiao Hua reaches when Xiao Li reaches a given floor -/
def xiao_hua_floor (xiao_li_floor : ℕ) : ℕ :=
  1 + ((xiao_li_floor - 1) * (3 - 1)) / (5 - 1)

/-- Theorem: When Xiao Li reaches the 25th floor, Xiao Hua will have reached the 13th floor -/
theorem xiao_hua_at_13 : xiao_hua_floor 25 = 13 := by
  sorry

end NUMINAMATH_CALUDE_xiao_hua_at_13_l1679_167998


namespace NUMINAMATH_CALUDE_angle_properties_l1679_167907

def angle_set (α : Real) : Set Real :=
  {x | ∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 3}

theorem angle_properties (α : Real) 
  (h : ∃ x y : Real, x = 1 ∧ y = Real.sqrt 3 ∧ x * Real.cos α = x ∧ y * Real.sin α = y) :
  (Real.sin (Real.pi - α) - Real.sin (Real.pi / 2 + α) = (Real.sqrt 3 - 1) / 2) ∧
  (angle_set α = {α}) := by
  sorry

end NUMINAMATH_CALUDE_angle_properties_l1679_167907


namespace NUMINAMATH_CALUDE_min_pencils_theorem_l1679_167980

def min_pencils_to_take (red blue green : ℕ) (red_goal blue_goal green_goal : ℕ) : ℕ :=
  (red + blue + green) - (red - red_goal).min 0 - (blue - blue_goal).min 0 - (green - green_goal).min 0 + 1

theorem min_pencils_theorem :
  min_pencils_to_take 15 13 8 1 2 3 = 22 :=
by sorry

end NUMINAMATH_CALUDE_min_pencils_theorem_l1679_167980


namespace NUMINAMATH_CALUDE_smallest_solution_is_negative_one_l1679_167925

-- Define the equation
def equation (x : ℝ) : Prop :=
  3 * x / (x - 3) + (3 * x^2 - 36) / (x + 3) = 15

-- Theorem statement
theorem smallest_solution_is_negative_one :
  (∃ x : ℝ, equation x) ∧ 
  (∀ y : ℝ, equation y → y ≥ -1) ∧
  equation (-1) :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_is_negative_one_l1679_167925


namespace NUMINAMATH_CALUDE_positive_X_value_l1679_167927

def hash (X Y : ℝ) : ℝ := X^2 + Y^2

theorem positive_X_value (X : ℝ) (h : hash X 7 = 290) : X = 17 := by
  sorry

end NUMINAMATH_CALUDE_positive_X_value_l1679_167927


namespace NUMINAMATH_CALUDE_profit_is_085_l1679_167915

/-- Calculates the total profit for Niko's sock reselling business --/
def calculate_profit : ℝ :=
  let initial_cost : ℝ := 9 * 2
  let discount_rate : ℝ := 0.1
  let discount : ℝ := initial_cost * discount_rate
  let cost_after_discount : ℝ := initial_cost - discount
  let shipping_storage : ℝ := 5
  let total_cost : ℝ := cost_after_discount + shipping_storage
  let resell_price_4_pairs : ℝ := 4 * (2 + 2 * 0.25)
  let resell_price_5_pairs : ℝ := 5 * (2 + 0.2)
  let total_resell_price : ℝ := resell_price_4_pairs + resell_price_5_pairs
  let sales_tax_rate : ℝ := 0.05
  let sales_tax : ℝ := total_resell_price * sales_tax_rate
  let total_revenue : ℝ := total_resell_price + sales_tax
  total_revenue - total_cost

theorem profit_is_085 : calculate_profit = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_profit_is_085_l1679_167915


namespace NUMINAMATH_CALUDE_april_earnings_l1679_167909

def rose_price : ℕ := 7
def initial_roses : ℕ := 9
def remaining_roses : ℕ := 4

theorem april_earnings : (initial_roses - remaining_roses) * rose_price = 35 := by
  sorry

end NUMINAMATH_CALUDE_april_earnings_l1679_167909


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_perpendicular_line_contained_line_l1679_167956

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def contains (p : Plane) (l : Line) : Prop := sorry

-- State the theorems
theorem perpendicular_lines_parallel (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n := by sorry

theorem perpendicular_line_contained_line (m n : Line) (α : Plane) :
  perpendicular m α → contains α n → perpendicular m n := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_perpendicular_line_contained_line_l1679_167956


namespace NUMINAMATH_CALUDE_sheila_hourly_rate_l1679_167991

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  eight_hour_days : Nat
  six_hour_days : Nat
  weekly_earnings : Nat

/-- Calculate Sheila's hourly rate --/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (8 * schedule.eight_hour_days + 6 * schedule.six_hour_days)

/-- Theorem: Sheila's hourly rate is $6 --/
theorem sheila_hourly_rate :
  let schedule : WorkSchedule := {
    eight_hour_days := 3,
    six_hour_days := 2,
    weekly_earnings := 216
  }
  hourly_rate schedule = 6 := by sorry

end NUMINAMATH_CALUDE_sheila_hourly_rate_l1679_167991


namespace NUMINAMATH_CALUDE_largest_ball_on_torus_l1679_167988

/-- The radius of the largest spherical ball that can be placed on top of a torus -/
def largest_ball_radius (inner_radius outer_radius : ℝ) (torus_center : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem stating that the largest ball radius for the given torus is 4 -/
theorem largest_ball_on_torus :
  let inner_radius : ℝ := 3
  let outer_radius : ℝ := 5
  let torus_center : ℝ × ℝ × ℝ := (4, 0, 1)
  largest_ball_radius inner_radius outer_radius torus_center = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_ball_on_torus_l1679_167988


namespace NUMINAMATH_CALUDE_exists_valid_sign_assignment_l1679_167903

/-- Represents a vertex in the triangular grid --/
structure Vertex :=
  (x : ℤ)
  (y : ℤ)

/-- Represents a triangle in the grid --/
structure Triangle :=
  (a : Vertex)
  (b : Vertex)
  (c : Vertex)

/-- The type of sign assignment functions --/
def SignAssignment := Vertex → Int

/-- Predicate to check if a triangle satisfies the sign rule --/
def satisfiesRule (f : SignAssignment) (t : Triangle) : Prop :=
  (f t.a = f t.b → f t.c = 1) ∧
  (f t.a ≠ f t.b → f t.c = -1)

/-- The set of all triangles in the grid --/
def allTriangles : Set Triangle := sorry

/-- Statement of the theorem --/
theorem exists_valid_sign_assignment :
  ∃ (f : SignAssignment),
    (∀ t ∈ allTriangles, satisfiesRule f t) ∧
    (∃ v w : Vertex, f v ≠ f w) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_sign_assignment_l1679_167903


namespace NUMINAMATH_CALUDE_total_books_read_is_48cs_l1679_167941

/-- The number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  c * s * 4 * 12

/-- Theorem: The total number of books read by the entire student body in one year is 48cs -/
theorem total_books_read_is_48cs (c s : ℕ) : total_books_read c s = 48 * c * s := by
  sorry

end NUMINAMATH_CALUDE_total_books_read_is_48cs_l1679_167941


namespace NUMINAMATH_CALUDE_fixed_fee_determination_l1679_167950

/-- Represents the billing system for an online service provider -/
structure BillingSystem where
  fixedFee : ℝ
  hourlyCharge : ℝ

/-- Calculates the total bill given the billing system and hours used -/
def calculateBill (bs : BillingSystem) (hours : ℝ) : ℝ :=
  bs.fixedFee + bs.hourlyCharge * hours

theorem fixed_fee_determination (bs : BillingSystem) 
  (h1 : calculateBill bs 1 = 18.70)
  (h2 : calculateBill bs 3 = 34.10) : 
  bs.fixedFee = 11.00 := by
  sorry

end NUMINAMATH_CALUDE_fixed_fee_determination_l1679_167950


namespace NUMINAMATH_CALUDE_prove_theta_value_l1679_167939

-- Define the angles in degrees
def angle_VEK : ℝ := 70
def angle_KEW : ℝ := 40
def angle_EVG : ℝ := 110

-- Define θ as a real number
def θ : ℝ := 40

-- Theorem statement
theorem prove_theta_value :
  angle_VEK = 70 ∧
  angle_KEW = 40 ∧
  angle_EVG = 110 →
  θ = 40 := by
  sorry


end NUMINAMATH_CALUDE_prove_theta_value_l1679_167939


namespace NUMINAMATH_CALUDE_sin_2theta_value_l1679_167975

theorem sin_2theta_value (θ : Real) (h : Real.tan θ + 1 / Real.tan θ = 4) : 
  Real.sin (2 * θ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l1679_167975


namespace NUMINAMATH_CALUDE_impossible_sum_16_l1679_167945

def standard_die := Finset.range 6

theorem impossible_sum_16 (roll1 roll2 : ℕ) :
  roll1 ∈ standard_die → roll2 ∈ standard_die → roll1 + roll2 ≠ 16 := by
  sorry

end NUMINAMATH_CALUDE_impossible_sum_16_l1679_167945


namespace NUMINAMATH_CALUDE_smallest_n_both_composite_l1679_167994

def is_composite (n : ℕ) : Prop := ∃ a b, a > 1 ∧ b > 1 ∧ a * b = n

theorem smallest_n_both_composite :
  (∀ n : ℕ, n > 0 ∧ n < 13 → ¬(is_composite (2*n - 1) ∧ is_composite (2*n + 1))) ∧
  (is_composite (2*13 - 1) ∧ is_composite (2*13 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_both_composite_l1679_167994


namespace NUMINAMATH_CALUDE_d_eq_4_sufficient_not_necessary_l1679_167957

/-- An arithmetic sequence with first term 2 and common difference d -/
def arithmetic_seq (n : ℕ) (d : ℝ) : ℝ := 2 + (n - 1) * d

/-- Condition for a_1, a_2, a_5 to form a geometric sequence -/
def is_geometric (d : ℝ) : Prop :=
  (arithmetic_seq 2 d)^2 = (arithmetic_seq 1 d) * (arithmetic_seq 5 d)

/-- d = 4 is a sufficient but not necessary condition for a_1, a_2, a_5 to form a geometric sequence -/
theorem d_eq_4_sufficient_not_necessary :
  (∀ d : ℝ, d = 4 → is_geometric d) ∧
  ¬(∀ d : ℝ, is_geometric d → d = 4) :=
sorry

end NUMINAMATH_CALUDE_d_eq_4_sufficient_not_necessary_l1679_167957


namespace NUMINAMATH_CALUDE_unique_number_with_specific_divisors_l1679_167921

theorem unique_number_with_specific_divisors : ∃! n : ℕ, 
  (9 ∣ n) ∧ (5 ∣ n) ∧ (Finset.card (Nat.divisors n) = 14) ∧ (n = 3645) := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_specific_divisors_l1679_167921


namespace NUMINAMATH_CALUDE_earthworm_catches_centipede_l1679_167960

/-- The time (in minutes) it takes for an earthworm to catch up with a centipede given their speeds and initial distance -/
def catch_up_time (centipede_speed earthworm_speed initial_distance : ℚ) : ℚ :=
  initial_distance / (earthworm_speed - centipede_speed)

/-- Theorem stating that under the given conditions, the earthworm catches up with the centipede in 24 minutes -/
theorem earthworm_catches_centipede :
  let centipede_speed : ℚ := 5 / 3  -- 5 meters in 3 minutes
  let earthworm_speed : ℚ := 5 / 2  -- 5 meters in 2 minutes
  let initial_distance : ℚ := 20    -- 20 meters ahead
  catch_up_time centipede_speed earthworm_speed initial_distance = 24 := by
  sorry

#eval catch_up_time (5/3) (5/2) 20

end NUMINAMATH_CALUDE_earthworm_catches_centipede_l1679_167960


namespace NUMINAMATH_CALUDE_total_sandwiches_l1679_167970

/-- The number of sandwiches made by each person and the total -/
def sandwiches : ℕ → ℕ
| 0 => 49  -- Billy
| 1 => 49 + (49 * 3 / 10)  -- Katelyn
| 2 => (sandwiches 1 * 3) / 5  -- Chloe
| 3 => 25  -- Emma
| 4 => 25 * 2  -- Stella
| _ => 0

/-- The theorem stating the total number of sandwiches made -/
theorem total_sandwiches : 
  sandwiches 0 + sandwiches 1 + sandwiches 2 + sandwiches 3 + sandwiches 4 = 226 := by
  sorry


end NUMINAMATH_CALUDE_total_sandwiches_l1679_167970


namespace NUMINAMATH_CALUDE_tan_equality_implies_sixty_degrees_l1679_167966

theorem tan_equality_implies_sixty_degrees (n : ℤ) :
  -90 < n ∧ n < 90 →
  Real.tan (n * π / 180) = Real.tan (240 * π / 180) →
  n = 60 := by
sorry

end NUMINAMATH_CALUDE_tan_equality_implies_sixty_degrees_l1679_167966


namespace NUMINAMATH_CALUDE_fraction_sum_l1679_167978

theorem fraction_sum : (3 : ℚ) / 9 + (7 : ℚ) / 12 = (11 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1679_167978


namespace NUMINAMATH_CALUDE_xiang_lake_one_millionth_closest_to_study_room_l1679_167938

/-- The combined area of Phase I and Phase II of Xiang Lake in square kilometers -/
def xiang_lake_area : ℝ := 10.6

/-- One million as a real number -/
def one_million : ℝ := 1000000

/-- Conversion factor from square kilometers to square meters -/
def km2_to_m2 : ℝ := 1000000

/-- Approximate area of a typical study room in square meters -/
def typical_study_room_area : ℝ := 10.6

/-- Theorem stating that one-millionth of Xiang Lake's area is closest to a typical study room's area -/
theorem xiang_lake_one_millionth_closest_to_study_room :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |xiang_lake_area * km2_to_m2 / one_million - typical_study_room_area| < ε :=
sorry

end NUMINAMATH_CALUDE_xiang_lake_one_millionth_closest_to_study_room_l1679_167938


namespace NUMINAMATH_CALUDE_gift_shop_combinations_l1679_167918

/-- The number of varieties of wrapping paper -/
def wrapping_paper_varieties : ℕ := 8

/-- The number of colors of ribbon -/
def ribbon_colors : ℕ := 3

/-- The number of types of gift cards -/
def gift_card_types : ℕ := 5

/-- The number of varieties of stickers -/
def sticker_varieties : ℕ := 5

/-- The total number of possible combinations -/
def total_combinations : ℕ := wrapping_paper_varieties * ribbon_colors * gift_card_types * sticker_varieties

theorem gift_shop_combinations : total_combinations = 600 := by
  sorry

end NUMINAMATH_CALUDE_gift_shop_combinations_l1679_167918


namespace NUMINAMATH_CALUDE_cloth_cost_price_per_meter_l1679_167905

/-- Given a cloth sale scenario, prove the cost price per meter. -/
theorem cloth_cost_price_per_meter
  (total_length : ℕ)
  (total_selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_length = 66)
  (h2 : total_selling_price = 660)
  (h3 : profit_per_meter = 5) :
  (total_selling_price - total_length * profit_per_meter) / total_length = 5 :=
by sorry

end NUMINAMATH_CALUDE_cloth_cost_price_per_meter_l1679_167905


namespace NUMINAMATH_CALUDE_system_solutions_l1679_167989

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  x^2 + y^2 = z^2 ∧ x*z = y^2 ∧ x*y = 10

-- State the theorem
theorem system_solutions :
  ∃ (x y z : ℝ), system x y z ∧
  ((x = Real.sqrt 10 ∧ y = Real.sqrt 10 ∧ z = Real.sqrt 10) ∨
   (x = -Real.sqrt 10 ∧ y = -Real.sqrt 10 ∧ z = -Real.sqrt 10)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l1679_167989


namespace NUMINAMATH_CALUDE_special_rectangle_area_l1679_167973

/-- Represents a rectangle with specific properties -/
structure SpecialRectangle where
  d : ℝ  -- diagonal length
  w : ℝ  -- width
  h : ℝ  -- height (length)
  h_eq_3w : h = 3 * w  -- length is three times the width
  diagonal_eq : d^2 = w^2 + h^2  -- Pythagorean theorem

/-- The area of a SpecialRectangle is (3/10) * d^2 -/
theorem special_rectangle_area (r : SpecialRectangle) : r.w * r.h = (3/10) * r.d^2 := by
  sorry

#check special_rectangle_area

end NUMINAMATH_CALUDE_special_rectangle_area_l1679_167973


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_k_range_l1679_167912

theorem empty_solution_set_implies_k_range (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 ≥ 0) → k ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_k_range_l1679_167912


namespace NUMINAMATH_CALUDE_initial_birds_l1679_167961

theorem initial_birds (initial_birds final_birds additional_birds : ℕ) 
  (h1 : additional_birds = 21)
  (h2 : final_birds = 35)
  (h3 : final_birds = initial_birds + additional_birds) : 
  initial_birds = 14 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_l1679_167961


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l1679_167976

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : x + 3 * y = 9) 
  (h2 : x * y = -15) : 
  x^2 + 9 * y^2 = 171 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l1679_167976


namespace NUMINAMATH_CALUDE_largest_remaining_circle_l1679_167958

/-- Represents a circle with a given diameter -/
structure Circle where
  diameter : ℝ
  diameter_pos : diameter > 0

/-- The problem setup -/
def plywood_problem (initial : Circle) (cutout1 : Circle) (cutout2 : Circle) : Prop :=
  initial.diameter = 30 ∧ cutout1.diameter = 20 ∧ cutout2.diameter = 10

/-- The theorem to be proved -/
theorem largest_remaining_circle 
  (initial : Circle) (cutout1 : Circle) (cutout2 : Circle) 
  (h : plywood_problem initial cutout1 cutout2) : 
  ∃ (largest : Circle), largest.diameter = 30 / 7 ∧ 
  ∀ (c : Circle), c.diameter ≤ largest.diameter :=
sorry

end NUMINAMATH_CALUDE_largest_remaining_circle_l1679_167958
