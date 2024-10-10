import Mathlib

namespace reciprocal_equals_self_is_negative_one_l3469_346903

theorem reciprocal_equals_self_is_negative_one (x : ℝ) :
  x < 0 ∧ 1 / x = x → x = -1 := by
  sorry

end reciprocal_equals_self_is_negative_one_l3469_346903


namespace alice_winning_strategy_l3469_346988

/-- Represents the peg game with n holes and k pegs. -/
structure PegGame where
  n : ℕ
  k : ℕ
  h1 : 1 ≤ k
  h2 : k < n

/-- Predicate that determines if Alice has a winning strategy. -/
def alice_wins (game : PegGame) : Prop :=
  ¬(Even game.n ∧ Even game.k)

/-- The main theorem about Alice's winning strategy in the peg game. -/
theorem alice_winning_strategy (game : PegGame) :
  alice_wins game ↔
  (∃ (strategy : Unit), 
    (∀ (bob_move : Unit), ∃ (alice_move : Unit), 
      -- Alice can always make a move that leads to a winning position
      true)) := by sorry

end alice_winning_strategy_l3469_346988


namespace value_of_y_l3469_346921

theorem value_of_y (x y : ℤ) (h1 : x^2 + x + 4 = y - 4) (h2 : x = -7) : y = 50 := by
  sorry

end value_of_y_l3469_346921


namespace xyz_value_l3469_346901

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 15)
  (h3 : x + y + z = 5) :
  x * y * z = 10 := by
sorry

end xyz_value_l3469_346901


namespace f_minimum_at_cos2x_neg_half_l3469_346919

noncomputable def f (x : ℝ) : ℝ := 9 / (8 * Real.cos (2 * x) + 16) - Real.sin x ^ 2

theorem f_minimum_at_cos2x_neg_half :
  ∀ x : ℝ, f x ≥ 0 ∧ (f x = 0 ↔ Real.cos (2 * x) = -1/2) :=
sorry

end f_minimum_at_cos2x_neg_half_l3469_346919


namespace min_sort_steps_l3469_346993

/-- Represents the color of a cow -/
inductive Color
| Purple
| White

/-- A configuration of cows -/
def Configuration (n : ℕ) := Fin (2 * n) → Color

/-- A valid swap operation on a configuration -/
def ValidSwap (n : ℕ) (c : Configuration n) (i j : ℕ) : Prop :=
  i < j ∧ j ≤ 2 * n ∧ j - i = 2 * n - j

/-- The number of steps required to sort a configuration -/
def SortSteps (n : ℕ) (c : Configuration n) : ℕ := sorry

/-- The theorem stating that n steps are always sufficient and sometimes necessary -/
theorem min_sort_steps (n : ℕ) :
  (∀ c : Configuration n, SortSteps n c ≤ n) ∧
  (∃ c : Configuration n, SortSteps n c = n) := by sorry

end min_sort_steps_l3469_346993


namespace tan_half_product_l3469_346958

theorem tan_half_product (a b : ℝ) :
  5 * (Real.cos a + Real.cos b) + 4 * (Real.cos a * Real.cos b + 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2) = 3 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -3) :=
by sorry

end tan_half_product_l3469_346958


namespace fraction_comparison_l3469_346970

theorem fraction_comparison : (17 : ℚ) / 14 > (31 : ℚ) / 11 := by sorry

end fraction_comparison_l3469_346970


namespace parabola_intersection_range_l3469_346945

/-- The parabola function -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - (4*m + 1)*x + 2*m - 1

theorem parabola_intersection_range (m : ℝ) :
  (∃ x₁ x₂, x₁ < 2 ∧ x₂ > 2 ∧ f m x₁ = 0 ∧ f m x₂ = 0) →  -- Intersects x-axis at two points
  (f m 0 < -1/2) →  -- Intersects y-axis below (0, -1/2)
  1/6 < m ∧ m < 1/4 :=
sorry

end parabola_intersection_range_l3469_346945


namespace fraction_equality_l3469_346962

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 3 / 5) 
  (h2 : r / t = 8 / 9) : 
  (3 * m^2 * r - n * t^2) / (5 * n * t^2 - 9 * m^2 * r) = -1 := by
  sorry

end fraction_equality_l3469_346962


namespace transformation_identity_l3469_346934

/-- Represents a 3D point -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Rotation 180° about y-axis -/
def rotateY180 (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := -p.z }

/-- Reflection through yz-plane -/
def reflectYZ (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

/-- Rotation 90° about z-axis -/
def rotateZ90 (p : Point3D) : Point3D :=
  { x := p.y, y := -p.x, z := p.z }

/-- Reflection through xz-plane -/
def reflectXZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

/-- Reflection through xy-plane -/
def reflectXY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

/-- The sequence of transformations -/
def transformSequence (p : Point3D) : Point3D :=
  reflectXY (reflectXZ (rotateZ90 (reflectYZ (rotateY180 p))))

theorem transformation_identity :
  transformSequence { x := 2, y := 2, z := 2 } = { x := 2, y := 2, z := 2 } := by
  sorry

end transformation_identity_l3469_346934


namespace union_of_A_and_B_l3469_346929

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define the union of A and B
def AUnionB : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = AUnionB := by sorry

end union_of_A_and_B_l3469_346929


namespace age_proof_l3469_346950

/-- Proves the ages of Desiree and her cousin given the conditions -/
theorem age_proof (desiree_age : ℝ) (cousin_age : ℝ) : 
  desiree_age = 2.99999835 →
  cousin_age = 1.499999175 →
  desiree_age = 2 * cousin_age →
  desiree_age + 30 = 0.6666666 * (cousin_age + 30) + 14 :=
by sorry

end age_proof_l3469_346950


namespace unique_two_digit_integer_l3469_346942

theorem unique_two_digit_integer (t : ℕ) : 
  (10 ≤ t ∧ t < 100) ∧ (11 * t) % 100 = 36 ↔ t = 76 := by sorry

end unique_two_digit_integer_l3469_346942


namespace man_speed_against_current_l3469_346994

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem stating that for the given conditions, 
    the man's speed against the current is 14 km/h. -/
theorem man_speed_against_current :
  speed_against_current 20 3 = 14 := by
  sorry

end man_speed_against_current_l3469_346994


namespace custom_operation_equation_l3469_346918

-- Define the custom operation *
def star (a b : ℤ) : ℤ := 2 * a + b

-- State the theorem
theorem custom_operation_equation :
  ∃ x : ℤ, star 3 (star 4 x) = -1 ∧ x = -15 := by
  sorry

end custom_operation_equation_l3469_346918


namespace ad_difference_l3469_346959

/-- Represents the number of ads on each web page -/
structure WebPageAds where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The conditions of the problem -/
def adConditions (w : WebPageAds) : Prop :=
  w.first = 12 ∧
  w.second = 2 * w.first ∧
  w.third > w.second ∧
  w.fourth = (3 * w.second) / 4 ∧
  (2 * (w.first + w.second + w.third + w.fourth)) / 3 = 68

theorem ad_difference (w : WebPageAds) (h : adConditions w) : 
  w.third - w.second = 24 := by
sorry

end ad_difference_l3469_346959


namespace trout_percentage_is_sixty_percent_l3469_346927

def total_fish : ℕ := 5
def trout_price : ℕ := 5
def bluegill_price : ℕ := 4
def sunday_earnings : ℕ := 23

theorem trout_percentage_is_sixty_percent :
  ∃ (trout blue_gill : ℕ),
    trout + blue_gill = total_fish ∧
    trout * trout_price + blue_gill * bluegill_price = sunday_earnings ∧
    (trout : ℚ) / (total_fish : ℚ) = 3/5 := by
  sorry

end trout_percentage_is_sixty_percent_l3469_346927


namespace complex_number_problem_l3469_346931

-- Define the complex number Z₁
def Z₁ (a : ℝ) : ℂ := 2 + a * Complex.I

-- Main theorem
theorem complex_number_problem (a : ℝ) (ha : a > 0) 
  (h_pure_imag : ∃ b : ℝ, Z₁ a ^ 2 = b * Complex.I) :
  a = 2 ∧ Complex.abs (Z₁ a / (1 - Complex.I)) = 2 := by
  sorry


end complex_number_problem_l3469_346931


namespace lawn_mowing_problem_l3469_346939

theorem lawn_mowing_problem (mary_time tom_time tom_work_time : ℝ) 
  (h1 : mary_time = 6)
  (h2 : tom_time = 4)
  (h3 : tom_work_time = 3) :
  1 - (tom_work_time / tom_time) = 1/4 := by sorry

end lawn_mowing_problem_l3469_346939


namespace three_number_average_l3469_346936

theorem three_number_average (a b c : ℝ) 
  (h1 : (a + b) / 2 = 26.5)
  (h2 : (b + c) / 2 = 34.5)
  (h3 : (a + c) / 2 = 29)
  (h4 : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (a + b + c) / 3 = 30 := by
sorry

end three_number_average_l3469_346936


namespace four_possible_ones_digits_l3469_346909

-- Define a function to check if a number is divisible by 6
def divisible_by_six (n : ℕ) : Prop := n % 6 = 0

-- Define a function to get the ones digit of a number
def ones_digit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem four_possible_ones_digits :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, ones_digit n ∈ s ∧ divisible_by_six n) ∧
    (∀ d, d ∈ s ↔ ∃ n, ones_digit n = d ∧ divisible_by_six n) ∧
    Finset.card s = 4 :=
sorry

end four_possible_ones_digits_l3469_346909


namespace constant_expression_l3469_346905

theorem constant_expression (x y k : ℝ) 
  (eq1 : x + 2*y = k + 2) 
  (eq2 : 2*x - 3*y = 3*k - 1) : 
  x + 9*y = 7 := by
sorry

end constant_expression_l3469_346905


namespace campground_distance_l3469_346943

/-- The distance traveled by Sue's family to the campground -/
def distance_to_campground (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: The distance to the campground is 300 miles -/
theorem campground_distance : 
  distance_to_campground 60 5 = 300 := by
  sorry

end campground_distance_l3469_346943


namespace f_composition_at_one_l3469_346961

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (1/2) * x - 1 else 2^x

theorem f_composition_at_one :
  f (f 1) = Real.sqrt 2 / 2 := by
  sorry

end f_composition_at_one_l3469_346961


namespace percentage_problem_l3469_346978

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.6 * x = 240 := by
  sorry

end percentage_problem_l3469_346978


namespace zoo_penguins_l3469_346955

theorem zoo_penguins (penguins : ℕ) (polar_bears : ℕ) : 
  polar_bears = 2 * penguins → 
  penguins + polar_bears = 63 → 
  penguins = 21 := by
sorry

end zoo_penguins_l3469_346955


namespace village_assistants_selection_l3469_346997

theorem village_assistants_selection (n : ℕ) (k : ℕ) (h_n : n = 10) (h_k : k = 3) :
  (Nat.choose n k) - (Nat.choose (n - 3) k) - 
  (2 * (Nat.choose (n - 2) (k - 1)) - (Nat.choose (n - 3) (k - 2))) = 49 := by
  sorry

end village_assistants_selection_l3469_346997


namespace midpoint_line_slope_l3469_346985

/-- The slope of the line containing the midpoints of two given line segments is -3/7 -/
theorem midpoint_line_slope :
  let midpoint1 := ((1 + 3) / 2, (2 + 6) / 2)
  let midpoint2 := ((4 + 7) / 2, (1 + 4) / 2)
  let slope := (midpoint2.2 - midpoint1.2) / (midpoint2.1 - midpoint1.1)
  slope = -3 / 7 := by sorry

end midpoint_line_slope_l3469_346985


namespace no_solution_for_inequality_l3469_346940

theorem no_solution_for_inequality :
  ¬∃ (x : ℝ), x > 0 ∧ x * Real.sqrt (10 - x) + Real.sqrt (10 * x - x^3) ≥ 10 := by
  sorry

end no_solution_for_inequality_l3469_346940


namespace a_over_b_equals_half_l3469_346914

theorem a_over_b_equals_half (a b : ℤ) (h : a + Real.sqrt b = Real.sqrt (15 + Real.sqrt 216)) : a / b = 1 / 2 := by
  sorry

end a_over_b_equals_half_l3469_346914


namespace min_value_a_plus_2b_l3469_346910

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (2*a + b)⁻¹ + (b + 1)⁻¹ = 1) : 
  ∀ x y, x > 0 → y > 0 → (2*x + y)⁻¹ + (y + 1)⁻¹ = 1 → a + 2*b ≤ x + 2*y :=
by sorry

end min_value_a_plus_2b_l3469_346910


namespace parabola_vertex_on_x_axis_l3469_346965

/-- A parabola with equation y = x^2 - 4x + c has its vertex on the x-axis if and only if c = 4 -/
theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + c = 0 ∧ ∀ y : ℝ, y^2 - 4*y + c ≥ x^2 - 4*x + c) ↔ c = 4 :=
sorry

end parabola_vertex_on_x_axis_l3469_346965


namespace parallel_lines_k_values_l3469_346984

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

/-- Definition of line l₁ -/
def l₁ (k : ℝ) (x y : ℝ) : Prop :=
  (k - 3) * x + (4 - k) * y + 1 = 0

/-- Definition of line l₂ -/
def l₂ (k : ℝ) (x y : ℝ) : Prop :=
  2 * (k - 3) * x - 2 * y + 3 = 0

theorem parallel_lines_k_values :
  ∀ k : ℝ, (∀ x y : ℝ, are_parallel (k - 3) (4 - k) (2 * (k - 3)) (-2)) →
  k = 3 ∨ k = 5 := by
  sorry

end parallel_lines_k_values_l3469_346984


namespace quadratic_roots_existence_l3469_346992

theorem quadratic_roots_existence : ∃ (p q : ℝ), 
  ((p - 1)^2 - 4*q > 0) ∧ 
  ((p + 1)^2 - 4*q > 0) ∧ 
  (p^2 - 4*q < 0) :=
sorry

end quadratic_roots_existence_l3469_346992


namespace book_rearrangement_combinations_l3469_346981

/-- The number of options for each day of the week --/
def daily_options : List Nat := [1, 2, 3, 3, 2]

/-- The total number of combinations --/
def total_combinations : Nat := daily_options.prod

/-- Theorem stating that the total number of combinations is 36 --/
theorem book_rearrangement_combinations :
  total_combinations = 36 := by
  sorry

end book_rearrangement_combinations_l3469_346981


namespace dannys_bottle_caps_l3469_346952

/-- Calculates the total number of bottle caps in Danny's collection -/
def total_bottle_caps (initial : ℕ) (found : ℕ) : ℕ :=
  initial + found

/-- Theorem stating that Danny's total bottle caps is 55 -/
theorem dannys_bottle_caps :
  total_bottle_caps 37 18 = 55 := by
  sorry

end dannys_bottle_caps_l3469_346952


namespace min_area_triangle_l3469_346969

theorem min_area_triangle (m n : ℝ) : 
  let l := {(x, y) : ℝ × ℝ | m * x + n * y - 1 = 0}
  let A := (1/m, 0)
  let B := (0, 1/n)
  let O := (0, 0)
  (∀ (x y : ℝ), (x, y) ∈ l → |m * x + n * y - 1| / Real.sqrt (m^2 + n^2) = Real.sqrt 3) →
  (∃ (S : ℝ), S = (1/2) * |1/m * 1/n| ∧ 
    (∀ (S' : ℝ), S' = (1/2) * |1/m' * 1/n'| → 
      m'^2 + n'^2 = 1/3 → S ≤ S') ∧ S = 3) :=
by sorry

end min_area_triangle_l3469_346969


namespace sphere_volume_ratio_l3469_346986

theorem sphere_volume_ratio (r R : ℝ) (h : R = 4 * r) :
  (4 / 3 * Real.pi * R^3) / (4 / 3 * Real.pi * r^3) = 64 := by
  sorry

end sphere_volume_ratio_l3469_346986


namespace floor_multiple_implies_integer_l3469_346953

theorem floor_multiple_implies_integer (r : ℝ) : 
  r ≥ 1 →
  (∀ (m n : ℕ+), n.val % m.val = 0 → (⌊n.val * r⌋ : ℤ) % (⌊m.val * r⌋ : ℤ) = 0) →
  ∃ (k : ℤ), r = k := by
  sorry

end floor_multiple_implies_integer_l3469_346953


namespace bug_meeting_point_l3469_346999

/-- Triangle with sides a, b, c and point S on perimeter --/
structure Triangle (a b c : ℝ) where
  S : ℝ
  h1 : 0 < a ∧ 0 < b ∧ 0 < c
  h2 : a + b > c ∧ b + c > a ∧ c + a > b
  h3 : 0 ≤ S ∧ S ≤ a + b + c

/-- The length of QS in the triangle --/
def qsLength (t : Triangle 7 8 9) : ℝ :=
  5

theorem bug_meeting_point (t : Triangle 7 8 9) : qsLength t = 5 := by
  sorry

end bug_meeting_point_l3469_346999


namespace sum_of_central_angles_is_360_l3469_346995

/-- A circle with an inscribed pentagon -/
structure PentagonInCircle where
  /-- The circle -/
  circle : Set ℝ × Set ℝ
  /-- The inscribed pentagon -/
  pentagon : Set (ℝ × ℝ)
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- The vertices of the pentagon -/
  vertices : Fin 5 → ℝ × ℝ
  /-- The lines from vertices to center -/
  lines : Fin 5 → Set (ℝ × ℝ)

/-- The sum of angles at the center formed by lines from pentagon vertices to circle center -/
def sumOfCentralAngles (p : PentagonInCircle) : ℝ := sorry

/-- Theorem: The sum of central angles in a pentagon inscribed in a circle is 360° -/
theorem sum_of_central_angles_is_360 (p : PentagonInCircle) : 
  sumOfCentralAngles p = 360 := by sorry

end sum_of_central_angles_is_360_l3469_346995


namespace tile_border_ratio_l3469_346987

/-- Represents the arrangement of tiles in a square garden -/
structure TileArrangement where
  n : ℕ               -- Number of tiles along one side of the garden
  s : ℝ               -- Side length of each tile in meters
  d : ℝ               -- Width of the border around each tile in meters
  h_positive_s : 0 < s
  h_positive_d : 0 < d

/-- The theorem stating the ratio of border width to tile side length -/
theorem tile_border_ratio (arr : TileArrangement) (h_n : arr.n = 30) 
  (h_coverage : (arr.n^2 * arr.s^2) / ((arr.n * arr.s + 2 * arr.n * arr.d)^2) = 0.81) :
  arr.d / arr.s = 1 / 18 := by
  sorry

end tile_border_ratio_l3469_346987


namespace tobias_swims_3000_meters_l3469_346991

/-- The number of meters Tobias swims in 3 hours with regular pauses -/
def tobias_swim_distance : ℕ :=
  let total_time : ℕ := 3 * 60  -- 3 hours in minutes
  let swim_pause_cycle : ℕ := 25 + 5  -- 25 min swim + 5 min pause
  let num_cycles : ℕ := total_time / swim_pause_cycle
  let total_swim_time : ℕ := num_cycles * 25  -- Total swimming time in minutes
  let meters_per_5min : ℕ := 100  -- Swims 100 meters every 5 minutes
  total_swim_time / 5 * meters_per_5min

/-- Theorem stating that Tobias swims 3000 meters -/
theorem tobias_swims_3000_meters : tobias_swim_distance = 3000 := by
  sorry

#eval tobias_swim_distance  -- This should output 3000

end tobias_swims_3000_meters_l3469_346991


namespace crayons_given_away_l3469_346916

theorem crayons_given_away (initial_crayons remaining_crayons : ℕ) 
  (h1 : initial_crayons = 106)
  (h2 : remaining_crayons = 52) : 
  initial_crayons - remaining_crayons = 54 := by
  sorry

end crayons_given_away_l3469_346916


namespace sum_of_numbers_greater_than_1_1_l3469_346932

def numbers : List ℚ := [1.4, 9/10, 1.2, 0.5, 13/10]

theorem sum_of_numbers_greater_than_1_1 : 
  (numbers.filter (λ x => x > 1.1)).sum = 3.9 := by
  sorry

end sum_of_numbers_greater_than_1_1_l3469_346932


namespace sequence_value_l3469_346963

theorem sequence_value (n : ℕ) (a : ℕ → ℕ) : 
  (∀ k, a k = 3 * k + 4) → a n = 13 → n = 3 := by
sorry

end sequence_value_l3469_346963


namespace two_digit_number_interchange_l3469_346928

theorem two_digit_number_interchange (x y : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ x - y = 8 → 
  (10 * x + y) - (10 * y + x) = 72 := by
  sorry

end two_digit_number_interchange_l3469_346928


namespace woodworker_solution_l3469_346925

/-- Represents the number of furniture items made by a woodworker. -/
structure FurnitureCount where
  chairs : ℕ
  tables : ℕ
  cabinets : ℕ

/-- Calculates the total number of legs used for a given furniture count. -/
def totalLegs (f : FurnitureCount) : ℕ :=
  4 * f.chairs + 4 * f.tables + 2 * f.cabinets

/-- The woodworker's furniture count satisfies the given conditions. -/
def isSolution (f : FurnitureCount) : Prop :=
  f.chairs = 6 ∧ f.cabinets = 4 ∧ totalLegs f = 80

theorem woodworker_solution :
  ∃ f : FurnitureCount, isSolution f ∧ f.tables = 12 := by
  sorry

end woodworker_solution_l3469_346925


namespace toms_age_l3469_346923

/-- Tom's age problem -/
theorem toms_age (s t : ℕ) : 
  t = 2 * s - 1 →  -- Tom's age is 1 year less than twice his sister's age
  t + s = 14 →     -- The sum of their ages is 14 years
  t = 9            -- Tom's age is 9 years
:= by sorry

end toms_age_l3469_346923


namespace smallest_non_factor_non_prime_l3469_346980

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_non_factor_non_prime : 
  ∃ (n : ℕ), 
    n > 0 ∧ 
    ¬(factorial 30 % n = 0) ∧ 
    ¬(Nat.Prime n) ∧
    (∀ m : ℕ, m > 0 ∧ m < n → 
      (factorial 30 % m = 0) ∨ (Nat.Prime m)) ∧
    n = 961 := by
  sorry

end smallest_non_factor_non_prime_l3469_346980


namespace inequality_ordering_l3469_346911

theorem inequality_ordering (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end inequality_ordering_l3469_346911


namespace total_coins_is_21_l3469_346949

/-- Represents the coin distribution pattern between Pete and Paul -/
def coin_distribution (x : ℕ) : Prop :=
  ∃ (paul_coins : ℕ) (pete_coins : ℕ),
    paul_coins = x ∧
    pete_coins = 6 * x ∧
    pete_coins = x * (x + 1) * (x + 2) / 6

/-- The total number of coins is 21 -/
theorem total_coins_is_21 : ∃ (x : ℕ), coin_distribution x ∧ x + 6 * x = 21 := by
  sorry

end total_coins_is_21_l3469_346949


namespace volume_cone_from_right_triangle_l3469_346947

/-- The volume of a cone formed by rotating a right triangle around its hypotenuse -/
theorem volume_cone_from_right_triangle (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let v := (1 / 3) * π * r^2 * h
  h = 2 ∧ r = 1 → v = (2 * π) / 3 := by sorry

end volume_cone_from_right_triangle_l3469_346947


namespace no_solution_cubic_inequality_l3469_346920

theorem no_solution_cubic_inequality :
  ¬∃ x : ℝ, x ≠ 2 ∧ (x^3 - 8) / (x - 2) < 0 := by
  sorry

end no_solution_cubic_inequality_l3469_346920


namespace chord_length_l3469_346924

/-- The length of the chord cut off by a line on a circle -/
theorem chord_length (x y : ℝ) : 
  let line := {(x, y) : ℝ × ℝ | x - Real.sqrt 2 * y - 1 = 0}
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + (y - 1)^2 = 2}
  let chord := line ∩ circle
  ∃ (a b : ℝ), (a, b) ∈ chord ∧ 
    ∃ (c d : ℝ), (c, d) ∈ chord ∧ 
      (a - c)^2 + (b - d)^2 = (4 * Real.sqrt 3 / 3)^2 :=
sorry

end chord_length_l3469_346924


namespace fraction_simplification_l3469_346983

theorem fraction_simplification : 1000^2 / (252^2 - 248^2) = 500 := by
  sorry

end fraction_simplification_l3469_346983


namespace sum_reciprocal_pairs_gt_one_l3469_346913

theorem sum_reciprocal_pairs_gt_one (a₁ a₂ a₃ : ℝ) (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1) :
  let S := a₁ + a₂ + a₃
  (a₁^2 / (a₁ - 1) > S) ∧ (a₂^2 / (a₂ - 1) > S) ∧ (a₃^2 / (a₃ - 1) > S) →
  1 / (a₁ + a₂) + 1 / (a₂ + a₃) + 1 / (a₃ + a₁) > 1 := by
  sorry

end sum_reciprocal_pairs_gt_one_l3469_346913


namespace james_cattle_profit_l3469_346937

def cattle_profit (num_cattle : ℕ) (purchase_price : ℕ) (feeding_cost_percentage : ℕ) 
                  (weight_per_cattle : ℕ) (selling_price_per_pound : ℕ) : ℕ :=
  let feeding_cost := purchase_price * feeding_cost_percentage / 100
  let total_cost := purchase_price + feeding_cost
  let selling_price_per_cattle := weight_per_cattle * selling_price_per_pound
  let total_selling_price := num_cattle * selling_price_per_cattle
  total_selling_price - total_cost

theorem james_cattle_profit :
  cattle_profit 100 40000 20 1000 2 = 112000 := by
  sorry

end james_cattle_profit_l3469_346937


namespace inequality_proof_l3469_346917

theorem inequality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a^4 / (4*a^4 + b^4 + c^4)) + (b^4 / (a^4 + 4*b^4 + c^4)) + (c^4 / (a^4 + b^4 + 4*c^4)) ≤ 1/2 :=
by sorry

end inequality_proof_l3469_346917


namespace felix_lifting_capacity_l3469_346951

/-- Felix's lifting capacity problem -/
theorem felix_lifting_capacity 
  (felix_lift_ratio : ℝ) 
  (brother_weight_ratio : ℝ) 
  (brother_lift_ratio : ℝ) 
  (brother_lift_weight : ℝ) 
  (h1 : felix_lift_ratio = 1.5)
  (h2 : brother_weight_ratio = 2)
  (h3 : brother_lift_ratio = 3)
  (h4 : brother_lift_weight = 600) :
  felix_lift_ratio * (brother_lift_weight / brother_lift_ratio / brother_weight_ratio) = 150 := by
  sorry


end felix_lifting_capacity_l3469_346951


namespace train_speed_with_stops_l3469_346922

/-- Proves that a train's average speed with stoppages is half of its speed without stoppages,
    given that it stops for half of each hour. -/
theorem train_speed_with_stops (D : ℝ) (h : D > 0) :
  let speed_without_stops : ℝ := 250
  let stop_ratio : ℝ := 0.5
  let time_without_stops : ℝ := D / speed_without_stops
  let time_with_stops : ℝ := time_without_stops / (1 - stop_ratio)
  let speed_with_stops : ℝ := D / time_with_stops
  speed_with_stops = speed_without_stops * (1 - stop_ratio) := by
sorry

end train_speed_with_stops_l3469_346922


namespace dolls_count_l3469_346956

/-- The number of dolls Jane has -/
def jane_dolls : ℕ := 13

/-- The difference between Jill's and Jane's dolls -/
def doll_difference : ℕ := 6

/-- The total number of dolls Jane and Jill have together -/
def total_dolls : ℕ := jane_dolls + (jane_dolls + doll_difference)

theorem dolls_count : total_dolls = 32 := by sorry

end dolls_count_l3469_346956


namespace median_in_80_84_interval_l3469_346977

/-- Represents the score intervals --/
inductive ScoreInterval
  | interval_65_69
  | interval_70_74
  | interval_75_79
  | interval_80_84
  | interval_85_89
  | interval_90_94

/-- The number of students in each score interval --/
def studentCount (interval : ScoreInterval) : Nat :=
  match interval with
  | .interval_65_69 => 6
  | .interval_70_74 => 10
  | .interval_75_79 => 25
  | .interval_80_84 => 30
  | .interval_85_89 => 20
  | .interval_90_94 => 10

/-- The total number of students --/
def totalStudents : Nat := 101

/-- The position of the median in the dataset --/
def medianPosition : Nat := (totalStudents + 1) / 2

/-- Theorem stating that the median score is in the 80-84 interval --/
theorem median_in_80_84_interval :
  ∃ k, k ≤ medianPosition ∧
       k > (studentCount ScoreInterval.interval_90_94 +
            studentCount ScoreInterval.interval_85_89) ∧
       k ≤ (studentCount ScoreInterval.interval_90_94 +
            studentCount ScoreInterval.interval_85_89 +
            studentCount ScoreInterval.interval_80_84) :=
  sorry

end median_in_80_84_interval_l3469_346977


namespace problem_solution_l3469_346989

theorem problem_solution : (29.7 + 83.45) - 0.3 = 112.85 := by
  sorry

end problem_solution_l3469_346989


namespace trig_identity_proof_l3469_346982

theorem trig_identity_proof : 
  (Real.sin (47 * π / 180) - Real.sin (17 * π / 180) * Real.cos (30 * π / 180)) / 
  Real.sin (73 * π / 180) = 1 / 2 := by
sorry

end trig_identity_proof_l3469_346982


namespace dandelion_seed_percentage_l3469_346966

/-- Represents the number of sunflowers Carla has -/
def num_sunflowers : ℕ := 6

/-- Represents the number of dandelions Carla has -/
def num_dandelions : ℕ := 8

/-- Represents the number of seeds per sunflower -/
def seeds_per_sunflower : ℕ := 9

/-- Represents the number of seeds per dandelion -/
def seeds_per_dandelion : ℕ := 12

/-- Calculates the total number of seeds from sunflowers -/
def total_sunflower_seeds : ℕ := num_sunflowers * seeds_per_sunflower

/-- Calculates the total number of seeds from dandelions -/
def total_dandelion_seeds : ℕ := num_dandelions * seeds_per_dandelion

/-- Calculates the total number of seeds -/
def total_seeds : ℕ := total_sunflower_seeds + total_dandelion_seeds

/-- Theorem: The percentage of seeds from dandelions is 64% -/
theorem dandelion_seed_percentage : 
  (total_dandelion_seeds : ℚ) / (total_seeds : ℚ) * 100 = 64 := by
  sorry

end dandelion_seed_percentage_l3469_346966


namespace even_function_quadratic_behavior_l3469_346957

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 3 * m * x + 3

-- State the theorem
theorem even_function_quadratic_behavior :
  ∀ m : ℝ, (∀ x : ℝ, f m x = f m (-x)) →
  (m = 0 ∧
   ∃ c : ℝ, c ∈ (Set.Ioo (-4) 2) ∧
   (∀ x y : ℝ, x ∈ (Set.Ioo (-4) c) → y ∈ (Set.Ioo (-4) c) → x < y → f m x < f m y) ∧
   (∀ x y : ℝ, x ∈ (Set.Ioo c 2) → y ∈ (Set.Ioo c 2) → x < y → f m x > f m y)) :=
by sorry

end even_function_quadratic_behavior_l3469_346957


namespace men_working_with_boys_l3469_346902

-- Define the work done by one man per day
def work_man : ℚ := 1 / 48

-- Define the work done by one boy per day
def work_boy : ℚ := 5 / 96

-- Define the total work to be done
def total_work : ℚ := 1

theorem men_working_with_boys : ℕ :=
  let men_count : ℕ := 1
  have h1 : 2 * work_man + 4 * work_boy = total_work / 4 := by sorry
  have h2 : men_count * work_man + 6 * work_boy = total_work / 3 := by sorry
  have h3 : 2 * work_boy = 5 * work_man := by sorry
  men_count

end men_working_with_boys_l3469_346902


namespace ann_initial_blocks_l3469_346975

/-- Given that Ann finds 44 blocks and ends with 53 blocks, prove that she initially had 9 blocks. -/
theorem ann_initial_blocks (found : ℕ) (final : ℕ) (h1 : found = 44) (h2 : final = 53) :
  final - found = 9 := by sorry

end ann_initial_blocks_l3469_346975


namespace total_correct_answers_l3469_346971

/-- Given a math test with 40 questions where 75% are answered correctly,
    and an English test with 50 questions where 98% are answered correctly,
    the total number of correctly answered questions is 79. -/
theorem total_correct_answers
  (math_questions : ℕ)
  (math_percentage : ℚ)
  (english_questions : ℕ)
  (english_percentage : ℚ)
  (h1 : math_questions = 40)
  (h2 : math_percentage = 75 / 100)
  (h3 : english_questions = 50)
  (h4 : english_percentage = 98 / 100) :
  ⌊math_questions * math_percentage⌋ + ⌊english_questions * english_percentage⌋ = 79 :=
by sorry

end total_correct_answers_l3469_346971


namespace inequality_proof_l3469_346973

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a = 1) : 
  a / Real.sqrt (a^2 + 1) + b / Real.sqrt (b^2 + 1) + c / Real.sqrt (c^2 + 1) ≤ 3/2 := by
sorry

end inequality_proof_l3469_346973


namespace triangle_perimeter_l3469_346974

theorem triangle_perimeter : ∀ (a b c : ℝ),
  a = 4 ∧ b = 8 ∧ c^2 - 14*c + 40 = 0 ∧
  a + b > c ∧ a + c > b ∧ b + c > a →
  a + b + c = 22 :=
by
  sorry

end triangle_perimeter_l3469_346974


namespace sand_bags_problem_l3469_346967

/-- Given that each bag has a capacity of 65 pounds and 12 bags are needed,
    prove that the total pounds of sand is 780. -/
theorem sand_bags_problem (bag_capacity : ℕ) (num_bags : ℕ) 
    (h1 : bag_capacity = 65) (h2 : num_bags = 12) : 
    bag_capacity * num_bags = 780 := by
  sorry

end sand_bags_problem_l3469_346967


namespace range_of_a_l3469_346976

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → |x + 1/x| > |a - 2| + 1) → 
  1 < a ∧ a < 3 :=
by sorry

end range_of_a_l3469_346976


namespace math_books_count_l3469_346990

/-- The number of math books on a shelf with 100 total books, 32 history books, and 25 geography books. -/
def math_books (total : ℕ) (history : ℕ) (geography : ℕ) : ℕ :=
  total - history - geography

/-- Theorem stating that there are 43 math books on the shelf. -/
theorem math_books_count : math_books 100 32 25 = 43 := by
  sorry

end math_books_count_l3469_346990


namespace even_sum_sufficient_not_necessary_l3469_346926

/-- A function is even if f(-x) = f(x) for all x in its domain --/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The sum of two functions --/
def FunctionSum (f g : ℝ → ℝ) : ℝ → ℝ := fun x ↦ f x + g x

theorem even_sum_sufficient_not_necessary :
  (∀ f g : ℝ → ℝ, IsEven f ∧ IsEven g → IsEven (FunctionSum f g)) ∧
  (∃ f g : ℝ → ℝ, IsEven (FunctionSum f g) ∧ (¬IsEven f ∨ ¬IsEven g)) := by
  sorry

#check even_sum_sufficient_not_necessary

end even_sum_sufficient_not_necessary_l3469_346926


namespace ratio_and_equation_imply_c_value_l3469_346908

theorem ratio_and_equation_imply_c_value 
  (a b c : ℝ) 
  (h1 : ∃ (k : ℝ), a = 2*k ∧ b = 3*k ∧ c = 7*k) 
  (h2 : a - b + 3 = c - 2*b) : 
  c = 21/2 := by
sorry

end ratio_and_equation_imply_c_value_l3469_346908


namespace cube_sum_problem_l3469_346930

theorem cube_sum_problem (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) :
  x^3 + y^3 = 640 := by
  sorry

end cube_sum_problem_l3469_346930


namespace break_even_books_l3469_346998

/-- Represents the fixed cost of making books -/
def fixed_cost : ℝ := 50000

/-- Represents the marketing cost per book -/
def marketing_cost_per_book : ℝ := 4

/-- Represents the selling price per book -/
def selling_price_per_book : ℝ := 9

/-- Calculates the total cost for a given number of books -/
def total_cost (num_books : ℝ) : ℝ :=
  fixed_cost + marketing_cost_per_book * num_books

/-- Calculates the revenue for a given number of books -/
def revenue (num_books : ℝ) : ℝ :=
  selling_price_per_book * num_books

/-- Theorem: The number of books needed to break even is 10000 -/
theorem break_even_books : 
  ∃ (x : ℝ), x = 10000 ∧ total_cost x = revenue x :=
by sorry

end break_even_books_l3469_346998


namespace cone_vertex_angle_l3469_346933

/-- Given a cone whose lateral surface development has a central angle of α radians,
    the vertex angle of its axial section is equal to 2 * arcsin(α / (2π)). -/
theorem cone_vertex_angle (α : ℝ) (h : 0 < α ∧ α < 2 * Real.pi) :
  let vertex_angle := 2 * Real.arcsin (α / (2 * Real.pi))
  vertex_angle = 2 * Real.arcsin (α / (2 * Real.pi)) := by
  sorry

end cone_vertex_angle_l3469_346933


namespace caterpillar_eggs_hatched_l3469_346912

theorem caterpillar_eggs_hatched (initial_caterpillars : ℕ) (caterpillars_left : ℕ) (final_caterpillars : ℕ) 
  (h1 : initial_caterpillars = 14)
  (h2 : caterpillars_left = 8)
  (h3 : final_caterpillars = 10) :
  initial_caterpillars + (caterpillars_left + final_caterpillars - initial_caterpillars) - caterpillars_left = final_caterpillars :=
by sorry

end caterpillar_eggs_hatched_l3469_346912


namespace smallest_number_in_sequence_l3469_346906

theorem smallest_number_in_sequence (x : ℝ) : 
  let second := 4 * x
  let third := 2 * x
  (x + second + third) / 3 = 77 →
  x = 33 := by sorry

end smallest_number_in_sequence_l3469_346906


namespace expression_simplification_l3469_346900

theorem expression_simplification (b : ℝ) : 
  ((3 * b + 10 - 5 * b^2) / 5) = -b^2 + (3 * b / 5) + 2 := by
  sorry

end expression_simplification_l3469_346900


namespace complex_modulus_product_l3469_346904

theorem complex_modulus_product : Complex.abs ((10 - 7*I) * (9 + 11*I)) = Real.sqrt 30098 := by
  sorry

end complex_modulus_product_l3469_346904


namespace max_candies_for_class_l3469_346915

def max_candies (num_students : ℕ) (mean_candies : ℕ) (min_candies : ℕ) : ℕ :=
  (num_students * mean_candies) - (min_candies * (num_students - 1))

theorem max_candies_for_class (num_students : ℕ) (mean_candies : ℕ) (min_candies : ℕ) 
  (h1 : num_students = 24)
  (h2 : mean_candies = 7)
  (h3 : min_candies = 3) :
  max_candies num_students mean_candies min_candies = 99 :=
by
  sorry

#eval max_candies 24 7 3

end max_candies_for_class_l3469_346915


namespace complex_modulus_problem_l3469_346968

theorem complex_modulus_problem (z : ℂ) (h : z * (2 + Complex.I) = 10 - 5 * Complex.I) :
  Complex.abs z = 5 := by
  sorry

end complex_modulus_problem_l3469_346968


namespace gcd_pow_minus_one_l3469_346907

theorem gcd_pow_minus_one (a n m : ℕ) (ha : a > 0) :
  Nat.gcd (a^n - 1) (a^m - 1) = a^(Nat.gcd n m) - 1 :=
by sorry

end gcd_pow_minus_one_l3469_346907


namespace hyperbola_sum_l3469_346946

/-- Proves that for a hyperbola with given parameters, the sum of h, k, a, and b equals 6 + 2√10 -/
theorem hyperbola_sum (h k : ℝ) (focus_y vertex_y : ℝ) : 
  h = 1 → 
  k = 2 → 
  focus_y = 9 → 
  vertex_y = -1 → 
  let a := |k - vertex_y|
  let c := |k - focus_y|
  let b := Real.sqrt (c^2 - a^2)
  h + k + a + b = 6 + 2 * Real.sqrt 10 := by
sorry

end hyperbola_sum_l3469_346946


namespace discount_difference_l3469_346964

def original_bill : ℝ := 15000

def single_discount_rate : ℝ := 0.3
def first_successive_discount_rate : ℝ := 0.25
def second_successive_discount_rate : ℝ := 0.06

def single_discount_amount : ℝ := original_bill * (1 - single_discount_rate)
def successive_discount_amount : ℝ := original_bill * (1 - first_successive_discount_rate) * (1 - second_successive_discount_rate)

theorem discount_difference :
  successive_discount_amount - single_discount_amount = 75 := by sorry

end discount_difference_l3469_346964


namespace quadratic_transformations_integer_roots_l3469_346979

/-- 
Given a quadratic equation x^2 + px + q = 0, where p and q are integers,
this function returns true if the equation has integer roots.
-/
def has_integer_roots (p q : ℤ) : Prop :=
  ∃ x y : ℤ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x ≠ y

/-- 
This theorem states that there exist initial integer values for p and q
such that the quadratic equation x^2 + px + q = 0 and its nine transformations
(where p and q are increased by 1 each time) all have integer roots.
-/
theorem quadratic_transformations_integer_roots :
  ∃ p q : ℤ, 
    (∀ i : ℕ, i ≤ 9 → has_integer_roots (p + i) (q + i)) :=
by sorry

end quadratic_transformations_integer_roots_l3469_346979


namespace division_of_fractions_l3469_346938

theorem division_of_fractions : (5 : ℚ) / 6 / (2 + 2 / 3) = 5 / 16 := by
  sorry

end division_of_fractions_l3469_346938


namespace sequence_a_properties_l3469_346972

def sequence_a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * sequence_a (n + 1) - sequence_a n

theorem sequence_a_properties :
  (∀ n : ℕ, ∃ k : ℤ, sequence_a n = k) ∧
  (∀ n : ℕ, 3 ∣ sequence_a n ↔ 3 ∣ n) := by
  sorry

end sequence_a_properties_l3469_346972


namespace linear_function_intersection_l3469_346941

/-- The linear function that intersects with two given lines -/
def linear_function (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

theorem linear_function_intersection : 
  ∃ k b : ℝ, 
    (linear_function k b 4 = -4 + 6) ∧ 
    (linear_function k b (1 + 1) = 1) ∧
    (∀ x : ℝ, linear_function k b x = (1/2) * x) := by
  sorry

end linear_function_intersection_l3469_346941


namespace ball_distribution_theorem_l3469_346960

/-- The number of ways to distribute 10 colored balls into two boxes -/
def distribute_balls : ℕ :=
  Nat.choose 10 4

/-- The total number of balls -/
def total_balls : ℕ := 10

/-- The number of red balls -/
def red_balls : ℕ := 5

/-- The number of white balls -/
def white_balls : ℕ := 3

/-- The number of green balls -/
def green_balls : ℕ := 2

/-- The capacity of the smaller box -/
def small_box_capacity : ℕ := 4

/-- The capacity of the larger box -/
def large_box_capacity : ℕ := 6

theorem ball_distribution_theorem :
  distribute_balls = 210 ∧
  total_balls = red_balls + white_balls + green_balls ∧
  total_balls = small_box_capacity + large_box_capacity :=
sorry

end ball_distribution_theorem_l3469_346960


namespace right_triangle_inequality_l3469_346996

theorem right_triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (pythagorean : a^2 + b^2 = c^2) : (a + b) / Real.sqrt 2 ≤ c := by
  sorry

end right_triangle_inequality_l3469_346996


namespace carlos_class_size_l3469_346944

theorem carlos_class_size :
  ∃! b : ℕ, 80 < b ∧ b < 150 ∧
  ∃ k₁ : ℕ, b = 3 * k₁ - 2 ∧
  ∃ k₂ : ℕ, b = 4 * k₂ - 3 ∧
  ∃ k₃ : ℕ, b = 5 * k₃ - 4 ∧
  b = 121 := by
sorry

end carlos_class_size_l3469_346944


namespace quadratic_polynomials_theorem_l3469_346935

theorem quadratic_polynomials_theorem (a b c d : ℝ) : 
  let f (x : ℝ) := x^2 + a*x + b
  let g (x : ℝ) := x^2 + c*x + d
  (∀ x, f x ≠ g x) →  -- f and g are distinct
  g (-a/2) = 0 →  -- x-coordinate of vertex of f is a root of g
  f (-c/2) = 0 →  -- x-coordinate of vertex of g is a root of f
  f 50 = -50 ∧ g 50 = -50 →  -- f and g intersect at (50, -50)
  (∃ x₁ x₂, ∀ x, f x ≥ f x₁ ∧ g x ≥ g x₂ ∧ f x₁ = g x₂) →  -- minimum value of f is the same as g
  a + c = -200 := by
sorry

end quadratic_polynomials_theorem_l3469_346935


namespace max_distance_sum_l3469_346948

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define a line passing through F₁
def line_through_F₁ (m : ℝ) (x y : ℝ) : Prop :=
  y = m * (x + 1)

-- Define the intersection points A and B
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ ellipse x y ∧ line_through_F₁ m x y}

-- Statement of the theorem
theorem max_distance_sum :
  ∀ (m : ℝ), ∃ (A B : ℝ × ℝ),
    A ∈ intersection_points m ∧
    B ∈ intersection_points m ∧
    A ≠ B ∧
    (∀ (A' B' : ℝ × ℝ),
      A' ∈ intersection_points m →
      B' ∈ intersection_points m →
      A' ≠ B' →
      dist A' F₂ + dist B' F₂ ≤ 5) ∧
    (∃ (m' : ℝ), ∃ (A' B' : ℝ × ℝ),
      A' ∈ intersection_points m' ∧
      B' ∈ intersection_points m' ∧
      A' ≠ B' ∧
      dist A' F₂ + dist B' F₂ = 5) :=
sorry


end max_distance_sum_l3469_346948


namespace solution_implies_a_value_l3469_346954

theorem solution_implies_a_value (x a : ℝ) : x = 1 ∧ 2 * x - a = 0 → a = 2 := by
  sorry

end solution_implies_a_value_l3469_346954
