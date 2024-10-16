import Mathlib

namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1915_191587

theorem imaginary_part_of_z (z : ℂ) : (3 - 4*I)*z = Complex.abs (4 + 3*I) → Complex.im z = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1915_191587


namespace NUMINAMATH_CALUDE_special_hyperbola_equation_l1915_191589

/-- A hyperbola with specific properties -/
structure SpecialHyperbola where
  /-- The center of the hyperbola is at the origin -/
  center_at_origin : True
  /-- The foci of the hyperbola are on the y-axis -/
  foci_on_y_axis : True
  /-- The eccentricity of the hyperbola is √5/2 -/
  eccentricity : ℝ
  eccentricity_value : eccentricity = Real.sqrt 5 / 2
  /-- A circle with radius 2 is tangent to the asymptote of the hyperbola -/
  circle_tangent_to_asymptote : True
  /-- One of the foci is the center of the circle -/
  focus_is_circle_center : True

/-- The equation of the special hyperbola -/
def hyperbola_equation (h : SpecialHyperbola) (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 4 = 1

/-- Theorem stating that the given hyperbola has the specified equation -/
theorem special_hyperbola_equation (h : SpecialHyperbola) :
  ∀ x y : ℝ, hyperbola_equation h x y ↔ y^2 / 16 - x^2 / 4 = 1 := by sorry

end NUMINAMATH_CALUDE_special_hyperbola_equation_l1915_191589


namespace NUMINAMATH_CALUDE_function_min_value_l1915_191530

/-- Given a function f(x) = (1/3)x³ - x + m with a maximum value of 1,
    prove that its minimum value is -1/3 -/
theorem function_min_value 
  (f : ℝ → ℝ) 
  (m : ℝ) 
  (h1 : ∀ x, f x = (1/3) * x^3 - x + m) 
  (h2 : ∃ x, f x = 1 ∧ ∀ y, f y ≤ 1) : 
  ∃ x, f x = -1/3 ∧ ∀ y, f y ≥ -1/3 :=
sorry

end NUMINAMATH_CALUDE_function_min_value_l1915_191530


namespace NUMINAMATH_CALUDE_wedding_champagne_bottles_l1915_191545

/-- The number of wedding guests -/
def num_guests : ℕ := 120

/-- The number of glasses of champagne per guest -/
def glasses_per_guest : ℕ := 2

/-- The number of servings per bottle of champagne -/
def servings_per_bottle : ℕ := 6

/-- The number of bottles of champagne needed for the wedding toast -/
def bottles_needed : ℕ := (num_guests * glasses_per_guest) / servings_per_bottle

theorem wedding_champagne_bottles : bottles_needed = 40 := by
  sorry

end NUMINAMATH_CALUDE_wedding_champagne_bottles_l1915_191545


namespace NUMINAMATH_CALUDE_monotonic_function_range_l1915_191528

def f (a x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ z, a ≤ z ∧ z ≤ b → f z = f x)

theorem monotonic_function_range (a : ℝ) :
  monotonic_on (f a) (-1) 2 → a ≤ -1 ∨ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_function_range_l1915_191528


namespace NUMINAMATH_CALUDE_square_sum_xy_l1915_191592

theorem square_sum_xy (x y : ℝ) 
  (h1 : 2 * x * (x + y) = 72) 
  (h2 : 3 * y * (x + y) = 108) : 
  (x + y)^2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_xy_l1915_191592


namespace NUMINAMATH_CALUDE_bookshelf_length_is_24_l1915_191585

/-- The length of one span in centimeters -/
def span_length : ℝ := 12

/-- The number of spans in the shorter side of the bookshelf -/
def bookshelf_spans : ℝ := 2

/-- The length of the shorter side of the bookshelf in centimeters -/
def bookshelf_length : ℝ := span_length * bookshelf_spans

theorem bookshelf_length_is_24 : bookshelf_length = 24 := by
  sorry

end NUMINAMATH_CALUDE_bookshelf_length_is_24_l1915_191585


namespace NUMINAMATH_CALUDE_smallest_integer_sum_product_squares_l1915_191543

theorem smallest_integer_sum_product_squares :
  ∃ (a : ℕ), a > 0 ∧ 
  (∃ (b : ℕ), 10 + a = b^2) ∧ 
  (∃ (c : ℕ), 10 * a = c^2) ∧
  (∀ (x : ℕ), x > 0 ∧ x < a → 
    (¬∃ (y : ℕ), 10 + x = y^2) ∨ 
    (¬∃ (z : ℕ), 10 * x = z^2)) ∧
  a = 90 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_sum_product_squares_l1915_191543


namespace NUMINAMATH_CALUDE_money_lending_problem_l1915_191529

theorem money_lending_problem (total : ℝ) (rate_A rate_B : ℝ) (time : ℝ) (interest_diff : ℝ) :
  total = 10000 ∧ 
  rate_A = 15 / 100 ∧ 
  rate_B = 18 / 100 ∧ 
  time = 2 ∧ 
  interest_diff = 360 →
  ∃ (amount_A amount_B : ℝ),
    amount_A + amount_B = total ∧
    amount_A * rate_A * time = amount_B * rate_B * time + interest_diff ∧
    amount_B = 4000 := by
  sorry

end NUMINAMATH_CALUDE_money_lending_problem_l1915_191529


namespace NUMINAMATH_CALUDE_remaining_volume_of_cube_l1915_191588

/-- The volume of the remaining part of a cube after cutting out smaller cubes --/
theorem remaining_volume_of_cube (edge_length : ℝ) (small_edge_length : ℝ) (num_small_cubes : ℕ) :
  edge_length = 9 →
  small_edge_length = 3 →
  num_small_cubes = 12 →
  edge_length ^ 3 - num_small_cubes * small_edge_length ^ 3 = 405 :=
by sorry

end NUMINAMATH_CALUDE_remaining_volume_of_cube_l1915_191588


namespace NUMINAMATH_CALUDE_election_total_votes_l1915_191551

/-- Represents the total number of votes in the election -/
def total_votes : ℕ := sorry

/-- Represents the vote percentage for Candidate A -/
def candidate_a_percentage : ℚ := 30 / 100

/-- Represents the vote percentage for Candidate B -/
def candidate_b_percentage : ℚ := 25 / 100

/-- Represents the vote difference between Candidate A and Candidate B -/
def vote_difference_a_b : ℕ := 1800

theorem election_total_votes : 
  (candidate_a_percentage - candidate_b_percentage) * total_votes = vote_difference_a_b ∧ 
  total_votes = 36000 := by sorry

end NUMINAMATH_CALUDE_election_total_votes_l1915_191551


namespace NUMINAMATH_CALUDE_divisors_of_2_pow_48_minus_1_l1915_191525

theorem divisors_of_2_pow_48_minus_1 :
  ∃! (a b : ℕ), 60 ≤ a ∧ a < b ∧ b ≤ 70 ∧
  (2^48 - 1) % a = 0 ∧ (2^48 - 1) % b = 0 ∧
  a = 63 ∧ b = 65 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_2_pow_48_minus_1_l1915_191525


namespace NUMINAMATH_CALUDE_orchard_trees_l1915_191596

theorem orchard_trees (total : ℕ) (pure_fuji : ℕ) (cross_pollinated : ℕ) (pure_gala : ℕ) :
  pure_gala = 39 →
  cross_pollinated = (total : ℚ) * (1 / 10) →
  pure_fuji = (total : ℚ) * (3 / 4) →
  pure_fuji + pure_gala + cross_pollinated = total →
  pure_fuji + cross_pollinated = 221 := by
sorry

end NUMINAMATH_CALUDE_orchard_trees_l1915_191596


namespace NUMINAMATH_CALUDE_emma_running_time_l1915_191554

theorem emma_running_time (emma_time : ℝ) (fernando_time : ℝ) : 
  fernando_time = 2 * emma_time →
  emma_time + fernando_time = 60 →
  emma_time = 20 := by
sorry

end NUMINAMATH_CALUDE_emma_running_time_l1915_191554


namespace NUMINAMATH_CALUDE_bike_tractor_speed_ratio_l1915_191579

/-- Given the conditions of the problem, prove that the ratio of the speed of the bike to the speed of the tractor is 2:1 -/
theorem bike_tractor_speed_ratio :
  ∀ (car_speed bike_speed tractor_speed : ℝ),
  car_speed = (9/5) * bike_speed →
  tractor_speed = 575 / 23 →
  car_speed = 540 / 6 →
  ∃ (k : ℝ), bike_speed = k * tractor_speed →
  bike_speed / tractor_speed = 2 := by
sorry

end NUMINAMATH_CALUDE_bike_tractor_speed_ratio_l1915_191579


namespace NUMINAMATH_CALUDE_remainder_of_N_mod_45_l1915_191508

def concatenate_integers (n : ℕ) : ℕ :=
  -- Definition of concatenating integers from 1 to n
  sorry

def N : ℕ := concatenate_integers 44

theorem remainder_of_N_mod_45 : N % 45 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_N_mod_45_l1915_191508


namespace NUMINAMATH_CALUDE_set_operations_l1915_191542

def A : Set ℝ := {x | 2 * x - 8 < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 6}

theorem set_operations :
  (A ∩ B = {x : ℝ | 0 < x ∧ x < 4}) ∧
  ((Aᶜ ∪ B) = {x : ℝ | 0 < x}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l1915_191542


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l1915_191560

theorem scientific_notation_equivalence :
  216000 = 2.16 * (10 ^ 5) :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l1915_191560


namespace NUMINAMATH_CALUDE_function_decreasing_iff_a_in_range_l1915_191571

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * (a - 3) * x + 1

-- Define the property of being decreasing on an interval
def is_decreasing_on (f : ℝ → ℝ) (lb : ℝ) : Prop :=
  ∀ x y, lb ≤ x → x < y → f y < f x

-- State the theorem
theorem function_decreasing_iff_a_in_range :
  ∀ a : ℝ, (is_decreasing_on (f a) (-2)) ↔ a ∈ Set.Icc (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_function_decreasing_iff_a_in_range_l1915_191571


namespace NUMINAMATH_CALUDE_distinct_subsets_remain_distinct_after_removal_l1915_191512

universe u

theorem distinct_subsets_remain_distinct_after_removal 
  {α : Type u} [DecidableEq α] (A : Finset α) (n : ℕ) 
  (subsets : Fin n → Finset α)
  (h_subset : ∀ i, (subsets i) ⊆ A)
  (h_distinct : ∀ i j, i ≠ j → subsets i ≠ subsets j) :
  ∃ a ∈ A, ∀ i j, i ≠ j → 
    (subsets i).erase a ≠ (subsets j).erase a :=
sorry

end NUMINAMATH_CALUDE_distinct_subsets_remain_distinct_after_removal_l1915_191512


namespace NUMINAMATH_CALUDE_new_selling_price_l1915_191516

theorem new_selling_price (old_price : ℝ) (old_profit_rate new_profit_rate : ℝ) : 
  old_price = 88 →
  old_profit_rate = 0.1 →
  new_profit_rate = 0.15 →
  let cost := old_price / (1 + old_profit_rate)
  let new_price := cost * (1 + new_profit_rate)
  new_price = 92 := by
sorry

end NUMINAMATH_CALUDE_new_selling_price_l1915_191516


namespace NUMINAMATH_CALUDE_min_distance_b_to_c_l1915_191556

/-- Calculates the minimum distance between points B and C given boat and river conditions -/
theorem min_distance_b_to_c 
  (boat_speed : ℝ) 
  (downstream_current : ℝ) 
  (upstream_current : ℝ) 
  (time_a_to_b : ℝ) 
  (max_time_b_to_c : ℝ) 
  (h1 : boat_speed = 42) 
  (h2 : downstream_current = 5) 
  (h3 : upstream_current = 7) 
  (h4 : time_a_to_b = 1 + 10/60) 
  (h5 : max_time_b_to_c = 2.5) : 
  ∃ (min_distance : ℝ), min_distance = 87.5 := by
  sorry

#check min_distance_b_to_c

end NUMINAMATH_CALUDE_min_distance_b_to_c_l1915_191556


namespace NUMINAMATH_CALUDE_six_lines_intersection_possibilities_l1915_191526

/-- Represents a line in a plane -/
structure Line

/-- Represents an intersection point of two lines -/
structure IntersectionPoint

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  lines : Finset Line
  intersections : Finset IntersectionPoint
  no_triple_intersections : ∀ p : IntersectionPoint, p ∈ intersections → 
    (∃! l1 l2 : Line, l1 ∈ lines ∧ l2 ∈ lines ∧ l1 ≠ l2 ∧ p ∈ intersections)

theorem six_lines_intersection_possibilities 
  (config : LineConfiguration) 
  (h_six_lines : config.lines.card = 6) :
  (∃ config' : LineConfiguration, config'.lines = config.lines ∧ config'.intersections.card = 12) ∧
  ¬(∃ config' : LineConfiguration, config'.lines = config.lines ∧ config'.intersections.card = 16) :=
sorry

end NUMINAMATH_CALUDE_six_lines_intersection_possibilities_l1915_191526


namespace NUMINAMATH_CALUDE_yellow_straight_probability_l1915_191562

structure Garden where
  roses : ℝ
  daffodils : ℝ
  tulips : ℝ
  green_prob : ℝ
  straight_prob : ℝ
  rose_straight_prob : ℝ
  daffodil_curved_prob : ℝ
  tulip_straight_prob : ℝ

def is_valid_garden (g : Garden) : Prop :=
  g.roses + g.daffodils + g.tulips = 1 ∧
  g.roses = 1/4 ∧
  g.daffodils = 1/2 ∧
  g.tulips = 1/4 ∧
  g.green_prob = 2/3 ∧
  g.straight_prob = 1/2 ∧
  g.rose_straight_prob = 1/6 ∧
  g.daffodil_curved_prob = 1/3 ∧
  g.tulip_straight_prob = 1/8

theorem yellow_straight_probability (g : Garden) 
  (h : is_valid_garden g) : 
  (1 - g.green_prob) * g.straight_prob = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_yellow_straight_probability_l1915_191562


namespace NUMINAMATH_CALUDE_count_solutions_l1915_191553

def is_solution (m n r : ℕ+) : Prop :=
  m * n + n * r + m * r = 2 * (m + n + r)

theorem count_solutions : 
  ∃! (solutions : Finset (ℕ+ × ℕ+ × ℕ+)), 
    (∀ (m n r : ℕ+), (m, n, r) ∈ solutions ↔ is_solution m n r) ∧ 
    solutions.card = 7 :=
sorry

end NUMINAMATH_CALUDE_count_solutions_l1915_191553


namespace NUMINAMATH_CALUDE_boat_length_boat_length_is_three_l1915_191595

/-- The length of a boat given its breadth, sinking depth, and the mass of a man. -/
theorem boat_length (breadth : ℝ) (sinking_depth : ℝ) (man_mass : ℝ) 
  (water_density : ℝ) (gravity : ℝ) : ℝ :=
  let volume := man_mass * gravity / (water_density * gravity)
  volume / (breadth * sinking_depth)

/-- Proof that the length of the boat is 3 meters given specific conditions. -/
theorem boat_length_is_three :
  boat_length 2 0.01 60 1000 9.81 = 3 := by
  sorry

end NUMINAMATH_CALUDE_boat_length_boat_length_is_three_l1915_191595


namespace NUMINAMATH_CALUDE_game_result_l1915_191523

def f (n : ℕ) : ℕ :=
  if n ^ 2 = n then 8
  else if n % 3 = 0 then 4
  else if n % 2 = 0 then 1
  else 0

def allie_rolls : List ℕ := [3, 4, 6, 1]
def betty_rolls : List ℕ := [4, 2, 5, 1]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem game_result : total_points allie_rolls * total_points betty_rolls = 117 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l1915_191523


namespace NUMINAMATH_CALUDE_correct_calculation_l1915_191534

theorem correct_calculation (x : ℚ) : x * 15 = 45 → x * 5 * 10 = 150 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1915_191534


namespace NUMINAMATH_CALUDE_circle_equation_l1915_191580

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def totalStudents : ℕ := 2500
def firstGradeStudents : ℕ := 1000
def secondGradeStudents : ℕ := 900
def thirdGradeStudents : ℕ := 600
def sampleSize : ℕ := 100

def centerA : Point := ⟨1, -1⟩

/-- The angle between line AB and AC in degrees -/
def angleBACDegrees : ℝ := 120

/-- The theorem to be proved -/
theorem circle_equation (a b : ℝ) (l : Line) (c : Circle) :
  (totalStudents = firstGradeStudents + secondGradeStudents + thirdGradeStudents) →
  (a * firstGradeStudents = sampleSize * 1000) →
  (b * thirdGradeStudents = sampleSize * 600) →
  (l.a = a ∧ l.b = b ∧ l.c = 8) →
  (c.center = centerA) →
  (∃ (B C : Point), B ≠ C ∧
    l.a * B.x + l.b * B.y + l.c = 0 ∧
    l.a * C.x + l.b * C.y + l.c = 0 ∧
    (B.x - c.center.x)^2 + (B.y - c.center.y)^2 = c.radius^2 ∧
    (C.x - c.center.x)^2 + (C.y - c.center.y)^2 = c.radius^2) →
  (angleBACDegrees = 120) →
  c.radius^2 = 18/17 := by
sorry


end NUMINAMATH_CALUDE_circle_equation_l1915_191580


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l1915_191582

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem lines_perp_to_plane_are_parallel 
  (a b : Line) (α : Plane) 
  (h1 : perp a α) (h2 : perp b α) : 
  parallel a b :=
sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l1915_191582


namespace NUMINAMATH_CALUDE_race_distance_is_140_l1915_191591

/-- The distance of a race, given the times of two runners and the difference in their finishing positions. -/
def race_distance (time_A time_B : ℕ) (difference : ℕ) : ℕ :=
  let speed_A := 140 / time_A
  let speed_B := 140 / time_B
  140

/-- Theorem stating that the race distance is 140 meters under the given conditions. -/
theorem race_distance_is_140 :
  race_distance 36 45 28 = 140 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_is_140_l1915_191591


namespace NUMINAMATH_CALUDE_cloth_selling_price_l1915_191552

/-- Calculates the total selling price of cloth given the quantity, cost price, and loss per metre -/
def total_selling_price (quantity : ℕ) (cost_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  quantity * (cost_price - loss_per_metre)

/-- Theorem stating that the total selling price of 200 metres of cloth
    with a cost price of Rs. 72 per metre and a loss of Rs. 12 per metre
    is Rs. 12,000 -/
theorem cloth_selling_price :
  total_selling_price 200 72 12 = 12000 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l1915_191552


namespace NUMINAMATH_CALUDE_mikes_typing_speed_reduction_l1915_191559

/-- Calculates the reduction in typing speed given the original speed, document length, and typing time. -/
def typing_speed_reduction (original_speed : ℕ) (document_length : ℕ) (typing_time : ℕ) : ℕ :=
  original_speed - (document_length / typing_time)

/-- Theorem stating that Mike's typing speed reduction is 20 words per minute. -/
theorem mikes_typing_speed_reduction :
  typing_speed_reduction 65 810 18 = 20 := by
  sorry

end NUMINAMATH_CALUDE_mikes_typing_speed_reduction_l1915_191559


namespace NUMINAMATH_CALUDE_specific_circle_distances_l1915_191502

/-- Two circles with given radii and distance between centers -/
structure TwoCircles where
  radius1 : ℝ
  radius2 : ℝ
  center_distance : ℝ

/-- The minimum and maximum distances between points on two circles -/
def circle_distances (c : TwoCircles) : ℝ × ℝ :=
  (c.center_distance - c.radius1 - c.radius2, c.center_distance + c.radius1 + c.radius2)

/-- Theorem stating the minimum and maximum distances for specific circle configuration -/
theorem specific_circle_distances :
  let c : TwoCircles := ⟨2, 3, 8⟩
  circle_distances c = (3, 13) := by sorry

end NUMINAMATH_CALUDE_specific_circle_distances_l1915_191502


namespace NUMINAMATH_CALUDE_same_color_probability_l1915_191535

/-- The probability of drawing two balls of the same color from a bag containing 4 green balls and 8 white balls. -/
theorem same_color_probability (green : ℕ) (white : ℕ) (h1 : green = 4) (h2 : white = 8) :
  let total := green + white
  let p_green := green / total
  let p_white := white / total
  let p_same_color := (p_green * (green - 1) / (total - 1)) + (p_white * (white - 1) / (total - 1))
  p_same_color = 17 / 33 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1915_191535


namespace NUMINAMATH_CALUDE_book_price_percentage_l1915_191521

/-- Given the original price and current price of a book, prove that the current price is 80% of the original price. -/
theorem book_price_percentage (original_price current_price : ℝ) 
  (h1 : original_price = 25)
  (h2 : current_price = 20) :
  current_price / original_price = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_book_price_percentage_l1915_191521


namespace NUMINAMATH_CALUDE_problem_solution_l1915_191575

def f (t : ℝ) : ℝ := t^2003 + 2002*t

theorem problem_solution (x y : ℝ) 
  (h1 : f (x - 1) = -1)
  (h2 : f (y - 2) = 1) : 
  x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1915_191575


namespace NUMINAMATH_CALUDE_rachel_apple_picking_l1915_191572

/-- Rachel's apple picking problem -/
theorem rachel_apple_picking (num_trees : ℕ) (apples_left : ℕ) (initial_apples : ℕ) : 
  num_trees = 3 → apples_left = 9 → initial_apples = 33 → 
  (initial_apples - apples_left) / num_trees = 8 := by
  sorry

end NUMINAMATH_CALUDE_rachel_apple_picking_l1915_191572


namespace NUMINAMATH_CALUDE_intersection_point_sum_l1915_191517

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem intersection_point_sum (h1 : f (-3) = 3) (h2 : f 1 = 3)
  (h3 : ∃! p : ℝ × ℝ, f p.1 = f (p.1 - 4) ∧ f p.1 = p.2) :
  ∃ p : ℝ × ℝ, f p.1 = f (p.1 - 4) ∧ f p.1 = p.2 ∧ p.1 + p.2 = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l1915_191517


namespace NUMINAMATH_CALUDE_max_value_of_f_l1915_191558

/-- The function f(x) = x(1-2x) -/
def f (x : ℝ) := x * (1 - 2 * x)

theorem max_value_of_f :
  ∃ (M : ℝ), M = 1/8 ∧ ∀ x, 0 < x → x < 1/2 → f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1915_191558


namespace NUMINAMATH_CALUDE_smallest_mu_inequality_l1915_191541

theorem smallest_mu_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  ∃ μ : ℝ, (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a^2 + b^2 + c^2 + d^2 ≥ 2*a*b + μ*b*c + 2*c*d) ∧
  (∀ μ' : ℝ, (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a^2 + b^2 + c^2 + d^2 ≥ 2*a*b + μ'*b*c + 2*c*d) → μ' ≥ μ) ∧
  μ = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_mu_inequality_l1915_191541


namespace NUMINAMATH_CALUDE_negation_equivalence_l1915_191593

theorem negation_equivalence (P Q : Prop) :
  ¬(P → ¬Q) ↔ (P ∧ Q) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1915_191593


namespace NUMINAMATH_CALUDE_truck_rental_miles_driven_l1915_191557

theorem truck_rental_miles_driven 
  (rental_fee : ℚ) 
  (charge_per_mile : ℚ) 
  (total_paid : ℚ) 
  (h1 : rental_fee = 2099 / 100)
  (h2 : charge_per_mile = 25 / 100)
  (h3 : total_paid = 9574 / 100) : 
  (total_paid - rental_fee) / charge_per_mile = 299 := by
sorry

#eval (9574 / 100 - 2099 / 100) / (25 / 100)

end NUMINAMATH_CALUDE_truck_rental_miles_driven_l1915_191557


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_log_sum_l1915_191565

theorem arithmetic_geometric_sequence_log_sum (a b c x y z : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z →
  (∃ d : ℝ, b - a = d ∧ c - b = d) →
  (∃ q : ℝ, y / x = q ∧ z / y = q) →
  (b - c) * Real.log x + (c - a) * Real.log y + (a - b) * Real.log z = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_log_sum_l1915_191565


namespace NUMINAMATH_CALUDE_gym_down_payment_down_payment_is_50_l1915_191547

/-- Calculates the down payment for a gym membership -/
theorem gym_down_payment (monthly_fee : ℕ) (total_payment : ℕ) : ℕ :=
  let months : ℕ := 3 * 12
  let total_monthly_payments : ℕ := months * monthly_fee
  total_payment - total_monthly_payments

/-- Proves that the down payment for the gym membership is $50 -/
theorem down_payment_is_50 :
  gym_down_payment 12 482 = 50 := by
  sorry

end NUMINAMATH_CALUDE_gym_down_payment_down_payment_is_50_l1915_191547


namespace NUMINAMATH_CALUDE_painting_equation_proof_l1915_191515

theorem painting_equation_proof (t : ℝ) : 
  let doug_rate : ℝ := 1 / 4
  let dave_rate : ℝ := 1 / 6
  let combined_rate : ℝ := doug_rate + dave_rate
  let break_time : ℝ := 1 / 2
  (combined_rate * (t - break_time) = 1) ↔ 
  ((1 / 4 + 1 / 6) * (t - 1 / 2) = 1) :=
by sorry

end NUMINAMATH_CALUDE_painting_equation_proof_l1915_191515


namespace NUMINAMATH_CALUDE_circles_tangent_implies_a_eq_plus_minus_one_l1915_191563

/-- Circle E with equation x^2 + y^2 = 4 -/
def circle_E : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

/-- Circle F with equation x^2 + (y-a)^2 = 1, parameterized by a -/
def circle_F (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - a)^2 = 1}

/-- Two circles are internally tangent if they have exactly one point in common -/
def internally_tangent (C1 C2 : Set (ℝ × ℝ)) : Prop :=
  ∃! p : ℝ × ℝ, p ∈ C1 ∧ p ∈ C2

/-- Main theorem: If circles E and F are internally tangent, then a = ±1 -/
theorem circles_tangent_implies_a_eq_plus_minus_one (a : ℝ) :
  internally_tangent (circle_E) (circle_F a) → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_circles_tangent_implies_a_eq_plus_minus_one_l1915_191563


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1915_191577

theorem quadratic_equation_solution (m : ℝ) 
  (x₁ x₂ : ℝ) -- Two real roots
  (h1 : x₁^2 - m*x₁ + 2*m - 1 = 0) -- x₁ satisfies the equation
  (h2 : x₂^2 - m*x₂ + 2*m - 1 = 0) -- x₂ satisfies the equation
  (h3 : x₁^2 + x₂^2 = 7) -- Given condition
  : m = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1915_191577


namespace NUMINAMATH_CALUDE_paper_clips_count_l1915_191570

/-- The number of paper clips in 2 cases -/
def paper_clips_in_two_cases (c b : ℕ) : ℕ := 2 * (c * b) * 600

/-- Theorem stating the number of paper clips in 2 cases -/
theorem paper_clips_count (c b : ℕ) :
  paper_clips_in_two_cases c b = 2 * (c * b) * 600 := by
  sorry

end NUMINAMATH_CALUDE_paper_clips_count_l1915_191570


namespace NUMINAMATH_CALUDE_initial_strawberry_weight_l1915_191506

/-- The weight of strawberries initially collected by Marco and his dad -/
def initial_weight : ℕ := sorry

/-- The additional weight of strawberries found -/
def additional_weight : ℕ := 30

/-- Marco's strawberry weight after finding more -/
def marco_weight : ℕ := 36

/-- Marco's dad's strawberry weight after finding more -/
def dad_weight : ℕ := 16

/-- Theorem stating that the initial weight of strawberries is 22 pounds -/
theorem initial_strawberry_weight : initial_weight = 22 := by
  sorry

end NUMINAMATH_CALUDE_initial_strawberry_weight_l1915_191506


namespace NUMINAMATH_CALUDE_convex_lattice_polygon_vertices_l1915_191549

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A convex polygon defined by its vertices -/
structure ConvexPolygon where
  vertices : List LatticePoint
  is_convex : Bool  -- Assume this is true for our polygon

/-- Checks if a point is inside or on the sides of a polygon -/
def is_inside_or_on_sides (point : LatticePoint) (polygon : ConvexPolygon) : Bool :=
  sorry  -- Implementation details omitted

theorem convex_lattice_polygon_vertices (polygon : ConvexPolygon) :
  (∀ point : LatticePoint, point ∉ polygon.vertices → ¬(is_inside_or_on_sides point polygon)) →
  List.length polygon.vertices ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_convex_lattice_polygon_vertices_l1915_191549


namespace NUMINAMATH_CALUDE_tube_length_doubles_pressure_l1915_191555

/-- The length of the tube that doubles the pressure at the bottom of a water-filled barrel. -/
theorem tube_length_doubles_pressure (h₁ : ℝ) (m : ℝ) (ρ : ℝ) (g : ℝ) :
  h₁ = 1.5 →  -- height of the barrel in meters
  m = 1000 →  -- mass of water in the barrel in kg
  ρ = 1000 →  -- density of water in kg/m³
  g = 9.8 →   -- acceleration due to gravity in m/s²
  ∃ h₂ : ℝ,   -- height of water in the tube
    h₂ = 1.5 ∧ ρ * g * (h₁ + h₂) = 2 * (ρ * g * h₁) :=
by sorry

end NUMINAMATH_CALUDE_tube_length_doubles_pressure_l1915_191555


namespace NUMINAMATH_CALUDE_polygon_sides_l1915_191518

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1915_191518


namespace NUMINAMATH_CALUDE_jesse_bananas_l1915_191597

/-- The number of friends Jesse shares his bananas with -/
def num_friends : ℕ := 3

/-- The number of bananas each friend would get if Jesse shares equally -/
def bananas_per_friend : ℕ := 7

/-- The total number of bananas Jesse has -/
def total_bananas : ℕ := num_friends * bananas_per_friend

theorem jesse_bananas : total_bananas = 21 := by
  sorry

end NUMINAMATH_CALUDE_jesse_bananas_l1915_191597


namespace NUMINAMATH_CALUDE_expression_evaluation_l1915_191511

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 1 / y) :
  (2 * x - 1 / x) * (y - 1 / y) = -2 * x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1915_191511


namespace NUMINAMATH_CALUDE_triangle_inequality_with_circumradius_and_altitudes_l1915_191568

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  -- Add conditions to ensure it's a valid triangle
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  pos_R : 0 < R
  pos_altitudes : 0 < h_a ∧ 0 < h_b ∧ 0 < h_c

-- State the theorem
theorem triangle_inequality_with_circumradius_and_altitudes (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 ≥ 2 * t.R * (t.h_a + t.h_b + t.h_c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_circumradius_and_altitudes_l1915_191568


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l1915_191561

/-- A rectangular prism is a three-dimensional shape with 6 faces, 12 edges, and 8 vertices. -/
structure RectangularPrism where
  faces : Nat
  edges : Nat
  vertices : Nat
  faces_eq : faces = 6
  edges_eq : edges = 12
  vertices_eq : vertices = 8

/-- The sum of faces, edges, and vertices of a rectangular prism is 26. -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  rp.faces + rp.edges + rp.vertices = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l1915_191561


namespace NUMINAMATH_CALUDE_mixed_repeating_decimal_denominator_l1915_191513

/-- Represents a mixed repeating decimal as a pair of natural numbers (m, k),
    where m is the number of non-repeating digits after the decimal point,
    and k is the length of the repeating part. -/
structure MixedRepeatingDecimal where
  m : ℕ
  k : ℕ+

/-- Represents an irreducible fraction as a pair of integers (p, q) -/
structure IrreducibleFraction where
  p : ℤ
  q : ℤ
  q_pos : q > 0
  coprime : Int.gcd p q = 1

/-- States that a given irreducible fraction represents a mixed repeating decimal -/
def represents (f : IrreducibleFraction) (d : MixedRepeatingDecimal) : Prop := sorry

/-- The main theorem: If an irreducible fraction represents a mixed repeating decimal,
    then its denominator is divisible by 2 or 5 or both -/
theorem mixed_repeating_decimal_denominator
  (f : IrreducibleFraction)
  (d : MixedRepeatingDecimal)
  (h : represents f d) :
  ∃ (a b : ℕ), f.q = 2^a * 5^b * (2^d.k.val - 1) := by
  sorry

end NUMINAMATH_CALUDE_mixed_repeating_decimal_denominator_l1915_191513


namespace NUMINAMATH_CALUDE_parade_runner_time_l1915_191539

/-- The time taken for a runner to travel from the front to the end of a moving parade -/
theorem parade_runner_time (parade_length : ℝ) (parade_speed : ℝ) (runner_speed : ℝ) :
  parade_length = 2 →
  parade_speed = 3 →
  runner_speed = 6 →
  (parade_length / (runner_speed - parade_speed)) * 60 = 40 := by
  sorry

end NUMINAMATH_CALUDE_parade_runner_time_l1915_191539


namespace NUMINAMATH_CALUDE_trig_identity_l1915_191503

theorem trig_identity (θ : Real) (h : Real.tan θ = 2) : 
  (Real.sin θ * Real.cos θ) / (1 + Real.sin θ ^ 2) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1915_191503


namespace NUMINAMATH_CALUDE_max_value_expression_l1915_191576

theorem max_value_expression (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  (x^2 - 2*x*y + 2*y^2) * (x^2 - 2*x*z + 2*z^2) * (y^2 - 2*y*z + 2*z^2) ≤ 12 ∧
  ∃ x y z, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 3 ∧
    (x^2 - 2*x*y + 2*y^2) * (x^2 - 2*x*z + 2*z^2) * (y^2 - 2*y*z + 2*z^2) = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1915_191576


namespace NUMINAMATH_CALUDE_optimal_selling_price_l1915_191531

/-- The optimal selling price problem -/
theorem optimal_selling_price (purchase_price : ℝ) (initial_price : ℝ) (initial_volume : ℝ) 
  (price_volume_relation : ℝ → ℝ) (profit_function : ℝ → ℝ) :
  purchase_price = 40 →
  initial_price = 50 →
  initial_volume = 50 →
  (∀ x, price_volume_relation x = initial_volume - x) →
  (∀ x, profit_function x = (initial_price + x) * (price_volume_relation x) - purchase_price * (price_volume_relation x)) →
  ∃ max_profit : ℝ, ∀ x, profit_function x ≤ max_profit ∧ profit_function 20 = max_profit →
  initial_price + 20 = 70 := by
sorry

end NUMINAMATH_CALUDE_optimal_selling_price_l1915_191531


namespace NUMINAMATH_CALUDE_marble_problem_l1915_191510

theorem marble_problem (a b : ℚ) : 
  b = 6 ∧ 
  a + (3 * a - b) + (4 * 3 * a) + (6 * 4 * 3 * a) = 240 →
  a = 123 / 44 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l1915_191510


namespace NUMINAMATH_CALUDE_sin_theta_value_l1915_191544

theorem sin_theta_value (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.sin (θ - π / 3) = 5 / 13) :
  Real.sin θ = (5 + 12 * Real.sqrt 3) / 26 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l1915_191544


namespace NUMINAMATH_CALUDE_rhombus_area_l1915_191594

/-- The area of a rhombus with vertices at (0, 3.5), (8, 0), (0, -3.5), and (-8, 0) is 56 square units. -/
theorem rhombus_area : 
  let vertices : List (ℝ × ℝ) := [(0, 3.5), (8, 0), (0, -3.5), (-8, 0)]
  let diag1 : ℝ := |3.5 - (-3.5)|
  let diag2 : ℝ := |8 - (-8)|
  (diag1 * diag2) / 2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1915_191594


namespace NUMINAMATH_CALUDE_grade_distribution_l1915_191505

theorem grade_distribution (frac_A frac_B frac_C frac_D : ℝ) 
  (h1 : frac_A = 0.6)
  (h2 : frac_B = 0.25)
  (h3 : frac_C = 0.1)
  (h4 : frac_D = 0.05) :
  frac_A + frac_B + frac_C + frac_D = 1 := by
  sorry

end NUMINAMATH_CALUDE_grade_distribution_l1915_191505


namespace NUMINAMATH_CALUDE_odd_function_theorem_l1915_191598

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem odd_function_theorem (a b k : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →  -- f is odd
  (∀ t, f a b (t^2 - 2*t) + f a b (2*t^2 - k) < 0) →  -- inequality holds for all t
  (a = 2 ∧ b = 1 ∧ k < -1/3) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_theorem_l1915_191598


namespace NUMINAMATH_CALUDE_connect_four_games_l1915_191540

/-- 
Given:
- The ratio of games Kaleb won to games he lost is 3:2
- Kaleb won 18 games

Prove: The total number of games played is 30
-/
theorem connect_four_games (games_won : ℕ) (games_lost : ℕ) : 
  (games_won : ℚ) / games_lost = 3 / 2 → 
  games_won = 18 → 
  games_won + games_lost = 30 := by
sorry

end NUMINAMATH_CALUDE_connect_four_games_l1915_191540


namespace NUMINAMATH_CALUDE_truck_speed_problem_l1915_191569

/-- 
Proves that given two trucks 1025 km apart, with Driver A starting at 90 km/h 
and Driver B starting 1 hour later, if Driver A has driven 145 km farther than 
Driver B when they meet, then Driver B's average speed is 485/6 km/h.
-/
theorem truck_speed_problem (distance : ℝ) (speed_A : ℝ) (extra_distance : ℝ) 
  (h1 : distance = 1025)
  (h2 : speed_A = 90)
  (h3 : extra_distance = 145) : 
  ∃ (speed_B : ℝ) (time : ℝ), 
    speed_B = 485 / 6 ∧ 
    time > 0 ∧
    speed_A * (time + 1) = speed_B * time + extra_distance ∧
    speed_A * (time + 1) + speed_B * time = distance :=
by sorry


end NUMINAMATH_CALUDE_truck_speed_problem_l1915_191569


namespace NUMINAMATH_CALUDE_meeting_distance_l1915_191501

/-- Represents the distance walked by a person -/
def distance_walked (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Two people walking towards each other from 35 miles apart, 
    one at 2 mph and the other at 5 mph, will meet when the faster one has walked 25 miles -/
theorem meeting_distance (initial_distance : ℝ) (speed_fred : ℝ) (speed_sam : ℝ) :
  initial_distance = 35 →
  speed_fred = 2 →
  speed_sam = 5 →
  ∃ (time : ℝ), 
    distance_walked speed_fred time + distance_walked speed_sam time = initial_distance ∧
    distance_walked speed_sam time = 25 := by
  sorry

end NUMINAMATH_CALUDE_meeting_distance_l1915_191501


namespace NUMINAMATH_CALUDE_school_population_l1915_191536

theorem school_population (b g t : ℕ) : 
  b = 4 * g → g = 5 * t → b + g + t = 26 * t := by
  sorry

end NUMINAMATH_CALUDE_school_population_l1915_191536


namespace NUMINAMATH_CALUDE_inscribed_rectangle_length_l1915_191584

/-- Right triangle PQR with inscribed rectangle ABCD -/
structure InscribedRectangle where
  /-- Length of side PQ -/
  pq : ℝ
  /-- Length of side QR -/
  qr : ℝ
  /-- Length of side PR -/
  pr : ℝ
  /-- Length of rectangle ABCD (parallel to PR) -/
  length : ℝ
  /-- Height of rectangle ABCD (parallel to PQ) -/
  height : ℝ
  /-- PQR is a right triangle -/
  is_right_triangle : pq ^ 2 + qr ^ 2 = pr ^ 2
  /-- Height is half the length -/
  height_half_length : height = length / 2
  /-- Rectangle fits in triangle -/
  fits_in_triangle : height ≤ pq ∧ length ≤ pr ∧ (pr - length) / (qr - height) = height / pq

/-- The length of the inscribed rectangle is 7.5 -/
theorem inscribed_rectangle_length (rect : InscribedRectangle) 
  (h_pq : rect.pq = 5) (h_qr : rect.qr = 12) (h_pr : rect.pr = 13) : 
  rect.length = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_length_l1915_191584


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1915_191581

theorem necessary_but_not_sufficient (a b : ℝ) (h : a > b) :
  (∃ c : ℝ, c ≥ 0 ∧ ¬(a * c > b * c)) ∧
  (∀ c : ℝ, a * c > b * c → c ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1915_191581


namespace NUMINAMATH_CALUDE_smoothie_time_theorem_l1915_191548

/-- Represents the time in minutes to chop each fruit type -/
structure ChoppingTimes where
  apple : ℕ
  banana : ℕ
  strawberry : ℕ
  mango : ℕ
  pineapple : ℕ

/-- Calculates the total time to make smoothies -/
def totalSmoothieTime (ct : ChoppingTimes) (blendTime : ℕ) (numSmoothies : ℕ) : ℕ :=
  (ct.apple + ct.banana + ct.strawberry + ct.mango + ct.pineapple + blendTime) * numSmoothies

/-- Theorem: The total time to make 5 smoothies is 115 minutes -/
theorem smoothie_time_theorem (ct : ChoppingTimes) (blendTime : ℕ) (numSmoothies : ℕ) :
  ct.apple = 2 →
  ct.banana = 3 →
  ct.strawberry = 4 →
  ct.mango = 5 →
  ct.pineapple = 6 →
  blendTime = 3 →
  numSmoothies = 5 →
  totalSmoothieTime ct blendTime numSmoothies = 115 := by
  sorry

end NUMINAMATH_CALUDE_smoothie_time_theorem_l1915_191548


namespace NUMINAMATH_CALUDE_unique_solution_equation_l1915_191532

theorem unique_solution_equation (x : ℝ) :
  x ≥ 0 ∧ (2021 * (x^2020)^(1/202) - 1 = 2020 * x) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l1915_191532


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l1915_191520

/-- Atomic weight of Copper in g/mol -/
def Cu_weight : ℝ := 63.546

/-- Atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.011

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 15.999

/-- Number of Copper atoms in the compound -/
def Cu_count : ℕ := 1

/-- Number of Carbon atoms in the compound -/
def C_count : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 3

/-- Molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := Cu_count * Cu_weight + C_count * C_weight + O_count * O_weight

theorem compound_molecular_weight : molecular_weight = 123.554 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l1915_191520


namespace NUMINAMATH_CALUDE_inscribed_circle_triangle_shortest_side_l1915_191586

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first segment of the divided side -/
  s1 : ℝ
  /-- The length of the second segment of the divided side -/
  s2 : ℝ
  /-- The length of the shortest side of the triangle -/
  shortest_side : ℝ

/-- Theorem stating that for a triangle with an inscribed circle of radius 5 units
    that divides one side into segments of 4 and 10 units, the shortest side is 30 units -/
theorem inscribed_circle_triangle_shortest_side
  (t : InscribedCircleTriangle)
  (h1 : t.r = 5)
  (h2 : t.s1 = 4)
  (h3 : t.s2 = 10) :
  t.shortest_side = 30 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_triangle_shortest_side_l1915_191586


namespace NUMINAMATH_CALUDE_finish_in_sixteen_days_l1915_191546

/-- Represents Jack's reading pattern and book information -/
structure ReadingPattern where
  totalPages : Nat
  weekdayPages : Nat
  weekendPages : Nat
  weekdaySkip : Nat
  weekendSkip : Nat

/-- Calculates the number of days it takes to read the book -/
def daysToFinish (pattern : ReadingPattern) : Nat :=
  sorry

/-- Theorem stating that it takes 16 days to finish the book with the given reading pattern -/
theorem finish_in_sixteen_days :
  daysToFinish { totalPages := 285
                , weekdayPages := 23
                , weekendPages := 35
                , weekdaySkip := 3
                , weekendSkip := 2 } = 16 := by
  sorry

end NUMINAMATH_CALUDE_finish_in_sixteen_days_l1915_191546


namespace NUMINAMATH_CALUDE_largest_three_digit_special_divisible_l1915_191574

theorem largest_three_digit_special_divisible : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) → m ≤ n) ∧
  (n % 6 = 0) ∧
  (∀ d : ℕ, d > 0 ∧ d ≤ 9 ∧ (n / 100 = d ∨ (n / 10) % 10 = d ∨ n % 10 = d) → n % d = 0) ∧
  n = 843 := by
sorry

end NUMINAMATH_CALUDE_largest_three_digit_special_divisible_l1915_191574


namespace NUMINAMATH_CALUDE_tan_problem_l1915_191567

noncomputable def α : ℝ := Real.arctan 3

theorem tan_problem (h : Real.tan (π - α) = -3) :
  (Real.tan α = 3) ∧
  ((Real.sin (π - α) - Real.cos (π + α) - Real.sin (2*π - α) + Real.cos (-α)) /
   (Real.sin (π/2 - α) + Real.cos (3*π/2 - α)) = -4) :=
by sorry

end NUMINAMATH_CALUDE_tan_problem_l1915_191567


namespace NUMINAMATH_CALUDE_min_sum_position_l1915_191527

/-- An arithmetic sequence {a_n} with sum of first n terms S_n -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- The theorem stating when S_n reaches its minimum value -/
theorem min_sum_position (seq : ArithmeticSequence) 
  (h1 : seq.a 2 = -2)
  (h2 : seq.S 4 = -4) :
  ∃ n : ℕ, (n = 2 ∨ n = 3) ∧ 
    (∀ m : ℕ, m ≥ 1 → seq.S n ≤ seq.S m) :=
sorry

end NUMINAMATH_CALUDE_min_sum_position_l1915_191527


namespace NUMINAMATH_CALUDE_first_five_terms_of_sequence_l1915_191514

def a (n : ℕ) : ℤ := (-1)^n + n

theorem first_five_terms_of_sequence :
  (List.range 5).map (fun i => a (i + 1)) = [0, 3, 2, 5, 4] := by
  sorry

end NUMINAMATH_CALUDE_first_five_terms_of_sequence_l1915_191514


namespace NUMINAMATH_CALUDE_trajectory_and_max_dot_product_l1915_191550

/-- Trajectory of point P satisfying given conditions -/
def trajectory (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

/-- Line segment AB with A on x-axis and B on y-axis -/
def lineSegmentAB (xA yB : ℝ) : Prop :=
  xA^2 + yB^2 = 9 ∧ xA ≥ 0 ∧ yB ≥ 0

/-- Point P satisfies BP = 2PA -/
def pointPCondition (xA yB x y : ℝ) : Prop :=
  (x - 0)^2 + (y - yB)^2 = 4 * ((x - xA)^2 + y^2)

/-- Line passing through (1,0) -/
def lineThroughOneZero (t x y : ℝ) : Prop :=
  x = t * y + 1

/-- Theorem stating the trajectory equation and maximum dot product -/
theorem trajectory_and_max_dot_product :
  ∀ xA yB x y t x1 y1 x2 y2 : ℝ,
  lineSegmentAB xA yB →
  pointPCondition xA yB x y →
  trajectory x y →
  lineThroughOneZero t x1 y1 →
  lineThroughOneZero t x2 y2 →
  trajectory x1 y1 →
  trajectory x2 y2 →
  (∀ x' y' : ℝ, trajectory x' y' → lineThroughOneZero t x' y' → 
    x1 * x2 + y1 * y2 ≥ x' * x' + y' * y') →
  x1 * x2 + y1 * y2 ≤ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_max_dot_product_l1915_191550


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l1915_191566

theorem polynomial_identity_sum (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 
    (x^2 + a₁*x + d₁) * (x^2 + a₂*x + d₂) * (x^2 + a₃*x + d₃)) : 
  a₁*d₁ + a₂*d₂ + a₃*d₃ = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l1915_191566


namespace NUMINAMATH_CALUDE_abs_sum_minus_product_equals_two_l1915_191504

theorem abs_sum_minus_product_equals_two
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  |a| / a + |b| / b + |c| / c - (a * b * c) / |a * b * c| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_minus_product_equals_two_l1915_191504


namespace NUMINAMATH_CALUDE_original_plus_increase_equals_current_l1915_191537

/-- The number of bacteria originally in the petri dish -/
def original_bacteria : ℕ := 600

/-- The current number of bacteria in the petri dish -/
def current_bacteria : ℕ := 8917

/-- The increase in the number of bacteria -/
def bacteria_increase : ℕ := 8317

/-- Theorem stating that the original number of bacteria plus the increase
    equals the current number of bacteria -/
theorem original_plus_increase_equals_current :
  original_bacteria + bacteria_increase = current_bacteria := by
  sorry

end NUMINAMATH_CALUDE_original_plus_increase_equals_current_l1915_191537


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1915_191500

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 20)
  (square_diff : x^2 - y^2 = 200)
  (diff_eq : x - y = 10) : x - y = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1915_191500


namespace NUMINAMATH_CALUDE_group_meal_cost_l1915_191509

/-- Calculates the total cost for a group to eat at a restaurant --/
def total_cost (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : ℕ :=
  (total_people - num_kids) * adult_meal_cost

/-- Theorem: The total cost for the given group is $15 --/
theorem group_meal_cost : total_cost 12 7 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_group_meal_cost_l1915_191509


namespace NUMINAMATH_CALUDE_octal_734_equals_decimal_476_l1915_191590

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ :=
  let ones := octal % 10
  let eights := (octal / 10) % 10
  let sixty_fours := octal / 100
  ones + 8 * eights + 64 * sixty_fours

/-- The octal number 734 is equal to 476 in decimal --/
theorem octal_734_equals_decimal_476 : octal_to_decimal 734 = 476 := by
  sorry

end NUMINAMATH_CALUDE_octal_734_equals_decimal_476_l1915_191590


namespace NUMINAMATH_CALUDE_even_function_inequality_l1915_191573

/-- A function f: ℝ → ℝ is even -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is increasing on (-∞, 0] -/
def IsIncreasingOnNegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

/-- Main theorem -/
theorem even_function_inequality (f : ℝ → ℝ) (a : ℝ)
  (h_even : IsEven f)
  (h_incr : IsIncreasingOnNegative f)
  (h_ineq : f a ≤ f 2) :
  a ≤ -2 ∨ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l1915_191573


namespace NUMINAMATH_CALUDE_exam_students_count_l1915_191519

theorem exam_students_count :
  ∀ (N : ℕ) (T : ℝ),
  (T = 80 * N) →
  ((T - 350) / (N - 5) = 90) →
  N = 10 := by
sorry

end NUMINAMATH_CALUDE_exam_students_count_l1915_191519


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l1915_191583

theorem chocolate_bars_count (large_box small_boxes chocolate_bars_per_small_box : ℕ) 
  (h1 : small_boxes = 21)
  (h2 : chocolate_bars_per_small_box = 25)
  (h3 : large_box = small_boxes * chocolate_bars_per_small_box) :
  large_box = 525 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_count_l1915_191583


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l1915_191522

def y : ℕ := 2^3 * 3^3 * 4^4 * 5^5 * 6^6 * 7^7 * 8^8 * 11^3

theorem smallest_multiplier_for_perfect_square :
  ∃! k : ℕ, k > 0 ∧ 
  (∃ m : ℕ, k * y = m^2) ∧
  (∀ j : ℕ, j > 0 → j < k → ¬∃ n : ℕ, j * y = n^2) ∧
  k = 2310 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l1915_191522


namespace NUMINAMATH_CALUDE_pablo_puzzle_days_l1915_191564

def puzzles_400 : ℕ := 15
def pieces_per_400 : ℕ := 400
def puzzles_700 : ℕ := 10
def pieces_per_700 : ℕ := 700
def pieces_per_hour : ℕ := 100
def hours_per_day : ℕ := 6

def total_pieces : ℕ := puzzles_400 * pieces_per_400 + puzzles_700 * pieces_per_700

def total_hours : ℕ := (total_pieces + pieces_per_hour - 1) / pieces_per_hour

def days_required : ℕ := (total_hours + hours_per_day - 1) / hours_per_day

theorem pablo_puzzle_days : days_required = 22 := by
  sorry

end NUMINAMATH_CALUDE_pablo_puzzle_days_l1915_191564


namespace NUMINAMATH_CALUDE_fair_coin_tosses_l1915_191533

theorem fair_coin_tosses (n : ℕ) : 
  (1 / 2 : ℝ) ^ n = (1 / 16 : ℝ) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_tosses_l1915_191533


namespace NUMINAMATH_CALUDE_matthews_walking_rate_l1915_191524

/-- Proves that Matthew's walking rate is 3 km per hour given the problem conditions -/
theorem matthews_walking_rate (total_distance : ℝ) (johnny_start_delay : ℝ) (johnny_rate : ℝ) (johnny_distance : ℝ) :
  total_distance = 45 →
  johnny_start_delay = 1 →
  johnny_rate = 4 →
  johnny_distance = 24 →
  ∃ (matthews_rate : ℝ),
    matthews_rate = 3 ∧
    matthews_rate * (johnny_distance / johnny_rate + johnny_start_delay) = total_distance - johnny_distance :=
by sorry

end NUMINAMATH_CALUDE_matthews_walking_rate_l1915_191524


namespace NUMINAMATH_CALUDE_volume_range_l1915_191578

/-- Pyramid S-ABCD with square base ABCD and isosceles right triangle side face SAD -/
structure Pyramid where
  /-- Side length of the square base ABCD -/
  base_side : ℝ
  /-- Length of SC -/
  sc_length : ℝ
  /-- The base ABCD is a square with side length 2 -/
  base_side_eq_two : base_side = 2
  /-- The side face SAD is an isosceles right triangle with SD as the hypotenuse -/
  sad_isosceles_right : True  -- This condition is implied by the structure
  /-- 2√2 ≤ SC ≤ 4 -/
  sc_range : 2 * Real.sqrt 2 ≤ sc_length ∧ sc_length ≤ 4

/-- Volume of the pyramid -/
def volume (p : Pyramid) : ℝ := sorry

/-- Theorem stating the range of the pyramid's volume -/
theorem volume_range (p : Pyramid) : 
  (4 * Real.sqrt 3) / 3 ≤ volume p ∧ volume p ≤ 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_range_l1915_191578


namespace NUMINAMATH_CALUDE_sum_difference_problem_l1915_191599

theorem sum_difference_problem (x y : ℤ) : x + y = 45 → x = 25 → x - y = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_problem_l1915_191599


namespace NUMINAMATH_CALUDE_unique_solution_iff_a_eq_plus_minus_two_l1915_191538

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop := x^2 + y^2 + z^2 + 4*y = 0
def equation2 (a x y z : ℝ) : Prop := x + a*y + a*z - a = 0

-- Define what it means for the system to have a unique solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! (x y z : ℝ), equation1 x y z ∧ equation2 a x y z

-- State the theorem
theorem unique_solution_iff_a_eq_plus_minus_two :
  ∀ a : ℝ, has_unique_solution a ↔ (a = 2 ∨ a = -2) := by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_a_eq_plus_minus_two_l1915_191538


namespace NUMINAMATH_CALUDE_sally_out_of_pocket_l1915_191507

theorem sally_out_of_pocket (provided : ℕ) (book_cost : ℕ) (students : ℕ) :
  provided = 320 →
  book_cost = 12 →
  students = 30 →
  (students * book_cost - provided : ℤ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_sally_out_of_pocket_l1915_191507
