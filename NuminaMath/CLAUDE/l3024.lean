import Mathlib

namespace f_monotone_intervals_cos_alpha_value_l3024_302441

noncomputable def f (x : ℝ) : ℝ := 
  1/2 * (Real.sin x + Real.cos x) * (Real.sin x - Real.cos x) + Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_monotone_intervals (k : ℤ) : 
  StrictMonoOn f (Set.Icc (-Real.pi/6 + k * Real.pi) (Real.pi/3 + k * Real.pi)) := by sorry

theorem cos_alpha_value (α : ℝ) 
  (h1 : f (α/2 + Real.pi/4) = Real.sqrt 3 / 3) 
  (h2 : -Real.pi/2 < α) 
  (h3 : α < 0) : 
  Real.cos α = (3 + Real.sqrt 6) / 6 := by sorry

end f_monotone_intervals_cos_alpha_value_l3024_302441


namespace number_division_sum_l3024_302436

theorem number_division_sum (x : ℝ) : (x / 2 + x + 2 = 62) ↔ (x = 40) := by
  sorry

end number_division_sum_l3024_302436


namespace cube_painting_probability_l3024_302488

/-- The number of colors used to paint the cube -/
def num_colors : ℕ := 3

/-- The number of faces on a cube -/
def num_faces : ℕ := 6

/-- The probability of each color for a single face -/
def color_probability : ℚ := 1 / 3

/-- The total number of possible color arrangements for the cube -/
def total_arrangements : ℕ := num_colors ^ num_faces

/-- The number of favorable arrangements where the cube can be placed with four vertical faces of the same color -/
def favorable_arrangements : ℕ := 75

/-- The probability of painting the cube such that it can be placed with four vertical faces of the same color -/
def probability_four_same : ℚ := favorable_arrangements / total_arrangements

theorem cube_painting_probability :
  probability_four_same = 25 / 243 := by sorry

end cube_painting_probability_l3024_302488


namespace complement_A_union_B_l3024_302496

-- Define the sets A and B
def A : Set ℝ := {x | x < -1 ∨ (2 ≤ x ∧ x < 3)}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 4}

-- State the theorem
theorem complement_A_union_B :
  (Set.univ \ A) ∪ B = {x : ℝ | x ≥ -2} := by sorry

end complement_A_union_B_l3024_302496


namespace paul_initial_books_l3024_302442

/-- The number of books Paul sold in the garage sale -/
def books_sold : ℕ := 78

/-- The number of books Paul has left after the sale -/
def books_left : ℕ := 37

/-- The initial number of books Paul had -/
def initial_books : ℕ := books_sold + books_left

theorem paul_initial_books : initial_books = 115 := by
  sorry

end paul_initial_books_l3024_302442


namespace acute_angle_range_l3024_302429

/-- Given two vectors a and b in ℝ², prove that the angle between them is acute
    if and only if x is in the specified range. -/
theorem acute_angle_range (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 4]
  (∀ i, a i * b i > 0) ↔ x ∈ Set.Ioo (-8) 2 ∪ Set.Ioi 2 := by
  sorry

end acute_angle_range_l3024_302429


namespace range_of_y_over_x_l3024_302450

def C (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 3 = 0

theorem range_of_y_over_x : 
  ∀ x y : ℝ, C x y → ∃ t : ℝ, y / x = t ∧ -Real.sqrt 3 / 3 ≤ t ∧ t ≤ Real.sqrt 3 / 3 :=
by sorry

end range_of_y_over_x_l3024_302450


namespace function_composition_equality_l3024_302412

theorem function_composition_equality (b : ℝ) (h1 : b > 0) : 
  let g : ℝ → ℝ := λ x ↦ b * x^2 - Real.cos (π * x)
  g (g 1) = -Real.cos π → b = 1 := by
sorry

end function_composition_equality_l3024_302412


namespace farm_tree_count_l3024_302481

/-- Represents the number of trees of each type that fell during the typhoon -/
structure FallenTrees where
  narra : ℕ
  mahogany : ℕ
  total : ℕ
  one_more_mahogany : mahogany = narra + 1
  sum_equals_total : narra + mahogany = total

/-- Calculates the final number of trees on the farm -/
def final_tree_count (initial_mahogany initial_narra total_fallen : ℕ) (fallen : FallenTrees) : ℕ :=
  let remaining := initial_mahogany + initial_narra - total_fallen
  let new_narra := 2 * fallen.narra
  let new_mahogany := 3 * fallen.mahogany
  remaining + new_narra + new_mahogany

/-- The theorem to be proved -/
theorem farm_tree_count :
  ∃ (fallen : FallenTrees),
    fallen.total = 5 ∧
    final_tree_count 50 30 5 fallen = 88 := by
  sorry

end farm_tree_count_l3024_302481


namespace counterexample_exists_l3024_302420

theorem counterexample_exists : ∃ (a b : ℝ), a > b ∧ a^2 ≤ b^2 := by
  sorry

end counterexample_exists_l3024_302420


namespace score_difference_l3024_302449

/-- Represents the test scores of three students -/
structure TestScores where
  meghan : ℕ
  jose : ℕ
  alisson : ℕ

/-- The properties of the test and scores -/
def ValidTestScores (s : TestScores) : Prop :=
  let totalQuestions : ℕ := 50
  let marksPerQuestion : ℕ := 2
  let maxScore : ℕ := totalQuestions * marksPerQuestion
  let wrongQuestions : ℕ := 5
  (s.jose = maxScore - wrongQuestions * marksPerQuestion) ∧ 
  (s.jose = s.alisson + 40) ∧
  (s.meghan + s.jose + s.alisson = 210) ∧
  (s.meghan < s.jose)

/-- The theorem stating the difference between Jose's and Meghan's scores -/
theorem score_difference (s : TestScores) (h : ValidTestScores s) : 
  s.jose - s.meghan = 20 := by
  sorry

end score_difference_l3024_302449


namespace square_of_one_forty_four_l3024_302411

/-- Represents a number in a given base -/
def BaseRepresentation (n : ℕ) (b : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ : ℕ), d₁ < b ∧ d₂ < b ∧ d₃ < b ∧ n = d₁ * b^2 + d₂ * b + d₃

/-- The number 144 in base b -/
def OneFortyFour (b : ℕ) : ℕ := b^2 + 4*b + 4

theorem square_of_one_forty_four (b : ℕ) :
  b > 4 →
  BaseRepresentation (OneFortyFour b) b →
  ∃ k : ℕ, OneFortyFour b = k^2 :=
sorry

end square_of_one_forty_four_l3024_302411


namespace trajectory_is_parallel_plane_l3024_302453

-- Define the type for a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the set of points P satisfying y = 3
def TrajectorySet : Set Point3D :=
  {p : Point3D | p.y = 3}

-- Define a plane parallel to xOz plane
def ParallelPlane (h : ℝ) : Set Point3D :=
  {p : Point3D | p.y = h}

-- Theorem statement
theorem trajectory_is_parallel_plane :
  ∃ h : ℝ, TrajectorySet = ParallelPlane h := by
  sorry

end trajectory_is_parallel_plane_l3024_302453


namespace cone_base_circumference_l3024_302454

/-- 
Given a right circular cone with volume 24π cubic centimeters and height 6 cm,
prove that the circumference of its base is 4√3π cm.
-/
theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = 24 * Real.pi ∧ h = 6 ∧ V = (1/3) * Real.pi * r^2 * h →
  2 * Real.pi * r = 4 * Real.sqrt 3 * Real.pi :=
by sorry

end cone_base_circumference_l3024_302454


namespace initial_average_weight_l3024_302477

/-- Proves the initially calculated average weight given the conditions of the problem -/
theorem initial_average_weight (n : ℕ) (misread_weight correct_weight : ℝ) (correct_avg : ℝ) :
  n = 20 ∧ 
  misread_weight = 56 ∧
  correct_weight = 61 ∧
  correct_avg = 58.65 →
  ∃ initial_avg : ℝ, 
    initial_avg * n + (correct_weight - misread_weight) = correct_avg * n ∧
    initial_avg = 58.4 := by
  sorry

end initial_average_weight_l3024_302477


namespace point_on_line_l3024_302456

/-- 
Given two points (m, n) and (m + 2, n + some_value) that lie on the line x = (y/2) - (2/5),
prove that some_value must equal 4.
-/
theorem point_on_line (m n some_value : ℝ) : 
  (m = n / 2 - 2 / 5) ∧ (m + 2 = (n + some_value) / 2 - 2 / 5) → some_value = 4 := by
  sorry

end point_on_line_l3024_302456


namespace train_length_l3024_302493

theorem train_length (bridge_length : ℝ) (total_time : ℝ) (on_bridge_time : ℝ) :
  bridge_length = 600 →
  total_time = 30 →
  on_bridge_time = 20 →
  (bridge_length + (bridge_length * on_bridge_time / total_time)) / (total_time - on_bridge_time) = 120 :=
by sorry

end train_length_l3024_302493


namespace soccer_team_games_l3024_302416

/-- Calculates the number of games played by a soccer team based on pizza slices and goals scored. -/
theorem soccer_team_games (pizzas : ℕ) (slices_per_pizza : ℕ) (goals_per_game : ℕ) 
  (h1 : pizzas = 6)
  (h2 : slices_per_pizza = 12)
  (h3 : goals_per_game = 9)
  (h4 : pizzas * slices_per_pizza = goals_per_game * (pizzas * slices_per_pizza / goals_per_game)) :
  pizzas * slices_per_pizza / goals_per_game = 8 := by
  sorry

end soccer_team_games_l3024_302416


namespace remainder_101_47_mod_100_l3024_302402

theorem remainder_101_47_mod_100 : 101^47 % 100 = 1 := by
  sorry

end remainder_101_47_mod_100_l3024_302402


namespace inequality_always_true_l3024_302455

theorem inequality_always_true (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 0) : c * a < c * b := by
  sorry

end inequality_always_true_l3024_302455


namespace nested_rectangles_exist_l3024_302421

/-- Represents a rectangle with integer sides --/
structure Rectangle where
  width : Nat
  height : Nat
  width_bound : width ≤ 100
  height_bound : height ≤ 100

/-- Checks if rectangle a can be nested inside rectangle b --/
def can_nest (a b : Rectangle) : Prop :=
  a.width ≤ b.width ∧ a.height ≤ b.height

theorem nested_rectangles_exist (rectangles : Finset Rectangle) 
  (h : rectangles.card = 101) :
  ∃ (A B C : Rectangle), A ∈ rectangles ∧ B ∈ rectangles ∧ C ∈ rectangles ∧
    can_nest A B ∧ can_nest B C := by
  sorry

end nested_rectangles_exist_l3024_302421


namespace parallelogram_diagonal_intersection_l3024_302471

/-- Given a parallelogram with opposite vertices (2, -3) and (14, 9),
    the diagonals intersect at the point (8, 3). -/
theorem parallelogram_diagonal_intersection :
  let a : ℝ × ℝ := (2, -3)
  let b : ℝ × ℝ := (14, 9)
  let midpoint : ℝ × ℝ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)
  midpoint = (8, 3) := by sorry

end parallelogram_diagonal_intersection_l3024_302471


namespace sum_of_roots_quadratic_sum_of_roots_2x2_minus_8x_plus_6_l3024_302425

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_roots_2x2_minus_8x_plus_6 :
  let f : ℝ → ℝ := λ x => 2*x^2 - 8*x + 6
  let r₁ := (-(-8) + Real.sqrt ((-8)^2 - 4*2*6)) / (2*2)
  let r₂ := (-(-8) - Real.sqrt ((-8)^2 - 4*2*6)) / (2*2)
  r₁ + r₂ = 4 :=
by sorry

end sum_of_roots_quadratic_sum_of_roots_2x2_minus_8x_plus_6_l3024_302425


namespace second_number_possibilities_l3024_302483

def is_valid_pair (x y : ℤ) : Prop :=
  (x = 14 ∨ y = 14) ∧ 2*x + 3*y = 94

theorem second_number_possibilities :
  ∃ (a b : ℤ), a ≠ b ∧ 
  (∀ x y, is_valid_pair x y → (x = 14 ∧ y = a) ∨ (y = 14 ∧ x = b)) :=
sorry

end second_number_possibilities_l3024_302483


namespace science_club_election_theorem_l3024_302435

def total_candidates : ℕ := 20
def past_officers : ℕ := 8
def positions_to_fill : ℕ := 6

def elections_with_at_least_two_past_officers : ℕ :=
  Nat.choose total_candidates positions_to_fill -
  (Nat.choose (total_candidates - past_officers) positions_to_fill +
   Nat.choose past_officers 1 * Nat.choose (total_candidates - past_officers) (positions_to_fill - 1))

theorem science_club_election_theorem :
  elections_with_at_least_two_past_officers = 31500 := by
  sorry

end science_club_election_theorem_l3024_302435


namespace tangent_line_to_circle_l3024_302444

theorem tangent_line_to_circle (r : ℝ) (h_pos : r > 0) :
  (∀ x y : ℝ, x + y = r → x^2 + y^2 = 4*r → 
    ∀ ε > 0, ∃ x' y' : ℝ, x' + y' = r ∧ (x' - x)^2 + (y' - y)^2 < ε^2 ∧ x'^2 + y'^2 ≠ 4*r) →
  r = 8 := by
sorry

end tangent_line_to_circle_l3024_302444


namespace total_cost_calculation_l3024_302484

def coffee_maker_price : ℝ := 70
def blender_price : ℝ := 100
def coffee_maker_discount : ℝ := 0.20
def blender_discount : ℝ := 0.15
def sales_tax_rate : ℝ := 0.08
def extended_warranty_cost : ℝ := 25
def shipping_fee : ℝ := 12

def total_cost : ℝ :=
  let discounted_coffee_maker := coffee_maker_price * (1 - coffee_maker_discount)
  let discounted_blender := blender_price * (1 - blender_discount)
  let subtotal := 2 * discounted_coffee_maker + discounted_blender
  let sales_tax := subtotal * sales_tax_rate
  subtotal + sales_tax + extended_warranty_cost + shipping_fee

theorem total_cost_calculation :
  total_cost = 249.76 := by sorry

end total_cost_calculation_l3024_302484


namespace points_per_game_l3024_302414

theorem points_per_game (total_points : ℕ) (num_games : ℕ) (points_per_game : ℕ) : 
  total_points = 24 → 
  num_games = 6 → 
  total_points = num_games * points_per_game → 
  points_per_game = 4 := by
sorry

end points_per_game_l3024_302414


namespace rectangle_area_increase_l3024_302428

theorem rectangle_area_increase (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  let original_area := x * y
  let new_length := 1.2 * x
  let new_width := 1.1 * y
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.32 := by
sorry

end rectangle_area_increase_l3024_302428


namespace triangle_area_sides_circumradius_l3024_302451

/-- The area of a triangle in terms of its sides and circumradius -/
theorem triangle_area_sides_circumradius (a b c R S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_circumradius : R = (a * b * c) / (4 * S))
  (h_area : S > 0) :
  S = (a * b * c) / (4 * R) := by
sorry


end triangle_area_sides_circumradius_l3024_302451


namespace decimal_123_to_binary_l3024_302464

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

theorem decimal_123_to_binary :
  decimal_to_binary 123 = [true, true, false, true, true, true, true] := by
  sorry

end decimal_123_to_binary_l3024_302464


namespace min_value_parallel_vectors_l3024_302494

/-- Given vectors a and b, with m > 0, n > 0, and a parallel to b, 
    the minimum value of 1/m + 8/n is 9/2 -/
theorem min_value_parallel_vectors (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (4 - n, 2)
  (∃ k : ℝ, a = k • b) →
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → 1 / m' + 8 / n' ≥ 9 / 2) ∧
  (∃ m' n' : ℝ, m' > 0 ∧ n' > 0 ∧ 1 / m' + 8 / n' = 9 / 2) :=
by sorry

end min_value_parallel_vectors_l3024_302494


namespace ratio_equality_l3024_302423

theorem ratio_equality : ∃ x : ℝ, (12 : ℝ) / 8 = x / 240 ∧ x = 360 := by
  sorry

end ratio_equality_l3024_302423


namespace max_donuts_is_17_seventeen_donuts_possible_l3024_302468

-- Define the prices and budget
def single_price : ℕ := 1
def pack4_price : ℕ := 3
def pack8_price : ℕ := 5
def budget : ℕ := 11

-- Define a function to calculate the number of donuts for a given combination
def donut_count (singles pack4 pack8 : ℕ) : ℕ :=
  singles + 4 * pack4 + 8 * pack8

-- Define a function to calculate the total cost for a given combination
def total_cost (singles pack4 pack8 : ℕ) : ℕ :=
  singles * single_price + pack4 * pack4_price + pack8 * pack8_price

-- Theorem stating that 17 is the maximum number of donuts that can be purchased
theorem max_donuts_is_17 :
  ∀ (singles pack4 pack8 : ℕ),
    total_cost singles pack4 pack8 ≤ budget →
    donut_count singles pack4 pack8 ≤ 17 :=
by
  sorry

-- Theorem stating that 17 donuts can actually be purchased
theorem seventeen_donuts_possible :
  ∃ (singles pack4 pack8 : ℕ),
    total_cost singles pack4 pack8 ≤ budget ∧
    donut_count singles pack4 pack8 = 17 :=
by
  sorry

end max_donuts_is_17_seventeen_donuts_possible_l3024_302468


namespace line_contains_point_l3024_302432

theorem line_contains_point (j : ℝ) : 
  (∀ x y : ℝ, -2 - 3*j*x = 7*y → x = 1/3 ∧ y = -3) → j = 19 := by
  sorry

end line_contains_point_l3024_302432


namespace sharon_journey_distance_l3024_302459

/-- Represents the journey from Sharon's house to her mother's house -/
structure Journey where
  distance : ℝ
  normalTime : ℝ
  reducedTime : ℝ
  speedReduction : ℝ

/-- The specific journey with given conditions -/
def sharonJourney : Journey where
  distance := 140  -- to be proved
  normalTime := 240
  reducedTime := 330
  speedReduction := 15

theorem sharon_journey_distance :
  ∀ j : Journey,
  j.normalTime = 240 ∧
  j.reducedTime = 330 ∧
  j.speedReduction = 15 ∧
  (j.distance / j.normalTime - j.speedReduction / 60) * (j.reducedTime - j.normalTime / 2) = j.distance / 2 →
  j.distance = 140 := by
  sorry

#check sharon_journey_distance

end sharon_journey_distance_l3024_302459


namespace max_triangle_area_l3024_302400

/-- Ellipse type -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Line type -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle type -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Function to calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Function to check if a point is on an ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop := sorry

/-- Function to check if a line intersects an ellipse at two distinct points -/
def intersectsAtTwoPoints (l : Line) (e : Ellipse) : Prop := sorry

/-- Theorem statement -/
theorem max_triangle_area 
  (e : Ellipse) 
  (h_eccentricity : e.a^2 - e.b^2 = e.a^2 / 2)
  (A : Point)
  (h_A_on_ellipse : isOnEllipse A e)
  (h_A_coords : A.x = 1 ∧ A.y = Real.sqrt 2)
  (l : Line)
  (h_l_slope : l.slope = Real.sqrt 2)
  (h_intersects : intersectsAtTwoPoints l e) :
  ∃ (B C : Point), 
    isOnEllipse B e ∧ 
    isOnEllipse C e ∧ 
    B ≠ C ∧
    ∀ (B' C' : Point), 
      isOnEllipse B' e → 
      isOnEllipse C' e → 
      B' ≠ C' →
      triangleArea ⟨A, B', C'⟩ ≤ Real.sqrt 2 ∧
      triangleArea ⟨A, B, C⟩ = Real.sqrt 2 := by
  sorry


end max_triangle_area_l3024_302400


namespace principal_calculation_l3024_302482

/-- Proves that given the specified conditions, the principal is 6200 --/
theorem principal_calculation (rate : ℚ) (time : ℕ) (interest_difference : ℚ) :
  rate = 5 / 100 →
  time = 10 →
  interest_difference = 3100 →
  ∃ (principal : ℚ), principal * rate * time = principal - interest_difference ∧ principal = 6200 := by
  sorry

end principal_calculation_l3024_302482


namespace quadratic_sum_of_coefficients_l3024_302426

/-- A quadratic function with a positive constant term -/
def f (a b k : ℝ) (hk : k > 0) (x : ℝ) : ℝ := a * x^2 + b * x + k

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

theorem quadratic_sum_of_coefficients 
  (a b k : ℝ) (hk : k > 0) : 
  (f' a b 0 = 0) → 
  (f' a b 1 = 2) → 
  a + b = 1 := by
sorry

end quadratic_sum_of_coefficients_l3024_302426


namespace power_tower_mod_500_l3024_302480

theorem power_tower_mod_500 : 
  5^(5^(5^5)) ≡ 125 [ZMOD 500] := by sorry

end power_tower_mod_500_l3024_302480


namespace point_in_third_quadrant_l3024_302473

def angle : ℝ := 2017

theorem point_in_third_quadrant :
  let x := Real.cos (angle * π / 180)
  let y := Real.sin (angle * π / 180)
  x < 0 ∧ y < 0 :=
by sorry

end point_in_third_quadrant_l3024_302473


namespace barn_paint_area_l3024_302419

/-- Represents the dimensions of a rectangular barn -/
structure BarnDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents the dimensions of a rectangular opening (door or window) -/
structure OpeningDimensions where
  width : ℝ
  height : ℝ

/-- Calculates the total area to be painted in a barn -/
def totalPaintArea (barn : BarnDimensions) (doors : List OpeningDimensions) (windows : List OpeningDimensions) : ℝ :=
  let wallArea := 2 * (barn.width * barn.height + barn.length * barn.height)
  let floorCeilingArea := 2 * (barn.width * barn.length)
  let doorArea := doors.map (fun d => d.width * d.height) |>.sum
  let windowArea := windows.map (fun w => w.width * w.height) |>.sum
  2 * (wallArea - doorArea - windowArea) + floorCeilingArea

/-- Theorem stating that the total area to be painted is 1588 sq yd -/
theorem barn_paint_area :
  let barn := BarnDimensions.mk 15 20 8
  let doors := [OpeningDimensions.mk 3 7, OpeningDimensions.mk 3 7]
  let windows := [OpeningDimensions.mk 2 4, OpeningDimensions.mk 2 4, OpeningDimensions.mk 2 4]
  totalPaintArea barn doors windows = 1588 := by
  sorry

end barn_paint_area_l3024_302419


namespace perfect_square_polynomial_l3024_302463

/-- A polynomial that is always a perfect square for integer inputs can be expressed as (dx + e)^2 -/
theorem perfect_square_polynomial
  (a b c : ℤ)
  (h : ∀ (x : ℤ), ∃ (y : ℤ), a * x^2 + b * x + c = y^2) :
  ∃ (d e : ℤ), ∀ (x : ℤ), a * x^2 + b * x + c = (d * x + e)^2 := by
  sorry

end perfect_square_polynomial_l3024_302463


namespace area_enclosed_by_cosine_curve_l3024_302476

theorem area_enclosed_by_cosine_curve : 
  let f (x : ℝ) := Real.cos x
  let area := ∫ x in (0)..(π/2), f x - ∫ x in (π/2)..(3*π/2), f x
  area = 3 := by
sorry

end area_enclosed_by_cosine_curve_l3024_302476


namespace consecutive_integers_sum_l3024_302409

theorem consecutive_integers_sum (a b c : ℤ) : 
  (b = a + 1) →
  (c = b + 1) →
  (a + c = 140) →
  (b - a = 2) →
  (a + b + c = 210) := by
sorry

end consecutive_integers_sum_l3024_302409


namespace sum_of_generated_numbers_eq_5994_l3024_302427

/-- The sum of all three-digit natural numbers created using digits 1, 2, and 3 -/
def sum_three_digit_numbers : ℕ := 5994

/-- The set of digits that can be used -/
def valid_digits : Finset ℕ := {1, 2, 3}

/-- A function to generate all possible three-digit numbers using the valid digits -/
def generate_numbers : Finset ℕ := sorry

/-- Theorem stating that the sum of all generated numbers equals sum_three_digit_numbers -/
theorem sum_of_generated_numbers_eq_5994 : 
  (generate_numbers.sum id) = sum_three_digit_numbers := by sorry

end sum_of_generated_numbers_eq_5994_l3024_302427


namespace quadratic_two_real_roots_l3024_302487

theorem quadratic_two_real_roots (b c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ ∀ z : ℝ, z^2 + b*z + c = 0 ↔ z = x ∨ z = y) ↔ b^2 - 4*c ≥ 0 := by
sorry

end quadratic_two_real_roots_l3024_302487


namespace yellow_bags_count_l3024_302461

/-- Represents the number of marbles in each type of bag -/
def marbles_per_bag : Fin 3 → ℕ
  | 0 => 10  -- Red bags
  | 1 => 50  -- Blue bags
  | 2 => 100 -- Yellow bags
  | _ => 0   -- This case is unreachable due to Fin 3

/-- The total number of bags -/
def total_bags : ℕ := 12

/-- The total number of marbles -/
def total_marbles : ℕ := 500

theorem yellow_bags_count :
  ∃ (red blue yellow : ℕ),
    red + blue + yellow = total_bags ∧
    red * marbles_per_bag 0 + blue * marbles_per_bag 1 + yellow * marbles_per_bag 2 = total_marbles ∧
    red = blue ∧
    yellow = 2 := by sorry

end yellow_bags_count_l3024_302461


namespace fifty_second_card_is_ace_l3024_302479

-- Define the card ranks
inductive Rank
| King | Queen | Jack | Ten | Nine | Eight | Seven | Six | Five | Four | Three | Two | Ace

-- Define the reversed order of cards
def reversedOrder : List Rank := [
  Rank.King, Rank.Queen, Rank.Jack, Rank.Ten, Rank.Nine, Rank.Eight, Rank.Seven,
  Rank.Six, Rank.Five, Rank.Four, Rank.Three, Rank.Two, Rank.Ace
]

-- Define the number of cards in a cycle
def cardsPerCycle : Nat := 13

-- Define the position we're interested in
def targetPosition : Nat := 52

-- Theorem: The 52nd card in the reversed deck is an Ace
theorem fifty_second_card_is_ace :
  (targetPosition - 1) % cardsPerCycle = cardsPerCycle - 1 →
  reversedOrder[(targetPosition - 1) % cardsPerCycle] = Rank.Ace :=
by
  sorry

#check fifty_second_card_is_ace

end fifty_second_card_is_ace_l3024_302479


namespace square_area_not_correlation_l3024_302495

/-- A relationship between two variables -/
structure Relationship (α β : Type) where
  relate : α → β → Prop

/-- A correlation is a relationship that is not deterministic -/
def IsCorrelation {α β : Type} (r : Relationship α β) : Prop :=
  ∃ (x : α) (y₁ y₂ : β), y₁ ≠ y₂ ∧ r.relate x y₁ ∧ r.relate x y₂

/-- The relationship between a square's side length and its area -/
def SquareAreaRelationship : Relationship ℝ ℝ :=
  { relate := λ side area => area = side ^ 2 }

/-- Theorem: The relationship between a square's side length and its area is not a correlation -/
theorem square_area_not_correlation : ¬ IsCorrelation SquareAreaRelationship := by
  sorry

end square_area_not_correlation_l3024_302495


namespace basic_astrophysics_degrees_l3024_302445

/-- Represents the allocation of a budget in a circle graph --/
def BudgetAllocation (total : ℝ) (allocated : ℝ) (degreesPerPercent : ℝ) : Prop :=
  total = 100 ∧ 
  allocated = 95 ∧ 
  degreesPerPercent = 360 / 100

/-- Theorem: The number of degrees representing the remaining budget (basic astrophysics) is 18 --/
theorem basic_astrophysics_degrees 
  (total allocated remaining : ℝ) 
  (degreesPerPercent : ℝ) 
  (h : BudgetAllocation total allocated degreesPerPercent) :
  remaining = 18 :=
sorry

end basic_astrophysics_degrees_l3024_302445


namespace gavin_blue_shirts_l3024_302413

/-- The number of blue shirts Gavin has -/
def blue_shirts (total : ℕ) (green : ℕ) : ℕ := total - green

theorem gavin_blue_shirts :
  let total_shirts : ℕ := 23
  let green_shirts : ℕ := 17
  blue_shirts total_shirts green_shirts = 6 := by
sorry

end gavin_blue_shirts_l3024_302413


namespace jill_age_l3024_302460

/-- Represents the ages of individuals in the problem -/
structure Ages where
  gina : ℕ
  helen : ℕ
  ian : ℕ
  jill : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.gina + 4 = ages.helen ∧
  ages.helen = ages.ian + 5 ∧
  ages.jill = ages.ian + 2 ∧
  ages.gina = 18

/-- The theorem stating Jill's age -/
theorem jill_age (ages : Ages) (h : problem_conditions ages) : ages.jill = 19 := by
  sorry

#check jill_age

end jill_age_l3024_302460


namespace janes_homework_l3024_302490

theorem janes_homework (x y z : ℝ) 
  (h1 : x - (y + z) = 15) 
  (h2 : x - y + z = 7) : 
  x - y = 11 := by
sorry

end janes_homework_l3024_302490


namespace square_root_of_two_l3024_302447

theorem square_root_of_two : Real.sqrt 2 = (Real.sqrt 2 : ℝ) := by
  sorry

end square_root_of_two_l3024_302447


namespace log_sum_difference_equals_two_l3024_302467

theorem log_sum_difference_equals_two :
  Real.log 50 / Real.log 10 + Real.log 20 / Real.log 10 - Real.log 10 / Real.log 10 = 2 := by
  sorry

end log_sum_difference_equals_two_l3024_302467


namespace optimal_garden_dimensions_l3024_302475

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  width : Real
  length : Real

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : Real :=
  d.width * d.length

/-- Calculates the perimeter of a rectangular garden -/
def gardenPerimeter (d : GardenDimensions) : Real :=
  2 * (d.width + d.length)

/-- Theorem: Optimal dimensions for a rectangular garden with minimum fencing -/
theorem optimal_garden_dimensions :
  ∃ (d : GardenDimensions),
    d.length = 2 * d.width ∧
    gardenArea d ≥ 500 ∧
    (∀ (d' : GardenDimensions),
      d'.length = 2 * d'.width →
      gardenArea d' ≥ 500 →
      gardenPerimeter d ≤ gardenPerimeter d') ∧
    d.width = 5 * Real.sqrt 10 ∧
    d.length = 10 * Real.sqrt 10 ∧
    gardenPerimeter d = 30 * Real.sqrt 10 :=
  sorry


end optimal_garden_dimensions_l3024_302475


namespace mitchell_gum_packets_l3024_302408

theorem mitchell_gum_packets (pieces_per_packet : ℕ) (pieces_chewed : ℕ) (pieces_left : ℕ) : 
  pieces_per_packet = 7 →
  pieces_left = 2 →
  pieces_chewed = 54 →
  (pieces_chewed + pieces_left) / pieces_per_packet = 8 :=
by
  sorry

end mitchell_gum_packets_l3024_302408


namespace peter_additional_miles_l3024_302458

/-- The number of additional miles Peter runs compared to Andrew each day -/
def additional_miles : ℝ := sorry

/-- Andrew's daily miles -/
def andrew_miles : ℝ := 2

/-- Number of days they run -/
def days : ℕ := 5

/-- Total miles run by both after 5 days -/
def total_miles : ℝ := 35

theorem peter_additional_miles :
  additional_miles = 3 ∧
  days * (andrew_miles + additional_miles) + days * andrew_miles = total_miles :=
sorry

end peter_additional_miles_l3024_302458


namespace no_five_points_configuration_l3024_302433

-- Define a type for points in space
variable (Point : Type)

-- Define a distance function between two points
variable (dist : Point → Point → ℝ)

-- Define the congruence transformation type
def CongruenceTransformation (Point : Type) := Point → Point

-- First congruence transformation
variable (t1 : CongruenceTransformation Point)

-- Second congruence transformation
variable (t2 : CongruenceTransformation Point)

-- Theorem statement
theorem no_five_points_configuration 
  (A B C D E : Point)
  (h1 : t1 A = B ∧ t1 B = A ∧ t1 C = C ∧ t1 D = D ∧ t1 E = E)
  (h2 : t2 A = B ∧ t2 B = C ∧ t2 C = D ∧ t2 D = E ∧ t2 E = A)
  (h3 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E)
  (h4 : ∀ X Y : Point, dist (t1 X) (t1 Y) = dist X Y)
  (h5 : ∀ X Y : Point, dist (t2 X) (t2 Y) = dist X Y) :
  False :=
sorry

end no_five_points_configuration_l3024_302433


namespace triangle_angle_sum_max_l3024_302418

theorem triangle_angle_sum_max (A C : Real) (h1 : 0 < A) (h2 : A < 2 * π / 3) (h3 : A + C = 2 * π / 3) :
  let S := (Real.sqrt 3 / 3) * Real.sin A * Real.sin C
  ∃ (max_S : Real), ∀ (A' C' : Real), 
    0 < A' → A' < 2 * π / 3 → A' + C' = 2 * π / 3 → 
    (Real.sqrt 3 / 3) * Real.sin A' * Real.sin C' ≤ max_S ∧
    max_S = Real.sqrt 3 / 4 := by
  sorry

end triangle_angle_sum_max_l3024_302418


namespace bargain_bin_books_l3024_302466

/-- Calculate the number of books in a bargain bin after sales and additions. -/
def booksInBin (initial : ℕ) (sold : ℕ) (added : ℕ) : ℕ :=
  initial - sold + added

/-- Theorem stating that for the given values, the number of books in the bin is 11. -/
theorem bargain_bin_books : booksInBin 4 3 10 = 11 := by
  sorry

end bargain_bin_books_l3024_302466


namespace stream_speed_l3024_302404

theorem stream_speed (swim_speed : ℝ) (upstream_time downstream_time : ℝ) :
  swim_speed = 12 ∧ 
  upstream_time = 2 * downstream_time ∧ 
  upstream_time > 0 ∧ 
  downstream_time > 0 →
  ∃ stream_speed : ℝ,
    stream_speed = 4 ∧
    (swim_speed - stream_speed) * upstream_time = (swim_speed + stream_speed) * downstream_time :=
by sorry

end stream_speed_l3024_302404


namespace condition_a_necessary_not_sufficient_l3024_302492

-- Define Condition A
def condition_a (x y : ℝ) : Prop :=
  2 < x + y ∧ x + y < 4 ∧ 0 < x * y ∧ x * y < 3

-- Define Condition B
def condition_b (x y : ℝ) : Prop :=
  0 < x ∧ x < 1 ∧ 2 < y ∧ y < 3

-- Theorem stating that Condition A is necessary but not sufficient for Condition B
theorem condition_a_necessary_not_sufficient :
  (∀ x y : ℝ, condition_b x y → condition_a x y) ∧
  (∃ x y : ℝ, condition_a x y ∧ ¬condition_b x y) := by
  sorry

end condition_a_necessary_not_sufficient_l3024_302492


namespace min_nSn_value_l3024_302472

/-- An arithmetic sequence with sum S_n for the first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- The sum function
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n : ℕ, S n = n * (2 * a 0 + (n - 1) * (a 1 - a 0)) / 2

/-- The main theorem stating the minimum value of nS_n -/
theorem min_nSn_value (seq : ArithmeticSequence) 
    (h1 : seq.S 10 = 0) 
    (h2 : seq.S 15 = 25) : 
  ∃ n : ℕ, ∀ m : ℕ, n * seq.S n ≤ m * seq.S m ∧ n * seq.S n = -49 := by
  sorry

end min_nSn_value_l3024_302472


namespace min_value_sum_l3024_302491

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  a + 2*b ≥ 9 := by
  sorry

end min_value_sum_l3024_302491


namespace martha_reading_challenge_l3024_302469

def pages_read : List Nat := [12, 18, 14, 20, 11, 13, 19, 15, 17]
def total_days : Nat := 10
def target_average : Nat := 15

theorem martha_reading_challenge :
  ∃ (x : Nat), 
    (List.sum pages_read + x) / total_days = target_average ∧
    x = 11 := by
  sorry

end martha_reading_challenge_l3024_302469


namespace polynomial_multiplication_l3024_302457

theorem polynomial_multiplication (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end polynomial_multiplication_l3024_302457


namespace xavier_yvonne_not_zelda_prob_l3024_302489

-- Define the probabilities of success for each person
def xavier_prob : ℚ := 1/4
def yvonne_prob : ℚ := 2/3
def zelda_prob : ℚ := 5/8

-- Define the probability of the desired outcome
def desired_outcome_prob : ℚ := xavier_prob * yvonne_prob * (1 - zelda_prob)

-- Theorem statement
theorem xavier_yvonne_not_zelda_prob : desired_outcome_prob = 1/16 := by
  sorry

end xavier_yvonne_not_zelda_prob_l3024_302489


namespace vacant_seats_l3024_302470

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) 
  (h1 : total_seats = 600) 
  (h2 : filled_percentage = 45/100) : 
  ℕ := by
  sorry

end vacant_seats_l3024_302470


namespace det_equality_l3024_302474

theorem det_equality (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = 7 →
  Matrix.det !![x - 2*z, y - 2*w; z, w] = 7 := by
  sorry

end det_equality_l3024_302474


namespace intersection_points_correct_l3024_302438

/-- The number of intersection points of segments joining m distinct points 
    on the positive x-axis to n distinct points on the positive y-axis, 
    where no three segments are concurrent. -/
def intersectionPoints (m n : ℕ) : ℕ :=
  m * n * (m - 1) * (n - 1) / 4

/-- Theorem stating that the number of intersection points is correct. -/
theorem intersection_points_correct (m n : ℕ) :
  intersectionPoints m n = m * n * (m - 1) * (n - 1) / 4 :=
by sorry

end intersection_points_correct_l3024_302438


namespace sandwich_combinations_l3024_302440

theorem sandwich_combinations (n_meat : Nat) (n_cheese : Nat) : 
  n_meat = 10 → n_cheese = 9 → n_meat * (n_cheese.choose 2) = 360 := by
  sorry

end sandwich_combinations_l3024_302440


namespace perfect_square_trinomial_m_value_l3024_302430

/-- A trinomial ax^2 + bx + c is a perfect square if there exists a real number d such that ax^2 + bx + c = (dx + e)^2 for all x -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ d e : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (d * x + e)^2

theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, is_perfect_square_trinomial 1 (-4) m → m = 4 := by
  sorry

end perfect_square_trinomial_m_value_l3024_302430


namespace quartic_equation_integer_roots_l3024_302434

theorem quartic_equation_integer_roots :
  let f (x : ℤ) (a : ℤ) := x^4 - 16*x^3 + (81-2*a)*x^2 + (16*a-142)*x + a^2 - 21*a + 68
  ∃ (a : ℤ), a = -4 ∧ (∀ x : ℤ, f x a = 0 ↔ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 7) := by
  sorry

end quartic_equation_integer_roots_l3024_302434


namespace sale_result_l3024_302407

/-- Represents the total number of cases of cat food sold during a sale. -/
def total_cases_sold (first_group : Nat) (second_group : Nat) (third_group : Nat) 
  (first_group_cases : Nat) (second_group_cases : Nat) (third_group_cases : Nat) : Nat :=
  first_group * first_group_cases + second_group * second_group_cases + third_group * third_group_cases

/-- Theorem stating that the total number of cases sold is 40 given the specific customer purchase patterns. -/
theorem sale_result : 
  total_cases_sold 8 4 8 3 2 1 = 40 := by
  sorry

#check sale_result

end sale_result_l3024_302407


namespace geometric_sequence_middle_term_l3024_302452

theorem geometric_sequence_middle_term 
  (a b c : ℝ) 
  (pos_a : 0 < a) 
  (pos_b : 0 < b) 
  (pos_c : 0 < c) 
  (geom_seq : b^2 = a * c) 
  (def_a : a = 5 + 2 * Real.sqrt 3) 
  (def_c : c = 5 - 2 * Real.sqrt 3) : 
  b = Real.sqrt 13 := by
sorry

end geometric_sequence_middle_term_l3024_302452


namespace f_of_tan_squared_l3024_302485

noncomputable def f (x : ℝ) : ℝ := 1 / ((x / (x - 1)))

theorem f_of_tan_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π/2) :
  f (Real.tan t ^ 2) = Real.tan t ^ 2 :=
by sorry

end f_of_tan_squared_l3024_302485


namespace even_odd_square_sum_l3024_302415

theorem even_odd_square_sum (a b : ℕ) :
  (Even (a * b) → ∃ c d : ℕ, a^2 + b^2 + c^2 = d^2) ∧
  (Odd (a * b) → ¬∃ c d : ℕ, a^2 + b^2 + c^2 = d^2) :=
by sorry

end even_odd_square_sum_l3024_302415


namespace division_problem_l3024_302446

theorem division_problem :
  ∃ (dividend : ℕ), 
    dividend = 11889708 ∧ 
    dividend / 12 = 990809 ∧ 
    dividend % 12 = 0 :=
by sorry

end division_problem_l3024_302446


namespace max_sides_1950_gon_l3024_302424

/-- A convex polygon with n sides --/
structure ConvexPolygon (n : ℕ) where
  -- Add necessary fields here
  sides : n > 2

/-- The result of drawing all diagonals in a convex polygon --/
def drawAllDiagonals (p : ConvexPolygon n) : Set (ConvexPolygon m) :=
  sorry

/-- The maximum number of sides among the resulting polygons after drawing all diagonals --/
def maxResultingSides (p : ConvexPolygon n) : ℕ :=
  sorry

theorem max_sides_1950_gon :
  ∀ (p : ConvexPolygon 1950),
  maxResultingSides p = 1949 :=
sorry

end max_sides_1950_gon_l3024_302424


namespace intersection_A_B_when_a_neg_one_range_of_a_when_A_subset_B_l3024_302499

/-- Definition of set A -/
def A (a : ℝ) : Set ℝ := {x : ℝ | 0 < 2*x + a ∧ 2*x + a ≤ 3}

/-- Definition of set B -/
def B : Set ℝ := {x : ℝ | -1/2 < x ∧ x < 2}

/-- Theorem for the intersection of A and B when a = -1 -/
theorem intersection_A_B_when_a_neg_one :
  A (-1) ∩ B = {x : ℝ | 1/2 < x ∧ x < 2} := by sorry

/-- Theorem for the range of a when A is a subset of B -/
theorem range_of_a_when_A_subset_B :
  ∀ a : ℝ, A a ⊆ B ↔ -1 < a ∧ a ≤ 1 := by sorry

end intersection_A_B_when_a_neg_one_range_of_a_when_A_subset_B_l3024_302499


namespace expected_count_in_sample_l3024_302431

/-- 
Given a population where 1/4 of the members have a certain characteristic,
prove that the expected number of individuals with that characteristic
in a random sample of 300 is 75.
-/
theorem expected_count_in_sample 
  (population_probability : ℚ) 
  (sample_size : ℕ) 
  (h1 : population_probability = 1 / 4) 
  (h2 : sample_size = 300) : 
  population_probability * sample_size = 75 := by
sorry

end expected_count_in_sample_l3024_302431


namespace chairs_moved_by_pat_l3024_302448

theorem chairs_moved_by_pat (total_chairs : ℕ) (careys_chairs : ℕ) (chairs_left : ℕ) 
  (h1 : total_chairs = 74)
  (h2 : careys_chairs = 28)
  (h3 : chairs_left = 17) :
  total_chairs - careys_chairs - chairs_left = 29 := by
  sorry

end chairs_moved_by_pat_l3024_302448


namespace quadratic_factorization_l3024_302462

theorem quadratic_factorization (a : ℤ) : 
  (∃ m n p q : ℤ, (15 : ℤ) * x^2 + a * x + (15 : ℤ) = (m * x + n) * (p * x + q) ∧ 
   Nat.Prime m.natAbs ∧ Nat.Prime p.natAbs) → 
  ∃ k : ℤ, a = 2 * k := by
  sorry

end quadratic_factorization_l3024_302462


namespace no_linear_term_condition_l3024_302443

theorem no_linear_term_condition (p q : ℝ) : 
  (∀ x : ℝ, (x^2 - p*x + q)*(x - 3) = x^3 + (-p-3)*x^2 + 0*x + (-3*q)) → 
  q + 3*p = 0 := by
sorry

end no_linear_term_condition_l3024_302443


namespace angle_in_third_quadrant_l3024_302417

theorem angle_in_third_quadrant (α : Real) : 
  (π / 2 < α ∧ α < π) → (π < π / 2 + α ∧ π / 2 + α < 3 * π / 2) := by
  sorry

end angle_in_third_quadrant_l3024_302417


namespace system_one_solution_system_two_solution_l3024_302465

-- System of equations (1)
theorem system_one_solution (x y : ℝ) : 
  3*x - 2*y = 6 ∧ 2*x + 3*y = 17 → x = 4 ∧ y = 3 := by
sorry

-- System of equations (2)
theorem system_two_solution (x y : ℝ) :
  x + 4*y = 14 ∧ (x-3)/4 - (y-3)/3 = 1/12 → x = 3 ∧ y = 11/4 := by
sorry

end system_one_solution_system_two_solution_l3024_302465


namespace smallest_multiple_l3024_302406

theorem smallest_multiple : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (∃ (k : ℕ), a = 5 * k) ∧ 
  (∃ (m : ℕ), a + 1 = 7 * m) ∧ 
  (∃ (n : ℕ), a + 2 = 9 * n) ∧ 
  (∃ (p : ℕ), a + 3 = 11 * p) ∧ 
  (∀ (b : ℕ), 
    (b > 0) ∧ 
    (∃ (k : ℕ), b = 5 * k) ∧ 
    (∃ (m : ℕ), b + 1 = 7 * m) ∧ 
    (∃ (n : ℕ), b + 2 = 9 * n) ∧ 
    (∃ (p : ℕ), b + 3 = 11 * p) 
    → b ≥ a) ∧
  a = 1735 :=
by sorry

end smallest_multiple_l3024_302406


namespace rectangular_field_area_l3024_302403

/-- Represents a rectangular field with given properties -/
structure RectangularField where
  breadth : ℝ
  length : ℝ
  perimeter : ℝ
  length_constraint : length = breadth + 30
  perimeter_constraint : perimeter = 2 * (length + breadth)

/-- Theorem: Area of the rectangular field with given constraints is 18000 square meters -/
theorem rectangular_field_area (field : RectangularField) (h : field.perimeter = 540) :
  field.length * field.breadth = 18000 := by
  sorry

#check rectangular_field_area

end rectangular_field_area_l3024_302403


namespace distribute_five_three_l3024_302422

/-- The number of ways to distribute n distinct elements into k distinct groups,
    where each group must contain at least one element. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 150 ways to distribute 5 distinct elements into 3 distinct groups,
    where each group must contain at least one element. -/
theorem distribute_five_three : distribute 5 3 = 150 := by
  sorry

end distribute_five_three_l3024_302422


namespace triangle_side_length_l3024_302401

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if A = 60°, b = 4, and the area is 2√3, then a = 2√3 -/
theorem triangle_side_length (a b c : ℝ) (A : Real) (S : ℝ) :
  A = π / 3 →  -- 60° in radians
  b = 4 →
  S = 2 * Real.sqrt 3 →
  S = 1 / 2 * b * c * Real.sin A →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
  a = 2 * Real.sqrt 3 := by
  sorry

end triangle_side_length_l3024_302401


namespace average_weight_increase_l3024_302410

theorem average_weight_increase (initial_count : ℕ) (replaced_weight original_weight : ℝ) :
  initial_count = 8 →
  replaced_weight = 65 →
  original_weight = 85 →
  (original_weight - replaced_weight) / initial_count = 2.5 := by
  sorry

end average_weight_increase_l3024_302410


namespace max_blocks_in_box_l3024_302486

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of blocks that can fit in a box -/
def maxBlocksFit (box : BoxDimensions) (block : BlockDimensions) : ℕ :=
  (box.length / block.length) * (box.width / block.width) * (box.height / block.height)

theorem max_blocks_in_box :
  let box : BoxDimensions := ⟨5, 4, 3⟩
  let block : BlockDimensions := ⟨1, 2, 2⟩
  maxBlocksFit box block = 12 := by
  sorry


end max_blocks_in_box_l3024_302486


namespace jimmy_stair_time_l3024_302497

/-- The sum of an arithmetic sequence -/
def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Jimmy's stair climbing time -/
theorem jimmy_stair_time : arithmeticSum 20 7 7 = 287 := by
  sorry

end jimmy_stair_time_l3024_302497


namespace spinner_divisible_by_three_probability_l3024_302439

/-- Represents the possible outcomes of the spinner -/
inductive SpinnerOutcome
  | One
  | Two
  | Four

/-- Represents a three-digit number formed by three spins -/
structure ThreeDigitNumber where
  hundreds : SpinnerOutcome
  tens : SpinnerOutcome
  units : SpinnerOutcome

/-- Converts a SpinnerOutcome to its numerical value -/
def spinnerValue (outcome : SpinnerOutcome) : Nat :=
  match outcome with
  | SpinnerOutcome.One => 1
  | SpinnerOutcome.Two => 2
  | SpinnerOutcome.Four => 4

/-- Checks if a ThreeDigitNumber is divisible by 3 -/
def isDivisibleByThree (n : ThreeDigitNumber) : Bool :=
  (spinnerValue n.hundreds + spinnerValue n.tens + spinnerValue n.units) % 3 = 0

/-- Calculates the probability of getting a number divisible by 3 -/
def probabilityDivisibleByThree : ℚ :=
  let totalOutcomes := 27  -- 3^3
  let favorableOutcomes := 6  -- Counted from the problem
  favorableOutcomes / totalOutcomes

/-- Main theorem: The probability of getting a number divisible by 3 is 2/9 -/
theorem spinner_divisible_by_three_probability :
  probabilityDivisibleByThree = 2 / 9 := by
  sorry

end spinner_divisible_by_three_probability_l3024_302439


namespace sum_difference_theorem_l3024_302478

/-- Rounds a number to the nearest multiple of 5, rounding 5s up -/
def roundToNearestFive (n : ℕ) : ℕ :=
  ((n + 2) / 5) * 5

/-- Sums all integers from 1 to n -/
def sumToN (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Sums all integers from 1 to n after rounding each to the nearest multiple of 5 -/
def sumRoundedToN (n : ℕ) : ℕ :=
  List.sum (List.map roundToNearestFive (List.range n))

theorem sum_difference_theorem :
  sumToN 100 - sumRoundedToN 100 = 4750 :=
sorry

end sum_difference_theorem_l3024_302478


namespace wash_time_proof_l3024_302405

/-- The number of weeks between each wash -/
def wash_interval : ℕ := 4

/-- The time in minutes it takes to wash the pillowcases -/
def wash_time : ℕ := 30

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- Calculates the total time spent washing pillowcases in a year -/
def total_wash_time_per_year : ℕ :=
  (weeks_per_year / wash_interval) * wash_time

theorem wash_time_proof :
  total_wash_time_per_year = 390 :=
by sorry

end wash_time_proof_l3024_302405


namespace odd_digits_in_560_base9_l3024_302437

/-- Converts a natural number from base 10 to base 9 --/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers --/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

theorem odd_digits_in_560_base9 :
  countOddDigits (toBase9 560) = 0 := by
  sorry

end odd_digits_in_560_base9_l3024_302437


namespace jude_current_age_l3024_302498

/-- Heath's age today -/
def heath_age_today : ℕ := 16

/-- The number of years in the future when the age comparison is made -/
def years_in_future : ℕ := 5

/-- Heath's age in the future -/
def heath_age_future : ℕ := heath_age_today + years_in_future

/-- Jude's age in the future -/
def jude_age_future : ℕ := heath_age_future / 3

/-- Jude's age today -/
def jude_age_today : ℕ := jude_age_future - years_in_future

theorem jude_current_age : jude_age_today = 2 := by
  sorry

end jude_current_age_l3024_302498
