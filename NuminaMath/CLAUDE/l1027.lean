import Mathlib

namespace isosceles_triangle_l1027_102712

theorem isosceles_triangle (A B C : Real) (h : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) : A = B := by
  sorry

end isosceles_triangle_l1027_102712


namespace reflect_x_axis_l1027_102774

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectX (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflect_x_axis (p : Point) (h : p = Point.mk 3 1) :
  reflectX p = Point.mk 3 (-1) := by
  sorry

end reflect_x_axis_l1027_102774


namespace soccer_team_strikers_l1027_102783

theorem soccer_team_strikers (goalies defenders midfielders strikers total : ℕ) : 
  goalies = 3 →
  defenders = 10 →
  midfielders = 2 * defenders →
  total = 40 →
  strikers = total - (goalies + defenders + midfielders) →
  strikers = 7 := by
sorry

end soccer_team_strikers_l1027_102783


namespace isosceles_right_triangle_line_equation_l1027_102753

/-- A line that passes through a point and forms an isosceles right triangle with coordinate axes -/
structure IsoscelesRightTriangleLine where
  -- The slope of the line
  slope : ℝ
  -- The y-intercept of the line
  y_intercept : ℝ
  -- The point that the line passes through
  point : ℝ × ℝ
  -- The line passes through the given point
  point_on_line : y_intercept = point.2 - slope * point.1
  -- The line forms an isosceles right triangle with coordinate axes
  isosceles_right_triangle : slope = 1 ∨ slope = -1

/-- The equation of a line that passes through (2,3) and forms an isosceles right triangle with coordinate axes -/
theorem isosceles_right_triangle_line_equation (l : IsoscelesRightTriangleLine) 
  (h : l.point = (2, 3)) :
  (∀ x y, y = l.slope * x + l.y_intercept ↔ x + y - 5 = 0) ∨
  (∀ x y, y = l.slope * x + l.y_intercept ↔ x - y + 1 = 0) := by
  sorry

end isosceles_right_triangle_line_equation_l1027_102753


namespace problem_statement_l1027_102756

theorem problem_statement : (12 : ℕ)^5 * 6^5 / 432^3 = 24 := by
  sorry

end problem_statement_l1027_102756


namespace sum_interior_angles_regular_pentagon_l1027_102733

/-- The sum of interior angles of a regular polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A regular pentagon has 5 sides -/
def regular_pentagon_sides : ℕ := 5

/-- Theorem: The sum of the interior angles of a regular pentagon is 540 degrees -/
theorem sum_interior_angles_regular_pentagon :
  sum_interior_angles regular_pentagon_sides = 540 := by
  sorry


end sum_interior_angles_regular_pentagon_l1027_102733


namespace square_side_length_l1027_102734

theorem square_side_length (m n : ℝ) :
  let area := 9*m^2 + 24*m*n + 16*n^2
  ∃ (side : ℝ), side ≥ 0 ∧ side^2 = area ∧ side = |3*m + 4*n| :=
by sorry

end square_side_length_l1027_102734


namespace calculation_proof_l1027_102771

theorem calculation_proof : 2359 + 180 / 60 * 3 - 359 = 2009 := by
  sorry

end calculation_proof_l1027_102771


namespace geometric_sequence_sixth_term_l1027_102707

theorem geometric_sequence_sixth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a = 512) 
  (h2 : a * r^7 = 2) : 
  a * r^5 = 16 := by
sorry

end geometric_sequence_sixth_term_l1027_102707


namespace merchant_profit_l1027_102701

theorem merchant_profit (cost : ℝ) (selling_price : ℝ) : 
  cost = 30 ∧ selling_price = 39 → 
  selling_price = cost + (cost * cost / 100) := by
sorry

end merchant_profit_l1027_102701


namespace adjacent_vertices_probability_l1027_102785

/-- A decagon is a polygon with 10 vertices -/
def Decagon := {n : ℕ // n = 10}

/-- The number of vertices in a decagon -/
def numVertices : Decagon → ℕ := fun _ ↦ 10

/-- The number of adjacent vertices for each vertex in a decagon -/
def numAdjacentVertices : Decagon → ℕ := fun _ ↦ 2

/-- The probability of selecting two adjacent vertices in a decagon -/
def probAdjacentVertices (d : Decagon) : ℚ :=
  (numAdjacentVertices d : ℚ) / ((numVertices d - 1) : ℚ)

theorem adjacent_vertices_probability (d : Decagon) :
  probAdjacentVertices d = 2/9 := by sorry

end adjacent_vertices_probability_l1027_102785


namespace remainder_55_power_55_plus_10_mod_8_l1027_102728

theorem remainder_55_power_55_plus_10_mod_8 : 55^55 + 10 ≡ 1 [ZMOD 8] := by
  sorry

end remainder_55_power_55_plus_10_mod_8_l1027_102728


namespace product_multiple_of_16_probability_l1027_102706

def S : Finset ℕ := {3, 4, 8, 16}

theorem product_multiple_of_16_probability :
  let pairs := S.powerset.filter (λ p : Finset ℕ => p.card = 2)
  let valid_pairs := pairs.filter (λ p : Finset ℕ => (p.prod id) % 16 = 0)
  (valid_pairs.card : ℚ) / pairs.card = 1 / 3 := by
sorry

end product_multiple_of_16_probability_l1027_102706


namespace tangent_line_equation_l1027_102724

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 4

-- Define the point of tangency
def point : ℝ × ℝ := (1, 1)

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x, (deriv f) x = 3*x^2 - 8*x) ∧
    (deriv f) (point.1) = m ∧
    f point.1 = point.2 ∧
    (∀ x, m * (x - point.1) + point.2 = -5 * x + 6) := by
  sorry

end tangent_line_equation_l1027_102724


namespace frog_jump_distance_l1027_102703

/-- The jumping contest problem -/
theorem frog_jump_distance 
  (grasshopper_jump : ℕ) 
  (grasshopper_frog_diff : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : grasshopper_jump = grasshopper_frog_diff + frog_jump) :
  frog_jump = 15 :=
by
  sorry

#check frog_jump_distance

end frog_jump_distance_l1027_102703


namespace supplement_of_beta_l1027_102794

def complementary_angles (α β : Real) : Prop := α + β = 90

theorem supplement_of_beta (α β : Real) 
  (h1 : complementary_angles α β) 
  (h2 : α = 30) : 
  180 - β = 120 := by
  sorry

end supplement_of_beta_l1027_102794


namespace minimum_cost_theorem_l1027_102765

/-- Represents the number and cost of diesel generators --/
structure DieselGenerators where
  totalCount : Nat
  typeACount : Nat
  typeBCount : Nat
  typeCCount : Nat
  typeACost : Nat
  typeBCost : Nat
  typeCCost : Nat

/-- Represents the irrigation capacity of the generators --/
def irrigationCapacity (g : DieselGenerators) : Nat :=
  4 * g.typeACount + 3 * g.typeBCount + 2 * g.typeCCount

/-- Represents the total cost of operating the generators --/
def operatingCost (g : DieselGenerators) : Nat :=
  g.typeACost * g.typeACount + g.typeBCost * g.typeBCount + g.typeCCost * g.typeCCount

/-- Theorem stating the minimum cost of operation --/
theorem minimum_cost_theorem (g : DieselGenerators) :
  g.totalCount = 10 ∧
  g.typeACount > 0 ∧ g.typeBCount > 0 ∧ g.typeCCount > 0 ∧
  g.typeACount + g.typeBCount + g.typeCCount = g.totalCount ∧
  irrigationCapacity g = 32 ∧
  g.typeACost = 130 ∧ g.typeBCost = 120 ∧ g.typeCCost = 100 →
  ∃ (minCost : Nat), minCost = 1190 ∧
    ∀ (h : DieselGenerators), 
      h.totalCount = 10 ∧
      h.typeACount > 0 ∧ h.typeBCount > 0 ∧ h.typeCCount > 0 ∧
      h.typeACount + h.typeBCount + h.typeCCount = h.totalCount ∧
      irrigationCapacity h = 32 ∧
      h.typeACost = 130 ∧ h.typeBCost = 120 ∧ h.typeCCost = 100 →
      operatingCost h ≥ minCost := by
  sorry

end minimum_cost_theorem_l1027_102765


namespace complex_number_in_first_quadrant_l1027_102719

theorem complex_number_in_first_quadrant :
  let z : ℂ := Complex.I / (2 + Complex.I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end complex_number_in_first_quadrant_l1027_102719


namespace subset_pairs_count_l1027_102776

/-- Given a fixed set S with n elements, this theorem states that the number of ordered pairs (A, B) 
    where A and B are subsets of S and A ⊆ B is equal to 3^n. -/
theorem subset_pairs_count (n : ℕ) : 
  (Finset.powerset (Finset.range n)).card = 3^n := by sorry

end subset_pairs_count_l1027_102776


namespace exist_natural_solution_l1027_102745

theorem exist_natural_solution :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 := by
sorry

end exist_natural_solution_l1027_102745


namespace chicken_distribution_problem_l1027_102772

/-- The multiple of Skylar's chickens that Quentin has 25 more than -/
def chicken_multiple (total colten skylar quentin : ℕ) : ℕ :=
  (quentin - 25) / skylar

/-- Proof of the chicken distribution problem -/
theorem chicken_distribution_problem (total colten skylar quentin : ℕ) 
  (h1 : total = 383)
  (h2 : colten = 37)
  (h3 : skylar = 3 * colten - 4)
  (h4 : quentin + skylar + colten = total)
  (h5 : ∃ m : ℕ, quentin = m * skylar + 25) :
  chicken_multiple total colten skylar quentin = 2 := by
  sorry

#eval chicken_multiple 383 37 107 239

end chicken_distribution_problem_l1027_102772


namespace fraction_equality_l1027_102746

theorem fraction_equality (p q : ℚ) (h : p / q = 4 / 5) :
  11 / 7 + (2 * q - p) / (2 * q + p) = 2 := by
  sorry

end fraction_equality_l1027_102746


namespace fifteen_plus_neg_twentythree_l1027_102744

-- Define the operation for adding a positive and negative rational number
def add_pos_neg (a b : ℚ) : ℚ := -(b - a)

-- Theorem statement
theorem fifteen_plus_neg_twentythree :
  15 + (-23) = add_pos_neg 15 23 :=
sorry

end fifteen_plus_neg_twentythree_l1027_102744


namespace f_is_linear_l1027_102790

def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

def f (x : ℝ) : ℝ := -2 * x

theorem f_is_linear : is_linear f := by sorry

end f_is_linear_l1027_102790


namespace original_group_size_l1027_102715

theorem original_group_size (initial_days work_days : ℕ) (absent_men : ℕ) : 
  initial_days = 15 →
  absent_men = 8 →
  work_days = 18 →
  ∃ (original_size : ℕ),
    original_size * initial_days = (original_size - absent_men) * work_days ∧
    original_size = 48 :=
by sorry

end original_group_size_l1027_102715


namespace product_remainder_zero_l1027_102714

theorem product_remainder_zero (a b c : ℕ) (ha : a = 1256) (hb : b = 7921) (hc : c = 70305) :
  (a * b * c) % 10 = 0 := by
  sorry

end product_remainder_zero_l1027_102714


namespace regular_polygon_sides_l1027_102784

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  n > 2 → 
  exterior_angle = 20 → 
  (n : ℝ) * exterior_angle = 360 →
  n = 18 := by
  sorry

end regular_polygon_sides_l1027_102784


namespace meals_sold_equals_twelve_l1027_102702

/-- Represents the number of meals sold during lunch in a restaurant -/
def meals_sold_during_lunch (lunch_meals : ℕ) (dinner_prep : ℕ) (dinner_available : ℕ) : ℕ :=
  lunch_meals + dinner_prep - dinner_available

/-- Theorem stating that the number of meals sold during lunch is 12 -/
theorem meals_sold_equals_twelve : 
  meals_sold_during_lunch 17 5 10 = 12 := by
  sorry

end meals_sold_equals_twelve_l1027_102702


namespace part1_part2_l1027_102760

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 4) ≤ 0
def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

-- Part 1
theorem part1 (m : ℝ) (h1 : m > 0) 
  (h2 : ∀ x, ¬(q m x) → ¬(p x)) 
  (h3 : ∃ x, ¬(p x) ∧ q m x) : 
  m ≥ 4 := by sorry

-- Part 2
theorem part2 (x : ℝ) 
  (h1 : p x ∨ q 5 x) 
  (h2 : ¬(p x ∧ q 5 x)) : 
  x ∈ Set.Icc (-3 : ℝ) (-2) ∪ Set.Ioc 4 7 := by sorry

end part1_part2_l1027_102760


namespace quadratic_inequality_implies_a_geq_5_l1027_102700

theorem quadratic_inequality_implies_a_geq_5 (a : ℝ) : 
  (∀ x : ℝ, 1 < x ∧ x < 5 → x^2 - 2*(a-2)*x + a < 0) → a ≥ 5 := by
sorry

end quadratic_inequality_implies_a_geq_5_l1027_102700


namespace function_properties_l1027_102742

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) + x) + (2^x - 1) / (2^x + 1) + 3

def g (x : ℝ) : ℝ := sorry

theorem function_properties :
  (∀ x : ℝ, f x + f (-x) = 6) ∧
  (∀ x : ℝ, g x + g (-x) = 6) ∧
  (∀ a b : ℝ, f a + f b > 6 → a + b > 0) := by sorry

end function_properties_l1027_102742


namespace problem_1_problem_2_problem_3_l1027_102748

-- Problem 1
theorem problem_1 : (-36 : ℚ) * (5/4 - 5/6 - 11/12) = 18 := by sorry

-- Problem 2
theorem problem_2 : (-2)^2 - 3 * (-1)^3 + 0 * (-2)^3 = 7 := by sorry

-- Problem 3
theorem problem_3 (x y : ℚ) (hx : x = -2) (hy : y = 1/2) :
  3 * x^2 * y - 2 * x * y^2 - 3/2 * (x^2 * y - 2 * x * y^2) = 5/2 := by sorry

end problem_1_problem_2_problem_3_l1027_102748


namespace picnic_blankets_theorem_l1027_102799

/-- The area of a blanket after a given number of folds -/
def folded_area (initial_area : ℕ) (num_folds : ℕ) : ℕ :=
  initial_area / 2^num_folds

/-- The total area of multiple blankets after folding -/
def total_folded_area (num_blankets : ℕ) (initial_area : ℕ) (num_folds : ℕ) : ℕ :=
  num_blankets * folded_area initial_area num_folds

theorem picnic_blankets_theorem :
  total_folded_area 3 64 4 = 12 := by
  sorry

end picnic_blankets_theorem_l1027_102799


namespace distance_to_third_side_l1027_102738

/-- Represents a point inside an equilateral triangle -/
structure PointInTriangle where
  /-- Distance to the first side -/
  d1 : ℝ
  /-- Distance to the second side -/
  d2 : ℝ
  /-- Distance to the third side -/
  d3 : ℝ
  /-- The sum of distances equals the triangle's height -/
  sum_eq_height : d1 + d2 + d3 = 5 * Real.sqrt 3

/-- Theorem: In an equilateral triangle with side length 10, if a point inside
    has distances 1 and 3 to two sides, its distance to the third side is 5√3 - 4 -/
theorem distance_to_third_side
  (P : PointInTriangle)
  (h1 : P.d1 = 1)
  (h2 : P.d2 = 3) :
  P.d3 = 5 * Real.sqrt 3 - 4 := by
  sorry


end distance_to_third_side_l1027_102738


namespace sum_power_mod_five_l1027_102757

theorem sum_power_mod_five (n : ℕ) :
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := by
  sorry

end sum_power_mod_five_l1027_102757


namespace calculate_expression_solve_equation_l1027_102739

-- Problem 1
theorem calculate_expression : -3^2 + 5 * (-8/5) - (-4)^2 / (-8) = -13 := by sorry

-- Problem 2
theorem solve_equation : 
  ∃ x : ℚ, (x + 1) / 2 - 2 = x / 4 ∧ x = -4/3 := by sorry

end calculate_expression_solve_equation_l1027_102739


namespace photo_arrangement_l1027_102768

/-- The number of ways to select and permute 3 people out of 8, keeping the rest in place -/
theorem photo_arrangement (n m : ℕ) (hn : n = 8) (hm : m = 3) : 
  (n.choose m) * (Nat.factorial m) = 336 := by
  sorry

end photo_arrangement_l1027_102768


namespace greg_trip_distance_l1027_102722

/-- Represents Greg's trip with given distances and speeds -/
structure GregTrip where
  workplace_to_market : ℝ
  market_to_friend : ℝ
  friend_to_aunt : ℝ
  aunt_to_grocery : ℝ
  grocery_to_home : ℝ

/-- Calculates the total distance of Greg's trip -/
def total_distance (trip : GregTrip) : ℝ :=
  trip.workplace_to_market + trip.market_to_friend + trip.friend_to_aunt + 
  trip.aunt_to_grocery + trip.grocery_to_home

/-- Theorem stating that Greg's total trip distance is 100 miles -/
theorem greg_trip_distance :
  ∃ (trip : GregTrip),
    trip.workplace_to_market = 30 ∧
    trip.market_to_friend = trip.workplace_to_market + 10 ∧
    trip.friend_to_aunt = 5 ∧
    trip.aunt_to_grocery = 7 ∧
    trip.grocery_to_home = 18 ∧
    total_distance trip = 100 :=
by
  sorry

end greg_trip_distance_l1027_102722


namespace a_value_proof_l1027_102727

theorem a_value_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h1 : a ^ b = b ^ a) (h2 : b = 4 * a) : a = (4 : ℝ) ^ (1/3) := by
  sorry

end a_value_proof_l1027_102727


namespace right_triangle_xy_length_l1027_102761

theorem right_triangle_xy_length 
  (X Y Z : ℝ × ℝ) 
  (right_angle : (X.1 - Y.1) * (X.1 - Z.1) + (X.2 - Y.2) * (X.2 - Z.2) = 0) 
  (yz_length : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 20)
  (tan_z_eq : (X.2 - Y.2) / (X.1 - Y.1) = 4 * (X.1 - Y.1) / 20) :
  Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = 5 * Real.sqrt 15 := by
sorry

end right_triangle_xy_length_l1027_102761


namespace combined_males_below_50_l1027_102786

/-- Represents an office branch with employee information -/
structure Branch where
  total_employees : ℕ
  male_percentage : ℚ
  male_over_50_percentage : ℚ

/-- Calculates the number of males below 50 in a branch -/
def males_below_50 (b : Branch) : ℚ :=
  b.total_employees * b.male_percentage * (1 - b.male_over_50_percentage)

/-- The given information about the three branches -/
def branch_A : Branch :=
  { total_employees := 4500
  , male_percentage := 60 / 100
  , male_over_50_percentage := 40 / 100 }

def branch_B : Branch :=
  { total_employees := 3500
  , male_percentage := 50 / 100
  , male_over_50_percentage := 55 / 100 }

def branch_C : Branch :=
  { total_employees := 2200
  , male_percentage := 35 / 100
  , male_over_50_percentage := 70 / 100 }

/-- The main theorem stating the combined number of males below 50 -/
theorem combined_males_below_50 :
  ⌊males_below_50 branch_A + males_below_50 branch_B + males_below_50 branch_C⌋ = 2638 := by
  sorry

end combined_males_below_50_l1027_102786


namespace jordan_scoring_breakdown_l1027_102754

/-- Represents the scoring statistics of a basketball player in a game. -/
structure ScoringStats where
  total_points : ℕ
  total_shots : ℕ
  total_hits : ℕ
  three_pointers_made : ℕ
  three_pointer_attempts : ℕ

/-- Calculates the number of 2-point shots and free throws made given scoring statistics. -/
def calculate_shots (stats : ScoringStats) : ℕ × ℕ := sorry

/-- Theorem stating that given Jordan's scoring statistics, he made 8 2-point shots and 3 free throws. -/
theorem jordan_scoring_breakdown (stats : ScoringStats) 
  (h1 : stats.total_points = 28)
  (h2 : stats.total_shots = 24)
  (h3 : stats.total_hits = 14)
  (h4 : stats.three_pointers_made = 3)
  (h5 : stats.three_pointer_attempts = 3) :
  calculate_shots stats = (8, 3) := by sorry

end jordan_scoring_breakdown_l1027_102754


namespace square_sum_equals_69_l1027_102763

/-- Given a system of equations, prove that x₀² + y₀² = 69 -/
theorem square_sum_equals_69 
  (x₀ y₀ c : ℝ) 
  (h1 : x₀ * y₀ = 6)
  (h2 : x₀^2 * y₀ + x₀ * y₀^2 + x₀ + y₀ + c = 2) :
  x₀^2 + y₀^2 = 69 := by
  sorry

end square_sum_equals_69_l1027_102763


namespace largest_angle_in_triangle_l1027_102796

theorem largest_angle_in_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 4/3 of a right angle
  a + b = 4/3 * 90
  -- One angle is 36° larger than the other
  → b = a + 36
  -- All angles are non-negative
  → a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0
  -- Sum of all angles in a triangle is 180°
  → a + b + c = 180
  -- The largest angle is 78°
  → max a (max b c) = 78 :=
by
  sorry

end largest_angle_in_triangle_l1027_102796


namespace sin_alpha_for_point_l1027_102750

theorem sin_alpha_for_point (α : Real) :
  let P : ℝ × ℝ := (1, -Real.sqrt 3)
  (∃ (t : ℝ), t > 0 ∧ P = (t * Real.cos α, t * Real.sin α)) →
  Real.sin α = -Real.sqrt 3 / 2 := by
sorry

end sin_alpha_for_point_l1027_102750


namespace pencil_count_l1027_102758

/-- The number of pencils Mitchell and Antonio have together -/
def total_pencils (mitchell_pencils : ℕ) (difference : ℕ) : ℕ :=
  mitchell_pencils + (mitchell_pencils - difference)

/-- Theorem stating the total number of pencils Mitchell and Antonio have -/
theorem pencil_count (mitchell_pencils : ℕ) (difference : ℕ) 
  (h1 : mitchell_pencils = 30)
  (h2 : difference = 6) : 
  total_pencils mitchell_pencils difference = 54 := by
  sorry

end pencil_count_l1027_102758


namespace problem_1_l1027_102798

theorem problem_1 : (1 : ℝ) * (1 - 2 * Real.sqrt 3) * (1 + 2 * Real.sqrt 3) - (1 + Real.sqrt 3)^2 = -15 - 2 * Real.sqrt 3 := by
  sorry

end problem_1_l1027_102798


namespace parabola_line_intersection_property_l1027_102755

/-- Parabola type representing y² = 4x -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ

/-- Line type representing y = k(x-1) -/
structure Line where
  k : ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector dot product -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem parabola_line_intersection_property
  (C : Parabola)
  (l : Line)
  (A B M N O : Point)
  (hC : C.focus = (1, 0) ∧ C.directrix = -1)
  (hl : l.k ≠ 0)
  (hA : A.y^2 = 4 * A.x ∧ A.y = l.k * (A.x - 1))
  (hB : B.y^2 = 4 * B.x ∧ B.y = l.k * (B.x - 1))
  (hM : M.x = -1 ∧ M.y * A.x = -A.y)
  (hN : N.x = -1 ∧ N.y * B.x = -B.y)
  (hO : O.x = 0 ∧ O.y = 0) :
  dot_product (M.x - O.x, M.y - O.y) (N.x - O.x, N.y - O.y) =
  dot_product (A.x - O.x, A.y - O.y) (B.x - O.x, B.y - O.y) :=
sorry

end parabola_line_intersection_property_l1027_102755


namespace money_in_pond_is_637_l1027_102743

/-- The amount of money in cents left in the pond after all calculations -/
def moneyInPond : ℕ :=
  let dimeValue : ℕ := 10
  let quarterValue : ℕ := 25
  let halfDollarValue : ℕ := 50
  let dollarValue : ℕ := 100
  let nickelValue : ℕ := 5
  let pennyValue : ℕ := 1
  let foreignCoinValue : ℕ := 25

  let cindyMoney : ℕ := 5 * dimeValue + 3 * halfDollarValue
  let ericMoney : ℕ := 3 * quarterValue + 2 * dollarValue + halfDollarValue
  let garrickMoney : ℕ := 8 * nickelValue + 7 * pennyValue
  let ivyMoney : ℕ := 60 * pennyValue + 5 * foreignCoinValue

  let totalBefore : ℕ := cindyMoney + ericMoney + garrickMoney + ivyMoney

  let beaumontRemoval : ℕ := 2 * dimeValue + 3 * nickelValue + 10 * pennyValue
  let ericRemoval : ℕ := quarterValue + halfDollarValue

  totalBefore - beaumontRemoval - ericRemoval

theorem money_in_pond_is_637 : moneyInPond = 637 := by
  sorry

end money_in_pond_is_637_l1027_102743


namespace solve_equation_l1027_102709

theorem solve_equation (x : ℝ) (h : (0.12 / x) * 2 = 12) : x = 0.02 := by
  sorry

end solve_equation_l1027_102709


namespace circle_center_transformation_l1027_102766

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 + d)

theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (-3, 4)
  let reflected := reflect_x initial_center
  let final_center := translate_up reflected 5
  final_center = (-3, 1) := by sorry

end circle_center_transformation_l1027_102766


namespace runners_meeting_time_l1027_102788

/-- Represents a runner with their lap time in minutes -/
structure Runner where
  name : String
  lapTime : Nat

/-- Calculates the earliest time (in minutes) when all runners meet at the starting point -/
def earliestMeetingTime (runners : List Runner) : Nat :=
  sorry

theorem runners_meeting_time :
  let runners : List Runner := [
    { name := "Laura", lapTime := 5 },
    { name := "Maria", lapTime := 8 },
    { name := "Charlie", lapTime := 10 },
    { name := "Zoe", lapTime := 2 }
  ]
  earliestMeetingTime runners = 40 := by
  sorry

end runners_meeting_time_l1027_102788


namespace quadratic_is_perfect_square_l1027_102797

theorem quadratic_is_perfect_square :
  ∃ (a b : ℝ), ∀ x, 9 * x^2 - 30 * x + 25 = (a * x + b)^2 := by
  sorry

end quadratic_is_perfect_square_l1027_102797


namespace circle_and_tangents_l1027_102752

-- Define the points A, B, and M
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 2)
def M : ℝ × ℝ := (3, 1)

-- Define circle C with AB as its diameter
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 4}

-- Define the tangent lines
def tangentLine1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 3}
def tangentLine2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | 3 * p.1 - 4 * p.2 - 5 = 0}

theorem circle_and_tangents :
  -- 1. Prove that C is the correct circle equation
  (∀ p : ℝ × ℝ, p ∈ C ↔ (p.1 - 1)^2 + (p.2 - 2)^2 = 4) ∧
  -- 2. Prove that tangentLine1 and tangentLine2 are tangent to C and pass through M
  (∀ p : ℝ × ℝ, p ∈ tangentLine1 → (p = M ∨ (∃! q : ℝ × ℝ, q ∈ C ∩ tangentLine1))) ∧
  (∀ p : ℝ × ℝ, p ∈ tangentLine2 → (p = M ∨ (∃! q : ℝ × ℝ, q ∈ C ∩ tangentLine2))) :=
sorry

end circle_and_tangents_l1027_102752


namespace no_perfect_squares_in_sequence_l1027_102777

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 3
  | (n + 2) => 6 * x (n + 1) - x n

theorem no_perfect_squares_in_sequence : ∀ n : ℕ, ¬∃ m : ℕ, x n = m ^ 2 := by
  sorry

end no_perfect_squares_in_sequence_l1027_102777


namespace promotion_payment_correct_l1027_102710

/-- Represents the payment calculation for a clothing factory promotion -/
def promotion_payment (suit_price tie_price : ℕ) (num_suits num_ties : ℕ) : ℕ × ℕ :=
  let option1 := suit_price * num_suits + tie_price * (num_ties - num_suits)
  let option2 := ((suit_price * num_suits + tie_price * num_ties) * 9) / 10
  (option1, option2)

theorem promotion_payment_correct (x : ℕ) (h : x > 20) :
  promotion_payment 200 40 20 x = (40 * x + 3200, 3600 + 36 * x) := by
  sorry

#eval promotion_payment 200 40 20 30

end promotion_payment_correct_l1027_102710


namespace simplify_and_evaluate_l1027_102781

theorem simplify_and_evaluate (a b : ℝ) : 
  (a^2 + a - 6 = 0) → 
  (b^2 + b - 6 = 0) → 
  a ≠ b →
  ((a / (a^2 - b^2) - 1 / (a + b)) / (1 / (a^2 - a * b))) = 6 := by
  sorry

end simplify_and_evaluate_l1027_102781


namespace conic_is_ellipse_l1027_102730

/-- Defines the equation of the conic section --/
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y - 2)^2) + Real.sqrt ((x - 6)^2 + (y + 4)^2) = 12

/-- Theorem stating that the equation describes an ellipse --/
theorem conic_is_ellipse : ∃ (a b x₀ y₀ : ℝ), 
  (∀ x y : ℝ, conic_equation x y ↔ 
    ((x - x₀) / a)^2 + ((y - y₀) / b)^2 = 1) ∧ 
  a > 0 ∧ b > 0 ∧ a ≠ b :=
sorry

end conic_is_ellipse_l1027_102730


namespace cubic_equation_integer_roots_l1027_102779

theorem cubic_equation_integer_roots :
  ∀ x : ℤ, x^3 - 3*x^2 - 10*x + 20 = 0 ↔ x = -2 ∨ x = 5 := by
  sorry

end cubic_equation_integer_roots_l1027_102779


namespace arithmetic_sequence_fourth_term_l1027_102713

/-- Given an arithmetic sequence where the sum of the third and fifth terms is 10,
    prove that the fourth term is 5 -/
theorem arithmetic_sequence_fourth_term
  (b : ℝ) -- third term
  (d : ℝ) -- common difference
  (h : b + (b + 2*d) = 10) -- sum of third and fifth terms is 10
  : b + d = 5 := by sorry

end arithmetic_sequence_fourth_term_l1027_102713


namespace ab_multiplier_l1027_102769

theorem ab_multiplier (a b : ℚ) (h1 : 6 * a = 20) (h2 : 7 * b = 20) : ∃ n : ℚ, n * (a * b) = 800 ∧ n = 84 := by
  sorry

end ab_multiplier_l1027_102769


namespace cube_volumes_sum_l1027_102705

theorem cube_volumes_sum (a b c : ℕ) (h : 6 * (a^2 + b^2 + c^2) = 564) :
  a^3 + b^3 + c^3 = 764 ∨ a^3 + b^3 + c^3 = 586 :=
by sorry

end cube_volumes_sum_l1027_102705


namespace min_rental_cost_is_2860_l1027_102732

/-- Represents a rental plan for cars --/
structure RentalPlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a rental plan is valid for transporting the given amount of goods --/
def isValidPlan (plan : RentalPlan) (totalGoods : ℕ) : Prop :=
  3 * plan.typeA + 4 * plan.typeB = totalGoods

/-- Calculates the rental cost for a given plan --/
def rentalCost (plan : RentalPlan) : ℕ :=
  300 * plan.typeA + 320 * plan.typeB

/-- Theorem stating that the minimum rental cost to transport 35 tons of goods is 2860 yuan --/
theorem min_rental_cost_is_2860 :
  ∃ (plan : RentalPlan),
    isValidPlan plan 35 ∧
    rentalCost plan = 2860 ∧
    ∀ (otherPlan : RentalPlan), isValidPlan otherPlan 35 → rentalCost plan ≤ rentalCost otherPlan :=
sorry

end min_rental_cost_is_2860_l1027_102732


namespace pipe_length_problem_l1027_102737

theorem pipe_length_problem (total_length : ℝ) (short_length : ℝ) (long_length : ℝ) : 
  total_length = 177 →
  long_length = 2 * short_length →
  total_length = short_length + long_length →
  long_length = 118 := by
sorry

end pipe_length_problem_l1027_102737


namespace genevieve_coffee_consumption_l1027_102704

-- Define the conversion factors
def ml_to_oz : ℝ := 0.0338
def l_to_oz : ℝ := 33.8

-- Define the thermos sizes
def small_thermos_ml : ℝ := 250
def medium_thermos_ml : ℝ := 400
def large_thermos_l : ℝ := 1

-- Calculate the amount of coffee in each thermos type in ounces
def small_thermos_oz : ℝ := small_thermos_ml * ml_to_oz
def medium_thermos_oz : ℝ := medium_thermos_ml * ml_to_oz
def large_thermos_oz : ℝ := large_thermos_l * l_to_oz

-- Define Genevieve's consumption
def genevieve_consumption : ℝ := small_thermos_oz + medium_thermos_oz + large_thermos_oz

-- Theorem statement
theorem genevieve_coffee_consumption :
  genevieve_consumption = 55.77 := by sorry

end genevieve_coffee_consumption_l1027_102704


namespace plan_a_fixed_charge_l1027_102770

/-- The fixed charge for the first 5 minutes under plan A -/
def fixed_charge : ℝ := 0.60

/-- The per-minute rate after the first 5 minutes under plan A -/
def rate_a : ℝ := 0.06

/-- The per-minute rate for plan B -/
def rate_b : ℝ := 0.08

/-- The duration at which both plans charge the same amount -/
def equal_duration : ℝ := 14.999999999999996

theorem plan_a_fixed_charge :
  fixed_charge = rate_b * equal_duration - rate_a * (equal_duration - 5) :=
by sorry

end plan_a_fixed_charge_l1027_102770


namespace minutes_to_seconds_l1027_102759

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes we're converting to seconds -/
def minutes : ℚ := 12.5

/-- Theorem stating that 12.5 minutes is equal to 750 seconds -/
theorem minutes_to_seconds : (minutes * seconds_per_minute : ℚ) = 750 := by
  sorry

end minutes_to_seconds_l1027_102759


namespace contest_prime_problem_l1027_102782

theorem contest_prime_problem : ∃! p : ℕ,
  Prime p ∧
  100 < p ∧ p < 500 ∧
  (∃ e : ℕ,
    e > 100 ∧
    e ≡ 2016 [ZMOD (p - 1)] ∧
    e - (p - 1) / 2 = 21 ∧
    2^2016 ≡ -(2^21) [ZMOD p]) ∧
  p = 211 := by
sorry

end contest_prime_problem_l1027_102782


namespace distance_between_centers_l1027_102780

/-- Two circles in the first quadrant, both tangent to both coordinate axes and passing through (4,1) -/
structure TangentCircles where
  C₁ : ℝ × ℝ  -- Center of first circle
  C₂ : ℝ × ℝ  -- Center of second circle
  h₁ : C₁.1 = C₁.2  -- Centers lie on angle bisector
  h₂ : C₂.1 = C₂.2  -- Centers lie on angle bisector
  h₃ : C₁.1 > 0 ∧ C₁.2 > 0  -- First circle in first quadrant
  h₄ : C₂.1 > 0 ∧ C₂.2 > 0  -- Second circle in first quadrant
  h₅ : (C₁.1 - 4)^2 + (C₁.2 - 1)^2 = C₁.1^2  -- First circle passes through (4,1)
  h₆ : (C₂.1 - 4)^2 + (C₂.2 - 1)^2 = C₂.1^2  -- Second circle passes through (4,1)

/-- The distance between the centers of two tangent circles is 8 -/
theorem distance_between_centers (tc : TangentCircles) : 
  Real.sqrt ((tc.C₁.1 - tc.C₂.1)^2 + (tc.C₁.2 - tc.C₂.2)^2) = 8 := by
  sorry

end distance_between_centers_l1027_102780


namespace simplify_trig_expression_l1027_102751

theorem simplify_trig_expression :
  (Real.sin (30 * π / 180) + Real.sin (60 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (60 * π / 180)) =
  Real.tan (45 * π / 180) := by sorry

end simplify_trig_expression_l1027_102751


namespace largest_divisor_of_n_squared_divisible_by_72_l1027_102787

theorem largest_divisor_of_n_squared_divisible_by_72 (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) :
  ∀ t : ℕ, t > 0 → t ∣ n → t ≤ 12 ∧ 12 ∣ n :=
by sorry

end largest_divisor_of_n_squared_divisible_by_72_l1027_102787


namespace dave_tshirts_l1027_102717

/-- The number of white T-shirt packs Dave bought -/
def white_packs : ℕ := 3

/-- The number of T-shirts in each white pack -/
def white_per_pack : ℕ := 6

/-- The number of blue T-shirt packs Dave bought -/
def blue_packs : ℕ := 2

/-- The number of T-shirts in each blue pack -/
def blue_per_pack : ℕ := 4

/-- The total number of T-shirts Dave bought -/
def total_tshirts : ℕ := white_packs * white_per_pack + blue_packs * blue_per_pack

theorem dave_tshirts : total_tshirts = 26 := by
  sorry

end dave_tshirts_l1027_102717


namespace find_multiple_l1027_102762

theorem find_multiple (x : ℝ) (m : ℝ) (h1 : x = 13) (h2 : x + x + 2*x + m*x = 104) : m = 4 := by
  sorry

end find_multiple_l1027_102762


namespace james_fish_catch_l1027_102721

/-- The total pounds of fish James caught -/
def total_fish (trout salmon tuna : ℕ) : ℕ := trout + salmon + tuna

/-- Proves that James caught 900 pounds of fish in total -/
theorem james_fish_catch : 
  let trout : ℕ := 200
  let salmon : ℕ := trout + trout / 2
  let tuna : ℕ := 2 * trout
  total_fish trout salmon tuna = 900 := by sorry

end james_fish_catch_l1027_102721


namespace profit_maximized_at_100_yuan_optimal_selling_price_l1027_102725

/-- Profit function given price increase -/
def profit (x : ℝ) : ℝ := (10 + x) * (400 - 20 * x)

/-- The price increase that maximizes profit -/
def optimal_price_increase : ℝ := 10

theorem profit_maximized_at_100_yuan :
  ∀ x : ℝ, profit x ≤ profit optimal_price_increase :=
sorry

/-- The optimal selling price is 100 yuan -/
theorem optimal_selling_price :
  90 + optimal_price_increase = 100 :=
sorry

end profit_maximized_at_100_yuan_optimal_selling_price_l1027_102725


namespace sin_plus_cos_range_l1027_102720

theorem sin_plus_cos_range : ∀ x : ℝ, -Real.sqrt 2 ≤ Real.sin x + Real.cos x ∧ Real.sin x + Real.cos x ≤ Real.sqrt 2 := by
  sorry

end sin_plus_cos_range_l1027_102720


namespace line_parameterization_values_l1027_102792

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = 3 * x + 2

/-- The parameterization of the line -/
def parameterization (t s m : ℝ) : ℝ × ℝ :=
  (-4 + t * m, s + t * (-7))

/-- Theorem stating the values of s and m for the given line and parameterization -/
theorem line_parameterization_values :
  ∃ (s m : ℝ), 
    (∀ t, line_equation (parameterization t s m).1 (parameterization t s m).2) ∧
    s = -10 ∧ 
    m = -7/3 := by
  sorry

end line_parameterization_values_l1027_102792


namespace longest_segment_proof_l1027_102718

/-- The total length of all segments in the rectangular spiral -/
def total_length : ℕ := 3000

/-- Predicate to check if a given length satisfies the spiral condition -/
def satisfies_spiral_condition (n : ℕ) : Prop :=
  n * (n + 1) ≤ total_length

/-- The longest line segment in the rectangular spiral -/
def longest_segment : ℕ := 54

theorem longest_segment_proof :
  satisfies_spiral_condition longest_segment ∧
  ∀ m : ℕ, m > longest_segment → ¬satisfies_spiral_condition m :=
by sorry

end longest_segment_proof_l1027_102718


namespace function_characterization_l1027_102764

/-- Euler's totient function -/
noncomputable def φ : ℕ+ → ℕ+ :=
  sorry

/-- The property that the function f satisfies -/
def satisfies_property (f : ℕ+ → ℕ+) : Prop :=
  ∀ (m n : ℕ+), m ≥ n → f (m * φ (n^3)) = f m * φ (n^3)

/-- The main theorem -/
theorem function_characterization :
  ∀ (f : ℕ+ → ℕ+), satisfies_property f →
  ∃ (b : ℕ+), ∀ (n : ℕ+), f n = b * n :=
sorry

end function_characterization_l1027_102764


namespace ten_faucets_fifty_gallons_l1027_102767

/-- The time (in seconds) it takes for a given number of faucets to fill a pool of a given volume. -/
def fill_time (num_faucets : ℕ) (volume : ℝ) : ℝ :=
  sorry

theorem ten_faucets_fifty_gallons
  (h1 : fill_time 5 200 = 15 * 60) -- Five faucets fill 200 gallons in 15 minutes
  (h2 : ∀ (n : ℕ) (v : ℝ), fill_time n v > 0) -- All fill times are positive
  (h3 : ∀ (n m : ℕ) (v : ℝ), n ≠ 0 → m ≠ 0 → fill_time n v * m = fill_time m v * n) -- Faucets dispense water at the same rate
  : fill_time 10 50 = 112.5 := by
  sorry

end ten_faucets_fifty_gallons_l1027_102767


namespace candy_mixture_problem_l1027_102773

/-- Given two types of candy mixed to produce a mixture selling at a certain price,
    calculate the total amount of mixture produced. -/
theorem candy_mixture_problem (x : ℝ) : 
  x > 0 ∧ 
  3.50 * x + 4.30 * 6.25 = 4.00 * (x + 6.25) → 
  x + 6.25 = 10 := by
  sorry

#check candy_mixture_problem

end candy_mixture_problem_l1027_102773


namespace five_thirteenths_period_l1027_102778

def decimal_expansion_period (n d : ℕ) : ℕ :=
  sorry

theorem five_thirteenths_period :
  decimal_expansion_period 5 13 = 6 := by
  sorry

end five_thirteenths_period_l1027_102778


namespace fourth_root_of_sqrt_fraction_l1027_102729

theorem fourth_root_of_sqrt_fraction : 
  (32 / 10000 : ℝ)^(1/4 * 1/2) = (2 : ℝ)^(1/8) / (5 : ℝ)^(1/2) := by sorry

end fourth_root_of_sqrt_fraction_l1027_102729


namespace sarah_today_cans_l1027_102726

/-- The number of cans Sarah collected yesterday -/
def sarah_yesterday : ℕ := 50

/-- The number of additional cans Lara collected compared to Sarah yesterday -/
def lara_extra_yesterday : ℕ := 30

/-- The number of cans Lara collected today -/
def lara_today : ℕ := 70

/-- The difference in total cans collected between yesterday and today -/
def fewer_today : ℕ := 20

/-- Theorem: Sarah collected 40 cans today -/
theorem sarah_today_cans : 
  sarah_yesterday + (sarah_yesterday + lara_extra_yesterday) - fewer_today - lara_today = 40 := by
  sorry

end sarah_today_cans_l1027_102726


namespace quadratic_root_k_value_l1027_102789

theorem quadratic_root_k_value : ∃ k : ℝ, 3^2 - k*3 - 6 = 0 ∧ k = 1 := by
  sorry

end quadratic_root_k_value_l1027_102789


namespace ellipse_m_range_l1027_102716

def is_ellipse_equation (m : ℝ) : Prop :=
  (m - 1 > 0) ∧ (3 - m > 0) ∧ (m - 1 ≠ 3 - m)

theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse_equation m → m ∈ Set.Ioo 1 2 ∪ Set.Ioo 2 3 :=
by sorry

end ellipse_m_range_l1027_102716


namespace expected_value_unfair_coin_expected_value_zero_l1027_102711

/-- The expected monetary value of a single flip of an unfair coin -/
theorem expected_value_unfair_coin (p_heads : ℝ) (p_tails : ℝ) 
  (value_heads : ℝ) (value_tails : ℝ) : ℝ :=
  p_heads * value_heads + p_tails * value_tails

/-- Proof that the expected monetary value of the specific unfair coin is 0 -/
theorem expected_value_zero : 
  expected_value_unfair_coin (2/3) (1/3) 5 (-10) = 0 := by
sorry

end expected_value_unfair_coin_expected_value_zero_l1027_102711


namespace sum_of_constants_l1027_102735

/-- Given a function y(x) = a + b/x, where a and b are constants,
    prove that a + b = -34 if y(-2) = 2 and y(-4) = 8 -/
theorem sum_of_constants (a b : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (a + b / x = 2 ↔ x = -2) ∧ (a + b / x = 8 ↔ x = -4)) →
  a + b = -34 := by
  sorry

end sum_of_constants_l1027_102735


namespace absolute_value_of_negative_2023_l1027_102740

theorem absolute_value_of_negative_2023 : |(-2023 : ℝ)| = 2023 := by
  sorry

end absolute_value_of_negative_2023_l1027_102740


namespace probability_not_triangle_l1027_102741

theorem probability_not_triangle (total : ℕ) (triangles : ℕ) 
  (h1 : total = 10) (h2 : triangles = 4) : 
  (total - triangles : ℚ) / total = 3 / 5 := by
  sorry

end probability_not_triangle_l1027_102741


namespace fraction_to_decimal_l1027_102775

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end fraction_to_decimal_l1027_102775


namespace grocery_store_diet_soda_l1027_102749

/-- The number of bottles of diet soda in a grocery store -/
def diet_soda_bottles (regular_soda_bottles : ℕ) (difference : ℕ) : ℕ :=
  regular_soda_bottles - difference

/-- Theorem: The grocery store has 4 bottles of diet soda -/
theorem grocery_store_diet_soda :
  diet_soda_bottles 83 79 = 4 := by
  sorry

end grocery_store_diet_soda_l1027_102749


namespace cosine_inequality_l1027_102708

theorem cosine_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1/2) : 
  Real.cos (1 + a) < Real.cos (1 - a) := by
sorry

end cosine_inequality_l1027_102708


namespace special_number_exists_l1027_102736

theorem special_number_exists : ∃ n : ℕ+, 
  (Nat.digits 10 n.val).length = 1000 ∧ 
  0 ∉ Nat.digits 10 n.val ∧
  ∃ pairs : List (ℕ × ℕ), 
    pairs.length = 500 ∧
    (pairs.map (λ p => p.1 * p.2)).sum ∣ n.val ∧
    ∀ d ∈ Nat.digits 10 n.val, ∃ p ∈ pairs, d = p.1 ∨ d = p.2 :=
by sorry

end special_number_exists_l1027_102736


namespace floor_y_length_l1027_102793

/-- Represents a rectangular floor with length and width -/
structure RectangularFloor where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular floor -/
def area (floor : RectangularFloor) : ℝ :=
  floor.length * floor.width

theorem floor_y_length 
  (floor_x floor_y : RectangularFloor)
  (equal_area : area floor_x = area floor_y)
  (x_dimensions : floor_x.length = 10 ∧ floor_x.width = 18)
  (y_width : floor_y.width = 9) :
  floor_y.length = 20 := by
sorry

end floor_y_length_l1027_102793


namespace binary_to_septal_conversion_l1027_102791

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its septal (base 7) representation -/
def decimal_to_septal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem binary_to_septal_conversion :
  let binary := [true, false, true, false, true, true]
  let decimal := binary_to_decimal binary
  let septal := decimal_to_septal decimal
  decimal = 53 ∧ septal = [1, 0, 4] :=
by sorry

end binary_to_septal_conversion_l1027_102791


namespace arithmetic_sequence_property_l1027_102795

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 1 + 2 * a 8 + a 15 = 96) :
  2 * a 9 - a 10 = 24 := by
  sorry

end arithmetic_sequence_property_l1027_102795


namespace paradise_park_ferris_wheel_capacity_l1027_102747

/-- The number of people that can ride a Ferris wheel simultaneously -/
def ferris_wheel_capacity (num_seats : ℕ) (people_per_seat : ℕ) : ℕ :=
  num_seats * people_per_seat

/-- Theorem: A Ferris wheel with 14 seats, each holding 6 people, can accommodate 84 people -/
theorem paradise_park_ferris_wheel_capacity :
  ferris_wheel_capacity 14 6 = 84 := by
  sorry

end paradise_park_ferris_wheel_capacity_l1027_102747


namespace mean_of_cubic_solutions_l1027_102723

theorem mean_of_cubic_solutions (x : ℝ) : 
  (x^3 + 3*x^2 - 44*x = 0) → 
  (∃ s : Finset ℝ, (∀ y ∈ s, y^3 + 3*y^2 - 44*y = 0) ∧ 
                   (s.card = 3) ∧ 
                   (s.sum id / s.card = -1)) :=
by sorry

end mean_of_cubic_solutions_l1027_102723


namespace unique_digit_divisibility_l1027_102731

theorem unique_digit_divisibility : ∃! A : ℕ, A < 10 ∧ 41 % A = 0 ∧ (273100 + A * 10 + 8) % 8 = 0 := by
  sorry

end unique_digit_divisibility_l1027_102731
