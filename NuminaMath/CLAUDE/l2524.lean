import Mathlib

namespace rectangle_diagonal_distance_sum_equal_l2524_252469

-- Define a rectangle in 2D space
structure Rectangle where
  a : ℝ  -- half-width
  b : ℝ  -- half-height

-- Define points A, B, C, D of the rectangle
def cornerA (r : Rectangle) : ℝ × ℝ := (-r.a, -r.b)
def cornerB (r : Rectangle) : ℝ × ℝ := (r.a, -r.b)
def cornerC (r : Rectangle) : ℝ × ℝ := (r.a, r.b)
def cornerD (r : Rectangle) : ℝ × ℝ := (-r.a, r.b)

-- Define the distance squared between two points
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- State the theorem
theorem rectangle_diagonal_distance_sum_equal (r : Rectangle) (p : ℝ × ℝ) :
  distanceSquared p (cornerA r) + distanceSquared p (cornerC r) =
  distanceSquared p (cornerB r) + distanceSquared p (cornerD r) := by
  sorry


end rectangle_diagonal_distance_sum_equal_l2524_252469


namespace cube_edge_assignment_impossibility_l2524_252493

/-- Represents the assignment of 1 or -1 to each edge of a cube -/
def CubeAssignment := Fin 12 → Int

/-- The set of possible sums for a face given an assignment -/
def possibleSums : Finset Int := {-4, -2, 0, 2, 4}

/-- The number of faces on a cube -/
def numFaces : Nat := 6

/-- Given a cube assignment, computes the sum for a specific face -/
def faceSum (assignment : CubeAssignment) (face : Fin 6) : Int :=
  sorry -- Implementation details omitted

theorem cube_edge_assignment_impossibility :
  ¬∃ (assignment : CubeAssignment),
    (∀ (i : Fin 12), assignment i = 1 ∨ assignment i = -1) ∧
    (∀ (face1 face2 : Fin 6), face1 ≠ face2 → faceSum assignment face1 ≠ faceSum assignment face2) :=
  sorry

#check cube_edge_assignment_impossibility

end cube_edge_assignment_impossibility_l2524_252493


namespace existence_of_prime_divisor_greater_than_ten_l2524_252451

/-- A function that returns the smallest prime divisor of a natural number -/
def smallest_prime_divisor (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem existence_of_prime_divisor_greater_than_ten (start : ℕ) 
  (h_start : is_four_digit start) :
  ∃ k : ℕ, k < 10 ∧ smallest_prime_divisor (start + k) > 10 := by
  sorry

end existence_of_prime_divisor_greater_than_ten_l2524_252451


namespace next_shared_meeting_l2524_252461

/-- The number of days between meetings for the drama club -/
def drama_interval : ℕ := 3

/-- The number of days between meetings for the choir -/
def choir_interval : ℕ := 5

/-- The number of days between meetings for the debate team -/
def debate_interval : ℕ := 7

/-- The theorem stating that the next shared meeting will occur in 105 days -/
theorem next_shared_meeting :
  Nat.lcm (Nat.lcm drama_interval choir_interval) debate_interval = 105 := by
  sorry

end next_shared_meeting_l2524_252461


namespace cookies_in_fridge_l2524_252459

theorem cookies_in_fridge (total cookies_to_tim cookies_to_mike : ℕ) 
  (h1 : total = 512)
  (h2 : cookies_to_tim = 30)
  (h3 : cookies_to_mike = 45)
  (h4 : cookies_to_anna = 3 * cookies_to_tim) :
  total - (cookies_to_tim + cookies_to_mike + cookies_to_anna) = 347 :=
by sorry

end cookies_in_fridge_l2524_252459


namespace distance_from_circle_center_to_line_l2524_252450

/-- The distance from the center of the circle x^2 + y^2 - 2x = 0 to the line 2x + y - 1 = 0 is √5/5 -/
theorem distance_from_circle_center_to_line :
  let circle_eq : ℝ → ℝ → Prop := λ x y ↦ x^2 + y^2 - 2*x = 0
  let line_eq : ℝ → ℝ → Prop := λ x y ↦ 2*x + y - 1 = 0
  ∃ (center_x center_y : ℝ), 
    (∀ x y, circle_eq x y ↔ (x - center_x)^2 + (y - center_y)^2 = 1) ∧
    (abs (2*center_x + center_y - 1) / Real.sqrt 5 = 1/5) :=
by sorry

end distance_from_circle_center_to_line_l2524_252450


namespace orange_boxes_theorem_l2524_252470

/-- Given 56 oranges that need to be stored in boxes, with each box containing 7 oranges,
    prove that the number of boxes required is 8. -/
theorem orange_boxes_theorem (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 56) (h2 : oranges_per_box = 7) :
  total_oranges / oranges_per_box = 8 := by
  sorry

end orange_boxes_theorem_l2524_252470


namespace vector_operation_result_l2524_252458

def v1 : Fin 3 → ℝ := ![-3, 2, -1]
def v2 : Fin 3 → ℝ := ![1, 10, -2]
def scalar : ℝ := 2

theorem vector_operation_result :
  scalar • v1 + v2 = ![(-5 : ℝ), 14, -4] := by sorry

end vector_operation_result_l2524_252458


namespace point_B_left_of_A_l2524_252487

theorem point_B_left_of_A : 8/13 < 5/8 := by sorry

end point_B_left_of_A_l2524_252487


namespace tan_difference_sum_l2524_252489

theorem tan_difference_sum (α β γ : Real) 
  (h1 : Real.tan α = 5)
  (h2 : Real.tan β = 2)
  (h3 : Real.tan γ = 3) :
  Real.tan (α - β + γ) = 18 := by
  sorry

end tan_difference_sum_l2524_252489


namespace isosceles_triangle_perimeter_l2524_252491

theorem isosceles_triangle_perimeter
  (equilateral_perimeter : ℝ)
  (isosceles_base : ℝ)
  (h1 : equilateral_perimeter = 60)
  (h2 : isosceles_base = 10)
  : ℝ := by
  sorry

#check isosceles_triangle_perimeter

end isosceles_triangle_perimeter_l2524_252491


namespace geometry_exam_average_score_l2524_252474

/-- Represents a student in the geometry exam -/
structure Student where
  name : String
  mistakes : ℕ
  score : ℚ

/-- Represents the geometry exam -/
structure GeometryExam where
  totalProblems : ℕ
  firstSectionProblems : ℕ
  firstSectionPoints : ℕ
  secondSectionPoints : ℕ
  firstSectionDeduction : ℕ
  secondSectionDeduction : ℕ

theorem geometry_exam_average_score 
  (exam : GeometryExam)
  (madeline leo brent nicholas : Student)
  (h_exam : exam.totalProblems = 15 ∧ 
            exam.firstSectionProblems = 5 ∧ 
            exam.firstSectionPoints = 3 ∧ 
            exam.secondSectionPoints = 1 ∧
            exam.firstSectionDeduction = 2 ∧
            exam.secondSectionDeduction = 1)
  (h_madeline : madeline.mistakes = 2)
  (h_leo : leo.mistakes = 2 * madeline.mistakes)
  (h_brent : brent.score = 25 ∧ brent.mistakes = leo.mistakes + 1)
  (h_nicholas : nicholas.mistakes = 3 * madeline.mistakes ∧ 
                nicholas.score = brent.score - 5) :
  (madeline.score + leo.score + brent.score + nicholas.score) / 4 = 22.25 := by
  sorry

end geometry_exam_average_score_l2524_252474


namespace complex_fraction_simplification_l2524_252427

theorem complex_fraction_simplification :
  let a := 6 + 7 / 2015
  let b := 4 + 5 / 2016
  let c := 7 + 2008 / 2015
  let d := 2 + 2011 / 2016
  let expression := a * b - c * d - 7 * (7 / 2015)
  expression = 5 / 144 := by sorry

end complex_fraction_simplification_l2524_252427


namespace stationery_cost_l2524_252468

/-- The cost of items at a stationery store -/
theorem stationery_cost (x y z : ℝ) 
  (h1 : 4 * x + y + 10 * z = 11) 
  (h2 : 3 * x + y + 7 * z = 8.9) : 
  x + y + z = 4.7 := by
  sorry

end stationery_cost_l2524_252468


namespace initial_shells_amount_l2524_252404

/-- The amount of shells initially in Jovana's bucket -/
def initial_shells : ℕ := sorry

/-- The amount of shells added to fill the bucket -/
def added_shells : ℕ := 12

/-- The total amount of shells after filling the bucket -/
def total_shells : ℕ := 17

/-- Theorem stating that the initial amount of shells is 5 pounds -/
theorem initial_shells_amount : initial_shells = 5 := by
  sorry

end initial_shells_amount_l2524_252404


namespace soda_price_proof_l2524_252483

/-- The regular price per can of soda -/
def regular_price : ℝ := sorry

/-- The discounted price per can when purchased in 24-can cases -/
def discounted_price : ℝ := regular_price * 0.85

/-- The total price of 72 cans purchased in 24-can cases -/
def total_price : ℝ := 18.36

theorem soda_price_proof :
  (72 * discounted_price = total_price) →
  regular_price = 0.30 := by
  sorry

end soda_price_proof_l2524_252483


namespace negation_of_universal_proposition_l2524_252479

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → 2^x > x^2) ↔ (∃ x : ℝ, x ≥ 0 ∧ 2^x ≤ x^2) :=
by sorry

end negation_of_universal_proposition_l2524_252479


namespace cone_surface_area_l2524_252435

/-- The surface area of a cone formed from a 270-degree sector of a circle with radius 20, divided by π, is 525. -/
theorem cone_surface_area (r : ℝ) (θ : ℝ) : 
  r = 20 → θ = 270 → (π * r^2 + π * r * (2 * π * r * θ / 360) / (2 * π)) / π = 525 := by
  sorry

end cone_surface_area_l2524_252435


namespace perfect_square_problem_l2524_252443

theorem perfect_square_problem :
  (∃ x : ℝ, 6^2024 = x^2) ∧
  (∀ y : ℝ, 7^2025 ≠ y^2) ∧
  (∃ z : ℝ, 8^2026 = z^2) ∧
  (∃ w : ℝ, 9^2027 = w^2) ∧
  (∃ v : ℝ, 10^2028 = v^2) := by
  sorry

end perfect_square_problem_l2524_252443


namespace cubic_function_properties_l2524_252423

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem cubic_function_properties (a b c x₀ : ℝ) 
  (h1 : f a b c (-1) = 0)
  (h2 : f a b c 1 = 0)
  (h3 : f a b c x₀ = 0)
  (h4 : 2 < x₀) (h5 : x₀ < 3) :
  (a + c = 0) ∧ (2 < c ∧ c < 3) ∧ (4*a + 2*b + c < -8) := by
  sorry


end cubic_function_properties_l2524_252423


namespace yard_length_ratio_l2524_252466

theorem yard_length_ratio : 
  ∀ (alex_length brianne_length derrick_length : ℝ),
  brianne_length = 6 * alex_length →
  brianne_length = 30 →
  derrick_length = 10 →
  alex_length / derrick_length = 1 / 2 := by
  sorry

end yard_length_ratio_l2524_252466


namespace square_area_is_eight_l2524_252492

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 = -2 * y^2 + 16 * x - 8 * y + 36

-- Define the property of being inscribed in a square with side parallel to x-axis
def inscribed_in_square (c : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (center_x center_y radius : ℝ),
    ∀ (x y : ℝ), c x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2

-- State the theorem
theorem square_area_is_eight :
  inscribed_in_square circle_equation →
  (∃ (side : ℝ), side^2 = 8 ∧
    ∀ (x y : ℝ), circle_equation x y →
      x ≥ -side/2 ∧ x ≤ side/2 ∧ y ≥ -side/2 ∧ y ≤ side/2) :=
sorry

end square_area_is_eight_l2524_252492


namespace five_students_three_locations_l2524_252421

/-- The number of ways for a given number of students to choose from a given number of locations. -/
def num_ways (num_students : ℕ) (num_locations : ℕ) : ℕ := num_locations ^ num_students

/-- Theorem: Five students choosing from three locations results in 243 different ways. -/
theorem five_students_three_locations : num_ways 5 3 = 243 := by
  sorry

end five_students_three_locations_l2524_252421


namespace triangle_sum_theorem_l2524_252441

noncomputable def triangle_sum (AB AC BC CX₁ : ℝ) : ℝ :=
  let M := BC / 2
  let NC := (5 / 13) * CX₁
  let X₁C := Real.sqrt (CX₁^2 - NC^2)
  let BN := BC - NC
  let X₁B := Real.sqrt (BN^2 + X₁C^2)
  let X₂X₁ := X₁B * (16 / 63)
  let ratio := 1 - (X₁B * (65 / 63) / AB)
  (X₁B + X₂X₁) / (1 - ratio)

theorem triangle_sum_theorem (AB AC BC CX₁ : ℝ) 
  (h1 : AB = 182) (h2 : AC = 182) (h3 : BC = 140) (h4 : CX₁ = 130) :
  triangle_sum AB AC BC CX₁ = 1106 / 5 :=
by sorry

end triangle_sum_theorem_l2524_252441


namespace half_radius_circle_y_l2524_252408

theorem half_radius_circle_y (x y : Real) : 
  (2 * Real.pi * x = 10 * Real.pi) →  -- Circumference of circle x is 10π
  (Real.pi * x^2 = Real.pi * y^2) →   -- Areas of circles x and y are equal
  (1/2) * y = 2.5 := by               -- Half of the radius of circle y is 2.5
sorry

end half_radius_circle_y_l2524_252408


namespace simplify_complex_fraction_l2524_252453

theorem simplify_complex_fraction (b : ℝ) (h : b ≠ 2) :
  2 - (1 / (1 + b / (2 - b))) = 1 + b / 2 := by
  sorry

end simplify_complex_fraction_l2524_252453


namespace largest_angle_measure_l2524_252456

def ConvexPentagon (a b c d e : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a + b + c + d + e = 540

theorem largest_angle_measure (a b c d e : ℝ) :
  ConvexPentagon a b c d e →
  c - 3 = a →
  e = 110 := by
  sorry

end largest_angle_measure_l2524_252456


namespace polynomial_product_equals_difference_of_cubes_l2524_252449

theorem polynomial_product_equals_difference_of_cubes (x : ℝ) :
  (x^4 + 30*x^2 + 225) * (x^2 - 15) = x^6 - 3375 := by
  sorry

end polynomial_product_equals_difference_of_cubes_l2524_252449


namespace alices_lawn_area_l2524_252457

/-- Represents a rectangular lawn with fence posts -/
structure Lawn :=
  (total_posts : ℕ)
  (post_spacing : ℕ)
  (long_side_posts : ℕ)
  (short_side_posts : ℕ)

/-- Calculates the area of the lawn given its specifications -/
def lawn_area (l : Lawn) : ℕ :=
  (l.post_spacing * (l.short_side_posts - 1)) * (l.post_spacing * (l.long_side_posts - 1))

/-- Theorem stating the area of Alice's lawn -/
theorem alices_lawn_area :
  ∀ (l : Lawn),
  l.total_posts = 24 →
  l.post_spacing = 5 →
  l.long_side_posts = 3 * l.short_side_posts →
  2 * (l.long_side_posts + l.short_side_posts - 2) = l.total_posts →
  lawn_area l = 825 := by
  sorry

#check alices_lawn_area

end alices_lawn_area_l2524_252457


namespace veranda_area_l2524_252472

/-- Calculates the area of a veranda surrounding a rectangular room -/
theorem veranda_area (room_length room_width veranda_width : ℝ) :
  room_length = 20 ∧ room_width = 12 ∧ veranda_width = 2 →
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = 144 :=
by sorry

end veranda_area_l2524_252472


namespace mark_sugar_intake_excess_l2524_252430

/-- Represents the calorie content and sugar information for a soft drink -/
structure SoftDrink where
  totalCalories : ℕ
  sugarPercentage : ℚ

/-- Represents the sugar content of a candy bar -/
structure CandyBar where
  sugarCalories : ℕ

theorem mark_sugar_intake_excess (drink : SoftDrink) (bar : CandyBar) 
    (h1 : drink.totalCalories = 2500)
    (h2 : drink.sugarPercentage = 5 / 100)
    (h3 : bar.sugarCalories = 25)
    (h4 : (drink.totalCalories : ℚ) * drink.sugarPercentage + 7 * bar.sugarCalories = 300)
    (h5 : (300 : ℚ) / 150 - 1 = 1) : 
    (300 : ℚ) / 150 - 1 = 1 := by sorry

end mark_sugar_intake_excess_l2524_252430


namespace windows_preference_l2524_252440

theorem windows_preference (total : ℕ) (mac_pref : ℕ) (no_pref : ℕ) : 
  total = 210 →
  mac_pref = 60 →
  no_pref = 90 →
  ∃ (windows_pref : ℕ),
    windows_pref = total - mac_pref - (mac_pref / 3) - no_pref ∧
    windows_pref = 40 := by
  sorry

end windows_preference_l2524_252440


namespace bounded_harmonic_constant_l2524_252434

/-- A function f: ℤ² → ℝ is harmonic if it satisfies the discrete Laplace equation -/
def Harmonic (f : ℤ × ℤ → ℝ) : Prop :=
  ∀ x y, f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1) = 4 * f (x, y)

/-- A function f: ℤ² → ℝ is bounded if there exists a positive constant M such that |f(x, y)| ≤ M for all (x, y) in ℤ² -/
def Bounded (f : ℤ × ℤ → ℝ) : Prop :=
  ∃ M > 0, ∀ x y, |f (x, y)| ≤ M

/-- If a function f: ℤ² → ℝ is both harmonic and bounded, then it is constant -/
theorem bounded_harmonic_constant (f : ℤ × ℤ → ℝ) (hf_harmonic : Harmonic f) (hf_bounded : Bounded f) :
  ∃ c : ℝ, ∀ x y, f (x, y) = c :=
sorry

end bounded_harmonic_constant_l2524_252434


namespace probability_two_qualified_products_l2524_252447

theorem probability_two_qualified_products (total : ℕ) (qualified : ℕ) (unqualified : ℕ) 
  (h1 : total = qualified + unqualified)
  (h2 : total = 10)
  (h3 : qualified = 8)
  (h4 : unqualified = 2) :
  let p := (qualified - 1) / (total - 1)
  p = 7 / 11 := by
sorry

end probability_two_qualified_products_l2524_252447


namespace complex_modulus_problem_l2524_252484

theorem complex_modulus_problem (z : ℂ) (h : (1 + 2*I)*z = (1 - I)) : 
  Complex.abs z = Real.sqrt 10 / 5 := by
sorry

end complex_modulus_problem_l2524_252484


namespace three_tangent_lines_l2524_252420

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define a line passing through (0,2)
def line_through_point (m b : ℝ) (x y : ℝ) : Prop := y = m*x + b ∧ 2 = b

-- Define the condition for a line to intersect the parabola at exactly one point
def intersects_once (m b : ℝ) : Prop :=
  ∃! x y, parabola x y ∧ line_through_point m b x y

-- The main theorem
theorem three_tangent_lines :
  ∃ L1 L2 L3 : ℝ × ℝ,
    L1 ≠ L2 ∧ L1 ≠ L3 ∧ L2 ≠ L3 ∧
    (∀ m b, intersects_once m b ↔ (m, b) = L1 ∨ (m, b) = L2 ∨ (m, b) = L3) :=
sorry

end three_tangent_lines_l2524_252420


namespace pirate_treasure_probability_l2524_252464

/-- The probability of finding treasure without traps on an island -/
def p_treasure_only : ℚ := 1 / 5

/-- The probability of finding neither treasure nor traps on an island -/
def p_neither : ℚ := 3 / 5

/-- The number of islands -/
def n_islands : ℕ := 7

/-- The number of islands with treasure only -/
def n_treasure_only : ℕ := 3

/-- The number of islands with neither treasure nor traps -/
def n_neither : ℕ := 4

theorem pirate_treasure_probability : 
  (Nat.choose n_islands n_treasure_only : ℚ) * 
  (p_treasure_only ^ n_treasure_only) * 
  (p_neither ^ n_neither) = 81 / 2225 := by
  sorry

end pirate_treasure_probability_l2524_252464


namespace arithmetic_sequence_common_difference_l2524_252463

def arithmetic_sequence (a : ℕ → ℝ) := 
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum1 : a 1 + a 2 = 4) 
  (h_sum2 : a 3 + a 4 = 16) : 
  ∃ d : ℝ, d = 3 ∧ ∀ n, a (n + 1) = a n + d := by
sorry

end arithmetic_sequence_common_difference_l2524_252463


namespace plywood_area_conservation_l2524_252444

theorem plywood_area_conservation (A W : ℝ) (h : A > 0 ∧ W > 0) :
  let L : ℝ := A / W
  let L' : ℝ := A / (2 * W)
  A = W * L ∧ A = (2 * W) * L' := by sorry

end plywood_area_conservation_l2524_252444


namespace unique_satisfying_function_l2524_252400

/-- A function f : [1, +∞) → [1, +∞) satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x ≥ 1, f x ≥ 1) ∧
  (∀ x ≥ 1, f x ≤ 2 * (x + 1)) ∧
  (∀ x ≥ 1, f (x + 1) = (1 / x) * ((f x)^2 - 1))

/-- The theorem stating that x + 1 is the unique function satisfying the conditions -/
theorem unique_satisfying_function :
  ∃! f : ℝ → ℝ, SatisfyingFunction f ∧ ∀ x ≥ 1, f x = x + 1 :=
sorry

end unique_satisfying_function_l2524_252400


namespace cos_sin_225_degrees_l2524_252405

theorem cos_sin_225_degrees :
  Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 ∧
  Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_sin_225_degrees_l2524_252405


namespace min_value_sum_squared_ratios_l2524_252402

theorem min_value_sum_squared_ratios (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b)^2 + (b / c)^2 + (c / a)^2 ≥ 3 ∧
  ((a / b)^2 + (b / c)^2 + (c / a)^2 = 3 ↔ a = b ∧ b = c) :=
by sorry

end min_value_sum_squared_ratios_l2524_252402


namespace alternating_sequence_sum_l2524_252411

def arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def alternating_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  arithmetic_sequence_sum a₁ (2 * d) ((n + 1) / 2) -
  arithmetic_sequence_sum (a₁ + d) (2 * d) (n / 2)

theorem alternating_sequence_sum :
  alternating_sum 2 3 19 = 29 := by
  sorry

end alternating_sequence_sum_l2524_252411


namespace square_plus_reciprocal_square_l2524_252467

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 := by
  sorry

end square_plus_reciprocal_square_l2524_252467


namespace D_72_equals_81_l2524_252439

/-- D(n) represents the number of ways to write a positive integer n as a product of integers greater than 1, where the order matters. -/
def D (n : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that D(72) = 81 -/
theorem D_72_equals_81 : D 72 = 81 := by
  sorry

end D_72_equals_81_l2524_252439


namespace tetrahedron_properties_l2524_252454

/-- A right prism with vertices A, B, C, A₁, B₁, C₁ -/
structure RightPrism (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C A₁ B₁ C₁ : V)
  (is_right_prism : sorry)

/-- Points P and P₁ on edges BB₁ and CC₁ respectively -/
structure PrismWithPoints (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] extends RightPrism V :=
  (P P₁ : V)
  (P_on_BB₁ : sorry)
  (P₁_on_CC₁ : sorry)
  (ratio_condition : sorry)

/-- The dihedral angle between two planes -/
def dihedral_angle (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (plane1 plane2 : Set V) : ℝ := sorry

/-- Theorem stating the properties of the tetrahedron AA₁PP₁ -/
theorem tetrahedron_properties 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (prism : PrismWithPoints V) :
  let A := prism.A
  let A₁ := prism.A₁
  let P := prism.P
  let P₁ := prism.P₁
  (dihedral_angle V {A, P₁, P} {A, A₁, P} = π / 2) ∧ 
  (dihedral_angle V {A₁, P, P₁} {A₁, A, P} = π / 2) ∧
  (dihedral_angle V {A, P, P₁} {A, A₁, P} + 
   dihedral_angle V {A, P, P₁} {A₁, P, P₁} + 
   dihedral_angle V {A₁, P₁, P} {A, A₁, P₁} = π) := by
  sorry

end tetrahedron_properties_l2524_252454


namespace probability_two_red_balls_l2524_252475

/-- The probability of picking two red balls from a bag containing 3 red, 4 blue, and 4 green balls. -/
theorem probability_two_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ)
  (h_total : total_balls = red_balls + blue_balls + green_balls)
  (h_red : red_balls = 3)
  (h_blue : blue_balls = 4)
  (h_green : green_balls = 4) :
  (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) = 3 / 55 := by
  sorry

end probability_two_red_balls_l2524_252475


namespace simplify_fraction_product_l2524_252446

theorem simplify_fraction_product : 4 * (15 / 5) * (25 / -75) = -4 := by
  sorry

end simplify_fraction_product_l2524_252446


namespace inverse_proportion_inequality_l2524_252436

/-- Given two points on the inverse proportion function y = -4/x, 
    if the x-coordinate of the first point is negative and 
    the x-coordinate of the second point is positive, 
    then the y-coordinate of the first point is greater than 
    the y-coordinate of the second point. -/
theorem inverse_proportion_inequality (x₁ x₂ y₁ y₂ : ℝ) : 
  x₁ < 0 → 0 < x₂ → y₁ = -4 / x₁ → y₂ = -4 / x₂ → y₁ > y₂ := by
  sorry

end inverse_proportion_inequality_l2524_252436


namespace coin_fraction_missing_l2524_252433

theorem coin_fraction_missing (x : ℚ) : x > 0 →
  let lost := (1 / 3 : ℚ) * x
  let found := (3 / 4 : ℚ) * lost
  let remaining := x - lost + found
  x - remaining = (1 / 12 : ℚ) * x := by
  sorry

end coin_fraction_missing_l2524_252433


namespace solution_system_l2524_252431

theorem solution_system (x y : ℝ) 
  (eq1 : x + y = 10) 
  (eq2 : x / y = 7 / 3) : 
  x = 7 ∧ y = 3 := by
sorry

end solution_system_l2524_252431


namespace first_discount_percentage_l2524_252488

/-- Proves that the first discount percentage is 10% given the conditions of the problem -/
theorem first_discount_percentage 
  (list_price : ℝ) 
  (final_price : ℝ) 
  (second_discount : ℝ) :
  list_price = 70 →
  final_price = 59.85 →
  second_discount = 0.05 →
  ∃ (first_discount : ℝ),
    final_price = list_price * (1 - first_discount) * (1 - second_discount) ∧
    first_discount = 0.1 :=
by sorry

end first_discount_percentage_l2524_252488


namespace michael_watermelon_weight_l2524_252426

/-- The weight of Michael's watermelon in pounds -/
def michael_watermelon : ℝ := 8

/-- The weight of Clay's watermelon in pounds -/
def clay_watermelon : ℝ := 3 * michael_watermelon

/-- The weight of John's watermelon in pounds -/
def john_watermelon : ℝ := 12

theorem michael_watermelon_weight :
  michael_watermelon = 8 ∧
  clay_watermelon = 3 * michael_watermelon ∧
  john_watermelon = clay_watermelon / 2 ∧
  john_watermelon = 12 := by
  sorry

end michael_watermelon_weight_l2524_252426


namespace intersection_of_A_and_B_l2524_252495

-- Define the sets A and B
def A : Set ℝ := {x | x < -3}
def B : Set ℝ := {x | x > -4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | -4 < x ∧ x < -3} := by sorry

end intersection_of_A_and_B_l2524_252495


namespace parabola_focal_line_properties_l2524_252417

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  pos_p : p > 0

/-- Point on a parabola -/
structure ParabolaPoint (para : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2 * para.p * x

/-- Line through focus intersecting parabola -/
structure FocalLine (para : Parabola) where
  A : ParabolaPoint para
  B : ParabolaPoint para

/-- Theorem statement -/
theorem parabola_focal_line_properties (para : Parabola) (l : FocalLine para) :
  ∃ (N : ℝ × ℝ) (P : ℝ × ℝ),
    -- 1. FN = 1/2 * AB
    (N.1 - para.p/2)^2 + N.2^2 = (1/2)^2 * ((l.A.x - l.B.x)^2 + (l.A.y - l.B.y)^2) ∧
    -- 2. Trajectory of P
    P.1 + para.p/2 = 0 := by
  sorry

end parabola_focal_line_properties_l2524_252417


namespace cylinder_radius_proof_l2524_252445

/-- The radius of a cylinder with specific properties -/
def cylinder_radius : ℝ := 12

/-- The original height of the cylinder -/
def original_height : ℝ := 4

/-- The increase in radius or height -/
def increase : ℝ := 8

theorem cylinder_radius_proof :
  (cylinder_radius + increase)^2 * original_height = 
  cylinder_radius^2 * (original_height + increase) :=
by sorry

end cylinder_radius_proof_l2524_252445


namespace intersection_P_Q_l2524_252407

def P : Set ℝ := {0, 1, 2, 3}
def Q : Set ℝ := {x : ℝ | |x| < 2}

theorem intersection_P_Q : P ∩ Q = {0, 1} := by sorry

end intersection_P_Q_l2524_252407


namespace inequality_proof_l2524_252422

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x^2 * y) / (y + z) + (y^2 * z) / (z + x) + (z^2 * x) / (x + y) ≥ (1/2) * (x^2 + y^2 + z^2) := by
  sorry

end inequality_proof_l2524_252422


namespace circle_center_transformation_l2524_252481

def initial_center : ℝ × ℝ := (8, -3)
def reflection_line (x y : ℝ) : Prop := y = x
def translation_vector : ℝ × ℝ := (2, -5)

def reflect_point (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def translate_point (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + v.1, p.2 + v.2)

theorem circle_center_transformation :
  let reflected := reflect_point initial_center
  let final := translate_point reflected translation_vector
  final = (-1, 3) := by sorry

end circle_center_transformation_l2524_252481


namespace inequality_proof_l2524_252498

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (1 + y * z) / (1 + x^2) + (1 + z * x) / (1 + y^2) + (1 + x * y) / (1 + z^2) ≥ 2 ∧
  ((1 + y * z) / (1 + x^2) + (1 + z * x) / (1 + y^2) + (1 + x * y) / (1 + z^2) = 2 ↔ x = 0 ∧ y = 0 ∧ z = 0) :=
by sorry

end inequality_proof_l2524_252498


namespace intersection_P_Q_l2524_252476

def P : Set ℝ := {-3, 0, 2, 4}
def Q : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} := by sorry

end intersection_P_Q_l2524_252476


namespace correct_average_weight_l2524_252415

theorem correct_average_weight (n : ℕ) (initial_avg : ℚ) (misread_weight : ℚ) (correct_weight : ℚ) :
  n = 20 →
  initial_avg = 58.4 →
  misread_weight = 56 →
  correct_weight = 66 →
  (n : ℚ) * initial_avg + (correct_weight - misread_weight) = n * 58.9 := by
  sorry

end correct_average_weight_l2524_252415


namespace smallest_positive_z_l2524_252448

open Real

theorem smallest_positive_z (x z : ℝ) (h1 : cos x = 0) (h2 : cos (x + z) = 1/2) :
  ∃ (z_min : ℝ), z_min = π/6 ∧ z_min > 0 ∧ ∀ (z' : ℝ), z' > 0 → cos x = 0 → cos (x + z') = 1/2 → z' ≥ z_min :=
sorry

end smallest_positive_z_l2524_252448


namespace sqrt_sum_eq_sum_iff_two_zero_l2524_252442

theorem sqrt_sum_eq_sum_iff_two_zero (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ (a = 0 ∧ b = 0) ∨ (a = 0 ∧ c = 0) ∨ (b = 0 ∧ c = 0) :=
by sorry

end sqrt_sum_eq_sum_iff_two_zero_l2524_252442


namespace balloon_problem_l2524_252409

-- Define the number of balloons each person has
def allan_initial : ℕ := sorry
def jake_initial : ℕ := 6
def allan_bought : ℕ := 3

-- Define the relationship between Allan's and Jake's balloons
theorem balloon_problem :
  allan_initial = 2 :=
by
  have h1 : jake_initial = (allan_initial + allan_bought) + 1 :=
    sorry
  sorry

end balloon_problem_l2524_252409


namespace imaginary_part_of_complex_fraction_l2524_252428

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := (1 - i) / (1 + i)
  Complex.im z = -1 := by sorry

end imaginary_part_of_complex_fraction_l2524_252428


namespace constant_term_product_l2524_252452

theorem constant_term_product (x : ℝ) : 
  (x^4 + x^2 + 7) * (2*x^5 + 3*x^3 + 10) = 70 + x * (2*x^8 + 3*x^6 + 20*x^4 + 3*x^7 + 10*x^5 + 7*2*x^5 + 7*3*x^3) :=
by sorry

end constant_term_product_l2524_252452


namespace polynomial_composition_factorization_l2524_252471

theorem polynomial_composition_factorization :
  ∀ (p : Polynomial ℤ),
  (Polynomial.degree p ≥ 1) →
  ∃ (q f g : Polynomial ℤ),
    (Polynomial.degree f ≥ 1) ∧
    (Polynomial.degree g ≥ 1) ∧
    (p.comp q = f * g) := by
  sorry

end polynomial_composition_factorization_l2524_252471


namespace tangent_parallel_implies_a_equals_one_l2524_252486

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem tangent_parallel_implies_a_equals_one (a : ℝ) :
  (f' a 1 = 4) → a = 1 := by
  sorry

end tangent_parallel_implies_a_equals_one_l2524_252486


namespace unique_digit_multiple_6_and_9_l2524_252494

def is_multiple_of_6_and_9 (n : ℕ) : Prop :=
  n % 6 = 0 ∧ n % 9 = 0

def five_digit_number (d : ℕ) : ℕ :=
  74820 + d

theorem unique_digit_multiple_6_and_9 :
  ∃! d : ℕ, d < 10 ∧ is_multiple_of_6_and_9 (five_digit_number d) :=
by sorry

end unique_digit_multiple_6_and_9_l2524_252494


namespace spherical_coordinate_reflection_l2524_252482

/-- Given a point with rectangular coordinates (3, 8, -6) and spherical coordinates (ρ, θ, φ),
    prove that the point with spherical coordinates (ρ, θ, -φ) has rectangular coordinates (-3, -8, -6). -/
theorem spherical_coordinate_reflection (ρ θ φ : ℝ) : 
  (ρ * Real.sin φ * Real.cos θ = 3 ∧ 
   ρ * Real.sin φ * Real.sin θ = 8 ∧ 
   ρ * Real.cos φ = -6) → 
  (ρ * Real.sin (-φ) * Real.cos θ = -3 ∧ 
   ρ * Real.sin (-φ) * Real.sin θ = -8 ∧ 
   ρ * Real.cos (-φ) = -6) := by
  sorry

end spherical_coordinate_reflection_l2524_252482


namespace shelter_adoption_percentage_l2524_252410

def initial_dogs : ℕ := 80
def returned_dogs : ℕ := 5
def final_dogs : ℕ := 53

def adoption_percentage : ℚ := 40

theorem shelter_adoption_percentage :
  (initial_dogs - (initial_dogs * adoption_percentage / 100) + returned_dogs : ℚ) = final_dogs :=
sorry

end shelter_adoption_percentage_l2524_252410


namespace cone_height_from_circular_sector_l2524_252424

/-- The height of a cone formed from a sector of a circular sheet -/
theorem cone_height_from_circular_sector (r : ℝ) (n : ℕ) (h : n > 0) : 
  let base_radius := r * Real.pi / (2 * n)
  let slant_height := r
  let height := Real.sqrt (slant_height^2 - base_radius^2)
  (r = 10 ∧ n = 4) → height = Real.sqrt 93.75 := by
  sorry

end cone_height_from_circular_sector_l2524_252424


namespace rectangles_in_5x5_grid_l2524_252416

/-- The number of dots in each row and column of the square array -/
def gridSize : Nat := 5

/-- The number of different rectangles that can be formed in the grid -/
def numRectangles : Nat := (gridSize.choose 2) * (gridSize.choose 2)

/-- Theorem stating the number of rectangles in a 5x5 grid -/
theorem rectangles_in_5x5_grid : numRectangles = 100 := by sorry

end rectangles_in_5x5_grid_l2524_252416


namespace smallest_next_divisor_after_437_l2524_252413

theorem smallest_next_divisor_after_437 (m : ℕ) (h1 : 10000 ≤ m ∧ m ≤ 99999) 
  (h2 : Odd m) (h3 : 437 ∣ m) :
  ∃ (d : ℕ), d ∣ m ∧ 437 < d ∧ d ≤ 874 ∧ ∀ (x : ℕ), x ∣ m → 437 < x → x ≥ d :=
by sorry

end smallest_next_divisor_after_437_l2524_252413


namespace apple_difference_l2524_252478

theorem apple_difference (jackie_apples adam_apples : ℕ) 
  (h1 : jackie_apples = 10) (h2 : adam_apples = 8) : 
  jackie_apples - adam_apples = 2 := by
  sorry

end apple_difference_l2524_252478


namespace initial_tangerines_count_l2524_252477

/-- The number of tangerines initially in the basket -/
def initial_tangerines : ℕ := sorry

/-- The number of tangerines Eunji ate -/
def eaten_tangerines : ℕ := 9

/-- The number of tangerines mother added -/
def added_tangerines : ℕ := 5

/-- The final number of tangerines in the basket -/
def final_tangerines : ℕ := 20

/-- Theorem stating that the initial number of tangerines was 24 -/
theorem initial_tangerines_count : initial_tangerines = 24 :=
by
  have h : initial_tangerines - eaten_tangerines + added_tangerines = final_tangerines := sorry
  sorry


end initial_tangerines_count_l2524_252477


namespace xiaodong_election_l2524_252403

theorem xiaodong_election (V : ℝ) (h : V > 0) : 
  let votes_needed := (3/4 : ℝ) * V
  let votes_calculated := (2/3 : ℝ) * V
  let votes_obtained := (5/6 : ℝ) * votes_calculated
  let votes_remaining := V - votes_calculated
  let additional_votes_needed := votes_needed - votes_obtained
  (additional_votes_needed / votes_remaining) = (7/12 : ℝ) := by
sorry

end xiaodong_election_l2524_252403


namespace water_flow_restrictor_l2524_252460

/-- Calculates the reduced flow rate given the original flow rate. -/
def reducedFlowRate (originalRate : ℝ) : ℝ :=
  0.6 * originalRate - 1

theorem water_flow_restrictor (originalRate : ℝ) 
    (h : originalRate = 5.0) : 
    reducedFlowRate originalRate = 2.0 := by
  sorry

end water_flow_restrictor_l2524_252460


namespace fd_length_l2524_252419

-- Define the triangle and arc
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r = 20 ∧ 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = r^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = r^2 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = r^2

-- Define the semicircle
def Semicircle (A B D : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ), O = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧
  (D.1 - O.1)^2 + (D.2 - O.2)^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4

-- Define the tangent point
def Tangent (C D O : ℝ × ℝ) : Prop :=
  (C.1 - D.1) * (D.1 - O.1) + (C.2 - D.2) * (D.2 - O.2) = 0

-- Define the intersection point
def Intersect (C D F B : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧
  F = (C.1 + t*(D.1 - C.1), C.2 + t*(D.2 - C.2)) ∧
  (F.1 - B.1)^2 + (F.2 - B.2)^2 = 20^2

-- Main theorem
theorem fd_length (A B C D F : ℝ × ℝ) :
  Triangle A B C →
  Semicircle A B D →
  Tangent C D ((A.1 + B.1)/2, (A.2 + B.2)/2) →
  Intersect C D F B →
  (F.1 - D.1)^2 + (F.2 - D.2)^2 = 16 := by
  sorry

end fd_length_l2524_252419


namespace third_degree_polynomial_property_l2524_252429

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial := ℝ → ℝ

/-- The property that |f(x)| = 15 for x ∈ {2, 4, 5, 6, 8, 9} -/
def HasAbsoluteValue15 (f : ThirdDegreePolynomial) : Prop :=
  ∀ x ∈ ({2, 4, 5, 6, 8, 9} : Set ℝ), |f x| = 15

theorem third_degree_polynomial_property (f : ThirdDegreePolynomial) 
  (h : HasAbsoluteValue15 f) : |f 0| = 135 := by
  sorry

end third_degree_polynomial_property_l2524_252429


namespace probability_one_defective_l2524_252480

def total_products : ℕ := 10
def quality_products : ℕ := 7
def defective_products : ℕ := 3
def selected_products : ℕ := 4

theorem probability_one_defective :
  (Nat.choose quality_products (selected_products - 1) * Nat.choose defective_products 1) /
  Nat.choose total_products selected_products = 1 / 2 :=
by sorry

end probability_one_defective_l2524_252480


namespace harmonic_mean_inequality_l2524_252401

theorem harmonic_mean_inequality (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 2) :
  1/m + 1/n ≥ 2 := by
sorry

end harmonic_mean_inequality_l2524_252401


namespace count_denominators_repeating_decimal_l2524_252438

/-- The number of different possible denominators for the fraction representation of a repeating decimal 0.ab̅ in lowest terms, where a and b are digits. -/
theorem count_denominators_repeating_decimal : ∃ (n : ℕ), n = 6 ∧ n = (Finset.image (λ (p : ℕ × ℕ) => (Nat.lcm 99 (10 * p.1 + p.2) / (10 * p.1 + p.2)).gcd 99) (Finset.filter (λ (p : ℕ × ℕ) => p.1 < 10 ∧ p.2 < 10) (Finset.product (Finset.range 10) (Finset.range 10)))).card := by
  sorry

end count_denominators_repeating_decimal_l2524_252438


namespace quadratic_two_distinct_roots_l2524_252432

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 8 = 0 ∧ x₂^2 + m*x₂ - 8 = 0 :=
by
  sorry


end quadratic_two_distinct_roots_l2524_252432


namespace min_distance_squared_l2524_252437

/-- The minimum squared distance from a point M(x,y,z) to N(1,1,1), 
    given specific conditions on x, y, and z -/
theorem min_distance_squared (x y z : ℝ) : 
  (∃ r : ℝ, y = x * r ∧ z = y * r) →  -- geometric progression condition
  (y * z = (x * y + x * z) / 2) →    -- arithmetic progression condition
  (z ≥ 1) →                          -- z ≥ 1 condition
  (x ≠ y ∧ y ≠ z ∧ x ≠ z) →          -- distinctness condition
  18 ≤ (x - 1)^2 + (y - 1)^2 + (z - 1)^2 :=
by sorry

end min_distance_squared_l2524_252437


namespace min_value_of_exponential_sum_l2524_252455

theorem min_value_of_exponential_sum (a b : ℝ) (h : a + b = 2) :
  ∃ m : ℝ, m = 6 ∧ ∀ x y : ℝ, x + y = 2 → 3^x + 3^y ≥ m :=
sorry

end min_value_of_exponential_sum_l2524_252455


namespace min_pieces_to_find_both_l2524_252496

-- Define the grid
def Grid := Fin 8 → Fin 8 → Bool

-- Define the properties of the grid
def has_fish (g : Grid) (i j : Fin 8) : Prop := g i j = true
def has_sausage (g : Grid) (i j : Fin 8) : Prop := g i j = true
def has_both (g : Grid) (i j : Fin 8) : Prop := has_fish g i j ∧ has_sausage g i j

-- Define the conditions
def valid_grid (g : Grid) : Prop :=
  (∃ i j k l m n : Fin 8, has_fish g i j ∧ has_fish g k l ∧ has_fish g m n ∧ 
    ¬(i = k ∧ j = l) ∧ ¬(i = m ∧ j = n) ∧ ¬(k = m ∧ l = n)) ∧
  (∃ i j k l : Fin 8, has_sausage g i j ∧ has_sausage g k l ∧ ¬(i = k ∧ j = l)) ∧
  (∃! i j : Fin 8, has_both g i j) ∧
  (∀ i j : Fin 6, ∃ k l m n : Fin 8, k ≥ i ∧ k < i + 6 ∧ l ≥ j ∧ l < j + 6 ∧
    m ≥ i ∧ m < i + 6 ∧ n ≥ j ∧ n < j + 6 ∧ has_fish g k l ∧ has_fish g m n ∧ ¬(k = m ∧ l = n)) ∧
  (∀ i j : Fin 6, ∃! k l : Fin 8, k ≥ i ∧ k < i + 3 ∧ l ≥ j ∧ l < j + 3 ∧ has_sausage g k l)

-- Define the theorem
theorem min_pieces_to_find_both (g : Grid) (h : valid_grid g) :
  ∃ s : Finset (Fin 8 × Fin 8), s.card = 5 ∧
    (∀ t : Finset (Fin 8 × Fin 8), t.card < 5 → 
      ∃ i j : Fin 8, has_both g i j ∧ (i, j) ∉ t) ∧
    (∀ i j : Fin 8, has_both g i j → (i, j) ∈ s) :=
sorry

end min_pieces_to_find_both_l2524_252496


namespace counterexample_exists_l2524_252418

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relation for a point being on a line or in a plane
variable (on_line : Point → Line → Prop)
variable (in_plane : Point → Plane → Prop)

-- Define the relation for a line being a subset of a plane
variable (line_subset_plane : Line → Plane → Prop)

-- Theorem statement
theorem counterexample_exists (l : Line) (α : Plane) (A : Point) 
  (h1 : ¬ line_subset_plane l α) 
  (h2 : on_line A l) :
  ¬ (∀ A, on_line A l → ¬ in_plane A α) :=
by sorry

end counterexample_exists_l2524_252418


namespace infinite_solutions_of_diophantine_equation_l2524_252473

theorem infinite_solutions_of_diophantine_equation :
  ∃ (S : Set (ℕ × ℕ × ℕ × ℕ)), Set.Infinite S ∧
    ∀ (x y z t : ℕ), (x, y, z, t) ∈ S → x^2 + y^2 = 5*(z^2 + t^2) := by
  sorry

end infinite_solutions_of_diophantine_equation_l2524_252473


namespace average_tickets_sold_l2524_252462

/-- Proves that the average number of tickets sold per member is 66 given the conditions -/
theorem average_tickets_sold (male_count : ℕ) (female_count : ℕ) 
  (male_female_ratio : female_count = 2 * male_count)
  (female_avg : ℝ) (male_avg : ℝ)
  (h_female_avg : female_avg = 70)
  (h_male_avg : male_avg = 58) :
  let total_tickets := female_count * female_avg + male_count * male_avg
  let total_members := male_count + female_count
  total_tickets / total_members = 66 := by
sorry

end average_tickets_sold_l2524_252462


namespace geometric_relations_l2524_252497

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (line_perp_plane : Line → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (plane_perp_plane : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem geometric_relations 
  (m l : Line) (α β : Plane) : 
  (line_perp_plane l α ∧ line_parallel_plane m α → perpendicular l m) ∧
  ¬(parallel m l ∧ line_in_plane m α → line_parallel_plane l α) ∧
  ¬(plane_perp_plane α β ∧ line_in_plane m α ∧ line_in_plane l β → perpendicular m l) ∧
  ¬(perpendicular m l ∧ line_in_plane m α ∧ line_in_plane l β → plane_perp_plane α β) :=
by sorry

end geometric_relations_l2524_252497


namespace problem_1_problem_2_l2524_252414

-- Problem 1
theorem problem_1 : -9 / 3 + (1 / 2 - 2 / 3) * 12 - |(-4)^3| = -69 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : 2 * (a^2 + 2*b^2) - 3 * (2*a^2 - b^2) = -4*a^2 + 7*b^2 := by sorry

end problem_1_problem_2_l2524_252414


namespace friendship_distribution_impossibility_l2524_252485

theorem friendship_distribution_impossibility :
  ∀ (students : Finset Nat) (f : Nat → Nat),
    Finset.card students = 25 →
    (∃ s₁ s₂ s₃ : Finset Nat, 
      Finset.card s₁ = 6 ∧ 
      Finset.card s₂ = 10 ∧ 
      Finset.card s₃ = 9 ∧
      s₁ ∪ s₂ ∪ s₃ = students ∧
      Disjoint s₁ s₂ ∧ Disjoint s₁ s₃ ∧ Disjoint s₂ s₃ ∧
      (∀ i ∈ s₁, f i = 3) ∧
      (∀ i ∈ s₂, f i = 4) ∧
      (∀ i ∈ s₃, f i = 5)) →
    False := by
  sorry


end friendship_distribution_impossibility_l2524_252485


namespace parallel_planes_sufficient_condition_l2524_252490

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relation for planes and lines
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- Define the relation for a line being within a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the intersection relation for lines
variable (line_intersect : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_sufficient_condition
  (α β : Plane) (m n l₁ l₂ : Line)
  (h_diff_planes : α ≠ β)
  (h_diff_lines : m ≠ n)
  (h_m_in_α : line_in_plane m α)
  (h_n_in_α : line_in_plane n α)
  (h_l₁_in_β : line_in_plane l₁ β)
  (h_l₂_in_β : line_in_plane l₂ β)
  (h_l₁_l₂_intersect : line_intersect l₁ l₂)
  (h_m_parallel_l₁ : line_parallel m l₁)
  (h_n_parallel_l₂ : line_parallel n l₂) :
  plane_parallel α β :=
sorry

end parallel_planes_sufficient_condition_l2524_252490


namespace expression_evaluation_l2524_252465

theorem expression_evaluation : (47 + 21)^2 - (47^2 + 21^2) - 7 * 47 = 1645 := by
  sorry

end expression_evaluation_l2524_252465


namespace two_digit_sum_ten_l2524_252412

/-- A two-digit number is a natural number between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- The digit sum of a natural number is the sum of its digits. -/
def DigitSum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

/-- There are exactly 9 two-digit numbers whose digits sum to 10. -/
theorem two_digit_sum_ten :
  ∃! (s : Finset ℕ), (∀ n ∈ s, TwoDigitNumber n ∧ DigitSum n = 10) ∧ s.card = 9 := by
sorry

end two_digit_sum_ten_l2524_252412


namespace complex_expression_equality_l2524_252406

theorem complex_expression_equality : 
  Real.sqrt (4/9) - Real.sqrt ((-2)^4) + (19/27 - 1)^(1/3) - (-1)^2017 = -5/3 := by
  sorry

end complex_expression_equality_l2524_252406


namespace quadratic_inequality_range_l2524_252425

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) := by
  sorry

end quadratic_inequality_range_l2524_252425


namespace circular_film_diameter_l2524_252499

theorem circular_film_diameter 
  (volume : ℝ) 
  (thickness : ℝ) 
  (π : ℝ) 
  (h1 : volume = 576) 
  (h2 : thickness = 0.2) 
  (h3 : π = Real.pi) : 
  let radius := Real.sqrt (volume / (thickness * π))
  2 * radius = 2 * Real.sqrt (2880 / π) :=
sorry

end circular_film_diameter_l2524_252499
