import Mathlib

namespace reciprocal_of_negative_2023_l1969_196961

theorem reciprocal_of_negative_2023 : ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end reciprocal_of_negative_2023_l1969_196961


namespace rectangle_shorter_side_l1969_196937

theorem rectangle_shorter_side (a b d : ℝ) : 
  a / b = 3 / 4 →  -- ratio of sides is 3:4
  d^2 = a^2 + b^2 →  -- Pythagorean theorem
  d = 9 →  -- diagonal is 9
  a = 5.4 :=  -- shorter side is 5.4
by sorry

end rectangle_shorter_side_l1969_196937


namespace camping_trip_percentage_l1969_196932

/-- Given a school where:
    - 20% of students went to the camping trip and took more than $100
    - 75% of students who went to the camping trip did not take more than $100
    Prove that 80% of all students went on the camping trip. -/
theorem camping_trip_percentage 
  (total_students : ℕ) 
  (students_more_than_100 : ℕ) 
  (students_not_more_than_100 : ℕ) 
  (h1 : students_more_than_100 = (20 : ℕ) * total_students / 100)
  (h2 : students_not_more_than_100 = (75 : ℕ) * (students_more_than_100 + students_not_more_than_100) / 100) :
  students_more_than_100 + students_not_more_than_100 = (80 : ℕ) * total_students / 100 := by
  sorry

end camping_trip_percentage_l1969_196932


namespace unshaded_area_of_intersecting_rectangles_l1969_196903

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the intersection of two rectangles -/
structure Intersection where
  width : ℝ
  height : ℝ

theorem unshaded_area_of_intersecting_rectangles
  (r1 : Rectangle)
  (r2 : Rectangle)
  (i : Intersection)
  (h1 : r1.width = 4 ∧ r1.height = 12)
  (h2 : r2.width = 5 ∧ r2.height = 10)
  (h3 : i.width = 4 ∧ i.height = 5) :
  area r1 + area r2 - (area r1 + area r2 - i.width * i.height) = 20 :=
sorry

end unshaded_area_of_intersecting_rectangles_l1969_196903


namespace cubic_divisibility_l1969_196952

theorem cubic_divisibility : ∃ (n : ℕ), n > 0 ∧ 84^3 % n = 0 ∧ n = 592704 := by
  sorry

end cubic_divisibility_l1969_196952


namespace cube_volume_problem_l1969_196977

theorem cube_volume_problem (s : ℝ) : 
  s > 0 →
  (s - 2) * s * (s + 2) = s^3 - 12 →
  s^3 = 27 := by
sorry

end cube_volume_problem_l1969_196977


namespace expansion_coefficients_l1969_196969

theorem expansion_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ = 1 ∧ a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -128) := by
  sorry

end expansion_coefficients_l1969_196969


namespace egypt_trip_total_cost_l1969_196959

def egypt_trip_cost (base_price upgrade_cost transportation_cost : ℕ) 
                    (individual_discount transportation_discount : ℚ) 
                    (num_people : ℕ) : ℚ :=
  let discounted_tour_price := base_price - individual_discount
  let total_per_person := discounted_tour_price + upgrade_cost
  let discounted_transportation := transportation_cost * (1 - transportation_discount)
  (total_per_person + discounted_transportation) * num_people

theorem egypt_trip_total_cost :
  egypt_trip_cost 147 65 80 14 (1/10) 2 = 540 := by
  sorry

end egypt_trip_total_cost_l1969_196959


namespace gulliver_kefir_consumption_l1969_196944

/-- Represents the total number of bottles of kefir Gulliver drinks -/
def total_kefir_bottles (initial_money : ℕ) (initial_price : ℕ) : ℕ :=
  initial_money * 6 / (7 * initial_price)

/-- Theorem stating the total number of kefir bottles Gulliver drinks -/
theorem gulliver_kefir_consumption :
  total_kefir_bottles 7000000 7 = 1166666 := by
  sorry

#eval total_kefir_bottles 7000000 7

end gulliver_kefir_consumption_l1969_196944


namespace min_value_h_l1969_196967

theorem min_value_h (x : ℝ) (hx : x > 0) : x + 1/x + 1/(x + 1/x)^2 ≥ 2.25 := by
  sorry

end min_value_h_l1969_196967


namespace power_division_nineteen_l1969_196925

theorem power_division_nineteen : 19^11 / 19^5 = 47045881 := by
  sorry

end power_division_nineteen_l1969_196925


namespace min_value_expression_l1969_196910

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2 + c^2) / (a*b + 2*b*c) ≥ 2 * Real.sqrt 5 / 5 :=
sorry

end min_value_expression_l1969_196910


namespace inequality_proofs_l1969_196919

theorem inequality_proofs 
  (a b c d : ℝ) 
  (hab : a > b) 
  (hcd : c > d) 
  (hac2bc2 : a * c^2 < b * c^2) 
  (hab_pos : a > b ∧ b > 0) 
  (hc_pos : c > 0) : 
  (a + c > b + d) ∧ 
  (a < b) ∧ 
  ((b + c) / (a + c) > b / a) := by
  sorry

end inequality_proofs_l1969_196919


namespace dividend_problem_l1969_196976

theorem dividend_problem (total : ℚ) (a b c : ℚ) 
  (h1 : total = 527)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) :
  a = 62 := by
sorry

end dividend_problem_l1969_196976


namespace simplify_trig_expression_l1969_196930

theorem simplify_trig_expression : 
  Real.sqrt (1 - Real.sin (160 * π / 180) ^ 2) = Real.cos (20 * π / 180) := by
  sorry

end simplify_trig_expression_l1969_196930


namespace max_red_socks_l1969_196978

theorem max_red_socks (total : ℕ) (red : ℕ) (blue : ℕ) : 
  total ≤ 2001 →
  total = red + blue →
  (red * (red - 1) + 2 * red * blue) / (total * (total - 1)) = 1/2 →
  red ≤ 990 :=
sorry

end max_red_socks_l1969_196978


namespace expression_value_theorem_l1969_196946

theorem expression_value_theorem (x : ℝ) (h : x = Real.sqrt (19 - 8 * Real.sqrt 3)) :
  (x^4 - 6*x^3 - 2*x^2 + 18*x + 23) / (x^2 - 8*x + 15) = 5 := by
  sorry

end expression_value_theorem_l1969_196946


namespace product_of_logs_l1969_196929

theorem product_of_logs (a b : ℕ+) : 
  (b - a = 870) →
  (Real.log b / Real.log a = 2) →
  (a + b : ℕ) = 930 := by
sorry

end product_of_logs_l1969_196929


namespace three_integers_sum_l1969_196999

theorem three_integers_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 125 →
  (a : ℕ) + b + c = 31 := by
  sorry

end three_integers_sum_l1969_196999


namespace line_through_parabola_vertex_l1969_196940

/-- The number of values of a for which the line y = x + a passes through
    the vertex of the parabola y = x^2 - 2ax + a^2 -/
theorem line_through_parabola_vertex :
  ∃! a : ℝ, ∀ x y : ℝ,
    (y = x + a) ∧ (y = x^2 - 2*a*x + a^2) →
    (x = a ∧ y = 0) := by sorry

end line_through_parabola_vertex_l1969_196940


namespace sequence_product_l1969_196943

theorem sequence_product (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →
  a 4 = 2 →
  a 2 * a 3 * a 5 * a 6 = 16 :=
by
  sorry

end sequence_product_l1969_196943


namespace impossible_to_reach_all_threes_l1969_196971

/-- Represents the state of the game at any point --/
structure GameState where
  numPiles : ℕ
  totalTokens : ℕ

/-- The invariant of the game --/
def invariant (state : GameState) : ℕ :=
  state.numPiles + state.totalTokens

/-- The initial state of the game --/
def initialState : GameState :=
  { numPiles := 1, totalTokens := 1001 }

/-- Theorem stating the impossibility of reaching a state with only piles of 3 tokens --/
theorem impossible_to_reach_all_threes :
  ¬∃ (k : ℕ), invariant initialState = 4 * k :=
sorry

end impossible_to_reach_all_threes_l1969_196971


namespace sum_a_b_is_8_l1969_196935

/-- A quadrilateral PQRS with specific properties -/
structure Quadrilateral where
  a : ℤ
  b : ℤ
  a_gt_b : a > b
  b_pos : b > 0
  is_rectangle : True  -- We assume PQRS is a rectangle
  area_is_32 : 2 * (a - b).natAbs * (a + b).natAbs = 32

/-- The sum of a and b in a quadrilateral with specific properties is 8 -/
theorem sum_a_b_is_8 (q : Quadrilateral) : q.a + q.b = 8 := by
  sorry

#check sum_a_b_is_8

end sum_a_b_is_8_l1969_196935


namespace linda_jeans_sold_l1969_196900

/-- The number of jeans sold by Linda -/
def jeans_sold : ℕ := 4

/-- The price of a pair of jeans in dollars -/
def jeans_price : ℕ := 11

/-- The price of a tee in dollars -/
def tees_price : ℕ := 8

/-- The number of tees sold -/
def tees_sold : ℕ := 7

/-- The total revenue in dollars -/
def total_revenue : ℕ := 100

theorem linda_jeans_sold :
  jeans_sold * jeans_price + tees_sold * tees_price = total_revenue :=
by sorry

end linda_jeans_sold_l1969_196900


namespace raghu_investment_l1969_196922

theorem raghu_investment (raghu trishul vishal : ℝ) : 
  vishal = 1.1 * trishul →
  trishul = 0.9 * raghu →
  raghu + trishul + vishal = 6358 →
  raghu = 2200 := by
sorry

end raghu_investment_l1969_196922


namespace lillian_candy_distribution_l1969_196948

theorem lillian_candy_distribution (initial_candies : ℕ) 
  (father_multiplier : ℕ) (num_friends : ℕ) : 
  initial_candies = 205 → 
  father_multiplier = 2 → 
  num_friends = 7 → 
  (initial_candies + father_multiplier * initial_candies) / num_friends = 87 := by
  sorry

end lillian_candy_distribution_l1969_196948


namespace polygon_formation_and_perimeter_l1969_196974

-- Define the structures
structure Triangle where
  A : Point
  B : Point
  C : Point

structure Parallelogram where
  O : Point
  X : Point
  Y : Point
  Z : Point

-- Define the function that creates a parallelogram from two points and O
def createParallelogram (O X Y : Point) : Parallelogram := sorry

-- Define the function that checks if a point is inside a triangle
def isPointInTriangle (p : Point) (t : Triangle) : Prop := sorry

-- Define the function that calculates the perimeter of a triangle
def trianglePerimeter (t : Triangle) : ℝ := sorry

-- Define the function that calculates the perimeter of a polygon
def polygonPerimeter (vertices : List Point) : ℝ := sorry

-- Main theorem
theorem polygon_formation_and_perimeter 
  (ABC DEF : Triangle) (O : Point) : 
  ∃ (polygon : List Point),
    (∀ X Y, isPointInTriangle X ABC → isPointInTriangle Y DEF →
      let p := createParallelogram O X Y
      (p.O ∈ polygon ∧ p.X ∈ polygon ∧ p.Y ∈ polygon ∧ p.Z ∈ polygon)) ∧
    (polygon.length = 6) ∧
    (polygonPerimeter polygon = trianglePerimeter ABC + trianglePerimeter DEF) :=
sorry

end polygon_formation_and_perimeter_l1969_196974


namespace book_reading_problem_l1969_196916

theorem book_reading_problem (n t k : ℕ) 
  (h1 : (k + 1) * n + k * (k + 1) / 2 = 374)
  (h2 : (k + 1) * t + k * (k + 1) / 2 = 319)
  (h3 : n > 0)
  (h4 : t > 0)
  (h5 : k > 0) :
  n + t = 53 := by
sorry

end book_reading_problem_l1969_196916


namespace cube_painting_probability_l1969_196914

/-- Represents the three possible colors for painting cube faces -/
inductive Color
  | Black
  | White
  | Red

/-- Represents a cube with six faces -/
structure Cube :=
  (faces : Fin 6 → Color)

/-- Checks if a cube has no adjacent red faces -/
def noAdjacentRed (c : Cube) : Prop := sorry

/-- Counts the number of valid cube paintings -/
def validPaintings : ℕ := sorry

/-- Checks if two cubes can be rotated to look identical -/
def canRotateIdentical (c1 c2 : Cube) : Prop := sorry

/-- Counts the number of ways two cubes can be painted to look identical after rotation -/
def identicalAppearances : ℕ := sorry

/-- The main theorem stating the probability of two cubes being painted and rotatable to look identical -/
theorem cube_painting_probability :
  (identicalAppearances : ℚ) / (validPaintings^2 : ℚ) = 1 / 5776 := by sorry

end cube_painting_probability_l1969_196914


namespace decagon_game_outcome_dodecagon_game_outcome_l1969_196927

/-- Represents the possible outcomes of the game -/
inductive GameOutcome
| FirstPlayerWins
| SecondPlayerWins

/-- Represents a regular polygon with alternating colored vertices -/
structure ColoredPolygon where
  sides : ℕ
  vertices_alternating_colors : sides > 0

/-- The game played on a colored polygon -/
def polygon_segment_game (p : ColoredPolygon) : GameOutcome :=
  sorry

/-- Theorem stating the outcome for a decagon -/
theorem decagon_game_outcome :
  polygon_segment_game ⟨10, by norm_num⟩ = GameOutcome.SecondPlayerWins :=
sorry

/-- Theorem stating the outcome for a dodecagon -/
theorem dodecagon_game_outcome :
  polygon_segment_game ⟨12, by norm_num⟩ = GameOutcome.FirstPlayerWins :=
sorry

end decagon_game_outcome_dodecagon_game_outcome_l1969_196927


namespace fraction_comparison_l1969_196905

theorem fraction_comparison : 
  (10 / 8 : ℚ) = 5 / 4 ∧ 
  (5 / 4 : ℚ) = 5 / 4 ∧ 
  (15 / 12 : ℚ) = 5 / 4 ∧ 
  (6 / 5 : ℚ) ≠ 5 / 4 ∧ 
  (50 / 40 : ℚ) = 5 / 4 := by
sorry

end fraction_comparison_l1969_196905


namespace equation_solution_l1969_196938

theorem equation_solution (y z w : ℝ) :
  let f : ℝ → ℝ := λ x => (x + y) / (y + z) - (z + w) / (w + x)
  let sol₁ := (-(w + y) + Real.sqrt ((w + y)^2 + 4*(z - w)*(z - y))) / 2
  let sol₂ := (-(w + y) - Real.sqrt ((w + y)^2 + 4*(z - w)*(z - y))) / 2
  (∀ x, f x = 0 ↔ x = sol₁ ∨ x = sol₂) ∧ (f sol₁ = 0 ∧ f sol₂ = 0) := by
  sorry

end equation_solution_l1969_196938


namespace probability_of_choosing_circle_l1969_196954

theorem probability_of_choosing_circle (total : ℕ) (circles : ℕ) 
  (h1 : total = 12) (h2 : circles = 5) : 
  (circles : ℚ) / total = 5 / 12 := by
  sorry

end probability_of_choosing_circle_l1969_196954


namespace det_of_matrix_is_one_l1969_196986

-- Define the determinant formula for a 2x2 matrix
def det_2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Define our specific matrix
def matrix : Matrix (Fin 2) (Fin 2) ℝ := !![5, 7; 2, 3]

-- Theorem statement
theorem det_of_matrix_is_one :
  det_2x2 (matrix 0 0) (matrix 0 1) (matrix 1 0) (matrix 1 1) = 1 := by
  sorry

end det_of_matrix_is_one_l1969_196986


namespace T_2021_2022_2023_even_l1969_196907

def T : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | 2 => 2
  | n + 3 => T (n + 2) + T (n + 1) + T n

theorem T_2021_2022_2023_even :
  Even (T 2021) ∧ Even (T 2022) ∧ Even (T 2023) := by sorry

end T_2021_2022_2023_even_l1969_196907


namespace quadratic_inequality_l1969_196906

/-- Quadratic function f(x) = -2(x+1)^2 + k -/
def f (k : ℝ) (x : ℝ) : ℝ := -2 * (x + 1)^2 + k

/-- Theorem stating the relationship between f(x) values at x = 2, -3, and -0.5 -/
theorem quadratic_inequality (k : ℝ) : f k 2 < f k (-3) ∧ f k (-3) < f k (-0.5) := by
  sorry

end quadratic_inequality_l1969_196906


namespace smallest_perimeter_l1969_196908

/-- Triangle PQR with intersection point J of angle bisectors of ∠Q and ∠R -/
structure TrianglePQR where
  PQ : ℕ+
  QR : ℕ+
  QJ : ℕ+
  isIsosceles : PQ = PQ
  angleIntersection : QJ = 10

/-- The perimeter of triangle PQR -/
def perimeter (t : TrianglePQR) : ℕ := 2 * t.PQ + t.QR

/-- The smallest possible perimeter of triangle PQR satisfying the given conditions -/
theorem smallest_perimeter :
  ∀ t : TrianglePQR, perimeter t ≥ 40 :=
sorry

end smallest_perimeter_l1969_196908


namespace lynn_travel_time_l1969_196987

-- Define the problem parameters
def walk_fraction : ℚ := 1/3
def bike_fraction : ℚ := 2/3
def bike_speed_multiplier : ℚ := 4
def walk_time : ℚ := 9

-- Define the theorem
theorem lynn_travel_time :
  let bike_time := walk_time / bike_speed_multiplier
  walk_time + bike_time = 11.25 := by
  sorry


end lynn_travel_time_l1969_196987


namespace quadratic_polynomial_solution_l1969_196917

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  is_quadratic : a ≠ 0

/-- Evaluation of a quadratic polynomial at a point -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem quadratic_polynomial_solution 
  (f g : QuadraticPolynomial) 
  (h1 : ∀ x, f.eval (g.eval x) = (f.eval x) * (g.eval x))
  (h2 : g.eval 3 = 40) :
  g.a = 1 ∧ g.b = 31/2 ∧ g.c = -31/2 := by
  sorry

end quadratic_polynomial_solution_l1969_196917


namespace common_inner_tangent_l1969_196953

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 16
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y + 24 = 0

-- Define the proposed tangent line
def tangent_line (x y : ℝ) : Prop := 3*x - 4*y - 20 = 0

-- Theorem statement
theorem common_inner_tangent :
  ∀ x y : ℝ, 
  (circle1 x y ∨ circle2 x y) → 
  (tangent_line x y ↔ 
    (∃ t : ℝ, 
      (circle1 (x + t) (y + t) ∧ tangent_line (x + t) (y + t)) ∨
      (circle2 (x + t) (y + t) ∧ tangent_line (x + t) (y + t))))
  := by sorry

end common_inner_tangent_l1969_196953


namespace circle_equation_l1969_196901

-- Define the center of the circle
def center : ℝ × ℝ := (-1, 2)

-- Define the radius of the circle
def radius : ℝ := 2

-- State the theorem
theorem circle_equation (x y : ℝ) :
  ((x - center.1)^2 + (y - center.2)^2 = radius^2) ↔
  ((x + 1)^2 + (y - 2)^2 = 4) :=
sorry

end circle_equation_l1969_196901


namespace square_root_of_difference_l1969_196945

theorem square_root_of_difference : 
  Real.sqrt (20212020 * 20202021 - 20212021 * 20202020) = 100 := by
  sorry

end square_root_of_difference_l1969_196945


namespace sum_of_roots_equation_l1969_196989

theorem sum_of_roots_equation (x : ℝ) : 
  (∃ a b : ℝ, x^2 - 5*x + 7 = 9 ∧ x = a ∨ x = b) → a + b = 5 := by
  sorry

end sum_of_roots_equation_l1969_196989


namespace white_surface_fraction_is_5_16_l1969_196934

/-- Represents a cube with given edge length -/
structure Cube :=
  (edge : ℕ)

/-- Represents the large cube constructed from smaller cubes -/
structure LargeCube :=
  (edge : ℕ)
  (smallCubes : ℕ)
  (redCubes : ℕ)
  (whiteCubes : ℕ)

/-- Calculates the surface area of a cube -/
def surfaceArea (c : Cube) : ℕ :=
  6 * c.edge * c.edge

/-- Calculates the fraction of white surface area -/
def whiteSurfaceFraction (lc : LargeCube) : ℚ :=
  sorry

/-- Theorem stating the fraction of white surface area -/
theorem white_surface_fraction_is_5_16 (lc : LargeCube) 
  (h1 : lc.edge = 4)
  (h2 : lc.smallCubes = 64)
  (h3 : lc.redCubes = 48)
  (h4 : lc.whiteCubes = 16) :
  whiteSurfaceFraction lc = 5 / 16 :=
sorry

end white_surface_fraction_is_5_16_l1969_196934


namespace bowling_team_average_weight_l1969_196933

theorem bowling_team_average_weight 
  (initial_players : ℕ) 
  (initial_average : ℝ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) 
  (h1 : initial_players = 7) 
  (h2 : initial_average = 94) 
  (h3 : new_player1_weight = 110) 
  (h4 : new_player2_weight = 60) : 
  (initial_players * initial_average + new_player1_weight + new_player2_weight) / (initial_players + 2) = 92 := by
  sorry

end bowling_team_average_weight_l1969_196933


namespace minimum_garden_width_l1969_196998

theorem minimum_garden_width (w : ℝ) (l : ℝ) :
  w > 0 →
  l = w + 10 →
  w * l ≥ 120 →
  w ≥ 10 :=
by sorry

end minimum_garden_width_l1969_196998


namespace quadratic_properties_l1969_196931

-- Define the quadratic function
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the derivative of the quadratic function
def quadratic_derivative (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

theorem quadratic_properties (a b c : ℝ) :
  -- The function has a minimum at x = 2
  (quadratic_derivative a b 2 = 0) →
  -- The function intersects x-axis at x₁ and x₂
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ 0 < x₂ ∧ quadratic a b c x₁ = 0 ∧ quadratic a b c x₂ = 0) →
  -- tan(CAO) - tan(CBO) = 1
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ 0 < x₂ ∧ quadratic a b c x₁ = 0 ∧ quadratic a b c x₂ = 0 ∧
    c / x₁ - (-c / x₂) = 1) →
  -- Conclusions
  (b + 4 * a = 0) ∧ (a = 1/4) ∧ (b = -1) :=
by sorry

end quadratic_properties_l1969_196931


namespace inequality_proof_l1969_196951

theorem inequality_proof (a b : ℤ) 
  (h1 : a > b) 
  (h2 : b > 1) 
  (h3 : (a + b) ∣ (a * b + 1)) 
  (h4 : (a - b) ∣ (a * b - 1)) : 
  a < Real.sqrt 3 * b := by
sorry

end inequality_proof_l1969_196951


namespace dance_troupe_average_age_l1969_196911

theorem dance_troupe_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℚ) 
  (avg_age_males : ℚ) 
  (h1 : num_females = 12) 
  (h2 : num_males = 18) 
  (h3 : avg_age_females = 25) 
  (h4 : avg_age_males = 30) : 
  (num_females * avg_age_females + num_males * avg_age_males) / (num_females + num_males) = 28 := by
  sorry

end dance_troupe_average_age_l1969_196911


namespace hollow_cube_side_length_l1969_196918

/-- Represents the number of cubes used to create a hollow cube -/
def hollow_cube_cubes (n : ℕ) : ℕ := 6 * n^2 - (n^2 + 4 * (n - 2))

/-- Theorem stating that if 98 cubes are used to make a hollow cube, its side length is 9 -/
theorem hollow_cube_side_length :
  ∃ (n : ℕ), hollow_cube_cubes n = 98 ∧ n = 9 :=
by sorry

end hollow_cube_side_length_l1969_196918


namespace wednesday_dressing_time_l1969_196975

/-- Represents the dressing times for each day of the school week -/
structure DressingTimes where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the average dressing time for the week -/
def weekAverage (times : DressingTimes) : ℚ :=
  (times.monday + times.tuesday + times.wednesday + times.thursday + times.friday) / 5

/-- Theorem: Given the dressing times for Monday, Tuesday, Thursday, and Friday,
    and the old average dressing time, the dressing time for Wednesday must be 3 minutes
    to maintain the same average over the entire week. -/
theorem wednesday_dressing_time
  (times : DressingTimes)
  (h_monday : times.monday = 2)
  (h_tuesday : times.tuesday = 4)
  (h_thursday : times.thursday = 4)
  (h_friday : times.friday = 2)
  (h_old_avg : weekAverage times = 3) :
  times.wednesday = 3 := by
  sorry

#check wednesday_dressing_time

end wednesday_dressing_time_l1969_196975


namespace quadrilateral_property_l1969_196966

-- Define the quadrilateral and its properties
structure Quadrilateral :=
  (area : ℝ)
  (pq : ℝ)
  (rs : ℝ)
  (d : ℝ)
  (m : ℕ)
  (n : ℕ)
  (p : ℕ)

-- Define the theorem
theorem quadrilateral_property (q : Quadrilateral) : 
  q.area = 15 ∧ q.pq = 6 ∧ q.rs = 8 ∧ q.d^2 = q.m + q.n * Real.sqrt q.p → 
  q.m + q.n + q.p = 81 := by
  sorry

end quadrilateral_property_l1969_196966


namespace intersection_sum_l1969_196921

/-- Given two graphs y = -2|x-a| + b and y = 2|x-c| + d intersecting at (1, 6) and (5, 2), prove a + c = 6 -/
theorem intersection_sum (a b c d : ℝ) : 
  (∀ x, -2*|x - a| + b = 2*|x - c| + d → x = 1 ∧ -2*|x - a| + b = 6 ∨ x = 5 ∧ -2*|x - a| + b = 2) →
  a + c = 6 := by
  sorry

end intersection_sum_l1969_196921


namespace line_x_axis_intersection_l1969_196968

/-- The line equation 2y + 5x = 15 -/
def line_equation (x y : ℝ) : Prop := 2 * y + 5 * x = 15

/-- A point is on the x-axis if its y-coordinate is 0 -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The intersection point of the line and the x-axis -/
def intersection_point : ℝ × ℝ := (3, 0)

theorem line_x_axis_intersection :
  let (x, y) := intersection_point
  line_equation x y ∧ on_x_axis x y :=
by sorry

end line_x_axis_intersection_l1969_196968


namespace binary_sum_equals_158_l1969_196913

/-- Converts a binary number (represented as a list of bits) to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number 1010101₂ -/
def binary1 : List Bool := [true, false, true, false, true, false, true]

/-- The second binary number 1001001₂ -/
def binary2 : List Bool := [true, false, false, true, false, false, true]

/-- Theorem stating that the sum of 1010101₂ and 1001001₂ is 158 in decimal -/
theorem binary_sum_equals_158 :
  binaryToDecimal binary1 + binaryToDecimal binary2 = 158 := by
  sorry

end binary_sum_equals_158_l1969_196913


namespace teacher_budget_theorem_l1969_196982

/-- Calculates the remaining budget for a teacher after purchasing school supplies. -/
def remaining_budget (last_year_budget : ℕ) (this_year_budget : ℕ) (supply1_cost : ℕ) (supply2_cost : ℕ) : ℕ :=
  (last_year_budget + this_year_budget) - (supply1_cost + supply2_cost)

/-- Proves that the remaining budget is 19 given the specific conditions. -/
theorem teacher_budget_theorem :
  remaining_budget 6 50 13 24 = 19 := by
  sorry

end teacher_budget_theorem_l1969_196982


namespace square_triangulation_l1969_196902

/-- A planar graph representing the configuration of points and lines in a square -/
structure SquareGraph where
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces (regions)

/-- The number of triangles formed in a square with 20 internal points -/
def num_triangles (g : SquareGraph) : ℕ := g.F - 1

theorem square_triangulation :
  ∀ g : SquareGraph,
  g.V = 24 →  -- 20 internal points + 4 vertices of the square
  2 * g.E = 3 * g.F + 1 →  -- Relation between edges and faces
  g.V - g.E + g.F = 2 →  -- Euler's formula for planar graphs
  num_triangles g = 42 := by
sorry

end square_triangulation_l1969_196902


namespace circles_intersect_l1969_196985

/-- The circles x^2 + y^2 = -4y and (x-1)^2 + y^2 = 1 are intersecting -/
theorem circles_intersect : ∃ (x y : ℝ),
  (x^2 + y^2 = -4*y) ∧ ((x-1)^2 + y^2 = 1) := by
  sorry


end circles_intersect_l1969_196985


namespace cow_chicken_problem_l1969_196909

theorem cow_chicken_problem (C H : ℕ) : 4*C + 2*H = 2*(C + H) + 10 → C = 5 :=
by sorry

end cow_chicken_problem_l1969_196909


namespace birds_nest_eggs_l1969_196936

theorem birds_nest_eggs (x : ℕ) : 
  (2 * x + 3 + 4 = 17) → x = 5 := by sorry

end birds_nest_eggs_l1969_196936


namespace pages_used_per_day_l1969_196904

/-- Given 5 notebooks with 40 pages each, lasting for 50 days, prove that 4 pages are used per day. -/
theorem pages_used_per_day (num_notebooks : ℕ) (pages_per_notebook : ℕ) (days_lasted : ℕ) :
  num_notebooks = 5 →
  pages_per_notebook = 40 →
  days_lasted = 50 →
  (num_notebooks * pages_per_notebook) / days_lasted = 4 :=
by sorry

end pages_used_per_day_l1969_196904


namespace markup_markdown_l1969_196956

theorem markup_markdown (original_price : ℝ) (markup1 markup2 markup3 markdown : ℝ) : 
  markup1 = 0.1 →
  markup2 = 0.1 →
  markup3 = 0.05 →
  original_price > 0 →
  original_price * (1 + markup1) * (1 + markup2) * (1 + markup3) * (1 - markdown) = original_price →
  ∀ x : ℕ, x < 22 → (1 - (x : ℝ) / 100) > 1 - markdown :=
by sorry

end markup_markdown_l1969_196956


namespace range_of_sum_l1969_196949

theorem range_of_sum (a b : ℝ) (ha : -2 < a ∧ a < -1) (hb : -1 < b ∧ b < 0) :
  -3 < a + b ∧ a + b < -1 := by
  sorry

end range_of_sum_l1969_196949


namespace product_of_sum_and_sum_of_cubes_l1969_196997

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a^3 + b^3 = 100) : 
  a * b = -3 := by
sorry

end product_of_sum_and_sum_of_cubes_l1969_196997


namespace box_sales_ratio_l1969_196955

theorem box_sales_ratio (thursday_sales : ℕ) 
  (h1 : thursday_sales = 1200)
  (h2 : ∃ wednesday_sales : ℕ, wednesday_sales = 2 * thursday_sales)
  (h3 : ∃ tuesday_sales : ℕ, tuesday_sales = 2 * wednesday_sales) :
  ∃ (tuesday_sales wednesday_sales : ℕ),
    tuesday_sales = 2 * wednesday_sales ∧
    wednesday_sales = 2 * thursday_sales :=
by
  sorry

end box_sales_ratio_l1969_196955


namespace valid_numbers_l1969_196915

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_valid_number (abcd : ℕ) : Prop :=
  1000 ≤ abcd ∧ abcd < 10000 ∧
  abcd % 11 = 0 ∧
  (abcd / 100 % 10 + abcd / 10 % 10 = abcd / 1000) ∧
  is_perfect_square ((abcd / 100 % 10) * 10 + (abcd / 10 % 10))

theorem valid_numbers :
  {abcd : ℕ | is_valid_number abcd} = {9812, 1012, 4048, 9361, 9097} :=
by sorry

end valid_numbers_l1969_196915


namespace parabola_hyperbola_intersection_l1969_196923

theorem parabola_hyperbola_intersection (p : ℝ) (m : ℝ) (a : ℝ) (b : ℝ) : 
  p > 0 → 
  m^2 = 2*p*1 →
  (1 - p/2)^2 + m^2 = 5^2 →
  2 * (-b/a) = -1 →
  a = 1/4 := by sorry

end parabola_hyperbola_intersection_l1969_196923


namespace combined_cost_theorem_l1969_196939

def wallet_cost : ℝ := 22
def purse_cost : ℝ := 4 * wallet_cost - 3

theorem combined_cost_theorem : wallet_cost + purse_cost = 107 := by
  sorry

end combined_cost_theorem_l1969_196939


namespace rectangular_field_area_l1969_196988

/-- Given a rectangular field with one side of 30 feet and three sides fenced using 
    a total of 78 feet of fencing, prove that the area of the field is 720 square feet. -/
theorem rectangular_field_area (L W : ℝ) : 
  L = 30 →  -- Length of uncovered side
  2 * W + L = 78 →  -- Total fencing equation
  L * W = 720 :=  -- Area of the field
by
  sorry

end rectangular_field_area_l1969_196988


namespace inequality_range_l1969_196992

theorem inequality_range (t : ℝ) (h1 : t > 0) :
  (∀ x > 0, Real.exp (2 * t * x) - (Real.log 2 + Real.log x) / t ≥ 0) ↔ t ≥ 1 / Real.exp 1 :=
sorry

end inequality_range_l1969_196992


namespace one_sixths_in_eleven_thirds_l1969_196950

theorem one_sixths_in_eleven_thirds : (11 / 3) / (1 / 6) = 22 := by sorry

end one_sixths_in_eleven_thirds_l1969_196950


namespace y_squared_value_l1969_196993

theorem y_squared_value (x y : ℤ) 
  (eq1 : 4 * x + y = 34) 
  (eq2 : 2 * x - y = 20) : 
  y ^ 2 = 4 := by
sorry

end y_squared_value_l1969_196993


namespace cos_x_plus_2y_equals_one_l1969_196958

theorem cos_x_plus_2y_equals_one 
  (x y a : ℝ) 
  (h1 : x ∈ Set.Icc (-π/4) (π/4)) 
  (h2 : y ∈ Set.Icc (-π/4) (π/4)) 
  (h3 : x^3 + Real.sin x - 2*a = 0) 
  (h4 : 4*y^3 + (1/2) * Real.sin (2*y) + a = 0) : 
  Real.cos (x + 2*y) = 1 := by
sorry

end cos_x_plus_2y_equals_one_l1969_196958


namespace extreme_value_condition_l1969_196942

/-- If f(x) = m cos x + (1/2) sin 2x reaches an extreme value at x = π/4, then m = 0 -/
theorem extreme_value_condition (m : ℝ) : 
  let f := fun (x : ℝ) => m * Real.cos x + (1/2) * Real.sin (2*x)
  (∃ (ε : ℝ), ∀ (h : ℝ), 0 < |h| → |h| < ε → f (π/4 + h) ≤ f (π/4)) →
  m = 0 := by
  sorry

end extreme_value_condition_l1969_196942


namespace means_and_sum_of_squares_l1969_196941

theorem means_and_sum_of_squares
  (x y z : ℝ)
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 7)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 385.5 := by
  sorry

end means_and_sum_of_squares_l1969_196941


namespace one_third_of_seven_times_nine_l1969_196994

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end one_third_of_seven_times_nine_l1969_196994


namespace intersection_M_N_l1969_196924

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x : ℝ | x^2 ≤ x}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end intersection_M_N_l1969_196924


namespace yulia_lemonade_stand_expenses_l1969_196995

/-- Yulia's lemonade stand financial calculation -/
theorem yulia_lemonade_stand_expenses 
  (net_profit : ℝ) 
  (lemonade_revenue : ℝ) 
  (babysitting_earnings : ℝ) 
  (h1 : net_profit = 44)
  (h2 : lemonade_revenue = 47)
  (h3 : babysitting_earnings = 31) :
  lemonade_revenue + babysitting_earnings - net_profit = 34 :=
by
  sorry

#check yulia_lemonade_stand_expenses

end yulia_lemonade_stand_expenses_l1969_196995


namespace blueberry_jelly_amount_l1969_196970

/-- The amount of strawberry jelly in grams -/
def strawberry_jelly : ℕ := 1792

/-- The total amount of jelly in grams -/
def total_jelly : ℕ := 6310

/-- The amount of blueberry jelly in grams -/
def blueberry_jelly : ℕ := total_jelly - strawberry_jelly

theorem blueberry_jelly_amount : blueberry_jelly = 4518 := by
  sorry

end blueberry_jelly_amount_l1969_196970


namespace triangle_area_l1969_196973

def a : ℝ × ℝ := (3, -2)
def b : ℝ × ℝ := (-1, 5)

theorem triangle_area : 
  let det := a.1 * b.2 - a.2 * b.1
  (1/2 : ℝ) * |det| = 13/2 := by sorry

end triangle_area_l1969_196973


namespace smallest_four_digit_divisible_by_smallest_odd_primes_l1969_196981

theorem smallest_four_digit_divisible_by_smallest_odd_primes : 
  ∃ (n : ℕ), 
    (1000 ≤ n) ∧ 
    (n < 10000) ∧ 
    (n % 3 = 0) ∧ 
    (n % 5 = 0) ∧ 
    (n % 7 = 0) ∧ 
    (n % 11 = 0) ∧
    (∀ m : ℕ, 
      (1000 ≤ m) ∧ 
      (m < 10000) ∧ 
      (m % 3 = 0) ∧ 
      (m % 5 = 0) ∧ 
      (m % 7 = 0) ∧ 
      (m % 11 = 0) → 
      n ≤ m) ∧
    n = 1155 :=
by sorry

end smallest_four_digit_divisible_by_smallest_odd_primes_l1969_196981


namespace max_value_expression_l1969_196926

theorem max_value_expression (a b c d : ℝ) 
  (ha : a ∈ Set.Icc (-5 : ℝ) 5)
  (hb : b ∈ Set.Icc (-5 : ℝ) 5)
  (hc : c ∈ Set.Icc (-5 : ℝ) 5)
  (hd : d ∈ Set.Icc (-5 : ℝ) 5) :
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 110 ∧
  ∃ (a₀ b₀ c₀ d₀ : ℝ),
    a₀ ∈ Set.Icc (-5 : ℝ) 5 ∧
    b₀ ∈ Set.Icc (-5 : ℝ) 5 ∧
    c₀ ∈ Set.Icc (-5 : ℝ) 5 ∧
    d₀ ∈ Set.Icc (-5 : ℝ) 5 ∧
    a₀ + 2*b₀ + c₀ + 2*d₀ - a₀*b₀ - b₀*c₀ - c₀*d₀ - d₀*a₀ = 110 :=
by
  sorry

end max_value_expression_l1969_196926


namespace smallest_area_right_triangle_l1969_196979

/-- The smallest possible area of a right triangle with sides 6 and 8 is 24 square units -/
theorem smallest_area_right_triangle (a b c : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → (1/2) * a * b = 24 := by sorry

end smallest_area_right_triangle_l1969_196979


namespace red_sweets_count_l1969_196984

theorem red_sweets_count (total : ℕ) (green : ℕ) (neither : ℕ) (red : ℕ) 
  (h1 : total = 285)
  (h2 : green = 59)
  (h3 : neither = 177)
  (h4 : total = red + green + neither) :
  red = 49 := by
  sorry

end red_sweets_count_l1969_196984


namespace parallel_vectors_x_coordinate_l1969_196980

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is parallel to c, 
    then the x-coordinate of c is -15. -/
theorem parallel_vectors_x_coordinate 
  (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) 
  (hb : b = (2, -1)) 
  (hc : c.2 = 3) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ (a.1 + 2*b.1, a.2 + 2*b.2) = (k * c.1, k * c.2)) : 
  c.1 = -15 := by
  sorry

end parallel_vectors_x_coordinate_l1969_196980


namespace sum_greater_than_6_is_random_event_l1969_196972

def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_sum_greater_than_6 (selection : List ℕ) : Bool :=
  selection.sum > 6

theorem sum_greater_than_6_is_random_event :
  ∃ (selection₁ selection₂ : List ℕ),
    selection₁.length = 3 ∧
    selection₂.length = 3 ∧
    (∀ n ∈ selection₁, n ∈ numbers) ∧
    (∀ n ∈ selection₂, n ∈ numbers) ∧
    is_sum_greater_than_6 selection₁ ∧
    ¬is_sum_greater_than_6 selection₂ :=
by
  sorry

end sum_greater_than_6_is_random_event_l1969_196972


namespace new_ratio_after_boarders_join_l1969_196928

theorem new_ratio_after_boarders_join (initial_boarders : ℕ) (new_boarders : ℕ) :
  initial_boarders = 60 →
  new_boarders = 15 →
  (2 : ℚ) / 5 = initial_boarders / (initial_boarders * 5 / 2) →
  (1 : ℚ) / 2 = (initial_boarders + new_boarders) / (initial_boarders * 5 / 2) :=
by sorry

end new_ratio_after_boarders_join_l1969_196928


namespace modulus_of_z_l1969_196947

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end modulus_of_z_l1969_196947


namespace inequality_solution_set_l1969_196983

theorem inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | (x - 5*a)*(x + a) > 0} = {x : ℝ | x < 5*a ∨ x > -a} := by
  sorry

end inequality_solution_set_l1969_196983


namespace part_one_part_two_l1969_196920

/-- Part I: Minimum value of m for maximum |f(x)| -/
theorem part_one (a : ℝ) (h_a : a ∈ Set.Icc 4 6) :
  ∃ m : ℝ, m ≥ 6 ∧ ∀ x ∈ Set.Icc 1 m, |x + a / x - 4| ≤ |m + a / m - 4| :=
sorry

/-- Part II: Upper bound for k -/
theorem part_two (a : ℝ) (h_a : a ∈ Set.Icc 1 2) (k : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 2 4 → x₂ ∈ Set.Icc 2 4 → x₁ < x₂ →
    |x₁ + a / x₁ - 4| - |x₂ + a / x₂ - 4| < k * x₁ + 3 - (k * x₂ + 3)) →
  k ≤ 6 - 4 * Real.sqrt 3 :=
sorry

end part_one_part_two_l1969_196920


namespace stationery_shop_sales_percentage_l1969_196912

theorem stationery_shop_sales_percentage (pen_sales pencil_sales marker_sales : ℝ) 
  (h_pen : pen_sales = 25)
  (h_pencil : pencil_sales = 30)
  (h_marker : marker_sales = 20)
  (h_total : pen_sales + pencil_sales + marker_sales + (100 - pen_sales - pencil_sales - marker_sales) = 100) :
  100 - pen_sales - pencil_sales - marker_sales = 25 := by
sorry

end stationery_shop_sales_percentage_l1969_196912


namespace teeth_removal_theorem_l1969_196965

theorem teeth_removal_theorem :
  let total_teeth : ℕ := 32
  let first_person_removed : ℕ := total_teeth / 4
  let second_person_removed : ℕ := total_teeth * 3 / 8
  let third_person_removed : ℕ := total_teeth / 2
  let fourth_person_removed : ℕ := 4
  first_person_removed + second_person_removed + third_person_removed + fourth_person_removed = 40 := by
  sorry

end teeth_removal_theorem_l1969_196965


namespace polynomial_real_root_l1969_196957

theorem polynomial_real_root (a : ℝ) : ∃ x : ℝ, x^4 - a*x^2 + a*x - 1 = 0 := by
  sorry

end polynomial_real_root_l1969_196957


namespace geometric_sequence_common_ratio_l1969_196960

/-- Given a geometric sequence {a_n} with S_3 = 9/2 and a_3 = 3/2, prove that the common ratio q satisfies q = 1 or q = -1/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h1 : S 3 = 9/2) 
  (h2 : a 3 = 3/2) : 
  ∃ q : ℚ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ (q = 1 ∨ q = -1/2) :=
sorry

end geometric_sequence_common_ratio_l1969_196960


namespace edge_projection_max_sum_l1969_196963

theorem edge_projection_max_sum (a b : ℝ) : 
  (∃ x y z : ℝ, x^2 + y^2 + z^2 = 7 ∧ x^2 + y^2 = 6 ∧ 
   a^2 = x^2 + 1 ∧ b^2 = y^2 + 1) →
  a + b ≤ 4 :=
by sorry

end edge_projection_max_sum_l1969_196963


namespace a_share_of_profit_l1969_196990

/-- Calculates the share of profit for an investor in a partnership business -/
def calculate_share_of_profit (investment_A investment_B investment_C total_profit : ℚ) : ℚ :=
  (investment_A / (investment_A + investment_B + investment_C)) * total_profit

/-- Theorem: A's share of the profit is 3780 given the investments and total profit -/
theorem a_share_of_profit (investment_A investment_B investment_C total_profit : ℚ) 
  (h1 : investment_A = 6300)
  (h2 : investment_B = 4200)
  (h3 : investment_C = 10500)
  (h4 : total_profit = 12600) :
  calculate_share_of_profit investment_A investment_B investment_C total_profit = 3780 := by
  sorry

end a_share_of_profit_l1969_196990


namespace hyperbola_and_k_range_l1969_196996

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + Real.sqrt 2

-- Define the dot product condition
def dot_product_condition (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 > 2

theorem hyperbola_and_k_range :
  ∃ (a b c : ℝ),
    (∀ x y, ellipse x y ↔ x^2 / a + y^2 / b = 1) ∧
    (c^2 = a / 2) ∧
    (∀ x y, hyperbola_C x y ↔ x^2 / 3 - y^2 = 1) ∧
    (∀ k,
      (∃ x1 y1 x2 y2,
        x1 ≠ x2 ∧
        hyperbola_C x1 y1 ∧
        hyperbola_C x2 y2 ∧
        line_l k x1 y1 ∧
        line_l k x2 y2 ∧
        dot_product_condition x1 y1 x2 y2) ↔
      (k ∈ Set.Ioo (-1 : ℝ) (-Real.sqrt 3 / 3) ∪ Set.Ioo (Real.sqrt 3 / 3) 1)) :=
sorry

end hyperbola_and_k_range_l1969_196996


namespace ellipse_perpendicular_triangle_area_l1969_196964

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/49 + y^2/24 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-5, 0)
def F₂ : ℝ × ℝ := (5, 0)

-- Define perpendicularity condition
def is_perpendicular (m n : ℝ) : Prop := (n / (m + 5)) * (n / (m - 5)) = -1

-- Theorem statement
theorem ellipse_perpendicular_triangle_area (m n : ℝ) :
  is_on_ellipse m n → is_perpendicular m n →
  (1/2 : ℝ) * |10 * n| = 24 := by sorry

end ellipse_perpendicular_triangle_area_l1969_196964


namespace inverse_function_value_l1969_196962

noncomputable def f (x : ℝ) : ℝ := x / (2 * x + 1)

noncomputable def f_inv : ℝ → ℝ := Function.invFun f

theorem inverse_function_value :
  f_inv 2 = -2/3 :=
sorry

end inverse_function_value_l1969_196962


namespace min_value_of_sum_of_roots_equality_condition_l1969_196991

theorem min_value_of_sum_of_roots (x : ℝ) : 
  Real.sqrt ((x - 2)^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ 4 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x : ℝ) : 
  Real.sqrt ((x - 2)^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) = 4 * Real.sqrt 2 ↔ x = 2 :=
by sorry

end min_value_of_sum_of_roots_equality_condition_l1969_196991
