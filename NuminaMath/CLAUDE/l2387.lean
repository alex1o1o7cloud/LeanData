import Mathlib

namespace max_height_triangle_DEF_l2387_238731

/-- Triangle DEF with side lengths -/
structure Triangle where
  DE : ℝ
  EF : ℝ
  FD : ℝ

/-- The maximum possible height of a table constructed from a triangle -/
def max_table_height (t : Triangle) : ℝ := sorry

/-- The given triangle DEF -/
def triangle_DEF : Triangle :=
  { DE := 25,
    EF := 28,
    FD := 33 }

theorem max_height_triangle_DEF :
  max_table_height triangle_DEF = 60 * Real.sqrt 129 / 61 := by sorry

end max_height_triangle_DEF_l2387_238731


namespace triangle_constant_sum_squares_l2387_238774

/-- Given a triangle XYZ where YZ = 10 and the length of median XM is 7,
    the value of XZ^2 + XY^2 is constant. -/
theorem triangle_constant_sum_squares (X Y Z M : ℝ × ℝ) :
  let d (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  (d Y Z = 10) →
  (M = ((Y.1 + Z.1) / 2, (Y.2 + Z.2) / 2)) →
  (d X M = 7) →
  ∃ (c : ℝ), ∀ (X' : ℝ × ℝ), d X' M = 7 → (d X' Y)^2 + (d X' Z)^2 = c :=
by sorry

end triangle_constant_sum_squares_l2387_238774


namespace prob_C_is_five_thirtysix_l2387_238707

/-- A spinner with 5 regions A, B, C, D, and E -/
structure Spinner :=
  (probA : ℚ)
  (probB : ℚ)
  (probC : ℚ)
  (probD : ℚ)
  (probE : ℚ)

/-- The properties of the spinner as given in the problem -/
def spinner_properties (s : Spinner) : Prop :=
  s.probA = 5/12 ∧
  s.probB = 1/6 ∧
  s.probC = s.probD ∧
  s.probE = s.probD ∧
  s.probA + s.probB + s.probC + s.probD + s.probE = 1

/-- The theorem stating that the probability of region C is 5/36 -/
theorem prob_C_is_five_thirtysix (s : Spinner) 
  (h : spinner_properties s) : s.probC = 5/36 := by
  sorry

end prob_C_is_five_thirtysix_l2387_238707


namespace intersection_points_on_circle_l2387_238776

/-- The intersection points of the parabolas y = (x + 2)^2 and x + 2 = (y - 1)^2 lie on a circle with r^2 = 2 -/
theorem intersection_points_on_circle : ∃ (c : ℝ × ℝ) (r : ℝ),
  (∀ x y : ℝ, y = (x + 2)^2 ∧ x + 2 = (y - 1)^2 →
    (x - c.1)^2 + (y - c.2)^2 = r^2) ∧
  r^2 = 2 := by
  sorry

end intersection_points_on_circle_l2387_238776


namespace ten_tables_seating_l2387_238790

/-- Calculates the number of people that can be seated at a given number of tables arranged in a row -/
def seatsInRow (numTables : ℕ) : ℕ :=
  if numTables = 0 then 0
  else if numTables = 1 then 6
  else if numTables = 2 then 10
  else if numTables = 3 then 14
  else 4 * numTables + 2

/-- Calculates the number of people that can be seated in a rectangular arrangement of tables -/
def seatsInRectangle (rows : ℕ) (tablesPerRow : ℕ) : ℕ :=
  rows * seatsInRow tablesPerRow

theorem ten_tables_seating :
  seatsInRectangle 2 5 = 80 :=
by sorry

end ten_tables_seating_l2387_238790


namespace f_sum_property_l2387_238714

def f (x : ℝ) : ℝ := 5*x^6 - 3*x^5 + 4*x^4 + x^3 - 2*x^2 - 2*x + 8

theorem f_sum_property : f 5 = 20 → f 5 + f (-5) = 68343 := by
  sorry

end f_sum_property_l2387_238714


namespace card_game_proof_l2387_238740

theorem card_game_proof (total_credits : ℕ) (red_cards : ℕ) (red_credit_value : ℕ) (blue_credit_value : ℕ)
  (h1 : total_credits = 84)
  (h2 : red_cards = 8)
  (h3 : red_credit_value = 3)
  (h4 : blue_credit_value = 5) :
  ∃ (blue_cards : ℕ), red_cards + blue_cards = 20 ∧ 
    red_cards * red_credit_value + blue_cards * blue_credit_value = total_credits :=
by
  sorry

end card_game_proof_l2387_238740


namespace reconstruct_axes_and_unit_l2387_238765

-- Define the parabola
def parabola : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

-- Define the concept of constructible points
def constructible (p : ℝ × ℝ) : Prop := sorry

-- Define the concept of constructible lines
def constructibleLine (l : Set (ℝ × ℝ)) : Prop := sorry

-- Define the x-axis
def xAxis : Set (ℝ × ℝ) := {p | p.2 = 0}

-- Define the y-axis
def yAxis : Set (ℝ × ℝ) := {p | p.1 = 0}

-- Define the unit point (1, 1)
def unitPoint : ℝ × ℝ := (1, 1)

-- Theorem stating that the coordinate axes and unit length can be reconstructed
theorem reconstruct_axes_and_unit : 
  ∃ (x y : Set (ℝ × ℝ)) (u : ℝ × ℝ),
    constructibleLine x ∧ 
    constructibleLine y ∧ 
    constructible u ∧
    x = xAxis ∧ 
    y = yAxis ∧ 
    u = unitPoint :=
  sorry

end reconstruct_axes_and_unit_l2387_238765


namespace replaced_person_weight_l2387_238772

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℚ) (new_person_weight : ℚ) : ℚ :=
  new_person_weight - (initial_count : ℚ) * average_increase

/-- Theorem stating the weight of the replaced person under the given conditions -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 (5/2) 85 = 65 := by
  sorry

end replaced_person_weight_l2387_238772


namespace min_cost_trees_l2387_238775

/-- The cost function for purchasing trees -/
def cost_function (x : ℕ) : ℕ := 20 * x + 12000

/-- The constraint on the number of cypress trees -/
def cypress_constraint (x : ℕ) : Prop := x ≥ 3 * (150 - x)

/-- The total number of trees to be purchased -/
def total_trees : ℕ := 150

/-- The theorem stating the minimum cost and optimal purchase -/
theorem min_cost_trees :
  ∃ (x : ℕ), 
    x ≤ total_trees ∧
    cypress_constraint x ∧
    (∀ (y : ℕ), y ≤ total_trees → cypress_constraint y → cost_function x ≤ cost_function y) ∧
    x = 113 ∧
    cost_function x = 14260 := by
  sorry

end min_cost_trees_l2387_238775


namespace line_relationship_sum_l2387_238711

/-- Represents a line in the form Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.A * l2.B = l1.B * l2.A

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.A * l2.A + l1.B * l2.B = 0

theorem line_relationship_sum (m n : ℝ) : 
  let l1 : Line := ⟨2, 2, -1⟩
  let l2 : Line := ⟨4, n, 3⟩
  let l3 : Line := ⟨m, 6, 1⟩
  parallel l1 l2 → perpendicular l1 l3 → m + n = -2 := by
  sorry

end line_relationship_sum_l2387_238711


namespace bertha_age_difference_l2387_238712

structure Grandparents where
  arthur : ℕ
  bertha : ℕ
  christoph : ℕ
  dolores : ℕ

def is_valid_grandparents (g : Grandparents) : Prop :=
  (max g.arthur (max g.bertha (max g.christoph g.dolores))) - 
  (min g.arthur (min g.bertha (min g.christoph g.dolores))) = 4 ∧
  g.arthur = g.bertha + 2 ∧
  g.christoph = g.dolores + 2 ∧
  g.bertha < g.dolores

theorem bertha_age_difference (g : Grandparents) (h : is_valid_grandparents g) :
  g.bertha + 2 = (g.arthur + g.bertha + g.christoph + g.dolores) / 4 := by
  sorry

#check bertha_age_difference

end bertha_age_difference_l2387_238712


namespace min_value_sum_of_reciprocals_l2387_238721

theorem min_value_sum_of_reciprocals (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (hsum : p + q + r + s + t + u = 10) :
  2/p + 3/q + 5/r + 7/s + 11/t + 13/u ≥ 23.875 ∧ 
  ∃ (p' q' r' s' t' u' : ℝ), 
    p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ s' > 0 ∧ t' > 0 ∧ u' > 0 ∧
    p' + q' + r' + s' + t' + u' = 10 ∧
    2/p' + 3/q' + 5/r' + 7/s' + 11/t' + 13/u' = 23.875 :=
by sorry

end min_value_sum_of_reciprocals_l2387_238721


namespace expression_value_l2387_238784

theorem expression_value (x y : ℝ) (h : x - y = 1) :
  x^4 - x*y^3 - x^3*y - 3*x^2*y + 3*x*y^2 + y^4 = 1 := by
  sorry

end expression_value_l2387_238784


namespace negation_of_implication_l2387_238702

theorem negation_of_implication (x : ℝ) : 
  (¬(x^2 = 1 → x = 1)) ↔ (x^2 ≠ 1 → x ≠ 1) := by sorry

end negation_of_implication_l2387_238702


namespace floor_equation_solution_l2387_238701

theorem floor_equation_solution :
  {x : ℚ | ⌊(8*x + 19)/7⌋ = (16*(x + 1))/11} =
  {1 + 1/16, 1 + 3/4, 2 + 7/16, 3 + 1/8, 3 + 13/16} := by
  sorry

end floor_equation_solution_l2387_238701


namespace trapezoid_segment_length_l2387_238797

-- Define the trapezoid and its properties
structure Trapezoid :=
  (AB CD : ℝ)
  (area_ratio : ℝ)
  (sum_parallel_sides : ℝ)
  (h_positive : AB > 0)
  (h_area_ratio : area_ratio = 5 / 3)
  (h_sum : AB + CD = sum_parallel_sides)

-- Theorem statement
theorem trapezoid_segment_length (t : Trapezoid) (h : t.sum_parallel_sides = 160) :
  t.AB = 100 :=
by sorry

end trapezoid_segment_length_l2387_238797


namespace line_equation_midpoint_line_equation_vector_ratio_l2387_238703

-- Define the point P
def P : ℝ × ℝ := (-3, 1)

-- Define the line l passing through P and intersecting x-axis at A and y-axis at B
def line_l (A B : ℝ × ℝ) : Prop :=
  A.2 = 0 ∧ B.1 = 0 ∧ ∃ t : ℝ, P = t • A + (1 - t) • B

-- Define the midpoint condition
def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the vector ratio condition
def vector_ratio (P A B : ℝ × ℝ) : Prop :=
  (A.1 - P.1, A.2 - P.2) = (2 * (P.1 - B.1), 2 * (P.2 - B.2))

-- Theorem for case I
theorem line_equation_midpoint (A B : ℝ × ℝ) :
  line_l A B → is_midpoint P A B →
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, (x - 3*y + 6 = 0) ↔ k * (x - A.1) = k * (y - A.2) :=
sorry

-- Theorem for case II
theorem line_equation_vector_ratio (A B : ℝ × ℝ) :
  line_l A B → vector_ratio P A B →
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, (x - 6*y + 9 = 0) ↔ k * (x - A.1) = k * (y - A.2) :=
sorry

end line_equation_midpoint_line_equation_vector_ratio_l2387_238703


namespace max_min_A_values_l2387_238743

open Complex Real

theorem max_min_A_values (z : ℂ) (h : abs (z - I) ≤ 1) :
  let A := (z.re : ℝ) * ((abs (z - I))^2 - 1)
  ∃ (max_A min_A : ℝ), 
    (∀ z', abs (z' - I) ≤ 1 → (z'.re : ℝ) * ((abs (z' - I))^2 - 1) ≤ max_A) ∧
    (∀ z', abs (z' - I) ≤ 1 → (z'.re : ℝ) * ((abs (z' - I))^2 - 1) ≥ min_A) ∧
    max_A = 2 * Real.sqrt 3 / 9 ∧
    min_A = -2 * Real.sqrt 3 / 9 :=
sorry

end max_min_A_values_l2387_238743


namespace irrational_among_given_numbers_l2387_238764

theorem irrational_among_given_numbers : 
  (∃ q : ℚ, |3 / (-8)| = q) ∧ 
  (∃ q : ℚ, |22 / 7| = q) ∧ 
  (∃ q : ℚ, 3.14 = q) ∧ 
  (∀ q : ℚ, |Real.sqrt 3| ≠ q) := by
  sorry

end irrational_among_given_numbers_l2387_238764


namespace largest_angle_in_circle_l2387_238759

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the angle between three points
def angle (A B C : Point) : ℝ := sorry

-- Define a function to check if a point is inside a circle
def isInside (p : Point) (c : Circle) : Prop := sorry

-- Define a function to check if a point is on the circumference of a circle
def isOnCircumference (p : Point) (c : Circle) : Prop := sorry

-- Define a function to check if three points form a diameter of a circle
def formsDiameter (A B C : Point) (circle : Circle) : Prop := sorry

theorem largest_angle_in_circle (circle : Circle) (A B : Point) 
  (hA : isInside A circle) (hB : isInside B circle) :
  ∃ C, isOnCircumference C circle ∧ 
    (∀ D, isOnCircumference D circle → angle A B C ≥ angle A B D) ∧
    formsDiameter A B C circle := by
  sorry

end largest_angle_in_circle_l2387_238759


namespace expression_evaluation_l2387_238738

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := -2
  ((2 * x + y) * (2 * x - y) - (2 * x - 3 * y)^2) / (-2 * y) = -16 := by
  sorry

end expression_evaluation_l2387_238738


namespace triangle_formation_count_l2387_238755

/-- The number of ways to choose k elements from a set of n elements -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of checkpoints on the first track -/
def checkpoints_track1 : ℕ := 6

/-- The number of checkpoints on the second track -/
def checkpoints_track2 : ℕ := 10

/-- The number of ways to form triangles by selecting one point from the first track
    and two points from the second track -/
def triangle_formations : ℕ := checkpoints_track1 * choose checkpoints_track2 2

theorem triangle_formation_count :
  triangle_formations = 270 := by sorry

end triangle_formation_count_l2387_238755


namespace cone_sphere_ratio_l2387_238792

/-- The ratio of a cone's height to its radius when its volume is one-third of a sphere with the same radius -/
theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) : 
  (1 / 3) * ((4 / 3) * Real.pi * r^3) = (1 / 3) * Real.pi * r^2 * h → h / r = 4 / 3 := by
  sorry

end cone_sphere_ratio_l2387_238792


namespace cube_decomposition_l2387_238725

/-- The smallest odd number in the decomposition of m³ -/
def smallest_odd (m : ℕ+) : ℕ := 2 * (m - 1) + 3

/-- The number of odd terms in the decomposition of m³ -/
def num_terms (m : ℕ+) : ℕ := (m + 2) * (m - 1) / 2

theorem cube_decomposition (m : ℕ+) :
  smallest_odd m = 91 → m = 10 := by sorry

end cube_decomposition_l2387_238725


namespace third_square_side_length_l2387_238770

/-- Given three squares with perimeters 60 cm, 48 cm, and 36 cm respectively,
    if the area of the third square is equal to the difference of the areas of the first two squares,
    then the side length of the third square is 9 cm. -/
theorem third_square_side_length 
  (s1 s2 s3 : ℝ) 
  (h1 : 4 * s1 = 60) 
  (h2 : 4 * s2 = 48) 
  (h3 : 4 * s3 = 36) 
  (h4 : s3^2 = s1^2 - s2^2) : 
  s3 = 9 := by
sorry

end third_square_side_length_l2387_238770


namespace quadratic_form_completion_l2387_238756

theorem quadratic_form_completion (z : ℝ) : ∃ (b : ℝ) (c : ℤ), z^2 - 6*z + 20 = (z + b)^2 + c ∧ c = 11 := by
  sorry

end quadratic_form_completion_l2387_238756


namespace factorial_ratio_l2387_238748

theorem factorial_ratio : Nat.factorial 10 / (Nat.factorial 4 * Nat.factorial 6) = 210 := by
  sorry

end factorial_ratio_l2387_238748


namespace car_average_speed_l2387_238723

/-- Given a car's speed for two hours, calculate its average speed. -/
theorem car_average_speed (speed1 speed2 : ℝ) (h1 : speed1 = 85) (h2 : speed2 = 45) :
  (speed1 + speed2) / 2 = 65 := by
  sorry

#check car_average_speed

end car_average_speed_l2387_238723


namespace min_sum_m_n_l2387_238778

theorem min_sum_m_n (m n : ℕ+) (h : 108 * m = n ^ 3) : 
  ∀ (k l : ℕ+), 108 * k = l ^ 3 → m + n ≤ k + l :=
sorry

end min_sum_m_n_l2387_238778


namespace bert_equals_kameron_in_40_days_l2387_238760

/-- The number of days required for Bert to have the same number of kangaroos as Kameron -/
def days_to_equal_kangaroos (kameron_kangaroos : ℕ) (bert_kangaroos : ℕ) (bert_buying_rate : ℕ) : ℕ :=
  (kameron_kangaroos - bert_kangaroos) / bert_buying_rate

/-- Proof that it takes 40 days for Bert to have the same number of kangaroos as Kameron -/
theorem bert_equals_kameron_in_40_days :
  days_to_equal_kangaroos 100 20 2 = 40 := by
  sorry

end bert_equals_kameron_in_40_days_l2387_238760


namespace union_of_A_and_B_l2387_238719

-- Define the sets A and B
def A : Set ℝ := { x | -1 < x ∧ x ≤ 1 }
def B : Set ℝ := { x | 0 < x ∧ x < 2 }

-- Theorem statement
theorem union_of_A_and_B :
  A ∪ B = { x | -1 < x ∧ x < 2 } := by sorry

end union_of_A_and_B_l2387_238719


namespace vector_problem_l2387_238724

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 6)
def c (x : ℝ) : ℝ × ℝ := 2 • a + b x

theorem vector_problem (x : ℝ) :
  (∃ y, b y ≠ r • a ∧ r ≠ 0) →  -- non-collinearity condition
  ‖a - b x‖ = 2 * Real.sqrt 5 →
  c x = (1, 10) := by
  sorry

end vector_problem_l2387_238724


namespace length_to_breadth_ratio_l2387_238710

/-- Represents a rectangular plot -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  area : ℝ
  length_multiple_of_breadth : ∃ (k : ℝ), length = k * breadth
  area_eq : area = length * breadth

/-- Theorem: The ratio of length to breadth is 3:1 for a rectangular plot with area 2028 and breadth 26 -/
theorem length_to_breadth_ratio (plot : RectangularPlot) 
  (h_area : plot.area = 2028)
  (h_breadth : plot.breadth = 26) :
  plot.length / plot.breadth = 3 := by
sorry

end length_to_breadth_ratio_l2387_238710


namespace perpendicular_when_a_neg_one_passes_through_zero_one_l2387_238754

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop :=
  (a^2 + a + 1) * x - y + 1 = 0

-- Define perpendicularity of two lines given their slopes
def perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

-- Theorem for statement A
theorem perpendicular_when_a_neg_one :
  perpendicular (-((-1)^2 + (-1) + 1)) 1 :=
sorry

-- Theorem for statement C
theorem passes_through_zero_one (a : ℝ) :
  line_l a 0 1 :=
sorry

end perpendicular_when_a_neg_one_passes_through_zero_one_l2387_238754


namespace mortdecai_mall_delivery_l2387_238771

/-- Represents the egg collection and distribution for Mortdecai in a week -/
structure EggDistribution where
  collected_per_day : ℕ  -- dozens of eggs collected on Tuesday and Thursday
  market_delivery : ℕ    -- dozens of eggs delivered to the market
  pie_usage : ℕ          -- dozens of eggs used for pie
  charity_donation : ℕ   -- dozens of eggs donated to charity

/-- Calculates the number of dozens of eggs delivered to the mall -/
def mall_delivery (ed : EggDistribution) : ℕ :=
  2 * ed.collected_per_day - (ed.market_delivery + ed.pie_usage + ed.charity_donation)

/-- Theorem stating that Mortdecai delivers 5 dozen eggs to the mall -/
theorem mortdecai_mall_delivery :
  let ed : EggDistribution := {
    collected_per_day := 8,
    market_delivery := 3,
    pie_usage := 4,
    charity_donation := 4
  }
  mall_delivery ed = 5 := by sorry

end mortdecai_mall_delivery_l2387_238771


namespace distance_covered_72min_10kmph_l2387_238787

/-- The distance covered by a man walking at a given speed for a given time. -/
def distanceCovered (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A man walking for 72 minutes at a speed of 10 km/hr covers a distance of 12 km. -/
theorem distance_covered_72min_10kmph :
  let speed : ℝ := 10  -- Speed in km/hr
  let time : ℝ := 72 / 60  -- Time in hours (72 minutes converted to hours)
  distanceCovered speed time = 12 := by
  sorry

end distance_covered_72min_10kmph_l2387_238787


namespace min_value_2a5_plus_a4_l2387_238726

/-- A geometric sequence with positive terms satisfying a specific condition -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  condition : 2 * a 4 + a 3 - 2 * a 2 - a 1 = 8

/-- The minimum value of 2a_5 + a_4 for the given geometric sequence -/
theorem min_value_2a5_plus_a4 (seq : GeometricSequence) :
  ∃ m : ℝ, m = 12 * Real.sqrt 3 ∧ ∀ x : ℝ, (2 * seq.a 5 + seq.a 4) ≥ m :=
sorry

end min_value_2a5_plus_a4_l2387_238726


namespace morse_high_school_seniors_l2387_238799

/-- The number of seniors at Morse High School -/
def num_seniors : ℕ := 300

/-- The number of students in the lower grades (freshmen, sophomores, and juniors) -/
def num_lower_grades : ℕ := 900

/-- The percentage of seniors who have cars -/
def senior_car_percentage : ℚ := 1/2

/-- The percentage of lower grade students who have cars -/
def lower_grade_car_percentage : ℚ := 1/10

/-- The percentage of all students who have cars -/
def total_car_percentage : ℚ := 1/5

theorem morse_high_school_seniors :
  (num_seniors * senior_car_percentage + num_lower_grades * lower_grade_car_percentage : ℚ) = 
  ((num_seniors + num_lower_grades) * total_car_percentage : ℚ) := by
  sorry

end morse_high_school_seniors_l2387_238799


namespace x_value_l2387_238742

theorem x_value (x : ℚ) (h : 1/3 - 1/4 = 4/x) : x = 48 := by
  sorry

end x_value_l2387_238742


namespace stamps_per_page_l2387_238795

theorem stamps_per_page (book1 book2 book3 : ℕ) 
  (h1 : book1 = 945) 
  (h2 : book2 = 1260) 
  (h3 : book3 = 1575) : 
  Nat.gcd book1 (Nat.gcd book2 book3) = 315 := by
  sorry

end stamps_per_page_l2387_238795


namespace dodecahedron_interior_diagonals_l2387_238741

/-- A dodecahedron is a 3-dimensional figure with 12 pentagonal faces and 20 vertices,
    where 3 faces meet at each vertex. -/
structure Dodecahedron where
  vertices : Nat
  faces : Nat
  faces_per_vertex : Nat
  vertices_eq : vertices = 20
  faces_eq : faces = 12
  faces_per_vertex_eq : faces_per_vertex = 3

/-- An interior diagonal of a dodecahedron is a segment connecting two vertices
    which do not lie on a common face. -/
def interior_diagonal (d : Dodecahedron) : Nat :=
  sorry

/-- The number of interior diagonals in a dodecahedron is 160. -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) :
  interior_diagonal d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l2387_238741


namespace linear_diophantine_equation_solutions_l2387_238727

theorem linear_diophantine_equation_solutions
  (a b c x₀ y₀ : ℤ)
  (h_gcd : Int.gcd a b = 1)
  (h_solution : a * x₀ + b * y₀ = c) :
  ∀ x y : ℤ, a * x + b * y = c →
    ∃ k : ℤ, x = x₀ + k * b ∧ y = y₀ - k * a :=
sorry

end linear_diophantine_equation_solutions_l2387_238727


namespace circle_equation_tangent_to_line_l2387_238732

/-- The equation of a circle with center (0, b) that is tangent to the line y = 2x + 1 at point (1, 3) -/
theorem circle_equation_tangent_to_line (b : ℝ) :
  (∀ x y : ℝ, y = 2 * x + 1 → (x - 1)^2 + (y - 3)^2 ≠ 0) →
  (1 : ℝ)^2 + (3 - b)^2 = (0 - 1)^2 + ((2 * 0 + 1) - b)^2 →
  (∀ x y : ℝ, (x : ℝ)^2 + (y - 7/2)^2 = 5/4 ↔ (x - 0)^2 + (y - b)^2 = (1 - 0)^2 + (3 - b)^2) :=
by sorry

end circle_equation_tangent_to_line_l2387_238732


namespace perfectville_run_difference_l2387_238798

theorem perfectville_run_difference (street_width : ℕ) (block_side : ℕ) : 
  street_width = 30 → block_side = 500 → 
  4 * (block_side + 2 * street_width) - 4 * block_side = 240 :=
by
  sorry

end perfectville_run_difference_l2387_238798


namespace twentieth_decimal_of_35_36_l2387_238789

/-- The fraction we're considering -/
def f : ℚ := 35 / 36

/-- The nth decimal digit in the decimal expansion of a rational number -/
noncomputable def nthDecimalDigit (q : ℚ) (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 20th decimal digit of 35/36 is 2 -/
theorem twentieth_decimal_of_35_36 : nthDecimalDigit f 20 = 2 := by sorry

end twentieth_decimal_of_35_36_l2387_238789


namespace repeating_decimal_equals_fraction_l2387_238761

/-- The repeating decimal 4.363636... -/
def repeating_decimal : ℚ := 4 + 36 / 99

/-- The fraction 144/33 -/
def fraction : ℚ := 144 / 33

/-- Theorem stating that the repeating decimal 4.363636... is equal to the fraction 144/33 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end repeating_decimal_equals_fraction_l2387_238761


namespace trivia_team_size_l2387_238773

theorem trivia_team_size :
  let members_absent : ℝ := 2
  let total_score : ℝ := 6
  let score_per_member : ℝ := 2
  let members_present : ℝ := total_score / score_per_member
  let total_members : ℝ := members_present + members_absent
  total_members = 5 := by
  sorry

end trivia_team_size_l2387_238773


namespace min_l_pieces_in_8x8_l2387_238728

/-- Represents an 8x8 square board --/
def Board := Fin 8 → Fin 8 → Bool

/-- Represents a three-cell L-shaped piece --/
structure LPiece where
  x : Fin 8
  y : Fin 8
  orientation : Fin 4

/-- Checks if an L-piece can be placed on the board --/
def canPlace (board : Board) (piece : LPiece) : Bool :=
  sorry

/-- Places an L-piece on the board --/
def placePiece (board : Board) (piece : LPiece) : Board :=
  sorry

/-- Checks if any more L-pieces can be placed on the board --/
def canPlaceMore (board : Board) : Bool :=
  sorry

/-- The main theorem --/
theorem min_l_pieces_in_8x8 :
  ∃ (pieces : List LPiece),
    pieces.length = 11 ∧
    (∃ (board : Board),
      (∀ p ∈ pieces, canPlace board p) ∧
      (∀ p ∈ pieces, board = placePiece board p) ∧
      ¬canPlaceMore board) ∧
    (∀ (pieces' : List LPiece),
      pieces'.length < 11 →
      ∀ (board : Board),
        (∀ p ∈ pieces', canPlace board p) →
        (∀ p ∈ pieces', board = placePiece board p) →
        canPlaceMore board) :=
  sorry

end min_l_pieces_in_8x8_l2387_238728


namespace nested_fraction_equality_l2387_238786

theorem nested_fraction_equality : 
  2 - (1 / (2 + (1 / (2 - (1 / 2))))) = -2/3 := by
  sorry

end nested_fraction_equality_l2387_238786


namespace specific_truck_toll_l2387_238745

/-- Calculates the toll for a truck crossing a bridge -/
def calculate_toll (x : ℕ) (w : ℝ) (peak_hours : Bool) : ℝ :=
  let y : ℝ := if peak_hours then 2 else 0
  3.50 + 0.50 * (x - 2 : ℝ) + 0.10 * w + y

/-- Theorem: The toll for a specific truck is $8.50 -/
theorem specific_truck_toll :
  calculate_toll 5 15 true = 8.50 := by
  sorry

end specific_truck_toll_l2387_238745


namespace cube_sum_greater_than_mixed_terms_l2387_238791

theorem cube_sum_greater_than_mixed_terms (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end cube_sum_greater_than_mixed_terms_l2387_238791


namespace tan_sum_identity_l2387_238747

theorem tan_sum_identity (x y z : Real) 
  (hx : x = 20 * π / 180)
  (hy : y = 30 * π / 180)
  (hz : z = 40 * π / 180)
  (h1 : Real.tan (60 * π / 180) = Real.sqrt 3)
  (h2 : Real.tan (30 * π / 180) = 1 / Real.sqrt 3) :
  Real.tan x * Real.tan y + Real.tan y * Real.tan z + Real.tan z * Real.tan x = 1 := by
  sorry

end tan_sum_identity_l2387_238747


namespace triangle_similarity_l2387_238730

theorem triangle_similarity (DC CB : ℝ) (AD AB ED : ℝ) (FC : ℝ) : 
  DC = 9 → 
  CB = 6 → 
  AB = (1/3) * AD → 
  ED = (2/3) * AD → 
  FC = 9 := by
sorry

end triangle_similarity_l2387_238730


namespace illumination_ways_l2387_238752

theorem illumination_ways (n : ℕ) (h : n = 6) : 2^n - 1 = 63 := by
  sorry

end illumination_ways_l2387_238752


namespace smallest_divisor_square_plus_divisor_square_l2387_238751

theorem smallest_divisor_square_plus_divisor_square (n : ℕ) : n ≥ 2 → (
  (∃ k d : ℕ, 
    k > 1 ∧ 
    k ∣ n ∧ 
    (∀ m : ℕ, m > 1 ∧ m ∣ n → k ≤ m) ∧ 
    d ∣ n ∧ 
    n = k^2 + d^2
  ) ↔ (n = 8 ∨ n = 20)
) := by sorry

end smallest_divisor_square_plus_divisor_square_l2387_238751


namespace sin_cos_equation_solution_l2387_238736

theorem sin_cos_equation_solution :
  ∃ x : ℝ, x = π / 14 ∧ Real.sin (3 * x) * Real.sin (4 * x) = Real.cos (3 * x) * Real.cos (4 * x) :=
by sorry

end sin_cos_equation_solution_l2387_238736


namespace max_material_a_units_l2387_238718

/-- Represents the quantities of materials A, B, and C --/
structure Materials where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given quantities satisfy the initial cost condition --/
def satisfiesInitialCost (m : Materials) : Prop :=
  3 * m.a + 5 * m.b + 7 * m.c = 62

/-- Checks if the given quantities satisfy the final cost condition --/
def satisfiesFinalCost (m : Materials) : Prop :=
  2 * m.a + 4 * m.b + 6 * m.c = 50

/-- Theorem stating the maximum number of units of material A --/
theorem max_material_a_units :
  ∃ (m : Materials),
    satisfiesInitialCost m ∧
    satisfiesFinalCost m ∧
    m.a = 5 ∧
    ∀ (m' : Materials),
      satisfiesInitialCost m' ∧
      satisfiesFinalCost m' →
      m'.a ≤ m.a :=
by sorry


end max_material_a_units_l2387_238718


namespace function_range_condition_l2387_238781

/-- Given functions f and g, prove that m ≥ 3/2 under specified conditions -/
theorem function_range_condition (m : ℝ) (h_m : m > 0) : 
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₂ ∈ Set.Icc (-1 : ℝ) 2, 
    ((1/2 : ℝ) ^ x₁) = m * x₂ - 1) → 
  m ≥ 3/2 := by
  sorry


end function_range_condition_l2387_238781


namespace min_value_system_l2387_238737

theorem min_value_system (x y k : ℝ) :
  (3 * x + y ≥ 0) →
  (4 * x + 3 * y ≥ k) →
  (∀ x' y', (3 * x' + y' ≥ 0) → (4 * x' + 3 * y' ≥ k) → (2 * x' + 4 * y' ≥ 2 * x + 4 * y)) →
  (2 * x + 4 * y = -6) →
  (k ≤ 0 ∧ ∀ m : ℤ, m > 0 → ¬(k ≥ m)) :=
by sorry

end min_value_system_l2387_238737


namespace f_max_value_l2387_238739

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sqrt (2 * x^3 + 7 * x^2 + 6 * x)) / (x^2 + 4 * x + 3)

theorem f_max_value :
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f x ≤ 1/2) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ f x = 1/2) :=
sorry

end f_max_value_l2387_238739


namespace square_root_of_25_l2387_238779

theorem square_root_of_25 : ∃ (a b : ℝ), a ≠ b ∧ a^2 = 25 ∧ b^2 = 25 := by
  sorry

#check square_root_of_25

end square_root_of_25_l2387_238779


namespace train_crossing_time_l2387_238753

/-- The time taken for a train to cross a stationary point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 150 → 
  train_speed_kmh = 180 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 3 := by
  sorry

end train_crossing_time_l2387_238753


namespace digit_452_of_7_19_is_6_l2387_238744

/-- The decimal representation of 7/19 is repeating -/
def decimal_rep_7_19_repeating : Prop := 
  ∃ (s : List Nat), s.length > 0 ∧ (7 : ℚ) / 19 = (s.map (λ n => (n : ℚ) / 10^s.length)).sum

/-- The 452nd digit after the decimal point in the decimal representation of 7/19 -/
def digit_452_of_7_19 : Nat := sorry

theorem digit_452_of_7_19_is_6 (h : decimal_rep_7_19_repeating) : 
  digit_452_of_7_19 = 6 := by sorry

end digit_452_of_7_19_is_6_l2387_238744


namespace boxes_with_neither_l2387_238750

theorem boxes_with_neither (total : ℕ) (crayons : ℕ) (markers : ℕ) (both : ℕ) : 
  total = 15 → crayons = 9 → markers = 6 → both = 4 →
  total - (crayons + markers - both) = 4 :=
by
  sorry

end boxes_with_neither_l2387_238750


namespace valid_arrangement_exists_l2387_238717

/-- Represents a 3x3 matrix of integers -/
def Matrix3x3 := Fin 3 → Fin 3 → ℤ

/-- Checks if two integers are coprime -/
def are_coprime (a b : ℤ) : Prop := Nat.gcd a.natAbs b.natAbs = 1

/-- Checks if the matrix satisfies the adjacency condition -/
def satisfies_adjacency_condition (m : Matrix3x3) : Prop :=
  ∀ i j i' j', (i = i' ∧ j.succ = j') ∨ (i = i' ∧ j = j'.succ) ∨
                (i.succ = i' ∧ j = j') ∨ (i = i'.succ ∧ j = j') ∨
                (i.succ = i' ∧ j.succ = j') ∨ (i.succ = i' ∧ j = j'.succ) ∨
                (i = i'.succ ∧ j.succ = j') ∨ (i = i'.succ ∧ j = j'.succ) →
                are_coprime (m i j) (m i' j')

/-- Checks if the matrix contains nine consecutive integers -/
def contains_consecutive_integers (m : Matrix3x3) : Prop :=
  ∃ start : ℤ, ∀ i j, ∃ k : ℕ, k < 9 ∧ m i j = start + k

/-- The main theorem stating the existence of a valid arrangement -/
theorem valid_arrangement_exists : ∃ m : Matrix3x3, 
  satisfies_adjacency_condition m ∧ contains_consecutive_integers m := by
  sorry

end valid_arrangement_exists_l2387_238717


namespace exists_always_last_card_l2387_238793

/-- Represents a card with a unique natural number -/
structure Card where
  number : ℕ
  unique : ℕ

/-- Represents the circular arrangement of cards -/
def CardArrangement := Vector Card 1000

/-- Simulates the card removal process -/
def removeCards (arrangement : CardArrangement) (startIndex : Fin 1000) : Card :=
  sorry

/-- Checks if a card is the last remaining for all starting positions except its own -/
def isAlwaysLast (arrangement : CardArrangement) (cardIndex : Fin 1000) : Prop :=
  ∀ i : Fin 1000, i ≠ cardIndex → removeCards arrangement i = arrangement.get cardIndex

/-- Main theorem: There exists a card arrangement where one card is always the last remaining -/
theorem exists_always_last_card : ∃ (arrangement : CardArrangement), ∃ (i : Fin 1000), isAlwaysLast arrangement i :=
  sorry

end exists_always_last_card_l2387_238793


namespace segment_length_sum_l2387_238706

theorem segment_length_sum (a : ℝ) : 
  let point1 := (3 * a, 2 * a - 5)
  let point2 := (5, -2)
  let distance := Real.sqrt ((point1.1 - point2.1)^2 + (point1.2 - point2.2)^2)
  distance = 3 * Real.sqrt 5 →
  ∃ (a1 a2 : ℝ), a1 ≠ a2 ∧ 
    (∀ x : ℝ, Real.sqrt ((3*x - 5)^2 + (2*x - 3)^2) = 3 * Real.sqrt 5 ↔ x = a1 ∨ x = a2) ∧
    a1 + a2 = 3.231 :=
by sorry

end segment_length_sum_l2387_238706


namespace area_of_region_is_4pi_l2387_238788

-- Define the region
def region (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y = -1

-- Define the area of the region
noncomputable def area_of_region : ℝ := sorry

-- Theorem statement
theorem area_of_region_is_4pi :
  area_of_region = 4 * Real.pi :=
sorry

end area_of_region_is_4pi_l2387_238788


namespace regression_analysis_appropriate_for_height_weight_l2387_238762

/-- Represents a statistical analysis method -/
inductive AnalysisMethod
  | ResidualAnalysis
  | RegressionAnalysis
  | IsoplethBarChart
  | IndependenceTest

/-- Represents a variable in the context of statistical analysis -/
structure Variable where
  name : String

/-- Represents a relationship between two variables -/
structure Relationship where
  var1 : Variable
  var2 : Variable
  correlated : Bool

/-- Determines if a given analysis method is appropriate for analyzing a relationship between two variables -/
def is_appropriate_method (method : AnalysisMethod) (rel : Relationship) : Prop :=
  method = AnalysisMethod.RegressionAnalysis ∧ rel.correlated = true

/-- Main theorem: Regression analysis is the appropriate method for analyzing the relationship between height and weight -/
theorem regression_analysis_appropriate_for_height_weight :
  let height : Variable := ⟨"height"⟩
  let weight : Variable := ⟨"weight"⟩
  let height_weight_rel : Relationship := ⟨height, weight, true⟩
  is_appropriate_method AnalysisMethod.RegressionAnalysis height_weight_rel :=
by
  sorry


end regression_analysis_appropriate_for_height_weight_l2387_238762


namespace gcd_sum_and_count_even_integers_l2387_238709

def sum_even_integers (a b : ℕ) : ℕ :=
  let first := if a % 2 = 0 then a else a + 1
  let last := if b % 2 = 0 then b else b - 1
  let n := (last - first) / 2 + 1
  n * (first + last) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  let first := if a % 2 = 0 then a else a + 1
  let last := if b % 2 = 0 then b else b - 1
  (last - first) / 2 + 1

theorem gcd_sum_and_count_even_integers :
  Nat.gcd (sum_even_integers 13 63) (count_even_integers 13 63) = 25 := by
  sorry

end gcd_sum_and_count_even_integers_l2387_238709


namespace fraction_problem_l2387_238767

theorem fraction_problem (x : ℚ) : 
  x / (4 * x - 9) = 3 / 4 → x = 27 / 8 := by
sorry

end fraction_problem_l2387_238767


namespace roses_picked_l2387_238708

theorem roses_picked (initial : ℕ) (sold : ℕ) (final : ℕ) : initial = 37 → sold = 16 → final = 40 → final - (initial - sold) = 19 := by
  sorry

end roses_picked_l2387_238708


namespace value_of_a_l2387_238715

theorem value_of_a (a b c : ℤ) (h1 : a + b = 10) (h2 : b + c = 8) (h3 : c = 4) : a = 6 := by
  sorry

end value_of_a_l2387_238715


namespace area_between_squares_l2387_238782

/-- The area of the region between two squares, where a smaller square is entirely contained within a larger square -/
theorem area_between_squares (larger_side smaller_side : ℝ) 
  (h1 : larger_side = 8) 
  (h2 : smaller_side = 4) 
  (h3 : smaller_side ≤ larger_side) : 
  larger_side ^ 2 - smaller_side ^ 2 = 48 := by
  sorry

end area_between_squares_l2387_238782


namespace intersection_complement_problem_l2387_238734

open Set

theorem intersection_complement_problem (U M N : Set ℕ) : 
  U = {0, 1, 2, 3, 4, 5} →
  M = {0, 3, 5} →
  N = {1, 4, 5} →
  M ∩ (U \ N) = {0, 3} := by
  sorry

end intersection_complement_problem_l2387_238734


namespace smallest_valid_number_l2387_238768

def is_valid (n : ℕ) : Prop :=
  ∀ k : ℕ, 2 ≤ k → k ≤ 12 → n % k = k - 1

theorem smallest_valid_number : 
  (is_valid 27719) ∧ (∀ m : ℕ, m < 27719 → ¬(is_valid m)) :=
sorry

end smallest_valid_number_l2387_238768


namespace range_of_x_minus_2y_l2387_238796

theorem range_of_x_minus_2y (x y : ℝ) (hx : -1 ≤ x ∧ x < 2) (hy : 0 < y ∧ y ≤ 1) :
  ∃ (z : ℝ), -3 ≤ z ∧ z < 2 ∧ ∃ (x' y' : ℝ), -1 ≤ x' ∧ x' < 2 ∧ 0 < y' ∧ y' ≤ 1 ∧ z = x' - 2*y' :=
by
  sorry

end range_of_x_minus_2y_l2387_238796


namespace percentage_difference_l2387_238700

theorem percentage_difference : (60 / 100 * 50) - (50 / 100 * 30) = 15 := by
  sorry

end percentage_difference_l2387_238700


namespace max_circular_triples_14_players_l2387_238783

/-- Represents a round-robin tournament --/
structure Tournament :=
  (num_players : ℕ)
  (games_per_player : ℕ)
  (no_draws : Bool)

/-- Calculates the maximum number of circular triples in a tournament --/
def max_circular_triples (t : Tournament) : ℕ :=
  sorry

/-- Theorem: In a 14-player round-robin tournament where each player plays 13 games
    and there are no draws, the maximum number of circular triples is 112 --/
theorem max_circular_triples_14_players :
  let t : Tournament := ⟨14, 13, true⟩
  max_circular_triples t = 112 := by sorry

end max_circular_triples_14_players_l2387_238783


namespace correct_dimes_calculation_l2387_238733

/-- Represents the number of dimes each sibling has -/
structure Dimes where
  barry : ℕ
  dan : ℕ
  emily : ℕ
  frank : ℕ

/-- Calculates the correct number of dimes for each sibling based on the given conditions -/
def calculate_dimes : Dimes :=
  let barry_dimes := 1000 / 10  -- $10.00 worth of dimes
  let dan_initial := barry_dimes / 2
  let dan_final := dan_initial + 2
  let emily_dimes := 2 * dan_initial
  let frank_dimes := emily_dimes - 7
  { barry := barry_dimes
  , dan := dan_final
  , emily := emily_dimes
  , frank := frank_dimes }

/-- Theorem stating that the calculated dimes match the expected values -/
theorem correct_dimes_calculation : 
  let dimes := calculate_dimes
  dimes.barry = 100 ∧ 
  dimes.dan = 52 ∧ 
  dimes.emily = 100 ∧ 
  dimes.frank = 93 := by
  sorry

end correct_dimes_calculation_l2387_238733


namespace joan_balloons_l2387_238704

/-- Joan and Melanie's blue balloons problem -/
theorem joan_balloons (joan_balloons : ℕ) (melanie_balloons : ℕ) (total_balloons : ℕ)
    (h1 : melanie_balloons = 41)
    (h2 : total_balloons = 81)
    (h3 : joan_balloons + melanie_balloons = total_balloons) :
  joan_balloons = 40 := by
  sorry

end joan_balloons_l2387_238704


namespace intersection_points_form_circle_l2387_238735

-- Define the system of equations
def equation1 (s x y : ℝ) : Prop := 3 * s * x - 5 * y - 7 * s = 0
def equation2 (s x y : ℝ) : Prop := 2 * x - 5 * s * y + 4 = 0

-- Define the set of points satisfying both equations
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ s : ℝ, equation1 s p.1 p.2 ∧ equation2 s p.1 p.2}

-- Theorem stating that the intersection points form a circle
theorem intersection_points_form_circle :
  ∃ c : ℝ × ℝ, ∃ r : ℝ, ∀ p ∈ intersection_points,
    (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 :=
sorry

end intersection_points_form_circle_l2387_238735


namespace tree_growth_condition_l2387_238713

/-- Represents the annual growth of a tree over 6 years -/
structure TreeGrowth where
  initial_height : ℝ
  annual_increase : ℝ

/-- Calculates the height of the tree after a given number of years -/
def height_after_years (t : TreeGrowth) (years : ℕ) : ℝ :=
  t.initial_height + t.annual_increase * years

/-- Theorem stating the condition for the tree's growth -/
theorem tree_growth_condition (t : TreeGrowth) : 
  t.initial_height = 4 ∧ 
  height_after_years t 6 = height_after_years t 4 + (1/7) * height_after_years t 4 →
  t.annual_increase = 2/5 :=
sorry

end tree_growth_condition_l2387_238713


namespace valentine_spending_percentage_l2387_238777

def total_students : ℕ := 30
def valentine_percentage : ℚ := 60 / 100
def valentine_cost : ℚ := 2
def total_money : ℚ := 40

theorem valentine_spending_percentage :
  (↑total_students * valentine_percentage * valentine_cost) / total_money * 100 = 90 := by
  sorry

end valentine_spending_percentage_l2387_238777


namespace g_at_negative_two_l2387_238746

def g (x : ℝ) : ℝ := 3 * x^5 - 4 * x^4 + 2 * x^3 - 5 * x^2 - x + 8

theorem g_at_negative_two : g (-2) = -186 := by
  sorry

end g_at_negative_two_l2387_238746


namespace shaded_area_percentage_l2387_238749

/-- Given two congruent squares with side length 12 that overlap to form a 12 by 20 rectangle,
    prove that 20% of the rectangle's area is shaded. -/
theorem shaded_area_percentage (side_length : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) :
  side_length = 12 →
  rectangle_width = 12 →
  rectangle_length = 20 →
  (side_length * side_length - rectangle_width * rectangle_length) / (rectangle_width * rectangle_length) * 100 = 20 := by
  sorry

end shaded_area_percentage_l2387_238749


namespace min_sum_of_primes_l2387_238780

/-- Given distinct positive integers a and b, where 20a + 17b and 17a + 20b
    are both prime numbers, the minimum sum of these prime numbers is 296. -/
theorem min_sum_of_primes (a b : ℕ+) (h_distinct : a ≠ b)
  (h_prime1 : Nat.Prime (20 * a + 17 * b))
  (h_prime2 : Nat.Prime (17 * a + 20 * b)) :
  (20 * a + 17 * b) + (17 * a + 20 * b) ≥ 296 := by
  sorry

end min_sum_of_primes_l2387_238780


namespace sum_of_leading_digits_of_roots_l2387_238794

/-- A function that returns the leading digit of a positive real number -/
def leadingDigit (x : ℝ) : ℕ :=
  sorry

/-- The number M, which is a 303-digit number consisting only of 5s -/
def M : ℕ := sorry

/-- The function g that returns the leading digit of the r-th root of M -/
def g (r : ℕ) : ℕ :=
  leadingDigit (M ^ (1 / r : ℝ))

/-- Theorem stating that the sum of g(2) to g(6) is 10 -/
theorem sum_of_leading_digits_of_roots :
  g 2 + g 3 + g 4 + g 5 + g 6 = 10 := by sorry

end sum_of_leading_digits_of_roots_l2387_238794


namespace only_101_prime_l2387_238729

/-- A number in the form 101010...101 with 2n+1 digits -/
def A (n : ℕ) : ℕ := (10^(2*n+2) - 1) / 99

/-- Predicate to check if a number is in the form 101010...101 -/
def is_alternating_101 (x : ℕ) : Prop :=
  ∃ n : ℕ, x = A n

/-- Main theorem: 101 is the only prime number with alternating 1s and 0s -/
theorem only_101_prime :
  ∀ p : ℕ, Prime p ∧ is_alternating_101 p ↔ p = 101 :=
sorry

end only_101_prime_l2387_238729


namespace complex_equation_solution_l2387_238766

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I) * z = 1 → z = (1 / 2 : ℂ) + Complex.I / 2 := by
  sorry

end complex_equation_solution_l2387_238766


namespace wheel_rotation_l2387_238757

/-- Given three wheels A, B, and C with radii 35 cm, 20 cm, and 8 cm respectively,
    where wheel A rotates through an angle of 72°, and all wheels rotate without slipping,
    prove that wheel C rotates through an angle of 315°. -/
theorem wheel_rotation (r_A r_B r_C : ℝ) (θ_A θ_C : ℝ) : 
  r_A = 35 →
  r_B = 20 →
  r_C = 8 →
  θ_A = 72 →
  r_A * θ_A = r_C * θ_C →
  θ_C = 315 := by
  sorry

#check wheel_rotation

end wheel_rotation_l2387_238757


namespace expression_equality_l2387_238720

theorem expression_equality (a b : ℝ) : -2 * (3 * a - b) + 3 * (2 * a + b) = 5 * b := by
  sorry

end expression_equality_l2387_238720


namespace garden_comparison_l2387_238763

-- Define the dimensions of Karl's garden
def karl_length : ℕ := 30
def karl_width : ℕ := 40

-- Define the dimensions of Makenna's garden
def makenna_side : ℕ := 35

-- Theorem to prove the comparison of areas and perimeters
theorem garden_comparison :
  (makenna_side * makenna_side - karl_length * karl_width = 25) ∧
  (2 * (karl_length + karl_width) = 4 * makenna_side) :=
by sorry

end garden_comparison_l2387_238763


namespace gcd_equality_implies_equal_l2387_238705

theorem gcd_equality_implies_equal (a b c : ℕ+) :
  a + Nat.gcd a b = b + Nat.gcd b c ∧
  b + Nat.gcd b c = c + Nat.gcd c a →
  a = b ∧ b = c := by
  sorry

end gcd_equality_implies_equal_l2387_238705


namespace expected_net_profit_l2387_238758

/-- The expected value of net profit from selling one electronic product -/
theorem expected_net_profit (purchase_price : ℝ) (pass_rate : ℝ) (profit_qualified : ℝ) (loss_defective : ℝ)
  (h1 : purchase_price = 10)
  (h2 : pass_rate = 0.95)
  (h3 : profit_qualified = 2)
  (h4 : loss_defective = 10) :
  profit_qualified * pass_rate + (-loss_defective) * (1 - pass_rate) = 1.4 := by
sorry

end expected_net_profit_l2387_238758


namespace function_value_at_2009_l2387_238722

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y - f (2 * x * y + 3) + 3 * f (x + y) - 3 * f x = -6 * x

/-- The main theorem stating that for a function satisfying the functional equation, f(2009) = 4021 -/
theorem function_value_at_2009 (f : ℝ → ℝ) (h : FunctionalEquation f) : f 2009 = 4021 := by
  sorry

end function_value_at_2009_l2387_238722


namespace james_payment_is_correct_l2387_238716

def james_total_payment (steak_price dessert_price drink_price : ℚ)
  (steak_discount : ℚ) (friend_steak_price friend_dessert_price friend_drink_price : ℚ)
  (friend_steak_discount : ℚ) (meal_tax_rate drink_tax_rate : ℚ)
  (james_tip_rate : ℚ) : ℚ :=
  let james_meal := steak_price * (1 - steak_discount)
  let friend_meal := friend_steak_price * (1 - friend_steak_discount)
  let james_total := james_meal + dessert_price + drink_price
  let friend_total := friend_meal + friend_dessert_price + friend_drink_price
  let james_tax := james_meal * meal_tax_rate + dessert_price * meal_tax_rate + drink_price * drink_tax_rate
  let friend_tax := friend_meal * meal_tax_rate + friend_dessert_price * meal_tax_rate + friend_drink_price * drink_tax_rate
  let total_bill := james_total + friend_total + james_tax + friend_tax
  let james_share := total_bill / 2
  let james_tip := james_share * james_tip_rate
  james_share + james_tip

theorem james_payment_is_correct :
  james_total_payment 16 5 3 0.1 14 4 2 0.05 0.08 0.05 0.2 = 265/10 := by sorry

end james_payment_is_correct_l2387_238716


namespace jaylen_cucumbers_count_l2387_238769

/-- The number of cucumbers Jaylen has -/
def jaylen_cucumbers (jaylen_carrots jaylen_bell_peppers jaylen_green_beans kristin_bell_peppers kristin_green_beans jaylen_total : ℕ) : ℕ :=
  jaylen_total - (jaylen_carrots + jaylen_bell_peppers + jaylen_green_beans)

theorem jaylen_cucumbers_count :
  ∀ (jaylen_carrots jaylen_bell_peppers jaylen_green_beans kristin_bell_peppers kristin_green_beans jaylen_total : ℕ),
  jaylen_carrots = 5 →
  jaylen_bell_peppers = 2 * kristin_bell_peppers →
  jaylen_green_beans = kristin_green_beans / 2 - 3 →
  kristin_bell_peppers = 2 →
  kristin_green_beans = 20 →
  jaylen_total = 18 →
  jaylen_cucumbers jaylen_carrots jaylen_bell_peppers jaylen_green_beans kristin_bell_peppers kristin_green_beans jaylen_total = 2 :=
by
  sorry

end jaylen_cucumbers_count_l2387_238769


namespace peaches_bought_is_seven_l2387_238785

/-- Represents the cost of fruits and the quantity purchased. -/
structure FruitPurchase where
  apple_cost : ℕ
  peach_cost : ℕ
  total_fruits : ℕ
  total_cost : ℕ

/-- Calculates the number of peaches bought given a FruitPurchase. -/
def peaches_bought (purchase : FruitPurchase) : ℕ :=
  let apple_count := purchase.total_fruits - (purchase.total_cost - purchase.apple_cost * purchase.total_fruits) / (purchase.peach_cost - purchase.apple_cost)
  purchase.total_fruits - apple_count

/-- Theorem stating that given the specific conditions, 7 peaches were bought. -/
theorem peaches_bought_is_seven : 
  ∀ (purchase : FruitPurchase), 
    purchase.apple_cost = 1000 → 
    purchase.peach_cost = 2000 → 
    purchase.total_fruits = 15 → 
    purchase.total_cost = 22000 → 
    peaches_bought purchase = 7 := by
  sorry


end peaches_bought_is_seven_l2387_238785
