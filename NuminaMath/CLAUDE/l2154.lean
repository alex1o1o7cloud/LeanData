import Mathlib

namespace company_employees_l2154_215446

theorem company_employees (december_employees : ℕ) (percentage_increase : ℚ) 
  (h1 : december_employees = 987)
  (h2 : percentage_increase = 127 / 1000) : 
  ∃ january_employees : ℕ, 
    (january_employees : ℚ) * (1 + percentage_increase) = december_employees ∧ 
    january_employees = 875 := by
  sorry

end company_employees_l2154_215446


namespace toris_initial_height_l2154_215492

/-- Given Tori's growth and current height, prove her initial height --/
theorem toris_initial_height (growth : ℝ) (current_height : ℝ) 
  (h1 : growth = 2.86)
  (h2 : current_height = 7.26) :
  current_height - growth = 4.40 := by
  sorry

end toris_initial_height_l2154_215492


namespace least_four_digit_multiple_of_seven_l2154_215477

theorem least_four_digit_multiple_of_seven : ∃ n : ℕ,
  n = 1001 ∧
  7 ∣ n ∧
  1000 ≤ n ∧
  n < 10000 ∧
  ∀ m : ℕ, 7 ∣ m → 1000 ≤ m → m < 10000 → n ≤ m :=
by sorry

end least_four_digit_multiple_of_seven_l2154_215477


namespace select_team_count_l2154_215486

/-- The number of ways to select a team of 7 people from a group of 7 boys and 9 girls, 
    with at least 3 girls in the team -/
def selectTeam (numBoys numGirls teamSize minGirls : ℕ) : ℕ :=
  (Finset.range (teamSize - minGirls + 1)).sum fun i =>
    Nat.choose numGirls (minGirls + i) * Nat.choose numBoys (teamSize - minGirls - i)

/-- Theorem stating that the number of ways to select the team is 10620 -/
theorem select_team_count : selectTeam 7 9 7 3 = 10620 := by
  sorry

end select_team_count_l2154_215486


namespace square_difference_l2154_215480

theorem square_difference (x y : ℝ) 
  (h1 : (x + y)^2 = 64) 
  (h2 : x * y = 15) : 
  (x - y)^2 = 4 := by
sorry

end square_difference_l2154_215480


namespace article_cost_price_l2154_215416

theorem article_cost_price (C : ℝ) (S : ℝ) : 
  S = 1.05 * C →
  S - 3 = 1.1 * (0.95 * C) →
  C = 600 := by
sorry

end article_cost_price_l2154_215416


namespace macaroons_remaining_l2154_215461

def remaining_macaroons (initial_red : ℕ) (initial_green : ℕ) (eaten_green : ℕ) : ℕ :=
  let eaten_red := 2 * eaten_green
  (initial_red - eaten_red) + (initial_green - eaten_green)

theorem macaroons_remaining :
  remaining_macaroons 50 40 15 = 45 := by
  sorry

end macaroons_remaining_l2154_215461


namespace vowel_word_count_l2154_215420

/-- The number of times vowels A and E appear -/
def vowel_count_ae : ℕ := 6

/-- The number of times vowels I, O, and U appear -/
def vowel_count_iou : ℕ := 5

/-- The length of the words to be formed -/
def word_length : ℕ := 6

/-- The total number of vowel choices for each position -/
def total_choices : ℕ := 2 * vowel_count_ae + 3 * vowel_count_iou

/-- Theorem stating the number of possible six-letter words -/
theorem vowel_word_count : (total_choices ^ word_length : ℕ) = 531441 := by
  sorry

end vowel_word_count_l2154_215420


namespace robin_gum_count_l2154_215410

theorem robin_gum_count (initial_gum : Real) (additional_gum : Real) : 
  initial_gum = 18.5 → additional_gum = 44.25 → initial_gum + additional_gum = 62.75 := by
  sorry

end robin_gum_count_l2154_215410


namespace ellipse_line_intersection_l2154_215485

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the line
def Line (k m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + m}

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define the vector from P to Q
def vector (P Q : ℝ × ℝ) : ℝ × ℝ :=
  (Q.1 - P.1, Q.2 - P.2)

-- Define the squared magnitude of a vector
def magnitude_squared (v : ℝ × ℝ) : ℝ :=
  v.1^2 + v.2^2

theorem ellipse_line_intersection :
  ∃ (k m : ℝ), 
    let C := Ellipse 2 (Real.sqrt 3)
    let l := Line k m
    let P := (2, 1)
    let M := (1, 3/2)
    (∀ x y, (x, y) ∈ C → (x^2 / 4) + (y^2 / 3) = 1) ∧ 
    (1, 3/2) ∈ C ∧
    (2, 1) ∈ l ∧
    (∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧ A ∈ l ∧ B ∈ l ∧ A ≠ B ∧
      dot_product (vector P A) (vector P B) = magnitude_squared (vector P M)) ∧
    k = 1/2 ∧ m = 1/2 := by
  sorry

end ellipse_line_intersection_l2154_215485


namespace circles_intersect_l2154_215457

-- Define the circles
def C1 (x y : ℝ) : Prop := (x + 2)^2 + (y - 2)^2 = 4
def C2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 16

-- Define the centers and radii
def center1 : ℝ × ℝ := (-2, 2)
def center2 : ℝ × ℝ := (2, 5)
def radius1 : ℝ := 2
def radius2 : ℝ := 4

-- Theorem statement
theorem circles_intersect :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  radius1 + radius2 > d ∧ d > abs (radius1 - radius2) := by sorry

end circles_intersect_l2154_215457


namespace range_of_x_l2154_215455

-- Define the function f
def f (x a : ℝ) := |x - 4| + |x - a|

-- State the theorem
theorem range_of_x (a : ℝ) (h1 : a > 1) 
  (h2 : ∃ (m : ℝ), ∀ x, f x a ≥ m ∧ ∃ y, f y a = m) 
  (h3 : (Classical.choose h2) = 3) :
  ∀ x, f x a ≤ 5 → 3 ≤ x ∧ x ≤ 8 := by
  sorry

#check range_of_x

end range_of_x_l2154_215455


namespace trilandia_sentinel_sites_l2154_215489

/-- Represents a triangular city with streets and sentinel sites. -/
structure TriangularCity where
  side_length : ℕ
  num_streets : ℕ

/-- Calculates the minimum number of sentinel sites required for a given triangular city. -/
def min_sentinel_sites (city : TriangularCity) : ℕ :=
  3 * (city.side_length / 2) - 1

/-- Theorem stating the minimum number of sentinel sites for Trilandia. -/
theorem trilandia_sentinel_sites :
  let trilandia : TriangularCity := ⟨2012, 6036⟩
  min_sentinel_sites trilandia = 3017 := by
  sorry

#eval min_sentinel_sites ⟨2012, 6036⟩

end trilandia_sentinel_sites_l2154_215489


namespace special_ellipse_equation_l2154_215433

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  -- The ellipse equation in the form (x²/a²) + (y²/b²) = 1
  a : ℝ
  b : ℝ
  -- Center at origin
  center_origin : True
  -- Foci on coordinate axis
  foci_on_axis : True
  -- Line y = x + 1 intersects the ellipse
  intersects_line : True
  -- OP ⊥ OQ where P and Q are intersection points
  op_perp_oq : True
  -- |PQ| = √10/2
  pq_length : True

/-- The theorem stating the possible equations of the special ellipse -/
theorem special_ellipse_equation (e : SpecialEllipse) :
  (∀ x y, x^2 + 3*y^2 = 2) ∨ (∀ x y, 3*x^2 + y^2 = 2) :=
sorry

end special_ellipse_equation_l2154_215433


namespace scientific_notation_of_11580000_l2154_215465

theorem scientific_notation_of_11580000 :
  (11580000 : ℝ) = 1.158 * (10 : ℝ)^7 := by
  sorry

end scientific_notation_of_11580000_l2154_215465


namespace x_value_proof_l2154_215441

theorem x_value_proof (y : ℝ) (x : ℝ) (h1 : y = -2) (h2 : (x - 2*y)^y = 0.001) :
  x = -4 + 10 * Real.sqrt 10 :=
sorry

end x_value_proof_l2154_215441


namespace probability_order_l2154_215403

-- Define the structure of a deck of cards
structure Card where
  suit : Fin 4
  rank : Fin 13

-- Define the deck
def standardDeck : Finset Card := sorry

-- Define the subsets of cards for each event
def fiveOfHearts : Finset Card := sorry
def jokers : Finset Card := sorry
def fives : Finset Card := sorry
def clubs : Finset Card := sorry
def redCards : Finset Card := sorry

-- Define the probability of drawing a card from a given set
def probability (subset : Finset Card) : ℚ :=
  (subset.card : ℚ) / (standardDeck.card : ℚ)

-- Theorem statement
theorem probability_order :
  probability fiveOfHearts < probability jokers ∧
  probability jokers < probability fives ∧
  probability fives < probability clubs ∧
  probability clubs < probability redCards :=
sorry

end probability_order_l2154_215403


namespace recycling_points_per_bag_l2154_215483

/-- Calculates the points earned per bag of recycled cans. -/
def points_per_bag (total_bags : ℕ) (unrecycled_bags : ℕ) (total_points : ℕ) : ℚ :=
  total_points / total_bags

theorem recycling_points_per_bag :
  let total_bags : ℕ := 4
  let unrecycled_bags : ℕ := 2
  let total_points : ℕ := 16
  points_per_bag total_bags unrecycled_bags total_points = 4 := by
  sorry

end recycling_points_per_bag_l2154_215483


namespace marias_number_l2154_215402

theorem marias_number (x : ℝ) : ((3 * (x - 3) + 3) / 3 = 10) → x = 12 := by
  sorry

end marias_number_l2154_215402


namespace expand_product_l2154_215400

theorem expand_product (x : ℝ) : (5*x + 3) * (3*x^2 + 4) = 15*x^3 + 9*x^2 + 20*x + 12 := by
  sorry

end expand_product_l2154_215400


namespace smallest_winning_number_l2154_215449

def bernardo_wins (N : ℕ) : Prop :=
  2 * N < 1000 ∧
  2 * N + 60 < 1000 ∧
  4 * N + 120 < 1000 ∧
  4 * N + 180 < 1000 ∧
  8 * N + 360 < 1000 ∧
  8 * N + 420 < 1000 ∧
  16 * N + 840 < 1000 ∧
  16 * N + 900 ≥ 1000

theorem smallest_winning_number :
  bernardo_wins 5 ∧ ∀ k : ℕ, k < 5 → ¬bernardo_wins k :=
sorry

end smallest_winning_number_l2154_215449


namespace area_of_triangle_formed_by_segment_l2154_215412

structure Rectangle where
  width : ℝ
  height : ℝ

structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

def Point := ℝ × ℝ

def Segment (p1 p2 : Point) := {p : Point | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (t * p1.1 + (1 - t) * p2.1, t * p1.2 + (1 - t) * p2.2)}

theorem area_of_triangle_formed_by_segment (rect : Rectangle) (tri : IsoscelesTriangle) 
  (h1 : rect.width = 15 ∧ rect.height = 10)
  (h2 : tri.base = 10 ∧ tri.height = 10)
  (h3 : (15, 0) = (rect.width, 0))
  (h4 : Segment (0, rect.height) (20, 10) ∩ Segment (15, 0) (25, 0) = {(15, 10)}) :
  (1 / 2) * rect.width * rect.height = 75 := by
  sorry

end area_of_triangle_formed_by_segment_l2154_215412


namespace min_value_of_expression_l2154_215417

theorem min_value_of_expression (a : ℝ) (h : 0 ≤ a ∧ a < 4) :
  ∃ m : ℝ, m = 1 ∧ ∀ x : ℝ, 0 ≤ x ∧ x < 4 → m ≤ |x - 2| + |3 - x| :=
by
  sorry

end min_value_of_expression_l2154_215417


namespace tan_150_degrees_l2154_215447

theorem tan_150_degrees :
  Real.tan (150 * π / 180) = -Real.sqrt 3 / 3 := by sorry

end tan_150_degrees_l2154_215447


namespace expand_squared_difference_product_expand_linear_factors_l2154_215472

/-- Theorem for the expansion of (2a-b)^2 * (2a+b)^2 -/
theorem expand_squared_difference_product (a b : ℝ) :
  (2*a - b)^2 * (2*a + b)^2 = 16*a^4 - 8*a^2*b^2 + b^4 := by sorry

/-- Theorem for the expansion of (3a+b-2)(3a-b+2) -/
theorem expand_linear_factors (a b : ℝ) :
  (3*a + b - 2) * (3*a - b + 2) = 9*a^2 - b^2 + 4*b - 4 := by sorry

end expand_squared_difference_product_expand_linear_factors_l2154_215472


namespace three_students_two_groups_l2154_215493

/-- The number of ways for students to sign up for activity groups. -/
def signUpWays (numStudents : ℕ) (numGroups : ℕ) : ℕ :=
  numGroups ^ numStudents

/-- Theorem: Three students signing up for two groups results in 8 ways. -/
theorem three_students_two_groups :
  signUpWays 3 2 = 8 := by
  sorry

end three_students_two_groups_l2154_215493


namespace only_height_weight_correlated_l2154_215440

-- Define the concept of a variable pair
structure VariablePair where
  var1 : String
  var2 : String

-- Define the concept of a functional relationship
def functionalRelationship (pair : VariablePair) : Prop := sorry

-- Define the concept of correlation
def correlated (pair : VariablePair) : Prop := sorry

-- Define the given variable pairs
def taxiFareDistance : VariablePair := ⟨"taxi fare", "distance traveled"⟩
def houseSizePrice : VariablePair := ⟨"house size", "house price"⟩
def heightWeight : VariablePair := ⟨"human height", "human weight"⟩
def ironSizeMass : VariablePair := ⟨"iron block size", "iron block mass"⟩

-- State the theorem
theorem only_height_weight_correlated :
  functionalRelationship taxiFareDistance →
  functionalRelationship houseSizePrice →
  (correlated heightWeight ∧ ¬functionalRelationship heightWeight) →
  functionalRelationship ironSizeMass →
  (correlated heightWeight ∧
   ¬correlated taxiFareDistance ∧
   ¬correlated houseSizePrice ∧
   ¬correlated ironSizeMass) := by
  sorry

end only_height_weight_correlated_l2154_215440


namespace intersection_of_spheres_integer_points_l2154_215426

theorem intersection_of_spheres_integer_points :
  let sphere1 := {(x, y, z) : ℤ × ℤ × ℤ | x^2 + y^2 + (z - 10)^2 ≤ 25}
  let sphere2 := {(x, y, z) : ℤ × ℤ × ℤ | x^2 + y^2 + (z - 4)^2 ≤ 36}
  ∃! p : ℤ × ℤ × ℤ, p ∈ sphere1 ∩ sphere2 :=
by
  sorry

end intersection_of_spheres_integer_points_l2154_215426


namespace constant_dot_product_l2154_215439

-- Define an equilateral triangle ABC with side length 2
def Triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = 2 ∧ dist B C = 2 ∧ dist C A = 2

-- Define a point P on side BC
def PointOnBC (B C P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • B + t • C

-- Vector dot product
def dot_product (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1) + (v.2 * w.2)

-- Vector addition
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 + w.1, v.2 + w.2)

-- Vector subtraction
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 - w.1, v.2 - w.2)

theorem constant_dot_product
  (A B C P : ℝ × ℝ)
  (h1 : Triangle A B C)
  (h2 : PointOnBC B C P) :
  dot_product (vec_sub P A) (vec_add (vec_sub B A) (vec_sub C A)) = 6 :=
sorry

end constant_dot_product_l2154_215439


namespace smallest_root_of_quadratic_l2154_215421

theorem smallest_root_of_quadratic (x : ℝ) :
  (12 * x^2 - 50 * x + 48 = 0) → (x ≥ 4/3) :=
by sorry

end smallest_root_of_quadratic_l2154_215421


namespace regular_polygon_perimeter_l2154_215425

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) : 
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  (360 : ℝ) / exterior_angle = n →
  n * side_length = 28 := by
  sorry

end regular_polygon_perimeter_l2154_215425


namespace chapters_read_l2154_215498

theorem chapters_read (num_books : ℕ) (chapters_per_book : ℕ) (total_chapters : ℕ) : 
  num_books = 10 → chapters_per_book = 24 → total_chapters = num_books * chapters_per_book →
  total_chapters = 240 :=
by sorry

end chapters_read_l2154_215498


namespace books_together_l2154_215494

/-- The number of books Tim and Mike have together -/
def total_books (tim_books mike_books : ℕ) : ℕ := tim_books + mike_books

/-- Theorem: Tim and Mike have 42 books together -/
theorem books_together : total_books 22 20 = 42 := by
  sorry

end books_together_l2154_215494


namespace chicken_chick_difference_l2154_215467

theorem chicken_chick_difference (total : ℕ) (chicks : ℕ) : 
  total = 821 → chicks = 267 → total - chicks - chicks = 287 := by
  sorry

end chicken_chick_difference_l2154_215467


namespace intersection_and_perpendicular_line_l2154_215414

-- Define the lines and conditions
def l1 (x y m : ℝ) : Prop := x + y - 3*m = 0
def l2 (x y m : ℝ) : Prop := 2*x - y + 2*m - 1 = 0
def y_intercept_l1 : ℝ := 3

-- Define the theorem
theorem intersection_and_perpendicular_line :
  ∃ (x y : ℝ),
    (∀ m : ℝ, l1 x y m ∧ l2 x y m) ∧
    x = 2/3 ∧ y = 7/3 ∧
    ∃ (k : ℝ), 3*x + 6*y + k = 0 ∧
    ∀ (x' y' : ℝ), l2 x' y' 1 → (x' - 2/3) + 2*(y' - 7/3) = 0 :=
by sorry

end intersection_and_perpendicular_line_l2154_215414


namespace min_mobots_correct_l2154_215432

/-- Represents a lawn as a grid with dimensions m and n -/
structure Lawn where
  m : ℕ
  n : ℕ

/-- Represents a mobot that can mow a lawn -/
inductive Mobot
  | east  : Mobot  -- Mobot moving east
  | north : Mobot  -- Mobot moving north

/-- Function to calculate the minimum number of mobots required -/
def minMobotsRequired (lawn : Lawn) : ℕ := min lawn.m lawn.n

/-- Theorem stating that minMobotsRequired gives the correct minimum number of mobots -/
theorem min_mobots_correct (lawn : Lawn) :
  ∀ (mobots : List Mobot), (∀ row col, row < lawn.m ∧ col < lawn.n → 
    ∃ mobot ∈ mobots, (mobot = Mobot.east ∧ ∃ r, r ≤ row) ∨ 
                       (mobot = Mobot.north ∧ ∃ c, c ≤ col)) →
  mobots.length ≥ minMobotsRequired lawn :=
sorry

end min_mobots_correct_l2154_215432


namespace min_max_sum_l2154_215491

theorem min_max_sum (p q r s t u : ℕ+) 
  (sum_eq : p + q + r + s + t + u = 2023) : 
  810 ≤ max (p + q) (max (q + r) (max (r + s) (max (s + t) (t + u)))) := by
  sorry

end min_max_sum_l2154_215491


namespace solution_product_l2154_215469

theorem solution_product (p q : ℝ) : 
  (p - 3) * (3 * p + 8) = p^2 - 5*p + 6 →
  (q - 3) * (3 * q + 8) = q^2 - 5*q + 6 →
  (p + 4) * (q + 4) = 7 := by
sorry

end solution_product_l2154_215469


namespace system_solution_l2154_215436

-- Define the system of equations
def equation1 (x y a b : ℝ) : Prop := x / (x - a) + y / (y - b) = 2
def equation2 (x y a b : ℝ) : Prop := a * x + b * y = 2 * a * b

-- State the theorem
theorem system_solution (a b : ℝ) (ha : a ≠ b) (hab : a + b ≠ 0) :
  ∃ x y : ℝ, equation1 x y a b ∧ equation2 x y a b ∧ x = 2 * a * b / (a + b) ∧ y = 2 * a * b / (a + b) :=
by sorry

end system_solution_l2154_215436


namespace doghouse_area_l2154_215466

/-- The area outside a regular hexagon that can be reached by a tethered point -/
theorem doghouse_area (side_length : ℝ) (rope_length : ℝ) (area : ℝ) :
  side_length = 2 →
  rope_length = 4 →
  area = 12 * Real.pi →
  area = (rope_length^2 * Real.pi * (2/3) + 2 * (side_length^2 * Real.pi * (1/6))) :=
by sorry

end doghouse_area_l2154_215466


namespace voting_difference_l2154_215451

/-- Represents the voting results for a company policy -/
structure VotingResults where
  total_employees : ℕ
  initial_for : ℕ
  initial_against : ℕ
  second_for : ℕ
  second_against : ℕ

/-- Conditions for the voting scenario -/
def voting_conditions (v : VotingResults) : Prop :=
  v.total_employees = 450 ∧
  v.initial_for + v.initial_against = v.total_employees ∧
  v.second_for + v.second_against = v.total_employees ∧
  v.initial_against > v.initial_for ∧
  v.second_for > v.second_against ∧
  (v.second_for - v.second_against) = 3 * (v.initial_against - v.initial_for) ∧
  v.second_for = (10 * v.initial_against) / 9

theorem voting_difference (v : VotingResults) 
  (h : voting_conditions v) : v.second_for - v.initial_for = 52 := by
  sorry

end voting_difference_l2154_215451


namespace two_circles_common_tangents_l2154_215428

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Number of common tangent lines between two circles -/
def commonTangentLines (c1 c2 : Circle) : ℕ :=
  sorry

/-- The main theorem -/
theorem two_circles_common_tangents :
  let c1 : Circle := { center := (1, 2), radius := 1 }
  let c2 : Circle := { center := (2, 5), radius := 3 }
  commonTangentLines c1 c2 = 2 := by
  sorry

end two_circles_common_tangents_l2154_215428


namespace linear_system_solution_l2154_215481

theorem linear_system_solution (m : ℚ) :
  let x : ℚ → ℚ := λ m => 6 * m + 1
  let y : ℚ → ℚ := λ m => -10 * m - 1
  (∀ m, x m + y m = -4 * m ∧ 2 * x m + y m = 2 * m + 1) ∧
  (x (1/2) - y (1/2) = 10) :=
by sorry

end linear_system_solution_l2154_215481


namespace fibonacci_like_sequence_l2154_215462

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) + a n

theorem fibonacci_like_sequence
  (a : ℕ → ℕ)
  (h_increasing : ∀ n : ℕ, a n < a (n + 1))
  (h_property : sequence_property a)
  (h_a7 : a 7 = 120) :
  a 8 = 194 := by
sorry

end fibonacci_like_sequence_l2154_215462


namespace cone_surface_area_and_volume_l2154_215454

/-- Represents a cone with given dimensions and properties -/
structure Cone where
  height : ℝ
  lateral_to_total_ratio : ℝ

/-- Calculates the surface area of the cone -/
def surface_area (c : Cone) : ℝ := sorry

/-- Calculates the volume of the cone -/
def volume (c : Cone) : ℝ := sorry

/-- Theorem stating the surface area and volume of a specific cone -/
theorem cone_surface_area_and_volume :
  let c := Cone.mk 96 (25/32)
  (surface_area c = 3584 * Real.pi) ∧ (volume c = 25088 * Real.pi) := by
  sorry

end cone_surface_area_and_volume_l2154_215454


namespace anne_solo_cleaning_time_l2154_215413

/-- Represents the time it takes Anne to clean the house alone -/
def anne_solo_time : ℝ := 12

/-- Represents Bruce's cleaning rate in houses per hour -/
noncomputable def bruce_rate : ℝ := sorry

/-- Represents Anne's cleaning rate in houses per hour -/
noncomputable def anne_rate : ℝ := sorry

/-- Bruce and Anne can clean the house in 4 hours together -/
axiom together_time : bruce_rate + anne_rate = 1 / 4

/-- If Anne's speed were doubled, they could clean the house in 3 hours -/
axiom double_anne_time : bruce_rate + 2 * anne_rate = 1 / 3

theorem anne_solo_cleaning_time : 
  1 / anne_rate = anne_solo_time :=
sorry

end anne_solo_cleaning_time_l2154_215413


namespace carls_weight_l2154_215442

theorem carls_weight (al ben carl ed : ℕ) 
  (h1 : al = ben + 25)
  (h2 : ben + 16 = carl)
  (h3 : ed = 146)
  (h4 : al = ed + 38) :
  carl = 175 := by
  sorry

end carls_weight_l2154_215442


namespace quadratic_factorization_l2154_215444

theorem quadratic_factorization (k : ℝ) : 
  (∀ x, x^2 - 8*x + 3 = 0 ↔ (x - 4)^2 = k) → k = 13 := by
  sorry

end quadratic_factorization_l2154_215444


namespace scientific_notation_of_248000_l2154_215471

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_248000 :
  toScientificNotation 248000 = ScientificNotation.mk 2.48 5 (by norm_num) :=
sorry

end scientific_notation_of_248000_l2154_215471


namespace solution_values_l2154_215456

def A : Set ℝ := {-1, 1}

def B (a b : ℝ) : Set ℝ := {x | x^2 - 2*a*x + b = 0}

theorem solution_values (a b : ℝ) : 
  B a b ≠ ∅ → A ∪ B a b = A → 
  ((a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = 1) ∨ (a = 0 ∧ b = -1)) :=
by sorry

end solution_values_l2154_215456


namespace unique_integer_with_properties_l2154_215422

theorem unique_integer_with_properties : ∃! n : ℕ+, 
  (∃ k : ℕ, n = 18 * k) ∧ 
  (30 < Real.sqrt n.val) ∧ 
  (Real.sqrt n.val < 30.5) := by
  sorry

end unique_integer_with_properties_l2154_215422


namespace tangent_line_at_point_p_l2154_215478

/-- The circle equation -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x + m*y = 0

/-- Point P is on the circle -/
def point_on_circle (m : ℝ) : Prop :=
  circle_equation 1 1 m

/-- The tangent line equation -/
def tangent_line_equation (x y : ℝ) : Prop :=
  x - 2*y + 1 = 0

/-- Theorem: The equation of the tangent line at point P(1,1) on the given circle is x - 2y + 1 = 0 -/
theorem tangent_line_at_point_p :
  ∃ m : ℝ, point_on_circle m →
  ∀ x y : ℝ, (x = 1 ∧ y = 1) →
  tangent_line_equation x y :=
sorry

end tangent_line_at_point_p_l2154_215478


namespace largest_x_value_l2154_215408

theorem largest_x_value (x : ℝ) : 
  (((15 * x^2 - 40 * x + 18) / (4 * x - 3) + 6 * x = 7 * x - 2) → x ≤ 1) ∧
  (∃ x : ℝ, (15 * x^2 - 40 * x + 18) / (4 * x - 3) + 6 * x = 7 * x - 2 ∧ x = 1) :=
by sorry

end largest_x_value_l2154_215408


namespace factorization_x4_minus_5x2_plus_4_l2154_215459

theorem factorization_x4_minus_5x2_plus_4 (x : ℝ) :
  x^4 - 5*x^2 + 4 = (x + 1)*(x - 1)*(x + 2)*(x - 2) := by
  sorry

end factorization_x4_minus_5x2_plus_4_l2154_215459


namespace expression_lower_bound_l2154_215405

theorem expression_lower_bound (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b + c) + 1 / (b + c) + 1 / (c + a)) ≥ 16 / 3 := by
  sorry

end expression_lower_bound_l2154_215405


namespace unique_420_sequence_l2154_215431

-- Define the sum of consecutive integers
def sum_consecutive (n : ℕ) (k : ℕ) : ℕ := k * n + k * (k - 1) / 2

-- Define a predicate for valid sequences
def valid_sequence (n : ℕ) (k : ℕ) : Prop :=
  k ≥ 2 ∧ (k % 2 = 0 ∨ k = 3) ∧ sum_consecutive n k = 420

-- The main theorem
theorem unique_420_sequence :
  ∃! (n k : ℕ), valid_sequence n k :=
sorry

end unique_420_sequence_l2154_215431


namespace pencil_pen_cost_l2154_215453

theorem pencil_pen_cost (x y : ℚ) : 
  (7 * x + 6 * y = 46.8) → 
  (3 * x + 5 * y = 32.2) → 
  (x = 2.4 ∧ y = 5) := by
sorry

end pencil_pen_cost_l2154_215453


namespace complement_union_theorem_l2154_215497

def I : Set Nat := {0, 1, 2, 3}
def A : Set Nat := {0, 1, 2}
def B : Set Nat := {2, 3}

theorem complement_union_theorem :
  (I \ A) ∪ (I \ B) = {0, 1, 3} := by sorry

end complement_union_theorem_l2154_215497


namespace firefighter_net_sag_l2154_215479

/-- The net sag for a person jumping onto a firefighter rescue net -/
def net_sag (m₁ m₂ h₁ h₂ x₁ : ℝ) (x₂ : ℝ) : Prop :=
  m₁ > 0 ∧ m₂ > 0 ∧ h₁ > 0 ∧ h₂ > 0 ∧ x₁ > 0 ∧ x₂ > 0 ∧
  28 * x₂^2 - x₂ - 29 = 0

theorem firefighter_net_sag (m₁ m₂ h₁ h₂ x₁ : ℝ) (hm₁ : m₁ = 78.75) (hm₂ : m₂ = 45)
    (hh₁ : h₁ = 15) (hh₂ : h₂ = 29) (hx₁ : x₁ = 1) :
  ∃ x₂, net_sag m₁ m₂ h₁ h₂ x₁ x₂ :=
by sorry

end firefighter_net_sag_l2154_215479


namespace weight_loss_in_april_l2154_215411

/-- Given Michael's weight loss plan:
  * total_weight: Total weight Michael wants to lose
  * march_loss: Weight lost in March
  * may_loss: Weight to lose in May
  * april_loss: Weight lost in April

  This theorem proves that the weight lost in April is equal to
  the total weight minus the weight lost in March and the weight to lose in May. -/
theorem weight_loss_in_april 
  (total_weight march_loss may_loss april_loss : ℕ) : 
  april_loss = total_weight - march_loss - may_loss := by
  sorry

#check weight_loss_in_april

end weight_loss_in_april_l2154_215411


namespace sin_cos_tan_relation_l2154_215401

theorem sin_cos_tan_relation (A : Real) (q : Real) 
  (h1 : Real.sin A = 3/5)
  (h2 : Real.cos A / Real.tan A = q/15) :
  q = 16 := by
sorry

end sin_cos_tan_relation_l2154_215401


namespace union_equals_universal_l2154_215464

-- Define the universal set U
def U : Finset Nat := {2, 3, 4, 5, 6}

-- Define set M
def M : Finset Nat := {3, 4, 5}

-- Define set N
def N : Finset Nat := {2, 4, 6}

-- Theorem statement
theorem union_equals_universal : M ∪ N = U := by sorry

end union_equals_universal_l2154_215464


namespace parabola_vertex_l2154_215423

/-- The vertex of the parabola y = -2(x-2)^2 - 5 is at the point (2, -5) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -2 * (x - 2)^2 - 5 → (2, -5) = (x, y) := by
  sorry

end parabola_vertex_l2154_215423


namespace sum_of_digits_divisible_by_13_l2154_215427

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_divisible_by_13 (n : ℕ) : 
  ∃ k ∈ Finset.range 79, 13 ∣ sum_of_digits (n + k) := by sorry

end sum_of_digits_divisible_by_13_l2154_215427


namespace smallest_four_digit_congruence_l2154_215468

theorem smallest_four_digit_congruence :
  ∃ (n : ℕ), 
    (1000 ≤ n ∧ n < 10000) ∧ 
    (75 * n) % 450 = 225 ∧
    (∀ m, (1000 ≤ m ∧ m < 10000) → (75 * m) % 450 = 225 → n ≤ m) ∧
    n = 1005 :=
by sorry

end smallest_four_digit_congruence_l2154_215468


namespace units_digit_of_T_is_zero_l2154_215419

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def T : ℕ := (List.range 99).foldl (λ acc i => acc + factorial (i + 3)) 0

theorem units_digit_of_T_is_zero : T % 10 = 0 := by sorry

end units_digit_of_T_is_zero_l2154_215419


namespace f_properties_l2154_215406

noncomputable def f (x : ℝ) : ℝ := (Real.sin x - Real.cos x) * Real.sin (2 * x) / Real.sin x

theorem f_properties :
  (∀ x : ℝ, f x ≠ 0 → x ∉ {y | ∃ k : ℤ, y = k * Real.pi}) ∧
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧
  (∀ k : ℤ, ∀ x : ℝ, Real.pi / 4 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 2 + k * Real.pi → f x ≥ 0) ∧
  (∃ m : ℝ, m > 0 ∧ m = 3 * Real.pi / 8 ∧ ∀ x : ℝ, f (x + m) = f (-x + m)) :=
by sorry

end f_properties_l2154_215406


namespace exists_unique_box_l2154_215475

/-- Represents a rectangular box with square base -/
structure Box where
  x : ℝ  -- side length of square base
  h : ℝ  -- height of box

/-- Calculates the surface area of the box -/
def surfaceArea (b : Box) : ℝ := 2 * b.x^2 + 4 * b.x * b.h

/-- Calculates the volume of the box -/
def volume (b : Box) : ℝ := b.x^2 * b.h

/-- Theorem stating the existence of a box meeting the given conditions -/
theorem exists_unique_box :
  ∃! b : Box,
    b.h = 2 * b.x + 2 ∧
    surfaceArea b ≥ 150 ∧
    volume b = 100 ∧
    b.x > 0 ∧
    b.h > 0 := by
  sorry

end exists_unique_box_l2154_215475


namespace smallest_n_for_fraction_inequality_l2154_215418

theorem smallest_n_for_fraction_inequality : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℤ), 0 < m → m < 2001 → 
    ∃ (k : ℤ), (m : ℚ) / 2001 < (k : ℚ) / n ∧ (k : ℚ) / n < ((m + 1) : ℚ) / 2002) ∧
  (∀ (n' : ℕ), 0 < n' → n' < n → 
    ∃ (m : ℤ), 0 < m ∧ m < 2001 ∧
      ∀ (k : ℤ), ¬((m : ℚ) / 2001 < (k : ℚ) / n' ∧ (k : ℚ) / n' < ((m + 1) : ℚ) / 2002)) ∧
  n = 4003 :=
by sorry

end smallest_n_for_fraction_inequality_l2154_215418


namespace min_third_side_right_triangle_l2154_215443

theorem min_third_side_right_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (a = 8 ∧ b = 15) ∨ (a = 8 ∧ c = 15) ∨ (b = 8 ∧ c = 15) →
  a^2 + b^2 = c^2 →
  min a (min b c) ≥ Real.sqrt 161 :=
by sorry

end min_third_side_right_triangle_l2154_215443


namespace calculate_expression_l2154_215484

theorem calculate_expression : -2⁻¹ * (-8) - Real.sqrt 9 - abs (-4) = -3 := by
  sorry

end calculate_expression_l2154_215484


namespace largest_triangle_perimeter_l2154_215458

theorem largest_triangle_perimeter :
  ∀ (x : ℕ),
  (7 + 8 > x) →
  (7 + x > 8) →
  (8 + x > 7) →
  (∀ y : ℕ, (7 + 8 > y) → (7 + y > 8) → (8 + y > 7) → y ≤ x) →
  7 + 8 + x = 29 :=
by
  sorry

end largest_triangle_perimeter_l2154_215458


namespace smallest_three_digit_square_ends_identical_l2154_215404

/-- A function that returns true if a number is a three-digit number -/
def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

/-- A function that returns true if the square of a number ends with three identical non-zero digits -/
def squareEndsWithThreeIdenticalNonZeroDigits (n : ℕ) : Prop :=
  ∃ d : ℕ, d ≠ 0 ∧ d < 10 ∧ n^2 % 1000 = 111 * d

/-- Theorem stating that 462 is the smallest three-digit number whose square ends with three identical non-zero digits -/
theorem smallest_three_digit_square_ends_identical : 
  (isThreeDigit 462 ∧ 
   squareEndsWithThreeIdenticalNonZeroDigits 462 ∧ 
   ∀ n : ℕ, isThreeDigit n → squareEndsWithThreeIdenticalNonZeroDigits n → 462 ≤ n) :=
by sorry

end smallest_three_digit_square_ends_identical_l2154_215404


namespace initial_number_proof_l2154_215448

theorem initial_number_proof (N : ℕ) : 
  (∃ k : ℕ, N - 7 = 15 * k) ∧ 
  (∀ m : ℕ, m < 7 → ¬∃ j : ℕ, N - m = 15 * j) → 
  N = 22 := by
sorry

end initial_number_proof_l2154_215448


namespace problem_1_problem_2_l2154_215424

-- Problem 1
theorem problem_1 (a b : ℤ) (h1 : a = 6) (h2 : b = -1) :
  2*a + 3*b - 2*a*b - a - 4*b - a*b = 25 := by sorry

-- Problem 2
theorem problem_2 (m n : ℤ) (h1 : |m| = 3) (h2 : |n| = 2) (h3 : m < n) :
  m^2 + 2*m*n + n^2 = 1 := by sorry

end problem_1_problem_2_l2154_215424


namespace regression_line_not_necessarily_through_sample_point_l2154_215434

/-- Given a set of sample data points and a regression line, 
    prove that the line doesn't necessarily pass through any sample point. -/
theorem regression_line_not_necessarily_through_sample_point 
  (n : ℕ) 
  (x y : Fin n → ℝ) 
  (a b : ℝ) : 
  ¬ (∀ (ε : ℝ), ε > 0 → 
    ∃ (i : Fin n), |y i - (b * x i + a)| < ε) :=
sorry

end regression_line_not_necessarily_through_sample_point_l2154_215434


namespace special_sequence_existence_l2154_215488

theorem special_sequence_existence : ∃ (a : ℕ → ℕ),
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, ∃ y, a n + 1 = y^2) ∧
  (∀ n, ∃ x, 3 * a n + 1 = x^2) ∧
  (∀ n, ∃ z, a n * a (n + 1) = z^2) :=
sorry

end special_sequence_existence_l2154_215488


namespace complex_equation_solution_l2154_215474

theorem complex_equation_solution (z : ℂ) (h : z * (3 - I) = 1 - I) : z = 2/5 - 1/5 * I := by
  sorry

end complex_equation_solution_l2154_215474


namespace product_decreasing_inequality_l2154_215435

theorem product_decreasing_inequality 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g) 
  (h_deriv : ∀ x, (deriv f x) * g x + f x * (deriv g x) < 0) 
  {a b x : ℝ} 
  (h_interval : a < x ∧ x < b) : 
  f x * g x > f b * g b :=
sorry

end product_decreasing_inequality_l2154_215435


namespace original_number_is_perfect_square_l2154_215429

theorem original_number_is_perfect_square :
  ∃ (n : ℕ), n^2 = 1296 ∧ ∃ (m : ℕ), (1296 + 148) = m^2 := by
  sorry

end original_number_is_perfect_square_l2154_215429


namespace circle_tangent_to_y_axis_l2154_215496

-- Define the circle's equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 5)^2 = 1

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 5)

-- Define what it means for a circle to be tangent to the y-axis
def tangent_to_y_axis (equation : (ℝ → ℝ → Prop)) : Prop :=
  ∃ y : ℝ, equation 0 y ∧ ∀ x ≠ 0, ¬equation x y

-- Theorem statement
theorem circle_tangent_to_y_axis :
  tangent_to_y_axis circle_equation ∧
  ∀ x y : ℝ, circle_equation x y → (x - circle_center.1)^2 + (y - circle_center.2)^2 = 1 :=
sorry

end circle_tangent_to_y_axis_l2154_215496


namespace cost_per_friend_is_1650_l2154_215409

/-- The cost per friend when buying erasers and pencils -/
def cost_per_friend (eraser_count : ℕ) (eraser_cost : ℕ) (pencil_count : ℕ) (pencil_cost : ℕ) (friend_count : ℕ) : ℚ :=
  ((eraser_count * eraser_cost + pencil_count * pencil_cost) : ℚ) / friend_count

/-- Theorem: The cost per friend for the given scenario is 1650 won -/
theorem cost_per_friend_is_1650 :
  cost_per_friend 5 200 7 800 4 = 1650 := by
  sorry

end cost_per_friend_is_1650_l2154_215409


namespace order_relation_l2154_215482

noncomputable def a (e : ℝ) : ℝ := 5 * Real.log (2^e)
noncomputable def b (e : ℝ) : ℝ := 2 * Real.log (5^e)
def c : ℝ := 10

theorem order_relation (e : ℝ) (h : e > 0) : c > a e ∧ a e > b e := by
  sorry

end order_relation_l2154_215482


namespace line_segment_ratio_l2154_215463

/-- Given points A, B, C, D, and E on a line in that order, prove that AC:DE = 5:3 -/
theorem line_segment_ratio (A B C D E : ℝ) : 
  (B - A = 3) → 
  (C - B = 7) → 
  (D - C = 4) → 
  (E - A = 20) → 
  (A < B) → (B < C) → (C < D) → (D < E) →
  (C - A) / (E - D) = 5 / 3 := by
  sorry

#check line_segment_ratio

end line_segment_ratio_l2154_215463


namespace roots_power_set_difference_l2154_215407

/-- The roots of the polynomial (x^101 - 1) / (x - 1) -/
def roots : Fin 100 → ℂ := sorry

/-- The set S of powers of roots -/
def S : Set ℂ := sorry

/-- The maximum number of unique values in S -/
def M : ℕ := sorry

/-- The minimum number of unique values in S -/
def N : ℕ := sorry

/-- The difference between the maximum and minimum number of unique values in S is 99 -/
theorem roots_power_set_difference : M - N = 99 := by sorry

end roots_power_set_difference_l2154_215407


namespace contest_paths_count_l2154_215445

/-- Represents the grid structure for the word "CONTEST" --/
inductive ContestGrid
| C : ContestGrid
| O : ContestGrid → ContestGrid
| N : ContestGrid → ContestGrid
| T : ContestGrid → ContestGrid
| E : ContestGrid → ContestGrid
| S : ContestGrid → ContestGrid

/-- Counts the number of paths to form "CONTEST" in the given grid --/
def countContestPaths (grid : ContestGrid) : ℕ :=
  match grid with
  | ContestGrid.C => 1
  | ContestGrid.O g => 2 * countContestPaths g
  | ContestGrid.N g => 2 * countContestPaths g
  | ContestGrid.T g => 2 * countContestPaths g
  | ContestGrid.E g => 2 * countContestPaths g
  | ContestGrid.S g => 2 * countContestPaths g

/-- The contest grid structure --/
def contestGrid : ContestGrid :=
  ContestGrid.S (ContestGrid.E (ContestGrid.T (ContestGrid.N (ContestGrid.O (ContestGrid.C)))))

theorem contest_paths_count :
  countContestPaths contestGrid = 127 :=
sorry

end contest_paths_count_l2154_215445


namespace prob_red_ball_l2154_215452

/-- The probability of drawing a red ball from a bag containing 1 red ball and 2 yellow balls is 1/3. -/
theorem prob_red_ball (num_red : ℕ) (num_yellow : ℕ) (h1 : num_red = 1) (h2 : num_yellow = 2) :
  (num_red : ℚ) / (num_red + num_yellow) = 1 / 3 := by
  sorry

end prob_red_ball_l2154_215452


namespace second_number_value_l2154_215495

theorem second_number_value (A B C : ℚ) : 
  A + B + C = 98 → 
  A / B = 2 / 3 → 
  B / C = 5 / 8 → 
  B = 30 := by
sorry

end second_number_value_l2154_215495


namespace special_circle_equation_l2154_215450

/-- A circle with center (a, b) and radius r satisfying specific conditions -/
structure SpecialCircle where
  a : ℝ
  b : ℝ
  r : ℝ
  center_on_line : a - 3 * b = 0
  tangent_to_y_axis : r = |a|
  chord_length : ((a - b) ^ 2 / 2) + 7 = r ^ 2

/-- The equation of a circle given its center (a, b) and radius r -/
def circle_equation (c : SpecialCircle) (x y : ℝ) : Prop :=
  (x - c.a) ^ 2 + (y - c.b) ^ 2 = c.r ^ 2

/-- The main theorem stating that a SpecialCircle satisfies one of two specific equations -/
theorem special_circle_equation (c : SpecialCircle) :
  (∀ x y, circle_equation c x y ↔ (x - 3) ^ 2 + (y - 1) ^ 2 = 9) ∨
  (∀ x y, circle_equation c x y ↔ (x + 3) ^ 2 + (y + 1) ^ 2 = 9) := by
  sorry

end special_circle_equation_l2154_215450


namespace sin_239_deg_l2154_215430

theorem sin_239_deg (a : ℝ) (h : Real.cos (31 * π / 180) = a) : 
  Real.sin (239 * π / 180) = -a := by
  sorry

end sin_239_deg_l2154_215430


namespace quad_pair_sum_l2154_215476

/-- Two distinct quadratic polynomials with specific properties -/
structure QuadraticPair where
  f : ℝ → ℝ
  g : ℝ → ℝ
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  hf : f = fun x ↦ x^2 + p*x + q
  hg : g = fun x ↦ x^2 + r*x + s
  distinct : f ≠ g
  vertex_root : g (-p/2) = 0 ∧ f (-r/2) = 0
  same_min : ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₁, f x₁ = m) ∧ (∀ x, g x ≥ m) ∧ (∃ x₂, g x₂ = m)
  intersection : f 50 = -200 ∧ g 50 = -200

/-- The sum of coefficients p and r is -200 -/
theorem quad_pair_sum (qp : QuadraticPair) : qp.p + qp.r = -200 := by
  sorry

end quad_pair_sum_l2154_215476


namespace immigrants_calculation_l2154_215470

/-- The number of people born in the country last year -/
def people_born : ℕ := 90171

/-- The total number of new people who began living in the country last year -/
def total_new_people : ℕ := 106491

/-- The number of people who immigrated to the country last year -/
def immigrants : ℕ := total_new_people - people_born

theorem immigrants_calculation :
  immigrants = 16320 := by
  sorry

end immigrants_calculation_l2154_215470


namespace calculation_problems_l2154_215438

theorem calculation_problems :
  ((-2 : ℤ) + 5 - abs (-8) + (-5) = -10) ∧
  ((-2 : ℤ)^2 * 5 - (-2)^3 / 4 = 22) := by
  sorry

end calculation_problems_l2154_215438


namespace parabola_b_value_l2154_215415

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- The y-intercept of a parabola -/
def yIntercept (p : Parabola) : ℝ := p.c

theorem parabola_b_value (p : Parabola) (h k : ℝ) :
  h > 0 ∧ k > 0 ∧
  vertex p = (h, k) ∧
  yIntercept p = -k ∧
  k = 2 * h →
  p.b = 8 := by sorry

end parabola_b_value_l2154_215415


namespace mashed_potatoes_suggestion_l2154_215487

theorem mashed_potatoes_suggestion (bacon_count : ℕ) (difference : ℕ) : 
  bacon_count = 394 → 
  difference = 63 → 
  bacon_count + difference = 457 :=
by
  sorry

end mashed_potatoes_suggestion_l2154_215487


namespace base_b_square_l2154_215460

theorem base_b_square (b : ℕ) : 
  (b + 5)^2 = 4*b^2 + 3*b + 6 → b = 8 := by
  sorry

end base_b_square_l2154_215460


namespace rectangle_length_calculation_l2154_215437

theorem rectangle_length_calculation (rectangle_width square_side : ℝ) :
  rectangle_width = 300 ∧ 
  square_side = 700 ∧ 
  (4 * square_side) = 2 * (2 * (rectangle_width + rectangle_length)) →
  rectangle_length = 400 :=
by
  sorry

#check rectangle_length_calculation

end rectangle_length_calculation_l2154_215437


namespace max_product_sum_2004_l2154_215490

theorem max_product_sum_2004 :
  (∃ (a b : ℤ), a + b = 2004 ∧ a * b = 1004004) ∧
  (∀ (x y : ℤ), x + y = 2004 → x * y ≤ 1004004) := by
  sorry

end max_product_sum_2004_l2154_215490


namespace shaded_area_of_square_pattern_l2154_215499

/-- Given a square with side length a, this theorem proves that the area of the shaded region
    formed by connecting vertices to midpoints of opposite sides in a pattern is (3/5) * a^2. -/
theorem shaded_area_of_square_pattern (a : ℝ) (h : a > 0) : ℝ :=
  let square_area := a^2
  let shaded_area := (3/5) * square_area
  shaded_area

#check shaded_area_of_square_pattern

end shaded_area_of_square_pattern_l2154_215499


namespace jacks_card_collection_l2154_215473

theorem jacks_card_collection :
  ∀ (football_cards baseball_cards total_cards : ℕ),
    baseball_cards = 3 * football_cards + 5 →
    baseball_cards = 95 →
    total_cards = baseball_cards + football_cards →
    total_cards = 125 := by
  sorry

end jacks_card_collection_l2154_215473
