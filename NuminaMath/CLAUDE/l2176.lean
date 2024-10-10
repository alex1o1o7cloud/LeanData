import Mathlib

namespace complex_unit_circle_representation_l2176_217668

theorem complex_unit_circle_representation (z : ℂ) (h1 : Complex.abs z = 1) (h2 : z ≠ -1) :
  ∃ t : ℝ, z = (1 + Complex.I * t) / (1 - Complex.I * t) := by
  sorry

end complex_unit_circle_representation_l2176_217668


namespace original_number_proof_l2176_217671

theorem original_number_proof (N : ℝ) (x : ℝ) : 
  (N * 1.2 = 480) → 
  (480 * 0.85 * x^2 = 5*x^3 + 24*x - 50) → 
  N = 400 := by
sorry

end original_number_proof_l2176_217671


namespace perpendicular_to_plane_implies_parallel_l2176_217683

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_plane_implies_parallel
  (m n : Line) (α : Plane) 
  (hm : m ≠ n)
  (hα : perpendicular m α)
  (hβ : perpendicular n α) :
  parallel m n :=
sorry

end perpendicular_to_plane_implies_parallel_l2176_217683


namespace quadratic_two_distinct_roots_l2176_217626

/-- 
Given a quadratic equation x^2 - 4x - a = 0, prove that it has two distinct real roots
if and only if a > -4.
-/
theorem quadratic_two_distinct_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x - a = 0 ∧ y^2 - 4*y - a = 0) ↔ a > -4 := by
  sorry

end quadratic_two_distinct_roots_l2176_217626


namespace min_value_quadratic_l2176_217657

theorem min_value_quadratic (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2) :
  ∃ m : ℝ, (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 2 := by
  sorry

end min_value_quadratic_l2176_217657


namespace green_peaches_count_l2176_217686

/-- Given a number of baskets, red peaches per basket, and total peaches,
    calculates the number of green peaches per basket. -/
def green_peaches_per_basket (num_baskets : ℕ) (red_per_basket : ℕ) (total_peaches : ℕ) : ℕ :=
  (total_peaches - num_baskets * red_per_basket) / num_baskets

/-- Proves that there are 4 green peaches in each basket given the problem conditions. -/
theorem green_peaches_count :
  green_peaches_per_basket 15 19 345 = 4 := by
  sorry

#eval green_peaches_per_basket 15 19 345

end green_peaches_count_l2176_217686


namespace oil_price_reduction_l2176_217658

/-- Proves that a 25% reduction in oil price allows purchasing 5 kg more oil for Rs. 1100 --/
theorem oil_price_reduction (original_price : ℝ) : 
  (original_price * 0.75 = 55) →  -- Reduced price is 55
  (1100 / 55 - 1100 / original_price = 5) := by
sorry

end oil_price_reduction_l2176_217658


namespace max_sum_of_product_l2176_217696

theorem max_sum_of_product (a b : ℤ) : 
  a ≠ b → a * b = -132 → a ≤ b → (∀ x y : ℤ, x ≠ y → x * y = -132 → x ≤ y → a + b ≥ x + y) → a + b = -1 :=
by sorry

end max_sum_of_product_l2176_217696


namespace james_oranges_l2176_217610

theorem james_oranges (pieces_per_orange : ℕ) (num_people : ℕ) (calories_per_orange : ℕ) (calories_per_person : ℕ) :
  pieces_per_orange = 8 →
  num_people = 4 →
  calories_per_orange = 80 →
  calories_per_person = 100 →
  (calories_per_person * num_people) / calories_per_orange * pieces_per_orange / pieces_per_orange = 5 :=
by sorry

end james_oranges_l2176_217610


namespace product_105_95_l2176_217699

theorem product_105_95 : 105 * 95 = 9975 := by
  sorry

end product_105_95_l2176_217699


namespace special_rectangle_exists_l2176_217624

/-- A rectangle with the given properties --/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  perimeter_equals_area : 2 * (length + width) = length * width
  width_is_length_minus_three : width = length - 3

/-- The theorem stating that a rectangle with length 6 and width 3 satisfies the conditions --/
theorem special_rectangle_exists : ∃ (r : SpecialRectangle), r.length = 6 ∧ r.width = 3 := by
  sorry

end special_rectangle_exists_l2176_217624


namespace minimum_agreement_for_budget_constraint_l2176_217642

/-- Represents a parliament budget allocation problem -/
structure ParliamentBudget where
  members : ℕ
  items : ℕ
  limit : ℝ

/-- Defines the minimum number of members required for agreement -/
def min_agreement (pb : ParliamentBudget) : ℕ := pb.members - pb.items + 1

/-- Theorem stating the minimum agreement required for the given problem -/
theorem minimum_agreement_for_budget_constraint 
  (pb : ParliamentBudget) 
  (h_members : pb.members = 2000) 
  (h_items : pb.items = 200) :
  min_agreement pb = 1991 := by
  sorry

#eval min_agreement { members := 2000, items := 200, limit := 0 }

end minimum_agreement_for_budget_constraint_l2176_217642


namespace max_students_distribution_l2176_217667

def number_of_pens : ℕ := 2010
def number_of_pencils : ℕ := 1050

theorem max_students_distribution (n : ℕ) :
  (n ∣ number_of_pens) ∧ 
  (n ∣ number_of_pencils) ∧ 
  (∀ m : ℕ, m > n → ¬(m ∣ number_of_pens) ∨ ¬(m ∣ number_of_pencils)) →
  n = 30 :=
by sorry

end max_students_distribution_l2176_217667


namespace shooting_probabilities_l2176_217640

/-- Let A and B be two individuals conducting 3 shooting trials each.
    The probability of A hitting the target in each trial is 1/2.
    The probability of B hitting the target in each trial is 2/3. -/
theorem shooting_probabilities 
  (probability_A : ℝ) 
  (probability_B : ℝ) 
  (h_prob_A : probability_A = 1/2) 
  (h_prob_B : probability_B = 2/3) :
  /- The probability that A hits the target exactly 2 times -/
  (3 : ℝ) * probability_A^2 * (1 - probability_A) = 3/8 ∧ 
  /- The probability that B hits the target at least 2 times -/
  (3 : ℝ) * probability_B^2 * (1 - probability_B) + probability_B^3 = 20/27 ∧ 
  /- The probability that B hits the target exactly 2 more times than A -/
  (3 : ℝ) * probability_B^2 * (1 - probability_B) * (1 - probability_A)^3 + 
  probability_B^3 * (3 : ℝ) * probability_A * (1 - probability_A)^2 = 1/6 :=
by sorry


end shooting_probabilities_l2176_217640


namespace drawing_red_is_certain_l2176_217675

/-- Represents a ball in the box -/
inductive Ball
  | Red

/-- Represents the box containing balls -/
def Box := List Ball

/-- Defines a certain event -/
def CertainEvent (event : Prop) : Prop :=
  ∀ (outcome : Prop), event = outcome

/-- The box contains exactly two red balls -/
def TwoRedBalls (box : Box) : Prop :=
  box = [Ball.Red, Ball.Red]

/-- Drawing a ball from the box -/
def DrawBall (box : Box) : Ball :=
  match box with
  | [] => Ball.Red  -- Default case, should not occur
  | (b :: _) => b

/-- The main theorem: Drawing a red ball from a box with two red balls is a certain event -/
theorem drawing_red_is_certain (box : Box) (h : TwoRedBalls box) :
  CertainEvent (DrawBall box = Ball.Red) := by
  sorry

end drawing_red_is_certain_l2176_217675


namespace sum_of_roots_l2176_217645

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - a^2 + a - 5 = 0)
  (hb : b^3 - 2*b^2 + 2*b + 4 = 0) : 
  a + b = 1 := by
sorry

end sum_of_roots_l2176_217645


namespace linear_function_property_l2176_217634

/-- A linear function is a function f : ℝ → ℝ such that f(ax + by) = af(x) + bf(y) for all x, y ∈ ℝ and a, b ∈ ℝ -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∀ (x y a b : ℝ), f (a * x + b * y) = a * f x + b * f y

theorem linear_function_property (f : ℝ → ℝ) (h_linear : LinearFunction f) 
    (h_given : f 10 - f 4 = 20) : f 16 - f 4 = 40 := by
  sorry

end linear_function_property_l2176_217634


namespace length_of_A_l2176_217672

-- Define the points
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (0, 14)
def C : ℝ × ℝ := (3, 6)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define the intersection of line segments
def intersect (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (1 - t) • p + t • r = q

-- Main theorem
theorem length_of_A'B' (A' B' : ℝ × ℝ) :
  line_y_eq_x A' →
  line_y_eq_x B' →
  intersect A A' C →
  intersect B B' C →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 90 * Real.sqrt 2 / 11 := by
  sorry

end length_of_A_l2176_217672


namespace team_win_percentage_l2176_217643

theorem team_win_percentage (total_games : ℕ) (wins_first_100 : ℕ) 
  (h1 : total_games ≥ 100)
  (h2 : wins_first_100 ≤ 100)
  (h3 : (wins_first_100 : ℝ) / 100 + (0.5 * (total_games - 100) : ℝ) / total_games = 0.7) :
  wins_first_100 = 70 := by
sorry

end team_win_percentage_l2176_217643


namespace percentage_calculation_l2176_217650

theorem percentage_calculation (whole : ℝ) (part : ℝ) (percentage : ℝ) 
  (h1 : whole = 800)
  (h2 : part = 200)
  (h3 : percentage = (part / whole) * 100) :
  percentage = 25 := by
  sorry

end percentage_calculation_l2176_217650


namespace cube_edge_length_l2176_217649

theorem cube_edge_length (V : ℝ) (s : ℝ) (h : V = 7) (h1 : V = s^3) :
  s = (7 : ℝ)^(1/3) := by sorry

end cube_edge_length_l2176_217649


namespace plane_equation_correct_l2176_217669

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by parametric equations -/
structure Line3D where
  x : ℝ → ℝ
  y : ℝ → ℝ
  z : ℝ → ℝ

/-- A plane in 3D space defined by the equation Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (plane : Plane) (point : Point3D) : Prop :=
  plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D = 0

/-- Check if a line is contained in a plane -/
def lineInPlane (plane : Plane) (line : Line3D) : Prop :=
  ∀ t, pointOnPlane plane ⟨line.x t, line.y t, line.z t⟩

/-- The given point that the plane passes through -/
def givenPoint : Point3D :=
  ⟨1, 4, -5⟩

/-- The given line that the plane contains -/
def givenLine : Line3D :=
  ⟨λ t => 4 * t + 2, λ t => -t + 1, λ t => 5 * t - 3⟩

/-- The plane we want to prove -/
def solutionPlane : Plane :=
  ⟨2, 7, 6, -66⟩

theorem plane_equation_correct :
  pointOnPlane solutionPlane givenPoint ∧
  lineInPlane solutionPlane givenLine ∧
  solutionPlane.A > 0 ∧
  Nat.gcd (Nat.gcd (Int.natAbs solutionPlane.A) (Int.natAbs solutionPlane.B))
          (Nat.gcd (Int.natAbs solutionPlane.C) (Int.natAbs solutionPlane.D)) = 1 :=
by sorry

end plane_equation_correct_l2176_217669


namespace sphere_radius_from_shadow_and_pole_l2176_217603

/-- The radius of a sphere given its shadow and a reference pole -/
theorem sphere_radius_from_shadow_and_pole 
  (sphere_shadow : ℝ) 
  (pole_height : ℝ) 
  (pole_shadow : ℝ) 
  (h_sphere_shadow : sphere_shadow = 15)
  (h_pole_height : pole_height = 1.5)
  (h_pole_shadow : pole_shadow = 3) :
  let tan_theta := pole_height / pole_shadow
  let radius := sphere_shadow * tan_theta
  radius = 7.5 := by sorry

end sphere_radius_from_shadow_and_pole_l2176_217603


namespace solution_set_correct_l2176_217690

def solution_set := {x : ℝ | 0 < x ∧ x < 1}

theorem solution_set_correct : 
  ∀ x : ℝ, x ∈ solution_set ↔ (x * (x + 2) > 0 ∧ |x| < 1) :=
by sorry

end solution_set_correct_l2176_217690


namespace article_price_fraction_l2176_217607

/-- Proves that selling an article at 2/3 of its original price results in a 10% loss,
    given that the original price has a 35% markup from the cost price. -/
theorem article_price_fraction (original_price cost_price : ℝ) :
  original_price = cost_price * (1 + 35 / 100) →
  original_price * (2 / 3) = cost_price * (1 - 10 / 100) := by
  sorry

end article_price_fraction_l2176_217607


namespace geometric_sequence_ratio_geometric_arithmetic_relation_l2176_217629

/-- A geometric sequence with first term a and common ratio q -/
def geometricSequence (a q : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => q * geometricSequence a q n

theorem geometric_sequence_ratio (a q : ℝ) (h : q ≠ 0) (h₁ : a ≠ 0) :
  ∀ n : ℕ, geometricSequence a q (n + 1) / geometricSequence a q n = q := by sorry

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => arithmeticSequence a d n + d

theorem geometric_arithmetic_relation (a q : ℝ) (h : q ≠ 0) (h₁ : a ≠ 0) :
  (∃ d : ℝ, arithmeticSequence 1 d 0 = 1 ∧
            arithmeticSequence 1 d 1 = geometricSequence a q 1 ∧
            arithmeticSequence 1 d 2 = geometricSequence a q 2 - 1) →
  (geometricSequence a q 2 + geometricSequence a q 3) / (geometricSequence a q 4 + geometricSequence a q 5) = 1/4 := by sorry

end geometric_sequence_ratio_geometric_arithmetic_relation_l2176_217629


namespace isosceles_triangles_count_l2176_217644

/-- The number of ways to choose three vertices of a regular nonagon to form an isosceles triangle -/
def isosceles_triangles_in_nonagon : ℕ := 33

/-- A regular nonagon has 9 sides -/
def nonagon_sides : ℕ := 9

/-- The number of ways to choose 2 vertices from a nonagon -/
def choose_two_vertices : ℕ := (nonagon_sides * (nonagon_sides - 1)) / 2

/-- The number of equilateral triangles in a nonagon -/
def equilateral_triangles : ℕ := 3

theorem isosceles_triangles_count :
  isosceles_triangles_in_nonagon = choose_two_vertices - equilateral_triangles :=
by sorry

end isosceles_triangles_count_l2176_217644


namespace correct_calculation_l2176_217646

theorem correct_calculation (x : ℕ) (h : x + 12 = 48) : x + 22 = 58 := by
  sorry

end correct_calculation_l2176_217646


namespace matrix_inverse_proof_l2176_217635

theorem matrix_inverse_proof : 
  let M : Matrix (Fin 3) (Fin 3) ℚ := ![![7/29, 5/29, 0], ![3/29, 2/29, 0], ![0, 0, 1]]
  let A : Matrix (Fin 3) (Fin 3) ℚ := ![![2, -5, 0], ![-3, 7, 0], ![0, 0, 1]]
  M * A = (1 : Matrix (Fin 3) (Fin 3) ℚ) := by sorry

end matrix_inverse_proof_l2176_217635


namespace peggy_stamp_count_l2176_217693

/-- The number of stamps Peggy has -/
def peggy_stamps : ℕ := 75

/-- The number of stamps Ernie has -/
def ernie_stamps : ℕ := 3 * peggy_stamps

/-- The number of stamps Bert has -/
def bert_stamps : ℕ := 4 * ernie_stamps

theorem peggy_stamp_count : 
  bert_stamps = peggy_stamps + 825 ∧ 
  ernie_stamps = 3 * peggy_stamps ∧ 
  bert_stamps = 4 * ernie_stamps →
  peggy_stamps = 75 := by sorry

end peggy_stamp_count_l2176_217693


namespace closest_point_on_line_l2176_217605

def v (t : ℝ) : ℝ × ℝ × ℝ := (3 + 8*t, -2 + 6*t, -4 - 2*t)

def a : ℝ × ℝ × ℝ := (5, 7, 3)

def direction : ℝ × ℝ × ℝ := (8, 6, -2)

theorem closest_point_on_line (t : ℝ) : 
  (t = 7/13) ↔ 
  (∀ s : ℝ, ‖v t - a‖ ≤ ‖v s - a‖) :=
sorry

end closest_point_on_line_l2176_217605


namespace farmer_bean_seedlings_l2176_217695

/-- Represents the farmer's planting scenario -/
structure FarmPlanting where
  bean_seedlings_per_row : ℕ
  pumpkin_seeds : ℕ
  pumpkin_seeds_per_row : ℕ
  radishes : ℕ
  radishes_per_row : ℕ
  rows_per_bed : ℕ
  total_beds : ℕ

/-- Calculates the total number of bean seedlings -/
def total_bean_seedlings (f : FarmPlanting) : ℕ :=
  let total_rows := f.total_beds * f.rows_per_bed
  let pumpkin_rows := f.pumpkin_seeds / f.pumpkin_seeds_per_row
  let radish_rows := f.radishes / f.radishes_per_row
  let bean_rows := total_rows - pumpkin_rows - radish_rows
  bean_rows * f.bean_seedlings_per_row

/-- Theorem stating that the farmer has 64 bean seedlings -/
theorem farmer_bean_seedlings :
  ∀ (f : FarmPlanting),
  f.bean_seedlings_per_row = 8 →
  f.pumpkin_seeds = 84 →
  f.pumpkin_seeds_per_row = 7 →
  f.radishes = 48 →
  f.radishes_per_row = 6 →
  f.rows_per_bed = 2 →
  f.total_beds = 14 →
  total_bean_seedlings f = 64 := by
  sorry

end farmer_bean_seedlings_l2176_217695


namespace sum_a_d_equals_one_l2176_217692

theorem sum_a_d_equals_one (a b c d : ℝ) 
  (h1 : a + b = 4) 
  (h2 : b + c = 5) 
  (h3 : c + d = 3) : 
  a + d = 1 := by
sorry

end sum_a_d_equals_one_l2176_217692


namespace arithmetic_sequence_properties_l2176_217698

/-- An arithmetic sequence with specific terms -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  third_term : a 3 = 9
  ninth_term : a 9 = 3

/-- The general term of the arithmetic sequence -/
def generalTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  15 - 3/2 * (n - 1)

/-- The term from which the sequence becomes negative -/
def negativeStartTerm (seq : ArithmeticSequence) : ℕ := 13

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = generalTerm seq n) ∧
  (∀ n, n ≥ negativeStartTerm seq → seq.a n < 0) ∧
  (∀ n, n < negativeStartTerm seq → seq.a n ≥ 0) :=
sorry

end arithmetic_sequence_properties_l2176_217698


namespace h_closed_form_l2176_217654

def h : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * h n + 2 * n

theorem h_closed_form (n : ℕ) : h n = 2^n + n^2 - n := by
  sorry

end h_closed_form_l2176_217654


namespace paula_four_hops_l2176_217621

def hop_distance (goal : ℚ) (remaining : ℚ) : ℚ :=
  (1 / 4) * remaining

def remaining_distance (goal : ℚ) (hopped : ℚ) : ℚ :=
  goal - hopped

theorem paula_four_hops :
  let goal : ℚ := 2
  let hop1 := hop_distance goal goal
  let hop2 := hop_distance goal (remaining_distance goal hop1)
  let hop3 := hop_distance goal (remaining_distance goal (hop1 + hop2))
  let hop4 := hop_distance goal (remaining_distance goal (hop1 + hop2 + hop3))
  hop1 + hop2 + hop3 + hop4 = 175 / 128 := by
  sorry

end paula_four_hops_l2176_217621


namespace charlies_garden_min_cost_l2176_217620

/-- Represents a rectangular region in the garden -/
structure Region where
  length : ℝ
  width : ℝ

/-- Calculates the area of a region -/
def area (r : Region) : ℝ := r.length * r.width

/-- Represents the cost of fertilizer per square meter for each vegetable type -/
structure FertilizerCost where
  lettuce : ℝ
  spinach : ℝ
  carrots : ℝ
  beans : ℝ
  tomatoes : ℝ

/-- The given garden layout -/
def garden_layout : List Region := [
  ⟨3, 1⟩,  -- Upper left
  ⟨4, 2⟩,  -- Lower right
  ⟨6, 2⟩,  -- Upper right
  ⟨2, 3⟩,  -- Middle center
  ⟨5, 4⟩   -- Bottom left
]

/-- The given fertilizer costs -/
def fertilizer_costs : FertilizerCost :=
  { lettuce := 2
  , spinach := 2.5
  , carrots := 3
  , beans := 3.5
  , tomatoes := 4
  }

/-- Calculates the minimum cost of fertilizers for the garden -/
def min_fertilizer_cost (layout : List Region) (costs : FertilizerCost) : ℝ :=
  sorry  -- Proof implementation goes here

/-- Theorem stating that the minimum fertilizer cost for Charlie's garden is $127 -/
theorem charlies_garden_min_cost :
  min_fertilizer_cost garden_layout fertilizer_costs = 127 := by
  sorry  -- Proof goes here

end charlies_garden_min_cost_l2176_217620


namespace canada_animal_population_l2176_217622

/-- Represents the population of different species in Canada -/
structure CanadaPopulation where
  humans : ℚ
  moose : ℚ
  beavers : ℚ
  caribou : ℚ
  wolves : ℚ
  grizzly_bears : ℚ

/-- The relationships between species in Canada -/
def population_relationships (p : CanadaPopulation) : Prop :=
  p.beavers = 2 * p.moose ∧
  p.humans = 19 * p.beavers ∧
  3 * p.caribou = 2 * p.moose ∧
  p.wolves = 4 * p.caribou ∧
  3 * p.grizzly_bears = p.wolves

/-- The theorem stating the combined population of animals given the human population -/
theorem canada_animal_population 
  (p : CanadaPopulation) 
  (h : population_relationships p) 
  (humans_pop : p.humans = 38) : 
  p.moose + p.beavers + p.caribou + p.wolves + p.grizzly_bears = 12.5 := by
  sorry

end canada_animal_population_l2176_217622


namespace max_value_M_l2176_217638

theorem max_value_M (x y z w : ℝ) (h : x + y + z + w = 1) :
  let M := x*w + 2*y*w + 3*x*y + 3*z*w + 4*x*z + 5*y*z
  ∃ (x₀ y₀ z₀ w₀ : ℝ), x₀ + y₀ + z₀ + w₀ = 1 ∧
    (∀ x y z w, x + y + z + w = 1 →
      x*w + 2*y*w + 3*x*y + 3*z*w + 4*x*z + 5*y*z ≤
      x₀*w₀ + 2*y₀*w₀ + 3*x₀*y₀ + 3*z₀*w₀ + 4*x₀*z₀ + 5*y₀*z₀) ∧
    x₀*w₀ + 2*y₀*w₀ + 3*x₀*y₀ + 3*z₀*w₀ + 4*x₀*z₀ + 5*y₀*z₀ = 3/2 :=
by sorry

end max_value_M_l2176_217638


namespace sum_of_values_l2176_217659

/-- A discrete random variable with two possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  prob₁ : ℝ
  prob₂ : ℝ
  h₁ : x₁ < x₂
  h₂ : prob₁ = (1 : ℝ) / 2
  h₃ : prob₂ = (1 : ℝ) / 2
  h₄ : prob₁ + prob₂ = 1

/-- Expected value of the discrete random variable -/
def expectation (X : DiscreteRV) : ℝ :=
  X.x₁ * X.prob₁ + X.x₂ * X.prob₂

/-- Variance of the discrete random variable -/
def variance (X : DiscreteRV) : ℝ :=
  (X.x₁ - expectation X)^2 * X.prob₁ + (X.x₂ - expectation X)^2 * X.prob₂

theorem sum_of_values (X : DiscreteRV) 
    (h_exp : expectation X = 2) 
    (h_var : variance X = (1 : ℝ) / 2) : 
  X.x₁ + X.x₂ = 3 := by
  sorry

end sum_of_values_l2176_217659


namespace symmetric_circle_equation_l2176_217687

/-- Given two circles C₁ and C₂, where C₁ has equation (x+1)²+(y-1)²=1 and C₂ is symmetric to C₁
    with respect to the line x-y-1=0, prove that the equation of C₂ is (x-2)²+(y+2)²=1 -/
theorem symmetric_circle_equation (x y : ℝ) :
  let C₁ : ℝ → ℝ → Prop := λ x y => (x + 1)^2 + (y - 1)^2 = 1
  let symmetry_line : ℝ → ℝ → Prop := λ x y => x - y - 1 = 0
  let C₂ : ℝ → ℝ → Prop := λ x y => (x - 2)^2 + (y + 2)^2 = 1
  (∀ x y, C₁ x y ↔ (x + 1)^2 + (y - 1)^2 = 1) →
  (∀ x₁ y₁ x₂ y₂, C₁ x₁ y₁ → C₂ x₂ y₂ → 
    ∃ x_sym y_sym, symmetry_line x_sym y_sym ∧
    (x₂ - x_sym = x_sym - x₁) ∧ (y₂ - y_sym = y_sym - y₁)) →
  (∀ x y, C₂ x y ↔ (x - 2)^2 + (y + 2)^2 = 1) :=
by sorry

end symmetric_circle_equation_l2176_217687


namespace rectangle_on_circle_l2176_217666

theorem rectangle_on_circle (R : ℝ) (x y : ℝ) :
  x^2 + y^2 = R^2 →
  x * y = (12 * R / 35) * (x + y) →
  ((x = 3 * R / 5 ∧ y = 4 * R / 5) ∨ (x = 4 * R / 5 ∧ y = 3 * R / 5)) :=
by sorry

end rectangle_on_circle_l2176_217666


namespace total_bonus_calculation_l2176_217641

def senior_bonus : ℕ := 1900
def junior_bonus : ℕ := 3100

theorem total_bonus_calculation : senior_bonus + junior_bonus = 5000 := by
  sorry

end total_bonus_calculation_l2176_217641


namespace total_portfolios_l2176_217689

theorem total_portfolios (num_students : ℕ) (portfolios_per_student : ℕ) 
  (h1 : num_students = 15)
  (h2 : portfolios_per_student = 8) :
  num_students * portfolios_per_student = 120 := by
  sorry

end total_portfolios_l2176_217689


namespace triangle_property_triangle_area_l2176_217628

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  angleSum : A + B + C = π
  positiveSides : 0 < a ∧ 0 < b ∧ 0 < c
  positiveAngles : 0 < A ∧ 0 < B ∧ 0 < C

theorem triangle_property (t : Triangle) 
  (h : Real.sin t.B * (Real.tan t.A + Real.tan t.C) = Real.tan t.A * Real.tan t.C) :
  t.b^2 = t.a * t.c :=
sorry

theorem triangle_area (t : Triangle) (h1 : t.a = 2 * t.c) (h2 : t.a = 2) :
  (1/2 : ℝ) * t.a * t.c * Real.sin t.B = Real.sqrt 7 / 4 :=
sorry

end triangle_property_triangle_area_l2176_217628


namespace annual_rent_per_square_foot_l2176_217631

/-- Calculates the annual rent per square foot for a shop given its dimensions and monthly rent -/
theorem annual_rent_per_square_foot
  (length : ℝ) (width : ℝ) (monthly_rent : ℝ)
  (h1 : length = 18)
  (h2 : width = 20)
  (h3 : monthly_rent = 3600) :
  monthly_rent * 12 / (length * width) = 120 := by
  sorry

end annual_rent_per_square_foot_l2176_217631


namespace roots_of_quadratic_l2176_217681

theorem roots_of_quadratic (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := by
  sorry

end roots_of_quadratic_l2176_217681


namespace street_lights_per_side_l2176_217619

/-- The number of neighborhoods in the town -/
def num_neighborhoods : ℕ := 10

/-- The number of roads in each neighborhood -/
def roads_per_neighborhood : ℕ := 4

/-- The total number of street lights in the town -/
def total_street_lights : ℕ := 20000

/-- The number of street lights on each opposite side of a road -/
def lights_per_side : ℚ := total_street_lights / (2 * num_neighborhoods * roads_per_neighborhood)

theorem street_lights_per_side :
  lights_per_side = 250 :=
sorry

end street_lights_per_side_l2176_217619


namespace pizza_distribution_l2176_217655

/-- Given the number of brothers, slices in small and large pizzas, and the number of each type of pizza ordered, 
    calculate the number of slices each brother can eat. -/
def slices_per_brother (num_brothers : ℕ) (slices_small : ℕ) (slices_large : ℕ) 
                       (num_small : ℕ) (num_large : ℕ) : ℕ :=
  (num_small * slices_small + num_large * slices_large) / num_brothers

/-- Theorem stating that under the given conditions, each brother can eat 12 slices of pizza. -/
theorem pizza_distribution :
  slices_per_brother 3 8 14 1 2 = 12 := by
  sorry

end pizza_distribution_l2176_217655


namespace sphere_surface_area_l2176_217630

theorem sphere_surface_area (R : ℝ) : 
  R > 0 → 
  (∃ (x : ℝ), x > 0 ∧ x < R ∧ 
    (∀ (y : ℝ), y > 0 → y < R → 
      2 * π * x^2 * (2 * Real.sqrt (R^2 - x^2)) ≥ 2 * π * y^2 * (2 * Real.sqrt (R^2 - y^2)))) →
  2 * π * R * (2 * Real.sqrt (R^2 - (R * Real.sqrt 6 / 3)^2)) = 16 * Real.sqrt 2 * π →
  4 * π * R^2 = 48 * π :=
by sorry

end sphere_surface_area_l2176_217630


namespace function_properties_l2176_217664

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
    (h1 : ∀ x, f (10 + x) = f (10 - x))
    (h2 : ∀ x, f (20 - x) = -f (20 + x)) :
    is_odd f ∧ is_periodic f 40 := by
  sorry

end function_properties_l2176_217664


namespace race_start_calculation_l2176_217651

/-- Given a kilometer race where runner A can give runner B a 200 meters start,
    and runner B can give runner C a 250 meters start,
    prove that runner A can give runner C a 400 meters start. -/
theorem race_start_calculation (Va Vb Vc : ℝ) 
  (h1 : Va / Vb = 1000 / 800)
  (h2 : Vb / Vc = 1000 / 750) :
  Va / Vc = 1000 / 600 :=
by sorry

end race_start_calculation_l2176_217651


namespace number_of_pickers_l2176_217697

/-- Given information about grape harvesting, calculate the number of pickers --/
theorem number_of_pickers (drums_per_day : ℕ) (total_drums : ℕ) (total_days : ℕ) 
  (h1 : drums_per_day = 108)
  (h2 : total_drums = 6264)
  (h3 : total_days = 58)
  (h4 : total_drums = drums_per_day * total_days) :
  total_drums / drums_per_day = 58 := by
  sorry

end number_of_pickers_l2176_217697


namespace circle_M_equation_l2176_217665

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the axis of symmetry of the parabola
def axis_of_symmetry (y : ℝ) : Prop := y = -1

-- Define the circle ⊙M
def circle_M (x y t : ℝ) : Prop := (x - t)^2 + (y - t^2/4)^2 = t^2

-- Define the tangency condition to y-axis
def tangent_to_y_axis (t : ℝ) : Prop := t = 2 ∨ t = -2

-- Define the tangency condition to axis of symmetry
def tangent_to_axis_of_symmetry (t : ℝ) : Prop := |1 + t^2/4| = |t|

-- Theorem statement
theorem circle_M_equation (x y t : ℝ) :
  parabola x y →
  axis_of_symmetry (-1) →
  circle_M x y t →
  tangent_to_y_axis t →
  tangent_to_axis_of_symmetry t →
  ∃ (sign : ℝ), sign = 1 ∨ sign = -1 ∧ x^2 + y^2 + sign*4*x - 2*y + 1 = 0 :=
sorry

end circle_M_equation_l2176_217665


namespace polynomial_root_properties_l2176_217674

def P (x p : ℂ) : ℂ := x^4 + 3*x^3 + 3*x + p

theorem polynomial_root_properties (p : ℝ) (x₁ : ℂ) 
  (h1 : Complex.abs x₁ = 1)
  (h2 : 2 * Complex.re x₁ = (Real.sqrt 17 - 3) / 2)
  (h3 : P x₁ p = 0) :
  p = -1 - 3 * x₁^3 - 3 * x₁ ∧
  x₁ = Complex.mk ((Real.sqrt 17 - 3) / 4) (Real.sqrt ((3 * Real.sqrt 17 - 5) / 8)) ∧
  ∀ n : ℕ+, x₁^(n : ℕ) ≠ 1 := by
sorry

end polynomial_root_properties_l2176_217674


namespace blue_button_probability_l2176_217608

/-- Represents a jar containing buttons of different colors. -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- The probability of selecting a blue button from a jar. -/
def blueProb (j : Jar) : ℚ :=
  j.blue / (j.red + j.blue)

/-- The initial state of Jar C. -/
def jarC : Jar := { red := 6, blue := 10 }

/-- The number of buttons transferred from Jar C to Jar D. -/
def transferred : ℕ := 4

/-- Jar C after the transfer. -/
def jarCAfter : Jar := { red := jarC.red - transferred / 2, blue := jarC.blue - transferred / 2 }

/-- Jar D after the transfer. -/
def jarD : Jar := { red := transferred / 2, blue := transferred / 2 }

/-- Theorem stating the probability of selecting blue buttons from both jars. -/
theorem blue_button_probability : 
  blueProb jarCAfter * blueProb jarD = 1 / 3 :=
sorry

end blue_button_probability_l2176_217608


namespace fourth_power_sum_l2176_217636

theorem fourth_power_sum (a b c : ℝ) 
  (sum_eq : a + b + c = 2)
  (sum_squares_eq : a^2 + b^2 + c^2 = 5)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 15.5 := by
  sorry

end fourth_power_sum_l2176_217636


namespace parallel_lines_a_equals_two_l2176_217632

/-- Two lines in the form Ax + By + C = 0 are parallel if and only if their slopes are equal -/
def parallel (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  A1 / B1 = A2 / B2

/-- Two lines in the form Ax + By + C = 0 are coincident if they have the same slope and y-intercept -/
def coincident (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  A1 / B1 = A2 / B2 ∧ C1 / B1 = C2 / B2

theorem parallel_lines_a_equals_two (a : ℝ) :
  parallel a (a + 2) 2 1 a (-2) ∧
  ¬coincident a (a + 2) 2 1 a (-2) →
  a = 2 := by
  sorry

end parallel_lines_a_equals_two_l2176_217632


namespace xy_equals_one_l2176_217684

theorem xy_equals_one (x y : ℝ) (h : x + y = 1/x + 1/y) (h_neq : x + y ≠ 0) : x * y = 1 := by
  sorry

end xy_equals_one_l2176_217684


namespace least_xy_value_l2176_217613

theorem least_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 8) : 
  (∀ a b : ℕ+, (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 8 → x * y ≤ a * b) ∧ x * y = 96 :=
sorry

end least_xy_value_l2176_217613


namespace wallpaper_overlap_theorem_l2176_217606

/-- The combined area of three walls with overlapping wallpaper -/
def combined_area (two_layer_area : ℝ) (three_layer_area : ℝ) (total_covered_area : ℝ) : ℝ :=
  total_covered_area + two_layer_area + 2 * three_layer_area

/-- Theorem stating the combined area of three walls with given overlapping conditions -/
theorem wallpaper_overlap_theorem (two_layer_area : ℝ) (three_layer_area : ℝ) (total_covered_area : ℝ)
    (h1 : two_layer_area = 40)
    (h2 : three_layer_area = 40)
    (h3 : total_covered_area = 180) :
    combined_area two_layer_area three_layer_area total_covered_area = 300 := by
  sorry

end wallpaper_overlap_theorem_l2176_217606


namespace matrix_power_four_l2176_217617

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 1, 1]

theorem matrix_power_four :
  A ^ 4 = !![(-14), -6; 3, (-17)] := by sorry

end matrix_power_four_l2176_217617


namespace triple_product_sum_two_l2176_217612

theorem triple_product_sum_two (x y z : ℝ) :
  (x * y + z = 2) ∧ (y * z + x = 2) ∧ (z * x + y = 2) →
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -2 ∧ y = -2 ∧ z = -2)) :=
by sorry

end triple_product_sum_two_l2176_217612


namespace units_digit_G_500_l2176_217600

-- Define the sequence G_n
def G (n : ℕ) : ℕ := 2^(3^n) + 1

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_G_500 : unitsDigit (G 500) = 3 := by
  sorry

end units_digit_G_500_l2176_217600


namespace sum_interior_angles_octagon_l2176_217688

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Formula for the sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: The sum of interior angles of an octagon is 1080° -/
theorem sum_interior_angles_octagon :
  sum_interior_angles octagon_sides = 1080 := by sorry

end sum_interior_angles_octagon_l2176_217688


namespace sculptures_not_on_display_sculptures_not_on_display_proof_l2176_217677

/-- Represents the total number of art pieces in the gallery -/
def total_art_pieces : ℕ := 3150

/-- Represents the fraction of art pieces on display -/
def fraction_on_display : ℚ := 1/3

/-- Represents the fraction of sculptures among displayed pieces -/
def fraction_sculptures_displayed : ℚ := 1/6

/-- Represents the fraction of paintings among pieces not on display -/
def fraction_paintings_not_displayed : ℚ := 1/3

/-- Represents that some sculptures are not on display -/
axiom some_sculptures_not_displayed : ∃ (n : ℕ), n > 0 ∧ n ≤ total_art_pieces

theorem sculptures_not_on_display : ℕ :=
  1400

theorem sculptures_not_on_display_proof : sculptures_not_on_display = 1400 := by
  sorry

end sculptures_not_on_display_sculptures_not_on_display_proof_l2176_217677


namespace quarter_circle_square_perimeter_l2176_217673

/-- The perimeter of a region bounded by quarter-circular arcs constructed at each corner of a square with sides measuring 4/π is equal to 8. -/
theorem quarter_circle_square_perimeter :
  let square_side : ℝ := 4 / Real.pi
  let quarter_circle_radius : ℝ := square_side
  let quarter_circle_count : ℕ := 4
  let region_perimeter : ℝ := quarter_circle_count * (Real.pi * quarter_circle_radius / 2)
  region_perimeter = 8 := by
  sorry

end quarter_circle_square_perimeter_l2176_217673


namespace lacustrine_glacial_monoliths_l2176_217602

-- Define the total number of monoliths
def total_monoliths : ℕ := 98

-- Define the probability of a monolith being sand
def prob_sand : ℚ := 1/7

-- Define the probability of a monolith being marine loam
def prob_marine_loam : ℚ := 9/14

-- Theorem statement
theorem lacustrine_glacial_monoliths :
  let sand_monoliths := (prob_sand * total_monoliths : ℚ).num
  let loam_monoliths := total_monoliths - sand_monoliths
  let marine_loam_monoliths := (prob_marine_loam * loam_monoliths : ℚ).num
  let lacustrine_glacial_loam_monoliths := loam_monoliths - marine_loam_monoliths
  sand_monoliths + lacustrine_glacial_loam_monoliths = 44 := by
  sorry

end lacustrine_glacial_monoliths_l2176_217602


namespace range_of_m_l2176_217685

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x > (1/2) → x^2 - m*x + 4 > 0) → m < 4 :=
by sorry

end range_of_m_l2176_217685


namespace condition_relationship_l2176_217661

theorem condition_relationship (a b : ℝ) : 
  (∀ a b, a > 0 ∧ b > 0 → a * b > 0) ∧  -- A is necessary for B
  (∃ a b, a * b > 0 ∧ ¬(a > 0 ∧ b > 0)) -- A is not sufficient for B
  := by sorry

end condition_relationship_l2176_217661


namespace intersection_implies_a_value_l2176_217647

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 1, a^2 + 4}

theorem intersection_implies_a_value :
  ∀ a : ℝ, A ∩ B a = {3} → a = 2 := by
  sorry

end intersection_implies_a_value_l2176_217647


namespace geometric_sequence_a5_l2176_217653

/-- A geometric sequence with common ratio 2 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem geometric_sequence_a5 (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 3 * a 11 = 16) :
  a 5 = 1 := by
  sorry

end geometric_sequence_a5_l2176_217653


namespace incircle_radius_not_less_than_one_l2176_217678

/-- Triangle ABC with sides a, b, c and incircle radius r -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ

/-- The theorem stating that the incircle radius of triangle ABC with BC = 3 and AC = 4 is not less than 1 -/
theorem incircle_radius_not_less_than_one (t : Triangle) (h1 : t.b = 3) (h2 : t.c = 4) : 
  t.r ≥ 1 := by
  sorry


end incircle_radius_not_less_than_one_l2176_217678


namespace logarithm_sum_simplification_l2176_217663

theorem logarithm_sum_simplification :
  (1 / (Real.log 3 / Real.log 12 + 1) +
   1 / (Real.log 2 / Real.log 8 + 1) +
   1 / (Real.log 9 / Real.log 18 + 1)) = 1 := by
  sorry

end logarithm_sum_simplification_l2176_217663


namespace greatest_real_part_of_cube_l2176_217676

theorem greatest_real_part_of_cube (z₁ z₂ z₃ z₄ z₅ : ℂ) : 
  z₁ = -1 ∧ 
  z₂ = -Real.sqrt 2 + I ∧ 
  z₃ = -1 + Real.sqrt 3 * I ∧ 
  z₄ = 2 * I ∧ 
  z₅ = -1 - Real.sqrt 3 * I → 
  (z₄^3).re ≥ (z₁^3).re ∧ 
  (z₄^3).re ≥ (z₂^3).re ∧ 
  (z₄^3).re ≥ (z₃^3).re ∧ 
  (z₄^3).re ≥ (z₅^3).re :=
by sorry

end greatest_real_part_of_cube_l2176_217676


namespace evaluate_expression_l2176_217691

theorem evaluate_expression : 48^3 + 3*(48^2)*4 + 3*48*(4^2) + 4^3 = 140608 := by
  sorry

end evaluate_expression_l2176_217691


namespace intersection_M_complement_N_l2176_217615

/-- The set M of real numbers less than 3 -/
def M : Set ℝ := {x : ℝ | x < 3}

/-- The set N of real numbers less than 1 -/
def N : Set ℝ := {x : ℝ | x < 1}

/-- Theorem stating that the intersection of M and the complement of N in ℝ
    is equal to the set of real numbers x where 1 ≤ x < 3 -/
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

end intersection_M_complement_N_l2176_217615


namespace total_seashells_eq_sum_l2176_217633

/-- The number of seashells Sam found on the beach -/
def total_seashells : ℕ := sorry

/-- The number of seashells Sam gave to Joan -/
def seashells_given : ℕ := 18

/-- The number of seashells Sam has left -/
def seashells_left : ℕ := 17

/-- Theorem stating that the total number of seashells is the sum of those given away and those left -/
theorem total_seashells_eq_sum : 
  total_seashells = seashells_given + seashells_left := by sorry

end total_seashells_eq_sum_l2176_217633


namespace closed_set_properties_l2176_217609

-- Define a closed set
def is_closed_set (M : Set Int) : Prop :=
  ∀ a b : Int, a ∈ M ∧ b ∈ M → (a + b) ∈ M ∧ (a - b) ∈ M

-- Define the set M = {-4, -2, 0, 2, 4}
def M : Set Int := {-4, -2, 0, 2, 4}

-- Define the set of positive integers
def positive_integers : Set Int := {n : Int | n > 0}

-- Define the set M = {n | n = 3k, k ∈ Z}
def M_3k : Set Int := {n : Int | ∃ k : Int, n = 3 * k}

theorem closed_set_properties :
  (¬ is_closed_set M) ∧
  (¬ is_closed_set positive_integers) ∧
  (is_closed_set M_3k) ∧
  (∃ A₁ A₂ : Set Int, is_closed_set A₁ ∧ is_closed_set A₂ ∧ ¬ is_closed_set (A₁ ∪ A₂)) :=
sorry

end closed_set_properties_l2176_217609


namespace function_root_iff_a_range_l2176_217627

/-- The function f(x) = 2ax - a + 3 has a root in (-1, 1) if and only if a ∈ (-∞, -3) ∪ (1, +∞) -/
theorem function_root_iff_a_range (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioo (-1) 1 ∧ 2 * a * x₀ - a + 3 = 0) ↔ 
  a ∈ Set.Iic (-3) ∪ Set.Ioi 1 :=
sorry

end function_root_iff_a_range_l2176_217627


namespace edward_sold_games_l2176_217618

theorem edward_sold_games (initial_games : ℕ) (boxes : ℕ) (games_per_box : ℕ) 
  (h1 : initial_games = 35)
  (h2 : boxes = 2)
  (h3 : games_per_box = 8) :
  initial_games - (boxes * games_per_box) = 19 := by
  sorry

end edward_sold_games_l2176_217618


namespace train_crossing_time_l2176_217616

-- Define constants
def train_length : Real := 120
def train_speed_kmh : Real := 70
def bridge_length : Real := 150

-- Define the theorem
theorem train_crossing_time :
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let crossing_time := total_distance / train_speed_ms
  ∃ ε > 0, abs (crossing_time - 13.89) < ε :=
by
  sorry

end train_crossing_time_l2176_217616


namespace triangle_with_consecutive_sides_and_area_l2176_217604

theorem triangle_with_consecutive_sides_and_area :
  ∃ (a b c S : ℕ), 
    (a + 1 = b) ∧ 
    (b + 1 = c) ∧ 
    (c + 1 = S) ∧
    (a = 3) ∧ (b = 4) ∧ (c = 5) ∧ (S = 6) ∧
    (2 * S = a * b) :=
by sorry

end triangle_with_consecutive_sides_and_area_l2176_217604


namespace expression_evaluation_l2176_217614

theorem expression_evaluation : (20 * 3 + 10) / (5 + 3) = 8.75 := by
  sorry

end expression_evaluation_l2176_217614


namespace sum_w_y_l2176_217601

theorem sum_w_y (w x y z : ℚ) 
  (eq1 : w * x * y = 10)
  (eq2 : w * y * z = 5)
  (eq3 : w * x * z = 45)
  (eq4 : x * y * z = 12) :
  w + y = 19/6 := by
  sorry

end sum_w_y_l2176_217601


namespace chord_length_l2176_217660

/-- The length of the chord cut by a circle on a line --/
theorem chord_length (r : ℝ) (a b c : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}
  let line := {(x, y) : ℝ × ℝ | ∃ t, x = a - b*t ∧ y = c + b*t}
  let chord := circle ∩ line
  r = 2 ∧ a = 2 ∧ b = 1/2 ∧ c = -1 →
  ∃ p q : ℝ × ℝ, p ∈ chord ∧ q ∈ chord ∧ p ≠ q ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = Real.sqrt 14 :=
by sorry

end chord_length_l2176_217660


namespace max_fraction_value_l2176_217652

def is_odd_integer (y : ℝ) : Prop := ∃ (k : ℤ), y = 2 * k + 1

theorem max_fraction_value (x y : ℝ) 
  (hx : -5 ≤ x ∧ x ≤ -3) 
  (hy : 3 ≤ y ∧ y ≤ 5) 
  (hy_odd : is_odd_integer y) : 
  (∀ z, -5 ≤ z ∧ z ≤ -3 → ∀ w, 3 ≤ w ∧ w ≤ 5 → is_odd_integer w → (x + y) / x ≥ (z + w) / z) ∧ 
  (x + y) / x ≤ 0.4 := by
  sorry

end max_fraction_value_l2176_217652


namespace square_circle_area_ratio_l2176_217648

theorem square_circle_area_ratio (s : ℝ) (h : s > 0) :
  let r : ℝ := s / 2
  let square_area : ℝ := s ^ 2
  let circle_area : ℝ := π * r ^ 2
  square_area / circle_area = 4 / π :=
by sorry

end square_circle_area_ratio_l2176_217648


namespace relay_race_total_time_l2176_217625

/-- Represents the data for each athlete in the relay race -/
structure AthleteData where
  distance : ℕ
  time : ℕ

/-- Calculates the total time of the relay race given the data for each athlete -/
def relay_race_time (athletes : Vector AthleteData 8) : ℕ :=
  athletes.toList.map (·.time) |>.sum

theorem relay_race_total_time : ∃ (athletes : Vector AthleteData 8),
  (athletes.get 0).distance = 200 ∧ (athletes.get 0).time = 55 ∧
  (athletes.get 1).distance = 300 ∧ (athletes.get 1).time = (athletes.get 0).time + 10 ∧
  (athletes.get 2).distance = 250 ∧ (athletes.get 2).time = (athletes.get 1).time - 15 ∧
  (athletes.get 3).distance = 150 ∧ (athletes.get 3).time = (athletes.get 0).time - 25 ∧
  (athletes.get 4).distance = 400 ∧ (athletes.get 4).time = 80 ∧
  (athletes.get 5).distance = 350 ∧ (athletes.get 5).time = (athletes.get 4).time - 20 ∧
  (athletes.get 6).distance = 275 ∧ (athletes.get 6).time = 70 ∧
  (athletes.get 7).distance = 225 ∧ (athletes.get 7).time = (athletes.get 6).time - 5 ∧
  relay_race_time athletes = 475 := by
  sorry

end relay_race_total_time_l2176_217625


namespace min_n_is_correct_l2176_217680

/-- The minimum positive integer n for which the expansion of (x^2 + 1/(3x^3))^n contains a constant term -/
def min_n : ℕ := 5

/-- Predicate to check if the expansion contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ (r : ℕ), 2 * n = 5 * r

theorem min_n_is_correct :
  (∀ m : ℕ, m > 0 ∧ m < min_n → ¬(has_constant_term m)) ∧
  has_constant_term min_n :=
sorry

end min_n_is_correct_l2176_217680


namespace triangle_side_length_l2176_217611

theorem triangle_side_length (a b : ℝ) (C : ℝ) (S : ℝ) :
  a = 1 →
  C = π / 4 →
  S = 2 * a →
  S = 1 / 2 * a * b * Real.sin C →
  b = 8 * Real.sqrt 2 := by
  sorry

end triangle_side_length_l2176_217611


namespace area_fraction_to_CD_l2176_217639

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- AB is parallel to CD and AB < CD
  AB : ℝ
  CD : ℝ
  h_parallel : AB < CD
  -- ∠BAD = 45° and ∠ABC = 135°
  angle_BAD : ℝ
  angle_ABC : ℝ
  h_angles : angle_BAD = π/4 ∧ angle_ABC = 3*π/4
  -- AD = BC = 100 m
  AD : ℝ
  BC : ℝ
  h_sides : AD = 100 ∧ BC = 100
  -- AB = 80 m
  h_AB : AB = 80
  -- CD > 100 m
  h_CD : CD > 100

/-- The fraction of the area closer to CD than to AB is approximately 3/4 -/
theorem area_fraction_to_CD (t : Trapezoid) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |((t.CD - t.AB) * t.AD / (2 * (t.AB + t.CD) * t.AD)) - 3/4| < ε :=
sorry

end area_fraction_to_CD_l2176_217639


namespace circle_intersection_radius_range_l2176_217637

/-- Given two intersecting circles O and M in a Cartesian coordinate system,
    where O has center (0, 0) and radius r (r > 0),
    and M has center (3, -4) and radius 2,
    the range of possible values for r is 3 < r < 7. -/
theorem circle_intersection_radius_range (r : ℝ) : 
  r > 0 ∧ 
  (∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ (x - 3)^2 + (y + 4)^2 = 4) →
  3 < r ∧ r < 7 := by
  sorry

#check circle_intersection_radius_range

end circle_intersection_radius_range_l2176_217637


namespace value_of_M_l2176_217682

theorem value_of_M : ∃ M : ℝ, (0.12 * M = 0.60 * 1500) ∧ (M = 7500) := by
  sorry

end value_of_M_l2176_217682


namespace calculator_problem_l2176_217662

theorem calculator_problem (x : ℝ) (hx : x ≠ 0) :
  (1 / (1/x - 1)) - 1 = -0.75 → x = 0.2 := by
  sorry

end calculator_problem_l2176_217662


namespace arccos_one_half_equals_pi_third_l2176_217656

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end arccos_one_half_equals_pi_third_l2176_217656


namespace evaluate_expression_l2176_217679

theorem evaluate_expression : (3^2)^2 - (2^3)^3 = -431 := by
  sorry

end evaluate_expression_l2176_217679


namespace correct_sampling_methods_l2176_217694

/-- Represents different sampling methods --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a survey with its characteristics --/
structure Survey where
  population : ℕ
  sample_size : ℕ
  has_groups : Bool

/-- Determines the most appropriate sampling method for a given survey --/
def best_sampling_method (s : Survey) : SamplingMethod :=
  if s.has_groups then SamplingMethod.Stratified
  else if s.population > 100 then SamplingMethod.Systematic
  else SamplingMethod.SimpleRandom

/-- The yogurt box survey --/
def yogurt_survey : Survey :=
  { population := 10, sample_size := 3, has_groups := false }

/-- The audience survey --/
def audience_survey : Survey :=
  { population := 1280, sample_size := 32, has_groups := false }

/-- The school staff survey --/
def staff_survey : Survey :=
  { population := 160, sample_size := 20, has_groups := true }

theorem correct_sampling_methods :
  best_sampling_method yogurt_survey = SamplingMethod.SimpleRandom ∧
  best_sampling_method audience_survey = SamplingMethod.Systematic ∧
  best_sampling_method staff_survey = SamplingMethod.Stratified :=
sorry

end correct_sampling_methods_l2176_217694


namespace prove_initial_person_count_l2176_217670

/-- The number of persons initially in a group, given that:
    - The average weight increases by 2.5 kg when a new person replaces someone
    - The replaced person weighs 70 kg
    - The new person weighs 90 kg
-/
def initialPersonCount : ℕ := 8

theorem prove_initial_person_count :
  let averageWeightIncrease : ℚ := 2.5
  let replacedPersonWeight : ℕ := 70
  let newPersonWeight : ℕ := 90
  averageWeightIncrease * initialPersonCount = newPersonWeight - replacedPersonWeight :=
by sorry

#eval initialPersonCount

end prove_initial_person_count_l2176_217670


namespace angle_330_equals_negative_30_l2176_217623

/-- Two angles have the same terminal side if they differ by a multiple of 360° --/
def same_terminal_side (α β : Real) : Prop :=
  ∃ k : Int, α = β + 360 * k

/-- The problem statement --/
theorem angle_330_equals_negative_30 :
  same_terminal_side 330 (-30) := by
  sorry

end angle_330_equals_negative_30_l2176_217623
