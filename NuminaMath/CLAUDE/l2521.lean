import Mathlib

namespace six_pieces_per_small_load_l2521_252197

/-- Given a total number of clothing pieces, the number of pieces in the first load,
    and the number of small loads, calculate the number of pieces in each small load. -/
def clothingPerSmallLoad (total : ℕ) (firstLoad : ℕ) (smallLoads : ℕ) : ℕ :=
  (total - firstLoad) / smallLoads

/-- Theorem stating that with 47 total pieces, 17 in the first load, and 5 small loads,
    each small load contains 6 pieces of clothing. -/
theorem six_pieces_per_small_load :
  clothingPerSmallLoad 47 17 5 = 6 := by
  sorry

end six_pieces_per_small_load_l2521_252197


namespace trig_identity_l2521_252104

theorem trig_identity (x y z : ℝ) : 
  Real.sin (x - y + z) * Real.cos y - Real.cos (x - y + z) * Real.sin y = Real.sin x := by
  sorry

end trig_identity_l2521_252104


namespace quadratic_equation_coefficients_l2521_252181

theorem quadratic_equation_coefficients 
  (a b c : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : (7 : ℝ) * ((7 : ℝ) * a + b) + c = 0) 
  (h3 : (-1 : ℝ) * ((-1 : ℝ) * a + b) + c = 0) :
  b = -6 ∧ c = -7 := by
sorry

end quadratic_equation_coefficients_l2521_252181


namespace series_convergence_l2521_252120

/-- The sum of the infinite series ∑(n=1 to ∞) [(n³+4n²+8n+8) / (3ⁿ·(n³+5))] converges to 1/2. -/
theorem series_convergence : 
  let f : ℕ → ℝ := λ n => (n^3 + 4*n^2 + 8*n + 8) / (3^n * (n^3 + 5))
  ∑' n, f n = 1/2 := by sorry

end series_convergence_l2521_252120


namespace square_areas_sum_l2521_252148

theorem square_areas_sum (a b c : ℕ) (ha : a = 2) (hb : b = 3) (hc : c = 6) :
  a^2 + b^2 + c^2 = 7^2 := by
  sorry

end square_areas_sum_l2521_252148


namespace set_as_interval_l2521_252143

def S : Set ℝ := {x : ℝ | -12 ≤ x ∧ x < 10 ∨ x > 11}

theorem set_as_interval : S = Set.Icc (-12) 10 ∪ Set.Ioi 11 := by sorry

end set_as_interval_l2521_252143


namespace carol_has_62_pennies_l2521_252147

/-- The number of pennies Alex currently has -/
def alex_pennies : ℕ := sorry

/-- The number of pennies Carol currently has -/
def carol_pennies : ℕ := sorry

/-- If Alex gives Carol two pennies, Carol will have four times as many pennies as Alex has -/
axiom condition1 : carol_pennies + 2 = 4 * (alex_pennies - 2)

/-- If Carol gives Alex two pennies, Carol will have three times as many pennies as Alex has -/
axiom condition2 : carol_pennies - 2 = 3 * (alex_pennies + 2)

/-- Carol has 62 pennies -/
theorem carol_has_62_pennies : carol_pennies = 62 := by sorry

end carol_has_62_pennies_l2521_252147


namespace conic_family_inscribed_in_square_l2521_252177

-- Define the square
def square : Set (ℝ × ℝ) :=
  {p | (p.1 = 3 ∨ p.1 = -3) ∧ p.2 ∈ [-3, 3] ∨
       (p.2 = 3 ∨ p.2 = -3) ∧ p.1 ∈ [-3, 3]}

-- Define the differential equation
def diff_eq (x y : ℝ) (dy_dx : ℝ) : Prop :=
  (9 - x^2) * dy_dx^2 = (9 - y^2)

-- Define a family of conics
def conic_family (C : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * (Real.cos C) * x * y = 9 * (Real.sin C)^2

-- State the theorem
theorem conic_family_inscribed_in_square :
  ∀ C : ℝ, ∃ p : ℝ × ℝ,
    p ∈ square ∧
    (∃ x y dy_dx : ℝ, diff_eq x y dy_dx ∧
      conic_family C x y ∧
      (x = p.1 ∧ y = p.2)) :=
sorry

end conic_family_inscribed_in_square_l2521_252177


namespace root_between_roots_l2521_252170

theorem root_between_roots (a b c r s : ℝ) 
  (hr : a * r^2 + b * r + c = 0)
  (hs : -a * s^2 + b * s + c = 0) :
  ∃ t : ℝ, (t > r ∧ t < s ∨ t > s ∧ t < r) ∧ a/2 * t^2 + b * t + c = 0 :=
by sorry

end root_between_roots_l2521_252170


namespace ascending_order_l2521_252150

theorem ascending_order (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 1) :
  b * c < a * c ∧ a * c < a * b ∧ a * b < a * b * c := by
  sorry

end ascending_order_l2521_252150


namespace circle_existence_theorem_l2521_252103

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if a point is inside a circle -/
def isInside (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

/-- Check if a point is on a circle -/
def isOn (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Check if three points are collinear -/
def areCollinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- The main theorem -/
theorem circle_existence_theorem (n : ℕ) (points : Fin n → Point) 
    (h1 : n ≥ 3) 
    (h2 : ∃ p1 p2 p3 : Fin n, ¬areCollinear (points p1) (points p2) (points p3)) :
    ∃ (c : Circle) (p1 p2 p3 : Fin n), 
      p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
      isOn (points p1) c ∧ isOn (points p2) c ∧ isOn (points p3) c ∧
      ∀ (p : Fin n), p ≠ p1 → p ≠ p2 → p ≠ p3 → ¬isInside (points p) c :=
  sorry

end circle_existence_theorem_l2521_252103


namespace select_students_theorem_l2521_252111

/-- Represents the number of students in each category for a group -/
structure GroupComposition :=
  (male : ℕ)
  (female : ℕ)

/-- Calculates the number of ways to select students from two groups with exactly one female -/
def selectStudentsWithOneFemale (groupA groupB : GroupComposition) : ℕ :=
  let selectOneFromA := groupA.female * groupA.male * (groupB.male.choose 2)
  let selectOneFromB := groupB.female * groupB.male * (groupA.male.choose 2)
  selectOneFromA + selectOneFromB

/-- The main theorem stating the number of ways to select students -/
theorem select_students_theorem (groupA groupB : GroupComposition) : 
  groupA.male = 5 → groupA.female = 3 → groupB.male = 6 → groupB.female = 2 →
  selectStudentsWithOneFemale groupA groupB = 345 := by
  sorry

#eval selectStudentsWithOneFemale ⟨5, 3⟩ ⟨6, 2⟩

end select_students_theorem_l2521_252111


namespace complex_power_four_l2521_252172

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_four (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end complex_power_four_l2521_252172


namespace boys_neither_happy_nor_sad_l2521_252101

/-- Given a group of children with various emotional states, prove the number of boys who are neither happy nor sad. -/
theorem boys_neither_happy_nor_sad
  (total_children : ℕ)
  (happy_children : ℕ)
  (sad_children : ℕ)
  (neither_children : ℕ)
  (total_boys : ℕ)
  (total_girls : ℕ)
  (happy_boys : ℕ)
  (sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neither_children = 20)
  (h5 : total_boys = 22)
  (h6 : total_girls = 38)
  (h7 : happy_boys = 6)
  (h8 : sad_girls = 4)
  (h9 : total_children = happy_children + sad_children + neither_children)
  (h10 : total_children = total_boys + total_girls) :
  total_boys - happy_boys - (sad_children - sad_girls) = 10 :=
by sorry

end boys_neither_happy_nor_sad_l2521_252101


namespace fraction_zero_implies_x_negative_three_l2521_252179

theorem fraction_zero_implies_x_negative_three (x : ℝ) :
  (x^2 - 9) / (x - 3) = 0 ∧ x - 3 ≠ 0 → x = -3 := by
  sorry

end fraction_zero_implies_x_negative_three_l2521_252179


namespace area_condition_implies_parallel_to_KL_l2521_252157

/-- A quadrilateral with non-parallel sides AB and CD -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (not_parallel : ¬ (B.1 - A.1) * (D.2 - C.2) = (B.2 - A.2) * (D.1 - C.1))

/-- The area of a triangle given by three points -/
noncomputable def triangleArea (P Q R : ℝ × ℝ) : ℝ := sorry

/-- The area of a quadrilateral -/
noncomputable def quadrilateralArea (q : Quadrilateral) : ℝ := sorry

/-- The intersection point of lines AB and CD -/
noncomputable def intersectionPoint (q : Quadrilateral) : ℝ × ℝ := sorry

/-- Point K on the extension of AB such that OK = AB -/
noncomputable def pointK (q : Quadrilateral) : ℝ × ℝ := sorry

/-- Point L on the extension of CD such that OL = CD -/
noncomputable def pointL (q : Quadrilateral) : ℝ × ℝ := sorry

/-- Check if three points are collinear -/
def collinear (P Q R : ℝ × ℝ) : Prop := sorry

/-- Check if a point is inside a quadrilateral -/
def isInside (X : ℝ × ℝ) (q : Quadrilateral) : Prop := sorry

/-- Check if two lines are parallel -/
def parallel (P Q R S : ℝ × ℝ) : Prop := sorry

theorem area_condition_implies_parallel_to_KL (q : Quadrilateral) (X : ℝ × ℝ) :
  isInside X q →
  triangleArea q.A q.B X + triangleArea q.C q.D X = (quadrilateralArea q) / 2 →
  ∃ P Q : ℝ × ℝ, collinear P Q X ∧ parallel P Q (pointK q) (pointL q) :=
sorry

end area_condition_implies_parallel_to_KL_l2521_252157


namespace unique_solution_implies_a_equals_four_l2521_252155

def A (a : ℝ) : Set ℝ := {x | a * x^2 + a * x + 1 = 0}

theorem unique_solution_implies_a_equals_four (a : ℝ) :
  (∃! x, x ∈ A a) → a = 4 := by
  sorry

end unique_solution_implies_a_equals_four_l2521_252155


namespace crayons_added_l2521_252151

theorem crayons_added (initial : ℕ) (final : ℕ) (added : ℕ) : 
  initial = 9 → final = 12 → added = final - initial → added = 3 := by sorry

end crayons_added_l2521_252151


namespace sara_apples_l2521_252161

theorem sara_apples (total : ℕ) (ali_factor : ℕ) (sara_apples : ℕ) 
  (h1 : total = 80)
  (h2 : ali_factor = 4)
  (h3 : total = sara_apples + ali_factor * sara_apples) :
  sara_apples = 16 := by
  sorry

end sara_apples_l2521_252161


namespace group_size_l2521_252184

theorem group_size (over_30 : ℕ) (prob_under_20 : ℚ) :
  over_30 = 90 →
  prob_under_20 = 7/16 →
  ∃ (total : ℕ),
    total = over_30 + (total - over_30) ∧
    (total - over_30) / total = prob_under_20 ∧
    total = 160 := by
  sorry

end group_size_l2521_252184


namespace square_of_binomial_constant_l2521_252186

theorem square_of_binomial_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) → 
  a = 25 := by
sorry

end square_of_binomial_constant_l2521_252186


namespace distance_between_vertices_l2521_252168

-- Define the equation
def equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + abs (y - 1) = 5

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 3)
def vertex2 : ℝ × ℝ := (0, -2)

-- Theorem statement
theorem distance_between_vertices :
  let (x1, y1) := vertex1
  let (x2, y2) := vertex2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 5 := by sorry

end distance_between_vertices_l2521_252168


namespace banana_cost_proof_l2521_252108

/-- The cost of Tony's purchase in dollars -/
def tony_cost : ℚ := 7

/-- The number of dozen apples Tony bought -/
def tony_apples : ℕ := 2

/-- The cost of Arnold's purchase in dollars -/
def arnold_cost : ℚ := 5

/-- The number of dozen apples Arnold bought -/
def arnold_apples : ℕ := 1

/-- The number of bunches of bananas each person bought -/
def bananas : ℕ := 1

/-- The cost of a bunch of bananas in dollars -/
def banana_cost : ℚ := 3

theorem banana_cost_proof :
  banana_cost = tony_cost - arnold_cost - (tony_apples - arnold_apples) * (tony_cost - arnold_cost) :=
by
  sorry

end banana_cost_proof_l2521_252108


namespace ladder_rungs_count_ladder_rungs_count_proof_l2521_252124

theorem ladder_rungs_count : ℕ → Prop :=
  fun n =>
    let middle_rung := n / 2
    let final_position := middle_rung + 5 - 7 + 8 + 7
    (n % 2 = 1) ∧ (final_position = n) → n = 27

-- The proof is omitted
theorem ladder_rungs_count_proof : ladder_rungs_count 27 := by sorry

end ladder_rungs_count_ladder_rungs_count_proof_l2521_252124


namespace smallest_candy_count_l2521_252146

theorem smallest_candy_count : ∃ (n : ℕ), 
  (100 ≤ n ∧ n < 1000) ∧ 
  (n + 6) % 9 = 0 ∧ 
  (n - 9) % 6 = 0 ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n) → (m + 6) % 9 ≠ 0 ∨ (m - 9) % 6 ≠ 0) ∧
  n = 111 := by
sorry

end smallest_candy_count_l2521_252146


namespace circle_center_l2521_252156

/-- Given a circle with equation x^2 + y^2 - 2mx - 3 = 0, where m < 0 and radius 2, 
    prove that its center is (-1, 0) -/
theorem circle_center (m : ℝ) (h1 : m < 0) :
  let eq := fun (x y : ℝ) ↦ x^2 + y^2 - 2*m*x - 3 = 0
  let r : ℝ := 2
  ∃ (C : ℝ × ℝ), C = (-1, 0) ∧ 
    (∀ (x y : ℝ), eq x y ↔ (x - C.1)^2 + (y - C.2)^2 = r^2) := by
  sorry

end circle_center_l2521_252156


namespace compacted_space_calculation_l2521_252180

/-- The number of cans Nick has -/
def num_cans : ℕ := 100

/-- The space each can takes up before compaction (in square inches) -/
def initial_space : ℝ := 30

/-- The percentage of space each can takes up after compaction -/
def compaction_ratio : ℝ := 0.35

/-- The total space occupied by all cans after compaction (in square inches) -/
def total_compacted_space : ℝ := num_cans * (initial_space * compaction_ratio)

theorem compacted_space_calculation :
  total_compacted_space = 1050 := by sorry

end compacted_space_calculation_l2521_252180


namespace shinyoung_read_most_l2521_252160

theorem shinyoung_read_most (shinyoung seokgi woong : ℚ) : 
  shinyoung = 1/3 ∧ seokgi = 1/4 ∧ woong = 1/5 → 
  shinyoung > seokgi ∧ shinyoung > woong := by
  sorry

end shinyoung_read_most_l2521_252160


namespace coffee_blend_weight_l2521_252122

/-- Represents the total weight of a coffee blend -/
def total_blend_weight (price_a price_b price_mix : ℚ) (weight_a : ℚ) : ℚ :=
  weight_a + (price_a * weight_a - price_mix * weight_a) / (price_mix - price_b)

/-- Theorem stating the total weight of the coffee blend -/
theorem coffee_blend_weight :
  total_blend_weight 9 8 (84/10) 8 = 20 := by
  sorry

end coffee_blend_weight_l2521_252122


namespace quadratic_root_sum_l2521_252133

theorem quadratic_root_sum (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + m*x + 2 = 0 ↔ x = x₁ ∨ x = x₂) → 
  x₁ + x₂ = -4 → 
  m = 4 := by
sorry

end quadratic_root_sum_l2521_252133


namespace number_of_valid_paths_l2521_252131

-- Define the grid dimensions
def grid_width : ℕ := 8
def grid_height : ℕ := 4

-- Define the blocked segments
def blocked_segments : List (ℕ × ℕ × ℕ × ℕ) := [(6, 2, 6, 3), (8, 2, 8, 3)]

-- Define a function to calculate valid paths
def valid_paths (width : ℕ) (height : ℕ) (blocked : List (ℕ × ℕ × ℕ × ℕ)) : ℕ :=
  sorry

-- Theorem statement
theorem number_of_valid_paths :
  valid_paths grid_width grid_height blocked_segments = 271 :=
sorry

end number_of_valid_paths_l2521_252131


namespace monotonic_increasing_cubic_linear_l2521_252127

/-- The function f(x) = x^3 - ax is monotonically increasing over ℝ if and only if a ≤ 0 -/
theorem monotonic_increasing_cubic_linear (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => x^3 - a*x)) ↔ a ≤ 0 := by sorry

end monotonic_increasing_cubic_linear_l2521_252127


namespace balanced_split_theorem_l2521_252105

/-- A finite collection of positive real numbers is balanced if each number
    is less than the sum of the others. -/
def IsBalanced (s : Finset ℝ) : Prop :=
  ∀ x ∈ s, x < (s.sum id - x)

/-- A finite collection of positive real numbers can be split into three parts
    with the property that the sum of the numbers in each part is less than
    the sum of the numbers in the two other parts. -/
def CanSplitIntoThreeParts (s : Finset ℝ) : Prop :=
  ∃ (a b c : Finset ℝ), a ∪ b ∪ c = s ∧ a ∩ b = ∅ ∧ b ∩ c = ∅ ∧ a ∩ c = ∅ ∧
    a.sum id < b.sum id + c.sum id ∧
    b.sum id < a.sum id + c.sum id ∧
    c.sum id < a.sum id + b.sum id

/-- The main theorem -/
theorem balanced_split_theorem (m : ℕ) (hm : m ≥ 3) :
  (∀ (s : Finset ℝ), s.card = m → IsBalanced s → CanSplitIntoThreeParts s) ↔ m ≠ 4 :=
sorry

end balanced_split_theorem_l2521_252105


namespace line_divides_l_shape_in_half_l2521_252153

/-- L-shaped region in the xy-plane -/
structure LShapedRegion where
  vertices : List (ℝ × ℝ) := [(0,0), (0,4), (4,4), (4,2), (6,2), (6,0)]

/-- Line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Calculate the area of a polygon given its vertices -/
def calculateArea (vertices : List (ℝ × ℝ)) : ℝ := sorry

/-- Calculate the slope of a line -/
def calculateSlope (l : Line) : ℝ := sorry

/-- Check if a line divides a region in half -/
def divides_in_half (l : Line) (r : LShapedRegion) : Prop := sorry

/-- Theorem: The line through (0,0) and (2,4) divides the L-shaped region in half -/
theorem line_divides_l_shape_in_half :
  let l : Line := { point1 := (0, 0), point2 := (2, 4) }
  let r : LShapedRegion := {}
  divides_in_half l r ∧ calculateSlope l = 2 := by sorry

end line_divides_l_shape_in_half_l2521_252153


namespace second_place_wins_l2521_252140

/-- Represents a hockey team's performance --/
structure TeamPerformance where
  wins : ℕ
  ties : ℕ

/-- Calculates points for a team based on wins and ties --/
def calculatePoints (team : TeamPerformance) : ℕ := 2 * team.wins + team.ties

/-- Represents the hockey league --/
structure HockeyLeague where
  firstPlace : TeamPerformance
  secondPlace : TeamPerformance
  elsasTeam : TeamPerformance

theorem second_place_wins (league : HockeyLeague) : 
  league.firstPlace = ⟨12, 4⟩ →
  league.elsasTeam = ⟨8, 10⟩ →
  league.secondPlace.ties = 1 →
  (calculatePoints league.firstPlace + calculatePoints league.secondPlace + calculatePoints league.elsasTeam) / 3 = 27 →
  league.secondPlace.wins = 13 := by
  sorry

#eval calculatePoints ⟨13, 1⟩  -- Expected output: 27

end second_place_wins_l2521_252140


namespace p_sufficient_not_necessary_for_q_l2521_252196

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (|x - 3| < 1 → x^2 + x - 6 > 0)) ∧
  (∃ x : ℝ, x^2 + x - 6 > 0 ∧ ¬(|x - 3| < 1)) :=
by sorry

end p_sufficient_not_necessary_for_q_l2521_252196


namespace distance_between_trees_l2521_252149

/-- Given a yard of length 275 meters with 26 trees planted at equal distances,
    including one at each end, the distance between consecutive trees is 11 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 275 →
  num_trees = 26 →
  yard_length / (num_trees - 1) = 11 :=
by
  sorry

end distance_between_trees_l2521_252149


namespace logarithm_sum_inequality_l2521_252173

theorem logarithm_sum_inequality : 
  Real.log 6 / Real.log 5 + Real.log 7 / Real.log 6 + Real.log 8 / Real.log 7 + Real.log 5 / Real.log 8 > 4 := by
  sorry

end logarithm_sum_inequality_l2521_252173


namespace speed_ratio_is_three_fourths_l2521_252175

/-- Represents the motion of objects A and B -/
structure Motion where
  vA : ℝ  -- Speed of A
  vB : ℝ  -- Speed of B

/-- The conditions of the problem -/
def satisfiesConditions (m : Motion) : Prop :=
  let distanceB := 800  -- Initial distance of B from O
  let t1 := 3           -- Time of first equidistance (in minutes)
  let t2 := 15          -- Time of second equidistance (in minutes)
  (t1 * m.vA = |distanceB - t1 * m.vB|) ∧   -- Equidistance at t1
  (t2 * m.vA = |distanceB - t2 * m.vB|)     -- Equidistance at t2

/-- The theorem to be proved -/
theorem speed_ratio_is_three_fourths :
  ∃ m : Motion, satisfiesConditions m ∧ m.vA / m.vB = 3 / 4 := by
  sorry


end speed_ratio_is_three_fourths_l2521_252175


namespace max_planes_from_three_parallel_lines_l2521_252198

/-- A line in 3D space -/
structure Line3D where
  -- We don't need to define the internal structure of a line for this problem

/-- A plane in 3D space -/
structure Plane where
  -- We don't need to define the internal structure of a plane for this problem

/-- Determines if two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Determines if a line lies on a plane -/
def lineOnPlane (l : Line3D) (p : Plane) : Prop :=
  sorry

/-- Determines if a plane is defined by two lines -/
def planeDefinedByLines (p : Plane) (l1 l2 : Line3D) : Prop :=
  sorry

/-- The main theorem: maximum number of planes defined by three parallel lines -/
theorem max_planes_from_three_parallel_lines (l1 l2 l3 : Line3D) 
  (h_parallel_12 : parallel l1 l2) 
  (h_parallel_23 : parallel l2 l3) 
  (h_parallel_13 : parallel l1 l3) :
  ∃ (p1 p2 p3 : Plane), 
    (∀ (p : Plane), (planeDefinedByLines p l1 l2 ∨ planeDefinedByLines p l2 l3 ∨ planeDefinedByLines p l1 l3) → 
      (p = p1 ∨ p = p2 ∨ p = p3)) ∧
    (∃ (p : Plane), planeDefinedByLines p l1 l2 ∧ planeDefinedByLines p l2 l3 ∧ planeDefinedByLines p l1 l3) →
      (p1 = p2 ∧ p2 = p3) :=
by
  sorry

end max_planes_from_three_parallel_lines_l2521_252198


namespace min_sum_squares_l2521_252187

theorem min_sum_squares (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  ∀ m : ℝ, m = a^2 + b^2 + c^2 → m ≥ 3 := by
  sorry

end min_sum_squares_l2521_252187


namespace tenth_square_area_l2521_252183

theorem tenth_square_area : 
  let initial_side : ℝ := 2
  let side_sequence : ℕ → ℝ := λ n => initial_side * (Real.sqrt 2) ^ (n - 1)
  let area : ℕ → ℝ := λ n => (side_sequence n) ^ 2
  area 10 = 2048 := by sorry

end tenth_square_area_l2521_252183


namespace fish_catch_theorem_l2521_252163

def mike_rate : ℕ := 30
def jim_rate : ℕ := 2 * mike_rate
def bob_rate : ℕ := (3 * jim_rate) / 2

def total_fish_caught (mike_rate jim_rate bob_rate : ℕ) : ℕ :=
  let fish_40_min := (mike_rate + jim_rate + bob_rate) * 2 / 3
  let fish_20_min := jim_rate * 1 / 3
  fish_40_min + fish_20_min

theorem fish_catch_theorem :
  total_fish_caught mike_rate jim_rate bob_rate = 140 := by
  sorry

end fish_catch_theorem_l2521_252163


namespace x0_value_l2521_252132

-- Define the function f
def f (x : ℝ) : ℝ := 13 - 8*x + x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -8 + 2*x

-- Theorem statement
theorem x0_value (x₀ : ℝ) (h : f' x₀ = 4) : x₀ = 6 := by
  sorry

end x0_value_l2521_252132


namespace midpoint_distance_is_1300_l2521_252185

/-- The distance from the school to the midpoint of the total path -/
def midpoint_distance (school_to_kindergarten_km : ℕ) (school_to_kindergarten_m : ℕ) (kindergarten_to_house_m : ℕ) : ℕ :=
  ((school_to_kindergarten_km * 1000 + school_to_kindergarten_m + kindergarten_to_house_m) / 2)

/-- Theorem stating that the midpoint distance is 1300 meters -/
theorem midpoint_distance_is_1300 :
  midpoint_distance 1 700 900 = 1300 := by
  sorry

#eval midpoint_distance 1 700 900

end midpoint_distance_is_1300_l2521_252185


namespace fraction_of_books_sold_l2521_252129

/-- Given a collection of books where some were sold and some remained unsold,
    this theorem proves the fraction of books sold. -/
theorem fraction_of_books_sold
  (price_per_book : ℝ)
  (unsold_books : ℕ)
  (total_revenue : ℝ)
  (h1 : price_per_book = 3.5)
  (h2 : unsold_books = 40)
  (h3 : total_revenue = 280.00000000000006) :
  (total_revenue / price_per_book) / ((total_revenue / price_per_book) + unsold_books : ℝ) = 2/3 := by
  sorry

#eval (280.00000000000006 / 3.5) / ((280.00000000000006 / 3.5) + 40)

end fraction_of_books_sold_l2521_252129


namespace isabellas_hair_length_l2521_252144

/-- Given Isabella's hair length at the end of the year and the amount it grew,
    prove that her initial hair length is equal to the final length minus the growth. -/
theorem isabellas_hair_length (final_length growth : ℕ) (h : final_length = 24 ∧ growth = 6) :
  final_length - growth = 18 := by
  sorry

end isabellas_hair_length_l2521_252144


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2521_252193

/-- An isosceles triangle with side lengths 4, 9, and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c => 
    (a = 4 ∧ b = 9 ∧ c = 9) →  -- Two sides are 9, one side is 4
    (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
    (b = c) →  -- Isosceles condition
    (a + b + c = 22)  -- Perimeter is 22

#check isosceles_triangle_perimeter

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : 
  ∃ (a b c : ℝ), isosceles_triangle_perimeter a b c :=
by
  sorry


end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2521_252193


namespace sqrt_difference_equals_negative_six_sqrt_two_l2521_252166

theorem sqrt_difference_equals_negative_six_sqrt_two :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) - Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = -6 * Real.sqrt 2 := by
  sorry

end sqrt_difference_equals_negative_six_sqrt_two_l2521_252166


namespace articles_produced_l2521_252116

/-- Given that x men working x hours a day for 2x days produce 2x³ articles,
    prove that y men working 2y hours a day for y days produce 2y³ articles. -/
theorem articles_produced (x y : ℕ) (h : x * x * (2 * x) = 2 * x^3) :
  y * (2 * y) * y = 2 * y^3 := by
  sorry

end articles_produced_l2521_252116


namespace walnut_trees_count_l2521_252130

/-- The number of walnut trees in the park after planting and removing trees -/
def final_tree_count (initial_trees : ℕ) (planted_group1 : ℕ) (planted_group2 : ℕ) (planted_group3 : ℕ) (removed_trees : ℕ) : ℕ :=
  initial_trees + planted_group1 + planted_group2 + planted_group3 - removed_trees

/-- Theorem stating that the final number of walnut trees in the park is 55 -/
theorem walnut_trees_count : final_tree_count 22 12 15 10 4 = 55 := by
  sorry

end walnut_trees_count_l2521_252130


namespace permutation_element_selection_l2521_252100

theorem permutation_element_selection (n : ℕ) (hn : n ≥ 10) :
  (Finset.range n).card.choose 3 = Nat.choose (n - 7) 3 :=
by sorry

end permutation_element_selection_l2521_252100


namespace five_divides_square_iff_five_divides_l2521_252134

theorem five_divides_square_iff_five_divides (a : ℤ) : 
  5 ∣ a^2 ↔ 5 ∣ a := by sorry

end five_divides_square_iff_five_divides_l2521_252134


namespace inequality_holds_for_all_z_l2521_252171

theorem inequality_holds_for_all_z (x y : ℝ) (hx : x > 0) :
  ∀ z : ℝ, y - z < Real.sqrt (z^2 + x^2) := by
  sorry

end inequality_holds_for_all_z_l2521_252171


namespace digit_deletion_divisibility_l2521_252126

theorem digit_deletion_divisibility (d : ℕ) (h : d > 0) : 
  ∃ (n n1 k a b c : ℕ), 
    n = 10^k * (10*a + b) + c ∧
    n1 = 10^k * a + c ∧
    0 < b ∧ b < 10 ∧
    c < 10^k ∧
    d ∣ n ∧
    d ∣ n1 :=
sorry

end digit_deletion_divisibility_l2521_252126


namespace no_prime_common_multiple_under_70_l2521_252165

theorem no_prime_common_multiple_under_70 : ¬ ∃ n : ℕ, 
  (10 ∣ n) ∧ (15 ∣ n) ∧ (n < 70) ∧ Nat.Prime n :=
by sorry

end no_prime_common_multiple_under_70_l2521_252165


namespace domain_intersection_is_closed_open_interval_l2521_252142

-- Define the domains of the two functions
def domain_sqrt (x : ℝ) : Prop := 4 - x^2 ≥ 0
def domain_ln (x : ℝ) : Prop := 4 - x > 0

-- Define the intersection of the domains
def domain_intersection (x : ℝ) : Prop := domain_sqrt x ∧ domain_ln x

-- Theorem statement
theorem domain_intersection_is_closed_open_interval :
  ∀ x, domain_intersection x ↔ x ∈ Set.Ici (-2) ∩ Set.Iio 1 :=
sorry

end domain_intersection_is_closed_open_interval_l2521_252142


namespace arithmetic_sequence_kth_term_l2521_252102

/-- Given an arithmetic sequence where the sum of the first n terms is 3n^2 + 2n,
    this theorem proves that the k-th term is 6k - 1. -/
theorem arithmetic_sequence_kth_term 
  (S : ℕ → ℝ) -- S represents the sum function of the sequence
  (h : ∀ n : ℕ, S n = 3 * n^2 + 2 * n) -- condition that sum of first n terms is 3n^2 + 2n
  (k : ℕ) -- k represents the index of the term we're looking for
  : S k - S (k-1) = 6 * k - 1 := by
  sorry


end arithmetic_sequence_kth_term_l2521_252102


namespace tabithas_age_l2521_252139

/-- Tabitha's hair color problem -/
theorem tabithas_age :
  ∀ (current_year : ℕ) (start_year : ℕ) (start_colors : ℕ) (future_year : ℕ) (future_colors : ℕ),
  start_year = 15 →
  start_colors = 2 →
  future_year = current_year + 3 →
  future_colors = 8 →
  future_colors = start_colors + (future_year - start_year) →
  current_year = 18 :=
by sorry

end tabithas_age_l2521_252139


namespace subset_intersection_cardinality_l2521_252106

theorem subset_intersection_cardinality (n m : ℕ) (Z : Finset ℕ) 
  (A : Fin m → Finset ℕ) : 
  (Z.card = n) →
  (∀ i : Fin m, A i ⊂ Z) →
  (∀ i j : Fin m, i ≠ j → (A i ∩ A j).card = 1) →
  m ≤ n := by
  sorry

end subset_intersection_cardinality_l2521_252106


namespace cubic_roots_sum_cubes_l2521_252164

theorem cubic_roots_sum_cubes (p q r : ℝ) : 
  (p^3 - 2*p^2 + 7*p - 1 = 0) → 
  (q^3 - 2*q^2 + 7*q - 1 = 0) → 
  (r^3 - 2*r^2 + 7*r - 1 = 0) → 
  (p + q + r = 2) →
  (p * q * r = 1) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = -3 := by
sorry

end cubic_roots_sum_cubes_l2521_252164


namespace min_cost_for_48_students_l2521_252135

/-- The minimum cost to purchase tickets for a group of students. -/
def min_ticket_cost (num_students : ℕ) (single_price : ℕ) (group_price : ℕ) : ℕ :=
  min
    ((num_students / 10) * group_price + (num_students % 10) * single_price)
    ((num_students / 10 + 1) * group_price)

/-- The minimum cost to purchase tickets for 48 students is 350 yuan. -/
theorem min_cost_for_48_students :
  min_ticket_cost 48 10 70 = 350 := by
  sorry

#eval min_ticket_cost 48 10 70

end min_cost_for_48_students_l2521_252135


namespace choose_3_from_13_l2521_252113

theorem choose_3_from_13 : Nat.choose 13 3 = 286 := by sorry

end choose_3_from_13_l2521_252113


namespace inequality_range_l2521_252141

theorem inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2*k*x^2 + k*x - 3/8 < 0) ↔ k ∈ Set.Ioc (-3) 0 :=
sorry

end inequality_range_l2521_252141


namespace car_distance_proof_l2521_252192

/-- Proves that a car traveling at 162 km/h for 5 hours covers a distance of 810 km -/
theorem car_distance_proof (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 162 → time = 5 → distance = speed * time → distance = 810 := by
sorry

end car_distance_proof_l2521_252192


namespace simple_interest_time_calculation_l2521_252188

theorem simple_interest_time_calculation (P : ℝ) (h1 : P > 0) : ∃ T : ℝ,
  (P * 5 * T) / 100 = P / 5 ∧ T = 4 := by
  sorry

end simple_interest_time_calculation_l2521_252188


namespace det_A_l2521_252189

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 1, -2; 8, 5, -4; 3, 3, 7]

def A' : Matrix (Fin 3) (Fin 3) ℤ := 
  Matrix.of (λ i j => 
    if i = 0 then A i j
    else A i j - A 0 j)

theorem det_A'_eq_55 : Matrix.det A' = 55 := by sorry

end det_A_l2521_252189


namespace race_result_l2521_252128

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a runner given time -/
def distance (r : Runner) (t : ℝ) : ℝ := r.speed * t

theorem race_result (a b : Runner) 
  (h1 : a.time = 240)
  (h2 : b.time = a.time + 10)
  (h3 : distance a a.time = 1000) :
  distance a a.time - distance b a.time = 40 := by
  sorry

end race_result_l2521_252128


namespace sum_of_squares_of_cubic_roots_l2521_252123

theorem sum_of_squares_of_cubic_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 5 * p + 15 = 0) →
  (3 * q^3 - 2 * q^2 + 5 * q + 15 = 0) →
  (3 * r^3 - 2 * r^2 + 5 * r + 15 = 0) →
  p^2 + q^2 + r^2 = -26/9 := by
sorry

end sum_of_squares_of_cubic_roots_l2521_252123


namespace movie_ticket_distribution_l2521_252114

theorem movie_ticket_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  (n.descFactorial k) = 720 :=
sorry

end movie_ticket_distribution_l2521_252114


namespace sum_of_a_equals_2673_l2521_252137

def a (n : ℕ) : ℕ :=
  if n % 15 = 0 ∧ n % 18 = 0 then 15
  else if n % 18 = 0 ∧ n % 12 = 0 then 16
  else if n % 12 = 0 ∧ n % 15 = 0 then 17
  else 0

theorem sum_of_a_equals_2673 :
  (Finset.range 3000).sum (fun n => a (n + 1)) = 2673 := by
  sorry

end sum_of_a_equals_2673_l2521_252137


namespace set_intersection_empty_implies_a_range_l2521_252158

def A (a : ℝ) := {x : ℝ | a - 1 < x ∧ x < 2*a + 1}
def B := {x : ℝ | 0 < x ∧ x < 1}

theorem set_intersection_empty_implies_a_range (a : ℝ) :
  A a ∩ B = ∅ ↔ a ≤ -1/2 ∨ a ≥ 2 :=
sorry

end set_intersection_empty_implies_a_range_l2521_252158


namespace crayfish_yield_theorem_l2521_252138

theorem crayfish_yield_theorem (last_year_total : ℝ) (this_year_total : ℝ) 
  (yield_difference : ℝ) (h1 : last_year_total = 4800) 
  (h2 : this_year_total = 6000) (h3 : yield_difference = 60) : 
  ∃ (x : ℝ), x = 300 ∧ this_year_total / x = last_year_total / (x - yield_difference) :=
by sorry

end crayfish_yield_theorem_l2521_252138


namespace sandy_obtained_45_marks_l2521_252194

/-- Calculates the total marks obtained by Sandy given the number of correct and incorrect sums. -/
def sandy_marks (total_sums : ℕ) (correct_sums : ℕ) : ℤ :=
  let incorrect_sums := total_sums - correct_sums
  3 * correct_sums - 2 * incorrect_sums

/-- Proves that Sandy obtained 45 marks given the problem conditions. -/
theorem sandy_obtained_45_marks :
  sandy_marks 30 21 = 45 := by
  sorry

#eval sandy_marks 30 21

end sandy_obtained_45_marks_l2521_252194


namespace twenty_cent_coins_count_l2521_252182

/-- Represents the coin collection of Alex -/
structure CoinCollection where
  total_coins : ℕ
  ten_cent_coins : ℕ
  twenty_cent_coins : ℕ
  total_is_sum : total_coins = ten_cent_coins + twenty_cent_coins
  all_coins_accounted : total_coins = 14

/-- Calculates the number of different values obtainable from a given coin collection -/
def different_values (c : CoinCollection) : ℕ :=
  27 - c.ten_cent_coins

/-- The main theorem stating that if there are 22 different obtainable values, 
    then there must be 9 20-cent coins -/
theorem twenty_cent_coins_count 
  (c : CoinCollection) 
  (h : different_values c = 22) : 
  c.twenty_cent_coins = 9 := by
  sorry

end twenty_cent_coins_count_l2521_252182


namespace complement_union_theorem_l2521_252119

-- Define the universal set U
def U : Finset Char := {'a', 'b', 'c', 'd'}

-- Define set A
def A : Finset Char := {'a', 'b'}

-- Define set B
def B : Finset Char := {'b', 'c', 'd'}

-- Theorem statement
theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {'a', 'c', 'd'} := by sorry

end complement_union_theorem_l2521_252119


namespace ratio_sum_equality_l2521_252191

theorem ratio_sum_equality (a b c x y z : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
  (h_abc : a^2 + b^2 + c^2 = 49)
  (h_xyz : x^2 + y^2 + z^2 = 64)
  (h_dot : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end ratio_sum_equality_l2521_252191


namespace intersection_of_M_and_N_l2521_252178

open Set

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | x * (x - 2) < 0}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_of_M_and_N_l2521_252178


namespace paper_clip_distribution_l2521_252107

theorem paper_clip_distribution (total_clips : ℕ) (num_boxes : ℕ) (clips_per_box : ℕ) : 
  total_clips = 81 → num_boxes = 9 → clips_per_box = total_clips / num_boxes → clips_per_box = 9 := by
  sorry

end paper_clip_distribution_l2521_252107


namespace kamal_present_age_l2521_252195

/-- Represents the present age of Kamal -/
def kamal_age : ℕ := sorry

/-- Represents the present age of Kamal's son -/
def son_age : ℕ := sorry

/-- The condition that 8 years ago, Kamal was 4 times as old as his son -/
axiom condition1 : kamal_age - 8 = 4 * (son_age - 8)

/-- The condition that after 8 years, Kamal will be twice as old as his son -/
axiom condition2 : kamal_age + 8 = 2 * (son_age + 8)

/-- Theorem stating that Kamal's present age is 40 years -/
theorem kamal_present_age : kamal_age = 40 := by sorry

end kamal_present_age_l2521_252195


namespace lg_equation_l2521_252162

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_equation : (lg 5)^2 + lg 2 * lg 50 = 1 := by
  sorry

end lg_equation_l2521_252162


namespace jamie_bathroom_theorem_l2521_252152

/-- The amount of liquid (in ounces) that triggers the need to use the bathroom -/
def bathroom_threshold : ℕ := 32

/-- The amount of liquid (in ounces) in a cup -/
def cup_ounces : ℕ := 8

/-- The amount of liquid (in ounces) in a pint -/
def pint_ounces : ℕ := 16

/-- The amount of liquid Jamie consumed before the test -/
def pre_test_consumption : ℕ := cup_ounces + pint_ounces

/-- The maximum amount Jamie can drink during the test without needing the bathroom -/
def max_test_consumption : ℕ := bathroom_threshold - pre_test_consumption

theorem jamie_bathroom_theorem : max_test_consumption = 8 := by
  sorry

end jamie_bathroom_theorem_l2521_252152


namespace triangle_inequality_l2521_252109

-- Define a triangle ABC in 2D space
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point M
def Point := ℝ × ℝ

-- Function to calculate distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Function to check if a point is inside a triangle
def isInside (t : Triangle) (p : Point) : Prop := sorry

-- Function to calculate the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_inequality (t : Triangle) (M : Point) 
  (h : isInside t M) : 
  min (distance M t.A) (min (distance M t.B) (distance M t.C)) + 
  distance M t.A + distance M t.B + distance M t.C < 
  perimeter t := by
  sorry

end triangle_inequality_l2521_252109


namespace integer_roots_of_polynomial_l2521_252176

/-- The set of all possible integer roots for the polynomial x^4 + 4x^3 + a_2 x^2 + a_1 x - 60 = 0 -/
def possible_roots : Set ℤ := {1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 10, -10, 12, -12, 15, -15, 20, -20, 30, -30, 60, -60}

/-- The polynomial x^4 + 4x^3 + a_2 x^2 + a_1 x - 60 -/
def polynomial (a₂ a₁ x : ℤ) : ℤ := x^4 + 4*x^3 + a₂*x^2 + a₁*x - 60

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  ∀ x : ℤ, polynomial a₂ a₁ x = 0 → x ∈ possible_roots :=
sorry

end integer_roots_of_polynomial_l2521_252176


namespace no_solution_fibonacci_equation_l2521_252174

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- State the theorem
theorem no_solution_fibonacci_equation :
  ∀ n : ℕ, n * (fib n) * (fib (n + 1)) ≠ (fib (n + 2) - 1)^2 :=
by sorry

end no_solution_fibonacci_equation_l2521_252174


namespace number_ordering_l2521_252154

theorem number_ordering (a b : ℝ) (ha : a > 0) (hb : 0 < b ∧ b < 1) :
  a^b > b^a ∧ b^a > Real.log b := by sorry

end number_ordering_l2521_252154


namespace unique_prime_digit_l2521_252145

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def number (B : ℕ) : ℕ := 303160 + B

theorem unique_prime_digit :
  ∃! B : ℕ, B < 10 ∧ is_prime (number B) :=
sorry

end unique_prime_digit_l2521_252145


namespace chandler_saves_49_weeks_l2521_252136

/-- The number of weeks it takes Chandler to save for a mountain bike --/
def weeks_to_save : ℕ :=
  let bike_cost : ℕ := 620
  let birthday_money : ℕ := 70 + 40 + 20
  let weekly_earnings : ℕ := 18
  let weekly_spending : ℕ := 8
  let weekly_savings : ℕ := weekly_earnings - weekly_spending
  ((bike_cost - birthday_money) + weekly_savings - 1) / weekly_savings

theorem chandler_saves_49_weeks :
  weeks_to_save = 49 :=
sorry

end chandler_saves_49_weeks_l2521_252136


namespace prop_negation_false_l2521_252121

theorem prop_negation_false (p q : Prop) : 
  ¬(¬(p ∧ q)) → (p ∧ q) := by
  sorry

end prop_negation_false_l2521_252121


namespace john_uber_profit_l2521_252118

/-- John's profit from driving Uber --/
def uber_profit (earnings depreciation : ℕ) : ℕ :=
  earnings - depreciation

/-- Depreciation of John's car --/
def car_depreciation (purchase_price trade_in_value : ℕ) : ℕ :=
  purchase_price - trade_in_value

theorem john_uber_profit :
  let earnings : ℕ := 30000
  let purchase_price : ℕ := 18000
  let trade_in_value : ℕ := 6000
  uber_profit earnings (car_depreciation purchase_price trade_in_value) = 18000 := by
sorry

end john_uber_profit_l2521_252118


namespace exam_failure_marks_l2521_252110

theorem exam_failure_marks (T : ℕ) (passing_mark : ℕ) : 
  (60 * T / 100 - 20 = passing_mark) →
  (passing_mark = 160) →
  (passing_mark - 40 * T / 100 = 40) :=
by sorry

end exam_failure_marks_l2521_252110


namespace union_complement_equals_set_l2521_252167

def U : Set ℕ := {x | x < 4}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem union_complement_equals_set : B ∪ (U \ A) = {2, 3} := by sorry

end union_complement_equals_set_l2521_252167


namespace carters_baseball_cards_l2521_252169

/-- Given that Marcus has 350 baseball cards and 95 more cards than Carter,
    prove that Carter has 255 baseball cards. -/
theorem carters_baseball_cards : 
  ∀ (marcus_cards carter_cards : ℕ), 
    marcus_cards = 350 → 
    marcus_cards = carter_cards + 95 →
    carter_cards = 255 := by
  sorry

end carters_baseball_cards_l2521_252169


namespace factorial_plus_one_divisible_implies_prime_l2521_252190

theorem factorial_plus_one_divisible_implies_prime (n : ℕ) :
  (Nat.factorial n + 1) % (n + 1) = 0 → Nat.Prime (n + 1) := by
  sorry

end factorial_plus_one_divisible_implies_prime_l2521_252190


namespace number_exists_l2521_252199

theorem number_exists : ∃ x : ℝ, x * 1.6 - (2 * 1.4) / 1.3 = 4 := by
  sorry

end number_exists_l2521_252199


namespace certain_number_proof_l2521_252115

theorem certain_number_proof : 
  ∃ (x : ℝ), x / 1.45 = 17.5 → x = 25.375 := by
  sorry

end certain_number_proof_l2521_252115


namespace restaurant_bill_calculation_l2521_252159

theorem restaurant_bill_calculation (adults children meal_cost : ℕ) 
  (h1 : adults = 2) 
  (h2 : children = 5) 
  (h3 : meal_cost = 3) : 
  (adults + children) * meal_cost = 21 := by
  sorry

end restaurant_bill_calculation_l2521_252159


namespace defective_draws_count_l2521_252117

/-- The number of ways to draw at least 3 defective products out of 5 from a batch of 50 products containing 4 defective ones -/
def defective_draws : ℕ := sorry

/-- Total number of products in the batch -/
def total_products : ℕ := 50

/-- Number of defective products in the batch -/
def defective_products : ℕ := 4

/-- Number of products drawn -/
def drawn_products : ℕ := 5

theorem defective_draws_count : defective_draws = 4186 := by sorry

end defective_draws_count_l2521_252117


namespace cos_315_degrees_l2521_252125

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end cos_315_degrees_l2521_252125


namespace solve_equation_l2521_252112

theorem solve_equation (n m x : ℚ) : 
  (5 / 7 : ℚ) = n / 91 ∧ 
  (5 / 7 : ℚ) = (m + n) / 105 ∧ 
  (5 / 7 : ℚ) = (x - m) / 140 → 
  x = 110 := by sorry

end solve_equation_l2521_252112
