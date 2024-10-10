import Mathlib

namespace parallelogram_side_sum_l1966_196603

/-- A parallelogram with side lengths 10, 12x-2, 5y+5, and 4 has x+y equal to 4/5 -/
theorem parallelogram_side_sum (x y : ℚ) : 
  (12 * x - 2 = 10) → (5 * y + 5 = 4) → x + y = 4/5 := by
  sorry

end parallelogram_side_sum_l1966_196603


namespace walnut_distribution_game_l1966_196693

-- Define the number of walnuts
def total_walnuts (n : ℕ) : ℕ := 2 * n + 1

-- Define Béja's division
def beja_division (n : ℕ) : ℕ × ℕ :=
  let total := total_walnuts n
  (2, total - 2)  -- Minimum possible division

-- Define Konia's subdivision
def konia_subdivision (a b : ℕ) : (ℕ × ℕ) × (ℕ × ℕ) :=
  ((1, a - 1), (1, b - 1))  -- Minimum possible subdivision

-- Define Konia's gain in each method
def konia_gain_method1 (n : ℕ) : ℕ :=
  let (a, b) := beja_division n
  let ((a1, a2), (b1, b2)) := konia_subdivision a b
  max a2 b2 + min a1 b1

def konia_gain_method2 (n : ℕ) : ℕ :=
  n  -- As proved in the solution

def konia_gain_method3 (n : ℕ) : ℕ :=
  n - 1  -- As proved in the solution

-- Theorem statement
theorem walnut_distribution_game (n : ℕ) (h : n ≥ 2) :
  konia_gain_method1 n > konia_gain_method2 n ∧
  konia_gain_method2 n > konia_gain_method3 n :=
sorry

end walnut_distribution_game_l1966_196693


namespace right_triangle_hypotenuse_l1966_196611

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 8 ∧ b = 15 ∧ c^2 = a^2 + b^2 → c = 17 := by
  sorry

end right_triangle_hypotenuse_l1966_196611


namespace strawberry_picking_l1966_196674

/-- Given the total number of strawberries picked by three people, the number
picked by two of them together, and the number picked by one person alone,
prove the number picked by the other two together. -/
theorem strawberry_picking (total : ℕ) (matthew_and_zac : ℕ) (zac_alone : ℕ)
  (h1 : total = 550)
  (h2 : matthew_and_zac = 250)
  (h3 : zac_alone = 200) :
  total - zac_alone = 350 := by
  sorry

end strawberry_picking_l1966_196674


namespace gcd_1729_1309_l1966_196673

theorem gcd_1729_1309 : Nat.gcd 1729 1309 = 7 := by
  sorry

end gcd_1729_1309_l1966_196673


namespace circumradius_of_specific_triangle_l1966_196697

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the circumcircle radius
def circumradius (t : Triangle) : ℝ := sorry

-- State the theorem
theorem circumradius_of_specific_triangle :
  ∀ t : Triangle,
  t.a = 2 →
  t.A = 2 * π / 3 →  -- 120° in radians
  circumradius t = 2 * Real.sqrt 3 / 3 := by sorry

end circumradius_of_specific_triangle_l1966_196697


namespace smaller_circle_area_l1966_196675

/-- Two externally tangent circles with common tangents -/
structure TangentCircles where
  r : ℝ  -- radius of smaller circle
  R : ℝ  -- radius of larger circle
  PA : ℝ  -- length of tangent segment PA
  AB : ℝ  -- length of tangent segment AB
  tangent : r > 0 ∧ R > 0 ∧ R > r  -- circles are externally tangent
  common_tangent : PA = AB  -- common tangent property
  length_condition : PA = 4  -- given length condition

/-- The area of the smaller circle in the TangentCircles configuration is 2π -/
theorem smaller_circle_area (tc : TangentCircles) : π * tc.r^2 = 2 * π := by
  sorry

#check smaller_circle_area

end smaller_circle_area_l1966_196675


namespace sam_found_18_seashells_l1966_196637

/-- The number of seashells Mary found -/
def mary_seashells : ℕ := 47

/-- The total number of seashells Sam and Mary found together -/
def total_seashells : ℕ := 65

/-- The number of seashells Sam found -/
def sam_seashells : ℕ := total_seashells - mary_seashells

theorem sam_found_18_seashells : sam_seashells = 18 := by
  sorry

end sam_found_18_seashells_l1966_196637


namespace puppies_per_cage_l1966_196605

theorem puppies_per_cage (initial_puppies : Nat) (sold_puppies : Nat) (num_cages : Nat)
  (h1 : initial_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : num_cages = 3)
  (h4 : initial_puppies > sold_puppies) :
  (initial_puppies - sold_puppies) / num_cages = 5 := by
  sorry

end puppies_per_cage_l1966_196605


namespace quadratic_roots_problem_l1966_196623

theorem quadratic_roots_problem (c : ℝ) 
  (h : ∃ r : ℝ, r^2 - 3*r + c = 0 ∧ (-r)^2 + 3*(-r) - c = 0) :
  ∃ x y : ℝ, x^2 - 3*x + c = 0 ∧ y^2 - 3*y + c = 0 ∧ x = 0 ∧ y = 3 :=
sorry

end quadratic_roots_problem_l1966_196623


namespace reappearance_is_lcm_l1966_196696

/-- The number of letters in the sequence -/
def num_letters : ℕ := 6

/-- The number of digits in the sequence -/
def num_digits : ℕ := 4

/-- The line number where the original sequence first reappears -/
def reappearance_line : ℕ := 12

/-- Theorem stating that the reappearance line is the LCM of the letter and digit cycle lengths -/
theorem reappearance_is_lcm : 
  reappearance_line = Nat.lcm num_letters num_digits := by sorry

end reappearance_is_lcm_l1966_196696


namespace lecture_scheduling_l1966_196616

theorem lecture_scheduling (n : ℕ) (h : n = 7) :
  (n.factorial / 2) = 2520 := by
  sorry

end lecture_scheduling_l1966_196616


namespace train_length_calculation_l1966_196618

theorem train_length_calculation (train_speed_kmph : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed_kmph = 72 →
  platform_length = 350 →
  crossing_time = 26 →
  let train_speed_mps := train_speed_kmph * (5/18)
  let total_distance := train_speed_mps * crossing_time
  let train_length := total_distance - platform_length
  train_length = 170 := by
sorry

end train_length_calculation_l1966_196618


namespace sum_of_coefficients_l1966_196633

theorem sum_of_coefficients (A B : ℝ) :
  (∀ x : ℝ, x ≠ 4 → A / (x - 4) + B * (x + 1) = (-4 * x^2 + 16 * x + 24) / (x - 4)) →
  A + B = 4 := by
sorry

end sum_of_coefficients_l1966_196633


namespace aquarium_has_hundred_fish_l1966_196655

/-- Represents the number of fish in an aquarium with specific conditions. -/
structure Aquarium where
  totalFish : ℕ
  clownfish : ℕ
  blowfish : ℕ
  blowfishInOwnTank : ℕ
  clownfishInDisplayTank : ℕ

/-- The aquarium satisfies the given conditions. -/
def validAquarium (a : Aquarium) : Prop :=
  a.clownfish = a.blowfish ∧
  a.blowfishInOwnTank = 26 ∧
  a.clownfishInDisplayTank = 16 ∧
  a.totalFish = a.clownfish + a.blowfish ∧
  a.clownfishInDisplayTank = (2 / 3 : ℚ) * (a.blowfish - a.blowfishInOwnTank)

/-- The theorem stating that a valid aquarium has 100 fish in total. -/
theorem aquarium_has_hundred_fish (a : Aquarium) (h : validAquarium a) : a.totalFish = 100 := by
  sorry

end aquarium_has_hundred_fish_l1966_196655


namespace elevator_min_trips_l1966_196643

def masses : List ℕ := [150, 60, 70, 71, 72, 100, 101, 102, 103]
def capacity : ℕ := 200

def is_valid_pair (m1 m2 : ℕ) : Prop := m1 + m2 ≤ capacity

def min_trips : ℕ := 5

theorem elevator_min_trips :
  (∀ (m1 m2 m3 : ℕ), m1 ∈ masses → m2 ∈ masses → m3 ∈ masses → m1 ≠ m2 → m2 ≠ m3 → m1 ≠ m3 → m1 + m2 + m3 > capacity) ∧
  (∃ (pairs : List (ℕ × ℕ)), 
    pairs.length = 4 ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ masses ∧ p.2 ∈ masses ∧ p.1 ≠ p.2 ∧ is_valid_pair p.1 p.2) ∧
    (∀ (m : ℕ), m ∈ masses → m = 150 ∨ (∃ (p : ℕ × ℕ), p ∈ pairs ∧ (m = p.1 ∨ m = p.2)))) →
  min_trips = 5 :=
by sorry

end elevator_min_trips_l1966_196643


namespace school_population_l1966_196681

/-- The total number of students in a school -/
def total_students : ℕ := sorry

/-- The number of boys in the school -/
def boys : ℕ := 75

/-- The number of girls in the school -/
def girls : ℕ := sorry

/-- Theorem stating the total number of students in the school -/
theorem school_population :
  (total_students = boys + girls) ∧ 
  (girls = (75 : ℚ) / 100 * total_students) →
  total_students = 300 := by sorry

end school_population_l1966_196681


namespace muffin_selection_problem_l1966_196654

theorem muffin_selection_problem :
  let n : ℕ := 6  -- Total number of muffins to select
  let k : ℕ := 4  -- Number of types of muffins
  Nat.choose (n + k - 1) (k - 1) = 84 := by
  sorry

end muffin_selection_problem_l1966_196654


namespace sin_2A_value_l1966_196658

theorem sin_2A_value (A : Real) (h : Real.cos (π/4 + A) = 5/13) : 
  Real.sin (2 * A) = 119/169 := by
  sorry

end sin_2A_value_l1966_196658


namespace quartic_polynomial_root_relation_l1966_196625

/-- Given a quartic polynomial ax^4 + bx^3 + cx^2 + dx + e = 0 with roots 4, -3, and 1, prove that (b+d)/a = -9/150 -/
theorem quartic_polynomial_root_relation (a b c d e : ℝ) (h1 : a ≠ 0) 
  (h2 : a * 4^4 + b * 4^3 + c * 4^2 + d * 4 + e = 0)
  (h3 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
  (h4 : a * 1^4 + b * 1^3 + c * 1^2 + d * 1 + e = 0) : 
  (b + d) / a = -9 / 150 := by
  sorry

end quartic_polynomial_root_relation_l1966_196625


namespace bakery_muffins_l1966_196662

/-- The number of muffins in each box -/
def muffins_per_box : ℕ := 5

/-- The number of available boxes -/
def available_boxes : ℕ := 10

/-- The number of additional boxes needed -/
def additional_boxes_needed : ℕ := 9

/-- The total number of muffins made by the bakery -/
def total_muffins : ℕ := muffins_per_box * (available_boxes + additional_boxes_needed)

theorem bakery_muffins : total_muffins = 95 := by
  sorry

end bakery_muffins_l1966_196662


namespace cubic_discriminant_example_l1966_196610

/-- The discriminant of a cubic equation ax^3 + bx^2 + cx + d -/
def cubic_discriminant (a b c d : ℝ) : ℝ :=
  -27 * a^2 * d^2 + 18 * a * b * c * d - 4 * b^3 * d + b^2 * c^2 - 4 * a * c^3

/-- The coefficients of the cubic equation x^3 - 2x^2 + 5x + 2 -/
def a : ℝ := 1
def b : ℝ := -2
def c : ℝ := 5
def d : ℝ := 2

theorem cubic_discriminant_example : cubic_discriminant a b c d = -640 := by
  sorry

end cubic_discriminant_example_l1966_196610


namespace game_not_fair_first_player_win_probability_limit_l1966_196692

/-- Represents a card game with n players and n cards. -/
structure CardGame where
  n : ℕ
  n_pos : 0 < n

/-- The probability of the first player winning the game. -/
noncomputable def firstPlayerWinProbability (game : CardGame) : ℝ :=
  Real.exp 1 / (game.n * (Real.exp 1 - 1))

/-- Theorem stating that the game is not fair for all players. -/
theorem game_not_fair (game : CardGame) : 
  ∃ (i j : ℕ), i ≠ j ∧ i ≤ game.n ∧ j ≤ game.n ∧ 
  (1 : ℝ) / game.n * (1 - (1 - 1 / game.n) ^ (game.n * (i - 1))) ≠ 
  (1 : ℝ) / game.n * (1 - (1 - 1 / game.n) ^ (game.n * (j - 1))) :=
sorry

/-- Theorem stating that the probability of the first player winning
    approaches e / (n * (e - 1)) as n becomes large. -/
theorem first_player_win_probability_limit (game : CardGame) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, game.n = n → 
  |firstPlayerWinProbability game - Real.exp 1 / (n * (Real.exp 1 - 1))| < ε :=
sorry

end game_not_fair_first_player_win_probability_limit_l1966_196692


namespace perpendicular_bisector_segments_theorem_l1966_196651

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  n_a : ℝ
  n_b : ℝ
  n_c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  h_order : a < b ∧ b < c

theorem perpendicular_bisector_segments_theorem (t : Triangle) :
  t.n_a > t.n_b ∧ t.n_c > t.n_b ∧
  ∃ (t1 t2 : Triangle), t1.n_a > t1.n_c ∧ t2.n_c > t2.n_a :=
by sorry

end perpendicular_bisector_segments_theorem_l1966_196651


namespace value_of_u_minus_v_l1966_196670

theorem value_of_u_minus_v (u v : ℚ) 
  (eq1 : 5 * u - 6 * v = 31)
  (eq2 : 3 * u + 5 * v = 4) : 
  u - v = 5.3 := by
  sorry

end value_of_u_minus_v_l1966_196670


namespace multiples_equality_l1966_196691

theorem multiples_equality (n : ℕ) : 
  let a : ℚ := (6 + 12 + 18 + 24 + 30 + 36 + 42) / 7
  let b : ℚ := 2 * n
  (a ^ 2 = b ^ 2) → n = 12 := by
  sorry

end multiples_equality_l1966_196691


namespace remainder_x6_minus_1_divided_by_x2_minus_x_plus_1_l1966_196679

theorem remainder_x6_minus_1_divided_by_x2_minus_x_plus_1 :
  ∃ q : Polynomial ℚ, (X^6 - 1) = (X^2 - X + 1) * q + (-2) := by sorry

end remainder_x6_minus_1_divided_by_x2_minus_x_plus_1_l1966_196679


namespace repeated_root_condition_l1966_196621

theorem repeated_root_condition (a : ℝ) : 
  (∃ x : ℝ, (3 / (x - 3) + a * x / (x^2 - 9) = 4 / (x + 3)) ∧ 
   (∀ ε > 0, ∃ y ≠ x, |y - x| < ε ∧ (3 / (y - 3) + a * y / (y^2 - 9) = 4 / (y + 3)))) ↔ 
  (a = -6 ∨ a = 8) := by
sorry

end repeated_root_condition_l1966_196621


namespace quadratic_equation_solution_l1966_196606

theorem quadratic_equation_solution :
  let a : ℝ := 2
  let b : ℝ := -8
  let c : ℝ := 5
  let x₁ : ℝ := 2 + Real.sqrt 6 / 2
  let x₂ : ℝ := 2 - Real.sqrt 6 / 2
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end quadratic_equation_solution_l1966_196606


namespace volume_object_A_l1966_196652

/-- Represents the volume of an object in a cube-shaped fishbowl --/
def volume_object (side_length : ℝ) (height_with_object : ℝ) (height_without_object : ℝ) : ℝ :=
  side_length^2 * (height_with_object - height_without_object)

/-- The volume of object (A) in the fishbowl is 2800 cubic centimeters --/
theorem volume_object_A :
  let side_length : ℝ := 20
  let height_with_object : ℝ := 16
  let height_without_object : ℝ := 9
  volume_object side_length height_with_object height_without_object = 2800 := by
  sorry

end volume_object_A_l1966_196652


namespace three_gorges_dam_capacity_scientific_notation_l1966_196629

-- Define the original number
def original_number : ℝ := 18200000

-- Define the scientific notation components
def coefficient : ℝ := 1.82
def exponent : ℤ := 7

-- Theorem statement
theorem three_gorges_dam_capacity_scientific_notation :
  original_number = coefficient * (10 : ℝ) ^ exponent :=
by sorry

end three_gorges_dam_capacity_scientific_notation_l1966_196629


namespace net_income_for_specific_case_l1966_196695

/-- Calculates the net income after tax for a tax resident --/
def net_income_after_tax (gross_income : ℝ) (tax_rate : ℝ) : ℝ :=
  gross_income * (1 - tax_rate)

/-- Theorem stating the net income after tax for a specific case --/
theorem net_income_for_specific_case :
  let gross_income : ℝ := 45000
  let tax_rate : ℝ := 0.13
  net_income_after_tax gross_income tax_rate = 39150 := by
  sorry

#eval net_income_after_tax 45000 0.13

end net_income_for_specific_case_l1966_196695


namespace polynomial_difference_divisibility_l1966_196682

theorem polynomial_difference_divisibility
  (a b c d x y : ℤ)
  (h : x ≠ y) :
  ∃ k : ℤ, a * x^3 + b * x^2 + c * x + d - (a * y^3 + b * y^2 + c * y + d) = (x - y) * k :=
sorry

end polynomial_difference_divisibility_l1966_196682


namespace function_symmetry_l1966_196607

/-- Given a function f(x) = ax + b*sin(x) + 1 where f(2017) = 7, prove that f(-2017) = -5 -/
theorem function_symmetry (a b : ℝ) :
  (let f := fun x => a * x + b * Real.sin x + 1
   f 2017 = 7) →
  (fun x => a * x + b * Real.sin x + 1) (-2017) = -5 := by
  sorry

end function_symmetry_l1966_196607


namespace overlap_probability_4x4_on_8x8_l1966_196617

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- Represents a square on the chessboard -/
structure Square :=
  (side : ℕ)

/-- The number of possible placements for a square on a chessboard -/
def num_placements (b : Chessboard) (s : Square) : ℕ :=
  (b.size - s.side + 1) * (b.size - s.side + 1)

/-- The number of non-overlapping placements for two squares -/
def num_non_overlapping (b : Chessboard) (s : Square) : ℕ :=
  2 * (b.size - s.side + 1) * (b.size - s.side + 1) - 4

/-- The probability of two squares overlapping -/
def overlap_probability (b : Chessboard) (s : Square) : ℚ :=
  1 - (num_non_overlapping b s : ℚ) / ((num_placements b s * num_placements b s) : ℚ)

theorem overlap_probability_4x4_on_8x8 :
  overlap_probability (Chessboard.mk 8) (Square.mk 4) = 529 / 625 := by
  sorry

end overlap_probability_4x4_on_8x8_l1966_196617


namespace unpainted_cubes_in_6x6x6_l1966_196638

/-- Represents a cube with painted rows --/
structure PaintedCube where
  size : Nat
  total_cubes : Nat
  painted_rows_per_face : Nat

/-- Calculates the number of unpainted cubes in a painted cube --/
def unpainted_cubes (c : PaintedCube) : Nat :=
  c.total_cubes - (6 * c.size * c.painted_rows_per_face - 12 * c.painted_rows_per_face)

/-- Theorem: In a 6x6x6 cube with 2 central rows painted on each face, there are 108 unpainted cubes --/
theorem unpainted_cubes_in_6x6x6 :
    let c : PaintedCube := { size := 6, total_cubes := 216, painted_rows_per_face := 2 }
    unpainted_cubes c = 108 := by
  sorry

end unpainted_cubes_in_6x6x6_l1966_196638


namespace max_vertices_with_unique_distances_five_vertices_with_unique_distances_exist_l1966_196615

/-- Represents a regular polygon with n sides inscribed in a circle -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A function to calculate the distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Checks if all pairwise distances in a list of points are unique -/
def allDistancesUnique (points : List (ℝ × ℝ)) : Prop := sorry

/-- The main theorem -/
theorem max_vertices_with_unique_distances
  (polygon : RegularPolygon 21)
  (selectedVertices : List (Fin 21)) :
  (allDistancesUnique (selectedVertices.map polygon.vertices)) →
  selectedVertices.length ≤ 5 := sorry

/-- The maximum number of vertices with unique distances is indeed achievable -/
theorem five_vertices_with_unique_distances_exist
  (polygon : RegularPolygon 21) :
  ∃ (selectedVertices : List (Fin 21)),
    selectedVertices.length = 5 ∧
    allDistancesUnique (selectedVertices.map polygon.vertices) := sorry

end max_vertices_with_unique_distances_five_vertices_with_unique_distances_exist_l1966_196615


namespace factor_implies_b_value_l1966_196630

/-- Q is a polynomial of degree 4 with a parameter b -/
def Q (b : ℝ) (x : ℝ) : ℝ := x^4 - 3*x^3 + b*x^2 - 12*x + 24

/-- Theorem: If x+2 is a factor of Q, then b = -22 -/
theorem factor_implies_b_value (b : ℝ) : 
  (∀ x, Q b x = 0 → x + 2 = 0) → b = -22 := by
  sorry

end factor_implies_b_value_l1966_196630


namespace min_bound_of_f_min_value_of_expression_l1966_196609

/-- The function f(x) = |x+1| + |x-1| -/
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

/-- The minimum value M such that f(x) ≤ M for all x ∈ ℝ is 2 -/
theorem min_bound_of_f : ∃ M : ℝ, (∀ x, f x ≤ M) ∧ (∀ N, (∀ x, f x ≤ N) → M ≤ N) ∧ M = 2 := by sorry

/-- Given positive a, b satisfying 3a + b = 2, the minimum value of 1/(2a) + 1/(a+b) is 2 -/
theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3 * a + b = 2) :
  ∃ m : ℝ, (1 / (2 * a) + 1 / (a + b) ≥ m) ∧
           (∀ x y, x > 0 → y > 0 → 3 * x + y = 2 → 1 / (2 * x) + 1 / (x + y) ≥ m) ∧
           m = 2 := by sorry

end min_bound_of_f_min_value_of_expression_l1966_196609


namespace equal_cost_sharing_l1966_196642

theorem equal_cost_sharing (A B C : ℝ) : 
  let total_cost := A + B + C
  let equal_share := total_cost / 3
  let leroy_additional_payment := equal_share - A
  leroy_additional_payment = (B + C - 2 * A) / 3 := by
sorry

end equal_cost_sharing_l1966_196642


namespace sweetsies_leftover_l1966_196641

theorem sweetsies_leftover (m : ℕ) : 
  (∃ k : ℕ, m = 11 * k + 8) →
  (∃ l : ℕ, 4 * m = 11 * l + 10) :=
by sorry

end sweetsies_leftover_l1966_196641


namespace tangency_points_form_circular_arc_l1966_196653

structure Segment where
  A : Point
  B : Point

structure Circle where
  center : Point
  radius : ℝ

def TangentCirclePair (s : Segment) (c1 c2 : Circle) : Prop :=
  -- Definition of tangent circle pair inscribed in segment
  sorry

def TangencyPoint (s : Segment) (c1 c2 : Circle) : Point :=
  -- Definition of tangency point between two circles
  sorry

def CircularArc (A B : Point) : Set Point :=
  -- Definition of a circular arc with endpoints A and B
  sorry

def AngleBisector (s : Segment) (arc : Set Point) : Set Point :=
  -- Definition of angle bisector between chord AB and segment arc
  sorry

theorem tangency_points_form_circular_arc (s : Segment) :
  ∃ (arc : Set Point), 
    (arc = CircularArc s.A s.B) ∧ 
    (arc = AngleBisector s arc) ∧
    (∀ (c1 c2 : Circle), TangentCirclePair s c1 c2 → 
      TangencyPoint s c1 c2 ∈ arc) := by
  sorry

end tangency_points_form_circular_arc_l1966_196653


namespace prime_power_modulo_l1966_196661

theorem prime_power_modulo (m : ℕ) (f : ℤ → ℤ) : m > 1 →
  (∃ x : ℤ, f x % m = 0) →
  (∃ y : ℤ, f y % m = 1) →
  (∃ z : ℤ, f z % m ≠ 0 ∧ f z % m ≠ 1) →
  (∃ p : ℕ, Prime p ∧ ∃ k : ℕ, m = p ^ k) :=
by sorry

end prime_power_modulo_l1966_196661


namespace remainder_of_product_of_nines_l1966_196644

def product_of_nines (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * (10^(i+1) - 1)) 1

theorem remainder_of_product_of_nines :
  product_of_nines 999 % 1000 = 109 := by
  sorry

end remainder_of_product_of_nines_l1966_196644


namespace age_difference_l1966_196671

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 13) : a = c + 13 := by
  sorry

end age_difference_l1966_196671


namespace minimum_gloves_needed_l1966_196677

theorem minimum_gloves_needed (participants : ℕ) (gloves_per_participant : ℕ) : 
  participants = 63 → gloves_per_participant = 2 → participants * gloves_per_participant = 126 := by
  sorry

end minimum_gloves_needed_l1966_196677


namespace two_copresidents_probability_l1966_196656

def club_sizes : List Nat := [6, 9, 10]
def copresident_counts : List Nat := [2, 3, 2]

def probability_two_copresidents (sizes : List Nat) (copresidents : List Nat) : ℚ :=
  let probabilities := List.zipWith (fun n p =>
    (Nat.choose p 2 * Nat.choose (n - p) 2) / Nat.choose n 4
  ) sizes copresidents
  (1 / 3 : ℚ) * (probabilities.sum)

theorem two_copresidents_probability :
  probability_two_copresidents club_sizes copresident_counts = 11 / 42 := by
  sorry

end two_copresidents_probability_l1966_196656


namespace floor_x_times_x_equals_90_l1966_196631

theorem floor_x_times_x_equals_90 :
  ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 90 ∧ x = 10 := by
  sorry

end floor_x_times_x_equals_90_l1966_196631


namespace polynomial_simplification_l1966_196684

theorem polynomial_simplification (x : ℝ) :
  (12 * x^10 + 5 * x^9 + 3 * x^7) + (2 * x^12 - x^10 + 2 * x^9 + 5 * x^6 + 7 * x^4 + 9 * x^2 + 4) =
  2 * x^12 + 11 * x^10 + 7 * x^9 + 3 * x^7 + 5 * x^6 + 7 * x^4 + 9 * x^2 + 4 :=
by sorry

end polynomial_simplification_l1966_196684


namespace sufficient_not_necessary_condition_l1966_196688

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x ≥ 2 ∧ y ≥ 2 → x + y ≥ 4) ∧
  (∃ x y : ℝ, x + y ≥ 4 ∧ (x < 2 ∨ y < 2)) :=
by sorry

end sufficient_not_necessary_condition_l1966_196688


namespace product_equals_328185_l1966_196685

theorem product_equals_328185 :
  let product := 9 * 11 * 13 * 15 * 17
  ∃ n : ℕ, n < 10 ∧ product = 300000 + n * 10000 + 8000 + 100 + 80 + 5 :=
by
  sorry

end product_equals_328185_l1966_196685


namespace deborah_finishes_first_l1966_196665

def lawn_problem (z r : ℝ) : Prop :=
  let jonathan_area := z
  let deborah_area := z / 3
  let ezekiel_area := z / 4
  let jonathan_rate := r
  let deborah_rate := r / 4
  let ezekiel_rate := r / 6
  let jonathan_time := jonathan_area / jonathan_rate
  let deborah_time := deborah_area / deborah_rate
  let ezekiel_time := ezekiel_area / ezekiel_rate
  (deborah_time < jonathan_time) ∧ (deborah_time < ezekiel_time)

theorem deborah_finishes_first (z r : ℝ) (hz : z > 0) (hr : r > 0) :
  lawn_problem z r :=
by
  sorry

end deborah_finishes_first_l1966_196665


namespace binomial_coefficient_ratio_l1966_196664

theorem binomial_coefficient_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₃ / a₂ = -2 := by
sorry

end binomial_coefficient_ratio_l1966_196664


namespace reciprocal_problem_l1966_196663

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 5) : 150 * (1 / x) = 240 := by
  sorry

end reciprocal_problem_l1966_196663


namespace parallel_line_theorem_perpendicular_line_theorem_l1966_196678

-- Define the intersection point
def intersection_point : ℝ × ℝ := (1, 3)

-- Define the given lines
def line_l1 (x y : ℝ) : Prop := x + y - 4 = 0
def line_l2 (x y : ℝ) : Prop := x - y + 2 = 0
def line_given (x y : ℝ) : Prop := 2*x - y - 1 = 0

-- Define the parallel and perpendicular lines
def line_parallel (x y : ℝ) : Prop := 2*x - y + 1 = 0
def line_perpendicular (x y : ℝ) : Prop := x + 2*y - 7 = 0

-- Theorem for the parallel line
theorem parallel_line_theorem :
  (∀ (x y : ℝ), line_l1 x y ∧ line_l2 x y → (x, y) = intersection_point) →
  line_parallel (intersection_point.1) (intersection_point.2) ∧
  (∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), line_parallel x y ↔ line_given (x + k) (y + k)) :=
sorry

-- Theorem for the perpendicular line
theorem perpendicular_line_theorem :
  (∀ (x y : ℝ), line_l1 x y ∧ line_l2 x y → (x, y) = intersection_point) →
  line_perpendicular (intersection_point.1) (intersection_point.2) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    line_perpendicular x₁ y₁ ∧ line_perpendicular x₂ y₂ ∧ x₁ ≠ x₂ →
    (y₂ - y₁) / (x₂ - x₁) * ((1 : ℝ) / 2) = -1) :=
sorry

end parallel_line_theorem_perpendicular_line_theorem_l1966_196678


namespace correct_product_and_multiplicand_l1966_196620

theorem correct_product_and_multiplicand :
  let incorrect_product : Nat := 1925817
  let correct_product : Nat := 1325813
  let multiplicand : Nat := 2839
  let multiplier : Nat := 467
  incorrect_product ≠ multiplicand * multiplier ∧
  (∃ (a b : Nat), incorrect_product = a * 100000 + 9 * 10000 + b * 1000 + 5 * 100 + 8 * 10 + 7 * 1) ∧
  correct_product = multiplicand * multiplier :=
by sorry

end correct_product_and_multiplicand_l1966_196620


namespace no_solution_to_digit_equation_l1966_196672

theorem no_solution_to_digit_equation :
  ¬ ∃ (L A R N C Y P U S : ℕ),
    (L ≠ 0 ∧ A ≠ 0 ∧ R ≠ 0 ∧ N ≠ 0 ∧ C ≠ 0 ∧ Y ≠ 0 ∧ P ≠ 0 ∧ U ≠ 0 ∧ S ≠ 0) ∧
    (L ≠ A ∧ L ≠ R ∧ L ≠ N ∧ L ≠ C ∧ L ≠ Y ∧ L ≠ P ∧ L ≠ U ∧ L ≠ S ∧
     A ≠ R ∧ A ≠ N ∧ A ≠ C ∧ A ≠ Y ∧ A ≠ P ∧ A ≠ U ∧ A ≠ S ∧
     R ≠ N ∧ R ≠ C ∧ R ≠ Y ∧ R ≠ P ∧ R ≠ U ∧ R ≠ S ∧
     N ≠ C ∧ N ≠ Y ∧ N ≠ P ∧ N ≠ U ∧ N ≠ S ∧
     C ≠ Y ∧ C ≠ P ∧ C ≠ U ∧ C ≠ S ∧
     Y ≠ P ∧ Y ≠ U ∧ Y ≠ S ∧
     P ≠ U ∧ P ≠ S ∧
     U ≠ S) ∧
    (1000 ≤ L * 1000 + A * 100 + R * 10 + N ∧ L * 1000 + A * 100 + R * 10 + N < 10000) ∧
    (100 ≤ A * 100 + C * 10 + A ∧ A * 100 + C * 10 + A < 1000) ∧
    (100 ≤ C * 100 + Y * 10 + P ∧ C * 100 + Y * 10 + P < 1000) ∧
    (100 ≤ R * 100 + U * 10 + S ∧ R * 100 + U * 10 + S < 1000) ∧
    ((L * 1000 + A * 100 + R * 10 + N) - (A * 100 + C * 10 + A)) / 
    ((C * 100 + Y * 10 + P) + (R * 100 + U * 10 + S)) = 
    C^(Y^P) * R^(U^S) := by
  sorry

end no_solution_to_digit_equation_l1966_196672


namespace tangent_line_triangle_area_l1966_196646

/-- A line tangent to the unit circle with intercept sum √3 forms a triangle with area 3/2 --/
theorem tangent_line_triangle_area :
  ∀ (a b : ℝ),
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a*x + b*y = 1) →  -- Line is tangent to unit circle
  a + b = Real.sqrt 3 →                           -- Sum of intercepts is √3
  (1/2) * |a*b| = 3/2 :=                          -- Area of triangle is 3/2
by sorry

end tangent_line_triangle_area_l1966_196646


namespace tan_alpha_on_unit_circle_l1966_196622

theorem tan_alpha_on_unit_circle (α : ℝ) : 
  ((-4/5 : ℝ)^2 + (3/5 : ℝ)^2 = 1) →  -- Point lies on the unit circle
  (∃ (t : ℝ), t > 0 ∧ t * Real.cos α = -4/5 ∧ t * Real.sin α = 3/5) →  -- Point is terminal point of angle α
  Real.tan α = -3/4 := by
sorry

end tan_alpha_on_unit_circle_l1966_196622


namespace peach_tree_average_production_l1966_196602

-- Define the number of apple trees
def num_apple_trees : ℕ := 30

-- Define the production of each apple tree in kg
def apple_production : ℕ := 150

-- Define the number of peach trees
def num_peach_trees : ℕ := 45

-- Define the total mass of fruit harvested in kg
def total_harvest : ℕ := 7425

-- Theorem to prove
theorem peach_tree_average_production :
  (total_harvest - num_apple_trees * apple_production) / num_peach_trees = 65 := by
  sorry

end peach_tree_average_production_l1966_196602


namespace find_P_l1966_196690

theorem find_P : ∃ P : ℕ+, (15 ^ 3 * 25 ^ 3 : ℕ) = 5 ^ 2 * P ^ 3 ∧ P = 375 := by
  sorry

end find_P_l1966_196690


namespace speed_ratio_l1966_196686

/-- The speed of object A in meters per minute -/
def v_A : ℝ := sorry

/-- The speed of object B in meters per minute -/
def v_B : ℝ := sorry

/-- The initial distance of B from point O in meters -/
def initial_distance_B : ℝ := 600

/-- The time when A and B are first equidistant from O in minutes -/
def t1 : ℝ := 4

/-- The time when A and B are again equidistant from O in minutes -/
def t2 : ℝ := 9

/-- Theorem stating that the ratio of A's speed to B's speed is 2/3 -/
theorem speed_ratio :
  v_A / v_B = 2 / 3 :=
sorry

end speed_ratio_l1966_196686


namespace assignment_theorem_l1966_196619

/-- The number of ways to assign 4 distinct tasks to 4 people selected from 6 volunteers -/
def assignment_ways : ℕ := 360

/-- The total number of volunteers -/
def total_volunteers : ℕ := 6

/-- The number of people to be selected -/
def selected_people : ℕ := 4

/-- The number of tasks to be assigned -/
def number_of_tasks : ℕ := 4

theorem assignment_theorem : 
  assignment_ways = (total_volunteers.factorial) / ((total_volunteers - selected_people).factorial) := by
  sorry

end assignment_theorem_l1966_196619


namespace dinner_price_proof_l1966_196648

theorem dinner_price_proof (original_price : ℝ) : 
  (original_price * 0.9 + original_price * 0.15) - 
  (original_price * 0.9 + original_price * 0.9 * 0.15) = 0.36 →
  original_price = 24 := by
sorry

end dinner_price_proof_l1966_196648


namespace females_in_coach_class_l1966_196667

theorem females_in_coach_class 
  (total_passengers : ℕ) 
  (female_percentage : ℚ) 
  (first_class_percentage : ℚ) 
  (first_class_male_ratio : ℚ) 
  (h1 : total_passengers = 120)
  (h2 : female_percentage = 45/100)
  (h3 : first_class_percentage = 10/100)
  (h4 : first_class_male_ratio = 1/3) :
  (total_passengers : ℚ) * female_percentage - 
  (total_passengers : ℚ) * first_class_percentage * (1 - first_class_male_ratio) = 46 := by
sorry

end females_in_coach_class_l1966_196667


namespace pet_store_cages_l1966_196687

/-- Given a pet store scenario, prove the number of cages used -/
theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 78)
  (h2 : sold_puppies = 30)
  (h3 : puppies_per_cage = 8) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 6 := by
  sorry

end pet_store_cages_l1966_196687


namespace football_purchase_problem_l1966_196650

/-- Represents the cost and quantity of footballs --/
structure FootballPurchase where
  costA : ℝ  -- Cost of one A brand football
  costB : ℝ  -- Cost of one B brand football
  quantityB : ℕ  -- Quantity of B brand footballs purchased

/-- Theorem statement for the football purchase problem --/
theorem football_purchase_problem (fp : FootballPurchase) : 
  fp.costB = fp.costA + 30 ∧ 
  2 * fp.costA + 3 * fp.costB = 340 ∧ 
  fp.quantityB ≤ 50 ∧
  54 * (50 - fp.quantityB) + 72 * fp.quantityB = 3060 →
  fp.quantityB = 20 := by
  sorry

#check football_purchase_problem

end football_purchase_problem_l1966_196650


namespace miss_aisha_height_l1966_196639

/-- Miss Aisha's height calculation -/
theorem miss_aisha_height :
  ∀ (h : ℝ),
  h > 0 →
  h / 3 + h / 4 + 25 = h →
  h = 60 := by
sorry

end miss_aisha_height_l1966_196639


namespace digit_removal_theorem_l1966_196604

/-- A function that removes digits from a natural number's decimal representation -/
def digit_removal (n : ℕ) : ℕ := sorry

/-- The number formed by 2005 9's -/
def N_2005 : ℕ := 10^2005 - 1

/-- The number formed by 2008 9's -/
def N_2008 : ℕ := 10^2008 - 1

/-- Theorem stating that N_2005^2009 can be obtained by removing digits from N_2008^2009 -/
theorem digit_removal_theorem :
  ∃ (f : ℕ → ℕ), f (N_2008^2009) = N_2005^2009 ∧ 
  (∀ n : ℕ, f n ≤ n) :=
sorry

end digit_removal_theorem_l1966_196604


namespace exactly_one_item_count_l1966_196600

/-- The number of households with exactly one item (car only, bike only, scooter only, or skateboard only) -/
def households_with_one_item (total : ℕ) (none : ℕ) (car_and_bike : ℕ) (car : ℕ) (bike : ℕ) (scooter : ℕ) (skateboard : ℕ) : ℕ :=
  (car - car_and_bike) + (bike - car_and_bike) + scooter + skateboard

theorem exactly_one_item_count :
  households_with_one_item 120 15 28 52 32 18 8 = 54 := by
  sorry

end exactly_one_item_count_l1966_196600


namespace coin_jar_problem_l1966_196660

/-- Represents the contents and value of a jar of coins. -/
structure CoinJar where
  pennies : ℕ
  nickels : ℕ
  quarters : ℕ
  total_coins : ℕ
  total_value : ℚ

/-- Theorem stating the conditions and result for the coin jar problem. -/
theorem coin_jar_problem (jar : CoinJar) : 
  jar.nickels = 3 * jar.pennies →
  jar.quarters = 4 * jar.nickels →
  jar.total_coins = jar.pennies + jar.nickels + jar.quarters →
  jar.total_coins = 240 →
  jar.total_value = jar.pennies * (1 : ℚ) / 100 + 
                    jar.nickels * (5 : ℚ) / 100 + 
                    jar.quarters * (25 : ℚ) / 100 →
  jar.total_value = (4740 : ℚ) / 100 := by
  sorry

end coin_jar_problem_l1966_196660


namespace flowerbed_width_l1966_196624

theorem flowerbed_width (width length perimeter : ℝ) : 
  width > 0 →
  length > 0 →
  length = 2 * width - 1 →
  perimeter = 2 * length + 2 * width →
  perimeter = 22 →
  width = 4 := by
sorry

end flowerbed_width_l1966_196624


namespace diana_hourly_wage_l1966_196645

/-- Represents Diana's work schedule and earnings --/
structure DianaWork where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculates Diana's hourly wage based on her work schedule and weekly earnings --/
def hourly_wage (d : DianaWork) : ℚ :=
  d.weekly_earnings / (d.monday_hours + d.tuesday_hours + d.wednesday_hours + d.thursday_hours + d.friday_hours)

/-- Theorem stating that Diana's hourly wage is $30 --/
theorem diana_hourly_wage :
  let d : DianaWork := {
    monday_hours := 10,
    tuesday_hours := 15,
    wednesday_hours := 10,
    thursday_hours := 15,
    friday_hours := 10,
    weekly_earnings := 1800
  }
  hourly_wage d = 30 := by sorry

end diana_hourly_wage_l1966_196645


namespace quadratic_inequality_solution_set_l1966_196627

theorem quadratic_inequality_solution_set (m : ℝ) (h : m > 1) :
  {x : ℝ | x^2 + (m - 1) * x - m ≥ 0} = {x : ℝ | x ≤ -m ∨ x ≥ 1} := by
  sorry

end quadratic_inequality_solution_set_l1966_196627


namespace stratified_sampling_proportional_l1966_196659

/-- Represents the number of employees in each title category -/
structure TitleCounts where
  senior : ℕ
  intermediate : ℕ
  junior : ℕ

/-- Represents the number of employees selected in each title category -/
structure SelectedCounts where
  senior : ℕ
  intermediate : ℕ
  junior : ℕ

/-- Checks if the selected counts are proportional to the total counts -/
def isProportionalSelection (total : TitleCounts) (selected : SelectedCounts) (sampleSize : ℕ) : Prop :=
  selected.senior * (total.senior + total.intermediate + total.junior) = 
    total.senior * sampleSize ∧
  selected.intermediate * (total.senior + total.intermediate + total.junior) = 
    total.intermediate * sampleSize ∧
  selected.junior * (total.senior + total.intermediate + total.junior) = 
    total.junior * sampleSize

theorem stratified_sampling_proportional 
  (total : TitleCounts)
  (selected : SelectedCounts)
  (h1 : total.senior = 15)
  (h2 : total.intermediate = 45)
  (h3 : total.junior = 90)
  (h4 : selected.senior = 3)
  (h5 : selected.intermediate = 9)
  (h6 : selected.junior = 18) :
  isProportionalSelection total selected 30 := by
  sorry

end stratified_sampling_proportional_l1966_196659


namespace andre_flowers_l1966_196626

theorem andre_flowers (initial_flowers : ℝ) (total_flowers : ℕ) 
  (h1 : initial_flowers = 67.0)
  (h2 : total_flowers = 157) :
  ↑total_flowers - initial_flowers = 90 := by
  sorry

end andre_flowers_l1966_196626


namespace go_complexity_vs_universe_atoms_l1966_196608

/-- The upper limit of the state space complexity of Go -/
def M : ℝ := 3^361

/-- The total number of atoms of ordinary matter in the observable universe -/
def N : ℝ := 10^80

/-- The logarithm base 10 of 3 -/
def lg3 : ℝ := 0.48

/-- Theorem stating that M/N is approximately equal to 10^93 -/
theorem go_complexity_vs_universe_atoms : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |M / N - 10^93| < ε := by sorry

end go_complexity_vs_universe_atoms_l1966_196608


namespace uncool_family_members_l1966_196699

theorem uncool_family_members (total : ℕ) (cool_dad : ℕ) (cool_mom : ℕ) (cool_sibling : ℕ)
  (cool_dad_and_mom : ℕ) (cool_mom_and_sibling : ℕ) (cool_dad_and_sibling : ℕ)
  (cool_all : ℕ) (h1 : total = 40) (h2 : cool_dad = 18) (h3 : cool_mom = 20)
  (h4 : cool_sibling = 10) (h5 : cool_dad_and_mom = 8) (h6 : cool_mom_and_sibling = 4)
  (h7 : cool_dad_and_sibling = 3) (h8 : cool_all = 2) :
  total - (cool_dad + cool_mom + cool_sibling - cool_dad_and_mom - cool_mom_and_sibling - cool_dad_and_sibling + cool_all) = 5 := by
sorry

end uncool_family_members_l1966_196699


namespace three_points_count_l1966_196680

/-- A configuration of points and lines -/
structure Configuration where
  points : Finset Nat
  lines : Finset (Finset Nat)
  point_count : points.card = 6
  line_count : lines.card = 4
  points_per_line : ∀ l ∈ lines, l.card = 3
  lines_contain_points : ∀ l ∈ lines, l ⊆ points

/-- The number of ways to choose three points on a line in the configuration -/
def three_points_on_line (config : Configuration) : Nat :=
  config.lines.sum (fun l => (l.card.choose 3))

theorem three_points_count (config : Configuration) :
  three_points_on_line config = 24 := by
  sorry

#check three_points_count

end three_points_count_l1966_196680


namespace sqrt_2023_minus_x_meaningful_l1966_196601

theorem sqrt_2023_minus_x_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2023 - x) ↔ x ≤ 2023 := by
  sorry

end sqrt_2023_minus_x_meaningful_l1966_196601


namespace bryans_mineral_samples_l1966_196666

theorem bryans_mineral_samples (samples_per_shelf : ℕ) (total_shelves : ℕ) 
  (h1 : samples_per_shelf = 128) 
  (h2 : total_shelves = 13) : 
  samples_per_shelf * total_shelves = 1664 := by
sorry

end bryans_mineral_samples_l1966_196666


namespace value_of_expression_l1966_196676

theorem value_of_expression (a b : ℝ) (ha : a = 3) (hb : b = 4) :
  (a^3 + 3*b^2) / 9 = 25 / 3 := by sorry

end value_of_expression_l1966_196676


namespace point_in_fourth_quadrant_l1966_196669

theorem point_in_fourth_quadrant (x y : ℝ) : 
  (1 - Complex.I) * x = 1 + y * Complex.I → x > 0 ∧ y < 0 :=
by sorry

end point_in_fourth_quadrant_l1966_196669


namespace wheel_rotation_l1966_196636

/-- Theorem: Rotation of a wheel in radians
  Given a wheel with radius 20 cm rotating counterclockwise,
  if a point on its circumference moves through an arc length of 40 cm,
  then the wheel has rotated 2 radians.
-/
theorem wheel_rotation (radius : ℝ) (arc_length : ℝ) (angle : ℝ) :
  radius = 20 →
  arc_length = 40 →
  angle = arc_length / radius →
  angle = 2 := by
sorry

end wheel_rotation_l1966_196636


namespace susan_gave_out_half_apples_l1966_196668

theorem susan_gave_out_half_apples (frank_apples : ℕ) (susan_apples : ℕ) (total_left : ℕ) :
  frank_apples = 36 →
  susan_apples = 3 * frank_apples →
  total_left = 78 →
  total_left = susan_apples + frank_apples - frank_apples / 3 - susan_apples * (1 - x) →
  x = (1 : ℚ) / 2 :=
by sorry

end susan_gave_out_half_apples_l1966_196668


namespace total_votes_is_120_l1966_196649

/-- The total number of votes cast in a school election -/
def total_votes : ℕ := sorry

/-- The number of votes Brenda received -/
def brenda_votes : ℕ := 45

/-- The fraction of total votes that Brenda received -/
def brenda_fraction : ℚ := 3/8

/-- Theorem stating that the total number of votes is 120 -/
theorem total_votes_is_120 : total_votes = 120 := by
  sorry

end total_votes_is_120_l1966_196649


namespace sister_age_when_john_is_50_l1966_196683

/-- Calculates the sister's age when John reaches a given age -/
def sisterAge (johnsCurrentAge : ℕ) (johnsFutureAge : ℕ) : ℕ :=
  let sisterCurrentAge := 2 * johnsCurrentAge
  sisterCurrentAge + (johnsFutureAge - johnsCurrentAge)

/-- Theorem stating that when John is 50, his sister will be 60, given their current ages -/
theorem sister_age_when_john_is_50 :
  sisterAge 10 50 = 60 := by sorry

end sister_age_when_john_is_50_l1966_196683


namespace simplify_expression_l1966_196694

theorem simplify_expression (x : ℝ) : 2 - (2 * (1 - (2 - (2 * (1 - x))))) = 4 * x := by
  sorry

end simplify_expression_l1966_196694


namespace similar_triangles_leg_ratio_l1966_196634

/-- Given two similar right triangles, where one has legs 12 and 9, and the other has
    corresponding legs y and 7, prove that y = 84/9 -/
theorem similar_triangles_leg_ratio (y : ℝ) : 
  (12 : ℝ) / y = 9 / 7 → y = 84 / 9 := by sorry

end similar_triangles_leg_ratio_l1966_196634


namespace twentieth_bend_is_71_l1966_196635

/-- The function that returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The spiral arrangement of natural numbers where bends occur at prime numbers -/
def spiralBend (n : ℕ) : ℕ := nthPrime n

theorem twentieth_bend_is_71 : spiralBend 20 = 71 := by sorry

end twentieth_bend_is_71_l1966_196635


namespace eight_digit_increasing_count_M_is_correct_l1966_196689

/-- The number of 8-digit positive integers with strictly increasing digits using only 1 through 8 -/
def M : ℕ := 1

/-- The set of valid digits -/
def validDigits : Finset ℕ := Finset.range 8

theorem eight_digit_increasing_count : 
  (Finset.powerset validDigits).filter (fun s => s.card = 8) = {validDigits} := by sorry

/-- The main theorem stating that M is correct -/
theorem M_is_correct : 
  M = Finset.card ((Finset.powerset validDigits).filter (fun s => s.card = 8)) := by sorry

end eight_digit_increasing_count_M_is_correct_l1966_196689


namespace triangle_properties_l1966_196698

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a < t.b ∧ t.b < t.c)
  (h2 : t.a / Real.sin t.A = 2 * t.b / Real.sqrt 3) :
  t.B = π / 3 ∧ 
  (t.a = 2 → t.c = 3 → t.b = Real.sqrt 7 ∧ 
    (1 / 2 : ℝ) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2) :=
by sorry

end triangle_properties_l1966_196698


namespace parallelogram_vector_operations_l1966_196640

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (A B C D : V)

-- Define the parallelogram condition
def is_parallelogram (A B C D : V) : Prop :=
  B - A = C - D ∧ D - A = C - B

-- Theorem statement
theorem parallelogram_vector_operations 
  (h : is_parallelogram A B C D) : 
  (B - A) + (D - A) = C - A ∧ (D - A) - (B - A) = D - B := by
  sorry

end parallelogram_vector_operations_l1966_196640


namespace complex_sum_theorem_l1966_196657

theorem complex_sum_theorem (a b : ℂ) : 
  a = 5 - 3*I → b = 2 + 4*I → a + 2*b = 9 + 5*I := by
  sorry

end complex_sum_theorem_l1966_196657


namespace riverdale_high_quiz_l1966_196628

theorem riverdale_high_quiz (total_contestants : ℕ) (total_students : ℕ) 
  (h1 : total_contestants = 234) 
  (h2 : total_students = 420) : 
  ∃ (freshmen juniors : ℕ), 
    freshmen + juniors = total_students ∧
    (3 * freshmen) / 7 + (3 * juniors) / 4 = total_contestants ∧
    freshmen = 64 ∧ 
    juniors = 356 := by
  sorry

end riverdale_high_quiz_l1966_196628


namespace smaller_cube_side_length_l1966_196614

theorem smaller_cube_side_length (a : ℕ) : 
  a > 0 ∧ 
  6 % a = 0 ∧ 
  6 * a^2 * (216 / a^3) = 432 → 
  a = 3 := by sorry

end smaller_cube_side_length_l1966_196614


namespace complex_equation_solution_l1966_196612

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end complex_equation_solution_l1966_196612


namespace puzzle_assembly_time_l1966_196632

/-- Represents the time taken to assemble a puzzle -/
def assemble_puzzle (initial_pieces : ℕ) (pieces_per_minute : ℕ) : ℕ :=
  (initial_pieces - 1) / (pieces_per_minute - 1)

theorem puzzle_assembly_time :
  let initial_pieces : ℕ := 121
  let two_piece_time : ℕ := 120  -- 2 hours in minutes
  let three_piece_time : ℕ := 60 -- 1 hour in minutes
  assemble_puzzle initial_pieces 2 = two_piece_time →
  assemble_puzzle initial_pieces 3 = three_piece_time :=
by sorry

end puzzle_assembly_time_l1966_196632


namespace mars_500_duration_notation_l1966_196613

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- Rounds a ScientificNotation to a specified number of significant figures -/
def roundToSignificantFigures (sn : ScientificNotation) (figures : ℕ) : ScientificNotation :=
  sorry

theorem mars_500_duration_notation (duration : ℝ) (h : duration = 12480) :
  (toScientificNotation duration).coefficient = 1.248 ∧
  (toScientificNotation duration).exponent = 4 ∧
  (roundToSignificantFigures (toScientificNotation duration) 3).coefficient = 1.25 :=
  sorry

end mars_500_duration_notation_l1966_196613


namespace fraction_of_wall_painted_l1966_196647

/-- 
Given that a wall can be painted in 60 minutes, 
this theorem proves that the fraction of the wall 
painted in 15 minutes is 1/4.
-/
theorem fraction_of_wall_painted 
  (total_time : ℕ) 
  (partial_time : ℕ) 
  (h1 : total_time = 60) 
  (h2 : partial_time = 15) : 
  (partial_time : ℚ) / total_time = 1 / 4 := by
  sorry

end fraction_of_wall_painted_l1966_196647
