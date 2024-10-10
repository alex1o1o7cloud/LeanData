import Mathlib

namespace sqrt_sum_equality_l2997_299796

theorem sqrt_sum_equality : ∃ (a b c : ℕ+), 
  (Real.sqrt 3 + (1 / Real.sqrt 3) + Real.sqrt 11 + (1 / Real.sqrt 11) + Real.sqrt 11 * Real.sqrt 3 = (a * Real.sqrt 3 + b * Real.sqrt 11) / c) ∧
  (∀ (a' b' c' : ℕ+), 
    (Real.sqrt 3 + (1 / Real.sqrt 3) + Real.sqrt 11 + (1 / Real.sqrt 11) + Real.sqrt 11 * Real.sqrt 3 = (a' * Real.sqrt 3 + b' * Real.sqrt 11) / c') → 
    c ≤ c') ∧
  a = 84 ∧ b = 44 ∧ c = 33 := by
  sorry

end sqrt_sum_equality_l2997_299796


namespace employee_hours_proof_l2997_299722

/-- The number of hours worked per week by both employees -/
def hours : ℕ := 40

/-- The hourly rate of the first employee -/
def rate1 : ℕ := 20

/-- The hourly rate of the second employee -/
def rate2 : ℕ := 22

/-- The hourly subsidy for hiring a disabled worker -/
def subsidy : ℕ := 6

/-- The weekly savings by hiring the cheaper employee -/
def savings : ℕ := 160

theorem employee_hours_proof :
  (rate1 * hours) - ((rate2 * hours) - (subsidy * hours)) = savings :=
by sorry

end employee_hours_proof_l2997_299722


namespace inscribed_polygon_sides_l2997_299706

/-- Represents a regular polygon with a given number of sides -/
structure RegularPolygon :=
  (sides : ℕ)

/-- Represents the configuration of polygons in the problem -/
structure PolygonConfiguration :=
  (central : RegularPolygon)
  (inscribed : RegularPolygon)
  (num_inscribed : ℕ)

/-- The sum of interior angles at a contact point -/
def contact_angle_sum : ℝ := 360

/-- The condition that the vertices of the central polygon touch the centers of the inscribed polygons -/
def touches_centers (config : PolygonConfiguration) : Prop :=
  sorry

/-- The theorem stating that in the given configuration, the number of sides of the inscribed polygons must be 6 -/
theorem inscribed_polygon_sides
  (config : PolygonConfiguration)
  (h1 : config.central.sides = 12)
  (h2 : config.num_inscribed = 6)
  (h3 : touches_centers config)
  (h4 : contact_angle_sum = 360) :
  config.inscribed.sides = 6 :=
sorry

end inscribed_polygon_sides_l2997_299706


namespace puzzle_pieces_l2997_299767

theorem puzzle_pieces (total_pieces : ℕ) (edge_difference : ℕ) (non_red_decrease : ℕ) : 
  total_pieces = 91 → 
  edge_difference = 24 → 
  non_red_decrease = 2 → 
  ∃ (red_pieces : ℕ) (non_red_pieces : ℕ), 
    red_pieces + non_red_pieces = total_pieces ∧ 
    non_red_pieces * non_red_decrease = edge_difference ∧
    red_pieces = 79 :=
by sorry

end puzzle_pieces_l2997_299767


namespace sqrt_four_fourth_power_sum_l2997_299742

theorem sqrt_four_fourth_power_sum : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 := by
  sorry

end sqrt_four_fourth_power_sum_l2997_299742


namespace number_with_specific_divisor_sum_l2997_299723

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem number_with_specific_divisor_sum :
  ∀ l m : ℕ,
  let n := 2^l * 3^m
  sum_of_divisors n = 403 →
  n = 144 := by
sorry

end number_with_specific_divisor_sum_l2997_299723


namespace correct_operation_l2997_299752

theorem correct_operation (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by
  sorry

end correct_operation_l2997_299752


namespace hemisphere_with_cylinder_surface_area_l2997_299772

/-- The total surface area of a hemisphere with a cylindrical protrusion -/
theorem hemisphere_with_cylinder_surface_area (r : ℝ) (h : r > 0) :
  let base_area := π * r^2
  let hemisphere_surface := 2 * π * r^2
  let cylinder_surface := 2 * π * r^2
  base_area + hemisphere_surface + cylinder_surface = 5 * π * r^2 := by
sorry

end hemisphere_with_cylinder_surface_area_l2997_299772


namespace five_million_times_eight_million_l2997_299750

theorem five_million_times_eight_million :
  (5000000 : ℕ) * 8000000 = 40000000000000 := by
  sorry

end five_million_times_eight_million_l2997_299750


namespace theta_value_l2997_299797

theorem theta_value : ∃! (Θ : ℕ), Θ ∈ Finset.range 10 ∧ (312 : ℚ) / Θ = 40 + 2 * Θ := by
  sorry

end theta_value_l2997_299797


namespace bakery_flour_usage_l2997_299759

theorem bakery_flour_usage : 
  let wheat_flour : ℝ := 0.2
  let white_flour : ℝ := 0.1
  let rye_flour : ℝ := 0.15
  let almond_flour : ℝ := 0.05
  let rice_flour : ℝ := 0.1
  wheat_flour + white_flour + rye_flour + almond_flour + rice_flour = 0.6 := by
  sorry

end bakery_flour_usage_l2997_299759


namespace min_value_on_circle_l2997_299746

theorem min_value_on_circle (x y : ℝ) (h : (x - 3)^2 + y^2 = 9) :
  ∃ (m : ℝ), (∀ (a b : ℝ), (a - 3)^2 + b^2 = 9 → -2*b - 3*a ≥ m) ∧
             (∃ (c d : ℝ), (c - 3)^2 + d^2 = 9 ∧ -2*d - 3*c = m) ∧
             m = -3 * Real.sqrt 13 - 9 :=
sorry

end min_value_on_circle_l2997_299746


namespace youseff_walk_time_l2997_299730

/-- The number of blocks from Youseff's home to his office -/
def distance : ℕ := 12

/-- The time in seconds it takes Youseff to ride his bike one block -/
def bike_time : ℕ := 20

/-- The additional time in minutes it takes Youseff to walk compared to biking -/
def additional_time : ℕ := 8

/-- The time in seconds it takes Youseff to walk one block -/
def walk_time : ℕ := sorry

theorem youseff_walk_time :
  walk_time = 60 :=
by sorry

end youseff_walk_time_l2997_299730


namespace triangle_sum_bounds_l2997_299703

theorem triangle_sum_bounds (A B C : Real) (hsum : A + B + C = Real.pi) (hpos : 0 < A ∧ 0 < B ∧ 0 < C) :
  let S := Real.sqrt (3 * Real.tan (A/2) * Real.tan (B/2) + 1) +
           Real.sqrt (3 * Real.tan (B/2) * Real.tan (C/2) + 1) +
           Real.sqrt (3 * Real.tan (C/2) * Real.tan (A/2) + 1)
  4 ≤ S ∧ S < 5 := by
  sorry

end triangle_sum_bounds_l2997_299703


namespace equilateral_triangle_condition_l2997_299711

/-- A function that checks if a natural number n satisfies the condition for forming an equilateral triangle with sticks of lengths 1 to n -/
def canFormEquilateralTriangle (n : ℕ) : Prop :=
  n ≥ 5 ∧ (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5)

/-- The sum of the first n natural numbers -/
def sumFirstN (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating the necessary and sufficient condition for forming an equilateral triangle -/
theorem equilateral_triangle_condition (n : ℕ) :
  (∃ (a b c : ℕ), a + b + c = sumFirstN n ∧ a = b ∧ b = c) ↔ canFormEquilateralTriangle n :=
by sorry

end equilateral_triangle_condition_l2997_299711


namespace unique_solution_quadratic_l2997_299787

/-- A quadratic equation qx^2 - 18x + 8 = 0 has only one solution when q = 81/8 -/
theorem unique_solution_quadratic :
  ∃! (x : ℝ), (81/8 : ℝ) * x^2 - 18 * x + 8 = 0 := by sorry

end unique_solution_quadratic_l2997_299787


namespace arc_cover_theorem_l2997_299762

/-- Represents an arc on a circle -/
structure Arc where
  start : ℝ  -- Start angle in degrees
  length : ℝ  -- Length of the arc in degrees

/-- A set of arcs covering a circle -/
def ArcCover := Set Arc

/-- Predicate to check if a set of arcs covers the entire circle -/
def covers_circle (cover : ArcCover) : Prop := sorry

/-- Predicate to check if any single arc in the set covers the entire circle -/
def has_complete_arc (cover : ArcCover) : Prop := sorry

/-- Calculate the total measure of a set of arcs -/
def total_measure (arcs : Set Arc) : ℝ := sorry

/-- Main theorem -/
theorem arc_cover_theorem (cover : ArcCover) 
  (h1 : covers_circle cover) 
  (h2 : ¬ has_complete_arc cover) : 
  ∃ (subset : Set Arc), subset ⊆ cover ∧ covers_circle subset ∧ total_measure subset ≤ 720 := by
  sorry

end arc_cover_theorem_l2997_299762


namespace inconvenient_transportation_probability_l2997_299738

/-- The probability of selecting exactly 4 villages with inconvenient transportation
    out of 10 randomly selected villages from a group of 15 villages,
    where 7 have inconvenient transportation, is equal to 1/30. -/
theorem inconvenient_transportation_probability :
  let total_villages : ℕ := 15
  let inconvenient_villages : ℕ := 7
  let selected_villages : ℕ := 10
  let target_inconvenient : ℕ := 4
  
  Fintype.card {s : Finset (Fin total_villages) //
    s.card = selected_villages ∧
    (s.filter (λ i => i.val < inconvenient_villages)).card = target_inconvenient} /
  Fintype.card {s : Finset (Fin total_villages) // s.card = selected_villages} = 1 / 30 :=
by sorry

end inconvenient_transportation_probability_l2997_299738


namespace palindrome_decomposition_l2997_299709

/-- A word is a list of characters -/
def Word := List Char

/-- A palindrome is a word that reads the same forward and backward -/
def isPalindrome (w : Word) : Prop :=
  w = w.reverse

/-- X is a word of length 2014 consisting of only 'A' and 'B' -/
def X : Word :=
  List.replicate 2014 'A'  -- Example word, actual content doesn't matter for the theorem

/-- Theorem: There exist at least 806 palindromes whose concatenation forms X -/
theorem palindrome_decomposition :
  ∃ (palindromes : List Word),
    palindromes.length ≥ 806 ∧
    (∀ p ∈ palindromes, isPalindrome p) ∧
    palindromes.join = X :=
  sorry


end palindrome_decomposition_l2997_299709


namespace trig_identity_l2997_299724

theorem trig_identity : 
  1 / Real.sin (70 * π / 180) - Real.sqrt 2 / Real.cos (70 * π / 180) = 
  -2 * Real.sin (25 * π / 180) / Real.sin (40 * π / 180) := by
  sorry

end trig_identity_l2997_299724


namespace arithmetic_sequence_sum_problem_l2997_299790

def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

def sum_of_arithmetic_sequence (a₁ d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_problem (i : ℕ) (k : ℕ) :
  (k > 0) →
  (k ≤ 10) →
  (sum_of_arithmetic_sequence 3 2 10 - arithmetic_sequence 3 2 (i + k) = 185) →
  (sum_of_arithmetic_sequence 3 2 10 = 200) :=
sorry

end arithmetic_sequence_sum_problem_l2997_299790


namespace triangle_area_is_two_symmetric_point_correct_line_equal_intercepts_l2997_299734

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

-- Function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ :=
  sorry

-- Function to check if a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  sorry

-- Function to find the symmetric point
def symmetricPoint (p : Point) (l : Line) : Point :=
  sorry

-- Function to check if a line has equal intercepts on both axes
def hasEqualIntercepts (l : Line) : Prop :=
  sorry

-- Theorem 1
theorem triangle_area_is_two :
  let l : Line := { a := 1, b := -1, c := -2 }
  let t : Triangle := { p1 := { x := 0, y := 0 }, p2 := { x := 2, y := 0 }, p3 := { x := 0, y := -2 } }
  triangleArea t = 2 :=
sorry

-- Theorem 2
theorem symmetric_point_correct :
  let l : Line := { a := -1, b := 1, c := 1 }
  let p : Point := { x := 0, y := 2 }
  symmetricPoint p l = { x := 1, y := 1 } :=
sorry

-- Theorem 3
theorem line_equal_intercepts :
  let l : Line := { a := 1, b := 1, c := -2 }
  pointOnLine { x := 1, y := 1 } l ∧ hasEqualIntercepts l :=
sorry

end triangle_area_is_two_symmetric_point_correct_line_equal_intercepts_l2997_299734


namespace newer_car_travels_195_miles_l2997_299770

/-- The distance traveled by the older car -/
def older_car_distance : ℝ := 150

/-- The percentage increase in distance for the newer car -/
def newer_car_percentage : ℝ := 0.30

/-- The distance traveled by the newer car -/
def newer_car_distance : ℝ := older_car_distance * (1 + newer_car_percentage)

/-- Theorem stating that the newer car travels 195 miles -/
theorem newer_car_travels_195_miles :
  newer_car_distance = 195 := by sorry

end newer_car_travels_195_miles_l2997_299770


namespace tangent_line_and_root_condition_l2997_299715

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- State the theorem
theorem tangent_line_and_root_condition (x : ℝ) :
  -- The tangent line at (2, 7)
  (∃ (m b : ℝ), f 2 = 7 ∧ 
    (∀ x, f x = m * x + b) ∧
    m = 12 ∧ b = -17) ∧
  -- Condition for three distinct real roots
  (∀ m : ℝ, (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) ↔
    -3 < m ∧ m < -2) :=
by sorry

end tangent_line_and_root_condition_l2997_299715


namespace sin_10_cos_20_cos_40_l2997_299735

theorem sin_10_cos_20_cos_40 :
  Real.sin (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (40 * π / 180) = 1 / 8 := by
  sorry

end sin_10_cos_20_cos_40_l2997_299735


namespace teachers_at_queen_high_school_l2997_299712

-- Define the given conditions
def total_students : ℕ := 1500
def classes_per_student : ℕ := 6
def classes_per_teacher : ℕ := 5
def students_per_class : ℕ := 35

-- Define the theorem
theorem teachers_at_queen_high_school :
  (total_students * classes_per_student / students_per_class + 4) / classes_per_teacher = 52 := by
  sorry


end teachers_at_queen_high_school_l2997_299712


namespace odd_sum_probability_l2997_299785

/-- The first 15 prime numbers -/
def first_15_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

/-- The number of ways to select 3 primes from the first 15 primes -/
def total_selections : Nat := Nat.choose 15 3

/-- The number of ways to select 3 primes from the first 15 primes such that their sum is odd -/
def odd_sum_selections : Nat := Nat.choose 14 2

theorem odd_sum_probability :
  (odd_sum_selections : ℚ) / total_selections = 1 / 5 := by sorry

end odd_sum_probability_l2997_299785


namespace pirate_treasure_distribution_l2997_299780

theorem pirate_treasure_distribution (x : ℕ) : 
  (x * (x + 1)) / 2 = 4 * x → x + 4 * x = 35 := by
  sorry

end pirate_treasure_distribution_l2997_299780


namespace marble_count_l2997_299719

theorem marble_count (allison_marbles angela_marbles albert_marbles : ℕ) : 
  allison_marbles = 28 →
  angela_marbles = allison_marbles + 8 →
  albert_marbles = 3 * angela_marbles →
  albert_marbles + allison_marbles = 136 :=
by
  sorry

end marble_count_l2997_299719


namespace triangle_properties_l2997_299776

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def altitude_equation (t : Triangle) : ℝ → ℝ → Prop :=
  fun x y => x + 5 * y - 3 = 0

def side_BC_equation (t : Triangle) : ℝ → ℝ → Prop :=
  fun x y => x + 2 * y - 10 = 0

theorem triangle_properties (t : Triangle) :
  t.A = (-2, 1) →
  t.B = (4, 3) →
  (t.C = (3, -2) → altitude_equation t t.A.1 t.A.2) ∧
  (∃ M : ℝ × ℝ, M = (3, 1) ∧ M.1 = (t.A.1 + t.C.1) / 2 ∧ M.2 = (t.A.2 + t.C.2) / 2 →
    side_BC_equation t t.B.1 t.B.2) :=
by
  sorry

end triangle_properties_l2997_299776


namespace tribe_leadership_structure_l2997_299743

theorem tribe_leadership_structure (n : ℕ) (h : n = 12) : 
  n * (n - 1) * (n - 2) * (Nat.choose (n - 3) 3) * (Nat.choose (n - 6) 3) = 2217600 :=
by sorry

end tribe_leadership_structure_l2997_299743


namespace complement_of_A_in_U_l2997_299756

def U : Set ℕ := {1,2,3,4,5,6,7}
def A : Set ℕ := {1,3,5,7}

theorem complement_of_A_in_U : 
  (U \ A) = {2,4,6} := by sorry

end complement_of_A_in_U_l2997_299756


namespace irrationality_of_sqrt_7_l2997_299769

theorem irrationality_of_sqrt_7 :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 7 = (p : ℚ) / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (-2 : ℚ) / 9 = (p : ℚ) / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (1 : ℚ) / 2 = (p : ℚ) / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ -4 = (p : ℚ) / q) :=
by sorry

end irrationality_of_sqrt_7_l2997_299769


namespace special_collection_loans_l2997_299786

theorem special_collection_loans (initial_books : ℕ) (final_books : ℕ) (return_rate : ℚ) :
  initial_books = 75 →
  final_books = 66 →
  return_rate = 70 / 100 →
  ∃ (loaned_books : ℕ), loaned_books = 30 ∧ 
    final_books = initial_books - (1 - return_rate) * loaned_books := by
  sorry

end special_collection_loans_l2997_299786


namespace triangle_arithmetic_sequence_l2997_299732

theorem triangle_arithmetic_sequence (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧ (A + B + C = π) →
  -- a, b, c are sides opposite to angles A, B, C respectively
  (a = 2 * Real.sin A) ∧ (b = 2 * Real.sin B) ∧ (c = 2 * Real.sin C) →
  -- a*cos(C), b*cos(B), c*cos(A) form an arithmetic sequence
  (a * Real.cos C + c * Real.cos A = 2 * b * Real.cos B) →
  -- Conclusions
  (B = π / 3) ∧
  (∀ x, x ∈ Set.Icc (-1/2) (1 + Real.sqrt 3) ↔ 
    ∃ A C, (0 < A) ∧ (A < 2*π/3) ∧ (C = 2*π/3 - A) ∧
    (x = 2 * Real.sin A * Real.sin A + Real.cos (A - C))) := by
  sorry

end triangle_arithmetic_sequence_l2997_299732


namespace grocery_store_costs_l2997_299753

theorem grocery_store_costs (total_cost : ℝ) (salary_fraction : ℝ) (delivery_fraction : ℝ) 
  (h1 : total_cost = 4000)
  (h2 : salary_fraction = 2/5)
  (h3 : delivery_fraction = 1/4) : 
  total_cost * (1 - salary_fraction) * (1 - delivery_fraction) = 1800 := by
  sorry

end grocery_store_costs_l2997_299753


namespace sqrt_inequality_l2997_299757

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) : 
  Real.sqrt a - Real.sqrt (a - 1) < Real.sqrt (a - 2) - Real.sqrt (a - 3) := by
  sorry

end sqrt_inequality_l2997_299757


namespace square_two_minus_sqrt_three_l2997_299740

theorem square_two_minus_sqrt_three (a b : ℚ) :
  (2 - Real.sqrt 3)^2 = a + b * Real.sqrt 3 → a + b = 3 := by
  sorry

end square_two_minus_sqrt_three_l2997_299740


namespace sufficient_not_necessary_l2997_299748

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x < 2}

theorem sufficient_not_necessary : 
  (∀ a : ℝ, a ∈ M → a ∈ N) ∧ 
  (∃ a : ℝ, a ∈ N ∧ a ∉ M) :=
by sorry

end sufficient_not_necessary_l2997_299748


namespace linear_function_properties_l2997_299747

/-- A linear function passing through two points and intersecting a horizontal line -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := fun x ↦ k * x + b

theorem linear_function_properties 
  (k b : ℝ) 
  (h_k : k ≠ 0)
  (h_point_A : LinearFunction k b 0 = 1)
  (h_point_B : LinearFunction k b 1 = 2)
  (h_intersect : ∃ x, LinearFunction k b x = 4) :
  (k = 1 ∧ b = 1) ∧ 
  (∃ x, x = 3 ∧ LinearFunction k b x = 4) ∧
  (∀ x, x < 3 → (2/3 * x + 2 > LinearFunction k b x ∧ 2/3 * x + 2 < 4)) := by
  sorry

#check linear_function_properties

end linear_function_properties_l2997_299747


namespace dans_initial_money_l2997_299754

/-- Dan's initial amount of money -/
def initial_money : ℕ := sorry

/-- Cost of the candy bar -/
def candy_cost : ℕ := 6

/-- Cost of the chocolate -/
def chocolate_cost : ℕ := 3

/-- Theorem stating that Dan's initial money is equal to the total spent -/
theorem dans_initial_money :
  initial_money = candy_cost + chocolate_cost ∧ candy_cost = chocolate_cost + 3 := by
  sorry

end dans_initial_money_l2997_299754


namespace brick_height_l2997_299708

/-- The surface area of a rectangular prism given its length, width, and height -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: The height of a rectangular prism with length 8 cm, width 6 cm, 
    and surface area 152 cm² is 2 cm -/
theorem brick_height : 
  ∃ (h : ℝ), h > 0 ∧ surface_area 8 6 h = 152 → h = 2 := by
sorry

end brick_height_l2997_299708


namespace negation_of_existence_exp_minus_x_minus_one_negation_l2997_299729

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem exp_minus_x_minus_one_negation :
  (¬∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by sorry

end negation_of_existence_exp_minus_x_minus_one_negation_l2997_299729


namespace prime_sequence_implies_composite_l2997_299704

theorem prime_sequence_implies_composite (p : ℕ) 
  (h1 : Nat.Prime p)
  (h2 : Nat.Prime (3*p + 2))
  (h3 : Nat.Prime (5*p + 4))
  (h4 : Nat.Prime (7*p + 6))
  (h5 : Nat.Prime (9*p + 8))
  (h6 : Nat.Prime (11*p + 10)) :
  ¬(Nat.Prime (6*p + 11)) :=
by sorry

end prime_sequence_implies_composite_l2997_299704


namespace twirly_tea_cups_capacity_l2997_299725

/-- The 'Twirly Tea Cups' ride problem -/
theorem twirly_tea_cups_capacity 
  (total_capacity : ℕ) 
  (num_teacups : ℕ) 
  (h1 : total_capacity = 63) 
  (h2 : num_teacups = 7) : 
  total_capacity / num_teacups = 9 := by
  sorry

end twirly_tea_cups_capacity_l2997_299725


namespace min_value_f_in_interval_l2997_299714

def f (x : ℝ) : ℝ := x^4 - 4*x + 3

theorem min_value_f_in_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 3 ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-2 : ℝ) 3 → f x ≤ f y) ∧
  f x = 0 :=
sorry

end min_value_f_in_interval_l2997_299714


namespace line_equation_in_triangle_l2997_299739

/-- Given a line passing through (-2b, 0) forming a triangular region in the second quadrant with area S, 
    its equation is 2Sx - b^2y + 4bS = 0 --/
theorem line_equation_in_triangle (b S : ℝ) (h_b : b ≠ 0) (h_S : S > 0) : 
  ∃ (m k : ℝ), 
    (∀ (x y : ℝ), y = m * x + k → 
      (x = -2*b ∧ y = 0) ∨ 
      (x = 0 ∧ y = 0) ∨ 
      (x = 0 ∧ y > 0)) ∧
    (1/2 * 2*b * (S/b) = S) ∧
    (∀ (x y : ℝ), 2*S*x - b^2*y + 4*b*S = 0 ↔ y = m * x + k) :=
by sorry

end line_equation_in_triangle_l2997_299739


namespace bill_donut_order_combinations_l2997_299744

/-- The number of combinations for selecting donuts satisfying the given conditions -/
def donut_combinations (total_donuts : ℕ) (donut_types : ℕ) (types_to_select : ℕ) : ℕ :=
  (donut_types.choose types_to_select) * 
  ((total_donuts - types_to_select + types_to_select - 1).choose (types_to_select - 1))

/-- Theorem stating that the number of combinations for Bill's donut order is 100 -/
theorem bill_donut_order_combinations : 
  donut_combinations 7 5 4 = 100 := by
sorry

end bill_donut_order_combinations_l2997_299744


namespace faulty_odometer_distance_l2997_299731

/-- Represents an odometer that skips certain digits --/
structure SkippingOdometer where
  reading : Nat
  skipped_digits : List Nat

/-- Calculates the actual distance traveled given a skipping odometer --/
def actual_distance (o : SkippingOdometer) : Nat :=
  sorry

/-- The theorem to be proved --/
theorem faulty_odometer_distance :
  let o : SkippingOdometer := { reading := 3509, skipped_digits := [4, 6] }
  actual_distance o = 2964 :=
sorry

end faulty_odometer_distance_l2997_299731


namespace arithmetic_sequence_common_difference_l2997_299736

theorem arithmetic_sequence_common_difference :
  let a : ℕ → ℤ := λ n => 2 - 3 * n
  ∀ n : ℕ, a (n + 1) - a n = -3 :=
by
  sorry

end arithmetic_sequence_common_difference_l2997_299736


namespace fraction_decrease_l2997_299798

theorem fraction_decrease (m n : ℝ) (h : m ≠ 0 ∧ n ≠ 0) : 
  (3*m + 3*n) / ((3*m) * (3*n)) = (1/3) * ((m + n) / (m * n)) := by
  sorry

end fraction_decrease_l2997_299798


namespace angle_subtraction_l2997_299720

/-- Represents an angle in degrees, minutes, and seconds -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)
  (seconds : ℕ)

/-- Converts an Angle to seconds -/
def angleToSeconds (a : Angle) : ℕ :=
  a.degrees * 3600 + a.minutes * 60 + a.seconds

/-- Converts seconds to an Angle -/
def secondsToAngle (s : ℕ) : Angle :=
  let d := s / 3600
  let m := (s % 3600) / 60
  let sec := s % 60
  ⟨d, m, sec⟩

theorem angle_subtraction :
  let a₁ : Angle := ⟨90, 0, 0⟩
  let a₂ : Angle := ⟨78, 28, 56⟩
  let result : Angle := ⟨11, 31, 4⟩
  angleToSeconds a₁ - angleToSeconds a₂ = angleToSeconds result := by
  sorry

end angle_subtraction_l2997_299720


namespace remainder_3125_div_98_l2997_299707

theorem remainder_3125_div_98 : 3125 % 98 = 87 := by
  sorry

end remainder_3125_div_98_l2997_299707


namespace addition_of_like_terms_l2997_299771

theorem addition_of_like_terms (a : ℝ) : 2 * a + a = 3 * a := by
  sorry

end addition_of_like_terms_l2997_299771


namespace min_value_3a_2b_min_value_3a_2b_achieved_l2997_299779

theorem min_value_3a_2b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (a + b)⁻¹ + (a - b)⁻¹ = 1) : 
  ∀ x y : ℝ, x > 0 → y > 0 → (x + y)⁻¹ + (x - y)⁻¹ = 1 → 3*x + 2*y ≥ 3 + Real.sqrt 5 :=
by sorry

theorem min_value_3a_2b_achieved (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (a + b)⁻¹ + (a - b)⁻¹ = 1) : 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x + y)⁻¹ + (x - y)⁻¹ = 1 ∧ 3*x + 2*y = 3 + Real.sqrt 5 :=
by sorry

end min_value_3a_2b_min_value_3a_2b_achieved_l2997_299779


namespace sum_reciprocal_inequality_l2997_299716

theorem sum_reciprocal_inequality (u v w : ℝ) (h : u + v + w = 3) :
  1 / (u^2 + 7) + 1 / (v^2 + 7) + 1 / (w^2 + 7) ≤ 3 / 8 := by
  sorry

end sum_reciprocal_inequality_l2997_299716


namespace negative_integer_solution_to_inequality_l2997_299761

theorem negative_integer_solution_to_inequality :
  ∀ x : ℤ, (x < 0 ∧ -2 * x < 4) ↔ x = -1 :=
sorry

end negative_integer_solution_to_inequality_l2997_299761


namespace add_base6_35_14_l2997_299702

/-- Converts a base 6 number to base 10 --/
def base6_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 6 --/
def base10_to_base6 (n : ℕ) : ℕ := sorry

/-- Addition in base 6 --/
def add_base6 (a b : ℕ) : ℕ :=
  base10_to_base6 (base6_to_base10 a + base6_to_base10 b)

theorem add_base6_35_14 : add_base6 35 14 = 53 := by sorry

end add_base6_35_14_l2997_299702


namespace candy_distribution_solution_l2997_299737

def candy_distribution (n : ℕ) : Prop :=
  let initial_candy : ℕ := 120
  let first_phase_passes : ℕ := 40
  let first_phase_candy := first_phase_passes
  let second_phase_candy := initial_candy - first_phase_candy
  let total_passes := first_phase_passes + (second_phase_candy / 2)
  (n ∣ total_passes) ∧ (n > 0) ∧ (n ≤ total_passes)

theorem candy_distribution_solution :
  candy_distribution 40 ∧ ∀ m : ℕ, m ≠ 40 → ¬(candy_distribution m) :=
sorry

end candy_distribution_solution_l2997_299737


namespace fraction_subtraction_l2997_299741

theorem fraction_subtraction : 3 / 5 - (2 / 15 + 1 / 3) = 2 / 15 := by
  sorry

end fraction_subtraction_l2997_299741


namespace cricket_team_average_age_l2997_299760

theorem cricket_team_average_age 
  (n : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age_diff : ℕ) 
  (h1 : n = 11) 
  (h2 : captain_age = 28) 
  (h3 : wicket_keeper_age_diff = 3) : 
  ∃ (team_avg : ℚ), 
    team_avg = 25 ∧ 
    (n : ℚ) * team_avg = 
      (captain_age : ℚ) + 
      ((captain_age : ℚ) + wicket_keeper_age_diff) + 
      ((n - 2 : ℚ) * (team_avg - 1)) := by
  sorry

end cricket_team_average_age_l2997_299760


namespace chair_count_difference_l2997_299726

/-- Represents the number of chairs of each color in a classroom. -/
structure ClassroomChairs where
  blue : Nat
  green : Nat
  white : Nat

/-- Theorem about the difference in chair counts in a classroom. -/
theorem chair_count_difference 
  (chairs : ClassroomChairs) 
  (h1 : chairs.blue = 10)
  (h2 : chairs.green = 3 * chairs.blue)
  (h3 : chairs.blue + chairs.green + chairs.white = 67) :
  chairs.blue + chairs.green - chairs.white = 13 := by
  sorry


end chair_count_difference_l2997_299726


namespace largest_number_l2997_299751

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

def number_85_9 : Nat := to_decimal [8, 5] 9
def number_210_6 : Nat := to_decimal [2, 1, 0] 6
def number_1000_4 : Nat := to_decimal [1, 0, 0, 0] 4
def number_11111_2 : Nat := to_decimal [1, 1, 1, 1, 1] 2

theorem largest_number :
  number_210_6 > number_85_9 ∧
  number_210_6 > number_1000_4 ∧
  number_210_6 > number_11111_2 := by
  sorry

end largest_number_l2997_299751


namespace jason_pears_count_l2997_299777

/-- The number of pears Jason picked -/
def jason_pears : ℕ := 105 - (47 + 12)

/-- The total number of pears picked -/
def total_pears : ℕ := 105

/-- The number of pears Keith picked -/
def keith_pears : ℕ := 47

/-- The number of pears Mike picked -/
def mike_pears : ℕ := 12

theorem jason_pears_count : jason_pears = 46 := by sorry

end jason_pears_count_l2997_299777


namespace two_white_prob_correct_at_least_one_white_prob_correct_l2997_299710

/-- Represents the outcome of drawing a ball -/
inductive Ball
| White
| Black

/-- Represents the state of the bag of balls -/
structure BagState where
  total : Nat
  white : Nat
  black : Nat

/-- The initial state of the bag -/
def initialBag : BagState :=
  { total := 5, white := 3, black := 2 }

/-- Calculates the probability of drawing two white balls in succession -/
def probTwoWhite (bag : BagState) : Rat :=
  (bag.white / bag.total) * ((bag.white - 1) / (bag.total - 1))

/-- Calculates the probability of drawing at least one white ball in two draws -/
def probAtLeastOneWhite (bag : BagState) : Rat :=
  1 - (bag.black / bag.total) * ((bag.black - 1) / (bag.total - 1))

theorem two_white_prob_correct :
  probTwoWhite initialBag = 3 / 10 := by sorry

theorem at_least_one_white_prob_correct :
  probAtLeastOneWhite initialBag = 9 / 10 := by sorry

end two_white_prob_correct_at_least_one_white_prob_correct_l2997_299710


namespace initial_necklaces_count_l2997_299793

theorem initial_necklaces_count (initial_earrings : ℕ) 
  (total_jewelry : ℕ) : 
  initial_earrings = 15 →
  total_jewelry = 57 →
  ∃ (initial_necklaces : ℕ),
    initial_necklaces = 15 ∧
    2 * initial_necklaces + initial_earrings + 
    (2/3 : ℚ) * initial_earrings + 
    (1/5 : ℚ) * ((2/3 : ℚ) * initial_earrings) = total_jewelry :=
by sorry

end initial_necklaces_count_l2997_299793


namespace pyramid_sphere_inequality_l2997_299713

/-- A regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  /-- The radius of the circumscribed sphere -/
  R : ℝ
  /-- The radius of the inscribed sphere -/
  r : ℝ
  /-- R is positive -/
  R_pos : 0 < R
  /-- r is positive -/
  r_pos : 0 < r

/-- 
For a regular quadrilateral pyramid inscribed in a sphere with radius R 
and circumscribed around a sphere with radius r, R ≥ (√2 + 1)r holds.
-/
theorem pyramid_sphere_inequality (p : RegularQuadrilateralPyramid) : 
  p.R ≥ (Real.sqrt 2 + 1) * p.r := by
  sorry

end pyramid_sphere_inequality_l2997_299713


namespace tangent_condition_intersection_condition_l2997_299745

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

-- Define the line
def line (x y m : ℝ) : Prop := y = x + m

-- Tangent condition
theorem tangent_condition (m : ℝ) : 
  (∃! p : ℝ × ℝ, ellipse p.1 p.2 ∧ line p.1 p.2 m) ↔ m^2 = 5 :=
sorry

-- Intersection condition
theorem intersection_condition (m : ℝ) :
  (∃ p q : ℝ × ℝ, p ≠ q ∧ 
   ellipse p.1 p.2 ∧ ellipse q.1 q.2 ∧ 
   line p.1 p.2 m ∧ line q.1 q.2 m ∧
   (p.1 - q.1)^2 + (p.2 - q.2)^2 = 4) ↔ 16 * m^2 = 30 :=
sorry

end tangent_condition_intersection_condition_l2997_299745


namespace arithmetic_sequence_sum_l2997_299768

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 4 + a 5 = 12) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 :=
by
  sorry

end arithmetic_sequence_sum_l2997_299768


namespace ellipse_C_and_point_T_l2997_299778

/-- The ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The circle M -/
def circle_M (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 4 = 0

/-- The line l passing through (1,0) and intersecting C at A and B -/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  y = m * (x - 1)

/-- The angle OTA equals OTB -/
def angle_OTA_eq_OTB (t : ℝ) (xA yA xB yB : ℝ) : Prop :=
  (yA / (xA - t)) + (yB / (xB - t)) = 0

theorem ellipse_C_and_point_T :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ b = 1 ∧
  (∃ (c : ℝ), c^2 = a^2 - b^2 ∧
    (∀ x y : ℝ, x + c*y - c = 0 → circle_M x y)) →
  (∀ x y : ℝ, ellipse_C x y a b ↔ x^2/4 + y^2 = 1) ∧
  (∃ t : ℝ, t = 4 ∧
    ∀ m xA yA xB yB : ℝ,
      line_l m xA yA ∧ line_l m xB yB ∧
      ellipse_C xA yA a b ∧ ellipse_C xB yB a b →
      angle_OTA_eq_OTB t xA yA xB yB) :=
by sorry

end ellipse_C_and_point_T_l2997_299778


namespace cosine_squared_inequality_l2997_299794

theorem cosine_squared_inequality (x y : ℝ) : 
  (Real.cos (x - y))^2 ≤ 4 * (1 - Real.sin x * Real.cos y) * (1 - Real.cos x * Real.sin y) := by
  sorry

end cosine_squared_inequality_l2997_299794


namespace power_four_times_four_equals_square_to_fourth_l2997_299728

theorem power_four_times_four_equals_square_to_fourth (a : ℝ) : a^4 * a^4 = (a^2)^4 := by
  sorry

end power_four_times_four_equals_square_to_fourth_l2997_299728


namespace isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l2997_299789

/-- An isosceles triangle with congruent sides of length 8 and perimeter 25 has a base of length 9. -/
theorem isosceles_triangle_base_length : ℝ → Prop :=
  fun base =>
    let congruent_side := 8
    let perimeter := 25
    (2 * congruent_side + base = perimeter) →
    base = 9

/-- Proof of the theorem -/
theorem isosceles_triangle_base_length_proof : isosceles_triangle_base_length 9 := by
  sorry

end isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l2997_299789


namespace f_properties_l2997_299733

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 + a

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.log x - 2 * a * x + 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f_derivative a x + (2 * a - 1) * x

theorem f_properties (a : ℝ) :
  (∀ x > 0, HasDerivAt (f a) (f_derivative a x) x) ∧
  (∀ x > 0, HasDerivAt (g a) ((1 - x) / x) x) ∧
  (∃ x₀ > 0, IsLocalMax (g a) x₀ ∧ g a x₀ = 0) ∧
  (∀ x₀ > 0, ¬ IsLocalMin (g a) x₀) ∧
  (∀ x > 1, f a x < 0) ↔ a ≥ (1 / 2 : ℝ) :=
by sorry

end f_properties_l2997_299733


namespace rectangle_division_possible_l2997_299782

theorem rectangle_division_possible : ∃ (w1 h1 w2 h2 w3 h3 : ℕ+), 
  (w1 * h1 : ℕ) + (w2 * h2 : ℕ) + (w3 * h3 : ℕ) = 100 * 70 ∧
  (w1 : ℕ) ≤ 100 ∧ (h1 : ℕ) ≤ 70 ∧
  (w2 : ℕ) ≤ 100 ∧ (h2 : ℕ) ≤ 70 ∧
  (w3 : ℕ) ≤ 100 ∧ (h3 : ℕ) ≤ 70 ∧
  2 * (w1 * h1 : ℕ) = (w2 * h2 : ℕ) ∧
  2 * (w2 * h2 : ℕ) = (w3 * h3 : ℕ) := by
  sorry

#check rectangle_division_possible

end rectangle_division_possible_l2997_299782


namespace arithmetic_sequence_ninth_term_l2997_299792

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The ninth term of an arithmetic sequence is 32, given that its third term is 20 and its sixth term is 26. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℕ)
  (h_arith : ArithmeticSequence a)
  (h_third : a 3 = 20)
  (h_sixth : a 6 = 26) :
  a 9 = 32 := by
  sorry


end arithmetic_sequence_ninth_term_l2997_299792


namespace smallest_cube_factor_l2997_299781

theorem smallest_cube_factor (z : ℕ) (hz : z.Prime ∧ z > 7) :
  let y := 19408850
  (∀ k : ℕ, k > 0 ∧ k < y → ¬∃ n : ℕ, (31360 * z) * k = n^3) ∧
  ∃ n : ℕ, (31360 * z) * y = n^3 :=
sorry

end smallest_cube_factor_l2997_299781


namespace clock_angle_l2997_299758

theorem clock_angle (hour_hand_angle hour_hand_movement minute_hand_movement : ℝ) :
  hour_hand_angle = 90 →
  hour_hand_movement = 15 →
  minute_hand_movement = 180 →
  180 - hour_hand_angle - hour_hand_movement = 75 :=
by sorry

end clock_angle_l2997_299758


namespace triangle_side_length_l2997_299721

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  a = 3 → b = Real.sqrt 13 → B = π / 3 → 
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) → c = 4 := by
  sorry

end triangle_side_length_l2997_299721


namespace expression_simplification_l2997_299705

theorem expression_simplification (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  ((x * y + 2) * (x * y - 2) + (x * y - 2)^2) / (x * y) = -8 := by
  sorry

end expression_simplification_l2997_299705


namespace simplify_exponential_fraction_l2997_299788

theorem simplify_exponential_fraction (n : ℕ) :
  (3^(n+3) - 3*(3^n)) / (3*(3^(n+2))) = 8/3 := by
  sorry

end simplify_exponential_fraction_l2997_299788


namespace ad_ratio_l2997_299795

/-- Represents the number of ads on each web page -/
structure WebPages :=
  (page1 : ℕ)
  (page2 : ℕ)
  (page3 : ℕ)
  (page4 : ℕ)

/-- Conditions of the problem -/
def adConditions (w : WebPages) : Prop :=
  w.page1 = 12 ∧
  w.page2 = 2 * w.page1 ∧
  w.page3 = w.page2 + 24 ∧
  2 * 68 = 3 * (w.page1 + w.page2 + w.page3 + w.page4)

/-- The theorem to be proved -/
theorem ad_ratio (w : WebPages) :
  adConditions w →
  (w.page4 : ℚ) / w.page2 = 3 / 4 := by
  sorry

end ad_ratio_l2997_299795


namespace taxi_theorem_l2997_299774

def taxi_distances : List ℤ := [5, 2, -4, -3, 6]
def fuel_rate : ℚ := 0.3
def base_fare : ℚ := 8
def base_distance : ℚ := 3
def extra_fare_rate : ℚ := 1.6

def final_position (distances : List ℤ) : ℤ :=
  distances.sum

def total_distance (distances : List ℤ) : ℕ :=
  distances.map Int.natAbs |>.sum

def fuel_consumed (distances : List ℤ) (rate : ℚ) : ℚ :=
  rate * (total_distance distances : ℚ)

def fare_for_distance (d : ℚ) : ℚ :=
  if d ≤ base_distance then base_fare
  else base_fare + extra_fare_rate * (d - base_distance)

def total_fare (distances : List ℤ) : ℚ :=
  distances.map (fun d => fare_for_distance (Int.natAbs d : ℚ)) |>.sum

theorem taxi_theorem :
  final_position taxi_distances = 6 ∧
  fuel_consumed taxi_distances fuel_rate = 6 ∧
  total_fare taxi_distances = 49.6 := by
  sorry

end taxi_theorem_l2997_299774


namespace corresponding_angles_not_always_equal_l2997_299700

-- Define the concept of corresponding angles
def corresponding_angles (α β : ℝ) : Prop := sorry

-- Theorem stating that the proposition "corresponding angles are equal" is false
theorem corresponding_angles_not_always_equal :
  ¬ ∀ α β : ℝ, corresponding_angles α β → α = β :=
sorry

end corresponding_angles_not_always_equal_l2997_299700


namespace imaginary_part_reciprocal_l2997_299791

theorem imaginary_part_reciprocal (a : ℝ) : Complex.im (1 / (a - Complex.I)) = 1 / (1 + a^2) := by
  sorry

end imaginary_part_reciprocal_l2997_299791


namespace octagon_interior_angle_l2997_299784

/-- The measure of each interior angle in a regular octagon -/
def interior_angle_octagon : ℝ := 135

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Formula for the sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

theorem octagon_interior_angle :
  interior_angle_octagon = (sum_interior_angles octagon_sides) / octagon_sides :=
by sorry

end octagon_interior_angle_l2997_299784


namespace hans_reservation_deposit_l2997_299755

/-- Calculates the total deposit for a restaurant reservation with given guest counts and fees -/
def calculate_deposit (num_kids num_adults num_seniors num_students num_employees : ℕ)
  (flat_fee kid_fee adult_fee senior_fee student_fee employee_fee : ℚ)
  (service_charge_rate : ℚ) : ℚ :=
  let base_deposit := flat_fee + 
    num_kids * kid_fee + 
    num_adults * adult_fee + 
    num_seniors * senior_fee + 
    num_students * student_fee + 
    num_employees * employee_fee
  let service_charge := base_deposit * service_charge_rate
  base_deposit + service_charge

/-- The total deposit for Hans' reservation is $128.63 -/
theorem hans_reservation_deposit :
  calculate_deposit 2 8 5 3 2 30 3 6 4 (9/2) (5/2) (1/20) = 12863/100 := by
  sorry

end hans_reservation_deposit_l2997_299755


namespace infinitely_many_planes_through_collinear_points_lines_in_different_planes_may_be_skew_unique_plane_through_parallel_lines_l2997_299766

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (on_line : Point → Line → Prop)
variable (on_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Statement B
theorem infinitely_many_planes_through_collinear_points 
  (A B C : Point) (m : Line) 
  (h1 : on_line A m) (h2 : on_line B m) (h3 : on_line C m) :
  ∃ (P : Set Plane), Infinite P ∧ ∀ p ∈ P, on_plane m p :=
sorry

-- Statement C
theorem lines_in_different_planes_may_be_skew 
  (m n : Line) (α β : Plane) 
  (h1 : on_plane m α) (h2 : on_plane n β) :
  ∃ (skew : Line → Line → Prop), skew m n :=
sorry

-- Statement D
theorem unique_plane_through_parallel_lines 
  (m n : Line) (h : parallel m n) :
  ∃! p : Plane, on_plane m p ∧ on_plane n p :=
sorry

end infinitely_many_planes_through_collinear_points_lines_in_different_planes_may_be_skew_unique_plane_through_parallel_lines_l2997_299766


namespace john_caffeine_consumption_l2997_299764

/-- The amount of caffeine John consumed from two energy drinks and a caffeine pill -/
theorem john_caffeine_consumption (first_drink_oz : ℝ) (first_drink_caffeine : ℝ) 
  (second_drink_oz : ℝ) (second_drink_caffeine_multiplier : ℝ) :
  first_drink_oz = 12 ∧ 
  first_drink_caffeine = 250 ∧ 
  second_drink_oz = 2 ∧ 
  second_drink_caffeine_multiplier = 3 →
  (let first_drink_caffeine_per_oz := first_drink_caffeine / first_drink_oz
   let second_drink_caffeine_per_oz := first_drink_caffeine_per_oz * second_drink_caffeine_multiplier
   let second_drink_caffeine := second_drink_caffeine_per_oz * second_drink_oz
   let total_drinks_caffeine := first_drink_caffeine + second_drink_caffeine
   let pill_caffeine := total_drinks_caffeine
   let total_caffeine := total_drinks_caffeine + pill_caffeine
   total_caffeine = 750) :=
by sorry

end john_caffeine_consumption_l2997_299764


namespace club_equation_solution_l2997_299749

/-- Define the ♣ operation -/
def club (A B : ℝ) : ℝ := 3 * A + 2 * B + 7

/-- Theorem stating that 17 is the unique solution to A ♣ 6 = 70 -/
theorem club_equation_solution :
  ∃! A : ℝ, club A 6 = 70 ∧ A = 17 := by
  sorry

end club_equation_solution_l2997_299749


namespace quadratic_inequality_condition_l2997_299701

theorem quadratic_inequality_condition (x : ℝ) :
  (∀ x, 0 < x ∧ x < 2 → x^2 - x - 6 < 0) ∧
  (∃ x, x^2 - x - 6 < 0 ∧ ¬(0 < x ∧ x < 2)) :=
sorry

end quadratic_inequality_condition_l2997_299701


namespace tangent_line_sum_l2997_299773

/-- Given a function f: ℝ → ℝ with a tangent line y = 1/2 * x + 2 at x = 1,
    prove that f(1) + f'(1) = 3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x, f 1 + (deriv f 1) * (x - 1) = 1/2 * x + 2) : 
    f 1 + deriv f 1 = 3 := by
  sorry

end tangent_line_sum_l2997_299773


namespace boys_tried_out_l2997_299717

/-- The number of boys who tried out for the basketball team -/
def num_boys : ℕ := sorry

/-- The number of girls who tried out for the basketball team -/
def num_girls : ℕ := 39

/-- The number of students who got called back -/
def called_back : ℕ := 26

/-- The number of students who didn't make the cut -/
def didnt_make_cut : ℕ := 17

theorem boys_tried_out : num_boys = 4 := by
  sorry

end boys_tried_out_l2997_299717


namespace triangular_square_iff_pell_solution_l2997_299718

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- A solution to the Pell's equation X^2 - 8Y^2 = 1 -/
def pell_solution (x : ℕ) : Prop := ∃ y : ℕ, x^2 - 8*y^2 = 1

/-- The main theorem: a triangular number is a perfect square iff it has the form (x^2 - 1)/8
    where x is a solution to the Pell's equation X^2 - 8Y^2 = 1 -/
theorem triangular_square_iff_pell_solution :
  ∀ n : ℕ, (∃ k : ℕ, triangular_number n = k^2) ↔ 
  (∃ x : ℕ, pell_solution x ∧ triangular_number n = (x^2 - 1) / 8) :=
sorry

end triangular_square_iff_pell_solution_l2997_299718


namespace coefficients_of_our_equation_l2997_299727

/-- Given a quadratic equation ax^2 + bx + c = 0, 
    returns the coefficients a, b, and c as a triple -/
def quadratic_coefficients (a b c : ℚ) : ℚ × ℚ × ℚ := (a, b, c)

/-- The quadratic equation 3x^2 - 6x - 1 = 0 -/
def our_equation := quadratic_coefficients 3 (-6) (-1)

theorem coefficients_of_our_equation :
  our_equation.2.1 = -6 ∧ our_equation.2.2 = -1 := by
  sorry

end coefficients_of_our_equation_l2997_299727


namespace dream_cost_in_illusions_l2997_299765

/-- Represents the price of an item in the dream market -/
structure DreamPrice where
  illusion : ℚ
  nap : ℚ
  nightmare : ℚ
  dream : ℚ

/-- The dream market pricing system satisfies the given conditions -/
def is_valid_pricing (p : DreamPrice) : Prop :=
  7 * p.illusion + 2 * p.nap + p.nightmare = 4 * p.dream ∧
  4 * p.illusion + 4 * p.nap + 2 * p.nightmare = 7 * p.dream

/-- The cost of one dream is equal to 10 illusions -/
theorem dream_cost_in_illusions (p : DreamPrice) : 
  is_valid_pricing p → p.dream = 10 * p.illusion := by
  sorry

end dream_cost_in_illusions_l2997_299765


namespace arithmetic_calculations_l2997_299775

theorem arithmetic_calculations :
  (14 - 25 + 12 - 17 = -16) ∧
  ((1/2 + 5/6 - 7/12) / (-1/36) = -27) :=
by sorry

end arithmetic_calculations_l2997_299775


namespace job_crop_production_l2997_299783

/-- Represents the land allocation of Job's farm --/
structure FarmLand where
  total : ℕ
  house_and_machinery : ℕ
  future_expansion : ℕ
  cattle : ℕ

/-- Calculates the land used for crop production --/
def crop_production (farm : FarmLand) : ℕ :=
  farm.total - (farm.house_and_machinery + farm.future_expansion + farm.cattle)

/-- Theorem stating that Job's land used for crop production is 70 hectares --/
theorem job_crop_production :
  let job_farm : FarmLand := {
    total := 150,
    house_and_machinery := 25,
    future_expansion := 15,
    cattle := 40
  }
  crop_production job_farm = 70 := by sorry

end job_crop_production_l2997_299783


namespace sum_of_integers_problem_l2997_299763

theorem sum_of_integers_problem : ∃ (a b : ℕ), 
  (a > 0) ∧ (b > 0) ∧
  (a * b + a + b - (a - b) = 120) ∧
  (Nat.gcd a b = 1) ∧
  (a < 25) ∧ (b < 25) ∧
  (a + b = 19) := by
  sorry

end sum_of_integers_problem_l2997_299763


namespace infinitely_many_common_terms_l2997_299799

/-- Sequence a_n defined recursively -/
def a : ℕ → ℤ
  | 0 => 2
  | 1 => 14
  | (n + 2) => 14 * a (n + 1) + a n

/-- Sequence b_n defined recursively -/
def b : ℕ → ℤ
  | 0 => 2
  | 1 => 14
  | (n + 2) => 6 * b (n + 1) - b n

/-- There are infinitely many common terms in sequences a and b -/
theorem infinitely_many_common_terms :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ a (2 * k + 1) = b (3 * k + 1) :=
by sorry

end infinitely_many_common_terms_l2997_299799
