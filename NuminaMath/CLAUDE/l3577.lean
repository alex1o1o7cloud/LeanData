import Mathlib

namespace arithmetic_sequence_sum_l3577_357763

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → a 2 + a 8 = 180 → a 3 + a 4 + a 5 + a 6 + a 7 = 450 := by
  sorry

end arithmetic_sequence_sum_l3577_357763


namespace telescope_cost_l3577_357772

theorem telescope_cost (joan karl : ℕ) 
  (h1 : joan + karl = 400)
  (h2 : 2 * joan = karl + 74) : 
  joan = 158 := by
sorry

end telescope_cost_l3577_357772


namespace nonagon_diagonals_l3577_357767

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular nine-sided polygon contains 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end nonagon_diagonals_l3577_357767


namespace number_of_complementary_sets_l3577_357769

/-- Represents a card in the deck -/
structure Card where
  shape : Fin 3
  color : Fin 3
  shade : Fin 3
  size : Fin 3

/-- The deck of all possible cards -/
def deck : Finset Card := sorry

/-- Checks if a set of three cards is complementary -/
def isComplementary (c1 c2 c3 : Card) : Prop := sorry

/-- The set of all complementary three-card sets -/
def complementarySets : Finset (Finset Card) := sorry

theorem number_of_complementary_sets :
  Finset.card complementarySets = 4536 := by sorry

end number_of_complementary_sets_l3577_357769


namespace fencing_cost_per_meter_l3577_357762

/-- Proves that the cost of fencing per meter for a rectangular plot is 26.5 Rs. -/
theorem fencing_cost_per_meter 
  (length : ℝ) 
  (breadth : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 70) 
  (h2 : length = breadth + 40) 
  (h3 : total_cost = 5300) : 
  total_cost / (2 * length + 2 * breadth) = 26.5 := by
  sorry

end fencing_cost_per_meter_l3577_357762


namespace probability_four_ones_eight_dice_l3577_357728

theorem probability_four_ones_eight_dice : 
  let n : ℕ := 8  -- number of dice
  let s : ℕ := 8  -- number of sides on each die
  let k : ℕ := 4  -- number of dice showing 1
  Nat.choose n k * (1 / s) ^ k * ((s - 1) / s) ^ (n - k) = 168070 / 16777216 := by
  sorry

end probability_four_ones_eight_dice_l3577_357728


namespace smallest_yellow_marbles_l3577_357757

theorem smallest_yellow_marbles (n : ℕ) (h1 : n % 10 = 0) (h2 : n ≥ 30) : ∃ (blue red green yellow : ℕ),
  blue = n / 2 ∧
  red = n / 5 ∧
  green = 8 ∧
  yellow = n - (blue + red + green) ∧
  yellow ≥ 1 ∧
  ∀ m : ℕ, m < n → ¬(∃ (b r g y : ℕ),
    b = m / 2 ∧
    r = m / 5 ∧
    g = 8 ∧
    y = m - (b + r + g) ∧
    y ≥ 1) :=
by sorry

end smallest_yellow_marbles_l3577_357757


namespace coefficient_m5n5_in_expansion_l3577_357783

theorem coefficient_m5n5_in_expansion : ∀ m n : ℕ,
  (Nat.choose 10 5 : ℕ) = 252 :=
by
  sorry

end coefficient_m5n5_in_expansion_l3577_357783


namespace new_person_age_l3577_357766

theorem new_person_age (T : ℕ) : 
  T > 0 →  -- Ensure total age is positive
  (T / 10 : ℚ) - ((T - 48 + 18) / 10 : ℚ) = 3 →
  18 = 18 := by sorry

end new_person_age_l3577_357766


namespace sugar_per_cookie_l3577_357731

theorem sugar_per_cookie (initial_cookies : ℕ) (initial_sugar_per_cookie : ℚ) 
  (new_cookies : ℕ) (total_sugar : ℚ) :
  initial_cookies = 50 →
  initial_sugar_per_cookie = 1 / 10 →
  new_cookies = 25 →
  total_sugar = initial_cookies * initial_sugar_per_cookie →
  total_sugar / new_cookies = 1 / 5 := by
sorry

end sugar_per_cookie_l3577_357731


namespace range_of_m_l3577_357718

-- Define the propositions p and q
def p (m : ℝ) : Prop := 4 < m ∧ m < 10

def q (m : ℝ) : Prop := 8 < m ∧ m < 12

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ (4 < m ∧ m ≤ 8) ∨ (10 ≤ m ∧ m < 12) := by
  sorry

end range_of_m_l3577_357718


namespace factorization_equality_l3577_357781

theorem factorization_equality (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2*y) * (x + 2*y) := by
  sorry

end factorization_equality_l3577_357781


namespace total_count_is_900_l3577_357710

/-- Represents the count of type A components -/
def a : ℕ := 400

/-- Represents the count of type B components -/
def b : ℕ := 300

/-- Represents the count of type C components -/
def c : ℕ := 200

/-- Represents the total sample size -/
def sample_size : ℕ := 45

/-- Represents the number of type C components sampled -/
def c_sampled : ℕ := 10

/-- Represents the total count of all components -/
def total_count : ℕ := a + b + c

/-- Theorem stating that the total count of all components is 900 -/
theorem total_count_is_900 : total_count = 900 := by
  sorry

end total_count_is_900_l3577_357710


namespace star_four_three_l3577_357784

def star (x y : ℝ) : ℝ := x^2 - x*y + y^2

theorem star_four_three : star 4 3 = 13 := by
  sorry

end star_four_three_l3577_357784


namespace equation_solution_l3577_357729

theorem equation_solution (x : ℝ) : (4 * x + 2) / (5 * x - 5) = 3 / 4 → x = -23 := by
  sorry

end equation_solution_l3577_357729


namespace g_formula_l3577_357741

noncomputable def g (a : ℝ) : ℝ :=
  let m := Real.exp (Real.log 2 * min a 2)
  let n := Real.exp (Real.log 2 * max (-2) a)
  n - m

theorem g_formula (a : ℝ) (ha : a ≥ 0) :
  g a = if a ≤ 2 then -3 else 1 - Real.exp (Real.log 2 * a) := by
  sorry

end g_formula_l3577_357741


namespace field_trip_adults_l3577_357789

theorem field_trip_adults (van_capacity : ℕ) (num_students : ℕ) (num_vans : ℕ) : 
  van_capacity = 8 → num_students = 22 → num_vans = 3 → 
  (num_vans * van_capacity - num_students : ℕ) = 2 := by
  sorry

end field_trip_adults_l3577_357789


namespace imaginary_part_of_i_times_two_plus_i_l3577_357730

theorem imaginary_part_of_i_times_two_plus_i (i : ℂ) : 
  (i * i = -1) → Complex.im (i * (2 + i)) = 2 := by
  sorry

end imaginary_part_of_i_times_two_plus_i_l3577_357730


namespace monkey_climb_height_l3577_357725

/-- The height of the tree that the monkey climbs -/
def tree_height : ℕ := 22

/-- The distance the monkey climbs up each hour -/
def climb_distance : ℕ := 3

/-- The distance the monkey slips back each hour -/
def slip_distance : ℕ := 2

/-- The total time it takes for the monkey to reach the top of the tree -/
def total_time : ℕ := 20

/-- Theorem stating that the height of the tree is 22 ft -/
theorem monkey_climb_height :
  tree_height = (total_time - 1) * (climb_distance - slip_distance) + climb_distance :=
by sorry

end monkey_climb_height_l3577_357725


namespace zoo_field_trip_l3577_357733

/-- Calculates the number of individuals left at the zoo after a field trip --/
theorem zoo_field_trip (initial_fifth_grade : ℕ) (merged_fifth_grade : ℕ) 
  (initial_chaperones : ℕ) (teachers : ℕ) (third_grade : ℕ) 
  (additional_chaperones : ℕ) (fifth_grade_left : ℕ) (third_grade_left : ℕ) 
  (chaperones_left : ℕ) : 
  initial_fifth_grade = 10 →
  merged_fifth_grade = 12 →
  initial_chaperones = 5 →
  teachers = 2 →
  third_grade = 15 →
  additional_chaperones = 3 →
  fifth_grade_left = 10 →
  third_grade_left = 6 →
  chaperones_left = 2 →
  initial_fifth_grade + merged_fifth_grade + initial_chaperones + teachers + 
    third_grade + additional_chaperones - 
    (fifth_grade_left + third_grade_left + chaperones_left) = 29 := by
  sorry


end zoo_field_trip_l3577_357733


namespace same_color_probability_l3577_357791

/-- The probability of drawing two balls of the same color with replacement -/
theorem same_color_probability (green red blue : ℕ) (h_green : green = 8) (h_red : red = 6) (h_blue : blue = 4) :
  let total := green + red + blue
  (green / total) ^ 2 + (red / total) ^ 2 + (blue / total) ^ 2 = 29 / 81 :=
by sorry

end same_color_probability_l3577_357791


namespace shirts_not_washed_l3577_357768

theorem shirts_not_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) : 
  short_sleeve = 39 → long_sleeve = 47 → washed = 20 → 
  short_sleeve + long_sleeve - washed = 66 := by
sorry

end shirts_not_washed_l3577_357768


namespace rectangular_to_polar_conversion_l3577_357720

theorem rectangular_to_polar_conversion :
  let x : ℝ := Real.sqrt 3
  let y : ℝ := -Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := if x > 0 ∧ y < 0
                then 2 * Real.pi - Real.arctan ((-y) / x)
                else 0  -- This else case is just a placeholder
  (r = Real.sqrt 6 ∧ θ = 7 * Real.pi / 4 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) := by
  sorry

end rectangular_to_polar_conversion_l3577_357720


namespace zero_existence_l3577_357798

open Real

-- Define the differential equation
def is_solution (y : ℝ → ℝ) : Prop :=
  ∀ x, (x^2 + 9) * (deriv^[2] y x) + (x^2 + 4) * y x = 0

-- Define the theorem
theorem zero_existence (y : ℝ → ℝ) 
  (h_sol : is_solution y) 
  (h_init1 : y 0 = 0) 
  (h_init2 : deriv y 0 = 1) :
  ∃ x ∈ Set.Icc (Real.sqrt (63/53) * π) (3*π/2), y x = 0 := by
  sorry

end zero_existence_l3577_357798


namespace product_of_cubic_fractions_l3577_357748

theorem product_of_cubic_fractions :
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 3) * (f 4) * (f 5) * (f 6) * (f 7) = 57 / 168 := by
  sorry

end product_of_cubic_fractions_l3577_357748


namespace diamond_property_false_l3577_357778

/-- The diamond operation for real numbers -/
def diamond (x y : ℝ) : ℝ := |x + y - 1|

/-- The statement that is false -/
theorem diamond_property_false : ∃ x y : ℝ, 2 * (diamond x y) ≠ diamond (2 * x) (2 * y) := by
  sorry

end diamond_property_false_l3577_357778


namespace total_triangles_is_18_l3577_357760

/-- Represents a figure with different types of triangles -/
structure TriangleFigure where
  smallest : Nat
  medium : Nat
  largest : Nat

/-- Calculates the total number of triangles in a TriangleFigure -/
def totalTriangles (figure : TriangleFigure) : Nat :=
  figure.smallest + figure.medium + figure.largest

/-- The given figure with 6 smallest, 7 medium, and 5 largest triangles -/
def givenFigure : TriangleFigure :=
  { smallest := 6, medium := 7, largest := 5 }

/-- Theorem stating that the total number of triangles in the given figure is 18 -/
theorem total_triangles_is_18 : totalTriangles givenFigure = 18 := by
  sorry

end total_triangles_is_18_l3577_357760


namespace largest_value_l3577_357787

theorem largest_value : 
  max (5 * Real.sqrt 2 - 7) (max (7 - 5 * Real.sqrt 2) (max |4/4 - 4/4| 0.1)) = 0.1 := by
  sorry

end largest_value_l3577_357787


namespace shaded_area_theorem_l3577_357794

/-- Represents a point on the grid -/
structure GridPoint where
  x : ℚ
  y : ℚ

/-- Represents a line segment on the grid -/
inductive GridSegment
  | Horizontal : GridPoint → GridPoint → GridSegment
  | Vertical : GridPoint → GridPoint → GridSegment
  | Diagonal : GridPoint → GridPoint → GridSegment
  | Midpoint : GridPoint → GridPoint → GridSegment

/-- Represents a shaded area on the grid -/
structure ShadedArea where
  boundary : List GridSegment

/-- The main theorem statement -/
theorem shaded_area_theorem (grid : List (List ℚ)) (shaded_areas : List ShadedArea) :
  (List.length shaded_areas = 2015) →
  (∀ area ∈ shaded_areas, ∀ segment ∈ area.boundary,
    match segment with
    | GridSegment.Horizontal p1 p2 => p1.y = p2.y ∧ (p2.x - p1.x).den = 1
    | GridSegment.Vertical p1 p2 => p1.x = p2.x ∧ (p2.y - p1.y).den = 1
    | GridSegment.Diagonal p1 p2 => (p2.x - p1.x).num = (p2.x - p1.x).den ∧ (p2.y - p1.y).num = (p2.y - p1.y).den
    | GridSegment.Midpoint p1 p2 => (p2.x - p1.x).num = 1 ∧ (p2.x - p1.x).den = 2 ∧ (p2.y - p1.y).num = 1 ∧ (p2.y - p1.y).den = 2
  ) →
  (∃ total_area : ℚ, total_area = 95/2) :=
by sorry

end shaded_area_theorem_l3577_357794


namespace periodic_function_2009_l3577_357746

/-- A function satisfying the given functional equation -/
def PeriodicFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) * (1 - f x) = 1 + f x

theorem periodic_function_2009 (f : ℝ → ℝ) 
  (h1 : PeriodicFunction f) 
  (h2 : f 5 = 2 + Real.sqrt 3) : 
  f 2009 = -2 + Real.sqrt 3 := by
  sorry

end periodic_function_2009_l3577_357746


namespace correct_statements_l3577_357705

theorem correct_statements (x : ℝ) : 
  (x ≥ 0 → x^2 ≥ x) ∧ 
  (x^2 ≥ 0 → abs x ≥ 0) ∧ 
  (x ≤ -1 → x^2 ≥ abs x) := by
  sorry

end correct_statements_l3577_357705


namespace richard_solves_1099_problems_l3577_357734

/-- The number of problems Richard solves in 2013 --/
def problems_solved_2013 : ℕ :=
  let days_in_2013 : ℕ := 365
  let problems_per_week : ℕ := 2 + 1 + 2 + 1 + 2 + 5 + 7
  let full_weeks : ℕ := days_in_2013 / 7
  let extra_day : ℕ := days_in_2013 % 7
  let normal_tuesday_problems : ℕ := 1
  let special_tuesday_problems : ℕ := 60
  full_weeks * problems_per_week + extra_day * normal_tuesday_problems + 
    (special_tuesday_problems - normal_tuesday_problems)

theorem richard_solves_1099_problems : problems_solved_2013 = 1099 := by
  sorry

end richard_solves_1099_problems_l3577_357734


namespace opposite_sign_square_root_l3577_357752

theorem opposite_sign_square_root (a b : ℝ) : 
  (|2*a - 4| + Real.sqrt (3*b + 12) = 0) → 
  Real.sqrt (2*a - 3*b) = 4 ∨ Real.sqrt (2*a - 3*b) = -4 :=
by sorry

end opposite_sign_square_root_l3577_357752


namespace expression_equality_l3577_357761

theorem expression_equality : 
  Real.sqrt 4 + |1 - Real.sqrt 3| - (1/2)⁻¹ + 2023^0 = Real.sqrt 3 := by
  sorry

end expression_equality_l3577_357761


namespace cannot_cut_square_l3577_357759

theorem cannot_cut_square (rectangle_area : ℝ) (square_area : ℝ) 
  (h_rectangle_area : rectangle_area = 582) 
  (h_square_area : square_area = 400) : ¬ ∃ (l w : ℝ), 
  l * w = rectangle_area ∧ 
  l / w = 3 / 2 ∧ 
  w ≥ Real.sqrt square_area := by
sorry

end cannot_cut_square_l3577_357759


namespace specific_committee_selection_l3577_357722

/-- The number of ways to choose a committee with a specific person included -/
def committee_selection (n m k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: There are 462 ways to choose a 6-person committee from 12 people with one specific person included -/
theorem specific_committee_selection :
  committee_selection 12 6 1 = 462 := by
  sorry

end specific_committee_selection_l3577_357722


namespace man_pants_count_l3577_357765

theorem man_pants_count (t_shirts : ℕ) (total_outfits : ℕ) (pants : ℕ) : 
  t_shirts = 8 → total_outfits = 72 → total_outfits = t_shirts * pants → pants = 9 := by
sorry

end man_pants_count_l3577_357765


namespace inscribed_cube_surface_area_l3577_357776

theorem inscribed_cube_surface_area (r : ℝ) (h : 4 * π * r^2 = π) :
  6 * (1 / (r * Real.sqrt 3))^2 = 2 := by sorry

end inscribed_cube_surface_area_l3577_357776


namespace sum_areas_eighteen_disks_l3577_357770

/-- The sum of areas of 18 congruent disks arranged on a unit circle --/
theorem sum_areas_eighteen_disks : ℝ := by
  -- Define the number of disks
  let n : ℕ := 18

  -- Define the radius of the large circle
  let R : ℝ := 1

  -- Define the central angle for each disk
  let central_angle : ℝ := 2 * Real.pi / n

  -- Define the radius of each small disk
  let r : ℝ := Real.tan (central_angle / 2)

  -- Define the area of a single disk
  let single_disk_area : ℝ := Real.pi * r^2

  -- Define the sum of areas of all disks
  let total_area : ℝ := n * single_disk_area

  -- The theorem statement
  have : total_area = 18 * Real.pi * (Real.tan (Real.pi / 18))^2 := by sorry

  -- Return the result
  exact total_area


end sum_areas_eighteen_disks_l3577_357770


namespace star_two_neg_three_l3577_357756

-- Define the ★ operation
def star (a b : ℤ) : ℤ := a * b^3 - 2*b + 2

-- Theorem statement
theorem star_two_neg_three : star 2 (-3) = -46 := by
  sorry

end star_two_neg_three_l3577_357756


namespace base6_to_base10_12345_l3577_357751

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- The list representation of 12345 in base 6 --/
def number : List Nat := [5, 4, 3, 2, 1]

theorem base6_to_base10_12345 :
  base6ToBase10 number = 1865 := by
  sorry

#eval base6ToBase10 number

end base6_to_base10_12345_l3577_357751


namespace future_cup_defense_l3577_357786

/-- Represents the defensive statistics of a class --/
structure DefensiveStats where
  avgGoalsConceded : ℝ
  stdDevGoalsConceded : ℝ

/-- Determines if one class has better average defensive performance than another --/
def betterAverageDefense (a b : DefensiveStats) : Prop :=
  a.avgGoalsConceded > b.avgGoalsConceded

/-- Determines if one class has less stable defensive performance than another --/
def lessStableDefense (a b : DefensiveStats) : Prop :=
  a.stdDevGoalsConceded > b.stdDevGoalsConceded

/-- Determines if a class has relatively consistent defensive performance --/
def consistentDefense (a : DefensiveStats) : Prop :=
  a.stdDevGoalsConceded < 0.5

theorem future_cup_defense 
  (classA classB : DefensiveStats)
  (hA : classA.avgGoalsConceded = 1.9 ∧ classA.stdDevGoalsConceded = 0.3)
  (hB : classB.avgGoalsConceded = 1.3 ∧ classB.stdDevGoalsConceded = 1.2) :
  betterAverageDefense classA classB ∧ 
  lessStableDefense classB classA ∧ 
  consistentDefense classA := by
  sorry

end future_cup_defense_l3577_357786


namespace endpoint_coordinate_sum_l3577_357745

/-- Given a line segment with midpoint (5, -8) and one endpoint at (7, 2),
    the sum of the coordinates of the other endpoint is -15. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
    (5 = (x + 7) / 2) →
    (-8 = (y + 2) / 2) →
    x + y = -15 := by
  sorry

end endpoint_coordinate_sum_l3577_357745


namespace sum_26_35_in_base7_l3577_357797

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of 26 and 35 in base 10, when converted to base 7, equals 85 -/
theorem sum_26_35_in_base7 : toBase7 (26 + 35) = 85 := by sorry

end sum_26_35_in_base7_l3577_357797


namespace largest_divisor_of_composite_l3577_357790

/-- A number is composite if it has a proper divisor -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- The largest integer that always divides n + n^4 - n^3 for composite n > 6 -/
def LargestDivisor : ℕ := 6

theorem largest_divisor_of_composite (n : ℕ) (h1 : IsComposite n) (h2 : n > 6) :
  (∀ d : ℕ, d > LargestDivisor → ∃ m : ℕ, IsComposite m ∧ m > 6 ∧ ¬(d ∣ (m + m^4 - m^3))) ∧
  (LargestDivisor ∣ (n + n^4 - n^3)) :=
sorry

end largest_divisor_of_composite_l3577_357790


namespace mark_deck_project_cost_l3577_357714

/-- The total cost of Mark's deck project -/
def deck_project_cost (length width : ℝ) (cost_A cost_B cost_sealant : ℝ) 
  (percent_A : ℝ) (tax_rate : ℝ) : ℝ :=
let total_area := length * width
let area_A := percent_A * total_area
let area_B := (1 - percent_A) * total_area
let cost_materials := cost_A * area_A + cost_B * area_B
let cost_sealant_total := cost_sealant * total_area
let subtotal := cost_materials + cost_sealant_total
subtotal * (1 + tax_rate)

/-- Theorem stating the total cost of Mark's deck project -/
theorem mark_deck_project_cost :
  deck_project_cost 30 40 3 5 1 0.6 0.07 = 6163.20 := by
  sorry


end mark_deck_project_cost_l3577_357714


namespace bug_path_tiles_l3577_357774

def width : ℕ := 15
def length : ℕ := 25
def total_tiles : ℕ := 375

theorem bug_path_tiles : 
  width + length - Nat.gcd width length = 35 := by sorry

end bug_path_tiles_l3577_357774


namespace expression_factorization_l3577_357785

theorem expression_factorization (b : ℝ) : 
  (8 * b^3 - 104 * b^2 + 9) - (9 * b^3 - 2 * b^2 + 9) = -b^2 * (b + 102) := by
  sorry

end expression_factorization_l3577_357785


namespace book_cost_l3577_357754

theorem book_cost (cost_of_three : ℝ) (h : cost_of_three = 45) : 
  (7 * (cost_of_three / 3) : ℝ) = 105 := by
  sorry

end book_cost_l3577_357754


namespace smith_cycling_time_comparison_l3577_357792

/-- Proves that the time taken for the second trip is 3/4 of the time taken for the first trip -/
theorem smith_cycling_time_comparison 
  (first_distance : ℝ) 
  (second_distance : ℝ) 
  (speed_multiplier : ℝ) 
  (h1 : first_distance = 90) 
  (h2 : second_distance = 270) 
  (h3 : speed_multiplier = 4) 
  (v : ℝ) 
  (hv : v > 0) : 
  (second_distance / (speed_multiplier * v)) / (first_distance / v) = 3/4 :=
by sorry

end smith_cycling_time_comparison_l3577_357792


namespace max_correct_answers_for_given_test_l3577_357704

/-- Represents a multiple choice test with scoring system -/
structure MCTest where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correctly answered questions -/
def max_correct_answers (test : MCTest) : ℕ :=
  sorry

/-- Theorem stating the maximum number of correct answers for the given test -/
theorem max_correct_answers_for_given_test :
  let test : MCTest := {
    total_questions := 60,
    correct_points := 3,
    incorrect_points := -2,
    total_score := 126
  }
  max_correct_answers test = 49 := by sorry

end max_correct_answers_for_given_test_l3577_357704


namespace unique_four_digit_number_l3577_357782

theorem unique_four_digit_number : ∃! x : ℕ,
  1000 ≤ x ∧ x ≤ 9999 ∧
  x % 7 = 0 ∧
  x % 29 = 0 ∧
  (19 * x) % 37 = 3 ∧
  x = 5075 := by
sorry

end unique_four_digit_number_l3577_357782


namespace total_crayons_l3577_357706

theorem total_crayons (billy jane mike sue : ℕ) 
  (h1 : billy = 62) 
  (h2 : jane = 52) 
  (h3 : mike = 78) 
  (h4 : sue = 97) : 
  billy + jane + mike + sue = 289 := by
  sorry

end total_crayons_l3577_357706


namespace circle_radius_l3577_357739

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x + m*y - 4 = 0

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop :=
  2*x + y = 0

-- Define the theorem
theorem circle_radius : 
  ∀ m : ℝ, 
  (∃ M N : ℝ × ℝ, 
    circle_equation M.1 M.2 m ∧ 
    circle_equation N.1 N.2 m ∧ 
    (∃ k : ℝ, symmetry_line ((M.1 + N.1)/2) ((M.2 + N.2)/2))) →
  (∃ center : ℝ × ℝ, ∀ x y : ℝ, 
    circle_equation x y m ↔ (x - center.1)^2 + (y - center.2)^2 = 3^2) :=
sorry

end circle_radius_l3577_357739


namespace no_real_roots_k_value_l3577_357737

theorem no_real_roots_k_value (k : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → k / (x - 1) + 3 ≠ x / (1 - x)) → k = -1 := by
  sorry

end no_real_roots_k_value_l3577_357737


namespace parallelogram_area_is_37_l3577_357758

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (v w : Fin 2 → ℤ) : ℕ :=
  (v 0 * w 1 - v 1 * w 0).natAbs

/-- Vectors v and w -/
def v : Fin 2 → ℤ := ![7, -5]
def w : Fin 2 → ℤ := ![13, -4]

/-- Theorem: The area of the parallelogram formed by v and w is 37 -/
theorem parallelogram_area_is_37 : parallelogramArea v w = 37 := by
  sorry

end parallelogram_area_is_37_l3577_357758


namespace inscribed_circle_radius_l3577_357750

theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 4) (hb : b = 9) (hc : c = 36) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 12 / 7 := by
  sorry

end inscribed_circle_radius_l3577_357750


namespace quadratic_solution_l3577_357753

theorem quadratic_solution (x : ℝ) : 
  (x = (7 + Real.sqrt 57) / 4 ∨ x = (7 - Real.sqrt 57) / 4) ↔ 
  2 * x^2 - 7 * x - 1 = 0 := by sorry

end quadratic_solution_l3577_357753


namespace large_rectangle_perimeter_l3577_357709

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.width)

/-- Calculates the perimeter of a square -/
def Square.perimeter (s : Square) : ℝ :=
  4 * s.side

/-- Theorem stating the perimeter of the large rectangle -/
theorem large_rectangle_perimeter 
  (square : Square)
  (small_rect : Rectangle)
  (h1 : square.perimeter = 24)
  (h2 : small_rect.perimeter = 16)
  (h3 : small_rect.length = square.side)
  (h4 : small_rect.width + square.side = small_rect.length) :
  let large_rect := Rectangle.mk (square.side + 2 * small_rect.length) (small_rect.width + square.side)
  large_rect.perimeter = 52 := by
  sorry


end large_rectangle_perimeter_l3577_357709


namespace base_b_divisibility_l3577_357723

theorem base_b_divisibility (b : ℤ) : b ∈ ({3, 4, 5, 6, 8} : Set ℤ) →
  (b * (2 * b^2 - b - 1)) % 4 ≠ 0 ↔ b = 3 ∨ b = 6 := by
  sorry

end base_b_divisibility_l3577_357723


namespace squares_in_3x3_lattice_l3577_357755

/-- A point in a 2D lattice -/
structure LatticePoint where
  x : ℕ
  y : ℕ

/-- A square lattice -/
structure SquareLattice where
  size : ℕ
  points : List LatticePoint

/-- A square formed by four lattice points -/
structure LatticeSquare where
  vertices : List LatticePoint

/-- Function to check if four points form a valid square in the lattice -/
def is_valid_square (l : SquareLattice) (s : LatticeSquare) : Prop :=
  sorry

/-- Function to count the number of valid squares in a lattice -/
def count_squares (l : SquareLattice) : ℕ :=
  sorry

/-- Theorem: The number of squares in a 3x3 square lattice is 5 -/
theorem squares_in_3x3_lattice :
  ∀ (l : SquareLattice), l.size = 3 → count_squares l = 5 := by
  sorry

end squares_in_3x3_lattice_l3577_357755


namespace arithmetic_sequence_sum_l3577_357736

/-- An arithmetic sequence with a positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 3 = 10 →
  a 1 * a 3 = 16 →
  a 11 + a 12 + a 13 = 105 := by
  sorry

end arithmetic_sequence_sum_l3577_357736


namespace cubic_equation_solution_l3577_357701

theorem cubic_equation_solution (p : ℝ) (a b c : ℝ) :
  (∀ x : ℝ, x^3 + p*x^2 + 3*x - 10 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  c - b = b - a →
  b - a > 0 →
  a = -1 ∧ b = -1 ∧ c = -1 ∧ p = 3 := by
  sorry

end cubic_equation_solution_l3577_357701


namespace infinite_series_sum_l3577_357796

/-- The sum of the infinite series ∑(1/(n(n+3))) for n from 1 to infinity is equal to 11/18. -/
theorem infinite_series_sum : ∑' (n : ℕ), 1 / (n * (n + 3 : ℝ)) = 11 / 18 := by sorry

end infinite_series_sum_l3577_357796


namespace ball_color_distribution_l3577_357771

theorem ball_color_distribution :
  ∀ (blue red green : ℕ),
  blue + red + green = 15 →
  (blue = red + 1 ∧ blue = green + 5) ∨
  (blue = red + 1 ∧ red = green) ∨
  (red = green ∧ blue = green + 5) →
  blue = 7 ∧ red = 6 ∧ green = 2 := by
sorry

end ball_color_distribution_l3577_357771


namespace sum_of_roots_theorem_l3577_357721

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 4

-- State the theorem
theorem sum_of_roots_theorem (a b : ℝ) 
  (h1 : f a = 14) 
  (h2 : f b = -14) : 
  a + b = -2 := by sorry

end sum_of_roots_theorem_l3577_357721


namespace equation_roots_property_l3577_357740

theorem equation_roots_property :
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧ 2 * x₁^2 - 5 = 20 ∧ 2 * x₂^2 - 5 = 20) ∧
  (∃ y₁ y₂ : ℝ, y₁ < 0 ∧ y₂ > 0 ∧ (3 * y₁ - 2)^2 = (2 * y₁ - 3)^2 ∧ (3 * y₂ - 2)^2 = (2 * y₂ - 3)^2) ∧
  (∃ z₁ z₂ : ℝ, z₁ < 0 ∧ z₂ > 0 ∧ (z₁^2 - 16 ≥ 0) ∧ (2 * z₁ - 2 ≥ 0) ∧ z₁^2 - 16 = 2 * z₁ - 2 ∧
                              (z₂^2 - 16 ≥ 0) ∧ (2 * z₂ - 2 ≥ 0) ∧ z₂^2 - 16 = 2 * z₂ - 2) :=
by sorry

end equation_roots_property_l3577_357740


namespace quadratic_inequality_solution_set_l3577_357717

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-2 : ℝ) 1 = {x | a * x^2 + b * x + c > 0}) :
  {x : ℝ | a * x^2 + (a + b) * x + c - a < 0} = 
    Set.Iic (-3 : ℝ) ∪ Set.Ioi (1 : ℝ) :=
by sorry

end quadratic_inequality_solution_set_l3577_357717


namespace inscribed_rectangles_area_sum_l3577_357726

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  bottom_left : ℝ × ℝ
  top_right : ℝ × ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ :=
  (r.top_right.1 - r.bottom_left.1) * (r.top_right.2 - r.bottom_left.2)

/-- Checks if a rectangle is inscribed in another rectangle -/
def is_inscribed (inner outer : Rectangle) : Prop :=
  inner.bottom_left.1 ≥ outer.bottom_left.1 ∧
  inner.bottom_left.2 ≥ outer.bottom_left.2 ∧
  inner.top_right.1 ≤ outer.top_right.1 ∧
  inner.top_right.2 ≤ outer.top_right.2

/-- Checks if two rectangles share a vertex on the given side -/
def share_vertex_on_side (r1 r2 outer : Rectangle) (side : ℝ) : Prop :=
  (r1.bottom_left.1 = side ∨ r1.top_right.1 = side) ∧
  (r2.bottom_left.1 = side ∨ r2.top_right.1 = side) ∧
  ∃ y, (r1.bottom_left.2 = y ∨ r1.top_right.2 = y) ∧
       (r2.bottom_left.2 = y ∨ r2.top_right.2 = y)

theorem inscribed_rectangles_area_sum (outer r1 r2 : Rectangle) :
  is_inscribed r1 outer →
  is_inscribed r2 outer →
  share_vertex_on_side r1 r2 outer outer.bottom_left.1 →
  area r1 + area r2 = area outer := by
  sorry

end inscribed_rectangles_area_sum_l3577_357726


namespace walking_speed_is_10_l3577_357738

/-- Represents the walking speed of person A in km/h -/
def walking_speed : ℝ := 10

/-- Represents the cycling speed of person B in km/h -/
def cycling_speed : ℝ := 20

/-- Represents the time difference in hours between when A starts walking and B starts cycling -/
def time_difference : ℝ := 6

/-- Represents the distance in km at which B catches up with A -/
def catch_up_distance : ℝ := 120

theorem walking_speed_is_10 : 
  walking_speed = 10 ∧ 
  cycling_speed = 20 ∧
  time_difference = 6 ∧
  catch_up_distance = 120 ∧
  ∃ t : ℝ, t > time_difference ∧ 
        walking_speed * t = catch_up_distance ∧ 
        cycling_speed * (t - time_difference) = catch_up_distance :=
by sorry

end walking_speed_is_10_l3577_357738


namespace bella_steps_theorem_l3577_357727

/-- The number of feet in a mile -/
def feet_per_mile : ℕ := 5280

/-- The distance between the two houses in miles -/
def distance_miles : ℕ := 3

/-- The length of Bella's step in feet -/
def step_length : ℕ := 3

/-- The ratio of Ella's speed to Bella's speed -/
def speed_ratio : ℕ := 4

/-- The number of steps Bella takes when they meet -/
def steps_taken : ℕ := 1056

theorem bella_steps_theorem :
  let total_distance_feet := distance_miles * feet_per_mile
  let combined_speed_ratio := speed_ratio + 1
  let bella_distance := total_distance_feet / combined_speed_ratio
  bella_distance / step_length = steps_taken := by
  sorry

end bella_steps_theorem_l3577_357727


namespace inequality_range_l3577_357732

theorem inequality_range (x : ℝ) :
  (∀ a : ℝ, a ≥ 1 → a * x^2 + (a - 3) * x + (a - 4) > 0) →
  x < -1 ∨ x > 3 := by
sorry

end inequality_range_l3577_357732


namespace cookie_distribution_l3577_357715

theorem cookie_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) : 
  total_cookies = 24 →
  num_people = 6 →
  cookies_per_person = total_cookies / num_people →
  cookies_per_person = 4 := by
sorry

end cookie_distribution_l3577_357715


namespace rectangle_width_l3577_357773

/-- Given a rectangle with perimeter 16 cm and width 2 cm longer than length, prove its width is 5 cm -/
theorem rectangle_width (length width : ℝ) : 
  (2 * (length + width) = 16) →  -- Perimeter is 16 cm
  (width = length + 2) →         -- Width is 2 cm longer than length
  width = 5 := by               -- Prove width is 5 cm
sorry

end rectangle_width_l3577_357773


namespace hemisphere_exposed_area_l3577_357779

/-- Given a hemisphere of radius r, where half of it is submerged in liquid,
    the total exposed surface area (including the circular top) is 2πr². -/
theorem hemisphere_exposed_area (r : ℝ) (hr : r > 0) :
  let exposed_area := π * r^2 + (π * r^2)
  exposed_area = 2 * π * r^2 := by
  sorry

end hemisphere_exposed_area_l3577_357779


namespace thirteen_power_mod_thirtyseven_l3577_357742

theorem thirteen_power_mod_thirtyseven (a : ℕ+) (h : 3 ∣ a.val) :
  (13 : ℤ)^(a.val) ≡ 1 [ZMOD 37] := by
  sorry

end thirteen_power_mod_thirtyseven_l3577_357742


namespace cube_monotone_l3577_357719

theorem cube_monotone (a b : ℝ) : a > b ↔ a^3 > b^3 := by
  sorry

end cube_monotone_l3577_357719


namespace quadratic_equation_solution_l3577_357708

theorem quadratic_equation_solution : 
  ∃ x : ℝ, 3 * x^2 - 6 * x + 3 = 0 ∧ x = 1 := by sorry

end quadratic_equation_solution_l3577_357708


namespace simplify_sqrt_expression_l3577_357712

theorem simplify_sqrt_expression :
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_expression_l3577_357712


namespace unique_pair_l3577_357777

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem unique_pair : 
  ∃! (a b : ℕ), 
    a > 0 ∧ 
    b > 0 ∧ 
    b > a ∧ 
    is_prime (b - a) ∧ 
    (a + b) % 10 = 3 ∧ 
    ∃ k : ℕ, a * b = k * k ∧
    a = 4 ∧
    b = 9 :=
by sorry

end unique_pair_l3577_357777


namespace singh_gain_l3577_357743

/-- Represents the game with three players and their monetary amounts -/
structure Game where
  initial_amount : ℚ
  ashtikar_final : ℚ
  singh_final : ℚ
  bhatia_final : ℚ

/-- Defines the conditions of the game -/
def game_conditions (g : Game) : Prop :=
  g.initial_amount = 70 ∧
  g.ashtikar_final / g.singh_final = 1 / 2 ∧
  g.singh_final / g.bhatia_final = 4 / 1 ∧
  g.ashtikar_final + g.singh_final + g.bhatia_final = 3 * g.initial_amount

/-- Theorem stating Singh's gain -/
theorem singh_gain (g : Game) (h : game_conditions g) : 
  g.singh_final - g.initial_amount = 50 := by
  sorry


end singh_gain_l3577_357743


namespace intersection_of_M_and_N_l3577_357795

def M : Set ℝ := {x | (x - 1)^2 < 4}
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_M_and_N : M ∩ N = {0, 1, 2} := by sorry

end intersection_of_M_and_N_l3577_357795


namespace inequality_equivalence_l3577_357799

theorem inequality_equivalence (x : ℝ) : 
  |2*x - 1| < |x| + 1 ↔ 0 < x ∧ x < 2 := by sorry

end inequality_equivalence_l3577_357799


namespace work_left_fraction_l3577_357749

theorem work_left_fraction (days_A days_B days_together : ℕ) 
  (h1 : days_A = 15)
  (h2 : days_B = 20)
  (h3 : days_together = 6) : 
  1 - (days_together : ℚ) * (1 / days_A + 1 / days_B) = 3 / 10 := by
  sorry

end work_left_fraction_l3577_357749


namespace point_outside_circle_if_line_intersects_l3577_357702

/-- A line intersects a circle at two distinct points if and only if
    the distance from the center of the circle to the line is less than the radius -/
axiom line_intersects_circle_iff_distance_lt_radius {a b : ℝ} :
  (∃ x₁ y₁ x₂ y₂ : ℝ, (x₁, y₁) ≠ (x₂, y₂) ∧
    a * x₁ + b * y₁ = 1 ∧ x₁^2 + y₁^2 = 1 ∧
    a * x₂ + b * y₂ = 1 ∧ x₂^2 + y₂^2 = 1) ↔
  (1 / (a^2 + b^2).sqrt < 1)

theorem point_outside_circle_if_line_intersects
  (a b : ℝ)
  (h_intersect : ∃ x₁ y₁ x₂ y₂ : ℝ, (x₁, y₁) ≠ (x₂, y₂) ∧
    a * x₁ + b * y₁ = 1 ∧ x₁^2 + y₁^2 = 1 ∧
    a * x₂ + b * y₂ = 1 ∧ x₂^2 + y₂^2 = 1) :
  a^2 + b^2 > 1 :=
sorry

end point_outside_circle_if_line_intersects_l3577_357702


namespace prob_kings_or_aces_l3577_357707

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of kings in a standard deck -/
def KingsInDeck : ℕ := 4

/-- Number of aces in a standard deck -/
def AcesInDeck : ℕ := 4

/-- Number of cards drawn -/
def CardsDrawn : ℕ := 3

/-- Probability of drawing three kings -/
def probThreeKings : ℚ :=
  (KingsInDeck / StandardDeck) * ((KingsInDeck - 1) / (StandardDeck - 1)) * ((KingsInDeck - 2) / (StandardDeck - 2))

/-- Probability of drawing exactly two aces -/
def probTwoAces : ℚ :=
  3 * (AcesInDeck / StandardDeck) * ((AcesInDeck - 1) / (StandardDeck - 1)) * ((StandardDeck - AcesInDeck) / (StandardDeck - 2))

/-- Probability of drawing three aces -/
def probThreeAces : ℚ :=
  (AcesInDeck / StandardDeck) * ((AcesInDeck - 1) / (StandardDeck - 1)) * ((AcesInDeck - 2) / (StandardDeck - 2))

/-- The probability of drawing either three kings or at least 2 aces when selecting 3 cards from a standard 52-card deck -/
theorem prob_kings_or_aces : probThreeKings + probTwoAces + probThreeAces = 43 / 33150 := by
  sorry

end prob_kings_or_aces_l3577_357707


namespace sara_flowers_l3577_357775

theorem sara_flowers (yellow_flowers : ℕ) (num_bouquets : ℕ) (red_flowers : ℕ) :
  yellow_flowers = 24 →
  num_bouquets = 8 →
  yellow_flowers % num_bouquets = 0 →
  red_flowers = yellow_flowers →
  red_flowers = 24 := by
sorry

end sara_flowers_l3577_357775


namespace ellipse_line_intersection_slope_product_l3577_357711

/-- Given an ellipse and a line intersecting it, proves the relationship between the slopes of the intersecting line and the line connecting the origin to the midpoint of the intersection points. -/
theorem ellipse_line_intersection_slope_product (k1 k2 : ℝ) 
  (h1 : k1 ≠ 0) 
  (h2 : ∃ (P1 P2 P : ℝ × ℝ), 
    (P1.1^2 + 2*P1.2^2 = 2) ∧ 
    (P2.1^2 + 2*P2.2^2 = 2) ∧ 
    (P = ((P1.1 + P2.1)/2, (P1.2 + P2.2)/2)) ∧ 
    (k1 = (P2.2 - P1.2)/(P2.1 - P1.1)) ∧ 
    (k2 = P.2/P.1)) : 
  k1 * k2 = -1/2 := by sorry

end ellipse_line_intersection_slope_product_l3577_357711


namespace circle_intersection_range_l3577_357724

def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}

def N (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}

theorem circle_intersection_range (r : ℝ) (h1 : r > 0) (h2 : M ∩ N r = N r) :
  r ∈ Set.Ioo 0 (2 - Real.sqrt 2) := by
sorry

end circle_intersection_range_l3577_357724


namespace carnival_activity_order_l3577_357788

/-- Represents an activity at the school carnival -/
inductive Activity
  | Dodgeball
  | MagicShow
  | PettingZoo
  | FacePainting

/-- Returns the popularity of an activity as a fraction -/
def popularity (a : Activity) : Rat :=
  match a with
  | Activity.Dodgeball => 3 / 8
  | Activity.MagicShow => 9 / 24
  | Activity.PettingZoo => 1 / 3
  | Activity.FacePainting => 5 / 12

/-- Checks if one activity is more popular than another -/
def morePopularThan (a b : Activity) : Prop :=
  popularity a > popularity b

theorem carnival_activity_order :
  morePopularThan Activity.FacePainting Activity.Dodgeball ∧
  morePopularThan Activity.Dodgeball Activity.MagicShow ∧
  morePopularThan Activity.MagicShow Activity.PettingZoo :=
by sorry

end carnival_activity_order_l3577_357788


namespace circle_centers_distance_l3577_357764

-- Define the circles and their properties
structure CirclePair where
  r : ℝ  -- radius of the smaller circle
  R : ℝ  -- radius of the larger circle
  common_chord : ℝ  -- length of the common chord

-- Define the theorem
theorem circle_centers_distance (c : CirclePair) :
  (c.r > 0) →  -- ensure positive radius
  (c.common_chord = c.r * Real.sqrt 2) →  -- common chord is side of square in smaller circle
  (c.R = c.r * Real.sqrt 2) →  -- radius of larger circle
  (∃ d : ℝ, (d = (c.r * (Real.sqrt 6 + Real.sqrt 2)) / 2) ∨
            (d = (c.r * (Real.sqrt 6 - Real.sqrt 2)) / 2)) :=
by sorry

end circle_centers_distance_l3577_357764


namespace annie_mike_toy_ratio_l3577_357744

/-- Represents the number of toys each person has -/
structure ToyCount where
  annie : ℕ
  mike : ℕ
  tom : ℕ

/-- Given the conditions of the problem, proves that the ratio of Annie's toys to Mike's toys is 4:1 -/
theorem annie_mike_toy_ratio 
  (tc : ToyCount) 
  (mike_toys : tc.mike = 6)
  (annie_multiple : ∃ k : ℕ, tc.annie = k * tc.mike)
  (annie_less_than_tom : tc.annie = tc.tom - 2)
  (total_toys : tc.annie + tc.mike + tc.tom = 56) :
  tc.annie / tc.mike = 4 := by
  sorry

#check annie_mike_toy_ratio

end annie_mike_toy_ratio_l3577_357744


namespace bus_journey_distance_l3577_357700

/-- Given a bus journey with two speeds, prove the distance covered at the slower speed. -/
theorem bus_journey_distance (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 250)
  (h2 : speed1 = 40)
  (h3 : speed2 = 60)
  (h4 : total_time = 5.4)
  (h5 : total_distance > 0)
  (h6 : speed1 > 0)
  (h7 : speed2 > 0)
  (h8 : total_time > 0)
  (h9 : speed1 < speed2) :
  ∃ (distance1 : ℝ), 
    distance1 / speed1 + (total_distance - distance1) / speed2 = total_time ∧ 
    distance1 = 148 := by
  sorry

end bus_journey_distance_l3577_357700


namespace dongwi_festival_cases_l3577_357703

/-- The number of cases in which Dongwi can go to play at the festival. -/
def num_cases (boys_schools : ℕ) (girls_schools : ℕ) : ℕ :=
  boys_schools + girls_schools

/-- Theorem stating that the number of cases for Dongwi to go to play is 7. -/
theorem dongwi_festival_cases :
  let boys_schools := 4
  let girls_schools := 3
  num_cases boys_schools girls_schools = 7 := by
  sorry

end dongwi_festival_cases_l3577_357703


namespace binary_representation_sum_of_exponents_l3577_357793

theorem binary_representation_sum_of_exponents (n : ℕ) (h : n = 2023) :
  (Nat.digits 2 n).sum = 48 := by
  sorry

end binary_representation_sum_of_exponents_l3577_357793


namespace prob_three_dice_sum_18_l3577_357780

/-- The probability of rolling a specific number on a standard die -/
def prob_single_die : ℚ := 1 / 6

/-- The number of faces on a standard die -/
def dice_faces : ℕ := 6

/-- The sum we're looking for -/
def target_sum : ℕ := 18

/-- The number of dice rolled -/
def num_dice : ℕ := 3

theorem prob_three_dice_sum_18 : 
  (prob_single_die ^ num_dice : ℚ) = 1 / 216 := by
  sorry

end prob_three_dice_sum_18_l3577_357780


namespace cubic_root_sum_l3577_357716

/-- Given a cubic equation with distinct real roots between 0 and 1, 
    prove that the sum of the reciprocals of one minus each root equals 2/3 -/
theorem cubic_root_sum (a b c : ℝ) : 
  (24 * a^3 - 38 * a^2 + 18 * a - 1 = 0) →
  (24 * b^3 - 38 * b^2 + 18 * b - 1 = 0) →
  (24 * c^3 - 38 * c^2 + 18 * c - 1 = 0) →
  (0 < a ∧ a < 1) →
  (0 < b ∧ b < 1) →
  (0 < c ∧ c < 1) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  1/(1-a) + 1/(1-b) + 1/(1-c) = 2/3 := by
sorry

end cubic_root_sum_l3577_357716


namespace school_sections_l3577_357747

theorem school_sections (num_boys num_girls : ℕ) 
  (h_boys : num_boys = 408) 
  (h_girls : num_girls = 312) : 
  (num_boys / (Nat.gcd num_boys num_girls)) + (num_girls / (Nat.gcd num_boys num_girls)) = 30 := by
  sorry

end school_sections_l3577_357747


namespace phantom_needs_43_more_l3577_357713

/-- The amount of money Phantom's mom gave him initially -/
def initial_amount : ℕ := 50

/-- The cost of one black printer ink -/
def black_ink_cost : ℕ := 11

/-- The number of black printer inks Phantom wants to buy -/
def black_ink_count : ℕ := 2

/-- The cost of one red printer ink -/
def red_ink_cost : ℕ := 15

/-- The number of red printer inks Phantom wants to buy -/
def red_ink_count : ℕ := 3

/-- The cost of one yellow printer ink -/
def yellow_ink_cost : ℕ := 13

/-- The number of yellow printer inks Phantom wants to buy -/
def yellow_ink_count : ℕ := 2

/-- The additional amount Phantom needs to ask his mom -/
def additional_amount : ℕ := 43

theorem phantom_needs_43_more :
  (black_ink_cost * black_ink_count +
   red_ink_cost * red_ink_count +
   yellow_ink_cost * yellow_ink_count) - initial_amount = additional_amount := by
  sorry

end phantom_needs_43_more_l3577_357713


namespace largest_and_smallest_A_l3577_357735

/-- A function that moves the last digit of a number to the first position -/
def moveLastDigitToFirst (n : ℕ) : ℕ :=
  let lastDigit := n % 10
  let restOfDigits := n / 10
  lastDigit * 10^8 + restOfDigits

/-- Theorem stating the largest and smallest A values -/
theorem largest_and_smallest_A :
  ∀ B : ℕ,
  (B > 22222222) →
  (Nat.gcd B 18 = 1) →
  (∃ A : ℕ, A = moveLastDigitToFirst B) →
  (∃ A_max A_min : ℕ,
    (A_max = moveLastDigitToFirst B → A_max ≤ 999999998) ∧
    (A_min = moveLastDigitToFirst B → A_min ≥ 122222224) ∧
    (∃ B_max B_min : ℕ,
      B_max > 22222222 ∧
      Nat.gcd B_max 18 = 1 ∧
      moveLastDigitToFirst B_max = 999999998 ∧
      B_min > 22222222 ∧
      Nat.gcd B_min 18 = 1 ∧
      moveLastDigitToFirst B_min = 122222224)) :=
by
  sorry

end largest_and_smallest_A_l3577_357735
