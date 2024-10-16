import Mathlib

namespace NUMINAMATH_CALUDE_surf_festival_average_l3228_322876

theorem surf_festival_average (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) : 
  day1 = 1500 →
  day2 = day1 + 600 →
  day3 = (2 : ℕ) * day1 / 5 →
  (day1 + day2 + day3) / 3 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_surf_festival_average_l3228_322876


namespace NUMINAMATH_CALUDE_quadrilateral_with_equal_opposite_sides_and_one_right_angle_not_necessarily_rectangle_l3228_322848

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of a quadrilateral
def has_opposite_sides_equal (q : Quadrilateral) : Prop := sorry
def has_one_right_angle (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem quadrilateral_with_equal_opposite_sides_and_one_right_angle_not_necessarily_rectangle :
  ∃ q : Quadrilateral, has_opposite_sides_equal q ∧ has_one_right_angle q ∧ ¬is_rectangle q := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_with_equal_opposite_sides_and_one_right_angle_not_necessarily_rectangle_l3228_322848


namespace NUMINAMATH_CALUDE_find_particular_number_l3228_322840

theorem find_particular_number (x : ℤ) : x - 29 + 64 = 76 → x = 41 := by
  sorry

end NUMINAMATH_CALUDE_find_particular_number_l3228_322840


namespace NUMINAMATH_CALUDE_min_value_expression_l3228_322883

theorem min_value_expression (a b c k : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_eq : a = k ∧ b = k ∧ c = k) : 
  (a + b + c) * ((a + b)⁻¹ + (a + c)⁻¹ + (b + c)⁻¹) = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3228_322883


namespace NUMINAMATH_CALUDE_squirrel_travel_time_l3228_322877

/-- Proves that a squirrel traveling at 6 miles per hour takes 30 minutes to cover 3 miles -/
theorem squirrel_travel_time :
  let speed : ℝ := 6  -- miles per hour
  let distance : ℝ := 3  -- miles
  let time_hours : ℝ := distance / speed
  let time_minutes : ℝ := time_hours * 60
  time_minutes = 30 := by sorry

end NUMINAMATH_CALUDE_squirrel_travel_time_l3228_322877


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3228_322809

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : geometric_sequence a q) 
  (h2 : ∀ n, a n > 0) 
  (h3 : q^2 = 4) : 
  (a 3 + a 4) / (a 5 + a 6) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3228_322809


namespace NUMINAMATH_CALUDE_gambler_outcome_l3228_322818

def gamble (initial_amount : ℚ) (bet_sequence : List Bool) : ℚ :=
  bet_sequence.foldl
    (fun amount win =>
      if win then amount + amount / 2
      else amount - amount / 2)
    initial_amount

theorem gambler_outcome :
  let initial_amount : ℚ := 100
  let bet_sequence : List Bool := [true, false, true, false]
  let final_amount := gamble initial_amount bet_sequence
  final_amount = 56.25 ∧ initial_amount - final_amount = 43.75 := by
  sorry

end NUMINAMATH_CALUDE_gambler_outcome_l3228_322818


namespace NUMINAMATH_CALUDE_odd_integers_square_divisibility_l3228_322861

theorem odd_integers_square_divisibility (m n : ℤ) :
  Odd m → Odd n → (m^2 - n^2 + 1) ∣ (n^2 - 1) → ∃ k : ℤ, m^2 - n^2 + 1 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_odd_integers_square_divisibility_l3228_322861


namespace NUMINAMATH_CALUDE_fraction_division_and_addition_l3228_322838

theorem fraction_division_and_addition :
  (5 : ℚ) / 6 / (9 : ℚ) / 10 + 1 / 15 = 402 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_and_addition_l3228_322838


namespace NUMINAMATH_CALUDE_function_properties_l3228_322860

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2 * a + 1) * x + 2

theorem function_properties :
  -- Part 1: When a = 0, the zero of the function is x = 2
  (∃ x : ℝ, f 0 x = 0 ∧ x = 2) ∧
  
  -- Part 2: When a = 1, the range of m for solutions in [1,3] is [-1/4, 2]
  (∀ m : ℝ, (∃ x : ℝ, x ∈ Set.Icc 1 3 ∧ f 1 x = m) ↔ m ∈ Set.Icc (-1/4) 2) ∧
  
  -- Part 3: When a > 0, the solution set of f(x) > 0
  (∀ a : ℝ, a > 0 →
    (a = 1/2 → {x : ℝ | f a x > 0} = {x : ℝ | x ≠ 2}) ∧
    (0 < a ∧ a < 1/2 → {x : ℝ | f a x > 0} = {x : ℝ | x > 1/a ∨ x < 2}) ∧
    (a > 1/2 → {x : ℝ | f a x > 0} = {x : ℝ | x < 1/a ∨ x > 2}))
  := by sorry


end NUMINAMATH_CALUDE_function_properties_l3228_322860


namespace NUMINAMATH_CALUDE_vegetable_pieces_count_l3228_322804

/-- Calculates the total number of vegetable pieces after cutting -/
def total_vegetable_pieces (bell_peppers onions zucchinis : ℕ) : ℕ :=
  let bell_pepper_thin := (bell_peppers / 4) * 20
  let bell_pepper_large := (bell_peppers - bell_peppers / 4) * 10
  let bell_pepper_small := (bell_pepper_large / 2) * 3
  let onion_thin := (onions / 2) * 18
  let onion_chunk := (onions - onions / 2) * 8
  let zucchini_thin := (zucchinis * 3 / 10) * 15
  let zucchini_chunk := (zucchinis - zucchinis * 3 / 10) * 8
  bell_pepper_thin + bell_pepper_large + bell_pepper_small + onion_thin + onion_chunk + zucchini_thin + zucchini_chunk

/-- Theorem stating that given the conditions, the total number of vegetable pieces is 441 -/
theorem vegetable_pieces_count : total_vegetable_pieces 10 7 15 = 441 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_pieces_count_l3228_322804


namespace NUMINAMATH_CALUDE_cube_of_square_of_third_smallest_prime_l3228_322813

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- State the theorem
theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime ^ 2) ^ 3 = 15625 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_square_of_third_smallest_prime_l3228_322813


namespace NUMINAMATH_CALUDE_boyden_family_ticket_cost_l3228_322879

/-- The cost of tickets for a family visit to a leisure park -/
def ticket_cost (adult_price : ℕ) (child_price : ℕ) (num_adults : ℕ) (num_children : ℕ) : ℕ :=
  adult_price * num_adults + child_price * num_children

theorem boyden_family_ticket_cost :
  let adult_price : ℕ := 19
  let child_price : ℕ := adult_price - 6
  let num_adults : ℕ := 2
  let num_children : ℕ := 3
  ticket_cost adult_price child_price num_adults num_children = 77 := by
  sorry

end NUMINAMATH_CALUDE_boyden_family_ticket_cost_l3228_322879


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l3228_322843

def M : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def N : Set ℝ := {x | x - 2 ≤ x ∧ x < 3}

theorem complement_M_intersect_N :
  ∀ x : ℝ, x ∈ (M ∩ N)ᶜ ↔ x < -2 ∨ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l3228_322843


namespace NUMINAMATH_CALUDE_valid_cube_assignment_exists_l3228_322821

/-- A cube is represented as a set of 8 vertices -/
def Cube := Fin 8

/-- An edge in the cube is a pair of vertices -/
def Edge := Prod Cube Cube

/-- The set of edges in a cube -/
def cubeEdges : Set Edge := sorry

/-- An assignment of natural numbers to the vertices of a cube -/
def Assignment := Cube → ℕ+

/-- Checks if one number divides another -/
def divides (a b : ℕ+) : Prop := ∃ k : ℕ+, b = a * k

/-- The main theorem stating the existence of a valid assignment -/
theorem valid_cube_assignment_exists : ∃ (f : Assignment),
  (∀ (e : Edge), e ∈ cubeEdges →
    (divides (f e.1) (f e.2) ∨ divides (f e.2) (f e.1))) ∧
  (∀ (v w : Cube), (Prod.mk v w) ∉ cubeEdges →
    ¬(divides (f v) (f w)) ∧ ¬(divides (f w) (f v))) := by
  sorry

end NUMINAMATH_CALUDE_valid_cube_assignment_exists_l3228_322821


namespace NUMINAMATH_CALUDE_negation_of_statement_l3228_322897

def S : Set Int := {1, -1, 0}

theorem negation_of_statement :
  (¬ ∀ x ∈ S, 2 * x + 1 > 0) ↔ (∃ x ∈ S, 2 * x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_statement_l3228_322897


namespace NUMINAMATH_CALUDE_max_3m_plus_4n_l3228_322865

theorem max_3m_plus_4n (m n : ℕ) : 
  (∃ (evens : Finset ℕ) (odds : Finset ℕ), 
    evens.card = m ∧ 
    odds.card = n ∧
    (∀ x ∈ evens, Even x ∧ x > 0) ∧
    (∀ y ∈ odds, Odd y ∧ y > 0) ∧
    (evens.sum id + odds.sum id = 1987)) →
  3 * m + 4 * n ≤ 221 :=
by sorry

end NUMINAMATH_CALUDE_max_3m_plus_4n_l3228_322865


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l3228_322816

theorem x_squared_plus_reciprocal (x : ℝ) (h : x ≠ 0) :
  x^4 + 1/x^4 = 47 → x^2 + 1/x^2 = 7 := by sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l3228_322816


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3228_322810

def is_valid (N : ℕ) : Prop :=
  ∀ k ∈ Finset.range 9, (N + k + 2) % (k + 2) = 0

theorem smallest_valid_number :
  ∃ N : ℕ, is_valid N ∧ ∀ M : ℕ, M < N → ¬ is_valid M :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3228_322810


namespace NUMINAMATH_CALUDE_min_first_row_sum_l3228_322869

/-- Represents a grid with 9 rows and 2004 columns -/
def Grid := Fin 9 → Fin 2004 → ℕ

/-- The condition that each integer from 1 to 2004 appears exactly 9 times in the grid -/
def validDistribution (g : Grid) : Prop :=
  ∀ n : Fin 2004, (Finset.univ.sum fun i => (Finset.univ.filter (fun j => g i j = n.val + 1)).card) = 9

/-- The condition that no integer appears more than 3 times in any column -/
def validColumn (g : Grid) : Prop :=
  ∀ j : Fin 2004, ∀ n : Fin 2004, (Finset.univ.filter (fun i => g i j = n.val + 1)).card ≤ 3

/-- The sum of the numbers in the first row -/
def firstRowSum (g : Grid) : ℕ :=
  Finset.univ.sum (fun j => g 0 j)

theorem min_first_row_sum :
  ∀ g : Grid, validDistribution g → validColumn g →
  firstRowSum g ≥ 2005004 :=
sorry

end NUMINAMATH_CALUDE_min_first_row_sum_l3228_322869


namespace NUMINAMATH_CALUDE_function_characterization_l3228_322801

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

def satisfies_condition (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, is_perfect_square (f (f a - b) + b * f (2 * a))

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

def solution_type_1 (f : ℤ → ℤ) : Prop :=
  (∀ n : ℤ, is_even n → f n = 0) ∧
  (∀ n : ℤ, ¬is_even n → is_perfect_square (f n))

def solution_type_2 (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, f n = n * n

theorem function_characterization (f : ℤ → ℤ) :
  satisfies_condition f → solution_type_1 f ∨ solution_type_2 f :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l3228_322801


namespace NUMINAMATH_CALUDE_chord_count_l3228_322874

/-- The number of different chords that can be drawn by connecting any two of ten points 
    on the circumference of a circle, where four of these points form a square. -/
def num_chords : ℕ := 45

/-- The total number of points on the circumference of the circle. -/
def total_points : ℕ := 10

/-- The number of points that form a square. -/
def square_points : ℕ := 4

theorem chord_count : 
  num_chords = (total_points * (total_points - 1)) / 2 :=
sorry

end NUMINAMATH_CALUDE_chord_count_l3228_322874


namespace NUMINAMATH_CALUDE_smallest_divisible_by_18_and_64_l3228_322839

theorem smallest_divisible_by_18_and_64 : ∀ n : ℕ, n > 0 → n % 18 = 0 → n % 64 = 0 → n ≥ 576 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_18_and_64_l3228_322839


namespace NUMINAMATH_CALUDE_find_number_l3228_322850

theorem find_number : ∃ x : ℕ, 5 + x = 20 ∧ x = 15 := by sorry

end NUMINAMATH_CALUDE_find_number_l3228_322850


namespace NUMINAMATH_CALUDE_problem_statement_l3228_322851

theorem problem_statement :
  (∀ x : ℝ, x^2 - 4*x + 5 > 0) ∧
  (∃ x : ℤ, 3*x^2 - 2*x - 1 = 0) ∧
  (¬ ∃ x : ℚ, x^2 = 5) ∧
  (¬ ∀ x : ℝ, x + 1/x > 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3228_322851


namespace NUMINAMATH_CALUDE_actual_distance_traveled_prove_actual_distance_l3228_322808

/-- The actual distance traveled by a person, given two different walking speeds and a distance difference. -/
theorem actual_distance_traveled (speed1 speed2 distance_diff : ℝ) (h1 : speed1 > 0) (h2 : speed2 > 0) 
  (h3 : speed2 > speed1) (h4 : distance_diff > 0) : ℝ :=
  let time := distance_diff / (speed2 - speed1)
  let actual_distance := speed1 * time
  actual_distance

/-- Proves that the actual distance traveled is 20 km under the given conditions. -/
theorem prove_actual_distance : 
  actual_distance_traveled 10 20 20 (by norm_num) (by norm_num) (by norm_num) (by norm_num) = 20 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_prove_actual_distance_l3228_322808


namespace NUMINAMATH_CALUDE_jefferson_bananas_l3228_322889

theorem jefferson_bananas (jefferson_bananas : ℕ) (walter_bananas : ℕ) : 
  walter_bananas = jefferson_bananas - (1/4 : ℚ) * jefferson_bananas →
  (jefferson_bananas + walter_bananas) / 2 = 49 →
  jefferson_bananas = 56 := by
sorry

end NUMINAMATH_CALUDE_jefferson_bananas_l3228_322889


namespace NUMINAMATH_CALUDE_hazel_eyed_brunettes_l3228_322868

/-- Represents the characteristics of students in a class -/
structure ClassCharacteristics where
  total_students : ℕ
  green_eyed_blondes : ℕ
  brunettes : ℕ
  hazel_eyed : ℕ

/-- Theorem: Number of hazel-eyed brunettes in the class -/
theorem hazel_eyed_brunettes (c : ClassCharacteristics) 
  (h1 : c.total_students = 60)
  (h2 : c.green_eyed_blondes = 20)
  (h3 : c.brunettes = 35)
  (h4 : c.hazel_eyed = 25) :
  c.total_students - (c.brunettes + c.green_eyed_blondes) = c.hazel_eyed - (c.total_students - c.brunettes) :=
by sorry

#check hazel_eyed_brunettes

end NUMINAMATH_CALUDE_hazel_eyed_brunettes_l3228_322868


namespace NUMINAMATH_CALUDE_prism_volume_approximation_l3228_322831

/-- Represents a right rectangular prism -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the volume of a rectangular prism -/
def volume (p : RectangularPrism) : ℝ := p.a * p.b * p.c

/-- The main theorem to prove -/
theorem prism_volume_approximation (p : RectangularPrism) 
  (h1 : p.a * p.b = 54)
  (h2 : p.b * p.c = 56)
  (h3 : p.a * p.c = 60) :
  round (volume p) = 426 := by
  sorry


end NUMINAMATH_CALUDE_prism_volume_approximation_l3228_322831


namespace NUMINAMATH_CALUDE_parallel_line_coordinates_l3228_322805

/-- Two points are parallel to the x-axis if and only if their y-coordinates are equal -/
def parallel_to_x_axis (p q : ℝ × ℝ) : Prop :=
  p.2 = q.2

theorem parallel_line_coordinates 
  (a b : ℝ) 
  (h : parallel_to_x_axis (5, a) (b, -2)) : 
  a = -2 ∧ b ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_coordinates_l3228_322805


namespace NUMINAMATH_CALUDE_point_B_in_fourth_quadrant_l3228_322858

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the third quadrant -/
def thirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Defines the fourth quadrant -/
def fourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Given point A in the third quadrant, prove point B is in the fourth quadrant -/
theorem point_B_in_fourth_quadrant (m n : ℝ) 
  (hA : thirdQuadrant ⟨-m, n⟩) : 
  fourthQuadrant ⟨m+1, n-1⟩ := by
  sorry


end NUMINAMATH_CALUDE_point_B_in_fourth_quadrant_l3228_322858


namespace NUMINAMATH_CALUDE_picklminster_to_quickville_distance_l3228_322806

/-- The distance between Picklminster and Quickville satisfies the given conditions -/
theorem picklminster_to_quickville_distance :
  ∃ (d : ℝ) (vA vB vC vD : ℝ),
    d > 0 ∧ vA > 0 ∧ vB > 0 ∧ vC > 0 ∧ vD > 0 ∧
    120 * vC = vA * (d - 120) ∧
    140 * vD = vA * (d - 140) ∧
    126 * vB = vC * (d - 126) ∧
    vB = vD ∧
    d = 210 :=
by
  sorry

#check picklminster_to_quickville_distance

end NUMINAMATH_CALUDE_picklminster_to_quickville_distance_l3228_322806


namespace NUMINAMATH_CALUDE_initial_money_calculation_l3228_322828

/-- The amount of money Monica and Sheila's mother gave them initially -/
def initial_money : ℝ := 50

/-- The cost of toilet paper -/
def toilet_paper_cost : ℝ := 12

/-- The cost of groceries -/
def groceries_cost : ℝ := 2 * toilet_paper_cost

/-- The amount of money left after buying toilet paper and groceries -/
def money_left : ℝ := initial_money - (toilet_paper_cost + groceries_cost)

/-- The cost of one pair of boots -/
def boot_cost : ℝ := 3 * money_left

/-- The additional money needed to buy two pairs of boots -/
def additional_money : ℝ := 2 * 35

theorem initial_money_calculation :
  initial_money = toilet_paper_cost + groceries_cost + money_left ∧
  2 * boot_cost = 2 * 3 * money_left ∧
  2 * boot_cost = 2 * 3 * money_left ∧
  2 * boot_cost - money_left = additional_money := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l3228_322828


namespace NUMINAMATH_CALUDE_quadratic_max_value_l3228_322880

/-- The quadratic function f(x) = -x^2 - 2x - 3 -/
def f (x : ℝ) : ℝ := -x^2 - 2*x - 3

theorem quadratic_max_value :
  (∀ x : ℝ, f x ≤ -2) ∧ (∃ x : ℝ, f x = -2) := by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l3228_322880


namespace NUMINAMATH_CALUDE_prob_jack_and_jill_selected_l3228_322803

/-- The probability of Jack being selected for the interview. -/
def prob_jack : ℝ := 0.20

/-- The probability of Jill being selected for the interview. -/
def prob_jill : ℝ := 0.15

/-- The number of workers in the hospital. -/
def num_workers : ℕ := 8

/-- The number of workers to be interviewed. -/
def num_interviewed : ℕ := 2

/-- Assumption that the selection of Jack and Jill are independent events. -/
axiom selection_independent : True

theorem prob_jack_and_jill_selected : 
  prob_jack * prob_jill = 0.03 := by sorry

end NUMINAMATH_CALUDE_prob_jack_and_jill_selected_l3228_322803


namespace NUMINAMATH_CALUDE_min_t_for_inequality_l3228_322873

theorem min_t_for_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (∀ x y, x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) ≤ (1 / (2 * Real.sqrt 6)) * (2 * x + 3 * y)) ∧
  (∀ ε > 0, ∃ x y, x ≥ 0 ∧ y ≥ 0 ∧ Real.sqrt (x * y) > (1 / (2 * Real.sqrt 6) - ε) * (2 * x + 3 * y)) :=
sorry

end NUMINAMATH_CALUDE_min_t_for_inequality_l3228_322873


namespace NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l3228_322833

theorem complex_exponential_to_rectangular : 2 * Real.sqrt 3 * Complex.exp (Complex.I * (13 * Real.pi / 6)) = 3 + Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l3228_322833


namespace NUMINAMATH_CALUDE_trig_product_equals_one_l3228_322882

theorem trig_product_equals_one :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_product_equals_one_l3228_322882


namespace NUMINAMATH_CALUDE_diary_ratio_l3228_322864

theorem diary_ratio (x : ℚ) : 
  (8 + x - (1/4) * (8 + x) = 18) → (x / 8 = 2) := by
  sorry

end NUMINAMATH_CALUDE_diary_ratio_l3228_322864


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3228_322844

theorem complex_modulus_problem (x y : ℝ) (h : Complex.I * Complex.mk x y = Complex.mk 3 4) :
  Complex.abs (Complex.mk x y) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3228_322844


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3228_322822

theorem solution_set_inequality (x : ℝ) : 
  (2 * x - 3) / (x + 2) ≤ 1 ↔ -2 < x ∧ x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3228_322822


namespace NUMINAMATH_CALUDE_flower_bed_perimeter_reduction_l3228_322829

/-- Represents a rectangular flower bed with length and width -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangular flower bed -/
def perimeter (fb : FlowerBed) : ℝ := 2 * (fb.length + fb.width)

/-- Theorem: The perimeter of a rectangular flower bed decreases by 17.5% 
    after reducing the length by 28% and the width by 28% -/
theorem flower_bed_perimeter_reduction (fb : FlowerBed) :
  let reduced_fb := FlowerBed.mk (fb.length * 0.72) (fb.width * 0.72)
  (perimeter fb - perimeter reduced_fb) / perimeter fb = 0.175 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_perimeter_reduction_l3228_322829


namespace NUMINAMATH_CALUDE_students_in_all_workshops_l3228_322895

theorem students_in_all_workshops (total : ℕ) (robotics dance music : ℕ) (at_least_two : ℕ) 
  (h_total : total = 25)
  (h_robotics : robotics = 15)
  (h_dance : dance = 12)
  (h_music : music = 10)
  (h_at_least_two : at_least_two = 11)
  (h_sum : robotics + dance + music - 2 * at_least_two ≤ total) :
  ∃ (only_one only_two all_three : ℕ),
    only_one + only_two + all_three = total ∧
    only_two + 3 * all_three = at_least_two ∧
    all_three = 1 :=
by sorry

end NUMINAMATH_CALUDE_students_in_all_workshops_l3228_322895


namespace NUMINAMATH_CALUDE_zeros_sum_inequality_l3228_322846

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * log x

theorem zeros_sum_inequality (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ < x₂ →
  f (1 / exp 2) x₁ = 0 → f (1 / exp 2) x₂ = 0 →
  log (x₁ + x₂) > log 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_zeros_sum_inequality_l3228_322846


namespace NUMINAMATH_CALUDE_count_not_divisible_1200_l3228_322823

def count_not_divisible (n : ℕ) : ℕ :=
  (n - 1) - ((n - 1) / 6 + (n - 1) / 8 - (n - 1) / 24)

theorem count_not_divisible_1200 :
  count_not_divisible 1200 = 900 := by
  sorry

end NUMINAMATH_CALUDE_count_not_divisible_1200_l3228_322823


namespace NUMINAMATH_CALUDE_triple_tilde_47_l3228_322817

-- Define the tilde operation
def tilde (N : ℝ) : ℝ := 0.4 * N + 2

-- State the theorem
theorem triple_tilde_47 : tilde (tilde (tilde 47)) = 6.128 := by sorry

end NUMINAMATH_CALUDE_triple_tilde_47_l3228_322817


namespace NUMINAMATH_CALUDE_rectangle_area_l3228_322853

theorem rectangle_area (x : ℝ) : 
  (2 * (x + 4) + 2 * (x - 2) = 56) → 
  ((x + 4) * (x - 2) = 187) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3228_322853


namespace NUMINAMATH_CALUDE_class_size_calculation_l3228_322894

theorem class_size_calculation (total : ℕ) 
  (h1 : (40 : ℚ) / 100 * total = (↑total * (40 : ℚ) / 100).floor)
  (h2 : (70 : ℚ) / 100 * ((40 : ℚ) / 100 * total) = 21) : 
  total = 75 := by
sorry

end NUMINAMATH_CALUDE_class_size_calculation_l3228_322894


namespace NUMINAMATH_CALUDE_inheritance_satisfies_tax_conditions_inheritance_uniqueness_l3228_322854

/-- The inheritance amount that satisfies the tax conditions -/
def inheritance : ℝ := 41379

/-- The total tax paid -/
def total_tax : ℝ := 15000

/-- Federal tax rate -/
def federal_tax_rate : ℝ := 0.25

/-- State tax rate -/
def state_tax_rate : ℝ := 0.15

/-- Theorem stating that the inheritance amount satisfies the tax conditions -/
theorem inheritance_satisfies_tax_conditions :
  federal_tax_rate * inheritance + 
  state_tax_rate * (inheritance - federal_tax_rate * inheritance) = 
  total_tax := by sorry

/-- Theorem stating that the inheritance amount is unique -/
theorem inheritance_uniqueness (x : ℝ) :
  federal_tax_rate * x + 
  state_tax_rate * (x - federal_tax_rate * x) = 
  total_tax →
  x = inheritance := by sorry

end NUMINAMATH_CALUDE_inheritance_satisfies_tax_conditions_inheritance_uniqueness_l3228_322854


namespace NUMINAMATH_CALUDE_xyz_divisible_by_55_l3228_322825

theorem xyz_divisible_by_55 (x y z a b c : ℤ) 
  (h1 : x^2 + y^2 = a^2) 
  (h2 : y^2 + z^2 = b^2) 
  (h3 : z^2 + x^2 = c^2) : 
  55 ∣ (x * y * z) := by
  sorry

end NUMINAMATH_CALUDE_xyz_divisible_by_55_l3228_322825


namespace NUMINAMATH_CALUDE_complex_equation_l3228_322891

theorem complex_equation (z : ℂ) : (Complex.I * z = 1 - 2 * Complex.I) → z = -2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_l3228_322891


namespace NUMINAMATH_CALUDE_largest_circle_at_A_l3228_322886

/-- Represents a pentagon with circles at its vertices -/
structure PentagonWithCircles where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  AE : ℝ
  radA : ℝ
  radB : ℝ
  radC : ℝ
  radD : ℝ
  radE : ℝ
  circle_contact : 
    AB = radA + radB ∧
    BC = radB + radC ∧
    CD = radC + radD ∧
    DE = radD + radE ∧
    AE = radE + radA

/-- The circle centered at A has the largest radius -/
theorem largest_circle_at_A (p : PentagonWithCircles) 
  (h1 : p.AB = 16) (h2 : p.BC = 14) (h3 : p.CD = 17) (h4 : p.DE = 13) (h5 : p.AE = 14) :
  p.radA = max p.radA (max p.radB (max p.radC (max p.radD p.radE))) := by
  sorry

end NUMINAMATH_CALUDE_largest_circle_at_A_l3228_322886


namespace NUMINAMATH_CALUDE_sqrt_sum_greater_than_sqrt_of_sum_l3228_322849

theorem sqrt_sum_greater_than_sqrt_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt a + Real.sqrt b > Real.sqrt (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_greater_than_sqrt_of_sum_l3228_322849


namespace NUMINAMATH_CALUDE_absolute_value_and_quadratic_equivalence_l3228_322856

theorem absolute_value_and_quadratic_equivalence :
  ∀ (b c : ℝ),
    (∀ x : ℝ, |x - 8| = 3 ↔ x^2 + b*x + c = 0) ↔
    (b = -16 ∧ c = 55) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_and_quadratic_equivalence_l3228_322856


namespace NUMINAMATH_CALUDE_half_pond_fill_time_l3228_322855

/-- Represents the growth of water hyacinth in a pond -/
def WaterHyacinthGrowth :=
  {growth : ℕ → ℝ // 
    (∀ n, growth (n + 1) = 2 * growth n) ∧ 
    (growth 10 = 1)}

theorem half_pond_fill_time (g : WaterHyacinthGrowth) : 
  g.val 9 = 1/2 := by sorry

end NUMINAMATH_CALUDE_half_pond_fill_time_l3228_322855


namespace NUMINAMATH_CALUDE_largest_smallest_three_digit_div_six_with_seven_l3228_322824

/-- A function that checks if a number contains the digit 7 --/
def contains_seven (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ (a = 7 ∨ b = 7 ∨ c = 7)

/-- A function that checks if a number is a three-digit number --/
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

/-- The main theorem --/
theorem largest_smallest_three_digit_div_six_with_seven :
  (∀ n : ℕ, is_three_digit n → n % 6 = 0 → contains_seven n → n ≤ 978) ∧
  (∀ n : ℕ, is_three_digit n → n % 6 = 0 → contains_seven n → 174 ≤ n) ∧
  is_three_digit 978 ∧ 978 % 6 = 0 ∧ contains_seven 978 ∧
  is_three_digit 174 ∧ 174 % 6 = 0 ∧ contains_seven 174 :=
by sorry

end NUMINAMATH_CALUDE_largest_smallest_three_digit_div_six_with_seven_l3228_322824


namespace NUMINAMATH_CALUDE_vexel_language_words_l3228_322872

def alphabet_size : ℕ := 26
def max_word_length : ℕ := 5

def words_with_z (n : ℕ) : ℕ :=
  alphabet_size^n - (alphabet_size - 1)^n

def total_words : ℕ :=
  (words_with_z 1) + (words_with_z 2) + (words_with_z 3) + (words_with_z 4) + (words_with_z 5)

theorem vexel_language_words :
  total_words = 2205115 :=
by sorry

end NUMINAMATH_CALUDE_vexel_language_words_l3228_322872


namespace NUMINAMATH_CALUDE_factorization_equality_l3228_322862

theorem factorization_equality (x y : ℝ) : -x^2*y + 6*y*x^2 - 9*y^3 = -y*(x - 3*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3228_322862


namespace NUMINAMATH_CALUDE_solution_implies_range_l3228_322820

/-- The function f(x) = x^2 - 4x - 2 -/
def f (x : ℝ) := x^2 - 4*x - 2

/-- The theorem stating that if x^2 - 4x - 2 - a > 0 has solutions in (1,4), then a < -2 -/
theorem solution_implies_range (a : ℝ) : 
  (∃ x ∈ Set.Ioo 1 4, f x > a) → a < -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_range_l3228_322820


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3228_322800

theorem smallest_number_with_given_remainders : ∃! n : ℕ, 
  (n % 6 = 2) ∧ (n % 5 = 3) ∧ (n % 7 = 1) ∧
  (∀ m : ℕ, m < n → ¬((m % 6 = 2) ∧ (m % 5 = 3) ∧ (m % 7 = 1))) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3228_322800


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l3228_322815

theorem triangle_perimeter_bound : 
  ∀ s : ℝ, 
  s > 0 → 
  s + 7 > 23 → 
  s + 23 > 7 → 
  7 + 23 > s → 
  s + 7 + 23 < 60 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l3228_322815


namespace NUMINAMATH_CALUDE_value_of_expression_l3228_322899

theorem value_of_expression (x : ℝ) (h : x^2 - 2*x = 1) : 
  2023 + 6*x - 3*x^2 = 2020 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3228_322899


namespace NUMINAMATH_CALUDE_smallest_multiple_l3228_322827

theorem smallest_multiple (n : ℕ) : n = 544 ↔ 
  (∃ k : ℕ, n = 17 * k) ∧ 
  (∃ m : ℕ, n = 53 * m + 7) ∧ 
  (∀ x : ℕ, x < n → ¬(∃ k m : ℕ, x = 17 * k ∧ x = 53 * m + 7)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l3228_322827


namespace NUMINAMATH_CALUDE_coins_on_side_for_36_circumference_l3228_322857

/-- The number of coins on one side of a square arrangement, given the total number of coins on the circumference. -/
def coins_on_one_side (circumference_coins : ℕ) : ℕ :=
  (circumference_coins + 4) / 4

/-- Theorem stating that for a square arrangement of coins with 36 coins on the circumference, there are 10 coins on one side. -/
theorem coins_on_side_for_36_circumference :
  coins_on_one_side 36 = 10 := by
  sorry

#eval coins_on_one_side 36  -- This should output 10

end NUMINAMATH_CALUDE_coins_on_side_for_36_circumference_l3228_322857


namespace NUMINAMATH_CALUDE_equation_solution_denominator_never_zero_l3228_322842

theorem equation_solution (x : ℝ) : 
  (x + 5) / (x^2 + 4*x + 10) = 0 ↔ x = -5 :=
by sorry

theorem denominator_never_zero (x : ℝ) : 
  x^2 + 4*x + 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_denominator_never_zero_l3228_322842


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3228_322875

/-- The line equation passing through a fixed point for all values of m -/
def line_equation (m x y : ℝ) : Prop :=
  (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

/-- The fixed point P -/
def P : ℝ × ℝ := (3, 1)

/-- Theorem stating that P lies on the line for all real m -/
theorem fixed_point_on_line : ∀ m : ℝ, line_equation m P.1 P.2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3228_322875


namespace NUMINAMATH_CALUDE_max_value_abc_l3228_322881

theorem max_value_abc (a b c : ℝ) (h : a + 3 * b + c = 5) :
  ∃ (max : ℝ), max = 25/3 ∧ ∀ (x y z : ℝ), x + 3 * y + z = 5 → x * y + x * z + y * z ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_l3228_322881


namespace NUMINAMATH_CALUDE_number_satisfying_condition_l3228_322834

theorem number_satisfying_condition (x : ℝ) : x = 40 ↔ 0.65 * x = 0.05 * 60 + 23 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_condition_l3228_322834


namespace NUMINAMATH_CALUDE_pentagon_to_squares_ratio_l3228_322812

-- Define the square structure
structure Square :=
  (side : ℝ)

-- Define the pentagon structure
structure Pentagon :=
  (area : ℝ)

-- Define the theorem
theorem pentagon_to_squares_ratio 
  (s : Square) 
  (p : Pentagon) 
  (h1 : s.side > 0)
  (h2 : p.area = s.side * s.side)
  : p.area / (3 * s.side * s.side) = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_pentagon_to_squares_ratio_l3228_322812


namespace NUMINAMATH_CALUDE_water_sulfuric_oxygen_equivalence_l3228_322898

/-- Represents the number of oxygen atoms in a molecule --/
def oxygenAtoms (molecule : String) : ℕ :=
  match molecule with
  | "H2SO4" => 4
  | "H2O" => 1
  | _ => 0

/-- Theorem stating that 4n water molecules have the same number of oxygen atoms as n sulfuric acid molecules --/
theorem water_sulfuric_oxygen_equivalence (n : ℕ) :
  n * oxygenAtoms "H2SO4" = 4 * n * oxygenAtoms "H2O" :=
by sorry


end NUMINAMATH_CALUDE_water_sulfuric_oxygen_equivalence_l3228_322898


namespace NUMINAMATH_CALUDE_base_5_minus_base_7_digits_l3228_322863

-- Define the number of digits in a given base
def numDigits (n : ℕ) (base : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log base n + 1

-- State the theorem
theorem base_5_minus_base_7_digits : 
  numDigits 2023 5 - numDigits 2023 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_base_5_minus_base_7_digits_l3228_322863


namespace NUMINAMATH_CALUDE_square_ratio_side_length_sum_l3228_322859

theorem square_ratio_side_length_sum (area_ratio : ℚ) :
  area_ratio = 245 / 35 →
  ∃ (a b c : ℕ), 
    (a * (b.sqrt : ℝ) / c : ℝ) ^ 2 = area_ratio ∧
    a = 1 ∧ b = 7 ∧ c = 1 ∧
    a + b + c = 9 :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_sum_l3228_322859


namespace NUMINAMATH_CALUDE_unique_integer_proof_l3228_322826

theorem unique_integer_proof : ∃! n : ℕ+, 
  (24 ∣ n) ∧ 
  (8 < (n : ℝ) ^ (1/3)) ∧ 
  ((n : ℝ) ^ (1/3) < 8.2) ∧ 
  n = 528 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_proof_l3228_322826


namespace NUMINAMATH_CALUDE_initial_cows_l3228_322807

theorem initial_cows (cows dogs : ℕ) : 
  cows = 2 * dogs →
  (3 / 4 : ℚ) * cows + (1 / 4 : ℚ) * dogs = 161 →
  cows = 184 := by
  sorry

end NUMINAMATH_CALUDE_initial_cows_l3228_322807


namespace NUMINAMATH_CALUDE_complex_number_problem_l3228_322878

theorem complex_number_problem (z : ℂ) : 
  z + Complex.abs z = 5 + Complex.I * Real.sqrt 3 → z = 11/5 + Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3228_322878


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l3228_322892

noncomputable def f (x : ℝ) : ℝ := (x^2 - 1)^2 + 2

theorem extreme_values_of_f :
  ∃ (a b c : ℝ), 
    (a = 0 ∧ b = 1 ∧ c = -1) ∧
    (∀ x : ℝ, f x ≥ 2) ∧
    (f a = 3 ∧ f b = 2 ∧ f c = 2) ∧
    (∀ x : ℝ, x ≠ a ∧ x ≠ b ∧ x ≠ c → f x < 3) :=
by
  sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l3228_322892


namespace NUMINAMATH_CALUDE_decreasing_g_implies_a_nonpositive_l3228_322888

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x

-- Define what it means for g to be decreasing on ℝ
def isDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- Theorem statement
theorem decreasing_g_implies_a_nonpositive :
  ∀ a : ℝ, isDecreasing (g a) → a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_decreasing_g_implies_a_nonpositive_l3228_322888


namespace NUMINAMATH_CALUDE_max_parts_properties_l3228_322885

/-- The maximum number of parts that can be produced from n blanks -/
def max_parts (n : ℕ) : ℕ :=
  let rec aux (blanks remaining : ℕ) : ℕ :=
    if remaining = 0 then blanks
    else
      let new_blanks := remaining / 3
      aux (blanks + remaining) new_blanks
  aux 0 n

theorem max_parts_properties :
  (max_parts 9 = 13) ∧
  (max_parts 14 = 20) ∧
  (max_parts 27 = 40 ∧ ∀ m < 27, max_parts m < 40) := by
  sorry

end NUMINAMATH_CALUDE_max_parts_properties_l3228_322885


namespace NUMINAMATH_CALUDE_vectors_collinear_l3228_322837

def a : ℝ × ℝ × ℝ := (-1, 2, 8)
def b : ℝ × ℝ × ℝ := (3, 7, -1)
def c₁ : ℝ × ℝ × ℝ := (4 * a.1 - 3 * b.1, 4 * a.2.1 - 3 * b.2.1, 4 * a.2.2 - 3 * b.2.2)
def c₂ : ℝ × ℝ × ℝ := (9 * b.1 - 12 * a.1, 9 * b.2.1 - 12 * a.2.1, 9 * b.2.2 - 12 * a.2.2)

theorem vectors_collinear : ∃ (k : ℝ), c₁ = (k * c₂.1, k * c₂.2.1, k * c₂.2.2) := by
  sorry

end NUMINAMATH_CALUDE_vectors_collinear_l3228_322837


namespace NUMINAMATH_CALUDE_smallest_divisible_by_5_and_24_l3228_322814

theorem smallest_divisible_by_5_and_24 : ∀ n : ℕ, n > 0 → n % 5 = 0 → n % 24 = 0 → n ≥ 120 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_5_and_24_l3228_322814


namespace NUMINAMATH_CALUDE_light_bulbs_problem_l3228_322852

theorem light_bulbs_problem (initial : ℕ) : 
  (initial - 16) / 2 = 12 → initial = 40 := by
  sorry

end NUMINAMATH_CALUDE_light_bulbs_problem_l3228_322852


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3228_322867

theorem cubic_equation_roots (x : ℝ) : 
  let r1 := 2 * Real.sin (2 * Real.pi / 9)
  let r2 := 2 * Real.sin (8 * Real.pi / 9)
  let r3 := 2 * Real.sin (14 * Real.pi / 9)
  (x - r1) * (x - r2) * (x - r3) = x^3 - 3*x + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3228_322867


namespace NUMINAMATH_CALUDE_cinema_rows_l3228_322887

def base8_to_decimal (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem cinema_rows :
  let total_seats : ℕ := base8_to_decimal 351
  let seats_per_row : ℕ := 3
  (total_seats / seats_per_row : ℕ) = 77 := by
sorry

end NUMINAMATH_CALUDE_cinema_rows_l3228_322887


namespace NUMINAMATH_CALUDE_hexomino_min_containing_rectangle_area_l3228_322819

/-- A hexomino is a polyomino of 6 connected unit squares. -/
def Hexomino : Type := Unit  -- Placeholder definition

/-- The minimum area of a rectangle that contains a given hexomino. -/
def minContainingRectangleArea (h : Hexomino) : ℝ := sorry

/-- Theorem: The minimum area of any rectangle containing a hexomino is 21/2. -/
theorem hexomino_min_containing_rectangle_area (h : Hexomino) :
  minContainingRectangleArea h = 21 / 2 := by sorry

end NUMINAMATH_CALUDE_hexomino_min_containing_rectangle_area_l3228_322819


namespace NUMINAMATH_CALUDE_correct_average_calculation_l3228_322841

theorem correct_average_calculation (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 →
  incorrect_avg = 20 →
  incorrect_num = 26 →
  correct_num = 86 →
  (n : ℚ) * incorrect_avg - incorrect_num + correct_num = n * 26 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_calculation_l3228_322841


namespace NUMINAMATH_CALUDE_sum_greater_than_product_l3228_322896

theorem sum_greater_than_product (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : Real.arctan x + Real.arctan y + Real.arctan z < π) : 
  x + y + z > x * y * z := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_product_l3228_322896


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_l3228_322832

theorem x_squared_plus_y_squared (x y : ℝ) 
  (h1 : x * y = 8)
  (h2 : x^2 * y + x * y^2 + x + y = 94) :
  x^2 + y^2 = 7540 / 81 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_l3228_322832


namespace NUMINAMATH_CALUDE_smallest_cube_ending_432_l3228_322835

theorem smallest_cube_ending_432 : 
  ∀ n : ℕ+, n.val^3 % 1000 = 432 → n.val ≥ 138 := by sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_432_l3228_322835


namespace NUMINAMATH_CALUDE_six_digit_number_divisibility_l3228_322830

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Represents the six-digit number formed by appending double of a three-digit number -/
def makeSixDigitNumber (n : ThreeDigitNumber) : Nat :=
  1000 * n.toNat + 2 * n.toNat

theorem six_digit_number_divisibility (n : ThreeDigitNumber) :
  (∃ k : Nat, makeSixDigitNumber n = 2 * k) ∧
  (∃ m : Nat, makeSixDigitNumber n = 3 * m ↔ ∃ l : Nat, n.toNat = 3 * l) :=
sorry

end NUMINAMATH_CALUDE_six_digit_number_divisibility_l3228_322830


namespace NUMINAMATH_CALUDE_last_digit_product_l3228_322811

/-- The last digit of (3^65 * 6^n * 7^71) is 4 for any non-negative integer n. -/
theorem last_digit_product (n : ℕ) : (3^65 * 6^n * 7^71) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_product_l3228_322811


namespace NUMINAMATH_CALUDE_pure_imaginary_modulus_l3228_322871

theorem pure_imaginary_modulus (b : ℝ) : 
  (Complex.I : ℂ).re * ((2 + b * Complex.I) * (2 - Complex.I)).re = 0 ∧ 
  (Complex.I : ℂ).im * ((2 + b * Complex.I) * (2 - Complex.I)).im ≠ 0 → 
  Complex.abs (1 + b * Complex.I) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_modulus_l3228_322871


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3228_322884

theorem inequality_solution_set (x : ℝ) :
  x * (2 * x^2 - 3 * x + 1) ≤ 0 ↔ x ≤ 0 ∨ (1/2 ≤ x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3228_322884


namespace NUMINAMATH_CALUDE_exists_tangent_circle_l3228_322870

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of two circles intersecting
def intersects (c1 c2 : Circle) : Prop :=
  ∃ (p : ℝ × ℝ), (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∧
                 (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2

-- Define the property of a point being on a circle
def onCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the property of a circle being tangent to another circle
def isTangent (c1 c2 : Circle) : Prop :=
  ∃ (p : ℝ × ℝ), onCircle p c1 ∧ onCircle p c2 ∧
    ∀ (q : ℝ × ℝ), q ≠ p → ¬(onCircle q c1 ∧ onCircle q c2)

-- Theorem statement
theorem exists_tangent_circle (S₁ S₂ S₃ : Circle) (O : ℝ × ℝ) :
  intersects S₁ S₂ ∧ intersects S₂ S₃ ∧ intersects S₃ S₁ ∧
  onCircle O S₁ ∧ onCircle O S₂ ∧ onCircle O S₃ →
  ∃ (S : Circle), isTangent S S₁ ∧ isTangent S S₂ ∧ isTangent S S₃ :=
sorry

end NUMINAMATH_CALUDE_exists_tangent_circle_l3228_322870


namespace NUMINAMATH_CALUDE_max_a_no_lattice_points_l3228_322847

def is_lattice_point (x y : ℚ) : Prop := ∃ (n m : ℤ), x = n ∧ y = m

theorem max_a_no_lattice_points :
  ∃ (a : ℚ), a = 17/51 ∧
  (∀ (m x : ℚ), 1/3 < m → m < a → 0 < x → x ≤ 50 → 
    ¬ is_lattice_point x (m * x + 3)) ∧
  (∀ (a' : ℚ), a' > a → 
    ∃ (m x : ℚ), 1/3 < m → m < a' → 0 < x → x ≤ 50 → 
      is_lattice_point x (m * x + 3)) :=
sorry

end NUMINAMATH_CALUDE_max_a_no_lattice_points_l3228_322847


namespace NUMINAMATH_CALUDE_rohan_monthly_salary_l3228_322802

/-- Rohan's monthly expenses and savings --/
structure RohanFinances where
  food_percent : ℝ
  rent_percent : ℝ
  entertainment_percent : ℝ
  conveyance_percent : ℝ
  taxes_percent : ℝ
  miscellaneous_percent : ℝ
  savings : ℝ

/-- Theorem: Rohan's monthly salary calculation --/
theorem rohan_monthly_salary (r : RohanFinances) 
  (h1 : r.food_percent = 0.40)
  (h2 : r.rent_percent = 0.20)
  (h3 : r.entertainment_percent = 0.10)
  (h4 : r.conveyance_percent = 0.10)
  (h5 : r.taxes_percent = 0.05)
  (h6 : r.miscellaneous_percent = 0.07)
  (h7 : r.savings = 1000) :
  ∃ (salary : ℝ), salary = 12500 ∧ 
    (1 - (r.food_percent + r.rent_percent + r.entertainment_percent + 
          r.conveyance_percent + r.taxes_percent + r.miscellaneous_percent)) * salary = r.savings :=
by sorry


end NUMINAMATH_CALUDE_rohan_monthly_salary_l3228_322802


namespace NUMINAMATH_CALUDE_probability_three_tails_l3228_322890

def coin_flips : ℕ := 8
def p_tails : ℚ := 3/5
def p_heads : ℚ := 2/5
def num_tails : ℕ := 3

theorem probability_three_tails :
  (Nat.choose coin_flips num_tails : ℚ) * p_tails ^ num_tails * p_heads ^ (coin_flips - num_tails) = 48624/390625 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_tails_l3228_322890


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_property_l3228_322845

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a straight line -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

/-- Checks if a point lies on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Checks if a point lies on a line -/
def on_line (l : Line) (p : Point) : Prop :=
  p.y = l.m * p.x + l.c

/-- Checks if a point lies on an asymptote of the hyperbola -/
def on_asymptote (h : Hyperbola) (p : Point) : Prop :=
  p.y = (h.b / h.a) * p.x ∨ p.y = -(h.b / h.a) * p.x

/-- The main theorem -/
theorem hyperbola_line_intersection_property
  (h : Hyperbola) (l : Line)
  (p q p' q' : Point)
  (hp : on_hyperbola h p)
  (hq : on_hyperbola h q)
  (hp' : on_asymptote h p')
  (hq' : on_asymptote h q')
  (hlp : on_line l p)
  (hlq : on_line l q)
  (hlp' : on_line l p')
  (hlq' : on_line l q') :
  |p.x - p'.x| = |q.x - q'.x| ∧ |p.y - p'.y| = |q.y - q'.y| :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_property_l3228_322845


namespace NUMINAMATH_CALUDE_line_plane_perp_sufficiency_not_necessity_l3228_322866

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicularity relation between a line and a plane
variable (line_perp_plane : Line → Plane → Prop)

-- Define the perpendicularity relation between two planes
variable (plane_perp_plane : Plane → Plane → Prop)

-- Define the relation of a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem line_plane_perp_sufficiency_not_necessity
  (α β : Plane) (m : Line)
  (h_diff : α ≠ β)
  (h_m_in_α : line_in_plane m α) :
  (line_perp_plane m β → plane_perp_plane α β) ∧
  ¬(plane_perp_plane α β → line_perp_plane m β) :=
sorry

end NUMINAMATH_CALUDE_line_plane_perp_sufficiency_not_necessity_l3228_322866


namespace NUMINAMATH_CALUDE_binomial_coefficient_x3y5_in_x_plus_y_8_l3228_322836

theorem binomial_coefficient_x3y5_in_x_plus_y_8 :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k) * (1 : ℕ)^k * (1 : ℕ)^(8 - k)) = 256 ∧
  (Nat.choose 8 3) = 56 :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_x3y5_in_x_plus_y_8_l3228_322836


namespace NUMINAMATH_CALUDE_smallest_divisible_by_fractions_l3228_322893

def is_divisible_by_fraction (n : ℕ) (a b : ℕ) : Prop :=
  ∃ k : ℕ, n * b = k * a

theorem smallest_divisible_by_fractions :
  ∀ n : ℕ, n > 0 →
    (is_divisible_by_fraction n 8 33 ∧
     is_divisible_by_fraction n 7 22 ∧
     is_divisible_by_fraction n 15 26) →
    n ≥ 120 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_fractions_l3228_322893
