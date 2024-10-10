import Mathlib

namespace crazy_silly_school_books_l3472_347291

/-- The number of books in the 'Crazy Silly School' series -/
def total_books : ℕ := 13

/-- The number of books already read -/
def books_read : ℕ := 9

/-- The number of books left to read -/
def books_left : ℕ := 4

/-- Theorem stating that the total number of books is equal to the sum of books read and books left -/
theorem crazy_silly_school_books : 
  total_books = books_read + books_left := by
  sorry

end crazy_silly_school_books_l3472_347291


namespace last_digit_of_product_l3472_347272

def last_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem last_digit_of_product (B : ℤ) : 
  B ≥ 0 ∧ B ≤ 9 →
  (last_digit (287 * 287 + B * B - 2 * 287 * B) = 4 ↔ B = 5 ∨ B = 9) :=
by sorry

end last_digit_of_product_l3472_347272


namespace factorization_proofs_l3472_347230

theorem factorization_proofs (x y : ℝ) :
  (x^2*y - 2*x*y + x*y^2 = x*y*(x - 2 + y)) ∧
  (x^2 - 3*x + 2 = (x - 1)*(x - 2)) ∧
  (4*x^4 - 64 = 4*(x^2 + 4)*(x + 2)*(x - 2)) := by
  sorry

end factorization_proofs_l3472_347230


namespace sqrt_neg_four_squared_l3472_347209

theorem sqrt_neg_four_squared : Real.sqrt ((-4)^2) = 4 := by
  sorry

end sqrt_neg_four_squared_l3472_347209


namespace point_transformation_identity_l3472_347236

def rotateZ90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflectXY (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def rotateX90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -z, y)

def reflectYZ (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

theorem point_transformation_identity :
  let initial_point : ℝ × ℝ × ℝ := (2, 2, 2)
  let transformed_point := reflectYZ (rotateX90 (reflectXY (rotateZ90 initial_point)))
  transformed_point = initial_point := by
  sorry

end point_transformation_identity_l3472_347236


namespace part_one_part_two_l3472_347216

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := |2 * x + b|

-- Part I
theorem part_one (b : ℝ) : 
  (∀ x, f b x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 2) → b = -1 := by sorry

-- Part II
theorem part_two :
  ∃ m : ℝ, ∀ x : ℝ, f (-1) (x + 3) + f (-1) (x + 1) ≥ m ∧ 
  ¬∃ m' : ℝ, (m' < m ∧ ∀ x : ℝ, f (-1) (x + 3) + f (-1) (x + 1) ≥ m') := by sorry

end part_one_part_two_l3472_347216


namespace true_propositions_l3472_347218

-- Define proposition p
def p : Prop := ∀ x y : ℝ, x > y → -x < -y

-- Define proposition q
def q : Prop := ∀ x y : ℝ, x < y → x^2 > y^2

-- Define the four compound propositions
def prop1 : Prop := p ∧ q
def prop2 : Prop := p ∨ q
def prop3 : Prop := p ∧ (¬q)
def prop4 : Prop := (¬p) ∨ q

-- Theorem stating which propositions are true
theorem true_propositions : prop2 ∧ prop3 ∧ ¬prop1 ∧ ¬prop4 := by
  sorry

end true_propositions_l3472_347218


namespace mary_found_four_eggs_l3472_347282

/-- The number of eggs Mary started with -/
def initial_eggs : ℕ := 27

/-- The number of eggs Mary ended up with -/
def final_eggs : ℕ := 31

/-- The number of eggs Mary found -/
def found_eggs : ℕ := final_eggs - initial_eggs

theorem mary_found_four_eggs : found_eggs = 4 := by
  sorry

end mary_found_four_eggs_l3472_347282


namespace largest_integer_less_than_100_remainder_5_mod_8_l3472_347226

theorem largest_integer_less_than_100_remainder_5_mod_8 :
  ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
by
  -- The proof goes here
  sorry

end largest_integer_less_than_100_remainder_5_mod_8_l3472_347226


namespace new_drive_free_space_calculation_l3472_347286

/-- Calculates the free space on a new external drive after file operations -/
def new_drive_free_space (initial_free : ℝ) (initial_used : ℝ) (deleted1 : ℝ) (deleted2 : ℝ) (added1 : ℝ) (added2 : ℝ) (new_drive_size : ℝ) : ℝ :=
  new_drive_size - (initial_used - (deleted1 + deleted2) + (added1 + added2))

/-- Theorem stating that the free space on the new drive is 313.5 GB -/
theorem new_drive_free_space_calculation :
  new_drive_free_space 75.8 210.3 34.5 29.7 13 27.4 500 = 313.5 := by
  sorry

#eval new_drive_free_space 75.8 210.3 34.5 29.7 13 27.4 500

end new_drive_free_space_calculation_l3472_347286


namespace expression_equals_seven_l3472_347215

theorem expression_equals_seven (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a * b + a * c + b * c)) = 7 := by
  sorry

end expression_equals_seven_l3472_347215


namespace x_range_given_sqrt_equality_l3472_347261

theorem x_range_given_sqrt_equality (x : ℝ) :
  Real.sqrt ((5 - x)^2) = x - 5 → x ≥ 5 := by
  sorry

end x_range_given_sqrt_equality_l3472_347261


namespace triangle_area_l3472_347207

/-- The area of a triangle with side lengths √29, √13, and √34 is 19/2 -/
theorem triangle_area (a b c : ℝ) (ha : a = Real.sqrt 29) (hb : b = Real.sqrt 13) (hc : c = Real.sqrt 34) :
  (1/2) * b * c * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*b*c))^2) = 19/2 := by
  sorry

end triangle_area_l3472_347207


namespace sqrt_equality_implies_one_and_six_l3472_347224

theorem sqrt_equality_implies_one_and_six (a b : ℕ) (ha : a > 0) (hb : b > 0) (hlt : a < b) :
  (Real.sqrt (1 + Real.sqrt (40 + 24 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b) →
  a = 1 ∧ b = 6 := by sorry

end sqrt_equality_implies_one_and_six_l3472_347224


namespace elon_has_13_teslas_l3472_347278

/-- The number of Teslas Chris has -/
def chris_teslas : ℕ := 6

/-- The number of Teslas Sam has -/
def sam_teslas : ℕ := chris_teslas / 2

/-- The number of Teslas Elon has -/
def elon_teslas : ℕ := sam_teslas + 10

theorem elon_has_13_teslas : elon_teslas = 13 := by
  sorry

end elon_has_13_teslas_l3472_347278


namespace bookstore_new_releases_fraction_l3472_347251

theorem bookstore_new_releases_fraction 
  (total_books : ℕ) 
  (historical_fiction_ratio : ℚ) 
  (historical_fiction_new_release_ratio : ℚ) 
  (other_new_release_ratio : ℚ) 
  (h1 : historical_fiction_ratio = 30 / 100)
  (h2 : historical_fiction_new_release_ratio = 40 / 100)
  (h3 : other_new_release_ratio = 50 / 100)
  (h4 : total_books > 0) :
  let historical_fiction_books := total_books * historical_fiction_ratio
  let historical_fiction_new_releases := historical_fiction_books * historical_fiction_new_release_ratio
  let other_books := total_books - historical_fiction_books
  let other_new_releases := other_books * other_new_release_ratio
  let total_new_releases := historical_fiction_new_releases + other_new_releases
  (historical_fiction_new_releases / total_new_releases : ℚ) = 12 / 47 := by
sorry

end bookstore_new_releases_fraction_l3472_347251


namespace rectangle_fold_theorem_l3472_347280

/-- Given a rectangle ABCD with AB = 4 and BC = 10, folded along a line through A
    such that A meets CD at point G where DG = 3, and C touches the extension of AB at point E,
    prove that the length of segment CE is 1. -/
theorem rectangle_fold_theorem (A B C D G E : ℝ × ℝ) : 
  let AB : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC : ℝ := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let DG : ℝ := Real.sqrt ((D.1 - G.1)^2 + (D.2 - G.2)^2)
  let CE : ℝ := Real.sqrt ((C.1 - E.1)^2 + (C.2 - E.2)^2)
  AB = 4 →
  BC = 10 →
  DG = 3 →
  (A.1 - B.1) * (C.1 - D.1) + (A.2 - B.2) * (C.2 - D.2) = 0 → -- rectangle condition
  (A.1 = G.1 ∧ A.2 = G.2) → -- A meets CD at G
  (E.1 - A.1) * (B.1 - A.1) + (E.2 - A.2) * (B.2 - A.2) ≥ 0 → -- C touches extension of AB
  CE = 1 := by
sorry

end rectangle_fold_theorem_l3472_347280


namespace ray_return_characterization_l3472_347277

/-- Represents a point in the triangular grid --/
structure GridPoint where
  a : ℕ
  b : ℕ

/-- Represents an equilateral triangle --/
structure EquilateralTriangle where
  sideLength : ℕ

/-- Checks if a GridPoint is on the triangular grid --/
def isOnGrid (p : GridPoint) : Prop :=
  p.a ≡ p.b [MOD 3]

/-- Checks if a line from origin to GridPoint doesn't pass through other grid points --/
def isDirectPath (p : GridPoint) : Prop :=
  Nat.gcd p.a p.b = 1

/-- Calculates the number of bounces for a ray to reach a GridPoint --/
def numberOfBounces (p : GridPoint) : ℕ :=
  2 * (p.a + p.b) - 3

/-- Theorem: Characterization of valid number of bounces for ray to return to A --/
theorem ray_return_characterization (n : ℕ) :
  (∃ (t : EquilateralTriangle) (p : GridPoint), 
    isOnGrid p ∧ isDirectPath p ∧ numberOfBounces p = n) ↔ 
  (n ≡ 1 [MOD 6] ∨ n ≡ 5 [MOD 6]) ∧ n ≠ 5 ∧ n ≠ 17 :=
sorry

end ray_return_characterization_l3472_347277


namespace line_vector_to_slope_intercept_l3472_347233

/-- Given a line in vector form, prove its slope-intercept form and find (m, b) -/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y + 4) = 0 →
  y = 2 * x - 10 ∧ (2, -10) = (2, -10) := by
  sorry

end line_vector_to_slope_intercept_l3472_347233


namespace quarterly_to_annual_compound_interest_l3472_347274

/-- Given an annual interest rate of 8% compounded quarterly, 
    prove that it's equivalent to an 8.24% annual rate compounded annually. -/
theorem quarterly_to_annual_compound_interest : 
  let quarterly_rate : ℝ := 0.08 / 4
  let effective_annual_rate : ℝ := (1 + quarterly_rate)^4 - 1
  ∀ ε > 0, |effective_annual_rate - 0.0824| < ε :=
by
  sorry

end quarterly_to_annual_compound_interest_l3472_347274


namespace function_inequality_implies_parameter_bound_l3472_347247

open Real

theorem function_inequality_implies_parameter_bound 
  (f g : ℝ → ℝ) 
  (h : ∀ x > 0, f x = 2 * x * log x ∧ g x = -x^2 + a * x - 3) 
  (h2 : ∀ x > 0, f x > g x) : 
  a < 4 := by
sorry

end function_inequality_implies_parameter_bound_l3472_347247


namespace sixth_term_is_36_l3472_347203

/-- The sequence of squares of natural numbers from 1 to 7 -/
def square_sequence : Fin 7 → ℕ := fun n => (n + 1)^2

/-- The 6th term of the square sequence is 36 -/
theorem sixth_term_is_36 : square_sequence 5 = 36 := by
  sorry

end sixth_term_is_36_l3472_347203


namespace max_m_value_l3472_347242

theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, x < m → x^2 - 2*x - 8 > 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x - 8 > 0 ∧ x ≥ m) ∧
  (∀ m' : ℝ, m' > m → 
    ¬((∀ x : ℝ, x < m' → x^2 - 2*x - 8 > 0) ∧ 
      (∃ x : ℝ, x^2 - 2*x - 8 > 0 ∧ x ≥ m'))) →
  m = 4 := by sorry

end max_m_value_l3472_347242


namespace min_value_theorem_l3472_347248

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  1/(a-1) + 4/(b-1) ≥ 4 := by
sorry

end min_value_theorem_l3472_347248


namespace jane_stopped_babysitting_16_years_ago_l3472_347219

/-- Represents a person with their current age and the age they started babysitting -/
structure Babysitter where
  current_age : ℕ
  start_age : ℕ

/-- Represents a person who was babysat -/
structure BabysatPerson where
  current_age : ℕ

def Babysitter.max_babysat_age (b : Babysitter) : ℕ := b.current_age / 2

def years_since_stopped_babysitting (b : Babysitter) (p : BabysatPerson) : ℕ :=
  b.current_age - p.current_age

theorem jane_stopped_babysitting_16_years_ago
  (jane : Babysitter)
  (oldest_babysat : BabysatPerson)
  (h1 : jane.current_age = 32)
  (h2 : jane.start_age = 16)
  (h3 : oldest_babysat.current_age = 24)
  (h4 : oldest_babysat.current_age ≤ jane.max_babysat_age) :
  years_since_stopped_babysitting jane oldest_babysat = 16 := by
  sorry

#check jane_stopped_babysitting_16_years_ago

end jane_stopped_babysitting_16_years_ago_l3472_347219


namespace range_of_m_l3472_347222

theorem range_of_m (x y m : ℝ) : 
  (2 * x - y = 5 * m) →
  (3 * x + 4 * y = 2 * m) →
  (x + y ≤ 5) →
  (2 * x + 7 * y < 18) →
  (-6 < m ∧ m ≤ 5) :=
by sorry

end range_of_m_l3472_347222


namespace mnp_value_l3472_347266

theorem mnp_value (a b x y : ℝ) (m n p : ℤ) 
  (h1 : a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1))
  (h2 : (a^m*x - a^n)*(a^p*y - a^3) = a^5*b^5) :
  m * n * p = 12 := by
  sorry

end mnp_value_l3472_347266


namespace allowance_spent_on_games_l3472_347231

theorem allowance_spent_on_games (total : ℝ) (books_frac snacks_frac music_frac : ℝ) : 
  total = 50 ∧ 
  books_frac = 1/4 ∧ 
  snacks_frac = 1/5 ∧ 
  music_frac = 2/5 → 
  total - (books_frac * total + snacks_frac * total + music_frac * total) = 7.5 := by
sorry

end allowance_spent_on_games_l3472_347231


namespace sequence_existence_and_bound_l3472_347287

theorem sequence_existence_and_bound (a : ℝ) (n : ℕ) :
  ∃! x : ℕ → ℝ, 
    (x 1 - x (n - 1) = 0) ∧ 
    (∀ i ∈ Finset.range n, (x (i - 1) + x i) / 2 = x i + (x i)^3 - a^3) ∧
    (∀ i ∈ Finset.range (n + 2), |x i| ≤ |a|) := by
  sorry

end sequence_existence_and_bound_l3472_347287


namespace cube_volume_percentage_l3472_347279

def box_length : ℕ := 8
def box_width : ℕ := 6
def box_height : ℕ := 12
def cube_side : ℕ := 4

def cubes_per_length : ℕ := box_length / cube_side
def cubes_per_width : ℕ := box_width / cube_side
def cubes_per_height : ℕ := box_height / cube_side

def total_cubes : ℕ := cubes_per_length * cubes_per_width * cubes_per_height

def cube_volume : ℕ := cube_side ^ 3
def total_cube_volume : ℕ := total_cubes * cube_volume

def box_volume : ℕ := box_length * box_width * box_height

theorem cube_volume_percentage :
  (total_cube_volume : ℚ) / (box_volume : ℚ) * 100 = 200 / 3 := by sorry

end cube_volume_percentage_l3472_347279


namespace inequality_and_equality_condition_l3472_347240

theorem inequality_and_equality_condition (x₁ x₂ : ℝ) 
  (h₁ : |x₁| ≤ 1) (h₂ : |x₂| ≤ 1) : 
  Real.sqrt (1 - x₁^2) + Real.sqrt (1 - x₂^2) ≤ 2 * Real.sqrt (1 - ((x₁ + x₂)/2)^2) ∧ 
  (Real.sqrt (1 - x₁^2) + Real.sqrt (1 - x₂^2) = 2 * Real.sqrt (1 - ((x₁ + x₂)/2)^2) ↔ x₁ = x₂) :=
by sorry

end inequality_and_equality_condition_l3472_347240


namespace inequality_not_always_true_l3472_347260

theorem inequality_not_always_true (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ¬ (∀ a b, a > b ∧ b > 0 → a + 1/a < b + 1/b) :=
by sorry

end inequality_not_always_true_l3472_347260


namespace unsold_bars_l3472_347252

theorem unsold_bars (total_bars : ℕ) (price_per_bar : ℕ) (total_sold : ℕ) :
  total_bars = 13 →
  price_per_bar = 6 →
  total_sold = 42 →
  total_bars - (total_sold / price_per_bar) = 6 := by
sorry

end unsold_bars_l3472_347252


namespace locus_is_hyperbola_branch_l3472_347254

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

-- Define a circle tangent to both circles
def tangentCircle (cx cy r : ℝ) : Prop :=
  ∀ (x y : ℝ), circle1 x y → (x - cx)^2 + (y - cy)^2 = r^2 ∨
                circle2 x y → (x - cx)^2 + (y - cy)^2 = r^2

-- Define the locus of centers
def locusOfCenters (x y : ℝ) : Prop :=
  ∃ (r : ℝ), tangentCircle x y r

-- Theorem statement
theorem locus_is_hyperbola_branch :
  ∃ (a b : ℝ), ∀ (x y : ℝ), locusOfCenters x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1 :=
sorry

end locus_is_hyperbola_branch_l3472_347254


namespace problem_solution_l3472_347297

theorem problem_solution (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) :
  a^2 - b^2 + 2*a*b = 64 := by
sorry

end problem_solution_l3472_347297


namespace second_polygon_sides_l3472_347284

theorem second_polygon_sides (p1 p2 : ℕ) (s : ℝ) :
  p1 = 50 →                          -- First polygon has 50 sides
  p1 * (3 * s) = p2 * s →            -- Same perimeter
  3 * s > 0 →                        -- Positive side length
  p2 = 150 := by sorry

end second_polygon_sides_l3472_347284


namespace sqrt_x_minus_9_meaningful_l3472_347258

theorem sqrt_x_minus_9_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 9) ↔ x ≥ 9 := by
  sorry

end sqrt_x_minus_9_meaningful_l3472_347258


namespace combined_distance_is_twelve_l3472_347245

/-- The combined distance walked by two ladies in Central Park -/
def combined_distance (distance_second : ℝ) : ℝ :=
  distance_second + 2 * distance_second

/-- Theorem: The combined distance walked by two ladies is 12 miles -/
theorem combined_distance_is_twelve : combined_distance 4 = 12 := by
  sorry

end combined_distance_is_twelve_l3472_347245


namespace team7_cups_l3472_347257

-- Define the number of teams
def num_teams : Nat := 7

-- Define the total amount of soup required
def total_soup : Nat := 2500

-- Define the amount made by the first team
def first_team : Nat := 450

-- Define the amount made by the second team
def second_team : Nat := 300

-- Define the relationship between teams 3+4 and team 7
def teams_34_7_relation (team7 : Nat) : Nat := 2 * team7

-- Define the relationship between teams 1+2 and teams 5+6
def teams_12_56_relation : Nat := first_team + second_team

-- Define the function to calculate the total soup made by all teams
def total_soup_made (team7 : Nat) : Nat :=
  first_team + second_team + teams_34_7_relation team7 + teams_12_56_relation + team7

-- Theorem stating that team 7 should prepare 334 cups to meet the total required
theorem team7_cups : ∃ (team7 : Nat), team7 = 334 ∧ total_soup_made team7 = total_soup := by
  sorry

end team7_cups_l3472_347257


namespace square_sum_range_l3472_347223

theorem square_sum_range (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + y = 1) :
  1/2 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 1 := by
  sorry

end square_sum_range_l3472_347223


namespace sum_a_b_range_m_solve_equation_l3472_347290

-- Define the system of equations
def system (a b m : ℝ) : Prop :=
  a + 2*b = 4 ∧ 2*a + b = 3 - m

-- Theorem 1: Express a + b in terms of m
theorem sum_a_b (a b m : ℝ) :
  system a b m → a + b = (7 - m) / 3 := by sorry

-- Theorem 2: Find the range of values for m
theorem range_m (a b m : ℝ) :
  system a b m → a - b > -4 → m < 3 := by sorry

-- Theorem 3: Solve the equation for positive integer m
theorem solve_equation (m : ℕ) (x : ℝ) :
  m < 3 → (m * x - (1 - x) / 2 = 5 ↔ x = 11/3 ∨ x = 2.2) := by sorry

end sum_a_b_range_m_solve_equation_l3472_347290


namespace problem_statement_l3472_347299

theorem problem_statement (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a + b - 1 / (2 * a) - 2 / b = 3 / 2) 
  (h4 : a < 1) : 
  b > 2 := by
sorry

end problem_statement_l3472_347299


namespace sum_of_squared_coefficients_l3472_347283

def original_expression (y : ℝ) : ℝ := 3 * (y^3 - 2*y^2 + 3) - 5 * (y^2 - 4*y + 2)

def simplified_expression (y : ℝ) : ℝ := 3*y^3 - 11*y^2 + 20*y - 1

theorem sum_of_squared_coefficients :
  (3^2 + (-11)^2 + 20^2 + (-1)^2 = 531) ∧
  (∀ y : ℝ, original_expression y = simplified_expression y) :=
by sorry

end sum_of_squared_coefficients_l3472_347283


namespace sum_of_digits_power_of_two_l3472_347239

-- Define s(n) as the sum of digits of n
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Define the property that n - s(n) is divisible by 9
def divisible_by_nine (n : ℕ) : Prop :=
  ∃ k : ℕ, n - sum_of_digits n = 9 * k

-- State the theorem
theorem sum_of_digits_power_of_two :
  (∀ n : ℕ, divisible_by_nine n) →
  2^2009 % 9 = 5 →
  sum_of_digits (sum_of_digits (sum_of_digits (2^2009))) < 9 →
  sum_of_digits (sum_of_digits (sum_of_digits (2^2009))) = 5 := by
  sorry

end sum_of_digits_power_of_two_l3472_347239


namespace cat_dog_ratio_l3472_347235

def kennel (num_dogs : ℕ) (num_cats : ℕ) : Prop :=
  num_cats = num_dogs - 6 ∧ num_dogs = 18

theorem cat_dog_ratio (num_dogs num_cats : ℕ) :
  kennel num_dogs num_cats →
  (num_cats : ℚ) / (num_dogs : ℚ) = 2 / 3 := by
  sorry

end cat_dog_ratio_l3472_347235


namespace quadratic_roots_of_nine_l3472_347289

theorem quadratic_roots_of_nine (x : ℝ) : x^2 = 9 ↔ x = 3 ∨ x = -3 := by
  sorry

end quadratic_roots_of_nine_l3472_347289


namespace not_outperformed_iff_ge_five_l3472_347293

/-- A directed graph representing a table tennis tournament. -/
structure TournamentGraph (n : ℕ) where
  (edges : Fin n → Fin n → Prop)
  (complete : ∀ i j : Fin n, i ≠ j → edges i j ∨ edges j i)

/-- Player i is not out-performed by player j. -/
def not_outperformed {n : ℕ} (G : TournamentGraph n) (i j : Fin n) : Prop :=
  ∃ k : Fin n, G.edges i k ∧ ¬G.edges j k

/-- The tournament satisfies the not out-performed condition for all players. -/
def all_not_outperformed (n : ℕ) : Prop :=
  ∃ G : TournamentGraph n, ∀ i j : Fin n, i ≠ j → not_outperformed G i j

/-- The main theorem: the not out-performed condition holds if and only if n ≥ 5. -/
theorem not_outperformed_iff_ge_five :
  ∀ n : ℕ, n ≥ 3 → (all_not_outperformed n ↔ n ≥ 5) :=
sorry

end not_outperformed_iff_ge_five_l3472_347293


namespace min_value_sum_reciprocals_and_cube_root_l3472_347227

theorem min_value_sum_reciprocals_and_cube_root (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  1/x + 1/y + 1/z + (x*y*z)^(1/3 : ℝ) ≥ 4 ∧ 
  (1/x + 1/y + 1/z + (x*y*z)^(1/3 : ℝ) = 4 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end min_value_sum_reciprocals_and_cube_root_l3472_347227


namespace frog_ratio_l3472_347229

/-- Given two ponds A and B with frogs, prove the ratio of frogs in A to B -/
theorem frog_ratio (total : ℕ) (pond_a : ℕ) (h1 : total = 48) (h2 : pond_a = 32) :
  (pond_a : ℚ) / ((total - pond_a) : ℚ) = 2 / 1 := by
  sorry

end frog_ratio_l3472_347229


namespace opposite_of_sqrt7_minus_3_l3472_347262

theorem opposite_of_sqrt7_minus_3 : 
  -(Real.sqrt 7 - 3) = 3 - Real.sqrt 7 := by sorry

end opposite_of_sqrt7_minus_3_l3472_347262


namespace students_in_both_groups_l3472_347256

theorem students_in_both_groups 
  (total : ℕ) 
  (math : ℕ) 
  (english : ℕ) 
  (h1 : total = 52) 
  (h2 : math = 32) 
  (h3 : english = 40) : 
  total = math + english - 20 :=
by sorry

end students_in_both_groups_l3472_347256


namespace item_cost_calculation_l3472_347212

theorem item_cost_calculation (total_items : ℕ) (total_cost : ℕ) : 
  total_items = 15 → total_cost = 30 → (total_cost / total_items : ℚ) = 2 := by
  sorry

end item_cost_calculation_l3472_347212


namespace candy_count_correct_l3472_347232

/-- Represents the number of pieces in each box of chocolates -/
def chocolate_boxes : List Nat := [500, 350, 700, 400, 450, 600]

/-- Represents the number of pieces in each box of lollipops -/
def lollipop_boxes : List Nat := [200, 300, 250, 350]

/-- Represents the number of pieces in each box of gummy bears -/
def gummy_bear_boxes : List Nat := [500, 550, 400, 600, 450]

/-- The total number of candy pieces in all boxes -/
def total_candies : Nat :=
  chocolate_boxes.sum + lollipop_boxes.sum + gummy_bear_boxes.sum

theorem candy_count_correct : total_candies = 6600 := by
  sorry

end candy_count_correct_l3472_347232


namespace stellas_album_has_50_pages_l3472_347273

/-- Calculates the number of pages in Stella's stamp album --/
def stellas_album_pages (stamps_per_first_page : ℕ) (stamps_per_other_page : ℕ) (total_stamps : ℕ) : ℕ :=
  let first_pages := 10
  let stamps_in_first_pages := first_pages * stamps_per_first_page
  let remaining_stamps := total_stamps - stamps_in_first_pages
  let other_pages := remaining_stamps / stamps_per_other_page
  first_pages + other_pages

/-- Theorem stating that Stella's album has 50 pages --/
theorem stellas_album_has_50_pages :
  stellas_album_pages (5 * 30) 50 3500 = 50 := by
  sorry

#eval stellas_album_pages (5 * 30) 50 3500

end stellas_album_has_50_pages_l3472_347273


namespace complex_fraction_simplification_l3472_347263

theorem complex_fraction_simplification : 
  let numerator := (11^4 + 400) * (25^4 + 400) * (37^4 + 400) * (49^4 + 400) * (61^4 + 400)
  let denominator := (5^4 + 400) * (17^4 + 400) * (29^4 + 400) * (41^4 + 400) * (53^4 + 400)
  numerator / denominator = 799 := by
  sorry

end complex_fraction_simplification_l3472_347263


namespace remainder_of_3_pow_20_mod_5_l3472_347220

theorem remainder_of_3_pow_20_mod_5 : 3^20 % 5 = 1 := by
  sorry

end remainder_of_3_pow_20_mod_5_l3472_347220


namespace plate_price_l3472_347271

/-- Given the conditions of Chenny's purchase, prove that each plate costs $2 -/
theorem plate_price (num_plates : ℕ) (spoon_price : ℚ) (num_spoons : ℕ) (total_paid : ℚ) :
  num_plates = 9 →
  spoon_price = 3/2 →
  num_spoons = 4 →
  total_paid = 24 →
  ∃ (plate_price : ℚ), plate_price * num_plates + spoon_price * num_spoons = total_paid ∧ plate_price = 2 := by
  sorry

end plate_price_l3472_347271


namespace fort_blocks_theorem_l3472_347264

/-- Represents the dimensions of a rectangular fort -/
structure FortDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks needed to build a fort with given dimensions and wall thickness -/
def blocksNeeded (d : FortDimensions) (wallThickness : ℕ) : ℕ :=
  let outerVolume := d.length * d.width * d.height
  let innerLength := d.length - 2 * wallThickness
  let innerWidth := d.width - 2 * wallThickness
  let innerHeight := d.height - wallThickness
  let innerVolume := innerLength * innerWidth * innerHeight
  outerVolume - innerVolume

/-- Theorem stating that a fort with given dimensions requires 480 blocks -/
theorem fort_blocks_theorem :
  let fortDimensions : FortDimensions := ⟨15, 8, 6⟩
  let wallThickness : ℕ := 3/2
  blocksNeeded fortDimensions wallThickness = 480 := by
  sorry

end fort_blocks_theorem_l3472_347264


namespace arithmetic_computation_l3472_347244

theorem arithmetic_computation : -7 * 3 - (-5 * -4) + (-9 * -6) = 13 := by
  sorry

end arithmetic_computation_l3472_347244


namespace arithmetic_sequence_sum_2012_l3472_347295

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  first_term : a 1 = -2012
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (n : ℤ) * seq.a 1 + (n * (n - 1) : ℤ) * (seq.a 2 - seq.a 1) / 2

theorem arithmetic_sequence_sum_2012 (seq : ArithmeticSequence) 
    (h : (sum_n seq 12 / 12 : ℚ) - (sum_n seq 10 / 10 : ℚ) = 2) :
    sum_n seq 2012 = -2012 := by
  sorry

end arithmetic_sequence_sum_2012_l3472_347295


namespace jeans_to_janes_money_ratio_l3472_347268

theorem jeans_to_janes_money_ratio (total : ℕ) (jeans_money : ℕ) :
  total = 76 →
  jeans_money = 57 →
  (jeans_money : ℚ) / (total - jeans_money : ℚ) = 3 / 1 :=
by sorry

end jeans_to_janes_money_ratio_l3472_347268


namespace incoming_students_l3472_347201

theorem incoming_students (n : ℕ) : n < 600 ∧ n % 26 = 25 ∧ n % 24 = 15 → n = 519 :=
by sorry

end incoming_students_l3472_347201


namespace fraction_equality_l3472_347211

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a / b = 2 / 3) 
  (h2 : c / b = 1 / 5) 
  (h3 : c / d = 7 / 15) : 
  a * b / (c * d) = 140 / 9 := by
  sorry

end fraction_equality_l3472_347211


namespace first_nonzero_digit_of_one_over_139_l3472_347288

/-- The first nonzero digit to the right of the decimal point in the decimal representation of 1/139 is 1. -/
theorem first_nonzero_digit_of_one_over_139 : ∃ (n : ℕ) (d : ℕ), 
  (1 : ℚ) / 139 = (n : ℚ) / 10^(d + 1) + (1 : ℚ) / (10 * 10^(d + 1)) + (r : ℚ) / (100 * 10^(d + 1)) ∧ 
  0 ≤ r ∧ r < 10 := by
  sorry

end first_nonzero_digit_of_one_over_139_l3472_347288


namespace multiples_count_multiples_of_4_or_5_not_20_l3472_347255

theorem multiples_count : Nat → Nat :=
  fun n => (n / 4 + n / 5 - n / 20)

theorem multiples_of_4_or_5_not_20 (upper_bound : Nat) 
  (h : upper_bound = 1500) : 
  multiples_count upper_bound = 600 := by
  sorry

end multiples_count_multiples_of_4_or_5_not_20_l3472_347255


namespace trig_identity_l3472_347294

theorem trig_identity (α : Real) (h : α ∈ Set.Ioo (-π) (-π/2)) : 
  Real.sqrt ((1 + Real.cos α) / (1 - Real.cos α)) - 
  Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = 
  2 / Real.tan α := by sorry

end trig_identity_l3472_347294


namespace smallest_solution_congruence_l3472_347250

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 29 = 17 % 29 ∧ 
  ∀ (y : ℕ), y > 0 → (5 * y) % 29 = 17 % 29 → x ≤ y :=
by sorry

end smallest_solution_congruence_l3472_347250


namespace wife_selection_probability_l3472_347292

theorem wife_selection_probability 
  (p_husband : ℝ) 
  (p_only_one : ℝ) 
  (h1 : p_husband = 1/7)
  (h2 : p_only_one = 0.28571428571428575) : 
  ∃ p_wife : ℝ, p_wife = 1/5 ∧ 
  p_only_one = p_husband * (1 - p_wife) + p_wife * (1 - p_husband) :=
sorry

end wife_selection_probability_l3472_347292


namespace profit_percentage_calculation_l3472_347265

/-- Calculate the profit percentage given the selling price and cost price -/
theorem profit_percentage_calculation (selling_price cost_price : ℚ) :
  selling_price = 800 ∧ cost_price = 640 →
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end profit_percentage_calculation_l3472_347265


namespace range_of_m_l3472_347270

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) : (B m ∩ A = B m) ↔ m ≤ 3 :=
  sorry

end range_of_m_l3472_347270


namespace arithmetic_geometric_sequence_l3472_347269

theorem arithmetic_geometric_sequence (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- distinct real numbers
  2 * b = a + c →  -- arithmetic sequence
  (a * b) ^ 2 = a * c * b * c →  -- geometric sequence
  a + b + c = 6 →  -- sum condition
  a = 4 := by sorry

end arithmetic_geometric_sequence_l3472_347269


namespace karls_trip_distance_l3472_347213

-- Define the problem parameters
def miles_per_gallon : ℚ := 30
def tank_capacity : ℚ := 16
def initial_distance : ℚ := 420
def gas_bought : ℚ := 10
def final_tank_fraction : ℚ := 3/4

-- Theorem statement
theorem karls_trip_distance :
  let initial_gas_used : ℚ := initial_distance / miles_per_gallon
  let remaining_gas : ℚ := tank_capacity - initial_gas_used
  let gas_after_refill : ℚ := remaining_gas + gas_bought
  let final_gas : ℚ := tank_capacity * final_tank_fraction
  gas_after_refill = final_gas →
  initial_distance = 420 := by
sorry

end karls_trip_distance_l3472_347213


namespace butterfly_failure_rate_l3472_347238

theorem butterfly_failure_rate 
  (total_caterpillars : ℕ) 
  (butterfly_price : ℚ) 
  (total_revenue : ℚ) : 
  total_caterpillars = 40 →
  butterfly_price = 3 →
  total_revenue = 72 →
  (total_caterpillars - (total_revenue / butterfly_price)) / total_caterpillars * 100 = 40 := by
  sorry

end butterfly_failure_rate_l3472_347238


namespace cubic_odd_and_increasing_l3472_347202

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem cubic_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y) :=
by sorry

end cubic_odd_and_increasing_l3472_347202


namespace negation_of_implication_l3472_347206

theorem negation_of_implication (a b c : ℝ) :
  ¬(a + b + c = 1 → a^2 + b^2 + c^2 ≤ 1/9) ↔ (a + b + c ≠ 1 → a^2 + b^2 + c^2 > 1/9) := by
sorry

end negation_of_implication_l3472_347206


namespace S_infinite_l3472_347275

/-- The set of positive integers n for which the number of positive divisors of 2^n - 1 is greater than n -/
def S : Set Nat :=
  {n : Nat | n > 0 ∧ (Nat.divisors (2^n - 1)).card > n}

/-- Theorem stating that the set S is infinite -/
theorem S_infinite : Set.Infinite S := by
  sorry

end S_infinite_l3472_347275


namespace opposite_of_negative_three_halves_l3472_347200

theorem opposite_of_negative_three_halves :
  -((-3 : ℚ) / 2) = 3 / 2 := by
  sorry

end opposite_of_negative_three_halves_l3472_347200


namespace inverse_function_root_uniqueness_l3472_347221

theorem inverse_function_root_uniqueness 
  (f : ℝ → ℝ) (h_inv : Function.Injective f) :
  ∀ m : ℝ, ∃! x : ℝ, f x = m ∨ ∀ y : ℝ, f y ≠ m :=
by sorry

end inverse_function_root_uniqueness_l3472_347221


namespace number_problem_l3472_347205

theorem number_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 20 → (40/100 : ℝ) * N = 240 := by
  sorry

end number_problem_l3472_347205


namespace min_value_expression_l3472_347237

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_constraint : a + b + c = 13) : 
  (a^2 + b^3 + c^4 + 2019) / (10*b + 123*c + 26) ≥ 4 := by
  sorry

end min_value_expression_l3472_347237


namespace square_configuration_angle_l3472_347267

/-- Theorem: In a configuration of three squares attached by their vertices to each other and to two vertical rods,
    where the sum of the white angles equals the sum of the gray angles, and given angles of 30°, 126°, 75°,
    and three 90° angles, the measure of the remaining angle x is 39°. -/
theorem square_configuration_angle (white_angles gray_angles : List ℝ)
  (h1 : white_angles.sum = gray_angles.sum)
  (h2 : white_angles.length = 4)
  (h3 : gray_angles.length = 3)
  (h4 : white_angles.take 3 = [30, 126, 75])
  (h5 : gray_angles = [90, 90, 90]) :
  white_angles[3] = 39 := by
  sorry

end square_configuration_angle_l3472_347267


namespace max_value_cos_sin_l3472_347217

theorem max_value_cos_sin (x : ℝ) : 2 * Real.cos x + 3 * Real.sin x ≤ Real.sqrt 13 := by
  sorry

end max_value_cos_sin_l3472_347217


namespace days_without_visits_l3472_347228

def days_in_year : ℕ := 366

def visit_period_1 : ℕ := 6
def visit_period_2 : ℕ := 8
def visit_period_3 : ℕ := 10

def days_with_visits (period : ℕ) : ℕ := days_in_year / period

def lcm_two (a b : ℕ) : ℕ := Nat.lcm a b
def lcm_three (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

def days_with_two_visits (period1 period2 : ℕ) : ℕ := days_in_year / (lcm_two period1 period2)

def days_with_three_visits (period1 period2 period3 : ℕ) : ℕ := days_in_year / (lcm_three period1 period2 period3)

theorem days_without_visits :
  days_in_year - 
  ((days_with_visits visit_period_1 + days_with_visits visit_period_2 + days_with_visits visit_period_3) -
   (days_with_two_visits visit_period_1 visit_period_2 + 
    days_with_two_visits visit_period_1 visit_period_3 + 
    days_with_two_visits visit_period_2 visit_period_3) +
   days_with_three_visits visit_period_1 visit_period_2 visit_period_3) = 257 :=
by sorry

end days_without_visits_l3472_347228


namespace remainder_of_power_minus_ninety_l3472_347298

theorem remainder_of_power_minus_ninety (n : ℕ) : (1 - 90) ^ 10 ≡ 1 [ZMOD 88] := by
  sorry

end remainder_of_power_minus_ninety_l3472_347298


namespace library_books_not_all_checked_out_l3472_347243

-- Define a type for books
variable (Book : Type)

-- Define a predicate for a book being in the library
variable (in_library : Book → Prop)

-- Define a predicate for a book being checked out
variable (checked_out : Book → Prop)

-- Theorem statement
theorem library_books_not_all_checked_out 
  (h : ¬∀ b : Book, in_library b → checked_out b) :
  (∃ b : Book, in_library b ∧ ¬checked_out b) ∧
  (¬∀ b : Book, in_library b → checked_out b) := by
  sorry

end library_books_not_all_checked_out_l3472_347243


namespace johnny_earnings_l3472_347225

/-- Represents Johnny's daily work schedule and earnings --/
structure DailyWork where
  hours1 : ℕ
  rate1 : ℕ
  hours2 : ℕ
  rate2 : ℕ
  hours3 : ℕ
  rate3 : ℕ

/-- Calculates the total earnings for a given number of days --/
def totalEarnings (work : DailyWork) (days : ℕ) : ℕ :=
  days * (work.hours1 * work.rate1 + work.hours2 * work.rate2 + work.hours3 * work.rate3)

/-- Johnny's work schedule --/
def johnnysWork : DailyWork :=
  { hours1 := 3
  , rate1 := 7
  , hours2 := 2
  , rate2 := 10
  , hours3 := 4
  , rate3 := 12 }

theorem johnny_earnings :
  totalEarnings johnnysWork 5 = 445 := by
  sorry

end johnny_earnings_l3472_347225


namespace meaningful_expression_range_l3472_347259

theorem meaningful_expression_range (x : ℝ) :
  (∃ y : ℝ, y = Real.sqrt (1 - x) ∧ x + 2 ≠ 0) ↔ x ≤ 1 ∧ x ≠ -2 := by sorry

end meaningful_expression_range_l3472_347259


namespace largest_ball_on_torus_l3472_347296

/-- The radius of the largest spherical ball that can be placed on top of a torus -/
def largest_ball_radius (inner_radius outer_radius : ℝ) : ℝ :=
  outer_radius - inner_radius

/-- The torus is formed by revolving a circle with radius 1 centered at (4,0,1) -/
def torus_center_radius : ℝ := 4

/-- The height of the torus center above the table -/
def torus_center_height : ℝ := 1

/-- Theorem: The radius of the largest spherical ball on a torus with inner radius 3 and outer radius 5 is 4 -/
theorem largest_ball_on_torus :
  largest_ball_radius 3 5 = 4 :=
by sorry

end largest_ball_on_torus_l3472_347296


namespace no_simultaneous_integer_roots_l3472_347208

theorem no_simultaneous_integer_roots :
  ¬ ∃ (b c : ℝ),
    (∃ (k l m n : ℤ),
      (k ≠ l ∧ m ≠ n) ∧
      (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = k ∨ x = l) ∧
      (∀ x : ℝ, 2*x^2 + (b+1)*x + (c+1) = 0 ↔ x = m ∨ x = n)) :=
by sorry

end no_simultaneous_integer_roots_l3472_347208


namespace triangular_number_200_l3472_347210

/-- Triangular number sequence -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 200th triangular number is 20100 -/
theorem triangular_number_200 : triangular_number 200 = 20100 := by
  sorry

end triangular_number_200_l3472_347210


namespace gcd_15_2015_l3472_347276

theorem gcd_15_2015 : Nat.gcd 15 2015 = 5 := by
  sorry

end gcd_15_2015_l3472_347276


namespace sufficient_condition_for_inequality_l3472_347234

theorem sufficient_condition_for_inequality (a : ℝ) :
  (a < 1) → (∀ x : ℝ, a ≤ |x| + |x - 1|) ∧
  ¬(∀ a : ℝ, (∀ x : ℝ, a ≤ |x| + |x - 1|) → (a < 1)) :=
by sorry

end sufficient_condition_for_inequality_l3472_347234


namespace seventeen_students_earlier_l3472_347281

/-- The number of students who came earlier than Hoseok, given the total number of students and the number of students who came later. -/
def students_earlier (total : ℕ) (later : ℕ) : ℕ :=
  total - later - 1

/-- Theorem stating that 17 students came earlier than Hoseok. -/
theorem seventeen_students_earlier :
  students_earlier 30 12 = 17 := by
  sorry

end seventeen_students_earlier_l3472_347281


namespace ancient_chinese_gcd_is_successive_differences_l3472_347285

/-- The algorithm used by ancient Chinese mathematicians to find the GCD of two positive integers -/
def ancient_chinese_gcd_algorithm : Type := sorry

/-- The method of successive differences -/
def successive_differences : Type := sorry

/-- Assertion that the ancient Chinese GCD algorithm is the method of successive differences -/
theorem ancient_chinese_gcd_is_successive_differences : 
  ancient_chinese_gcd_algorithm = successive_differences := by sorry

end ancient_chinese_gcd_is_successive_differences_l3472_347285


namespace sheet_width_l3472_347241

theorem sheet_width (w : ℝ) 
  (h1 : w > 0)
  (h2 : (w - 4) * 24 / (w * 30) = 64 / 100) : 
  w = 20 := by sorry

end sheet_width_l3472_347241


namespace tiffany_monday_bags_l3472_347214

/-- The number of bags Tiffany had on Monday -/
def monday_bags : ℕ := sorry

/-- The number of bags Tiffany found on Tuesday -/
def tuesday_bags : ℕ := 3

/-- The number of bags Tiffany found on Wednesday -/
def wednesday_bags : ℕ := 7

/-- The total number of bags Tiffany had -/
def total_bags : ℕ := 20

/-- Theorem stating that Tiffany had 10 bags on Monday -/
theorem tiffany_monday_bags : 
  monday_bags + tuesday_bags + wednesday_bags = total_bags ∧ monday_bags = 10 := by
  sorry

end tiffany_monday_bags_l3472_347214


namespace cubic_root_sum_l3472_347204

theorem cubic_root_sum (r s t : ℝ) : 
  r^3 - 15*r^2 + 13*r - 8 = 0 →
  s^3 - 15*s^2 + 13*s - 8 = 0 →
  t^3 - 15*t^2 + 13*t - 8 = 0 →
  r / (1/r + s*t) + s / (1/s + r*t) + t / (1/t + r*s) = 199/9 := by
sorry

end cubic_root_sum_l3472_347204


namespace inequality_proof_l3472_347249

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  (1 - x^2)⁻¹ + (1 - y^2)⁻¹ ≥ 2 * (1 - x*y)⁻¹ := by
  sorry

end inequality_proof_l3472_347249


namespace cabbage_distribution_l3472_347246

/-- Given a cabbage patch with 12 rows and 180 total heads of cabbage,
    prove that there are 15 heads of cabbage in each row. -/
theorem cabbage_distribution (rows : ℕ) (total_heads : ℕ) (heads_per_row : ℕ) : 
  rows = 12 → total_heads = 180 → heads_per_row * rows = total_heads → heads_per_row = 15 := by
  sorry

end cabbage_distribution_l3472_347246


namespace sqrt_sum_equals_eight_l3472_347253

theorem sqrt_sum_equals_eight : 
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 8 := by
  sorry

end sqrt_sum_equals_eight_l3472_347253
