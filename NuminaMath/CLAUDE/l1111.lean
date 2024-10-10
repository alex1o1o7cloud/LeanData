import Mathlib

namespace equation_represents_parabola_l1111_111148

/-- The equation represents a parabola if it can be transformed into the form y = ax² + bx + c, where a ≠ 0 -/
def is_parabola (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation |y - 3| = √((x+1)² + y²) -/
def given_equation (x y : ℝ) : Prop :=
  |y - 3| = Real.sqrt ((x + 1)^2 + y^2)

theorem equation_represents_parabola :
  ∃ f : ℝ → ℝ, (∀ x y, given_equation x y ↔ y = f x) ∧ is_parabola f :=
sorry

end equation_represents_parabola_l1111_111148


namespace square_perimeter_l1111_111197

theorem square_perimeter (s : ℝ) (h : s > 0) :
  (5 * s / 2 = 40) → (4 * s = 64) := by
  sorry

end square_perimeter_l1111_111197


namespace range_not_real_l1111_111166

/-- Given real numbers a and b satisfying ab = a + b + 3, 
    the range of (a-1)b is not equal to R. -/
theorem range_not_real : ¬ (∀ (y : ℝ), ∃ (a b : ℝ), a * b = a + b + 3 ∧ (a - 1) * b = y) := by
  sorry

end range_not_real_l1111_111166


namespace hyperbola_center_l1111_111129

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) : 
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  f1 = (2, 3) → f2 = (-4, 7) → center = (-1, 5) := by
sorry

end hyperbola_center_l1111_111129


namespace quadratic_solution_difference_l1111_111161

theorem quadratic_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (x₁^2 - 5*x₁ + 11 = x₁ + 53) ∧ 
  (x₂^2 - 5*x₂ + 11 = x₂ + 53) ∧ 
  (x₁ ≠ x₂) ∧
  (|x₁ - x₂| = 2 * Real.sqrt 51) := by
  sorry

end quadratic_solution_difference_l1111_111161


namespace rain_probabilities_l1111_111169

/-- The probability of rain in place A -/
def prob_A : ℝ := 0.2

/-- The probability of rain in place B -/
def prob_B : ℝ := 0.3

/-- The probability of no rain in both places A and B -/
def prob_neither : ℝ := (1 - prob_A) * (1 - prob_B)

/-- The probability of rain in exactly one of places A or B -/
def prob_exactly_one : ℝ := prob_A * (1 - prob_B) + (1 - prob_A) * prob_B

/-- The probability of rain in at least one of places A or B -/
def prob_at_least_one : ℝ := 1 - prob_neither

/-- The probability of rain in at most one of places A or B -/
def prob_at_most_one : ℝ := prob_neither + prob_exactly_one

theorem rain_probabilities :
  prob_neither = 0.56 ∧
  prob_exactly_one = 0.38 ∧
  prob_at_least_one = 0.44 ∧
  prob_at_most_one = 0.94 := by
  sorry

end rain_probabilities_l1111_111169


namespace emma_room_coverage_l1111_111187

/-- Represents the dimensions of Emma's room --/
structure RoomDimensions where
  rectangleLength : ℝ
  rectangleWidth : ℝ
  triangleBase : ℝ
  triangleHeight : ℝ

/-- Represents the tiles used to cover the room --/
structure Tiles where
  squareTiles : ℕ
  triangularTiles : ℕ
  squareTileArea : ℝ
  triangularTileBase : ℝ
  triangularTileHeight : ℝ

/-- Calculates the fraction of the room covered by tiles --/
def fractionalCoverage (room : RoomDimensions) (tiles : Tiles) : ℚ :=
  sorry

/-- Theorem stating that the fractional coverage of Emma's room is 3/20 --/
theorem emma_room_coverage :
  let room : RoomDimensions := {
    rectangleLength := 12,
    rectangleWidth := 20,
    triangleBase := 10,
    triangleHeight := 8
  }
  let tiles : Tiles := {
    squareTiles := 40,
    triangularTiles := 4,
    squareTileArea := 1,
    triangularTileBase := 1,
    triangularTileHeight := 1
  }
  fractionalCoverage room tiles = 3/20 := by
  sorry

end emma_room_coverage_l1111_111187


namespace quadratic_equation_m_value_l1111_111192

/-- Given that (m+1)x^(m^2+1) - 2x - 5 = 0 is a quadratic equation in x 
    and m + 1 ≠ 0, prove that m = 1 -/
theorem quadratic_equation_m_value (m : ℝ) : 
  (∃ a b c : ℝ, ∀ x : ℝ, (m + 1) * x^(m^2 + 1) - 2*x - 5 = a*x^2 + b*x + c) ∧ 
  (m + 1 ≠ 0) → 
  m = 1 :=
sorry

end quadratic_equation_m_value_l1111_111192


namespace range_of_a_l1111_111198

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → a > 2 * x - 1) → a ≥ 3 := by
  sorry

end range_of_a_l1111_111198


namespace ferris_wheel_capacity_l1111_111170

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := 4

/-- The total number of people that can ride the wheel at the same time -/
def total_people : ℕ := 20

/-- The number of people each seat can hold -/
def people_per_seat : ℕ := total_people / num_seats

theorem ferris_wheel_capacity : people_per_seat = 5 := by
  sorry

end ferris_wheel_capacity_l1111_111170


namespace wall_construction_l1111_111185

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 1800

/-- Rate of the first bricklayer in bricks per hour -/
def rate1 : ℚ := total_bricks / 12

/-- Rate of the second bricklayer in bricks per hour -/
def rate2 : ℚ := total_bricks / 15

/-- Combined rate reduction when working together -/
def rate_reduction : ℕ := 15

/-- Time taken to complete the wall together -/
def time_taken : ℕ := 6

theorem wall_construction :
  (time_taken : ℚ) * (rate1 + rate2 - rate_reduction) = total_bricks := by sorry

end wall_construction_l1111_111185


namespace geometric_sequence_middle_term_l1111_111172

/-- Given a geometric sequence with first term 1, last term 9, and middle terms a, b, c, prove that b = 3 -/
theorem geometric_sequence_middle_term 
  (a b c : ℝ) 
  (h_sequence : ∃ (r : ℝ), r > 0 ∧ a = 1 * r ∧ b = a * r ∧ c = b * r ∧ 9 = c * r) : 
  b = 3 := by
sorry

end geometric_sequence_middle_term_l1111_111172


namespace smaller_root_of_equation_l1111_111143

theorem smaller_root_of_equation : 
  let f (x : ℝ) := (x - 1/3)^2 + (x - 1/3)*(x - 2/3)
  ∃ y, f y = 0 ∧ y ≤ 1/3 ∧ ∀ z, f z = 0 → z ≥ 1/3 := by
  sorry

end smaller_root_of_equation_l1111_111143


namespace gcd_factorial_problem_l1111_111120

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 9) / (Nat.factorial 4)) = 2520 := by
  sorry

end gcd_factorial_problem_l1111_111120


namespace function_equality_l1111_111182

theorem function_equality (a b : ℝ) : 
  (∀ x, (x^2 + 4*x + 3 = (a*x + b)^2 + 4*(a*x + b) + 3)) → 
  ((a + b = -8) ∨ (a + b = 4)) := by
sorry

end function_equality_l1111_111182


namespace solution_set_inequality_l1111_111116

theorem solution_set_inequality (x : ℝ) : 
  (x * (x - 1) < 0) ↔ (0 < x ∧ x < 1) := by sorry

end solution_set_inequality_l1111_111116


namespace carnation_fraction_l1111_111102

def flower_bouquet (total : ℝ) (pink_roses red_roses pink_carnations red_carnations : ℝ) : Prop :=
  pink_roses + red_roses + pink_carnations + red_carnations = total ∧
  pink_roses + pink_carnations = (7/10) * total ∧
  pink_roses = (1/2) * (pink_roses + pink_carnations) ∧
  red_carnations = (5/6) * (red_roses + red_carnations)

theorem carnation_fraction (total : ℝ) (pink_roses red_roses pink_carnations red_carnations : ℝ) 
  (h : flower_bouquet total pink_roses red_roses pink_carnations red_carnations) :
  (pink_carnations + red_carnations) / total = 3/5 :=
sorry

end carnation_fraction_l1111_111102


namespace sarah_bus_time_l1111_111112

-- Define the problem parameters
def leave_time : Nat := 7 * 60 + 45  -- 7:45 AM in minutes
def return_time : Nat := 17 * 60 + 15  -- 5:15 PM in minutes
def num_classes : Nat := 8
def class_duration : Nat := 45
def lunch_break : Nat := 30
def extracurricular_time : Nat := 90  -- 1 hour and 30 minutes in minutes

-- Define the theorem
theorem sarah_bus_time :
  let total_time := return_time - leave_time
  let school_time := num_classes * class_duration + lunch_break + extracurricular_time
  total_time - school_time = 90 := by
  sorry

end sarah_bus_time_l1111_111112


namespace thirtieth_term_is_351_l1111_111138

/-- Arithmetic sequence with first term 3 and common difference 12 -/
def arithmeticSequence (n : ℕ) : ℤ :=
  3 + (n - 1) * 12

/-- The 30th term of the arithmetic sequence is 351 -/
theorem thirtieth_term_is_351 : arithmeticSequence 30 = 351 := by
  sorry

end thirtieth_term_is_351_l1111_111138


namespace arflaser_wavelength_scientific_notation_l1111_111137

theorem arflaser_wavelength_scientific_notation :
  ∀ (wavelength : ℝ),
  wavelength = 0.000000193 →
  ∃ (a : ℝ) (n : ℤ),
    wavelength = a * (10 : ℝ) ^ n ∧
    1 ≤ a ∧ a < 10 ∧
    a = 1.93 ∧ n = -7 :=
by sorry

end arflaser_wavelength_scientific_notation_l1111_111137


namespace range_of_m_for_decreasing_function_l1111_111158

-- Define a decreasing function
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem range_of_m_for_decreasing_function (f : ℝ → ℝ) (m : ℝ) 
  (h_decreasing : DecreasingFunction f) (h_inequality : f (m - 1) > f (2 * m - 1)) :
  m > 0 := by sorry

end range_of_m_for_decreasing_function_l1111_111158


namespace imaginary_unit_power_2016_l1111_111175

theorem imaginary_unit_power_2016 (i : ℂ) (h : i^2 = -1) : i^2016 = 1 := by
  sorry

end imaginary_unit_power_2016_l1111_111175


namespace total_soldiers_on_great_wall_l1111_111176

-- Define the parameters
def wall_length : ℕ := 7300
def tower_interval : ℕ := 5
def soldiers_per_tower : ℕ := 2

-- Theorem statement
theorem total_soldiers_on_great_wall :
  (wall_length / tower_interval) * soldiers_per_tower = 2920 :=
by sorry

end total_soldiers_on_great_wall_l1111_111176


namespace day_after_2005_squared_days_l1111_111101

theorem day_after_2005_squared_days (start_day : ℕ) : 
  start_day % 7 = 0 → (start_day + 2005^2) % 7 = 6 := by
  sorry

end day_after_2005_squared_days_l1111_111101


namespace vasya_number_digits_l1111_111134

theorem vasya_number_digits (x : ℝ) (h_pos : x > 0) 
  (h_kolya : 10^8 ≤ x^3 ∧ x^3 < 10^9) (h_petya : 10^10 ≤ x^4 ∧ x^4 < 10^11) :
  10^32 ≤ x^12 ∧ x^12 < 10^33 := by
  sorry

end vasya_number_digits_l1111_111134


namespace max_value_problem_min_value_problem_l1111_111180

-- Problem 1
theorem max_value_problem (x : ℝ) (h : 0 < x ∧ x < 2) : x * (4 - 2*x) ≤ 2 := by
  sorry

-- Problem 2
theorem min_value_problem (x : ℝ) (h : x > 3/2) : x + 8 / (2*x - 3) ≥ 11/2 := by
  sorry

end max_value_problem_min_value_problem_l1111_111180


namespace solution_set_f_solution_set_g_l1111_111146

-- Define the quadratic functions
def f (x : ℝ) := x^2 - 3*x - 4
def g (x : ℝ) := x^2 - x - 6

-- Define the solution sets
def S₁ : Set ℝ := {x | -1 < x ∧ x < 4}
def S₂ : Set ℝ := {x | x < -2 ∨ x > 3}

-- Theorem statements
theorem solution_set_f : {x : ℝ | f x < 0} = S₁ := by sorry

theorem solution_set_g : {x : ℝ | g x > 0} = S₂ := by sorry

end solution_set_f_solution_set_g_l1111_111146


namespace inequality_proof_l1111_111171

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  1 / (x + y) + 4 / (y + z) + 9 / (x + z) ≥ 18 / (x + y + z) := by
  sorry

end inequality_proof_l1111_111171


namespace man_speed_against_current_l1111_111124

/-- A river with three sections and a man traveling along it -/
structure River :=
  (current_speed1 : ℝ)
  (current_speed2 : ℝ)
  (current_speed3 : ℝ)
  (man_speed_with_current1 : ℝ)

/-- Calculate the man's speed against the current in each section -/
def speed_against_current (r : River) : ℝ × ℝ × ℝ :=
  let speed_still_water := r.man_speed_with_current1 - r.current_speed1
  (speed_still_water - r.current_speed1,
   speed_still_water - r.current_speed2,
   speed_still_water - r.current_speed3)

/-- Theorem stating the man's speed against the current in each section -/
theorem man_speed_against_current (r : River) 
  (h1 : r.current_speed1 = 1.5)
  (h2 : r.current_speed2 = 2.5)
  (h3 : r.current_speed3 = 3.5)
  (h4 : r.man_speed_with_current1 = 25) :
  speed_against_current r = (22, 21, 20) :=
sorry


end man_speed_against_current_l1111_111124


namespace exists_arithmetic_progression_of_5_primes_exists_arithmetic_progression_of_6_primes_l1111_111190

/-- An arithmetic progression of primes -/
def ArithmeticProgressionOfPrimes (n : ℕ) : Prop :=
  ∃ (a d : ℕ), ∀ k : Fin n, Prime (a + k * d)

/-- There exists an arithmetic progression of 5 primes -/
theorem exists_arithmetic_progression_of_5_primes :
  ArithmeticProgressionOfPrimes 5 := by
  sorry

/-- There exists an arithmetic progression of 6 primes -/
theorem exists_arithmetic_progression_of_6_primes :
  ArithmeticProgressionOfPrimes 6 := by
  sorry

end exists_arithmetic_progression_of_5_primes_exists_arithmetic_progression_of_6_primes_l1111_111190


namespace contrapositive_odd_sum_even_l1111_111144

theorem contrapositive_odd_sum_even :
  (¬(∃ (a b : ℤ), Odd a ∧ Odd b ∧ ¬(Even (a + b))) ↔
   (∀ (a b : ℤ), ¬(Even (a + b)) → ¬(Odd a ∧ Odd b))) :=
by sorry

end contrapositive_odd_sum_even_l1111_111144


namespace intersection_of_A_and_B_l1111_111178

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x < 3}
def B : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | -1 < x ∧ x < 3} := by
  sorry

end intersection_of_A_and_B_l1111_111178


namespace max_distance_sum_l1111_111127

/-- Given m ∈ R, for points A on the line x + my = 0 and B on the line mx - y - m + 3 = 0,
    where these lines intersect at point P, the maximum value of |PA| + |PB| is 2√5. -/
theorem max_distance_sum (m : ℝ) : 
  ∃ (A B P : ℝ × ℝ), 
    (A.1 + m * A.2 = 0) ∧ 
    (m * B.1 - B.2 - m + 3 = 0) ∧ 
    (P.1 + m * P.2 = 0) ∧ 
    (m * P.1 - P.2 - m + 3 = 0) ∧
    (∀ (A' B' : ℝ × ℝ), 
      (A'.1 + m * A'.2 = 0) → 
      (m * B'.1 - B'.2 - m + 3 = 0) → 
      Real.sqrt ((P.1 - A'.1)^2 + (P.2 - A'.2)^2) + Real.sqrt ((P.1 - B'.1)^2 + (P.2 - B'.2)^2) ≤ 2 * Real.sqrt 5) ∧
    (Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 2 * Real.sqrt 5) :=
by sorry

end max_distance_sum_l1111_111127


namespace total_sheets_required_l1111_111123

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0 to 9) -/
def digit_count : ℕ := 10

/-- The number of sheets required for one character -/
def sheets_per_char : ℕ := 1

/-- Theorem: The total number of sheets required to write all uppercase and lowercase 
    English alphabets and digits from 0 to 9 is 62 -/
theorem total_sheets_required : 
  sheets_per_char * (2 * alphabet_size + digit_count) = 62 := by
  sorry

end total_sheets_required_l1111_111123


namespace quadratic_inequality_l1111_111188

-- Define the quadratic function
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_inequality (b c : ℝ) (h : f b c 0 = f b c 2) :
  f b c (3/2) < f b c 0 ∧ f b c 0 < f b c (-2) := by
  sorry

end quadratic_inequality_l1111_111188


namespace least_subtraction_for_divisibility_by_10_l1111_111117

theorem least_subtraction_for_divisibility_by_10 :
  ∃ (n : ℕ), n = 2 ∧ 
  (427398 - n) % 10 = 0 ∧
  ∀ (m : ℕ), m < n → (427398 - m) % 10 ≠ 0 :=
by sorry

end least_subtraction_for_divisibility_by_10_l1111_111117


namespace special_polyhedron_ratio_l1111_111189

/-- A polyhedron with specific properties -/
structure SpecialPolyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ
  x : ℝ
  y : ℝ
  all_faces_isosceles : Prop
  edge_lengths : Prop
  vertex_degrees : Prop
  equal_dihedral_angles : Prop

/-- The theorem statement -/
theorem special_polyhedron_ratio 
  (P : SpecialPolyhedron)
  (h_faces : P.faces = 12)
  (h_edges : P.edges = 18)
  (h_vertices : P.vertices = 8)
  (h_isosceles : P.all_faces_isosceles)
  (h_edge_lengths : P.edge_lengths)
  (h_vertex_degrees : P.vertex_degrees)
  (h_dihedral_angles : P.equal_dihedral_angles)
  : P.x / P.y = 3 / 5 :=
sorry

end special_polyhedron_ratio_l1111_111189


namespace symmetry_of_shifted_even_function_l1111_111154

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define the axis of symmetry for a function
def axis_of_symmetry (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- State the theorem
theorem symmetry_of_shifted_even_function :
  is_even (λ x => f (x + 1)) → axis_of_symmetry f 1 := by
  sorry

end symmetry_of_shifted_even_function_l1111_111154


namespace final_sum_after_operations_l1111_111111

theorem final_sum_after_operations (x y S : ℝ) (h : x + y = S) :
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 := by sorry

end final_sum_after_operations_l1111_111111


namespace white_balls_count_l1111_111126

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 ∧
  green = 30 ∧
  yellow = 10 ∧
  red = 47 ∧
  purple = 3 ∧
  prob_not_red_purple = 1/2 →
  total - (green + yellow + red + purple) = 10 := by
sorry

end white_balls_count_l1111_111126


namespace odd_prime_sum_of_squares_l1111_111196

theorem odd_prime_sum_of_squares (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ (x y m : ℤ), 1 + x^2 + y^2 = m * p ∧ 0 < m ∧ m < p := by
  sorry

end odd_prime_sum_of_squares_l1111_111196


namespace count_valid_numbers_eq_441_l1111_111128

/-- The count of valid digits for hundreds place (1-4, 7-9) -/
def valid_hundreds : Nat := 7

/-- The count of valid digits for tens place (0-4, 7-9) -/
def valid_tens : Nat := 7

/-- The count of valid digits for units place (1-9) -/
def valid_units : Nat := 9

/-- The count of three-digit whole numbers with no 5's and 6's in the tens and hundreds places -/
def count_valid_numbers : Nat := valid_hundreds * valid_tens * valid_units

theorem count_valid_numbers_eq_441 : count_valid_numbers = 441 := by
  sorry

end count_valid_numbers_eq_441_l1111_111128


namespace max_n_is_eleven_l1111_111165

/-- A coloring of integers from 1 to 14 with two colors -/
def Coloring := Fin 14 → Bool

/-- Check if there exist pairs of numbers with the same color and given difference -/
def hasPairsWithDifference (c : Coloring) (k : Nat) (color : Bool) : Prop :=
  ∃ i j, i < j ∧ j ≤ 14 ∧ j - i = k ∧ c i = color ∧ c j = color

/-- The property that a coloring satisfies the conditions for a given n -/
def validColoring (c : Coloring) (n : Nat) : Prop :=
  ∀ k, k ≤ n → hasPairsWithDifference c k true ∧ hasPairsWithDifference c k false

/-- The main theorem: the maximum possible n is 11 -/
theorem max_n_is_eleven :
  (∃ c : Coloring, validColoring c 11) ∧
  (∀ c : Coloring, ¬validColoring c 12) :=
sorry

end max_n_is_eleven_l1111_111165


namespace percentage_less_than_500000_l1111_111121

-- Define the population categories
structure PopulationCategory where
  name : String
  percentage : ℝ

-- Define the theorem
theorem percentage_less_than_500000 (categories : List PopulationCategory)
  (h1 : categories.length = 3)
  (h2 : ∃ c ∈ categories, c.name = "less than 200,000" ∧ c.percentage = 35)
  (h3 : ∃ c ∈ categories, c.name = "200,000 to 499,999" ∧ c.percentage = 40)
  (h4 : ∃ c ∈ categories, c.name = "500,000 or more" ∧ c.percentage = 25)
  : (categories.filter (λ c => c.name = "less than 200,000" ∨ c.name = "200,000 to 499,999")).foldl (λ acc c => acc + c.percentage) 0 = 75 := by
  sorry

end percentage_less_than_500000_l1111_111121


namespace even_function_extension_l1111_111110

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem even_function_extension
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_neg : ∀ x < 0, f x = x - x^4) :
  ∀ x > 0, f x = -x^4 - x :=
by sorry

end even_function_extension_l1111_111110


namespace condition_2_condition_4_condition_1_not_sufficient_condition_3_not_sufficient_l1111_111167

-- Define the types for planes and lines
variable {Point : Type*}
variable {Line : Type*}
variable {Plane : Type*}

-- Define the necessary relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the planes α and β
variable (α β : Plane)

-- Theorem for condition ②
theorem condition_2 
  (h : ∀ l : Line, contains α l → line_parallel_plane l β) :
  parallel α β :=
sorry

-- Theorem for condition ④
theorem condition_4 
  (a b : Line)
  (h1 : perpendicular a α)
  (h2 : perpendicular b β)
  (h3 : line_parallel a b) :
  parallel α β :=
sorry

-- Theorem for condition ①
theorem condition_1_not_sufficient 
  (h : ∃ S : Set Line, (∀ l ∈ S, contains α l ∧ line_parallel_plane l β) ∧ Set.Infinite S) :
  ¬(parallel α β → True) :=
sorry

-- Theorem for condition ③
theorem condition_3_not_sufficient 
  (a b : Line)
  (h1 : contains α a)
  (h2 : contains β b)
  (h3 : line_parallel_plane a β)
  (h4 : line_parallel_plane b α) :
  ¬(parallel α β → True) :=
sorry

end condition_2_condition_4_condition_1_not_sufficient_condition_3_not_sufficient_l1111_111167


namespace mean_equality_problem_l1111_111168

theorem mean_equality_problem (x : ℝ) : 
  (8 + 16 + 24) / 3 = (10 + x) / 2 → x = 22 := by
  sorry

end mean_equality_problem_l1111_111168


namespace fraction_zero_implies_x_is_one_l1111_111136

theorem fraction_zero_implies_x_is_one (x : ℝ) :
  (x - 1) / (x - 3) = 0 → x = 1 := by
  sorry

end fraction_zero_implies_x_is_one_l1111_111136


namespace equal_share_problem_l1111_111109

theorem equal_share_problem (total_amount : ℚ) (num_people : ℕ) :
  total_amount = 3.75 →
  num_people = 3 →
  total_amount / num_people = 1.25 := by
  sorry

end equal_share_problem_l1111_111109


namespace total_yardage_progress_l1111_111135

def team_a_moves : List Int := [-5, 8, -3, 6]
def team_b_moves : List Int := [4, -2, 9, -7]

theorem total_yardage_progress : 
  (team_a_moves.sum + team_b_moves.sum) = 10 := by sorry

end total_yardage_progress_l1111_111135


namespace line_perpendicular_to_plane_and_parallel_line_l1111_111186

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)

-- Define the axioms
variable (different_lines : ∀ a b l : Line, a ≠ b ∧ b ≠ l ∧ a ≠ l)
variable (non_coincident_planes : ∀ α β : Plane, α ≠ β)

-- State the theorem
theorem line_perpendicular_to_plane_and_parallel_line 
  (a b l : Line) (α : Plane) :
  parallel a b → perpendicular_line_plane l α → perpendicular_line_line l b :=
sorry

end line_perpendicular_to_plane_and_parallel_line_l1111_111186


namespace optimal_discount_sequence_l1111_111184

/-- The price of the coffee bag before discounts -/
def initial_price : ℝ := 18

/-- The fixed discount amount -/
def fixed_discount : ℝ := 3

/-- The percentage discount as a decimal -/
def percentage_discount : ℝ := 0.15

/-- The price after applying fixed discount then percentage discount -/
def price_fixed_then_percentage : ℝ := (initial_price - fixed_discount) * (1 - percentage_discount)

/-- The price after applying percentage discount then fixed discount -/
def price_percentage_then_fixed : ℝ := initial_price * (1 - percentage_discount) - fixed_discount

theorem optimal_discount_sequence :
  price_fixed_then_percentage - price_percentage_then_fixed = 0.45 := by
  sorry

end optimal_discount_sequence_l1111_111184


namespace min_value_sum_reciprocal_squares_l1111_111193

/-- Given two internally tangent circles C₁ and C₂ with equations x² + y² + 2ax + a² - 4 = 0 and 
    x² + y² - 2by + b² - 1 = 0 respectively, where a, b ∈ ℝ and ab ≠ 0, 
    the minimum value of 1/a² + 1/b² is 9 -/
theorem min_value_sum_reciprocal_squares (a b : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2*a*x + a^2 - 4 = 0 ∧ x^2 + y^2 - 2*b*y + b^2 - 1 = 0) →
  a ≠ 0 →
  b ≠ 0 →
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x + a^2 - 4 ≠ 0 ∨ x^2 + y^2 - 2*b*y + b^2 - 1 ≠ 0 ∨ 
    (x^2 + y^2 + 2*a*x + a^2 - 4 = 0 ∧ x^2 + y^2 - 2*b*y + b^2 - 1 = 0)) →
  (1 / a^2 + 1 / b^2) ≥ 9 :=
by sorry

#check min_value_sum_reciprocal_squares

end min_value_sum_reciprocal_squares_l1111_111193


namespace eight_digit_rotation_l1111_111162

def is_coprime (a b : Nat) : Prop := Nat.gcd a b = 1

def rotate_last_to_first (n : Nat) : Nat :=
  let d := n % 10
  let k := n / 10
  d * 10^7 + k

theorem eight_digit_rotation (A B : Nat) :
  (∃ B : Nat, 
    B > 44444444 ∧ 
    is_coprime B 12 ∧ 
    A = rotate_last_to_first B) →
  (A ≤ 99999998 ∧ A ≥ 14444446) :=
by sorry

end eight_digit_rotation_l1111_111162


namespace jello_bathtub_cost_is_270_l1111_111133

/-- Represents the cost in dollars to fill a bathtub with jello --/
def jelloBathtubCost (jelloMixPerPound : Real) (bathtubCapacity : Real) 
  (cubicFeetToGallons : Real) (poundsPerGallon : Real) (jelloMixCost : Real) : Real :=
  jelloMixPerPound * bathtubCapacity * cubicFeetToGallons * poundsPerGallon * jelloMixCost

/-- Theorem stating the cost to fill a bathtub with jello is $270 --/
theorem jello_bathtub_cost_is_270 :
  jelloBathtubCost 1.5 6 7.5 8 0.5 = 270 := by
  sorry

#eval jelloBathtubCost 1.5 6 7.5 8 0.5

end jello_bathtub_cost_is_270_l1111_111133


namespace exam_theorem_l1111_111199

def exam_problem (total_boys : ℕ) (overall_average : ℚ) (passed_boys : ℕ) (failed_average : ℚ) : Prop :=
  let passed_average : ℚ := (total_boys * overall_average - (total_boys - passed_boys) * failed_average) / passed_boys
  passed_average = 39

theorem exam_theorem : exam_problem 120 36 105 15 := by
  sorry

end exam_theorem_l1111_111199


namespace sum_of_powers_of_i_equals_zero_l1111_111145

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i_equals_zero :
  i^2022 + i^2023 + i^2024 + i^2025 = 0 :=
by
  sorry


end sum_of_powers_of_i_equals_zero_l1111_111145


namespace complex_modulus_problem_l1111_111131

/-- Given a complex number z satisfying zi = (2+i)^2, prove that |z| = 5 -/
theorem complex_modulus_problem (z : ℂ) (h : z * Complex.I = (2 + Complex.I)^2) : 
  Complex.abs z = 5 := by
  sorry

end complex_modulus_problem_l1111_111131


namespace centers_form_square_l1111_111108

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Represents a square -/
structure Square :=
  (center : Point)
  (side_length : ℝ)

/-- Function to construct squares on the sides of a parallelogram -/
def construct_squares (p : Parallelogram) : Square × Square × Square × Square :=
  sorry

/-- Function to check if four points form a square -/
def is_square (p q r s : Point) : Prop :=
  sorry

/-- Theorem: The centers of squares constructed on the sides of a parallelogram form a square -/
theorem centers_form_square (p : Parallelogram) :
  let (sq1, sq2, sq3, sq4) := construct_squares p
  is_square sq1.center sq2.center sq3.center sq4.center :=
sorry

end centers_form_square_l1111_111108


namespace arithmetic_sequence_seventh_term_l1111_111103

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_third : a 3 = 50)
  (h_fifth : a 5 = 30) :
  a 7 = 10 :=
sorry

end arithmetic_sequence_seventh_term_l1111_111103


namespace circle_equation_l1111_111114

open Real

/-- A circle C in polar coordinates -/
structure PolarCircle where
  center : ℝ × ℝ
  passesThrough : ℝ × ℝ

/-- The equation of a line in polar form -/
def polarLine (θ₀ : ℝ) (k : ℝ) : ℝ → ℝ → Prop :=
  fun ρ θ ↦ ρ * sin (θ - θ₀) = k

theorem circle_equation (C : PolarCircle) 
  (h1 : C.passesThrough = (2 * sqrt 2, π/4))
  (h2 : C.center.1 = 2 ∧ C.center.2 = 0)
  (h3 : polarLine (π/3) (-sqrt 3) C.center.1 C.center.2) :
  ∀ θ, ∃ ρ, ρ = 4 * cos θ ∧ (ρ * cos θ - C.center.1)^2 + (ρ * sin θ - C.center.2)^2 = (2 * sqrt 2 - C.center.1)^2 + (2 * sqrt 2 - C.center.2)^2 := by
  sorry

end circle_equation_l1111_111114


namespace quadratic_inequality_l1111_111194

theorem quadratic_inequality (x : ℝ) : 9 * x^2 + 6 * x - 8 > 0 ↔ x < -4/3 ∨ x > 2/3 := by
  sorry

end quadratic_inequality_l1111_111194


namespace investment_growth_l1111_111147

/-- Calculates the final amount after simple interest is applied --/
def final_amount (principal : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal + principal * rate * time

/-- Proves that an investment of $1000 at 10% simple interest for 3 years results in $1300 --/
theorem investment_growth :
  final_amount 1000 (1/10) 3 = 1300 := by
  sorry

end investment_growth_l1111_111147


namespace mothers_day_discount_percentage_l1111_111179

/-- Calculates the discount percentage for a Mother's day special at a salon -/
theorem mothers_day_discount_percentage 
  (regular_price : ℝ) 
  (num_services : ℕ) 
  (discounted_total : ℝ) 
  (h1 : regular_price = 40)
  (h2 : num_services = 5)
  (h3 : discounted_total = 150) : 
  (1 - discounted_total / (regular_price * num_services)) * 100 = 25 := by
  sorry

end mothers_day_discount_percentage_l1111_111179


namespace quadratic_properties_l1111_111155

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

/-- Theorem stating that f satisfies the required conditions -/
theorem quadratic_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by
  sorry

end quadratic_properties_l1111_111155


namespace bus_average_speed_l1111_111115

/-- The average speed of a bus catching up to a bicycle -/
theorem bus_average_speed (bicycle_speed : ℝ) (initial_distance : ℝ) (catch_up_time : ℝ) :
  bicycle_speed = 15 →
  initial_distance = 195 →
  catch_up_time = 3 →
  (initial_distance + bicycle_speed * catch_up_time) / catch_up_time = 80 :=
by sorry

end bus_average_speed_l1111_111115


namespace routes_between_plains_cities_l1111_111181

theorem routes_between_plains_cities
  (total_cities : Nat)
  (mountainous_cities : Nat)
  (plains_cities : Nat)
  (total_routes : Nat)
  (mountainous_routes : Nat)
  (h1 : total_cities = 100)
  (h2 : mountainous_cities = 30)
  (h3 : plains_cities = 70)
  (h4 : mountainous_cities + plains_cities = total_cities)
  (h5 : total_routes = 150)
  (h6 : mountainous_routes = 21) :
  ∃ (plains_routes : Nat),
    plains_routes = 81 ∧
    plains_routes + mountainous_routes + (total_routes - plains_routes - mountainous_routes) = total_routes :=
by sorry

end routes_between_plains_cities_l1111_111181


namespace permutations_of_four_l1111_111139

theorem permutations_of_four (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end permutations_of_four_l1111_111139


namespace partnership_capital_share_l1111_111142

theorem partnership_capital_share (Y : ℚ) : 
  (1 / 3 : ℚ) + (1 / 4 : ℚ) + Y + (1 - ((1 / 3 : ℚ) + (1 / 4 : ℚ) + Y)) = 1 →
  (1 / 3 : ℚ) + (1 / 4 : ℚ) + Y = 1 →
  Y = 5 / 12 := by
sorry

end partnership_capital_share_l1111_111142


namespace expression_rationality_expression_rationality_iff_l1111_111156

theorem expression_rationality (x : ℚ) : ∃ (k : ℚ), 
  x^2 + (Real.sqrt (x^2 + 1))^2 - 1 / (x^2 + (Real.sqrt (x^2 + 1))^2) = k := by
  sorry

theorem expression_rationality_iff : 
  ∀ x : ℝ, (∃ k : ℚ, x^2 + (Real.sqrt (x^2 + 1))^2 - 1 / (x^2 + (Real.sqrt (x^2 + 1))^2) = k) ↔ 
  ∃ q : ℚ, x = q := by
  sorry

end expression_rationality_expression_rationality_iff_l1111_111156


namespace negation_of_universal_proposition_l1111_111157

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^3 > x^2) ↔ ∃ x : ℝ, x^3 ≤ x^2 := by sorry

end negation_of_universal_proposition_l1111_111157


namespace larger_integer_value_l1111_111164

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) :
  max a b = 21 := by
sorry

end larger_integer_value_l1111_111164


namespace regression_line_equation_l1111_111105

/-- Regression line parameters -/
structure RegressionParams where
  x_bar : ℝ
  y_bar : ℝ
  slope : ℝ

/-- Regression line equation -/
def regression_line (params : RegressionParams) (x : ℝ) : ℝ :=
  params.slope * x + (params.y_bar - params.slope * params.x_bar)

/-- Theorem: Given the slope, x̄, and ȳ, prove the regression line equation -/
theorem regression_line_equation (params : RegressionParams)
  (h1 : params.x_bar = 4)
  (h2 : params.y_bar = 5)
  (h3 : params.slope = 2) :
  ∀ x, regression_line params x = 2 * x - 3 := by
  sorry

#check regression_line_equation

end regression_line_equation_l1111_111105


namespace christmas_tree_lights_l1111_111107

theorem christmas_tree_lights (T : ℝ) : ∃ (R Y G B : ℝ),
  R = 0.30 * T ∧
  Y = 0.45 * T ∧
  G = 110 ∧
  T = R + Y + G + B ∧
  B = 0.25 * T - 110 :=
by sorry

end christmas_tree_lights_l1111_111107


namespace angle_cosine_in_3d_space_l1111_111125

/-- Given a point P(x, y, z) in the first octant of 3D space, if the cosines of the angles between OP
    and the x-axis (α) and y-axis (β) are 1/3 and 1/5 respectively, then the cosine of the angle
    between OP and the z-axis (γ) is √(191)/15. -/
theorem angle_cosine_in_3d_space (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) :
  let magnitude := Real.sqrt (x^2 + y^2 + z^2)
  (x / magnitude = 1 / 3) → (y / magnitude = 1 / 5) → (z / magnitude = Real.sqrt 191 / 15) := by
  sorry

end angle_cosine_in_3d_space_l1111_111125


namespace median_list_i_equals_eight_l1111_111153

def list_i : List ℕ := [9, 2, 4, 7, 10, 11]
def list_ii : List ℕ := [3, 3, 4, 6, 7, 10]

def median (l : List ℕ) : ℚ := sorry
def mode (l : List ℕ) : ℕ := sorry

theorem median_list_i_equals_eight :
  median list_i = 8 :=
by
  have h1 : median list_i = median list_ii + mode list_ii := sorry
  sorry

#check median_list_i_equals_eight

end median_list_i_equals_eight_l1111_111153


namespace intersection_points_form_line_l1111_111149

theorem intersection_points_form_line (s : ℝ) :
  ∃ (x y : ℝ),
    (2 * x - 3 * y = 6 * s - 5) ∧
    (3 * x + y = 9 * s + 4) ∧
    (y = 3 * x + 16 / 11) :=
by sorry

end intersection_points_form_line_l1111_111149


namespace parabola_midpoint_trajectory_and_line_l1111_111122

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16*x

-- Define the trajectory E
def trajectory_E (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (3, 2)

theorem parabola_midpoint_trajectory_and_line :
  -- Part 1: Prove that the trajectory E is y² = 4x
  (∀ x y : ℝ, (∃ x₀ y₀ : ℝ, parabola x₀ y₀ ∧ x = x₀ ∧ y = y₀/2) → trajectory_E x y) ∧
  -- Part 2: Prove that the line l passing through P and intersecting E at A and B (where P is the midpoint of AB) has the equation x - y - 1 = 0
  (∀ A B : ℝ × ℝ,
    let (x₁, y₁) := A
    let (x₂, y₂) := B
    trajectory_E x₁ y₁ ∧ 
    trajectory_E x₂ y₂ ∧ 
    x₁ + x₂ = 2 * point_P.1 ∧
    y₁ + y₂ = 2 * point_P.2 →
    line_l x₁ y₁ ∧ line_l x₂ y₂) :=
by sorry

end parabola_midpoint_trajectory_and_line_l1111_111122


namespace probability_white_ball_l1111_111119

theorem probability_white_ball (n : ℕ) : 
  (2 : ℚ) / (n + 2) = 2 / 5 → (n : ℚ) / (n + 2) = 3 / 5 := by
  sorry

end probability_white_ball_l1111_111119


namespace philips_school_days_l1111_111151

/-- Given the following conditions:
  - The distance from Philip's house to school is 2.5 miles
  - The distance from Philip's house to the market is 2 miles
  - Philip makes two round trips to school each day he goes to school
  - Philip makes one round trip to the market during weekends
  - Philip's car's mileage for a typical week is 44 miles

  Prove that Philip makes round trips to school 4 days a week. -/
theorem philips_school_days :
  ∀ (school_distance market_distance : ℚ)
    (daily_school_trips weekly_market_trips : ℕ)
    (weekly_mileage : ℚ),
  school_distance = 5/2 →
  market_distance = 2 →
  daily_school_trips = 2 →
  weekly_market_trips = 1 →
  weekly_mileage = 44 →
  ∃ (days : ℕ),
    days = 4 ∧
    weekly_mileage = (2 * school_distance * daily_school_trips * days : ℚ) + (2 * market_distance * weekly_market_trips) :=
by sorry

end philips_school_days_l1111_111151


namespace A_power_95_l1111_111130

def A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 0; 0, 0, -1; 0, 1, 0]

theorem A_power_95 : A^95 = !![0, 0, 0; 0, 0, 1; 0, -1, 0] := by
  sorry

end A_power_95_l1111_111130


namespace julie_order_amount_l1111_111159

/-- The amount of food ordered by Julie -/
def julie_order : ℝ := 10

/-- The amount of food ordered by Letitia -/
def letitia_order : ℝ := 20

/-- The amount of food ordered by Anton -/
def anton_order : ℝ := 30

/-- The tip percentage -/
def tip_percentage : ℝ := 0.20

/-- The individual tip amount paid by each person -/
def individual_tip : ℝ := 4

theorem julie_order_amount :
  julie_order = 10 ∧
  letitia_order = 20 ∧
  anton_order = 30 ∧
  tip_percentage = 0.20 ∧
  individual_tip = 4 →
  tip_percentage * (julie_order + letitia_order + anton_order) = 3 * individual_tip :=
by sorry

end julie_order_amount_l1111_111159


namespace cupcakes_brought_is_correct_l1111_111183

/-- The number of cupcakes Dani brought to her 2nd-grade class. -/
def cupcakes_brought : ℕ := 30

/-- The total number of students in the class, including Dani. -/
def total_students : ℕ := 27

/-- The number of teachers in the class. -/
def teachers : ℕ := 1

/-- The number of teacher's aids in the class. -/
def teacher_aids : ℕ := 1

/-- The number of students who called in sick. -/
def sick_students : ℕ := 3

/-- The number of cupcakes left after distribution. -/
def leftover_cupcakes : ℕ := 4

/-- Theorem stating that the number of cupcakes Dani brought is correct. -/
theorem cupcakes_brought_is_correct :
  cupcakes_brought = 
    (total_students - sick_students + teachers + teacher_aids) + leftover_cupcakes :=
by
  sorry

end cupcakes_brought_is_correct_l1111_111183


namespace f_order_l1111_111160

def f (x : ℝ) : ℝ := sorry

axiom f_even : ∀ x, f x = f (-x)
axiom f_periodic : ∀ x, f (x + 2) = f x
axiom f_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x^(1/1998)

theorem f_order : f (101/17) < f (98/19) ∧ f (98/19) < f (104/15) := by sorry

end f_order_l1111_111160


namespace students_without_glasses_l1111_111191

theorem students_without_glasses (total_students : ℕ) (percent_with_glasses : ℚ) 
  (h1 : total_students = 325)
  (h2 : percent_with_glasses = 40/100) : 
  ↑total_students * (1 - percent_with_glasses) = 195 := by
  sorry

end students_without_glasses_l1111_111191


namespace sqrt_equation_solution_l1111_111177

theorem sqrt_equation_solution (c : ℝ) :
  Real.sqrt (9 + Real.sqrt (27 + 9*c)) + Real.sqrt (3 + Real.sqrt (3 + c)) = 3 + 3 * Real.sqrt 3 →
  c = 33 := by
sorry

end sqrt_equation_solution_l1111_111177


namespace system_solution_l1111_111173

theorem system_solution (a b : ℝ) : 
  (a * 1 - b * 2 = -1) → 
  (a * 1 + b * 2 = 7) → 
  3 * a - 4 * b = 1 := by
sorry

end system_solution_l1111_111173


namespace xy_value_l1111_111152

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 8) : x * y = 8 := by
  sorry

end xy_value_l1111_111152


namespace cubic_expansion_sum_l1111_111141

theorem cubic_expansion_sum (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x, (2*x + 1)^3 = a₀*x^3 + a₁*x^2 + a₂*x + a₃) →
  a₁ + a₃ = 13 := by
sorry

end cubic_expansion_sum_l1111_111141


namespace horse_speed_l1111_111174

theorem horse_speed (field_area : ℝ) (run_time : ℝ) (horse_speed : ℝ) : 
  field_area = 576 →
  run_time = 8 →
  horse_speed = (4 * Real.sqrt field_area) / run_time →
  horse_speed = 12 :=
by
  sorry

end horse_speed_l1111_111174


namespace negation_equivalence_l1111_111150

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) := by sorry

end negation_equivalence_l1111_111150


namespace angle_value_in_triangle_l1111_111195

/-- Given a triangle ABC where ∠ABC = 120°, and two angles are 3y° and y°, prove that y = 30 -/
theorem angle_value_in_triangle (y : ℝ) : 
  (3 * y + y = 120) → y = 30 := by sorry

end angle_value_in_triangle_l1111_111195


namespace problem_statements_l1111_111113

theorem problem_statements :
  (∃ x : ℝ, x^3 < 1) ∧
  ¬(∃ x : ℚ, x^2 = 2) ∧
  ¬(∀ x : ℕ, x^3 > x^2) ∧
  (∀ x : ℝ, x^2 + 1 > 0) :=
by sorry

end problem_statements_l1111_111113


namespace cos_alpha_proof_l1111_111140

def angle_alpha : ℝ := sorry

def point_P : ℝ × ℝ := (-4, 3)

theorem cos_alpha_proof :
  point_P.1 = -4 ∧ point_P.2 = 3 →
  point_P ∈ {p : ℝ × ℝ | ∃ r : ℝ, r > 0 ∧ p = (r * Real.cos angle_alpha, r * Real.sin angle_alpha)} →
  Real.cos angle_alpha = -4/5 := by
  sorry

#check cos_alpha_proof

end cos_alpha_proof_l1111_111140


namespace race_participants_race_result_l1111_111106

theorem race_participants (group_size : ℕ) (start_position : ℕ) (end_position : ℕ) : ℕ :=
  let total_groups := start_position + end_position - 1
  total_groups * group_size

theorem race_result : race_participants 3 7 5 = 33 := by
  sorry

end race_participants_race_result_l1111_111106


namespace infinitely_many_m_with_coprime_binomial_l1111_111104

theorem infinitely_many_m_with_coprime_binomial (k l : ℕ+) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ m ∈ S, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1 := by
  sorry

end infinitely_many_m_with_coprime_binomial_l1111_111104


namespace fraction_to_decimal_l1111_111132

theorem fraction_to_decimal (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 208 / 10000 :=
by
  sorry

end fraction_to_decimal_l1111_111132


namespace f_greater_than_one_iff_l1111_111118

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else Real.sqrt x

theorem f_greater_than_one_iff (x₀ : ℝ) :
  f x₀ > 1 ↔ x₀ < -1 ∨ x₀ > 1 := by sorry

end f_greater_than_one_iff_l1111_111118


namespace second_red_probability_l1111_111163

/-- Represents a bag of marbles -/
structure Bag where
  white : ℕ
  black : ℕ
  red : ℕ
  green : ℕ

/-- The probability of drawing a red marble as the second marble -/
def second_red_prob (bagA bagB bagC : Bag) : ℚ :=
  let total_A := bagA.white + bagA.black
  let total_B := bagB.red + bagB.green
  let total_C := bagC.red + bagC.green
  let prob_white_A := bagA.white / total_A
  let prob_black_A := bagA.black / total_A
  let prob_red_B := bagB.red / total_B
  let prob_red_C := bagC.red / total_C
  prob_white_A * prob_red_B + prob_black_A * prob_red_C

theorem second_red_probability :
  let bagA : Bag := { white := 4, black := 5, red := 0, green := 0 }
  let bagB : Bag := { white := 0, black := 0, red := 3, green := 7 }
  let bagC : Bag := { white := 0, black := 0, red := 5, green := 3 }
  second_red_prob bagA bagB bagC = 12 / 25 := by
  sorry

end second_red_probability_l1111_111163


namespace a_greater_than_b_l1111_111100

theorem a_greater_than_b : 
  let a := (-12) * (-23) * (-34) * (-45)
  let b := (-123) * (-234) * (-345)
  a > b := by
sorry

end a_greater_than_b_l1111_111100
