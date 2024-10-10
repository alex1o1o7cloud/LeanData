import Mathlib

namespace parallel_vectors_x_value_l2666_266628

/-- Given two vectors a and b in ℝ², if they are parallel and a = (3, 4) and b = (x, 1/2), then x = 3/8 -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) :
  a = (3, 4) →
  b = (x, 1/2) →
  ∃ (k : ℝ), a = k • b →
  x = 3/8 := by
sorry

end parallel_vectors_x_value_l2666_266628


namespace sqrt_sum_equals_ten_l2666_266652

theorem sqrt_sum_equals_ten : 
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 10 := by
  sorry

end sqrt_sum_equals_ten_l2666_266652


namespace fleas_perished_count_l2666_266657

/-- Represents the count of fleas in an ear -/
structure FleaCount where
  adultA : ℕ
  adultB : ℕ
  nymphA : ℕ
  nymphB : ℕ

/-- Represents the survival rates for different flea types -/
structure SurvivalRates where
  adultA : ℚ
  adultB : ℚ
  nymphA : ℚ
  nymphB : ℚ

def rightEar : FleaCount := {
  adultA := 42,
  adultB := 80,
  nymphA := 37,
  nymphB := 67
}

def leftEar : FleaCount := {
  adultA := 29,
  adultB := 64,
  nymphA := 71,
  nymphB := 45
}

def survivalRates : SurvivalRates := {
  adultA := 3/4,
  adultB := 3/5,
  nymphA := 2/5,
  nymphB := 11/20
}

/-- Calculates the number of fleas that perished in an ear -/
def fleaPerished (ear : FleaCount) (rates : SurvivalRates) : ℚ :=
  ear.adultA * (1 - rates.adultA) +
  ear.adultB * (1 - rates.adultB) +
  ear.nymphA * (1 - rates.nymphA) +
  ear.nymphB * (1 - rates.nymphB)

theorem fleas_perished_count :
  ⌊fleaPerished rightEar survivalRates + fleaPerished leftEar survivalRates⌋ = 190 := by
  sorry

end fleas_perished_count_l2666_266657


namespace base6_calculation_l2666_266622

/-- Represents a number in base 6 --/
def Base6 : Type := ℕ

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : Base6 := sorry

/-- Adds two numbers in base 6 --/
def addBase6 (a b : Base6) : Base6 := sorry

/-- Subtracts two numbers in base 6 --/
def subBase6 (a b : Base6) : Base6 := sorry

/-- Theorem: 15₆ - 4₆ + 20₆ = 31₆ in base 6 --/
theorem base6_calculation : 
  let a := toBase6 15
  let b := toBase6 4
  let c := toBase6 20
  let d := toBase6 31
  addBase6 (subBase6 a b) c = d := by sorry

end base6_calculation_l2666_266622


namespace semicircle_perimeter_l2666_266645

/-- The perimeter of a semi-circle with radius 2.1 cm is π * 2.1 + 4.2 cm. -/
theorem semicircle_perimeter :
  let r : ℝ := 2.1
  (π * r + 2 * r) = π * 2.1 + 4.2 := by sorry

end semicircle_perimeter_l2666_266645


namespace sinusoidal_midline_l2666_266602

theorem sinusoidal_midline (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) →
  d = 1 := by
sorry

end sinusoidal_midline_l2666_266602


namespace part_one_part_two_l2666_266692

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x - a| + a
def g (x : ℝ) : ℝ := |2 * x - 1|

-- Part I
theorem part_one : 
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Part II
theorem part_two :
  (∀ x : ℝ, f a x + g x ≥ 3) → a ≥ 2 := by sorry

end part_one_part_two_l2666_266692


namespace biased_coin_probability_l2666_266665

def binomial (n k : ℕ) : ℕ := (Nat.choose n k)

theorem biased_coin_probability : ∃ (h : ℚ), 
  (0 < h ∧ h < 1) ∧ 
  (binomial 6 2 : ℚ) * h^2 * (1-h)^4 = (binomial 6 3 : ℚ) * h^3 * (1-h)^3 → 
  (binomial 6 4 : ℚ) * h^4 * (1-h)^2 = 19440 / 117649 :=
sorry

end biased_coin_probability_l2666_266665


namespace no_solutions_for_exponential_equations_l2666_266601

theorem no_solutions_for_exponential_equations :
  (∀ n : ℕ, n > 1 → ¬∃ (p m : ℕ), Nat.Prime p ∧ Odd p ∧ m > 0 ∧ p^n + 1 = 2^m) ∧
  (∀ n : ℕ, n > 2 → ¬∃ (p m : ℕ), Nat.Prime p ∧ Odd p ∧ m > 0 ∧ p^n - 1 = 2^m) := by
  sorry

end no_solutions_for_exponential_equations_l2666_266601


namespace local_minimum_at_zero_l2666_266688

/-- The function f(x) = (x^2 - 1)^3 + 1 has a local minimum at x = 0 -/
theorem local_minimum_at_zero (f : ℝ → ℝ) (h : f = λ x => (x^2 - 1)^3 + 1) :
  ∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f 0 ≤ f x :=
sorry

end local_minimum_at_zero_l2666_266688


namespace consecutive_numbers_sum_l2666_266649

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) = 60) → 
  ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  sorry

end consecutive_numbers_sum_l2666_266649


namespace square_perimeter_ratio_l2666_266683

theorem square_perimeter_ratio (x y : ℝ) (h : x * Real.sqrt 2 = 1.5 * y * Real.sqrt 2) :
  (4 * x) / (4 * y) = 1.5 := by
  sorry

end square_perimeter_ratio_l2666_266683


namespace inequality_proof_l2666_266664

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt ((a^2 + b^2 + c^2 + d^2) / 4) ≥ (((a*b*c + a*b*d + a*c*d + b*c*d) / 4) ^ (1/3)) := by
  sorry

end inequality_proof_l2666_266664


namespace root_sum_inverse_complement_l2666_266620

def cubic_polynomial (x : ℝ) : ℝ := 45 * x^3 - 75 * x^2 + 33 * x - 2

theorem root_sum_inverse_complement (a b c : ℝ) : 
  (cubic_polynomial a = 0) → 
  (cubic_polynomial b = 0) → 
  (cubic_polynomial c = 0) → 
  (a ≠ b) → (b ≠ c) → (a ≠ c) → 
  (0 < a) → (a < 1) → 
  (0 < b) → (b < 1) → 
  (0 < c) → (c < 1) → 
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 60) :=
by sorry

end root_sum_inverse_complement_l2666_266620


namespace remainder_equality_l2666_266607

theorem remainder_equality (x : ℕ+) (y : ℤ) 
  (h1 : ∃ k : ℤ, 200 = k * x.val + 5) 
  (h2 : ∃ m : ℤ, y = m * x.val + 5) : 
  y % x.val = 5 := by
  sorry

end remainder_equality_l2666_266607


namespace two_pencils_length_l2666_266615

/-- The length of a pencil in cubes -/
def PencilLength : ℕ := 12

/-- The total length of two pencils -/
def TotalLength : ℕ := PencilLength + PencilLength

/-- Theorem: The total length of two pencils, each 12 cubes long, is 24 cubes -/
theorem two_pencils_length : TotalLength = 24 := by
  sorry

end two_pencils_length_l2666_266615


namespace tim_weekly_earnings_l2666_266621

/-- Tim's daily tasks -/
def daily_tasks : ℕ := 100

/-- Pay per task in dollars -/
def pay_per_task : ℚ := 6/5

/-- Number of working days per week -/
def working_days_per_week : ℕ := 6

/-- Tim's weekly earnings in dollars -/
def weekly_earnings : ℚ := daily_tasks * pay_per_task * working_days_per_week

theorem tim_weekly_earnings : weekly_earnings = 720 := by sorry

end tim_weekly_earnings_l2666_266621


namespace grid_arithmetic_sequence_l2666_266698

theorem grid_arithmetic_sequence (row : Fin 7 → ℚ) (col1 col2 : Fin 5 → ℚ) :
  -- The row forms an arithmetic sequence
  (∀ i : Fin 6, row (i + 1) - row i = row 1 - row 0) →
  -- The first column forms an arithmetic sequence
  (∀ i : Fin 4, col1 (i + 1) - col1 i = col1 1 - col1 0) →
  -- The second column forms an arithmetic sequence
  (∀ i : Fin 4, col2 (i + 1) - col2 i = col2 1 - col2 0) →
  -- Given values
  row 0 = 25 →
  col1 2 = 16 →
  col1 3 = 20 →
  col2 4 = -21 →
  -- The fourth element in the row is the same as the first element in the first column
  row 3 = col1 0 →
  -- The last element in the row is the same as the first element in the second column
  row 6 = col2 0 →
  -- M is the first element in the second column
  col2 0 = 1021 / 12 := by
sorry

end grid_arithmetic_sequence_l2666_266698


namespace problem_statement_l2666_266606

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) :
  (x - 2)^2 + 25/((x - 2)^2) = -x + 5 := by
  sorry

end problem_statement_l2666_266606


namespace four_number_theorem_l2666_266680

theorem four_number_theorem (a b c d : ℕ+) (h : a * b = c * d) :
  ∃ (p q r s : ℕ+), a = p * q ∧ b = r * s ∧ c = p * s ∧ d = q * r := by
  sorry

end four_number_theorem_l2666_266680


namespace margie_change_l2666_266697

theorem margie_change : 
  let banana_cost : ℚ := 0.30
  let orange_cost : ℚ := 0.40
  let banana_count : ℕ := 5
  let orange_count : ℕ := 3
  let paid_amount : ℚ := 10.00
  let total_cost : ℚ := banana_cost * banana_count + orange_cost * orange_count
  let change : ℚ := paid_amount - total_cost
  change = 7.30 := by sorry

end margie_change_l2666_266697


namespace opposite_lime_is_black_l2666_266691

-- Define the colors
inductive Color
  | Purple
  | Cyan
  | Magenta
  | Silver
  | Black
  | Lime

-- Define a square with a color
structure Square where
  color : Color

-- Define a cube made of squares
structure Cube where
  squares : List Square
  hinged : squares.length = 6

-- Define the opposite face relation
def oppositeFace (c : Cube) (f1 f2 : Square) : Prop :=
  f1 ∈ c.squares ∧ f2 ∈ c.squares ∧ f1 ≠ f2

-- Theorem statement
theorem opposite_lime_is_black (c : Cube) :
  ∃ (lime_face black_face : Square),
    lime_face.color = Color.Lime ∧
    black_face.color = Color.Black ∧
    oppositeFace c lime_face black_face :=
  sorry


end opposite_lime_is_black_l2666_266691


namespace factor_expression_l2666_266647

theorem factor_expression (m : ℝ) : 2 * m^2 - 2 = 2 * (m + 1) * (m - 1) := by
  sorry

end factor_expression_l2666_266647


namespace problem_solution_l2666_266629

theorem problem_solution (a b c d : ℝ) : 
  a^2 + b^2 + c^2 + 4 = d + Real.sqrt (2*a + 2*b + 2*c - d) → d = 23/4 := by
  sorry

end problem_solution_l2666_266629


namespace weight_problem_l2666_266666

/-- Given three weights a, b, and c, prove that their average weights satisfy the given conditions --/
theorem weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →  -- average weight of a, b, and c is 45 kg
  (b + c) / 2 = 44 →      -- average weight of b and c is 44 kg
  b = 33 →                -- weight of b is 33 kg
  (a + b) / 2 = 40        -- average weight of a and b is 40 kg
:= by sorry

end weight_problem_l2666_266666


namespace problem_solution_l2666_266656

theorem problem_solution (A B : ℝ) (h1 : B + A + B = 814.8) (h2 : A = 10 * B) : A - B = 611.1 := by
  sorry

end problem_solution_l2666_266656


namespace vector_relation_l2666_266671

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define points A, B, and C
variable (A B C : V)

-- Define the theorem
theorem vector_relation (h1 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B) 
                        (h2 : C - A = (3/5) • (B - A)) : 
  C - A = -(3/2) • (C - B) := by
  sorry

end vector_relation_l2666_266671


namespace arithmetic_sequence_condition_l2666_266639

/-- Four real numbers are in arithmetic sequence -/
def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r

/-- The sum of the first and last terms equals the sum of the middle terms -/
def sum_property (a b c d : ℝ) : Prop :=
  a + d = b + c

theorem arithmetic_sequence_condition (a b c d : ℝ) :
  (is_arithmetic_sequence a b c d → sum_property a b c d) ∧
  ¬(sum_property a b c d → is_arithmetic_sequence a b c d) :=
sorry

end arithmetic_sequence_condition_l2666_266639


namespace composite_function_ratio_l2666_266619

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem composite_function_ratio : 
  (f (g (f 3))) / (g (f (g 3))) = 59 / 35 := by
  sorry

end composite_function_ratio_l2666_266619


namespace degree_of_polynomial_power_l2666_266613

/-- The degree of the polynomial (5x^3 + 7)^10 is 30. -/
theorem degree_of_polynomial_power : 
  Polynomial.degree ((5 * X ^ 3 + 7 : Polynomial ℝ) ^ 10) = 30 := by
  sorry

end degree_of_polynomial_power_l2666_266613


namespace twelve_sided_polygon_area_l2666_266676

/-- A 12-sided polygon composed of squares and triangles on a grid --/
structure TwelveSidedPolygon where
  center_square : ℝ  -- Area of the center square
  corner_triangles : ℝ  -- Number of corner triangles
  side_triangles : ℝ  -- Number of effective side triangles
  unit_square_area : ℝ  -- Area of a unit square
  unit_triangle_area : ℝ  -- Area of a unit right triangle

/-- The area of the 12-sided polygon --/
def polygon_area (p : TwelveSidedPolygon) : ℝ :=
  p.center_square * p.unit_square_area +
  p.corner_triangles * p.unit_triangle_area +
  p.side_triangles * p.unit_square_area

/-- Theorem stating that the area of the specific 12-sided polygon is 13 square units --/
theorem twelve_sided_polygon_area :
  ∀ (p : TwelveSidedPolygon),
  p.center_square = 9 ∧
  p.corner_triangles = 4 ∧
  p.side_triangles = 4 ∧
  p.unit_square_area = 1 ∧
  p.unit_triangle_area = 1/2 →
  polygon_area p = 13 := by
  sorry

end twelve_sided_polygon_area_l2666_266676


namespace polynomial_simplification_l2666_266672

theorem polynomial_simplification (x : ℝ) :
  (3 * x^6 + 2 * x^5 - x^4 + 3 * x^2 + 15) - (x^6 + 4 * x^5 + 3 * x^3 - 2 * x^2 + 20) =
  2 * x^6 - 2 * x^5 - x^4 + 5 * x^2 - 5 := by
  sorry

end polynomial_simplification_l2666_266672


namespace expression_increase_l2666_266661

theorem expression_increase (x y : ℝ) : 
  let original := 3 * x^2 * y
  let new_x := 1.2 * x
  let new_y := 2.4 * y
  let new_expression := 3 * new_x^2 * new_y
  new_expression = 3.456 * original := by
sorry

end expression_increase_l2666_266661


namespace vector_parallel_condition_l2666_266682

-- Define the vectors
def a : ℝ × ℝ := (2, -1)
def b (m : ℝ) : ℝ × ℝ := (-1, m)
def c : ℝ × ℝ := (-1, 2)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem vector_parallel_condition (m : ℝ) :
  parallel (a.1 + (b m).1, a.2 + (b m).2) c → m = -1 := by
  sorry

end vector_parallel_condition_l2666_266682


namespace surface_area_increase_after_cube_removal_l2666_266616

/-- Represents a rectangular solid with given dimensions -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cube with given side length -/
structure Cube where
  sideLength : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Calculates the surface area of a cube -/
def cubeSurfaceArea (cube : Cube) : ℝ :=
  6 * cube.sideLength^2

/-- Theorem: Removing a cube from the center of a rectangular solid increases surface area -/
theorem surface_area_increase_after_cube_removal 
  (solid : RectangularSolid) 
  (cube : Cube) 
  (h1 : solid.length = 4) 
  (h2 : solid.width = 3) 
  (h3 : solid.height = 5) 
  (h4 : cube.sideLength = 2) :
  surfaceArea solid + cubeSurfaceArea cube = surfaceArea solid + 24 := by
  sorry

end surface_area_increase_after_cube_removal_l2666_266616


namespace queen_jack_hands_count_l2666_266640

/-- The number of queens in a standard deck --/
def num_queens : ℕ := 4

/-- The number of jacks in a standard deck --/
def num_jacks : ℕ := 4

/-- The total number of queens and jacks --/
def total_queens_jacks : ℕ := num_queens + num_jacks

/-- The number of cards in a hand --/
def hand_size : ℕ := 5

/-- The number of 5-card hands containing only queens and jacks --/
def num_queen_jack_hands : ℕ := Nat.choose total_queens_jacks hand_size

theorem queen_jack_hands_count : num_queen_jack_hands = 56 := by
  sorry

end queen_jack_hands_count_l2666_266640


namespace complex_sum_of_parts_l2666_266653

theorem complex_sum_of_parts (a b : ℝ) : (Complex.I * (1 - Complex.I) = Complex.mk a b) → a + b = 2 := by
  sorry

end complex_sum_of_parts_l2666_266653


namespace library_problem_l2666_266686

theorem library_problem (total_books : ℕ) (books_per_student : ℕ) 
  (students_day1 : ℕ) (students_day2 : ℕ) (students_day3 : ℕ) : ℕ :=
  by
  have h1 : total_books = 120 := by sorry
  have h2 : books_per_student = 5 := by sorry
  have h3 : students_day1 = 4 := by sorry
  have h4 : students_day2 = 5 := by sorry
  have h5 : students_day3 = 6 := by sorry
  
  have remaining_books : ℕ := total_books - (students_day1 + students_day2 + students_day3) * books_per_student
  
  exact remaining_books / books_per_student

end library_problem_l2666_266686


namespace election_votes_theorem_l2666_266667

theorem election_votes_theorem (total_votes : ℕ) 
  (h1 : (13 : ℚ) / 20 * total_votes = 39 + (total_votes - 39)) : 
  total_votes = 60 := by
  sorry

end election_votes_theorem_l2666_266667


namespace granola_discounted_price_l2666_266677

/-- Calculates the discounted price per bag of granola given the following conditions:
    - Cost of ingredients per bag
    - Total number of bags made
    - Original selling price per bag
    - Number of bags sold at original price
    - Total net profit -/
def discounted_price (cost_per_bag : ℚ) (total_bags : ℕ) (original_price : ℚ)
                     (bags_sold_full_price : ℕ) (net_profit : ℚ) : ℚ :=
  let total_cost := cost_per_bag * total_bags
  let full_price_revenue := original_price * bags_sold_full_price
  let total_revenue := net_profit + total_cost
  let discounted_revenue := total_revenue - full_price_revenue
  let discounted_bags := total_bags - bags_sold_full_price
  discounted_revenue / discounted_bags

theorem granola_discounted_price :
  discounted_price 3 20 6 15 50 = 4 := by
  sorry

end granola_discounted_price_l2666_266677


namespace basketball_handshakes_l2666_266668

/-- Calculates the total number of handshakes in a basketball game scenario --/
def total_handshakes (players_per_team : ℕ) (num_referees : ℕ) (num_coaches : ℕ) : ℕ :=
  let player_handshakes := players_per_team * players_per_team
  let player_referee_handshakes := 2 * players_per_team * num_referees
  let coach_handshakes := num_coaches * (2 * players_per_team + num_referees)
  player_handshakes + player_referee_handshakes + coach_handshakes

/-- Theorem stating that the total number of handshakes in the given scenario is 102 --/
theorem basketball_handshakes :
  total_handshakes 6 3 2 = 102 := by
  sorry

#eval total_handshakes 6 3 2

end basketball_handshakes_l2666_266668


namespace jessicas_balloons_l2666_266624

theorem jessicas_balloons (joan_balloons sally_balloons total_balloons : ℕ) 
  (h1 : joan_balloons = 9)
  (h2 : sally_balloons = 5)
  (h3 : total_balloons = 16) :
  total_balloons - (joan_balloons + sally_balloons) = 2 := by
  sorry

end jessicas_balloons_l2666_266624


namespace quadratic_equation_solution_l2666_266689

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 2 * x^2 - 3 * x - 5
  ∃ x₁ x₂ : ℝ, x₁ = 5/2 ∧ x₂ = -1 ∧ f x₁ = 0 ∧ f x₂ = 0 := by
  sorry

end quadratic_equation_solution_l2666_266689


namespace regression_equation_properties_l2666_266630

-- Define the concept of a regression equation
structure RegressionEquation where
  -- Add necessary fields here
  mk :: -- Constructor

-- Define the property of temporality for regression equations
def has_temporality (eq : RegressionEquation) : Prop := sorry

-- Define the concept of sample values affecting applicability
def sample_values_affect_applicability (eq : RegressionEquation) : Prop := sorry

-- Theorem stating the correct properties of regression equations
theorem regression_equation_properties :
  ∀ (eq : RegressionEquation),
    (has_temporality eq) ∧
    (sample_values_affect_applicability eq) := by
  sorry

end regression_equation_properties_l2666_266630


namespace possible_values_of_y_l2666_266699

theorem possible_values_of_y (x y : ℝ) :
  |x - Real.sin (Real.log y)| = x + Real.sin (Real.log y) →
  ∃ n : ℤ, y = Real.exp (2 * π * ↑n) :=
sorry

end possible_values_of_y_l2666_266699


namespace sin_cos_sum_equals_sqrt3_over_2_l2666_266660

theorem sin_cos_sum_equals_sqrt3_over_2 :
  Real.sin (20 * π / 180) * Real.sin (50 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (40 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end sin_cos_sum_equals_sqrt3_over_2_l2666_266660


namespace point_on_line_value_l2666_266626

theorem point_on_line_value (x y : ℝ) (h1 : y = x + 2) (h2 : 1 < y) (h3 : y < 3) :
  Real.sqrt (y^2 - 8*x) + Real.sqrt (y^2 + 2*x + 5) = 5 := by
  sorry

end point_on_line_value_l2666_266626


namespace cafeteria_red_apples_l2666_266603

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := sorry

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 32

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 2

/-- The number of extra apples -/
def extra_apples : ℕ := 73

/-- Theorem stating that the number of red apples ordered is 43 -/
theorem cafeteria_red_apples : red_apples = 43 := by sorry

end cafeteria_red_apples_l2666_266603


namespace discriminant_of_quadratic_l2666_266690

/-- The discriminant of a quadratic polynomial ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The quadratic polynomial 2x^2 + (4 - 1/2)x + 1 -/
def quadratic_polynomial (x : ℚ) : ℚ := 2*x^2 + (4 - 1/2)*x + 1

theorem discriminant_of_quadratic : 
  discriminant 2 (4 - 1/2) 1 = 17/4 := by
  sorry

end discriminant_of_quadratic_l2666_266690


namespace only_set_A_is_right_triangle_l2666_266658

-- Define a function to check if three numbers form a right triangle
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

-- Define the sets of numbers
def set_A : List ℕ := [5, 12, 13]
def set_B : List ℕ := [3, 4, 6]
def set_C : List ℕ := [4, 5, 6]
def set_D : List ℕ := [5, 7, 9]

-- Theorem to prove
theorem only_set_A_is_right_triangle :
  (is_right_triangle 5 12 13) ∧
  ¬(is_right_triangle 3 4 6) ∧
  ¬(is_right_triangle 4 5 6) ∧
  ¬(is_right_triangle 5 7 9) :=
sorry

end only_set_A_is_right_triangle_l2666_266658


namespace closest_to_product_l2666_266694

def options : List ℝ := [7, 42, 74, 84, 737]

def product : ℝ := 1.8 * (40.3 + 0.07)

theorem closest_to_product : 
  ∃ (x : ℝ), x ∈ options ∧ 
  ∀ (y : ℝ), y ∈ options → |x - product| ≤ |y - product| ∧ 
  x = 74 :=
by sorry

end closest_to_product_l2666_266694


namespace fewer_heads_probability_l2666_266687

/-- The probability of getting fewer heads than tails when flipping 12 fair coins -/
def fewer_heads_prob : ℚ := 1586 / 4096

/-- The number of coins being flipped -/
def num_coins : ℕ := 12

theorem fewer_heads_probability :
  fewer_heads_prob = (2^num_coins - (num_coins.choose (num_coins / 2))) / (2 * 2^num_coins) :=
sorry

end fewer_heads_probability_l2666_266687


namespace min_value_f_exists_min_f_l2666_266654

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

-- Theorem for the minimum value of f(x)
theorem min_value_f (a : ℝ) :
  (a = 1 → ∀ x ∈ Set.Icc (-1) 0, f a x ≥ 5) ∧
  (a ≤ -1 → ∀ x ∈ Set.Icc (-1) 0, f a x ≥ 6 + 2*a) ∧
  (-1 < a ∧ a < 0 → ∀ x ∈ Set.Icc (-1) 0, f a x ≥ 5 - a^2) :=
by sorry

-- Theorem for the existence of x that achieves the minimum
theorem exists_min_f (a : ℝ) :
  (a = 1 → ∃ x ∈ Set.Icc (-1) 0, f a x = 5) ∧
  (a ≤ -1 → ∃ x ∈ Set.Icc (-1) 0, f a x = 6 + 2*a) ∧
  (-1 < a ∧ a < 0 → ∃ x ∈ Set.Icc (-1) 0, f a x = 5 - a^2) :=
by sorry

end min_value_f_exists_min_f_l2666_266654


namespace absolute_value_inequality_l2666_266635

theorem absolute_value_inequality (m : ℝ) : 
  (∀ x : ℝ, |x + 5| ≥ m + 2) → m ≤ -2 := by sorry

end absolute_value_inequality_l2666_266635


namespace mixed_committee_probability_l2666_266638

/-- The number of members in the Book club -/
def total_members : ℕ := 24

/-- The number of boys in the Book club -/
def num_boys : ℕ := 12

/-- The number of girls in the Book club -/
def num_girls : ℕ := 12

/-- The size of the committee -/
def committee_size : ℕ := 5

/-- The probability of choosing a committee with at least one boy and one girl -/
def probability_mixed_committee : ℚ := 171 / 177

theorem mixed_committee_probability :
  (1 : ℚ) - (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size : ℚ) / 
  (Nat.choose total_members committee_size : ℚ) = probability_mixed_committee := by
  sorry

end mixed_committee_probability_l2666_266638


namespace geometric_series_sum_l2666_266679

/-- Given a sequence a_n and its partial sum S_n, prove that S_20 = 400 -/
theorem geometric_series_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = (a n + 1)^2 / 4) →
  (∀ n, a n = 2 * n - 1) →
  S 20 = 400 := by
sorry

end geometric_series_sum_l2666_266679


namespace max_npk_l2666_266659

/-- Represents a single digit integer -/
def SingleDigit := {n : ℕ | 1 ≤ n ∧ n ≤ 9}

/-- Converts a single digit to a two-digit number with repeated digits -/
def toTwoDigit (m : SingleDigit) : ℕ := 11 * m

/-- Checks if a number is three digits -/
def isThreeDigits (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

/-- The result of multiplying a two-digit number by a single digit -/
def result (m k : SingleDigit) : ℕ := toTwoDigit m * k

theorem max_npk :
  ∀ m k : SingleDigit,
    m ≠ k →
    isThreeDigits (result m k) →
    ∀ m' k' : SingleDigit,
      m' ≠ k' →
      isThreeDigits (result m' k') →
      result m' k' ≤ 891 :=
sorry

end max_npk_l2666_266659


namespace geometric_sequence_common_ratio_l2666_266627

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h1 : a 3 = 3/2) 
  (h2 : S 3 = 9/2) : 
  ∃ q : ℚ, (q = 1 ∨ q = -1/2) ∧ 
    (∀ n : ℕ, n ≥ 1 → a n = a 1 * q^(n-1)) ∧
    (∀ n : ℕ, n ≥ 1 → S n = a 1 * (1 - q^n) / (1 - q)) :=
sorry

end geometric_sequence_common_ratio_l2666_266627


namespace smallest_congruent_number_l2666_266674

theorem smallest_congruent_number : ∃ n : ℕ, 
  n > 1 ∧ 
  n % 6 = 1 ∧ 
  n % 7 = 1 ∧ 
  n % 8 = 1 ∧
  (∀ m : ℕ, m > 1 → m % 6 = 1 → m % 7 = 1 → m % 8 = 1 → n ≤ m) ∧
  n = 169 := by
  sorry

end smallest_congruent_number_l2666_266674


namespace intersection_M_N_l2666_266673

def M : Set ℝ := {-4, -3, -2, -1, 0, 1}

def N : Set ℝ := {x : ℝ | x^2 + 3*x < 0}

theorem intersection_M_N : M ∩ N = {-2, -1} := by sorry

end intersection_M_N_l2666_266673


namespace bus_distance_l2666_266618

theorem bus_distance (total_distance : ℝ) (plane_fraction : ℝ) (train_bus_ratio : ℝ)
  (h1 : total_distance = 900)
  (h2 : plane_fraction = 1 / 3)
  (h3 : train_bus_ratio = 2 / 3) :
  let plane_distance := total_distance * plane_fraction
  let bus_distance := (total_distance - plane_distance) / (1 + train_bus_ratio)
  bus_distance = 360 := by
sorry

end bus_distance_l2666_266618


namespace pet_store_cages_l2666_266646

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : puppies_per_cage = 5) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end pet_store_cages_l2666_266646


namespace equation_solution_l2666_266605

theorem equation_solution (m : ℕ) : 
  ((1^m : ℚ) / (5^m)) * ((1^16 : ℚ) / (4^16)) = 1 / (2 * (10^31)) → m = 31 := by
  sorry

end equation_solution_l2666_266605


namespace rice_weight_in_pounds_l2666_266696

/-- Given rice divided equally into 4 containers, with each container having 33 ounces,
    and 1 pound being equal to 16 ounces, the total weight of rice in pounds is 8.25. -/
theorem rice_weight_in_pounds :
  let num_containers : ℕ := 4
  let ounces_per_container : ℚ := 33
  let ounces_per_pound : ℚ := 16
  let total_ounces : ℚ := num_containers * ounces_per_container
  let total_pounds : ℚ := total_ounces / ounces_per_pound
  total_pounds = 8.25 := by sorry

end rice_weight_in_pounds_l2666_266696


namespace reciprocal_of_2024_l2666_266634

theorem reciprocal_of_2024 : (2024⁻¹ : ℚ) = 1 / 2024 := by
  sorry

end reciprocal_of_2024_l2666_266634


namespace unique_outstanding_wins_all_l2666_266631

variable {α : Type*} [Fintype α] [DecidableEq α]

-- Define the winning relation
variable (wins : α → α → Prop)

-- Assumption: Every pair of contestants has a clear winner
axiom clear_winner (a b : α) : a ≠ b → (wins a b ∨ wins b a) ∧ ¬(wins a b ∧ wins b a)

-- Define what it means to be an outstanding contestant
def is_outstanding (a : α) : Prop :=
  ∀ b : α, b ≠ a → wins a b ∨ (∃ c : α, wins c b ∧ wins a c)

-- Theorem: If there is a unique outstanding contestant, they win against all others
theorem unique_outstanding_wins_all (a : α) :
  (is_outstanding wins a ∧ ∀ b : α, is_outstanding wins b → b = a) →
  ∀ b : α, b ≠ a → wins a b :=
by sorry

end unique_outstanding_wins_all_l2666_266631


namespace inverse_sum_theorem_l2666_266643

-- Define a function f: ℝ → ℝ with an inverse
def f : ℝ → ℝ := sorry

-- Assume f is bijective (has an inverse)
axiom f_bijective : Function.Bijective f

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- Condition: f(x) + f(-x) = 4 for all x
axiom f_condition (x : ℝ) : f x + f (-x) = 4

-- Theorem to prove
theorem inverse_sum_theorem (x : ℝ) : f_inv (x - 3) + f_inv (7 - x) = 0 := by sorry

end inverse_sum_theorem_l2666_266643


namespace zero_location_l2666_266617

theorem zero_location (x y : ℝ) (h : x^5 < y^8 ∧ y^8 < y^3 ∧ y^3 < x^6) :
  x^5 < 0 ∧ 0 < y^8 := by
  sorry

end zero_location_l2666_266617


namespace solution_set_inequality_l2666_266670

theorem solution_set_inequality (x : ℝ) :
  (1 / (x - 1) ≥ -1) ↔ (x ≤ 0 ∨ x > 1) := by sorry

end solution_set_inequality_l2666_266670


namespace johnson_family_reunion_ratio_l2666_266623

theorem johnson_family_reunion_ratio :
  ∀ (total_adults : ℕ) (total_children : ℕ),
  total_children = 45 →
  (total_adults / 3 : ℚ) + 10 = total_adults →
  (total_adults : ℚ) / total_children = 1 / 3 :=
by
  sorry

end johnson_family_reunion_ratio_l2666_266623


namespace matrix_N_property_l2666_266684

variable (N : Matrix (Fin 2) (Fin 2) ℝ)

theorem matrix_N_property (h1 : N.mulVec ![3, -2] = ![4, 1])
                          (h2 : N.mulVec ![-2, 4] = ![0, 2]) :
  N.mulVec ![7, 0] = ![14, 7] := by
  sorry

end matrix_N_property_l2666_266684


namespace expression_simplification_l2666_266685

theorem expression_simplification (y : ℝ) : 
  4*y + 9*y^2 + 8 - (3 - 4*y - 9*y^2) = 18*y^2 + 8*y + 5 := by
  sorry

end expression_simplification_l2666_266685


namespace fraction_simplification_l2666_266633

theorem fraction_simplification (x y z : ℚ) :
  x = 3 ∧ y = 4 ∧ z = 2 →
  (10 * x * y^3) / (15 * x^2 * y * z) = 16 / 9 := by
sorry

end fraction_simplification_l2666_266633


namespace intersection_with_y_axis_l2666_266644

/-- The intersection point of y = -4x + 2 with the y-axis is (0, 2) -/
theorem intersection_with_y_axis :
  let f (x : ℝ) := -4 * x + 2
  (0, f 0) = (0, 2) := by sorry

end intersection_with_y_axis_l2666_266644


namespace intersection_of_A_and_B_l2666_266604

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l2666_266604


namespace triangle_angle_calculation_l2666_266681

theorem triangle_angle_calculation (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  C = 2 * B →        -- Angle C is double angle B
  A = 3 * B →        -- Angle A is thrice angle B
  B = 30 :=          -- Angle B is 30°
by
  sorry

end triangle_angle_calculation_l2666_266681


namespace ages_sum_l2666_266693

theorem ages_sum (a b c : ℕ+) (h1 : b = c) (h2 : a * b * c = 72) : a + b + c = 14 := by
  sorry

end ages_sum_l2666_266693


namespace pure_imaginary_implies_a_zero_l2666_266611

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Given complex number z as a function of real number a -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 2*a) (a - 2)

/-- Theorem: If z(a) is a pure imaginary number, then a = 0 -/
theorem pure_imaginary_implies_a_zero : 
  ∀ a : ℝ, is_pure_imaginary (z a) → a = 0 := by
  sorry

end pure_imaginary_implies_a_zero_l2666_266611


namespace triangle_angle_proof_l2666_266662

theorem triangle_angle_proof (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α = 100 →          -- One angle is 100°
  β = 2 * γ →        -- One angle is twice the other
  γ = 26 :=          -- The smallest angle is 26°
by sorry

end triangle_angle_proof_l2666_266662


namespace sin_pi_12_plus_theta_l2666_266669

theorem sin_pi_12_plus_theta (θ : Real) 
  (h : Real.cos (5 * Real.pi / 12 - θ) = 1 / 3) : 
  Real.sin (Real.pi / 12 + θ) = 1 / 3 := by
  sorry

end sin_pi_12_plus_theta_l2666_266669


namespace princess_daphne_necklaces_l2666_266650

def total_cost : ℕ := 240000
def necklace_cost : ℕ := 40000

def number_of_necklaces : ℕ := 3

theorem princess_daphne_necklaces :
  ∃ (n : ℕ), n * necklace_cost + 3 * necklace_cost = total_cost ∧ n = number_of_necklaces :=
by sorry

end princess_daphne_necklaces_l2666_266650


namespace exists_config_with_more_than_20_components_l2666_266655

/-- A configuration of diagonals on an 8x8 grid -/
def DiagonalConfiguration := Fin 8 → Fin 8 → Bool

/-- A point on the 8x8 grid -/
structure GridPoint where
  x : Fin 8
  y : Fin 8

/-- Two points are connected if they are in the same cell or adjacent cells with connecting diagonals -/
def connected (config : DiagonalConfiguration) (p1 p2 : GridPoint) : Prop :=
  sorry

/-- A connected component is a maximal set of connected points -/
def ConnectedComponent (config : DiagonalConfiguration) := Set GridPoint

/-- The number of connected components in a configuration -/
def numComponents (config : DiagonalConfiguration) : ℕ :=
  sorry

/-- There exists a configuration with more than 20 connected components -/
theorem exists_config_with_more_than_20_components :
  ∃ (config : DiagonalConfiguration), numComponents config > 20 :=
sorry

end exists_config_with_more_than_20_components_l2666_266655


namespace olympiad_scores_l2666_266614

theorem olympiad_scores (scores : Fin 20 → ℕ) 
  (distinct : ∀ i j, i ≠ j → scores i ≠ scores j)
  (sum_condition : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → scores i < scores j + scores k) :
  ∀ i, scores i > 18 := by
  sorry

end olympiad_scores_l2666_266614


namespace box_weights_l2666_266632

theorem box_weights (a b c : ℝ) 
  (hab : a + b = 122)
  (hbc : b + c = 125)
  (hca : c + a = 127) : 
  a + b + c = 187 := by
  sorry

end box_weights_l2666_266632


namespace salt_problem_l2666_266642

theorem salt_problem (a x : ℝ) (h : a - x = 2 * (a - 2 * x)) : x = a / 3 := by
  sorry

end salt_problem_l2666_266642


namespace not_always_possible_to_reduce_box_dimension_counterexample_exists_l2666_266678

/-- Represents a rectangular parallelepiped with dimensions length, width, and height -/
structure Parallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a box containing parallelepipeds -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  contents : List Parallelepiped

/-- Predicate to check if a parallelepiped fits in a box -/
def fits_in_box (p : Parallelepiped) (b : Box) : Prop :=
  p.length ≤ b.length ∧ p.width ≤ b.width ∧ p.height ≤ b.height

/-- Predicate to check if all parallelepipeds in a list fit in a box -/
def all_fit_in_box (ps : List Parallelepiped) (b : Box) : Prop :=
  ∀ p ∈ ps, fits_in_box p b

/-- Function to reduce one dimension of each parallelepiped -/
def reduce_parallelepipeds (ps : List Parallelepiped) : List Parallelepiped :=
  ps.map fun p => 
    let reduced_length := p.length * 0.99
    let reduced_width := p.width * 0.99
    let reduced_height := p.height * 0.99
    ⟨reduced_length, reduced_width, reduced_height⟩

/-- Theorem stating that it's not always possible to reduce a box dimension -/
theorem not_always_possible_to_reduce_box_dimension 
  (original_box : Box) 
  (reduced_parallelepipeds : List Parallelepiped) : Prop :=
  ∃ (reduced_box : Box), 
    (reduced_box.length < original_box.length ∨ 
     reduced_box.width < original_box.width ∨ 
     reduced_box.height < original_box.height) ∧
    all_fit_in_box reduced_parallelepipeds reduced_box →
    False

/-- Main theorem -/
theorem counterexample_exists : ∃ (original_box : Box) (original_parallelepipeds : List Parallelepiped),
  all_fit_in_box original_parallelepipeds original_box ∧
  not_always_possible_to_reduce_box_dimension original_box (reduce_parallelepipeds original_parallelepipeds) := by
  sorry

end not_always_possible_to_reduce_box_dimension_counterexample_exists_l2666_266678


namespace selling_to_buying_price_ratio_l2666_266651

theorem selling_to_buying_price_ratio 
  (natasha_money : ℕ) 
  (natasha_carla_ratio : ℕ) 
  (carla_cosima_ratio : ℕ) 
  (profit : ℕ) 
  (h1 : natasha_money = 60)
  (h2 : natasha_carla_ratio = 3)
  (h3 : carla_cosima_ratio = 2)
  (h4 : profit = 36) :
  let carla_money := natasha_money / natasha_carla_ratio
  let cosima_money := carla_money / carla_cosima_ratio
  let total_money := natasha_money + carla_money + cosima_money
  let selling_price := total_money + profit
  ∃ (a b : ℕ), a = 7 ∧ b = 5 ∧ selling_price * b = total_money * a :=
by sorry

end selling_to_buying_price_ratio_l2666_266651


namespace production_decrease_l2666_266600

/-- The number of cars originally planned for production -/
def original_plan : ℕ := 200

/-- The number of doors per car -/
def doors_per_car : ℕ := 5

/-- The total number of doors produced after reductions -/
def total_doors : ℕ := 375

/-- The reduction factor due to pandemic -/
def pandemic_reduction : ℚ := 1/2

theorem production_decrease (x : ℕ) : 
  (pandemic_reduction * (original_plan - x : ℚ)) * doors_per_car = total_doors → 
  x = 50 := by sorry

end production_decrease_l2666_266600


namespace birth_death_rate_interval_l2666_266636

/-- Prove that the time interval for birth and death rates is 2 seconds given the conditions --/
theorem birth_death_rate_interval (birth_rate death_rate net_increase_per_day : ℕ) 
  (h1 : birth_rate = 6)
  (h2 : death_rate = 2)
  (h3 : net_increase_per_day = 172800) :
  (24 * 60 * 60) / ((birth_rate - death_rate) * net_increase_per_day) = 2 := by
  sorry


end birth_death_rate_interval_l2666_266636


namespace q_is_false_l2666_266625

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q := by
  sorry

end q_is_false_l2666_266625


namespace min_value_expression_min_value_is_17_2_l2666_266637

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 1 → 
    a^2 + 4 * b^2 + 1 / (a * b) ≤ x^2 + 4 * y^2 + 1 / (x * y) :=
by sorry

theorem min_value_is_17_2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) :
  a^2 + 4 * b^2 + 1 / (a * b) = 17 / 2 :=
by sorry

end min_value_expression_min_value_is_17_2_l2666_266637


namespace wire_division_proof_l2666_266609

/-- Calculates the number of equal parts a wire can be divided into -/
def wire_parts (total_length : ℕ) (part_length : ℕ) : ℕ :=
  total_length / part_length

/-- Proves that a wire of 64 inches divided into 16-inch parts results in 4 parts -/
theorem wire_division_proof :
  wire_parts 64 16 = 4 := by
  sorry

end wire_division_proof_l2666_266609


namespace square_area_difference_l2666_266610

theorem square_area_difference (area_B : ℝ) (side_diff : ℝ) : 
  area_B = 81 → 
  side_diff = 4 → 
  let side_B := Real.sqrt area_B
  let side_A := side_B + side_diff
  side_A * side_A = 169 := by
  sorry

end square_area_difference_l2666_266610


namespace tom_seashell_collection_l2666_266648

/-- The number of days Tom spent at the beach -/
def days_at_beach : ℕ := 5

/-- The number of seashells Tom found each day -/
def seashells_per_day : ℕ := 7

/-- The total number of seashells Tom found during his beach trip -/
def total_seashells : ℕ := days_at_beach * seashells_per_day

theorem tom_seashell_collection :
  total_seashells = 35 :=
by sorry

end tom_seashell_collection_l2666_266648


namespace reeboks_sold_count_l2666_266663

def quota : ℕ := 1000
def adidas_price : ℕ := 45
def nike_price : ℕ := 60
def reebok_price : ℕ := 35
def nike_sold : ℕ := 8
def adidas_sold : ℕ := 6
def above_goal : ℕ := 65

theorem reeboks_sold_count :
  ∃ (reebok_sold : ℕ),
    reebok_sold * reebok_price + nike_sold * nike_price + adidas_sold * adidas_price = quota + above_goal ∧
    reebok_sold = 9 := by
  sorry

end reeboks_sold_count_l2666_266663


namespace johnson_calls_l2666_266641

def days_in_year : ℕ := 365

def call_frequencies : List ℕ := [2, 3, 6, 7]

/-- 
Calculates the number of days in a year where no calls are received, 
given a list of call frequencies (in days) for each grandchild.
-/
def days_without_calls (frequencies : List ℕ) (total_days : ℕ) : ℕ :=
  sorry

theorem johnson_calls : 
  days_without_calls call_frequencies days_in_year = 61 := by sorry

end johnson_calls_l2666_266641


namespace parabola_intersection_sum_l2666_266608

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  parabola_eq : y^2 = 4*x

/-- Theorem: For a parabola y^2 = 4x, if a line passing through its focus intersects 
    the parabola at points A and B, and the distance |AB| = 12, then x₁ + x₂ = 10 -/
theorem parabola_intersection_sum (A B : ParabolaPoint) 
  (focus_line : A.x ≠ B.x → (A.y - B.y) / (A.x - B.x) = (A.y + B.y) / (A.x + B.x - 2))
  (distance : (A.x - B.x)^2 + (A.y - B.y)^2 = 12^2) :
  A.x + B.x = 10 := by
  sorry

end parabola_intersection_sum_l2666_266608


namespace smallest_non_phd_count_l2666_266675

/-- The tournament structure -/
structure Tournament where
  total_participants : ℕ
  phd_participants : ℕ
  non_phd_participants : ℕ
  total_points : ℕ
  phd_points : ℕ
  non_phd_points : ℕ

/-- The theorem to prove -/
theorem smallest_non_phd_count (t : Tournament) : 
  199 ≤ t.total_participants ∧ 
  t.total_participants ≤ 229 ∧
  t.total_participants = t.phd_participants + t.non_phd_participants ∧
  t.total_points = t.total_participants * (t.total_participants - 1) / 2 ∧
  t.phd_points = t.phd_participants * (t.phd_participants - 1) / 2 ∧
  t.non_phd_points = t.non_phd_participants * (t.non_phd_participants - 1) / 2 ∧
  2 * (t.phd_points + t.non_phd_points) = t.total_points →
  t.non_phd_participants ≥ 105 ∧ 
  ∃ (t' : Tournament), t'.non_phd_participants = 105 ∧ 
    199 ≤ t'.total_participants ∧ 
    t'.total_participants ≤ 229 ∧
    t'.total_participants = t'.phd_participants + t'.non_phd_participants ∧
    t'.total_points = t'.total_participants * (t'.total_participants - 1) / 2 ∧
    t'.phd_points = t'.phd_participants * (t'.phd_participants - 1) / 2 ∧
    t'.non_phd_points = t'.non_phd_participants * (t'.non_phd_participants - 1) / 2 ∧
    2 * (t'.phd_points + t'.non_phd_points) = t'.total_points :=
by sorry

end smallest_non_phd_count_l2666_266675


namespace ball_bounces_on_table_l2666_266612

/-- Represents a rectangular table -/
structure Table where
  length : ℕ
  width : ℕ

/-- Calculates the number of bounces required for a ball to travel
    from one corner to the opposite corner of a rectangular table,
    moving at a 45° angle and bouncing off sides at 45° -/
def numberOfBounces (t : Table) : ℕ :=
  t.length + t.width - 2

theorem ball_bounces_on_table (t : Table) (h1 : t.length = 5) (h2 : t.width = 2) :
  numberOfBounces t = 5 := by
  sorry

#eval numberOfBounces { length := 5, width := 2 }

end ball_bounces_on_table_l2666_266612


namespace rectangle_area_l2666_266695

/-- The area of a rectangle with length 2 and width 4 is 8 -/
theorem rectangle_area : ∀ (length width area : ℝ), 
  length = 2 → width = 4 → area = length * width → area = 8 := by
  sorry

end rectangle_area_l2666_266695
