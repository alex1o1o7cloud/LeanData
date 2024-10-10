import Mathlib

namespace monroe_made_200_granola_bars_l682_68284

/-- The number of granola bars Monroe made -/
def total_granola_bars : ℕ := sorry

/-- The number of granola bars eaten by Monroe and her husband -/
def eaten_by_parents : ℕ := 80

/-- The number of children in Monroe's family -/
def number_of_children : ℕ := 6

/-- The number of granola bars each child received -/
def bars_per_child : ℕ := 20

/-- Theorem stating that Monroe made 200 granola bars -/
theorem monroe_made_200_granola_bars :
  total_granola_bars = eaten_by_parents + number_of_children * bars_per_child :=
sorry

end monroe_made_200_granola_bars_l682_68284


namespace roots_relation_l682_68268

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 4

-- Define the polynomial j(x)
def j (b c d x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

-- Theorem statement
theorem roots_relation (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0) →
  (∀ x : ℝ, h x = 0 → ∃ y : ℝ, j b c d y = 0 ∧ y = x^3) →
  b = -8 ∧ c = 36 ∧ d = -64 := by
sorry

end roots_relation_l682_68268


namespace triangle_side_length_l682_68253

/-- Given a triangle ABC with the following properties:
  - f(x) = 2sin(2x + π/6) + 1
  - f(A) = 2
  - b = 1
  - Area of triangle ABC is √3/2
  Prove that a = √3 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = 2 * Real.sin (2 * x + Real.pi / 6) + 1) →
  f A = 2 →
  b = 1 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 2 →
  a = Real.sqrt 3 := by 
  sorry

end triangle_side_length_l682_68253


namespace particle_in_semicircle_probability_l682_68264

theorem particle_in_semicircle_probability (AB BC : Real) (h1 : AB = 2) (h2 : BC = 1) :
  let rectangle_area := AB * BC
  let semicircle_radius := AB / 2
  let semicircle_area := π * semicircle_radius^2 / 2
  semicircle_area / rectangle_area = π / 4 := by sorry

end particle_in_semicircle_probability_l682_68264


namespace toys_after_game_purchase_l682_68291

theorem toys_after_game_purchase (initial_amount : ℕ) (game_cost : ℕ) (toy_cost : ℕ) : 
  initial_amount = 63 → game_cost = 48 → toy_cost = 3 → 
  (initial_amount - game_cost) / toy_cost = 5 := by
  sorry

end toys_after_game_purchase_l682_68291


namespace perfect_square_polynomial_l682_68298

theorem perfect_square_polynomial (n : ℤ) : 
  ∃ (m : ℤ), n^4 + 6*n^3 + 11*n^2 + 3*n + 31 = m^2 ↔ n = 10 := by
sorry

end perfect_square_polynomial_l682_68298


namespace curve_self_intersection_l682_68243

/-- The x-coordinate of a point on the curve for a given t -/
def x (t : ℝ) : ℝ := 2 * t^2 - 3

/-- The y-coordinate of a point on the curve for a given t -/
def y (t : ℝ) : ℝ := 2 * t^4 - 9 * t^2 + 6

/-- Theorem stating that (-1, -1) is a self-intersection point of the curve -/
theorem curve_self_intersection :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ x t₁ = x t₂ ∧ y t₁ = y t₂ ∧ x t₁ = -1 ∧ y t₁ = -1 := by
  sorry

end curve_self_intersection_l682_68243


namespace complex_subtraction_l682_68254

theorem complex_subtraction (a b : ℂ) (h1 : a = 5 + I) (h2 : b = 2 - 3 * I) :
  a - 3 * b = 11 - 8 * I := by
  sorry

end complex_subtraction_l682_68254


namespace sum_ten_consecutive_naturals_odd_l682_68266

theorem sum_ten_consecutive_naturals_odd (n : ℕ) : ∃ k : ℕ, (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8) + (n + 9)) = 2 * k + 1 := by
  sorry

end sum_ten_consecutive_naturals_odd_l682_68266


namespace equal_number_of_boys_and_girls_l682_68228

/-- Represents a school with boys and girls -/
structure School where
  boys : ℕ
  girls : ℕ
  boys_age_sum : ℕ
  girls_age_sum : ℕ

/-- The average age of boys -/
def boys_avg (s : School) : ℚ := s.boys_age_sum / s.boys

/-- The average age of girls -/
def girls_avg (s : School) : ℚ := s.girls_age_sum / s.girls

/-- The average age of all students -/
def total_avg (s : School) : ℚ := (s.boys_age_sum + s.girls_age_sum) / (s.boys + s.girls)

/-- The theorem stating that the number of boys equals the number of girls -/
theorem equal_number_of_boys_and_girls (s : School) 
  (h1 : boys_avg s ≠ girls_avg s) 
  (h2 : (boys_avg s + girls_avg s) / 2 = total_avg s) : 
  s.boys = s.girls := by sorry

end equal_number_of_boys_and_girls_l682_68228


namespace real_part_of_complex_number_l682_68292

theorem real_part_of_complex_number (z : ℂ) : z = (Complex.I^3) / (1 + 2 * Complex.I) → Complex.re z = -2/5 := by
  sorry

end real_part_of_complex_number_l682_68292


namespace triangle_radius_equality_l682_68250

/-- For a triangle with sides a, b, c, circumradius R, inradius r, and semi-perimeter p,
    prove that ab + bc + ac = r² + p² + 4Rr -/
theorem triangle_radius_equality (a b c R r p : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ r > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_semi_perimeter : p = (a + b + c) / 2)
  (h_circumradius : R = (a * b * c) / (4 * (p * (p - a) * (p - b) * (p - c))^(1/2)))
  (h_inradius : r = (p * (p - a) * (p - b) * (p - c))^(1/2) / p) :
  a * b + b * c + a * c = r^2 + p^2 + 4 * R * r := by
sorry

end triangle_radius_equality_l682_68250


namespace arithmetic_sequence_n_l682_68275

/-- Given an arithmetic sequence {a_n} with a_1 = 20, a_n = 54, and S_n = 999, prove that n = 27 -/
theorem arithmetic_sequence_n (a : ℕ → ℝ) (n : ℕ) (S_n : ℝ) : 
  (∀ k, a (k + 1) - a k = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 20 →
  a n = 54 →
  S_n = 999 →
  n = 27 := by
sorry

end arithmetic_sequence_n_l682_68275


namespace ball_probabilities_l682_68265

/-- The number of red balls in the bag -/
def num_red_balls : ℕ := 4

/-- The number of white balls in the bag -/
def num_white_balls : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red_balls + num_white_balls

/-- The probability of picking a red ball in a single draw -/
def prob_red : ℚ := num_red_balls / total_balls

/-- The probability of picking a white ball in a single draw -/
def prob_white : ℚ := num_white_balls / total_balls

theorem ball_probabilities :
  -- Statement A
  (Nat.choose 2 1 * Nat.choose 4 2 : ℚ) / Nat.choose 6 3 = 3 / 5 ∧
  -- Statement B
  (6 : ℚ) * prob_red * (1 - prob_red) = 4 / 3 ∧
  -- Statement C
  (4 : ℚ) / 6 * 3 / 5 = 2 / 5 ∧
  -- Statement D
  1 - (1 - prob_red) ^ 3 = 26 / 27 := by
  sorry

end ball_probabilities_l682_68265


namespace triangle_isosceles_l682_68285

theorem triangle_isosceles (A B C : ℝ) (h : 2 * Real.sin A * Real.cos B = Real.sin C) : A = B :=
sorry

end triangle_isosceles_l682_68285


namespace candy_bar_difference_l682_68201

theorem candy_bar_difference (lena kevin nicole : ℕ) : 
  lena = 16 → 
  lena + 5 = 3 * kevin → 
  nicole = kevin + 4 → 
  lena - nicole = 5 := by
sorry

end candy_bar_difference_l682_68201


namespace nilpotent_matrix_cube_zero_l682_68260

/-- Given a 2x2 matrix B with real entries such that B^4 = 0, prove that B^3 = 0 -/
theorem nilpotent_matrix_cube_zero (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : B ^ 3 = 0 := by
  sorry

end nilpotent_matrix_cube_zero_l682_68260


namespace lovers_watches_prime_sum_squares_l682_68263

theorem lovers_watches_prime_sum_squares :
  ∃ (x y : Fin 12 → ℕ) (m : Fin 12 → ℕ),
    (∀ i : Fin 12, Nat.Prime (x i)) ∧
    (∀ i : Fin 12, Nat.Prime (y i)) ∧
    (∀ i j : Fin 12, i ≠ j → x i ≠ x j) ∧
    (∀ i j : Fin 12, i ≠ j → y i ≠ y j) ∧
    (∀ i : Fin 12, x i ≠ y i) ∧
    (∀ k : Fin 12, x k + x (k.succ) = y k + y (k.succ)) ∧
    (∀ k : Fin 12, ∃ (m_k : ℕ), x k + x (k.succ) = m_k ^ 2) :=
by sorry


end lovers_watches_prime_sum_squares_l682_68263


namespace total_ebook_readers_l682_68294

/-- The number of eBook readers Anna bought -/
def anna_readers : ℕ := 50

/-- The number of eBook readers John bought initially -/
def john_initial_readers : ℕ := anna_readers - 15

/-- The number of eBook readers John lost -/
def john_lost_readers : ℕ := 3

/-- The number of eBook readers John has after losing some -/
def john_final_readers : ℕ := john_initial_readers - john_lost_readers

/-- The total number of eBook readers John and Anna have together -/
def total_readers : ℕ := anna_readers + john_final_readers

theorem total_ebook_readers :
  total_readers = 82 :=
by sorry

end total_ebook_readers_l682_68294


namespace triangle_area_l682_68289

/-- Given a triangle ABC with side lengths AB = 1 and BC = 3, and the dot product of vectors AB and BC equal to -1, 
    prove that the area of the triangle is √2. -/
theorem triangle_area (A B C : ℝ × ℝ) : 
  let AB := ((B.1 - A.1), (B.2 - A.2))
  let BC := ((C.1 - B.1), (C.2 - B.2))
  (AB.1^2 + AB.2^2 = 1) →
  (BC.1^2 + BC.2^2 = 9) →
  (AB.1 * BC.1 + AB.2 * BC.2 = -1) →
  (abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)) / 2 = Real.sqrt 2) :=
by sorry


end triangle_area_l682_68289


namespace max_table_sum_l682_68251

def primes : List Nat := [2, 3, 5, 7, 17, 19]

def is_valid_arrangement (top bottom : List Nat) : Prop :=
  top.length = 3 ∧ bottom.length = 3 ∧ 
  (∀ x ∈ top, x ∈ primes) ∧ (∀ x ∈ bottom, x ∈ primes) ∧
  (∀ x ∈ top, x ∉ bottom) ∧ (∀ x ∈ bottom, x ∉ top)

def table_sum (top bottom : List Nat) : Nat :=
  (top.sum * bottom.sum)

theorem max_table_sum :
  ∀ top bottom : List Nat,
  is_valid_arrangement top bottom →
  table_sum top bottom ≤ 682 :=
sorry

end max_table_sum_l682_68251


namespace line_plane_perpendicular_parallel_l682_68202

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (planePerp : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular_parallel 
  (m n : Line) (α β γ : Plane)
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : α ≠ γ)
  (h4 : β ≠ γ)
  (h5 : perpendicular m β)
  (h6 : parallel m α) :
  planePerp α β :=
sorry

end line_plane_perpendicular_parallel_l682_68202


namespace sqrt_equation_solution_l682_68213

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (x + 8) = 10 → x = 92 := by
  sorry

end sqrt_equation_solution_l682_68213


namespace sequence_periodicity_l682_68235

theorem sequence_periodicity (a : ℕ → ℚ) (h1 : ∀ n : ℕ, |a (n + 1) - 2 * a n| = 2)
    (h2 : ∀ n : ℕ, |a n| ≤ 2) :
  ∃ k l : ℕ, k < l ∧ a k = a l :=
sorry

end sequence_periodicity_l682_68235


namespace max_x_value_l682_68247

theorem max_x_value (x : ℝ) : 
  (((5*x - 20) / (4*x - 5))^2 + ((5*x - 20) / (4*x - 5)) = 20) → 
  x ≤ 9/5 :=
by sorry

end max_x_value_l682_68247


namespace inverse_function_b_value_l682_68257

theorem inverse_function_b_value (f : ℝ → ℝ) (b : ℝ) :
  (∀ x, f x = 5 - b * x) →
  (∃ g : ℝ → ℝ, Function.LeftInverse g f ∧ Function.RightInverse g f ∧ g (-3) = 3) →
  b = 8 / 3 := by
sorry

end inverse_function_b_value_l682_68257


namespace upstream_speed_l682_68238

theorem upstream_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still = 45) 
  (h2 : speed_downstream = 53) : 
  speed_still - (speed_downstream - speed_still) = 37 := by
  sorry

end upstream_speed_l682_68238


namespace union_M_complement_N_equals_R_l682_68290

-- Define the sets M and N
def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | x^2 - x < 0}

-- State the theorem
theorem union_M_complement_N_equals_R : M ∪ (Set.univ \ N) = Set.univ := by sorry

end union_M_complement_N_equals_R_l682_68290


namespace octagon_area_given_equal_perimeter_and_square_area_l682_68258

/-- Given a square and a regular octagon with equal perimeters, 
    if the area of the square is 16, then the area of the octagon is 8 + 4√2 -/
theorem octagon_area_given_equal_perimeter_and_square_area (a b : ℝ) : 
  a > 0 → b > 0 → 4 * a = 8 * b → a^2 = 16 → 
  2 * (1 + Real.sqrt 2) * b^2 = 8 + 4 * Real.sqrt 2 := by
  sorry

end octagon_area_given_equal_perimeter_and_square_area_l682_68258


namespace school_student_count_l682_68273

/-- The number of classrooms in the school -/
def num_classrooms : ℕ := 24

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 5

/-- The total number of students in the school -/
def total_students : ℕ := num_classrooms * students_per_classroom

theorem school_student_count : total_students = 120 := by
  sorry

end school_student_count_l682_68273


namespace parameterization_validity_l682_68288

/-- The slope of the line -/
def m : ℚ := 7/4

/-- The y-intercept of the line -/
def b : ℚ := -14/4

/-- The line equation -/
def line_eq (x y : ℚ) : Prop := y = m * x + b

/-- Vector parameterization A -/
def param_A (t : ℚ) : ℚ × ℚ := (3 + 4*t, -5/4 + 7*t)

/-- Vector parameterization B -/
def param_B (t : ℚ) : ℚ × ℚ := (7 + 8*t, 7/4 + 14*t)

/-- Vector parameterization C -/
def param_C (t : ℚ) : ℚ × ℚ := (2 + 14*t, 1/2 + 7*t)

/-- Vector parameterization D -/
def param_D (t : ℚ) : ℚ × ℚ := (-1 + 8*t, -27/4 - 15*t)

/-- Vector parameterization E -/
def param_E (t : ℚ) : ℚ × ℚ := (4 - 7*t, 9/2 + 5*t)

theorem parameterization_validity :
  (∀ t, line_eq (param_A t).1 (param_A t).2) ∧
  (∀ t, line_eq (param_B t).1 (param_B t).2) ∧
  ¬(∀ t, line_eq (param_C t).1 (param_C t).2) ∧
  ¬(∀ t, line_eq (param_D t).1 (param_D t).2) ∧
  ¬(∀ t, line_eq (param_E t).1 (param_E t).2) :=
by sorry

end parameterization_validity_l682_68288


namespace average_weight_increase_l682_68206

/-- Proves that replacing a person weighing 45 kg with a person weighing 93 kg
    in a group of 8 people increases the average weight by 6 kg. -/
theorem average_weight_increase (initial_average : ℝ) :
  let group_size : ℕ := 8
  let old_weight : ℝ := 45
  let new_weight : ℝ := 93
  let weight_difference : ℝ := new_weight - old_weight
  let average_increase : ℝ := weight_difference / group_size
  average_increase = 6 := by
  sorry

#check average_weight_increase

end average_weight_increase_l682_68206


namespace seating_arrangements_l682_68225

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem seating_arrangements (n : ℕ) (k : ℕ) (h : n = 10 ∧ k = 3) :
  factorial n - factorial (n - k + 1) * factorial k = 3598560 :=
sorry

end seating_arrangements_l682_68225


namespace expression_simplification_l682_68229

theorem expression_simplification (x y : ℝ) (h : 2 * x + y - 3 = 0) :
  ((3 * x) / (x - y) + x / (x + y)) / (x / (x^2 - y^2)) = 6 :=
by sorry

end expression_simplification_l682_68229


namespace lindys_speed_l682_68299

/-- Proves that Lindy's speed is 9 feet per second given the problem conditions --/
theorem lindys_speed (initial_distance : ℝ) (jack_speed christina_speed : ℝ) 
  (lindy_distance : ℝ) : ℝ :=
by
  -- Define the given conditions
  have h1 : initial_distance = 240 := by sorry
  have h2 : jack_speed = 5 := by sorry
  have h3 : christina_speed = 3 := by sorry
  have h4 : lindy_distance = 270 := by sorry

  -- Calculate the time it takes for Jack and Christina to meet
  let total_speed := jack_speed + christina_speed
  let time_to_meet := initial_distance / total_speed

  -- Calculate Lindy's speed
  let lindy_speed := lindy_distance / time_to_meet

  -- Prove that Lindy's speed is 9 feet per second
  have h5 : lindy_speed = 9 := by sorry

  exact lindy_speed

end lindys_speed_l682_68299


namespace least_sum_of_bases_l682_68272

theorem least_sum_of_bases (a b : ℕ+) : 
  (6 * a.val + 3 = 3 * b.val + 6) →
  (∀ (a' b' : ℕ+), (6 * a'.val + 3 = 3 * b'.val + 6) → (a'.val + b'.val ≥ a.val + b.val)) →
  a.val + b.val = 20 :=
sorry

end least_sum_of_bases_l682_68272


namespace min_sum_distances_on_BC_l682_68212

/-- Four distinct points on a line -/
structure FourPoints where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  ordered : A < B ∧ B < C ∧ C < D

/-- Sum of distances from a point X to A, B, C, and D -/
def sumOfDistances (fp : FourPoints) (X : ℝ) : ℝ :=
  |X - fp.A| + |X - fp.B| + |X - fp.C| + |X - fp.D|

/-- The point that minimizes the sum of distances is on the segment BC -/
theorem min_sum_distances_on_BC (fp : FourPoints) :
  ∃ (X : ℝ), fp.B ≤ X ∧ X ≤ fp.C ∧
  ∀ (Y : ℝ), sumOfDistances fp X ≤ sumOfDistances fp Y :=
sorry

end min_sum_distances_on_BC_l682_68212


namespace factorization_problem_1_factorization_problem_2_l682_68274

-- Problem 1
theorem factorization_problem_1 (x : ℝ) :
  2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) :
  (x - y)^3 - 16 * (x - y) = (x - y) * (x - y + 4) * (x - y - 4) := by sorry

end factorization_problem_1_factorization_problem_2_l682_68274


namespace given_expression_is_proper_algebraic_notation_l682_68230

/-- A predicate that determines if an expression meets the requirements for algebraic notation -/
def is_proper_algebraic_notation (expression : String) : Prop := 
  expression = "(3πm)/4"

/-- The given expression -/
def given_expression : String := "(3πm)/4"

/-- Theorem stating that the given expression meets the requirements for algebraic notation -/
theorem given_expression_is_proper_algebraic_notation : 
  is_proper_algebraic_notation given_expression := by
  sorry

end given_expression_is_proper_algebraic_notation_l682_68230


namespace race_length_is_1000_l682_68208

/-- The length of a race given Aubrey's and Violet's positions -/
def race_length (violet_distance_covered : ℕ) (violet_distance_to_finish : ℕ) : ℕ :=
  violet_distance_covered + violet_distance_to_finish

/-- Theorem stating that the race length is 1000 meters -/
theorem race_length_is_1000 :
  race_length 721 279 = 1000 := by
  sorry

end race_length_is_1000_l682_68208


namespace gwen_math_problems_l682_68280

theorem gwen_math_problems 
  (science_problems : ℕ) 
  (finished_problems : ℕ) 
  (remaining_problems : ℕ) 
  (h1 : science_problems = 11)
  (h2 : finished_problems = 24)
  (h3 : remaining_problems = 5)
  (h4 : finished_problems + remaining_problems = science_problems + math_problems) :
  math_problems = 18 :=
by
  sorry

end gwen_math_problems_l682_68280


namespace expression_simplification_l682_68295

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 - 3) :
  (x - 3) / (x - 2) / (x + 2 - 5 / (x - 2)) = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l682_68295


namespace min_colors_theorem_l682_68226

/-- A coloring of positive integers -/
def Coloring (k : ℕ) := ℕ+ → Fin k

/-- A function from positive integers to positive integers -/
def IntFunction := ℕ+ → ℕ+

/-- The property that f(m+n) = f(m) + f(n) for integers of the same color -/
def SameColorAdditive (f : IntFunction) (c : Coloring k) : Prop :=
  ∀ m n : ℕ+, c m = c n → f (m + n) = f m + f n

/-- The property that there exist m and n such that f(m+n) ≠ f(m) + f(n) -/
def ExistsDifferentSum (f : IntFunction) : Prop :=
  ∃ m n : ℕ+, f (m + n) ≠ f m + f n

/-- The main theorem -/
theorem min_colors_theorem :
  (∃ k : ℕ+, ∃ c : Coloring k, ∃ f : IntFunction,
    SameColorAdditive f c ∧ ExistsDifferentSum f) ∧
  (∀ k : ℕ+, k < 3 → ¬∃ c : Coloring k, ∃ f : IntFunction,
    SameColorAdditive f c ∧ ExistsDifferentSum f) :=
sorry

end min_colors_theorem_l682_68226


namespace van_helsing_earnings_l682_68231

/-- Van Helsing's vampire and werewolf removal earnings problem -/
theorem van_helsing_earnings : ∀ (v w : ℕ),
  w = 4 * v →  -- There were 4 times as many werewolves as vampires
  w = 8 →      -- 8 werewolves were removed
  5 * (v / 2) + 10 * 8 = 85  -- Total earnings calculation
  := by sorry

end van_helsing_earnings_l682_68231


namespace m_intersect_n_eq_m_l682_68282

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = 2^x}
def N : Set ℝ := {y | ∃ x : ℝ, y = x^2}

-- State the theorem
theorem m_intersect_n_eq_m : M ∩ N = M := by sorry

end m_intersect_n_eq_m_l682_68282


namespace train_speed_equation_l682_68209

theorem train_speed_equation (x : ℝ) (h : x > 0) : 
  700 / x - 700 / (2.8 * x) = 3.6 ↔ 
  (∃ (t_express t_highspeed : ℝ),
    t_express = 700 / x ∧
    t_highspeed = 700 / (2.8 * x) ∧
    t_express - t_highspeed = 3.6) :=
by sorry

end train_speed_equation_l682_68209


namespace fractional_equation_solution_l682_68269

theorem fractional_equation_solution :
  ∃! x : ℝ, x ≠ 3 ∧ (1 - x) / (x - 3) = 1 / (3 - x) - 2 := by
  sorry

end fractional_equation_solution_l682_68269


namespace square_min_rotation_l682_68237

/-- A square is a geometric shape with four equal sides and four right angles. -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- The minimum rotation angle for a square to coincide with itself. -/
def minRotationAngle (s : Square) : ℝ := 90

/-- Theorem stating that the minimum rotation angle for a square to coincide with itself is 90 degrees. -/
theorem square_min_rotation (s : Square) : minRotationAngle s = 90 := by
  sorry

end square_min_rotation_l682_68237


namespace power_equation_solution_l682_68255

theorem power_equation_solution : 2^5 - 7 = 3^3 + (-2) := by sorry

end power_equation_solution_l682_68255


namespace arithmetic_sequence_common_difference_l682_68262

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The common difference of an arithmetic sequence. -/
def common_difference (a : ℕ → ℝ) : ℝ :=
  a 2 - a 1

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum1 : a 3 + a 6 = 11)
  (h_sum2 : a 5 + a 8 = 39) :
  common_difference a = 7 := by
sorry

end arithmetic_sequence_common_difference_l682_68262


namespace train_speed_l682_68219

theorem train_speed (length time : ℝ) (h1 : length = 300) (h2 : time = 15) :
  length / time = 20 := by
  sorry

end train_speed_l682_68219


namespace greatest_integer_fraction_inequality_l682_68232

theorem greatest_integer_fraction_inequality : 
  (∀ y : ℤ, (8 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 12) :=
by sorry

end greatest_integer_fraction_inequality_l682_68232


namespace inequality_solution_set_l682_68277

theorem inequality_solution_set : 
  {x : ℝ | x^2 - x - 6 < 0} = Set.Ioo (-2) 3 := by sorry

end inequality_solution_set_l682_68277


namespace intersection_length_l682_68281

/-- The curve C in the Cartesian plane -/
def C (x y : ℝ) : Prop := x^2 + y^2/4 = 1

/-- The line l passing through (0, 1) -/
def l (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

/-- Point A on the intersection of C and l -/
def A (k x₁ y₁ : ℝ) : Prop := C x₁ y₁ ∧ l k x₁ y₁

/-- Point B on the intersection of C and l -/
def B (k x₂ y₂ : ℝ) : Prop := C x₂ y₂ ∧ l k x₂ y₂

/-- The condition that OA · AB = 0 -/
def orthogonal (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

/-- The main theorem -/
theorem intersection_length 
  (k x₁ y₁ x₂ y₂ : ℝ) 
  (hA : A k x₁ y₁) 
  (hB : B k x₂ y₂) 
  (hO : orthogonal x₁ y₁ x₂ y₂) : 
  ((x₂ - x₁)^2 + (y₂ - y₁)^2) = (4*Real.sqrt 65/17)^2 := by
  sorry


end intersection_length_l682_68281


namespace craig_seashells_l682_68218

theorem craig_seashells : ∃ (c : ℕ), c = 54 ∧ c > 0 ∧ ∃ (b : ℕ), 
  (c : ℚ) / (b : ℚ) = 9 / 7 ∧ b = c - 12 := by
  sorry

end craig_seashells_l682_68218


namespace annual_income_calculation_l682_68214

theorem annual_income_calculation (total : ℝ) (p1 : ℝ) (rate1 : ℝ) (rate2 : ℝ)
  (h1 : total = 2500)
  (h2 : p1 = 500.0000000000002)
  (h3 : rate1 = 0.05)
  (h4 : rate2 = 0.06) :
  let p2 := total - p1
  let income1 := p1 * rate1
  let income2 := p2 * rate2
  income1 + income2 = 145 := by sorry

end annual_income_calculation_l682_68214


namespace fraction_of_silver_knights_with_shields_l682_68207

theorem fraction_of_silver_knights_with_shields :
  ∀ (total_knights : ℕ) (silver_knights : ℕ) (golden_knights : ℕ) (knights_with_shields : ℕ)
    (silver_knights_with_shields : ℕ) (golden_knights_with_shields : ℕ),
  total_knights > 0 →
  silver_knights + golden_knights = total_knights →
  silver_knights = (3 * total_knights) / 8 →
  knights_with_shields = total_knights / 4 →
  silver_knights_with_shields + golden_knights_with_shields = knights_with_shields →
  silver_knights_with_shields * golden_knights = 3 * golden_knights_with_shields * silver_knights →
  silver_knights_with_shields * 7 = silver_knights * 3 :=
by sorry

end fraction_of_silver_knights_with_shields_l682_68207


namespace all_parameterizations_valid_l682_68222

def line_equation (x y : ℝ) : Prop := y = 2 * x - 4

def valid_parameterization (p : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, line_equation (p.1 + t * v.1) (p.2 + t * v.2)

theorem all_parameterizations_valid :
  valid_parameterization (3, -2) (1, 2) ∧
  valid_parameterization (4, 0) (2, 4) ∧
  valid_parameterization (0, -4) (1, 2) ∧
  valid_parameterization (1, -1) (0.5, 1) ∧
  valid_parameterization (-1, -6) (-2, -4) :=
sorry

end all_parameterizations_valid_l682_68222


namespace base_n_representation_of_d_l682_68245

/-- Represents a number in base n --/
structure BaseN (n : ℕ) where
  digits : List ℕ
  all_less : ∀ d ∈ digits, d < n

/-- Convert a base-n number to its decimal representation --/
def toDecimal (n : ℕ) (b : BaseN n) : ℕ :=
  b.digits.enum.foldl (fun acc (i, d) => acc + d * n ^ i) 0

theorem base_n_representation_of_d (n : ℕ) (c d : ℕ) :
  n > 8 →
  n ^ 2 - c * n + d = 0 →
  toDecimal n ⟨[2, 1], by sorry⟩ = c →
  toDecimal n ⟨[0, 1, 1], by sorry⟩ = d :=
by sorry

end base_n_representation_of_d_l682_68245


namespace seating_arrangements_count_l682_68256

-- Define the number of sibling pairs
def num_sibling_pairs : ℕ := 4

-- Define the number of seats in each row
def seats_per_row : ℕ := 4

-- Define the number of rows in the van
def num_rows : ℕ := 2

-- Define the derangement function for 4 objects
def derangement_4 : ℕ := 9

-- Theorem statement
theorem seating_arrangements_count :
  (seats_per_row.factorial) * derangement_4 * (2^num_sibling_pairs) = 3456 := by
  sorry


end seating_arrangements_count_l682_68256


namespace cookfire_logs_added_l682_68217

/-- The number of logs added to a cookfire each hour, given the initial number of logs,
    burn rate, duration, and final number of logs. -/
def logsAddedPerHour (initialLogs burnRate duration finalLogs : ℕ) : ℕ :=
  let logsAfterBurning := initialLogs - burnRate * duration
  (finalLogs - logsAfterBurning + burnRate * (duration - 1)) / duration

theorem cookfire_logs_added (x : ℕ) :
  logsAddedPerHour 6 3 3 3 = 2 :=
sorry

end cookfire_logs_added_l682_68217


namespace terminating_decimal_thirteen_over_sixtwentyfive_l682_68233

theorem terminating_decimal_thirteen_over_sixtwentyfive :
  (13 : ℚ) / 625 = (208 : ℚ) / 10000 :=
by sorry

end terminating_decimal_thirteen_over_sixtwentyfive_l682_68233


namespace absolute_value_inequality_solution_set_l682_68248

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 1| > 1} = Set.Ioi (-2) ∪ Set.Ioi 0 :=
by sorry

end absolute_value_inequality_solution_set_l682_68248


namespace simplify_tan_product_l682_68270

theorem simplify_tan_product : (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end simplify_tan_product_l682_68270


namespace probability_alice_has_ball_after_three_turns_l682_68297

-- Define the probabilities
def alice_pass : ℚ := 1/3
def alice_keep : ℚ := 2/3
def bob_pass : ℚ := 1/4
def bob_keep : ℚ := 3/4

-- Define the game state after three turns
def alice_has_ball_after_three_turns : ℚ :=
  alice_keep^3 + alice_pass * bob_pass * alice_keep + alice_keep * alice_pass * bob_pass

-- Theorem statement
theorem probability_alice_has_ball_after_three_turns :
  alice_has_ball_after_three_turns = 11/27 := by sorry

end probability_alice_has_ball_after_three_turns_l682_68297


namespace imaginary_part_of_z_l682_68279

theorem imaginary_part_of_z (m : ℝ) :
  let z := (2 + m * Complex.I) / (1 + Complex.I)
  (z.re = 0) → z.im = -2 := by
sorry

end imaginary_part_of_z_l682_68279


namespace circle_condition_implies_m_range_necessary_but_not_sufficient_condition_implies_a_range_l682_68210

-- Define the equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*m*x + 5*m^2 + m - 2 = 0

-- Define the condition for being a circle
def is_circle (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_equation x y m

-- Define the inequality condition
def inequality_condition (m a : ℝ) : Prop :=
  (m - a) * (m - a - 4) < 0

theorem circle_condition_implies_m_range :
  (∀ m : ℝ, is_circle m → m > -2 ∧ m < 1) :=
sorry

theorem necessary_but_not_sufficient_condition_implies_a_range :
  (∀ a : ℝ, (∀ m : ℝ, inequality_condition m a → is_circle m) ∧
            (∃ m : ℝ, is_circle m ∧ ¬inequality_condition m a) →
   a ≥ -3 ∧ a ≤ -2) :=
sorry

end circle_condition_implies_m_range_necessary_but_not_sufficient_condition_implies_a_range_l682_68210


namespace position_after_2004_seconds_l682_68221

/-- Represents the position of the particle -/
structure Position :=
  (x : ℕ) (y : ℕ)

/-- Defines the movement pattern of the particle -/
def nextPosition (p : Position) : Position :=
  if p.x = p.y then Position.mk (p.x + 1) p.y
  else if p.x > p.y then Position.mk p.x (p.y + 1)
  else Position.mk p.x (p.y - 1)

/-- Calculates the position after n seconds -/
def positionAfterSeconds (n : ℕ) : Position :=
  match n with
  | 0 => Position.mk 0 0
  | 1 => Position.mk 0 1
  | n + 2 => nextPosition (positionAfterSeconds (n + 1))

/-- The main theorem stating the position after 2004 seconds -/
theorem position_after_2004_seconds :
  positionAfterSeconds 2004 = Position.mk 20 44 := by
  sorry


end position_after_2004_seconds_l682_68221


namespace dormitory_to_city_distance_l682_68224

theorem dormitory_to_city_distance :
  ∀ (D : ℝ),
  (1/3 : ℝ) * D + (3/5 : ℝ) * D + 2 = D →
  D = 30 := by
sorry

end dormitory_to_city_distance_l682_68224


namespace vector_collinearity_l682_68296

def a (k : ℝ) : Fin 2 → ℝ := ![1, k]
def b : Fin 2 → ℝ := ![2, 2]

theorem vector_collinearity (k : ℝ) :
  (∀ (i : Fin 2), (a k + b) i = (a k) i * (3 : ℝ)) → k = 1 := by
  sorry

end vector_collinearity_l682_68296


namespace trophy_cost_l682_68242

theorem trophy_cost (x y : ℕ) (hx : x < 10) (hy : y < 10) :
  let total_cents : ℕ := 1000 * x + 9990 + y
  (72 ∣ total_cents) →
  (total_cents : ℚ) / (72 * 100) = 11.11 := by
sorry

end trophy_cost_l682_68242


namespace circle_radii_order_l682_68236

/-- Given three circles A, B, and C with the following properties:
    - Circle A has a circumference of 6π
    - Circle B has an area of 16π
    - Circle C has a radius of 2
    Prove that the radii of the circles are ordered as r_C < r_A < r_B -/
theorem circle_radii_order (r_A r_B r_C : ℝ) : 
  (2 * π * r_A = 6 * π) →  -- Circumference of A
  (π * r_B^2 = 16 * π) →   -- Area of B
  (r_C = 2) →              -- Radius of C
  r_C < r_A ∧ r_A < r_B := by
sorry

end circle_radii_order_l682_68236


namespace remainder_theorem_l682_68239

-- Define the polynomial q(x)
def q (D E F : ℝ) (x : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 8

-- State the theorem
theorem remainder_theorem (D E F : ℝ) :
  (∃ p : ℝ → ℝ, ∀ x, q D E F x = (x - 2) * p x + 12) →
  (∃ p : ℝ → ℝ, ∀ x, q D E F x = (x + 2) * p x + 4) :=
by sorry

end remainder_theorem_l682_68239


namespace largest_three_digit_base_7_is_342_l682_68241

/-- The largest decimal number represented by a three-digit base-7 number -/
def largest_three_digit_base_7 : ℕ := 342

/-- The base of the number system -/
def base : ℕ := 7

/-- The number of digits -/
def num_digits : ℕ := 3

/-- Theorem: The largest decimal number represented by a three-digit base-7 number is 342 -/
theorem largest_three_digit_base_7_is_342 :
  largest_three_digit_base_7 = (base ^ num_digits - 1) := by sorry

end largest_three_digit_base_7_is_342_l682_68241


namespace increasing_quadratic_implies_a_ge_five_l682_68293

/-- Given a function f(x) = -x^2 + 2(a-1)x + 2 that is increasing on the interval (-∞, 4),
    prove that a ≥ 5. -/
theorem increasing_quadratic_implies_a_ge_five (a : ℝ) :
  (∀ x < 4, Monotone (fun x => -x^2 + 2*(a-1)*x + 2)) →
  a ≥ 5 := by
  sorry

end increasing_quadratic_implies_a_ge_five_l682_68293


namespace mapping_preimage_property_l682_68246

theorem mapping_preimage_property (A B : Type) (f : A → B) :
  ∃ (b : B), ∃ (a1 a2 : A), a1 ≠ a2 ∧ f a1 = b ∧ f a2 = b :=
sorry

end mapping_preimage_property_l682_68246


namespace stating_dodgeball_tournament_teams_l682_68200

/-- Represents the total points scored in a dodgeball tournament. -/
def total_points : ℕ := 1151

/-- Points awarded for a win in the tournament. -/
def win_points : ℕ := 15

/-- Points awarded for a tie in the tournament. -/
def tie_points : ℕ := 11

/-- Points awarded for a loss in the tournament. -/
def loss_points : ℕ := 0

/-- The number of teams in the tournament. -/
def num_teams : ℕ := 12

/-- 
Theorem stating that given the tournament conditions, 
the number of teams must be 12.
-/
theorem dodgeball_tournament_teams : 
  ∀ n : ℕ, 
    (n * (n - 1) / 2) * win_points ≤ total_points ∧ 
    total_points ≤ (n * (n - 1) / 2) * (win_points + tie_points) / 2 →
    n = num_teams :=
by sorry

end stating_dodgeball_tournament_teams_l682_68200


namespace fifth_term_of_sequence_l682_68205

def geometric_sequence (a₀ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₀ * r^n

theorem fifth_term_of_sequence (x : ℝ) :
  let a₀ := 3
  let r := 3 * x^2
  geometric_sequence a₀ r 4 = 243 * x^8 :=
by sorry

end fifth_term_of_sequence_l682_68205


namespace binomial_coefficient_two_l682_68215

theorem binomial_coefficient_two (n : ℕ+) : Nat.choose n.val 2 = n.val * (n.val - 1) / 2 := by
  sorry

end binomial_coefficient_two_l682_68215


namespace smallest_three_digit_multiple_of_13_l682_68283

theorem smallest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 104 → ¬(∃ k : ℕ, n = 13 * k) :=
by
  sorry

#check smallest_three_digit_multiple_of_13

end smallest_three_digit_multiple_of_13_l682_68283


namespace james_total_matches_l682_68249

/-- The number of boxes in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of boxes James has -/
def james_dozens : ℕ := 5

/-- The number of matches in each box -/
def matches_per_box : ℕ := 20

/-- Theorem: James has 1200 matches in total -/
theorem james_total_matches :
  james_dozens * dozen * matches_per_box = 1200 := by
  sorry

end james_total_matches_l682_68249


namespace fisher_min_score_l682_68211

/-- Represents a student's scores and eligibility requirements -/
structure StudentScores where
  algebra_sem1 : ℝ
  algebra_sem2 : ℝ
  statistics : ℝ
  algebra_required_avg : ℝ
  statistics_required : ℝ

/-- Determines if a student is eligible for the geometry class -/
def is_eligible (s : StudentScores) : Prop :=
  (s.algebra_sem1 + s.algebra_sem2) / 2 ≥ s.algebra_required_avg ∧
  s.statistics ≥ s.statistics_required

/-- Calculates the minimum score needed in the second semester of Algebra -/
def min_algebra_sem2_score (s : StudentScores) : ℝ :=
  2 * s.algebra_required_avg - s.algebra_sem1

/-- Theorem stating the minimum score Fisher needs in the second semester of Algebra -/
theorem fisher_min_score (fisher : StudentScores)
  (h1 : fisher.algebra_required_avg = 85)
  (h2 : fisher.statistics_required = 80)
  (h3 : fisher.algebra_sem1 = 84)
  (h4 : fisher.statistics = 82) :
  min_algebra_sem2_score fisher = 86 ∧ 
  is_eligible { fisher with algebra_sem2 := min_algebra_sem2_score fisher } :=
sorry


end fisher_min_score_l682_68211


namespace thelmas_tomato_slices_l682_68234

def slices_per_meal : ℕ := 20
def people_to_feed : ℕ := 8
def tomatoes_needed : ℕ := 20

def slices_per_tomato : ℕ := (slices_per_meal * people_to_feed) / tomatoes_needed

theorem thelmas_tomato_slices : slices_per_tomato = 8 := by
  sorry

end thelmas_tomato_slices_l682_68234


namespace constant_sign_of_root_combination_l682_68278

/-- Represents a polynomial of degree 4 -/
structure Polynomial4 where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ

/-- The roots of P(x) - t for a given t -/
def roots (P : Polynomial4) (t : ℝ) : Fin 4 → ℝ := sorry

/-- Predicate to check if P(x) - t has four distinct real roots -/
def has_four_distinct_real_roots (P : Polynomial4) (t : ℝ) : Prop := sorry

theorem constant_sign_of_root_combination (P : Polynomial4) :
  ∀ t₁ t₂ : ℝ, has_four_distinct_real_roots P t₁ → has_four_distinct_real_roots P t₂ →
  (roots P t₁ 0 + roots P t₁ 3 - roots P t₁ 1 - roots P t₁ 2) *
  (roots P t₂ 0 + roots P t₂ 3 - roots P t₂ 1 - roots P t₂ 2) > 0 :=
sorry

end constant_sign_of_root_combination_l682_68278


namespace sufficient_not_necessary_l682_68227

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 1 → x^2 + x - 2 > 0) ∧ 
  (∃ x : ℝ, x^2 + x - 2 > 0 ∧ x ≤ 1) := by
  sorry

end sufficient_not_necessary_l682_68227


namespace wang_shifu_not_yuan_dramatist_l682_68287

/-- The set of four great dramatists of the Yuan Dynasty -/
def YuanDramatists : Set String :=
  {"Guan Hanqing", "Zheng Guangzu", "Bai Pu", "Ma Zhiyuan"}

/-- Wang Shifu -/
def WangShifu : String := "Wang Shifu"

/-- Theorem stating that Wang Shifu is not one of the four great dramatists of the Yuan Dynasty -/
theorem wang_shifu_not_yuan_dramatist :
  WangShifu ∉ YuanDramatists := by
  sorry

end wang_shifu_not_yuan_dramatist_l682_68287


namespace max_handshakes_l682_68204

theorem max_handshakes (N : ℕ) (h1 : N > 4) : ∃ (max_shaken : ℕ),
  (∃ (not_shaken : Fin N → Prop),
    (∃ (a b : Fin N), a ≠ b ∧ not_shaken a ∧ not_shaken b ∧
      ∀ (x : Fin N), not_shaken x → (x = a ∨ x = b)) ∧
    (∀ (x : Fin N), ¬(not_shaken x) →
      ∀ (y : Fin N), y ≠ x → ∃ (shaken : Prop), shaken)) ∧
  max_shaken = N - 2 ∧
  ∀ (k : ℕ), k > max_shaken →
    ¬(∃ (not_shaken : Fin N → Prop),
      (∃ (a b : Fin N), a ≠ b ∧ not_shaken a ∧ not_shaken b ∧
        ∀ (x : Fin N), not_shaken x → (x = a ∨ x = b)) ∧
      (∀ (x : Fin N), ¬(not_shaken x) →
        ∀ (y : Fin N), y ≠ x → ∃ (shaken : Prop), shaken))
  := by sorry

end max_handshakes_l682_68204


namespace circle_center_and_radius_l682_68271

def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

def is_center (h k : ℝ) : Prop := ∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 4

theorem circle_center_and_radius :
  is_center 2 0 ∧ ∀ x y : ℝ, circle_equation x y → (x - 2)^2 + y^2 ≤ 4 := by sorry

end circle_center_and_radius_l682_68271


namespace election_votes_proof_l682_68286

theorem election_votes_proof (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (62 * total_votes) / 100 - (38 * total_votes) / 100 = 360) : 
  (62 * total_votes) / 100 = 930 := by
sorry

end election_votes_proof_l682_68286


namespace average_sale_l682_68240

def sales : List ℕ := [5420, 5660, 6200, 6350, 6500]
def projected_sale : ℕ := 6470

theorem average_sale :
  (sales.sum + projected_sale) / (sales.length + 1) = 6100 := by
  sorry

end average_sale_l682_68240


namespace trig_identity_l682_68252

theorem trig_identity (α : Real) (h : Real.tan α = 3) :
  2 * (Real.sin α)^2 + 4 * (Real.sin α) * (Real.cos α) - 9 * (Real.cos α)^2 = 21/10 := by
  sorry

end trig_identity_l682_68252


namespace odd_function_value_l682_68276

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value (f : ℝ → ℝ) (h_odd : IsOdd f)
    (h_periodic : ∀ x, f (x + 4) = f x + f 2) (h_f_neg_one : f (-1) = -2) :
    f 2013 = 2 := by
  sorry

end odd_function_value_l682_68276


namespace plywood_cut_perimeter_difference_l682_68244

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the plywood and its division -/
structure Plywood where
  length : ℝ
  width : ℝ
  num_pieces : ℕ

/-- Represents a way of cutting the plywood -/
structure CutMethod where
  piece : Rectangle
  is_valid : Bool

theorem plywood_cut_perimeter_difference 
  (p : Plywood) 
  (h1 : p.length = 10 ∧ p.width = 5)
  (h2 : p.num_pieces = 5) :
  ∃ (m1 m2 : CutMethod), 
    m1.is_valid ∧ m2.is_valid ∧ 
    perimeter m1.piece - perimeter m2.piece = 8 ∧
    ∀ (m : CutMethod), m.is_valid → 
      perimeter m.piece ≤ perimeter m1.piece ∧
      perimeter m.piece ≥ perimeter m2.piece := by
  sorry


end plywood_cut_perimeter_difference_l682_68244


namespace intersection_A_complement_B_l682_68259

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-2, 0, 1}
def B : Set Int := {-1, 0, 2}

theorem intersection_A_complement_B : A ∩ (U \ B) = {-2, 1} := by
  sorry

end intersection_A_complement_B_l682_68259


namespace product_of_solutions_l682_68220

theorem product_of_solutions (x : ℝ) : 
  (∀ x, -49 = -2*x^2 + 6*x) → 
  (∃ α β : ℝ, (α * β = -24.5) ∧ (-49 = -2*α^2 + 6*α) ∧ (-49 = -2*β^2 + 6*β)) :=
by sorry

end product_of_solutions_l682_68220


namespace cos_210_degrees_l682_68216

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_degrees_l682_68216


namespace lock_settings_count_l682_68261

/-- The number of digits on each dial of the lock -/
def numDigits : ℕ := 10

/-- The number of dials on the lock -/
def numDials : ℕ := 4

/-- Calculates the number of different settings for the lock -/
def lockSettings : ℕ := numDigits * (numDigits - 1) * (numDigits - 2) * (numDigits - 3)

/-- Theorem stating that the number of different settings for the lock is 5040 -/
theorem lock_settings_count : lockSettings = 5040 := by
  sorry

end lock_settings_count_l682_68261


namespace al2s3_weight_l682_68203

/-- The molecular weight of a compound in grams per mole -/
def molecular_weight (al_weight s_weight : ℝ) : ℝ :=
  2 * al_weight + 3 * s_weight

/-- The total weight of a given number of moles of a compound -/
def total_weight (moles : ℝ) (mol_weight : ℝ) : ℝ :=
  moles * mol_weight

/-- Theorem: The molecular weight of 3 moles of Al2S3 is 450.51 grams -/
theorem al2s3_weight : 
  let al_weight := 26.98
  let s_weight := 32.07
  let mol_weight := molecular_weight al_weight s_weight
  total_weight 3 mol_weight = 450.51 := by
sorry


end al2s3_weight_l682_68203


namespace arithmetic_sequence_seventh_term_l682_68223

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_fourth : a 4 = 5)
  (h_sum : a 5 + a 6 = 11) :
  a 7 = 6 := by
sorry

end arithmetic_sequence_seventh_term_l682_68223


namespace intersection_of_A_and_B_l682_68267

-- Define set A
def A : Set ℕ := {4, 5, 6, 7}

-- Define set B
def B : Set ℕ := {x | 3 ≤ x ∧ x < 6}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {4, 5} := by
  sorry

end intersection_of_A_and_B_l682_68267
