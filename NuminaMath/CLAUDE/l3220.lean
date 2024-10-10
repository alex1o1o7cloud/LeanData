import Mathlib

namespace quadrilateral_rod_count_l3220_322023

theorem quadrilateral_rod_count : 
  let a : ℕ := 5
  let b : ℕ := 12
  let c : ℕ := 20
  let valid_length (d : ℕ) : Prop := 
    1 ≤ d ∧ d ≤ 40 ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c ∧
    a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a
  (Finset.filter valid_length (Finset.range 41)).card = 30 :=
by sorry

end quadrilateral_rod_count_l3220_322023


namespace triangle_formation_l3220_322087

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem triangle_formation :
  can_form_triangle 3 4 5 ∧
  ¬can_form_triangle 1 1 2 ∧
  ¬can_form_triangle 1 4 6 ∧
  ¬can_form_triangle 2 3 7 := by
  sorry

end triangle_formation_l3220_322087


namespace euro_equation_solution_l3220_322079

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- State the theorem
theorem euro_equation_solution :
  ∀ y : ℝ, euro y (euro 7 5) = 560 → y = 4 := by
  sorry

end euro_equation_solution_l3220_322079


namespace trapezoid_area_l3220_322069

theorem trapezoid_area (large_triangle_area small_triangle_area : ℝ)
  (num_trapezoids : ℕ) (h1 : large_triangle_area = 36)
  (h2 : small_triangle_area = 4) (h3 : num_trapezoids = 4) :
  (large_triangle_area - small_triangle_area) / num_trapezoids = 8 := by
  sorry

end trapezoid_area_l3220_322069


namespace line_perpendicular_to_plane_implies_perpendicular_to_contained_line_l3220_322066

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationship operators
variable (contained_in : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane_implies_perpendicular_to_contained_line
  (m n : Line) (α : Plane)
  (h1 : contained_in m α)
  (h2 : perpendicular n α) :
  perpendicular_lines m n :=
sorry

end line_perpendicular_to_plane_implies_perpendicular_to_contained_line_l3220_322066


namespace f_not_monotonic_range_l3220_322046

/-- The function f(x) = x³ - 12x -/
def f (x : ℝ) : ℝ := x^3 - 12*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 12

/-- A function is not monotonic on an interval if its derivative has a zero in that interval -/
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x, a < x ∧ x < b ∧ f' x = 0

/-- The theorem stating the range of k for which f is not monotonic on (k, k+2) -/
theorem f_not_monotonic_range :
  ∀ k : ℝ, not_monotonic f k (k+2) ↔ (k > -4 ∧ k < -2) ∨ (k > 0 ∧ k < 2) :=
sorry

end f_not_monotonic_range_l3220_322046


namespace twelfth_term_of_specific_sequence_l3220_322090

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence (α : Type*) [Field α] where
  first_term : α
  common_difference : α

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence ℚ) (n : ℕ) : ℚ :=
  seq.first_term + (n - 1 : ℚ) * seq.common_difference

theorem twelfth_term_of_specific_sequence :
  let seq := ArithmeticSequence.mk (1/2 : ℚ) ((5/6 - 1/2) : ℚ)
  nth_term seq 2 = 5/6 → nth_term seq 3 = 7/6 → nth_term seq 12 = 25/6 := by
  sorry


end twelfth_term_of_specific_sequence_l3220_322090


namespace not_well_placed_2_pow_2011_l3220_322068

/-- Represents the first number in a row of the triangular table -/
def first_in_row (row : ℕ) : ℕ := (row - 1)^2 + 1

/-- Represents the first number in a column of the triangular table -/
def first_in_column (col : ℕ) : ℕ := (col - 1)^2 + 1

/-- A number is well-placed if it equals the sum of the first number in its row and the first number in its column -/
def is_well_placed (n : ℕ) : Prop :=
  ∃ (row col : ℕ), n = first_in_row row + first_in_column col

theorem not_well_placed_2_pow_2011 : ¬ is_well_placed (2^2011) := by
  sorry

end not_well_placed_2_pow_2011_l3220_322068


namespace max_cookies_eaten_l3220_322058

/-- Given two people sharing 30 cookies, where one eats twice as many as the other,
    the maximum number of cookies the person eating fewer can have is 10. -/
theorem max_cookies_eaten (total : ℕ) (andy bella : ℕ) : 
  total = 30 →
  bella = 2 * andy →
  andy + bella = total →
  andy ≤ 10 ∧ ∃ (n : ℕ), n = 10 ∧ n ≤ andy :=
by sorry

end max_cookies_eaten_l3220_322058


namespace x_squared_minus_y_squared_l3220_322038

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5/8) (h2 : x - y = 3/8) : x^2 - y^2 = 15/64 := by
  sorry

end x_squared_minus_y_squared_l3220_322038


namespace abc_inequality_l3220_322049

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end abc_inequality_l3220_322049


namespace circles_intersect_l3220_322018

/-- Two circles are intersecting if the distance between their centers is greater than the absolute 
    difference of their radii and less than the sum of their radii. -/
def are_intersecting (r1 r2 d : ℝ) : Prop :=
  d > |r1 - r2| ∧ d < r1 + r2

/-- Given two circles with radii 5 and 8, and distance between centers 8, 
    prove that they are intersecting. -/
theorem circles_intersect : are_intersecting 5 8 8 := by
  sorry

end circles_intersect_l3220_322018


namespace centroid_coincidence_l3220_322005

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Calculates the centroid of a triangle -/
def triangleCentroid (t : Triangle) : Point := sorry

/-- Theorem: The centroid of a triangle coincides with the centroid of its subtriangles -/
theorem centroid_coincidence (ABC : Triangle) : 
  let D : Point := sorry -- D is the foot of the altitude from C to AB
  let ACD : Triangle := ⟨ABC.A, ABC.C, D⟩
  let BCD : Triangle := ⟨ABC.B, ABC.C, D⟩
  let M1 : Point := triangleCentroid ACD
  let M2 : Point := triangleCentroid BCD
  let Z : Point := triangleCentroid ABC
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧
    Z.x = t * M1.x + (1 - t) * M2.x ∧
    Z.y = t * M1.y + (1 - t) * M2.y ∧
    t = (triangleArea BCD) / (triangleArea ACD + triangleArea BCD) :=
by sorry

end centroid_coincidence_l3220_322005


namespace axis_of_symmetry_l3220_322055

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2*x

-- Theorem stating that the axis of symmetry is x = 1
theorem axis_of_symmetry :
  ∀ y : ℝ, ∃ x : ℝ, parabola (x + 1) = parabola (1 - x) := by
  sorry

end axis_of_symmetry_l3220_322055


namespace count_valid_numbers_l3220_322073

/-- A function that generates all valid four-digit even numbers greater than 2000
    using digits 0, 1, 2, 3, 4, 5 without repetition -/
def validNumbers : Finset Nat := sorry

/-- The cardinality of the set of valid numbers -/
theorem count_valid_numbers : Finset.card validNumbers = 120 := by sorry

end count_valid_numbers_l3220_322073


namespace ryan_english_study_time_l3220_322006

/-- The number of hours Ryan spends on learning Chinese daily -/
def chinese_hours : ℕ := 5

/-- The number of additional hours Ryan spends on learning English compared to Chinese -/
def additional_english_hours : ℕ := 2

/-- The number of hours Ryan spends on learning English daily -/
def english_hours : ℕ := chinese_hours + additional_english_hours

theorem ryan_english_study_time : english_hours = 7 := by
  sorry

end ryan_english_study_time_l3220_322006


namespace tim_score_theorem_l3220_322003

/-- Sum of the first n even numbers -/
def sumFirstNEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- A number is recognizable if it's 90 (for this specific problem) -/
def isRecognizable (x : ℕ) : Prop := x = 90

/-- A number is a square number if it's the square of some integer -/
def isSquareNumber (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem tim_score_theorem :
  ∃ n : ℕ, isSquareNumber n ∧ isRecognizable (sumFirstNEvenNumbers n) ∧
  ∀ m : ℕ, m < n → ¬(isSquareNumber m ∧ isRecognizable (sumFirstNEvenNumbers m)) :=
by sorry

end tim_score_theorem_l3220_322003


namespace game_theory_proof_l3220_322089

theorem game_theory_proof (x y : ℝ) : 
  (x + y + (24 - x - y) = 24) →
  (2*x - 24 = 2) →
  (4*y - 24 = 4) →
  (∀ (a b c : ℝ), (a + b + c = 24) → (a = 8 ∧ b = 8 ∧ c = 8)) →
  (x = 13 ∧ y = 7 ∧ 24 - x - y = 4) :=
by sorry

end game_theory_proof_l3220_322089


namespace jake_has_seven_peaches_l3220_322032

-- Define the number of peaches and apples for Steven
def steven_peaches : ℕ := 19
def steven_apples : ℕ := 14

-- Define Jake's peaches and apples in relation to Steven's
def jake_peaches : ℕ := steven_peaches - 12
def jake_apples : ℕ := steven_apples + 79

-- Theorem to prove
theorem jake_has_seven_peaches : jake_peaches = 7 := by
  sorry

end jake_has_seven_peaches_l3220_322032


namespace probability_three_blue_jellybeans_l3220_322027

def total_jellybeans : ℕ := 20
def initial_blue_jellybeans : ℕ := 10

def probability_all_blue : ℚ := 2 / 19

theorem probability_three_blue_jellybeans :
  let p1 := initial_blue_jellybeans / total_jellybeans
  let p2 := (initial_blue_jellybeans - 1) / (total_jellybeans - 1)
  let p3 := (initial_blue_jellybeans - 2) / (total_jellybeans - 2)
  p1 * p2 * p3 = probability_all_blue := by
  sorry

end probability_three_blue_jellybeans_l3220_322027


namespace largest_n_divisibility_l3220_322091

theorem largest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ (n + 10) ∣ (n^3 + 100) ∧ ∀ (m : ℕ), m > n → ¬((m + 10) ∣ (m^3 + 100)) :=
by
  use 890
  sorry

end largest_n_divisibility_l3220_322091


namespace number_of_divisors_of_45_l3220_322008

theorem number_of_divisors_of_45 : Nat.card {d : ℕ | d ∣ 45} = 6 := by
  sorry

end number_of_divisors_of_45_l3220_322008


namespace prob_shortest_diagonal_11_gon_l3220_322031

/-- The number of sides in our regular polygon -/
def n : ℕ := 11

/-- The total number of diagonals in an n-sided regular polygon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in an n-sided regular polygon -/
def shortest_diagonals (n : ℕ) : ℕ := n

/-- The probability of selecting a shortest diagonal in an n-sided regular polygon -/
def prob_shortest_diagonal (n : ℕ) : ℚ :=
  shortest_diagonals n / total_diagonals n

/-- Theorem: The probability of selecting a shortest diagonal in a regular 11-sided polygon is 1/4 -/
theorem prob_shortest_diagonal_11_gon :
  prob_shortest_diagonal n = 1 / 4 := by sorry

end prob_shortest_diagonal_11_gon_l3220_322031


namespace gcf_4320_2550_l3220_322071

theorem gcf_4320_2550 : Nat.gcd 4320 2550 = 30 := by
  sorry

end gcf_4320_2550_l3220_322071


namespace cos_alpha_minus_pi_fourth_l3220_322076

theorem cos_alpha_minus_pi_fourth (α : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : Real.tan (α + π/4) = -3) : 
  Real.cos (α - π/4) = 3 * Real.sqrt 10 / 10 := by
sorry

end cos_alpha_minus_pi_fourth_l3220_322076


namespace john_received_120_l3220_322050

def grandpa_amount : ℕ := 30

def grandma_amount : ℕ := 3 * grandpa_amount

def total_amount : ℕ := grandpa_amount + grandma_amount

theorem john_received_120 : total_amount = 120 := by
  sorry

end john_received_120_l3220_322050


namespace largest_n_with_unique_k_l3220_322074

theorem largest_n_with_unique_k : ∃ (k : ℤ), 
  (5 : ℚ)/12 < (7 : ℚ)/(7 + k) ∧ (7 : ℚ)/(7 + k) < 4/9 ∧ 
  (∀ (m : ℕ) (j : ℤ), m > 7 → 
    ((5 : ℚ)/12 < (m : ℚ)/(m + j) ∧ (m : ℚ)/(m + j) < 4/9) → 
    (∃ (l : ℤ), l ≠ j ∧ (5 : ℚ)/12 < (m : ℚ)/(m + l) ∧ (m : ℚ)/(m + l) < 4/9)) :=
by sorry

end largest_n_with_unique_k_l3220_322074


namespace positive_abc_l3220_322041

theorem positive_abc (a b c : ℝ) 
  (sum_pos : a + b + c > 0)
  (sum_prod_pos : a * b + b * c + c * a > 0)
  (prod_pos : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
sorry

end positive_abc_l3220_322041


namespace ellipse_foci_distance_l3220_322096

/-- The distance between the foci of the ellipse 9x^2 + 36y^2 = 1296 is 12√3 -/
theorem ellipse_foci_distance (x y : ℝ) :
  (9 * x^2 + 36 * y^2 = 1296) →
  (∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = (12 * Real.sqrt 3)^2 ∧
    ∀ (p : ℝ × ℝ), 9 * p.1^2 + 36 * p.2^2 = 1296 →
      Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) +
      Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) =
      2 * Real.sqrt (144)) :=
by sorry

end ellipse_foci_distance_l3220_322096


namespace tangent_circle_value_l3220_322010

/-- A line in polar coordinates -/
def polar_line (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ + Real.sqrt 3 * ρ * Real.sin θ + 1 = 0

/-- A circle in polar coordinates -/
def polar_circle (a ρ θ : ℝ) : Prop :=
  ρ = 2 * a * Real.cos θ ∧ a > 0

/-- Tangency condition between a line and a circle -/
def is_tangent (a : ℝ) : Prop :=
  ∃ ρ θ, polar_line ρ θ ∧ polar_circle a ρ θ

theorem tangent_circle_value :
  ∃ a, is_tangent a ∧ a = 1 :=
sorry

end tangent_circle_value_l3220_322010


namespace arithmetic_sequence_12th_term_l3220_322042

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_4th : a 4 = 1) :
  a 12 = 15 := by
sorry

end arithmetic_sequence_12th_term_l3220_322042


namespace factor_expression_l3220_322099

theorem factor_expression (b : ℝ) : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) := by
  sorry

end factor_expression_l3220_322099


namespace consecutive_integers_base_equation_l3220_322019

/-- Converts a number from base b to base 10 -/
def toBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

theorem consecutive_integers_base_equation :
  ∀ C D : ℕ,
  C > 0 →
  D = C + 1 →
  toBase10 154 C + toBase10 52 D = toBase10 76 (C + D) →
  C + D = 11 := by sorry

end consecutive_integers_base_equation_l3220_322019


namespace expansion_coefficient_condition_l3220_322030

/-- The coefficient of the r-th term in the expansion of (2x + 1/x)^n -/
def coefficient (n : ℕ) (r : ℕ) : ℚ :=
  2^(n-r) * (n.choose r)

theorem expansion_coefficient_condition (n : ℕ) :
  (coefficient n 2 = 2 * coefficient n 3) → n = 5 := by
  sorry

end expansion_coefficient_condition_l3220_322030


namespace greatest_power_of_three_in_factorial_l3220_322036

theorem greatest_power_of_three_in_factorial :
  (∃ (n : ℕ), n = 9 ∧ 
   ∀ (k : ℕ), 3^k ∣ Nat.factorial 22 → k ≤ n) ∧
   (∀ (m : ℕ), m > 9 → ¬(3^m ∣ Nat.factorial 22)) := by
sorry

end greatest_power_of_three_in_factorial_l3220_322036


namespace student_correct_problems_l3220_322065

/-- Represents the number of problems solved correctly by a student. -/
def correct_problems (total : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (final_score : ℤ) : ℕ :=
  sorry

/-- Theorem stating that given the problem conditions, the number of correctly solved problems is 31. -/
theorem student_correct_problems :
  correct_problems 80 5 3 8 = 31 := by sorry

end student_correct_problems_l3220_322065


namespace equation_solution_l3220_322017

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (x - 16))) = 55 ∧ x = 15 := by
  sorry

end equation_solution_l3220_322017


namespace sams_weight_l3220_322059

/-- Given the weights of Tyler, Sam, Peter, and Alex, prove Sam's weight --/
theorem sams_weight (tyler sam peter alex : ℝ) : 
  tyler = sam + 25 →
  peter = tyler / 2 →
  alex = 2 * (sam + peter) →
  peter = 65 →
  sam = 105 := by
  sorry

end sams_weight_l3220_322059


namespace school_average_age_l3220_322001

theorem school_average_age 
  (total_students : ℕ) 
  (boys_avg_age girls_avg_age : ℚ) 
  (num_girls : ℕ) :
  total_students = 640 →
  boys_avg_age = 12 →
  girls_avg_age = 11 →
  num_girls = 160 →
  let num_boys := total_students - num_girls
  let total_age := boys_avg_age * num_boys + girls_avg_age * num_girls
  total_age / total_students = 11.75 := by
sorry

end school_average_age_l3220_322001


namespace inscribed_circle_cycle_l3220_322028

structure Triangle :=
  (A B C : Point)

structure Circle :=
  (center : Point)
  (radius : ℝ)

def inscribed_circle (T : Triangle) (i : ℕ) : Circle :=
  sorry

theorem inscribed_circle_cycle (T : Triangle) :
  inscribed_circle T 7 = inscribed_circle T 1 :=
sorry

end inscribed_circle_cycle_l3220_322028


namespace pages_ratio_is_one_to_two_l3220_322045

-- Define the total number of pages in the book
def total_pages : ℕ := 120

-- Define the number of pages read yesterday
def pages_read_yesterday : ℕ := 12

-- Define the number of pages read today
def pages_read_today : ℕ := 2 * pages_read_yesterday

-- Define the number of pages to be read tomorrow
def pages_to_read_tomorrow : ℕ := 42

-- Theorem statement
theorem pages_ratio_is_one_to_two :
  let pages_read_so_far := pages_read_yesterday + pages_read_today
  let remaining_pages := total_pages - pages_read_so_far
  (pages_to_read_tomorrow : ℚ) / remaining_pages = 1 / 2 := by
sorry

end pages_ratio_is_one_to_two_l3220_322045


namespace polycarp_kolka_numbers_l3220_322053

/-- The smallest 5-digit number composed of distinct even digits -/
def polycarp_number : ℕ := 20468

/-- Kolka's incorrect 5-digit number -/
def kolka_number : ℕ := 20486

/-- Checks if a number is a 5-digit number -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

/-- Checks if a number is composed of distinct even digits -/
def has_distinct_even_digits (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    n = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    Even a ∧ Even b ∧ Even c ∧ Even d ∧ Even e

theorem polycarp_kolka_numbers :
  (is_five_digit polycarp_number) ∧
  (has_distinct_even_digits polycarp_number) ∧
  (∀ n : ℕ, is_five_digit n → has_distinct_even_digits n → n ≥ polycarp_number) ∧
  (is_five_digit kolka_number) ∧
  (has_distinct_even_digits kolka_number) ∧
  (kolka_number - polycarp_number < 100) ∧
  (kolka_number ≠ polycarp_number) →
  kolka_number = 20486 :=
by sorry

end polycarp_kolka_numbers_l3220_322053


namespace smallest_candy_count_l3220_322077

theorem smallest_candy_count : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n ≤ 999) ∧ 
  (n + 7) % 9 = 0 ∧ 
  (n - 9) % 7 = 0 ∧
  n = 110 ∧
  ∀ (m : ℕ), (m ≥ 100 ∧ m ≤ 999) → 
    (m + 7) % 9 = 0 → (m - 9) % 7 = 0 → m ≥ n :=
by sorry

end smallest_candy_count_l3220_322077


namespace min_value_quadratic_l3220_322026

theorem min_value_quadratic (a b : ℝ) :
  2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a - 4 * b + 2044 ≥ 1976 ∧
  (2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a - 4 * b + 2044 = 1976 ↔ a = 8 ∧ b = 2) := by
  sorry

end min_value_quadratic_l3220_322026


namespace quadratic_equation_a_value_l3220_322025

theorem quadratic_equation_a_value (a b c : ℝ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c →
    (x = -2 ∧ y = -3) ∨ (x = 1 ∧ y = 0)) →
  (∀ x y : ℝ, y = a * (x + 2)^2 - 3) →
  a = 1/3 := by sorry

end quadratic_equation_a_value_l3220_322025


namespace oliver_bath_water_usage_l3220_322060

/-- Calculates the weekly water usage for baths given the bucket capacity, 
    number of buckets to fill the tub, number of buckets removed, and days per week -/
def weekly_water_usage (bucket_capacity : ℕ) (buckets_to_fill : ℕ) (buckets_removed : ℕ) (days_per_week : ℕ) : ℕ :=
  (buckets_to_fill - buckets_removed) * bucket_capacity * days_per_week

/-- Theorem stating that given the specific conditions, the weekly water usage is 9240 ounces -/
theorem oliver_bath_water_usage :
  weekly_water_usage 120 14 3 7 = 9240 := by
  sorry

end oliver_bath_water_usage_l3220_322060


namespace expression_factorization_l3220_322062

theorem expression_factorization (x : ℝ) : 
  (18 * x^6 + 50 * x^4 - 8) - (2 * x^6 - 6 * x^4 - 8) = 8 * x^4 * (2 * x^2 + 7) := by
  sorry

end expression_factorization_l3220_322062


namespace f_monotone_increasing_l3220_322040

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.log x / Real.log (1/2))^2 - 2 * (Real.log x / Real.log (1/2)) + 1

theorem f_monotone_increasing :
  ∀ x y, x ≥ Real.sqrt 2 / 2 → y ≥ Real.sqrt 2 / 2 → x < y → f x < f y :=
sorry

end f_monotone_increasing_l3220_322040


namespace perpendicular_planes_condition_l3220_322075

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and between lines
variable (perp_planes : Plane → Plane → Prop)
variable (perp_lines : Line → Line → Prop)

-- Define the relation for a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- Define the relation for two lines being different
variable (different : Line → Line → Prop)

-- Define the relation for two lines intersecting
variable (intersect : Line → Line → Prop)

-- State the theorem
theorem perpendicular_planes_condition 
  (α β : Plane) (m n l₁ l₂ : Line)
  (h1 : in_plane m α)
  (h2 : in_plane n α)
  (h3 : different m n)
  (h4 : in_plane l₁ β)
  (h5 : in_plane l₂ β)
  (h6 : intersect l₁ l₂)
  (h7 : perp_lines m l₁)
  (h8 : perp_lines m l₂) :
  perp_planes α β :=
sorry

end perpendicular_planes_condition_l3220_322075


namespace circle_equation_l3220_322084

theorem circle_equation (r : ℝ) (h1 : r = 6) :
  ∃ (a b : ℝ),
    (∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = r^2) ∧
    (b = r) ∧
    (∃ (x y : ℝ), x^2 + y^2 - 6*y + 8 = 0 ∧ (x - a)^2 + (y - b)^2 = (r - 1)^2) →
    ((a = 4 ∨ a = -4) ∧ b = 6) :=
by sorry

end circle_equation_l3220_322084


namespace sum_of_xy_on_circle_l3220_322033

theorem sum_of_xy_on_circle (x y : ℝ) (h : x^2 + y^2 = 16*x - 12*y + 20) : x + y = 2 := by
  sorry

end sum_of_xy_on_circle_l3220_322033


namespace power_mean_inequality_l3220_322086

theorem power_mean_inequality (a b : ℝ) (n : ℕ) 
  (ha : a > 0) (hb : b > 0) (hn : n > 0) : 
  (a^n + b^n) / 2 ≥ ((a + b) / 2)^n := by
  sorry

end power_mean_inequality_l3220_322086


namespace point_coordinates_l3220_322054

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the conditions for the point
def on_x_axis (p : Point) : Prop := p.2 = 0
def right_of_origin (p : Point) : Prop := p.1 > 0
def distance_from_origin (p : Point) (d : ℝ) : Prop := p.1^2 + p.2^2 = d^2

-- Theorem statement
theorem point_coordinates :
  ∀ (p : Point),
    on_x_axis p →
    right_of_origin p →
    distance_from_origin p 3 →
    p = (3, 0) :=
by
  sorry


end point_coordinates_l3220_322054


namespace cylinder_reciprocal_sum_l3220_322004

theorem cylinder_reciprocal_sum (r h : ℝ) (volume_eq : π * r^2 * h = 2) (surface_area_eq : 2 * π * r * h + 2 * π * r^2 = 12) :
  1 / r + 1 / h = 3 := by
sorry

end cylinder_reciprocal_sum_l3220_322004


namespace A_intersect_B_equals_one_two_l3220_322016

-- Define the universal set U
def U : Set ℤ := {x : ℤ | |x - 1| < 3}

-- Define set A
def A : Set ℤ := {1, 2, 3}

-- Define the complement of B in U
def C_U_B : Set ℤ := {-1, 3}

-- Theorem to prove
theorem A_intersect_B_equals_one_two : A ∩ (U \ C_U_B) = {1, 2} := by sorry

end A_intersect_B_equals_one_two_l3220_322016


namespace dog_catches_rabbit_l3220_322011

/-- Proves that a dog chasing a rabbit catches up in 4 minutes under given conditions -/
theorem dog_catches_rabbit (dog_speed rabbit_speed : ℝ) (head_start : ℝ) :
  dog_speed = 24 ∧ rabbit_speed = 15 ∧ head_start = 0.6 →
  (head_start / (dog_speed - rabbit_speed)) * 60 = 4 := by
  sorry

end dog_catches_rabbit_l3220_322011


namespace cost_to_fill_displays_l3220_322052

/-- Represents the inventory and pricing of a jewelry store -/
structure JewelryStore where
  necklace_capacity : ℕ
  current_necklaces : ℕ
  ring_capacity : ℕ
  current_rings : ℕ
  bracelet_capacity : ℕ
  current_bracelets : ℕ
  necklace_price : ℕ
  ring_price : ℕ
  bracelet_price : ℕ

/-- Calculates the total cost to fill all displays in the jewelry store -/
def total_cost_to_fill (store : JewelryStore) : ℕ :=
  ((store.necklace_capacity - store.current_necklaces) * store.necklace_price) +
  ((store.ring_capacity - store.current_rings) * store.ring_price) +
  ((store.bracelet_capacity - store.current_bracelets) * store.bracelet_price)

/-- Theorem stating that the total cost to fill all displays is $183 -/
theorem cost_to_fill_displays (store : JewelryStore) 
  (h1 : store.necklace_capacity = 12)
  (h2 : store.current_necklaces = 5)
  (h3 : store.ring_capacity = 30)
  (h4 : store.current_rings = 18)
  (h5 : store.bracelet_capacity = 15)
  (h6 : store.current_bracelets = 8)
  (h7 : store.necklace_price = 4)
  (h8 : store.ring_price = 10)
  (h9 : store.bracelet_price = 5) :
  total_cost_to_fill store = 183 := by
  sorry

end cost_to_fill_displays_l3220_322052


namespace find_m_l3220_322014

theorem find_m : ∃ m : ℕ, 
  (1 ^ (m + 1) / 5 ^ (m + 1)) * (1 ^ 18 / 4 ^ 18) = 1 / (2 * 10 ^ 35) ∧ m = 34 := by
  sorry

end find_m_l3220_322014


namespace sarahs_reading_capacity_l3220_322078

/-- Sarah's reading problem -/
theorem sarahs_reading_capacity 
  (pages_per_hour : ℕ) 
  (pages_per_book : ℕ) 
  (available_hours : ℕ) 
  (h1 : pages_per_hour = 120) 
  (h2 : pages_per_book = 360) 
  (h3 : available_hours = 8) :
  (available_hours * pages_per_hour) / pages_per_book = 2 :=
sorry

end sarahs_reading_capacity_l3220_322078


namespace roots_equation_t_value_l3220_322093

theorem roots_equation_t_value (n s : ℝ) (u v : ℝ) : 
  u^2 - n*u + 6 = 0 →
  v^2 - n*v + 6 = 0 →
  (u + 2/v)^2 - s*(u + 2/v) + t = 0 →
  (v + 2/u)^2 - s*(v + 2/u) + t = 0 →
  t = 32/3 := by
sorry

end roots_equation_t_value_l3220_322093


namespace johns_spending_l3220_322063

theorem johns_spending (total : ℚ) 
  (h1 : total = 24)
  (h2 : total * (1/4) + total * (1/3) + 6 + bakery = total) : 
  bakery / total = 1/6 :=
by sorry

end johns_spending_l3220_322063


namespace infinite_series_sum_l3220_322080

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) is equal to 9/4. -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (3*n - 2) / (n * (n + 1) * (n + 3))) = 9/4 := by
  sorry


end infinite_series_sum_l3220_322080


namespace scale_division_l3220_322082

/-- Given a scale of length 198 inches divided into 8 equal parts, 
    prove that the length of each part is 24.75 inches. -/
theorem scale_division (total_length : ℝ) (num_parts : ℕ) 
  (h1 : total_length = 198) 
  (h2 : num_parts = 8) :
  total_length / num_parts = 24.75 := by
  sorry

end scale_division_l3220_322082


namespace brick_length_calculation_l3220_322064

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- The problem statement -/
theorem brick_length_calculation (wall : Dimensions) (brick_width : ℝ) (brick_height : ℝ) 
    (num_bricks : ℕ) (h_wall : wall = ⟨800, 600, 22.5⟩) (h_brick_width : brick_width = 11.25) 
    (h_brick_height : brick_height = 6) (h_num_bricks : num_bricks = 2000) :
    ∃ (brick_length : ℝ), 
      volume wall = num_bricks * volume ⟨brick_length, brick_width, brick_height⟩ ∧ 
      brick_length = 80 := by
  sorry

end brick_length_calculation_l3220_322064


namespace exactly_three_statements_true_l3220_322034

-- Define the polyline distance function
def polyline_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Define points A, B, M, and N
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (1, 0)
def M : ℝ × ℝ := (-1, 0)
def N : ℝ × ℝ := (1, 0)

-- Statement 1
def statement_1 : Prop :=
  polyline_distance A.1 A.2 B.1 B.2 = 5

-- Statement 2
def statement_2 : Prop :=
  ∃ (S : Set (ℝ × ℝ)), S = {p : ℝ × ℝ | polyline_distance p.1 p.2 0 0 = 1} ∧
  ¬(∃ (center : ℝ × ℝ) (radius : ℝ), S = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2})

-- Statement 3
def statement_3 : Prop :=
  ∀ (C : ℝ × ℝ), (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ C = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))) →
    polyline_distance A.1 A.2 C.1 C.2 + polyline_distance C.1 C.2 B.1 B.2 = polyline_distance A.1 A.2 B.1 B.2

-- Statement 4
def statement_4 : Prop :=
  {p : ℝ × ℝ | polyline_distance p.1 p.2 M.1 M.2 = polyline_distance p.1 p.2 N.1 N.2} =
  {p : ℝ × ℝ | p.1 = 0}

-- Main theorem
theorem exactly_three_statements_true :
  (statement_1 ∧ ¬statement_2 ∧ statement_3 ∧ statement_4) := by sorry

end exactly_three_statements_true_l3220_322034


namespace correct_calculation_l3220_322007

theorem correct_calculation (x : ℝ) : 3 * x - 10 = 50 → 3 * x + 10 = 70 := by
  sorry

end correct_calculation_l3220_322007


namespace arithmetic_sequence_common_difference_l3220_322021

/-- An arithmetic sequence {a_n} with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 4 + seq.a 5 = 24) 
  (h2 : seq.S 6 = 48) : 
  common_difference seq = 4 := by
sorry

end arithmetic_sequence_common_difference_l3220_322021


namespace sqrt_eq_condition_l3220_322029

theorem sqrt_eq_condition (x y : ℝ) (h : x * y ≠ 0) :
  Real.sqrt (4 * x^2 * y^3) = -2 * x * y * Real.sqrt y ↔ x < 0 ∧ y > 0 := by
sorry

end sqrt_eq_condition_l3220_322029


namespace cube_sum_theorem_l3220_322035

-- Define a cube with 8 vertices
structure Cube :=
  (vertices : Fin 8 → ℝ)

-- Define the sum of numbers on a face
def face_sum (c : Cube) (v1 v2 v3 v4 : Fin 8) : ℝ :=
  c.vertices v1 + c.vertices v2 + c.vertices v3 + c.vertices v4

-- Define the sum of all face sums
def total_face_sum (c : Cube) : ℝ :=
  face_sum c 0 1 2 3 +
  face_sum c 0 1 4 5 +
  face_sum c 0 3 4 7 +
  face_sum c 1 2 5 6 +
  face_sum c 2 3 6 7 +
  face_sum c 4 5 6 7

-- Define the sum of all vertex values
def vertex_sum (c : Cube) : ℝ :=
  c.vertices 0 + c.vertices 1 + c.vertices 2 + c.vertices 3 +
  c.vertices 4 + c.vertices 5 + c.vertices 6 + c.vertices 7

-- Theorem statement
theorem cube_sum_theorem (c : Cube) :
  total_face_sum c = 2019 → vertex_sum c = 673 :=
by sorry

end cube_sum_theorem_l3220_322035


namespace journey_time_proof_l3220_322002

-- Define the journey segments
inductive Segment
| Uphill
| Flat
| Downhill

-- Define the journey parameters
def total_distance : ℝ := 50
def uphill_speed : ℝ := 3

-- Define the ratios
def length_ratio (s : Segment) : ℝ :=
  match s with
  | .Uphill => 1
  | .Flat => 2
  | .Downhill => 3

def time_ratio (s : Segment) : ℝ :=
  match s with
  | .Uphill => 4
  | .Flat => 5
  | .Downhill => 6

-- Define the theorem
theorem journey_time_proof :
  let total_ratio : ℝ := (length_ratio .Uphill) + (length_ratio .Flat) + (length_ratio .Downhill)
  let uphill_distance : ℝ := total_distance * (length_ratio .Uphill) / total_ratio
  let uphill_time : ℝ := uphill_distance / uphill_speed
  let time_ratio_sum : ℝ := (time_ratio .Uphill) + (time_ratio .Flat) + (time_ratio .Downhill)
  let total_time : ℝ := uphill_time * time_ratio_sum / (time_ratio .Uphill)
  total_time = 10 + 5 / 12 :=
by sorry


end journey_time_proof_l3220_322002


namespace cosine_period_proof_l3220_322070

/-- Given a cosine function y = a cos(bx + c) + d where a, b, c, and d are positive constants,
    and the graph covers three periods from 0 to 3π, prove that b = 2. -/
theorem cosine_period_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_period : (3 : ℝ) * (2 * π / b) = 3 * π) : b = 2 := by
  sorry

end cosine_period_proof_l3220_322070


namespace solution_set_abs_inequality_l3220_322072

theorem solution_set_abs_inequality (x : ℝ) :
  (|1 - 2*x| < 3) ↔ (-1 < x ∧ x < 2) :=
by sorry

end solution_set_abs_inequality_l3220_322072


namespace min_sum_with_constraints_l3220_322015

theorem min_sum_with_constraints (x y z : ℝ) 
  (hx : x ≥ 5) (hy : y ≥ 6) (hz : z ≥ 7) 
  (h_sum_sq : x^2 + y^2 + z^2 ≥ 125) : 
  x + y + z ≥ 19 ∧ ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ ≥ 5 ∧ y₀ ≥ 6 ∧ z₀ ≥ 7 ∧ 
    x₀^2 + y₀^2 + z₀^2 ≥ 125 ∧ 
    x₀ + y₀ + z₀ = 19 := by
  sorry

end min_sum_with_constraints_l3220_322015


namespace smallest_non_five_divisible_unit_digit_l3220_322022

def is_divisible_by_five (n : ℕ) : Prop := n % 5 = 0

def units_digit (n : ℕ) : ℕ := n % 10

def is_digit (d : ℕ) : Prop := d < 10

theorem smallest_non_five_divisible_unit_digit : 
  ∀ d : ℕ, is_digit d → 
  (∀ n : ℕ, is_divisible_by_five n → units_digit n ≠ d) → 
  d ≥ 1 :=
sorry

end smallest_non_five_divisible_unit_digit_l3220_322022


namespace impossible_to_use_all_parts_l3220_322095

theorem impossible_to_use_all_parts (p q r : ℕ) : 
  ¬∃ (x y z : ℕ), 
    (2 * x + 2 * z = 2 * p + 2 * r + 2) ∧ 
    (2 * x + y = 2 * p + q + 1) ∧ 
    (y + z = q + r) :=
by sorry

end impossible_to_use_all_parts_l3220_322095


namespace parabola_equation_l3220_322097

theorem parabola_equation (p : ℝ) (h_p : p > 0) :
  (∃ x y : ℝ, y^2 = 2*p*x ∧ 
    (x + p/2)^2 + y^2 = 100 ∧ 
    y^2 = 36) → 
  p = 2 ∨ p = 18 := by
sorry

end parabola_equation_l3220_322097


namespace probability_two_non_defective_pens_l3220_322092

/-- The probability of selecting two non-defective pens from a box with defective pens -/
theorem probability_two_non_defective_pens 
  (total_pens : ℕ) 
  (defective_pens : ℕ) 
  (selected_pens : ℕ) 
  (h1 : total_pens = 10) 
  (h2 : defective_pens = 2) 
  (h3 : selected_pens = 2) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 28 / 45 :=
by sorry

end probability_two_non_defective_pens_l3220_322092


namespace problem_solution_l3220_322044

theorem problem_solution :
  let x : ℤ := 5
  let y : ℤ := x + 3
  let z : ℤ := 3 * y + 1
  z = 25 := by sorry

end problem_solution_l3220_322044


namespace min_reciprocal_sum_l3220_322020

theorem min_reciprocal_sum (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : x^2 + y^2 = x*y*(x^2*y^2 + 2)) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = a*b*(a^2*b^2 + 2) → 
  1/x + 1/y ≤ 1/a + 1/b :=
by sorry

end min_reciprocal_sum_l3220_322020


namespace area_of_specific_region_l3220_322047

/-- The area of a specific region in a circle with an inscribed regular hexagon -/
theorem area_of_specific_region (r : ℝ) (s : ℝ) (h_r : r = 3) (h_s : s = 2) :
  let circle_area := π * r^2
  let hexagon_side := s
  let sector_angle := 120
  let sector_area := (sector_angle / 360) * circle_area
  let triangle_area := (1/2) * r^2 * Real.sin (sector_angle * π / 180)
  sector_area - triangle_area = 3 * π - (9 * Real.sqrt 3) / 4 := by
  sorry

end area_of_specific_region_l3220_322047


namespace consecutive_sequence_unique_l3220_322039

/-- Three consecutive natural numbers forming an arithmetic and geometric sequence -/
def ConsecutiveSequence (a b c : ℕ) : Prop :=
  (b = a + 1) ∧ (c = b + 1) ∧
  (b + 2)^2 = (a + 1) * (c + 5)

theorem consecutive_sequence_unique :
  ∀ a b c : ℕ, ConsecutiveSequence a b c → a = 1 ∧ b = 2 ∧ c = 3 :=
by
  sorry

end consecutive_sequence_unique_l3220_322039


namespace cubic_roots_relation_l3220_322024

theorem cubic_roots_relation (p q r : ℂ) (u v w : ℂ) : 
  (∀ x : ℂ, x^3 + 4*x^2 + 5*x - 14 = (x - p) * (x - q) * (x - r)) →
  (∀ x : ℂ, x^3 + u*x^2 + v*x + w = (x - (p + q)) * (x - (q + r)) * (x - (r + p))) →
  w = 34 := by
sorry

end cubic_roots_relation_l3220_322024


namespace max_value_and_minimum_l3220_322094

noncomputable def f (x a b c : ℝ) : ℝ := |x + a| - |x - b| + c

theorem max_value_and_minimum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmax : ∀ x, f x a b c ≤ 10) 
  (hmax_exists : ∃ x, f x a b c = 10) : 
  (a + b + c = 10) ∧ 
  (∀ a' b' c', a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 10 → 
    1/4 * (a - 1)^2 + (b - 2)^2 + (c - 3)^2 ≤ 1/4 * (a' - 1)^2 + (b' - 2)^2 + (c' - 3)^2) ∧
  (1/4 * (a - 1)^2 + (b - 2)^2 + (c - 3)^2 = 8/3) ∧
  (a = 11/3 ∧ b = 8/3 ∧ c = 11/3) :=
by sorry

end max_value_and_minimum_l3220_322094


namespace oyster_consumption_l3220_322037

/-- The number of oysters Squido eats -/
def squido_oysters : ℕ := 200

/-- The number of oysters Crabby eats -/
def crabby_oysters : ℕ := 2 * squido_oysters

/-- The total number of oysters eaten by Crabby and Squido -/
def total_oysters : ℕ := squido_oysters + crabby_oysters

theorem oyster_consumption :
  total_oysters = 600 :=
sorry

end oyster_consumption_l3220_322037


namespace structure_surface_area_270_l3220_322067

def surface_area_cube (side_length : ℝ) : ℝ := 6 * side_length^2

def structure_surface_area (large_side : ℝ) (medium_side : ℝ) (small_side : ℝ) : ℝ :=
  surface_area_cube large_side +
  4 * surface_area_cube medium_side +
  4 * surface_area_cube small_side

theorem structure_surface_area_270 :
  structure_surface_area 5 2 1 = 270 := by
  sorry

end structure_surface_area_270_l3220_322067


namespace smallest_factorization_coefficient_l3220_322012

theorem smallest_factorization_coefficient (b : ℕ) : 
  (∃ m n : ℤ, x^2 + b*x + 2023 = (x + m) * (x + n)) → b ≥ 136 :=
by sorry

end smallest_factorization_coefficient_l3220_322012


namespace function_transformation_l3220_322056

theorem function_transformation (f : ℝ → ℝ) (h : f 1 = 3) : f (-(-1)) + 1 = 4 := by
  sorry

end function_transformation_l3220_322056


namespace inequality_solution_l3220_322085

theorem inequality_solution (x : ℝ) (h : x ≠ 5) :
  (x * (x^2 + x + 1)) / ((x - 5)^2) ≥ 15 ↔ x ∈ Set.Iio 5 ∪ Set.Ioi 5 :=
by sorry

end inequality_solution_l3220_322085


namespace arithmetic_calculations_l3220_322088

theorem arithmetic_calculations :
  (-9 + 5 * (-6) - 18 / (-3) = -33) ∧
  ((-3/4 - 5/8 + 9/12) * (-24) + (-8) / (2/3) = 3) := by
sorry

end arithmetic_calculations_l3220_322088


namespace table_length_is_77_l3220_322098

/-- Represents the dimensions and placement of sheets on a table. -/
structure TableSetup where
  tableWidth : ℕ
  tableLength : ℕ
  sheetWidth : ℕ
  sheetHeight : ℕ
  sheetCount : ℕ

/-- Checks if the given setup satisfies the conditions of the problem. -/
def isValidSetup (setup : TableSetup) : Prop :=
  setup.tableWidth = 80 ∧
  setup.sheetWidth = 8 ∧
  setup.sheetHeight = 5 ∧
  setup.sheetWidth + setup.sheetCount = setup.tableWidth ∧
  setup.sheetHeight + setup.sheetCount = setup.tableLength

/-- The main theorem stating that if the setup is valid, the table length must be 77. -/
theorem table_length_is_77 (setup : TableSetup) :
  isValidSetup setup → setup.tableLength = 77 := by
  sorry

#check table_length_is_77

end table_length_is_77_l3220_322098


namespace path_area_is_775_l3220_322081

/-- Represents the dimensions and cost of a rectangular field with a surrounding path. -/
structure FieldWithPath where
  fieldLength : ℝ
  fieldWidth : ℝ
  pathWidth : ℝ
  pathCostPerSqm : ℝ
  totalPathCost : ℝ

/-- Calculates the area of the path surrounding a rectangular field. -/
def pathArea (f : FieldWithPath) : ℝ :=
  let totalLength := f.fieldLength + 2 * f.pathWidth
  let totalWidth := f.fieldWidth + 2 * f.pathWidth
  totalLength * totalWidth - f.fieldLength * f.fieldWidth

/-- Theorem stating that the area of the path is 775 sq m for the given field dimensions. -/
theorem path_area_is_775 (f : FieldWithPath)
  (h1 : f.fieldLength = 95)
  (h2 : f.fieldWidth = 55)
  (h3 : f.pathWidth = 2.5)
  (h4 : f.pathCostPerSqm = 2)
  (h5 : f.totalPathCost = 1550) :
  pathArea f = 775 := by
  sorry

end path_area_is_775_l3220_322081


namespace fraction_addition_l3220_322083

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end fraction_addition_l3220_322083


namespace expand_product_l3220_322048

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end expand_product_l3220_322048


namespace geometric_series_sum_l3220_322061

/-- The sum of a finite geometric series -/
def geometricSum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_series_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometricSum a r n = 4/3 := by
sorry

end geometric_series_sum_l3220_322061


namespace set_operations_l3220_322000

def A : Set ℕ := {x | x > 0 ∧ x < 9}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {3, 4, 5, 6}

theorem set_operations :
  (A ∩ B = {1, 2, 3}) ∧
  (A ∩ C = {3, 4, 5, 6}) ∧
  (A ∩ (B ∪ C) = {1, 2, 3, 4, 5, 6}) ∧
  (A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8}) := by
  sorry

end set_operations_l3220_322000


namespace invalid_atomic_number_difference_l3220_322051

/-- Represents a period in the periodic table -/
inductive Period
| Second
| Third
| Fourth
| Fifth
| Sixth

/-- Represents an element in the periodic table -/
structure Element where
  atomicNumber : ℕ
  period : Period

/-- The difference in atomic numbers between elements in groups VIA and IA in the same period -/
def atomicNumberDifference (p : Period) : ℕ :=
  match p with
  | Period.Second => 5
  | Period.Third => 5
  | Period.Fourth => 15
  | Period.Fifth => 15
  | Period.Sixth => 29

theorem invalid_atomic_number_difference (X Y : Element) 
  (h1 : X.period = Y.period)
  (h2 : Y.atomicNumber = X.atomicNumber + atomicNumberDifference X.period) :
  Y.atomicNumber - X.atomicNumber ≠ 9 := by
  sorry

#check invalid_atomic_number_difference

end invalid_atomic_number_difference_l3220_322051


namespace lamplighter_monkey_speed_l3220_322043

/-- A Lamplighter monkey's movement characteristics -/
structure LamplighterMonkey where
  swingingSpeed : ℝ
  runningSpeed : ℝ
  runningTime : ℝ
  swingingTime : ℝ
  totalDistance : ℝ

/-- Theorem: Given the characteristics of a Lamplighter monkey's movement,
    prove that its running speed is 15 feet per second -/
theorem lamplighter_monkey_speed (monkey : LamplighterMonkey)
  (h1 : monkey.swingingSpeed = 10)
  (h2 : monkey.runningTime = 5)
  (h3 : monkey.swingingTime = 10)
  (h4 : monkey.totalDistance = 175) :
  monkey.runningSpeed = 15 := by
  sorry

#check lamplighter_monkey_speed

end lamplighter_monkey_speed_l3220_322043


namespace cos_90_degrees_zero_l3220_322013

theorem cos_90_degrees_zero : Real.cos (π / 2) = 0 := by
  sorry

end cos_90_degrees_zero_l3220_322013


namespace f_of_three_equals_nine_l3220_322057

theorem f_of_three_equals_nine (f : ℝ → ℝ) (h : ∀ x, f x = x^2) : f 3 = 9 := by
  sorry

end f_of_three_equals_nine_l3220_322057


namespace sum_geq_three_cube_root_three_l3220_322009

theorem sum_geq_three_cube_root_three
  (a b c : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (h : a^3 + b^3 + c^3 = a^2 * b^2 * c^2) :
  a + b + c ≥ 3 * (3 : ℝ)^(1/3) :=
by sorry

end sum_geq_three_cube_root_three_l3220_322009
