import Mathlib

namespace algebraic_expression_value_l1840_184033

theorem algebraic_expression_value (a b : ℝ) : 
  (a * 1^3 + b * 1 + 1 = 5) → (a * (-1)^3 + b * (-1) + 1 = -3) := by
  sorry

end algebraic_expression_value_l1840_184033


namespace factorization_of_3x_squared_minus_12_l1840_184072

theorem factorization_of_3x_squared_minus_12 (x : ℝ) :
  3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := by
  sorry

end factorization_of_3x_squared_minus_12_l1840_184072


namespace pulley_distance_l1840_184032

theorem pulley_distance (r₁ r₂ t d : ℝ) 
  (hr₁ : r₁ = 14)
  (hr₂ : r₂ = 4)
  (ht : t = 24)
  (hd : d = Real.sqrt ((r₁ - r₂)^2 + t^2)) :
  d = 26 := by
  sorry

end pulley_distance_l1840_184032


namespace investment_growth_l1840_184040

/-- Compound interest function -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Problem statement -/
theorem investment_growth :
  let principal : ℝ := 8000
  let rate : ℝ := 0.04
  let time : ℕ := 10
  abs (compound_interest principal rate time - 11841.92) < 0.01 := by
  sorry

end investment_growth_l1840_184040


namespace sibling_ages_sum_l1840_184049

/-- Given four positive integers representing ages of siblings, prove that their sum is 24 --/
theorem sibling_ages_sum (x y z : ℕ) (h1 : 2 * x^2 + y^2 + z^2 = 194) (h2 : x > y) (h3 : y > z) :
  x + x + y + z = 24 := by
  sorry

end sibling_ages_sum_l1840_184049


namespace roots_are_cosines_of_triangle_angles_l1840_184037

-- Define the polynomial p(x)
def p (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the condition
def condition (a b c : ℝ) : Prop := a^2 - 2*b - 2*c = 1

-- Theorem statement
theorem roots_are_cosines_of_triangle_angles 
  (a b c : ℝ) 
  (h_positive_roots : ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    ∀ t : ℝ, p a b c t = 0 ↔ t = x ∨ t = y ∨ t = z) :
  condition a b c ↔ 
  ∃ A B C : ℝ, 
    0 < A ∧ A < π/2 ∧
    0 < B ∧ B < π/2 ∧
    0 < C ∧ C < π/2 ∧
    A + B + C = π ∧
    (∀ t : ℝ, p a b c t = 0 ↔ t = Real.cos A ∨ t = Real.cos B ∨ t = Real.cos C) :=
by sorry

end roots_are_cosines_of_triangle_angles_l1840_184037


namespace painted_cube_one_third_blue_iff_three_l1840_184070

/-- Represents a cube with side length n, painted blue on all faces and cut into n^3 unit cubes -/
structure PaintedCube where
  n : ℕ

/-- The total number of faces of all unit cubes -/
def PaintedCube.totalFaces (c : PaintedCube) : ℕ := 6 * c.n^3

/-- The number of blue faces among all unit cubes -/
def PaintedCube.blueFaces (c : PaintedCube) : ℕ := 6 * c.n^2

/-- The condition that exactly one-third of the total faces are blue -/
def PaintedCube.oneThirdBlue (c : PaintedCube) : Prop :=
  3 * c.blueFaces = c.totalFaces

theorem painted_cube_one_third_blue_iff_three (c : PaintedCube) :
  c.oneThirdBlue ↔ c.n = 3 := by sorry

end painted_cube_one_third_blue_iff_three_l1840_184070


namespace max_value_theorem_l1840_184011

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (2 * x + y)) + (y / (x + 2 * y)) ≤ 2/3 := by sorry

end max_value_theorem_l1840_184011


namespace camping_trip_percentage_l1840_184066

theorem camping_trip_percentage (total_students : ℕ) 
  (h1 : (22 : ℝ) / 100 * total_students = (25 : ℝ) / 100 * ((88 : ℝ) / 100 * total_students)) 
  (h2 : (75 : ℝ) / 100 * ((88 : ℝ) / 100 * total_students) + (25 : ℝ) / 100 * ((88 : ℝ) / 100 * total_students) = (88 : ℝ) / 100 * total_students) : 
  (88 : ℝ) / 100 * total_students = (22 : ℝ) / (25 / 100) := by
  sorry

end camping_trip_percentage_l1840_184066


namespace probability_two_white_balls_l1840_184075

def total_balls : ℕ := 7 + 8
def white_balls : ℕ := 7
def black_balls : ℕ := 8

theorem probability_two_white_balls :
  (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1)) = 1 / 5 :=
by sorry

end probability_two_white_balls_l1840_184075


namespace specific_bulb_probability_l1840_184009

/-- The number of light bulbs -/
def num_bulbs : ℕ := 4

/-- The number of bulbs to be installed -/
def num_installed : ℕ := 3

/-- The number of ways to arrange n items taken k at a time -/
def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The probability of installing a specific bulb at a specific vertex -/
def probability : ℚ := (permutations (num_bulbs - 1) (num_installed - 1)) / (permutations num_bulbs num_installed)

theorem specific_bulb_probability : probability = 1 / 4 := by sorry

end specific_bulb_probability_l1840_184009


namespace union_of_sets_l1840_184053

theorem union_of_sets (S T : Set ℕ) (h1 : S = {0, 1}) (h2 : T = {0}) : 
  S ∪ T = {0, 1} := by
sorry

end union_of_sets_l1840_184053


namespace stratified_sampling_probability_l1840_184085

/-- The probability of selecting an individual in a sampling method -/
def sampling_probability (m : ℕ) (sample_size : ℕ) : ℚ := 1 / sample_size

theorem stratified_sampling_probability (m : ℕ) (h : m ≥ 3) :
  sampling_probability m 3 = 1 / 3 := by sorry

end stratified_sampling_probability_l1840_184085


namespace computer_price_decrease_l1840_184068

/-- The price of a computer after a certain number of years, given an initial price and a rate of decrease every 3 years. -/
def price_after_years (initial_price : ℝ) (decrease_rate : ℝ) (years : ℕ) : ℝ :=
  initial_price * (1 - decrease_rate) ^ (years / 3)

/-- Theorem stating that the price of a computer initially priced at 8100 yuan,
    decreasing by 1/3 every 3 years, will be 2400 yuan after 9 years. -/
theorem computer_price_decrease :
  price_after_years 8100 (1/3) 9 = 2400 := by
  sorry

end computer_price_decrease_l1840_184068


namespace calculation_proof_l1840_184023

theorem calculation_proof : 1.23 * 67 + 8.2 * 12.3 - 90 * 0.123 = 172.2 := by
  sorry

end calculation_proof_l1840_184023


namespace ellipse_point_properties_l1840_184034

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the point P
structure Point (x₀ y₀ : ℝ) where
  inside_ellipse : 0 < x₀^2 / 2 + y₀^2
  inside_ellipse' : x₀^2 / 2 + y₀^2 < 1

-- Define the line passing through P
def line (x₀ y₀ x y : ℝ) : Prop := x₀ * x / 2 + y₀ * y = 1

-- Theorem statement
theorem ellipse_point_properties {x₀ y₀ : ℝ} (P : Point x₀ y₀) :
  -- 1. Range of |PF₁| + |PF₂|
  ∃ (PF₁ PF₂ : ℝ), 2 ≤ PF₁ + PF₂ ∧ PF₁ + PF₂ < 2 * Real.sqrt 2 ∧
  -- 2. No common points between the line and ellipse
  ∀ (x y : ℝ), line x₀ y₀ x y → ¬ ellipse x y :=
sorry

end ellipse_point_properties_l1840_184034


namespace k_not_determined_l1840_184089

theorem k_not_determined (k r : ℝ) (a : ℝ → ℝ) :
  (∀ r, a r = (k * r)^3) →
  (a (r / 2) = 0.125 * a r) →
  True
:= by sorry

end k_not_determined_l1840_184089


namespace elevator_movement_l1840_184062

/-- Represents the number of floors in a building --/
def TotalFloors : ℕ := 13

/-- Represents the initial floor of the elevator --/
def InitialFloor : ℕ := 9

/-- Represents the first upward movement of the elevator --/
def FirstUpwardMovement : ℕ := 3

/-- Represents the second upward movement of the elevator --/
def SecondUpwardMovement : ℕ := 8

/-- Represents the final floor of the elevator (top floor) --/
def FinalFloor : ℕ := 13

theorem elevator_movement (x : ℕ) : 
  InitialFloor - x + FirstUpwardMovement + SecondUpwardMovement = FinalFloor → 
  x = 7 := by
sorry

end elevator_movement_l1840_184062


namespace tangent_line_triangle_area_l1840_184099

-- Define the function f(x) = x^3 - x + 1
def f (x : ℝ) : ℝ := x^3 - x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem statement
theorem tangent_line_triangle_area :
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  let tangent_line (x : ℝ) : ℝ := m * (x - x₀) + y₀
  let x_intercept : ℝ := -y₀ / m + x₀
  let y_intercept : ℝ := tangent_line 0
  (1/2) * x_intercept * y_intercept = 1/2 :=
sorry

end tangent_line_triangle_area_l1840_184099


namespace max_page_number_with_25_threes_l1840_184001

/-- Counts the occurrences of a specific digit in a number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ := sorry

/-- Counts the total occurrences of a specific digit in numbers from 1 to n -/
def countDigitUpTo (n : ℕ) (d : ℕ) : ℕ := sorry

/-- The maximum page number that can be reached with a given number of '3's -/
def maxPageNumber (threes : ℕ) : ℕ := sorry

theorem max_page_number_with_25_threes :
  maxPageNumber 25 = 139 := by sorry

end max_page_number_with_25_threes_l1840_184001


namespace ball_drawing_properties_l1840_184036

/-- Probability of drawing a red ball on the nth draw -/
def P (n : ℕ) : ℚ :=
  1/2 + 1/(2^(2*n + 1))

/-- Sum of the first n terms of the sequence P_n -/
def S (n : ℕ) : ℚ :=
  (1/6) * (3*n + 1 - (1/4)^n)

theorem ball_drawing_properties :
  (P 2 = 17/32) ∧
  (∀ n : ℕ, 4 * P (n+2) + P n = 5 * P (n+1)) ∧
  (∀ n : ℕ, S n = (1/6) * (3*n + 1 - (1/4)^n)) :=
by sorry

end ball_drawing_properties_l1840_184036


namespace pigeon_win_conditions_l1840_184017

/-- The game result for the pigeon -/
inductive GameResult
| Win
| Lose

/-- Determines the game result for the pigeon given the board size, egg count, and seagull's square size -/
def pigeonWins (n : ℕ) (m : ℕ) (k : ℕ) : GameResult :=
  if k ≤ n ∧ n ≤ 2 * k - 1 ∧ m ≥ k^2 then GameResult.Win
  else if n ≥ 2 * k ∧ m ≥ k^2 + 1 then GameResult.Win
  else GameResult.Lose

/-- Theorem stating the conditions for the pigeon to win -/
theorem pigeon_win_conditions (n : ℕ) (m : ℕ) (k : ℕ) (h : n ≥ k) :
  (k ≤ n ∧ n ≤ 2 * k - 1 → (pigeonWins n m k = GameResult.Win ↔ m ≥ k^2)) ∧
  (n ≥ 2 * k → (pigeonWins n m k = GameResult.Win ↔ m ≥ k^2 + 1)) := by
  sorry

end pigeon_win_conditions_l1840_184017


namespace work_relation_l1840_184015

/-- Represents an isothermal process of a gas -/
structure IsothermalProcess where
  pressure : ℝ → ℝ
  volume : ℝ → ℝ
  work : ℝ

/-- The work done on a gas during an isothermal process -/
def work_done (p : IsothermalProcess) : ℝ := p.work

/-- Condition: The volume in process 1-2 is twice the volume in process 3-4 for any given pressure -/
def volume_relation (p₁₂ p₃₄ : IsothermalProcess) : Prop :=
  ∀ t, p₁₂.volume t = 2 * p₃₄.volume t

/-- Theorem: The work done on the gas in process 1-2 is twice the work done in process 3-4 -/
theorem work_relation (p₁₂ p₃₄ : IsothermalProcess) 
  (h : volume_relation p₁₂ p₃₄) : 
  work_done p₁₂ = 2 * work_done p₃₄ := by
  sorry

end work_relation_l1840_184015


namespace students_passed_test_l1840_184005

/-- Represents the result of a proficiency test -/
structure TestResult where
  total_students : ℕ
  passing_score : ℕ
  passed_students : ℕ

/-- The proficiency test result for the university -/
def university_test : TestResult :=
  { total_students := 1000
  , passing_score := 70
  , passed_students := 600 }

/-- Theorem stating the number of students who passed the test -/
theorem students_passed_test : university_test.passed_students = 600 := by
  sorry

#check students_passed_test

end students_passed_test_l1840_184005


namespace girls_count_l1840_184080

theorem girls_count (boys girls : ℕ) : 
  (boys : ℚ) / girls = 8 / 5 →
  boys + girls = 351 →
  girls = 135 := by
sorry

end girls_count_l1840_184080


namespace poster_height_l1840_184029

/-- Given a rectangular poster with width 4 inches and area 28 square inches, its height is 7 inches. -/
theorem poster_height (width : ℝ) (area : ℝ) (height : ℝ) 
    (h_width : width = 4)
    (h_area : area = 28)
    (h_rect_area : area = width * height) : height = 7 := by
  sorry

end poster_height_l1840_184029


namespace percentage_problem_l1840_184077

theorem percentage_problem (p : ℝ) : 
  (25 / 100 * 840 = p / 100 * 1500 - 15) → p = 15 := by sorry

end percentage_problem_l1840_184077


namespace coin_probability_l1840_184016

theorem coin_probability (p q : ℝ) (hq : q = 1 - p) 
  (h : (Nat.choose 10 5 : ℝ) * p^5 * q^5 = (Nat.choose 10 6 : ℝ) * p^6 * q^4) : 
  p = 6/11 := by
sorry

end coin_probability_l1840_184016


namespace opposites_and_reciprocals_l1840_184050

theorem opposites_and_reciprocals (a b c d : ℝ) 
  (h1 : a = -b) -- a and b are opposites
  (h2 : c * d = 1) -- c and d are reciprocals
  : 3 * (a + b) - 4 * c * d = -4 := by
  sorry

end opposites_and_reciprocals_l1840_184050


namespace system_solutions_l1840_184084

def is_solution (x y z : ℝ) : Prop :=
  x^3 + y^3 + z^3 = 8 ∧
  x^2 + y^2 + z^2 = 22 ∧
  1/x + 1/y + 1/z + z/(x*y) = 0

theorem system_solutions :
  (is_solution 3 2 (-3)) ∧
  (is_solution (-3) 2 3) ∧
  (is_solution 2 3 (-3)) ∧
  (is_solution 2 (-3) 3) :=
by sorry

end system_solutions_l1840_184084


namespace inequality_proof_l1840_184045

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a + b + c = 1) 
  (h5 : ∀ x : ℝ, |x - a| + |x - 1| ≥ (a^2 + b^2 + c^2) / (b + c)) : 
  a ≤ Real.sqrt 2 - 1 := by
sorry

end inequality_proof_l1840_184045


namespace parallel_vectors_imply_m_value_l1840_184058

/-- Given vectors a and b in R², prove that if 2a + b is parallel to a - 2b, then m = -1/2 --/
theorem parallel_vectors_imply_m_value (m : ℝ) :
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (2, -1)
  (∃ (k : ℝ), k ≠ 0 ∧ (2 • a + b) = k • (a - 2 • b)) →
  m = -1/2 := by
  sorry

end parallel_vectors_imply_m_value_l1840_184058


namespace jellybean_problem_l1840_184086

theorem jellybean_problem (initial_quantity : ℝ) : 
  (0.75^3 * initial_quantity = 27) → initial_quantity = 64 := by
  sorry

end jellybean_problem_l1840_184086


namespace triangle_inequality_theorem_not_necessary_condition_l1840_184091

/-- Proposition P: segments of lengths a, b, c can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Proposition Q: a² + b² + c² < 2(ab + bc + ca) -/
def inequality_holds (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 < 2*(a*b + b*c + c*a)

theorem triangle_inequality_theorem (a b c : ℝ) :
  can_form_triangle a b c → inequality_holds a b c :=
sorry

theorem not_necessary_condition :
  ∃ a b c : ℝ, inequality_holds a b c ∧ ¬can_form_triangle a b c :=
sorry

end triangle_inequality_theorem_not_necessary_condition_l1840_184091


namespace book_purchase_ratio_l1840_184004

/-- Represents the number of people who purchased only book A -/
def C : ℕ := 1000

/-- Represents the number of people who purchased both books A and B -/
def AB : ℕ := 500

/-- Represents the total number of people who purchased book A -/
def A : ℕ := C + AB

/-- Represents the total number of people who purchased book B -/
def B : ℕ := AB + (A / 2 - AB)

theorem book_purchase_ratio : (AB : ℚ) / (B - AB : ℚ) = 2 := by sorry

end book_purchase_ratio_l1840_184004


namespace quadratic_solution_difference_squared_l1840_184088

theorem quadratic_solution_difference_squared : 
  ∀ α β : ℝ, α ≠ β → α^2 = 2*α + 2 → β^2 = 2*β + 2 → (α - β)^2 = 12 := by
  sorry

end quadratic_solution_difference_squared_l1840_184088


namespace solution_difference_l1840_184019

-- Define the equation
def equation (r : ℝ) : Prop :=
  (r^2 - 6*r - 20) / (r + 3) = 3*r + 10

-- Define the solutions
def solutions : Set ℝ :=
  {r : ℝ | equation r ∧ r ≠ -3}

-- Theorem statement
theorem solution_difference :
  ∃ (r₁ r₂ : ℝ), r₁ ∈ solutions ∧ r₂ ∈ solutions ∧ r₁ ≠ r₂ ∧ |r₁ - r₂| = 20 :=
by sorry

end solution_difference_l1840_184019


namespace composite_number_probability_l1840_184012

/-- Represents a standard 6-sided die -/
def StandardDie : Type := Fin 6

/-- Represents the special die with only prime numbers -/
def SpecialDie : Type := Fin 3

/-- The total number of possible outcomes when rolling 6 dice -/
def TotalOutcomes : ℕ := 6^5 * 3

/-- The number of non-composite outcomes -/
def NonCompositeOutcomes : ℕ := 4

/-- The probability of getting a composite number -/
def CompositeNumberProbability : ℚ := 5831 / 5832

/-- Theorem stating the probability of getting a composite number when rolling 6 dice
    (5 standard 6-sided dice and 1 special die with prime numbers 2, 3, 5) and
    multiplying their face values -/
theorem composite_number_probability :
  (TotalOutcomes - NonCompositeOutcomes : ℚ) / TotalOutcomes = CompositeNumberProbability :=
sorry

end composite_number_probability_l1840_184012


namespace line_slope_and_inclination_l1840_184043

/-- Given a line l passing through points A(1,2) and B(4, 2+√3), 
    prove its slope and angle of inclination. -/
theorem line_slope_and_inclination :
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (4, 2 + Real.sqrt 3)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  let angle := Real.arctan slope
  slope = Real.sqrt 3 / 3 ∧ angle = π / 6 := by
  sorry


end line_slope_and_inclination_l1840_184043


namespace expansion_equality_l1840_184039

theorem expansion_equality (a b : ℝ) : (a - b) * (-a - b) = b^2 - a^2 := by
  sorry

end expansion_equality_l1840_184039


namespace max_books_purchasable_l1840_184064

theorem max_books_purchasable (book_price : ℚ) (budget : ℚ) : 
  book_price = 15 → budget = 200 → 
    ↑(⌊budget / book_price⌋) = (13 : ℤ) := by
  sorry

end max_books_purchasable_l1840_184064


namespace line_through_points_eq_target_l1840_184055

/-- The equation of a line passing through two points -/
def line_equation (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

/-- The specific points given in the problem -/
def point1 : ℝ × ℝ := (-1, 0)
def point2 : ℝ × ℝ := (0, 1)

/-- The equation we want to prove -/
def target_equation (x y : ℝ) : Prop := x - y + 1 = 0

/-- Theorem stating that the line equation passing through the given points
    is equivalent to the target equation -/
theorem line_through_points_eq_target :
  ∀ x y : ℝ, line_equation point1.1 point1.2 point2.1 point2.2 x y ↔ target_equation x y :=
by sorry

end line_through_points_eq_target_l1840_184055


namespace charlies_data_usage_l1840_184079

/-- Represents Charlie's cell phone data usage problem -/
theorem charlies_data_usage 
  (data_limit : ℝ) 
  (extra_cost_per_gb : ℝ)
  (week1_usage : ℝ)
  (week2_usage : ℝ)
  (week3_usage : ℝ)
  (extra_charge : ℝ)
  (h1 : data_limit = 8)
  (h2 : extra_cost_per_gb = 10)
  (h3 : week1_usage = 2)
  (h4 : week2_usage = 3)
  (h5 : week3_usage = 5)
  (h6 : extra_charge = 120)
  : ∃ (week4_usage : ℝ), 
    week4_usage = 10 ∧ 
    (week1_usage + week2_usage + week3_usage + week4_usage - data_limit) * extra_cost_per_gb = extra_charge :=
sorry

end charlies_data_usage_l1840_184079


namespace product_sum_6545_l1840_184094

theorem product_sum_6545 : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 6545 ∧ 
  a + b = 162 := by
sorry

end product_sum_6545_l1840_184094


namespace min_value_divisible_by_72_l1840_184060

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

theorem min_value_divisible_by_72 (x y : ℕ) (h1 : x ≥ 4) 
  (h2 : is_divisible_by (98348 * 10 + x * 10 + y) 72) : y = 6 := by
  sorry

end min_value_divisible_by_72_l1840_184060


namespace lunch_spending_solution_l1840_184081

def lunch_spending (your_spending : ℚ) : ℚ × ℚ × ℚ × ℚ :=
  (your_spending, 
   your_spending + 15, 
   your_spending - 20, 
   2 * your_spending)

theorem lunch_spending_solution : 
  ∃! (your_spending : ℚ), 
    let (you, friend1, friend2, friend3) := lunch_spending your_spending
    you + friend1 + friend2 + friend3 = 150 ∧
    friend1 = you + 15 ∧
    friend2 = you - 20 ∧
    friend3 = 2 * you :=
by
  sorry

#eval lunch_spending 31

end lunch_spending_solution_l1840_184081


namespace remainder_theorem_l1840_184069

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 25 * k - 1) :
  (n^2 + 3*n + 5) % 25 = 3 := by
  sorry

end remainder_theorem_l1840_184069


namespace binomial_expansion_theorem_l1840_184027

theorem binomial_expansion_theorem (n k : ℕ) (a b : ℝ) (h1 : n ≥ 2) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : a = k * b) (h5 : k > 0) :
  (n * (a - b)^(n-1) * (-b) + n * (n-1) / 2 * (a - b)^(n-2) * (-b)^2 = 0) → n = 2 * k + 1 := by
sorry

end binomial_expansion_theorem_l1840_184027


namespace smallest_of_five_consecutive_sum_100_l1840_184021

theorem smallest_of_five_consecutive_sum_100 (n : ℕ) :
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) → n = 18 := by
  sorry

end smallest_of_five_consecutive_sum_100_l1840_184021


namespace fourth_intersection_point_l1840_184018

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The curve xy = 2 -/
def onCurve (p : Point) : Prop := p.x * p.y = 2

/-- A circle in the 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A point lies on a circle -/
def onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The theorem statement -/
theorem fourth_intersection_point
  (c : Circle)
  (h1 : onCurve ⟨2, 1⟩ ∧ onCircle ⟨2, 1⟩ c)
  (h2 : onCurve ⟨-4, -1/2⟩ ∧ onCircle ⟨-4, -1/2⟩ c)
  (h3 : onCurve ⟨1/2, 4⟩ ∧ onCircle ⟨1/2, 4⟩ c)
  (h4 : ∃ (p : Point), onCurve p ∧ onCircle p c ∧ p ≠ ⟨2, 1⟩ ∧ p ≠ ⟨-4, -1/2⟩ ∧ p ≠ ⟨1/2, 4⟩) :
  ∃ (p : Point), p = ⟨-1, -2⟩ ∧ onCurve p ∧ onCircle p c :=
sorry

end fourth_intersection_point_l1840_184018


namespace symmetric_line_l1840_184013

/-- Given a line l with equation x - y + 1 = 0, prove that its symmetric line l' 
    with respect to x = 2 has the equation x + y - 5 = 0 -/
theorem symmetric_line (x y : ℝ) : 
  (x - y + 1 = 0) → 
  (∃ x' y', x' + y' - 5 = 0 ∧ x' = 4 - x ∧ y' = y) :=
by sorry

end symmetric_line_l1840_184013


namespace marbles_remaining_l1840_184026

theorem marbles_remaining (total_marbles : ℕ) (total_bags : ℕ) (bags_removed : ℕ) : 
  total_marbles = 28 →
  total_bags = 4 →
  bags_removed = 1 →
  total_marbles % total_bags = 0 →
  (total_bags - bags_removed) * (total_marbles / total_bags) = 21 := by
  sorry

end marbles_remaining_l1840_184026


namespace units_digit_factorial_product_l1840_184047

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_factorial_product :
  units_digit (factorial 1 * factorial 2 * factorial 3 * factorial 4) = 8 := by
  sorry

end units_digit_factorial_product_l1840_184047


namespace rectangular_solid_volume_l1840_184083

theorem rectangular_solid_volume 
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 15)
  (h_front : front_area = 20)
  (h_bottom : bottom_area = 12) :
  ∃ (a b c : ℝ), a * b = side_area ∧ b * c = front_area ∧ c * a = bottom_area ∧ a * b * c = 60 :=
by sorry

end rectangular_solid_volume_l1840_184083


namespace lucky_number_2015_l1840_184024

/-- A function that returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ+) : ℕ := sorry

/-- A function that returns true if a positive integer is a "lucky number" (sum of digits is 8) -/
def isLuckyNumber (n : ℕ+) : Prop := sumOfDigits n = 8

/-- A function that returns the nth "lucky number" -/
def nthLuckyNumber (n : ℕ+) : ℕ+ := sorry

theorem lucky_number_2015 : nthLuckyNumber 106 = 2015 := by sorry

end lucky_number_2015_l1840_184024


namespace digit_sum_problem_l1840_184078

theorem digit_sum_problem (X Y Z : ℕ) : 
  X < 10 → Y < 10 → Z < 10 →
  100 * X + 10 * Y + Z + 100 * X + 10 * Y + Z + 10 * Y + Z = 1675 →
  X + Y + Z = 15 := by
sorry

end digit_sum_problem_l1840_184078


namespace bullets_shot_l1840_184031

/-- Proves that given 5 guys with 25 bullets each, if they all shoot an equal number of bullets
    and the remaining bullets equal the initial amount per person,
    then each person must have shot 20 bullets. -/
theorem bullets_shot (num_guys : Nat) (initial_bullets : Nat) (bullets_shot : Nat) : 
  num_guys = 5 →
  initial_bullets = 25 →
  (num_guys * initial_bullets) - (num_guys * bullets_shot) = initial_bullets →
  bullets_shot = 20 := by
  sorry

#check bullets_shot

end bullets_shot_l1840_184031


namespace negative_four_squared_l1840_184096

theorem negative_four_squared : -4^2 = -16 := by
  sorry

end negative_four_squared_l1840_184096


namespace water_pressure_force_on_trapezoidal_dam_l1840_184057

/-- The force of water pressure on a trapezoidal dam -/
theorem water_pressure_force_on_trapezoidal_dam 
  (ρ : Real) (g : Real) (a b h : Real) : 
  ρ = 1000 →
  g = 10 →
  a = 6.9 →
  b = 11.4 →
  h = 5.0 →
  ρ * g * h^2 * (b / 2 - (b - a) * h / (6 * h)) = 1050000 := by
  sorry

end water_pressure_force_on_trapezoidal_dam_l1840_184057


namespace square_field_area_l1840_184061

/-- Proves that a square field with specific barbed wire conditions has an area of 27889 square meters -/
theorem square_field_area (wire_cost_per_meter : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  wire_cost_per_meter = 2.0 →
  total_cost = 1332 →
  gate_width = 1 →
  num_gates = 2 →
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    total_cost = wire_cost_per_meter * (4 * side_length - (↑num_gates * gate_width)) ∧
    side_length^2 = 27889 :=
by sorry

end square_field_area_l1840_184061


namespace school_days_per_week_l1840_184042

theorem school_days_per_week 
  (daily_usage_per_class : ℕ) 
  (weekly_total_usage : ℕ) 
  (num_classes : ℕ) 
  (h1 : daily_usage_per_class = 200)
  (h2 : weekly_total_usage = 9000)
  (h3 : num_classes = 9) :
  weekly_total_usage / (daily_usage_per_class * num_classes) = 5 := by
sorry

end school_days_per_week_l1840_184042


namespace equation_solutions_l1840_184063

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 1 = 8 ↔ x = 3 ∨ x = -3) ∧
  (∀ x : ℝ, (x + 4)^3 = -64 ↔ x = -8) := by
sorry

end equation_solutions_l1840_184063


namespace two_and_half_in_one_and_three_fourths_l1840_184098

theorem two_and_half_in_one_and_three_fourths : 
  (1 + 3/4) / (2 + 1/2) = 7/10 := by sorry

end two_and_half_in_one_and_three_fourths_l1840_184098


namespace unique_digit_subtraction_l1840_184010

theorem unique_digit_subtraction :
  ∃! (I K S : ℕ),
    I < 10 ∧ K < 10 ∧ S < 10 ∧
    100 * K + 10 * I + S ≥ 100 ∧
    100 * S + 10 * I + K ≥ 100 ∧
    100 * S + 10 * K + I ≥ 100 ∧
    (100 * K + 10 * I + S) - (100 * S + 10 * I + K) = 100 * S + 10 * K + I :=
by sorry

end unique_digit_subtraction_l1840_184010


namespace parabola_midpoint_trajectory_l1840_184003

theorem parabola_midpoint_trajectory (x y : ℝ) : 
  let parabola := {(x, y) : ℝ × ℝ | x^2 = 4*y}
  let focus := (0, 1)
  ∀ (p : ℝ × ℝ), p ∈ parabola → 
    let midpoint := ((p.1 + focus.1)/2, (p.2 + focus.2)/2)
    midpoint.1^2 = 2*midpoint.2 - 1 :=
by sorry

end parabola_midpoint_trajectory_l1840_184003


namespace smores_per_person_l1840_184030

/-- Proves that given the conditions of the S'mores problem, each person will eat 3 S'mores -/
theorem smores_per_person 
  (total_people : ℕ) 
  (cost_per_set : ℚ) 
  (smores_per_set : ℕ) 
  (total_cost : ℚ) 
  (h1 : total_people = 8)
  (h2 : cost_per_set = 3)
  (h3 : smores_per_set = 4)
  (h4 : total_cost = 18) :
  (total_cost / cost_per_set * smores_per_set) / total_people = 3 := by
sorry

end smores_per_person_l1840_184030


namespace four_digit_repeat_count_l1840_184025

theorem four_digit_repeat_count : ∀ n : ℕ, (20 ≤ n ∧ n ≤ 99) → (Finset.range 100 \ Finset.range 20).card = 80 := by
  sorry

end four_digit_repeat_count_l1840_184025


namespace modular_inverse_of_5_mod_24_l1840_184046

theorem modular_inverse_of_5_mod_24 :
  ∃ a : ℕ, a < 24 ∧ (5 * a) % 24 = 1 ∧ a = 5 := by
  sorry

end modular_inverse_of_5_mod_24_l1840_184046


namespace special_right_triangle_hypotenuse_l1840_184073

/-- A right triangle with specific properties -/
structure SpecialRightTriangle where
  /-- Length of the shorter leg -/
  short_leg : ℝ
  /-- Length of the longer leg -/
  long_leg : ℝ
  /-- Length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The longer leg is 2 feet longer than twice the shorter leg -/
  leg_relation : long_leg = 2 * short_leg + 2
  /-- The area of the triangle is 96 square feet -/
  area_constraint : (1 / 2) * short_leg * long_leg = 96
  /-- Pythagorean theorem holds -/
  pythagorean : short_leg ^ 2 + long_leg ^ 2 = hypotenuse ^ 2

/-- Theorem: The hypotenuse of the special right triangle is √388 feet -/
theorem special_right_triangle_hypotenuse (t : SpecialRightTriangle) :
  t.hypotenuse = Real.sqrt 388 := by
  sorry

end special_right_triangle_hypotenuse_l1840_184073


namespace marble_problem_l1840_184014

theorem marble_problem (a : ℕ) 
  (angela : ℕ) 
  (brian : ℕ) 
  (caden : ℕ) 
  (daryl : ℕ) 
  (h1 : angela = a) 
  (h2 : brian = 3 * a) 
  (h3 : caden = 2 * brian) 
  (h4 : daryl = 5 * caden) 
  (h5 : angela + brian + caden + daryl = 120) : 
  a = 3 := by
sorry

end marble_problem_l1840_184014


namespace train_speed_l1840_184008

/-- Proves that a train of given length crossing a bridge of given length in a given time has a specific speed in km/hr -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 110)
  (h2 : bridge_length = 265)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed

end train_speed_l1840_184008


namespace f_has_maximum_for_negative_x_l1840_184067

/-- The function f(x) = 2x + 1/x - 1 has a maximum value when x < 0 -/
theorem f_has_maximum_for_negative_x :
  ∃ (M : ℝ), ∀ (x : ℝ), x < 0 → (2 * x + 1 / x - 1 : ℝ) ≤ M := by
  sorry

end f_has_maximum_for_negative_x_l1840_184067


namespace dairy_farm_husk_consumption_l1840_184095

/-- Given a dairy farm scenario where multiple cows eat multiple bags of husk over multiple days,
    this theorem proves that the number of days for one cow to eat one bag is the same as the
    total number of days for all cows to eat all bags. -/
theorem dairy_farm_husk_consumption
  (num_cows : ℕ)
  (num_bags : ℕ)
  (num_days : ℕ)
  (h_cows : num_cows = 46)
  (h_bags : num_bags = 46)
  (h_days : num_days = 46)
  : num_days = (num_days * num_cows) / num_cows :=
by
  sorry

#check dairy_farm_husk_consumption

end dairy_farm_husk_consumption_l1840_184095


namespace draw_three_one_probability_l1840_184006

/-- The probability of drawing exactly 3 balls of one color and 1 of the other color
    from a bin containing 10 black balls and 8 white balls, when 4 balls are drawn at random -/
theorem draw_three_one_probability (black_balls : ℕ) (white_balls : ℕ) (total_draw : ℕ) :
  black_balls = 10 →
  white_balls = 8 →
  total_draw = 4 →
  (Nat.choose black_balls 3 * Nat.choose white_balls 1 +
   Nat.choose black_balls 1 * Nat.choose white_balls 3) /
  Nat.choose (black_balls + white_balls) total_draw = 1 / 2 :=
by sorry

end draw_three_one_probability_l1840_184006


namespace chorus_students_l1840_184041

theorem chorus_students (total : ℕ) (band : ℕ) (both : ℕ) (neither : ℕ) :
  total = 50 →
  band = 26 →
  both = 2 →
  neither = 8 →
  ∃ chorus : ℕ, chorus = 18 ∧ chorus + band - both = total - neither :=
by sorry

end chorus_students_l1840_184041


namespace remainder_of_3_pow_19_mod_10_l1840_184022

theorem remainder_of_3_pow_19_mod_10 : 3^19 % 10 = 7 := by
  sorry

end remainder_of_3_pow_19_mod_10_l1840_184022


namespace stating_min_additional_games_is_ten_l1840_184002

/-- Represents the number of games initially played -/
def initial_games : ℕ := 5

/-- Represents the number of games initially won by the Wolves -/
def initial_wolves_wins : ℕ := 2

/-- Represents the minimum winning percentage required for the Wolves -/
def min_winning_percentage : ℚ := 4/5

/-- 
Determines if a given number of additional games results in the Wolves
winning at least the minimum required percentage of all games
-/
def meets_winning_percentage (additional_games : ℕ) : Prop :=
  (initial_wolves_wins + additional_games : ℚ) / (initial_games + additional_games) ≥ min_winning_percentage

/-- 
Theorem stating that 10 is the minimum number of additional games
needed for the Wolves to meet the minimum winning percentage
-/
theorem min_additional_games_is_ten :
  (∀ n < 10, ¬(meets_winning_percentage n)) ∧ meets_winning_percentage 10 :=
sorry

end stating_min_additional_games_is_ten_l1840_184002


namespace orange_gumdrops_after_replacement_l1840_184076

/-- Represents the number of gumdrops of each color in a jar -/
structure GumdropJar where
  blue : ℕ
  brown : ℕ
  red : ℕ
  yellow : ℕ
  green : ℕ
  orange : ℕ

/-- Calculates the total number of gumdrops in the jar -/
def GumdropJar.total (jar : GumdropJar) : ℕ :=
  jar.blue + jar.brown + jar.red + jar.yellow + jar.green + jar.orange

/-- Represents the initial distribution of gumdrops -/
def initial_jar : GumdropJar :=
  { blue := 40
    brown := 15
    red := 10
    yellow := 5
    green := 20
    orange := 10 }

/-- Theorem stating that after replacing a third of blue gumdrops with orange,
    the number of orange gumdrops will be 23 -/
theorem orange_gumdrops_after_replacement (jar : GumdropJar)
    (h1 : jar = initial_jar)
    (h2 : jar.total = 100)
    (h3 : jar.blue / 3 = 13) :
  (⟨jar.blue - 13, jar.brown, jar.red, jar.yellow, jar.green, jar.orange + 13⟩ : GumdropJar).orange = 23 := by
  sorry

end orange_gumdrops_after_replacement_l1840_184076


namespace prob_diff_colors_is_11_18_l1840_184087

def num_blue : ℕ := 6
def num_yellow : ℕ := 4
def num_red : ℕ := 2
def total_chips : ℕ := num_blue + num_yellow + num_red

def prob_diff_colors : ℚ :=
  (num_blue : ℚ) / total_chips * ((num_yellow + num_red) : ℚ) / total_chips +
  (num_yellow : ℚ) / total_chips * ((num_blue + num_red) : ℚ) / total_chips +
  (num_red : ℚ) / total_chips * ((num_blue + num_yellow) : ℚ) / total_chips

theorem prob_diff_colors_is_11_18 : prob_diff_colors = 11 / 18 := by
  sorry

end prob_diff_colors_is_11_18_l1840_184087


namespace p_equiv_simplified_p_sum_of_squares_of_coefficients_l1840_184020

/-- The polynomial p(x) defined by the given expression -/
def p (x : ℝ) : ℝ := 5 * (x^2 - 3*x + 4) - 8 * (x^3 - x^2 + 2*x - 3)

/-- The simplified form of p(x) -/
def simplified_p (x : ℝ) : ℝ := -8*x^3 + 13*x^2 - 31*x + 44

/-- Theorem stating that p(x) is equivalent to its simplified form -/
theorem p_equiv_simplified_p : p = simplified_p := by sorry

/-- Theorem proving the sum of squares of coefficients of simplified_p is 3130 -/
theorem sum_of_squares_of_coefficients :
  (-8)^2 + 13^2 + (-31)^2 + 44^2 = 3130 := by sorry

end p_equiv_simplified_p_sum_of_squares_of_coefficients_l1840_184020


namespace tree_planting_schedule_l1840_184028

theorem tree_planting_schedule (total_trees : ℕ) (days_saved : ℕ) : 
  total_trees = 960 →
  days_saved = 4 →
  ∃ (original_plan : ℕ),
    original_plan = 120 ∧
    (total_trees / original_plan) - (total_trees / (2 * original_plan)) = days_saved :=
by
  sorry

end tree_planting_schedule_l1840_184028


namespace min_sum_and_inequality_range_l1840_184048

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ 4 * a + b = a * b

-- Define the minimum value of a + b
def min_sum (a b : ℝ) : ℝ := a + b

-- Define the inequality condition
def inequality_condition (a b t : ℝ) : Prop :=
  ∀ x : ℝ, |x - a| + |x - b| ≥ t^2 - 2*t

-- Theorem statement
theorem min_sum_and_inequality_range :
  ∃ a b : ℝ, conditions a b ∧
    (∀ a' b' : ℝ, conditions a' b' → min_sum a b ≤ min_sum a' b') ∧
    min_sum a b = 9 ∧
    (∀ t : ℝ, inequality_condition a b t ↔ -1 ≤ t ∧ t ≤ 3) :=
sorry

end min_sum_and_inequality_range_l1840_184048


namespace fraction_multiplication_l1840_184090

theorem fraction_multiplication : (-1/6 + 3/4 - 5/12) * 48 = 8 := by sorry

end fraction_multiplication_l1840_184090


namespace fifteenth_entry_is_29_l1840_184056

/-- r₁₁(m) is the remainder when m is divided by 11 -/
def r₁₁ (m : ℕ) : ℕ := m % 11

/-- List of nonnegative integers n that satisfy r₁₁(7n) ≤ 5 -/
def satisfying_list : List ℕ :=
  (List.range (100 : ℕ)).filter (fun n => r₁₁ (7 * n) ≤ 5)

theorem fifteenth_entry_is_29 : satisfying_list[14] = 29 := by
  sorry

end fifteenth_entry_is_29_l1840_184056


namespace brandon_skittles_l1840_184092

/-- 
Given Brandon's initial number of Skittles and the number of Skittles he loses,
prove that his final number of Skittles is equal to the difference between
the initial number and the number lost.
-/
theorem brandon_skittles (initial : ℕ) (lost : ℕ) :
  initial ≥ lost → initial - lost = initial - lost :=
by sorry

end brandon_skittles_l1840_184092


namespace travel_time_calculation_l1840_184059

/-- Travel time calculation given distance and average speed -/
theorem travel_time_calculation 
  (distance : ℝ) 
  (average_speed : ℝ) 
  (h1 : distance = 790) 
  (h2 : average_speed = 50) :
  distance / average_speed = 15.8 := by
  sorry

end travel_time_calculation_l1840_184059


namespace area_between_squares_l1840_184065

/-- The area of the region inside a large square but outside a smaller square -/
theorem area_between_squares (large_side : ℝ) (small_side : ℝ) 
  (h_large : large_side = 10)
  (h_small : small_side = 4)
  (h_placement : ∃ (x y : ℝ), x ^ 2 + y ^ 2 = (large_side / 2) ^ 2 ∧ 
                 0 ≤ x ∧ x ≤ small_side ∧ 0 ≤ y ∧ y ≤ small_side) :
  large_side ^ 2 - small_side ^ 2 = 84 := by
sorry

end area_between_squares_l1840_184065


namespace max_area_at_two_l1840_184052

open Real

noncomputable def tangentArea (m : ℝ) : ℝ :=
  if 1 ≤ m ∧ m ≤ 2 then
    4 * (4 - m) * (exp m)
  else if 2 < m ∧ m ≤ 5 then
    8 * (exp m)
  else
    0

theorem max_area_at_two :
  ∀ m : ℝ, 1 ≤ m ∧ m ≤ 5 → tangentArea m ≤ tangentArea 2 := by
  sorry

#check max_area_at_two

end max_area_at_two_l1840_184052


namespace blue_part_length_l1840_184082

/-- Proves that the blue part of a pencil is 3.5 cm long given specific conditions -/
theorem blue_part_length (total_length : ℝ) (black_ratio : ℝ) (white_ratio : ℝ)
  (h1 : total_length = 8)
  (h2 : black_ratio = 1 / 8)
  (h3 : white_ratio = 1 / 2)
  (h4 : black_ratio * total_length + white_ratio * (total_length - black_ratio * total_length) +
    (total_length - black_ratio * total_length - white_ratio * (total_length - black_ratio * total_length)) = total_length) :
  total_length - black_ratio * total_length - white_ratio * (total_length - black_ratio * total_length) = 3.5 := by
  sorry

end blue_part_length_l1840_184082


namespace cylinder_fill_cost_l1840_184093

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The cost to fill a cylinder with gasoline -/
def fillCost (c : Cylinder) (price : ℝ) : ℝ := c.radius^2 * c.height * price

/-- The theorem statement -/
theorem cylinder_fill_cost 
  (canB canN : Cylinder) 
  (h_radius : canN.radius = 2 * canB.radius) 
  (h_height : canN.height = canB.height / 2) 
  (h_half_cost : fillCost { radius := canB.radius, height := canB.height / 2 } (8 / (π * canB.radius^2 * canB.height)) = 4) :
  fillCost canN (8 / (π * canB.radius^2 * canB.height)) = 16 := by
  sorry


end cylinder_fill_cost_l1840_184093


namespace inequality_system_solution_l1840_184000

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x, (x - b < 0 ∧ x + a > 0) ↔ (2 < x ∧ x < 3)) → 
  a + b = 1 := by
sorry

end inequality_system_solution_l1840_184000


namespace always_even_l1840_184051

theorem always_even (m n : ℤ) : 
  ∃ k : ℤ, (2*m + 1)^2 + 3*(2*m + 1)*(2*n + 1) = 2*k := by
sorry

end always_even_l1840_184051


namespace cylinder_surface_area_doubling_l1840_184071

theorem cylinder_surface_area_doubling (r h : ℝ) : 
  r > 0 → h > 0 →
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 300 →
  8 * Real.pi * r^2 + 4 * Real.pi * r * h = 900 →
  h = r →
  2 * Real.pi * r^2 + 2 * Real.pi * r * (2 * h) = 450 :=
by sorry

end cylinder_surface_area_doubling_l1840_184071


namespace quadratic_equation_properties_l1840_184044

theorem quadratic_equation_properties :
  ∃ (p q : ℝ),
    (∀ (r : ℝ), (∀ (x : ℝ), x^2 - (r+7)*x + r + 87 = 0 →
      (∃ (y : ℝ), y ≠ x ∧ y^2 - (r+7)*y + r + 87 = 0) ∧
      x < 0) ↔ p < r ∧ r < q) ∧
    p^2 + q^2 = 8098 :=
by sorry

end quadratic_equation_properties_l1840_184044


namespace triangle_area_l1840_184038

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the area of the triangle is √2 under the following conditions:
    1. b = a*cos(C) + c*cos(B)
    2. CA · CB = 1 (dot product)
    3. c = 2 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = a * Real.cos C + c * Real.cos B →
  a * c * Real.cos B = 1 →
  c = 2 →
  (1/2) * a * b * Real.sin C = Real.sqrt 2 := by
  sorry

end triangle_area_l1840_184038


namespace soap_brand_ratio_l1840_184054

def total_households : ℕ := 240
def households_neither : ℕ := 80
def households_only_A : ℕ := 60
def households_both : ℕ := 25

theorem soap_brand_ratio :
  ∃ (households_only_B : ℕ),
    households_only_A + households_only_B + households_both + households_neither = total_households ∧
    households_only_B / households_both = 3 := by
  sorry

end soap_brand_ratio_l1840_184054


namespace job_completion_time_l1840_184074

/-- Represents the time (in minutes) it takes to complete a job when working together,
    given the individual completion times of two workers. -/
def time_working_together (sylvia_time carla_time : ℚ) : ℚ :=
  1 / (1 / sylvia_time + 1 / carla_time)

/-- Theorem stating that if Sylvia takes 45 minutes and Carla takes 30 minutes to complete a job individually,
    then together they will complete the job in 18 minutes. -/
theorem job_completion_time :
  time_working_together 45 30 = 18 := by
  sorry

#eval time_working_together 45 30

end job_completion_time_l1840_184074


namespace more_stable_performance_l1840_184007

/-- Represents a person's shooting performance -/
structure ShootingPerformance where
  average : ℝ
  variance : ℝ

/-- Determines if the first performance is more stable than the second -/
def isMoreStable (p1 p2 : ShootingPerformance) : Prop :=
  p1.variance < p2.variance

/-- Theorem: Given two shooting performances with the same average,
    the one with smaller variance is more stable -/
theorem more_stable_performance 
  (personA personB : ShootingPerformance)
  (h_same_average : personA.average = personB.average)
  (h_variance_A : personA.variance = 1.4)
  (h_variance_B : personB.variance = 0.6) :
  isMoreStable personB personA :=
sorry

end more_stable_performance_l1840_184007


namespace gold_asymmetric_probability_l1840_184035

/-- Represents a coin --/
inductive Coin
| Gold
| Silver

/-- Represents the symmetry of a coin --/
inductive Symmetry
| Symmetric
| Asymmetric

/-- The probability of getting heads for the asymmetric coin --/
def asymmetricHeadsProbability : ℝ := 0.6

/-- The result of a coin flip --/
inductive FlipResult
| Heads
| Tails

/-- Represents the sequence of coin flips --/
structure FlipSequence where
  goldResult : FlipResult
  silverResult1 : FlipResult
  silverResult2 : FlipResult

/-- The observed flip sequence --/
def observedFlips : FlipSequence := {
  goldResult := FlipResult.Heads,
  silverResult1 := FlipResult.Tails,
  silverResult2 := FlipResult.Heads
}

/-- The probability that the gold coin is asymmetric given the observed flip sequence --/
def probGoldAsymmetric (flips : FlipSequence) : ℝ := sorry

theorem gold_asymmetric_probability :
  probGoldAsymmetric observedFlips = 6/10 := by sorry

end gold_asymmetric_probability_l1840_184035


namespace rectangle_ratio_l1840_184097

/-- Represents the configuration of rectangles around a quadrilateral -/
structure RectangleConfiguration where
  s : ℝ  -- side length of the inner quadrilateral
  x : ℝ  -- shorter side of each rectangle
  y : ℝ  -- longer side of each rectangle
  h1 : s > 0  -- side length is positive
  h2 : x > 0  -- rectangle sides are positive
  h3 : y > 0
  h4 : (s + 2*x)^2 = 4*s^2  -- area relation
  h5 : s + 2*y = 2*s  -- relation for y sides

/-- The ratio of y to x is 1 in the given configuration -/
theorem rectangle_ratio (config : RectangleConfiguration) : config.y / config.x = 1 := by
  sorry

end rectangle_ratio_l1840_184097
