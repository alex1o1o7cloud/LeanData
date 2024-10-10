import Mathlib

namespace multiply_658217_by_99999_l2490_249056

theorem multiply_658217_by_99999 : 658217 * 99999 = 65821034183 := by
  sorry

end multiply_658217_by_99999_l2490_249056


namespace rose_discount_percentage_l2490_249012

theorem rose_discount_percentage (dozen_count : ℕ) (cost_per_rose : ℕ) (final_amount : ℕ) : 
  dozen_count = 5 → 
  cost_per_rose = 6 → 
  final_amount = 288 → 
  (1 - (final_amount : ℚ) / (dozen_count * 12 * cost_per_rose)) * 100 = 20 := by
  sorry

end rose_discount_percentage_l2490_249012


namespace negative_two_inequality_l2490_249042

theorem negative_two_inequality (a b : ℝ) (h : a < b) : -2 * a > -2 * b := by
  sorry

end negative_two_inequality_l2490_249042


namespace parallelogram_smaller_angle_l2490_249057

theorem parallelogram_smaller_angle (x : ℝ) : 
  x > 0 ∧ x < 180 ∧ x + (x + 90) = 180 → x = 45 := by
  sorry

end parallelogram_smaller_angle_l2490_249057


namespace tiangong_survey_method_l2490_249062

/-- Represents the types of survey methods --/
inductive SurveyMethod
  | Comprehensive
  | Sampling

/-- Represents the requirements for the survey --/
structure SurveyRequirements where
  high_precision : Bool
  no_errors_allowed : Bool

/-- Determines the appropriate survey method based on the given requirements --/
def appropriate_survey_method (requirements : SurveyRequirements) : SurveyMethod :=
  if requirements.high_precision && requirements.no_errors_allowed then
    SurveyMethod.Comprehensive
  else
    SurveyMethod.Sampling

theorem tiangong_survey_method :
  let requirements : SurveyRequirements := ⟨true, true⟩
  appropriate_survey_method requirements = SurveyMethod.Comprehensive :=
by sorry

end tiangong_survey_method_l2490_249062


namespace union_of_sets_l2490_249075

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem union_of_sets :
  ∃ x : ℝ, (A x ∩ B x = {9}) → (A x ∪ B x = {-8, -7, -4, 4, 9}) := by
  sorry

end union_of_sets_l2490_249075


namespace sum_of_x_and_y_l2490_249080

theorem sum_of_x_and_y (x y : ℝ) 
  (eq1 : x^2 + x*y + y = 14) 
  (eq2 : y^2 + x*y + x = 28) : 
  x + y = -7 ∨ x + y = 6 := by
sorry

end sum_of_x_and_y_l2490_249080


namespace corey_candy_count_l2490_249072

theorem corey_candy_count (total : ℕ) (difference : ℕ) (corey : ℕ) : 
  total = 66 → difference = 8 → corey + (corey + difference) = total → corey = 29 := by
  sorry

end corey_candy_count_l2490_249072


namespace inequality_proof_l2490_249067

theorem inequality_proof (n : ℕ) (a b c x y z : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
  (h_max : a = max a (max b (max c (max x (max y z)))))
  (h_sum : a + b + c = x + y + z)
  (h_prod : a * b * c = x * y * z) :
  a^n + b^n + c^n ≥ x^n + y^n + z^n := by
sorry

end inequality_proof_l2490_249067


namespace imaginary_part_of_z_squared_l2490_249006

/-- Given a complex number z = 3 - i, prove that the imaginary part of z² is -6 -/
theorem imaginary_part_of_z_squared (z : ℂ) (h : z = 3 - I) : 
  (z^2).im = -6 := by
  sorry

end imaginary_part_of_z_squared_l2490_249006


namespace beatrice_book_cost_l2490_249069

/-- Calculates the total cost of books given the pricing rules and number of books purchased. -/
def book_cost (regular_price : ℕ) (discount : ℕ) (regular_quantity : ℕ) (total_quantity : ℕ) : ℕ :=
  let regular_cost := regular_price * regular_quantity
  let discounted_quantity := total_quantity - regular_quantity
  let discounted_price := regular_price - discount
  let discounted_cost := discounted_quantity * discounted_price
  regular_cost + discounted_cost

/-- Proves that given the specific pricing rules and Beatrice's purchase, the total cost is $370. -/
theorem beatrice_book_cost :
  let regular_price := 20
  let discount := 2
  let regular_quantity := 5
  let total_quantity := 20
  book_cost regular_price discount regular_quantity total_quantity = 370 := by
  sorry

#eval book_cost 20 2 5 20  -- This should output 370

end beatrice_book_cost_l2490_249069


namespace cos_300_degrees_l2490_249038

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end cos_300_degrees_l2490_249038


namespace geometric_sequence_ratio_l2490_249094

/-- Given a geometric sequence {a_n} with common ratio q = 1/2 and sum of first n terms S_n,
    prove that S_4 / a_4 = 15 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence condition
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- Sum formula
  q = (1 : ℝ) / 2 →  -- Common ratio
  S 4 / a 4 = 15 :=
by sorry

end geometric_sequence_ratio_l2490_249094


namespace megan_homework_pages_l2490_249049

def remaining_pages (total_problems completed_problems problems_per_page : ℕ) : ℕ :=
  ((total_problems - completed_problems) + problems_per_page - 1) / problems_per_page

theorem megan_homework_pages : remaining_pages 40 26 7 = 2 := by
  sorry

end megan_homework_pages_l2490_249049


namespace polynomial_equation_solution_l2490_249016

theorem polynomial_equation_solution (x : ℝ) : 
  let p : ℝ → ℝ := λ x => (1 + Real.sqrt 109) / 2
  p (x^2) - p (x^2 - 3) = (p x)^2 + 27 := by
  sorry

end polynomial_equation_solution_l2490_249016


namespace max_diagonals_regular_1000_gon_l2490_249013

/-- The number of sides in the regular polygon -/
def n : ℕ := 1000

/-- The number of different diagonal lengths in a regular n-gon -/
def num_diagonal_lengths (n : ℕ) : ℕ := n / 2

/-- The total number of diagonals in a regular n-gon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals of each length in a regular n-gon -/
def diagonals_per_length (n : ℕ) : ℕ := n

/-- The maximum number of diagonals that can be selected such that among any three of the chosen diagonals, at least two have the same length -/
def max_selected_diagonals (n : ℕ) : ℕ := 2 * diagonals_per_length n

theorem max_diagonals_regular_1000_gon :
  max_selected_diagonals n = 2000 :=
sorry

end max_diagonals_regular_1000_gon_l2490_249013


namespace emily_gave_cards_l2490_249039

/-- The number of cards Martha starts with -/
def initial_cards : ℕ := 3

/-- The number of cards Martha ends up with -/
def final_cards : ℕ := 79

/-- The number of cards Emily gave to Martha -/
def cards_from_emily : ℕ := final_cards - initial_cards

theorem emily_gave_cards : cards_from_emily = 76 := by
  sorry

end emily_gave_cards_l2490_249039


namespace perpendicular_length_l2490_249002

/-- Given a triangle ABC with angle ABC = 135°, AB = 2, and BC = 5,
    if perpendiculars are constructed to AB at A and to BC at C meeting at point D,
    then CD = 5√2 -/
theorem perpendicular_length (A B C D : ℝ × ℝ) : 
  let angleABC : ℝ := 135 * π / 180
  let AB : ℝ := 2
  let BC : ℝ := 5
  ∀ (perpAB : (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0)
    (perpBC : (D.1 - C.1) * (B.1 - C.1) + (D.2 - C.2) * (B.2 - C.2) = 0),
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 5 * Real.sqrt 2 :=
by sorry

end perpendicular_length_l2490_249002


namespace bagel_store_spending_l2490_249064

theorem bagel_store_spending (B D : ℝ) : 
  D = (9/10) * B →
  B = D + 15 →
  B + D = 285 :=
by sorry

end bagel_store_spending_l2490_249064


namespace shirt_cost_l2490_249035

theorem shirt_cost (jeans_cost shirt_cost : ℚ) : 
  (3 * jeans_cost + 2 * shirt_cost = 69) →
  (2 * jeans_cost + 3 * shirt_cost = 76) →
  shirt_cost = 18 := by
  sorry

end shirt_cost_l2490_249035


namespace number_added_problem_l2490_249044

theorem number_added_problem (x : ℝ) : 
  3 * (2 * 5 + x) = 57 → x = 9 := by
sorry

end number_added_problem_l2490_249044


namespace coupon_savings_difference_l2490_249045

/-- Represents the savings from Coupon A (15% off the listed price) -/
def savingsA (price : ℝ) : ℝ := 0.15 * price

/-- Represents the savings from Coupon B ($30 off the listed price) -/
def savingsB : ℝ := 30

/-- Represents the savings from Coupon C (25% off the amount exceeding $100) -/
def savingsC (price : ℝ) : ℝ := 0.25 * (price - 100)

/-- The theorem stating the difference between max and min prices where Coupon A is optimal -/
theorem coupon_savings_difference : 
  ∃ (x y : ℝ), 
    x > 100 ∧ y > 100 ∧
    (∀ p, p > 100 → savingsA p ≥ savingsB → savingsA p ≥ savingsC p → p ≥ x) ∧
    (∀ p, p > 100 → savingsA p ≥ savingsB → savingsA p ≥ savingsC p → p ≤ y) ∧
    y - x = 50 := by
  sorry

end coupon_savings_difference_l2490_249045


namespace no_partition_with_translation_l2490_249077

theorem no_partition_with_translation (A B : Set ℝ) (a : ℝ) : 
  A ⊆ Set.Icc 0 1 → 
  B ⊆ Set.Icc 0 1 → 
  A ∩ B = ∅ → 
  B = {x | ∃ y ∈ A, x = y + a} → 
  False :=
sorry

end no_partition_with_translation_l2490_249077


namespace corner_to_triangle_ratio_is_one_l2490_249093

/-- Represents a square board with four equally spaced lines passing through its center -/
structure Board :=
  (side_length : ℝ)
  (is_square : side_length > 0)

/-- Represents the area of a triangular section in the board -/
def triangular_area (b : Board) : ℝ := sorry

/-- Represents the area of a corner region in the board -/
def corner_area (b : Board) : ℝ := sorry

/-- Theorem stating that the ratio of corner area to triangular area is 1 for a board with side length 2 -/
theorem corner_to_triangle_ratio_is_one :
  ∀ (b : Board), b.side_length = 2 → corner_area b / triangular_area b = 1 := by sorry

end corner_to_triangle_ratio_is_one_l2490_249093


namespace salary_problem_l2490_249023

theorem salary_problem (A B : ℝ) 
  (h1 : A + B = 2000)
  (h2 : 0.05 * A = 0.15 * B) :
  A = 1500 := by
sorry

end salary_problem_l2490_249023


namespace parallelepiped_net_theorem_l2490_249043

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

/-- Represents a net of a parallelepiped -/
structure Net :=
  (squares : ℕ)

/-- Function to unfold a parallelepiped into a net -/
def unfold (p : Parallelepiped) : Net :=
  { squares := 2 * (p.length * p.width + p.length * p.height + p.width * p.height) }

/-- Function to remove one square from a net -/
def remove_square (n : Net) : Net :=
  { squares := n.squares - 1 }

/-- Theorem stating that a 2 × 1 × 1 parallelepiped unfolds into a net with 10 squares,
    and removing one square results in a valid net with 9 squares -/
theorem parallelepiped_net_theorem :
  let p : Parallelepiped := ⟨2, 1, 1⟩
  let full_net : Net := unfold p
  let cut_net : Net := remove_square full_net
  full_net.squares = 10 ∧ cut_net.squares = 9 := by
  sorry


end parallelepiped_net_theorem_l2490_249043


namespace students_who_left_zoo_l2490_249055

/-- Proves the number of students who left the zoo given the initial conditions and remaining individuals --/
theorem students_who_left_zoo 
  (initial_students : Nat) 
  (initial_chaperones : Nat) 
  (initial_teachers : Nat) 
  (remaining_individuals : Nat) 
  (chaperones_who_left : Nat)
  (h1 : initial_students = 20)
  (h2 : initial_chaperones = 5)
  (h3 : initial_teachers = 2)
  (h4 : remaining_individuals = 15)
  (h5 : chaperones_who_left = 2) :
  initial_students - (remaining_individuals - chaperones_who_left - initial_teachers) = 9 := by
  sorry


end students_who_left_zoo_l2490_249055


namespace triangle_side_inequality_l2490_249030

/-- Triangle inequality for side lengths a, b, c -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The main inequality to be proved -/
def main_inequality (a b c : ℝ) : ℝ :=
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a)

theorem triangle_side_inequality (a b c : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (tri : triangle_inequality a b c) : 
  main_inequality a b c ≥ 0 ∧ 
  (main_inequality a b c = 0 ↔ a = b ∧ b = c) := by
  sorry


end triangle_side_inequality_l2490_249030


namespace congruence_characterization_l2490_249041

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- Characterization of integers satisfying the given congruence -/
theorem congruence_characterization (n : ℕ) (h : n > 2) :
  (phi n / 2) % 6 = 1 ↔ 
    n = 3 ∨ n = 4 ∨ n = 6 ∨ 
    (∃ (p k : ℕ), p.Prime ∧ p % 12 = 11 ∧ (n = p^(2*k) ∨ n = 2 * p^(2*k))) :=
  sorry

end congruence_characterization_l2490_249041


namespace simple_interest_solution_l2490_249008

/-- Simple interest calculation -/
def simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) : Prop :=
  let principal := interest / (rate * time / 100)
  principal = 8935

/-- Theorem stating the solution to the simple interest problem -/
theorem simple_interest_solution :
  simple_interest_problem 4020.75 9 5 := by
  sorry

end simple_interest_solution_l2490_249008


namespace triangle_angle_measure_l2490_249029

theorem triangle_angle_measure (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_area : (1 / (4 * Real.sqrt 3)) * (b^2 + c^2 - a^2) = 
            (1 / 2) * b * c * Real.sin (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))) :
  Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) = π / 6 := by
sorry

end triangle_angle_measure_l2490_249029


namespace equation_solution_l2490_249091

theorem equation_solution :
  ∃ x : ℚ, x - 1/2 = 7/8 - 2/3 ∧ x = 17/24 := by sorry

end equation_solution_l2490_249091


namespace sqrt_equation_solution_l2490_249096

theorem sqrt_equation_solution :
  ∀ a b : ℕ+,
    a < b →
    Real.sqrt (1 + Real.sqrt (45 + 16 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b →
    a = 1 ∧ b = 5 := by
  sorry

end sqrt_equation_solution_l2490_249096


namespace maria_towels_result_l2490_249097

/-- The number of towels Maria ended up with after shopping and giving some to her mother. -/
def towels_maria_kept (green_towels white_towels towels_given : ℕ) : ℕ :=
  green_towels + white_towels - towels_given

/-- Theorem stating that Maria ended up with 22 towels. -/
theorem maria_towels_result :
  towels_maria_kept 35 21 34 = 22 := by
  sorry

end maria_towels_result_l2490_249097


namespace northern_village_population_l2490_249095

/-- The number of people in the western village -/
def western_village : ℕ := 7488

/-- The number of people in the southern village -/
def southern_village : ℕ := 6912

/-- The total number of people conscripted from all three villages -/
def total_conscripted : ℕ := 300

/-- The number of people conscripted from the northern village -/
def northern_conscripted : ℕ := 108

/-- The number of people in the northern village -/
def northern_village : ℕ := 4206

theorem northern_village_population :
  (northern_conscripted : ℚ) / total_conscripted =
  northern_village / (northern_village + western_village + southern_village) :=
sorry

end northern_village_population_l2490_249095


namespace rectangles_in_5x5_grid_l2490_249074

/-- The number of dots on each side of the square grid -/
def gridSize : ℕ := 5

/-- The number of different rectangles that can be formed in a square grid -/
def numRectangles (n : ℕ) : ℕ := (n + 1).choose 2 * (n + 1).choose 2

/-- Theorem stating the number of rectangles in a 5x5 grid -/
theorem rectangles_in_5x5_grid : numRectangles gridSize = 225 := by
  sorry

end rectangles_in_5x5_grid_l2490_249074


namespace workday_meeting_percentage_l2490_249046

/-- Represents the duration of a workday in hours -/
def workday_hours : ℝ := 10

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℝ := 60

/-- Calculates the total workday time in minutes -/
def workday_minutes : ℝ := workday_hours * 60

/-- Calculates the duration of the second meeting in minutes -/
def second_meeting_minutes : ℝ := 2 * first_meeting_minutes

/-- Calculates the total time spent in meetings in minutes -/
def total_meeting_minutes : ℝ := first_meeting_minutes + second_meeting_minutes

/-- Theorem: The percentage of the workday spent in meetings is 30% -/
theorem workday_meeting_percentage :
  (total_meeting_minutes / workday_minutes) * 100 = 30 := by sorry

end workday_meeting_percentage_l2490_249046


namespace gdp_growth_time_l2490_249001

theorem gdp_growth_time (initial_gdp : ℝ) (growth_rate : ℝ) (target_gdp : ℝ) :
  initial_gdp = 8000 →
  growth_rate = 0.1 →
  target_gdp = 16000 →
  (∀ n : ℕ, n < 5 → initial_gdp * (1 + growth_rate) ^ n ≤ target_gdp) ∧
  initial_gdp * (1 + growth_rate) ^ 5 > target_gdp :=
by sorry

end gdp_growth_time_l2490_249001


namespace distance_to_school_l2490_249000

/-- The distance from the neighborhood to the school in meters. -/
def school_distance : ℝ := 960

/-- The initial speed of student A in meters per minute. -/
def speed_A_initial : ℝ := 40

/-- The initial speed of student B in meters per minute. -/
def speed_B_initial : ℝ := 60

/-- The speed of student A after increasing in meters per minute. -/
def speed_A_increased : ℝ := 60

/-- The speed of student B after decreasing in meters per minute. -/
def speed_B_decreased : ℝ := 40

/-- The time difference in minutes between A and B's arrival at school. -/
def time_difference : ℝ := 2

/-- Theorem stating that given the conditions, the distance to school is 960 meters. -/
theorem distance_to_school :
  ∀ (distance : ℝ),
  (∃ (time_A time_B : ℝ),
    distance / 2 = speed_A_initial * time_A
    ∧ distance / 2 = speed_A_increased * (time_B - time_A)
    ∧ distance = speed_B_initial * time_A + speed_B_decreased * (time_B - time_A)
    ∧ time_B + time_difference = time_A)
  → distance = school_distance :=
by sorry

end distance_to_school_l2490_249000


namespace gcd_lcm_product_l2490_249026

theorem gcd_lcm_product (a b c : ℕ+) :
  let D := Nat.gcd a (Nat.gcd b c)
  let m := Nat.lcm a (Nat.lcm b c)
  D * m = a * b * c := by
sorry

end gcd_lcm_product_l2490_249026


namespace log_identity_l2490_249048

theorem log_identity (a : ℝ) (h : a = Real.log 3 / Real.log 4) : 
  2^a + 2^(-a) = 4 * Real.sqrt 3 / 3 := by
  sorry

end log_identity_l2490_249048


namespace solution_form_l2490_249021

open Real

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 1 → y > 1 → f x - f y = (y - x) * f (x * y)

/-- The theorem stating that any function satisfying the equation must be of the form k/x -/
theorem solution_form (f : ℝ → ℝ) (h : SatisfiesEquation f) :
    ∃ k : ℝ, ∀ x, x > 1 → f x = k / x := by
  sorry

end solution_form_l2490_249021


namespace max_d_value_l2490_249010

def a (n : ℕ+) : ℕ := 100 + n^2 + 3*n

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (m : ℕ+), d m = 13 ∧ ∀ (n : ℕ+), d n ≤ 13 :=
sorry

end max_d_value_l2490_249010


namespace angle_B_is_pi_over_six_l2490_249025

/-- Triangle ABC with given properties -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The law of sines for a triangle -/
axiom law_of_sines (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- Theorem: In triangle ABC, if angle A = 120°, side a = 2, and side b = (2√3)/3, then angle B = π/6 -/
theorem angle_B_is_pi_over_six (t : Triangle) 
  (h1 : t.A = 2 * π / 3)  -- 120° in radians
  (h2 : t.a = 2)
  (h3 : t.b = 2 * Real.sqrt 3 / 3) :
  t.B = π / 6 := by
  sorry


end angle_B_is_pi_over_six_l2490_249025


namespace arithmetic_sequence_30th_term_l2490_249054

/-- 
Given an arithmetic sequence where:
- a₁ is the first term
- a₂₀ is the 20th term
- a₃₀ is the 30th term
This theorem states that if a₁ = 3 and a₂₀ = 41, then a₃₀ = 61.
-/
theorem arithmetic_sequence_30th_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n m : ℕ, a (n + m) - a n = m * (a 2 - a 1))
  (h_first : a 1 = 3)
  (h_twentieth : a 20 = 41) : 
  a 30 = 61 := by
sorry

end arithmetic_sequence_30th_term_l2490_249054


namespace sum_of_angles_x_and_y_l2490_249007

-- Define a circle divided into 16 equal arcs
def circle_arcs : ℕ := 16

-- Define the span of angle x
def x_span : ℕ := 3

-- Define the span of angle y
def y_span : ℕ := 5

-- Theorem statement
theorem sum_of_angles_x_and_y (x y : Real) :
  (x = (360 / circle_arcs * x_span) / 2) →
  (y = (360 / circle_arcs * y_span) / 2) →
  x + y = 90 := by
  sorry

end sum_of_angles_x_and_y_l2490_249007


namespace max_height_of_smaller_box_l2490_249089

/-- The maximum height of a smaller box that can fit in a larger box --/
theorem max_height_of_smaller_box 
  (large_length : ℝ) (large_width : ℝ) (large_height : ℝ)
  (small_length : ℝ) (small_width : ℝ)
  (max_boxes : ℕ) :
  large_length = 6 →
  large_width = 5 →
  large_height = 4 →
  small_length = 0.6 →
  small_width = 0.5 →
  max_boxes = 1000 →
  ∃ (h : ℝ), h ≤ 0.4 ∧ 
    (max_boxes : ℝ) * small_length * small_width * h ≤ 
    large_length * large_width * large_height :=
by sorry

end max_height_of_smaller_box_l2490_249089


namespace math_competition_team_selection_l2490_249061

theorem math_competition_team_selection (n : ℕ) (k : ℕ) (total : ℕ) (exclude : ℕ) :
  n = 10 →
  k = 3 →
  total = Nat.choose (n - 1) k →
  exclude = Nat.choose (n - 3) k →
  total - exclude = 49 := by
  sorry

end math_competition_team_selection_l2490_249061


namespace reciprocal_sum_one_third_three_fourths_l2490_249053

theorem reciprocal_sum_one_third_three_fourths (x : ℚ) :
  x = (1/3 + 3/4)⁻¹ → x = 12/13 :=
by sorry

end reciprocal_sum_one_third_three_fourths_l2490_249053


namespace dove_hatching_fraction_l2490_249066

theorem dove_hatching_fraction (initial_doves : ℕ) (eggs_per_dove : ℕ) (total_doves_after : ℕ) :
  initial_doves = 20 →
  eggs_per_dove = 3 →
  total_doves_after = 65 →
  (total_doves_after - initial_doves : ℚ) / (initial_doves * eggs_per_dove) = 3 / 4 :=
by sorry

end dove_hatching_fraction_l2490_249066


namespace smallest_n_satisfying_inequality_l2490_249015

theorem smallest_n_satisfying_inequality : ∃ n : ℕ, 
  (∀ m : ℕ, 2006^1003 < m^2006 → n ≤ m) ∧ 2006^1003 < n^2006 ∧ n = 45 := by
  sorry

end smallest_n_satisfying_inequality_l2490_249015


namespace prime_square_mod_six_l2490_249051

theorem prime_square_mod_six (p : ℕ) (h_prime : Nat.Prime p) (h_gt_five : p > 5) :
  p^2 % 6 = 1 := by
  sorry

end prime_square_mod_six_l2490_249051


namespace product_nine_sum_undetermined_l2490_249047

theorem product_nine_sum_undetermined : 
  ∃ (a b c d : ℤ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a * b * c * d = 9 ∧
    ¬∃! (s : ℤ), s = a + b + c + d :=
by sorry

end product_nine_sum_undetermined_l2490_249047


namespace flower_pots_total_cost_l2490_249063

def flower_pots_cost (n : ℕ) (price_difference : ℚ) (largest_pot_price : ℚ) : ℚ :=
  let smallest_pot_price := largest_pot_price - (n - 1 : ℚ) * price_difference
  (n : ℚ) * smallest_pot_price + ((n - 1) * n / 2 : ℚ) * price_difference

theorem flower_pots_total_cost :
  flower_pots_cost 6 (3/10) (85/40) = 33/4 :=
sorry

end flower_pots_total_cost_l2490_249063


namespace johnny_weekly_earnings_l2490_249083

/-- Represents Johnny's dog walking business --/
structure DogWalker where
  dogs_per_walk : ℕ
  pay_30min : ℕ
  pay_60min : ℕ
  hours_per_day : ℕ
  long_walks_per_day : ℕ
  work_days_per_week : ℕ

/-- Calculates Johnny's weekly earnings --/
def weekly_earnings (dw : DogWalker) : ℕ :=
  sorry

/-- Johnny's specific situation --/
def johnny : DogWalker := {
  dogs_per_walk := 3,
  pay_30min := 15,
  pay_60min := 20,
  hours_per_day := 4,
  long_walks_per_day := 6,
  work_days_per_week := 5
}

/-- Theorem stating Johnny's weekly earnings --/
theorem johnny_weekly_earnings : weekly_earnings johnny = 1500 :=
  sorry

end johnny_weekly_earnings_l2490_249083


namespace marble_jar_problem_l2490_249050

theorem marble_jar_problem (g y : ℕ) : 
  (g - 1 : ℚ) / (g + y - 1 : ℚ) = 1 / 8 →
  (g : ℚ) / (g + y - 3 : ℚ) = 1 / 6 →
  g + y = 9 := by
sorry

end marble_jar_problem_l2490_249050


namespace sin_2alpha_value_l2490_249020

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (π/4 - α) = -3/5) : 
  Real.sin (2 * α) = -7/25 := by
  sorry

end sin_2alpha_value_l2490_249020


namespace smaller_circle_area_l2490_249082

-- Define the radius of the smaller circle
def r : ℝ := sorry

-- Define the radius of the larger circle
def R : ℝ := 3 * r

-- Define the length of the common tangent
def tangent_length : ℝ := 5

-- Theorem statement
theorem smaller_circle_area : 
  r^2 + tangent_length^2 = (R - r)^2 → π * r^2 = 25 * π / 3 := by
  sorry

end smaller_circle_area_l2490_249082


namespace triangle_inequality_l2490_249098

/-- Given a triangle with sides a, b, c and angle γ opposite side c,
    prove that c ≥ (a + b) * sin(γ/2) --/
theorem triangle_inequality (a b c γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angle : 0 < γ ∧ γ < π)
  (h_opposite : γ = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) :
  c ≥ (a + b) * Real.sin (γ / 2) := by
  sorry

end triangle_inequality_l2490_249098


namespace mike_ride_length_l2490_249019

-- Define the taxi fare structure
structure TaxiFare where
  start_fee : ℝ
  per_mile_fee : ℝ
  toll_fee : ℝ

-- Define the problem parameters
def mike_fare : TaxiFare := ⟨2.5, 0.25, 0⟩
def annie_fare : TaxiFare := ⟨2.5, 0.25, 5⟩
def annie_miles : ℝ := 14

-- Define the function to calculate the total fare
def total_fare (fare : TaxiFare) (miles : ℝ) : ℝ :=
  fare.start_fee + fare.per_mile_fee * miles + fare.toll_fee

-- Theorem statement
theorem mike_ride_length :
  ∃ (mike_miles : ℝ),
    total_fare mike_fare mike_miles = total_fare annie_fare annie_miles ∧
    mike_miles = 34 := by
  sorry

end mike_ride_length_l2490_249019


namespace count_quadruples_l2490_249081

theorem count_quadruples : 
  let S := {q : Fin 10 × Fin 10 × Fin 10 × Fin 10 | true}
  Fintype.card S = 10000 := by sorry

end count_quadruples_l2490_249081


namespace swimming_pool_area_l2490_249034

/-- Represents a rectangular swimming pool with given properties -/
structure SwimmingPool where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  length_condition : length = 2 * width + 40
  perimeter_condition : perimeter = 2 * (length + width)
  perimeter_value : perimeter = 800

/-- Calculates the area of a rectangular swimming pool -/
def pool_area (pool : SwimmingPool) : ℝ :=
  pool.width * pool.length

/-- Theorem stating that a swimming pool with the given properties has an area of 33600 square feet -/
theorem swimming_pool_area (pool : SwimmingPool) : pool_area pool = 33600 := by
  sorry

end swimming_pool_area_l2490_249034


namespace unique_integer_solution_l2490_249085

theorem unique_integer_solution :
  ∃! z : ℤ, (5 * z ≤ 2 * z - 8) ∧ (-3 * z ≥ 18) ∧ (7 * z ≤ -3 * z - 21) := by
  sorry

end unique_integer_solution_l2490_249085


namespace simplify_expression_l2490_249032

theorem simplify_expression (w : ℝ) : 2*w + 3 - 4*w - 5 + 6*w + 7 - 8*w - 9 = -4*w - 4 := by
  sorry

end simplify_expression_l2490_249032


namespace minutes_to_seconds_l2490_249004

theorem minutes_to_seconds (minutes : Real) (seconds_per_minute : Nat) :
  minutes * seconds_per_minute = 468 → minutes = 7.8 ∧ seconds_per_minute = 60 := by
  sorry

end minutes_to_seconds_l2490_249004


namespace max_phi_symmetric_sine_l2490_249033

/-- Given a function f(x) = 2sin(4x + φ) where φ < 0, if the graph of f(x) is symmetric
    about the line x = π/24, then the maximum value of φ is -2π/3. -/
theorem max_phi_symmetric_sine (φ : ℝ) (hφ : φ < 0) :
  (∀ x : ℝ, 2 * Real.sin (4 * x + φ) = 2 * Real.sin (4 * (π / 12 - x) + φ)) →
  (∃ (φ_max : ℝ), φ_max = -2 * π / 3 ∧ φ ≤ φ_max ∧ ∀ ψ, ψ < 0 → ψ ≤ φ_max) :=
by sorry

end max_phi_symmetric_sine_l2490_249033


namespace mountain_climbing_speed_ratio_l2490_249079

/-- Proves that the ratio of ascending to descending speeds is 3:4 given the conditions of the problem -/
theorem mountain_climbing_speed_ratio 
  (s : ℝ) -- Total distance of the mountain path
  (x : ℝ) -- Jia's ascending speed
  (y : ℝ) -- Yi's descending speed
  (h1 : s > 0) -- The distance is positive
  (h2 : x > 0) -- Ascending speed is positive
  (h3 : y > 0) -- Descending speed is positive
  (h4 : s / x - s / (x + y) = 16) -- Time difference for Jia after meeting
  (h5 : s / y - s / (x + y) = 9) -- Time difference for Yi after meeting
  : x / y = 3 / 4 := by sorry

end mountain_climbing_speed_ratio_l2490_249079


namespace max_value_and_monotonicity_l2490_249071

noncomputable def f (x : ℝ) : ℝ := (3 * Real.log (x + 2) - Real.log (x - 2)) / 2

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x - 1) - f x

theorem max_value_and_monotonicity (h : ∀ x, f x ≥ f 4) :
  (∀ x ∈ Set.Icc 3 7, f x ≤ f 7) ∧
  (∀ a ≥ 1, Monotone (F a) ∧ ∀ a < 1, ¬Monotone (F a)) := by sorry

end max_value_and_monotonicity_l2490_249071


namespace highest_average_speed_l2490_249005

def time_periods : Fin 5 → String
| 0 => "8-9 am"
| 1 => "9-10 am"
| 2 => "10-11 am"
| 3 => "2-3 pm"
| 4 => "3-4 pm"

def distances : Fin 5 → ℝ
| 0 => 50
| 1 => 70
| 2 => 60
| 3 => 80
| 4 => 40

def average_speed (i : Fin 5) : ℝ := distances i

def highest_speed_period : Fin 5 := 3

theorem highest_average_speed :
  ∀ (i : Fin 5), average_speed highest_speed_period ≥ average_speed i :=
by sorry

end highest_average_speed_l2490_249005


namespace min_diff_integers_avg_l2490_249027

theorem min_diff_integers_avg (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧  -- Five different positive integers
  (a + b + c + d + e) / 5 = 5 ∧    -- Average is 5
  ∀ x y z w v : ℕ,                 -- For any other set of 5 different positive integers
    x < y ∧ y < z ∧ z < w ∧ w < v ∧
    (x + y + z + w + v) / 5 = 5 →
    (e - a) ≤ (v - x) →            -- with minimum difference
  (b + c + d) / 3 = 5 :=           -- Average of middle three is 5
by sorry

end min_diff_integers_avg_l2490_249027


namespace creatures_conference_handshakes_l2490_249059

def num_goblins : ℕ := 25
def num_elves : ℕ := 18
def num_fairies : ℕ := 20

def handshakes_among (n : ℕ) : ℕ := n * (n - 1) / 2

def handshakes_between (n : ℕ) (m : ℕ) : ℕ := n * m

def total_handshakes : ℕ :=
  handshakes_among num_goblins +
  handshakes_among num_elves +
  handshakes_between num_goblins num_fairies +
  handshakes_between num_elves num_fairies

theorem creatures_conference_handshakes :
  total_handshakes = 1313 := by sorry

end creatures_conference_handshakes_l2490_249059


namespace waiter_tables_l2490_249086

theorem waiter_tables (customers_per_table : ℕ) (total_customers : ℕ) (h1 : customers_per_table = 8) (h2 : total_customers = 48) :
  total_customers / customers_per_table = 6 := by
  sorry

end waiter_tables_l2490_249086


namespace floor_sqrt_80_l2490_249052

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by sorry

end floor_sqrt_80_l2490_249052


namespace expansion_coefficient_l2490_249065

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion term
def expansionTerm (a : ℝ) (r : ℕ) : ℝ := 
  (-1)^r * a^(8 - r) * binomial 8 r

-- State the theorem
theorem expansion_coefficient (a : ℝ) : 
  (expansionTerm a 4 = 70) → (a = 1 ∨ a = -1) := by sorry

end expansion_coefficient_l2490_249065


namespace last_number_systematic_sampling_l2490_249037

/-- Systematic sampling function -/
def systematicSampling (totalEmployees : ℕ) (sampleSize : ℕ) (firstNumber : ℕ) : ℕ :=
  let interval := totalEmployees / sampleSize
  firstNumber + (sampleSize - 1) * interval

/-- Theorem: Last number in systematic sampling -/
theorem last_number_systematic_sampling :
  systematicSampling 1000 50 15 = 995 := by
  sorry

#eval systematicSampling 1000 50 15

end last_number_systematic_sampling_l2490_249037


namespace symmetric_curve_equation_l2490_249031

-- Define the original curve
def original_curve (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line of symmetry
def symmetry_line : ℝ := 2

-- Define the symmetric point
def symmetric_point (x y : ℝ) : ℝ × ℝ := (4 - x, y)

-- Theorem statement
theorem symmetric_curve_equation :
  ∀ x y : ℝ, original_curve (4 - x) y → y^2 = 16 - 4*x :=
by sorry

end symmetric_curve_equation_l2490_249031


namespace log_inequalities_l2490_249073

/-- Proves the inequalities for logarithms with different bases -/
theorem log_inequalities :
  (Real.log 4 / Real.log 8 > Real.log 4 / Real.log 9) ∧
  (Real.log 4 / Real.log 9 > Real.log 4 / Real.log 10) ∧
  (Real.log 4 / Real.log 0.3 < 0.3^2) ∧
  (0.3^2 < 2^0.4) := by
  sorry

end log_inequalities_l2490_249073


namespace sum_of_x_solutions_is_zero_l2490_249070

theorem sum_of_x_solutions_is_zero :
  ∀ x₁ x₂ : ℝ,
  (∃ y : ℝ, y = 8 ∧ x₁^2 + y^2 = 169) ∧
  (∃ y : ℝ, y = 8 ∧ x₂^2 + y^2 = 169) ∧
  (∀ x : ℝ, (∃ y : ℝ, y = 8 ∧ x^2 + y^2 = 169) → (x = x₁ ∨ x = x₂)) →
  x₁ + x₂ = 0 :=
by sorry

end sum_of_x_solutions_is_zero_l2490_249070


namespace laces_for_shoes_l2490_249003

theorem laces_for_shoes (num_pairs : ℕ) (laces_per_pair : ℕ) (h1 : num_pairs = 26) (h2 : laces_per_pair = 2) :
  num_pairs * laces_per_pair = 52 := by
  sorry

end laces_for_shoes_l2490_249003


namespace parabolas_similar_l2490_249099

/-- Two parabolas are similar if there exists a homothety that transforms one into the other -/
theorem parabolas_similar (a : ℝ) : 
  (∃ (x y : ℝ), y = 2 * x^2 → (∃ (x' y' : ℝ), y' = x'^2 ∧ x' = 2*x ∧ y' = 2*y)) := by
  sorry

#check parabolas_similar

end parabolas_similar_l2490_249099


namespace smallest_non_prime_non_square_no_small_factors_l2490_249084

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬(p ∣ n)

theorem smallest_non_prime_non_square_no_small_factors :
  (∀ m : ℕ, m < 4091 →
    is_prime m ∨
    is_perfect_square m ∨
    ¬(has_no_prime_factor_less_than m 60)) ∧
  ¬(is_prime 4091) ∧
  ¬(is_perfect_square 4091) ∧
  has_no_prime_factor_less_than 4091 60 :=
sorry

end smallest_non_prime_non_square_no_small_factors_l2490_249084


namespace chord_intersection_probability_is_one_twelfth_l2490_249011

/-- The number of points evenly spaced around the circle -/
def n : ℕ := 2020

/-- The probability that two randomly chosen chords intersect -/
def chord_intersection_probability : ℚ := 1 / 12

/-- Theorem stating that the probability of two randomly chosen chords intersecting is 1/12 -/
theorem chord_intersection_probability_is_one_twelfth :
  chord_intersection_probability = 1 / 12 := by sorry

end chord_intersection_probability_is_one_twelfth_l2490_249011


namespace rectangular_solid_pythagorean_l2490_249024

/-- A rectangular solid with given dimensions and body diagonal -/
structure RectangularSolid where
  p : ℝ  -- length
  q : ℝ  -- width
  r : ℝ  -- height
  d : ℝ  -- body diagonal length

/-- The Pythagorean theorem for rectangular solids -/
theorem rectangular_solid_pythagorean (solid : RectangularSolid) :
  solid.p^2 + solid.q^2 + solid.r^2 = solid.d^2 := by
  sorry


end rectangular_solid_pythagorean_l2490_249024


namespace equivalent_workout_l2490_249009

/-- Represents the weight of a single dumbbell in pounds -/
def dumbbell_weight : ℕ → ℕ
| 0 => 15
| 1 => 20
| _ => 0

/-- Calculates the total weight lifted given the dumbbell type and number of repetitions -/
def total_weight (dumbbell_type : ℕ) (repetitions : ℕ) : ℕ :=
  2 * dumbbell_weight dumbbell_type * repetitions

/-- Proves that lifting two 15-pound weights 16 times is equivalent to lifting two 20-pound weights 12 times -/
theorem equivalent_workout : total_weight 0 16 = total_weight 1 12 := by
  sorry

end equivalent_workout_l2490_249009


namespace integer_solution_exists_l2490_249017

theorem integer_solution_exists (a b : ℤ) : ∃ (x y z t : ℤ), 
  (x + y + 2*z + 2*t = a) ∧ (2*x - 2*y + z - t = b) := by
  sorry

end integer_solution_exists_l2490_249017


namespace unique_solution_quadratic_system_l2490_249028

theorem unique_solution_quadratic_system :
  ∃! x : ℚ, (10 * x^2 + 9 * x - 2 = 0) ∧ (30 * x^2 + 59 * x - 6 = 0) ∧ (x = 1/5) :=
by sorry

end unique_solution_quadratic_system_l2490_249028


namespace sum_of_squares_divisible_by_three_l2490_249022

theorem sum_of_squares_divisible_by_three (a b : ℤ) : 
  (3 ∣ a^2 + b^2) → (3 ∣ a) ∧ (3 ∣ b) := by
sorry

end sum_of_squares_divisible_by_three_l2490_249022


namespace fair_coin_same_side_probability_l2490_249060

theorem fair_coin_same_side_probability :
  let n : ℕ := 10
  let p : ℝ := 1 / 2
  (p ^ n : ℝ) = 1 / 1024 := by sorry

end fair_coin_same_side_probability_l2490_249060


namespace strokes_over_par_tom_strokes_over_par_l2490_249036

theorem strokes_over_par (rounds : ℕ) (avg_strokes : ℕ) (par_value : ℕ) : ℕ :=
  let total_strokes := rounds * avg_strokes
  let total_par := rounds * par_value
  total_strokes - total_par

theorem tom_strokes_over_par :
  strokes_over_par 9 4 3 = 9 := by
  sorry

end strokes_over_par_tom_strokes_over_par_l2490_249036


namespace painting_time_proof_l2490_249018

/-- Given a painting job with a total number of rooms, time per room, and rooms already painted,
    calculate the time needed to paint the remaining rooms. -/
def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

/-- Theorem stating that for the given painting scenario, the time to paint the remaining rooms is 16 hours. -/
theorem painting_time_proof :
  time_to_paint_remaining 10 8 8 = 16 := by
  sorry


end painting_time_proof_l2490_249018


namespace existence_of_periodic_even_function_l2490_249088

theorem existence_of_periodic_even_function :
  ∃ f : ℝ → ℝ,
    (f 0 ≠ 0) ∧
    (∀ x : ℝ, f x = f (-x)) ∧
    (∀ x : ℝ, f (x + π) = f x) := by
  sorry

end existence_of_periodic_even_function_l2490_249088


namespace vowel_initials_probability_l2490_249090

/-- Represents the set of possible initials --/
def Initials : Type := Char

/-- The set of all possible initials --/
def all_initials : Finset Initials := sorry

/-- The set of vowel initials --/
def vowel_initials : Finset Initials := sorry

/-- The number of students in the class --/
def class_size : ℕ := 30

/-- No two students have the same initials --/
axiom unique_initials : class_size ≤ Finset.card all_initials

/-- The probability of picking a student with vowel initials --/
def vowel_probability : ℚ := (Finset.card vowel_initials : ℚ) / (Finset.card all_initials : ℚ)

/-- Main theorem: The probability of picking a student with vowel initials is 5/26 --/
theorem vowel_initials_probability : vowel_probability = 5 / 26 := by sorry

end vowel_initials_probability_l2490_249090


namespace triangle_area_problem_l2490_249087

theorem triangle_area_problem (x : ℝ) (h1 : x > 0) 
  (h2 : (1/2) * x * (3*x) = 96) : x = 8 := by
  sorry

end triangle_area_problem_l2490_249087


namespace smallest_positive_root_floor_l2490_249040

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - Real.cos x + 2 * Real.tan x

def is_smallest_positive_root (s : ℝ) : Prop :=
  s > 0 ∧ g s = 0 ∧ ∀ x, 0 < x ∧ x < s → g x ≠ 0

theorem smallest_positive_root_floor :
  ∃ s, is_smallest_positive_root s ∧ ⌊s⌋ = 3 := by sorry

end smallest_positive_root_floor_l2490_249040


namespace smallest_ending_in_9_divisible_by_11_l2490_249078

def ends_in_9 (n : ℕ) : Prop := n % 10 = 9

def is_smallest_ending_in_9_divisible_by_11 (n : ℕ) : Prop :=
  n > 0 ∧ ends_in_9 n ∧ n % 11 = 0 ∧
  ∀ m : ℕ, m > 0 → ends_in_9 m → m % 11 = 0 → m ≥ n

theorem smallest_ending_in_9_divisible_by_11 :
  is_smallest_ending_in_9_divisible_by_11 319 := by
sorry

end smallest_ending_in_9_divisible_by_11_l2490_249078


namespace birds_in_second_tree_l2490_249058

/-- Represents the number of birds in each tree -/
structure TreeBirds where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The initial state of birds in the trees -/
def initial_state : TreeBirds := sorry

/-- The state after birds have flown away -/
def final_state : TreeBirds := sorry

theorem birds_in_second_tree :
  /- Total number of birds initially -/
  initial_state.first + initial_state.second + initial_state.third = 60 →
  /- Birds that flew away from each tree -/
  initial_state.first - final_state.first = 6 →
  initial_state.second - final_state.second = 8 →
  initial_state.third - final_state.third = 4 →
  /- Equal number of birds in each tree after flying away -/
  final_state.first = final_state.second →
  final_state.second = final_state.third →
  /- The number of birds originally in the second tree was 22 -/
  initial_state.second = 22 := by
  sorry

end birds_in_second_tree_l2490_249058


namespace exam_mode_l2490_249076

def scores : List Nat := [7, 10, 9, 8, 7, 9, 9, 8]

def mode (l : List α) [DecidableEq α] : Option α :=
  l.foldl (fun acc x =>
    match acc with
    | none => some x
    | some y => if l.count x > l.count y then some x else some y
  ) none

theorem exam_mode :
  mode scores = some 9 := by
  sorry

end exam_mode_l2490_249076


namespace largest_divisor_of_consecutive_even_product_l2490_249014

theorem largest_divisor_of_consecutive_even_product : ∃ (n : ℕ), 
  (∀ (k : ℕ), k > 0 → 16 ∣ (2*k) * (2*k + 2) * (2*k + 4)) ∧ 
  (∀ (m : ℕ), m > 16 → ∃ (j : ℕ), j > 0 ∧ ¬(m ∣ (2*j) * (2*j + 2) * (2*j + 4))) :=
by sorry

end largest_divisor_of_consecutive_even_product_l2490_249014


namespace imo_inequality_l2490_249068

theorem imo_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end imo_inequality_l2490_249068


namespace math_competition_problem_l2490_249092

theorem math_competition_problem (a b c : ℕ) 
  (h1 : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9)
  (h2 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h3 : (1/a + 1/b + 1/c - 1/a*1/b - 1/a*1/c - 1/b*1/c + 1/a*1/b*1/c : ℚ) = 7/15) :
  ((1 - 1/a) * (1 - 1/b) * (1 - 1/c) : ℚ) = 8/15 := by
  sorry

end math_competition_problem_l2490_249092
