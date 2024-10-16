import Mathlib

namespace NUMINAMATH_CALUDE_number_of_possible_D_values_l2627_262793

-- Define the type for digits (0-9)
def Digit := Fin 10

-- Define the addition operation
def add (a b : Digit) : ℕ := a.val + b.val

-- Define the property of being distinct
def distinct (a b c d : Digit) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- Define the main theorem
theorem number_of_possible_D_values :
  ∃ (s : Finset Digit),
    (∀ d ∈ s, ∃ (a b c e : Digit),
      distinct a b c d ∧
      add a b = d.val ∧
      add c e = d.val) ∧
    s.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_possible_D_values_l2627_262793


namespace NUMINAMATH_CALUDE_vector_dot_product_l2627_262717

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 4)
def c : ℝ × ℝ := (3, 2)

theorem vector_dot_product :
  (2 • a + b) • c = -3 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_l2627_262717


namespace NUMINAMATH_CALUDE_marble_theorem_l2627_262762

/-- Represents a jar containing marbles -/
structure Jar where
  red : ℕ
  yellow : ℕ

/-- The problem setup -/
def marble_problem : Prop :=
  ∃ (jar1 jar2 : Jar),
    -- Ratio conditions
    jar1.red * 2 = jar1.yellow * 7 ∧
    jar2.red * 3 = jar2.yellow * 5 ∧
    -- Total yellow marbles
    jar1.yellow + jar2.yellow = 50 ∧
    -- Total marbles in Jar 2 is 20 more than Jar 1
    jar2.red + jar2.yellow = jar1.red + jar1.yellow + 20 ∧
    -- The conclusion we want to prove
    jar1.red = jar2.red + 2

theorem marble_theorem : marble_problem := by
  sorry

end NUMINAMATH_CALUDE_marble_theorem_l2627_262762


namespace NUMINAMATH_CALUDE_point_on_exponential_graph_tan_value_l2627_262799

theorem point_on_exponential_graph_tan_value :
  ∀ a : ℝ, (3 : ℝ)^a = 9 → Real.tan (a * π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_exponential_graph_tan_value_l2627_262799


namespace NUMINAMATH_CALUDE_average_questions_correct_l2627_262747

def dongwoos_group : List Nat := [16, 22, 30, 26, 18, 20]

theorem average_questions_correct : 
  (List.sum dongwoos_group) / (List.length dongwoos_group) = 22 := by
  sorry

end NUMINAMATH_CALUDE_average_questions_correct_l2627_262747


namespace NUMINAMATH_CALUDE_solutions_of_equation_l2627_262737

theorem solutions_of_equation : 
  {z : ℂ | z^6 - 9*z^3 + 8 = 0} = {2, 1} := by sorry

end NUMINAMATH_CALUDE_solutions_of_equation_l2627_262737


namespace NUMINAMATH_CALUDE_smallest_subtraction_for_divisibility_l2627_262757

theorem smallest_subtraction_for_divisibility :
  ∃! x : ℕ, x ≤ 100 ∧ (427751 - x) % 101 = 0 ∧ ∀ y : ℕ, y < x → (427751 - y) % 101 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_subtraction_for_divisibility_l2627_262757


namespace NUMINAMATH_CALUDE_equation_solution_l2627_262787

theorem equation_solution (x : ℝ) (hx : x ≠ 0) : 
  (9 * x)^18 = (27 * x)^9 + 81 * x ↔ x = 1/3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2627_262787


namespace NUMINAMATH_CALUDE_average_difference_l2627_262709

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 60) : 
  c - a = 30 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l2627_262709


namespace NUMINAMATH_CALUDE_triangle_area_comparison_l2627_262778

theorem triangle_area_comparison : 
  let a : Real := 3
  let b : Real := 5
  let c : Real := 6
  let p : Real := (a + b + c) / 2
  let area_A : Real := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let area_B : Real := (3 * Real.sqrt 14) / 2
  area_A = 2 * Real.sqrt 14 ∧ area_A / area_B = 4 / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_area_comparison_l2627_262778


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l2627_262704

theorem right_triangle_acute_angles (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ -- Angles are positive
  a + b = 90 ∧ -- Sum of acute angles in a right triangle
  b = 4 * a -- Ratio of angles is 4:1
  → a = 18 ∧ b = 72 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l2627_262704


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_l2627_262798

/-- Represents the job titles in the school --/
inductive JobTitle
| Senior
| Intermediate
| Clerk

/-- Represents the school staff distribution --/
structure StaffDistribution where
  total : ℕ
  senior : ℕ
  intermediate : ℕ
  clerk : ℕ
  sum_eq_total : senior + intermediate + clerk = total

/-- Represents the sample distribution --/
structure SampleDistribution where
  total : ℕ
  senior : ℕ
  intermediate : ℕ
  clerk : ℕ
  sum_eq_total : senior + intermediate + clerk = total

/-- Checks if a sample distribution is correctly stratified --/
def isCorrectlySampled (staff : StaffDistribution) (sample : SampleDistribution) : Prop :=
  sample.senior * staff.total = staff.senior * sample.total ∧
  sample.intermediate * staff.total = staff.intermediate * sample.total ∧
  sample.clerk * staff.total = staff.clerk * sample.total

/-- The main theorem to prove --/
theorem stratified_sampling_correct 
  (staff : StaffDistribution)
  (sample : SampleDistribution)
  (h_staff : staff = { 
    total := 150, 
    senior := 45, 
    intermediate := 90, 
    clerk := 15, 
    sum_eq_total := by norm_num
  })
  (h_sample : sample = {
    total := 10,
    senior := 3,
    intermediate := 6,
    clerk := 1,
    sum_eq_total := by norm_num
  }) : 
  isCorrectlySampled staff sample := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_correct_l2627_262798


namespace NUMINAMATH_CALUDE_last_digit_of_product_l2627_262789

theorem last_digit_of_product : (3^101 * 5^89 * 6^127 * 7^139 * 11^79 * 13^67 * 17^53) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_product_l2627_262789


namespace NUMINAMATH_CALUDE_max_xy_collinear_vectors_l2627_262736

def vector_a (x : ℝ) : ℝ × ℝ := (1, x^2)
def vector_b (y : ℝ) : ℝ × ℝ := (-2, y^2 - 2)

def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (k * a.1 = b.1 ∧ k * a.2 = b.2)

theorem max_xy_collinear_vectors (x y : ℝ) :
  collinear (vector_a x) (vector_b y) →
  x * y ≤ Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_xy_collinear_vectors_l2627_262736


namespace NUMINAMATH_CALUDE_largest_n_binomial_sum_l2627_262705

theorem largest_n_binomial_sum : 
  (∃ n : ℕ, (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n) ∧ 
    (∀ m : ℕ, m > n → Nat.choose 10 3 + Nat.choose 10 4 ≠ Nat.choose 11 m)) → 
  (∃ n : ℕ, n = 7 ∧ (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n) ∧ 
    (∀ m : ℕ, m > n → Nat.choose 10 3 + Nat.choose 10 4 ≠ Nat.choose 11 m)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_sum_l2627_262705


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2627_262780

-- Define the variables
def distance_day1 : ℝ := 240
def distance_day2 : ℝ := 420
def time_difference : ℝ := 3

-- Define the theorem
theorem average_speed_calculation :
  ∃ (v : ℝ), v > 0 ∧
  distance_day2 / v = distance_day1 / v + time_difference ∧
  v = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l2627_262780


namespace NUMINAMATH_CALUDE_triangle_circumradius_l2627_262767

/-- Given a triangle with side lengths 8, 15, and 17, its circumradius is 8.5 -/
theorem triangle_circumradius : ∀ (a b c : ℝ), 
  a = 8 ∧ b = 15 ∧ c = 17 →
  (a^2 + b^2 = c^2) →
  (c / 2 = 8.5) := by
  sorry

#check triangle_circumradius

end NUMINAMATH_CALUDE_triangle_circumradius_l2627_262767


namespace NUMINAMATH_CALUDE_bryden_receives_correct_amount_l2627_262748

/-- The amount Bryden receives for selling state quarters -/
def bryden_receive (num_quarters : ℕ) (face_value : ℚ) (collector_offer_percent : ℕ) : ℚ :=
  num_quarters * face_value * (collector_offer_percent : ℚ) / 100

/-- Theorem stating that Bryden receives $31.25 for selling five state quarters -/
theorem bryden_receives_correct_amount :
  bryden_receive 5 (1/4) 2500 = 125/4 :=
sorry

end NUMINAMATH_CALUDE_bryden_receives_correct_amount_l2627_262748


namespace NUMINAMATH_CALUDE_exists_n_order_of_two_congruent_l2627_262776

/-- The order of 2 in n! -/
def v (n : ℕ) : ℕ := sorry

/-- For any positive integers a and m, there exists n > 1 such that v(n) ≡ a (mod m) -/
theorem exists_n_order_of_two_congruent (a m : ℕ+) : ∃ n : ℕ, n > 1 ∧ v n % m = a % m := by
  sorry

end NUMINAMATH_CALUDE_exists_n_order_of_two_congruent_l2627_262776


namespace NUMINAMATH_CALUDE_rectangle_area_l2627_262720

/-- A rectangle with length thrice its breadth and perimeter 88 meters has an area of 363 square meters. -/
theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let perimeter := 2 * (l + b)
  perimeter = 88 → l * b = 363 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2627_262720


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2627_262728

/-- A hyperbola is defined by its standard equation and properties. -/
structure Hyperbola where
  /-- The coefficient of x² in the standard equation -/
  a : ℝ
  /-- The coefficient of y² in the standard equation -/
  b : ℝ
  /-- A point that the hyperbola passes through -/
  point : ℝ × ℝ
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ

/-- The standard equation of a hyperbola holds for its defining point. -/
def satisfies_equation (h : Hyperbola) : Prop :=
  h.a * h.point.1^2 - h.b * h.point.2^2 = 1

/-- The asymptote slope is related to the coefficients in the standard equation. -/
def asymptote_condition (h : Hyperbola) : Prop :=
  h.asymptote_slope^2 = h.a / h.b

/-- The theorem stating the standard equation of the hyperbola. -/
theorem hyperbola_equation (h : Hyperbola)
    (point_cond : h.point = (4, Real.sqrt 3))
    (slope_cond : h.asymptote_slope = 1/2)
    (eq_cond : satisfies_equation h)
    (asym_cond : asymptote_condition h) :
    h.a = 1/4 ∧ h.b = 1 :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2627_262728


namespace NUMINAMATH_CALUDE_parabola_vertex_l2627_262746

/-- The vertex of the parabola y = 4x^2 + 16x + 20 is (-2, 4) -/
theorem parabola_vertex :
  let f : ℝ → ℝ := λ x => 4 * x^2 + 16 * x + 20
  ∃! (m n : ℝ), (∀ x, f x ≥ f m) ∧ f m = n ∧ m = -2 ∧ n = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2627_262746


namespace NUMINAMATH_CALUDE_total_dolls_l2627_262701

theorem total_dolls (hannah_ratio : ℝ) (sister_dolls : ℝ) : 
  hannah_ratio = 5.5 →
  sister_dolls = 8.5 →
  hannah_ratio * sister_dolls + sister_dolls = 55.25 := by
  sorry

end NUMINAMATH_CALUDE_total_dolls_l2627_262701


namespace NUMINAMATH_CALUDE_selection_theorem_l2627_262715

/-- The number of athletes who can play both basketball and soccer -/
def both_sports (total : ℕ) (basketball : ℕ) (soccer : ℕ) : ℕ :=
  basketball + soccer - total

/-- The number of athletes who can only play basketball -/
def only_basketball (total : ℕ) (basketball : ℕ) (soccer : ℕ) : ℕ :=
  basketball - both_sports total basketball soccer

/-- The number of athletes who can only play soccer -/
def only_soccer (total : ℕ) (basketball : ℕ) (soccer : ℕ) : ℕ :=
  soccer - both_sports total basketball soccer

/-- The number of ways to select two athletes for basketball and soccer -/
def selection_ways (total : ℕ) (basketball : ℕ) (soccer : ℕ) : ℕ :=
  let b := both_sports total basketball soccer
  let ob := only_basketball total basketball soccer
  let os := only_soccer total basketball soccer
  Nat.choose b 2 + b * ob + b * os + ob * os

theorem selection_theorem (total basketball soccer : ℕ) 
  (h1 : total = 9) (h2 : basketball = 5) (h3 : soccer = 6) :
  selection_ways total basketball soccer = 28 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l2627_262715


namespace NUMINAMATH_CALUDE_characterize_solution_set_l2627_262707

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∃ (n : ℕ), n ≥ 2 ∧ ∀ (x y : ℝ), f (x + y^n) = f x + (f y)^n

/-- The set of functions that satisfy the functional equation -/
def SolutionSet : Set (ℝ → ℝ) :=
  {f | SatisfiesFunctionalEquation f}

/-- The zero function -/
def ZeroFunction : ℝ → ℝ := fun _ ↦ 0

/-- The identity function -/
def IdentityFunction : ℝ → ℝ := fun x ↦ x

/-- The negation function -/
def NegationFunction : ℝ → ℝ := fun x ↦ -x

/-- The main theorem characterizing the solution set -/
theorem characterize_solution_set :
  SolutionSet = {ZeroFunction, IdentityFunction, NegationFunction} := by sorry

end NUMINAMATH_CALUDE_characterize_solution_set_l2627_262707


namespace NUMINAMATH_CALUDE_prime_divides_mn_minus_one_l2627_262708

theorem prime_divides_mn_minus_one (m n p : ℕ) 
  (h_m_pos : 0 < m) 
  (h_n_pos : 0 < n) 
  (h_p_prime : Nat.Prime p) 
  (h_m_lt_n : m < n) 
  (h_n_lt_p : n < p) 
  (h_p_div_m_sq : p ∣ (m^2 + 1)) 
  (h_p_div_n_sq : p ∣ (n^2 + 1)) : 
  p ∣ (m * n - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_mn_minus_one_l2627_262708


namespace NUMINAMATH_CALUDE_troy_computer_worth_l2627_262769

/-- The worth of Troy's new computer -/
def new_computer_worth (initial_savings selling_price additional_needed : ℕ) : ℕ :=
  initial_savings + selling_price + additional_needed

/-- Theorem: The worth of Troy's new computer is $80 -/
theorem troy_computer_worth :
  new_computer_worth 50 20 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_troy_computer_worth_l2627_262769


namespace NUMINAMATH_CALUDE_point_coordinate_sum_l2627_262738

/-- Given two points A and B, where A is at (2, 1) and B is on the line y = 6,
    and the slope of segment AB is 4/5, prove that the sum of the x- and y-coordinates of B is 14.25 -/
theorem point_coordinate_sum (B : ℝ × ℝ) : 
  B.2 = 6 → -- B is on the line y = 6
  (B.2 - 1) / (B.1 - 2) = 4 / 5 → -- slope of AB is 4/5
  B.1 + B.2 = 14.25 := by sorry

end NUMINAMATH_CALUDE_point_coordinate_sum_l2627_262738


namespace NUMINAMATH_CALUDE_jessie_muffins_theorem_l2627_262740

/-- The number of muffins made when Jessie and her friends each receive an equal amount -/
def total_muffins (num_friends : ℕ) (muffins_per_person : ℕ) : ℕ :=
  (num_friends + 1) * muffins_per_person

/-- Theorem stating that when Jessie has 4 friends and each person gets 4 muffins, the total is 20 -/
theorem jessie_muffins_theorem :
  total_muffins 4 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jessie_muffins_theorem_l2627_262740


namespace NUMINAMATH_CALUDE_system_solutions_l2627_262703

def has_solution (a : ℝ) : Prop :=
  ∃ (x y : ℝ), y > 0 ∧ x ≥ 0 ∧ y - 2 = a * (x - 4) ∧ 2 * x / (|y| + y) = Real.sqrt x

theorem system_solutions (a : ℝ) :
  (a ≤ 0 ∨ a = 1/4) →
    (has_solution a ∧
     ∃ (x y : ℝ), (x = 0 ∧ y = 2 - 4*a) ∨ (x = 4 ∧ y = 2)) ∧
  ((0 < a ∧ a < 1/4) ∨ (1/4 < a ∧ a < 1/2)) →
    (has_solution a ∧
     ∃ (x y : ℝ), (x = 0 ∧ y = 2 - 4*a) ∨ (x = 4 ∧ y = 2) ∨ (x = ((1-2*a)/a)^2 ∧ y = (1-2*a)/a)) ∧
  (a ≥ 1/2) →
    (has_solution a ∧
     ∃ (x y : ℝ), x = 4 ∧ y = 2) :=
by sorry


end NUMINAMATH_CALUDE_system_solutions_l2627_262703


namespace NUMINAMATH_CALUDE_circle_xy_bounds_l2627_262788

/-- The circle defined by x² + y² - 4x - 4y + 6 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 4*p.2 + 6 = 0}

/-- The product function xy for points on the circle -/
def xy_product (p : ℝ × ℝ) : ℝ := p.1 * p.2

theorem circle_xy_bounds :
  (∃ p ∈ Circle, ∀ q ∈ Circle, xy_product q ≤ xy_product p) ∧
  (∃ p ∈ Circle, ∀ q ∈ Circle, xy_product p ≤ xy_product q) ∧
  (∃ p ∈ Circle, xy_product p = 9) ∧
  (∃ p ∈ Circle, xy_product p = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_xy_bounds_l2627_262788


namespace NUMINAMATH_CALUDE_x_plus_reciprocal_x_l2627_262741

theorem x_plus_reciprocal_x (x : ℝ) (hx_pos : x > 0) 
  (hx_eq : x^10 + x^5 + 1/x^5 + 1/x^10 = 15250) : 
  x + 1/x = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_reciprocal_x_l2627_262741


namespace NUMINAMATH_CALUDE_exam_question_count_exam_question_count_proof_l2627_262726

theorem exam_question_count (marks_per_correct : ℕ) (marks_per_incorrect : ℕ) 
  (total_marks : ℕ) (correct_answers : ℕ) (total_questions : ℕ) : Prop :=
  (marks_per_correct = 4) →
  (marks_per_incorrect = 1) →
  (total_marks = 120) →
  (correct_answers = 40) →
  (marks_per_correct * correct_answers - marks_per_incorrect * (total_questions - correct_answers) = total_marks) →
  total_questions = 80

-- Proof
theorem exam_question_count_proof : 
  exam_question_count 4 1 120 40 80 := by sorry

end NUMINAMATH_CALUDE_exam_question_count_exam_question_count_proof_l2627_262726


namespace NUMINAMATH_CALUDE_ratio_sum_to_base_l2627_262739

theorem ratio_sum_to_base (x y : ℝ) (h : y / x = 3 / 7) : (x + y) / x = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_to_base_l2627_262739


namespace NUMINAMATH_CALUDE_shirt_original_price_l2627_262713

/-- Proves that if a shirt's price after a 15% discount is $68, then its original price was $80. -/
theorem shirt_original_price (discounted_price : ℝ) (discount_rate : ℝ) : 
  discounted_price = 68 → discount_rate = 0.15 → 
  discounted_price = (1 - discount_rate) * 80 := by
  sorry

end NUMINAMATH_CALUDE_shirt_original_price_l2627_262713


namespace NUMINAMATH_CALUDE_work_completion_l2627_262716

/-- The number of men in the first group -/
def men_first : ℕ := 15

/-- The number of days for the first group to complete the work -/
def days_first : ℚ := 25

/-- The number of days for the second group to complete the work -/
def days_second : ℚ := 37/2

/-- The total amount of work in man-days -/
def total_work : ℚ := men_first * days_first

/-- The number of men in the second group -/
def men_second : ℕ := 20

theorem work_completion :
  (men_second : ℚ) * days_second = total_work :=
sorry

end NUMINAMATH_CALUDE_work_completion_l2627_262716


namespace NUMINAMATH_CALUDE_square_sum_power_of_two_l2627_262795

theorem square_sum_power_of_two (x y z : ℕ) (h : x^2 + y^2 = 2^z) :
  ∃ n : ℕ, x = 2^n ∧ y = 2^n ∧ z = 2*n + 1 := by
sorry

end NUMINAMATH_CALUDE_square_sum_power_of_two_l2627_262795


namespace NUMINAMATH_CALUDE_power_six_mod_eleven_l2627_262719

theorem power_six_mod_eleven : 6^2045 % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_six_mod_eleven_l2627_262719


namespace NUMINAMATH_CALUDE_sequence_2011th_term_l2627_262733

theorem sequence_2011th_term (a : ℕ → ℝ) 
  (h1 : a 1 = 0)
  (h2 : ∀ n : ℕ, a n + a (n + 1) = 2) : 
  a 2011 = 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_2011th_term_l2627_262733


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l2627_262771

theorem sum_of_cubes_of_roots (P : ℝ → ℝ) (x₁ x₂ x₃ : ℝ) :
  P = (fun x ↦ x^3 - 3*x - 1) →
  P x₁ = 0 →
  P x₂ = 0 →
  P x₃ = 0 →
  x₁^3 + x₂^3 + x₃^3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l2627_262771


namespace NUMINAMATH_CALUDE_pencil_count_l2627_262745

/-- The total number of pencils after adding more to an initial amount -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem: Given 33 initial pencils and 27 added pencils, the total is 60 -/
theorem pencil_count : total_pencils 33 27 = 60 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l2627_262745


namespace NUMINAMATH_CALUDE_evaluate_expression_l2627_262756

theorem evaluate_expression : (1 / ((5^2)^4)) * 5^11 * 2 = 250 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2627_262756


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l2627_262744

/-- A geometric sequence with a_2 = 2 and a_8 = 128 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  a 2 = 2 ∧ a 8 = 128

/-- The general formula for the sequence -/
def GeneralFormula (a : ℕ → ℝ) : Prop :=
  (∀ n, a n = 2^(n-1)) ∨ (∀ n, a n = -(-2)^(n-1))

/-- The sum of the first n terms -/
def SumFormula (S : ℕ → ℝ) : Prop :=
  (∀ n, S n = 2^n - 1) ∨ (∀ n, S n = (1/3) * ((-2)^n - 1))

theorem geometric_sequence_properties
  (a : ℕ → ℝ) (S : ℕ → ℝ) (h : GeometricSequence a) :
  GeneralFormula a ∧ SumFormula S :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l2627_262744


namespace NUMINAMATH_CALUDE_remainder_equality_l2627_262754

theorem remainder_equality (A B D S T u v : ℕ) 
  (h1 : A > B)
  (h2 : S = A % D)
  (h3 : T = B % D)
  (h4 : u = (A + B) % D)
  (h5 : v = (S + T) % D) :
  u = v := by
  sorry

end NUMINAMATH_CALUDE_remainder_equality_l2627_262754


namespace NUMINAMATH_CALUDE_octahedral_die_red_faces_l2627_262721

theorem octahedral_die_red_faces (n : ℕ) (k : ℕ) (opposite_pairs : ℕ) :
  n = 8 →
  k = 2 →
  opposite_pairs = 4 →
  Nat.choose n k - opposite_pairs = 24 :=
by sorry

end NUMINAMATH_CALUDE_octahedral_die_red_faces_l2627_262721


namespace NUMINAMATH_CALUDE_prob_ace_ten_king_standard_deck_l2627_262796

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (aces : ℕ)
  (tens : ℕ)
  (kings : ℕ)

/-- The probability of drawing an Ace, then a Ten, then a King without replacement -/
def prob_ace_ten_king (d : Deck) : ℚ :=
  (d.aces : ℚ) / d.total_cards *
  (d.tens : ℚ) / (d.total_cards - 1) *
  (d.kings : ℚ) / (d.total_cards - 2)

/-- Theorem stating the probability of drawing an Ace, then a Ten, then a King from a standard deck -/
theorem prob_ace_ten_king_standard_deck : 
  prob_ace_ten_king {total_cards := 52, aces := 4, tens := 4, kings := 4} = 2 / 16575 := by
  sorry


end NUMINAMATH_CALUDE_prob_ace_ten_king_standard_deck_l2627_262796


namespace NUMINAMATH_CALUDE_find_V_l2627_262722

-- Define the relationship between R, V, and W
def relationship (R V W : ℚ) : Prop :=
  ∃ c : ℚ, c ≠ 0 ∧ R * W = c * V

-- State the theorem
theorem find_V : 
  (∃ R₀ V₀ W₀ : ℚ, R₀ = 6 ∧ V₀ = 2 ∧ W₀ = 3 ∧ relationship R₀ V₀ W₀) →
  (∃ R₁ V₁ W₁ : ℚ, R₁ = 25 ∧ W₁ = 5 ∧ relationship R₁ V₁ W₁ ∧ V₁ = 125 / 9) :=
by sorry

end NUMINAMATH_CALUDE_find_V_l2627_262722


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l2627_262779

theorem merchant_profit_percentage (cost_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : 
  markup_percentage = 75 →
  discount_percentage = 40 →
  cost_price > 0 →
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 5 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l2627_262779


namespace NUMINAMATH_CALUDE_factory_sampling_is_systematic_l2627_262712

/-- Represents a sampling method used in quality control --/
inductive SamplingMethod
| Systematic
| Random
| Stratified
| Cluster

/-- Represents a factory production line --/
structure ProductionLine where
  conveyor_belt : Bool
  inspection_interval : ℕ
  fixed_position : Bool

/-- Determines the sampling method based on the production line characteristics --/
def determine_sampling_method (line : ProductionLine) : SamplingMethod :=
  if line.conveyor_belt ∧ line.inspection_interval > 0 ∧ line.fixed_position then
    SamplingMethod.Systematic
  else
    SamplingMethod.Random -- Default to Random for simplicity

/-- Theorem stating that the described sampling method is Systematic Sampling --/
theorem factory_sampling_is_systematic (factory : ProductionLine) 
  (h1 : factory.conveyor_belt = true)
  (h2 : factory.inspection_interval = 10)
  (h3 : factory.fixed_position = true) :
  determine_sampling_method factory = SamplingMethod.Systematic := by
  sorry


end NUMINAMATH_CALUDE_factory_sampling_is_systematic_l2627_262712


namespace NUMINAMATH_CALUDE_candy_count_third_set_l2627_262781

/-- Represents a set of candies with hard candies, chocolates, and gummy candies -/
structure CandySet where
  hard : ℕ
  chocolate : ℕ
  gummy : ℕ

/-- The total number of candies in a set -/
def CandySet.total (s : CandySet) : ℕ := s.hard + s.chocolate + s.gummy

theorem candy_count_third_set (set1 set2 set3 : CandySet) : 
  /- Total number of each type is equal across all sets -/
  (set1.hard + set2.hard + set3.hard = set1.chocolate + set2.chocolate + set3.chocolate) ∧
  (set1.hard + set2.hard + set3.hard = set1.gummy + set2.gummy + set3.gummy) ∧
  /- First set conditions -/
  (set1.chocolate = set1.gummy) ∧
  (set1.hard = set1.chocolate + 7) ∧
  /- Second set conditions -/
  (set2.hard = set2.chocolate) ∧
  (set2.gummy = set2.hard - 15) ∧
  /- Third set condition -/
  (set3.hard = 0) →
  /- Conclusion: total number of candies in the third set is 29 -/
  set3.total = 29 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_third_set_l2627_262781


namespace NUMINAMATH_CALUDE_train_crossing_bridge_time_l2627_262753

/-- Represents the problem of a train crossing a bridge -/
def TrainCrossingBridge (train_length : ℝ) (train_speed_kmph : ℝ) (bridge_length : ℝ) : Prop :=
  let total_distance : ℝ := train_length + bridge_length
  let train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
  let crossing_time : ℝ := total_distance / train_speed_mps
  crossing_time = 72.5

/-- Theorem stating that a train 250 meters long, running at 72 kmph, 
    takes 72.5 seconds to cross a bridge 1,200 meters in length -/
theorem train_crossing_bridge_time :
  TrainCrossingBridge 250 72 1200 := by
  sorry

#check train_crossing_bridge_time

end NUMINAMATH_CALUDE_train_crossing_bridge_time_l2627_262753


namespace NUMINAMATH_CALUDE_cookie_theorem_l2627_262790

/-- The number of combinations when selecting 8 cookies from 4 types, with at least one of each type -/
def cookieCombinations : ℕ := 46

/-- The function that calculates the number of combinations -/
def calculateCombinations (totalCookies : ℕ) (cookieTypes : ℕ) : ℕ :=
  sorry

theorem cookie_theorem :
  calculateCombinations 8 4 = cookieCombinations :=
by sorry

end NUMINAMATH_CALUDE_cookie_theorem_l2627_262790


namespace NUMINAMATH_CALUDE_bananas_left_l2627_262731

/-- The number of bananas in a dozen -/
def dozen : ℕ := 12

/-- The number of bananas Anthony ate -/
def eaten : ℕ := 2

/-- Theorem: The number of bananas left is 10 -/
theorem bananas_left : dozen - eaten = 10 := by
  sorry

end NUMINAMATH_CALUDE_bananas_left_l2627_262731


namespace NUMINAMATH_CALUDE_betty_strawberries_l2627_262765

/-- Proves that Betty picked 16 strawberries given the conditions of the problem -/
theorem betty_strawberries : ∃ (B N : ℕ),
  let M := B + 20
  let total_strawberries := B + M + N
  let jars := 40 / 4
  let strawberries_per_jar := 7
  B + 20 = 2 * N ∧
  total_strawberries = jars * strawberries_per_jar ∧
  B = 16 := by
  sorry


end NUMINAMATH_CALUDE_betty_strawberries_l2627_262765


namespace NUMINAMATH_CALUDE_rhombus_diagonal_sum_l2627_262763

/-- A rhombus with specific properties -/
structure Rhombus where
  longer_diagonal : ℝ
  shorter_diagonal : ℝ
  area : ℝ
  diagonal_diff : longer_diagonal - shorter_diagonal = 4
  area_eq : area = 6
  positive_diagonals : longer_diagonal > 0 ∧ shorter_diagonal > 0

/-- The sum of diagonals in a rhombus with given properties is 8 -/
theorem rhombus_diagonal_sum (r : Rhombus) : r.longer_diagonal + r.shorter_diagonal = 8 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_sum_l2627_262763


namespace NUMINAMATH_CALUDE_lollipop_challenge_l2627_262770

def joann_lollipops (n : ℕ) : ℕ := 8 + 2 * n

def tom_lollipops (n : ℕ) : ℕ := 5 * 2^(n - 1)

def total_lollipops : ℕ := 
  (Finset.range 7).sum joann_lollipops + (Finset.range 7).sum tom_lollipops

theorem lollipop_challenge : total_lollipops = 747 := by
  sorry

end NUMINAMATH_CALUDE_lollipop_challenge_l2627_262770


namespace NUMINAMATH_CALUDE_only_zero_solution_l2627_262723

theorem only_zero_solution (x y z : ℤ) :
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_only_zero_solution_l2627_262723


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2627_262761

/-- A quadratic function passing through two given points -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + 4

theorem quadratic_function_properties :
  ∃ (a b : ℝ),
    (QuadraticFunction a b (-1) = 3) ∧
    (QuadraticFunction a b 2 = 18) ∧
    (a = 2 ∧ b = 3) ∧
    (let vertex_x := -b / (2 * a)
     let vertex_y := QuadraticFunction a b vertex_x
     vertex_x = -3/4 ∧ vertex_y = 23/8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2627_262761


namespace NUMINAMATH_CALUDE_simplify_fraction_l2627_262772

theorem simplify_fraction : 18 * (8 / 12) * (1 / 27) = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2627_262772


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l2627_262714

-- Define the logarithm functions
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log_sum_equals_two : lg 0.01 + log2 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l2627_262714


namespace NUMINAMATH_CALUDE_walmart_sales_l2627_262792

theorem walmart_sales (thermometer_price hot_water_bottle_price total_sales : ℕ)
  (thermometer_ratio : ℕ) (h1 : thermometer_price = 2)
  (h2 : hot_water_bottle_price = 6) (h3 : total_sales = 1200)
  (h4 : thermometer_ratio = 7) :
  ∃ (thermometers hot_water_bottles : ℕ),
    thermometer_price * thermometers + hot_water_bottle_price * hot_water_bottles = total_sales ∧
    thermometers = thermometer_ratio * hot_water_bottles ∧
    hot_water_bottles = 60 := by
  sorry

end NUMINAMATH_CALUDE_walmart_sales_l2627_262792


namespace NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l2627_262710

def trailing_zeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_500_trailing_zeroes :
  trailing_zeroes 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l2627_262710


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l2627_262732

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q →
  (a 3)^2 + 4 * (a 3) + 1 = 0 →
  (a 7)^2 + 4 * (a 7) + 1 = 0 →
  a 5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l2627_262732


namespace NUMINAMATH_CALUDE_range_of_a_l2627_262758

theorem range_of_a (a : ℝ) : 
  ((∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∨ 
   (∃ x : ℝ, x^2 - x + a = 0)) ∧ 
  ¬((∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧ 
    (∃ x : ℝ, x^2 - x + a = 0)) ↔ 
  a < 0 ∨ (1/4 < a ∧ a < 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2627_262758


namespace NUMINAMATH_CALUDE_initial_green_balls_l2627_262743

theorem initial_green_balls (pink_balls : ℕ) (added_green_balls : ℕ) :
  pink_balls = 23 →
  added_green_balls = 14 →
  ∃ initial_green_balls : ℕ, 
    initial_green_balls + added_green_balls = pink_balls ∧
    initial_green_balls = 9 :=
by sorry

end NUMINAMATH_CALUDE_initial_green_balls_l2627_262743


namespace NUMINAMATH_CALUDE_boys_not_adjacent_girls_adjacent_girls_not_at_ends_l2627_262766

/-- The number of boys in the group -/
def num_boys : Nat := 3

/-- The number of girls in the group -/
def num_girls : Nat := 2

/-- The total number of people in the group -/
def total_people : Nat := num_boys + num_girls

/-- Calculates the number of ways to arrange n distinct objects -/
def permutations (n : Nat) : Nat := Nat.factorial n

/-- Theorem stating the number of ways boys are not adjacent -/
theorem boys_not_adjacent : 
  permutations num_girls * permutations num_boys = 12 := by sorry

/-- Theorem stating the number of ways girls are adjacent -/
theorem girls_adjacent : 
  permutations (total_people - num_girls + 1) * permutations num_girls = 48 := by sorry

/-- Theorem stating the number of ways girls are not at the ends -/
theorem girls_not_at_ends : 
  (total_people - 2) * permutations num_boys = 36 := by sorry

end NUMINAMATH_CALUDE_boys_not_adjacent_girls_adjacent_girls_not_at_ends_l2627_262766


namespace NUMINAMATH_CALUDE_z_value_theorem_l2627_262785

theorem z_value_theorem (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ y ≠ x) 
  (eq : 1 / x - 1 / y = 1 / z) : z = (x * y) / (y - x) := by
  sorry

end NUMINAMATH_CALUDE_z_value_theorem_l2627_262785


namespace NUMINAMATH_CALUDE_intersection_points_sum_l2627_262783

theorem intersection_points_sum (m : ℕ) (h : m = 17) : 
  ∃ (x : ℕ), 
    (∀ y : ℕ, (y ≡ 6*x + 3 [MOD m] ↔ y ≡ 13*x + 8 [MOD m])) ∧ 
    x = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_sum_l2627_262783


namespace NUMINAMATH_CALUDE_abigail_initial_fences_l2627_262735

/-- The number of fences Abigail can build in 8 hours -/
def fences_in_8_hours : ℕ := 8 * 60 / 30

/-- The total number of fences after 8 hours of building -/
def total_fences : ℕ := 26

/-- The number of fences Abigail built initially -/
def initial_fences : ℕ := total_fences - fences_in_8_hours

theorem abigail_initial_fences : initial_fences = 10 := by
  sorry

end NUMINAMATH_CALUDE_abigail_initial_fences_l2627_262735


namespace NUMINAMATH_CALUDE_x_divisibility_l2627_262702

def x : ℕ := 128 + 192 + 256 + 320 + 576 + 704 + 6464

theorem x_divisibility :
  (∃ k : ℕ, x = 8 * k) ∧
  (∃ k : ℕ, x = 16 * k) ∧
  (∃ k : ℕ, x = 32 * k) ∧
  (∃ k : ℕ, x = 64 * k) :=
by sorry

end NUMINAMATH_CALUDE_x_divisibility_l2627_262702


namespace NUMINAMATH_CALUDE_four_fixed_points_iff_c_in_range_l2627_262774

/-- A quadratic function f(x) = x^2 - cx + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - c*x + c

/-- The composition of f with itself -/
def f_comp_f (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- Predicate for f ∘ f having four distinct fixed points -/
def has_four_distinct_fixed_points (c : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ x₄ : ℝ, 
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f_comp_f c x₁ = x₁ ∧ f_comp_f c x₂ = x₂ ∧ f_comp_f c x₃ = x₃ ∧ f_comp_f c x₄ = x₄

theorem four_fixed_points_iff_c_in_range :
  ∀ c : ℝ, has_four_distinct_fixed_points c ↔ (c < -1 ∨ c > 3) :=
sorry

end NUMINAMATH_CALUDE_four_fixed_points_iff_c_in_range_l2627_262774


namespace NUMINAMATH_CALUDE_max_value_problem_l2627_262775

theorem max_value_problem (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 3) :
  (x * y) / (x + y) + (x * z) / (x + z) + (y * z) / (y + z) ≤ 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_max_value_problem_l2627_262775


namespace NUMINAMATH_CALUDE_students_walking_home_l2627_262794

theorem students_walking_home (bus car bike scooter : ℚ) : 
  bus = 1/2 → car = 1/4 → bike = 1/10 → scooter = 1/8 → 
  1 - (bus + car + bike + scooter) = 1/40 := by
sorry

end NUMINAMATH_CALUDE_students_walking_home_l2627_262794


namespace NUMINAMATH_CALUDE_fence_cost_square_plot_l2627_262768

theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 58) :
  let side_length : ℝ := Real.sqrt area
  let perimeter : ℝ := 4 * side_length
  let total_cost : ℝ := perimeter * price_per_foot
  total_cost = 3944 := by
sorry

end NUMINAMATH_CALUDE_fence_cost_square_plot_l2627_262768


namespace NUMINAMATH_CALUDE_system_solution_l2627_262750

theorem system_solution (x y z : ℤ) : 
  (x^2 = y*z + 1 ∧ y^2 = z*x + 1 ∧ z^2 = x*y + 1) ↔ 
  ((x = 1 ∧ y = 0 ∧ z = -1) ∨
   (x = 1 ∧ y = -1 ∧ z = 0) ∨
   (x = 0 ∧ y = 1 ∧ z = -1) ∨
   (x = 0 ∧ y = -1 ∧ z = 1) ∨
   (x = -1 ∧ y = 1 ∧ z = 0) ∨
   (x = -1 ∧ y = 0 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2627_262750


namespace NUMINAMATH_CALUDE_product_of_smallest_primes_l2627_262729

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def smallest_one_digit_primes : (ℕ × ℕ) :=
  (2, 3)

def smallest_two_digit_prime : ℕ :=
  11

theorem product_of_smallest_primes :
  let (p1, p2) := smallest_one_digit_primes
  p1 * p2 * smallest_two_digit_prime = 66 ∧
  is_prime p1 ∧ is_prime p2 ∧ is_prime smallest_two_digit_prime ∧
  p1 < 10 ∧ p2 < 10 ∧ smallest_two_digit_prime ≥ 10 ∧ smallest_two_digit_prime < 100 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_smallest_primes_l2627_262729


namespace NUMINAMATH_CALUDE_mikes_marbles_l2627_262784

/-- Given that Mike initially has 8 orange marbles and gives 4 to Sam,
    prove that Mike now has 4 orange marbles. -/
theorem mikes_marbles (initial_marbles : ℕ) (marbles_given : ℕ) (remaining_marbles : ℕ) :
  initial_marbles = 8 →
  marbles_given = 4 →
  remaining_marbles = initial_marbles - marbles_given →
  remaining_marbles = 4 := by
  sorry

end NUMINAMATH_CALUDE_mikes_marbles_l2627_262784


namespace NUMINAMATH_CALUDE_isabel_total_songs_l2627_262791

/-- The number of country albums Isabel bought -/
def country_albums : ℕ := 6

/-- The number of pop albums Isabel bought -/
def pop_albums : ℕ := 2

/-- The number of jazz albums Isabel bought -/
def jazz_albums : ℕ := 4

/-- The number of rock albums Isabel bought -/
def rock_albums : ℕ := 3

/-- The number of songs in each country album -/
def songs_per_country_album : ℕ := 9

/-- The number of songs in each pop album -/
def songs_per_pop_album : ℕ := 9

/-- The number of songs in each jazz album -/
def songs_per_jazz_album : ℕ := 12

/-- The number of songs in each rock album -/
def songs_per_rock_album : ℕ := 14

/-- The total number of songs Isabel bought -/
def total_songs : ℕ := 
  country_albums * songs_per_country_album +
  pop_albums * songs_per_pop_album +
  jazz_albums * songs_per_jazz_album +
  rock_albums * songs_per_rock_album

theorem isabel_total_songs : total_songs = 162 := by
  sorry

end NUMINAMATH_CALUDE_isabel_total_songs_l2627_262791


namespace NUMINAMATH_CALUDE_painted_cubes_4x4x4_l2627_262786

/-- The number of unit cubes with at least one face painted in a 4x4x4 cube -/
def painted_cubes (n : Nat) : Nat :=
  n^3 - (n - 2)^3

/-- The proposition that the number of painted cubes in a 4x4x4 cube is 41 -/
theorem painted_cubes_4x4x4 :
  painted_cubes 4 = 41 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_4x4x4_l2627_262786


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2627_262752

theorem complex_modulus_problem (z : ℂ) (h : z * (Complex.I + 1) = Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2627_262752


namespace NUMINAMATH_CALUDE_inequality_proof_l2627_262751

theorem inequality_proof (x y : ℝ) (h : x^12 + y^12 ≤ 2) :
  x^2 + y^2 + x^2 * y^2 ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2627_262751


namespace NUMINAMATH_CALUDE_negation_existential_quadratic_l2627_262773

theorem negation_existential_quadratic :
  (¬ ∃ x : ℝ, x^2 + 2*x - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_existential_quadratic_l2627_262773


namespace NUMINAMATH_CALUDE_matrix_equality_zero_l2627_262764

open Matrix

theorem matrix_equality_zero (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℝ) 
  (h1 : A * B = B) (h2 : det (A - 1) ≠ 0) : B = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equality_zero_l2627_262764


namespace NUMINAMATH_CALUDE_sasha_leaves_picked_l2627_262725

/-- The number of apple trees along the road -/
def apple_trees : ℕ := 17

/-- The number of poplar trees along the road -/
def poplar_trees : ℕ := 20

/-- The index of the apple tree from which Sasha starts picking leaves -/
def start_index : ℕ := 8

/-- The total number of trees along the road -/
def total_trees : ℕ := apple_trees + poplar_trees

/-- The number of leaves Sasha picked -/
def leaves_picked : ℕ := total_trees - (start_index - 1)

theorem sasha_leaves_picked : leaves_picked = 24 := by
  sorry

end NUMINAMATH_CALUDE_sasha_leaves_picked_l2627_262725


namespace NUMINAMATH_CALUDE_marcus_gathered_25_bottles_l2627_262742

-- Define the total number of milk bottles
def total_bottles : ℕ := 45

-- Define the number of bottles John gathered
def john_bottles : ℕ := 20

-- Define Marcus' bottles as the difference between total and John's
def marcus_bottles : ℕ := total_bottles - john_bottles

-- Theorem to prove
theorem marcus_gathered_25_bottles : marcus_bottles = 25 := by
  sorry

end NUMINAMATH_CALUDE_marcus_gathered_25_bottles_l2627_262742


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2627_262734

theorem sum_of_fractions_equals_one (x y z : ℝ) (h : x * y * z = 1) :
  (1 / (1 + x + x * y)) + (1 / (1 + y + y * z)) + (1 / (1 + z + z * x)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2627_262734


namespace NUMINAMATH_CALUDE_area_bound_l2627_262749

-- Define the points and circles
variable (A B C D K L M N : Point)
variable (I I_A I_B I_C I_D : Circle)

-- Define the convex quadrilateral ABCD
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the inscribed circle I
def is_inscribed_circle (I : Circle) (A B C D : Point) : Prop := sorry

-- Define tangent points
def is_tangent_point (K L M N : Point) (I : Circle) (A B C D : Point) : Prop := sorry

-- Define incircles of triangles
def is_incircle (I_A I_B I_C I_D : Circle) (A B C D K L M N : Point) : Prop := sorry

-- Define common external tangent lines
def common_external_tangent (I_AB I_BC I_CD I_AD : Line) (I_A I_B I_C I_D : Circle) : Prop := sorry

-- Define the area S of the quadrilateral formed by I_AB, I_BC, I_CD, and I_AD
def area_S (I_AB I_BC I_CD I_AD : Line) : ℝ := sorry

-- Define the radius r of circle I
def radius_r (I : Circle) : ℝ := sorry

-- Theorem statement
theorem area_bound 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_inscribed_circle I A B C D)
  (h3 : is_tangent_point K L M N I A B C D)
  (h4 : is_incircle I_A I_B I_C I_D A B C D K L M N)
  (h5 : common_external_tangent I_AB I_BC I_CD I_AD I_A I_B I_C I_D)
  (S : ℝ)
  (h6 : S = area_S I_AB I_BC I_CD I_AD)
  (r : ℝ)
  (h7 : r = radius_r I) :
  S ≤ (12 - 8 * Real.sqrt 2) * r^2 := by sorry

end NUMINAMATH_CALUDE_area_bound_l2627_262749


namespace NUMINAMATH_CALUDE_max_multicolored_sets_l2627_262700

/-- A color distribution is a list of positive integers representing the number of points of each color. -/
def ColorDistribution := List Nat

/-- The number of multi-colored sets for a given color distribution. -/
def multiColoredSets (d : ColorDistribution) : Nat :=
  d.prod

/-- Predicate to check if a color distribution is valid for the problem. -/
def isValidDistribution (d : ColorDistribution) : Prop :=
  d.length > 0 ∧ 
  d.sum = 2012 ∧ 
  d.Nodup ∧
  d.all (· > 0)

/-- The theorem stating that 61 colors maximize the number of multi-colored sets. -/
theorem max_multicolored_sets : 
  ∃ (d : ColorDistribution), isValidDistribution d ∧ d.length = 61 ∧
  ∀ (d' : ColorDistribution), isValidDistribution d' → d'.length ≠ 61 → 
    multiColoredSets d ≥ multiColoredSets d' :=
  sorry

end NUMINAMATH_CALUDE_max_multicolored_sets_l2627_262700


namespace NUMINAMATH_CALUDE_sally_balloons_l2627_262759

theorem sally_balloons (sally_balloons fred_balloons : ℕ) : 
  fred_balloons = 3 * sally_balloons →
  fred_balloons = 18 →
  sally_balloons = 6 := by
sorry

end NUMINAMATH_CALUDE_sally_balloons_l2627_262759


namespace NUMINAMATH_CALUDE_molly_age_when_stopped_l2627_262711

/-- Calculates the age when a person stops riding their bike daily, given their starting age,
    daily riding distance, total distance covered, and days in a year. -/
def age_when_stopped (starting_age : ℕ) (daily_distance : ℕ) (total_distance : ℕ) (days_per_year : ℕ) : ℕ :=
  starting_age + (total_distance / daily_distance) / days_per_year

/-- Theorem stating that given the specified conditions, Molly's age when she stopped riding
    her bike daily is 16 years old. -/
theorem molly_age_when_stopped :
  let starting_age : ℕ := 13
  let daily_distance : ℕ := 3
  let total_distance : ℕ := 3285
  let days_per_year : ℕ := 365
  age_when_stopped starting_age daily_distance total_distance days_per_year = 16 := by
  sorry


end NUMINAMATH_CALUDE_molly_age_when_stopped_l2627_262711


namespace NUMINAMATH_CALUDE_seed_ratio_proof_l2627_262718

def total_seeds : ℕ := 120
def left_seeds : ℕ := 20
def additional_seeds : ℕ := 30
def remaining_seeds : ℕ := 30

theorem seed_ratio_proof :
  let used_seeds := total_seeds - remaining_seeds
  let right_seeds := used_seeds - left_seeds - additional_seeds
  (right_seeds : ℚ) / left_seeds = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_seed_ratio_proof_l2627_262718


namespace NUMINAMATH_CALUDE_inequality_proof_l2627_262727

theorem inequality_proof (a : ℝ) : 
  (a^2 + 5)^2 + 4*a*(10 - a) - 8*a^3 ≥ 0 ∧ 
  ((a^2 + 5)^2 + 4*a*(10 - a) - 8*a^3 = 0 ↔ a = 5 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2627_262727


namespace NUMINAMATH_CALUDE_cost_of_socks_socks_cost_proof_l2627_262730

theorem cost_of_socks (initial_amount : ℕ) (shirt_cost : ℕ) (remaining_amount : ℕ) : ℕ :=
  initial_amount - shirt_cost - remaining_amount

theorem socks_cost_proof (initial_amount : ℕ) (shirt_cost : ℕ) (remaining_amount : ℕ) 
    (h1 : initial_amount = 100)
    (h2 : shirt_cost = 24)
    (h3 : remaining_amount = 65) :
  cost_of_socks initial_amount shirt_cost remaining_amount = 11 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_socks_socks_cost_proof_l2627_262730


namespace NUMINAMATH_CALUDE_max_probability_zero_units_digit_l2627_262760

def probability_zero_units_digit (N : ℕ+) : ℚ :=
  let q2 := (N / 2 : ℚ) / N
  let q5 := (N / 5 : ℚ) / N
  let q10 := (N / 10 : ℚ) / N
  q10 * (2 - q10) + 2 * (q2 - q10) * (q5 - q10)

theorem max_probability_zero_units_digit :
  ∀ N : ℕ+, probability_zero_units_digit N ≤ 27/100 := by
  sorry

end NUMINAMATH_CALUDE_max_probability_zero_units_digit_l2627_262760


namespace NUMINAMATH_CALUDE_calculation_proof_l2627_262706

theorem calculation_proof : 2 * (-1/4) - |1 - Real.sqrt 3| + (-2023)^0 = 3/2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2627_262706


namespace NUMINAMATH_CALUDE_class_mean_score_l2627_262777

theorem class_mean_score 
  (n : ℕ) 
  (h1 : n > 15) 
  (overall_mean : ℝ) 
  (h2 : overall_mean = 10) 
  (group_mean : ℝ) 
  (h3 : group_mean = 16) : 
  let remaining_mean := (n * overall_mean - 15 * group_mean) / (n - 15)
  remaining_mean = (10 * n - 240) / (n - 15) := by
sorry

end NUMINAMATH_CALUDE_class_mean_score_l2627_262777


namespace NUMINAMATH_CALUDE_largest_three_digit_congruence_l2627_262755

theorem largest_three_digit_congruence :
  ∃ (n : ℕ), 
    n = 998 ∧ 
    100 ≤ n ∧ n < 1000 ∧ 
    (70 * n) % 350 = 210 ∧
    ∀ (m : ℕ), 100 ≤ m ∧ m < 1000 ∧ (70 * m) % 350 = 210 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_congruence_l2627_262755


namespace NUMINAMATH_CALUDE_only_prime_with_alternating_base14_rep_l2627_262724

/-- Represents a number in base-14 with alternating 1s and 0s -/
def alternatingBaseRepresentation (n : ℕ) : ℕ :=
  (14^(2*n+2) - 1) / (14^2 - 1)

/-- Checks if a number is prime -/
def isPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

theorem only_prime_with_alternating_base14_rep :
  ∃! p : ℕ, isPrime p ∧ ∃ n : ℕ, alternatingBaseRepresentation n = p :=
by
  -- The unique prime is 197
  use 197
  sorry -- Proof omitted

#eval alternatingBaseRepresentation 1  -- Should evaluate to 197

end NUMINAMATH_CALUDE_only_prime_with_alternating_base14_rep_l2627_262724


namespace NUMINAMATH_CALUDE_difference_of_squares_l2627_262797

theorem difference_of_squares (a b : ℝ) : (2*a + b) * (b - 2*a) = b^2 - 4*a^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2627_262797


namespace NUMINAMATH_CALUDE_sosnovka_petrovka_distance_l2627_262782

/-- The distance between two points on a road --/
def distance (a b : ℕ) : ℕ := max a b - min a b

theorem sosnovka_petrovka_distance :
  ∀ (A B P S : ℕ),
  distance A P = 70 →
  distance A B = 20 →
  distance B S = 130 →
  distance S P = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_sosnovka_petrovka_distance_l2627_262782
