import Mathlib

namespace seating_arrangements_l3426_342673

theorem seating_arrangements (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 5) :
  (n.factorial / (n - k).factorial) = 720 := by
  sorry

end seating_arrangements_l3426_342673


namespace quadratic_touches_x_axis_l3426_342665

/-- A quadratic function that touches the x-axis -/
def touches_x_axis (a : ℝ) : Prop :=
  ∃ x : ℝ, 2 * x^2 - 8 * x + a = 0 ∧
  ∀ y : ℝ, 2 * y^2 - 8 * y + a ≥ 0

/-- The value of 'a' for which the quadratic function touches the x-axis is 8 -/
theorem quadratic_touches_x_axis :
  ∃! a : ℝ, touches_x_axis a ∧ a = 8 :=
sorry

end quadratic_touches_x_axis_l3426_342665


namespace tangent_line_properties_l3426_342644

-- Define the curve
def f (x : ℝ) : ℝ := 4 * x^2 - 6 * x + 3

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 8 * x - 6

theorem tangent_line_properties :
  -- Part a: Tangent line parallel to y = 2x at (1, 1)
  (f' 1 = 2 ∧ f 1 = 1) ∧
  -- Part b: Tangent line perpendicular to y = x/4 at (1/4, 7/4)
  (f' (1/4) = -4 ∧ f (1/4) = 7/4) :=
by sorry

end tangent_line_properties_l3426_342644


namespace min_value_expression_l3426_342690

theorem min_value_expression (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  c^2 + d^2 + 4/c^2 + 2*d/c ≥ 2 * Real.sqrt 3 ∧
  ∃ (c₀ d₀ : ℝ), c₀ ≠ 0 ∧ d₀ ≠ 0 ∧ c₀^2 + d₀^2 + 4/c₀^2 + 2*d₀/c₀ = 2 * Real.sqrt 3 :=
sorry

end min_value_expression_l3426_342690


namespace descending_order_l3426_342626

-- Define the numbers
def a : ℝ := 0.8
def b : ℝ := 0.878
def c : ℝ := 0.877
def d : ℝ := 0.87

-- Theorem statement
theorem descending_order : b > c ∧ c > d ∧ d > a := by sorry

end descending_order_l3426_342626


namespace divisibility_by_36_l3426_342664

theorem divisibility_by_36 (n : ℤ) (h1 : n ≥ 5) (h2 : ¬ 2 ∣ n) (h3 : ¬ 3 ∣ n) : 
  36 ∣ (n^2 - 1) := by
  sorry

end divisibility_by_36_l3426_342664


namespace no_solution_exists_l3426_342659

open Function Set

-- Define the property that a function must satisfy
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y) = f x - y

-- State the theorem
theorem no_solution_exists :
  ¬ ∃ f : ℝ → ℝ, Continuous f ∧ SatisfiesFunctionalEquation f :=
sorry

end no_solution_exists_l3426_342659


namespace no_torn_cards_l3426_342607

/-- The number of baseball cards Mary initially had -/
def initial_cards : ℕ := 18

/-- The number of baseball cards Fred gave to Mary -/
def fred_cards : ℕ := 26

/-- The number of baseball cards Mary bought -/
def bought_cards : ℕ := 40

/-- The total number of baseball cards Mary has now -/
def total_cards : ℕ := 84

/-- The number of torn baseball cards in Mary's initial collection -/
def torn_cards : ℕ := initial_cards - (total_cards - fred_cards - bought_cards)

theorem no_torn_cards : torn_cards = 0 := by
  sorry

end no_torn_cards_l3426_342607


namespace baker_new_cakes_l3426_342668

theorem baker_new_cakes (initial_cakes sold_cakes current_cakes : ℕ) 
  (h1 : initial_cakes = 121)
  (h2 : sold_cakes = 105)
  (h3 : current_cakes = 186) :
  current_cakes - (initial_cakes - sold_cakes) = 170 := by
  sorry

end baker_new_cakes_l3426_342668


namespace prob_less_than_8_l3426_342698

/-- The probability of an archer scoring less than 8 in a single shot -/
theorem prob_less_than_8 (p_10 p_9 p_8 : ℝ) 
  (h1 : p_10 = 0.24)
  (h2 : p_9 = 0.28)
  (h3 : p_8 = 0.19) : 
  1 - (p_10 + p_9 + p_8) = 0.29 := by
  sorry

end prob_less_than_8_l3426_342698


namespace molecular_weight_8_moles_Al2O3_l3426_342641

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Aluminum atoms in one molecule of Al2O3 -/
def num_Al_atoms : ℕ := 2

/-- The number of Oxygen atoms in one molecule of Al2O3 -/
def num_O_atoms : ℕ := 3

/-- The number of moles of Al2O3 -/
def num_moles : ℕ := 8

/-- The molecular weight of Al2O3 in g/mol -/
def molecular_weight_Al2O3 : ℝ := 
  num_Al_atoms * atomic_weight_Al + num_O_atoms * atomic_weight_O

theorem molecular_weight_8_moles_Al2O3 : 
  num_moles * molecular_weight_Al2O3 = 815.68 := by
  sorry

end molecular_weight_8_moles_Al2O3_l3426_342641


namespace reading_time_calculation_l3426_342675

def total_reading_time (total_chapters : ℕ) (reading_time_per_chapter : ℕ) : ℕ :=
  let chapters_read := total_chapters - (total_chapters / 3)
  (chapters_read * reading_time_per_chapter) / 60

theorem reading_time_calculation :
  total_reading_time 31 20 = 7 := by
  sorry

end reading_time_calculation_l3426_342675


namespace factor_theorem_quadratic_l3426_342606

theorem factor_theorem_quadratic (t : ℝ) : 
  (∃ (p : ℝ → ℝ), ∀ x, 4*x^2 - 8*x + 3 = (x - t) * p x) ↔ (t = 1.5 ∨ t = 0.5) :=
by sorry

end factor_theorem_quadratic_l3426_342606


namespace symmetry_of_graphs_l3426_342616

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define a real number a
variable (a : ℝ)

-- Define symmetry about a vertical line
def symmetricAboutVerticalLine (g h : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x y, g x = y ↔ h (2*c - x) = y

-- State the theorem
theorem symmetry_of_graphs :
  symmetricAboutVerticalLine (fun x ↦ f (a - x)) (fun x ↦ f (x - a)) a :=
sorry

end symmetry_of_graphs_l3426_342616


namespace line_through_points_l3426_342632

def line_equation (k m x : ℝ) : ℝ := k * x + m

theorem line_through_points (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : ∃ n : ℕ, b = n * a) :
  (∃ m : ℝ, line_equation k m a = a ∧ line_equation k m b = 8 * b) →
  k ∈ ({9, 15} : Set ℝ) := by
  sorry

end line_through_points_l3426_342632


namespace sixth_term_is_64_l3426_342680

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  sum_2_4 : a 2 + a 4 = 20
  sum_3_5 : a 3 + a 5 = 40

/-- The sixth term of the geometric sequence is 64 -/
theorem sixth_term_is_64 (seq : GeometricSequence) : seq.a 6 = 64 := by
  sorry

end sixth_term_is_64_l3426_342680


namespace cubic_function_constraint_l3426_342691

def f (a b c x : ℝ) : ℝ := 4 * x^3 + a * x^2 + b * x + c

theorem cubic_function_constraint (a b c : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ f a b c x ∧ f a b c x ≤ 1) →
  a = 0 ∧ b = -3 ∧ c = 0 := by
  sorry

end cubic_function_constraint_l3426_342691


namespace hyundai_dodge_ratio_l3426_342655

theorem hyundai_dodge_ratio (total : ℕ) (dodge : ℕ) (kia : ℕ) (hyundai : ℕ) :
  total = 400 →
  dodge = total / 2 →
  kia = 100 →
  hyundai = total - dodge - kia →
  (hyundai : ℚ) / dodge = 1 / 2 := by
sorry

end hyundai_dodge_ratio_l3426_342655


namespace line_passes_through_fixed_point_l3426_342669

/-- If k, -1, and b form an arithmetic sequence, then the line y = kx + b passes through (1, -2) -/
theorem line_passes_through_fixed_point (k b : ℝ) :
  (∃ d : ℝ, k = -1 - d ∧ b = -1 + d) →
  k * 1 + b = -2 :=
by sorry

end line_passes_through_fixed_point_l3426_342669


namespace program_result_l3426_342682

def program_output (a₀ b₀ : ℕ) : ℕ × ℕ :=
  let a₁ := a₀ + b₀
  let b₁ := b₀ * a₁
  (a₁, b₁)

theorem program_result :
  program_output 1 3 = (4, 12) := by sorry

end program_result_l3426_342682


namespace baseball_card_difference_l3426_342625

theorem baseball_card_difference (marcus_cards carter_cards : ℕ) 
  (h1 : marcus_cards = 210) 
  (h2 : carter_cards = 152) : 
  marcus_cards - carter_cards = 58 := by
sorry

end baseball_card_difference_l3426_342625


namespace problem_statement_l3426_342622

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -7)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 8) :
  b / (a + b) + c / (b + c) + a / (c + a) = 9 := by
  sorry

end problem_statement_l3426_342622


namespace first_character_lines_l3426_342618

/-- The number of lines for each character in Jerry's skit script --/
structure ScriptLines where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of Jerry's skit script --/
def script_conditions (lines : ScriptLines) : Prop :=
  lines.first = lines.second + 8 ∧
  lines.third = 2 ∧
  lines.second = 6 + 3 * lines.third

/-- Theorem stating that the first character has 20 lines --/
theorem first_character_lines (lines : ScriptLines) 
  (h : script_conditions lines) : lines.first = 20 := by
  sorry

#check first_character_lines

end first_character_lines_l3426_342618


namespace apple_orange_price_l3426_342678

theorem apple_orange_price (x y z : ℝ) 
  (eq1 : 24 * x = 28 * y)
  (eq2 : 45 * x + 60 * y = 1350 * z) :
  30 * x + 40 * y = 118.2857 * z :=
by sorry

end apple_orange_price_l3426_342678


namespace elective_schemes_count_l3426_342646

/-- The number of courses offered by the school -/
def total_courses : ℕ := 10

/-- The number of conflicting courses (A, B, C) -/
def conflicting_courses : ℕ := 3

/-- The number of courses each student must elect -/
def courses_to_elect : ℕ := 3

/-- The number of different elective schemes available for a student -/
def elective_schemes : ℕ := Nat.choose (total_courses - conflicting_courses) courses_to_elect +
                             conflicting_courses * Nat.choose (total_courses - conflicting_courses) (courses_to_elect - 1)

theorem elective_schemes_count :
  elective_schemes = 98 :=
sorry

end elective_schemes_count_l3426_342646


namespace binomial_coefficient_two_l3426_342658

theorem binomial_coefficient_two (n : ℕ+) : Nat.choose n.val 2 = n.val * (n.val - 1) / 2 := by
  sorry

end binomial_coefficient_two_l3426_342658


namespace harmonic_progression_logarithm_equality_l3426_342688

/-- Given x, y, z form a harmonic progression, 
    prove that lg (x+z) + lg (x-2y+z) = 2 lg (x-z) -/
theorem harmonic_progression_logarithm_equality 
  (x y z : ℝ) 
  (h : (1/x + 1/z)/2 = 1/y) : 
  Real.log (x+z) + Real.log (x-2*y+z) = 2 * Real.log (x-z) := by
  sorry

end harmonic_progression_logarithm_equality_l3426_342688


namespace expression_evaluation_l3426_342679

theorem expression_evaluation :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) * Real.sqrt (3^2 + 1^2) = 3280 * Real.sqrt 10 := by
  sorry

end expression_evaluation_l3426_342679


namespace percentage_increase_l3426_342648

theorem percentage_increase (x : ℝ) (h : x = 114.4) : 
  (x - 88) / 88 * 100 = 30 := by
  sorry

end percentage_increase_l3426_342648


namespace vector_addition_l3426_342699

theorem vector_addition : 
  let v1 : Fin 3 → ℝ := ![4, -9, 2]
  let v2 : Fin 3 → ℝ := ![-3, 8, -5]
  v1 + v2 = ![1, -1, -3] := by
sorry

end vector_addition_l3426_342699


namespace qin_jiushao_v3_value_l3426_342629

/-- Qin Jiushao's algorithm for polynomial evaluation -/
def qinJiushao (f : ℝ → ℝ) (x : ℝ) : ℕ → ℝ
  | 0 => x + 3
  | 1 => qinJiushao f x 0 * x - 1
  | 2 => qinJiushao f x 1 * x
  | 3 => qinJiushao f x 2 * x + 2
  | 4 => qinJiushao f x 3 * x - 1
  | _ => 0

/-- The polynomial f(x) = x^5 + 3x^4 - x^3 + 2x - 1 -/
def f (x : ℝ) : ℝ := x^5 + 3*x^4 - x^3 + 2*x - 1

theorem qin_jiushao_v3_value :
  qinJiushao f 2 2 = 18 := by sorry

end qin_jiushao_v3_value_l3426_342629


namespace line_equation_with_parallel_intersections_l3426_342696

/-- The equation of a line passing through point P(1,2) and intersecting two parallel lines,
    forming a line segment of length √2. --/
theorem line_equation_with_parallel_intersections
  (l : Set (ℝ × ℝ))  -- The line we're looking for
  (P : ℝ × ℝ)        -- Point P
  (l₁ l₂ : Set (ℝ × ℝ))  -- The two parallel lines
  (A B : ℝ × ℝ)      -- Points of intersection
  (h_P : P = (1, 2))
  (h_l₁ : l₁ = {(x, y) : ℝ × ℝ | 4*x + 3*y + 1 = 0})
  (h_l₂ : l₂ = {(x, y) : ℝ × ℝ | 4*x + 3*y + 6 = 0})
  (h_A : A ∈ l ∩ l₁)
  (h_B : B ∈ l ∩ l₂)
  (h_dist : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 2)
  (h_P_on_l : P ∈ l) :
  l = {(x, y) : ℝ × ℝ | 7*x - y - 5 = 0} ∨
  l = {(x, y) : ℝ × ℝ | x + 7*y - 15 = 0} :=
sorry

end line_equation_with_parallel_intersections_l3426_342696


namespace spending_theorem_l3426_342639

-- Define the fraction of savings spent on the stereo
def stereo_fraction : ℚ := 1/4

-- Define the fraction less spent on the television compared to the stereo
def tv_fraction_less : ℚ := 2/3

-- Calculate the fraction spent on the television
def tv_fraction : ℚ := stereo_fraction - tv_fraction_less * stereo_fraction

-- Define the total fraction spent on both items
def total_fraction : ℚ := stereo_fraction + tv_fraction

-- Theorem statement
theorem spending_theorem : total_fraction = 1/3 := by
  sorry

end spending_theorem_l3426_342639


namespace concatenation_equation_solution_l3426_342631

theorem concatenation_equation_solution :
  ∃ x : ℕ, x + (10 * x + x) = 12 ∧ x = 1 := by
  sorry

end concatenation_equation_solution_l3426_342631


namespace candy_consumption_theorem_l3426_342620

/-- Represents the number of candies eaten by each person -/
structure CandyConsumption where
  andrey : ℕ
  boris : ℕ
  denis : ℕ

/-- Represents the relative eating rates of the three people -/
structure EatingRates where
  andrey_boris : ℚ  -- Ratio of Andrey's rate to Boris's rate
  andrey_denis : ℚ  -- Ratio of Andrey's rate to Denis's rate

theorem candy_consumption_theorem (rates : EatingRates) (total : ℕ) : 
  rates.andrey_boris = 4/3 → 
  rates.andrey_denis = 6/7 → 
  total = 70 → 
  ∃ (consumption : CandyConsumption), 
    consumption.andrey = 24 ∧ 
    consumption.boris = 18 ∧ 
    consumption.denis = 28 ∧
    consumption.andrey + consumption.boris + consumption.denis = total :=
by sorry

end candy_consumption_theorem_l3426_342620


namespace cubic_two_roots_l3426_342614

/-- A cubic function with a parameter c -/
def f (c : ℝ) : ℝ → ℝ := fun x ↦ x^3 - 3*x + c

/-- The derivative of f -/
def f' (c : ℝ) : ℝ → ℝ := fun x ↦ 3*x^2 - 3

theorem cubic_two_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f c x₁ = 0 ∧ f c x₂ = 0 ∧
    ∀ x, f c x = 0 → x = x₁ ∨ x = x₂) →
  c = 2 ∨ c = -2 := by
  sorry

end cubic_two_roots_l3426_342614


namespace max_difference_bounded_l3426_342695

theorem max_difference_bounded (a : Fin 2017 → ℝ) 
  (h1 : a 1 = a 2017)
  (h2 : ∀ i : Fin 2015, |a i + a (i + 2) - 2 * a (i + 1)| ≤ 1) :
  ∃ M : ℝ, M = 508032 ∧ 
  (∀ i j : Fin 2017, i < j → |a i - a j| ≤ M) ∧
  (∃ i j : Fin 2017, i < j ∧ |a i - a j| = M) := by
sorry

end max_difference_bounded_l3426_342695


namespace election_votes_calculation_l3426_342676

theorem election_votes_calculation (total_votes : ℕ) : 
  (total_votes : ℚ) * (55 / 100) - (total_votes : ℚ) * (30 / 100) = 174 →
  total_votes = 696 := by
  sorry

end election_votes_calculation_l3426_342676


namespace solve_exponential_equation_l3426_342609

theorem solve_exponential_equation :
  ∀ x : ℝ, (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^x = (256 : ℝ)^(10 : ℝ) → x = 5 := by
  sorry

end solve_exponential_equation_l3426_342609


namespace polynomial_factorization_l3426_342683

theorem polynomial_factorization (x : ℝ) :
  x^8 + x^4 + 1 = (x^2 - Real.sqrt 3 * x + 1) * (x^2 + Real.sqrt 3 * x + 1) * (x^2 - x + 1) * (x^2 + x + 1) := by
  sorry

end polynomial_factorization_l3426_342683


namespace rachel_homework_l3426_342684

theorem rachel_homework (reading_pages math_pages : ℕ) : 
  reading_pages = math_pages + 6 →
  reading_pages = 14 →
  math_pages = 8 := by sorry

end rachel_homework_l3426_342684


namespace painting_cost_conversion_l3426_342666

/-- Given exchange rates and the cost of a painting in Namibian dollars, 
    prove its cost in Euros -/
theorem painting_cost_conversion 
  (usd_to_nam : ℝ) 
  (usd_to_eur : ℝ) 
  (painting_cost_nam : ℝ) 
  (h1 : usd_to_nam = 7) 
  (h2 : usd_to_eur = 0.9) 
  (h3 : painting_cost_nam = 140) : 
  painting_cost_nam / usd_to_nam * usd_to_eur = 18 := by
  sorry

end painting_cost_conversion_l3426_342666


namespace regular_price_is_100_l3426_342634

/-- The regular price of one bag -/
def regular_price : ℝ := 100

/-- The promotional price of the fourth bag -/
def fourth_bag_price : ℝ := 5

/-- The total cost for four bags -/
def total_cost : ℝ := 305

/-- Theorem stating that the regular price of one bag is $100 -/
theorem regular_price_is_100 :
  3 * regular_price + fourth_bag_price = total_cost :=
by sorry

end regular_price_is_100_l3426_342634


namespace employee_count_proof_l3426_342627

/-- The number of employees in the room -/
def num_employees : ℕ := 25000

/-- The initial percentage of managers as a rational number -/
def initial_manager_percentage : ℚ := 99 / 100

/-- The final percentage of managers as a rational number -/
def final_manager_percentage : ℚ := 98 / 100

/-- The number of managers that leave the room -/
def managers_leaving : ℕ := 250

theorem employee_count_proof :
  (initial_manager_percentage * num_employees : ℚ) - managers_leaving = 
  final_manager_percentage * num_employees :=
by sorry

end employee_count_proof_l3426_342627


namespace number_exceeding_percentage_l3426_342647

theorem number_exceeding_percentage : ∃ x : ℝ, x = 0.35 * x + 245 := by
  sorry

end number_exceeding_percentage_l3426_342647


namespace rectangle_unique_property_l3426_342638

-- Define the properties
def opposite_sides_equal (shape : Type) : Prop := sorry
def opposite_angles_equal (shape : Type) : Prop := sorry
def diagonals_equal (shape : Type) : Prop := sorry
def opposite_sides_parallel (shape : Type) : Prop := sorry

-- Define rectangles and parallelograms
class Rectangle (shape : Type) : Prop :=
  (opp_sides_eq : opposite_sides_equal shape)
  (opp_angles_eq : opposite_angles_equal shape)
  (diag_eq : diagonals_equal shape)
  (opp_sides_para : opposite_sides_parallel shape)

class Parallelogram (shape : Type) : Prop :=
  (opp_sides_eq : opposite_sides_equal shape)
  (opp_angles_eq : opposite_angles_equal shape)
  (opp_sides_para : opposite_sides_parallel shape)

-- Theorem statement
theorem rectangle_unique_property :
  ∀ (shape : Type),
    Rectangle shape →
    Parallelogram shape →
    (diagonals_equal shape ↔ Rectangle shape) ∧
    (¬(opposite_sides_equal shape ↔ Rectangle shape)) ∧
    (¬(opposite_angles_equal shape ↔ Rectangle shape)) ∧
    (¬(opposite_sides_parallel shape ↔ Rectangle shape)) :=
sorry

end rectangle_unique_property_l3426_342638


namespace marbles_combination_l3426_342637

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of marbles -/
def total_marbles : ℕ := 9

/-- The number of marbles to choose -/
def marbles_to_choose : ℕ := 4

/-- Theorem stating that choosing 4 marbles from 9 results in 126 ways -/
theorem marbles_combination :
  choose total_marbles marbles_to_choose = 126 := by
  sorry

end marbles_combination_l3426_342637


namespace yellow_given_popped_l3426_342617

/-- Represents the types of kernels in the bag -/
inductive KernelType
  | White
  | Yellow
  | Brown

/-- The probability of selecting a kernel of a given type -/
def selectionProb (k : KernelType) : ℚ :=
  match k with
  | KernelType.White => 3/5
  | KernelType.Yellow => 1/5
  | KernelType.Brown => 1/5

/-- The probability of a kernel popping given its type -/
def poppingProb (k : KernelType) : ℚ :=
  match k with
  | KernelType.White => 1/3
  | KernelType.Yellow => 3/4
  | KernelType.Brown => 1/2

/-- The probability of selecting and popping a kernel of a given type -/
def selectAndPopProb (k : KernelType) : ℚ :=
  selectionProb k * poppingProb k

/-- The total probability of selecting and popping any kernel -/
def totalPopProb : ℚ :=
  selectAndPopProb KernelType.White + selectAndPopProb KernelType.Yellow + selectAndPopProb KernelType.Brown

/-- The probability that a popped kernel is yellow -/
theorem yellow_given_popped :
  selectAndPopProb KernelType.Yellow / totalPopProb = 1/3 := by
  sorry


end yellow_given_popped_l3426_342617


namespace chocolate_chip_cookies_l3426_342610

/-- The number of cookies in each bag -/
def cookies_per_bag : ℕ := 9

/-- The number of oatmeal cookies -/
def oatmeal_cookies : ℕ := 41

/-- The number of baggies that can be made with all cookies -/
def total_baggies : ℕ := 6

/-- The theorem stating the number of chocolate chip cookies -/
theorem chocolate_chip_cookies : 
  cookies_per_bag * total_baggies - oatmeal_cookies = 13 := by
  sorry

end chocolate_chip_cookies_l3426_342610


namespace minimum_travel_time_for_problem_scenario_l3426_342693

/-- Represents the travel scenario between two cities -/
structure TravelScenario where
  distance : ℝ
  num_people : ℕ
  num_bicycles : ℕ
  cyclist_speed : ℝ
  pedestrian_speed : ℝ

/-- The minimum time for all people to reach the destination -/
def minimum_travel_time (scenario : TravelScenario) : ℝ :=
  sorry

/-- The specific travel scenario from the problem -/
def problem_scenario : TravelScenario :=
  { distance := 45
    num_people := 3
    num_bicycles := 2
    cyclist_speed := 15
    pedestrian_speed := 5 }

theorem minimum_travel_time_for_problem_scenario :
  minimum_travel_time problem_scenario = 3 :=
by sorry

end minimum_travel_time_for_problem_scenario_l3426_342693


namespace quadratic_roots_theorem_l3426_342619

theorem quadratic_roots_theorem (c : ℝ) :
  (∀ x : ℝ, x^2 - 3*x + c = 0 ↔ x = (3 + Real.sqrt c) / 2 ∨ x = (3 - Real.sqrt c) / 2) →
  c = 9/5 :=
by sorry

end quadratic_roots_theorem_l3426_342619


namespace sum_three_fourths_power_inequality_l3426_342681

theorem sum_three_fourths_power_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^(3/4) + b^(3/4) + c^(3/4) > (a + b + c)^(3/4) := by
  sorry

end sum_three_fourths_power_inequality_l3426_342681


namespace shoe_selection_theorem_l3426_342652

/-- The number of pairs of shoes in the bag -/
def total_pairs : ℕ := 10

/-- The number of shoes randomly selected -/
def selected_shoes : ℕ := 4

/-- The number of ways to select 4 shoes such that none of them form a pair -/
def no_pairs : ℕ := 3360

/-- The number of ways to select 4 shoes such that exactly 2 pairs are formed -/
def two_pairs : ℕ := 45

/-- The number of ways to select 4 shoes such that 2 shoes form a pair and the other 2 do not -/
def one_pair : ℕ := 1440

theorem shoe_selection_theorem :
  (Nat.choose total_pairs selected_shoes * 2^selected_shoes = no_pairs) ∧
  (Nat.choose total_pairs 2 = two_pairs) ∧
  (total_pairs * Nat.choose (total_pairs - 1) 2 * 2^2 = one_pair) :=
by sorry

end shoe_selection_theorem_l3426_342652


namespace total_bills_and_coins_l3426_342657

/-- Represents the payment details for a grocery bill -/
structure GroceryPayment where
  totalBill : ℕ
  billValue : ℕ
  coinValue : ℕ
  numBills : ℕ
  numCoins : ℕ

/-- Theorem stating the total number of bills and coins used in the payment -/
theorem total_bills_and_coins (payment : GroceryPayment) 
  (h1 : payment.totalBill = 285)
  (h2 : payment.billValue = 20)
  (h3 : payment.coinValue = 5)
  (h4 : payment.numBills = 11)
  (h5 : payment.numCoins = 11)
  : payment.numBills + payment.numCoins = 22 := by
  sorry


end total_bills_and_coins_l3426_342657


namespace y_in_terms_of_x_l3426_342670

theorem y_in_terms_of_x (x y : ℚ) : x - 2 = 4 * (y - 1) + 3 → y = (1/4) * x - (1/4) := by
  sorry

end y_in_terms_of_x_l3426_342670


namespace hike_taxi_count_hike_taxi_count_is_six_l3426_342605

/-- Calculates the number of taxis required for a hike --/
theorem hike_taxi_count (total_people : ℕ) (car_count : ℕ) (van_count : ℕ) 
  (people_per_car : ℕ) (people_per_van : ℕ) (people_per_taxi : ℕ) : ℕ :=
  let people_in_cars := car_count * people_per_car
  let people_in_vans := van_count * people_per_van
  let people_in_taxis := total_people - (people_in_cars + people_in_vans)
  people_in_taxis / people_per_taxi

/-- Proves that 6 taxis were required for the hike --/
theorem hike_taxi_count_is_six : 
  hike_taxi_count 58 3 2 4 5 6 = 6 := by
  sorry

end hike_taxi_count_hike_taxi_count_is_six_l3426_342605


namespace sphere_cylinder_volume_difference_l3426_342694

/-- The volume of the space inside a sphere and outside an inscribed right cylinder -/
theorem sphere_cylinder_volume_difference (r_sphere : ℝ) (r_cylinder : ℝ) :
  r_sphere = 6 →
  r_cylinder = 4 →
  let h_cylinder := 2 * Real.sqrt (r_sphere ^ 2 - r_cylinder ^ 2)
  let v_sphere := (4 / 3) * Real.pi * r_sphere ^ 3
  let v_cylinder := Real.pi * r_cylinder ^ 2 * h_cylinder
  v_sphere - v_cylinder = (288 - 64 * Real.sqrt 5) * Real.pi := by
  sorry

end sphere_cylinder_volume_difference_l3426_342694


namespace A_equals_set_l3426_342697

def A : Set ℝ :=
  {x | ∃ a b c : ℝ, a * b * c ≠ 0 ∧ 
       x = a / |a| + |b| / b + |c| / c + (a * b * c) / |a * b * c|}

theorem A_equals_set : A = {-4, 0, 4} := by
  sorry

end A_equals_set_l3426_342697


namespace senior_mean_score_l3426_342685

-- Define the total number of students
def total_students : ℕ := 120

-- Define the overall mean score
def overall_mean : ℝ := 110

-- Define the relationship between number of seniors and juniors
def junior_senior_ratio : ℝ := 0.75

-- Define the relationship between senior and junior mean scores
def senior_junior_mean_ratio : ℝ := 1.4

-- Theorem statement
theorem senior_mean_score :
  ∃ (seniors juniors : ℕ) (senior_mean junior_mean : ℝ),
    seniors + juniors = total_students ∧
    juniors = Int.floor (junior_senior_ratio * seniors) ∧
    senior_mean = senior_junior_mean_ratio * junior_mean ∧
    (seniors * senior_mean + juniors * junior_mean) / total_students = overall_mean ∧
    Int.floor senior_mean = 124 := by
  sorry

end senior_mean_score_l3426_342685


namespace triangle_third_side_length_l3426_342612

theorem triangle_third_side_length 
  (a b x : ℕ) 
  (h1 : a = 2) 
  (h2 : b = 6) 
  (h3 : Even x) 
  (h4 : x > (b - a)) 
  (h5 : x < (b + a)) : 
  x = 6 := by
sorry

end triangle_third_side_length_l3426_342612


namespace pyramid_volume_l3426_342624

/-- Given a triangular pyramid SABC with base ABC being an equilateral triangle
    with side length a and edge SA = b, where the lateral faces are congruent,
    this theorem proves the possible volumes of the pyramid based on the
    relationship between a and b. -/
theorem pyramid_volume (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let V1 := a^2 / 12 * Real.sqrt (3 * b^2 - a^2)
  let V2 := a^2 * Real.sqrt 3 / 12 * Real.sqrt (b^2 - a^2)
  let V3 := a^2 * Real.sqrt 3 / 12 * Real.sqrt (b^2 - 3 * a^2)
  (a / Real.sqrt 3 < b ∧ b ≤ a → volume_pyramid = V1) ∧
  (a < b ∧ b ≤ a * Real.sqrt 3 → volume_pyramid = V1 ∨ volume_pyramid = V2) ∧
  (b > a * Real.sqrt 3 → volume_pyramid = V1 ∨ volume_pyramid = V2 ∨ volume_pyramid = V3) :=
by sorry

def volume_pyramid : ℝ := sorry

end pyramid_volume_l3426_342624


namespace x_gt_1_sufficient_not_necessary_for_abs_x_gt_1_l3426_342613

theorem x_gt_1_sufficient_not_necessary_for_abs_x_gt_1 :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧
  (∃ x : ℝ, |x| > 1 ∧ x ≤ 1) :=
by sorry

end x_gt_1_sufficient_not_necessary_for_abs_x_gt_1_l3426_342613


namespace power_of_product_l3426_342623

theorem power_of_product (x y : ℝ) : (x^2 * y)^3 = x^6 * y^3 := by
  sorry

end power_of_product_l3426_342623


namespace greatest_integer_inequality_l3426_342602

theorem greatest_integer_inequality (y : ℤ) : 
  (5 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 7 :=
by sorry

end greatest_integer_inequality_l3426_342602


namespace jay_used_zero_l3426_342651

/-- Represents the amount of paint in a gallon -/
def gallon : ℚ := 1

/-- Represents the amount of paint Dexter used in gallons -/
def dexter_used : ℚ := 3/8

/-- Represents the amount of paint left in gallons -/
def paint_left : ℚ := 1

/-- Represents the amount of paint Jay used in gallons -/
def jay_used : ℚ := gallon - dexter_used - paint_left

theorem jay_used_zero : jay_used = 0 := by sorry

end jay_used_zero_l3426_342651


namespace ellipse_equation_1_ellipse_equation_2_x_axis_ellipse_equation_2_y_axis_l3426_342662

-- Define the ellipse type
structure Ellipse where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  c : ℝ  -- distance from center to focus
  e : ℝ  -- eccentricity

-- Define the standard equation of an ellipse
def standardEquation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

-- Theorem for the first condition
theorem ellipse_equation_1 :
  ∀ E : Ellipse,
  E.c = 6 →
  E.e = 2/3 →
  (∀ x y : ℝ, standardEquation E x y ↔ y^2/81 + x^2/45 = 1) :=
sorry

-- Theorem for the second condition (foci on x-axis)
theorem ellipse_equation_2_x_axis :
  ∀ E : Ellipse,
  E.a = 5 →
  E.c = 3 →
  (∀ x y : ℝ, standardEquation E x y ↔ x^2/25 + y^2/16 = 1) :=
sorry

-- Theorem for the second condition (foci on y-axis)
theorem ellipse_equation_2_y_axis :
  ∀ E : Ellipse,
  E.a = 5 →
  E.c = 3 →
  (∀ x y : ℝ, standardEquation E y x ↔ y^2/25 + x^2/16 = 1) :=
sorry

end ellipse_equation_1_ellipse_equation_2_x_axis_ellipse_equation_2_y_axis_l3426_342662


namespace right_triangle_hypotenuse_l3426_342649

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 90 → 
    b = 120 → 
    c^2 = a^2 + b^2 → 
    c = 150 := by
sorry

end right_triangle_hypotenuse_l3426_342649


namespace parallelogram_height_l3426_342635

/-- The height of a parallelogram with given base and area -/
theorem parallelogram_height (base area height : ℝ) 
  (h_base : base = 14)
  (h_area : area = 336)
  (h_formula : area = base * height) : height = 24 := by
  sorry

end parallelogram_height_l3426_342635


namespace complement_of_A_relative_to_U_l3426_342636

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4, 5}

theorem complement_of_A_relative_to_U :
  (U \ A) = {2} := by sorry

end complement_of_A_relative_to_U_l3426_342636


namespace seventy_third_digit_is_zero_l3426_342671

/-- The number consisting of 112 ones -/
def number_of_ones : ℕ := 11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111

/-- The square of the number consisting of 112 ones -/
def square_of_ones : ℕ := number_of_ones * number_of_ones

/-- The seventy-third digit from the end of a natural number -/
def seventy_third_digit_from_end (n : ℕ) : ℕ :=
  (n / 10^72) % 10

theorem seventy_third_digit_is_zero :
  seventy_third_digit_from_end square_of_ones = 0 := by
  sorry

end seventy_third_digit_is_zero_l3426_342671


namespace tan_sum_of_roots_l3426_342621

theorem tan_sum_of_roots (α β : Real) : 
  (∃ x y : Real, x^2 + 6*x + 7 = 0 ∧ y^2 + 6*y + 7 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) → 
  Real.tan (α + β) = 1 := by
sorry

end tan_sum_of_roots_l3426_342621


namespace sequence_2011th_term_l3426_342654

def sequence_term (n : ℕ) : ℕ → ℕ
  | 0 => 52
  | (m + 1) => 
    let prev := sequence_term n m
    let last_digit := prev % 10
    let remaining := prev / 10
    last_digit ^ 2 + 2 * remaining

def is_cyclic (seq : ℕ → ℕ) (start : ℕ) (length : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ start → seq k = seq (k % length + start)

theorem sequence_2011th_term :
  ∃ (start length : ℕ),
    start > 0 ∧
    length > 0 ∧
    is_cyclic (sequence_term 0) start length ∧
    sequence_term 0 2010 = 18 := by
  sorry

end sequence_2011th_term_l3426_342654


namespace eleven_students_like_sports_l3426_342643

/-- The number of students who like basketball or cricket or both -/
def students_basketball_or_cricket (basketball : ℕ) (cricket : ℕ) (both : ℕ) : ℕ :=
  basketball + cricket - both

/-- Theorem stating that given the conditions, 11 students like basketball or cricket or both -/
theorem eleven_students_like_sports : students_basketball_or_cricket 9 8 6 = 11 := by
  sorry

end eleven_students_like_sports_l3426_342643


namespace simple_interest_time_proof_l3426_342600

/-- The simple interest rate per annum -/
def simple_interest_rate : ℚ := 8 / 100

/-- The principal amount for simple interest -/
def simple_principal : ℚ := 1750.000000000002

/-- The principal amount for compound interest -/
def compound_principal : ℚ := 4000

/-- The compound interest rate per annum -/
def compound_interest_rate : ℚ := 10 / 100

/-- The time period for compound interest in years -/
def compound_time : ℕ := 2

/-- Function to calculate compound interest -/
def compound_interest (p : ℚ) (r : ℚ) (t : ℕ) : ℚ :=
  p * ((1 + r) ^ t - 1)

/-- The time period for simple interest in years -/
def simple_time : ℕ := 3

theorem simple_interest_time_proof :
  simple_principal * simple_interest_rate * simple_time =
  (1 / 2) * compound_interest compound_principal compound_interest_rate compound_time :=
by sorry

end simple_interest_time_proof_l3426_342600


namespace largest_four_digit_divisible_by_five_l3426_342686

theorem largest_four_digit_divisible_by_five : ∃ n : ℕ,
  n = 9995 ∧
  n ≥ 1000 ∧ n < 10000 ∧
  n % 5 = 0 ∧
  ∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 5 = 0 → m ≤ n :=
by sorry

end largest_four_digit_divisible_by_five_l3426_342686


namespace airline_capacity_is_2482_l3426_342601

/-- Calculates the number of passengers an airline can accommodate daily --/
def airline_capacity (small_planes medium_planes large_planes : ℕ)
  (small_rows small_seats small_flights small_occupancy : ℕ)
  (medium_rows medium_seats medium_flights medium_occupancy : ℕ)
  (large_rows large_seats large_flights large_occupancy : ℕ) : ℕ :=
  let small_capacity := small_planes * small_rows * small_seats * small_flights * small_occupancy / 100
  let medium_capacity := medium_planes * medium_rows * medium_seats * medium_flights * medium_occupancy / 100
  let large_capacity := large_planes * large_rows * large_seats * large_flights * large_occupancy / 100
  small_capacity + medium_capacity + large_capacity

/-- The airline's daily passenger capacity is 2482 --/
theorem airline_capacity_is_2482 :
  airline_capacity 2 2 1 15 6 3 80 25 8 2 90 35 10 4 95 = 2482 := by
  sorry

end airline_capacity_is_2482_l3426_342601


namespace ten_streets_intersections_l3426_342628

/-- The number of intersections created by n non-parallel streets -/
def intersections (n : ℕ) : ℕ := n.choose 2

/-- The theorem stating that 10 non-parallel streets create 45 intersections -/
theorem ten_streets_intersections : intersections 10 = 45 := by
  sorry

end ten_streets_intersections_l3426_342628


namespace rhombus_converse_and_inverse_false_l3426_342660

-- Define what it means for a polygon to be a rhombus
def is_rhombus (p : Polygon) : Prop := sorry

-- Define what it means for a polygon to have all sides of equal length
def has_equal_sides (p : Polygon) : Prop := sorry

-- Define a polygon (we don't need to specify its properties here)
def Polygon : Type := sorry

theorem rhombus_converse_and_inverse_false :
  (∃ p : Polygon, has_equal_sides p ∧ ¬is_rhombus p) ∧
  (∃ p : Polygon, ¬is_rhombus p ∧ has_equal_sides p) :=
sorry

end rhombus_converse_and_inverse_false_l3426_342660


namespace race_finish_distance_l3426_342663

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance : ℝ

/-- Represents the state of the race -/
structure Race where
  sasha : Runner
  lyosha : Runner
  kolya : Runner
  raceLength : ℝ

/-- The theorem to be proved -/
theorem race_finish_distance (r : Race) : 
  r.raceLength = 100 ∧ 
  r.sasha.distance - r.lyosha.distance = 10 ∧
  r.lyosha.distance - r.kolya.distance = 10 ∧
  r.sasha.distance = r.raceLength →
  r.sasha.distance - r.kolya.distance = 19 := by
  sorry

end race_finish_distance_l3426_342663


namespace vector_difference_magnitude_l3426_342667

/-- Given vectors a and b in R^2, prove that their difference has magnitude 1 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) :
  a.1 = Real.cos (15 * π / 180) ∧
  a.2 = Real.sin (15 * π / 180) ∧
  b.1 = Real.sin (15 * π / 180) ∧
  b.2 = Real.cos (15 * π / 180) →
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 1 := by
  sorry

#check vector_difference_magnitude

end vector_difference_magnitude_l3426_342667


namespace no_positive_integer_solution_l3426_342650

theorem no_positive_integer_solution :
  ¬ ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ 3 * a^2 = b^2 + 1 := by
  sorry

end no_positive_integer_solution_l3426_342650


namespace dayan_20th_term_dayan_even_term_formula_l3426_342687

def dayan_sequence : ℕ → ℕ
| 0 => 0
| 1 => 2
| 2 => 4
| 3 => 8
| 4 => 12
| 5 => 18
| 6 => 24
| 7 => 32
| 8 => 40
| 9 => 50
| n + 10 => dayan_sequence n  -- placeholder for terms beyond 10th

theorem dayan_20th_term : dayan_sequence 19 = 200 := by
  sorry

theorem dayan_even_term_formula (n : ℕ) : dayan_sequence (2 * n - 1) = 2 * n^2 := by
  sorry

end dayan_20th_term_dayan_even_term_formula_l3426_342687


namespace sheep_to_horse_ratio_l3426_342653

def daily_horse_food_per_horse : ℕ := 230
def total_daily_horse_food : ℕ := 12880
def number_of_sheep : ℕ := 56

theorem sheep_to_horse_ratio :
  (total_daily_horse_food / daily_horse_food_per_horse = number_of_sheep) →
  (number_of_sheep : ℚ) / (total_daily_horse_food / daily_horse_food_per_horse : ℚ) = 1 := by
sorry

end sheep_to_horse_ratio_l3426_342653


namespace fraction_evaluation_l3426_342603

theorem fraction_evaluation (a b : ℝ) (ha : a = 7) (hb : b = 3) :
  5 / (a + b) = 1 / 2 := by
  sorry

end fraction_evaluation_l3426_342603


namespace factorization_2m_cubed_minus_8m_l3426_342615

theorem factorization_2m_cubed_minus_8m (m : ℝ) : 2*m^3 - 8*m = 2*m*(m+2)*(m-2) := by
  sorry

end factorization_2m_cubed_minus_8m_l3426_342615


namespace unique_triple_solution_l3426_342640

theorem unique_triple_solution : 
  ∀ (a b p : ℕ+), 
    Nat.Prime p.val → 
    (a.val + b.val : ℕ) ^ p.val = p.val ^ a.val + p.val ^ b.val → 
    a = 1 ∧ b = 1 ∧ p = 2 := by
  sorry

end unique_triple_solution_l3426_342640


namespace max_min_difference_r_l3426_342656

theorem max_min_difference_r (p q r : ℝ) 
  (sum_condition : p + q + r = 5)
  (sum_squares_condition : p^2 + q^2 + r^2 = 27) :
  ∃ (r_max r_min : ℝ),
    (∀ r' : ℝ, p + q + r' = 5 ∧ p^2 + q^2 + r'^2 = 27 → r' ≤ r_max) ∧
    (∀ r' : ℝ, p + q + r' = 5 ∧ p^2 + q^2 + r'^2 = 27 → r' ≥ r_min) ∧
    r_max - r_min = 8 * Real.sqrt 7 / 3 :=
by sorry

end max_min_difference_r_l3426_342656


namespace no_natural_solution_l3426_342604

theorem no_natural_solution :
  ¬ ∃ (x y : ℕ), x^4 - y^4 = x^3 + y^3 := by
  sorry

end no_natural_solution_l3426_342604


namespace power_function_not_through_origin_l3426_342674

theorem power_function_not_through_origin (m : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (m^2 - 3*m + 3) * x^(m^2 - m - 2) ≠ 0) →
  m = 1 ∨ m = 2 := by
  sorry

end power_function_not_through_origin_l3426_342674


namespace interview_score_is_85_l3426_342672

/-- Calculate the interview score based on individual scores and their proportions -/
def interview_score (basic_knowledge : ℝ) (communication_skills : ℝ) (work_attitude : ℝ) 
  (basic_prop : ℝ) (comm_prop : ℝ) (attitude_prop : ℝ) : ℝ :=
  basic_knowledge * basic_prop + communication_skills * comm_prop + work_attitude * attitude_prop

/-- Theorem: The interview score for the given scores and proportions is 85 points -/
theorem interview_score_is_85 :
  interview_score 85 80 88 0.2 0.3 0.5 = 85 := by
  sorry

#eval interview_score 85 80 88 0.2 0.3 0.5

end interview_score_is_85_l3426_342672


namespace smallest_positive_angle_theorem_l3426_342608

theorem smallest_positive_angle_theorem (θ : Real) : 
  (θ > 0) → 
  (10 * Real.sin θ * (Real.cos θ)^3 - 10 * (Real.sin θ)^3 * Real.cos θ = Real.sqrt 2) →
  (∀ φ, φ > 0 → 10 * Real.sin φ * (Real.cos φ)^3 - 10 * (Real.sin φ)^3 * Real.cos φ = Real.sqrt 2 → θ ≤ φ) →
  θ = (1/4) * Real.arcsin ((2 * Real.sqrt 2) / 5) :=
by sorry

end smallest_positive_angle_theorem_l3426_342608


namespace parallel_lines_a_value_l3426_342633

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- Definition of the first line -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0

/-- Definition of the second line -/
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + a^2 - 1 = 0

/-- The theorem to be proved -/
theorem parallel_lines_a_value :
  ∃ a : ℝ, (∀ x y : ℝ, line1 a x y ↔ line2 a x y) ∧ a = -1 :=
sorry

end parallel_lines_a_value_l3426_342633


namespace replacement_count_l3426_342611

-- Define the replacement percentage
def replacement_percentage : ℝ := 0.2

-- Define the final milk percentage
def final_milk_percentage : ℝ := 0.5120000000000001

-- Define the function to calculate the remaining milk percentage after n replacements
def remaining_milk (n : ℕ) : ℝ := (1 - replacement_percentage) ^ n

-- Theorem statement
theorem replacement_count : ∃ n : ℕ, remaining_milk n = final_milk_percentage ∧ n = 3 := by
  sorry

end replacement_count_l3426_342611


namespace erica_money_proof_l3426_342677

def total_money : ℕ := 91
def sam_money : ℕ := 38

theorem erica_money_proof :
  total_money - sam_money = 53 := by
  sorry

end erica_money_proof_l3426_342677


namespace solve_cricket_problem_l3426_342630

def cricket_problem (W : ℝ) : Prop :=
  let crickets_90F : ℝ := 4
  let crickets_100F : ℝ := 2 * crickets_90F
  let prop_90F : ℝ := 0.8
  let prop_100F : ℝ := 1 - prop_90F
  let total_crickets : ℝ := 72
  W * (crickets_90F * prop_90F + crickets_100F * prop_100F) = total_crickets

theorem solve_cricket_problem :
  ∃ W : ℝ, cricket_problem W ∧ W = 15 := by
  sorry

end solve_cricket_problem_l3426_342630


namespace max_sum_of_squares_l3426_342692

theorem max_sum_of_squares (x y z : ℝ) 
  (h1 : x + y = z - 1) 
  (h2 : x * y = z^2 - 7*z + 14) : 
  (∃ (max : ℝ), ∀ (x' y' z' : ℝ), 
    x' + y' = z' - 1 → 
    x' * y' = z'^2 - 7*z' + 14 → 
    x'^2 + y'^2 ≤ max ∧ 
    (x'^2 + y'^2 = max ↔ z' = 3) ∧ 
    max = 2) :=
sorry

end max_sum_of_squares_l3426_342692


namespace min_distance_sum_l3426_342689

noncomputable def circle_C (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 8

noncomputable def circle_D (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 5

def origin : ℝ × ℝ := (0, 0)

theorem min_distance_sum (P Q : ℝ × ℝ) (hP : circle_C P.1 P.2) (hQ : circle_D Q.1 Q.2) :
  ∃ (min_val : ℝ), min_val = Real.sqrt 5 ∧
  ∀ (P' Q' : ℝ × ℝ), circle_C P'.1 P'.2 → circle_D Q'.1 Q'.2 →
    Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) + 
    (Real.sqrt 2 / 2) * Real.sqrt (P'.1^2 + P'.2^2) ≥ min_val :=
by sorry

end min_distance_sum_l3426_342689


namespace hearts_then_king_probability_l3426_342645

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size : cards.card = 52)
  (suits : ∀ c ∈ cards, c.1 ∈ Finset.range 4)
  (ranks : ∀ c ∈ cards, c.2 ∈ Finset.range 13)

/-- The probability of drawing a specific sequence of cards from a shuffled deck -/
def draw_probability (d : Deck) (seq : List (Nat × Nat)) : ℚ :=
  sorry

/-- Hearts suit is represented by 0 -/
def hearts : Nat := 0

/-- King rank is represented by 12 (0-indexed) -/
def king : Nat := 12

theorem hearts_then_king_probability :
  ∀ d : Deck, 
    draw_probability d [(hearts, 0), (hearts, 1), (hearts, 2), (hearts, 3), (0, king)] = 286 / 124900 := by
  sorry

end hearts_then_king_probability_l3426_342645


namespace rectangle_area_problem_l3426_342661

theorem rectangle_area_problem : ∃ (x y : ℝ), 
  (x + 3.5) * (y - 1.5) = x * y ∧ 
  (x - 3.5) * (y + 2) = x * y ∧ 
  x * y = 294 := by
sorry

end rectangle_area_problem_l3426_342661


namespace point_movement_to_y_axis_l3426_342642

/-- Given a point P that is moved 1 unit to the right to point M on the y-axis, 
    prove that M has coordinates (0, -2) -/
theorem point_movement_to_y_axis (m : ℝ) : 
  let P : ℝ × ℝ := (m + 2, 2 * m + 4)
  let M : ℝ × ℝ := (P.1 + 1, P.2)
  M.1 = 0 → M = (0, -2) := by
  sorry

end point_movement_to_y_axis_l3426_342642
