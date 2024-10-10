import Mathlib

namespace exists_number_with_nine_nines_squared_l3109_310926

theorem exists_number_with_nine_nines_squared : ∃ n : ℕ, 
  ∃ k : ℕ, n^2 = 999999999 * 10^k + m ∧ m < 10^k :=
sorry

end exists_number_with_nine_nines_squared_l3109_310926


namespace bob_dogs_count_l3109_310981

/-- Represents the number of cats Bob has -/
def num_cats : ℕ := 4

/-- Represents the portion of the food bag a single cat receives -/
def cat_portion : ℚ := 125 / 4000

/-- Represents the portion of the food bag a single dog receives -/
def dog_portion : ℚ := num_cats * cat_portion

/-- The number of dogs Bob has -/
def num_dogs : ℕ := 7

theorem bob_dogs_count :
  (num_dogs : ℚ) * dog_portion + (num_cats : ℚ) * cat_portion = 1 :=
sorry

#check bob_dogs_count

end bob_dogs_count_l3109_310981


namespace cu2co3_2_weight_calculation_l3109_310984

-- Define the chemical equation coefficients
def cu_no3_2_coeff : ℚ := 2
def na2co3_coeff : ℚ := 3
def cu2co3_2_coeff : ℚ := 1

-- Define the available moles of reactants
def cu_no3_2_moles : ℚ := 1.85
def na2co3_moles : ℚ := 3.21

-- Define the molar mass of Cu2(CO3)2
def cu2co3_2_molar_mass : ℚ := 247.12

-- Define the function to calculate the limiting reactant
def limiting_reactant (cu_no3_2 : ℚ) (na2co3 : ℚ) : ℚ :=
  min (cu_no3_2 / cu_no3_2_coeff) (na2co3 / na2co3_coeff)

-- Define the function to calculate the moles of Cu2(CO3)2 produced
def cu2co3_2_produced (limiting : ℚ) : ℚ :=
  limiting * (cu2co3_2_coeff / cu_no3_2_coeff)

-- Define the function to calculate the weight of Cu2(CO3)2 produced
def cu2co3_2_weight (moles : ℚ) : ℚ :=
  moles * cu2co3_2_molar_mass

-- Theorem statement
theorem cu2co3_2_weight_calculation :
  cu2co3_2_weight (cu2co3_2_produced (limiting_reactant cu_no3_2_moles na2co3_moles)) = 228.586 := by
  sorry

end cu2co3_2_weight_calculation_l3109_310984


namespace election_votes_theorem_l3109_310931

theorem election_votes_theorem (total_votes : ℕ) (winner_votes second_votes third_votes : ℕ) :
  (winner_votes : ℚ) = 45 / 100 * total_votes ∧
  (second_votes : ℚ) = 35 / 100 * total_votes ∧
  winner_votes = second_votes + 150 ∧
  winner_votes + second_votes + third_votes = total_votes →
  total_votes = 1500 ∧ winner_votes = 675 ∧ second_votes = 525 ∧ third_votes = 300 := by
  sorry

end election_votes_theorem_l3109_310931


namespace olivias_chips_quarters_l3109_310911

/-- The number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- The total amount Olivia pays in dollars -/
def total_dollars : ℕ := 4

/-- The number of quarters Olivia pays for soda -/
def quarters_for_soda : ℕ := 12

/-- The number of quarters Olivia pays for chips -/
def quarters_for_chips : ℕ := total_dollars * quarters_per_dollar - quarters_for_soda

theorem olivias_chips_quarters : quarters_for_chips = 4 := by
  sorry

end olivias_chips_quarters_l3109_310911


namespace bisection_solves_x_squared_minus_two_program_flowchart_l3109_310997

/-- Represents different types of flowcharts -/
inductive FlowchartType
  | Process
  | Program
  | KnowledgeStructure
  | OrganizationalStructure

/-- Represents a method for solving equations -/
inductive SolvingMethod
  | Bisection
  | Newton
  | Secant

/-- Represents an equation to be solved -/
structure Equation where
  f : ℝ → ℝ

/-- Determines the type of flowchart used to solve an equation using a specific method -/
def flowchartTypeForSolving (eq : Equation) (method : SolvingMethod) : FlowchartType :=
  sorry

/-- The theorem stating that solving x^2 - 2 = 0 using the bisection method results in a program flowchart -/
theorem bisection_solves_x_squared_minus_two_program_flowchart :
  flowchartTypeForSolving { f := fun x => x^2 - 2 } SolvingMethod.Bisection = FlowchartType.Program :=
  sorry

end bisection_solves_x_squared_minus_two_program_flowchart_l3109_310997


namespace c_minus_three_equals_negative_two_l3109_310967

/-- An invertible function g : ℝ → ℝ -/
def g : ℝ → ℝ :=
  sorry

/-- c is a real number such that g(c) = 3 and g(3) = c -/
def c : ℝ :=
  sorry

theorem c_minus_three_equals_negative_two (h1 : Function.Injective g) (h2 : g c = 3) (h3 : g 3 = c) :
  c - 3 = -2 :=
sorry

end c_minus_three_equals_negative_two_l3109_310967


namespace car_speed_problem_l3109_310913

/-- A car travels uphill and downhill. This theorem proves the downhill speed given certain conditions. -/
theorem car_speed_problem (uphill_speed : ℝ) (total_distance : ℝ) (total_time uphill_time downhill_time : ℝ) 
  (h1 : uphill_speed = 30)
  (h2 : total_distance = 650)
  (h3 : total_time = 15)
  (h4 : uphill_time = 5)
  (h5 : downhill_time = 5) :
  ∃ downhill_speed : ℝ, 
    downhill_speed * downhill_time + uphill_speed * uphill_time = total_distance ∧ 
    downhill_speed = 100 :=
by sorry

end car_speed_problem_l3109_310913


namespace susan_spending_l3109_310937

theorem susan_spending (initial_amount : ℝ) (h1 : initial_amount = 600) : 
  let after_clothes := initial_amount / 2
  let after_books := after_clothes / 2
  after_books = 150 := by
sorry

end susan_spending_l3109_310937


namespace field_ratio_proof_l3109_310906

/-- Proves that for a rectangular field with length 24 meters and width 13.5 meters,
    the ratio of twice the width to the length is 9:8. -/
theorem field_ratio_proof (length width : ℝ) : 
  length = 24 → width = 13.5 → (2 * width) / length = 9 / 8 := by
  sorry

end field_ratio_proof_l3109_310906


namespace quadratic_not_in_third_quadrant_l3109_310993

/-- A linear function passing through the first, third, and fourth quadrants -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  a_nonzero : a ≠ 0
  passes_first_quadrant : ∃ x > 0, -a * x + b > 0
  passes_third_quadrant : ∃ x < 0, -a * x + b < 0
  passes_fourth_quadrant : ∃ x > 0, -a * x + b < 0

/-- The corresponding quadratic function -/
def quadratic_function (f : LinearFunction) (x : ℝ) : ℝ :=
  -f.a * x^2 + f.b * x

/-- Theorem stating that the quadratic function does not pass through the third quadrant -/
theorem quadratic_not_in_third_quadrant (f : LinearFunction) :
  ¬∃ x < 0, quadratic_function f x < 0 := by
  sorry


end quadratic_not_in_third_quadrant_l3109_310993


namespace original_list_size_l3109_310927

theorem original_list_size (n : ℕ) (m : ℚ) : 
  (m + 3) * (n + 1) = m * n + 20 →
  (m + 1) * (n + 2) = (m + 3) * (n + 1) + 2 →
  n = 7 := by
sorry

end original_list_size_l3109_310927


namespace factors_72_l3109_310995

/-- The number of distinct positive factors of 72 -/
def num_factors_72 : ℕ := sorry

/-- Theorem stating that the number of distinct positive factors of 72 is 12 -/
theorem factors_72 : num_factors_72 = 12 := by sorry

end factors_72_l3109_310995


namespace probability_sum_10_l3109_310916

def die_faces : Nat := 6

def total_outcomes : Nat := die_faces * die_faces

def favorable_outcomes : Nat := 3 * 2 - 1

theorem probability_sum_10 : 
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 6 := by sorry

end probability_sum_10_l3109_310916


namespace smaller_area_with_center_l3109_310919

/-- Represents a circular sector with a central angle of 60 degrees -/
structure Sector60 where
  radius : ℝ
  center : ℝ × ℝ

/-- Represents the line that cuts the sector -/
structure CuttingLine where
  slope : ℝ
  intercept : ℝ

/-- Represents the two parts after cutting the sector -/
structure SectorParts where
  part_with_center : Set (ℝ × ℝ)
  other_part : Set (ℝ × ℝ)

/-- Function to cut the sector -/
def cut_sector (s : Sector60) (l : CuttingLine) : SectorParts := sorry

/-- Function to calculate perimeter of a part -/
def perimeter (part : Set (ℝ × ℝ)) : ℝ := sorry

/-- Function to calculate area of a part -/
def area (part : Set (ℝ × ℝ)) : ℝ := sorry

/-- The main theorem -/
theorem smaller_area_with_center (s : Sector60) :
  ∃ (l : CuttingLine),
    let parts := cut_sector s l
    perimeter parts.part_with_center = perimeter parts.other_part →
    area parts.part_with_center < area parts.other_part :=
  sorry

end smaller_area_with_center_l3109_310919


namespace change_ratio_for_quadratic_function_l3109_310983

/-- Given a function f(x) = 2x^2 - 4, prove that the ratio of change in y to change in x
    between the points (1, -2) and (1 + Δx, -2 + Δy) is equal to 4 + 2Δx -/
theorem change_ratio_for_quadratic_function (Δx Δy : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 - 4
  f 1 = -2 →
  f (1 + Δx) = -2 + Δy →
  Δy / Δx = 4 + 2 * Δx :=
by
  sorry

end change_ratio_for_quadratic_function_l3109_310983


namespace smallest_norm_w_l3109_310968

variable (w : ℝ × ℝ)

def v : ℝ × ℝ := (4, 2)

theorem smallest_norm_w (h : ‖w + v‖ = 10) :
  ∃ (w_min : ℝ × ℝ), ‖w_min‖ = 10 - 2 * Real.sqrt 5 ∧ ∀ w', ‖w' + v‖ = 10 → ‖w'‖ ≥ ‖w_min‖ :=
sorry

end smallest_norm_w_l3109_310968


namespace even_function_sum_a_b_l3109_310964

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 3) * x + 3

-- Define the property of being an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Main theorem
theorem even_function_sum_a_b :
  ∀ a b : ℝ,
  (∀ x, x ∈ Set.Icc (2 * a - 3) (4 - a) → f a b x = f a b (-x)) →
  a + b = 2 :=
by sorry

end even_function_sum_a_b_l3109_310964


namespace okinawa_sales_ratio_l3109_310933

/-- Proves the ratio of Okinawa-flavored milk tea sales to total sales -/
theorem okinawa_sales_ratio (total_sales : ℕ) (winter_melon_sales : ℕ) (chocolate_sales : ℕ) 
  (h1 : total_sales = 50)
  (h2 : winter_melon_sales = 2 * total_sales / 5)
  (h3 : chocolate_sales = 15)
  (h4 : winter_melon_sales + chocolate_sales + (total_sales - winter_melon_sales - chocolate_sales) = total_sales) :
  (total_sales - winter_melon_sales - chocolate_sales) / total_sales = 3 / 10 := by
  sorry

end okinawa_sales_ratio_l3109_310933


namespace three_number_sum_l3109_310973

theorem three_number_sum (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : (a + b + c) / 3 = a + 8)
  (h4 : (a + b + c) / 3 = c - 18)
  (h5 : b = 12) :
  a + b + c = 66 := by
  sorry

end three_number_sum_l3109_310973


namespace log_arithmetic_mean_implies_geometric_mean_geometric_mean_not_implies_log_arithmetic_mean_l3109_310900

-- Define the arithmetic mean of logarithms
def log_arithmetic_mean (x y z : ℝ) : Prop :=
  Real.log y = (Real.log x + Real.log z) / 2

-- Define the geometric mean
def geometric_mean (x y z : ℝ) : Prop :=
  y ^ 2 = x * z

theorem log_arithmetic_mean_implies_geometric_mean
  (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) :
  log_arithmetic_mean x y z → geometric_mean x y z :=
sorry

theorem geometric_mean_not_implies_log_arithmetic_mean :
  ∃ x y z : ℝ, geometric_mean x y z ∧ ¬log_arithmetic_mean x y z :=
sorry

end log_arithmetic_mean_implies_geometric_mean_geometric_mean_not_implies_log_arithmetic_mean_l3109_310900


namespace jackie_apple_count_l3109_310915

/-- Given that Adam has 9 apples and 3 more apples than Jackie, prove that Jackie has 6 apples. -/
theorem jackie_apple_count (adam_apple_count : ℕ) (adam_extra_apples : ℕ) (jackie_apple_count : ℕ)
  (h1 : adam_apple_count = 9)
  (h2 : adam_apple_count = jackie_apple_count + adam_extra_apples)
  (h3 : adam_extra_apples = 3) :
  jackie_apple_count = 6 := by
  sorry

end jackie_apple_count_l3109_310915


namespace stamps_collection_theorem_l3109_310930

def kylie_stamps : ℕ := 34
def nelly_stamps_difference : ℕ := 44

def total_stamps : ℕ := kylie_stamps + (kylie_stamps + nelly_stamps_difference)

theorem stamps_collection_theorem : total_stamps = 112 := by
  sorry

end stamps_collection_theorem_l3109_310930


namespace solution_set_f_geq_3_range_of_a_l3109_310935

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f x ≥ a^2 - a} = {a : ℝ | -1 ≤ a ∧ a ≤ 2} := by sorry

end solution_set_f_geq_3_range_of_a_l3109_310935


namespace middle_school_students_l3109_310925

theorem middle_school_students (elementary : ℕ) (middle : ℕ) : 
  elementary = 4 * middle - 3 →
  elementary + middle = 247 →
  middle = 50 := by
sorry

end middle_school_students_l3109_310925


namespace number_equation_solution_l3109_310956

theorem number_equation_solution :
  ∃ x : ℝ, x - (1002 / 20.04) = 1295 ∧ x = 1345 := by
  sorry

end number_equation_solution_l3109_310956


namespace semicircle_area_l3109_310969

theorem semicircle_area (diameter : ℝ) (h : diameter = 3) : 
  let radius : ℝ := diameter / 2
  let semicircle_area : ℝ := (π * radius^2) / 2
  semicircle_area = 9 * π / 8 := by
sorry

end semicircle_area_l3109_310969


namespace problem_statement_l3109_310910

theorem problem_statement (x y : ℝ) (h1 : x = 3) (h2 : y = 1) :
  let n := x - y^(2*(x+y))
  n = 2 := by sorry

end problem_statement_l3109_310910


namespace equation_roots_l3109_310946

theorem equation_roots : 
  let f : ℝ → ℝ := λ x => (x - 3)^2 - (x - 3)
  (f 3 = 0 ∧ f 4 = 0) ∧ ∀ x : ℝ, f x = 0 → (x = 3 ∨ x = 4) :=
by sorry

end equation_roots_l3109_310946


namespace coordinates_wrt_origin_l3109_310988

/-- The coordinates of a point with respect to the origin are the same as its given coordinates. -/
theorem coordinates_wrt_origin (x y : ℝ) : 
  let A : ℝ × ℝ := (x, y)
  A = A :=
by sorry

end coordinates_wrt_origin_l3109_310988


namespace smallest_square_enclosing_circle_l3109_310920

theorem smallest_square_enclosing_circle (r : ℝ) (h : r = 5) : 
  (2 * r) ^ 2 = 100 := by
  sorry

end smallest_square_enclosing_circle_l3109_310920


namespace disjunction_not_implies_both_true_l3109_310939

theorem disjunction_not_implies_both_true :
  ¬(∀ (p q : Prop), (p ∨ q) → (p ∧ q)) := by sorry

end disjunction_not_implies_both_true_l3109_310939


namespace acid_concentration_problem_l3109_310934

theorem acid_concentration_problem (acid1 acid2 acid3 : ℝ) (water : ℝ) :
  acid1 = 10 →
  acid2 = 20 →
  acid3 = 30 →
  acid1 / (acid1 + (water * (1/20))) = 1/20 →
  acid2 / (acid2 + (water * (13/30))) = 7/30 →
  acid3 / (acid3 + water) = 21/200 :=
by sorry

end acid_concentration_problem_l3109_310934


namespace unique_divisible_by_101_l3109_310965

theorem unique_divisible_by_101 : ∃! n : ℕ, 
  201300 ≤ n ∧ n < 201400 ∧ n % 101 = 0 :=
by
  sorry

end unique_divisible_by_101_l3109_310965


namespace two_digit_reverse_sum_l3109_310928

theorem two_digit_reverse_sum (x y n : ℕ) : 
  (10 ≤ x ∧ x < 100) →  -- x is a two-digit integer
  (10 ≤ y ∧ y < 100) →  -- y is a two-digit integer
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ x = 10 * a + b ∧ y = 10 * b + a) →  -- y is obtained by reversing the digits of x
  x^2 + y^2 = n^2 →  -- x^2 + y^2 = n^2
  x + y + n = 132 := by
sorry

end two_digit_reverse_sum_l3109_310928


namespace parabola_directrix_l3109_310972

/-- Given a parabola with equation x² = 4y, its directrix has equation y = -1 -/
theorem parabola_directrix (x y : ℝ) : x^2 = 4*y → (∃ (k : ℝ), k = -1 ∧ y = k) := by
  sorry

end parabola_directrix_l3109_310972


namespace M_definition_sum_of_digits_M_l3109_310948

def M : ℕ := sorry

-- M is the smallest positive integer divisible by every positive integer less than 8
theorem M_definition : 
  M > 0 ∧ 
  (∀ k : ℕ, k > 0 → k < 8 → M % k = 0) ∧
  (∀ n : ℕ, n > 0 → (∀ k : ℕ, k > 0 → k < 8 → n % k = 0) → n ≥ M) :=
sorry

-- Function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- Theorem stating that the sum of digits of M is 6
theorem sum_of_digits_M : sum_of_digits M = 6 :=
sorry

end M_definition_sum_of_digits_M_l3109_310948


namespace marikas_fathers_age_l3109_310938

/-- Given that Marika was 10 years old in 2006 and her father's age was five times her age,
    prove that the year when Marika's father's age will be twice her age is 2036. -/
theorem marikas_fathers_age (marika_birth_year : ℕ) (father_birth_year : ℕ) : 
  marika_birth_year = 1996 →
  father_birth_year = 1956 →
  ∃ (year : ℕ), year = 2036 ∧ 
    (year - father_birth_year) = 2 * (year - marika_birth_year) :=
by sorry

end marikas_fathers_age_l3109_310938


namespace chess_pieces_count_l3109_310949

theorem chess_pieces_count (black_pieces : ℕ) (prob_black : ℚ) (white_pieces : ℕ) : 
  black_pieces = 6 → 
  prob_black = 1 / 5 → 
  (black_pieces : ℚ) / ((black_pieces : ℚ) + (white_pieces : ℚ)) = prob_black →
  white_pieces = 24 := by
sorry

end chess_pieces_count_l3109_310949


namespace smallest_n_factorial_divisible_by_2016_smallest_n_factorial_divisible_by_2016_pow_10_l3109_310962

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem smallest_n_factorial_divisible_by_2016 :
  ∀ n : ℕ, n < 8 → ¬(factorial n % 2016 = 0) ∧ (factorial 8 % 2016 = 0) := by sorry

theorem smallest_n_factorial_divisible_by_2016_pow_10 :
  ∀ n : ℕ, n < 63 → ¬(factorial n % (2016^10) = 0) ∧ (factorial 63 % (2016^10) = 0) := by sorry

end smallest_n_factorial_divisible_by_2016_smallest_n_factorial_divisible_by_2016_pow_10_l3109_310962


namespace set_equality_implies_values_l3109_310999

def A (a b : ℝ) : Set ℝ := {2, a, b}
def B (b : ℝ) : Set ℝ := {0, 2, b^2 - 2}

theorem set_equality_implies_values (a b : ℝ) :
  A a b = B b → ((a = 0 ∧ b = -1) ∨ (a = -2 ∧ b = 0)) :=
by sorry

end set_equality_implies_values_l3109_310999


namespace symmetry_of_point_l3109_310986

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricToOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

/-- The theorem statement -/
theorem symmetry_of_point :
  let A : Point := ⟨3, -2⟩
  let A' : Point := ⟨-3, 2⟩
  symmetricToOrigin A A' := by sorry

end symmetry_of_point_l3109_310986


namespace geometric_sequence_increasing_condition_l3109_310952

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : geometric_sequence a q) (h_pos : a 1 > 0) :
  (increasing_sequence a → q > 0) ∧
  ¬(q > 0 → increasing_sequence a) :=
by sorry

end geometric_sequence_increasing_condition_l3109_310952


namespace max_triangle_area_l3109_310982

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | (p.1^2 + (p.2 - 2)^2 = 4)}

-- Define the line l
def l : Set (ℝ × ℝ) := {p | p.1 - Real.sqrt 3 * p.2 + Real.sqrt 3 = 0}

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define a function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_triangle_area :
  ∀ P ∈ C, P ≠ A → P ≠ B →
  triangleArea P A B ≤ (4 * Real.sqrt 13 + Real.sqrt 39) / 4 :=
sorry

end max_triangle_area_l3109_310982


namespace no_positive_integers_divisible_by_three_l3109_310912

theorem no_positive_integers_divisible_by_three (n : ℕ) : 
  n > 0 ∧ 3 ∣ n → ¬(28 - 6 * n > 14) :=
by sorry

end no_positive_integers_divisible_by_three_l3109_310912


namespace polynomial_identity_l3109_310977

theorem polynomial_identity (g : ℝ → ℝ) (h : ∀ x, g (x^2 + 2) = x^4 + 6*x^2 + 4) :
  ∀ x, g (x^2 - 2) = x^4 - 2*x^2 - 4 := by
  sorry

end polynomial_identity_l3109_310977


namespace number_of_boys_l3109_310940

theorem number_of_boys (total_students : ℕ) (boys_fraction : ℚ) (boys_count : ℕ) : 
  total_students = 12 →
  boys_fraction = 2/3 →
  boys_count = (total_students : ℚ) * boys_fraction →
  boys_count = 8 := by
sorry

end number_of_boys_l3109_310940


namespace f_minus_one_eq_neg_two_l3109_310961

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_minus_one_eq_neg_two
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_pos : ∀ x > 0, f x = 2^x) :
  f (-1) = -2 := by
sorry

end f_minus_one_eq_neg_two_l3109_310961


namespace simplest_form_iff_coprime_l3109_310953

/-- A fraction is a pair of integers where the denominator is non-zero -/
structure Fraction where
  numerator : Int
  denominator : Int
  denom_nonzero : denominator ≠ 0

/-- A fraction is in its simplest form if it cannot be reduced further -/
def is_simplest_form (f : Fraction) : Prop :=
  ∀ k : Int, k ≠ 0 → ¬(k ∣ f.numerator ∧ k ∣ f.denominator)

/-- Two integers are coprime if their greatest common divisor is 1 -/
def are_coprime (a b : Int) : Prop :=
  Int.gcd a b = 1

/-- Theorem: A fraction is in its simplest form if and only if its numerator and denominator are coprime -/
theorem simplest_form_iff_coprime (f : Fraction) :
  is_simplest_form f ↔ are_coprime f.numerator f.denominator := by
  sorry

end simplest_form_iff_coprime_l3109_310953


namespace house_transaction_loss_l3109_310921

def initial_value : ℝ := 12000
def loss_percentage : ℝ := 0.15
def gain_percentage : ℝ := 0.20

theorem house_transaction_loss :
  let first_sale := initial_value * (1 - loss_percentage)
  let second_sale := first_sale * (1 + gain_percentage)
  second_sale - initial_value = 240 := by sorry

end house_transaction_loss_l3109_310921


namespace divisibility_criterion_l3109_310905

theorem divisibility_criterion (A m k : ℕ) (h_pos : A > 0) (h_m_pos : m > 0) (h_k_pos : k > 0) :
  let g := k * m + 1
  let remainders : List ℕ := sorry
  let sum_remainders := remainders.sum
  (A % m = 0) ↔ (sum_remainders % m = 0) := by
  sorry

end divisibility_criterion_l3109_310905


namespace triangle_area_circumradius_angles_l3109_310966

theorem triangle_area_circumradius_angles 
  (α β γ : Real) (R : Real) (S_Δ : Real) :
  (α + β + γ = π) →
  (R > 0) →
  (S_Δ > 0) →
  (S_Δ = 2 * R^2 * Real.sin α * Real.sin β * Real.sin γ) :=
by sorry

end triangle_area_circumradius_angles_l3109_310966


namespace trajectory_of_moving_circle_l3109_310960

-- Define the two circles
def C₁ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 169
def C₂ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 9

-- Define the moving circle
def MovingCircle (x y r : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), C₁ x₀ y₀ ∧ C₂ x₀ y₀ ∧
  ((x - x₀)^2 + (y - y₀)^2 = r^2) ∧
  ((x - 4)^2 + y^2 = (13 - r)^2) ∧
  ((x + 4)^2 + y^2 = (r + 3)^2)

-- Theorem statement
theorem trajectory_of_moving_circle :
  ∀ (x y : ℝ), (∃ (r : ℝ), MovingCircle x y r) →
  (x^2 / 64 + y^2 / 48 = 1) :=
sorry

end trajectory_of_moving_circle_l3109_310960


namespace island_area_l3109_310992

/-- The area of a rectangular island with width 5 miles and length 10 miles is 50 square miles. -/
theorem island_area : 
  let width : ℝ := 5
  let length : ℝ := 10
  width * length = 50 := by sorry

end island_area_l3109_310992


namespace election_votes_l3109_310904

theorem election_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (52 : ℚ) / 100 * total_votes - (48 : ℚ) / 100 * total_votes = 288) : 
  ((52 : ℚ) / 100 * total_votes : ℚ).floor = 3744 := by
sorry

end election_votes_l3109_310904


namespace chord_length_circle_line_l3109_310917

/-- The chord length cut by a circle from a line -/
theorem chord_length_circle_line (r : ℝ) (a b c : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}
  let line := {(x, y) : ℝ × ℝ | ∃ t, x = a + b*t ∧ y = c + t}
  let chord_length := 2 * Real.sqrt (r^2 - (a^2 + b^2 - 2*a*c + c^2) / (b^2 + 1))
  r = 3 ∧ a = 1 ∧ b = 2 ∧ c = 2 → chord_length = 12 * Real.sqrt 5 / 5 := by
sorry

end chord_length_circle_line_l3109_310917


namespace specific_figure_area_l3109_310945

/-- A fifteen-sided figure drawn on a 1 cm × 1 cm graph paper -/
structure FifteenSidedFigure where
  /-- The number of full unit squares within the figure -/
  full_squares : ℕ
  /-- The number of rectangles within the figure -/
  rectangles : ℕ
  /-- The width of each rectangle in cm -/
  rectangle_width : ℝ
  /-- The height of each rectangle in cm -/
  rectangle_height : ℝ
  /-- The figure has fifteen sides -/
  sides : ℕ
  sides_eq : sides = 15

/-- The area of the fifteen-sided figure in cm² -/
def figure_area (f : FifteenSidedFigure) : ℝ :=
  f.full_squares + f.rectangles * f.rectangle_width * f.rectangle_height

/-- Theorem stating that the area of the specific fifteen-sided figure is 15 cm² -/
theorem specific_figure_area :
  ∃ f : FifteenSidedFigure, figure_area f = 15 := by
  sorry

end specific_figure_area_l3109_310945


namespace new_average_after_removal_l3109_310970

theorem new_average_after_removal (numbers : List ℝ) : 
  numbers.length = 12 → 
  numbers.sum / numbers.length = 90 → 
  80 ∈ numbers → 
  84 ∈ numbers → 
  let remaining := numbers.filter (λ x => x ≠ 80 ∧ x ≠ 84)
  remaining.sum / remaining.length = 91.6 := by
sorry

end new_average_after_removal_l3109_310970


namespace problem_solution_l3109_310943

theorem problem_solution : ((-4)^2) * (((-1)^2023) + (3/4) + ((-1/2)^3)) = -6 := by
  sorry

end problem_solution_l3109_310943


namespace equivalent_ratios_l3109_310942

theorem equivalent_ratios (x : ℚ) : (3 : ℚ) / 12 = 3 / x → x = 12 := by
  sorry

end equivalent_ratios_l3109_310942


namespace x_amount_proof_l3109_310918

def total_amount : ℝ := 5000
def ratio_x : ℝ := 2
def ratio_y : ℝ := 8

theorem x_amount_proof :
  let total_ratio := ratio_x + ratio_y
  let amount_per_part := total_amount / total_ratio
  let x_amount := amount_per_part * ratio_x
  x_amount = 1000 := by sorry

end x_amount_proof_l3109_310918


namespace geometric_sequence_sum_l3109_310950

/-- A geometric sequence with its sum function -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1  -- Geometric sequence property
  sum_property : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))  -- Sum formula

/-- Theorem: For a geometric sequence with S_5 = 3 and S_10 = 9, S_15 = 21 -/
theorem geometric_sequence_sum (seq : GeometricSequence) 
  (h1 : seq.S 5 = 3) 
  (h2 : seq.S 10 = 9) : 
  seq.S 15 = 21 := by
  sorry

end geometric_sequence_sum_l3109_310950


namespace probability_white_balls_l3109_310975

def total_balls : ℕ := 16
def white_balls : ℕ := 8
def black_balls : ℕ := 8
def drawn_balls : ℕ := 2

theorem probability_white_balls (total_balls white_balls black_balls drawn_balls : ℕ) 
  (h1 : total_balls = white_balls + black_balls)
  (h2 : total_balls = 16)
  (h3 : white_balls = 8)
  (h4 : black_balls = 8)
  (h5 : drawn_balls = 2) :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 7 / 30 ∧
  1 - (Nat.choose black_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 23 / 30 :=
by sorry

end probability_white_balls_l3109_310975


namespace average_of_numbers_l3109_310959

def numbers : List ℕ := [1200, 1300, 1400, 1510, 1520, 1530, 1200]

theorem average_of_numbers : (numbers.sum / numbers.length : ℚ) = 1380 := by
  sorry

end average_of_numbers_l3109_310959


namespace expected_hearts_in_modified_deck_l3109_310991

/-- A circular arrangement of cards -/
structure CircularDeck :=
  (total : ℕ)
  (hearts : ℕ)
  (h_total : total ≥ hearts)

/-- Expected number of adjacent heart pairs in a circular arrangement -/
def expected_adjacent_hearts (deck : CircularDeck) : ℚ :=
  (deck.hearts : ℚ) * (deck.hearts - 1) / (deck.total - 1)

theorem expected_hearts_in_modified_deck :
  let deck := CircularDeck.mk 40 10 (by norm_num)
  expected_adjacent_hearts deck = 30 / 13 := by
sorry

end expected_hearts_in_modified_deck_l3109_310991


namespace square_area_problem_l3109_310990

theorem square_area_problem (x : ℝ) (h : 3.5 * x * (x - 30) = 2 * x^2) : x^2 = 4900 := by
  sorry

end square_area_problem_l3109_310990


namespace shaded_area_regular_octagon_l3109_310922

/-- The area of the shaded region in a regular octagon with side length 8 cm,
    formed by connecting the midpoints of consecutive sides. -/
theorem shaded_area_regular_octagon (s : ℝ) (h : s = 8) :
  let outer_area := 2 * (1 + Real.sqrt 2) * s^2
  let inner_side := s * (1 - Real.sqrt 2 / 2)
  let inner_area := 2 * (1 + Real.sqrt 2) * inner_side^2
  outer_area - inner_area = 128 * (1 + Real.sqrt 2) - 2 * (1 + Real.sqrt 2) * (8 - 4 * Real.sqrt 2)^2 :=
by sorry

end shaded_area_regular_octagon_l3109_310922


namespace parabola_equation_l3109_310980

/-- A parabola with focus on the x-axis, vertex at the origin, and opening to the right -/
structure RightParabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  h1 : ∀ x y, eq x y ↔ y^2 = 2*p*x
  h2 : p > 0

/-- The point (1, 2) lies on the parabola -/
def PassesThroughPoint (par : RightParabola) : Prop :=
  par.eq 1 2

theorem parabola_equation (par : RightParabola) (h : PassesThroughPoint par) :
  par.p = 2 ∧ ∀ x y, par.eq x y ↔ y^2 = 4*x := by
  sorry

end parabola_equation_l3109_310980


namespace inequality_equivalence_l3109_310998

theorem inequality_equivalence (x : ℝ) : (3 + x) * (2 - x) < 0 ↔ x > 2 ∨ x < -3 := by sorry

end inequality_equivalence_l3109_310998


namespace absolute_value_equation_solution_l3109_310994

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |2*x + 3| - |x - 1| = 4*x - 3 :=
by
  -- The unique solution is 7/3
  use 7/3
  sorry

end absolute_value_equation_solution_l3109_310994


namespace max_square_garden_area_l3109_310979

theorem max_square_garden_area (perimeter : ℕ) (side_length : ℕ) : 
  perimeter = 160 →
  4 * side_length = perimeter →
  ∀ s : ℕ, 4 * s ≤ perimeter → s ^ 2 ≤ side_length ^ 2 :=
by
  sorry

#check max_square_garden_area

end max_square_garden_area_l3109_310979


namespace cos_2theta_plus_pi_l3109_310978

theorem cos_2theta_plus_pi (θ : Real) (h : Real.tan θ = 2) : 
  Real.cos (2 * θ + Real.pi) = 3 / 5 := by
  sorry

end cos_2theta_plus_pi_l3109_310978


namespace constant_sum_of_squares_l3109_310971

/-- Defines an ellipse C with equation x²/4 + y² = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Defines a point P on the major axis of C -/
def point_P (m : ℝ) : ℝ × ℝ := (m, 0)

/-- Defines the direction vector of line l -/
def direction_vector : ℝ × ℝ := (2, 1)

/-- Defines the line l passing through P with the given direction vector -/
def line_l (m t : ℝ) : ℝ × ℝ := (m + 2*t, t)

/-- Defines the intersection points of line l and ellipse C -/
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_l m t ∧ ellipse_C p.1 p.2}

/-- States that |PA|² + |PB|² is constant for all valid m -/
theorem constant_sum_of_squares (m : ℝ) (hm : -2 ≤ m ∧ m ≤ 2) :
  ∃ A B, A ∈ intersection_points m ∧ B ∈ intersection_points m ∧ A ≠ B ∧
    (A.1 - m)^2 + A.2^2 + (B.1 - m)^2 + B.2^2 = 5 :=
sorry

end constant_sum_of_squares_l3109_310971


namespace jenny_activities_alignment_l3109_310941

def dance_interval : ℕ := 6
def karate_interval : ℕ := 12
def swimming_interval : ℕ := 15
def painting_interval : ℕ := 20
def library_interval : ℕ := 18
def sick_days : ℕ := 7

def next_alignment_day : ℕ := 187

theorem jenny_activities_alignment :
  let intervals := [dance_interval, karate_interval, swimming_interval, painting_interval, library_interval]
  let lcm_intervals := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm dance_interval karate_interval) swimming_interval) painting_interval) library_interval
  next_alignment_day = lcm_intervals + sick_days := by
  sorry

end jenny_activities_alignment_l3109_310941


namespace triangle_yz_length_l3109_310944

/-- Given a triangle XYZ where cos(2X-Z) + sin(X+Z) = 2 and XY = 6, prove that YZ = 3 -/
theorem triangle_yz_length (X Y Z : ℝ) (h1 : 0 < X ∧ 0 < Y ∧ 0 < Z)
  (h2 : X + Y + Z = π) (h3 : Real.cos (2*X - Z) + Real.sin (X + Z) = 2) (h4 : 6 = 6) : 
  3 = 3 := by
  sorry

end triangle_yz_length_l3109_310944


namespace optimal_production_plan_l3109_310902

/-- Represents a production plan with quantities of products A and B. -/
structure ProductionPlan where
  productA : ℕ
  productB : ℕ

/-- Represents the available raw materials and profit data. -/
structure FactoryData where
  rawMaterialA : ℝ
  rawMaterialB : ℝ
  totalProducts : ℕ
  materialAForProductA : ℝ
  materialBForProductA : ℝ
  materialAForProductB : ℝ
  materialBForProductB : ℝ
  profitProductA : ℕ
  profitProductB : ℕ

/-- Checks if a production plan is valid given the factory data. -/
def isValidPlan (plan : ProductionPlan) (data : FactoryData) : Prop :=
  plan.productA + plan.productB = data.totalProducts ∧
  plan.productA * data.materialAForProductA + plan.productB * data.materialAForProductB ≤ data.rawMaterialA ∧
  plan.productA * data.materialBForProductA + plan.productB * data.materialBForProductB ≤ data.rawMaterialB

/-- Calculates the profit for a given production plan. -/
def calculateProfit (plan : ProductionPlan) (data : FactoryData) : ℕ :=
  plan.productA * data.profitProductA + plan.productB * data.profitProductB

/-- The main theorem to prove. -/
theorem optimal_production_plan (data : FactoryData)
  (h_data : data.rawMaterialA = 66 ∧ data.rawMaterialB = 66.4 ∧ data.totalProducts = 90 ∧
            data.materialAForProductA = 0.5 ∧ data.materialBForProductA = 0.8 ∧
            data.materialAForProductB = 1.2 ∧ data.materialBForProductB = 0.6 ∧
            data.profitProductA = 30 ∧ data.profitProductB = 20) :
  ∃ (optimalPlan : ProductionPlan),
    isValidPlan optimalPlan data ∧
    calculateProfit optimalPlan data = 2420 ∧
    ∀ (plan : ProductionPlan), isValidPlan plan data → calculateProfit plan data ≤ 2420 :=
  sorry

end optimal_production_plan_l3109_310902


namespace sugar_cups_in_lemonade_l3109_310923

theorem sugar_cups_in_lemonade (total_cups : ℕ) (sugar_ratio water_ratio : ℕ) : 
  total_cups = 84 → sugar_ratio = 1 → water_ratio = 2 → 
  (sugar_ratio * total_cups) / (sugar_ratio + water_ratio) = 28 := by
sorry

end sugar_cups_in_lemonade_l3109_310923


namespace polynomial_coefficients_l3109_310947

theorem polynomial_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 = a₀ + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3 + a₄*(1-x)^4 + a₅*(1-x)^5) →
  (a₃ = -10 ∧ a₁ + a₃ + a₅ = -16) := by
  sorry

end polynomial_coefficients_l3109_310947


namespace friends_basketball_score_l3109_310996

theorem friends_basketball_score 
  (total_points : ℕ) 
  (edward_points : ℕ) 
  (h1 : total_points = 13) 
  (h2 : edward_points = 7) : 
  total_points - edward_points = 6 := by
  sorry

end friends_basketball_score_l3109_310996


namespace rectangular_field_dimensions_l3109_310963

theorem rectangular_field_dimensions (m : ℝ) : 
  (2 * m + 9) * (m - 4) = 88 → m = 7.5 := by
  sorry

end rectangular_field_dimensions_l3109_310963


namespace function_inequality_l3109_310909

-- Define the function f
variable {f : ℝ → ℝ}

-- State the theorem
theorem function_inequality
  (h : ∀ x y : ℝ, f y - f x ≤ (y - x)^2)
  (n : ℕ)
  (hn : n > 0)
  (a b : ℝ) :
  |f b - f a| ≤ (1 / n : ℝ) * (b - a)^2 := by
  sorry

end function_inequality_l3109_310909


namespace odd_function_negative_domain_l3109_310954

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_positive : ∀ x > 0, f x = x * (1 + x)) :
  ∀ x < 0, f x = x * (1 - x) :=
by sorry

end odd_function_negative_domain_l3109_310954


namespace tin_addition_theorem_l3109_310957

/-- Proves that adding 1.5 kg of pure tin to a 12 kg alloy containing 45% copper 
    will result in a new alloy containing 40% copper. -/
theorem tin_addition_theorem (initial_mass : ℝ) (initial_copper_percentage : ℝ) 
    (final_copper_percentage : ℝ) (tin_added : ℝ) : 
    initial_mass = 12 →
    initial_copper_percentage = 0.45 →
    final_copper_percentage = 0.4 →
    tin_added = 1.5 →
    initial_mass * initial_copper_percentage = 
    final_copper_percentage * (initial_mass + tin_added) := by
  sorry

#check tin_addition_theorem

end tin_addition_theorem_l3109_310957


namespace girls_tried_out_l3109_310932

/-- The number of girls who tried out for the basketball team -/
def girls : ℕ := 39

/-- The number of boys who tried out for the basketball team -/
def boys : ℕ := 4

/-- The number of students who got called back -/
def called_back : ℕ := 26

/-- The number of students who didn't make the cut -/
def didnt_make_cut : ℕ := 17

/-- The total number of students who tried out -/
def total_students : ℕ := called_back + didnt_make_cut

theorem girls_tried_out : girls = total_students - boys := by
  sorry

end girls_tried_out_l3109_310932


namespace initial_roses_l3109_310989

theorem initial_roses (initial thrown_away added final : ℕ) 
  (h1 : thrown_away = 4)
  (h2 : added = 25)
  (h3 : final = 23)
  (h4 : initial - thrown_away + added = final) : initial = 2 :=
by sorry

end initial_roses_l3109_310989


namespace area_between_tangent_circles_l3109_310985

/-- The area of the region between two tangent circles -/
theorem area_between_tangent_circles (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : r₂ = 3 * r₁) :
  π * r₂^2 - π * r₁^2 = 32 * π * r₁^2 :=
sorry

end area_between_tangent_circles_l3109_310985


namespace vector_dot_product_equation_l3109_310901

-- Define the vectors a and b
def a : ℝ × ℝ := (2, -1)
def b (x : ℝ) : ℝ × ℝ := (3, x)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem vector_dot_product_equation (x : ℝ) :
  dot_product a (b x) = 3 → x = 3 := by
  sorry

end vector_dot_product_equation_l3109_310901


namespace max_attempts_l3109_310955

/-- The number of unique arrangements of a four-digit number containing one 2, one 9, and two 6s -/
def password_arrangements : ℕ := sorry

/-- The maximum number of attempts needed to find the correct password -/
theorem max_attempts : password_arrangements = 12 := by sorry

end max_attempts_l3109_310955


namespace peters_birdseed_calculation_l3109_310974

/-- The amount of birdseed needed for a week given the number of birds and their daily consumption --/
def birdseed_needed (parakeet_count : ℕ) (parrot_count : ℕ) (finch_count : ℕ) 
  (parakeet_consumption : ℕ) (parrot_consumption : ℕ) (days : ℕ) : ℕ :=
  let finch_consumption := parakeet_consumption / 2
  let daily_total := parakeet_count * parakeet_consumption + 
                     parrot_count * parrot_consumption + 
                     finch_count * finch_consumption
  daily_total * days

/-- Theorem stating that Peter needs to buy 266 grams of birdseed for a week --/
theorem peters_birdseed_calculation :
  birdseed_needed 3 2 4 2 14 7 = 266 := by
  sorry


end peters_birdseed_calculation_l3109_310974


namespace exponential_function_property_l3109_310907

theorem exponential_function_property (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → (fun x => a^x) (x + y) = (fun x => a^x) x * (fun x => a^x) y :=
by sorry

end exponential_function_property_l3109_310907


namespace arrangements_with_A_or_B_at_ends_eq_84_l3109_310924

/-- The number of ways to arrange n distinct objects --/
def permutations (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange 5 people in a row with at least one of A or B at the ends --/
def arrangements_with_A_or_B_at_ends : ℕ :=
  permutations 5 - (3 * 2 * permutations 3)

theorem arrangements_with_A_or_B_at_ends_eq_84 :
  arrangements_with_A_or_B_at_ends = 84 := by
  sorry

end arrangements_with_A_or_B_at_ends_eq_84_l3109_310924


namespace remainder_set_different_l3109_310914

theorem remainder_set_different (a b c : ℤ) 
  (ha : 0 < a ∧ a < c - 1) 
  (hb : 1 < b ∧ b < c) : 
  let r : ℤ → ℤ := λ k => (k * b) % c
  (∀ k, 0 ≤ k ∧ k ≤ a → 0 ≤ r k ∧ r k < c) →
  {k : ℤ | 0 ≤ k ∧ k ≤ a}.image r ≠ {k : ℤ | 0 ≤ k ∧ k ≤ a} := by
  sorry

end remainder_set_different_l3109_310914


namespace knitting_productivity_l3109_310976

theorem knitting_productivity (girl1_work_time girl1_break_time girl2_work_time girl2_break_time : ℕ) 
  (h1 : girl1_work_time = 5)
  (h2 : girl1_break_time = 1)
  (h3 : girl2_work_time = 7)
  (h4 : girl2_break_time = 1)
  : (girl1_work_time * (girl1_work_time + girl1_break_time)) / 
    (girl2_work_time * (girl2_work_time + girl2_break_time)) = 20 / 21 :=
by sorry

end knitting_productivity_l3109_310976


namespace bianca_next_day_miles_l3109_310929

/-- The number of miles Bianca ran on the first day -/
def first_day_miles : ℕ := 8

/-- The total number of miles Bianca ran over two days -/
def total_miles : ℕ := 12

/-- The number of miles Bianca ran on the next day -/
def next_day_miles : ℕ := total_miles - first_day_miles

theorem bianca_next_day_miles :
  next_day_miles = 4 :=
by sorry

end bianca_next_day_miles_l3109_310929


namespace parallelogram_area_l3109_310958

/-- The area of a parallelogram with vertices at (0, 0), (3, 0), (5, 12), and (8, 12) is 36 square units. -/
theorem parallelogram_area : 
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (3, 0)
  let v3 : ℝ × ℝ := (5, 12)
  let v4 : ℝ × ℝ := (8, 12)
  let base : ℝ := v2.1 - v1.1
  let height : ℝ := v3.2 - v1.2
  base * height = 36 := by
  sorry

end parallelogram_area_l3109_310958


namespace cracked_seashells_l3109_310987

/-- The number of cracked seashells given the conditions from the problem -/
theorem cracked_seashells (mary_shells keith_shells total_shells : ℕ) 
  (h1 : mary_shells = 2)
  (h2 : keith_shells = 5)
  (h3 : total_shells = 7) :
  mary_shells + keith_shells - total_shells = 0 := by
  sorry

end cracked_seashells_l3109_310987


namespace sum_of_square_areas_l3109_310908

/-- Given two squares ABCD and DEFG where CE = 14 and AG = 2, prove that the sum of their areas is 100 -/
theorem sum_of_square_areas (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 14 → a - b = 2 → a^2 + b^2 = 100 := by
  sorry

end sum_of_square_areas_l3109_310908


namespace intersection_point_x_coordinate_l3109_310951

/-- The x-coordinate of the intersection point of two linear functions -/
theorem intersection_point_x_coordinate (k b : ℝ) (h : k ≠ b) : 
  ∃ x : ℝ, k * x + b = b * x + k ∧ x = 1 := by
  sorry

end intersection_point_x_coordinate_l3109_310951


namespace brothers_ratio_l3109_310936

theorem brothers_ratio (aaron_brothers bennett_brothers : ℕ) 
  (h1 : aaron_brothers = 4) 
  (h2 : bennett_brothers = 6) : 
  (bennett_brothers : ℚ) / aaron_brothers = 3 / 2 := by
  sorry

end brothers_ratio_l3109_310936


namespace geometric_sequence_property_l3109_310903

/-- A geometric sequence with positive terms -/
structure PositiveGeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : ∀ n, a n > 0
  h_geometric : ∀ n, a (n + 1) = q * a n

/-- The property that 2a_1, (1/2)a_3, and a_2 form an arithmetic sequence -/
def ArithmeticProperty (s : PositiveGeometricSequence) : Prop :=
  2 * s.a 1 + s.a 2 = 2 * ((1/2) * s.a 3)

theorem geometric_sequence_property (s : PositiveGeometricSequence) 
  (h_arith : ArithmeticProperty s) : s.q = 2 := by
  sorry

end geometric_sequence_property_l3109_310903
