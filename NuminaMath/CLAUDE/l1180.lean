import Mathlib

namespace cnc_machine_profit_l1180_118035

/-- Represents the profit function for a CNC machine -/
def profit_function (x : ℕ+) : ℤ := -2 * x.val ^ 2 + 40 * x.val - 98

/-- Represents when the machine starts generating profit -/
def profit_start : ℕ+ := 3

/-- Represents the year of maximum average annual profit -/
def max_avg_profit_year : ℕ+ := 7

/-- Represents the year of maximum total profit -/
def max_total_profit_year : ℕ+ := 10

/-- Theorem stating the properties of the CNC machine profit -/
theorem cnc_machine_profit :
  (∀ x : ℕ+, profit_function x = -2 * x.val ^ 2 + 40 * x.val - 98) ∧
  (∀ x : ℕ+, x < profit_start → profit_function x ≤ 0) ∧
  (∀ x : ℕ+, x ≥ profit_start → profit_function x > 0) ∧
  (∀ x : ℕ+, x ≠ max_avg_profit_year → 
    (profit_function x : ℚ) / x.val ≤ (profit_function max_avg_profit_year : ℚ) / max_avg_profit_year.val) ∧
  (∀ x : ℕ+, profit_function x ≤ profit_function max_total_profit_year) :=
by sorry

end cnc_machine_profit_l1180_118035


namespace distance_between_trees_l1180_118093

/-- Given a yard of length 150 meters with 11 trees planted at equal distances,
    including one tree at each end, the distance between consecutive trees is 15 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 150 ∧ num_trees = 11 →
  (yard_length / (num_trees - 1 : ℝ)) = 15 := by
  sorry

end distance_between_trees_l1180_118093


namespace function_equation_solution_l1180_118057

/-- Given a function f: ℝ → ℝ satisfying f(x-f(y)) = 1 - x - y for all x, y ∈ ℝ,
    prove that f(x) = 1/2 - x for all x ∈ ℝ. -/
theorem function_equation_solution (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x - f y) = 1 - x - y) : 
    ∀ x : ℝ, f x = 1/2 - x := by
  sorry

end function_equation_solution_l1180_118057


namespace dans_remaining_money_l1180_118094

theorem dans_remaining_money (initial_amount : ℚ) (candy_price : ℚ) (gum_price : ℚ) :
  initial_amount = 3.75 →
  candy_price = 1.25 →
  gum_price = 0.80 →
  initial_amount - (candy_price + gum_price) = 1.70 := by
  sorry

end dans_remaining_money_l1180_118094


namespace owen_june_burger_expense_l1180_118017

/-- The amount Owen spent on burgers in June -/
def owen_burger_expense (burgers_per_day : ℕ) (burger_cost : ℕ) (days_in_june : ℕ) : ℕ :=
  burgers_per_day * days_in_june * burger_cost

/-- Theorem stating that Owen's burger expense in June is 720 dollars -/
theorem owen_june_burger_expense :
  owen_burger_expense 2 12 30 = 720 :=
by sorry

end owen_june_burger_expense_l1180_118017


namespace conic_section_properties_l1180_118012

-- Define the equation C
def C (x y k : ℝ) : Prop := x^2 / (16 + k) - y^2 / (9 - k) = 1

-- Theorem statement
theorem conic_section_properties :
  -- The equation cannot represent a circle
  (∀ k : ℝ, ¬∃ r : ℝ, ∀ x y : ℝ, C x y k ↔ x^2 + y^2 = r^2) ∧
  -- When k > 9, the equation represents an ellipse with foci on the x-axis
  (∀ k : ℝ, k > 9 → ∃ a b : ℝ, a > b ∧ b > 0 ∧ ∀ x y : ℝ, C x y k ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  -- When -16 < k < 9, the equation represents a hyperbola with foci on the x-axis
  (∀ k : ℝ, -16 < k ∧ k < 9 → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, C x y k ↔ x^2 / a^2 - y^2 / b^2 = 1) ∧
  -- When the equation represents an ellipse or a hyperbola, the focal distance is always 10
  (∀ k : ℝ, (k > 9 ∨ (-16 < k ∧ k < 9)) → 
    ∃ c : ℝ, c = 5 ∧ 
    (∀ x y : ℝ, C x y k → 
      (k > 9 → ∃ a b : ℝ, a > b ∧ b > 0 ∧ c^2 = a^2 - b^2) ∧
      (-16 < k ∧ k < 9 → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ c^2 = a^2 + b^2))) :=
sorry

end conic_section_properties_l1180_118012


namespace angle_measure_when_complement_is_half_supplement_l1180_118064

theorem angle_measure_when_complement_is_half_supplement :
  ∀ x : ℝ,
  (x > 0) →
  (x ≤ 180) →
  (90 - x = (180 - x) / 2) →
  x = 90 := by
sorry

end angle_measure_when_complement_is_half_supplement_l1180_118064


namespace search_rescue_selection_methods_l1180_118003

def chinese_ships : ℕ := 4
def chinese_planes : ℕ := 3
def foreign_ships : ℕ := 5
def foreign_planes : ℕ := 2

def units_per_side : ℕ := 2
def total_units : ℕ := 4
def required_planes : ℕ := 1

theorem search_rescue_selection_methods :
  (chinese_ships.choose units_per_side * chinese_planes.choose required_planes * foreign_ships.choose units_per_side) +
  (chinese_ships.choose units_per_side * foreign_ships.choose (units_per_side - 1) * foreign_planes.choose required_planes) = 180 := by
  sorry

end search_rescue_selection_methods_l1180_118003


namespace pure_imaginary_product_l1180_118034

theorem pure_imaginary_product (a : ℝ) : 
  (∃ b : ℝ, (a + Complex.I) * (2 - Complex.I) = Complex.I * b) → a = -1/2 := by
  sorry

end pure_imaginary_product_l1180_118034


namespace parallel_vectors_y_value_l1180_118074

def vector_a : ℝ × ℝ := (2, 3)
def vector_b (y : ℝ) : ℝ × ℝ := (4, -1 + y)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_y_value :
  parallel vector_a (vector_b y) → y = 7 := by
  sorry

end parallel_vectors_y_value_l1180_118074


namespace fruit_box_ratio_l1180_118054

/-- Proves that the ratio of peaches to oranges is 1:2 given the conditions of the fruit box problem -/
theorem fruit_box_ratio : 
  ∀ (total_fruits oranges apples peaches : ℕ),
  total_fruits = 56 →
  oranges = total_fruits / 4 →
  apples = 35 →
  apples = 5 * peaches →
  (peaches : ℚ) / oranges = 1 / 2 := by
sorry

end fruit_box_ratio_l1180_118054


namespace boxes_per_crate_l1180_118042

theorem boxes_per_crate 
  (total_crates : ℕ) 
  (original_machines_per_box : ℕ) 
  (machines_removed_per_box : ℕ) 
  (total_machines_removed : ℕ) 
  (h1 : total_crates = 10)
  (h2 : original_machines_per_box = 4)
  (h3 : machines_removed_per_box = 1)
  (h4 : total_machines_removed = 60) :
  total_machines_removed / total_crates = 6 := by
sorry

end boxes_per_crate_l1180_118042


namespace complex_fraction_equality_l1180_118083

theorem complex_fraction_equality : (2 * Complex.I) / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end complex_fraction_equality_l1180_118083


namespace student_score_average_l1180_118023

/-- Given a student's scores in mathematics, physics, and chemistry, prove that the average of mathematics and chemistry scores is 26. -/
theorem student_score_average (math physics chem : ℕ) : 
  math + physics = 32 →
  chem = physics + 20 →
  (math + chem) / 2 = 26 := by
  sorry

end student_score_average_l1180_118023


namespace jane_farm_eggs_l1180_118058

/-- Calculates the number of eggs laid per chicken per week -/
def eggs_per_chicken_per_week (num_chickens : ℕ) (price_per_dozen : ℚ) (total_revenue : ℚ) (num_weeks : ℕ) : ℚ :=
  (total_revenue / price_per_dozen * 12) / (num_chickens * num_weeks)

theorem jane_farm_eggs : 
  let num_chickens : ℕ := 10
  let price_per_dozen : ℚ := 2
  let total_revenue : ℚ := 20
  let num_weeks : ℕ := 2
  eggs_per_chicken_per_week num_chickens price_per_dozen total_revenue num_weeks = 6 := by
  sorry

#eval eggs_per_chicken_per_week 10 2 20 2

end jane_farm_eggs_l1180_118058


namespace george_score_l1180_118078

theorem george_score (n : ℕ) (avg_without : ℚ) (avg_with : ℚ) : 
  n = 20 → 
  avg_without = 75 → 
  avg_with = 76 → 
  (n - 1) * avg_without + 95 = n * avg_with :=
by
  sorry

end george_score_l1180_118078


namespace absolute_value_equation_implies_power_l1180_118016

theorem absolute_value_equation_implies_power (x : ℝ) :
  |x| = 3 * x + 1 → (4 * x + 2)^2005 = 1 := by
  sorry

end absolute_value_equation_implies_power_l1180_118016


namespace function_forms_correctness_l1180_118043

-- Define a linear function
def linear_function (a b x : ℝ) : ℝ := a * x + b

-- Define a special case of linear function
def linear_function_special (a x : ℝ) : ℝ := a * x

-- Define a quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define special cases of quadratic function
def quadratic_function_special1 (a c x : ℝ) : ℝ := a * x^2 + c
def quadratic_function_special2 (a x : ℝ) : ℝ := a * x^2

-- Theorem stating the correctness of these function definitions
theorem function_forms_correctness (a b c x : ℝ) (h : a ≠ 0) :
  (∃ y, y = linear_function a b x) ∧
  (∃ y, y = linear_function_special a x) ∧
  (∃ y, y = quadratic_function a b c x) ∧
  (∃ y, y = quadratic_function_special1 a c x) ∧
  (∃ y, y = quadratic_function_special2 a x) :=
sorry

end function_forms_correctness_l1180_118043


namespace intersection_point_x_coordinate_l1180_118066

noncomputable def f (x : ℝ) := Real.exp (x - 1)

theorem intersection_point_x_coordinate 
  (A B C E : ℝ × ℝ) 
  (hA : A = (1, f 1)) 
  (hB : B = (Real.exp 3, f (Real.exp 3))) 
  (hC : C.2 = (2/3) * A.2 + (1/3) * B.2) 
  (hE : E.2 = f E.1 ∧ E.2 = C.2) :
  E.1 = Real.log ((2/3) + (1/3) * Real.exp (Real.exp 3 - 1)) + 1 := by
  sorry

end intersection_point_x_coordinate_l1180_118066


namespace longest_tape_measure_l1180_118048

theorem longest_tape_measure (a b c : ℕ) 
  (ha : a = 600) (hb : b = 500) (hc : c = 1200) : 
  Nat.gcd a (Nat.gcd b c) = 100 := by
  sorry

end longest_tape_measure_l1180_118048


namespace puzzle_solution_l1180_118027

theorem puzzle_solution (A B C : ℤ) 
  (eq1 : A + C = 10)
  (eq2 : A + B + 1 = C + 10)
  (eq3 : A + 1 = B) :
  A = 6 ∧ B = 7 ∧ C = 4 := by
  sorry

end puzzle_solution_l1180_118027


namespace two_pencils_one_pen_cost_l1180_118067

-- Define the cost of a pencil and a pen
variable (pencil_cost pen_cost : ℚ)

-- Define the given conditions
axiom condition1 : 3 * pencil_cost + pen_cost = 3
axiom condition2 : 3 * pencil_cost + 4 * pen_cost = (15/2)

-- State the theorem to be proved
theorem two_pencils_one_pen_cost : 
  2 * pencil_cost + pen_cost = (5/2) := by
sorry

end two_pencils_one_pen_cost_l1180_118067


namespace ratio_equality_l1180_118079

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 49)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 64)
  (dot_product : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
  sorry

end ratio_equality_l1180_118079


namespace triangle_theorem_l1180_118031

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : a^2 = b^2 + c^2 - 2*b*c*Real.cos A

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) : 
  (2 * t.b = t.c + 2 * t.a * Real.cos t.C) → 
  (t.A = π / 3 ∧ 
   (1/2 * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 / 3 ∧ t.a = 3 → 
    t.a + t.b + t.c = 8)) := by
  sorry

end triangle_theorem_l1180_118031


namespace prob_shortest_diagonal_15_sided_l1180_118098

/-- The number of sides in the regular polygon -/
def n : ℕ := 15

/-- The total number of diagonals in a regular n-sided polygon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in a regular n-sided polygon -/
def shortest_diagonals (n : ℕ) : ℕ := n

/-- The probability of selecting a shortest diagonal in a regular n-sided polygon -/
def prob_shortest_diagonal (n : ℕ) : ℚ :=
  shortest_diagonals n / total_diagonals n

theorem prob_shortest_diagonal_15_sided :
  prob_shortest_diagonal n = 1 / 6 := by
  sorry

end prob_shortest_diagonal_15_sided_l1180_118098


namespace sequence_term_equality_l1180_118086

theorem sequence_term_equality (n : ℕ) : 
  2 * Real.log 5 + Real.log 3 = Real.log (4 * 19 - 1) :=
by sorry

end sequence_term_equality_l1180_118086


namespace alice_bob_difference_zero_l1180_118032

/-- Represents the vacation expenses problem -/
def vacation_expenses (alice_paid bob_paid charlie_paid : ℝ) (a b : ℝ) : Prop :=
  let total_paid := alice_paid + bob_paid + charlie_paid
  let equal_share := total_paid / 3
  -- Alice's balance after giving 'a' to Charlie
  (alice_paid - a = equal_share) ∧
  -- Bob's balance after giving 'b' to Charlie
  (bob_paid - b = equal_share) ∧
  -- Charlie's balance after receiving 'a' from Alice and 'b' from Bob
  (charlie_paid + a + b = equal_share)

/-- Theorem stating that the difference between what Alice and Bob give to Charlie is zero -/
theorem alice_bob_difference_zero 
  (alice_paid bob_paid charlie_paid : ℝ) 
  (h_alice : alice_paid = 180) 
  (h_bob : bob_paid = 240) 
  (h_charlie : charlie_paid = 120) :
  ∃ a b : ℝ, vacation_expenses alice_paid bob_paid charlie_paid a b ∧ a - b = 0 :=
sorry

end alice_bob_difference_zero_l1180_118032


namespace barrel_division_l1180_118090

theorem barrel_division (length width height : ℝ) (volume_small_barrel : ℝ) : 
  length = 6.4 ∧ width = 9 ∧ height = 5.2 ∧ volume_small_barrel = 1 →
  ⌈length * width * height / volume_small_barrel⌉ = 300 := by
  sorry

end barrel_division_l1180_118090


namespace max_ratio_on_unit_circle_l1180_118099

theorem max_ratio_on_unit_circle :
  let a : ℂ := Real.sqrt 17
  let b : ℂ := Complex.I * Real.sqrt 19
  (∃ (k : ℝ), k = 4/3 ∧
    ∀ (z : ℂ), Complex.abs z = 1 →
      Complex.abs (a - z) / Complex.abs (b - z) ≤ k) ∧
    ∀ (k' : ℝ), (∀ (z : ℂ), Complex.abs z = 1 →
      Complex.abs (a - z) / Complex.abs (b - z) ≤ k') →
      k' ≥ 4/3 :=
by sorry

end max_ratio_on_unit_circle_l1180_118099


namespace remaining_money_l1180_118028

def octal_to_decimal (n : ℕ) : ℕ := sorry

def john_savings : ℕ := octal_to_decimal 5372

def ticket_cost : ℕ := 1200

theorem remaining_money :
  john_savings - ticket_cost = 1610 := by sorry

end remaining_money_l1180_118028


namespace valid_pairs_l1180_118095

def is_valid_pair (a b : ℕ) : Prop :=
  (a^2 + b) % (b^2 - a) = 0 ∧ (b^2 + a) % (a^2 - b) = 0

theorem valid_pairs :
  ∀ a b : ℕ, is_valid_pair a b ↔ 
    ((a = 1 ∧ b = 2) ∨ 
     (a = 2 ∧ b = 1) ∨ 
     (a = 2 ∧ b = 2) ∨ 
     (a = 2 ∧ b = 3) ∨ 
     (a = 3 ∧ b = 2) ∨ 
     (a = 3 ∧ b = 3)) :=
by sorry

end valid_pairs_l1180_118095


namespace expression_value_l1180_118062

theorem expression_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (sum_zero : x + y + z = 0) (sum_prod_nonzero : x*y + x*z + y*z ≠ 0) :
  (x^5 + y^5 + z^5) / (x*y*z * (x*y + x*z + y*z)) = -5 := by
  sorry

end expression_value_l1180_118062


namespace product_sum_equation_l1180_118088

theorem product_sum_equation : (3 * 4 * 5) * (1/3 + 1/4 + 1/5 + 1) = 107 := by
  sorry

end product_sum_equation_l1180_118088


namespace parallel_plane_sufficient_not_necessary_l1180_118037

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Main theorem
theorem parallel_plane_sufficient_not_necessary
  (m n : Line) (α β : Plane)
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_m_subset_α : subset m α)
  (h_n_subset_α : subset n α) :
  (∀ l : Line, subset l α → parallel_line_plane l β) ∧
  ∃ m n : Line, subset m α ∧ subset n α ∧ 
    parallel_line_plane m β ∧ parallel_line_plane n β ∧
    ¬ parallel_plane_plane α β :=
sorry

end parallel_plane_sufficient_not_necessary_l1180_118037


namespace existence_of_graph_with_chromatic_number_without_clique_l1180_118096

/-- A graph is a structure with vertices and an edge relation -/
structure Graph (V : Type) where
  edge : V → V → Prop

/-- The chromatic number of a graph is the minimum number of colors needed to color its vertices 
    such that no two adjacent vertices have the same color -/
def chromaticNumber (G : Graph V) : ℕ := sorry

/-- An n-clique in a graph is a complete subgraph with n vertices -/
def hasClique (G : Graph V) (n : ℕ) : Prop := sorry

theorem existence_of_graph_with_chromatic_number_without_clique :
  ∀ n : ℕ, n > 3 → ∃ (V : Type) (G : Graph V), chromaticNumber G = n ∧ ¬hasClique G n := by
  sorry

end existence_of_graph_with_chromatic_number_without_clique_l1180_118096


namespace num_teams_is_nine_l1180_118060

/-- The number of games in a round-robin tournament with n teams -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of games played in the tournament -/
def total_games : ℕ := 36

/-- Theorem: The number of teams in the tournament is 9 -/
theorem num_teams_is_nine : ∃ (n : ℕ), n > 0 ∧ num_games n = total_games ∧ n = 9 := by
  sorry

end num_teams_is_nine_l1180_118060


namespace textbook_completion_date_l1180_118007

/-- Represents the number of problems solved on a given day -/
def problems_solved (day : ℕ) : ℕ → ℕ
| 0 => day + 1  -- September 6
| n + 1 => day - n  -- Subsequent days

/-- Calculates the total problems solved up to a given day -/
def total_solved (day : ℕ) : ℕ :=
  (List.range (day + 1)).map (problems_solved day) |>.sum

theorem textbook_completion_date 
  (total_problems : ℕ) 
  (problems_left_day3 : ℕ) 
  (h1 : total_problems = 91)
  (h2 : problems_left_day3 = 46)
  (h3 : total_solved 2 = total_problems - problems_left_day3) :
  total_solved 6 = total_problems := by
  sorry

#eval total_solved 6  -- Should output 91

end textbook_completion_date_l1180_118007


namespace investment_rate_calculation_investment_rate_proof_l1180_118022

/-- Calculates the required interest rate for the remaining investment --/
theorem investment_rate_calculation 
  (total_investment : ℝ) 
  (first_investment : ℝ) 
  (second_investment : ℝ) 
  (first_rate : ℝ) 
  (second_rate : ℝ) 
  (desired_income : ℝ) : ℝ :=
  let remaining_investment := total_investment - first_investment - second_investment
  let first_income := first_investment * first_rate / 100
  let second_income := second_investment * second_rate / 100
  let remaining_income := desired_income - first_income - second_income
  let required_rate := remaining_income / remaining_investment * 100
  required_rate

/-- Proves that the required interest rate is approximately 7.05% --/
theorem investment_rate_proof 
  (h1 : total_investment = 15000)
  (h2 : first_investment = 6000)
  (h3 : second_investment = 4500)
  (h4 : first_rate = 3)
  (h5 : second_rate = 4.5)
  (h6 : desired_income = 700) :
  ∃ ε > 0, |investment_rate_calculation total_investment first_investment second_investment first_rate second_rate desired_income - 7.05| < ε :=
by
  sorry

end investment_rate_calculation_investment_rate_proof_l1180_118022


namespace parallel_lines_equal_angles_with_plane_l1180_118018

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the relation for a line forming an angle with a plane
variable (angle_with_plane : Line → Plane → ℝ)

-- State the theorem
theorem parallel_lines_equal_angles_with_plane
  (m n : Line) (α : Plane) :
  (parallel m n → angle_with_plane m α = angle_with_plane n α) ∧
  ¬(angle_with_plane m α = angle_with_plane n α → parallel m n) :=
sorry

end parallel_lines_equal_angles_with_plane_l1180_118018


namespace average_reading_time_l1180_118000

/-- Given that Emery reads 5 times faster than Serena and takes 20 days to read a book,
    prove that the average number of days for both to read the book is 60 days. -/
theorem average_reading_time (emery_days : ℕ) (emery_speed : ℕ) :
  emery_days = 20 →
  emery_speed = 5 →
  (emery_days + emery_speed * emery_days) / 2 = 60 := by
  sorry

end average_reading_time_l1180_118000


namespace chord_length_l1180_118092

-- Define the line l
def line_l (x y : ℝ) : Prop := y = -3/4 * x + 5/4

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Theorem statement
theorem chord_length :
  ∃ (chord_length : ℝ),
    (∀ x y : ℝ, line_l x y → circle_O x y → 
      chord_length = 2 * Real.sqrt 3) :=
by sorry

end chord_length_l1180_118092


namespace ice_cream_scoop_cost_l1180_118052

/-- The cost of a "Build Your Own Hot Brownie" dessert --/
structure BrownieDessert where
  brownieCost : ℝ
  syrupCost : ℝ
  nutsCost : ℝ
  iceCreamScoops : ℕ
  totalCost : ℝ

/-- The specific dessert order made by Juanita --/
def juanitaOrder : BrownieDessert where
  brownieCost := 2.50
  syrupCost := 0.50
  nutsCost := 1.50
  iceCreamScoops := 2
  totalCost := 7.00

/-- Theorem stating that each scoop of ice cream costs $1.00 --/
theorem ice_cream_scoop_cost (order : BrownieDessert) : 
  order.brownieCost = 2.50 →
  order.syrupCost = 0.50 →
  order.nutsCost = 1.50 →
  order.iceCreamScoops = 2 →
  order.totalCost = 7.00 →
  (order.totalCost - (order.brownieCost + 2 * order.syrupCost + order.nutsCost)) / order.iceCreamScoops = 1.00 := by
  sorry

end ice_cream_scoop_cost_l1180_118052


namespace tan_22_5_identity_l1180_118097

theorem tan_22_5_identity : 
  (Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2) = 1/2 := by sorry

end tan_22_5_identity_l1180_118097


namespace negative_fraction_comparison_l1180_118070

theorem negative_fraction_comparison : -5/4 < -4/5 := by
  sorry

end negative_fraction_comparison_l1180_118070


namespace isosceles_triangle_lateral_side_length_l1180_118020

/-- Given an isosceles triangle with vertex angle α and the sum of two different heights l,
    the length of a lateral side is l * tan(α/2) / (1 + 2 * sin(α/2)). -/
theorem isosceles_triangle_lateral_side_length
  (α l : ℝ) (h_α : 0 < α ∧ α < π) (h_l : l > 0) :
  ∃ (side_length : ℝ),
    side_length = l * Real.tan (α / 2) / (1 + 2 * Real.sin (α / 2)) ∧
    ∃ (height1 height2 : ℝ),
      height1 + height2 = l ∧
      height1 ≠ height2 ∧
      ∃ (base : ℝ),
        height1 = side_length * Real.cos (α / 2) ∧
        height2 = base / 2 * Real.tan (α / 2) :=
by sorry

end isosceles_triangle_lateral_side_length_l1180_118020


namespace find_divisor_l1180_118008

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 15968 →
  quotient = 89 →
  remainder = 37 →
  dividend = divisor * quotient + remainder →
  divisor = 179 := by
sorry

end find_divisor_l1180_118008


namespace gas_purchase_l1180_118089

theorem gas_purchase (price_nc : ℝ) (amount : ℝ) : 
  price_nc > 0 →
  amount > 0 →
  price_nc * amount + (price_nc + 1) * amount = 50 →
  price_nc = 2 →
  amount = 10 := by
sorry

end gas_purchase_l1180_118089


namespace binomial_expansion_sum_l1180_118030

theorem binomial_expansion_sum (x : ℝ) :
  ∃ (a a₁ a₂ a₃ a₄ a₅ : ℝ),
    (2*x - 1)^5 = a*x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅ ∧
    a₂ + a₃ = 40 := by
  sorry

end binomial_expansion_sum_l1180_118030


namespace salary_after_raise_l1180_118011

/-- 
Given an original salary and a percentage increase, 
calculate the new salary after a raise.
-/
theorem salary_after_raise 
  (original_salary : ℝ) 
  (percentage_increase : ℝ) 
  (new_salary : ℝ) : 
  original_salary = 55 ∧ 
  percentage_increase = 9.090909090909092 ∧
  new_salary = original_salary * (1 + percentage_increase / 100) →
  new_salary = 60 :=
by sorry

end salary_after_raise_l1180_118011


namespace largest_invertible_interval_containing_two_l1180_118056

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 8

-- State the theorem
theorem largest_invertible_interval_containing_two :
  ∃ (a : ℝ), a ≤ 2 ∧ 
  (∀ (x y : ℝ), a ≤ x ∧ x < y → g x < g y) ∧
  (∀ (b : ℝ), b < a → ¬(∀ (x y : ℝ), b ≤ x ∧ x < y → g x < g y)) :=
by sorry

end largest_invertible_interval_containing_two_l1180_118056


namespace express_delivery_growth_rate_l1180_118009

theorem express_delivery_growth_rate 
  (initial_revenue : ℝ)
  (final_revenue : ℝ)
  (years : ℕ)
  (h1 : initial_revenue = 400)
  (h2 : final_revenue = 576)
  (h3 : years = 2) :
  ∃ (growth_rate : ℝ), 
    growth_rate = 0.2 ∧ 
    initial_revenue * (1 + growth_rate) ^ years = final_revenue :=
sorry

end express_delivery_growth_rate_l1180_118009


namespace video_recorder_wholesale_cost_l1180_118013

theorem video_recorder_wholesale_cost :
  ∀ (wholesale_cost : ℝ),
  (∃ (retail_price employee_price : ℝ),
    retail_price = 1.20 * wholesale_cost ∧
    employee_price = 0.85 * retail_price ∧
    employee_price = 204) →
  wholesale_cost = 200 :=
by sorry

end video_recorder_wholesale_cost_l1180_118013


namespace cone_volume_from_inscribed_cylinder_and_frustum_l1180_118072

/-- Given a cone with an inscribed cylinder and a truncated cone (frustum), 
    this theorem proves the volume of the original cone. -/
theorem cone_volume_from_inscribed_cylinder_and_frustum 
  (V_cylinder : ℝ) 
  (V_frustum : ℝ) 
  (h_cylinder : V_cylinder = 21) 
  (h_frustum : V_frustum = 91) : 
  ∃ (V_cone : ℝ), V_cone = 94.5 ∧ 
  ∃ (R r H h : ℝ), 
    R > 0 ∧ r > 0 ∧ H > 0 ∧ h > 0 ∧
    V_cylinder = π * r^2 * h ∧
    V_frustum = (1/3) * π * (R^2 + R*r + r^2) * (H - h) ∧
    R / r = 3 ∧
    h / H = 1/3 ∧
    V_cone = (1/3) * π * R^2 * H := by
  sorry

end cone_volume_from_inscribed_cylinder_and_frustum_l1180_118072


namespace simultaneous_equations_solution_l1180_118004

theorem simultaneous_equations_solution :
  ∀ x y : ℝ, 
    (2 * x - 3 * y = 0.4 * (x + y)) →
    (5 * y = 1.2 * x) →
    (x = 0 ∧ y = 0) :=
by
  sorry

end simultaneous_equations_solution_l1180_118004


namespace sqrt_16_l1180_118075

theorem sqrt_16 : {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end sqrt_16_l1180_118075


namespace correct_arrangement_count_l1180_118036

/-- The number of ways to arrange 3 girls and 5 boys in a row -/
def arrangement_count (n_girls : ℕ) (n_boys : ℕ) : ℕ × ℕ × ℕ :=
  let total := n_girls + n_boys
  let adjacent := (Nat.factorial n_girls) * (Nat.factorial (total - n_girls + 1))
  let not_adjacent := (Nat.factorial n_boys) * (Nat.choose (n_boys + 1) n_girls) * (Nat.factorial n_girls)
  let boys_fixed := Nat.choose total n_girls
  (adjacent, not_adjacent, boys_fixed)

/-- Theorem stating the correct number of arrangements for 3 girls and 5 boys -/
theorem correct_arrangement_count :
  arrangement_count 3 5 = (4320, 14400, 336) := by
  sorry

end correct_arrangement_count_l1180_118036


namespace second_coaster_speed_l1180_118087

/-- The speed of the second rollercoaster given the speeds of the other coasters and the average speed -/
theorem second_coaster_speed 
  (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h1 : x₁ = 50)
  (h3 : x₃ = 73)
  (h4 : x₄ = 70)
  (h5 : x₅ = 40)
  (avg : (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 59) :
  x₂ = 62 := by
  sorry

end second_coaster_speed_l1180_118087


namespace candy_distribution_count_l1180_118033

/-- The number of ways to distribute n distinct items among k bags, where each bag must receive at least one item. -/
def distribute (n k : ℕ) : ℕ := k^n - k * ((k-1)^n - (k-1))

/-- The number of ways to distribute 9 distinct pieces of candy among 3 bags, where each bag must receive at least one piece of candy. -/
def candy_distribution : ℕ := distribute 9 3

theorem candy_distribution_count : candy_distribution = 18921 := by
  sorry

end candy_distribution_count_l1180_118033


namespace different_ball_counts_l1180_118019

/-- Represents a box in the game -/
structure Box :=
  (id : Nat)

/-- Represents a pair of boxes -/
structure BoxPair :=
  (box1 : Box)
  (box2 : Box)

/-- The game state -/
structure GameState :=
  (boxes : Finset Box)
  (pairs : Finset BoxPair)
  (ballCount : Box → Nat)

/-- The theorem statement -/
theorem different_ball_counts (n : Nat) (h : n = 2018) :
  ∃ (finalState : GameState),
    finalState.boxes.card = n ∧
    finalState.pairs.card = 2 * n - 2 ∧
    ∀ (b1 b2 : Box), b1 ∈ finalState.boxes → b2 ∈ finalState.boxes → b1 ≠ b2 →
      finalState.ballCount b1 ≠ finalState.ballCount b2 :=
by sorry

end different_ball_counts_l1180_118019


namespace power_division_equality_l1180_118021

theorem power_division_equality : (3^2)^4 / 3^2 = 729 := by
  sorry

end power_division_equality_l1180_118021


namespace largest_c_value_l1180_118006

theorem largest_c_value : ∃ (c_max : ℚ), c_max = 4 ∧ 
  (∀ c : ℚ, (3 * c + 4) * (c - 2) = 9 * c → c ≤ c_max) ∧
  ((3 * c_max + 4) * (c_max - 2) = 9 * c_max) := by
  sorry

end largest_c_value_l1180_118006


namespace impossibility_of_triangular_section_l1180_118045

theorem impossibility_of_triangular_section (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = 7^2 →
  b^2 + c^2 = 8^2 →
  c^2 + a^2 = 11^2 →
  False :=
by sorry

end impossibility_of_triangular_section_l1180_118045


namespace square_area_to_side_ratio_l1180_118076

theorem square_area_to_side_ratio :
  ∀ (s1 s2 : ℝ), s1 > 0 → s2 > 0 →
  (s1^2 / s2^2 = 243 / 75) →
  ∃ (a b c : ℕ), 
    (s1 / s2 = a * Real.sqrt b / c) ∧
    (a = 9 ∧ b = 1 ∧ c = 5) ∧
    (a + b + c = 15) := by
  sorry

end square_area_to_side_ratio_l1180_118076


namespace pencils_per_box_l1180_118059

theorem pencils_per_box (total_pencils : ℕ) (num_boxes : ℕ) 
  (h1 : total_pencils = 648) 
  (h2 : num_boxes = 162) 
  (h3 : total_pencils % num_boxes = 0) : 
  total_pencils / num_boxes = 4 := by
sorry

end pencils_per_box_l1180_118059


namespace no_four_digit_number_divisible_by_94_sum_l1180_118055

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_two_digits (n : ℕ) : ℕ := n / 100

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem no_four_digit_number_divisible_by_94_sum :
  ¬ ∃ (n : ℕ), is_four_digit n ∧ 
    n % (first_two_digits n + last_two_digits n) = 0 ∧
    first_two_digits n + last_two_digits n = 94 := by
  sorry

end no_four_digit_number_divisible_by_94_sum_l1180_118055


namespace zeros_before_nonzero_digit_l1180_118029

theorem zeros_before_nonzero_digit (n : ℕ) (m : ℕ) : 
  (Nat.log 10 (2^n * 5^m)).pred = n.max m := by sorry

end zeros_before_nonzero_digit_l1180_118029


namespace median_triangle_theorem_l1180_118041

/-- Given a triangle ABC with area 1 and medians s_a, s_b, s_c, there exists a triangle
    with sides s_a, s_b, s_c, and its area is 4/3 times the area of triangle ABC. -/
theorem median_triangle_theorem (A B C : ℝ × ℝ) (s_a s_b s_c : ℝ) :
  let triangle_area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  let median_a := ((B.1 + C.1) / 2 - A.1, (B.2 + C.2) / 2 - A.2)
  let median_b := ((A.1 + C.1) / 2 - B.1, (A.2 + C.2) / 2 - B.2)
  let median_c := ((A.1 + B.1) / 2 - C.1, (A.2 + B.2) / 2 - C.2)
  triangle_area = 1 ∧
  s_a = Real.sqrt (median_a.1^2 + median_a.2^2) ∧
  s_b = Real.sqrt (median_b.1^2 + median_b.2^2) ∧
  s_c = Real.sqrt (median_c.1^2 + median_c.2^2) →
  ∃ (D E F : ℝ × ℝ),
    let new_triangle_area := abs ((D.1 * (E.2 - F.2) + E.1 * (F.2 - D.2) + F.1 * (D.2 - E.2)) / 2)
    Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) = s_a ∧
    Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2) = s_b ∧
    Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2) = s_c ∧
    new_triangle_area = 4/3 * triangle_area := by
  sorry


end median_triangle_theorem_l1180_118041


namespace work_rate_proof_l1180_118010

/-- Given work rates for individuals and pairs, prove the work rate for a specific pair -/
theorem work_rate_proof 
  (c_rate : ℚ)
  (bc_rate : ℚ)
  (ca_rate : ℚ)
  (h1 : c_rate = 1 / 24)
  (h2 : bc_rate = 1 / 3)
  (h3 : ca_rate = 1 / 4) :
  ∃ (ab_rate : ℚ), ab_rate = 1 / 2 := by
  sorry


end work_rate_proof_l1180_118010


namespace ellens_snack_calories_l1180_118024

/-- Calculates the calories of an afternoon snack given the total daily allowance and the calories consumed in other meals. -/
def afternoon_snack_calories (daily_allowance breakfast lunch dinner : ℕ) : ℕ :=
  daily_allowance - breakfast - lunch - dinner

/-- Proves that Ellen's afternoon snack was 130 calories given her daily allowance and other meal calorie counts. -/
theorem ellens_snack_calories :
  afternoon_snack_calories 2200 353 885 832 = 130 := by
  sorry

end ellens_snack_calories_l1180_118024


namespace arithmetic_sequence_sum_l1180_118081

theorem arithmetic_sequence_sum : 
  ∀ (a₁ aₙ d n : ℕ) (S : ℕ),
    a₁ = 1 →                   -- First term is 1
    aₙ = 25 →                  -- Last term is 25
    d = 2 →                    -- Common difference is 2
    aₙ = a₁ + (n - 1) * d →    -- Formula for the nth term of an arithmetic sequence
    S = n * (a₁ + aₙ) / 2 →    -- Formula for the sum of an arithmetic sequence
    S = 169 :=                 -- The sum is 169
by sorry

end arithmetic_sequence_sum_l1180_118081


namespace pizza_fraction_eaten_l1180_118044

/-- Calculates the fraction of pizza eaten given the calorie information and consumption --/
theorem pizza_fraction_eaten 
  (lettuce_cal : ℕ) 
  (dressing_cal : ℕ) 
  (crust_cal : ℕ) 
  (cheese_cal : ℕ) 
  (total_consumed : ℕ) 
  (h1 : lettuce_cal = 50)
  (h2 : dressing_cal = 210)
  (h3 : crust_cal = 600)
  (h4 : cheese_cal = 400)
  (h5 : total_consumed = 330) :
  (total_consumed - (lettuce_cal + 2 * lettuce_cal + dressing_cal) / 4) / 
  (crust_cal + crust_cal / 3 + cheese_cal) = 1 / 5 := by
  sorry

#check pizza_fraction_eaten

end pizza_fraction_eaten_l1180_118044


namespace geometric_sequence_problem_l1180_118015

-- Define a geometric sequence
def is_geometric_sequence (x y z : ℝ) : Prop :=
  ∃ q : ℝ, y = x * q ∧ z = y * q

-- Define the problem statement
theorem geometric_sequence_problem (a b c d : ℝ) 
  (h : is_geometric_sequence a c d) :
  (is_geometric_sequence (a*b) (b+c) (c+d) ∨
   is_geometric_sequence (a*b) (b*c) (c*d) ∨
   is_geometric_sequence (a*b) (b-c) (-d)) ∧
  ¬(is_geometric_sequence (a*b) (b+c) (c+d) ∧
    is_geometric_sequence (a*b) (b*c) (c*d)) ∧
  ¬(is_geometric_sequence (a*b) (b+c) (c+d) ∧
    is_geometric_sequence (a*b) (b-c) (-d)) ∧
  ¬(is_geometric_sequence (a*b) (b*c) (c*d) ∧
    is_geometric_sequence (a*b) (b-c) (-d)) :=
by sorry

end geometric_sequence_problem_l1180_118015


namespace mississippi_arrangements_l1180_118084

def word : String := "MISSISSIPPI"

def letter_counts : List (Char × Nat) := [('M', 1), ('I', 4), ('S', 4), ('P', 2)]

def total_letters : Nat := 11

def arrangements_starting_with_p : Nat := 6300

theorem mississippi_arrangements :
  (List.sum (letter_counts.map (fun p => p.2)) = total_letters) →
  (List.length letter_counts = 4) →
  (List.any letter_counts (fun p => p.1 = 'P' ∧ p.2 ≥ 1)) →
  (arrangements_starting_with_p = (Nat.factorial (total_letters - 1)) / 
    (List.prod (letter_counts.map (fun p => 
      if p.1 = 'P' then Nat.factorial (p.2 - 1) else Nat.factorial p.2)))) :=
by sorry

end mississippi_arrangements_l1180_118084


namespace pillow_average_cost_l1180_118085

theorem pillow_average_cost (n : ℕ) (avg_cost : ℚ) (additional_cost : ℚ) :
  n = 4 →
  avg_cost = 5 →
  additional_cost = 10 →
  (n * avg_cost + additional_cost) / (n + 1) = 6 := by
sorry

end pillow_average_cost_l1180_118085


namespace f_increasing_when_x_greater_than_one_l1180_118082

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 5

theorem f_increasing_when_x_greater_than_one :
  ∀ x : ℝ, x > 1 → (deriv f) x > 0 := by sorry

end f_increasing_when_x_greater_than_one_l1180_118082


namespace orchestra_students_l1180_118050

theorem orchestra_students (band_students : ℕ → ℕ) (choir_students : ℕ) (total_students : ℕ) :
  (∀ x : ℕ, band_students x = 2 * x) →
  choir_students = 28 →
  total_students = 88 →
  ∃ x : ℕ, x + band_students x + choir_students = total_students ∧ x = 20 :=
by sorry

end orchestra_students_l1180_118050


namespace sandwich_percentage_l1180_118040

theorem sandwich_percentage (total_weight : ℝ) (condiment_weight : ℝ) 
  (h1 : total_weight = 150)
  (h2 : condiment_weight = 45) :
  (total_weight - condiment_weight) / total_weight * 100 = 70 := by
  sorry

end sandwich_percentage_l1180_118040


namespace train_length_calculation_l1180_118014

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length_calculation (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 27 → 
  ∃ (length_m : ℝ), abs (length_m - 450.09) < 0.01 ∧ length_m = speed_kmh * (1000 / 3600) * time_s := by
  sorry

end train_length_calculation_l1180_118014


namespace intersection_A_B_l1180_118077

def A : Set ℝ := {x | ∃ n : ℤ, x = 2 * n - 1}
def B : Set ℝ := {x | x^2 - 4*x < 0}

theorem intersection_A_B : A ∩ B = {1, 3} := by sorry

end intersection_A_B_l1180_118077


namespace highlighted_area_theorem_l1180_118005

theorem highlighted_area_theorem (circle_area : ℝ) (angle1 : ℝ) (angle2 : ℝ) :
  circle_area = 20 →
  angle1 = 60 →
  angle2 = 30 →
  (angle1 + angle2) / 360 * circle_area = 5 :=
by sorry

end highlighted_area_theorem_l1180_118005


namespace fraction_evaluation_l1180_118061

theorem fraction_evaluation (x : ℝ) (h : x = 3) : (x^6 + 8*x^3 + 16) / (x^3 + 4) = 31 := by
  sorry

end fraction_evaluation_l1180_118061


namespace imaginary_part_of_complex_fraction_l1180_118049

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (1 - 3*I) / (1 - I) → z.im = -1 := by
  sorry

end imaginary_part_of_complex_fraction_l1180_118049


namespace cubic_function_unique_determination_l1180_118051

def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem cubic_function_unique_determination 
  (f : ℝ → ℝ) 
  (h_cubic : ∃ a b c d : ℝ, ∀ x, f x = cubic_function a b c d x) 
  (h_max : f 1 = 4 ∧ (deriv f) 1 = 0)
  (h_min : f 3 = 0 ∧ (deriv f) 3 = 0)
  (h_origin : f 0 = 0) :
  ∀ x, f x = x^3 - 6*x^2 + 9*x :=
sorry

end cubic_function_unique_determination_l1180_118051


namespace quadratic_vertex_l1180_118080

/-- The quadratic function f(x) = 2(x-1)^2 + 5 -/
def f (x : ℝ) : ℝ := 2 * (x - 1)^2 + 5

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (1, 5)

/-- Theorem: The vertex of the quadratic function f(x) = 2(x-1)^2 + 5 is (1, 5) -/
theorem quadratic_vertex : 
  ∀ x : ℝ, f x ≥ f (vertex.1) ∧ f (vertex.1) = vertex.2 := by
  sorry

end quadratic_vertex_l1180_118080


namespace angle_C_measure_l1180_118026

-- Define the angles A, B, and C
variable (A B C : ℝ)

-- Define the parallel lines condition
variable (p_parallel_q : Bool)

-- State the theorem
theorem angle_C_measure :
  p_parallel_q = true →
  A = (1/4) * B →
  B + C = 180 →
  C = 36 := by
  sorry

end angle_C_measure_l1180_118026


namespace not_perfect_square_l1180_118038

theorem not_perfect_square (n d : ℕ+) (h : d ∣ (2 * n ^ 2)) :
  ¬∃ (x : ℕ), (n : ℝ) ^ 2 + d = (x : ℝ) ^ 2 := by
  sorry

end not_perfect_square_l1180_118038


namespace ant_return_probability_l1180_118046

/-- Represents a vertex in a tetrahedron -/
inductive Vertex : Type
  | A | B | C | D

/-- Represents the state of the ant's position -/
structure AntState :=
  (position : Vertex)
  (distance : ℕ)

/-- The probability of choosing any edge at a vertex -/
def edgeProbability : ℚ := 1 / 3

/-- The total distance the ant needs to travel -/
def totalDistance : ℕ := 4

/-- Function to calculate the probability of the ant being at a specific vertex after a certain distance -/
noncomputable def probabilityAtVertex (v : Vertex) (d : ℕ) : ℚ :=
  sorry

/-- Theorem stating the probability of the ant returning to vertex A after 4 moves -/
theorem ant_return_probability :
  probabilityAtVertex Vertex.A totalDistance = 7 / 27 :=
sorry

end ant_return_probability_l1180_118046


namespace cube_surface_area_l1180_118025

/-- The surface area of a cube given the sum of its edge lengths -/
theorem cube_surface_area (sum_of_edges : ℝ) (h : sum_of_edges = 36) : 
  6 * (sum_of_edges / 12)^2 = 54 := by
  sorry

#check cube_surface_area

end cube_surface_area_l1180_118025


namespace greatest_power_of_three_l1180_118091

def p : ℕ := (List.range 35).foldl (· * ·) 1

theorem greatest_power_of_three (k : ℕ) : k ≤ 15 ∧ 3^k ∣ p ∧ ∀ m > k, ¬(3^m ∣ p) :=
sorry

end greatest_power_of_three_l1180_118091


namespace hyperbola_triangle_perimeter_l1180_118069

def hyperbola_equation (x y : ℝ) : Prop := x^2 / 9 - y^2 / 7 = 1

def is_focus (F : ℝ × ℝ) (C : (ℝ × ℝ → Prop)) : Prop := sorry

def is_right_branch (P : ℝ × ℝ) (C : (ℝ × ℝ → Prop)) : Prop := sorry

def distance (P Q : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_triangle_perimeter 
  (C : ℝ × ℝ → Prop)
  (F₁ F₂ P : ℝ × ℝ) :
  (∀ x y, C (x, y) ↔ hyperbola_equation x y) →
  is_focus F₁ C ∧ is_focus F₂ C →
  F₁.1 < F₂.1 →
  is_right_branch P C →
  distance P F₁ = 8 →
  distance P F₁ + distance P F₂ + distance F₁ F₂ = 18 :=
sorry

end hyperbola_triangle_perimeter_l1180_118069


namespace largest_common_divisor_510_399_l1180_118068

theorem largest_common_divisor_510_399 : Nat.gcd 510 399 = 57 := by
  sorry

end largest_common_divisor_510_399_l1180_118068


namespace coin_distribution_ways_l1180_118053

/-- The number of coin denominations available -/
def num_denominations : ℕ := 4

/-- The number of boys receiving coins -/
def num_boys : ℕ := 6

/-- Theorem stating the number of ways to distribute coins -/
theorem coin_distribution_ways : (num_denominations ^ num_boys : ℕ) = 4096 := by
  sorry

end coin_distribution_ways_l1180_118053


namespace square_vector_properties_l1180_118002

/-- Given a square ABCD with side length 2 and vectors a and b satisfying the given conditions,
    prove that a · b = 2 and (b - 4a) ⊥ b -/
theorem square_vector_properties (a b : ℝ × ℝ) :
  let A := (0, 0)
  let B := (2, 0)
  let C := (2, 2)
  let D := (0, 2)
  let AB := B - A
  let BC := C - B
  AB = 2 • a →
  BC = b - 2 • a →
  a • b = 2 ∧ (b - 4 • a) • b = 0 := by
  sorry

end square_vector_properties_l1180_118002


namespace two_pump_fill_time_l1180_118001

theorem two_pump_fill_time (small_pump_time large_pump_time : ℝ) 
  (h_small : small_pump_time = 3)
  (h_large : large_pump_time = 1/4)
  (h_positive : small_pump_time > 0 ∧ large_pump_time > 0) :
  1 / (1 / small_pump_time + 1 / large_pump_time) = 3/13 := by
  sorry

end two_pump_fill_time_l1180_118001


namespace corrected_mean_problem_l1180_118039

/-- Calculates the corrected mean of a set of observations after fixing an error in one observation -/
def corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n : ℚ) * original_mean + (correct_value - incorrect_value) / (n : ℚ)

/-- Theorem stating that the corrected mean for the given problem is 32.5 -/
theorem corrected_mean_problem :
  corrected_mean 50 32 23 48 = 32.5 := by
  sorry

end corrected_mean_problem_l1180_118039


namespace four_thirteenths_cycle_sum_l1180_118071

/-- Represents a repeating decimal with a two-digit cycle -/
structure RepeatingDecimal where
  whole : ℕ
  cycle : ℕ × ℕ

/-- Converts a fraction to a repeating decimal -/
def fractionToRepeatingDecimal (n d : ℕ) : RepeatingDecimal :=
  sorry

/-- Extracts the cycle digits from a repeating decimal -/
def getCycleDigits (r : RepeatingDecimal) : ℕ × ℕ :=
  r.cycle

theorem four_thirteenths_cycle_sum :
  let r := fractionToRepeatingDecimal 4 13
  let (c, d) := getCycleDigits r
  c + d = 3 := by
    sorry

end four_thirteenths_cycle_sum_l1180_118071


namespace tank_insulation_cost_l1180_118065

/-- Calculates the surface area of a rectangular prism -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Calculates the cost of insulating a rectangular tank -/
def insulationCost (l w h costPerSqFt : ℝ) : ℝ :=
  surfaceArea l w h * costPerSqFt

/-- Theorem: The cost of insulating a rectangular tank with given dimensions is $1240 -/
theorem tank_insulation_cost :
  insulationCost 5 3 2 20 = 1240 := by
  sorry

end tank_insulation_cost_l1180_118065


namespace justine_fewer_than_ylona_l1180_118047

/-- The number of rubber bands each person had initially and after Bailey's distribution --/
structure RubberBands where
  justine_initial : ℕ
  bailey_initial : ℕ
  ylona_initial : ℕ
  bailey_final : ℕ

/-- The conditions of the rubber band problem --/
def rubber_band_problem (rb : RubberBands) : Prop :=
  rb.justine_initial = rb.bailey_initial + 10 ∧
  rb.justine_initial < rb.ylona_initial ∧
  rb.bailey_final = rb.bailey_initial - 4 ∧
  rb.bailey_final = 8 ∧
  rb.ylona_initial = 24

/-- Theorem stating that Justine had 2 fewer rubber bands than Ylona initially --/
theorem justine_fewer_than_ylona (rb : RubberBands) 
  (h : rubber_band_problem rb) : 
  rb.ylona_initial - rb.justine_initial = 2 := by
  sorry

end justine_fewer_than_ylona_l1180_118047


namespace skew_lines_angle_equals_dihedral_angle_l1180_118073

-- Define the dihedral angle
def dihedral_angle (α l β : Line3) : ℝ := sorry

-- Define perpendicularity between a line and a plane
def perpendicular (m : Line3) (α : Plane3) : Prop := sorry

-- Define the angle between two skew lines
def skew_line_angle (m n : Line3) : ℝ := sorry

-- Theorem statement
theorem skew_lines_angle_equals_dihedral_angle 
  (α l β : Line3) (m n : Line3) :
  dihedral_angle α l β = 60 →
  perpendicular m α →
  perpendicular n β →
  skew_line_angle m n = 60 := by sorry

end skew_lines_angle_equals_dihedral_angle_l1180_118073


namespace intersection_midpoint_sum_l1180_118063

/-- Given a line y = ax + b that intersects y = x^2 at two distinct points,
    if the midpoint of these points is (5, 101), then a + b = -41 -/
theorem intersection_midpoint_sum (a b : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    a * x₁ + b = x₁^2 ∧ 
    a * x₂ + b = x₂^2 ∧ 
    (x₁ + x₂) / 2 = 5 ∧ 
    (x₁^2 + x₂^2) / 2 = 101) →
  a + b = -41 := by
sorry

end intersection_midpoint_sum_l1180_118063
