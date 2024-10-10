import Mathlib

namespace square_side_difference_sum_l1167_116759

theorem square_side_difference_sum (a b c : ℝ) 
  (ha : a^2 = 25) (hb : b^2 = 81) (hc : c^2 = 64) : 
  (b - a) + (b - c) = 5 := by sorry

end square_side_difference_sum_l1167_116759


namespace mass_of_Fe2CO3_3_l1167_116713

/-- The molar mass of iron in g/mol -/
def molar_mass_Fe : ℝ := 55.845

/-- The molar mass of carbon in g/mol -/
def molar_mass_C : ℝ := 12.011

/-- The molar mass of oxygen in g/mol -/
def molar_mass_O : ℝ := 15.999

/-- The number of moles of Fe2(CO3)3 -/
def num_moles : ℝ := 12

/-- The molar mass of Fe2(CO3)3 in g/mol -/
def molar_mass_Fe2CO3_3 : ℝ :=
  2 * molar_mass_Fe + 3 * molar_mass_C + 9 * molar_mass_O

/-- The mass of Fe2(CO3)3 in grams -/
def mass_Fe2CO3_3 : ℝ := num_moles * molar_mass_Fe2CO3_3

theorem mass_of_Fe2CO3_3 : mass_Fe2CO3_3 = 3500.568 := by
  sorry

end mass_of_Fe2CO3_3_l1167_116713


namespace unique_ages_solution_l1167_116796

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem unique_ages_solution :
  ∃! (a b c : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a = 2 * b ∧
    b = c - 7 ∧
    is_prime (a + b + c) ∧
    a + b + c < 70 ∧
    sum_of_digits (a + b + c) = 13 ∧
    a = 30 ∧ b = 15 ∧ c = 22 :=
sorry

end unique_ages_solution_l1167_116796


namespace train_passing_jogger_l1167_116797

theorem train_passing_jogger (jogger_speed train_speed : ℝ) 
  (train_length initial_distance : ℝ) :
  jogger_speed = 9 →
  train_speed = 45 →
  train_length = 120 →
  initial_distance = 240 →
  (train_speed - jogger_speed) * (5 / 18) * 
    ((initial_distance + train_length) / ((train_speed - jogger_speed) * (5 / 18))) = 36 := by
  sorry

end train_passing_jogger_l1167_116797


namespace geometric_sequence_in_arithmetic_progression_l1167_116785

theorem geometric_sequence_in_arithmetic_progression (x : ℚ) (hx : x > 0) :
  ∃ (i j k : ℕ), i < j ∧ j < k ∧ (x + i) * (x + k) = (x + j)^2 := by
  sorry

end geometric_sequence_in_arithmetic_progression_l1167_116785


namespace marble_probability_l1167_116743

theorem marble_probability (total : ℕ) (blue red : ℕ) (h1 : total = 20) (h2 : blue = 6) (h3 : red = 9) :
  let white := total - (blue + red)
  (red + white : ℚ) / total = 7 / 10 := by
sorry

end marble_probability_l1167_116743


namespace expected_pairs_eq_63_l1167_116714

/-- The number of students in the gathering -/
def n : ℕ := 15

/-- The probability of any pair of students liking each other -/
def p : ℚ := 3/5

/-- The expected number of pairs that like each other -/
def expected_pairs : ℚ := p * (n.choose 2)

theorem expected_pairs_eq_63 : expected_pairs = 63 := by sorry

end expected_pairs_eq_63_l1167_116714


namespace twentieth_term_of_sequence_l1167_116750

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1 : ℤ) * d

theorem twentieth_term_of_sequence (a₁ d : ℤ) (h₁ : a₁ = 8) (h₂ : d = -3) :
  arithmeticSequenceTerm a₁ d 20 = -49 := by
  sorry

end twentieth_term_of_sequence_l1167_116750


namespace central_cell_value_l1167_116710

/-- Represents a 3x3 table of real numbers -/
def Table := Fin 3 → Fin 3 → ℝ

/-- The product of numbers in a row equals 10 -/
def row_product (t : Table) : Prop :=
  ∀ i : Fin 3, (t i 0) * (t i 1) * (t i 2) = 10

/-- The product of numbers in a column equals 10 -/
def col_product (t : Table) : Prop :=
  ∀ j : Fin 3, (t 0 j) * (t 1 j) * (t 2 j) = 10

/-- The product of numbers in any 2x2 square equals 3 -/
def square_product (t : Table) : Prop :=
  ∀ i j : Fin 2, (t i j) * (t i (j+1)) * (t (i+1) j) * (t (i+1) (j+1)) = 3

theorem central_cell_value (t : Table) 
  (h_row : row_product t) 
  (h_col : col_product t) 
  (h_square : square_product t) : 
  t 1 1 = 0.00081 := by
  sorry

end central_cell_value_l1167_116710


namespace added_number_after_doubling_l1167_116782

theorem added_number_after_doubling (original : ℕ) (added : ℕ) : 
  original = 5 → 3 * (2 * original + added) = 57 → added = 9 := by
  sorry

end added_number_after_doubling_l1167_116782


namespace triangle_side_length_proof_l1167_116749

noncomputable def triangle_side_length (A B C : Real) : Real :=
  let AB : Real := 1
  let AC : Real := 2
  let cos_B_plus_sin_C : Real := 1
  (2 * Real.sqrt 21 + 3) / 5

theorem triangle_side_length_proof (A B C : Real) :
  let AB : Real := 1
  let AC : Real := 2
  let cos_B_plus_sin_C : Real := 1
  triangle_side_length A B C = (2 * Real.sqrt 21 + 3) / 5 :=
by
  sorry

end triangle_side_length_proof_l1167_116749


namespace complex_square_sum_l1167_116725

theorem complex_square_sum (a b : ℝ) : (1 + Complex.I) ^ 2 = a + b * Complex.I → a + b = 4 := by
  sorry

end complex_square_sum_l1167_116725


namespace quadratic_inequality_solution_range_l1167_116764

theorem quadratic_inequality_solution_range (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 2 * x + 6 * k < 0) ↔ k < -Real.sqrt 6 / 6 := by
  sorry

end quadratic_inequality_solution_range_l1167_116764


namespace complex_number_system_l1167_116763

theorem complex_number_system (x y z : ℂ) 
  (h1 : x + y + z = 3)
  (h2 : x^2 + y^2 + z^2 = 3)
  (h3 : x^3 + y^3 + z^3 = 3) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end complex_number_system_l1167_116763


namespace vector_identity_l1167_116744

variable {V : Type*} [AddCommGroup V]

/-- For any four points A, B, C, and D in a vector space, 
    CB + AD - AB = CD -/
theorem vector_identity (A B C D : V) : C - B + (D - A) - (B - A) = D - C := by
  sorry

end vector_identity_l1167_116744


namespace odd_function_negative_domain_l1167_116746

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_positive : ∀ x ≥ 0, f x = x^2 + 2*x) :
  ∀ x < 0, f x = -x^2 + 2*x :=
by sorry

end odd_function_negative_domain_l1167_116746


namespace intersection_implies_a_value_l1167_116781

def M (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def P (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value :
  ∀ a : ℝ, M a ∩ P a = {-3} → a = -1 := by
sorry

end intersection_implies_a_value_l1167_116781


namespace line_parameterization_vector_l1167_116767

/-- The line equation y = (4x - 7) / 3 -/
def line_equation (x y : ℝ) : Prop := y = (4 * x - 7) / 3

/-- The parameterization of the line -/
def parameterization (v d : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (v.1 + t * d.1, v.2 + t * d.2)

/-- The distance constraint -/
def distance_constraint (p : ℝ × ℝ) (t : ℝ) : Prop :=
  p.1 ≥ 5 → ‖(p.1 - 5, p.2 - 2)‖ = 2 * t

/-- The theorem statement -/
theorem line_parameterization_vector :
  ∃ (v : ℝ × ℝ), ∀ (x y t : ℝ),
    let p := parameterization v (6/5, 8/5) t
    line_equation p.1 p.2 ∧
    distance_constraint p t :=
by sorry

end line_parameterization_vector_l1167_116767


namespace arithmetic_sequence_property_l1167_116748

/-- An arithmetic sequence -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : arithmeticSequence a) 
  (h_sum : a 2 + a 12 = 32) : 
  2 * a 3 + a 15 = 48 := by
  sorry

end arithmetic_sequence_property_l1167_116748


namespace rectangle_dimensions_l1167_116795

theorem rectangle_dimensions (x : ℝ) : 
  x > 3 →
  (x - 3) * (3 * x + 6) = 9 * x - 9 →
  x = (21 + Real.sqrt 549) / 6 := by
sorry

end rectangle_dimensions_l1167_116795


namespace wage_increase_percentage_l1167_116716

theorem wage_increase_percentage (old_wage new_wage : ℝ) (h1 : old_wage = 20) (h2 : new_wage = 28) :
  (new_wage - old_wage) / old_wage * 100 = 40 := by
sorry

end wage_increase_percentage_l1167_116716


namespace calculation_proof_l1167_116752

theorem calculation_proof : (3.6 * 0.3) / 0.2 = 5.4 := by
  sorry

end calculation_proof_l1167_116752


namespace condition1_coordinates_condition2_coordinates_l1167_116741

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given point A with coordinates dependent on parameter a -/
def A (a : ℝ) : Point := ⟨3*a + 2, 2*a - 4⟩

/-- Fixed point B -/
def B : Point := ⟨3, 4⟩

/-- Condition 1: Line AB is parallel to x-axis -/
def parallel_to_x_axis (A B : Point) : Prop :=
  A.y = B.y

/-- Condition 2: Distance from A to both coordinate axes is equal -/
def equal_distance_to_axes (A : Point) : Prop :=
  |A.x| = |A.y|

/-- Theorem for Condition 1 -/
theorem condition1_coordinates :
  ∃ a : ℝ, parallel_to_x_axis (A a) B → A a = ⟨14, 4⟩ := by sorry

/-- Theorem for Condition 2 -/
theorem condition2_coordinates :
  ∃ a : ℝ, equal_distance_to_axes (A a) → 
    (A a = ⟨-16, -16⟩ ∨ A a = ⟨3.2, -3.2⟩) := by sorry

end condition1_coordinates_condition2_coordinates_l1167_116741


namespace parallel_line_plane_condition_l1167_116704

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the subset relation for lines and planes
variable (subset_line_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_line_plane_condition 
  (α β : Plane) (m : Line) :
  parallel_planes α β → 
  ¬subset_line_plane m β → 
  parallel_line_plane m α :=
sorry

end parallel_line_plane_condition_l1167_116704


namespace prob_at_least_one_value_l1167_116709

/-- The probability of event A occurring -/
def prob_A : ℝ := 0.4

/-- The probability of event B occurring -/
def prob_B : ℝ := 0.5

/-- Events A and B are independent -/
axiom events_independent : True

/-- The probability of at least one of the events A or B occurring -/
def prob_at_least_one : ℝ := 1 - (1 - prob_A) * (1 - prob_B)

/-- Theorem: The probability of at least one of the events A or B occurring is 0.7 -/
theorem prob_at_least_one_value : prob_at_least_one = 0.7 := by
  sorry

end prob_at_least_one_value_l1167_116709


namespace sqrt_sum_equation_l1167_116755

theorem sqrt_sum_equation (x : ℝ) 
  (h : Real.sqrt (49 - x^2) - Real.sqrt (25 - x^2) = 3) : 
  Real.sqrt (49 - x^2) + Real.sqrt (25 - x^2) = 8 := by
  sorry

end sqrt_sum_equation_l1167_116755


namespace extreme_value_M_inequality_condition_l1167_116702

open Real

noncomputable def f (x : ℝ) : ℝ := log x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := x + 1/x + a * (1/x)

noncomputable def M (x : ℝ) : ℝ := x - log x + 2/x

theorem extreme_value_M :
  (∀ x > 0, M x ≥ 3 - log 2) ∧
  (M 2 = 3 - log 2) ∧
  (∀ b : ℝ, ∃ x > 0, M x > b) :=
sorry

theorem inequality_condition (m : ℝ) :
  (∀ x > 0, 1 / (x + 1/x) ≤ 1 / (2 + m * (log x)^2)) ↔ 0 ≤ m ∧ m ≤ 1 :=
sorry

end extreme_value_M_inequality_condition_l1167_116702


namespace equation_one_solutions_equation_two_solutions_equation_three_solutions_equation_four_solutions_l1167_116708

-- (1) (3x-1)^2 = (x+1)^2
theorem equation_one_solutions (x : ℝ) :
  (3*x - 1)^2 = (x + 1)^2 ↔ x = 0 ∨ x = 1 := by sorry

-- (2) (x-1)^2+2x(x-1)=0
theorem equation_two_solutions (x : ℝ) :
  (x - 1)^2 + 2*x*(x - 1) = 0 ↔ x = 1 ∨ x = 1/3 := by sorry

-- (3) x^2 - 4x + 1 = 0
theorem equation_three_solutions (x : ℝ) :
  x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 := by sorry

-- (4) 2x^2 + 7x - 4 = 0
theorem equation_four_solutions (x : ℝ) :
  2*x^2 + 7*x - 4 = 0 ↔ x = 1/2 ∨ x = -4 := by sorry

end equation_one_solutions_equation_two_solutions_equation_three_solutions_equation_four_solutions_l1167_116708


namespace kolya_purchase_l1167_116778

/-- Represents the price of an item in kopecks -/
def ItemPrice (rubles : ℕ) : ℕ := 100 * rubles + 99

/-- Represents Kolya's total purchase in kopecks -/
def TotalPurchase : ℕ := 200 * 100 + 83

/-- Predicate to check if a given number of items satisfies the purchase conditions -/
def ValidPurchase (n : ℕ) : Prop :=
  ∃ (rubles : ℕ), n * ItemPrice rubles = TotalPurchase

theorem kolya_purchase :
  {n : ℕ | ValidPurchase n} = {17, 117} := by sorry

end kolya_purchase_l1167_116778


namespace system_and_expression_solution_l1167_116728

theorem system_and_expression_solution :
  -- System of equations
  (∃ (x y : ℝ), 2*x + y = 4 ∧ x + 2*y = 5 ∧ x = 1 ∧ y = 2) ∧
  -- Simplified expression evaluation
  (let x : ℝ := -2; (x^2 + 1) / x = -5/2) :=
sorry

end system_and_expression_solution_l1167_116728


namespace cubic_sum_minus_product_l1167_116793

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 13) 
  (sum_prod_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 1027 := by
  sorry

end cubic_sum_minus_product_l1167_116793


namespace length_eb_is_two_l1167_116707

/-- An equilateral triangle with points on its sides -/
structure TriangleWithPoints where
  -- The side length of the equilateral triangle
  side : ℝ
  -- Lengths of segments
  ad : ℝ
  de : ℝ
  ef : ℝ
  fa : ℝ
  -- Conditions
  equilateral : side > 0
  d_on_ab : ad ≤ side
  e_on_bc : de ≤ side
  f_on_ca : fa ≤ side
  ad_value : ad = 4
  de_value : de = 8
  ef_value : ef = 10
  fa_value : fa = 6

/-- The length of segment EB in the triangle -/
def length_eb (t : TriangleWithPoints) : ℝ := 2

/-- Theorem: The length of EB is 2 -/
theorem length_eb_is_two (t : TriangleWithPoints) : length_eb t = 2 := by
  sorry

end length_eb_is_two_l1167_116707


namespace hens_count_l1167_116758

/-- Represents the number of hens in the farm -/
def num_hens : ℕ := sorry

/-- Represents the number of cows in the farm -/
def num_cows : ℕ := sorry

/-- The total number of heads in the farm -/
def total_heads : ℕ := 48

/-- The total number of feet in the farm -/
def total_feet : ℕ := 140

/-- Each hen has 1 head and 2 feet -/
def hen_head_feet : ℕ × ℕ := (1, 2)

/-- Each cow has 1 head and 4 feet -/
def cow_head_feet : ℕ × ℕ := (1, 4)

theorem hens_count : num_hens = 26 :=
  by sorry

end hens_count_l1167_116758


namespace problem_statement_l1167_116739

theorem problem_statement (a b : ℝ) (h1 : a * b = 2) (h2 : a - b = 3) :
  a^3 * b - 2 * a^2 * b^2 + a * b^3 = 18 := by
sorry

end problem_statement_l1167_116739


namespace group_size_proof_l1167_116733

theorem group_size_proof (total : ℕ) (older : ℕ) (prob : ℚ) 
  (h1 : older = 90)
  (h2 : prob = 40/130)
  (h3 : prob = (total - older) / total) :
  total = 130 := by
  sorry

end group_size_proof_l1167_116733


namespace max_value_of_symmetric_f_l1167_116766

def f (a b x : ℝ) : ℝ := (1 - x^2) * (x^2 + a*x + b)

theorem max_value_of_symmetric_f (a b : ℝ) :
  (∀ x : ℝ, f a b x = f a b (-4 - x)) →
  (∃ x : ℝ, f a b x = 0 ∧ f a b (-4 - x) = 0) →
  (∃ m : ℝ, ∀ x : ℝ, f a b x ≤ m ∧ ∃ x₀ : ℝ, f a b x₀ = m) →
  (∃ m : ℝ, (∀ x : ℝ, f a b x ≤ m) ∧ (∃ x₀ : ℝ, f a b x₀ = m) ∧ m = 16) :=
sorry

end max_value_of_symmetric_f_l1167_116766


namespace profit_per_meter_l1167_116774

/-- Calculate the profit per meter of cloth -/
theorem profit_per_meter (meters_sold : ℕ) (selling_price : ℕ) (cost_price_per_meter : ℕ) :
  meters_sold = 45 →
  selling_price = 4500 →
  cost_price_per_meter = 88 →
  (selling_price - meters_sold * cost_price_per_meter) / meters_sold = 12 :=
by sorry

end profit_per_meter_l1167_116774


namespace train_length_l1167_116754

/-- The length of a train given its speed, time to pass a platform, and platform length -/
theorem train_length (train_speed : Real) (pass_time : Real) (platform_length : Real) :
  train_speed = 45 * (5/18) ∧ 
  pass_time = 44 ∧ 
  platform_length = 190 →
  (train_speed * pass_time) - platform_length = 360 := by
  sorry


end train_length_l1167_116754


namespace absolute_value_equality_l1167_116717

theorem absolute_value_equality (x : ℝ) (h : x < -2) :
  |x - Real.sqrt ((x + 2)^2)| = -2*x - 2 := by
  sorry

end absolute_value_equality_l1167_116717


namespace two_times_larger_by_one_l1167_116788

theorem two_times_larger_by_one (a : ℝ) : 
  (2 * a + 1) = (2 * a) + 1 := by sorry

end two_times_larger_by_one_l1167_116788


namespace breakfast_cost_theorem_l1167_116783

/-- The cost of breakfast for Francis and Kiera -/
def breakfast_cost (muffin_price fruit_cup_price : ℕ) 
  (francis_muffins francis_fruit_cups : ℕ)
  (kiera_muffins kiera_fruit_cups : ℕ) : ℕ :=
  (francis_muffins + kiera_muffins) * muffin_price +
  (francis_fruit_cups + kiera_fruit_cups) * fruit_cup_price

/-- Theorem stating the total cost of Francis and Kiera's breakfast -/
theorem breakfast_cost_theorem : 
  breakfast_cost 2 3 2 2 2 1 = 17 := by
  sorry

end breakfast_cost_theorem_l1167_116783


namespace fourth_task_completion_time_l1167_116732

-- Define the start and end times
def start_time : ℕ := 12 * 60  -- 12:00 PM in minutes
def end_time : ℕ := 15 * 60    -- 3:00 PM in minutes

-- Define the number of tasks completed
def num_tasks : ℕ := 3

-- Theorem to prove
theorem fourth_task_completion_time 
  (h1 : end_time - start_time = num_tasks * (end_time - start_time) / num_tasks) -- Tasks are equally time-consuming
  (h2 : (end_time - start_time) % num_tasks = 0) -- Ensures division is exact
  : end_time + (end_time - start_time) / num_tasks = 16 * 60 := -- 4:00 PM in minutes
by
  sorry

end fourth_task_completion_time_l1167_116732


namespace divisibility_implies_equality_l1167_116742

theorem divisibility_implies_equality (a b : ℕ) :
  (4 * a * b - 1) ∣ (4 * a^2 - 1)^2 → a = b := by
  sorry

end divisibility_implies_equality_l1167_116742


namespace rectangle_tileability_l1167_116773

/-- A rectangle can be tiled with 1 × b tiles -/
def IsTileable (m n b : ℕ) : Prop := sorry

/-- For an even b, there exists M such that for all m, n > M with mn even, 
    an m × n rectangle is (1, b)-tileable -/
theorem rectangle_tileability (b : ℕ) (h_even : Even b) : 
  ∃ M : ℕ, ∀ m n : ℕ, m > M → n > M → Even (m * n) → IsTileable m n b := by sorry

end rectangle_tileability_l1167_116773


namespace netball_points_calculation_l1167_116700

theorem netball_points_calculation 
  (w d : ℕ) 
  (h1 : w > d) 
  (h2 : 7 * w + 3 * d = 44) : 
  5 * w + 2 * d = 31 := by
sorry

end netball_points_calculation_l1167_116700


namespace sylvia_earnings_l1167_116736

-- Define the work durations
def monday_hours : ℚ := 5/2
def tuesday_minutes : ℕ := 40
def wednesday_start : ℕ := 9 * 60 + 15  -- 9:15 AM in minutes
def wednesday_end : ℕ := 11 * 60 + 50   -- 11:50 AM in minutes
def thursday_minutes : ℕ := 45

-- Define the hourly rate
def hourly_rate : ℚ := 4

-- Define the function to calculate total earnings
def total_earnings : ℚ :=
  let total_minutes : ℚ := 
    monday_hours * 60 + 
    tuesday_minutes + 
    (wednesday_end - wednesday_start) + 
    thursday_minutes
  let total_hours : ℚ := total_minutes / 60
  total_hours * hourly_rate

-- Theorem statement
theorem sylvia_earnings : total_earnings = 26 := by
  sorry

end sylvia_earnings_l1167_116736


namespace min_large_buses_correct_l1167_116760

/-- The minimum number of large buses required to transport students --/
def min_large_buses (total_students : ℕ) (large_capacity : ℕ) (small_capacity : ℕ) (min_small_buses : ℕ) : ℕ :=
  let remaining_students := total_students - min_small_buses * small_capacity
  (remaining_students + large_capacity - 1) / large_capacity

theorem min_large_buses_correct (total_students : ℕ) (large_capacity : ℕ) (small_capacity : ℕ) (min_small_buses : ℕ)
  (h1 : total_students = 523)
  (h2 : large_capacity = 45)
  (h3 : small_capacity = 30)
  (h4 : min_small_buses = 5) :
  min_large_buses total_students large_capacity small_capacity min_small_buses = 9 := by
  sorry

#eval min_large_buses 523 45 30 5

end min_large_buses_correct_l1167_116760


namespace circle_center_point_satisfies_center_circle_center_is_4_2_l1167_116706

/-- The center of a circle given by the equation x^2 - 8x + y^2 - 4y = 16 is (4, 2) -/
theorem circle_center (x y : ℝ) : 
  x^2 - 8*x + y^2 - 4*y = 16 → (x - 4)^2 + (y - 2)^2 = 36 := by
  sorry

/-- The point (4, 2) satisfies the center condition of the circle -/
theorem point_satisfies_center : 
  (4 : ℝ)^2 - 8*4 + (2 : ℝ)^2 - 4*2 = 16 := by
  sorry

/-- The center of the circle with equation x^2 - 8x + y^2 - 4y = 16 is (4, 2) -/
theorem circle_center_is_4_2 : 
  ∃! (c : ℝ × ℝ), ∀ (x y : ℝ), x^2 - 8*x + y^2 - 4*y = 16 → (x - c.1)^2 + (y - c.2)^2 = 36 ∧ c = (4, 2) := by
  sorry

end circle_center_point_satisfies_center_circle_center_is_4_2_l1167_116706


namespace arithmetic_simplification_l1167_116790

theorem arithmetic_simplification :
  2 - (-3) - 6 - (-8) - 10 - (-12) = 9 := by sorry

end arithmetic_simplification_l1167_116790


namespace consecutive_zeros_in_prime_power_l1167_116770

theorem consecutive_zeros_in_prime_power (p : Nat) (h : Nat.Prime p) :
  ∃ n : Nat, n > 0 ∧ p^n % 10^2002 = 0 ∧ p^n % 10^2003 ≠ 0 := by
  sorry

end consecutive_zeros_in_prime_power_l1167_116770


namespace quadratic_form_sum_l1167_116735

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → 
  a + h + k = -6 := by sorry

end quadratic_form_sum_l1167_116735


namespace gloves_with_pair_count_l1167_116753

-- Define the number of glove pairs
def num_pairs : ℕ := 4

-- Define the total number of gloves
def total_gloves : ℕ := 2 * num_pairs

-- Define the number of gloves to pick
def gloves_to_pick : ℕ := 4

-- Theorem statement
theorem gloves_with_pair_count :
  (Nat.choose total_gloves gloves_to_pick) - (2^num_pairs) = 54 := by
  sorry

end gloves_with_pair_count_l1167_116753


namespace sector_shape_area_l1167_116721

theorem sector_shape_area (r : ℝ) (h : r = 12) : 
  let circle_area := π * r^2
  let sector_90 := (90 / 360) * circle_area
  let sector_120 := (120 / 360) * circle_area
  sector_90 + sector_120 = 84 * π := by
sorry

end sector_shape_area_l1167_116721


namespace shirts_sold_proof_l1167_116780

/-- The number of shirts sold by Sab and Dane -/
def num_shirts : ℕ := 18

/-- The number of pairs of shoes sold -/
def num_shoes : ℕ := 6

/-- The price of each pair of shoes in dollars -/
def price_shoes : ℕ := 3

/-- The price of each shirt in dollars -/
def price_shirts : ℕ := 2

/-- The earnings of each person (Sab and Dane) in dollars -/
def earnings_per_person : ℕ := 27

theorem shirts_sold_proof : 
  num_shirts = 18 ∧ 
  num_shoes * price_shoes + num_shirts * price_shirts = 2 * earnings_per_person := by
  sorry

end shirts_sold_proof_l1167_116780


namespace negation_equivalence_l1167_116798

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 1 < 0) ↔ (∀ x : ℝ, x^2 - 1 ≥ 0) := by sorry

end negation_equivalence_l1167_116798


namespace both_selected_probability_l1167_116757

theorem both_selected_probability (ram_prob ravi_prob : ℚ) 
  (h1 : ram_prob = 2/7) 
  (h2 : ravi_prob = 1/5) : 
  ram_prob * ravi_prob = 2/35 := by
sorry

end both_selected_probability_l1167_116757


namespace lawrence_county_camp_kids_l1167_116703

/-- The number of kids going to camp during summer break in Lawrence county --/
def kids_at_camp (total_kids : ℕ) (kids_at_home : ℕ) : ℕ :=
  total_kids - kids_at_home

/-- Theorem stating the number of kids going to camp in Lawrence county --/
theorem lawrence_county_camp_kids :
  kids_at_camp 313473 274865 = 38608 := by
  sorry

end lawrence_county_camp_kids_l1167_116703


namespace megan_initial_bottles_l1167_116756

/-- The number of water bottles Megan had initially -/
def initial_bottles : ℕ := sorry

/-- The number of water bottles Megan drank -/
def bottles_drank : ℕ := 3

/-- The number of water bottles Megan had left -/
def bottles_left : ℕ := 14

theorem megan_initial_bottles : 
  initial_bottles = bottles_left + bottles_drank :=
by sorry

end megan_initial_bottles_l1167_116756


namespace quadratic_solution_difference_squared_l1167_116747

theorem quadratic_solution_difference_squared :
  ∀ d e : ℝ,
  (4 * d^2 + 8 * d - 48 = 0) →
  (4 * e^2 + 8 * e - 48 = 0) →
  d ≠ e →
  (d - e)^2 = 49 := by
sorry

end quadratic_solution_difference_squared_l1167_116747


namespace oak_trees_after_planting_total_oak_trees_after_planting_l1167_116711

/-- The number of oak trees in a park after planting new trees is equal to the sum of the initial number of trees and the number of newly planted trees. -/
theorem oak_trees_after_planting (initial_trees newly_planted_trees : ℕ) :
  initial_trees + newly_planted_trees = initial_trees + newly_planted_trees :=
by sorry

/-- The park initially has 5 oak trees. -/
def initial_oak_trees : ℕ := 5

/-- The number of oak trees to be planted is 4. -/
def oak_trees_to_plant : ℕ := 4

/-- The total number of oak trees after planting is 9. -/
theorem total_oak_trees_after_planting :
  initial_oak_trees + oak_trees_to_plant = 9 :=
by sorry

end oak_trees_after_planting_total_oak_trees_after_planting_l1167_116711


namespace greatest_integer_in_set_l1167_116726

/-- A set of consecutive even integers -/
def ConsecutiveEvenSet : Type := List Nat

/-- The median of a list of numbers -/
def median (l : List Nat) : Nat :=
  sorry

/-- Check if a list contains only even numbers -/
def allEven (l : List Nat) : Prop :=
  sorry

/-- Check if a list contains consecutive even integers -/
def isConsecutiveEven (l : List Nat) : Prop :=
  sorry

theorem greatest_integer_in_set (s : ConsecutiveEvenSet) 
  (h1 : median s = 150)
  (h2 : s.head! = 140)
  (h3 : allEven s)
  (h4 : isConsecutiveEven s) :
  s.getLast! = 152 :=
sorry

end greatest_integer_in_set_l1167_116726


namespace largest_inexpressible_number_l1167_116776

def is_expressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

theorem largest_inexpressible_number : 
  (∀ k : ℕ, k > 19 ∧ k ≤ 50 → is_expressible k) ∧
  ¬is_expressible 19 :=
sorry

end largest_inexpressible_number_l1167_116776


namespace triangle_perimeter_l1167_116727

/-- An equilateral triangle with inscribed circles and square -/
structure TriangleWithInscriptions where
  /-- Side length of the equilateral triangle -/
  side : ℝ
  /-- Radius of each inscribed circle -/
  circle_radius : ℝ
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- The circle radius is 4 -/
  h_circle_radius : circle_radius = 4
  /-- The square side is equal to the triangle side minus twice the diameter of two circles -/
  h_square_side : square_side = side - 4 * circle_radius
  /-- The triangle side is composed of two parts touching the circles and the diameter of two circles -/
  h_side : side = 2 * (circle_radius * Real.sqrt 3) + 2 * circle_radius

/-- The perimeter of the triangle is 24 + 24√3 -/
theorem triangle_perimeter (t : TriangleWithInscriptions) : 
  3 * t.side = 24 + 24 * Real.sqrt 3 := by
  sorry

end triangle_perimeter_l1167_116727


namespace min_value_quadratic_l1167_116761

theorem min_value_quadratic (x : ℝ) :
  ∃ (min_z : ℝ), min_z = 12 ∧ ∀ z : ℝ, z = 4*x^2 + 8*x + 16 → z ≥ min_z :=
by
  sorry

end min_value_quadratic_l1167_116761


namespace fencing_requirement_l1167_116720

theorem fencing_requirement (area : ℝ) (uncovered_side : ℝ) (fencing : ℝ) : 
  area = 880 →
  uncovered_side = 25 →
  fencing = uncovered_side + 2 * (area / uncovered_side) →
  fencing = 95.4 := by
sorry

end fencing_requirement_l1167_116720


namespace divisibility_implication_l1167_116705

theorem divisibility_implication (a b : ℕ) (h1 : a < 1000) (h2 : b^10 ∣ a^21) : b ∣ a^2 := by
  sorry

end divisibility_implication_l1167_116705


namespace base8_531_to_base7_l1167_116751

/-- Converts a number from base 8 to base 10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 --/
def base10ToBase7 (n : ℕ) : List ℕ := sorry

/-- Checks if a list of digits represents a valid base 7 number --/
def isValidBase7 (digits : List ℕ) : Prop :=
  digits.all (· < 7)

theorem base8_531_to_base7 :
  let base10 := base8ToBase10 531
  let base7 := base10ToBase7 base10
  isValidBase7 base7 ∧ base7 = [1, 0, 0, 2] :=
by sorry

end base8_531_to_base7_l1167_116751


namespace dot_product_constant_l1167_116738

/-- Definition of ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Definition of curve E -/
def curve_E (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Definition of point D -/
def point_D : ℝ × ℝ := (-2, 0)

/-- Definition of line passing through origin -/
def line_through_origin (k : ℝ) (x y : ℝ) : Prop := y = k * x

/-- Theorem: Dot product of DA and DB is constant -/
theorem dot_product_constant (A B : ℝ × ℝ) :
  curve_E A.1 A.2 →
  curve_E B.1 B.2 →
  (∃ k : ℝ, line_through_origin k A.1 A.2 ∧ line_through_origin k B.1 B.2) →
  ((A.1 + 2) * (B.1 + 2) + (A.2 * B.2) = 3) :=
sorry

end dot_product_constant_l1167_116738


namespace product_of_sqrt5_plus_minus_2_l1167_116729

theorem product_of_sqrt5_plus_minus_2 :
  let a := Real.sqrt 5 + 2
  let b := Real.sqrt 5 - 2
  a * b = 1 := by sorry

end product_of_sqrt5_plus_minus_2_l1167_116729


namespace geometric_arithmetic_sequence_problem_l1167_116786

theorem geometric_arithmetic_sequence_problem :
  ∀ a b c : ℝ,
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (∃ r : ℝ, b = a * r ∧ c = b * r) →
  (a * b * c = 512) →
  (2 * b = (a - 2) + (c - 2)) →
  ((a = 4 ∧ b = 8 ∧ c = 16) ∨ (a = 16 ∧ b = 8 ∧ c = 4)) :=
by sorry


end geometric_arithmetic_sequence_problem_l1167_116786


namespace quadratic_inequality_implies_m_range_l1167_116730

theorem quadratic_inequality_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - m > 0) → m < -1 := by
  sorry

end quadratic_inequality_implies_m_range_l1167_116730


namespace partner_contribution_b_contribution_is_31500_l1167_116784

/-- Given a business partnership scenario, calculate partner B's capital contribution. -/
theorem partner_contribution (a_initial : ℕ) (a_months : ℕ) (b_months : ℕ) (profit_ratio_a : ℕ) (profit_ratio_b : ℕ) : ℕ :=
  let total_months := a_months
  let b_contribution := (a_initial * total_months * profit_ratio_b) / (b_months * profit_ratio_a)
  b_contribution

/-- Prove that B's contribution is 31500 rupees given the specific scenario. -/
theorem b_contribution_is_31500 :
  partner_contribution 3500 12 2 2 3 = 31500 := by
  sorry

end partner_contribution_b_contribution_is_31500_l1167_116784


namespace perpendicular_line_through_point_l1167_116722

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the point A
def point_A : ℝ × ℝ := (2, -3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y - 8 = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  ∀ (x y : ℝ),
    (perpendicular_line x y ∧ (x, y) = point_A) →
    (∀ (x' y' : ℝ), given_line x' y' → (x - x') * (x' - x) + (y - y') * (y' - y) = 0) :=
sorry

end perpendicular_line_through_point_l1167_116722


namespace function_identity_implies_constant_relation_l1167_116765

-- Define the functions and constants
variable (f g : ℝ → ℝ)
variable (a b c : ℝ)

-- State the theorem
theorem function_identity_implies_constant_relation 
  (h : ∀ (x y : ℝ), f x * g y = a * x * y + b * x + c * y + 1) : 
  a = b * c := by sorry

end function_identity_implies_constant_relation_l1167_116765


namespace cubic_inequality_l1167_116769

theorem cubic_inequality (x : ℝ) : x^3 + 3*x^2 + x - 5 > 0 ↔ x > 1 := by sorry

end cubic_inequality_l1167_116769


namespace tan_alpha_plus_pi_4_eq_neg_3_implies_cos_2alpha_plus_2sin_2alpha_eq_1_l1167_116779

theorem tan_alpha_plus_pi_4_eq_neg_3_implies_cos_2alpha_plus_2sin_2alpha_eq_1 
  (α : ℝ) (h : Real.tan (α + π/4) = -3) : 
  Real.cos (2*α) + 2 * Real.sin (2*α) = 1 := by
  sorry

end tan_alpha_plus_pi_4_eq_neg_3_implies_cos_2alpha_plus_2sin_2alpha_eq_1_l1167_116779


namespace sound_propagation_all_directions_l1167_116792

/-- Represents the medium through which sound travels -/
inductive Medium
| Air
| Water
| Solid

/-- Represents a direction in 3D space -/
structure Direction where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents sound as a mechanical wave -/
structure Sound where
  medium : Medium
  frequency : ℝ
  amplitude : ℝ

/-- Represents the propagation of sound in a medium -/
def SoundPropagation (s : Sound) (d : Direction) : Prop :=
  match s.medium with
  | Medium.Air => true
  | Medium.Water => true
  | Medium.Solid => true

/-- Theorem stating that sound propagates in all directions in a classroom -/
theorem sound_propagation_all_directions 
  (s : Sound) 
  (h1 : s.medium = Medium.Air) 
  (h2 : ∀ (d : Direction), SoundPropagation s d) : 
  ∀ (d : Direction), SoundPropagation s d :=
sorry

end sound_propagation_all_directions_l1167_116792


namespace g_range_l1167_116794

noncomputable def g (x : ℝ) : ℝ := (Real.arccos x)^4 + (Real.arcsin x)^4

theorem g_range : 
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, 
  ∃ y ∈ Set.Icc (Real.pi^4 / 16) ((3 * Real.pi^4) / 32), 
  g x = y ∧ 
  ∀ z, g x = z → z ∈ Set.Icc (Real.pi^4 / 16) ((3 * Real.pi^4) / 32) :=
sorry

end g_range_l1167_116794


namespace ten_thousand_one_divides_same_first_last_four_digits_l1167_116791

-- Define an 8-digit number type
def EightDigitNumber := { n : ℕ // 10000000 ≤ n ∧ n < 100000000 }

-- Define the property of having the same first and last four digits
def SameFirstLastFourDigits (n : EightDigitNumber) : Prop :=
  ∃ (a b c d : ℕ), 
    0 ≤ a ∧ a < 10 ∧
    0 ≤ b ∧ b < 10 ∧
    0 ≤ c ∧ c < 10 ∧
    0 ≤ d ∧ d < 10 ∧
    n.val = 10000000 * a + 1000000 * b + 100000 * c + 10000 * d +
            1000 * a + 100 * b + 10 * c + d

-- Theorem statement
theorem ten_thousand_one_divides_same_first_last_four_digits 
  (n : EightDigitNumber) (h : SameFirstLastFourDigits n) : 
  10001 ∣ n.val :=
sorry

end ten_thousand_one_divides_same_first_last_four_digits_l1167_116791


namespace passing_percentage_is_fifty_l1167_116777

def student_score : ℕ := 200
def failed_by : ℕ := 20
def max_marks : ℕ := 440

def passing_score : ℕ := student_score + failed_by

def passing_percentage : ℚ := (passing_score : ℚ) / (max_marks : ℚ) * 100

theorem passing_percentage_is_fifty : passing_percentage = 50 := by
  sorry

end passing_percentage_is_fifty_l1167_116777


namespace hyperbola_dot_product_bound_l1167_116731

/-- The hyperbola with center at origin, left focus at (-2,0), and equation x²/a² - y² = 1 where a > 0 -/
structure Hyperbola where
  a : ℝ
  h_a_pos : a > 0

/-- A point on the right branch of the hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 = 1
  h_right_branch : x ≥ h.a

/-- The theorem stating that the dot product of OP and FP is bounded below -/
theorem hyperbola_dot_product_bound (h : Hyperbola) (p : HyperbolaPoint h) :
  p.x * (p.x + 2) + p.y * p.y ≥ 3 + 2 * Real.sqrt 3 := by sorry

end hyperbola_dot_product_bound_l1167_116731


namespace arithmetic_sequence_problem_l1167_116723

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : arithmetic_sequence a)
  (h2 : a 1 = 1/3)
  (h3 : a 2 + a 5 = 4)
  (h4 : ∃ n : ℕ, a n = 33) :
  (∃ n : ℕ, a n = 33 ∧ n = 50) ∧
  (∃ S : ℚ, S = (50 * 51) / 3 ∧ S = 850) :=
sorry

end arithmetic_sequence_problem_l1167_116723


namespace two_digit_number_swap_sum_theorem_l1167_116771

/-- Represents a two-digit number with distinct non-zero digits -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  tens_not_zero : tens ≠ 0
  units_not_zero : units ≠ 0
  distinct_digits : tens ≠ units
  is_two_digit : tens < 10 ∧ units < 10

/-- The value of a TwoDigitNumber -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- The value of a TwoDigitNumber with swapped digits -/
def TwoDigitNumber.swapped_value (n : TwoDigitNumber) : Nat :=
  10 * n.units + n.tens

theorem two_digit_number_swap_sum_theorem 
  (a b c : TwoDigitNumber) 
  (h : a.value + b.value + c.value = 41) :
  a.swapped_value + b.swapped_value + c.swapped_value = 113 := by
  sorry

end two_digit_number_swap_sum_theorem_l1167_116771


namespace marbleCombinations_eq_twelve_l1167_116799

/-- The number of ways to select 4 marbles from a set of 5 indistinguishable red marbles,
    4 indistinguishable blue marbles, and 2 indistinguishable black marbles -/
def marbleCombinations : ℕ :=
  let red := 5
  let blue := 4
  let black := 2
  let totalSelect := 4
  (Finset.filter (fun t : ℕ × ℕ × ℕ => 
    t.1 + t.2.1 + t.2.2 = totalSelect ∧ 
    t.1 ≤ red ∧ 
    t.2.1 ≤ blue ∧ 
    t.2.2 ≤ black
  ) (Finset.product (Finset.range (red + 1)) (Finset.product (Finset.range (blue + 1)) (Finset.range (black + 1))))).card

theorem marbleCombinations_eq_twelve : marbleCombinations = 12 := by
  sorry

end marbleCombinations_eq_twelve_l1167_116799


namespace reciprocal_of_negative_half_l1167_116712

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_negative_half : reciprocal (-1/2) = -2 := by
  sorry

end reciprocal_of_negative_half_l1167_116712


namespace unique_prime_double_squares_l1167_116775

theorem unique_prime_double_squares : 
  ∃! (p : ℕ), 
    Prime p ∧ 
    (∃ (x : ℕ), p + 7 = 2 * x^2) ∧ 
    (∃ (y : ℕ), p^2 + 7 = 2 * y^2) ∧ 
    p = 11 := by
  sorry

end unique_prime_double_squares_l1167_116775


namespace zeros_before_first_nonzero_digit_l1167_116740

theorem zeros_before_first_nonzero_digit (n : ℕ) (d : ℕ) (h : n = 5 ∧ d = 1600) :
  (n : ℚ) / d = 0.003125 :=
sorry

end zeros_before_first_nonzero_digit_l1167_116740


namespace polygon_sides_l1167_116724

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 = 3 * 360) → 
  n = 8 := by
sorry

end polygon_sides_l1167_116724


namespace remaining_potatoes_l1167_116718

def initial_potatoes : ℕ := 8
def eaten_potatoes : ℕ := 3

theorem remaining_potatoes : initial_potatoes - eaten_potatoes = 5 := by
  sorry

end remaining_potatoes_l1167_116718


namespace student_guinea_pig_difference_is_126_l1167_116762

/-- The number of students in each fourth-grade classroom -/
def students_per_classroom : ℕ := 24

/-- The number of guinea pigs in each fourth-grade classroom -/
def guinea_pigs_per_classroom : ℕ := 3

/-- The number of fourth-grade classrooms -/
def number_of_classrooms : ℕ := 6

/-- The difference between the total number of students and the total number of guinea pigs -/
def student_guinea_pig_difference : ℕ :=
  (students_per_classroom * number_of_classrooms) - (guinea_pigs_per_classroom * number_of_classrooms)

theorem student_guinea_pig_difference_is_126 :
  student_guinea_pig_difference = 126 := by
  sorry

end student_guinea_pig_difference_is_126_l1167_116762


namespace crazy_silly_school_theorem_l1167_116715

/-- Represents the 'Crazy Silly School' series collection --/
structure CrazySillyCollection where
  books : ℕ
  movies : ℕ
  videoGames : ℕ
  audiobooks : ℕ

/-- Represents the completed items in the collection --/
structure CompletedItems where
  booksRead : ℕ
  moviesWatched : ℕ
  gamesPlayed : ℕ
  audiobooksListened : ℕ
  halfReadBooks : ℕ
  halfWatchedMovies : ℕ

/-- Calculates the portions left to complete in the collection --/
def portionsLeftToComplete (collection : CrazySillyCollection) (completed : CompletedItems) : ℚ :=
  let totalPortions := collection.books + collection.movies + collection.videoGames + collection.audiobooks
  let completedPortions := completed.booksRead - completed.halfReadBooks / 2 +
                           completed.moviesWatched - completed.halfWatchedMovies / 2 +
                           completed.gamesPlayed +
                           completed.audiobooksListened
  totalPortions - completedPortions

/-- Theorem stating the number of portions left to complete in the 'Crazy Silly School' series --/
theorem crazy_silly_school_theorem (collection : CrazySillyCollection) (completed : CompletedItems) :
  collection.books = 22 ∧
  collection.movies = 10 ∧
  collection.videoGames = 8 ∧
  collection.audiobooks = 15 ∧
  completed.booksRead = 12 ∧
  completed.moviesWatched = 6 ∧
  completed.gamesPlayed = 3 ∧
  completed.audiobooksListened = 7 ∧
  completed.halfReadBooks = 2 ∧
  completed.halfWatchedMovies = 1 →
  portionsLeftToComplete collection completed = 28.5 := by
  sorry

end crazy_silly_school_theorem_l1167_116715


namespace work_completion_time_l1167_116737

/-- The time taken for three workers to complete a task together,
    given their individual completion times. -/
theorem work_completion_time
  (x_time y_time z_time : ℝ)
  (hx : x_time = 10)
  (hy : y_time = 15)
  (hz : z_time = 20)
  : (1 : ℝ) / ((1 / x_time) + (1 / y_time) + (1 / z_time)) = 60 / 13 := by
  sorry

end work_completion_time_l1167_116737


namespace extreme_value_and_intersection_l1167_116787

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (a - Real.log x) / x

def g (x : ℝ) : ℝ := -1

theorem extreme_value_and_intersection (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ x ≤ Real.exp 1 ∧ f a x = g x) →
  (∀ (x : ℝ), x > 0 → f a x ≥ -Real.exp (-a - 1)) ∧
  (f a (Real.exp (a + 1)) = -Real.exp (-a - 1)) ∧
  (a ≤ -1 ∨ (0 ≤ a ∧ a ≤ Real.exp 1)) :=
sorry

end extreme_value_and_intersection_l1167_116787


namespace unique_pen_distribution_l1167_116734

/-- Represents a distribution of pens among students -/
structure PenDistribution where
  num_students : ℕ
  pens_per_student : ℕ → ℕ
  total_pens : ℕ

/-- The condition that among any four pens, at least two belong to the same person -/
def four_pens_condition (d : PenDistribution) : Prop :=
  ∀ (s : Finset ℕ), s.card = 4 → ∃ i ∈ s, d.pens_per_student i ≥ 2

/-- The condition that among any five pens, no more than three belong to the same person -/
def five_pens_condition (d : PenDistribution) : Prop :=
  ∀ (s : Finset ℕ), s.card = 5 → ∀ i ∈ s, d.pens_per_student i ≤ 3

/-- The theorem stating the unique distribution satisfying the given conditions -/
theorem unique_pen_distribution :
  ∀ (d : PenDistribution),
    d.total_pens = 9 →
    four_pens_condition d →
    five_pens_condition d →
    d.num_students = 3 ∧ (∀ i, i < d.num_students → d.pens_per_student i = 3) :=
by sorry

end unique_pen_distribution_l1167_116734


namespace chocolate_bars_per_box_l1167_116719

theorem chocolate_bars_per_box (total_bars : ℕ) (total_boxes : ℕ) 
  (h1 : total_bars = 849) (h2 : total_boxes = 170) :
  total_bars / total_boxes = 5 := by
  sorry

end chocolate_bars_per_box_l1167_116719


namespace oxygen_weight_in_N2O_l1167_116701

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of nitrogen atoms in N2O -/
def num_N : ℕ := 2

/-- The number of oxygen atoms in N2O -/
def num_O : ℕ := 1

/-- The molecular weight of the oxygen part in N2O -/
def molecular_weight_O_part : ℝ := num_O * atomic_weight_O

theorem oxygen_weight_in_N2O : 
  molecular_weight_O_part = 16.00 := by sorry

end oxygen_weight_in_N2O_l1167_116701


namespace solution_set_for_m_equals_one_m_range_for_inequality_l1167_116772

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + m| + |2*x - 1|

-- Theorem 1
theorem solution_set_for_m_equals_one (x : ℝ) :
  f 1 x ≥ 3 ↔ x ≤ -1 ∨ x ≥ 1 := by sorry

-- Theorem 2
theorem m_range_for_inequality (m : ℝ) (h1 : m > 0) :
  (∀ x ∈ Set.Icc m (2*m^2), (1/2) * f m x ≤ |x + 1|) →
  1/2 < m ∧ m ≤ 1 := by sorry

end solution_set_for_m_equals_one_m_range_for_inequality_l1167_116772


namespace A_value_l1167_116745

/-- Rounds a natural number down to the nearest tens -/
def round_down_to_tens (n : ℕ) : ℕ :=
  (n / 10) * 10

/-- Given a natural number n = A567 where A is unknown, 
    if n rounds down to 2560, then A = 2 -/
theorem A_value (n : ℕ) (h : round_down_to_tens n = 2560) : 
  n / 1000 = 2 := by sorry

end A_value_l1167_116745


namespace area_ratio_of_similar_triangles_l1167_116768

-- Define two triangles
variable (T1 T2 : Set (ℝ × ℝ))

-- Define similarity ratio
variable (k : ℝ)

-- Define the property of similarity
def are_similar (T1 T2 : Set (ℝ × ℝ)) (k : ℝ) : Prop := sorry

-- Define the area of a triangle
def area (T : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_ratio_of_similar_triangles 
  (h_similar : are_similar T1 T2 k) 
  (h_k_pos : k > 0) :
  area T2 / area T1 = k^2 := sorry

end area_ratio_of_similar_triangles_l1167_116768


namespace annual_insurance_payment_l1167_116789

/-- The number of quarters in a year -/
def quarters_per_year : ℕ := 4

/-- The quarterly insurance payment in dollars -/
def quarterly_payment : ℕ := 378

/-- The annual insurance payment in dollars -/
def annual_payment : ℕ := quarterly_payment * quarters_per_year

theorem annual_insurance_payment :
  annual_payment = 1512 :=
by sorry

end annual_insurance_payment_l1167_116789
