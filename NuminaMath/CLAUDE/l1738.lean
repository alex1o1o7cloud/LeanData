import Mathlib

namespace simplify_t_l1738_173828

theorem simplify_t (t : ℝ) : t = 1 / (3 - Real.rpow 3 (1/3)) → t = (3 + Real.rpow 3 (1/3)) / 6 := by
  sorry

end simplify_t_l1738_173828


namespace book_pages_difference_l1738_173821

theorem book_pages_difference : 
  let purple_books : ℕ := 8
  let orange_books : ℕ := 7
  let blue_books : ℕ := 5
  let purple_pages_per_book : ℕ := 320
  let orange_pages_per_book : ℕ := 640
  let blue_pages_per_book : ℕ := 450
  let total_purple_pages := purple_books * purple_pages_per_book
  let total_orange_pages := orange_books * orange_pages_per_book
  let total_blue_pages := blue_books * blue_pages_per_book
  let total_orange_blue_pages := total_orange_pages + total_blue_pages
  total_orange_blue_pages - total_purple_pages = 4170 := by
sorry

end book_pages_difference_l1738_173821


namespace min_sum_of_product_2310_l1738_173853

theorem min_sum_of_product_2310 (a b c : ℕ+) : 
  a * b * c = 2310 → (∀ x y z : ℕ+, x * y * z = 2310 → a + b + c ≤ x + y + z) → a + b + c = 40 :=
sorry

end min_sum_of_product_2310_l1738_173853


namespace sqrt_x_minus_6_meaningful_l1738_173825

theorem sqrt_x_minus_6_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 6) ↔ x ≥ 6 := by sorry

end sqrt_x_minus_6_meaningful_l1738_173825


namespace intersection_A_B_l1738_173858

open Set

def A : Set ℝ := {x | (x - 2) / x ≤ 0 ∧ x ≠ 0}
def B : Set ℝ := Icc (-1 : ℝ) 1

theorem intersection_A_B : A ∩ B = Ioc 0 1 := by
  sorry

end intersection_A_B_l1738_173858


namespace hamburgers_left_over_l1738_173802

theorem hamburgers_left_over (total : ℕ) (served : ℕ) (left_over : ℕ) : 
  total = 9 → served = 3 → left_over = total - served → left_over = 6 := by
sorry

end hamburgers_left_over_l1738_173802


namespace janice_age_problem_l1738_173818

theorem janice_age_problem :
  ∀ x : ℕ,
  (x + 12 = 8 * (x - 2)) → x = 4 :=
by
  sorry

end janice_age_problem_l1738_173818


namespace uncle_fyodor_cannot_always_win_l1738_173888

/-- Represents a sandwich with sausage and cheese -/
structure Sandwich :=
  (hasSausage : Bool)

/-- Represents the state of the game -/
structure GameState :=
  (sandwiches : List Sandwich)
  (turn : Nat)

/-- Uncle Fyodor's move: eat one sandwich from either end -/
def uncleFyodorMove (state : GameState) : GameState :=
  sorry

/-- Matroskin's move: remove sausage from one sandwich or do nothing -/
def matroskinMove (state : GameState) : GameState :=
  sorry

/-- Play the game until all sandwiches are eaten -/
def playGame (initialState : GameState) : GameState :=
  sorry

/-- Check if Uncle Fyodor wins (last sandwich eaten contains sausage) -/
def uncleFyodorWins (finalState : GameState) : Bool :=
  sorry

/-- Theorem: There exists a natural number N for which Uncle Fyodor cannot guarantee a win -/
theorem uncle_fyodor_cannot_always_win :
  ∃ N : Nat, ∀ uncleFyodorStrategy : GameState → GameState,
    ∃ matroskinStrategy : GameState → GameState,
      let initialState := GameState.mk (List.replicate N (Sandwich.mk true)) 0
      ¬(uncleFyodorWins (playGame initialState)) :=
by
  sorry

end uncle_fyodor_cannot_always_win_l1738_173888


namespace stating_max_bulbs_on_theorem_l1738_173889

/-- Represents the maximum number of bulbs that can be turned on in an n × n grid -/
def maxBulbsOn (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n^2 / 2
  else
    (n^2 - 1) / 2

/-- 
Theorem stating the maximum number of bulbs that can be turned on in an n × n grid,
given the constraints of the problem.
-/
theorem max_bulbs_on_theorem (n : ℕ) :
  ∀ (pressed : Finset (ℕ × ℕ)),
    (∀ (i j : ℕ), (i, j) ∈ pressed → i < n ∧ j < n) →
    (∀ (i j k l : ℕ), (i, j) ∈ pressed → (k, l) ∈ pressed → (i = k ∨ j = l) → i = k ∧ j = l) →
    (∃ (final_state : Finset (ℕ × ℕ)),
      (∀ (i j : ℕ), (i, j) ∈ final_state → i < n ∧ j < n) ∧
      final_state.card ≤ maxBulbsOn n) :=
by
  sorry

#check max_bulbs_on_theorem

end stating_max_bulbs_on_theorem_l1738_173889


namespace smallest_multiples_of_17_l1738_173827

theorem smallest_multiples_of_17 :
  (∃ n : ℕ, n * 17 = 34 ∧ ∀ m : ℕ, m * 17 ≥ 10 ∧ m * 17 < 100 → m * 17 ≥ 34) ∧
  (∃ n : ℕ, n * 17 = 1003 ∧ ∀ m : ℕ, m * 17 ≥ 1000 ∧ m * 17 < 10000 → m * 17 ≥ 1003) :=
by sorry

end smallest_multiples_of_17_l1738_173827


namespace rationalize_denominator_l1738_173822

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 6 + Real.sqrt 5) =
  6 * Real.sqrt 2 - 2 * Real.sqrt 15 + Real.sqrt 30 - 5 := by
  sorry

end rationalize_denominator_l1738_173822


namespace hyperbola_solution_is_three_halves_l1738_173872

/-- The set of all real numbers m that satisfy the conditions of the hyperbola problem -/
def hyperbola_solution : Set ℝ :=
  {m : ℝ | m > 0 ∧ 2 * m^2 + 3 * m = 9}

/-- The theorem stating that the solution set contains only 3/2 -/
theorem hyperbola_solution_is_three_halves : hyperbola_solution = {3/2} := by
  sorry

end hyperbola_solution_is_three_halves_l1738_173872


namespace simplify_trig_expression_l1738_173806

theorem simplify_trig_expression (α : ℝ) :
  2 * Real.sin α * Real.cos α * (Real.cos α ^ 2 - Real.sin α ^ 2) = (1/2) * Real.sin (4 * α) := by
  sorry

end simplify_trig_expression_l1738_173806


namespace cycle_loss_percentage_l1738_173883

/-- Calculates the percentage of loss given the cost price and selling price -/
def percentage_loss (cost_price selling_price : ℚ) : ℚ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Theorem: The percentage of loss for a cycle with cost price 1200 and selling price 1020 is 15% -/
theorem cycle_loss_percentage :
  let cost_price : ℚ := 1200
  let selling_price : ℚ := 1020
  percentage_loss cost_price selling_price = 15 := by
  sorry

#eval percentage_loss 1200 1020

end cycle_loss_percentage_l1738_173883


namespace remainder_sum_divided_by_11_l1738_173824

theorem remainder_sum_divided_by_11 : 
  (2 * (8735 + 8736 + 8737 + 8738 + 8739)) % 11 = 8 := by
  sorry

end remainder_sum_divided_by_11_l1738_173824


namespace n_is_even_l1738_173851

-- Define a type for points in space
def Point : Type := ℝ × ℝ × ℝ

-- Define a function to check if four points are coplanar
def are_coplanar (p q r s : Point) : Prop := sorry

-- Define a function to check if a point is inside a tetrahedron
def is_interior_point (p q r s t : Point) : Prop := sorry

-- Define the main theorem
theorem n_is_even (n : ℕ) (P : Fin n → Point) (Q : Point) :
  (∀ (i j k l : Fin n), i ≠ j → j ≠ k → k ≠ l → i ≠ k → j ≠ l → i ≠ l → 
    ¬ are_coplanar (P i) (P j) (P k) (P l)) →
  (∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → 
    ∃ (l : Fin n), l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ 
      is_interior_point Q (P i) (P j) (P k) (P l)) →
  Even n := by
  sorry

end n_is_even_l1738_173851


namespace sin_power_five_expansion_l1738_173840

theorem sin_power_five_expansion (b₁ b₂ b₃ b₄ b₅ : ℝ) : 
  (∀ θ : ℝ, Real.sin θ ^ 5 = b₁ * Real.sin θ + b₂ * Real.sin (2 * θ) + 
    b₃ * Real.sin (3 * θ) + b₄ * Real.sin (4 * θ) + b₅ * Real.sin (5 * θ)) → 
  b₁ ^ 2 + b₂ ^ 2 + b₃ ^ 2 + b₄ ^ 2 + b₅ ^ 2 = 63 / 128 := by
  sorry

end sin_power_five_expansion_l1738_173840


namespace box_counting_l1738_173866

theorem box_counting (initial_boxes : ℕ) (boxes_per_operation : ℕ) (final_nonempty_boxes : ℕ) :
  initial_boxes = 2013 →
  boxes_per_operation = 13 →
  final_nonempty_boxes = 2013 →
  initial_boxes + boxes_per_operation * final_nonempty_boxes = 28182 := by
  sorry

#check box_counting

end box_counting_l1738_173866


namespace average_difference_due_to_input_error_l1738_173894

theorem average_difference_due_to_input_error :
  ∀ (data_points : ℕ) (incorrect_value : ℝ) (correct_value : ℝ),
    data_points = 30 →
    incorrect_value = 105 →
    correct_value = 15 →
    (incorrect_value - correct_value) / data_points = 3 :=
by
  sorry

end average_difference_due_to_input_error_l1738_173894


namespace quadratic_equation_solution_l1738_173820

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = 1/3 ∧ 
  (∀ x : ℝ, 3*x^2 - 4*x + 1 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end quadratic_equation_solution_l1738_173820


namespace defective_units_shipped_l1738_173884

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ)
  (h1 : defective_rate = 0.04)
  (h2 : shipped_rate = 0.04) :
  (defective_rate * shipped_rate * total_units) / total_units = 0.0016 := by
  sorry

end defective_units_shipped_l1738_173884


namespace alice_pears_l1738_173809

/-- The number of pears Alice sold -/
def sold : ℕ := sorry

/-- The number of pears Alice poached -/
def poached : ℕ := sorry

/-- The number of pears Alice canned -/
def canned : ℕ := sorry

/-- The total number of pears -/
def total : ℕ := 42

theorem alice_pears :
  (canned = poached + poached / 5) ∧
  (poached = sold / 2) ∧
  (sold + poached + canned = total) →
  sold = 20 := by sorry

end alice_pears_l1738_173809


namespace range_of_a_minus_b_l1738_173856

theorem range_of_a_minus_b (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 2) :
  ∀ x, -3 < x ∧ x < 0 ↔ ∃ a b, -1 < a ∧ a < b ∧ b < 2 ∧ x = a - b :=
by sorry

end range_of_a_minus_b_l1738_173856


namespace equation_solution_l1738_173873

theorem equation_solution :
  ∃ y : ℚ, (2 * y + 3 * y = 500 - (4 * y + 5 * y)) ∧ (y = 250 / 7) := by
  sorry

end equation_solution_l1738_173873


namespace negation_of_universal_nonnegative_l1738_173832

theorem negation_of_universal_nonnegative :
  (¬ ∀ x : ℝ, x ≥ 0) ↔ (∃ x : ℝ, x < 0) := by sorry

end negation_of_universal_nonnegative_l1738_173832


namespace total_cost_theorem_l1738_173878

/-- Represents the cost of utensils in Moneda -/
structure UtensilCost where
  teaspoon : ℕ
  tablespoon : ℕ
  dessertSpoon : ℕ

/-- Represents the number of utensils Clara has -/
structure UtensilCount where
  teaspoon : ℕ
  tablespoon : ℕ
  dessertSpoon : ℕ

/-- Calculates the total cost of exchanged utensils and souvenirs in euros -/
def totalCostInEuros (costs : UtensilCost) (counts : UtensilCount) 
  (monedaToEuro : ℚ) (souvenirCostDollars : ℕ) (euroToDollar : ℚ) : ℚ :=
  sorry

/-- Theorem stating the total cost in euros -/
theorem total_cost_theorem (costs : UtensilCost) (counts : UtensilCount) 
  (monedaToEuro : ℚ) (souvenirCostDollars : ℕ) (euroToDollar : ℚ) :
  costs.teaspoon = 9 ∧ costs.tablespoon = 12 ∧ costs.dessertSpoon = 18 ∧
  counts.teaspoon = 7 ∧ counts.tablespoon = 10 ∧ counts.dessertSpoon = 12 ∧
  monedaToEuro = 0.04 ∧ souvenirCostDollars = 40 ∧ euroToDollar = 1.15 →
  totalCostInEuros costs counts monedaToEuro souvenirCostDollars euroToDollar = 50.74 :=
by sorry

end total_cost_theorem_l1738_173878


namespace combined_population_l1738_173892

def wellington_population : ℕ := 900

def port_perry_population : ℕ := 7 * wellington_population

def lazy_harbor_population : ℕ := 2 * wellington_population + 600

def newbridge_population : ℕ := 3 * (port_perry_population - wellington_population)

theorem combined_population :
  port_perry_population + lazy_harbor_population + newbridge_population = 24900 := by
  sorry

end combined_population_l1738_173892


namespace inequality_range_l1738_173815

theorem inequality_range (a : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  ∀ x : ℝ, x^2 + a*x > 4*x + a - 3 ↔ x < -1 ∨ x > 3 := by
sorry

end inequality_range_l1738_173815


namespace zoo_bus_distribution_l1738_173864

theorem zoo_bus_distribution (total_people : ℕ) (num_buses : ℕ) 
  (h1 : total_people = 219) (h2 : num_buses = 3) :
  total_people / num_buses = 73 := by
  sorry

end zoo_bus_distribution_l1738_173864


namespace arctan_sum_of_cubic_roots_l1738_173850

theorem arctan_sum_of_cubic_roots (x₁ x₂ x₃ : ℝ) : 
  x₁^3 - 10*x₁ + 11 = 0 →
  x₂^3 - 10*x₂ + 11 = 0 →
  x₃^3 - 10*x₃ + 11 = 0 →
  -5 < x₁ ∧ x₁ < 5 →
  -5 < x₂ ∧ x₂ < 5 →
  -5 < x₃ ∧ x₃ < 5 →
  Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = π/4 := by
sorry

end arctan_sum_of_cubic_roots_l1738_173850


namespace sugar_recipe_reduction_sugar_mixed_number_l1738_173846

theorem sugar_recipe_reduction : 
  let original_sugar : ℚ := 31/4
  let reduced_sugar : ℚ := (1/3) * original_sugar
  reduced_sugar = 31/12 := by sorry

theorem sugar_mixed_number :
  let reduced_sugar : ℚ := 31/12
  ∃ (whole : ℕ) (numerator : ℕ) (denominator : ℕ),
    reduced_sugar = whole + (numerator : ℚ) / denominator ∧
    whole = 2 ∧ numerator = 7 ∧ denominator = 12 := by sorry

end sugar_recipe_reduction_sugar_mixed_number_l1738_173846


namespace original_number_is_ten_l1738_173843

theorem original_number_is_ten : ∃ x : ℝ, 3 * (2 * x + 8) = 84 ∧ x = 10 := by
  sorry

end original_number_is_ten_l1738_173843


namespace line_slope_intercept_sum_l1738_173886

/-- Given a line passing through points (1, 3) and (-3, -1), 
    prove that the sum of its slope and y-intercept is 3. -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℝ), 
  (3 = m * 1 + b) →  -- Point (1, 3) satisfies the line equation
  (-1 = m * (-3) + b) →  -- Point (-3, -1) satisfies the line equation
  m + b = 3 := by
  sorry

end line_slope_intercept_sum_l1738_173886


namespace a_gt_b_neither_necessary_nor_sufficient_for_a_sq_gt_b_sq_l1738_173823

theorem a_gt_b_neither_necessary_nor_sufficient_for_a_sq_gt_b_sq :
  ¬(∀ a b : ℝ, a > b → a^2 > b^2) ∧ 
  ¬(∀ a b : ℝ, a^2 > b^2 → a > b) := by
  sorry

end a_gt_b_neither_necessary_nor_sufficient_for_a_sq_gt_b_sq_l1738_173823


namespace initial_milk_water_ratio_l1738_173838

theorem initial_milk_water_ratio 
  (M W : ℝ) 
  (h1 : M + W = 45) 
  (h2 : M / (W + 18) = 4/3) : 
  M / W = 4 := by
sorry

end initial_milk_water_ratio_l1738_173838


namespace function_properties_l1738_173847

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 10

theorem function_properties (m : ℝ) (h1 : m > 1) (h2 : f m m = 1) :
  ∃ (g : ℝ → ℝ),
    (∀ x, g x = x^2 - 6*x + 10) ∧
    (∀ x ∈ Set.Icc 3 5, g x ≤ 5) ∧
    (∀ x ∈ Set.Icc 3 5, g x ≥ 1) ∧
    (∃ x ∈ Set.Icc 3 5, g x = 5) ∧
    (∃ x ∈ Set.Icc 3 5, g x = 1) :=
by sorry

end function_properties_l1738_173847


namespace total_stamps_l1738_173834

theorem total_stamps (x y : ℕ) (hx : x = 34) (hy : y = x + 44) : x + y = 112 := by
  sorry

end total_stamps_l1738_173834


namespace division_problem_l1738_173861

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 222 →
  quotient = 17 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 13 := by sorry

end division_problem_l1738_173861


namespace abs_neg_five_equals_five_l1738_173877

theorem abs_neg_five_equals_five : |(-5 : ℤ)| = 5 := by
  sorry

end abs_neg_five_equals_five_l1738_173877


namespace workshop_workers_count_l1738_173887

/-- Proves that the total number of workers in a workshop is 49 given specific salary conditions. -/
theorem workshop_workers_count :
  let average_salary : ℕ := 8000
  let technician_salary : ℕ := 20000
  let other_salary : ℕ := 6000
  let technician_count : ℕ := 7
  ∃ (total_workers : ℕ) (other_workers : ℕ),
    total_workers = technician_count + other_workers ∧
    total_workers * average_salary = technician_count * technician_salary + other_workers * other_salary ∧
    total_workers = 49 := by
  sorry

#check workshop_workers_count

end workshop_workers_count_l1738_173887


namespace more_silver_than_gold_fish_l1738_173837

theorem more_silver_than_gold_fish (x g s r : ℕ) : 
  x = g + s + r →
  x - g = (2 * x) / 3 - 1 →
  x - r = (2 * x) / 3 + 4 →
  s = g + 2 := by
sorry

end more_silver_than_gold_fish_l1738_173837


namespace homework_time_difference_l1738_173897

/-- Proves that the difference in time taken by Sarah and Samuel to finish their homework is 48 minutes -/
theorem homework_time_difference (samuel_time sarah_time_hours : ℝ) : 
  samuel_time = 30 → 
  sarah_time_hours = 1.3 → 
  sarah_time_hours * 60 - samuel_time = 48 := by
sorry

end homework_time_difference_l1738_173897


namespace vessel_base_length_vessel_problem_solution_l1738_173899

/-- Given a cube immersed in a rectangular vessel, calculates the length of the vessel's base. -/
theorem vessel_base_length (cube_edge : ℝ) (vessel_width : ℝ) (water_rise : ℝ) : ℝ :=
  let cube_volume := cube_edge^3
  let vessel_length := cube_volume / (vessel_width * water_rise)
  vessel_length

/-- Proves that for a 15 cm cube in a vessel of width 15 cm causing 11.25 cm water rise, 
    the vessel's base length is 20 cm. -/
theorem vessel_problem_solution : 
  vessel_base_length 15 15 11.25 = 20 := by
  sorry

end vessel_base_length_vessel_problem_solution_l1738_173899


namespace function_inequality_l1738_173845

noncomputable def f (x : ℝ) := x^2 - Real.pi * x

theorem function_inequality (α β γ : ℝ) 
  (h_α : α ∈ Set.Ioo 0 Real.pi) 
  (h_β : β ∈ Set.Ioo 0 Real.pi) 
  (h_γ : γ ∈ Set.Ioo 0 Real.pi)
  (h_sin_α : Real.sin α = 1/3)
  (h_tan_β : Real.tan β = 5/4)
  (h_cos_γ : Real.cos γ = -1/3) :
  f α > f β ∧ f β > f γ := by
  sorry

end function_inequality_l1738_173845


namespace product_prices_and_min_units_l1738_173808

/-- Represents the unit price of product A in yuan -/
def price_A : ℝ := sorry

/-- Represents the unit price of product B in yuan -/
def price_B : ℝ := sorry

/-- Represents the total number of units produced (in thousands) -/
def total_units : ℕ := 80

/-- Represents the relationship between the sales revenue of A and B -/
axiom revenue_relation : 2 * price_A = 3 * price_B

/-- Represents the difference in sales revenue between A and B -/
axiom revenue_difference : 3 * price_A - 2 * price_B = 1500

/-- Represents the minimum number of units of A to be sold (in thousands) -/
def min_units_A : ℕ := sorry

/-- Theorem stating the unit prices of A and B, and the minimum units of A to be sold -/
theorem product_prices_and_min_units : 
  price_A = 900 ∧ price_B = 600 ∧ 
  (∀ m : ℕ, m ≥ min_units_A → 
    900 * m + 600 * (total_units - m) ≥ 54000) ∧
  min_units_A = 2 := by sorry

end product_prices_and_min_units_l1738_173808


namespace negation_of_universal_proposition_l1738_173800

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + 1 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l1738_173800


namespace parallel_vectors_m_l1738_173816

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given two vectors a and b, where a = (m, 4) and b = (3, -2),
    if a is parallel to b, then m = -6 -/
theorem parallel_vectors_m (m : ℝ) :
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (3, -2)
  parallel a b → m = -6 := by
sorry

end parallel_vectors_m_l1738_173816


namespace age_difference_l1738_173805

/-- Proves the age difference between a man and his son --/
theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 22 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 24 := by
  sorry

end age_difference_l1738_173805


namespace negation_of_implication_l1738_173817

theorem negation_of_implication (a : ℝ) :
  ¬(a > 1 → a^2 > 1) ↔ (a ≤ 1 → a^2 ≤ 1) := by sorry

end negation_of_implication_l1738_173817


namespace sum_of_valid_starting_numbers_l1738_173879

def machine_rule (n : ℕ) : ℕ :=
  if n % 2 = 0 then n + 5 else 2 * n

def iterate_machine (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => machine_rule (iterate_machine n k)

def valid_starting_numbers : List ℕ :=
  (List.range 55).filter (λ n => iterate_machine n 4 = 54)

theorem sum_of_valid_starting_numbers :
  valid_starting_numbers.sum = 39 :=
sorry

end sum_of_valid_starting_numbers_l1738_173879


namespace min_value_of_function_l1738_173819

theorem min_value_of_function (x : ℝ) (h : x > 2) : 
  4 / (x - 2) + x ≥ 6 ∧ ∃ y > 2, 4 / (y - 2) + y = 6 := by
  sorry

end min_value_of_function_l1738_173819


namespace function_inequality_l1738_173848

theorem function_inequality (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f)
  (h_pos : ∀ x, f x > 0)
  (h_ineq : ∀ x, f x < x * deriv f x) :
  2 * f 1 < f 2 := by
  sorry

end function_inequality_l1738_173848


namespace count_integer_lengths_for_specific_triangle_l1738_173813

-- Define the right triangle DEF
structure RightTriangle where
  DE : ℝ
  EF : ℝ
  right_angle : DE > 0 ∧ EF > 0

-- Define the function to count integer lengths
def count_integer_lengths (t : RightTriangle) : ℕ :=
  sorry

-- Theorem statement
theorem count_integer_lengths_for_specific_triangle :
  ∃ (t : RightTriangle), t.DE = 24 ∧ t.EF = 25 ∧ count_integer_lengths t = 14 :=
sorry

end count_integer_lengths_for_specific_triangle_l1738_173813


namespace correct_departure_time_l1738_173814

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Calculates the difference between two times in minutes -/
def timeDiffMinutes (t1 t2 : Time) : Int :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

/-- The performance start time -/
def performanceTime : Time := { hours := 8, minutes := 30 }

/-- The travel time in minutes -/
def travelTime : Nat := 20

/-- The latest departure time -/
def latestDepartureTime : Time := { hours := 8, minutes := 10 }

theorem correct_departure_time :
  timeDiffMinutes performanceTime latestDepartureTime = travelTime := by
  sorry

end correct_departure_time_l1738_173814


namespace puzzle_ratio_is_three_to_one_l1738_173885

/-- Given a total puzzle-solving time, warm-up time, and number of additional puzzles,
    calculates the ratio of time spent on each additional puzzle to the warm-up time. -/
def puzzle_time_ratio (total_time warm_up_time : ℕ) (num_puzzles : ℕ) : ℚ :=
  let remaining_time := total_time - warm_up_time
  let time_per_puzzle := remaining_time / num_puzzles
  (time_per_puzzle : ℚ) / warm_up_time

/-- Proves that for the given conditions, the ratio of time spent on each additional puzzle
    to the warm-up puzzle is 3:1. -/
theorem puzzle_ratio_is_three_to_one :
  puzzle_time_ratio 70 10 2 = 3 / 1 := by
  sorry

end puzzle_ratio_is_three_to_one_l1738_173885


namespace union_of_A_and_B_l1738_173857

-- Define the sets A and B
def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 3 ∨ x = 4} := by
  sorry

end union_of_A_and_B_l1738_173857


namespace pants_bought_with_tshirts_l1738_173893

/-- Given the price relationships of pants and t-shirts, prove that 1 pant was bought with 6 t-shirts -/
theorem pants_bought_with_tshirts (x : ℚ) :
  (∃ (p t : ℚ), p > 0 ∧ t > 0 ∧ 
    x * p + 6 * t = 750 ∧
    p + 12 * t = 750 ∧
    8 * t = 400) →
  x = 1 := by
sorry

end pants_bought_with_tshirts_l1738_173893


namespace f_value_at_5_l1738_173811

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

-- State the theorem
theorem f_value_at_5 (a b c m : ℝ) :
  f a b c (-5) = m → f a b c 5 = -m + 4 := by
  sorry

end f_value_at_5_l1738_173811


namespace four_roots_condition_l1738_173869

/-- If the equation x^2 - 4|x| + 5 = m has four distinct real roots, then 1 < m < 5 -/
theorem four_roots_condition (m : ℝ) : 
  (∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x : ℝ, x^2 - 4*|x| + 5 = m ↔ (x = a ∨ x = b ∨ x = c ∨ x = d))) →
  1 < m ∧ m < 5 := by
sorry


end four_roots_condition_l1738_173869


namespace island_population_l1738_173875

theorem island_population (centipedes humans sheep : ℕ) : 
  centipedes = 100 →
  centipedes = 2 * humans →
  sheep = humans / 2 →
  sheep + humans = 75 := by
sorry

end island_population_l1738_173875


namespace M_intersect_N_eq_open_interval_l1738_173849

-- Define the sets M and N
def M : Set ℝ := {x | (x + 2) * (x - 1) < 0}
def N : Set ℝ := {x | x + 1 < 0}

-- State the theorem
theorem M_intersect_N_eq_open_interval : M ∩ N = {x | -2 < x ∧ x < -1} := by sorry

end M_intersect_N_eq_open_interval_l1738_173849


namespace division_problem_l1738_173854

theorem division_problem (x : ℝ) (h : x = 1) : 4 / (1 + 3/x) = 1 := by
  sorry

end division_problem_l1738_173854


namespace distance_to_reflection_over_x_axis_distance_B_to_B_l1738_173876

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection_over_x_axis (x y : ℝ) :
  Real.sqrt ((x - x)^2 + ((-y) - y)^2) = 2 * abs y := by sorry

/-- The specific case for point B(1, 4) --/
theorem distance_B_to_B'_is_8 :
  Real.sqrt ((1 - 1)^2 + ((-4) - 4)^2) = 8 := by sorry

end distance_to_reflection_over_x_axis_distance_B_to_B_l1738_173876


namespace xy_equals_two_l1738_173839

theorem xy_equals_two (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)
  (h : x^2 + 2/x = y + 2/y) : x * y = 2 := by
  sorry

end xy_equals_two_l1738_173839


namespace jessica_roses_thrown_away_l1738_173874

/-- The number of roses Jessica threw away -/
def roses_thrown_away (initial : ℕ) (added : ℕ) (final : ℕ) : ℕ :=
  initial + added - final

/-- Proof that Jessica threw away 4 roses -/
theorem jessica_roses_thrown_away :
  roses_thrown_away 2 25 23 = 4 := by
  sorry

end jessica_roses_thrown_away_l1738_173874


namespace sum_evaluation_l1738_173841

theorem sum_evaluation : 
  4/3 + 8/9 + 16/27 + 32/81 + 64/243 + 128/729 - 8 = -1/729 := by sorry

end sum_evaluation_l1738_173841


namespace division_of_decimals_l1738_173865

theorem division_of_decimals : (0.45 : ℚ) / (0.005 : ℚ) = 90 := by sorry

end division_of_decimals_l1738_173865


namespace video_vote_ratio_l1738_173860

theorem video_vote_ratio : 
  let up_votes : ℕ := 18
  let down_votes : ℕ := 4
  let ratio : ℚ := up_votes / down_votes
  ratio = 9 / 2 := by
  sorry

end video_vote_ratio_l1738_173860


namespace blue_faces_proportion_l1738_173881

/-- Given a cube of side length n, prove that if one-third of the faces of its unit cubes are blue, then n = 3 -/
theorem blue_faces_proportion (n : ℕ) : n ≥ 1 →
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by sorry

end blue_faces_proportion_l1738_173881


namespace geometric_sum_remainder_main_theorem_l1738_173890

theorem geometric_sum_remainder (n : ℕ) (a r : ℤ) (m : ℕ) (h : m > 0) :
  (a * (r^n - 1) / (r - 1)) % m = ((a * (r^n % m - 1)) / (r - 1)) % m :=
sorry

theorem main_theorem :
  (((3^1005 - 1) / 2) : ℤ) % 500 = 121 :=
sorry

end geometric_sum_remainder_main_theorem_l1738_173890


namespace composite_expression_prime_case_n_one_l1738_173863

theorem composite_expression (n : ℕ) :
  n > 1 → ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 3^(2*n+1) - 2^(2*n+1) - 6^n = a * b :=
sorry

theorem prime_case_n_one :
  3^3 - 2^3 - 6 = 13 :=
sorry

end composite_expression_prime_case_n_one_l1738_173863


namespace arithmetic_series_sum_l1738_173830

theorem arithmetic_series_sum : ∀ (a₁ aₙ : ℤ) (d : ℤ) (n : ℕ),
  a₁ = -41 →
  aₙ = 1 →
  d = 2 →
  n = (aₙ - a₁) / d + 1 →
  (n : ℤ) * (a₁ + aₙ) / 2 = -440 := by
  sorry

end arithmetic_series_sum_l1738_173830


namespace problem_1_problem_2_l1738_173895

-- Problem 1
theorem problem_1 : (π - 1)^0 - Real.sqrt 8 + |(- 2) * Real.sqrt 2| = 1 := by
  sorry

-- Problem 2
theorem problem_2 : ∀ x : ℝ, 3 * x - 2 > x + 4 ↔ x > 3 := by
  sorry

end problem_1_problem_2_l1738_173895


namespace chord_intersection_length_l1738_173852

/-- In a circle with radius R, chord AB of length a, diameter AC, and chord PQ perpendicular to AC
    intersecting AB at M with PM : MQ = 3 : 1, prove that AM = (4R²a) / (16R² - 3a²) -/
theorem chord_intersection_length (R a : ℝ) (h1 : R > 0) (h2 : a > 0) (h3 : a < 2*R) :
  ∃ (AM : ℝ), AM = (4 * R^2 * a) / (16 * R^2 - 3 * a^2) :=
sorry

end chord_intersection_length_l1738_173852


namespace horse_tile_problem_representation_l1738_173870

/-- Represents the equation for the horse and tile problem -/
def horse_tile_equation (x : ℝ) : Prop :=
  3 * x + (1/3) * (100 - x) = 100

/-- The total number of horses -/
def total_horses : ℝ := 100

/-- The total number of tiles -/
def total_tiles : ℝ := 100

/-- The number of tiles a big horse can pull -/
def big_horse_capacity : ℝ := 3

/-- The number of small horses needed to pull one tile -/
def small_horses_per_tile : ℝ := 3

/-- Theorem stating that the equation correctly represents the problem -/
theorem horse_tile_problem_representation :
  ∀ x, x ≥ 0 ∧ x ≤ total_horses →
  horse_tile_equation x ↔
    (x * big_horse_capacity + (total_horses - x) / small_horses_per_tile = total_tiles) :=
by sorry

end horse_tile_problem_representation_l1738_173870


namespace carrie_tomatoes_l1738_173836

/-- The number of tomatoes Carrie harvested -/
def tomatoes : ℕ := sorry

/-- The number of carrots Carrie harvested -/
def carrots : ℕ := 350

/-- The price of a tomato in dollars -/
def tomato_price : ℚ := 1

/-- The price of a carrot in dollars -/
def carrot_price : ℚ := 3/2

/-- The total revenue from selling all tomatoes and carrots in dollars -/
def total_revenue : ℚ := 725

theorem carrie_tomatoes : 
  tomatoes = 200 :=
sorry

end carrie_tomatoes_l1738_173836


namespace cupcake_difference_l1738_173859

theorem cupcake_difference (morning_cupcakes afternoon_cupcakes total_cupcakes : ℕ) : 
  morning_cupcakes = 20 →
  total_cupcakes = 55 →
  afternoon_cupcakes = total_cupcakes - morning_cupcakes →
  afternoon_cupcakes - morning_cupcakes = 15 := by
  sorry

end cupcake_difference_l1738_173859


namespace factorization_difference_of_squares_l1738_173801

theorem factorization_difference_of_squares (m x y : ℝ) : m * x^2 - m * y^2 = m * (x + y) * (x - y) := by
  sorry

end factorization_difference_of_squares_l1738_173801


namespace white_square_arc_length_bound_l1738_173891

/-- Represents a circle on a chessboard --/
structure ChessboardCircle where
  center : ℝ × ℝ
  radius : ℝ
  encloses_white_square : Bool

/-- Represents the portion of a circle's circumference passing through white squares --/
def white_square_arc_length (c : ChessboardCircle) : ℝ := sorry

/-- The theorem to be proved --/
theorem white_square_arc_length_bound 
  (c : ChessboardCircle) 
  (h1 : c.radius = 1) 
  (h2 : c.encloses_white_square = true) : 
  white_square_arc_length c ≤ (1/3) * (2 * Real.pi * c.radius) := by
  sorry

end white_square_arc_length_bound_l1738_173891


namespace complex_equation_solution_l1738_173829

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I + 1) / (Complex.I - 1) = Complex.mk a b → b = 1 := by
  sorry

end complex_equation_solution_l1738_173829


namespace greatest_divisor_with_remainders_l1738_173807

theorem greatest_divisor_with_remainders : Nat.gcd (1657 - 6) (2037 - 5) = 127 := by
  sorry

end greatest_divisor_with_remainders_l1738_173807


namespace sqrt_meaningful_range_l1738_173855

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 3 * x - 5) ↔ x ≥ 5 / 3 :=
by sorry

end sqrt_meaningful_range_l1738_173855


namespace parabola_transformation_l1738_173844

-- Define the original function
def f (x : ℝ) : ℝ := (x - 3)^2 - 4

-- Define the transformation
def transform (g : ℝ → ℝ) : ℝ → ℝ := λ x => g (x - 1) + 2

-- Define the expected result function
def expected_result (x : ℝ) : ℝ := (x - 4)^2 - 2

-- Theorem statement
theorem parabola_transformation :
  ∀ x, transform f x = expected_result x :=
sorry

end parabola_transformation_l1738_173844


namespace product_difference_l1738_173831

theorem product_difference (a b : ℕ+) : 
  a * b = 323 → a = 17 → b - a = 2 :=
by
  sorry

end product_difference_l1738_173831


namespace algebraic_expression_equality_l1738_173803

theorem algebraic_expression_equality (x y : ℝ) : 
  x + 2 * y + 1 = 3 → 2 * x + 4 * y + 1 = 5 := by
  sorry

end algebraic_expression_equality_l1738_173803


namespace product_of_roots_l1738_173804

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 20 → ∃ y : ℝ, (x + 3) * (x - 4) = 20 ∧ (y + 3) * (y - 4) = 20 ∧ x * y = -32 := by
  sorry

end product_of_roots_l1738_173804


namespace max_product_of_sums_l1738_173833

theorem max_product_of_sums (a b c d e f : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  a + b + c + d + e + f = 45 →
  (a + b + c) * (d + e + f) ≤ 550 := by
sorry

end max_product_of_sums_l1738_173833


namespace number_puzzle_l1738_173880

theorem number_puzzle (y : ℝ) (h : y ≠ 0) : y = (1 / y) * (-y) + 5 → y = 4 := by
  sorry

end number_puzzle_l1738_173880


namespace largest_and_smallest_numbers_l1738_173868

-- Define the numbers in their respective bases
def num1 : ℕ := 63  -- 111111₂ in decimal
def num2 : ℕ := 78  -- 210₆ in decimal
def num3 : ℕ := 64  -- 1000₄ in decimal
def num4 : ℕ := 65  -- 81₈ in decimal

-- Theorem statement
theorem largest_and_smallest_numbers :
  (num2 = max num1 (max num2 (max num3 num4))) ∧
  (num1 = min num1 (min num2 (min num3 num4))) := by
  sorry

end largest_and_smallest_numbers_l1738_173868


namespace real_part_of_z_l1738_173842

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  z.re = 1 := by sorry

end real_part_of_z_l1738_173842


namespace bear_climbing_problem_l1738_173812

/-- Represents the mountain climbing problem with two bears -/
structure MountainClimb where
  S : ℝ  -- Total distance from base to summit in meters
  VA : ℝ  -- Bear A's ascending speed
  VB : ℝ  -- Bear B's ascending speed
  meetingTime : ℝ  -- Time when bears meet (in hours)
  meetingDistance : ℝ  -- Distance from summit where bears meet (in meters)

/-- The theorem statement for the mountain climbing problem -/
theorem bear_climbing_problem (m : MountainClimb) : 
  m.VA > m.VB ∧  -- Bear A is faster than Bear B
  m.meetingTime = 2 ∧  -- Bears meet after 2 hours
  m.meetingDistance = 1600 ∧  -- Bears meet 1600 meters from summit
  m.S - 1600 = 2 * m.meetingTime * (m.VA + m.VB) ∧  -- Meeting condition
  (m.S + 800) / (m.S - 1600) = 5 / 4 →  -- Condition when Bear B reaches summit
  (m.S / m.VA + m.S / (2 * m.VA)) = 14 / 5  -- Total time for Bear A
  := by sorry

end bear_climbing_problem_l1738_173812


namespace quadratic_form_minimum_l1738_173862

theorem quadratic_form_minimum : 
  ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 9 ≥ -10 ∧ 
  ∃ x₀ y₀ : ℝ, 3 * x₀^2 + 4 * x₀ * y₀ + 2 * y₀^2 - 6 * x₀ + 8 * y₀ + 9 = -10 := by
  sorry

#check quadratic_form_minimum

end quadratic_form_minimum_l1738_173862


namespace cubic_inequality_l1738_173898

theorem cubic_inequality (x : ℝ) :
  x^3 - 12*x^2 + 47*x - 60 < 0 ↔ 3 < x ∧ x < 5 := by sorry

end cubic_inequality_l1738_173898


namespace ellipse_equation_l1738_173810

/-- Given a circle and an ellipse with specific properties, prove the equation of the ellipse -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 + y^2 = 4 → 
    (x^2 / a^2 + y^2 / b^2 = 1 ∧ 
     ((x = 0 ∧ y = b) ∨ (x = 0 ∧ y = -b) ∨ 
      (y = 0 ∧ x^2 = a^2 - b^2) ∨ (y = 0 ∧ x^2 = a^2 - b^2)))) →
  a^2 = 8 ∧ b^2 = 4 := by
sorry

end ellipse_equation_l1738_173810


namespace smallest_sum_is_28_l1738_173826

/-- Converts a number from base 6 to base 10 --/
def base6To10 (x y z : Nat) : Nat :=
  36 * x + 6 * y + z

/-- Converts a number from base b to base 10 --/
def baseBTo10 (b : Nat) : Nat :=
  3 * b + 3

/-- Represents the conditions of the problem --/
def validConfiguration (x y z b : Nat) : Prop :=
  x ≤ 5 ∧ y ≤ 5 ∧ z ≤ 5 ∧ b > 6 ∧ base6To10 x y z = baseBTo10 b

theorem smallest_sum_is_28 :
  ∃ x y z b, validConfiguration x y z b ∧
  ∀ x' y' z' b', validConfiguration x' y' z' b' →
    x + y + z + b ≤ x' + y' + z' + b' ∧
    x + y + z + b = 28 :=
sorry

end smallest_sum_is_28_l1738_173826


namespace quadratic_inequality_l1738_173835

theorem quadratic_inequality (x : ℝ) : x^2 - x - 12 < 0 ↔ -3 < x ∧ x < 4 := by
  sorry

end quadratic_inequality_l1738_173835


namespace equal_pairs_infinity_l1738_173896

def infinite_sequence (a : ℤ → ℝ) : Prop :=
  ∀ n : ℤ, a n = (1/4) * (a (n-1) + a (n+1))

theorem equal_pairs_infinity (a : ℤ → ℝ) :
  infinite_sequence a →
  (∃ i j : ℤ, i ≠ j ∧ a i = a j) →
  ∃ f : ℕ → (ℤ × ℤ), (∀ n : ℕ, (f n).1 ≠ (f n).2 ∧ a (f n).1 = a (f n).2) ∧
                      (∀ m n : ℕ, m ≠ n → f m ≠ f n) :=
by sorry

end equal_pairs_infinity_l1738_173896


namespace sphere_center_sum_l1738_173867

-- Define the points and constants
variable (a b c p q r α β γ : ℝ)

-- Define the conditions
variable (h1 : p^3 = α)
variable (h2 : q^3 = β)
variable (h3 : r^3 = γ)

-- Define the plane equation
variable (h4 : a/α + b/β + c/γ = 1)

-- Define that (p,q,r) is the center of the sphere passing through O, A, B, C
variable (h5 : p^2 + q^2 + r^2 = (p - α)^2 + q^2 + r^2)
variable (h6 : p^2 + q^2 + r^2 = p^2 + (q - β)^2 + r^2)
variable (h7 : p^2 + q^2 + r^2 = p^2 + q^2 + (r - γ)^2)

-- Theorem statement
theorem sphere_center_sum :
  a/p^3 + b/q^3 + c/r^3 = 1 :=
sorry

end sphere_center_sum_l1738_173867


namespace marks_towers_count_l1738_173871

/-- The number of sandcastles on Mark's beach -/
def marks_castles : ℕ := 20

/-- The number of sandcastles on Jeff's beach -/
def jeffs_castles : ℕ := 3 * marks_castles

/-- The number of towers on each of Jeff's sandcastles -/
def jeffs_towers_per_castle : ℕ := 5

/-- The total number of sandcastles and towers on both beaches -/
def total_count : ℕ := 580

/-- The number of towers on each of Mark's sandcastles -/
def marks_towers_per_castle : ℕ := 10

theorem marks_towers_count : 
  marks_castles + (marks_castles * marks_towers_per_castle) + 
  jeffs_castles + (jeffs_castles * jeffs_towers_per_castle) = total_count := by
  sorry

end marks_towers_count_l1738_173871


namespace ellipse_intersection_ratio_l1738_173882

/-- First ellipse -/
def ellipse1 (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Second ellipse -/
def ellipse2 (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

/-- Point P on the first ellipse -/
def P : ℝ × ℝ := sorry

/-- Point O is the origin -/
def O : ℝ × ℝ := (0, 0)

/-- Q is the intersection of ray PO with the second ellipse -/
noncomputable def Q : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem ellipse_intersection_ratio :
  ellipse1 P.1 P.2 →
  ellipse2 Q.1 Q.2 →
  distance P Q / distance O P = 3 := by sorry

end ellipse_intersection_ratio_l1738_173882
