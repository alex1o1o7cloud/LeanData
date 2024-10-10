import Mathlib

namespace average_marks_combined_classes_l3985_398546

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) 
  (h1 : n1 = 30) (h2 : n2 = 50) (h3 : avg1 = 40) (h4 : avg2 = 90) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 71.25 := by
  sorry

end average_marks_combined_classes_l3985_398546


namespace book_words_per_page_l3985_398571

theorem book_words_per_page :
  ∀ (words_per_page : ℕ),
    words_per_page ≤ 120 →
    (150 * words_per_page) % 221 = 210 →
    words_per_page = 48 := by
  sorry

end book_words_per_page_l3985_398571


namespace mike_lego_bridge_l3985_398583

/-- Calculates the number of bricks of other types Mike needs for his LEGO bridge. -/
def other_bricks (type_a : ℕ) (total : ℕ) : ℕ :=
  total - (type_a + type_a / 2)

/-- Theorem stating that Mike will use 90 bricks of other types for his LEGO bridge. -/
theorem mike_lego_bridge :
  ∀ (type_a : ℕ) (total : ℕ),
    type_a ≥ 40 →
    total = 150 →
    other_bricks type_a total = 90 := by
  sorry

end mike_lego_bridge_l3985_398583


namespace molecular_weight_N2O5H3P_l3985_398556

-- Define atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_P : ℝ := 30.97

-- Define the molecular formula
def N_count : ℕ := 2
def O_count : ℕ := 5
def H_count : ℕ := 3
def P_count : ℕ := 1

-- Theorem statement
theorem molecular_weight_N2O5H3P :
  N_count * atomic_weight_N +
  O_count * atomic_weight_O +
  H_count * atomic_weight_H +
  P_count * atomic_weight_P = 142.02 := by
  sorry

end molecular_weight_N2O5H3P_l3985_398556


namespace outfit_combinations_l3985_398523

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) : shirts = 5 → pants = 3 → shirts * pants = 15 := by
  sorry

end outfit_combinations_l3985_398523


namespace parametric_to_cartesian_l3985_398554

/-- Prove that the given parametric equations are equivalent to the Cartesian equation -/
theorem parametric_to_cartesian (t : ℝ) (x y : ℝ) (h1 : t ≠ 0) (h2 : x ≠ 1) 
  (h3 : x = 1 - 1/t) (h4 : y = 1 - t^2) : 
  y = x * (x - 2) / (x - 1)^2 := by
  sorry

end parametric_to_cartesian_l3985_398554


namespace z_value_l3985_398501

theorem z_value (x y z : ℝ) 
  (h1 : (x + y) / 2 = 4) 
  (h2 : x + y + z = 0) : 
  z = -8 := by
sorry

end z_value_l3985_398501


namespace problem_solution_l3985_398577

theorem problem_solution (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p) 
  (h_equation : p / (q - r) + q / (r - p) + r / (p - q) = 3) :
  p / (q - r)^2 + q / (r - p)^2 + r / (p - q)^2 = 0 := by
  sorry

end problem_solution_l3985_398577


namespace twenty_paise_coins_count_l3985_398582

theorem twenty_paise_coins_count (total_coins : ℕ) (total_value : ℚ) :
  total_coins = 342 →
  total_value = 71 →
  ∃ (coins_20 coins_25 : ℕ),
    coins_20 + coins_25 = total_coins ∧
    (20 * coins_20 + 25 * coins_25 : ℚ) / 100 = total_value ∧
    coins_20 = 290 := by
  sorry

end twenty_paise_coins_count_l3985_398582


namespace roots_positive_implies_b_in_range_l3985_398502

/-- A quadratic function f(x) = x² - 2x + b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + b

/-- The discriminant of f(x) -/
def discriminant (b : ℝ) : ℝ := 4 - 4*b

theorem roots_positive_implies_b_in_range (b : ℝ) :
  (∀ x : ℝ, f b x = 0 → x > 0) →
  0 < b ∧ b ≤ 1 := by sorry

end roots_positive_implies_b_in_range_l3985_398502


namespace salary_calculation_l3985_398515

/-- Represents the number of turbans given as part of the yearly salary -/
def turbans_per_year : ℕ := sorry

/-- The price of a turban in rupees -/
def turban_price : ℕ := 110

/-- The base salary in rupees for a full year -/
def base_salary : ℕ := 90

/-- The amount in rupees received by the servant after 9 months -/
def received_amount : ℕ := 40

/-- The number of months the servant worked -/
def months_worked : ℕ := 9

/-- The total number of months in a year -/
def months_in_year : ℕ := 12

theorem salary_calculation :
  (months_worked : ℚ) / months_in_year * (base_salary + turbans_per_year * turban_price) =
  received_amount + turban_price ∧ turbans_per_year = 1 := by sorry

end salary_calculation_l3985_398515


namespace absent_percentage_l3985_398516

theorem absent_percentage (total_students : ℕ) (boys : ℕ) (girls : ℕ)
  (h_total : total_students = 180)
  (h_boys : boys = 100)
  (h_girls : girls = 80)
  (h_sum : total_students = boys + girls)
  (absent_boys_fraction : ℚ)
  (absent_girls_fraction : ℚ)
  (h_absent_boys : absent_boys_fraction = 1 / 5)
  (h_absent_girls : absent_girls_fraction = 1 / 4) :
  (absent_boys_fraction * boys + absent_girls_fraction * girls) / total_students = 2 / 9 := by
sorry

end absent_percentage_l3985_398516


namespace train_journey_time_l3985_398519

theorem train_journey_time (S T D : ℝ) (h1 : D = S * T) (h2 : D = (S / 2) * (T + 4)) :
  T + 4 = 8 := by sorry

end train_journey_time_l3985_398519


namespace rational_product_sum_integer_sum_product_max_rational_product_negative_sum_comparison_l3985_398555

theorem rational_product_sum (a b : ℚ) : 
  a * b = 6 → a + b ≠ 0 := by sorry

theorem integer_sum_product_max (a b : ℤ) : 
  a + b = -5 → a * b ≤ 6 := by sorry

theorem rational_product_negative_sum_comparison (a b : ℚ) : 
  a * b < 0 → 
  (∃ (x y : ℚ), x * y < 0 ∧ x + y < 0) ∧ 
  (∃ (x y : ℚ), x * y < 0 ∧ x + y = 0) ∧ 
  (∃ (x y : ℚ), x * y < 0 ∧ x + y > 0) := by sorry

end rational_product_sum_integer_sum_product_max_rational_product_negative_sum_comparison_l3985_398555


namespace zeros_of_composite_function_l3985_398553

noncomputable section

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then k * x + 1 else Real.log x

-- Define the composite function g
def g (k : ℝ) (x : ℝ) : ℝ := f k (f k x + 1)

-- Theorem statement
theorem zeros_of_composite_function (k : ℝ) :
  (k > 0 → (∃ x₁ x₂ x₃ x₄ : ℝ, g k x₁ = 0 ∧ g k x₂ = 0 ∧ g k x₃ = 0 ∧ g k x₄ = 0 ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄)) ∧
  (k < 0 → (∃! x : ℝ, g k x = 0)) :=
sorry

end

end zeros_of_composite_function_l3985_398553


namespace blocks_lost_l3985_398590

/-- Given Carol's initial and final block counts, prove the number of blocks lost. -/
theorem blocks_lost (initial : ℕ) (final : ℕ) (h1 : initial = 42) (h2 : final = 17) :
  initial - final = 25 := by
  sorry

end blocks_lost_l3985_398590


namespace square_equation_solution_l3985_398518

theorem square_equation_solution :
  ∃! x : ℤ, (2020 + x)^2 = x^2 :=
by
  use -1010
  sorry

end square_equation_solution_l3985_398518


namespace sum_of_common_ratios_l3985_398544

/-- Given two nonconstant geometric sequences with different common ratios,
    if 3(a₃ - b₃) = 4(a₂ - b₂), then the sum of their common ratios is 4/3 -/
theorem sum_of_common_ratios (k a₂ a₃ b₂ b₃ p r : ℝ) 
    (h1 : k ≠ 0)
    (h2 : p ≠ 1)
    (h3 : r ≠ 1)
    (h4 : p ≠ r)
    (h5 : a₂ = k * p)
    (h6 : a₃ = k * p^2)
    (h7 : b₂ = k * r)
    (h8 : b₃ = k * r^2)
    (h9 : 3 * (a₃ - b₃) = 4 * (a₂ - b₂)) :
  p + r = 4/3 := by
  sorry

end sum_of_common_ratios_l3985_398544


namespace distance_between_points_l3985_398505

/-- The distance between points (1, 3) and (-5, 7) is 2√13 units. -/
theorem distance_between_points : 
  let pointA : ℝ × ℝ := (1, 3)
  let pointB : ℝ × ℝ := (-5, 7)
  Real.sqrt ((pointB.1 - pointA.1)^2 + (pointB.2 - pointA.2)^2) = 2 * Real.sqrt 13 :=
by sorry

end distance_between_points_l3985_398505


namespace complement_of_union_l3985_398584

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {4, 5}
def B : Set Nat := {3, 4}

theorem complement_of_union :
  (U \ (A ∪ B)) = {1, 2, 6} := by sorry

end complement_of_union_l3985_398584


namespace nested_radical_value_l3985_398526

/-- The value of the infinite nested radical sqrt(3 - sqrt(3 - sqrt(3 - ...))) -/
noncomputable def nestedRadical : ℝ :=
  Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt 3))))))

/-- Theorem stating that the nested radical equals (-1 + sqrt(13)) / 2 -/
theorem nested_radical_value : nestedRadical = (-1 + Real.sqrt 13) / 2 := by
  sorry

end nested_radical_value_l3985_398526


namespace cards_nell_has_left_nell_remaining_cards_l3985_398592

/-- Calculates the number of cards Nell has left after giving some to Jeff -/
theorem cards_nell_has_left (nell_initial : ℕ) (jeff_initial : ℕ) (jeff_final : ℕ) : ℕ :=
  let cards_transferred := jeff_final - jeff_initial
  nell_initial - cards_transferred

/-- Proves that Nell has 252 cards left after giving some to Jeff -/
theorem nell_remaining_cards : 
  cards_nell_has_left 528 11 287 = 252 := by
  sorry


end cards_nell_has_left_nell_remaining_cards_l3985_398592


namespace jacob_rental_cost_l3985_398535

/-- Calculates the total cost of renting a car given the daily rate, per-mile rate, number of days, and miles driven. -/
def total_rental_cost (daily_rate : ℚ) (mile_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mile_rate * miles

/-- Proves that Jacob's total car rental cost is $237.5 given the specified conditions. -/
theorem jacob_rental_cost :
  let daily_rate : ℚ := 30
  let mile_rate : ℚ := 1/4
  let rental_days : ℕ := 5
  let miles_driven : ℕ := 350
  total_rental_cost daily_rate mile_rate rental_days miles_driven = 237.5 := by
sorry

end jacob_rental_cost_l3985_398535


namespace coefficient_of_x_fourth_l3985_398503

theorem coefficient_of_x_fourth (x : ℝ) : 
  let expr := 5*(x^4 - 2*x^5) + 3*(x^2 - 3*x^4 + 2*x^6) - (2*x^5 - 3*x^4)
  ∃ (a b c d e f : ℝ), expr = a*x^6 + b*x^5 + (-1)*x^4 + d*x^3 + e*x^2 + f*x + c :=
by sorry

end coefficient_of_x_fourth_l3985_398503


namespace painting_cost_cny_l3985_398563

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℚ := 8

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℚ := 7

/-- Cost of the painting in Namibian dollars -/
def painting_cost_nad : ℚ := 160

/-- Theorem stating the cost of the painting in Chinese yuan -/
theorem painting_cost_cny : 
  (painting_cost_nad / usd_to_nad) * usd_to_cny = 140 := by
  sorry

end painting_cost_cny_l3985_398563


namespace factors_imply_absolute_value_l3985_398569

-- Define the polynomial
def p (h k x : ℝ) : ℝ := 3 * x^3 - h * x - 3 * k

-- Define the factors
def f₁ (x : ℝ) : ℝ := x + 3
def f₂ (x : ℝ) : ℝ := x - 2

-- Theorem statement
theorem factors_imply_absolute_value (h k : ℝ) :
  (∀ x, p h k x = 0 → f₁ x = 0 ∨ f₂ x = 0) →
  |3 * h - 4 * k| = 615 := by
  sorry

end factors_imply_absolute_value_l3985_398569


namespace angle_equivalence_l3985_398509

def angle_with_same_terminal_side (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 - 120

theorem angle_equivalence :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 360 ∧ angle_with_same_terminal_side θ ∧ θ = 240 := by
sorry

end angle_equivalence_l3985_398509


namespace triangle_angle_measure_l3985_398566

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 3a cos C = 2c cos A and tan A = 1/3, then angle B measures 135°. -/
theorem triangle_angle_measure (a b c A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C → -- angles are positive
  A + B + C = π → -- sum of angles in a triangle
  3 * a * Real.cos C = 2 * c * Real.cos A →
  Real.tan A = 1 / 3 →
  B = π / 4 * 3 := by
sorry

end triangle_angle_measure_l3985_398566


namespace prop_one_prop_three_prop_five_l3985_398559

-- Proposition 1
theorem prop_one (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (a + 1) / (b + 1) > a / b) : a < b := by
  sorry

-- Proposition 3
theorem prop_three : ∀ x : ℝ, x^2 - 2*x + 1 ≥ 0 := by
  sorry

-- Proposition 5
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (λ acc (i, b) => acc + if b then 2^i else 0) 0

theorem prop_five : binary_to_decimal [true, false, true, true, true] = 23 := by
  sorry

end prop_one_prop_three_prop_five_l3985_398559


namespace no_nonzero_real_solution_l3985_398565

theorem no_nonzero_real_solution :
  ¬ ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 1/a + 1/b = 2/(a+b) := by
  sorry

end no_nonzero_real_solution_l3985_398565


namespace triangle_properties_l3985_398598

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.a * Real.sin t.C = Real.sqrt 3 * t.c * Real.cos t.A ∧
  t.b = 2 ∧
  (1 / 2) * t.b * t.c * Real.sin t.A = Real.sqrt 3

-- State the theorem
theorem triangle_properties (t : Triangle) :
  satisfies_conditions t → t.A = π / 3 ∧ t.a = 2 := by
  sorry

end triangle_properties_l3985_398598


namespace firm_employee_count_l3985_398579

-- Define the initial number of Democrats and Republicans
def initial_democrats : ℕ := sorry
def initial_republicans : ℕ := sorry

-- Define the conditions
axiom condition1 : initial_democrats + 1 = initial_republicans - 1
axiom condition2 : initial_democrats + 4 = 2 * (initial_republicans - 4)

-- Define the total number of employees
def total_employees : ℕ := initial_democrats + initial_republicans

-- Theorem to prove
theorem firm_employee_count : total_employees = 18 := by
  sorry

end firm_employee_count_l3985_398579


namespace solution_set_f_greater_than_two_range_of_t_l3985_398533

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x > 1 ∨ x < -5} :=
sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (11/2)*t} = {t : ℝ | 1/2 ≤ t ∧ t ≤ 5} :=
sorry

end solution_set_f_greater_than_two_range_of_t_l3985_398533


namespace max_value_on_edge_l3985_398542

/-- A 2D grid represented as a function from pairs of integers to real numbers. -/
def Grid := ℤ × ℤ → ℝ

/-- Predicate to check if a cell is on the edge of the grid. -/
def isOnEdge (m n : ℕ) (i j : ℤ) : Prop :=
  i = 0 ∨ i = m - 1 ∨ j = 0 ∨ j = n - 1

/-- The set of valid coordinates in an m × n grid. -/
def validCoords (m n : ℕ) : Set (ℤ × ℤ) :=
  {(i, j) | 0 ≤ i ∧ i < m ∧ 0 ≤ j ∧ j < n}

/-- Predicate to check if a grid satisfies the arithmetic mean property. -/
def satisfiesArithmeticMeanProperty (g : Grid) (m n : ℕ) : Prop :=
  ∀ (i j : ℤ), (i, j) ∈ validCoords m n → ¬isOnEdge m n i j →
    g (i, j) = (g (i-1, j) + g (i+1, j) + g (i, j-1) + g (i, j+1)) / 4

/-- Theorem: The maximum value in a grid satisfying the arithmetic mean property
    must be on the edge. -/
theorem max_value_on_edge (g : Grid) (m n : ℕ) 
    (h_mean : satisfiesArithmeticMeanProperty g m n)
    (h_distinct : ∀ (i j k l : ℤ), (i, j) ≠ (k, l) → 
      (i, j) ∈ validCoords m n → (k, l) ∈ validCoords m n → g (i, j) ≠ g (k, l))
    (h_finite : m > 0 ∧ n > 0) :
    ∃ (i j : ℤ), (i, j) ∈ validCoords m n ∧ isOnEdge m n i j ∧
      ∀ (k l : ℤ), (k, l) ∈ validCoords m n → g (i, j) ≥ g (k, l) :=
  sorry

end max_value_on_edge_l3985_398542


namespace sum_of_integers_l3985_398574

theorem sum_of_integers (a b c d e : ℤ) 
  (eq1 : a - b + c - e = 7)
  (eq2 : b - c + d + e = 9)
  (eq3 : c - d + a - e = 5)
  (eq4 : d - a + b + e = 1) :
  a + b + c + d + e = 11 := by
  sorry

end sum_of_integers_l3985_398574


namespace sqrt_simplification_l3985_398595

theorem sqrt_simplification (a : ℝ) (ha : a > 0) : a^2 * Real.sqrt a = a^(5/2) := by
  sorry

end sqrt_simplification_l3985_398595


namespace programmer_debug_time_l3985_398525

/-- Proves that given a 48-hour work week, where 1/4 of the time is spent on flow charts
    and 3/8 on coding, the remaining time spent on debugging is 18 hours. -/
theorem programmer_debug_time (total_hours : ℝ) (flow_chart_fraction : ℝ) (coding_fraction : ℝ) :
  total_hours = 48 →
  flow_chart_fraction = 1/4 →
  coding_fraction = 3/8 →
  total_hours * (1 - flow_chart_fraction - coding_fraction) = 18 :=
by sorry

end programmer_debug_time_l3985_398525


namespace soccer_game_water_consumption_l3985_398531

/-- Proves that the number of water bottles consumed is 4, given the initial quantities,
    remaining bottles, and the relationship between water and soda consumption. -/
theorem soccer_game_water_consumption
  (initial_water : Nat)
  (initial_soda : Nat)
  (remaining_bottles : Nat)
  (h1 : initial_water = 12)
  (h2 : initial_soda = 34)
  (h3 : remaining_bottles = 30)
  (h4 : ∀ w s, w + s = initial_water + initial_soda - remaining_bottles → s = 3 * w) :
  initial_water + initial_soda - remaining_bottles - 3 * (initial_water + initial_soda - remaining_bottles) / 4 = 4 :=
by sorry

end soccer_game_water_consumption_l3985_398531


namespace A_inter_B_eq_a_geq_2_l3985_398528

-- Define sets A, B, and C
def A : Set ℝ := {x | (x + 4) * (x - 2) > 0}
def B : Set ℝ := {y | ∃ x, y = x^2 - 2*x + 2}
def C (a : ℝ) : Set ℝ := {x | -4 ≤ x ∧ x ≤ a}

-- Define the complement of A relative to C
def C_R_A (a : ℝ) : Set ℝ := {x | x ∈ C a ∧ x ∉ A}

-- Theorem for part (I)
theorem A_inter_B_eq : A ∩ B = {x | x > 2} := by sorry

-- Theorem for part (II)
theorem a_geq_2 (a : ℝ) : C_R_A a ⊆ C a → a ≥ 2 := by sorry

end A_inter_B_eq_a_geq_2_l3985_398528


namespace cos_sin_equation_solution_l3985_398529

theorem cos_sin_equation_solution (n : ℕ) :
  ∀ x : ℝ, (Real.cos x)^n - (Real.sin x)^n = 1 ↔
    (n % 2 = 0 ∧ ∃ k : ℤ, x = k * Real.pi) ∨
    (n % 2 = 1 ∧ (∃ k : ℤ, x = 2 * k * Real.pi ∨ x = (3 / 2 + 2 * k) * Real.pi)) :=
by sorry

end cos_sin_equation_solution_l3985_398529


namespace min_xy_value_l3985_398524

theorem min_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 6) :
  ∃ (x₀ y₀ : ℕ+), (1 : ℚ) / x₀ + (1 : ℚ) / (3 * y₀) = (1 : ℚ) / 6 ∧
    x₀.val * y₀.val = 48 ∧
    ∀ (a b : ℕ+), (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 6 →
      x₀.val * y₀.val ≤ a.val * b.val :=
by sorry

end min_xy_value_l3985_398524


namespace smallest_right_triangle_area_l3985_398513

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let area1 := (1/2) * a * b
  let c := Real.sqrt (b^2 - a^2)
  let area2 := (1/2) * a * c
  min area1 area2 = 6 * Real.sqrt 7 := by
sorry

end smallest_right_triangle_area_l3985_398513


namespace no_integer_solutions_specific_solution_l3985_398591

theorem no_integer_solutions : ¬∃ (m n : ℤ), m^3 + 6*m^2 + 5*m = 27*n^3 + 9*n^2 + 9*n + 1 := by
  sorry

-- Additional fact mentioned in the problem
theorem specific_solution : (31 * 26)^3 + 6*(31*26)^2 + 5*(31*26) = 27*n^3 + 9*n^2 + 9*n + 1 := by
  sorry

end no_integer_solutions_specific_solution_l3985_398591


namespace reeyas_average_is_73_l3985_398543

def reeyas_scores : List ℝ := [55, 67, 76, 82, 85]

theorem reeyas_average_is_73 : 
  (reeyas_scores.sum / reeyas_scores.length : ℝ) = 73 := by
  sorry

end reeyas_average_is_73_l3985_398543


namespace some_number_value_l3985_398539

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = n * 25 * 45 * 49) : n = 7 := by
  sorry

end some_number_value_l3985_398539


namespace student_trip_cost_is_1925_l3985_398594

/-- Calculates the amount each student needs for a trip given fundraising conditions -/
def student_trip_cost (num_students : ℕ) 
                      (misc_expenses : ℕ) 
                      (day1_raised : ℕ) 
                      (day2_raised : ℕ) 
                      (day3_raised : ℕ) 
                      (additional_days : ℕ) 
                      (additional_per_student : ℕ) : ℕ :=
  let first_three_days := day1_raised + day2_raised + day3_raised
  let next_days_total := (first_three_days / 2) * additional_days
  let total_raised := first_three_days + next_days_total
  let total_needed := total_raised + misc_expenses + (num_students * additional_per_student)
  total_needed / num_students

/-- Theorem stating that given the specific conditions, each student needs $1925 for the trip -/
theorem student_trip_cost_is_1925 : 
  student_trip_cost 6 3000 600 900 400 4 475 = 1925 := by
  sorry

#eval student_trip_cost 6 3000 600 900 400 4 475

end student_trip_cost_is_1925_l3985_398594


namespace ursula_initial_money_l3985_398558

/-- Calculates the initial amount of money Ursula had given the purchase details --/
def initial_money (num_hotdogs : ℕ) (price_hotdog : ℚ) (num_salads : ℕ) (price_salad : ℚ) (change : ℚ) : ℚ :=
  num_hotdogs * price_hotdog + num_salads * price_salad + change

/-- Proves that Ursula's initial money was $20.00 given the purchase details --/
theorem ursula_initial_money :
  initial_money 5 (3/2) 3 (5/2) 5 = 20 := by
  sorry

end ursula_initial_money_l3985_398558


namespace arthur_muffins_l3985_398522

theorem arthur_muffins (initial_muffins additional_muffins : ℕ) 
  (h1 : initial_muffins = 35)
  (h2 : additional_muffins = 48) :
  initial_muffins + additional_muffins = 83 :=
by sorry

end arthur_muffins_l3985_398522


namespace factorization_demonstrates_transformation_l3985_398549

/-- Represents a quadratic equation of the form ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the method used to solve the equation -/
inductive SolvingMethod
  | Factorization

/-- Represents the mathematical idea demonstrated by the solving method -/
inductive MathematicalIdea
  | Transformation
  | Function
  | CombiningNumbersAndShapes
  | Axiomatic

/-- Solves a quadratic equation using the given method -/
def solveQuadratic (eq : QuadraticEquation) (method : SolvingMethod) : Set ℝ :=
  sorry

/-- Determines the mathematical idea demonstrated by the solving method -/
def demonstratedIdea (eq : QuadraticEquation) (method : SolvingMethod) : MathematicalIdea :=
  sorry

theorem factorization_demonstrates_transformation : 
  let eq : QuadraticEquation := { a := 3, b := -6, c := 0 }
  demonstratedIdea eq SolvingMethod.Factorization = MathematicalIdea.Transformation :=
by sorry

end factorization_demonstrates_transformation_l3985_398549


namespace quadratic_not_through_point_l3985_398512

def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

theorem quadratic_not_through_point (p q : ℝ) :
  f p q 1 = 1 → f p q 3 = 1 → f p q 4 ≠ 5 := by
  sorry

end quadratic_not_through_point_l3985_398512


namespace contrapositive_equivalence_l3985_398575

def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, x^2 + x - m = 0

theorem contrapositive_equivalence :
  (∀ m : ℝ, m ≤ 0 → has_real_roots m) ↔
  (∀ m : ℝ, ¬(has_real_roots m) → m > 0) :=
by sorry

end contrapositive_equivalence_l3985_398575


namespace wardrobe_cost_calculation_l3985_398532

def wardrobe_cost (skirt_price blouse_price jacket_price pant_price : ℝ)
  (skirt_discount jacket_discount : ℝ) (tax_rate : ℝ) : ℝ :=
  let skirt_total := 4 * (skirt_price * (1 - skirt_discount))
  let blouse_total := 6 * blouse_price
  let jacket_total := 2 * (jacket_price - jacket_discount)
  let pant_total := 2 * pant_price + 0.5 * pant_price
  let subtotal := skirt_total + blouse_total + jacket_total + pant_total
  subtotal * (1 + tax_rate)

theorem wardrobe_cost_calculation :
  wardrobe_cost 25 18 45 35 0.1 5 0.07 = 391.09 := by
  sorry

end wardrobe_cost_calculation_l3985_398532


namespace cube_root_square_l3985_398534

theorem cube_root_square (x : ℝ) : (x + 5) ^ (1/3 : ℝ) = 3 → (x + 5)^2 = 729 := by
  sorry

end cube_root_square_l3985_398534


namespace complex_equation_solution_l3985_398514

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I = 11 + 7 * Complex.I) : 
  z = 7 - 11 * Complex.I := by
sorry

end complex_equation_solution_l3985_398514


namespace village_apple_trees_l3985_398527

theorem village_apple_trees (total_sample_percentage : ℚ) 
  (apple_trees_in_sample : ℕ) (total_apple_trees : ℕ) : 
  total_sample_percentage = 1/10 →
  apple_trees_in_sample = 80 →
  total_apple_trees = apple_trees_in_sample / total_sample_percentage →
  total_apple_trees = 800 := by
  sorry

end village_apple_trees_l3985_398527


namespace twice_largest_two_digit_is_190_l3985_398551

def largest_two_digit (a b c : Nat) : Nat :=
  max (10 * max a (max b c) + min (max a b) (max b c))
      (10 * min (max a b) (max b c) + max a (max b c))

theorem twice_largest_two_digit_is_190 :
  largest_two_digit 3 5 9 * 2 = 190 := by
  sorry

end twice_largest_two_digit_is_190_l3985_398551


namespace binomial_divisibility_l3985_398588

theorem binomial_divisibility (k : ℕ) (hk : k ≥ 2) :
  (∃ m : ℕ, Nat.choose (2^k) 2 + Nat.choose (2^k) 3 = 2^(3*k) * m) ∧
  (∀ n : ℕ, Nat.choose (2^k) 2 + Nat.choose (2^k) 3 ≠ 2^(3*k + 1) * n) :=
sorry

end binomial_divisibility_l3985_398588


namespace largest_x_abs_equation_l3985_398507

theorem largest_x_abs_equation : 
  ∀ x : ℝ, |x - 3| = 14.5 → x ≤ 17.5 ∧ ∃ y : ℝ, |y - 3| = 14.5 ∧ y = 17.5 := by
  sorry

end largest_x_abs_equation_l3985_398507


namespace ice_cream_sales_l3985_398573

theorem ice_cream_sales (sales : List ℝ) (mean : ℝ) : 
  sales.length = 6 →
  sales = [100, 92, 109, 96, 103, 105] →
  mean = 100.1 →
  (sales.sum + (7 * mean - sales.sum)) / 7 = mean →
  7 * mean - sales.sum = 95.7 :=
by sorry

end ice_cream_sales_l3985_398573


namespace divisibility_pairs_l3985_398580

theorem divisibility_pairs : 
  ∀ m n : ℕ+, 
    (∀ k : ℕ+, k ≤ n → m.val % k = 0) ∧ 
    (m.val % (n + 1) ≠ 0) ∧ 
    (m.val % (n + 2) ≠ 0) ∧ 
    (m.val % (n + 3) ≠ 0) →
    ((n = 1 ∧ Nat.gcd m.val 6 = 1) ∨ 
     (n = 2 ∧ Nat.gcd m.val 12 = 1)) :=
by sorry

end divisibility_pairs_l3985_398580


namespace sarah_walked_4_6_miles_l3985_398572

/-- The distance Sarah walked in miles -/
def sarah_distance (blocks_west : ℕ) (blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Theorem stating that Sarah walked 4.6 miles -/
theorem sarah_walked_4_6_miles :
  sarah_distance 8 15 (1/5) = 46/10 := by
  sorry

end sarah_walked_4_6_miles_l3985_398572


namespace triangle_properties_l3985_398541

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  Real.sqrt 3 * Real.cos t.B = t.b * Real.sin t.C

def condition2 (t : Triangle) : Prop :=
  2 * t.a - t.c = 2 * t.b * Real.cos t.C

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h_b : t.b = 2 * Real.sqrt 3)
  (h_cond1 : condition1 t)
  (h_cond2 : condition2 t) :
  (∃ (area : ℝ), t.a = 2 → area = 2 * Real.sqrt 3) ∧
  (2 * Real.sqrt 3 < t.a + t.c ∧ t.a + t.c ≤ 4 * Real.sqrt 3) := by
  sorry

end triangle_properties_l3985_398541


namespace ten_integer_segments_l3985_398593

/-- Right triangle DEF with integer leg lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- The number of distinct integer lengths of line segments from E to DF -/
def num_integer_segments (t : RightTriangle) : ℕ :=
  sorry

/-- Our specific right triangle -/
def triangle : RightTriangle :=
  { de := 18, ef := 24 }

theorem ten_integer_segments : num_integer_segments triangle = 10 := by
  sorry

end ten_integer_segments_l3985_398593


namespace intersection_A_B_l3985_398586

def A : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {y | ∃ x ∈ A, y = 3 * x - 2}

theorem intersection_A_B : A ∩ B = {1, 4} := by sorry

end intersection_A_B_l3985_398586


namespace sine_ratio_equals_two_l3985_398560

/-- Triangle ABC with vertices A(-1, 0), C(1, 0), and B on the ellipse x²/4 + y²/3 = 1 -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_A : A = (-1, 0)
  h_C : C = (1, 0)
  h_B : (B.1^2 / 4) + (B.2^2 / 3) = 1

/-- The sine of an angle in the triangle -/
noncomputable def sin_angle (t : Triangle) (v : Fin 3) : ℝ :=
  sorry

/-- Theorem stating that (sin A + sin C) / sin B = 2 for the given triangle -/
theorem sine_ratio_equals_two (t : Triangle) :
  (sin_angle t 0 + sin_angle t 2) / sin_angle t 1 = 2 :=
sorry

end sine_ratio_equals_two_l3985_398560


namespace min_value_sum_reciprocals_l3985_398521

/-- Given a line 2ax - by + 2 = 0 passing through the center of the circle (x + 1)^2 + (y - 2)^2 = 4,
    where a > 0 and b > 0, the minimum value of 1/a + 1/b is 4 -/
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : 2 * a * (-1) - b * 2 + 2 = 0) : 
  (∀ x y, (x + 1)^2 + (y - 2)^2 = 4 → 2 * a * x - b * y + 2 = 0) → 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 2 * a' * (-1) - b' * 2 + 2 = 0 → 1 / a' + 1 / b' ≥ 1 / a + 1 / b) →
  1 / a + 1 / b = 4 := by
sorry

end min_value_sum_reciprocals_l3985_398521


namespace card_play_combinations_l3985_398552

/-- Represents the number of ways to play 5 cards (2 twos and 3 aces) -/
def ways_to_play_cards : ℕ :=
  Nat.factorial 5 + 
  Nat.factorial 2 + 
  Nat.factorial 4 + 
  (Nat.choose 3 2 * Nat.factorial 3) + 
  Nat.factorial 3 + 
  (Nat.choose 3 2 * Nat.factorial 4)

/-- Theorem stating that the number of ways to play the cards is 242 -/
theorem card_play_combinations : ways_to_play_cards = 242 := by
  sorry

end card_play_combinations_l3985_398552


namespace jerry_age_l3985_398597

theorem jerry_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 24 → 
  mickey_age = 4 * jerry_age - 8 → 
  jerry_age = 8 := by
sorry

end jerry_age_l3985_398597


namespace total_items_in_jar_l3985_398500

-- Define the number of candies
def total_candies : ℕ := 3409
def chocolate_candies : ℕ := 1462
def gummy_candies : ℕ := 1947

-- Define the number of secret eggs
def total_eggs : ℕ := 145
def eggs_with_one_prize : ℕ := 98
def eggs_with_two_prizes : ℕ := 38
def eggs_with_three_prizes : ℕ := 9

-- Theorem to prove
theorem total_items_in_jar : 
  total_candies + 
  (eggs_with_one_prize * 1 + eggs_with_two_prizes * 2 + eggs_with_three_prizes * 3) = 3610 := by
  sorry

end total_items_in_jar_l3985_398500


namespace cos_double_angle_given_tan_l3985_398564

theorem cos_double_angle_given_tan (α : Real) (h : Real.tan α = 3) : 
  Real.cos (2 * α) = -4/5 := by
  sorry

end cos_double_angle_given_tan_l3985_398564


namespace rhombus_area_l3985_398557

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) : 
  (1 / 2 : ℝ) * d1 * d2 = 24 := by
  sorry

end rhombus_area_l3985_398557


namespace magnitude_of_z_is_one_l3985_398585

/-- Given a complex number z defined as z = (1-i)/(1+i) + 2i, prove that its magnitude |z| is equal to 1 -/
theorem magnitude_of_z_is_one : 
  let z : ℂ := (1 - Complex.I) / (1 + Complex.I) + 2 * Complex.I
  Complex.abs z = 1 := by
  sorry

end magnitude_of_z_is_one_l3985_398585


namespace team_division_probabilities_l3985_398599

def totalTeams : ℕ := 8
def weakTeams : ℕ := 3
def groupSize : ℕ := 4

/-- The probability that one of the groups has exactly 2 weak teams -/
def prob_exactly_two_weak : ℚ := 6/7

/-- The probability that group A has at least 2 weak teams -/
def prob_at_least_two_weak : ℚ := 1/2

theorem team_division_probabilities :
  (totalTeams = 8 ∧ weakTeams = 3 ∧ groupSize = 4) →
  (prob_exactly_two_weak = 6/7 ∧ prob_at_least_two_weak = 1/2) := by
  sorry

end team_division_probabilities_l3985_398599


namespace a_2012_value_l3985_398510

-- Define the sequence
def a : ℕ → ℚ
  | 0 => 2
  | (n + 1) => a n / (1 + a n)

-- State the theorem
theorem a_2012_value : a 2012 = 2 / 4025 := by
  sorry

end a_2012_value_l3985_398510


namespace place_value_decomposition_l3985_398520

theorem place_value_decomposition :
  (286 = 200 + 80 + 6) ∧
  (7560 = 7000 + 500 + 60) ∧
  (2048 = 2000 + 40 + 8) ∧
  (8009 = 8000 + 9) ∧
  (3070 = 3000 + 70) := by
  sorry

end place_value_decomposition_l3985_398520


namespace amount_ratio_l3985_398545

/-- Given three amounts a, b, and c in rupees, prove that the ratio of a to b is 3:1 -/
theorem amount_ratio (a b c : ℕ) : 
  a + b + c = 645 →
  b = c + 25 →
  b = 134 →
  a / b = 3 := by
  sorry

end amount_ratio_l3985_398545


namespace proportional_function_and_point_l3985_398596

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x + 2

-- State the theorem
theorem proportional_function_and_point :
  -- Conditions
  (∃ k : ℝ, ∀ x y : ℝ, y - 2 = k * x) →  -- y-2 is directly proportional to x
  (f 1 = 6) →                           -- When x=1, y=6
  (f (-3/4) = -1) →                     -- Point P(a, -1) is on the graph of the function
  -- Conclusions
  ((∀ x : ℝ, f x = 4 * x + 2) ∧         -- The function expression is y = 4x + 2
   (-3/4 : ℝ) = -3/4)                   -- The value of a is -3/4
  := by sorry

end proportional_function_and_point_l3985_398596


namespace fruit_sales_problem_l3985_398561

-- Define the variables and constants
def june_total_A : ℝ := 12000
def june_total_B : ℝ := 9000
def july_total_quantity : ℝ := 5000
def july_min_total : ℝ := 23400
def cost_A : ℝ := 2.7
def cost_B : ℝ := 3.5

-- Define the theorem
theorem fruit_sales_problem :
  ∃ (june_price_A : ℝ) (july_quantity_A : ℝ) (july_profit : ℝ),
    -- Conditions
    (june_total_A / june_price_A - june_total_B / (1.5 * june_price_A) = 1000) ∧
    (0.7 * june_price_A * july_quantity_A + 0.6 * 1.5 * june_price_A * (july_total_quantity - july_quantity_A) ≥ july_min_total) ∧
    -- Conclusions
    (june_price_A = 6) ∧
    (july_quantity_A = 3000) ∧
    (july_profit = (0.7 * june_price_A - cost_A) * july_quantity_A + 
                   (0.6 * 1.5 * june_price_A - cost_B) * (july_total_quantity - july_quantity_A)) ∧
    (july_profit = 8300) :=
by
  sorry

end fruit_sales_problem_l3985_398561


namespace function_property_l3985_398504

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x * y) = f x * f y)
  (h2 : ∀ x y : ℝ, f (x + y) ≤ 2 * (f x + f y))
  (h3 : f 2 = 4) : 
  f 3 ≤ 9 := by sorry

end function_property_l3985_398504


namespace quadratic_roots_sum_bound_l3985_398562

theorem quadratic_roots_sum_bound (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁^2 + p*r₁ + 15 = 0 → r₂^2 + p*r₂ + 15 = 0 → |r₁ + r₂| > 2 * Real.sqrt 15 := by
  sorry

end quadratic_roots_sum_bound_l3985_398562


namespace min_value_abc_l3985_398540

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 1/a + 1/b + 1/c = 9) :
  a^4 * b^3 * c^2 ≥ 1/10368 ∧ ∃ (a₀ b₀ c₀ : ℝ), 
    a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 1/a₀ + 1/b₀ + 1/c₀ = 9 ∧ a₀^4 * b₀^3 * c₀^2 = 1/10368 :=
by sorry

end min_value_abc_l3985_398540


namespace perpendicular_lines_parallel_l3985_398537

-- Define the space we're working in
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define lines and planes
def Line (V : Type*) [NormedAddCommGroup V] := V → Prop
def Plane (V : Type*) [NormedAddCommGroup V] := V → Prop

-- Define perpendicular relation between a line and a plane
def Perpendicular (l : Line V) (p : Plane V) : Prop := sorry

-- Define parallel relation between two lines
def Parallel (l1 l2 : Line V) : Prop := sorry

-- Theorem statement
theorem perpendicular_lines_parallel 
  (m n : Line V) (α : Plane V) 
  (hm : m ≠ n) 
  (h1 : Perpendicular m α) 
  (h2 : Perpendicular n α) : 
  Parallel m n :=
sorry

end perpendicular_lines_parallel_l3985_398537


namespace sum_of_even_coefficients_l3985_398536

theorem sum_of_even_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 = a₀ + a₁*(1+x) + a₂*(1+x)^2 + a₃*(1+x)^3 + a₄*(1+x)^4 + a₅*(1+x)^5) →
  a₀ + a₂ + a₄ = -16 := by
sorry

end sum_of_even_coefficients_l3985_398536


namespace preimage_of_3_1_l3985_398547

/-- The mapping f: ℝ² → ℝ² defined by f(x,y) = (x+y, x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

/-- Theorem stating that (2,1) is the pre-image of (3,1) under the mapping f -/
theorem preimage_of_3_1 : f (2, 1) = (3, 1) := by sorry

end preimage_of_3_1_l3985_398547


namespace complex_product_real_imag_parts_l3985_398530

theorem complex_product_real_imag_parts : ∃ (a b : ℝ), 
  (Complex.mk a b = (2 * Complex.I - 1) / Complex.I) ∧ (a * b = 2) := by
  sorry

end complex_product_real_imag_parts_l3985_398530


namespace sequence_length_730_l3985_398548

/-- Given a sequence of real numbers satisfying certain conditions, prove that the length of the sequence is 730. -/
theorem sequence_length_730 (n : ℕ+) (b : ℕ → ℝ) : 
  b 0 = 45 → 
  b 1 = 81 → 
  b n = 0 → 
  (∀ k : ℕ, 1 ≤ k ∧ k < n → b (k + 1) = b (k - 1) - 5 / b k) → 
  n = 730 := by
  sorry

end sequence_length_730_l3985_398548


namespace polygon_diagonals_l3985_398589

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem polygon_diagonals (m n : ℕ) : 
  m + n = 33 →
  diagonals m + diagonals n = 243 →
  max m n = 21 →
  diagonals (max m n) = 189 := by
sorry

end polygon_diagonals_l3985_398589


namespace cos_pi_sixth_minus_alpha_l3985_398587

theorem cos_pi_sixth_minus_alpha (α : ℝ) (h : Real.sin (α + π / 3) = 1 / 2) :
  Real.cos (π / 6 - α) = 1 / 2 := by
  sorry

end cos_pi_sixth_minus_alpha_l3985_398587


namespace vector_magnitude_l3985_398576

/-- Given two vectors a and b in ℝ², prove that |b| = √3 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  (let angle := Real.pi / 3
   let a_x := 1
   let a_y := Real.sqrt 2
   a = (a_x, a_y) ∧ 
   Real.cos angle = a.1 * b.1 + a.2 * b.2 / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) ∧
   (a.1 * (a.1 - 2 * b.1) + a.2 * (a.2 - 2 * b.2) = 0)) →
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 3 := by
sorry

end vector_magnitude_l3985_398576


namespace rational_solutions_quadratic_l3985_398568

theorem rational_solutions_quadratic (m : ℕ+) : 
  (∃ x : ℚ, m * x^2 + 25 * x + m = 0) ↔ (m = 10 ∨ m = 12) :=
by sorry

end rational_solutions_quadratic_l3985_398568


namespace chord_intersection_probability_2010_l3985_398567

/-- Given a circle with n distinct points evenly placed around it,
    this function returns the probability that when four distinct points
    are randomly chosen, the chord formed by two of these points
    intersects the chord formed by the other two points. -/
def chord_intersection_probability (n : ℕ) : ℚ :=
  1 / 3

/-- Theorem stating that for a circle with 2010 distinct points,
    the probability of chord intersection is 1/3 -/
theorem chord_intersection_probability_2010 :
  chord_intersection_probability 2010 = 1 / 3 := by
  sorry

end chord_intersection_probability_2010_l3985_398567


namespace directrix_of_hyperbola_l3985_398581

/-- The directrix of the hyperbola xy = 1 -/
def directrix_equation (x y : ℝ) : Prop :=
  y = -x + Real.sqrt 2 ∨ y = -x - Real.sqrt 2

/-- The hyperbola equation xy = 1 -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x * y = 1

/-- Theorem stating that the directrix of the hyperbola xy = 1 has the equation y = -x ± √2 -/
theorem directrix_of_hyperbola (x y : ℝ) :
  hyperbola_equation x y → directrix_equation x y :=
sorry

end directrix_of_hyperbola_l3985_398581


namespace quadratic_equation_proof_l3985_398508

theorem quadratic_equation_proof :
  ∃ (x y : ℝ), x + y = 10 ∧ |x - y| = 6 ∧ x^2 - 10*x + 16 = 0 ∧ y^2 - 10*y + 16 = 0 := by
  sorry

end quadratic_equation_proof_l3985_398508


namespace camryn_trumpet_practice_l3985_398506

/-- Represents the number of days between Camryn's practices for each instrument -/
structure PracticeSchedule where
  trumpet : ℕ
  flute : ℕ

/-- Checks if the practice schedule satisfies the given conditions -/
def is_valid_schedule (schedule : PracticeSchedule) : Prop :=
  schedule.flute = 3 ∧
  schedule.trumpet > 1 ∧
  schedule.trumpet < 33 ∧
  Nat.lcm schedule.trumpet schedule.flute = 33

theorem camryn_trumpet_practice (schedule : PracticeSchedule) :
  is_valid_schedule schedule → schedule.trumpet = 11 := by
  sorry

#check camryn_trumpet_practice

end camryn_trumpet_practice_l3985_398506


namespace gcd_problem_l3985_398578

theorem gcd_problem (a b c : ℕ) : 
  a * b * c = 2^4 * 3^2 * 5^3 →
  Nat.gcd a b = 15 →
  Nat.gcd a c = 5 →
  Nat.gcd b c = 20 →
  (a = 15 ∧ b = 60 ∧ c = 20) := by
sorry

end gcd_problem_l3985_398578


namespace line_passes_through_fixed_point_l3985_398538

theorem line_passes_through_fixed_point 
  (a b c : ℝ) 
  (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_sum : 1/a + 1/b = 1/c) : 
  ∃ (x y : ℝ), x/a + y/b = 1 ∧ x = c ∧ y = c :=
by sorry


end line_passes_through_fixed_point_l3985_398538


namespace electricity_cost_for_1200_watts_l3985_398570

/-- Calculates the total cost of electricity usage based on tiered pricing, late fees, and discounts --/
def calculate_electricity_cost (usage : ℕ) : ℚ :=
  let tier1_limit : ℕ := 300
  let tier2_limit : ℕ := 800
  let tier1_rate : ℚ := 4
  let tier2_rate : ℚ := 3.5
  let tier3_rate : ℚ := 3
  let late_fee_tier1 : ℚ := 150
  let late_fee_tier2 : ℚ := 200
  let late_fee_tier3 : ℚ := 250
  let discount_lower : ℕ := 900
  let discount_upper : ℕ := 1100

  let tier1_cost := min usage tier1_limit * tier1_rate
  let tier2_cost := max 0 (min usage tier2_limit - tier1_limit) * tier2_rate
  let tier3_cost := max 0 (usage - tier2_limit) * tier3_rate

  let total_electricity_cost := tier1_cost + tier2_cost + tier3_cost

  let late_fee := 
    if usage ≤ 600 then late_fee_tier1
    else if usage ≤ 1000 then late_fee_tier2
    else late_fee_tier3

  let total_cost := total_electricity_cost + late_fee

  -- No discount applied as usage is not in the 3rd highest quartile

  total_cost

theorem electricity_cost_for_1200_watts :
  calculate_electricity_cost 1200 = 4400 := by
  sorry

end electricity_cost_for_1200_watts_l3985_398570


namespace system_solutions_l3985_398511

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  x + y + z = 17 ∧ x*y + y*z + z*x = 94 ∧ x*y*z = 168

-- Define the set of solutions
def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(1, 4, -12), (1, -12, 4), (4, 1, -12), (4, -12, 1), (-12, 1, 4), (-12, 4, 1)}

-- Theorem statement
theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ (x, y, z) ∈ solution_set :=
sorry

end system_solutions_l3985_398511


namespace dice_probability_l3985_398550

def num_dice : ℕ := 5
def num_faces : ℕ := 12
def num_divisible_by_three : ℕ := 4  -- 3, 6, 9, 12 are divisible by 3

def prob_divisible_by_three : ℚ := num_divisible_by_three / num_faces
def prob_not_divisible_by_three : ℚ := 1 - prob_divisible_by_three

def exactly_three_divisible_probability : ℚ :=
  (Nat.choose num_dice 3 : ℚ) * 
  (prob_divisible_by_three ^ 3) * 
  (prob_not_divisible_by_three ^ 2)

theorem dice_probability : 
  exactly_three_divisible_probability = 40 / 243 := by
  sorry

end dice_probability_l3985_398550


namespace complex_modulus_one_l3985_398517

theorem complex_modulus_one (z : ℂ) (h : 3 * z^6 + 2 * Complex.I * z^5 - 2 * z - 3 * Complex.I = 0) : Complex.abs z = 1 := by
  sorry

end complex_modulus_one_l3985_398517
