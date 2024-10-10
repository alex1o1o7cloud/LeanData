import Mathlib

namespace parabola_properties_l1654_165458

/-- Parabola intersecting x-axis -/
def parabola (m : ℝ) (x : ℝ) : ℝ := x^2 - (m^2 + 4) * x - 2 * m^2 - 12

/-- Discriminant of the parabola -/
def discriminant (m : ℝ) : ℝ := (m^2 + 4)^2 + 4 * (2 * m^2 + 12)

/-- Chord length of the parabola intersecting x-axis -/
def chord_length (m : ℝ) : ℝ := m^2 + 8

theorem parabola_properties (m : ℝ) :
  (∀ m, discriminant m > 0) ∧
  (chord_length m = m^2 + 8) ∧
  (∀ m, chord_length m ≥ 8) ∧
  (chord_length 0 = 8) := by
  sorry

end parabola_properties_l1654_165458


namespace maddie_purchase_cost_l1654_165456

/-- Calculates the total cost of Maddie's beauty product purchase --/
def calculate_total_cost (
  palette_price : ℝ)
  (palette_count : ℕ)
  (palette_discount : ℝ)
  (lipstick_price : ℝ)
  (lipstick_count : ℕ)
  (hair_color_price : ℝ)
  (hair_color_count : ℕ)
  (hair_color_discount : ℝ)
  (sales_tax_rate : ℝ) : ℝ :=
  let palette_cost := palette_price * palette_count * (1 - palette_discount)
  let lipstick_cost := lipstick_price * (lipstick_count - 1)
  let hair_color_cost := hair_color_price * hair_color_count * (1 - hair_color_discount)
  let subtotal := palette_cost + lipstick_cost + hair_color_cost
  let total := subtotal * (1 + sales_tax_rate)
  total

/-- Theorem stating that the total cost of Maddie's purchase is $58.64 --/
theorem maddie_purchase_cost :
  calculate_total_cost 15 3 0.2 2.5 4 4 3 0.1 0.08 = 58.64 := by
  sorry

end maddie_purchase_cost_l1654_165456


namespace cube_volume_from_surface_area_l1654_165454

/-- Given a cube with surface area 150 square inches, its volume is 125 cubic inches -/
theorem cube_volume_from_surface_area :
  ∀ (edge_length : ℝ),
  (6 * edge_length^2 = 150) →
  edge_length^3 = 125 :=
by
  sorry

end cube_volume_from_surface_area_l1654_165454


namespace g_monotone_decreasing_l1654_165485

/-- The function g(x) defined in terms of the parameter a -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * (1 - a) * x^2 - 3 * a * x

/-- The derivative of g(x) with respect to x -/
def g' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 4 * (1 - a) * x - 3 * a

/-- Theorem stating the conditions for g(x) to be monotonically decreasing -/
theorem g_monotone_decreasing (a : ℝ) : 
  (∀ x : ℝ, x < a / 3 → g' a x ≤ 0) ↔ a ∈ Set.Iic (-1) ∪ {0} :=
sorry

end g_monotone_decreasing_l1654_165485


namespace inequality_implication_l1654_165484

theorem inequality_implication (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1)^2) :
  y * (y - 1) ≤ x^2 := by
  sorry

end inequality_implication_l1654_165484


namespace parabola_intersection_length_l1654_165493

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (A B : ℝ × ℝ) : Prop :=
  (A.1 - focus.1) * (B.2 - focus.2) = (B.1 - focus.1) * (A.2 - focus.2)

-- Define the condition that A and B are on the parabola
def points_on_parabola (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2

-- Define the sum of x-coordinates condition
def sum_of_x_coordinates (A B : ℝ × ℝ) : Prop :=
  A.1 + B.1 = 3

-- The main theorem
theorem parabola_intersection_length (A B : ℝ × ℝ) :
  line_through_focus A B →
  points_on_parabola A B →
  sum_of_x_coordinates A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5 := by sorry

end parabola_intersection_length_l1654_165493


namespace race_distance_difference_l1654_165498

theorem race_distance_difference (race_distance : ℝ) (a_time b_time : ℝ) : 
  race_distance = 80 →
  a_time = 20 →
  b_time = 25 →
  let a_speed := race_distance / a_time
  let b_speed := race_distance / b_time
  let b_distance := b_speed * a_time
  race_distance - b_distance = 16 := by sorry

end race_distance_difference_l1654_165498


namespace intersection_range_l1654_165492

/-- Given two curves C₁ and C₂, prove the range of m for which they intersect at exactly one point above the x-axis. -/
theorem intersection_range (a : ℝ) (h_a : a > 0) :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (x^2 / a^2 + y^2 = 1 ∧ y^2 = 2*(x + m) ∧ y > 0) →
    (0 < a ∧ a < 1 → (m = (a^2 + 1) / 2 ∨ (-a < m ∧ m ≤ a))) ∧
    (a ≥ 1 → (-a < m ∧ m < a)) :=
by sorry

end intersection_range_l1654_165492


namespace geometric_sequence_ratio_l1654_165481

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The property that three terms form an arithmetic sequence -/
def ArithmeticSequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  ArithmeticSequence (3 * a 1) ((1/2) * a 3) (2 * a 2) →
  (a 2014 + a 2015) / (a 2012 + a 2013) = 9 := by
  sorry

end geometric_sequence_ratio_l1654_165481


namespace glee_club_gender_ratio_l1654_165480

/-- Given a glee club with total members and female members, 
    prove the ratio of female to male members -/
theorem glee_club_gender_ratio (total : ℕ) (female : ℕ) 
    (h1 : total = 18) (h2 : female = 12) :
    (female : ℚ) / ((total - female) : ℚ) = 2 / 1 := by
  sorry

end glee_club_gender_ratio_l1654_165480


namespace jenny_reads_three_books_l1654_165427

/-- Represents the number of books Jenny can read given the conditions --/
def books_jenny_can_read (days : ℕ) (reading_speed : ℕ) (reading_time : ℚ) 
  (book1_words : ℕ) (book2_words : ℕ) (book3_words : ℕ) : ℕ :=
  let total_words := book1_words + book2_words + book3_words
  let total_reading_hours := (days : ℚ) * reading_time
  let words_read := (reading_speed : ℚ) * total_reading_hours
  if words_read ≥ total_words then 3 else 
    if words_read ≥ book1_words + book2_words then 2 else
      if words_read ≥ book1_words then 1 else 0

/-- Theorem stating that Jenny can read exactly 3 books in 10 days --/
theorem jenny_reads_three_books : 
  books_jenny_can_read 10 100 (54/60) 200 400 300 = 3 := by
  sorry

end jenny_reads_three_books_l1654_165427


namespace extremum_derivative_zero_not_sufficient_l1654_165470

-- Define a differentiable function f on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define what it means for a point to be an extremum
def IsExtremum (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ ∀ x, f x ≥ f x₀

-- Theorem statement
theorem extremum_derivative_zero
  (x₀ : ℝ)
  (h_extremum : IsExtremum f x₀) :
  deriv f x₀ = 0 :=
sorry

-- Counter-example to show the converse is not always true
def counter_example : ℝ → ℝ := fun x ↦ x^3

theorem not_sufficient
  (h_deriv_zero : deriv counter_example 0 = 0)
  (h_not_extremum : ¬ IsExtremum counter_example 0) :
  ∃ f : ℝ → ℝ, ∃ x₀ : ℝ, deriv f x₀ = 0 ∧ ¬ IsExtremum f x₀ :=
sorry

end extremum_derivative_zero_not_sufficient_l1654_165470


namespace difference_of_squares_l1654_165431

theorem difference_of_squares : (635 : ℕ)^2 - (365 : ℕ)^2 = 270000 := by
  sorry

end difference_of_squares_l1654_165431


namespace tube_length_doubles_pressure_l1654_165475

/-- The length of a vertical tube required to double the pressure at the bottom of a water-filled barrel -/
theorem tube_length_doubles_pressure 
  (h₁ : ℝ) -- Initial height of water in the barrel
  (m : ℝ) -- Mass of water in the barrel
  (a : ℝ) -- Cross-sectional area of the tube
  (ρ : ℝ) -- Density of water
  (g : ℝ) -- Acceleration due to gravity
  (h₁_val : h₁ = 1.5) -- Given height of the barrel
  (m_val : m = 1000) -- Given mass of water
  (a_val : a = 1e-4) -- Given cross-sectional area (1 cm² = 1e-4 m²)
  (ρ_val : ρ = 1000) -- Given density of water
  : ∃ (h₂ : ℝ), h₂ = h₁ ∧ ρ * g * (h₁ + h₂) = 2 * (ρ * g * h₁) :=
by sorry

end tube_length_doubles_pressure_l1654_165475


namespace unique_solution_condition_smallest_divisor_double_factorial_divides_sum_double_factorial_l1654_165444

-- Definition of double factorial
def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

-- Theorem for the first part of the problem
theorem unique_solution_condition (a : ℝ) :
  (∃! x y : ℝ, x + y - 144 = 0 ∧ x * y - 5184 - 0.1 * a^2 = 0) ↔ a = 0 := by sorry

-- Theorem for the second part of the problem
theorem smallest_divisor_double_factorial :
  ∀ n : ℕ, n > 2022 → n ∣ (double_factorial 2021 + double_factorial 2022) → n ≥ 2023 := by sorry

-- Theorem that 2023 divides the sum of double factorials
theorem divides_sum_double_factorial :
  2023 ∣ (double_factorial 2021 + double_factorial 2022) := by sorry

end unique_solution_condition_smallest_divisor_double_factorial_divides_sum_double_factorial_l1654_165444


namespace optimal_sale_l1654_165432

/-- Represents the selling price and number of items that maximize profit while meeting constraints --/
def OptimalSale : Type := ℕ × ℕ

/-- Calculates the number of items sold given a selling price --/
def itemsSold (initialItems : ℕ) (initialPrice : ℕ) (newPrice : ℕ) : ℕ :=
  initialItems - 10 * (newPrice - initialPrice)

/-- Calculates the total cost given a number of items and cost per item --/
def totalCost (items : ℕ) (costPerItem : ℕ) : ℕ := items * costPerItem

/-- Calculates the profit given selling price, cost per item, and number of items sold --/
def profit (sellingPrice : ℕ) (costPerItem : ℕ) (itemsSold : ℕ) : ℕ :=
  (sellingPrice - costPerItem) * itemsSold

/-- Theorem stating the optimal selling price and number of items to purchase --/
theorem optimal_sale (initialItems : ℕ) (initialPrice : ℕ) (costPerItem : ℕ) (targetProfit : ℕ) (maxCost : ℕ)
    (h_initialItems : initialItems = 500)
    (h_initialPrice : initialPrice = 50)
    (h_costPerItem : costPerItem = 40)
    (h_targetProfit : targetProfit = 8000)
    (h_maxCost : maxCost = 10000) :
    ∃ (sale : OptimalSale),
      let (sellingPrice, itemsToBuy) := sale
      profit sellingPrice costPerItem (itemsSold initialItems initialPrice sellingPrice) = targetProfit ∧
      totalCost itemsToBuy costPerItem < maxCost ∧
      sellingPrice = 80 ∧
      itemsToBuy = 200 := by
  sorry

end optimal_sale_l1654_165432


namespace local_minimum_at_two_l1654_165457

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- State the theorem
theorem local_minimum_at_two :
  ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f x ≥ f 2 :=
sorry

end local_minimum_at_two_l1654_165457


namespace cube_roots_of_negative_one_l1654_165496

theorem cube_roots_of_negative_one :
  let z₁ : ℂ := -1
  let z₂ : ℂ := (1 + Complex.I * Real.sqrt 3) / 2
  let z₃ : ℂ := (1 - Complex.I * Real.sqrt 3) / 2
  (∀ z : ℂ, z^3 = -1 ↔ z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by sorry

end cube_roots_of_negative_one_l1654_165496


namespace quadratic_properties_l1654_165494

-- Define the quadratic function
def quadratic (a b t : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + t - 1

theorem quadratic_properties :
  ∀ (a b t : ℝ), t < 0 →
  -- Part 1
  (quadratic a b (-2) 1 = -4 ∧ quadratic a b (-2) (-1) = 0) → (a = 1 ∧ b = -2) ∧
  -- Part 2
  (2 * a - b = 1) → 
    ∃ (k p : ℝ), k ≠ 0 ∧ 
      ∀ (x : ℝ), (quadratic a b (-2) x = k * x + p) → 
        ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic a b (-2) x1 = k * x1 + p ∧ quadratic a b (-2) x2 = k * x2 + p ∧
  -- Part 3
  ∀ (m n : ℝ), m > 0 ∧ n > 0 →
    quadratic a b t (-1) = t ∧ quadratic a b t m = t - n ∧
    ((1/2) * n - 2 * t = (1/2) * (m + 1) * (quadratic a b t m - quadratic a b t (-1))) →
    (∀ (x : ℝ), -1 ≤ x ∧ x ≤ m → quadratic a b t x ≤ quadratic a b t (-1)) →
    ((0 < a ∧ a ≤ 1/3) ∨ (-1 ≤ a ∧ a < 0)) :=
by sorry

end quadratic_properties_l1654_165494


namespace a_value_l1654_165410

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2

-- Define the derivative of f
def f_prime (a : ℝ) (x : ℝ) : ℝ := 2 * a * x

-- Theorem statement
theorem a_value (a : ℝ) : f_prime a 1 = 4 → a = 2 := by
  sorry

end a_value_l1654_165410


namespace number_satisfying_condition_l1654_165441

theorem number_satisfying_condition : ∃ x : ℤ, (x - 29) / 13 = 15 ∧ x = 224 := by
  sorry

end number_satisfying_condition_l1654_165441


namespace only_vertical_angles_true_l1654_165422

-- Define the propositions
def proposition1 := "Non-intersecting lines are parallel lines"
def proposition2 := "Corresponding angles are equal"
def proposition3 := "If the squares of two real numbers are equal, then the two real numbers are also equal"
def proposition4 := "Vertical angles are equal"

-- Define a function to check if a proposition is true
def is_true (p : String) : Prop :=
  p = proposition4

-- Theorem statement
theorem only_vertical_angles_true :
  (is_true proposition1 = false) ∧
  (is_true proposition2 = false) ∧
  (is_true proposition3 = false) ∧
  (is_true proposition4 = true) :=
by
  sorry


end only_vertical_angles_true_l1654_165422


namespace x_seventh_plus_27x_squared_l1654_165450

theorem x_seventh_plus_27x_squared (x : ℝ) (h : x^3 - 3*x = 7) :
  x^7 + 27*x^2 = 76*x^2 + 270*x + 483 := by
  sorry

end x_seventh_plus_27x_squared_l1654_165450


namespace sock_drawer_theorem_l1654_165404

/-- The minimum number of socks needed to guarantee at least n pairs when selecting from m colors -/
def min_socks_for_pairs (m n : ℕ) : ℕ := m + 1 + 2 * (n - 1)

/-- The number of colors of socks in the drawer -/
def num_colors : ℕ := 4

/-- The number of pairs we want to guarantee -/
def required_pairs : ℕ := 15

theorem sock_drawer_theorem :
  min_socks_for_pairs num_colors required_pairs = 33 :=
sorry

end sock_drawer_theorem_l1654_165404


namespace two_circles_exist_l1654_165468

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions for the circle
def satisfiesConditions (c : Circle) : Prop :=
  let (a, b) := c.center
  -- Circle is tangent to the x-axis
  c.radius = |b| ∧
  -- Center is on the line 3x - y = 0
  3 * a = b ∧
  -- Intersects x - y = 0 to form a chord of length 2√7
  2 * c.radius^2 = (a - b)^2 + 14

-- State the theorem
theorem two_circles_exist :
  ∃ (c1 c2 : Circle),
    satisfiesConditions c1 ∧
    satisfiesConditions c2 ∧
    c1.center = (1, 3) ∧
    c2.center = (-1, -3) ∧
    c1.radius = 3 ∧
    c2.radius = 3 :=
  sorry

end two_circles_exist_l1654_165468


namespace present_age_ratio_l1654_165477

/-- Given A's present age and the future ratio of ages, prove the present ratio of ages -/
theorem present_age_ratio (a b : ℕ) (h1 : a = 15) (h2 : (a + 6) * 5 = (b + 6) * 7) :
  5 * b = 3 * a := by
  sorry

end present_age_ratio_l1654_165477


namespace cream_fraction_is_three_tenths_l1654_165463

-- Define the initial contents of the cups
def initial_A : ℚ := 8
def initial_B : ℚ := 6
def initial_C : ℚ := 4

-- Define the transfer fractions
def transfer_A_to_B : ℚ := 1/3
def transfer_B_to_A : ℚ := 1/2
def transfer_A_to_C : ℚ := 1/4
def transfer_C_to_A : ℚ := 1/3

-- Define the function to calculate the final fraction of cream in Cup A
def final_cream_fraction (
  initial_A initial_B initial_C : ℚ
) (
  transfer_A_to_B transfer_B_to_A transfer_A_to_C transfer_C_to_A : ℚ
) : ℚ :=
  sorry -- The actual calculation would go here

-- Theorem statement
theorem cream_fraction_is_three_tenths :
  final_cream_fraction
    initial_A initial_B initial_C
    transfer_A_to_B transfer_B_to_A transfer_A_to_C transfer_C_to_A
  = 3/10 := by
  sorry

end cream_fraction_is_three_tenths_l1654_165463


namespace square_inequality_l1654_165406

theorem square_inequality (α : ℝ) (x : ℝ) (h1 : α ≥ 0) (h2 : (x + 1)^2 ≥ α * (α + 1)) :
  x^2 ≥ α * (α - 1) := by
sorry

end square_inequality_l1654_165406


namespace arccos_cos_three_l1654_165405

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 := by
  sorry

end arccos_cos_three_l1654_165405


namespace candy_bar_distribution_l1654_165407

theorem candy_bar_distribution (total_candy_bars : ℝ) (num_people : ℝ) 
  (h1 : total_candy_bars = 5.0) 
  (h2 : num_people = 3.0) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |total_candy_bars / num_people - 1.67| < ε :=
sorry

end candy_bar_distribution_l1654_165407


namespace caterpillars_left_tree_l1654_165491

/-- Proves that the number of caterpillars that left the tree is 8 --/
theorem caterpillars_left_tree (initial : ℕ) (hatched : ℕ) (final : ℕ) : 
  initial = 14 → hatched = 4 → final = 10 → initial + hatched - final = 8 := by
  sorry

end caterpillars_left_tree_l1654_165491


namespace rationalize_and_product_l1654_165417

theorem rationalize_and_product : ∃ (A B C : ℤ),
  (((2 : ℝ) + Real.sqrt 5) / ((3 : ℝ) - Real.sqrt 5) = (A : ℝ) / 4 + (B : ℝ) / 4 * Real.sqrt (C : ℝ)) ∧
  A * B * C = 275 := by
  sorry

end rationalize_and_product_l1654_165417


namespace mod_congruence_problem_l1654_165471

theorem mod_congruence_problem (n : ℕ) : 
  (123^2 * 947) % 60 = n ∧ 0 ≤ n ∧ n < 60 → n = 3 := by
  sorry

end mod_congruence_problem_l1654_165471


namespace system_of_equations_solutions_l1654_165420

theorem system_of_equations_solutions :
  -- System 1
  (∃ x y : ℚ, x - 2*y = 0 ∧ 3*x + 2*y = 8 ∧ x = 2 ∧ y = 1) ∧
  -- System 2
  (∃ x y : ℚ, 3*x - 5*y = 9 ∧ 2*x + 3*y = -6 ∧ x = -3/19 ∧ y = -36/19) := by
  sorry


end system_of_equations_solutions_l1654_165420


namespace solve_equation_1_solve_equation_2_l1654_165437

-- First equation
theorem solve_equation_1 : 
  ∃ x : ℚ, 15 - (7 - 5 * x) = 2 * x + (5 - 3 * x) ↔ x = -1/2 := by sorry

-- Second equation
theorem solve_equation_2 : 
  ∃ x : ℚ, (x - 3) / 2 - (2 * x - 3) / 5 = 1 ↔ x = 19 := by sorry

end solve_equation_1_solve_equation_2_l1654_165437


namespace brendas_age_l1654_165497

theorem brendas_age (addison janet brenda : ℚ) 
  (h1 : addison = 4 * brenda) 
  (h2 : janet = brenda + 8) 
  (h3 : addison = janet) : 
  brenda = 8/3 := by
sorry

end brendas_age_l1654_165497


namespace square_sum_proof_l1654_165473

theorem square_sum_proof (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end square_sum_proof_l1654_165473


namespace economics_law_tournament_l1654_165445

theorem economics_law_tournament (n : ℕ) (m : ℕ) : 
  220 < n → n < 254 →
  m < n →
  (n - 2 * m)^2 = n →
  ∀ k : ℕ, (220 < k ∧ k < 254 ∧ k < n ∧ (k - 2 * (n - k))^2 = k) → n - m ≤ k - (n - k) →
  n - m = 105 := by
sorry

end economics_law_tournament_l1654_165445


namespace sum_of_squares_and_product_l1654_165446

theorem sum_of_squares_and_product (x y : ℝ) : 
  x = 2 / (Real.sqrt 3 + 1) →
  y = 2 / (Real.sqrt 3 - 1) →
  x^2 + x*y + y^2 = 10 := by sorry

end sum_of_squares_and_product_l1654_165446


namespace tan_negative_3645_degrees_l1654_165416

theorem tan_negative_3645_degrees : Real.tan ((-3645 : ℝ) * π / 180) = -1 := by
  sorry

end tan_negative_3645_degrees_l1654_165416


namespace remaining_soup_feeds_twenty_adults_l1654_165476

/-- Represents the number of adults a can of soup can feed -/
def adults_per_can : ℕ := 4

/-- Represents the number of children a can of soup can feed -/
def children_per_can : ℕ := 7

/-- Represents the total number of cans of soup -/
def total_cans : ℕ := 10

/-- Represents the number of children fed -/
def children_fed : ℕ := 35

/-- Calculates the number of adults that can be fed with the remaining soup -/
def adults_fed_with_remaining_soup : ℕ := 
  adults_per_can * (total_cans - (children_fed / children_per_can))

theorem remaining_soup_feeds_twenty_adults : 
  adults_fed_with_remaining_soup = 20 := by sorry

end remaining_soup_feeds_twenty_adults_l1654_165476


namespace certain_number_proof_l1654_165487

theorem certain_number_proof (x y z N : ℤ) : 
  x < y → y < z →
  y - x > N →
  Even x →
  Odd y →
  Odd z →
  (∀ w, w - x ≥ 13 → w ≥ z) →
  (∃ u v, u < v ∧ v < z ∧ v - u > N ∧ Even u ∧ Odd v ∧ v - x < 13) →
  N ≤ 10 :=
sorry

end certain_number_proof_l1654_165487


namespace unused_sector_angle_l1654_165419

/-- Given a circular piece of paper with radius r, from which a cone is formed
    with base radius 10 cm and volume 500π cm³, prove that the central angle
    of the unused sector is approximately 130.817°. -/
theorem unused_sector_angle (r : ℝ) : 
  r > 0 →
  (1 / 3 * π * 10^2 * (r^2 - 10^2).sqrt = 500 * π) →
  abs (360 - (20 * π / (2 * π * r)) * 360 - 130.817) < 0.001 := by
sorry


end unused_sector_angle_l1654_165419


namespace first_year_after_2020_with_sum_4_l1654_165409

/-- Sum of digits of a number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Check if a year is after 2020 and has sum of digits equal to 4 -/
def isValidYear (year : ℕ) : Prop :=
  year > 2020 ∧ sumOfDigits year = 4

/-- 2022 is the first year after 2020 with sum of digits equal to 4 -/
theorem first_year_after_2020_with_sum_4 :
  (∀ y : ℕ, y < 2022 → ¬(isValidYear y)) ∧ isValidYear 2022 := by
  sorry

#eval sumOfDigits 2020  -- Should output 4
#eval sumOfDigits 2022  -- Should output 4

end first_year_after_2020_with_sum_4_l1654_165409


namespace equation_linear_implies_a_equals_one_l1654_165453

theorem equation_linear_implies_a_equals_one (a : ℝ) :
  (∀ x, (a^2 - 1) * x^2 - a*x - x + 2 = 0 → ∃ m b, (a^2 - 1) * x^2 - a*x - x + 2 = m*x + b) →
  a = 1 :=
by sorry

end equation_linear_implies_a_equals_one_l1654_165453


namespace zeros_in_square_of_nines_l1654_165447

/-- The number of zeros in the decimal expansion of (10^8 - 1)² is 7 -/
theorem zeros_in_square_of_nines : ∃ n : ℕ, n = 7 ∧ 
  (∃ m : ℕ, (10^8 - 1)^2 = m * 10^n + k ∧ k < 10^n ∧ k % 10 ≠ 0) := by
  sorry

end zeros_in_square_of_nines_l1654_165447


namespace dandelion_picking_l1654_165411

theorem dandelion_picking (billy_initial : ℕ) (george_initial : ℕ) (average : ℕ) : 
  billy_initial = 36 →
  george_initial = billy_initial / 3 →
  average = 34 →
  (billy_initial + george_initial + 2 * (average - (billy_initial + george_initial) / 2)) / 2 = average →
  average - (billy_initial + george_initial) / 2 = 10 :=
by sorry

end dandelion_picking_l1654_165411


namespace intersection_of_three_lines_l1654_165462

theorem intersection_of_three_lines (k : ℝ) : 
  (∃ x y : ℝ, y = 7 * x - 2 ∧ y = -3 * x + 14 ∧ y = 4 * x + k) → k = 2.8 := by
  sorry

end intersection_of_three_lines_l1654_165462


namespace straight_line_probability_l1654_165495

/-- The number of dots in each row or column of the grid -/
def gridSize : ℕ := 5

/-- The total number of dots in the grid -/
def totalDots : ℕ := gridSize * gridSize

/-- The number of dots required to form a line -/
def dotsInLine : ℕ := 4

/-- The number of possible straight lines containing four dots in a 5x5 grid -/
def numStraightLines : ℕ := 16

/-- The total number of ways to choose 4 dots from 25 dots -/
def totalWaysToChoose : ℕ := Nat.choose totalDots dotsInLine

/-- The probability of selecting four dots that form a straight line -/
def probabilityOfStraightLine : ℚ := numStraightLines / totalWaysToChoose

theorem straight_line_probability :
  probabilityOfStraightLine = 16 / 12650 := by sorry

end straight_line_probability_l1654_165495


namespace min_n_is_correct_l1654_165415

/-- The minimum positive integer n for which (x^5 + 1/x)^n contains a constant term -/
def min_n : ℕ := 6

/-- Predicate to check if (x^5 + 1/x)^n contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ r : ℕ, 5 * n = 6 * r

theorem min_n_is_correct :
  (has_constant_term min_n) ∧
  (∀ m : ℕ, m < min_n → ¬(has_constant_term m)) :=
by sorry

end min_n_is_correct_l1654_165415


namespace consecutive_integers_transformation_l1654_165489

/-- Sum of squares of first m positive integers -/
def sum_of_squares (m : ℕ) : ℕ := m * (m + 1) * (2 * m + 1) / 6

/-- Sum of squares of 2n consecutive integers starting from k -/
def consecutive_sum_of_squares (n k : ℕ) : ℕ :=
  sum_of_squares (k + 2*n - 1) - sum_of_squares (k - 1)

theorem consecutive_integers_transformation (n : ℕ) :
  ∀ k m : ℕ, ∃ t : ℕ, 2^t * consecutive_sum_of_squares n 1 ≠ consecutive_sum_of_squares n k := by
  sorry

end consecutive_integers_transformation_l1654_165489


namespace number_problem_l1654_165465

/-- Given a number N, if N/p = 8, N/q = 18, and p - q = 0.2777777777777778, then N = 4 -/
theorem number_problem (N p q : ℝ) 
  (h1 : N / p = 8)
  (h2 : N / q = 18)
  (h3 : p - q = 0.2777777777777778) : 
  N = 4 := by sorry

end number_problem_l1654_165465


namespace no_single_solution_quadratic_inequality_l1654_165499

theorem no_single_solution_quadratic_inequality :
  ¬ ∃ (b : ℝ), ∃! (x : ℝ), |x^2 + 3*b*x + 4*b| ≤ 5 := by
  sorry

end no_single_solution_quadratic_inequality_l1654_165499


namespace trig_simplification_l1654_165425

theorem trig_simplification (α : Real) :
  Real.sin (-α) * Real.cos (π + α) * Real.tan (2 * π + α) = Real.sin α ^ 2 := by
  sorry

end trig_simplification_l1654_165425


namespace sum_of_max_min_g_l1654_165460

def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |3*x - 15|

theorem sum_of_max_min_g : 
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 3 10, g x ≤ max) ∧ 
    (∃ x ∈ Set.Icc 3 10, g x = max) ∧
    (∀ x ∈ Set.Icc 3 10, min ≤ g x) ∧ 
    (∃ x ∈ Set.Icc 3 10, g x = min) ∧
    max + min = -21 := by
  sorry

end sum_of_max_min_g_l1654_165460


namespace one_third_of_recipe_l1654_165451

theorem one_third_of_recipe (original_amount : ℚ) (reduced_amount : ℚ) : 
  original_amount = 27/4 → reduced_amount = original_amount / 3 → reduced_amount = 9/4 :=
by sorry

end one_third_of_recipe_l1654_165451


namespace min_value_expression_l1654_165478

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 5) :
  ((x + 1) * (2*y + 1)) / Real.sqrt (x*y) ≥ 4 * Real.sqrt 3 :=
sorry

end min_value_expression_l1654_165478


namespace savings_ratio_l1654_165434

/-- Represents the number of cans collected from different sources -/
structure CanCollection where
  home : ℕ
  grandparents : ℕ
  neighbor : ℕ
  office : ℕ

/-- Calculates the total number of cans collected -/
def total_cans (c : CanCollection) : ℕ :=
  c.home + c.grandparents + c.neighbor + c.office

/-- Represents the problem setup -/
structure RecyclingProblem where
  collection : CanCollection
  price_per_can : ℚ
  savings_amount : ℚ

/-- Main theorem: The ratio of savings to total amount collected is 1:2 -/
theorem savings_ratio (p : RecyclingProblem)
  (h1 : p.collection.home = 12)
  (h2 : p.collection.grandparents = 3 * p.collection.home)
  (h3 : p.collection.neighbor = 46)
  (h4 : p.collection.office = 250)
  (h5 : p.price_per_can = 1/4)
  (h6 : p.savings_amount = 43) :
  p.savings_amount / (p.price_per_can * total_cans p.collection) = 1/2 := by
  sorry

end savings_ratio_l1654_165434


namespace calculation_proof_system_of_equations_proof_l1654_165436

-- Part 1: Calculation proof
theorem calculation_proof :
  -2^2 - |2 - Real.sqrt 5| + (8 : ℝ)^(1/3) = -Real.sqrt 5 := by sorry

-- Part 2: System of equations proof
theorem system_of_equations_proof :
  ∃ (x y : ℝ), 2*x + y = 5 ∧ x - 3*y = 6 ∧ x = 3 ∧ y = -1 := by sorry

end calculation_proof_system_of_equations_proof_l1654_165436


namespace problem_statement_l1654_165428

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

-- State the theorem
theorem problem_statement (m : ℝ) :
  (∀ x, f m (x + 2) ≥ 0 ↔ x ∈ Set.Icc (-1 : ℝ) 1) →
  (m = 1 ∧
   ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
     1/a + 1/(2*b) + 1/(3*c) = m →
     a + 2*b + 3*c ≥ 9) :=
by sorry

end problem_statement_l1654_165428


namespace transformations_map_correctly_l1654_165486

-- Define points in 2D space
def C : ℝ × ℝ := (3, -2)
def D : ℝ × ℝ := (4, -5)
def C' : ℝ × ℝ := (-3, 2)
def D' : ℝ × ℝ := (-4, 5)

-- Define translation
def translate (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - 6, p.2 + 4)

-- Define 180° clockwise rotation
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Theorem stating that both transformations map C to C' and D to D'
theorem transformations_map_correctly :
  (translate C = C' ∧ translate D = D') ∧
  (rotate180 C = C' ∧ rotate180 D = D') :=
by sorry

end transformations_map_correctly_l1654_165486


namespace logarithm_expression_equality_l1654_165467

theorem logarithm_expression_equality : 2 * Real.log 2 / Real.log 10 + Real.log (5/8) / Real.log 10 - Real.log 25 / Real.log 10 = -1 := by
  sorry

end logarithm_expression_equality_l1654_165467


namespace circle_equation_proof_l1654_165424

/-- The standard equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
def standard_circle_equation (h k r x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Given a circle with center (1, -2) and radius 6, its standard equation is (x-1)^2 + (y+2)^2 = 36 -/
theorem circle_equation_proof :
  ∀ x y : ℝ, standard_circle_equation 1 (-2) 6 x y ↔ (x - 1)^2 + (y + 2)^2 = 36 := by
  sorry

end circle_equation_proof_l1654_165424


namespace quadratic_inequality_range_l1654_165400

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0) → 
  -2 ≤ a ∧ a < 6/5 := by
sorry

end quadratic_inequality_range_l1654_165400


namespace sum_of_ages_l1654_165442

/-- Represents the ages of two people P and Q -/
structure Ages where
  p : ℝ
  q : ℝ

/-- The condition that P's age is thrice Q's age when P was as old as Q is now -/
def age_relation (ages : Ages) : Prop :=
  ages.p = 3 * (ages.q - (ages.p - ages.q))

/-- Theorem stating the sum of P and Q's ages given the conditions -/
theorem sum_of_ages :
  ∀ (ages : Ages),
    ages.q = 37.5 →
    age_relation ages →
    ages.p + ages.q = 93.75 := by
  sorry


end sum_of_ages_l1654_165442


namespace f_max_at_neg_four_l1654_165402

-- Define the function
def f (x : ℝ) : ℝ := -x^2 - 8*x + 16

-- State the theorem
theorem f_max_at_neg_four :
  ∀ x : ℝ, f x ≤ f (-4) :=
by sorry

end f_max_at_neg_four_l1654_165402


namespace first_term_of_special_arithmetic_sequence_l1654_165466

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of the first 60 terms is 500
  sum_first_60 : (60 : ℚ) / 2 * (2 * a + 59 * d) = 500
  -- Sum of the next 60 terms (61 to 120) is 2900
  sum_next_60 : (60 : ℚ) / 2 * (2 * (a + 60 * d) + 59 * d) = 2900

/-- The first term of the arithmetic sequence with given properties is -34/3 -/
theorem first_term_of_special_arithmetic_sequence (seq : ArithmeticSequence) : seq.a = -34/3 := by
  sorry

end first_term_of_special_arithmetic_sequence_l1654_165466


namespace rachel_treasures_l1654_165472

theorem rachel_treasures (points_per_treasure : ℕ) (second_level_treasures : ℕ) (total_score : ℕ) :
  points_per_treasure = 9 →
  second_level_treasures = 2 →
  total_score = 63 →
  ∃ (first_level_treasures : ℕ),
    first_level_treasures * points_per_treasure + second_level_treasures * points_per_treasure = total_score ∧
    first_level_treasures = 5 :=
by
  sorry

#check rachel_treasures

end rachel_treasures_l1654_165472


namespace smaller_number_in_ratio_l1654_165448

theorem smaller_number_in_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a / b = 3 / 8 → (a - 24) / (b - 24) = 4 / 9 → a = 72 := by
sorry

end smaller_number_in_ratio_l1654_165448


namespace unknown_bill_denomination_l1654_165418

/-- Represents the number of bills of each denomination --/
structure BillCount where
  twenty : Nat
  ten : Nat
  unknown : Nat

/-- Represents the total value of bills --/
def totalValue (b : BillCount) (unknownDenom : Nat) : Nat :=
  20 * b.twenty + 10 * b.ten + unknownDenom * b.unknown

/-- The problem statement --/
theorem unknown_bill_denomination (b : BillCount) (h1 : b.twenty = 10) (h2 : b.ten = 8) (h3 : b.unknown = 4) :
  ∃ (x : Nat), x = 5 ∧ totalValue b x = 300 := by
  sorry

end unknown_bill_denomination_l1654_165418


namespace adlai_has_two_dogs_l1654_165413

/-- The number of legs a dog has -/
def dog_legs : ℕ := 4

/-- The number of legs a chicken has -/
def chicken_legs : ℕ := 2

/-- The total number of animal legs -/
def total_legs : ℕ := 10

/-- The number of chickens Adlai has -/
def num_chickens : ℕ := 1

/-- Theorem stating that Adlai has 2 dogs -/
theorem adlai_has_two_dogs : 
  ∃ (num_dogs : ℕ), num_dogs * dog_legs + num_chickens * chicken_legs = total_legs ∧ num_dogs = 2 :=
sorry

end adlai_has_two_dogs_l1654_165413


namespace necessary_not_sufficient_l1654_165488

theorem necessary_not_sufficient : 
  (∃ x : ℝ, |x - 1| < 2 ∧ ¬(x * (3 - x) > 0)) ∧
  (∀ x : ℝ, x * (3 - x) > 0 → |x - 1| < 2) :=
by sorry

end necessary_not_sufficient_l1654_165488


namespace expression_simplification_l1654_165474

theorem expression_simplification (p : ℤ) :
  ((7 * p + 2) - 3 * p * 2) * 4 + (5 - 2 / 2) * (8 * p - 12) = 36 * p - 40 := by
  sorry

end expression_simplification_l1654_165474


namespace inequality_solution_l1654_165443

theorem inequality_solution (x : ℝ) : 
  (3 * x - 2 ≥ 0) → 
  (|Real.sqrt (3 * x - 2) - 3| > 1 ↔ (x > 6 ∨ (2/3 ≤ x ∧ x < 2))) :=
by sorry

end inequality_solution_l1654_165443


namespace consecutive_integers_sum_of_squares_l1654_165452

theorem consecutive_integers_sum_of_squares (n : ℤ) : 
  n * (n + 1) * (n + 2) = 12 * (3 * n + 3) → 
  n^2 + (n + 1)^2 + (n + 2)^2 = 29 := by
  sorry

end consecutive_integers_sum_of_squares_l1654_165452


namespace trajectory_of_G_l1654_165459

/-- The ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Point P on the ellipse C -/
def point_P (x₀ y₀ : ℝ) : Prop := ellipse_C x₀ y₀

/-- Relation between vectors PG and GO -/
def vector_relation (x₀ y₀ x y : ℝ) : Prop :=
  (x - x₀, y - y₀) = (2 * (-x), 2 * (-y))

/-- Trajectory of point G -/
def trajectory_G (x y : ℝ) : Prop := 9*x^2/4 + 3*y^2 = 1

theorem trajectory_of_G (x₀ y₀ x y : ℝ) :
  point_P x₀ y₀ → vector_relation x₀ y₀ x y → trajectory_G x y := by sorry

end trajectory_of_G_l1654_165459


namespace absolute_value_equation_solution_l1654_165469

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 4| = 3 - x :=
by
  -- Proof goes here
  sorry

end absolute_value_equation_solution_l1654_165469


namespace solution_set_f_leq_5_range_of_m_l1654_165423

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x + 3| + |2*x - 1|

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set_f_leq_5 :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -7/4 ≤ x ∧ x ≤ 3/4} := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, f x < |m - 2|) → (m > 6 ∨ m < -2) := by sorry

end solution_set_f_leq_5_range_of_m_l1654_165423


namespace conical_container_height_l1654_165435

theorem conical_container_height (d : ℝ) (n : ℕ) (h r : ℝ) : 
  d = 64 ∧ n = 4 ∧ (π * d^2 / 4) = n * (π * r * (d / 2)) ∧ h^2 + r^2 = (d / 2)^2 
  → h = 8 * Real.sqrt 15 := by sorry

end conical_container_height_l1654_165435


namespace yardwork_earnings_contribution_l1654_165401

def earnings : List ℕ := [18, 22, 30, 35, 45]
def max_contribution : ℕ := 40
def num_friends : ℕ := 5

theorem yardwork_earnings_contribution :
  let total := (earnings.sum - 45 + max_contribution)
  let equal_share := total / num_friends
  35 - equal_share = 6 := by sorry

end yardwork_earnings_contribution_l1654_165401


namespace square_difference_equals_product_l1654_165449

theorem square_difference_equals_product : (51 + 15)^2 - (51^2 + 15^2) = 1530 := by
  sorry

end square_difference_equals_product_l1654_165449


namespace function_range_condition_l1654_165412

def f (a x : ℝ) : ℝ := (a^2 - 2*a - 3)*x^2 + (a - 3)*x + 1

theorem function_range_condition (a : ℝ) :
  (∀ x, ∃ y, f a x = y) ∧ (∀ y, ∃ x, f a x = y) ↔ a > 3 ∨ a < -1 :=
sorry

end function_range_condition_l1654_165412


namespace all_transformed_points_in_S_l1654_165403

def S : Set ℂ := {z | -1 ≤ z.re ∧ z.re ≤ 1 ∧ -1 ≤ z.im ∧ z.im ≤ 1}

theorem all_transformed_points_in_S :
  ∀ z ∈ S, (1/2 + 1/2*I) * z ∈ S := by
  sorry

end all_transformed_points_in_S_l1654_165403


namespace cubic_equation_complex_root_l1654_165421

theorem cubic_equation_complex_root (k : ℝ) : 
  (∃ z : ℂ, z^3 + 2*(k-1)*z^2 + 9*z + 5*(k-1) = 0 ∧ Complex.abs z = Real.sqrt 5) →
  k = 2 ∨ k = -2/3 := by
sorry

end cubic_equation_complex_root_l1654_165421


namespace craig_travel_difference_l1654_165490

theorem craig_travel_difference : 
  let bus_distance : ℝ := 3.83
  let walk_distance : ℝ := 0.17
  bus_distance - walk_distance = 3.66 := by
  sorry

end craig_travel_difference_l1654_165490


namespace M_equals_interval_inequality_holds_l1654_165483

def f (x : ℝ) := |x + 2| + |x - 2|

def M : Set ℝ := {x | f x ≤ 6}

theorem M_equals_interval : M = Set.Icc (-3) 3 := by sorry

theorem inequality_holds (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  Real.sqrt 3 * |a + b| ≤ |a * b + 3| := by sorry

end M_equals_interval_inequality_holds_l1654_165483


namespace inequality_properties_l1654_165414

theorem inequality_properties (x y : ℝ) (h : x > y) : x^3 > y^3 ∧ Real.log x > Real.log y := by
  sorry

end inequality_properties_l1654_165414


namespace gcd_factorial_seven_eight_l1654_165408

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end gcd_factorial_seven_eight_l1654_165408


namespace circle_center_radius_sum_l1654_165455

/-- Given a circle D with equation x^2 + 14y + 63 = -y^2 - 12x, 
    where (a, b) is the center and r is the radius, 
    prove that a + b + r = -13 + √22 -/
theorem circle_center_radius_sum (x y a b r : ℝ) : 
  (∀ x y, x^2 + 14*y + 63 = -y^2 - 12*x) →
  ((x - a)^2 + (y - b)^2 = r^2) →
  a + b + r = -13 + Real.sqrt 22 := by
sorry

end circle_center_radius_sum_l1654_165455


namespace purely_imaginary_z_l1654_165479

theorem purely_imaginary_z (α : ℝ) : 
  let z : ℂ := Complex.mk (Real.sin α) (-(1 - Real.cos α))
  z.re = 0 → ∃ k : ℤ, α = (2 * k + 1) * Real.pi := by sorry

end purely_imaginary_z_l1654_165479


namespace spinner_probability_l1654_165464

/-- Represents a game board based on an equilateral triangle -/
structure GameBoard where
  /-- The number of regions formed by altitudes and one median -/
  total_regions : ℕ
  /-- The number of shaded regions -/
  shaded_regions : ℕ
  /-- Ensure the number of shaded regions is less than or equal to the total regions -/
  h_valid : shaded_regions ≤ total_regions

/-- Calculate the probability of landing in a shaded region -/
def probability (board : GameBoard) : ℚ :=
  board.shaded_regions / board.total_regions

/-- The main theorem to be proved -/
theorem spinner_probability (board : GameBoard) 
  (h_total : board.total_regions = 12)
  (h_shaded : board.shaded_regions = 3) : 
  probability board = 1/4 := by
  sorry

end spinner_probability_l1654_165464


namespace solution_count_l1654_165438

/-- The number of distinct ordered pairs of non-negative integers (a, b) that sum to 50 -/
def count_solutions : ℕ := 51

/-- Predicate for valid solutions -/
def is_valid_solution (a b : ℕ) : Prop := a + b = 50

theorem solution_count :
  (∃! (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s ↔ is_valid_solution p.1 p.2) ∧ 
    s.card = count_solutions) :=
sorry

end solution_count_l1654_165438


namespace line_through_two_points_l1654_165426

/-- 
Given two distinct points P₁(x₁, y₁) and P₂(x₂, y₂) in the plane,
the equation (x-x₁)(y₂-y₁) = (y-y₁)(x₂-x₁) represents the line passing through these points.
-/
theorem line_through_two_points (x₁ y₁ x₂ y₂ : ℝ) (h : (x₁, y₁) ≠ (x₂, y₂)) :
  ∀ x y : ℝ, (x - x₁) * (y₂ - y₁) = (y - y₁) * (x₂ - x₁) ↔ 
  ∃ t : ℝ, x = x₁ + t * (x₂ - x₁) ∧ y = y₁ + t * (y₂ - y₁) :=
by sorry

end line_through_two_points_l1654_165426


namespace quadratic_equation_roots_l1654_165429

theorem quadratic_equation_roots (k : ℝ) :
  let f := fun x : ℝ => x^2 - 2*(k-1)*x + k^2
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  (k ≤ 1/2) ∧
  (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁*x₂ + x₁ + x₂ - 1 = 0 → k = -3) :=
by sorry

end quadratic_equation_roots_l1654_165429


namespace total_rent_is_245_l1654_165430

-- Define the oxen-months for each person
def a_oxen_months : ℕ := 10 * 7
def b_oxen_months : ℕ := 12 * 5
def c_oxen_months : ℕ := 15 * 3

-- Define the total oxen-months
def total_oxen_months : ℕ := a_oxen_months + b_oxen_months + c_oxen_months

-- Define c's payment
def c_payment : ℚ := 62.99999999999999

-- Define the cost per oxen-month
def cost_per_oxen_month : ℚ := c_payment / c_oxen_months

-- Theorem to prove
theorem total_rent_is_245 : 
  ∃ (total_rent : ℚ), total_rent = cost_per_oxen_month * total_oxen_months ∧ 
                       total_rent = 245 := by
  sorry

end total_rent_is_245_l1654_165430


namespace blue_fish_ratio_l1654_165439

/-- Given a fish tank with the following properties:
  - The total number of fish is 60.
  - Half of the blue fish have spots.
  - There are 10 blue, spotted fish.
  Prove that the ratio of blue fish to the total number of fish is 1/3. -/
theorem blue_fish_ratio (total_fish : ℕ) (blue_spotted_fish : ℕ) 
  (h1 : total_fish = 60)
  (h2 : blue_spotted_fish = 10)
  (h3 : blue_spotted_fish * 2 = blue_spotted_fish + (total_fish - blue_spotted_fish * 2)) :
  (blue_spotted_fish * 2 : ℚ) / total_fish = 1 / 3 := by
  sorry

end blue_fish_ratio_l1654_165439


namespace binomial_probability_l1654_165440

/-- A random variable following a binomial distribution with parameters n and p -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  X : ℝ → ℝ  -- The random variable

/-- The probability mass function for a binomial distribution -/
def pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem binomial_probability (X : BinomialDistribution 6 (1/2)) :
  pmf 6 (1/2) 3 = 5/16 := by
  sorry

end binomial_probability_l1654_165440


namespace point_on_bisector_implies_a_eq_neg_five_l1654_165461

/-- A point P with coordinates (x, y) is on the bisector of the second and fourth quadrants if x + y = 0 -/
def on_bisector (x y : ℝ) : Prop := x + y = 0

/-- Given that point P (a+3, 7+a) is on the bisector of the second and fourth quadrants, prove that a = -5 -/
theorem point_on_bisector_implies_a_eq_neg_five (a : ℝ) :
  on_bisector (a + 3) (7 + a) → a = -5 := by
  sorry

end point_on_bisector_implies_a_eq_neg_five_l1654_165461


namespace arrangement_count_l1654_165482

/-- The number of distinct arrangements of 9 indistinguishable objects and 3 indistinguishable objects in a row of 12 positions -/
def distinct_arrangements : ℕ := 220

/-- The total number of positions -/
def total_positions : ℕ := 12

/-- The number of indistinguishable objects of the first type (armchairs) -/
def first_object_count : ℕ := 9

/-- The number of indistinguishable objects of the second type (benches) -/
def second_object_count : ℕ := 3

theorem arrangement_count :
  distinct_arrangements = (total_positions.choose second_object_count) :=
by sorry

end arrangement_count_l1654_165482


namespace factor_quadratic_l1654_165433

theorem factor_quadratic (t : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, -6 * x^2 + 17 * x + 7 = k * (x - t)) ↔ 
  (t = (17 + Real.sqrt 457) / 12 ∨ t = (17 - Real.sqrt 457) / 12) :=
by sorry

end factor_quadratic_l1654_165433
