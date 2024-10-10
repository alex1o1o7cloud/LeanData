import Mathlib

namespace specific_triangle_perimeter_l703_70313

/-- Triangle with parallel lines intersecting its interior --/
structure TriangleWithParallelLines where
  -- Side lengths of the original triangle
  PQ : ℝ
  QR : ℝ
  PR : ℝ
  -- Lengths of segments formed by parallel lines intersecting the triangle
  m_P_length : ℝ
  m_Q_length : ℝ
  m_R_length : ℝ

/-- Calculate the perimeter of the inner triangle formed by parallel lines --/
def innerTrianglePerimeter (t : TriangleWithParallelLines) : ℝ :=
  sorry

/-- Theorem statement for the specific triangle problem --/
theorem specific_triangle_perimeter :
  let t : TriangleWithParallelLines := {
    PQ := 150,
    QR := 275,
    PR := 225,
    m_P_length := 65,
    m_Q_length := 55,
    m_R_length := 25
  }
  innerTrianglePerimeter t = 755 := by
    sorry

end specific_triangle_perimeter_l703_70313


namespace triangle_area_from_circle_and_chord_data_l703_70340

/-- Given a circle and a triangle circumscribed around it, this theorem proves
    the area of the triangle based on given measurements. -/
theorem triangle_area_from_circle_and_chord_data (R : ℝ) (chord_length : ℝ) (center_to_chord : ℝ) (perimeter : ℝ)
  (h1 : chord_length = 16)
  (h2 : center_to_chord = 15)
  (h3 : perimeter = 200)
  (h4 : R^2 = center_to_chord^2 + (chord_length/2)^2) :
  R * (perimeter / 2) = 1700 := by
  sorry

end triangle_area_from_circle_and_chord_data_l703_70340


namespace sin_2x_derivative_l703_70309

theorem sin_2x_derivative (x : ℝ) : 
  deriv (fun x => Real.sin (2 * x)) x = 2 * Real.cos (2 * x) := by
  sorry

end sin_2x_derivative_l703_70309


namespace plot_length_is_61_l703_70363

def rectangular_plot_length (breadth : ℝ) (length_difference : ℝ) (fencing_cost_per_meter : ℝ) (total_fencing_cost : ℝ) : ℝ :=
  breadth + length_difference

theorem plot_length_is_61 (breadth : ℝ) :
  let length_difference : ℝ := 22
  let fencing_cost_per_meter : ℝ := 26.50
  let total_fencing_cost : ℝ := 5300
  let length := rectangular_plot_length breadth length_difference fencing_cost_per_meter total_fencing_cost
  let perimeter := 2 * (length + breadth)
  fencing_cost_per_meter * perimeter = total_fencing_cost →
  length = 61 := by
sorry

end plot_length_is_61_l703_70363


namespace binomial_coefficient_20_10_l703_70306

theorem binomial_coefficient_20_10 
  (h1 : Nat.choose 18 8 = 31824)
  (h2 : Nat.choose 18 9 = 48620)
  (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 172822 := by
sorry

end binomial_coefficient_20_10_l703_70306


namespace alyssa_toy_cost_l703_70311

/-- Calculates the total cost of toys with various discounts and special offers -/
def total_cost (football_price marbles_price puzzle_price toy_car_price board_game_price 
                stuffed_animal_price action_figure_price : ℝ) : ℝ :=
  let marbles_discounted := marbles_price * (1 - 0.05)
  let puzzle_discounted := puzzle_price * (1 - 0.10)
  let toy_car_discounted := toy_car_price * (1 - 0.15)
  let stuffed_animals_total := stuffed_animal_price * 1.5
  let action_figures_total := action_figure_price * (1 + 0.4)
  football_price + marbles_discounted + puzzle_discounted + toy_car_discounted + 
  board_game_price + stuffed_animals_total + action_figures_total

/-- Theorem stating the total cost of Alyssa's toys -/
theorem alyssa_toy_cost : 
  total_cost 5.71 6.59 4.25 3.95 10.49 8.99 12.39 = 60.468 := by
  sorry

end alyssa_toy_cost_l703_70311


namespace divisibility_proof_l703_70302

def is_valid_number (r b c : Nat) : Prop :=
  r < 10 ∧ b < 10 ∧ c < 10

def number_value (r b c : Nat) : Nat :=
  523000 + r * 100 + b * 10 + c

theorem divisibility_proof (r b c : Nat) 
  (h1 : is_valid_number r b c) 
  (h2 : r * b * c = 180) 
  (h3 : (number_value r b c) % 89 = 0) : 
  (number_value r b c) % 5886 = 0 := by
sorry

end divisibility_proof_l703_70302


namespace sum_of_vertices_l703_70376

/-- A configuration of numbers on a triangle -/
structure TriangleConfig where
  vertices : Fin 3 → ℕ
  sides : Fin 3 → ℕ
  sum_property : ∀ i : Fin 3, vertices i + sides i + vertices (i + 1) = 17

/-- The set of numbers to be used in the triangle -/
def triangle_numbers : Finset ℕ := {1, 3, 5, 7, 9, 11}

/-- The theorem stating the sum of numbers at the vertices -/
theorem sum_of_vertices (config : TriangleConfig) 
  (h : ∀ n, n ∈ (Finset.image config.vertices Finset.univ ∪ Finset.image config.sides Finset.univ) → n ∈ triangle_numbers) :
  config.vertices 0 + config.vertices 1 + config.vertices 2 = 15 := by
  sorry


end sum_of_vertices_l703_70376


namespace exponent_calculation_l703_70378

theorem exponent_calculation : ((15^15 / 15^14)^3 * 3^3) / 3^3 = 3375 := by
  sorry

end exponent_calculation_l703_70378


namespace set_equality_implies_m_values_l703_70369

def A : Set ℝ := {x | x^2 - 3*x - 10 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x - 1 = 0}

theorem set_equality_implies_m_values (m : ℝ) : A ∪ B m = A → m = 0 ∨ m = -1/2 ∨ m = 1/5 := by
  sorry

end set_equality_implies_m_values_l703_70369


namespace min_value_sum_l703_70368

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : 
  x + 3 * y ≥ 4 + 8 * Real.sqrt 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧ 
    x₀ + 3 * y₀ = 4 + 8 * Real.sqrt 3 := by
  sorry

end min_value_sum_l703_70368


namespace total_amount_proof_l703_70389

/-- Calculates the total amount of money given the number of 50 and 500 rupee notes -/
def totalAmount (n50 : ℕ) (n500 : ℕ) : ℕ := n50 * 50 + n500 * 500

/-- Proves that the total amount of money is 10350 rupees given the specified conditions -/
theorem total_amount_proof :
  let total_notes : ℕ := 108
  let n50 : ℕ := 97
  let n500 : ℕ := total_notes - n50
  totalAmount n50 n500 = 10350 := by
  sorry


end total_amount_proof_l703_70389


namespace quadratic_function_uniqueness_l703_70329

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_uniqueness 
  (f : ℝ → ℝ) 
  (h_quad : is_quadratic f)
  (h_sol : ∀ x : ℝ, f x > 0 ↔ 0 < x ∧ x < 4)
  (h_max : ∀ x : ℝ, x ∈ Set.Icc (-1) 5 → f x ≤ 12)
  (h_attain : ∃ x : ℝ, x ∈ Set.Icc (-1) 5 ∧ f x = 12) :
  ∀ x : ℝ, f x = -3 * x^2 + 12 * x :=
sorry

end quadratic_function_uniqueness_l703_70329


namespace power_zero_eq_one_l703_70332

theorem power_zero_eq_one (a : ℝ) (h : a ≠ 0) : a ^ 0 = 1 := by
  sorry

end power_zero_eq_one_l703_70332


namespace complex_fraction_simplification_l703_70303

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  ((1 + i)^10) / (1 - i) = -16 + 16*i :=
by sorry

end complex_fraction_simplification_l703_70303


namespace correct_purchase_and_savings_l703_70395

/-- Represents the purchase of notebooks by a school -/
structure NotebookPurchase where
  type1 : ℕ  -- number of notebooks of first type
  type2 : ℕ  -- number of notebooks of second type

/-- Calculates the total cost of notebooks without discount -/
def totalCost (purchase : NotebookPurchase) : ℕ :=
  3 * purchase.type1 + 2 * purchase.type2

/-- Calculates the discounted cost of notebooks -/
def discountedCost (purchase : NotebookPurchase) : ℚ :=
  3 * purchase.type1 * (8/10) + 2 * purchase.type2 * (9/10)

/-- Theorem stating the correct purchase and savings -/
theorem correct_purchase_and_savings :
  ∃ (purchase : NotebookPurchase),
    totalCost purchase = 460 ∧
    purchase.type1 = 2 * purchase.type2 + 20 ∧
    purchase.type1 = 120 ∧
    purchase.type2 = 50 ∧
    460 - discountedCost purchase = 82 := by
  sorry


end correct_purchase_and_savings_l703_70395


namespace acid_mixture_theorem_l703_70304

/-- Represents an acid solution with a given concentration and volume -/
structure AcidSolution where
  concentration : ℝ
  volume : ℝ

/-- Calculates the amount of pure acid in a solution -/
def pureAcid (solution : AcidSolution) : ℝ :=
  solution.concentration * solution.volume

/-- Theorem: Mixing 4L of 60% acid with 16L of 75% acid yields 20L of 72% acid -/
theorem acid_mixture_theorem :
  let solution1 : AcidSolution := { concentration := 0.60, volume := 4 }
  let solution2 : AcidSolution := { concentration := 0.75, volume := 16 }
  let finalSolution : AcidSolution := { concentration := 0.72, volume := 20 }
  pureAcid solution1 + pureAcid solution2 = pureAcid finalSolution :=
by sorry

end acid_mixture_theorem_l703_70304


namespace function_upper_bound_l703_70343

theorem function_upper_bound 
  (a r : ℝ) 
  (ha : a > 1) 
  (hr : r > 1) 
  (f : ℝ → ℝ) 
  (hf : ∀ x > 0, f x ^ 2 ≤ a * x * f (x / a))
  (hf_small : ∀ x, 0 < x → x < 1 / 2^2005 → f x < 2^2005) :
  ∀ x > 0, f x ≤ a^(1 - r) * x^r := by
sorry

end function_upper_bound_l703_70343


namespace complex_fraction_sum_l703_70388

theorem complex_fraction_sum (a b : ℝ) : 
  (1 + 2*I) / (Complex.mk a b) = 1 + I → a + b = 2 := by
  sorry

end complex_fraction_sum_l703_70388


namespace complex_modulus_l703_70322

theorem complex_modulus (z : ℂ) : z = (1 + Complex.I) / (2 - Complex.I) → Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end complex_modulus_l703_70322


namespace max_value_x3_minus_y3_l703_70319

theorem max_value_x3_minus_y3 (x y : ℝ) 
  (h1 : 3 * (x^3 + y^3) = x + y) 
  (h2 : x + y = 1) : 
  ∃ (max : ℝ), max = 7/27 ∧ ∀ (a b : ℝ), 3 * (a^3 + b^3) = a + b → a + b = 1 → a^3 - b^3 ≤ max :=
by sorry

end max_value_x3_minus_y3_l703_70319


namespace exists_consecutive_numbers_with_properties_l703_70327

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of two consecutive numbers with given properties -/
theorem exists_consecutive_numbers_with_properties :
  ∃ n : ℕ, sum_of_digits n = 8 ∧ (n + 1) % 8 = 0 := by
  sorry

end exists_consecutive_numbers_with_properties_l703_70327


namespace smallest_square_area_for_two_rectangles_l703_70386

/-- The smallest square area that can contain two non-overlapping rectangles -/
theorem smallest_square_area_for_two_rectangles :
  ∀ (w₁ h₁ w₂ h₂ : ℕ),
    w₁ = 2 ∧ h₁ = 4 ∧ w₂ = 3 ∧ h₂ = 5 →
    ∃ (s : ℕ),
      s^2 = 81 ∧
      ∀ (a : ℕ),
        (a ≥ w₁ ∧ a ≥ h₁ ∧ a ≥ w₂ ∧ a ≥ h₂ ∧ a ≥ w₁ + w₂ ∧ a ≥ h₁ + h₂) →
        a^2 ≥ s^2 :=
by sorry

end smallest_square_area_for_two_rectangles_l703_70386


namespace probability_A_makes_basket_on_kth_shot_l703_70320

/-- The probability that player A takes k shots to make the basket -/
def P (k : ℕ) : ℝ :=
  (0.24 ^ (k - 1)) * 0.4

/-- Theorem stating the probability formula for player A making a basket on the k-th shot -/
theorem probability_A_makes_basket_on_kth_shot (k : ℕ) :
  P k = (0.24 ^ (k - 1)) * 0.4 :=
by
  sorry

#check probability_A_makes_basket_on_kth_shot

end probability_A_makes_basket_on_kth_shot_l703_70320


namespace range_of_k_l703_70350

theorem range_of_k (k : ℝ) : 
  (∀ x : ℝ, |x - 2| + |x - 3| > |k - 1|) ↔ k ∈ Set.Ioo 0 2 :=
by
  sorry

end range_of_k_l703_70350


namespace real_number_inequalities_l703_70393

-- Define the propositions
theorem real_number_inequalities (a b c : ℝ) : 
  -- Proposition A
  ((a * c^2 > b * c^2) → (a > b)) ∧ 
  -- Proposition B (negation)
  (∃ a b c : ℝ, a > b ∧ a * c^2 ≤ b * c^2) ∧ 
  -- Proposition C (negation)
  (∃ a b : ℝ, a > b ∧ 1/a ≥ 1/b) ∧ 
  -- Proposition D
  ((a > b ∧ b > 0) → (a^2 > a*b ∧ a*b > b^2)) :=
by sorry

end real_number_inequalities_l703_70393


namespace calculate_expression_l703_70373

theorem calculate_expression : 2003^3 - 2001^3 - 6 * 2003^2 + 24 * 1001 = -4 := by
  sorry

end calculate_expression_l703_70373


namespace simplify_expression_l703_70396

theorem simplify_expression (x : ℝ) : 
  2*x - 3*(2 - x) + 4*(1 + 3*x) - 5*(1 - x^2) = -5*x^2 + 17*x - 7 := by
  sorry

end simplify_expression_l703_70396


namespace fraction_undefined_values_l703_70339

def undefined_values (a : ℝ) : Prop :=
  a^3 - 4*a = 0

theorem fraction_undefined_values :
  {a : ℝ | undefined_values a} = {-2, 0, 2} := by
  sorry

end fraction_undefined_values_l703_70339


namespace special_function_properties_l703_70394

/-- A function satisfying specific properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, ∀ y > 0, f (x * y) = f x + f y) ∧
  (∀ x > 1, f x < 0) ∧
  (f 3 = -1)

theorem special_function_properties
  (f : ℝ → ℝ)
  (hf : SpecialFunction f) :
  f 1 = 0 ∧
  f (1/9) = 2 ∧
  (∀ x y, x > 0 → y > 0 → x < y → f y < f x) ∧
  (∀ x, f x + f (2 - x) < 2 ↔ 1 - 2 * Real.sqrt 2 / 3 < x ∧ x < 1 + 2 * Real.sqrt 2 / 3) :=
by sorry

end special_function_properties_l703_70394


namespace myrtle_egg_count_l703_70372

/-- The number of eggs Myrtle has after her trip -/
def myrtle_eggs (num_hens : ℕ) (eggs_per_hen : ℕ) (days_gone : ℕ) (neighbor_took : ℕ) (eggs_dropped : ℕ) : ℕ :=
  num_hens * eggs_per_hen * days_gone - neighbor_took - eggs_dropped

/-- Proof that Myrtle has 46 eggs given the conditions -/
theorem myrtle_egg_count : myrtle_eggs 3 3 7 12 5 = 46 := by
  sorry

end myrtle_egg_count_l703_70372


namespace shaded_area_concentric_circles_l703_70377

theorem shaded_area_concentric_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 6) (h₂ : r₂ = 3) :
  let area_triangles := 4 * (1/2 * r₂ * r₂)
  let area_small_sectors := 4 * (1/4 * Real.pi * r₂^2)
  area_triangles + area_small_sectors = 18 + 9 * Real.pi := by
  sorry

end shaded_area_concentric_circles_l703_70377


namespace athul_rowing_problem_l703_70323

/-- Athul's rowing problem -/
theorem athul_rowing_problem 
  (v : ℝ) -- Athul's speed in still water (km/h)
  (d : ℝ) -- Distance rowed upstream (km)
  (h1 : v + 1 = 24 / 4) -- Downstream speed equation
  (h2 : v - 1 = d / 4) -- Upstream speed equation
  : d = 16 := by
  sorry

end athul_rowing_problem_l703_70323


namespace inequality_solution_set_l703_70367

theorem inequality_solution_set (x : ℝ) : 
  (1 + x) * (2 - x) * (3 + x^2) > 0 ↔ -1 < x ∧ x < 2 :=
sorry

end inequality_solution_set_l703_70367


namespace basketball_weight_l703_70325

/-- Given that eight identical basketballs weigh the same as four identical watermelons,
    and one watermelon weighs 32 pounds, prove that one basketball weighs 16 pounds. -/
theorem basketball_weight (watermelon_weight : ℝ) (basketball_weight : ℝ) : 
  watermelon_weight = 32 →
  8 * basketball_weight = 4 * watermelon_weight →
  basketball_weight = 16 := by
sorry

end basketball_weight_l703_70325


namespace correct_total_carrots_l703_70381

/-- The total number of carrots Bianca has after picking, throwing out, and picking again -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) : ℕ :=
  initial - thrown_out + picked_next_day

/-- Theorem stating that the total number of carrots is correct -/
theorem correct_total_carrots (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ)
  (h1 : initial ≥ thrown_out) :
  total_carrots initial thrown_out picked_next_day = initial - thrown_out + picked_next_day :=
by
  sorry

#eval total_carrots 23 10 47  -- Should evaluate to 60

end correct_total_carrots_l703_70381


namespace inverse_variation_problem_l703_70315

-- Define the inverse variation relationship
def inverse_variation (y z : ℝ) : Prop := ∃ k : ℝ, y^2 * Real.sqrt z = k

-- Define the theorem
theorem inverse_variation_problem (y₁ y₂ z₁ z₂ : ℝ) 
  (h1 : inverse_variation y₁ z₁)
  (h2 : y₁ = 3)
  (h3 : z₁ = 4)
  (h4 : y₂ = 6) :
  z₂ = 1/4 := by
sorry

end inverse_variation_problem_l703_70315


namespace max_pens_purchased_l703_70359

/-- Represents the prices and quantities of pens and mechanical pencils -/
structure PriceQuantity where
  pen_price : ℕ
  pencil_price : ℕ
  pen_quantity : ℕ
  pencil_quantity : ℕ

/-- Represents the pricing conditions given in the problem -/
def pricing_conditions (p : PriceQuantity) : Prop :=
  2 * p.pen_price + 5 * p.pencil_price = 75 ∧
  3 * p.pen_price + 2 * p.pencil_price = 85

/-- Represents the promotion and quantity conditions -/
def promotion_conditions (p : PriceQuantity) : Prop :=
  p.pencil_quantity = 2 * p.pen_quantity + 8 ∧
  p.pen_price * p.pen_quantity + p.pencil_price * (p.pencil_quantity - p.pen_quantity) < 670

/-- Theorem stating the maximum number of pens that can be purchased -/
theorem max_pens_purchased (p : PriceQuantity) 
  (h1 : pricing_conditions p) 
  (h2 : promotion_conditions p) : 
  p.pen_quantity ≤ 20 :=
sorry

end max_pens_purchased_l703_70359


namespace sum_of_roots_quadratic_l703_70355

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ - 3 = 0 → 
  x₂^2 - 4*x₂ - 3 = 0 → 
  x₁ + x₂ = 4 := by
sorry

end sum_of_roots_quadratic_l703_70355


namespace relay_arrangements_count_l703_70353

/-- Represents the number of people in the class -/
def class_size : ℕ := 5

/-- Represents the number of people needed for the relay -/
def relay_size : ℕ := 4

/-- Represents the number of options for the first runner -/
def first_runner_options : ℕ := 3

/-- Represents the number of options for the last runner -/
def last_runner_options : ℕ := 2

/-- Calculates the number of relay arrangements given the constraints -/
def relay_arrangements : ℕ := 24

/-- Theorem stating that the number of relay arrangements is 24 -/
theorem relay_arrangements_count : 
  relay_arrangements = 24 := by sorry

end relay_arrangements_count_l703_70353


namespace earnings_difference_is_250_l703_70305

/-- Represents the investment and return ratios for three investors -/
structure InvestmentData where
  a_invest : ℕ
  b_invest : ℕ
  c_invest : ℕ
  a_return : ℕ
  b_return : ℕ
  c_return : ℕ

/-- Calculates the earnings difference between investors B and A -/
def earningsDifference (data : InvestmentData) (total_earnings : ℕ) : ℕ :=
  sorry

/-- Theorem stating the earnings difference between B and A -/
theorem earnings_difference_is_250 :
  let data : InvestmentData := {
    a_invest := 3, b_invest := 4, c_invest := 5,
    a_return := 6, b_return := 5, c_return := 4
  }
  earningsDifference data 7250 = 250 := by sorry

end earnings_difference_is_250_l703_70305


namespace reciprocal_of_recurring_decimal_l703_70354

/-- The decimal representation of the recurring decimal 0.363636... -/
def recurring_decimal : ℚ := 36 / 99

/-- The reciprocal of the common fraction form of 0.363636... -/
def reciprocal : ℚ := 11 / 4

theorem reciprocal_of_recurring_decimal : 
  (recurring_decimal)⁻¹ = reciprocal := by sorry

end reciprocal_of_recurring_decimal_l703_70354


namespace sufficient_unnecessary_condition_for_hyperbola_l703_70348

/-- The equation of a conic section -/
def conic_equation (k x y : ℝ) : Prop :=
  x^2 / (k - 2) + y^2 / (5 - k) = 1

/-- Condition for the equation to represent a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  k < 2 ∨ k > 5

/-- Statement that k < 1 is a sufficient and unnecessary condition for a hyperbola -/
theorem sufficient_unnecessary_condition_for_hyperbola :
  (∀ k, k < 1 → is_hyperbola k) ∧
  ∃ k, is_hyperbola k ∧ ¬(k < 1) :=
by sorry

end sufficient_unnecessary_condition_for_hyperbola_l703_70348


namespace circle_center_and_radius_l703_70392

/-- Given a circle with equation x^2 + y^2 - 2x + 4y + 3 = 0, 
    its center is at (1, -2) and its radius is √2 -/
theorem circle_center_and_radius :
  let f : ℝ × ℝ → ℝ := λ (x, y) => x^2 + y^2 - 2*x + 4*y + 3
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -2) ∧ 
    radius = Real.sqrt 2 ∧
    ∀ (p : ℝ × ℝ), f p = 0 ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 :=
by sorry

end circle_center_and_radius_l703_70392


namespace dog_drying_ratio_l703_70324

/-- The time (in minutes) it takes to dry a short-haired dog -/
def short_hair_time : ℕ := 10

/-- The number of short-haired dogs -/
def num_short_hair : ℕ := 6

/-- The number of full-haired dogs -/
def num_full_hair : ℕ := 9

/-- The total time (in minutes) it takes to dry all dogs -/
def total_time : ℕ := 240

/-- The ratio of time to dry a full-haired dog to a short-haired dog -/
def drying_ratio : ℚ := 2

theorem dog_drying_ratio :
  ∃ (full_hair_time : ℕ),
    full_hair_time = short_hair_time * (drying_ratio.num / drying_ratio.den) ∧
    num_short_hair * short_hair_time + num_full_hair * full_hair_time = total_time :=
by sorry

end dog_drying_ratio_l703_70324


namespace audrey_needs_eight_limes_l703_70345

/-- The number of tablespoons in a cup -/
def tablespoons_per_cup : ℚ := 16

/-- The amount of key lime juice in the original recipe, in cups -/
def original_recipe_juice : ℚ := 1/4

/-- The factor by which Audrey increases the amount of juice -/
def juice_increase_factor : ℚ := 2

/-- The amount of juice one key lime yields, in tablespoons -/
def juice_per_lime : ℚ := 1

/-- Calculates the number of key limes Audrey needs for her pie -/
def key_limes_needed : ℚ :=
  (original_recipe_juice * juice_increase_factor * tablespoons_per_cup) / juice_per_lime

/-- Theorem stating that Audrey needs 8 key limes for her pie -/
theorem audrey_needs_eight_limes : key_limes_needed = 8 := by
  sorry

end audrey_needs_eight_limes_l703_70345


namespace pure_imaginary_square_l703_70358

theorem pure_imaginary_square (a : ℝ) : 
  (∃ b : ℝ, (1 + a * Complex.I)^2 = b * Complex.I) → (a = 1 ∨ a = -1) := by
  sorry

end pure_imaginary_square_l703_70358


namespace sqrt_two_inequality_l703_70351

theorem sqrt_two_inequality (m n : ℕ) (h : (m : ℝ) / n < Real.sqrt 2) :
  (m : ℝ) / n < Real.sqrt 2 * (1 - 1 / (4 * n^2)) :=
by sorry

end sqrt_two_inequality_l703_70351


namespace mirror_area_l703_70380

/-- Given a rectangular frame with outer dimensions 100 cm by 140 cm and a uniform frame width of 12 cm,
    the area of the rectangular mirror that fits exactly inside the frame is 8816 cm². -/
theorem mirror_area (frame_width frame_height frame_thickness : ℕ) 
  (hw : frame_width = 100)
  (hh : frame_height = 140)
  (ht : frame_thickness = 12) :
  (frame_width - 2 * frame_thickness) * (frame_height - 2 * frame_thickness) = 8816 :=
by sorry

end mirror_area_l703_70380


namespace zoo_visitors_l703_70308

theorem zoo_visitors (visitors_saturday : ℕ) (visitors_that_day : ℕ) : 
  visitors_saturday = 3750 →
  visitors_saturday = 3 * visitors_that_day →
  visitors_that_day = 1250 := by
sorry

end zoo_visitors_l703_70308


namespace quadratic_minimum_l703_70364

theorem quadratic_minimum (x : ℝ) :
  let y := 4 * x^2 + 8 * x + 16
  ∀ x', 4 * x'^2 + 8 * x' + 16 ≥ 12 ∧ (4 * (-1)^2 + 8 * (-1) + 16 = 12) := by
  sorry

end quadratic_minimum_l703_70364


namespace perimeter_of_special_region_l703_70321

/-- The perimeter of a region bounded by four arcs, each being three-quarters of a circle
    constructed on the sides of a unit square, is equal to 3π. -/
theorem perimeter_of_special_region : Real := by
  -- Define the side length of the square
  let square_side : Real := 1

  -- Define the radius of each circle (half the side length)
  let circle_radius : Real := square_side / 2

  -- Define the length of a full circle with this radius
  let full_circle_length : Real := 2 * Real.pi * circle_radius

  -- Define the length of three-quarters of this circle
  let arc_length : Real := (3 / 4) * full_circle_length

  -- Define the perimeter as four times the arc length
  let perimeter : Real := 4 * arc_length

  -- Prove that this perimeter equals 3π
  sorry

end perimeter_of_special_region_l703_70321


namespace equilateral_triangle_third_vertex_y_coord_l703_70331

/-- Given an equilateral triangle with two vertices at (0,3) and (10,3),
    prove that the y-coordinate of the third vertex in the first quadrant is 3 + 5√3. -/
theorem equilateral_triangle_third_vertex_y_coord :
  let v1 : ℝ × ℝ := (0, 3)
  let v2 : ℝ × ℝ := (10, 3)
  let side_length : ℝ := 10
  ∃ (x y : ℝ), 
    x > 0 ∧ y > 3 ∧
    (x - 0)^2 + (y - 3)^2 = side_length^2 ∧
    (x - 10)^2 + (y - 3)^2 = side_length^2 ∧
    y = 3 + 5 * Real.sqrt 3 :=
by sorry

end equilateral_triangle_third_vertex_y_coord_l703_70331


namespace used_car_clients_l703_70384

theorem used_car_clients (num_cars : ℕ) (selections_per_car : ℕ) (cars_per_client : ℕ) : 
  num_cars = 16 → 
  selections_per_car = 3 → 
  cars_per_client = 2 → 
  (num_cars * selections_per_car) / cars_per_client = 24 := by
sorry

end used_car_clients_l703_70384


namespace complex_distance_range_l703_70383

theorem complex_distance_range (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min max : ℝ), min = 3 ∧ max = 5 ∧
  (∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 →
    min ≤ Complex.abs (w - 2 - 2*I) ∧ Complex.abs (w - 2 - 2*I) ≤ max) :=
by sorry

end complex_distance_range_l703_70383


namespace line_plane_perpendicularity_l703_70333

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (plane_perp : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (a : Line) (α β : Plane) 
  (h : subset a α) : 
  (∀ (b : Line), subset b α → (perp b β → plane_perp α β)) ∧ 
  (∃ (c : Line), subset c α ∧ plane_perp α β ∧ ¬perp c β) :=
sorry

end line_plane_perpendicularity_l703_70333


namespace book_sorting_terminates_and_sorts_width_l703_70318

/-- Represents a book with height and width -/
structure Book where
  height : ℕ
  width : ℕ

/-- The state of the bookshelf -/
structure BookshelfState where
  books : List Book
  n : ℕ

/-- Predicate to check if books are sorted by increasing width -/
def sortedByWidth (state : BookshelfState) : Prop :=
  ∀ i j, i < j → i < state.n → j < state.n →
    (state.books.get ⟨i, by sorry⟩).width < (state.books.get ⟨j, by sorry⟩).width

/-- Predicate to check if a swap is valid -/
def canSwap (state : BookshelfState) (i : ℕ) : Prop :=
  i + 1 < state.n ∧
  (state.books.get ⟨i, by sorry⟩).width > (state.books.get ⟨i + 1, by sorry⟩).width ∧
  (state.books.get ⟨i, by sorry⟩).height < (state.books.get ⟨i + 1, by sorry⟩).height

/-- The main theorem -/
theorem book_sorting_terminates_and_sorts_width
  (initial : BookshelfState)
  (h_n : initial.n ≥ 2)
  (h_unique : ∀ i j, i ≠ j → i < initial.n → j < initial.n →
    (initial.books.get ⟨i, by sorry⟩).height ≠ (initial.books.get ⟨j, by sorry⟩).height ∧
    (initial.books.get ⟨i, by sorry⟩).width ≠ (initial.books.get ⟨j, by sorry⟩).width)
  (h_initial_height : ∀ i j, i < j → i < initial.n → j < initial.n →
    (initial.books.get ⟨i, by sorry⟩).height < (initial.books.get ⟨j, by sorry⟩).height) :
  ∃ (final : BookshelfState),
    (∀ i, ¬canSwap final i) ∧
    sortedByWidth final :=
by sorry

end book_sorting_terminates_and_sorts_width_l703_70318


namespace expression_equality_l703_70365

theorem expression_equality : 
  |1 - Real.sqrt 3| + 3 * Real.tan (30 * π / 180) - (1/2)⁻¹ + (3 - π)^0 = 3.732 + Real.sqrt 3 := by
  sorry

end expression_equality_l703_70365


namespace expression_equals_one_l703_70307

theorem expression_equals_one (a b c : ℝ) (h : b^2 = a*c) :
  (a^2 * b^2 * c^2) / (a^3 + b^3 + c^3) * (1/a^3 + 1/b^3 + 1/c^3) = 1 := by
  sorry

end expression_equals_one_l703_70307


namespace quadratic_root_implies_a_value_l703_70342

theorem quadratic_root_implies_a_value (a : ℝ) : 
  (4^2 - 3*4 = a^2) → (a = 2 ∨ a = -2) := by
  sorry

end quadratic_root_implies_a_value_l703_70342


namespace even_increasing_function_inequality_l703_70356

-- Define an even function on ℝ
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define an increasing function on [0,+∞)
def increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

theorem even_increasing_function_inequality (f : ℝ → ℝ) (k : ℝ) 
  (h_even : even_function f) 
  (h_increasing : increasing_on_nonneg f) 
  (h_inequality : f k > f 2) : 
  k > 2 ∨ k < -2 :=
sorry

end even_increasing_function_inequality_l703_70356


namespace find_c_and_d_l703_70330

/-- Definition of the polynomial g(x) -/
def g (c d x : ℝ) : ℝ := c * x^3 - 8 * x^2 + d * x - 7

/-- Theorem stating the conditions and the result to be proved -/
theorem find_c_and_d :
  ∀ c d : ℝ,
  g c d 2 = -7 →
  g c d (-1) = -25 →
  c = 2 ∧ d = 8 := by
sorry

end find_c_and_d_l703_70330


namespace star_equation_solution_l703_70382

-- Define the * operation
def star (a b : ℝ) : ℝ := 4 * a + 2 * b

-- State the theorem
theorem star_equation_solution :
  ∃ y : ℝ, star 3 (star 4 y) = -2 ∧ y = -11.5 := by
  sorry

end star_equation_solution_l703_70382


namespace tangent_line_equation_l703_70349

/-- The equation of a cubic curve -/
def f (x : ℝ) : ℝ := x^3 - 2*x + 3

/-- The derivative of the cubic curve -/
def f' (x : ℝ) : ℝ := 3*x^2 - 2

/-- The point on the curve where we want to find the tangent line -/
def x₀ : ℝ := 1

/-- The y-coordinate of the point on the curve -/
def y₀ : ℝ := f x₀

/-- The slope of the tangent line at the point (x₀, y₀) -/
def m : ℝ := f' x₀

theorem tangent_line_equation :
  ∀ x y : ℝ, (x - x₀) = m * (y - y₀) ↔ x - y + 1 = 0 :=
by sorry

end tangent_line_equation_l703_70349


namespace joan_sold_26_books_l703_70370

/-- The number of books Joan sold in the yard sale -/
def books_sold (initial_books remaining_books : ℕ) : ℕ :=
  initial_books - remaining_books

/-- Theorem: Joan sold 26 books in the yard sale -/
theorem joan_sold_26_books :
  books_sold 33 7 = 26 := by
  sorry

end joan_sold_26_books_l703_70370


namespace ashley_champagne_bottles_l703_70334

/-- The number of bottles of champagne needed for a wedding toast --/
def bottles_needed (guests : ℕ) (glasses_per_guest : ℕ) (servings_per_bottle : ℕ) : ℕ :=
  (guests * glasses_per_guest) / servings_per_bottle

/-- Theorem: Ashley needs 40 bottles of champagne for her wedding toast --/
theorem ashley_champagne_bottles : 
  bottles_needed 120 2 6 = 40 := by
  sorry

end ashley_champagne_bottles_l703_70334


namespace smallest_integer_with_remainders_l703_70371

theorem smallest_integer_with_remainders : ∃ (n : ℕ), 
  (∀ (m : ℕ), m > 0 ∧ m % 6 = 2 ∧ m % 8 = 3 → n ≤ m) ∧
  n > 0 ∧ n % 6 = 2 ∧ n % 8 = 3 :=
by
  -- The proof would go here
  sorry

end smallest_integer_with_remainders_l703_70371


namespace fifteenth_digit_of_sum_one_seventh_one_eleventh_l703_70374

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ := sorry

/-- The sum of decimal representations of two rational numbers -/
def sum_decimal_representations (q₁ q₂ : ℚ) : ℕ → ℕ := sorry

/-- The nth digit after the decimal point in a decimal representation -/
def nth_digit_after_decimal (rep : ℕ → ℕ) (n : ℕ) : ℕ := sorry

theorem fifteenth_digit_of_sum_one_seventh_one_eleventh :
  nth_digit_after_decimal (sum_decimal_representations (1/7) (1/11)) 15 = 2 := by
  sorry

end fifteenth_digit_of_sum_one_seventh_one_eleventh_l703_70374


namespace f_is_odd_and_increasing_l703_70341

def f (x : ℝ) : ℝ := x * |x|

theorem f_is_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end f_is_odd_and_increasing_l703_70341


namespace part_one_part_two_l703_70337

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 1

-- Part (1)
theorem part_one (m n : ℝ) :
  (∀ x, f m x < 0 ↔ -2 < x ∧ x < n) →
  m = 3/2 ∧ n = 1/2 := by sorry

-- Part (2)
theorem part_two (m : ℝ) :
  (∀ x ∈ Set.Icc m (m+1), f m x < 0) →
  m > -Real.sqrt 2 / 2 ∧ m < 0 := by sorry

end part_one_part_two_l703_70337


namespace det_equation_solution_l703_70375

/-- Definition of 2nd order determinant -/
def det2 (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem: If |x+1 1-x; 1-x x+1| = 8, then x = 2 -/
theorem det_equation_solution (x : ℝ) : 
  det2 (x + 1) (1 - x) (1 - x) (x + 1) = 8 → x = 2 := by
  sorry

end det_equation_solution_l703_70375


namespace complex_modulus_problem_l703_70300

theorem complex_modulus_problem (z : ℂ) (x : ℝ) 
  (h1 : z * Complex.I = 2 * Complex.I + x)
  (h2 : z.im = 2) : 
  Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l703_70300


namespace simplify_and_evaluate_l703_70316

theorem simplify_and_evaluate (m : ℤ) (h : m = -1) :
  -(m^2 - 3*m) + 2*(m^2 - m - 1) = -2 := by
  sorry

end simplify_and_evaluate_l703_70316


namespace alpha_plus_beta_equals_81_l703_70399

theorem alpha_plus_beta_equals_81 
  (α β : ℝ) 
  (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 72*x + 945) / (x^2 + 45*x - 3240)) : 
  α + β = 81 := by
sorry

end alpha_plus_beta_equals_81_l703_70399


namespace three_digit_swap_solution_l703_70344

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def swap_digits (n : ℕ) : ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  100 * c + 10 * b + a

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem three_digit_swap_solution :
  ∀ A B : ℕ,
    is_three_digit A →
    is_three_digit B →
    B = swap_digits A →
    A / B = 3 →
    A % B = 7 * sum_of_digits A →
    ((A = 421 ∧ B = 124) ∨ (A = 842 ∧ B = 248)) :=
by sorry

end three_digit_swap_solution_l703_70344


namespace problem_polygon_area_l703_70385

/-- A point in a 2D grid --/
structure GridPoint where
  x : Int
  y : Int

/-- A polygon defined by a list of grid points --/
def Polygon := List GridPoint

/-- The polygon described in the problem --/
def problemPolygon : Polygon := [
  ⟨0, 0⟩, ⟨10, 0⟩, ⟨20, 0⟩, ⟨30, 10⟩, ⟨20, 30⟩,
  ⟨10, 30⟩, ⟨0, 30⟩, ⟨0, 20⟩, ⟨10, 20⟩, ⟨10, 10⟩
]

/-- Calculate the area of a polygon given its vertices --/
def calculatePolygonArea (p : Polygon) : Int :=
  sorry

theorem problem_polygon_area :
  calculatePolygonArea problemPolygon = 9 := by
  sorry

end problem_polygon_area_l703_70385


namespace probability_sum_14_correct_l703_70328

/-- Represents a standard deck of 52 cards -/
def StandardDeck : Nat := 52

/-- Represents the number of cards with values 2 through 10 in a standard deck -/
def NumberCards : Nat := 36

/-- Represents the number of pairs of number cards that sum to 14 -/
def PairsSummingTo14 : Nat := 76

/-- The probability of selecting two number cards that sum to 14 from a standard deck -/
def probability_sum_14 : ℚ := 19 / 663

theorem probability_sum_14_correct : 
  (PairsSummingTo14 : ℚ) / (StandardDeck * (StandardDeck - 1)) = probability_sum_14 := by
  sorry

end probability_sum_14_correct_l703_70328


namespace units_digit_of_special_three_digit_number_l703_70336

/-- The product of digits of a three-digit number -/
def P (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

/-- The sum of digits of a three-digit number -/
def S (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

/-- A three-digit number is between 100 and 999 -/
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem units_digit_of_special_three_digit_number (N : ℕ) 
  (h1 : is_three_digit N) 
  (h2 : N = P N + S N) : 
  N % 10 = 9 := by
sorry

end units_digit_of_special_three_digit_number_l703_70336


namespace prime_divides_29_power_plus_one_l703_70390

theorem prime_divides_29_power_plus_one (p : ℕ) : 
  Nat.Prime p ∧ p ∣ 29^p + 1 ↔ p = 2 ∨ p = 3 ∨ p = 5 := by
  sorry

end prime_divides_29_power_plus_one_l703_70390


namespace second_smallest_five_digit_pascal_correct_l703_70357

/-- Binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Predicate to check if a number is five digits -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

/-- Predicate to check if a number appears in Pascal's triangle -/
def in_pascal_triangle (n : ℕ) : Prop := ∃ (row col : ℕ), binomial row col = n

/-- The second smallest five-digit number in Pascal's triangle -/
def second_smallest_five_digit_pascal : ℕ := 31465

theorem second_smallest_five_digit_pascal_correct :
  is_five_digit second_smallest_five_digit_pascal ∧
  in_pascal_triangle second_smallest_five_digit_pascal ∧
  ∃ (m : ℕ), is_five_digit m ∧ 
             in_pascal_triangle m ∧ 
             m < second_smallest_five_digit_pascal ∧
             ∀ (k : ℕ), is_five_digit k ∧ 
                        in_pascal_triangle k ∧ 
                        k ≠ m → 
                        second_smallest_five_digit_pascal ≤ k :=
by sorry

end second_smallest_five_digit_pascal_correct_l703_70357


namespace final_rope_length_l703_70397

/-- Represents the weekly rope transactions in feet -/
def weekly_transactions : List ℝ :=
  [6, 18, 14, -9, 8, -1, 3, -10]

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

/-- Calculates the total rope length in inches after all transactions -/
def total_rope_length : ℝ :=
  (weekly_transactions.sum * feet_to_inches)

theorem final_rope_length :
  total_rope_length = 348 := by sorry

end final_rope_length_l703_70397


namespace opposite_of_negative_fraction_opposite_of_negative_one_over_2023_l703_70360

theorem opposite_of_negative_fraction (n : ℕ) (hn : n ≠ 0) :
  (-(1 : ℚ) / n) + (1 : ℚ) / n = 0 :=
by sorry

theorem opposite_of_negative_one_over_2023 :
  (-(1 : ℚ) / 2023) + (1 : ℚ) / 2023 = 0 :=
by sorry

end opposite_of_negative_fraction_opposite_of_negative_one_over_2023_l703_70360


namespace imaginary_unit_multiplication_l703_70387

-- Define the complex number i
def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_unit_multiplication :
  i * (1 + i) = -1 + i := by sorry

end imaginary_unit_multiplication_l703_70387


namespace simplify_trig_expression_l703_70347

theorem simplify_trig_expression :
  Real.sqrt (1 - 2 * Real.sin (Real.pi + 4) * Real.cos (Real.pi + 4)) = Real.cos 4 - Real.sin 4 := by
  sorry

end simplify_trig_expression_l703_70347


namespace triangle_theorem_l703_70314

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.cos t.C = 1/4)
  (h2 : t.a^2 = t.b^2 + (1/2) * t.c^2) :
  Real.sin (t.A - t.B) = Real.sqrt 15 / 8 ∧
  (t.c = Real.sqrt 10 → t.a = 3 ∧ t.b = 2) := by
  sorry

end triangle_theorem_l703_70314


namespace intersection_of_A_and_B_l703_70379

def A : Set ℤ := {1, 2}
def B : Set ℤ := {x | x^2 - 5*x + 4 < 0}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end intersection_of_A_and_B_l703_70379


namespace f_minimum_value_l703_70312

open Real

noncomputable def f (x : ℝ) : ℝ := (3 * sin x - 4 * cos x - 10) * (3 * sin x + 4 * cos x - 10)

theorem f_minimum_value :
  ∃ (min : ℝ), (∀ (x : ℝ), f x ≥ min) ∧ (min = 25 / 9 - 10 - 80 * Real.sqrt 2 / 3 - 116) := by
  sorry

end f_minimum_value_l703_70312


namespace car_speed_problem_l703_70326

/-- Proves that given a car traveling for two hours with an average speed of 40 km/h,
    and a speed of 60 km/h in the second hour, the speed in the first hour must be 20 km/h. -/
theorem car_speed_problem (speed_first_hour : ℝ) (speed_second_hour : ℝ) (average_speed : ℝ) :
  speed_second_hour = 60 →
  average_speed = 40 →
  (speed_first_hour + speed_second_hour) / 2 = average_speed →
  speed_first_hour = 20 :=
by sorry

end car_speed_problem_l703_70326


namespace arithmetic_sequence_theorem_l703_70362

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

-- Define the property of a_1 and a_2 being roots of the equation
def roots_property (a : ℕ → ℝ) : Prop :=
  (a 1)^2 - (a 3) * (a 1) + (a 4) = 0 ∧
  (a 2)^2 - (a 3) * (a 2) + (a 4) = 0

-- Theorem statement
theorem arithmetic_sequence_theorem (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d → roots_property a → ∀ n : ℕ, a n = 2 * n := by
  sorry

end arithmetic_sequence_theorem_l703_70362


namespace find_first_fraction_l703_70335

def compound_ratio : ℚ := 0.07142857142857142
def second_fraction : ℚ := 1/3
def third_fraction : ℚ := 3/8

theorem find_first_fraction :
  ∃ (first_fraction : ℚ), first_fraction * second_fraction * third_fraction = compound_ratio :=
sorry

end find_first_fraction_l703_70335


namespace sum_square_units_digits_2023_l703_70366

def first_odd_integers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def square_units_digit (n : ℕ) : ℕ :=
  (n ^ 2) % 10

def sum_square_units_digits (n : ℕ) : ℕ :=
  (first_odd_integers n).map square_units_digit |> List.sum

theorem sum_square_units_digits_2023 :
  sum_square_units_digits 2023 % 10 = 5 := by
  sorry

end sum_square_units_digits_2023_l703_70366


namespace square_area_is_56_l703_70317

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 = -2 * y^2 + 24 * x + 8 * y + 36

-- Define the property that the circle is inscribed in a square with sides parallel to axes
def inscribed_in_square (center_x center_y radius : ℝ) : Prop :=
  ∃ (side_length : ℝ), side_length = 2 * radius

-- Theorem statement
theorem square_area_is_56 :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    inscribed_in_square center_x center_y radius ∧
    4 * radius^2 = 56 := by
  sorry

end square_area_is_56_l703_70317


namespace garden_area_increase_l703_70310

/-- Proves that adding 60 feet of fence to a rectangular garden of 80x20 feet
    to make it square increases the area by 2625 square feet. -/
theorem garden_area_increase : 
  ∀ (original_length original_width added_fence : ℕ),
    original_length = 80 →
    original_width = 20 →
    added_fence = 60 →
    let original_perimeter := 2 * (original_length + original_width)
    let new_perimeter := original_perimeter + added_fence
    let new_side := new_perimeter / 4
    let original_area := original_length * original_width
    let new_area := new_side * new_side
    new_area - original_area = 2625 := by
  sorry

end garden_area_increase_l703_70310


namespace only_group_d_forms_triangle_l703_70346

/-- A group of three sticks --/
structure StickGroup where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a group of sticks can form a triangle --/
def canFormTriangle (g : StickGroup) : Prop :=
  g.a + g.b > g.c ∧ g.b + g.c > g.a ∧ g.c + g.a > g.b

/-- The given groups of sticks --/
def groupA : StickGroup := ⟨1, 2, 6⟩
def groupB : StickGroup := ⟨2, 2, 4⟩
def groupC : StickGroup := ⟨1, 2, 3⟩
def groupD : StickGroup := ⟨2, 3, 4⟩

/-- Theorem: Only group D can form a triangle --/
theorem only_group_d_forms_triangle :
  ¬(canFormTriangle groupA) ∧
  ¬(canFormTriangle groupB) ∧
  ¬(canFormTriangle groupC) ∧
  canFormTriangle groupD :=
sorry

end only_group_d_forms_triangle_l703_70346


namespace event_probability_l703_70352

noncomputable def probability_event (a b : Real) : Real :=
  (min b (3/2) - max a 0) / (b - a)

theorem event_probability : probability_event 0 2 = 3/4 := by
  sorry

end event_probability_l703_70352


namespace last_three_average_l703_70338

theorem last_three_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 65 →
  (list.take 4).sum / 4 = 60 →
  (list.drop 4).sum / 3 = 215 / 3 := by
sorry

end last_three_average_l703_70338


namespace circle_equation_from_center_and_chord_l703_70391

/-- The equation of a circle given its center and a chord on a line. -/
theorem circle_equation_from_center_and_chord (x y : ℝ) :
  let center : ℝ × ℝ := (4, 7)
  let chord_length : ℝ := 8
  let line_eq : ℝ → ℝ → ℝ := fun x y => 3 * x - 4 * y + 1
  (∃ (a b : ℝ), (a - 4)^2 + (b - 7)^2 = 25 ∧ 
                line_eq a b = 0 ∧ 
                (a - center.1)^2 + (b - center.2)^2 = (chord_length / 2)^2) →
  (x - 4)^2 + (y - 7)^2 = 25 := by
sorry

end circle_equation_from_center_and_chord_l703_70391


namespace polynomial_roots_interlace_l703_70361

theorem polynomial_roots_interlace (p₁ p₂ q₁ q₂ : ℝ) 
  (h : (q₁ - q₂)^2 + (p₁ - p₂)*(p₁*q₂ - p₂*q₁) < 0) :
  let f := fun x : ℝ => x^2 + p₁*x + q₁
  let g := fun x : ℝ => x^2 + p₂*x + q₂
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ < y₂ ∧ g y₁ = 0 ∧ g y₂ = 0) ∧
  (∃ x y : ℝ, (f x = 0 ∧ y₁ < x ∧ x < y₂) ∧ (g y = 0 ∧ x₁ < y ∧ y < x₂)) :=
by sorry

end polynomial_roots_interlace_l703_70361


namespace convex_polygon_as_intersection_of_halfplanes_l703_70301

-- Define a convex polygon
def ConvexPolygon (P : Set (ℝ × ℝ)) : Prop := sorry

-- Define a half-plane
def HalfPlane (H : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem convex_polygon_as_intersection_of_halfplanes 
  (P : Set (ℝ × ℝ)) (h : ConvexPolygon P) :
  ∃ (n : ℕ) (H : Fin n → Set (ℝ × ℝ)), 
    (∀ i, HalfPlane (H i)) ∧ 
    P = ⋂ i, H i :=
sorry

end convex_polygon_as_intersection_of_halfplanes_l703_70301


namespace beehives_for_candles_l703_70398

/-- Given that 3 beehives make enough wax for 12 candles, 
    prove that 24 hives are needed to make 96 candles. -/
theorem beehives_for_candles : 
  (3 : ℚ) * 96 / 12 = 24 := by sorry

end beehives_for_candles_l703_70398
