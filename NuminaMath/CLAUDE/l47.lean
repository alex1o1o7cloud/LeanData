import Mathlib

namespace NUMINAMATH_CALUDE_damien_jogging_days_l47_4702

/-- Represents the number of miles Damien jogs per day -/
def miles_per_day : ℕ := 5

/-- Represents the total number of miles Damien jogs over three weeks -/
def total_miles : ℕ := 75

/-- Calculates the number of days Damien jogs over three weeks -/
def days_jogged : ℕ := total_miles / miles_per_day

theorem damien_jogging_days :
  days_jogged = 15 := by sorry

end NUMINAMATH_CALUDE_damien_jogging_days_l47_4702


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l47_4789

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ) :
  (∀ x : ℝ, (x - 1)^4 * (x + 2)^8 = a*x^12 + a₁*x^11 + a₂*x^10 + a₃*x^9 + a₄*x^8 + 
    a₅*x^7 + a₆*x^6 + a₇*x^5 + a₈*x^4 + a₉*x^3 + a₁₀*x^2 + a₁₁*x + a₁₂) →
  a + a₂ + a₄ + a₆ + a₈ + a₁₀ + a₁₂ = 8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l47_4789


namespace NUMINAMATH_CALUDE_tan_function_product_l47_4745

theorem tan_function_product (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + π / 2))) → 
  a * Real.tan (b * π / 8) = 1 → 
  a * b = 2 := by sorry

end NUMINAMATH_CALUDE_tan_function_product_l47_4745


namespace NUMINAMATH_CALUDE_min_distance_to_equidistant_point_l47_4724

/-- Circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 9 = 0

/-- Circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

/-- Point P is equidistant from C₁ and C₂ -/
def equidistant (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y + 9 = x^2 + y^2 + 2*x + 2*y + 1

/-- The minimum distance from the origin to any point equidistant from C₁ and C₂ is 4/5 -/
theorem min_distance_to_equidistant_point :
  ∃ (x₀ y₀ : ℝ), equidistant x₀ y₀ ∧
    ∀ (x y : ℝ), equidistant x y → x₀^2 + y₀^2 ≤ x^2 + y^2 ∧
    x₀^2 + y₀^2 = (4/5)^2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_equidistant_point_l47_4724


namespace NUMINAMATH_CALUDE_range_of_a_l47_4796

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := (x - t) * abs x

-- State the theorem
theorem range_of_a (t : ℝ) (h_t : t ∈ Set.Ioo 0 2) :
  (∀ x ∈ Set.Icc (-1) 2, ∃ a : ℝ, f t x > x + a) →
  (∃ a : ℝ, ∀ x ∈ Set.Icc (-1) 2, f t x > x + a ∧ a ≤ -1/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l47_4796


namespace NUMINAMATH_CALUDE_parabola_symmetric_point_l47_4721

/-- Given a parabola y = x^2 + 4x - m where (1, 2) is a point on the parabola,
    and a point B symmetric to (1, 2) with respect to the axis of symmetry,
    prove that the coordinates of B are (-5, 2). -/
theorem parabola_symmetric_point (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + 4*x - m
  let A : ℝ × ℝ := (1, 2)
  let axis_of_symmetry : ℝ := -2
  let B : ℝ × ℝ := (-5, 2)
  (f A.1 = A.2) →  -- A is on the parabola
  (A.1 - axis_of_symmetry = axis_of_symmetry - B.1) →  -- A and B are symmetric
  (A.2 = B.2) →  -- y-coordinates of A and B are equal
  B = (-5, 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetric_point_l47_4721


namespace NUMINAMATH_CALUDE_exists_prime_divisor_l47_4756

-- Define the sequence a_n
def a (c : ℕ+) : ℕ → ℕ
  | 0 => c
  | n + 1 => (a c n)^3 - 4 * c * (a c n)^2 + 5 * c^2 * (a c n) + c

-- State the theorem
theorem exists_prime_divisor (c : ℕ+) (n : ℕ) (hn : n ≥ 2) :
  ∃ p : ℕ, Prime p ∧ p ∣ a c (n - 1) ∧ ∀ k : ℕ, k < n - 1 → ¬(p ∣ a c k) := by
  sorry

end NUMINAMATH_CALUDE_exists_prime_divisor_l47_4756


namespace NUMINAMATH_CALUDE_M_intersect_N_empty_l47_4727

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 - x > 0}

-- Define set N
def N : Set ℝ := {x : ℝ | (x - 1) / x < 0}

-- Theorem statement
theorem M_intersect_N_empty : M ∩ N = ∅ := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_empty_l47_4727


namespace NUMINAMATH_CALUDE_fraction_equality_l47_4739

theorem fraction_equality (w z : ℝ) (h : (1/w + 1/z)/(1/w - 1/z) = 2023) : 
  (w + z)/(w - z) = -2023 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l47_4739


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l47_4788

theorem cubic_sum_theorem (a b : ℝ) (h : a^3 + b^3 + 3*a*b = 1) : 
  a + b = 1 ∨ a + b = -2 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l47_4788


namespace NUMINAMATH_CALUDE_log_square_plus_one_neither_sufficient_nor_necessary_l47_4749

theorem log_square_plus_one_neither_sufficient_nor_necessary (a b : ℝ) :
  ¬(∀ a b : ℝ, (Real.log (a^2 + 1) < Real.log (b^2 + 1)) → (a < b)) ∧
  ¬(∀ a b : ℝ, (a < b) → (Real.log (a^2 + 1) < Real.log (b^2 + 1))) :=
sorry

end NUMINAMATH_CALUDE_log_square_plus_one_neither_sufficient_nor_necessary_l47_4749


namespace NUMINAMATH_CALUDE_profit_percent_for_cost_selling_ratio_l47_4758

theorem profit_percent_for_cost_selling_ratio (cost_price selling_price : ℝ) :
  cost_price > 0 →
  selling_price > cost_price →
  cost_price / selling_price = 2 / 3 →
  (selling_price - cost_price) / cost_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_for_cost_selling_ratio_l47_4758


namespace NUMINAMATH_CALUDE_mobile_purchase_price_l47_4726

/-- Represents the purchase and sale of items with profit or loss -/
def ItemTransaction (purchase_price : ℚ) (profit_percent : ℚ) : ℚ :=
  purchase_price * (1 + profit_percent / 100)

theorem mobile_purchase_price :
  let grinder_price : ℚ := 15000
  let grinder_loss_percent : ℚ := 5
  let mobile_profit_percent : ℚ := 10
  let total_profit : ℚ := 50

  ∃ mobile_price : ℚ,
    (ItemTransaction grinder_price (-grinder_loss_percent) +
     ItemTransaction mobile_price mobile_profit_percent) -
    (grinder_price + mobile_price) = total_profit ∧
    mobile_price = 8000 :=
by sorry

end NUMINAMATH_CALUDE_mobile_purchase_price_l47_4726


namespace NUMINAMATH_CALUDE_extraneous_root_implies_m_value_l47_4710

theorem extraneous_root_implies_m_value (m : ℝ) : 
  (∃ x : ℝ, (x - 1) / (x + 4) = m / (x + 4) ∧ x + 4 = 0) → m = -5 :=
by sorry

end NUMINAMATH_CALUDE_extraneous_root_implies_m_value_l47_4710


namespace NUMINAMATH_CALUDE_cylinder_no_triangular_front_view_l47_4722

-- Define the set of solid geometries
inductive SolidGeometry
| Cylinder
| Cone
| Tetrahedron
| TriangularPrism

-- Define a function that determines if a solid geometry can have a triangular front view
def canHaveTriangularFrontView (s : SolidGeometry) : Prop :=
  match s with
  | SolidGeometry.Cylinder => False
  | _ => True

-- Theorem statement
theorem cylinder_no_triangular_front_view :
  ∀ s : SolidGeometry, ¬(canHaveTriangularFrontView s) ↔ s = SolidGeometry.Cylinder :=
by sorry

end NUMINAMATH_CALUDE_cylinder_no_triangular_front_view_l47_4722


namespace NUMINAMATH_CALUDE_five_number_sum_problem_l47_4709

theorem five_number_sum_problem :
  ∃! (a b c d e : ℕ),
    (∀ (s : Finset ℕ), s.card = 4 → s ⊆ {a, b, c, d, e} →
      (s.sum id = 44 ∨ s.sum id = 45 ∨ s.sum id = 46 ∨ s.sum id = 47)) ∧
    ({a, b, c, d, e} : Finset ℕ).card = 5 ∧
    a = 13 ∧ b = 12 ∧ c = 11 ∧ d = 11 ∧ e = 10 :=
by sorry

end NUMINAMATH_CALUDE_five_number_sum_problem_l47_4709


namespace NUMINAMATH_CALUDE_price_after_nine_years_l47_4761

def initial_price : ℝ := 640
def decrease_rate : ℝ := 0.25
def years : ℕ := 9
def price_after_n_years (n : ℕ) : ℝ := initial_price * (1 - decrease_rate) ^ (n / 3)

theorem price_after_nine_years :
  price_after_n_years years = 270 := by
  sorry

end NUMINAMATH_CALUDE_price_after_nine_years_l47_4761


namespace NUMINAMATH_CALUDE_inequality_equivalence_l47_4753

theorem inequality_equivalence (x : ℝ) : 
  (x / (x + 1) + 2 * x / ((x + 1) * (2 * x + 1)) + 3 * x / ((x + 1) * (2 * x + 1) * (3 * x + 1)) > 1) ↔ 
  (x < -1 ∨ (-1/2 < x ∧ x < -1/3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l47_4753


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l47_4760

theorem polar_to_rectangular_conversion (r θ : ℝ) (h1 : r = 5) (h2 : θ = 5 * π / 4) :
  (r * Real.cos θ, r * Real.sin θ) = (-5 * Real.sqrt 2 / 2, -5 * Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l47_4760


namespace NUMINAMATH_CALUDE_rahul_savings_l47_4742

/-- Rahul's savings problem -/
theorem rahul_savings (nsc ppf : ℕ) : 
  (1/3 : ℚ) * nsc = (1/2 : ℚ) * ppf →
  nsc + ppf = 180000 →
  ppf = 72000 := by
sorry

end NUMINAMATH_CALUDE_rahul_savings_l47_4742


namespace NUMINAMATH_CALUDE_sum_equals_17b_l47_4769

theorem sum_equals_17b (b : ℝ) (a c d : ℝ) 
  (ha : a = 3 * b) 
  (hc : c = 2 * a) 
  (hd : d = c + b) : 
  a + b + c + d = 17 * b := by
sorry

end NUMINAMATH_CALUDE_sum_equals_17b_l47_4769


namespace NUMINAMATH_CALUDE_max_product_of_areas_l47_4759

theorem max_product_of_areas (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  a + b + c + d = 1 →
  a * b * c * d ≤ 1 / 256 :=
sorry

end NUMINAMATH_CALUDE_max_product_of_areas_l47_4759


namespace NUMINAMATH_CALUDE_p_satisfies_equation_l47_4781

/-- The polynomial p(x) that satisfies the given equation -/
def p (x : ℝ) : ℝ := -2*x^4 - 2*x^3 + 5*x^2 - 2*x + 2

/-- The theorem stating that p(x) satisfies the given equation -/
theorem p_satisfies_equation (x : ℝ) :
  4*x^4 + 2*x^3 - 6*x + 4 + p x = 2*x^4 + 5*x^2 - 8*x + 6 := by
  sorry

end NUMINAMATH_CALUDE_p_satisfies_equation_l47_4781


namespace NUMINAMATH_CALUDE_blueberry_tart_fraction_l47_4793

theorem blueberry_tart_fraction (total : Real) (cherry : Real) (peach : Real)
  (h1 : total = 0.91)
  (h2 : cherry = 0.08)
  (h3 : peach = 0.08) :
  total - (cherry + peach) = 0.75 := by
sorry

end NUMINAMATH_CALUDE_blueberry_tart_fraction_l47_4793


namespace NUMINAMATH_CALUDE_total_undeveloped_area_l47_4768

def undeveloped_sections : ℕ := 3
def section_area : ℕ := 2435

theorem total_undeveloped_area : undeveloped_sections * section_area = 7305 := by
  sorry

end NUMINAMATH_CALUDE_total_undeveloped_area_l47_4768


namespace NUMINAMATH_CALUDE_boxes_of_apples_l47_4762

/-- The number of boxes of apples after processing a delivery -/
def number_of_boxes (apples_per_crate : ℕ) (crates_delivered : ℕ) (rotten_apples : ℕ) (apples_per_box : ℕ) : ℕ :=
  ((apples_per_crate * crates_delivered - rotten_apples) / apples_per_box)

/-- Theorem stating that the number of boxes is 50 given the specific conditions -/
theorem boxes_of_apples :
  number_of_boxes 42 12 4 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_boxes_of_apples_l47_4762


namespace NUMINAMATH_CALUDE_power_multiplication_l47_4786

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l47_4786


namespace NUMINAMATH_CALUDE_non_multiples_count_is_412_l47_4755

/-- The count of three-digit numbers that are not multiples of 3, 5, or 7 -/
def non_multiples_count : ℕ :=
  let total_three_digit := 999 - 100 + 1
  let multiples_3 := (999 - 100) / 3 + 1
  let multiples_5 := (995 - 100) / 5 + 1
  let multiples_7 := (994 - 105) / 7 + 1
  let multiples_15 := (990 - 105) / 15 + 1
  let multiples_21 := (987 - 105) / 21 + 1
  let multiples_35 := (980 - 105) / 35 + 1
  let multiples_105 := (945 - 105) / 105 + 1
  let total_multiples := multiples_3 + multiples_5 + multiples_7 - multiples_15 - multiples_21 - multiples_35 + multiples_105
  total_three_digit - total_multiples

theorem non_multiples_count_is_412 : non_multiples_count = 412 := by
  sorry

end NUMINAMATH_CALUDE_non_multiples_count_is_412_l47_4755


namespace NUMINAMATH_CALUDE_equation_solution_l47_4785

theorem equation_solution (x : ℝ) (h : x ≠ -2) :
  (4 * x^2 - 3 * x + 2) / (x + 2) = 4 * x - 5 ↔ x = 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l47_4785


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l47_4792

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = 2/5 + (1/3 : ℝ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = 2/5 - (1/3 : ℝ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l47_4792


namespace NUMINAMATH_CALUDE_pizza_problem_l47_4701

/-- The number of triple cheese pizzas purchased -/
def T : ℕ := 10

/-- The number of meat lovers pizzas purchased -/
def M : ℕ := 9

/-- The standard price of a pizza in dollars -/
def standard_price : ℕ := 5

/-- The total cost in dollars -/
def total_cost : ℕ := 55

/-- The cost of triple cheese pizzas under the special pricing -/
def triple_cheese_cost (n : ℕ) : ℕ := (n / 2) * standard_price

/-- The cost of meat lovers pizzas under the special pricing -/
def meat_lovers_cost (n : ℕ) : ℕ := ((n / 3) * 2) * standard_price

theorem pizza_problem : 
  triple_cheese_cost T + meat_lovers_cost M = total_cost :=
sorry

end NUMINAMATH_CALUDE_pizza_problem_l47_4701


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l47_4719

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a 1 * q^n

theorem geometric_sequence_solution (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 * a 2 * a 3 = 27 →
  a 2 + a 4 = 30 →
  ((a 1 = 1 ∧ ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a 1 * q^n) ∧ q = 3) ∨
   (a 1 = -1 ∧ ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a 1 * q^n) ∧ q = -3)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l47_4719


namespace NUMINAMATH_CALUDE_negation_of_P_is_true_l47_4725

theorem negation_of_P_is_true : ∀ (x : ℝ), (x - 1)^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_P_is_true_l47_4725


namespace NUMINAMATH_CALUDE_angle_sum_in_circle_l47_4729

/-- Given a circle with four angles around its center measured as 7x°, 3x°, 4x°, and x°,
    prove that x = 24°. -/
theorem angle_sum_in_circle (x : ℝ) : 
  (7 * x + 3 * x + 4 * x + x : ℝ) = 360 → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_circle_l47_4729


namespace NUMINAMATH_CALUDE_cake_piece_volume_l47_4770

/-- The volume of a piece of cake -/
theorem cake_piece_volume (diameter : ℝ) (thickness : ℝ) (num_pieces : ℕ) 
  (h1 : diameter = 16)
  (h2 : thickness = 1/2)
  (h3 : num_pieces = 8) :
  (π * (diameter/2)^2 * thickness) / num_pieces = 4 * π := by
  sorry

end NUMINAMATH_CALUDE_cake_piece_volume_l47_4770


namespace NUMINAMATH_CALUDE_ordering_proof_l47_4737

theorem ordering_proof (x a y : ℝ) (h1 : x < a) (h2 : a < y) (h3 : y < 0) :
  x^3 < y*a^2 ∧ y*a^2 < a*y ∧ a*y < x^2 := by
  sorry

end NUMINAMATH_CALUDE_ordering_proof_l47_4737


namespace NUMINAMATH_CALUDE_draw_ball_one_probability_l47_4703

/-- The number of balls in the box -/
def total_balls : ℕ := 5

/-- The number of balls drawn -/
def drawn_balls : ℕ := 2

/-- The number of ways to draw the specific ball (number 1) -/
def favorable_outcomes : ℕ := 4

/-- The total number of ways to draw 2 balls out of 5 -/
def total_outcomes : ℕ := 10

/-- The probability of drawing ball number 1 -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem draw_ball_one_probability :
  probability = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_draw_ball_one_probability_l47_4703


namespace NUMINAMATH_CALUDE_A_subset_B_l47_4787

-- Define set A
def A : Set ℤ := {x | ∃ k : ℕ, x = 7 * k + 3}

-- Define set B
def B : Set ℤ := {x | ∃ k : ℤ, x = 7 * k - 4}

-- Theorem stating A is a subset of B
theorem A_subset_B : A ⊆ B := by
  sorry

end NUMINAMATH_CALUDE_A_subset_B_l47_4787


namespace NUMINAMATH_CALUDE_square_binomial_minus_square_l47_4790

theorem square_binomial_minus_square : 15^2 + 2*(15*5) + 5^2 - 3^2 = 391 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_minus_square_l47_4790


namespace NUMINAMATH_CALUDE_acid_solution_concentration_l47_4778

theorem acid_solution_concentration 
  (P : ℝ) -- Original acid concentration percentage
  (h1 : 0 ≤ P ∧ P ≤ 100) -- Ensure P is a valid percentage
  (h2 : 0.5 * P + 0.5 * 20 = 35) -- Equation representing the mixing process
  : P = 50 := by
  sorry

end NUMINAMATH_CALUDE_acid_solution_concentration_l47_4778


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l47_4743

/-- The coefficient of x^3 in the expansion of (1+2x)^6 is 160 -/
theorem coefficient_x_cubed_in_binomial_expansion : 
  (Finset.range 7).sum (fun k => (Nat.choose 6 k) * 2^k * if k = 3 then 1 else 0) = 160 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l47_4743


namespace NUMINAMATH_CALUDE_linear_regression_approximation_l47_4731

/-- Linear regression problem -/
theorem linear_regression_approximation 
  (b : ℝ) -- Slope of the regression line
  (x_mean y_mean : ℝ) -- Mean values of x and y
  (h1 : y_mean = b * x_mean + 0.2) -- Regression line passes through (x_mean, y_mean)
  (h2 : x_mean = 4) -- Given mean of x
  (h3 : y_mean = 5) -- Given mean of y
  : b * 2 + 0.2 = 2.6 := by
  sorry

end NUMINAMATH_CALUDE_linear_regression_approximation_l47_4731


namespace NUMINAMATH_CALUDE_set_equality_implies_power_l47_4794

theorem set_equality_implies_power (m n : ℝ) : 
  let P : Set ℝ := {1, m}
  let Q : Set ℝ := {2, -n}
  P = Q → m^n = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_power_l47_4794


namespace NUMINAMATH_CALUDE_common_divisors_count_l47_4757

def a : Nat := 12600
def b : Nat := 14400

theorem common_divisors_count : (Nat.divisors (Nat.gcd a b)).card = 45 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_count_l47_4757


namespace NUMINAMATH_CALUDE_prom_tip_percentage_l47_4765

theorem prom_tip_percentage : 
  let ticket_cost : ℕ := 100
  let dinner_cost : ℕ := 120
  let limo_hourly_rate : ℕ := 80
  let limo_hours : ℕ := 6
  let total_cost : ℕ := 836
  let tip_percentage : ℚ := (total_cost - (2 * ticket_cost + dinner_cost + limo_hourly_rate * limo_hours)) / dinner_cost * 100
  tip_percentage = 30 := by sorry

end NUMINAMATH_CALUDE_prom_tip_percentage_l47_4765


namespace NUMINAMATH_CALUDE_chopped_cube_height_l47_4712

/-- Represents a 3D cube with a chopped corner --/
structure ChoppedCube where
  side_length : ℝ
  cut_ratio : ℝ

/-- The height of the remaining solid when the chopped face is placed on a table --/
def remaining_height (c : ChoppedCube) : ℝ :=
  c.side_length - c.cut_ratio * c.side_length

/-- Theorem stating that for a 2x2x2 cube with a corner chopped at midpoints, 
    the remaining height is 1 unit --/
theorem chopped_cube_height :
  let c : ChoppedCube := { side_length := 2, cut_ratio := 1/2 }
  remaining_height c = 1 := by
  sorry

end NUMINAMATH_CALUDE_chopped_cube_height_l47_4712


namespace NUMINAMATH_CALUDE_watch_cost_price_l47_4705

theorem watch_cost_price (loss_rate : ℝ) (gain_rate : ℝ) (price_difference : ℝ) :
  loss_rate = 0.1 →
  gain_rate = 0.04 →
  price_difference = 190 →
  ∃ (cost_price : ℝ),
    cost_price * (1 - loss_rate) = cost_price * 0.9 ∧
    cost_price * (1 + gain_rate) = cost_price * 1.04 ∧
    cost_price * (1 + gain_rate) - cost_price * (1 - loss_rate) = price_difference ∧
    cost_price = 1357.14 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l47_4705


namespace NUMINAMATH_CALUDE_student_pet_difference_l47_4707

/-- Represents a fourth-grade classroom at Pine Hill Elementary -/
structure Classroom where
  students : ℕ
  rabbits : ℕ
  hamsters : ℕ

/-- Creates a standard fourth-grade classroom -/
def standard_classroom : Classroom :=
  { students := 20, rabbits := 2, hamsters := 1 }

/-- Calculates the total number of pets in a classroom -/
def pets_in_classroom (c : Classroom) : ℕ :=
  c.rabbits + c.hamsters

/-- Calculates the total number of students in n classrooms -/
def total_students (n : ℕ) : ℕ :=
  n * standard_classroom.students

/-- Calculates the total number of pets in n classrooms -/
def total_pets (n : ℕ) : ℕ :=
  n * pets_in_classroom standard_classroom

/-- The main theorem: difference between students and pets in 5 classrooms is 85 -/
theorem student_pet_difference : total_students 5 - total_pets 5 = 85 := by
  sorry

end NUMINAMATH_CALUDE_student_pet_difference_l47_4707


namespace NUMINAMATH_CALUDE_final_chicken_count_l47_4711

def chicken_farm (initial_chickens : ℕ) 
                 (disease_A_infection_rate : ℚ)
                 (disease_A_death_rate : ℚ)
                 (disease_B_infection_rate : ℚ)
                 (disease_B_death_rate : ℚ)
                 (purchase_multiplier : ℚ) : ℕ :=
  sorry

theorem final_chicken_count : 
  chicken_farm 800 (15/100) (45/100) (20/100) (30/100) (25/2) = 1939 :=
sorry

end NUMINAMATH_CALUDE_final_chicken_count_l47_4711


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_4410_l47_4791

theorem largest_perfect_square_factor_of_4410 : 
  ∃ (n : ℕ), n * n = 441 ∧ n * n ∣ 4410 ∧ ∀ (m : ℕ), m * m ∣ 4410 → m * m ≤ n * n := by
  sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_4410_l47_4791


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_inequality_l47_4740

theorem smallest_n_for_sqrt_inequality : 
  ∃ (n : ℕ), n > 0 ∧ Real.sqrt n - Real.sqrt (n - 1) < 0.02 ∧ 
  ∀ (m : ℕ), m > 0 ∧ m < n → Real.sqrt m - Real.sqrt (m - 1) ≥ 0.02 :=
by
  use 626
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_inequality_l47_4740


namespace NUMINAMATH_CALUDE_bridge_length_l47_4751

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time_s : Real) :
  train_length = 120 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time_s) - train_length = 255 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l47_4751


namespace NUMINAMATH_CALUDE_finite_state_machine_cannot_generate_sqrt_two_l47_4732

/-- Represents a finite state machine -/
structure FiniteStateMachine where
  states : Finset ℕ
  initialState : ℕ
  transition : ℕ → ℕ
  output : ℕ → ℕ

/-- Represents an infinite sequence of natural numbers -/
def InfiniteSequence := ℕ → ℕ

/-- The decimal representation of √2 -/
noncomputable def sqrtTwoDecimal : InfiniteSequence :=
  sorry

/-- A sequence is eventually periodic if there exist n and p such that
    for all k ≥ n, f(k) = f(k+p) -/
def EventuallyPeriodic (f : InfiniteSequence) : Prop :=
  ∃ n p : ℕ, p > 0 ∧ ∀ k ≥ n, f k = f (k + p)

/-- The main theorem: No finite state machine can generate the decimal representation of √2 -/
theorem finite_state_machine_cannot_generate_sqrt_two :
  ∀ (fsm : FiniteStateMachine),
  ¬∃ (f : InfiniteSequence),
    (∀ n, f n = fsm.output (fsm.transition^[n] fsm.initialState)) ∧
    (f = sqrtTwoDecimal) :=
  sorry

end NUMINAMATH_CALUDE_finite_state_machine_cannot_generate_sqrt_two_l47_4732


namespace NUMINAMATH_CALUDE_simplify_expression_l47_4741

theorem simplify_expression : 
  (5^5 + 5^3 + 5) / (5^4 - 2*5^2 + 5) = 651 / 116 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l47_4741


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l47_4747

theorem constant_term_binomial_expansion (n : ℕ+) :
  (∃ k : ℕ, k ≤ n ∧ 3*n = 4*k) → n ≠ 6 := by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l47_4747


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l47_4750

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∀ x, (a - 1) * x < a - 1 ↔ x > 1) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l47_4750


namespace NUMINAMATH_CALUDE_division_remainder_proof_l47_4717

theorem division_remainder_proof (divisor quotient dividend remainder : ℕ) : 
  divisor = 21 →
  quotient = 14 →
  dividend = 301 →
  dividend = divisor * quotient + remainder →
  remainder = 7 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l47_4717


namespace NUMINAMATH_CALUDE_quadratic_inequality_rational_inequality_l47_4716

-- Problem 1
theorem quadratic_inequality (x : ℝ) :
  2 * x^2 - 3 * x + 1 < 0 ↔ 1/2 < x ∧ x < 1 :=
by sorry

-- Problem 2
theorem rational_inequality (x : ℝ) :
  2 * x / (x + 1) ≥ 1 ↔ x ≥ 1 ∨ x < -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_rational_inequality_l47_4716


namespace NUMINAMATH_CALUDE_half_vector_MN_l47_4746

/-- Given two vectors OM and ON in ℝ², prove that half of vector MN equals (-4, 1/2) -/
theorem half_vector_MN (OM ON : ℝ × ℝ) (h1 : OM = (3, -2)) (h2 : ON = (-5, -1)) :
  (1 / 2 : ℝ) • (ON - OM) = (-4, 1/2) := by
  sorry

end NUMINAMATH_CALUDE_half_vector_MN_l47_4746


namespace NUMINAMATH_CALUDE_grape_bowl_comparison_l47_4780

theorem grape_bowl_comparison (rob_grapes allie_grapes allyn_grapes : ℕ) : 
  rob_grapes = 25 →
  allie_grapes = rob_grapes + 2 →
  rob_grapes + allie_grapes + allyn_grapes = 83 →
  allyn_grapes - allie_grapes = 4 :=
by sorry

end NUMINAMATH_CALUDE_grape_bowl_comparison_l47_4780


namespace NUMINAMATH_CALUDE_roots_sum_condition_l47_4734

theorem roots_sum_condition (a b : ℤ) (α : ℝ) :
  (0 ≤ α ∧ α < 2 * Real.pi) →
  (∀ x : ℝ, x^2 + a * x + 2 * b^2 = 0 ↔ x = Real.sin α ∨ x = Real.cos α) →
  a + b = 1 ∨ a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_condition_l47_4734


namespace NUMINAMATH_CALUDE_total_ninja_stars_l47_4797

/-- The number of ninja throwing stars each person has -/
structure NinjaStars where
  eric : ℕ
  chad : ℕ
  mark : ℕ
  jennifer : ℕ

/-- The initial distribution of ninja throwing stars -/
def initial_distribution : NinjaStars where
  eric := 4
  chad := 4 * 2
  mark := 0
  jennifer := 0

/-- The final distribution of ninja throwing stars after all transactions -/
def final_distribution : NinjaStars :=
  let chad_after_selling := initial_distribution.chad - 2
  let chad_final := chad_after_selling + 3
  let mark_jennifer_share := 10 / 2
  { eric := initial_distribution.eric,
    chad := chad_final,
    mark := mark_jennifer_share,
    jennifer := mark_jennifer_share }

/-- The theorem stating the total number of ninja throwing stars -/
theorem total_ninja_stars :
  final_distribution.eric +
  final_distribution.chad +
  final_distribution.mark +
  final_distribution.jennifer = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_total_ninja_stars_l47_4797


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l47_4779

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {1, 3}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l47_4779


namespace NUMINAMATH_CALUDE_revenue_change_l47_4744

theorem revenue_change
  (T : ℝ) -- Original tax rate (as a percentage)
  (C : ℝ) -- Original consumption
  (h1 : T > 0)
  (h2 : C > 0) :
  let new_tax_rate := T * (1 - 0.19)
  let new_consumption := C * (1 + 0.15)
  let original_revenue := T / 100 * C
  let new_revenue := new_tax_rate / 100 * new_consumption
  (new_revenue - original_revenue) / original_revenue = -0.0685 :=
sorry

end NUMINAMATH_CALUDE_revenue_change_l47_4744


namespace NUMINAMATH_CALUDE_base5_division_l47_4752

/-- Converts a base 5 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 5 * acc + d) 0

/-- Converts a decimal number to its base 5 representation as a list of digits -/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else go (m / 5) ((m % 5) :: acc)
    go n []

/-- Theorem stating that the quotient of 1121₅ ÷ 12₅ in base 5 is equal to 43₅ -/
theorem base5_division :
  toBase5 (toDecimal [1, 1, 2, 1] / toDecimal [1, 2]) = [4, 3] := by
  sorry

end NUMINAMATH_CALUDE_base5_division_l47_4752


namespace NUMINAMATH_CALUDE_equation_result_l47_4736

theorem equation_result (x y : ℤ) (h1 : 2 * x - y = 20) (h2 : 3 * y^2 = 48) :
  3 * x + y = 40 := by
  sorry

end NUMINAMATH_CALUDE_equation_result_l47_4736


namespace NUMINAMATH_CALUDE_unique_solution_for_alpha_minus_one_l47_4798

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the functional equation
def SatisfiesEquation (f : RealFunction) (α : ℝ) : Prop :=
  ∀ x y : ℝ, f (f (x + y)) = f (x + y) + f x * f y + α * x * y

-- State the theorem
theorem unique_solution_for_alpha_minus_one (α : ℝ) (hα : α ≠ 0) :
  (∃ f : RealFunction, SatisfiesEquation f α) ↔ (α = -1 ∧ ∃ f : RealFunction, SatisfiesEquation f (-1) ∧ ∀ x, f x = x) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_for_alpha_minus_one_l47_4798


namespace NUMINAMATH_CALUDE_heartsuit_ratio_theorem_l47_4774

-- Define the ♡ operation
def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

-- Theorem statement
theorem heartsuit_ratio_theorem : 
  (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_heartsuit_ratio_theorem_l47_4774


namespace NUMINAMATH_CALUDE_product_of_fractions_l47_4772

theorem product_of_fractions : (2 : ℚ) / 3 * 5 / 8 * 1 / 4 = 5 / 48 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l47_4772


namespace NUMINAMATH_CALUDE_session_comparison_l47_4700

theorem session_comparison (a b : ℝ) : 
  a > 0 → -- Assuming a is positive (number of people)
  b = 0.9 * (1.1 * a) → 
  a > b := by
sorry

end NUMINAMATH_CALUDE_session_comparison_l47_4700


namespace NUMINAMATH_CALUDE_sum_k_squared_over_3_to_k_l47_4782

open Real

/-- The sum of the infinite series k^2 / 3^k from k = 1 to infinity is 7 -/
theorem sum_k_squared_over_3_to_k (S : ℝ) : 
  (∑' k, (k : ℝ)^2 / 3^k) = 7 :=
by sorry

end NUMINAMATH_CALUDE_sum_k_squared_over_3_to_k_l47_4782


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l47_4733

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 222 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 222 := by sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l47_4733


namespace NUMINAMATH_CALUDE_simplify_fraction_l47_4708

theorem simplify_fraction (x : ℝ) (hx : x = Real.sqrt 3) :
  (1 / (1 + x)) * (1 / (1 - x)) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l47_4708


namespace NUMINAMATH_CALUDE_cubic_roots_problem_l47_4777

theorem cubic_roots_problem (a b : ℝ) (r s : ℝ) : 
  (r^3 + a*r + b = 0) →
  (s^3 + a*s + b = 0) →
  ((r+3)^3 + a*(r+3) + b + 360 = 0) →
  ((s-2)^3 + a*(s-2) + b + 360 = 0) →
  (b = -1330/27 ∨ b = -6340/27) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_problem_l47_4777


namespace NUMINAMATH_CALUDE_circle_condition_l47_4766

-- Define the equation of the circle
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y + m = 0

-- Theorem statement
theorem circle_condition (m : ℝ) :
  (∃ x y : ℝ, circle_equation x y m) ↔ m < 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l47_4766


namespace NUMINAMATH_CALUDE_frank_kibble_ratio_l47_4706

-- Define the problem parameters
def initial_kibble : ℕ := 12
def remaining_kibble : ℕ := 7
def mary_total : ℕ := 2
def frank_afternoon : ℕ := 1

-- Define Frank's late evening amount
def frank_late_evening : ℕ := initial_kibble - remaining_kibble - mary_total - frank_afternoon

-- Theorem statement
theorem frank_kibble_ratio :
  frank_late_evening = 2 * frank_afternoon :=
sorry

end NUMINAMATH_CALUDE_frank_kibble_ratio_l47_4706


namespace NUMINAMATH_CALUDE_tourist_survival_l47_4773

theorem tourist_survival (initial : ℕ) (eaten : ℕ) (poison_fraction : ℚ) (recovery_fraction : ℚ) : 
  initial = 30 →
  eaten = 2 →
  poison_fraction = 1/2 →
  recovery_fraction = 1/7 →
  (initial - eaten - (initial - eaten) * poison_fraction + 
   (initial - eaten) * poison_fraction * recovery_fraction : ℚ) = 16 := by
sorry

end NUMINAMATH_CALUDE_tourist_survival_l47_4773


namespace NUMINAMATH_CALUDE_dividend_calculation_l47_4776

theorem dividend_calculation (quotient divisor remainder : ℝ) 
  (hq : quotient = -427.86)
  (hd : divisor = 52.7)
  (hr : remainder = -14.5) :
  (quotient * divisor) + remainder = -22571.122 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l47_4776


namespace NUMINAMATH_CALUDE_total_thumbtacks_l47_4771

/-- The number of boards tested -/
def boards_tested : ℕ := 120

/-- The number of thumbtacks used per board -/
def tacks_per_board : ℕ := 3

/-- The number of thumbtacks remaining in each can after testing -/
def tacks_remaining_per_can : ℕ := 30

/-- The number of cans used -/
def num_cans : ℕ := 3

/-- Theorem stating that the total number of thumbtacks in three full cans is 450 -/
theorem total_thumbtacks :
  boards_tested * tacks_per_board + num_cans * tacks_remaining_per_can = 450 :=
by sorry

end NUMINAMATH_CALUDE_total_thumbtacks_l47_4771


namespace NUMINAMATH_CALUDE_integer_triple_solution_l47_4735

theorem integer_triple_solution (x y z : ℤ) :
  x * (y + z) = y^2 + z^2 - 2 ∧
  y * (z + x) = z^2 + x^2 - 2 ∧
  z * (x + y) = x^2 + y^2 - 2 →
  (x = 1 ∧ y = 0 ∧ z = -1) ∨
  (x = 1 ∧ y = -1 ∧ z = 0) ∨
  (x = 0 ∧ y = 1 ∧ z = -1) ∨
  (x = 0 ∧ y = -1 ∧ z = 1) ∨
  (x = -1 ∧ y = 1 ∧ z = 0) ∨
  (x = -1 ∧ y = 0 ∧ z = 1) :=
by sorry


end NUMINAMATH_CALUDE_integer_triple_solution_l47_4735


namespace NUMINAMATH_CALUDE_cooking_participants_l47_4718

/-- The number of people in a curriculum group with various activities -/
structure CurriculumGroup where
  yoga : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  cookingAndYoga : ℕ
  allCurriculums : ℕ
  cookingAndWeaving : ℕ

/-- The total number of people studying cooking in the curriculum group -/
def totalCooking (g : CurriculumGroup) : ℕ :=
  g.cookingOnly + (g.cookingAndYoga - g.allCurriculums) + 
  (g.cookingAndWeaving - g.allCurriculums) + g.allCurriculums

/-- Theorem stating that the number of people studying cooking is 9 -/
theorem cooking_participants (g : CurriculumGroup) 
  (h1 : g.yoga = 25)
  (h2 : g.weaving = 8)
  (h3 : g.cookingOnly = 2)
  (h4 : g.cookingAndYoga = 7)
  (h5 : g.allCurriculums = 3)
  (h6 : g.cookingAndWeaving = 3) :
  totalCooking g = 9 := by
  sorry

end NUMINAMATH_CALUDE_cooking_participants_l47_4718


namespace NUMINAMATH_CALUDE_chief_permutations_l47_4784

/-- The number of letters in the word CHIEF -/
def word_length : ℕ := 5

/-- The total number of permutations of the word CHIEF -/
def total_permutations : ℕ := Nat.factorial word_length

/-- The number of permutations where I appears after E -/
def permutations_i_after_e : ℕ := total_permutations / 2

theorem chief_permutations :
  permutations_i_after_e = total_permutations / 2 :=
by sorry

end NUMINAMATH_CALUDE_chief_permutations_l47_4784


namespace NUMINAMATH_CALUDE_appended_number_divisibility_l47_4767

theorem appended_number_divisibility : ∃ n : ℕ, 
  27700 ≤ n ∧ n ≤ 27799 ∧ 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 12 → n % k = 0) :=
sorry

end NUMINAMATH_CALUDE_appended_number_divisibility_l47_4767


namespace NUMINAMATH_CALUDE_half_angle_formulas_l47_4730

/-- For a triangle with sides a, b, c, angle α opposite side a, and semi-perimeter p = (a + b + c) / 2,
    we prove the half-angle formulas for cos and sin. -/
theorem half_angle_formulas (a b c : ℝ) (α : Real) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_triangle : a < b + c ∧ b < a + c ∧ c < a + b) 
    (h_angle : α = Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) : 
  let p := (a + b + c) / 2
  (Real.cos (α / 2))^2 = p * (p - a) / (b * c) ∧ 
  (Real.sin (α / 2))^2 = (p - b) * (p - c) / (b * c) := by
sorry

end NUMINAMATH_CALUDE_half_angle_formulas_l47_4730


namespace NUMINAMATH_CALUDE_widgets_per_shipping_box_l47_4720

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.width * d.length * d.height

/-- Represents the packing scenario at the Widget Factory -/
structure WidgetPacking where
  cartonDimensions : BoxDimensions
  shippingBoxDimensions : BoxDimensions
  widgetsPerCarton : ℕ

/-- Theorem stating the number of widgets that can be shipped in each shipping box -/
theorem widgets_per_shipping_box (p : WidgetPacking) 
  (h1 : p.cartonDimensions = BoxDimensions.mk 4 4 5)
  (h2 : p.shippingBoxDimensions = BoxDimensions.mk 20 20 20)
  (h3 : p.widgetsPerCarton = 3) : 
  (boxVolume p.shippingBoxDimensions / boxVolume p.cartonDimensions) * p.widgetsPerCarton = 300 := by
  sorry


end NUMINAMATH_CALUDE_widgets_per_shipping_box_l47_4720


namespace NUMINAMATH_CALUDE_five_mile_taxi_cost_l47_4738

/-- Calculates the cost of a taxi ride given the base fare, cost per mile, and distance traveled. -/
def taxi_cost (base_fare : ℝ) (cost_per_mile : ℝ) (distance : ℝ) : ℝ :=
  base_fare + cost_per_mile * distance

/-- Proves that a 5-mile taxi ride costs $2.75 given the specified base fare and cost per mile. -/
theorem five_mile_taxi_cost :
  let base_fare : ℝ := 1.50
  let cost_per_mile : ℝ := 0.25
  let distance : ℝ := 5
  taxi_cost base_fare cost_per_mile distance = 2.75 := by
  sorry

end NUMINAMATH_CALUDE_five_mile_taxi_cost_l47_4738


namespace NUMINAMATH_CALUDE_volume_of_T_l47_4754

/-- The solid T in ℝ³ -/
def T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p
                   (|x| + |y| ≤ 2) ∧ (|x| + |z| ≤ 2) ∧ (|y| + |z| ≤ 2)}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The volume of T is 1664/81 -/
theorem volume_of_T : volume T = 1664 / 81 := by sorry

end NUMINAMATH_CALUDE_volume_of_T_l47_4754


namespace NUMINAMATH_CALUDE_saltwater_animals_count_l47_4723

-- Define the number of saltwater aquariums
def saltwater_aquariums : ℕ := 22

-- Define the number of animals per aquarium
def animals_per_aquarium : ℕ := 46

-- Theorem to prove the number of saltwater animals
theorem saltwater_animals_count :
  saltwater_aquariums * animals_per_aquarium = 1012 := by
  sorry

end NUMINAMATH_CALUDE_saltwater_animals_count_l47_4723


namespace NUMINAMATH_CALUDE_f_neg_six_value_l47_4775

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f (-x) = f x
axiom f_periodic : ∀ x, f (x + 6) = f x
axiom f_defined : ∀ x, -3 ≤ x → x ≤ 3 → f x = (x + 1) * (x - (1/2 : ℝ))

-- Theorem to prove
theorem f_neg_six_value : f (-6) = -1/2 := by sorry

end NUMINAMATH_CALUDE_f_neg_six_value_l47_4775


namespace NUMINAMATH_CALUDE_remainder_2007_div_25_l47_4763

theorem remainder_2007_div_25 : 2007 % 25 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2007_div_25_l47_4763


namespace NUMINAMATH_CALUDE_alternating_color_probability_l47_4728

/-- The number of white balls in the box -/
def num_white_balls : ℕ := 6

/-- The number of black balls in the box -/
def num_black_balls : ℕ := 6

/-- The total number of balls in the box -/
def total_balls : ℕ := num_white_balls + num_black_balls

/-- The number of ways to arrange white and black balls -/
def total_arrangements : ℕ := Nat.choose total_balls num_white_balls

/-- The number of arrangements where colors alternate -/
def alternating_arrangements : ℕ := 2

/-- The probability of drawing balls with alternating colors -/
def prob_alternating_colors : ℚ := alternating_arrangements / total_arrangements

theorem alternating_color_probability :
  prob_alternating_colors = 1 / 462 :=
by sorry

end NUMINAMATH_CALUDE_alternating_color_probability_l47_4728


namespace NUMINAMATH_CALUDE_interval_for_72_students_8_sample_l47_4795

/-- The interval of segmentation in systematic sampling -/
def interval_of_segmentation (total_students : ℕ) (sample_size : ℕ) : ℕ :=
  total_students / sample_size

/-- Theorem: The interval of segmentation for 72 students and sample size 8 is 9 -/
theorem interval_for_72_students_8_sample : 
  interval_of_segmentation 72 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_interval_for_72_students_8_sample_l47_4795


namespace NUMINAMATH_CALUDE_sampling_survey_most_suitable_l47_4704

/-- Represents a survey method -/
inductive SurveyMethod
  | ComprehensiveSurvey
  | SamplingSurvey

/-- Represents the characteristics of a nationwide survey -/
structure NationwideSurvey where
  population : Nat  -- Number of students
  geographical_spread : Nat  -- Measure of how spread out the students are
  resource_constraints : Nat  -- Measure of available resources for the survey

/-- Determines the most suitable survey method for a given nationwide survey -/
def most_suitable_survey_method (survey : NationwideSurvey) : SurveyMethod :=
  sorry

/-- Theorem stating that for a nationwide survey of primary and secondary school students' 
    homework time, the most suitable survey method is a sampling survey -/
theorem sampling_survey_most_suitable (survey : NationwideSurvey) :
  most_suitable_survey_method survey = SurveyMethod.SamplingSurvey :=
  sorry

end NUMINAMATH_CALUDE_sampling_survey_most_suitable_l47_4704


namespace NUMINAMATH_CALUDE_sammy_cheese_ratio_l47_4783

/-- Represents the number of slices in each pizza -/
def slices_per_pizza : ℕ := 12

/-- Represents the number of pizzas ordered -/
def total_pizzas : ℕ := 2

/-- Represents the number of slices Dean ate from the Hawaiian pizza -/
def dean_slices : ℕ := slices_per_pizza / 2

/-- Represents the number of slices Frank ate from the Hawaiian pizza -/
def frank_slices : ℕ := 3

/-- Represents the total number of slices left over -/
def leftover_slices : ℕ := 11

/-- Theorem stating the ratio of slices Sammy ate from the cheese pizza to the total slices of the cheese pizza -/
theorem sammy_cheese_ratio :
  ∃ (sammy_slices : ℕ),
    sammy_slices = slices_per_pizza - (leftover_slices - (slices_per_pizza - (dean_slices + frank_slices))) ∧
    sammy_slices * 3 = slices_per_pizza :=
by sorry

end NUMINAMATH_CALUDE_sammy_cheese_ratio_l47_4783


namespace NUMINAMATH_CALUDE_surface_area_of_cut_and_rearranged_solid_l47_4764

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the cuts made on the solid -/
structure Cuts where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Calculates the surface area of the new solid formed by cutting and rearranging -/
def surfaceArea (d : Dimensions) (c : Cuts) : ℝ :=
  2 * d.length * d.width +  -- top and bottom
  2 * d.length * d.height +  -- front and back
  2 * d.width * d.height    -- sides

/-- The main theorem to prove -/
theorem surface_area_of_cut_and_rearranged_solid
  (d : Dimensions)
  (c : Cuts)
  (h1 : d.length = 2 ∧ d.width = 1 ∧ d.height = 1)
  (h2 : c.first = 1/4 ∧ c.second = 5/12 ∧ c.third = 19/36) :
  surfaceArea d c = 10 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_cut_and_rearranged_solid_l47_4764


namespace NUMINAMATH_CALUDE_crate_stacking_probability_l47_4715

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Calculates the number of possible arrangements for a given configuration -/
def arrangements (a b c : ℕ) : ℕ := sorry

/-- The probability of stacking crates to a specific height -/
def stack_probability (crate_dims : CrateDimensions) (num_crates target_height : ℕ) : ℚ :=
  sorry

theorem crate_stacking_probability :
  let crate_dims : CrateDimensions := ⟨2, 5, 7⟩
  let num_crates : ℕ := 8
  let target_height : ℕ := 36
  stack_probability crate_dims num_crates target_height = 98 / 6561 := by sorry

end NUMINAMATH_CALUDE_crate_stacking_probability_l47_4715


namespace NUMINAMATH_CALUDE_max_silver_tokens_l47_4714

/-- Represents the state of tokens -/
structure TokenState where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange booth -/
structure ExchangeBooth where
  input_color : String
  input_amount : ℕ
  output_silver : ℕ
  output_other_color : String
  output_other_amount : ℕ

/-- Performs a single exchange at a booth if possible -/
def exchange (state : TokenState) (booth : ExchangeBooth) : Option TokenState :=
  sorry

/-- Performs all possible exchanges until no more are possible -/
def exchange_all (initial_state : TokenState) (booths : List ExchangeBooth) : TokenState :=
  sorry

/-- The main theorem to prove -/
theorem max_silver_tokens : 
  let initial_state : TokenState := ⟨100, 65, 0⟩
  let booths : List ExchangeBooth := [
    ⟨"red", 3, 1, "blue", 2⟩,
    ⟨"blue", 4, 1, "red", 2⟩
  ]
  let final_state := exchange_all initial_state booths
  final_state.silver = 65 :=
sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l47_4714


namespace NUMINAMATH_CALUDE_cannot_tile_with_sphinxes_l47_4748

/-- Represents a sphinx shape -/
structure Sphinx :=
  (angles : Finset ℝ)
  (smallTriangles : ℕ)

/-- Represents the large equilateral triangle -/
structure LargeTriangle :=
  (sideLength : ℕ)
  (smallTriangles : ℕ)
  (grayTriangles : ℕ)
  (whiteTriangles : ℕ)

/-- Theorem stating the impossibility of tiling the large triangle with sphinxes -/
theorem cannot_tile_with_sphinxes (s : Sphinx) (t : LargeTriangle) : 
  s.angles = {60, 120, 240} ∧ 
  s.smallTriangles = 6 ∧
  t.sideLength = 6 ∧
  t.smallTriangles = 21 ∧
  t.grayTriangles = 15 ∧
  t.whiteTriangles = 21 →
  ¬ ∃ (n : ℕ), n * s.smallTriangles = t.smallTriangles :=
by sorry

end NUMINAMATH_CALUDE_cannot_tile_with_sphinxes_l47_4748


namespace NUMINAMATH_CALUDE_intersection_M_N_l47_4799

def M : Set ℕ := {0, 2, 4, 8}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_M_N : M ∩ N = {0, 4, 8} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l47_4799


namespace NUMINAMATH_CALUDE_leap_year_53_mondays_probability_l47_4713

/-- A leap year has 366 days -/
def leapYearDays : ℕ := 366

/-- A leap year has 52 weeks plus 2 days -/
def leapYearWeeksAndDays : ℕ × ℕ := (52, 2)

/-- There are 7 possible days for a year to start on -/
def possibleStartDays : ℕ := 7

/-- The probability of a randomly selected leap year having 53 Mondays -/
def probLeapYear53Mondays : ℚ := 2 / 7

theorem leap_year_53_mondays_probability :
  probLeapYear53Mondays = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_leap_year_53_mondays_probability_l47_4713
