import Mathlib

namespace NUMINAMATH_CALUDE_meeting_point_x_coordinate_l305_30548

-- Define the river boundaries
def river_left : ℝ := 0
def river_right : ℝ := 25

-- Define the current speed
def current_speed : ℝ := 2

-- Define the starting positions
def mallard_start : ℝ × ℝ := (0, 0)
def wigeon_start : ℝ × ℝ := (25, 0)

-- Define the meeting point y-coordinate
def meeting_y : ℝ := 22

-- Define the speeds relative to water
def mallard_speed : ℝ := 4
def wigeon_speed : ℝ := 3

-- Theorem statement
theorem meeting_point_x_coordinate :
  ∃ (x : ℝ), 
    x > river_left ∧ 
    x < river_right ∧ 
    (∃ (t : ℝ), t > 0 ∧
      (mallard_start.1 + mallard_speed * t * Real.cos (Real.arctan ((meeting_y - mallard_start.2) / (x - mallard_start.1))) = x) ∧
      (wigeon_start.1 - wigeon_speed * t * Real.cos (Real.arctan ((meeting_y - wigeon_start.2) / (wigeon_start.1 - x))) = x) ∧
      (mallard_start.2 + (mallard_speed * Real.sin (Real.arctan ((meeting_y - mallard_start.2) / (x - mallard_start.1))) + current_speed) * t = meeting_y) ∧
      (wigeon_start.2 + (wigeon_speed * Real.sin (Real.arctan ((meeting_y - wigeon_start.2) / (wigeon_start.1 - x))) + current_speed) * t = meeting_y)) ∧
    x = 100 / 7 := by
  sorry

end NUMINAMATH_CALUDE_meeting_point_x_coordinate_l305_30548


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_length_l305_30583

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  /-- The length of a leg of the triangle -/
  leg : ℝ
  /-- The length of the hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- Condition that the hypotenuse is √2 times the leg -/
  hyp_leg_relation : hypotenuse = leg * Real.sqrt 2
  /-- Condition that the leg is positive -/
  leg_pos : leg > 0

/-- The theorem stating that for an isosceles right triangle with area 64, its hypotenuse is 16 -/
theorem isosceles_right_triangle_hypotenuse_length
  (t : IsoscelesRightTriangle)
  (area_eq : t.leg * t.leg / 2 = 64) :
  t.hypotenuse = 16 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_length_l305_30583


namespace NUMINAMATH_CALUDE_quadratic_factorization_l305_30534

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l305_30534


namespace NUMINAMATH_CALUDE_solution_is_five_binomial_coefficient_identity_l305_30514

-- Define A_x
def A (x : ℕ) : ℕ := x * (x - 1) * (x - 2)

-- Part 1: Prove that the solution to 3A_x^3 = 2A_{x+1}^2 + 6A_x^2 is x = 5
theorem solution_is_five : ∃ (x : ℕ), x > 3 ∧ 3 * (A x)^3 = 2 * (A (x + 1))^2 + 6 * (A x)^2 ∧ x = 5 := by
  sorry

-- Part 2: Prove that kC_n^k = nC_{n-1}^{k-1}
theorem binomial_coefficient_identity (n k : ℕ) (h : k ≤ n) : 
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_is_five_binomial_coefficient_identity_l305_30514


namespace NUMINAMATH_CALUDE_max_min_product_l305_30535

theorem max_min_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (sum_eq : a + b + c = 12) (prod_sum_eq : a * b + b * c + c * a = 30) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 9 * Real.sqrt 2 ∧
  ∃ (a' b' c' : ℝ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧
    a' + b' + c' = 12 ∧ a' * b' + b' * c' + c' * a' = 30 ∧
    min (a' * b') (min (b' * c') (c' * a')) = 9 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_min_product_l305_30535


namespace NUMINAMATH_CALUDE_subset_implies_m_values_l305_30556

def A (m : ℝ) : Set ℝ := {-1, 3, m^2}
def B (m : ℝ) : Set ℝ := {3, 2*m - 1}

theorem subset_implies_m_values (m : ℝ) : B m ⊆ A m → m = 0 ∨ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_values_l305_30556


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l305_30578

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem
theorem f_derivative_at_one : 
  (deriv f) 1 = 0 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l305_30578


namespace NUMINAMATH_CALUDE_wood_stove_burn_rate_l305_30555

/-- Wood stove burning rate problem -/
theorem wood_stove_burn_rate 
  (morning_duration : ℝ) 
  (afternoon_duration : ℝ)
  (morning_rate : ℝ) 
  (starting_wood : ℝ) 
  (ending_wood : ℝ) : 
  morning_duration = 4 →
  afternoon_duration = 4 →
  morning_rate = 2 →
  starting_wood = 30 →
  ending_wood = 3 →
  ∃ (afternoon_rate : ℝ), 
    afternoon_rate = (starting_wood - ending_wood - morning_duration * morning_rate) / afternoon_duration ∧ 
    afternoon_rate = 4.75 := by
  sorry

end NUMINAMATH_CALUDE_wood_stove_burn_rate_l305_30555


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l305_30592

theorem rectangle_dimension_change (b h : ℝ) (h_pos : 0 < h) (b_pos : 0 < b) :
  let new_base := 1.1 * b
  let new_height := h * (new_base * h) / (b * h) / new_base
  (h - new_height) / h = 1 / 11 := by sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l305_30592


namespace NUMINAMATH_CALUDE_base5_242_equals_base10_72_l305_30515

-- Define a function to convert a base 5 number to base 10
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

-- Theorem stating that 242 in base 5 is equal to 72 in base 10
theorem base5_242_equals_base10_72 :
  base5ToBase10 [2, 4, 2] = 72 := by
  sorry

end NUMINAMATH_CALUDE_base5_242_equals_base10_72_l305_30515


namespace NUMINAMATH_CALUDE_root_equation_m_value_l305_30516

theorem root_equation_m_value :
  ∀ m : ℝ, ((-4)^2 + m * (-4) - 20 = 0) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_m_value_l305_30516


namespace NUMINAMATH_CALUDE_equation_solutions_l305_30599

-- Define the equations
def equation1 (x : ℚ) : Prop := (1 - x) / 3 - 2 = x / 6

def equation2 (x : ℚ) : Prop := (x + 1) / (1/4) - (x - 2) / (1/2) = 5

-- State the theorem
theorem equation_solutions :
  (∃ x : ℚ, equation1 x ∧ x = -10/3) ∧
  (∃ x : ℚ, equation2 x ∧ x = -3/2) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l305_30599


namespace NUMINAMATH_CALUDE_combined_height_sara_joe_l305_30532

/-- The combined height of Sara and Joe is 120 inches -/
theorem combined_height_sara_joe : 
  ∀ (sara_height joe_height : ℕ),
  joe_height = 2 * sara_height + 6 →
  joe_height = 82 →
  sara_height + joe_height = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_combined_height_sara_joe_l305_30532


namespace NUMINAMATH_CALUDE_ball_distribution_proof_l305_30531

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem ball_distribution_proof (total_balls : ℕ) (box1_num box2_num : ℕ) : 
  total_balls = 7 ∧ box1_num = 2 ∧ box2_num = 3 →
  (choose total_balls box1_num.succ.succ) + 
  (choose total_balls box1_num.succ) + 
  (choose total_balls box1_num) = 91 := by
sorry

end NUMINAMATH_CALUDE_ball_distribution_proof_l305_30531


namespace NUMINAMATH_CALUDE_fifth_month_sale_l305_30564

theorem fifth_month_sale
  (sale1 sale2 sale3 sale4 sale6 : ℕ)
  (average : ℚ)
  (h1 : sale1 = 2500)
  (h2 : sale2 = 6500)
  (h3 : sale3 = 9855)
  (h4 : sale4 = 7230)
  (h6 : sale6 = 11915)
  (h_avg : average = 7500)
  (h_total : (sale1 + sale2 + sale3 + sale4 + sale6 + sale5) / 6 = average) :
  sale5 = 7000 := by
sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l305_30564


namespace NUMINAMATH_CALUDE_max_profit_at_70_best_selling_price_l305_30563

/-- Represents the profit function for a product with given pricing and demand characteristics -/
def profit (x : ℕ) : ℝ :=
  (50 + x - 40) * (50 - x)

/-- Theorem stating that the maximum profit occurs when the selling price is 70 yuan -/
theorem max_profit_at_70 :
  ∀ x : ℕ, x < 50 → x > 0 → profit x ≤ profit 20 :=
sorry

/-- Corollary stating that the best selling price is 70 yuan -/
theorem best_selling_price :
  ∃ x : ℕ, x < 50 ∧ x > 0 ∧ ∀ y : ℕ, y < 50 → y > 0 → profit y ≤ profit x :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_70_best_selling_price_l305_30563


namespace NUMINAMATH_CALUDE_manuscript_has_100_pages_l305_30594

/-- Represents the pricing and revision structure of a typing service --/
structure TypingService where
  initial_cost : ℕ  -- Cost per page for initial typing
  revision_cost : ℕ  -- Cost per page for each revision

/-- Represents the manuscript details --/
structure Manuscript where
  total_pages : ℕ
  once_revised : ℕ
  twice_revised : ℕ

/-- Calculates the total cost for typing and revising a manuscript --/
def total_cost (service : TypingService) (manuscript : Manuscript) : ℕ :=
  service.initial_cost * manuscript.total_pages +
  service.revision_cost * manuscript.once_revised +
  2 * service.revision_cost * manuscript.twice_revised

/-- Theorem stating that given the conditions, the manuscript has 100 pages --/
theorem manuscript_has_100_pages (service : TypingService) (manuscript : Manuscript) :
  service.initial_cost = 5 →
  service.revision_cost = 4 →
  manuscript.once_revised = 30 →
  manuscript.twice_revised = 20 →
  total_cost service manuscript = 780 →
  manuscript.total_pages = 100 :=
by
  sorry


end NUMINAMATH_CALUDE_manuscript_has_100_pages_l305_30594


namespace NUMINAMATH_CALUDE_equation_solution_l305_30544

theorem equation_solution (x : ℝ) (h : x ≠ -1) :
  (x^2 + x + 1) / (x + 1) = x + 3 ↔ x = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l305_30544


namespace NUMINAMATH_CALUDE_flag_combinations_l305_30510

def available_colors : ℕ := 6
def stripes : ℕ := 3

theorem flag_combinations : (available_colors * (available_colors - 1) * (available_colors - 2)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_flag_combinations_l305_30510


namespace NUMINAMATH_CALUDE_exact_two_out_of_three_germinate_l305_30511

/-- The probability of a single seed germinating -/
def p : ℚ := 4/5

/-- The total number of seeds -/
def n : ℕ := 3

/-- The number of seeds we want to germinate -/
def k : ℕ := 2

/-- The probability of exactly k out of n seeds germinating -/
def prob_k_out_of_n (p : ℚ) (n k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem exact_two_out_of_three_germinate :
  prob_k_out_of_n p n k = 48/125 := by sorry

end NUMINAMATH_CALUDE_exact_two_out_of_three_germinate_l305_30511


namespace NUMINAMATH_CALUDE_james_candy_payment_l305_30508

/-- Proves that James paid $20 for candy given the conditions of the problem -/
theorem james_candy_payment (
  num_packs : ℕ)
  (price_per_pack : ℕ)
  (change_received : ℕ)
  (h1 : num_packs = 3)
  (h2 : price_per_pack = 3)
  (h3 : change_received = 11)
  : num_packs * price_per_pack + change_received = 20 := by
  sorry

end NUMINAMATH_CALUDE_james_candy_payment_l305_30508


namespace NUMINAMATH_CALUDE_vegetarian_eaters_l305_30591

/-- Given a family with the following characteristics:
  - Total number of people: 45
  - Number of people who eat only vegetarian: 22
  - Number of people who eat only non-vegetarian: 15
  - Number of people who eat both vegetarian and non-vegetarian: 8
  Prove that the number of people who eat vegetarian meals is 30. -/
theorem vegetarian_eaters (total : ℕ) (only_veg : ℕ) (only_nonveg : ℕ) (both : ℕ)
  (h1 : total = 45)
  (h2 : only_veg = 22)
  (h3 : only_nonveg = 15)
  (h4 : both = 8) :
  only_veg + both = 30 := by
  sorry

end NUMINAMATH_CALUDE_vegetarian_eaters_l305_30591


namespace NUMINAMATH_CALUDE_problem_statement_l305_30541

theorem problem_statement (a b : ℝ) :
  (4 / (Real.sqrt 6 + Real.sqrt 2) - 1 / (Real.sqrt 3 + Real.sqrt 2) = Real.sqrt a - Real.sqrt b) →
  a - b = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l305_30541


namespace NUMINAMATH_CALUDE_f_properties_l305_30504

def f (x : ℝ) : ℝ := x^3 + 3*x^2

theorem f_properties :
  (f (-1) = 2) →
  (deriv f (-1) = -3) →
  (∃ (y : ℝ), y ∈ Set.Icc (-16) 4 ↔ ∃ (x : ℝ), x ∈ Set.Icc (-4) 0 ∧ f x = y) ∧
  (∀ (t : ℝ), (∀ (x y : ℝ), t ≤ x ∧ x < y ∧ y ≤ t + 1 → f x > f y) ↔ t ∈ Set.Icc (-2) (-1)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l305_30504


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l305_30551

theorem complex_fraction_sum : (1 - Complex.I) / (1 + Complex.I)^2 + (1 + Complex.I) / (1 - Complex.I)^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l305_30551


namespace NUMINAMATH_CALUDE_intersection_equals_B_l305_30561

def A : Set ℝ := {x | x^2 < 4}
def B : Set ℝ := {0, 1}

theorem intersection_equals_B : A ∩ B = B := by sorry

end NUMINAMATH_CALUDE_intersection_equals_B_l305_30561


namespace NUMINAMATH_CALUDE_x_plus_twice_y_l305_30582

theorem x_plus_twice_y (x y z : ℚ) : 
  x = y / 3 → y = z / 4 → z = 100 → x + 2 * y = 175 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_twice_y_l305_30582


namespace NUMINAMATH_CALUDE_smallest_repeating_block_of_8_11_l305_30537

theorem smallest_repeating_block_of_8_11 : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → (8 * 10^k) % 11 = (8 * 10^(k + n)) % 11) ∧
  (∀ (m : ℕ), m > 0 → m < n → ∃ (k : ℕ), k > 0 ∧ (8 * 10^k) % 11 ≠ (8 * 10^(k + m)) % 11) ∧
  n = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_repeating_block_of_8_11_l305_30537


namespace NUMINAMATH_CALUDE_expression_comparison_l305_30553

theorem expression_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  (∃ a b : ℝ, (a + 1/a) * (b + 1/b) > (Real.sqrt (a * b) + 1 / Real.sqrt (a * b))^2) ∧
  (∃ a b : ℝ, (a + 1/a) * (b + 1/b) > ((a + b)/2 + 2/(a + b))^2) ∧
  (∃ a b : ℝ, ((a + b)/2 + 2/(a + b))^2 > (a + 1/a) * (b + 1/b)) :=
by sorry

end NUMINAMATH_CALUDE_expression_comparison_l305_30553


namespace NUMINAMATH_CALUDE_sugar_consumption_reduction_l305_30523

theorem sugar_consumption_reduction (initial_price new_price : ℚ) 
  (h1 : initial_price = 10)
  (h2 : new_price = 13) :
  let reduction_percentage := (1 - initial_price / new_price) * 100
  reduction_percentage = 300 / 13 := by
sorry

end NUMINAMATH_CALUDE_sugar_consumption_reduction_l305_30523


namespace NUMINAMATH_CALUDE_ratio_equality_implies_sum_ratio_l305_30547

theorem ratio_equality_implies_sum_ratio (x y z : ℝ) :
  x / 3 = y / (-4) ∧ y / (-4) = z / 7 →
  (3 * x + y + z) / y = -3 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_implies_sum_ratio_l305_30547


namespace NUMINAMATH_CALUDE_ellipse_properties_l305_30577

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A hyperbola formed from an ellipse -/
structure Hyperbola (e : Ellipse) where
  is_equilateral : Bool

/-- A triangle formed by the left focus, right focus, and two points on the ellipse -/
structure Triangle (e : Ellipse) where
  perimeter : ℝ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ := sorry

/-- The maximum area of a triangle formed by the foci and two points on the ellipse -/
def max_triangle_area (e : Ellipse) (t : Triangle e) : ℝ := sorry

theorem ellipse_properties (e : Ellipse) (h : Hyperbola e) (t : Triangle e) :
  h.is_equilateral = true → t.perimeter = 8 →
    eccentricity e = Real.sqrt 2 / 2 ∧ max_triangle_area e t = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l305_30577


namespace NUMINAMATH_CALUDE_right_triangle_sets_l305_30533

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_sets :
  is_pythagorean_triple 8 15 17 ∧
  is_pythagorean_triple 7 24 25 ∧
  is_pythagorean_triple 3 4 5 ∧
  ¬ is_pythagorean_triple 2 3 4 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l305_30533


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l305_30589

/-- A hyperbola with the property that the distance from its vertex to its asymptote
    is 1/4 of the length of its imaginary axis has eccentricity 2. -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a > 0) (k : b > 0) :
  (a * b / Real.sqrt (a^2 + b^2) = 1/4 * (2*b)) → (Real.sqrt (a^2 + b^2) / a = 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l305_30589


namespace NUMINAMATH_CALUDE_commodity_cost_proof_l305_30587

def total_cost (price1 price2 : ℕ) : ℕ := price1 + price2

theorem commodity_cost_proof (price1 price2 : ℕ) 
  (h1 : price1 = 477)
  (h2 : price1 = price2 + 127) :
  total_cost price1 price2 = 827 := by
  sorry

end NUMINAMATH_CALUDE_commodity_cost_proof_l305_30587


namespace NUMINAMATH_CALUDE_range_of_m_l305_30545

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x y : ℝ, x^2 / (2*m) - y^2 / (m-2) = 1 → m > 2

def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  x₁^2 + (2*m-3)*x₁ + 1 = 0 ∧ x₂^2 + (2*m-3)*x₂ + 1 = 0

-- State the theorem
theorem range_of_m : 
  (∀ m : ℝ, ¬(p m ∧ q m)) → 
  (∀ m : ℝ, p m ∨ q m) → 
  ∀ m : ℝ, (2 < m ∧ m ≤ 5/2) ∨ m < 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l305_30545


namespace NUMINAMATH_CALUDE_alkaline_probability_is_two_fifths_l305_30538

/-- The number of total solutions -/
def total_solutions : ℕ := 5

/-- The number of alkaline solutions -/
def alkaline_solutions : ℕ := 2

/-- The probability of selecting an alkaline solution -/
def alkaline_probability : ℚ := alkaline_solutions / total_solutions

theorem alkaline_probability_is_two_fifths :
  alkaline_probability = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_alkaline_probability_is_two_fifths_l305_30538


namespace NUMINAMATH_CALUDE_part_one_part_two_l305_30580

-- Define the function f
def f (x k m : ℝ) : ℝ := |x^2 - k*x - m|

-- Theorem for part (1)
theorem part_one (k m : ℝ) :
  m = 2 * k^2 →
  (∀ x y, 1 < x ∧ x < y → f x k m < f y k m) →
  -1 ≤ k ∧ k ≤ 1/2 := by sorry

-- Theorem for part (2)
theorem part_two (k m a b : ℝ) :
  (∀ x, x ∈ Set.Icc a b → f x k m ≤ 1) →
  b - a ≤ 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l305_30580


namespace NUMINAMATH_CALUDE_base_k_conversion_l305_30597

theorem base_k_conversion (k : ℕ) : k > 0 ∧ 1 * k^2 + 3 * k + 2 = 42 ↔ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_k_conversion_l305_30597


namespace NUMINAMATH_CALUDE_cost_price_calculation_l305_30522

theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  selling_price = 21000 →
  discount_rate = 0.10 →
  profit_rate = 0.08 →
  (selling_price * (1 - discount_rate)) / (1 + profit_rate) = 17500 := by
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l305_30522


namespace NUMINAMATH_CALUDE_set_equality_from_intersection_union_equality_l305_30519

theorem set_equality_from_intersection_union_equality (A : Set α) :
  ∃ X, (X ∩ A = X ∪ A) → (X = A) := by sorry

end NUMINAMATH_CALUDE_set_equality_from_intersection_union_equality_l305_30519


namespace NUMINAMATH_CALUDE_courtyard_width_l305_30585

/-- The width of a rectangular courtyard given its length and paving stone requirements -/
theorem courtyard_width (length : Real) (num_stones : Nat) (stone_length stone_width : Real) 
  (h1 : length = 40)
  (h2 : num_stones = 132)
  (h3 : stone_length = 2.5)
  (h4 : stone_width = 2) :
  length * (num_stones * stone_length * stone_width / length) = 16.5 := by
  sorry

#check courtyard_width

end NUMINAMATH_CALUDE_courtyard_width_l305_30585


namespace NUMINAMATH_CALUDE_incorrect_operation_l305_30530

theorem incorrect_operation (a b c d e f : ℝ) 
  (h1 : a = Real.sqrt 2)
  (h2 : b = Real.sqrt 3)
  (h3 : c = Real.sqrt 5)
  (h4 : d = Real.sqrt 6)
  (h5 : e = Real.sqrt (1/2))
  (h6 : f = Real.sqrt 8)
  (prop1 : a * b = d)
  (prop2 : a / e = 2)
  (prop3 : a + f = 3 * a) :
  a + b ≠ c := by
  sorry

end NUMINAMATH_CALUDE_incorrect_operation_l305_30530


namespace NUMINAMATH_CALUDE_imaginary_sum_zero_l305_30503

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_sum_zero :
  i^13 + i^18 + i^23 + i^28 + i^33 + i^38 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_sum_zero_l305_30503


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l305_30570

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 - 6*p^3 + 23*p^2 - 72*p + 8 = 0) →
  (q^4 - 6*q^3 + 23*q^2 - 72*q + 8 = 0) →
  (r^4 - 6*r^3 + 23*r^2 - 72*r + 8 = 0) →
  (s^4 - 6*s^3 + 23*s^2 - 72*s + 8 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = -9 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l305_30570


namespace NUMINAMATH_CALUDE_water_in_sport_formulation_l305_30573

/-- Represents the ratio of ingredients in a flavored drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio :=
  { flavoring := 3 * standard_ratio.flavoring,
    corn_syrup := standard_ratio.corn_syrup,
    water := 60 * standard_ratio.flavoring }

/-- The amount of water in ounces given the amount of corn syrup in the sport formulation -/
def water_amount (corn_syrup_oz : ℚ) : ℚ :=
  corn_syrup_oz * (sport_ratio.water / sport_ratio.corn_syrup)

theorem water_in_sport_formulation :
  water_amount 2 = 120 :=
sorry

end NUMINAMATH_CALUDE_water_in_sport_formulation_l305_30573


namespace NUMINAMATH_CALUDE_ordered_pairs_count_l305_30540

theorem ordered_pairs_count : 
  { p : ℤ × ℤ | (p.1 : ℤ) ^ 2019 + (p.2 : ℤ) ^ 2 = 2 * (p.2 : ℤ) }.Finite ∧ 
  { p : ℤ × ℤ | (p.1 : ℤ) ^ 2019 + (p.2 : ℤ) ^ 2 = 2 * (p.2 : ℤ) }.ncard = 3 :=
by sorry

end NUMINAMATH_CALUDE_ordered_pairs_count_l305_30540


namespace NUMINAMATH_CALUDE_remaining_three_average_l305_30574

theorem remaining_three_average (total : ℕ) (all_avg first_four_avg next_three_avg following_two_avg : ℚ) :
  total = 12 →
  all_avg = 6.30 →
  first_four_avg = 5.60 →
  next_three_avg = 4.90 →
  following_two_avg = 7.25 →
  (total * all_avg - (4 * first_four_avg + 3 * next_three_avg + 2 * following_two_avg)) / 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remaining_three_average_l305_30574


namespace NUMINAMATH_CALUDE_winter_spending_calculation_l305_30576

/-- The amount spent by the Surf City government at the end of November 1988, in millions of dollars. -/
def spent_end_november : ℝ := 3.3

/-- The amount spent by the Surf City government at the end of February 1989, in millions of dollars. -/
def spent_end_february : ℝ := 7.0

/-- The amount spent during December, January, and February, in millions of dollars. -/
def winter_spending : ℝ := spent_end_february - spent_end_november

theorem winter_spending_calculation : winter_spending = 3.7 := by
  sorry

end NUMINAMATH_CALUDE_winter_spending_calculation_l305_30576


namespace NUMINAMATH_CALUDE_parabola_coefficient_l305_30558

def quadratic_function (a b c : ℤ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_coefficient
  (a b c : ℤ)
  (vertex_x vertex_y : ℝ)
  (point_x point_y : ℝ)
  (h_vertex : ∀ x, quadratic_function a b c x ≥ quadratic_function a b c vertex_x)
  (h_vertex_y : quadratic_function a b c vertex_x = vertex_y)
  (h_point : quadratic_function a b c point_x = point_y)
  (h_vertex_coords : vertex_x = 2 ∧ vertex_y = 3)
  (h_point_coords : point_x = 1 ∧ point_y = 0) :
  a = -3 := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l305_30558


namespace NUMINAMATH_CALUDE_min_value_theorem_l305_30521

theorem min_value_theorem (a : ℝ) (h : 8 * a^2 + 7 * a + 6 = 5) :
  ∃ (m : ℝ), (∀ x, 8 * x^2 + 7 * x + 6 = 5 → 3 * x + 2 ≥ m) ∧ (∃ y, 8 * y^2 + 7 * y + 6 = 5 ∧ 3 * y + 2 = m) ∧ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l305_30521


namespace NUMINAMATH_CALUDE_distance_between_points_l305_30505

/-- The distance between two points (-3, 5) and (4, -9) is √245 -/
theorem distance_between_points : Real.sqrt 245 = Real.sqrt ((4 - (-3))^2 + (-9 - 5)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l305_30505


namespace NUMINAMATH_CALUDE_problem_statement_l305_30575

theorem problem_statement (θ : ℝ) : 
  ((∀ x : ℝ, x^2 - 2*x*Real.sin θ + 1 ≥ 0) ∨ 
   (∀ α β : ℝ, Real.sin (α + β) ≤ Real.sin α + Real.sin β)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l305_30575


namespace NUMINAMATH_CALUDE_no_valid_partition_l305_30546

/-- A partition of integers into three subsets -/
def IntPartition := ℤ → Fin 3

/-- Property that n, n-50, and n+1987 belong to different subsets -/
def ValidPartition (p : IntPartition) : Prop :=
  ∀ n : ℤ, p n ≠ p (n - 50) ∧ p n ≠ p (n + 1987) ∧ p (n - 50) ≠ p (n + 1987)

/-- Theorem stating the impossibility of such a partition -/
theorem no_valid_partition : ¬ ∃ p : IntPartition, ValidPartition p := by
  sorry

end NUMINAMATH_CALUDE_no_valid_partition_l305_30546


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l305_30569

theorem rectangle_dimensions (x : ℝ) : 
  (x - 3 > 0) →
  (3 * x + 4 > 0) →
  ((x - 3) * (3 * x + 4) = 12 * x - 9) →
  (x = (17 + 5 * Real.sqrt 13) / 6) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l305_30569


namespace NUMINAMATH_CALUDE_power_five_137_mod_8_l305_30509

theorem power_five_137_mod_8 : 5^137 % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_five_137_mod_8_l305_30509


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_l305_30550

theorem smallest_resolvable_debt (pig_value goat_value : ℕ) 
  (pig_value_pos : pig_value > 0) (goat_value_pos : goat_value > 0) :
  ∃ (debt : ℕ), debt > 0 ∧ 
  (∃ (p g : ℤ), debt = pig_value * p + goat_value * g) ∧
  (∀ (d : ℕ), d > 0 → (∃ (p g : ℤ), d = pig_value * p + goat_value * g) → d ≥ debt) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_l305_30550


namespace NUMINAMATH_CALUDE_P_on_angle_bisector_PQ_parallel_to_x_axis_l305_30562

-- Define points P and Q
def P (a : ℝ) : ℝ × ℝ := (a + 1, 2 * a - 3)
def Q : ℝ × ℝ := (2, 3)

-- Theorem for the first condition
theorem P_on_angle_bisector :
  ∃ a : ℝ, P a = (5, 5) ∧ (P a).1 = (P a).2 := by sorry

-- Theorem for the second condition
theorem PQ_parallel_to_x_axis :
  ∃ a : ℝ, (P a).2 = Q.2 → |((P a).1 - Q.1)| = 2 := by sorry

end NUMINAMATH_CALUDE_P_on_angle_bisector_PQ_parallel_to_x_axis_l305_30562


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l305_30559

theorem smallest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 13 ∣ n → 104 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l305_30559


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_23_l305_30542

theorem greatest_three_digit_multiple_of_23 : ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 23 = 0 → n ≤ 991 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_23_l305_30542


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l305_30536

/-- Acme T-Shirt Company's pricing function -/
def acme_price (x : ℕ) : ℕ := 60 + 8 * x

/-- Delta T-shirt Company's pricing function -/
def delta_price (x : ℕ) : ℕ := 12 * x

/-- The minimum number of shirts for which Acme is cheaper than Delta -/
def min_shirts_for_acme_cheaper : ℕ := 16

theorem acme_cheaper_at_min_shirts :
  acme_price min_shirts_for_acme_cheaper < delta_price min_shirts_for_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_for_acme_cheaper →
    acme_price n ≥ delta_price n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l305_30536


namespace NUMINAMATH_CALUDE_tiffany_phone_pictures_l305_30557

theorem tiffany_phone_pictures :
  ∀ (phone_pics camera_pics total_pics num_albums pics_per_album : ℕ),
    camera_pics = 13 →
    num_albums = 5 →
    pics_per_album = 4 →
    total_pics = num_albums * pics_per_album →
    total_pics = phone_pics + camera_pics →
    phone_pics = 7 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_phone_pictures_l305_30557


namespace NUMINAMATH_CALUDE_households_using_only_brand_A_l305_30518

/-- The number of households that use only brand A soap -/
def only_brand_A : ℕ := 60

/-- The number of households that use only brand B soap -/
def only_brand_B : ℕ := 75

/-- The number of households that use both brand A and brand B soap -/
def both_brands : ℕ := 25

/-- The number of households that use neither brand A nor brand B soap -/
def neither_brand : ℕ := 80

/-- The total number of households surveyed -/
def total_households : ℕ := 240

/-- Theorem stating that the number of households using only brand A soap is 60 -/
theorem households_using_only_brand_A :
  only_brand_A = total_households - only_brand_B - both_brands - neither_brand :=
by sorry

end NUMINAMATH_CALUDE_households_using_only_brand_A_l305_30518


namespace NUMINAMATH_CALUDE_trap_existence_for_specific_feeders_l305_30502

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- A feeder is an interval that contains infinitely many terms of the sequence. -/
def IsFeeder (s : Sequence) (a b : ℝ) : Prop :=
  ∀ n : ℕ, ∃ m ≥ n, a ≤ s m ∧ s m ≤ b

/-- A trap is an interval that contains all but finitely many terms of the sequence. -/
def IsTrap (s : Sequence) (a b : ℝ) : Prop :=
  ∃ N : ℕ, ∀ n ≥ N, a ≤ s n ∧ s n ≤ b

/-- Main theorem about traps in sequences with specific feeders. -/
theorem trap_existence_for_specific_feeders (s : Sequence) 
  (h1 : IsFeeder s 0 1) (h2 : IsFeeder s 9 10) : 
  (¬ ∃ a : ℝ, IsTrap s a (a + 1)) ∧
  (∃ a : ℝ, IsTrap s a (a + 9)) := by sorry


end NUMINAMATH_CALUDE_trap_existence_for_specific_feeders_l305_30502


namespace NUMINAMATH_CALUDE_square_perimeter_l305_30517

theorem square_perimeter (s : ℝ) (h1 : s > 0) (h2 : s^2 = 625) : 4 * s = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l305_30517


namespace NUMINAMATH_CALUDE_fruit_weight_assignment_is_correct_l305_30567

/-- Represents the fruits in the problem -/
inductive Fruit
  | orange
  | banana
  | mandarin
  | peach
  | apple

/-- Assigns weights to fruits -/
def weight_assignment : Fruit → ℕ
  | Fruit.orange   => 280
  | Fruit.banana   => 170
  | Fruit.mandarin => 100
  | Fruit.peach    => 200
  | Fruit.apple    => 150

/-- The set of possible weights -/
def possible_weights : Set ℕ := {100, 150, 170, 200, 280}

theorem fruit_weight_assignment_is_correct :
  (∀ f : Fruit, weight_assignment f ∈ possible_weights) ∧
  (weight_assignment Fruit.peach < weight_assignment Fruit.orange) ∧
  (weight_assignment Fruit.apple < weight_assignment Fruit.banana) ∧
  (weight_assignment Fruit.banana < weight_assignment Fruit.peach) ∧
  (weight_assignment Fruit.mandarin < weight_assignment Fruit.banana) ∧
  (weight_assignment Fruit.apple + weight_assignment Fruit.banana > weight_assignment Fruit.orange) ∧
  (∀ w : Fruit → ℕ, 
    (∀ f : Fruit, w f ∈ possible_weights) →
    (w Fruit.peach < w Fruit.orange) →
    (w Fruit.apple < w Fruit.banana) →
    (w Fruit.banana < w Fruit.peach) →
    (w Fruit.mandarin < w Fruit.banana) →
    (w Fruit.apple + w Fruit.banana > w Fruit.orange) →
    w = weight_assignment) :=
by sorry

end NUMINAMATH_CALUDE_fruit_weight_assignment_is_correct_l305_30567


namespace NUMINAMATH_CALUDE_cyclists_meeting_time_l305_30572

/-- Two cyclists moving in opposite directions on a circular track meet at the starting point -/
theorem cyclists_meeting_time
  (circumference : ℝ)
  (speed1 : ℝ)
  (speed2 : ℝ)
  (h1 : circumference = 675)
  (h2 : speed1 = 7)
  (h3 : speed2 = 8) :
  circumference / (speed1 + speed2) = 45 :=
by sorry

end NUMINAMATH_CALUDE_cyclists_meeting_time_l305_30572


namespace NUMINAMATH_CALUDE_parallel_lines_parameter_l305_30571

/-- Given two lines in the plane, prove that the parameter 'a' must equal 4 for the lines to be parallel -/
theorem parallel_lines_parameter (a : ℝ) : 
  (∀ x y : ℝ, 3 * x + (1 - a) * y + 1 = 0 ↔ x - y + 2 = 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_parameter_l305_30571


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_2_to_20_l305_30524

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_2_to_20 :
  arithmetic_sequence_sum 2 2 10 = 110 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_2_to_20_l305_30524


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l305_30539

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a)
    (h_sum : a 6 + a 8 = 10)
    (h_a3 : a 3 = 1) :
    a 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l305_30539


namespace NUMINAMATH_CALUDE_bounded_recurrence_sequence_is_constant_two_l305_30529

/-- A sequence of natural numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → a n = (a (n - 1) + a (n - 2)) / Nat.gcd (a (n - 1)) (a (n - 2))

/-- A sequence is bounded if there exists an upper bound for all its terms -/
def BoundedSequence (a : ℕ → ℕ) : Prop :=
  ∃ M : ℕ, ∀ n : ℕ, a n ≤ M

theorem bounded_recurrence_sequence_is_constant_two (a : ℕ → ℕ) 
  (h_recurrence : RecurrenceSequence a) (h_bounded : BoundedSequence a) :
  ∀ n : ℕ, a n = 2 := by
  sorry

end NUMINAMATH_CALUDE_bounded_recurrence_sequence_is_constant_two_l305_30529


namespace NUMINAMATH_CALUDE_max_value_d_l305_30590

theorem max_value_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10)
  (sum_prod_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 ∧ 
  ∃ a b c d, a + b + c + d = 10 ∧ 
             a*b + a*c + a*d + b*c + b*d + c*d = 20 ∧
             d = (5 + Real.sqrt 105) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_d_l305_30590


namespace NUMINAMATH_CALUDE_angle_B_value_l305_30552

theorem angle_B_value (a b c : ℝ) (h : a^2 + c^2 - b^2 = Real.sqrt 3 * a * c) :
  let B := Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))
  B = π / 6 := by
sorry

end NUMINAMATH_CALUDE_angle_B_value_l305_30552


namespace NUMINAMATH_CALUDE_orchard_fruit_count_l305_30543

def apple_trees : ℕ := 50
def orange_trees : ℕ := 30
def apple_baskets_per_tree : ℕ := 25
def orange_baskets_per_tree : ℕ := 15
def apples_per_basket : ℕ := 18
def oranges_per_basket : ℕ := 12

theorem orchard_fruit_count :
  let total_apples := apple_trees * apple_baskets_per_tree * apples_per_basket
  let total_oranges := orange_trees * orange_baskets_per_tree * oranges_per_basket
  total_apples = 22500 ∧ total_oranges = 5400 := by
  sorry

end NUMINAMATH_CALUDE_orchard_fruit_count_l305_30543


namespace NUMINAMATH_CALUDE_simplify_expression_l305_30579

theorem simplify_expression :
  (3/2) * Real.sqrt 5 - (1/3) * Real.sqrt 6 + (1/2) * (-Real.sqrt 5 + 2 * Real.sqrt 6) =
  Real.sqrt 5 + (2/3) * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l305_30579


namespace NUMINAMATH_CALUDE_sum_squares_quadratic_roots_l305_30588

theorem sum_squares_quadratic_roots : 
  let a := 1
  let b := -10
  let c := 9
  let s₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let s₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  s₁^2 + s₂^2 = 82 :=
by sorry

end NUMINAMATH_CALUDE_sum_squares_quadratic_roots_l305_30588


namespace NUMINAMATH_CALUDE_right_prism_cross_section_type_l305_30513

/-- Represents a right prism -/
structure RightPrism where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a cross-section of a prism -/
inductive CrossSection
  | GeneralTrapezoid
  | IsoscelesTrapezoid
  | Other

/-- Function to determine the type of cross-section through the centers of base faces -/
def crossSectionThroughCenters (prism : RightPrism) : CrossSection :=
  sorry

/-- Theorem stating that the cross-section through the centers of base faces
    of a right prism is either a general trapezoid or an isosceles trapezoid -/
theorem right_prism_cross_section_type (prism : RightPrism) :
  (crossSectionThroughCenters prism = CrossSection.GeneralTrapezoid) ∨
  (crossSectionThroughCenters prism = CrossSection.IsoscelesTrapezoid) :=
by
  sorry

end NUMINAMATH_CALUDE_right_prism_cross_section_type_l305_30513


namespace NUMINAMATH_CALUDE_complex_modulus_l305_30598

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l305_30598


namespace NUMINAMATH_CALUDE_speed_conversion_l305_30525

/-- Proves that a speed of 36.003 km/h is equivalent to 10.0008 meters per second. -/
theorem speed_conversion (speed_kmh : ℝ) (speed_ms : ℝ) : 
  speed_kmh = 36.003 ∧ speed_ms = 10.0008 → speed_kmh * (1000 / 3600) = speed_ms := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l305_30525


namespace NUMINAMATH_CALUDE_e_value_proof_l305_30595

theorem e_value_proof (a b c : ℕ) (e : ℚ) 
  (h1 : a = 105)
  (h2 : b = 126)
  (h3 : c = 63)
  (h4 : a^3 - b^2 + c^2 = 21 * 25 * 45 * e) :
  e = 47.7 := by
  sorry

end NUMINAMATH_CALUDE_e_value_proof_l305_30595


namespace NUMINAMATH_CALUDE_max_value_expression_l305_30554

theorem max_value_expression (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x / abs x + abs y / y - (x * y) / abs (x * y) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l305_30554


namespace NUMINAMATH_CALUDE_train_length_l305_30526

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 180 → time_s = 7 → speed_kmh * (1000 / 3600) * time_s = 350 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l305_30526


namespace NUMINAMATH_CALUDE_no_solution_exists_l305_30566

theorem no_solution_exists : ∀ k : ℕ, k^6 + k^4 + k^2 ≠ 10^(k+1) + 9 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l305_30566


namespace NUMINAMATH_CALUDE_magic_square_sum_l305_30593

/-- Represents a 3x3 magic square with five unknown values -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- The magic sum (sum of each row, column, and diagonal) -/
def magicSum (sq : MagicSquare) : ℕ := 15 + sq.b + 27

/-- Conditions for the magic square -/
def isMagicSquare (sq : MagicSquare) : Prop :=
  magicSum sq = 15 + sq.b + 27
  ∧ magicSum sq = 24 + sq.a + sq.d
  ∧ magicSum sq = sq.e + 18 + sq.c
  ∧ magicSum sq = 15 + sq.a + sq.c
  ∧ magicSum sq = sq.b + sq.a + 18
  ∧ magicSum sq = 27 + sq.d + sq.c
  ∧ magicSum sq = 15 + sq.a + sq.c
  ∧ magicSum sq = 27 + sq.a + sq.e

theorem magic_square_sum (sq : MagicSquare) (h : isMagicSquare sq) : sq.d + sq.e = 47 := by
  sorry


end NUMINAMATH_CALUDE_magic_square_sum_l305_30593


namespace NUMINAMATH_CALUDE_line_BC_equation_triangle_ABC_area_l305_30520

-- Define the points of the triangle
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (5, -2)
def C : ℝ × ℝ := (3, 5)

-- Define the line equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def ABC : Triangle := { A := A, B := B, C := C }

-- Theorem for the equation of line BC
theorem line_BC_equation (t : Triangle) (l : Line) : 
  t = ABC → l.a = 7 ∧ l.b = 2 ∧ l.c = -31 → 
  l.a * t.B.1 + l.b * t.B.2 + l.c = 0 ∧
  l.a * t.C.1 + l.b * t.C.2 + l.c = 0 :=
sorry

-- Theorem for the area of triangle ABC
theorem triangle_ABC_area (t : Triangle) : 
  t = ABC → (1/2) * |t.A.1 * (t.B.2 - t.C.2) + t.B.1 * (t.C.2 - t.A.2) + t.C.1 * (t.A.2 - t.B.2)| = 29/2 :=
sorry

end NUMINAMATH_CALUDE_line_BC_equation_triangle_ABC_area_l305_30520


namespace NUMINAMATH_CALUDE_money_distribution_l305_30527

/-- Given three people A, B, and C with money amounts a, b, and c respectively,
    if their total amount is 500, B and C together have 310, and C has 10,
    then A and C together have 200. -/
theorem money_distribution (a b c : ℕ) : 
  a + b + c = 500 → b + c = 310 → c = 10 → a + c = 200 := by
  sorry

#check money_distribution

end NUMINAMATH_CALUDE_money_distribution_l305_30527


namespace NUMINAMATH_CALUDE_divisor_count_relation_l305_30565

-- Define a function to count divisors
def count_divisors (x : ℕ) : ℕ := sorry

-- Theorem statement
theorem divisor_count_relation (n : ℕ) :
  n > 0 → count_divisors (210 * n^3) = 210 → count_divisors (64 * n^5) = 22627 :=
by sorry

end NUMINAMATH_CALUDE_divisor_count_relation_l305_30565


namespace NUMINAMATH_CALUDE_trig_identity_l305_30500

theorem trig_identity : 
  (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.sin (70 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l305_30500


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l305_30581

theorem restaurant_bill_proof : 
  ∀ (n : ℕ) (total_friends : ℕ) (paying_friends : ℕ) (extra_amount : ℕ),
    total_friends = 10 →
    paying_friends = 9 →
    extra_amount = 3 →
    n = (paying_friends * (n / total_friends + extra_amount)) →
    n = 270 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l305_30581


namespace NUMINAMATH_CALUDE_walnut_trees_after_planting_l305_30501

/-- The number of walnut trees in the park after planting -/
def total_trees (initial_trees planted_trees : ℕ) : ℕ :=
  initial_trees + planted_trees

/-- Theorem: The total number of walnut trees after planting is 55 -/
theorem walnut_trees_after_planting :
  total_trees 22 33 = 55 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_after_planting_l305_30501


namespace NUMINAMATH_CALUDE_lowest_degree_polynomial_l305_30528

/-- A polynomial with coefficients in ℤ -/
def IntPolynomial := ℕ → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial) : ℕ := sorry

/-- The set of coefficients of a polynomial -/
def coefficients (p : IntPolynomial) : Set ℤ := sorry

/-- Predicate for a polynomial satisfying the given conditions -/
def satisfies_conditions (p : IntPolynomial) : Prop :=
  ∃ b : ℤ, (∃ x ∈ coefficients p, x < b) ∧
            (∃ y ∈ coefficients p, y > b) ∧
            b ∉ coefficients p

/-- The main theorem -/
theorem lowest_degree_polynomial :
  ∃ p : IntPolynomial, satisfies_conditions p ∧
    degree p = 4 ∧
    ∀ q : IntPolynomial, satisfies_conditions q → degree q ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_lowest_degree_polynomial_l305_30528


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l305_30549

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : ArithmeticSequence a)
  (h_condition : a 1 + 3 * a 8 + a 15 = 120) :
  2 * a 6 - a 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l305_30549


namespace NUMINAMATH_CALUDE_molecular_weight_8_moles_Al2O3_l305_30596

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

/-- Theorem: The molecular weight of 8 moles of Al2O3 is 815.68 grams -/
theorem molecular_weight_8_moles_Al2O3 :
  num_moles * molecular_weight_Al2O3 = 815.68 := by
  sorry


end NUMINAMATH_CALUDE_molecular_weight_8_moles_Al2O3_l305_30596


namespace NUMINAMATH_CALUDE_alex_is_26_l305_30560

-- Define the ages as natural numbers
def inez_age : ℕ := 18
def zack_age : ℕ := inez_age + 5
def jose_age : ℕ := zack_age - 3
def alex_age : ℕ := jose_age + 6

-- Theorem to prove
theorem alex_is_26 : alex_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_alex_is_26_l305_30560


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l305_30507

/-- The function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (a-2)*x^2 + a*x - 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*(a-2)*x + a

/-- Theorem: If f'(x) is even, then the tangent line at (1, f(1)) is 5x - y - 3 = 0 -/
theorem tangent_line_at_one (a : ℝ) :
  (∀ x, f' a x = f' a (-x)) →
  ∃ m b, ∀ x y, y = m*x + b ↔ y - f a 1 = (f' a 1) * (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l305_30507


namespace NUMINAMATH_CALUDE_complement_of_M_l305_30506

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 3, 5}

theorem complement_of_M : Mᶜ = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l305_30506


namespace NUMINAMATH_CALUDE_range_m_prop_p_range_m_prop_p_not_q_l305_30584

/-- Proposition p: For all real x, x²-2mx-3m > 0 -/
def prop_p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*m*x - 3*m > 0

/-- Proposition q: There exists a real x such that x²+4mx+1 < 0 -/
def prop_q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 4*m*x + 1 < 0

/-- The range of m for which proposition p is true -/
theorem range_m_prop_p : 
  {m : ℝ | prop_p m} = Set.Ioo (-3) 0 :=
sorry

/-- The range of m for which proposition p is true and proposition q is false -/
theorem range_m_prop_p_not_q : 
  {m : ℝ | prop_p m ∧ ¬(prop_q m)} = Set.Ico (-1/2) 0 :=
sorry

end NUMINAMATH_CALUDE_range_m_prop_p_range_m_prop_p_not_q_l305_30584


namespace NUMINAMATH_CALUDE_three_digit_numbers_satisfying_condition_l305_30568

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def satisfies_condition (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  n = 2 * ((10 * a + b) + (10 * b + c) + (10 * a + c))

def solution_set : Set ℕ := {134, 144, 150, 288, 294}

theorem three_digit_numbers_satisfying_condition :
  ∀ n : ℕ, is_valid_number n ∧ satisfies_condition n ↔ n ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_three_digit_numbers_satisfying_condition_l305_30568


namespace NUMINAMATH_CALUDE_complementary_angle_l305_30586

theorem complementary_angle (A : ℝ) (h : A = 25) : 90 - A = 65 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angle_l305_30586


namespace NUMINAMATH_CALUDE_problem_solution_l305_30512

def solution : Set (ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) :=
  {(3, 2, 1, 3, 2, 1), (6, 1, 1, 2, 2, 2), (7, 1, 1, 3, 3, 1), (8, 1, 1, 5, 2, 1),
   (2, 2, 2, 6, 1, 1), (3, 3, 1, 7, 1, 1), (5, 2, 1, 8, 1, 1)}

def satisfies_conditions (t : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) : Prop :=
  let (a, b, c, x, y, z) := t
  a + b + c = x * y * z ∧
  x + y + z = a * b * c ∧
  a ≥ b ∧ b ≥ c ∧ c ≥ 1 ∧
  x ≥ y ∧ y ≥ z ∧ z ≥ 1

theorem problem_solution :
  ∀ t : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ, satisfies_conditions t ↔ t ∈ solution := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_problem_solution_l305_30512
