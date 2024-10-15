import Mathlib

namespace NUMINAMATH_CALUDE_half_vector_MN_l2673_267369

/-- Given two vectors OM and ON in ℝ², prove that half of vector MN is (1/2, -4) -/
theorem half_vector_MN (OM ON : ℝ × ℝ) (h1 : OM = (-2, 3)) (h2 : ON = (-1, -5)) :
  (1 / 2 : ℝ) • (ON - OM) = (1/2, -4) := by
  sorry

end NUMINAMATH_CALUDE_half_vector_MN_l2673_267369


namespace NUMINAMATH_CALUDE_total_spent_is_20_27_l2673_267378

/-- Calculates the total amount spent on items with discount and tax --/
def totalSpent (initialAmount : ℚ) (candyPrice : ℚ) (chocolatePrice : ℚ) (gumPrice : ℚ) 
  (chipsPrice : ℚ) (discountRate : ℚ) (taxRate : ℚ) : ℚ :=
  let discountedCandyPrice := candyPrice * (1 - discountRate)
  let subtotal := discountedCandyPrice + chocolatePrice + gumPrice + chipsPrice
  let tax := subtotal * taxRate
  subtotal + tax

/-- Theorem stating that the total amount spent is $20.27 --/
theorem total_spent_is_20_27 : 
  totalSpent 50 7 6 3 4 (10/100) (5/100) = 2027/100 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_20_27_l2673_267378


namespace NUMINAMATH_CALUDE_largest_root_bound_l2673_267347

/-- A polynomial of degree 4 with constrained coefficients -/
def ConstrainedPoly (b a₂ a₁ a₀ : ℝ) : ℝ → ℝ :=
  fun x ↦ x^4 + b*x^3 + a₂*x^2 + a₁*x + a₀

/-- The set of all constrained polynomials -/
def ConstrainedPolySet : Set (ℝ → ℝ) :=
  {p | ∃ b a₂ a₁ a₀, |b| < 3 ∧ |a₂| < 2 ∧ |a₁| < 2 ∧ |a₀| < 2 ∧ p = ConstrainedPoly b a₂ a₁ a₀}

theorem largest_root_bound :
  (∃ p ∈ ConstrainedPolySet, ∃ r, 3 < r ∧ r < 4 ∧ p r = 0) ∧
  (∀ p ∈ ConstrainedPolySet, ∀ r ≥ 4, p r ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_largest_root_bound_l2673_267347


namespace NUMINAMATH_CALUDE_monomial_properties_l2673_267398

/-- Represents a monomial in two variables -/
structure Monomial (α : Type*) [Ring α] where
  coefficient : α
  exponent_a : ℕ
  exponent_b : ℕ

/-- Calculate the degree of a monomial -/
def Monomial.degree {α : Type*} [Ring α] (m : Monomial α) : ℕ :=
  m.exponent_a + m.exponent_b

/-- The monomial -2a²b -/
def example_monomial : Monomial ℤ :=
  { coefficient := -2
    exponent_a := 2
    exponent_b := 1 }

theorem monomial_properties :
  (example_monomial.coefficient = -2) ∧
  (example_monomial.degree = 3) := by
  sorry

end NUMINAMATH_CALUDE_monomial_properties_l2673_267398


namespace NUMINAMATH_CALUDE_crayon_selection_count_l2673_267390

/-- The number of crayons in the box -/
def total_crayons : ℕ := 15

/-- The number of crayons Karl must select -/
def crayons_to_select : ℕ := 5

/-- The number of non-red crayons to select -/
def non_red_to_select : ℕ := crayons_to_select - 1

/-- The number of non-red crayons available -/
def available_non_red : ℕ := total_crayons - 1

theorem crayon_selection_count :
  (Nat.choose available_non_red non_red_to_select) = 1001 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_count_l2673_267390


namespace NUMINAMATH_CALUDE_opposite_roots_quadratic_l2673_267391

theorem opposite_roots_quadratic (k : ℝ) : 
  (∃ x y : ℝ, x^2 + (k^2 - 1)*x + k + 1 = 0 ∧ 
               y^2 + (k^2 - 1)*y + k + 1 = 0 ∧ 
               x = -y ∧ x ≠ y) → 
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_opposite_roots_quadratic_l2673_267391


namespace NUMINAMATH_CALUDE_squirrel_acorns_l2673_267371

theorem squirrel_acorns (num_squirrels : ℕ) (acorns_collected : ℕ) (acorns_needed_per_squirrel : ℕ) :
  num_squirrels = 7 →
  acorns_collected = 875 →
  acorns_needed_per_squirrel = 170 →
  (acorns_needed_per_squirrel * num_squirrels - acorns_collected) / num_squirrels = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l2673_267371


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2673_267361

theorem simplify_square_roots : Real.sqrt 81 - Real.sqrt 144 = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2673_267361


namespace NUMINAMATH_CALUDE_distance_covered_l2673_267363

/-- Proves that the distance covered is 100 km given the conditions of the problem -/
theorem distance_covered (usual_speed usual_time increased_speed : ℝ) : 
  usual_speed = 20 →
  increased_speed = 25 →
  usual_speed * usual_time = increased_speed * (usual_time - 1) →
  usual_speed * usual_time = 100 := by
  sorry

#check distance_covered

end NUMINAMATH_CALUDE_distance_covered_l2673_267363


namespace NUMINAMATH_CALUDE_intersection_parallel_line_l2673_267343

/-- The equation of a line passing through the intersection of two given lines and parallel to a third line -/
theorem intersection_parallel_line (a b c d e f g h i : ℝ) :
  (∃ x y : ℝ, a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →  -- Intersection exists
  (g ≠ 0 ∨ h ≠ 0) →  -- Third line is not degenerate
  (a * h ≠ b * g ∨ d * h ≠ e * g) →  -- At least one of the first two lines is not parallel to the third
  (∃ k : ℝ, k ≠ 0 ∧ 
    ∀ x y : ℝ, (a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →
    ∃ t : ℝ, g * x + h * y + i + t * (a * x + b * y + c) = 0 ∧
            g * x + h * y + i + t * (d * x + e * y + f) = 0) →
  ∃ j : ℝ, ∀ x y : ℝ, 
    (a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →
    g * x + h * y + j = 0
  := by sorry

end NUMINAMATH_CALUDE_intersection_parallel_line_l2673_267343


namespace NUMINAMATH_CALUDE_baseball_card_money_ratio_l2673_267396

/-- Proves the ratio of Lisa's money to Charlotte's money given the conditions of the baseball card purchase problem -/
theorem baseball_card_money_ratio :
  let card_cost : ℕ := 100
  let patricia_money : ℕ := 6
  let lisa_money : ℕ := 5 * patricia_money
  let additional_money_needed : ℕ := 49
  let total_money : ℕ := card_cost - additional_money_needed
  let charlotte_money : ℕ := total_money - lisa_money - patricia_money
  (lisa_money : ℚ) / (charlotte_money : ℚ) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_baseball_card_money_ratio_l2673_267396


namespace NUMINAMATH_CALUDE_parabola_minimum_y_value_l2673_267319

/-- The minimum y-value of the parabola y = 3x^2 + 6x + 4 is 1 -/
theorem parabola_minimum_y_value :
  let f : ℝ → ℝ := fun x ↦ 3 * x^2 + 6 * x + 4
  ∃ x₀ : ℝ, ∀ x : ℝ, f x₀ ≤ f x ∧ f x₀ = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_minimum_y_value_l2673_267319


namespace NUMINAMATH_CALUDE_sqrt_sum_reciprocal_implies_fraction_l2673_267365

theorem sqrt_sum_reciprocal_implies_fraction (x : ℝ) (h : Real.sqrt x + 1 / Real.sqrt x = 3) :
  x / (x^2 + 2018*x + 1) = 1 / 2025 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_reciprocal_implies_fraction_l2673_267365


namespace NUMINAMATH_CALUDE_cultural_shirt_production_theorem_l2673_267311

/-- Represents the production and pricing of cultural shirts --/
structure CulturalShirtProduction where
  first_batch_cost : ℝ
  second_batch_cost : ℝ
  second_batch_quantity_multiplier : ℝ
  second_batch_cost_increase : ℝ
  discount_rate : ℝ
  discount_quantity : ℕ
  target_profit_margin : ℝ

/-- Calculates the cost per shirt in the first batch and the price per shirt for a given profit margin --/
def calculate_shirt_costs_and_price (prod : CulturalShirtProduction) :
  (ℝ × ℝ) :=
  sorry

/-- Theorem stating the correct cost and price for the given conditions --/
theorem cultural_shirt_production_theorem (prod : CulturalShirtProduction)
  (h1 : prod.first_batch_cost = 3000)
  (h2 : prod.second_batch_cost = 6600)
  (h3 : prod.second_batch_quantity_multiplier = 2)
  (h4 : prod.second_batch_cost_increase = 3)
  (h5 : prod.discount_rate = 0.6)
  (h6 : prod.discount_quantity = 30)
  (h7 : prod.target_profit_margin = 0.5) :
  calculate_shirt_costs_and_price prod = (30, 50) :=
  sorry

end NUMINAMATH_CALUDE_cultural_shirt_production_theorem_l2673_267311


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l2673_267339

/-- Given a triangle with two sides of length 51 and 67 units, and the third side being an integer,
    the minimum possible perimeter is 135 units. -/
theorem min_perimeter_triangle (a b x : ℕ) (ha : a = 51) (hb : b = 67) : 
  (a + b > x ∧ a + x > b ∧ b + x > a) → (∀ y : ℕ, (a + b > y ∧ a + y > b ∧ b + y > a) → x ≤ y) →
  a + b + x = 135 := by
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l2673_267339


namespace NUMINAMATH_CALUDE_sixth_television_is_three_l2673_267399

def selected_televisions : List Nat := [20, 26, 24, 19, 23, 3]

theorem sixth_television_is_three : 
  selected_televisions.length = 6 ∧ selected_televisions.getLast? = some 3 :=
sorry

end NUMINAMATH_CALUDE_sixth_television_is_three_l2673_267399


namespace NUMINAMATH_CALUDE_abc_value_l2673_267336

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (eq1 : a + 1/b = 5)
  (eq2 : b + 1/c = 2)
  (eq3 : c + 1/a = 3) :
  a * b * c = 1 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l2673_267336


namespace NUMINAMATH_CALUDE_inequality_solution_l2673_267354

theorem inequality_solution :
  ∃! x : ℝ, (Real.sqrt (x^3 + 2*x - 58) + 5) * |x^3 - 7*x^2 + 13*x - 3| ≤ 0 ∧ x = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2673_267354


namespace NUMINAMATH_CALUDE_average_of_next_sequence_l2673_267342

def consecutive_integers_average (a b : ℕ) : Prop :=
  (a > 0) ∧ 
  (b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4)) / 5)

theorem average_of_next_sequence (a b : ℕ) :
  consecutive_integers_average a b →
  ((b + (b + 1) + (b + 2) + (b + 3) + (b + 4)) / 5 : ℚ) = a + 4 := by
  sorry

end NUMINAMATH_CALUDE_average_of_next_sequence_l2673_267342


namespace NUMINAMATH_CALUDE_checkerboard_probability_l2673_267308

/-- Represents a rectangular checkerboard -/
structure Checkerboard where
  length : ℕ
  width : ℕ

/-- Calculates the total number of squares on the checkerboard -/
def totalSquares (board : Checkerboard) : ℕ :=
  board.length * board.width

/-- Calculates the number of squares not touching or adjacent to any edge -/
def innerSquares (board : Checkerboard) : ℕ :=
  (board.length - 4) * (board.width - 4)

/-- The probability of choosing a square not touching or adjacent to any edge -/
def innerSquareProbability (board : Checkerboard) : ℚ :=
  innerSquares board / totalSquares board

theorem checkerboard_probability :
  ∃ (board : Checkerboard), board.length = 10 ∧ board.width = 6 ∧
  innerSquareProbability board = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_probability_l2673_267308


namespace NUMINAMATH_CALUDE_root_implies_a_value_l2673_267324

theorem root_implies_a_value (a : ℝ) : (2 * (-1)^2 + a * (-1) - 1 = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_value_l2673_267324


namespace NUMINAMATH_CALUDE_stating_pipeline_equation_l2673_267314

/-- Represents the total length of the pipeline in meters -/
def total_length : ℝ := 3000

/-- Represents the increase in daily work efficiency as a decimal -/
def efficiency_increase : ℝ := 0.25

/-- Represents the number of days the project is completed ahead of schedule -/
def days_ahead : ℝ := 20

/-- 
Theorem stating that the equation correctly represents the relationship 
between the original daily pipeline laying rate and the given conditions
-/
theorem pipeline_equation (x : ℝ) : 
  total_length / ((1 + efficiency_increase) * x) - total_length / x = days_ahead := by
  sorry

end NUMINAMATH_CALUDE_stating_pipeline_equation_l2673_267314


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2673_267331

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 250 →
  train_speed_kmh = 60 →
  crossing_time = 20 →
  ∃ (bridge_length : ℝ), (abs (bridge_length - 83.4) < 0.1) ∧
    (bridge_length = train_speed_kmh * 1000 / 3600 * crossing_time - train_length) :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l2673_267331


namespace NUMINAMATH_CALUDE_f_20_5_l2673_267317

/-- 
  f(n, m) represents the number of possible increasing arithmetic sequences 
  that can be formed by selecting m terms from the numbers 1, 2, 3, ..., n
-/
def f (n m : ℕ) : ℕ :=
  sorry

/-- Helper function to check if a sequence is valid -/
def is_valid_sequence (seq : List ℕ) (n : ℕ) : Prop :=
  sorry

theorem f_20_5 : f 20 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_f_20_5_l2673_267317


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l2673_267348

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l2673_267348


namespace NUMINAMATH_CALUDE_power_three_124_mod_7_l2673_267300

theorem power_three_124_mod_7 : 3^124 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_three_124_mod_7_l2673_267300


namespace NUMINAMATH_CALUDE_expression_equality_l2673_267353

theorem expression_equality : 
  2013 * (2015/2014) + 2014 * (2016/2015) + 4029/(2014 * 2015) = 4029 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2673_267353


namespace NUMINAMATH_CALUDE_smallest_n_for_P_less_than_threshold_l2673_267303

/-- The probability of drawing n-1 white marbles followed by a red marble -/
def P (n : ℕ) : ℚ := 1 / (n * (n + 1))

/-- The number of boxes -/
def num_boxes : ℕ := 3000

theorem smallest_n_for_P_less_than_threshold :
  (∀ k < 55, P k ≥ 1 / num_boxes) ∧
  P 55 < 1 / num_boxes :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_P_less_than_threshold_l2673_267303


namespace NUMINAMATH_CALUDE_jessica_almonds_l2673_267382

theorem jessica_almonds : ∃ (j : ℕ), 
  (∃ (l : ℕ), j = l + 8 ∧ l = j / 3) → j = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_jessica_almonds_l2673_267382


namespace NUMINAMATH_CALUDE_parabola_properties_l2673_267387

/-- A parabola passing through (-1, 0) and (m, 0) opening downwards -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  h_a_neg : a < 0
  h_m_bounds : 1 < m ∧ m < 2
  h_pass_through : a * (-1)^2 + b * (-1) + c = 0 ∧ a * m^2 + b * m + c = 0

/-- The properties of the parabola -/
theorem parabola_properties (p : Parabola) :
  (p.b > 0) ∧ 
  (∀ x₁ x₂ y₁ y₂ : ℝ, 
    (p.a * x₁^2 + p.b * x₁ + p.c = y₁) → 
    (p.a * x₂^2 + p.b * x₂ + p.c = y₂) → 
    x₁ < x₂ → 
    x₁ + x₂ > 1 → 
    y₁ > y₂) ∧
  (p.a ≤ -1 → 
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    p.a * x₁^2 + p.b * x₁ + p.c = 1 ∧ 
    p.a * x₂^2 + p.b * x₂ + p.c = 1) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2673_267387


namespace NUMINAMATH_CALUDE_student_average_greater_than_true_average_l2673_267356

theorem student_average_greater_than_true_average 
  (x y w : ℤ) (h : x < w ∧ w < y) : 
  (x + w + 2 * y) / 4 > (x + y + w) / 3 := by
  sorry

end NUMINAMATH_CALUDE_student_average_greater_than_true_average_l2673_267356


namespace NUMINAMATH_CALUDE_problem_types_not_mutually_exclusive_l2673_267330

/-- Represents a mathematical problem type -/
inductive ProblemType
  | Proof
  | Computation
  | Construction

/-- Represents a mathematical problem -/
structure Problem where
  type : ProblemType
  hasProofElement : Bool
  hasComputationElement : Bool
  hasConstructionElement : Bool

/-- Theorem stating that problem types are not mutually exclusive -/
theorem problem_types_not_mutually_exclusive :
  ∃ (p : Problem), (p.type = ProblemType.Proof ∨ p.type = ProblemType.Computation ∨ p.type = ProblemType.Construction) ∧
    p.hasProofElement ∧ p.hasComputationElement ∧ p.hasConstructionElement :=
sorry

end NUMINAMATH_CALUDE_problem_types_not_mutually_exclusive_l2673_267330


namespace NUMINAMATH_CALUDE_geometry_relations_l2673_267333

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : subset m β) :
  (parallel_planes α β → perpendicular_lines l m) ∧
  ¬(perpendicular_lines l m → parallel_planes α β) ∧
  ¬(perpendicular_planes α β → parallel_lines l m) ∧
  (parallel_lines l m → perpendicular_planes α β) :=
sorry

end NUMINAMATH_CALUDE_geometry_relations_l2673_267333


namespace NUMINAMATH_CALUDE_find_b_l2673_267332

-- Define the ratio relationship
def ratio_relation (x y z : ℚ) : Prop :=
  ∃ (k : ℚ), x = 4 * k ∧ y = 3 * k ∧ z = 7 * k

-- Define the main theorem
theorem find_b (x y z b : ℚ) :
  ratio_relation x y z →
  y = 15 * b - 5 * z + 25 →
  z = 21 →
  b = 89 / 15 := by
  sorry


end NUMINAMATH_CALUDE_find_b_l2673_267332


namespace NUMINAMATH_CALUDE_range_of_M_l2673_267327

theorem range_of_M (x y z : ℝ) (h1 : x + y + z = 30) (h2 : 3 * x + y - z = 50)
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  let M := 5 * x + 4 * y + 2 * z
  ∀ m, (m = M) → 120 ≤ m ∧ m ≤ 130 :=
by sorry

end NUMINAMATH_CALUDE_range_of_M_l2673_267327


namespace NUMINAMATH_CALUDE_at_least_one_is_one_l2673_267316

theorem at_least_one_is_one (a b c : ℝ) 
  (sum_eq : a + b + c = 1/a + 1/b + 1/c) 
  (product_eq : a * b * c = 1) : 
  a = 1 ∨ b = 1 ∨ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_is_one_l2673_267316


namespace NUMINAMATH_CALUDE_pizza_topping_options_l2673_267392

/-- Represents the number of topping options for each category --/
structure ToppingOptions where
  cheese : Nat
  meat : Nat
  vegetable : Nat

/-- Represents the restriction between pepperoni and peppers --/
def hasPepperoniPepperRestriction : Bool := true

/-- Calculates the total number of topping combinations --/
def totalCombinations (options : ToppingOptions) (restriction : Bool) : Nat :=
  if restriction then
    options.cheese * (options.meat - 1) * options.vegetable +
    options.cheese * 1 * (options.vegetable - 1)
  else
    options.cheese * options.meat * options.vegetable

/-- The main theorem to prove --/
theorem pizza_topping_options :
  ∃ (options : ToppingOptions),
    options.cheese = 3 ∧
    options.vegetable = 5 ∧
    hasPepperoniPepperRestriction = true ∧
    totalCombinations options hasPepperoniPepperRestriction = 57 ∧
    options.meat = 4 := by
  sorry


end NUMINAMATH_CALUDE_pizza_topping_options_l2673_267392


namespace NUMINAMATH_CALUDE_difference_of_squares_l2673_267326

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2673_267326


namespace NUMINAMATH_CALUDE_missing_interior_angle_l2673_267367

theorem missing_interior_angle (n : ℕ) (sum_without_one : ℝ) (missing_angle : ℝ) :
  n = 18 →
  sum_without_one = 2750 →
  (n - 2) * 180 = sum_without_one + missing_angle →
  missing_angle = 130 :=
by sorry

end NUMINAMATH_CALUDE_missing_interior_angle_l2673_267367


namespace NUMINAMATH_CALUDE_prob_white_both_urns_l2673_267380

/-- Represents an urn with a certain number of black and white balls -/
structure Urn :=
  (black : ℕ)
  (white : ℕ)

/-- Calculates the probability of drawing a white ball from an urn -/
def prob_white (u : Urn) : ℚ :=
  u.white / (u.black + u.white)

/-- The probability of drawing white balls from both urns is 7/30 -/
theorem prob_white_both_urns (urn1 urn2 : Urn)
  (h1 : urn1 = Urn.mk 6 4)
  (h2 : urn2 = Urn.mk 5 7) :
  prob_white urn1 * prob_white urn2 = 7 / 30 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_both_urns_l2673_267380


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2673_267320

def M : Set ℕ := {1, 2, 3, 4}
def N : Set ℕ := {2, 4, 6, 8}

theorem intersection_of_M_and_N :
  M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2673_267320


namespace NUMINAMATH_CALUDE_power_of_two_preserves_order_l2673_267394

theorem power_of_two_preserves_order (a b : ℝ) : a > b → (2 : ℝ) ^ a > (2 : ℝ) ^ b := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_preserves_order_l2673_267394


namespace NUMINAMATH_CALUDE_x_value_l2673_267340

theorem x_value (w y z x : ℤ) 
  (hw : w = 90)
  (hz : z = 4 * w + 40)
  (hy : y = 3 * z + 15)
  (hx : x = 2 * y + 6) : 
  x = 2436 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2673_267340


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l2673_267377

/-- Calculates the profit percentage when selling a number of articles at the cost price of a different number of articles. -/
def profit_percentage (articles_sold : ℕ) (articles_cost_price : ℕ) : ℚ :=
  ((articles_cost_price : ℚ) - (articles_sold : ℚ)) / (articles_sold : ℚ) * 100

/-- Theorem stating that when a shopkeeper sells 50 articles at the cost price of 60 articles, the profit percentage is 20%. -/
theorem shopkeeper_profit : profit_percentage 50 60 = 20 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l2673_267377


namespace NUMINAMATH_CALUDE_gear_speed_ratio_l2673_267321

/-- Represents the number of teeth on a gear -/
structure Gear where
  teeth : ℕ

/-- Represents the angular speed of a gear in revolutions per minute -/
structure AngularSpeed where
  rpm : ℝ

/-- Represents a system of four meshed gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear

/-- The theorem stating the ratio of angular speeds for four meshed gears -/
theorem gear_speed_ratio (system : GearSystem) 
  (ωA ωB ωC ωD : AngularSpeed) :
  ωA.rpm * system.A.teeth = ωB.rpm * system.B.teeth ∧
  ωB.rpm * system.B.teeth = ωC.rpm * system.C.teeth ∧
  ωC.rpm * system.C.teeth = ωD.rpm * system.D.teeth →
  ∃ (k : ℝ), k > 0 ∧
    ωA.rpm = k * (system.B.teeth * system.C.teeth * system.D.teeth) ∧
    ωB.rpm = k * (system.A.teeth * system.C.teeth * system.D.teeth) ∧
    ωC.rpm = k * (system.A.teeth * system.B.teeth * system.D.teeth) ∧
    ωD.rpm = k * (system.A.teeth * system.B.teeth * system.C.teeth) :=
by sorry

end NUMINAMATH_CALUDE_gear_speed_ratio_l2673_267321


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2673_267373

/-- Two vectors in R² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (m, 2)
  parallel a b → m = -Real.sqrt 2 ∨ m = Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2673_267373


namespace NUMINAMATH_CALUDE_fraction_multiplication_cube_l2673_267397

theorem fraction_multiplication_cube : (1 / 2 : ℚ)^3 * (1 / 7 : ℚ) = 1 / 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_cube_l2673_267397


namespace NUMINAMATH_CALUDE_seeds_in_first_plot_l2673_267359

/-- The number of seeds planted in the first plot -/
def seeds_plot1 : ℕ := sorry

/-- The number of seeds planted in the second plot -/
def seeds_plot2 : ℕ := 200

/-- The percentage of seeds that germinated in the first plot -/
def germination_rate_plot1 : ℚ := 30 / 100

/-- The percentage of seeds that germinated in the second plot -/
def germination_rate_plot2 : ℚ := 35 / 100

/-- The percentage of total seeds that germinated -/
def total_germination_rate : ℚ := 32 / 100

/-- Theorem stating that the number of seeds planted in the first plot is 300 -/
theorem seeds_in_first_plot : 
  (germination_rate_plot1 * seeds_plot1 + germination_rate_plot2 * seeds_plot2 : ℚ) = 
  total_germination_rate * (seeds_plot1 + seeds_plot2) ∧ 
  seeds_plot1 = 300 := by sorry

end NUMINAMATH_CALUDE_seeds_in_first_plot_l2673_267359


namespace NUMINAMATH_CALUDE_franks_money_l2673_267388

/-- Frank's initial amount of money -/
def initial_money : ℝ := 11

/-- Amount Frank spent on a game -/
def game_cost : ℝ := 3

/-- Frank's allowance -/
def allowance : ℝ := 14

/-- Frank's final amount of money -/
def final_money : ℝ := 22

theorem franks_money :
  initial_money - game_cost + allowance = final_money :=
by sorry

end NUMINAMATH_CALUDE_franks_money_l2673_267388


namespace NUMINAMATH_CALUDE_floor_sqrt_50_l2673_267368

theorem floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_l2673_267368


namespace NUMINAMATH_CALUDE_inequality_direction_change_l2673_267395

theorem inequality_direction_change (a b x : ℝ) (h : x < 0) :
  (a < b) ↔ (a * x > b * x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_direction_change_l2673_267395


namespace NUMINAMATH_CALUDE_chocolate_ratio_problem_l2673_267362

/-- The number of dark chocolate bars sold given the ratio and white chocolate bars sold -/
def dark_chocolate_bars (white_ratio : ℕ) (dark_ratio : ℕ) (white_bars : ℕ) : ℕ :=
  (dark_ratio * white_bars) / white_ratio

/-- Theorem: Given a ratio of 4:3 for white to dark chocolate and 20 white chocolate bars sold,
    the number of dark chocolate bars sold is 15 -/
theorem chocolate_ratio_problem :
  dark_chocolate_bars 4 3 20 = 15 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_ratio_problem_l2673_267362


namespace NUMINAMATH_CALUDE_regression_line_equation_l2673_267360

/-- Given a regression line with slope 1.2 passing through the point (4, 5),
    prove that its equation is ŷ = 1.2x + 0.2 -/
theorem regression_line_equation 
  (slope : ℝ) 
  (center_x : ℝ) 
  (center_y : ℝ) 
  (h1 : slope = 1.2) 
  (h2 : center_x = 4) 
  (h3 : center_y = 5) : 
  ∃ (a : ℝ), ∀ (x y : ℝ), y = slope * x + a ↔ (x = center_x ∧ y = center_y) ∨ y = 1.2 * x + 0.2 :=
sorry

end NUMINAMATH_CALUDE_regression_line_equation_l2673_267360


namespace NUMINAMATH_CALUDE_candy_container_problem_l2673_267386

theorem candy_container_problem (V₁ V₂ n₁ : ℝ) (h₁ : V₁ = 72) (h₂ : V₂ = 216) (h₃ : n₁ = 30) :
  let n₂ := (n₁ / V₁) * V₂
  n₂ = 90 := by
sorry

end NUMINAMATH_CALUDE_candy_container_problem_l2673_267386


namespace NUMINAMATH_CALUDE_square_difference_l2673_267323

theorem square_difference (x y : ℚ) 
  (h1 : x + y = 3/8) 
  (h2 : x - y = 1/8) : 
  x^2 - y^2 = 3/64 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2673_267323


namespace NUMINAMATH_CALUDE_square_cut_perimeter_l2673_267310

/-- Given a square with perimeter 64 inches, prove that cutting a right triangle
    with hypotenuse equal to one side and translating it results in a new figure
    with perimeter 32 + 16√2 inches. -/
theorem square_cut_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 64) :
  let side_length : ℝ := square_perimeter / 4
  let triangle_leg : ℝ := side_length * Real.sqrt 2 / 2
  let new_perimeter : ℝ := 2 * side_length + 2 * triangle_leg
  new_perimeter = 32 + 16 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_square_cut_perimeter_l2673_267310


namespace NUMINAMATH_CALUDE_complex_product_theorem_l2673_267393

theorem complex_product_theorem (a : ℝ) (z₁ z₂ : ℂ) : 
  z₁ = a - 2*I ∧ z₂ = -1 + a*I ∧ (∃ b : ℝ, z₁ + z₂ = b*I) → z₁ * z₂ = 1 + 3*I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l2673_267393


namespace NUMINAMATH_CALUDE_number_ordering_l2673_267318

def a : ℕ := 62398
def b : ℕ := 63298
def c : ℕ := 62389
def d : ℕ := 63289

theorem number_ordering : b > d ∧ d > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_number_ordering_l2673_267318


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2673_267312

/-- A quadratic function f(x) = ax^2 - bx satisfying certain conditions -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - b * x

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties (a b : ℝ) :
  (f a b 2 = 0) →
  (∃ x : ℝ, (f a b x = x) ∧ (∀ y : ℝ, f a b y = y → y = x)) →
  ((∀ x : ℝ, f a b x = -1/2 * x^2 + x) ∧
   (∀ x : ℝ, x ∈ Set.Icc 0 3 → f a b x ≤ 1/2) ∧
   (∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ f a b x = 1/2)) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l2673_267312


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l2673_267351

theorem divisibility_by_seven (a b : ℤ) : (10 * a + b) % 7 = 0 ↔ (a - 2 * b) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l2673_267351


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l2673_267389

/-- Definition of an ellipse with given major axis length and eccentricity -/
structure Ellipse where
  major_axis : ℝ
  eccentricity : ℝ

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) : Prop :=
  (∀ x y : ℝ, x^2 / 16 + y^2 / 7 = 1) ∨ (∀ x y : ℝ, x^2 / 7 + y^2 / 16 = 1)

/-- Theorem stating that an ellipse with major axis 8 and eccentricity 3/4 satisfies the standard equation -/
theorem ellipse_standard_equation (e : Ellipse) (h1 : e.major_axis = 8) (h2 : e.eccentricity = 3/4) :
  standard_equation e := by
  sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l2673_267389


namespace NUMINAMATH_CALUDE_hyperbola_conditions_exclusive_or_conditions_l2673_267375

-- Define proposition p
def p (k : ℝ) : Prop := k^2 - 8*k - 20 ≤ 0

-- Define proposition q
def q (k : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 4 - k > 0 ∧ 1 - k < 0 ∧
  ∀ (x y : ℝ), x^2 / (4-k) + y^2 / (1-k) = 1 ↔ (x/a)^2 - (y/b)^2 = 1

theorem hyperbola_conditions (k : ℝ) : q k ↔ 1 < k ∧ k < 4 := by sorry

theorem exclusive_or_conditions (k : ℝ) : (p k ∨ q k) ∧ ¬(p k ∧ q k) ↔ 
  (-2 ≤ k ∧ k ≤ 1) ∨ (4 ≤ k ∧ k ≤ 10) := by sorry

end NUMINAMATH_CALUDE_hyperbola_conditions_exclusive_or_conditions_l2673_267375


namespace NUMINAMATH_CALUDE_servant_service_duration_l2673_267374

/-- Represents the servant's employment contract and actual service --/
structure ServantContract where
  yearlyPayment : ℕ  -- Payment in Rupees for a full year of service
  uniformPrice : ℕ   -- Price of the uniform in Rupees
  actualPayment : ℕ  -- Actual payment received in Rupees
  actualUniform : Bool -- Whether the servant received the uniform

/-- Calculates the number of months served based on the contract and actual payment --/
def monthsServed (contract : ServantContract) : ℚ :=
  let totalYearlyValue := contract.yearlyPayment + contract.uniformPrice
  let actualTotalReceived := contract.actualPayment + 
    (if contract.actualUniform then contract.uniformPrice else 0)
  let fractionWorked := (totalYearlyValue - actualTotalReceived) / contract.yearlyPayment
  12 * (1 - fractionWorked)

/-- Theorem stating that given the problem conditions, the servant served for approximately 3 months --/
theorem servant_service_duration (contract : ServantContract) 
  (h1 : contract.yearlyPayment = 900)
  (h2 : contract.uniformPrice = 100)
  (h3 : contract.actualPayment = 650)
  (h4 : contract.actualUniform = true) :
  ∃ (m : ℕ), m = 3 ∧ abs (monthsServed contract - m) < 1 := by
  sorry

end NUMINAMATH_CALUDE_servant_service_duration_l2673_267374


namespace NUMINAMATH_CALUDE_investment_income_is_500_l2673_267341

/-- Calculates the total yearly income from a set of investments -/
def totalYearlyIncome (totalAmount : ℝ) (firstInvestment : ℝ) (firstRate : ℝ) 
                      (secondInvestment : ℝ) (secondRate : ℝ) (remainderRate : ℝ) : ℝ :=
  let remainderInvestment := totalAmount - firstInvestment - secondInvestment
  firstInvestment * firstRate + secondInvestment * secondRate + remainderInvestment * remainderRate

/-- Theorem: The total yearly income from the given investment strategy is $500 -/
theorem investment_income_is_500 : 
  totalYearlyIncome 10000 4000 0.05 3500 0.04 0.064 = 500 := by
  sorry

end NUMINAMATH_CALUDE_investment_income_is_500_l2673_267341


namespace NUMINAMATH_CALUDE_cookie_area_theorem_l2673_267302

/-- Represents a rectangular cookie with length and width -/
structure Cookie where
  length : ℝ
  width : ℝ

/-- Calculates the area of a cookie -/
def Cookie.area (c : Cookie) : ℝ := c.length * c.width

/-- Calculates the circumference of two cookies placed horizontally -/
def combined_circumference (c : Cookie) : ℝ := 2 * (2 * c.length + c.width)

theorem cookie_area_theorem (c : Cookie) 
  (h1 : combined_circumference c = 70)
  (h2 : c.width = 15) : 
  c.area = 150 := by
  sorry

end NUMINAMATH_CALUDE_cookie_area_theorem_l2673_267302


namespace NUMINAMATH_CALUDE_election_probabilities_l2673_267304

structure Student where
  name : String
  prob_elected : ℚ

def A : Student := { name := "A", prob_elected := 4/5 }
def B : Student := { name := "B", prob_elected := 3/5 }
def C : Student := { name := "C", prob_elected := 7/10 }

def students : List Student := [A, B, C]

-- Probability that exactly one student is elected
def prob_exactly_one_elected (students : List Student) : ℚ :=
  sorry

-- Probability that at most two students are elected
def prob_at_most_two_elected (students : List Student) : ℚ :=
  sorry

theorem election_probabilities :
  (prob_exactly_one_elected students = 47/250) ∧
  (prob_at_most_two_elected students = 83/125) := by
  sorry

end NUMINAMATH_CALUDE_election_probabilities_l2673_267304


namespace NUMINAMATH_CALUDE_counterexample_odd_composite_plus_two_prime_l2673_267301

theorem counterexample_odd_composite_plus_two_prime :
  ∃ n : ℕ, 
    Odd n ∧ 
    ¬ Prime n ∧ 
    n > 1 ∧ 
    ¬ Prime (n + 2) ∧
    n = 25 :=
by
  sorry


end NUMINAMATH_CALUDE_counterexample_odd_composite_plus_two_prime_l2673_267301


namespace NUMINAMATH_CALUDE_odd_power_sum_divisible_l2673_267357

theorem odd_power_sum_divisible (x y : ℤ) :
  ∀ n : ℕ, Odd n → (x + y) ∣ (x^n + y^n) := by
  sorry

end NUMINAMATH_CALUDE_odd_power_sum_divisible_l2673_267357


namespace NUMINAMATH_CALUDE_chopped_cube_height_l2673_267322

theorem chopped_cube_height (cube_side_length : ℝ) (h_side : cube_side_length = 2) :
  let chopped_corner_height := cube_side_length - (1 / Real.sqrt 3)
  chopped_corner_height = (5 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_chopped_cube_height_l2673_267322


namespace NUMINAMATH_CALUDE_team_arrangement_solution_l2673_267358

/-- Represents the arrangement of team members in rows. -/
structure TeamArrangement where
  totalMembers : ℕ
  numRows : ℕ
  firstRowMembers : ℕ
  h1 : totalMembers = (numRows * (2 * firstRowMembers + numRows - 1)) / 2
  h2 : numRows > 16

/-- The solution to the team arrangement problem. -/
theorem team_arrangement_solution :
  ∃ (arr : TeamArrangement),
    arr.totalMembers = 1000 ∧
    arr.numRows = 25 ∧
    arr.firstRowMembers = 28 :=
  sorry


end NUMINAMATH_CALUDE_team_arrangement_solution_l2673_267358


namespace NUMINAMATH_CALUDE_knights_seating_probability_correct_l2673_267328

/-- The probability of three knights being seated with empty chairs on either side
    when randomly placed around a circular table with n chairs. -/
def knights_seating_probability (n : ℕ) : ℚ :=
  if n ≥ 6 then
    (n - 4 : ℚ) * (n - 5) / ((n - 1 : ℚ) * (n - 2))
  else
    0

/-- Theorem stating the probability of three knights being seated with empty chairs
    on either side when randomly placed around a circular table with n chairs. -/
theorem knights_seating_probability_correct (n : ℕ) (h : n ≥ 6) :
  knights_seating_probability n =
    (n - 4 : ℚ) * (n - 5) / ((n - 1 : ℚ) * (n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_knights_seating_probability_correct_l2673_267328


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l2673_267352

theorem complex_modulus_equality (x y : ℝ) :
  (Complex.I + 1) * x = Complex.I * y + 1 →
  Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l2673_267352


namespace NUMINAMATH_CALUDE_freezer_ice_cubes_l2673_267307

/-- The minimum number of ice cubes in Jerry's freezer -/
def min_ice_cubes (num_cups : ℕ) (ice_per_cup : ℕ) : ℕ :=
  num_cups * ice_per_cup

/-- Theorem stating that the minimum number of ice cubes is the product of cups and ice per cup -/
theorem freezer_ice_cubes (num_cups : ℕ) (ice_per_cup : ℕ) :
  min_ice_cubes num_cups ice_per_cup = num_cups * ice_per_cup :=
by sorry

end NUMINAMATH_CALUDE_freezer_ice_cubes_l2673_267307


namespace NUMINAMATH_CALUDE_jennys_bottle_cap_distance_l2673_267305

theorem jennys_bottle_cap_distance (x : ℝ) : 
  (x + (1/3) * x) + 21 = (15 + 2 * 15) → x = 18 := by sorry

end NUMINAMATH_CALUDE_jennys_bottle_cap_distance_l2673_267305


namespace NUMINAMATH_CALUDE_s_range_l2673_267338

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x)^3

theorem s_range :
  {y : ℝ | ∃ x ≠ 2, s x = y} = {y : ℝ | y < 0 ∨ y > 0} :=
by sorry

end NUMINAMATH_CALUDE_s_range_l2673_267338


namespace NUMINAMATH_CALUDE_sweet_bitter_fruits_problem_l2673_267309

/-- Represents the problem of buying sweet and bitter fruits --/
theorem sweet_bitter_fruits_problem 
  (x y : ℕ) -- x is the number of sweet fruits, y is the number of bitter fruits
  (h1 : x + y = 99) -- total number of fruits
  (h2 : 3 * x + (1/3) * y = 97) -- total cost in wen
  : 
  -- The system of equations correctly represents the problem
  (x + y = 99 ∧ 3 * x + (1/3) * y = 97) := by
  sorry


end NUMINAMATH_CALUDE_sweet_bitter_fruits_problem_l2673_267309


namespace NUMINAMATH_CALUDE_circle_bisector_properties_l2673_267379

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point A
def A : ℝ × ℝ := (6, 0)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define a point P on the circle
def P (x y : ℝ) : Prop := Circle x y

-- Define point M on the bisector of ∠POA and on PA
def M (x y : ℝ) (px py : ℝ) : Prop :=
  P px py ∧ 
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
  x = t * px + (1 - t) * A.1 ∧
  y = t * py + (1 - t) * A.2 ∧
  (x - O.1) * (A.1 - O.1) + (y - O.2) * (A.2 - O.2) = 
  (px - O.1) * (A.1 - O.1) + (py - O.2) * (A.2 - O.2)

-- Theorem statement
theorem circle_bisector_properties 
  (x y px py : ℝ) 
  (h_m : M x y px py) :
  (∃ (ma pm : ℝ), ma / pm = 3 ∧ 
    ma^2 = (x - A.1)^2 + (y - A.2)^2 ∧
    pm^2 = (x - px)^2 + (y - py)^2) ∧
  (x - 2/3)^2 + y^2 = 9/4 :=
sorry

end NUMINAMATH_CALUDE_circle_bisector_properties_l2673_267379


namespace NUMINAMATH_CALUDE_bianca_carrots_l2673_267385

/-- The number of carrots Bianca picked the next day -/
def carrots_picked_next_day (initial_carrots thrown_out_carrots final_total : ℕ) : ℕ :=
  final_total - (initial_carrots - thrown_out_carrots)

/-- Theorem stating that Bianca picked 47 carrots the next day -/
theorem bianca_carrots : carrots_picked_next_day 23 10 60 = 47 := by
  sorry

end NUMINAMATH_CALUDE_bianca_carrots_l2673_267385


namespace NUMINAMATH_CALUDE_power_multiplication_l2673_267313

theorem power_multiplication (a : ℝ) : a^3 * a = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2673_267313


namespace NUMINAMATH_CALUDE_parents_disagreeing_with_tuition_increase_l2673_267346

theorem parents_disagreeing_with_tuition_increase 
  (total_parents : ℕ) 
  (agree_percentage : ℚ) 
  (h1 : total_parents = 800) 
  (h2 : agree_percentage = 1/5) : 
  (1 - agree_percentage) * total_parents = 640 := by
  sorry

end NUMINAMATH_CALUDE_parents_disagreeing_with_tuition_increase_l2673_267346


namespace NUMINAMATH_CALUDE_fraction_under_eleven_l2673_267329

theorem fraction_under_eleven (total : ℕ) (between_eleven_and_thirteen : ℚ) (thirteen_and_above : ℕ) :
  total = 45 →
  between_eleven_and_thirteen = 2 / 5 →
  thirteen_and_above = 12 →
  (total : ℚ) - between_eleven_and_thirteen * total - (thirteen_and_above : ℚ) = 1 / 3 * total :=
by sorry

end NUMINAMATH_CALUDE_fraction_under_eleven_l2673_267329


namespace NUMINAMATH_CALUDE_runners_meet_time_l2673_267384

/-- The circumference of the circular track in meters -/
def track_length : ℝ := 600

/-- The speeds of the four runners in meters per second -/
def runner_speeds : List ℝ := [5.0, 5.5, 6.0, 6.5]

/-- The time in seconds for the runners to meet again -/
def meeting_time : ℝ := 1200

/-- Theorem stating that the given meeting time is the minimum time for the runners to meet again -/
theorem runners_meet_time : 
  meeting_time = (track_length / (runner_speeds[1] - runner_speeds[0])) ∧
  meeting_time = (track_length / (runner_speeds[2] - runner_speeds[1])) ∧
  meeting_time = (track_length / (runner_speeds[3] - runner_speeds[2])) ∧
  (∀ t : ℝ, t > 0 → t < meeting_time → 
    ∃ i j : Fin 4, i ≠ j ∧ 
    (runner_speeds[i] * t) % track_length ≠ (runner_speeds[j] * t) % track_length) :=
by sorry

end NUMINAMATH_CALUDE_runners_meet_time_l2673_267384


namespace NUMINAMATH_CALUDE_cube_root_simplification_l2673_267381

theorem cube_root_simplification (ω : ℂ) :
  ω ≠ 1 →
  ω^3 = 1 →
  (1 - 2*ω + 3*ω^2)^3 + (2 + 3*ω - 4*ω^2)^3 = -83 := by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l2673_267381


namespace NUMINAMATH_CALUDE_chef_nut_purchase_l2673_267349

/-- The weight of almonds bought by the chef in kilograms -/
def almond_weight : ℝ := 0.14

/-- The weight of pecans bought by the chef in kilograms -/
def pecan_weight : ℝ := 0.38

/-- The total weight of nuts bought by the chef in kilograms -/
def total_nut_weight : ℝ := almond_weight + pecan_weight

theorem chef_nut_purchase : total_nut_weight = 0.52 := by
  sorry

end NUMINAMATH_CALUDE_chef_nut_purchase_l2673_267349


namespace NUMINAMATH_CALUDE_existence_of_special_number_l2673_267325

theorem existence_of_special_number :
  ∃ N : ℕ, 
    (∃ a b : ℕ, a < 150 ∧ b < 150 ∧ b = a + 1 ∧ ¬(a ∣ N) ∧ ¬(b ∣ N)) ∧
    (∀ k : ℕ, k ≤ 150 → (k ∣ N) ∨ k = a ∨ k = b) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l2673_267325


namespace NUMINAMATH_CALUDE_mary_weight_loss_l2673_267383

/-- Given Mary's weight changes, prove her initial weight loss --/
theorem mary_weight_loss (initial_weight final_weight : ℝ) 
  (h1 : initial_weight = 99)
  (h2 : final_weight = 81) : 
  ∃ x : ℝ, x = 10.5 ∧ initial_weight - x + 2*x - 3*x + 3 = final_weight :=
by sorry

end NUMINAMATH_CALUDE_mary_weight_loss_l2673_267383


namespace NUMINAMATH_CALUDE_grid_number_is_333_l2673_267345

/-- Represents a shape type -/
inductive Shape : Type
| A
| B
| C

/-- Represents a row in the grid -/
structure Row :=
  (shape : Shape)
  (count : Nat)

/-- The problem setup -/
def grid_setup : List Row :=
  [⟨Shape.A, 3⟩, ⟨Shape.B, 3⟩, ⟨Shape.C, 3⟩]

/-- Converts a list of rows to a natural number -/
def rows_to_number (rows : List Row) : Nat :=
  rows.foldl (fun acc row => acc * 10 + row.count) 0

/-- The main theorem -/
theorem grid_number_is_333 :
  rows_to_number grid_setup = 333 := by
  sorry

end NUMINAMATH_CALUDE_grid_number_is_333_l2673_267345


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l2673_267344

/-- The vertex of the parabola y = -x^2 + 4x - 5 has coordinates (2, -1) -/
theorem parabola_vertex_coordinates :
  let f (x : ℝ) := -x^2 + 4*x - 5
  ∃! (a b : ℝ), (∀ x, f x ≤ f a) ∧ f a = b ∧ a = 2 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l2673_267344


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l2673_267315

theorem sqrt_sum_fractions : 
  Real.sqrt (2 * ((1 : ℝ) / 25 + (1 : ℝ) / 36)) = (Real.sqrt 122) / 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l2673_267315


namespace NUMINAMATH_CALUDE_arithmetic_sequence_specific_sum_l2673_267335

/-- An arithmetic sequence with sum S_n of its first n terms. -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n : ℕ, S n = n * (a 0 + a (n - 1)) / 2

/-- Given an arithmetic sequence with specific sum values, prove n = 4. -/
theorem arithmetic_sequence_specific_sum (seq : ArithmeticSequence) 
  (h1 : seq.S 6 = 36)
  (h2 : seq.S 12 = 144)
  (h3 : ∃ n : ℕ, seq.S (6 * n) = 576) :
  ∃ n : ℕ, n = 4 ∧ seq.S (6 * n) = 576 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_specific_sum_l2673_267335


namespace NUMINAMATH_CALUDE_c_value_theorem_l2673_267334

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation for c
def c_equation (a b : ℕ+) : ℂ := (a + b * i) ^ 3 - 107 * i

-- State the theorem
theorem c_value_theorem (a b c : ℕ+) :
  (c_equation a b).re = c ∧ (c_equation a b).im = 0 → c = 198 := by
  sorry

end NUMINAMATH_CALUDE_c_value_theorem_l2673_267334


namespace NUMINAMATH_CALUDE_sum_even_minus_odd_product_equals_6401_l2673_267370

def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

def product_of_odd_integers (a b : ℕ) : ℕ :=
  if ∃ n ∈ Finset.range (b - a + 1), Even (a + n) then 0 else 1

theorem sum_even_minus_odd_product_equals_6401 :
  sum_of_integers 100 150 + count_even_integers 100 150 - product_of_odd_integers 100 150 = 6401 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_minus_odd_product_equals_6401_l2673_267370


namespace NUMINAMATH_CALUDE_certain_number_proof_l2673_267376

theorem certain_number_proof (X : ℝ) : 0.8 * X - 0.35 * 300 = 31 → X = 170 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2673_267376


namespace NUMINAMATH_CALUDE_sum_first_six_primes_l2673_267364

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The sum of the first n prime numbers -/
def sumFirstNPrimes (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of the first 6 prime numbers is 41 -/
theorem sum_first_six_primes : sumFirstNPrimes 6 = 41 := by sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_l2673_267364


namespace NUMINAMATH_CALUDE_quadratic_equation_one_l2673_267337

theorem quadratic_equation_one (x : ℝ) : 2 * (2 * x - 1)^2 = 8 ↔ x = 3/2 ∨ x = -1/2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_one_l2673_267337


namespace NUMINAMATH_CALUDE_range_of_g_l2673_267372

open Set
open Function

def g (x : ℝ) : ℝ := 3 * (x - 4)

theorem range_of_g :
  range g = {y : ℝ | y < -27 ∨ y > -27} :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l2673_267372


namespace NUMINAMATH_CALUDE_b_alone_time_l2673_267355

-- Define the work rates
def work_rate_b : ℚ := 1
def work_rate_a : ℚ := 2 * work_rate_b
def work_rate_c : ℚ := 3 * work_rate_a

-- Define the total work (completed job)
def total_work : ℚ := 1

-- Define the time taken by all three together
def total_time : ℚ := 9

-- Theorem to prove
theorem b_alone_time (h1 : work_rate_a = 2 * work_rate_b)
                     (h2 : work_rate_c = 3 * work_rate_a)
                     (h3 : (work_rate_a + work_rate_b + work_rate_c) * total_time = total_work) :
  total_work / work_rate_b = 81 := by
  sorry


end NUMINAMATH_CALUDE_b_alone_time_l2673_267355


namespace NUMINAMATH_CALUDE_l_shape_area_l2673_267350

/-- The area of an "L" shape formed by removing a smaller rectangle from a larger rectangle -/
theorem l_shape_area (large_length large_width small_length_diff small_width_diff : ℕ) : 
  large_length = 10 →
  large_width = 7 →
  small_length_diff = 3 →
  small_width_diff = 3 →
  (large_length * large_width) - ((large_length - small_length_diff) * (large_width - small_width_diff)) = 42 := by
sorry

end NUMINAMATH_CALUDE_l_shape_area_l2673_267350


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2673_267306

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first four terms of the sequence equals 30. -/
def SumEquals30 (a : ℕ → ℝ) : Prop :=
  a 1 + a 2 + a 3 + a 4 = 30

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumEquals30 a) : a 2 + a 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2673_267306


namespace NUMINAMATH_CALUDE_movie_theater_attendance_l2673_267366

theorem movie_theater_attendance 
  (total_seats : ℕ) 
  (empty_seats : ℕ) 
  (h1 : total_seats = 750) 
  (h2 : empty_seats = 218) : 
  total_seats - empty_seats = 532 := by
sorry

end NUMINAMATH_CALUDE_movie_theater_attendance_l2673_267366
