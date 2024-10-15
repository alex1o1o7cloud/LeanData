import Mathlib

namespace NUMINAMATH_CALUDE_other_polynomial_form_l4070_407031

/-- Given two polynomials with a specified difference, this theorem proves the form of the other polynomial. -/
theorem other_polynomial_form (a b c d : ℝ) 
  (diff : ℝ) -- The difference between the two polynomials
  (poly1 : ℝ) -- One of the polynomials
  (h1 : diff = c^2 * d^2 - a^2 * b^2) -- Condition on the difference
  (h2 : poly1 = a^2 * b^2 + c^2 * d^2 - 2*a*b*c*d) -- Condition on one polynomial
  : ∃ (poly2 : ℝ), (poly2 = 2*c^2*d^2 - 2*a*b*c*d ∨ poly2 = 2*a^2*b^2 - 2*a*b*c*d) ∧ 
    ((poly1 - poly2 = diff) ∨ (poly2 - poly1 = diff)) :=
by
  sorry

end NUMINAMATH_CALUDE_other_polynomial_form_l4070_407031


namespace NUMINAMATH_CALUDE_pyramidal_stack_logs_example_l4070_407036

/-- Calculates the total number of logs in a pyramidal stack. -/
def pyramidal_stack_logs (bottom_row : ℕ) (top_row : ℕ) (difference : ℕ) : ℕ :=
  let n := (bottom_row - top_row) / difference + 1
  n * (bottom_row + top_row) / 2

/-- Proves that the total number of logs in the given pyramidal stack is 60. -/
theorem pyramidal_stack_logs_example : pyramidal_stack_logs 15 5 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_pyramidal_stack_logs_example_l4070_407036


namespace NUMINAMATH_CALUDE_max_value_under_constraint_l4070_407040

theorem max_value_under_constraint (x y : ℝ) :
  x^2 + y^2 ≤ 5 →
  3*|x + y| + |4*y + 9| + |7*y - 3*x - 18| ≤ 27 + 6*Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_under_constraint_l4070_407040


namespace NUMINAMATH_CALUDE_points_form_circle_l4070_407032

theorem points_form_circle :
  ∀ (t : ℝ), ∃ (x y : ℝ), x = Real.cos t ∧ y = Real.sin t → x^2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_points_form_circle_l4070_407032


namespace NUMINAMATH_CALUDE_prime_arithmetic_sequence_ones_digit_l4070_407049

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def ones_digit (n : ℕ) : ℕ := n % 10

def arithmetic_sequence (a b c d : ℕ) : Prop :=
  ∃ (r : ℕ), r > 0 ∧ b = a + r ∧ c = b + r ∧ d = c + r

theorem prime_arithmetic_sequence_ones_digit
  (p q r s : ℕ)
  (h_prime : is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s)
  (h_seq : arithmetic_sequence p q r s)
  (h_p_gt_5 : p > 5)
  (h_diff : ∃ (d : ℕ), d = 10 ∧ q = p + d ∧ r = q + d ∧ s = r + d) :
  ones_digit p = 1 ∨ ones_digit p = 3 ∨ ones_digit p = 7 ∨ ones_digit p = 9 :=
sorry

end NUMINAMATH_CALUDE_prime_arithmetic_sequence_ones_digit_l4070_407049


namespace NUMINAMATH_CALUDE_ellipse_line_slope_l4070_407063

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (2, 0)

-- Define a line passing through a point with a given slope
def line_through_point (m : ℝ) (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  y - p.2 = m * (x - p.1)

-- Define a circle passing through three points
def circle_through_points (p1 p2 p3 : ℝ × ℝ) (center : ℝ × ℝ) : Prop :=
  (p1.1 - center.1)^2 + (p1.2 - center.2)^2 =
  (p2.1 - center.1)^2 + (p2.2 - center.2)^2 ∧
  (p1.1 - center.1)^2 + (p1.2 - center.2)^2 =
  (p3.1 - center.1)^2 + (p3.2 - center.2)^2

-- Theorem statement
theorem ellipse_line_slope :
  ∀ (A B : ℝ × ℝ) (m : ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    line_through_point m right_focus A.1 A.2 ∧
    line_through_point m right_focus B.1 B.2 ∧
    (∃ (t : ℝ), circle_through_points A B (-Real.sqrt 7, 0) (0, t)) →
    m = Real.sqrt 2 / 2 ∨ m = -Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_line_slope_l4070_407063


namespace NUMINAMATH_CALUDE_car_overtake_distance_l4070_407083

/-- Proves that the initial distance between two cars is 10 miles given their speeds and overtaking time -/
theorem car_overtake_distance (speed_a speed_b time_to_overtake : ℝ) 
  (h1 : speed_a = 58)
  (h2 : speed_b = 50)
  (h3 : time_to_overtake = 2.25)
  (h4 : (speed_a - speed_b) * time_to_overtake = initial_distance + 8) :
  initial_distance = 10 := by sorry


end NUMINAMATH_CALUDE_car_overtake_distance_l4070_407083


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l4070_407064

def geometric_sequence (a : ℕ → ℝ) :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  geometric_sequence a → a 1 = 1 → a 5 = 9 → a 3 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l4070_407064


namespace NUMINAMATH_CALUDE_janabel_widget_sales_l4070_407071

theorem janabel_widget_sales :
  let a : ℕ → ℕ := fun n => 2 * n - 1
  let S : ℕ → ℕ := fun n => n * (a 1 + a n) / 2
  S 20 = 400 := by
  sorry

end NUMINAMATH_CALUDE_janabel_widget_sales_l4070_407071


namespace NUMINAMATH_CALUDE_negation_of_all_cuboids_are_prisms_l4070_407094

-- Define the universe of shapes
variable {Shape : Type}

-- Define properties
variable (isCuboid : Shape → Prop)
variable (isPrism : Shape → Prop)
variable (hasLateralFaces : Shape → ℕ → Prop)

-- The theorem
theorem negation_of_all_cuboids_are_prisms :
  (¬ ∀ x : Shape, isCuboid x → (isPrism x ∧ hasLateralFaces x 4)) ↔ 
  (∃ x : Shape, isCuboid x ∧ ¬(isPrism x ∧ hasLateralFaces x 4)) := by
sorry

end NUMINAMATH_CALUDE_negation_of_all_cuboids_are_prisms_l4070_407094


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l4070_407045

theorem arithmetic_mean_of_fractions :
  (3 : ℚ) / 7 + (5 : ℚ) / 9 = (31 : ℚ) / 63 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l4070_407045


namespace NUMINAMATH_CALUDE_article_price_l4070_407077

theorem article_price (selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : selling_price = 110)
  (h2 : gain_percent = 10) : 
  ∃ (original_price : ℝ), 
    selling_price = original_price * (1 + gain_percent / 100) ∧ 
    original_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_article_price_l4070_407077


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l4070_407013

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l4070_407013


namespace NUMINAMATH_CALUDE_fraction_existence_and_nonexistence_l4070_407019

theorem fraction_existence_and_nonexistence :
  (∀ n : ℕ+, ∃ a b : ℕ+, (Real.sqrt n : ℝ) ≤ (a : ℝ) / (b : ℝ) ∧
                         (a : ℝ) / (b : ℝ) ≤ Real.sqrt (n + 1) ∧
                         (b : ℝ) ≤ Real.sqrt n + 1) ∧
  (∃ f : ℕ → ℕ+, ∀ k : ℕ, ∀ a b : ℕ+,
    (Real.sqrt (f k) : ℝ) ≤ (a : ℝ) / (b : ℝ) →
    (a : ℝ) / (b : ℝ) ≤ Real.sqrt (f k + 1) →
    (b : ℝ) > Real.sqrt (f k)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_existence_and_nonexistence_l4070_407019


namespace NUMINAMATH_CALUDE_correct_statements_l4070_407051

-- Define a differentiable function
variable (f : ℝ → ℝ) (hf : Differentiable ℝ f)

-- Define extremum
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ f x ≥ f x₀

-- Define inductive and deductive reasoning
def InductiveReasoning : Prop :=
  ∃ (specific general : Prop), specific → general

def DeductiveReasoning : Prop :=
  ∃ (general specific : Prop), general → specific

-- Define synthetic and analytic methods
def SyntheticMethod : Prop :=
  ∃ (cause effect : Prop), cause → effect

def AnalyticMethod : Prop :=
  ∃ (effect cause : Prop), effect → cause

-- Theorem statement
theorem correct_statements
  (x₀ : ℝ)
  (h_extremum : HasExtremumAt f x₀) :
  (deriv f x₀ = 0) ∧
  InductiveReasoning ∧
  DeductiveReasoning ∧
  SyntheticMethod ∧
  AnalyticMethod :=
sorry

end NUMINAMATH_CALUDE_correct_statements_l4070_407051


namespace NUMINAMATH_CALUDE_inverse_graph_point_l4070_407014

-- Define a function f with an inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- Define the condition that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- Define the condition that the graph of y = x - f(x) passes through (2,5)
axiom graph_condition : 2 - f 2 = 5

-- Theorem to prove
theorem inverse_graph_point :
  (∀ x, f_inv (f x) = x ∧ f (f_inv x) = x) →
  (2 - f 2 = 5) →
  f_inv (-3) + 3 = 5 :=
by sorry

end NUMINAMATH_CALUDE_inverse_graph_point_l4070_407014


namespace NUMINAMATH_CALUDE_matrix_equation_holds_l4070_407065

def M : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 2, 4]

theorem matrix_equation_holds :
  M^3 - 2 • M^2 + (-12) • M = 3 • !![1, 2; 2, 4] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_holds_l4070_407065


namespace NUMINAMATH_CALUDE_softball_players_count_l4070_407044

theorem softball_players_count (cricket hockey football total : ℕ) 
  (h1 : cricket = 22)
  (h2 : hockey = 15)
  (h3 : football = 21)
  (h4 : total = 77) :
  total - (cricket + hockey + football) = 19 := by
  sorry

end NUMINAMATH_CALUDE_softball_players_count_l4070_407044


namespace NUMINAMATH_CALUDE_karthik_weight_upper_bound_l4070_407033

-- Define the lower and upper bounds for Karthik's weight according to different opinions
def karthik_lower_bound : ℝ := 55
def brother_lower_bound : ℝ := 50
def brother_upper_bound : ℝ := 60
def father_upper_bound : ℝ := 58

-- Define the average weight
def average_weight : ℝ := 56.5

-- Define Karthik's upper bound as a variable
def karthik_upper_bound : ℝ := sorry

-- Theorem statement
theorem karthik_weight_upper_bound :
  karthik_lower_bound < karthik_upper_bound ∧
  brother_lower_bound < karthik_upper_bound ∧
  karthik_upper_bound ≤ brother_upper_bound ∧
  karthik_upper_bound ≤ father_upper_bound ∧
  average_weight = (karthik_lower_bound + karthik_upper_bound) / 2 →
  karthik_upper_bound = 58 := by sorry

end NUMINAMATH_CALUDE_karthik_weight_upper_bound_l4070_407033


namespace NUMINAMATH_CALUDE_saree_price_calculation_l4070_407035

/-- Calculate the final price after applying multiple discounts and a tax increase --/
def finalPrice (initialPrice : ℝ) (discounts : List ℝ) (taxRate : ℝ) (finalDiscount : ℝ) : ℝ :=
  let priceAfterDiscounts := discounts.foldl (fun price discount => price * (1 - discount)) initialPrice
  let priceAfterTax := priceAfterDiscounts * (1 + taxRate)
  priceAfterTax * (1 - finalDiscount)

/-- The theorem stating the final price of the sarees --/
theorem saree_price_calculation :
  let initialPrice : ℝ := 495
  let discounts : List ℝ := [0.20, 0.15, 0.10]
  let taxRate : ℝ := 0.05
  let finalDiscount : ℝ := 0.03
  abs (finalPrice initialPrice discounts taxRate finalDiscount - 308.54) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_saree_price_calculation_l4070_407035


namespace NUMINAMATH_CALUDE_binary_representation_of_51_l4070_407050

/-- Converts a natural number to its binary representation as a list of booleans -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- The decimal number to be converted -/
def decimalNumber : ℕ := 51

/-- The expected binary representation -/
def expectedBinary : List Bool := [true, true, false, false, true, true]

/-- Theorem stating that the binary representation of 51 is [true, true, false, false, true, true] -/
theorem binary_representation_of_51 :
  toBinary decimalNumber = expectedBinary := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_of_51_l4070_407050


namespace NUMINAMATH_CALUDE_regular_polygon_inscribed_circle_l4070_407021

theorem regular_polygon_inscribed_circle (n : ℕ) (R : ℝ) (h : R > 0) :
  (1 / 2 : ℝ) * n * R^2 * Real.sin (2 * Real.pi / n) = 4 * R^2 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_inscribed_circle_l4070_407021


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l4070_407097

/-- Proves that 11,580,000 is equal to 1.158 × 10^7 in scientific notation -/
theorem scientific_notation_equivalence : 
  11580000 = 1.158 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l4070_407097


namespace NUMINAMATH_CALUDE_point_C_coordinates_l4070_407092

def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (7, 2)

theorem point_C_coordinates :
  ∀ C : ℝ × ℝ,
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B) →
  (dist A C = 2 * dist C B) →
  C = (5, 2/3) :=
by sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l4070_407092


namespace NUMINAMATH_CALUDE_ab_nonnegative_l4070_407084

theorem ab_nonnegative (a b : ℚ) (ha : |a| = -a) (hb : |b| ≠ b) : a * b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_nonnegative_l4070_407084


namespace NUMINAMATH_CALUDE_cone_radius_l4070_407041

theorem cone_radius (r l : ℝ) : 
  r > 0 → l > 0 →
  π * l = 2 * π * r →
  π * r^2 + π * r * l = 3 * π →
  r = 1 := by
sorry

end NUMINAMATH_CALUDE_cone_radius_l4070_407041


namespace NUMINAMATH_CALUDE_burger_share_length_l4070_407072

-- Define the length of a foot in inches
def foot_in_inches : ℕ := 12

-- Define the burger length in feet
def burger_length_feet : ℕ := 1

-- Define the number of people sharing the burger
def num_people : ℕ := 2

-- Theorem to prove
theorem burger_share_length :
  (burger_length_feet * foot_in_inches) / num_people = 6 := by
  sorry

end NUMINAMATH_CALUDE_burger_share_length_l4070_407072


namespace NUMINAMATH_CALUDE_gas_cost_per_gallon_l4070_407093

/-- Proves that the cost of gas per gallon is $4, given the specified conditions. -/
theorem gas_cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) :
  miles_per_gallon = 32 →
  total_miles = 304 →
  total_cost = 38 →
  total_cost / (total_miles / miles_per_gallon) = 4 := by
  sorry

#check gas_cost_per_gallon

end NUMINAMATH_CALUDE_gas_cost_per_gallon_l4070_407093


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l4070_407009

/-- Represents a rectangular plot with a given breadth and area -/
structure RectangularPlot where
  breadth : ℝ
  area : ℝ
  length_is_thrice_breadth : length = 3 * breadth
  area_formula : area = length * breadth

/-- The breadth of a rectangular plot with thrice length and 2700 sq m area is 30 m -/
theorem rectangular_plot_breadth (plot : RectangularPlot) 
  (h_area : plot.area = 2700) : plot.breadth = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l4070_407009


namespace NUMINAMATH_CALUDE_remaining_time_is_three_l4070_407018

/-- Represents the time needed to finish plowing a field with two tractors -/
def time_to_finish (time_a time_b worked_time : ℚ) : ℚ :=
  let rate_a : ℚ := 1 / time_a
  let rate_b : ℚ := 1 / time_b
  let remaining_work : ℚ := 1 - (rate_a * worked_time)
  let combined_rate : ℚ := rate_a + rate_b
  remaining_work / combined_rate

/-- Theorem stating that the remaining time to finish plowing is 3 hours -/
theorem remaining_time_is_three :
  time_to_finish 20 15 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remaining_time_is_three_l4070_407018


namespace NUMINAMATH_CALUDE_existence_and_digit_sum_l4070_407090

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Proves the existence of N and its properties -/
theorem existence_and_digit_sum :
  ∃ N : ℕ, N^2 = 36^50 * 50^36 ∧ sum_of_digits N = 54 := by sorry

end NUMINAMATH_CALUDE_existence_and_digit_sum_l4070_407090


namespace NUMINAMATH_CALUDE_cone_height_ratio_l4070_407089

theorem cone_height_ratio (base_circumference : Real) (original_height : Real) (shorter_volume : Real) :
  base_circumference = 20 * Real.pi →
  original_height = 40 →
  shorter_volume = 160 * Real.pi →
  ∃ (shorter_height : Real),
    (1 / 3) * Real.pi * ((base_circumference / (2 * Real.pi)) ^ 2) * shorter_height = shorter_volume ∧
    shorter_height / original_height = 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l4070_407089


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l4070_407026

theorem smallest_number_divisible (n : ℕ) : n = 746 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ k₃ k₄ : ℕ, 
    m - 18 = 8 * k₁ ∧ 
    m - 18 = 14 * k₂ ∧ 
    m - 18 = 26 * k₃ ∧ 
    m - 18 = 28 * k₄)) ∧
  (∃ k₁ k₂ k₃ k₄ : ℕ, 
    n - 18 = 8 * k₁ ∧ 
    n - 18 = 14 * k₂ ∧ 
    n - 18 = 26 * k₃ ∧ 
    n - 18 = 28 * k₄) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l4070_407026


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l4070_407020

theorem perfect_square_trinomial (x y k : ℝ) : 
  (∃ a : ℝ, x^2 + k*x*y + 64*y^2 = a^2) → k = 16 ∨ k = -16 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l4070_407020


namespace NUMINAMATH_CALUDE_sum_from_interest_and_discount_l4070_407010

/-- Given a sum, rate, and time, if the simple interest is 88 and the true discount is 80, then the sum is 880. -/
theorem sum_from_interest_and_discount (P r t : ℝ) 
  (h1 : P * r * t / 100 = 88)
  (h2 : P * r * t / (100 + r * t) = 80) : 
  P = 880 := by
  sorry

#check sum_from_interest_and_discount

end NUMINAMATH_CALUDE_sum_from_interest_and_discount_l4070_407010


namespace NUMINAMATH_CALUDE_negative_pi_less_than_negative_three_l4070_407053

theorem negative_pi_less_than_negative_three :
  π > 3 → -π < -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_pi_less_than_negative_three_l4070_407053


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4070_407027

/-- Given a hyperbola C: x²/a² - y²/b² = 1 (a > 0, b > 0) with focal length 2√5,
    and a parabola y = (1/4)x² + 1/4 tangent to its asymptote,
    prove that the equation of the hyperbola C is x²/4 - y² = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_focal : 5 = a^2 + b^2)
  (h_tangent : ∃ (x : ℝ), (1/4) * x^2 + (1/4) = (b/a) * x) :
  a = 2 ∧ b = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4070_407027


namespace NUMINAMATH_CALUDE_park_trees_l4070_407095

theorem park_trees (pine_percentage : ℝ) (non_pine_count : ℕ) 
  (h1 : pine_percentage = 0.7)
  (h2 : non_pine_count = 105) : 
  ∃ (total_trees : ℕ), 
    (↑non_pine_count : ℝ) = (1 - pine_percentage) * (total_trees : ℝ) ∧ 
    total_trees = 350 :=
by sorry

end NUMINAMATH_CALUDE_park_trees_l4070_407095


namespace NUMINAMATH_CALUDE_permutation_distinct_differences_l4070_407037

theorem permutation_distinct_differences (n : ℕ+) :
  (∃ (a : Fin n → Fin n), Function.Bijective a ∧
    (∀ (i j : Fin n), i ≠ j → |a i - i| ≠ |a j - j|)) ↔
  (∃ (k : ℕ), n = 4 * k ∨ n = 4 * k + 1) :=
by sorry

end NUMINAMATH_CALUDE_permutation_distinct_differences_l4070_407037


namespace NUMINAMATH_CALUDE_percy_dish_cost_l4070_407067

/-- The cost of a meal for three people with a 10% tip --/
def meal_cost (leticia_cost scarlett_cost percy_cost : ℝ) : ℝ :=
  (leticia_cost + scarlett_cost + percy_cost) * 1.1

/-- The theorem stating the cost of Percy's dish --/
theorem percy_dish_cost : 
  ∃ (percy_cost : ℝ), 
    meal_cost 10 13 percy_cost = 44 ∧ 
    percy_cost = 17 := by
  sorry

end NUMINAMATH_CALUDE_percy_dish_cost_l4070_407067


namespace NUMINAMATH_CALUDE_initial_money_calculation_l4070_407062

theorem initial_money_calculation (initial_money : ℚ) : 
  (initial_money * (1 - 1/3) * (1 - 1/5) * (1 - 1/4) = 100) → 
  initial_money = 250 := by
sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l4070_407062


namespace NUMINAMATH_CALUDE_negation_of_exists_exponential_nonpositive_l4070_407007

theorem negation_of_exists_exponential_nonpositive :
  (¬ ∃ x : ℝ, Real.exp x ≤ 0) ↔ (∀ x : ℝ, Real.exp x > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_exists_exponential_nonpositive_l4070_407007


namespace NUMINAMATH_CALUDE_line_direction_vector_l4070_407003

/-- The direction vector of a parameterized line -/
def direction_vector (line : ℝ → ℝ × ℝ) : ℝ × ℝ := sorry

/-- The point on the line at t = 0 -/
def initial_point (line : ℝ → ℝ × ℝ) : ℝ × ℝ := sorry

theorem line_direction_vector :
  let line (t : ℝ) : ℝ × ℝ := 
    (4 + 3 * t / Real.sqrt 34, 2 + 5 * t / Real.sqrt 34)
  let y (x : ℝ) : ℝ := (5 * x - 7) / 3
  ∀ (x : ℝ), x ≥ 4 → 
    let point := (x, y x)
    let dist := Real.sqrt ((x - 4)^2 + (y x - 2)^2)
    point = initial_point line + dist • direction_vector line ∧
    direction_vector line = (3 / Real.sqrt 34, 5 / Real.sqrt 34) :=
by sorry

end NUMINAMATH_CALUDE_line_direction_vector_l4070_407003


namespace NUMINAMATH_CALUDE_gcd_n_minus_three_n_plus_three_eq_one_l4070_407085

theorem gcd_n_minus_three_n_plus_three_eq_one (n : ℕ+) 
  (h : (Nat.divisors (n.val^2 - 9)).card = 6) : 
  Nat.gcd (n.val - 3) (n.val + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_minus_three_n_plus_three_eq_one_l4070_407085


namespace NUMINAMATH_CALUDE_germination_probability_l4070_407076

/-- The germination rate of seeds -/
def germination_rate : ℝ := 0.8

/-- The number of seeds sown -/
def num_seeds : ℕ := 5

/-- The probability of exactly k successes in n trials with probability p -/
def bernoulli_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of at least 4 out of 5 seeds germinating -/
def prob_at_least_4 : ℝ :=
  bernoulli_prob num_seeds 4 germination_rate + bernoulli_prob num_seeds 5 germination_rate

theorem germination_probability :
  prob_at_least_4 = 0.73728 := by sorry

end NUMINAMATH_CALUDE_germination_probability_l4070_407076


namespace NUMINAMATH_CALUDE_alex_candles_left_l4070_407060

theorem alex_candles_left (initial_candles used_candles : ℕ) 
  (h1 : initial_candles = 44)
  (h2 : used_candles = 32) :
  initial_candles - used_candles = 12 := by
  sorry

end NUMINAMATH_CALUDE_alex_candles_left_l4070_407060


namespace NUMINAMATH_CALUDE_fraction_less_than_mode_l4070_407029

def data_list : List ℕ := [3, 4, 5, 5, 5, 5, 7, 11, 21]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

def count_less_than_mode (l : List ℕ) : ℕ :=
  l.filter (· < mode l) |>.length

theorem fraction_less_than_mode :
  (count_less_than_mode data_list : ℚ) / data_list.length = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_mode_l4070_407029


namespace NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_1540_l4070_407091

theorem sum_largest_smallest_prime_factors_1540 : 
  ∃ (p q : ℕ), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p ∣ 1540 ∧ 
    q ∣ 1540 ∧ 
    (∀ r : ℕ, Nat.Prime r → r ∣ 1540 → p ≤ r ∧ r ≤ q) ∧ 
    p + q = 13 :=
by sorry

end NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_1540_l4070_407091


namespace NUMINAMATH_CALUDE_impossible_table_filling_l4070_407058

theorem impossible_table_filling (n : ℕ) (h : n ≥ 3) :
  ¬ ∃ (table : Fin n → Fin (n + 3) → ℕ),
    (∀ i j, table i j ∈ Finset.range (n * (n + 3) + 1)) ∧
    (∀ i j₁ j₂, j₁ ≠ j₂ → table i j₁ ≠ table i j₂) ∧
    (∀ i, ∃ j₁ j₂ j₃, j₁ ≠ j₂ ∧ j₁ ≠ j₃ ∧ j₂ ≠ j₃ ∧
      table i j₁ * table i j₂ = table i j₃) :=
by sorry

end NUMINAMATH_CALUDE_impossible_table_filling_l4070_407058


namespace NUMINAMATH_CALUDE_average_growth_rate_is_20_percent_l4070_407039

/-- Represents the monthly revenue growth rate as a real number between 0 and 1 -/
def MonthlyGrowthRate : Type := { r : ℝ // 0 ≤ r ∧ r ≤ 1 }

/-- The revenue in February in millions of yuan -/
def february_revenue : ℝ := 4

/-- The revenue increase rate from February to March -/
def march_increase_rate : ℝ := 0.1

/-- The revenue in May in millions of yuan -/
def may_revenue : ℝ := 633.6

/-- The number of months between March and May -/
def months_between : ℕ := 2

/-- Calculate the average monthly growth rate from March to May -/
def calculate_growth_rate (feb_rev : ℝ) (march_inc : ℝ) (may_rev : ℝ) (months : ℕ) : MonthlyGrowthRate :=
  sorry

theorem average_growth_rate_is_20_percent :
  calculate_growth_rate february_revenue march_increase_rate may_revenue months_between = ⟨0.2, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_average_growth_rate_is_20_percent_l4070_407039


namespace NUMINAMATH_CALUDE_work_group_size_work_group_size_is_9_l4070_407075

theorem work_group_size (days1 : ℕ) (days2 : ℕ) (men2 : ℕ) : ℕ :=
  let work_constant := men2 * days2
  let men1 := work_constant / days1
  men1

theorem work_group_size_is_9 :
  work_group_size 80 36 20 = 9 := by
  sorry

end NUMINAMATH_CALUDE_work_group_size_work_group_size_is_9_l4070_407075


namespace NUMINAMATH_CALUDE_complex_square_simplification_l4070_407099

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 7 - 24 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l4070_407099


namespace NUMINAMATH_CALUDE_fermat_numbers_coprime_l4070_407008

theorem fermat_numbers_coprime (m n : ℕ) (h : m ≠ n) : 
  Nat.gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fermat_numbers_coprime_l4070_407008


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l4070_407043

theorem trigonometric_inequality (h : 3 * Real.pi / 8 ∈ Set.Ioo 0 (Real.pi / 2)) :
  Real.sin (Real.cos (3 * Real.pi / 8)) < Real.cos (Real.sin (3 * Real.pi / 8)) ∧
  Real.cos (Real.sin (3 * Real.pi / 8)) < Real.sin (Real.sin (3 * Real.pi / 8)) ∧
  Real.sin (Real.sin (3 * Real.pi / 8)) < Real.cos (Real.cos (3 * Real.pi / 8)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l4070_407043


namespace NUMINAMATH_CALUDE_bee_count_l4070_407078

theorem bee_count (legs_per_bee : ℕ) (total_legs : ℕ) (h1 : legs_per_bee = 6) (h2 : total_legs = 12) :
  total_legs / legs_per_bee = 2 := by
  sorry

end NUMINAMATH_CALUDE_bee_count_l4070_407078


namespace NUMINAMATH_CALUDE_total_weight_calculation_l4070_407082

/-- Given the weight of apples and the ratio of pears to apples, 
    calculate the total weight of apples and pears. -/
def total_weight (apple_weight : ℝ) (pear_to_apple_ratio : ℝ) : ℝ :=
  apple_weight + pear_to_apple_ratio * apple_weight

/-- Theorem stating that the total weight of apples and pears is equal to
    the weight of apples plus three times the weight of apples, 
    given that there are three times as many pears as apples. -/
theorem total_weight_calculation (apple_weight : ℝ) :
  total_weight apple_weight 3 = apple_weight + 3 * apple_weight :=
by
  sorry

#eval total_weight 240 3  -- Should output 960

end NUMINAMATH_CALUDE_total_weight_calculation_l4070_407082


namespace NUMINAMATH_CALUDE_red_faces_up_possible_l4070_407079

/-- Represents a cubic block with one red face and five white faces -/
structure Block where
  redFaceUp : Bool

/-- Represents an n x n chessboard with cubic blocks -/
structure Chessboard (n : ℕ) where
  blocks : Matrix (Fin n) (Fin n) Block

/-- Represents a rotation of blocks in a row or column -/
inductive Rotation
  | Row : Fin n → Rotation
  | Column : Fin n → Rotation

/-- Applies a rotation to the chessboard -/
def applyRotation (board : Chessboard n) (rot : Rotation) : Chessboard n :=
  sorry

/-- Checks if all blocks on the chessboard have their red faces up -/
def allRedFacesUp (board : Chessboard n) : Bool :=
  sorry

/-- Theorem stating that it's possible to turn all red faces up after a finite number of rotations -/
theorem red_faces_up_possible (n : ℕ) :
  ∃ (rotations : List Rotation), ∀ (initial : Chessboard n),
    allRedFacesUp (rotations.foldl applyRotation initial) := by
  sorry

end NUMINAMATH_CALUDE_red_faces_up_possible_l4070_407079


namespace NUMINAMATH_CALUDE_f_not_satisfy_double_property_l4070_407047

-- Define the function f(x) = x + 1
def f (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem f_not_satisfy_double_property : ∃ x : ℝ, f (2 * x) ≠ 2 * f x := by
  sorry

end NUMINAMATH_CALUDE_f_not_satisfy_double_property_l4070_407047


namespace NUMINAMATH_CALUDE_perpendicular_bisector_and_parallel_line_l4070_407046

/-- Given two points A and B in the plane, this theorem proves:
    1. The equation of the perpendicular bisector of AB
    2. The equation of a line passing through P and parallel to AB -/
theorem perpendicular_bisector_and_parallel_line 
  (A B P : ℝ × ℝ) 
  (hA : A = (8, -6)) 
  (hB : B = (2, 2)) 
  (hP : P = (2, -3)) : 
  (∃ (a b c : ℝ), a * 3 = b * 4 ∧ c = 23 ∧ 
    (∀ (x y : ℝ), (a * x + b * y + c = 0) ↔ 
      (x - (A.1 + B.1) / 2)^2 + (y - (A.2 + B.2) / 2)^2 = 
      ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4)) ∧
  (∃ (d e f : ℝ), d * 4 = -e * 3 ∧ f = 1 ∧
    (∀ (x y : ℝ), (d * x + e * y + f = 0) ↔ 
      (y - P.2) = ((B.2 - A.2) / (B.1 - A.1)) * (x - P.1))) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_and_parallel_line_l4070_407046


namespace NUMINAMATH_CALUDE_shaded_area_is_32_l4070_407052

/-- Represents a rectangle in the grid --/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the grid configuration --/
structure Grid where
  totalWidth : ℕ
  totalHeight : ℕ
  rectangles : List Rectangle

def triangleArea (base height : ℕ) : ℕ :=
  base * height / 2

def rectangleArea (r : Rectangle) : ℕ :=
  r.width * r.height

def totalGridArea (g : Grid) : ℕ :=
  g.rectangles.foldl (fun acc r => acc + rectangleArea r) 0

theorem shaded_area_is_32 (g : Grid) 
    (h1 : g.totalWidth = 16)
    (h2 : g.totalHeight = 8)
    (h3 : g.rectangles = [⟨5, 4⟩, ⟨6, 6⟩, ⟨5, 8⟩])
    (h4 : triangleArea g.totalWidth g.totalHeight = 64) :
    totalGridArea g - triangleArea g.totalWidth g.totalHeight = 32 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_32_l4070_407052


namespace NUMINAMATH_CALUDE_triangle_problem_l4070_407061

theorem triangle_problem (a b c A B C : Real) (h1 : (2 * a - b) / c = Real.cos B / Real.cos C) :
  let f := fun x => 2 * Real.sin x * Real.cos x * Real.cos C + 2 * Real.sin x * Real.sin x * Real.sin C - Real.sqrt 3 / 2
  C = π / 3 ∧ Set.Icc (f 0) (f (π / 2)) = Set.Icc (-(Real.sqrt 3) / 2) 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l4070_407061


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l4070_407042

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l4070_407042


namespace NUMINAMATH_CALUDE_max_value_of_complex_expression_l4070_407088

theorem max_value_of_complex_expression (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (M : ℝ), M = 4 ∧ ∀ w : ℂ, Complex.abs w = 1 → Complex.abs (w + 2 * Real.sqrt 2 + Complex.I) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_complex_expression_l4070_407088


namespace NUMINAMATH_CALUDE_value_of_T_l4070_407086

-- Define the variables
variable (M A T H E : ℤ)

-- Define the conditions
def condition_H : H = 8 := by sorry
def condition_MATH : M + A + T + H = 47 := by sorry
def condition_MEET : M + E + E + T = 62 := by sorry
def condition_TEAM : T + E + A + M = 58 := by sorry

-- Theorem to prove
theorem value_of_T : T = 9 := by sorry

end NUMINAMATH_CALUDE_value_of_T_l4070_407086


namespace NUMINAMATH_CALUDE_probability_second_red_given_first_red_l4070_407005

def total_balls : ℕ := 10
def red_balls : ℕ := 6
def white_balls : ℕ := 4

theorem probability_second_red_given_first_red :
  let p_first_red := red_balls / total_balls
  let p_both_red := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))
  let p_second_red_given_first_red := p_both_red / p_first_red
  p_second_red_given_first_red = 5 / 9 :=
sorry

end NUMINAMATH_CALUDE_probability_second_red_given_first_red_l4070_407005


namespace NUMINAMATH_CALUDE_smallest_Y_for_binary_multiple_of_15_l4070_407011

def is_binary_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_Y_for_binary_multiple_of_15 :
  (∃ U : ℕ, is_binary_number U ∧ U % 15 = 0 ∧ U = 15 * 74) ∧
  (∀ Y : ℕ, Y < 74 → ¬∃ U : ℕ, is_binary_number U ∧ U % 15 = 0 ∧ U = 15 * Y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_Y_for_binary_multiple_of_15_l4070_407011


namespace NUMINAMATH_CALUDE_max_value_implies_a_l4070_407056

def f (a : ℝ) (x : ℝ) : ℝ := -4 * x^2 + 4 * a * x - 4 * a - a^2

theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ -5) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = -5) →
  a = 5/4 ∨ a = -5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l4070_407056


namespace NUMINAMATH_CALUDE_min_b_for_q_half_or_more_l4070_407030

def q (b : ℕ) : ℚ :=
  (Nat.choose (40 - b) 2 + Nat.choose (b - 1) 2) / 1225

theorem min_b_for_q_half_or_more : 
  ∀ b : ℕ, 1 ≤ b ∧ b ≤ 41 → (q b ≥ 1/2 ↔ b ≥ 7) :=
sorry

end NUMINAMATH_CALUDE_min_b_for_q_half_or_more_l4070_407030


namespace NUMINAMATH_CALUDE_lowest_possible_score_l4070_407080

/-- Represents a set of test scores -/
structure TestScores where
  scores : List Nat
  all_valid : ∀ s ∈ scores, s ≤ 100

/-- Calculates the average of a list of scores -/
def average (ts : TestScores) : Rat :=
  (ts.scores.sum : Rat) / ts.scores.length

/-- The problem statement -/
theorem lowest_possible_score 
  (first_two : TestScores)
  (h1 : first_two.scores = [82, 75])
  (h2 : first_two.scores.length = 2)
  : ∃ (last_two : TestScores),
    last_two.scores.length = 2 ∧ 
    (∃ (s : Nat), s ∈ last_two.scores ∧ s = 83) ∧
    average (TestScores.mk (first_two.scores ++ last_two.scores) sorry) = 85 ∧
    (∀ (other_last_two : TestScores),
      other_last_two.scores.length = 2 →
      average (TestScores.mk (first_two.scores ++ other_last_two.scores) sorry) = 85 →
      ∀ (s : Nat), s ∈ other_last_two.scores → s ≥ 83) := by
  sorry

end NUMINAMATH_CALUDE_lowest_possible_score_l4070_407080


namespace NUMINAMATH_CALUDE_order_of_exponentials_l4070_407025

theorem order_of_exponentials :
  let a : ℝ := (2 : ℝ) ^ (4/5)
  let b : ℝ := (4 : ℝ) ^ (2/7)
  let c : ℝ := (25 : ℝ) ^ (1/5)
  b < a ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_order_of_exponentials_l4070_407025


namespace NUMINAMATH_CALUDE_products_not_equal_l4070_407068

def is_valid_table (t : Fin 10 → Fin 10 → ℕ) : Prop :=
  ∀ i j, 102 ≤ t i j ∧ t i j ≤ 201 ∧ (∀ i' j', (i ≠ i' ∨ j ≠ j') → t i j ≠ t i' j')

def row_product (t : Fin 10 → Fin 10 → ℕ) (i : Fin 10) : ℕ :=
  (Finset.univ.prod fun j => t i j)

def col_product (t : Fin 10 → Fin 10 → ℕ) (j : Fin 10) : ℕ :=
  (Finset.univ.prod fun i => t i j)

def row_products (t : Fin 10 → Fin 10 → ℕ) : Finset ℕ :=
  Finset.image (row_product t) Finset.univ

def col_products (t : Fin 10 → Fin 10 → ℕ) : Finset ℕ :=
  Finset.image (col_product t) Finset.univ

theorem products_not_equal :
  ∀ t : Fin 10 → Fin 10 → ℕ, is_valid_table t → row_products t ≠ col_products t :=
sorry

end NUMINAMATH_CALUDE_products_not_equal_l4070_407068


namespace NUMINAMATH_CALUDE_specific_quadrilateral_area_l4070_407028

/-- Represents a quadrilateral ABCD with given side lengths and a right angle at C -/
structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  AD : ℝ
  right_angle_at_C : Bool

/-- Calculates the area of the quadrilateral ABCD -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific quadrilateral is 106 -/
theorem specific_quadrilateral_area :
  ∃ (q : Quadrilateral),
    q.AB = 15 ∧
    q.BC = 5 ∧
    q.CD = 12 ∧
    q.AD = 13 ∧
    q.right_angle_at_C = true ∧
    area q = 106 := by
  sorry

end NUMINAMATH_CALUDE_specific_quadrilateral_area_l4070_407028


namespace NUMINAMATH_CALUDE_smallest_difference_fraction_l4070_407017

theorem smallest_difference_fraction :
  ∀ p q : ℕ, 
    0 < q → q < 1001 → 
    |123 / 1001 - (p : ℚ) / q| ≥ |123 / 1001 - 94 / 765| := by
  sorry

end NUMINAMATH_CALUDE_smallest_difference_fraction_l4070_407017


namespace NUMINAMATH_CALUDE_fuchsia_purple_or_blue_count_l4070_407055

/-- Represents the survey results about fuchsia color perception --/
structure FuchsiaSurvey where
  total : ℕ
  like_pink : ℕ
  like_pink_and_purple : ℕ
  like_none : ℕ
  like_all : ℕ

/-- Calculates the number of people who believe fuchsia is "like purple" or "like blue" --/
def purple_or_blue (survey : FuchsiaSurvey) : ℕ :=
  survey.total - survey.like_none - (survey.like_pink - survey.like_pink_and_purple)

/-- Theorem stating that for the given survey results, 64 people believe fuchsia is "like purple" or "like blue" --/
theorem fuchsia_purple_or_blue_count :
  let survey : FuchsiaSurvey := {
    total := 150,
    like_pink := 90,
    like_pink_and_purple := 47,
    like_none := 23,
    like_all := 20
  }
  purple_or_blue survey = 64 := by
  sorry

end NUMINAMATH_CALUDE_fuchsia_purple_or_blue_count_l4070_407055


namespace NUMINAMATH_CALUDE_triangle_inradius_l4070_407001

/-- The inradius of a triangle with side lengths 7, 11, and 14 is 3√10 / 4 -/
theorem triangle_inradius (a b c : ℝ) (ha : a = 7) (hb : b = 11) (hc : c = 14) :
  let s := (a + b + c) / 2
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  A / s = (3 * Real.sqrt 10) / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_inradius_l4070_407001


namespace NUMINAMATH_CALUDE_tom_monthly_fluid_intake_l4070_407081

/-- Represents Tom's daily fluid intake --/
structure DailyFluidIntake where
  soda : Nat
  water : Nat
  juice : Nat
  sports_drink : Nat

/-- Represents Tom's additional weekend fluid intake --/
structure WeekendExtraFluidIntake where
  smoothie : Nat

/-- Represents the structure of a month --/
structure Month where
  weeks : Nat
  days_per_week : Nat
  weekdays_per_week : Nat
  weekend_days_per_week : Nat

def weekday_intake (d : DailyFluidIntake) : Nat :=
  d.soda * 12 + d.water + d.juice * 8 + d.sports_drink * 16

def weekend_intake (d : DailyFluidIntake) (w : WeekendExtraFluidIntake) : Nat :=
  weekday_intake d + w.smoothie

def total_monthly_intake (d : DailyFluidIntake) (w : WeekendExtraFluidIntake) (m : Month) : Nat :=
  (weekday_intake d * m.weekdays_per_week * m.weeks) +
  (weekend_intake d w * m.weekend_days_per_week * m.weeks)

theorem tom_monthly_fluid_intake :
  let tom_daily := DailyFluidIntake.mk 5 64 3 2
  let tom_weekend := WeekendExtraFluidIntake.mk 32
  let month := Month.mk 4 7 5 2
  total_monthly_intake tom_daily tom_weekend month = 5296 := by
  sorry

end NUMINAMATH_CALUDE_tom_monthly_fluid_intake_l4070_407081


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l4070_407038

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geom : IsGeometricSequence a)
  (h_fifth : a 5 = Nat.factorial 7)
  (h_eighth : a 8 = Nat.factorial 8) :
  a 1 = 315 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l4070_407038


namespace NUMINAMATH_CALUDE_triangle_side_length_l4070_407066

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively,
    if c = 2a, b = 4, and cos B = 1/4, then c = 4 -/
theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : c = 2 * a) 
  (h2 : b = 4) 
  (h3 : Real.cos B = 1 / 4) : 
  c = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4070_407066


namespace NUMINAMATH_CALUDE_equipment_purchase_problem_l4070_407054

/-- Equipment purchase problem -/
theorem equipment_purchase_problem 
  (price_A : ℕ)
  (price_B : ℕ)
  (discount_B : ℕ)
  (total_units : ℕ)
  (min_B : ℕ)
  (h1 : price_A = 40)
  (h2 : 30 * price_B - 5 * discount_B = 1425)
  (h3 : 50 * price_B - 25 * discount_B = 2125)
  (h4 : total_units = 90)
  (h5 : min_B = 15) :
  ∃ (units_A units_B : ℕ),
    units_A + units_B = total_units ∧
    units_B ≥ min_B ∧
    units_B ≤ 2 * units_A ∧
    units_A * price_A + (min units_B 25) * price_B + 
      (max (units_B - 25) 0) * (price_B - discount_B) = 3675 ∧
    ∀ (a b : ℕ),
      a + b = total_units →
      b ≥ min_B →
      b ≤ 2 * a →
      a * price_A + (min b 25) * price_B + 
        (max (b - 25) 0) * (price_B - discount_B) ≥ 3675 := by
  sorry

end NUMINAMATH_CALUDE_equipment_purchase_problem_l4070_407054


namespace NUMINAMATH_CALUDE_f_max_at_three_halves_l4070_407069

/-- The quadratic function f(x) = -3x^2 + 9x - 1 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x - 1

/-- The theorem states that f(x) attains its maximum value when x = 3/2 -/
theorem f_max_at_three_halves :
  ∃ (c : ℝ), c = 3/2 ∧ ∀ (x : ℝ), f x ≤ f c :=
by
  sorry

end NUMINAMATH_CALUDE_f_max_at_three_halves_l4070_407069


namespace NUMINAMATH_CALUDE_cubic_polynomial_with_irrational_product_of_roots_l4070_407023

theorem cubic_polynomial_with_irrational_product_of_roots :
  ∃ (a b c : ℚ) (u v : ℝ),
    (u^3 + a*u^2 + b*u + c = 0) ∧
    (v^3 + a*v^2 + b*v + c = 0) ∧
    ((u*v)^3 + a*(u*v)^2 + b*(u*v) + c = 0) ∧
    ¬(∃ (q : ℚ), u*v = q) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_with_irrational_product_of_roots_l4070_407023


namespace NUMINAMATH_CALUDE_parallel_planes_condition_l4070_407016

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallelLines : Line → Line → Prop)
variable (parallelPlanes : Plane → Plane → Prop)
variable (perpendicularLineToPlane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_condition 
  (m n : Line) (α β : Plane) :
  perpendicularLineToPlane m α →
  perpendicularLineToPlane n β →
  parallelLines m n →
  parallelPlanes α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_condition_l4070_407016


namespace NUMINAMATH_CALUDE_cloth_loss_problem_l4070_407024

/-- Calculates the loss per metre of cloth given the total quantity sold,
    total selling price, and cost price per metre. -/
def loss_per_metre (quantity : ℕ) (selling_price total_cost_price : ℚ) : ℚ :=
  (total_cost_price - selling_price) / quantity

theorem cloth_loss_problem (quantity : ℕ) (selling_price cost_price_per_metre : ℚ) 
  (h1 : quantity = 200)
  (h2 : selling_price = 12000)
  (h3 : cost_price_per_metre = 66) :
  loss_per_metre quantity selling_price (quantity * cost_price_per_metre) = 6 := by
sorry

end NUMINAMATH_CALUDE_cloth_loss_problem_l4070_407024


namespace NUMINAMATH_CALUDE_complex_equation_implies_ratio_l4070_407004

theorem complex_equation_implies_ratio (m n : ℝ) :
  (2 + m * Complex.I) * (n - 2 * Complex.I) = -4 - 3 * Complex.I →
  m / n = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_implies_ratio_l4070_407004


namespace NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l4070_407000

/-- Hyperbola C with equation x²/16 - y²/4 = 1 -/
def hyperbola_C (x y : ℝ) : Prop := x^2 / 16 - y^2 / 4 = 1

/-- Point P with coordinates (0, 3) -/
def point_P : ℝ × ℝ := (0, 3)

/-- Line l passing through point P -/
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x + 3

/-- Condition for point A to be on the hyperbola C and line l -/
def point_A_condition (k : ℝ) (x y : ℝ) : Prop :=
  hyperbola_C x y ∧ y = line_l k x ∧ x > 0

/-- Condition for point B to be on the hyperbola C and line l -/
def point_B_condition (k : ℝ) (x y : ℝ) : Prop :=
  hyperbola_C x y ∧ y = line_l k x ∧ x > 0

/-- Condition for point D to be on line l -/
def point_D_condition (k : ℝ) (x y : ℝ) : Prop :=
  y = line_l k x ∧ (x, y) ≠ point_P

/-- Condition for the cross ratio equality |PA| * |DB| = |PB| * |DA| -/
def cross_ratio_condition (xa ya xb yb xd yd : ℝ) : Prop :=
  (xa - 0) * (xd - xb) = (xb - 0) * (xd - xa)

theorem hyperbola_intersection_theorem (k : ℝ) 
  (xa ya xb yb xd yd : ℝ) :
  point_A_condition k xa ya →
  point_B_condition k xb yb →
  point_D_condition k xd yd →
  (xa ≠ xb) →
  cross_ratio_condition xa ya xb yb xd yd →
  yd = -4/3 := by sorry


end NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l4070_407000


namespace NUMINAMATH_CALUDE_distance_to_bus_stand_l4070_407057

/-- The distance to the bus stand in kilometers -/
def distance : ℝ := 13.5

/-- The time at which the bus arrives in hours -/
def bus_arrival_time : ℝ := 2.5

/-- Theorem stating that the distance to the bus stand is 13.5 km -/
theorem distance_to_bus_stand :
  (distance = 5 * (bus_arrival_time + 0.2)) ∧
  (distance = 6 * (bus_arrival_time - 0.25)) :=
by sorry

end NUMINAMATH_CALUDE_distance_to_bus_stand_l4070_407057


namespace NUMINAMATH_CALUDE_smallest_a_value_l4070_407074

theorem smallest_a_value (a b : ℝ) : 
  a ≥ 0 → b ≥ 0 → 
  (∀ x : ℤ, Real.sin (a * (x : ℝ) + b) = Real.sin (37 * (x : ℝ))) → 
  ∀ a' ≥ 0, (∀ x : ℤ, Real.sin (a' * (x : ℝ) + b) = Real.sin (37 * (x : ℝ))) → 
  a' ≥ 37 := by
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l4070_407074


namespace NUMINAMATH_CALUDE_oil_production_theorem_l4070_407012

/-- Oil production per person for different regions --/
def oil_production_problem : Prop :=
  let west_production := 55.084
  let non_west_production := 214.59
  let russia_production := 1038.33
  let total_production := 13737.1
  let russia_percentage := 0.09
  let russia_population := 147000000

  let russia_total_production := total_production * russia_percentage
  let russia_per_person := russia_total_production / russia_population

  (west_production = 55.084) ∧
  (non_west_production = 214.59) ∧
  (russia_per_person = 1038.33)

theorem oil_production_theorem : oil_production_problem := by
  sorry

end NUMINAMATH_CALUDE_oil_production_theorem_l4070_407012


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l4070_407098

theorem chess_tournament_participants : ∃ n : ℕ, 
  n > 0 ∧ 
  (n * (n - 1)) / 2 = 171 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l4070_407098


namespace NUMINAMATH_CALUDE_middle_number_problem_l4070_407070

theorem middle_number_problem (a b c : ℕ) : 
  a < b ∧ b < c ∧ 
  a + b = 16 ∧ 
  a + c = 21 ∧ 
  b + c = 27 → 
  b = 11 := by
sorry

end NUMINAMATH_CALUDE_middle_number_problem_l4070_407070


namespace NUMINAMATH_CALUDE_rectangle_area_is_144_l4070_407034

-- Define the radius of the circles
def circle_radius : ℝ := 3

-- Define the rectangle
structure Rectangle where
  length : ℝ
  width : ℝ

-- Define the property that circles touch the sides of the rectangle
def circles_touch_sides (r : Rectangle) : Prop :=
  r.length = 4 * circle_radius ∧ r.width = 4 * circle_radius

-- Define the area of the rectangle
def rectangle_area (r : Rectangle) : ℝ :=
  r.length * r.width

-- Theorem statement
theorem rectangle_area_is_144 (r : Rectangle) 
  (h : circles_touch_sides r) : rectangle_area r = 144 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_144_l4070_407034


namespace NUMINAMATH_CALUDE_number_problem_l4070_407087

theorem number_problem : ∃ x : ℚ, x^2 + 95 = (x - 15)^2 ∧ x = 13/3 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4070_407087


namespace NUMINAMATH_CALUDE_jimmy_bread_packs_l4070_407002

/-- The number of packs of bread needed for a given number of sandwiches -/
def bread_packs_needed (num_sandwiches : ℕ) (slices_per_sandwich : ℕ) (slices_per_pack : ℕ) (initial_slices : ℕ) : ℕ :=
  ((num_sandwiches * slices_per_sandwich - initial_slices) + slices_per_pack - 1) / slices_per_pack

/-- Theorem: Jimmy needs 4 packs of bread for his picnic -/
theorem jimmy_bread_packs : bread_packs_needed 8 2 4 0 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_bread_packs_l4070_407002


namespace NUMINAMATH_CALUDE_solution_set_characterization_l4070_407073

/-- An odd function satisfying certain conditions -/
def OddFunctionWithConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x < 0, 2 * x * (deriv f (2 * x)) + f (2 * x) < 0) ∧
  f (-2) = 0

/-- The solution set of xf(2x) < 0 -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | x * f (2 * x) < 0}

/-- The main theorem -/
theorem solution_set_characterization (f : ℝ → ℝ) 
  (hf : OddFunctionWithConditions f) : 
  SolutionSet f = {x | -1 < x ∧ x < 1 ∧ x ≠ 0} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l4070_407073


namespace NUMINAMATH_CALUDE_jills_speed_l4070_407059

/-- Proves that Jill's speed was 8 km/h given the conditions of the problem -/
theorem jills_speed (jack_distance1 jack_distance2 jack_speed1 jack_speed2 : ℝ)
  (h1 : jack_distance1 = 12)
  (h2 : jack_distance2 = 12)
  (h3 : jack_speed1 = 12)
  (h4 : jack_speed2 = 6)
  (jill_distance jill_time : ℝ)
  (h5 : jill_distance = jack_distance1 + jack_distance2)
  (h6 : jill_time = jack_distance1 / jack_speed1 + jack_distance2 / jack_speed2) :
  jill_distance / jill_time = 8 :=
sorry

end NUMINAMATH_CALUDE_jills_speed_l4070_407059


namespace NUMINAMATH_CALUDE_triangle_side_relation_l4070_407015

theorem triangle_side_relation (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_angle : Real.cos (2 * Real.pi / 3) = -1/2) :
  a^2 + a*c + c^2 - b^2 = 0 := by sorry

end NUMINAMATH_CALUDE_triangle_side_relation_l4070_407015


namespace NUMINAMATH_CALUDE_energy_drink_consumption_l4070_407096

/-- Represents the relationship between coding hours and energy drink consumption -/
def energy_drink_relation (hours : ℝ) (drinks : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ hours * drinks = k

theorem energy_drink_consumption 
  (h1 : energy_drink_relation 8 3)
  (h2 : energy_drink_relation 10 x) :
  x = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_energy_drink_consumption_l4070_407096


namespace NUMINAMATH_CALUDE_solve_equation_l4070_407048

theorem solve_equation (x y : ℝ) : y = 2 / (5 * x + 3) → y = 2 → x = -2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4070_407048


namespace NUMINAMATH_CALUDE_area_equality_l4070_407006

-- Define the areas of various shapes
variable (S_Quadrilateral_BHCG : ℝ)
variable (S_Quadrilateral_AGDH : ℝ)
variable (S_Triangle_ABG : ℝ)
variable (S_Triangle_DCG : ℝ)
variable (S_Triangle_DEH : ℝ)
variable (S_Triangle_AFH : ℝ)
variable (S_Triangle_AOG : ℝ)
variable (S_Triangle_DOG : ℝ)
variable (S_Triangle_DOH : ℝ)
variable (S_Triangle_AOH : ℝ)
variable (S_Shaded : ℝ)
variable (S_Triangle_EFH : ℝ)
variable (S_Triangle_BCG : ℝ)

-- State the theorem
theorem area_equality 
  (h1 : S_Quadrilateral_BHCG / S_Quadrilateral_AGDH = 1 / 4)
  (h2 : S_Triangle_ABG + S_Triangle_DCG + S_Triangle_DEH + S_Triangle_AFH = 
        S_Triangle_AOG + S_Triangle_DOG + S_Triangle_DOH + S_Triangle_AOH)
  (h3 : S_Triangle_ABG + S_Triangle_DCG + S_Triangle_DEH + S_Triangle_AFH = S_Shaded)
  (h4 : S_Triangle_EFH + S_Triangle_BCG = S_Quadrilateral_BHCG)
  (h5 : S_Quadrilateral_BHCG = 1/4 * S_Shaded) :
  S_Quadrilateral_AGDH = S_Shaded :=
by sorry

end NUMINAMATH_CALUDE_area_equality_l4070_407006


namespace NUMINAMATH_CALUDE_unfair_coin_expected_value_l4070_407022

/-- Given an unfair coin with the following properties:
  * Probability of heads: 2/3
  * Probability of tails: 1/3
  * Gain on heads: $5
  * Loss on tails: $12
  This theorem proves that the expected value of a single coin flip
  is -2/3 dollars. -/
theorem unfair_coin_expected_value :
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let gain_heads : ℚ := 5
  let loss_tails : ℚ := 12
  p_heads * gain_heads + p_tails * (-loss_tails) = -2/3 := by
sorry

end NUMINAMATH_CALUDE_unfair_coin_expected_value_l4070_407022
