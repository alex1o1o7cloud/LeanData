import Mathlib

namespace NUMINAMATH_CALUDE_sin_theta_value_l1145_114511

theorem sin_theta_value (θ : Real) 
  (h1 : 10 * Real.tan θ = 2 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.sin θ = (-5 + Real.sqrt 29) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l1145_114511


namespace NUMINAMATH_CALUDE_crayons_count_l1145_114594

/-- The number of rows of crayons -/
def num_rows : ℕ := 7

/-- The number of crayons in each row -/
def crayons_per_row : ℕ := 30

/-- The total number of crayons -/
def total_crayons : ℕ := num_rows * crayons_per_row

theorem crayons_count : total_crayons = 210 := by
  sorry

end NUMINAMATH_CALUDE_crayons_count_l1145_114594


namespace NUMINAMATH_CALUDE_money_left_over_l1145_114526

/-- The amount of money left over after purchasing bread, peanut butter, and honey with a discount coupon. -/
theorem money_left_over (bread_price : ℝ) (peanut_butter_price : ℝ) (honey_price : ℝ)
  (bread_quantity : ℕ) (peanut_butter_quantity : ℕ) (honey_quantity : ℕ)
  (discount : ℝ) (initial_money : ℝ) :
  bread_price = 2.35 →
  peanut_butter_price = 3.10 →
  honey_price = 4.50 →
  bread_quantity = 4 →
  peanut_butter_quantity = 2 →
  honey_quantity = 1 →
  discount = 2 →
  initial_money = 20 →
  initial_money - (bread_price * bread_quantity + peanut_butter_price * peanut_butter_quantity + 
    honey_price * honey_quantity - discount) = 1.90 := by
  sorry

end NUMINAMATH_CALUDE_money_left_over_l1145_114526


namespace NUMINAMATH_CALUDE_odd_function_extension_l1145_114565

-- Define an odd function f
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_extension
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_pos : ∀ x > 0, f x = x^3 + x + 1) :
  ∀ x < 0, f x = x^3 + x - 1 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_extension_l1145_114565


namespace NUMINAMATH_CALUDE_binary_110101_equals_53_l1145_114507

-- Define the binary number as a list of bits
def binary_number : List Bool := [true, true, false, true, false, true]

-- Define a function to convert binary to decimal
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Theorem statement
theorem binary_110101_equals_53 :
  binary_to_decimal binary_number = 53 := by
  sorry

end NUMINAMATH_CALUDE_binary_110101_equals_53_l1145_114507


namespace NUMINAMATH_CALUDE_parabola_vertex_l1145_114585

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = x^2 - 2x + 3 is (1, 2) -/
theorem parabola_vertex : 
  (∀ x : ℝ, f x = (x - vertex.1)^2 + vertex.2) ∧ 
  (∀ x : ℝ, f x ≥ f vertex.1) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1145_114585


namespace NUMINAMATH_CALUDE_davids_math_marks_l1145_114512

/-- Represents the marks obtained by David in various subjects -/
structure Marks where
  english : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  mathematics : ℕ

/-- Calculates the average marks given the total marks and number of subjects -/
def average (total : ℕ) (subjects : ℕ) : ℚ :=
  (total : ℚ) / (subjects : ℚ)

/-- Theorem stating that given David's marks in other subjects and his average,
    his Mathematics marks must be 60 -/
theorem davids_math_marks (m : Marks) (h1 : m.english = 72) (h2 : m.physics = 35)
    (h3 : m.chemistry = 62) (h4 : m.biology = 84)
    (h5 : average (m.english + m.physics + m.chemistry + m.biology + m.mathematics) 5 = 62.6) :
    m.mathematics = 60 := by
  sorry

#check davids_math_marks

end NUMINAMATH_CALUDE_davids_math_marks_l1145_114512


namespace NUMINAMATH_CALUDE_bridge_length_l1145_114534

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) : 
  train_length = 100 →
  train_speed_kmh = 45 →
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time_s - train_length = 275 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l1145_114534


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1145_114541

theorem fraction_to_decimal : (5 : ℚ) / 50 = 0.1 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1145_114541


namespace NUMINAMATH_CALUDE_expression_evaluation_l1145_114539

theorem expression_evaluation : (4 * 6) / (12 * 16) * (8 * 12 * 16) / (4 * 6 * 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1145_114539


namespace NUMINAMATH_CALUDE_salad_dressing_ratio_l1145_114577

theorem salad_dressing_ratio (bowl_capacity : ℝ) (oil_density : ℝ) (vinegar_density : ℝ) 
  (total_weight : ℝ) (oil_fraction : ℝ) :
  bowl_capacity = 150 →
  oil_fraction = 2/3 →
  oil_density = 5 →
  vinegar_density = 4 →
  total_weight = 700 →
  (total_weight - oil_fraction * bowl_capacity * oil_density) / vinegar_density / bowl_capacity = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_salad_dressing_ratio_l1145_114577


namespace NUMINAMATH_CALUDE_total_fence_cost_l1145_114551

/-- Represents the cost of building a fence for a pentagonal plot -/
def fence_cost (a b c d e : ℕ) (pa pb pc pd pe : ℕ) : ℕ :=
  a * pa + b * pb + c * pc + d * pd + e * pe

/-- Theorem stating the total cost of the fence -/
theorem total_fence_cost :
  fence_cost 9 12 15 11 13 45 55 60 50 65 = 3360 := by
  sorry

end NUMINAMATH_CALUDE_total_fence_cost_l1145_114551


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l1145_114568

theorem quadratic_form_equivalence (k : ℝ) :
  (∃ (a b : ℝ), ∀ (x : ℝ), (3*k - 2)*x*(x + k) + k^2*(k - 1) = (a*x + b)^2) ↔ k = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l1145_114568


namespace NUMINAMATH_CALUDE_number_of_factors_of_M_l1145_114556

def M : Nat := 2^5 * 3^4 * 5^3 * 11^2

theorem number_of_factors_of_M : 
  (Finset.filter (·∣M) (Finset.range (M + 1))).card = 360 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_M_l1145_114556


namespace NUMINAMATH_CALUDE_log_seven_eighteen_l1145_114505

theorem log_seven_eighteen (a b : ℝ) 
  (h1 : Real.log 2 / Real.log 10 = a) 
  (h2 : Real.log 3 / Real.log 10 = b) : 
  Real.log 18 / Real.log 7 = (2*a + 4*b) / (1 + 2*a) := by
  sorry

end NUMINAMATH_CALUDE_log_seven_eighteen_l1145_114505


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l1145_114509

theorem number_puzzle_solution : 
  ∃ x : ℚ, x - (3/5) * x = 50 ∧ x = 125 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l1145_114509


namespace NUMINAMATH_CALUDE_arrange_five_photos_l1145_114580

theorem arrange_five_photos (n : ℕ) (h : n = 5) : Nat.factorial n = 120 := by
  sorry

end NUMINAMATH_CALUDE_arrange_five_photos_l1145_114580


namespace NUMINAMATH_CALUDE_calculate_interest_rate_l1145_114538

/-- Given simple interest, principal, and time, calculate the interest rate. -/
theorem calculate_interest_rate 
  (simple_interest principal time rate : ℝ) 
  (h1 : simple_interest = 400)
  (h2 : principal = 1200)
  (h3 : time = 4)
  (h4 : simple_interest = principal * rate * time / 100) :
  rate = 400 * 100 / (1200 * 4) :=
by sorry

end NUMINAMATH_CALUDE_calculate_interest_rate_l1145_114538


namespace NUMINAMATH_CALUDE_continuous_function_composite_power_l1145_114593

theorem continuous_function_composite_power (k : ℝ) :
  (∃ f : ℝ → ℝ, Continuous f ∧ ∀ x, f (f x) = k * x^9) → k ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_continuous_function_composite_power_l1145_114593


namespace NUMINAMATH_CALUDE_min_value_of_f_l1145_114502

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x ∈ Set.Icc (-1) 0, f x ≥ m) ∧ (∃ x ∈ Set.Icc (-1) 0, f x = m) ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1145_114502


namespace NUMINAMATH_CALUDE_cubic_difference_l1145_114572

theorem cubic_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 65) : 
  a^3 - b^3 = 511 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_l1145_114572


namespace NUMINAMATH_CALUDE_memorial_visitors_equation_l1145_114590

theorem memorial_visitors_equation (x : ℕ) (h1 : x + (2 * x + 56) = 589) : 2 * x + 56 = 589 - x := by
  sorry

end NUMINAMATH_CALUDE_memorial_visitors_equation_l1145_114590


namespace NUMINAMATH_CALUDE_F_sum_positive_l1145_114591

/-- The function f(x) = ax^2 + bx + 1 -/
noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- The function F(x) defined piecewise based on f(x) -/
noncomputable def F (a b x : ℝ) : ℝ :=
  if x > 0 then f a b x else -f a b x

/-- Theorem stating that F(m) + F(n) > 0 under given conditions -/
theorem F_sum_positive (a b m n : ℝ) : 
  f a b (-1) = 0 → 
  (∀ x, f a b x ≥ 0) → 
  m * n < 0 → 
  m + n > 0 → 
  a > 0 → 
  (∀ x, f a b x = f a b (-x)) → 
  F a b m + F a b n > 0 := by
  sorry

end NUMINAMATH_CALUDE_F_sum_positive_l1145_114591


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1145_114571

/-- Represents a hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a line with equation x - √3y + 2 = 0 -/
def special_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 2 = 0

theorem hyperbola_real_axis_length
  (h : Hyperbola)
  (focus_on_line : ∃ (x y : ℝ), special_line x y ∧ x^2 / h.a^2 - y^2 / h.b^2 = 1)
  (perpendicular_to_asymptote : h.b / h.a = Real.sqrt 3) :
  2 * h.a = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1145_114571


namespace NUMINAMATH_CALUDE_tims_earnings_per_visit_l1145_114584

theorem tims_earnings_per_visit
  (visitors_per_day : ℕ)
  (regular_days : ℕ)
  (total_earnings : ℚ)
  (h1 : visitors_per_day = 100)
  (h2 : regular_days = 6)
  (h3 : total_earnings = 18)
  : 
  let total_visitors := visitors_per_day * regular_days + 2 * (visitors_per_day * regular_days)
  total_earnings / total_visitors = 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_tims_earnings_per_visit_l1145_114584


namespace NUMINAMATH_CALUDE_min_max_abs_quadratic_l1145_114547

theorem min_max_abs_quadratic :
  ∃ y : ℝ, ∀ z : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |x^2 - x*y + x| ≤ (⨆ x ∈ {x : ℝ | 0 ≤ x ∧ x ≤ 2}, |x^2 - x*z + x|)) ∧
  (⨆ x ∈ {x : ℝ | 0 ≤ x ∧ x ≤ 2}, |x^2 - x*y + x|) = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_max_abs_quadratic_l1145_114547


namespace NUMINAMATH_CALUDE_min_transport_time_l1145_114531

/-- The minimum time required for transporting goods between two cities --/
theorem min_transport_time (distance : ℝ) (num_trains : ℕ) (speed : ℝ) 
  (h1 : distance = 400)
  (h2 : num_trains = 17)
  (h3 : speed > 0) :
  (distance / speed + (num_trains - 1) * (speed / 20)^2 / speed) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_transport_time_l1145_114531


namespace NUMINAMATH_CALUDE_insufficient_comparisons_l1145_114529

/-- Represents a comparison of three elements -/
structure TripleComparison (α : Type) where
  a : α
  b : α
  c : α

/-- The type of all possible orderings of n distinct elements -/
def Orderings (n : ℕ) := Fin n → Fin n

/-- The number of possible orderings for n distinct elements -/
def num_orderings (n : ℕ) : ℕ := n.factorial

/-- The maximum number of orderings that can be eliminated by a single triple comparison -/
def max_eliminated_by_comparison (n : ℕ) : ℕ := (n - 2).factorial

/-- The number of comparisons allowed -/
def num_comparisons : ℕ := 9

/-- The number of distinct elements to be ordered -/
def num_elements : ℕ := 5

/-- Theorem stating that the given number of comparisons is insufficient -/
theorem insufficient_comparisons :
  ∃ (remaining : ℕ), remaining > 1 ∧
  remaining ≤ num_orderings num_elements - num_comparisons * max_eliminated_by_comparison num_elements :=
sorry

end NUMINAMATH_CALUDE_insufficient_comparisons_l1145_114529


namespace NUMINAMATH_CALUDE_function_not_in_first_quadrant_l1145_114578

theorem function_not_in_first_quadrant
  (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : b < -1) :
  ∀ x y : ℝ, y = a^x + b → ¬(x > 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_function_not_in_first_quadrant_l1145_114578


namespace NUMINAMATH_CALUDE_range_of_m_for_odd_function_with_conditions_l1145_114564

/-- An odd function f: ℝ → ℝ satisfying certain conditions -/
def OddFunctionWithConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (3/2 + x) = f (3/2 - x)) ∧
  (f 5 > -2) ∧
  (∃ m : ℝ, f 2 = m - 3/m)

/-- The range of m for the given function f -/
def RangeOfM (f : ℝ → ℝ) : Set ℝ :=
  {m : ℝ | m < -1 ∨ (0 < m ∧ m < 3)}

/-- Theorem stating the range of m for a function satisfying the given conditions -/
theorem range_of_m_for_odd_function_with_conditions (f : ℝ → ℝ) 
  (h : OddFunctionWithConditions f) : 
  ∃ m : ℝ, f 2 = m - 3/m ∧ m ∈ RangeOfM f := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_odd_function_with_conditions_l1145_114564


namespace NUMINAMATH_CALUDE_probability_is_one_third_l1145_114503

/-- Right triangle XYZ with XY = 10 and XZ = 6 -/
structure RightTriangle where
  xy : ℝ
  xz : ℝ
  xy_eq : xy = 10
  xz_eq : xz = 6

/-- Random point Q in the interior of triangle XYZ -/
def RandomPoint (t : RightTriangle) : Type := Unit

/-- Area of triangle QYZ -/
def AreaQYZ (t : RightTriangle) (q : RandomPoint t) : ℝ := sorry

/-- Area of triangle XYZ -/
def AreaXYZ (t : RightTriangle) : ℝ := sorry

/-- Probability that area of QYZ is less than one-third of area of XYZ -/
def Probability (t : RightTriangle) : ℝ := sorry

/-- Theorem: The probability is equal to 1/3 -/
theorem probability_is_one_third (t : RightTriangle) :
  Probability t = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_third_l1145_114503


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1145_114570

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_a1 : a 1 = 1) 
  (h_a3 : a 3 = -3) : 
  ∀ n : ℕ, a n = -2 * n + 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1145_114570


namespace NUMINAMATH_CALUDE_fox_distribution_l1145_114519

/-- The fox distribution problem -/
theorem fox_distribution
  (m : ℕ) (a : ℝ) (x y : ℝ)
  (h_positive : m > 1 ∧ a > 0)
  (h_distribution : ∀ (n : ℕ), n > 0 → n * a + (x - (n - 1) * y - n * a) / m = y) :
  x = (m - 1)^2 * a ∧ y = (m - 1) * a ∧ (m - 1 : ℝ) = x / y :=
by sorry

end NUMINAMATH_CALUDE_fox_distribution_l1145_114519


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l1145_114563

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : 4/x + 3/y = 1) : 
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 4/x₀ + 3/y₀ = 1 ∧ x₀ * y₀ = 48 ∧ ∀ x' y', x' > 0 → y' > 0 → 4/x' + 3/y' = 1 → x' * y' ≥ 48) ∧
  (∃ (x₁ y₁ : ℝ), x₁ > 0 ∧ y₁ > 0 ∧ 4/x₁ + 3/y₁ = 1 ∧ x₁ + y₁ = 7 + 4 * Real.sqrt 3 ∧ ∀ x' y', x' > 0 → y' > 0 → 4/x' + 3/y' = 1 → x' + y' ≥ 7 + 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l1145_114563


namespace NUMINAMATH_CALUDE_shoe_store_sale_l1145_114535

theorem shoe_store_sale (sneakers sandals boots : ℕ) 
  (h1 : sneakers = 2) 
  (h2 : sandals = 4) 
  (h3 : boots = 11) : 
  sneakers + sandals + boots = 17 := by
  sorry

end NUMINAMATH_CALUDE_shoe_store_sale_l1145_114535


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1145_114542

theorem complex_fraction_equality : (2 * Complex.I) / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1145_114542


namespace NUMINAMATH_CALUDE_cutting_tool_geometry_l1145_114574

theorem cutting_tool_geometry (A B C : ℝ × ℝ) : 
  let r : ℝ := 6
  let AB : ℝ := 5
  let BC : ℝ := 3
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = AB^2 →
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = BC^2 →
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 →
  A.1^2 + A.2^2 = r^2 →
  B.1^2 + B.2^2 = r^2 →
  C.1^2 + C.2^2 = r^2 →
  (B.1^2 + B.2^2 = 4.16 ∧ A = (2, 5.4) ∧ C = (5, 0.4)) := by
sorry

end NUMINAMATH_CALUDE_cutting_tool_geometry_l1145_114574


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1145_114508

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ (b : ℝ), (2 - a * Complex.I) / (1 + Complex.I) = Complex.I * b) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1145_114508


namespace NUMINAMATH_CALUDE_asset_value_increase_l1145_114520

theorem asset_value_increase (initial_value : ℝ) (h : initial_value > 0) :
  let year1_increase := 0.2
  let year2_increase := 0.3
  let year1_value := initial_value * (1 + year1_increase)
  let year2_value := year1_value * (1 + year2_increase)
  let total_increase := (year2_value - initial_value) / initial_value
  total_increase = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_asset_value_increase_l1145_114520


namespace NUMINAMATH_CALUDE_line_perp_two_planes_implies_parallel_l1145_114559

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- State the theorem
theorem line_perp_two_planes_implies_parallel 
  (h1 : perpendicular n α) 
  (h2 : perpendicular n β) : 
  parallel α β := by sorry

end NUMINAMATH_CALUDE_line_perp_two_planes_implies_parallel_l1145_114559


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_prism_height_isosceles_trapezoid_prism_height_proof_l1145_114596

/-- Represents a prism with a base that is an isosceles trapezoid inscribed around a circle -/
structure IsoscelesTrapezoidPrism where
  r : ℝ  -- radius of the inscribed circle
  α : ℝ  -- acute angle of the trapezoid

/-- 
Theorem: The height of the prism is 2r tan(α) given:
- The base is an isosceles trapezoid inscribed around a circle with radius r
- The acute angle of the trapezoid is α
- A plane passing through one side of the base and the acute angle endpoint 
  of the opposite side of the top plane forms an angle α with the base plane
-/
theorem isosceles_trapezoid_prism_height 
  (prism : IsoscelesTrapezoidPrism) : ℝ :=
  2 * prism.r * Real.tan prism.α

-- Proof
theorem isosceles_trapezoid_prism_height_proof
  (prism : IsoscelesTrapezoidPrism) :
  isosceles_trapezoid_prism_height prism = 2 * prism.r * Real.tan prism.α := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_prism_height_isosceles_trapezoid_prism_height_proof_l1145_114596


namespace NUMINAMATH_CALUDE_count_perfect_squares_l1145_114566

/-- The number of positive perfect square factors of (2^12)(3^15)(5^18)(7^8) -/
def num_perfect_square_factors : ℕ := 2800

/-- The product in question -/
def product : ℕ := 2^12 * 3^15 * 5^18 * 7^8

/-- A function that counts the number of positive perfect square factors of a natural number -/
def count_perfect_square_factors (n : ℕ) : ℕ := sorry

theorem count_perfect_squares :
  count_perfect_square_factors product = num_perfect_square_factors := by sorry

end NUMINAMATH_CALUDE_count_perfect_squares_l1145_114566


namespace NUMINAMATH_CALUDE_proposition_truth_l1145_114514

theorem proposition_truth (p q : Prop) 
  (h1 : ¬p) 
  (h2 : p ∨ q) : 
  ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_proposition_truth_l1145_114514


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1145_114517

theorem min_value_sum_of_reciprocals (n : ℕ) (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / (1 + a^n) + 1 / (1 + b^n)) ≥ 1 ∧ 
  (1 / (1 + 1^n) + 1 / (1 + 1^n) = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1145_114517


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l1145_114523

theorem quadratic_roots_sum_squares (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 1 = 0 → x₂^2 - 3*x₂ + 1 = 0 → x₁^2 + 3*x₁*x₂ + x₂^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l1145_114523


namespace NUMINAMATH_CALUDE_smallest_number_of_nuts_l1145_114546

theorem smallest_number_of_nuts (N : ℕ) : N = 320 ↔ 
  N > 0 ∧
  N % 11 = 1 ∧
  N % 13 = 8 ∧
  N % 17 = 3 ∧
  N > 41 ∧
  (∀ M : ℕ, M > 0 ∧ M % 11 = 1 ∧ M % 13 = 8 ∧ M % 17 = 3 ∧ M > 41 → N ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_nuts_l1145_114546


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_x_range_of_x_for_inequality_l1145_114587

-- Define the function f
def f (x : ℝ) := |2*x - 1| - |x + 1|

-- Theorem for part I
theorem solution_set_f_greater_than_x :
  {x : ℝ | f x > x} = {x : ℝ | x < 0} := by sorry

-- Theorem for part II
theorem range_of_x_for_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x : ℝ, (1/a + 4/b) ≥ f x) →
  (∀ x : ℝ, f x ≤ 9) →
  (∀ x : ℝ, -7 ≤ x ∧ x ≤ 11) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_x_range_of_x_for_inequality_l1145_114587


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1145_114532

/-- Given a principal amount and an interest rate, if the amount after 2 years is 720
    and after 7 years is 1020, then the principal amount is 600. -/
theorem simple_interest_problem (P R : ℚ) 
  (h1 : P + (P * R * 2) / 100 = 720)
  (h2 : P + (P * R * 7) / 100 = 1020) : 
  P = 600 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1145_114532


namespace NUMINAMATH_CALUDE_metallic_sheet_width_l1145_114550

/-- Given a rectangular metallic sheet with length 48 m, from which squares of side 
    length 6 m are cut from each corner to form an open box, prove that if the 
    volume of the resulting box is 5184 m³, then the width of the original 
    metallic sheet is 36 m. -/
theorem metallic_sheet_width (sheet_length : ℝ) (cut_square_side : ℝ) (box_volume : ℝ) 
  (sheet_width : ℝ) :
  sheet_length = 48 →
  cut_square_side = 6 →
  box_volume = 5184 →
  box_volume = (sheet_length - 2 * cut_square_side) * 
               (sheet_width - 2 * cut_square_side) * 
               cut_square_side →
  sheet_width = 36 :=
by sorry

end NUMINAMATH_CALUDE_metallic_sheet_width_l1145_114550


namespace NUMINAMATH_CALUDE_women_average_age_l1145_114592

theorem women_average_age (n : ℕ) (A : ℝ) (age1 age2 : ℕ) :
  n = 10 ∧ 
  age1 = 10 ∧ 
  age2 = 12 ∧ 
  (n * A - age1 - age2 + 2 * ((n * (A + 2)) - (n * A - age1 - age2))) / 2 = 21 :=
by sorry

end NUMINAMATH_CALUDE_women_average_age_l1145_114592


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1145_114527

theorem gcd_of_specific_numbers : Nat.gcd 333333 888888888 = 3 := by sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1145_114527


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1145_114543

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_m_value :
  ∀ m : ℝ, parallel (m, 4) (3, -2) → m = -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1145_114543


namespace NUMINAMATH_CALUDE_inequality_solution_range_of_a_l1145_114500

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x - 3|

-- Theorem for the solution of the inequality
theorem inequality_solution :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -1/4 ≤ x ∧ x ≤ 9/4} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, ∃ y, Real.log y = f x + a} = {a : ℝ | a > -2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_of_a_l1145_114500


namespace NUMINAMATH_CALUDE_exam_score_below_mean_l1145_114569

/-- Given an exam with a mean score and a known score above the mean,
    calculate the score that is a certain number of standard deviations below the mean. -/
theorem exam_score_below_mean
  (mean : ℝ)
  (score_above : ℝ)
  (sd_above : ℝ)
  (sd_below : ℝ)
  (h1 : mean = 74)
  (h2 : score_above = 98)
  (h3 : sd_above = 3)
  (h4 : sd_below = 2)
  (h5 : score_above = mean + sd_above * ((score_above - mean) / sd_above)) :
  mean - sd_below * ((score_above - mean) / sd_above) = 58 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_below_mean_l1145_114569


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_zero_l1145_114588

theorem imaginary_unit_sum_zero (i : ℂ) (hi : i^2 = -1) : i + i^2 + i^3 + i^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_zero_l1145_114588


namespace NUMINAMATH_CALUDE_min_value_of_a_l1145_114558

theorem min_value_of_a (a : ℝ) : 
  (∀ x > a, 2 * x + 1 / (x - a)^2 ≥ 7) → a ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l1145_114558


namespace NUMINAMATH_CALUDE_line_passes_through_parabola_vertex_l1145_114581

/-- The number of values of b for which the line y = 2x + b passes through
    the vertex of the parabola y = x^2 - 2bx + b^2 is exactly 1. -/
theorem line_passes_through_parabola_vertex :
  ∃! b : ℝ, ∀ x y : ℝ,
    (y = 2 * x + b) ∧ (y = x^2 - 2 * b * x + b^2) →
    (x = b ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_parabola_vertex_l1145_114581


namespace NUMINAMATH_CALUDE_smaug_gold_coins_l1145_114575

/-- Represents the number of coins in Smaug's hoard -/
structure DragonHoard where
  gold : ℕ
  silver : ℕ
  copper : ℕ

/-- Represents the value of different coin types in terms of copper coins -/
structure CoinValues where
  silver_to_copper : ℕ
  gold_to_silver : ℕ

/-- Calculates the total value of a hoard in copper coins -/
def hoard_value (hoard : DragonHoard) (values : CoinValues) : ℕ :=
  hoard.gold * values.gold_to_silver * values.silver_to_copper +
  hoard.silver * values.silver_to_copper +
  hoard.copper

/-- Theorem stating that Smaug has 100 gold coins -/
theorem smaug_gold_coins : 
  ∀ (hoard : DragonHoard) (values : CoinValues),
    hoard.silver = 60 →
    hoard.copper = 33 →
    values.silver_to_copper = 8 →
    values.gold_to_silver = 3 →
    hoard_value hoard values = 2913 →
    hoard.gold = 100 := by
  sorry

end NUMINAMATH_CALUDE_smaug_gold_coins_l1145_114575


namespace NUMINAMATH_CALUDE_mcdonald_farm_weeks_l1145_114536

/-- The number of weeks required for Mcdonald's farm to produce the total number of eggs -/
def weeks_required (saly_eggs ben_eggs total_eggs : ℕ) : ℕ :=
  total_eggs / (saly_eggs + ben_eggs + ben_eggs / 2)

/-- Theorem stating that the number of weeks required is 4 -/
theorem mcdonald_farm_weeks : weeks_required 10 14 124 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mcdonald_farm_weeks_l1145_114536


namespace NUMINAMATH_CALUDE_gcd_problem_l1145_114513

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 1177) :
  Int.gcd (3 * b^2 + 34 * b + 76) (b + 14) = 2 := by sorry

end NUMINAMATH_CALUDE_gcd_problem_l1145_114513


namespace NUMINAMATH_CALUDE_nine_cakes_l1145_114586

/-- Represents the arrangement of cakes on a round table. -/
def CakeArrangement (n : ℕ) := Fin n

/-- Represents the action of eating every third cake. -/
def eatEveryThird (n : ℕ) (i : Fin n) : Fin n :=
  ⟨(i + 3) % n, by sorry⟩

/-- Represents the number of laps needed to eat all cakes. -/
def lapsToEatAll (n : ℕ) : ℕ := 7

/-- The last cake eaten is the same as the first one encountered. -/
def lastIsFirst (n : ℕ) : Prop :=
  ∃ (i : Fin n), (lapsToEatAll n).iterate (eatEveryThird n) i = i

/-- The main theorem stating that there are 9 cakes on the table. -/
theorem nine_cakes :
  ∃ (n : ℕ), n = 9 ∧ 
  lapsToEatAll n = 7 ∧
  lastIsFirst n :=
sorry

end NUMINAMATH_CALUDE_nine_cakes_l1145_114586


namespace NUMINAMATH_CALUDE_stamp_exchange_theorem_l1145_114524

/-- Represents the number of stamp collectors and countries -/
def n : ℕ := 26

/-- The minimum number of letters needed to exchange stamps -/
def min_letters (n : ℕ) : ℕ := 2 * (n - 1)

/-- Theorem stating the minimum number of letters needed for stamp exchange -/
theorem stamp_exchange_theorem :
  min_letters n = 50 :=
by sorry

end NUMINAMATH_CALUDE_stamp_exchange_theorem_l1145_114524


namespace NUMINAMATH_CALUDE_larger_number_problem_l1145_114597

theorem larger_number_problem (x y : ℝ) : 3 * y = 4 * x → y - x = 8 → y = 32 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1145_114597


namespace NUMINAMATH_CALUDE_sum_square_values_l1145_114582

theorem sum_square_values (K M : ℕ) : 
  K * (K + 1) / 2 = M^2 →
  M < 200 →
  K > M →
  (K = 8 ∨ K = 49) ∧ 
  (∀ n : ℕ, n * (n + 1) / 2 = M^2 ∧ M < 200 ∧ n > M → n = 8 ∨ n = 49) :=
by sorry

end NUMINAMATH_CALUDE_sum_square_values_l1145_114582


namespace NUMINAMATH_CALUDE_hostel_provisions_l1145_114557

/-- Given a hostel with provisions for a certain number of men, 
    calculate the initial number of days the provisions were planned for. -/
theorem hostel_provisions 
  (initial_men : ℕ) 
  (men_left : ℕ) 
  (days_after_leaving : ℕ) 
  (h1 : initial_men = 250)
  (h2 : men_left = 50)
  (h3 : days_after_leaving = 60) :
  (initial_men * (initial_men - men_left) * days_after_leaving) / 
  ((initial_men - men_left) * initial_men) = 48 := by
sorry

end NUMINAMATH_CALUDE_hostel_provisions_l1145_114557


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1145_114504

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (x + 8) - (4 / Real.sqrt (x + 8)) = 3 ∧ x = 8 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1145_114504


namespace NUMINAMATH_CALUDE_special_line_equation_l1145_114510

/-- A line passing through point (1,4) with the sum of its x and y intercepts equal to zero -/
structure SpecialLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through (1,4) -/
  passes_through_point : slope * 1 + y_intercept = 4
  /-- The sum of x and y intercepts is zero -/
  sum_of_intercepts_zero : (-y_intercept / slope) + y_intercept = 0

/-- The equation of a SpecialLine is either 4x - y = 0 or x - y + 3 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (l.slope = 4 ∧ l.y_intercept = 0) ∨ (l.slope = 1 ∧ l.y_intercept = 3) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l1145_114510


namespace NUMINAMATH_CALUDE_train_acceleration_time_l1145_114552

/-- Proves that a train starting from rest, accelerating uniformly at 3 m/s², 
    and traveling a distance of 27 m takes sqrt(18) seconds. -/
theorem train_acceleration_time : ∀ (s a : ℝ),
  s = 27 →  -- distance traveled
  a = 3 →   -- acceleration rate
  ∃ t : ℝ,
    s = (1/2) * a * t^2 ∧  -- kinematic equation for uniform acceleration from rest
    t = Real.sqrt 18 := by
  sorry

end NUMINAMATH_CALUDE_train_acceleration_time_l1145_114552


namespace NUMINAMATH_CALUDE_tournament_teams_count_l1145_114515

/-- Represents a football tournament with the given conditions -/
structure FootballTournament where
  n : ℕ  -- number of teams
  winner_points : ℕ
  last_place_points : ℕ
  winner_points_eq : winner_points = 26
  last_place_points_eq : last_place_points = 20

/-- Theorem stating that under the given conditions, the number of teams must be 12 -/
theorem tournament_teams_count (t : FootballTournament) : t.n = 12 := by
  sorry

end NUMINAMATH_CALUDE_tournament_teams_count_l1145_114515


namespace NUMINAMATH_CALUDE_bookshop_inventory_l1145_114537

theorem bookshop_inventory (books_sold : ℕ) (percentage_sold : ℚ) (initial_stock : ℕ) : 
  books_sold = 280 → percentage_sold = 2/5 → initial_stock * percentage_sold = books_sold → 
  initial_stock = 700 := by
sorry

end NUMINAMATH_CALUDE_bookshop_inventory_l1145_114537


namespace NUMINAMATH_CALUDE_unemployment_rate_calculation_l1145_114521

theorem unemployment_rate_calculation (previous_employment_rate previous_unemployment_rate : ℝ)
  (h1 : previous_employment_rate + previous_unemployment_rate = 100)
  (h2 : previous_employment_rate > 0)
  (h3 : previous_unemployment_rate > 0) :
  let new_employment_rate := 0.85 * previous_employment_rate
  let new_unemployment_rate := 1.1 * previous_unemployment_rate
  new_unemployment_rate = 66 :=
by
  sorry

#check unemployment_rate_calculation

end NUMINAMATH_CALUDE_unemployment_rate_calculation_l1145_114521


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1145_114554

-- Define the vectors
def a (x : ℝ) : Fin 2 → ℝ := ![x, 1]
def b (y : ℝ) : Fin 2 → ℝ := ![1, y]
def c : Fin 2 → ℝ := ![2, -4]

-- Define the conditions
def perpendicular (v w : Fin 2 → ℝ) : Prop := 
  (v 0) * (w 0) + (v 1) * (w 1) = 0

def parallel (v w : Fin 2 → ℝ) : Prop := 
  ∃ (k : ℝ), v = fun i ↦ k * (w i)

-- Theorem statement
theorem vector_sum_magnitude (x y : ℝ) 
  (h1 : perpendicular (a x) c) 
  (h2 : parallel (b y) c) : 
  Real.sqrt ((a x 0 + b y 0)^2 + (a x 1 + b y 1)^2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1145_114554


namespace NUMINAMATH_CALUDE_point_coordinates_l1145_114576

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance of a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance of a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (p : Point) 
  (h1 : SecondQuadrant p) 
  (h2 : DistanceToXAxis p = 3) 
  (h3 : DistanceToYAxis p = 1) : 
  p = Point.mk (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1145_114576


namespace NUMINAMATH_CALUDE_two_distinct_prime_factors_l1145_114516

def append_threes (n : ℕ) : ℕ :=
  12320 * 4^(10*n + 1) + (4^(10*n + 1) - 1) / 3

theorem two_distinct_prime_factors (n : ℕ) : 
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ 
   append_threes n = p * q) ↔ n = 0 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_prime_factors_l1145_114516


namespace NUMINAMATH_CALUDE_most_suitable_sampling_method_l1145_114555

-- Define the population structure
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

-- Define the sampling method
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | RemoveOneElderlyThenStratified

-- Define the suitability of a sampling method
def isMostSuitable (pop : Population) (sampleSize : ℕ) (method : SamplingMethod) : Prop :=
  method = SamplingMethod.RemoveOneElderlyThenStratified

-- Theorem statement
theorem most_suitable_sampling_method 
  (pop : Population)
  (h1 : pop.elderly = 28)
  (h2 : pop.middleAged = 54)
  (h3 : pop.young = 81)
  (sampleSize : ℕ)
  (h4 : sampleSize = 36) :
  isMostSuitable pop sampleSize SamplingMethod.RemoveOneElderlyThenStratified :=
by
  sorry

end NUMINAMATH_CALUDE_most_suitable_sampling_method_l1145_114555


namespace NUMINAMATH_CALUDE_octagon_diagonals_l1145_114598

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l1145_114598


namespace NUMINAMATH_CALUDE_hari_well_digging_time_l1145_114545

theorem hari_well_digging_time 
  (jake_time : ℝ) 
  (paul_time : ℝ) 
  (combined_time : ℝ) 
  (h : jake_time = 16)
  (i : paul_time = 24)
  (j : combined_time = 8)
  : ∃ (hari_time : ℝ), 
    1 / jake_time + 1 / paul_time + 1 / hari_time = 1 / combined_time ∧ 
    hari_time = 48 := by
  sorry

end NUMINAMATH_CALUDE_hari_well_digging_time_l1145_114545


namespace NUMINAMATH_CALUDE_special_trapezoid_base_ratio_l1145_114583

/-- A trapezoid with an inscribed and circumscribed circle, and one angle of 60 degrees -/
structure SpecialTrapezoid where
  /-- The trapezoid has an inscribed circle -/
  has_inscribed_circle : Bool
  /-- The trapezoid has a circumscribed circle -/
  has_circumscribed_circle : Bool
  /-- One angle of the trapezoid is 60 degrees -/
  has_60_degree_angle : Bool
  /-- The length of the longer base of the trapezoid -/
  longer_base : ℝ
  /-- The length of the shorter base of the trapezoid -/
  shorter_base : ℝ

/-- The ratio of the bases of a special trapezoid is 3:1 -/
theorem special_trapezoid_base_ratio (t : SpecialTrapezoid) :
  t.has_inscribed_circle ∧ t.has_circumscribed_circle ∧ t.has_60_degree_angle →
  t.longer_base / t.shorter_base = 3 := by
  sorry

end NUMINAMATH_CALUDE_special_trapezoid_base_ratio_l1145_114583


namespace NUMINAMATH_CALUDE_inverse_proportionality_l1145_114579

/-- Two real numbers are inversely proportional if their product is constant. -/
theorem inverse_proportionality (x y k : ℝ) (h : x * y = k) :
  ∃ (c : ℝ), ∀ (x' y' : ℝ), x' * y' = k → y' = c / x' :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportionality_l1145_114579


namespace NUMINAMATH_CALUDE_weight_ratio_after_loss_l1145_114553

def jakes_current_weight : ℝ := 152
def combined_weight : ℝ := 212
def weight_loss : ℝ := 32

def sisters_weight : ℝ := combined_weight - jakes_current_weight
def jakes_new_weight : ℝ := jakes_current_weight - weight_loss

theorem weight_ratio_after_loss : 
  jakes_new_weight / sisters_weight = 2 := by sorry

end NUMINAMATH_CALUDE_weight_ratio_after_loss_l1145_114553


namespace NUMINAMATH_CALUDE_smallest_whole_number_gt_100_odd_factors_l1145_114573

theorem smallest_whole_number_gt_100_odd_factors : ∀ n : ℕ, n > 100 → (∃ k : ℕ, n = k^2) → ∀ m : ℕ, m > 100 → (∃ j : ℕ, m = j^2) → n ≤ m → n = 121 := by
  sorry

end NUMINAMATH_CALUDE_smallest_whole_number_gt_100_odd_factors_l1145_114573


namespace NUMINAMATH_CALUDE_total_protein_consumed_l1145_114567

-- Define the protein content for each food item
def collagen_protein_per_2_scoops : ℚ := 18
def protein_powder_per_scoop : ℚ := 21
def steak_protein : ℚ := 56
def greek_yogurt_protein : ℚ := 15
def almond_protein_per_quarter_cup : ℚ := 6

-- Define the consumption quantities
def collagen_scoops : ℚ := 1
def protein_powder_scoops : ℚ := 2
def steak_quantity : ℚ := 1
def greek_yogurt_servings : ℚ := 1
def almond_cups : ℚ := 1/2

-- Theorem statement
theorem total_protein_consumed :
  (collagen_scoops / 2 * collagen_protein_per_2_scoops) +
  (protein_powder_scoops * protein_powder_per_scoop) +
  (steak_quantity * steak_protein) +
  (greek_yogurt_servings * greek_yogurt_protein) +
  (almond_cups / (1/4) * almond_protein_per_quarter_cup) = 134 := by
  sorry

end NUMINAMATH_CALUDE_total_protein_consumed_l1145_114567


namespace NUMINAMATH_CALUDE_value_of_2a_minus_3b_l1145_114530

-- Define the functions f, g, and h
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 6
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem value_of_2a_minus_3b (a b : ℝ) :
  (∀ x, h a b x = x - 9) →
  2 * a - 3 * b = 22 := by
  sorry

end NUMINAMATH_CALUDE_value_of_2a_minus_3b_l1145_114530


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l1145_114595

theorem square_area_equal_perimeter_triangle (s : ℝ) :
  let triangle_perimeter := 5.5 + 5.5 + 7
  let square_side := triangle_perimeter / 4
  s = square_side → s^2 = 20.25 := by
sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l1145_114595


namespace NUMINAMATH_CALUDE_v_2002_equals_5_l1145_114560

def g : ℕ → ℕ 
  | 1 => 5
  | 2 => 3
  | 3 => 6
  | 4 => 2
  | 5 => 1
  | 6 => 7
  | 7 => 4
  | _ => 0  -- Default case for inputs not in the table

def v : ℕ → ℕ
  | 0 => 5
  | n + 1 => g (v n)

theorem v_2002_equals_5 : v 2002 = 5 := by
  sorry

end NUMINAMATH_CALUDE_v_2002_equals_5_l1145_114560


namespace NUMINAMATH_CALUDE_probability_at_least_two_A_plus_specific_l1145_114518

def probability_at_least_two_A_plus (p_physics : ℚ) (p_chemistry : ℚ) (p_politics : ℚ) : ℚ :=
  let p_not_physics := 1 - p_physics
  let p_not_chemistry := 1 - p_chemistry
  let p_not_politics := 1 - p_politics
  p_physics * p_chemistry * p_not_politics +
  p_physics * p_not_chemistry * p_politics +
  p_not_physics * p_chemistry * p_politics +
  p_physics * p_chemistry * p_politics

theorem probability_at_least_two_A_plus_specific :
  probability_at_least_two_A_plus (7/8) (3/4) (5/12) = 151/192 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_two_A_plus_specific_l1145_114518


namespace NUMINAMATH_CALUDE_expression_equals_eight_l1145_114533

theorem expression_equals_eight (a : ℝ) (h : a = 2) : 
  (a^3 + (3*a)^3) / (a^2 - a*(3*a) + (3*a)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_eight_l1145_114533


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l1145_114589

theorem geometric_series_first_term (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a^2 / (1 - r^2) = 80) : a = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l1145_114589


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1145_114549

theorem interest_rate_calculation (P t : ℝ) (diff : ℝ) : 
  P = 10000 → 
  t = 2 → 
  diff = 49 → 
  P * (1 + 7/100)^t - P - (P * 7 * t / 100) = diff :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1145_114549


namespace NUMINAMATH_CALUDE_favorite_subject_problem_l1145_114562

theorem favorite_subject_problem (total_students : ℕ) 
  (math_fraction : ℚ) (english_fraction : ℚ) (science_fraction : ℚ)
  (h_total : total_students = 30)
  (h_math : math_fraction = 1 / 5)
  (h_english : english_fraction = 1 / 3)
  (h_science : science_fraction = 1 / 7) :
  total_students - 
  (↑total_students * math_fraction).floor - 
  (↑total_students * english_fraction).floor - 
  ((↑total_students - (↑total_students * math_fraction).floor - (↑total_students * english_fraction).floor) * science_fraction).floor = 12 := by
sorry

end NUMINAMATH_CALUDE_favorite_subject_problem_l1145_114562


namespace NUMINAMATH_CALUDE_alice_flour_measurement_l1145_114544

/-- The number of times Alice needs to fill her measuring cup to get the required amount of flour -/
def number_of_fills (total_flour : ℚ) (cup_capacity : ℚ) : ℚ :=
  total_flour / cup_capacity

/-- Theorem: Alice needs to fill her ⅓ cup measuring cup 10 times to get 3⅓ cups of flour -/
theorem alice_flour_measurement :
  number_of_fills (3 + 1/3) (1/3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_alice_flour_measurement_l1145_114544


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_9_l1145_114599

/-- Represents a four-digit number in the form 5BB3 where B is a single digit -/
def fourDigitNumber (B : Nat) : Nat :=
  5000 + 100 * B + 10 * B + 3

/-- Checks if a number is divisible by 9 -/
def isDivisibleBy9 (n : Nat) : Prop :=
  n % 9 = 0

/-- B is a single digit -/
def isSingleDigit (B : Nat) : Prop :=
  B ≥ 0 ∧ B ≤ 9

theorem four_digit_divisible_by_9 :
  ∃ B : Nat, isSingleDigit B ∧ isDivisibleBy9 (fourDigitNumber B) → B = 5 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_9_l1145_114599


namespace NUMINAMATH_CALUDE_translation_result_l1145_114501

def initial_point : ℝ × ℝ := (-4, 3)
def translation : ℝ × ℝ := (-2, -2)

def translate (p : ℝ × ℝ) (t : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + t.1, p.2 + t.2)

theorem translation_result :
  translate initial_point translation = (-6, 1) := by sorry

end NUMINAMATH_CALUDE_translation_result_l1145_114501


namespace NUMINAMATH_CALUDE_sphere_volume_coefficient_l1145_114506

theorem sphere_volume_coefficient (cube_side : ℝ) (L : ℝ) : 
  cube_side = 3 → 
  (4 * π * (((6 * cube_side^2) / (4 * π))^(1/2)))^2 = 6 * cube_side^2 →
  (4/3) * π * (((6 * cube_side^2) / (4 * π))^(3/2)) = L * Real.sqrt 15 / Real.sqrt π →
  L = 84 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_coefficient_l1145_114506


namespace NUMINAMATH_CALUDE_pencils_bought_with_promotion_l1145_114561

/-- Represents the number of pencils Petya's mom gave him money for -/
def pencils_mom_paid_for : ℕ := 49

/-- Represents the number of additional pencils Petya could buy with the promotion -/
def additional_pencils : ℕ := 12

/-- Represents the total number of pencils Petya could buy with the promotion -/
def total_pencils_bought : ℕ := pencils_mom_paid_for + additional_pencils

theorem pencils_bought_with_promotion :
  pencils_mom_paid_for = 49 ∧ 
  additional_pencils = 12 ∧
  total_pencils_bought = pencils_mom_paid_for + additional_pencils :=
by sorry

end NUMINAMATH_CALUDE_pencils_bought_with_promotion_l1145_114561


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l1145_114548

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l1145_114548


namespace NUMINAMATH_CALUDE_amount_after_two_years_l1145_114522

-- Define the initial amount
def initial_amount : ℚ := 64000

-- Define the annual increase rate
def annual_rate : ℚ := 1 / 8

-- Define the time period in years
def years : ℕ := 2

-- Define the function to calculate the amount after n years
def amount_after_years (initial : ℚ) (rate : ℚ) (n : ℕ) : ℚ :=
  initial * (1 + rate) ^ n

-- Theorem statement
theorem amount_after_two_years :
  amount_after_years initial_amount annual_rate years = 81000 := by
  sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l1145_114522


namespace NUMINAMATH_CALUDE_shopping_mall_uses_systematic_sampling_l1145_114525

/-- Represents a sampling method with given characteristics -/
structure SamplingMethod where
  initialSelection : Bool  -- True if initial selection is random
  fixedInterval : Bool     -- True if subsequent selections are at fixed intervals
  equalGroups : Bool       -- True if population is divided into equal-sized groups

/-- Definition of systematic sampling method -/
def isSystematicSampling (method : SamplingMethod) : Prop :=
  method.initialSelection ∧ method.fixedInterval ∧ method.equalGroups

/-- The sampling method used by the shopping mall -/
def shoppingMallMethod : SamplingMethod :=
  { initialSelection := true,  -- Randomly select one stub
    fixedInterval := true,     -- Sequentially take stubs at fixed intervals (every 50)
    equalGroups := true }      -- Each group has 50 invoice stubs

/-- Theorem stating that the shopping mall's method is systematic sampling -/
theorem shopping_mall_uses_systematic_sampling :
  isSystematicSampling shoppingMallMethod := by
  sorry


end NUMINAMATH_CALUDE_shopping_mall_uses_systematic_sampling_l1145_114525


namespace NUMINAMATH_CALUDE_complex_modulus_proof_l1145_114528

theorem complex_modulus_proof : 
  let z : ℂ := Complex.mk (3/4) (-5/6)
  ‖z‖ = Real.sqrt 127 / 12 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_proof_l1145_114528


namespace NUMINAMATH_CALUDE_rick_irons_31_clothes_l1145_114540

/-- Calculates the total number of clothes ironed by Rick -/
def totalClothesIroned (shirtsPerHour dressShirtsHours pantsPerHour dressPantsHours jacketsPerHour jacketsHours : ℕ) : ℕ :=
  shirtsPerHour * dressShirtsHours + pantsPerHour * dressPantsHours + jacketsPerHour * jacketsHours

/-- Proves that Rick irons 31 pieces of clothing given the conditions -/
theorem rick_irons_31_clothes :
  totalClothesIroned 4 3 3 5 2 2 = 31 := by
  sorry

#eval totalClothesIroned 4 3 3 5 2 2

end NUMINAMATH_CALUDE_rick_irons_31_clothes_l1145_114540
