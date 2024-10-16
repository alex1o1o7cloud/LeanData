import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_l433_43318

theorem inequality_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -1) :
  (x / (x + 1) + (x - 3) / (3 * x) ≥ 4) ↔ 
  (x > -1.5 ∧ x < -1) ∨ (x > -0.25) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l433_43318


namespace NUMINAMATH_CALUDE_calculate_overall_profit_specific_profit_l433_43300

/-- Calculate the overall profit or loss from selling a refrigerator and a mobile phone -/
theorem calculate_overall_profit (refrigerator_cost mobile_cost : ℝ) 
  (refrigerator_loss_percent mobile_profit_percent : ℝ) : ℝ :=
  let refrigerator_loss := refrigerator_cost * (refrigerator_loss_percent / 100)
  let refrigerator_sell := refrigerator_cost - refrigerator_loss
  let mobile_profit := mobile_cost * (mobile_profit_percent / 100)
  let mobile_sell := mobile_cost + mobile_profit
  let total_cost := refrigerator_cost + mobile_cost
  let total_sell := refrigerator_sell + mobile_sell
  total_sell - total_cost

/-- Prove that the overall profit is 120 Rs given the specific conditions -/
theorem specific_profit : calculate_overall_profit 15000 8000 4 9 = 120 := by
  sorry

end NUMINAMATH_CALUDE_calculate_overall_profit_specific_profit_l433_43300


namespace NUMINAMATH_CALUDE_smallest_positive_d_l433_43353

theorem smallest_positive_d : ∃ d : ℝ, d > 0 ∧
  (5 * Real.sqrt 5)^2 + (d + 5)^2 = (5 * d)^2 ∧
  ∀ d' : ℝ, d' > 0 → (5 * Real.sqrt 5)^2 + (d' + 5)^2 = (5 * d')^2 → d ≤ d' :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_d_l433_43353


namespace NUMINAMATH_CALUDE_rational_triple_theorem_l433_43327

/-- The set of triples that satisfy the conditions -/
def valid_triples : Set (ℚ × ℚ × ℚ) :=
  {(1, 1, 1), (1, 2, 2), (2, 4, 4), (2, 3, 6), (3, 3, 3)}

/-- A predicate that checks if a triple of rationals satisfies the conditions -/
def satisfies_conditions (p q r : ℚ) : Prop :=
  p > 0 ∧ q > 0 ∧ r > 0 ∧
  (p + q + r).isInt ∧
  (1/p + 1/q + 1/r).isInt ∧
  (p * q * r).isInt

theorem rational_triple_theorem :
  ∀ p q r : ℚ, satisfies_conditions p q r ↔ (p, q, r) ∈ valid_triples :=
by sorry

end NUMINAMATH_CALUDE_rational_triple_theorem_l433_43327


namespace NUMINAMATH_CALUDE_units_digit_of_power_of_six_l433_43380

theorem units_digit_of_power_of_six (n : ℕ) : (6^n) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_of_six_l433_43380


namespace NUMINAMATH_CALUDE_value_of_a_l433_43313

/-- Converts paise to rupees -/
def paise_to_rupees (paise : ℚ) : ℚ := paise / 100

/-- The problem statement -/
theorem value_of_a (a : ℚ) (h : (0.5 / 100) * a = 75) : 
  paise_to_rupees a = 150 := by sorry

end NUMINAMATH_CALUDE_value_of_a_l433_43313


namespace NUMINAMATH_CALUDE_skater_speeds_l433_43315

theorem skater_speeds (V₁ V₂ : ℝ) (h1 : V₁ > 0) (h2 : V₂ > 0) 
  (h3 : (V₁ + V₂) / |V₁ - V₂| = 4) (h4 : V₁ = 6 ∨ V₂ = 6) :
  (V₁ = 10 ∧ V₂ = 6) ∨ (V₁ = 6 ∧ V₂ = 3.6) := by
  sorry

end NUMINAMATH_CALUDE_skater_speeds_l433_43315


namespace NUMINAMATH_CALUDE_sequence_product_l433_43303

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

/-- The main theorem -/
theorem sequence_product (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  (∀ n : ℕ, a n ≠ 0) →
  (2 * a 3 - a 7 ^ 2 + 2 * a n = 0) →
  b 7 = a 7 →
  b 6 * b 8 = 16 := by
sorry

end NUMINAMATH_CALUDE_sequence_product_l433_43303


namespace NUMINAMATH_CALUDE_polynomial_zeros_evaluation_l433_43386

theorem polynomial_zeros_evaluation (r s : ℝ) : 
  r^2 - 3*r + 1 = 0 → 
  s^2 - 3*s + 1 = 0 → 
  (1 : ℝ)^2 - 18*(1 : ℝ) + 1 = -16 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_zeros_evaluation_l433_43386


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l433_43383

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (x ^ (1/4) = 16 / (9 - x ^ (1/4))) ↔ (x = 4096 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l433_43383


namespace NUMINAMATH_CALUDE_no_special_numbers_l433_43326

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem no_special_numbers :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 27 ∧ n % 3 = 0 ∧ Even n :=
by sorry

end NUMINAMATH_CALUDE_no_special_numbers_l433_43326


namespace NUMINAMATH_CALUDE_part_one_part_two_l433_43338

-- Define the conditions P and Q
def P (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def Q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem part_one (x : ℝ) (h : P x 1 ∧ Q x) : 2 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) 
  (h : ∀ x, ¬(P x a) → ¬(Q x)) 
  (h_not_nec : ∃ x, ¬(P x a) ∧ Q x) : 
  1 < a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l433_43338


namespace NUMINAMATH_CALUDE_fraction_equality_l433_43393

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 2 * b) (h2 : b ≠ 0) :
  let x := a / b
  (2 * a + b) / (a + 2 * b) = (2 * x + 1) / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l433_43393


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l433_43311

theorem smallest_angle_in_triangle (x y z : ℝ) : 
  x + y + z = 180 →  -- Sum of angles in a triangle is 180°
  x + y = 45 →       -- Sum of two angles is 45°
  y = x - 5 →        -- One angle is 5° less than the other
  x > 0 ∧ y > 0 ∧ z > 0 →  -- All angles are positive
  min x (min y z) = 20 :=  -- The smallest angle is 20°
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l433_43311


namespace NUMINAMATH_CALUDE_double_average_l433_43330

theorem double_average (n : Nat) (original_avg : Nat) (h1 : n = 12) (h2 : original_avg = 36) :
  let total := n * original_avg
  let doubled_total := 2 * total
  let new_avg := doubled_total / n
  new_avg = 72 := by
sorry

end NUMINAMATH_CALUDE_double_average_l433_43330


namespace NUMINAMATH_CALUDE_locus_of_Q_l433_43362

-- Define the circle E
def circle_E (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

-- Define point F
def point_F : ℝ × ℝ := (1, 0)

-- Define a point P on the circle E
def point_P (x y : ℝ) : Prop := circle_E x y

-- Define point Q as the intersection of the perpendicular bisector of PF and the radius PE
def point_Q (x y : ℝ) (px py : ℝ) : Prop :=
  point_P px py ∧
  -- Q is on the perpendicular bisector of PF
  (x - (px + 1)/2) * (px - 1) + (y - py/2) * py = 0 ∧
  -- Q is on the radius PE
  (x + 1) * (px + 1) + y * py = 16

-- The locus equation
def locus_equation (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- The theorem statement
theorem locus_of_Q :
  ∀ x y px py : ℝ, point_Q x y px py → locus_equation x y :=
sorry

end NUMINAMATH_CALUDE_locus_of_Q_l433_43362


namespace NUMINAMATH_CALUDE_det_trig_matrix_zero_l433_43306

theorem det_trig_matrix_zero (a b : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![![1, Real.sin (a - b), Real.sin a],
                                       ![Real.sin (a - b), 1, Real.sin b],
                                       ![Real.sin a, Real.sin b, 1]]
  Matrix.det M = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_trig_matrix_zero_l433_43306


namespace NUMINAMATH_CALUDE_function_comparison_l433_43317

theorem function_comparison (a b : ℝ) (f g : ℝ → ℝ) 
  (h_diff_f : Differentiable ℝ f)
  (h_diff_g : Differentiable ℝ g)
  (h_deriv : ∀ x ∈ Set.Icc a b, deriv f x > deriv g x)
  (h_eq : f a = g a)
  (h_le : a ≤ b) :
  ∀ x ∈ Set.Icc a b, f x ≥ g x :=
sorry

end NUMINAMATH_CALUDE_function_comparison_l433_43317


namespace NUMINAMATH_CALUDE_jasmine_milk_purchase_jasmine_milk_purchase_holds_l433_43388

/-- Proves that Jasmine bought 2 gallons of milk given the problem conditions -/
theorem jasmine_milk_purchase : ℝ → Prop :=
  fun gallons_of_milk =>
    let coffee_pounds : ℝ := 4
    let coffee_price_per_pound : ℝ := 2.5
    let milk_price_per_gallon : ℝ := 3.5
    let total_cost : ℝ := 17
    coffee_pounds * coffee_price_per_pound + gallons_of_milk * milk_price_per_gallon = total_cost →
    gallons_of_milk = 2

/-- The theorem holds -/
theorem jasmine_milk_purchase_holds : jasmine_milk_purchase 2 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_milk_purchase_jasmine_milk_purchase_holds_l433_43388


namespace NUMINAMATH_CALUDE_quadratic_roots_arithmetic_progression_l433_43366

theorem quadratic_roots_arithmetic_progression 
  (a b c : ℝ) 
  (p₁ p₂ q₁ q₂ : ℝ) 
  (h₁ : a * p₁^2 + b * p₁ + c = 0)
  (h₂ : a * p₂^2 + b * p₂ + c = 0)
  (h₃ : c * q₁^2 + b * q₁ + a = 0)
  (h₄ : c * q₂^2 + b * q₂ + a = 0)
  (h₅ : p₁ ≠ p₂)
  (h₆ : q₁ ≠ q₂)
  (h₇ : ∃ d : ℝ, q₁ = p₁ + d ∧ p₂ = p₁ + 2*d ∧ q₂ = p₁ + 3*d) :
  a + c = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_arithmetic_progression_l433_43366


namespace NUMINAMATH_CALUDE_log_product_equality_l433_43371

theorem log_product_equality : (Real.log 2 / Real.log 3 + Real.log 5 / Real.log 3) * (Real.log 9 / Real.log 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equality_l433_43371


namespace NUMINAMATH_CALUDE_ivan_number_properties_l433_43384

def sum_of_digits (n : ℕ) : ℕ := sorry

def num_digits (n : ℕ) : ℕ := sorry

theorem ivan_number_properties (n : ℕ) (h : n > 0) :
  let x := (sum_of_digits n)^2
  (num_digits n ≤ 3 → x < 730) ∧
  (num_digits n = 4 → x < n) ∧
  (num_digits n ≥ 5 → x < n) ∧
  (∀ m : ℕ, m > 0 → (sum_of_digits x)^2 = m → (m = 1 ∨ m = 81)) :=
by sorry

#check ivan_number_properties

end NUMINAMATH_CALUDE_ivan_number_properties_l433_43384


namespace NUMINAMATH_CALUDE_transformed_area_l433_43352

-- Define the transformation matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 8, -2]

-- Define the original region T and its area
def area_T : ℝ := 12

-- Define the transformed region T' and its area
def area_T' : ℝ := |A.det| * area_T

-- Theorem statement
theorem transformed_area :
  area_T' = 456 :=
sorry

end NUMINAMATH_CALUDE_transformed_area_l433_43352


namespace NUMINAMATH_CALUDE_no_triple_with_three_coprime_roots_l433_43329

theorem no_triple_with_three_coprime_roots : ¬∃ (a b c x₁ x₂ x₃ : ℤ),
  (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
  (Int.gcd x₁ x₂ = 1 ∧ Int.gcd x₁ x₃ = 1 ∧ Int.gcd x₂ x₃ = 1) ∧
  (x₁^3 - a^2*x₁^2 + b^2*x₁ - a*b + 3*c = 0) ∧
  (x₂^3 - a^2*x₂^2 + b^2*x₂ - a*b + 3*c = 0) ∧
  (x₃^3 - a^2*x₃^2 + b^2*x₃ - a*b + 3*c = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_no_triple_with_three_coprime_roots_l433_43329


namespace NUMINAMATH_CALUDE_angle_properties_l433_43308

theorem angle_properties (θ : Real) (h1 : π/2 < θ ∧ θ < π) (h2 : Real.tan (2*θ) = -2*Real.sqrt 2) :
  (Real.tan θ = -Real.sqrt 2 / 2) ∧
  ((2 * (Real.cos (θ/2))^2 - Real.sin θ - Real.tan (5*π/4)) / (Real.sqrt 2 * Real.sin (θ + π/4)) = 3 + 2*Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_angle_properties_l433_43308


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_of_fourth_powers_l433_43322

theorem largest_power_of_two_dividing_difference_of_fourth_powers :
  ∃ k : ℕ, k = 5 ∧ 2^k = (Nat.gcd (15^4 - 9^4) (2^32)) :=
by sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_of_fourth_powers_l433_43322


namespace NUMINAMATH_CALUDE_no_root_greater_than_three_l433_43395

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the theorem
theorem no_root_greater_than_three (a b c : ℝ) :
  (quadratic a b c (-1) = -1) →
  (quadratic a b c 0 = 2) →
  (quadratic a b c 2 = 2) →
  (quadratic a b c 4 = -6) →
  ∀ x > 3, quadratic a b c x ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_root_greater_than_three_l433_43395


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l433_43319

theorem gcf_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l433_43319


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l433_43378

theorem least_sum_of_bases (c d : ℕ) (h1 : c > 0) (h2 : d > 0) 
  (h3 : 3 * c + 6 = 6 * d + 3) : 
  (∀ x y : ℕ, x > 0 → y > 0 → 3 * x + 6 = 6 * y + 3 → c + d ≤ x + y) ∧ c + d = 5 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l433_43378


namespace NUMINAMATH_CALUDE_chicken_cage_problem_l433_43325

theorem chicken_cage_problem :
  ∃ (chickens cages : ℕ),
    chickens = 25 ∧ cages = 6 ∧
    (4 * cages + 1 = chickens) ∧
    (5 * (cages - 1) = chickens) :=
by sorry

end NUMINAMATH_CALUDE_chicken_cage_problem_l433_43325


namespace NUMINAMATH_CALUDE_pie_weight_l433_43324

theorem pie_weight (total_weight : ℝ) (eaten_fraction : ℝ) (eaten_weight : ℝ) : 
  eaten_fraction = 1/6 →
  eaten_weight = 240 →
  total_weight = eaten_weight / eaten_fraction →
  (1 - eaten_fraction) * total_weight = 1200 :=
by
  sorry

end NUMINAMATH_CALUDE_pie_weight_l433_43324


namespace NUMINAMATH_CALUDE_parabola_midpoint_trajectory_and_intersection_l433_43387

/-- Parabola C -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Trajectory D -/
def trajectory_D (x y : ℝ) : Prop := y^2 = x

/-- Line l with slope 1 passing through (1, 0) -/
def line_l (x y : ℝ) : Prop := y = x - 1

/-- The focus of parabola C -/
def focus_C : ℝ × ℝ := (1, 0)

/-- The statement to prove -/
theorem parabola_midpoint_trajectory_and_intersection :
  (∀ x y : ℝ, parabola_C x y → ∃ x' y' : ℝ, trajectory_D x' y' ∧ y' = y / 2 ∧ x' = x) ∧
  (∃ A B : ℝ × ℝ,
    trajectory_D A.1 A.2 ∧ trajectory_D B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = 10) :=
sorry

end NUMINAMATH_CALUDE_parabola_midpoint_trajectory_and_intersection_l433_43387


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l433_43344

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 →
  (((3 / (x - 1)) - x - 1) / ((x^2 - 4*x + 4) / (x - 1))) = (2 + x) / (2 - x) ∧
  (((3 / (0 - 1)) - 0 - 1) / ((0^2 - 4*0 + 4) / (0 - 1))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l433_43344


namespace NUMINAMATH_CALUDE_age_ratio_proof_l433_43309

-- Define Rahul's and Deepak's ages
def rahul_future_age : ℕ := 26
def years_to_future : ℕ := 10
def deepak_current_age : ℕ := 12

-- Define the ratio we want to prove
def target_ratio : Rat := 4 / 3

-- Theorem statement
theorem age_ratio_proof :
  (rahul_future_age - years_to_future : ℚ) / deepak_current_age = target_ratio := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l433_43309


namespace NUMINAMATH_CALUDE_work_completion_time_l433_43333

/-- Given two workers a and b, where a is thrice as fast as b, proves that if a can complete
    a work alone in 40 days, then a and b together can complete the work in 30 days. -/
theorem work_completion_time
  (rate_a rate_b : ℝ)  -- Rates at which workers a and b work
  (h1 : rate_a = 3 * rate_b)  -- a is thrice as fast as b
  (h2 : rate_a * 40 = 1)  -- a alone completes the work in 40 days
  : (rate_a + rate_b) * 30 = 1 :=  -- a and b together complete the work in 30 days
by
  sorry


end NUMINAMATH_CALUDE_work_completion_time_l433_43333


namespace NUMINAMATH_CALUDE_object_is_cylinder_l433_43328

-- Define the properties of the object
structure GeometricObject where
  front_view : Type
  side_view : Type
  top_view : Type
  front_is_square : front_view = Square
  side_is_square : side_view = Square
  front_side_equal : front_view = side_view
  top_is_circle : top_view = Circle

-- Define the theorem
theorem object_is_cylinder (obj : GeometricObject) : obj = Cylinder := by
  sorry

end NUMINAMATH_CALUDE_object_is_cylinder_l433_43328


namespace NUMINAMATH_CALUDE_sum_of_series_l433_43350

theorem sum_of_series (a₁ : ℝ) (r : ℝ) (n : ℕ) (d : ℝ) :
  let geometric_sum := a₁ / (1 - r)
  let arithmetic_sum := n * (2 * a₁ + (n - 1) * d) / 2
  geometric_sum + arithmetic_sum = 115 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_series_l433_43350


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_implies_m_gt_one_m_gt_one_implies_point_in_first_quadrant_l433_43334

/-- A point P(x, y) is in the first quadrant if and only if x > 0 and y > 0 -/
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- The x-coordinate of point P is m - 1 -/
def x_coord (m : ℝ) : ℝ := m - 1

/-- The y-coordinate of point P is m + 2 -/
def y_coord (m : ℝ) : ℝ := m + 2

/-- Theorem: If point P(m-1, m+2) is in the first quadrant, then m > 1 -/
theorem point_in_first_quadrant_implies_m_gt_one (m : ℝ) :
  in_first_quadrant (x_coord m) (y_coord m) → m > 1 :=
by sorry

/-- Theorem: If m > 1, then point P(m-1, m+2) is in the first quadrant -/
theorem m_gt_one_implies_point_in_first_quadrant (m : ℝ) :
  m > 1 → in_first_quadrant (x_coord m) (y_coord m) :=
by sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_implies_m_gt_one_m_gt_one_implies_point_in_first_quadrant_l433_43334


namespace NUMINAMATH_CALUDE_projected_revenue_increase_l433_43377

theorem projected_revenue_increase 
  (last_year_revenue : ℝ) 
  (h1 : actual_revenue = 0.75 * last_year_revenue) 
  (h2 : actual_revenue = 0.60 * projected_revenue) 
  (projected_revenue := last_year_revenue * (1 + projected_increase / 100)) :
  projected_increase = 25 := by
sorry

end NUMINAMATH_CALUDE_projected_revenue_increase_l433_43377


namespace NUMINAMATH_CALUDE_distribute_5_4_l433_43332

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 cousins into 4 rooms -/
theorem distribute_5_4 : distribute 5 4 = 67 := by sorry

end NUMINAMATH_CALUDE_distribute_5_4_l433_43332


namespace NUMINAMATH_CALUDE_orchard_problem_l433_43323

theorem orchard_problem (total_trees : ℕ) (pure_fuji : ℕ) (pure_gala : ℕ) :
  (pure_fuji : ℚ) = 3 / 4 * total_trees →
  (pure_fuji : ℚ) + 1 / 10 * total_trees = 221 →
  pure_gala = 39 := by
  sorry

end NUMINAMATH_CALUDE_orchard_problem_l433_43323


namespace NUMINAMATH_CALUDE_no_real_solutions_cube_root_equation_l433_43365

theorem no_real_solutions_cube_root_equation :
  ¬∃ x : ℝ, (x ^ (1/3 : ℝ)) = 15 / (6 - x ^ (1/3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_cube_root_equation_l433_43365


namespace NUMINAMATH_CALUDE_solve_cost_price_l433_43373

def cost_price_problem (C : ℝ) : Prop :=
  let S := 1.05 * C
  let C_new := 0.95 * C
  let S_new := S - 4
  S_new = 1.1 * C_new ∧ C = 800

theorem solve_cost_price : ∃ C : ℝ, cost_price_problem C :=
  sorry

end NUMINAMATH_CALUDE_solve_cost_price_l433_43373


namespace NUMINAMATH_CALUDE_triangles_in_3x7_rectangle_l433_43354

/-- The number of small triangles created by cutting a rectangle --/
def num_triangles (length width : ℕ) : ℕ :=
  let total_squares := length * width
  let corner_squares := 4
  let cut_squares := total_squares - corner_squares
  let triangles_per_square := 4
  cut_squares * triangles_per_square

/-- Theorem stating the number of triangles for a 3x7 rectangle --/
theorem triangles_in_3x7_rectangle :
  num_triangles 3 7 = 68 := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_3x7_rectangle_l433_43354


namespace NUMINAMATH_CALUDE_visible_bird_legs_count_l433_43345

theorem visible_bird_legs_count :
  let crows : ℕ := 4
  let pigeons : ℕ := 3
  let flamingos : ℕ := 5
  let sparrows : ℕ := 8
  let crow_legs : ℕ := 2
  let pigeon_legs : ℕ := 2
  let flamingo_legs : ℕ := 3
  let sparrow_legs : ℕ := 2
  crows * crow_legs + pigeons * pigeon_legs + flamingos * flamingo_legs + sparrows * sparrow_legs = 45 :=
by sorry

end NUMINAMATH_CALUDE_visible_bird_legs_count_l433_43345


namespace NUMINAMATH_CALUDE_inverse_g_87_l433_43391

def g (x : ℝ) : ℝ := 3 * x^3 + 6

theorem inverse_g_87 : g⁻¹ 87 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_87_l433_43391


namespace NUMINAMATH_CALUDE_rocket_coaster_total_cars_l433_43316

/-- Represents a roller coaster with two types of cars -/
structure RollerCoaster where
  four_passenger_cars : ℕ
  six_passenger_cars : ℕ
  total_capacity : ℕ

/-- The Rocket Coaster satisfies the given conditions -/
def rocket_coaster : RollerCoaster :=
  { four_passenger_cars := 9,
    six_passenger_cars := 6,
    total_capacity := 72 }

/-- The total number of cars on the Rocket Coaster -/
def total_cars (rc : RollerCoaster) : ℕ :=
  rc.four_passenger_cars + rc.six_passenger_cars

/-- Theorem stating that the total number of cars on the Rocket Coaster is 15 -/
theorem rocket_coaster_total_cars :
  total_cars rocket_coaster = 15 ∧
  4 * rocket_coaster.four_passenger_cars + 6 * rocket_coaster.six_passenger_cars = rocket_coaster.total_capacity :=
by sorry

#eval total_cars rocket_coaster -- Should output 15

end NUMINAMATH_CALUDE_rocket_coaster_total_cars_l433_43316


namespace NUMINAMATH_CALUDE_right_triangle_with_reversed_digits_l433_43320

theorem right_triangle_with_reversed_digits : ∀ a b c : ℕ,
  a = 56 ∧ c = 65 ∧ 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ 10 ≤ c ∧ c < 100 →
  a^2 + b^2 = c^2 →
  b = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_with_reversed_digits_l433_43320


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_for_purely_imaginary_l433_43304

-- Define a complex number
def complex (a b : ℝ) := a + b * Complex.I

-- Define what it means for a complex number to be purely imaginary
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem necessary_not_sufficient_condition_for_purely_imaginary (a b : ℝ) :
  (isPurelyImaginary (complex a b) → a = 0) ∧
  ¬(a = 0 → isPurelyImaginary (complex a b)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_for_purely_imaginary_l433_43304


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l433_43368

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  x^2 - 72*y^2 - 12*x + 144 = 0

/-- Definition of an ellipse in standard form -/
def is_ellipse (a b h k : ℝ) (x y : ℝ) : Prop :=
  ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1

/-- Theorem stating that the given equation represents an ellipse -/
theorem conic_is_ellipse :
  ∃ a b h k : ℝ, a > 0 ∧ b > 0 ∧
  ∀ x y : ℝ, conic_equation x y ↔ is_ellipse a b h k x y :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l433_43368


namespace NUMINAMATH_CALUDE_system_solution_l433_43331

theorem system_solution (x y b : ℝ) : 
  (5 * x + y = b) → 
  (3 * x + 4 * y = 3 * b) → 
  (x = 3) → 
  (b = 60) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l433_43331


namespace NUMINAMATH_CALUDE_rainbow_population_proof_l433_43392

/-- The number of settlements in Solar Valley -/
def num_settlements : ℕ := 10

/-- The population of Zhovtnevo -/
def zhovtnevo_population : ℕ := 1000

/-- The amount by which Zhovtnevo's population exceeds the average -/
def excess_population : ℕ := 90

/-- The population of Rainbow settlement -/
def rainbow_population : ℕ := 900

theorem rainbow_population_proof :
  rainbow_population = 
    (num_settlements * zhovtnevo_population - num_settlements * excess_population) / (num_settlements - 1) :=
by sorry

end NUMINAMATH_CALUDE_rainbow_population_proof_l433_43392


namespace NUMINAMATH_CALUDE_magic_square_base_5_l433_43341

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a b c d e f g h i : ℕ)

/-- Converts a number from base 5 to base 10 -/
def toBase10 (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 4
  | 10 => 5
  | 11 => 6
  | 12 => 7
  | 13 => 8
  | 14 => 9
  | _ => 0  -- For simplicity, we only define the conversion for numbers used in the square

/-- Checks if the given square is magic in base 5 -/
def isMagicSquare (s : MagicSquare) : Prop :=
  let a' := toBase10 s.a
  let b' := toBase10 s.b
  let c' := toBase10 s.c
  let d' := toBase10 s.d
  let e' := toBase10 s.e
  let f' := toBase10 s.f
  let g' := toBase10 s.g
  let h' := toBase10 s.h
  let i' := toBase10 s.i
  -- Row sums are equal
  (a' + b' + c' = d' + e' + f') ∧
  (d' + e' + f' = g' + h' + i') ∧
  -- Column sums are equal
  (a' + d' + g' = b' + e' + h') ∧
  (b' + e' + h' = c' + f' + i') ∧
  -- Diagonal sums are equal
  (a' + e' + i' = c' + e' + g')

theorem magic_square_base_5 : 
  isMagicSquare ⟨13, 1, 11, 3, 10, 12, 4, 14, 2⟩ := by
  sorry


end NUMINAMATH_CALUDE_magic_square_base_5_l433_43341


namespace NUMINAMATH_CALUDE_compare_fractions_l433_43358

theorem compare_fractions : -3/2 > -(5/3) := by sorry

end NUMINAMATH_CALUDE_compare_fractions_l433_43358


namespace NUMINAMATH_CALUDE_relationship_abc_l433_43337

theorem relationship_abc (a b c : ℝ) 
  (ha : a = (2/3)^(-(1/3 : ℝ))) 
  (hb : b = (5/3)^(-(2/3 : ℝ))) 
  (hc : c = (3/2)^(2/3 : ℝ)) : 
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l433_43337


namespace NUMINAMATH_CALUDE_philip_monthly_mileage_l433_43355

/-- Represents Philip's driving routine and calculates the total monthly mileage -/
def philipDrivingMileage : ℕ :=
  let schoolRoundTrip : ℕ := 5 /- 2.5 * 2, rounded to nearest integer -/
  let workOneWay : ℕ := 8
  let marketRoundTrip : ℕ := 2
  let gymRoundTrip : ℕ := 4
  let friendRoundTrip : ℕ := 6
  let weekdayMileage : ℕ := (schoolRoundTrip * 2 + workOneWay * 2) * 5
  let saturdayMileage : ℕ := marketRoundTrip + gymRoundTrip + friendRoundTrip
  let weeklyMileage : ℕ := weekdayMileage + saturdayMileage
  let weeksInMonth : ℕ := 4
  weeklyMileage * weeksInMonth

/-- Theorem stating that Philip's total monthly mileage is 468 miles -/
theorem philip_monthly_mileage : philipDrivingMileage = 468 := by
  sorry

end NUMINAMATH_CALUDE_philip_monthly_mileage_l433_43355


namespace NUMINAMATH_CALUDE_perfect_square_a_value_of_a_l433_43398

theorem perfect_square_a : ∃ n : ℕ, 1995^2 + 1995^2 * 1996^2 + 1996^2 = n^2 :=
by
  use 3982021
  sorry

theorem value_of_a : 1995^2 + 1995^2 * 1996^2 + 1996^2 = 3982021^2 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_a_value_of_a_l433_43398


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l433_43376

theorem trigonometric_equation_solution (x : ℝ) :
  (4 * Real.sin x ^ 4 + Real.cos (4 * x) = 1 + 12 * Real.cos x ^ 4) ↔
  (∃ k : ℤ, x = π / 3 * (3 * ↑k + 1) ∨ x = π / 3 * (3 * ↑k - 1)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l433_43376


namespace NUMINAMATH_CALUDE_tonys_fever_l433_43361

/-- Given Tony's normal temperature, temperature increase due to sickness, and fever threshold,
    calculate how many degrees above the fever threshold Tony's temperature is. -/
theorem tonys_fever (normal_temp : ℝ) (temp_increase : ℝ) (fever_threshold : ℝ) : 
  normal_temp = 95 → temp_increase = 10 → fever_threshold = 100 →
  normal_temp + temp_increase - fever_threshold = 5 :=
by sorry

end NUMINAMATH_CALUDE_tonys_fever_l433_43361


namespace NUMINAMATH_CALUDE_zeros_of_f_l433_43302

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 9

-- Theorem stating that the zeros of f(x) are ±3
theorem zeros_of_f :
  ∀ x : ℝ, f x = 0 ↔ x = 3 ∨ x = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_zeros_of_f_l433_43302


namespace NUMINAMATH_CALUDE_inequality_solution_range_l433_43389

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 2 ≥ 0) → 0 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l433_43389


namespace NUMINAMATH_CALUDE_angle_bisector_length_l433_43364

/-- Given a triangle ABC, this theorem states that the length of the angle bisector
    from vertex C to the opposite side AB can be calculated using the formula:
    l₃ = (2ab)/(a+b) * cos(C/2), where a and b are the lengths of sides BC and AC
    respectively, and C is the angle at vertex C. -/
theorem angle_bisector_length (a b C l₃ : ℝ) :
  (a > 0) → (b > 0) → (C > 0) → (C < π) →
  l₃ = (2 * a * b) / (a + b) * Real.cos (C / 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_bisector_length_l433_43364


namespace NUMINAMATH_CALUDE_square_difference_fourth_power_l433_43369

theorem square_difference_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fourth_power_l433_43369


namespace NUMINAMATH_CALUDE_math_class_students_count_l433_43360

theorem math_class_students_count :
  ∃! n : ℕ, 0 < n ∧ n < 50 ∧ n % 8 = 6 ∧ n % 5 = 1 ∧ n = 46 := by
  sorry

end NUMINAMATH_CALUDE_math_class_students_count_l433_43360


namespace NUMINAMATH_CALUDE_projection_ratio_l433_43385

def projection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  !![9/41, -20/41; -20/41, 16/41]

theorem projection_ratio :
  ∀ (a b : ℚ),
  (a ≠ 0) →
  (projection_matrix.vecMul ![a, b] = ![a, b]) →
  b / a = 8 / 5 := by
sorry

end NUMINAMATH_CALUDE_projection_ratio_l433_43385


namespace NUMINAMATH_CALUDE_park_tree_density_l433_43370

/-- Given a rectangular park with length, width, and number of trees, 
    calculate the area occupied by each tree. -/
def area_per_tree (length width num_trees : ℕ) : ℚ :=
  (length * width : ℚ) / num_trees

/-- Theorem stating that in a park of 1000 feet long and 2000 feet wide, 
    with 100,000 trees, each tree occupies 20 square feet. -/
theorem park_tree_density :
  area_per_tree 1000 2000 100000 = 20 := by
  sorry

#eval area_per_tree 1000 2000 100000

end NUMINAMATH_CALUDE_park_tree_density_l433_43370


namespace NUMINAMATH_CALUDE_no_snow_probability_l433_43343

theorem no_snow_probability (p : ℚ) (h : p = 2/3) : (1 - p)^5 = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l433_43343


namespace NUMINAMATH_CALUDE_sector_circumference_l433_43339

/-- The circumference of a sector with central angle 60° and radius 15 cm is 5(6 + π) cm. -/
theorem sector_circumference :
  let θ : ℝ := 60  -- Central angle in degrees
  let r : ℝ := 15  -- Radius in cm
  let arc_length : ℝ := (θ / 360) * (2 * π * r)
  let circumference : ℝ := arc_length + 2 * r
  circumference = 5 * (6 + π) := by sorry

end NUMINAMATH_CALUDE_sector_circumference_l433_43339


namespace NUMINAMATH_CALUDE_parabola_perpendicular_range_l433_43363

/-- Given point A(0,2) and two points B and C on the parabola y^2 = x + 4 such that AB ⟂ BC,
    the y-coordinate of point C satisfies y ≤ 0 or y ≥ 4. -/
theorem parabola_perpendicular_range (B C : ℝ × ℝ) : 
  let A : ℝ × ℝ := (0, 2)
  let on_parabola (p : ℝ × ℝ) := p.2^2 = p.1 + 4
  let perpendicular (p q r : ℝ × ℝ) := 
    (q.2 - p.2) * (r.2 - q.2) = -(q.1 - p.1) * (r.1 - q.1)
  on_parabola B ∧ on_parabola C ∧ perpendicular A B C →
  C.2 ≤ 0 ∨ C.2 ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_perpendicular_range_l433_43363


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l433_43321

/-- Given a circle with equation x^2 - 8x + y^2 - 4y = -4, prove its center and radius -/
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (4, 2) ∧
    radius = 4 ∧
    ∀ (x y : ℝ), x^2 - 8*x + y^2 - 4*y = -4 ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l433_43321


namespace NUMINAMATH_CALUDE_charity_fundraiser_revenue_l433_43381

theorem charity_fundraiser_revenue (total_tickets : ℕ) (total_revenue : ℕ) 
  (h_total_tickets : total_tickets = 170)
  (h_total_revenue : total_revenue = 2917) :
  ∃ (full_price : ℕ) (full_count : ℕ) (quarter_count : ℕ),
    full_count + quarter_count = total_tickets ∧
    full_count * full_price + quarter_count * (full_price / 4) = total_revenue ∧
    full_count * full_price = 1748 :=
by sorry

end NUMINAMATH_CALUDE_charity_fundraiser_revenue_l433_43381


namespace NUMINAMATH_CALUDE_quadratic_inequalities_solution_sets_l433_43396

theorem quadratic_inequalities_solution_sets :
  (∀ x : ℝ, -3 * x^2 + x + 1 > 0 ↔ x ∈ Set.Ioo ((1 - Real.sqrt 13) / 6) ((1 + Real.sqrt 13) / 6)) ∧
  (∀ x : ℝ, x^2 - 2*x + 1 ≤ 0 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_solution_sets_l433_43396


namespace NUMINAMATH_CALUDE_total_green_peaches_is_fourteen_l433_43375

/-- The number of baskets containing peaches. -/
def num_baskets : ℕ := 7

/-- The number of green peaches in each basket. -/
def green_peaches_per_basket : ℕ := 2

/-- The total number of green peaches in all baskets. -/
def total_green_peaches : ℕ := num_baskets * green_peaches_per_basket

theorem total_green_peaches_is_fourteen : total_green_peaches = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_green_peaches_is_fourteen_l433_43375


namespace NUMINAMATH_CALUDE_exists_z_satisfying_conditions_l433_43348

-- Define the complex function g
def g (z : ℂ) : ℂ := z^2 + 2*Complex.I*z + 2

-- State the theorem
theorem exists_z_satisfying_conditions : 
  ∃ z : ℂ, Complex.im z > 0 ∧ 
    (∃ a b : ℤ, g z = ↑a + ↑b * Complex.I ∧ 
      abs a ≤ 5 ∧ abs b ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_exists_z_satisfying_conditions_l433_43348


namespace NUMINAMATH_CALUDE_right_triangle_sets_l433_43372

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  is_pythagorean_triple 7 24 25 ∧
  is_pythagorean_triple 6 8 10 ∧
  is_pythagorean_triple 9 12 15 ∧
  ¬ is_pythagorean_triple 3 4 6 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l433_43372


namespace NUMINAMATH_CALUDE_complex_expression_value_l433_43335

theorem complex_expression_value : 
  let x : ℂ := Complex.exp (Complex.I * (2 * Real.pi / 9))
  (2 * x + x^3) * (2 * x^3 + x^9) * (2 * x^6 + x^18) * 
  (2 * x^2 + x^6) * (2 * x^5 + x^15) * (2 * x^7 + x^21) = 557 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_value_l433_43335


namespace NUMINAMATH_CALUDE_four_solutions_l433_43394

/-- The system of equations has exactly 4 distinct real solutions -/
theorem four_solutions (x y z w : ℝ) : 
  (x = z - w + x * z ∧
   y = w - x + y * w ∧
   z = x - y + x * z ∧
   w = y - z + y * w) →
  ∃! (s : Finset (ℝ × ℝ × ℝ × ℝ)), s.card = 4 ∧ ∀ (a b c d : ℝ), (a, b, c, d) ∈ s ↔
    (a = c - d + a * c ∧
     b = d - a + b * d ∧
     c = a - b + a * c ∧
     d = b - c + b * d) :=
by sorry

end NUMINAMATH_CALUDE_four_solutions_l433_43394


namespace NUMINAMATH_CALUDE_profit_achieved_l433_43347

/-- The minimum number of disks Maria needs to sell to make a profit of $120 -/
def disks_to_sell : ℕ := 219

/-- The cost of buying 5 disks -/
def buy_price : ℚ := 6

/-- The selling price of 4 disks -/
def sell_price : ℚ := 7

/-- The desired profit -/
def target_profit : ℚ := 120

theorem profit_achieved :
  let cost_per_disk : ℚ := buy_price / 5
  let revenue_per_disk : ℚ := sell_price / 4
  let profit_per_disk : ℚ := revenue_per_disk - cost_per_disk
  (disks_to_sell : ℚ) * profit_per_disk ≥ target_profit ∧
  ∀ n : ℕ, (n : ℚ) * profit_per_disk < target_profit → n < disks_to_sell :=
by sorry

end NUMINAMATH_CALUDE_profit_achieved_l433_43347


namespace NUMINAMATH_CALUDE_bus_capacity_problem_l433_43367

/-- Proves that given two buses with a capacity of 150 people each, where one bus is 70% full
    and the total number of people in both buses is 195, the percentage of capacity full
    for the other bus is 60%. -/
theorem bus_capacity_problem (bus_capacity : ℕ) (total_people : ℕ) (second_bus_percentage : ℚ) :
  bus_capacity = 150 →
  total_people = 195 →
  second_bus_percentage = 70/100 →
  ∃ (first_bus_percentage : ℚ),
    first_bus_percentage * bus_capacity + second_bus_percentage * bus_capacity = total_people ∧
    first_bus_percentage = 60/100 :=
by sorry

end NUMINAMATH_CALUDE_bus_capacity_problem_l433_43367


namespace NUMINAMATH_CALUDE_angle_range_theorem_l433_43359

theorem angle_range_theorem (θ : Real) 
  (h1 : 0 ≤ θ) (h2 : θ < 2 * Real.pi) 
  (h3 : Real.sin θ ^ 3 - Real.cos θ ^ 3 ≥ Real.cos θ - Real.sin θ) : 
  Real.pi / 4 ≤ θ ∧ θ ≤ 5 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_range_theorem_l433_43359


namespace NUMINAMATH_CALUDE_equality_equivalence_l433_43349

theorem equality_equivalence (a b c d : ℝ) : 
  (a - b)^2 + (c - d)^2 = 0 ↔ (a = b ∧ c = d) := by sorry

end NUMINAMATH_CALUDE_equality_equivalence_l433_43349


namespace NUMINAMATH_CALUDE_sin_over_two_minus_cos_max_value_l433_43307

theorem sin_over_two_minus_cos_max_value (x : ℝ) : 
  (Real.sin x) / (2 - Real.cos x) ≤ Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_over_two_minus_cos_max_value_l433_43307


namespace NUMINAMATH_CALUDE_inequality_solution_set_l433_43357

theorem inequality_solution_set (x : ℝ) :
  (3*x - 1) / (2 - x) ≥ 0 ↔ x ∈ {y : ℝ | 1/3 ≤ y ∧ y < 2} :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l433_43357


namespace NUMINAMATH_CALUDE_patricks_pencil_loss_percentage_l433_43382

/-- Calculates the overall loss percentage for Patrick's pencil sales -/
theorem patricks_pencil_loss_percentage : 
  let type_a_count : ℕ := 30
  let type_b_count : ℕ := 40
  let type_c_count : ℕ := 10
  let type_a_cost : ℚ := 1
  let type_b_cost : ℚ := 2
  let type_c_cost : ℚ := 3
  let type_a_discount : ℚ := 0.5
  let type_b_discount : ℚ := 1
  let type_c_discount : ℚ := 1.5
  let total_cost : ℚ := type_a_count * type_a_cost + type_b_count * type_b_cost + type_c_count * type_c_cost
  let total_revenue : ℚ := type_a_count * (type_a_cost - type_a_discount) + 
                           type_b_count * (type_b_cost - type_b_discount) + 
                           type_c_count * (type_c_cost - type_c_discount)
  let additional_loss : ℚ := type_a_count * (type_a_cost - type_a_discount)
  let total_loss : ℚ := total_cost - total_revenue + additional_loss
  let loss_percentage : ℚ := (total_loss / total_cost) * 100
  ∃ ε > 0, |loss_percentage - 60.71| < ε :=
by sorry

end NUMINAMATH_CALUDE_patricks_pencil_loss_percentage_l433_43382


namespace NUMINAMATH_CALUDE_most_likely_gender_combination_l433_43346

theorem most_likely_gender_combination (n : ℕ) (p : ℝ) : 
  n = 5 → p = 1/2 → 2 * (n.choose 3) * p^n = 5/8 := by sorry

end NUMINAMATH_CALUDE_most_likely_gender_combination_l433_43346


namespace NUMINAMATH_CALUDE_employed_females_percentage_l433_43310

theorem employed_females_percentage (total_population : ℝ) 
  (h1 : total_population > 0) 
  (employed_percentage : ℝ) 
  (h2 : employed_percentage = 72) 
  (employed_males_percentage : ℝ) 
  (h3 : employed_males_percentage = 36) : 
  (employed_percentage - employed_males_percentage) / employed_percentage * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l433_43310


namespace NUMINAMATH_CALUDE_marcus_points_l433_43356

/-- Proves that Marcus scored 28 points in the basketball game -/
theorem marcus_points (total_points : ℕ) (other_players : ℕ) (avg_points : ℕ) : 
  total_points = 63 → other_players = 5 → avg_points = 7 → 
  total_points - (other_players * avg_points) = 28 := by
  sorry

end NUMINAMATH_CALUDE_marcus_points_l433_43356


namespace NUMINAMATH_CALUDE_solve_equation_for_m_l433_43301

theorem solve_equation_for_m : ∃ m : ℝ, (m - 5)^3 = (1/27)⁻¹ ∧ m = 8 := by sorry

end NUMINAMATH_CALUDE_solve_equation_for_m_l433_43301


namespace NUMINAMATH_CALUDE_sugar_water_inequality_l433_43340

theorem sugar_water_inequality (a b c d : ℝ) : 
  a > b ∧ b > 0 ∧ c > d ∧ d > 0 → 
  (b + d) / (a + d) < (b + c) / (a + c) ∧
  ∀ (a b : ℝ), a > 0 ∧ b > 0 → 
  a / (1 + a + b) + b / (1 + a + b) < a / (1 + a) + b / (1 + b) := by
sorry

end NUMINAMATH_CALUDE_sugar_water_inequality_l433_43340


namespace NUMINAMATH_CALUDE_uncool_relatives_count_l433_43351

/-- Given a club with the following characteristics:
  * 50 total people
  * 25 people with cool dads
  * 28 people with cool moms
  * 10 people with cool siblings
  * 15 people with both cool dads and cool moms
  * 5 people with both cool dads and cool siblings
  * 7 people with both cool moms and cool siblings
  * 3 people with cool dads, cool moms, and cool siblings
Prove that the number of people with all uncool relatives is 11. -/
theorem uncool_relatives_count (total : Nat) (cool_dad : Nat) (cool_mom : Nat) (cool_sibling : Nat)
    (cool_dad_and_mom : Nat) (cool_dad_and_sibling : Nat) (cool_mom_and_sibling : Nat)
    (cool_all : Nat) (h1 : total = 50) (h2 : cool_dad = 25) (h3 : cool_mom = 28)
    (h4 : cool_sibling = 10) (h5 : cool_dad_and_mom = 15) (h6 : cool_dad_and_sibling = 5)
    (h7 : cool_mom_and_sibling = 7) (h8 : cool_all = 3) :
  total - (cool_dad + cool_mom + cool_sibling - cool_dad_and_mom - cool_dad_and_sibling
           - cool_mom_and_sibling + cool_all) = 11 := by
  sorry

end NUMINAMATH_CALUDE_uncool_relatives_count_l433_43351


namespace NUMINAMATH_CALUDE_right_triangular_pyramid_area_relation_l433_43305

/-- Represents a triangular pyramid with right angles at each vertex -/
structure RightTriangularPyramid where
  /-- The length of the first base edge -/
  a : ℝ
  /-- The length of the second base edge -/
  b : ℝ
  /-- The length of the third base edge -/
  c : ℝ
  /-- The length of the first edge from apex to base -/
  e₁ : ℝ
  /-- The length of the second edge from apex to base -/
  e₂ : ℝ
  /-- The length of the third edge from apex to base -/
  e₃ : ℝ
  /-- The area of the first lateral face -/
  t₁ : ℝ
  /-- The area of the second lateral face -/
  t₂ : ℝ
  /-- The area of the third lateral face -/
  t₃ : ℝ
  /-- The area of the base -/
  T : ℝ
  /-- Condition: right angles at vertices -/
  right_angles : a^2 = e₁^2 + e₂^2 ∧ b^2 = e₂^2 + e₃^2 ∧ c^2 = e₃^2 + e₁^2
  /-- Condition: lateral face areas -/
  lateral_areas : t₁ = (1/2) * e₁ * e₂ ∧ t₂ = (1/2) * e₂ * e₃ ∧ t₃ = (1/2) * e₃ * e₁
  /-- Condition: base area -/
  base_area : T = (1/4) * Real.sqrt ((a+b+c)*(a+b-c)*(a-b+c)*(b+c-a))

/-- The square of the base area is equal to the sum of the squares of the lateral face areas -/
theorem right_triangular_pyramid_area_relation (p : RightTriangularPyramid) :
  p.T^2 = p.t₁^2 + p.t₂^2 + p.t₃^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangular_pyramid_area_relation_l433_43305


namespace NUMINAMATH_CALUDE_odd_function_implies_m_eq_neg_one_l433_43390

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The logarithm function with base a -/
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

/-- The given function f -/
noncomputable def f (a m : ℝ) (x : ℝ) : ℝ := log a ((1 - m*x) / (x - 1))

theorem odd_function_implies_m_eq_neg_one (a m : ℝ) 
    (h1 : a > 0) (h2 : a ≠ 1) (h3 : IsOddFunction (f a m)) : m = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_implies_m_eq_neg_one_l433_43390


namespace NUMINAMATH_CALUDE_train_passing_time_l433_43399

/-- Given a train of length 420 meters traveling at 63 km/hr,
    prove that it takes 24 seconds to pass a stationary point. -/
theorem train_passing_time (train_length : Real) (train_speed_kmh : Real) :
  train_length = 420 ∧ train_speed_kmh = 63 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l433_43399


namespace NUMINAMATH_CALUDE_crates_in_third_trip_is_two_l433_43314

/-- Represents the problem of distributing crates across multiple trips. -/
structure CrateDistribution where
  total_crates : ℕ
  min_crate_weight : ℕ
  max_trip_weight : ℕ

/-- Calculates the number of crates in the third trip. -/
def crates_in_third_trip (cd : CrateDistribution) : ℕ :=
  cd.total_crates - 2 * (cd.max_trip_weight / cd.min_crate_weight)

/-- Theorem stating that for the given conditions, the number of crates in the third trip is 2. -/
theorem crates_in_third_trip_is_two :
  let cd : CrateDistribution := {
    total_crates := 12,
    min_crate_weight := 120,
    max_trip_weight := 600
  }
  crates_in_third_trip cd = 2 := by
  sorry

end NUMINAMATH_CALUDE_crates_in_third_trip_is_two_l433_43314


namespace NUMINAMATH_CALUDE_platform_length_l433_43379

/-- The length of a platform given train crossing times -/
theorem platform_length (train_length : ℝ) (post_time : ℝ) (platform_time : ℝ) :
  train_length = 150 →
  post_time = 15 →
  platform_time = 25 →
  ∃ (platform_length : ℝ),
    platform_length = 100 ∧
    train_length / post_time = (train_length + platform_length) / platform_time :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l433_43379


namespace NUMINAMATH_CALUDE_harry_pizza_order_cost_l433_43374

def pizza_order (large_pizza_cost : ℕ) (topping_cost : ℕ) (num_pizzas : ℕ) (num_toppings : ℕ) (tip_percentage : ℚ) : ℚ :=
  let pizza_with_toppings_cost := large_pizza_cost + topping_cost * num_toppings
  let total_pizza_cost := pizza_with_toppings_cost * num_pizzas
  let tip := (tip_percentage / 100) * total_pizza_cost
  total_pizza_cost + tip

theorem harry_pizza_order_cost :
  pizza_order 14 2 2 3 25 = 50 := by
  sorry

end NUMINAMATH_CALUDE_harry_pizza_order_cost_l433_43374


namespace NUMINAMATH_CALUDE_green_guards_with_shields_l433_43397

theorem green_guards_with_shields (total : ℝ) (green : ℝ) (yellow : ℝ) (special : ℝ) 
  (h1 : green = (3/8) * total)
  (h2 : yellow = (5/8) * total)
  (h3 : special = (1/5) * total)
  (h4 : ∃ (r s : ℝ), (green * (r/s) + yellow * (r/(3*s)) = special) ∧ (r/s > 0) ∧ (s ≠ 0)) :
  ∃ (r s : ℝ), (r/s = 12/35) ∧ (green * (r/s) = (3/5) * special) := by
  sorry

end NUMINAMATH_CALUDE_green_guards_with_shields_l433_43397


namespace NUMINAMATH_CALUDE_sqrt_real_range_l433_43342

theorem sqrt_real_range (x : ℝ) : (∃ y : ℝ, y^2 = x - 3) → x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_real_range_l433_43342


namespace NUMINAMATH_CALUDE_egyptian_fraction_sum_l433_43336

theorem egyptian_fraction_sum : ∃ (b₂ b₃ b₄ b₅ b₆ b₇ b₈ : ℕ),
  (5 : ℚ) / 8 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040 + b₈ / 40320 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧ b₇ < 7 ∧ b₈ < 8 ∧
  b₂ + b₃ + b₄ + b₅ + b₆ + b₇ + b₈ = 4 := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_sum_l433_43336


namespace NUMINAMATH_CALUDE_sum_of_squares_l433_43312

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 52) : x^2 + y^2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l433_43312
