import Mathlib

namespace box_length_l905_90519

/-- The length of a box with given dimensions and cube requirements -/
theorem box_length (width : ℝ) (height : ℝ) (cube_volume : ℝ) (num_cubes : ℕ)
  (h_width : width = 12)
  (h_height : height = 3)
  (h_cube_volume : cube_volume = 3)
  (h_num_cubes : num_cubes = 108) :
  width * height * (num_cubes : ℝ) * cube_volume / (width * height) = 9 := by
  sorry

end box_length_l905_90519


namespace odd_square_minus_one_div_eight_l905_90549

theorem odd_square_minus_one_div_eight (a : ℕ) (h1 : a > 0) (h2 : Odd a) :
  ∃ k : ℤ, a^2 - 1 = 8 * k :=
sorry

end odd_square_minus_one_div_eight_l905_90549


namespace difference_of_fractions_numerator_l905_90568

theorem difference_of_fractions_numerator : 
  let a := 2024
  let b := 2023
  let diff := a / b - b / a
  let p := (a^2 - b^2) / (a * b)
  p = 4047 := by sorry

end difference_of_fractions_numerator_l905_90568


namespace community_cleaning_event_l905_90570

theorem community_cleaning_event (total : ℝ) : 
  (0.3 * total = total * 0.3) →
  (0.6 * total = 2 * (total * 0.3)) →
  (total - (total * 0.3 + 0.6 * total) = 200) →
  total = 2000 := by
sorry

end community_cleaning_event_l905_90570


namespace soup_weight_proof_l905_90530

theorem soup_weight_proof (initial_weight : ℝ) : 
  (((initial_weight / 2) / 2) / 2 = 5) → initial_weight = 40 := by
  sorry

end soup_weight_proof_l905_90530


namespace equation_solution_l905_90528

theorem equation_solution : 
  ∃ t : ℝ, 3 * 3^t + Real.sqrt (9 * 9^t) = 18 ∧ t = 1 := by sorry

end equation_solution_l905_90528


namespace binomial_coefficient_1000_1000_l905_90505

theorem binomial_coefficient_1000_1000 : Nat.choose 1000 1000 = 1 := by sorry

end binomial_coefficient_1000_1000_l905_90505


namespace larger_ssr_not_better_fit_l905_90592

/-- Represents a simple linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ
  x : List ℝ
  y : List ℝ

/-- Calculates the sum of squared residuals for a given model -/
def sumSquaredResiduals (model : LinearRegression) : ℝ :=
  sorry

/-- Represents the goodness of fit of a model -/
def goodnessOfFit (model : LinearRegression) : ℝ :=
  sorry

theorem larger_ssr_not_better_fit (model1 model2 : LinearRegression) :
  sumSquaredResiduals model1 > sumSquaredResiduals model2 →
  goodnessOfFit model1 ≤ goodnessOfFit model2 :=
sorry

end larger_ssr_not_better_fit_l905_90592


namespace max_profit_is_850_l905_90572

def fruit_problem (m : ℝ) : Prop :=
  let total_weight : ℝ := 200
  let profit_A : ℝ := 20 - 16
  let profit_B : ℝ := 25 - 20
  let total_profit : ℝ := m * profit_A + (total_weight - m) * profit_B
  0 ≤ m ∧ m ≤ total_weight ∧ m ≥ 3 * (total_weight - m) →
  total_profit ≤ 850

theorem max_profit_is_850 :
  ∃ m : ℝ, fruit_problem m ∧
  (∀ n : ℝ, fruit_problem n → 
    m * (20 - 16) + (200 - m) * (25 - 20) ≥ n * (20 - 16) + (200 - n) * (25 - 20)) :=
sorry

end max_profit_is_850_l905_90572


namespace complex_sum_modulus_l905_90542

theorem complex_sum_modulus (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 1) 
  (h2 : Complex.abs z₂ = 1) 
  (h3 : Complex.abs (z₁ - z₂) = Real.sqrt 3) : 
  Complex.abs (z₁ + z₂) = 1 := by
  sorry

end complex_sum_modulus_l905_90542


namespace quadratic_equation_roots_specific_root_condition_l905_90598

theorem quadratic_equation_roots (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x + k + 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ↔ k ≤ 3 :=
by sorry

theorem specific_root_condition (k : ℝ) (x₁ x₂ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x + k + 1
  (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂ ∧ 3/x₁ + 3/x₂ = x₁*x₂ - 4) → k = -3 :=
by sorry

end quadratic_equation_roots_specific_root_condition_l905_90598


namespace unique_solution_exists_l905_90547

theorem unique_solution_exists (y : ℝ) (h : y > 0) :
  ∃! x : ℝ, (2 ^ (4 * x + 2)) * (4 ^ (2 * x + 3)) = 8 ^ (3 * x + 4) * y :=
by sorry

end unique_solution_exists_l905_90547


namespace students_playing_both_sports_l905_90583

def total_students : ℕ := 470
def football_players : ℕ := 325
def cricket_players : ℕ := 175
def neither_players : ℕ := 50

theorem students_playing_both_sports : ℕ := by
  sorry

#check students_playing_both_sports = 80

end students_playing_both_sports_l905_90583


namespace length_MN_circle_P_equation_l905_90578

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define the intersection points M and N
def intersection_points (M N : ℝ × ℝ) : Prop :=
  line_l M.1 M.2 ∧ circle_C M.1 M.2 ∧
  line_l N.1 N.2 ∧ circle_C N.1 N.2 ∧
  M ≠ N

-- Theorem for the length of MN
theorem length_MN (M N : ℝ × ℝ) (h : intersection_points M N) :
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 2 :=
sorry

-- Define the circle P
def circle_P (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Theorem for the equation of circle P
theorem circle_P_equation (M N : ℝ × ℝ) (h : intersection_points M N) :
  ∀ x y : ℝ, circle_P x y ↔ 
    ((x - (M.1 + N.1) / 2)^2 + (y - (M.2 + N.2) / 2)^2 = 
     ((M.1 - N.1)^2 + (M.2 - N.2)^2) / 4) :=
sorry

end length_MN_circle_P_equation_l905_90578


namespace simplify_and_evaluate_l905_90520

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (1 - x / (x + 1)) / ((x^2 - 1) / (x^2 + 2*x + 1)) = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l905_90520


namespace will_buttons_count_l905_90504

theorem will_buttons_count (mari_buttons : ℕ) (kendra_buttons : ℕ) (sue_buttons : ℕ) (will_buttons : ℕ) : 
  mari_buttons = 8 →
  kendra_buttons = 5 * mari_buttons + 4 →
  sue_buttons = kendra_buttons / 2 →
  will_buttons = 2 * (kendra_buttons + sue_buttons) →
  will_buttons = 132 :=
by
  sorry

end will_buttons_count_l905_90504


namespace log_problem_l905_90579

theorem log_problem (x y : ℝ) (h1 : Real.log (x * y^4) = 1) (h2 : Real.log (x^3 * y) = 1) :
  Real.log (x^3 * y^2) = 13/11 := by
  sorry

end log_problem_l905_90579


namespace cubic_expansion_result_l905_90557

theorem cubic_expansion_result (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (Real.sqrt 3 * x - Real.sqrt 2)^3 = a₀ * x^3 + a₁ * x^2 + a₂ * x + a₃) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end cubic_expansion_result_l905_90557


namespace symmetry_implies_coordinates_l905_90539

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposites and their y-coordinates are the same. -/
def symmetric_wrt_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

theorem symmetry_implies_coordinates (a b : ℝ) :
  symmetric_wrt_y_axis (a, 3) (-2, b) → a = 2 ∧ b = 3 := by
  sorry

end symmetry_implies_coordinates_l905_90539


namespace inequality_theorem_l905_90538

/-- The function f(x, y) = ax² + 2bxy + cy² -/
def f (a b c x y : ℝ) : ℝ := a * x^2 + 2 * b * x * y + c * y^2

/-- The main theorem -/
theorem inequality_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_nonneg : ∀ (x y : ℝ), 0 ≤ f a b c x y) :
  ∀ (x₁ x₂ y₁ y₂ : ℝ),
    Real.sqrt (f a b c x₁ y₁ * f a b c x₂ y₂) * f a b c (x₁ - x₂) (y₁ - y₂) ≥
    (a * c - b^2) * (x₁ * y₂ - x₂ * y₁)^2 := by
  sorry

end inequality_theorem_l905_90538


namespace reciprocal_sum_theorem_l905_90523

def sum_of_reciprocals (a b : ℕ+) : ℚ := (a⁻¹ : ℚ) + (b⁻¹ : ℚ)

theorem reciprocal_sum_theorem (a b : ℕ+) 
  (sum_cond : a + b = 45)
  (lcm_cond : Nat.lcm a b = 120)
  (hcf_cond : Nat.gcd a b = 5) :
  sum_of_reciprocals a b = 3/40 := by
  sorry

end reciprocal_sum_theorem_l905_90523


namespace inequality_proof_l905_90500

/-- The function f(x) defined as |x-m| + |x+3| -/
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| + |x + 3|

/-- Theorem stating that given the conditions, 1/(m+n) + 1/t ≥ 2 -/
theorem inequality_proof (m n t : ℝ) (hm : m > 0) (hn : n > 0) (ht : t > 0) 
  (h_min : ∀ x, f m x ≥ 5 - n - t) : 
  1 / (m + n) + 1 / t ≥ 2 := by
  sorry


end inequality_proof_l905_90500


namespace semicircle_perimeter_approx_l905_90595

/-- The perimeter of a semicircle with radius 20 is approximately 102.83 -/
theorem semicircle_perimeter_approx :
  let r : ℝ := 20
  let perimeter : ℝ := π * r + 2 * r
  ∃ ε > 0, abs (perimeter - 102.83) < ε :=
by sorry

end semicircle_perimeter_approx_l905_90595


namespace space_diagonals_specific_polyhedron_l905_90577

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  sorry

/-- Theorem stating the number of space diagonals in the specific polyhedron -/
theorem space_diagonals_specific_polyhedron :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 70,
    faces := 42,
    triangular_faces := 30,
    quadrilateral_faces := 12
  }
  space_diagonals Q = 341 := by
  sorry

end space_diagonals_specific_polyhedron_l905_90577


namespace vector_addition_scalar_mult_l905_90584

/-- Given plane vectors a and b, prove that 3a + b equals (-2, 6) -/
theorem vector_addition_scalar_mult 
  (a b : ℝ × ℝ) 
  (ha : a = (-1, 2)) 
  (hb : b = (1, 0)) : 
  (3 : ℝ) • a + b = (-2, 6) := by
  sorry

end vector_addition_scalar_mult_l905_90584


namespace fifteen_percent_of_600_is_90_l905_90562

theorem fifteen_percent_of_600_is_90 :
  ∀ x : ℝ, (15 / 100) * x = 90 → x = 600 := by
  sorry

end fifteen_percent_of_600_is_90_l905_90562


namespace bake_sale_ratio_l905_90537

/-- Given a bake sale where 104 items were sold in total, with 48 cookies sold,
    prove that the ratio of brownies to cookies sold is 7:6. -/
theorem bake_sale_ratio : 
  let total_items : ℕ := 104
  let cookies_sold : ℕ := 48
  let brownies_sold : ℕ := total_items - cookies_sold
  (brownies_sold : ℚ) / (cookies_sold : ℚ) = 7 / 6 := by
  sorry

end bake_sale_ratio_l905_90537


namespace mod_eight_difference_l905_90503

theorem mod_eight_difference (n : ℕ) : (47^n - 23^n) % 8 = 0 :=
sorry

end mod_eight_difference_l905_90503


namespace polynomial_coefficient_sum_l905_90507

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 2 := by
sorry

end polynomial_coefficient_sum_l905_90507


namespace polynomial_factorization_l905_90596

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 2*x + 1) * (x^2 + 8*x + 15) + (x^2 + 6*x + 5) = (x + 1) * (x + 5) * (x + 2)^2 := by
  sorry

end polynomial_factorization_l905_90596


namespace gain_percent_is_112_5_l905_90513

/-- Represents the ratio of selling price to cost price -/
def price_ratio : ℚ := 5 / 2

/-- Represents the discount factor applied to the selling price -/
def discount_factor : ℚ := 85 / 100

/-- Calculates the gain percent based on the given conditions -/
def gain_percent : ℚ := (price_ratio * discount_factor - 1) * 100

/-- Theorem stating the gain percent under the given conditions -/
theorem gain_percent_is_112_5 : gain_percent = 112.5 := by
  sorry

end gain_percent_is_112_5_l905_90513


namespace smallest_non_even_units_digit_l905_90580

def EvenUnitsDigits : Set Nat := {0, 2, 4, 6, 8}

theorem smallest_non_even_units_digit : 
  (∀ d : Nat, d < 10 → d ∉ EvenUnitsDigits → 1 ≤ d) ∧ 1 ∉ EvenUnitsDigits := by
  sorry

end smallest_non_even_units_digit_l905_90580


namespace slide_total_boys_l905_90551

theorem slide_total_boys (initial : ℕ) (second : ℕ) (third : ℕ) 
  (h1 : initial = 87) 
  (h2 : second = 46) 
  (h3 : third = 29) : 
  initial + second + third = 162 := by
  sorry

end slide_total_boys_l905_90551


namespace circle_symmetry_symmetric_circle_correct_l905_90501

/-- Given two circles in the xy-plane, this theorem states that they are symmetric with respect to the line y = x. -/
theorem circle_symmetry (x y : ℝ) : 
  ((x - 3)^2 + (y + 1)^2 = 2) ↔ ((y + 1)^2 + (x - 3)^2 = 2) := by sorry

/-- The equation of the circle symmetric to (x-3)^2 + (y+1)^2 = 2 with respect to y = x -/
def symmetric_circle_equation (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 3)^2 = 2

theorem symmetric_circle_correct (x y : ℝ) : 
  symmetric_circle_equation x y ↔ ((y - 3)^2 + (x + 1)^2 = 2) := by sorry

end circle_symmetry_symmetric_circle_correct_l905_90501


namespace angle_identity_l905_90506

/-- If the terminal side of angle α passes through point P(-2, 1) in the rectangular coordinate system, 
    then cos²α - sin(2α) = 8/5 -/
theorem angle_identity (α : ℝ) : 
  (∃ (x y : ℝ), x = -2 ∧ y = 1 ∧ y / x = Real.tan α) → 
  Real.cos α ^ 2 - Real.sin (2 * α) = 8 / 5 := by
  sorry

end angle_identity_l905_90506


namespace combined_average_marks_l905_90591

theorem combined_average_marks (n1 n2 : ℕ) (avg1 avg2 : ℝ) : 
  n1 = 26 → 
  n2 = 50 → 
  avg1 = 40 → 
  avg2 = 60 → 
  let total_students := n1 + n2
  let total_marks := n1 * avg1 + n2 * avg2
  abs ((total_marks / total_students) - 53.16) < 0.01 := by
sorry

end combined_average_marks_l905_90591


namespace police_emergency_number_prime_factor_l905_90502

/-- A police emergency number is a positive integer that ends with 133 in decimal representation. -/
def PoliceEmergencyNumber (n : ℕ+) : Prop :=
  ∃ k : ℕ, n = k * 1000 + 133

/-- Theorem: Every police emergency number has a prime factor greater than 7. -/
theorem police_emergency_number_prime_factor
  (n : ℕ+) (h : PoliceEmergencyNumber n) :
  ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n.val :=
by sorry

end police_emergency_number_prime_factor_l905_90502


namespace son_work_time_l905_90560

/-- Given a task that can be completed by a man in 7 days or by the man and his son together in 3 days, 
    this theorem proves that the son can complete the task alone in 5.25 days. -/
theorem son_work_time (man_time : ℝ) (combined_time : ℝ) (son_time : ℝ) : 
  man_time = 7 → combined_time = 3 → son_time = 21 / 4 := by
  sorry

end son_work_time_l905_90560


namespace incircle_and_inscribed_circles_inequality_l905_90554

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the theorem
theorem incircle_and_inscribed_circles_inequality 
  (triangle : Triangle) 
  (incircle : Circle) 
  (inscribed_circle1 inscribed_circle2 inscribed_circle3 : Circle) :
  -- Conditions
  (incircle.radius > 0) →
  (inscribed_circle1.radius > 0) →
  (inscribed_circle2.radius > 0) →
  (inscribed_circle3.radius > 0) →
  (inscribed_circle1.radius < incircle.radius) →
  (inscribed_circle2.radius < incircle.radius) →
  (inscribed_circle3.radius < incircle.radius) →
  -- Theorem statement
  inscribed_circle1.radius + inscribed_circle2.radius + inscribed_circle3.radius ≥ incircle.radius :=
by
  sorry

end incircle_and_inscribed_circles_inequality_l905_90554


namespace lucas_numbers_l905_90508

theorem lucas_numbers (a b : ℤ) : 
  3 * a + 4 * b = 140 → (a = 20 ∨ b = 20) → a = 20 ∧ b = 20 := by
  sorry

end lucas_numbers_l905_90508


namespace cube_neg_iff_neg_l905_90589

theorem cube_neg_iff_neg (x : ℝ) : x^3 < 0 ↔ x < 0 := by sorry

end cube_neg_iff_neg_l905_90589


namespace aruns_weight_lower_limit_l905_90575

theorem aruns_weight_lower_limit 
  (lower_bound : ℝ) 
  (upper_bound : ℝ) 
  (h1 : lower_bound > 65)
  (h2 : upper_bound ≤ 68)
  (h3 : (lower_bound + upper_bound) / 2 = 67) :
  lower_bound = 66 := by
sorry

end aruns_weight_lower_limit_l905_90575


namespace range_of_a_l905_90525

/-- Proposition p: The function y=(a-1)x is increasing -/
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (a - 1) * x < (a - 1) * y

/-- Proposition q: The inequality -x^2+2x-2≤a holds true for all real numbers x -/
def q (a : ℝ) : Prop := ∀ x : ℝ, -x^2 + 2*x - 2 ≤ a

/-- The main theorem stating the range of a -/
theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) : 
  a ∈ Set.Icc (-1 : ℝ) 1 :=
sorry

end range_of_a_l905_90525


namespace complex_fraction_simplification_l905_90564

theorem complex_fraction_simplification :
  let z₁ : ℂ := 3 + 3*I
  let z₂ : ℂ := -1 + 3*I
  z₁ / z₂ = (-1.2 : ℝ) - 1.2*I := by sorry

end complex_fraction_simplification_l905_90564


namespace ratio_equality_l905_90511

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 49)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 64)
  (dot_product : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end ratio_equality_l905_90511


namespace min_value_product_l905_90582

theorem min_value_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 8) :
  (2 * a + 3 * b) * (2 * b + 3 * c) * (2 * c + 3 * a) ≥ 288 := by
  sorry

end min_value_product_l905_90582


namespace eight_p_plus_one_composite_l905_90563

theorem eight_p_plus_one_composite (p : ℕ) (h1 : Nat.Prime p) (h2 : Nat.Prime (8 * p - 1)) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = 8 * p + 1 :=
by sorry

end eight_p_plus_one_composite_l905_90563


namespace mary_added_candy_l905_90594

/-- Proof that Mary added 10 pieces of candy to her collection --/
theorem mary_added_candy (megan_candy : ℕ) (mary_total : ℕ) (h1 : megan_candy = 5) (h2 : mary_total = 25) :
  mary_total - (3 * megan_candy) = 10 := by
  sorry

end mary_added_candy_l905_90594


namespace magic_balls_theorem_l905_90576

theorem magic_balls_theorem :
  ∃ (n : ℕ), 5 + 4 * n = 2005 :=
by sorry

end magic_balls_theorem_l905_90576


namespace prob_at_least_one_red_l905_90585

/-- Represents a box containing red and white balls -/
structure Box where
  red_balls : ℕ
  white_balls : ℕ

/-- Calculates the probability of drawing a specific color ball from a box -/
def prob_draw (b : Box) (color : String) : ℚ :=
  if color = "red" then
    b.red_balls / (b.red_balls + b.white_balls)
  else if color = "white" then
    b.white_balls / (b.red_balls + b.white_balls)
  else
    0

/-- Theorem: The probability of drawing at least one red ball from two boxes,
    each containing 2 red balls and 1 white ball, is equal to 8/9 -/
theorem prob_at_least_one_red (box_a box_b : Box) 
  (ha : box_a.red_balls = 2 ∧ box_a.white_balls = 1)
  (hb : box_b.red_balls = 2 ∧ box_b.white_balls = 1) : 
  1 - (prob_draw box_a "white" * prob_draw box_b "white") = 8/9 :=
sorry

end prob_at_least_one_red_l905_90585


namespace smallest_digit_sum_of_product_l905_90512

/-- Given two two-digit positive integers with all digits different and both less than 50,
    the smallest possible sum of digits of their product (a four-digit number) is 20. -/
theorem smallest_digit_sum_of_product (m n : ℕ) : 
  10 ≤ m ∧ m < 50 ∧ 10 ≤ n ∧ n < 50 ∧ 
  (∀ d₁ d₂ d₃ d₄, m = 10 * d₁ + d₂ ∧ n = 10 * d₃ + d₄ → d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄) →
  1000 ≤ m * n ∧ m * n < 10000 →
  20 ≤ (m * n / 1000 + (m * n / 100) % 10 + (m * n / 10) % 10 + m * n % 10) ∧
  ∀ p q : ℕ, 10 ≤ p ∧ p < 50 ∧ 10 ≤ q ∧ q < 50 →
    (∀ e₁ e₂ e₃ e₄, p = 10 * e₁ + e₂ ∧ q = 10 * e₃ + e₄ → e₁ ≠ e₂ ∧ e₁ ≠ e₃ ∧ e₁ ≠ e₄ ∧ e₂ ≠ e₃ ∧ e₂ ≠ e₄ ∧ e₃ ≠ e₄) →
    1000 ≤ p * q ∧ p * q < 10000 →
    (p * q / 1000 + (p * q / 100) % 10 + (p * q / 10) % 10 + p * q % 10) ≥ 20 :=
by sorry

end smallest_digit_sum_of_product_l905_90512


namespace inheritance_tax_theorem_inheritance_uniqueness_l905_90531

/-- The original amount of inheritance --/
def inheritance : ℝ := 41379

/-- The total amount of taxes paid --/
def total_taxes : ℝ := 15000

/-- Theorem stating that the inheritance amount satisfies the tax conditions --/
theorem inheritance_tax_theorem :
  0.25 * inheritance + 0.15 * (0.75 * inheritance) = total_taxes :=
by sorry

/-- Theorem proving that the inheritance amount is unique --/
theorem inheritance_uniqueness (x : ℝ) :
  0.25 * x + 0.15 * (0.75 * x) = total_taxes → x = inheritance :=
by sorry

end inheritance_tax_theorem_inheritance_uniqueness_l905_90531


namespace circle_square_area_l905_90534

theorem circle_square_area (r : ℝ) (s : ℝ) (hr : r = 1) (hs : s = 2) :
  let circle_area := π * r^2
  let square_area := s^2
  let square_diagonal := s * Real.sqrt 2
  circle_area - square_area = 0 := by sorry

end circle_square_area_l905_90534


namespace stagecoach_encounter_l905_90514

/-- The number of stagecoaches traveling daily from Bratislava to Brașov -/
def daily_coaches_bratislava_to_brasov : ℕ := 2

/-- The number of stagecoaches traveling daily from Brașov to Bratislava -/
def daily_coaches_brasov_to_bratislava : ℕ := 2

/-- The number of days the journey takes -/
def journey_duration : ℕ := 10

/-- The number of stagecoaches encountered when traveling from Bratislava to Brașov -/
def encountered_coaches : ℕ := daily_coaches_brasov_to_bratislava * journey_duration

theorem stagecoach_encounter :
  encountered_coaches = 20 :=
sorry

end stagecoach_encounter_l905_90514


namespace jennas_profit_l905_90571

/-- Calculates the total profit for Jenna's wholesale business --/
def calculate_profit (buy_price sell_price rent tax_rate worker_salary num_workers num_widgets : ℝ) : ℝ :=
  let total_revenue := sell_price * num_widgets
  let total_cost := buy_price * num_widgets
  let gross_profit := total_revenue - total_cost
  let total_expenses := rent + (worker_salary * num_workers)
  let net_profit_before_tax := gross_profit - total_expenses
  let taxes := tax_rate * net_profit_before_tax
  net_profit_before_tax - taxes

/-- Theorem stating that Jenna's total profit is $4,000 given the specified conditions --/
theorem jennas_profit :
  calculate_profit 3 8 10000 0.2 2500 4 5000 = 4000 := by
  sorry

end jennas_profit_l905_90571


namespace adam_ferris_wheel_cost_l905_90569

/-- The amount of money Adam spent on the ferris wheel ride -/
def ferris_wheel_cost (initial_tickets : ℕ) (remaining_tickets : ℕ) (ticket_price : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) * ticket_price

/-- Theorem: Adam spent 81 dollars on the ferris wheel ride -/
theorem adam_ferris_wheel_cost :
  ferris_wheel_cost 13 4 9 = 81 := by
  sorry

end adam_ferris_wheel_cost_l905_90569


namespace simplify_fraction_l905_90527

theorem simplify_fraction : 
  ((2^1010)^2 - (2^1008)^2) / ((2^1009)^2 - (2^1007)^2) = 4 := by
  sorry

end simplify_fraction_l905_90527


namespace count_integer_lengths_specific_triangle_l905_90546

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ  -- length of first leg
  b : ℕ  -- length of second leg
  c : ℕ  -- length of hypotenuse
  right_angle : c^2 = a^2 + b^2  -- Pythagorean theorem

/-- Counts the number of distinct integer lengths of line segments
    that can be drawn from a vertex to the opposite side -/
def count_integer_lengths (t : RightTriangle) : ℕ :=
  -- Implementation details omitted
  sorry

/-- The main theorem -/
theorem count_integer_lengths_specific_triangle :
  ∃ t : RightTriangle, t.a = 15 ∧ t.b = 20 ∧ count_integer_lengths t = 9 :=
by
  sorry

end count_integer_lengths_specific_triangle_l905_90546


namespace simplify_expression_l905_90532

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^(2/3) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / ((1/3) * a^(1/6) * b^(5/6)) = -9 * a :=
by sorry

end simplify_expression_l905_90532


namespace correct_equation_l905_90593

theorem correct_equation (a b : ℝ) : 3 * a^2 * b - 4 * b * a^2 = -a^2 * b := by
  sorry

end correct_equation_l905_90593


namespace cube_root_of_sqrt_64_l905_90524

theorem cube_root_of_sqrt_64 : ∃ (x : ℝ), x^3 = Real.sqrt 64 ∧ (x = 2 ∨ x = -2) := by
  sorry

end cube_root_of_sqrt_64_l905_90524


namespace smallest_number_l905_90588

theorem smallest_number (s : Finset ℚ) (hs : s = {-5, 1, -1, 0}) : 
  ∀ x ∈ s, -5 ≤ x :=
by
  sorry

end smallest_number_l905_90588


namespace rectangle_area_change_l905_90590

theorem rectangle_area_change (original_area : ℝ) : 
  original_area = 540 →
  (0.9 * 1.2 * original_area : ℝ) = 583.2 :=
by sorry

end rectangle_area_change_l905_90590


namespace smaug_silver_coins_l905_90533

/-- Represents the number of coins of each type in Smaug's hoard -/
structure DragonHoard where
  gold : ℕ
  silver : ℕ
  copper : ℕ

/-- Calculates the total value of the hoard in copper coins -/
def hoardValue (h : DragonHoard) : ℕ :=
  h.gold * 3 * 8 + h.silver * 8 + h.copper

/-- Theorem stating that Smaug has 60 silver coins -/
theorem smaug_silver_coins :
  ∃ h : DragonHoard,
    h.gold = 100 ∧
    h.copper = 33 ∧
    hoardValue h = 2913 ∧
    h.silver = 60 := by
  sorry

end smaug_silver_coins_l905_90533


namespace bike_license_count_l905_90548

/-- The number of possible letters for a bike license -/
def num_letters : ℕ := 3

/-- The number of digits in a bike license -/
def num_digits : ℕ := 4

/-- The number of possible digits for each position (0-9) -/
def digits_per_position : ℕ := 10

/-- The total number of possible bike licenses -/
def total_licenses : ℕ := num_letters * (digits_per_position ^ num_digits)

theorem bike_license_count : total_licenses = 30000 := by
  sorry

end bike_license_count_l905_90548


namespace sqrt_equation_solution_l905_90553

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt x / 15 = 4 → x = 3600 := by sorry

end sqrt_equation_solution_l905_90553


namespace larger_number_with_given_hcf_lcm_factors_l905_90567

theorem larger_number_with_given_hcf_lcm_factors (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  Nat.gcd a b = 120 ∧
  ∃ k : ℕ, Nat.lcm a b = 120 * 13 * 17 * 23 * k ∧ k = 1 →
  max a b = 26520 :=
by sorry

end larger_number_with_given_hcf_lcm_factors_l905_90567


namespace imaginary_part_of_z_l905_90509

theorem imaginary_part_of_z (z : ℂ) : z = (Complex.I : ℂ) / (1 - Complex.I) → z.im = 1 / 2 := by
  sorry

end imaginary_part_of_z_l905_90509


namespace textbooks_on_sale_textbooks_on_sale_is_five_l905_90516

/-- Proves the number of textbooks bought on sale given the conditions of the problem -/
theorem textbooks_on_sale (sale_price : ℕ) (online_total : ℕ) (bookstore_multiplier : ℕ) (total_spent : ℕ) : ℕ :=
  let sale_count := (total_spent - online_total - (bookstore_multiplier * online_total)) / sale_price
  sale_count

#check textbooks_on_sale 10 40 3 210 = 5

/-- The main theorem that proves the number of textbooks bought on sale is 5 -/
theorem textbooks_on_sale_is_five : textbooks_on_sale 10 40 3 210 = 5 := by
  sorry

end textbooks_on_sale_textbooks_on_sale_is_five_l905_90516


namespace scientific_notation_of_1050000_l905_90587

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  norm : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_1050000 :
  toScientificNotation 1050000 = ScientificNotation.mk 1.05 6 (by norm_num) :=
sorry

end scientific_notation_of_1050000_l905_90587


namespace motorcycle_license_count_l905_90559

/-- The number of possible letters for a motorcycle license -/
def num_letters : ℕ := 3

/-- The number of digits in a motorcycle license -/
def num_digits : ℕ := 6

/-- The number of possible choices for each digit -/
def choices_per_digit : ℕ := 10

/-- The total number of possible motorcycle licenses -/
def total_licenses : ℕ := num_letters * (choices_per_digit ^ num_digits)

theorem motorcycle_license_count :
  total_licenses = 3000000 := by
  sorry

end motorcycle_license_count_l905_90559


namespace exactly_six_valid_tuples_l905_90556

def is_valid_tuple (t : Fin 4 → Fin 4) : Prop :=
  (∃ (σ : Equiv (Fin 4) (Fin 4)), ∀ i, t i = σ i) ∧
  (t 0 = 1 ∨ t 1 ≠ 1 ∨ t 2 = 2 ∨ t 3 ≠ 4) ∧
  ¬(t 0 = 1 ∧ t 1 ≠ 1) ∧
  ¬(t 0 = 1 ∧ t 2 = 2) ∧
  ¬(t 0 = 1 ∧ t 3 ≠ 4) ∧
  ¬(t 1 ≠ 1 ∧ t 2 = 2) ∧
  ¬(t 1 ≠ 1 ∧ t 3 ≠ 4) ∧
  ¬(t 2 = 2 ∧ t 3 ≠ 4)

theorem exactly_six_valid_tuples :
  ∃! (s : Finset (Fin 4 → Fin 4)), s.card = 6 ∧ ∀ t, t ∈ s ↔ is_valid_tuple t :=
sorry

end exactly_six_valid_tuples_l905_90556


namespace tree_spacing_l905_90515

theorem tree_spacing (road_length : ℝ) (num_trees : ℕ) (space_between : ℝ) :
  road_length = 157 ∧ num_trees = 13 ∧ space_between = 12 →
  (road_length - space_between * (num_trees - 1)) / num_trees = 1 :=
by sorry

end tree_spacing_l905_90515


namespace jessica_seashells_l905_90597

/-- The number of seashells Jessica gave to Joan -/
def seashells_given : ℕ := 6

/-- The number of seashells Jessica kept -/
def seashells_kept : ℕ := 2

/-- The initial number of seashells Jessica found -/
def initial_seashells : ℕ := seashells_given + seashells_kept

theorem jessica_seashells : initial_seashells = 8 := by
  sorry

end jessica_seashells_l905_90597


namespace swim_meet_car_capacity_l905_90518

/-- Represents the transportation details for the swimming club's trip --/
structure SwimMeetTransport where
  num_cars : ℕ
  num_vans : ℕ
  people_per_car : ℕ
  people_per_van : ℕ
  max_per_van : ℕ
  additional_capacity : ℕ

/-- Calculates the maximum capacity per car given the transport details --/
def max_capacity_per_car (t : SwimMeetTransport) : ℕ :=
  let total_people := t.num_cars * t.people_per_car + t.num_vans * t.people_per_van
  let total_capacity := total_people + t.additional_capacity
  let van_capacity := t.num_vans * t.max_per_van
  (total_capacity - van_capacity) / t.num_cars

/-- Theorem stating that the maximum capacity per car is 6 for the given scenario --/
theorem swim_meet_car_capacity :
  let t : SwimMeetTransport := {
    num_cars := 2,
    num_vans := 3,
    people_per_car := 5,
    people_per_van := 3,
    max_per_van := 8,
    additional_capacity := 17
  }
  max_capacity_per_car t = 6 := by
  sorry

end swim_meet_car_capacity_l905_90518


namespace divisibility_condition_l905_90581

theorem divisibility_condition (a b : ℕ+) :
  (a.val * b.val^2 + b.val + 7) ∣ (a.val^2 * b.val + a.val + b.val) ↔
  (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k.val^2 ∧ b = 7 * k.val) :=
by sorry

end divisibility_condition_l905_90581


namespace floor_ceiling_difference_l905_90517

theorem floor_ceiling_difference : ⌊(1.999 : ℝ)⌋ - ⌈(3.001 : ℝ)⌉ = -3 := by
  sorry

end floor_ceiling_difference_l905_90517


namespace positive_roots_range_l905_90574

theorem positive_roots_range (m : ℝ) :
  (∀ x : ℝ, x^2 + (m+2)*x + m+5 = 0 → x > 0) ↔ -5 < m ∧ m ≤ -4 := by
  sorry

end positive_roots_range_l905_90574


namespace quarter_difference_l905_90521

/-- Represents the number and value of coins in Sally's savings jar. -/
structure CoinJar where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  total_coins : ℕ
  total_value : ℕ

/-- Checks if a CoinJar configuration is valid according to the problem constraints. -/
def is_valid_jar (jar : CoinJar) : Prop :=
  jar.total_coins = 150 ∧
  jar.total_value = 2000 ∧
  jar.total_coins = jar.nickels + jar.dimes + jar.quarters ∧
  jar.total_value = 5 * jar.nickels + 10 * jar.dimes + 25 * jar.quarters

/-- Finds the maximum number of quarters possible in a valid CoinJar. -/
def max_quarters (jar : CoinJar) : ℕ := sorry

/-- Finds the minimum number of quarters possible in a valid CoinJar. -/
def min_quarters (jar : CoinJar) : ℕ := sorry

/-- Theorem stating the difference between max and min quarters is 62. -/
theorem quarter_difference (jar : CoinJar) (h : is_valid_jar jar) :
  max_quarters jar - min_quarters jar = 62 := by sorry

end quarter_difference_l905_90521


namespace angle_of_inclination_slope_one_l905_90550

/-- The angle of inclination of a line with slope 1 in the Cartesian coordinate system is π/4 -/
theorem angle_of_inclination_slope_one :
  let line := {(x, y) : ℝ × ℝ | x - y - 3 = 0}
  let slope : ℝ := 1
  let angle_of_inclination := Real.arctan slope
  angle_of_inclination = π / 4 := by
  sorry

end angle_of_inclination_slope_one_l905_90550


namespace cubic_polynomial_value_at_6_l905_90522

/-- A cubic polynomial satisfying specific conditions -/
def cubic_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, p x = a*x^3 + b*x^2 + c*x + d) ∧
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 5 → p n = 1 / (n^2 : ℝ))

/-- Theorem stating that a cubic polynomial satisfying given conditions has p(6) = 0 -/
theorem cubic_polynomial_value_at_6 (p : ℝ → ℝ) (h : cubic_polynomial p) : p 6 = 0 := by
  sorry

end cubic_polynomial_value_at_6_l905_90522


namespace ratio_problem_l905_90565

theorem ratio_problem (a b : ℤ) : 
  (a : ℚ) / b = 1 / 4 → 
  (a + 6 : ℚ) / b = 1 / 2 → 
  b = 24 := by
sorry

end ratio_problem_l905_90565


namespace completing_square_equivalence_l905_90555

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 2*x - 8 = 0 ↔ (x - 1)^2 = 9 := by
sorry

end completing_square_equivalence_l905_90555


namespace perfect_square_factors_of_8640_l905_90529

/-- The number of positive integer factors of 8640 that are perfect squares -/
def num_perfect_square_factors (n : ℕ) : ℕ :=
  (Finset.range 4).card * (Finset.range 2).card * (Finset.range 1).card

/-- The prime factorization of 8640 -/
def prime_factorization (n : ℕ) : List (ℕ × ℕ) :=
  [(2, 6), (3, 3), (5, 1)]

theorem perfect_square_factors_of_8640 :
  num_perfect_square_factors 8640 = 8 ∧ prime_factorization 8640 = [(2, 6), (3, 3), (5, 1)] := by
  sorry

end perfect_square_factors_of_8640_l905_90529


namespace dodecahedral_die_expected_value_l905_90535

/-- A fair dodecahedral die with faces numbered from 1 to 12 -/
def DodecahedralDie : Finset ℕ := Finset.range 12

/-- The probability of each outcome for a fair die -/
def prob (n : ℕ) : ℚ := 1 / 12

/-- The expected value of rolling the die -/
def expected_value : ℚ := (DodecahedralDie.sum (fun i => prob i * (i + 1)))

/-- Theorem: The expected value of rolling a fair dodecahedral die is 6.5 -/
theorem dodecahedral_die_expected_value :
  expected_value = 13 / 2 := by sorry

end dodecahedral_die_expected_value_l905_90535


namespace raft_sticks_total_l905_90573

def simon_sticks : ℕ := 36

def gerry_sticks : ℕ := (2 * simon_sticks) / 3

def micky_sticks : ℕ := simon_sticks + gerry_sticks + 9

def darryl_sticks : ℕ := simon_sticks + gerry_sticks + micky_sticks + 1

def total_sticks : ℕ := simon_sticks + gerry_sticks + micky_sticks + darryl_sticks

theorem raft_sticks_total : total_sticks = 259 := by
  sorry

end raft_sticks_total_l905_90573


namespace roots_relation_l905_90552

theorem roots_relation (n r : ℝ) (c d : ℝ) : 
  (c^2 - n*c + 3 = 0) → 
  (d^2 - n*d + 3 = 0) → 
  ((c + 1/d)^2 - r*(c + 1/d) + s = 0) → 
  ((d + 1/c)^2 - r*(d + 1/c) + s = 0) → 
  s = 16/3 := by sorry

end roots_relation_l905_90552


namespace mame_probability_theorem_l905_90543

/-- Represents a piece of paper with 8 possible surfaces (4 on each side) -/
structure Paper :=
  (surfaces : Fin 8)

/-- The probability of a specific surface being on top -/
def probability_on_top (paper : Paper) : ℚ := 1 / 8

/-- The surface with "MAME" written on it -/
def mame_surface : Fin 8 := 0

theorem mame_probability_theorem :
  probability_on_top { surfaces := mame_surface } = 1 / 8 := by
  sorry

end mame_probability_theorem_l905_90543


namespace min_colors_theorem_l905_90561

def is_multiple (m n : ℕ) : Prop := ∃ k : ℕ, m = k * n

def valid_coloring (f : ℕ → ℕ) : Prop :=
  ∀ m n, 2 ≤ n ∧ n < m ∧ m ≤ 31 → is_multiple m n → f m ≠ f n

theorem min_colors_theorem :
  ∃ (k : ℕ) (f : ℕ → ℕ),
    (∀ n, 2 ≤ n ∧ n ≤ 31 → f n < k) ∧
    valid_coloring f ∧
    (∀ k' < k, ¬∃ f', (∀ n, 2 ≤ n ∧ n ≤ 31 → f' n < k') ∧ valid_coloring f') ∧
    k = 4 :=
sorry

end min_colors_theorem_l905_90561


namespace tunnel_length_l905_90558

/-- The length of a tunnel given train parameters -/
theorem tunnel_length (train_length : ℝ) (time_diff : ℝ) (train_speed : ℝ) :
  train_length = 2 →
  time_diff = 4 →
  train_speed = 30 →
  train_length = train_speed * time_diff / 60 := by
  sorry

#check tunnel_length

end tunnel_length_l905_90558


namespace hand_count_theorem_l905_90536

def special_deck_size : ℕ := 60
def hand_size : ℕ := 12

def number_of_hands : ℕ := Nat.choose special_deck_size hand_size

theorem hand_count_theorem (C : ℕ) (h : C < 10) :
  ∃ (B : ℕ), number_of_hands = 192 * (10^6) + B * (10^5) + C * (10^4) + 3210 :=
by sorry

end hand_count_theorem_l905_90536


namespace inscribed_triangle_area_bound_l905_90540

/-- A convex polygon with area 1 -/
structure ConvexPolygon where
  area : ℝ
  isConvex : Bool
  area_eq_one : area = 1
  is_convex : isConvex = true

/-- A triangle inscribed in a convex polygon -/
structure InscribedTriangle (p : ConvexPolygon) where
  area : ℝ
  is_inscribed : Bool

/-- Theorem: Any convex polygon with area 1 contains a triangle with area at least 3/8 -/
theorem inscribed_triangle_area_bound (p : ConvexPolygon) : 
  ∃ (t : InscribedTriangle p), t.area ≥ 3/8 := by
  sorry

end inscribed_triangle_area_bound_l905_90540


namespace man_mass_on_boat_l905_90544

/-- The mass of a man causing a boat to sink by a certain depth --/
def mass_of_man (boat_length boat_breadth sinking_depth water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * sinking_depth * water_density

/-- Theorem stating the mass of the man in the given problem --/
theorem man_mass_on_boat : 
  let boat_length : ℝ := 3
  let boat_breadth : ℝ := 2
  let sinking_depth : ℝ := 0.018  -- 1.8 cm converted to meters
  let water_density : ℝ := 1000
  mass_of_man boat_length boat_breadth sinking_depth water_density = 108 := by
  sorry

end man_mass_on_boat_l905_90544


namespace oranges_remaining_l905_90599

theorem oranges_remaining (initial_oranges removed_oranges : ℕ) 
  (h1 : initial_oranges = 96)
  (h2 : removed_oranges = 45) :
  initial_oranges - removed_oranges = 51 := by
sorry

end oranges_remaining_l905_90599


namespace must_divide_five_l905_90510

theorem must_divide_five (a b c d : ℕ+) 
  (h1 : Nat.gcd a b = 40)
  (h2 : Nat.gcd b c = 45)
  (h3 : Nat.gcd c d = 75)
  (h4 : 120 < Nat.gcd d a ∧ Nat.gcd d a < 150) :
  5 ∣ a := by
  sorry

end must_divide_five_l905_90510


namespace ring_price_is_7_10_l905_90541

/-- Represents the sales at a craft fair -/
structure CraftFairSales where
  necklace_price : ℝ
  earrings_price : ℝ
  ring_price : ℝ
  necklaces_sold : ℕ
  rings_sold : ℕ
  earrings_sold : ℕ
  bracelets_sold : ℕ
  total_sales : ℝ

/-- The cost of a bracelet is twice the cost of a ring -/
def bracelet_price (sales : CraftFairSales) : ℝ := 2 * sales.ring_price

/-- Theorem stating that the ring price is $7.10 given the conditions -/
theorem ring_price_is_7_10 (sales : CraftFairSales) 
  (h1 : sales.necklace_price = 12)
  (h2 : sales.earrings_price = 10)
  (h3 : sales.necklaces_sold = 4)
  (h4 : sales.rings_sold = 8)
  (h5 : sales.earrings_sold = 5)
  (h6 : sales.bracelets_sold = 6)
  (h7 : sales.total_sales = 240)
  (h8 : sales.necklace_price * sales.necklaces_sold + 
        sales.ring_price * sales.rings_sold + 
        sales.earrings_price * sales.earrings_sold + 
        bracelet_price sales * sales.bracelets_sold = sales.total_sales) :
  sales.ring_price = 7.1 := by
  sorry


end ring_price_is_7_10_l905_90541


namespace last_digit_largest_power_of_3_dividing_27_factorial_l905_90545

/-- The largest power of 3 that divides n! -/
def largestPowerOf3DividingFactorial (n : ℕ) : ℕ :=
  sorry

/-- The last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ :=
  n % 10

theorem last_digit_largest_power_of_3_dividing_27_factorial :
  lastDigit (3^(largestPowerOf3DividingFactorial 27)) = 3 := by
  sorry

end last_digit_largest_power_of_3_dividing_27_factorial_l905_90545


namespace beverly_bottle_caps_l905_90586

/-- The number of groups of bottle caps in Beverly's collection -/
def num_groups : ℕ := 7

/-- The number of bottle caps in each group -/
def caps_per_group : ℕ := 5

/-- The total number of bottle caps in Beverly's collection -/
def total_caps : ℕ := num_groups * caps_per_group

theorem beverly_bottle_caps : total_caps = 35 := by
  sorry

end beverly_bottle_caps_l905_90586


namespace parabola_c_range_l905_90526

/-- The range of c for a parabola with specific properties -/
theorem parabola_c_range (b c : ℝ) : 
  (∀ x, x^2 + b*x + c = 0 → -1 < x ∧ x < 3) →  -- roots within (-1, 3)
  (∃ x, -1 < x ∧ x < 3 ∧ x^2 + b*x + c = 0 ∧ -x^2 - b*x - c = 0) →  -- equal roots exist
  b = -4 →  -- axis of symmetry at x = 2
  (-5 < c ∧ c ≤ 3) ∨ c = 4 :=
by sorry

end parabola_c_range_l905_90526


namespace trouser_discount_proof_l905_90566

/-- The final percent decrease in price for a trouser with given original price and discount -/
def final_percent_decrease (original_price discount_percent : ℝ) : ℝ :=
  discount_percent

theorem trouser_discount_proof (original_price discount_percent : ℝ) 
  (h1 : original_price = 100)
  (h2 : discount_percent = 30) :
  final_percent_decrease original_price discount_percent = 30 := by
  sorry

#eval final_percent_decrease 100 30

end trouser_discount_proof_l905_90566
