import Mathlib

namespace sum_of_i_powers_l4122_412228

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^13 + i^18 + i^23 + i^28 + i^33 = i := by
  sorry

end sum_of_i_powers_l4122_412228


namespace multiply_mixed_number_l4122_412210

theorem multiply_mixed_number : 8 * (12 + 2/5) = 99 + 1/5 := by
  sorry

end multiply_mixed_number_l4122_412210


namespace rectangle_placement_l4122_412270

theorem rectangle_placement (a b c d : ℝ) 
  (h1 : a < c) (h2 : c < d) (h3 : d < b) (h4 : a * b < c * d) :
  (∃ (θ : ℝ), 0 < θ ∧ θ < π / 2 ∧ 
    b * Real.cos θ + a * Real.sin θ ≤ c ∧
    b * Real.sin θ + a * Real.cos θ ≤ d) ↔ 
  (b^2 - a^2)^2 ≤ (b*d - a*c)^2 + (b*c - a*d)^2 := by sorry

end rectangle_placement_l4122_412270


namespace rope_purchase_difference_l4122_412277

def inches_per_foot : ℕ := 12

def last_week_purchase : ℕ := 6

def this_week_purchase_inches : ℕ := 96

theorem rope_purchase_difference :
  last_week_purchase - (this_week_purchase_inches / inches_per_foot) = 2 :=
by sorry

end rope_purchase_difference_l4122_412277


namespace peach_difference_l4122_412282

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 14

/-- The number of peaches Jill has -/
def jill_peaches : ℕ := 5

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - 6

/-- Jake has more peaches than Jill -/
axiom jake_more_than_jill : jake_peaches > jill_peaches

theorem peach_difference : jake_peaches - jill_peaches = 3 := by
  sorry

end peach_difference_l4122_412282


namespace line_passes_through_point_min_length_AB_min_dot_product_l4122_412248

-- Define the line l: mx + y - 1 - 2m = 0
def line_l (m : ℝ) (x y : ℝ) : Prop := m * x + y - 1 - 2 * m = 0

-- Define the circle O: x^2 + y^2 = r^2
def circle_O (r : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = r^2

-- Theorem 1: The line l passes through the point (2, 1) for all m
theorem line_passes_through_point (m : ℝ) : line_l m 2 1 := by sorry

-- Theorem 2: When r = 4, the minimum length of AB is 2√11
theorem min_length_AB (A B : ℝ × ℝ) 
  (hA : circle_O 4 A.1 A.2) (hB : circle_O 4 B.1 B.2) 
  (hl : ∃ m : ℝ, line_l m A.1 A.2 ∧ line_l m B.1 B.2) :
  ∃ min_length : ℝ, min_length = 2 * Real.sqrt 11 ∧ 
  ∀ m : ℝ, line_l m A.1 A.2 → line_l m B.1 B.2 → 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ min_length := by sorry

-- Theorem 3: When r = 4, the minimum value of OA · OB is -16
theorem min_dot_product (A B : ℝ × ℝ) 
  (hA : circle_O 4 A.1 A.2) (hB : circle_O 4 B.1 B.2) 
  (hl : ∃ m : ℝ, line_l m A.1 A.2 ∧ line_l m B.1 B.2) :
  ∃ min_dot : ℝ, min_dot = -16 ∧ 
  ∀ m : ℝ, line_l m A.1 A.2 → line_l m B.1 B.2 → 
  A.1 * B.1 + A.2 * B.2 ≥ min_dot := by sorry

end line_passes_through_point_min_length_AB_min_dot_product_l4122_412248


namespace diamonds_in_F_10_l4122_412211

/-- Number of diamonds in figure F_n -/
def diamonds (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 1 + 3 * (n * (n - 1) / 2)

/-- Theorem stating that F_10 contains 136 diamonds -/
theorem diamonds_in_F_10 : diamonds 10 = 136 := by
  sorry

end diamonds_in_F_10_l4122_412211


namespace equation_solution_l4122_412214

theorem equation_solution : ∃! x : ℝ, 4*x - 2*x + 1 - 3 = 0 :=
by
  use 1
  constructor
  · -- Prove that 1 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end equation_solution_l4122_412214


namespace parallel_vector_scalar_l4122_412256

/-- Given vectors a, b, and c in ℝ², prove that if a + kb is parallel to c, then k = 1/2 -/
theorem parallel_vector_scalar (a b c : ℝ × ℝ) (h : a = (2, -1) ∧ b = (1, 1) ∧ c = (-5, 1)) :
  (∃ k : ℝ, (a.1 + k * b.1, a.2 + k * b.2).1 * c.2 = (a.1 + k * b.1, a.2 + k * b.2).2 * c.1) →
  (∃ k : ℝ, k = 1/2) :=
by sorry

end parallel_vector_scalar_l4122_412256


namespace inverse_arcsin_function_l4122_412254

theorem inverse_arcsin_function (f : ℝ → ℝ) (h : ∀ x, f x = Real.arcsin (2 * x + 1)) :
  f⁻¹ (π / 6) = -1 / 4 := by
  sorry

end inverse_arcsin_function_l4122_412254


namespace smallest_n_congruence_l4122_412299

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 
  (634 * n ≡ 1275 * n [ZMOD 30]) ∧ 
  (∀ (m : ℕ), m > 0 → (634 * m ≡ 1275 * m [ZMOD 30]) → n ≤ m) ∧ 
  n = 30 := by
  sorry

end smallest_n_congruence_l4122_412299


namespace fraction_evaluation_l4122_412262

theorem fraction_evaluation : (3020 - 2890)^2 / 196 = 86 := by sorry

end fraction_evaluation_l4122_412262


namespace avg_sq_feet_per_person_approx_l4122_412271

/-- The population of the United States -/
def us_population : ℕ := 226504825

/-- The area of the United States in square miles -/
def us_area_sq_miles : ℕ := 3615122

/-- The number of square feet in a square mile -/
def sq_feet_per_sq_mile : ℕ := 5280 * 5280

/-- The average square feet per person in the United States -/
def avg_sq_feet_per_person : ℚ :=
  (us_area_sq_miles * sq_feet_per_sq_mile : ℚ) / us_population

/-- Theorem stating that the average square feet per person is approximately 500000 -/
theorem avg_sq_feet_per_person_approx :
  ∃ ε > 0, abs (avg_sq_feet_per_person - 500000) < ε := by
  sorry

end avg_sq_feet_per_person_approx_l4122_412271


namespace sum_of_first_89_l4122_412257

/-- The sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: The sum of the first 89 natural numbers is 4005 -/
theorem sum_of_first_89 : sum_of_first_n 89 = 4005 := by
  sorry

end sum_of_first_89_l4122_412257


namespace bisecting_line_slope_intercept_sum_l4122_412225

/-- Triangle XYZ with vertices X(1, 9), Y(3, 1), and Z(9, 1) -/
structure Triangle where
  X : ℝ × ℝ := (1, 9)
  Y : ℝ × ℝ := (3, 1)
  Z : ℝ × ℝ := (9, 1)

/-- A line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The line that bisects the area of the triangle -/
def bisectingLine (t : Triangle) : Line :=
  sorry

theorem bisecting_line_slope_intercept_sum (t : Triangle) :
  (bisectingLine t).slope + (bisectingLine t).yIntercept = -3 := by
  sorry

end bisecting_line_slope_intercept_sum_l4122_412225


namespace simplify_fraction_l4122_412253

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) :
  (x + 2) / (x^2 - 2*x) / ((8*x / (x - 2)) + x - 2) = 1 / (x * (x + 2)) := by
  sorry

end simplify_fraction_l4122_412253


namespace age_ratio_proof_l4122_412265

/-- Given the ages of three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  a + b + c = 17 →  -- total age is 17
  b = 6 →  -- b is 6 years old
  b = 2 * c  -- the ratio of b's age to c's age is 2:1
  := by sorry

end age_ratio_proof_l4122_412265


namespace sqrt_sum_inequality_l4122_412220

theorem sqrt_sum_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  Real.sqrt (x / 2) + Real.sqrt (y / 2) ≤ Real.sqrt (x + y) := by
  sorry

end sqrt_sum_inequality_l4122_412220


namespace positive_real_inequality_l4122_412280

theorem positive_real_inequality (x : ℝ) (hx : x > 0) :
  x + 1 / x ≥ 2 ∧ (x + 1 / x = 2 ↔ x = 1) := by
  sorry

end positive_real_inequality_l4122_412280


namespace number_problem_l4122_412212

theorem number_problem (x : ℝ) : 
  3 - (1/4 * 2) - (1/3 * 3) - (1/7 * x) = 27 → 
  (10/100) * x = 17.85 := by
sorry

end number_problem_l4122_412212


namespace circle_equation_l4122_412245

/-- The equation of a circle with center (1,1) passing through the origin (0,0) -/
theorem circle_equation : 
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), (x - 1)^2 + (y - 1)^2 = r^2 ∧ 0^2 + 0^2 = r^2) → 
  (x - 1)^2 + (y - 1)^2 = 2 :=
by sorry

end circle_equation_l4122_412245


namespace right_triangle_solution_l4122_412289

-- Define the right triangle
def RightTriangle (a b c h : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ h > 0 ∧ a^2 + b^2 = c^2 ∧ h^2 = (a^2 * b^2) / c^2

-- Define the conditions
def TriangleConditions (a b c h e d : ℝ) : Prop :=
  RightTriangle a b c h ∧ 
  (c^2 / (2*h) - h/2 = e) ∧  -- Difference between hypotenuse segments
  (a - b = d)                -- Difference between legs

-- Theorem statement
theorem right_triangle_solution (e d : ℝ) (he : e = 37.0488) (hd : d = 31) :
  ∃ (a b c h : ℝ), TriangleConditions a b c h e d ∧ 
    (a = 40 ∧ b = 9 ∧ c = 41) := by
  sorry

end right_triangle_solution_l4122_412289


namespace happy_boys_count_l4122_412235

theorem happy_boys_count (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (neutral_children : ℕ) (total_boys : ℕ) (total_girls : ℕ) (sad_girls : ℕ) 
  (neutral_boys : ℕ) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  total_boys = 16 →
  total_girls = 44 →
  sad_girls = 4 →
  neutral_boys = 4 →
  ∃ (happy_boys : ℕ), happy_boys > 0 →
  happy_boys = 6 :=
by sorry

end happy_boys_count_l4122_412235


namespace line_through_point_with_equal_intercepts_l4122_412290

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def has_equal_intercepts (l : Line) : Prop :=
  l.a = l.b ∨ l.a = -l.b

theorem line_through_point_with_equal_intercepts :
  ∃ (l1 l2 : Line),
    (passes_through l1 ⟨1, 2⟩ ∧ has_equal_intercepts l1) ∧
    (passes_through l2 ⟨1, 2⟩ ∧ has_equal_intercepts l2) ∧
    ((l1.a = 1 ∧ l1.b = 1 ∧ l1.c = -3) ∨ (l2.a = 2 ∧ l2.b = -1 ∧ l2.c = 0)) :=
sorry

end line_through_point_with_equal_intercepts_l4122_412290


namespace modular_inverse_of_7_mod_31_l4122_412215

theorem modular_inverse_of_7_mod_31 : ∃ x : ℕ, x ≤ 30 ∧ (7 * x) % 31 = 1 :=
by
  use 9
  sorry

end modular_inverse_of_7_mod_31_l4122_412215


namespace composite_8n_plus_3_l4122_412269

theorem composite_8n_plus_3 (n : ℕ) (x y : ℕ) 
  (h1 : 8 * n + 1 = x^2) 
  (h2 : 24 * n + 1 = y^2) 
  (h3 : n > 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 8 * n + 3 = a * b := by
  sorry

end composite_8n_plus_3_l4122_412269


namespace system_of_equations_solution_l4122_412295

theorem system_of_equations_solution (x y z : ℤ) 
  (eq1 : x + y + z = 600)
  (eq2 : x - y = 200)
  (eq3 : x + z = 500) :
  x = 300 ∧ y = 100 ∧ z = 200 :=
by sorry

end system_of_equations_solution_l4122_412295


namespace max_area_of_divided_rectangle_l4122_412243

/-- Given a large rectangle divided into 8 smaller rectangles with specific perimeters,
    prove that its maximum area is 512 square centimeters. -/
theorem max_area_of_divided_rectangle :
  ∀ (pA pB pC pD pE : ℝ) (area : ℝ → ℝ),
  pA = 26 →
  pB = 28 →
  pC = 30 →
  pD = 32 →
  pE = 34 →
  (∀ x, area x ≤ 512) →
  (∃ x, area x = 512) :=
by sorry

end max_area_of_divided_rectangle_l4122_412243


namespace remainder_theorem_l4122_412276

theorem remainder_theorem (P K Q R K' Q' S' T : ℕ) 
  (h1 : P = K * Q + R)
  (h2 : Q = K' * Q' + S')
  (h3 : R * Q' = T)
  (h4 : Q' ≠ 0) :
  P % (K * K') = K * S' + T / Q' :=
sorry

end remainder_theorem_l4122_412276


namespace incorrect_division_result_l4122_412266

theorem incorrect_division_result (dividend : ℕ) :
  dividend / 36 = 32 →
  dividend / 48 = 24 :=
by
  sorry

end incorrect_division_result_l4122_412266


namespace function_increasing_implies_omega_bound_l4122_412247

theorem function_increasing_implies_omega_bound 
  (ω : ℝ) 
  (h_pos : ω > 0)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = (1/2) * Real.sin (ω * x / 2) * Real.cos (ω * x / 2))
  (h_increasing : StrictMonoOn f (Set.Icc (-π/3) (π/4))) :
  ω ≤ 3/2 :=
sorry

end function_increasing_implies_omega_bound_l4122_412247


namespace expand_and_simplify_l4122_412202

theorem expand_and_simplify (a : ℝ) : 3*a*(2*a^2 - 4*a) - 2*a^2*(3*a + 4) = -20*a^2 := by
  sorry

end expand_and_simplify_l4122_412202


namespace sum_two_digit_integers_mod_1000_l4122_412259

/-- The sum of all four-digit integers formed using exactly two different digits -/
def S : ℕ := sorry

/-- Theorem stating that S mod 1000 = 370 -/
theorem sum_two_digit_integers_mod_1000 : S % 1000 = 370 := by sorry

end sum_two_digit_integers_mod_1000_l4122_412259


namespace students_like_both_desserts_l4122_412233

/-- Proves the number of students who like both apple pie and chocolate cake -/
theorem students_like_both_desserts 
  (total : ℕ) 
  (like_apple : ℕ) 
  (like_chocolate : ℕ) 
  (like_neither : ℕ) 
  (h1 : total = 50)
  (h2 : like_apple = 22)
  (h3 : like_chocolate = 20)
  (h4 : like_neither = 15) :
  total - like_neither - (like_apple + like_chocolate - (total - like_neither)) = 7 := by
  sorry

end students_like_both_desserts_l4122_412233


namespace sequence_type_l4122_412226

theorem sequence_type (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = 2 * n^2 - 2 * n) → 
  (∃ d : ℝ, d = 4 ∧ ∀ n, a (n + 1) - a n = d) := by
sorry

end sequence_type_l4122_412226


namespace basketball_shot_probability_l4122_412230

theorem basketball_shot_probability :
  let p_at_least_one : ℝ := 0.9333333333333333
  let p_free_throw : ℝ := 4/5
  let p_high_school : ℝ := 1/2
  let p_pro : ℝ := 1/3
  (1 - (1 - p_free_throw) * (1 - p_high_school) * (1 - p_pro) = p_at_least_one) :=
by sorry

end basketball_shot_probability_l4122_412230


namespace marbles_remaining_l4122_412242

theorem marbles_remaining (initial : ℝ) (lost : ℝ) (given_away : ℝ) (found : ℝ) : 
  initial = 150 → 
  lost = 58.5 → 
  given_away = 37.2 → 
  found = 10.8 → 
  initial - lost - given_away + found = 65.1 := by
sorry

end marbles_remaining_l4122_412242


namespace fraction_of_x_l4122_412236

theorem fraction_of_x (w x y f : ℝ) : 
  2/w + f*x = 2/y → 
  w*x = y → 
  (w + x)/2 = 0.5 → 
  f = 2/x - 2 := by
sorry

end fraction_of_x_l4122_412236


namespace tangent_line_intersection_product_l4122_412283

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = x / 2
def asymptote2 (x y : ℝ) : Prop := y = -x / 2

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop := hyperbola P.1 P.2

-- Define a line tangent to the hyperbola at point P
def tangent_line (l : ℝ × ℝ → Prop) (P : ℝ × ℝ) : Prop :=
  point_on_hyperbola P ∧ l P

-- Define intersection points of the tangent line with asymptotes
def intersection_points (l : ℝ × ℝ → Prop) (M N : ℝ × ℝ) : Prop :=
  l M ∧ l N ∧ asymptote1 M.1 M.2 ∧ asymptote2 N.1 N.2

-- Theorem statement
theorem tangent_line_intersection_product (P M N : ℝ × ℝ) (l : ℝ × ℝ → Prop) :
  point_on_hyperbola P →
  tangent_line l P →
  intersection_points l M N →
  M.1 * N.1 + M.2 * N.2 = 3 := by sorry

end tangent_line_intersection_product_l4122_412283


namespace complex_fraction_power_l4122_412213

theorem complex_fraction_power (i : ℂ) : i * i = -1 → (((1 + i) / (1 - i)) ^ 2018 : ℂ) = -1 := by
  sorry

end complex_fraction_power_l4122_412213


namespace notebook_cost_is_50_l4122_412279

def mean_expenditure : ℝ := 500
def num_days : ℕ := 7
def other_days_expenditure : List ℝ := [450, 600, 400, 500, 550, 300]
def pen_cost : ℝ := 30
def earphone_cost : ℝ := 620

def total_week_expenditure : ℝ := mean_expenditure * num_days
def other_days_total : ℝ := other_days_expenditure.sum
def friday_expenditure : ℝ := total_week_expenditure - other_days_total

theorem notebook_cost_is_50 :
  friday_expenditure - (pen_cost + earphone_cost) = 50 := by
  sorry

end notebook_cost_is_50_l4122_412279


namespace pentagon_tiles_18gon_l4122_412273

-- Define the pentagon
structure Pentagon where
  side_length : ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  angle5 : ℝ
  sum_of_angles : angle1 + angle2 + angle3 + angle4 + angle5 = 540

-- Define the regular 18-gon
structure Regular18Gon where
  side_length : ℝ
  interior_angle : ℝ
  interior_angle_eq : interior_angle = 160

-- Theorem statement
theorem pentagon_tiles_18gon (c : ℝ) (h : c > 0) :
  ∃ (p : Pentagon) (g : Regular18Gon),
    p.side_length = c ∧
    g.side_length = c ∧
    p.angle1 = 60 ∧
    p.angle2 = 160 ∧
    p.angle3 = 80 ∧
    p.angle4 = 100 ∧
    p.angle5 = 140 ∧
    (∃ (n : ℕ), n = 18 ∧ n * p.side_length = 18 * g.side_length) :=
  sorry

end pentagon_tiles_18gon_l4122_412273


namespace farm_tax_calculation_l4122_412244

/-- The farm tax calculation problem -/
theorem farm_tax_calculation 
  (tax_percentage : Real) 
  (total_tax_collected : Real) 
  (willam_land_percentage : Real) : 
  tax_percentage = 0.4 →
  total_tax_collected = 3840 →
  willam_land_percentage = 0.3125 →
  willam_land_percentage * (total_tax_collected / tax_percentage) = 3000 := by
  sorry

#check farm_tax_calculation

end farm_tax_calculation_l4122_412244


namespace sqrt_equality_implies_t_values_l4122_412274

theorem sqrt_equality_implies_t_values (t : ℝ) :
  (Real.sqrt (5 * Real.sqrt (t - 5)) = (10 - t + t^2)^(1/4)) →
  (t = 13 + Real.sqrt 34 ∨ t = 13 - Real.sqrt 34) :=
by sorry

end sqrt_equality_implies_t_values_l4122_412274


namespace shoes_to_sell_l4122_412208

def monthly_goal : ℕ := 80
def sold_last_week : ℕ := 27
def sold_this_week : ℕ := 12

theorem shoes_to_sell : monthly_goal - (sold_last_week + sold_this_week) = 41 := by
  sorry

end shoes_to_sell_l4122_412208


namespace partial_fraction_coefficient_sum_l4122_412278

theorem partial_fraction_coefficient_sum :
  ∀ (A B C D E : ℝ),
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5)) →
  A + B + C + D + E = 0 :=
by
  sorry

end partial_fraction_coefficient_sum_l4122_412278


namespace product_of_integers_l4122_412204

theorem product_of_integers (p q r : ℕ+) : 
  p + q + r = 30 → 
  (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 420 / (p * q * r) = 1 → 
  p * q * r = 1800 := by
sorry

end product_of_integers_l4122_412204


namespace adjacent_sum_theorem_l4122_412209

/-- Represents a 3x3 table with numbers from 1 to 9 -/
def Table := Fin 3 → Fin 3 → Fin 9

/-- Checks if a table contains each number from 1 to 9 exactly once -/
def is_valid (t : Table) : Prop :=
  ∀ n : Fin 9, ∃! (i j : Fin 3), t i j = n

/-- Checks if the table has 1, 2, 3, and 4 in the correct positions -/
def correct_positions (t : Table) : Prop :=
  t 0 0 = 0 ∧ t 2 0 = 1 ∧ t 0 2 = 2 ∧ t 2 2 = 3

/-- Returns the sum of adjacent numbers to the given position -/
def adjacent_sum (t : Table) (i j : Fin 3) : ℕ :=
  (if i > 0 then (t (i-1) j).val + 1 else 0) +
  (if i < 2 then (t (i+1) j).val + 1 else 0) +
  (if j > 0 then (t i (j-1)).val + 1 else 0) +
  (if j < 2 then (t i (j+1)).val + 1 else 0)

/-- The main theorem -/
theorem adjacent_sum_theorem (t : Table) :
  is_valid t →
  correct_positions t →
  (∃ i j : Fin 3, t i j = 4 ∧ adjacent_sum t i j = 9) →
  (∃ i j : Fin 3, t i j = 5 ∧ adjacent_sum t i j = 29) :=
by sorry

end adjacent_sum_theorem_l4122_412209


namespace sum_of_three_numbers_l4122_412223

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum1 : a + b = 35)
  (sum2 : b + c = 42)
  (sum3 : c + a = 58) :
  a + b + c = 67.5 := by
  sorry

end sum_of_three_numbers_l4122_412223


namespace quadratic_completed_square_l4122_412267

theorem quadratic_completed_square (b : ℝ) (m : ℝ) :
  (∀ x, x^2 + b*x + 1/6 = (x + m)^2 + 1/12) → b = -Real.sqrt 3 / 3 := by
  sorry

end quadratic_completed_square_l4122_412267


namespace total_buttons_is_1600_l4122_412296

/-- The number of 3-button shirts ordered -/
def shirts_3_button : ℕ := 200

/-- The number of 5-button shirts ordered -/
def shirts_5_button : ℕ := 200

/-- The number of buttons on a 3-button shirt -/
def buttons_per_3_button_shirt : ℕ := 3

/-- The number of buttons on a 5-button shirt -/
def buttons_per_5_button_shirt : ℕ := 5

/-- The total number of buttons used for all shirts -/
def total_buttons : ℕ := shirts_3_button * buttons_per_3_button_shirt + 
                          shirts_5_button * buttons_per_5_button_shirt

theorem total_buttons_is_1600 : total_buttons = 1600 := by
  sorry

end total_buttons_is_1600_l4122_412296


namespace ellipse_circle_tangent_contained_l4122_412275

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_a_ge_b : a ≥ b

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_pos_r : 0 < r

/-- Check if a point (x, y) is on the ellipse -/
def Ellipse.contains (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Check if a point (x, y) is on or inside the circle -/
def Circle.contains (c : Circle) (x y : ℝ) : Prop :=
  (x - c.h)^2 + (y - c.k)^2 ≤ c.r^2

/-- Check if the circle is tangent to the ellipse -/
def is_tangent (e : Ellipse) (c : Circle) : Prop :=
  ∃ x y : ℝ, e.contains x y ∧ (x - c.h)^2 + (y - c.k)^2 = c.r^2 ∧
    ∀ x' y' : ℝ, e.contains x' y' → (x' - c.h)^2 + (y' - c.k)^2 ≥ c.r^2

/-- Check if the circle is entirely contained within the ellipse -/
def is_contained (e : Ellipse) (c : Circle) : Prop :=
  ∀ x y : ℝ, c.contains x y → e.contains x y

/-- Main theorem: The circle with radius 2 centered at a focus of the ellipse
    is tangent to the ellipse and contained within it -/
theorem ellipse_circle_tangent_contained (e : Ellipse) (c : Circle)
    (h_e : e.a = 6 ∧ e.b = 5)
    (h_c : c.h = Real.sqrt 11 ∧ c.k = 0 ∧ c.r = 2) :
    is_tangent e c ∧ is_contained e c := by
  sorry

end ellipse_circle_tangent_contained_l4122_412275


namespace add_three_preserves_inequality_l4122_412232

theorem add_three_preserves_inequality (a b : ℝ) : a > b → a + 3 > b + 3 := by
  sorry

end add_three_preserves_inequality_l4122_412232


namespace square_side_ratio_sum_l4122_412222

theorem square_side_ratio_sum (area_ratio : ℚ) : 
  area_ratio = 128 / 50 →
  ∃ (p q r : ℕ), 
    (p * Real.sqrt q : ℝ) / r = Real.sqrt (area_ratio) ∧
    p + q + r = 14 :=
by sorry

end square_side_ratio_sum_l4122_412222


namespace airplane_hovering_time_l4122_412252

/-- Calculates the total hovering time for an airplane over two days -/
theorem airplane_hovering_time 
  (mountain_day1 : ℕ) 
  (central_day1 : ℕ) 
  (eastern_day1 : ℕ) 
  (additional_time : ℕ) 
  (h1 : mountain_day1 = 3)
  (h2 : central_day1 = 4)
  (h3 : eastern_day1 = 2)
  (h4 : additional_time = 2) :
  mountain_day1 + central_day1 + eastern_day1 + 
  (mountain_day1 + additional_time) + 
  (central_day1 + additional_time) + 
  (eastern_day1 + additional_time) = 24 := by
sorry


end airplane_hovering_time_l4122_412252


namespace absolute_value_condition_l4122_412268

theorem absolute_value_condition (x : ℝ) : |x - 1| = 1 - x → x ≤ 1 := by
  sorry

end absolute_value_condition_l4122_412268


namespace julie_work_hours_l4122_412287

/-- Given Julie's work schedule and earnings, calculate her required weekly hours during the school year --/
theorem julie_work_hours (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ)
                         (school_year_weeks : ℕ) (school_year_earnings : ℕ) :
  summer_weeks = 12 →
  summer_hours_per_week = 48 →
  summer_earnings = 5000 →
  school_year_weeks = 48 →
  school_year_earnings = 5000 →
  (summer_hours_per_week * summer_weeks * school_year_earnings) / (summer_earnings * school_year_weeks) = 12 := by
  sorry

end julie_work_hours_l4122_412287


namespace minimum_cost_theorem_l4122_412219

/-- Represents the cost and quantity of prizes A and B --/
structure PrizePurchase where
  costA : ℕ  -- Cost of prize A
  costB : ℕ  -- Cost of prize B
  quantityA : ℕ  -- Quantity of prize A
  quantityB : ℕ  -- Quantity of prize B

/-- Conditions for the prize purchase problem --/
def PrizePurchaseConditions (p : PrizePurchase) : Prop :=
  3 * p.costA + 2 * p.costB = 390 ∧  -- Condition 1
  4 * p.costA = 5 * p.costB + 60 ∧  -- Condition 2
  p.quantityA + p.quantityB = 30 ∧  -- Condition 3
  p.quantityA ≥ p.quantityB / 2 ∧  -- Condition 4
  p.costA * p.quantityA + p.costB * p.quantityB ≤ 2170  -- Condition 5

/-- The theorem to be proved --/
theorem minimum_cost_theorem (p : PrizePurchase) 
  (h : PrizePurchaseConditions p) : 
  p.costA = 90 ∧ p.costB = 60 ∧ 
  p.quantityA * p.costA + p.quantityB * p.costB ≥ 2100 :=
sorry

end minimum_cost_theorem_l4122_412219


namespace charlie_feather_collection_l4122_412229

/-- The number of sets of wings Charlie needs to make -/
def num_sets : ℕ := 2

/-- The number of feathers required for each set of wings -/
def feathers_per_set : ℕ := 900

/-- The number of feathers Charlie already has -/
def feathers_collected : ℕ := 387

/-- The total number of additional feathers Charlie needs to collect -/
def additional_feathers_needed : ℕ := num_sets * feathers_per_set - feathers_collected

theorem charlie_feather_collection :
  additional_feathers_needed = 1413 := by sorry

end charlie_feather_collection_l4122_412229


namespace ceiling_times_self_156_l4122_412284

theorem ceiling_times_self_156 :
  ∃! (y : ℝ), ⌈y⌉ * y = 156 :=
by
  -- The proof goes here
  sorry

end ceiling_times_self_156_l4122_412284


namespace total_birds_count_l4122_412206

/-- Proves that given the specified conditions, the total number of birds is 185 -/
theorem total_birds_count (chickens ducks : ℕ) 
  (h1 : ducks = 4 * chickens + 10) 
  (h2 : ducks = 150) : 
  chickens + ducks = 185 := by
  sorry

end total_birds_count_l4122_412206


namespace range_of_a_l4122_412250

/-- Given sets A and B, where "x ∈ B" is a sufficient but not necessary condition for "x ∈ A",
    this theorem proves that the range of values for a is [0, 1]. -/
theorem range_of_a (A B : Set ℝ) (a : ℝ) : 
  A = {x : ℝ | x^2 - x - 2 ≤ 0} →
  B = {x : ℝ | |x - a| ≤ 1} →
  (∀ x, x ∈ B → x ∈ A) →
  ¬(∀ x, x ∈ A → x ∈ B) →
  0 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l4122_412250


namespace exponential_equation_solution_l4122_412293

theorem exponential_equation_solution :
  ∃ x : ℝ, (2 : ℝ) ^ (x + 6) = 64 ^ (x - 1) ∧ x = 2.4 := by
  sorry

end exponential_equation_solution_l4122_412293


namespace f_2013_equals_2_l4122_412205

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def satisfies_recurrence (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 4) = f x + 2 * f 2

theorem f_2013_equals_2 (f : ℝ → ℝ) 
  (h1 : is_even_function f)
  (h2 : satisfies_recurrence f)
  (h3 : f (-1) = 2) :
  f 2013 = 2 := by
  sorry

end f_2013_equals_2_l4122_412205


namespace product_equals_square_l4122_412241

theorem product_equals_square : 
  250 * 9.996 * 3.996 * 500 = (4998 : ℝ)^2 := by
  sorry

end product_equals_square_l4122_412241


namespace f_negative_a_l4122_412272

theorem f_negative_a (a : ℝ) (f : ℝ → ℝ) (h : f = λ x ↦ x^3 * Real.cos x + 1) (h_fa : f a = 11) :
  f (-a) = -9 := by
sorry

end f_negative_a_l4122_412272


namespace marble_difference_l4122_412217

theorem marble_difference (e d : ℕ) (h1 : e > d) (h2 : e = (d - 8) + 30) : e - d = 22 := by
  sorry

end marble_difference_l4122_412217


namespace sqrt_identity_l4122_412201

theorem sqrt_identity (a b : ℝ) (h : a^2 ≥ b ∧ a ≥ 0 ∧ b ≥ 0) :
  (∀ (s : Bool), Real.sqrt (a + (if s then 1 else -1) * Real.sqrt b) = 
    Real.sqrt ((a + Real.sqrt (a^2 - b)) / 2) + 
    (if s then 1 else -1) * Real.sqrt ((a - Real.sqrt (a^2 - b)) / 2)) := by
  sorry

end sqrt_identity_l4122_412201


namespace logarithm_simplification_l4122_412298

theorem logarithm_simplification 
  (m n p q x z : ℝ) 
  (hm : m > 0) (hn : n > 0) (hp : p > 0) (hq : q > 0) (hx : x > 0) (hz : z > 0) : 
  Real.log (m / n) + Real.log (n / p) + Real.log (p / q) - Real.log (m * x / (q * z)) = Real.log (z / x) := by
  sorry

end logarithm_simplification_l4122_412298


namespace unique_solution_system_l4122_412291

theorem unique_solution_system (x y z : ℝ) :
  (x - 1) * (y - 1) * (z - 1) = x * y * z - 1 ∧
  (x - 2) * (y - 2) * (z - 2) = x * y * z - 2 →
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end unique_solution_system_l4122_412291


namespace gcd_1189_264_l4122_412231

theorem gcd_1189_264 : Nat.gcd 1189 264 = 1 := by
  sorry

end gcd_1189_264_l4122_412231


namespace base_9_addition_multiplication_l4122_412240

/-- Converts a number from base 9 to base 10 --/
def base9ToBase10 (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 9^2 + tens * 9 + ones

/-- Converts a number from base 10 to base 9 --/
def base10ToBase9 (n : ℕ) : ℕ :=
  sorry -- Implementation not provided, as it's not required for the statement

theorem base_9_addition_multiplication :
  let a := base9ToBase10 436
  let b := base9ToBase10 782
  let c := base9ToBase10 204
  let d := base9ToBase10 12
  base10ToBase9 ((a + b + c) * d) = 18508 := by
  sorry

end base_9_addition_multiplication_l4122_412240


namespace product_of_fractions_l4122_412238

theorem product_of_fractions : (1 : ℚ) / 3 * 3 / 5 * 5 / 7 * 7 / 9 = 1 / 9 := by
  sorry

end product_of_fractions_l4122_412238


namespace log_ratio_simplification_l4122_412292

theorem log_ratio_simplification (x : ℝ) 
  (h1 : 5 * x^3 > 0) (h2 : 7 * x - 3 > 0) : 
  (Real.log (Real.sqrt (7 * x - 3)) / Real.log (5 * x^3)) / Real.log (7 * x - 3) = 1/2 := by
  sorry

end log_ratio_simplification_l4122_412292


namespace players_who_quit_l4122_412239

theorem players_who_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives_after : ℕ) :
  initial_players = 16 →
  lives_per_player = 8 →
  total_lives_after = 72 →
  initial_players - (total_lives_after / lives_per_player) = 7 :=
by sorry

end players_who_quit_l4122_412239


namespace cost_difference_is_two_point_five_l4122_412285

/-- Represents the pizza sharing scenario between Bob and Samantha -/
structure PizzaSharing where
  totalSlices : ℕ
  plainPizzaCost : ℚ
  oliveCost : ℚ
  bobOliveSlices : ℕ
  bobPlainSlices : ℕ

/-- Calculates the cost difference between Bob and Samantha's payments -/
def costDifference (ps : PizzaSharing) : ℚ :=
  let totalCost := ps.plainPizzaCost + ps.oliveCost
  let costPerSlice := totalCost / ps.totalSlices
  let bobCost := costPerSlice * (ps.bobOliveSlices + ps.bobPlainSlices)
  let samanthaCost := costPerSlice * (ps.totalSlices - ps.bobOliveSlices - ps.bobPlainSlices)
  bobCost - samanthaCost

/-- Theorem stating that the cost difference is $2.5 -/
theorem cost_difference_is_two_point_five :
  let ps : PizzaSharing := {
    totalSlices := 12,
    plainPizzaCost := 12,
    oliveCost := 3,
    bobOliveSlices := 4,
    bobPlainSlices := 3
  }
  costDifference ps = 5/2 := by sorry

end cost_difference_is_two_point_five_l4122_412285


namespace circle_area_decrease_l4122_412200

/-- Given three circles with radii r1, r2, and r3, prove that the decrease in their combined area
    when each radius is reduced by 50% is equal to 75% of their original combined area. -/
theorem circle_area_decrease (r1 r2 r3 : ℝ) (hr1 : r1 > 0) (hr2 : r2 > 0) (hr3 : r3 > 0) :
  let original_area := π * (r1^2 + r2^2 + r3^2)
  let new_area := π * ((r1/2)^2 + (r2/2)^2 + (r3/2)^2)
  original_area - new_area = (3/4) * original_area :=
by sorry

end circle_area_decrease_l4122_412200


namespace line_segments_in_proportion_l4122_412249

theorem line_segments_in_proportion :
  let a : ℝ := 2
  let b : ℝ := Real.sqrt 5
  let c : ℝ := 2 * Real.sqrt 3
  let d : ℝ := Real.sqrt 15
  a * d = b * c := by sorry

end line_segments_in_proportion_l4122_412249


namespace car_speed_l4122_412297

/-- Given a car that travels 390 miles in 6 hours, prove its speed is 65 miles per hour -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 390 ∧ time = 6 ∧ speed = distance / time → speed = 65 := by
  sorry

end car_speed_l4122_412297


namespace sandy_marbles_l4122_412264

/-- The number of marbles in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of red marbles Jessica has -/
def jessica_dozens : ℕ := 3

/-- The number of times more red marbles Sandy has compared to Jessica -/
def sandy_multiplier : ℕ := 4

/-- Theorem stating the number of red marbles Sandy has -/
theorem sandy_marbles : jessica_dozens * dozen * sandy_multiplier = 144 := by
  sorry

end sandy_marbles_l4122_412264


namespace rabbit_farm_number_l4122_412288

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem rabbit_farm_number : 
  ∃! n : ℕ, is_six_digit n ∧ 
            is_perfect_square n ∧ 
            is_perfect_cube n ∧ 
            is_prime (n - 6) ∧ 
            n = 117649 :=
by sorry

end rabbit_farm_number_l4122_412288


namespace base_2_representation_of_125_l4122_412216

theorem base_2_representation_of_125 :
  ∃ (a b c d e f g : ℕ),
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 0 ∧ g = 1) ∧
    125 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end base_2_representation_of_125_l4122_412216


namespace tylenol_dosage_l4122_412234

/-- Calculates the mg per pill given the total dosage and number of pills -/
def mg_per_pill (dosage_mg : ℕ) (dosage_interval_hours : ℕ) (duration_days : ℕ) (total_pills : ℕ) : ℚ :=
  let doses_per_day := 24 / dosage_interval_hours
  let total_doses := doses_per_day * duration_days
  let total_mg := dosage_mg * total_doses
  (total_mg : ℚ) / total_pills

theorem tylenol_dosage :
  mg_per_pill 1000 6 14 112 = 500 := by
  sorry

end tylenol_dosage_l4122_412234


namespace K_bounds_l4122_412255

-- Define the variables and constraints
def K (x y z : ℝ) : ℝ := 5*x - 6*y + 7*z

def constraint1 (x y z : ℝ) : Prop := 4*x + y + 2*z = 4
def constraint2 (x y z : ℝ) : Prop := 3*x + 6*y - 2*z = 6

-- State the theorem
theorem K_bounds :
  ∀ x y z : ℝ,
    x ≥ 0 → y ≥ 0 → z ≥ 0 →
    constraint1 x y z →
    constraint2 x y z →
    -5 ≤ K x y z ∧ K x y z ≤ 7 :=
by
  sorry

end K_bounds_l4122_412255


namespace monotonic_increase_interval_l4122_412221

/-- A power function that passes through the point (3, 9) -/
def f (x : ℝ) : ℝ := x^2

/-- The point (3, 9) lies on the graph of f -/
axiom point_on_graph : f 3 = 9

/-- Theorem: The interval of monotonic increase for f is [0, +∞) -/
theorem monotonic_increase_interval :
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y :=
sorry

end monotonic_increase_interval_l4122_412221


namespace vasily_salary_higher_than_fedor_l4122_412237

/-- Represents the salary distribution for graduates --/
structure SalaryDistribution where
  high : ℝ  -- Proportion earning 60,000 rubles
  very_high : ℝ  -- Proportion earning 80,000 rubles
  low : ℝ  -- Proportion earning 25,000 rubles (not in field)
  medium : ℝ  -- Proportion earning 40,000 rubles

/-- Calculates the expected salary given a salary distribution --/
def expected_salary (dist : SalaryDistribution) : ℝ :=
  60000 * dist.high + 80000 * dist.very_high + 25000 * dist.low + 40000 * dist.medium

/-- Calculates Fedor's salary after a given number of years --/
def fedor_salary (years : ℕ) : ℝ :=
  25000 + 3000 * years

/-- Main theorem statement --/
theorem vasily_salary_higher_than_fedor :
  let total_students : ℝ := 300
  let successful_students : ℝ := 270
  let grad_prob : ℝ := successful_students / total_students
  let salary_dist : SalaryDistribution := {
    high := 1/5,
    very_high := 1/10,
    low := 1/20,
    medium := 1 - (1/5 + 1/10 + 1/20)
  }
  let vasily_expected_salary : ℝ := 
    grad_prob * expected_salary salary_dist + (1 - grad_prob) * 25000
  let fedor_final_salary : ℝ := fedor_salary 4
  vasily_expected_salary = 45025 ∧ 
  vasily_expected_salary - fedor_final_salary = 8025 := by
  sorry


end vasily_salary_higher_than_fedor_l4122_412237


namespace geometric_sequence_ratio_l4122_412218

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h_pos : q > 0) 
  (h_geom : ∀ n, a (n + 1) = q * a n) 
  (h_eq : a 3 * a 9 = 2 * (a 5)^2) : 
  q = Real.sqrt 2 := by sorry

end geometric_sequence_ratio_l4122_412218


namespace cards_not_in_box_l4122_412246

theorem cards_not_in_box (total_cards : ℕ) (cards_per_box : ℕ) (boxes_given : ℕ) (boxes_kept : ℕ) : 
  total_cards = 75 →
  cards_per_box = 10 →
  boxes_given = 2 →
  boxes_kept = 5 →
  total_cards - (cards_per_box * (boxes_given + boxes_kept)) = 5 := by
sorry

end cards_not_in_box_l4122_412246


namespace alpha_beta_cosine_l4122_412258

theorem alpha_beta_cosine (α β : Real)
  (h_α : α ∈ Set.Ioo 0 (π / 3))
  (h_β : β ∈ Set.Ioo (π / 6) (π / 2))
  (eq_α : 5 * Real.sqrt 3 * Real.sin α + 5 * Real.cos α = 8)
  (eq_β : Real.sqrt 2 * Real.sin β + Real.sqrt 6 * Real.cos β = 2) :
  Real.cos (α + π / 6) = 3 / 5 ∧ Real.cos (α + β) = - Real.sqrt 2 / 10 := by
  sorry

end alpha_beta_cosine_l4122_412258


namespace greatest_a_no_integral_solution_l4122_412207

theorem greatest_a_no_integral_solution :
  (∀ a : ℤ, (∀ x : ℤ, ¬(|x + 1| < a - (3/2))) → a ≤ 1) ∧
  (∃ x : ℤ, |x + 1| < 2 - (3/2)) :=
sorry

end greatest_a_no_integral_solution_l4122_412207


namespace tangerine_orange_difference_l4122_412263

def initial_oranges : ℕ := 5
def initial_tangerines : ℕ := 17
def removed_oranges : ℕ := 2
def removed_tangerines : ℕ := 10
def added_oranges : ℕ := 3
def added_tangerines : ℕ := 6

theorem tangerine_orange_difference :
  (initial_tangerines - removed_tangerines + added_tangerines) -
  (initial_oranges - removed_oranges + added_oranges) = 7 := by
sorry

end tangerine_orange_difference_l4122_412263


namespace jeff_score_problem_l4122_412294

theorem jeff_score_problem (scores : List ℝ) (desired_average : ℝ) : 
  scores = [89, 92, 88, 95, 91] → 
  desired_average = 93 → 
  (scores.sum + 103) / 6 = desired_average :=
by sorry

end jeff_score_problem_l4122_412294


namespace inequality_solution_l4122_412227

theorem inequality_solution (x : ℝ) (h : x ≠ 1) :
  1 / (x - 1) ≤ 1 ↔ x < 1 ∨ x ≥ 2 := by
  sorry

end inequality_solution_l4122_412227


namespace solve_equation_l4122_412261

theorem solve_equation (x : ℝ) : (1 / 2) * (1 / 7) * x = 14 → x = 196 := by
  sorry

end solve_equation_l4122_412261


namespace louisa_average_speed_l4122_412224

/-- Proves that given the conditions of Louisa's travel, her average speed was 50 miles per hour -/
theorem louisa_average_speed :
  ∀ (v : ℝ), 
    v > 0 →
    350 / v - 200 / v = 3 →
    v = 50 :=
by
  sorry

end louisa_average_speed_l4122_412224


namespace not_divisible_by_five_l4122_412251

theorem not_divisible_by_five (n : ℤ) : ¬ (5 ∣ (n^2 + n + 1)) := by
  sorry

end not_divisible_by_five_l4122_412251


namespace min_prime_angle_in_linear_pair_l4122_412203

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem min_prime_angle_in_linear_pair (a b : ℕ) :
  a + b = 180 →
  is_prime a →
  is_prime b →
  a > b →
  b ≥ 7 :=
sorry

end min_prime_angle_in_linear_pair_l4122_412203


namespace cinema_seats_l4122_412281

theorem cinema_seats (rows : ℕ) (seats_per_row : ℕ) (h1 : rows = 21) (h2 : seats_per_row = 26) :
  rows * seats_per_row = 546 := by
  sorry

end cinema_seats_l4122_412281


namespace sphere_volume_ratio_l4122_412286

theorem sphere_volume_ratio (r R : ℝ) (h : r > 0) (H : R > 0) :
  (4 * Real.pi * r^2) / (4 * Real.pi * R^2) = 4/9 →
  ((4/3) * Real.pi * r^3) / ((4/3) * Real.pi * R^3) = 8/27 := by
sorry

end sphere_volume_ratio_l4122_412286


namespace quadratic_non_real_roots_l4122_412260

theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
  sorry

end quadratic_non_real_roots_l4122_412260
