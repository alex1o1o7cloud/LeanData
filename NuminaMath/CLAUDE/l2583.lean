import Mathlib

namespace hawks_score_l2583_258377

theorem hawks_score (total_points eagles_points hawks_points : ℕ) : 
  total_points = 82 →
  eagles_points - hawks_points = 18 →
  eagles_points + hawks_points = total_points →
  hawks_points = 32 := by
sorry

end hawks_score_l2583_258377


namespace cube_root_simplification_l2583_258336

theorem cube_root_simplification : Real.rpow (4^6 * 5^3 * 7^3) (1/3) = 560 := by
  sorry

end cube_root_simplification_l2583_258336


namespace f_monotone_and_roots_sum_l2583_258339

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2*x - 2*(a+1)*Real.exp x + a*Real.exp (2*x)

theorem f_monotone_and_roots_sum (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a = 1 ∧
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ = a * Real.exp (2*x₁) → f a x₂ = a * Real.exp (2*x₂) → x₁ + x₂ > 2 :=
sorry

end f_monotone_and_roots_sum_l2583_258339


namespace probability_three_even_dice_l2583_258364

def num_dice : ℕ := 5
def faces_per_die : ℕ := 20
def target_even : ℕ := 3

theorem probability_three_even_dice :
  let p_even : ℚ := 1 / 2  -- Probability of rolling an even number on a single die
  let p_arrangement : ℚ := p_even ^ target_even * (1 - p_even) ^ (num_dice - target_even)
  let num_arrangements : ℕ := Nat.choose num_dice target_even
  num_arrangements * p_arrangement = 5 / 16 := by
sorry

end probability_three_even_dice_l2583_258364


namespace negative_real_inequality_l2583_258319

theorem negative_real_inequality (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  a > b ↔ a - 1 / a > b - 1 / b := by sorry

end negative_real_inequality_l2583_258319


namespace unique_quadratic_m_l2583_258300

def is_quadratic_coefficient (m : ℝ) : Prop :=
  |m| = 2 ∧ m - 2 ≠ 0

theorem unique_quadratic_m :
  ∃! m : ℝ, is_quadratic_coefficient m ∧ m = -2 :=
sorry

end unique_quadratic_m_l2583_258300


namespace quadratic_factor_difference_l2583_258360

/-- Given a quadratic expression that can be factored, prove the difference of its factors' constants -/
theorem quadratic_factor_difference (a b : ℤ) : 
  (∀ y, 2 * y^2 + 5 * y - 12 = (2 * y + a) * (y + b)) → 
  a - b = -7 := by
  sorry

end quadratic_factor_difference_l2583_258360


namespace quadrilateral_interior_angles_sum_l2583_258346

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A quadrilateral is a polygon with 4 sides -/
def is_quadrilateral (n : ℕ) : Prop := n = 4

theorem quadrilateral_interior_angles_sum :
  ∀ n : ℕ, is_quadrilateral n → sum_interior_angles n = 360 := by
  sorry

end quadrilateral_interior_angles_sum_l2583_258346


namespace M_mod_1000_eq_9_l2583_258325

/-- The number of 8-digit positive integers with strictly increasing digits -/
def M : ℕ := Nat.choose 9 8

/-- The theorem stating that M modulo 1000 equals 9 -/
theorem M_mod_1000_eq_9 : M % 1000 = 9 := by
  sorry

end M_mod_1000_eq_9_l2583_258325


namespace flower_bed_area_l2583_258345

theorem flower_bed_area (a b c : ℝ) (ha : a = 6) (hb : b = 8) (hc : c = 10) :
  (1/2) * a * b = 24 := by
  sorry

end flower_bed_area_l2583_258345


namespace output_value_2003_l2583_258352

/-- The annual growth rate of the company's output value -/
def growth_rate : ℝ := 0.10

/-- The initial output value of the company in 2000 (in millions of yuan) -/
def initial_value : ℝ := 10

/-- The number of years between 2000 and 2003 -/
def years : ℕ := 3

/-- The expected output value of the company in 2003 (in millions of yuan) -/
def expected_value : ℝ := 13.31

/-- Theorem stating that the company's output value in 2003 will be 13.31 million yuan -/
theorem output_value_2003 : 
  initial_value * (1 + growth_rate) ^ years = expected_value := by
  sorry

end output_value_2003_l2583_258352


namespace deepak_age_l2583_258387

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 5 / 7 →
  arun_age + 6 = 36 →
  deepak_age = 42 := by
sorry

end deepak_age_l2583_258387


namespace speeding_ticket_percentage_l2583_258361

theorem speeding_ticket_percentage
  (exceed_limit_percent : ℝ)
  (no_ticket_percent : ℝ)
  (h1 : exceed_limit_percent = 14.285714285714285)
  (h2 : no_ticket_percent = 30) :
  (1 - no_ticket_percent / 100) * exceed_limit_percent = 10 :=
by sorry

end speeding_ticket_percentage_l2583_258361


namespace dividend_calculation_l2583_258358

theorem dividend_calculation (remainder quotient divisor dividend : ℕ) : 
  remainder = 5 →
  divisor = 3 * quotient →
  divisor = 3 * remainder + 3 →
  dividend = divisor * quotient + remainder →
  dividend = 113 := by
sorry

end dividend_calculation_l2583_258358


namespace rectangular_plot_breadth_l2583_258332

/-- The breadth of a rectangular plot given its area and length-breadth relationship -/
theorem rectangular_plot_breadth (area : ℝ) (length_ratio : ℝ) : 
  area = 360 → length_ratio = 0.75 → ∃ breadth : ℝ, 
    area = (length_ratio * breadth) * breadth ∧ breadth = 4 * Real.sqrt 30 := by
  sorry

end rectangular_plot_breadth_l2583_258332


namespace min_coach_handshakes_l2583_258399

/-- Represents the total number of handshakes at the event -/
def total_handshakes : ℕ := 300

/-- Calculates the number of athlete handshakes given the total number of athletes -/
def athlete_handshakes (n : ℕ) : ℕ := (3 * n * n) / 4

/-- Calculates the number of coach handshakes given the total number of athletes -/
def coach_handshakes (n : ℕ) : ℕ := n

/-- Theorem stating the minimum number of coach handshakes -/
theorem min_coach_handshakes :
  ∃ n : ℕ, 
    athlete_handshakes n + coach_handshakes n = total_handshakes ∧
    coach_handshakes n = 20 ∧
    ∀ m : ℕ, 
      athlete_handshakes m + coach_handshakes m = total_handshakes →
      coach_handshakes m ≥ coach_handshakes n :=
by sorry

end min_coach_handshakes_l2583_258399


namespace range_of_x_minus_2y_l2583_258393

theorem range_of_x_minus_2y (x y : ℝ) 
  (hx : 30 < x ∧ x < 42) 
  (hy : 16 < y ∧ y < 24) : 
  ∀ z, z ∈ Set.Ioo (-18 : ℝ) 10 ↔ ∃ (x' y' : ℝ), 
    30 < x' ∧ x' < 42 ∧ 
    16 < y' ∧ y' < 24 ∧ 
    z = x' - 2*y' :=
by sorry

end range_of_x_minus_2y_l2583_258393


namespace shaded_area_is_four_point_five_l2583_258317

/-- The area of a shape composed of a large isosceles right triangle and a crescent (lune) -/
theorem shaded_area_is_four_point_five 
  (large_triangle_leg : ℝ) 
  (semicircle_diameter : ℝ) 
  (π : ℝ) 
  (h1 : large_triangle_leg = 2)
  (h2 : semicircle_diameter = 2)
  (h3 : π = 3) : 
  (1/2 * large_triangle_leg * large_triangle_leg) + 
  ((1/2 * π * (semicircle_diameter/2)^2) - (1/2 * (semicircle_diameter/2) * (semicircle_diameter/2))) = 4.5 := by
  sorry

#check shaded_area_is_four_point_five

end shaded_area_is_four_point_five_l2583_258317


namespace equation_solvability_l2583_258351

theorem equation_solvability (n : ℕ) (hn : Odd n) :
  (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 4 / n = 1 / x + 1 / y) ↔
  (∃ d : ℕ, d > 0 ∧ d ∣ n ∧ ∃ k : ℕ, d = 4 * k + 3) :=
by sorry

end equation_solvability_l2583_258351


namespace compound_interest_proof_l2583_258390

/-- The compound interest rate that turns $1200 into $1348.32 in 2 years with annual compounding -/
def compound_interest_rate : ℝ :=
  0.06

theorem compound_interest_proof (initial_sum final_sum : ℝ) (years : ℕ) :
  initial_sum = 1200 →
  final_sum = 1348.32 →
  years = 2 →
  final_sum = initial_sum * (1 + compound_interest_rate) ^ years :=
by sorry

end compound_interest_proof_l2583_258390


namespace sqrt_sum_equals_nine_implies_product_l2583_258330

theorem sqrt_sum_equals_nine_implies_product (x : ℝ) :
  (Real.sqrt (7 + x) + Real.sqrt (28 - x) = 9) →
  ((7 + x) * (28 - x) = 529) := by
sorry

end sqrt_sum_equals_nine_implies_product_l2583_258330


namespace function_equality_implies_a_value_l2583_258397

/-- The function f(x) = x -/
def f (x : ℝ) : ℝ := x

/-- The function g(x) = ax^2 - x, parameterized by a -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x

/-- The theorem stating that under given conditions, a = 3/2 -/
theorem function_equality_implies_a_value :
  ∀ (a : ℝ), a > 0 →
  (∀ x₁ ∈ Set.Icc 1 2, ∃ x₂ ∈ Set.Icc 1 2, f x₁ * f x₂ = g a x₁ * g a x₂) →
  a = 3/2 := by sorry

end function_equality_implies_a_value_l2583_258397


namespace projection_onto_orthogonal_vector_l2583_258347

/-- Given orthogonal vectors a and b in R^2, and the projection of (4, -2) onto a,
    prove that the projection of (4, -2) onto b is (24/5, -2/5). -/
theorem projection_onto_orthogonal_vector 
  (a b : ℝ × ℝ) 
  (h_orthogonal : a.1 * b.1 + a.2 * b.2 = 0) 
  (h_proj_a : (4 : ℝ) * a.1 + (-2 : ℝ) * a.2 = (-4/5 : ℝ) * (a.1^2 + a.2^2)) :
  (4 : ℝ) * b.1 + (-2 : ℝ) * b.2 = (24/5 : ℝ) * (b.1^2 + b.2^2) :=
by sorry

end projection_onto_orthogonal_vector_l2583_258347


namespace percentage_of_50_to_125_l2583_258308

theorem percentage_of_50_to_125 : 
  (50 : ℝ) / 125 * 100 = 40 :=
by sorry

end percentage_of_50_to_125_l2583_258308


namespace fraction_sum_equality_l2583_258318

theorem fraction_sum_equality : 
  let a : ℕ := 1
  let b : ℕ := 6
  let c : ℕ := 7
  let d : ℕ := 3
  let e : ℕ := 5
  let f : ℕ := 2
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
   d ≠ e ∧ d ≠ f ∧
   e ≠ f) →
  (Nat.gcd a b = 1 ∧ Nat.gcd c d = 1 ∧ Nat.gcd e f = 1) →
  (a : ℚ) / b + (c : ℚ) / d = (e : ℚ) / f :=
by sorry

end fraction_sum_equality_l2583_258318


namespace number_line_real_bijection_l2583_258375

-- Define the number line as a type
def NumberLine : Type := ℝ

-- Define a point on the number line
def Point : Type := NumberLine

-- State the theorem
theorem number_line_real_bijection : 
  ∃ f : Point → ℝ, Function.Bijective f :=
sorry

end number_line_real_bijection_l2583_258375


namespace new_shipment_bears_l2583_258388

theorem new_shipment_bears (initial_stock : ℕ) (bears_per_shelf : ℕ) (num_shelves : ℕ) : 
  initial_stock = 6 → bears_per_shelf = 6 → num_shelves = 4 → 
  num_shelves * bears_per_shelf - initial_stock = 18 :=
by
  sorry

end new_shipment_bears_l2583_258388


namespace sum_floor_series_l2583_258338

theorem sum_floor_series (n : ℕ+) :
  (∑' k : ℕ, ⌊(n + 2^k : ℝ) / 2^(k+1)⌋) = n := by sorry

end sum_floor_series_l2583_258338


namespace optimal_price_reduction_l2583_258337

/-- Represents the daily sales and profit of mooncakes -/
structure MooncakeSales where
  initialSales : ℕ
  initialProfit : ℕ
  priceReduction : ℕ
  salesIncrease : ℕ
  targetProfit : ℕ

/-- Calculates the daily profit based on price reduction -/
def dailyProfit (s : MooncakeSales) (x : ℕ) : ℕ :=
  (s.initialProfit - x) * (s.initialSales + (s.salesIncrease * x) / s.priceReduction)

/-- Theorem stating that a 6 yuan price reduction achieves the target profit -/
theorem optimal_price_reduction (s : MooncakeSales) 
    (h1 : s.initialSales = 80)
    (h2 : s.initialProfit = 30)
    (h3 : s.priceReduction = 5)
    (h4 : s.salesIncrease = 20)
    (h5 : s.targetProfit = 2496) :
    dailyProfit s 6 = s.targetProfit := by
  sorry

#check optimal_price_reduction

end optimal_price_reduction_l2583_258337


namespace quarter_circle_roll_path_length_l2583_258382

/-- The length of the path traveled by a point on a quarter-circle when rolled along a straight line -/
theorem quarter_circle_roll_path_length (r : ℝ) (h : r = 3 / Real.pi) :
  let path_length := 3 * (Real.pi * r / 2)
  path_length = 4.5 := by sorry

end quarter_circle_roll_path_length_l2583_258382


namespace exam_average_l2583_258313

theorem exam_average (successful_count unsuccessful_count : ℕ)
                     (successful_avg unsuccessful_avg : ℚ)
                     (h1 : successful_count = 20)
                     (h2 : unsuccessful_count = 20)
                     (h3 : successful_avg = 42)
                     (h4 : unsuccessful_avg = 38) :
  let total_count := successful_count + unsuccessful_count
  let total_points := successful_count * successful_avg + unsuccessful_count * unsuccessful_avg
  total_points / total_count = 40 := by
sorry

end exam_average_l2583_258313


namespace train_passing_time_l2583_258342

theorem train_passing_time (fast_train_length slow_train_length : ℝ)
  (fast_train_passing_time : ℝ) (h1 : fast_train_length = 315)
  (h2 : slow_train_length = 300) (h3 : fast_train_passing_time = 21) :
  slow_train_length / (fast_train_length / fast_train_passing_time) = 20 :=
by sorry

end train_passing_time_l2583_258342


namespace infinite_solutions_imply_d_value_l2583_258324

theorem infinite_solutions_imply_d_value (d : ℚ) :
  (∀ (x : ℚ), 3 * (5 + 2 * d * x) = 15 * x + 15) → d = 5 / 2 := by
  sorry

end infinite_solutions_imply_d_value_l2583_258324


namespace m_range_l2583_258372

/-- The proposition p: The solution set of the inequality |x|+|x-1| > m is R -/
def p (m : ℝ) : Prop :=
  ∀ x, |x| + |x - 1| > m

/-- The proposition q: f(x)=(5-2m)^x is an increasing function -/
def q (m : ℝ) : Prop :=
  ∀ x y, x < y → (5 - 2*m)^x < (5 - 2*m)^y

/-- The range of m given the conditions -/
theorem m_range :
  ∃ m, (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ 1 ≤ m ∧ m < 2 :=
sorry

end m_range_l2583_258372


namespace simplified_expression_terms_l2583_258304

def polynomial_terms (n : ℕ) : ℕ := Nat.choose (n + 4 - 1) (4 - 1)

theorem simplified_expression_terms :
  polynomial_terms 5 = 56 := by sorry

end simplified_expression_terms_l2583_258304


namespace opposite_of_sqrt_4_l2583_258355

theorem opposite_of_sqrt_4 : -(Real.sqrt 4) = -2 := by sorry

end opposite_of_sqrt_4_l2583_258355


namespace parabola_coefficients_l2583_258343

/-- A parabola with given properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ
  vertex_y : ℝ
  point_x : ℝ
  point_y : ℝ
  vertex_property : vertex_y = a * vertex_x^2 + b * vertex_x + c
  point_property : point_y = a * point_x^2 + b * point_x + c
  symmetry_property : b = -2 * a * vertex_x

/-- The theorem stating the values of a, b, and c for the given parabola -/
theorem parabola_coefficients (p : Parabola)
  (h_vertex : p.vertex_x = 2 ∧ p.vertex_y = 4)
  (h_point : p.point_x = 0 ∧ p.point_y = 5) :
  p.a = 1/4 ∧ p.b = -1 ∧ p.c = 5 := by
  sorry

end parabola_coefficients_l2583_258343


namespace ruler_measurement_l2583_258350

/-- Represents a ruler with marks at specific positions -/
structure Ruler :=
  (marks : List ℝ)

/-- Checks if a length can be measured using the given ruler -/
def can_measure (r : Ruler) (length : ℝ) : Prop :=
  ∃ (coeffs : List ℤ), length = (List.zip r.marks coeffs).foldl (λ acc (m, c) => acc + m * c) 0

theorem ruler_measurement (r : Ruler) (h : r.marks = [0, 7, 11]) :
  (can_measure r 8) ∧ (can_measure r 5) := by
  sorry

end ruler_measurement_l2583_258350


namespace percentage_less_than_l2583_258396

theorem percentage_less_than (p t j : ℝ) 
  (ht : t = p * (1 - 0.0625))
  (hj : j = t * (1 - 0.20)) : 
  j = p * (1 - 0.25) := by
sorry

end percentage_less_than_l2583_258396


namespace tan_two_alpha_l2583_258374

theorem tan_two_alpha (α : Real) 
  (h : (Real.sin (Real.pi - α) + Real.sin (Real.pi / 2 - α)) / (Real.sin α - Real.cos α) = 1 / 2) : 
  Real.tan (2 * α) = 3 / 4 := by
  sorry

end tan_two_alpha_l2583_258374


namespace min_value_trig_expression_equality_condition_l2583_258385

theorem min_value_trig_expression (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 20)^2 ≥ 236.137 := by
  sorry

theorem equality_condition (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 20)^2 = 236.137 ↔ 
  (Real.cos α = 10 / Real.sqrt 500 ∧ Real.sin α = 20 / Real.sqrt 500 ∧ β = Real.pi/2 - α) := by
  sorry

end min_value_trig_expression_equality_condition_l2583_258385


namespace boat_speed_in_still_water_l2583_258303

/-- The speed of a boat in still water, given downstream travel information and current speed. -/
theorem boat_speed_in_still_water
  (current_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : current_speed = 5)
  (h2 : downstream_distance = 5)
  (h3 : downstream_time = 1/5) :
  let downstream_speed := (boat_speed : ℝ) + current_speed
  downstream_distance = downstream_speed * downstream_time →
  boat_speed = 20 :=
by
  sorry

end boat_speed_in_still_water_l2583_258303


namespace irreducible_fraction_to_mersenne_form_l2583_258386

theorem irreducible_fraction_to_mersenne_form 
  (p q : ℕ+) 
  (h_q_odd : q.val % 2 = 1) : 
  ∃ (n k : ℕ+), (p : ℚ) / q = (n : ℚ) / (2^k.val - 1) :=
sorry

end irreducible_fraction_to_mersenne_form_l2583_258386


namespace log_50_between_consecutive_integers_l2583_258371

theorem log_50_between_consecutive_integers :
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < b ∧ a + b = 3 :=
by sorry

end log_50_between_consecutive_integers_l2583_258371


namespace problem_solution_l2583_258367

theorem problem_solution (a b : ℚ) 
  (eq1 : 8*a + 3*b = -1)
  (eq2 : a = b - 3) : 
  5*b = 115/11 := by
sorry

end problem_solution_l2583_258367


namespace warehouse_problem_l2583_258362

/-- Represents the time (in hours) it takes a team to move all goods in a warehouse -/
structure TeamSpeed :=
  (hours : ℝ)
  (positive : hours > 0)

/-- Represents the division of Team C's time between helping Team A and Team B -/
structure TeamCHelp :=
  (helpA : ℝ)
  (helpB : ℝ)
  (positive_helpA : helpA > 0)
  (positive_helpB : helpB > 0)

/-- The main theorem stating the solution to the warehouse problem -/
theorem warehouse_problem 
  (speedA : TeamSpeed) 
  (speedB : TeamSpeed) 
  (speedC : TeamSpeed)
  (h_speedA : speedA.hours = 6)
  (h_speedB : speedB.hours = 7)
  (h_speedC : speedC.hours = 14) :
  ∃ (help : TeamCHelp),
    help.helpA = 7/4 ∧ 
    help.helpB = 7/2 ∧
    help.helpA + help.helpB = speedA.hours * speedB.hours / (speedA.hours + speedB.hours) ∧
    1 / speedA.hours + 1 / speedC.hours * help.helpA = 1 ∧
    1 / speedB.hours + 1 / speedC.hours * help.helpB = 1 :=
sorry

end warehouse_problem_l2583_258362


namespace interest_calculation_l2583_258354

/-- Calculates the total interest earned from two investments -/
def total_interest (total_investment : ℚ) (rate1 rate2 : ℚ) (amount1 : ℚ) : ℚ :=
  let amount2 := total_investment - amount1
  amount1 * rate1 + amount2 * rate2

/-- Proves that the total interest earned is $490 given the specified conditions -/
theorem interest_calculation :
  let total_investment : ℚ := 8000
  let rate1 : ℚ := 8 / 100
  let rate2 : ℚ := 5 / 100
  let amount1 : ℚ := 3000
  total_interest total_investment rate1 rate2 amount1 = 490 := by
  sorry

end interest_calculation_l2583_258354


namespace restaurant_earnings_l2583_258378

theorem restaurant_earnings : 
  let meals_1 := 10
  let price_1 := 8
  let meals_2 := 5
  let price_2 := 10
  let meals_3 := 20
  let price_3 := 4
  meals_1 * price_1 + meals_2 * price_2 + meals_3 * price_3 = 210 :=
by sorry

end restaurant_earnings_l2583_258378


namespace y1_gt_y2_l2583_258395

/-- A quadratic function with a positive leading coefficient and symmetric axis at x = 1 -/
structure SymmetricQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  sym_axis : b = -2 * a

/-- The y-coordinate of the quadratic function at a given x -/
def y_coord (q : SymmetricQuadratic) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- Theorem stating that y₁ > y₂ for the given quadratic function -/
theorem y1_gt_y2 (q : SymmetricQuadratic) (y₁ y₂ : ℝ)
  (h1 : y_coord q (-1) = y₁)
  (h2 : y_coord q 2 = y₂) :
  y₁ > y₂ := by
  sorry

end y1_gt_y2_l2583_258395


namespace polynomial_remainder_l2583_258322

/-- The polynomial p(x) = x^3 - 4x^2 + 3x + 2 -/
def p (x : ℝ) : ℝ := x^3 - 4*x^2 + 3*x + 2

/-- The remainder when p(x) is divided by (x - 1) -/
def remainder : ℝ := p 1

theorem polynomial_remainder : remainder = 2 := by
  sorry

end polynomial_remainder_l2583_258322


namespace scientific_notation_million_l2583_258356

theorem scientific_notation_million (x : ℝ) (h : x = 1464.3) :
  x * (10 : ℝ)^6 = 1.4643 * (10 : ℝ)^7 := by
  sorry

end scientific_notation_million_l2583_258356


namespace solution_set_inequality_l2583_258353

theorem solution_set_inequality (x : ℝ) : 
  (2*x - 1) / x < 0 ↔ 0 < x ∧ x < 1/2 :=
by sorry

end solution_set_inequality_l2583_258353


namespace circumcenters_not_concyclic_l2583_258326

-- Define a point in 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define a function to check if a quadrilateral is convex
def isConvex (q : Quadrilateral) : Prop := sorry

-- Define a function to get the circumcenter of a triangle
def circumcenter (p1 p2 p3 : Point) : Point := sorry

-- Define a function to check if points are distinct
def areDistinct (p1 p2 p3 p4 : Point) : Prop := sorry

-- Define a function to check if points are concyclic
def areConcyclic (p1 p2 p3 p4 : Point) : Prop := sorry

-- Theorem statement
theorem circumcenters_not_concyclic (q : Quadrilateral) 
  (h_convex : isConvex q)
  (O_A : Point) (O_B : Point) (O_C : Point) (O_D : Point)
  (h_O_A : O_A = circumcenter q.B q.C q.D)
  (h_O_B : O_B = circumcenter q.C q.D q.A)
  (h_O_C : O_C = circumcenter q.D q.A q.B)
  (h_O_D : O_D = circumcenter q.A q.B q.C)
  (h_distinct : areDistinct O_A O_B O_C O_D) :
  ¬(areConcyclic O_A O_B O_C O_D) := by
  sorry

end circumcenters_not_concyclic_l2583_258326


namespace sides_divisible_by_three_l2583_258344

/-- A convex polygon divided into triangles by non-intersecting diagonals. -/
structure TriangulatedPolygon where
  /-- The number of sides of the polygon. -/
  sides : ℕ
  /-- The number of triangles in the triangulation. -/
  triangles : ℕ
  /-- The property that each vertex is a vertex of an odd number of triangles. -/
  odd_vertex_property : Bool

/-- 
Theorem: If a convex polygon is divided into triangles by non-intersecting diagonals,
and each vertex of the polygon is a vertex of an odd number of these triangles,
then the number of sides of the polygon is divisible by 3.
-/
theorem sides_divisible_by_three (p : TriangulatedPolygon) 
  (h : p.odd_vertex_property = true) : 
  ∃ k : ℕ, p.sides = 3 * k :=
sorry

end sides_divisible_by_three_l2583_258344


namespace simplify_expression_l2583_258328

theorem simplify_expression (a : ℝ) : (1 + a) * (1 - a) + a * (a - 2) = 1 - 2 * a := by
  sorry

end simplify_expression_l2583_258328


namespace stationery_box_sheets_l2583_258376

theorem stationery_box_sheets (S E : ℕ) : 
  S - (S / 3 + 50) = 50 →
  E = S / 3 + 50 →
  S = 150 := by
sorry

end stationery_box_sheets_l2583_258376


namespace occupancy_is_75_percent_l2583_258349

/-- Represents an apartment complex -/
structure ApartmentComplex where
  buildings : Nat
  studio_per_building : Nat
  two_person_per_building : Nat
  four_person_per_building : Nat
  current_occupancy : Nat

/-- Calculate the maximum occupancy of an apartment complex -/
def max_occupancy (complex : ApartmentComplex) : Nat :=
  complex.buildings * (complex.studio_per_building + 2 * complex.two_person_per_building + 4 * complex.four_person_per_building)

/-- Calculate the occupancy percentage of an apartment complex -/
def occupancy_percentage (complex : ApartmentComplex) : Rat :=
  (complex.current_occupancy : Rat) / (max_occupancy complex)

/-- The main theorem stating that the occupancy percentage is 75% -/
theorem occupancy_is_75_percent (complex : ApartmentComplex) 
  (h1 : complex.buildings = 4)
  (h2 : complex.studio_per_building = 10)
  (h3 : complex.two_person_per_building = 20)
  (h4 : complex.four_person_per_building = 5)
  (h5 : complex.current_occupancy = 210) :
  occupancy_percentage complex = 3/4 := by
  sorry

end occupancy_is_75_percent_l2583_258349


namespace superhero_speed_in_miles_per_hour_l2583_258315

-- Define the superhero's speed in kilometers per minute
def superhero_speed_km_per_min : ℝ := 1000

-- Define the conversion factor from kilometers to miles
def km_to_miles : ℝ := 0.6

-- Define the number of minutes in an hour
def minutes_per_hour : ℝ := 60

-- Theorem statement
theorem superhero_speed_in_miles_per_hour :
  superhero_speed_km_per_min * minutes_per_hour * km_to_miles = 36000 := by
  sorry

end superhero_speed_in_miles_per_hour_l2583_258315


namespace smallest_x_value_l2583_258380

theorem smallest_x_value (x y : ℕ+) (h : (4 : ℚ) / 5 = y / (200 + x)) : 
  ∀ z : ℕ+, (4 : ℚ) / 5 = (y : ℚ) / (200 + z) → x ≤ z :=
by sorry

#check smallest_x_value

end smallest_x_value_l2583_258380


namespace patio_layout_change_l2583_258348

/-- Represents a rectangular patio layout --/
structure PatioLayout where
  rows : ℕ
  columns : ℕ
  total_tiles : ℕ
  is_rectangular : rows * columns = total_tiles

/-- The change in patio layout --/
def change_layout (initial : PatioLayout) (row_increase : ℕ) : PatioLayout :=
  { rows := initial.rows + row_increase,
    columns := initial.total_tiles / (initial.rows + row_increase),
    total_tiles := initial.total_tiles,
    is_rectangular := sorry }

theorem patio_layout_change (initial : PatioLayout) 
  (h1 : initial.total_tiles = 30)
  (h2 : initial.rows = 5) :
  let final := change_layout initial 4
  initial.columns - final.columns = 3 := by sorry

end patio_layout_change_l2583_258348


namespace shyne_garden_theorem_l2583_258307

/-- Represents the number of plants that can be grown from one packet of seeds for each type of plant. -/
structure PlantsPerPacket where
  eggplants : ℕ
  sunflowers : ℕ
  tomatoes : ℕ
  peas : ℕ
  cucumbers : ℕ

/-- Represents the number of seed packets bought for each type of plant. -/
structure PacketsBought where
  eggplants : ℕ
  sunflowers : ℕ
  tomatoes : ℕ
  peas : ℕ
  cucumbers : ℕ

/-- Represents the percentage of plants that can be grown in each season. -/
structure PlantingPercentages where
  spring_eggplants_peas : ℚ
  summer_sunflowers_cucumbers : ℚ
  both_seasons_tomatoes : ℚ

/-- Calculates the total number of plants Shyne can potentially grow across spring and summer. -/
def totalPlants (plantsPerPacket : PlantsPerPacket) (packetsBought : PacketsBought) (percentages : PlantingPercentages) : ℕ :=
  sorry

/-- Theorem stating that Shyne can potentially grow 366 plants across spring and summer. -/
theorem shyne_garden_theorem (plantsPerPacket : PlantsPerPacket) (packetsBought : PacketsBought) (percentages : PlantingPercentages) :
  plantsPerPacket.eggplants = 14 ∧
  plantsPerPacket.sunflowers = 10 ∧
  plantsPerPacket.tomatoes = 16 ∧
  plantsPerPacket.peas = 20 ∧
  plantsPerPacket.cucumbers = 18 ∧
  packetsBought.eggplants = 6 ∧
  packetsBought.sunflowers = 8 ∧
  packetsBought.tomatoes = 7 ∧
  packetsBought.peas = 9 ∧
  packetsBought.cucumbers = 5 ∧
  percentages.spring_eggplants_peas = 3/5 ∧
  percentages.summer_sunflowers_cucumbers = 7/10 ∧
  percentages.both_seasons_tomatoes = 4/5 →
  totalPlants plantsPerPacket packetsBought percentages = 366 :=
by
  sorry

end shyne_garden_theorem_l2583_258307


namespace unique_function_satisfying_conditions_l2583_258366

theorem unique_function_satisfying_conditions :
  ∃! f : ℝ → ℝ, 
    (∀ x : ℝ, f (x + 1) ≥ f x + 1) ∧ 
    (∀ x y : ℝ, f (x * y) ≥ f x * f y) ∧
    (∀ x : ℝ, f x = x) := by
  sorry

end unique_function_satisfying_conditions_l2583_258366


namespace quadratic_no_real_roots_l2583_258340

theorem quadratic_no_real_roots : 
  ∀ x : ℝ, 2 * x^2 - 3 * x + (3/2) ≠ 0 := by
sorry

end quadratic_no_real_roots_l2583_258340


namespace range_of_valid_m_l2583_258357

/-- The set A as defined in the problem -/
def A : Set ℝ := {y | ∃ x ∈ Set.Icc (-1/2) 2, y = x^2 - (3/2)*x + 1}

/-- The set B as defined in the problem -/
def B (m : ℝ) : Set ℝ := {x | |x - m| ≥ 1}

/-- The range of values for m that satisfies the condition A ⊆ B -/
def valid_m : Set ℝ := {m | A ⊆ B m}

/-- Theorem stating that the range of valid m is (-∞, -9/16] ∪ [3, +∞) -/
theorem range_of_valid_m : valid_m = Set.Iic (-9/16) ∪ Set.Ici 3 := by sorry

end range_of_valid_m_l2583_258357


namespace complex_z_value_l2583_258359

-- Define the operation for 2x2 matrices
def matrixOp (a b c d : ℂ) : ℂ := a * d - b * c

-- Theorem statement
theorem complex_z_value (z : ℂ) :
  matrixOp z (1 - Complex.I) (1 + Complex.I) 1 = Complex.I →
  z = 2 + Complex.I :=
by sorry

end complex_z_value_l2583_258359


namespace sticker_collection_value_l2583_258391

theorem sticker_collection_value (total_stickers : ℕ) (sample_size : ℕ) (sample_value : ℕ) 
  (h1 : total_stickers = 18)
  (h2 : sample_size = 6)
  (h3 : sample_value = 24) :
  (total_stickers : ℚ) * (sample_value : ℚ) / (sample_size : ℚ) = 72 := by
  sorry

end sticker_collection_value_l2583_258391


namespace tunnel_length_proof_l2583_258398

/-- Represents the scale of a map -/
structure MapScale where
  ratio : ℚ

/-- Represents a length on a map -/
structure MapLength where
  length : ℚ
  unit : String

/-- Represents an actual length in reality -/
structure ActualLength where
  length : ℚ
  unit : String

/-- Converts a MapLength to an ActualLength based on a given MapScale -/
def convertMapLengthToActual (scale : MapScale) (mapLength : MapLength) : ActualLength :=
  { length := mapLength.length * scale.ratio
    unit := "cm" }

/-- Converts centimeters to kilometers -/
def cmToKm (cm : ℚ) : ℚ :=
  cm / 100000

theorem tunnel_length_proof (scale : MapScale) (mapLength : MapLength) :
  scale.ratio = 38000 →
  mapLength.length = 7 →
  mapLength.unit = "cm" →
  let actualLength := convertMapLengthToActual scale mapLength
  cmToKm actualLength.length = 2.66 := by
    sorry

end tunnel_length_proof_l2583_258398


namespace log_inequality_implies_order_l2583_258301

theorem log_inequality_implies_order (x y : ℝ) :
  (Real.log x / Real.log (1/2)) < (Real.log y / Real.log (1/2)) ∧
  (Real.log y / Real.log (1/2)) < 0 →
  1 < y ∧ y < x :=
by sorry

end log_inequality_implies_order_l2583_258301


namespace complex_power_sum_l2583_258311

theorem complex_power_sum (z : ℂ) (h : z + (1 / z) = 2 * Real.cos (5 * π / 180)) :
  z^1000 + (1 / z^1000) = 2 * Real.cos (20 * π / 180) :=
by sorry

end complex_power_sum_l2583_258311


namespace quadratic_roots_property_l2583_258341

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ - 3 = 0 ∧ x₂^2 + x₂ - 3 = 0 → x₁^3 - 4*x₂^2 + 19 = 0 := by
  sorry

end quadratic_roots_property_l2583_258341


namespace omega_set_classification_l2583_258389

-- Define the concept of an Ω set
def is_omega_set (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (p₁ : ℝ × ℝ), p₁ ∈ M → ∃ (p₂ : ℝ × ℝ), p₂ ∈ M ∧ p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

-- Define the sets
def set1 : Set (ℝ × ℝ) := {p | p.2 = 1 / p.1}
def set2 : Set (ℝ × ℝ) := {p | p.2 = (p.1 - 1) / Real.exp p.1}
def set3 : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt (1 - p.1^2)}
def set4 : Set (ℝ × ℝ) := {p | p.2 = p.1^2 - 2*p.1 + 2}
def set5 : Set (ℝ × ℝ) := {p | p.2 = Real.cos p.1 + Real.sin p.1}

-- State the theorem
theorem omega_set_classification :
  (¬ is_omega_set set1) ∧
  (is_omega_set set2) ∧
  (is_omega_set set3) ∧
  (¬ is_omega_set set4) ∧
  (is_omega_set set5) := by
  sorry

end omega_set_classification_l2583_258389


namespace total_frog_eyes_l2583_258370

/-- The number of frogs in the pond -/
def total_frogs : ℕ := 6

/-- The number of eyes for Species A frogs -/
def eyes_species_a : ℕ := 2

/-- The number of eyes for Species B frogs -/
def eyes_species_b : ℕ := 3

/-- The number of eyes for Species C frogs -/
def eyes_species_c : ℕ := 4

/-- The number of Species A frogs -/
def frogs_species_a : ℕ := 2

/-- The number of Species B frogs -/
def frogs_species_b : ℕ := 1

/-- The number of Species C frogs -/
def frogs_species_c : ℕ := 3

theorem total_frog_eyes : 
  frogs_species_a * eyes_species_a + 
  frogs_species_b * eyes_species_b + 
  frogs_species_c * eyes_species_c = 19 := by
  sorry

end total_frog_eyes_l2583_258370


namespace ace_king_probability_l2583_258381

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Aces in a standard deck -/
def num_aces : ℕ := 4

/-- The number of Kings in a standard deck -/
def num_kings : ℕ := 4

/-- The probability of drawing an Ace followed by a King from a standard deck -/
theorem ace_king_probability : 
  (num_aces : ℚ) / deck_size * num_kings / (deck_size - 1) = 4 / 663 := by
sorry

end ace_king_probability_l2583_258381


namespace tangent_lines_problem_l2583_258369

theorem tangent_lines_problem (num_not_enclosed : ℕ) (lines_less_than_30 : ℕ) :
  num_not_enclosed = 68 →
  lines_less_than_30 = 4 →
  ∃ (num_tangent_lines : ℕ),
    num_tangent_lines = 30 - lines_less_than_30 ∧
    num_tangent_lines * 2 = num_not_enclosed :=
by sorry

end tangent_lines_problem_l2583_258369


namespace tan_five_pi_four_l2583_258392

theorem tan_five_pi_four : Real.tan (5 * π / 4) = 1 := by
  sorry

end tan_five_pi_four_l2583_258392


namespace inequality_and_equality_condition_l2583_258314

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (a + 2) * (b + 2) ≥ c * d ∧ 
  (∃ (a₀ b₀ c₀ d₀ : ℝ), a₀^2 + b₀^2 + c₀^2 + d₀^2 = 4 ∧ 
    (a₀ + 2) * (b₀ + 2) = c₀ * d₀ ∧ 
    a₀ = -2 ∧ b₀ = -2 ∧ c₀ = 1 ∧ d₀ = 1) :=
by sorry

end inequality_and_equality_condition_l2583_258314


namespace contrapositive_true_l2583_258312

theorem contrapositive_true : 
  (∀ x : ℝ, (x^2 ≤ 0 → x ≥ 0)) := by sorry

end contrapositive_true_l2583_258312


namespace solutions_of_f_eq_quarter_solution_set_of_f_leq_two_l2583_258331

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else Real.log x / Real.log 4

-- Theorem for the solutions of f(x) = 1/4
theorem solutions_of_f_eq_quarter :
  {x : ℝ | f x = 1/4} = {2, Real.sqrt 2} :=
sorry

-- Theorem for the solution set of f(x) ≤ 2
theorem solution_set_of_f_leq_two :
  {x : ℝ | f x ≤ 2} = Set.Icc (-1) 16 :=
sorry

end solutions_of_f_eq_quarter_solution_set_of_f_leq_two_l2583_258331


namespace dot_only_count_l2583_258383

/-- Represents the number of letters in an alphabet with specific characteristics. -/
structure Alphabet where
  total : ℕ
  dot_and_line : ℕ
  line_only : ℕ
  dot_only : ℕ

/-- Theorem stating that in an alphabet with given properties, 
    the number of letters containing only a dot is 3. -/
theorem dot_only_count (α : Alphabet) 
  (h_total : α.total = 40)
  (h_dot_and_line : α.dot_and_line = 13)
  (h_line_only : α.line_only = 24)
  (h_all_covered : α.total = α.dot_and_line + α.line_only + α.dot_only) :
  α.dot_only = 3 := by
  sorry

end dot_only_count_l2583_258383


namespace range_of_a_l2583_258368

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) → (0 < a ∧ a < 1) :=
by sorry

end range_of_a_l2583_258368


namespace saturday_exclamation_l2583_258334

/-- Represents the alien's exclamation as a string of 'A's and 'U's -/
def Exclamation := String

/-- Transforms a single character in the exclamation -/
def transformChar (c : Char) : Char :=
  match c with
  | 'A' => 'U'
  | 'U' => 'A'
  | _ => c

/-- Transforms the second half of the exclamation -/
def transformSecondHalf (s : String) : String :=
  s.map transformChar

/-- Generates the next day's exclamation based on the current day -/
def nextDayExclamation (current : Exclamation) : Exclamation :=
  let n := current.length
  let firstHalf := current.take (n / 2)
  let secondHalf := current.drop (n / 2)
  firstHalf ++ transformSecondHalf secondHalf

/-- Generates the nth day's exclamation -/
def nthDayExclamation (n : Nat) : Exclamation :=
  match n with
  | 0 => "A"
  | n + 1 => nextDayExclamation (nthDayExclamation n)

theorem saturday_exclamation :
  nthDayExclamation 5 = "АУУАУААУУААУАУААУУААУААААУУААУАА" :=
by sorry

end saturday_exclamation_l2583_258334


namespace max_points_is_36_l2583_258335

/-- Represents a tournament with 8 teams where each team plays every other team twice -/
structure Tournament where
  num_teams : Nat
  games_per_pair : Nat
  win_points : Nat
  draw_points : Nat
  loss_points : Nat

/-- Calculate the total number of games in the tournament -/
def total_games (t : Tournament) : Nat :=
  (t.num_teams * (t.num_teams - 1) / 2) * t.games_per_pair

/-- Calculate the maximum possible points for each of the top three teams -/
def max_points_top_three (t : Tournament) : Nat :=
  let games_against_others := (t.num_teams - 3) * t.games_per_pair
  let points_against_others := games_against_others * t.win_points
  let games_among_top_three := 2 * t.games_per_pair
  let points_among_top_three := games_among_top_three * t.draw_points
  points_against_others + points_among_top_three

/-- The theorem to be proved -/
theorem max_points_is_36 (t : Tournament) 
  (h1 : t.num_teams = 8)
  (h2 : t.games_per_pair = 2)
  (h3 : t.win_points = 3)
  (h4 : t.draw_points = 1)
  (h5 : t.loss_points = 0) :
  max_points_top_three t = 36 := by
  sorry

end max_points_is_36_l2583_258335


namespace nineteen_power_nineteen_not_sum_of_cube_and_fourth_power_l2583_258302

theorem nineteen_power_nineteen_not_sum_of_cube_and_fourth_power :
  ¬ ∃ (x y : ℤ), 19^19 = x^3 + y^4 := by
  sorry

end nineteen_power_nineteen_not_sum_of_cube_and_fourth_power_l2583_258302


namespace not_circle_iff_a_eq_zero_l2583_258316

/-- The equation of a potential circle -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - a*x + 2*y + 1 = 0

/-- The condition for the equation to represent a circle -/
def is_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that the equation does not represent a circle iff a = 0 -/
theorem not_circle_iff_a_eq_zero (a : ℝ) :
  ¬(is_circle a) ↔ a = 0 :=
sorry

end not_circle_iff_a_eq_zero_l2583_258316


namespace marble_weight_calculation_l2583_258384

/-- Given two pieces of marble of equal weight and a third piece,
    if the total weight is 0.75 tons and the third piece weighs 0.08333333333333333 ton,
    then the weight of each of the first two pieces is 0.33333333333333335 ton. -/
theorem marble_weight_calculation (w : ℝ) : 
  2 * w + 0.08333333333333333 = 0.75 → w = 0.33333333333333335 := by
  sorry

end marble_weight_calculation_l2583_258384


namespace spencer_total_distance_l2583_258305

/-- The total distance Spencer walked throughout the day -/
def total_distance (d1 d2 d3 d4 d5 d6 d7 : ℝ) : ℝ :=
  d1 + d2 + d3 + d4 + d5 + d6 + d7

/-- Theorem: Given Spencer's walking distances, the total distance is 8.6 miles -/
theorem spencer_total_distance :
  total_distance 1.2 0.6 0.9 1.7 2.1 1.3 0.8 = 8.6 := by
  sorry

end spencer_total_distance_l2583_258305


namespace harry_weekly_earnings_l2583_258379

/-- Represents Harry's dog-walking schedule and earnings --/
structure DogWalker where
  mon_wed_fri_dogs : ℕ
  tuesday_dogs : ℕ
  thursday_dogs : ℕ
  pay_per_dog : ℕ

/-- Calculates the weekly earnings of a dog walker --/
def weekly_earnings (dw : DogWalker) : ℕ :=
  (3 * dw.mon_wed_fri_dogs + dw.tuesday_dogs + dw.thursday_dogs) * dw.pay_per_dog

/-- Harry's specific dog-walking schedule --/
def harry : DogWalker :=
  { mon_wed_fri_dogs := 7
    tuesday_dogs := 12
    thursday_dogs := 9
    pay_per_dog := 5 }

/-- Theorem stating Harry's weekly earnings --/
theorem harry_weekly_earnings :
  weekly_earnings harry = 210 := by
  sorry

end harry_weekly_earnings_l2583_258379


namespace no_real_roots_condition_implies_inequality_g_no_intersect_l2583_258394

/-- A quadratic function that doesn't intersect with y = x -/
structure NoIntersectQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  no_intersect : ∀ x : ℝ, a * x^2 + b * x + c ≠ x

def f (q : NoIntersectQuadratic) (x : ℝ) : ℝ := q.a * x^2 + q.b * x + q.c

theorem no_real_roots (q : NoIntersectQuadratic) : ∀ x : ℝ, f q (f q x) ≠ x := by sorry

theorem condition_implies_inequality (q : NoIntersectQuadratic) (h : q.a + q.b + q.c = 0) :
  ∀ x : ℝ, f q (f q x) < x := by sorry

def g (q : NoIntersectQuadratic) (x : ℝ) : ℝ := q.a * x^2 - q.b * x + q.c

theorem g_no_intersect (q : NoIntersectQuadratic) : ∀ x : ℝ, g q x ≠ -x := by sorry

end no_real_roots_condition_implies_inequality_g_no_intersect_l2583_258394


namespace perpendicular_parallel_implies_perpendicular_l2583_258373

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations for perpendicular and parallel
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_implies_perpendicular
  (α β : Plane) (l : Line)
  (h1 : α ≠ β)
  (h2 : perpendicular l α)
  (h3 : parallel α β) :
  perpendicular l β :=
sorry

end perpendicular_parallel_implies_perpendicular_l2583_258373


namespace multiplication_commutative_l2583_258320

theorem multiplication_commutative (a b : ℝ) : a * b = b * a := by
  sorry

end multiplication_commutative_l2583_258320


namespace martha_lasagna_cost_l2583_258323

/-- The cost of ingredients for Martha's lasagna --/
theorem martha_lasagna_cost : 
  let cheese_weight : ℝ := 1.5
  let meat_weight : ℝ := 0.5
  let cheese_price : ℝ := 6
  let meat_price : ℝ := 8
  cheese_weight * cheese_price + meat_weight * meat_price = 13 := by
  sorry

end martha_lasagna_cost_l2583_258323


namespace intersection_of_A_and_B_l2583_258329

def A : Set Int := {-1, 0, 1, 2, 3}
def B : Set Int := {-3, -1, 1, 3, 5}

theorem intersection_of_A_and_B : A ∩ B = {-1, 1, 3} := by
  sorry

end intersection_of_A_and_B_l2583_258329


namespace a_range_theorem_l2583_258306

theorem a_range_theorem (a : ℝ) : 
  (∀ x : ℝ, a^2 * x - 2*(a - x - 4) < 0) ↔ -2 < a ∧ a ≤ 2 :=
by sorry

end a_range_theorem_l2583_258306


namespace shekars_weighted_average_sum_of_weightages_is_one_l2583_258333

/-- Represents the subjects and their corresponding scores and weightages -/
structure Subject where
  name : String
  score : ℝ
  weightage : ℝ

/-- Calculates the weighted average of a list of subjects -/
def weightedAverage (subjects : List Subject) : ℝ :=
  (subjects.map (fun s => s.score * s.weightage)).sum

/-- Shekar's subjects with their scores and weightages -/
def shekarsSubjects : List Subject := [
  ⟨"Mathematics", 76, 0.15⟩,
  ⟨"Science", 65, 0.15⟩,
  ⟨"Social Studies", 82, 0.20⟩,
  ⟨"English", 67, 0.20⟩,
  ⟨"Biology", 75, 0.10⟩,
  ⟨"Computer Science", 89, 0.10⟩,
  ⟨"History", 71, 0.10⟩
]

/-- Theorem stating that Shekar's weighted average marks is 74.45 -/
theorem shekars_weighted_average :
  weightedAverage shekarsSubjects = 74.45 := by
  sorry

/-- Proof that the sum of weightages is 1 -/
theorem sum_of_weightages_is_one :
  (shekarsSubjects.map (fun s => s.weightage)).sum = 1 := by
  sorry

end shekars_weighted_average_sum_of_weightages_is_one_l2583_258333


namespace trajectory_is_parabola_l2583_258365

/-- The trajectory of point M(x,y) satisfying the distance condition -/
def trajectory_equation (x y : ℝ) : Prop :=
  ((x - 4)^2 + y^2)^(1/2) = |x + 3| + 1

/-- The theorem stating the equation of the trajectory -/
theorem trajectory_is_parabola (x y : ℝ) :
  trajectory_equation x y → y^2 = 16 * x := by
  sorry

end trajectory_is_parabola_l2583_258365


namespace cone_surface_area_l2583_258310

/-- The surface area of a cone, given its lateral surface properties -/
theorem cone_surface_area (r : ℝ) (h : ℝ) : 
  (r * r * π + r * h * π = 16 * π / 9) →
  (h * h + r * r = 2 * 2) →
  (2 * π * r = 4 * π / 3) →
  (r * h * π = 4 * π / 3) →
  (r * r * π + r * h * π = 16 * π / 9) :=
by sorry

end cone_surface_area_l2583_258310


namespace pencils_bought_l2583_258321

-- Define the cost of a single pencil and notebook
variable (P N : ℝ)

-- Define the number of pencils in the second case
variable (X : ℝ)

-- Conditions from the problem
axiom cost_condition1 : 96 * P + 24 * N = 520
axiom cost_condition2 : X * P + 4 * N = 60
axiom sum_condition : P + N = 15.512820512820513

-- Theorem to prove
theorem pencils_bought : X = 3 := by
  sorry

end pencils_bought_l2583_258321


namespace chess_and_go_problem_l2583_258363

theorem chess_and_go_problem (chess_price go_price : ℝ) 
  (h1 : 6 * chess_price + 5 * go_price = 190)
  (h2 : 8 * chess_price + 10 * go_price = 320)
  (budget : ℝ) (total_sets : ℕ)
  (h3 : budget ≤ 1800)
  (h4 : total_sets = 100) :
  chess_price = 15 ∧ 
  go_price = 20 ∧ 
  ∃ (min_chess : ℕ), min_chess ≥ 40 ∧ 
    chess_price * min_chess + go_price * (total_sets - min_chess) ≤ budget :=
by sorry

end chess_and_go_problem_l2583_258363


namespace f_properties_l2583_258309

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 2*x

-- Theorem statement
theorem f_properties :
  (∀ x₁ x₂ : ℝ, x₁ ≥ 1 ∧ x₂ ≥ 1 ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x : ℝ, x ≥ 2 ∧ x ≤ 5 → f x ≤ 0) ∧
  (∀ x : ℝ, x ≥ 2 ∧ x ≤ 5 → f x ≥ -15) ∧
  (∃ x : ℝ, x ≥ 2 ∧ x ≤ 5 ∧ f x = 0) ∧
  (∃ x : ℝ, x ≥ 2 ∧ x ≤ 5 ∧ f x = -15) :=
by
  sorry

end f_properties_l2583_258309


namespace sphere_in_cube_surface_area_l2583_258327

theorem sphere_in_cube_surface_area (cube_edge : ℝ) (h : cube_edge = 2) :
  let sphere_radius := cube_edge / 2
  let sphere_surface_area := 4 * Real.pi * sphere_radius ^ 2
  sphere_surface_area = 4 * Real.pi :=
by sorry

end sphere_in_cube_surface_area_l2583_258327
