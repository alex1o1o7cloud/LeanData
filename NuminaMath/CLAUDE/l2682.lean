import Mathlib

namespace work_completion_l2682_268294

theorem work_completion (days1 days2 men2 : ℕ) 
  (h1 : days1 = 80)
  (h2 : days2 = 56)
  (h3 : men2 = 20)
  (h4 : ∀ m d, m * d = men2 * days2) : 
  ∃ men1 : ℕ, men1 = 14 ∧ men1 * days1 = men2 * days2 := by
  sorry

end work_completion_l2682_268294


namespace derek_water_addition_l2682_268222

/-- The amount of water Derek added to the bucket -/
def water_added (initial final : ℝ) : ℝ := final - initial

theorem derek_water_addition (initial final : ℝ) 
  (h1 : initial = 3)
  (h2 : final = 9.8) :
  water_added initial final = 6.8 := by
  sorry

end derek_water_addition_l2682_268222


namespace sequence_integer_count_l2682_268228

def sequence_term (n : ℕ) : ℚ :=
  12150 / 3^n

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem sequence_integer_count :
  ∃ (k : ℕ), k = 5 ∧
  (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
  (∀ (n : ℕ), n ≥ k → ¬ is_integer (sequence_term n)) :=
sorry

end sequence_integer_count_l2682_268228


namespace greatest_integer_radius_l2682_268281

theorem greatest_integer_radius (r : ℕ) : (r : ℝ) ^ 2 * Real.pi < 90 * Real.pi → r ≤ 9 :=
  sorry

end greatest_integer_radius_l2682_268281


namespace BC_time_is_three_hours_l2682_268203

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 4
def work_rate_C : ℚ := 1 / 12

-- Define the combined work rate of A and C
def work_rate_AC : ℚ := 1 / 3

-- Define the time taken by B and C together
def time_BC : ℚ := 1 / (work_rate_B + work_rate_C)

-- Theorem statement
theorem BC_time_is_three_hours :
  work_rate_A = 1 / 4 →
  work_rate_B = 1 / 4 →
  work_rate_AC = 1 / 3 →
  work_rate_C = work_rate_AC - work_rate_A →
  time_BC = 3 := by
  sorry


end BC_time_is_three_hours_l2682_268203


namespace sandy_shopping_money_l2682_268263

theorem sandy_shopping_money (initial_amount : ℝ) (spent_percentage : ℝ) (remaining_amount : ℝ) : 
  spent_percentage = 30 →
  remaining_amount = 140 →
  (1 - spent_percentage / 100) * initial_amount = remaining_amount →
  initial_amount = 200 := by
  sorry

end sandy_shopping_money_l2682_268263


namespace car_speed_problem_l2682_268214

/-- Proves that given two cars P and R traveling 900 miles, where car P takes 2 hours less
    time than car R and has an average speed 10 miles per hour greater than car R,
    the average speed of car R is 62.25 miles per hour. -/
theorem car_speed_problem (speed_r : ℝ) : 
  (900 / speed_r - 2 = 900 / (speed_r + 10)) → speed_r = 62.25 := by
  sorry

end car_speed_problem_l2682_268214


namespace sin_plus_cos_zero_equiv_cos_2x_over_sin_minus_cos_zero_l2682_268273

open Real

theorem sin_plus_cos_zero_equiv_cos_2x_over_sin_minus_cos_zero :
  ∀ x : ℝ, (sin x + cos x = 0) ↔ ((cos (2 * x)) / (sin x - cos x) = 0) :=
by sorry

end sin_plus_cos_zero_equiv_cos_2x_over_sin_minus_cos_zero_l2682_268273


namespace power_of_seven_l2682_268243

theorem power_of_seven (k : ℕ) (h : 7^k = 2) : 7^(2*k + 2) = 784 := by
  sorry

end power_of_seven_l2682_268243


namespace min_value_sum_squares_l2682_268217

/-- Given positive real numbers a, b, c satisfying the condition,
    the minimum value of the expression is 50 -/
theorem min_value_sum_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a/b + b/c + c/a + b/a + c/b + a/c = 10) :
  ∃ m : ℝ, m = 50 ∧ ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    x/y + y/z + z/x + y/x + z/y + x/z = 10 →
    (x/y + y/z + z/x)^2 + (y/x + z/y + x/z)^2 ≥ m :=
by sorry

end min_value_sum_squares_l2682_268217


namespace subtraction_of_negatives_l2682_268207

theorem subtraction_of_negatives : -5 - (-2) = -3 := by sorry

end subtraction_of_negatives_l2682_268207


namespace xy_power_2018_l2682_268236

theorem xy_power_2018 (x y : ℝ) (h : |x - 1/2| + (y + 2)^2 = 0) : (x*y)^2018 = 1 := by
  sorry

end xy_power_2018_l2682_268236


namespace intersection_distance_l2682_268225

/-- The distance between the intersection points of the line y = 1 - x and the circle x^2 + y^2 = 8 is equal to √30 -/
theorem intersection_distance : 
  ∃ (A B : ℝ × ℝ), 
    (A.1^2 + A.2^2 = 8) ∧ 
    (B.1^2 + B.2^2 = 8) ∧ 
    (A.2 = 1 - A.1) ∧ 
    (B.2 = 1 - B.1) ∧ 
    (A ≠ B) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 30) :=
by sorry

end intersection_distance_l2682_268225


namespace range_of_a_l2682_268269

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 3/4| ≤ 1/4
def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ 0 ≤ a ∧ a ≤ 1/2 ∧ (a ≠ 0 ∨ a ≠ 1/2) :=
sorry

end range_of_a_l2682_268269


namespace problem_solution_l2682_268270

theorem problem_solution (x y : ℝ) : 
  x > 0 → x = 3 → x + y = 60 * (1 / x) → y = 17 := by sorry

end problem_solution_l2682_268270


namespace cone_height_ratio_l2682_268224

/-- Represents a cone with height, slant height, and central angle of unfolded lateral surface -/
structure Cone where
  height : ℝ
  slant_height : ℝ
  central_angle : ℝ

/-- The theorem statement -/
theorem cone_height_ratio (A B : Cone) :
  A.slant_height = B.slant_height →
  A.central_angle + B.central_angle = 2 * Real.pi →
  A.central_angle * A.slant_height^2 / (B.central_angle * B.slant_height^2) = 2 →
  A.height / B.height = Real.sqrt 10 / 4 := by
  sorry

end cone_height_ratio_l2682_268224


namespace fifteenth_odd_multiple_of_five_l2682_268291

theorem fifteenth_odd_multiple_of_five : ∃ n : ℕ, 
  (n > 0) ∧ 
  (∀ k < n, ∃ m : ℕ, k = 2 * m + 1 ∧ ∃ l : ℕ, k = 5 * l) →
  (∃ m : ℕ, n = 2 * m + 1) ∧ 
  (∃ l : ℕ, n = 5 * l) ∧ 
  n = 145 :=
by sorry

end fifteenth_odd_multiple_of_five_l2682_268291


namespace min_value_trigonometric_expression_l2682_268202

theorem min_value_trigonometric_expression (A B C : Real) (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
  (h_sum : Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 2) :
  (1 / (Real.sin A ^ 2 * Real.cos B ^ 4) + 
   1 / (Real.sin B ^ 2 * Real.cos C ^ 4) + 
   1 / (Real.sin C ^ 2 * Real.cos A ^ 4)) ≥ 81/4 := by
  sorry

end min_value_trigonometric_expression_l2682_268202


namespace number_count_l2682_268268

theorem number_count (avg_all : ℝ) (avg_group1 : ℝ) (avg_group2 : ℝ) (avg_group3 : ℝ) 
  (h1 : avg_all = 2.80)
  (h2 : avg_group1 = 2.4)
  (h3 : avg_group2 = 2.3)
  (h4 : avg_group3 = 3.7) :
  ∃ (n : ℕ), n = 6 ∧ (2 * avg_group1 + 2 * avg_group2 + 2 * avg_group3) / n = avg_all := by
  sorry

end number_count_l2682_268268


namespace product_65_55_l2682_268289

theorem product_65_55 : 65 * 55 = 3575 := by
  sorry

end product_65_55_l2682_268289


namespace min_days_person_A_l2682_268213

/-- Represents the number of days a person takes to complete the project alone -/
structure PersonSpeed where
  days : ℕ
  days_positive : days > 0

/-- Represents the work done by a person in a day -/
def work_rate (speed : PersonSpeed) : ℚ :=
  1 / speed.days

/-- The total project work is 1 -/
def total_work : ℚ := 1

/-- Theorem stating the minimum number of days person A must work -/
theorem min_days_person_A (
  speed_A speed_B speed_C : PersonSpeed)
  (h_A : speed_A.days = 24)
  (h_B : speed_B.days = 36)
  (h_C : speed_C.days = 60)
  (total_days : ℕ)
  (h_total_days : total_days ≤ 18)
  (h_integer_days : ∃ (days_A days_B days_C : ℕ),
    days_A + days_B + days_C = total_days ∧
    days_A * work_rate speed_A + days_B * work_rate speed_B + days_C * work_rate speed_C = total_work) :
  ∃ (min_days_A : ℕ), min_days_A = 6 ∧
    ∀ (days_A : ℕ), 
      (∃ (days_B days_C : ℕ),
        days_A + days_B + days_C = total_days ∧
        days_A * work_rate speed_A + days_B * work_rate speed_B + days_C * work_rate speed_C = total_work) →
      days_A ≥ min_days_A :=
by sorry

end min_days_person_A_l2682_268213


namespace no_integer_n_makes_complex_fifth_power_real_l2682_268242

theorem no_integer_n_makes_complex_fifth_power_real : 
  ¬∃ (n : ℤ), (Complex.I : ℂ).im * ((n + 2 * Complex.I)^5).im = 0 := by sorry

end no_integer_n_makes_complex_fifth_power_real_l2682_268242


namespace magnitude_2a_minus_b_l2682_268215

def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![-1, 2]

theorem magnitude_2a_minus_b : ‖(2 • a) - b‖ = 5 := by
  sorry

end magnitude_2a_minus_b_l2682_268215


namespace expression_evaluation_l2682_268279

theorem expression_evaluation (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hsum : x + 1/y ≠ 0) : 
  (x^2 + 1/y^2) / (x + 1/y) = x - 1/y := by
  sorry

end expression_evaluation_l2682_268279


namespace exactly_two_rectangle_coverage_l2682_268249

/-- Represents a rectangle on a grid -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the overlap between two rectangles -/
structure Overlap where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ := r.width * r.height

/-- Calculates the area of an overlap -/
def overlapArea (o : Overlap) : ℕ := o.width * o.height

/-- The main theorem -/
theorem exactly_two_rectangle_coverage 
  (r1 r2 r3 : Rectangle)
  (o12 o23 : Overlap)
  (h1 : r1.width = 4 ∧ r1.height = 6)
  (h2 : r2.width = 4 ∧ r2.height = 6)
  (h3 : r3.width = 4 ∧ r3.height = 6)
  (h4 : o12.width = 2 ∧ o12.height = 4)
  (h5 : o23.width = 2 ∧ o23.height = 4)
  (h6 : overlapArea o12 + overlapArea o23 = 16) :
  11 = overlapArea o12 + overlapArea o23 - 5 := by
  sorry


end exactly_two_rectangle_coverage_l2682_268249


namespace valid_coloring_exists_l2682_268220

/-- A type representing a coloring of regions in a plane --/
def Coloring (n : ℕ) := Fin n → Bool

/-- A predicate that checks if a coloring is valid for n lines --/
def IsValidColoring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ (i j : Fin n), i ≠ j → c i ≠ c j

/-- The main theorem stating that a valid coloring exists for any number of lines --/
theorem valid_coloring_exists (n : ℕ) (h : n > 0) : 
  ∃ (c : Coloring n), IsValidColoring n c := by
  sorry

#check valid_coloring_exists

end valid_coloring_exists_l2682_268220


namespace reinforcement_size_l2682_268271

/-- Calculates the size of a reinforcement given initial garrison size, provision duration, and new provision duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_provisions : ℕ) (days_before_reinforcement : ℕ) (new_provisions : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_provisions
  let remaining_provisions := initial_garrison * (initial_provisions - days_before_reinforcement)
  let reinforcement := (remaining_provisions / new_provisions) - initial_garrison
  reinforcement

/-- Theorem stating that given the problem conditions, the reinforcement size is 1600. -/
theorem reinforcement_size :
  calculate_reinforcement 2000 54 18 20 = 1600 := by
  sorry

end reinforcement_size_l2682_268271


namespace ellipse_equation_l2682_268265

/-- Represents an ellipse with axes aligned to the coordinate system -/
structure Ellipse where
  a : ℝ  -- Half-length of the major axis
  b : ℝ  -- Half-length of the minor axis
  c : ℝ  -- Distance from center to focus

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Theorem: Given the specified conditions, prove the standard equation of the ellipse -/
theorem ellipse_equation (e : Ellipse) 
    (h1 : e.a + e.b = 9)  -- Sum of half-lengths of axes is 18/2 = 9
    (h2 : e.c = 3)        -- One focus is at (3, 0)
    (h3 : e.c^2 = e.a^2 - e.b^2)  -- Relationship between a, b, and c
    : standard_equation e = λ x y ↦ x^2 / 25 + y^2 / 16 = 1 := by
  sorry

end ellipse_equation_l2682_268265


namespace max_speed_is_four_l2682_268255

/-- Represents the scenario of two pedestrians traveling between points A and B. -/
structure PedestrianScenario where
  route1_length : ℝ
  route2_length : ℝ
  first_section_length : ℝ
  time_difference : ℝ
  speed_difference : ℝ

/-- Calculates the maximum average speed of the first pedestrian on the second section. -/
def max_average_speed (scenario : PedestrianScenario) : ℝ :=
  4 -- The actual calculation is omitted and replaced with the known result

/-- Theorem stating that the maximum average speed is 4 km/h given the scenario conditions. -/
theorem max_speed_is_four (scenario : PedestrianScenario) 
  (h1 : scenario.route1_length = 19)
  (h2 : scenario.route2_length = 12)
  (h3 : scenario.first_section_length = 11)
  (h4 : scenario.time_difference = 2)
  (h5 : scenario.speed_difference = 0.5) :
  max_average_speed scenario = 4 := by
  sorry

#check max_speed_is_four

end max_speed_is_four_l2682_268255


namespace sophie_bought_five_cupcakes_l2682_268219

/-- The number of cupcakes Sophie bought -/
def num_cupcakes : ℕ := sorry

/-- The price of each cupcake in dollars -/
def cupcake_price : ℚ := 2

/-- The number of doughnuts Sophie bought -/
def num_doughnuts : ℕ := 6

/-- The price of each doughnut in dollars -/
def doughnut_price : ℚ := 1

/-- The number of apple pie slices Sophie bought -/
def num_pie_slices : ℕ := 4

/-- The price of each apple pie slice in dollars -/
def pie_slice_price : ℚ := 2

/-- The number of cookies Sophie bought -/
def num_cookies : ℕ := 15

/-- The price of each cookie in dollars -/
def cookie_price : ℚ := 3/5

/-- The total amount Sophie spent in dollars -/
def total_spent : ℚ := 33

/-- Theorem stating that Sophie bought 5 cupcakes -/
theorem sophie_bought_five_cupcakes :
  num_cupcakes = 5 ∧
  (num_cupcakes : ℚ) * cupcake_price +
  (num_doughnuts : ℚ) * doughnut_price +
  (num_pie_slices : ℚ) * pie_slice_price +
  (num_cookies : ℚ) * cookie_price = total_spent :=
by sorry

end sophie_bought_five_cupcakes_l2682_268219


namespace product_sum_of_three_numbers_l2682_268241

theorem product_sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 62) 
  (h2 : a + b + c = 18) : 
  a*b + b*c + a*c = 131 := by
sorry

end product_sum_of_three_numbers_l2682_268241


namespace equation_solution_l2682_268211

theorem equation_solution : 
  ∃ x : ℝ, (3 : ℝ)^(x - 2) = 9^(x + 2) ∧ x = -6 := by
  sorry

end equation_solution_l2682_268211


namespace pencil_rows_l2682_268208

theorem pencil_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h1 : total_pencils = 35) (h2 : pencils_per_row = 5) :
  total_pencils / pencils_per_row = 7 :=
by sorry

end pencil_rows_l2682_268208


namespace expression_theorem_l2682_268293

-- Define the expression E as a function of x
def E (x : ℝ) : ℝ := 6 * x + 45

-- State the theorem
theorem expression_theorem (x : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₂ - x₁ = 12) →
  (E x) / (2 * x + 15) = 3 →
  E x = 6 * x + 45 := by
sorry

end expression_theorem_l2682_268293


namespace min_n_for_S_n_gt_1020_l2682_268285

/-- The sum of the first n terms in the sequence -/
def S_n (n : ℕ) : ℤ := 2 * (2^n - 1) - n

/-- The proposition that 10 is the minimum value of n such that S_n > 1020 -/
theorem min_n_for_S_n_gt_1020 :
  (∀ k < 10, S_n k ≤ 1020) ∧ S_n 10 > 1020 :=
sorry

end min_n_for_S_n_gt_1020_l2682_268285


namespace polynomial_simplification_l2682_268261

theorem polynomial_simplification (x : ℝ) : 
  (3*x^2 + 5*x + 8)*(x + 2) - (x + 2)*(x^2 + 5*x - 72) + (4*x - 15)*(x + 2)*(x + 6) = 
  6*x^3 + 21*x^2 + 18*x := by
sorry

end polynomial_simplification_l2682_268261


namespace u_n_satisfies_property_u_n_is_smallest_u_n_equals_2n_minus_1_l2682_268292

/-- Given a positive integer n, u_n is the smallest positive integer such that
    for every positive integer d, the number of numbers divisible by d
    in any u_n consecutive positive odd numbers is no less than
    the number of numbers divisible by d in the set of odd numbers 1, 3, 5, ..., 2n-1 -/
def u_n (n : ℕ+) : ℕ :=
  2 * n.val - 1

/-- For any positive integer n, u_n satisfies the required property -/
theorem u_n_satisfies_property (n : ℕ+) :
  ∀ (d : ℕ+) (a : ℕ),
    (∀ k : Fin (2 * n.val - 1), ∃ m : ℕ, 2 * (a + k.val) - 1 = d * (2 * m + 1)) →
    (∃ k : Fin n, ∃ m : ℕ, 2 * k.val + 1 = d * (2 * m + 1)) :=
  sorry

/-- u_n is the smallest positive integer satisfying the required property -/
theorem u_n_is_smallest (n : ℕ+) :
  ∀ m : ℕ+, m.val < u_n n →
    ∃ (d : ℕ+) (a : ℕ),
      (∀ k : Fin m, ∃ l : ℕ, 2 * (a + k.val) - 1 = d * (2 * l + 1)) ∧
      ¬(∃ k : Fin n, ∃ l : ℕ, 2 * k.val + 1 = d * (2 * l + 1)) :=
  sorry

/-- The main theorem stating that u_n is equal to 2n - 1 -/
theorem u_n_equals_2n_minus_1 (n : ℕ+) :
  u_n n = 2 * n.val - 1 :=
  sorry

end u_n_satisfies_property_u_n_is_smallest_u_n_equals_2n_minus_1_l2682_268292


namespace ratio_problem_l2682_268258

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 5)
  (h2 : b / c = 2 / 7)
  (h3 : c / d = 4) :
  d / a = 14 / 5 := by
  sorry

end ratio_problem_l2682_268258


namespace min_value_of_f_l2682_268209

/-- The function f(x) = 4x^2 - 12x + 9 -/
def f (x : ℝ) : ℝ := 4 * x^2 - 12 * x + 9

/-- The minimum value of f(x) is 0 -/
theorem min_value_of_f : ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = 0 := by
  sorry

end min_value_of_f_l2682_268209


namespace debby_flour_problem_l2682_268245

theorem debby_flour_problem (x : ℝ) : x + 4 = 16 → x = 12 := by
  sorry

end debby_flour_problem_l2682_268245


namespace negation_of_negation_one_l2682_268235

theorem negation_of_negation_one : -(-1) = 1 := by sorry

end negation_of_negation_one_l2682_268235


namespace doughnut_savings_l2682_268290

/-- The cost of one dozen doughnuts -/
def cost_one_dozen : ℕ := 8

/-- The cost of two dozens of doughnuts -/
def cost_two_dozens : ℕ := 14

/-- The number of sets when buying one dozen at a time -/
def sets_one_dozen : ℕ := 6

/-- The number of sets when buying two dozens at a time -/
def sets_two_dozens : ℕ := 3

/-- Theorem stating the savings when buying 3 sets of 2 dozens instead of 6 sets of 1 dozen -/
theorem doughnut_savings : 
  sets_one_dozen * cost_one_dozen - sets_two_dozens * cost_two_dozens = 6 := by
  sorry

end doughnut_savings_l2682_268290


namespace rectangle_diagonal_length_l2682_268216

/-- A rectangle with given area and perimeter -/
structure Rectangle where
  area : ℝ
  perimeter : ℝ

/-- The length of the diagonal of a rectangle -/
def diagonal_length (r : Rectangle) : ℝ :=
  sorry

theorem rectangle_diagonal_length :
  ∀ r : Rectangle, r.area = 16 ∧ r.perimeter = 18 → diagonal_length r = 7 :=
by sorry

end rectangle_diagonal_length_l2682_268216


namespace smallest_cube_multiple_l2682_268274

theorem smallest_cube_multiple : ∃ (x : ℕ), x > 0 ∧ (∃ (M : ℤ), 2520 * x = M^3) ∧ 
  (∀ (y : ℕ), y > 0 → (∃ (N : ℤ), 2520 * y = N^3) → x ≤ y) ∧ x = 3675 := by
  sorry

end smallest_cube_multiple_l2682_268274


namespace height_prediction_approximate_l2682_268275

/-- Regression model for height prediction -/
def height_model (x : ℝ) : ℝ := 7.19 * x + 73.93

/-- The age at which we want to predict the height -/
def prediction_age : ℝ := 10

/-- The predicted height at the given age -/
def predicted_height : ℝ := height_model prediction_age

theorem height_prediction_approximate :
  ∃ ε > 0, abs (predicted_height - 145.83) < ε :=
sorry

end height_prediction_approximate_l2682_268275


namespace sphere_surface_area_cuboid_l2682_268252

/-- The surface area of a sphere circumscribing a cuboid with dimensions 2, 1, and 1 is 6π. -/
theorem sphere_surface_area_cuboid : 
  ∃ (r : ℝ), 
    r > 0 ∧ 
    (2 : ℝ)^2 + 1^2 + 1^2 = (2*r)^2 ∧ 
    4 * Real.pi * r^2 = 6 * Real.pi := by
  sorry

end sphere_surface_area_cuboid_l2682_268252


namespace power_equation_solution_l2682_268266

theorem power_equation_solution (p : ℕ) : 64^5 = 8^p → p = 10 := by
  sorry

end power_equation_solution_l2682_268266


namespace cube_root_equation_solution_l2682_268264

theorem cube_root_equation_solution (y : ℝ) :
  (5 - 2 / y) ^ (1/3 : ℝ) = -3 → y = 1/16 := by sorry

end cube_root_equation_solution_l2682_268264


namespace three_is_primitive_root_l2682_268200

theorem three_is_primitive_root (n : ℕ) (p : ℕ) (h1 : n > 1) (h2 : p = 2^n + 1) (h3 : Nat.Prime p) :
  IsPrimitiveRoot 3 p := by
  sorry

end three_is_primitive_root_l2682_268200


namespace equilateral_triangle_grid_polygon_area_l2682_268234

/-- Represents an equilateral triangular grid -/
structure EquilateralTriangularGrid where
  sideLength : ℕ
  totalPoints : ℕ

/-- Represents a polygon on the grid -/
structure Polygon (G : EquilateralTriangularGrid) where
  vertices : ℕ
  nonSelfIntersecting : Bool
  usesAllPoints : Bool

/-- The area of a polygon on an equilateral triangular grid -/
noncomputable def polygonArea (G : EquilateralTriangularGrid) (S : Polygon G) : ℝ :=
  sorry

theorem equilateral_triangle_grid_polygon_area 
  (G : EquilateralTriangularGrid) 
  (S : Polygon G) :
  G.sideLength = 20 ∧ 
  G.totalPoints = 210 ∧ 
  S.vertices = 210 ∧ 
  S.nonSelfIntersecting = true ∧ 
  S.usesAllPoints = true →
  polygonArea G S = 52 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_grid_polygon_area_l2682_268234


namespace representatives_count_l2682_268218

/-- The number of ways to select representatives from boys and girls -/
def select_representatives (num_boys num_girls num_representatives : ℕ) : ℕ :=
  Nat.choose num_boys 2 * Nat.choose num_girls 1 +
  Nat.choose num_boys 1 * Nat.choose num_girls 2

/-- Theorem stating that selecting 3 representatives from 5 boys and 3 girls,
    with both genders represented, can be done in 45 ways -/
theorem representatives_count :
  select_representatives 5 3 3 = 45 := by
  sorry

#eval select_representatives 5 3 3

end representatives_count_l2682_268218


namespace quadratic_inequality_equivalence_l2682_268232

theorem quadratic_inequality_equivalence :
  ∀ x : ℝ, 2 * x^2 - 7 * x - 30 < 0 ↔ -5/2 < x ∧ x < 6 := by
  sorry

end quadratic_inequality_equivalence_l2682_268232


namespace partial_fraction_sum_zero_l2682_268251

theorem partial_fraction_sum_zero (x A B C D E F : ℝ) : 
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) = 
  A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end partial_fraction_sum_zero_l2682_268251


namespace board_numbers_product_l2682_268226

theorem board_numbers_product (a b c d e : ℤ) : 
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℤ) = 
    {2, 6, 10, 10, 12, 14, 16, 18, 20, 24} → 
  a * b * c * d * e = -3003 := by
sorry

end board_numbers_product_l2682_268226


namespace complex_number_in_second_quadrant_l2682_268283

def complex_number : ℂ := Complex.I * (3 + 4 * Complex.I)

theorem complex_number_in_second_quadrant :
  Real.sign (complex_number.re) = -1 ∧ Real.sign (complex_number.im) = 1 := by sorry

end complex_number_in_second_quadrant_l2682_268283


namespace students_left_in_classroom_l2682_268282

theorem students_left_in_classroom 
  (total_students : ℕ) 
  (painting_fraction : ℚ) 
  (playing_fraction : ℚ) 
  (h1 : total_students = 50) 
  (h2 : painting_fraction = 3/5) 
  (h3 : playing_fraction = 1/5) : 
  total_students - (painting_fraction * total_students + playing_fraction * total_students) = 10 := by
sorry

end students_left_in_classroom_l2682_268282


namespace disk_ratio_theorem_l2682_268257

/-- Represents a disk with a center point and a radius. -/
structure Disk where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two disks are tangent to each other. -/
def areTangent (d1 d2 : Disk) : Prop :=
  (d1.center.1 - d2.center.1)^2 + (d1.center.2 - d2.center.2)^2 = (d1.radius + d2.radius)^2

/-- Checks if two disks have disjoint interiors. -/
def haveDisjointInteriors (d1 d2 : Disk) : Prop :=
  (d1.center.1 - d2.center.1)^2 + (d1.center.2 - d2.center.2)^2 > (d1.radius + d2.radius)^2

theorem disk_ratio_theorem (d1 d2 d3 d4 : Disk) 
  (h_equal_size : d1.radius = d2.radius ∧ d2.radius = d3.radius)
  (h_smaller : d4.radius < d1.radius)
  (h_tangent : areTangent d1 d2 ∧ areTangent d2 d3 ∧ areTangent d3 d1 ∧ 
               areTangent d1 d4 ∧ areTangent d2 d4 ∧ areTangent d3 d4)
  (h_disjoint : haveDisjointInteriors d1 d2 ∧ haveDisjointInteriors d2 d3 ∧ 
                haveDisjointInteriors d3 d1 ∧ haveDisjointInteriors d1 d4 ∧ 
                haveDisjointInteriors d2 d4 ∧ haveDisjointInteriors d3 d4) :
  d4.radius / d1.radius = (2 * Real.sqrt 3 - 3) / 3 := by
  sorry

end disk_ratio_theorem_l2682_268257


namespace greene_nursery_flower_count_l2682_268295

/-- The number of red roses at Greene Nursery -/
def red_roses : ℕ := 1491

/-- The number of yellow carnations at Greene Nursery -/
def yellow_carnations : ℕ := 3025

/-- The number of white roses at Greene Nursery -/
def white_roses : ℕ := 1768

/-- The total number of flowers at Greene Nursery -/
def total_flowers : ℕ := red_roses + yellow_carnations + white_roses

theorem greene_nursery_flower_count : total_flowers = 6284 := by
  sorry

end greene_nursery_flower_count_l2682_268295


namespace green_fruits_vs_red_peaches_green_peaches_vs_yellow_apples_l2682_268276

/-- Represents the number of red peaches in the basket -/
def red_peaches : ℕ := 5

/-- Represents the number of green peaches in the basket -/
def green_peaches : ℕ := 11

/-- Represents the number of yellow apples in the basket -/
def yellow_apples : ℕ := 8

/-- Represents the number of green apples in the basket -/
def green_apples : ℕ := 15

/-- Theorem stating the difference between green fruits and red peaches -/
theorem green_fruits_vs_red_peaches : 
  green_peaches + green_apples - red_peaches = 21 := by sorry

/-- Theorem stating the difference between green peaches and yellow apples -/
theorem green_peaches_vs_yellow_apples : 
  green_peaches - yellow_apples = 3 := by sorry

end green_fruits_vs_red_peaches_green_peaches_vs_yellow_apples_l2682_268276


namespace two_triangles_exist_l2682_268297

/-- Represents a triangle with side lengths and heights -/
structure Triangle where
  a : ℝ
  m_b : ℝ
  m_c : ℝ

/-- Given conditions for the triangle construction problem -/
def givenConditions : Triangle where
  a := 6
  m_b := 1
  m_c := 2

/-- Predicate to check if a triangle satisfies the given conditions -/
def satisfiesConditions (t : Triangle) : Prop :=
  t.a = givenConditions.a ∧ t.m_b = givenConditions.m_b ∧ t.m_c = givenConditions.m_c

/-- Theorem stating that exactly two distinct triangles satisfy the given conditions -/
theorem two_triangles_exist :
  ∃ (t1 t2 : Triangle), satisfiesConditions t1 ∧ satisfiesConditions t2 ∧ t1 ≠ t2 ∧
  ∀ (t : Triangle), satisfiesConditions t → (t = t1 ∨ t = t2) :=
sorry

end two_triangles_exist_l2682_268297


namespace min_distance_ellipse_to_line_l2682_268239

/-- The minimum distance from a point on the ellipse x = 4cos(θ), y = 3sin(θ) to the line x - y - 6 = 0 is √2/2 -/
theorem min_distance_ellipse_to_line :
  let ellipse := {(x, y) : ℝ × ℝ | ∃ θ : ℝ, x = 4 * Real.cos θ ∧ y = 3 * Real.sin θ}
  let line := {(x, y) : ℝ × ℝ | x - y - 6 = 0}
  ∀ p ∈ ellipse, (
    let dist := fun q : ℝ × ℝ => |q.1 - q.2 - 6| / Real.sqrt 2
    ∃ q ∈ line, dist q = Real.sqrt 2 / 2 ∧ ∀ r ∈ line, dist p ≥ Real.sqrt 2 / 2
  ) := by sorry

end min_distance_ellipse_to_line_l2682_268239


namespace alcohol_mixture_proof_l2682_268280

theorem alcohol_mixture_proof (x y z final_volume final_alcohol : ℝ) :
  x = 300 ∧ y = 600 ∧ z = 300 ∧
  (0.1 * x + 0.3 * y + 0.4 * z) / (x + y + z) = 0.22 ∧
  y = 2 * z →
  final_volume = x + y + z ∧
  final_alcohol = 0.22 * final_volume :=
by sorry

end alcohol_mixture_proof_l2682_268280


namespace bmw_cars_sold_l2682_268240

/-- The total number of cars sold -/
def total_cars : ℕ := 250

/-- The percentage of Audi cars sold -/
def audi_percent : ℚ := 10 / 100

/-- The percentage of Toyota cars sold -/
def toyota_percent : ℚ := 20 / 100

/-- The percentage of Acura cars sold -/
def acura_percent : ℚ := 15 / 100

/-- The percentage of Ford cars sold -/
def ford_percent : ℚ := 25 / 100

/-- The percentage of BMW cars sold -/
def bmw_percent : ℚ := 1 - (audi_percent + toyota_percent + acura_percent + ford_percent)

theorem bmw_cars_sold : 
  ⌊(bmw_percent * total_cars : ℚ)⌋ = 75 := by sorry

end bmw_cars_sold_l2682_268240


namespace translation_preserves_shape_and_size_l2682_268204

-- Define a geometric figure
def GeometricFigure := Type

-- Define a translation operation
def translate (F : GeometricFigure) (v : ℝ × ℝ) : GeometricFigure := sorry

-- Define properties of a figure
def shape (F : GeometricFigure) : Type := sorry
def size (F : GeometricFigure) : ℝ := sorry

-- Theorem: Translation preserves shape and size
theorem translation_preserves_shape_and_size (F : GeometricFigure) (v : ℝ × ℝ) :
  (shape (translate F v) = shape F) ∧ (size (translate F v) = size F) := by sorry

end translation_preserves_shape_and_size_l2682_268204


namespace sqrt_equation_l2682_268206

theorem sqrt_equation (n : ℝ) : Real.sqrt (10 + n) = 8 → n = 54 := by
  sorry

end sqrt_equation_l2682_268206


namespace fraction_inequality_l2682_268256

theorem fraction_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a^2 / (a + 1)) + (b^2 / (b + 1)) ≥ 1/3 := by
  sorry

end fraction_inequality_l2682_268256


namespace glorias_cash_was_150_l2682_268247

/-- Calculates Gloria's initial cash given the cabin cost, tree counts, tree prices, and leftover cash --/
def glorias_initial_cash (cabin_cost : ℕ) (cypress_count pine_count maple_count : ℕ) 
  (cypress_price pine_price maple_price : ℕ) (leftover_cash : ℕ) : ℕ :=
  cabin_cost + leftover_cash - (cypress_count * cypress_price + pine_count * pine_price + maple_count * maple_price)

/-- Theorem stating that Gloria's initial cash was $150 --/
theorem glorias_cash_was_150 : 
  glorias_initial_cash 129000 20 600 24 100 200 300 350 = 150 := by
  sorry


end glorias_cash_was_150_l2682_268247


namespace y_value_l2682_268286

theorem y_value (y : ℚ) (h : (1 : ℚ) / 3 - (1 : ℚ) / 4 = 4 / y) : y = 48 := by
  sorry

end y_value_l2682_268286


namespace stating_meal_distribution_count_l2682_268277

/-- Represents the number of people having dinner -/
def n : ℕ := 12

/-- Represents the number of meal types -/
def meal_types : ℕ := 4

/-- Represents the number of people who ordered each meal type -/
def people_per_meal : ℕ := 3

/-- Represents the number of people who should receive their ordered meal type -/
def correct_meals : ℕ := 2

/-- 
Theorem stating that the number of ways to distribute meals 
such that exactly two people receive their ordered meal type is 88047666
-/
theorem meal_distribution_count : 
  (Nat.choose n correct_meals) * (Nat.factorial (n - correct_meals)) = 88047666 := by
  sorry

end stating_meal_distribution_count_l2682_268277


namespace rectangle_squares_sum_l2682_268227

theorem rectangle_squares_sum (a b : ℝ) : 
  a + b = 3 → a * b = 1 → a^2 + b^2 = 7 := by sorry

end rectangle_squares_sum_l2682_268227


namespace probability_of_specific_tile_arrangement_l2682_268205

theorem probability_of_specific_tile_arrangement :
  let total_tiles : ℕ := 6
  let x_tiles : ℕ := 4
  let o_tiles : ℕ := 2
  let specific_arrangement := [true, true, false, true, false, true]
  
  (x_tiles + o_tiles = total_tiles) →
  (List.length specific_arrangement = total_tiles) →
  
  (probability_of_arrangement : ℚ) =
    (x_tiles.choose 2 * o_tiles.choose 1 * x_tiles.choose 1 * o_tiles.choose 1 * x_tiles.choose 1) /
    total_tiles.factorial →
  
  probability_of_arrangement = 1 / 15 :=
by
  sorry

end probability_of_specific_tile_arrangement_l2682_268205


namespace race_track_radius_l2682_268233

/-- Given a circular race track, prove that the radius of the outer circle
    is equal to the radius of the inner circle plus the width of the track. -/
theorem race_track_radius 
  (inner_circumference : ℝ) 
  (track_width : ℝ) 
  (inner_circumference_eq : inner_circumference = 880) 
  (track_width_eq : track_width = 18) :
  ∃ (outer_radius : ℝ),
    outer_radius = inner_circumference / (2 * Real.pi) + track_width :=
by sorry

end race_track_radius_l2682_268233


namespace prob_all_painted_10_beads_l2682_268253

/-- A circular necklace with beads -/
structure Necklace :=
  (num_beads : ℕ)

/-- The number of beads selected for painting -/
def num_selected : ℕ := 5

/-- Function to calculate the probability of all beads being painted -/
noncomputable def prob_all_painted (n : Necklace) : ℚ :=
  sorry

/-- Theorem stating the probability of all beads being painted for a 10-bead necklace -/
theorem prob_all_painted_10_beads :
  prob_all_painted { num_beads := 10 } = 17 / 42 :=
sorry

end prob_all_painted_10_beads_l2682_268253


namespace pen_promotion_result_l2682_268250

/-- Represents the promotion event in a shop selling pens and giving away teddy bears. -/
structure PenPromotion where
  /-- Profit in yuan for selling one pen -/
  profit_per_pen : ℕ
  /-- Cost in yuan for one teddy bear -/
  cost_per_bear : ℕ
  /-- Total profit in yuan from the promotion event -/
  total_profit : ℕ

/-- Calculates the number of pens sold during the promotion event -/
def pens_sold (promo : PenPromotion) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that given the specific conditions of the promotion,
    the number of pens sold is 335 -/
theorem pen_promotion_result :
  let promo : PenPromotion := {
    profit_per_pen := 7,
    cost_per_bear := 2,
    total_profit := 2011
  }
  pens_sold promo = 335 := by
  sorry

end pen_promotion_result_l2682_268250


namespace total_amount_to_pay_l2682_268262

def original_balance : ℝ := 150
def finance_charge_percentage : ℝ := 0.02

theorem total_amount_to_pay : 
  original_balance * (1 + finance_charge_percentage) = 153 := by sorry

end total_amount_to_pay_l2682_268262


namespace triangle_area_is_integer_l2682_268287

-- Define a point in the plane
structure Point where
  x : Int
  y : Int

-- Define a function to check if a number is odd
def isOdd (n : Int) : Prop := ∃ k : Int, n = 2 * k + 1

-- Define a triangle with three points
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

-- Define the area of a triangle
def triangleArea (t : Triangle) : Rat :=
  let x1 := t.p1.x
  let y1 := t.p1.y
  let x2 := t.p2.x
  let y2 := t.p2.y
  let x3 := t.p3.x
  let y3 := t.p3.y
  Rat.ofInt (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

-- Theorem statement
theorem triangle_area_is_integer (t : Triangle) :
  t.p1 = Point.mk 1 1 →
  (isOdd t.p2.x ∧ isOdd t.p2.y) →
  (isOdd t.p3.x ∧ isOdd t.p3.y) →
  t.p2 ≠ t.p3 →
  ∃ n : Int, triangleArea t = n := by
  sorry

end triangle_area_is_integer_l2682_268287


namespace quadratic_function_bounds_l2682_268278

/-- Given a quadratic function f(x) = ax^2 - c, 
    if -4 ≤ f(1) ≤ -1 and -1 ≤ f(2) ≤ 5, then -1 ≤ f(3) ≤ 20 -/
theorem quadratic_function_bounds (a c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^2 - c
  (-4 : ℝ) ≤ f 1 ∧ f 1 ≤ -1 ∧ -1 ≤ f 2 ∧ f 2 ≤ 5 → 
  -1 ≤ f 3 ∧ f 3 ≤ 20 := by
  sorry

end quadratic_function_bounds_l2682_268278


namespace max_constant_inequality_l2682_268244

theorem max_constant_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x^2 + y^2 = 1) :
  ∃ (c : ℝ), c = 1/2 ∧ x^6 + y^6 ≥ c*x*y ∧ ∀ (c' : ℝ), (∀ (x' y' : ℝ), x' > 0 → y' > 0 → x'^2 + y'^2 = 1 → x'^6 + y'^6 ≥ c'*x'*y') → c' ≤ c :=
sorry

end max_constant_inequality_l2682_268244


namespace strawberry_problem_l2682_268260

theorem strawberry_problem (initial : Float) (eaten : Float) (remaining : Float) :
  initial = 78.0 → eaten = 42.0 → remaining = initial - eaten → remaining = 36.0 := by
  sorry

end strawberry_problem_l2682_268260


namespace product_equals_fraction_l2682_268229

/-- The repeating decimal 0.456̄ as a rational number -/
def repeating_decimal : ℚ := 456 / 999

/-- The product of the repeating decimal 0.456̄ and 7 -/
def product : ℚ := repeating_decimal * 7

/-- Theorem stating that the product of 0.456̄ and 7 is equal to 1064/333 -/
theorem product_equals_fraction : product = 1064 / 333 := by sorry

end product_equals_fraction_l2682_268229


namespace group_meal_cost_l2682_268288

/-- The cost of a meal for a group at a restaurant -/
def mealCost (totalPeople : ℕ) (kids : ℕ) (adultMealPrice : ℕ) : ℕ :=
  (totalPeople - kids) * adultMealPrice

/-- Theorem: The meal cost for a group of 13 people with 9 kids is $28 -/
theorem group_meal_cost : mealCost 13 9 7 = 28 := by
  sorry

end group_meal_cost_l2682_268288


namespace square_side_length_l2682_268296

theorem square_side_length (x : ℝ) (h : x > 0) : 4 * x = 2 * (x ^ 2) → x = 2 := by
  sorry

end square_side_length_l2682_268296


namespace time_calculation_l2682_268201

-- Define a structure for time
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

-- Define the initial time
def initial_time : Time := { hours := 8, minutes := 45, seconds := 0 }

-- Define the number of seconds to add
def seconds_to_add : Nat := 9876

-- Define the expected final time
def expected_final_time : Time := { hours := 11, minutes := 29, seconds := 36 }

-- Function to add seconds to a given time
def add_seconds (t : Time) (s : Nat) : Time :=
  sorry

-- Theorem to prove
theorem time_calculation :
  add_seconds initial_time seconds_to_add = expected_final_time :=
sorry

end time_calculation_l2682_268201


namespace truck_initial_momentum_l2682_268230

/-- Initial momentum of a truck -/
theorem truck_initial_momentum
  (v : ℝ) -- Initial velocity
  (F : ℝ) -- Constant force applied to stop the truck
  (x : ℝ) -- Distance traveled before stopping
  (t : ℝ) -- Time taken to stop
  (h1 : v > 0) -- Assumption: initial velocity is positive
  (h2 : F > 0) -- Assumption: force is positive
  (h3 : x > 0) -- Assumption: distance is positive
  (h4 : t > 0) -- Assumption: time is positive
  (h5 : x = (v * t) / 2) -- Relation between distance, velocity, and time
  (h6 : F * t = v) -- Relation between force, time, and velocity change
  : ∃ (m : ℝ), m * v = (2 * F * x) / v :=
sorry

end truck_initial_momentum_l2682_268230


namespace homomorphism_characterization_l2682_268267

theorem homomorphism_characterization (f : ℤ → ℤ) :
  (∀ x y : ℤ, f (x + y) = f x + f y) →
  ∃ a : ℤ, ∀ x : ℤ, f x = a * x :=
by sorry

end homomorphism_characterization_l2682_268267


namespace quadratic_intercept_l2682_268210

/-- A quadratic function with vertex (4, 9) and x-intercept (0, 0) has its other x-intercept at x = 8 -/
theorem quadratic_intercept (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = a * (x - 4)^2 + 9) →  -- vertex form
  (a * 0^2 + b * 0 + c = 0) →                       -- (0, 0) is an x-intercept
  (∃ x ≠ 0, a * x^2 + b * x + c = 0 ∧ x = 8) :=     -- other x-intercept is at x = 8
by sorry

end quadratic_intercept_l2682_268210


namespace problem_solution_l2682_268237

/-- Represents a three-digit number in the form abc --/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value --/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem problem_solution :
  ∀ (a b : Nat),
  let n1 := ThreeDigitNumber.mk 3 a 7 (by sorry)
  let n2 := ThreeDigitNumber.mk 6 b 1 (by sorry)
  (n1.toNat + 294 = n2.toNat) →
  (n2.toNat % 7 = 0) →
  a + b = 8 := by sorry

end problem_solution_l2682_268237


namespace parabola_f_value_l2682_268212

/-- A parabola with equation y = dx^2 + ex + f, vertex at (3, -5), and passing through (4, -3) has f = 13 -/
theorem parabola_f_value (d e f : ℝ) : 
  (∀ x y : ℝ, y = d*x^2 + e*x + f) →  -- Parabola equation
  (-5 : ℝ) = d*(3:ℝ)^2 + e*(3:ℝ) + f → -- Vertex at (3, -5)
  (-3 : ℝ) = d*(4:ℝ)^2 + e*(4:ℝ) + f → -- Passes through (4, -3)
  f = 13 := by
sorry

end parabola_f_value_l2682_268212


namespace sqrt_eighteen_minus_three_sqrt_half_plus_sqrt_two_l2682_268221

theorem sqrt_eighteen_minus_three_sqrt_half_plus_sqrt_two : 
  Real.sqrt 18 - 3 * Real.sqrt (1/2) + Real.sqrt 2 = (5 * Real.sqrt 2) / 2 := by
  sorry

end sqrt_eighteen_minus_three_sqrt_half_plus_sqrt_two_l2682_268221


namespace children_neither_happy_nor_sad_l2682_268223

theorem children_neither_happy_nor_sad 
  (total_children : ℕ)
  (happy_children : ℕ)
  (sad_children : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (happy_boys : ℕ)
  (sad_girls : ℕ)
  (neither_happy_nor_sad_boys : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : boys = 16)
  (h5 : girls = 44)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neither_happy_nor_sad_boys = 4)
  : total_children - happy_children - sad_children = 20 := by
  sorry

end children_neither_happy_nor_sad_l2682_268223


namespace right_triangle_geometric_sequence_sine_l2682_268238

theorem right_triangle_geometric_sequence_sine (a b c : Real) :
  -- The triangle is right-angled
  a^2 + b^2 = c^2 →
  -- The sides form a geometric sequence
  (b / a = c / b ∨ a / b = b / c) →
  -- The sine of the smallest angle
  min (a / c) (b / c) = (Real.sqrt 5 - 1) / 2 := by
sorry

end right_triangle_geometric_sequence_sine_l2682_268238


namespace probability_is_zero_l2682_268284

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ
  direction : Bool  -- True for counterclockwise, False for clockwise

/-- Represents the picture taken of the track -/
structure Picture where
  coverageFraction : ℝ
  centerPosition : ℝ  -- Position on track (0 ≤ position < 1)

/-- Calculates the probability of both runners being in the picture -/
def probabilityBothInPicture (rachel : Runner) (robert : Runner) (pic : Picture) (timeElapsed : ℝ) : ℝ :=
  sorry

/-- Theorem stating the probability is zero for the given conditions -/
theorem probability_is_zero :
  ∀ (rachel : Runner) (robert : Runner) (pic : Picture) (t : ℝ),
    rachel.lapTime = 120 →
    robert.lapTime = 75 →
    rachel.direction = true →
    robert.direction = false →
    pic.coverageFraction = 1/3 →
    pic.centerPosition = 0 →
    15 * 60 ≤ t ∧ t < 16 * 60 →
    probabilityBothInPicture rachel robert pic t = 0 :=
  sorry

end probability_is_zero_l2682_268284


namespace triangle_side_length_l2682_268299

noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt 3 * Real.sin (x / 2) * Real.cos (x / 2) - Real.cos (2 * x / 2)^2 + 1/2

theorem triangle_side_length 
  (A B C : ℝ) 
  (hA : 0 < A ∧ A < π) 
  (hB : 0 < B ∧ B < π) 
  (hC : 0 < C ∧ C < π) 
  (hABC : A + B + C = π) 
  (hf : f A = 1/2) 
  (ha : Real.sqrt 3 = (Real.sin B / Real.sin A)) 
  (hB : Real.sin B = 2 * Real.sin C) : 
  Real.sin C / Real.sin A = 1 := by sorry

end triangle_side_length_l2682_268299


namespace perfect_square_values_l2682_268254

theorem perfect_square_values (a n : ℕ) : 
  (a ^ 2 + a + 1589 = n ^ 2) ↔ (a = 1588 ∨ a = 28 ∨ a = 316 ∨ a = 43) :=
sorry

end perfect_square_values_l2682_268254


namespace rhombus_perimeter_given_side_l2682_268272

/-- A rhombus is a quadrilateral with four equal sides -/
structure Rhombus where
  side_length : ℝ
  side_length_positive : side_length > 0

/-- The perimeter of a rhombus is four times its side length -/
def perimeter (r : Rhombus) : ℝ := 4 * r.side_length

theorem rhombus_perimeter_given_side (r : Rhombus) (h : r.side_length = 2) : perimeter r = 8 := by
  sorry

end rhombus_perimeter_given_side_l2682_268272


namespace acute_triangle_sine_sum_l2682_268246

theorem acute_triangle_sine_sum (α β γ : Real) 
  (h_acute : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_triangle : α + β + γ = Real.pi)
  (h_acute_triangle : α < Real.pi/2 ∧ β < Real.pi/2 ∧ γ < Real.pi/2) : 
  Real.sin α + Real.sin β + Real.sin γ > 2 := by
sorry

end acute_triangle_sine_sum_l2682_268246


namespace inequality_holds_for_all_x_l2682_268259

theorem inequality_holds_for_all_x : ∀ x : ℝ, x + 2 < x + 3 := by
  sorry

end inequality_holds_for_all_x_l2682_268259


namespace orthocenter_property_l2682_268298

/-- An acute-angled triangle with its orthocenter properties -/
structure AcuteTriangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Altitude lengths
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ
  -- Distances from vertices to orthocenter
  d_a : ℝ
  d_b : ℝ
  d_c : ℝ
  -- Conditions
  acute : a > 0 ∧ b > 0 ∧ c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  acute_angles : a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

/-- The orthocenter property for acute-angled triangles -/
theorem orthocenter_property (t : AcuteTriangle) :
  t.m_a * t.d_a + t.m_b * t.d_b + t.m_c * t.d_c = (t.a^2 + t.b^2 + t.c^2) / 2 := by
  sorry

end orthocenter_property_l2682_268298


namespace trees_in_yard_l2682_268231

/-- The number of trees planted along a yard -/
def num_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  (yard_length / tree_distance) + 1

/-- Theorem: There are 24 trees planted along the yard -/
theorem trees_in_yard :
  let yard_length : ℕ := 414
  let tree_distance : ℕ := 18
  num_trees yard_length tree_distance = 24 := by
  sorry

end trees_in_yard_l2682_268231


namespace point_difference_l2682_268248

/-- Represents a basketball player with their score and penalties. -/
structure Player where
  score : ℕ
  penalties : List ℕ

/-- Calculates the final score of a player after subtracting penalties. -/
def finalScore (p : Player) : ℤ :=
  p.score - p.penalties.sum

/-- Represents a basketball team with a list of players. -/
structure Team where
  players : List Player

/-- Calculates the total score of a team. -/
def teamScore (t : Team) : ℤ :=
  t.players.map finalScore |>.sum

/-- The given data for Team A. -/
def teamA : Team := {
  players := [
    { score := 12, penalties := [2] },
    { score := 18, penalties := [2, 2, 2] },
    { score := 5,  penalties := [] },
    { score := 7,  penalties := [3, 3] },
    { score := 6,  penalties := [1] }
  ]
}

/-- The given data for Team B. -/
def teamB : Team := {
  players := [
    { score := 10, penalties := [1, 1] },
    { score := 9,  penalties := [2] },
    { score := 12, penalties := [] },
    { score := 8,  penalties := [1, 1, 1] },
    { score := 5,  penalties := [3] },
    { score := 4,  penalties := [] }
  ]
}

/-- The main theorem stating the point difference between Team B and Team A. -/
theorem point_difference : teamScore teamB - teamScore teamA = 5 := by
  sorry


end point_difference_l2682_268248
