import Mathlib

namespace proportion_solution_l229_22966

theorem proportion_solution (x : ℝ) : (0.6 / x = 5 / 8) → x = 0.96 := by
  sorry

end proportion_solution_l229_22966


namespace ellipse_properties_l229_22903

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Definition of the line l -/
def line_l (k m x y : ℝ) : Prop := y = k * x + m

/-- Theorem stating the properties of the ellipse and its intersections -/
theorem ellipse_properties :
  ∀ (k m : ℝ),
  m > 0 →
  (∃ (A B : ℝ × ℝ),
    ellipse_C A.1 A.2 ∧
    ellipse_C B.1 B.2 ∧
    line_l k m A.1 A.2 ∧
    line_l k m B.1 B.2 ∧
    (k = 1/2 ∨ k = -1/2) →
    (∃ (c : ℝ), A.1^2 + A.2^2 + B.1^2 + B.2^2 = c) ∧
    (∃ (area : ℝ), area ≤ 1 ∧
      (k = 1/2 ∨ k = -1/2) →
      area = 1)) :=
sorry

end ellipse_properties_l229_22903


namespace seven_at_eight_equals_28_div_9_l229_22922

/-- The '@' operation for positive integers -/
def at_op (a b : ℕ+) : ℚ :=
  (a.val * b.val : ℚ) / (a.val + b.val + 3 : ℚ)

/-- Theorem: 7 @ 8 = 28/9 -/
theorem seven_at_eight_equals_28_div_9 : 
  at_op ⟨7, by norm_num⟩ ⟨8, by norm_num⟩ = 28 / 9 := by
  sorry

end seven_at_eight_equals_28_div_9_l229_22922


namespace collinear_points_k_value_l229_22934

/-- Three points are collinear if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- If the points (2, 3), (7, k), and (15, 4) are collinear, then k = 44/13. -/
theorem collinear_points_k_value :
  collinear 2 3 7 k 15 4 → k = 44 / 13 :=
by sorry

end collinear_points_k_value_l229_22934


namespace parabola_point_distance_l229_22962

theorem parabola_point_distance (x₀ y₀ : ℝ) : 
  y₀^2 = 8 * x₀ →  -- Point (x₀, y₀) is on the parabola y² = 8x
  (x₀ - 2)^2 + y₀^2 = 3^2 →  -- Distance from (x₀, y₀) to focus (2, 0) is 3
  |y₀| = 2 * Real.sqrt 2 := by
sorry

end parabola_point_distance_l229_22962


namespace rationalize_denominator_l229_22935

theorem rationalize_denominator : 3 / Real.sqrt 48 = Real.sqrt 3 / 4 := by
  sorry

end rationalize_denominator_l229_22935


namespace exponential_inequality_l229_22928

theorem exponential_inequality (x a b : ℝ) 
  (h_x_pos : x > 0) 
  (h_ineq : 0 < b^x ∧ b^x < a^x ∧ a^x < 1) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0) : 
  1 > a ∧ a > b := by sorry

end exponential_inequality_l229_22928


namespace john_repair_results_l229_22911

/-- Represents the repair job details for John --/
structure RepairJob where
  totalCars : ℕ
  standardRepairCars : ℕ
  standardRepairTime : ℕ
  longerRepairPercent : ℚ
  hourlyRate : ℚ

/-- Calculates the total repair time and money earned for a given repair job --/
def calculateRepairResults (job : RepairJob) : ℚ × ℚ :=
  let standardTime := job.standardRepairCars * job.standardRepairTime
  let longerRepairTime := job.standardRepairTime * (1 + job.longerRepairPercent)
  let longerRepairCars := job.totalCars - job.standardRepairCars
  let longerTime := longerRepairCars * longerRepairTime
  let totalMinutes := standardTime + longerTime
  let totalHours := totalMinutes / 60
  let moneyEarned := totalHours * job.hourlyRate
  (totalHours, moneyEarned)

/-- Theorem stating that for John's specific repair job, the total repair time is 11 hours and he earns $330 --/
theorem john_repair_results :
  let job : RepairJob := {
    totalCars := 10,
    standardRepairCars := 6,
    standardRepairTime := 50,
    longerRepairPercent := 4/5,
    hourlyRate := 30
  }
  calculateRepairResults job = (11, 330) := by sorry

end john_repair_results_l229_22911


namespace pure_imaginary_condition_l229_22963

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ b : ℝ, (2 - a * I) / (1 + I) = b * I) ↔ a = 2 :=
by sorry

end pure_imaginary_condition_l229_22963


namespace product_of_four_consecutive_integers_l229_22929

theorem product_of_four_consecutive_integers (X : ℤ) :
  X * (X + 1) * (X + 2) * (X + 3) = (X^2 + 3*X + 1)^2 - 1 := by
  sorry

end product_of_four_consecutive_integers_l229_22929


namespace closest_integer_to_cube_root_250_l229_22910

theorem closest_integer_to_cube_root_250 :
  ∃ (n : ℤ), ∀ (m : ℤ), |n - (250 : ℝ)^(1/3)| ≤ |m - (250 : ℝ)^(1/3)| ∧ n = 6 :=
sorry

end closest_integer_to_cube_root_250_l229_22910


namespace part_one_disproof_part_two_proof_l229_22968

-- Part 1
theorem part_one_disproof (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z ≥ 3) :
  ¬ (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z ≥ 3 → 1/x + 1/y + 1/z ≤ 3) :=
sorry

-- Part 2
theorem part_two_proof (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z ≤ 3) :
  1/x + 1/y + 1/z ≥ 3 :=
sorry

end part_one_disproof_part_two_proof_l229_22968


namespace optimal_timing_problem_l229_22941

/-- Represents the optimal timing problem for three people traveling between two points. -/
theorem optimal_timing_problem (distance : ℝ) (walking_speed : ℝ) (bicycle_speed : ℝ) 
  (h_distance : distance = 15)
  (h_walking_speed : walking_speed = 6)
  (h_bicycle_speed : bicycle_speed = 15) :
  ∃ (optimal_time : ℝ),
    optimal_time = 3 / 11 ∧
    (∀ (t : ℝ), 
      let time_A := distance / walking_speed + (distance - walking_speed * t) / bicycle_speed
      let time_B := t + (distance - bicycle_speed * t) / walking_speed
      let time_C := distance / walking_speed - t
      (time_A = time_B ∧ time_B = time_C) → t = optimal_time) :=
by sorry

end optimal_timing_problem_l229_22941


namespace initial_children_on_bus_l229_22978

/-- Given that 14 more children got on a bus at a bus stop, 
    resulting in a total of 78 children, prove that there were 
    initially 64 children on the bus. -/
theorem initial_children_on_bus : 
  ∀ (initial : ℕ), initial + 14 = 78 → initial = 64 := by
  sorry

end initial_children_on_bus_l229_22978


namespace fraction_cubed_l229_22959

theorem fraction_cubed : (3 / 4 : ℚ) ^ 3 = 27 / 64 := by
  sorry

end fraction_cubed_l229_22959


namespace polynomial_no_real_roots_l229_22954

theorem polynomial_no_real_roots :
  ∀ x : ℝ, 4 * x^8 - 2 * x^7 + x^6 - 3 * x^4 + x^2 - x + 1 > 0 := by
  sorry

end polynomial_no_real_roots_l229_22954


namespace prob_sum_three_two_dice_l229_22938

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := num_faces * num_faces

/-- The number of ways to roll a sum of 3 with two dice -/
def favorable_outcomes : ℕ := 2

/-- The probability of an event occurring -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

/-- Theorem: The probability of rolling a sum of 3 with two fair dice is 1/18 -/
theorem prob_sum_three_two_dice : 
  probability favorable_outcomes total_outcomes = 1 / 18 := by sorry

end prob_sum_three_two_dice_l229_22938


namespace area_between_concentric_circles_l229_22904

theorem area_between_concentric_circles (r : ℝ) (h1 : r = 2) (h2 : r > 0) : 
  π * (5 * r)^2 - π * r^2 = 96 * π := by
  sorry

end area_between_concentric_circles_l229_22904


namespace prob_not_adjacent_seven_chairs_l229_22916

/-- The number of chairs in the row -/
def n : ℕ := 7

/-- The number of ways two people can sit next to each other in a row of n chairs -/
def adjacent_seating (n : ℕ) : ℕ := n - 1

/-- The total number of ways two people can choose seats from n chairs -/
def total_seating (n : ℕ) : ℕ := n.choose 2

/-- The probability that Mary and James don't sit next to each other
    when randomly choosing seats in a row of n chairs -/
def prob_not_adjacent (n : ℕ) : ℚ :=
  1 - (adjacent_seating n : ℚ) / (total_seating n : ℚ)

theorem prob_not_adjacent_seven_chairs :
  prob_not_adjacent n = 5/7 := by sorry

end prob_not_adjacent_seven_chairs_l229_22916


namespace vacation_savings_l229_22958

def total_income : ℝ := 72800
def total_expenses : ℝ := 54200
def deposit_rate : ℝ := 0.1

theorem vacation_savings : 
  let remaining := total_income - total_expenses
  let deposit := deposit_rate * remaining
  total_income - total_expenses - deposit = 16740 := by
  sorry

end vacation_savings_l229_22958


namespace rectangular_solid_surface_area_l229_22936

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- The surface area of a rectangular solid with dimensions a, b, and c. -/
def surface_area (a b c : ℕ) : ℕ :=
  2 * (a * b + b * c + c * a)

/-- The volume of a rectangular solid with dimensions a, b, and c. -/
def volume (a b c : ℕ) : ℕ :=
  a * b * c

theorem rectangular_solid_surface_area (a b c : ℕ) :
  is_prime a ∧ is_prime b ∧ is_prime c ∧ volume a b c = 308 →
  surface_area a b c = 226 := by
  sorry

end rectangular_solid_surface_area_l229_22936


namespace predict_sales_at_34_degrees_l229_22991

/-- Represents the linear regression model for cold drink sales -/
structure ColdDrinkSalesModel where
  /-- Calculates the predicted sales volume based on temperature -/
  predict : ℝ → ℝ

/-- Theorem: Given the linear regression model ŷ = 2x + 60, 
    the predicted sales volume for a day with highest temperature 34°C is 128 cups -/
theorem predict_sales_at_34_degrees 
  (model : ColdDrinkSalesModel)
  (h_model : ∀ x, model.predict x = 2 * x + 60) :
  model.predict 34 = 128 := by
  sorry

end predict_sales_at_34_degrees_l229_22991


namespace min_value_reciprocal_sum_l229_22915

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  (1 / x + 1 / y) ≥ 1 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x * y = 4 ∧ 1 / x + 1 / y = 1 :=
sorry

end min_value_reciprocal_sum_l229_22915


namespace art_collection_unique_paintings_l229_22907

theorem art_collection_unique_paintings
  (shared : ℕ)
  (andrew_total : ℕ)
  (john_unique : ℕ)
  (h1 : shared = 15)
  (h2 : andrew_total = 25)
  (h3 : john_unique = 8) :
  andrew_total - shared + john_unique = 18 :=
by sorry

end art_collection_unique_paintings_l229_22907


namespace adam_final_spend_l229_22971

/-- Represents a purchased item with its weight and price per kilogram -/
structure Item where
  weight : Float
  price_per_kg : Float

/-- Calculates the total cost of purchases before discounts -/
def total_cost (items : List Item) : Float :=
  items.foldl (λ acc item => acc + item.weight * item.price_per_kg) 0

/-- Applies the almonds and walnuts discount if eligible -/
def apply_nuts_discount (almonds_cost cashews_cost total : Float) : Float :=
  if almonds_cost + cashews_cost ≥ 2.5 * 10 then
    total - 0.1 * (almonds_cost + cashews_cost)
  else
    total

/-- Applies the overall purchase discount if eligible -/
def apply_overall_discount (total : Float) : Float :=
  if total > 100 then total * 0.95 else total

/-- Theorem stating that Adam's final spend is $69.1 -/
theorem adam_final_spend :
  let items : List Item := [
    { weight := 1.5, price_per_kg := 12 },  -- almonds
    { weight := 1,   price_per_kg := 10 },  -- walnuts
    { weight := 0.5, price_per_kg := 20 },  -- cashews
    { weight := 1,   price_per_kg := 8 },   -- raisins
    { weight := 1.5, price_per_kg := 6 },   -- apricots
    { weight := 0.8, price_per_kg := 15 },  -- pecans
    { weight := 0.7, price_per_kg := 7 }    -- dates
  ]
  let initial_total := total_cost items
  let almonds_cost := 1.5 * 12
  let walnuts_cost := 1 * 10
  let after_nuts_discount := apply_nuts_discount almonds_cost walnuts_cost initial_total
  let final_total := apply_overall_discount after_nuts_discount
  final_total = 69.1 := by
  sorry

end adam_final_spend_l229_22971


namespace abs_neg_2023_l229_22933

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by sorry

end abs_neg_2023_l229_22933


namespace cone_apex_angle_l229_22901

theorem cone_apex_angle (r : ℝ) (h : ℝ) (l : ℝ) (θ : ℝ) : 
  r > 0 → h > 0 → l > 0 →
  l = 2 * r →  -- ratio of lateral area to base area is 2
  h = r * Real.sqrt 3 →  -- derived from Pythagorean theorem
  θ = 2 * Real.arctan (1 / Real.sqrt 3) →  -- definition of apex angle
  θ = π / 3  -- 60 degrees in radians
:= by sorry

end cone_apex_angle_l229_22901


namespace congruence_solution_l229_22990

theorem congruence_solution : ∃ n : ℕ, n ≤ 4 ∧ n ≡ -2323 [ZMOD 5] ∧ n = 2 := by
  sorry

end congruence_solution_l229_22990


namespace science_homework_duration_l229_22989

/-- Calculates the time remaining for science homework given the total time and time spent on other subjects. -/
def science_homework_time (total_time math_time english_time history_time project_time : ℕ) : ℕ :=
  total_time - (math_time + english_time + history_time + project_time)

/-- Proves that given the specified times for total work and other subjects, the remaining time for science homework is 50 minutes. -/
theorem science_homework_duration :
  science_homework_time 180 45 30 25 30 = 50 := by
  sorry

end science_homework_duration_l229_22989


namespace laticia_socks_l229_22945

/-- Proves that Laticia knitted 13 pairs of socks in the first week -/
theorem laticia_socks (x : ℕ) : x + (x + 4) + (x + 2) + (x - 1) = 57 → x = 13 := by
  sorry

end laticia_socks_l229_22945


namespace min_a_value_l229_22961

theorem min_a_value (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → x^2 + a*x + 1 ≥ 0) →
  a ≥ -5/2 :=
sorry

end min_a_value_l229_22961


namespace quadratic_equation_roots_l229_22956

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ x = -2) → 
  m = -12 ∧ ∃ y : ℝ, y^2 - 4*y + m = 0 ∧ y = 6 := by
  sorry

end quadratic_equation_roots_l229_22956


namespace divide_and_power_l229_22943

theorem divide_and_power : (5 / (1 / 5)) ^ 3 = 15625 := by sorry

end divide_and_power_l229_22943


namespace simplify_fraction_product_l229_22952

theorem simplify_fraction_product : 16 * (-24 / 5) * (45 / 56) = -2160 / 7 := by
  sorry

end simplify_fraction_product_l229_22952


namespace perimeter_ratio_of_squares_with_diagonal_ratio_l229_22973

theorem perimeter_ratio_of_squares_with_diagonal_ratio (d : ℝ) :
  let d1 := d
  let d2 := 4 * d
  let s1 := d1 / Real.sqrt 2
  let s2 := d2 / Real.sqrt 2
  let p1 := 4 * s1
  let p2 := 4 * s2
  p2 / p1 = 8 := by sorry

end perimeter_ratio_of_squares_with_diagonal_ratio_l229_22973


namespace walter_chores_l229_22983

/-- The number of days Walter worked -/
def total_days : ℕ := 10

/-- Walter's earnings for a regular day -/
def regular_pay : ℕ := 3

/-- Walter's earnings for an exceptional day -/
def exceptional_pay : ℕ := 5

/-- Walter's total earnings -/
def total_earnings : ℕ := 36

/-- The number of days Walter did chores exceptionally well -/
def exceptional_days : ℕ := 3

/-- The number of days Walter did regular chores -/
def regular_days : ℕ := total_days - exceptional_days

theorem walter_chores :
  regular_days * regular_pay + exceptional_days * exceptional_pay = total_earnings ∧
  regular_days + exceptional_days = total_days :=
by sorry

end walter_chores_l229_22983


namespace square_sum_proof_l229_22985

theorem square_sum_proof (a b : ℝ) (h1 : a - b = 6) (h2 : a * b = 32) : a^2 + b^2 = 100 := by
  sorry

end square_sum_proof_l229_22985


namespace function_properties_l229_22982

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

-- Define the theorem
theorem function_properties (a m n : ℝ) 
  (h_m_pos : m > 0) (h_n_pos : n > 0) (h_m_neq_n : m ≠ n)
  (h_fm : f a m = 3) (h_fn : f a n = 3) :
  0 < a ∧ a < Real.exp 2 ∧ a^2 < m * n ∧ m * n < a * Real.exp 2 := by
  sorry

end

end function_properties_l229_22982


namespace problem_solution_l229_22906

theorem problem_solution (x : ℝ) (h : (1 : ℝ) / 4 + 4 * ((1 : ℝ) / 2013 + 1 / x) = 7 / 4) :
  1872 + 48 * ((2013 : ℝ) * x / (x + 2013)) = 2000 := by
  sorry

end problem_solution_l229_22906


namespace almonds_problem_l229_22996

theorem almonds_problem (lily_almonds jack_almonds : ℕ) : 
  lily_almonds = jack_almonds + 8 →
  jack_almonds = lily_almonds / 3 →
  lily_almonds = 12 := by
sorry

end almonds_problem_l229_22996


namespace min_sum_abs_values_l229_22924

def matrix_condition (a b c d : ℤ) : Prop :=
  let M : Matrix (Fin 2) (Fin 2) ℤ := !![a, b; c, d]
  M ^ 2 = !![5, 0; 0, 5]

theorem min_sum_abs_values (a b c d : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h_matrix : matrix_condition a b c d) :
  (∀ a' b' c' d' : ℤ, a' ≠ 0 → b' ≠ 0 → c' ≠ 0 → d' ≠ 0 → 
    matrix_condition a' b' c' d' → 
    |a| + |b| + |c| + |d| ≤ |a'| + |b'| + |c'| + |d'|) ∧
  |a| + |b| + |c| + |d| = 6 :=
sorry

end min_sum_abs_values_l229_22924


namespace partition_large_rectangle_l229_22923

/-- Definition of a "good" rectangle -/
inductive GoodRectangle
  | square : GoodRectangle
  | rectangle : GoodRectangle

/-- Predicate to check if a rectangle can be partitioned into good rectangles -/
def can_partition (a b : ℕ) : Prop :=
  ∃ (num_squares num_rectangles : ℕ),
    2 * 2 * num_squares + 1 * 11 * num_rectangles = a * b

/-- Theorem: Any rectangle with integer sides greater than 100 can be partitioned into good rectangles -/
theorem partition_large_rectangle (a b : ℕ) (ha : a > 100) (hb : b > 100) :
  can_partition a b := by
  sorry


end partition_large_rectangle_l229_22923


namespace class_size_problem_l229_22972

theorem class_size_problem (x y : ℕ) : 
  y = x / 6 →  -- Initial condition: absent = 1/6 of present
  y = (x - 1) / 5 →  -- Condition after one student leaves
  x + y = 7  -- Total number of students
  := by sorry

end class_size_problem_l229_22972


namespace bells_lcm_l229_22932

def church_interval : ℕ := 18
def school_interval : ℕ := 24
def city_hall_interval : ℕ := 30

theorem bells_lcm :
  Nat.lcm (Nat.lcm church_interval school_interval) city_hall_interval = 360 := by
  sorry

end bells_lcm_l229_22932


namespace grid_equal_sums_l229_22955

/-- Given a, b, c, prove that there exist x, y, z, t, u, v such that all rows, columns, and diagonals in a 3x3 grid sum to the same value -/
theorem grid_equal_sums (a b c : ℚ) : ∃ (x y z t u v : ℚ),
  (x + a + b = x + y + c) ∧
  (y + z + t = b + z + c) ∧
  (u + t + v = a + t + c) ∧
  (x + y + c = y + z + t) ∧
  (x + a + b = a + z + v) ∧
  (x + y + c = u + t + v) ∧
  (x + a + b = b + z + c) :=
by sorry

end grid_equal_sums_l229_22955


namespace complex_determinant_solution_l229_22986

/-- Definition of the determinant operation -/
def det (a b c d : ℂ) : ℂ := a * d - b * c

/-- Theorem stating that z = 2 - i satisfies the given condition -/
theorem complex_determinant_solution :
  ∃ z : ℂ, det z (1 + 2*I) (1 - I) (1 + I) = 0 ∧ z = 2 - I :=
by sorry

end complex_determinant_solution_l229_22986


namespace game_a_higher_prob_l229_22918

def prob_heads : ℚ := 3/4
def prob_tails : ℚ := 1/4

def game_a_win_prob : ℚ := prob_heads^4 + prob_tails^4

def game_b_win_prob : ℚ := prob_heads^3 * prob_tails^2 + prob_tails^3 * prob_heads^2

theorem game_a_higher_prob : game_a_win_prob = game_b_win_prob + 1/4 := by
  sorry

end game_a_higher_prob_l229_22918


namespace evaluate_expression_l229_22950

theorem evaluate_expression : (3^3)^2 + 1 = 730 := by
  sorry

end evaluate_expression_l229_22950


namespace subtraction_of_negatives_l229_22921

theorem subtraction_of_negatives : -1 - 2 = -3 := by
  sorry

end subtraction_of_negatives_l229_22921


namespace function_periodicity_l229_22900

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_periodicity (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x| ≤ 1)
  (h2 : ∀ x, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  is_periodic f 1 := by
sorry

end function_periodicity_l229_22900


namespace ellipse_min_major_axis_l229_22964

/-- Given an ellipse where the maximum area of a triangle formed by a point
    on the ellipse and its two foci is 1, the minimum value of its major axis is 2√2. -/
theorem ellipse_min_major_axis (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_ellipse : a^2 = b^2 + c^2) (h_area : b * c = 1) :
  2 * a ≥ 2 * Real.sqrt 2 := by
sorry

end ellipse_min_major_axis_l229_22964


namespace simplify_expression_solve_fractional_equation_l229_22939

-- Problem 1
theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  (12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1) = 2 * x - 4 * x^2 := by
  sorry

-- Problem 2
theorem solve_fractional_equation :
  ∃ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 ∧ 5 / (x^2 + x) - 1 / (x^2 - x) = 0 ∧ x = 3/2 := by
  sorry

end simplify_expression_solve_fractional_equation_l229_22939


namespace size_relationship_l229_22979

theorem size_relationship (a₁ a₂ b₁ b₂ : ℝ) (h1 : a₁ < a₂) (h2 : b₁ < b₂) :
  a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end size_relationship_l229_22979


namespace water_tank_capacity_l229_22912

theorem water_tank_capacity (x : ℚ) : 
  (1 / 3 : ℚ) * x + 16 = x → x = 24 := by
  sorry

end water_tank_capacity_l229_22912


namespace common_area_rectangle_circle_l229_22980

/-- The area of the region common to a rectangle and a circle with the same center -/
theorem common_area_rectangle_circle (rectangle_width : ℝ) (rectangle_height : ℝ) (circle_radius : ℝ) : 
  rectangle_width = 10 →
  rectangle_height = 4 →
  circle_radius = 5 →
  (rectangle_width / 2 = circle_radius) →
  (rectangle_height / 2 < circle_radius) →
  let common_area := rectangle_width * rectangle_height + 2 * π * (rectangle_height / 2)^2
  common_area = 40 + 4 * π := by
  sorry


end common_area_rectangle_circle_l229_22980


namespace first_1000_decimals_are_zero_l229_22914

theorem first_1000_decimals_are_zero (a : ℕ) (n : ℕ) 
    (ha : a = 35 ∨ a = 37) (hn : n = 1999 ∨ n = 2000) :
  ∃ (k : ℕ), (6 + Real.sqrt a)^n = k + (1 / 10^1000) * (Real.sqrt a) := by
  sorry

end first_1000_decimals_are_zero_l229_22914


namespace cubic_sum_reciprocal_l229_22988

theorem cubic_sum_reciprocal (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end cubic_sum_reciprocal_l229_22988


namespace coat_price_proof_l229_22995

theorem coat_price_proof (price : ℝ) : 
  (price - 250 = price * 0.5) → price = 500 := by
  sorry

end coat_price_proof_l229_22995


namespace conditional_probability_l229_22960

theorem conditional_probability (P_AB P_A : ℝ) (h1 : P_AB = 2/15) (h2 : P_A = 2/5) :
  P_AB / P_A = 1/3 := by
  sorry

end conditional_probability_l229_22960


namespace negative_a_fifth_times_a_l229_22977

theorem negative_a_fifth_times_a (a : ℝ) : (-a)^5 * a = -a^6 := by
  sorry

end negative_a_fifth_times_a_l229_22977


namespace F_2_f_3_equals_341_l229_22953

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 2
def F (a b : ℝ) : ℝ := b^3 - a

-- State the theorem
theorem F_2_f_3_equals_341 : F 2 (f 3) = 341 := by sorry

end F_2_f_3_equals_341_l229_22953


namespace not_necessarily_divisible_by_twenty_l229_22974

theorem not_necessarily_divisible_by_twenty (k : ℤ) (n : ℤ) : 
  n = k * (k + 1) * (k + 2) → (∃ m : ℤ, n = 5 * m) → 
  ¬(∀ (k : ℤ), ∃ (m : ℤ), n = 20 * m) := by
  sorry

end not_necessarily_divisible_by_twenty_l229_22974


namespace total_silver_dollars_l229_22917

/-- The number of silver dollars owned by Mr. Chiu -/
def chiu_dollars : ℕ := 56

/-- The number of silver dollars owned by Mr. Phung -/
def phung_dollars : ℕ := chiu_dollars + 16

/-- The number of silver dollars owned by Mr. Ha -/
def ha_dollars : ℕ := phung_dollars + 5

/-- The total number of silver dollars owned by all three -/
def total_dollars : ℕ := chiu_dollars + phung_dollars + ha_dollars

theorem total_silver_dollars :
  total_dollars = 205 := by sorry

end total_silver_dollars_l229_22917


namespace dividing_line_equation_l229_22947

/-- Represents a circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the region S formed by the union of nine unit circles -/
def region_S : Set (ℝ × ℝ) :=
  sorry

/-- The line with slope 4 that divides region S into two equal areas -/
def dividing_line : ℝ → ℝ :=
  sorry

/-- Theorem stating that the dividing line has the equation 4x - y = 3 -/
theorem dividing_line_equation :
  ∀ x y, dividing_line y = x ↔ 4 * x - y = 3 :=
sorry

end dividing_line_equation_l229_22947


namespace chris_money_before_birthday_l229_22994

/-- The amount of money Chris had before his birthday -/
def money_before_birthday : ℕ := sorry

/-- The amount Chris received from his grandmother -/
def grandmother_gift : ℕ := 25

/-- The amount Chris received from his aunt and uncle -/
def aunt_uncle_gift : ℕ := 20

/-- The amount Chris received from his parents -/
def parents_gift : ℕ := 75

/-- The total amount Chris has now -/
def total_money_now : ℕ := 279

/-- Theorem stating that Chris had $159 before his birthday -/
theorem chris_money_before_birthday :
  money_before_birthday = 159 :=
by sorry

end chris_money_before_birthday_l229_22994


namespace solve_system_for_p_l229_22942

theorem solve_system_for_p (p q : ℚ) 
  (eq1 : 2 * p + 5 * q = 10)
  (eq2 : 5 * p + 2 * q = 20) : 
  p = 80 / 21 := by sorry

end solve_system_for_p_l229_22942


namespace euler_identity_complex_power_exp_sum_bound_l229_22997

-- Define the complex exponential function
noncomputable def cexp (x : ℝ) : ℂ := Complex.exp (x * Complex.I)

-- Euler's formula
axiom euler_formula (x : ℝ) : cexp x = Complex.cos x + Complex.I * Complex.sin x

-- Theorems to prove
theorem euler_identity : cexp π + 1 = 0 := by sorry

theorem complex_power : (1/2 + Complex.I * (Real.sqrt 3)/2) ^ 2022 = 1 := by sorry

theorem exp_sum_bound (x : ℝ) : Complex.abs (cexp x + cexp (-x)) ≤ 2 := by sorry

end euler_identity_complex_power_exp_sum_bound_l229_22997


namespace package_volume_calculation_l229_22998

/-- Calculates the total volume needed to package a collection given box dimensions and cost constraints. -/
theorem package_volume_calculation 
  (box_length : ℝ) 
  (box_width : ℝ) 
  (box_height : ℝ) 
  (box_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : box_length = 20) 
  (h2 : box_width = 20) 
  (h3 : box_height = 15) 
  (h4 : box_cost = 0.9) 
  (h5 : total_cost = 459) :
  (total_cost / box_cost) * (box_length * box_width * box_height) = 3060000 := by
  sorry

end package_volume_calculation_l229_22998


namespace function_inequality_l229_22925

open Real

theorem function_inequality (a x : ℝ) (ha : a ≥ 1) (hx : x > 0) :
  a * exp x + 2 * x - 1 ≥ (x + a * exp 1) * x := by
  sorry

end function_inequality_l229_22925


namespace max_x_value_l229_22926

theorem max_x_value (x : ℝ) : 
  ((5*x - 20)/(4*x - 5))^2 + (5*x - 20)/(4*x - 5) = 20 → x ≤ 9/5 :=
by sorry

end max_x_value_l229_22926


namespace arccos_one_half_l229_22902

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end arccos_one_half_l229_22902


namespace function_composition_ratio_l229_22919

def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 2 * x - 3

theorem function_composition_ratio :
  (f (g (f 3))) / (g (f (g 3))) = 59 / 19 := by
  sorry

end function_composition_ratio_l229_22919


namespace smallest_positive_a_l229_22984

theorem smallest_positive_a (a : ℝ) : 
  a > 0 ∧ 
  (⌊2016 * a⌋ : ℤ) - (⌈a⌉ : ℤ) + 1 = 2016 ∧ 
  ∀ b : ℝ, b > 0 → (⌊2016 * b⌋ : ℤ) - (⌈b⌉ : ℤ) + 1 = 2016 → a ≤ b → 
  a = 2017 / 2016 := by
sorry

end smallest_positive_a_l229_22984


namespace f_properties_l229_22940

/-- The function f(m, n) represents the absolute difference between 
    the areas of black and white parts in a right triangle with legs m and n. -/
def f (m n : ℕ+) : ℝ :=
  sorry

theorem f_properties :
  (∀ m n : ℕ+, Even m.val → Even n.val → f m n = 0) ∧
  (∀ m n : ℕ+, Odd m.val → Odd n.val → f m n = 1/2) ∧
  (∀ m n : ℕ+, f m n ≤ (1/2 : ℝ) * max m.val n.val) ∧
  (∀ c : ℝ, ∃ m n : ℕ+, f m n ≥ c) :=
by sorry

end f_properties_l229_22940


namespace sqrt_sum_floor_equality_l229_22949

theorem sqrt_sum_floor_equality (n : ℤ) : 
  ⌊Real.sqrt (n : ℝ) + Real.sqrt ((n + 1) : ℝ)⌋ = ⌊Real.sqrt ((4 * n + 2) : ℝ)⌋ :=
sorry

end sqrt_sum_floor_equality_l229_22949


namespace lottery_ambo_probability_l229_22970

theorem lottery_ambo_probability (n : ℕ) : 
  (n ≥ 5) →
  (Nat.choose 5 2 : ℚ) / (Nat.choose n 2 : ℚ) = 5 / 473 →
  n = 44 :=
by sorry

end lottery_ambo_probability_l229_22970


namespace odd_function_property_l229_22967

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f 3 - f 2 = 1) :
  f (-2) - f (-3) = 1 := by
  sorry

end odd_function_property_l229_22967


namespace classroom_gpa_l229_22944

/-- Given a classroom where one-third of the students have a GPA of 54 and the remaining two-thirds have a GPA of 45, the GPA of the whole class is 48. -/
theorem classroom_gpa : 
  ∀ (n : ℕ) (total_gpa : ℝ),
  n > 0 →
  total_gpa = (n / 3 : ℝ) * 54 + (2 * n / 3 : ℝ) * 45 →
  total_gpa / n = 48 :=
by
  sorry

end classroom_gpa_l229_22944


namespace remainder_of_binary_division_l229_22905

def binary_number : ℕ := 101110100101

theorem remainder_of_binary_division (n : ℕ) (h : n = binary_number) :
  n % 8 = 5 := by
  sorry

end remainder_of_binary_division_l229_22905


namespace correct_calculation_l229_22999

theorem correct_calculation (x : ℝ) (h : (x * 5) + 7 = 27) : (x + 5) * 7 = 63 := by
  sorry

end correct_calculation_l229_22999


namespace cost_per_sandwich_is_correct_l229_22987

-- Define the problem parameters
def sandwiches_per_loaf : ℕ := 10
def total_sandwiches : ℕ := 50
def bread_cost : ℚ := 4
def meat_cost : ℚ := 5
def cheese_cost : ℚ := 4
def meat_packs_per_loaf : ℕ := 2
def cheese_packs_per_loaf : ℕ := 2
def cheese_coupon : ℚ := 1
def meat_coupon : ℚ := 1
def discount_threshold : ℚ := 60
def discount_rate : ℚ := 0.1

-- Define the function to calculate the cost per sandwich
def cost_per_sandwich : ℚ :=
  let loaves := total_sandwiches / sandwiches_per_loaf
  let meat_packs := loaves * meat_packs_per_loaf
  let cheese_packs := loaves * cheese_packs_per_loaf
  let total_cost := loaves * bread_cost + meat_packs * meat_cost + cheese_packs * cheese_cost
  let discounted_cost := total_cost - cheese_coupon - meat_coupon
  let final_cost := if discounted_cost > discount_threshold
                    then discounted_cost * (1 - discount_rate)
                    else discounted_cost
  final_cost / total_sandwiches

-- Theorem to prove
theorem cost_per_sandwich_is_correct :
  cost_per_sandwich = 1.944 := by sorry

end cost_per_sandwich_is_correct_l229_22987


namespace cone_height_l229_22913

/-- Given a cone with slant height 10 and base radius 5, its height is 5√3 -/
theorem cone_height (l r h : ℝ) (hl : l = 10) (hr : r = 5) 
  (h_def : h = Real.sqrt (l^2 - r^2)) : h = 5 * Real.sqrt 3 := by
  sorry


end cone_height_l229_22913


namespace order_of_numbers_l229_22992

theorem order_of_numbers (m n : ℝ) (hm : m < 0) (hn : n > 0) (hmn : m + n < 0) :
  -m > n ∧ n > -n ∧ -n > m := by sorry

end order_of_numbers_l229_22992


namespace prime_power_cube_plus_one_l229_22965

def is_solution (x y z : ℕ+) : Prop :=
  z.val.Prime ∧ z^(x.val) = y^3 + 1

theorem prime_power_cube_plus_one :
  ∀ x y z : ℕ+, is_solution x y z ↔ (x, y, z) = (1, 1, 2) ∨ (x, y, z) = (2, 2, 3) :=
sorry

end prime_power_cube_plus_one_l229_22965


namespace exists_compound_interest_l229_22976

/-- Represents the compound interest scenario -/
def compound_interest (P : ℝ) : Prop :=
  let r : ℝ := 0.06  -- annual interest rate
  let n : ℝ := 12    -- number of compounding periods per year
  let t : ℝ := 0.25  -- time in years (3 months)
  let A : ℝ := 1014.08  -- final amount after 3 months
  let two_month_amount : ℝ := P * (1 + r / n) ^ (2 * n * (t / 3))
  A = P * (1 + r / n) ^ (n * t) ∧ 
  (A - two_month_amount) * 100 = 13

/-- Theorem stating the existence of an initial investment satisfying the compound interest scenario -/
theorem exists_compound_interest : ∃ P : ℝ, compound_interest P :=
  sorry

end exists_compound_interest_l229_22976


namespace base4_division_theorem_l229_22909

/-- Convert a number from base 4 to base 10 -/
def base4To10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (4 ^ i)) 0

/-- Convert a number from base 10 to base 4 -/
def base10To4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Division in base 4 -/
def divBase4 (a b : List Nat) : List Nat :=
  base10To4 (base4To10 a / base4To10 b)

theorem base4_division_theorem :
  divBase4 [3, 1, 2, 2] [1, 2] = [2, 0, 1] := by sorry

end base4_division_theorem_l229_22909


namespace expression_equality_l229_22946

theorem expression_equality (x : ℝ) (Q : ℝ) (h : 2 * (5 * x + 3 * Real.sqrt 2) = Q) :
  4 * (10 * x + 6 * Real.sqrt 2) = 4 * Q := by
  sorry

end expression_equality_l229_22946


namespace residue_calculation_l229_22957

theorem residue_calculation (m : ℕ) (h : m = 17) : 
  (220 * 18 - 28 * 5 + 4) % m = 12 := by
  sorry

end residue_calculation_l229_22957


namespace ellipse_hyperbola_m_range_l229_22948

/-- An ellipse with equation x²/5 + y²/m = 1 -/
def isEllipse (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2/5 + y^2/m = 1 ∧ m ≠ 0 ∧ m ≠ 5

/-- A hyperbola with equation x²/5 + y²/(m-6) = 1 -/
def isHyperbola (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2/5 + y^2/(m-6) = 1 ∧ m ≠ 6

/-- The range of valid m values -/
def validRange (m : ℝ) : Prop :=
  (0 < m ∧ m < 5) ∨ (5 < m ∧ m < 6)

theorem ellipse_hyperbola_m_range :
  ∀ m : ℝ, (isEllipse m ∧ isHyperbola m) ↔ validRange m :=
by sorry

end ellipse_hyperbola_m_range_l229_22948


namespace tangent_line_y_intercept_l229_22993

/-- The function representing the curve y = x^3 + 2x - 1 -/
def f (x : ℝ) : ℝ := x^3 + 2*x - 1

/-- The derivative of the function f -/
def f' (x : ℝ) : ℝ := 3*x^2 + 2

/-- The point P on the curve -/
def P : ℝ × ℝ := (1, 2)

/-- The slope of the tangent line at point P -/
def k : ℝ := f' P.1

/-- The y-intercept of the tangent line -/
def b : ℝ := P.2 - k * P.1

theorem tangent_line_y_intercept :
  b = -3 :=
sorry

end tangent_line_y_intercept_l229_22993


namespace library_purchase_theorem_l229_22975

-- Define the types of books
inductive BookType
| SocialScience
| Children

-- Define the price function
def price : BookType → ℕ
| BookType.SocialScience => 40
| BookType.Children => 20

-- Define the total cost function
def totalCost (ss_count : ℕ) (c_count : ℕ) : ℕ :=
  ss_count * price BookType.SocialScience + c_count * price BookType.Children

-- Define the valid purchase plan predicate
def isValidPurchasePlan (ss_count : ℕ) (c_count : ℕ) : Prop :=
  ss_count + c_count ≥ 70 ∧
  c_count = ss_count + 20 ∧
  totalCost ss_count c_count ≤ 2000

-- State the theorem
theorem library_purchase_theorem :
  (totalCost 20 40 = 1600) ∧
  (20 * price BookType.SocialScience = 30 * price BookType.Children + 200) ∧
  (∀ ss_count c_count : ℕ, isValidPurchasePlan ss_count c_count ↔ 
    (ss_count = 25 ∧ c_count = 45) ∨ (ss_count = 26 ∧ c_count = 46)) :=
sorry

end library_purchase_theorem_l229_22975


namespace tourist_group_size_proof_l229_22930

/-- Represents the number of people a large room can accommodate -/
def large_room_capacity : ℕ := 3

/-- Represents the number of large rooms rented -/
def large_rooms_rented : ℕ := 8

/-- Represents the total number of people in the tourist group -/
def tourist_group_size : ℕ := large_rooms_rented * large_room_capacity

theorem tourist_group_size_proof :
  (∀ n : ℕ, n ≠ tourist_group_size → 
    (∃ m k : ℕ, n = 3 * m + 2 * k ∧ m + k < large_rooms_rented) ∨
    (∃ m k : ℕ, n = 3 * m + 2 * k ∧ m > large_rooms_rented)) →
  tourist_group_size = 24 := by sorry

end tourist_group_size_proof_l229_22930


namespace min_fraction_sum_l229_22937

def Digits := Finset.range 8

theorem min_fraction_sum (A B C D : ℕ) 
  (hA : A ∈ Digits) (hB : B ∈ Digits) (hC : C ∈ Digits) (hD : D ∈ Digits)
  (hAB : A ≠ B) (hAC : A ≠ C) (hAD : A ≠ D) (hBC : B ≠ C) (hBD : B ≠ D) (hCD : C ≠ D)
  (hB_pos : B > 0) (hD_pos : D > 0) :
  (A : ℚ) / B + (C : ℚ) / D ≥ 11 / 28 :=
sorry

end min_fraction_sum_l229_22937


namespace ali_baba_strategy_l229_22981

/-- A game with diamonds where players split piles. -/
structure DiamondGame where
  total_diamonds : ℕ
  
/-- The number of moves required to end the game. -/
def moves_to_end (game : DiamondGame) : ℕ :=
  game.total_diamonds - 1

/-- Determines if the second player wins the game. -/
def second_player_wins (game : DiamondGame) : Prop :=
  Even (moves_to_end game)

/-- Theorem: In a game with 2017 diamonds, the second player wins. -/
theorem ali_baba_strategy (game : DiamondGame) (h : game.total_diamonds = 2017) :
  second_player_wins game := by
  sorry

#eval moves_to_end { total_diamonds := 2017 }

end ali_baba_strategy_l229_22981


namespace base_10_to_base_7_l229_22969

theorem base_10_to_base_7 : ∃ (a b c d : ℕ), 
  803 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ 
  a < 7 ∧ b < 7 ∧ c < 7 ∧ d < 7 ∧
  a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 5 :=
by sorry

end base_10_to_base_7_l229_22969


namespace count_valid_pairs_l229_22927

def is_valid_pair (a b : ℂ) : Prop :=
  a^4 * b^7 = 1 ∧ a^8 * b^3 = 1

theorem count_valid_pairs :
  ∃! (n : ℕ), ∃ (S : Finset (ℂ × ℂ)),
    Finset.card S = n ∧
    (∀ (p : ℂ × ℂ), p ∈ S ↔ is_valid_pair p.1 p.2) ∧
    n = 16 :=
by sorry

end count_valid_pairs_l229_22927


namespace equation_solution_l229_22951

theorem equation_solution : 
  ∃! x : ℚ, (x + 1) / 3 - 1 = (5 * x - 1) / 6 :=
by
  use -1
  constructor
  · -- Prove that -1 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end equation_solution_l229_22951


namespace inequality_system_solution_l229_22931

theorem inequality_system_solution :
  {x : ℝ | x + 3 ≥ 2 ∧ (3 * x - 1) / 2 < 4} = {x : ℝ | -1 ≤ x ∧ x < 3} := by
  sorry

end inequality_system_solution_l229_22931


namespace symmetry_wrt_x_axis_l229_22920

/-- Given a point P with coordinates (3,2), prove that its symmetrical point
    with respect to the x-axis has coordinates (3,-2) -/
theorem symmetry_wrt_x_axis :
  let P : ℝ × ℝ := (3, 2)
  let symmetry_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetry_x P = (3, -2) := by sorry

end symmetry_wrt_x_axis_l229_22920


namespace any_proof_to_contradiction_l229_22908

theorem any_proof_to_contradiction (P : Prop) : P → ∃ (proof : ¬P → False), P :=
  sorry

end any_proof_to_contradiction_l229_22908
