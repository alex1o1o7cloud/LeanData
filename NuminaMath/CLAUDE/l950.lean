import Mathlib

namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l950_95041

/-- A line with equation y = 3x - 2 does not pass through the second quadrant. -/
theorem line_not_in_second_quadrant :
  ∀ x y : ℝ, y = 3 * x - 2 → ¬(x > 0 ∧ y > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l950_95041


namespace NUMINAMATH_CALUDE_sqrt_four_cubed_sum_l950_95022

theorem sqrt_four_cubed_sum : Real.sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_cubed_sum_l950_95022


namespace NUMINAMATH_CALUDE_evaluate_expression_l950_95089

theorem evaluate_expression (x y z : ℚ) : 
  x = 1/4 → y = 1/3 → z = 12 → x^3 * y^4 * z = 1/432 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l950_95089


namespace NUMINAMATH_CALUDE_yogurt_refund_l950_95056

theorem yogurt_refund (total_packs : ℕ) (expired_percentage : ℚ) (cost_per_pack : ℕ) : 
  total_packs = 80 → 
  expired_percentage = 40 / 100 → 
  cost_per_pack = 12 → 
  (total_packs : ℚ) * expired_percentage * cost_per_pack = 384 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_refund_l950_95056


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l950_95050

theorem complex_fraction_equality : (2 * Complex.I) / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l950_95050


namespace NUMINAMATH_CALUDE_circle_constant_value_l950_95004

-- Define the circle equation
def circle_equation (x y c : ℝ) : Prop :=
  x^2 + 4*x + y^2 + 8*y + c = 0

-- Define the center of the circle
def circle_center (x y : ℝ) : Prop :=
  x = -2 ∧ y = -4

-- Define the radius of the circle
def circle_radius (r : ℝ) : Prop :=
  r = 5

-- Theorem statement
theorem circle_constant_value :
  ∀ (c : ℝ), 
  (∀ (x y : ℝ), circle_equation x y c → 
    ∃ (h k : ℝ), circle_center h k ∧ 
    ∃ (r : ℝ), circle_radius r ∧ 
    (x - h)^2 + (y - k)^2 = r^2) →
  c = -5 := by sorry

end NUMINAMATH_CALUDE_circle_constant_value_l950_95004


namespace NUMINAMATH_CALUDE_valid_lineups_count_l950_95049

-- Define the total number of players
def total_players : ℕ := 15

-- Define the number of players in a starting lineup
def lineup_size : ℕ := 6

-- Define the number of players who can't play together
def restricted_players : ℕ := 3

-- Define the function to calculate the number of valid lineups
def valid_lineups : ℕ := sorry

-- Theorem statement
theorem valid_lineups_count :
  valid_lineups = 3300 :=
sorry

end NUMINAMATH_CALUDE_valid_lineups_count_l950_95049


namespace NUMINAMATH_CALUDE_log_inequality_l950_95052

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + x^2) < x^2 / (1 + x^2) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l950_95052


namespace NUMINAMATH_CALUDE_linear_function_through_points_l950_95060

/-- A linear function passing through point A(1, -1) -/
def f (a : ℝ) (x : ℝ) : ℝ := -x + a

theorem linear_function_through_points :
  ∃ (a : ℝ), f a 1 = -1 ∧ f a (-2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_through_points_l950_95060


namespace NUMINAMATH_CALUDE_five_points_two_small_triangles_l950_95039

-- Define a triangular region with unit area
def UnitTriangle : Set (ℝ × ℝ) := sorry

-- Define a function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem five_points_two_small_triangles 
  (points : Finset (ℝ × ℝ)) 
  (h1 : points.card = 5) 
  (h2 : ∀ p ∈ points, p ∈ UnitTriangle) : 
  ∃ (t1 t2 : Finset (ℝ × ℝ)), 
    t1 ⊆ points ∧ t2 ⊆ points ∧ 
    t1.card = 3 ∧ t2.card = 3 ∧ 
    t1 ≠ t2 ∧
    (∃ (p1 p2 p3 : ℝ × ℝ), p1 ∈ t1 ∧ p2 ∈ t1 ∧ p3 ∈ t1 ∧ triangleArea p1 p2 p3 ≤ 1/4) ∧
    (∃ (q1 q2 q3 : ℝ × ℝ), q1 ∈ t2 ∧ q2 ∈ t2 ∧ q3 ∈ t2 ∧ triangleArea q1 q2 q3 ≤ 1/4) :=
by sorry

end NUMINAMATH_CALUDE_five_points_two_small_triangles_l950_95039


namespace NUMINAMATH_CALUDE_tallest_giraffe_height_is_96_l950_95092

/-- The height of the shortest giraffe in inches -/
def shortest_giraffe_height : ℕ := 68

/-- The height difference between the tallest and shortest giraffes in inches -/
def height_difference : ℕ := 28

/-- The number of adult giraffes at the zoo -/
def num_giraffes : ℕ := 14

/-- The height of the tallest giraffe in inches -/
def tallest_giraffe_height : ℕ := shortest_giraffe_height + height_difference

theorem tallest_giraffe_height_is_96 : tallest_giraffe_height = 96 := by
  sorry

end NUMINAMATH_CALUDE_tallest_giraffe_height_is_96_l950_95092


namespace NUMINAMATH_CALUDE_expression_evaluation_l950_95064

theorem expression_evaluation :
  let x : ℝ := 3 * Real.sqrt 3 + 2 * Real.sqrt 2
  let y : ℝ := 3 * Real.sqrt 3 - 2 * Real.sqrt 2
  ((x * (x + y) + 2 * y * (x + y)) / (x * y * (x + 2 * y))) / ((x * y) / (x + 2 * y)) = 108 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l950_95064


namespace NUMINAMATH_CALUDE_class_average_problem_l950_95068

theorem class_average_problem (total_students : ℕ) 
  (high_score_students : ℕ) (zero_score_students : ℕ) 
  (high_score : ℕ) (class_average : ℕ) :
  total_students = 25 →
  high_score_students = 3 →
  zero_score_students = 5 →
  high_score = 95 →
  class_average = 42 →
  let remaining_students := total_students - (high_score_students + zero_score_students)
  let total_score := total_students * class_average
  let high_score_total := high_score_students * high_score
  let remaining_score := total_score - high_score_total
  remaining_score / remaining_students = 45 := by
sorry

end NUMINAMATH_CALUDE_class_average_problem_l950_95068


namespace NUMINAMATH_CALUDE_ratio_of_squares_to_difference_l950_95023

theorem ratio_of_squares_to_difference (a b : ℝ) : 
  0 < b → 0 < a → a > b → (a^2 + b^2 = 7 * (a - b)) → (a / b = Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_squares_to_difference_l950_95023


namespace NUMINAMATH_CALUDE_factorization_equality_l950_95018

theorem factorization_equality (x y : ℝ) : 2 * x^2 * y - 8 * y = 2 * y * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l950_95018


namespace NUMINAMATH_CALUDE_work_rate_increase_l950_95003

theorem work_rate_increase (total_time hours_worked : ℝ)
  (original_items additional_items : ℕ) :
  total_time = 10 ∧ 
  hours_worked = 6 ∧ 
  original_items = 1250 ∧ 
  additional_items = 150 →
  let original_rate := original_items / total_time
  let items_processed := original_rate * hours_worked
  let remaining_items := original_items - items_processed + additional_items
  let remaining_time := total_time - hours_worked
  let new_rate := remaining_items / remaining_time
  (new_rate - original_rate) / original_rate * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_work_rate_increase_l950_95003


namespace NUMINAMATH_CALUDE_gifted_subscribers_l950_95090

/-- Calculates the number of gifted subscribers for a Twitch streamer --/
theorem gifted_subscribers
  (initial_subscribers : ℕ)
  (income_per_subscriber : ℕ)
  (current_monthly_income : ℕ)
  (h1 : initial_subscribers = 150)
  (h2 : income_per_subscriber = 9)
  (h3 : current_monthly_income = 1800) :
  current_monthly_income / income_per_subscriber - initial_subscribers = 50 :=
by sorry

end NUMINAMATH_CALUDE_gifted_subscribers_l950_95090


namespace NUMINAMATH_CALUDE_cheaper_fluid_cost_l950_95001

/-- Represents the cost of cleaning fluids and drum quantities -/
structure CleaningSupplies where
  total_drums : ℕ
  expensive_drums : ℕ
  cheap_drums : ℕ
  expensive_cost : ℚ
  total_cost : ℚ

/-- Theorem stating that given the conditions, the cheaper fluid costs $20 per drum -/
theorem cheaper_fluid_cost (supplies : CleaningSupplies)
  (h1 : supplies.total_drums = 7)
  (h2 : supplies.expensive_drums + supplies.cheap_drums = supplies.total_drums)
  (h3 : supplies.expensive_cost = 30)
  (h4 : supplies.total_cost = 160)
  (h5 : supplies.cheap_drums = 5) :
  (supplies.total_cost - supplies.expensive_cost * supplies.expensive_drums) / supplies.cheap_drums = 20 :=
by sorry

end NUMINAMATH_CALUDE_cheaper_fluid_cost_l950_95001


namespace NUMINAMATH_CALUDE_parabola_c_value_l950_95079

/-- Given a parabola y = ax^2 + bx + c with vertex (3, -5) passing through (1, -3),
    prove that c = -0.5 -/
theorem parabola_c_value (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →   -- Parabola equation
  (3, -5) = (3, a * 3^2 + b * 3 + c) →     -- Vertex condition
  -3 = a * 1^2 + b * 1 + c →               -- Point condition
  c = -0.5 := by
sorry


end NUMINAMATH_CALUDE_parabola_c_value_l950_95079


namespace NUMINAMATH_CALUDE_norm_scalar_multiple_l950_95073

variable (n : ℕ)
variable (v : Fin n → ℝ)

theorem norm_scalar_multiple
  (h : ‖v‖ = 6) :
  ‖(5 : ℝ) • v‖ = 30 := by
sorry

end NUMINAMATH_CALUDE_norm_scalar_multiple_l950_95073


namespace NUMINAMATH_CALUDE_fly_probabilities_l950_95072

def fly_move (x y : ℕ) : Prop := x ≤ 8 ∧ y ≤ 10

def prob_reach (x y : ℕ) : ℚ := (Nat.choose (x + y) x : ℚ) / 2^(x + y)

def prob_through (x1 y1 x2 y2 x3 y3 : ℕ) : ℚ :=
  (Nat.choose (x1 + y1) x1 * Nat.choose (x3 - x2 + y3 - y2) (x3 - x2) : ℚ) / 2^(x3 + y3)

def inside_circle (x y cx cy r : ℝ) : Prop :=
  (x - cx)^2 + (y - cy)^2 ≤ r^2

theorem fly_probabilities :
  let p1 := prob_reach 8 10
  let p2 := prob_through 5 6 6 6 8 10
  let p3 := (2 * Nat.choose 9 2 * Nat.choose 9 6 + 2 * Nat.choose 9 3 * Nat.choose 9 5 + Nat.choose 9 4 ^ 2 : ℚ) / 2^18
  (p1 = (Nat.choose 18 8 : ℚ) / 2^18) ∧
  (p2 = (Nat.choose 11 5 * Nat.choose 6 2 : ℚ) / 2^18) ∧
  (∀ x y, fly_move x y → inside_circle x y 4 5 3 → prob_reach x y ≤ p3) := by
  sorry

end NUMINAMATH_CALUDE_fly_probabilities_l950_95072


namespace NUMINAMATH_CALUDE_parallel_planes_line_sufficient_not_necessary_l950_95034

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relations
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (line_in_plane : Line → Plane → Prop)

theorem parallel_planes_line_sufficient_not_necessary 
  (α β : Plane) (l : Line) 
  (h_distinct : α ≠ β) 
  (h_l_in_α : line_in_plane l α) :
  (∀ α β l, plane_parallel α β → line_parallel_plane l β) ∧ 
  (∃ α β l, line_parallel_plane l β ∧ ¬plane_parallel α β) := by
  sorry


end NUMINAMATH_CALUDE_parallel_planes_line_sufficient_not_necessary_l950_95034


namespace NUMINAMATH_CALUDE_fourth_person_height_l950_95043

theorem fourth_person_height :
  ∀ (h₁ h₂ h₃ h₄ : ℝ),
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →
  h₂ - h₁ = 2 →
  h₃ - h₂ = 2 →
  h₄ - h₃ = 6 →
  (h₁ + h₂ + h₃ + h₄) / 4 = 79 →
  h₄ = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l950_95043


namespace NUMINAMATH_CALUDE_cube_equals_nine_times_implies_fifth_power_l950_95062

theorem cube_equals_nine_times_implies_fifth_power (w : ℕ+) 
  (h : w.val ^ 3 = 9 * w.val) : w.val ^ 5 = 243 := by
  sorry

end NUMINAMATH_CALUDE_cube_equals_nine_times_implies_fifth_power_l950_95062


namespace NUMINAMATH_CALUDE_aaron_initial_cards_l950_95016

theorem aaron_initial_cards (found : ℕ) (final : ℕ) (h1 : found = 62) (h2 : final = 67) :
  final - found = 5 := by
  sorry

end NUMINAMATH_CALUDE_aaron_initial_cards_l950_95016


namespace NUMINAMATH_CALUDE_box_matching_problem_l950_95030

/-- Represents the problem of matching box bodies and bottoms --/
theorem box_matching_problem (total_tinplates : ℕ) 
  (bodies_per_tinplate : ℕ) (bottoms_per_tinplate : ℕ) 
  (bottoms_per_body : ℕ) (bodies_tinplates : ℕ) (bottoms_tinplates : ℕ) :
  total_tinplates = 36 →
  bodies_per_tinplate = 25 →
  bottoms_per_tinplate = 40 →
  bottoms_per_body = 2 →
  bodies_tinplates = 16 →
  bottoms_tinplates = 20 →
  bodies_tinplates + bottoms_tinplates = total_tinplates ∧
  bodies_per_tinplate * bodies_tinplates * bottoms_per_body = 
    bottoms_per_tinplate * bottoms_tinplates :=
by sorry

end NUMINAMATH_CALUDE_box_matching_problem_l950_95030


namespace NUMINAMATH_CALUDE_A_sufficient_for_B_l950_95055

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}

theorem A_sufficient_for_B : ∀ x : ℝ, x ∈ A → x ∈ B := by
  sorry

end NUMINAMATH_CALUDE_A_sufficient_for_B_l950_95055


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l950_95087

-- Define the quadratic function types
def QuadraticFunction (a b c : ℝ) := λ x : ℝ => a * x^2 + b * x + c

-- Define the solution set type
def SolutionSet := Set ℝ

-- State the theorem
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h1 : SolutionSet) 
  (h2 : h1 = {x : ℝ | x < (1/3) ∨ x > (1/2)}) 
  (h3 : h1 = {x : ℝ | QuadraticFunction a b c x < 0}) :
  {x : ℝ | QuadraticFunction c (-b) a x > 0} = Set.Ioo (-3) (-2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l950_95087


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_b_l950_95095

def b (n : ℕ) : ℕ := n.factorial + 3 * n

theorem max_gcd_consecutive_b : (∃ n : ℕ, Nat.gcd (b n) (b (n + 1)) = 14) ∧ 
  (∀ n : ℕ, Nat.gcd (b n) (b (n + 1)) ≤ 14) :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_b_l950_95095


namespace NUMINAMATH_CALUDE_profit_per_meter_is_55_l950_95033

/-- Profit per meter of cloth -/
def profit_per_meter (total_meters : ℕ) (total_profit : ℕ) : ℚ :=
  total_profit / total_meters

/-- Theorem: The profit per meter of cloth is 55 rupees -/
theorem profit_per_meter_is_55
  (total_meters : ℕ)
  (selling_price : ℕ)
  (total_profit : ℕ)
  (h1 : total_meters = 40)
  (h2 : selling_price = 8200)
  (h3 : total_profit = 2200) :
  profit_per_meter total_meters total_profit = 55 := by
  sorry

end NUMINAMATH_CALUDE_profit_per_meter_is_55_l950_95033


namespace NUMINAMATH_CALUDE_f_neg_one_equals_two_l950_95086

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) := f x + 4

-- State the theorem
theorem f_neg_one_equals_two
  (h_odd : ∀ x, f (-x) = -f x)  -- f is an odd function
  (h_g_one : g 1 = 2)           -- g(1) = 2
  : f (-1) = 2 := by
  sorry


end NUMINAMATH_CALUDE_f_neg_one_equals_two_l950_95086


namespace NUMINAMATH_CALUDE_apartment_cost_l950_95080

/-- The cost of a room on the first floor of Krystiana's apartment building. -/
def first_floor_cost : ℝ := 8.75

/-- The number of rooms on each floor. -/
def rooms_per_floor : ℕ := 3

/-- The additional cost for a room on the second floor compared to the first floor. -/
def second_floor_additional_cost : ℝ := 20

theorem apartment_cost (total_earnings : ℝ) 
  (h_total : total_earnings = 165) :
  first_floor_cost * rooms_per_floor + 
  (first_floor_cost + second_floor_additional_cost) * rooms_per_floor + 
  (2 * first_floor_cost) * rooms_per_floor = total_earnings := by
  sorry

end NUMINAMATH_CALUDE_apartment_cost_l950_95080


namespace NUMINAMATH_CALUDE_at_most_one_greater_than_one_l950_95070

theorem at_most_one_greater_than_one (x y : ℝ) (h : x + y < 2) :
  ¬(x > 1 ∧ y > 1) := by
  sorry

end NUMINAMATH_CALUDE_at_most_one_greater_than_one_l950_95070


namespace NUMINAMATH_CALUDE_subset_condition_l950_95099

theorem subset_condition (A B : Set ℕ) (m : ℕ) : 
  A = {0, 1, 2} → 
  B = {1, m} → 
  B ⊆ A → 
  m = 0 ∨ m = 2 := by
sorry

end NUMINAMATH_CALUDE_subset_condition_l950_95099


namespace NUMINAMATH_CALUDE_compute_expression_l950_95042

theorem compute_expression : 20 * (256 / 4 + 64 / 16 + 16 / 64 + 2) = 1405 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l950_95042


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l950_95017

theorem average_of_a_and_b (a b c : ℝ) 
  (h1 : (a + b) / 2 = 50)
  (h2 : (b + c) / 2 = 70)
  (h3 : c - a = 40) :
  (a + b) / 2 = 50 := by
sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l950_95017


namespace NUMINAMATH_CALUDE_expression_equals_two_l950_95051

theorem expression_equals_two (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 2) :
  (2 * x^2 - x) / ((x + 1) * (x - 2)) - (4 + x) / ((x + 1) * (x - 2)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_two_l950_95051


namespace NUMINAMATH_CALUDE_root_sum_fraction_l950_95084

theorem root_sum_fraction (p q r : ℝ) : 
  p^3 - 6*p^2 + 11*p - 6 = 0 →
  q^3 - 6*q^2 + 11*q - 6 = 0 →
  r^3 - 6*r^2 + 11*r - 6 = 0 →
  (p / (p*q + 2)) + (q / (p*r + 2)) + (r / (q*p + 2)) = 3/4 := by
sorry

end NUMINAMATH_CALUDE_root_sum_fraction_l950_95084


namespace NUMINAMATH_CALUDE_tunnel_length_l950_95015

/-- The length of a tunnel given train parameters -/
theorem tunnel_length (train_length : ℝ) (train_speed : ℝ) (time : ℝ) :
  train_length = 90 →
  train_speed = 160 →
  time = 3 →
  train_speed * time - train_length = 390 := by
  sorry

end NUMINAMATH_CALUDE_tunnel_length_l950_95015


namespace NUMINAMATH_CALUDE_red_square_area_equals_cross_area_l950_95007

/-- Represents a square flag with a symmetric cross -/
structure CrossFlag where
  /-- Side length of the flag -/
  side : ℝ
  /-- Ratio of the cross arm width to the flag side length -/
  arm_ratio : ℝ
  /-- The cross (arms + center) occupies 49% of the flag area -/
  cross_area_constraint : 4 * arm_ratio * (1 - arm_ratio) = 0.49

theorem red_square_area_equals_cross_area (flag : CrossFlag) :
  4 * flag.arm_ratio^2 = 4 * flag.arm_ratio * (1 - flag.arm_ratio) := by
  sorry

#check red_square_area_equals_cross_area

end NUMINAMATH_CALUDE_red_square_area_equals_cross_area_l950_95007


namespace NUMINAMATH_CALUDE_triangle_inequality_l950_95012

theorem triangle_inequality (a b c : ℝ) 
  (triangle_cond : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (abc_cond : a * b * c = 1) : 
  (Real.sqrt (b + c - a)) / a + (Real.sqrt (c + a - b)) / b + (Real.sqrt (a + b - c)) / c ≥ a + b + c :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l950_95012


namespace NUMINAMATH_CALUDE_eight_steps_result_l950_95078

def alternate_divide_multiply (n : ℕ) : ℕ → ℕ
  | 0 => n
  | i + 1 => if i % 2 = 0 then (alternate_divide_multiply n i) / 2 else (alternate_divide_multiply n i) * 3

theorem eight_steps_result :
  alternate_divide_multiply 10000000 8 = 2^3 * 3^4 * 5^7 := by
  sorry

end NUMINAMATH_CALUDE_eight_steps_result_l950_95078


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l950_95065

theorem trigonometric_expression_value (θ : Real) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l950_95065


namespace NUMINAMATH_CALUDE_total_votes_l950_95024

theorem total_votes (veggies_votes : ℕ) (meat_votes : ℕ) 
  (h1 : veggies_votes = 337) (h2 : meat_votes = 335) : 
  veggies_votes + meat_votes = 672 := by
  sorry

end NUMINAMATH_CALUDE_total_votes_l950_95024


namespace NUMINAMATH_CALUDE_min_even_integers_l950_95076

theorem min_even_integers (a b c d e f g : ℤ) : 
  a + b = 29 → 
  a + b + c + d = 47 → 
  a + b + c + d + e + f + g = 66 → 
  (∃ (count : ℕ), count ≥ 1 ∧ 
    count = (if Even a then 1 else 0) + 
            (if Even b then 1 else 0) + 
            (if Even c then 1 else 0) + 
            (if Even d then 1 else 0) + 
            (if Even e then 1 else 0) + 
            (if Even f then 1 else 0) + 
            (if Even g then 1 else 0) ∧
    ∀ (other_count : ℕ), 
      other_count = (if Even a then 1 else 0) + 
                    (if Even b then 1 else 0) + 
                    (if Even c then 1 else 0) + 
                    (if Even d then 1 else 0) + 
                    (if Even e then 1 else 0) + 
                    (if Even f then 1 else 0) + 
                    (if Even g then 1 else 0) →
      count ≤ other_count) :=
by sorry

end NUMINAMATH_CALUDE_min_even_integers_l950_95076


namespace NUMINAMATH_CALUDE_x_squared_in_set_l950_95058

theorem x_squared_in_set (x : ℝ) : x^2 ∈ ({0, -1, x} : Set ℝ) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_in_set_l950_95058


namespace NUMINAMATH_CALUDE_unique_grid_solution_l950_95094

-- Define the grid
def Grid := Fin 3 → Fin 3 → Option Char

-- Define adjacency
def adjacent (i j k l : Fin 3) : Prop :=
  (i = k ∧ j.val + 1 = l.val) ∨
  (i = k ∧ j.val = l.val + 1) ∨
  (i.val + 1 = k.val ∧ j = l) ∨
  (i.val = k.val + 1 ∧ j = l) ∨
  (i.val + 1 = k.val ∧ j.val + 1 = l.val) ∨
  (i.val + 1 = k.val ∧ j.val = l.val + 1) ∨
  (i.val = k.val + 1 ∧ j.val + 1 = l.val) ∨
  (i.val = k.val + 1 ∧ j.val = l.val + 1)

-- Define the constraints
def valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ∈ [none, some 'A', some 'B', some 'C']) ∧
  (∀ i, ∃! j, g i j = some 'A') ∧
  (∀ i, ∃! j, g i j = some 'B') ∧
  (∀ i, ∃! j, g i j = some 'C') ∧
  (∀ j, ∃! i, g i j = some 'A') ∧
  (∀ j, ∃! i, g i j = some 'B') ∧
  (∀ j, ∃! i, g i j = some 'C') ∧
  (∀ i j k l, adjacent i j k l → g i j ≠ g k l) ∧
  (g 0 1 = none ∧ g 1 0 = none)

-- Define the diagonal string
def diagonal_string (g : Grid) : String :=
  String.mk [
    (g 0 0).getD 'X',
    (g 1 1).getD 'X',
    (g 2 2).getD 'X'
  ]

-- The theorem to prove
theorem unique_grid_solution :
  ∀ g : Grid, valid_grid g → diagonal_string g = "XXC" := by
  sorry

end NUMINAMATH_CALUDE_unique_grid_solution_l950_95094


namespace NUMINAMATH_CALUDE_polynomial_no_x_x2_terms_l950_95061

theorem polynomial_no_x_x2_terms (m n : ℚ) : 
  (∀ x, 3 * (x^3 + 1/3 * x^2 + n * x) - (m * x^2 - 6 * x - 1) = 
        3 * x^3 + 1) → 
  m + n = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_no_x_x2_terms_l950_95061


namespace NUMINAMATH_CALUDE_penny_sock_cost_l950_95047

/-- Given Penny's shopping scenario, prove the cost of each pair of socks. -/
theorem penny_sock_cost (initial_amount : ℚ) (num_sock_pairs : ℕ) (hat_cost remaining_amount : ℚ) :
  initial_amount = 20 →
  num_sock_pairs = 4 →
  hat_cost = 7 →
  remaining_amount = 5 →
  ∃ (sock_cost : ℚ), 
    initial_amount - hat_cost - (num_sock_pairs : ℚ) * sock_cost = remaining_amount ∧
    sock_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_penny_sock_cost_l950_95047


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_lines_l950_95071

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_perpendicular_lines 
  (m n : Line) (α β : Plane) 
  (h_distinct_lines : m ≠ n) 
  (h_distinct_planes : α ≠ β) 
  (h_m_perp_n : perpendicular_lines m n) 
  (h_m_perp_α : perpendicular_line_plane m α) 
  (h_n_perp_β : perpendicular_line_plane n β) : 
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_lines_l950_95071


namespace NUMINAMATH_CALUDE_smallest_root_of_quadratic_l950_95091

theorem smallest_root_of_quadratic (x : ℝ) :
  (12 * x^2 - 44 * x + 40 = 0) → (x ≥ 5/3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_root_of_quadratic_l950_95091


namespace NUMINAMATH_CALUDE_no_bounded_function_satisfying_inequality_l950_95067

theorem no_bounded_function_satisfying_inequality :
  ¬ ∃ f : ℝ → ℝ, (∀ x : ℝ, ∃ M : ℝ, |f x| ≤ M) ∧ 
    (f 1 > 0) ∧ 
    (∀ x y : ℝ, (f (x + y))^2 ≥ (f x)^2 + 2 * f (x * y) + (f y)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_no_bounded_function_satisfying_inequality_l950_95067


namespace NUMINAMATH_CALUDE_range_of_a_l950_95046

/-- Given a system of equations and an inequality, prove the range of values for a. -/
theorem range_of_a (a x y : ℝ) 
  (eq1 : x + y = 3 * a + 4)
  (eq2 : x - y = 7 * a - 4)
  (ineq : 3 * x - 2 * y < 11) :
  a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l950_95046


namespace NUMINAMATH_CALUDE_joe_initial_cars_l950_95040

/-- The number of cars Joe will have after getting more -/
def total_cars : ℕ := 62

/-- The number of additional cars Joe will get -/
def additional_cars : ℕ := 12

/-- Joe's initial number of cars -/
def initial_cars : ℕ := total_cars - additional_cars

theorem joe_initial_cars : initial_cars = 50 := by sorry

end NUMINAMATH_CALUDE_joe_initial_cars_l950_95040


namespace NUMINAMATH_CALUDE_nine_twin_functions_l950_95044

-- Define the function f(x) = 2x^2 + 1
def f (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the range set
def range_set : Set ℝ := {5, 19}

-- Define the property for a valid domain
def is_valid_domain (D : Set ℝ) : Prop :=
  (∀ x ∈ D, f x ∈ range_set) ∧ 
  (∀ y ∈ range_set, ∃ x ∈ D, f x = y)

-- State the theorem
theorem nine_twin_functions :
  ∃! (domains : Finset (Set ℝ)), 
    Finset.card domains = 9 ∧ 
    (∀ D ∈ domains, is_valid_domain D) ∧
    (∀ D : Set ℝ, is_valid_domain D → D ∈ domains) :=
sorry

end NUMINAMATH_CALUDE_nine_twin_functions_l950_95044


namespace NUMINAMATH_CALUDE_repeating_decimal_problem_l950_95045

theorem repeating_decimal_problem (a b : ℕ) (h1 : a < 10) (h2 : b < 10) : 
  66 * (1 + (10 * a + b) / 99) - 66 * (1 + (10 * a + b) / 100) = 1/2 → 
  10 * a + b = 75 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_problem_l950_95045


namespace NUMINAMATH_CALUDE_decagon_adjacent_probability_l950_95048

/-- A decagon is a polygon with 10 vertices -/
def Decagon := {n : ℕ // n = 10}

/-- The number of ways to choose 2 distinct vertices from a decagon -/
def totalChoices (d : Decagon) : ℕ := (d.val.choose 2)

/-- The number of ways to choose 2 adjacent vertices from a decagon -/
def adjacentChoices (d : Decagon) : ℕ := 2 * d.val

/-- The probability of choosing two adjacent vertices in a decagon -/
def adjacentProbability (d : Decagon) : ℚ :=
  (adjacentChoices d : ℚ) / (totalChoices d : ℚ)

theorem decagon_adjacent_probability (d : Decagon) :
  adjacentProbability d = 4/9 := by sorry

end NUMINAMATH_CALUDE_decagon_adjacent_probability_l950_95048


namespace NUMINAMATH_CALUDE_max_value_x_2y_plus_1_l950_95059

theorem max_value_x_2y_plus_1 (x y : ℝ) 
  (hx : |x - 1| ≤ 1) 
  (hy : |y - 2| ≤ 1) : 
  |x - 2*y + 1| ≤ 5 := by sorry

end NUMINAMATH_CALUDE_max_value_x_2y_plus_1_l950_95059


namespace NUMINAMATH_CALUDE_complex_division_result_l950_95014

theorem complex_division_result : (1 + 2*I) / (1 - 2*I) = -3/5 + 4/5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l950_95014


namespace NUMINAMATH_CALUDE_tens_digit_of_19_power_2023_l950_95075

theorem tens_digit_of_19_power_2023 : ∃ n : ℕ, 19^2023 ≡ 50 + n [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_19_power_2023_l950_95075


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l950_95026

/-- A function f is decreasing on an open interval (a,b) if for all x, y in (a,b), x < y implies f(x) > f(y) -/
def DecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → f x > f y

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h1 : DecreasingOn f (-1) 1)
  (h2 : f (1 - a) < f (3 * a - 1)) :
  0 < a ∧ a < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l950_95026


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l950_95037

theorem infinite_geometric_series_first_term 
  (r : ℚ) (S : ℚ) (h1 : r = -1/3) (h2 : S = 9) :
  let a := S * (1 - r)
  a = 12 := by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l950_95037


namespace NUMINAMATH_CALUDE_personalized_pencil_cost_l950_95020

/-- The cost of personalized pencils with a discount for large orders -/
theorem personalized_pencil_cost 
  (base_cost : ℝ)  -- Cost for 100 pencils
  (base_quantity : ℕ)  -- Base quantity (100 pencils)
  (discount_threshold : ℕ)  -- Threshold for discount (1000 pencils)
  (discount_rate : ℝ)  -- Discount rate (5%)
  (order_quantity : ℕ)  -- Quantity ordered (2500 pencils)
  (h1 : base_cost = 30)
  (h2 : base_quantity = 100)
  (h3 : discount_threshold = 1000)
  (h4 : discount_rate = 0.05)
  (h5 : order_quantity = 2500) :
  let cost_per_pencil := base_cost / base_quantity
  let full_cost := cost_per_pencil * order_quantity
  let discounted_cost := full_cost * (1 - discount_rate)
  (if order_quantity > discount_threshold then discounted_cost else full_cost) = 712.5 := by
  sorry


end NUMINAMATH_CALUDE_personalized_pencil_cost_l950_95020


namespace NUMINAMATH_CALUDE_pinterest_group_pins_l950_95025

/-- Calculates the number of pins in a Pinterest group after one month -/
def pinsAfterOneMonth (
  groupSize : ℕ
  ) (averageDailyContribution : ℕ
  ) (weeklyDeletionRate : ℕ
  ) (initialPins : ℕ
  ) : ℕ :=
  let daysInMonth : ℕ := 30
  let weeksInMonth : ℕ := 4
  let monthlyContribution := groupSize * averageDailyContribution * daysInMonth
  let monthlyDeletion := groupSize * weeklyDeletionRate * weeksInMonth
  initialPins + monthlyContribution - monthlyDeletion

theorem pinterest_group_pins :
  pinsAfterOneMonth 20 10 5 1000 = 6600 := by
  sorry

end NUMINAMATH_CALUDE_pinterest_group_pins_l950_95025


namespace NUMINAMATH_CALUDE_root_in_interval_l950_95088

theorem root_in_interval : ∃ x : ℝ, 2 < x ∧ x < 3 ∧ Real.log x + x - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l950_95088


namespace NUMINAMATH_CALUDE_chess_game_draw_probability_l950_95010

theorem chess_game_draw_probability
  (p_a_not_lose : ℝ)
  (p_b_not_lose : ℝ)
  (h_a : p_a_not_lose = 0.8)
  (h_b : p_b_not_lose = 0.7)
  (h_game : ∀ (p_a_win p_draw : ℝ),
    p_a_win + p_draw = p_a_not_lose ∧
    (1 - p_a_win) = p_b_not_lose) :
  ∃ (p_draw : ℝ), p_draw = 0.5 := by
sorry

end NUMINAMATH_CALUDE_chess_game_draw_probability_l950_95010


namespace NUMINAMATH_CALUDE_parabola_focus_l950_95021

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y = -4 * x^2 + 4 * x - 1

/-- The focus of a parabola -/
def is_focus (f_x f_y : ℝ) : Prop :=
  f_x = 1/2 ∧ f_y = -1/8

/-- Theorem: The focus of the parabola y = -4x^2 + 4x - 1 is (1/2, -1/8) -/
theorem parabola_focus :
  ∃ (f_x f_y : ℝ), (∀ x y, parabola_equation x y → is_focus f_x f_y) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l950_95021


namespace NUMINAMATH_CALUDE_reflection_of_P_is_correct_l950_95082

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflection_of_P_is_correct : 
  let P : Point := { x := 2, y := -3 }
  reflectXAxis P = { x := 2, y := 3 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_P_is_correct_l950_95082


namespace NUMINAMATH_CALUDE_commodity_tax_consumption_l950_95085

theorem commodity_tax_consumption (T C : ℝ) (h1 : T > 0) (h2 : C > 0) : 
  let new_tax := 0.75 * T
  let new_revenue := 0.825 * T * C
  let new_consumption := C * (1 + 10 / 100)
  new_tax * new_consumption = new_revenue := by sorry

end NUMINAMATH_CALUDE_commodity_tax_consumption_l950_95085


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l950_95077

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l950_95077


namespace NUMINAMATH_CALUDE_fraction_increase_l950_95027

theorem fraction_increase (x y : ℝ) (square : ℝ) :
  (2 * x * y) / (x + square) = (1 / 5) * (2 * (5 * x) * (5 * y)) / (5 * x + 5 * square) →
  square = 3 * y :=
by sorry

end NUMINAMATH_CALUDE_fraction_increase_l950_95027


namespace NUMINAMATH_CALUDE_gcf_of_180_250_300_l950_95069

theorem gcf_of_180_250_300 : Nat.gcd 180 (Nat.gcd 250 300) = 10 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_180_250_300_l950_95069


namespace NUMINAMATH_CALUDE_factorization_example_l950_95013

theorem factorization_example (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

#check factorization_example

end NUMINAMATH_CALUDE_factorization_example_l950_95013


namespace NUMINAMATH_CALUDE_fred_likes_twelve_pairs_l950_95057

theorem fred_likes_twelve_pairs : 
  (Finset.filter (fun n : Fin 100 => n.val % 8 = 0) Finset.univ).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_fred_likes_twelve_pairs_l950_95057


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_l950_95006

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line
def line (x y : ℝ) : Prop := y = x + 1

theorem ellipse_parabola_intersection :
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
    ellipse 0 1 ∧  -- Vertex of ellipse at (0, 1)
    parabola 0 1 ∧  -- Focus of parabola at (0, 1)
    line x₁ y₁ ∧
    line x₂ y₂ ∧
    parabola x₁ y₁ ∧
    parabola x₂ y₂ ∧
    x₁ * x₂ = -4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_l950_95006


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l950_95002

def is_multiple_of_29 (n : ℕ) : Prop := ∃ k : ℕ, n = 29 * k

def last_two_digits_are_29 (n : ℕ) : Prop := n % 100 = 29

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem smallest_number_with_conditions : 
  (is_multiple_of_29 51729) ∧ 
  (last_two_digits_are_29 51729) ∧ 
  (sum_of_digits 51729 = 29) ∧
  (∀ m : ℕ, m < 51729 → 
    ¬(is_multiple_of_29 m ∧ last_two_digits_are_29 m ∧ sum_of_digits m = 29)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l950_95002


namespace NUMINAMATH_CALUDE_triangle_coordinate_difference_l950_95036

/-- Triangle ABC with vertices A(0,10), B(4,0), C(10,0), and a vertical line
    intersecting AC at R and BC at S. If the area of triangle RSC is 20,
    then the positive difference between the x and y coordinates of R is 4√10 - 10. -/
theorem triangle_coordinate_difference (R : ℝ × ℝ) :
  let A : ℝ × ℝ := (0, 10)
  let B : ℝ × ℝ := (4, 0)
  let C : ℝ × ℝ := (10, 0)
  let S : ℝ × ℝ := (R.1, 0)  -- S has same x-coordinate as R and y-coordinate 0
  -- R is on line AC
  (10 - R.1) / (0 - R.2) = 1 →
  -- RS is vertical (same x-coordinate)
  R.1 = S.1 →
  -- Area of triangle RSC is 20
  abs ((R.1 - 10) * R.2) / 2 = 20 →
  -- The positive difference between x and y coordinates of R
  abs (R.2 - R.1) = 4 * Real.sqrt 10 - 10 :=
by sorry

end NUMINAMATH_CALUDE_triangle_coordinate_difference_l950_95036


namespace NUMINAMATH_CALUDE_expression_value_l950_95081

theorem expression_value : (5^8 - 3^7) * (1^6 + (-1)^5)^11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l950_95081


namespace NUMINAMATH_CALUDE_percentage_of_indian_children_l950_95097

theorem percentage_of_indian_children 
  (total_men : ℕ) 
  (total_women : ℕ) 
  (total_children : ℕ) 
  (percent_indian_men : ℚ) 
  (percent_indian_women : ℚ) 
  (percent_non_indian : ℚ) :
  total_men = 500 →
  total_women = 300 →
  total_children = 500 →
  percent_indian_men = 10 / 100 →
  percent_indian_women = 60 / 100 →
  percent_non_indian = 55.38461538461539 / 100 →
  (↑(total_men * 10 + total_women * 60 + total_children * 70) / ↑(total_men + total_women + total_children) : ℚ) = 1 - percent_non_indian :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_indian_children_l950_95097


namespace NUMINAMATH_CALUDE_negation_equivalence_l950_95005

theorem negation_equivalence :
  (¬ ∀ x : ℝ, ∃ n : ℕ+, (n : ℝ) ≥ x^2) ↔ (∃ x : ℝ, ∀ n : ℕ+, (n : ℝ) < x^2) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l950_95005


namespace NUMINAMATH_CALUDE_hannah_bought_three_sweatshirts_l950_95074

/-- Represents the purchase of sweatshirts and T-shirts by Hannah -/
structure Purchase where
  sweatshirts : ℕ
  tshirts : ℕ
  sweatshirt_cost : ℕ
  tshirt_cost : ℕ
  total_spent : ℕ

/-- Hannah's specific purchase -/
def hannahs_purchase : Purchase where
  sweatshirts := 0  -- We'll prove this should be 3
  tshirts := 2
  sweatshirt_cost := 15
  tshirt_cost := 10
  total_spent := 65

/-- The theorem stating that Hannah bought 3 sweatshirts -/
theorem hannah_bought_three_sweatshirts :
  ∃ (p : Purchase), p.tshirts = 2 ∧ p.sweatshirt_cost = 15 ∧ p.tshirt_cost = 10 ∧ p.total_spent = 65 ∧ p.sweatshirts = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_hannah_bought_three_sweatshirts_l950_95074


namespace NUMINAMATH_CALUDE_largest_coin_through_hole_l950_95035

-- Define the diameter of coins
def diameter (coin : String) : ℝ :=
  match coin with
  | "1 kopeck" => 1
  | "20 kopeck" => 2
  | _ => 0

-- Define a circular hole
structure CircularHole where
  diameter : ℝ

-- Define a function to check if a coin can pass through a hole when paper is folded
def canPassThroughWhenFolded (coin : String) (hole : CircularHole) : Prop :=
  diameter coin ≤ 2 * hole.diameter

theorem largest_coin_through_hole :
  let hole : CircularHole := ⟨diameter "1 kopeck"⟩
  canPassThroughWhenFolded "20 kopeck" hole := by
  sorry

end NUMINAMATH_CALUDE_largest_coin_through_hole_l950_95035


namespace NUMINAMATH_CALUDE_initial_kittens_l950_95032

/-- The number of kittens Tim gave to Jessica -/
def kittens_to_jessica : ℕ := 3

/-- The number of kittens Tim gave to Sara -/
def kittens_to_sara : ℕ := 6

/-- The number of kittens Tim has left -/
def kittens_left : ℕ := 9

/-- Theorem: Tim's initial number of kittens was 18 -/
theorem initial_kittens : 
  kittens_to_jessica + kittens_to_sara + kittens_left = 18 := by
  sorry

end NUMINAMATH_CALUDE_initial_kittens_l950_95032


namespace NUMINAMATH_CALUDE_triangle_problem_l950_95028

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_problem (t : Triangle) 
  (h1 : (t.a + t.b) * (Real.sin t.A - Real.sin t.B) = (t.c - t.b) * Real.sin t.C)
  (h2 : 2 * t.c = 3 * t.b)
  (h3 : 1/2 * t.b * t.c * Real.sin t.A = 6 * Real.sqrt 3) :
  t.A = π/3 ∧ t.a = 2 * Real.sqrt (21/3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l950_95028


namespace NUMINAMATH_CALUDE_pass_through_walls_l950_95019

theorem pass_through_walls (k : ℕ) (n : ℕ) : 
  k = 8 → 
  (k * Real.sqrt (k / ((k - 1) * k + (k - 1))) = Real.sqrt (k * (k / n))) ↔ 
  n = 63 := by
sorry

end NUMINAMATH_CALUDE_pass_through_walls_l950_95019


namespace NUMINAMATH_CALUDE_square_sum_value_l950_95083

theorem square_sum_value (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 12) : a^2 + b^2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l950_95083


namespace NUMINAMATH_CALUDE_ferris_wheel_rides_l950_95054

theorem ferris_wheel_rides (rollercoaster_rides catapult_rides : ℕ) 
  (rollercoaster_cost catapult_cost ferris_wheel_cost total_tickets : ℕ) :
  rollercoaster_rides = 3 →
  catapult_rides = 2 →
  rollercoaster_cost = 4 →
  catapult_cost = 4 →
  ferris_wheel_cost = 1 →
  total_tickets = 21 →
  (total_tickets - (rollercoaster_rides * rollercoaster_cost + catapult_rides * catapult_cost)) / ferris_wheel_cost = 1 :=
by sorry

end NUMINAMATH_CALUDE_ferris_wheel_rides_l950_95054


namespace NUMINAMATH_CALUDE_bankers_discount_calculation_l950_95093

/-- Banker's discount calculation -/
theorem bankers_discount_calculation
  (true_discount : ℝ)
  (sum_due : ℝ)
  (h1 : true_discount = 60)
  (h2 : sum_due = 360) :
  true_discount + (true_discount^2 / sum_due) = 70 :=
by sorry

end NUMINAMATH_CALUDE_bankers_discount_calculation_l950_95093


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l950_95009

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l950_95009


namespace NUMINAMATH_CALUDE_solve_halloween_decorations_l950_95031

/-- Represents the Halloween decoration problem --/
def halloween_decorations 
  (skulls : ℕ) 
  (broomsticks : ℕ) 
  (spiderwebs : ℕ) 
  (cauldrons : ℕ) 
  (total_planned : ℕ) : Prop :=
  let pumpkins := 2 * spiderwebs
  let total_put_up := skulls + broomsticks + spiderwebs + pumpkins + cauldrons
  let left_to_put_up := total_planned - total_put_up
  left_to_put_up = 30

/-- Theorem stating the solution to the Halloween decoration problem --/
theorem solve_halloween_decorations : 
  halloween_decorations 12 4 12 1 83 :=
by
  sorry

#check solve_halloween_decorations

end NUMINAMATH_CALUDE_solve_halloween_decorations_l950_95031


namespace NUMINAMATH_CALUDE_existence_of_abc_l950_95063

theorem existence_of_abc (n : ℕ) (A : Finset ℕ) :
  A ⊆ Finset.range (5^n + 1) →
  A.card = 4*n + 2 →
  ∃ a b c : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a < b ∧ b < c ∧ c + 2*a > 3*b :=
sorry

end NUMINAMATH_CALUDE_existence_of_abc_l950_95063


namespace NUMINAMATH_CALUDE_kids_savings_l950_95008

-- Define the number of coins each child has
def teagan_pennies : ℕ := 200
def rex_nickels : ℕ := 100
def toni_dimes : ℕ := 330

-- Define the value of each coin type in cents
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10

-- Define the total savings in cents
def total_savings : ℕ := 
  teagan_pennies * penny_value + 
  rex_nickels * nickel_value + 
  toni_dimes * dime_value

-- Theorem to prove
theorem kids_savings : total_savings = 4000 := by
  sorry

end NUMINAMATH_CALUDE_kids_savings_l950_95008


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l950_95029

/-- The axis of symmetry of the parabola y = x^2 + 4x - 5 is the line x = -2 -/
theorem parabola_axis_of_symmetry :
  let f : ℝ → ℝ := fun x ↦ x^2 + 4*x - 5
  ∃ (a : ℝ), a = -2 ∧ ∀ (x y : ℝ), f (a + x) = f (a - x) := by
  sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l950_95029


namespace NUMINAMATH_CALUDE_equal_numbers_l950_95066

theorem equal_numbers (x : Fin 2011 → ℝ) (x' : Fin 2011 → ℝ)
  (h1 : ∀ i : Fin 2011, x i + x (i + 1) = 2 * x' i)
  (h2 : ∃ σ : Equiv.Perm (Fin 2011), ∀ i, x' i = x (σ i)) :
  ∀ i j : Fin 2011, x i = x j :=
sorry

end NUMINAMATH_CALUDE_equal_numbers_l950_95066


namespace NUMINAMATH_CALUDE_boy_girl_ratio_l950_95038

/-- Represents the number of students in the class -/
def total_students : ℕ := 25

/-- Represents the difference between the number of boys and girls -/
def boy_girl_difference : ℕ := 9

/-- Theorem stating that the ratio of boys to girls is 17:8 -/
theorem boy_girl_ratio :
  ∃ (boys girls : ℕ),
    boys + girls = total_students ∧
    boys = girls + boy_girl_difference ∧
    boys = 17 ∧
    girls = 8 :=
by sorry

end NUMINAMATH_CALUDE_boy_girl_ratio_l950_95038


namespace NUMINAMATH_CALUDE_box_paperclips_relation_small_box_medium_box_large_box_l950_95011

/-- Represents the number of paperclips a box can hold based on its volume -/
noncomputable def paperclips (volume : ℝ) : ℝ :=
  50 * (volume / 16)

theorem box_paperclips_relation (v : ℝ) :
  paperclips v = 50 * (v / 16) :=
by sorry

theorem small_box : paperclips 16 = 50 :=
by sorry

theorem medium_box : paperclips 32 = 100 :=
by sorry

theorem large_box : paperclips 64 = 200 :=
by sorry

end NUMINAMATH_CALUDE_box_paperclips_relation_small_box_medium_box_large_box_l950_95011


namespace NUMINAMATH_CALUDE_consecutive_integer_roots_l950_95098

theorem consecutive_integer_roots (p q : ℤ) : 
  (∃ x y : ℤ, x^2 - p*x + q = 0 ∧ y^2 - p*y + q = 0 ∧ y = x + 1) →
  Prime q →
  (p = 3 ∨ p = -3) ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integer_roots_l950_95098


namespace NUMINAMATH_CALUDE_inequality_proof_l950_95000

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b^3 / (a^2 + 8*b*c)) + (c^3 / (b^2 + 8*c*a)) + (a^3 / (c^2 + 8*a*b)) ≥ (1/9) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l950_95000


namespace NUMINAMATH_CALUDE_discount_problem_l950_95053

/-- Calculate the original price given the discounted price and discount rate -/
def originalPrice (discountedPrice : ℚ) (discountRate : ℚ) : ℚ :=
  discountedPrice / (1 - discountRate)

/-- The problem statement -/
theorem discount_problem (item1_discounted : ℚ) (item1_rate : ℚ)
                         (item2_discounted : ℚ) (item2_rate : ℚ)
                         (item3_discounted : ℚ) (item3_rate : ℚ) :
  item1_discounted = 4400 →
  item1_rate = 56 / 100 →
  item2_discounted = 3900 →
  item2_rate = 35 / 100 →
  item3_discounted = 2400 →
  item3_rate = 20 / 100 →
  originalPrice item1_discounted item1_rate +
  originalPrice item2_discounted item2_rate +
  originalPrice item3_discounted item3_rate = 19000 := by
  sorry

end NUMINAMATH_CALUDE_discount_problem_l950_95053


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l950_95096

theorem smallest_x_absolute_value_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 2 ∧
  ∀ (y : ℝ), y * |y| = 3 * y + 2 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l950_95096
