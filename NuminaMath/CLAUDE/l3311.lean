import Mathlib

namespace NUMINAMATH_CALUDE_star_op_example_l3311_331178

/-- Custom binary operation ☼ defined for rational numbers -/
def star_op (a b : ℚ) : ℚ := a^3 - 2*a*b + 4

/-- Theorem stating that 4 ☼ (-9) = 140 -/
theorem star_op_example : star_op 4 (-9) = 140 := by
  sorry

end NUMINAMATH_CALUDE_star_op_example_l3311_331178


namespace NUMINAMATH_CALUDE_f_plus_g_at_2_l3311_331188

-- Define f and g as functions from ℝ to ℝ
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem f_plus_g_at_2 (hf : is_even f) (hg : is_odd g) 
  (h : ∀ x, f x - g x = x^3 + 2^(-x)) : 
  f 2 + g 2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_f_plus_g_at_2_l3311_331188


namespace NUMINAMATH_CALUDE_angle_c_not_five_sixths_pi_l3311_331148

theorem angle_c_not_five_sixths_pi (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_eq1 : 3 * Real.sin A + 4 * Real.cos B = 6) 
  (h_eq2 : 3 * Real.cos A + 4 * Real.sin B = 1) : 
  C ≠ 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_c_not_five_sixths_pi_l3311_331148


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l3311_331156

-- Define the radius, height, and volumes
variable (r : ℝ) -- radius of the base (shared by cone and sphere)
variable (h : ℝ) -- height of the cone
variable (V_sphere V_cone : ℝ) -- volumes of sphere and cone

-- Define the theorem
theorem cone_sphere_ratio :
  (V_sphere = (4/3) * Real.pi * r^3) →  -- volume formula for sphere
  (V_cone = (1/3) * Real.pi * r^2 * h) →  -- volume formula for cone
  (V_cone = (1/3) * V_sphere) →  -- given condition
  (h / r = 4/3) :=  -- conclusion to prove
by
  sorry  -- proof omitted

end NUMINAMATH_CALUDE_cone_sphere_ratio_l3311_331156


namespace NUMINAMATH_CALUDE_value_of_two_over_x_l3311_331184

theorem value_of_two_over_x (x : ℂ) (h : 1 - 5 / x + 9 / x^2 = 0) :
  2 / x = Complex.ofReal (5 / 9) - Complex.I * Complex.ofReal (Real.sqrt 11 / 9) ∨
  2 / x = Complex.ofReal (5 / 9) + Complex.I * Complex.ofReal (Real.sqrt 11 / 9) := by
sorry

end NUMINAMATH_CALUDE_value_of_two_over_x_l3311_331184


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3311_331171

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder -/
def major_axis_length (cylinder_radius : ℝ) (major_axis_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_axis_ratio

/-- Theorem: The length of the major axis of the ellipse is 5 -/
theorem ellipse_major_axis_length :
  major_axis_length 2 1.25 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3311_331171


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3311_331144

theorem arithmetic_calculations :
  ((-20) + (-14) - (-18) - 13 = -29) ∧
  ((-6) * (-2) / (1/8) = 96) ∧
  ((-24) * ((-3/4) - (5/6) + (7/8)) = 17) ∧
  (-(1^4) - (1 - 0.5) * (1/3) * ((-3)^2) = -5/2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3311_331144


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3311_331136

theorem max_value_of_expression (p q r s : ℕ) : 
  p ∈ ({1, 2, 3, 4} : Finset ℕ) → 
  q ∈ ({1, 2, 3, 4} : Finset ℕ) → 
  r ∈ ({1, 2, 3, 4} : Finset ℕ) → 
  s ∈ ({1, 2, 3, 4} : Finset ℕ) → 
  p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s → 
  p^q + r^s ≤ 83 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3311_331136


namespace NUMINAMATH_CALUDE_cosine_sine_inequality_l3311_331114

theorem cosine_sine_inequality (a b c : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a + b + c = π / 2) : 
  Real.cos a + Real.cos b + Real.cos c > Real.sin a + Real.sin b + Real.sin c := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_inequality_l3311_331114


namespace NUMINAMATH_CALUDE_concert_revenue_calculation_l3311_331100

/-- Calculates the total revenue from concert ticket sales given specific conditions --/
theorem concert_revenue_calculation (ticket_price : ℝ) 
  (first_ten_discount : ℝ) (next_twenty_discount : ℝ)
  (military_discount : ℝ) (student_discount : ℝ) (senior_discount : ℝ)
  (total_buyers : ℕ) (military_buyers : ℕ) (student_buyers : ℕ) (senior_buyers : ℕ) :
  ticket_price = 20 →
  first_ten_discount = 0.4 →
  next_twenty_discount = 0.15 →
  military_discount = 0.25 →
  student_discount = 0.2 →
  senior_discount = 0.1 →
  total_buyers = 85 →
  military_buyers = 8 →
  student_buyers = 12 →
  senior_buyers = 9 →
  (10 * (ticket_price * (1 - first_ten_discount)) +
   20 * (ticket_price * (1 - next_twenty_discount)) +
   military_buyers * (ticket_price * (1 - military_discount)) +
   student_buyers * (ticket_price * (1 - student_discount)) +
   senior_buyers * (ticket_price * (1 - senior_discount)) +
   (total_buyers - (10 + 20 + military_buyers + student_buyers + senior_buyers)) * ticket_price) = 1454 := by
  sorry


end NUMINAMATH_CALUDE_concert_revenue_calculation_l3311_331100


namespace NUMINAMATH_CALUDE_line_equation_for_triangle_l3311_331154

/-- Given a line passing through (a, 0) that forms a triangle with area T' in the first quadrant,
    prove that its equation is 2T'x - a^2y + 2aT' = 0 --/
theorem line_equation_for_triangle (a T' : ℝ) (h_a : a > 0) (h_T' : T' > 0) :
  ∃ (x y : ℝ → ℝ), ∀ t : ℝ,
    (x t = a ∧ y t = 0) ∨
    (x t = 0 ∧ y t = 2 * T' / a) ∨
    (x t ≥ 0 ∧ y t ≥ 0 ∧ 2 * T' * x t - a^2 * y t + 2 * a * T' = 0) :=
sorry

end NUMINAMATH_CALUDE_line_equation_for_triangle_l3311_331154


namespace NUMINAMATH_CALUDE_trivia_team_members_l3311_331176

/-- Represents a trivia team with its total members and points scored. -/
structure TriviaTeam where
  totalMembers : ℕ
  absentMembers : ℕ
  pointsPerMember : ℕ
  totalPoints : ℕ

/-- Theorem stating the total members in the trivia team -/
theorem trivia_team_members (team : TriviaTeam)
  (h1 : team.absentMembers = 4)
  (h2 : team.pointsPerMember = 8)
  (h3 : team.totalPoints = 64)
  : team.totalMembers = 12 := by
  sorry

#check trivia_team_members

end NUMINAMATH_CALUDE_trivia_team_members_l3311_331176


namespace NUMINAMATH_CALUDE_ball_picking_probabilities_l3311_331172

/-- The probability of picking ball 3 using method one -/
def P₁ : ℚ := 1/3

/-- The probability of picking ball 3 using method two -/
def P₂ : ℚ := 1/2

/-- The probability of picking ball 3 using method three -/
def P₃ : ℚ := 2/3

/-- Theorem stating the relationships between P₁, P₂, and P₃ -/
theorem ball_picking_probabilities :
  (P₁ < P₂) ∧ (P₁ < P₃) ∧ (P₂ ≠ P₃) ∧ (2 * P₁ = P₃) := by
  sorry

end NUMINAMATH_CALUDE_ball_picking_probabilities_l3311_331172


namespace NUMINAMATH_CALUDE_marble_problem_l3311_331122

theorem marble_problem (total : ℝ) (red blue yellow purple white : ℝ) : 
  red + blue + yellow + purple + white = total ∧
  red = 0.25 * total ∧
  blue = 0.15 * total ∧
  yellow = 0.20 * total ∧
  purple = 0.05 * total ∧
  white = 50 ∧
  total = 143 →
  blue + (red / 3) = 33 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l3311_331122


namespace NUMINAMATH_CALUDE_complex_subtraction_l3311_331111

theorem complex_subtraction (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 4 - I) :
  a - 2*b = -3 - I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l3311_331111


namespace NUMINAMATH_CALUDE_product_of_exponents_l3311_331101

theorem product_of_exponents (p r s : ℕ) : 
  3^p + 3^5 = 252 → 
  2^r + 58 = 122 → 
  5^3 * 6^s = 117000 → 
  p * r * s = 36 := by
  sorry

end NUMINAMATH_CALUDE_product_of_exponents_l3311_331101


namespace NUMINAMATH_CALUDE_max_daily_profit_l3311_331113

/-- The daily profit function for a store selling an item -/
noncomputable def daily_profit (x : ℕ) : ℝ :=
  if x ≥ 1 ∧ x ≤ 30 then
    -x^2 + 52*x + 620
  else if x ≥ 31 ∧ x ≤ 60 then
    -40*x + 2480
  else
    0

/-- The maximum daily profit and the day it occurs -/
theorem max_daily_profit :
  ∃ (max_profit : ℝ) (max_day : ℕ),
    max_profit = 1296 ∧
    max_day = 26 ∧
    (∀ x : ℕ, x ≥ 1 ∧ x ≤ 60 → daily_profit x ≤ max_profit) ∧
    daily_profit max_day = max_profit :=
by sorry

end NUMINAMATH_CALUDE_max_daily_profit_l3311_331113


namespace NUMINAMATH_CALUDE_weight_of_new_person_l3311_331185

theorem weight_of_new_person 
  (n : ℕ) 
  (initial_average : ℝ) 
  (weight_increase : ℝ) 
  (replaced_weight : ℝ) : 
  n = 8 → 
  weight_increase = 5 → 
  replaced_weight = 35 → 
  (n * initial_average + (n * weight_increase + replaced_weight)) / n = 
    initial_average + weight_increase → 
  n * weight_increase + replaced_weight = 75 := by
sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l3311_331185


namespace NUMINAMATH_CALUDE_quadratic_function_sum_l3311_331199

theorem quadratic_function_sum (a b : ℝ) : 
  a > 0 → 
  (∀ x ∈ Set.Icc 2 3, (a * x^2 - 2 * a * x + 1 + b) ≤ 4) →
  (∀ x ∈ Set.Icc 2 3, (a * x^2 - 2 * a * x + 1 + b) ≥ 1) →
  (∃ x ∈ Set.Icc 2 3, a * x^2 - 2 * a * x + 1 + b = 4) →
  (∃ x ∈ Set.Icc 2 3, a * x^2 - 2 * a * x + 1 + b = 1) →
  a + b = 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_sum_l3311_331199


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_three_l3311_331126

/-- Given a function y = x(1-ax) with maximum value 1/12 for 0 < x < 1/a, prove that a = 3 -/
theorem max_value_implies_a_equals_three (a : ℝ) : 
  (∃ (max_y : ℝ), max_y = 1/12 ∧ 
    (∀ x : ℝ, 0 < x → x < 1/a → x * (1 - a*x) ≤ max_y) ∧
    (∃ x : ℝ, 0 < x ∧ x < 1/a ∧ x * (1 - a*x) = max_y)) →
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_three_l3311_331126


namespace NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l3311_331109

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_16_to_30 :
  sum_of_squares 30 - sum_of_squares 15 = 8215 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l3311_331109


namespace NUMINAMATH_CALUDE_final_student_score_problem_solution_l3311_331103

theorem final_student_score (total_students : ℕ) (graded_students : ℕ) 
  (initial_average : ℚ) (final_average : ℚ) : ℚ :=
  let remaining_students := total_students - graded_students
  let initial_total := initial_average * graded_students
  let final_total := final_average * total_students
  (final_total - initial_total) / remaining_students

theorem problem_solution :
  final_student_score 20 19 75 78 = 135 := by sorry

end NUMINAMATH_CALUDE_final_student_score_problem_solution_l3311_331103


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3311_331143

theorem intersection_of_sets : 
  let P : Set ℤ := {-3, -2, 0, 2}
  let Q : Set ℤ := {-1, -2, -3, 0, 1}
  P ∩ Q = {-3, -2, 0} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3311_331143


namespace NUMINAMATH_CALUDE_no_solutions_exist_l3311_331108

-- Define the greatest prime factor function
def greatest_prime_factor (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_solutions_exist : 
  ¬ ∃ (n : ℕ), n > 1 ∧ 
  (greatest_prime_factor n = Real.sqrt n) ∧ 
  (greatest_prime_factor (n + 60) = Real.sqrt (n + 60)) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_exist_l3311_331108


namespace NUMINAMATH_CALUDE_binomial_20_17_l3311_331141

theorem binomial_20_17 : Nat.choose 20 17 = 1140 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_17_l3311_331141


namespace NUMINAMATH_CALUDE_lilliputian_matchboxes_in_gulliver_matchbox_l3311_331151

/-- The scale factor between Gulliver's homeland and Lilliput -/
def scaleFactor : ℕ := 12

/-- The dimensions of a matchbox (length, width, height) -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculate the volume of a matchbox given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- The number of Lilliputian matchboxes that fit in one dimension -/
def fitInOneDimension : ℕ := scaleFactor

theorem lilliputian_matchboxes_in_gulliver_matchbox (g : Dimensions) (l : Dimensions)
    (h_scale : l.length = g.length / scaleFactor ∧ 
               l.width = g.width / scaleFactor ∧ 
               l.height = g.height / scaleFactor) :
    (volume g) / (volume l) = 1728 := by
  sorry

end NUMINAMATH_CALUDE_lilliputian_matchboxes_in_gulliver_matchbox_l3311_331151


namespace NUMINAMATH_CALUDE_chromatic_flow_duality_l3311_331145

/-- A planar multigraph -/
structure PlanarMultigraph where
  -- Add necessary fields

/-- The dual of a planar multigraph -/
def dual (G : PlanarMultigraph) : PlanarMultigraph :=
  sorry

/-- The chromatic number of a planar multigraph -/
def chromaticNumber (G : PlanarMultigraph) : ℕ :=
  sorry

/-- The flow number of a planar multigraph -/
def flowNumber (G : PlanarMultigraph) : ℕ :=
  sorry

/-- Theorem: The chromatic number of a planar multigraph equals the flow number of its dual -/
theorem chromatic_flow_duality (G : PlanarMultigraph) :
    chromaticNumber G = flowNumber (dual G) :=
  sorry

end NUMINAMATH_CALUDE_chromatic_flow_duality_l3311_331145


namespace NUMINAMATH_CALUDE_r_daily_earning_l3311_331105

/-- Given the daily earnings of three individuals p, q, and r, prove that r earns 40 per day. -/
theorem r_daily_earning (p q r : ℕ) : 
  (9 * (p + q + r) = 1890) →
  (5 * (p + r) = 600) →
  (7 * (q + r) = 910) →
  r = 40 := by
  sorry

end NUMINAMATH_CALUDE_r_daily_earning_l3311_331105


namespace NUMINAMATH_CALUDE_range_of_a_l3311_331158

-- Define the function g
def g (x : ℝ) : ℝ := |x - 1| - |x - 2|

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, g x < a^2 + a + 1) →
  (a < -1 ∨ a > 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3311_331158


namespace NUMINAMATH_CALUDE_negation_of_forall_cubic_l3311_331180

theorem negation_of_forall_cubic (P : ℝ → Prop) :
  (¬ ∀ x < 0, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x < 0, x^3 - x^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_cubic_l3311_331180


namespace NUMINAMATH_CALUDE_fifth_bush_berries_l3311_331198

def berry_sequence : ℕ → ℕ
  | 0 => 3
  | 1 => 4
  | 2 => 7
  | 3 => 12
  | n + 4 => berry_sequence (n + 3) + (berry_sequence (n + 3) - berry_sequence (n + 2) + 2)

theorem fifth_bush_berries : berry_sequence 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_fifth_bush_berries_l3311_331198


namespace NUMINAMATH_CALUDE_sequence_convergence_and_general_term_l3311_331194

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The sequence a_n -/
noncomputable def a (x y : ℝ) : ℕ → ℝ
  | 0 => x
  | 1 => y
  | n + 2 => (a x y (n + 1) * a x y n + 1) / (a x y (n + 1) + a x y n)

/-- The general term formula for a_n when n ≥ 2 -/
noncomputable def a_general_term (x y : ℝ) (n : ℕ) : ℝ :=
  let num := 2 * ((y - 1) / (y + 1)) ^ (fib (n - 1)) * ((x - 1) / (x + 1)) ^ (fib (n - 2))
  let den := 1 - ((y - 1) / (y + 1)) ^ (fib (n - 1)) * ((x - 1) / (x + 1)) ^ (fib (n - 2))
  num / den - 1

theorem sequence_convergence_and_general_term (x y : ℝ) :
  (∃ n₀ : ℕ+, ∀ n ≥ n₀, a x y n = 1 ∨ a x y n = -1) ↔
    ((x = 1 ∧ y ≠ -1) ∨ (x = -1 ∧ y ≠ 1) ∨ (y = 1 ∧ x ≠ -1) ∨ (y = -1 ∧ x ≠ 1)) ∧
  ∀ n ≥ 2, a x y n = a_general_term x y n :=
by sorry

end NUMINAMATH_CALUDE_sequence_convergence_and_general_term_l3311_331194


namespace NUMINAMATH_CALUDE_value_of_expression_l3311_331138

theorem value_of_expression (x : ℝ) (h : x = 5) : 4 * x - 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3311_331138


namespace NUMINAMATH_CALUDE_deck_size_proof_l3311_331165

theorem deck_size_proof (r b : ℕ) : 
  (r : ℚ) / (r + b) = 1/4 →
  (r : ℚ) / (r + b + 6) = 1/6 →
  r + b = 12 := by
sorry

end NUMINAMATH_CALUDE_deck_size_proof_l3311_331165


namespace NUMINAMATH_CALUDE_constant_speed_calculation_l3311_331170

/-- Proves that a journey of 2304 kilometers completed in 36 hours at a constant speed results in a speed of 64 km/h -/
theorem constant_speed_calculation (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 2304 →
  time = 36 →
  speed = distance / time →
  speed = 64 := by
sorry

end NUMINAMATH_CALUDE_constant_speed_calculation_l3311_331170


namespace NUMINAMATH_CALUDE_square_sum_problem_l3311_331118

theorem square_sum_problem (a b c d m n : ℕ+) 
  (sum_eq : a + b + c + d = m^2)
  (sum_squares_eq : a^2 + b^2 + c^2 + d^2 = 1989)
  (max_eq : max a (max b (max c d)) = n^2) :
  m = 9 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_problem_l3311_331118


namespace NUMINAMATH_CALUDE_strengthened_inequality_l3311_331116

theorem strengthened_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  27 * (a^3 + b^3 + c^3) + 1 ≥ 12 * (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_strengthened_inequality_l3311_331116


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3311_331102

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 8) * (Real.sqrt 6 / Real.sqrt 9) = Real.sqrt 35 / 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3311_331102


namespace NUMINAMATH_CALUDE_cuboid_volume_doubled_l3311_331191

/-- The volume of a cuboid after doubling its dimensions -/
theorem cuboid_volume_doubled (original_volume : ℝ) : 
  original_volume = 36 → 8 * original_volume = 288 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_doubled_l3311_331191


namespace NUMINAMATH_CALUDE_divisors_of_420_l3311_331186

/-- The sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of divisors function -/
def num_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating the sum and number of divisors for 420 -/
theorem divisors_of_420 : 
  sum_of_divisors 420 = 1344 ∧ num_of_divisors 420 = 24 := by sorry

end NUMINAMATH_CALUDE_divisors_of_420_l3311_331186


namespace NUMINAMATH_CALUDE_bus_seat_capacity_l3311_331142

/-- Represents the seating configuration of a bus -/
structure BusSeating where
  left_seats : Nat
  right_seats : Nat
  back_seat_capacity : Nat
  total_capacity : Nat

/-- Calculates the number of people each regular seat can hold -/
def seat_capacity (bus : BusSeating) : Nat :=
  sorry

/-- Theorem stating that for the given bus configuration, each seat can hold 3 people -/
theorem bus_seat_capacity :
  let bus : BusSeating := {
    left_seats := 15,
    right_seats := 12,
    back_seat_capacity := 10,
    total_capacity := 91
  }
  seat_capacity bus = 3 := by sorry

end NUMINAMATH_CALUDE_bus_seat_capacity_l3311_331142


namespace NUMINAMATH_CALUDE_video_game_problem_l3311_331131

theorem video_game_problem (total_games : ℕ) (price_per_game : ℕ) (total_earnings : ℕ) :
  total_games - (total_earnings / price_per_game) = total_games - (total_earnings / price_per_game) :=
by sorry

end NUMINAMATH_CALUDE_video_game_problem_l3311_331131


namespace NUMINAMATH_CALUDE_onions_sold_l3311_331163

theorem onions_sold (initial : ℕ) (left : ℕ) (sold : ℕ) : 
  initial = 98 → left = 33 → sold = initial - left → sold = 65 := by
sorry

end NUMINAMATH_CALUDE_onions_sold_l3311_331163


namespace NUMINAMATH_CALUDE_unique_solution_fraction_equation_l3311_331197

theorem unique_solution_fraction_equation :
  ∃! x : ℝ, (x ≠ 3 ∧ x ≠ 4) ∧ (3 / (x - 3) = 4 / (x - 4)) ∧ x = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_fraction_equation_l3311_331197


namespace NUMINAMATH_CALUDE_basketball_substitutions_remainder_l3311_331190

/-- Represents the number of ways to make substitutions in a basketball game -/
def substitution_ways (total_players starters max_substitutions : ℕ) : ℕ :=
  sorry

/-- The main theorem about the basketball substitutions problem -/
theorem basketball_substitutions_remainder :
  substitution_ways 15 5 4 % 1000 = 301 := by sorry

end NUMINAMATH_CALUDE_basketball_substitutions_remainder_l3311_331190


namespace NUMINAMATH_CALUDE_brother_got_two_l3311_331152

-- Define the type for grades
inductive Grade : Type
  | one : Grade
  | two : Grade
  | three : Grade
  | four : Grade
  | five : Grade

-- Define the sneezing function
def grandmother_sneezes (statement : Prop) : Prop := sorry

-- Define the brother's grade
def brothers_grade : Grade := sorry

-- Theorem statement
theorem brother_got_two :
  -- Condition 1: When the brother tells the truth, the grandmother sneezes
  (∀ (statement : Prop), statement → grandmother_sneezes statement) →
  -- Condition 2: The brother said he got a "5", but the grandmother didn't sneeze
  (¬ grandmother_sneezes (brothers_grade = Grade.five)) →
  -- Condition 3: The brother said he got a "4", and the grandmother sneezed
  (grandmother_sneezes (brothers_grade = Grade.four)) →
  -- Condition 4: The brother said he got at least a "3", but the grandmother didn't sneeze
  (¬ grandmother_sneezes (brothers_grade = Grade.three ∨ brothers_grade = Grade.four ∨ brothers_grade = Grade.five)) →
  -- Conclusion: The brother's grade is 2
  brothers_grade = Grade.two :=
by
  sorry

end NUMINAMATH_CALUDE_brother_got_two_l3311_331152


namespace NUMINAMATH_CALUDE_passing_percentage_l3311_331150

def max_marks : ℕ := 300
def obtained_marks : ℕ := 160
def failed_by : ℕ := 20

theorem passing_percentage :
  (((obtained_marks + failed_by : ℚ) / max_marks) * 100 : ℚ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_l3311_331150


namespace NUMINAMATH_CALUDE_factoring_quadratic_l3311_331179

theorem factoring_quadratic (x : ℝ) : 60 * x + 90 - 15 * x^2 = 15 * (-x^2 + 4 * x + 6) := by
  sorry

end NUMINAMATH_CALUDE_factoring_quadratic_l3311_331179


namespace NUMINAMATH_CALUDE_more_ones_than_zeros_mod_500_l3311_331112

/-- A function that returns the number of 1's in the binary representation of a natural number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- A function that returns the number of digits in the binary representation of a natural number -/
def binaryDigits (n : ℕ) : ℕ := sorry

/-- The set of positive integers less than or equal to 1000 whose binary representation has more 1's than 0's -/
def M : Finset ℕ := sorry

theorem more_ones_than_zeros_mod_500 : M.card % 500 = 61 := by sorry

end NUMINAMATH_CALUDE_more_ones_than_zeros_mod_500_l3311_331112


namespace NUMINAMATH_CALUDE_sequence_x_perfect_square_l3311_331133

def perfect_square_sequence (s : ℕ → ℕ) : Prop :=
  ∀ n, ∃ m, s n = m^2

def sequence_x : ℕ → ℤ
| 0 => 0
| 1 => 3
| (n + 2) => 4 * sequence_x (n + 1) - sequence_x n

theorem sequence_x_perfect_square :
  perfect_square_sequence (λ n => (sequence_x (n + 1) * sequence_x (n - 1) + 9).natAbs) := by
  sorry

end NUMINAMATH_CALUDE_sequence_x_perfect_square_l3311_331133


namespace NUMINAMATH_CALUDE_square_of_binomial_formula_l3311_331130

theorem square_of_binomial_formula (x y : ℝ) :
  (2*x + y) * (y - 2*x) = y^2 - (2*x)^2 :=
by sorry

end NUMINAMATH_CALUDE_square_of_binomial_formula_l3311_331130


namespace NUMINAMATH_CALUDE_polygon_sides_l3311_331149

theorem polygon_sides (sum_interior_angles : ℕ) : sum_interior_angles = 1440 → ∃ n : ℕ, n = 10 ∧ (n - 2) * 180 = sum_interior_angles :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l3311_331149


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3311_331121

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 12 (x + 1) = Nat.choose 12 (2 * x - 1)) → (x = 2 ∨ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3311_331121


namespace NUMINAMATH_CALUDE_cos_equality_solution_l3311_331146

theorem cos_equality_solution (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (1230 * π / 180) → n = 150 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_solution_l3311_331146


namespace NUMINAMATH_CALUDE_time_after_adding_seconds_l3311_331167

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time (4:45:00 a.m.) -/
def initialTime : Time :=
  { hours := 4, minutes := 45, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 12345

/-- The resulting time after adding seconds -/
def resultTime : Time :=
  { hours := 8, minutes := 30, seconds := 45 }

theorem time_after_adding_seconds :
  addSeconds initialTime secondsToAdd = resultTime := by
  sorry

end NUMINAMATH_CALUDE_time_after_adding_seconds_l3311_331167


namespace NUMINAMATH_CALUDE_closest_fraction_l3311_331181

def medals_won : ℚ := 23 / 150

def options : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (closest : ℚ), closest ∈ options ∧
  ∀ (x : ℚ), x ∈ options → |medals_won - closest| ≤ |medals_won - x| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_fraction_l3311_331181


namespace NUMINAMATH_CALUDE_parameterization_valid_iff_l3311_331153

/-- A parameterization of a line is represented by an initial point and a direction vector -/
structure Parameterization where
  x₀ : ℝ
  y₀ : ℝ
  dx : ℝ
  dy : ℝ

/-- The line y = 2x - 4 -/
def line (x : ℝ) : ℝ := 2 * x - 4

/-- A parameterization is valid for the line y = 2x - 4 -/
def is_valid_parameterization (p : Parameterization) : Prop :=
  line p.x₀ = p.y₀ ∧ ∃ (t : ℝ), p.dx = t * 1 ∧ p.dy = t * 2

/-- Theorem: A parameterization is valid if and only if it satisfies the conditions -/
theorem parameterization_valid_iff (p : Parameterization) :
  is_valid_parameterization p ↔ 
  (line p.x₀ = p.y₀ ∧ ∃ (t : ℝ), p.dx = t * 1 ∧ p.dy = t * 2) :=
by sorry

end NUMINAMATH_CALUDE_parameterization_valid_iff_l3311_331153


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3311_331134

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, (- (1/2) * x^2 + 2*x > m*x) ↔ (0 < x ∧ x < 2)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3311_331134


namespace NUMINAMATH_CALUDE_equation_solutions_l3311_331177

theorem equation_solutions : 
  ∀ (n m : ℤ), n^4 - 2*n^2 = m^2 + 38 ↔ 
  ((m = 5 ∧ n = -3) ∨ (m = -5 ∧ n = -3) ∨ (m = 5 ∧ n = 3) ∨ (m = -5 ∧ n = 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3311_331177


namespace NUMINAMATH_CALUDE_dogs_return_simultaneously_l3311_331193

theorem dogs_return_simultaneously
  (L : ℝ)  -- Distance between the two people
  (v V u : ℝ)  -- Speeds of slow person, fast person, and dogs respectively
  (h1 : 0 < v)
  (h2 : v < V)
  (h3 : 0 < u)
  (h4 : V < u) :
  (2 * L * u) / ((u + V) * (u + v)) = (2 * L * u) / ((u + v) * (u + V)) :=
by sorry

end NUMINAMATH_CALUDE_dogs_return_simultaneously_l3311_331193


namespace NUMINAMATH_CALUDE_pencil_cost_to_selling_ratio_l3311_331125

/-- Given a purchase of 90 pencils sold at a loss equal to the selling price of 40 pencils,
    the ratio of the cost of 90 pencils to the selling price of 90 pencils is 13:1. -/
theorem pencil_cost_to_selling_ratio :
  ∀ (C S : ℝ),
  C > 0 → S > 0 →
  90 * C - 40 * S = 90 * S →
  (90 * C) / (90 * S) = 13 / 1 := by
sorry

end NUMINAMATH_CALUDE_pencil_cost_to_selling_ratio_l3311_331125


namespace NUMINAMATH_CALUDE_marker_notebook_cost_l3311_331135

theorem marker_notebook_cost :
  ∀ (m n : ℕ),
  (10 * m + 5 * n = 120) →
  (m > n) →
  (m = 10 ∧ n = 4) →
  (m + n = 14) :=
by sorry

end NUMINAMATH_CALUDE_marker_notebook_cost_l3311_331135


namespace NUMINAMATH_CALUDE_weight_replacement_l3311_331161

theorem weight_replacement (n : ℕ) (new_weight : ℝ) (avg_increase : ℝ) 
  (h1 : n = 8)
  (h2 : new_weight = 68)
  (h3 : avg_increase = 1) :
  n * avg_increase = new_weight - (new_weight - n * avg_increase) :=
by
  sorry

#check weight_replacement

end NUMINAMATH_CALUDE_weight_replacement_l3311_331161


namespace NUMINAMATH_CALUDE_sin_pi_6_minus_2alpha_l3311_331124

theorem sin_pi_6_minus_2alpha (α : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = (Real.cos (α - π/6), 1/2))
  (hb : b = (1, -2 * Real.sin α))
  (hab : a.1 * b.1 + a.2 * b.2 = 1/3) :
  Real.sin (π/6 - 2*α) = -7/9 := by
sorry

end NUMINAMATH_CALUDE_sin_pi_6_minus_2alpha_l3311_331124


namespace NUMINAMATH_CALUDE_eighth_row_interior_sum_l3311_331137

/-- Sum of all elements in row n of Pascal's Triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^(n-1)

/-- Sum of interior numbers in row n of Pascal's Triangle -/
def pascal_interior_sum (n : ℕ) : ℕ := pascal_row_sum n - 2

theorem eighth_row_interior_sum :
  pascal_interior_sum 8 = 126 := by sorry

end NUMINAMATH_CALUDE_eighth_row_interior_sum_l3311_331137


namespace NUMINAMATH_CALUDE_rectangle_ellipse_theorem_l3311_331110

/-- Represents a rectangle ABCD with an inscribed ellipse K -/
structure RectangleWithEllipse where
  -- Length of side AB
  ab : ℝ
  -- Length of side AD
  ad : ℝ
  -- Point M on AB where the minor axis of K intersects
  am : ℝ
  -- Point L on AB where the minor axis of K intersects
  lb : ℝ
  -- Ensure AB = 2
  ab_eq_two : ab = 2
  -- Ensure AD < √2
  ad_lt_sqrt_two : ad < Real.sqrt 2
  -- Ensure M and L are on AB
  m_l_on_ab : am + lb = ab

/-- The theorem to be proved -/
theorem rectangle_ellipse_theorem (rect : RectangleWithEllipse) :
  rect.am^2 - rect.lb^2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ellipse_theorem_l3311_331110


namespace NUMINAMATH_CALUDE_sum_x_y_equals_six_l3311_331174

theorem sum_x_y_equals_six (x y : ℝ) 
  (h1 : x^2 + y^2 = 8*x + 4*y - 20) 
  (h2 : x + y = 6) : 
  x + y = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_six_l3311_331174


namespace NUMINAMATH_CALUDE_inverse_125_mod_79_l3311_331169

theorem inverse_125_mod_79 (h : (5⁻¹ : ZMod 79) = 39) : (125⁻¹ : ZMod 79) = 69 := by
  sorry

end NUMINAMATH_CALUDE_inverse_125_mod_79_l3311_331169


namespace NUMINAMATH_CALUDE_binary_1101001101_equals_base4_311310_l3311_331183

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem binary_1101001101_equals_base4_311310 :
  let binary : List Bool := [true, true, false, true, false, false, true, true, false, true]
  decimal_to_base4 (binary_to_decimal binary) = [3, 1, 1, 3, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_binary_1101001101_equals_base4_311310_l3311_331183


namespace NUMINAMATH_CALUDE_permutation_exists_16_no_permutation_exists_15_l3311_331104

def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_valid_permutation (perm : List ℕ) (max_sum : ℕ) : Prop :=
  perm.length = numbers.length ∧
  perm.toFinset = numbers.toFinset ∧
  ∀ i, i + 2 < perm.length → perm[i]! + perm[i+1]! + perm[i+2]! ≤ max_sum

theorem permutation_exists_16 : ∃ perm, is_valid_permutation perm 16 :=
sorry

theorem no_permutation_exists_15 : ¬∃ perm, is_valid_permutation perm 15 :=
sorry

end NUMINAMATH_CALUDE_permutation_exists_16_no_permutation_exists_15_l3311_331104


namespace NUMINAMATH_CALUDE_product_reciprocal_sum_l3311_331140

theorem product_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_prod : x * y = 16) (h_recip : 1 / x = 3 * (1 / y)) : 
  x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_reciprocal_sum_l3311_331140


namespace NUMINAMATH_CALUDE_fish_selection_probabilities_l3311_331132

/-- The number of fish in the aquarium -/
def total_fish : ℕ := 6

/-- The number of black fish in the aquarium -/
def black_fish : ℕ := 4

/-- The number of red fish in the aquarium -/
def red_fish : ℕ := 2

/-- The number of days the teacher has classes -/
def class_days : ℕ := 4

/-- The probability of selecting fish of the same color in two consecutive draws -/
def prob_same_color : ℚ := 5 / 9

/-- The probability of selecting fish of different colors in two consecutive draws on exactly 2 out of 4 days -/
def prob_diff_color_two_days : ℚ := 800 / 2187

theorem fish_selection_probabilities :
  (prob_same_color = (black_fish / total_fish) ^ 2 + (red_fish / total_fish) ^ 2) ∧
  (prob_diff_color_two_days = 
    (class_days.choose 2 : ℚ) * prob_same_color ^ 2 * (1 - prob_same_color) ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_fish_selection_probabilities_l3311_331132


namespace NUMINAMATH_CALUDE_factorization_equality_l3311_331117

theorem factorization_equality (x y : ℝ) : 5 * x^2 + 6 * x * y - 8 * y^2 = (x + 2 * y) * (5 * x - 4 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3311_331117


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l3311_331147

/-- Proves that mixing 300 mL of 10% alcohol solution with 100 mL of 30% alcohol solution results in a 15% alcohol solution -/
theorem alcohol_mixture_proof :
  let solution_x_volume : ℝ := 300
  let solution_x_concentration : ℝ := 0.10
  let solution_y_volume : ℝ := 100
  let solution_y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.15
  let total_volume := solution_x_volume + solution_y_volume
  let total_alcohol := solution_x_volume * solution_x_concentration + solution_y_volume * solution_y_concentration
  total_alcohol / total_volume = target_concentration := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l3311_331147


namespace NUMINAMATH_CALUDE_water_bottle_shortage_l3311_331115

/-- Represents the water bottle consumption during a soccer match --/
structure WaterBottleConsumption where
  initial_bottles : ℕ
  first_break_players : ℕ
  first_break_bottles_per_player : ℕ
  second_break_players : ℕ
  second_break_bottles_per_player : ℕ
  second_break_extra_bottles : ℕ
  third_break_players : ℕ
  third_break_bottles_per_player : ℕ

/-- Calculates the shortage of water bottles after the match --/
def calculate_shortage (consumption : WaterBottleConsumption) : ℤ :=
  let total_used := 
    consumption.first_break_players * consumption.first_break_bottles_per_player +
    consumption.second_break_players * consumption.second_break_bottles_per_player +
    consumption.second_break_extra_bottles +
    consumption.third_break_players * consumption.third_break_bottles_per_player
  consumption.initial_bottles - total_used

/-- Theorem stating that there is a shortage of 4 bottles given the match conditions --/
theorem water_bottle_shortage : 
  ∃ (consumption : WaterBottleConsumption), 
    consumption.initial_bottles = 48 ∧
    consumption.first_break_players = 11 ∧
    consumption.first_break_bottles_per_player = 2 ∧
    consumption.second_break_players = 14 ∧
    consumption.second_break_bottles_per_player = 1 ∧
    consumption.second_break_extra_bottles = 4 ∧
    consumption.third_break_players = 12 ∧
    consumption.third_break_bottles_per_player = 1 ∧
    calculate_shortage consumption = -4 := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_shortage_l3311_331115


namespace NUMINAMATH_CALUDE_total_population_theorem_l3311_331123

/-- Represents the population of a school -/
structure SchoolPopulation where
  b : ℕ  -- number of boys
  g : ℕ  -- number of girls
  t : ℕ  -- number of teachers

/-- Checks if the school population satisfies the given conditions -/
def isValidPopulation (p : SchoolPopulation) : Prop :=
  p.b = 4 * p.g ∧ p.g = 2 * p.t

/-- Calculates the total population of the school -/
def totalPopulation (p : SchoolPopulation) : ℕ :=
  p.b + p.g + p.t

/-- Theorem stating that for a valid school population, 
    the total population is equal to 11b/8 -/
theorem total_population_theorem (p : SchoolPopulation) 
  (h : isValidPopulation p) : 
  (totalPopulation p : ℚ) = 11 * (p.b : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_total_population_theorem_l3311_331123


namespace NUMINAMATH_CALUDE_matchsticks_left_l3311_331182

/-- Calculates the number of matchsticks left in a box after Elvis and Ralph make their squares -/
theorem matchsticks_left (initial_count : ℕ) (elvis_square_size elvis_squares : ℕ) (ralph_square_size ralph_squares : ℕ) : 
  initial_count = 50 → 
  elvis_square_size = 4 → 
  ralph_square_size = 8 → 
  elvis_squares = 5 → 
  ralph_squares = 3 → 
  initial_count - (elvis_square_size * elvis_squares + ralph_square_size * ralph_squares) = 6 := by
sorry

end NUMINAMATH_CALUDE_matchsticks_left_l3311_331182


namespace NUMINAMATH_CALUDE_darnell_call_minutes_l3311_331129

/-- Represents the monthly phone usage and plans for Darnell -/
structure PhoneUsage where
  unlimited_plan_cost : ℝ
  alt_plan_text_cost : ℝ
  alt_plan_text_limit : ℝ
  alt_plan_call_cost : ℝ
  alt_plan_call_limit : ℝ
  texts_sent : ℝ
  alt_plan_savings : ℝ

/-- Calculates the number of minutes Darnell spends on the phone each month -/
def calculate_call_minutes (usage : PhoneUsage) : ℝ :=
  sorry

/-- Theorem stating that given the conditions, Darnell spends 60 minutes on the phone each month -/
theorem darnell_call_minutes (usage : PhoneUsage) 
  (h1 : usage.unlimited_plan_cost = 12)
  (h2 : usage.alt_plan_text_cost = 1)
  (h3 : usage.alt_plan_text_limit = 30)
  (h4 : usage.alt_plan_call_cost = 3)
  (h5 : usage.alt_plan_call_limit = 20)
  (h6 : usage.texts_sent = 60)
  (h7 : usage.alt_plan_savings = 1) :
  calculate_call_minutes usage = 60 :=
sorry

end NUMINAMATH_CALUDE_darnell_call_minutes_l3311_331129


namespace NUMINAMATH_CALUDE_window_purchase_savings_l3311_331168

/-- Represents the window store's pricing and discount structure -/
structure WindowStore where
  regularPrice : ℕ := 120
  freeWindowThreshold : ℕ := 5
  bulkDiscountThreshold : ℕ := 10
  bulkDiscountRate : ℚ := 0.05

/-- Calculates the cost of windows for an individual purchase -/
def individualCost (store : WindowStore) (quantity : ℕ) : ℚ :=
  let freeWindows := quantity / store.freeWindowThreshold
  let paidWindows := quantity - freeWindows
  let basePrice := paidWindows * store.regularPrice
  if quantity > store.bulkDiscountThreshold
  then basePrice * (1 - store.bulkDiscountRate)
  else basePrice

/-- Calculates the cost of windows for a collective purchase -/
def collectiveCost (store : WindowStore) (quantities : List ℕ) : ℚ :=
  let totalQuantity := quantities.sum
  let freeWindows := totalQuantity / store.freeWindowThreshold
  let paidWindows := totalQuantity - freeWindows
  let basePrice := paidWindows * store.regularPrice
  basePrice * (1 - store.bulkDiscountRate)

/-- Theorem statement for the window purchase problem -/
theorem window_purchase_savings (store : WindowStore) :
  let gregQuantity := 9
  let susanQuantity := 13
  let individualTotal := individualCost store gregQuantity + individualCost store susanQuantity
  let collectiveTotal := collectiveCost store [gregQuantity, susanQuantity]
  individualTotal - collectiveTotal = 162 := by
  sorry

end NUMINAMATH_CALUDE_window_purchase_savings_l3311_331168


namespace NUMINAMATH_CALUDE_folded_paper_area_ratio_l3311_331192

/-- Represents a rectangular piece of paper with specific folding properties -/
structure FoldedPaper where
  width : ℝ
  length : ℝ
  area : ℝ
  foldedArea : ℝ
  lengthIsDoubleWidth : length = 2 * width
  areaIsLengthTimesWidth : area = length * width
  foldedAreaCalculation : foldedArea = area - 2 * (width * width / 4)

/-- Theorem stating that the ratio of folded area to original area is 1/2 -/
theorem folded_paper_area_ratio 
    (paper : FoldedPaper) : paper.foldedArea / paper.area = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_area_ratio_l3311_331192


namespace NUMINAMATH_CALUDE_different_color_pairs_count_l3311_331189

def white_socks : ℕ := 4
def brown_socks : ℕ := 4
def blue_socks : ℕ := 2
def gray_socks : ℕ := 5

def total_socks : ℕ := white_socks + brown_socks + blue_socks + gray_socks

def different_color_pairs : ℕ := 
  white_socks * brown_socks + 
  white_socks * blue_socks + 
  white_socks * gray_socks + 
  brown_socks * blue_socks + 
  brown_socks * gray_socks + 
  blue_socks * gray_socks

theorem different_color_pairs_count : different_color_pairs = 82 := by
  sorry

end NUMINAMATH_CALUDE_different_color_pairs_count_l3311_331189


namespace NUMINAMATH_CALUDE_no_solution_lcm_equation_l3311_331173

theorem no_solution_lcm_equation :
  ¬∃ (n m : ℕ), Nat.lcm (n^2) m + Nat.lcm n (m^2) = 2019 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_lcm_equation_l3311_331173


namespace NUMINAMATH_CALUDE_line_translation_l3311_331139

-- Define the original line
def original_line (x : ℝ) : ℝ := 2 * x + 1

-- Define the translation amount
def translation : ℝ := -2

-- Define the translated line
def translated_line (x : ℝ) : ℝ := 2 * x - 1

-- Theorem stating that the translation of the original line results in the translated line
theorem line_translation :
  ∀ x : ℝ, translated_line x = original_line x + translation :=
sorry

end NUMINAMATH_CALUDE_line_translation_l3311_331139


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_equation_l3311_331106

theorem equal_roots_quadratic_equation (x m : ℝ) : 
  (∃ r : ℝ, ∀ x, x^2 - 5*x + m = 0 ↔ x = r) → m = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_equation_l3311_331106


namespace NUMINAMATH_CALUDE_diameter_of_circle_with_radius_seven_l3311_331128

/-- The diameter of a circle is twice its radius -/
def diameter (radius : ℝ) : ℝ := 2 * radius

/-- For a circle with radius 7, the diameter is 14 -/
theorem diameter_of_circle_with_radius_seven :
  diameter 7 = 14 := by sorry

end NUMINAMATH_CALUDE_diameter_of_circle_with_radius_seven_l3311_331128


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l3311_331164

/-- Represents a hyperbola in the form x²/a² - y²/b² = 1 --/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a parabola in the form y² = 2px --/
structure Parabola where
  p : ℝ
  h_pos_p : p > 0

/-- The focal length of a hyperbola is the distance from the center to a focus --/
def focal_length (h : Hyperbola) : ℝ := sorry

/-- The left vertex of a hyperbola --/
def left_vertex (h : Hyperbola) : ℝ × ℝ := sorry

/-- The focus of a parabola --/
def parabola_focus (p : Parabola) : ℝ × ℝ := sorry

/-- The directrix of a parabola --/
def parabola_directrix (p : Parabola) : ℝ × ℝ → Prop := sorry

/-- An asymptote of a hyperbola --/
def hyperbola_asymptote (h : Hyperbola) : ℝ × ℝ → Prop := sorry

/-- The distance between two points --/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_focal_length 
  (h : Hyperbola) 
  (p : Parabola) 
  (h_distance : distance (left_vertex h) (parabola_focus p) = 4)
  (h_intersection : ∃ (pt : ℝ × ℝ), hyperbola_asymptote h pt ∧ parabola_directrix p pt ∧ pt = (-2, -1)) :
  focal_length h = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l3311_331164


namespace NUMINAMATH_CALUDE_operation_terminates_l3311_331120

/-- A sequence of positive integers -/
def Sequence := List Nat

/-- Represents the operation of replacing adjacent numbers -/
inductive Operation
  | replaceLeft (x y : Nat) : Operation  -- Replaces (x, y) with (y+1, x)
  | replaceRight (x y : Nat) : Operation -- Replaces (x, y) with (x-1, x)

/-- Applies an operation to a sequence -/
def applyOperation (s : Sequence) (op : Operation) : Sequence :=
  match s, op with
  | x::y::rest, Operation.replaceLeft x' y' => if x > y then (y+1)::x::rest else s
  | x::y::rest, Operation.replaceRight x' y' => if x > y then (x-1)::x::rest else s
  | _, _ => s

/-- Theorem: The process of applying operations terminates after finite iterations -/
theorem operation_terminates (s : Sequence) : 
  ∃ (n : Nat), ∀ (ops : List Operation), ops.length > n → 
    (ops.foldl applyOperation s = s) := by
  sorry


end NUMINAMATH_CALUDE_operation_terminates_l3311_331120


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l3311_331159

theorem magnitude_of_complex_power : 
  Complex.abs ((2/3 : ℂ) + (5/6 : ℂ) * Complex.I) ^ 8 = (41^4 : ℝ) / 1679616 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l3311_331159


namespace NUMINAMATH_CALUDE_annie_candy_cost_l3311_331107

/-- Calculates the total cost of candies Annie bought for her classmates --/
def total_candy_cost (candy_a_cost candy_b_cost candy_c_cost : ℚ) 
                     (classmates : ℕ) 
                     (a_per_person b_per_person c_per_person : ℕ) : ℚ :=
  let cost_per_person := a_per_person * candy_a_cost + 
                         b_per_person * candy_b_cost + 
                         c_per_person * candy_c_cost
  cost_per_person * classmates

theorem annie_candy_cost : 
  total_candy_cost 0.1 0.15 0.2 35 3 2 1 = 28 := by
  sorry

end NUMINAMATH_CALUDE_annie_candy_cost_l3311_331107


namespace NUMINAMATH_CALUDE_customer_payment_l3311_331175

def cost_price : ℝ := 6425
def markup_percentage : ℝ := 24

theorem customer_payment (cost : ℝ) (markup : ℝ) :
  cost = cost_price →
  markup = markup_percentage →
  cost * (1 + markup / 100) = 7967 := by
  sorry

end NUMINAMATH_CALUDE_customer_payment_l3311_331175


namespace NUMINAMATH_CALUDE_trajectory_of_M_l3311_331119

/-- The trajectory of point M satisfying the given conditions -/
theorem trajectory_of_M (x y : ℝ) (h : x ≥ 3/2) :
  (∀ (t : ℝ), t^2 + y^2 = 1 → 
    Real.sqrt ((x - t)^2 + y^2) = Real.sqrt ((x - 2)^2 + y^2) + 1) →
  3 * x^2 - y^2 - 8 * x + 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_M_l3311_331119


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_when_a_2_a_values_when_A_union_B_equals_A_l3311_331166

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 + 2*x - 3 = 0}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - (a+1)*x + a = 0}

-- Define the complement of B in ℝ
def C_ℝB (a : ℝ) : Set ℝ := {x : ℝ | x ∉ B a}

-- Statement 1
theorem intersection_A_complement_B_when_a_2 :
  A ∩ C_ℝB 2 = {-3} :=
sorry

-- Statement 2
theorem a_values_when_A_union_B_equals_A :
  {a : ℝ | A ∪ B a = A} = {-3, 1} :=
sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_when_a_2_a_values_when_A_union_B_equals_A_l3311_331166


namespace NUMINAMATH_CALUDE_cos_pi_4_minus_alpha_l3311_331160

theorem cos_pi_4_minus_alpha (α : Real) (h : Real.sin (α - 7 * Real.pi / 4) = 1 / 2) :
  Real.cos (Real.pi / 4 - α) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_4_minus_alpha_l3311_331160


namespace NUMINAMATH_CALUDE_solution_to_equation_one_no_solution_to_equation_two_l3311_331157

-- Problem 1
theorem solution_to_equation_one (x : ℝ) : 
  (3 / x) - (2 / (x - 2)) = 0 ↔ x = 6 :=
sorry

-- Problem 2
theorem no_solution_to_equation_two :
  ¬∃ (x : ℝ), (3 / (4 - x)) + 2 = ((1 - x) / (x - 4)) :=
sorry

end NUMINAMATH_CALUDE_solution_to_equation_one_no_solution_to_equation_two_l3311_331157


namespace NUMINAMATH_CALUDE_cone_height_l3311_331196

theorem cone_height (r : Real) (h : Real) :
  (3 : Real) * (2 * Real.pi / 3) = 2 * Real.pi * r →
  h ^ 2 + r ^ 2 = 3 ^ 2 →
  h = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_l3311_331196


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3311_331187

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_roots_range (a b c : ℝ) (ha : a ≠ 0) :
  f a b c (-1) = -2 →
  f a b c (-1/2) = -1/4 →
  f a b c 0 = 1 →
  f a b c (1/2) = 7/4 →
  f a b c 1 = 2 →
  f a b c (3/2) = 7/4 →
  f a b c 2 = 1 →
  f a b c (5/2) = -1/4 →
  f a b c 3 = -2 →
  ∃ x₁ x₂ : ℝ, f a b c x₁ = 0 ∧ f a b c x₂ = 0 ∧ 
    -1/2 < x₁ ∧ x₁ < 0 ∧ 2 < x₂ ∧ x₂ < 5/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3311_331187


namespace NUMINAMATH_CALUDE_quadratic_minimum_interval_l3311_331162

theorem quadratic_minimum_interval (m : ℝ) : 
  (∀ x ∈ Set.Icc m (m + 2), x^2 - 4*x + 3 ≥ 5/4) ∧ 
  (∃ x ∈ Set.Icc m (m + 2), x^2 - 4*x + 3 = 5/4) →
  m = -3/2 ∨ m = 7/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_interval_l3311_331162


namespace NUMINAMATH_CALUDE_jill_net_salary_l3311_331127

/-- Represents Jill's financial situation --/
structure JillFinances where
  net_salary : ℝ
  discretionary_income : ℝ
  vacation_fund_percent : ℝ
  savings_percent : ℝ
  socializing_percent : ℝ
  remaining_amount : ℝ

/-- Theorem stating Jill's net monthly salary given her financial conditions --/
theorem jill_net_salary (j : JillFinances) 
  (h1 : j.discretionary_income = j.net_salary / 5)
  (h2 : j.vacation_fund_percent = 0.3)
  (h3 : j.savings_percent = 0.2)
  (h4 : j.socializing_percent = 0.35)
  (h5 : j.remaining_amount = 108)
  (h6 : (1 - (j.vacation_fund_percent + j.savings_percent + j.socializing_percent)) * j.discretionary_income = j.remaining_amount) :
  j.net_salary = 3600 := by
  sorry

#check jill_net_salary

end NUMINAMATH_CALUDE_jill_net_salary_l3311_331127


namespace NUMINAMATH_CALUDE_min_value_problem_l3311_331155

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (y : ℝ), y = (a + 1 / (2015 * a)) * (b + 1 / (2015 * b)) ∧
    y ≥ (2 * Real.sqrt 2016 - 2) / 2015 ∧
    (∀ (z : ℝ), z = (a + 1 / (2015 * a)) * (b + 1 / (2015 * b)) → z ≥ y) := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l3311_331155


namespace NUMINAMATH_CALUDE_complex_absolute_value_l3311_331195

theorem complex_absolute_value (z : ℂ) : z = 2 / (1 - Complex.I * Real.sqrt 3) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l3311_331195
