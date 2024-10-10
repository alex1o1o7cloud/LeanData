import Mathlib

namespace red_pens_count_l1205_120545

theorem red_pens_count (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h1 : total = 240)
  (h2 : red + blue = total)
  (h3 : blue = red - 2) : 
  red = 121 := by
  sorry

end red_pens_count_l1205_120545


namespace equation_solution_l1205_120500

theorem equation_solution : ∃ x : ℝ, (x / (2 * x - 3) + 5 / (3 - 2 * x) = 4) ∧ (x = 1) := by
  sorry

end equation_solution_l1205_120500


namespace greater_solution_of_quadratic_l1205_120554

theorem greater_solution_of_quadratic (x : ℝ) : 
  x^2 + 20*x - 96 = 0 → x ≤ 4 :=
sorry

end greater_solution_of_quadratic_l1205_120554


namespace smallest_n_mod_congruence_l1205_120591

theorem smallest_n_mod_congruence :
  ∃ (n : ℕ), n > 0 ∧ (17 * n) % 7 = 1234 % 7 ∧
  ∀ (m : ℕ), m > 0 ∧ (17 * m) % 7 = 1234 % 7 → n ≤ m :=
by sorry

end smallest_n_mod_congruence_l1205_120591


namespace twenty_bulb_series_string_possibilities_l1205_120599

/-- Represents a string of decorative lights -/
structure LightString where
  num_bulbs : ℕ
  is_series : Bool

/-- Calculates the number of ways a light string can be non-functioning -/
def non_functioning_possibilities (ls : LightString) : ℕ :=
  if ls.is_series then 2^ls.num_bulbs - 1 else 0

/-- Theorem stating the number of non-functioning possibilities for a specific light string -/
theorem twenty_bulb_series_string_possibilities :
  ∃ (ls : LightString), ls.num_bulbs = 20 ∧ ls.is_series = true ∧ non_functioning_possibilities ls = 2^20 - 1 :=
sorry

end twenty_bulb_series_string_possibilities_l1205_120599


namespace lg_difference_equals_two_l1205_120551

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_difference_equals_two : lg 25 - lg (1/4) = 2 := by
  sorry

end lg_difference_equals_two_l1205_120551


namespace one_third_blue_faces_iff_three_l1205_120524

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  sideLength : n > 0

/-- The total number of faces of all unit cubes when a cube of side length n is cut into n^3 unit cubes -/
def totalFaces (c : Cube n) : ℕ := 6 * n^3

/-- The number of blue faces when a cube of side length n is painted on all sides and cut into n^3 unit cubes -/
def blueFaces (c : Cube n) : ℕ := 6 * n^2

/-- The theorem stating that exactly one-third of the faces are blue if and only if n = 3 -/
theorem one_third_blue_faces_iff_three (c : Cube n) :
  3 * blueFaces c = totalFaces c ↔ n = 3 :=
sorry

end one_third_blue_faces_iff_three_l1205_120524


namespace sum_103_odd_numbers_from_63_l1205_120590

/-- The sum of the first n odd numbers starting from a given odd number -/
def sumOddNumbers (start : ℕ) (n : ℕ) : ℕ :=
  n * (2 * start + n - 1)

/-- Theorem: The sum of the first 103 odd numbers starting from 63 is 17015 -/
theorem sum_103_odd_numbers_from_63 :
  sumOddNumbers 63 103 = 17015 := by
  sorry

end sum_103_odd_numbers_from_63_l1205_120590


namespace translated_segment_endpoint_l1205_120552

/-- Given a segment AB with endpoints A(-4, -1) and B(1, 1), when translated to segment A'B' where A' has coordinates (-2, 2), prove that the coordinates of B' are (3, 4). -/
theorem translated_segment_endpoint (A B A' B' : ℝ × ℝ) : 
  A = (-4, -1) → 
  B = (1, 1) → 
  A' = (-2, 2) → 
  (A'.1 - A.1 = B'.1 - B.1 ∧ A'.2 - A.2 = B'.2 - B.2) → 
  B' = (3, 4) := by
  sorry

end translated_segment_endpoint_l1205_120552


namespace production_days_l1205_120535

/-- Given the average daily production for n days and the effect of adding one more day's production,
    prove the value of n. -/
theorem production_days (n : ℕ) : 
  (∀ (P : ℕ), P / n = 60 → (P + 90) / (n + 1) = 65) → n = 5 := by
  sorry

end production_days_l1205_120535


namespace exact_power_pair_l1205_120548

theorem exact_power_pair : 
  ∀ (a b : ℕ), 
  (∀ (n : ℕ), ∃ (c : ℕ), a^n + b^n = c^(n+1)) → 
  (a = 2 ∧ b = 2) := by
sorry

end exact_power_pair_l1205_120548


namespace number_of_possible_a_values_l1205_120547

theorem number_of_possible_a_values : ∃ (S : Finset ℕ),
  (∀ a ∈ S, ∃ b c d : ℕ,
    a > b ∧ b > c ∧ c > d ∧
    a + b + c + d = 2060 ∧
    a^2 - b^2 + c^2 - d^2 = 1987) ∧
  (∀ a : ℕ, (∃ b c d : ℕ,
    a > b ∧ b > c ∧ c > d ∧
    a + b + c + d = 2060 ∧
    a^2 - b^2 + c^2 - d^2 = 1987) → a ∈ S) ∧
  Finset.card S = 513 :=
sorry

end number_of_possible_a_values_l1205_120547


namespace wades_tips_per_customer_l1205_120572

/-- Wade's tips per customer calculation -/
theorem wades_tips_per_customer :
  ∀ (tips_per_customer : ℚ),
  (28 : ℚ) * tips_per_customer +  -- Friday tips
  (3 * 28 : ℚ) * tips_per_customer +  -- Saturday tips (3 times Friday)
  (36 : ℚ) * tips_per_customer =  -- Sunday tips
  (296 : ℚ) →  -- Total tips
  tips_per_customer = 2 := by
sorry

end wades_tips_per_customer_l1205_120572


namespace work_completion_time_l1205_120585

theorem work_completion_time (b_completion_time b_work_days a_remaining_time : ℕ) 
  (hb : b_completion_time = 15)
  (hbw : b_work_days = 10)
  (ha : a_remaining_time = 7) : 
  ∃ (a_completion_time : ℕ), a_completion_time = 21 ∧
  (a_completion_time : ℚ)⁻¹ * a_remaining_time = 1 - (b_work_days : ℚ) / b_completion_time :=
by sorry

end work_completion_time_l1205_120585


namespace max_M_value_l1205_120502

/-- Given a system of equations and conditions, prove the maximum value of M --/
theorem max_M_value (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (heq1 : x - 2*y = z - 2*u) (heq2 : 2*y*z = u*x) (hyz : z ≥ y) :
  ∃ (M : ℝ), M > 0 ∧ M ≤ z/y ∧ ∀ (N : ℝ), (N > 0 ∧ N ≤ z/y → N ≤ 6 + 4*Real.sqrt 2) :=
by sorry

end max_M_value_l1205_120502


namespace min_t_value_fixed_point_BD_l1205_120567

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the triangle area function
def triangle_area (t angle_AOB : ℝ) : Prop := t * Real.tan angle_AOB > 0

-- Theorem for minimum value of t
theorem min_t_value (a : ℝ) : 
  ∃ (t : ℝ), triangle_area t (Real.arctan ((4*a)/(a^2 - 4))) ∧ 
  t ≥ -2 ∧ 
  (t = -2 ↔ a = 2) := 
sorry

-- Theorem for fixed point of line BD when a = -1
theorem fixed_point_BD (x y : ℝ) : 
  parabola_C x y → 
  ∃ (x' y' : ℝ), parabola_C x' (-y') ∧ 
  (y - y' = (4 / (y' + y)) * (x - x'^2/4)) → 
  x = 1 ∧ y = 0 := 
sorry

end min_t_value_fixed_point_BD_l1205_120567


namespace binomial_sum_l1205_120515

theorem binomial_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) : 
  (∀ x : ℝ, (3 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 233 := by
  sorry

end binomial_sum_l1205_120515


namespace max_students_for_equal_distribution_l1205_120526

theorem max_students_for_equal_distribution (pens pencils : ℕ) 
  (h1 : pens = 1001) (h2 : pencils = 910) : 
  Nat.gcd pens pencils = 91 := by
sorry

end max_students_for_equal_distribution_l1205_120526


namespace sum_perfect_square_l1205_120507

theorem sum_perfect_square (K M : ℕ) : 
  K > 0 → M < 100 → K * (K + 1) = M^2 → (K = 8 ∨ K = 35) := by
  sorry

end sum_perfect_square_l1205_120507


namespace perfect_square_binomial_l1205_120543

theorem perfect_square_binomial (x : ℝ) : 
  ∃ (a b : ℝ), 16 * x^2 - 40 * x + 25 = (a * x + b)^2 := by
sorry

end perfect_square_binomial_l1205_120543


namespace ratio_problem_l1205_120520

theorem ratio_problem (q r s t u : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 3 / 4)
  (h4 : u / q = 1 / 2) :
  t / u = 16 / 9 := by
  sorry

end ratio_problem_l1205_120520


namespace mans_speed_with_current_is_15_l1205_120568

/-- Given a man's speed against a current and the speed of the current,
    calculate the man's speed with the current. -/
def mans_speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem stating that given the specific conditions,
    the man's speed with the current is 15 km/hr. -/
theorem mans_speed_with_current_is_15
  (speed_against_current : ℝ)
  (current_speed : ℝ)
  (h1 : speed_against_current = 8.6)
  (h2 : current_speed = 3.2) :
  mans_speed_with_current speed_against_current current_speed = 15 := by
  sorry

#eval mans_speed_with_current 8.6 3.2

end mans_speed_with_current_is_15_l1205_120568


namespace intersection_equals_interval_l1205_120549

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 2^x}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - 2*x)}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- Define the open-closed interval (0, 1/2]
def open_closed_interval : Set ℝ := {x | 0 < x ∧ x ≤ 1/2}

-- Theorem statement
theorem intersection_equals_interval : M_intersect_N = open_closed_interval := by
  sorry

end intersection_equals_interval_l1205_120549


namespace target_hit_probability_l1205_120556

theorem target_hit_probability (p1 p2 : ℝ) (h1 : p1 = 1/2) (h2 : p2 = 1/3) :
  1 - (1 - p1) * (1 - p2) = 2/3 := by
  sorry

end target_hit_probability_l1205_120556


namespace water_evaporation_rate_l1205_120532

/-- Calculates the daily evaporation rate of water in a glass -/
theorem water_evaporation_rate 
  (initial_amount : ℝ) 
  (evaporation_period : ℕ) 
  (evaporation_percentage : ℝ) :
  initial_amount = 12 →
  evaporation_period = 22 →
  evaporation_percentage = 5.5 →
  (initial_amount * evaporation_percentage / 100) / evaporation_period = 0.03 :=
by
  sorry

end water_evaporation_rate_l1205_120532


namespace isosceles_triangle_most_stable_l1205_120542

-- Define the shapes
inductive Shape
  | RegularPentagon
  | Square
  | Trapezoid
  | IsoscelesTriangle

-- Define the stability property
def is_stable (s : Shape) : Prop :=
  match s with
  | Shape.RegularPentagon => false
  | Shape.Square => false
  | Shape.Trapezoid => false
  | Shape.IsoscelesTriangle => true

-- Theorem statement
theorem isosceles_triangle_most_stable :
  ∀ s : Shape, is_stable s → s = Shape.IsoscelesTriangle :=
by sorry

end isosceles_triangle_most_stable_l1205_120542


namespace simplify_expression_l1205_120506

theorem simplify_expression (x : ℝ) : (3*x)^4 - (2*x)*(x^3) = 79*x^4 := by
  sorry

end simplify_expression_l1205_120506


namespace susan_age_indeterminate_l1205_120511

/-- Represents a person's age at different points in time -/
structure PersonAge where
  current : ℕ
  eightYearsAgo : ℕ
  inFifteenYears : ℕ

/-- The given conditions of the problem -/
axiom james : PersonAge
axiom janet : PersonAge
axiom susan : ℕ → Prop

axiom james_age_condition : james.inFifteenYears = 37
axiom james_janet_age_relation : james.eightYearsAgo = 2 * janet.eightYearsAgo
axiom susan_birth_condition : ∃ (age : ℕ), susan age

/-- The statement that Susan's age in 5 years cannot be determined -/
theorem susan_age_indeterminate : ¬∃ (age : ℕ), ∀ (current_age : ℕ), susan current_age → current_age + 5 = age := by
  sorry

end susan_age_indeterminate_l1205_120511


namespace unique_square_board_state_l1205_120587

/-- Represents the state of numbers on the board -/
def BoardState := List Nat

/-- The process of replacing a number with its proper divisors -/
def replace_with_divisors (a : Nat) : BoardState :=
  sorry

/-- The full process of repeatedly replacing numbers until no more replacements are possible -/
def process (initial : BoardState) : BoardState :=
  sorry

/-- Theorem: The only natural number N for which the described process
    can result in exactly N^2 numbers on the board is 1 -/
theorem unique_square_board_state (N : Nat) :
  (∃ (final : BoardState), process [N] = final ∧ final.length = N^2) ↔ N = 1 :=
sorry

end unique_square_board_state_l1205_120587


namespace prime_factor_count_l1205_120558

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_factor_count (x : ℕ) : 
  is_prime x → 
  (∃ (n : ℕ), 2^22 * x^7 * 11^2 = n ∧ (Nat.factors n).length = 31) → 
  x = 7 :=
sorry

end prime_factor_count_l1205_120558


namespace sin_cos_sixth_power_l1205_120583

theorem sin_cos_sixth_power (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 2) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 13 / 16 := by
  sorry

end sin_cos_sixth_power_l1205_120583


namespace five_objects_two_groups_l1205_120573

/-- The number of ways to partition n indistinguishable objects into k indistinguishable groups -/
def partition_count (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 3 ways to partition 5 indistinguishable objects into 2 indistinguishable groups -/
theorem five_objects_two_groups : partition_count 5 2 = 3 := by sorry

end five_objects_two_groups_l1205_120573


namespace anne_had_fifteen_sweettarts_l1205_120519

/-- The number of Sweettarts Anne had initially -/
def annes_initial_sweettarts (num_friends : ℕ) (sweettarts_per_friend : ℕ) : ℕ :=
  num_friends * sweettarts_per_friend

/-- Theorem stating that Anne had 15 Sweettarts initially -/
theorem anne_had_fifteen_sweettarts :
  annes_initial_sweettarts 3 5 = 15 := by
  sorry

end anne_had_fifteen_sweettarts_l1205_120519


namespace point_movement_l1205_120509

/-- Given three points A, B, and C on a number line, where:
    - B is 4 units to the right of A
    - C is 2 units to the left of B
    - C represents the number -3
    Prove that A represents the number -5 -/
theorem point_movement (A B C : ℝ) 
  (h1 : B = A + 4)
  (h2 : C = B - 2)
  (h3 : C = -3) :
  A = -5 := by sorry

end point_movement_l1205_120509


namespace symmetry_of_curves_l1205_120523

/-- The original curve E -/
def E (x y : ℝ) : Prop := x^2 + 2*x*y + y^2 + 3*x + y = 0

/-- The line of symmetry l -/
def l (x y : ℝ) : Prop := 2*x - y - 1 = 0

/-- The symmetric curve E' -/
def E' (x y : ℝ) : Prop := x^2 + 14*x*y + 49*y^2 - 21*x + 103*y + 54 = 0

/-- Theorem stating that E' is symmetric to E with respect to l -/
theorem symmetry_of_curves :
  ∀ (x y x' y' : ℝ),
    E x y →
    l ((x + x') / 2) ((y + y') / 2) →
    E' x' y' :=
sorry

end symmetry_of_curves_l1205_120523


namespace max_sum_of_sides_l1205_120570

variable (A B C a b c : ℝ)

-- Define the triangle ABC
def is_triangle (A B C a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the given condition
def given_condition (A B C a b c : ℝ) : Prop :=
  (2 * a - c) / b = Real.cos C / Real.cos B

-- Theorem statement
theorem max_sum_of_sides 
  (h_triangle : is_triangle A B C a b c)
  (h_condition : given_condition A B C a b c)
  (h_b : b = 4) :
  ∃ (max : ℝ), max = 8 ∧ a + c ≤ max :=
sorry

end max_sum_of_sides_l1205_120570


namespace shaded_area_proof_l1205_120580

theorem shaded_area_proof (side_length : ℝ) (circle_radius1 circle_radius2 circle_radius3 : ℝ) :
  side_length = 30 ∧ 
  circle_radius1 = 5 ∧ 
  circle_radius2 = 4 ∧ 
  circle_radius3 = 3 →
  (side_length^2 / 9) * 5 = 500 := by
sorry

end shaded_area_proof_l1205_120580


namespace average_height_is_141_l1205_120522

def student_heights : List ℝ := [145, 142, 138, 136, 143, 146, 138, 144, 137, 141]

theorem average_height_is_141 :
  (student_heights.sum / student_heights.length : ℝ) = 141 := by
  sorry

end average_height_is_141_l1205_120522


namespace complex_modulus_problem_l1205_120577

theorem complex_modulus_problem (a : ℝ) (h1 : a > 0) :
  Complex.abs ((a - Complex.I) / Complex.I) = 2 → a = Real.sqrt 3 := by
  sorry

end complex_modulus_problem_l1205_120577


namespace integer_pairs_satisfying_equation_l1205_120541

theorem integer_pairs_satisfying_equation :
  ∀ (x y : ℤ), x^2 = y^2 + 2*y + 13 ↔ (x = 4 ∧ y = 1) ∨ (x = -4 ∧ y = -5) := by
  sorry

end integer_pairs_satisfying_equation_l1205_120541


namespace face_mask_profit_l1205_120539

/-- Calculates the total profit from selling face masks given the specified conditions. -/
theorem face_mask_profit :
  let num_boxes : ℕ := 3
  let discount_rate : ℚ := 1/5
  let original_price : ℚ := 8
  let masks_per_box : List ℕ := [25, 30, 35]
  let selling_price : ℚ := 3/5

  let discounted_price := original_price * (1 - discount_rate)
  let total_cost := num_boxes * discounted_price
  let total_masks := masks_per_box.sum
  let total_revenue := total_masks * selling_price
  let profit := total_revenue - total_cost

  profit = 348/10 :=
by sorry

end face_mask_profit_l1205_120539


namespace larger_number_problem_l1205_120588

theorem larger_number_problem (x y : ℕ) 
  (h1 : x * y = 40)
  (h2 : x + y = 13)
  (h3 : Even x ∨ Even y) :
  max x y = 8 := by
sorry

end larger_number_problem_l1205_120588


namespace congruence_solution_l1205_120594

theorem congruence_solution (x : ℤ) 
  (h1 : (2 + x) % (2^4) = 3^2 % (2^4))
  (h2 : (3 + x) % (3^4) = 2^3 % (3^4))
  (h3 : (4 + x) % (2^3) = 3^3 % (2^3)) :
  x % 24 = 23 := by sorry

end congruence_solution_l1205_120594


namespace imaginary_part_of_z_l1205_120569

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = 1 + Complex.I) :
  Complex.im z = -1 := by sorry

end imaginary_part_of_z_l1205_120569


namespace slope_implies_y_coordinate_l1205_120518

/-- Given two points P and Q in a coordinate plane, if the slope of the line through P and Q is -3/2, then the y-coordinate of Q is -2. -/
theorem slope_implies_y_coordinate (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ = -2 →
  y₁ = 7 →
  x₂ = 4 →
  (y₂ - y₁) / (x₂ - x₁) = -3/2 →
  y₂ = -2 :=
by sorry

end slope_implies_y_coordinate_l1205_120518


namespace circle_cut_and_reform_l1205_120538

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a point inside a circle
def PointInside (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 < c.radius^2

-- Define the theorem
theorem circle_cut_and_reform (c : Circle) (a : ℝ × ℝ) (h : PointInside c a) :
  ∃ (part1 part2 : Set (ℝ × ℝ)), 
    (part1 ∪ part2 = {p | PointInside c p}) ∧
    (∃ (new_circle : Circle), new_circle.center = a ∧
      part1 ∪ part2 = {p | PointInside new_circle p}) :=
sorry

end circle_cut_and_reform_l1205_120538


namespace correct_det_calculation_l1205_120564

/-- Definition of 2x2 determinant -/
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

/-- Polynomial M -/
def M (m n : ℝ) : ℝ := m^2 - 2*m*n

/-- Polynomial N -/
def N (m n : ℝ) : ℝ := 3*m^2 - m*n

/-- Theorem: The correct calculation of |M N; 1 3| equals -5mn -/
theorem correct_det_calculation (m n : ℝ) :
  det2x2 (M m n) (N m n) 1 3 = -5*m*n := by
  sorry

end correct_det_calculation_l1205_120564


namespace subtraction_problem_sum_l1205_120586

theorem subtraction_problem_sum (K L M N : ℕ) : 
  K < 10 → L < 10 → M < 10 → N < 10 →
  6000 + 100 * K + L - (900 + N) = 2011 →
  K + L + M + N = 17 := by
sorry

end subtraction_problem_sum_l1205_120586


namespace diagonals_concurrent_l1205_120503

-- Define a regular 12-gon
def Regular12gon (P : Fin 12 → ℝ × ℝ) : Prop :=
  ∀ i j : Fin 12, dist (P i) (P ((i + 1) % 12)) = dist (P j) (P ((j + 1) % 12))

-- Define a diagonal in the 12-gon
def Diagonal (P : Fin 12 → ℝ × ℝ) (i j : Fin 12) : Set (ℝ × ℝ) :=
  {x | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = (1 - t) • (P i) + t • (P j)}

-- Define concurrency of three lines
def Concurrent (L₁ L₂ L₃ : Set (ℝ × ℝ)) : Prop :=
  ∃ x : ℝ × ℝ, x ∈ L₁ ∧ x ∈ L₂ ∧ x ∈ L₃

-- Theorem statement
theorem diagonals_concurrent (P : Fin 12 → ℝ × ℝ) (h : Regular12gon P) :
  Concurrent (Diagonal P 0 8) (Diagonal P 11 3) (Diagonal P 1 10) :=
sorry

end diagonals_concurrent_l1205_120503


namespace quadratic_solution_l1205_120508

theorem quadratic_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : ∀ x : ℝ, x^2 + 2*a*x + b = 0 ↔ x = a ∨ x = b) :
  a = 1 ∧ b = -3 := by
  sorry

end quadratic_solution_l1205_120508


namespace min_socks_for_fifteen_pairs_l1205_120537

/-- Represents the number of socks of each color in the room -/
structure SockCollection where
  red : Nat
  green : Nat
  blue : Nat
  yellow : Nat
  black : Nat

/-- The minimum number of socks needed to guarantee a certain number of pairs -/
def minSocksForPairs (socks : SockCollection) (pairs : Nat) : Nat :=
  5 + 5 * 2 * (pairs - 1) + 1

/-- Theorem stating the minimum number of socks needed for 15 pairs -/
theorem min_socks_for_fifteen_pairs (socks : SockCollection)
    (h1 : socks.red = 120)
    (h2 : socks.green = 100)
    (h3 : socks.blue = 70)
    (h4 : socks.yellow = 50)
    (h5 : socks.black = 30) :
    minSocksForPairs socks 15 = 146 := by
  sorry

#eval minSocksForPairs { red := 120, green := 100, blue := 70, yellow := 50, black := 30 } 15

end min_socks_for_fifteen_pairs_l1205_120537


namespace area_between_circles_l1205_120597

/-- Given two concentric circles where the outer radius is twice the inner radius
    and the width between circles is 3, prove the area between circles is 27π -/
theorem area_between_circles (r : ℝ) (h1 : r > 0) (h2 : 2 * r - r = 3) :
  π * (2 * r)^2 - π * r^2 = 27 * π := by
  sorry

end area_between_circles_l1205_120597


namespace jana_height_l1205_120513

theorem jana_height (jess_height kelly_height jana_height : ℕ) : 
  jess_height = 72 →
  kelly_height = jess_height - 3 →
  jana_height = kelly_height + 5 →
  jana_height = 74 := by
  sorry

end jana_height_l1205_120513


namespace action_figure_fraction_l1205_120512

theorem action_figure_fraction (total_toys dolls : ℕ) : 
  total_toys = 24 → 
  dolls = 18 → 
  (total_toys - dolls : ℚ) / total_toys = 1 / 4 := by
  sorry

end action_figure_fraction_l1205_120512


namespace min_max_sum_reciprocals_l1205_120527

open Real

theorem min_max_sum_reciprocals (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 3) :
  let f := (1 / (x + y) + 1 / (x + z) + 1 / (y + z))
  ∃ (min_val : ℝ), (∀ a b c, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3 →
    (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≥ min_val) ∧
  min_val = (3 / 2) ∧
  ¬∃ (max_val : ℝ), ∀ a b c, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3 →
    (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≤ max_val :=
by sorry

end min_max_sum_reciprocals_l1205_120527


namespace fraction_evaluation_l1205_120516

theorem fraction_evaluation : 
  (⌈(21 / 8 : ℚ) - ⌈(35 / 21 : ℚ)⌉⌉ : ℚ) / 
  (⌈(35 / 8 : ℚ) + ⌈(8 * 21 / 35 : ℚ)⌉⌉ : ℚ) = 1 / 10 := by
  sorry

end fraction_evaluation_l1205_120516


namespace unique_total_prices_l1205_120550

def gift_prices : List ℕ := [2, 5, 8, 11, 14]
def box_prices : List ℕ := [3, 5, 7, 9, 11]

def total_prices : List ℕ :=
  List.eraseDups (List.map (λ (p : ℕ × ℕ) => p.1 + p.2) (List.product gift_prices box_prices))

theorem unique_total_prices :
  total_prices.length = 19 := by sorry

end unique_total_prices_l1205_120550


namespace missing_fraction_sum_l1205_120557

theorem missing_fraction_sum (sum : ℚ) (f1 f2 f3 f4 f5 f6 : ℚ) :
  sum = 45/100 →
  f1 = 1/3 →
  f2 = 1/2 →
  f3 = -5/6 →
  f4 = 1/5 →
  f5 = -9/20 →
  f6 = -9/20 →
  ∃ x : ℚ, x = 23/20 ∧ sum = f1 + f2 + f3 + f4 + f5 + f6 + x :=
by sorry

end missing_fraction_sum_l1205_120557


namespace sum_in_base5_l1205_120521

-- Define a function to convert from base 10 to base 5
def toBase5 (n : ℕ) : List ℕ := sorry

-- Define a function to interpret a list of digits as a number in base 5
def fromBase5 (digits : List ℕ) : ℕ := sorry

theorem sum_in_base5 : 
  toBase5 (12 + 47) = [2, 1, 4] := by sorry

end sum_in_base5_l1205_120521


namespace no_valid_solution_l1205_120596

theorem no_valid_solution : ¬∃ (x y z : ℕ+), 
  (x * y * z = 4 * (x + y + z)) ∧ 
  (x * y = z + x) ∧ 
  ∃ (k : ℕ+), x * y * z = k * k :=
by sorry

end no_valid_solution_l1205_120596


namespace ellipse_major_axis_length_l1205_120530

/-- The length of the major axis of the ellipse 2x^2 + y^2 = 8 is 4√2 -/
theorem ellipse_major_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | 2 * x^2 + y^2 = 8}
  ∃ a b : ℝ, a > b ∧ a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), (x, y) ∈ ellipse ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
    2 * a = 4 * Real.sqrt 2 :=
by sorry

end ellipse_major_axis_length_l1205_120530


namespace sufficient_not_necessary_implies_necessary_not_sufficient_l1205_120566

theorem sufficient_not_necessary_implies_necessary_not_sufficient 
  (p q : Prop) (h : (p → q) ∧ ¬(q → p)) : 
  (¬q → ¬p) ∧ ¬(¬p → ¬q) := by
  sorry

end sufficient_not_necessary_implies_necessary_not_sufficient_l1205_120566


namespace solve_sqrt_equation_l1205_120528

theorem solve_sqrt_equation (x : ℝ) (h : x > 0) :
  Real.sqrt ((3 / x) + 3) = 2 → x = 3 := by
  sorry

end solve_sqrt_equation_l1205_120528


namespace soda_difference_l1205_120575

def julio_orange : ℕ := 4
def julio_grape : ℕ := 7
def mateo_orange : ℕ := 1
def mateo_grape : ℕ := 3
def sophia_orange : ℕ := 6
def sophia_strawberry : ℕ := 5

def orange_soda_volume : ℚ := 2
def grape_soda_volume : ℚ := 2
def sophia_orange_volume : ℚ := 1.5
def sophia_strawberry_volume : ℚ := 2.5

def julio_total : ℚ := julio_orange * orange_soda_volume + julio_grape * grape_soda_volume
def mateo_total : ℚ := mateo_orange * orange_soda_volume + mateo_grape * grape_soda_volume
def sophia_total : ℚ := sophia_orange * sophia_orange_volume + sophia_strawberry * sophia_strawberry_volume

theorem soda_difference :
  (max julio_total (max mateo_total sophia_total)) - (min julio_total (min mateo_total sophia_total)) = 14 := by
  sorry

end soda_difference_l1205_120575


namespace course_selection_problem_l1205_120565

theorem course_selection_problem (n : ℕ) (k : ℕ) (m : ℕ) : 
  n = 6 → k = 3 → m = 1 → 
  (n.choose m) * ((n - m).choose (k - m)) * ((n - k).choose (k - m)) = 180 := by
  sorry

end course_selection_problem_l1205_120565


namespace number_of_elements_in_set_l1205_120534

theorem number_of_elements_in_set (initial_avg : ℚ) (incorrect_num : ℚ) (correct_num : ℚ) (correct_avg : ℚ) (n : ℕ) : 
  initial_avg = 18 →
  incorrect_num = 26 →
  correct_num = 66 →
  correct_avg = 22 →
  n * initial_avg + (correct_num - incorrect_num) = n * correct_avg →
  n = 10 := by
sorry

end number_of_elements_in_set_l1205_120534


namespace balloon_height_per_ounce_l1205_120533

/-- Calculates the height increase per ounce of helium for a balloon flight --/
theorem balloon_height_per_ounce 
  (total_money : ℚ)
  (sheet_cost : ℚ)
  (rope_cost : ℚ)
  (propane_cost : ℚ)
  (helium_price_per_ounce : ℚ)
  (max_height : ℚ)
  (h1 : total_money = 200)
  (h2 : sheet_cost = 42)
  (h3 : rope_cost = 18)
  (h4 : propane_cost = 14)
  (h5 : helium_price_per_ounce = 3/2)
  (h6 : max_height = 9492) :
  (max_height / ((total_money - (sheet_cost + rope_cost + propane_cost)) / helium_price_per_ounce)) = 113 := by
  sorry

end balloon_height_per_ounce_l1205_120533


namespace distinct_prime_factors_count_l1205_120536

def n : ℕ := 97 * 101 * 104 * 107 * 109

theorem distinct_prime_factors_count : Nat.card (Nat.factors n).toFinset = 6 := by
  sorry

end distinct_prime_factors_count_l1205_120536


namespace job_completion_time_l1205_120510

/-- Given two people P and Q working on a job, this theorem proves the time
    it takes P to complete the job alone, given the time it takes Q alone
    and the time it takes them working together. -/
theorem job_completion_time
  (time_Q : ℝ)
  (time_PQ : ℝ)
  (h1 : time_Q = 6)
  (h2 : time_PQ = 2.4) :
  ∃ (time_P : ℝ), time_P = 4 ∧ 1 / time_P + 1 / time_Q = 1 / time_PQ :=
by sorry

end job_completion_time_l1205_120510


namespace tree_distance_l1205_120546

/-- Given 6 equally spaced trees along a straight road, where the distance between
    the first and fourth tree is 60 feet, the distance between the first and last
    tree is 100 feet. -/
theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 6) (h2 : d = 60) :
  (n - 1) * d / 3 = 100 := by
  sorry

end tree_distance_l1205_120546


namespace A_intersect_B_l1205_120505

def A : Set ℤ := {1, 2, 3, 4}

def B : Set ℤ := {y | ∃ x ∈ A, y = 3 * x - 2}

theorem A_intersect_B : A ∩ B = {1, 4} := by sorry

end A_intersect_B_l1205_120505


namespace shells_per_friend_l1205_120584

/-- Given the number of shells collected by Jillian, Savannah, and Clayton,
    and the number of friends to distribute the shells to,
    prove that each friend receives 27 shells. -/
theorem shells_per_friend
  (jillian_shells : ℕ)
  (savannah_shells : ℕ)
  (clayton_shells : ℕ)
  (num_friends : ℕ)
  (h1 : jillian_shells = 29)
  (h2 : savannah_shells = 17)
  (h3 : clayton_shells = 8)
  (h4 : num_friends = 2) :
  (jillian_shells + savannah_shells + clayton_shells) / num_friends = 27 :=
by
  sorry

#check shells_per_friend

end shells_per_friend_l1205_120584


namespace product_of_polynomials_l1205_120562

theorem product_of_polynomials (g h : ℚ) : 
  (∀ d : ℚ, (7 * d^2 - 4 * d + g) * (3 * d^2 + h * d - 9) = 
    21 * d^4 - 49 * d^3 - 44 * d^2 + 17 * d - 24) → 
  g + h = -107/24 := by sorry

end product_of_polynomials_l1205_120562


namespace interest_rate_is_twelve_percent_l1205_120555

/-- Calculates the simple interest rate given the principal, interest, and time. -/
def calculate_interest_rate (principal interest : ℕ) (time : ℕ) : ℚ :=
  (interest * 100 : ℚ) / (principal * time)

/-- Proves that the interest rate is 12% given the specified conditions. -/
theorem interest_rate_is_twelve_percent 
  (principal : ℕ) 
  (interest : ℕ) 
  (time : ℕ) 
  (h1 : principal = 875)
  (h2 : interest = 2100)
  (h3 : time = 20) :
  calculate_interest_rate principal interest time = 12 := by
  sorry

#eval calculate_interest_rate 875 2100 20

end interest_rate_is_twelve_percent_l1205_120555


namespace five_students_three_colleges_l1205_120574

/-- The number of ways for students to apply to colleges -/
def applicationWays (numStudents : ℕ) (numColleges : ℕ) : ℕ :=
  numColleges ^ numStudents

/-- Theorem: 5 students applying to 3 colleges results in 3^5 different ways -/
theorem five_students_three_colleges : 
  applicationWays 5 3 = 3^5 := by
  sorry

end five_students_three_colleges_l1205_120574


namespace total_candies_l1205_120571

/-- Represents the number of candies Lillian has initially -/
def initial_candies : ℕ := 88

/-- Represents the number of candies Lillian receives from her father -/
def additional_candies : ℕ := 5

/-- Theorem stating the total number of candies Lillian has after receiving more -/
theorem total_candies : initial_candies + additional_candies = 93 := by
  sorry

end total_candies_l1205_120571


namespace video_game_sales_theorem_l1205_120576

/-- Given a total number of video games, number of non-working games, and a price per working game,
    calculate the total money that can be earned by selling the working games. -/
def total_money_earned (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : ℕ :=
  (total_games - non_working_games) * price_per_game

/-- Theorem stating that given 10 total games, 8 non-working games, and a price of $6 per working game,
    the total money earned is $12. -/
theorem video_game_sales_theorem :
  total_money_earned 10 8 6 = 12 := by
  sorry

end video_game_sales_theorem_l1205_120576


namespace last_ten_digits_periodicity_l1205_120501

theorem last_ten_digits_periodicity (n : ℕ) (h : n ≥ 10) :
  2^n % 10^10 = 2^(n + 4 * 10^9) % 10^10 := by
  sorry

end last_ten_digits_periodicity_l1205_120501


namespace bridge_length_l1205_120563

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 130 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length = 245 := by
  sorry

end bridge_length_l1205_120563


namespace unique_valid_sequence_l1205_120517

def IsValidSequence (a : ℕ → ℕ) : Prop :=
  (∀ m n : ℕ, m ≠ n → a m ≠ a n) ∧
  (∀ n : ℕ, a n % a (a n) = 0)

theorem unique_valid_sequence :
  ∀ a : ℕ → ℕ, IsValidSequence a → (∀ n : ℕ, a n = n) :=
by sorry

end unique_valid_sequence_l1205_120517


namespace isosceles_triangle_perimeter_l1205_120561

-- Define an isosceles triangle with sides a, b, and c
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)
  validTriangle : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter of a triangle
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, 
  ((t.a = 8 ∧ t.b = 8 ∧ t.c = 4) ∨ (t.a = 8 ∧ t.b = 4 ∧ t.c = 8) ∨ (t.a = 4 ∧ t.b = 8 ∧ t.c = 8)) →
  perimeter t = 20 := by
  sorry


end isosceles_triangle_perimeter_l1205_120561


namespace exponent_operations_l1205_120504

theorem exponent_operations (a : ℝ) : 
  (a^4 * a^3 = a^7) ∧ 
  ((a^2)^3 ≠ a^5) ∧ 
  (3*a^2 - a^2 ≠ 2) ∧ 
  ((a - b)^2 ≠ a^2 - b^2) :=
sorry

end exponent_operations_l1205_120504


namespace max_discarded_grapes_l1205_120578

theorem max_discarded_grapes (n : ℕ) : ∃ (q : ℕ), n = 7 * q + 6 ∧ 
  ∀ (r : ℕ), r < 7 → n ≠ 7 * (q + 1) + r :=
by sorry

end max_discarded_grapes_l1205_120578


namespace maggie_bouncy_balls_l1205_120553

/-- The number of bouncy balls Maggie kept -/
def total_bouncy_balls : ℝ :=
  let yellow_packs : ℝ := 8.0
  let green_packs_given : ℝ := 4.0
  let green_packs_bought : ℝ := 4.0
  let balls_per_pack : ℝ := 10.0
  yellow_packs * balls_per_pack + (green_packs_bought - green_packs_given) * balls_per_pack

/-- Theorem stating that Maggie kept 80.0 bouncy balls -/
theorem maggie_bouncy_balls : total_bouncy_balls = 80.0 := by
  sorry

end maggie_bouncy_balls_l1205_120553


namespace triangle_condition_right_triangle_condition_l1205_120559

/-- Given vectors in 2D space -/
def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (5 - m, -(4 + m))

/-- Vector subtraction -/
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

/-- Dot product of 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Condition for three points to form a triangle -/
def forms_triangle (m : ℝ) : Prop :=
  let AB := vec_sub OB OA
  let AC := vec_sub (OC m) OA
  let BC := vec_sub (OC m) OB
  AB.1 / AB.2 ≠ AC.1 / AC.2 ∧ AB.1 / AB.2 ≠ BC.1 / BC.2 ∧ AC.1 / AC.2 ≠ BC.1 / BC.2

/-- Theorem: Condition for A, B, and C to form a triangle -/
theorem triangle_condition : ∀ m : ℝ, forms_triangle m ↔ m ≠ -1 := by sorry

/-- Theorem: Condition for ABC to be a right triangle with angle A as the right angle -/
theorem right_triangle_condition : 
  ∀ m : ℝ, dot_product (vec_sub OB OA) (vec_sub (OC m) OA) = 0 ↔ m = 3/2 := by sorry

end triangle_condition_right_triangle_condition_l1205_120559


namespace probability_not_face_card_l1205_120531

theorem probability_not_face_card (total_cards : ℕ) (red_cards : ℕ) (spades_cards : ℕ)
  (red_face_cards : ℕ) (spades_face_cards : ℕ) :
  total_cards = 52 →
  red_cards = 26 →
  spades_cards = 13 →
  red_face_cards = 6 →
  spades_face_cards = 3 →
  (red_cards + spades_cards - (red_face_cards + spades_face_cards)) / (red_cards + spades_cards) = 10 / 13 := by
  sorry

end probability_not_face_card_l1205_120531


namespace remainder_negation_l1205_120529

theorem remainder_negation (a : ℤ) : 
  (a % 1999 = 1) → ((-a) % 1999 = 1998) := by
  sorry

end remainder_negation_l1205_120529


namespace smallest_sum_of_squares_l1205_120540

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 221 → ∀ a b : ℕ, a^2 - b^2 = 221 → x^2 + y^2 ≤ a^2 + b^2 → x^2 + y^2 = 229 :=
sorry

end smallest_sum_of_squares_l1205_120540


namespace trig_system_solution_l1205_120598

theorem trig_system_solution (x y : ℝ) 
  (h1 : Real.tan x * Real.tan y = 1/6)
  (h2 : Real.sin x * Real.sin y = 1/(5 * Real.sqrt 2)) :
  Real.cos (x + y) = 1/Real.sqrt 2 ∧ 
  Real.cos (x - y) = 7/(5 * Real.sqrt 2) := by
sorry

end trig_system_solution_l1205_120598


namespace points_on_line_l1205_120525

-- Define the line y = -3x + b
def line (x : ℝ) (b : ℝ) : ℝ := -3 * x + b

-- Define the points
def point1 (y₁ : ℝ) (b : ℝ) : Prop := y₁ = line (-2) b
def point2 (y₂ : ℝ) (b : ℝ) : Prop := y₂ = line (-1) b
def point3 (y₃ : ℝ) (b : ℝ) : Prop := y₃ = line 1 b

-- Theorem statement
theorem points_on_line (y₁ y₂ y₃ b : ℝ) 
  (h1 : point1 y₁ b) (h2 : point2 y₂ b) (h3 : point3 y₃ b) :
  y₁ > y₂ ∧ y₂ > y₃ := by sorry

end points_on_line_l1205_120525


namespace power_of_power_l1205_120581

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l1205_120581


namespace volunteer_distribution_l1205_120514

def distribute_volunteers (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem volunteer_distribution :
  distribute_volunteers 7 3 = 6 :=
sorry

end volunteer_distribution_l1205_120514


namespace fixed_points_of_f_composition_l1205_120592

def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 2*x

theorem fixed_points_of_f_composition :
  ∀ x : ℝ, f (f x) = f x ↔ x = 0 ∨ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 := by
  sorry

end fixed_points_of_f_composition_l1205_120592


namespace pentagon_square_side_ratio_l1205_120593

/-- Given a regular pentagon and a square with the same perimeter of 20 inches,
    the ratio of the side length of the pentagon to the side length of the square is 4/5. -/
theorem pentagon_square_side_ratio :
  ∀ (p s : ℝ), 
    p > 0 → s > 0 →
    5 * p = 20 →  -- Perimeter of pentagon
    4 * s = 20 →  -- Perimeter of square
    p / s = 4 / 5 := by
  sorry

end pentagon_square_side_ratio_l1205_120593


namespace polly_tweets_l1205_120595

/-- Represents the tweet rate (tweets per minute) for each of Polly's activities -/
structure TweetRate where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Represents the duration (in minutes) of each of Polly's activities -/
structure ActivityDuration where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Calculates the total number of tweets given the tweet rates and activity durations -/
def totalTweets (rate : TweetRate) (duration : ActivityDuration) : ℕ :=
  rate.happy * duration.happy + rate.hungry * duration.hungry + rate.mirror * duration.mirror

/-- Theorem stating that given Polly's specific tweet rates and activity durations, 
    the total number of tweets is 1340 -/
theorem polly_tweets : 
  ∀ (rate : TweetRate) (duration : ActivityDuration),
  rate.happy = 18 ∧ rate.hungry = 4 ∧ rate.mirror = 45 ∧
  duration.happy = 20 ∧ duration.hungry = 20 ∧ duration.mirror = 20 →
  totalTweets rate duration = 1340 := by
sorry

end polly_tweets_l1205_120595


namespace min_value_problem_l1205_120579

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 6*x*y - 1 = 0) :
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 6*a*b - 1 = 0 → x + 2*y ≤ a + 2*b) ∧ 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + 6*a*b - 1 = 0 ∧ x + 2*y = a + 2*b) ∧
  x + 2*y = 2 * Real.sqrt 2 / 3 :=
sorry

end min_value_problem_l1205_120579


namespace bucket_capacity_reduction_l1205_120589

theorem bucket_capacity_reduction (current_buckets : ℕ) (reduction_factor : ℚ) : 
  current_buckets = 25 → 
  reduction_factor = 2 / 5 →
  ↑(Nat.ceil ((current_buckets : ℚ) / reduction_factor)) = 63 := by
  sorry

end bucket_capacity_reduction_l1205_120589


namespace statues_painted_l1205_120544

theorem statues_painted (total_paint : ℚ) (paint_per_statue : ℚ) :
  total_paint = 7/16 →
  paint_per_statue = 1/16 →
  (total_paint / paint_per_statue : ℚ) = 7 :=
by sorry

end statues_painted_l1205_120544


namespace salary_ratio_proof_l1205_120560

/-- Proves that the ratio of Shyam's monthly salary to Abhinav's monthly salary is 2:1 -/
theorem salary_ratio_proof (ram_salary shyam_salary abhinav_annual_salary : ℕ) : 
  ram_salary = 25600 →
  abhinav_annual_salary = 192000 →
  10 * ram_salary = 8 * shyam_salary →
  ∃ (k : ℕ), shyam_salary = k * (abhinav_annual_salary / 12) →
  shyam_salary / (abhinav_annual_salary / 12) = 2 := by
  sorry

end salary_ratio_proof_l1205_120560


namespace ad_space_width_l1205_120582

def ad_problem (num_spaces : ℕ) (length : ℝ) (cost_per_sqft : ℝ) (total_cost : ℝ) : Prop :=
  ∃ w : ℝ,
    w > 0 ∧
    num_spaces * length * w * cost_per_sqft = total_cost ∧
    w = 5

theorem ad_space_width :
  ad_problem 30 12 60 108000 :=
sorry

end ad_space_width_l1205_120582
