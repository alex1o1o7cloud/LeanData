import Mathlib

namespace pet_shop_kittens_l74_7453

theorem pet_shop_kittens (num_puppies : ℕ) (puppy_cost kitten_cost total_stock : ℚ) : 
  num_puppies = 2 →
  puppy_cost = 20 →
  kitten_cost = 15 →
  total_stock = 100 →
  (total_stock - num_puppies * puppy_cost) / kitten_cost = 4 :=
by sorry

end pet_shop_kittens_l74_7453


namespace total_handshakes_at_convention_l74_7428

def num_gremlins : ℕ := 30
def num_imps : ℕ := 25
def num_reconciled_imps : ℕ := 10
def num_unreconciled_imps : ℕ := 15

def handshakes_among_gremlins : ℕ := num_gremlins * (num_gremlins - 1) / 2
def handshakes_among_reconciled_imps : ℕ := num_reconciled_imps * (num_reconciled_imps - 1) / 2
def handshakes_between_gremlins_and_imps : ℕ := num_gremlins * num_imps

theorem total_handshakes_at_convention : 
  handshakes_among_gremlins + handshakes_among_reconciled_imps + handshakes_between_gremlins_and_imps = 1230 := by
  sorry

end total_handshakes_at_convention_l74_7428


namespace difference_of_reciprocals_l74_7452

theorem difference_of_reciprocals (p q : ℚ) 
  (hp : 4 / p = 8) (hq : 4 / q = 18) : p - q = 5 / 18 := by
  sorry

end difference_of_reciprocals_l74_7452


namespace function_properties_no_zeros_l74_7421

noncomputable section

def f (a : ℝ) (x : ℝ) := a * Real.log x - x
def g (a : ℝ) (x : ℝ) := a * Real.exp x - x

theorem function_properties (a : ℝ) (ha : a > 0) :
  (∀ x > 1, ∀ y > x, f a y < f a x) ∧
  (∃ x > 2, ∀ y > 2, g a x ≤ g a y) →
  a ∈ Set.Ioo 0 (1 / Real.exp 2) :=
sorry

theorem no_zeros (a : ℝ) (ha : a > 0) :
  (∀ x > 0, f a x ≠ 0) ∧ (∀ x, g a x ≠ 0) →
  a ∈ Set.Ioo (1 / Real.exp 1) (Real.exp 1) :=
sorry

end function_properties_no_zeros_l74_7421


namespace sqrt_sum_equality_l74_7414

theorem sqrt_sum_equality (x : ℝ) :
  Real.sqrt (x^2 - 2*x + 4) + Real.sqrt (x^2 + 2*x + 4) =
  Real.sqrt ((x-1)^2 + 3) + Real.sqrt ((x+1)^2 + 3) :=
by sorry

end sqrt_sum_equality_l74_7414


namespace tagged_fish_in_second_catch_l74_7449

theorem tagged_fish_in_second_catch 
  (total_fish : ℕ) 
  (initially_tagged : ℕ) 
  (second_catch : ℕ) 
  (h1 : total_fish = 1500) 
  (h2 : initially_tagged = 60) 
  (h3 : second_catch = 50) :
  (initially_tagged : ℚ) / total_fish * second_catch = 2 := by
  sorry

end tagged_fish_in_second_catch_l74_7449


namespace student_count_equality_l74_7465

/-- Proves that the number of students in class A equals the number of students in class C
    given the average ages of each class and the overall average age. -/
theorem student_count_equality (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (14 * a + 13 * b + 12 * c) / (a + b + c) = 13 → a = c := by
  sorry

end student_count_equality_l74_7465


namespace a2_value_l74_7457

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 2

-- Define the geometric sequence property for a_1, a_2, and a_5
def geometric_property (a : ℕ → ℝ) : Prop :=
  (a 2 / a 1) = (a 5 / a 2)

-- Theorem statement
theorem a2_value (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_geom : geometric_property a) : 
  a 2 = 3 := by
  sorry

end a2_value_l74_7457


namespace parallelogram_area_36_24_l74_7442

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 36 cm and height 24 cm is 864 cm² -/
theorem parallelogram_area_36_24 :
  parallelogram_area 36 24 = 864 := by
  sorry

end parallelogram_area_36_24_l74_7442


namespace cat_walking_time_l74_7483

/-- Proves that the total time for Jenny's cat walking process is 28 minutes -/
theorem cat_walking_time (resisting_time : ℝ) (walking_distance : ℝ) (walking_rate : ℝ) : 
  resisting_time = 20 →
  walking_distance = 64 →
  walking_rate = 8 →
  resisting_time + walking_distance / walking_rate = 28 :=
by sorry

end cat_walking_time_l74_7483


namespace tan_alpha_value_l74_7471

theorem tan_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
  sorry

end tan_alpha_value_l74_7471


namespace complex_number_quadrant_l74_7418

theorem complex_number_quadrant : 
  let z : ℂ := (5 * Complex.I) / (1 - 2 * Complex.I)
  (z.re < 0 ∧ z.im > 0) := by
  sorry

end complex_number_quadrant_l74_7418


namespace x_plus_y_value_l74_7494

theorem x_plus_y_value (x y : ℤ) (hx : -x = 3) (hy : |y| = 5) : x + y = 2 ∨ x + y = -8 := by
  sorry

end x_plus_y_value_l74_7494


namespace num_triples_eq_three_l74_7497

/-- The number of triples (a, b, c) of positive integers satisfying a + ab + abc = 11 -/
def num_triples : Nat :=
  (Finset.filter (fun t : Nat × Nat × Nat =>
    let (a, b, c) := t
    a > 0 ∧ b > 0 ∧ c > 0 ∧ a + a * b + a * b * c = 11)
    (Finset.product (Finset.range 12) (Finset.product (Finset.range 12) (Finset.range 12)))).card

/-- Theorem stating that there are exactly 3 triples (a, b, c) of positive integers
    satisfying a + ab + abc = 11 -/
theorem num_triples_eq_three : num_triples = 3 := by
  sorry

end num_triples_eq_three_l74_7497


namespace power_calculation_l74_7443

theorem power_calculation : 16^10 * 8^12 / 4^28 = 2^20 := by
  sorry

end power_calculation_l74_7443


namespace picasso_prints_probability_l74_7438

/-- The probability of arranging 4 specific items consecutively in a random arrangement of n items -/
def consecutive_probability (n : ℕ) (k : ℕ) : ℚ :=
  if n < k then 0
  else (k.factorial * (n - k + 1).factorial) / n.factorial

theorem picasso_prints_probability :
  consecutive_probability 12 4 = 1 / 55 := by
  sorry

end picasso_prints_probability_l74_7438


namespace equation_solution_l74_7458

theorem equation_solution : ∀ x : ℝ, x^2 - 2*x - 3 = x + 7 → x = 5 ∨ x = -2 := by
  sorry

end equation_solution_l74_7458


namespace function_value_comparison_l74_7456

theorem function_value_comparison (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x = x^2 + 2*x*(deriv f 2)) : f (-1) > f 1 := by
  sorry

end function_value_comparison_l74_7456


namespace valerie_light_bulb_purchase_l74_7404

/-- Calculates the money left over after buying light bulbs --/
def money_left_over (small_bulbs : ℕ) (large_bulbs : ℕ) (small_cost : ℕ) (large_cost : ℕ) (total_money : ℕ) : ℕ :=
  total_money - (small_bulbs * small_cost + large_bulbs * large_cost)

/-- Theorem: Valerie will have $24 left over after buying light bulbs --/
theorem valerie_light_bulb_purchase :
  money_left_over 3 1 8 12 60 = 24 := by
  sorry

end valerie_light_bulb_purchase_l74_7404


namespace point_on_y_axis_equal_distance_to_axes_l74_7450

-- Define point P with parameter a
def P (a : ℝ) : ℝ × ℝ := (2 + a, 3 * a - 6)

-- Theorem for part 1
theorem point_on_y_axis (a : ℝ) :
  P a = (0, -12) ↔ (P a).1 = 0 :=
sorry

-- Theorem for part 2
theorem equal_distance_to_axes (a : ℝ) :
  (P a = (6, 6) ∨ P a = (3, -3)) ↔ abs (P a).1 = abs (P a).2 :=
sorry

end point_on_y_axis_equal_distance_to_axes_l74_7450


namespace exponential_function_max_min_sum_l74_7495

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem exponential_function_max_min_sum (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ max (f a 0) (f a 1)) ∧
  (∀ x ∈ Set.Icc 0 1, f a x ≥ min (f a 0) (f a 1)) ∧
  (max (f a 0) (f a 1) + min (f a 0) (f a 1) = 3) →
  a = 2 :=
by
  sorry

end exponential_function_max_min_sum_l74_7495


namespace rectangle_area_similarity_l74_7431

theorem rectangle_area_similarity (R1_side : ℝ) (R1_area : ℝ) (R2_diagonal : ℝ) :
  R1_side = 3 →
  R1_area = 24 →
  R2_diagonal = 20 →
  ∃ (R2_area : ℝ), R2_area = 3200 / 73 := by
  sorry

end rectangle_area_similarity_l74_7431


namespace sara_coin_collection_value_l74_7498

/-- Calculates the total value in cents of a coin collection --/
def total_cents (quarters dimes nickels pennies : ℕ) : ℕ :=
  quarters * 25 + dimes * 10 + nickels * 5 + pennies

/-- Proves that Sara's coin collection totals 453 cents --/
theorem sara_coin_collection_value :
  total_cents 11 8 15 23 = 453 := by
  sorry

end sara_coin_collection_value_l74_7498


namespace tan_double_angle_special_case_l74_7402

/-- Given a function f(x) = sin x + cos x with f'(x) = 3f(x), prove that tan 2x = -4/3 -/
theorem tan_double_angle_special_case (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.sin x + Real.cos x) 
  (h2 : ∀ x, deriv f x = 3 * f x) : 
  ∀ x, Real.tan (2 * x) = -4/3 := by sorry

end tan_double_angle_special_case_l74_7402


namespace simplify_trig_expression_l74_7489

theorem simplify_trig_expression :
  Real.sqrt (2 + Real.cos (20 * π / 180) - Real.sin (10 * π / 180)^2) = Real.sqrt 3 * Real.cos (10 * π / 180) := by
  sorry

end simplify_trig_expression_l74_7489


namespace find_M_l74_7459

theorem find_M : ∃ M : ℝ, (0.2 * M = 0.6 * 1230) ∧ (M = 3690) := by sorry

end find_M_l74_7459


namespace contrapositive_geometric_sequence_l74_7405

/-- A sequence (a, b, c) is geometric if there exists a common ratio r such that b = ar and c = br -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- The main theorem: The contrapositive of "If (a,b,c) is geometric, then b^2 = ac" 
    is equivalent to "If b^2 ≠ ac, then (a,b,c) is not geometric" -/
theorem contrapositive_geometric_sequence (a b c : ℝ) :
  (¬(b^2 = a*c) → ¬(IsGeometricSequence a b c)) ↔
  (IsGeometricSequence a b c → b^2 = a*c) :=
sorry

end contrapositive_geometric_sequence_l74_7405


namespace system_solution_l74_7441

theorem system_solution (x y z : ℝ) : 
  x^4 + y^2 + 4 = 5*y*z ∧
  y^4 + z^2 + 4 = 5*z*x ∧
  z^4 + x^2 + 4 = 5*x*y →
  (x = y ∧ y = z ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2)) :=
by sorry

end system_solution_l74_7441


namespace different_color_probability_l74_7446

theorem different_color_probability (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ)
  (h1 : total_balls = white_balls + black_balls)
  (h2 : white_balls = 3)
  (h3 : black_balls = 1) :
  (white_balls * black_balls) / ((total_balls * (total_balls - 1)) / 2) = 1 / 2 :=
by sorry

end different_color_probability_l74_7446


namespace ellipse_major_axis_length_l74_7482

/-- The equation of an ellipse in its standard form -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 = 1

/-- The length of the major axis of an ellipse -/
def major_axis_length : ℝ := 8

/-- Theorem: The length of the major axis of the ellipse x^2/16 + y^2 = 1 is 8 -/
theorem ellipse_major_axis_length :
  ∀ x y : ℝ, is_ellipse x y → major_axis_length = 8 :=
by sorry

end ellipse_major_axis_length_l74_7482


namespace lcm_36_105_l74_7479

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l74_7479


namespace problem_statement_l74_7427

theorem problem_statement : (10 * 7)^3 + (45 * 5)^2 = 393625 := by
  sorry

end problem_statement_l74_7427


namespace number_of_small_boxes_l74_7475

/-- Given a large box containing small boxes of chocolates, this theorem proves
    the number of small boxes given the total number of chocolates and
    the number of chocolates per small box. -/
theorem number_of_small_boxes
  (total_chocolates : ℕ)
  (chocolates_per_box : ℕ)
  (h1 : total_chocolates = 400)
  (h2 : chocolates_per_box = 25)
  (h3 : total_chocolates % chocolates_per_box = 0) :
  total_chocolates / chocolates_per_box = 16 := by
  sorry

#check number_of_small_boxes

end number_of_small_boxes_l74_7475


namespace magic_act_disappearance_ratio_l74_7415

theorem magic_act_disappearance_ratio :
  ∀ (total_performances : ℕ) 
    (total_reappearances : ℕ) 
    (double_reappearance_prob : ℚ),
  total_performances = 100 →
  total_reappearances = 110 →
  double_reappearance_prob = 1/5 →
  (total_performances - 
   (total_reappearances - total_performances * double_reappearance_prob)) / 
   total_performances = 1/10 := by
sorry

end magic_act_disappearance_ratio_l74_7415


namespace max_k_value_l74_7492

theorem max_k_value (A B C : ℕ) (k : ℕ+) : 
  (A ≠ 0) →
  (A < 10) → 
  (B < 10) → 
  (C < 10) → 
  (k * (10 * A + B) = 100 * A + 10 * C + B) → 
  k ≤ 19 :=
sorry

end max_k_value_l74_7492


namespace quadratic_inequality_solution_set_l74_7493

theorem quadratic_inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) := by
  sorry

end quadratic_inequality_solution_set_l74_7493


namespace histogram_group_width_l74_7480

/-- Represents a group in a frequency histogram -/
structure HistogramGroup where
  a : ℝ
  b : ℝ
  m : ℝ  -- frequency
  h : ℝ  -- height
  h_pos : h > 0
  m_pos : m > 0
  a_lt_b : a < b

/-- 
The absolute value of the group width |a-b| in a frequency histogram 
is equal to the frequency m divided by the height h.
-/
theorem histogram_group_width (g : HistogramGroup) : 
  |g.b - g.a| = g.m / g.h := by
  sorry

end histogram_group_width_l74_7480


namespace tire_price_problem_l74_7473

theorem tire_price_problem (total_cost : ℝ) (fifth_tire_cost : ℝ) :
  total_cost = 485 →
  fifth_tire_cost = 5 →
  ∃ (regular_price : ℝ),
    4 * regular_price + fifth_tire_cost = total_cost ∧
    regular_price = 120 := by
  sorry

end tire_price_problem_l74_7473


namespace regular_polygon_sides_l74_7485

theorem regular_polygon_sides (n : ℕ) (h_regular : n ≥ 3) 
  (h_interior_angle : (n - 2) * 180 / n = 140) : n = 9 := by
  sorry

end regular_polygon_sides_l74_7485


namespace thumbtack_probability_estimate_l74_7422

-- Define the structure for the frequency table entry
structure FrequencyEntry :=
  (throws : ℕ)
  (touchingGround : ℕ)
  (frequency : ℚ)

-- Define the frequency table
def frequencyTable : List FrequencyEntry := [
  ⟨40, 20, 1/2⟩,
  ⟨120, 50, 417/1000⟩,
  ⟨320, 146, 456/1000⟩,
  ⟨480, 219, 456/1000⟩,
  ⟨720, 328, 456/1000⟩,
  ⟨800, 366, 458/1000⟩,
  ⟨920, 421, 458/1000⟩,
  ⟨1000, 463, 463/1000⟩
]

-- Define the function to estimate the probability
def estimateProbability (table : List FrequencyEntry) : ℚ :=
  -- Implementation details omitted
  sorry

-- Theorem statement
theorem thumbtack_probability_estimate :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  |estimateProbability frequencyTable - 46/100| < ε :=
sorry

end thumbtack_probability_estimate_l74_7422


namespace polynomial_divisibility_l74_7468

theorem polynomial_divisibility (n : ℕ) (hn : n > 0) :
  ∃ q : Polynomial ℝ, x^(2*n+1) - (2*n+1)*x^(n+1) + (2*n+1)*x^n - 1 = (x - 1)^3 * q := by
  sorry

end polynomial_divisibility_l74_7468


namespace sum_s_1_to_321_l74_7496

-- Define s(n) as the sum of all odd digits of n
def s (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_s_1_to_321 : 
  (Finset.range 321).sum s + s 321 = 1727 := by sorry

end sum_s_1_to_321_l74_7496


namespace pencil_distribution_l74_7426

/-- Given a class with 8 students and 120 pencils, prove that when the pencils are divided equally,
    each student receives 15 pencils. -/
theorem pencil_distribution (num_students : ℕ) (num_pencils : ℕ) (pencils_per_student : ℕ) 
    (h1 : num_students = 8)
    (h2 : num_pencils = 120)
    (h3 : num_pencils = num_students * pencils_per_student) :
  pencils_per_student = 15 := by
  sorry

end pencil_distribution_l74_7426


namespace intersection_sum_l74_7448

/-- Given two lines that intersect at (4,3), prove that a + b = 7/4 -/
theorem intersection_sum (a b : ℚ) : 
  (∀ x y : ℚ, x = (3/4) * y + a ↔ y = (3/4) * x + b) → 
  (4 = (3/4) * 3 + a ∧ 3 = (3/4) * 4 + b) →
  a + b = 7/4 := by
  sorry

end intersection_sum_l74_7448


namespace employee_reduction_percentage_l74_7451

/-- Theorem: Employee Reduction Percentage

Given:
- The number of employees decreased.
- The average salary increased by 10%.
- The total salary remained constant.

Prove:
The percentage decrease in the number of employees is (1 - 1/1.1) * 100%.
-/
theorem employee_reduction_percentage 
  (E : ℝ) -- Initial number of employees
  (E' : ℝ) -- Number of employees after reduction
  (S : ℝ) -- Initial average salary
  (h1 : E' < E) -- Number of employees decreased
  (h2 : E' * (1.1 * S) = E * S) -- Total salary remained constant
  : (E - E') / E * 100 = (1 - 1 / 1.1) * 100 := by
  sorry

#check employee_reduction_percentage

end employee_reduction_percentage_l74_7451


namespace exactly_one_integer_solution_l74_7413

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the property for (3n+i)^6 to be an integer
def is_integer_power (n : ℤ) : Prop :=
  ∃ m : ℤ, (3 * n + i : ℂ)^6 = m

-- Theorem statement
theorem exactly_one_integer_solution :
  ∃! n : ℤ, is_integer_power n :=
sorry

end exactly_one_integer_solution_l74_7413


namespace field_distance_l74_7460

theorem field_distance (D : ℝ) (mary edna lucy : ℝ) : 
  mary = (3/8) * D →
  edna = (2/3) * mary →
  lucy = (5/6) * edna →
  lucy + 4 = mary →
  D = 24 := by
sorry

end field_distance_l74_7460


namespace algebraic_expression_equality_l74_7436

theorem algebraic_expression_equality (a b : ℝ) : 
  (2*a + (1/2)*b)^2 - 4*(a^2 + b^2) = (2*a + (1/2)*b)^2 - 4*(a^2 + b^2) := by
  sorry

end algebraic_expression_equality_l74_7436


namespace cos_660_degrees_l74_7430

theorem cos_660_degrees : Real.cos (660 * π / 180) = 1 / 2 := by
  sorry

end cos_660_degrees_l74_7430


namespace elberta_amount_l74_7407

def granny_smith : ℕ := 63

def anjou : ℕ := granny_smith / 3

def elberta : ℕ := anjou + 2

theorem elberta_amount : elberta = 23 := by
  sorry

end elberta_amount_l74_7407


namespace value_of_M_l74_7423

theorem value_of_M : 
  let M := (Real.sqrt (Real.sqrt 7 + 3) - Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 - 1) + Real.sqrt (4 - 2 * Real.sqrt 3)
  M = (3 - Real.sqrt 6 + Real.sqrt 42) / 6 := by
  sorry

end value_of_M_l74_7423


namespace unknown_cube_edge_length_l74_7439

/-- The edge length of the unknown cube -/
def x : ℝ := 6

/-- The volume of a cube given its edge length -/
def cube_volume (edge : ℝ) : ℝ := edge ^ 3

theorem unknown_cube_edge_length :
  let cube1_edge : ℝ := 8
  let cube2_edge : ℝ := 10
  let new_cube_edge : ℝ := 12
  cube_volume new_cube_edge = cube_volume cube1_edge + cube_volume cube2_edge + cube_volume x :=
by sorry

end unknown_cube_edge_length_l74_7439


namespace commission_for_8000_l74_7429

/-- Represents the commission structure of a bank -/
structure BankCommission where
  /-- Fixed fee for any withdrawal -/
  fixed_fee : ℝ
  /-- Proportional fee rate for withdrawal amount -/
  prop_rate : ℝ

/-- Calculates the commission for a given withdrawal amount -/
def calculate_commission (bc : BankCommission) (amount : ℝ) : ℝ :=
  bc.fixed_fee + bc.prop_rate * amount

theorem commission_for_8000 :
  ∀ (bc : BankCommission),
    calculate_commission bc 5000 = 110 →
    calculate_commission bc 11000 = 230 →
    calculate_commission bc 8000 = 170 := by
  sorry

end commission_for_8000_l74_7429


namespace intersection_point_is_unique_l74_7486

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-14/17, 96/17)

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 4

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := 2 * y = -7 * x - 2

theorem intersection_point_is_unique :
  (∀ x y : ℚ, line1 x y ∧ line2 x y ↔ (x, y) = intersection_point) :=
sorry

end intersection_point_is_unique_l74_7486


namespace sound_propagation_at_10C_l74_7408

-- Define the relationship between temperature and speed of sound
def speed_of_sound (temp : Int) : Int :=
  match temp with
  | -20 => 318
  | -10 => 324
  | 0 => 330
  | 10 => 336
  | 20 => 342
  | 30 => 348
  | _ => 0  -- For temperatures not in the data set

-- Theorem statement
theorem sound_propagation_at_10C :
  speed_of_sound 10 * 4 = 1344 := by
  sorry


end sound_propagation_at_10C_l74_7408


namespace reciprocal_of_point_six_repeating_l74_7401

def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  d / (1 - (1/10))

theorem reciprocal_of_point_six_repeating :
  (repeating_decimal_to_fraction (6/10))⁻¹ = 3/2 := by
  sorry

end reciprocal_of_point_six_repeating_l74_7401


namespace asian_games_touring_routes_l74_7462

theorem asian_games_touring_routes :
  let total_cities : ℕ := 7
  let cities_to_visit : ℕ := 5
  let mandatory_cities : ℕ := 2
  let remaining_cities : ℕ := total_cities - mandatory_cities
  let cities_to_choose : ℕ := cities_to_visit - mandatory_cities
  let gaps : ℕ := cities_to_choose + 1

  (remaining_cities.factorial / (remaining_cities - cities_to_choose).factorial) *
  (gaps.choose mandatory_cities) = 600 :=
by sorry

end asian_games_touring_routes_l74_7462


namespace binomial_representation_l74_7487

theorem binomial_representation (n : ℕ) :
  ∃ x y z : ℕ, n = Nat.choose x 1 + Nat.choose y 2 + Nat.choose z 3 ∧
  ((0 ≤ x ∧ x < y ∧ y < z) ∨ (x = 0 ∧ y = 0 ∧ 0 < z)) :=
sorry

end binomial_representation_l74_7487


namespace sam_initial_yellow_marbles_l74_7445

/-- The number of yellow marbles Sam had initially -/
def initial_yellow_marbles : ℝ := 86.0

/-- The number of yellow marbles Joan gave to Sam -/
def joan_yellow_marbles : ℝ := 25.0

/-- The total number of yellow marbles Sam has now -/
def total_yellow_marbles : ℝ := 111

theorem sam_initial_yellow_marbles :
  initial_yellow_marbles + joan_yellow_marbles = total_yellow_marbles :=
by sorry

end sam_initial_yellow_marbles_l74_7445


namespace sum_two_longest_altitudes_l74_7461

/-- A right triangle with sides 6, 8, and 10 units -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  side_a : a = 6
  side_b : b = 8
  side_c : c = 10

/-- The sum of the lengths of the two longest altitudes in the given right triangle is 14 -/
theorem sum_two_longest_altitudes (t : RightTriangle) : 
  max t.a t.b + min t.a t.b = 14 := by
  sorry

end sum_two_longest_altitudes_l74_7461


namespace angle_measure_l74_7433

theorem angle_measure (x : ℝ) : 
  (90 - x = (180 - x) / 3 + 20) → x = 75 := by
  sorry

end angle_measure_l74_7433


namespace sin_675_degrees_l74_7417

theorem sin_675_degrees : Real.sin (675 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_675_degrees_l74_7417


namespace binomial_30_3_squared_l74_7454

theorem binomial_30_3_squared : (Nat.choose 30 3)^2 = 16483600 := by
  sorry

end binomial_30_3_squared_l74_7454


namespace intersecting_circles_sum_l74_7420

/-- Two circles intersect at points A and B, with their centers on a line -/
structure IntersectingCircles where
  m : ℝ
  n : ℝ
  /-- Point A coordinates -/
  pointA : ℝ × ℝ := (1, 3)
  /-- Point B coordinates -/
  pointB : ℝ × ℝ := (m, n)
  /-- The centers of both circles are on the line x - y - 2 = 0 -/
  centers_on_line : ∀ (x y : ℝ), x - y - 2 = 0

/-- The sum of m and n for the intersecting circles is 4 -/
theorem intersecting_circles_sum (ic : IntersectingCircles) : ic.m + ic.n = 4 := by
  sorry

end intersecting_circles_sum_l74_7420


namespace paperclip_production_l74_7410

/-- Given that 8 identical machines can produce 560 paperclips per minute,
    prove that 12 machines running at the same rate will produce 5040 paperclips in 6 minutes. -/
theorem paperclip_production 
  (rate : ℕ → ℕ → ℕ) -- rate function: number of machines → minutes → number of paperclips
  (h1 : rate 8 1 = 560) -- 8 machines produce 560 paperclips in 1 minute
  (h2 : ∀ n m, rate n m = n * rate 1 m) -- machines work at the same rate
  (h3 : ∀ n m k, rate n (m * k) = k * rate n m) -- linear scaling with time
  : rate 12 6 = 5040 :=
by sorry

end paperclip_production_l74_7410


namespace locus_is_circle_l74_7400

/-- Given two fixed points A and B in a plane, the locus of points C 
    satisfying $\overrightarrow{AC} \cdot \overrightarrow{BC} = 1$ is a circle. -/
theorem locus_is_circle (A B : ℝ × ℝ) : 
  {C : ℝ × ℝ | (C.1 - A.1, C.2 - A.2) • (C.1 - B.1, C.2 - B.2) = 1} = 
  {C : ℝ × ℝ | ∃ (center : ℝ × ℝ) (radius : ℝ), 
    (C.1 - center.1)^2 + (C.2 - center.2)^2 = radius^2} :=
sorry

end locus_is_circle_l74_7400


namespace air_conditioner_costs_and_minimum_cost_l74_7424

/-- Represents the cost and quantity of air conditioners -/
structure AirConditioner :=
  (costA : ℕ) -- Cost of type A
  (costB : ℕ) -- Cost of type B
  (quantityA : ℕ) -- Quantity of type A
  (quantityB : ℕ) -- Quantity of type B

/-- Conditions for air conditioner purchase -/
def satisfiesConditions (ac : AirConditioner) : Prop :=
  ac.costA * 3 + ac.costB * 2 = 39000 ∧
  ac.costA * 4 = ac.costB * 5 + 6000 ∧
  ac.quantityA + ac.quantityB = 30 ∧
  ac.quantityA * 2 ≥ ac.quantityB ∧
  ac.costA * ac.quantityA + ac.costB * ac.quantityB ≤ 217000

/-- Total cost of air conditioners -/
def totalCost (ac : AirConditioner) : ℕ :=
  ac.costA * ac.quantityA + ac.costB * ac.quantityB

/-- Theorem stating the correct costs and minimum total cost -/
theorem air_conditioner_costs_and_minimum_cost :
  ∃ (ac : AirConditioner),
    satisfiesConditions ac ∧
    ac.costA = 9000 ∧
    ac.costB = 6000 ∧
    (∀ (ac' : AirConditioner), satisfiesConditions ac' → totalCost ac ≤ totalCost ac') ∧
    totalCost ac = 210000 :=
  sorry

end air_conditioner_costs_and_minimum_cost_l74_7424


namespace solution_triplets_l74_7470

theorem solution_triplets (x y z : ℝ) :
  (2 * x^3 + 1 = 3 * z * x) ∧
  (2 * y^3 + 1 = 3 * x * y) ∧
  (2 * z^3 + 1 = 3 * y * z) →
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1/2 ∧ y = -1/2 ∧ z = -1/2)) :=
by sorry

end solution_triplets_l74_7470


namespace zeros_when_m_zero_one_zero_in_interval_l74_7466

/-- The function f(x) defined in terms of m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - (2*m + 1)*x + m*(m + 1)

/-- Theorem stating the zeros of f(x) when m = 0 -/
theorem zeros_when_m_zero :
  ∃ (x₁ x₂ : ℝ), x₁ = 0 ∧ x₂ = 1 ∧ f 0 x₁ = 0 ∧ f 0 x₂ = 0 :=
sorry

/-- Theorem stating the range of m for which f(x) has exactly one zero in (1,3) -/
theorem one_zero_in_interval (m : ℝ) :
  (∃! x, 1 < x ∧ x < 3 ∧ f m x = 0) ↔ (0 < m ∧ m ≤ 1) ∨ (2 ≤ m ∧ m < 3) :=
sorry

end zeros_when_m_zero_one_zero_in_interval_l74_7466


namespace cubic_polynomial_bound_l74_7444

theorem cubic_polynomial_bound (p q r : ℝ) : 
  ∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ∧ |x^3 + p*x^2 + q*x + r| ≥ (1/4 : ℝ) := by
sorry

end cubic_polynomial_bound_l74_7444


namespace painter_problem_l74_7476

theorem painter_problem (total_rooms : ℕ) (time_per_room : ℕ) (time_left : ℕ) 
  (h1 : total_rooms = 11)
  (h2 : time_per_room = 7)
  (h3 : time_left = 63) :
  total_rooms - (time_left / time_per_room) = 2 := by
  sorry

end painter_problem_l74_7476


namespace polynomial_identity_l74_7474

theorem polynomial_identity (p : ℝ → ℝ) 
  (h1 : ∀ x, p (x^2 + 1) = (p x)^2 + 1) 
  (h2 : p 0 = 0) : 
  ∀ x, p x = x := by sorry

end polynomial_identity_l74_7474


namespace stating_plant_distribution_theorem_l74_7467

/-- Represents the number of ways to distribute plants among lamps -/
def plant_distribution_ways : ℕ := 9

/-- The number of cactus plants -/
def num_cactus : ℕ := 3

/-- The number of bamboo plants -/
def num_bamboo : ℕ := 2

/-- The number of blue lamps -/
def num_blue_lamps : ℕ := 3

/-- The number of green lamps -/
def num_green_lamps : ℕ := 2

/-- 
Theorem stating that the number of ways to distribute the plants among the lamps is 9,
given the specified numbers of plants and lamps.
-/
theorem plant_distribution_theorem : 
  plant_distribution_ways = 9 := by sorry

end stating_plant_distribution_theorem_l74_7467


namespace addition_problems_l74_7434

theorem addition_problems :
  (189 + (-9) = 180) ∧
  ((-25) + 56 + (-39) = -8) ∧
  (41 + (-22) + (-33) + 19 = 5) ∧
  ((-0.5) + 13/4 + 2.75 + (-11/2) = 0) := by
  sorry

end addition_problems_l74_7434


namespace faster_watch_gain_rate_l74_7435

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ

/-- Calculates the difference in minutes between two times -/
def timeDifference (t1 t2 : Time) : ℕ := sorry

/-- Calculates the number of hours between two times -/
def hoursBetween (t1 t2 : Time) : ℕ := sorry

theorem faster_watch_gain_rate (alarmSetTime correctAlarmTime fasterAlarmTime : Time) 
  (h1 : alarmSetTime = ⟨22, 0⟩)  -- Alarm set at 10:00 PM
  (h2 : correctAlarmTime = ⟨4, 0⟩)  -- Correct watch shows 4:00 AM
  (h3 : fasterAlarmTime = ⟨4, 12⟩)  -- Faster watch shows 4:12 AM
  : (timeDifference correctAlarmTime fasterAlarmTime) / 
    (hoursBetween alarmSetTime correctAlarmTime) = 2 := by sorry

end faster_watch_gain_rate_l74_7435


namespace consecutive_integers_sum_l74_7447

theorem consecutive_integers_sum (n : ℕ) : 
  (n + 2 = 9) → (n + (n + 1) + (n + 2) = 24) := by
  sorry

end consecutive_integers_sum_l74_7447


namespace marble_problem_l74_7478

theorem marble_problem (x : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ)
  (h1 : angela = x)
  (h2 : brian = 3 * x)
  (h3 : caden = 2 * brian)
  (h4 : daryl = 4 * caden)
  (h5 : angela + brian + caden + daryl = 144) :
  x = 72 / 17 := by
sorry

end marble_problem_l74_7478


namespace simplify_expression_l74_7409

theorem simplify_expression (a : ℝ) : 3 * a^5 * (4 * a^7) = 12 * a^12 := by
  sorry

end simplify_expression_l74_7409


namespace no_integer_solutions_for_equation_l74_7416

theorem no_integer_solutions_for_equation :
  ¬ ∃ (x y : ℤ), x^2 - 7*y = 10 := by
  sorry

end no_integer_solutions_for_equation_l74_7416


namespace constant_term_expansion_l74_7463

/-- The constant term in the expansion of y^3 * (x + 1/(x^2*y))^n if it exists -/
def constantTerm (n : ℕ+) : ℕ :=
  if n = 9 then 84 else 0

theorem constant_term_expansion (n : ℕ+) :
  (∃ k : ℕ, k ≠ 0 ∧ constantTerm n = k) →
  constantTerm n = 84 :=
by sorry

end constant_term_expansion_l74_7463


namespace second_month_sale_l74_7437

theorem second_month_sale (
  average_sale : ℕ)
  (month1_sale : ℕ)
  (month3_sale : ℕ)
  (month4_sale : ℕ)
  (month5_sale : ℕ)
  (month6_sale : ℕ)
  (h1 : average_sale = 6500)
  (h2 : month1_sale = 6635)
  (h3 : month3_sale = 7230)
  (h4 : month4_sale = 6562)
  (h5 : month6_sale = 4791)
  : ∃ (month2_sale : ℕ),
    month2_sale = 13782 ∧
    (month1_sale + month2_sale + month3_sale + month4_sale + month5_sale + month6_sale) / 6 = average_sale :=
by sorry

end second_month_sale_l74_7437


namespace work_distance_is_ten_l74_7469

/-- Calculates the one-way distance to work given gas tank capacity, remaining fuel fraction, and fuel efficiency. -/
def distance_to_work (tank_capacity : ℚ) (remaining_fraction : ℚ) (miles_per_gallon : ℚ) : ℚ :=
  (tank_capacity * (1 - remaining_fraction) * miles_per_gallon) / 2

/-- Proves that given the specified conditions, Jim's work is 10 miles away from his house. -/
theorem work_distance_is_ten :
  let tank_capacity : ℚ := 12
  let remaining_fraction : ℚ := 2/3
  let miles_per_gallon : ℚ := 5
  distance_to_work tank_capacity remaining_fraction miles_per_gallon = 10 := by
  sorry


end work_distance_is_ten_l74_7469


namespace problem_solution_l74_7491

theorem problem_solution (x : ℚ) : 
  3 + 1 / (2 + 1 / (3 + 3 / (4 + x))) = 225/73 → x = -647/177 :=
by
  sorry

end problem_solution_l74_7491


namespace max_product_l74_7403

def digits : Finset Nat := {3, 5, 6, 8, 9}

def isValidPair (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def threeDigitNum (a b c : Nat) : Nat := 100 * a + 10 * b + c
def twoDigitNum (d e : Nat) : Nat := 10 * d + e

def product (a b c d e : Nat) : Nat :=
  threeDigitNum a b c * twoDigitNum d e

theorem max_product :
  ∀ a b c d e,
    isValidPair a b c d e →
    product a b c d e ≤ product 8 5 9 6 3 :=
by sorry

end max_product_l74_7403


namespace jane_mean_score_l74_7425

def jane_scores : List ℝ := [85, 88, 90, 92, 95, 100]

theorem jane_mean_score : 
  (jane_scores.sum / jane_scores.length : ℝ) = 550 / 6 := by
  sorry

end jane_mean_score_l74_7425


namespace no_integer_b_with_two_distinct_roots_l74_7412

theorem no_integer_b_with_two_distinct_roots :
  ¬ ∃ (b : ℤ), ∃ (x y : ℤ), x ≠ y ∧
    x^4 + 4*x^3 + b*x^2 + 16*x + 8 = 0 ∧
    y^4 + 4*y^3 + b*y^2 + 16*y + 8 = 0 :=
by sorry

end no_integer_b_with_two_distinct_roots_l74_7412


namespace hotel_room_cost_l74_7488

theorem hotel_room_cost (total_rooms : ℕ) (double_room_cost : ℕ) (total_revenue : ℕ) (single_rooms : ℕ) :
  total_rooms = 260 →
  double_room_cost = 60 →
  total_revenue = 14000 →
  single_rooms = 64 →
  ∃ (single_room_cost : ℕ),
    single_room_cost = 35 ∧
    single_room_cost * single_rooms + double_room_cost * (total_rooms - single_rooms) = total_revenue :=
by sorry

end hotel_room_cost_l74_7488


namespace determinant_transformation_l74_7432

theorem determinant_transformation (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = 3 →
  Matrix.det !![p, 5*p + 4*q; r, 5*r + 4*s] = 12 := by
sorry

end determinant_transformation_l74_7432


namespace base6_arithmetic_l74_7440

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 * 6^3 + d2 * 6^2 + d3 * 6 + d4

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : ℕ) : ℕ :=
  let d1 := n / (6^3)
  let d2 := (n / (6^2)) % 6
  let d3 := (n / 6) % 6
  let d4 := n % 6
  d1 * 1000 + d2 * 100 + d3 * 10 + d4

/-- The main theorem to prove --/
theorem base6_arithmetic : 
  base10ToBase6 (base6ToBase10 4512 - base6ToBase10 2324 + base6ToBase10 1432) = 4020 := by
  sorry

end base6_arithmetic_l74_7440


namespace lou_senior_first_cookies_l74_7464

/-- Represents the cookie jar situation --/
structure CookieJar where
  total : ℕ
  louSeniorFirst : ℕ
  louSeniorSecond : ℕ
  louieJunior : ℕ
  remaining : ℕ

/-- The cookie jar problem --/
def cookieJarProblem : CookieJar :=
  { total := 22
  , louSeniorFirst := 3  -- This is what we want to prove
  , louSeniorSecond := 1
  , louieJunior := 7
  , remaining := 11 }

/-- Theorem stating that Lou Senior took 3 cookies the first time --/
theorem lou_senior_first_cookies :
  cookieJarProblem.total - cookieJarProblem.louSeniorFirst - 
  cookieJarProblem.louSeniorSecond - cookieJarProblem.louieJunior = 
  cookieJarProblem.remaining :=
by sorry

end lou_senior_first_cookies_l74_7464


namespace new_ratio_after_addition_l74_7481

theorem new_ratio_after_addition (a b : ℤ) : 
  (a : ℚ) / b = 1 / 4 →
  b = 72 →
  (a + 6 : ℚ) / b = 1 / 3 := by
sorry

end new_ratio_after_addition_l74_7481


namespace n_fifth_minus_n_divisible_by_30_l74_7419

theorem n_fifth_minus_n_divisible_by_30 (n : ℤ) : 30 ∣ (n^5 - n) := by
  sorry

end n_fifth_minus_n_divisible_by_30_l74_7419


namespace sum_of_reciprocals_l74_7406

theorem sum_of_reciprocals (x y : ℚ) :
  (1 / x + 1 / y = 4) → (1 / x - 1 / y = -6) → (x + y = -4 / 5) := by
  sorry

end sum_of_reciprocals_l74_7406


namespace sun_radius_scientific_notation_l74_7472

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

/-- The sun's radius in kilometers -/
def sun_radius_km : ℝ := 696000

/-- The sun's radius in meters -/
def sun_radius_m : ℝ := sun_radius_km * km_to_m

theorem sun_radius_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), a ≥ 1 ∧ a < 10 ∧ sun_radius_m = a * (10 : ℝ) ^ n ∧ a = 6.96 ∧ n = 8 :=
sorry

end sun_radius_scientific_notation_l74_7472


namespace right_triangle_tangent_l74_7477

theorem right_triangle_tangent (n : ℕ) (a h : ℝ) (α : ℝ) :
  Odd n →
  0 < n →
  0 < a →
  0 < h →
  0 < α →
  α < π / 2 →
  Real.tan α = (4 * n * h) / ((n^2 - 1) * a) := by
  sorry

end right_triangle_tangent_l74_7477


namespace birthday_cake_icing_l74_7499

/-- Represents a 3D cube --/
structure Cube :=
  (size : ℕ)

/-- Represents the icing configuration on the cube --/
structure IcingConfig :=
  (top : Bool)
  (bottom : Bool)
  (side1 : Bool)
  (side2 : Bool)
  (side3 : Bool)
  (side4 : Bool)

/-- Counts the number of unit cubes with exactly two iced sides --/
def countTwoSidedIcedCubes (c : Cube) (ic : IcingConfig) : ℕ :=
  sorry

/-- The main theorem --/
theorem birthday_cake_icing (c : Cube) (ic : IcingConfig) :
  c.size = 5 →
  ic.top = true →
  ic.bottom = true →
  ic.side1 = true →
  ic.side2 = true →
  ic.side3 = false →
  ic.side4 = false →
  countTwoSidedIcedCubes c ic = 20 :=
sorry

end birthday_cake_icing_l74_7499


namespace rectangle_division_l74_7484

/-- If a rectangle with an area of 59.6 square centimeters is divided into 4 equal parts, 
    then the area of one part is 14.9 square centimeters. -/
theorem rectangle_division (total_area : ℝ) (num_parts : ℕ) (area_of_part : ℝ) : 
  total_area = 59.6 → 
  num_parts = 4 → 
  area_of_part = total_area / num_parts → 
  area_of_part = 14.9 := by
sorry

end rectangle_division_l74_7484


namespace math_paths_count_l74_7490

/-- Represents the number of adjacent positions a letter can move to -/
def adjacent_positions : ℕ := 8

/-- Represents the length of the word "MATH" -/
def word_length : ℕ := 4

/-- Calculates the number of paths to spell "MATH" -/
def num_paths : ℕ := adjacent_positions ^ (word_length - 1)

/-- Theorem stating that the number of paths to spell "MATH" is 512 -/
theorem math_paths_count : num_paths = 512 := by sorry

end math_paths_count_l74_7490


namespace intersection_x_sum_l74_7455

/-- The sum of x-coordinates of intersection points of two congruences -/
theorem intersection_x_sum : ∃ (S : Finset ℤ),
  (∀ x ∈ S, ∃ y : ℤ, 
    (y ≡ 7*x + 3 [ZMOD 20] ∧ y ≡ 13*x + 17 [ZMOD 20]) ∧
    (x ≥ 0 ∧ x < 20)) ∧
  (∀ x : ℤ, x ≥ 0 → x < 20 →
    (∃ y : ℤ, y ≡ 7*x + 3 [ZMOD 20] ∧ y ≡ 13*x + 17 [ZMOD 20]) →
    x ∈ S) ∧
  S.sum id = 12 :=
sorry

end intersection_x_sum_l74_7455


namespace profit_increase_condition_l74_7411

/-- The selling price function -/
def price (t : ℤ) : ℚ := (1/4) * t + 30

/-- The daily sales volume function -/
def sales_volume (t : ℤ) : ℚ := 120 - 2 * t

/-- The daily profit function after donation -/
def profit (t : ℤ) (n : ℚ) : ℚ :=
  (price t - 20 - n) * sales_volume t

/-- The derivative of the profit function with respect to t -/
def profit_derivative (t : ℤ) (n : ℚ) : ℚ :=
  -t + 2*n + 10

theorem profit_increase_condition (n : ℚ) :
  (∀ t : ℤ, 1 ≤ t ∧ t ≤ 28 → profit_derivative t n > 0) ↔
  (8.75 < n ∧ n ≤ 9.25) :=
sorry

end profit_increase_condition_l74_7411
