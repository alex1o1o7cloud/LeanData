import Mathlib

namespace NUMINAMATH_CALUDE_find_x_l491_49152

theorem find_x : ∃ x : ℝ, (5 * x) / (180 / 3) + 80 = 81 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l491_49152


namespace NUMINAMATH_CALUDE_one_cow_one_bag_days_l491_49194

/-- Given that 52 cows eat 104 bags of husk in 78 days, 
    prove that it takes 39 days for one cow to eat one bag of husk. -/
theorem one_cow_one_bag_days (cows : ℕ) (bags : ℕ) (days : ℕ) 
  (h1 : cows = 52) 
  (h2 : bags = 104) 
  (h3 : days = 78) : 
  (bags * days) / (cows * bags) = 39 := by
  sorry

end NUMINAMATH_CALUDE_one_cow_one_bag_days_l491_49194


namespace NUMINAMATH_CALUDE_range_of_2alpha_minus_beta_l491_49130

theorem range_of_2alpha_minus_beta (α β : ℝ) 
  (h1 : π < α + β ∧ α + β < (4*π)/3)
  (h2 : -π < α - β ∧ α - β < -π/3) :
  ∀ x, (-π < x ∧ x < π/6) ↔ ∃ α' β', 
    (π < α' + β' ∧ α' + β' < (4*π)/3) ∧
    (-π < α' - β' ∧ α' - β' < -π/3) ∧
    x = 2*α' - β' :=
by sorry

end NUMINAMATH_CALUDE_range_of_2alpha_minus_beta_l491_49130


namespace NUMINAMATH_CALUDE_volume_of_specific_polyhedron_l491_49146

/-- Represents a cube in 3D space -/
structure Cube where
  edge_length : ℝ

/-- Represents a convex polyhedron formed by planes passing through midpoints of cube edges -/
structure ConvexPolyhedron where
  cube : Cube

/-- Calculate the volume of the convex polyhedron -/
def volume (p : ConvexPolyhedron) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific convex polyhedron -/
theorem volume_of_specific_polyhedron :
  ∀ (c : Cube) (p : ConvexPolyhedron),
    c.edge_length = 2 →
    p.cube = c →
    volume p = 32 / 3 :=
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_polyhedron_l491_49146


namespace NUMINAMATH_CALUDE_exists_special_box_l491_49126

/-- A rectangular box with integer dimensions (a, b, c) where the volume is four times the surface area -/
def SpecialBox (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ a * b * c = 8 * (a * b + b * c + c * a)

/-- There exists at least one ordered triple (a, b, c) satisfying the SpecialBox conditions -/
theorem exists_special_box : ∃ (a b c : ℕ), SpecialBox a b c := by
  sorry

end NUMINAMATH_CALUDE_exists_special_box_l491_49126


namespace NUMINAMATH_CALUDE_propositions_3_and_4_are_true_l491_49153

theorem propositions_3_and_4_are_true :
  (∃ n : ℝ, ∀ m : ℝ, m * n = m) ∧
  (∀ n : ℝ, ∃ m : ℝ, m^2 < n) :=
by sorry

end NUMINAMATH_CALUDE_propositions_3_and_4_are_true_l491_49153


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l491_49195

/-- For a geometric sequence with common ratio q, the condition a_5 * a_6 < a_4^2 is necessary but not sufficient for 0 < q < 1 -/
theorem geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence definition
  (a 5 * a 6 < a 4 ^ 2) →       -- given condition
  (∃ q', 0 < q' ∧ q' < 1 ∧ ¬(a 5 * a 6 < a 4 ^ 2 → 0 < q' ∧ q' < 1)) ∧
  (0 < q ∧ q < 1 → a 5 * a 6 < a 4 ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l491_49195


namespace NUMINAMATH_CALUDE_find_n_l491_49128

theorem find_n (x y n : ℝ) (h1 : x = 3) (h2 : y = 27) (h3 : n^(n / (2 + x)) = y) : n = 15 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l491_49128


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_l491_49103

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number is 465 -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_l491_49103


namespace NUMINAMATH_CALUDE_domain_of_composite_function_l491_49111

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-1) 2

-- State the theorem
theorem domain_of_composite_function :
  {x : ℝ | f (3 - 2*x) ∈ Set.range f} = Set.Icc (1/2) 2 :=
sorry

end NUMINAMATH_CALUDE_domain_of_composite_function_l491_49111


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l491_49186

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 3 + a 4 + a 5 = 12 → a 1 + a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l491_49186


namespace NUMINAMATH_CALUDE_new_person_weight_l491_49190

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 45 →
  ∃ (new_weight : ℝ),
    new_weight = initial_count * weight_increase + replaced_weight ∧
    new_weight = 65 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l491_49190


namespace NUMINAMATH_CALUDE_polygon_sides_l491_49166

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 2 * 360 + 180 → n = 7 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l491_49166


namespace NUMINAMATH_CALUDE_circle_point_perpendicular_l491_49107

theorem circle_point_perpendicular (m : ℝ) : m > 0 →
  (∃ (P : ℝ × ℝ), P.1^2 + P.2^2 = 1 ∧ 
    ((P.1 + m) * (P.1 - m) + (P.2 - 2) * (P.2 - 2) = 0)) →
  (3 : ℝ) - 1 = 2 := by sorry

end NUMINAMATH_CALUDE_circle_point_perpendicular_l491_49107


namespace NUMINAMATH_CALUDE_unique_k_square_sum_l491_49187

theorem unique_k_square_sum : ∃! (k : ℕ), k ≠ 1 ∧
  (∃ (n : ℕ), k = n^2 + (n+1)^2) ∧
  (∃ (m : ℕ), k^4 = m^2 + (m+1)^2) :=
by sorry

end NUMINAMATH_CALUDE_unique_k_square_sum_l491_49187


namespace NUMINAMATH_CALUDE_equation_transformation_correctness_l491_49183

theorem equation_transformation_correctness :
  -- Option A is incorrect
  (∀ x : ℝ, 3 + x = 7 → x ≠ 7 + 3) ∧
  -- Option B is incorrect
  (∀ x : ℝ, 5 * x = -4 → x ≠ -5/4) ∧
  -- Option C is incorrect
  (∀ x : ℝ, 7/4 * x = 3 → x ≠ 3 * 7/4) ∧
  -- Option D is correct
  (∀ x : ℝ, -(x - 2) / 4 = 1 → -(x - 2) = 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_transformation_correctness_l491_49183


namespace NUMINAMATH_CALUDE_labourer_savings_is_30_l491_49170

/-- Calculates the savings of a labourer after clearing debt -/
def labourerSavings (monthlyIncome : ℕ) (initialExpenditure : ℕ) (initialMonths : ℕ)
  (reducedExpenditure : ℕ) (reducedMonths : ℕ) : ℕ :=
  let initialDebt := initialMonths * initialExpenditure - initialMonths * monthlyIncome
  let availableAmount := reducedMonths * monthlyIncome - reducedMonths * reducedExpenditure
  availableAmount - initialDebt

/-- The labourer's savings after clearing debt is 30 -/
theorem labourer_savings_is_30 :
  labourerSavings 78 85 6 60 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_labourer_savings_is_30_l491_49170


namespace NUMINAMATH_CALUDE_fourteen_machines_four_minutes_l491_49188

/-- The number of bottles produced by a given number of machines in a given time -/
def bottles_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  let base_machines := 6
  let base_production := 270
  let production_per_machine_per_minute := base_production / base_machines
  machines * production_per_machine_per_minute * minutes

/-- Theorem stating that 14 machines produce 2520 bottles in 4 minutes -/
theorem fourteen_machines_four_minutes :
  bottles_produced 14 4 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_fourteen_machines_four_minutes_l491_49188


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l491_49131

/-- Given that the solution set of ax^2 - bx - 1 > 0 is {x | -1/2 < x < -1/3},
    prove that the solution set of x^2 - bx - a ≥ 0 is {x | x ≥ 3 ∨ x ≤ 2} -/
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : ∀ x : ℝ, ax^2 - b*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :
  ∀ x : ℝ, x^2 - b*x - a ≥ 0 ↔ x ≥ 3 ∨ x ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l491_49131


namespace NUMINAMATH_CALUDE_point_on_line_l491_49113

/-- Given two points A and B in the Cartesian plane, if a point C satisfies the vector equation
    OC = s*OA + t*OB where s + t = 1, then C lies on the line passing through A and B. -/
theorem point_on_line (A B C : ℝ × ℝ) (s t : ℝ) :
  A = (2, 1) →
  B = (-1, -2) →
  C = s • A + t • B →
  s + t = 1 →
  C.1 - C.2 = 1 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l491_49113


namespace NUMINAMATH_CALUDE_complex_number_equality_l491_49105

theorem complex_number_equality (z : ℂ) : z * (1 + Complex.I * Real.sqrt 3) = Complex.abs (1 + Complex.I * Real.sqrt 3) → z = 1 - Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l491_49105


namespace NUMINAMATH_CALUDE_coins_sold_proof_l491_49119

def beth_initial_coins : ℕ := 250
def carl_gift_coins : ℕ := 75
def sell_percentage : ℚ := 60 / 100

theorem coins_sold_proof :
  let total_coins := beth_initial_coins + carl_gift_coins
  ⌊(sell_percentage * total_coins : ℚ)⌋ = 195 := by sorry

end NUMINAMATH_CALUDE_coins_sold_proof_l491_49119


namespace NUMINAMATH_CALUDE_range_of_a_l491_49163

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → (0 < a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l491_49163


namespace NUMINAMATH_CALUDE_rectangle_area_l491_49144

/-- The area of a rectangle with width 81/4 cm and height 148/9 cm is 333 cm². -/
theorem rectangle_area : 
  let width : ℚ := 81 / 4
  let height : ℚ := 148 / 9
  (width * height : ℚ) = 333 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l491_49144


namespace NUMINAMATH_CALUDE_store_sale_revenue_l491_49196

/-- Calculates the amount left after a store's inventory sale --/
theorem store_sale_revenue (total_items : ℕ) (category_a_items : ℕ) (category_b_items : ℕ) (category_c_items : ℕ)
  (price_a : ℝ) (price_b : ℝ) (price_c : ℝ)
  (discount_a : ℝ) (discount_b : ℝ) (discount_c : ℝ)
  (sales_percent_a : ℝ) (sales_percent_b : ℝ) (sales_percent_c : ℝ)
  (return_rate : ℝ) (advertising_cost : ℝ) (creditors_amount : ℝ) :
  total_items = category_a_items + category_b_items + category_c_items →
  category_a_items = 1000 →
  category_b_items = 700 →
  category_c_items = 300 →
  price_a = 50 →
  price_b = 75 →
  price_c = 100 →
  discount_a = 0.8 →
  discount_b = 0.7 →
  discount_c = 0.6 →
  sales_percent_a = 0.85 →
  sales_percent_b = 0.75 →
  sales_percent_c = 0.9 →
  return_rate = 0.03 →
  advertising_cost = 2000 →
  creditors_amount = 15000 →
  ∃ (revenue : ℝ), revenue = 13172.50 ∧ 
    revenue = (category_a_items * sales_percent_a * price_a * (1 - discount_a) * (1 - return_rate) +
               category_b_items * sales_percent_b * price_b * (1 - discount_b) * (1 - return_rate) +
               category_c_items * sales_percent_c * price_c * (1 - discount_c) * (1 - return_rate)) -
              advertising_cost - creditors_amount :=
by
  sorry


end NUMINAMATH_CALUDE_store_sale_revenue_l491_49196


namespace NUMINAMATH_CALUDE_selene_total_cost_l491_49182

/-- Calculate the total cost of Selene's purchase --/
def calculate_total_cost (camera_price : ℚ) (camera_count : ℕ) (frame_price : ℚ) (frame_count : ℕ)
  (card_price : ℚ) (card_count : ℕ) (camera_discount : ℚ) (frame_discount : ℚ) (card_discount : ℚ)
  (camera_frame_tax : ℚ) (card_tax : ℚ) : ℚ :=
  let camera_total := camera_price * camera_count
  let frame_total := frame_price * frame_count
  let card_total := card_price * card_count
  let camera_discounted := camera_total * (1 - camera_discount)
  let frame_discounted := frame_total * (1 - frame_discount)
  let card_discounted := card_total * (1 - card_discount)
  let camera_frame_subtotal := camera_discounted + frame_discounted
  let camera_frame_taxed := camera_frame_subtotal * (1 + camera_frame_tax)
  let card_taxed := card_discounted * (1 + card_tax)
  camera_frame_taxed + card_taxed

/-- Theorem stating that Selene's total cost is $691.72 --/
theorem selene_total_cost :
  calculate_total_cost 110 2 120 3 30 4 (7/100) (5/100) (10/100) (6/100) (4/100) = 69172/100 := by
  sorry

end NUMINAMATH_CALUDE_selene_total_cost_l491_49182


namespace NUMINAMATH_CALUDE_inequality_proof_l491_49165

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (2 * x^2) / (y + z) + (2 * y^2) / (z + x) + (2 * z^2) / (x + y) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l491_49165


namespace NUMINAMATH_CALUDE_b_nonnegative_l491_49114

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem b_nonnegative 
  (a b c m₁ m₂ : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : f a b c 1 = 0) 
  (h4 : a^2 + (f a b c m₁ + f a b c m₂) * a + f a b c m₁ * f a b c m₂ = 0) :
  b ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_b_nonnegative_l491_49114


namespace NUMINAMATH_CALUDE_cookie_boxes_problem_l491_49122

theorem cookie_boxes_problem (n : ℕ) : n = 12 ↔ 
  n > 0 ∧ 
  n - 11 ≥ 1 ∧ 
  n - 2 ≥ 1 ∧ 
  (n - 11) + (n - 2) < n ∧
  ∀ m : ℕ, m > n → ¬(m > 0 ∧ m - 11 ≥ 1 ∧ m - 2 ≥ 1 ∧ (m - 11) + (m - 2) < m) :=
by sorry

end NUMINAMATH_CALUDE_cookie_boxes_problem_l491_49122


namespace NUMINAMATH_CALUDE_derivative_of_f_at_1_l491_49192

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_of_f_at_1 : 
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_at_1_l491_49192


namespace NUMINAMATH_CALUDE_negative_three_less_than_negative_one_l491_49179

theorem negative_three_less_than_negative_one : -3 < -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_less_than_negative_one_l491_49179


namespace NUMINAMATH_CALUDE_solution_using_determinants_l491_49104

/-- Definition of 2x2 determinant -/
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

/-- System of equations -/
def equation1 (x y : ℝ) : Prop := 2 * x - y = 1
def equation2 (x y : ℝ) : Prop := 3 * x + 2 * y = 11

/-- Determinants for the system -/
def D : ℝ := det2x2 2 (-1) 3 2
def D_x : ℝ := det2x2 1 (-1) 11 2
def D_y : ℝ := det2x2 2 1 3 11

/-- Theorem: Solution of the system using determinant method -/
theorem solution_using_determinants :
  ∃ (x y : ℝ), equation1 x y ∧ equation2 x y ∧ x = D_x / D ∧ y = D_y / D :=
sorry

end NUMINAMATH_CALUDE_solution_using_determinants_l491_49104


namespace NUMINAMATH_CALUDE_function_composition_equality_l491_49108

theorem function_composition_equality (a b c d : ℝ) :
  let f := fun (x : ℝ) => a * x + b
  let g := fun (x : ℝ) => c * x + d
  (∀ x, f (g x) = g (f x)) ↔ (b = d ∨ a = c + 1) :=
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l491_49108


namespace NUMINAMATH_CALUDE_hexagon_exterior_angle_sum_hexagon_exterior_angle_sum_proof_l491_49115

/-- The sum of the exterior angles of a hexagon is 360 degrees. -/
theorem hexagon_exterior_angle_sum : ℝ :=
  360

#check hexagon_exterior_angle_sum

/-- Proof of the theorem -/
theorem hexagon_exterior_angle_sum_proof :
  hexagon_exterior_angle_sum = 360 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_exterior_angle_sum_hexagon_exterior_angle_sum_proof_l491_49115


namespace NUMINAMATH_CALUDE_triangle_inequalities_l491_49151

/-- Given a triangle with side lengths a, b, c, circumradius R, and inradius r,
    prove the inequalities abc ≥ (a+b-c)(a-b+c)(-a+b+c) and R ≥ 2r -/
theorem triangle_inequalities (a b c R r : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_circumradius : R > 0)
  (h_inradius : r > 0)
  (h_area : 4 * R * (r * (a + b + c) / 2) = a * b * c) :
  a * b * c ≥ (a + b - c) * (a - b + c) * (-a + b + c) ∧ R ≥ 2 * r := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l491_49151


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_equals_six_l491_49175

theorem mean_equality_implies_y_equals_six :
  let mean1 := (4 + 8 + 16) / 3
  let mean2 := (10 + 12 + y) / 3
  mean1 = mean2 → y = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_equals_six_l491_49175


namespace NUMINAMATH_CALUDE_probability_two_blue_l491_49156

/-- Represents a jar with red and blue buttons -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- Calculates the total number of buttons in a jar -/
def Jar.total (j : Jar) : ℕ := j.red + j.blue

/-- Represents the state of both jars after button removal -/
structure JarState where
  c : Jar
  d : Jar

/-- Defines the initial state of Jar C -/
def initial_jar_c : Jar := { red := 6, blue := 10 }

/-- Defines the button removal process -/
def remove_buttons (j : Jar) (n : ℕ) : JarState :=
  { c := { red := j.red - n, blue := j.blue - n },
    d := { red := n, blue := n } }

/-- Theorem stating the probability of choosing two blue buttons -/
theorem probability_two_blue (n : ℕ) : 
  let initial_total := initial_jar_c.total
  let final_state := remove_buttons initial_jar_c n
  final_state.c.total = (3 * initial_total) / 4 →
  (final_state.c.blue : ℚ) / final_state.c.total * 
  (final_state.d.blue : ℚ) / final_state.d.total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_blue_l491_49156


namespace NUMINAMATH_CALUDE_collinear_points_sum_l491_49112

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p2 = p1 + t • (p3 - p1) ∨ p1 = p2 + t • (p3 - p2)

theorem collinear_points_sum (a b : ℝ) :
  collinear (2, a, b) (a, 3, b) (a, b, 4) → a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l491_49112


namespace NUMINAMATH_CALUDE_arccos_cos_2x_solution_set_l491_49164

theorem arccos_cos_2x_solution_set :
  ∀ x : ℝ, (Real.arccos (Real.cos (2 * x)) = x) ↔ 
    (∃ k : ℤ, x = 2 * k * π ∨ x = 2 * π / 3 + 2 * k * π ∨ x = -(2 * π / 3) + 2 * k * π) :=
by sorry

end NUMINAMATH_CALUDE_arccos_cos_2x_solution_set_l491_49164


namespace NUMINAMATH_CALUDE_problem_statement_l491_49134

theorem problem_statement (a b : ℝ) (h1 : 2*a + b = -3) (h2 : 2*a - b = 2) :
  4*a^2 - b^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l491_49134


namespace NUMINAMATH_CALUDE_rectangle_longest_side_l491_49160

/-- A rectangle with perimeter 240 feet and area equal to eight times its perimeter has its longest side equal to 80 feet. -/
theorem rectangle_longest_side (l w : ℝ) : 
  (l > 0) → 
  (w > 0) → 
  (2 * l + 2 * w = 240) → 
  (l * w = 8 * (2 * l + 2 * w)) → 
  (max l w = 80) := by
sorry

end NUMINAMATH_CALUDE_rectangle_longest_side_l491_49160


namespace NUMINAMATH_CALUDE_strawberry_harvest_l491_49180

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ :=
  d.length * d.width

/-- Calculates the total number of plants in the garden -/
def totalPlants (d : GardenDimensions) (density : ℝ) : ℝ :=
  gardenArea d * density

/-- Calculates the total number of strawberries harvested -/
def totalStrawberries (d : GardenDimensions) (density : ℝ) (yield : ℝ) : ℝ :=
  totalPlants d density * yield

/-- Theorem: The total number of strawberries harvested is 5400 -/
theorem strawberry_harvest (d : GardenDimensions) (density : ℝ) (yield : ℝ)
    (h1 : d.length = 10)
    (h2 : d.width = 9)
    (h3 : density = 5)
    (h4 : yield = 12) :
    totalStrawberries d density yield = 5400 := by
  sorry

#eval totalStrawberries ⟨10, 9⟩ 5 12

end NUMINAMATH_CALUDE_strawberry_harvest_l491_49180


namespace NUMINAMATH_CALUDE_limit_proof_l491_49121

open Real

theorem limit_proof (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧
  ∀ x : ℝ, 0 < |x + 7/2| ∧ |x + 7/2| < δ →
    |(2*x^2 + 13*x + 21) / (2*x + 7) + 1/2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_proof_l491_49121


namespace NUMINAMATH_CALUDE_difference_of_squares_divided_l491_49127

theorem difference_of_squares_divided : (311^2 - 297^2) / 14 = 608 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_divided_l491_49127


namespace NUMINAMATH_CALUDE_bottle_caps_difference_l491_49198

-- Define the number of bottle caps found and thrown away each day
def monday_found : ℕ := 36
def monday_thrown : ℕ := 45
def tuesday_found : ℕ := 58
def tuesday_thrown : ℕ := 30
def wednesday_found : ℕ := 80
def wednesday_thrown : ℕ := 70

-- Define the final number of bottle caps left
def final_caps : ℕ := 65

-- Theorem to prove
theorem bottle_caps_difference :
  (monday_found + tuesday_found + wednesday_found) -
  (monday_thrown + tuesday_thrown + wednesday_thrown) = 29 :=
by sorry

end NUMINAMATH_CALUDE_bottle_caps_difference_l491_49198


namespace NUMINAMATH_CALUDE_pattern_cannot_form_cube_l491_49197

/-- Represents a square in the pattern -/
structure Square :=
  (id : ℕ)

/-- Represents the pattern of squares -/
structure Pattern :=
  (center : Square)
  (top : Square)
  (left : Square)
  (right : Square)
  (front : Square)

/-- Represents a cube -/
structure Cube :=
  (faces : Fin 6 → Square)

/-- Defines the given pattern -/
def given_pattern : Pattern :=
  { center := ⟨0⟩
  , top := ⟨1⟩
  , left := ⟨2⟩
  , right := ⟨3⟩
  , front := ⟨4⟩ }

/-- Theorem stating that the given pattern cannot form a cube -/
theorem pattern_cannot_form_cube :
  ¬ ∃ (c : Cube), c.faces 0 = given_pattern.center ∧
                  c.faces 1 = given_pattern.top ∧
                  c.faces 2 = given_pattern.left ∧
                  c.faces 3 = given_pattern.right ∧
                  c.faces 4 = given_pattern.front :=
by
  sorry


end NUMINAMATH_CALUDE_pattern_cannot_form_cube_l491_49197


namespace NUMINAMATH_CALUDE_bananas_in_E_l491_49120

/-- The number of baskets -/
def num_baskets : ℕ := 5

/-- The average number of fruits per basket -/
def avg_fruits_per_basket : ℕ := 25

/-- The number of fruits in basket A -/
def fruits_in_A : ℕ := 15

/-- The number of fruits in basket B -/
def fruits_in_B : ℕ := 30

/-- The number of fruits in basket C -/
def fruits_in_C : ℕ := 20

/-- The number of fruits in basket D -/
def fruits_in_D : ℕ := 25

/-- Theorem: The number of bananas in basket E is 35 -/
theorem bananas_in_E : 
  num_baskets * avg_fruits_per_basket - (fruits_in_A + fruits_in_B + fruits_in_C + fruits_in_D) = 35 := by
  sorry

end NUMINAMATH_CALUDE_bananas_in_E_l491_49120


namespace NUMINAMATH_CALUDE_first_discount_percentage_l491_49177

theorem first_discount_percentage
  (list_price : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ)
  (h1 : list_price = 70)
  (h2 : final_price = 61.74)
  (h3 : second_discount = 0.01999999999999997)
  : ∃ (first_discount : ℝ),
    first_discount = 0.1 ∧
    final_price = list_price * (1 - first_discount) * (1 - second_discount) :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l491_49177


namespace NUMINAMATH_CALUDE_mrs_white_orchard_yield_l491_49109

/-- Represents the dimensions and crop yields of Mrs. White's orchard -/
structure Orchard where
  length_paces : ℕ
  width_paces : ℕ
  feet_per_pace : ℕ
  tomato_yield_per_sqft : ℚ
  cucumber_yield_per_sqft : ℚ

/-- Calculates the expected crop yield from the orchard -/
def expected_yield (o : Orchard) : ℚ :=
  let area_sqft := (o.length_paces * o.feet_per_pace) * (o.width_paces * o.feet_per_pace)
  let half_area_sqft := area_sqft / 2
  let tomato_yield := half_area_sqft * o.tomato_yield_per_sqft
  let cucumber_yield := half_area_sqft * o.cucumber_yield_per_sqft
  tomato_yield + cucumber_yield

/-- Mrs. White's orchard -/
def mrs_white_orchard : Orchard :=
  { length_paces := 10
  , width_paces := 30
  , feet_per_pace := 3
  , tomato_yield_per_sqft := 3/4
  , cucumber_yield_per_sqft := 2/5 }

theorem mrs_white_orchard_yield :
  expected_yield mrs_white_orchard = 1552.5 := by
  sorry

end NUMINAMATH_CALUDE_mrs_white_orchard_yield_l491_49109


namespace NUMINAMATH_CALUDE_math_city_intersections_l491_49116

/-- Represents a city with a given number of streets -/
structure City where
  numStreets : ℕ
  noParallel : Bool
  noTripleIntersections : Bool

/-- Calculates the number of intersections in a city -/
def numIntersections (c : City) : ℕ :=
  (c.numStreets.pred * c.numStreets.pred) / 2

theorem math_city_intersections (c : City) 
  (h1 : c.numStreets = 10)
  (h2 : c.noParallel = true)
  (h3 : c.noTripleIntersections = true) :
  numIntersections c = 45 := by
  sorry

end NUMINAMATH_CALUDE_math_city_intersections_l491_49116


namespace NUMINAMATH_CALUDE_inequality_proof_l491_49150

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_cond : |Real.sqrt (a * b) - Real.sqrt (c * d)| ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l491_49150


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l491_49138

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = 1 - 5 * Complex.I) : 
  z.im = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l491_49138


namespace NUMINAMATH_CALUDE_students_representing_x_percent_of_boys_l491_49141

theorem students_representing_x_percent_of_boys 
  (total_population : ℝ) 
  (boys_percentage : ℝ) 
  (x : ℝ) 
  (h1 : total_population = 113.38934190276818)
  (h2 : boys_percentage = 70) :
  (x / 100) * (boys_percentage / 100 * total_population) = 
  (x / 100) * 79.37253933173772 :=
by
  sorry

end NUMINAMATH_CALUDE_students_representing_x_percent_of_boys_l491_49141


namespace NUMINAMATH_CALUDE_sin_minus_cos_sqrt_two_l491_49132

theorem sin_minus_cos_sqrt_two (x : Real) :
  0 ≤ x ∧ x < 2 * Real.pi ∧ Real.sin x - Real.cos x = Real.sqrt 2 → x = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_cos_sqrt_two_l491_49132


namespace NUMINAMATH_CALUDE_beth_comic_books_percentage_l491_49171

theorem beth_comic_books_percentage
  (total_books : ℕ)
  (novel_percentage : ℚ)
  (graphic_novels : ℕ)
  (h1 : total_books = 120)
  (h2 : novel_percentage = 65/100)
  (h3 : graphic_novels = 18) :
  (total_books - (novel_percentage * total_books).floor - graphic_novels) / total_books = 1/5 := by
sorry

end NUMINAMATH_CALUDE_beth_comic_books_percentage_l491_49171


namespace NUMINAMATH_CALUDE_sean_money_difference_l491_49169

theorem sean_money_difference (fritz_money : ℕ) (rick_sean_total : ℕ) : 
  fritz_money = 40 →
  rick_sean_total = 96 →
  ∃ (sean_money : ℕ),
    sean_money > fritz_money / 2 ∧
    3 * sean_money + sean_money = rick_sean_total ∧
    sean_money - fritz_money / 2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_sean_money_difference_l491_49169


namespace NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l491_49191

theorem smallest_x_for_perfect_cube : ∃ (x : ℕ+), 
  (∀ (y : ℕ+), 2520 * y.val = (M : ℕ)^3 → x ≤ y) ∧ 
  (∃ (M : ℕ), 2520 * x.val = M^3) ∧
  x.val = 3675 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l491_49191


namespace NUMINAMATH_CALUDE_lynne_magazines_l491_49181

def num_books : ℕ := 9
def book_cost : ℕ := 7
def magazine_cost : ℕ := 4
def total_spent : ℕ := 75

theorem lynne_magazines :
  ∃ (num_magazines : ℕ),
    num_magazines * magazine_cost + num_books * book_cost = total_spent ∧
    num_magazines = 3 := by
  sorry

end NUMINAMATH_CALUDE_lynne_magazines_l491_49181


namespace NUMINAMATH_CALUDE_lawrence_marbles_l491_49118

theorem lawrence_marbles (num_friends : ℕ) (marbles_per_friend : ℕ) 
  (h1 : num_friends = 64) (h2 : marbles_per_friend = 86) : 
  num_friends * marbles_per_friend = 5504 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_marbles_l491_49118


namespace NUMINAMATH_CALUDE_cos_300_degrees_l491_49173

theorem cos_300_degrees : Real.cos (300 * π / 180) = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l491_49173


namespace NUMINAMATH_CALUDE_parking_probability_probability_equals_actual_l491_49123

/-- The probability of finding 3 consecutive empty spaces in a row of 18 spaces 
    where 14 spaces are randomly occupied -/
theorem parking_probability : ℝ := by
  -- Define the total number of spaces
  let total_spaces : ℕ := 18
  -- Define the number of occupied spaces
  let occupied_spaces : ℕ := 14
  -- Define the number of consecutive empty spaces needed
  let required_empty_spaces : ℕ := 3
  
  -- Calculate the probability
  -- We're not providing the actual calculation here, just the structure
  sorry

-- The actual probability value
def actual_probability : ℚ := 171 / 204

-- Prove that the calculated probability equals the actual probability
theorem probability_equals_actual : parking_probability = actual_probability := by
  sorry

end NUMINAMATH_CALUDE_parking_probability_probability_equals_actual_l491_49123


namespace NUMINAMATH_CALUDE_expression_evaluation_l491_49106

theorem expression_evaluation : (75 / 1.5) * (500 / 25) - (300 / 0.03) + (125 * 4 / 0.1) = -4000 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l491_49106


namespace NUMINAMATH_CALUDE_a_25_mod_26_l491_49110

/-- Definition of a_n as the integer obtained by concatenating all integers from 1 to n -/
def a (n : ℕ) : ℕ := sorry

/-- Theorem stating that a_25 mod 26 = 13 -/
theorem a_25_mod_26 : a 25 % 26 = 13 := by sorry

end NUMINAMATH_CALUDE_a_25_mod_26_l491_49110


namespace NUMINAMATH_CALUDE_line_problem_l491_49102

/-- A line in the xy-plane defined by y = 2x + 4 -/
def line (x : ℝ) : ℝ := 2 * x + 4

/-- Point P on the x-axis -/
def P (p : ℝ) : ℝ × ℝ := (p, 0)

/-- Point R where the line intersects the y-axis -/
def R : ℝ × ℝ := (0, line 0)

/-- Point Q where the line intersects the vertical line through P -/
def Q (p : ℝ) : ℝ × ℝ := (p, line p)

/-- Area of the quadrilateral OPQR -/
def area_OPQR (p : ℝ) : ℝ := p * (p + 4)

theorem line_problem (p : ℝ) (h : p > 0) :
  R.2 = 4 ∧
  Q p = (p, 2 * p + 4) ∧
  area_OPQR p = p * (p + 4) ∧
  (p = 8 → area_OPQR p = 96) ∧
  (area_OPQR p = 77 → p = 7) := by
  sorry

end NUMINAMATH_CALUDE_line_problem_l491_49102


namespace NUMINAMATH_CALUDE_parents_can_catch_kolya_l491_49137

/-- Represents a point in the park --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a person in the park --/
structure Person :=
  (position : Point)
  (speed : ℝ)

/-- Represents the park with its alleys --/
structure Park :=
  (square_side : ℝ)
  (alley_length : ℝ)

/-- Checks if a point is on an alley --/
def is_on_alley (park : Park) (p : Point) : Prop :=
  (p.x = 0 ∨ p.x = park.square_side ∨ p.x = park.square_side / 2) ∨
  (p.y = 0 ∨ p.y = park.square_side ∨ p.y = park.square_side / 2)

/-- Represents the state of the chase --/
structure ChaseState :=
  (park : Park)
  (kolya : Person)
  (parent1 : Person)
  (parent2 : Person)

/-- Defines what it means for parents to catch Kolya --/
def parents_catch_kolya (state : ChaseState) : Prop :=
  ∃ (t : ℝ), t > 0 ∧ 
  ∃ (final_kolya final_parent1 final_parent2 : Point),
    is_on_alley state.park final_kolya ∧
    is_on_alley state.park final_parent1 ∧
    is_on_alley state.park final_parent2 ∧
    (final_kolya = final_parent1 ∨ final_kolya = final_parent2)

/-- The main theorem stating that parents can catch Kolya --/
theorem parents_can_catch_kolya (initial_state : ChaseState) :
  initial_state.kolya.speed = 3 * initial_state.parent1.speed ∧
  initial_state.kolya.speed = 3 * initial_state.parent2.speed ∧
  initial_state.park.square_side > 0 ∧
  initial_state.park.alley_length > 0 →
  parents_catch_kolya initial_state :=
sorry

end NUMINAMATH_CALUDE_parents_can_catch_kolya_l491_49137


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l491_49184

def polynomial (x : ℝ) : ℝ :=
  -3 * (x^8 - 2*x^5 + 4*x^3 - 6) + 5 * (x^4 + 3*x^2 - 2*x + 1) - 4 * (x^6 - 5*x + 7)

theorem sum_of_coefficients :
  polynomial 1 = 12 :=
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l491_49184


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l491_49117

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain 
  (f : ℝ → ℝ) 
  (h_odd : IsOdd f)
  (h_pos : ∀ x > 0, f x = -x + 1) :
  ∀ x < 0, f x = -x - 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l491_49117


namespace NUMINAMATH_CALUDE_rational_sum_l491_49148

theorem rational_sum (a b : ℚ) (h : |3 - a| + (b + 2)^2 = 0) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_l491_49148


namespace NUMINAMATH_CALUDE_max_value_quadratic_max_value_sum_products_l491_49124

-- Part 1
theorem max_value_quadratic (a : ℝ) (h : a > 0) :
  ∃ (max : ℝ), ∀ (x : ℝ), x > 0 → a > 2*x → x*(a - 2*x) ≤ max ∧
  ∃ (x_max : ℝ), x_max > 0 ∧ a > 2*x_max ∧ x_max*(a - 2*x_max) = max :=
sorry

-- Part 2
theorem max_value_sum_products (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 4) :
  a*b + b*c + a*c ≤ 4 ∧ ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
  a'^2 + b'^2 + c'^2 = 4 ∧ a'*b' + b'*c' + a'*c' = 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_max_value_sum_products_l491_49124


namespace NUMINAMATH_CALUDE_original_number_proof_l491_49125

/-- Given a number n formed by adding a digit h in the 10's place of 284,
    where n is divisible by 6 and h = 1, prove that the original number
    without the 10's digit is 284. -/
theorem original_number_proof (n : ℕ) (h : ℕ) :
  n = 2000 + h * 100 + 84 →
  h = 1 →
  n % 6 = 0 →
  2000 + 84 = 284 :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l491_49125


namespace NUMINAMATH_CALUDE_dvd_pack_discounted_price_l491_49174

/-- The price of a DVD pack after discount -/
def price_after_discount (original_price discount : ℕ) : ℕ :=
  original_price - discount

/-- Theorem: The price of a DVD pack after a $25 discount is $51, given that the original price is $76 -/
theorem dvd_pack_discounted_price :
  price_after_discount 76 25 = 51 := by
  sorry

end NUMINAMATH_CALUDE_dvd_pack_discounted_price_l491_49174


namespace NUMINAMATH_CALUDE_wire_length_proof_l491_49193

theorem wire_length_proof (shorter_piece longer_piece total_length : ℝ) : 
  shorter_piece = 30 ∧ 
  shorter_piece = (3/5) * longer_piece ∧
  total_length = shorter_piece + longer_piece →
  total_length = 80 := by
sorry

end NUMINAMATH_CALUDE_wire_length_proof_l491_49193


namespace NUMINAMATH_CALUDE_intersection_line_canonical_l491_49176

-- Define the planes
def plane1 (x y z : ℝ) : Prop := 3 * x + 4 * y + 3 * z + 1 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x - 4 * y - 2 * z + 4 = 0

-- Define the canonical form of the line
def canonical_line (x y z : ℝ) : Prop := (x + 1) / 4 = (y - 1/2) / 12 ∧ (y - 1/2) / 12 = z / (-20)

-- Theorem statement
theorem intersection_line_canonical : 
  ∀ x y z : ℝ, plane1 x y z ∧ plane2 x y z → canonical_line x y z :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_line_canonical_l491_49176


namespace NUMINAMATH_CALUDE_line_L_equation_ellipse_C_equation_l491_49157

-- Define the line L
def line_L (x y : ℝ) : Prop := x/4 + y/2 = 1

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

-- Theorem for line L
theorem line_L_equation :
  ∀ (a b : ℝ), a > 0 ∧ b > 0 →
  (∀ (x y : ℝ), x/a + y/b = 1 → (x = 2 ∧ y = 1)) →
  (1/2 * a * b = 4) →
  (∀ (x y : ℝ), line_L x y ↔ x/a + y/b = 1) :=
sorry

-- Theorem for ellipse C
theorem ellipse_C_equation :
  let e : ℝ := 0.8
  let c : ℝ := 4
  let a : ℝ := c / e
  let b : ℝ := Real.sqrt (a^2 - c^2)
  ∀ (x y : ℝ), ellipse_C x y ↔ x^2/a^2 + y^2/b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_line_L_equation_ellipse_C_equation_l491_49157


namespace NUMINAMATH_CALUDE_angle_between_points_after_one_second_l491_49167

/-- Represents the angular velocity of a rotating point. -/
structure AngularVelocity where
  value : ℝ
  positive : value > 0

/-- Represents a rotating point on a circle. -/
structure RotatingPoint where
  velocity : AngularVelocity

/-- Calculates the angle between two rotating points after 1 second. -/
def angleBetweenPoints (p1 p2 : RotatingPoint) : ℝ := sorry

/-- Theorem stating the angle between two rotating points after 1 second. -/
theorem angle_between_points_after_one_second 
  (p1 p2 : RotatingPoint) 
  (h1 : p1.velocity.value - p2.velocity.value = 2 * Real.pi / 60)  -- Two more revolutions per minute
  (h2 : 1 / p1.velocity.value - 1 / p2.velocity.value = 5)  -- 5 seconds faster revolution
  : angleBetweenPoints p1 p2 = 12 * Real.pi / 180 ∨ 
    angleBetweenPoints p1 p2 = 60 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_points_after_one_second_l491_49167


namespace NUMINAMATH_CALUDE_min_value_problem_l491_49159

theorem min_value_problem (i j k l m n o p : ℝ) 
  (h1 : i * j * k * l = 16) 
  (h2 : m * n * o * p = 25) : 
  (i * m)^2 + (j * n)^2 + (k * o)^2 + (l * p)^2 ≥ 160 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l491_49159


namespace NUMINAMATH_CALUDE_expression_value_l491_49147

theorem expression_value (x y z : ℝ) (hx : x = 2) (hy : y = -3) (hz : z = 1) :
  x^2 + y^2 - z^2 + 2*x*y + 2*y*z = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l491_49147


namespace NUMINAMATH_CALUDE_theater_attendance_l491_49143

/-- Proves the number of children attending a theater given total attendance and revenue --/
theorem theater_attendance (adults children : ℕ) 
  (total_attendance : adults + children = 280)
  (total_revenue : 60 * adults + 25 * children = 14000) :
  children = 80 := by sorry

end NUMINAMATH_CALUDE_theater_attendance_l491_49143


namespace NUMINAMATH_CALUDE_xNotEqual1_is_valid_l491_49133

/-- Valid conditional operators -/
inductive ConditionalOperator
  | gt  -- >
  | ge  -- >=
  | lt  -- <
  | ne  -- <>
  | le  -- <=
  | eq  -- =

/-- A conditional expression -/
structure ConditionalExpression where
  operator : ConditionalOperator
  value : ℝ

/-- Check if a conditional expression is valid -/
def isValidConditionalExpression (expr : ConditionalExpression) : Prop :=
  expr.operator ∈ [ConditionalOperator.gt, ConditionalOperator.ge, ConditionalOperator.lt, 
                   ConditionalOperator.ne, ConditionalOperator.le, ConditionalOperator.eq]

/-- The specific conditional expression "x <> 1" -/
def xNotEqual1 : ConditionalExpression :=
  { operator := ConditionalOperator.ne, value := 1 }

/-- Theorem: "x <> 1" is a valid conditional expression -/
theorem xNotEqual1_is_valid : isValidConditionalExpression xNotEqual1 := by
  sorry

end NUMINAMATH_CALUDE_xNotEqual1_is_valid_l491_49133


namespace NUMINAMATH_CALUDE_log_inequality_l491_49189

theorem log_inequality (x : ℝ) (h : x > 0) :
  9.280 * (Real.log x / Real.log 7) - Real.log 7 * (Real.log x / Real.log 3) > Real.log 0.25 / Real.log 2 ↔ 
  x < 3^(2 / (Real.log 7 / Real.log 3 - Real.log 3 / Real.log 7)) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l491_49189


namespace NUMINAMATH_CALUDE_min_value_of_expression_l491_49140

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 * y * (4*x + 3*y) = 3) : 
  ∃ (min : ℝ), min = 2 * Real.sqrt 3 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 
  x'^2 * y' * (4*x' + 3*y') = 3 → 2*x' + 3*y' ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l491_49140


namespace NUMINAMATH_CALUDE_range_of_a_l491_49129

/-- Given sets A and B, where A is [-2, 4) and B is {x | x^2 - ax - 4 ≤ 0},
    if B is a subset of A, then a is in the range [0, 3). -/
theorem range_of_a (a : ℝ) : 
  let A : Set ℝ := {x | -2 ≤ x ∧ x < 4}
  let B : Set ℝ := {x | x^2 - a*x - 4 ≤ 0}
  B ⊆ A → 0 ≤ a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l491_49129


namespace NUMINAMATH_CALUDE_charge_per_mile_calculation_l491_49172

/-- Proves that the charge per mile is $0.25 given the rental fee, total amount paid, and miles driven -/
theorem charge_per_mile_calculation (rental_fee total_paid miles_driven : ℚ) 
  (h1 : rental_fee = 20.99)
  (h2 : total_paid = 95.74)
  (h3 : miles_driven = 299) :
  (total_paid - rental_fee) / miles_driven = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_charge_per_mile_calculation_l491_49172


namespace NUMINAMATH_CALUDE_inequality_equivalence_l491_49155

theorem inequality_equivalence (x y : ℝ) (h : x ≠ 0) :
  (y - x)^2 < x^2 ↔ y > 0 ∧ y < 2*x := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l491_49155


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l491_49101

theorem max_value_cos_sin (x : Real) : 3 * Real.cos x + Real.sin x ≤ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l491_49101


namespace NUMINAMATH_CALUDE_range_of_x_l491_49145

def p (x : ℝ) := Real.log (x^2 - 2*x - 2) ≥ 0

def q (x : ℝ) := 0 < x ∧ x < 4

theorem range_of_x (x : ℝ) (hp : p x) (hq : ¬q x) : x ≥ 4 ∨ x ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l491_49145


namespace NUMINAMATH_CALUDE_division_theorem_l491_49161

theorem division_theorem (dividend divisor remainder quotient : ℕ) : 
  dividend = 141 →
  divisor = 17 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  quotient = 8 := by
sorry

end NUMINAMATH_CALUDE_division_theorem_l491_49161


namespace NUMINAMATH_CALUDE_tuzik_meets_ivan_l491_49135

/-- The time when Tuzik reaches Ivan -/
def meeting_time : Real :=
  -- Define the meeting time as 47 minutes after 12:00
  47

/-- Proof that Tuzik reaches Ivan at the calculated meeting time -/
theorem tuzik_meets_ivan (total_distance : Real) (ivan_speed : Real) (tuzik_speed : Real) 
  (ivan_start_time : Real) (tuzik_start_time : Real) :
  total_distance = 12000 →  -- 12 km in meters
  ivan_speed = 1 →          -- 1 m/s
  tuzik_speed = 9 →         -- 9 m/s
  ivan_start_time = 0 →     -- 12:00 represented as 0 minutes
  tuzik_start_time = 30 →   -- 12:30 represented as 30 minutes
  meeting_time = 47 := by
  sorry

#check tuzik_meets_ivan

end NUMINAMATH_CALUDE_tuzik_meets_ivan_l491_49135


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l491_49168

theorem cubic_sum_theorem (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a * b + a * c + b * c = 2) 
  (h3 : a * b * c = 3) : 
  a^3 + b^3 + c^3 = 9 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l491_49168


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l491_49154

theorem smallest_integer_satisfying_inequality : 
  ∃ x : ℤ, (∀ y : ℤ, (5 : ℚ) / 8 < (y + 3 : ℚ) / 15 → x ≤ y) ∧ (5 : ℚ) / 8 < (x + 3 : ℚ) / 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l491_49154


namespace NUMINAMATH_CALUDE_range_of_m_l491_49158

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x > 0, m^2 + 2*m - 1 ≤ x + 1/x

def q (m : ℝ) : Prop := ∀ x, (5 - m^2)^x < (5 - m^2)^(x + 1)

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) →
  (-3 ≤ m ∧ m ≤ -2) ∨ (1 < m ∧ m < 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l491_49158


namespace NUMINAMATH_CALUDE_certain_number_proof_l491_49185

theorem certain_number_proof (A B C X : ℝ) : 
  A / B = 5 / 6 →
  B / C = 6 / 8 →
  C = 42 →
  A + C = B + X →
  X = 36.75 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l491_49185


namespace NUMINAMATH_CALUDE_problem_statement_l491_49142

theorem problem_statement (x y : ℚ) 
  (h1 : 3 * x + 4 * y = 0)
  (h2 : x = y + 3) :
  5 * y = -45 / 7 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l491_49142


namespace NUMINAMATH_CALUDE_ball_arrangements_count_l491_49162

def num_red_balls : ℕ := 2
def num_yellow_balls : ℕ := 3
def num_white_balls : ℕ := 4
def total_balls : ℕ := num_red_balls + num_yellow_balls + num_white_balls

theorem ball_arrangements_count :
  (Nat.factorial total_balls) / (Nat.factorial num_red_balls * Nat.factorial num_yellow_balls * Nat.factorial num_white_balls) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_ball_arrangements_count_l491_49162


namespace NUMINAMATH_CALUDE_complex_square_one_plus_i_l491_49139

theorem complex_square_one_plus_i : (1 + Complex.I) ^ 2 = 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_one_plus_i_l491_49139


namespace NUMINAMATH_CALUDE_min_socks_for_different_colors_l491_49149

theorem min_socks_for_different_colors :
  let total_blue_socks : ℕ := 6
  let total_red_socks : ℕ := 6
  let min_socks : ℕ := 7
  ∀ (selected : ℕ), selected ≥ min_socks →
    ∃ (blue red : ℕ), blue + red = selected ∧
      blue ≤ total_blue_socks ∧
      red ≤ total_red_socks ∧
      (blue > 0 ∧ red > 0) :=
by sorry

end NUMINAMATH_CALUDE_min_socks_for_different_colors_l491_49149


namespace NUMINAMATH_CALUDE_book_probabilities_l491_49136

/-- Represents the book collection with given properties -/
structure BookCollection where
  total : ℕ
  liberal_arts : ℕ
  hardcover : ℕ
  softcover_science : ℕ
  total_eq : total = 100
  liberal_arts_eq : liberal_arts = 40
  hardcover_eq : hardcover = 70
  softcover_science_eq : softcover_science = 20

/-- Calculates the probability of selecting a liberal arts hardcover book -/
def prob_liberal_arts_hardcover (bc : BookCollection) : ℚ :=
  (bc.hardcover - bc.softcover_science : ℚ) / bc.total

/-- Calculates the probability of selecting a liberal arts book then a hardcover book -/
def prob_liberal_arts_then_hardcover (bc : BookCollection) : ℚ :=
  (bc.liberal_arts : ℚ) / bc.total * (bc.hardcover : ℚ) / bc.total

/-- Main theorem stating the probabilities -/
theorem book_probabilities (bc : BookCollection) :
    prob_liberal_arts_hardcover bc = 3/10 ∧
    prob_liberal_arts_then_hardcover bc = 28/100 := by
  sorry

end NUMINAMATH_CALUDE_book_probabilities_l491_49136


namespace NUMINAMATH_CALUDE_pasture_fence_posts_l491_49199

/-- Calculates the number of posts needed for a given length of fence -/
def posts_for_length (length : ℕ) (post_spacing : ℕ) : ℕ :=
  (length / post_spacing) + 1

/-- The pasture dimensions -/
def pasture_width : ℕ := 36
def pasture_length : ℕ := 75

/-- The spacing between posts -/
def post_spacing : ℕ := 15

/-- The total number of posts required for the pasture -/
def total_posts : ℕ :=
  posts_for_length pasture_width post_spacing +
  2 * (posts_for_length pasture_length post_spacing - 1)

theorem pasture_fence_posts :
  total_posts = 14 := by sorry

end NUMINAMATH_CALUDE_pasture_fence_posts_l491_49199


namespace NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l491_49178

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The statement that "a = 0" is a necessary but not sufficient condition for "a + bi to be purely imaginary". -/
theorem a_zero_necessary_not_sufficient (a b : ℝ) :
  (∀ z : ℂ, z = a + b * I → is_purely_imaginary z → a = 0) ∧
  (∃ z : ℂ, z = a + b * I ∧ a = 0 ∧ ¬is_purely_imaginary z) :=
sorry

end NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l491_49178


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l491_49100

theorem absolute_value_inequality (a b : ℝ) (h : |a - b| > 2) :
  ∀ x : ℝ, |x - a| + |x - b| > 2 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l491_49100
