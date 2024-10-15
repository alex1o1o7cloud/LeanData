import Mathlib

namespace NUMINAMATH_CALUDE_farrah_matchsticks_l2109_210982

/-- Calculates the total number of matchsticks given the number of boxes, matchboxes per box, and sticks per matchbox. -/
def total_matchsticks (x y z : ℕ) : ℕ := x * y * z

/-- Theorem stating that for the given values, the total number of matchsticks is 300,000. -/
theorem farrah_matchsticks :
  let x : ℕ := 10
  let y : ℕ := 50
  let z : ℕ := 600
  total_matchsticks x y z = 300000 := by
  sorry

end NUMINAMATH_CALUDE_farrah_matchsticks_l2109_210982


namespace NUMINAMATH_CALUDE_sphere_speed_l2109_210927

-- Define constants
def Q : Real := -20e-6
def q : Real := 50e-6
def AB : Real := 2
def AC : Real := 3
def m : Real := 0.2
def g : Real := 10
def k : Real := 9e9

-- Define the theorem
theorem sphere_speed (BC : Real) (v : Real) 
  (h1 : BC^2 = AC^2 - AB^2)  -- Pythagorean theorem
  (h2 : v^2 = (2/m) * (k*Q*q * (1/AB - 1/BC) + m*g*AB)) : -- Energy conservation
  v = 5 := by
sorry

end NUMINAMATH_CALUDE_sphere_speed_l2109_210927


namespace NUMINAMATH_CALUDE_park_tree_density_l2109_210922

/-- Given a rectangular park with length, width, and number of trees, 
    calculate the area occupied by each tree. -/
def area_per_tree (length width num_trees : ℕ) : ℚ :=
  (length * width : ℚ) / num_trees

/-- Theorem stating that in a park of 1000 feet long and 2000 feet wide, 
    with 100,000 trees, each tree occupies 20 square feet. -/
theorem park_tree_density :
  area_per_tree 1000 2000 100000 = 20 := by
  sorry

#eval area_per_tree 1000 2000 100000

end NUMINAMATH_CALUDE_park_tree_density_l2109_210922


namespace NUMINAMATH_CALUDE_fraction_equality_l2109_210904

theorem fraction_equality (a b : ℝ) (h : 1/a - 1/b = 1) :
  (a + a*b - b) / (a - 2*a*b - b) = 0 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2109_210904


namespace NUMINAMATH_CALUDE_theo_daily_consumption_l2109_210943

/-- Represents the daily water consumption of the three siblings -/
structure SiblingWaterConsumption where
  theo : ℕ
  mason : ℕ
  roxy : ℕ

/-- The total water consumption of the siblings over a week -/
def weeklyConsumption (swc : SiblingWaterConsumption) : ℕ :=
  7 * (swc.theo + swc.mason + swc.roxy)

/-- Theorem stating Theo's daily water consumption -/
theorem theo_daily_consumption :
  ∃ (swc : SiblingWaterConsumption),
    swc.mason = 7 ∧
    swc.roxy = 9 ∧
    weeklyConsumption swc = 168 ∧
    swc.theo = 8 := by
  sorry

end NUMINAMATH_CALUDE_theo_daily_consumption_l2109_210943


namespace NUMINAMATH_CALUDE_system_solution_l2109_210912

theorem system_solution (x y b : ℝ) : 
  (5 * x + y = b) → 
  (3 * x + 4 * y = 3 * b) → 
  (x = 3) → 
  (b = 60) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2109_210912


namespace NUMINAMATH_CALUDE_galya_number_puzzle_l2109_210944

theorem galya_number_puzzle (k : ℝ) (N : ℝ) : 
  (((k * N + N) / N) - N = k - 7729) → N = 7730 :=
by
  sorry

end NUMINAMATH_CALUDE_galya_number_puzzle_l2109_210944


namespace NUMINAMATH_CALUDE_not_prime_4k4_plus_1_and_k4_plus_4_l2109_210936

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem not_prime_4k4_plus_1_and_k4_plus_4 (k : ℕ) : 
  ¬(is_prime (4 * k^4 + 1)) ∧ ¬(is_prime (k^4 + 4)) := by
  sorry


end NUMINAMATH_CALUDE_not_prime_4k4_plus_1_and_k4_plus_4_l2109_210936


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l2109_210974

theorem complex_expression_evaluation :
  let i : ℂ := Complex.I
  ((2 + i) * (3 + i)) / (1 + i) = 5 := by sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l2109_210974


namespace NUMINAMATH_CALUDE_matrix_determinant_and_fraction_sum_l2109_210994

theorem matrix_determinant_and_fraction_sum (p q r : ℝ) :
  let M := ![![p, 2*q, r],
             ![q, r, p],
             ![r, p, q]]
  Matrix.det M = 0 →
  (p / (2*q + r) + 2*q / (p + r) + r / (p + q) = -4) ∨
  (p / (2*q + r) + 2*q / (p + r) + r / (p + q) = 11/6) :=
by sorry

end NUMINAMATH_CALUDE_matrix_determinant_and_fraction_sum_l2109_210994


namespace NUMINAMATH_CALUDE_ice_cream_frozen_yoghurt_cost_difference_l2109_210924

/-- Calculates the difference in cost between ice cream and frozen yoghurt purchases -/
theorem ice_cream_frozen_yoghurt_cost_difference :
  let ice_cream_cartons : ℕ := 10
  let frozen_yoghurt_cartons : ℕ := 4
  let ice_cream_price : ℕ := 4
  let frozen_yoghurt_price : ℕ := 1
  let ice_cream_total_cost := ice_cream_cartons * ice_cream_price
  let frozen_yoghurt_total_cost := frozen_yoghurt_cartons * frozen_yoghurt_price
  ice_cream_total_cost - frozen_yoghurt_total_cost = 36 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_frozen_yoghurt_cost_difference_l2109_210924


namespace NUMINAMATH_CALUDE_units_digit_of_power_of_six_l2109_210998

theorem units_digit_of_power_of_six (n : ℕ) : (6^n) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_of_six_l2109_210998


namespace NUMINAMATH_CALUDE_applicants_theorem_l2109_210970

/-- The number of applicants with less than 4 years' experience and no degree -/
def applicants_less_exp_no_degree (total : ℕ) (exp : ℕ) (deg : ℕ) (exp_and_deg : ℕ) : ℕ :=
  total - (exp + deg - exp_and_deg)

theorem applicants_theorem (total : ℕ) (exp : ℕ) (deg : ℕ) (exp_and_deg : ℕ)
  (h_total : total = 30)
  (h_exp : exp = 10)
  (h_deg : deg = 18)
  (h_exp_and_deg : exp_and_deg = 9) :
  applicants_less_exp_no_degree total exp deg exp_and_deg = 11 :=
by
  sorry

#eval applicants_less_exp_no_degree 30 10 18 9

end NUMINAMATH_CALUDE_applicants_theorem_l2109_210970


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l2109_210989

def original_soup_price : ℚ := 7.50 / 3
def original_bread_price : ℚ := 5 / 2
def new_soup_price : ℚ := 8 / 4
def new_bread_price : ℚ := 6 / 3

def original_bundle_avg : ℚ := (original_soup_price + original_bread_price) / 2
def new_bundle_avg : ℚ := (new_soup_price + new_bread_price) / 2

theorem price_decrease_percentage :
  (original_bundle_avg - new_bundle_avg) / original_bundle_avg * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l2109_210989


namespace NUMINAMATH_CALUDE_value_of_equation_l2109_210932

theorem value_of_equation (x y V : ℝ) 
  (eq1 : x + |x| + y = V)
  (eq2 : x + |y| - y = 6)
  (eq3 : x + y = 12) :
  V = 18 := by
  sorry

end NUMINAMATH_CALUDE_value_of_equation_l2109_210932


namespace NUMINAMATH_CALUDE_runner_speed_proof_l2109_210978

def total_distance : ℝ := 1000
def total_time : ℝ := 380
def first_segment_distance : ℝ := 720
def first_segment_speed : ℝ := 3

def second_segment_speed : ℝ := 2

theorem runner_speed_proof :
  let first_segment_time := first_segment_distance / first_segment_speed
  let second_segment_distance := total_distance - first_segment_distance
  let second_segment_time := total_time - first_segment_time
  second_segment_speed = second_segment_distance / second_segment_time :=
by
  sorry

end NUMINAMATH_CALUDE_runner_speed_proof_l2109_210978


namespace NUMINAMATH_CALUDE_minimal_sum_distances_l2109_210972

noncomputable section

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Point in 2D plane -/
def Point := ℝ × ℝ

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Inverse point with respect to a circle -/
def inverse_point (c : Circle) (p : Point) : Point := sorry

/-- Line defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Perpendicular bisector of two points -/
def perpendicular_bisector (p q : Point) : Line := sorry

/-- Intersection point of a line and a circle -/
def line_circle_intersection (l : Line) (c : Circle) : Option Point := sorry

/-- Theorem: Minimal sum of distances from two fixed points to a point on a circle -/
theorem minimal_sum_distances (c : Circle) (p q : Point) 
  (h1 : distance c.center p = distance c.center q) 
  (h2 : distance c.center p < c.radius ∧ distance c.center q < c.radius) :
  ∃ z : Point, 
    (distance c.center z = c.radius) ∧ 
    (∀ w : Point, distance c.center w = c.radius → 
      distance p z + distance q z ≤ distance p w + distance q w) :=
sorry

end

end NUMINAMATH_CALUDE_minimal_sum_distances_l2109_210972


namespace NUMINAMATH_CALUDE_solve_equation_for_m_l2109_210946

theorem solve_equation_for_m : ∃ m : ℝ, (m - 5)^3 = (1/27)⁻¹ ∧ m = 8 := by sorry

end NUMINAMATH_CALUDE_solve_equation_for_m_l2109_210946


namespace NUMINAMATH_CALUDE_zeros_of_f_l2109_210947

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 9

-- Theorem stating that the zeros of f(x) are ±3
theorem zeros_of_f :
  ∀ x : ℝ, f x = 0 ↔ x = 3 ∨ x = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_zeros_of_f_l2109_210947


namespace NUMINAMATH_CALUDE_exists_valid_pairs_l2109_210933

def digits_ge_6 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≥ 6

def a (k : ℕ) : ℕ :=
  (10^k - 3) * 10^2 + 97

theorem exists_valid_pairs :
  ∃ n : ℕ, ∀ k ≥ n, 
    digits_ge_6 (a k) ∧ 
    digits_ge_6 7 ∧ 
    digits_ge_6 (a k * 7) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_pairs_l2109_210933


namespace NUMINAMATH_CALUDE_smallest_positive_d_l2109_210937

theorem smallest_positive_d : ∃ d : ℝ, d > 0 ∧
  (5 * Real.sqrt 5)^2 + (d + 5)^2 = (5 * d)^2 ∧
  ∀ d' : ℝ, d' > 0 → (5 * Real.sqrt 5)^2 + (d' + 5)^2 = (5 * d')^2 → d ≤ d' :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_d_l2109_210937


namespace NUMINAMATH_CALUDE_boat_distribution_problem_l2109_210987

/-- Represents the boat distribution problem from "Nine Chapters on the Mathematical Art" --/
theorem boat_distribution_problem (x : ℕ) : 
  (∀ (total_boats : ℕ) (large_boat_capacity : ℕ) (small_boat_capacity : ℕ) (total_students : ℕ),
    total_boats = 8 ∧ 
    large_boat_capacity = 6 ∧ 
    small_boat_capacity = 4 ∧ 
    total_students = 38 ∧ 
    x ≤ total_boats ∧
    x * small_boat_capacity + (total_boats - x) * large_boat_capacity = total_students) →
  4 * x + 6 * (8 - x) = 38 :=
by sorry

end NUMINAMATH_CALUDE_boat_distribution_problem_l2109_210987


namespace NUMINAMATH_CALUDE_marble_selection_ways_l2109_210954

theorem marble_selection_ways (n m : ℕ) (h1 : n = 9) (h2 : m = 4) :
  Nat.choose n m = 126 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l2109_210954


namespace NUMINAMATH_CALUDE_product_inequality_l2109_210903

theorem product_inequality (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0) (h : A * B * C = 1) :
  (A - 1 + 1/B) * (B - 1 + 1/C) * (C - 1 + 1/A) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l2109_210903


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_irrational_l2109_210902

/-- Given two consecutive even integers and their sum, the square root of the sum of their squares is irrational -/
theorem sqrt_sum_squares_irrational (x : ℤ) : 
  let a : ℤ := 2 * x
  let b : ℤ := 2 * x + 2
  let c : ℤ := a + b
  let D : ℤ := a^2 + b^2 + c^2
  Irrational (Real.sqrt (D : ℝ)) := by
sorry


end NUMINAMATH_CALUDE_sqrt_sum_squares_irrational_l2109_210902


namespace NUMINAMATH_CALUDE_equation_solution_l2109_210956

theorem equation_solution : ∃! x : ℚ, (x^2 + 2*x + 3) / (x + 4) = x + 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2109_210956


namespace NUMINAMATH_CALUDE_min_value_sqrt_inequality_l2109_210941

theorem min_value_sqrt_inequality :
  ∃ (a : ℝ), (∀ (x y : ℝ), x > 0 ∧ y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) ∧
  (∀ (b : ℝ), (∀ (x y : ℝ), x > 0 ∧ y > 0 → Real.sqrt x + Real.sqrt y ≤ b * Real.sqrt (x + y)) → a ≤ b) ∧
  a = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_inequality_l2109_210941


namespace NUMINAMATH_CALUDE_prop_analysis_l2109_210988

-- Define the original proposition
def original_prop (x y : ℝ) : Prop := (x + y = 5) → (x = 3 ∧ y = 2)

-- Define the converse
def converse (x y : ℝ) : Prop := (x = 3 ∧ y = 2) → (x + y = 5)

-- Define the inverse
def inverse (x y : ℝ) : Prop := (x + y ≠ 5) → (x ≠ 3 ∨ y ≠ 2)

-- Define the contrapositive
def contrapositive (x y : ℝ) : Prop := (x ≠ 3 ∨ y ≠ 2) → (x + y ≠ 5)

-- Theorem stating the truth values of converse, inverse, and contrapositive
theorem prop_analysis :
  (∀ x y : ℝ, converse x y) ∧
  (¬ ∀ x y : ℝ, inverse x y) ∧
  (∀ x y : ℝ, contrapositive x y) :=
by sorry

end NUMINAMATH_CALUDE_prop_analysis_l2109_210988


namespace NUMINAMATH_CALUDE_rectangle_area_change_l2109_210913

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let new_length := 1.1 * L
  let new_breadth := 0.9 * B
  let original_area := L * B
  let new_area := new_length * new_breadth
  new_area = 0.99 * original_area := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l2109_210913


namespace NUMINAMATH_CALUDE_complement_A_in_U_l2109_210965

def U : Set ℕ := {x | (x + 1) / (x - 5) ≤ 0}

def A : Set ℕ := {1, 2, 4}

theorem complement_A_in_U : 
  {x ∈ U | x ∉ A} = {0, 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l2109_210965


namespace NUMINAMATH_CALUDE_loaves_sold_is_one_l2109_210958

/-- Represents the baker's sales and prices --/
structure BakerSales where
  usual_pastries : ℕ
  usual_bread : ℕ
  today_pastries : ℕ
  pastry_price : ℚ
  bread_price : ℚ
  sales_difference : ℚ

/-- Calculates the number of loaves of bread sold today --/
def loaves_sold_today (s : BakerSales) : ℚ :=
  ((s.usual_pastries * s.pastry_price + s.usual_bread * s.bread_price) -
   (s.today_pastries * s.pastry_price + s.sales_difference)) / s.bread_price

/-- Theorem stating that the number of loaves sold today is 1 --/
theorem loaves_sold_is_one (s : BakerSales)
  (h1 : s.usual_pastries = 20)
  (h2 : s.usual_bread = 10)
  (h3 : s.today_pastries = 14)
  (h4 : s.pastry_price = 2)
  (h5 : s.bread_price = 4)
  (h6 : s.sales_difference = 48) :
  loaves_sold_today s = 1 := by
  sorry

#eval loaves_sold_today {
  usual_pastries := 20,
  usual_bread := 10,
  today_pastries := 14,
  pastry_price := 2,
  bread_price := 4,
  sales_difference := 48
}

end NUMINAMATH_CALUDE_loaves_sold_is_one_l2109_210958


namespace NUMINAMATH_CALUDE_smallest_rational_l2109_210975

theorem smallest_rational (a b c d : ℚ) (ha : a = 1) (hb : b = 0) (hc : c = -1/2) (hd : d = -3) :
  d < c ∧ c < b ∧ b < a :=
by sorry

end NUMINAMATH_CALUDE_smallest_rational_l2109_210975


namespace NUMINAMATH_CALUDE_sum_of_sequences_l2109_210926

def sequence1 : List Nat := [2, 12, 22, 32, 42]
def sequence2 : List Nat := [10, 20, 30, 40, 50]

theorem sum_of_sequences : (sequence1.sum + sequence2.sum = 260) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_l2109_210926


namespace NUMINAMATH_CALUDE_ones_digit_of_19_power_l2109_210951

theorem ones_digit_of_19_power (n : ℕ) : 19^(19 * (13^13)) ≡ 9 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_19_power_l2109_210951


namespace NUMINAMATH_CALUDE_cloth_cost_price_l2109_210969

theorem cloth_cost_price
  (meters_sold : ℕ)
  (selling_price : ℕ)
  (loss_per_meter : ℕ)
  (h1 : meters_sold = 450)
  (h2 : selling_price = 18000)
  (h3 : loss_per_meter = 5) :
  (selling_price + meters_sold * loss_per_meter) / meters_sold = 45 := by
sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l2109_210969


namespace NUMINAMATH_CALUDE_molecular_weight_independent_of_moles_l2109_210996

/-- The molecular weight of an acid in g/mol -/
def molecular_weight : ℝ := 408

/-- The number of moles of the acid -/
def moles : ℝ := 6

/-- Theorem stating that the molecular weight is independent of the number of moles -/
theorem molecular_weight_independent_of_moles :
  molecular_weight = molecular_weight := by sorry

end NUMINAMATH_CALUDE_molecular_weight_independent_of_moles_l2109_210996


namespace NUMINAMATH_CALUDE_M_intersect_N_l2109_210992

def M : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

theorem M_intersect_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l2109_210992


namespace NUMINAMATH_CALUDE_min_cost_50_percent_alloy_l2109_210966

/-- Represents a gold alloy with its gold percentage and cost per ounce -/
structure GoldAlloy where
  percentage : Rat
  cost : Rat

/-- Theorem stating the minimum cost per ounce to create a 50% gold alloy -/
theorem min_cost_50_percent_alloy 
  (alloy40 : GoldAlloy) 
  (alloy60 : GoldAlloy)
  (alloy90 : GoldAlloy)
  (h1 : alloy40.percentage = 40/100)
  (h2 : alloy60.percentage = 60/100)
  (h3 : alloy90.percentage = 90/100)
  (h4 : alloy40.cost = 200)
  (h5 : alloy60.cost = 300)
  (h6 : alloy90.cost = 400) :
  ∃ (x y z : Rat),
    x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
    (x * alloy40.percentage + y * alloy60.percentage + z * alloy90.percentage) / (x + y + z) = 1/2 ∧
    (x * alloy40.cost + y * alloy60.cost + z * alloy90.cost) / (x + y + z) = 240 ∧
    ∀ (a b c : Rat),
      a ≥ 0 → b ≥ 0 → c ≥ 0 →
      (a * alloy40.percentage + b * alloy60.percentage + c * alloy90.percentage) / (a + b + c) = 1/2 →
      (a * alloy40.cost + b * alloy60.cost + c * alloy90.cost) / (a + b + c) ≥ 240 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_50_percent_alloy_l2109_210966


namespace NUMINAMATH_CALUDE_men_to_women_ratio_l2109_210900

/-- Proves that the ratio of men to women workers is 1:3 given the problem conditions -/
theorem men_to_women_ratio (woman_wage : ℝ) (num_women : ℕ) : 
  let man_wage := 2 * woman_wage
  let women_earnings := num_women * woman_wage * 30
  let men_earnings := (num_women / 3) * man_wage * 20
  women_earnings = 21600 → men_earnings = 14400 := by
  sorry

end NUMINAMATH_CALUDE_men_to_women_ratio_l2109_210900


namespace NUMINAMATH_CALUDE_parabola_line_distance_l2109_210957

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Line structure -/
structure Line where
  k : ℝ

/-- Problem statement -/
theorem parabola_line_distance (parab : Parabola) (l : Line) : 
  (parab.p / 2 = 4) →
  (∃ x y : ℝ, x^2 = 2 * parab.p * y ∧ y = l.k * (x + 1)) →
  (∀ x y : ℝ, x^2 = 2 * parab.p * y ∧ y = l.k * (x + 1) → x = -1) →
  let focus_distance := dist (0, parab.p / 2) (x, l.k * (x + 1))
  (∃ x : ℝ, focus_distance = 1 ∨ focus_distance = 4 ∨ focus_distance = Real.sqrt 17) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_distance_l2109_210957


namespace NUMINAMATH_CALUDE_inverse_variation_cube_and_sqrt_l2109_210979

/-- Given that x^3 varies inversely with √x, prove that y = 1/16384 when x = 64, 
    given that y = 16 when x = 4 -/
theorem inverse_variation_cube_and_sqrt (y : ℝ → ℝ) :
  (∀ x : ℝ, x > 0 → ∃ k : ℝ, y x * (x^3 * x.sqrt) = k) →
  y 4 = 16 →
  y 64 = 1 / 16384 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_and_sqrt_l2109_210979


namespace NUMINAMATH_CALUDE_toll_constant_value_l2109_210960

/-- Represents the toll formula for a bridge -/
def toll_formula (constant : ℝ) (x : ℕ) : ℝ := constant + 0.50 * (x - 2)

/-- Calculates the number of axles for a truck given its wheel configuration -/
def calculate_axles (front_wheels : ℕ) (other_wheels : ℕ) : ℕ :=
  1 + (other_wheels / 4)

theorem toll_constant_value :
  ∃ (constant : ℝ),
    let x := calculate_axles 2 16
    toll_formula constant x = 4 ∧ constant = 2.50 := by
  sorry

end NUMINAMATH_CALUDE_toll_constant_value_l2109_210960


namespace NUMINAMATH_CALUDE_line_perpendicular_to_parallel_plane_l2109_210952

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_parallel_plane
  (m : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : parallel α β) :
  perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_parallel_plane_l2109_210952


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l2109_210905

theorem ratio_of_sum_and_difference (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a + b = 5 * (a - b)) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l2109_210905


namespace NUMINAMATH_CALUDE_complex_expression_value_l2109_210910

theorem complex_expression_value : 
  let x : ℂ := Complex.exp (Complex.I * (2 * Real.pi / 9))
  (2 * x + x^3) * (2 * x^3 + x^9) * (2 * x^6 + x^18) * 
  (2 * x^2 + x^6) * (2 * x^5 + x^15) * (2 * x^7 + x^21) = 557 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_value_l2109_210910


namespace NUMINAMATH_CALUDE_cookie_problem_indeterminate_l2109_210901

/-- Represents the number of cookies Paco had and ate --/
structure CookieCount where
  initialSweet : ℕ
  initialSalty : ℕ
  eatenSweet : ℕ
  eatenSalty : ℕ

/-- Represents the conditions of the cookie problem --/
def CookieProblem (c : CookieCount) : Prop :=
  c.initialSalty = 6 ∧
  c.eatenSweet = 20 ∧
  c.eatenSalty = 34 ∧
  c.eatenSalty = c.eatenSweet + 14

theorem cookie_problem_indeterminate :
  ∀ (c : CookieCount), CookieProblem c →
    (c.initialSweet ≥ 20 ∧
     ∀ (n : ℕ), n ≥ 20 → ∃ (c' : CookieCount), CookieProblem c' ∧ c'.initialSweet = n) :=
by sorry

#check cookie_problem_indeterminate

end NUMINAMATH_CALUDE_cookie_problem_indeterminate_l2109_210901


namespace NUMINAMATH_CALUDE_students_per_classroom_l2109_210955

/-- Given a school trip scenario, calculate the number of students per classroom. -/
theorem students_per_classroom
  (num_classrooms : ℕ)
  (seats_per_bus : ℕ)
  (num_buses : ℕ)
  (h1 : num_classrooms = 67)
  (h2 : seats_per_bus = 6)
  (h3 : num_buses = 737) :
  (num_buses * seats_per_bus) / num_classrooms = 66 :=
by sorry

end NUMINAMATH_CALUDE_students_per_classroom_l2109_210955


namespace NUMINAMATH_CALUDE_line_circle_intersection_point_on_line_point_inside_circle_l2109_210917

/-- The line y = kx + 1 intersects the circle x^2 + y^2 = 2 but doesn't pass through its center -/
theorem line_circle_intersection (k : ℝ) : 
  ∃ (x y : ℝ), y = k * x + 1 ∧ x^2 + y^2 = 2 ∧ (x ≠ 0 ∨ y ≠ 0) := by
  sorry

/-- The point (0, 1) is always on the line y = kx + 1 -/
theorem point_on_line (k : ℝ) : k * 0 + 1 = 1 := by
  sorry

/-- The point (0, 1) is inside the circle x^2 + y^2 = 2 -/
theorem point_inside_circle : 0^2 + 1^2 < 2 := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_point_on_line_point_inside_circle_l2109_210917


namespace NUMINAMATH_CALUDE_honey_barrel_problem_l2109_210953

theorem honey_barrel_problem (total_weight : ℝ) (half_removed_weight : ℝ) 
  (h1 : total_weight = 56)
  (h2 : half_removed_weight = 34) :
  ∃ (honey_weight barrel_weight : ℝ),
    honey_weight = 44 ∧
    barrel_weight = 12 ∧
    honey_weight + barrel_weight = total_weight ∧
    honey_weight / 2 + barrel_weight = half_removed_weight := by
  sorry

end NUMINAMATH_CALUDE_honey_barrel_problem_l2109_210953


namespace NUMINAMATH_CALUDE_ratio_p_to_r_l2109_210925

theorem ratio_p_to_r (p q r s : ℚ) 
  (h1 : p / q = 5 / 4)
  (h2 : r / s = 3 / 2)
  (h3 : s / q = 1 / 5) :
  p / r = 25 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_p_to_r_l2109_210925


namespace NUMINAMATH_CALUDE_isosceles_base_length_isosceles_x_bounds_l2109_210945

/-- An isosceles triangle with perimeter 20 -/
structure IsoscelesTriangle where
  /-- Length of the equal sides -/
  x : ℝ
  /-- Length of the base -/
  y : ℝ
  /-- The triangle has perimeter 20 -/
  perimeter_eq : 2 * x + y = 20
  /-- The equal sides are longer than 5 and shorter than 10 -/
  x_bounds : 5 < x ∧ x < 10

/-- The base length of an isosceles triangle with perimeter 20 is 20 - 2x -/
theorem isosceles_base_length (t : IsoscelesTriangle) : t.y = 20 - 2 * t.x := by
  sorry

/-- The base length formula is valid only when 5 < x < 10 -/
theorem isosceles_x_bounds (t : IsoscelesTriangle) : 5 < t.x ∧ t.x < 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_base_length_isosceles_x_bounds_l2109_210945


namespace NUMINAMATH_CALUDE_range_of_m_l2109_210997

/-- The function f(x) = x^2 - 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

/-- Set A: range of a for which f(x) has no real roots -/
def A : Set ℝ := {a | ∀ x, f a x ≠ 0}

/-- Set B: range of a for which f(x) is not monotonic on (m, m+3) -/
def B (m : ℝ) : Set ℝ := {a | ∃ x y, m < x ∧ x < y ∧ y < m + 3 ∧ (f a x - f a y) * (x - y) < 0}

/-- Theorem: If x ∈ A is a sufficient but not necessary condition for x ∈ B, 
    then -2 ≤ m ≤ -1 -/
theorem range_of_m (m : ℝ) : 
  (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) → -2 ≤ m ∧ m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2109_210997


namespace NUMINAMATH_CALUDE_triangles_in_3x7_rectangle_l2109_210938

/-- The number of small triangles created by cutting a rectangle --/
def num_triangles (length width : ℕ) : ℕ :=
  let total_squares := length * width
  let corner_squares := 4
  let cut_squares := total_squares - corner_squares
  let triangles_per_square := 4
  cut_squares * triangles_per_square

/-- Theorem stating the number of triangles for a 3x7 rectangle --/
theorem triangles_in_3x7_rectangle :
  num_triangles 3 7 = 68 := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_3x7_rectangle_l2109_210938


namespace NUMINAMATH_CALUDE_uncool_relatives_count_l2109_210923

/-- Given a club with the following characteristics:
  * 50 total people
  * 25 people with cool dads
  * 28 people with cool moms
  * 10 people with cool siblings
  * 15 people with both cool dads and cool moms
  * 5 people with both cool dads and cool siblings
  * 7 people with both cool moms and cool siblings
  * 3 people with cool dads, cool moms, and cool siblings
Prove that the number of people with all uncool relatives is 11. -/
theorem uncool_relatives_count (total : Nat) (cool_dad : Nat) (cool_mom : Nat) (cool_sibling : Nat)
    (cool_dad_and_mom : Nat) (cool_dad_and_sibling : Nat) (cool_mom_and_sibling : Nat)
    (cool_all : Nat) (h1 : total = 50) (h2 : cool_dad = 25) (h3 : cool_mom = 28)
    (h4 : cool_sibling = 10) (h5 : cool_dad_and_mom = 15) (h6 : cool_dad_and_sibling = 5)
    (h7 : cool_mom_and_sibling = 7) (h8 : cool_all = 3) :
  total - (cool_dad + cool_mom + cool_sibling - cool_dad_and_mom - cool_dad_and_sibling
           - cool_mom_and_sibling + cool_all) = 11 := by
  sorry

end NUMINAMATH_CALUDE_uncool_relatives_count_l2109_210923


namespace NUMINAMATH_CALUDE_tuesday_lesson_duration_is_one_hour_l2109_210986

/-- Represents the duration of each lesson on Tuesday in hours -/
def tuesday_lesson_duration : ℝ := 1

/-- The total number of hours Adam spent at school over the three days -/
def total_hours : ℝ := 12

/-- The number of lessons Adam had on Monday -/
def monday_lessons : ℕ := 6

/-- The duration of each lesson on Monday in hours -/
def monday_lesson_duration : ℝ := 0.5

/-- The number of lessons Adam had on Tuesday -/
def tuesday_lessons : ℕ := 3

/-- Theorem stating that the duration of each lesson on Tuesday is 1 hour -/
theorem tuesday_lesson_duration_is_one_hour :
  tuesday_lesson_duration = 1 ∧
  total_hours = (monday_lessons : ℝ) * monday_lesson_duration +
                (tuesday_lessons : ℝ) * tuesday_lesson_duration +
                2 * (tuesday_lessons : ℝ) * tuesday_lesson_duration :=
by sorry

end NUMINAMATH_CALUDE_tuesday_lesson_duration_is_one_hour_l2109_210986


namespace NUMINAMATH_CALUDE_simple_interest_rate_l2109_210934

/-- Calculate the interest rate given principal, time, and simple interest -/
theorem simple_interest_rate (principal time simple_interest : ℝ) :
  principal > 0 ∧ time > 0 ∧ simple_interest > 0 →
  (simple_interest * 100) / (principal * time) = 9 ∧
  principal = 8965 ∧ time = 5 ∧ simple_interest = 4034.25 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l2109_210934


namespace NUMINAMATH_CALUDE_solution_product_l2109_210964

theorem solution_product (p q : ℝ) : 
  (p - 6) * (3 * p + 12) = p^2 + 2 * p - 72 →
  (q - 6) * (3 * q + 12) = q^2 + 2 * q - 72 →
  p ≠ q →
  (p + 2) * (q + 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_product_l2109_210964


namespace NUMINAMATH_CALUDE_vector_decomposition_l2109_210959

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![(-2), 4, 7]
def p : Fin 3 → ℝ := ![0, 1, 2]
def q : Fin 3 → ℝ := ![1, 0, 1]
def r : Fin 3 → ℝ := ![(-1), 2, 4]

/-- Theorem: x can be decomposed as 2p - q + r -/
theorem vector_decomposition :
  x = fun i => 2 * p i - q i + r i := by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l2109_210959


namespace NUMINAMATH_CALUDE_white_surface_area_fraction_l2109_210967

theorem white_surface_area_fraction (cube_edge : ℕ) (total_unit_cubes : ℕ) 
  (white_unit_cubes : ℕ) (black_unit_cubes : ℕ) :
  cube_edge = 4 →
  total_unit_cubes = 64 →
  white_unit_cubes = 48 →
  black_unit_cubes = 16 →
  white_unit_cubes + black_unit_cubes = total_unit_cubes →
  (black_unit_cubes : ℚ) * 3 / (6 * cube_edge^2 : ℚ) = 1/2 →
  (6 * cube_edge^2 - black_unit_cubes * 3 : ℚ) / (6 * cube_edge^2 : ℚ) = 1/2 := by
  sorry

#check white_surface_area_fraction

end NUMINAMATH_CALUDE_white_surface_area_fraction_l2109_210967


namespace NUMINAMATH_CALUDE_area_of_triangle_l2109_210939

-- Define the triangle ABC and point P
structure Triangle :=
  (A B C P : ℝ × ℝ)

-- Define the properties of the triangle
def isRightTriangle (t : Triangle) : Prop :=
  -- Right angle at B
  sorry

def pointOnHypotenuse (t : Triangle) : Prop :=
  -- P is on AC
  sorry

def angleABP (t : Triangle) : ℝ :=
  -- Angle ABP in radians
  sorry

def lengthAP (t : Triangle) : ℝ :=
  -- Length of AP
  sorry

def lengthCP (t : Triangle) : ℝ :=
  -- Length of CP
  sorry

def areaABC (t : Triangle) : ℝ :=
  -- Area of triangle ABC
  sorry

-- Theorem statement
theorem area_of_triangle (t : Triangle) 
  (h1 : isRightTriangle t)
  (h2 : pointOnHypotenuse t)
  (h3 : angleABP t = π / 6)  -- 30° in radians
  (h4 : lengthAP t = 2)
  (h5 : lengthCP t = 1) :
  areaABC t = 9 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_l2109_210939


namespace NUMINAMATH_CALUDE_triangle_construction_theorem_l2109_210935

-- Define a triangle as a structure with three points
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a function to check if a point is on a line segment
def isOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop := sorry

-- Define a function to check if two line segments are parallel
def areParallel (A B : ℝ × ℝ) (C D : ℝ × ℝ) : Prop := sorry

-- Define a function to count valid triangles
def countValidTriangles (ABC X'Y'Z' : Triangle) : ℕ := sorry

-- Define a function to count valid triangles including extensions
def countValidTrianglesWithExtensions (ABC X'Y'Z' : Triangle) : ℕ := sorry

theorem triangle_construction_theorem (ABC X'Y'Z' : Triangle) :
  (countValidTriangles ABC X'Y'Z' = 2 ∨
   countValidTriangles ABC X'Y'Z' = 1 ∨
   countValidTriangles ABC X'Y'Z' = 0) ∧
  countValidTrianglesWithExtensions ABC X'Y'Z' = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_construction_theorem_l2109_210935


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2109_210940

/-- A geometric sequence with a_2 = 5 and a_6 = 33 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q ∧ a 2 = 5 ∧ a 6 = 33

/-- The product of a_3 and a_5 in the geometric sequence is 165 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : geometric_sequence a) :
  a 3 * a 5 = 165 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2109_210940


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l2109_210962

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem f_monotonicity_and_extrema :
  (∀ x y, -2 < x → x < y → f x < f y) ∧
  (∀ x y, x < y → y < -2 → f y < f x) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 0, f x ≤ f 0) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 0, -1 / Real.exp 2 ≤ f x) ∧
  f 0 = 1 ∧
  f (-2) = -1 / Real.exp 2 := by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l2109_210962


namespace NUMINAMATH_CALUDE_parallel_line_x_coordinate_l2109_210977

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if two points form a line parallel to the y-axis -/
def parallelToYAxis (p q : Point) : Prop :=
  p.x = q.x

/-- The theorem statement -/
theorem parallel_line_x_coordinate 
  (a : ℝ) 
  (P : Point) 
  (Q : Point) 
  (h1 : P = ⟨a, -5⟩) 
  (h2 : Q = ⟨4, 3⟩) 
  (h3 : parallelToYAxis P Q) : 
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_x_coordinate_l2109_210977


namespace NUMINAMATH_CALUDE_lamp_savings_l2109_210928

theorem lamp_savings (num_lamps : ℕ) (original_price : ℚ) (discount_rate : ℚ) (additional_discount : ℚ) :
  num_lamps = 3 →
  original_price = 15 →
  discount_rate = 0.25 →
  additional_discount = 5 →
  num_lamps * original_price - (num_lamps * (original_price * (1 - discount_rate)) - additional_discount) = 16.25 :=
by sorry

end NUMINAMATH_CALUDE_lamp_savings_l2109_210928


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2109_210991

theorem greatest_divisor_with_remainders : 
  let a := 150 - 50
  let b := 230 - 5
  let c := 175 - 25
  Nat.gcd a (Nat.gcd b c) = 25 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2109_210991


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2109_210981

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2109_210981


namespace NUMINAMATH_CALUDE_product_inequality_l2109_210908

theorem product_inequality (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_one : x + y + z = 1) : 
  (1 - x) * (1 - y) * (1 - z) ≥ 8 * x * y * z := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l2109_210908


namespace NUMINAMATH_CALUDE_complex_number_conditions_l2109_210929

theorem complex_number_conditions (α : ℂ) :
  α ≠ 1 →
  Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1) →
  Complex.abs (α^3 - 1) = 6 * Complex.abs (α - 1) →
  α = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_number_conditions_l2109_210929


namespace NUMINAMATH_CALUDE_intersection_M_N_l2109_210909

def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}
def N : Set ℝ := {-2, -1, 0, 1}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2109_210909


namespace NUMINAMATH_CALUDE_additional_bottles_needed_l2109_210907

def medium_bottle_capacity : ℕ := 50
def giant_bottle_capacity : ℕ := 750
def bottles_already_owned : ℕ := 3

theorem additional_bottles_needed : 
  (giant_bottle_capacity / medium_bottle_capacity) - bottles_already_owned = 12 := by
  sorry

end NUMINAMATH_CALUDE_additional_bottles_needed_l2109_210907


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l2109_210976

theorem pentagon_area_sum (a b : ℤ) (h1 : 0 < b) (h2 : b < a) :
  let F := (a, b)
  let G := (b, a)
  let H := (-b, a)
  let I := (-b, -a)
  let J := (-a, -b)
  let pentagon_area := (a^2 : ℝ) + 3 * (a * b)
  pentagon_area = 550 → a + b = 24 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_sum_l2109_210976


namespace NUMINAMATH_CALUDE_last_number_proof_l2109_210916

theorem last_number_proof (a b c d : ℝ) : 
  (a + b + c) / 3 = 6 →
  (b + c + d) / 3 = 3 →
  a + d = 13 →
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_last_number_proof_l2109_210916


namespace NUMINAMATH_CALUDE_dog_food_consumption_l2109_210920

/-- The amount of dog food eaten by two dogs per day, given that each dog eats 0.125 scoop per day. -/
theorem dog_food_consumption (dog1_consumption dog2_consumption : ℝ) 
  (h1 : dog1_consumption = 0.125)
  (h2 : dog2_consumption = 0.125) : 
  dog1_consumption + dog2_consumption = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_consumption_l2109_210920


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l2109_210950

/-- The function f(x) = x^3 + 3ax^2 - 6ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 - 6*a*x + 2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x - 6*a

theorem minimum_value_implies_a (a : ℝ) :
  ∃ x₀ : ℝ, x₀ > 1 ∧ x₀ < 3 ∧
  (∀ x : ℝ, f a x ≥ f a x₀) ∧
  (f_derivative a x₀ = 0) →
  a = -2 := by sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l2109_210950


namespace NUMINAMATH_CALUDE_granger_grocery_bill_l2109_210919

/-- Calculates the total cost of a grocery shopping trip -/
def total_cost (spam_price peanut_butter_price bread_price : ℕ) 
               (spam_quantity peanut_butter_quantity bread_quantity : ℕ) : ℕ :=
  spam_price * spam_quantity + 
  peanut_butter_price * peanut_butter_quantity + 
  bread_price * bread_quantity

/-- Proves that Granger's grocery bill is $59 -/
theorem granger_grocery_bill : 
  total_cost 3 5 2 12 3 4 = 59 := by
  sorry

end NUMINAMATH_CALUDE_granger_grocery_bill_l2109_210919


namespace NUMINAMATH_CALUDE_inequality_preservation_l2109_210968

theorem inequality_preservation (x y : ℝ) (h : x > y) : 2 * x + 1 > 2 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l2109_210968


namespace NUMINAMATH_CALUDE_position_after_four_steps_l2109_210915

/-- Given a number line where the distance from 0 to 30 is divided into 6 equal steps,
    the position reached after 4 steps is 20. -/
theorem position_after_four_steps :
  ∀ (total_distance : ℝ) (total_steps : ℕ) (steps_taken : ℕ),
    total_distance = 30 →
    total_steps = 6 →
    steps_taken = 4 →
    (total_distance / total_steps) * steps_taken = 20 :=
by
  sorry

#check position_after_four_steps

end NUMINAMATH_CALUDE_position_after_four_steps_l2109_210915


namespace NUMINAMATH_CALUDE_min_dwarves_at_risk_l2109_210980

/-- Represents the color of a hat -/
inductive HatColor
| Black
| White

/-- Represents a dwarf with a hat -/
structure Dwarf :=
  (hat : HatColor)

/-- A line of dwarves -/
def DwarfLine := List Dwarf

/-- The probability of guessing correctly for a single dwarf -/
def guessProb : ℚ := 1/2

/-- The minimum number of dwarves at risk given a strategy -/
def minRisk (p : ℕ) (strategy : DwarfLine → ℕ) : ℕ :=
  min p (strategy (List.replicate p (Dwarf.mk HatColor.Black)))

theorem min_dwarves_at_risk (p : ℕ) (h : p > 0) :
  ∃ (strategy : DwarfLine → ℕ), minRisk p strategy = 1 :=
sorry

end NUMINAMATH_CALUDE_min_dwarves_at_risk_l2109_210980


namespace NUMINAMATH_CALUDE_inequality_solution_l2109_210914

theorem inequality_solution (a : ℝ) (x : ℝ) : 
  (x + 1) * ((a - 1) * x - 1) > 0 ↔ 
    (a < 0 ∧ -1 < x ∧ x < 1 / (a - 1)) ∨
    (0 < a ∧ a < 1 ∧ 1 / (a - 1) < x ∧ x < -1) ∨
    (a = 1 ∧ x < -1) ∨
    (a > 1 ∧ (x < -1 ∨ x > 1 / (a - 1))) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2109_210914


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l2109_210973

theorem quadratic_distinct_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 3*x + 2*m = 0 ∧ y^2 - 3*y + 2*m = 0) ↔ m < 9/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l2109_210973


namespace NUMINAMATH_CALUDE_crypto_puzzle_solution_l2109_210985

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem crypto_puzzle_solution :
  ∀ (A B C : ℕ),
    is_digit A →
    is_digit B →
    is_digit C →
    A + B + 1 = C + 10 →
    B = A + 2 →
    A ≠ B ∧ A ≠ C ∧ B ≠ C →
    C = 1 :=
by sorry

end NUMINAMATH_CALUDE_crypto_puzzle_solution_l2109_210985


namespace NUMINAMATH_CALUDE_alex_score_l2109_210984

-- Define the total number of shots
def total_shots : ℕ := 40

-- Define the success rates
def three_point_success_rate : ℚ := 1/4
def two_point_success_rate : ℚ := 1/5

-- Define the point values
def three_point_value : ℕ := 3
def two_point_value : ℕ := 2

-- Theorem statement
theorem alex_score :
  ∀ x y : ℕ,
  x + y = total_shots →
  (x : ℚ) * three_point_success_rate * three_point_value +
  (y : ℚ) * two_point_success_rate * two_point_value = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_alex_score_l2109_210984


namespace NUMINAMATH_CALUDE_expression_factorization_l2109_210948

theorem expression_factorization (a b c d : ℝ) : 
  a^3 * (b^2 - d^2) + b^3 * (c^2 - a^2) + c^3 * (d^2 - b^2) + d^3 * (a^2 - c^2) = 
  (a - b) * (b - c) * (c - d) * (d - a) * 
  (a^2 + a*b + a*c + a*d + b^2 + b*c + b*d + c^2 + c*d + d^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2109_210948


namespace NUMINAMATH_CALUDE_percentage_increase_proof_l2109_210983

def initial_earnings : ℝ := 40
def new_earnings : ℝ := 60

theorem percentage_increase_proof :
  (new_earnings - initial_earnings) / initial_earnings * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_proof_l2109_210983


namespace NUMINAMATH_CALUDE_egyptian_fraction_sum_l2109_210911

theorem egyptian_fraction_sum : ∃ (b₂ b₃ b₄ b₅ b₆ b₇ b₈ : ℕ),
  (5 : ℚ) / 8 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040 + b₈ / 40320 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧ b₇ < 7 ∧ b₈ < 8 ∧
  b₂ + b₃ + b₄ + b₅ + b₆ + b₇ + b₈ = 4 := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_sum_l2109_210911


namespace NUMINAMATH_CALUDE_charity_fundraiser_revenue_l2109_210999

theorem charity_fundraiser_revenue (total_tickets : ℕ) (total_revenue : ℕ) 
  (h_total_tickets : total_tickets = 170)
  (h_total_revenue : total_revenue = 2917) :
  ∃ (full_price : ℕ) (full_count : ℕ) (quarter_count : ℕ),
    full_count + quarter_count = total_tickets ∧
    full_count * full_price + quarter_count * (full_price / 4) = total_revenue ∧
    full_count * full_price = 1748 :=
by sorry

end NUMINAMATH_CALUDE_charity_fundraiser_revenue_l2109_210999


namespace NUMINAMATH_CALUDE_divisor_power_expression_l2109_210963

theorem divisor_power_expression (k : ℕ) : 
  (30 ^ k : ℕ) ∣ 929260 → 3 ^ k - k ^ 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_power_expression_l2109_210963


namespace NUMINAMATH_CALUDE_unit_vector_d_l2109_210918

def d : ℝ × ℝ := (12, -5)

theorem unit_vector_d :
  let magnitude := Real.sqrt (d.1 ^ 2 + d.2 ^ 2)
  (d.1 / magnitude, d.2 / magnitude) = (12 / 13, -5 / 13) := by
  sorry

end NUMINAMATH_CALUDE_unit_vector_d_l2109_210918


namespace NUMINAMATH_CALUDE_fraction_scaling_l2109_210971

theorem fraction_scaling (a b : ℝ) :
  (2*a + 2*b) / ((2*a)^2 + (2*b)^2) = (1/2) * ((a + b) / (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_scaling_l2109_210971


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l2109_210906

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (x ^ (1/4) = 16 / (9 - x ^ (1/4))) ↔ (x = 4096 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l2109_210906


namespace NUMINAMATH_CALUDE_greatest_partition_number_l2109_210942

/-- A partition of the positive integers into k subsets -/
def Partition (k : ℕ) := Fin k → Set ℕ

/-- The property that a partition satisfies the sum condition for all n ≥ 15 -/
def SatisfiesSumCondition (p : Partition k) : Prop :=
  ∀ (n : ℕ) (i : Fin k), n ≥ 15 →
    ∃ (x y : ℕ), x ∈ p i ∧ y ∈ p i ∧ x ≠ y ∧ x + y = n

/-- The main theorem stating that 3 is the greatest k satisfying the conditions -/
theorem greatest_partition_number :
  (∃ (p : Partition 3), SatisfiesSumCondition p) ∧
  (∀ k > 3, ¬∃ (p : Partition k), SatisfiesSumCondition p) :=
sorry

end NUMINAMATH_CALUDE_greatest_partition_number_l2109_210942


namespace NUMINAMATH_CALUDE_most_likely_gender_combination_l2109_210931

theorem most_likely_gender_combination (n : ℕ) (p : ℝ) : 
  n = 5 → p = 1/2 → 2 * (n.choose 3) * p^n = 5/8 := by sorry

end NUMINAMATH_CALUDE_most_likely_gender_combination_l2109_210931


namespace NUMINAMATH_CALUDE_solution_value_l2109_210961

-- Define the function E
def E (a b c : ℝ) : ℝ := a * b^2 + c

-- State the theorem
theorem solution_value : ∃ a : ℝ, 2*a + E a 3 2 = 4 + E a 5 3 ∧ a = -5/14 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l2109_210961


namespace NUMINAMATH_CALUDE_unique_solution_l2109_210990

/-- Represents the pictures in the table --/
inductive Picture : Type
| Cat : Picture
| Chicken : Picture
| Crab : Picture
| Bear : Picture
| Goat : Picture

/-- Assignment of digits to pictures --/
def PictureAssignment := Picture → Fin 10

/-- Checks if all pictures are assigned different digits --/
def is_valid_assignment (assignment : PictureAssignment) : Prop :=
  ∀ p q : Picture, p ≠ q → assignment p ≠ assignment q

/-- Checks if the assignment satisfies the row and column sums --/
def satisfies_sums (assignment : PictureAssignment) : Prop :=
  assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Crab = 10 ∧
  assignment Picture.Goat + assignment Picture.Goat + assignment Picture.Crab + assignment Picture.Bear + assignment Picture.Bear = 16 ∧
  assignment Picture.Cat + assignment Picture.Bear + assignment Picture.Goat + assignment Picture.Goat + assignment Picture.Crab = 13 ∧
  assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Chicken + assignment Picture.Chicken + assignment Picture.Goat = 17 ∧
  assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Goat = 11

/-- The theorem to be proved --/
theorem unique_solution :
  ∃! assignment : PictureAssignment,
    is_valid_assignment assignment ∧
    satisfies_sums assignment ∧
    assignment Picture.Cat = 1 ∧
    assignment Picture.Chicken = 5 ∧
    assignment Picture.Crab = 2 ∧
    assignment Picture.Bear = 4 ∧
    assignment Picture.Goat = 3 :=
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2109_210990


namespace NUMINAMATH_CALUDE_triangle_isosceles_l2109_210921

/-- A triangle with side lengths satisfying a specific equation is isosceles. -/
theorem triangle_isosceles (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_triangle : (a - c)^2 + (a - c) * b = 0) : a = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_isosceles_l2109_210921


namespace NUMINAMATH_CALUDE_congruence_problem_l2109_210995

theorem congruence_problem (N : ℕ) (h1 : N > 1) 
  (h2 : 69 ≡ 90 [MOD N]) (h3 : 90 ≡ 125 [MOD N]) : 
  81 ≡ 4 [MOD N] := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l2109_210995


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l2109_210993

theorem x_squared_plus_reciprocal (x : ℝ) (h : x^4 + 1/x^4 = 240) : 
  x^2 + 1/x^2 = Real.sqrt 242 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l2109_210993


namespace NUMINAMATH_CALUDE_alternating_arrangements_count_alternating_arrangements_proof_l2109_210949

/-- The number of ways to arrange 2 men and 2 women in a row,
    such that no two men or two women are adjacent. -/
def alternating_arrangements : ℕ := 8

/-- The number of men in the arrangement. -/
def num_men : ℕ := 2

/-- The number of women in the arrangement. -/
def num_women : ℕ := 2

/-- Theorem stating that the number of alternating arrangements
    of 2 men and 2 women is 8. -/
theorem alternating_arrangements_count :
  alternating_arrangements = 8 ∧
  num_men = 2 ∧
  num_women = 2 := by
  sorry

/-- Proof that the number of alternating arrangements is correct. -/
theorem alternating_arrangements_proof :
  alternating_arrangements = 2 * (Nat.factorial num_men) * (Nat.factorial num_women) := by
  sorry

end NUMINAMATH_CALUDE_alternating_arrangements_count_alternating_arrangements_proof_l2109_210949


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2109_210930

-- Define a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the function y = x(2x - 1)
def f (x : ℝ) : ℝ := x * (2 * x - 1)

-- Theorem statement
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l2109_210930
