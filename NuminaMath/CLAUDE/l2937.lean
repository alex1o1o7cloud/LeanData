import Mathlib

namespace luthers_line_has_17_pieces_l2937_293776

/-- Represents Luther's latest clothing line -/
structure ClothingLine where
  silk_pieces : ℕ
  cashmere_pieces : ℕ
  blended_pieces : ℕ

/-- Calculates the total number of pieces in the clothing line -/
def total_pieces (line : ClothingLine) : ℕ :=
  line.silk_pieces + line.cashmere_pieces + line.blended_pieces

/-- Theorem: Luther's latest line has 17 pieces -/
theorem luthers_line_has_17_pieces :
  ∃ (line : ClothingLine),
    line.silk_pieces = 10 ∧
    line.cashmere_pieces = line.silk_pieces / 2 ∧
    line.blended_pieces = 2 ∧
    total_pieces line = 17 := by
  sorry

end luthers_line_has_17_pieces_l2937_293776


namespace consecutive_integers_product_sum_l2937_293757

theorem consecutive_integers_product_sum (x : ℕ) :
  x > 0 ∧ x * (x + 1) = 930 → x + (x + 1) = 61 := by
  sorry

end consecutive_integers_product_sum_l2937_293757


namespace smallest_d_value_l2937_293789

theorem smallest_d_value (c d : ℕ+) (h1 : c - d = 8) 
  (h2 : Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16) : 
  d ≥ 4 ∧ ∃ (c' d' : ℕ+), c' - d' = 8 ∧ 
    Nat.gcd ((c'^3 + d'^3) / (c' + d')) (c' * d') = 16 ∧ d' = 4 :=
sorry

end smallest_d_value_l2937_293789


namespace six_star_nine_l2937_293711

-- Define the star operation
def star (a b : ℕ) : ℚ :=
  (a * b : ℚ) / (a + b - 3 : ℚ)

-- Theorem statement
theorem six_star_nine :
  (∀ a b : ℕ, a > 0 ∧ b > 0 ∧ a + b > 3) →
  star 6 9 = 9 / 2 := by
sorry

end six_star_nine_l2937_293711


namespace platform_height_l2937_293707

/-- Given two configurations of identical rectangular prisms on a platform,
    prove that the platform height is 37 inches. -/
theorem platform_height (l w : ℝ) : 
  l + 37 - w = 40 → w + 37 - l = 34 → 37 = 37 := by
  sorry

end platform_height_l2937_293707


namespace line_perpendicular_to_parallel_planes_l2937_293772

structure Space where
  Plane : Type
  Line : Type
  parallel : Plane → Plane → Prop
  perpendicular : Line → Plane → Prop
  subset : Line → Plane → Prop

theorem line_perpendicular_to_parallel_planes 
  (S : Space) (α β : S.Plane) (m : S.Line) : 
  S.perpendicular m α → S.parallel α β → S.perpendicular m β := by
  sorry

end line_perpendicular_to_parallel_planes_l2937_293772


namespace triangle_side_difference_l2937_293719

/-- Given a triangle ABC with side lengths satisfying specific conditions, prove that b - a = 0 --/
theorem triangle_side_difference (a b : ℤ) : 
  a > 1 → 
  b > 1 → 
  ∃ (AB BC CA : ℝ), 
    AB = b^2 - 1 ∧ 
    BC = a^2 ∧ 
    CA = 2*a ∧ 
    AB + BC > CA ∧ 
    BC + CA > AB ∧ 
    CA + AB > BC → 
    b - a = 0 := by
  sorry

end triangle_side_difference_l2937_293719


namespace find_number_l2937_293754

theorem find_number : ∃ N : ℚ, (5/6 * N) - (5/16 * N) = 250 ∧ N = 480 := by
  sorry

end find_number_l2937_293754


namespace product_of_numbers_l2937_293751

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 58) : x * y = 21 := by
  sorry

end product_of_numbers_l2937_293751


namespace simplify_and_evaluate_l2937_293783

theorem simplify_and_evaluate (a b : ℚ) (h1 : a = -1/3) (h2 : b = -2) :
  ((3*a + b)^2 - (3*a + b)*(3*a - b)) / (2*b) = -3 := by
  sorry

end simplify_and_evaluate_l2937_293783


namespace square_difference_l2937_293700

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end square_difference_l2937_293700


namespace polynomial_simplification_l2937_293778

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^6 + 3 * x^5 + 2 * x^4 + x + 15) - (x^6 + 4 * x^5 + 5 * x^4 - 2 * x^3 + 20) = 
  x^6 - x^5 - 3 * x^4 + 2 * x^3 + x - 5 := by
  sorry

end polynomial_simplification_l2937_293778


namespace equation_implies_ratio_one_third_l2937_293794

theorem equation_implies_ratio_one_third 
  (a x y : ℝ) 
  (h_distinct : a ≠ x ∧ x ≠ y ∧ a ≠ y) 
  (h_eq : Real.sqrt (a * (x - a)) + Real.sqrt (a * (y - a)) = Real.sqrt (x - a) - Real.sqrt (a - y)) :
  (3 * x^2 + x * y - y^2) / (x^2 - x * y + y^2) = 1/3 :=
by sorry

end equation_implies_ratio_one_third_l2937_293794


namespace circle_intersection_tangent_slope_l2937_293749

noncomputable def C₁ (x y : ℝ) := x^2 + y^2 - 6*x + 4*y + 9 = 0

noncomputable def C₂ (m x y : ℝ) := (x + m)^2 + (y + m + 5)^2 = 2*m^2 + 8*m + 10

def on_coordinate_axes (x y : ℝ) := x = 0 ∨ y = 0

theorem circle_intersection_tangent_slope 
  (m : ℝ) (h_m : m ≠ -3) (x₀ y₀ : ℝ) (h_axes : on_coordinate_axes x₀ y₀)
  (h_tangent : ∃ (T₁_x T₁_y T₂_x T₂_y : ℝ), 
    C₁ T₁_x T₁_y ∧ C₂ m T₂_x T₂_y ∧ 
    (x₀ - T₁_x)^2 + (y₀ - T₁_y)^2 = (x₀ - T₂_x)^2 + (y₀ - T₂_y)^2) :
  (m = 5 → ∃! (n : ℕ), n = 2 ∧ ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧ C₂ m x₁ y₁ ∧ C₂ m x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) ∧
  (x₀ + y₀ + 1 = 0 ∧ ((x₀ = 0 ∧ y₀ = -1) ∨ (x₀ = -1 ∧ y₀ = 0))) ∧
  (∀ (k : ℝ), (∀ (x y : ℝ), C₁ x y → (y + 2 = k * (x - 3)) → 
    (∀ (m : ℝ), m ≠ -3 → ∃ (x' y' : ℝ), C₂ m x' y' ∧ y' + 2 = k * (x' - 3))) → k > 0) :=
sorry

end circle_intersection_tangent_slope_l2937_293749


namespace problem_solution_l2937_293758

theorem problem_solution (a b : ℕ+) : 
  Nat.lcm a b = 2520 → 
  Nat.gcd a b = 24 → 
  a = 240 → 
  b = 252 := by
sorry

end problem_solution_l2937_293758


namespace problem_statement_l2937_293761

theorem problem_statement (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a + b)^2014 = 1 := by
  sorry

end problem_statement_l2937_293761


namespace min_value_expression_l2937_293717

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y < 27) :
  (Real.sqrt x + Real.sqrt y) / Real.sqrt (x * y) + 1 / Real.sqrt (27 - x - y) ≥ 1 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b < 27 ∧
    (Real.sqrt a + Real.sqrt b) / Real.sqrt (a * b) + 1 / Real.sqrt (27 - a - b) = 1 :=
by sorry

end min_value_expression_l2937_293717


namespace vector_magnitude_problem_l2937_293777

/-- Given two vectors on a plane with specific properties, prove the magnitude of a third vector. -/
theorem vector_magnitude_problem (a b m : ℝ × ℝ) : 
  (a.1 * b.1 + a.2 * b.2 = -1/2) →  -- angle between a and b is 120°
  (a.1^2 + a.2^2 = 1) →             -- magnitude of a is 1
  (b.1^2 + b.2^2 = 4) →             -- magnitude of b is 2
  (m.1 * a.1 + m.2 * a.2 = 1) →     -- m · a = 1
  (m.1 * b.1 + m.2 * b.2 = 1) →     -- m · b = 1
  m.1^2 + m.2^2 = 7/3 :=            -- |m|^2 = (√21/3)^2 = 21/9 = 7/3
by sorry

end vector_magnitude_problem_l2937_293777


namespace hypotenuse_length_l2937_293736

/-- A right triangle with specific properties -/
structure RightTriangle where
  /-- The hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- The perimeter of the triangle -/
  perimeter : ℝ
  /-- The side opposite to the 30° angle -/
  opposite_30 : ℝ
  /-- Condition: The triangle has a right angle -/
  right_angle : True
  /-- Condition: One angle is 30° -/
  angle_30 : True
  /-- Condition: The perimeter is 120 units -/
  perimeter_120 : perimeter = 120
  /-- Condition: The side opposite to 30° is half the hypotenuse -/
  opposite_half_hypotenuse : opposite_30 = hypotenuse / 2

/-- Theorem: The hypotenuse of the specified right triangle is 40(3 - √3) -/
theorem hypotenuse_length (t : RightTriangle) : t.hypotenuse = 40 * (3 - Real.sqrt 3) := by
  sorry

end hypotenuse_length_l2937_293736


namespace decimal_34_to_binary_l2937_293705

def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_34_to_binary :
  decimal_to_binary 34 = [1, 0, 0, 0, 1, 0] := by
  sorry

end decimal_34_to_binary_l2937_293705


namespace fraction_to_zero_power_l2937_293773

theorem fraction_to_zero_power :
  let f : ℚ := -574839201 / 1357924680
  f ≠ 0 →
  f^0 = 1 := by sorry

end fraction_to_zero_power_l2937_293773


namespace train_length_l2937_293716

/-- The length of a train given its speed, the speed of a man moving in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (crossing_time : ℝ) : 
  train_speed = 25 →
  man_speed = 2 →
  crossing_time = 44 →
  (train_speed + man_speed) * crossing_time * (1000 / 3600) = 330 := by
  sorry

#check train_length

end train_length_l2937_293716


namespace intersection_complement_problem_l2937_293703

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {1, 3, 5}

theorem intersection_complement_problem :
  N ∩ (U \ M) = {3, 5} := by sorry

end intersection_complement_problem_l2937_293703


namespace arithmetic_geometric_sum_l2937_293771

/-- An arithmetic sequence with common difference 2 -/
def arithmeticSeq (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- a_1, a_3, and a_4 form a geometric sequence -/
def geometricSubseq (a : ℕ → ℤ) : Prop :=
  a 3 ^ 2 = a 1 * a 4

theorem arithmetic_geometric_sum (a : ℕ → ℤ) 
  (h_arith : arithmeticSeq a) (h_geom : geometricSubseq a) : 
  a 2 + a 3 = -10 := by
  sorry

end arithmetic_geometric_sum_l2937_293771


namespace prob_at_least_one_white_l2937_293781

/-- The number of red balls in the bag -/
def num_red : ℕ := 3

/-- The number of white balls in the bag -/
def num_white : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red + num_white

/-- The number of balls drawn -/
def drawn : ℕ := 3

/-- The probability of drawing at least one white ball -/
theorem prob_at_least_one_white :
  (1 - (Nat.choose num_red drawn / Nat.choose total_balls drawn : ℚ)) = 9/10 := by
  sorry

end prob_at_least_one_white_l2937_293781


namespace problem_solution_l2937_293713

theorem problem_solution (x y : ℝ) 
  (h1 : x = 153) 
  (h2 : x^3*y - 4*x^2*y + 4*x*y = 350064) : 
  y = 40/3967 := by sorry

end problem_solution_l2937_293713


namespace mean_proportional_sqrt45_and_7_3_pi_l2937_293798

theorem mean_proportional_sqrt45_and_7_3_pi :
  let a := Real.sqrt 45
  let b := 7/3 * Real.pi
  Real.sqrt (a * b) = Real.sqrt (7 * Real.sqrt 5 * Real.pi) := by
  sorry

end mean_proportional_sqrt45_and_7_3_pi_l2937_293798


namespace car_acceleration_at_one_second_l2937_293784

-- Define the velocity function
def v (t : ℝ) : ℝ := -t^2 + 10*t

-- Define the acceleration function as the derivative of velocity
def a (t : ℝ) : ℝ := -2*t + 10

-- Theorem statement
theorem car_acceleration_at_one_second :
  a 1 = 8 := by
  sorry

end car_acceleration_at_one_second_l2937_293784


namespace train_speed_calculation_l2937_293726

/-- The speed of a train in km/hr -/
def train_speed : ℝ := 90

/-- The length of the train in meters -/
def train_length : ℝ := 750

/-- The time taken to cross the platform in minutes -/
def crossing_time : ℝ := 1

/-- The length of the platform in meters -/
def platform_length : ℝ := train_length

theorem train_speed_calculation :
  train_speed = (2 * train_length) / crossing_time * 60 / 1000 :=
by sorry

end train_speed_calculation_l2937_293726


namespace large_pizza_slices_l2937_293718

def large_pizza_cost : ℝ := 10
def first_topping_cost : ℝ := 2
def next_two_toppings_cost : ℝ := 1
def remaining_toppings_cost : ℝ := 0.5
def num_toppings : ℕ := 7
def cost_per_slice : ℝ := 2

def total_pizza_cost : ℝ :=
  large_pizza_cost + first_topping_cost + 2 * next_two_toppings_cost + 
  (num_toppings - 3 : ℝ) * remaining_toppings_cost

theorem large_pizza_slices :
  (total_pizza_cost / cost_per_slice : ℝ) = 8 :=
sorry

end large_pizza_slices_l2937_293718


namespace tree_planting_event_l2937_293744

theorem tree_planting_event (boys girls : ℕ) : 
  girls - boys = 400 →
  boys = 600 →
  girls > boys →
  (60 : ℚ) / 100 * (boys + girls) = 960 := by
  sorry

end tree_planting_event_l2937_293744


namespace train_passengers_count_l2937_293762

/-- Represents the number of passengers in each carriage of a train -/
structure TrainCarriages :=
  (c1 c2 c3 c4 c5 : ℕ)

/-- Defines the condition for the number of neighbours a passenger has -/
def valid_neighbours (tc : TrainCarriages) : Prop :=
  ∀ i : Fin 5, 
    let neighbours := match i with
      | 0 => tc.c1 - 1 + tc.c2
      | 1 => tc.c1 + tc.c2 - 1 + tc.c3
      | 2 => tc.c2 + tc.c3 - 1 + tc.c4
      | 3 => tc.c3 + tc.c4 - 1 + tc.c5
      | 4 => tc.c4 + tc.c5 - 1
    (neighbours = 5 ∨ neighbours = 10)

/-- The main theorem stating that under the given conditions, 
    the total number of passengers is 17 -/
theorem train_passengers_count (tc : TrainCarriages) 
  (h1 : tc.c1 ≥ 1 ∧ tc.c2 ≥ 1 ∧ tc.c3 ≥ 1 ∧ tc.c4 ≥ 1 ∧ tc.c5 ≥ 1)
  (h2 : valid_neighbours tc) : 
  tc.c1 + tc.c2 + tc.c3 + tc.c4 + tc.c5 = 17 := by
  sorry

end train_passengers_count_l2937_293762


namespace min_value_sum_of_squares_l2937_293770

theorem min_value_sum_of_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a ≥ 6 ∧
  ((a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a = 6 ↔ a = b ∧ b = c) :=
by sorry

end min_value_sum_of_squares_l2937_293770


namespace book_purchase_theorem_l2937_293792

theorem book_purchase_theorem (total_A total_B both only_B : ℕ) 
  (h1 : total_A = 2 * total_B)
  (h2 : both = 500)
  (h3 : both = 2 * only_B) :
  total_A - both = 1000 := by
  sorry

end book_purchase_theorem_l2937_293792


namespace expression_simplification_l2937_293704

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 + 2) :
  (a / (a^2 - 4*a + 4) + (a + 2) / (2*a - a^2)) / (2 / (a^2 - 2*a)) = Real.sqrt 2 := by
  sorry

end expression_simplification_l2937_293704


namespace tree_table_profit_l2937_293785

/-- Calculates the profit from selling tables made from chopped trees --/
theorem tree_table_profit
  (trees : ℕ)
  (planks_per_tree : ℕ)
  (planks_per_table : ℕ)
  (price_per_table : ℕ)
  (labor_cost : ℕ)
  (h1 : trees = 30)
  (h2 : planks_per_tree = 25)
  (h3 : planks_per_table = 15)
  (h4 : price_per_table = 300)
  (h5 : labor_cost = 3000)
  : (trees * planks_per_tree / planks_per_table) * price_per_table - labor_cost = 12000 := by
  sorry

#check tree_table_profit

end tree_table_profit_l2937_293785


namespace inscribed_circle_radius_l2937_293732

/-- Configuration of semicircles and inscribed circle -/
structure SemicircleConfig where
  R : ℝ  -- Radius of larger semicircle
  r : ℝ  -- Radius of smaller semicircle
  x : ℝ  -- Radius of inscribed circle

/-- The inscribed circle is tangent to both semicircles and the diameter -/
def is_tangent (config : SemicircleConfig) : Prop :=
  ∃ (O O₁ O₂ : ℝ × ℝ),
    let d := config.R - config.x
    let h := Real.sqrt (d^2 - config.x^2)
    (config.R + config.r)^2 = d^2 + (config.r + config.x)^2 ∧
    h^2 + config.x^2 = config.R^2 ∧
    h^2 + (config.r + config.x)^2 = (config.R + config.r)^2

theorem inscribed_circle_radius
  (config : SemicircleConfig)
  (h₁ : config.R = 18)
  (h₂ : config.r = 9)
  (h₃ : is_tangent config) :
  config.x = 8 :=
sorry

end inscribed_circle_radius_l2937_293732


namespace triangle_area_range_l2937_293735

-- Define the triangle ABC
structure Triangle :=
  (AB : ℝ)
  (BC : ℝ)
  (CA : ℝ)

-- Define the variable points P and Q
structure VariablePoints :=
  (P : ℝ) -- distance AP
  (Q : ℝ) -- distance AQ

-- Define the perpendiculars x and y
structure Perpendiculars :=
  (x : ℝ)
  (y : ℝ)

-- Define the main theorem
theorem triangle_area_range (ABC : Triangle) (PQ : VariablePoints) (perp : Perpendiculars) :
  ABC.AB = 4 ∧ ABC.BC = 5 ∧ ABC.CA = 3 →
  0 < PQ.P ∧ PQ.P ≤ ABC.AB →
  0 < PQ.Q ∧ PQ.Q ≤ ABC.CA →
  perp.x = PQ.Q / 2 →
  perp.y = PQ.P / 2 →
  PQ.P * PQ.Q = 6 →
  6 ≤ 2 * perp.y + 3 * perp.x ∧ 2 * perp.y + 3 * perp.x ≤ 6.5 :=
by sorry


end triangle_area_range_l2937_293735


namespace pi_estimation_l2937_293769

theorem pi_estimation (n m : ℕ) (h1 : n = 100) (h2 : m = 31) :
  let π_est := 4 * (n : ℝ) / (m : ℝ) - 3
  π_est = 81 / 25 := by
  sorry

end pi_estimation_l2937_293769


namespace three_digit_divisible_by_15_l2937_293764

theorem three_digit_divisible_by_15 : 
  (Finset.filter (fun k => 100 ≤ 15 * k ∧ 15 * k ≤ 999) (Finset.range 1000)).card = 60 := by
  sorry

end three_digit_divisible_by_15_l2937_293764


namespace inequality_equivalence_l2937_293709

theorem inequality_equivalence (x : ℝ) : (x - 1) / 2 ≤ -1 ↔ x ≤ -1 := by
  sorry

end inequality_equivalence_l2937_293709


namespace power_tower_mod_500_l2937_293752

theorem power_tower_mod_500 : 5^(5^(5^2)) ≡ 25 [ZMOD 500] := by
  sorry

end power_tower_mod_500_l2937_293752


namespace age_sum_proof_l2937_293712

theorem age_sum_proof (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 72 → a + b + c = 14 := by
  sorry

end age_sum_proof_l2937_293712


namespace polynomial_factorization_l2937_293739

theorem polynomial_factorization (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end polynomial_factorization_l2937_293739


namespace max_value_of_expression_l2937_293737

theorem max_value_of_expression (t : ℝ) : 
  (∃ (t_max : ℝ), ∀ (t : ℝ), ((3^t - 5*t)*t)/(9^t) ≤ ((3^t_max - 5*t_max)*t_max)/(9^t_max)) ∧ 
  (∃ (t_0 : ℝ), ((3^t_0 - 5*t_0)*t_0)/(9^t_0) = 1/20) := by
  sorry

end max_value_of_expression_l2937_293737


namespace max_k_value_l2937_293775

open Real

noncomputable def f (x : ℝ) : ℝ := x * (1 + log x)

theorem max_k_value (k : ℤ) :
  (∀ x > 2, k * (x - 2) < f x) → k ≤ 4 :=
by sorry

end max_k_value_l2937_293775


namespace joan_piano_time_l2937_293706

/-- Represents the time Joan spent on various activities during her music practice -/
structure MusicPractice where
  total_time : ℕ
  writing_time : ℕ
  reading_time : ℕ
  exercising_time : ℕ

/-- Calculates the time spent on the piano given Joan's music practice schedule -/
def time_on_piano (practice : MusicPractice) : ℕ :=
  practice.total_time - (practice.writing_time + practice.reading_time + practice.exercising_time)

/-- Theorem stating that Joan spent 30 minutes on the piano -/
theorem joan_piano_time :
  let practice : MusicPractice := {
    total_time := 120,
    writing_time := 25,
    reading_time := 38,
    exercising_time := 27
  }
  time_on_piano practice = 30 := by sorry

end joan_piano_time_l2937_293706


namespace circle_area_from_diameter_endpoints_l2937_293708

/-- The area of a circle with diameter endpoints C(-2, 3) and D(4, -1) is 13π. -/
theorem circle_area_from_diameter_endpoints :
  let C : ℝ × ℝ := (-2, 3)
  let D : ℝ × ℝ := (4, -1)
  let diameter_squared := (D.1 - C.1)^2 + (D.2 - C.2)^2
  let radius_squared := diameter_squared / 4
  let circle_area := π * radius_squared
  circle_area = 13 * π :=
by sorry

end circle_area_from_diameter_endpoints_l2937_293708


namespace xyz_value_l2937_293729

theorem xyz_value (x y z : ℝ) 
  (h1 : 2 * x + 3 * y + z = 13) 
  (h2 : 4 * x^2 + 9 * y^2 + z^2 - 2 * x + 15 * y + 3 * z = 82) : 
  x * y * z = 12 := by
sorry

end xyz_value_l2937_293729


namespace fraction_sum_equality_l2937_293756

theorem fraction_sum_equality : 
  (2 : ℚ) / 100 + 5 / 1000 + 5 / 10000 + 3 * (4 / 1000) = 375 / 10000 := by
  sorry

end fraction_sum_equality_l2937_293756


namespace ribbon_length_for_circular_sign_l2937_293731

/-- Given a circular region with area 616 square inches, using π ≈ 22/7,
    and adding 10% extra to the circumference, prove that the amount of
    ribbon needed (rounded up to the nearest inch) is 97 inches. -/
theorem ribbon_length_for_circular_sign :
  let area : ℝ := 616
  let π_approx : ℝ := 22 / 7
  let radius : ℝ := Real.sqrt (area / π_approx)
  let circumference : ℝ := 2 * π_approx * radius
  let extra_ribbon : ℝ := 0.1 * circumference
  let total_ribbon : ℝ := circumference + extra_ribbon
  ⌈total_ribbon⌉ = 97 := by
sorry

end ribbon_length_for_circular_sign_l2937_293731


namespace max_interval_length_l2937_293714

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- State the theorem
theorem max_interval_length 
  (a b : ℝ) 
  (h1 : a ≤ b) 
  (h2 : ∀ x ∈ Set.Icc a b, -3 ≤ f x ∧ f x ≤ 1) 
  (h3 : ∃ x ∈ Set.Icc a b, f x = -3) 
  (h4 : ∃ x ∈ Set.Icc a b, f x = 1) :
  b - a ≤ 4 :=
sorry

end max_interval_length_l2937_293714


namespace angle_range_l2937_293725

theorem angle_range (α : Real) :
  (|Real.sin (4 * Real.pi - α)| = Real.sin (Real.pi + α)) →
  ∃ k : ℤ, 2 * k * Real.pi - Real.pi ≤ α ∧ α ≤ 2 * k * Real.pi :=
by sorry

end angle_range_l2937_293725


namespace downstream_speed_calculation_l2937_293730

/-- Represents the speed of a person rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ

/-- Calculates the downstream speed given upstream and still water speeds -/
def downstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.upstream

/-- Theorem stating that given the specified upstream and still water speeds, 
    the downstream speed is 53 kmph -/
theorem downstream_speed_calculation (s : RowingSpeed) 
  (h1 : s.upstream = 37) 
  (h2 : s.stillWater = 45) : 
  downstreamSpeed s = 53 := by
  sorry

#eval downstreamSpeed { upstream := 37, stillWater := 45 }

end downstream_speed_calculation_l2937_293730


namespace value_range_of_f_l2937_293774

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 2*x

-- Define the domain
def domain : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem value_range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -3 ≤ y ∧ y ≤ 1} := by sorry

end value_range_of_f_l2937_293774


namespace sum_of_three_consecutive_integers_l2937_293787

theorem sum_of_three_consecutive_integers (a b c : ℤ) : 
  (a + 1 = b) → (b + 1 = c) → (c = 12) → (a + b + c = 33) := by
  sorry

end sum_of_three_consecutive_integers_l2937_293787


namespace knight_return_even_moves_l2937_293745

/-- Represents a chess square --/
structure ChessSquare :=
  (color : Bool)

/-- Represents a knight's move on a chess board --/
def knightMove (start : ChessSquare) : ChessSquare :=
  { color := ¬start.color }

/-- Represents a sequence of knight moves --/
def knightMoves (start : ChessSquare) (n : ℕ) : ChessSquare :=
  match n with
  | 0 => start
  | m + 1 => knightMove (knightMoves start m)

/-- Theorem: If a knight returns to its starting square after n moves, then n is even --/
theorem knight_return_even_moves (start : ChessSquare) (n : ℕ) :
  knightMoves start n = start → Even n :=
by sorry

end knight_return_even_moves_l2937_293745


namespace sum_binomial_congruence_l2937_293786

theorem sum_binomial_congruence (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  (∑' j, Nat.choose p j * Nat.choose (p + j) j) ≡ (2^p + 1) [ZMOD p^2] := by
  sorry

end sum_binomial_congruence_l2937_293786


namespace increments_of_z_l2937_293723

noncomputable def z (x y : ℝ) : ℝ := x^2 * y

theorem increments_of_z (x y Δx Δy : ℝ) 
  (hx : x = 1) (hy : y = 2) (hΔx : Δx = 0.1) (hΔy : Δy = -0.2) :
  let Δx_z := z (x + Δx) y - z x y
  let Δy_z := z x (y + Δy) - z x y
  let Δz := z (x + Δx) (y + Δy) - z x y
  (Δx_z = 0.42) ∧ (Δy_z = -0.2) ∧ (Δz = 0.178) := by
  sorry

end increments_of_z_l2937_293723


namespace ring_stack_height_is_117_l2937_293750

/-- Calculates the distance from the top of the top ring to the bottom of the bottom ring in a stack of linked rings. -/
def ring_stack_height (top_diameter : ℝ) (top_thickness : ℝ) (bottom_diameter : ℝ) 
  (diameter_decrease : ℝ) (thickness_decrease : ℝ) : ℝ :=
  sorry

/-- The distance from the top of the top ring to the bottom of the bottom ring is 117 cm. -/
theorem ring_stack_height_is_117 : 
  ring_stack_height 30 2 10 2 0.1 = 117 := by sorry

end ring_stack_height_is_117_l2937_293750


namespace total_savings_percentage_l2937_293715

/-- Calculates the total savings percentage given the original prices and discount rates -/
theorem total_savings_percentage
  (jacket_price shirt_price hat_price : ℝ)
  (jacket_discount shirt_discount hat_discount : ℝ)
  (h_jacket_price : jacket_price = 100)
  (h_shirt_price : shirt_price = 50)
  (h_hat_price : hat_price = 30)
  (h_jacket_discount : jacket_discount = 0.3)
  (h_shirt_discount : shirt_discount = 0.6)
  (h_hat_discount : hat_discount = 0.5) :
  (jacket_price * jacket_discount + shirt_price * shirt_discount + hat_price * hat_discount) /
  (jacket_price + shirt_price + hat_price) * 100 = 41.67 :=
by sorry

end total_savings_percentage_l2937_293715


namespace blithe_toys_proof_l2937_293748

/-- The number of toys Blithe lost -/
def lost_toys : ℕ := 6

/-- The number of toys Blithe found -/
def found_toys : ℕ := 9

/-- The number of toys Blithe had after losing and finding toys -/
def final_toys : ℕ := 43

/-- The initial number of toys Blithe had -/
def initial_toys : ℕ := 40

theorem blithe_toys_proof :
  initial_toys - lost_toys + found_toys = final_toys :=
by sorry

end blithe_toys_proof_l2937_293748


namespace dolly_additional_tickets_l2937_293793

/-- The number of additional tickets Dolly needs to buy for amusement park rides -/
theorem dolly_additional_tickets : ℕ := by
  -- Define the number of rides Dolly wants for each attraction
  let ferris_wheel_rides : ℕ := 2
  let roller_coaster_rides : ℕ := 3
  let log_ride_rides : ℕ := 7

  -- Define the cost in tickets for each attraction
  let ferris_wheel_cost : ℕ := 2
  let roller_coaster_cost : ℕ := 5
  let log_ride_cost : ℕ := 1

  -- Define the number of tickets Dolly currently has
  let current_tickets : ℕ := 20

  -- Calculate the total number of tickets needed
  let total_tickets_needed : ℕ := 
    ferris_wheel_rides * ferris_wheel_cost +
    roller_coaster_rides * roller_coaster_cost +
    log_ride_rides * log_ride_cost

  -- Calculate the additional tickets needed
  let additional_tickets : ℕ := total_tickets_needed - current_tickets

  -- Prove that the additional tickets needed is 6
  have h : additional_tickets = 6 := by sorry

  exact 6

end dolly_additional_tickets_l2937_293793


namespace difference_between_results_l2937_293765

theorem difference_between_results (x : ℝ) (h : x = 15) : 2 * x - (26 - x) = 19 := by
  sorry

end difference_between_results_l2937_293765


namespace sum_of_reciprocal_ratios_ge_two_l2937_293799

theorem sum_of_reciprocal_ratios_ge_two (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  b / a + a / b ≥ 2 := by
  sorry

end sum_of_reciprocal_ratios_ge_two_l2937_293799


namespace tenth_term_is_39_l2937_293790

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  second_term : a + d = 7
  fifth_term : a + 4 * d = 19

/-- The tenth term of the arithmetic sequence is 39 -/
theorem tenth_term_is_39 (seq : ArithmeticSequence) : seq.a + 9 * seq.d = 39 := by
  sorry

end tenth_term_is_39_l2937_293790


namespace function_and_minimum_value_l2937_293734

def f (x : ℝ) := x^2 - x - 2

def g (x a : ℝ) := f (x + a) + x

theorem function_and_minimum_value :
  (∀ x, f (x - 1) = x^2 - 3 * x) →
  (∀ x, f x = x^2 - x - 2) ∧
  (∀ a,
    (a ≥ 1 → ∀ x ∈ Set.Icc (-1) 3, g x a ≥ a^2 - 3 * a - 1) ∧
    (-3 < a ∧ a < 1 → ∀ x ∈ Set.Icc (-1) 3, g x a ≥ -a - 2) ∧
    (a ≤ -3 → ∀ x ∈ Set.Icc (-1) 3, g x a ≥ a^2 + 5 * a + 7)) :=
by sorry

end function_and_minimum_value_l2937_293734


namespace simplify_expression_l2937_293747

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 + 9 = 45*x + 27 := by
  sorry

end simplify_expression_l2937_293747


namespace arithmetic_sequence_sum_l2937_293702

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 15 = 48 →
  a 3 + 3 * a 8 + a 13 = 120 :=
by
  sorry

end arithmetic_sequence_sum_l2937_293702


namespace rectangle_perimeter_l2937_293721

/-- A rectangle with area 120 square feet and shorter sides of 8 feet has a perimeter of 46 feet -/
theorem rectangle_perimeter (area : ℝ) (short_side : ℝ) (long_side : ℝ) (perimeter : ℝ) : 
  area = 120 →
  short_side = 8 →
  area = long_side * short_side →
  perimeter = 2 * long_side + 2 * short_side →
  perimeter = 46 := by
  sorry

#check rectangle_perimeter

end rectangle_perimeter_l2937_293721


namespace problem_solution_l2937_293740

-- Define the propositions
def p : Prop := ∀ x > 0, 3^x > 1
def q : Prop := ∀ a, a < -2 → (∃ x ∈ Set.Icc (-1) 2, a * x + 3 = 0) ∧
                    ¬(∀ a, (∃ x ∈ Set.Icc (-1) 2, a * x + 3 = 0) → a < -2)

-- Theorem statement
theorem problem_solution :
  (¬p ↔ ∃ x > 0, 3^x ≤ 1) ∧
  ¬p ∧
  q :=
sorry

end problem_solution_l2937_293740


namespace stratified_sampling_l2937_293796

theorem stratified_sampling (total_employees : ℕ) (male_employees : ℕ) (sample_size : ℕ) :
  total_employees = 750 →
  male_employees = 300 →
  sample_size = 45 →
  (sample_size - (male_employees * sample_size / total_employees) : ℕ) = 27 := by sorry

end stratified_sampling_l2937_293796


namespace negation_of_implication_l2937_293795

theorem negation_of_implication (x y : ℝ) : 
  ¬(x = 0 ∧ y = 0 → x * y = 0) ↔ (¬(x = 0 ∧ y = 0) → x * y ≠ 0) := by
sorry

end negation_of_implication_l2937_293795


namespace complex_linear_combination_l2937_293788

theorem complex_linear_combination (a b : ℂ) (h1 : a = 3 + 2*I) (h2 : b = 2 - 3*I) :
  2*a + 3*b = 12 - 5*I := by
  sorry

end complex_linear_combination_l2937_293788


namespace vector_magnitude_l2937_293797

/-- Given vectors a and b, if a is collinear with a + b, then |a - b| = 2√5 -/
theorem vector_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![3, x]
  (∃ (k : ℝ), a = k • (a + b)) →
  ‖a - b‖ = 2 * Real.sqrt 5 := by
sorry

end vector_magnitude_l2937_293797


namespace line_direction_vector_l2937_293746

def point_a : ℝ × ℝ := (-3, 1)
def point_b : ℝ × ℝ := (2, 5)

def direction_vector (b : ℝ) : ℝ × ℝ := (1, b)

theorem line_direction_vector : 
  ∃ (b : ℝ), (point_b.1 - point_a.1, point_b.2 - point_a.2) = 
    (point_b.1 - point_a.1) • direction_vector b ∧ b = 4/5 := by
  sorry

end line_direction_vector_l2937_293746


namespace average_age_combined_l2937_293701

theorem average_age_combined (n_students : Nat) (n_parents : Nat)
  (avg_age_students : ℚ) (avg_age_parents : ℚ)
  (h1 : n_students = 50)
  (h2 : n_parents = 75)
  (h3 : avg_age_students = 10)
  (h4 : avg_age_parents = 40) :
  (n_students * avg_age_students + n_parents * avg_age_parents) / (n_students + n_parents : ℚ) = 28 := by
  sorry

end average_age_combined_l2937_293701


namespace fraction_sum_integer_l2937_293766

theorem fraction_sum_integer (n : ℕ+) (h : ∃ (k : ℤ), (1/4 : ℚ) + (1/5 : ℚ) + (1/10 : ℚ) + (1/(n : ℚ)) = k) : n = 20 := by
  sorry

end fraction_sum_integer_l2937_293766


namespace exactly_two_valid_positions_l2937_293743

/-- Represents a position where an additional square can be placed -/
inductive Position
| Left : Position
| Right : Position
| Top : Position
| Bottom : Position
| FrontLeft : Position
| FrontRight : Position

/-- Represents the 'F' shape configuration -/
structure FShape :=
  (squares : Fin 6 → Unit)

/-- Represents the modified shape with an additional square -/
structure ModifiedShape :=
  (base : FShape)
  (additional_square : Position)

/-- Predicate to check if a modified shape can be folded into a valid 3D structure -/
def can_fold_to_valid_structure (shape : ModifiedShape) : Prop :=
  sorry

/-- The main theorem stating there are exactly two valid positions -/
theorem exactly_two_valid_positions :
  ∃ (p₁ p₂ : Position), p₁ ≠ p₂ ∧
    (∀ (shape : ModifiedShape),
      can_fold_to_valid_structure shape ↔ shape.additional_square = p₁ ∨ shape.additional_square = p₂) :=
sorry

end exactly_two_valid_positions_l2937_293743


namespace mathlon_solution_l2937_293779

/-- A Mathlon competition with M events and three participants -/
structure Mathlon where
  M : ℕ
  p₁ : ℕ
  p₂ : ℕ
  p₃ : ℕ
  scoreA : ℕ
  scoreB : ℕ
  scoreC : ℕ
  B_won_100m : Bool

/-- The conditions of the Mathlon problem -/
def mathlon_conditions (m : Mathlon) : Prop :=
  m.M > 0 ∧
  m.p₁ > m.p₂ ∧ m.p₂ > m.p₃ ∧ m.p₃ > 0 ∧
  m.scoreA = 22 ∧ m.scoreB = 9 ∧ m.scoreC = 9 ∧
  m.B_won_100m = true

/-- The theorem to prove -/
theorem mathlon_solution (m : Mathlon) (h : mathlon_conditions m) : 
  m.M = 5 ∧ ∃ (events : Fin m.M → Fin 3), 
    (∃ i, events i = 1) ∧  -- B wins one event (100m)
    (∃ i, events i = 2)    -- C is second in one event (high jump)
    := by sorry

end mathlon_solution_l2937_293779


namespace area_of_curve_l2937_293759

-- Define the curve
def curve (x y : ℝ) : Prop := |x - 1| + |y - 1| = 1

-- Define the area enclosed by the curve
noncomputable def enclosed_area : ℝ := sorry

-- Theorem statement
theorem area_of_curve : enclosed_area = 2 := by sorry

end area_of_curve_l2937_293759


namespace M_factor_count_l2937_293728

def M : ℕ := 2^6 * 3^5 * 5^3 * 7^4 * 11^1

def count_factors (n : ℕ) : ℕ := sorry

theorem M_factor_count : count_factors M = 1680 := by sorry

end M_factor_count_l2937_293728


namespace rem_five_sevenths_three_fourths_l2937_293760

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem rem_five_sevenths_three_fourths :
  rem (5/7) (3/4) = 5/7 := by sorry

end rem_five_sevenths_three_fourths_l2937_293760


namespace infinitely_many_n_divides_2_pow_n_plus_2_l2937_293738

theorem infinitely_many_n_divides_2_pow_n_plus_2 :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, n > 0 ∧ n ∣ 2^n + 2 :=
by sorry

end infinitely_many_n_divides_2_pow_n_plus_2_l2937_293738


namespace fourth_month_sales_l2937_293753

def sales_1 : ℕ := 2500
def sales_2 : ℕ := 6500
def sales_3 : ℕ := 9855
def sales_5 : ℕ := 7000
def sales_6 : ℕ := 11915
def average_sale : ℕ := 7500
def num_months : ℕ := 6

theorem fourth_month_sales (sales_4 : ℕ) : 
  (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = average_sale → 
  sales_4 = 14230 := by
  sorry

end fourth_month_sales_l2937_293753


namespace distinct_pair_count_l2937_293724

theorem distinct_pair_count :
  let S := Finset.range 15
  (S.card * (S.card - 1) : ℕ) = 210 := by sorry

end distinct_pair_count_l2937_293724


namespace no_solution_iff_b_geq_neg_four_thirds_l2937_293733

theorem no_solution_iff_b_geq_neg_four_thirds (b : ℝ) : 
  (∀ a x : ℝ, a > 1 → a^(2 - 2*x^2) + (b + 4)*a^(1 - x^2) + 3*b + 4 ≠ 0) ↔ 
  b ≥ -4/3 :=
sorry

end no_solution_iff_b_geq_neg_four_thirds_l2937_293733


namespace amount_left_after_purchases_l2937_293767

def calculate_discounted_price (price : ℚ) (discount_percent : ℚ) : ℚ :=
  price * (1 - discount_percent / 100)

def initial_amount : ℚ := 60

def frame_price : ℚ := 15
def frame_discount : ℚ := 10

def wheel_price : ℚ := 25
def wheel_discount : ℚ := 5

def seat_price : ℚ := 8
def seat_discount : ℚ := 15

def handlebar_price : ℚ := 5
def handlebar_discount : ℚ := 0

def bell_price : ℚ := 3
def bell_discount : ℚ := 0

def hat_price : ℚ := 10
def hat_discount : ℚ := 25

def total_cost : ℚ :=
  calculate_discounted_price frame_price frame_discount +
  calculate_discounted_price wheel_price wheel_discount +
  calculate_discounted_price seat_price seat_discount +
  calculate_discounted_price handlebar_price handlebar_discount +
  calculate_discounted_price bell_price bell_discount +
  calculate_discounted_price hat_price hat_discount

theorem amount_left_after_purchases :
  initial_amount - total_cost = 45 / 100 := by sorry

end amount_left_after_purchases_l2937_293767


namespace tv_show_production_cost_l2937_293722

/-- Calculates the total cost of producing all episodes of a TV show with the given conditions -/
theorem tv_show_production_cost :
  let num_seasons : ℕ := 5
  let first_season_cost : ℕ := 100000
  let other_season_cost : ℕ := 2 * first_season_cost
  let first_season_episodes : ℕ := 12
  let other_season_episodes : ℕ := first_season_episodes + (first_season_episodes / 2)
  let last_season_episodes : ℕ := 24
  
  let first_season_total : ℕ := first_season_episodes * first_season_cost
  let other_seasons_episodes : ℕ := other_season_episodes * (num_seasons - 2) + last_season_episodes
  let other_seasons_total : ℕ := other_seasons_episodes * other_season_cost
  
  first_season_total + other_seasons_total = 16800000 :=
by sorry

end tv_show_production_cost_l2937_293722


namespace average_of_middle_two_l2937_293791

theorem average_of_middle_two (n₁ n₂ n₃ n₄ n₅ n₆ : ℝ) : 
  (n₁ + n₂ + n₃ + n₄ + n₅ + n₆) / 6 = 3.95 →
  (n₁ + n₂) / 2 = 3.6 →
  (n₅ + n₆) / 2 = 4.400000000000001 →
  (n₃ + n₄) / 2 = 3.85 :=
by sorry

end average_of_middle_two_l2937_293791


namespace ellipse_m_range_l2937_293742

/-- The equation of an ellipse with parameter m -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (16 - m) + y^2 / (m + 4) = 1

/-- The condition for the equation to represent an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  (16 - m > 0) ∧ (m + 4 > 0) ∧ (16 - m ≠ m + 4)

/-- Theorem stating the range of m for which the equation represents an ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m ↔ (m > -4 ∧ m < 16 ∧ m ≠ 6) :=
sorry

end ellipse_m_range_l2937_293742


namespace power_five_mod_thirteen_l2937_293782

theorem power_five_mod_thirteen : 5^2006 % 13 = 12 := by
  sorry

end power_five_mod_thirteen_l2937_293782


namespace cost_price_calculation_l2937_293710

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 207)
  (h2 : profit_percentage = 0.15) : 
  ∃ (cost_price : ℝ), cost_price = 180 ∧ selling_price = cost_price * (1 + profit_percentage) :=
sorry

end cost_price_calculation_l2937_293710


namespace quadratic_real_roots_range_l2937_293720

theorem quadratic_real_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 2 * x^2 - 3 * x + m = 0 ∧ 2 * y^2 - 3 * y + m = 0) ↔ m ≤ 9/8 :=
by sorry

end quadratic_real_roots_range_l2937_293720


namespace tshirt_purchase_cost_l2937_293727

theorem tshirt_purchase_cost : 
  let num_fandoms : ℕ := 4
  let shirts_per_fandom : ℕ := 5
  let original_price : ℚ := 15
  let initial_discount : ℚ := 0.2
  let additional_discount : ℚ := 0.1
  let seasonal_discount : ℚ := 0.25
  let seasonal_discount_portion : ℚ := 0.5
  let tax_rate : ℚ := 0.1

  let total_shirts := num_fandoms * shirts_per_fandom
  let original_total := total_shirts * original_price
  let after_initial_discount := original_total * (1 - initial_discount)
  let after_additional_discount := after_initial_discount * (1 - additional_discount)
  let seasonal_discount_amount := (original_total * seasonal_discount_portion) * seasonal_discount
  let after_all_discounts := after_additional_discount - seasonal_discount_amount
  let final_cost := after_all_discounts * (1 + tax_rate)

  final_cost = 196.35 := by sorry

end tshirt_purchase_cost_l2937_293727


namespace triangle_side_length_l2937_293741

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  B = π / 6 →  -- 30° in radians
  (1 / 2) * a * c * Real.sin B = 3 / 2 →  -- Area formula
  Real.sin A + Real.sin C = 2 * Real.sin B →  -- Given condition
  b = Real.sqrt 3 + 1 := by
  sorry

end triangle_side_length_l2937_293741


namespace tshirts_sold_l2937_293763

def profit_per_tshirt : ℕ := 62
def total_tshirt_profit : ℕ := 11346

theorem tshirts_sold : ℕ := by
  have h : profit_per_tshirt * 183 = total_tshirt_profit := by sorry
  exact 183

#check tshirts_sold

end tshirts_sold_l2937_293763


namespace function_value_at_two_l2937_293755

/-- Given a function f: ℝ → ℝ satisfying f(x) + 2f(1/x) = 3x for all x ∈ ℝ, prove that f(2) = -3/2 -/
theorem function_value_at_two (f : ℝ → ℝ) (h : ∀ x : ℝ, f x + 2 * f (1/x) = 3 * x) : f 2 = -3/2 := by
  sorry

end function_value_at_two_l2937_293755


namespace line_vector_at_4_l2937_293780

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_at_4 :
  (∃ (a d : ℝ × ℝ × ℝ),
    (∀ t : ℝ, line_vector t = a + t • d) ∧
    line_vector (-2) = (2, 6, 16) ∧
    line_vector 1 = (-1, -4, -8)) →
  line_vector 4 = (-4, -10, -32) :=
by sorry

end line_vector_at_4_l2937_293780


namespace symmetry_implies_axis_l2937_293768

/-- A function g : ℝ → ℝ with the property that g(x) = g(3-x) for all x ∈ ℝ -/
def SymmetricFunction (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x = g (3 - x)

/-- The line x = 1.5 is an axis of symmetry for g -/
def IsAxisOfSymmetry (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x = g (3 - x)

theorem symmetry_implies_axis (g : ℝ → ℝ) (h : SymmetricFunction g) :
  IsAxisOfSymmetry g := by sorry

end symmetry_implies_axis_l2937_293768
