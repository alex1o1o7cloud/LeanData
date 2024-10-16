import Mathlib

namespace NUMINAMATH_CALUDE_passengers_taken_proof_l3463_346339

/-- The number of trains per hour -/
def trains_per_hour : ℕ := 12

/-- The number of passengers each train leaves at the station -/
def passengers_left_per_train : ℕ := 200

/-- The total number of passengers stepping on and off in an hour -/
def total_passengers_per_hour : ℕ := 6240

/-- The number of passengers each train takes from the station -/
def passengers_taken_per_train : ℕ := 320

theorem passengers_taken_proof :
  passengers_taken_per_train * trains_per_hour + 
  passengers_left_per_train * trains_per_hour = 
  total_passengers_per_hour :=
by sorry

end NUMINAMATH_CALUDE_passengers_taken_proof_l3463_346339


namespace NUMINAMATH_CALUDE_equal_sequence_l3463_346399

theorem equal_sequence (x : Fin 2011 → ℝ) (x' : Fin 2011 → ℝ) 
  (h1 : ∀ i : Fin 2011, x i + x (i + 1) = 2 * x' i)
  (h2 : ∃ σ : Equiv.Perm (Fin 2011), ∀ i, x' i = x (σ i)) :
  ∀ i j : Fin 2011, x i = x j :=
by sorry

end NUMINAMATH_CALUDE_equal_sequence_l3463_346399


namespace NUMINAMATH_CALUDE_probability_all_different_digits_l3463_346376

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_all_different_digits (n : ℕ) : Prop :=
  is_three_digit_number n ∧
  let digits := [n / 100, (n / 10) % 10, n % 10]
  digits.toFinset.card = 3

def count_three_digit_numbers : ℕ := 999 - 100 + 1

def count_numbers_with_all_different_digits : ℕ := 675

theorem probability_all_different_digits :
  (count_numbers_with_all_different_digits : ℚ) / count_three_digit_numbers = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_different_digits_l3463_346376


namespace NUMINAMATH_CALUDE_butterfingers_count_l3463_346331

theorem butterfingers_count (total : ℕ) (snickers : ℕ) (mars : ℕ) (butterfingers : ℕ) : 
  total = 12 → snickers = 3 → mars = 2 → total = snickers + mars + butterfingers →
  butterfingers = 7 := by
sorry

end NUMINAMATH_CALUDE_butterfingers_count_l3463_346331


namespace NUMINAMATH_CALUDE_sum_of_two_squares_l3463_346306

theorem sum_of_two_squares (u : ℕ) (h : Odd u) :
  ∃ (a b : ℕ), (3^(3*u) - 1) / (3^u - 1) = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_l3463_346306


namespace NUMINAMATH_CALUDE_banana_price_reduction_theorem_l3463_346397

/-- Represents the price reduction scenario for bananas -/
structure BananaPriceReduction where
  reduced_price_per_dozen : ℝ
  additional_bananas : ℕ
  additional_cost : ℝ

/-- Calculates the percentage reduction in banana prices -/
def calculate_percentage_reduction (scenario : BananaPriceReduction) : ℝ :=
  -- The implementation is not provided as per the instructions
  sorry

/-- Theorem stating that the percentage reduction is 60% given the specified conditions -/
theorem banana_price_reduction_theorem (scenario : BananaPriceReduction) 
  (h1 : scenario.reduced_price_per_dozen = 3.84)
  (h2 : scenario.additional_bananas = 50)
  (h3 : scenario.additional_cost = 40) : 
  calculate_percentage_reduction scenario = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_price_reduction_theorem_l3463_346397


namespace NUMINAMATH_CALUDE_service_cost_per_vehicle_service_cost_is_2_10_l3463_346348

/-- Represents the problem of calculating service cost per vehicle at a fuel station -/
theorem service_cost_per_vehicle (fuel_cost_per_liter : ℝ) (mini_van_tank : ℝ) 
  (truck_tank_multiplier : ℝ) (total_cost : ℝ) (num_mini_vans : ℕ) (num_trucks : ℕ) : ℝ :=
  let truck_tank := mini_van_tank * (1 + truck_tank_multiplier)
  let total_fuel := num_mini_vans * mini_van_tank + num_trucks * truck_tank
  let fuel_cost := total_fuel * fuel_cost_per_liter
  let total_service_cost := total_cost - fuel_cost
  total_service_cost / (num_mini_vans + num_trucks)

/-- The service cost per vehicle is $2.10 -/
theorem service_cost_is_2_10 : 
  service_cost_per_vehicle 0.7 65 1.2 347.2 3 2 = 2.1 := by
  sorry

end NUMINAMATH_CALUDE_service_cost_per_vehicle_service_cost_is_2_10_l3463_346348


namespace NUMINAMATH_CALUDE_no_intersection_at_roots_l3463_346325

theorem no_intersection_at_roots : ∀ x : ℝ, 
  (x^2 - 3*x + 2 = 0) → 
  ¬(x^2 - 1 = 3*x - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_no_intersection_at_roots_l3463_346325


namespace NUMINAMATH_CALUDE_ratio_problem_l3463_346387

theorem ratio_problem (A B C D : ℝ) 
  (hA : A = 0.40 * B) 
  (hB : B = 0.25 * C) 
  (hD : D = 0.60 * C) : 
  A / D = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3463_346387


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l3463_346316

theorem fixed_point_exponential_function (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 3
  f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l3463_346316


namespace NUMINAMATH_CALUDE_det_B_squared_minus_3B_l3463_346320

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : 
  Matrix.det ((B ^ 2) - 3 • B) = -704 := by sorry

end NUMINAMATH_CALUDE_det_B_squared_minus_3B_l3463_346320


namespace NUMINAMATH_CALUDE_trig_identity_l3463_346321

theorem trig_identity (α : ℝ) : 
  (1 + 1 / Real.cos (2 * α) + Real.tan (2 * α)) * (1 - 1 / Real.cos (2 * α) + Real.tan (2 * α)) = 2 * Real.tan (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3463_346321


namespace NUMINAMATH_CALUDE_cylinder_height_relation_l3463_346361

-- Define the cylinders
def Cylinder (r h : ℝ) := r > 0 ∧ h > 0

-- Theorem statement
theorem cylinder_height_relation 
  (r₁ h₁ r₂ h₂ : ℝ) 
  (cyl₁ : Cylinder r₁ h₁) 
  (cyl₂ : Cylinder r₂ h₂) 
  (volume_eq : r₁^2 * h₁ = r₂^2 * h₂) 
  (radius_relation : r₂ = 1.2 * r₁) : 
  h₁ = 1.44 * h₂ := by
sorry

end NUMINAMATH_CALUDE_cylinder_height_relation_l3463_346361


namespace NUMINAMATH_CALUDE_area_ratio_circumference_ratio_l3463_346358

-- Define a circular park
structure CircularPark where
  diameter : ℝ
  diameter_pos : diameter > 0

-- Define the enlarged park
def enlargedPark (park : CircularPark) : CircularPark :=
  { diameter := 3 * park.diameter
    diameter_pos := by
      have h : park.diameter > 0 := park.diameter_pos
      linarith }

-- Theorem for area ratio
theorem area_ratio (park : CircularPark) :
  (enlargedPark park).diameter^2 / park.diameter^2 = 9 := by
sorry

-- Theorem for circumference ratio
theorem circumference_ratio (park : CircularPark) :
  (enlargedPark park).diameter / park.diameter = 3 := by
sorry

end NUMINAMATH_CALUDE_area_ratio_circumference_ratio_l3463_346358


namespace NUMINAMATH_CALUDE_water_dumped_calculation_l3463_346370

/-- Calculates the amount of water dumped out of a bathtub given specific conditions --/
theorem water_dumped_calculation (faucet_rate : ℝ) (evaporation_rate : ℝ) (time : ℝ) (water_left : ℝ) : 
  faucet_rate = 40 ∧ 
  evaporation_rate = 200 / 60 ∧ 
  time = 9 * 60 ∧ 
  water_left = 7800 → 
  (faucet_rate * time - evaporation_rate * time - water_left) / 1000 = 12 := by
  sorry


end NUMINAMATH_CALUDE_water_dumped_calculation_l3463_346370


namespace NUMINAMATH_CALUDE_amaya_total_score_l3463_346300

/-- Represents the scores in different subjects -/
structure Scores where
  music : ℕ
  social_studies : ℕ
  arts : ℕ
  maths : ℕ

/-- Calculates the total score across all subjects -/
def total_score (s : Scores) : ℕ :=
  s.music + s.social_studies + s.arts + s.maths

/-- Theorem stating the total score given the conditions -/
theorem amaya_total_score :
  ∀ s : Scores,
  s.music = 70 →
  s.social_studies = s.music + 10 →
  s.maths = s.arts - 20 →
  s.maths = (9 * s.arts) / 10 →
  total_score s = 530 := by
  sorry

#check amaya_total_score

end NUMINAMATH_CALUDE_amaya_total_score_l3463_346300


namespace NUMINAMATH_CALUDE_extended_segment_endpoint_l3463_346341

/-- Given a segment AB with endpoints A(3, 3) and B(15, 9), extended through B to point C
    such that BC = 1/2 * AB, the coordinates of point C are (21, 12). -/
theorem extended_segment_endpoint (A B C : ℝ × ℝ) : 
  A = (3, 3) → B = (15, 9) → 
  (C.1 - B.1, C.2 - B.2) = (1/2 * (B.1 - A.1), 1/2 * (B.2 - A.2)) →
  C = (21, 12) := by
  sorry

end NUMINAMATH_CALUDE_extended_segment_endpoint_l3463_346341


namespace NUMINAMATH_CALUDE_unique_solution_for_a_l3463_346301

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2|

-- Define the function p(x, a)
def p (x a : ℝ) : ℝ := |x| + a

-- Define the domain of f
def D_f : Set ℝ := {x | x ≠ 2 ∧ x ≠ -1}

-- Theorem statement
theorem unique_solution_for_a (a : ℝ) :
  a ∈ Set.Ioo (-2) 0 ∪ Set.Ioo 0 2 ↔
  ∃! (x : ℝ), x ∈ D_f ∧ f x = p x a :=
sorry

end NUMINAMATH_CALUDE_unique_solution_for_a_l3463_346301


namespace NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l3463_346362

-- Define the conversion factor from yards to feet
def yards_to_feet : ℝ := 3

-- Define the volume in cubic yards
def volume_cubic_yards : ℝ := 6

-- Theorem to prove
theorem cubic_yards_to_cubic_feet :
  volume_cubic_yards * (yards_to_feet ^ 3) = 162 := by
  sorry

end NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l3463_346362


namespace NUMINAMATH_CALUDE_expand_product_l3463_346356

theorem expand_product (x : ℝ) : (x + 3) * (x + 4 + 6) = x^2 + 13*x + 30 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3463_346356


namespace NUMINAMATH_CALUDE_right_triangle_three_colors_l3463_346389

-- Define the color type
inductive Color
| Red
| Green
| Blue

-- Define the point type
structure Point where
  x : Int
  y : Int

-- Define the coloring function
def coloring : Point → Color := sorry

-- Define the property that all three colors are present
def all_colors_present : Prop :=
  ∃ (p1 p2 p3 : Point), coloring p1 ≠ coloring p2 ∧ coloring p2 ≠ coloring p3 ∧ coloring p3 ≠ coloring p1

-- Define a right-angled triangle
def is_right_triangle (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0 ∨
  (p1.x - p2.x) * (p3.x - p2.x) + (p1.y - p2.y) * (p3.y - p2.y) = 0 ∨
  (p1.x - p3.x) * (p2.x - p3.x) + (p1.y - p3.y) * (p2.y - p3.y) = 0

-- Theorem statement
theorem right_triangle_three_colors (h : all_colors_present) :
  ∃ (p1 p2 p3 : Point), is_right_triangle p1 p2 p3 ∧
    coloring p1 ≠ coloring p2 ∧ coloring p2 ≠ coloring p3 ∧ coloring p3 ≠ coloring p1 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_three_colors_l3463_346389


namespace NUMINAMATH_CALUDE_sock_order_ratio_l3463_346312

/-- Represents the number of pairs of socks and their prices -/
structure SockOrder where
  grey_pairs : ℕ
  white_pairs : ℕ
  white_price : ℝ

/-- Calculates the total cost of a sock order -/
def total_cost (order : SockOrder) : ℝ :=
  order.grey_pairs * (3 * order.white_price) + order.white_pairs * order.white_price

theorem sock_order_ratio (order : SockOrder) :
  order.grey_pairs = 6 →
  total_cost { grey_pairs := order.white_pairs, white_pairs := order.grey_pairs, white_price := order.white_price } = 1.25 * total_cost order →
  (order.grey_pairs : ℚ) / order.white_pairs = 6 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sock_order_ratio_l3463_346312


namespace NUMINAMATH_CALUDE_carrot_weight_calculation_l3463_346347

/-- The weight of carrots installed by the merchant -/
def carrot_weight : ℝ := sorry

/-- The total weight of all vegetables installed -/
def total_weight : ℝ := 36

/-- The weight of zucchini installed -/
def zucchini_weight : ℝ := 13

/-- The weight of broccoli installed -/
def broccoli_weight : ℝ := 8

/-- The weight of vegetables sold -/
def sold_weight : ℝ := 18

theorem carrot_weight_calculation :
  (carrot_weight + zucchini_weight + broccoli_weight = total_weight) ∧
  (total_weight = 2 * sold_weight) →
  carrot_weight = 15 := by sorry

end NUMINAMATH_CALUDE_carrot_weight_calculation_l3463_346347


namespace NUMINAMATH_CALUDE_next_two_terms_l3463_346364

def arithmetic_sequence (a₀ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₀ + n * d

def is_arithmetic_sequence (seq : ℕ → ℕ) (a₀ d : ℕ) : Prop :=
  ∀ n, seq n = arithmetic_sequence a₀ d n

theorem next_two_terms
  (seq : ℕ → ℕ)
  (h : is_arithmetic_sequence seq 3 4)
  (h0 : seq 0 = 3)
  (h1 : seq 1 = 7)
  (h2 : seq 2 = 11)
  (h3 : seq 3 = 15)
  (h4 : seq 4 = 19)
  (h5 : seq 5 = 23) :
  seq 6 = 27 ∧ seq 7 = 31 := by
sorry

end NUMINAMATH_CALUDE_next_two_terms_l3463_346364


namespace NUMINAMATH_CALUDE_max_cube_sum_on_unit_circle_l3463_346382

theorem max_cube_sum_on_unit_circle :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧
  (∀ x y : ℝ, x^2 + y^2 = 1 → |x^3| + |y^3| ≤ M) ∧
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ |x^3| + |y^3| = M) := by
sorry

end NUMINAMATH_CALUDE_max_cube_sum_on_unit_circle_l3463_346382


namespace NUMINAMATH_CALUDE_polynomial_inequality_l3463_346391

/-- A polynomial satisfying the given property -/
def GoodPolynomial (p : ℝ → ℝ) : Prop :=
  ∀ x, p (x + 1) - p x = x^100

/-- The main theorem to prove -/
theorem polynomial_inequality (p : ℝ → ℝ) (hp : GoodPolynomial p) :
  ∀ t, 0 ≤ t → t ≤ 1/2 → p (1 - t) ≥ p t := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l3463_346391


namespace NUMINAMATH_CALUDE_toy_store_revenue_l3463_346304

theorem toy_store_revenue (december : ℝ) (november january : ℝ) 
  (h1 : november = (3/5) * december) 
  (h2 : january = (1/3) * november) : 
  december = (5/2) * ((november + january) / 2) := by
  sorry

end NUMINAMATH_CALUDE_toy_store_revenue_l3463_346304


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l3463_346326

/-- Represents the selling and buying of a bicycle through two transactions -/
def bicycle_sales (initial_cost : ℝ) : Prop :=
  let first_sale := initial_cost * 1.5
  let final_sale := first_sale * 1.25
  final_sale = 225

theorem bicycle_cost_price : ∃ (initial_cost : ℝ), 
  bicycle_sales initial_cost ∧ initial_cost = 120 := by sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l3463_346326


namespace NUMINAMATH_CALUDE_test_passing_difference_l3463_346336

theorem test_passing_difference (total : ℕ) (arithmetic : ℕ) (algebra : ℕ) (geometry : ℕ)
  (arithmetic_correct : ℚ) (algebra_correct : ℚ) (geometry_correct : ℚ) (passing_grade : ℚ)
  (h1 : total = 90)
  (h2 : arithmetic = 20)
  (h3 : algebra = 40)
  (h4 : geometry = 30)
  (h5 : arithmetic_correct = 60 / 100)
  (h6 : algebra_correct = 50 / 100)
  (h7 : geometry_correct = 70 / 100)
  (h8 : passing_grade = 65 / 100)
  (h9 : total = arithmetic + algebra + geometry) :
  ⌈total * passing_grade⌉ - (⌊arithmetic * arithmetic_correct⌋ + ⌊algebra * algebra_correct⌋ + ⌊geometry * geometry_correct⌋) = 6 := by
  sorry

end NUMINAMATH_CALUDE_test_passing_difference_l3463_346336


namespace NUMINAMATH_CALUDE_russells_earnings_l3463_346323

/-- Proof of Russell's earnings --/
theorem russells_earnings (vika_earnings breanna_earnings saheed_earnings kayla_earnings russell_earnings : ℕ) : 
  vika_earnings = 84 →
  kayla_earnings = vika_earnings - 30 →
  saheed_earnings = 4 * kayla_earnings →
  breanna_earnings = saheed_earnings + (saheed_earnings / 4) →
  russell_earnings = 2 * (breanna_earnings - kayla_earnings) →
  russell_earnings = 432 := by
  sorry

end NUMINAMATH_CALUDE_russells_earnings_l3463_346323


namespace NUMINAMATH_CALUDE_product_sum_difference_problem_l3463_346390

theorem product_sum_difference_problem (P Q R S : ℕ) : 
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S →
  P > 0 ∧ Q > 0 ∧ R > 0 ∧ S > 0 →
  P * Q = 72 →
  R * S = 72 →
  P + Q = R - S →
  P = 4 ∨ Q = 4 ∨ R = 4 ∨ S = 4 :=
by sorry

#check product_sum_difference_problem

end NUMINAMATH_CALUDE_product_sum_difference_problem_l3463_346390


namespace NUMINAMATH_CALUDE_expression_rationality_expression_rationality_iff_l3463_346318

theorem expression_rationality (x : ℚ) : ∃ (k : ℚ), 
  x^2 + (Real.sqrt (x^2 + 1))^2 - 1 / (x^2 + (Real.sqrt (x^2 + 1))^2) = k := by
  sorry

theorem expression_rationality_iff : 
  ∀ x : ℝ, (∃ k : ℚ, x^2 + (Real.sqrt (x^2 + 1))^2 - 1 / (x^2 + (Real.sqrt (x^2 + 1))^2) = k) ↔ 
  ∃ q : ℚ, x = q := by
  sorry

end NUMINAMATH_CALUDE_expression_rationality_expression_rationality_iff_l3463_346318


namespace NUMINAMATH_CALUDE_multiplication_value_problem_l3463_346340

theorem multiplication_value_problem : 
  ∃ x : ℝ, (4.5 / 6) * x = 9 ∧ x = 12 := by
sorry

end NUMINAMATH_CALUDE_multiplication_value_problem_l3463_346340


namespace NUMINAMATH_CALUDE_grocery_store_bottles_l3463_346359

/-- The total number of soda bottles in a grocery store. -/
def total_bottles (regular : ℕ) (diet : ℕ) (lite : ℕ) : ℕ :=
  regular + diet + lite

/-- Theorem stating that the total number of bottles is 110. -/
theorem grocery_store_bottles : total_bottles 57 26 27 = 110 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_bottles_l3463_346359


namespace NUMINAMATH_CALUDE_task_completion_time_l3463_346392

theorem task_completion_time (a b c : ℝ) 
  (h1 : 1/a + 1/b = 1/2)
  (h2 : 1/b + 1/c = 1/4)
  (h3 : 1/c + 1/a = 5/12) :
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_task_completion_time_l3463_346392


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l3463_346398

theorem negative_sixty_four_to_four_thirds (x : ℝ) : x = (-64)^(4/3) → x = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l3463_346398


namespace NUMINAMATH_CALUDE_intersecting_squares_area_difference_l3463_346351

theorem intersecting_squares_area_difference : 
  let s1 : ℕ := 12
  let s2 : ℕ := 9
  let s3 : ℕ := 7
  let s4 : ℕ := 3
  s1^2 + s3^2 - (s2^2 + s4^2) = 103 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_squares_area_difference_l3463_346351


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3463_346329

/-- Given an ellipse with equation 4x^2 + y^2 = 16, its major axis has length 8 -/
theorem ellipse_major_axis_length :
  ∀ (x y : ℝ), 4 * x^2 + y^2 = 16 → ∃ (a b : ℝ), 
    a > b ∧ 
    x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    2 * a = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3463_346329


namespace NUMINAMATH_CALUDE_initial_number_proof_l3463_346308

theorem initial_number_proof (N : ℤ) : (N + 3) % 24 = 0 → N = 21 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l3463_346308


namespace NUMINAMATH_CALUDE_largest_angle_convex_hexagon_l3463_346344

theorem largest_angle_convex_hexagon (x : ℝ) :
  (x + 2) + (2 * x + 3) + (3 * x + 4) + (4 * x + 5) + (5 * x + 6) + (6 * x + 7) = 720 →
  max (x + 2) (max (2 * x + 3) (max (3 * x + 4) (max (4 * x + 5) (max (5 * x + 6) (6 * x + 7))))) = 205 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_convex_hexagon_l3463_346344


namespace NUMINAMATH_CALUDE_system_solution_unique_l3463_346342

theorem system_solution_unique :
  ∃! (x y : ℝ), 
    3 * x^2 + 4 * x * y + 12 * y^2 + 16 * y = -6 ∧
    x^2 - 12 * x * y + 4 * y^2 - 10 * x + 12 * y = -7 ∧
    x = 1/2 ∧ y = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3463_346342


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l3463_346322

theorem roots_quadratic_equation (m n : ℝ) : 
  (m^2 + 5*m + 3 = 0) → 
  (n^2 + 5*n + 3 = 0) → 
  m * Real.sqrt (n / m) + n * Real.sqrt (m / n) = -2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l3463_346322


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3463_346381

/-- A sequence is geometric if the ratio between any two consecutive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence {a_n} where a_4 + a_6 = 8, 
    prove that a_1a_7 + 2a_3a_7 + a_3a_9 = 64. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
    (h_geometric : IsGeometric a) 
    (h_sum : a 4 + a 6 = 8) : 
    a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3463_346381


namespace NUMINAMATH_CALUDE_max_x_minus_y_l3463_346350

theorem max_x_minus_y (x y : ℝ) (h : 2 * (x^2 + y^2) = x + y + 2*x*y) : 
  ∃ (M : ℝ), M = 2 ∧ ∀ (a b : ℝ), 2 * (a^2 + b^2) = a + b + 2*a*b → a - b ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l3463_346350


namespace NUMINAMATH_CALUDE_probability_of_second_defective_given_first_defective_l3463_346327

-- Define the total number of items
def total_items : ℕ := 20

-- Define the number of good items
def good_items : ℕ := 16

-- Define the number of defective items
def defective_items : ℕ := 4

-- Define the probability of drawing a defective item on the first draw
def prob_first_defective : ℚ := defective_items / total_items

-- Define the probability of drawing a defective item on the second draw given the first was defective
def prob_second_defective_given_first_defective : ℚ := (defective_items - 1) / (total_items - 1)

-- Theorem statement
theorem probability_of_second_defective_given_first_defective :
  prob_second_defective_given_first_defective = 3 / 19 :=
sorry

end NUMINAMATH_CALUDE_probability_of_second_defective_given_first_defective_l3463_346327


namespace NUMINAMATH_CALUDE_convergence_rate_l3463_346338

def v : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => (3/2) * v n - (3/2) * (v n)^2

theorem convergence_rate (k : ℕ) : k ≥ 5 ↔ |v k - 1/2| ≤ 1/2^20 :=
sorry

end NUMINAMATH_CALUDE_convergence_rate_l3463_346338


namespace NUMINAMATH_CALUDE_triangle_angle_B_is_pi_over_three_l3463_346352

theorem triangle_angle_B_is_pi_over_three 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_given : (c - b) / (c - a) = Real.sin A / (Real.sin C + Real.sin B)) :
  B = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_B_is_pi_over_three_l3463_346352


namespace NUMINAMATH_CALUDE_combined_work_rate_l3463_346360

/-- The combined work rate of three workers given their individual work rates -/
theorem combined_work_rate 
  (rate_A : ℚ) 
  (rate_B : ℚ) 
  (rate_C : ℚ) 
  (h_A : rate_A = 1 / 12)
  (h_B : rate_B = 1 / 6)
  (h_C : rate_C = 1 / 18) : 
  rate_A + rate_B + rate_C = 11 / 36 := by
  sorry

#check combined_work_rate

end NUMINAMATH_CALUDE_combined_work_rate_l3463_346360


namespace NUMINAMATH_CALUDE_fourth_term_largest_l3463_346307

theorem fourth_term_largest (x : ℝ) : 
  (5/8 < x ∧ x < 20/21) ↔ 
  (∀ k : ℕ, k ≠ 4 → 
    Nat.choose 10 3 * (5^7) * (3*x)^3 ≥ Nat.choose 10 (k-1) * (5^(10-(k-1))) * (3*x)^(k-1)) :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_largest_l3463_346307


namespace NUMINAMATH_CALUDE_absolute_value_and_exponents_l3463_346353

theorem absolute_value_and_exponents : |-4| + (Real.pi - Real.sqrt 2) ^ 0 - (1/2)⁻¹ = 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_exponents_l3463_346353


namespace NUMINAMATH_CALUDE_greatest_five_digit_divisible_by_sum_of_digits_l3463_346345

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

/-- Checks if a number is divisible by the sum of its digits -/
def isDivisibleBySumOfDigits (n : ℕ) : Prop :=
  n % sumOfDigits n = 0

/-- Theorem: 99972 is the greatest five-digit number divisible by the sum of its digits -/
theorem greatest_five_digit_divisible_by_sum_of_digits :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ isDivisibleBySumOfDigits n → n ≤ 99972 :=
by sorry

end NUMINAMATH_CALUDE_greatest_five_digit_divisible_by_sum_of_digits_l3463_346345


namespace NUMINAMATH_CALUDE_greatest_n_with_divisibility_conditions_l3463_346314

theorem greatest_n_with_divisibility_conditions :
  ∃ (n : ℕ), n < 1000 ∧
  (Int.floor (Real.sqrt n) - 2 : ℤ) ∣ (n - 4 : ℤ) ∧
  (Int.floor (Real.sqrt n) + 2 : ℤ) ∣ (n + 4 : ℤ) ∧
  (∀ (m : ℕ), m < 1000 →
    (Int.floor (Real.sqrt m) - 2 : ℤ) ∣ (m - 4 : ℤ) →
    (Int.floor (Real.sqrt m) + 2 : ℤ) ∣ (m + 4 : ℤ) →
    m ≤ n) ∧
  n = 956 :=
sorry

end NUMINAMATH_CALUDE_greatest_n_with_divisibility_conditions_l3463_346314


namespace NUMINAMATH_CALUDE_alpha_squared_gt_beta_squared_l3463_346335

theorem alpha_squared_gt_beta_squared 
  (α β : Real) 
  (h1 : α ∈ Set.Icc (-π/2) (π/2)) 
  (h2 : β ∈ Set.Icc (-π/2) (π/2)) 
  (h3 : α * Real.sin α - β * Real.sin β > 0) : 
  α^2 > β^2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_squared_gt_beta_squared_l3463_346335


namespace NUMINAMATH_CALUDE_clerical_staff_reduction_l3463_346380

theorem clerical_staff_reduction (total_employees : ℕ) 
  (initial_clerical_fraction : ℚ) (final_clerical_fraction : ℚ) 
  (h1 : total_employees = 3600)
  (h2 : initial_clerical_fraction = 1/3)
  (h3 : final_clerical_fraction = 1/5) : 
  ∃ (f : ℚ), 
    (initial_clerical_fraction * total_employees) * (1 - f) = 
    final_clerical_fraction * (total_employees - initial_clerical_fraction * total_employees * f) ∧ 
    f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_clerical_staff_reduction_l3463_346380


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3463_346357

theorem cube_root_equation_solution (y : ℝ) : 
  (15 * y + (15 * y + 15) ^ (1/3 : ℝ)) ^ (1/3 : ℝ) = 15 → y = 224 := by
sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3463_346357


namespace NUMINAMATH_CALUDE_interest_rate_first_part_l3463_346368

/-- Given a total amount of 3200, divided into two parts where the first part is 800
    and the second part is at 5% interest rate, and the total annual interest is 144,
    prove that the interest rate of the first part is 3%. -/
theorem interest_rate_first_part (total : ℕ) (first_part : ℕ) (second_part : ℕ) 
  (second_rate : ℚ) (total_interest : ℕ) :
  total = 3200 →
  first_part = 800 →
  second_part = total - first_part →
  second_rate = 5 / 100 →
  total_interest = 144 →
  ∃ (first_rate : ℚ), 
    first_rate * first_part / 100 + second_rate * second_part = total_interest ∧
    first_rate = 3 / 100 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_first_part_l3463_346368


namespace NUMINAMATH_CALUDE_negation_of_all_birds_can_fly_l3463_346385

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (bird : U → Prop)
variable (can_fly : U → Prop)

-- State the theorem
theorem negation_of_all_birds_can_fly :
  (¬ ∀ (x : U), bird x → can_fly x) ↔ (∃ (x : U), bird x ∧ ¬ can_fly x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_birds_can_fly_l3463_346385


namespace NUMINAMATH_CALUDE_area_difference_triangles_l3463_346396

/-- Given a right-angled triangle with base 3 and height 9, and another right-angled triangle
    with base 6 and height 9, prove that the difference between the areas of the triangles formed
    by a line intersecting both hypotenuses is 13.5 square units. -/
theorem area_difference_triangles (A B C D F H : ℝ × ℝ) : 
  -- ΔFAH and ΔHBC are right triangles
  (F.1 - A.1) * (H.2 - A.2) = (H.1 - A.1) * (F.2 - A.2) →
  (H.1 - B.1) * (C.2 - B.2) = (C.1 - B.1) * (H.2 - B.2) →
  -- AH = 6
  (H.1 - A.1)^2 + (H.2 - A.2)^2 = 36 →
  -- HB = 3
  (B.1 - H.1)^2 + (B.2 - H.2)^2 = 9 →
  -- FC = 9
  (C.1 - F.1)^2 + (C.2 - F.2)^2 = 81 →
  -- AC and HF intersect at D
  ∃ t : ℝ, D = (1 - t) • A + t • C ∧ ∃ s : ℝ, D = (1 - s) • H + s • F →
  -- The difference between the areas of ΔADF and ΔBDC is 13.5
  abs ((A.1 * (F.2 - D.2) + D.1 * (A.2 - F.2) + F.1 * (D.2 - A.2)) / 2 -
       (B.1 * (C.2 - D.2) + D.1 * (B.2 - C.2) + C.1 * (D.2 - B.2)) / 2) = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_area_difference_triangles_l3463_346396


namespace NUMINAMATH_CALUDE_power_multiplication_l3463_346333

theorem power_multiplication (x : ℝ) : x^5 * x^6 = x^11 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3463_346333


namespace NUMINAMATH_CALUDE_unique_p_for_natural_roots_l3463_346311

def cubic_equation (p : ℝ) (x : ℝ) : ℝ :=
  5 * x^3 - 5 * (p + 1) * x^2 + (71 * p - 1) * x + 1 - 66 * p

theorem unique_p_for_natural_roots :
  ∃! p : ℝ, p = 76 ∧
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  cubic_equation p x = 0 ∧
  cubic_equation p y = 0 ∧
  cubic_equation p z = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_p_for_natural_roots_l3463_346311


namespace NUMINAMATH_CALUDE_max_value_condition_l3463_346305

/-- The expression 2005 - (x + y)^2 takes its maximum value when x = -y -/
theorem max_value_condition (x y : ℝ) : 
  (∀ a b : ℝ, 2005 - (x + y)^2 ≥ 2005 - (a + b)^2) → x = -y := by
sorry

end NUMINAMATH_CALUDE_max_value_condition_l3463_346305


namespace NUMINAMATH_CALUDE_largest_valid_number_l3463_346349

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100 = 8) ∧
  (∀ d, 0 < d ∧ d < 10 ∧ (n / 10 % 10 = d ∨ n % 10 = d) → n % d = 0)

theorem largest_valid_number : 
  is_valid_number 864 ∧ ∀ n, is_valid_number n → n ≤ 864 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l3463_346349


namespace NUMINAMATH_CALUDE_factorization_proofs_l3463_346363

theorem factorization_proofs (x y a b : ℝ) : 
  (2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2) ∧ 
  (a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a + b) * (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proofs_l3463_346363


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3463_346393

theorem min_value_quadratic (x : ℝ) : 
  ∃ (m : ℝ), m = 702 ∧ ∀ x, 3 * x^2 - 18 * x + 729 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3463_346393


namespace NUMINAMATH_CALUDE_central_angle_measure_l3463_346328

-- Define the sector properties
def arc_length : ℝ := 4
def sector_area : ℝ := 2

-- Theorem statement
theorem central_angle_measure :
  ∀ (r θ : ℝ),
  r > 0 →
  sector_area = 1/2 * r * arc_length →
  arc_length = r * θ →
  θ = 4 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_measure_l3463_346328


namespace NUMINAMATH_CALUDE_closest_fraction_l3463_346303

def medals_won : ℚ := 23 / 150

def fractions : List ℚ := [1/6, 1/7, 1/8, 1/9, 1/10]

theorem closest_fraction :
  (fractions.argmin (λ x => |x - medals_won|)).get! = 1/7 := by sorry

end NUMINAMATH_CALUDE_closest_fraction_l3463_346303


namespace NUMINAMATH_CALUDE_pinball_spend_proof_l3463_346355

def half_dollar : ℚ := 0.5

def wednesday_spend : ℕ := 4
def thursday_spend : ℕ := 14

def total_spend : ℚ := (wednesday_spend * half_dollar) + (thursday_spend * half_dollar)

theorem pinball_spend_proof : total_spend = 9 := by
  sorry

end NUMINAMATH_CALUDE_pinball_spend_proof_l3463_346355


namespace NUMINAMATH_CALUDE_tan_product_from_cos_sum_diff_l3463_346313

theorem tan_product_from_cos_sum_diff (α β : Real) 
  (h1 : Real.cos (α + β) = 1/5)
  (h2 : Real.cos (α - β) = 3/5) : 
  Real.tan α * Real.tan β = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_from_cos_sum_diff_l3463_346313


namespace NUMINAMATH_CALUDE_suit_price_increase_l3463_346354

/-- Proves that the percentage increase in the price of a suit was 20% --/
theorem suit_price_increase (original_price : ℝ) (coupon_discount : ℝ) (final_price : ℝ) :
  original_price = 150 →
  coupon_discount = 0.2 →
  final_price = 144 →
  ∃ (increase_percentage : ℝ),
    increase_percentage = 20 ∧
    final_price = (1 - coupon_discount) * (original_price * (1 + increase_percentage / 100)) :=
by sorry

end NUMINAMATH_CALUDE_suit_price_increase_l3463_346354


namespace NUMINAMATH_CALUDE_expression_independence_l3463_346372

theorem expression_independence (x a b c : ℝ) 
  (hxa : x ≠ a) (hxb : x ≠ b) (hxc : x ≠ c) : 
  (x - a) * (x - b) * (x - c) * 
  ((a - b) / (x - c) + (b - c) / (x - a) + (c - a) / (x - b)) = 
  (b - a) * (a - c) * (c - b) := by
  sorry

end NUMINAMATH_CALUDE_expression_independence_l3463_346372


namespace NUMINAMATH_CALUDE_total_amount_distributed_l3463_346337

/-- Given an equal distribution of money among 22 persons, where each person
    receives Rs 1,950, prove that the total amount distributed is Rs 42,900. -/
theorem total_amount_distributed
  (num_persons : ℕ) (amount_per_person : ℕ) (h1 : num_persons = 22) (h2 : amount_per_person = 1950) :
  num_persons * amount_per_person = 42900 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_distributed_l3463_346337


namespace NUMINAMATH_CALUDE_grazing_problem_solution_l3463_346343

/-- Represents the grazing scenario with oxen and rent -/
structure GrazingScenario where
  a_oxen : ℕ
  a_months : ℕ
  b_oxen : ℕ
  b_months : ℕ
  c_oxen : ℕ
  c_months : ℕ
  total_rent : ℚ
  c_rent : ℚ

/-- Calculates the total oxen-months for a given scenario -/
def total_oxen_months (s : GrazingScenario) : ℕ :=
  s.a_oxen * s.a_months + s.b_oxen * s.b_months + s.c_oxen * s.c_months

/-- Theorem stating the solution to the grazing problem -/
theorem grazing_problem_solution (s : GrazingScenario) 
  (h1 : s.a_oxen = 10)
  (h2 : s.a_months = 7)
  (h3 : s.b_oxen = 12)
  (h4 : s.c_oxen = 15)
  (h5 : s.c_months = 3)
  (h6 : s.total_rent = 245)
  (h7 : s.c_rent = 62.99999999999999)
  : s.b_months = 5 := by
  sorry


end NUMINAMATH_CALUDE_grazing_problem_solution_l3463_346343


namespace NUMINAMATH_CALUDE_log_comparison_l3463_346377

theorem log_comparison : Real.log 2009 / Real.log 2008 > Real.log 2010 / Real.log 2009 := by sorry

end NUMINAMATH_CALUDE_log_comparison_l3463_346377


namespace NUMINAMATH_CALUDE_solution_set_equality_l3463_346346

theorem solution_set_equality : 
  {x : ℝ | (x - 3) * (x + 2) < 0} = {x : ℝ | -2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3463_346346


namespace NUMINAMATH_CALUDE_satisfying_polynomial_form_l3463_346334

/-- A polynomial with real coefficients satisfying the given equality for all real a, b, c -/
def SatisfyingPolynomial (p : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, p (a + b - 2*c) + p (b + c - 2*a) + p (c + a - 2*b) = 
               3 * (p (a - b) + p (b - c) + p (c - a))

/-- The theorem stating that any satisfying polynomial has the form a₂x² + a₁x -/
theorem satisfying_polynomial_form (p : ℝ → ℝ) (h : SatisfyingPolynomial p) :
  ∃ a₂ a₁ : ℝ, ∀ x : ℝ, p x = a₂ * x^2 + a₁ * x :=
sorry

end NUMINAMATH_CALUDE_satisfying_polynomial_form_l3463_346334


namespace NUMINAMATH_CALUDE_logarithm_sum_l3463_346395

theorem logarithm_sum (a b : ℝ) (ha : a = Real.log 8) (hb : b = Real.log 25) :
  5^(a/b) + 2^(b/a) = 2 * Real.sqrt 2 + 5^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_l3463_346395


namespace NUMINAMATH_CALUDE_unique_solution_system_l3463_346309

theorem unique_solution_system :
  ∃! (x y : ℝ), (x + y = (3 - x) + (3 - y)) ∧ (x - y = (x - 2) + (y - 2)) ∧ x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3463_346309


namespace NUMINAMATH_CALUDE_triangle_arithmetic_geometric_sequence_l3463_346332

theorem triangle_arithmetic_geometric_sequence (A B C : ℝ) (a b c : ℝ) : 
  -- Angles form an arithmetic sequence
  2 * B = A + C →
  -- Sum of angles in a triangle
  A + B + C = π →
  -- Sides form a geometric sequence
  b^2 = a * c →
  -- Law of cosines
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  -- Conclusions
  Real.cos B = 1 / 2 ∧ Real.sin A * Real.sin C = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_geometric_sequence_l3463_346332


namespace NUMINAMATH_CALUDE_derivative_independent_of_function_value_l3463_346374

variable (f : ℝ → ℝ)
variable (x₀ : ℝ)

theorem derivative_independent_of_function_value :
  ∃ (g : ℝ → ℝ), g x₀ ≠ f x₀ ∧ HasDerivAt g (deriv f x₀) x₀ :=
sorry

end NUMINAMATH_CALUDE_derivative_independent_of_function_value_l3463_346374


namespace NUMINAMATH_CALUDE_basketball_score_proof_l3463_346384

theorem basketball_score_proof (two_points : ℕ) (three_points : ℕ) (free_throws : ℕ) :
  (3 * three_points = 2 * (2 * two_points)) →
  (free_throws = 2 * two_points) →
  (2 * two_points + 3 * three_points + free_throws = 72) →
  free_throws = 18 := by
sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l3463_346384


namespace NUMINAMATH_CALUDE_adult_ticket_price_is_five_l3463_346386

/-- Represents the ticket sales for a baseball game at a community center -/
structure TicketSales where
  total_tickets : ℕ
  adult_tickets : ℕ
  child_ticket_price : ℕ
  total_revenue : ℕ

/-- The price of an adult ticket given the ticket sales information -/
def adult_ticket_price (sales : TicketSales) : ℕ :=
  (sales.total_revenue - (sales.total_tickets - sales.adult_tickets) * sales.child_ticket_price) / sales.adult_tickets

/-- Theorem stating that the adult ticket price is $5 given the specific sales information -/
theorem adult_ticket_price_is_five :
  let sales : TicketSales := {
    total_tickets := 85,
    adult_tickets := 35,
    child_ticket_price := 2,
    total_revenue := 275
  }
  adult_ticket_price sales = 5 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_price_is_five_l3463_346386


namespace NUMINAMATH_CALUDE_cake_pieces_l3463_346373

theorem cake_pieces (cake_length : ℕ) (cake_width : ℕ) (piece_length : ℕ) (piece_width : ℕ) :
  cake_length = 24 →
  cake_width = 20 →
  piece_length = 3 →
  piece_width = 2 →
  (cake_length * cake_width) / (piece_length * piece_width) = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_cake_pieces_l3463_346373


namespace NUMINAMATH_CALUDE_tom_teaching_years_l3463_346302

theorem tom_teaching_years :
  ∀ (tom_years devin_years : ℕ),
    tom_years + devin_years = 70 →
    devin_years = tom_years / 2 - 5 →
    tom_years = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_tom_teaching_years_l3463_346302


namespace NUMINAMATH_CALUDE_max_pots_is_ten_l3463_346366

/-- Represents the number of items Susan can buy -/
structure Purchase where
  pins : ℕ
  pans : ℕ
  pots : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  3 * p.pins + 4 * p.pans + 9 * p.pots

/-- Checks if a purchase is valid according to the problem constraints -/
def isValidPurchase (p : Purchase) : Prop :=
  p.pins ≥ 1 ∧ p.pans ≥ 1 ∧ p.pots ≥ 1 ∧ totalCost p = 100

/-- Theorem stating that the maximum number of pots Susan can buy is 10 -/
theorem max_pots_is_ten :
  ∀ p : Purchase, isValidPurchase p → p.pots ≤ 10 ∧ 
  ∃ q : Purchase, isValidPurchase q ∧ q.pots = 10 :=
sorry

end NUMINAMATH_CALUDE_max_pots_is_ten_l3463_346366


namespace NUMINAMATH_CALUDE_michael_earnings_l3463_346394

/-- Calculates the total money earned from selling birdhouses --/
def total_money_earned (large_price medium_price small_price : ℕ) 
                       (large_sold medium_sold small_sold : ℕ) : ℕ :=
  large_price * large_sold + medium_price * medium_sold + small_price * small_sold

/-- Theorem: Michael's earnings from selling birdhouses --/
theorem michael_earnings : 
  total_money_earned 22 16 7 2 2 3 = 97 := by sorry

end NUMINAMATH_CALUDE_michael_earnings_l3463_346394


namespace NUMINAMATH_CALUDE_negation_equivalence_l3463_346315

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀ ≤ 0 ∧ x₀^2 ≥ 0) ↔ (∀ x : ℝ, x ≤ 0 → x^2 < 0) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3463_346315


namespace NUMINAMATH_CALUDE_mans_speed_with_stream_l3463_346310

/-- Given a man's rowing speed against the stream and his speed in still water,
    calculate his speed with the stream. -/
theorem mans_speed_with_stream
  (speed_against_stream : ℝ)
  (speed_still_water : ℝ)
  (h1 : speed_against_stream = 4)
  (h2 : speed_still_water = 5) :
  speed_still_water + (speed_still_water - speed_against_stream) = 6 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_with_stream_l3463_346310


namespace NUMINAMATH_CALUDE_min_value_z_l3463_346330

theorem min_value_z (x y : ℝ) (h1 : x - y + 5 ≥ 0) (h2 : x + y ≥ 0) (h3 : x ≤ 3) :
  ∀ z : ℝ, z = (x + y + 2) / (x + 3) → z ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_min_value_z_l3463_346330


namespace NUMINAMATH_CALUDE_min_omega_for_overlapping_sine_graphs_l3463_346371

/-- Given a function f(x) = sin(ωx + π/3) where ω > 0, if the graph of y = f(x) is shifted
    to the right by 2π/3 units and overlaps with the original graph, then the minimum
    value of ω is 3. -/
theorem min_omega_for_overlapping_sine_graphs (ω : ℝ) (f : ℝ → ℝ) :
  ω > 0 →
  (∀ x, f x = Real.sin (ω * x + π / 3)) →
  (∀ x, f (x + 2 * π / 3) = f x) →
  3 ≤ ω ∧ ∀ ω', (ω' > 0 ∧ ∀ x, f (x + 2 * π / 3) = f x) → ω ≤ ω' :=
by sorry

end NUMINAMATH_CALUDE_min_omega_for_overlapping_sine_graphs_l3463_346371


namespace NUMINAMATH_CALUDE_cost_of_18_pencils_13_notebooks_l3463_346379

/-- The cost of a pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a notebook -/
def notebook_cost : ℝ := sorry

/-- The first given condition: 9 pencils and 11 notebooks cost $6.05 -/
axiom condition1 : 9 * pencil_cost + 11 * notebook_cost = 6.05

/-- The second given condition: 6 pencils and 4 notebooks cost $2.68 -/
axiom condition2 : 6 * pencil_cost + 4 * notebook_cost = 2.68

/-- Theorem: The cost of 18 pencils and 13 notebooks is $8.45 -/
theorem cost_of_18_pencils_13_notebooks :
  18 * pencil_cost + 13 * notebook_cost = 8.45 := by sorry

end NUMINAMATH_CALUDE_cost_of_18_pencils_13_notebooks_l3463_346379


namespace NUMINAMATH_CALUDE_arrangement_count_5_2_l3463_346317

/-- The number of ways to arrange n distinct objects and m pairs of 2 distinct objects each in a row,
    where the objects within each pair must be adjacent -/
def arrangementCount (n m : ℕ) : ℕ :=
  Nat.factorial (n + m) * (Nat.factorial 2)^m

/-- Theorem: The number of ways to arrange 5 distinct objects and 2 pairs of 2 distinct objects each
    in a row, where the objects within each pair must be adjacent, is equal to 7! * (2!)^2 -/
theorem arrangement_count_5_2 :
  arrangementCount 5 2 = 20160 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_5_2_l3463_346317


namespace NUMINAMATH_CALUDE_gcf_of_54_and_72_l3463_346383

theorem gcf_of_54_and_72 : Nat.gcd 54 72 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_54_and_72_l3463_346383


namespace NUMINAMATH_CALUDE_ron_sold_twelve_tickets_l3463_346369

/-- Represents the ticket sales problem with Ron and Kathy --/
structure TicketSales where
  ron_price : ℝ
  kathy_price : ℝ
  total_tickets : ℕ
  total_income : ℝ

/-- Theorem stating that Ron sold 12 tickets given the problem conditions --/
theorem ron_sold_twelve_tickets (ts : TicketSales) 
  (h1 : ts.ron_price = 2)
  (h2 : ts.kathy_price = 4.5)
  (h3 : ts.total_tickets = 20)
  (h4 : ts.total_income = 60) : 
  ∃ (ron_tickets : ℕ) (kathy_tickets : ℕ), 
    ron_tickets + kathy_tickets = ts.total_tickets ∧ 
    ron_tickets * ts.ron_price + kathy_tickets * ts.kathy_price = ts.total_income ∧
    ron_tickets = 12 := by
  sorry

end NUMINAMATH_CALUDE_ron_sold_twelve_tickets_l3463_346369


namespace NUMINAMATH_CALUDE_jeremy_oranges_l3463_346367

theorem jeremy_oranges (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) : 
  tuesday = 3 * monday →
  wednesday = 70 →
  monday + tuesday + wednesday = 470 →
  monday = 100 := by
sorry

end NUMINAMATH_CALUDE_jeremy_oranges_l3463_346367


namespace NUMINAMATH_CALUDE_number_puzzle_l3463_346324

theorem number_puzzle : ∃ x : ℤ, (x - 10 = 15) ∧ (x + 5 = 30) := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3463_346324


namespace NUMINAMATH_CALUDE_y_equivalent_condition_l3463_346378

theorem y_equivalent_condition (x y : ℝ) :
  y = 2 * x + 4 →
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 8) ↔ 
  (y ∈ Set.Icc (-6) 6 ∪ Set.Icc 14 26) :=
by sorry

end NUMINAMATH_CALUDE_y_equivalent_condition_l3463_346378


namespace NUMINAMATH_CALUDE_repeating_decimal_three_three_six_l3463_346319

/-- Represents a repeating decimal where the decimal part repeats infinitely -/
def RepeatingDecimal (whole : ℤ) (repeating : ℕ) : ℚ :=
  whole + (repeating : ℚ) / (99 : ℚ)

/-- The statement that 3.363636... equals 37/11 -/
theorem repeating_decimal_three_three_six : RepeatingDecimal 3 36 = 37 / 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_three_three_six_l3463_346319


namespace NUMINAMATH_CALUDE_evaluate_expression_l3463_346365

theorem evaluate_expression (x y : ℝ) (hx : x = 3) (hy : y = 0) : y * (y - 3 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3463_346365


namespace NUMINAMATH_CALUDE_olivia_weekly_earnings_l3463_346375

/-- Calculates the total earnings for a week given an hourly rate and hours worked on specific days -/
def weeklyEarnings (hourlyRate : ℕ) (mondayHours wednesdayHours fridayHours : ℕ) : ℕ :=
  hourlyRate * (mondayHours + wednesdayHours + fridayHours)

/-- Proves that Olivia's earnings for the week equal $117 -/
theorem olivia_weekly_earnings :
  weeklyEarnings 9 4 3 6 = 117 := by
  sorry

end NUMINAMATH_CALUDE_olivia_weekly_earnings_l3463_346375


namespace NUMINAMATH_CALUDE_cafe_prices_l3463_346388

/-- The cost of items at a roadside cafe -/
structure CafePrices where
  sandwich : ℕ
  coffee : ℕ
  donut : ℕ

/-- The problem statement -/
theorem cafe_prices (p : CafePrices) : 
  4 * p.sandwich + p.coffee + 10 * p.donut = 169 ∧ 
  3 * p.sandwich + p.coffee + 7 * p.donut = 126 →
  p.sandwich + p.coffee + p.donut = 40 := by
  sorry

end NUMINAMATH_CALUDE_cafe_prices_l3463_346388
