import Mathlib

namespace parabola_vertex_l2296_229688

/-- The parabola defined by y = -(x+2)^2 + 6 has its vertex at (-2, 6) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -(x + 2)^2 + 6 → (∀ t : ℝ, y ≤ -(t + 2)^2 + 6) → (x = -2 ∧ y = 6) :=
by sorry

end parabola_vertex_l2296_229688


namespace odd_number_probability_l2296_229605

-- Define a fair six-sided die
def FairDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the set of odd numbers on the die
def OddNumbers : Finset ℕ := {1, 3, 5}

-- Theorem: The probability of rolling an odd number is 1/2
theorem odd_number_probability :
  (Finset.card OddNumbers : ℚ) / (Finset.card FairDie : ℚ) = 1 / 2 := by
  sorry

end odd_number_probability_l2296_229605


namespace intersection_A_B_l2296_229623

def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 2}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} := by sorry

end intersection_A_B_l2296_229623


namespace fibonacci_inequality_l2296_229627

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_inequality (n : ℕ) :
  (fibonacci (n + 1) : ℝ) ^ (1 / n : ℝ) ≥ 1 + (fibonacci n : ℝ) ^ (-1 / n : ℝ) := by
  sorry

end fibonacci_inequality_l2296_229627


namespace complex_order_multiplication_property_l2296_229624

-- Define the order relation on complex numbers
def complex_order (z1 z2 : ℂ) : Prop :=
  z1.re > z2.re ∨ (z1.re = z2.re ∧ z1.im > z2.im)

-- Define the statement to be proven false
theorem complex_order_multiplication_property (z z1 z2 : ℂ) :
  ¬(complex_order z 0 → complex_order z1 z2 → complex_order (z * z1) (z * z2)) :=
sorry

end complex_order_multiplication_property_l2296_229624


namespace two_rotations_top_left_to_top_right_l2296_229625

/-- Represents the corners of a rectangle --/
inductive Corner
  | TopLeft
  | TopRight
  | BottomRight
  | BottomLeft

/-- Represents the rotation of a rectangle around a regular pentagon --/
def rotateAroundPentagon (n : ℕ) (startCorner : Corner) : Corner :=
  match n % 4 with
  | 0 => startCorner
  | 1 => match startCorner with
    | Corner.TopLeft => Corner.TopRight
    | Corner.TopRight => Corner.BottomRight
    | Corner.BottomRight => Corner.BottomLeft
    | Corner.BottomLeft => Corner.TopLeft
  | 2 => match startCorner with
    | Corner.TopLeft => Corner.BottomRight
    | Corner.TopRight => Corner.BottomLeft
    | Corner.BottomRight => Corner.TopLeft
    | Corner.BottomLeft => Corner.TopRight
  | 3 => match startCorner with
    | Corner.TopLeft => Corner.BottomLeft
    | Corner.TopRight => Corner.TopLeft
    | Corner.BottomRight => Corner.TopRight
    | Corner.BottomLeft => Corner.BottomRight
  | _ => startCorner  -- This case should never occur due to % 4

/-- Theorem stating that after two full rotations, an object at the top left corner ends up at the top right corner --/
theorem two_rotations_top_left_to_top_right :
  rotateAroundPentagon 2 Corner.TopLeft = Corner.TopRight :=
by sorry


end two_rotations_top_left_to_top_right_l2296_229625


namespace three_Z_five_equals_fourteen_l2296_229652

-- Define the operation Z
def Z (a b : ℤ) : ℤ := b + 11*a - a^2 - a*b

-- Theorem statement
theorem three_Z_five_equals_fourteen : Z 3 5 = 14 := by
  sorry

end three_Z_five_equals_fourteen_l2296_229652


namespace goals_scored_over_two_days_l2296_229663

/-- The total number of goals scored by Gina and Tom over two days -/
def total_goals (gina_day1 gina_day2 tom_day1 tom_day2 : ℕ) : ℕ :=
  gina_day1 + gina_day2 + tom_day1 + tom_day2

/-- Theorem stating the total number of goals scored by Gina and Tom -/
theorem goals_scored_over_two_days :
  ∃ (gina_day1 gina_day2 tom_day1 tom_day2 : ℕ),
    gina_day1 = 2 ∧
    tom_day1 = gina_day1 + 3 ∧
    tom_day2 = 6 ∧
    gina_day2 = tom_day2 - 2 ∧
    total_goals gina_day1 gina_day2 tom_day1 tom_day2 = 17 :=
by
  sorry

end goals_scored_over_two_days_l2296_229663


namespace root_expression_equals_five_l2296_229613

theorem root_expression_equals_five (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (h1 : a - 5 * Real.sqrt a + 2 = 0)
  (h2 : b - 5 * Real.sqrt b + 2 = 0) :
  (a * Real.sqrt a + b * Real.sqrt b) / (a - b) *
  (2 / Real.sqrt a - 2 / Real.sqrt b) /
  (Real.sqrt a - (a + b) / Real.sqrt b) +
  5 * (5 * Real.sqrt a - a) / (b + 2) = 5 := by
sorry

end root_expression_equals_five_l2296_229613


namespace base_4_representation_of_253_base_4_to_decimal_3331_l2296_229643

/-- Converts a natural number to its base 4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n < 4 then [n]
  else (n % 4) :: toBase4 (n / 4)

/-- Converts a list of base 4 digits to its decimal representation -/
def fromBase4 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 4 * acc) 0

theorem base_4_representation_of_253 :
  toBase4 253 = [1, 3, 3, 3] :=
by sorry

theorem base_4_to_decimal_3331 :
  fromBase4 [1, 3, 3, 3] = 253 :=
by sorry

end base_4_representation_of_253_base_4_to_decimal_3331_l2296_229643


namespace f_negative_when_x_greater_than_one_third_l2296_229637

def f (x : ℝ) := -3 * x + 1

theorem f_negative_when_x_greater_than_one_third :
  ∀ x : ℝ, x > 1/3 → f x < 0 := by
sorry

end f_negative_when_x_greater_than_one_third_l2296_229637


namespace area_triangle_OCD_l2296_229697

/-- Given a trapezoid ABCD and a parallelogram ABGH inscribed within it,
    this theorem calculates the area of triangle OCD. -/
theorem area_triangle_OCD (S_ABCD S_ABGH : ℝ) (h1 : S_ABCD = 320) (h2 : S_ABGH = 80) :
  ∃ (S_OCD : ℝ), S_OCD = 45 :=
by sorry

end area_triangle_OCD_l2296_229697


namespace bakery_chairs_count_l2296_229683

theorem bakery_chairs_count :
  let indoor_tables : ℕ := 8
  let outdoor_tables : ℕ := 12
  let chairs_per_indoor_table : ℕ := 3
  let chairs_per_outdoor_table : ℕ := 3
  let total_chairs := indoor_tables * chairs_per_indoor_table + outdoor_tables * chairs_per_outdoor_table
  total_chairs = 60 := by
  sorry

end bakery_chairs_count_l2296_229683


namespace bowling_ball_weight_l2296_229660

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 7 * b = 3 * c)  -- Seven bowling balls weigh the same as three canoes
  (h2 : 2 * c = 56)     -- Two canoes weigh 56 pounds
  : b = 12 :=           -- One bowling ball weighs 12 pounds
by
  sorry

#check bowling_ball_weight

end bowling_ball_weight_l2296_229660


namespace mardi_gras_necklaces_l2296_229665

/-- Proves that the total number of necklaces caught is 49 given the problem conditions --/
theorem mardi_gras_necklaces : 
  ∀ (boudreaux rhonda latch cecilia : ℕ),
  boudreaux = 12 →
  rhonda = boudreaux / 2 →
  latch = 3 * rhonda - 4 →
  cecilia = latch + 3 →
  ∃ (k : ℕ), boudreaux + rhonda + latch + cecilia = 7 * k →
  boudreaux + rhonda + latch + cecilia = 49 := by
sorry

end mardi_gras_necklaces_l2296_229665


namespace first_pair_price_is_22_l2296_229657

/-- The price of the first pair of shoes -/
def first_pair_price : ℝ := 22

/-- The price of the second pair of shoes -/
def second_pair_price : ℝ := 1.5 * first_pair_price

/-- The total price of both pairs of shoes -/
def total_price : ℝ := 55

/-- Theorem stating that the price of the first pair of shoes is $22 -/
theorem first_pair_price_is_22 :
  first_pair_price = 22 ∧
  second_pair_price = 1.5 * first_pair_price ∧
  total_price = first_pair_price + second_pair_price :=
by sorry

end first_pair_price_is_22_l2296_229657


namespace rope_cutting_l2296_229689

theorem rope_cutting (total_length : ℝ) (ratio_short : ℝ) (ratio_long : ℝ) :
  total_length = 35 →
  ratio_short = 3 →
  ratio_long = 4 →
  (ratio_long / (ratio_short + ratio_long)) * total_length = 20 :=
by
  sorry

end rope_cutting_l2296_229689


namespace range_of_a_l2296_229674

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀ + 2*a*x₀ + 2 - a = 0) → 
  a ≤ -2 ∨ a = 1 := by
sorry

end range_of_a_l2296_229674


namespace largest_consecutive_sum_bound_l2296_229600

def is_permutation (σ : Fin 100 → ℕ) : Prop :=
  Function.Bijective σ ∧ ∀ i, σ i ∈ Finset.range 101

def consecutive_sum (σ : Fin 100 → ℕ) (start : Fin 91) : ℕ :=
  (Finset.range 10).sum (λ i => σ (start + i))

theorem largest_consecutive_sum_bound :
  (∃ A : ℕ, A = 505 ∧
    (∀ σ : Fin 100 → ℕ, is_permutation σ →
      ∃ start : Fin 91, consecutive_sum σ start ≥ A) ∧
    ∀ B : ℕ, B > A →
      ∃ σ : Fin 100 → ℕ, is_permutation σ ∧
        ∀ start : Fin 91, consecutive_sum σ start < B) :=
sorry

end largest_consecutive_sum_bound_l2296_229600


namespace house_painting_cost_l2296_229634

/-- Calculates the total cost of painting a house given the areas and costs per square foot for different rooms. -/
def total_painting_cost (living_room_area : ℕ) (living_room_cost : ℕ)
                        (bedroom_area : ℕ) (bedroom_cost : ℕ)
                        (kitchen_area : ℕ) (kitchen_cost : ℕ)
                        (bathroom_area : ℕ) (bathroom_cost : ℕ) : ℕ :=
  living_room_area * living_room_cost +
  2 * bedroom_area * bedroom_cost +
  kitchen_area * kitchen_cost +
  2 * bathroom_area * bathroom_cost

/-- Theorem stating that the total cost of painting the house is 49500 Rs. -/
theorem house_painting_cost :
  total_painting_cost 600 30 450 25 300 20 100 15 = 49500 := by
  sorry

#eval total_painting_cost 600 30 450 25 300 20 100 15

end house_painting_cost_l2296_229634


namespace triangle_regions_l2296_229671

theorem triangle_regions (p : ℕ) (h_prime : Nat.Prime p) (h_ge_3 : p ≥ 3) :
  let num_lines := 3 * p
  (num_lines * (num_lines + 1)) / 2 + 1 = 3 * p^2 - 3 * p + 1 := by
  sorry

end triangle_regions_l2296_229671


namespace hyperbola_asymptotes_equation_l2296_229672

/-- Given a hyperbola and a circle intersecting to form a square, 
    prove the equation of the asymptotes of the hyperbola. -/
theorem hyperbola_asymptotes_equation 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c = Real.sqrt (a^2 + b^2)) 
  (h_hyperbola : ∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1 → x^2 + y^2 = c^2 → x^2 = y^2) :
  ∃ k : ℝ, k = Real.sqrt (Real.sqrt 2 - 1) ∧ 
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1 → y = k * x ∨ y = -k * x) :=
by sorry

end hyperbola_asymptotes_equation_l2296_229672


namespace find_m_value_l2296_229659

theorem find_m_value (m : ℝ) (h1 : m ≠ 0) :
  (∀ x : ℝ, (x^2 - m) * (x + m) = x^3 + m * (x^2 - x - 7)) →
  m = 7 := by
sorry

end find_m_value_l2296_229659


namespace carwash_problem_l2296_229640

theorem carwash_problem (car_price truck_price suv_price : ℕ) 
  (total_raised num_suvs num_trucks : ℕ) : 
  car_price = 5 → 
  truck_price = 6 → 
  suv_price = 7 → 
  total_raised = 100 → 
  num_suvs = 5 → 
  num_trucks = 5 → 
  ∃ num_cars : ℕ, 
    num_cars * car_price + num_trucks * truck_price + num_suvs * suv_price = total_raised ∧ 
    num_cars = 7 :=
by
  sorry

end carwash_problem_l2296_229640


namespace max_value_when_m_1_solution_when_m_neg_2_l2296_229639

-- Define the function f(x, m)
def f (x m : ℝ) : ℝ := |m * x + 1| - |x - 1|

-- Theorem 1: Maximum value of f(x) when m = 1
theorem max_value_when_m_1 :
  ∃ (max : ℝ), max = 2 ∧ ∀ (x : ℝ), f x 1 ≤ max :=
sorry

-- Theorem 2: Solution to f(x) ≥ 1 when m = -2
theorem solution_when_m_neg_2 :
  ∀ (x : ℝ), f x (-2) ≥ 1 ↔ x ≤ -1 ∨ x ≥ 1 :=
sorry

end max_value_when_m_1_solution_when_m_neg_2_l2296_229639


namespace max_valid_sequence_length_l2296_229662

/-- A sequence of integers satisfying the given conditions -/
def ValidSequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  (∀ i : ℕ, i + 2 < n → a i + a (i + 1) + a (i + 2) > 0) ∧
  (∀ i : ℕ, i + 4 < n → a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) < 0)

/-- The maximum length of a valid sequence is 6 -/
theorem max_valid_sequence_length :
  (∃ (a : ℕ → ℤ), ValidSequence a 6) ∧
  (∀ n : ℕ, n > 6 → ¬∃ (a : ℕ → ℤ), ValidSequence a n) :=
sorry

end max_valid_sequence_length_l2296_229662


namespace simple_random_for_small_population_systematic_for_large_uniform_population_stratified_for_population_with_strata_l2296_229673

-- Define the sampling methods
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

-- Define a structure for a sampling scenario
structure SamplingScenario where
  populationSize : ℕ
  sampleSize : ℕ
  hasStrata : Bool
  uniformDistribution : Bool

-- Define the function to determine the most appropriate sampling method
def mostAppropriateSamplingMethod (scenario : SamplingScenario) : SamplingMethod :=
  sorry

-- Theorem for the first scenario
theorem simple_random_for_small_population :
  mostAppropriateSamplingMethod { populationSize := 20, sampleSize := 4, hasStrata := false, uniformDistribution := true } = SamplingMethod.SimpleRandom :=
  sorry

-- Theorem for the second scenario
theorem systematic_for_large_uniform_population :
  mostAppropriateSamplingMethod { populationSize := 1280, sampleSize := 32, hasStrata := false, uniformDistribution := true } = SamplingMethod.Systematic :=
  sorry

-- Theorem for the third scenario
theorem stratified_for_population_with_strata :
  mostAppropriateSamplingMethod { populationSize := 180, sampleSize := 15, hasStrata := true, uniformDistribution := false } = SamplingMethod.Stratified :=
  sorry

end simple_random_for_small_population_systematic_for_large_uniform_population_stratified_for_population_with_strata_l2296_229673


namespace arithmetic_to_geometric_iff_rational_l2296_229621

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticProgression (a d : ℚ) : ℕ → ℚ := fun n ↦ a + n * d

/-- A geometric progression is a sequence where each term after the first
    is found by multiplying the previous term by a fixed, non-zero number. -/
def GeometricProgression (a r : ℚ) : ℕ → ℚ := fun n ↦ a * r^n

/-- A subsequence of a sequence is a sequence that can be derived from the original
    sequence by deleting some or no elements without changing the order of the
    remaining elements. -/
def Subsequence (f g : ℕ → ℚ) : Prop :=
  ∃ h : ℕ → ℕ, Monotone h ∧ ∀ n, f (h n) = g n

theorem arithmetic_to_geometric_iff_rational (a d : ℚ) (hd : d ≠ 0) :
  (∃ (b r : ℚ) (hr : r ≠ 1), Subsequence (ArithmeticProgression a d) (GeometricProgression b r)) ↔
  ∃ q : ℚ, a = q * d := by sorry

end arithmetic_to_geometric_iff_rational_l2296_229621


namespace simplify_expression_1_simplify_expression_2_simplify_expression_3_simplify_expression_4_l2296_229620

-- (1)
theorem simplify_expression_1 : 3 * Real.sqrt 20 - Real.sqrt 45 - Real.sqrt (1/5) = (14 * Real.sqrt 5) / 5 := by sorry

-- (2)
theorem simplify_expression_2 : (Real.sqrt 6 * Real.sqrt 3) / Real.sqrt 2 - 1 = 2 := by sorry

-- (3)
theorem simplify_expression_3 : Real.sqrt 16 + 327 - 2 * Real.sqrt (1/4) = 330 := by sorry

-- (4)
theorem simplify_expression_4 : (Real.sqrt 3 - Real.sqrt 5) * (Real.sqrt 5 + Real.sqrt 3) - (Real.sqrt 5 - Real.sqrt 3)^2 = 2 * Real.sqrt 15 - 6 := by sorry

end simplify_expression_1_simplify_expression_2_simplify_expression_3_simplify_expression_4_l2296_229620


namespace absolute_value_multiplication_l2296_229684

theorem absolute_value_multiplication : -2 * |(-3)| = -6 := by
  sorry

end absolute_value_multiplication_l2296_229684


namespace inequality_proof_l2296_229661

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a + 1 / b ≥ 4 / (a + b) := by
  sorry

end inequality_proof_l2296_229661


namespace polynomial_existence_l2296_229633

theorem polynomial_existence : ∃ (P : ℤ → ℤ), 
  (∃ (a b c d e f g h i : ℤ), ∀ x, P x = a*x^8 + b*x^7 + c*x^6 + d*x^5 + e*x^4 + f*x^3 + g*x^2 + h*x + i) ∧ 
  (∀ x : ℤ, P x ≠ 0) ∧
  (∀ n : ℕ, n > 0 → ∃ x : ℤ, (n : ℤ) ∣ P x) := by
sorry

end polynomial_existence_l2296_229633


namespace geometric_series_common_ratio_l2296_229638

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4 / 7
  let a₂ : ℚ := 16 / 21
  let r : ℚ := a₂ / a₁
  r = 4 / 3 := by sorry

end geometric_series_common_ratio_l2296_229638


namespace circle_radius_l2296_229681

/-- Given a circle with equation x^2 + y^2 - 2ax + 2 = 0 and center (2, 0), its radius is √2 -/
theorem circle_radius (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*a*x + 2 = 0 ↔ (x - 2)^2 + y^2 = 2) → 
  (∃ r : ℝ, r > 0 ∧ r^2 = 2) :=
by sorry

end circle_radius_l2296_229681


namespace sheep_in_pen_l2296_229667

theorem sheep_in_pen (total : ℕ) (rounded_up : ℕ) (wandered_off : ℕ) : 
  wandered_off = 9 →
  wandered_off = total / 10 →
  rounded_up = total * 9 / 10 →
  rounded_up = 81 := by
sorry

end sheep_in_pen_l2296_229667


namespace inequality_solution_l2296_229658

theorem inequality_solution (x : ℝ) (h1 : x > 0) 
  (h2 : x * Real.sqrt (16 - x) + Real.sqrt (16 * x - x^3) ≥ 16) : x = 4 := by
  sorry

end inequality_solution_l2296_229658


namespace cone_volume_and_surface_area_l2296_229650

/-- Represents a cone with given slant height and height --/
structure Cone where
  slant_height : ℝ
  height : ℝ

/-- Calculate the volume of a cone --/
def volume (c : Cone) : ℝ := sorry

/-- Calculate the surface area of a cone --/
def surface_area (c : Cone) : ℝ := sorry

/-- Theorem stating the volume and surface area of a specific cone --/
theorem cone_volume_and_surface_area :
  let c : Cone := { slant_height := 15, height := 9 }
  (volume c = 432 * Real.pi) ∧ (surface_area c = 324 * Real.pi) := by sorry

end cone_volume_and_surface_area_l2296_229650


namespace solve_school_supplies_problem_l2296_229692

/-- Represents the price and quantity of pens and notebooks -/
structure Supplies where
  pen_price : ℚ
  notebook_price : ℚ
  pen_quantity : ℕ
  notebook_quantity : ℕ

/-- Calculates the total cost of supplies -/
def total_cost (s : Supplies) : ℚ :=
  s.pen_price * s.pen_quantity + s.notebook_price * s.notebook_quantity

/-- Represents the conditions of the problem -/
structure ProblemConditions where
  xiaofang_cost : ℚ
  xiaoliang_cost : ℚ
  xiaofang_supplies : Supplies
  xiaoliang_supplies : Supplies
  reward_fund : ℚ
  prize_sets : ℕ

/-- Theorem stating the solution to the problem -/
theorem solve_school_supplies_problem (c : ProblemConditions)
  (h1 : c.xiaofang_cost = 18)
  (h2 : c.xiaoliang_cost = 22)
  (h3 : c.xiaofang_supplies.pen_quantity = 2)
  (h4 : c.xiaofang_supplies.notebook_quantity = 3)
  (h5 : c.xiaoliang_supplies.pen_quantity = 3)
  (h6 : c.xiaoliang_supplies.notebook_quantity = 2)
  (h7 : c.reward_fund = 400)
  (h8 : c.prize_sets = 20)
  (h9 : total_cost c.xiaofang_supplies = c.xiaofang_cost)
  (h10 : total_cost c.xiaoliang_supplies = c.xiaoliang_cost) :
  ∃ (pen_price notebook_price : ℚ) (combinations : ℕ),
    pen_price = 6 ∧
    notebook_price = 2 ∧
    combinations = 4 ∧
    (∀ x y : ℕ, (x * pen_price + y * notebook_price) * c.prize_sets = c.reward_fund →
      (x = 0 ∧ y = 10) ∨ (x = 1 ∧ y = 7) ∨ (x = 2 ∧ y = 4) ∨ (x = 3 ∧ y = 1)) :=
by sorry

end solve_school_supplies_problem_l2296_229692


namespace window_width_calculation_l2296_229698

/-- Calculates the width of each window in a room given the room dimensions,
    door dimensions, number of windows, window height, cost per square foot,
    and total cost of whitewashing. -/
theorem window_width_calculation (room_length room_width room_height : ℝ)
                                 (door_height door_width : ℝ)
                                 (num_windows : ℕ)
                                 (window_height : ℝ)
                                 (cost_per_sqft total_cost : ℝ) :
  room_length = 25 ∧ room_width = 15 ∧ room_height = 12 ∧
  door_height = 6 ∧ door_width = 3 ∧
  num_windows = 3 ∧
  window_height = 3 ∧
  cost_per_sqft = 9 ∧
  total_cost = 8154 →
  ∃ (window_width : ℝ),
    window_width = 4 ∧
    total_cost = (2 * (room_length + room_width) * room_height -
                  door_height * door_width -
                  num_windows * window_height * window_width) * cost_per_sqft :=
by sorry

end window_width_calculation_l2296_229698


namespace james_running_distance_l2296_229629

/-- Proves that given the conditions of the problem, the initial running distance was 600 miles per week -/
theorem james_running_distance (initial_distance : ℝ) : 
  (initial_distance + 40 * 3 = 1.2 * initial_distance) → 
  initial_distance = 600 := by
  sorry

end james_running_distance_l2296_229629


namespace volleyball_team_selection_l2296_229646

def total_players : ℕ := 16
def triplets : ℕ := 3
def captain : ℕ := 1
def starters : ℕ := 6

def remaining_players : ℕ := total_players - triplets - captain
def players_to_choose : ℕ := starters - triplets - captain

theorem volleyball_team_selection :
  Nat.choose remaining_players players_to_choose = 66 := by
  sorry

end volleyball_team_selection_l2296_229646


namespace crocodile_count_l2296_229686

/-- The number of frogs in the pond -/
def num_frogs : ℕ := 20

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := 52

/-- The number of eyes each animal (frog or crocodile) has -/
def eyes_per_animal : ℕ := 2

/-- The number of crocodiles in the pond -/
def num_crocodiles : ℕ := 6

theorem crocodile_count :
  num_crocodiles * eyes_per_animal + num_frogs * eyes_per_animal = total_eyes :=
by sorry

end crocodile_count_l2296_229686


namespace pythagorean_triple_check_l2296_229615

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_check : 
  ¬ is_pythagorean_triple 12 15 18 ∧
  is_pythagorean_triple 3 4 5 ∧
  ¬ is_pythagorean_triple 6 9 15 :=
sorry

end pythagorean_triple_check_l2296_229615


namespace mass_of_compound_l2296_229653

/-- Molar mass of potassium in g/mol -/
def molar_mass_K : ℝ := 39.10

/-- Molar mass of aluminum in g/mol -/
def molar_mass_Al : ℝ := 26.98

/-- Molar mass of sulfur in g/mol -/
def molar_mass_S : ℝ := 32.07

/-- Molar mass of oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Molar mass of hydrogen in g/mol -/
def molar_mass_H : ℝ := 1.01

/-- Number of moles of the compound -/
def num_moles : ℝ := 15

/-- Molar mass of potassium aluminum sulfate dodecahydrate (KAl(SO4)2·12H2O) in g/mol -/
def molar_mass_compound : ℝ := 
  molar_mass_K + molar_mass_Al + 2 * molar_mass_S + 32 * molar_mass_O + 24 * molar_mass_H

/-- Mass of the compound in grams -/
def mass_compound : ℝ := num_moles * molar_mass_compound

theorem mass_of_compound : mass_compound = 9996.9 := by
  sorry

end mass_of_compound_l2296_229653


namespace expression_simplification_l2296_229618

theorem expression_simplification (x y : ℝ) (hx : x = 2) (hy : y = 1/2) :
  (x + y) * (x - y) + (x - y)^2 - (x^2 - 3*x*y) = 5 := by sorry

end expression_simplification_l2296_229618


namespace geometric_progression_fourth_term_l2296_229632

theorem geometric_progression_fourth_term 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 2^(1/4 : ℝ)) 
  (h₂ : a₂ = 2^(1/8 : ℝ)) 
  (h₃ : a₃ = 2^(1/16 : ℝ)) 
  (h_geom : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) :
  ∃ a₄ : ℝ, a₄ = a₃ * (a₃ / a₂) ∧ a₄ = 2^(-1/16 : ℝ) :=
sorry

end geometric_progression_fourth_term_l2296_229632


namespace associative_property_only_l2296_229619

theorem associative_property_only (a b c : ℕ) : 
  (a + b) + c = a + (b + c) ↔ 
  ∃ (x y z : ℕ), x + y + z = x + (y + z) ∧ x = 57 ∧ y = 24 ∧ z = 76 :=
by sorry

end associative_property_only_l2296_229619


namespace candy_distribution_l2296_229655

theorem candy_distribution (x : ℚ) 
  (h1 : 3 * x = mia_candies)
  (h2 : 4 * mia_candies = noah_candies)
  (h3 : 6 * noah_candies = olivia_candies)
  (h4 : x + mia_candies + noah_candies + olivia_candies = 468) :
  x = 117 / 22 := by
  sorry

end candy_distribution_l2296_229655


namespace smallest_multiplier_for_perfect_square_l2296_229682

theorem smallest_multiplier_for_perfect_square (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 2 → ¬(∃ k : ℕ, 1152 * m = k * k)) ∧ 
  (∃ k : ℕ, 1152 * 2 = k * k) :=
by sorry

end smallest_multiplier_for_perfect_square_l2296_229682


namespace factorization_problem_1_factorization_problem_2_l2296_229669

-- Problem 1
theorem factorization_problem_1 (p q : ℝ) :
  6 * p^3 * q - 10 * p^2 = 2 * p^2 * (3 * p * q - 5) := by sorry

-- Problem 2
theorem factorization_problem_2 (a : ℝ) :
  a^4 - 8 * a^2 + 16 = (a + 2)^2 * (a - 2)^2 := by sorry

end factorization_problem_1_factorization_problem_2_l2296_229669


namespace intersection_probability_l2296_229664

/-- A regular decagon is a 10-sided polygon with all sides equal and all angles equal. -/
def RegularDecagon : Type := Unit

/-- The number of vertices in a regular decagon. -/
def num_vertices : ℕ := 10

/-- The number of diagonals in a regular decagon. -/
def num_diagonals (d : RegularDecagon) : ℕ := 35

/-- The number of ways to choose 2 diagonals from a regular decagon. -/
def num_diagonal_pairs (d : RegularDecagon) : ℕ := 595

/-- The number of sets of 4 points that determine intersecting diagonals. -/
def num_intersecting_sets (d : RegularDecagon) : ℕ := 210

/-- The probability that two randomly chosen diagonals of a regular decagon intersect inside the decagon. -/
theorem intersection_probability (d : RegularDecagon) : 
  (num_intersecting_sets d : ℚ) / (num_diagonal_pairs d) = 210 / 595 := by sorry

end intersection_probability_l2296_229664


namespace neither_alive_probability_l2296_229614

/-- The probability that a man will be alive for 10 more years -/
def prob_man_alive : ℚ := 1/4

/-- The probability that a woman will be alive for 10 more years -/
def prob_woman_alive : ℚ := 1/3

/-- The probability that neither the man nor the woman will be alive for 10 more years -/
def prob_neither_alive : ℚ := (1 - prob_man_alive) * (1 - prob_woman_alive)

theorem neither_alive_probability : prob_neither_alive = 1/2 := by
  sorry

end neither_alive_probability_l2296_229614


namespace shooter_probabilities_l2296_229668

/-- A shooter has a probability of hitting the target in a single shot -/
def hit_probability : ℝ := 0.5

/-- The number of shots taken -/
def num_shots : ℕ := 4

/-- The probability of hitting the target exactly k times in n shots -/
def prob_exact_hits (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * hit_probability ^ k * (1 - hit_probability) ^ (n - k)

/-- The probability of hitting the target at least once in n shots -/
def prob_at_least_one_hit (n : ℕ) : ℝ :=
  1 - (1 - hit_probability) ^ n

theorem shooter_probabilities :
  (prob_exact_hits num_shots 3 = 1/4) ∧
  (prob_at_least_one_hit num_shots = 15/16) := by
  sorry

end shooter_probabilities_l2296_229668


namespace min_value_a_l2296_229630

theorem min_value_a (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  ∃ (a : ℝ), ∀ (x y : ℝ), x > 1 → y > 1 → 
    Real.log (x * y) ≤ Real.log a * Real.sqrt (Real.log x ^ 2 + Real.log y ^ 2) ∧
    ∀ (b : ℝ), (∀ (x y : ℝ), x > 1 → y > 1 → 
      Real.log (x * y) ≤ Real.log b * Real.sqrt (Real.log x ^ 2 + Real.log y ^ 2)) → 
    a ≤ b :=
by
  sorry

end min_value_a_l2296_229630


namespace perpendicular_intersects_side_l2296_229670

/-- A regular polygon with n sides inscribed in a circle -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry
  is_inscribed : sorry

/-- The opposite side of a vertex in a regular polygon -/
def opposite_side (p : RegularPolygon 101) (i : Fin 101) : Set (ℝ × ℝ) :=
  sorry

/-- The perpendicular from a vertex to the line containing the opposite side -/
def perpendicular (p : RegularPolygon 101) (i : Fin 101) : Set (ℝ × ℝ) :=
  sorry

/-- The intersection point of the perpendicular and the line containing the opposite side -/
def intersection_point (p : RegularPolygon 101) (i : Fin 101) : ℝ × ℝ :=
  sorry

/-- Theorem: In a regular 101-gon inscribed in a circle, there exists at least one vertex 
    such that the perpendicular from this vertex to the line containing the opposite side 
    intersects the opposite side itself, not its extension -/
theorem perpendicular_intersects_side (p : RegularPolygon 101) : 
  ∃ i : Fin 101, intersection_point p i ∈ opposite_side p i :=
sorry

end perpendicular_intersects_side_l2296_229670


namespace jolene_total_earnings_l2296_229679

/-- The amount of money Jolene raised through babysitting and car washing -/
def jolene_earnings (num_families : ℕ) (babysitting_rate : ℕ) (num_cars : ℕ) (car_wash_rate : ℕ) : ℕ :=
  num_families * babysitting_rate + num_cars * car_wash_rate

/-- Theorem stating that Jolene raised $180 given the specified conditions -/
theorem jolene_total_earnings :
  jolene_earnings 4 30 5 12 = 180 := by
  sorry

end jolene_total_earnings_l2296_229679


namespace square_area_from_vertices_l2296_229628

/-- The area of a square with adjacent vertices at (0,5) and (5,0) is 50 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (0, 5)
  let p2 : ℝ × ℝ := (5, 0)
  let side_length := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let area := side_length^2
  area = 50 := by sorry


end square_area_from_vertices_l2296_229628


namespace five_digit_subtraction_l2296_229694

theorem five_digit_subtraction (a b c d e : ℕ) : 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 →
  a > 0 →
  (a * 10000 + b * 1000 + c * 100 + d * 10 + e) - 
  (e * 10000 + d * 1000 + c * 100 + b * 10 + a) = 
  (10072 : ℕ) →
  a > e →
  (∀ a' e' : ℕ, a' < 10 ∧ e' < 10 ∧ a' > e' → a' - e' ≥ a - e) →
  a = 9 ∧ e = 7 := by
sorry

end five_digit_subtraction_l2296_229694


namespace min_value_of_f_l2296_229636

def f (x : ℝ) := 3 * x^2 - 6 * x + 9

theorem min_value_of_f : 
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 6 := by
  sorry

end min_value_of_f_l2296_229636


namespace min_value_of_sum_of_products_min_value_is_achievable_l2296_229656

def is_permutation_of_1_to_9 (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℕ) : Prop :=
  ({a₁, a₂, a₃, b₁, b₂, b₃, c₁, c₂, c₃} : Finset ℕ) = Finset.range 9

theorem min_value_of_sum_of_products 
  (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℕ) 
  (h : is_permutation_of_1_to_9 a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃) : 
  a₁ * a₂ * a₃ + b₁ * b₂ * b₃ + c₁ * c₂ * c₃ ≥ 214 :=
sorry

theorem min_value_is_achievable : 
  ∃ a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℕ, 
    is_permutation_of_1_to_9 a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ ∧ 
    a₁ * a₂ * a₃ + b₁ * b₂ * b₃ + c₁ * c₂ * c₃ = 214 :=
sorry

end min_value_of_sum_of_products_min_value_is_achievable_l2296_229656


namespace marble_collection_proof_l2296_229631

/-- The number of blue marbles collected by the three friends --/
def total_blue_marbles (mary_blue : ℕ) (jenny_blue : ℕ) (anie_blue : ℕ) : ℕ :=
  mary_blue + jenny_blue + anie_blue

theorem marble_collection_proof 
  (jenny_red : ℕ) 
  (jenny_blue : ℕ) 
  (mary_red : ℕ) 
  (mary_blue : ℕ) 
  (anie_red : ℕ) 
  (anie_blue : ℕ) : 
  jenny_red = 30 →
  jenny_blue = 25 →
  mary_red = 2 * jenny_red →
  anie_red = mary_red + 20 →
  anie_blue = 2 * jenny_blue →
  mary_blue = anie_blue / 2 →
  total_blue_marbles mary_blue jenny_blue anie_blue = 100 := by
  sorry

#check marble_collection_proof

end marble_collection_proof_l2296_229631


namespace quadrilateral_wx_length_l2296_229691

-- Define the quadrilateral WXYZ
structure Quadrilateral :=
  (W X Y Z : ℝ × ℝ)

-- Define the circle
def Circle := (ℝ × ℝ) → Prop

-- Define the inscribed property
def inscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

-- Define the diameter property
def is_diameter (W Z : ℝ × ℝ) (c : Circle) : Prop := sorry

-- Define the angle measure
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the length of a segment
def segment_length (A B : ℝ × ℝ) : ℝ := sorry

theorem quadrilateral_wx_length 
  (q : Quadrilateral) 
  (c : Circle) 
  (h1 : inscribed q c)
  (h2 : is_diameter q.W q.Z c)
  (h3 : segment_length q.W q.Z = 2)
  (h4 : segment_length q.X q.Z = segment_length q.Y q.W)
  (h5 : angle_measure q.W q.X q.Y = 72 * π / 180) :
  segment_length q.W q.X = Real.cos (18 * π / 180) * Real.sqrt (2 * (1 - Real.sin (18 * π / 180))) := by
  sorry

end quadrilateral_wx_length_l2296_229691


namespace roof_ratio_l2296_229695

theorem roof_ratio (length width : ℝ) : 
  length * width = 576 →
  length - width = 36 →
  length / width = 4 :=
by
  sorry

end roof_ratio_l2296_229695


namespace third_hour_speed_l2296_229699

/-- Calculates the average speed for the third hour given total distance, total time, and speeds for the first two hours -/
def average_speed_third_hour (total_distance : ℝ) (total_time : ℝ) (speed_first_hour : ℝ) (speed_second_hour : ℝ) : ℝ :=
  let distance_first_two_hours := speed_first_hour + speed_second_hour
  let distance_third_hour := total_distance - distance_first_two_hours
  distance_third_hour

/-- Proves that the average speed for the third hour is 30 mph given the problem conditions -/
theorem third_hour_speed : 
  let total_distance : ℝ := 120
  let total_time : ℝ := 3
  let speed_first_hour : ℝ := 40
  let speed_second_hour : ℝ := 50
  average_speed_third_hour total_distance total_time speed_first_hour speed_second_hour = 30 := by
  sorry


end third_hour_speed_l2296_229699


namespace dogs_and_video_games_percentage_l2296_229675

theorem dogs_and_video_games_percentage 
  (total_students : ℕ) 
  (dogs_preference : ℕ) 
  (dogs_and_movies_percent : ℚ) : 
  total_students = 30 →
  dogs_preference = 18 →
  dogs_and_movies_percent = 10 / 100 →
  (dogs_preference - (dogs_and_movies_percent * total_students).num) / total_students = 1 / 2 := by
sorry

end dogs_and_video_games_percentage_l2296_229675


namespace count_triangles_eq_29_l2296_229616

/-- The number of non-similar triangles with angles (in degrees) that are distinct
    positive integers in an arithmetic progression with an even common difference -/
def count_triangles : ℕ :=
  let angle_sum := 180
  let middle_angle := angle_sum / 3
  let max_difference := middle_angle - 1
  (max_difference / 2)

theorem count_triangles_eq_29 : count_triangles = 29 := by
  sorry

end count_triangles_eq_29_l2296_229616


namespace symmetric_point_coordinates_l2296_229651

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin of the coordinate system -/
def origin : Point := ⟨0, 0⟩

/-- Given point P -/
def P : Point := ⟨-1, -2⟩

/-- Symmetry about the origin -/
def symmetricAboutOrigin (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

/-- Theorem: The point symmetrical to P(-1, -2) about the origin has coordinates (1, 2) -/
theorem symmetric_point_coordinates :
  symmetricAboutOrigin P = Point.mk 1 2 := by
  sorry

end symmetric_point_coordinates_l2296_229651


namespace students_with_A_or_B_l2296_229610

theorem students_with_A_or_B (fraction_A fraction_B : ℝ) 
  (h1 : fraction_A = 0.7)
  (h2 : fraction_B = 0.2) : 
  fraction_A + fraction_B = 0.9 := by
  sorry

end students_with_A_or_B_l2296_229610


namespace f_a_equals_two_l2296_229696

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_a_equals_two (a : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ a → f x = f (-x)) →
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ a → f x = x^2 + 1) →
  f a = 2 := by sorry

end f_a_equals_two_l2296_229696


namespace fraction_simplification_l2296_229607

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) : (x^2 - y^2) / (x - y) = x + y := by
  sorry

end fraction_simplification_l2296_229607


namespace work_by_concurrent_forces_l2296_229612

/-- Work done by concurrent forces -/
theorem work_by_concurrent_forces :
  let F₁ : ℝ × ℝ := (Real.log 2, Real.log 2)
  let F₂ : ℝ × ℝ := (Real.log 5, Real.log 2)
  let s : ℝ × ℝ := (2 * Real.log 5, 1)
  let F : ℝ × ℝ := (F₁.1 + F₂.1, F₁.2 + F₂.2)
  let W : ℝ := F.1 * s.1 + F.2 * s.2
  W = 2 :=
by sorry

end work_by_concurrent_forces_l2296_229612


namespace f_13_equals_214_l2296_229604

/-- The function f defined as f(n) = n^2 + 2n + 19 -/
def f (n : ℕ) : ℕ := n^2 + 2*n + 19

/-- Theorem stating that f(13) equals 214 -/
theorem f_13_equals_214 : f 13 = 214 := by
  sorry

end f_13_equals_214_l2296_229604


namespace team_allocation_proof_l2296_229602

/-- Proves that given the initial team sizes and total transfer, 
    the number of people allocated to Team A that makes its size 
    twice Team B's size is 23 -/
theorem team_allocation_proof 
  (initial_a initial_b transfer : ℕ) 
  (h_initial_a : initial_a = 31)
  (h_initial_b : initial_b = 26)
  (h_transfer : transfer = 24) :
  ∃ (x : ℕ), 
    x ≤ transfer ∧ 
    initial_a + x = 2 * (initial_b + (transfer - x)) ∧ 
    x = 23 := by
  sorry

end team_allocation_proof_l2296_229602


namespace abc_inequality_l2296_229606

/-- Given a + 2b + 3c = 4, prove two statements about a, b, and c -/
theorem abc_inequality (a b c : ℝ) (h : a + 2*b + 3*c = 4) :
  (∀ (ha : a > 0) (hb : b > 0) (hc : c > 0), 1/a + 2/b + 3/c ≥ 9) ∧
  (∃ (m : ℝ), m = 4/3 ∧ ∀ (x y z : ℝ), x + 2*y + 3*z = 4 → |1/2*x + y| + |z| ≥ m) :=
by sorry

end abc_inequality_l2296_229606


namespace b_41_mod_49_l2296_229680

/-- The sequence b_n defined as 6^n + 8^n -/
def b (n : ℕ) : ℕ := 6^n + 8^n

/-- The theorem stating that b_41 is congruent to 35 modulo 49 -/
theorem b_41_mod_49 : b 41 ≡ 35 [ZMOD 49] := by sorry

end b_41_mod_49_l2296_229680


namespace sufficient_condition_for_inequality_not_necessary_condition_l2296_229693

theorem sufficient_condition_for_inequality (x : ℝ) :
  (-1 < x ∧ x < 5) → (6 / (x + 1) ≥ 1) :=
by
  sorry

theorem not_necessary_condition (x : ℝ) :
  (6 / (x + 1) ≥ 1) → ¬(-1 < x ∧ x < 5) :=
by
  sorry

end sufficient_condition_for_inequality_not_necessary_condition_l2296_229693


namespace m_range_l2296_229609

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, (|x - m| < 1 ↔ 1/3 < x ∧ x < 1/2)) ↔ 
  (-1/2 ≤ m ∧ m ≤ 4/3) :=
sorry

end m_range_l2296_229609


namespace all_log_monotonic_exists_divisible_by_2_and_5_exists_log2_positive_all_statements_true_l2296_229611

-- Define logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- 1. All logarithmic functions are monotonic
theorem all_log_monotonic (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  StrictMono (log a) := by sorry

-- 2. There exists an integer divisible by both 2 and 5
theorem exists_divisible_by_2_and_5 :
  ∃ n : ℤ, 2 ∣ n ∧ 5 ∣ n := by sorry

-- 3. There exists a real number x such that log₂x > 0
theorem exists_log2_positive :
  ∃ x : ℝ, log 2 x > 0 := by sorry

-- All statements are true
theorem all_statements_true :
  (∀ a : ℝ, a > 0 → a ≠ 1 → StrictMono (log a)) ∧
  (∃ n : ℤ, 2 ∣ n ∧ 5 ∣ n) ∧
  (∃ x : ℝ, log 2 x > 0) := by sorry

end all_log_monotonic_exists_divisible_by_2_and_5_exists_log2_positive_all_statements_true_l2296_229611


namespace one_square_remains_l2296_229645

/-- Represents a grid with its dimensions and number of squares -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (squares : ℕ)

/-- Represents the state of points on the grid -/
structure GridState :=
  (grid : Grid)
  (removed_points : ℕ)
  (remaining_squares : ℕ)

/-- Function to calculate the number of additional points to remove -/
def pointsToRemove (initial : GridState) (target : ℕ) : ℕ :=
  sorry

theorem one_square_remains (g : Grid) (initial : GridState) : 
  g.rows = 4 ∧ g.cols = 4 ∧ g.squares = 30 ∧ 
  initial.grid = g ∧ initial.removed_points = 4 →
  pointsToRemove initial 1 = 4 :=
sorry

end one_square_remains_l2296_229645


namespace problem_solution_l2296_229690

theorem problem_solution (x y z : ℝ) 
  (eq1 : 12 * x - 9 * y^2 = 7)
  (eq2 : 6 * y - 9 * z^2 = -2)
  (eq3 : 12 * z - 9 * x^2 = 4) :
  6 * x^2 + 9 * y^2 + 12 * z^2 = 9 := by
sorry

end problem_solution_l2296_229690


namespace line_ellipse_intersection_l2296_229687

/-- The line y = kx + 1 (k ∈ ℝ) always has a common point with the curve x²/5 + y²/m = 1
    if and only if m ≥ 1 and m ≠ 5, where m is a non-negative real number. -/
theorem line_ellipse_intersection (m : ℝ) (h_m_nonneg : m ≥ 0) :
  (∀ k : ℝ, ∃ x y : ℝ, y = k * x + 1 ∧ x^2 / 5 + y^2 / m = 1) ↔ m ≥ 1 ∧ m ≠ 5 :=
sorry

end line_ellipse_intersection_l2296_229687


namespace union_of_sets_l2296_229644

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 4, 5}
  A ∪ B = {1, 2, 3, 4, 5} := by sorry

end union_of_sets_l2296_229644


namespace elective_schemes_count_l2296_229685

/-- The number of elective courses available. -/
def total_courses : ℕ := 10

/-- The number of mutually exclusive courses. -/
def exclusive_courses : ℕ := 3

/-- The number of courses each student must elect. -/
def courses_to_choose : ℕ := 3

/-- Calculates the number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- 
Theorem: The number of ways to choose 3 courses out of 10, 
where 3 specific courses are mutually exclusive, is 98.
-/
theorem elective_schemes_count : 
  choose (total_courses - exclusive_courses) courses_to_choose + 
  exclusive_courses * choose (total_courses - exclusive_courses) (courses_to_choose - 1) = 98 := by
  sorry


end elective_schemes_count_l2296_229685


namespace goose_eggs_count_l2296_229608

/-- The number of goose eggs laid at the pond -/
def total_eggs : ℕ := 1125

/-- The fraction of eggs that hatched -/
def hatched_fraction : ℚ := 1/3

/-- The fraction of hatched geese that survived the first month -/
def survived_month_fraction : ℚ := 4/5

/-- The fraction of geese that survived the first month but did not survive the first year -/
def not_survived_year_fraction : ℚ := 3/5

/-- The number of geese that survived the first year -/
def survived_year : ℕ := 120

theorem goose_eggs_count :
  (↑survived_year : ℚ) = (↑total_eggs * hatched_fraction * survived_month_fraction * (1 - not_survived_year_fraction)) ∧
  ∀ n : ℕ, n ≠ total_eggs → 
    (↑survived_year : ℚ) ≠ (↑n * hatched_fraction * survived_month_fraction * (1 - not_survived_year_fraction)) :=
by sorry

end goose_eggs_count_l2296_229608


namespace same_color_probability_l2296_229676

/-- The number of color options for neckties -/
def necktie_colors : ℕ := 6

/-- The number of color options for shirts -/
def shirt_colors : ℕ := 5

/-- The number of color options for hats -/
def hat_colors : ℕ := 4

/-- The number of color options for socks -/
def sock_colors : ℕ := 3

/-- The number of colors available for all item types -/
def common_colors : ℕ := 3

/-- The probability of selecting items of the same color for a box -/
theorem same_color_probability : 
  (common_colors : ℚ) / (necktie_colors * shirt_colors * hat_colors * sock_colors) = 1 / 120 := by
  sorry

end same_color_probability_l2296_229676


namespace events_mutually_exclusive_not_complementary_l2296_229649

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 3

/-- Represents the number of girls in the group -/
def num_girls : ℕ := 2

/-- Represents the number of students selected -/
def num_selected : ℕ := 2

/-- Represents the event of selecting exactly one boy -/
def event_one_boy : Set (Fin num_boys × Fin num_girls) := sorry

/-- Represents the event of selecting exactly two boys -/
def event_two_boys : Set (Fin num_boys × Fin num_girls) := sorry

/-- The main theorem stating that the two events are mutually exclusive but not complementary -/
theorem events_mutually_exclusive_not_complementary :
  (event_one_boy ∩ event_two_boys = ∅) ∧ 
  (event_one_boy ∪ event_two_boys ≠ Set.univ) :=
sorry

end events_mutually_exclusive_not_complementary_l2296_229649


namespace max_five_cent_coins_l2296_229622

theorem max_five_cent_coins (x y z : ℕ) : 
  x + y + z = 25 →
  x + 2*y + 5*z = 60 →
  z ≤ 8 :=
by sorry

end max_five_cent_coins_l2296_229622


namespace bakery_sugar_amount_l2296_229677

/-- Given the ratios of ingredients in a bakery storage room, prove the amount of sugar. -/
theorem bakery_sugar_amount (sugar flour baking_soda : ℝ) 
  (h1 : sugar / flour = 3 / 8)
  (h2 : flour / baking_soda = 10 / 1)
  (h3 : flour / (baking_soda + 60) = 8 / 1) :
  sugar = 900 := by
  sorry

end bakery_sugar_amount_l2296_229677


namespace multiplication_equations_l2296_229654

theorem multiplication_equations : 
  (30 * 30 = 900) ∧
  (30 * 40 = 1200) ∧
  (40 * 70 = 2800) ∧
  (50 * 70 = 3500) ∧
  (60 * 70 = 4200) ∧
  (4 * 90 = 360) := by
  sorry

end multiplication_equations_l2296_229654


namespace simplify_and_express_negative_exponents_l2296_229626

theorem simplify_and_express_negative_exponents 
  (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = 2 * x⁻¹ * y⁻¹ * z⁻¹ := by
  sorry

end simplify_and_express_negative_exponents_l2296_229626


namespace percentage_increase_l2296_229647

theorem percentage_increase (t : ℝ) (P : ℝ) : 
  t = 80 →
  (t + (P / 100) * t) - (t - (25 / 100) * t) = 30 →
  P = 12.5 := by
sorry

end percentage_increase_l2296_229647


namespace symmetric_points_sum_l2296_229641

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposites
    and their y-coordinates are equal -/
def symmetric_wrt_y_axis (a b : ℝ × ℝ) : Prop :=
  a.1 = -b.1 ∧ a.2 = b.2

theorem symmetric_points_sum (x y : ℝ) :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (x - 4, 6 + y)
  symmetric_wrt_y_axis a b → x + y = -3 := by
sorry

end symmetric_points_sum_l2296_229641


namespace bob_gave_terry_24_bushels_l2296_229601

/-- Represents the number of bushels Bob grew -/
def total_bushels : ℕ := 50

/-- Represents the number of ears per bushel -/
def ears_per_bushel : ℕ := 14

/-- Represents the number of ears Bob has left -/
def ears_left : ℕ := 357

/-- Calculates the number of bushels Bob gave to Terry -/
def bushels_given_to_terry : ℕ :=
  ((total_bushels * ears_per_bushel) - ears_left) / ears_per_bushel

theorem bob_gave_terry_24_bushels :
  bushels_given_to_terry = 24 := by
  sorry

end bob_gave_terry_24_bushels_l2296_229601


namespace vacuum_tube_alignment_l2296_229635

theorem vacuum_tube_alignment :
  ∃ (f g : Fin 7 → Fin 7), 
    ∀ (r : Fin 7), ∃ (k : Fin 7), f k = g ((r + k) % 7) := by
  sorry

end vacuum_tube_alignment_l2296_229635


namespace complex_equation_solution_l2296_229617

theorem complex_equation_solution (a : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : (a + i) * (1 + i) = 2 * i) : a = 1 := by
  sorry

end complex_equation_solution_l2296_229617


namespace water_level_correct_water_level_rate_initial_water_level_l2296_229642

/-- Represents the water level function in a reservoir -/
def water_level (x : ℝ) : ℝ := 6 + 0.3 * x

theorem water_level_correct (x : ℝ) (h : x ≥ 0) :
  water_level x = 6 + 0.3 * x :=
by sorry

/-- The water level rises at a constant rate of 0.3 meters per hour -/
theorem water_level_rate (x y : ℝ) (hx : x ≥ 0) (hy : y > x) :
  (water_level y - water_level x) / (y - x) = 0.3 :=
by sorry

/-- The initial water level is 6 meters -/
theorem initial_water_level : water_level 0 = 6 :=
by sorry

end water_level_correct_water_level_rate_initial_water_level_l2296_229642


namespace diophantine_equation_solutions_l2296_229603

theorem diophantine_equation_solutions (n : ℕ) : n ∈ ({1, 2, 3} : Set ℕ) ↔ 
  ∃ (a b c : ℤ), a^n + b^n = c^n + n ∧ n ≤ 6 := by
  sorry

end diophantine_equation_solutions_l2296_229603


namespace steve_reading_time_l2296_229666

/-- Calculates the number of weeks needed to read a book given the total pages and pages read per week. -/
def weeks_to_read (total_pages : ℕ) (pages_per_day : ℕ) (reading_days_per_week : ℕ) : ℕ :=
  total_pages / (pages_per_day * reading_days_per_week)

/-- Proves that it takes 7 weeks to read a 2100-page book when reading 100 pages on 3 days per week. -/
theorem steve_reading_time : weeks_to_read 2100 100 3 = 7 := by
  sorry

end steve_reading_time_l2296_229666


namespace intersection_M_N_l2296_229678

def M : Set ℝ := {0, 1, 3}
def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N : M ∩ N = {1} := by sorry

end intersection_M_N_l2296_229678


namespace inequality_solution_set_l2296_229648

theorem inequality_solution_set (x : ℝ) : 
  5 * x^2 + 7 * x > 3 ↔ x < -1 ∨ x > 3/5 := by sorry

end inequality_solution_set_l2296_229648
