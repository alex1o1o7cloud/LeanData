import Mathlib

namespace subset_relation_and_complement_l2202_220259

open Set

theorem subset_relation_and_complement (S A B : Set α) :
  (∀ x, x ∈ (S \ A) → x ∈ B) →
  (A ⊇ (S \ B) ∧ A ≠ (S \ B)) :=
by sorry

end subset_relation_and_complement_l2202_220259


namespace base5_division_theorem_l2202_220244

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 5 + d) 0

/-- Converts a base 10 number to base 5 --/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

theorem base5_division_theorem :
  let dividend := [2, 1, 3, 4, 2]  -- 21342₅
  let divisor := [2, 3]            -- 23₅
  let quotient := [4, 0, 4, 3]     -- 4043₅
  (base5ToBase10 dividend) / (base5ToBase10 divisor) = base5ToBase10 quotient := by
  sorry

end base5_division_theorem_l2202_220244


namespace solve_equation_l2202_220288

theorem solve_equation (x : ℝ) (y : ℝ) (h1 : y = (x^2 - 9) / (x - 3)) 
  (h2 : y = 3*x + 1) (h3 : x ≠ 3) : x = 1 := by
  sorry

end solve_equation_l2202_220288


namespace hexagon_division_existence_l2202_220277

/-- A hexagon is a polygon with six sides -/
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- A line is represented by two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- A triangle is represented by three points -/
structure Triangle where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- Predicate to check if two triangles are congruent -/
def areCongruentTriangles (t1 t2 : Triangle) : Prop := sorry

/-- Predicate to check if a line divides a hexagon into four congruent triangles -/
def dividesIntoFourCongruentTriangles (h : Hexagon) (l : Line) : Prop :=
  ∃ t1 t2 t3 t4 : Triangle,
    areCongruentTriangles t1 t2 ∧
    areCongruentTriangles t1 t3 ∧
    areCongruentTriangles t1 t4

/-- Theorem stating that there exists a hexagon that can be divided by a single line into four congruent triangles -/
theorem hexagon_division_existence :
  ∃ (h : Hexagon) (l : Line), dividesIntoFourCongruentTriangles h l := by sorry

end hexagon_division_existence_l2202_220277


namespace gcd_lcm_product_l2202_220242

theorem gcd_lcm_product (a b : ℕ) (ha : a = 90) (hb : b = 135) :
  (Nat.gcd a b) * (Nat.lcm a b) = 12150 := by
  sorry

end gcd_lcm_product_l2202_220242


namespace work_completion_time_l2202_220252

/-- The number of days it takes worker A to complete the work -/
def days_A : ℚ := 10

/-- The efficiency ratio of worker B compared to worker A -/
def efficiency_ratio : ℚ := 1.75

/-- The number of days it takes worker B to complete the work -/
def days_B : ℚ := 40 / 7

theorem work_completion_time :
  days_A * efficiency_ratio = days_B :=
sorry

end work_completion_time_l2202_220252


namespace success_arrangements_l2202_220219

/-- The number of permutations of a multiset -/
def multiset_permutations (n : ℕ) (repeats : List ℕ) : ℕ :=
  Nat.factorial n / (repeats.map Nat.factorial).prod

/-- The number of ways to arrange the letters of SUCCESS -/
theorem success_arrangements : multiset_permutations 7 [3, 2] = 420 := by
  sorry

end success_arrangements_l2202_220219


namespace total_books_read_l2202_220235

def summer_reading (june july august : ℕ) : Prop :=
  june = 8 ∧ july = 2 * june ∧ august = july - 3

theorem total_books_read (june july august : ℕ) 
  (h : summer_reading june july august) : june + july + august = 37 := by
  sorry

end total_books_read_l2202_220235


namespace car_speed_second_half_l2202_220261

/-- Calculates the speed of a car during the second half of a journey given the total distance,
    speed for the first half, and average speed for the entire journey. -/
theorem car_speed_second_half
  (total_distance : ℝ)
  (first_half_distance : ℝ)
  (first_half_speed : ℝ)
  (average_speed : ℝ)
  (h1 : total_distance = 320)
  (h2 : first_half_distance = 160)
  (h3 : first_half_speed = 90)
  (h4 : average_speed = 84.70588235294117)
  (h5 : first_half_distance * 2 = total_distance) :
  let second_half_speed := (total_distance / average_speed - first_half_distance / first_half_speed)⁻¹ * first_half_distance
  second_half_speed = 80 := by
sorry


end car_speed_second_half_l2202_220261


namespace triangle_sine_inequality_l2202_220233

theorem triangle_sine_inequality (a b c : Real) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a + b > c ∧ b + c > a ∧ c + a > b)
  (h3 : a + b + c ≤ 2 * Real.pi) : 
  Real.sin a + Real.sin b > Real.sin c ∧
  Real.sin b + Real.sin c > Real.sin a ∧
  Real.sin c + Real.sin a > Real.sin b :=
by sorry

end triangle_sine_inequality_l2202_220233


namespace endpoint_sum_coordinates_endpoint_sum_coordinates_proof_l2202_220206

/-- Given a line segment with one endpoint (6, 2) and midpoint (3, 7),
    the sum of coordinates of the other endpoint is 12. -/
theorem endpoint_sum_coordinates : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop :=
  fun endpoint1 midpoint endpoint2 =>
    endpoint1 = (6, 2) ∧
    midpoint = (3, 7) ∧
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) →
    endpoint2.1 + endpoint2.2 = 12
    
#check endpoint_sum_coordinates

theorem endpoint_sum_coordinates_proof : 
  ∃ (endpoint2 : ℝ × ℝ), endpoint_sum_coordinates (6, 2) (3, 7) endpoint2 := by
  sorry

end endpoint_sum_coordinates_endpoint_sum_coordinates_proof_l2202_220206


namespace rancher_lasso_probability_l2202_220290

/-- The probability of a rancher placing a lasso around a cow's neck in a single throw. -/
def single_throw_probability : ℚ := 1 / 2

/-- The number of attempts the rancher makes. -/
def number_of_attempts : ℕ := 3

/-- The probability of the rancher placing a lasso around a cow's neck at least once in the given number of attempts. -/
def success_probability : ℚ := 7 / 8

theorem rancher_lasso_probability :
  (1 : ℚ) - (1 - single_throw_probability) ^ number_of_attempts = success_probability :=
sorry

end rancher_lasso_probability_l2202_220290


namespace existence_of_function_l2202_220213

theorem existence_of_function (a : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x y : ℝ, x + f y = a * (y + f x)) ↔ (a = 1 ∨ a = -1) := by
  sorry

end existence_of_function_l2202_220213


namespace prob_three_odd_dice_l2202_220284

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The number of odd numbers on each die -/
def numOddSides : ℕ := 4

/-- The number of dice that should show an odd number -/
def targetOddDice : ℕ := 3

/-- The probability of rolling exactly three odd numbers when five 8-sided dice are rolled -/
theorem prob_three_odd_dice : 
  (numOddSides / numSides) ^ targetOddDice * 
  ((numSides - numOddSides) / numSides) ^ (numDice - targetOddDice) * 
  (Nat.choose numDice targetOddDice) = 5 / 16 := by
  sorry

end prob_three_odd_dice_l2202_220284


namespace remainder_s_1024_mod_1000_l2202_220225

-- Define the polynomial q(x)
def q (x : ℤ) : ℤ := (x^1025 - 1) / (x - 1)

-- Define the divisor polynomial
def divisor (x : ℤ) : ℤ := x^6 + x^5 + 3*x^4 + x^3 + x^2 + x + 1

-- Define s(x) as the polynomial remainder
noncomputable def s (x : ℤ) : ℤ := q x % divisor x

-- Theorem statement
theorem remainder_s_1024_mod_1000 : |s 1024| % 1000 = 824 := by sorry

end remainder_s_1024_mod_1000_l2202_220225


namespace power_function_symmetry_l2202_220253

/-- A function f is a power function if it can be written as f(x) = ax^n for some constant a and real number n. -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), ∀ x, f x = a * x^n

/-- A function f is symmetric about the y-axis if f(x) = f(-x) for all x. -/
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- Given that f(x) = (t^2 - t + 1)x^((t+3)/5) is a power function and 
    symmetric about the y-axis, prove that t = 1. -/
theorem power_function_symmetry (t : ℝ) : 
  let f := fun (x : ℝ) ↦ (t^2 - t + 1) * x^((t+3)/5)
  is_power_function f ∧ symmetric_about_y_axis f → t = 1 := by
  sorry

end power_function_symmetry_l2202_220253


namespace wise_stock_price_l2202_220211

/-- Given the conditions of Mr. Wise's stock purchase, prove the price of the stock he bought 400 shares of. -/
theorem wise_stock_price (total_value : ℝ) (price_known : ℝ) (total_shares : ℕ) (shares_unknown : ℕ) :
  total_value = 1950 →
  price_known = 4.5 →
  total_shares = 450 →
  shares_unknown = 400 →
  ∃ (price_unknown : ℝ),
    price_unknown * shares_unknown + price_known * (total_shares - shares_unknown) = total_value ∧
    price_unknown = 4.3125 :=
by sorry

end wise_stock_price_l2202_220211


namespace rationalize_denominator_sum_l2202_220226

theorem rationalize_denominator_sum :
  ∃ (A B C D E F : ℤ),
    (F > 0) ∧
    (∀ (x : ℝ), x > 0 → (1 / (Real.sqrt 5 + Real.sqrt 2 + Real.sqrt 11) = 
      (A * Real.sqrt 2 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F)) ∧
    (A + B + C + D + E + F = 136) :=
by sorry

end rationalize_denominator_sum_l2202_220226


namespace rectangle_area_ratio_l2202_220202

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively, 
    where a/c = b/d = 3/5, the ratio of the area of Rectangle A to the area 
    of Rectangle B is 9:25. -/
theorem rectangle_area_ratio 
  (a b c d : ℝ) 
  (h1 : a / c = 3 / 5) 
  (h2 : b / d = 3 / 5) :
  (a * b) / (c * d) = 9 / 25 := by
  sorry


end rectangle_area_ratio_l2202_220202


namespace unique_right_triangle_perimeter_area_ratio_l2202_220249

theorem unique_right_triangle_perimeter_area_ratio :
  ∃! (a b : ℝ), a > 0 ∧ b > 0 ∧
  (a + b + Real.sqrt (a^2 + b^2)) / ((1/2) * a * b) = 5 := by
  sorry

end unique_right_triangle_perimeter_area_ratio_l2202_220249


namespace three_consecutive_heads_probability_l2202_220266

theorem three_consecutive_heads_probability (p : ℝ) :
  p = (1 : ℝ) / 2 →  -- probability of heads on a single flip
  p * p * p = (1 : ℝ) / 8 :=  -- probability of three consecutive heads
by
  sorry

end three_consecutive_heads_probability_l2202_220266


namespace jones_wardrobe_l2202_220268

/-- The ratio of shirts to pants in Mr. Jones' wardrobe -/
def shirt_to_pants_ratio : ℕ := 6

/-- The number of pants Mr. Jones owns -/
def number_of_pants : ℕ := 40

/-- The total number of pieces of clothes Mr. Jones owns -/
def total_clothes : ℕ := shirt_to_pants_ratio * number_of_pants + number_of_pants

theorem jones_wardrobe : total_clothes = 280 := by
  sorry

end jones_wardrobe_l2202_220268


namespace joint_equation_solver_l2202_220270

/-- Given two equations and two solutions, prove the value of a specific expression --/
theorem joint_equation_solver (a b : ℤ) :
  (a * (-3) + 5 * (-1) = 15) →
  (4 * (-3) - b * (-1) = -2) →
  (a * 5 + 5 * 4 = 15) →
  (4 * 5 - b * 4 = -2) →
  a^2018 + (-1/10 * b : ℚ)^2019 = 0 := by
  sorry

end joint_equation_solver_l2202_220270


namespace inequality_condition_on_a_l2202_220294

theorem inequality_condition_on_a :
  ∀ a : ℝ, (∀ x : ℝ, (a - 3) * x^2 + 2 * (a - 3) * x - 4 < 0) ↔ a ∈ Set.Ioc (-1) 3 :=
by sorry

end inequality_condition_on_a_l2202_220294


namespace compound_ratio_example_l2202_220222

def ratio (a b : ℤ) := (a, b)

def compound_ratio (r1 r2 r3 : ℤ × ℤ) : ℤ × ℤ :=
  let (a1, b1) := r1
  let (a2, b2) := r2
  let (a3, b3) := r3
  (a1 * a2 * a3, b1 * b2 * b3)

def simplify_ratio (r : ℤ × ℤ) : ℤ × ℤ :=
  let (a, b) := r
  let gcd := Int.gcd a b
  (a / gcd, b / gcd)

theorem compound_ratio_example : 
  simplify_ratio (compound_ratio (ratio 2 3) (ratio 6 11) (ratio 11 2)) = (2, 1) := by
  sorry

end compound_ratio_example_l2202_220222


namespace equation_solution_l2202_220265

theorem equation_solution : ∃! y : ℚ, 7 * (4 * y + 5) - 4 = -3 * (2 - 9 * y) := by
  use (-37 : ℚ)
  constructor
  · -- Prove that y = -37 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end equation_solution_l2202_220265


namespace traffic_to_driving_ratio_l2202_220281

theorem traffic_to_driving_ratio (total_time driving_time : ℝ) 
  (h1 : total_time = 15)
  (h2 : driving_time = 5) :
  (total_time - driving_time) / driving_time = 2 := by
  sorry

end traffic_to_driving_ratio_l2202_220281


namespace fractional_parts_sum_not_one_l2202_220248

theorem fractional_parts_sum_not_one (x : ℚ) : 
  ¬(x - ⌊x⌋ + x^2 - ⌊x^2⌋ = 1) := by sorry

end fractional_parts_sum_not_one_l2202_220248


namespace prime_or_composite_a4_3a2_9_l2202_220271

theorem prime_or_composite_a4_3a2_9 (a : ℕ) :
  (a = 1 ∨ a = 2 → Nat.Prime (a^4 - 3*a^2 + 9)) ∧
  (a > 2 → ¬Nat.Prime (a^4 - 3*a^2 + 9)) :=
by sorry

end prime_or_composite_a4_3a2_9_l2202_220271


namespace gcd_from_lcm_and_ratio_l2202_220210

theorem gcd_from_lcm_and_ratio (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : X * 5 = Y * 2) : 
  Nat.gcd X Y = 18 := by
  sorry

end gcd_from_lcm_and_ratio_l2202_220210


namespace stating_spheres_fit_funnel_iff_l2202_220200

/-- Represents a conical funnel with two spheres inside it -/
structure ConicalFunnelWithSpheres where
  α : ℝ  -- Half of the axial section angle
  R : ℝ  -- Radius of the larger sphere
  r : ℝ  -- Radius of the smaller sphere
  h_angle_positive : 0 < α
  h_angle_less_than_pi_half : α < π / 2
  h_R_positive : 0 < R
  h_r_positive : 0 < r
  h_R_greater_r : r < R

/-- 
The necessary and sufficient condition for two spheres to be placed in a conical funnel 
such that they both touch its lateral surface
-/
def spheres_fit_condition (funnel : ConicalFunnelWithSpheres) : Prop :=
  Real.sin funnel.α ≤ (funnel.R - funnel.r) / funnel.R

/-- 
Theorem stating the necessary and sufficient condition for two spheres 
to fit in a conical funnel touching its lateral surface
-/
theorem spheres_fit_funnel_iff (funnel : ConicalFunnelWithSpheres) :
  (∃ (pos_R pos_r : ℝ), 
    pos_R > 0 ∧ pos_r > 0 ∧ pos_R = funnel.R ∧ pos_r = funnel.r ∧
    (∃ (config : ℝ × ℝ), 
      (config.1 > 0 ∧ config.2 > 0) ∧
      (config.1 + pos_R) * Real.sin funnel.α = pos_R ∧
      (config.2 + pos_r) * Real.sin funnel.α = pos_r ∧
      config.1 + pos_R + pos_r = config.2)) ↔
  spheres_fit_condition funnel :=
sorry

end stating_spheres_fit_funnel_iff_l2202_220200


namespace bridget_apples_l2202_220276

theorem bridget_apples : ∃ (x : ℕ), 
  x > 0 ∧ 
  (2 * x) % 3 = 0 ∧ 
  (2 * x) / 3 - 5 = 2 ∧ 
  x = 11 := by
  sorry

end bridget_apples_l2202_220276


namespace cost_of_apple_l2202_220280

/-- The cost of fruit problem -/
theorem cost_of_apple (banana_cost orange_cost : ℚ)
  (apple_count banana_count orange_count : ℕ)
  (average_cost : ℚ)
  (h1 : banana_cost = 1)
  (h2 : orange_cost = 3)
  (h3 : apple_count = 12)
  (h4 : banana_count = 4)
  (h5 : orange_count = 4)
  (h6 : average_cost = 2)
  (h7 : average_cost * (apple_count + banana_count + orange_count : ℚ) =
        apple_cost * apple_count + banana_cost * banana_count + orange_cost * orange_count) :
  apple_cost = 2 :=
sorry

end cost_of_apple_l2202_220280


namespace lemon_price_increase_l2202_220291

/-- Proves that the increase in lemon price is $4 given the conditions of Erick's fruit sale --/
theorem lemon_price_increase :
  ∀ (x : ℝ),
    (80 * (8 + x) + 140 * (7 + x / 2) = 2220) →
    x = 4 := by
  sorry

end lemon_price_increase_l2202_220291


namespace train_speed_calculation_l2202_220237

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  bridge_length = 140 →
  crossing_time = 23.998080153587715 →
  ∃ (speed : ℝ), abs (speed - 36) < 0.1 ∧ speed = (train_length + bridge_length) / crossing_time * 3.6 :=
by sorry

end train_speed_calculation_l2202_220237


namespace system_solution_unique_l2202_220292

theorem system_solution_unique :
  ∃! (x y z : ℝ), 5 * x + 3 * y = 65 ∧ 2 * y - z = 11 ∧ 3 * x + 4 * z = 57 :=
by
  -- The proof goes here
  sorry

end system_solution_unique_l2202_220292


namespace pedro_squares_difference_l2202_220289

theorem pedro_squares_difference (jesus_squares linden_squares pedro_squares : ℕ) 
  (h1 : jesus_squares = 60)
  (h2 : linden_squares = 75)
  (h3 : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 := by
  sorry

end pedro_squares_difference_l2202_220289


namespace calculation_proof_l2202_220296

theorem calculation_proof : (0.0048 * 3.5) / (0.05 * 0.1 * 0.004) = 840 := by
  sorry

end calculation_proof_l2202_220296


namespace gcd_of_lcm_and_ratio_l2202_220204

theorem gcd_of_lcm_and_ratio (A B : ℕ+) : 
  Nat.lcm A B = 180 → 
  A.val * 6 = B.val * 5 → 
  Nat.gcd A B = 6 := by
sorry

end gcd_of_lcm_and_ratio_l2202_220204


namespace complex_equation_solution_l2202_220278

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 3 - Complex.I → z = -1 - 3 * Complex.I := by
  sorry

end complex_equation_solution_l2202_220278


namespace two_times_greater_l2202_220260

theorem two_times_greater (a b : ℚ) (h : a > b) : 2 * a > 2 * b := by
  sorry

end two_times_greater_l2202_220260


namespace mrs_hilt_fountain_trips_l2202_220238

/-- Calculates the total distance walked to a water fountain given the one-way distance and number of trips -/
def total_distance_walked (one_way_distance : ℕ) (num_trips : ℕ) : ℕ :=
  2 * one_way_distance * num_trips

/-- Proves that given a distance of 30 feet from desk to fountain and 4 trips to the fountain, the total distance walked is 240 feet -/
theorem mrs_hilt_fountain_trips :
  total_distance_walked 30 4 = 240 := by
  sorry

end mrs_hilt_fountain_trips_l2202_220238


namespace sales_tax_difference_l2202_220267

theorem sales_tax_difference (price : ℝ) (high_rate low_rate : ℝ) :
  price = 50 →
  high_rate = 0.0725 →
  low_rate = 0.0675 →
  price * high_rate - price * low_rate = 0.25 :=
by
  sorry

end sales_tax_difference_l2202_220267


namespace AB_vector_l2202_220282

def OA : ℝ × ℝ := (1, -2)
def OB : ℝ × ℝ := (-3, 1)

theorem AB_vector : (OB.1 - OA.1, OB.2 - OA.2) = (-4, 3) := by
  sorry

end AB_vector_l2202_220282


namespace car_journey_time_l2202_220224

/-- Proves that given a car traveling 210 km in 7 hours for the forward journey,
    and increasing its speed by 12 km/hr for the return journey,
    the time taken for the return journey is 5 hours. -/
theorem car_journey_time (distance : ℝ) (forward_time : ℝ) (speed_increase : ℝ) :
  distance = 210 →
  forward_time = 7 →
  speed_increase = 12 →
  (distance / (distance / forward_time + speed_increase)) = 5 := by
  sorry

end car_journey_time_l2202_220224


namespace trig_identity_l2202_220214

theorem trig_identity (θ : Real) (h : Real.tan (θ - Real.pi) = 2) :
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 := by
  sorry

end trig_identity_l2202_220214


namespace g_of_neg_one_eq_neg_seven_l2202_220283

/-- Given a function g(x) = 5x - 2, prove that g(-1) = -7 -/
theorem g_of_neg_one_eq_neg_seven :
  let g : ℝ → ℝ := fun x ↦ 5 * x - 2
  g (-1) = -7 := by sorry

end g_of_neg_one_eq_neg_seven_l2202_220283


namespace solve_cookies_problem_l2202_220247

def cookies_problem (total_baked : ℕ) (kristy_ate : ℕ) (friend1_took : ℕ) (friend2_took : ℕ) (friend3_took : ℕ) (cookies_left : ℕ) : Prop :=
  let cookies_taken := kristy_ate + friend1_took + friend2_took + friend3_took
  let cookies_given_away := total_baked - cookies_left
  let brother_cookies := cookies_given_away - cookies_taken
  brother_cookies = 1

theorem solve_cookies_problem :
  cookies_problem 22 2 3 5 5 6 := by
  sorry

end solve_cookies_problem_l2202_220247


namespace no_real_solutions_l2202_220220

theorem no_real_solutions :
  ∀ x y : ℝ, x^2 + 2*y^2 - 6*x - 8*y + 21 ≠ 0 := by
  sorry

end no_real_solutions_l2202_220220


namespace area_of_parallelogram_EFGH_l2202_220262

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Calculates the magnitude of the cross product of two 2D vectors -/
def crossProductMagnitude (v1 v2 : Vector2D) : ℝ :=
  |v1.x * v2.y - v1.y * v2.x|

/-- Theorem: Area of parallelogram EFGH -/
theorem area_of_parallelogram_EFGH : 
  let EF : Vector2D := ⟨3, 1⟩
  let EG : Vector2D := ⟨1, 5⟩
  crossProductMagnitude EF EG = 14 := by
  sorry

#check area_of_parallelogram_EFGH

end area_of_parallelogram_EFGH_l2202_220262


namespace P_initial_investment_l2202_220216

/-- Represents the initial investment of P in rupees -/
def P_investment : ℕ := sorry

/-- Represents Q's investment in rupees -/
def Q_investment : ℕ := 9000

/-- Represents the number of months P's investment was active -/
def P_months : ℕ := 12

/-- Represents the number of months Q's investment was active -/
def Q_months : ℕ := 8

/-- Represents P's share in the profit ratio -/
def P_share : ℕ := 2

/-- Represents Q's share in the profit ratio -/
def Q_share : ℕ := 3

/-- Theorem stating that P's initial investment is 4000 rupees -/
theorem P_initial_investment :
  (P_investment * P_months) * Q_share = (Q_investment * Q_months) * P_share ∧
  P_investment = 4000 := by
  sorry

end P_initial_investment_l2202_220216


namespace f_symmetry_f_max_min_on_interval_l2202_220255

def f (x : ℝ) : ℝ := x^3 - 27*x

theorem f_symmetry (x : ℝ) : f (-x) = -f x := by sorry

theorem f_max_min_on_interval :
  let a : ℝ := -4
  let b : ℝ := 5
  (∃ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, f y ≤ f x) ∧
  (∃ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, f x ≤ f y) ∧
  (∃ x ∈ Set.Icc a b, f x = 54) ∧
  (∃ x ∈ Set.Icc a b, f x = -54) := by sorry

end f_symmetry_f_max_min_on_interval_l2202_220255


namespace circle_center_l2202_220272

/-- The center of a circle with equation x^2 + 4x + y^2 - 6y = 12 is (-2, 3) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + 4*x + y^2 - 6*y = 12) → (x + 2)^2 + (y - 3)^2 = 25 := by
  sorry

end circle_center_l2202_220272


namespace jury_duty_days_is_25_l2202_220258

/-- Calculates the total number of days spent on jury duty -/
def totalJuryDutyDays (
  jurySelectionDays : ℕ)
  (trialMultiplier : ℕ)
  (trialDailyHours : ℕ)
  (deliberationDays : List ℕ)
  (deliberationDailyHours : ℕ) : ℕ :=
  let trialDays := jurySelectionDays * trialMultiplier
  let totalDeliberationHours := deliberationDays.sum * (deliberationDailyHours - 2)
  let totalDeliberationDays := (totalDeliberationHours + deliberationDailyHours - 1) / deliberationDailyHours
  jurySelectionDays + trialDays + totalDeliberationDays

/-- Theorem stating that the total jury duty days is 25 -/
theorem jury_duty_days_is_25 :
  totalJuryDutyDays 2 4 9 [6, 4, 5] 14 = 25 := by
  sorry

#eval totalJuryDutyDays 2 4 9 [6, 4, 5] 14

end jury_duty_days_is_25_l2202_220258


namespace hakimi_age_l2202_220212

/-- Given three friends Hakimi, Jared, and Molly, this theorem proves Hakimi's age
    based on the given conditions. -/
theorem hakimi_age (hakimi_age jared_age molly_age : ℕ) : 
  (hakimi_age + jared_age + molly_age) / 3 = 40 →  -- Average age is 40
  jared_age = hakimi_age + 10 →  -- Jared is 10 years older than Hakimi
  molly_age = 30 →  -- Molly's age is 30
  hakimi_age = 40 :=  -- Hakimi's age is 40
by
  sorry

end hakimi_age_l2202_220212


namespace chocolate_gain_percent_l2202_220236

theorem chocolate_gain_percent (C S : ℝ) (h : 165 * C = 150 * S) : 
  (S - C) / C * 100 = 10 := by sorry

end chocolate_gain_percent_l2202_220236


namespace not_p_sufficient_not_necessary_for_not_q_l2202_220254

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 = 3*x + 4
def q (x : ℝ) : Prop := x = Real.sqrt (3*x + 4)

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x, ¬(q x) → ¬(p x)) :=
sorry

end not_p_sufficient_not_necessary_for_not_q_l2202_220254


namespace trigonometric_identities_l2202_220241

theorem trigonometric_identities (α : Real) (h_acute : 0 < α ∧ α < Real.pi / 2) 
  (h_sin : Real.sin α = 3 / 5) : 
  (Real.cos α = 4 / 5) ∧ 
  (Real.cos (α + Real.pi / 6) = (4 * Real.sqrt 3 - 3) / 10) := by
  sorry

end trigonometric_identities_l2202_220241


namespace train_car_speed_ratio_l2202_220230

/-- Given a bus that travels 320 km in 5 hours, and its speed is 4/5 of the train's speed,
    and a car that travels 525 km in 7 hours, prove that the ratio of the train's speed
    to the car's speed is 16:15 -/
theorem train_car_speed_ratio :
  ∀ (bus_speed train_speed car_speed : ℝ),
    bus_speed = 320 / 5 →
    bus_speed = (4 / 5) * train_speed →
    car_speed = 525 / 7 →
    train_speed / car_speed = 16 / 15 := by
  sorry

end train_car_speed_ratio_l2202_220230


namespace average_after_removal_l2202_220257

theorem average_after_removal (numbers : Finset ℕ) (sum : ℕ) :
  Finset.card numbers = 15 →
  sum / 15 = 100 →
  sum = Finset.sum numbers id →
  80 ∈ numbers →
  90 ∈ numbers →
  95 ∈ numbers →
  (sum - 80 - 90 - 95) / (Finset.card numbers - 3) = 1235 / 12 :=
by sorry

end average_after_removal_l2202_220257


namespace cities_under_50k_l2202_220203

/-- City population distribution -/
structure CityDistribution where
  small : ℝ  -- Percentage of cities with fewer than 5,000 residents
  medium : ℝ  -- Percentage of cities with 5,000 to 49,999 residents
  large : ℝ  -- Percentage of cities with 50,000 or more residents

/-- The given city distribution -/
def givenDistribution : CityDistribution where
  small := 20
  medium := 35
  large := 45

/-- Theorem: The percentage of cities with fewer than 50,000 residents is 55% -/
theorem cities_under_50k (d : CityDistribution) 
  (h1 : d.small = 20) 
  (h2 : d.medium = 35) 
  (h3 : d.large = 45) : 
  d.small + d.medium = 55 := by
  sorry

#check cities_under_50k

end cities_under_50k_l2202_220203


namespace sum_of_squares_l2202_220209

theorem sum_of_squares (a b c d e f : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (h1 : b * c * d * e * f / a = 1 / 2)
  (h2 : a * c * d * e * f / b = 1 / 4)
  (h3 : a * b * d * e * f / c = 1 / 8)
  (h4 : a * b * c * e * f / d = 2)
  (h5 : a * b * c * d * f / e = 4)
  (h6 : a * b * c * d * e / f = 8) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 119 / 8 := by
sorry

end sum_of_squares_l2202_220209


namespace prob_at_least_one_boy_and_girl_l2202_220239

-- Define the probability of having a boy or girl
def prob_boy_or_girl : ℚ := 1 / 2

-- Define the number of children in the family
def num_children : ℕ := 4

-- The theorem to prove
theorem prob_at_least_one_boy_and_girl : 
  (1 : ℚ) - 2 * (prob_boy_or_girl ^ num_children) = 7 / 8 := by
  sorry

end prob_at_least_one_boy_and_girl_l2202_220239


namespace sarah_brother_books_l2202_220246

/-- The number of books Sarah's brother bought -/
def brothers_books (sarah_paperbacks sarah_hardbacks : ℕ) : ℕ :=
  (sarah_paperbacks / 3) + (sarah_hardbacks * 2)

/-- Theorem: Sarah's brother bought 10 books in total -/
theorem sarah_brother_books :
  brothers_books 6 4 = 10 := by
  sorry

end sarah_brother_books_l2202_220246


namespace mouse_testes_most_appropriate_l2202_220286

-- Define the possible experimental materials
inductive ExperimentalMaterial
| AscarisEggs
| ChickenLiver
| MouseTestes
| OnionEpidermis

-- Define the cell division processes
inductive CellDivisionProcess
| Mitosis
| Meiosis
| NoDivision

-- Define the property of continuous cell formation
def hasContinuousCellFormation : ExperimentalMaterial → Prop
| ExperimentalMaterial.MouseTestes => True
| _ => False

-- Define the cell division process for each material
def cellDivisionProcess : ExperimentalMaterial → CellDivisionProcess
| ExperimentalMaterial.AscarisEggs => CellDivisionProcess.Mitosis
| ExperimentalMaterial.ChickenLiver => CellDivisionProcess.Mitosis
| ExperimentalMaterial.MouseTestes => CellDivisionProcess.Meiosis
| ExperimentalMaterial.OnionEpidermis => CellDivisionProcess.NoDivision

-- Define the property of being appropriate for observing meiosis
def isAppropriateForMeiosis (material : ExperimentalMaterial) : Prop :=
  cellDivisionProcess material = CellDivisionProcess.Meiosis ∧ hasContinuousCellFormation material

-- Theorem statement
theorem mouse_testes_most_appropriate :
  ∀ material : ExperimentalMaterial,
    isAppropriateForMeiosis material → material = ExperimentalMaterial.MouseTestes :=
by
  sorry

end mouse_testes_most_appropriate_l2202_220286


namespace unique_positive_number_l2202_220227

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x^2 = (Real.sqrt 16)^3 := by
  sorry

end unique_positive_number_l2202_220227


namespace dawn_monthly_payments_l2202_220285

/-- Dawn's annual salary in dollars -/
def annual_salary : ℕ := 48000

/-- Dawn's monthly savings rate as a fraction -/
def savings_rate : ℚ := 1/10

/-- Dawn's monthly savings in dollars -/
def monthly_savings : ℕ := 400

/-- The number of months in a year -/
def months_in_year : ℕ := 12

theorem dawn_monthly_payments :
  (annual_salary / months_in_year : ℚ) * savings_rate = monthly_savings :=
sorry

end dawn_monthly_payments_l2202_220285


namespace tangent_line_at_x_1_l2202_220298

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 2*x - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 4*x - 2

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) → y = 5*x - 5 :=
by
  sorry

end tangent_line_at_x_1_l2202_220298


namespace goals_scored_l2202_220274

def bruce_goals : ℕ := 4

def michael_goals : ℕ := 3 * bruce_goals

def total_goals : ℕ := bruce_goals + michael_goals

theorem goals_scored : total_goals = 16 := by
  sorry

end goals_scored_l2202_220274


namespace same_color_probability_l2202_220299

/-- The number of pairs of shoes -/
def num_pairs : ℕ := 9

/-- The total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes to be selected -/
def selection_size : ℕ := 2

/-- The probability of selecting two shoes of the same color -/
theorem same_color_probability : 
  (num_pairs : ℚ) / (total_shoes.choose selection_size) = 1 / 17 := by
  sorry

end same_color_probability_l2202_220299


namespace cube_planes_parallel_l2202_220250

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Given two planes in a cube, determines if they are parallel -/
def are_planes_parallel (cube : Cube) (plane1 plane2 : Plane) : Prop :=
  -- The definition of parallel planes
  ∃ (k : ℝ), k ≠ 0 ∧ plane1.normal = k • plane2.normal

/-- Constructs the plane AB1D1 in the cube -/
def plane_AB1D1 (cube : Cube) : Plane :=
  -- Definition of plane AB1D1
  sorry

/-- Constructs the plane BC1D in the cube -/
def plane_BC1D (cube : Cube) : Plane :=
  -- Definition of plane BC1D
  sorry

/-- Theorem stating that in a cube, plane AB1D1 is parallel to plane BC1D -/
theorem cube_planes_parallel (cube : Cube) : 
  are_planes_parallel cube (plane_AB1D1 cube) (plane_BC1D cube) := by
  sorry

end cube_planes_parallel_l2202_220250


namespace injective_implies_different_outputs_injective_implies_at_most_one_preimage_l2202_220231

-- Define the function f from set A to set B
variable {A B : Type*} (f : A → B)

-- Define injectivity
def Injective (f : A → B) : Prop :=
  ∀ x₁ x₂ : A, f x₁ = f x₂ → x₁ = x₂

-- Theorem 1: If f is injective and x₁ ≠ x₂, then f(x₁) ≠ f(x₂)
theorem injective_implies_different_outputs
  (hf : Injective f) :
  ∀ x₁ x₂ : A, x₁ ≠ x₂ → f x₁ ≠ f x₂ := by
sorry

-- Theorem 2: If f is injective, then for any b ∈ B, there is at most one pre-image in A
theorem injective_implies_at_most_one_preimage
  (hf : Injective f) :
  ∀ b : B, ∃! x : A, f x = b := by
sorry

end injective_implies_different_outputs_injective_implies_at_most_one_preimage_l2202_220231


namespace missing_items_count_l2202_220269

def initial_tshirts : ℕ := 9

def initial_sweaters (t : ℕ) : ℕ := 2 * t

def final_sweaters : ℕ := 3

def final_tshirts (t : ℕ) : ℕ := 3 * t

def missing_items (init_t init_s final_t final_s : ℕ) : ℕ :=
  if final_t > init_t
  then init_s - final_s
  else (init_t - final_t) + (init_s - final_s)

theorem missing_items_count :
  missing_items initial_tshirts (initial_sweaters initial_tshirts) 
                (final_tshirts initial_tshirts) final_sweaters = 15 := by
  sorry

end missing_items_count_l2202_220269


namespace parallel_vectors_magnitude_l2202_220293

/-- Given two planar vectors a and b, where a is parallel to b,
    prove that the magnitude of 3a + b is √5 -/
theorem parallel_vectors_magnitude (y : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, y]
  (a 0 * b 1 = a 1 * b 0) →  -- Parallel condition
  Real.sqrt ((3 * a 0 + b 0)^2 + (3 * a 1 + b 1)^2) = Real.sqrt 5 := by
sorry

end parallel_vectors_magnitude_l2202_220293


namespace cheese_warehouse_problem_l2202_220279

theorem cheese_warehouse_problem (total_rats : ℕ) (cheese_first_night : ℕ) (rats_second_night : ℕ) :
  total_rats > rats_second_night →
  cheese_first_night = 10 →
  rats_second_night = 7 →
  (rats_second_night : ℚ) * (cheese_first_night : ℚ) / (2 * total_rats : ℚ) = 1 →
  cheese_first_night + 1 = 11 := by
  sorry

#check cheese_warehouse_problem

end cheese_warehouse_problem_l2202_220279


namespace prob_diff_tens_digits_l2202_220228

/-- The probability of selecting 6 different integers from 10 to 59 with different tens digits -/
theorem prob_diff_tens_digits : ℝ := by
  -- Define the range of integers
  let range : Set ℕ := {n : ℕ | 10 ≤ n ∧ n ≤ 59}

  -- Define the number of integers to be selected
  let k : ℕ := 6

  -- Define the function that returns the tens digit of a number
  let tens_digit (n : ℕ) : ℕ := n / 10

  -- Define the probability
  let prob : ℝ := (5 * 10 * 9 * 10^4 : ℝ) / (Nat.choose 50 6 : ℝ)

  -- State that the probability is equal to 1500000/5296900
  have h : prob = 1500000 / 5296900 := by sorry

  -- Return the probability
  exact prob

end prob_diff_tens_digits_l2202_220228


namespace solve_transactions_problem_l2202_220297

def transactions_problem (mabel_monday : ℕ) : Prop :=
  let mabel_tuesday : ℕ := mabel_monday + mabel_monday / 10
  let anthony_tuesday : ℕ := 2 * mabel_tuesday
  let cal_tuesday : ℕ := (2 * anthony_tuesday + 2) / 3  -- Rounded up
  let jade_tuesday : ℕ := cal_tuesday + 17
  let isla_wednesday : ℕ := mabel_tuesday + cal_tuesday - 12
  let tim_thursday : ℕ := jade_tuesday + isla_wednesday + (jade_tuesday + isla_wednesday) / 2 + 1  -- Rounded up
  (mabel_monday = 100) → (tim_thursday = 614)

theorem solve_transactions_problem :
  transactions_problem 100 := by sorry

end solve_transactions_problem_l2202_220297


namespace hangar_length_proof_l2202_220275

/-- The length of an airplane hangar given the number of planes it can fit and the length of each plane. -/
def hangar_length (num_planes : ℕ) (plane_length : ℕ) : ℕ :=
  num_planes * plane_length

/-- Theorem stating that a hangar fitting 7 planes of 40 feet each is 280 feet long. -/
theorem hangar_length_proof :
  hangar_length 7 40 = 280 := by
  sorry

end hangar_length_proof_l2202_220275


namespace statement_equivalence_l2202_220201

theorem statement_equivalence (P Q : Prop) : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) := by sorry

end statement_equivalence_l2202_220201


namespace sum_b_plus_c_l2202_220256

theorem sum_b_plus_c (a b c d : ℝ) 
  (h1 : a + b = 12)
  (h2 : c + d = 3)
  (h3 : a + d = 6) :
  b + c = 9 := by
  sorry

end sum_b_plus_c_l2202_220256


namespace opposite_of_negative_nine_l2202_220245

theorem opposite_of_negative_nine : 
  (-(- 9 : ℤ)) = (9 : ℤ) := by sorry

end opposite_of_negative_nine_l2202_220245


namespace quadratic_root_problem_l2202_220243

theorem quadratic_root_problem (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x - k = 0 ∧ x = 0) → 
  (∃ y : ℝ, y^2 + 2*y - k = 0 ∧ y = -2) :=
by sorry

end quadratic_root_problem_l2202_220243


namespace hyperbola_equation_l2202_220205

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (x : ℝ), x^2 + y^2 = 4 ∧ x = -2) →
  (b / a = Real.sqrt 3) →
  (∀ (x y : ℝ), x^2 - y^2 / 3 = 1) :=
by sorry

end hyperbola_equation_l2202_220205


namespace two_true_propositions_l2202_220295

theorem two_true_propositions (a b c : ℝ) : 
  (∃! n : ℕ, n = 2 ∧ 
    n = (if (a > b → a*c^2 > b*c^2) then 1 else 0) +
        (if (a*c^2 > b*c^2 → a > b) then 1 else 0) +
        (if (a ≤ b → a*c^2 ≤ b*c^2) then 1 else 0) +
        (if (a*c^2 ≤ b*c^2 → a ≤ b) then 1 else 0)) :=
by
  sorry

end two_true_propositions_l2202_220295


namespace emily_lives_emily_final_lives_l2202_220218

/-- Calculates the final number of lives in Emily's video game. -/
theorem emily_lives (initial : ℕ) (lost : ℕ) (gained : ℕ) :
  initial ≥ lost →
  initial - lost + gained = initial + gained - lost :=
by
  sorry

/-- Proves that Emily ends up with 41 lives. -/
theorem emily_final_lives : 
  let initial : ℕ := 42
  let lost : ℕ := 25
  let gained : ℕ := 24
  initial ≥ lost →
  initial - lost + gained = 41 :=
by
  sorry

end emily_lives_emily_final_lives_l2202_220218


namespace inequality_solution_l2202_220208

theorem inequality_solution (x y : ℝ) :
  (y^2 - 4*x*y + 4*x^2 < x^2) ↔ (x < y ∧ y < 3*x ∧ x > 0) :=
by sorry

end inequality_solution_l2202_220208


namespace hyperbola_eccentricity_l2202_220287

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (h_arith_mean : (a + b) / 2 = 5 / 2) (h_geom_mean : Real.sqrt (a * b) = Real.sqrt 6) :
  let c := Real.sqrt (a^2 - b^2)
  (c / a) = Real.sqrt 13 / 3 := by sorry

end hyperbola_eccentricity_l2202_220287


namespace gumball_ratio_l2202_220229

/-- Represents the gumball problem scenario -/
structure GumballScenario where
  alicia_gumballs : ℕ
  pedro_multiplier : ℚ
  remaining_gumballs : ℕ

/-- The specific scenario given in the problem -/
def problem_scenario : GumballScenario :=
  { alicia_gumballs := 20
  , pedro_multiplier := 3
  , remaining_gumballs := 60 }

/-- Calculates the total number of gumballs initially in the bowl -/
def total_gumballs (s : GumballScenario) : ℚ :=
  s.alicia_gumballs * (2 + s.pedro_multiplier)

/-- Calculates Pedro's additional gumballs -/
def pedro_additional_gumballs (s : GumballScenario) : ℚ :=
  s.alicia_gumballs * s.pedro_multiplier

/-- The main theorem to prove -/
theorem gumball_ratio (s : GumballScenario) :
  s.alicia_gumballs = 20 →
  s.remaining_gumballs = 60 →
  (total_gumballs s * (3/5) : ℚ) = s.remaining_gumballs →
  (pedro_additional_gumballs s) / s.alicia_gumballs = 3 :=
by sorry

#check gumball_ratio problem_scenario

end gumball_ratio_l2202_220229


namespace consecutive_composite_sequence_l2202_220207

theorem consecutive_composite_sequence (n : ℕ) : ∃ r : ℕ, ∀ k ∈ Finset.range n, ¬(Nat.Prime (r + k + 1)) :=
sorry

end consecutive_composite_sequence_l2202_220207


namespace circle_area_tripled_l2202_220240

theorem circle_area_tripled (r n : ℝ) : 
  (π * (r + n)^2 = 3 * π * r^2) → r = n/2 * (Real.sqrt 3 - 1) := by
  sorry

end circle_area_tripled_l2202_220240


namespace function_properties_l2202_220273

-- Define the function f
def f (x : ℝ) : ℝ := |x - 10| - |x - 25|

-- Define the theorem
theorem function_properties (a : ℝ) 
  (h : ∀ x, f x < 10 * a + 10) : 
  a > 1/2 ∧ ∃ (min_value : ℝ), min_value = 9 ∧ 
  ∀ a, a > 1/2 → 2 * a + 27 / (a^2) ≥ min_value :=
by sorry

end function_properties_l2202_220273


namespace decreasing_function_odd_product_l2202_220234

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Statement 1
theorem decreasing_function (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0) :
  ∀ x y : ℝ, x < y → f y < f x :=
sorry

-- Define an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Statement 3
theorem odd_product (h : is_odd f) :
  is_odd (λ x => f x * f (|x|)) :=
sorry

end decreasing_function_odd_product_l2202_220234


namespace encoded_xyz_value_l2202_220232

/-- Represents a digit in the base-6 encoding system -/
inductive Digit : Type
| U | V | W | X | Y | Z

/-- Converts a Digit to its corresponding natural number value -/
def digit_to_nat (d : Digit) : ℕ :=
  match d with
  | Digit.U => 0
  | Digit.V => 1
  | Digit.W => 2
  | Digit.X => 3
  | Digit.Y => 4
  | Digit.Z => 5

/-- Represents a three-digit number in the base-6 encoding system -/
structure EncodedNumber :=
  (hundreds : Digit)
  (tens : Digit)
  (ones : Digit)

/-- Converts an EncodedNumber to its base-10 value -/
def to_base_10 (n : EncodedNumber) : ℕ :=
  36 * (digit_to_nat n.hundreds) + 6 * (digit_to_nat n.tens) + (digit_to_nat n.ones)

/-- The theorem to be proved -/
theorem encoded_xyz_value :
  ∀ (v x y z : Digit),
    v ≠ x → v ≠ y → v ≠ z → x ≠ y → x ≠ z → y ≠ z →
    to_base_10 (EncodedNumber.mk v x z) + 1 = to_base_10 (EncodedNumber.mk v x y) →
    to_base_10 (EncodedNumber.mk v x y) + 1 = to_base_10 (EncodedNumber.mk v v y) →
    to_base_10 (EncodedNumber.mk x y z) = 184 :=
sorry

end encoded_xyz_value_l2202_220232


namespace P_n_formula_S_3_formula_geometric_sequence_condition_l2202_220223

-- Define the sequence and expansion operation
def Sequence := List ℝ

def expand_by_sum (s : Sequence) : Sequence :=
  match s with
  | [] => []
  | [x] => [x]
  | x::y::rest => x :: (x+y) :: expand_by_sum (y::rest)

-- Define P_n and S_n
def P (n : ℕ) (a b c : ℝ) : ℕ := 
  (expand_by_sum^[n] [a, b, c]).length

def S (n : ℕ) (a b c : ℝ) : ℝ := 
  (expand_by_sum^[n] [a, b, c]).sum

-- Theorem statements
theorem P_n_formula (n : ℕ) (a b c : ℝ) : 
  P n a b c = 2^(n+1) + 1 := by sorry

theorem S_3_formula (a b c : ℝ) :
  S 3 a b c = 14*a + 27*b + 14*c := by sorry

theorem geometric_sequence_condition (a b c : ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, S (n+1) a b c = r * S n a b c) ↔ 
  ((a + c = 0 ∧ b ≠ 0) ∨ (2*b + a + c = 0 ∧ b ≠ 0)) := by sorry

end P_n_formula_S_3_formula_geometric_sequence_condition_l2202_220223


namespace quadratic_equation_roots_as_coefficients_l2202_220264

theorem quadratic_equation_roots_as_coefficients :
  ∀ (A B : ℝ),
  (∀ x : ℝ, x^2 + A*x + B = 0 ↔ x = A ∨ x = B) →
  ((A = 0 ∧ B = 0) ∨ (A = 1 ∧ B = -2)) :=
by sorry

end quadratic_equation_roots_as_coefficients_l2202_220264


namespace marks_ratio_l2202_220217

theorem marks_ratio (P S W : ℚ) 
  (h1 : P / S = 4 / 5) 
  (h2 : S / W = 5 / 2) : 
  P / W = 2 / 1 := by
sorry

end marks_ratio_l2202_220217


namespace ratio_equality_l2202_220221

theorem ratio_equality : ∃ x : ℚ, (x / (2/5)) = ((3/7) / (6/5)) ∧ x = 1/7 := by
  sorry

end ratio_equality_l2202_220221


namespace arithmetic_sequence_common_difference_l2202_220215

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_first_term : a 1 = 1) 
  (h_sum : a 3 + a 5 = 14) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 2 :=
sorry

end arithmetic_sequence_common_difference_l2202_220215


namespace positive_multiple_of_seven_find_x_l2202_220263

theorem positive_multiple_of_seven (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 7 * k ∧ k > 0

theorem find_x : ∃ x : ℕ, 
  positive_multiple_of_seven x ∧ 
  x^2 > 50 ∧ 
  x < 30 ∧
  (x = 14 ∨ x = 21 ∨ x = 28) :=
by sorry

end positive_multiple_of_seven_find_x_l2202_220263


namespace rth_term_of_arithmetic_progression_l2202_220251

def sum_of_n_terms (n : ℕ) : ℕ := 2*n + 3*n^2 + n^3

theorem rth_term_of_arithmetic_progression (r : ℕ) :
  sum_of_n_terms r - sum_of_n_terms (r - 1) = 3*r^2 + 5*r - 2 :=
by sorry

end rth_term_of_arithmetic_progression_l2202_220251
