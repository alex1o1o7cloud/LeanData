import Mathlib

namespace high_five_count_l903_90328

def number_of_people : ℕ := 12

/-- The number of unique pairs (high-fives) in a group of n people -/
def number_of_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

theorem high_five_count :
  number_of_pairs number_of_people = 66 :=
by sorry

end high_five_count_l903_90328


namespace power_equality_l903_90301

theorem power_equality (n : ℕ) : 4^6 = 8^n → n = 4 := by sorry

end power_equality_l903_90301


namespace complex_in_fourth_quadrant_l903_90306

/-- A complex number z is in the fourth quadrant if its real part is positive and its imaginary part is negative. -/
def in_fourth_quadrant (z : ℂ) : Prop := 0 < z.re ∧ z.im < 0

/-- Given -1 < m < 1, the complex number (1-i) + m(1+i) is in the fourth quadrant. -/
theorem complex_in_fourth_quadrant (m : ℝ) (h : -1 < m ∧ m < 1) :
  in_fourth_quadrant ((1 - Complex.I) + m * (1 + Complex.I)) := by
  sorry

end complex_in_fourth_quadrant_l903_90306


namespace circle_center_is_two_neg_three_l903_90317

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*x + y^2 + 6*y - 11 = 0

/-- The center of a circle -/
def circle_center (h k : ℝ) : ℝ × ℝ := (h, k)

/-- Theorem: The center of the circle defined by x^2 - 4x + y^2 + 6y - 11 = 0 is (2, -3) -/
theorem circle_center_is_two_neg_three :
  ∃ (r : ℝ), ∀ (x y : ℝ), circle_equation x y ↔ (x - 2)^2 + (y - (-3))^2 = r^2 :=
sorry

end circle_center_is_two_neg_three_l903_90317


namespace binary_multiplication_addition_l903_90338

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n

theorem binary_multiplication_addition :
  let a := [true, false, false, true, true]  -- 11001₂
  let b := [false, true, true]               -- 110₂
  let c := [false, true, false, true]        -- 1010₂
  let result := [false, true, true, true, true, true, false, true]  -- 10111110₂
  (binary_to_nat a * binary_to_nat b + binary_to_nat c) = binary_to_nat result := by
  sorry

end binary_multiplication_addition_l903_90338


namespace factor_implies_h_value_l903_90308

theorem factor_implies_h_value (h : ℝ) (m : ℝ) : 
  (∃ k : ℝ, m^2 - h*m - 24 = (m - 8) * k) → h = 5 := by
sorry

end factor_implies_h_value_l903_90308


namespace min_value_of_a_l903_90381

theorem min_value_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 2 3 ∧ (1 + a * x) / (x * 2^x) ≥ 1) → 
  a ≥ 7/2 :=
sorry

end min_value_of_a_l903_90381


namespace find_n_l903_90361

-- Define the polynomial
def p (x y : ℝ) := (x^2 - y)^7

-- Define the fourth term of the expansion
def fourth_term (x y : ℝ) := -35 * x^8 * y^3

-- Define the fifth term of the expansion
def fifth_term (x y : ℝ) := 35 * x^6 * y^4

-- Theorem statement
theorem find_n (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m * n = 7)
  (h4 : fourth_term m n = fifth_term m n) : n = (49 : ℝ)^(1/3) := by
  sorry

end find_n_l903_90361


namespace function_equation_solution_l903_90302

/-- A function satisfying the given functional equation -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x * f y - f (x * y)) / 2 = x + y + 1

/-- The main theorem stating that any function satisfying the equation must be f(x) = x + 2 -/
theorem function_equation_solution (f : ℝ → ℝ) (h : satisfies_equation f) : 
  ∀ x : ℝ, f x = x + 2 := by
  sorry

end function_equation_solution_l903_90302


namespace arithmetic_sequence_formula_l903_90323

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence with given conditions,
    the general term formula is a_n = 4 - 2n. -/
theorem arithmetic_sequence_formula 
  (a : ℕ → ℤ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_3 : a 3 = -2) 
  (h_7 : a 7 = -10) :
  ∃ c : ℤ, ∀ n : ℕ, a n = 4 - 2 * n :=
sorry

end arithmetic_sequence_formula_l903_90323


namespace sqrt_inequality_l903_90332

theorem sqrt_inequality : Real.sqrt 3 - Real.sqrt 2 > Real.sqrt 6 - Real.sqrt 5 := by
  sorry

end sqrt_inequality_l903_90332


namespace arithmetic_equality_l903_90342

theorem arithmetic_equality : (18 / (5 + 2 - 3)) * 4 = 18 := by
  sorry

end arithmetic_equality_l903_90342


namespace prime_relation_l903_90375

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem prime_relation (p q : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (h : 13 * p + 1 = q + 2) : 
  q = 39 := by sorry

end prime_relation_l903_90375


namespace absolute_value_equation_solution_l903_90321

theorem absolute_value_equation_solution : 
  {x : ℝ | |x - 5| = 3*x + 6} = {-11/2, -1/4} := by sorry

end absolute_value_equation_solution_l903_90321


namespace first_problem_answer_l903_90330

/-- Given three math problems with the following properties:
    1. The second problem's answer is twice the first problem's answer.
    2. The third problem's answer is 400 less than the sum of the first two problems' answers.
    3. The total of all three answers is 3200.
    Prove that the answer to the first math problem is 600. -/
theorem first_problem_answer (a : ℕ) : 
  (∃ b c : ℕ, 
    b = 2 * a ∧ 
    c = a + b - 400 ∧ 
    a + b + c = 3200) → 
  a = 600 :=
by sorry

end first_problem_answer_l903_90330


namespace pot_height_problem_l903_90399

/-- Given two similar right-angled triangles, where one triangle has a height of 20 inches and
    a base of 10 inches, and the other triangle has a base of 20 inches,
    prove that the height of the second triangle is 40 inches. -/
theorem pot_height_problem (h₁ h₂ : ℝ) (b₁ b₂ : ℝ) :
  h₁ = 20 → b₁ = 10 → b₂ = 20 → (h₁ / b₁ = h₂ / b₂) → h₂ = 40 :=
by sorry

end pot_height_problem_l903_90399


namespace identity_implies_equality_l903_90390

theorem identity_implies_equality (a b c d : ℝ) :
  (∀ x : ℝ, a * x + b = c * x + d) → (a = c ∧ b = d) := by
  sorry

end identity_implies_equality_l903_90390


namespace union_of_sets_l903_90345

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3, 4}
  let B : Set ℕ := {2, 4, 5, 6}
  A ∪ B = {1, 2, 3, 4, 5, 6} := by
sorry

end union_of_sets_l903_90345


namespace x_value_proof_l903_90358

theorem x_value_proof (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 18) : x = 14 := by
  sorry

end x_value_proof_l903_90358


namespace faye_remaining_money_is_correct_l903_90363

/-- Calculates Faye's remaining money after her shopping spree -/
def faye_remaining_money (original_money : ℚ) : ℚ :=
  let father_gift := 3 * original_money
  let mother_gift := 2 * father_gift
  let grandfather_gift := 4 * original_money
  let total_money := original_money + father_gift + mother_gift + grandfather_gift
  let muffin_cost := 15 * 1.75
  let cookie_cost := 10 * 2.5
  let juice_cost := 2 * 4
  let candy_cost := 25 * 0.25
  let total_item_cost := muffin_cost + cookie_cost + juice_cost + candy_cost
  let tip := 0.15 * (muffin_cost + cookie_cost)
  let total_spent := total_item_cost + tip
  total_money - total_spent

theorem faye_remaining_money_is_correct : 
  faye_remaining_money 20 = 206.81 := by sorry

end faye_remaining_money_is_correct_l903_90363


namespace all_equations_have_one_negative_one_positive_root_l903_90341

-- Define the equations
def equation1 (x : ℝ) : Prop := 4 * x^2 - 6 = 34
def equation2 (x : ℝ) : Prop := (3*x-2)^2 = (x+1)^2
def equation3 (x : ℝ) : Prop := (x^2-12).sqrt = (2*x-2).sqrt

-- Define the property of having one negative and one positive root
def has_one_negative_one_positive_root (f : ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ f x ∧ f y

-- Theorem statement
theorem all_equations_have_one_negative_one_positive_root :
  has_one_negative_one_positive_root equation1 ∧
  has_one_negative_one_positive_root equation2 ∧
  has_one_negative_one_positive_root equation3 :=
sorry

end all_equations_have_one_negative_one_positive_root_l903_90341


namespace equation_solution_l903_90391

theorem equation_solution : 
  ∃ y : ℝ, (7 * y / (y + 4) - 5 / (y + 4) = 2 / (y + 4)) ∧ y = 1 := by
  sorry

end equation_solution_l903_90391


namespace inverse_of_complex_expression_l903_90315

theorem inverse_of_complex_expression (i : ℂ) (h : i^2 = -1) :
  (3*i - 2*i⁻¹)⁻¹ = -i/5 := by sorry

end inverse_of_complex_expression_l903_90315


namespace fractional_equation_solution_l903_90377

theorem fractional_equation_solution :
  ∃! x : ℚ, (2 - x) / (x - 3) + 3 = 2 / (3 - x) ∧ x = 5 / 2 := by
  sorry

end fractional_equation_solution_l903_90377


namespace popsicle_consumption_l903_90333

/-- The number of Popsicles eaten in a given time period -/
def popsicles_eaten (rate : ℚ) (time : ℚ) : ℚ :=
  time / rate

/-- Proves that eating 1 Popsicle every 20 minutes for 6 hours results in 18 Popsicles -/
theorem popsicle_consumption : popsicles_eaten (20 / 60) 6 = 18 := by
  sorry

#eval popsicles_eaten (20 / 60) 6

end popsicle_consumption_l903_90333


namespace sum_of_three_numbers_l903_90393

theorem sum_of_three_numbers : 4.75 + 0.303 + 0.432 = 5.485 := by
  sorry

end sum_of_three_numbers_l903_90393


namespace square_area_increase_l903_90396

/-- The increase in area of a square when its side length increases by 6 -/
theorem square_area_increase (a : ℝ) : 
  (a + 6)^2 - a^2 = 12*a + 36 := by
  sorry

end square_area_increase_l903_90396


namespace container_capacity_l903_90359

theorem container_capacity (C : ℝ) 
  (h1 : 0.30 * C + 54 = 0.75 * C) : C = 120 := by
  sorry

end container_capacity_l903_90359


namespace sum_squared_l903_90349

theorem sum_squared (x y : ℝ) (h1 : 2*x*(x+y) = 58) (h2 : 3*y*(x+y) = 111) : 
  (x + y)^2 = 28561 / 25 := by
sorry

end sum_squared_l903_90349


namespace min_value_of_2x_plus_y_min_value_is_sqrt_3_l903_90354

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + 2*x*y - 1 = 0) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a^2 + 2*a*b - 1 = 0 → 2*x + y ≤ 2*a + b :=
by
  sorry

theorem min_value_is_sqrt_3 (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + 2*x*y - 1 = 0) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + 2*a*b - 1 = 0 ∧ 2*a + b = Real.sqrt 3 :=
by
  sorry

end min_value_of_2x_plus_y_min_value_is_sqrt_3_l903_90354


namespace park_pairings_l903_90340

/-- The number of unique pairings in a group of 12 people where two specific individuals do not interact -/
theorem park_pairings (n : ℕ) (h : n = 12) : 
  (n.choose 2) - 1 = 65 := by
  sorry

end park_pairings_l903_90340


namespace theorem_1_theorem_2_theorem_3_l903_90353

-- Define the set S
def S (m l : ℝ) := {x : ℝ | m ≤ x ∧ x ≤ l}

-- Define the property that x^2 ∈ S for all x ∈ S
def closed_under_square (m l : ℝ) :=
  ∀ x, x ∈ S m l → x^2 ∈ S m l

-- Theorem 1
theorem theorem_1 (m l : ℝ) (h : closed_under_square m l) :
  m = 1 → S m l = {1} := by sorry

-- Theorem 2
theorem theorem_2 (m l : ℝ) (h : closed_under_square m l) :
  m = -1/2 → 1/4 ≤ l ∧ l ≤ 1 := by sorry

-- Theorem 3
theorem theorem_3 (m l : ℝ) (h : closed_under_square m l) :
  l = 1/2 → -Real.sqrt 2 / 2 ≤ m ∧ m ≤ 0 := by sorry

end theorem_1_theorem_2_theorem_3_l903_90353


namespace reseat_twelve_women_l903_90378

/-- Number of ways to reseat n women under given conditions -/
def T : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 => T (n + 2) + T (n + 1) + T n

/-- Theorem stating that the number of ways to reseat 12 women is 927 -/
theorem reseat_twelve_women : T 12 = 927 := by
  sorry

end reseat_twelve_women_l903_90378


namespace tom_jogging_distance_l903_90355

/-- Tom's jogging rate in miles per minute -/
def jogging_rate : ℚ := 1 / 15

/-- Time Tom jogs in minutes -/
def jogging_time : ℚ := 45

/-- Distance Tom jogs in miles -/
def jogging_distance : ℚ := jogging_rate * jogging_time

theorem tom_jogging_distance :
  jogging_distance = 3 := by sorry

end tom_jogging_distance_l903_90355


namespace all_vints_are_xaffs_l903_90370

-- Define the types
variable (Zibb Xaff Yurn Worb Vint : Type)

-- Define the conditions
variable (h1 : Zibb → Xaff)
variable (h2 : Yurn → Xaff)
variable (h3 : Worb → Zibb)
variable (h4 : Yurn → Worb)
variable (h5 : Worb → Vint)
variable (h6 : Vint → Yurn)

-- Theorem to prove
theorem all_vints_are_xaffs : Vint → Xaff := by sorry

end all_vints_are_xaffs_l903_90370


namespace tan_theta_range_l903_90312

theorem tan_theta_range (θ : Real) (a : Real) 
  (h1 : -π/2 < θ ∧ θ < π/2) 
  (h2 : Real.sin θ + Real.cos θ = a) 
  (h3 : 0 < a ∧ a < 1) : 
  -1 < Real.tan θ ∧ Real.tan θ < 0 := by
  sorry

end tan_theta_range_l903_90312


namespace diagonal_passes_through_720_cubes_l903_90343

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_through (l w h : ℕ) : ℕ :=
  l + w + h - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l) + Nat.gcd l (Nat.gcd w h)

/-- Theorem stating that for a 120 × 360 × 400 rectangular solid, 
    an internal diagonal passes through 720 unit cubes -/
theorem diagonal_passes_through_720_cubes :
  cubes_passed_through 120 360 400 = 720 := by
  sorry

end diagonal_passes_through_720_cubes_l903_90343


namespace martins_walk_l903_90322

/-- The distance between Martin's house and Lawrence's house -/
def distance : ℝ := 12

/-- The time Martin spent walking -/
def time : ℝ := 6

/-- Martin's walking speed -/
def speed : ℝ := 2

/-- Theorem: The distance between Martin's house and Lawrence's house is 12 miles -/
theorem martins_walk : distance = speed * time := by
  sorry

end martins_walk_l903_90322


namespace product_factorization_l903_90350

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def product : ℕ := (List.range 9).foldl (λ acc i => acc * factorial (20 + i)) 1

theorem product_factorization :
  ∃ (n : ℕ), n > 0 ∧ product = 825 * n^3 := by sorry

end product_factorization_l903_90350


namespace decimal_to_fraction_l903_90339

theorem decimal_to_fraction :
  (2.35 : ℚ) = 47 / 20 := by sorry

end decimal_to_fraction_l903_90339


namespace equation_solution_l903_90320

theorem equation_solution (x : ℝ) : (x + 5)^2 = 16 ↔ x = -1 ∨ x = -9 := by
  sorry

end equation_solution_l903_90320


namespace unknown_blanket_rate_l903_90305

/-- Proves that given the specified blanket purchases and average price, the unknown rate is 285 --/
theorem unknown_blanket_rate (num_blankets_1 num_blankets_2 num_unknown : ℕ)
                              (price_1 price_2 avg_price : ℚ) :
  num_blankets_1 = 3 →
  num_blankets_2 = 5 →
  num_unknown = 2 →
  price_1 = 100 →
  price_2 = 150 →
  avg_price = 162 →
  let total_blankets := num_blankets_1 + num_blankets_2 + num_unknown
  let total_cost := avg_price * total_blankets
  let known_cost := num_blankets_1 * price_1 + num_blankets_2 * price_2
  let unknown_cost := total_cost - known_cost
  unknown_cost / num_unknown = 285 := by sorry

end unknown_blanket_rate_l903_90305


namespace units_digit_of_7_pow_2050_l903_90348

theorem units_digit_of_7_pow_2050 : 7^2050 % 10 = 9 := by
  sorry

end units_digit_of_7_pow_2050_l903_90348


namespace isosceles_triangle_l903_90329

theorem isosceles_triangle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_equation : a^2 - b^2 + a*c - b*c = 0) : 
  a = b ∨ b = c ∨ c = a := by
  sorry

end isosceles_triangle_l903_90329


namespace M_eq_open_interval_compare_values_l903_90304

/-- The function f(x) = |x| - |2x - 1| -/
def f (x : ℝ) : ℝ := |x| - |2*x - 1|

/-- The set M is defined as the solution set of f(x) > -1 -/
def M : Set ℝ := {x | f x > -1}

/-- Theorem stating that M is the open interval (0, 2) -/
theorem M_eq_open_interval : M = Set.Ioo 0 2 := by sorry

/-- Theorem comparing a^2 - a + 1 and 1/a for a ∈ M -/
theorem compare_values (a : ℝ) (h : a ∈ M) :
  (0 < a ∧ a < 1 → a^2 - a + 1 < 1/a) ∧
  (a = 1 → a^2 - a + 1 = 1/a) ∧
  (1 < a ∧ a < 2 → a^2 - a + 1 > 1/a) := by sorry

end M_eq_open_interval_compare_values_l903_90304


namespace cyclist_travel_time_l903_90316

/-- Proves that a cyclist who travels 2.5 miles in 10 minutes will take 20 minutes
    to travel 4 miles when their speed is reduced by 20% due to a headwind -/
theorem cyclist_travel_time (initial_distance : ℝ) (initial_time : ℝ) 
  (new_distance : ℝ) (speed_reduction : ℝ) :
  initial_distance = 2.5 →
  initial_time = 10 →
  new_distance = 4 →
  speed_reduction = 0.2 →
  (new_distance / ((initial_distance / initial_time) * (1 - speed_reduction))) = 20 := by
  sorry

#check cyclist_travel_time

end cyclist_travel_time_l903_90316


namespace find_a_value_l903_90310

def A : Set ℝ := {2, 4}
def B (a : ℝ) : Set ℝ := {a, a^2 + 3}

theorem find_a_value (h : A ∩ B a = {2}) : a = 2 := by
  sorry

end find_a_value_l903_90310


namespace sum_of_x_and_y_is_8_l903_90357

def is_greatest_factorial_under_1000 (n : ℕ) : Prop :=
  n.factorial < 1000 ∧ ∀ m : ℕ, m > n → m.factorial ≥ 1000

theorem sum_of_x_and_y_is_8 :
  ∀ x y : ℕ,
    x > 0 →
    y > 1 →
    is_greatest_factorial_under_1000 x →
    x + y = 8 :=
by
  sorry

end sum_of_x_and_y_is_8_l903_90357


namespace ball_drawing_properties_l903_90324

/-- Represents the number of balls drawn -/
def n : ℕ := 3

/-- Represents the initial number of red balls -/
def r : ℕ := 5

/-- Represents the initial number of black balls -/
def b : ℕ := 2

/-- Represents the total number of balls -/
def total : ℕ := r + b

/-- Represents the random variable for the number of red balls drawn without replacement -/
def X : Fin (n + 1) → ℝ := sorry

/-- Represents the random variable for the number of black balls drawn without replacement -/
def Y : Fin (n + 1) → ℝ := sorry

/-- Represents the random variable for the number of red balls drawn with replacement -/
def ξ : Fin (n + 1) → ℝ := sorry

/-- The expected value of X -/
noncomputable def E_X : ℝ := sorry

/-- The expected value of Y -/
noncomputable def E_Y : ℝ := sorry

/-- The expected value of ξ -/
noncomputable def E_ξ : ℝ := sorry

/-- The variance of X -/
noncomputable def D_X : ℝ := sorry

/-- The variance of ξ -/
noncomputable def D_ξ : ℝ := sorry

theorem ball_drawing_properties :
  (E_X / E_Y = r / b) ∧ (E_X = E_ξ) ∧ (D_X < D_ξ) := by sorry

end ball_drawing_properties_l903_90324


namespace inequality_proof_l903_90318

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (x^4)/(y*(1-y^2)) + (y^4)/(z*(1-z^2)) + (z^4)/(x*(1-x^2)) ≥ 1/8 := by
  sorry

end inequality_proof_l903_90318


namespace log_expression_equals_one_l903_90376

theorem log_expression_equals_one : 
  (((1 - Real.log 3 / Real.log 6) ^ 2 + (Real.log 2 / Real.log 6) * (Real.log 18 / Real.log 6)) / (Real.log 4 / Real.log 6)) = 1 := by
  sorry

end log_expression_equals_one_l903_90376


namespace count_even_factors_div_by_5_l903_90389

/-- The number of even natural-number factors divisible by 5 of 2^3 * 5^2 * 11^1 -/
def num_even_factors_div_by_5 : ℕ :=
  let n : ℕ := 2^3 * 5^2 * 11^1
  -- Define the function here
  12

theorem count_even_factors_div_by_5 :
  num_even_factors_div_by_5 = 12 := by sorry

end count_even_factors_div_by_5_l903_90389


namespace solution_set_f_nonnegative_range_of_a_l903_90395

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x| - 2

-- Theorem 1: Solution set of f(x) ≥ 0
theorem solution_set_f_nonnegative :
  {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -3 ∨ x ≥ 1} :=
sorry

-- Theorem 2: Range of values for a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x ≤ |x| + a) → a ≥ -3 :=
sorry

end solution_set_f_nonnegative_range_of_a_l903_90395


namespace batsman_average_after_12th_innings_l903_90385

/-- Represents a batsman's cricket performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored : ℚ) / (b.innings + 1 : ℚ)

theorem batsman_average_after_12th_innings 
  (b : Batsman)
  (h1 : b.innings = 11)
  (h2 : newAverage b 65 = b.average + 3)
  : newAverage b 65 = 32 := by
  sorry

end batsman_average_after_12th_innings_l903_90385


namespace codemaster_combinations_l903_90300

/-- The number of colors available for the pegs -/
def num_colors : ℕ := 8

/-- The number of slots in the code -/
def num_slots : ℕ := 5

/-- Theorem: The number of possible secret codes in Codemaster -/
theorem codemaster_combinations : num_colors ^ num_slots = 32768 := by
  sorry

end codemaster_combinations_l903_90300


namespace complex_fraction_simplification_l903_90325

/-- Given that i² = -1, prove that (2 - 3i) / (4 - 5i) = 23/41 - (2/41)i -/
theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (2 - 3*i) / (4 - 5*i) = 23/41 - (2/41)*i := by sorry

end complex_fraction_simplification_l903_90325


namespace unique_root_condition_l903_90369

/-- The characteristic equation of a thermal energy process -/
def characteristic_equation (x t : ℝ) : Prop := x^3 - 3*x = t

/-- The condition for a unique root -/
def has_unique_root (t : ℝ) : Prop :=
  ∃! x, characteristic_equation x t

/-- The main theorem about the uniqueness and magnitude of the root -/
theorem unique_root_condition (t : ℝ) :
  has_unique_root t ↔ abs t > 2 ∧ ∀ x, characteristic_equation x t → abs x > 2 :=
sorry

end unique_root_condition_l903_90369


namespace selection_theorem_l903_90394

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The total number of people to choose from -/
def total_people : ℕ := num_boys + num_girls

/-- The number of people to be selected -/
def select_count : ℕ := 4

/-- The number of ways to select 4 people from 4 boys and 3 girls,
    such that the selection includes at least one boy and one girl -/
def selection_methods : ℕ := 34

theorem selection_theorem :
  (Nat.choose total_people select_count) - (Nat.choose num_boys select_count) = selection_methods :=
sorry

end selection_theorem_l903_90394


namespace cute_six_digit_integers_l903_90360

def is_permutation (n : ℕ) (digits : List ℕ) : Prop :=
  digits.length = 6 ∧ digits.toFinset = Finset.range 6

def first_k_digits_divisible (digits : List ℕ) : Prop :=
  ∀ k : ℕ, k ≤ 6 → k ∣ (digits.take k).foldl (λ acc d => acc * 10 + d) 0

def is_cute (digits : List ℕ) : Prop :=
  is_permutation 6 digits ∧ first_k_digits_divisible digits

theorem cute_six_digit_integers :
  ∃! (s : Finset (List ℕ)), s.card = 2 ∧ ∀ digits, digits ∈ s ↔ is_cute digits :=
sorry

end cute_six_digit_integers_l903_90360


namespace school_population_l903_90364

/-- Given a school with boys, girls, and teachers, prove the total population. -/
theorem school_population (b g t : ℕ) : 
  b = 4 * g ∧ g = 8 * t → b + g + t = (41 * b) / 32 := by
  sorry

end school_population_l903_90364


namespace certain_number_proof_l903_90347

theorem certain_number_proof (x : ℝ) : 0.28 * x + 0.45 * 250 = 224.5 → x = 400 := by
  sorry

end certain_number_proof_l903_90347


namespace right_triangle_hypotenuse_l903_90335

theorem right_triangle_hypotenuse (base height hypotenuse : ℝ) :
  base = 3 →
  height * base / 2 = 6 →
  base^2 + height^2 = hypotenuse^2 →
  hypotenuse = 5 := by
  sorry

end right_triangle_hypotenuse_l903_90335


namespace quadratic_equation_solution_l903_90392

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  (x + 2) * (x - 3) = 2 * x - 6

-- Define the general form of the equation
def general_form (x : ℝ) : Prop :=
  x^2 - 3*x = 0

-- Theorem statement
theorem quadratic_equation_solution :
  (∀ x, quadratic_equation x ↔ general_form x) ∧
  (∃ x₁ x₂, x₁ = 0 ∧ x₂ = 3 ∧ ∀ x, general_form x ↔ (x = x₁ ∨ x = x₂)) :=
sorry

end quadratic_equation_solution_l903_90392


namespace three_digit_divisibility_l903_90398

def is_valid (x y z : Nat) : Prop :=
  let n := 579000 + 100 * x + 10 * y + z
  n % 5 = 0 ∧ n % 7 = 0 ∧ n % 9 = 0

theorem three_digit_divisibility :
  ∀ x y z : Nat,
    x < 10 ∧ y < 10 ∧ z < 10 →
    is_valid x y z ↔ (x = 6 ∧ y = 0 ∧ z = 0) ∨ 
                     (x = 2 ∧ y = 8 ∧ z = 5) ∨ 
                     (x = 9 ∧ y = 1 ∧ z = 5) :=
by sorry

#check three_digit_divisibility

end three_digit_divisibility_l903_90398


namespace shaded_perimeter_is_32_l903_90372

/-- Given two squares ABCD and BEFG sharing vertex B, with E on BC, G on AB,
    this structure represents the configuration described in the problem. -/
structure SquareConfiguration where
  -- Side length of square ABCD
  x : ℝ
  -- Assertion that E is on BC and G is on AB
  h_e_on_bc : True
  h_g_on_ab : True
  -- Length of CG is 9
  h_cg_length : 9 = 9
  -- Area of shaded region (ABCD - BEFG) is 47
  h_shaded_area : x^2 - (81 - x^2) = 47

/-- Theorem stating that under the given conditions, 
    the perimeter of the shaded region (which is the same as ABCD) is 32. -/
theorem shaded_perimeter_is_32 (config : SquareConfiguration) :
  4 * config.x = 32 := by
  sorry

#check shaded_perimeter_is_32

end shaded_perimeter_is_32_l903_90372


namespace special_triangle_sides_l903_90314

/-- An isosceles triangle with perimeter 60 and centroid on the inscribed circle. -/
structure SpecialTriangle where
  -- Two equal sides
  a : ℝ
  -- Third side
  b : ℝ
  -- Perimeter is 60
  perimeter_eq : 2 * a + b = 60
  -- a > 0 and b > 0
  a_pos : a > 0
  b_pos : b > 0
  -- Centroid on inscribed circle condition
  centroid_on_inscribed : 3 * (a * b) = 60 * (a + b - (2 * a + b) / 2)

/-- The sides of a special triangle are 25, 25, and 10. -/
theorem special_triangle_sides (t : SpecialTriangle) : t.a = 25 ∧ t.b = 10 := by
  sorry

end special_triangle_sides_l903_90314


namespace find_divisor_l903_90362

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h1 : dividend = 689)
  (h2 : quotient = 19)
  (h3 : remainder = 5)
  (h4 : dividend = quotient * (dividend / quotient) + remainder) :
  dividend / quotient = 36 := by
  sorry

end find_divisor_l903_90362


namespace absolute_value_calculation_l903_90388

theorem absolute_value_calculation : |-3| * 2 - (-1) = 7 := by
  sorry

end absolute_value_calculation_l903_90388


namespace train_bridge_crossing_time_l903_90379

/-- Proves that a train with given length and speed takes the calculated time to cross a bridge of given length. -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 90) 
  (h2 : train_speed_kmh = 45) 
  (h3 : bridge_length = 285) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

#check train_bridge_crossing_time

end train_bridge_crossing_time_l903_90379


namespace polynomial_division_theorem_l903_90387

theorem polynomial_division_theorem (x : ℝ) : 
  8 * x^3 + 4 * x^2 - 6 * x - 9 = (x + 3) * (8 * x^2 - 20 * x + 54) - 171 := by
  sorry

end polynomial_division_theorem_l903_90387


namespace student_number_problem_l903_90382

theorem student_number_problem : ∃ x : ℤ, 2 * x - 148 = 110 ∧ x = 129 := by
  sorry

end student_number_problem_l903_90382


namespace system_solution_l903_90367

theorem system_solution : 
  ∃ (x y : ℚ), (4 * x - 7 * y = -9) ∧ (5 * x + 3 * y = -11) ∧ (x = -104/47) ∧ (y = 1/47) := by
  sorry

end system_solution_l903_90367


namespace smallest_positive_a_l903_90326

/-- A function with period 20 -/
def IsPeriodic20 (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x - 20) = g x

/-- The property we want to prove for the smallest positive a -/
def HasProperty (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g ((x - a) / 10) = g (x / 10)

theorem smallest_positive_a (g : ℝ → ℝ) (h : IsPeriodic20 g) :
  (∃ a > 0, HasProperty g a) →
  (∃ a > 0, HasProperty g a ∧ ∀ b, 0 < b → b < a → ¬HasProperty g b) →
  (∃! a, a = 200 ∧ a > 0 ∧ HasProperty g a ∧ ∀ b, 0 < b → b < a → ¬HasProperty g b) :=
sorry

end smallest_positive_a_l903_90326


namespace arithmetic_geometric_sequence_l903_90344

-- Define an arithmetic sequence with common difference 2
def arithmetic_seq (a : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, a (n + 1) = a n + 2

-- Define a geometric sequence for three terms
def geometric_seq (x y z : ℤ) : Prop :=
  y * y = x * z

-- Theorem statement
theorem arithmetic_geometric_sequence (a : ℤ → ℤ) :
  arithmetic_seq a →
  geometric_seq (a 1) (a 3) (a 4) →
  a 1 = -8 := by
sorry

end arithmetic_geometric_sequence_l903_90344


namespace complex_arithmetic_l903_90311

theorem complex_arithmetic (A M S : ℂ) (P : ℝ) 
  (hA : A = 5 - 2*I) 
  (hM : M = -3 + 2*I) 
  (hS : S = 2*I) 
  (hP : P = 3) : 
  A - M + S - P = 5 - 2*I := by
  sorry

end complex_arithmetic_l903_90311


namespace representative_selection_count_l903_90386

def female_students : ℕ := 5
def male_students : ℕ := 7
def total_representatives : ℕ := 5
def max_female_representatives : ℕ := 2

theorem representative_selection_count :
  (Nat.choose male_students total_representatives) +
  (Nat.choose female_students 1 * Nat.choose male_students 4) +
  (Nat.choose female_students 2 * Nat.choose male_students 3) = 546 := by
  sorry

end representative_selection_count_l903_90386


namespace intersecting_lines_l903_90397

theorem intersecting_lines (x y : ℝ) : 
  (2*x - y)^2 - (x + 3*y)^2 = 0 ↔ (x = 4*y ∨ x = -2/3*y) := by
  sorry

end intersecting_lines_l903_90397


namespace stratified_sampling_correct_l903_90307

/-- Represents the number of students to be chosen from a class in stratified sampling -/
def stratified_sample (total_students : ℕ) (class_size : ℕ) (sample_size : ℕ) : ℕ :=
  (class_size * sample_size) / total_students

theorem stratified_sampling_correct (class1_size class2_size sample_size : ℕ) 
  (h1 : class1_size = 54)
  (h2 : class2_size = 42)
  (h3 : sample_size = 16) :
  (stratified_sample (class1_size + class2_size) class1_size sample_size = 9) ∧
  (stratified_sample (class1_size + class2_size) class2_size sample_size = 7) := by
  sorry

end stratified_sampling_correct_l903_90307


namespace hyperbola_property_l903_90356

/-- The hyperbola with equation x^2 - y^2 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p | p.1^2 - p.2^2 = 1}

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- The length of the semi-major axis of the hyperbola -/
def a : ℝ := sorry

theorem hyperbola_property (P : ℝ × ℝ) (h₁ : P ∈ Hyperbola)
    (h₂ : (P.1 - F₁.1, P.2 - F₁.2) • (P.1 - F₂.1, P.2 - F₂.2) = 0) :
    ‖(P.1 - F₁.1, P.2 - F₁.2) + (P.1 - F₂.1, P.2 - F₂.2)‖ = 2 * a := by
  sorry

end hyperbola_property_l903_90356


namespace unfactorizable_quartic_l903_90327

theorem unfactorizable_quartic : ¬ ∃ (a b c d : ℤ), ∀ (x : ℝ),
  x^4 + 2*x^2 + 2*x + 2 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end unfactorizable_quartic_l903_90327


namespace interview_room_occupancy_l903_90303

/-- Given a waiting room and an interview room, prove that the number of people in the interview room is 5. -/
theorem interview_room_occupancy (waiting_room interview_room : ℕ) : interview_room = 5 :=
  by
  -- Define the initial number of people in the waiting room
  have initial_waiting : waiting_room = 22 := by sorry
  
  -- Define the number of new arrivals
  have new_arrivals : ℕ := 3
  
  -- Define the relationship between waiting room and interview room after new arrivals
  have after_arrivals : waiting_room + new_arrivals = 5 * interview_room := by sorry
  
  sorry -- Proof goes here

end interview_room_occupancy_l903_90303


namespace polar_equation_represents_line_and_circle_l903_90383

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ * Real.sin θ = Real.sin (2 * θ)

-- Define a line in Cartesian coordinates (x-axis in this case)
def is_line (x y : ℝ) : Prop := y = 0

-- Define a circle in Cartesian coordinates
def is_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Theorem statement
theorem polar_equation_represents_line_and_circle :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (is_line x₁ y₁ ∧ polar_equation x₁ 0) ∧
    (is_circle x₂ y₂ ∧ ∃ (ρ θ : ℝ), polar_equation ρ θ ∧ x₂ = ρ * Real.cos θ ∧ y₂ = ρ * Real.sin θ) :=
sorry

end polar_equation_represents_line_and_circle_l903_90383


namespace find_a_l903_90319

-- Define the universal set U
def U (a : ℤ) : Set ℤ := {2, 4, a^2 - a + 1}

-- Define set A
def A (a : ℤ) : Set ℤ := {a + 4, 4}

-- Define the complement of A relative to U
def complement_A (a : ℤ) : Set ℤ := U a \ A a

-- Theorem statement
theorem find_a : ∃ (a : ℤ), 
  (U a = {2, 4, a^2 - a + 1}) ∧ 
  (A a = {a + 4, 4}) ∧ 
  (complement_A a = {7}) ∧ 
  (a = -2) :=
sorry

end find_a_l903_90319


namespace sphere_radius_from_hole_l903_90373

/-- Given a spherical hole with a width of 30 cm at the top and a depth of 10 cm,
    the radius of the sphere that created this hole is 16.25 cm. -/
theorem sphere_radius_from_hole (hole_width : ℝ) (hole_depth : ℝ) (sphere_radius : ℝ) : 
  hole_width = 30 → 
  hole_depth = 10 → 
  sphere_radius = (hole_width ^ 2 / 4 + hole_depth ^ 2) / (2 * hole_depth) → 
  sphere_radius = 16.25 := by
sorry

end sphere_radius_from_hole_l903_90373


namespace pentagon_area_l903_90380

-- Define the pentagon
structure Pentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  angle : ℝ

-- Define the given pentagon
def given_pentagon : Pentagon :=
  { side1 := 18
  , side2 := 25
  , side3 := 30
  , side4 := 28
  , side5 := 22
  , angle := 110 }

-- Define the area calculation function
noncomputable def calculate_area (p : Pentagon) : ℝ := sorry

-- Theorem stating the area of the given pentagon
theorem pentagon_area :
  ∃ ε > 0, |calculate_area given_pentagon - 738| < ε := by sorry

end pentagon_area_l903_90380


namespace quadratic_equation_solution_l903_90374

theorem quadratic_equation_solution (m n : ℤ) :
  m^2 + 2*m*n + 2*n^2 - 4*n + 4 = 0 → m = -2 ∧ n = 2 := by
  sorry

end quadratic_equation_solution_l903_90374


namespace triangle_angle_measurement_l903_90334

theorem triangle_angle_measurement (A B C : ℝ) : 
  -- Triangle ABC exists
  A + B + C = 180 →
  -- Measure of angle C is three times the measure of angle B
  C = 3 * B →
  -- Angle B measures 30°
  B = 30 →
  -- Then the measure of angle A is 60°
  A = 60 := by
sorry

end triangle_angle_measurement_l903_90334


namespace skating_speed_ratio_l903_90337

theorem skating_speed_ratio (v_f v_s : ℝ) (h1 : v_f > 0) (h2 : v_s > 0) :
  (v_f + v_s) / (v_f - v_s) = 5 → v_f / v_s = 3/2 := by
  sorry

end skating_speed_ratio_l903_90337


namespace series_sum_l903_90371

/-- r is the positive real solution to x³ - ¼x - 1 = 0 -/
def r : ℝ := sorry

/-- T is the sum of the infinite series r + 2r⁴ + 3r⁷ + 4r¹⁰ + ... -/
noncomputable def T : ℝ := sorry

/-- The equation that r satisfies -/
axiom r_eq : r^3 - (1/4)*r - 1 = 0

/-- The main theorem: T equals 4 / (1 + 4/r) -/
theorem series_sum : T = 4 / (1 + 4/r) := by sorry

end series_sum_l903_90371


namespace lunch_break_duration_l903_90346

/- Define the workshop as a unit (100%) -/
def workshop : ℝ := 1

/- Define the working rates -/
variable (p : ℝ) -- Paula's painting rate (workshop/hour)
variable (h : ℝ) -- Combined rate of helpers (workshop/hour)

/- Define the lunch break duration in hours -/
variable (L : ℝ)

/- Monday's work -/
axiom monday_work : (9 - L) * (p + h) = 0.6 * workshop

/- Tuesday's work -/
axiom tuesday_work : (7 - L) * h = 0.3 * workshop

/- Wednesday's work -/
axiom wednesday_work : (10 - L) * p = 0.1 * workshop

/- The sum of work done on all three days equals the whole workshop -/
axiom total_work : 0.6 * workshop + 0.3 * workshop + 0.1 * workshop = workshop

/- Theorem: The lunch break is 48 minutes -/
theorem lunch_break_duration : L * 60 = 48 := by
  sorry

end lunch_break_duration_l903_90346


namespace parabola_comparison_l903_90365

theorem parabola_comparison : ∀ x : ℝ, 
  x^2 - (1/3)*x + 3 < x^2 + (1/3)*x + 4 := by sorry

end parabola_comparison_l903_90365


namespace unique_valid_square_l903_90336

/-- A number is a square with exactly two non-zero digits, one of which is 3 -/
def is_valid_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 ∧ 
  (∃ a b : ℕ, a ≠ 0 ∧ b ≠ 0 ∧ (a = 3 ∨ b = 3) ∧ n = 10 * a + b) ∧
  (∀ c d e : ℕ, n ≠ 100 * c + 10 * d + e ∨ c = 0 ∨ d = 0 ∨ e = 0)

theorem unique_valid_square : 
  ∀ n : ℕ, is_valid_square n ↔ n = 36 :=
by sorry

end unique_valid_square_l903_90336


namespace torturie_problem_l903_90309

/-- The number of the last remaining prisoner in the Torturie problem -/
def lastPrisoner (n : ℕ) : ℕ :=
  2 * n - 2^(Nat.log2 n + 1) + 1

/-- The Torturie problem statement -/
theorem torturie_problem (n : ℕ) (h : n > 0) :
  lastPrisoner n = 
    let k := Nat.log2 n
    2 * n - 2^(k + 1) + 1 :=
by sorry

end torturie_problem_l903_90309


namespace inequality_solution_l903_90331

open Real

theorem inequality_solution (x y : ℝ) : 
  (Real.sqrt 3 * Real.tan x - (Real.sin y) ^ (1/4) - 
   Real.sqrt ((3 / (Real.cos x)^2) + (Real.sin y)^(1/2) - 6) ≥ Real.sqrt 3) ↔ 
  (∃ (n k : ℤ), x = π/4 + n*π ∧ y = k*π) :=
sorry

end inequality_solution_l903_90331


namespace min_money_lost_l903_90384

def check_amount : ℕ := 1270
def bill_10 : ℕ := 10
def bill_50 : ℕ := 50
def total_bills_used : ℕ := 15

def money_lost (t f : ℕ) : ℕ :=
  check_amount - (t * bill_10 + f * bill_50)

theorem min_money_lost :
  ∃ (t f : ℕ),
    (t + f = total_bills_used) ∧
    ((t = f + 1) ∨ (t = f - 1)) ∧
    (∀ (t' f' : ℕ),
      (t' + f' = total_bills_used) ∧
      ((t' = f' + 1) ∨ (t' = f' - 1)) →
      money_lost t f ≤ money_lost t' f') ∧
    money_lost t f = 800 :=
by sorry

end min_money_lost_l903_90384


namespace train_platform_length_l903_90313

/-- Given a train passing a pole in t seconds and a platform in 6t seconds at constant velocity,
    prove that the length of the platform is 5 times the length of the train. -/
theorem train_platform_length
  (t : ℝ)
  (train_length : ℝ)
  (platform_length : ℝ)
  (velocity : ℝ)
  (h1 : velocity = train_length / t)
  (h2 : velocity = (train_length + platform_length) / (6 * t))
  : platform_length = 5 * train_length := by
  sorry

end train_platform_length_l903_90313


namespace mass_of_ccl4_produced_l903_90352

-- Define the chemical equation
def balanced_equation : String := "CaC2 + 4 Cl2O → CCl4 + CaCl2O"

-- Define the number of moles of reaction
def reaction_moles : ℝ := 8

-- Define molar masses
def molar_mass_carbon : ℝ := 12.01
def molar_mass_chlorine : ℝ := 35.45

-- Define the molar mass of CCl4
def molar_mass_ccl4 : ℝ := molar_mass_carbon + 4 * molar_mass_chlorine

-- Theorem statement
theorem mass_of_ccl4_produced : 
  reaction_moles * molar_mass_ccl4 = 1230.48 := by sorry

end mass_of_ccl4_produced_l903_90352


namespace gwen_bookcase_distribution_l903_90351

/-- Given a bookcase with mystery and picture book shelves, 
    calculates the number of books on each shelf. -/
def books_per_shelf (mystery_shelves : ℕ) (picture_shelves : ℕ) (total_books : ℕ) : ℕ :=
  total_books / (mystery_shelves + picture_shelves)

/-- Proves that Gwen's bookcase has 4 books on each shelf. -/
theorem gwen_bookcase_distribution :
  books_per_shelf 5 3 32 = 4 := by
  sorry

#eval books_per_shelf 5 3 32

end gwen_bookcase_distribution_l903_90351


namespace largest_n_is_correct_l903_90368

/-- Represents the coefficients of the quadratic expression 6x^2 + nx + 48 -/
structure QuadraticCoeffs where
  n : ℤ

/-- Represents the coefficients of the linear factors (2x + A)(3x + B) -/
structure LinearFactors where
  A : ℤ
  B : ℤ

/-- Checks if the given linear factors produce the quadratic expression -/
def is_valid_factorization (q : QuadraticCoeffs) (f : LinearFactors) : Prop :=
  (2 * f.B + 3 * f.A = q.n) ∧ (f.A * f.B = 48)

/-- The largest value of n for which the quadratic can be factored -/
def largest_n : ℤ := 99

theorem largest_n_is_correct : 
  (∀ q : QuadraticCoeffs, ∃ f : LinearFactors, is_valid_factorization q f → q.n ≤ largest_n) ∧
  (∃ q : QuadraticCoeffs, ∃ f : LinearFactors, is_valid_factorization q f ∧ q.n = largest_n) :=
sorry

end largest_n_is_correct_l903_90368


namespace mango_lassi_price_l903_90366

/-- The cost of a mango lassi at Delicious Delhi restaurant --/
def mango_lassi_cost (samosa_cost pakora_cost tip_percentage total_cost : ℚ) : ℚ :=
  total_cost - (samosa_cost + pakora_cost + (samosa_cost + pakora_cost) * tip_percentage / 100)

/-- Theorem stating the cost of the mango lassi --/
theorem mango_lassi_price :
  mango_lassi_cost 6 12 25 25 = (5/2 : ℚ) := by sorry

end mango_lassi_price_l903_90366
