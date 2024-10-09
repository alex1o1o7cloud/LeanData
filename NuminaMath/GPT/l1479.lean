import Mathlib

namespace mrs_hilt_total_payment_l1479_147942

noncomputable def total_hotdogs : ℕ := 12
noncomputable def cost_first_4 : ℝ := 4 * 0.60
noncomputable def cost_next_5 : ℝ := 5 * 0.75
noncomputable def cost_last_3 : ℝ := 3 * 0.90
noncomputable def total_cost : ℝ := cost_first_4 + cost_next_5 + cost_last_3

theorem mrs_hilt_total_payment : total_cost = 8.85 := by
  -- proof goes here
  sorry

end mrs_hilt_total_payment_l1479_147942


namespace find_a_l1479_147965

noncomputable def f (a x : ℝ) : ℝ := (a * x^2 + 1) / (x + 1)

theorem find_a (a : ℝ) (h1 : ∃ t, t = (f a 1 - 1) / (1 - 0) ∧ t = ((3 * a - 1) / 4)) : a = -1 :=
by
  -- Auxiliary steps to frame the Lean theorem precisely
  let f1 := f a 1
  have h2 : f1 = (a + 1) / 2 := sorry
  have slope_tangent : ∀ t : ℝ, t = (3 * a - 1) / 4 := sorry
  have tangent_eq : (∀ (x y : ℝ), y - f1 = ((3 * a - 1) / 4) * (x - 1)) := sorry
  have pass_point : ∀ (x y : ℝ), (x, y) = (0, 1) -> (1 : ℝ) - ((a + 1) / 2) = ((1 - 3 * a) / 4) := sorry
  exact sorry

end find_a_l1479_147965


namespace two_digit_numbers_solution_l1479_147949

theorem two_digit_numbers_solution :
  ∀ (N : ℕ), (∃ (x y : ℕ), (N = 10 * x + y) ∧ (x < 10) ∧ (y < 10) ∧ 4 * x + 2 * y = N / 2) →
    (N = 32 ∨ N = 64 ∨ N = 96) := 
by
  sorry

end two_digit_numbers_solution_l1479_147949


namespace negation_of_p_range_of_m_if_p_false_l1479_147968

open Real

noncomputable def neg_p (m : ℝ) : Prop :=
∀ x : ℝ, x^2 - m*x - m > 0

theorem negation_of_p (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 - m*x - m ≤ 0) ↔ neg_p m := 
by sorry

theorem range_of_m_if_p_false : 
  (∀ m : ℝ, neg_p m → (-4 < m ∧ m < 0)) :=
by sorry

end negation_of_p_range_of_m_if_p_false_l1479_147968


namespace simplify_expression_l1479_147958

theorem simplify_expression (k : ℤ) (c d : ℤ) 
(h1 : (5 * k + 15) / 5 = c * k + d) 
(h2 : ∀ k, d + c * k = k + 3) : 
c / d = 1 / 3 := 
by 
  sorry

end simplify_expression_l1479_147958


namespace max_product_price_l1479_147902

/-- Conditions: 
1. Company C sells 50 products.
2. The average retail price of the products is $2,500.
3. No product sells for less than $800.
4. Exactly 20 products sell for less than $2,000.
Goal:
Prove that the greatest possible selling price of the most expensive product is $51,000.
-/
theorem max_product_price (n : ℕ) (avg_price : ℝ) (min_price : ℝ) (threshold_price : ℝ) (num_below_threshold : ℕ) :
  n = 50 → 
  avg_price = 2500 → 
  min_price = 800 → 
  threshold_price = 2000 → 
  num_below_threshold = 20 → 
  ∃ max_price : ℝ, max_price = 51000 :=
by 
  sorry

end max_product_price_l1479_147902


namespace projection_matrix_ratio_l1479_147901

theorem projection_matrix_ratio
  (x y : ℚ)
  (h1 : (4/29) * x - (10/29) * y = x)
  (h2 : -(10/29) * x + (25/29) * y = y) :
  y / x = -5/2 :=
by
  sorry

end projection_matrix_ratio_l1479_147901


namespace ratio_of_oranges_l1479_147927

def num_good_oranges : ℕ := 24
def num_bad_oranges : ℕ := 8
def ratio_good_to_bad : ℕ := num_good_oranges / num_bad_oranges

theorem ratio_of_oranges : ratio_good_to_bad = 3 := by
  show 24 / 8 = 3
  sorry

end ratio_of_oranges_l1479_147927


namespace vertical_asymptote_sum_l1479_147938

theorem vertical_asymptote_sum :
  (∀ x : ℝ, 4*x^2 + 6*x + 3 = 0 → x = -1 / 2 ∨ x = -1) →
  (-1 / 2 + -1) = -3 / 2 :=
by
  intro h
  sorry

end vertical_asymptote_sum_l1479_147938


namespace ab_div_c_eq_one_l1479_147946

theorem ab_div_c_eq_one (A B C : ℕ) (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (hne1 : A ≠ B) (hne2 : A ≠ C) (hne3 : B ≠ C) :
  (1 - 1 / (6 + 1 / (6 + 1 / 6)) = 1 / (A + 1 / (B + 1 / 1))) → (A + B) / C = 1 :=
by sorry

end ab_div_c_eq_one_l1479_147946


namespace MN_equal_l1479_147979

def M : Set ℝ := {x | ∃ (m : ℤ), x = Real.sin ((2 * m - 3) * Real.pi / 6)}
def N : Set ℝ := {y | ∃ (n : ℤ), y = Real.cos (n * Real.pi / 3)}

theorem MN_equal : M = N := by
  sorry

end MN_equal_l1479_147979


namespace distance_between_street_lights_l1479_147941

theorem distance_between_street_lights :
  ∀ (n : ℕ) (L : ℝ), n = 18 → L = 16.4 → 8 > 0 →
  (L / (8 : ℕ) = 2.05) :=
by
  intros n L h_n h_L h_nonzero
  sorry

end distance_between_street_lights_l1479_147941


namespace surface_area_of_cube_edge_8_l1479_147918

-- Definition of surface area of a cube
def surface_area_of_cube (edge_length : ℕ) : ℕ :=
  6 * (edge_length * edge_length)

-- Theorem to prove the surface area for a cube with edge length of 8 cm is 384 cm²
theorem surface_area_of_cube_edge_8 : surface_area_of_cube 8 = 384 :=
by
  -- The proof will be inserted here. We use sorry to indicate the missing proof.
  sorry

end surface_area_of_cube_edge_8_l1479_147918


namespace vector_dot_product_l1479_147957

def vector := ℝ × ℝ

def collinear (a b : vector) : Prop :=
  ∃ (k : ℝ), a = (k * b.1, k * b.2)

noncomputable def dot_product (v w : vector) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_dot_product (k : ℝ) (h_collinear : collinear (3 / 2, 1) (3, k))
  (h_k : k = 2) :
  dot_product ((3 / 2, 1) - (3, k)) (2 * (3 / 2, 1) + (3, k)) = -13 :=
by
  sorry

end vector_dot_product_l1479_147957


namespace g_prime_positive_l1479_147980

noncomputable def f (a x : ℝ) := a * x - a * x ^ 2 - Real.log x

noncomputable def g (a x : ℝ) := -2 * (a * x - a * x ^ 2 - Real.log x) - (2 * a + 1) * x ^ 2 + a * x

def g_zero (a x1 x2 : ℝ) := g a x1 = 0 ∧ g a x2 = 0

def x1_x2_condition (x1 x2 : ℝ) := x1 < x2 ∧ x2 < 4 * x1

theorem g_prime_positive (a x1 x2 : ℝ) (h1 : g_zero a x1 x2) (h2 : x1_x2_condition x1 x2) :
  (deriv (g a) ((2 * x1 + x2) / 3)) > 0 := by
  sorry

end g_prime_positive_l1479_147980


namespace vertex_and_maximum_l1479_147971

-- Define the quadratic equation
def quadratic (x : ℝ) : ℝ := -3 * x^2 + 6 * x - 9

-- Prove that the vertex of the parabola quadratic is (1, -6) and it is a maximum point
theorem vertex_and_maximum :
  (∃ x y : ℝ, (quadratic x = y) ∧ (x = 1) ∧ (y = -6)) ∧
  (∀ x : ℝ, quadratic x ≤ quadratic 1) :=
sorry

end vertex_and_maximum_l1479_147971


namespace gcd_9011_4379_l1479_147990

def a : ℕ := 9011
def b : ℕ := 4379

theorem gcd_9011_4379 : Nat.gcd a b = 1 := by
  sorry

end gcd_9011_4379_l1479_147990


namespace sum_of_integers_with_product_5_pow_4_l1479_147924

theorem sum_of_integers_with_product_5_pow_4 :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * b * c * d = 5^4 ∧
  a + b + c + d = 156 :=
by sorry

end sum_of_integers_with_product_5_pow_4_l1479_147924


namespace quadratic_inequality_solution_l1479_147991

theorem quadratic_inequality_solution
  (x : ℝ) 
  (h1 : ∀ x, x^2 + 2 * x - 3 > 0 ↔ x < -3 ∨ x > 1) :
  (2 * x^2 - 3 * x - 2 < 0) ↔ (-1 / 2 < x ∧ x < 2) :=
by {
  sorry
}

end quadratic_inequality_solution_l1479_147991


namespace four_xyz_value_l1479_147986

theorem four_xyz_value (x y z : ℝ) (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 4 * x * y * z = 48 := by
  sorry

end four_xyz_value_l1479_147986


namespace gp_values_l1479_147963

theorem gp_values (p : ℝ) (hp : 0 < p) :
  let a := -p - 12
  let b := 2 * Real.sqrt p
  let c := p - 5
  (b / a = c / b) ↔ p = 4 :=
by
  sorry

end gp_values_l1479_147963


namespace years_since_mothers_death_l1479_147994

noncomputable def jessica_age_at_death (x : ℕ) : ℕ := 40 - x
noncomputable def mother_age_at_death (x : ℕ) : ℕ := 2 * jessica_age_at_death x

theorem years_since_mothers_death (x : ℕ) : mother_age_at_death x + x = 70 ↔ x = 10 :=
by
  sorry

end years_since_mothers_death_l1479_147994


namespace a_b_product_l1479_147943

theorem a_b_product (a b : ℝ) (h1 : 2 * a - b = 1) (h2 : 2 * b - a = 7) : (a + b) * (a - b) = -16 :=
by
  -- The proof would be provided here.
  sorry

end a_b_product_l1479_147943


namespace total_listening_days_l1479_147904

-- Definitions
variables {x y z t : ℕ}

-- Problem statement
theorem total_listening_days (x y z t : ℕ) : (x + y + z) * t = ((x + y + z) * t) :=
by sorry

end total_listening_days_l1479_147904


namespace lemons_minus_pears_l1479_147973

theorem lemons_minus_pears
  (apples : ℕ)
  (pears : ℕ)
  (tangerines : ℕ)
  (lemons : ℕ)
  (watermelons : ℕ)
  (h1 : apples = 8)
  (h2 : pears = 5)
  (h3 : tangerines = 12)
  (h4 : lemons = 17)
  (h5 : watermelons = 10) :
  lemons - pears = 12 := 
sorry

end lemons_minus_pears_l1479_147973


namespace asian_population_percentage_in_west_is_57_l1479_147929

variable (NE MW South West : ℕ)

def total_asian_population (NE MW South West : ℕ) : ℕ :=
  NE + MW + South + West

def west_asian_population_percentage
  (NE MW South West : ℕ) (total_asian_population : ℕ) : ℚ :=
  (West : ℚ) / (total_asian_population : ℚ) * 100

theorem asian_population_percentage_in_west_is_57 :
  total_asian_population 2 3 4 12 = 21 →
  west_asian_population_percentage 2 3 4 12 21 = 57 :=
by
  intros
  sorry

end asian_population_percentage_in_west_is_57_l1479_147929


namespace limit_for_regular_pay_l1479_147989

theorem limit_for_regular_pay 
  (x : ℕ) 
  (regular_pay_rate : ℕ := 3) 
  (overtime_pay_rate : ℕ := 6) 
  (total_pay : ℕ := 186) 
  (overtime_hours : ℕ := 11) 
  (H : 3 * x + (6 * 11) = 186) 
  :
  x = 40 :=
sorry

end limit_for_regular_pay_l1479_147989


namespace fraction_sum_l1479_147972

theorem fraction_sum (x a b : ℕ) (h1 : x = 36 / 99) (h2 : a = 4) (h3 : b = 11) (h4 : Nat.gcd a b = 1) : a + b = 15 :=
by
  sorry

end fraction_sum_l1479_147972


namespace elephant_weight_equivalence_l1479_147982

-- Define the conditions as variables
def elephants := 1000000000
def buildings := 25000

-- Define the question and expected answer
def expected_answer := 40000

-- State the theorem
theorem elephant_weight_equivalence:
  (elephants / buildings = expected_answer) :=
by
  sorry

end elephant_weight_equivalence_l1479_147982


namespace guests_did_not_come_l1479_147956

theorem guests_did_not_come 
  (total_cookies : ℕ) 
  (prepared_guests : ℕ) 
  (cookies_per_guest : ℕ) 
  (total_cookies_eq : total_cookies = 18) 
  (prepared_guests_eq : prepared_guests = 10)
  (cookies_per_guest_eq : cookies_per_guest = 18) 
  (total_cookies_computation : total_cookies = cookies_per_guest) :
  prepared_guests - total_cookies / cookies_per_guest = 9 :=
by
  sorry

end guests_did_not_come_l1479_147956


namespace prop_logic_example_l1479_147995

theorem prop_logic_example (p q : Prop) (h : ¬ (¬ p ∨ ¬ q)) : (p ∧ q) ∧ (p ∨ q) :=
by {
  sorry
}

end prop_logic_example_l1479_147995


namespace binom_15_4_l1479_147921

theorem binom_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end binom_15_4_l1479_147921


namespace initial_money_amount_l1479_147947

theorem initial_money_amount (x : ℕ) (h : x + 16 = 18) : x = 2 := by
  sorry

end initial_money_amount_l1479_147947


namespace behavior_on_neg_interval_l1479_147961

variable (f : ℝ → ℝ)

-- condition 1: f is an odd function
def odd_function : Prop :=
  ∀ x, f (-x) = -f x

-- condition 2: f is increasing on [3, 7]
def increasing_3_7 : Prop :=
  ∀ x y, (3 ≤ x ∧ x < y ∧ y ≤ 7) → f x < f y

-- condition 3: minimum value of f on [3, 7] is 5
def minimum_3_7 : Prop :=
  ∃ a, 3 ≤ a ∧ a ≤ 7 ∧ f a = 5

-- Use the above conditions to prove the required property on [-7, -3].
theorem behavior_on_neg_interval 
  (h1 : odd_function f) 
  (h2 : increasing_3_7 f) 
  (h3 : minimum_3_7 f) : 
  (∀ x y, (-7 ≤ x ∧ x < y ∧ y ≤ -3) → f x < f y) 
  ∧ ∀ x, -7 ≤ x ∧ x ≤ -3 → f x ≤ -5 :=
sorry

end behavior_on_neg_interval_l1479_147961


namespace probability_green_face_l1479_147906

def faces : ℕ := 6
def green_faces : ℕ := 3

theorem probability_green_face : (green_faces : ℚ) / (faces : ℚ) = 1 / 2 := by
  sorry

end probability_green_face_l1479_147906


namespace maximum_bunnies_drum_l1479_147975

-- Define the conditions as provided in the problem
def drumsticks := ℕ -- Natural number type for simplicity
def drum := ℕ -- Natural number type for simplicity

structure Bunny :=
(drum_size : drum)
(stick_length : drumsticks)

def max_drumming_bunnies (bunnies : List Bunny) : ℕ := 
  -- Actual implementation to find the maximum number of drumming bunnies
  sorry

theorem maximum_bunnies_drum (bunnies : List Bunny) (h_size : bunnies.length = 7) : max_drumming_bunnies bunnies = 6 :=
by
  -- Proof of the theorem
  sorry

end maximum_bunnies_drum_l1479_147975


namespace remaining_laps_l1479_147920

def track_length : ℕ := 9
def initial_laps : ℕ := 6
def total_distance : ℕ := 99

theorem remaining_laps : (total_distance - (initial_laps * track_length)) / track_length = 5 := by
  sorry

end remaining_laps_l1479_147920


namespace circle_circumference_ratio_l1479_147955

theorem circle_circumference_ratio (q r p : ℝ) (hq : p = q + r) : 
  (2 * Real.pi * q + 2 * Real.pi * r) / (2 * Real.pi * p) = 1 :=
by
  sorry

end circle_circumference_ratio_l1479_147955


namespace find_side_b_in_triangle_l1479_147911

theorem find_side_b_in_triangle 
  (A B : ℝ) (a : ℝ)
  (h_cosA : Real.cos A = -1/2)
  (h_B : B = Real.pi / 4)
  (h_a : a = 3) :
  ∃ b, b = Real.sqrt 6 :=
by
  sorry

end find_side_b_in_triangle_l1479_147911


namespace sarah_numbers_sum_l1479_147932

-- Definition of x and y being integers with their respective ranges
def isTwoDigit (x : ℕ) : Prop := 10 ≤ x ∧ x ≤ 99
def isThreeDigit (y : ℕ) : Prop := 100 ≤ y ∧ y ≤ 999

-- The condition relating x and y
def formedNumber (x y : ℕ) : Prop := 1000 * x + y = 7 * x * y

-- The Lean 4 statement for the proof problem
theorem sarah_numbers_sum (x y : ℕ) (H1 : isTwoDigit x) (H2 : isThreeDigit y) (H3 : formedNumber x y) : x + y = 1074 :=
  sorry

end sarah_numbers_sum_l1479_147932


namespace number_of_terms_in_arithmetic_sequence_l1479_147959

/-- Define the conditions. -/
def a : ℕ := 2
def d : ℕ := 5
def a_n : ℕ := 57

/-- Define the proof problem. -/
theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, a_n = a + (n - 1) * d ∧ n = 12 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l1479_147959


namespace gcd_is_3_l1479_147997

def gcd_6273_14593 : ℕ := Nat.gcd 6273 14593

theorem gcd_is_3 : gcd_6273_14593 = 3 :=
by
  sorry

end gcd_is_3_l1479_147997


namespace tina_mother_age_l1479_147917

variable {x : ℕ}

theorem tina_mother_age (h1 : 10 + x = 2 * x - 20) : 2010 + x = 2040 :=
by 
  sorry

end tina_mother_age_l1479_147917


namespace faster_speed_l1479_147916

theorem faster_speed (x : ℝ) (h1 : 40 = 8 * 5) (h2 : 60 = x * 5) : x = 12 :=
sorry

end faster_speed_l1479_147916


namespace find_f_5_l1479_147951

theorem find_f_5 : 
  ∀ (f : ℝ → ℝ) (y : ℝ), 
  (∀ x, f x = 2 * x ^ 2 + y) ∧ f 2 = 60 -> f 5 = 102 :=
by
  sorry

end find_f_5_l1479_147951


namespace intercepts_congruence_l1479_147922

theorem intercepts_congruence (m : ℕ) (h : m = 29) (x0 y0 : ℕ) (hx : 0 ≤ x0 ∧ x0 < m) (hy : 0 ≤ y0 ∧ y0 < m) 
  (h1 : 5 * x0 % m = (2 * 0 + 3) % m)  (h2 : (5 * 0) % m = (2 * y0 + 3) % m) : 
  x0 + y0 = 31 := by
  sorry

end intercepts_congruence_l1479_147922


namespace right_triangle_angle_l1479_147915

theorem right_triangle_angle (x : ℝ) (h1 : x + 5 * x = 90) : 5 * x = 75 :=
by
  sorry

end right_triangle_angle_l1479_147915


namespace max_value_of_expression_l1479_147983

-- Define the variables and constraints
variables {a b c d : ℤ}
variables (S : finset ℤ) (a_val b_val c_val d_val : ℤ)

axiom h1 : S = {0, 1, 2, 4, 5}
axiom h2 : a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S
axiom h3 : ∀ x ∈ S, x = a → x ≠ b ∧ x ≠ c ∧ x ≠ d
axiom h4 : ∀ x ∈ S, x = b → x ≠ a ∧ x ≠ c ∧ x ≠ d
axiom h5 : ∀ x ∈ S, x = c → x ≠ a ∧ x ≠ b ∧ x ≠ d
axiom h6 : ∀ x ∈ S, x = d → x ≠ a ∧ x ≠ b ∧ x ≠ c

-- The main theorem to be proven
theorem max_value_of_expression : (∃ a b c d, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
  (∀ x ∈ S, (x = a → x ≠ b ∧ x ≠ c ∧ x ≠ d) ∧ 
             (x = b → x ≠ a ∧ x ≠ c ∧ x ≠ d) ∧ 
             (x = c → x ≠ a ∧ x ≠ b ∧ x ≠ d) ∧ 
             (x = d → x ≠ a ∧ x ≠ b ∧ x ≠ c)) ∧
  (c * a^b - d = 20)) :=
sorry

end max_value_of_expression_l1479_147983


namespace ec_value_l1479_147930

theorem ec_value (AB AD : ℝ) (EFGH1 EFGH2 : ℝ) (x : ℝ)
  (h1 : AB = 2)
  (h2 : AD = 1)
  (h3 : EFGH1 = 1 / 2 * AB)
  (h4 : EFGH2 = 1 / 2 * AD)
  (h5 : 1 + 2 * x = 1)
  : x = 1 / 3 :=
by sorry

end ec_value_l1479_147930


namespace color_property_l1479_147985

theorem color_property (k : ℕ) (h : k ≥ 1) : k = 1 ∨ k = 2 :=
by
  sorry

end color_property_l1479_147985


namespace insert_arithmetic_sequence_l1479_147978

theorem insert_arithmetic_sequence (d a b : ℤ) 
  (h1 : (-1) + 3 * d = 8) 
  (h2 : a = (-1) + d) 
  (h3 : b = a + d) : 
  a = 2 ∧ b = 5 := by
  sorry

end insert_arithmetic_sequence_l1479_147978


namespace fractions_addition_l1479_147937

theorem fractions_addition : (1 / 6 - 5 / 12 + 3 / 8) = 1 / 8 :=
by
  sorry

end fractions_addition_l1479_147937


namespace work_completion_days_l1479_147966

open Real

theorem work_completion_days (days_A : ℝ) (days_B : ℝ) (amount_total : ℝ) (amount_C : ℝ) :
  days_A = 6 ∧ days_B = 8 ∧ amount_total = 5000 ∧ amount_C = 625.0000000000002 →
  (1 / days_A) + (1 / days_B) + (amount_C / amount_total * (1)) = 5 / 12 →
  1 / ((1 / days_A) + (1 / days_B) + (amount_C / amount_total * (1))) = 2.4 :=
  sorry

end work_completion_days_l1479_147966


namespace total_area_expanded_dining_area_l1479_147976

noncomputable def expanded_dining_area_total : ℝ :=
  let rectangular_area := 35
  let radius := 4
  let semi_circular_area := (1 / 2) * Real.pi * (radius^2)
  rectangular_area + semi_circular_area

theorem total_area_expanded_dining_area :
  expanded_dining_area_total = 60.13272 := by
  sorry

end total_area_expanded_dining_area_l1479_147976


namespace symmetry_propositions_l1479_147970

noncomputable def verify_symmetry_conditions (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  Prop :=
  -- This defines the propositions to be proven
  (∀ x : ℝ, a^x - 1 = a^(-x) - 1) ∧
  (∀ x : ℝ, a^(x - 2) = a^(2 - x)) ∧
  (∀ x : ℝ, a^(x + 2) = a^(2 - x))

-- Create the problem statement
theorem symmetry_propositions (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  verify_symmetry_conditions a h1 h2 :=
sorry

end symmetry_propositions_l1479_147970


namespace min_value_proof_l1479_147936

noncomputable def min_value_condition (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (1 / (a + 3) + 1 / (b + 3) = 1 / 4)

theorem min_value_proof : ∃ a b : ℝ, min_value_condition a b ∧ a + 3 * b = 4 + 8 * Real.sqrt 3 := by
  sorry

end min_value_proof_l1479_147936


namespace average_of_first_40_results_l1479_147907

theorem average_of_first_40_results 
  (A : ℝ)
  (avg_other_30 : ℝ := 40)
  (avg_all_70 : ℝ := 34.285714285714285) : A = 30 :=
by 
  let sum1 := A * 40
  let sum2 := avg_other_30 * 30
  let combined_sum := sum1 + sum2
  let combined_avg := combined_sum / 70
  have h1 : combined_avg = avg_all_70 := by sorry
  have h2 : combined_avg = 34.285714285714285 := by sorry
  have h3 : combined_sum = (A * 40) + (40 * 30) := by sorry
  have h4 : (A * 40) + 1200 = 2400 := by sorry
  have h5 : A * 40 = 1200 := by sorry
  have h6 : A = 1200 / 40 := by sorry
  have h7 : A = 30 := by sorry
  exact h7

end average_of_first_40_results_l1479_147907


namespace max_y_for_f_eq_0_l1479_147908

-- Define f(x, y, z) as the remainder when (x - y)! is divided by (x + z).
def f (x y z : ℕ) : ℕ :=
  Nat.factorial (x - y) % (x + z)

-- Conditions given in the problem
variable (x y z : ℕ)
variable (hx : x = 100)
variable (hz : z = 50)

theorem max_y_for_f_eq_0 : 
  f x y z = 0 → y ≤ 75 :=
by
  rw [hx, hz]
  sorry

end max_y_for_f_eq_0_l1479_147908


namespace f_f_five_eq_five_l1479_147984

-- Define the function and its properties
noncomputable def f : ℝ → ℝ := sorry

-- Hypotheses
axiom h1 : ∀ x : ℝ, f (x + 2) = -f x
axiom h2 : f 1 = -5

-- Theorem to prove
theorem f_f_five_eq_five : f (f 5) = 5 :=
sorry

end f_f_five_eq_five_l1479_147984


namespace smallest_N_l1479_147944

theorem smallest_N (l m n : ℕ) (N : ℕ) (h1 : N = l * m * n) (h2 : (l - 1) * (m - 1) * (n - 1) = 300) : 
  N = 462 :=
sorry

end smallest_N_l1479_147944


namespace john_took_more_chickens_l1479_147934

theorem john_took_more_chickens (john_chickens mary_chickens ray_chickens : ℕ)
  (h_ray : ray_chickens = 10)
  (h_mary : mary_chickens = ray_chickens + 6)
  (h_john : john_chickens = mary_chickens + 5) :
  john_chickens - ray_chickens = 11 := by
  sorry

end john_took_more_chickens_l1479_147934


namespace monthly_rent_is_1300_l1479_147967

def shop_length : ℕ := 10
def shop_width : ℕ := 10
def annual_rent_per_square_foot : ℕ := 156

def area_of_shop : ℕ := shop_length * shop_width
def annual_rent_for_shop : ℕ := annual_rent_per_square_foot * area_of_shop

def monthly_rent : ℕ := annual_rent_for_shop / 12

theorem monthly_rent_is_1300 : monthly_rent = 1300 := by
  sorry

end monthly_rent_is_1300_l1479_147967


namespace correct_average_marks_l1479_147926

theorem correct_average_marks 
  (avg_marks : ℝ) 
  (num_students : ℕ) 
  (incorrect_marks : ℕ → (ℝ × ℝ)) :
  avg_marks = 85 →
  num_students = 50 →
  incorrect_marks 0 = (95, 45) →
  incorrect_marks 1 = (78, 58) →
  incorrect_marks 2 = (120, 80) →
  (∃ corrected_avg_marks : ℝ, corrected_avg_marks = 82.8) :=
by
  sorry

end correct_average_marks_l1479_147926


namespace range_of_a_l1479_147996

theorem range_of_a :
  (∀ t : ℝ, 0 < t ∧ t ≤ 2 → (t / (t^2 + 9) ≤ a ∧ a ≤ (t + 2) / t^2)) →
  (2 / 13 ≤ a ∧ a ≤ 1) :=
by
  intro h
  -- Proof of the theorem goes here
  sorry

end range_of_a_l1479_147996


namespace non_empty_solution_set_l1479_147913

theorem non_empty_solution_set (a : ℝ) (h : a > 0) : (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 :=
by
  sorry

end non_empty_solution_set_l1479_147913


namespace total_voters_l1479_147993

theorem total_voters (x : ℝ)
  (h1 : 0.35 * x + 80 = (0.35 * x + 80) + 0.65 * x - (0.65 * x - 0.45 * (x + 80)))
  (h2 : 0.45 * (x + 80) = 0.65 * x) : 
  x + 80 = 260 := by
  -- We'll provide the proof here
  sorry

end total_voters_l1479_147993


namespace lily_final_balance_l1479_147900

noncomputable def initial_balance : ℝ := 55
noncomputable def shirt_cost : ℝ := 7
noncomputable def shoes_cost : ℝ := 3 * shirt_cost
noncomputable def book_cost : ℝ := 4
noncomputable def books_amount : ℝ := 5
noncomputable def gift_fraction : ℝ := 0.20

noncomputable def remaining_balance : ℝ :=
  initial_balance - 
  shirt_cost - 
  shoes_cost - 
  books_amount * book_cost - 
  gift_fraction * (initial_balance - shirt_cost - shoes_cost - books_amount * book_cost)

theorem lily_final_balance : remaining_balance = 5.60 := 
by 
  sorry

end lily_final_balance_l1479_147900


namespace complex_pure_imaginary_l1479_147914

theorem complex_pure_imaginary (a : ℝ) : 
  ((a^2 - 3*a + 2) = 0) → (a = 2) := 
  by 
  sorry

end complex_pure_imaginary_l1479_147914


namespace johns_profit_is_200_l1479_147964

def num_woodburnings : ℕ := 20
def price_per_woodburning : ℕ := 15
def cost_of_wood : ℕ := 100
def total_revenue : ℕ := num_woodburnings * price_per_woodburning
def profit : ℕ := total_revenue - cost_of_wood

theorem johns_profit_is_200 : profit = 200 :=
by
  -- proof steps go here
  sorry

end johns_profit_is_200_l1479_147964


namespace tiles_touching_walls_of_room_l1479_147928

theorem tiles_touching_walls_of_room (length width : Nat) 
    (hl : length = 10) (hw : width = 5) : 
    2 * length + 2 * width - 4 = 26 := by
  sorry

end tiles_touching_walls_of_room_l1479_147928


namespace transportation_cost_l1479_147945

-- Definitions for the conditions
def number_of_original_bags : ℕ := 80
def weight_of_original_bag : ℕ := 50
def total_cost_original : ℕ := 6000

def scale_factor_bags : ℕ := 3
def scale_factor_weight : ℚ := 3 / 5

-- Derived quantities
def number_of_new_bags : ℕ := scale_factor_bags * number_of_original_bags
def weight_of_new_bag : ℚ := scale_factor_weight * weight_of_original_bag
def cost_per_original_bag : ℚ := total_cost_original / number_of_original_bags
def cost_per_new_bag : ℚ := cost_per_original_bag * (weight_of_new_bag / weight_of_original_bag)

-- Final cost calculation
def total_cost_new : ℚ := number_of_new_bags * cost_per_new_bag

-- The statement that needs to be proved
theorem transportation_cost : total_cost_new = 10800 := sorry

end transportation_cost_l1479_147945


namespace total_stickers_l1479_147923

-- Definitions for the given conditions
def stickers_per_page : ℕ := 10
def number_of_pages : ℕ := 22

-- The theorem to be proven
theorem total_stickers : stickers_per_page * number_of_pages = 220 := by
  sorry

end total_stickers_l1479_147923


namespace cone_sector_central_angle_l1479_147953

noncomputable def base_radius := 1
noncomputable def slant_height := 2
noncomputable def circumference (r : ℝ) := 2 * Real.pi * r
noncomputable def arc_length (r : ℝ) := circumference r
noncomputable def central_angle (l : ℝ) (s : ℝ) := l / s

theorem cone_sector_central_angle : central_angle (arc_length base_radius) slant_height = Real.pi := 
by 
  -- Here we acknowledge that the proof would go, but it is left out as per instructions.
  sorry

end cone_sector_central_angle_l1479_147953


namespace const_seq_is_arithmetic_not_geometric_l1479_147939

-- Define the sequence
def const_seq (n : ℕ) : ℕ := 0

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (seq : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, seq (n + 1) = seq n + d

-- Define what it means for a sequence to be geometric
def is_geometric_sequence (seq : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, ∀ n : ℕ, seq (n + 1) = seq n * r

-- The proof statement
theorem const_seq_is_arithmetic_not_geometric :
  is_arithmetic_sequence const_seq ∧ ¬ is_geometric_sequence const_seq :=
by
  sorry

end const_seq_is_arithmetic_not_geometric_l1479_147939


namespace find_c_l1479_147931

theorem find_c (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 310) : C = 10 :=
by
  sorry

end find_c_l1479_147931


namespace rabbit_carrot_count_l1479_147962

theorem rabbit_carrot_count
  (r h : ℕ)
  (hr : r = h - 3)
  (eq_carrots : 4 * r = 5 * h) :
  4 * r = 36 :=
by
  sorry

end rabbit_carrot_count_l1479_147962


namespace count_four_digit_numbers_with_thousands_digit_one_l1479_147999

theorem count_four_digit_numbers_with_thousands_digit_one : 
  ∃ N : ℕ, N = 1000 ∧ (∀ n : ℕ, 1000 ≤ n ∧ n < 2000 → (n / 1000 = 1)) :=
sorry

end count_four_digit_numbers_with_thousands_digit_one_l1479_147999


namespace vasechkin_result_l1479_147988

theorem vasechkin_result (x : ℕ) (h : (x / 2 * 7) - 1001 = 7) : (x / 8) ^ 2 - 1001 = 295 :=
by
  sorry

end vasechkin_result_l1479_147988


namespace find_width_l1479_147948

variable (a b : ℝ)

def perimeter : ℝ := 6 * a + 4 * b
def length : ℝ := 2 * a + b
def width : ℝ := a + b

theorem find_width (h : perimeter a b = 6 * a + 4 * b)
                   (h₂ : length a b = 2 * a + b) : width a b = (perimeter a b) / 2 - length a b := by
  sorry

end find_width_l1479_147948


namespace total_population_expression_l1479_147935

variables (b g t: ℕ)

-- Assuming the given conditions
def condition1 := b = 4 * g
def condition2 := g = 8 * t

-- The theorem to prove
theorem total_population_expression (h1 : condition1 b g) (h2 : condition2 g t) :
    b + g + t = 41 * b / 32 := sorry

end total_population_expression_l1479_147935


namespace b_ne_d_l1479_147903

-- Conditions
def P (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def Q (x : ℝ) (c d : ℝ) : ℝ := x^2 + c * x + d

def PQ_eq_QP_no_real_roots (a b c d : ℝ) : Prop := 
  ∀ (x : ℝ), P (Q x c d) a b ≠ Q (P x a b) c d

-- Goal
theorem b_ne_d (a b c d : ℝ) (h : PQ_eq_QP_no_real_roots a b c d) : b ≠ d := 
sorry

end b_ne_d_l1479_147903


namespace shaded_area_correct_l1479_147905

def diameter := 3 -- inches
def pattern_length := 18 -- inches equivalent to 1.5 feet

def radius := diameter / 2 -- radius calculation

noncomputable def area_of_one_circle := Real.pi * (radius ^ 2)
def number_of_circles := pattern_length / diameter
noncomputable def total_shaded_area := number_of_circles * area_of_one_circle

theorem shaded_area_correct :
  total_shaded_area = 13.5 * Real.pi :=
  by
  sorry

end shaded_area_correct_l1479_147905


namespace age_difference_l1479_147954

def A := 10
def B := 8
def C := B / 2
def total_age (A B C : ℕ) : Prop := A + B + C = 22

theorem age_difference (A B C : ℕ) (hB : B = 8) (hC : B = 2 * C) (h_total : total_age A B C) : A - B = 2 := by
  sorry

end age_difference_l1479_147954


namespace balance_after_6_months_l1479_147912

noncomputable def final_balance : ℝ :=
  let balance_m1 := 5000 * (1 + 0.04 / 12)
  let balance_m2 := (balance_m1 + 1000) * (1 + 0.042 / 12)
  let balance_m3 := balance_m2 * (1 + 0.038 / 12)
  let balance_m4 := (balance_m3 - 1500) * (1 + 0.05 / 12)
  let balance_m5 := (balance_m4 + 750) * (1 + 0.052 / 12)
  let balance_m6 := (balance_m5 - 1000) * (1 + 0.045 / 12)
  balance_m6

theorem balance_after_6_months : final_balance = 4371.51 := sorry

end balance_after_6_months_l1479_147912


namespace log_9_256_eq_4_log_2_3_l1479_147977

noncomputable def logBase9Base2Proof : Prop :=
  (Real.log 256 / Real.log 9 = 4 * (Real.log 3 / Real.log 2))

theorem log_9_256_eq_4_log_2_3 : logBase9Base2Proof :=
by
  sorry

end log_9_256_eq_4_log_2_3_l1479_147977


namespace values_of_x_for_f_l1479_147969

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def is_monotonically_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem values_of_x_for_f (f : ℝ → ℝ) 
  (h1 : is_even_function f) 
  (h2 : is_monotonically_increasing_on_nonneg f) : 
  (∀ x : ℝ, f (2*x - 1) < f 3 ↔ (-1 < x ∧ x < 2)) :=
by
  sorry

end values_of_x_for_f_l1479_147969


namespace total_tweets_l1479_147998

-- Conditions and Definitions
def tweets_happy_per_minute := 18
def tweets_hungry_per_minute := 4
def tweets_reflection_per_minute := 45
def minutes_each_period := 20

-- Proof Problem Statement
theorem total_tweets : 
  (minutes_each_period * tweets_happy_per_minute) + 
  (minutes_each_period * tweets_hungry_per_minute) + 
  (minutes_each_period * tweets_reflection_per_minute) = 1340 :=
by
  sorry

end total_tweets_l1479_147998


namespace inverse_negative_exchange_l1479_147987

theorem inverse_negative_exchange (f1 f2 f3 f4 : ℝ → ℝ) (hx1 : ∀ x, f1 x = x - (1/x))
  (hx2 : ∀ x, f2 x = x + (1/x)) (hx3 : ∀ x, f3 x = Real.log x)
  (hx4 : ∀ x, f4 x = if 0 < x ∧ x < 1 then x else if x = 1 then 0 else -(1/x)) :
  (∀ x, f1 (1/x) = -f1 x) ∧ (∀ x, f2 (1/x) = -f2 x) ∧ (∀ x, f3 (1/x) = -f3 x) ∧
  (∀ x, f4 (1/x) = -f4 x) ↔ True := by 
  sorry

end inverse_negative_exchange_l1479_147987


namespace two_colonies_same_time_l1479_147981

def doubles_in_size_every_day (P : ℕ → ℕ) : Prop :=
∀ n, P (n + 1) = 2 * P n

def reaches_habitat_limit_in (f : ℕ → ℕ) (days limit : ℕ) : Prop :=
f days = limit

theorem two_colonies_same_time (P : ℕ → ℕ) (Q : ℕ → ℕ) (limit : ℕ) (days : ℕ)
  (h1 : doubles_in_size_every_day P)
  (h2 : reaches_habitat_limit_in P days limit)
  (h3 : ∀ n, Q n = 2 * P n) :
  reaches_habitat_limit_in Q days limit :=
sorry

end two_colonies_same_time_l1479_147981


namespace equilateral_triangle_dot_product_l1479_147992

noncomputable def dot_product_sum (a b c : ℝ) := 
  a * b + b * c + c * a

theorem equilateral_triangle_dot_product 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : A = 1)
  (h2 : B = 1)
  (h3 : C = 1)
  (h4 : a = 1)
  (h5 : b = 1)
  (h6 : c = 1) :
  dot_product_sum a b c = 1 / 2 :=
by 
  sorry

end equilateral_triangle_dot_product_l1479_147992


namespace A_alone_completes_one_work_in_32_days_l1479_147950

def amount_of_work_per_day_by_B : ℝ := sorry
def amount_of_work_per_day_by_A : ℝ := 3 * amount_of_work_per_day_by_B
def total_work : ℝ := (amount_of_work_per_day_by_A + amount_of_work_per_day_by_B) * 24

theorem A_alone_completes_one_work_in_32_days :
  total_work = amount_of_work_per_day_by_A * 32 :=
by
  sorry

end A_alone_completes_one_work_in_32_days_l1479_147950


namespace correct_option_C_l1479_147952

noncomputable def question := "Which of the following operations is correct?"
noncomputable def option_A := (-2)^2
noncomputable def option_B := (-2)^3
noncomputable def option_C := (-1/2)^3
noncomputable def option_D := (-7/3)^3
noncomputable def correct_answer := -1/8

theorem correct_option_C :
  option_C = correct_answer := by
  sorry

end correct_option_C_l1479_147952


namespace neither_rain_nor_snow_l1479_147933

theorem neither_rain_nor_snow 
  (p_rain : ℚ)
  (p_snow : ℚ)
  (independent : Prop) 
  (h_rain : p_rain = 4/10)
  (h_snow : p_snow = 1/5)
  (h_independent : independent)
  : (1 - p_rain) * (1 - p_snow) = 12 / 25 := 
by
  sorry

end neither_rain_nor_snow_l1479_147933


namespace no_integers_satisfy_eq_l1479_147910

theorem no_integers_satisfy_eq (m n : ℤ) : m^2 ≠ n^5 - 4 := 
by {
  sorry
}

end no_integers_satisfy_eq_l1479_147910


namespace xyz_inequality_l1479_147909

theorem xyz_inequality (x y z : ℝ) (h : x + y + z = 0) : 
  6 * (x^3 + y^3 + z^3)^2 ≤ (x^2 + y^2 + z^2)^3 := 
by sorry

end xyz_inequality_l1479_147909


namespace geometric_probability_l1479_147919

theorem geometric_probability :
  let total_length := (1/2 - 0)
  let favorable_length := (1/3 - 0)
  total_length > 0 ∧ favorable_length > 0 →
  (favorable_length / total_length) = 2 / 3 :=
by
  intros _ _ _
  sorry

end geometric_probability_l1479_147919


namespace product_of_sums_of_squares_l1479_147940

-- Given conditions as definitions
def sum_of_squares (a b : ℤ) : ℤ := a^2 + b^2

-- Prove that the product of two sums of squares is also a sum of squares
theorem product_of_sums_of_squares (a b n k : ℤ) (K P : ℤ) (hK : K = sum_of_squares a b) (hP : P = sum_of_squares n k) :
    K * P = (a * n + b * k)^2 + (a * k - b * n)^2 := 
by
  sorry

end product_of_sums_of_squares_l1479_147940


namespace larger_integer_is_7sqrt14_l1479_147925

theorem larger_integer_is_7sqrt14 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a / b = 7 / 3) (h2 : a * b = 294) : max a b = 7 * Real.sqrt 14 :=
by 
  sorry

end larger_integer_is_7sqrt14_l1479_147925


namespace frustum_surface_area_l1479_147974

noncomputable def total_surface_area_of_frustum
  (R r h : ℝ) : ℝ :=
  let s := Real.sqrt (h^2 + (R - r)^2)
  let A_lateral := Real.pi * (R + r) * s
  let A_top := Real.pi * r^2
  let A_bottom := Real.pi * R^2
  A_lateral + A_top + A_bottom

theorem frustum_surface_area :
  total_surface_area_of_frustum 8 2 5 = 10 * Real.pi * Real.sqrt 61 + 68 * Real.pi :=
  sorry

end frustum_surface_area_l1479_147974


namespace complement_of_M_in_U_l1479_147960

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 4, 6}
def complement_U_M : Set ℕ := {1, 3, 5}

theorem complement_of_M_in_U :
  (U \ M) = complement_U_M :=
by
  sorry

end complement_of_M_in_U_l1479_147960
