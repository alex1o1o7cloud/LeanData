import Mathlib

namespace NUMINAMATH_GPT_sequence_problem_l799_79997

theorem sequence_problem 
  (a : ℕ → ℕ) 
  (h1 : a 1 = 5) 
  (h2 : ∀ n : ℕ, a (n + 1) - a n = 3 + 4 * (n - 1)) : 
  a 50 = 4856 :=
sorry

end NUMINAMATH_GPT_sequence_problem_l799_79997


namespace NUMINAMATH_GPT_find_greatest_number_l799_79975

def numbers := [0.07, -0.41, 0.8, 0.35, -0.9]

theorem find_greatest_number :
  ∃ x ∈ numbers, x > 0.7 ∧ ∀ y ∈ numbers, y > 0.7 → y = 0.8 :=
by
  sorry

end NUMINAMATH_GPT_find_greatest_number_l799_79975


namespace NUMINAMATH_GPT_smallest_integral_k_l799_79965

theorem smallest_integral_k (k : ℤ) :
  (297 - 108 * k < 0) ↔ (k ≥ 3) :=
sorry

end NUMINAMATH_GPT_smallest_integral_k_l799_79965


namespace NUMINAMATH_GPT_speed_of_stream_l799_79931

-- Conditions
variables (v : ℝ) -- speed of the stream in kmph
variables (boat_speed_still_water : ℝ := 10) -- man's speed in still water in kmph
variables (distance : ℝ := 90) -- distance traveled down the stream in km
variables (time : ℝ := 5) -- time taken to travel the distance down the stream in hours

-- Proof statement
theorem speed_of_stream : v = 8 :=
  by
    -- effective speed down the stream = boat_speed_still_water + v
    -- given that distance = speed * time
    -- 90 = (10 + v) * 5
    -- solving for v
    sorry

end NUMINAMATH_GPT_speed_of_stream_l799_79931


namespace NUMINAMATH_GPT_x1_x2_in_M_l799_79941

-- Definitions of the set M and the condition x ∈ M
def M : Set ℕ := { x | ∃ a b : ℤ, x = a^2 + b^2 }

-- Statement of the problem
theorem x1_x2_in_M (x1 x2 : ℕ) (h1 : x1 ∈ M) (h2 : x2 ∈ M) : (x1 * x2) ∈ M :=
sorry

end NUMINAMATH_GPT_x1_x2_in_M_l799_79941


namespace NUMINAMATH_GPT_find_matrix_calculate_M5_alpha_l799_79980

-- Define the matrix M, eigenvalues, eigenvectors and vector α
def M : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 2], ![3, 2]]
def alpha : Fin 2 → ℝ := ![-1, 1]
def e1 : Fin 2 → ℝ := ![2, 3]
def e2 : Fin 2 → ℝ := ![1, -1]
def lambda1 : ℝ := 4
def lambda2 : ℝ := -1

-- Conditions: eigenvalues and their corresponding eigenvectors
axiom h1 : M.mulVec e1 = lambda1 • e1
axiom h2 : M.mulVec e2 = lambda2 • e2

-- Condition: given vector α
axiom h3 : alpha = - e2

-- Prove that M is the matrix given by the components
theorem find_matrix : M = ![![1, 2], ![3, 2]] :=
sorry

-- Prove that M^5 times α equals the given vector
theorem calculate_M5_alpha : (M^5).mulVec alpha = ![-1, 1] :=
sorry

end NUMINAMATH_GPT_find_matrix_calculate_M5_alpha_l799_79980


namespace NUMINAMATH_GPT_car_rental_budget_l799_79991

def daily_rental_cost : ℝ := 30.0
def cost_per_mile : ℝ := 0.18
def total_miles : ℝ := 250.0

theorem car_rental_budget : daily_rental_cost + (cost_per_mile * total_miles) = 75.0 :=
by 
  sorry

end NUMINAMATH_GPT_car_rental_budget_l799_79991


namespace NUMINAMATH_GPT_teams_dig_tunnel_in_10_days_l799_79989

theorem teams_dig_tunnel_in_10_days (hA : ℝ) (hB : ℝ) (work_A : hA = 15) (work_B : hB = 30) : 
  (1 / (1 / hA + 1 / hB)) = 10 := 
by
  sorry

end NUMINAMATH_GPT_teams_dig_tunnel_in_10_days_l799_79989


namespace NUMINAMATH_GPT_percent_increase_from_may_to_june_l799_79905

noncomputable def profit_increase_from_march_to_april (P : ℝ) : ℝ := 1.30 * P
noncomputable def profit_decrease_from_april_to_may (P : ℝ) : ℝ := 1.04 * P
noncomputable def profit_increase_from_march_to_june (P : ℝ) : ℝ := 1.56 * P

theorem percent_increase_from_may_to_june (P : ℝ) :
  (1.04 * P * (1 + 0.50)) = 1.56 * P :=
by
  sorry

end NUMINAMATH_GPT_percent_increase_from_may_to_june_l799_79905


namespace NUMINAMATH_GPT_john_spent_fraction_at_arcade_l799_79981

theorem john_spent_fraction_at_arcade 
  (allowance : ℝ) (spent_arcade : ℝ) (spent_candy_store : ℝ) 
  (h1 : allowance = 3.45)
  (h2 : spent_candy_store = 0.92)
  (h3 : 3.45 - spent_arcade - (1/3) * (3.45 - spent_arcade) = spent_candy_store) :
  spent_arcade / allowance = 2.07 / 3.45 :=
by
  sorry

end NUMINAMATH_GPT_john_spent_fraction_at_arcade_l799_79981


namespace NUMINAMATH_GPT_simplify_expression_l799_79968

theorem simplify_expression (b : ℝ) (h1 : b ≠ 1) (h2 : b ≠ 1 / 2) :
  (1 / 2 - 1 / (1 + b / (1 - 2 * b))) = (3 * b - 1) / (2 * (1 - b)) :=
sorry

end NUMINAMATH_GPT_simplify_expression_l799_79968


namespace NUMINAMATH_GPT_mod_residue_l799_79954

theorem mod_residue : (250 * 15 - 337 * 5 + 22) % 13 = 7 := by
  sorry

end NUMINAMATH_GPT_mod_residue_l799_79954


namespace NUMINAMATH_GPT_miranda_pillows_l799_79929

-- Define the conditions in the problem
def pounds_per_pillow := 2
def feathers_per_pound := 300
def total_feathers := 3600

-- Define the goal in terms of these conditions
def num_pillows : Nat :=
  (total_feathers / feathers_per_pound) / pounds_per_pillow

-- Prove that the number of pillows Miranda can stuff is 6
theorem miranda_pillows : num_pillows = 6 :=
by
  sorry

end NUMINAMATH_GPT_miranda_pillows_l799_79929


namespace NUMINAMATH_GPT_find_quarters_l799_79988

def num_pennies := 123
def num_nickels := 85
def num_dimes := 35
def cost_per_scoop_cents := 300  -- $3 = 300 cents
def num_family_members := 5
def leftover_cents := 48

def total_cost_cents := num_family_members * cost_per_scoop_cents
def total_initial_cents := total_cost_cents + leftover_cents

-- Values of coins in cents
def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25

def total_pennies_value := num_pennies * penny_value
def total_nickels_value := num_nickels * nickel_value
def total_dimes_value := num_dimes * dime_value
def total_initial_excluding_quarters := total_pennies_value + total_nickels_value + total_dimes_value

def total_quarters_value := total_initial_cents - total_initial_excluding_quarters
def num_quarters := total_quarters_value / quarter_value

theorem find_quarters : num_quarters = 26 := by
  sorry

end NUMINAMATH_GPT_find_quarters_l799_79988


namespace NUMINAMATH_GPT_fraction_evaluation_l799_79944

theorem fraction_evaluation :
  let p := 8579
  let q := 6960
  p.gcd q = 1 ∧ (32 / 30 - 30 / 32 + 32 / 29) = p / q :=
by
  sorry

end NUMINAMATH_GPT_fraction_evaluation_l799_79944


namespace NUMINAMATH_GPT_arithmetic_seq_sum_mul_3_l799_79998

-- Definition of the arithmetic sequence
def arithmetic_sequence := [101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121]

-- Prove that 3 times the sum of the arithmetic sequence is 3663
theorem arithmetic_seq_sum_mul_3 : 
  3 * (arithmetic_sequence.sum) = 3663 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_mul_3_l799_79998


namespace NUMINAMATH_GPT_max_x2_y2_on_circle_l799_79993

noncomputable def max_value_on_circle : ℝ :=
  12 + 8 * Real.sqrt 2

theorem max_x2_y2_on_circle (x y : ℝ) (h : x^2 - 4 * x - 4 + y^2 = 0) : 
  x^2 + y^2 ≤ max_value_on_circle := 
by
  sorry

end NUMINAMATH_GPT_max_x2_y2_on_circle_l799_79993


namespace NUMINAMATH_GPT_max_area_of_triangle_l799_79923

noncomputable def max_area_triangle (a A : ℝ) : ℝ :=
  let bcsinA := sorry
  1 / 2 * bcsinA

theorem max_area_of_triangle (a A : ℝ) (hab : a = 4) (hAa : A = Real.pi / 3) :
  max_area_triangle a A = 4 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_max_area_of_triangle_l799_79923


namespace NUMINAMATH_GPT_max_gcd_13n_plus_4_8n_plus_3_l799_79992

theorem max_gcd_13n_plus_4_8n_plus_3 (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, k = 9 ∧ gcd (13 * n + 4) (8 * n + 3) = k := 
sorry

end NUMINAMATH_GPT_max_gcd_13n_plus_4_8n_plus_3_l799_79992


namespace NUMINAMATH_GPT_liam_finishes_on_wednesday_l799_79960

theorem liam_finishes_on_wednesday :
  let start_day := 3  -- Wednesday, where 0 represents Sunday
  let total_books := 20
  let total_days := (total_books * (total_books + 1)) / 2
  (total_days % 7) = 0 :=
by sorry

end NUMINAMATH_GPT_liam_finishes_on_wednesday_l799_79960


namespace NUMINAMATH_GPT_sets_of_earrings_l799_79963

namespace EarringsProblem

variables (magnets buttons gemstones earrings : ℕ)

theorem sets_of_earrings (h1 : gemstones = 24)
                         (h2 : gemstones = 3 * buttons)
                         (h3 : buttons = magnets / 2)
                         (h4 : earrings = magnets / 2)
                         (h5 : ∀ n : ℕ, n % 2 = 0 → ∃ k, n = 2 * k) :
  earrings = 8 :=
by
  sorry

end EarringsProblem

end NUMINAMATH_GPT_sets_of_earrings_l799_79963


namespace NUMINAMATH_GPT_mother_age_twice_xiaoming_in_18_years_l799_79974

-- Definitions based on conditions
def xiaoming_age_now : ℕ := 6
def mother_age_now : ℕ := 30

theorem mother_age_twice_xiaoming_in_18_years : 
    ∀ (n : ℕ), xiaoming_age_now + n = 24 → mother_age_now + n = 2 * (xiaoming_age_now + n) → n = 18 :=
by
  intro n hn hm
  sorry

end NUMINAMATH_GPT_mother_age_twice_xiaoming_in_18_years_l799_79974


namespace NUMINAMATH_GPT_simplify_expression_l799_79930

theorem simplify_expression (x : ℝ) : 7 * x + 15 - 3 * x + 2 = 4 * x + 17 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l799_79930


namespace NUMINAMATH_GPT_fence_perimeter_l799_79986

noncomputable def posts (n : ℕ) := 36
noncomputable def space_between_posts (d : ℕ) := 6
noncomputable def length_is_twice_width (l w : ℕ) := l = 2 * w

theorem fence_perimeter (n d w l perimeter : ℕ)
  (h1 : posts n = 36)
  (h2 : space_between_posts d = 6)
  (h3 : length_is_twice_width l w)
  : perimeter = 216 :=
sorry

end NUMINAMATH_GPT_fence_perimeter_l799_79986


namespace NUMINAMATH_GPT_rate_of_increase_twice_l799_79951

theorem rate_of_increase_twice {x : ℝ} (h : (1 + x)^2 = 2) : x = (Real.sqrt 2) - 1 :=
sorry

end NUMINAMATH_GPT_rate_of_increase_twice_l799_79951


namespace NUMINAMATH_GPT_third_vertex_y_coordinate_correct_l799_79904

noncomputable def third_vertex_y_coordinate (x1 y1 x2 y2 : ℝ) (h : y1 = y2) (h_dist : |x1 - x2| = 10) : ℝ :=
  y1 + 5 * Real.sqrt 3

theorem third_vertex_y_coordinate_correct : 
  third_vertex_y_coordinate 3 4 13 4 rfl (by norm_num) = 4 + 5 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_third_vertex_y_coordinate_correct_l799_79904


namespace NUMINAMATH_GPT_decrease_in_length_l799_79977

theorem decrease_in_length (L B : ℝ) (h₀ : L ≠ 0) (h₁ : B ≠ 0)
  (h₂ : ∃ (A' : ℝ), A' = 0.72 * L * B)
  (h₃ : ∃ B' : ℝ, B' = B * 0.9) :
  ∃ (x : ℝ), x = 20 :=
by
  sorry

end NUMINAMATH_GPT_decrease_in_length_l799_79977


namespace NUMINAMATH_GPT_max_pencils_l799_79925

theorem max_pencils 
  (p : ℕ → ℝ)
  (h_price1 : ∀ n : ℕ, n ≤ 10 → p n = 0.75 * n)
  (h_price2 : ∀ n : ℕ, n > 10 → p n = 0.75 * 10 + 0.65 * (n - 10))
  (budget : ℝ) (h_budget : budget = 10) :
  ∃ n : ℕ, p n ≤ budget ∧ (∀ m : ℕ, p m ≤ budget → m ≤ 13) :=
by {
  sorry
}

end NUMINAMATH_GPT_max_pencils_l799_79925


namespace NUMINAMATH_GPT_verify_equation_l799_79934

theorem verify_equation : (3^2 + 5^2)^2 = 16^2 + 30^2 := by
  sorry

end NUMINAMATH_GPT_verify_equation_l799_79934


namespace NUMINAMATH_GPT_xy_product_l799_79939

theorem xy_product (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 :=
sorry

end NUMINAMATH_GPT_xy_product_l799_79939


namespace NUMINAMATH_GPT_g_x_squared_minus_3_l799_79955

theorem g_x_squared_minus_3 (g : ℝ → ℝ)
  (h : ∀ x : ℝ, g (x^2 - 1) = x^4 - 4 * x^2 + 4) :
  ∀ x : ℝ, g (x^2 - 3) = x^4 - 6 * x^2 + 11 :=
by
  sorry

end NUMINAMATH_GPT_g_x_squared_minus_3_l799_79955


namespace NUMINAMATH_GPT_length_BE_l799_79966

-- Definitions and Conditions
def is_square (ABCD : Type) (side_length : ℝ) : Prop :=
  side_length = 2

def triangle_area (base : ℝ) (height : ℝ) : ℝ :=
  0.5 * base * height

def rectangle_area (length : ℝ) (width : ℝ) : ℝ :=
  length * width

-- Problem statement in Lean
theorem length_BE 
(ABCD : Type) (side_length : ℝ) 
(JKHG : Type) (BC : ℝ) (x : ℝ) 
(E : Type) (E_on_BC : E) 
(area_fact : rectangle_area BC x = 2 * triangle_area x BC) 
(h1 : is_square ABCD side_length) 
(h2 : BC = 2) : 
x = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_length_BE_l799_79966


namespace NUMINAMATH_GPT_problem_statement_l799_79937

noncomputable def g (x : ℝ) : ℝ := 3^(x + 1)

theorem problem_statement (x : ℝ) : g (x + 1) - 2 * g x = g x := by
  -- The proof here is omitted
  sorry

end NUMINAMATH_GPT_problem_statement_l799_79937


namespace NUMINAMATH_GPT_P_eq_CU_M_union_CU_N_l799_79928

open Set

-- Definitions of U, M, N
def U : Set (ℝ × ℝ) := { p | True }
def M : Set (ℝ × ℝ) := { p | p.2 ≠ p.1 }
def N : Set (ℝ × ℝ) := { p | p.2 ≠ -p.1 }
def CU_M : Set (ℝ × ℝ) := { p | p.2 = p.1 }
def CU_N : Set (ℝ × ℝ) := { p | p.2 = -p.1 }

-- Theorem statement
theorem P_eq_CU_M_union_CU_N :
  { p : ℝ × ℝ | p.2^2 ≠ p.1^2 } = CU_M ∪ CU_N :=
sorry

end NUMINAMATH_GPT_P_eq_CU_M_union_CU_N_l799_79928


namespace NUMINAMATH_GPT_spencer_total_distance_l799_79949

def d1 : ℝ := 1.2
def d2 : ℝ := 0.6
def d3 : ℝ := 0.9
def d4 : ℝ := 1.7
def d5 : ℝ := 2.1
def d6 : ℝ := 1.3
def d7 : ℝ := 0.8

theorem spencer_total_distance : d1 + d2 + d3 + d4 + d5 + d6 + d7 = 8.6 :=
by
  sorry

end NUMINAMATH_GPT_spencer_total_distance_l799_79949


namespace NUMINAMATH_GPT_factorial_div_l799_79945

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_div : (factorial 4) / (factorial (4 - 3)) = 24 := by
  sorry

end NUMINAMATH_GPT_factorial_div_l799_79945


namespace NUMINAMATH_GPT_egg_weight_probability_l799_79920

theorem egg_weight_probability : 
  let P_lt_30 := 0.3
  let P_30_40 := 0.5
  P_lt_30 + P_30_40 ≤ 1 → (1 - (P_lt_30 + P_30_40) = 0.2) := by
  intro h
  sorry

end NUMINAMATH_GPT_egg_weight_probability_l799_79920


namespace NUMINAMATH_GPT_solve_quadratic_l799_79946

theorem solve_quadratic (y : ℝ) :
  y^2 - 3 * y - 10 = -(y + 2) * (y + 6) ↔ (y = -1/2 ∨ y = -2) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l799_79946


namespace NUMINAMATH_GPT_add_two_inequality_l799_79947

theorem add_two_inequality (a b : ℝ) (h : a > b) : a + 2 > b + 2 :=
sorry

end NUMINAMATH_GPT_add_two_inequality_l799_79947


namespace NUMINAMATH_GPT_prime_dvd_square_l799_79913

theorem prime_dvd_square (p n : ℕ) (hp : Nat.Prime p) (h : p ∣ n^2) : p ∣ n :=
  sorry

end NUMINAMATH_GPT_prime_dvd_square_l799_79913


namespace NUMINAMATH_GPT_product_of_a_values_has_three_solutions_eq_20_l799_79903

noncomputable def f (x : ℝ) : ℝ := abs ((x^2 - 10 * x + 25) / (x - 5) - (x^2 - 3 * x) / (3 - x))

def has_three_solutions (a : ℝ) : Prop :=
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ abs (abs (f x1) - 5) = a ∧ abs (abs (f x2) - 5) = a ∧ abs (abs (f x3) - 5) = a)

theorem product_of_a_values_has_three_solutions_eq_20 :
  ∃ a1 a2 : ℝ, has_three_solutions a1 ∧ has_three_solutions a2 ∧ a1 * a2 = 20 :=
sorry

end NUMINAMATH_GPT_product_of_a_values_has_three_solutions_eq_20_l799_79903


namespace NUMINAMATH_GPT_part1_part2_part3_l799_79983

noncomputable def f : ℝ → ℝ := sorry -- Given f is a function on ℝ with domain (0, +∞)

axiom domain_pos (x : ℝ) : 0 < x
axiom pos_condition (x : ℝ) (h : 1 < x) : 0 < f x
axiom functional_eq (x y : ℝ) : f (x * y) = f x + f y
axiom specific_value : f (1/3) = -1

-- (1) Prove: f(1/x) = -f(x)
theorem part1 (x : ℝ) (hx : 0 < x) : f (1 / x) = - f x := sorry

-- (2) Prove: f(x) is an increasing function on its domain
theorem part2 (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (h : x1 < x2) : f x1 < f x2 := sorry

-- (3) Prove the range of x for the inequality
theorem part3 (x : ℝ) (hx : 0 < x) (hx2 : 0 < x - 2) : 
  f x - f (1 / (x - 2)) ≥ 2 ↔ 1 + Real.sqrt 10 ≤ x := sorry

end NUMINAMATH_GPT_part1_part2_part3_l799_79983


namespace NUMINAMATH_GPT_problem1_problem2_l799_79901

-- Definition and conditions
def i := Complex.I

-- Problem 1
theorem problem1 : (2 + 2 * i) / (1 - i)^2 + (Real.sqrt 2 / (1 + i)) ^ 2010 = -1 := 
by
  sorry

-- Problem 2
theorem problem2 : (4 - i^5) * (6 + 2 * i^7) + (7 + i^11) * (4 - 3 * i) = 47 - 39 * i := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l799_79901


namespace NUMINAMATH_GPT_determine_weights_l799_79909

-- Definitions
variable {W : Type} [AddCommGroup W] [OrderedAddCommMonoid W]
variable (w : Fin 20 → W) -- List of weights for 20 people
variable (s : W) -- Total sum of weights
variable (lower upper : W) -- Lower and upper weight limits

-- Conditions
def weight_constraints : Prop :=
  (∀ i, lower ≤ w i ∧ w i ≤ upper) ∧ (Finset.univ.sum w = s)

-- Problem statement
theorem determine_weights (w : Fin 20 → ℝ) :
  weight_constraints w 60 90 3040 →
  ∃ w : Fin 20 → ℝ, weight_constraints w 60 90 3040 := by
  sorry

end NUMINAMATH_GPT_determine_weights_l799_79909


namespace NUMINAMATH_GPT_smallest_n_for_inequality_l799_79978

theorem smallest_n_for_inequality (n : ℕ) : 5 + 3 * n > 300 ↔ n = 99 := by
  sorry

end NUMINAMATH_GPT_smallest_n_for_inequality_l799_79978


namespace NUMINAMATH_GPT_prime_iff_good_fractions_l799_79969

def isGoodFraction (n : ℕ) (a b : ℕ) : Prop := a > 0 ∧ b > 0 ∧ (a + b = n)

def canBeExpressedUsingGoodFractions (n : ℕ) (a b : ℕ) : Prop :=
  ∃ (expressedFraction : ℕ → ℕ → Prop), expressedFraction a b ∧
  ∀ x y, expressedFraction x y → isGoodFraction n x y

theorem prime_iff_good_fractions {n : ℕ} (hn : n > 1) :
  Prime n ↔
    ∀ a b : ℕ, b < n → (a > 0 ∧ b > 0) → canBeExpressedUsingGoodFractions n a b :=
sorry

end NUMINAMATH_GPT_prime_iff_good_fractions_l799_79969


namespace NUMINAMATH_GPT_simplify_expression_l799_79967

theorem simplify_expression (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) (h4 : x ≠ 5):
  ((x^2 - 4 * x + 3) / (x^2 - 6 * x + 9)) / ((x^2 - 6 * x + 8) / (x^2 - 8 * x + 15)) = 
  (x - 1) * (x - 5) / ((x - 3) * (x - 4) * (x - 2)) :=
sorry

end NUMINAMATH_GPT_simplify_expression_l799_79967


namespace NUMINAMATH_GPT_paper_holes_symmetric_l799_79956

-- Define the initial conditions
def folded_paper : Type := sorry -- Specific structure to represent the paper and its folds

def paper_fold_bottom_to_top (paper : folded_paper) : folded_paper := sorry
def paper_fold_right_half_to_left (paper : folded_paper) : folded_paper := sorry
def paper_fold_diagonal (paper : folded_paper) : folded_paper := sorry

-- Define a function that represents punching a hole near the folded edge
def punch_hole_near_folded_edge (paper : folded_paper) : folded_paper := sorry

-- Initial paper
def initial_paper : folded_paper := sorry

-- Folded and punched paper
def paper_after_folds_and_punch : folded_paper :=
  punch_hole_near_folded_edge (
    paper_fold_diagonal (
      paper_fold_right_half_to_left (
        paper_fold_bottom_to_top initial_paper)))

-- Unfolding the paper
def unfold_diagonal (paper : folded_paper) : folded_paper := sorry
def unfold_right_half (paper : folded_paper) : folded_paper := sorry
def unfold_bottom_to_top (paper : folded_paper) : folded_paper := sorry

def paper_after_unfolding : folded_paper :=
  unfold_bottom_to_top (
    unfold_right_half (
      unfold_diagonal paper_after_folds_and_punch))

-- Definition of hole pattern 'eight_symmetric_holes'
def eight_symmetric_holes (paper : folded_paper) : Prop := sorry

-- The proof problem
theorem paper_holes_symmetric :
  eight_symmetric_holes paper_after_unfolding := sorry

end NUMINAMATH_GPT_paper_holes_symmetric_l799_79956


namespace NUMINAMATH_GPT_least_pos_int_x_l799_79911

theorem least_pos_int_x (x : ℕ) (h1 : ∃ k : ℤ, (3 * x + 43) = 53 * k) 
  : x = 21 :=
sorry

end NUMINAMATH_GPT_least_pos_int_x_l799_79911


namespace NUMINAMATH_GPT_exercise_l799_79940

theorem exercise (a b : ℕ) (h1 : 656 = 3 * 7^2 + a * 7 + b) (h2 : 656 = 3 * 10^2 + a * 10 + b) : 
  (a * b) / 15 = 1 :=
by
  sorry

end NUMINAMATH_GPT_exercise_l799_79940


namespace NUMINAMATH_GPT_jessie_final_position_l799_79953

theorem jessie_final_position :
  ∃ y : ℕ,
  (0 + 6 * 4 = 24) ∧
  (y = 24) :=
by
  sorry

end NUMINAMATH_GPT_jessie_final_position_l799_79953


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l799_79962

noncomputable def a_n (a1 d : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d
noncomputable def S_n (a1 d : ℕ) (n : ℕ) : ℕ := n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_sum (a1 d : ℕ) 
  (h1 : a1 + d = 6) 
  (h2 : (a1 + 2 * d)^2 = a1 * (a1 + 6 * d)) 
  (h3 : d ≠ 0) : 
  S_n a1 d 8 = 88 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l799_79962


namespace NUMINAMATH_GPT_henry_time_proof_l799_79936

-- Define the time Dawson took to run the first leg of the course
def dawson_time : ℝ := 38

-- Define the average time they took to run a leg of the course
def average_time : ℝ := 22.5

-- Define the time Henry took to run the second leg of the course
def henry_time : ℝ := 7

-- Prove that Henry took 7 seconds to run the second leg
theorem henry_time_proof : 
  (dawson_time + henry_time) / 2 = average_time :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_henry_time_proof_l799_79936


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l799_79999

theorem quadratic_inequality_solution {x : ℝ} :
  (x^2 + x - 6 ≤ 0) ↔ (-3 ≤ x ∧ x ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l799_79999


namespace NUMINAMATH_GPT_black_area_after_transformations_l799_79957

theorem black_area_after_transformations :
  let initial_fraction : ℝ := 1
  let transformation_factor : ℝ := 3 / 4
  let number_of_transformations : ℕ := 5
  let final_fraction : ℝ := transformation_factor ^ number_of_transformations
  final_fraction = 243 / 1024 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_black_area_after_transformations_l799_79957


namespace NUMINAMATH_GPT_ab_value_l799_79950

theorem ab_value (a b : ℚ) 
  (h1 : (a + b) ^ 2 + |b + 5| = b + 5) 
  (h2 : 2 * a - b + 1 = 0) : 
  a * b = -1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_ab_value_l799_79950


namespace NUMINAMATH_GPT_find_blue_chips_l799_79906

def num_chips_satisfies (n m : ℕ) : Prop :=
  (n > m) ∧ (n + m > 2) ∧ (n + m < 50) ∧
  (n * (n - 1) + m * (m - 1)) = 2 * n * m

theorem find_blue_chips (n : ℕ) :
  (∃ m : ℕ, num_chips_satisfies n m) → 
  n = 3 ∨ n = 6 ∨ n = 10 ∨ n = 15 ∨ n = 21 ∨ n = 28 :=
by
  sorry

end NUMINAMATH_GPT_find_blue_chips_l799_79906


namespace NUMINAMATH_GPT_perfect_squares_unique_l799_79952

theorem perfect_squares_unique (n : ℕ) (h1 : ∃ k : ℕ, 20 * n = k^2) (h2 : ∃ p : ℕ, 5 * n + 275 = p^2) :
  n = 125 :=
by
  sorry

end NUMINAMATH_GPT_perfect_squares_unique_l799_79952


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_range_l799_79996

theorem quadratic_distinct_real_roots_range (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 2 * x - 1 = 0 ∧ ∃ y : ℝ, y ≠ x ∧ k * y^2 - 2 * y - 1 = 0) ↔ (k > -1 ∧ k ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_range_l799_79996


namespace NUMINAMATH_GPT_pyramid_volume_l799_79943

theorem pyramid_volume
  (FB AC FA FC AB BC : ℝ)
  (hFB : FB = 12)
  (hAC : AC = 4)
  (hFA : FA = 7)
  (hFC : FC = 7)
  (hAB : AB = 7)
  (hBC : BC = 7) :
  (1/3 * AC * (1/2 * FB * 3)) = 24 := by sorry

end NUMINAMATH_GPT_pyramid_volume_l799_79943


namespace NUMINAMATH_GPT_minimum_n_for_obtuse_triangle_l799_79914

def α₀ : ℝ := 60 
def β₀ : ℝ := 59.999
def γ₀ : ℝ := 60.001

def α (n : ℕ) : ℝ := (-2)^n * (α₀ - 60) + 60
def β (n : ℕ) : ℝ := (-2)^n * (β₀ - 60) + 60
def γ (n : ℕ) : ℝ := (-2)^n * (γ₀ - 60) + 60

theorem minimum_n_for_obtuse_triangle : ∃ n : ℕ, β n > 90 ∧ ∀ m : ℕ, m < n → β m ≤ 90 :=
by sorry

end NUMINAMATH_GPT_minimum_n_for_obtuse_triangle_l799_79914


namespace NUMINAMATH_GPT_solve_abs_quadratic_l799_79932

theorem solve_abs_quadratic :
  ∀ x : ℝ, abs (x^2 - 4 * x + 4) = 3 - x ↔ (x = (3 + Real.sqrt 5) / 2 ∨ x = (3 - Real.sqrt 5) / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_abs_quadratic_l799_79932


namespace NUMINAMATH_GPT_find_constant_term_l799_79979

theorem find_constant_term (x y C : ℤ) 
    (h1 : 5 * x + y = 19) 
    (h2 : 3 * x + 2 * y = 10) 
    (h3 : C = x + 3 * y) 
    : C = 1 := 
by 
  sorry

end NUMINAMATH_GPT_find_constant_term_l799_79979


namespace NUMINAMATH_GPT_candy_lasts_for_days_l799_79995

-- Definitions based on conditions
def candy_from_neighbors : ℕ := 75
def candy_from_sister : ℕ := 130
def candy_traded : ℕ := 25
def candy_lost : ℕ := 15
def candy_eaten_per_day : ℕ := 7

-- Total candy calculation
def total_candy : ℕ := candy_from_neighbors + candy_from_sister - candy_traded - candy_lost
def days_candy_lasts : ℕ := total_candy / candy_eaten_per_day

-- Proof statement
theorem candy_lasts_for_days : days_candy_lasts = 23 := by
  -- sorry is used to skip the actual proof
  sorry

end NUMINAMATH_GPT_candy_lasts_for_days_l799_79995


namespace NUMINAMATH_GPT_no_triangle_with_heights_1_2_3_l799_79972

open Real

theorem no_triangle_with_heights_1_2_3 :
  ¬(∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
     ∃ (k : ℝ), k > 0 ∧ 
       a * k = 1 ∧ b * (k / 2) = 2 ∧ c * (k / 3) = 3 ∧
       (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
by 
  sorry

end NUMINAMATH_GPT_no_triangle_with_heights_1_2_3_l799_79972


namespace NUMINAMATH_GPT_value_of_expression_l799_79922

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : a^2 - b^2 + 6 * b = 9 :=
  sorry

end NUMINAMATH_GPT_value_of_expression_l799_79922


namespace NUMINAMATH_GPT_divisor_of_number_l799_79926

theorem divisor_of_number (n d q p : ℤ) 
  (h₁ : n = d * q + 3)
  (h₂ : n ^ 2 = d * p + 3) : 
  d = 6 := 
sorry

end NUMINAMATH_GPT_divisor_of_number_l799_79926


namespace NUMINAMATH_GPT_balloon_ratio_l799_79908

/-- Janice has 6 water balloons. --/
def Janice_balloons : Nat := 6

/-- Randy has half as many water balloons as Janice. --/
def Randy_balloons : Nat := Janice_balloons / 2

/-- Cynthia has 12 water balloons. --/
def Cynthia_balloons : Nat := 12

/-- The ratio of Cynthia's water balloons to Randy's water balloons is 4:1. --/
theorem balloon_ratio : Cynthia_balloons / Randy_balloons = 4 := by
  sorry

end NUMINAMATH_GPT_balloon_ratio_l799_79908


namespace NUMINAMATH_GPT_min_k_inequality_l799_79982

theorem min_k_inequality (α β : ℝ) (hα : 0 < α) (hα2 : α < 2 * Real.pi / 3)
  (hβ : 0 < β) (hβ2 : β < 2 * Real.pi / 3) :
  4 * Real.cos α ^ 2 + 2 * Real.cos α * Real.cos β + 4 * Real.cos β ^ 2
  - 3 * Real.cos α - 3 * Real.cos β - 6 < 0 :=
by
  sorry

end NUMINAMATH_GPT_min_k_inequality_l799_79982


namespace NUMINAMATH_GPT_original_price_of_cycle_l799_79958

theorem original_price_of_cycle (SP : ℝ) (gain_percent : ℝ) (original_price : ℝ) 
  (hSP : SP = 1260) (hgain : gain_percent = 0.40) (h_eq : SP = original_price * (1 + gain_percent)) :
  original_price = 900 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_cycle_l799_79958


namespace NUMINAMATH_GPT_length_of_square_side_l799_79990

-- Definitions based on conditions
def perimeter_of_triangle : ℝ := 46
def total_perimeter : ℝ := 78
def perimeter_of_square : ℝ := total_perimeter - perimeter_of_triangle

-- Lean statement for the problem
theorem length_of_square_side : perimeter_of_square / 4 = 8 := by
  sorry

end NUMINAMATH_GPT_length_of_square_side_l799_79990


namespace NUMINAMATH_GPT_radius_of_circle_l799_79971

theorem radius_of_circle:
  (∃ (r: ℝ), 
    (∀ (x: ℝ), (x^2 + r - x) = 0 → 1 - 4 * r = 0)
  ) → r = 1 / 4 := 
sorry

end NUMINAMATH_GPT_radius_of_circle_l799_79971


namespace NUMINAMATH_GPT_mints_ratio_l799_79912

theorem mints_ratio (n : ℕ) (green_mints red_mints : ℕ) (h1 : green_mints + red_mints = n) (h2 : green_mints = 3 * (n / 4)) : green_mints / red_mints = 3 :=
by
  sorry

end NUMINAMATH_GPT_mints_ratio_l799_79912


namespace NUMINAMATH_GPT_no_form3000001_is_perfect_square_l799_79942

theorem no_form3000001_is_perfect_square (n : ℕ) : 
  ∀ k : ℤ, (3 * 10^n + 1 ≠ k^2) :=
by
  sorry

end NUMINAMATH_GPT_no_form3000001_is_perfect_square_l799_79942


namespace NUMINAMATH_GPT_exists_xy_interval_l799_79959

theorem exists_xy_interval (a b : ℝ) : 
  ∃ (x y : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ |x * y - a * x - b * y| ≥ 1 / 3 :=
sorry

end NUMINAMATH_GPT_exists_xy_interval_l799_79959


namespace NUMINAMATH_GPT_product_of_roots_l799_79973

noncomputable def quadratic_equation (x : ℝ) : Prop :=
  (x + 4) * (x - 5) = 22

theorem product_of_roots :
  ∀ x1 x2 : ℝ, quadratic_equation x1 → quadratic_equation x2 → (x1 * x2 = -42) := 
by
  sorry

end NUMINAMATH_GPT_product_of_roots_l799_79973


namespace NUMINAMATH_GPT_jake_spent_more_l799_79964

def cost_of_balloons (helium_count : ℕ) (foil_count : ℕ) (helium_price : ℝ) (foil_price : ℝ) : ℝ :=
  helium_count * helium_price + foil_count * foil_price

theorem jake_spent_more 
  (allan_helium : ℕ) (allan_foil : ℕ) (jake_helium : ℕ) (jake_foil : ℕ)
  (helium_price : ℝ) (foil_price : ℝ)
  (h_allan_helium : allan_helium = 2) (h_allan_foil : allan_foil = 3) 
  (h_jake_helium : jake_helium = 4) (h_jake_foil : jake_foil = 2)
  (h_helium_price : helium_price = 1.5) (h_foil_price : foil_price = 2.5) :
  cost_of_balloons jake_helium jake_foil helium_price foil_price - 
  cost_of_balloons allan_helium allan_foil helium_price foil_price = 0.5 := 
by
  sorry

end NUMINAMATH_GPT_jake_spent_more_l799_79964


namespace NUMINAMATH_GPT_multiple_of_3804_l799_79921

theorem multiple_of_3804 (n : ℕ) (hn : 0 < n) : 
  ∃ k : ℕ, (n^3 - n) * (5^(8*n+4) + 3^(4*n+2)) = k * 3804 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_3804_l799_79921


namespace NUMINAMATH_GPT_range_of_m_l799_79924

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : x > 0) (hy : y > 0) (hxy : (1/x) + (4/y) = 1) :
  (x + y > m^2 + 8 * m) → (-9 < m ∧ m < 1) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_m_l799_79924


namespace NUMINAMATH_GPT_fabian_total_cost_l799_79935

def mouse_cost : ℕ := 20

def keyboard_cost : ℕ := 2 * mouse_cost

def headphones_cost : ℕ := mouse_cost + 15

def usb_hub_cost : ℕ := 36 - mouse_cost

def total_cost : ℕ := mouse_cost + keyboard_cost + headphones_cost + usb_hub_cost

theorem fabian_total_cost : total_cost = 111 := 
by 
  unfold total_cost mouse_cost keyboard_cost headphones_cost usb_hub_cost
  sorry

end NUMINAMATH_GPT_fabian_total_cost_l799_79935


namespace NUMINAMATH_GPT_problem_1_l799_79987

theorem problem_1 :
  (5 / ((1 / (1 * 2)) + (1 / (2 * 3)) + (1 / (3 * 4)) + (1 / (4 * 5)) + (1 / (5 * 6)))) = 6 := by
  sorry

end NUMINAMATH_GPT_problem_1_l799_79987


namespace NUMINAMATH_GPT_equation_no_solution_B_l799_79916

theorem equation_no_solution_B :
  ¬(∃ x : ℝ, |-3 * x| + 5 = 0) :=
sorry

end NUMINAMATH_GPT_equation_no_solution_B_l799_79916


namespace NUMINAMATH_GPT_peter_class_students_l799_79994

def total_students (students_with_two_hands students_with_one_hand students_with_three_hands : ℕ) : ℕ :=
  students_with_two_hands + students_with_one_hand + students_with_three_hands + 1

theorem peter_class_students
  (students_with_two_hands students_with_one_hand students_with_three_hands : ℕ)
  (total_hands_without_peter : ℕ) :

  students_with_two_hands = 10 →
  students_with_one_hand = 3 →
  students_with_three_hands = 1 →
  total_hands_without_peter = 20 →
  total_students students_with_two_hands students_with_one_hand students_with_three_hands = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_peter_class_students_l799_79994


namespace NUMINAMATH_GPT_son_daughter_eggs_per_morning_l799_79961

-- Define the given conditions in Lean 4
def trays_per_week : Nat := 2
def eggs_per_tray : Nat := 24
def eggs_per_night_rhea_husband : Nat := 4
def nights_per_week : Nat := 7
def uneaten_eggs_per_week : Nat := 6

-- Define the total eggs bought per week
def total_eggs_per_week : Nat := trays_per_week * eggs_per_tray

-- Define the eggs eaten per week by Rhea and her husband
def eggs_eaten_per_week_rhea_husband : Nat := eggs_per_night_rhea_husband * nights_per_week

-- Prove the number of eggs eaten by son and daughter every morning
theorem son_daughter_eggs_per_morning :
  (total_eggs_per_week - eggs_eaten_per_week_rhea_husband - uneaten_eggs_per_week) = 14 :=
sorry

end NUMINAMATH_GPT_son_daughter_eggs_per_morning_l799_79961


namespace NUMINAMATH_GPT_range_of_a_l799_79910

noncomputable def f (x a : ℝ) : ℝ := x^3 - a*x^2 + 4

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) → 3 < a :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l799_79910


namespace NUMINAMATH_GPT_cricket_runs_l799_79933

theorem cricket_runs (x a b c d : ℕ) 
    (h1 : a = 1 * x) 
    (h2 : b = 3 * x) 
    (h3 : c = 5 * x) 
    (h4 : d = 4 * x) 
    (total_runs : 1 * x + 3 * x + 5 * x + 4 * x = 234) :
  a = 18 ∧ b = 54 ∧ c = 90 ∧ d = 72 := by
  sorry

end NUMINAMATH_GPT_cricket_runs_l799_79933


namespace NUMINAMATH_GPT_expression_value_as_fraction_l799_79907

theorem expression_value_as_fraction :
  2 + (3 / (2 + (5 / (4 + (7 / 3))))) = 91 / 19 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_as_fraction_l799_79907


namespace NUMINAMATH_GPT_problem_126_times_3_pow_6_l799_79948

theorem problem_126_times_3_pow_6 (p : ℝ) (h : 126 * 3^8 = p) : 
  126 * 3^6 = (1 / 9) * p := 
by {
  -- Placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_problem_126_times_3_pow_6_l799_79948


namespace NUMINAMATH_GPT_find_price_of_turban_l799_79970

-- Define the main variables and conditions
def price_of_turban (T : ℝ) : Prop :=
  ((3 / 4) * 90 + T = 60 + T) → T = 30

-- State the theorem with the given conditions and aim to find T
theorem find_price_of_turban (T : ℝ) (h1 : 90 + T = 120) :  price_of_turban T :=
by
  intros
  sorry


end NUMINAMATH_GPT_find_price_of_turban_l799_79970


namespace NUMINAMATH_GPT_find_b_for_continuity_at_2_l799_79902

noncomputable def f (x : ℝ) (b : ℝ) :=
if x ≤ 2 then 3 * x^2 + 1 else b * x - 6

theorem find_b_for_continuity_at_2
  (b : ℝ) 
  (h_cont : ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) :
  b = 19 / 2 := by sorry

end NUMINAMATH_GPT_find_b_for_continuity_at_2_l799_79902


namespace NUMINAMATH_GPT_find_S3m_l799_79985
  
-- Arithmetic sequence with given properties
variable (m : ℕ)
variable (S : ℕ → ℕ)
variable (a : ℕ → ℕ)

-- Define the conditions
axiom Sm : S m = 30
axiom S2m : S (2 * m) = 100

-- Problem statement to prove
theorem find_S3m : S (3 * m) = 170 :=
by
  sorry

end NUMINAMATH_GPT_find_S3m_l799_79985


namespace NUMINAMATH_GPT_min_sum_abc_l799_79919

theorem min_sum_abc (a b c : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a * b * c + b * c + c = 2014) : a + b + c = 40 :=
sorry

end NUMINAMATH_GPT_min_sum_abc_l799_79919


namespace NUMINAMATH_GPT_sarahs_brother_apples_l799_79927

theorem sarahs_brother_apples (x : ℝ) (hx : 5 * x = 45.0) : x = 9.0 :=
by
  sorry

end NUMINAMATH_GPT_sarahs_brother_apples_l799_79927


namespace NUMINAMATH_GPT_remainder_7_times_10_pow_20_plus_1_pow_20_mod_9_l799_79984

theorem remainder_7_times_10_pow_20_plus_1_pow_20_mod_9 :
  (7 * 10 ^ 20 + 1 ^ 20) % 9 = 8 :=
by
  -- need to note down the known conditions to help guide proof writing.
  -- condition: 1 ^ 20 = 1
  -- condition: 10 % 9 = 1

  sorry

end NUMINAMATH_GPT_remainder_7_times_10_pow_20_plus_1_pow_20_mod_9_l799_79984


namespace NUMINAMATH_GPT_last_two_digits_of_floor_l799_79917

def last_two_digits (n : Nat) : Nat :=
  n % 100

theorem last_two_digits_of_floor :
  let x := 10^93
  let y := 10^31
  last_two_digits (Nat.floor (x / (y + 3))) = 8 :=
by
  sorry

end NUMINAMATH_GPT_last_two_digits_of_floor_l799_79917


namespace NUMINAMATH_GPT_correct_option_c_l799_79900

theorem correct_option_c (a : ℝ) : (-2 * a) ^ 3 = -8 * a ^ 3 :=
sorry

end NUMINAMATH_GPT_correct_option_c_l799_79900


namespace NUMINAMATH_GPT_restaurant_total_cost_l799_79938

theorem restaurant_total_cost :
  let vegetarian_cost := 5
  let chicken_cost := 7
  let steak_cost := 10
  let kids_cost := 3
  let tax_rate := 0.10
  let tip_rate := 0.15
  let num_vegetarians := 3
  let num_chicken_lovers := 4
  let num_steak_enthusiasts := 2
  let num_kids_hot_dog := 3
  let subtotal := (num_vegetarians * vegetarian_cost) + (num_chicken_lovers * chicken_cost) + (num_steak_enthusiasts * steak_cost) + (num_kids_hot_dog * kids_cost)
  let tax := subtotal * tax_rate
  let tip := subtotal * tip_rate
  let total_cost := subtotal + tax + tip
  total_cost = 90 :=
by sorry

end NUMINAMATH_GPT_restaurant_total_cost_l799_79938


namespace NUMINAMATH_GPT_no_solution_for_equation_l799_79918

theorem no_solution_for_equation :
  ¬ ∃ x : ℝ, (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_equation_l799_79918


namespace NUMINAMATH_GPT_prime_fraction_identity_l799_79976

theorem prime_fraction_identity : ∀ (p q : ℕ),
  Prime p → Prime q → p = 2 → q = 2 →
  (pq + p^p + q^q) / (p + q) = 3 :=
by
  intros p q hp hq hp2 hq2
  sorry

end NUMINAMATH_GPT_prime_fraction_identity_l799_79976


namespace NUMINAMATH_GPT_correct_operation_l799_79915

variables (a b : ℝ)

theorem correct_operation : 5 * a * b - 3 * a * b = 2 * a * b :=
by sorry

end NUMINAMATH_GPT_correct_operation_l799_79915
