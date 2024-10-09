import Mathlib

namespace ratio_of_cream_l1009_100958

theorem ratio_of_cream (coffee_init : ℕ) (joe_coffee_drunk : ℕ) (cream_added : ℕ) (joann_total_drunk : ℕ) 
  (joann_coffee_init : ℕ := coffee_init)
  (joe_coffee_init : ℕ := coffee_init) (joann_cream_init : ℕ := cream_added)
  (joe_cream_init : ℕ := cream_added)
  (joann_drunk_cream_ratio : ℚ := joann_cream_init / (joann_coffee_init + joann_cream_init)) :
  (joe_cream_init / (joann_cream_init - joann_total_drunk * (joann_drunk_cream_ratio))) = 
  (6 / 5) := 
by
  sorry

end ratio_of_cream_l1009_100958


namespace john_cuts_his_grass_to_l1009_100957

theorem john_cuts_his_grass_to (growth_rate monthly_cost annual_cost cut_height : ℝ)
  (h : ℝ) : 
  growth_rate = 0.5 ∧ monthly_cost = 100 ∧ annual_cost = 300 ∧ cut_height = 4 →
  h = 2 := by
  intros conditions
  sorry

end john_cuts_his_grass_to_l1009_100957


namespace smallest_positive_integer_l1009_100960

-- Definitions of the conditions
def condition1 (k : ℕ) : Prop := k % 10 = 9
def condition2 (k : ℕ) : Prop := k % 9 = 8
def condition3 (k : ℕ) : Prop := k % 8 = 7
def condition4 (k : ℕ) : Prop := k % 7 = 6
def condition5 (k : ℕ) : Prop := k % 6 = 5
def condition6 (k : ℕ) : Prop := k % 5 = 4
def condition7 (k : ℕ) : Prop := k % 4 = 3
def condition8 (k : ℕ) : Prop := k % 3 = 2
def condition9 (k : ℕ) : Prop := k % 2 = 1

-- Statement of the problem
theorem smallest_positive_integer : ∃ k : ℕ, 
  k > 0 ∧
  condition1 k ∧ 
  condition2 k ∧ 
  condition3 k ∧ 
  condition4 k ∧ 
  condition5 k ∧ 
  condition6 k ∧ 
  condition7 k ∧ 
  condition8 k ∧ 
  condition9 k ∧
  k = 2519 := 
sorry

end smallest_positive_integer_l1009_100960


namespace triangle_inequality_inequality_equality_condition_l1009_100978

variable (a b c : ℝ)

-- indicating triangle inequality conditions
variable (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)

theorem triangle_inequality_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a^2*b*(a - b) + b^2*c*(b - c) + c^2*a*(c - a) ≥ 0 :=
sorry

theorem equality_condition (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a^2*b*(a - b) + b^2*c*(b - c) + c^2*a*(c - a) = 0 ↔ a = b ∧ b = c :=
sorry

end triangle_inequality_inequality_equality_condition_l1009_100978


namespace arithmetic_sequence_sum_l1009_100979

theorem arithmetic_sequence_sum (a d x y : ℤ) 
  (h1 : a = 3) (h2 : d = 5) 
  (h3 : x = a + d) 
  (h4 : y = x + d) 
  (h5 : y = 18) 
  (h6 : x = 13) : x + y = 31 := by
  sorry

end arithmetic_sequence_sum_l1009_100979


namespace volume_ratio_octahedron_cube_l1009_100954

theorem volume_ratio_octahedron_cube 
  (s : ℝ) -- edge length of the octahedron
  (h := s * Real.sqrt 2 / 2) -- height of one of the pyramids forming the octahedron
  (volume_O := s^3 * Real.sqrt 2 / 3) -- volume of the octahedron
  (a := (2 * s) / Real.sqrt 3) -- edge length of the cube
  (volume_C := (a ^ 3)) -- volume of the cube
  (diag_C : ℝ := 2 * s) -- diagonal of the cube
  (h_diag : diag_C = (a * Real.sqrt 3)) -- relation of diagonal to edge length of the cube
  (ratio := volume_O / volume_C) -- ratio of the volumes
  (desired_ratio := 3 / 8) -- given ratio in simplified form
  (m := 3) -- first part of the ratio
  (n := 8) -- second part of the ratio
  (rel_prime : Nat.gcd m n = 1) -- m and n are relatively prime
  (correct_ratio : ratio = desired_ratio) -- the ratio is correct
  : m + n = 11 :=
by
  sorry 

end volume_ratio_octahedron_cube_l1009_100954


namespace max_candies_l1009_100987

theorem max_candies (V M S : ℕ) (hv : V = 35) (hm : 1 ≤ M ∧ M < 35) (hs : S = 35 + M) (heven : Even S) : V + M + S = 136 :=
sorry

end max_candies_l1009_100987


namespace multiple_of_6_is_multiple_of_3_l1009_100955

theorem multiple_of_6_is_multiple_of_3 (n : ℕ) : (∃ k : ℕ, n = 6 * k) → (∃ m : ℕ, n = 3 * m) :=
by
  sorry

end multiple_of_6_is_multiple_of_3_l1009_100955


namespace first_runner_meets_conditions_l1009_100924

noncomputable def first_runner_time := 11

theorem first_runner_meets_conditions (T : ℕ) (second_runner_time third_runner_time : ℕ) (meet_time : ℕ)
  (h1 : second_runner_time = 4)
  (h2 : third_runner_time = 11 / 2)
  (h3 : meet_time = 44)
  (h4 : meet_time % T = 0)
  (h5 : meet_time % second_runner_time = 0)
  (h6 : meet_time % third_runner_time = 0) : 
  T = first_runner_time :=
by
  sorry

end first_runner_meets_conditions_l1009_100924


namespace rectangle_square_division_l1009_100900

theorem rectangle_square_division (n : ℕ) 
  (a b c d : ℕ) 
  (h1 : a * b = n) 
  (h2 : c * d = n + 76)
  (h3 : ∃ u v : ℕ, gcd a c = u ∧ gcd b d = v ∧ u * v * a^2 = u * v * c^2 ∧ u * v * b^2 = u * v * d^2) : 
  n = 324 := sorry

end rectangle_square_division_l1009_100900


namespace intersection_of_A_and_B_is_5_and_8_l1009_100971

def A : Set ℕ := {4, 5, 6, 8}
def B : Set ℕ := {5, 7, 8, 9}

theorem intersection_of_A_and_B_is_5_and_8 : A ∩ B = {5, 8} :=
  by sorry

end intersection_of_A_and_B_is_5_and_8_l1009_100971


namespace find_a4_l1009_100914

def arithmetic_sequence_sum (n : ℕ) (a₁ d : ℤ) : ℤ := n / 2 * (2 * a₁ + (n - 1) * d)

theorem find_a4 (a₁ d : ℤ) (S₅ S₉ : ℤ) 
  (h₁ : arithmetic_sequence_sum 5 a₁ d = 35)
  (h₂ : arithmetic_sequence_sum 9 a₁ d = 117) :
  (a₁ + 3 * d) = 20 := 
sorry

end find_a4_l1009_100914


namespace find_constant_k_eq_l1009_100991

theorem find_constant_k_eq : ∃ k : ℤ, (-x^2 - (k + 11)*x - 8 = -(x - 2)*(x - 4)) ↔ (k = -17) :=
by
  sorry

end find_constant_k_eq_l1009_100991


namespace students_arrangement_l1009_100903

def num_students := 5
def num_females := 2
def num_males := 3
def female_A_cannot_end := true
def only_two_males_next_to_each_other := true

theorem students_arrangement (h1: num_students = 5)
                             (h2: num_females = 2)
                             (h3: num_males = 3)
                             (h4: female_A_cannot_end = true)
                             (h5: only_two_males_next_to_each_other = true) :
    ∃ n, n = 48 :=
by
  sorry

end students_arrangement_l1009_100903


namespace income_growth_rate_l1009_100986

noncomputable def income_growth_eq (x : ℝ) : Prop :=
  3.2 * (1 + x)^2 = 3.7

theorem income_growth_rate :
  ∃ x : ℝ, income_growth_eq x :=
sorry

end income_growth_rate_l1009_100986


namespace dice_composite_probability_l1009_100992

theorem dice_composite_probability (m n : ℕ) (h : Nat.gcd m n = 1) :
  (∃ m n : ℕ, (m * 36 = 29 * n) ∧ Nat.gcd m n = 1) → m + n = 65 :=
by {
  sorry
}

end dice_composite_probability_l1009_100992


namespace ribbon_per_box_l1009_100993

def total_ribbon : ℝ := 4.5
def remaining_ribbon : ℝ := 1
def number_of_boxes : ℕ := 5

theorem ribbon_per_box :
  (total_ribbon - remaining_ribbon) / number_of_boxes = 0.7 :=
by
  sorry

end ribbon_per_box_l1009_100993


namespace minimize_frac_inv_l1009_100913

theorem minimize_frac_inv (a b : ℕ) (h1: 4 * a + b = 30) (h2: a > 0) (h3: b > 0) :
  (a, b) = (5, 10) :=
sorry

end minimize_frac_inv_l1009_100913


namespace avg_ac_l1009_100941

-- Define the ages of a, b, and c as variables A, B, and C
variables (A B C : ℕ)

-- Define the conditions
def avg_abc (A B C : ℕ) : Prop := (A + B + C) / 3 = 26
def age_b (B : ℕ) : Prop := B = 20

-- State the theorem to prove
theorem avg_ac {A B C : ℕ} (h1 : avg_abc A B C) (h2 : age_b B) : (A + C) / 2 = 29 := 
by sorry

end avg_ac_l1009_100941


namespace exists_solution_iff_l1009_100952

theorem exists_solution_iff (m : ℝ) (x y : ℝ) :
  ((y = (3 * m + 2) * x + 1) ∧ (y = (5 * m - 4) * x + 5)) ↔ m ≠ 3 :=
by sorry

end exists_solution_iff_l1009_100952


namespace number_of_gummies_l1009_100907

-- Define the necessary conditions
def lollipop_cost : ℝ := 1.5
def lollipop_count : ℕ := 4
def gummy_cost : ℝ := 2.0
def initial_money : ℝ := 15.0
def money_left : ℝ := 5.0

-- Total cost of lollipops and total amount spent on candies
noncomputable def total_lollipop_cost := lollipop_count * lollipop_cost
noncomputable def total_spent := initial_money - money_left
noncomputable def total_gummy_cost := total_spent - total_lollipop_cost
noncomputable def gummy_count := total_gummy_cost / gummy_cost

-- Main theorem statement
theorem number_of_gummies : gummy_count = 2 := 
by
  sorry -- Proof to be added

end number_of_gummies_l1009_100907


namespace sum_of_coefficients_l1009_100972

def original_function (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 4

def transformed_function (x : ℝ) : ℝ := 3 * (x + 2)^2 - 2 * (x + 2) + 4 + 5

theorem sum_of_coefficients : (3 : ℝ) + 10 + 17 = 30 :=
by
  sorry

end sum_of_coefficients_l1009_100972


namespace quadratic_two_distinct_real_roots_l1009_100934

theorem quadratic_two_distinct_real_roots : 
  ∀ (a b c : ℝ), a = 1 ∧ b = -5 ∧ c = 6 → 
  b^2 - 4 * a * c > 0 :=
by
  sorry

end quadratic_two_distinct_real_roots_l1009_100934


namespace running_current_each_unit_l1009_100968

theorem running_current_each_unit (I : ℝ) (h1 : ∀i, i = 2 * I) (h2 : ∀i, i * 3 = 6 * I) (h3 : 6 * I = 240) : I = 40 :=
by
  sorry

end running_current_each_unit_l1009_100968


namespace empty_seats_in_theater_l1009_100967

theorem empty_seats_in_theater :
  let total_seats := 750
  let occupied_seats := 532
  total_seats - occupied_seats = 218 :=
by
  sorry

end empty_seats_in_theater_l1009_100967


namespace sum_divisible_by_49_l1009_100931

theorem sum_divisible_by_49
  {x y z : ℤ} 
  (hx : x % 7 ≠ 0)
  (hy : y % 7 ≠ 0)
  (hz : z % 7 ≠ 0)
  (h : 7 ^ 3 ∣ (x ^ 7 + y ^ 7 + z ^ 7)) : 7^2 ∣ (x + y + z) :=
by
  sorry

end sum_divisible_by_49_l1009_100931


namespace cost_of_carrots_and_cauliflower_l1009_100923

variable {p c f o : ℝ}

theorem cost_of_carrots_and_cauliflower
  (h1 : p + c + f + o = 30)
  (h2 : o = 3 * p)
  (h3 : f = p + c) : 
  c + f = 14 := 
by
  sorry

end cost_of_carrots_and_cauliflower_l1009_100923


namespace rectangle_dimensions_l1009_100953

-- Definitions from conditions
def is_rectangle (length width : ℝ) : Prop :=
  3 * width = length ∧ 3 * width^2 = 8 * width

-- The theorem to prove
theorem rectangle_dimensions :
  ∃ (length width : ℝ), is_rectangle length width ∧ width = 8 / 3 ∧ length = 8 := by
  sorry

end rectangle_dimensions_l1009_100953


namespace percentage_decrease_second_year_l1009_100928

-- Define initial population
def initial_population : ℝ := 14999.999999999998

-- Define the population at the end of the first year after 12% increase
def population_end_year_1 : ℝ := initial_population * 1.12

-- Define the final population at the end of the second year
def final_population : ℝ := 14784.0

-- Define the proof statement
theorem percentage_decrease_second_year :
  ∃ D : ℝ, final_population = population_end_year_1 * (1 - D / 100) ∧ D = 12 :=
by
  sorry

end percentage_decrease_second_year_l1009_100928


namespace angle_C_eq_pi_div_3_side_c_eq_7_l1009_100910

theorem angle_C_eq_pi_div_3 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C) :
  C = Real.pi / 3 :=
sorry

theorem side_c_eq_7 
  (a b c : ℝ) 
  (h1 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C)
  (h1a : a = 5) 
  (h1b : b = 8) 
  (h2 : C = Real.pi / 3) :
  c = 7 :=
sorry

end angle_C_eq_pi_div_3_side_c_eq_7_l1009_100910


namespace ray_climbing_stairs_l1009_100918

theorem ray_climbing_stairs (n : ℕ) (h1 : n % 4 = 3) (h2 : n % 5 = 2) (h3 : 10 < n) : n = 27 :=
sorry

end ray_climbing_stairs_l1009_100918


namespace polynomial_identity_equals_neg_one_l1009_100974

theorem polynomial_identity_equals_neg_one
  (a a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (5 * x + 4)^3 = a + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a + a₂) - (a₁ + a₃) = -1 :=
by
  intro h
  sorry

end polynomial_identity_equals_neg_one_l1009_100974


namespace quadratic_condition_l1009_100963

theorem quadratic_condition (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (0 ≤ a ∧ a < 3) :=
sorry

end quadratic_condition_l1009_100963


namespace inequality_neg_multiply_l1009_100922

theorem inequality_neg_multiply {a b : ℝ} (h : a > b) : -2 * a < -2 * b :=
sorry

end inequality_neg_multiply_l1009_100922


namespace length_of_MN_l1009_100996

-- Define the lengths and trapezoid properties
variables (a b: ℝ)

-- Define the problem statement
theorem length_of_MN (a b: ℝ) :
  ∃ (MN: ℝ), ∀ (M N: ℝ) (is_trapezoid : True),
  (MN = 3 * a * b / (a + 2 * b)) :=
sorry

end length_of_MN_l1009_100996


namespace max_gold_coins_l1009_100917

variables (planks : ℕ)
          (windmill_planks windmill_gold : ℕ)
          (steamboat_planks steamboat_gold : ℕ)
          (airplane_planks airplane_gold : ℕ)

theorem max_gold_coins (h_planks: planks = 130)
                       (h_windmill: windmill_planks = 5 ∧ windmill_gold = 6)
                       (h_steamboat: steamboat_planks = 7 ∧ steamboat_gold = 8)
                       (h_airplane: airplane_planks = 14 ∧ airplane_gold = 19) :
  ∃ (gold : ℕ), gold = 172 :=
by
  sorry

end max_gold_coins_l1009_100917


namespace birds_flew_up_l1009_100937

theorem birds_flew_up (initial_birds new_birds total_birds : ℕ) 
    (h_initial : initial_birds = 29) 
    (h_total : total_birds = 42) : 
    new_birds = total_birds - initial_birds := 
by 
    sorry

end birds_flew_up_l1009_100937


namespace exist_polynomials_unique_polynomials_l1009_100984

-- Problem statement: the function 'f'
variable (f : ℝ → ℝ → ℝ → ℝ)

-- Condition: f(w, w, w) = 0 for all w ∈ ℝ
axiom f_ww_ww_ww (w : ℝ) : f w w w = 0

-- Statement for existence of A, B, C
theorem exist_polynomials (f : ℝ → ℝ → ℝ → ℝ)
  (hf : ∀ w : ℝ, f w w w = 0) : 
  ∃ A B C : ℝ → ℝ → ℝ → ℝ, 
  (∀ w : ℝ, A w w w + B w w w + C w w w = 0) ∧ 
  ∀ x y z : ℝ, f x y z = A x y z * (x - y) + B x y z * (y - z) + C x y z * (z - x) :=
sorry

-- Statement for uniqueness of A, B, C
theorem unique_polynomials (f : ℝ → ℝ → ℝ → ℝ) 
  (A B C A' B' C' : ℝ → ℝ → ℝ → ℝ)
  (hf: ∀ w : ℝ, f w w w = 0)
  (h1 : ∀ w : ℝ, A w w w + B w w w + C w w w = 0)
  (h2 : ∀ x y z : ℝ, f x y z = A x y z * (x - y) + B x y z * (y - z) + C x y z * (z - x))
  (h3 : ∀ w : ℝ, A' w w w + B' w w w + C' w w w = 0)
  (h4 : ∀ x y z : ℝ, f x y z = A' x y z * (x - y) + B' x y z * (y - z) + C' x y z * (z - x)) : 
  A = A' ∧ B = B' ∧ C = C' :=
sorry

end exist_polynomials_unique_polynomials_l1009_100984


namespace slow_train_speed_l1009_100945

/-- Given the conditions of two trains traveling towards each other and their meeting times,
     prove the speed of the slow train. -/
theorem slow_train_speed :
  let distance_AB := 901
  let slow_train_departure := 5 + 30 / 60 -- 5:30 AM in decimal hours
  let fast_train_departure := 9 + 30 / 60 -- 9:30 AM in decimal hours
  let meeting_time := 16 + 30 / 60 -- 4:30 PM in decimal hours
  let fast_train_speed := 58 -- speed in km/h
  let slow_train_time := meeting_time - slow_train_departure
  let fast_train_time := meeting_time - fast_train_departure
  let fast_train_distance := fast_train_speed * fast_train_time
  let slow_train_distance := distance_AB - fast_train_distance
  let slow_train_speed := slow_train_distance / slow_train_time
  slow_train_speed = 45 := sorry

end slow_train_speed_l1009_100945


namespace inequality_solution_l1009_100947

theorem inequality_solution {x : ℝ} : (1 / 2 - (x - 2) / 3 > 1) → (x < 1 / 2) :=
by {
  sorry
}

end inequality_solution_l1009_100947


namespace donuts_selection_l1009_100983

theorem donuts_selection :
  (∃ g c p : ℕ, g + c + p = 6 ∧ g ≥ 1 ∧ c ≥ 1 ∧ p ≥ 1) →
  ∃ k : ℕ, k = 10 :=
by {
  -- The mathematical proof steps are omitted according to the instructions
  sorry
}

end donuts_selection_l1009_100983


namespace average_speed_is_50_l1009_100981

-- Defining the conditions
def totalDistance : ℕ := 250
def totalTime : ℕ := 5

-- Defining the average speed
def averageSpeed := totalDistance / totalTime

-- The theorem statement
theorem average_speed_is_50 : averageSpeed = 50 := sorry

end average_speed_is_50_l1009_100981


namespace donation_to_second_orphanage_l1009_100938

variable (total_donation : ℝ) (first_donation : ℝ) (third_donation : ℝ)

theorem donation_to_second_orphanage :
  total_donation = 650 ∧ first_donation = 175 ∧ third_donation = 250 →
  (total_donation - first_donation - third_donation = 225) := by
  sorry

end donation_to_second_orphanage_l1009_100938


namespace average_of_first_12_is_14_l1009_100956

-- Definitions based on given conditions
def average_of_25 := 19
def sum_of_25 := average_of_25 * 25

def average_of_last_12 := 17
def sum_of_last_12 := average_of_last_12 * 12

def result_13 := 103

-- Main proof statement to be checked
theorem average_of_first_12_is_14 (A : ℝ) (h1 : sum_of_25 = sum_of_25) (h2 : sum_of_last_12 = sum_of_last_12) (h3 : result_13 = 103) :
  (A * 12 + result_13 + sum_of_last_12 = sum_of_25) → (A = 14) :=
by
  sorry

end average_of_first_12_is_14_l1009_100956


namespace sum_of_roots_ln_abs_eq_l1009_100911

theorem sum_of_roots_ln_abs_eq (m : ℝ) (x1 x2 : ℝ) (hx1 : Real.log (|x1|) = m) (hx2 : Real.log (|x2|) = m) : x1 + x2 = 0 :=
sorry

end sum_of_roots_ln_abs_eq_l1009_100911


namespace transformed_expression_value_l1009_100942

-- Defining the new operations according to the problem's conditions
def new_minus (a b : ℕ) : ℕ := a + b
def new_plus (a b : ℕ) : ℕ := a * b
def new_times (a b : ℕ) : ℕ := a / b
def new_div (a b : ℕ) : ℕ := a - b

-- Problem statement
theorem transformed_expression_value : new_minus 6 (new_plus 9 (new_times 8 (new_div 3 25))) = 5 :=
sorry

end transformed_expression_value_l1009_100942


namespace fraction_value_l1009_100985

theorem fraction_value (a : ℕ) (h : a > 0) (h_eq : (a:ℝ) / (a + 35) = 0.7) : a = 82 :=
by
  -- Steps to prove the theorem here
  sorry

end fraction_value_l1009_100985


namespace hyperbola_eccentricity_l1009_100976

theorem hyperbola_eccentricity 
  (a b c e : ℝ)
  (h_hyperbola : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_positive_a : a > 0)
  (h_positive_b : b > 0)
  (h_asymptote_parallel : b = 2 * a)
  (h_c_squared : c^2 = a^2 + b^2)
  (h_e_def : e = c / a) :
  e = Real.sqrt 5 :=
sorry

end hyperbola_eccentricity_l1009_100976


namespace garin_homework_pages_l1009_100962

theorem garin_homework_pages (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) : 
    pages_per_day = 19 → 
    days = 24 → 
    total_pages = pages_per_day * days → 
    total_pages = 456 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end garin_homework_pages_l1009_100962


namespace quarters_remaining_l1009_100925

-- Define the number of quarters Sally originally had
def initialQuarters : Nat := 760

-- Define the number of quarters Sally spent
def spentQuarters : Nat := 418

-- Prove that the number of quarters she has now is 342
theorem quarters_remaining : initialQuarters - spentQuarters = 342 :=
by
  sorry

end quarters_remaining_l1009_100925


namespace poly_comp_eq_l1009_100932

variable {K : Type*} [Field K]

theorem poly_comp_eq {Q1 Q2 : Polynomial K} (P : Polynomial K) (hP : ¬P.degree = 0) :
  Q1.comp P = Q2.comp P → Q1 = Q2 :=
by
  intro h
  sorry

end poly_comp_eq_l1009_100932


namespace cards_given_l1009_100965

/-- Martha starts with 3 cards. She ends up with 79 cards after receiving some from Emily. We need to prove that Emily gave her 76 cards. -/
theorem cards_given (initial_cards final_cards cards_given : ℕ) (h1 : initial_cards = 3) (h2 : final_cards = 79) (h3 : final_cards = initial_cards + cards_given) :
  cards_given = 76 :=
sorry

end cards_given_l1009_100965


namespace vertical_line_divides_triangle_l1009_100970

theorem vertical_line_divides_triangle (k : ℝ) :
  let triangle_area := 1 / 2 * |0 * (1 - 1) + 1 * (1 - 0) + 9 * (0 - 1)|
  let left_triangle_area := 1 / 2 * |0 * (1 - 1) + k * (1 - 0) + 1 * (0 - 1)|
  let right_triangle_area := triangle_area - left_triangle_area
  triangle_area = 4 
  ∧ left_triangle_area = 2
  ∧ right_triangle_area = 2
  ∧ (k = 5) ∨ (k = -3) → 
  k = 5 :=
by
  sorry

end vertical_line_divides_triangle_l1009_100970


namespace find_m_if_f_even_l1009_100959

theorem find_m_if_f_even (m : ℝ) (f : ℝ → ℝ) : 
  (∀ x : ℝ, f x = x^4 + (m - 1) * x + 1) ∧ (∀ x : ℝ, f x = f (-x)) → m = 1 := 
by 
  sorry

end find_m_if_f_even_l1009_100959


namespace apples_to_pears_l1009_100988

theorem apples_to_pears :
  (3 / 4) * 12 = 9 → (2 / 3) * 6 = 4 :=
by {
  sorry
}

end apples_to_pears_l1009_100988


namespace dino_finances_l1009_100939

def earnings_per_gig (hours: ℕ) (rate: ℕ) : ℕ := hours * rate

def dino_total_income : ℕ :=
  earnings_per_gig 20 10 + -- Earnings from the first gig
  earnings_per_gig 30 20 + -- Earnings from the second gig
  earnings_per_gig 5 40    -- Earnings from the third gig

def dino_expenses : ℕ := 500

def dino_net_income : ℕ :=
  dino_total_income - dino_expenses

theorem dino_finances : 
  dino_net_income = 500 :=
by
  -- Here, the actual proof would be constructed.
  sorry

end dino_finances_l1009_100939


namespace train_length_l1009_100919

theorem train_length (L V : ℝ) (h1 : L = V * 26) (h2 : L + 150 = V * 39) : L = 300 := by
  sorry

end train_length_l1009_100919


namespace total_pears_picked_l1009_100909

variables (jason_keith_mike_morning : ℕ)
variables (alicia_tina_nicola_afternoon : ℕ)
variables (days : ℕ)
variables (total_pears : ℕ)

def one_day_total (jason_keith_mike_morning alicia_tina_nicola_afternoon : ℕ) : ℕ :=
  jason_keith_mike_morning + alicia_tina_nicola_afternoon

theorem total_pears_picked (hjkm: jason_keith_mike_morning = 46 + 47 + 12)
                           (hatn: alicia_tina_nicola_afternoon = 28 + 33 + 52)
                           (hdays: days = 3)
                           (htotal: total_pears = 654):
  total_pears = (one_day_total  (46 + 47 + 12)  (28 + 33 + 52)) * 3 := 
sorry

end total_pears_picked_l1009_100909


namespace ray_initial_cents_l1009_100975

theorem ray_initial_cents :
  ∀ (initial_cents : ℕ), 
    (∃ (peter_cents : ℕ), 
      peter_cents = 30 ∧
      ∃ (randi_cents : ℕ),
        randi_cents = 2 * peter_cents ∧
        randi_cents = peter_cents + 60 ∧
        peter_cents + randi_cents = initial_cents
    ) →
    initial_cents = 90 := 
by
    intros initial_cents h
    obtain ⟨peter_cents, hp, ⟨randi_cents, hr1, hr2, hr3⟩⟩ := h
    sorry

end ray_initial_cents_l1009_100975


namespace Bobby_has_27_pairs_l1009_100926

-- Define the number of shoes Becky has
variable (B : ℕ)

-- Define the number of shoes Bonny has as 13, with the relationship to Becky's shoes
def Bonny_shoes : Prop := 2 * B - 5 = 13

-- Define the number of shoes Bobby has given Becky's count
def Bobby_shoes := 3 * B

-- Prove that Bobby has 27 pairs of shoes given the conditions
theorem Bobby_has_27_pairs (hB : Bonny_shoes B) : Bobby_shoes B = 27 := 
by 
  sorry

end Bobby_has_27_pairs_l1009_100926


namespace power_function_point_l1009_100933

theorem power_function_point (a : ℝ) (h : (2 : ℝ) ^ a = (1 / 2 : ℝ)) : a = -1 :=
by sorry

end power_function_point_l1009_100933


namespace jiwon_walk_distance_l1009_100901

theorem jiwon_walk_distance : 
  (13 * 90) * 0.45 = 526.5 := by
  sorry

end jiwon_walk_distance_l1009_100901


namespace tan_neg_3780_eq_zero_l1009_100994

theorem tan_neg_3780_eq_zero : Real.tan (-3780 * Real.pi / 180) = 0 := 
by 
  sorry

end tan_neg_3780_eq_zero_l1009_100994


namespace units_digit_2_104_5_205_11_302_l1009_100916

theorem units_digit_2_104_5_205_11_302 : 
  ((2 ^ 104) * (5 ^ 205) * (11 ^ 302)) % 10 = 0 :=
by
  sorry

end units_digit_2_104_5_205_11_302_l1009_100916


namespace circle_symmetric_about_line_l1009_100940

theorem circle_symmetric_about_line :
  ∃ b : ℝ, (∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 4 = 0 → y = 2*x + b) → b = 4 :=
by
  sorry

end circle_symmetric_about_line_l1009_100940


namespace two_digit_multiples_of_4_and_9_l1009_100906

theorem two_digit_multiples_of_4_and_9 :
  ∃ (count : ℕ), 
    (∀ (n : ℕ), (10 ≤ n ∧ n ≤ 99) → (n % 4 = 0 ∧ n % 9 = 0) → (n = 36 ∨ n = 72)) ∧ count = 2 :=
by
  sorry

end two_digit_multiples_of_4_and_9_l1009_100906


namespace sqrt_meaningful_range_l1009_100929

theorem sqrt_meaningful_range (x : ℝ) : x + 1 ≥ 0 ↔ x ≥ -1 :=
by sorry

end sqrt_meaningful_range_l1009_100929


namespace area_of_ADFE_l1009_100973

namespace Geometry

open Classical

noncomputable def area_triangle (A B C : Type) [Field A] (area_DBF area_BFC area_FCE : A) : A :=
  let total_area := area_DBF + area_BFC + area_FCE
  let area := (105 : A) / 4
  total_area + area

theorem area_of_ADFE (A B C D E F : Type) [Field A] 
  (area_DBF : A) (area_BFC : A) (area_FCE : A) : 
  area_DBF = 4 → area_BFC = 6 → area_FCE = 5 → 
  area_triangle A B C area_DBF area_BFC area_FCE = (15 : A) + (105 : A) / 4 := 
by 
  intros 
  sorry

end area_of_ADFE_l1009_100973


namespace new_releases_fraction_is_2_over_5_l1009_100951

def fraction_new_releases (total_books : ℕ) (frac_historical_fiction : ℚ) (frac_new_historical_fiction : ℚ) (frac_new_non_historical_fiction : ℚ) : ℚ :=
  let num_historical_fiction := frac_historical_fiction * total_books
  let num_new_historical_fiction := frac_new_historical_fiction * num_historical_fiction
  let num_non_historical_fiction := total_books - num_historical_fiction
  let num_new_non_historical_fiction := frac_new_non_historical_fiction * num_non_historical_fiction
  let total_new_releases := num_new_historical_fiction + num_new_non_historical_fiction
  num_new_historical_fiction / total_new_releases

theorem new_releases_fraction_is_2_over_5 :
  ∀ (total_books : ℕ), total_books > 0 →
    fraction_new_releases total_books (40 / 100) (40 / 100) (40 / 100) = 2 / 5 :=
by 
  intro total_books h
  sorry

end new_releases_fraction_is_2_over_5_l1009_100951


namespace limit_of_sequence_N_of_epsilon_l1009_100999

theorem limit_of_sequence (a_n : ℕ → ℝ) (a : ℝ) (h : ∀ n, a_n n = (7 * n - 1) / (n + 1)) :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) ↔ a = 7 := sorry

theorem N_of_epsilon (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, N = ⌈8 / ε⌉ := sorry

end limit_of_sequence_N_of_epsilon_l1009_100999


namespace sum_of_integers_l1009_100921

variable (x y : ℕ)

theorem sum_of_integers (h1 : x - y = 10) (h2 : x * y = 56) : x + y = 18 := 
by 
  sorry

end sum_of_integers_l1009_100921


namespace arithmetic_sequence_sixth_term_l1009_100964

theorem arithmetic_sequence_sixth_term (a d : ℤ) 
    (sum_first_five : a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = 15)
    (fourth_term : a + 3 * d = 4) : a + 5 * d = 6 :=
by
  sorry

end arithmetic_sequence_sixth_term_l1009_100964


namespace inequality_bound_l1009_100930

theorem inequality_bound (a x : ℝ) (h1 : |a| ≤ 1) (h2 : |x| ≤ 1) :
  |x^2 - ax - a^2| ≤ 5 / 4 :=
sorry

end inequality_bound_l1009_100930


namespace line_through_midpoint_l1009_100977

theorem line_through_midpoint (x y : ℝ)
  (ellipse : x^2 / 25 + y^2 / 16 = 1)
  (midpoint : P = (2, 1)) :
  ∃ (A B : ℝ × ℝ),
    A ≠ B ∧
    P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧
    (A.1^2 / 25 + A.2^2 / 16 = 1) ∧
    (B.1^2 / 25 + B.2^2 / 16 = 1) ∧
    (x = 32*y - 25*x - 89) :=
sorry

end line_through_midpoint_l1009_100977


namespace jane_current_age_l1009_100982

noncomputable def JaneAge : ℕ := 34

theorem jane_current_age : 
  ∃ J : ℕ, 
    (∀ t : ℕ, t ≥ 18 ∧ t - 18 ≤ JaneAge - 18 → t ≤ JaneAge / 2) ∧
    (JaneAge - 12 = 23 - 12 * 2) ∧
    (23 = 23) →
    J = 34 := by
  sorry

end jane_current_age_l1009_100982


namespace necessary_not_sufficient_to_form_triangle_l1009_100948

-- Define the vectors and the condition
variables (a b c : ℝ × ℝ)

-- Define the condition that these vectors form a closed loop (triangle)
def forms_closed_loop (a b c : ℝ × ℝ) : Prop :=
  a + b + c = (0, 0)

-- Prove that the condition is necessary but not sufficient
theorem necessary_not_sufficient_to_form_triangle :
  forms_closed_loop a b c → ∃ (x : ℝ × ℝ), a ≠ x ∧ b ≠ -2 * x ∧ c ≠ x :=
sorry

end necessary_not_sufficient_to_form_triangle_l1009_100948


namespace solve_for_x_l1009_100961

theorem solve_for_x (x : ℚ) (h : x + 3 * x = 300 - (4 * x + 5 * x)) : x = 300 / 13 :=
by
  sorry

end solve_for_x_l1009_100961


namespace payment_difference_correct_l1009_100944

noncomputable def initial_debt : ℝ := 12000

noncomputable def planA_interest_rate : ℝ := 0.08
noncomputable def planA_compounding_periods : ℕ := 2

noncomputable def planB_interest_rate : ℝ := 0.08

noncomputable def planA_payment_years : ℕ := 4
noncomputable def planA_remaining_years : ℕ := 4

noncomputable def planB_years : ℕ := 8

-- Amount accrued in Plan A after 4 years
noncomputable def planA_amount_after_first_period : ℝ :=
  initial_debt * (1 + planA_interest_rate / planA_compounding_periods) ^ (planA_compounding_periods * planA_payment_years)

-- Amount paid at the end of first period (two-thirds of total)
noncomputable def planA_first_payment : ℝ :=
  (2/3) * planA_amount_after_first_period

-- Remaining debt after first payment
noncomputable def planA_remaining_debt : ℝ :=
  planA_amount_after_first_period - planA_first_payment

-- Amount accrued on remaining debt after 8 years (second 4-year period)
noncomputable def planA_second_payment : ℝ :=
  planA_remaining_debt * (1 + planA_interest_rate / planA_compounding_periods) ^ (planA_compounding_periods * planA_remaining_years)

-- Total payment under Plan A
noncomputable def total_payment_planA : ℝ :=
  planA_first_payment + planA_second_payment

-- Total payment under Plan B
noncomputable def total_payment_planB : ℝ :=
  initial_debt * (1 + planB_interest_rate * planB_years)

-- Positive difference between payments
noncomputable def payment_difference : ℝ :=
  total_payment_planB - total_payment_planA

theorem payment_difference_correct :
  payment_difference = 458.52 :=
by
  sorry

end payment_difference_correct_l1009_100944


namespace geometric_sequence_fraction_l1009_100908

noncomputable def a_n : ℕ → ℝ := sorry -- geometric sequence {a_n}
noncomputable def S : ℕ → ℝ := sorry   -- sequence sum S_n
def q : ℝ := sorry                     -- common ratio

theorem geometric_sequence_fraction (h_sequence: ∀ n, 2 * S (n - 1) = S n + S (n + 1))
  (h_q: ∀ n, a_n (n + 1) = q * a_n n)
  (h_q_neg2: q = -2) :
  (a_n 5 + a_n 7) / (a_n 3 + a_n 5) = 4 :=
by 
  sorry

end geometric_sequence_fraction_l1009_100908


namespace fresh_grapes_weight_l1009_100902

/-- Given fresh grapes containing 90% water by weight, 
    and dried grapes containing 20% water by weight,
    if the weight of dried grapes obtained from a certain amount of fresh grapes is 2.5 kg,
    then the weight of the fresh grapes used is 20 kg.
-/
theorem fresh_grapes_weight (F D : ℝ)
  (hD : D = 2.5)
  (fresh_water_content : ℝ := 0.90)
  (dried_water_content : ℝ := 0.20)
  (fresh_solid_content : ℝ := 1 - fresh_water_content)
  (dried_solid_content : ℝ := 1 - dried_water_content)
  (solid_mass_constancy : fresh_solid_content * F = dried_solid_content * D) : 
  F = 20 := 
  sorry

end fresh_grapes_weight_l1009_100902


namespace hours_per_day_l1009_100998

theorem hours_per_day (H : ℕ) : 
  (42 * 12 * H = 30 * 14 * 6) → 
  H = 5 := by
  sorry

end hours_per_day_l1009_100998


namespace triangle_sides_possible_k_l1009_100935

noncomputable def f (x k : ℝ) : ℝ := x^2 - 4*x + 4 + k^2

theorem triangle_sides_possible_k (a b c k : ℝ) (ha : 0 ≤ a) (hb : a ≤ 3) (ha' : 0 ≤ b) (hb' : b ≤ 3) (ha'' : 0 ≤ c) (hb'' : c ≤ 3) :
  (f a k + f b k > f c k) ∧ (f a k + f c k > f b k) ∧ (f b k + f c k > f a k) ↔ k = 3 ∨ k = 4 :=
by
  sorry

end triangle_sides_possible_k_l1009_100935


namespace gcd_of_items_l1009_100990

def numPens : ℕ := 891
def numPencils : ℕ := 810
def numNotebooks : ℕ := 1080
def numErasers : ℕ := 972

theorem gcd_of_items :
  Nat.gcd (Nat.gcd (Nat.gcd numPens numPencils) numNotebooks) numErasers = 27 :=
by
  sorry

end gcd_of_items_l1009_100990


namespace exists_abcd_for_n_gt_one_l1009_100915

theorem exists_abcd_for_n_gt_one (n : Nat) (h : n > 1) :
  ∃ a b c d : Nat, a + b = 4 * n ∧ c + d = 4 * n ∧ a * b - c * d = 4 * n := 
by
  sorry

end exists_abcd_for_n_gt_one_l1009_100915


namespace original_decimal_l1009_100927

theorem original_decimal (x : ℝ) : (10 * x = x + 2.7) → x = 0.3 := 
by
    intro h
    sorry

end original_decimal_l1009_100927


namespace largest_consecutive_sum_55_l1009_100943

theorem largest_consecutive_sum_55 :
  ∃ n a : ℕ, (n * (a + (n - 1) / 2) = 55) ∧ (n = 10) ∧ (∀ m : ℕ, ∀ b : ℕ, (m * (b + (m - 1) / 2) = 55) → (m ≤ 10)) :=
by 
  sorry

end largest_consecutive_sum_55_l1009_100943


namespace find_solution_l1009_100966

open Nat

def binomial (n : ℕ) (k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

noncomputable def expression (n : ℕ) : ℕ :=
  1 + binomial n 1 + binomial n 2 + binomial n 3

theorem find_solution (n : ℕ) (h : n > 3) :
  expression n ∣ 2 ^ 2000 ↔ n = 7 ∨ n = 23 :=
by
  sorry

end find_solution_l1009_100966


namespace ratio_of_ages_l1009_100949

-- Given conditions
def present_age_sum (H J : ℕ) : Prop :=
  H + J = 43

def present_ages (H J : ℕ) : Prop := 
  H = 27 ∧ J = 16

def multiple_of_age (H J k : ℕ) : Prop :=
  H - 5 = k * (J - 5)

-- Prove that the ratio of Henry's age to Jill's age 5 years ago was 2:1
theorem ratio_of_ages (H J k : ℕ) 
  (h_sum : present_age_sum H J)
  (h_present : present_ages H J)
  (h_multiple : multiple_of_age H J k) :
  (H - 5) / (J - 5) = 2 :=
by
  sorry

end ratio_of_ages_l1009_100949


namespace area_of_rectangle_l1009_100912

-- Define the conditions
variable {S1 S2 S3 S4 : ℝ} -- side lengths of the four squares

-- The conditions:
-- 1. Four non-overlapping squares
-- 2. The area of the shaded square is 4 square inches
def conditions (S1 S2 S3 S4 : ℝ) : Prop :=
    S1^2 = 4 -- Given that one of the squares has an area of 4 square inches

-- The proof problem:
theorem area_of_rectangle (S1 S2 S3 S4 : ℝ) (h1 : 2 * S1 = S2) (h2 : 2 * S2 = S3) (h3 : conditions S1 S2 S3 S4) : 
    S1^2 + S2^2 + S3^2 = 24 :=
by
  sorry

end area_of_rectangle_l1009_100912


namespace smallest_area_of_right_triangle_l1009_100980

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l1009_100980


namespace expression_numerator_l1009_100920

theorem expression_numerator (p q : ℕ) (E : ℕ) 
  (h1 : p * 5 = q * 4)
  (h2 : (18 / 7) + (E / (2 * q + p)) = 3) : E = 6 := 
by 
  sorry

end expression_numerator_l1009_100920


namespace carnival_ticket_count_l1009_100995

theorem carnival_ticket_count (ferris_wheel_rides bumper_car_rides ride_cost : ℕ) 
  (h1 : ferris_wheel_rides = 7) 
  (h2 : bumper_car_rides = 3) 
  (h3 : ride_cost = 5) : 
  ferris_wheel_rides + bumper_car_rides * ride_cost = 50 := 
by {
  -- proof omitted
  sorry
}

end carnival_ticket_count_l1009_100995


namespace exponentiation_rule_l1009_100904

theorem exponentiation_rule (b : ℝ) : (-2 * b) ^ 3 = -8 * b ^ 3 :=
by sorry

end exponentiation_rule_l1009_100904


namespace find_ab_l1009_100969

noncomputable def poly (x a b : ℝ) := x^4 + a * x^3 - 5 * x^2 + b * x - 6

theorem find_ab (a b : ℝ) (h : poly 2 a b = 0) : (a = 0 ∧ b = 4) :=
by
  sorry

end find_ab_l1009_100969


namespace incenter_sum_equals_one_l1009_100936

noncomputable def incenter (A B C : Point) : Point := sorry -- Definition goes here

def side_length (A B C : Point) (a b c : ℝ) : Prop :=
  -- Definitions relating to side lengths go here
  sorry

theorem incenter_sum_equals_one (A B C I : Point) (a b c IA IB IC : ℝ) (h_incenter : I = incenter A B C)
    (h_sides : side_length A B C a b c) :
    (IA ^ 2 / (b * c)) + (IB ^ 2 / (a * c)) + (IC ^ 2 / (a * b)) = 1 :=
  sorry

end incenter_sum_equals_one_l1009_100936


namespace sum_series_equals_4_div_9_l1009_100946

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l1009_100946


namespace sum_of_five_distinct_integers_product_2022_l1009_100950

theorem sum_of_five_distinct_integers_product_2022 :
  ∃ (a b c d e : ℤ), 
    a * b * c * d * e = 2022 ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧ 
    (a + b + c + d + e = 342 ∨
     a + b + c + d + e = 338 ∨
     a + b + c + d + e = 336 ∨
     a + b + c + d + e = -332) :=
by 
  sorry

end sum_of_five_distinct_integers_product_2022_l1009_100950


namespace determine_ratio_l1009_100905

-- Definition of the given conditions.
def total_length : ℕ := 69
def longer_length : ℕ := 46
def ratio_of_lengths (shorter_length longer_length : ℕ) : ℕ := longer_length / shorter_length

-- The theorem we need to prove.
theorem determine_ratio (x : ℕ) (m : ℕ) (h1 : longer_length = m * x) (h2 : x + longer_length = total_length) : 
  ratio_of_lengths x longer_length = 2 :=
by
  sorry

end determine_ratio_l1009_100905


namespace opposite_pairs_l1009_100997

theorem opposite_pairs :
  (3^2 = 9) ∧ (-3^2 = -9) ∧
  ¬ ((3^2 = 9 ∧ -2^3 = -8) ∧ 9 = -(-8)) ∧
  ¬ ((3^2 = 9 ∧ (-3)^2 = 9) ∧ 9 = -9) ∧
  ¬ ((-3^2 = -9 ∧ -(-3)^2 = -9) ∧ -9 = -(-9)) :=
by
  sorry

end opposite_pairs_l1009_100997


namespace pq_sub_l1009_100989

-- Assuming the conditions
theorem pq_sub (p q : ℚ) 
  (h₁ : 3 / p = 4) 
  (h₂ : 3 / q = 18) : 
  p - q = 7 / 12 := 
  sorry

end pq_sub_l1009_100989
