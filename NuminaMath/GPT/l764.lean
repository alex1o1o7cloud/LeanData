import Mathlib

namespace joseph_total_distance_l764_76496

-- Distance Joseph runs on Monday
def d1 : ℕ := 900

-- Increment each day
def increment : ℕ := 200

-- Adjust distance calculation
def d2 := d1 + increment
def d3 := d2 + increment

-- Total distance calculation
def total_distance := d1 + d2 + d3

-- Prove that the total distance is 3300 meters
theorem joseph_total_distance : total_distance = 3300 :=
by sorry

end joseph_total_distance_l764_76496


namespace frame_dimension_ratio_l764_76471

theorem frame_dimension_ratio (W H x : ℕ) (h1 : W = 20) (h2 : H = 30) (h3 : 2 * (W + 2 * x) * (H + 6 * x) - W * H = 2 * (W * H)) :
  (W + 2 * x) / (H + 6 * x) = 1/2 :=
by sorry

end frame_dimension_ratio_l764_76471


namespace bus_speed_proof_l764_76493
noncomputable def speed_of_bus (train_length : ℝ) (train_speed_kmph : ℝ) (time_to_pass : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let relative_speed_mps := train_length / time_to_pass
  let bus_speed_mps := relative_speed_mps - train_speed_mps
  bus_speed_mps * 3.6

theorem bus_speed_proof : 
  speed_of_bus 220 90 5.279577633789296 = 60 :=
by
  sorry

end bus_speed_proof_l764_76493


namespace number_of_candy_packages_l764_76452

theorem number_of_candy_packages (total_candies pieces_per_package : ℕ) 
  (h_total_candies : total_candies = 405)
  (h_pieces_per_package : pieces_per_package = 9) :
  total_candies / pieces_per_package = 45 := by
  sorry

end number_of_candy_packages_l764_76452


namespace recipe_calls_for_eight_cups_of_sugar_l764_76481

def cups_of_flour : ℕ := 6
def cups_of_salt : ℕ := 7
def additional_sugar_needed (salt : ℕ) : ℕ := salt + 1

theorem recipe_calls_for_eight_cups_of_sugar :
  additional_sugar_needed cups_of_salt = 8 :=
by
  -- condition 1: cups_of_flour = 6
  -- condition 2: cups_of_salt = 7
  -- condition 4: additional_sugar_needed = salt + 1
  -- prove formula: 7 + 1 = 8
  sorry

end recipe_calls_for_eight_cups_of_sugar_l764_76481


namespace arithmetic_progression_term_l764_76400

variable (n r : ℕ)

-- Given the sum of the first n terms of an arithmetic progression is S_n = 3n + 4n^2
def S (n : ℕ) : ℕ := 3 * n + 4 * n^2

-- Prove that the r-th term of the sequence is 8r - 1
theorem arithmetic_progression_term :
  (S r) - (S (r - 1)) = 8 * r - 1 :=
by
  sorry

end arithmetic_progression_term_l764_76400


namespace length_of_tangent_point_to_circle_l764_76443

theorem length_of_tangent_point_to_circle :
  let P := (2, 3)
  let O := (0, 0)
  let r := 1
  let OP := Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2)
  let tangent_length := Real.sqrt (OP^2 - r^2)
  tangent_length = 2 * Real.sqrt 3 := by
  sorry

end length_of_tangent_point_to_circle_l764_76443


namespace time_rachel_is_13_l764_76412

-- Definitions based on problem conditions
def time_matt := 12
def time_patty := time_matt / 3
def time_rachel := 2 * time_patty + 5

-- Theorem statement to prove Rachel's time to paint the house
theorem time_rachel_is_13 : time_rachel = 13 := 
by 
  sorry

end time_rachel_is_13_l764_76412


namespace julia_ink_containers_l764_76409

-- Definitions based on conditions
def total_posters : Nat := 60
def posters_remaining : Nat := 45
def lost_containers : Nat := 1

-- Required to be proven statement
theorem julia_ink_containers : 
  (total_posters - posters_remaining) = 15 → 
  posters_remaining / 15 = 3 := 
by 
  sorry

end julia_ink_containers_l764_76409


namespace factorization_of_x_squared_minus_one_l764_76403

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l764_76403


namespace rem_value_is_correct_l764_76483

def rem (x y : ℚ) : ℚ :=
  x - y * (Int.floor (x / y))

theorem rem_value_is_correct : rem (-5/9) (7/3) = 16/9 := by
  sorry

end rem_value_is_correct_l764_76483


namespace find_constants_and_extrema_l764_76408

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

theorem find_constants_and_extrema (a b c : ℝ) (h : a ≠ 0) 
    (ext1 : deriv (f a b c) 1 = 0) (ext2 : deriv (f a b c) (-1) = 0) (value1 : f a b c 1 = -1) :
    a = -1/2 ∧ b = 0 ∧ c = 1/2 ∧ 
    (∃ x : ℝ, x = 1 ∧ deriv (deriv (f a b c)) x < 0) ∧
    (∃ x : ℝ, x = -1 ∧ deriv (deriv (f a b c)) x > 0) :=
sorry

end find_constants_and_extrema_l764_76408


namespace moles_of_CaCl2_l764_76468

/-- 
We are given the reaction: CaCO3 + 2 HCl → CaCl2 + CO2 + H2O 
with 2 moles of HCl and 1 mole of CaCO3. We need to prove that the number 
of moles of CaCl2 formed is 1.
-/
theorem moles_of_CaCl2 (HCl: ℝ) (CaCO3: ℝ) (reaction: CaCO3 + 2 * HCl = 1): CaCO3 = 1 → HCl = 2 → CaCl2 = 1 :=
by
  -- importing the required context for chemical equations and stoichiometry
  sorry

end moles_of_CaCl2_l764_76468


namespace zoe_takes_correct_amount_of_money_l764_76478

def numberOfPeople : ℕ := 6
def costPerSoda : ℝ := 0.5
def costPerPizza : ℝ := 1.0

def totalCost : ℝ := (numberOfPeople * costPerSoda) + (numberOfPeople * costPerPizza)

theorem zoe_takes_correct_amount_of_money : totalCost = 9 := sorry

end zoe_takes_correct_amount_of_money_l764_76478


namespace inequality_geq_27_l764_76492

theorem inequality_geq_27 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_eq : a + b + c + 2 = a * b * c) : (a + 1) * (b + 1) * (c + 1) ≥ 27 := 
    sorry

end inequality_geq_27_l764_76492


namespace prize_distribution_correct_l764_76450

def probability_A_correct : ℚ := 3 / 4
def probability_B_correct : ℚ := 4 / 5
def total_prize : ℚ := 190

-- Calculation of expected prizes
def probability_A_only_correct : ℚ := probability_A_correct * (1 - probability_B_correct)
def probability_B_only_correct : ℚ := probability_B_correct * (1 - probability_A_correct)
def probability_both_correct : ℚ := probability_A_correct * probability_B_correct

def normalized_probability : ℚ := probability_A_only_correct + probability_B_only_correct + probability_both_correct

def expected_prize_A : ℚ := (probability_A_only_correct / normalized_probability * total_prize) + (probability_both_correct / normalized_probability * (total_prize / 2))
def expected_prize_B : ℚ := (probability_B_only_correct / normalized_probability * total_prize) + (probability_both_correct / normalized_probability * (total_prize / 2))

theorem prize_distribution_correct :
  expected_prize_A = 90 ∧ expected_prize_B = 100 := 
by
  sorry

end prize_distribution_correct_l764_76450


namespace discriminant_of_quadratic_l764_76474

-- Define the quadratic equation coefficients
def a : ℚ := 5
def b : ℚ := 5 + 1/2
def c : ℚ := 1/2

-- Define the discriminant function
def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

-- State the theorem
theorem discriminant_of_quadratic :
  discriminant a b c = 81 / 4 :=
by
  -- We provide the result of the computation directly
  sorry

end discriminant_of_quadratic_l764_76474


namespace largest_n_unique_k_l764_76449

theorem largest_n_unique_k : ∃! (n : ℕ), ∃ (k : ℤ),
  (7 / 16 : ℚ) < (n : ℚ) / (n + k : ℚ) ∧ (n : ℚ) / (n + k : ℚ) < (8 / 17 : ℚ) ∧ n = 112 := 
sorry

end largest_n_unique_k_l764_76449


namespace smallest_prime_dividing_sum_l764_76426

theorem smallest_prime_dividing_sum (h1 : Odd 7) (h2 : Odd 9) 
    (h3 : ∀ {a b : ℤ}, Odd a → Odd b → Even (a + b)) :
  ∃ p : ℕ, Prime p ∧ p ∣ (7 ^ 15 + 9 ^ 7) ∧ p = 2 := 
by
  sorry

end smallest_prime_dividing_sum_l764_76426


namespace ratio_of_height_to_radius_max_volume_l764_76423

theorem ratio_of_height_to_radius_max_volume (r h : ℝ) (h_surface_area : 2 * Real.pi * r^2 + 2 * Real.pi * r * h = 6 * Real.pi) :
  (exists (max_r : ℝ) (max_h : ℝ), 2 * r * max_r + 2 * r * max_h = 6 * Real.pi ∧ 
                                  max_r = 1 ∧ 
                                  max_h = 2 ∧ 
                                  (max_h / max_r) = 2) :=
by
  sorry

end ratio_of_height_to_radius_max_volume_l764_76423


namespace quadratic_inequality_solution_l764_76454

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x ^ 2 + 4 * x + 1 > 0) ↔ (a > 4) :=
sorry

end quadratic_inequality_solution_l764_76454


namespace simplify_eval_expression_l764_76463

theorem simplify_eval_expression : 
  ∀ (a b : ℤ), a = -1 → b = 4 → ((a - b)^2 - 2 * a * (a + b) + (a + 2 * b) * (a - 2 * b)) = -32 := 
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end simplify_eval_expression_l764_76463


namespace sandrine_washed_160_dishes_l764_76422

-- Define the number of pears picked by Charles
def charlesPears : ℕ := 50

-- Define the number of bananas cooked by Charles as 3 times the number of pears he picked
def charlesBananas : ℕ := 3 * charlesPears

-- Define the number of dishes washed by Sandrine as 10 more than the number of bananas Charles cooked
def sandrineDishes : ℕ := charlesBananas + 10

-- Prove that Sandrine washed 160 dishes
theorem sandrine_washed_160_dishes : sandrineDishes = 160 := by
  -- The proof is omitted
  sorry

end sandrine_washed_160_dishes_l764_76422


namespace sculpture_and_base_height_l764_76473

def height_in_inches (feet: ℕ) (inches: ℕ) : ℕ :=
  feet * 12 + inches

theorem sculpture_and_base_height
  (sculpture_feet: ℕ) (sculpture_inches: ℕ) (base_inches: ℕ)
  (hf: sculpture_feet = 2)
  (hi: sculpture_inches = 10)
  (hb: base_inches = 8)
  : height_in_inches sculpture_feet sculpture_inches + base_inches = 42 :=
by
  -- Placeholder for the proof
  sorry

end sculpture_and_base_height_l764_76473


namespace find_product_xy_l764_76430

theorem find_product_xy (x y : ℝ) 
  (h1 : (9 + 10 + 11 + x + y) / 5 = 10)
  (h2 : ((9 - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (x - 10)^2 + (y - 10)^2) / 5 = 4) :
  x * y = 191 :=
sorry

end find_product_xy_l764_76430


namespace total_apples_eaten_l764_76404

theorem total_apples_eaten : (1 / 2) * 16 + (1 / 3) * 15 + (1 / 4) * 20 = 18 := by
  sorry

end total_apples_eaten_l764_76404


namespace number_of_incorrect_inequalities_l764_76440

theorem number_of_incorrect_inequalities (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) :
  (ite (|a| > |b|) 0 1) + (ite (a < b) 0 1) + (ite (a + b < ab) 0 1) + (ite (a^3 > b^3) 0 1) = 3 :=
sorry

end number_of_incorrect_inequalities_l764_76440


namespace value_of_72_a_in_terms_of_m_and_n_l764_76438

theorem value_of_72_a_in_terms_of_m_and_n (a m n : ℝ) (hm : 2^a = m) (hn : 3^a = n) :
  72^a = m^3 * n^2 :=
by sorry

end value_of_72_a_in_terms_of_m_and_n_l764_76438


namespace largest_multiple_of_7_less_than_neg85_l764_76464

theorem largest_multiple_of_7_less_than_neg85 : ∃ n : ℤ, (∃ k : ℤ, n = 7 * k) ∧ n < -85 ∧ n = -91 :=
by
  sorry

end largest_multiple_of_7_less_than_neg85_l764_76464


namespace problem_1_problem_2_l764_76460

-- Define the factorial and permutation functions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def permutation (n k : ℕ) : ℕ :=
  factorial n / factorial (n - k)

-- Problem 1 statement
theorem problem_1 : permutation 6 6 - permutation 5 5 = 600 := by
  sorry

-- Problem 2 statement
theorem problem_2 : 
  15 * permutation 5 5 * (10^5) + 15 * permutation 4 4 * 11111 =
  15 * permutation 5 5 * (10^5) + 15 * permutation 4 4 * 11111 := by
  sorry

end problem_1_problem_2_l764_76460


namespace range_of_m_l764_76484

noncomputable def proposition (m : ℝ) : Prop := ∀ x : ℝ, 4^x - 2^(x + 1) + m = 0

theorem range_of_m (m : ℝ) (h : ¬¬proposition m) : m ≤ 1 :=
by
  sorry

end range_of_m_l764_76484


namespace sqrt49_times_sqrt25_eq_5sqrt7_l764_76465

noncomputable def sqrt49_times_sqrt25 : ℝ :=
  Real.sqrt (49 * Real.sqrt 25)

theorem sqrt49_times_sqrt25_eq_5sqrt7 :
  sqrt49_times_sqrt25 = 5 * Real.sqrt 7 :=
by
sorry

end sqrt49_times_sqrt25_eq_5sqrt7_l764_76465


namespace calculate_expression_l764_76482

theorem calculate_expression : 1000 * 2.998 * 2.998 * 100 = (29980)^2 := 
by
  sorry

end calculate_expression_l764_76482


namespace find_hundreds_digit_l764_76410

theorem find_hundreds_digit :
  ∃ n : ℕ, (n % 37 = 0) ∧ (n % 173 = 0) ∧ (10000 ≤ n) ∧ (n < 100000) ∧ ((n / 1000) % 10 = 3) ∧ (((n / 100) % 10) = 2) :=
sorry

end find_hundreds_digit_l764_76410


namespace expression_value_range_l764_76475

theorem expression_value_range (a b c d e : ℝ) (h₁ : 0 ≤ a) (h₂ : a ≤ 1) (h₃ : 0 ≤ b) (h₄ : b ≤ 1) (h₅ : 0 ≤ c) (h₆ : c ≤ 1) (h₇ : 0 ≤ d) (h₈ : d ≤ 1) (h₉ : 0 ≤ e) (h₁₀ : e ≤ 1) :
  4 * Real.sqrt (2 / 3) ≤ (Real.sqrt (a^2 + (1 - b)^2 + e^2) + Real.sqrt (b^2 + (1 - c)^2 + e^2) + Real.sqrt (c^2 + (1 - d)^2 + e^2) + Real.sqrt (d^2 + (1 - a)^2 + e^2)) ∧ 
  (Real.sqrt (a^2 + (1 - b)^2 + e^2) + Real.sqrt (b^2 + (1 - c)^2 + e^2) + Real.sqrt (c^2 + (1 - d)^2 + e^2) + Real.sqrt (d^2 + (1 - a)^2 + e^2)) ≤ 8 :=
sorry

end expression_value_range_l764_76475


namespace find_r_l764_76425

variable (p r s : ℝ)

theorem find_r (h : ∀ x : ℝ, (y : ℝ) = x^2 + p * x + r + s → (y = 10 ↔ x = -p / 2)) : r = 10 - s + p^2 / 4 := by
  sorry

end find_r_l764_76425


namespace johns_average_speed_l764_76435

-- Definitions based on conditions
def cycling_distance_uphill := 3 -- in km
def cycling_time_uphill := 45 / 60 -- in hr (45 minutes)

def cycling_distance_downhill := 3 -- in km
def cycling_time_downhill := 15 / 60 -- in hr (15 minutes)

def walking_distance := 2 -- in km
def walking_time := 20 / 60 -- in hr (20 minutes)

-- Definition for total distance traveled
def total_distance := cycling_distance_uphill + cycling_distance_downhill + walking_distance

-- Definition for total time spent traveling
def total_time := cycling_time_uphill + cycling_time_downhill + walking_time

-- Definition for average speed
def average_speed := total_distance / total_time

-- Proof statement
theorem johns_average_speed : average_speed = 6 := by
  sorry

end johns_average_speed_l764_76435


namespace problem_I_problem_II_l764_76429

noncomputable def f (x a : ℝ) : ℝ := 2 / x + a * Real.log x

theorem problem_I (a : ℝ) (h : a > 0) (h' : (2:ℝ) = (1 / (4 / a)) * (a^2) / 8):
  ∃ x : ℝ, f x a = f (1 / 2) a := sorry

theorem problem_II (a : ℝ) (h : a > 0) (h' : ∃ x : ℝ, f x a < 2) :
  (True : Prop) := sorry

end problem_I_problem_II_l764_76429


namespace quadratic_complete_square_l764_76457

theorem quadratic_complete_square (a b c : ℝ) :
  (8*x^2 - 48*x - 288) = a*(x + b)^2 + c → a + b + c = -355 := 
  by
  sorry

end quadratic_complete_square_l764_76457


namespace find_angle_BEC_l764_76462

theorem find_angle_BEC (A B C D E : Type) (angle_A angle_B angle_D angle_DEC angle_C angle_CED angle_BEC : ℝ) 
  (hA : angle_A = 50) (hB : angle_B = 90) (hD : angle_D = 70) (hDEC : angle_DEC = 20)
  (h_quadrilateral_sum: angle_A + angle_B + angle_C + angle_D = 360)
  (h_C : angle_C = 150)
  (h_CED : angle_CED = angle_C - angle_DEC)
  (h_BEC: angle_BEC = 180 - angle_B - angle_CED) : angle_BEC = 110 :=
by
  -- Definitions according to the given problem
  have h1 : angle_C = 360 - (angle_A + angle_B + angle_D) := by sorry
  have h2 : angle_CED = angle_C - angle_DEC := by sorry
  have h3 : angle_BEC = 180 - angle_B - angle_CED := by sorry

  -- Proving the required angle
  have h_goal : angle_BEC = 110 := by
    sorry  -- Actual proof steps go here

  exact h_goal

end find_angle_BEC_l764_76462


namespace min_moves_queens_switch_places_l764_76436

-- Assume a type representing the board positions
inductive Position where
| first_rank | last_rank 

-- Assume a type representing the queens
inductive Queen where
| black | white

-- Function to count minimum moves for switching places
def min_moves_to_switch_places : ℕ :=
  sorry

theorem min_moves_queens_switch_places :
  min_moves_to_switch_places = 23 :=
  sorry

end min_moves_queens_switch_places_l764_76436


namespace worm_length_difference_l764_76414

theorem worm_length_difference
  (worm1 worm2: ℝ)
  (h_worm1: worm1 = 0.8)
  (h_worm2: worm2 = 0.1) :
  worm1 - worm2 = 0.7 :=
by
  -- starting the proof
  sorry

end worm_length_difference_l764_76414


namespace farm_field_area_l764_76455

theorem farm_field_area
  (plough_per_day_planned plough_per_day_actual fields_left : ℕ)
  (D : ℕ) 
  (condition1 : plough_per_day_planned = 100)
  (condition2 : plough_per_day_actual = 85)
  (condition3 : fields_left = 40)
  (additional_days : ℕ) 
  (condition4 : additional_days = 2)
  (initial_days : D + additional_days = 85 * (D + 2) + 40) :
  (100 * D + fields_left = 1440) :=
by
  sorry

end farm_field_area_l764_76455


namespace problem1_problem2_l764_76487

variable (k : ℝ)

-- Definitions of proposition p and q
def p (k : ℝ) : Prop := ∀ x : ℝ, x^2 - k*x + 2*k + 5 ≥ 0

def q (k : ℝ) : Prop := (4 - k > 0) ∧ (1 - k < 0)

-- Theorem statements based on the proof problem
theorem problem1 (hq : q k) : 1 < k ∧ k < 4 :=
by sorry

theorem problem2 (hp_q : p k ∨ q k) (hp_and_q_false : ¬(p k ∧ q k)) : 
  (-2 ≤ k ∧ k ≤ 1) ∨ (4 ≤ k ∧ k ≤ 10) :=
by sorry

end problem1_problem2_l764_76487


namespace quarterly_to_annual_interest_rate_l764_76428

theorem quarterly_to_annual_interest_rate :
  ∃ s : ℝ, (1 + 0.02)^4 = 1 + s / 100 ∧ abs (s - 8.24) < 0.01 :=
by
  sorry

end quarterly_to_annual_interest_rate_l764_76428


namespace proof_equivalent_triples_l764_76445

noncomputable def valid_triples := 
  { (a, b, c) : ℝ × ℝ × ℝ |
    a * b + b * c + c * a = 1 ∧
    a^2 * b + c = b^2 * c + a ∧
    a^2 * b + c = c^2 * a + b }

noncomputable def desired_solutions := 
  { (a, b, c) |
    (a = 0 ∧ b = 1 ∧ c = 1) ∨
    (a = 0 ∧ b = 1 ∧ c = -1) ∨
    (a = 0 ∧ b = -1 ∧ c = 1) ∨
    (a = 0 ∧ b = -1 ∧ c = -1) ∨

    (a = 1 ∧ b = 1 ∧ c = 0) ∨
    (a = 1 ∧ b = -1 ∧ c = 0) ∨
    (a = -1 ∧ b = 1 ∧ c = 0) ∨
    (a = -1 ∧ b = -1 ∧ c = 0) ∨

    (a = 1 ∧ b = 0 ∧ c = 1) ∨
    (a = 1 ∧ b = 0 ∧ c = -1) ∨
    (a = -1 ∧ b = 0 ∧ c = 1) ∨
    (a = -1 ∧ b = 0 ∧ c = -1) ∨

    ((a = (Real.sqrt 3) / 3 ∧ b = (Real.sqrt 3) / 3 ∧ 
      c = (Real.sqrt 3) / 3) ∨
     (a = -(Real.sqrt 3) / 3 ∧ b = -(Real.sqrt 3) / 3 ∧ 
      c = -(Real.sqrt 3) / 3)) }

theorem proof_equivalent_triples :
  valid_triples = desired_solutions :=
sorry

end proof_equivalent_triples_l764_76445


namespace josh_ribbon_shortfall_l764_76467

-- Define the total amount of ribbon Josh has
def total_ribbon : ℝ := 18

-- Define the number of gifts
def num_gifts : ℕ := 6

-- Define the ribbon requirements for each gift
def ribbon_per_gift_wrapping : ℝ := 2
def ribbon_per_bow : ℝ := 1.5
def ribbon_per_tag : ℝ := 0.25
def ribbon_per_trim : ℝ := 0.5

-- Calculate the total ribbon required for all the tasks
def total_ribbon_needed : ℝ :=
  (ribbon_per_gift_wrapping * num_gifts) +
  (ribbon_per_bow * num_gifts) +
  (ribbon_per_tag * num_gifts) +
  (ribbon_per_trim * num_gifts)

-- Calculate the ribbon shortfall
def ribbon_shortfall : ℝ :=
  total_ribbon_needed - total_ribbon

-- Prove that Josh will be short by 7.5 yards of ribbon
theorem josh_ribbon_shortfall : ribbon_shortfall = 7.5 := by
  sorry

end josh_ribbon_shortfall_l764_76467


namespace prob_dominant_trait_one_child_prob_at_least_one_dominant_trait_two_children_l764_76432

-- Define the probability of a genotype given two mixed genotype (rd) parents producing a child.
def prob_genotype_dd : ℚ := (1/2) * (1/2)
def prob_genotype_rr : ℚ := (1/2) * (1/2)
def prob_genotype_rd : ℚ := 2 * (1/2) * (1/2)

-- Assertion that the probability of a child displaying the dominant characteristic (dd or rd) is 3/4.
theorem prob_dominant_trait_one_child : 
  prob_genotype_dd + prob_genotype_rd = 3/4 := sorry

-- Define the probability of two children both being rr.
def prob_both_rr_two_children : ℚ := prob_genotype_rr * prob_genotype_rr

-- Assertion that the probability of at least one of two children displaying the dominant characteristic is 15/16.
theorem prob_at_least_one_dominant_trait_two_children : 
  1 - prob_both_rr_two_children = 15/16 := sorry

end prob_dominant_trait_one_child_prob_at_least_one_dominant_trait_two_children_l764_76432


namespace vlad_taller_by_41_inches_l764_76424

/-- Vlad's height is 6 feet and 3 inches. -/
def vlad_height_feet : ℕ := 6

def vlad_height_inches : ℕ := 3

/-- Vlad's sister's height is 2 feet and 10 inches. -/
def sister_height_feet : ℕ := 2

def sister_height_inches : ℕ := 10

/-- There are 12 inches in a foot. -/
def inches_in_a_foot : ℕ := 12

/-- Convert height in feet and inches to total inches. -/
def convert_to_inches (feet inches : ℕ) : ℕ :=
  feet * inches_in_a_foot + inches

/-- Proof that Vlad is 41 inches taller than his sister. -/
theorem vlad_taller_by_41_inches : convert_to_inches vlad_height_feet vlad_height_inches - convert_to_inches sister_height_feet sister_height_inches = 41 :=
by
  -- Start the proof
  sorry

end vlad_taller_by_41_inches_l764_76424


namespace slips_with_3_count_l764_76499

def number_of_slips_with_3 (x : ℕ) : Prop :=
  let total_slips := 15
  let expected_value := 4.6
  let prob_3 := (x : ℚ) / total_slips
  let prob_8 := (total_slips - x : ℚ) / total_slips
  let E := prob_3 * 3 + prob_8 * 8
  E = expected_value

theorem slips_with_3_count : ∃ x : ℕ, number_of_slips_with_3 x ∧ x = 10 :=
by
  sorry

end slips_with_3_count_l764_76499


namespace highest_power_of_3_divides_l764_76490

def A_n (n : ℕ) : ℕ := (10^(3^n) - 1) / 9

theorem highest_power_of_3_divides (n : ℕ) : ∃ k : ℕ, A_n n = 3^n * k ∧ ¬ (3 * A_n n = 3^(n+1) * k)
:= by
  sorry

end highest_power_of_3_divides_l764_76490


namespace sum_of_sequence_l764_76479

def sequence_t (n : ℕ) : ℚ :=
  if n % 2 = 1 then 1 / 7^n else 2 / 7^n

theorem sum_of_sequence :
  (∑' n:ℕ, sequence_t (n + 1)) = 3 / 16 :=
by
  sorry

end sum_of_sequence_l764_76479


namespace combination_property_l764_76421

theorem combination_property (x : ℕ) (hx : 2 * x - 1 ≤ 11 ∧ x ≤ 11) :
  (Nat.choose 11 (2 * x - 1) = Nat.choose 11 x) → (x = 1 ∨ x = 4) :=
by
  sorry

end combination_property_l764_76421


namespace Melanie_dimes_and_coins_l764_76497

-- Define all given conditions
def d1 : Nat := 7
def d2 : Nat := 8
def d3 : Nat := 4
def r : Float := 2.5

-- State the theorem to prove
theorem Melanie_dimes_and_coins :
  let d_t := d1 + d2 + d3
  let c_t := Float.ofNat d_t * r
  d_t = 19 ∧ c_t = 47.5 :=
by
  sorry

end Melanie_dimes_and_coins_l764_76497


namespace passes_through_point_l764_76433

theorem passes_through_point (a : ℝ) (h : a > 0) (h2 : a ≠ 1) : 
  (2, 1) ∈ {p : ℝ × ℝ | ∃ a, p.snd = a * p.fst - 2} :=
sorry

end passes_through_point_l764_76433


namespace seventh_rack_dvds_l764_76415

def rack_dvds : ℕ → ℕ
| 0 => 3
| 1 => 4
| n + 2 => ((rack_dvds (n + 1)) - (rack_dvds n)) * 2 + (rack_dvds (n + 1))

theorem seventh_rack_dvds : rack_dvds 6 = 66 := 
by
  sorry

end seventh_rack_dvds_l764_76415


namespace find_p_l764_76485

noncomputable def p (x1 x2 x3 x4 n : ℝ) :=
  (x1 + x3) * (x2 + x3) + (x1 + x4) * (x2 + x4)

theorem find_p (x1 x2 x3 x4 n : ℝ) (h1 : x1 ≠ x2)
(h2 : (x1 + x3) * (x1 + x4) = n - 10)
(h3 : (x2 + x3) * (x2 + x4) = n - 10) :
  p x1 x2 x3 x4 n = n - 20 :=
sorry

end find_p_l764_76485


namespace calculation_correct_l764_76441

theorem calculation_correct : 67897 * 67898 - 67896 * 67899 = 2 := by
  sorry

end calculation_correct_l764_76441


namespace intersection_point_a_l764_76491

theorem intersection_point_a : ∃ (x y : ℝ), y = 4 * x - 32 ∧ y = -6 * x + 8 ∧ x = 4 ∧ y = -16 :=
sorry

end intersection_point_a_l764_76491


namespace eggs_left_in_box_l764_76411

theorem eggs_left_in_box (initial_eggs : ℕ) (taken_eggs : ℕ) (remaining_eggs : ℕ) : 
  initial_eggs = 47 → taken_eggs = 5 → remaining_eggs = initial_eggs - taken_eggs → remaining_eggs = 42 :=
by
  sorry

end eggs_left_in_box_l764_76411


namespace parking_lot_motorcycles_l764_76442

theorem parking_lot_motorcycles
  (x y : ℕ)
  (h1 : x + y = 24)
  (h2 : 3 * x + 4 * y = 86) : x = 10 :=
by
  sorry

end parking_lot_motorcycles_l764_76442


namespace least_number_with_remainder_l764_76476

theorem least_number_with_remainder (N : ℕ) : (∃ k : ℕ, N = 12 * k + 4) → N = 256 :=
by
  intro h
  sorry

end least_number_with_remainder_l764_76476


namespace sum_of_x_y_l764_76448

theorem sum_of_x_y (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_x_y_l764_76448


namespace log_of_y_pow_x_eq_neg4_l764_76494

theorem log_of_y_pow_x_eq_neg4 (x y : ℝ) (h : |x - 8 * y| + (4 * y - 1) ^ 2 = 0) : 
  Real.logb 2 (y ^ x) = -4 :=
sorry

end log_of_y_pow_x_eq_neg4_l764_76494


namespace geom_seq_inequality_l764_76446

-- Define S_n as a.sum of the first n terms of a geometric sequence with ratio q and first term a_1
noncomputable def S (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
if h : q = 1 then (n + 1) * a_1 else a_1 * (1 - q ^ (n + 1)) / (1 - q)

-- Define a_n for geometric sequence
noncomputable def a_seq (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
a_1 * q ^ n

-- The main theorem to prove
theorem geom_seq_inequality (a_1 : ℝ) (q : ℝ) (n : ℕ) (hq_pos : 0 < q) :
  S a_1 q (n + 1) * a_seq a_1 q n > S a_1 q n * a_seq a_1 q (n + 1) :=
by {
  sorry -- Placeholder for actual proof
}

end geom_seq_inequality_l764_76446


namespace volume_difference_l764_76427

-- Define the dimensions of the first bowl
def length1 : ℝ := 14
def width1 : ℝ := 16
def height1 : ℝ := 9

-- Define the dimensions of the second bowl
def length2 : ℝ := 14
def width2 : ℝ := 16
def height2 : ℝ := 4

-- Define the volumes of the two bowls assuming they are rectangular prisms
def volume1 : ℝ := length1 * width1 * height1
def volume2 : ℝ := length2 * width2 * height2

-- Statement to prove the volume difference
theorem volume_difference : volume1 - volume2 = 1120 := by
  sorry

end volume_difference_l764_76427


namespace gina_minutes_of_netflix_l764_76495

-- Define the conditions given in the problem
def gina_chooses_three_times_as_often (g s : ℕ) : Prop :=
  g = 3 * s

def total_shows_watched (g s : ℕ) : Prop :=
  g + s = 24

def duration_per_show : ℕ := 50

-- The theorem that encapsulates the problem statement and the correct answer
theorem gina_minutes_of_netflix (g s : ℕ) (h1 : gina_chooses_three_times_as_often g s) 
    (h2 : total_shows_watched g s) :
    g * duration_per_show = 900 :=
by
  sorry

end gina_minutes_of_netflix_l764_76495


namespace avg_cost_of_6_toys_l764_76486

-- Define the given conditions
def dhoni_toys_count : ℕ := 5
def dhoni_toys_avg_cost : ℝ := 10
def sixth_toy_cost : ℝ := 16
def sales_tax_rate : ℝ := 0.10

-- Define the supposed answer
def supposed_avg_cost : ℝ := 11.27

-- Define the problem in Lean 4 statement
theorem avg_cost_of_6_toys :
  (dhoni_toys_count * dhoni_toys_avg_cost + sixth_toy_cost * (1 + sales_tax_rate)) / (dhoni_toys_count + 1) = supposed_avg_cost :=
by
  -- Proof goes here, replace with actual proof
  sorry

end avg_cost_of_6_toys_l764_76486


namespace marble_count_l764_76417

-- Define the variables for the number of marbles
variables (o p y : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := o = 1.3 * p
def condition2 : Prop := y = 1.5 * o

-- Define the total number of marbles based on the conditions
def total_marbles : ℝ := o + p + y

-- The theorem statement that needs to be proved
theorem marble_count (h1 : condition1 o p) (h2 : condition2 o y) : total_marbles o p y = 3.269 * o :=
by sorry

end marble_count_l764_76417


namespace problem1_problem2_l764_76477

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2 * a * x - 3

theorem problem1 (a : ℝ) (h : f a (a + 1) - f a a = 9) : a = 2 :=
by sorry

theorem problem2 (a : ℝ) (h : ∃ x, f a x = -4 ∧ ∀ y, f a y ≥ -4) : a = 1 ∨ a = -1 :=
by sorry

end problem1_problem2_l764_76477


namespace tangent_line_value_l764_76451

theorem tangent_line_value {a : ℝ} (h : a > 0) : 
  (∀ θ ρ, (ρ * (Real.cos θ + Real.sin θ) = a) → (ρ = 2 * Real.cos θ)) → 
  a = 1 + Real.sqrt 2 :=
sorry

end tangent_line_value_l764_76451


namespace problem1_problem2_l764_76431

noncomputable def f (x a b : ℝ) : ℝ := x^2 - (a+1)*x + b

theorem problem1 (h : ∀ x : ℝ, f x (-4) (-10) < 0 ↔ -5 < x ∧ x < 2) : f x (-4) (-10) < 0 :=
sorry

theorem problem2 (a : ℝ) : 
  (a > 1 → ∀ x : ℝ, f x a a > 0 ↔ x < 1 ∨ x > a) ∧
  (a = 1 → ∀ x : ℝ, f x a a > 0 ↔ x ≠ 1) ∧
  (a < 1 → ∀ x : ℝ, f x a a > 0 ↔ x < a ∨ x > 1) :=
sorry

end problem1_problem2_l764_76431


namespace essentially_different_proportions_l764_76401

theorem essentially_different_proportions (x y z t : α) [DecidableEq α] 
  (h1 : x ≠ y) (h2 : x ≠ z) (h3 : x ≠ t) (h4 : y ≠ z) (h5 : y ≠ t) (h6 : z ≠ t) : 
  ∃ n : ℕ, n = 3 := by
  sorry

end essentially_different_proportions_l764_76401


namespace systematic_sampling_seventeenth_group_l764_76444

theorem systematic_sampling_seventeenth_group :
  ∀ (total_students : ℕ) (sample_size : ℕ) (first_number : ℕ) (interval : ℕ),
  total_students = 800 →
  sample_size = 50 →
  first_number = 8 →
  interval = total_students / sample_size →
  first_number + 16 * interval = 264 :=
by
  intros total_students sample_size first_number interval h1 h2 h3 h4
  sorry

end systematic_sampling_seventeenth_group_l764_76444


namespace sequence_equiv_l764_76456

theorem sequence_equiv (n : ℕ) (hn : n > 0) : ∃ (p : ℕ), p > 0 ∧ (4 * p + 5 = (3^n)^2) :=
by
  sorry

end sequence_equiv_l764_76456


namespace find_m_l764_76489

noncomputable def vec_add (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem find_m (m : ℝ) (h : dot_product (vec_add (-1, 2) (m, 1)) (-1, 2) = 0) : m = 7 :=
  by 
  sorry

end find_m_l764_76489


namespace probability_bijection_l764_76458

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 2, 3, 4, 5}

theorem probability_bijection : 
  let total_mappings := 5^4
  let bijections := 5 * 4 * 3 * 2
  let probability := bijections / total_mappings
  probability = 24 / 125 := 
by
  sorry

end probability_bijection_l764_76458


namespace simplify_expression_l764_76439

theorem simplify_expression :
  (Real.sqrt (Real.sqrt (81)) - Real.sqrt (8 + 1 / 2)) ^ 2 = (35 / 2) - 3 * Real.sqrt 34 :=
by
  sorry

end simplify_expression_l764_76439


namespace range_of_a_l764_76407

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - 3 < 0 → x > a) ∧ (∃ x : ℝ, x > a ∧ ¬(x^2 - 2 * x - 3 < 0)) → a ≤ -1 :=
by
  sorry

end range_of_a_l764_76407


namespace toothpicks_15th_stage_l764_76469
-- Import the required library

-- Define the arithmetic sequence based on the provided conditions.
def toothpicks (n : ℕ) : ℕ :=
  if n = 1 then 5 else 3 * (n - 1) + 5

-- State the theorem
theorem toothpicks_15th_stage : toothpicks 15 = 47 :=
by {
  -- Provide the proof here, but currently using sorry as instructed
  sorry
}

end toothpicks_15th_stage_l764_76469


namespace satisfies_conditions_l764_76453

noncomputable def m := 29 / 3

def real_part (m : ℝ) : ℝ := m^2 - 8*m + 15
def imag_part (m : ℝ) : ℝ := m^2 - 5*m - 14

theorem satisfies_conditions (m : ℝ) 
  (real_cond : m < 3 ∨ m > 5) 
  (imag_cond : -2 < m ∧ m < 7)
  (line_cond : real_part m = imag_part m): 
  m = 29 / 3 :=
by {
  sorry
}

end satisfies_conditions_l764_76453


namespace sum_of_numbers_l764_76472

-- Definitions for the numbers involved
def n1 : Nat := 1235
def n2 : Nat := 2351
def n3 : Nat := 3512
def n4 : Nat := 5123

-- Proof statement
theorem sum_of_numbers :
  n1 + n2 + n3 + n4 = 12221 := by
  sorry

end sum_of_numbers_l764_76472


namespace magician_method_N_2k_magician_method_values_l764_76420

-- (a) Prove that if there is a method for N = k, then there is a method for N = 2k.
theorem magician_method_N_2k (k : ℕ) (method_k : Prop) : 
  (∃ method_N_k : Prop, method_k → method_N_k) → 
  (∃ method_N_2k : Prop, method_k → method_N_2k) :=
sorry

-- (b) Find all values of N for which the magician and the assistant have a method.
theorem magician_method_values (N : ℕ) : 
  (∃ method : Prop, method) ↔ (∃ m : ℕ, N = 2^m) :=
sorry

end magician_method_N_2k_magician_method_values_l764_76420


namespace compute_fraction_l764_76470

theorem compute_fraction :
  ((1/3)^4 * (1/5) = (1/405)) :=
by
  sorry

end compute_fraction_l764_76470


namespace domain_of_sqrt_log_l764_76419

theorem domain_of_sqrt_log {x : ℝ} : (2 < x ∧ x ≤ 5 / 2) ↔ 
  (5 - 2 * x > 0 ∧ 0 ≤ Real.logb (1 / 2) (5 - 2 * x)) :=
sorry

end domain_of_sqrt_log_l764_76419


namespace distance_point_parabola_focus_l764_76461

theorem distance_point_parabola_focus (P : ℝ × ℝ) (x y : ℝ) (hP : P = (3, y)) (h_parabola : y^2 = 4 * 3) :
    dist P (0, -1) = 4 :=
by
  sorry

end distance_point_parabola_focus_l764_76461


namespace radius_of_circle_l764_76466

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = Real.pi * r ^ 2) : r = 6 := 
by 
  sorry

end radius_of_circle_l764_76466


namespace simplify_expression_l764_76480

theorem simplify_expression (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ 2) (h4 : x ≠ 3) :
  ((x - 5) / (x - 3) - ((x^2 + 2 * x + 1) / (x^2 + x)) / ((x + 1) / (x - 2)) = 
  -6 / (x^2 - 3 * x)) :=
by
  sorry

end simplify_expression_l764_76480


namespace minimum_period_l764_76416

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem minimum_period (ω : ℝ) (hω : ω > 0) 
  (h : ∀ x1 x2 : ℝ, |f ω x1 - f ω x2| = 2 → |x1 - x2| = Real.pi / 2) :
  ∃ T > 0, ∀ x : ℝ, f ω (x + T) = f ω x ∧ T = Real.pi := sorry

end minimum_period_l764_76416


namespace total_votes_cast_l764_76406

theorem total_votes_cast (V : ℝ) (h1 : V > 0) (h2 : 0.35 * V = candidate_votes) (h3 : candidate_votes + 2400 = rival_votes) (h4 : candidate_votes + rival_votes = V) : V = 8000 := 
by
  sorry

end total_votes_cast_l764_76406


namespace base6_sum_l764_76437

-- Define each of the numbers in base 6
def base6_555 : ℕ := 5 * 6^2 + 5 * 6^1 + 5 * 6^0
def base6_55 : ℕ := 5 * 6^1 + 5 * 6^0
def base6_5 : ℕ := 5 * 6^0
def base6_1103 : ℕ := 1 * 6^3 + 1 * 6^2 + 0 * 6^1 + 3 * 6^0 

-- The problem statement is to prove the sum equals the expected result in base 6
theorem base6_sum : base6_555 + base6_55 + base6_5 = base6_1103 :=
by
  sorry

end base6_sum_l764_76437


namespace comic_stack_ways_l764_76459

-- Define the factorial function for convenience
noncomputable def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- Conditions: Define the number of each type of comic book
def batman_comics := 7
def superman_comics := 4
def wonder_woman_comics := 5
def flash_comics := 3

-- The total number of comic books
def total_comics := batman_comics + superman_comics + wonder_woman_comics + flash_comics

-- Proof problem: The number of ways to stack the comics
theorem comic_stack_ways :
  (factorial batman_comics) * (factorial superman_comics) * (factorial wonder_woman_comics) * (factorial flash_comics) * (factorial 4) = 1102489600 := sorry

end comic_stack_ways_l764_76459


namespace eliot_account_balance_l764_76434

variable (A E : ℝ)

-- Condition 1: Al has more money than Eliot.
axiom h1 : A > E

-- Condition 2: The difference between their accounts is 1/12 of the sum of their accounts.
axiom h2 : A - E = (1 / 12) * (A + E)

-- Condition 3: If Al's account were increased by 10% and Eliot's by 20%, Al would have exactly $21 more than Eliot.
axiom h3 : 1.1 * A = 1.2 * E + 21

-- Conjecture: Eliot has $210 in his account.
theorem eliot_account_balance : E = 210 :=
by
  sorry

end eliot_account_balance_l764_76434


namespace sum_of_coordinates_reflection_l764_76488

theorem sum_of_coordinates_reflection (y : ℝ) : 
  let C := (3, y)
  let D := (3, -y)
  (C.1 + C.2 + D.1 + D.2) = 6 :=
by
  let C := (3, y)
  let D := (3, -y)
  have h : C.1 + C.2 + D.1 + D.2 = 6 := sorry
  exact h

end sum_of_coordinates_reflection_l764_76488


namespace average_headcount_spring_terms_l764_76498

def spring_headcount_02_03 := 10900
def spring_headcount_03_04 := 10500
def spring_headcount_04_05 := 10700

theorem average_headcount_spring_terms :
  (spring_headcount_02_03 + spring_headcount_03_04 + spring_headcount_04_05) / 3 = 10700 := by
  sorry

end average_headcount_spring_terms_l764_76498


namespace Misha_earnings_needed_l764_76402

-- Define the conditions and the goal in Lean 4
def Misha_current_dollars : ℕ := 34
def Misha_target_dollars : ℕ := 47

theorem Misha_earnings_needed : Misha_target_dollars - Misha_current_dollars = 13 := by
  sorry

end Misha_earnings_needed_l764_76402


namespace spinsters_count_l764_76405

theorem spinsters_count (S C : ℕ) (h_ratio : S / C = 2 / 7) (h_diff : C = S + 55) : S = 22 :=
by
  sorry

end spinsters_count_l764_76405


namespace problem_solution_l764_76447

theorem problem_solution :
  ∃ a b c d : ℚ, 
  4 * a + 2 * b + 5 * c + 8 * d = 67 ∧ 
  4 * (d + c) = b ∧ 
  2 * b + 3 * c = a ∧ 
  c + 1 = d ∧ 
  a * b * c * d = (1201 * 572 * 19 * 124) / (105 ^ 4) :=
sorry

end problem_solution_l764_76447


namespace basketball_court_perimeter_l764_76413

variables {Width Length : ℕ}

def width := 17
def length := 31

def perimeter (width length : ℕ) := 2 * (length + width)

theorem basketball_court_perimeter : 
  perimeter width length = 96 :=
sorry

end basketball_court_perimeter_l764_76413


namespace sum_of_three_exists_l764_76418

theorem sum_of_three_exists (n : ℤ) (X : Finset ℤ) 
  (hX_card : X.card = n + 2) 
  (hX_abs : ∀ x ∈ X, abs x ≤ n) : 
  ∃ a b c : ℤ, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ c = a + b := 
by 
  sorry

end sum_of_three_exists_l764_76418
