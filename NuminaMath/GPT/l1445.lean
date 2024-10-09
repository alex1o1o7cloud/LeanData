import Mathlib

namespace always_in_range_l1445_144599

noncomputable def g (x k : ℝ) : ℝ := x^2 + 2 * k * x + 1

theorem always_in_range (k : ℝ) : 
  ∃ x : ℝ, g x k = 3 :=
by
  sorry

end always_in_range_l1445_144599


namespace flowers_given_to_mother_l1445_144504

-- Definitions based on conditions:
def Alissa_flowers : Nat := 16
def Melissa_flowers : Nat := 16
def flowers_left : Nat := 14

-- The proof problem statement:
theorem flowers_given_to_mother :
  Alissa_flowers + Melissa_flowers - flowers_left = 18 := by
  sorry

end flowers_given_to_mother_l1445_144504


namespace profit_maximization_problem_l1445_144585

-- Step 1: Define the data points and linear function
def data_points : List (ℝ × ℝ) := [(65, 70), (70, 60), (75, 50), (80, 40)]

-- Step 2: Define the linear function between y and x
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Step 3: Define cost and profit function
def cost_per_kg : ℝ := 60
def profit_function (y x : ℝ) : ℝ := y * (x - cost_per_kg)

-- Step 4: The main problem statement
theorem profit_maximization_problem :
  ∃ (k b : ℝ), 
  (∀ (x₁ x₂ : ℝ), (x₁, y₁) ∈ data_points ∧ (x₂, y₂) ∈ data_points → linear_function k b x₁ = y₁ ∧ linear_function k b x₂ = y₂) ∧
  ∃ (x : ℝ), profit_function (linear_function k b x) x = 600 ∧
  ∀ x : ℝ, -2 * x^2 + 320 * x - 12000 ≤ -2 * 80^2 + 320 * 80 - 12000
  :=
sorry

end profit_maximization_problem_l1445_144585


namespace find_k_of_inverse_proportion_l1445_144517

theorem find_k_of_inverse_proportion (k x y : ℝ) (h : y = k / x) (hx : x = 2) (hy : y = 6) : k = 12 :=
by
  sorry

end find_k_of_inverse_proportion_l1445_144517


namespace cube_painted_faces_l1445_144545

noncomputable def painted_faces_count (side_length painted_cubes_edge middle_cubes_edge : ℕ) : ℕ :=
  let total_corners := 8
  let total_edges := 12
  total_corners + total_edges * middle_cubes_edge

theorem cube_painted_faces :
  ∀ side_length : ℕ, side_length = 4 →
  ∀ painted_cubes_edge middle_cubes_edge total_cubes : ℕ,
  total_cubes = side_length * side_length * side_length →
  painted_cubes_edge = 3 →
  middle_cubes_edge = 2 →
  painted_faces_count side_length painted_cubes_edge middle_cubes_edge = 32 := sorry

end cube_painted_faces_l1445_144545


namespace sequence_term_position_l1445_144575

theorem sequence_term_position (n : ℕ) (h : 2 * Real.sqrt 5 = Real.sqrt (3 * n - 1)) : n = 7 :=
sorry

end sequence_term_position_l1445_144575


namespace balloons_initial_count_l1445_144551

theorem balloons_initial_count (x : ℕ) (h : x + 13 = 60) : x = 47 :=
by
  -- proof skipped
  sorry

end balloons_initial_count_l1445_144551


namespace distance_of_hyperbola_vertices_l1445_144535

-- Define the hyperbola equation condition
def hyperbola : Prop := ∃ (y x : ℝ), (y^2 / 16) - (x^2 / 9) = 1

-- Define a variable for the distance between the vertices
def distance_between_vertices (a : ℝ) : ℝ := 2 * a

-- The main statement to be proved
theorem distance_of_hyperbola_vertices :
  hyperbola → distance_between_vertices 4 = 8 :=
by
  intro h
  sorry

end distance_of_hyperbola_vertices_l1445_144535


namespace cutting_stick_ways_l1445_144526

theorem cutting_stick_ways :
  ∃ (s : Finset (ℕ × ℕ)), 
  (∀ a ∈ s, 2 * a.1 + 3 * a.2 = 14) ∧
  s.card = 2 := 
by
  sorry

end cutting_stick_ways_l1445_144526


namespace average_monthly_growth_rate_correct_l1445_144521

theorem average_monthly_growth_rate_correct:
  (∃ x : ℝ, 30000 * (1 + x)^2 = 36300) ↔ 3 * (1 + x)^2 = 3.63 := 
by {
  sorry -- proof placeholder
}

end average_monthly_growth_rate_correct_l1445_144521


namespace repeating_decimal_as_fraction_l1445_144564

def repeating_decimal := 567 / 999

theorem repeating_decimal_as_fraction : repeating_decimal = 21 / 37 := by
  sorry

end repeating_decimal_as_fraction_l1445_144564


namespace num_men_in_second_group_l1445_144579

def total_work_hours_week (men: ℕ) (hours_per_day: ℕ) (days_per_week: ℕ) : ℕ :=
  men * hours_per_day * days_per_week

def earnings_per_man_hour (total_earnings: ℕ) (total_work_hours: ℕ) : ℚ :=
  total_earnings / total_work_hours

def required_man_hours (total_earnings: ℕ) (earnings_per_hour: ℚ) : ℚ :=
  total_earnings / earnings_per_hour

def number_of_men (total_man_hours: ℚ) (hours_per_day: ℕ) (days_per_week: ℕ) : ℚ :=
  total_man_hours / (hours_per_day * days_per_week)

theorem num_men_in_second_group :
  let hours_per_day_1 := 10
  let hours_per_day_2 := 6
  let days_per_week := 7
  let men_1 := 4
  let earnings_1 := 1000
  let earnings_2 := 1350
  let work_hours_1 := total_work_hours_week men_1 hours_per_day_1 days_per_week
  let rate_1 := earnings_per_man_hour earnings_1 work_hours_1
  let work_hours_2 := required_man_hours earnings_2 rate_1
  number_of_men work_hours_2 hours_per_day_2 days_per_week = 9 := by
  sorry

end num_men_in_second_group_l1445_144579


namespace ratio_of_sides_l1445_144589

theorem ratio_of_sides (s r : ℝ) (h : s^2 = 2 * r^2 * Real.sqrt 2) : r / s = 1 / Real.sqrt (2 * Real.sqrt 2) := 
by
  sorry

end ratio_of_sides_l1445_144589


namespace incorrect_transformation_l1445_144570

theorem incorrect_transformation (a b : ℕ) (h : a ≠ 0 ∧ b ≠ 0) (h_eq : a / 2 = b / 3) :
  (∃ k : ℕ, 2 * a = 3 * b → false) ∧ 
  (a / b = 2 / 3) ∧ 
  (b / a = 3 / 2) ∧
  (3 * a = 2 * b) :=
by
  sorry

end incorrect_transformation_l1445_144570


namespace pier_to_village_trip_l1445_144549

theorem pier_to_village_trip :
  ∃ (x t : ℝ), 
  (x / 10 + x / 8 = t + 1 / 60) ∧
  (5 * t / 2 + 4 * t / 2 = x) ∧
  (x = 6) ∧
  (t = 4 / 3) :=
by
  sorry

end pier_to_village_trip_l1445_144549


namespace anya_more_erasers_l1445_144584

theorem anya_more_erasers (anya_erasers andrea_erasers : ℕ)
  (h1 : anya_erasers = 4 * andrea_erasers)
  (h2 : andrea_erasers = 4) :
  anya_erasers - andrea_erasers = 12 := by
  sorry

end anya_more_erasers_l1445_144584


namespace equivalent_problem_l1445_144510

-- Definitions that correspond to conditions
def valid_n (n : ℕ) : Prop := n < 13 ∧ (4 * n) % 13 = 1

-- The equivalent proof problem
theorem equivalent_problem (n : ℕ) (h : valid_n n) : ((3 ^ n) ^ 4 - 3) % 13 = 6 := by
  sorry

end equivalent_problem_l1445_144510


namespace sum_of_consecutive_even_integers_divisible_by_three_l1445_144507

theorem sum_of_consecutive_even_integers_divisible_by_three (n : ℤ) : 
  ∃ p : ℤ, Prime p ∧ p = 3 ∧ p ∣ (n + (n + 2) + (n + 4)) :=
by 
  sorry

end sum_of_consecutive_even_integers_divisible_by_three_l1445_144507


namespace average_income_l1445_144529

/-- The daily incomes of the cab driver over 5 days. --/
def incomes : List ℕ := [400, 250, 650, 400, 500]

/-- Prove that the average income of the cab driver over these 5 days is $440. --/
theorem average_income : (incomes.sum / incomes.length) = 440 := by
  sorry

end average_income_l1445_144529


namespace rectangular_block_height_l1445_144505

theorem rectangular_block_height (l w h : ℕ) 
  (volume_eq : l * w * h = 42) 
  (perimeter_eq : 2 * l + 2 * w = 18) : 
  h = 3 :=
by
  sorry

end rectangular_block_height_l1445_144505


namespace length_of_second_train_is_approximately_159_98_l1445_144594

noncomputable def length_of_second_train : ℝ :=
  let length_first_train := 110 -- meters
  let speed_first_train := 60 -- km/hr
  let speed_second_train := 40 -- km/hr
  let time_to_cross := 9.719222462203025 -- seconds
  let km_per_hr_to_m_per_s := 5 / 18 -- conversion factor from km/hr to m/s
  let relative_speed := (speed_first_train + speed_second_train) * km_per_hr_to_m_per_s -- relative speed in m/s
  let total_distance := relative_speed * time_to_cross -- total distance covered
  total_distance - length_first_train -- length of the second train

theorem length_of_second_train_is_approximately_159_98 :
  abs (length_of_second_train - 159.98) < 0.01 := 
by
  sorry -- Placeholder for the actual proof

end length_of_second_train_is_approximately_159_98_l1445_144594


namespace range_of_a_l1445_144548

theorem range_of_a (a : ℝ) :
  (∀ (x y : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ y ∧ y ≤ 4 → 2 * x^2 - 2 * a * x * y + y^2 ≥ 0) →
  a ≤ Real.sqrt 2 :=
sorry

end range_of_a_l1445_144548


namespace number_of_extreme_points_l1445_144534

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + 3 * x^2 + 4 * x - a

theorem number_of_extreme_points (a : ℝ) : 
  (∀ x : ℝ, (3 * x^2 + 6 * x + 4) > 0) →
  0 = 0 :=
by
  intro h
  sorry

end number_of_extreme_points_l1445_144534


namespace probability_of_winning_l1445_144586

variable (P_A P_B P_C P_M_given_A P_M_given_B P_M_given_C : ℝ)

theorem probability_of_winning :
  P_A = 0.6 →
  P_B = 0.3 →
  P_C = 0.1 →
  P_M_given_A = 0.1 →
  P_M_given_B = 0.2 →
  P_M_given_C = 0.3 →
  (P_A * P_M_given_A + P_B * P_M_given_B + P_C * P_M_given_C) = 0.15 :=
by sorry

end probability_of_winning_l1445_144586


namespace find_ratio_of_d1_and_d2_l1445_144550

theorem find_ratio_of_d1_and_d2
  (x y d1 d2 : ℝ)
  (h1 : x + 4 * d1 = y)
  (h2 : x + 5 * d2 = y)
  (h3 : d1 ≠ 0)
  (h4 : d2 ≠ 0) :
  d1 / d2 = 5 / 4 := 
by 
  sorry

end find_ratio_of_d1_and_d2_l1445_144550


namespace sequence_value_x_l1445_144540

theorem sequence_value_x (a1 a2 a3 a4 a5 a6 : ℕ) 
  (h1 : a1 = 2) 
  (h2 : a2 = 5) 
  (h3 : a3 = 11) 
  (h4 : a4 = 20) 
  (h5 : a6 = 47)
  (h6 : a2 - a1 = 3) 
  (h7 : a3 - a2 = 6) 
  (h8 : a4 - a3 = 9) 
  (h9 : a6 - a5 = 15) : 
  a5 = 32 :=
sorry

end sequence_value_x_l1445_144540


namespace evaluate_fraction_sqrt_l1445_144514

theorem evaluate_fraction_sqrt :
  (Real.sqrt ((1 / 8) + (1 / 18)) = (Real.sqrt 26) / 12) :=
by
  sorry

end evaluate_fraction_sqrt_l1445_144514


namespace fraction_sum_l1445_144583

theorem fraction_sum : (3 / 8) + (9 / 12) = 9 / 8 :=
by
  sorry

end fraction_sum_l1445_144583


namespace prime_p_is_2_l1445_144524

theorem prime_p_is_2 (p q r : ℕ) 
  (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (h_sum : p + q = r) (h_lt : p < q) : 
  p = 2 :=
sorry

end prime_p_is_2_l1445_144524


namespace solution_for_x_l1445_144573

theorem solution_for_x (x : ℝ) : 
  (∀ (y : ℝ), 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3 / 2 :=
by 
  -- Proof should go here
  sorry

end solution_for_x_l1445_144573


namespace value_of_a_l1445_144531

theorem value_of_a (a : ℝ) (A B : ℝ × ℝ) (hA : A = (a - 2, 2 * a + 7)) (hB : B = (1, 5)) (h_parallel : (A.1 = B.1)) : a = 3 :=
by {
  sorry
}

end value_of_a_l1445_144531


namespace time_to_pass_pole_l1445_144539

def length_of_train : ℝ := 240
def length_of_platform : ℝ := 650
def time_to_pass_platform : ℝ := 89

theorem time_to_pass_pole (length_of_train length_of_platform time_to_pass_platform : ℝ) 
  (h_train : length_of_train = 240)
  (h_platform : length_of_platform = 650)
  (h_time : time_to_pass_platform = 89)
  : (length_of_train / ((length_of_train + length_of_platform) / time_to_pass_platform)) = 24 := by
  -- Let the speed of the train be v, hence
  -- v = (length_of_train + length_of_platform) / time_to_pass_platform
  -- What we need to prove is  
  -- length_of_train / v = 24
  sorry

end time_to_pass_pole_l1445_144539


namespace dasha_rectangle_problem_l1445_144578

variables (a b c : ℕ)

theorem dasha_rectangle_problem
  (h1 : a > 0) 
  (h2 : a * (b + c) + a * (b - a) + a^2 + a * (c - a) = 43) 
  : (a = 1 ∧ b + c = 22) ∨ (a = 43 ∧ b + c = 2) :=
by
  sorry

end dasha_rectangle_problem_l1445_144578


namespace value_calculation_l1445_144546

-- Definition of constants used in the problem
def a : ℝ := 1.3333
def b : ℝ := 3.615
def expected_value : ℝ := 4.81998845

-- The proposition to be proven
theorem value_calculation : a * b = expected_value :=
by sorry

end value_calculation_l1445_144546


namespace quadratic_real_roots_implies_k_range_l1445_144580

theorem quadratic_real_roots_implies_k_range (k : ℝ) 
  (h : ∃ x : ℝ, k * x^2 + 2 * x - 1 = 0)
  (hk : k ≠ 0) : k ≥ -1 ∧ k ≠ 0 :=
sorry

end quadratic_real_roots_implies_k_range_l1445_144580


namespace largest_of_given_numbers_l1445_144562

theorem largest_of_given_numbers :
  ∀ (a b c d e : ℝ), a = 0.998 → b = 0.9899 → c = 0.99 → d = 0.981 → e = 0.995 →
  a > b ∧ a > c ∧ a > d ∧ a > e :=
by
  intros a b c d e Ha Hb Hc Hd He
  rw [Ha, Hb, Hc, Hd, He]
  exact ⟨ by norm_num, by norm_num, by norm_num, by norm_num ⟩

end largest_of_given_numbers_l1445_144562


namespace arithmetic_expression_evaluation_l1445_144536

theorem arithmetic_expression_evaluation : 1997 * (2000 / 2000) - 2000 * (1997 / 1997) = -3 := 
by
  sorry

end arithmetic_expression_evaluation_l1445_144536


namespace find_interest_rate_of_first_investment_l1445_144518

noncomputable def total_interest : ℚ := 73
noncomputable def interest_rate_7_percent : ℚ := 0.07
noncomputable def invested_400 : ℚ := 400
noncomputable def interest_7_percent := invested_400 * interest_rate_7_percent
noncomputable def interest_first_investment := total_interest - interest_7_percent
noncomputable def invested_first : ℚ := invested_400 - 100
noncomputable def interest_first : ℚ := 45  -- calculated as total_interest - interest_7_percent

theorem find_interest_rate_of_first_investment (r : ℚ) :
  interest_first = invested_first * r * 1 → 
  r = 0.15 :=
by
  sorry

end find_interest_rate_of_first_investment_l1445_144518


namespace intersection_points_in_decagon_l1445_144597

-- Define the number of sides for a regular decagon
def n : ℕ := 10

-- The formula to calculate the number of ways to choose 4 vertices from n vertices
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The statement that needs to be proven
theorem intersection_points_in_decagon : choose 10 4 = 210 := by
  sorry

end intersection_points_in_decagon_l1445_144597


namespace simplified_fraction_of_num_l1445_144563

def num : ℚ := 368 / 100

theorem simplified_fraction_of_num : num = 92 / 25 := by
  sorry

end simplified_fraction_of_num_l1445_144563


namespace dot_product_example_l1445_144577

variables (a : ℝ × ℝ) (b : ℝ × ℝ)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_example 
  (ha : a = (-1, 1)) 
  (hb : b = (3, -2)) : dot_product a b = -5 := by
  sorry

end dot_product_example_l1445_144577


namespace set_intersection_l1445_144590

theorem set_intersection (A B : Set ℝ) 
  (hA : A = { x : ℝ | 0 < x ∧ x < 5 }) 
  (hB : B = { x : ℝ | -1 ≤ x ∧ x < 4 }) : 
  (A ∩ B) = { x : ℝ | 0 < x ∧ x < 4 } :=
by
  sorry

end set_intersection_l1445_144590


namespace length_of_second_parallel_side_l1445_144509

-- Define the given conditions
def parallel_side1 : ℝ := 20
def distance : ℝ := 14
def area : ℝ := 266

-- Define the theorem to prove the length of the second parallel side
theorem length_of_second_parallel_side (x : ℝ) 
  (h : area = (1 / 2) * (parallel_side1 + x) * distance) : 
  x = 18 :=
sorry

end length_of_second_parallel_side_l1445_144509


namespace math_proof_l1445_144511

noncomputable def math_problem (x : ℝ) : ℝ :=
  (3 / (2 * x) * (1 / 2) * (2 / 5) * 5020) - ((2 ^ 3) * (1 / (3 * x + 2)) * 250) + Real.sqrt (900 / x)

theorem math_proof :
  math_problem 4 = 60.393 :=
by
  sorry

end math_proof_l1445_144511


namespace evaluate_K_l1445_144576

theorem evaluate_K : ∃ K : ℕ, 32^2 * 4^4 = 2^K ∧ K = 18 := by
  use 18
  sorry

end evaluate_K_l1445_144576


namespace evaporation_amount_l1445_144512

noncomputable def water_evaporated_per_day (total_water: ℝ) (percentage_evaporated: ℝ) (days: ℕ) : ℝ :=
  (percentage_evaporated / 100) * total_water / days

theorem evaporation_amount :
  water_evaporated_per_day 10 7 50 = 0.014 :=
by
  sorry

end evaporation_amount_l1445_144512


namespace smallest_and_second_smallest_four_digit_numbers_divisible_by_35_l1445_144591

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999
def divisible_by_35 (n : ℕ) : Prop := n % 35 = 0

theorem smallest_and_second_smallest_four_digit_numbers_divisible_by_35 :
  ∃ a b : ℕ, 
    is_four_digit a ∧ 
    is_four_digit b ∧ 
    divisible_by_35 a ∧ 
    divisible_by_35 b ∧ 
    a < b ∧ 
    ∀ c : ℕ, is_four_digit c → divisible_by_35 c → a ≤ c → (c = a ∨ c = b) :=
by
  sorry

end smallest_and_second_smallest_four_digit_numbers_divisible_by_35_l1445_144591


namespace win_sector_area_l1445_144525

noncomputable def radius : ℝ := 8
noncomputable def probability : ℝ := 1 / 4
noncomputable def total_area : ℝ := Real.pi * radius^2

theorem win_sector_area :
  ∃ (W : ℝ), W = probability * total_area ∧ W = 16 * Real.pi :=
by
  -- Proof skipped
  sorry

end win_sector_area_l1445_144525


namespace narrow_black_stripes_count_l1445_144555

theorem narrow_black_stripes_count (w n : ℕ) (b : ℕ) 
  (h1 : b = w + 7) 
  (h2 : w + n = b + 1) 
  : n = 8 :=
by sorry

end narrow_black_stripes_count_l1445_144555


namespace remaining_numbers_l1445_144515

-- Define the problem statement in Lean 4
theorem remaining_numbers (S S5 S3 : ℝ) (A3 : ℝ) 
  (h1 : S / 8 = 20) 
  (h2 : S5 / 5 = 12) 
  (h3 : S3 = S - S5) 
  (h4 : A3 = 100 / 3) : 
  S3 / A3 = 3 :=
sorry

end remaining_numbers_l1445_144515


namespace quadratic_two_distinct_real_roots_l1445_144538

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x^2 - 6 * x + k = 0) ↔ k < 9 :=
by
  sorry

end quadratic_two_distinct_real_roots_l1445_144538


namespace prime_numbers_solution_l1445_144554

theorem prime_numbers_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h1 : Nat.Prime (p + q)) (h2 : Nat.Prime (p^2 + q^2 - q)) : p = 3 ∧ q = 2 :=
by
  sorry

end prime_numbers_solution_l1445_144554


namespace larger_number_is_34_l1445_144556

theorem larger_number_is_34 (x y : ℕ) (h1 : x + y = 56) (h2 : y = x + 12) : y = 34 :=
by
  sorry

end larger_number_is_34_l1445_144556


namespace abs_eq_implies_y_eq_half_l1445_144519

theorem abs_eq_implies_y_eq_half (y : ℝ) (h : |y - 3| = |y + 2|) : y = 1 / 2 :=
by 
  sorry

end abs_eq_implies_y_eq_half_l1445_144519


namespace gcd_five_pentagonal_and_n_plus_one_l1445_144587

-- Definition of the nth pentagonal number
def pentagonal_number (n : ℕ) : ℕ :=
  (n * (3 * n - 1)) / 2

-- Proof statement
theorem gcd_five_pentagonal_and_n_plus_one (n : ℕ) (h : 0 < n) : 
  Nat.gcd (5 * pentagonal_number n) (n + 1) = 1 :=
sorry

end gcd_five_pentagonal_and_n_plus_one_l1445_144587


namespace manolo_makes_45_masks_in_four_hours_l1445_144567

noncomputable def face_masks_in_four_hour_shift : ℕ :=
  let first_hour_rate := 4
  let subsequent_hour_rate := 6
  let first_hour_face_masks := 60 / first_hour_rate
  let subsequent_hours_face_masks_per_hour := 60 / subsequent_hour_rate
  let total_face_masks :=
    first_hour_face_masks + subsequent_hours_face_masks_per_hour * (4 - 1)
  total_face_masks

theorem manolo_makes_45_masks_in_four_hours :
  face_masks_in_four_hour_shift = 45 :=
 by sorry

end manolo_makes_45_masks_in_four_hours_l1445_144567


namespace range_of_a_l1445_144572

noncomputable def y (a x : ℝ) : ℝ := a * Real.exp x + 3 * x
noncomputable def y_prime (a x : ℝ) : ℝ := a * Real.exp x + 3

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, a * Real.exp x + 3 = 0 ∧ a * Real.exp x + 3 * x < 0) → a < -3 :=
by
  sorry

end range_of_a_l1445_144572


namespace sufficient_but_not_necessary_l1445_144558

def p (x : ℝ) : Prop := |x - 4| > 2
def q (x : ℝ) : Prop := x > 1

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 6 → x > 1) ∧ ¬(∀ x, x > 1 → 2 ≤ x ∧ x ≤ 6) :=
  sorry

end sufficient_but_not_necessary_l1445_144558


namespace no_such_coins_l1445_144543

theorem no_such_coins (p1 p2 : ℝ) (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1)
  (cond1 : (1 - p1) * (1 - p2) = p1 * p2)
  (cond2 : p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :
  false :=
  sorry

end no_such_coins_l1445_144543


namespace area_of_square_field_l1445_144557

def side_length : ℕ := 7
def expected_area : ℕ := 49

theorem area_of_square_field : (side_length * side_length) = expected_area := 
by
  -- The proof steps will be filled here
  sorry

end area_of_square_field_l1445_144557


namespace union_complement_eq_l1445_144506

/-- The universal set U and sets A and B as given in the problem. -/
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

/-- The lean statement of our proof problem. -/
theorem union_complement_eq : A ∪ (U \ B) = {0, 1, 2, 3} := by
  sorry

end union_complement_eq_l1445_144506


namespace cos_90_eq_zero_l1445_144500

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l1445_144500


namespace white_balls_count_l1445_144592

theorem white_balls_count (W B R : ℕ) (h1 : B = W + 14) (h2 : R = 3 * (B - W)) (h3 : W + B + R = 1000) : W = 472 :=
sorry

end white_balls_count_l1445_144592


namespace coin_value_permutations_l1445_144596

theorem coin_value_permutations : 
  let digits := [1, 2, 2, 4, 4, 5, 9]
  let odd_digits := [1, 5, 9]
  let permutations (l : List ℕ) := Nat.factorial (l.length) / (l.filter (· = 2)).length.factorial / (l.filter (· = 4)).length.factorial
  3 * permutations (digits.erase 1 ++ digits.erase 5 ++ digits.erase 9) = 540 := by
  let digits := [1, 2, 2, 4, 4, 5, 9]
  let odd_digits := [1, 5, 9]
  let permutations (l : List ℕ) := Nat.factorial (l.length) / (l.filter (· = 2)).length.factorial / (l.filter (· = 4)).length.factorial
  show 3 * permutations (digits.erase 1 ++ digits.erase 5 ++ digits.erase 9) = 540
  
  -- Steps for the proof can be filled in
  -- sorry in place to indicate incomplete proof steps
  sorry

end coin_value_permutations_l1445_144596


namespace nth_equation_l1445_144508

theorem nth_equation (n : ℕ) (hn : n > 0) : 9 * n + (n - 1) = 10 * n - 1 :=
sorry

end nth_equation_l1445_144508


namespace Jana_taller_than_Kelly_l1445_144566

-- Definitions and given conditions
def Jess_height := 72
def Jana_height := 74
def Kelly_height := Jess_height - 3

-- Proof statement
theorem Jana_taller_than_Kelly : Jana_height - Kelly_height = 5 := by
  sorry

end Jana_taller_than_Kelly_l1445_144566


namespace minimize_average_cost_l1445_144520

noncomputable def average_comprehensive_cost (x : ℝ) : ℝ :=
  560 + 48 * x + 2160 * 10^6 / (2000 * x)

theorem minimize_average_cost : 
  ∃ x_min : ℝ, x_min ≥ 10 ∧ 
  ∀ x ≥ 10, average_comprehensive_cost x ≥ average_comprehensive_cost x_min :=
sorry

end minimize_average_cost_l1445_144520


namespace simplify_expression_l1445_144559

theorem simplify_expression : ( (3 + 4 + 5 + 6) / 3 ) + ( (3 * 6 + 9) / 4 ) = 12.75 := by
  sorry

end simplify_expression_l1445_144559


namespace number_of_chickens_l1445_144547

variable (C P : ℕ) (legs_total : ℕ := 48) (legs_pig : ℕ := 4) (legs_chicken : ℕ := 2) (number_pigs : ℕ := 9)

theorem number_of_chickens (h1 : P = number_pigs)
                           (h2 : legs_pig * P + legs_chicken * C = legs_total) :
                           C = 6 :=
by
  sorry

end number_of_chickens_l1445_144547


namespace sphere_tangent_radius_l1445_144598

variables (a b : ℝ) (h : b > a)

noncomputable def radius (a b : ℝ) : ℝ := a * (b - a) / Real.sqrt (b^2 - a^2)

theorem sphere_tangent_radius (a b : ℝ) (h : b > a) : 
  radius a b = a * (b - a) / Real.sqrt (b^2 - a^2) :=
by sorry

end sphere_tangent_radius_l1445_144598


namespace larger_integer_value_l1445_144565

theorem larger_integer_value (a b : ℕ) (h1 : a * b = 189) (h2 : a / gcd a b = 7 ∧ b / gcd a b = 3 ∨ a / gcd a b = 3 ∧ b / gcd a b = 7) : max a b = 21 :=
by
  sorry

end larger_integer_value_l1445_144565


namespace darry_small_ladder_climbs_l1445_144503

-- Define the constants based on the conditions
def full_ladder_steps := 11
def full_ladder_climbs := 10
def small_ladder_steps := 6
def total_steps := 152

-- Darry's total steps climbed via full ladder
def full_ladder_total_steps := full_ladder_steps * full_ladder_climbs

-- Define x as the number of times Darry climbed the smaller ladder
variable (x : ℕ)

-- Prove that x = 7 given the conditions
theorem darry_small_ladder_climbs (h : full_ladder_total_steps + small_ladder_steps * x = total_steps) : x = 7 :=
by 
  sorry

end darry_small_ladder_climbs_l1445_144503


namespace q_true_or_false_l1445_144516

variable (p q : Prop)

theorem q_true_or_false (h1 : ¬ (p ∧ q)) (h2 : ¬ p) : q ∨ ¬ q :=
by
  sorry

end q_true_or_false_l1445_144516


namespace returns_to_start_point_after_fourth_passenger_distance_after_last_passenger_total_earnings_l1445_144544

noncomputable def driving_distances : List ℤ := [-5, 3, 6, -4, 7, -2]

def fare (distance : ℕ) : ℕ :=
  if distance ≤ 3 then 8 else 8 + 2 * (distance - 3)

theorem returns_to_start_point_after_fourth_passenger :
  List.sum (driving_distances.take 4) = 0 :=
by
  sorry

theorem distance_after_last_passenger :
  List.sum driving_distances = 5 :=
by
  sorry

theorem total_earnings :
  (fare 5 + fare 3 + fare 6 + fare 4 + fare 7 + fare 2) = 68 :=
by
  sorry

end returns_to_start_point_after_fourth_passenger_distance_after_last_passenger_total_earnings_l1445_144544


namespace mean_temperature_is_0_5_l1445_144553

def temperatures : List ℝ := [-3.5, -2.25, 0, 3.75, 4.5]

theorem mean_temperature_is_0_5 :
  (temperatures.sum / temperatures.length) = 0.5 :=
by
  sorry

end mean_temperature_is_0_5_l1445_144553


namespace defective_units_l1445_144569

-- Conditions given in the problem
variable (D : ℝ) (h1 : 0.05 * D = 0.35)

-- The percent of the units produced that are defective is 7%
theorem defective_units (h1 : 0.05 * D = 0.35) : D = 7 := sorry

end defective_units_l1445_144569


namespace white_balls_in_bag_l1445_144552

   theorem white_balls_in_bag (m : ℕ) (h : m ≤ 7) :
     (2 * (m * (m - 1) / 2) / (7 * 6 / 2)) + ((m * (7 - m)) / (7 * 6 / 2)) = 6 / 7 → m = 3 :=
   by
     intros h_eq
     sorry
   
end white_balls_in_bag_l1445_144552


namespace confidence_95_implies_K2_gt_3_841_l1445_144560

-- Conditions
def confidence_no_relationship (K2 : ℝ) : Prop := K2 ≤ 3.841
def confidence_related_95 (K2 : ℝ) : Prop := K2 > 3.841
def confidence_related_99 (K2 : ℝ) : Prop := K2 > 6.635

theorem confidence_95_implies_K2_gt_3_841 (K2 : ℝ) :
  confidence_related_95 K2 ↔ K2 > 3.841 :=
by sorry

end confidence_95_implies_K2_gt_3_841_l1445_144560


namespace rectangle_area_l1445_144593

theorem rectangle_area (length : ℝ) (width_dm : ℝ) (width_m : ℝ) (h1 : length = 8) (h2 : width_dm = 50) (h3 : width_m = width_dm / 10) : 
  (length * width_m = 40) :=
by {
  sorry
}

end rectangle_area_l1445_144593


namespace difference_between_x_and_y_l1445_144571

theorem difference_between_x_and_y (x y : ℕ) (h₁ : 3 ^ x * 4 ^ y = 59049) (h₂ : x = 10) : x - y = 10 := by
  sorry

end difference_between_x_and_y_l1445_144571


namespace intersection_complement_l1445_144537

open Set

variable {α : Type*}
noncomputable def A : Set ℝ := {x | x^2 ≥ 1}
noncomputable def B : Set ℝ := {x | (x - 2) / x ≤ 0}

theorem intersection_complement :
  A ∩ (compl B) = (Iic (-1)) ∪ (Ioi 2) := by
sorry

end intersection_complement_l1445_144537


namespace arithmetic_seq_a12_l1445_144561

theorem arithmetic_seq_a12 (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 4 = 1)
  (h2 : a 7 + a 9 = 16)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  a 12 = 15 :=
by sorry

end arithmetic_seq_a12_l1445_144561


namespace logarithm_identity_l1445_144574

theorem logarithm_identity (k x : ℝ) (hk : 0 < k ∧ k ≠ 1) (hx : 0 < x) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 7) = 3 → x = 343 :=
by
  intro h
  sorry

end logarithm_identity_l1445_144574


namespace greatest_possible_q_minus_r_l1445_144581

theorem greatest_possible_q_minus_r :
  ∃ (q r : ℕ), 945 = 21 * q + r ∧ 0 ≤ r ∧ r < 21 ∧ q - r = 45 :=
by
  sorry

end greatest_possible_q_minus_r_l1445_144581


namespace apples_per_bucket_l1445_144588

theorem apples_per_bucket (total_apples buckets : ℕ) (h1 : total_apples = 56) (h2 : buckets = 7) : 
  (total_apples / buckets) = 8 :=
by
  sorry

end apples_per_bucket_l1445_144588


namespace triangle_side_length_l1445_144502

theorem triangle_side_length (a b c : ℝ) (A : ℝ) 
  (h_a : a = 2) (h_c : c = 2) (h_A : A = 30) :
  b = 2 * Real.sqrt 3 :=
by
  sorry

end triangle_side_length_l1445_144502


namespace circuit_length_is_365_l1445_144501

-- Definitions based on given conditions
def runs_morning := 7
def runs_afternoon := 3
def total_distance_week := 25550
def total_runs_day := runs_morning + runs_afternoon
def total_runs_week := total_runs_day * 7

-- Statement of the problem to be proved
theorem circuit_length_is_365 :
  total_distance_week / total_runs_week = 365 :=
sorry

end circuit_length_is_365_l1445_144501


namespace fran_ate_15_green_macaroons_l1445_144532

variable (total_red total_green initial_remaining green_macaroons_eaten : ℕ)

-- Conditions as definitions
def initial_red_macaroons := 50
def initial_green_macaroons := 40
def total_macaroons := 90
def remaining_macaroons := 45

-- Total eaten macaroons
def total_eaten_macaroons (G : ℕ) := G + 2 * G

-- The proof statement
theorem fran_ate_15_green_macaroons
  (h1 : total_red = initial_red_macaroons)
  (h2 : total_green = initial_green_macaroons)
  (h3 : initial_remaining = remaining_macaroons)
  (h4 : total_macaroons = initial_red_macaroons + initial_green_macaroons)
  (h5 : initial_remaining = total_macaroons - total_eaten_macaroons green_macaroons_eaten):
  green_macaroons_eaten = 15 :=
  by
  sorry

end fran_ate_15_green_macaroons_l1445_144532


namespace fraction_identity_l1445_144527

theorem fraction_identity (a b : ℚ) (h : a / b = 3 / 4) : (b - a) / b = 1 / 4 :=
by
  sorry

end fraction_identity_l1445_144527


namespace cos_squared_alpha_plus_pi_over_4_l1445_144582

theorem cos_squared_alpha_plus_pi_over_4 (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) :
  Real.cos (α + Real.pi / 4) ^ 2 = 1 / 6 :=
by
  sorry

end cos_squared_alpha_plus_pi_over_4_l1445_144582


namespace range_of_y_eq_4_sin_squared_x_minus_2_l1445_144568

theorem range_of_y_eq_4_sin_squared_x_minus_2 : 
  (∀ x : ℝ, y = 4 * (Real.sin x)^2 - 2) → 
  (∃ a b : ℝ, ∀ x : ℝ, y ∈ Set.Icc a b ∧ a = -2 ∧ b = 2) :=
sorry

end range_of_y_eq_4_sin_squared_x_minus_2_l1445_144568


namespace option_C_l1445_144541

theorem option_C (a b c : ℝ) (h₀ : a > b) (h₁ : b > c) (h₂ : c > 0) :
  (b + c) / (a + c) > b / a :=
sorry

end option_C_l1445_144541


namespace corrected_sum_l1445_144533

theorem corrected_sum : 37541 + 43839 ≠ 80280 → 37541 + 43839 = 81380 :=
by
  sorry

end corrected_sum_l1445_144533


namespace problem1_problem2_l1445_144528

-- Define points A, B, C
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 1, y := -2}
def B : Point := {x := 2, y := 1}
def C : Point := {x := 3, y := 2}

-- Function to compute vector difference
def vector_sub (p1 p2 : Point) : Point :=
  {x := p1.x - p2.x, y := p1.y - p2.y}

-- Function to compute vector scalar multiplication
def scalar_mul (k : ℝ) (p : Point) : Point :=
  {x := k * p.x, y := k * p.y}

-- Function to add two vectors
def vec_add (p1 p2 : Point) : Point :=
  {x := p1.x + p2.x, y := p1.y + p2.y}

-- Problem 1
def result_vector : Point :=
  let AB := vector_sub B A
  let AC := vector_sub C A
  let BC := vector_sub C B
  vec_add (scalar_mul 3 AB) (vec_add (scalar_mul (-2) AC) BC)

-- Prove the coordinates are (0, 2)
theorem problem1 : result_vector = {x := 0, y := 2} := by
  sorry

-- Problem 2
def D : Point :=
  let BC := vector_sub C B
  {x := 1 + BC.x, y := (-2) + BC.y}

-- Prove the coordinates are (2, -1)
theorem problem2 : D = {x := 2, y := -1} := by
  sorry

end problem1_problem2_l1445_144528


namespace tenth_number_drawn_eq_195_l1445_144530

noncomputable def total_students : Nat := 1000
noncomputable def sample_size : Nat := 50
noncomputable def first_selected_number : Nat := 15  -- Note: 0015 is 15 in natural number

theorem tenth_number_drawn_eq_195 
  (h1 : total_students = 1000)
  (h2 : sample_size = 50)
  (h3 : first_selected_number = 15) :
  15 + (20 * 9) = 195 := 
by
  sorry

end tenth_number_drawn_eq_195_l1445_144530


namespace sqrt8_same_type_as_sqrt2_l1445_144522

theorem sqrt8_same_type_as_sqrt2 :
  (∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 8) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 4) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 6) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 10) :=
by
  sorry

end sqrt8_same_type_as_sqrt2_l1445_144522


namespace sin_30_is_half_l1445_144542

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l1445_144542


namespace values_of_x_minus_y_l1445_144523

theorem values_of_x_minus_y (x y : ℤ) (h1 : |x| = 5) (h2 : |y| = 3) (h3 : y > x) : x - y = -2 ∨ x - y = -8 :=
  sorry

end values_of_x_minus_y_l1445_144523


namespace same_terminal_side_l1445_144595

theorem same_terminal_side : 
  let θ1 := 23 * Real.pi / 3
  let θ2 := 5 * Real.pi / 3
  (∃ k : ℤ, θ1 - 2 * k * Real.pi = θ2) :=
sorry

end same_terminal_side_l1445_144595


namespace particle_max_height_and_time_l1445_144513

theorem particle_max_height_and_time (t : ℝ) (s : ℝ) 
  (height_eq : s = 180 * t - 18 * t^2) :
  ∃ t₁ : ℝ, ∃ s₁ : ℝ, s₁ = 450 ∧ t₁ = 5 ∧ s = 180 * t₁ - 18 * t₁^2 :=
sorry

end particle_max_height_and_time_l1445_144513
