import Mathlib

namespace omitted_angle_of_convex_polygon_l1318_131804

theorem omitted_angle_of_convex_polygon (calculated_sum : ℕ) (omitted_angle : ℕ)
    (h₁ : calculated_sum = 2583) (h₂ : omitted_angle = 2700 - 2583) :
    omitted_angle = 117 :=
by
  sorry

end omitted_angle_of_convex_polygon_l1318_131804


namespace tan_alpha_trigonometric_expression_l1318_131884

variable (α : ℝ)
variable (h1 : Real.sin (Real.pi + α) = 3 / 5)
variable (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2)

theorem tan_alpha (h1 : Real.sin (Real.pi + α) = 3 / 5) (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : Real.tan α = 3 / 4 := 
sorry

theorem trigonometric_expression (h1 : Real.sin (Real.pi + α) = 3 / 5) (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  (Real.sin ((Real.pi + α) / 2) - Real.cos ((Real.pi + α) / 2)) / 
  (Real.sin ((Real.pi - α) / 2) - Real.cos ((Real.pi - α) / 2)) = -1 / 2 := 
sorry

end tan_alpha_trigonometric_expression_l1318_131884


namespace unique_two_digit_integer_l1318_131863

theorem unique_two_digit_integer (t : ℕ) (h : 11 * t % 100 = 36) (ht : 10 ≤ t ∧ t ≤ 99) : t = 76 :=
by
  sorry

end unique_two_digit_integer_l1318_131863


namespace factorize_m_square_minus_16_l1318_131890

-- Define the expression
def expr (m : ℝ) : ℝ := m^2 - 16

-- Define the factorized form
def factorized_expr (m : ℝ) : ℝ := (m + 4) * (m - 4)

-- State the theorem
theorem factorize_m_square_minus_16 (m : ℝ) : expr m = factorized_expr m :=
by
  sorry

end factorize_m_square_minus_16_l1318_131890


namespace trigonometric_identity_simplification_l1318_131891

theorem trigonometric_identity_simplification :
  (Real.sin (15 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) + Real.cos (15 * Real.pi / 180) * Real.sin (105 * Real.pi / 180) = 1) :=
by sorry

end trigonometric_identity_simplification_l1318_131891


namespace arith_seq_a4a6_equals_4_l1318_131896

variable (a : ℕ → ℝ) (d : ℝ)
variable (h2 : a 2 = a 1 + d)
variable (h4 : a 4 = a 1 + 3 * d)
variable (h6 : a 6 = a 1 + 5 * d)
variable (h8 : a 8 = a 1 + 7 * d)
variable (h10 : a 10 = a 1 + 9 * d)
variable (condition : (a 2)^2 + 2 * a 2 * a 8 + a 6 * a 10 = 16)

theorem arith_seq_a4a6_equals_4 : a 4 * a 6 = 4 := by
  sorry

end arith_seq_a4a6_equals_4_l1318_131896


namespace largest_of_seven_consecutive_l1318_131849

theorem largest_of_seven_consecutive (n : ℕ) 
  (h1: n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 3010) :
  n + 6 = 433 :=
by 
  sorry

end largest_of_seven_consecutive_l1318_131849


namespace largest_angle_of_consecutive_integer_angles_of_hexagon_l1318_131862

theorem largest_angle_of_consecutive_integer_angles_of_hexagon 
  (angles : Fin 6 → ℝ)
  (h_consecutive : ∃ (x : ℝ), angles = ![
    x - 3, x - 2, x - 1, x, x + 1, x + 2 ])
  (h_sum : (angles 0 + angles 1 + angles 2 + angles 3 + angles 4 + angles 5) = 720) :
  (angles 5 = 122.5) :=
by
  sorry

end largest_angle_of_consecutive_integer_angles_of_hexagon_l1318_131862


namespace average_price_of_rackets_l1318_131829

theorem average_price_of_rackets (total_amount : ℝ) (number_of_pairs : ℕ) (average_price : ℝ) 
  (h1 : total_amount = 588) (h2 : number_of_pairs = 60) : average_price = 9.80 :=
by
  sorry

end average_price_of_rackets_l1318_131829


namespace proposition_1_proposition_2_proposition_3_proposition_4_l1318_131893

axiom p1 : Prop
axiom p2 : Prop
axiom p3 : Prop
axiom p4 : Prop

axiom p1_true : p1 = true
axiom p2_false : p2 = false
axiom p3_false : p3 = false
axiom p4_true : p4 = true

theorem proposition_1 : (p1 ∧ p4) = true := by sorry
theorem proposition_2 : (p1 ∧ p2) = false := by sorry
theorem proposition_3 : (¬p2 ∨ p3) = true := by sorry
theorem proposition_4 : (¬p3 ∨ ¬p4) = true := by sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l1318_131893


namespace sufficient_and_necessary_condition_l1318_131869

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem sufficient_and_necessary_condition (a b : ℝ) : (a + b > 0) ↔ (f a + f b > 0) :=
by sorry

end sufficient_and_necessary_condition_l1318_131869


namespace smallest_whole_number_larger_than_triangle_perimeter_l1318_131875

theorem smallest_whole_number_larger_than_triangle_perimeter :
  (∀ s : ℝ, 16 < s ∧ s < 30 → ∃ n : ℕ, n = 60) :=
by
  sorry

end smallest_whole_number_larger_than_triangle_perimeter_l1318_131875


namespace cauchy_schwarz_inequality_l1318_131848

theorem cauchy_schwarz_inequality (a b x y : ℝ) :
  (a^2 + b^2) * (x^2 + y^2) ≥ (a * x + b * y)^2 :=
by
  sorry

end cauchy_schwarz_inequality_l1318_131848


namespace increasing_interval_a_geq_neg2_l1318_131871

theorem increasing_interval_a_geq_neg2
  (f : ℝ → ℝ)
  (h : ∀ x, f x = x^2 + 2 * (a - 2) * x + 5)
  (h_inc : ∀ x > 4, f (x + 1) > f x) :
  a ≥ -2 :=
sorry

end increasing_interval_a_geq_neg2_l1318_131871


namespace mod_inverse_13_1728_l1318_131856

theorem mod_inverse_13_1728 :
  (13 * 133) % 1728 = 1 := by
  sorry

end mod_inverse_13_1728_l1318_131856


namespace geometric_series_product_l1318_131880

theorem geometric_series_product (y : ℝ) :
  (∑'n : ℕ, (1 / 3 : ℝ) ^ n) * (∑'n : ℕ, (- 1 / 3 : ℝ) ^ n)
  = ∑'n : ℕ, (y⁻¹ : ℝ) ^ n ↔ y = 9 :=
by
  sorry

end geometric_series_product_l1318_131880


namespace factor_w4_minus_16_l1318_131843

theorem factor_w4_minus_16 (w : ℝ) : (w^4 - 16) = (w - 2) * (w + 2) * (w^2 + 4) :=
by
    sorry

end factor_w4_minus_16_l1318_131843


namespace total_cost_price_l1318_131858

theorem total_cost_price (SP1 SP2 SP3 : ℝ) (P1 P2 P3 : ℝ) 
  (h1 : SP1 = 120) (h2 : SP2 = 150) (h3 : SP3 = 200)
  (h4 : P1 = 0.20) (h5 : P2 = 0.25) (h6 : P3 = 0.10) : (SP1 / (1 + P1) + SP2 / (1 + P2) + SP3 / (1 + P3) = 401.82) :=
by
  sorry

end total_cost_price_l1318_131858


namespace sum_of_three_squares_l1318_131847

theorem sum_of_three_squares (s t : ℤ) (h1 : 3 * s + 2 * t = 27)
                             (h2 : 2 * s + 3 * t = 23) (h3 : s + 2 * t = 13) :
  3 * s = 21 :=
sorry

end sum_of_three_squares_l1318_131847


namespace proportion_fourth_number_l1318_131882

theorem proportion_fourth_number (x y : ℝ) (h₀ : 0.75 * y = 5 * x) (h₁ : x = 1.65) : y = 11 :=
by
  sorry

end proportion_fourth_number_l1318_131882


namespace winning_candidate_percentage_l1318_131807

theorem winning_candidate_percentage
  (majority_difference : ℕ)
  (total_valid_votes : ℕ)
  (P : ℕ)
  (h1 : majority_difference = 192)
  (h2 : total_valid_votes = 480)
  (h3 : 960 * P = 67200) : 
  P = 70 := by
  sorry

end winning_candidate_percentage_l1318_131807


namespace original_number_of_men_l1318_131841

theorem original_number_of_men 
    (x : ℕ) 
    (h : x * 40 = (x - 5) * 60) : x = 15 := 
sorry

end original_number_of_men_l1318_131841


namespace ratio_of_adults_to_children_is_24_over_25_l1318_131814

theorem ratio_of_adults_to_children_is_24_over_25
  (a c : ℕ) (h₁ : a ≥ 1) (h₂ : c ≥ 1) 
  (h₃ : 30 * a + 18 * c = 2340) 
  (h₄ : c % 5 = 0) :
  a = 48 ∧ c = 50 ∧ (a / c : ℚ) = 24 / 25 :=
sorry

end ratio_of_adults_to_children_is_24_over_25_l1318_131814


namespace total_books_l1318_131851

theorem total_books (shelves_mystery shelves_picture : ℕ) (books_per_shelf : ℕ) 
    (h_mystery : shelves_mystery = 5) (h_picture : shelves_picture = 4) (h_books_per_shelf : books_per_shelf = 6) : 
    shelves_mystery * books_per_shelf + shelves_picture * books_per_shelf = 54 := 
by 
  sorry

end total_books_l1318_131851


namespace no_consecutive_nat_mul_eq_25k_plus_1_l1318_131822

theorem no_consecutive_nat_mul_eq_25k_plus_1 (k : ℕ) : 
  ¬ ∃ n : ℕ, n * (n + 1) = 25 * k + 1 :=
sorry

end no_consecutive_nat_mul_eq_25k_plus_1_l1318_131822


namespace nineteen_power_six_l1318_131879

theorem nineteen_power_six :
    19^11 / 19^5 = 47045881 := by
  sorry

end nineteen_power_six_l1318_131879


namespace possible_values_of_a_plus_b_l1318_131868

theorem possible_values_of_a_plus_b (a b : ℤ)
  (h1 : ∃ α : ℝ, 0 ≤ α ∧ α < 2 * Real.pi ∧ (∃ (sinα cosα : ℝ), sinα = Real.sin α ∧ cosα = Real.cos α ∧ (sinα + cosα = -a) ∧ (sinα * cosα = 2 * b^2))) :
  a + b = 1 ∨ a + b = -1 := 
sorry

end possible_values_of_a_plus_b_l1318_131868


namespace length_first_train_l1318_131885

noncomputable def length_second_train : ℝ := 200
noncomputable def speed_first_train_kmh : ℝ := 42
noncomputable def speed_second_train_kmh : ℝ := 30
noncomputable def time_seconds : ℝ := 14.998800095992321

noncomputable def speed_first_train_ms : ℝ := speed_first_train_kmh * 1000 / 3600
noncomputable def speed_second_train_ms : ℝ := speed_second_train_kmh * 1000 / 3600

noncomputable def relative_speed : ℝ := speed_first_train_ms + speed_second_train_ms
noncomputable def combined_length : ℝ := relative_speed * time_seconds

theorem length_first_train : combined_length - length_second_train = 99.9760019198464 :=
by
  sorry

end length_first_train_l1318_131885


namespace distance_BC_400m_l1318_131852

-- Define the hypotheses
variables
  (starting_from_same_time : Prop) -- Sam and Nik start from points A and B respectively at the same time
  (constant_speeds : Prop) -- They travel towards each other at constant speeds along the same route
  (meeting_point_C : Prop) -- They meet at point C, which is 600 m away from starting point A
  (speed_Sam : ℕ) (speed_Sam_value : speed_Sam = 50) -- The speed of Sam is 50 meters per minute
  (time_Sam : ℕ) (time_Sam_value : time_Sam = 20) -- It took Sam 20 minutes to cover the distance between A and B

-- Define the statement to be proven
theorem distance_BC_400m
  (d_AB : ℕ) (d_AB_value : d_AB = speed_Sam * time_Sam)
  (d_AC : ℕ) (d_AC_value : d_AC = 600)
  (d_BC : ℕ) (d_BC_value : d_BC = d_AB - d_AC) :
  d_BC = 400 := by
  sorry

end distance_BC_400m_l1318_131852


namespace average_of_rest_equals_40_l1318_131818

-- Defining the initial conditions
def total_students : ℕ := 20
def high_scorers : ℕ := 2
def low_scorers : ℕ := 3
def class_average : ℚ := 40

-- The target function to calculate the average of the rest of the students
def average_rest_students (total_students high_scorers low_scorers : ℕ) (class_average : ℚ) : ℚ :=
  let total_marks := total_students * class_average
  let high_scorer_marks := 100 * high_scorers
  let low_scorer_marks := 0 * low_scorers
  let rest_marks := total_marks - (high_scorer_marks + low_scorer_marks)
  let rest_students := total_students - high_scorers - low_scorers
  rest_marks / rest_students

-- The theorem to prove that the average of the rest of the students is 40
theorem average_of_rest_equals_40 : average_rest_students total_students high_scorers low_scorers class_average = 40 := 
by
  sorry

end average_of_rest_equals_40_l1318_131818


namespace tennis_players_l1318_131823

theorem tennis_players (total_members badminton_players neither_players both_players : ℕ)
  (h1 : total_members = 80)
  (h2 : badminton_players = 48)
  (h3 : neither_players = 7)
  (h4 : both_players = 21) :
  total_members - neither_players = badminton_players - both_players + (total_members - neither_players - badminton_players + both_players) + both_players →
  ((total_members - neither_players) - (badminton_players - both_players) - both_players) + both_players = 46 :=
by
  intros h
  sorry

end tennis_players_l1318_131823


namespace total_pages_l1318_131865

-- Definitions based on conditions
def math_pages : ℕ := 10
def extra_reading_pages : ℕ := 3
def reading_pages : ℕ := math_pages + extra_reading_pages

-- Statement of the proof problem
theorem total_pages : math_pages + reading_pages = 23 := by 
  sorry

end total_pages_l1318_131865


namespace div_pow_two_sub_one_l1318_131805

theorem div_pow_two_sub_one {k n : ℕ} (hk : 0 < k) (hn : 0 < n) :
  (3^k ∣ 2^n - 1) ↔ (∃ m : ℕ, n = 2 * 3^(k-1) * m) :=
by
  sorry

end div_pow_two_sub_one_l1318_131805


namespace simplify_expression_l1318_131845

theorem simplify_expression (x : ℝ) : 
  (2 * x - 3 * (2 + x) + 4 * (2 - x) - 5 * (2 + 3 * x)) = -20 * x - 8 :=
by
  sorry

end simplify_expression_l1318_131845


namespace ratio_simplified_l1318_131854

variable (a b c : ℕ)
variable (n m p : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : p > 0)

theorem ratio_simplified (h_ratio : a^n = 3 * c^p ∧ b^m = 4 * c^p ∧ c^p = 7 * c^p) :
  (a^n + b^m + c^p) / c^p = 2 := sorry

end ratio_simplified_l1318_131854


namespace number_of_possible_A2_eq_one_l1318_131881

noncomputable def unique_possible_A2 (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  (A^4 = 0) → (A^2 = 0)

theorem number_of_possible_A2_eq_one (A : Matrix (Fin 2) (Fin 2) ℝ) :
  unique_possible_A2 A :=
by 
  sorry

end number_of_possible_A2_eq_one_l1318_131881


namespace locus_of_Q_is_circle_l1318_131857

variables {A B C P Q : ℝ}

def point_on_segment (A B C : ℝ) : Prop := C > A ∧ C < B

def variable_point_on_circle (A B P : ℝ) : Prop := (P - A) * (P - B) = 0

def ratio_condition (C P Q A B : ℝ) : Prop := (P - C) / (C - Q) = (A - C) / (C - B)

def locus_of_Q_circle (A B C P Q : ℝ) : Prop := ∃ B', (C > A ∧ C < B) → (P - A) * (P - B) = 0 → (P - C) / (C - Q) = (A - C) / (C - B) → (Q - B') * (Q - B) = 0

theorem locus_of_Q_is_circle (A B C P Q : ℝ) :
  point_on_segment A B C →
  variable_point_on_circle A B P →
  ratio_condition C P Q A B →
  locus_of_Q_circle A B C P Q :=
by
  sorry

end locus_of_Q_is_circle_l1318_131857


namespace other_liquid_cost_l1318_131886

-- Definitions based on conditions
def total_fuel_gallons : ℕ := 12
def fuel_price_per_gallon : ℝ := 8
def oil_price_per_gallon : ℝ := 15
def fuel_cost : ℝ := total_fuel_gallons * fuel_price_per_gallon
def other_liquid_price_per_gallon (x : ℝ) : Prop :=
  (7 * x + 5 * oil_price_per_gallon = fuel_cost) ∨
  (7 * oil_price_per_gallon + 5 * x = fuel_cost)

-- Question: The cost of the other liquid per gallon
theorem other_liquid_cost :
  ∃ x, other_liquid_price_per_gallon x ∧ x = 3 :=
sorry

end other_liquid_cost_l1318_131886


namespace total_inflation_time_l1318_131888

theorem total_inflation_time (time_per_ball : ℕ) (alexia_balls : ℕ) (extra_balls : ℕ) : 
  time_per_ball = 20 → alexia_balls = 20 → extra_balls = 5 →
  (alexia_balls * time_per_ball) + ((alexia_balls + extra_balls) * time_per_ball) = 900 :=
by 
  intros h1 h2 h3
  sorry

end total_inflation_time_l1318_131888


namespace reduced_price_proof_l1318_131887

noncomputable def reduced_price (P: ℝ) := 0.88 * P

theorem reduced_price_proof :
  ∃ R P : ℝ, R = reduced_price P ∧ 1200 / R = 1200 / P + 6 ∧ R = 24 :=
by
  sorry

end reduced_price_proof_l1318_131887


namespace number_of_ways_to_express_n_as_sum_l1318_131833

noncomputable def P (n k : ℕ) : ℕ := sorry
noncomputable def Q (n k : ℕ) : ℕ := sorry

theorem number_of_ways_to_express_n_as_sum (n : ℕ) (k : ℕ) (h : k ≥ 2) : P n k = Q n k := sorry

end number_of_ways_to_express_n_as_sum_l1318_131833


namespace roots_in_interval_l1318_131809

theorem roots_in_interval (f : ℝ → ℝ)
  (h : ∀ x, f x = 4 * x ^ 2 - (3 * m + 1) * x - m - 2) :
  (forall (x1 x2 : ℝ), (f x1 = 0 ∧ f x2 = 0) → -1 < x1 ∧ x1 < 2 ∧ -1 < x2 ∧ x2 < 2) ↔ -1 < m ∧ m < 12 / 7 :=
sorry

end roots_in_interval_l1318_131809


namespace avg_primes_between_30_and_50_l1318_131839

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_30_and_50 : List ℕ := [31, 37, 41, 43, 47]

def sum_primes : ℕ := primes_between_30_and_50.sum

def count_primes : ℕ := primes_between_30_and_50.length

def average_primes : ℚ := (sum_primes : ℚ) / (count_primes : ℚ)

theorem avg_primes_between_30_and_50 : average_primes = 39.8 := by
  sorry

end avg_primes_between_30_and_50_l1318_131839


namespace quadratic_no_ten_powers_of_2_values_l1318_131831

theorem quadratic_no_ten_powers_of_2_values 
  (a b : ℝ) :
  ¬ ∃ (j : ℤ), ∀ k : ℤ, j ≤ k ∧ k < j + 10 → ∃ n : ℕ, (k^2 + a * k + b) = 2 ^ n :=
by sorry

end quadratic_no_ten_powers_of_2_values_l1318_131831


namespace GIMPS_meaning_l1318_131878

/--
  Curtis Cooper's team discovered the largest prime number known as \( 2^{74,207,281} - 1 \), which is a Mersenne prime.
  GIMPS stands for "Great Internet Mersenne Prime Search."

  Prove that GIMPS means "Great Internet Mersenne Prime Search".
-/
theorem GIMPS_meaning : GIMPS = "Great Internet Mersenne Prime Search" :=
  sorry

end GIMPS_meaning_l1318_131878


namespace train_pass_platform_time_l1318_131826

-- Define the conditions given in the problem.
def train_length : ℕ := 1200
def platform_length : ℕ := 1100
def time_to_cross_tree : ℕ := 120

-- Define the calculation for speed.
def speed := train_length / time_to_cross_tree

-- Define the combined length of train and platform.
def combined_length := train_length + platform_length

-- Define the expected time to pass the platform.
def expected_time_to_pass_platform := combined_length / speed

-- The theorem to prove.
theorem train_pass_platform_time :
  expected_time_to_pass_platform = 230 :=
by {
  -- Placeholder for the proof.
  sorry
}

end train_pass_platform_time_l1318_131826


namespace necessary_and_sufficient_condition_l1318_131873

noncomputable def f (a x : ℝ) : ℝ := a * x - x^2

theorem necessary_and_sufficient_condition (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≤ 1) ↔ (0 < a ∧ a ≤ 2) := by
  sorry

end necessary_and_sufficient_condition_l1318_131873


namespace appropriate_presentation_length_l1318_131821

-- Definitions and conditions
def ideal_speaking_rate : ℕ := 160
def min_minutes : ℕ := 20
def max_minutes : ℕ := 40
def appropriate_words_range (words : ℕ) : Prop :=
  words ≥ (min_minutes * ideal_speaking_rate) ∧ words ≤ (max_minutes * ideal_speaking_rate)

-- Statement to prove
theorem appropriate_presentation_length : appropriate_words_range 5000 :=
by sorry

end appropriate_presentation_length_l1318_131821


namespace exponent_product_l1318_131810

theorem exponent_product (a : ℝ) (m n : ℕ)
  (h1 : a^m = 2) (h2 : a^n = 5) : a^(2*m + n) = 20 :=
sorry

end exponent_product_l1318_131810


namespace total_days_spent_on_islands_l1318_131817

-- Define the conditions and question in Lean 4
def first_expedition_A_weeks := 3
def second_expedition_A_weeks := first_expedition_A_weeks + 2
def last_expedition_A_weeks := second_expedition_A_weeks * 2

def first_expedition_B_weeks := 5
def second_expedition_B_weeks := first_expedition_B_weeks - 3
def last_expedition_B_weeks := first_expedition_B_weeks

def total_weeks_on_island_A := first_expedition_A_weeks + second_expedition_A_weeks + last_expedition_A_weeks
def total_weeks_on_island_B := first_expedition_B_weeks + second_expedition_B_weeks + last_expedition_B_weeks

def total_weeks := total_weeks_on_island_A + total_weeks_on_island_B
def total_days := total_weeks * 7

theorem total_days_spent_on_islands : total_days = 210 :=
by
  -- We skip the proof part
  sorry

end total_days_spent_on_islands_l1318_131817


namespace value_of_abc_l1318_131813

theorem value_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + 1 / b = 5) (h2 : b + 1 / c = 2) (h3 : c + 1 / a = 3) : 
  abc = 1 :=
by
  sorry

end value_of_abc_l1318_131813


namespace factor_expression_l1318_131824

theorem factor_expression (x : ℝ) : 3 * x * (x - 5) + 7 * (x - 5) - 2 * (x - 5) = (3 * x + 5) * (x - 5) :=
by
  sorry

end factor_expression_l1318_131824


namespace fruit_platter_has_thirty_fruits_l1318_131830

-- Define the conditions
def at_least_five_apples (g_apple r_apple y_apple : ℕ) : Prop :=
  g_apple + r_apple + y_apple ≥ 5

def at_most_five_oranges (r_orange y_orange : ℕ) : Prop :=
  r_orange + y_orange ≤ 5

def kiwi_grape_constraints (g_kiwi p_grape : ℕ) : Prop :=
  g_kiwi + p_grape ≥ 8 ∧ g_kiwi + p_grape ≤ 12 ∧ g_kiwi = p_grape

def at_least_one_each_grape (g_grape p_grape : ℕ) : Prop :=
  g_grape ≥ 1 ∧ p_grape ≥ 1

-- The final statement to prove
theorem fruit_platter_has_thirty_fruits :
  ∃ (g_apple r_apple y_apple r_orange y_orange g_kiwi p_grape g_grape : ℕ),
    at_least_five_apples g_apple r_apple y_apple ∧
    at_most_five_oranges r_orange y_orange ∧
    kiwi_grape_constraints g_kiwi p_grape ∧
    at_least_one_each_grape g_grape p_grape ∧
    g_apple + r_apple + y_apple + r_orange + y_orange + g_kiwi + p_grape + g_grape = 30 :=
sorry

end fruit_platter_has_thirty_fruits_l1318_131830


namespace petya_numbers_board_l1318_131834

theorem petya_numbers_board (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k → k < n → (∀ d : ℕ, 4 ∣ 10 ^ d → ¬(4 ∣ k))) 
  (h3 : ∀ k : ℕ, 0 ≤ k → k < n→ (∀ d : ℕ, 7 ∣ 10 ^ d → ¬(7 ∣ (k + n - 1)))) : 
  ∃ x : ℕ, (x = 2021) := 
by
  sorry

end petya_numbers_board_l1318_131834


namespace total_trucks_l1318_131850

-- Define the number of trucks Namjoon has
def trucks_namjoon : ℕ := 3

-- Define the number of trucks Taehyung has
def trucks_taehyung : ℕ := 2

-- Prove that together, Namjoon and Taehyung have 5 trucks
theorem total_trucks : trucks_namjoon + trucks_taehyung = 5 := by 
  sorry

end total_trucks_l1318_131850


namespace geometric_series_sum_l1318_131801

theorem geometric_series_sum :
  (1 / 3 - 1 / 6 + 1 / 12 - 1 / 24 + 1 / 48 - 1 / 96) = 7 / 32 :=
by
  sorry

end geometric_series_sum_l1318_131801


namespace comparison_of_exponential_and_power_l1318_131883

theorem comparison_of_exponential_and_power :
  let a := 2 ^ 0.6
  let b := 0.6 ^ 2
  a > b :=
by
  let a := 2 ^ 0.6
  let b := 0.6 ^ 2
  sorry

end comparison_of_exponential_and_power_l1318_131883


namespace total_area_of_squares_l1318_131838

-- Condition 1: Definition of the side length
def side_length (s : ℝ) : Prop := s = 12

-- Condition 2: Definition of the center of one square coinciding with the vertex of another
-- Here, we assume the positions are fixed so this condition is given
def coincide_center_vertex (s₁ s₂ : ℝ) : Prop := s₁ = s₂ 

-- The main theorem statement
theorem total_area_of_squares
  (s₁ s₂ : ℝ) 
  (h₁ : side_length s₁)
  (h₂ : side_length s₂)
  (h₃ : coincide_center_vertex s₁ s₂) :
  (2 * s₁^2) - (s₁^2 / 4) = 252 :=
by
  sorry

end total_area_of_squares_l1318_131838


namespace length_of_unfenced_side_l1318_131828

theorem length_of_unfenced_side
  (L W : ℝ)
  (h1 : L * W = 200)
  (h2 : 2 * W + L = 50) :
  L = 10 :=
sorry

end length_of_unfenced_side_l1318_131828


namespace loaves_count_l1318_131864

theorem loaves_count 
  (init_loaves : ℕ)
  (sold_percent : ℕ) 
  (bulk_purchase : ℕ)
  (bulk_discount_percent : ℕ)
  (evening_purchase : ℕ)
  (evening_discount_percent : ℕ)
  (final_loaves : ℕ)
  (h1 : init_loaves = 2355)
  (h2 : sold_percent = 30)
  (h3 : bulk_purchase = 750)
  (h4 : bulk_discount_percent = 20)
  (h5 : evening_purchase = 489)
  (h6 : evening_discount_percent = 15)
  (h7 : final_loaves = 2888) :
  let mid_morning_sold := init_loaves * sold_percent / 100
  let loaves_after_sale := init_loaves - mid_morning_sold
  let bulk_discount_loaves := bulk_purchase * bulk_discount_percent / 100
  let loaves_after_bulk_purchase := loaves_after_sale + bulk_purchase
  let evening_discount_loaves := evening_purchase * evening_discount_percent / 100
  let loaves_after_evening_purchase := loaves_after_bulk_purchase + evening_purchase
  loaves_after_evening_purchase = final_loaves :=
by
  sorry

end loaves_count_l1318_131864


namespace one_million_div_one_fourth_l1318_131870

theorem one_million_div_one_fourth : (1000000 : ℝ) / (1 / 4) = 4000000 := by
  sorry

end one_million_div_one_fourth_l1318_131870


namespace trees_falling_count_l1318_131855

/-- Definition of the conditions of the problem. --/
def initial_mahogany_trees : ℕ := 50
def initial_narra_trees : ℕ := 30
def trees_on_farm_after_typhoon : ℕ := 88

/-- The mathematical proof problem statement in Lean 4:
Prove the total number of trees that fell during the typhoon (N + M) is equal to 5,
given the conditions.
--/
theorem trees_falling_count (M N : ℕ) 
  (h1 : M = N + 1)
  (h2 : (initial_mahogany_trees - M + 3 * M) + (initial_narra_trees - N + 2 * N) = trees_on_farm_after_typhoon) :
  N + M = 5 := sorry

end trees_falling_count_l1318_131855


namespace betty_height_in_feet_l1318_131874

theorem betty_height_in_feet (dog_height carter_height betty_height : ℕ) (h1 : dog_height = 24) 
  (h2 : carter_height = 2 * dog_height) (h3 : betty_height = carter_height - 12) : betty_height / 12 = 3 :=
by
  sorry

end betty_height_in_feet_l1318_131874


namespace neg_sub_eq_sub_l1318_131820

theorem neg_sub_eq_sub (a b : ℝ) : - (a - b) = b - a := 
by
  sorry

end neg_sub_eq_sub_l1318_131820


namespace congruence_theorem_l1318_131859

def triangle_congruent_SSA (a b : ℝ) (gamma : ℝ) :=
  b * b = a * a + (-2 * a * 5 * Real.cos gamma) + 25

theorem congruence_theorem : triangle_congruent_SSA 3 5 (150 * Real.pi / 180) :=
by
  -- Proof is omitted, based on the problem's instruction.
  sorry

end congruence_theorem_l1318_131859


namespace remainder_7_pow_93_mod_12_l1318_131898

theorem remainder_7_pow_93_mod_12 : 7 ^ 93 % 12 = 7 := 
by
  -- the sequence repeats every two terms: 7, 1, 7, 1, ...
  sorry

end remainder_7_pow_93_mod_12_l1318_131898


namespace existence_of_unusual_100_digit_numbers_l1318_131819

theorem existence_of_unusual_100_digit_numbers :
  ∃ (n₁ n₂ : ℕ), 
  (n₁ = 10^100 - 1) ∧ (n₂ = 5 * 10^99 - 1) ∧ 
  (∀ x : ℕ, x = n₁ → (x^3 % 10^100 = x) ∧ (x^2 % 10^100 ≠ x)) ∧
  (∀ x : ℕ, x = n₂ → (x^3 % 10^100 = x) ∧ (x^2 % 10^100 ≠ x)) := 
sorry

end existence_of_unusual_100_digit_numbers_l1318_131819


namespace circle_center_sum_l1318_131811

theorem circle_center_sum (h k : ℝ) :
  (∀ x y : ℝ, (x - h) ^ 2 + (y - k) ^ 2 = x ^ 2 + y ^ 2 - 6 * x - 8 * y + 38) → h + k = 7 :=
by sorry

end circle_center_sum_l1318_131811


namespace cuboid_surface_area_two_cubes_l1318_131899

noncomputable def cuboid_surface_area (b : ℝ) : ℝ :=
  let l := 2 * b
  let w := b
  let h := b
  2 * (l * w + l * h + w * h)

theorem cuboid_surface_area_two_cubes (b : ℝ) : cuboid_surface_area b = 10 * b^2 := by
  sorry

end cuboid_surface_area_two_cubes_l1318_131899


namespace katy_books_ratio_l1318_131815

theorem katy_books_ratio (J : ℕ) (H1 : 8 + J + (J - 3) = 37) : J / 8 = 2 := 
by
  sorry

end katy_books_ratio_l1318_131815


namespace krish_spent_on_sweets_l1318_131832

noncomputable def initial_amount := 200.50
noncomputable def amount_per_friend := 25.20
noncomputable def remaining_amount := 114.85

noncomputable def total_given_to_friends := amount_per_friend * 2
noncomputable def amount_before_sweets := initial_amount - total_given_to_friends
noncomputable def amount_spent_on_sweets := amount_before_sweets - remaining_amount

theorem krish_spent_on_sweets : amount_spent_on_sweets = 35.25 :=
by
  sorry

end krish_spent_on_sweets_l1318_131832


namespace linear_function_no_third_quadrant_l1318_131840

theorem linear_function_no_third_quadrant (m : ℝ) (h : ∀ x y : ℝ, x < 0 → y < 0 → y ≠ -2 * x + 1 - m) : 
  m ≤ 1 :=
by
  sorry

end linear_function_no_third_quadrant_l1318_131840


namespace square_diagonal_cut_l1318_131877

/--
Given a square with side length 10,
prove that cutting along the diagonal results in two 
right-angled isosceles triangles with dimensions 10, 10, 10*sqrt(2).
-/
theorem square_diagonal_cut (side_length : ℕ) (triangle_side1 triangle_side2 hypotenuse : ℝ) 
  (h_side : side_length = 10)
  (h_triangle_side1 : triangle_side1 = 10) 
  (h_triangle_side2 : triangle_side2 = 10)
  (h_hypotenuse : hypotenuse = 10 * Real.sqrt 2) : 
  triangle_side1 = side_length ∧ triangle_side2 = side_length ∧ hypotenuse = side_length * Real.sqrt 2 :=
by
  sorry

end square_diagonal_cut_l1318_131877


namespace find_a3_a4_a5_l1318_131872

open Real

variables {a : ℕ → ℝ} (q : ℝ)

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

noncomputable def a_1 : ℝ := 3

def sum_of_first_three (a : ℕ → ℝ) : Prop :=
  a 0 + a 1 + a 2 = 21

def all_terms_positive (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < a n

theorem find_a3_a4_a5 (h1 : is_geometric_sequence a) (h2 : a 0 = a_1) (h3 : sum_of_first_three a) (h4 : all_terms_positive a) :
  a 2 + a 3 + a 4 = 84 :=
sorry

end find_a3_a4_a5_l1318_131872


namespace average_difference_l1318_131836

theorem average_difference : 
  (500 + 1000) / 2 - (100 + 500) / 2 = 450 := 
by
  sorry

end average_difference_l1318_131836


namespace total_trees_l1318_131802

-- Definitions based on the conditions
def ava_trees : ℕ := 9
def lily_trees : ℕ := ava_trees - 3

-- Theorem stating the total number of apple trees planted by Ava and Lily
theorem total_trees : ava_trees + lily_trees = 15 := by
  -- We skip the proof for now
  sorry

end total_trees_l1318_131802


namespace problem_1_problem_2_l1318_131842

open Set

-- First problem: when a = 2
theorem problem_1:
  ∀ (x : ℝ), 2 * x^2 - x - 1 > 0 ↔ (x < -(1 / 2) ∨ x > 1) :=
by
  sorry

-- Second problem: when a > -1
theorem problem_2 (a : ℝ) (h : a > -1) :
  ∀ (x : ℝ), 
    (if a = 0 then x - 1 > 0 else if a > 0 then  a * x ^ 2 + (1 - a) * x - 1 > 0 ↔ (x < -1 / a ∨ x > 1) 
    else a * x ^ 2 + (1 - a) * x - 1 > 0 ↔ (1 < x ∧ x < -1 / a)) :=
by
  sorry

end problem_1_problem_2_l1318_131842


namespace price_of_case_bulk_is_12_l1318_131892

noncomputable def price_per_can_grocery_store : ℚ := 6 / 12
noncomputable def price_per_can_bulk : ℚ := price_per_can_grocery_store - 0.25
def cans_per_case_bulk : ℕ := 48
noncomputable def price_per_case_bulk : ℚ := price_per_can_bulk * cans_per_case_bulk

theorem price_of_case_bulk_is_12 : price_per_case_bulk = 12 :=
by
  sorry

end price_of_case_bulk_is_12_l1318_131892


namespace base_conversion_l1318_131837

theorem base_conversion (b : ℕ) (h_pos : b > 0) :
  (1 * 6 ^ 2 + 2 * 6 ^ 1 + 5 * 6 ^ 0 = 2 * b ^ 2 + 2 * b + 1) → b = 4 :=
by
  sorry

end base_conversion_l1318_131837


namespace balloons_remaining_proof_l1318_131846

-- The initial number of balloons the clown has
def initial_balloons : ℕ := 3 * 12

-- The number of boys who buy balloons
def boys : ℕ := 3

-- The number of girls who buy balloons
def girls : ℕ := 12

-- The total number of children buying balloons
def total_children : ℕ := boys + girls

-- The remaining number of balloons after sales
def remaining_balloons (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

-- Problem statement: Proof that the remaining balloons are 21 given the conditions
theorem balloons_remaining_proof : remaining_balloons initial_balloons total_children = 21 := sorry

end balloons_remaining_proof_l1318_131846


namespace time_difference_halfway_point_l1318_131816

theorem time_difference_halfway_point 
  (T_d : ℝ) 
  (T_s : ℝ := 2 * T_d) 
  (H_d : ℝ := T_d / 2) 
  (H_s : ℝ := T_s / 2) 
  (diff_time : ℝ := H_s - H_d) : 
  T_d = 35 →
  T_s = 2 * T_d →
  diff_time = 17.5 :=
by
  intros h1 h2
  sorry

end time_difference_halfway_point_l1318_131816


namespace number_of_handshakes_l1318_131808

-- Definitions based on the conditions:
def number_of_teams : ℕ := 4
def number_of_women_per_team : ℕ := 2
def total_women : ℕ := number_of_teams * number_of_women_per_team

-- Each woman shakes hands with all others except her partner
def handshakes_per_woman : ℕ := total_women - 1 - (number_of_women_per_team - 1)

-- Calculate total handshakes, considering each handshake is counted twice
def total_handshakes : ℕ := (total_women * handshakes_per_woman) / 2

-- Statement to prove
theorem number_of_handshakes :
  total_handshakes = 24 := 
sorry

end number_of_handshakes_l1318_131808


namespace angle_between_clock_hands_at_7_25_l1318_131803

theorem angle_between_clock_hands_at_7_25 : 
  let degrees_per_hour := 30
  let minute_hand_position := (25 / 60 * 360 : ℝ)
  let hour_hand_position := (7 * degrees_per_hour + (25 / 60 * degrees_per_hour) : ℝ)
  abs (hour_hand_position - minute_hand_position) = 72.5 
  := by
  let degrees_per_hour := 30
  let minute_hand_position := (25 / 60 * 360 : ℝ)
  let hour_hand_position := (7 * degrees_per_hour + (25 / 60 * degrees_per_hour) : ℝ)
  sorry

end angle_between_clock_hands_at_7_25_l1318_131803


namespace front_crawl_speed_l1318_131860
   
   def swim_condition := 
     ∃ F : ℝ, -- Speed of front crawl in yards per minute
     (∃ t₁ t₂ d₁ d₂ : ℝ, -- t₁ is time for front crawl, t₂ is time for breaststroke, d₁ and d₂ are distances
               t₁ = 8 ∧
               t₂ = 4 ∧
               d₁ = t₁ * F ∧
               d₂ = t₂ * 35 ∧
               d₁ + d₂ = 500 ∧
               t₁ + t₂ = 12) ∧
     F = 45
   
   theorem front_crawl_speed : swim_condition :=
     by
       sorry -- Proof goes here, with given conditions satisfying F = 45
   
end front_crawl_speed_l1318_131860


namespace sufficient_condition_for_m_ge_9_l1318_131806

theorem sufficient_condition_for_m_ge_9
  (x m : ℝ)
  (p : |x - 4| ≤ 6)
  (q : x ≤ 1 + m)
  (h_sufficient : ∀ x, |x - 4| ≤ 6 → x ≤ 1 + m)
  (h_not_necessary : ∃ x, ¬(|x - 4| ≤ 6) ∧ x ≤ 1 + m) :
  m ≥ 9 := 
sorry

end sufficient_condition_for_m_ge_9_l1318_131806


namespace total_lunch_bill_l1318_131827

theorem total_lunch_bill (cost_hotdog cost_salad : ℝ) (h_hd : cost_hotdog = 5.36) (h_sd : cost_salad = 5.10) :
  cost_hotdog + cost_salad = 10.46 :=
by
  sorry

end total_lunch_bill_l1318_131827


namespace books_per_bookshelf_l1318_131894

theorem books_per_bookshelf (total_books bookshelves : ℕ) (h_total_books : total_books = 34) (h_bookshelves : bookshelves = 2) : total_books / bookshelves = 17 :=
by
  sorry

end books_per_bookshelf_l1318_131894


namespace common_difference_arithmetic_sequence_l1318_131825

variable (n d : ℝ) (a : ℝ := 7 - 2 * d) (an : ℝ := 37) (Sn : ℝ := 198)

theorem common_difference_arithmetic_sequence :
  7 + (n - 3) * d = 37 ∧ 
  396 = n * (44 - 2 * d) ∧
  Sn = n / 2 * (a + an) →
  (∃ d : ℝ, 7 + (n - 3) * d = 37 ∧ 396 = n * (44 - 2 * d)) :=
by
  sorry

end common_difference_arithmetic_sequence_l1318_131825


namespace velocity_at_t4_acceleration_is_constant_l1318_131812

noncomputable def s (t : ℝ) : ℝ := 3 * t^2 - 3 * t + 8

def v (t : ℝ) : ℝ := 6 * t - 3

def a : ℝ := 6

theorem velocity_at_t4 : v 4 = 21 := by 
  sorry

theorem acceleration_is_constant : a = 6 := by 
  sorry

end velocity_at_t4_acceleration_is_constant_l1318_131812


namespace B_completes_work_in_18_days_l1318_131867

variable {A B : ℝ}
variable (x : ℝ)

-- Conditions provided
def A_works_twice_as_fast_as_B (h1 : A = 2 * B) : Prop := true
def together_finish_work_in_6_days (h2 : B = 1 / x ∧ A = 2 / x ∧ 1 / 6 = (A + B)) : Prop := true

-- Theorem to prove: It takes B 18 days to complete the work independently
theorem B_completes_work_in_18_days (h1 : A = 2 * B) (h2 : B = 1 / x ∧ A = 2 / x ∧ 1 / 6 = (A + B)) : x = 18 := by
  sorry

end B_completes_work_in_18_days_l1318_131867


namespace max_gcd_of_15n_plus_4_and_9n_plus_2_eq_2_l1318_131876

theorem max_gcd_of_15n_plus_4_and_9n_plus_2_eq_2 (n : ℕ) (hn : n > 0) :
  ∃ m, m = Nat.gcd (15 * n + 4) (9 * n + 2) ∧ m ≤ 2 :=
by
  sorry

end max_gcd_of_15n_plus_4_and_9n_plus_2_eq_2_l1318_131876


namespace percentage_of_180_out_of_360_equals_50_l1318_131835

theorem percentage_of_180_out_of_360_equals_50 :
  (180 / 360 : ℚ) * 100 = 50 := 
sorry

end percentage_of_180_out_of_360_equals_50_l1318_131835


namespace least_number_to_add_l1318_131853

theorem least_number_to_add (k n : ℕ) (h : k = 1015) (m : n = 25) : 
  ∃ x : ℕ, (k + x) % n = 0 ∧ x = 10 := by
  sorry

end least_number_to_add_l1318_131853


namespace conic_section_is_hyperbola_l1318_131897

-- Definitions for the conditions in the problem
def conic_section_equation (x y : ℝ) := (x - 4) ^ 2 = 5 * (y + 2) ^ 2 - 45

-- The theorem that we need to prove
theorem conic_section_is_hyperbola : ∀ x y : ℝ, (conic_section_equation x y) → "H" = "H" :=
by
  intro x y h
  sorry

end conic_section_is_hyperbola_l1318_131897


namespace travel_time_by_raft_l1318_131889

variable (U V : ℝ) -- U: speed of the steamboat, V: speed of the river current
variable (S : ℝ) -- S: distance between cities A and B

-- Conditions
variable (h1 : S = 12 * U - 15 * V) -- Distance calculation, city B to city A
variable (h2 : S = 8 * U + 10 * V)  -- Distance calculation, city A to city B
variable (T : ℝ) -- Time taken on a raft

-- Proof problem
theorem travel_time_by_raft : T = 60 :=
by
  sorry


end travel_time_by_raft_l1318_131889


namespace xiao_zhang_return_distance_xiao_zhang_no_refuel_needed_l1318_131866

def total_distance : ℕ :=
  15 - 3 + 16 - 11 + 10 - 12 + 4 - 15 + 16 - 18

def fuel_consumption_per_km : ℝ := 0.6
def initial_fuel : ℝ := 72.2

theorem xiao_zhang_return_distance :
  total_distance = 2 := by
  sorry

theorem xiao_zhang_no_refuel_needed :
  (initial_fuel - fuel_consumption_per_km * (|15| + |3| + |16| + |11| + |10| + |12| + |4| + |15| + |16| + |18|)) >= 0 := by
  sorry

end xiao_zhang_return_distance_xiao_zhang_no_refuel_needed_l1318_131866


namespace average_score_l1318_131861

theorem average_score (T : ℝ) (M F : ℝ) (avgM avgF : ℝ) 
  (h1 : M = 0.4 * T) 
  (h2 : M + F = T) 
  (h3 : avgM = 75) 
  (h4 : avgF = 80) : 
  (75 * M + 80 * F) / T = 78 := 
  by 
  sorry

end average_score_l1318_131861


namespace convex_quadrilateral_division_l1318_131895

-- Definitions for convex quadrilateral and some basic geometric objects.
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Quadrilateral :=
  (A B C D : Point)
  (convex : ∀ (X Y Z : Point), (X ≠ Y) ∧ (Y ≠ Z) ∧ (Z ≠ X))

-- Definitions for lines and midpoints.
def is_midpoint (M X Y : Point) : Prop :=
  M.x = (X.x + Y.x) / 2 ∧ M.y = (X.y + Y.y) / 2

-- Preliminary to determining equal area division.
def equal_area_division (Q : Quadrilateral) (L : Point → Point → Prop) : Prop :=
  ∃ F,
    is_midpoint F Q.A Q.B ∧
    -- Assuming some way to relate area with F and L
    L Q.D F ∧
    -- Placeholder for equality of areas (details depend on how we calculate area)
    sorry

-- Problem statement in Lean 4
theorem convex_quadrilateral_division (Q : Quadrilateral) :
  ∃ L, equal_area_division Q L :=
by
  -- Proof will be constructed here based on steps in the solution
  sorry

end convex_quadrilateral_division_l1318_131895


namespace find_angle_C_l1318_131800

open Real

theorem find_angle_C (a b C A B : ℝ) 
  (h1 : a^2 + b^2 = 6 * a * b * cos C)
  (h2 : sin C ^ 2 = 2 * sin A * sin B) :
  C = π / 3 := 
  sorry

end find_angle_C_l1318_131800


namespace fractions_order_l1318_131844

theorem fractions_order :
  let frac1 := (21 : ℚ) / (17 : ℚ)
  let frac2 := (23 : ℚ) / (19 : ℚ)
  let frac3 := (25 : ℚ) / (21 : ℚ)
  frac3 < frac2 ∧ frac2 < frac1 :=
by sorry

end fractions_order_l1318_131844
