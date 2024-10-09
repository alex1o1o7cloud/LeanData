import Mathlib

namespace ages_of_patients_l1671_167114

theorem ages_of_patients (x y : ℕ) 
  (h1 : x - y = 44) 
  (h2 : x * y = 1280) : 
  (x = 64 ∧ y = 20) ∨ (x = 20 ∧ y = 64) := by
  sorry

end ages_of_patients_l1671_167114


namespace sum_even_squares_sum_odd_squares_l1671_167136

open scoped BigOperators

def sumOfSquaresEven (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, (2 * (i + 1))^2

def sumOfSquaresOdd (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, (2 * i + 1)^2

theorem sum_even_squares (n : ℕ) :
  sumOfSquaresEven n = (2 * n * (n - 1) * (2 * n - 1)) / 3 := by
    sorry

theorem sum_odd_squares (n : ℕ) :
  sumOfSquaresOdd n = (n * (4 * n^2 - 1)) / 3 := by
    sorry

end sum_even_squares_sum_odd_squares_l1671_167136


namespace range_of_a_l1671_167103

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (ax^2 - ax + 1 ≤ 0)) ↔ 0 ≤ a ∧ a < 4 :=
by
  sorry

end range_of_a_l1671_167103


namespace inequality_transform_l1671_167109

variable {x y : ℝ}

theorem inequality_transform (h : x < y) : - (x / 2) > - (y / 2) :=
sorry

end inequality_transform_l1671_167109


namespace peter_reads_more_books_l1671_167150

-- Definitions and conditions
def total_books : ℕ := 20
def peter_percentage_read : ℕ := 40
def brother_percentage_read : ℕ := 10

def percentage_to_count (percentage : ℕ) (total : ℕ) : ℕ := (percentage * total) / 100

-- Main statement to prove
theorem peter_reads_more_books :
  percentage_to_count peter_percentage_read total_books - percentage_to_count brother_percentage_read total_books = 6 :=
by
  sorry

end peter_reads_more_books_l1671_167150


namespace derivative_of_f_l1671_167154

def f (x : ℝ) : ℝ := 2 * x + 3

theorem derivative_of_f :
  ∀ x : ℝ, (deriv f x) = 2 :=
by 
  sorry

end derivative_of_f_l1671_167154


namespace lcm_of_6_8_10_l1671_167181

theorem lcm_of_6_8_10 : Nat.lcm (Nat.lcm 6 8) 10 = 120 := 
  by sorry

end lcm_of_6_8_10_l1671_167181


namespace time_period_simple_interest_l1671_167144

theorem time_period_simple_interest 
  (P : ℝ) (R18 R12 : ℝ) (additional_interest : ℝ) (T : ℝ) :
  P = 2500 →
  R18 = 0.18 →
  R12 = 0.12 →
  additional_interest = 300 →
  P * R18 * T = P * R12 * T + additional_interest →
  T = 2 :=
by
  intros P_val R18_val R12_val add_int_val interest_eq
  rw [P_val, R18_val, R12_val, add_int_val] at interest_eq
  -- Continue the proof here
  sorry

end time_period_simple_interest_l1671_167144


namespace chromosome_structure_l1671_167120

-- Definitions related to the conditions of the problem
def chromosome : Type := sorry  -- Define type for chromosome (hypothetical representation)
def has_centromere (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome has centromere
def contains_one_centromere (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome contains one centromere
def has_one_chromatid (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome has one chromatid
def has_two_chromatids (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome has two chromatids
def is_chromatin (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome is chromatin

-- Define the problem statement
theorem chromosome_structure (c : chromosome) :
  contains_one_centromere c ∧ ¬has_one_chromatid c ∧ ¬has_two_chromatids c ∧ ¬is_chromatin c := sorry

end chromosome_structure_l1671_167120


namespace trigonometric_identity_l1671_167156

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) : 2 * Real.sin θ + Real.sin θ * Real.cos θ = 2 := by
  sorry

end trigonometric_identity_l1671_167156


namespace sets_given_to_friend_l1671_167151

theorem sets_given_to_friend (total_cards : ℕ) (total_given_away : ℕ) (sets_brother : ℕ) 
  (sets_sister : ℕ) (cards_per_set : ℕ) (sets_friend : ℕ) 
  (h1 : total_cards = 365) 
  (h2 : total_given_away = 195) 
  (h3 : sets_brother = 8) 
  (h4 : sets_sister = 5) 
  (h5 : cards_per_set = 13) 
  (h6 : total_given_away = (sets_brother + sets_sister + sets_friend) * cards_per_set) : 
  sets_friend = 2 :=
by
  sorry

end sets_given_to_friend_l1671_167151


namespace number_of_int_pairs_l1671_167177

theorem number_of_int_pairs (x y : ℤ) (h : x^2 + 2 * y^2 < 25) : 
  ∃ S : Finset (ℤ × ℤ), S.card = 55 ∧ ∀ (a : ℤ × ℤ), a ∈ S ↔ a.1^2 + 2 * a.2^2 < 25 :=
sorry

end number_of_int_pairs_l1671_167177


namespace min_number_of_trials_sum_15_min_number_of_trials_sum_at_least_15_l1671_167128

noncomputable def min_trials_sum_of_15 : ℕ :=
  15

noncomputable def min_trials_sum_at_least_15 : ℕ :=
  8

theorem min_number_of_trials_sum_15 (x : ℕ) :
  (∀ (x : ℕ), (103/108 : ℝ)^x < (1/2 : ℝ) → x >= min_trials_sum_of_15) := sorry

theorem min_number_of_trials_sum_at_least_15 (x : ℕ) :
  (∀ (x : ℕ), (49/54 : ℝ)^x < (1/2 : ℝ) → x >= min_trials_sum_at_least_15) := sorry

end min_number_of_trials_sum_15_min_number_of_trials_sum_at_least_15_l1671_167128


namespace tickets_needed_to_ride_l1671_167130

noncomputable def tickets_required : Float :=
let ferris_wheel := 3.5
let roller_coaster := 8.0
let bumper_cars := 5.0
let additional_ride_discount := 0.5
let newspaper_coupon := 1.5
let teacher_discount := 2.0

let total_cost_without_discounts := ferris_wheel + roller_coaster + bumper_cars
let total_additional_discounts := additional_ride_discount * 2
let total_coupons_discounts := newspaper_coupon + teacher_discount

let total_cost_with_discounts := total_cost_without_discounts - total_additional_discounts - total_coupons_discounts
total_cost_with_discounts

theorem tickets_needed_to_ride : tickets_required = 12.0 := by
  sorry

end tickets_needed_to_ride_l1671_167130


namespace lcm_nuts_bolts_l1671_167190

theorem lcm_nuts_bolts : Nat.lcm 13 8 = 104 := 
sorry

end lcm_nuts_bolts_l1671_167190


namespace sum_of_a_for_quadratic_has_one_solution_l1671_167183

noncomputable def discriminant (a : ℝ) : ℝ := (a + 12)^2 - 4 * 3 * 16

theorem sum_of_a_for_quadratic_has_one_solution : 
  (∀ a : ℝ, discriminant a = 0) → 
  (-12 + 8 * Real.sqrt 3) + (-12 - 8 * Real.sqrt 3) = -24 :=
by
  intros h
  simp [discriminant] at h
  sorry

end sum_of_a_for_quadratic_has_one_solution_l1671_167183


namespace common_roots_correct_l1671_167149

noncomputable section
def common_roots_product (A B : ℝ) : ℝ :=
  let p := sorry
  let q := sorry
  p * q

theorem common_roots_correct (A B : ℝ) (h1 : ∀ x, x^3 + 2*A*x + 20 = 0 → x = p ∨ x = q ∨ x = r) 
    (h2 : ∀ x, x^3 + B*x^2 + 100 = 0 → x = p ∨ x = q ∨ x = s)
    (h_sum1 : p + q + r = 0) 
    (h_sum2 : p + q + s = -B)
    (h_prod1 : p * q * r = -20) 
    (h_prod2 : p * q * s = -100) : 
    common_roots_product A B = 10 * (2000)^(1/3) ∧ 15 = 10 + 3 + 2 :=
by
  sorry

end common_roots_correct_l1671_167149


namespace mean_score_for_exam_l1671_167113

variable (M SD : ℝ)

-- Define the conditions
def condition1 : Prop := 58 = M - 2 * SD
def condition2 : Prop := 98 = M + 3 * SD

-- The problem statement
theorem mean_score_for_exam (h1 : condition1 M SD) (h2 : condition2 M SD) : M = 74 :=
sorry

end mean_score_for_exam_l1671_167113


namespace new_bookstore_acquisition_l1671_167108

theorem new_bookstore_acquisition (x : ℝ) 
  (h1 : (1 / 2) * x + (1 / 4) * x + 50 = x - 200) : x = 1000 :=
by {
  sorry
}

end new_bookstore_acquisition_l1671_167108


namespace trapezium_area_correct_l1671_167160

-- Define the lengths of the parallel sides and the distance between them
def a := 24  -- length of the first parallel side in cm
def b := 14  -- length of the second parallel side in cm
def h := 18  -- distance between the parallel sides in cm

-- Define the area calculation function for the trapezium
def trapezium_area (a b h : ℕ) : ℕ :=
  1 / 2 * (a + b) * h

-- The theorem to prove that the area of the given trapezium is 342 square centimeters
theorem trapezium_area_correct : trapezium_area a b h = 342 :=
  sorry

end trapezium_area_correct_l1671_167160


namespace find_angle_D_l1671_167189

theorem find_angle_D (A B C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = 2 * D)
  (h3 : B = C + 40) : D = 70 := by
  sorry

end find_angle_D_l1671_167189


namespace find_xy_integers_l1671_167135

theorem find_xy_integers (x y : ℤ) (h : x^3 + 2 * x * y = 7) :
  (x, y) = (-7, -25) ∨ (x, y) = (-1, -4) ∨ (x, y) = (1, 3) ∨ (x, y) = (7, -24) :=
sorry

end find_xy_integers_l1671_167135


namespace tan_subtraction_l1671_167143

theorem tan_subtraction (α β : ℝ) (h₁ : Real.tan α = 11) (h₂ : Real.tan β = 5) : 
  Real.tan (α - β) = 3 / 28 := 
  sorry

end tan_subtraction_l1671_167143


namespace base5_addition_l1671_167165

theorem base5_addition : 
  (14 : ℕ) + (132 : ℕ) = (101 : ℕ) :=
by {
  sorry
}

end base5_addition_l1671_167165


namespace zero_in_M_l1671_167152

def M : Set Int := {-1, 0, 1}

theorem zero_in_M : 0 ∈ M :=
  by
  -- Proof is omitted
  sorry

end zero_in_M_l1671_167152


namespace probability_of_same_color_correct_l1671_167134

def total_plates : ℕ := 13
def red_plates : ℕ := 7
def blue_plates : ℕ := 6

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def total_ways_to_choose_two : ℕ := choose total_plates 2
noncomputable def ways_to_choose_two_red : ℕ := choose red_plates 2
noncomputable def ways_to_choose_two_blue : ℕ := choose blue_plates 2

noncomputable def ways_to_choose_two_same_color : ℕ :=
  ways_to_choose_two_red + ways_to_choose_two_blue

noncomputable def probability_same_color : ℚ :=
  ways_to_choose_two_same_color / total_ways_to_choose_two

theorem probability_of_same_color_correct :
  probability_same_color = 4 / 9 := by
  sorry

end probability_of_same_color_correct_l1671_167134


namespace minimum_value_l1671_167198

open Real

theorem minimum_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 / (y - 2) + y^2 / (x - 2)) ≥ 12 :=
sorry

end minimum_value_l1671_167198


namespace find_pairs_l1671_167106

theorem find_pairs (a b : ℕ) (h1 : a + b = 60) (h2 : Nat.lcm a b = 72) : (a = 24 ∧ b = 36) ∨ (a = 36 ∧ b = 24) := 
sorry

end find_pairs_l1671_167106


namespace edge_length_in_mm_l1671_167100

-- Definitions based on conditions
def cube_volume (a : ℝ) : ℝ := a^3

axiom volume_of_dice : cube_volume 2 = 8

-- Statement of the theorem to be proved
theorem edge_length_in_mm : ∃ (a : ℝ), cube_volume a = 8 ∧ a * 10 = 20 := sorry

end edge_length_in_mm_l1671_167100


namespace tom_sold_price_l1671_167101

noncomputable def original_price : ℝ := 200
noncomputable def tripled_price (price : ℝ) : ℝ := 3 * price
noncomputable def sold_price (price : ℝ) : ℝ := 0.4 * price

theorem tom_sold_price : sold_price (tripled_price original_price) = 240 := 
by
  sorry

end tom_sold_price_l1671_167101


namespace find_positive_integers_l1671_167158

noncomputable def positive_integer_solutions_ineq (x : ℕ) : Prop :=
  x > 0 ∧ (x : ℝ) < 4

theorem find_positive_integers (x : ℕ) : 
  (x > 0 ∧ (↑x - 3)/3 < 7 - 5*(↑x)/3) ↔ positive_integer_solutions_ineq x :=
by
  sorry

end find_positive_integers_l1671_167158


namespace packages_delivered_by_third_butcher_l1671_167192

theorem packages_delivered_by_third_butcher 
  (x y z : ℕ) 
  (h1 : x = 10) 
  (h2 : y = 7) 
  (h3 : 4 * x + 4 * y + 4 * z = 100) : 
  z = 8 :=
by { sorry }

end packages_delivered_by_third_butcher_l1671_167192


namespace handshakes_count_l1671_167184

def num_teams : ℕ := 4
def players_per_team : ℕ := 2
def total_players : ℕ := num_teams * players_per_team
def shakeable_players (total : ℕ) : ℕ := total * (total - players_per_team) / 2

theorem handshakes_count :
  shakeable_players total_players = 24 :=
by
  sorry

end handshakes_count_l1671_167184


namespace students_not_yet_pictured_l1671_167115

def students_in_class : ℕ := 24
def students_before_lunch : ℕ := students_in_class / 3
def students_after_lunch_before_gym : ℕ := 10
def total_students_pictures_taken : ℕ := students_before_lunch + students_after_lunch_before_gym

theorem students_not_yet_pictured : total_students_pictures_taken = 18 → students_in_class - total_students_pictures_taken = 6 := by
  intros h
  rw [h]
  rfl

end students_not_yet_pictured_l1671_167115


namespace sum_a1_a3_a5_l1671_167111

-- Definitions
variable (a : ℕ → ℕ)
variable (b : ℕ → ℕ)

-- Conditions
axiom initial_condition : a 1 = 16
axiom relationship_ak_bk : ∀ k, b k = a k / 2
axiom ak_next : ∀ k, a (k + 1) = a k + 2 * (b k)

-- Theorem Statement
theorem sum_a1_a3_a5 : a 1 + a 3 + a 5 = 336 :=
by
  sorry

end sum_a1_a3_a5_l1671_167111


namespace total_pamphlets_correct_l1671_167123

-- Define the individual printing rates and hours
def Mike_pre_break_rate := 600
def Mike_pre_break_hours := 9
def Mike_post_break_rate := Mike_pre_break_rate / 3
def Mike_post_break_hours := 2

def Leo_pre_break_rate := 2 * Mike_pre_break_rate
def Leo_pre_break_hours := Mike_pre_break_hours / 3
def Leo_post_first_break_rate := Leo_pre_break_rate / 2
def Leo_post_second_break_rate := Leo_post_first_break_rate / 2

def Sally_pre_break_rate := 3 * Mike_pre_break_rate
def Sally_pre_break_hours := Mike_post_break_hours / 2
def Sally_post_break_rate := Leo_post_first_break_rate
def Sally_post_break_hours := 1

-- Calculate the total number of pamphlets printed by each person
def Mike_pamphlets := 
  (Mike_pre_break_rate * Mike_pre_break_hours) + (Mike_post_break_rate * Mike_post_break_hours)

def Leo_pamphlets := 
  (Leo_pre_break_rate * 1) + (Leo_post_first_break_rate * 1) + (Leo_post_second_break_rate * 1)

def Sally_pamphlets := 
  (Sally_pre_break_rate * Sally_pre_break_hours) + (Sally_post_break_rate * Sally_post_break_hours)

-- Calculate the total number of pamphlets printed by all three
def total_pamphlets := Mike_pamphlets + Leo_pamphlets + Sally_pamphlets

theorem total_pamphlets_correct : total_pamphlets = 10700 := by
  sorry

end total_pamphlets_correct_l1671_167123


namespace regression_line_passes_through_center_l1671_167191

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 1.5 * x - 15

-- Define the condition of the sample center point
def sample_center (x_bar y_bar : ℝ) : Prop :=
  y_bar = regression_eq x_bar

-- The proof goal
theorem regression_line_passes_through_center (x_bar y_bar : ℝ) (h : sample_center x_bar y_bar) :
  y_bar = 1.5 * x_bar - 15 :=
by
  -- Using the given condition as hypothesis
  exact h

end regression_line_passes_through_center_l1671_167191


namespace sum_q_p_is_minus_12_l1671_167188

noncomputable def p (x : ℝ) : ℝ := x^2 - 3 * x + 2

noncomputable def q (x : ℝ) : ℝ := -x^2

theorem sum_q_p_is_minus_12 :
  (q (p 0) + q (p 1) + q (p 2) + q (p 3) + q (p 4)) = -12 :=
by
  sorry

end sum_q_p_is_minus_12_l1671_167188


namespace total_salary_correct_l1671_167153

-- Define the daily salaries
def owner_salary : ℕ := 20
def manager_salary : ℕ := 15
def cashier_salary : ℕ := 10
def clerk_salary : ℕ := 5
def bagger_salary : ℕ := 3

-- Define the number of employees
def num_owners : ℕ := 1
def num_managers : ℕ := 3
def num_cashiers : ℕ := 5
def num_clerks : ℕ := 7
def num_baggers : ℕ := 9

-- Define the total salary calculation
def total_daily_salary : ℕ :=
  (num_owners * owner_salary) +
  (num_managers * manager_salary) +
  (num_cashiers * cashier_salary) +
  (num_clerks * clerk_salary) +
  (num_baggers * bagger_salary)

-- The theorem we need to prove
theorem total_salary_correct :
  total_daily_salary = 177 :=
by
  -- Proof can be filled in later
  sorry

end total_salary_correct_l1671_167153


namespace ex_sq_sum_l1671_167178

theorem ex_sq_sum (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = -1) : x^2 + y^2 = 11 :=
by
  sorry

end ex_sq_sum_l1671_167178


namespace inequality_proof_l1671_167185

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 1/3) : 
  (1 - a) * (1 - b) ≤ 25/36 :=
by
  sorry

end inequality_proof_l1671_167185


namespace quadratic_root_relationship_l1671_167166

noncomputable def roots_of_quadratic (a b c: ℚ) (h_nonzero: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h_root_relation: ∀ (s₁ s₂ : ℚ), s₁ + s₂ = -c ∧ s₁ * s₂ = a → (3 * s₁) + (3 * s₂) = -a ∧ (3 * s₁) * (3 * s₂) = b) : Prop :=
  b / c = 27

theorem quadratic_root_relationship (a b c : ℚ) (h_nonzero: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_root_relation: ∀ (s₁ s₂ : ℚ), s₁ + s₂ = -c ∧ s₁ * s₂ = a → (3 * s₁) + (3 * s₂) = -a ∧ (3 * s₁) * (3 * s₂) = b) : 
  roots_of_quadratic a b c h_nonzero h_root_relation := 
by 
  sorry

end quadratic_root_relationship_l1671_167166


namespace geom_sequence_next_term_l1671_167182

def geom_seq (a r : ℕ → ℤ) (i : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n * r i

theorem geom_sequence_next_term (y : ℤ) (a : ℕ → ℤ) (r : ℕ → ℤ) (n : ℕ) : 
  geom_seq a r 0 →
  a 0 = 3 →
  a 1 = 9 * y^2 →
  a 2 = 27 * y^4 →
  a 3 = 81 * y^6 →
  r 0 = 3 * y^2 →
  a 4 = 243 * y^8 :=
by
  intro h_seq h1 h2 h3 h4 hr
  sorry

end geom_sequence_next_term_l1671_167182


namespace solve_equation_l1671_167148

theorem solve_equation (x : ℝ) (h1 : x + 2 ≠ 0) (h2 : 3 - x ≠ 0) :
  (3 * x - 5) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -15 / 2 :=
by
  sorry

end solve_equation_l1671_167148


namespace exp_log_pb_eq_log_ba_l1671_167171

noncomputable def log_b (b a : ℝ) := Real.log a / Real.log b

theorem exp_log_pb_eq_log_ba (a b p : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : p = log_b b (log_b b a) / log_b b a) :
  a^p = log_b b a :=
by
  sorry

end exp_log_pb_eq_log_ba_l1671_167171


namespace radius_of_circle_l1671_167140

theorem radius_of_circle (r : ℝ) (h : 3 * (2 * Real.pi * r) = 2 * (Real.pi * r ^ 2)) : r = 3 := by
  sorry

end radius_of_circle_l1671_167140


namespace train_speed_proof_l1671_167193

noncomputable def train_speed_kmh (length_train : ℝ) (time_crossing : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := length_train / time_crossing
  let train_speed_ms := relative_speed - man_speed_ms
  train_speed_ms * (3600 / 1000)

theorem train_speed_proof :
  train_speed_kmh 150 8 7 = 60.5 :=
by
  sorry

end train_speed_proof_l1671_167193


namespace fg_sqrt2_eq_neg5_l1671_167107

noncomputable def f (x : ℝ) : ℝ := 4 - 3 * x
noncomputable def g (x : ℝ) : ℝ := x^2 + 1

theorem fg_sqrt2_eq_neg5 : f (g (Real.sqrt 2)) = -5 := by
  sorry

end fg_sqrt2_eq_neg5_l1671_167107


namespace ab_sum_l1671_167110

theorem ab_sum (a b : ℝ) (h₁ : ∀ x : ℝ, (x + a) * (x + 8) = x^2 + b * x + 24) (h₂ : 8 * a = 24) : a + b = 14 :=
by
  sorry

end ab_sum_l1671_167110


namespace table_seating_problem_l1671_167197

theorem table_seating_problem 
  (n : ℕ) 
  (label : ℕ → ℕ) 
  (h1 : label 31 = 31) 
  (h2 : label (31 - 17 + n) = 14) 
  (h3 : label (31 + 16) = 7) 
  : n = 41 :=
sorry

end table_seating_problem_l1671_167197


namespace find_y_l1671_167146

noncomputable def x : Real := 1.6666666666666667
def y : Real := 5

theorem find_y (h : x ≠ 0) (h1 : (x * y) / 3 = x^2) : y = 5 := 
by sorry

end find_y_l1671_167146


namespace cost_price_percentage_l1671_167121

theorem cost_price_percentage (MP CP : ℝ) 
  (h1 : MP * 0.9 = CP * (72 / 70))
  (h2 : CP / MP * 100 = 87.5) :
  CP / MP = 0.875 :=
by {
  sorry
}

end cost_price_percentage_l1671_167121


namespace range_of_t_l1671_167164

noncomputable def a_n (n : ℕ) (t : ℝ) : ℝ := -n + t
noncomputable def b_n (n : ℕ) : ℝ := 3^(n-3)
noncomputable def c_n (n : ℕ) (t : ℝ) : ℝ := 
  let a := a_n n t 
  let b := b_n n
  (a + b) / 2 + (|a - b|) / 2

theorem range_of_t (t : ℝ) (h : ∀ n : ℕ, n > 0 → c_n n t ≥ c_n 3 t) : 10/3 < t ∧ t < 5 :=
    sorry

end range_of_t_l1671_167164


namespace problem1_problem2_l1671_167157

open Real

-- Proof problem for the first expression
theorem problem1 : 
  (-2^2 * (1 / 4) + 4 / (4/9) + (-1) ^ 2023 = 7) :=
by 
  sorry

-- Proof problem for the second expression
theorem problem2 : 
  (-1 ^ 4 + abs (2 - (-3)^2) + (1/2) / (-3/2) = 17/3) :=
by 
  sorry

end problem1_problem2_l1671_167157


namespace area_ratio_l1671_167147

noncomputable def AreaOfTrapezoid (AD BC : ℝ) (R : ℝ) : ℝ :=
  let s_π := Real.pi
  let height1 := 2 -- One of the heights considered
  let height2 := 14 -- Another height considered
  (AD + BC) / 2 * height1  -- First case area
  -- Here we assume the area uses sine which is arc-related, but provide fixed coefficients for area representation

noncomputable def AreaOfRectangle (R : ℝ) : ℝ :=
  let d := 2 * R
  -- Using the equation for area discussed
  d * d / 2

theorem area_ratio (AD BC : ℝ) (R : ℝ) (hAD : AD = 16) (hBC : BC = 12) (hR : R = 10) :
  let area_trap := AreaOfTrapezoid AD BC R
  let area_rect := AreaOfRectangle R
  area_trap / area_rect = 1 / 2 ∨ area_trap / area_rect = 49 / 50 :=
by
  sorry

end area_ratio_l1671_167147


namespace find_k_l1671_167133

def a : ℝ × ℝ := (2, 1)
def b (k : ℝ) : ℝ × ℝ := (-2, k)
def vec_op (a b : ℝ × ℝ) : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)

noncomputable def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_k (k : ℝ) : dot_prod a (vec_op a (b k)) = 0 → k = 14 :=
by
  sorry

end find_k_l1671_167133


namespace total_age_difference_is_twelve_l1671_167124

variable {A B C : ℕ}

theorem total_age_difference_is_twelve (h1 : A + B > B + C) (h2 : C = A - 12) :
  (A + B) - (B + C) = 12 :=
by
  sorry

end total_age_difference_is_twelve_l1671_167124


namespace ratio_a_c_l1671_167122

theorem ratio_a_c {a b c : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : ∀ x : ℝ, x^2 + a * x + b = 0 → 2 * x^2 + 2 * a * x + 2 * b = 0)
  (h2 : ∀ x : ℝ, x^2 + b * x + c = 0 → x^2 + b * x + c = 0) :
  a / c = 1 / 8 :=
by
  sorry

end ratio_a_c_l1671_167122


namespace find_distance_to_school_l1671_167169

variable (v d : ℝ)
variable (h_rush_hour : d = v * (1 / 2))
variable (h_no_traffic : d = (v + 20) * (1 / 4))

theorem find_distance_to_school (h_rush_hour : d = v * (1 / 2)) (h_no_traffic : d = (v + 20) * (1 / 4)) : d = 10 := by
  sorry

end find_distance_to_school_l1671_167169


namespace total_nickels_l1671_167163

-- Definition of the number of original nickels Mary had
def original_nickels := 7

-- Definition of the number of nickels her dad gave her
def added_nickels := 5

-- Prove that the total number of nickels Mary has now is 12
theorem total_nickels : original_nickels + added_nickels = 12 := by
  sorry

end total_nickels_l1671_167163


namespace frozen_yogurt_price_l1671_167180

variable (F G S : ℝ) -- Define the variables F, G, S as real numbers

-- Define the conditions given in the problem
variable (h1 : 5 * F + 2 * G + 5 * S = 55)
variable (h2 : S = 5)
variable (h3 : G = 1 / 2 * F)

-- State the proof goal
theorem frozen_yogurt_price : F = 5 :=
by
  sorry

end frozen_yogurt_price_l1671_167180


namespace smallest_a_l1671_167167

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

theorem smallest_a (a : ℕ) (h1 : is_factor 112 (a * 43 * 62 * 1311)) (h2 : is_factor 33 (a * 43 * 62 * 1311)) : a = 1848 :=
by
  sorry

end smallest_a_l1671_167167


namespace certain_number_is_60_l1671_167102

theorem certain_number_is_60 
  (A J C : ℕ) 
  (h1 : A = 4) 
  (h2 : C = 8) 
  (h3 : A = (1 / 2) * J) :
  3 * (A + J + C) = 60 :=
by sorry

end certain_number_is_60_l1671_167102


namespace least_positive_integer_remainder_l1671_167176

theorem least_positive_integer_remainder :
  ∃ n : ℕ, (n > 0) ∧ (n % 5 = 1) ∧ (n % 4 = 2) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 5 = 1) ∧ (m % 4 = 2) → n ≤ m) :=
sorry

end least_positive_integer_remainder_l1671_167176


namespace students_study_both_l1671_167159

-- Define variables and conditions
variable (total_students G B G_and_B : ℕ)
variable (G_percent B_percent : ℝ)
variable (total_students_eq : total_students = 300)
variable (G_percent_eq : G_percent = 0.8)
variable (B_percent_eq : B_percent = 0.5)
variable (G_eq : G = G_percent * total_students)
variable (B_eq : B = B_percent * total_students)
variable (students_eq : total_students = G + B - G_and_B)

-- Theorem statement
theorem students_study_both :
  G_and_B = 90 :=
by
  sorry

end students_study_both_l1671_167159


namespace cost_price_eq_l1671_167168

variables (x : ℝ)

def f (x : ℝ) : ℝ := x * (1 + 0.30)
def g (x : ℝ) : ℝ := f x * 0.80

theorem cost_price_eq (h : g x = 2080) : x * (1 + 0.30) * 0.80 = 2080 :=
by sorry

end cost_price_eq_l1671_167168


namespace smallest_positive_n_l1671_167126

theorem smallest_positive_n (n : ℕ) (h1 : 0 < n) (h2 : gcd (8 * n - 3) (6 * n + 4) > 1) : n = 1 :=
sorry

end smallest_positive_n_l1671_167126


namespace sally_initial_cards_l1671_167195

variable (initial_cards : ℕ)

-- Define the conditions
def cards_given := 41
def cards_lost := 20
def cards_now := 48

-- Define the proof problem
theorem sally_initial_cards :
  initial_cards + cards_given - cards_lost = cards_now → initial_cards = 27 :=
by
  intro h
  sorry

end sally_initial_cards_l1671_167195


namespace part1_part2_l1671_167104

open Real

def f (x a : ℝ) : ℝ :=
  x^2 + a * x + 3

theorem part1 (x : ℝ) (h : x^2 - 4 * x + 3 < 0) :
  1 < x ∧ x < 3 :=
  sorry

theorem part2 (a : ℝ) (h : ∀ x, f x a > 0) :
  -2 * sqrt 3 < a ∧ a < 2 * sqrt 3 :=
  sorry

end part1_part2_l1671_167104


namespace constant_function_solution_l1671_167142

theorem constant_function_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, 2 * f x = f (x + y) + f (x + 2 * y)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c := by
  sorry

end constant_function_solution_l1671_167142


namespace tim_meditation_time_l1671_167145

-- Definitions of the conditions:
def time_reading_week (t_reading : ℕ) : Prop := t_reading = 14
def twice_as_much_reading (t_reading t_meditate : ℕ) : Prop := t_reading = 2 * t_meditate

-- The theorem to prove:
theorem tim_meditation_time (t_reading t_meditate_per_day : ℕ) 
  (h1 : time_reading_week t_reading)
  (h2 : twice_as_much_reading t_reading (7 * t_meditate_per_day)) :
  t_meditate_per_day = 1 :=
by
  sorry

end tim_meditation_time_l1671_167145


namespace woman_away_time_l1671_167173

noncomputable def angle_hour_hand (n : ℝ) : ℝ := 150 + n / 2
noncomputable def angle_minute_hand (n : ℝ) : ℝ := 6 * n

theorem woman_away_time : 
  (∀ n : ℝ, abs (angle_hour_hand n - angle_minute_hand n) = 120) → 
  abs ((540 / 11 : ℝ) - (60 / 11 : ℝ)) = 43.636 :=
by sorry

end woman_away_time_l1671_167173


namespace polynomial_coeff_sum_abs_l1671_167196

theorem polynomial_coeff_sum_abs (a a_1 a_2 a_3 a_4 a_5 : ℤ) :
    (2 * x - 1)^5 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
    |a_1| + |a_2| + |a_3| + |a_4| + |a_5| = 242 := by 
  sorry

end polynomial_coeff_sum_abs_l1671_167196


namespace segment_proportionality_l1671_167137

variable (a b c x : ℝ)

theorem segment_proportionality (ha : a ≠ 0) (hc : c ≠ 0) 
  (h : x = a * (b / c)) : 
  (x / a) = (b / c) := 
by
  sorry

end segment_proportionality_l1671_167137


namespace average_weight_of_removed_onions_l1671_167132

theorem average_weight_of_removed_onions (total_weight_40_onions : ℝ := 7680)
    (average_weight_35_onions : ℝ := 190)
    (number_of_onions_removed : ℕ := 5)
    (total_onions_initial : ℕ := 40)
    (total_number_of_remaining_onions : ℕ := 35) :
    (total_weight_40_onions - total_number_of_remaining_onions * average_weight_35_onions) / number_of_onions_removed = 206 :=
by
    sorry

end average_weight_of_removed_onions_l1671_167132


namespace max_candy_remainder_l1671_167179

theorem max_candy_remainder (x : ℕ) : x % 11 < 11 ∧ (∀ r : ℕ, r < 11 → x % 11 ≤ r) → x % 11 = 10 := 
sorry

end max_candy_remainder_l1671_167179


namespace complete_square_proof_l1671_167119

def complete_square (x : ℝ) : Prop :=
  x^2 - 2 * x - 8 = 0 -> (x - 1)^2 = 9

theorem complete_square_proof (x : ℝ) :
  complete_square x :=
sorry

end complete_square_proof_l1671_167119


namespace determine_moles_Al2O3_formed_l1671_167174

noncomputable def initial_moles_Al : ℝ := 10
noncomputable def initial_moles_Fe2O3 : ℝ := 6
noncomputable def balanced_eq (moles_Al moles_Fe2O3 moles_Al2O3 moles_Fe : ℝ) : Prop :=
  2 * moles_Al + moles_Fe2O3 = moles_Al2O3 + 2 * moles_Fe

theorem determine_moles_Al2O3_formed :
  ∃ moles_Al2O3 : ℝ, balanced_eq 10 6 moles_Al2O3 (moles_Al2O3 * 2) ∧ moles_Al2O3 = 5 := 
  by 
  sorry

end determine_moles_Al2O3_formed_l1671_167174


namespace John_lost_3_ebook_readers_l1671_167162

-- Definitions based on the conditions
def A : Nat := 50  -- Anna bought 50 eBook readers
def J : Nat := A - 15  -- John bought 15 less than Anna
def total : Nat := 82  -- Total eBook readers now

-- The number of eBook readers John has after the loss:
def J_after_loss : Nat := total - A

-- The number of eBook readers John lost:
def John_loss : Nat := J - J_after_loss

theorem John_lost_3_ebook_readers : John_loss = 3 :=
by
  sorry

end John_lost_3_ebook_readers_l1671_167162


namespace mortar_shell_hits_the_ground_at_50_seconds_l1671_167175

noncomputable def mortar_shell_firing_equation (x : ℝ) : ℝ :=
  - (1 / 5) * x^2 + 10 * x

theorem mortar_shell_hits_the_ground_at_50_seconds : 
  ∃ x : ℝ, mortar_shell_firing_equation x = 0 ∧ x = 50 :=
by
  sorry

end mortar_shell_hits_the_ground_at_50_seconds_l1671_167175


namespace matrix_pow_A_50_l1671_167138

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![5, 2], ![-16, -6]]

theorem matrix_pow_A_50 :
  A ^ 50 = ![![301, 100], ![-800, -249]] :=
by
  sorry

end matrix_pow_A_50_l1671_167138


namespace men_in_group_l1671_167172

theorem men_in_group (A : ℝ) (n : ℕ) (h : n > 0) 
  (inc_avg : ↑n * A + 2 * 32 - (21 + 23) = ↑n * (A + 1)) : n = 20 :=
sorry

end men_in_group_l1671_167172


namespace proportional_segments_l1671_167116

-- Define the problem
theorem proportional_segments :
  ∀ (a b c d : ℕ), a = 6 → b = 9 → c = 12 → (a * d = b * c) → d = 18 :=
by
  intros a b c d ha hb hc hrat
  rw [ha, hb, hc] at hrat
  exact sorry

end proportional_segments_l1671_167116


namespace real_roots_prime_equation_l1671_167129

noncomputable def has_rational_roots (p q : ℕ) : Prop :=
  ∃ x : ℚ, x^2 + p^2 * x + q^3 = 0

theorem real_roots_prime_equation (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  has_rational_roots p q ↔ (p = 3 ∧ q = 2) :=
sorry

end real_roots_prime_equation_l1671_167129


namespace max_value_of_expression_l1671_167155

theorem max_value_of_expression (x y z : ℤ) 
  (h1 : x * y + x + y = 20) 
  (h2 : y * z + y + z = 6) 
  (h3 : x * z + x + z = 2) : 
  x^2 + y^2 + z^2 ≤ 84 :=
sorry

end max_value_of_expression_l1671_167155


namespace circle_radius_three_points_on_line_l1671_167170

theorem circle_radius_three_points_on_line :
  ∀ R : ℝ,
  (∀ x y : ℝ, (x - 1)^2 + (y + 1)^2 = R^2 → (4 * x + 3 * y = 11) → (dist (x, y) (1, -1) = 1)) →
  R = 3
:= sorry

end circle_radius_three_points_on_line_l1671_167170


namespace y_value_on_line_l1671_167194

theorem y_value_on_line (x y : ℝ) (k : ℝ → ℝ)
  (h1 : k 0 = 0)
  (h2 : ∀ x, k x = (1/5) * x)
  (hx1 : k x = 1)
  (hx2 : k 5 = y) :
  y = 1 :=
sorry

end y_value_on_line_l1671_167194


namespace min_value_5_5_l1671_167118

noncomputable def given_expression (x y z : ℝ) : ℝ :=
  (6 * z) / (2 * x + y) + (6 * x) / (y + 2 * z) + (4 * y) / (x + z)

theorem min_value_5_5 (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x + y + z = 1) :
  given_expression x y z ≥ 5.5 :=
sorry

end min_value_5_5_l1671_167118


namespace range_j_l1671_167127

def h (x : ℝ) : ℝ := 2 * x + 3

def j (x : ℝ) : ℝ := h (h (h (h x)))

theorem range_j : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → 61 ≤ j x ∧ j x ≤ 93) := 
by 
  sorry

end range_j_l1671_167127


namespace number_of_elements_l1671_167131

theorem number_of_elements
  (init_avg : ℕ → ℝ)
  (correct_avg : ℕ → ℝ)
  (incorrect_num correct_num : ℝ)
  (h1 : ∀ n : ℕ, init_avg n = 17)
  (h2 : ∀ n : ℕ, correct_avg n = 20)
  (h3 : incorrect_num = 26)
  (h4 : correct_num = 56)
  : ∃ n : ℕ, n = 10 := sorry

end number_of_elements_l1671_167131


namespace intersection_M_N_l1671_167112

-- Define the set M based on the given condition
def M : Set ℝ := { x | x^2 > 1 }

-- Define the set N based on the given elements
def N : Set ℝ := { x | x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 }

-- Prove that the intersection of M and N is {-2, 2}
theorem intersection_M_N : M ∩ N = { -2, 2 } := by
  sorry

end intersection_M_N_l1671_167112


namespace question1_question2_l1671_167161

-- Define required symbols and parameters
variables {x : ℝ} {b c : ℝ}

-- Statement 1: Proving b + c given the conditions on the inequality
theorem question1 (h : ∀ x, -1 < x ∧ x < 3 → 5*x^2 - b*x + c < 0) : b + c = -25 := sorry

-- Statement 2: Proving the solution set for the given inequality
theorem question2 (h : ∀ x, (2 * x - 5) / (x + 4) ≥ 0 → (x ≥ 5 / 2 ∨ x < -4)) : 
  {x | (2 * x - 5) / (x + 4) ≥ 0} = {x | x ≥ 5/2 ∨ x < -4} := sorry

end question1_question2_l1671_167161


namespace correct_operation_B_l1671_167187

variable (a : ℝ)

theorem correct_operation_B :
  2 * a^2 * a^4 = 2 * a^6 :=
by sorry

end correct_operation_B_l1671_167187


namespace average_age_of_new_men_is_30_l1671_167125

noncomputable def average_age_of_two_new_men (A : ℝ) : ℝ :=
  let total_age_before : ℝ := 8 * A
  let total_age_after : ℝ := 8 * (A + 2)
  let age_of_replaced_men : ℝ := 21 + 23
  let total_age_of_new_men : ℝ := total_age_after - total_age_before + age_of_replaced_men
  total_age_of_new_men / 2

theorem average_age_of_new_men_is_30 (A : ℝ) : 
  average_age_of_two_new_men A = 30 :=
by 
  sorry

end average_age_of_new_men_is_30_l1671_167125


namespace circle_to_ellipse_scaling_l1671_167186

theorem circle_to_ellipse_scaling :
  ∀ (x' y' : ℝ), (4 * x')^2 + y'^2 = 16 → x'^2 / 16 + y'^2 / 4 = 1 :=
by
  intro x' y'
  intro h
  sorry

end circle_to_ellipse_scaling_l1671_167186


namespace length_of_hypotenuse_l1671_167105

theorem length_of_hypotenuse (a b : ℝ) (h1 : a = 15) (h2 : b = 21) : 
hypotenuse_length = Real.sqrt (a^2 + b^2) :=
by
  rw [h1, h2]
  sorry

end length_of_hypotenuse_l1671_167105


namespace royWeight_l1671_167199

-- Define the problem conditions
def johnWeight : ℕ := 81
def johnHeavierBy : ℕ := 77

-- Define the main proof problem
theorem royWeight : (johnWeight - johnHeavierBy) = 4 := by
  sorry

end royWeight_l1671_167199


namespace workers_are_280_women_l1671_167117

variables (W : ℕ) 
          (workers_without_retirement_plan : ℕ := W / 3)
          (women_without_retirement_plan : ℕ := (workers_without_retirement_plan * 1) / 10)
          (workers_with_retirement_plan : ℕ := W * 2 / 3)
          (men_with_retirement_plan : ℕ := (workers_with_retirement_plan * 4) / 10)
          (total_men : ℕ := (workers_without_retirement_plan * 9) / 30)
          (total_workers := total_men / (9 / 30))
          (number_of_women : ℕ := total_workers - 120)

theorem workers_are_280_women : total_workers = 400 ∧ number_of_women = 280 :=
by sorry

end workers_are_280_women_l1671_167117


namespace find_four_numbers_l1671_167141

theorem find_four_numbers (a b c d : ℕ) : 
  a + b + c + d = 45 ∧ (∃ k : ℕ, a + 2 = k ∧ b - 2 = k ∧ 2 * c = k ∧ d / 2 = k) → (a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20) :=
by
  sorry

end find_four_numbers_l1671_167141


namespace Matilda_correct_age_l1671_167139

def Louis_age : ℕ := 14
def Jerica_age : ℕ := 2 * Louis_age
def Matilda_age : ℕ := Jerica_age + 7

theorem Matilda_correct_age : Matilda_age = 35 :=
by
  -- Proof needs to be filled here
  sorry

end Matilda_correct_age_l1671_167139
