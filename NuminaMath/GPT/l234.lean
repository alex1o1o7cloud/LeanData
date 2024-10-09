import Mathlib

namespace sarah_total_distance_walked_l234_23434

noncomputable def total_distance : ℝ :=
  let rest_time : ℝ := 1 / 3
  let total_time : ℝ := 3.5
  let time_spent_walking : ℝ := total_time - rest_time -- time spent walking
  let uphill_speed : ℝ := 3 -- in mph
  let downhill_speed : ℝ := 4 -- in mph
  let d := time_spent_walking * (uphill_speed * downhill_speed) / (uphill_speed + downhill_speed) -- half distance D
  2 * d

theorem sarah_total_distance_walked :
  total_distance = 10.858 := sorry

end sarah_total_distance_walked_l234_23434


namespace find_S_l234_23410

variable {R k : ℝ}

theorem find_S (h : |k + R| / |R| = 0) : S = 1 :=
by
  let S := |k + 2*R| / |2*k + R|
  have h1 : k + R = 0 := by sorry
  have h2 : k = -R := by sorry
  sorry

end find_S_l234_23410


namespace find_function_satisfying_condition_l234_23476

theorem find_function_satisfying_condition :
  ∃ c : ℝ, ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (f x + 2 * y) = 6 * x + f (f y - x)) → 
                          (∀ x : ℝ, f x = 2 * x + c) :=
sorry

end find_function_satisfying_condition_l234_23476


namespace correct_transformation_l234_23425

theorem correct_transformation (a b m : ℝ) (h : m ≠ 0) : (am / bm) = (a / b) :=
by sorry

end correct_transformation_l234_23425


namespace hoseok_divides_number_l234_23418

theorem hoseok_divides_number (x : ℕ) (h : x / 6 = 11) : x = 66 := by
  sorry

end hoseok_divides_number_l234_23418


namespace fraction_identity_l234_23493

theorem fraction_identity (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 5) : (a + b) / c = 1 :=
by
  sorry

end fraction_identity_l234_23493


namespace diameter_of_large_circle_is_19_312_l234_23475

noncomputable def diameter_large_circle (r_small : ℝ) (n : ℕ) : ℝ :=
  let side_length_inner_octagon := 2 * r_small
  let radius_inner_octagon := side_length_inner_octagon / (2 * Real.sin (Real.pi / n)) / 2
  let radius_large_circle := radius_inner_octagon + r_small
  2 * radius_large_circle

theorem diameter_of_large_circle_is_19_312 :
  diameter_large_circle 4 8 = 19.312 :=
by
  sorry

end diameter_of_large_circle_is_19_312_l234_23475


namespace dice_probability_green_l234_23484

theorem dice_probability_green :
  let total_faces := 6
  let green_faces := 3
  let probability := green_faces / total_faces
  probability = 1 / 2 :=
by
  let total_faces := 6
  let green_faces := 3
  let probability := green_faces / total_faces
  have h : probability = 1 / 2 := by sorry
  exact h

end dice_probability_green_l234_23484


namespace smallest_clock_equiv_to_square_greater_than_10_l234_23465

def clock_equiv (h k : ℕ) : Prop :=
  (h % 12) = (k % 12)

theorem smallest_clock_equiv_to_square_greater_than_10 : ∃ h > 10, clock_equiv h (h * h) ∧ ∀ h' > 10, clock_equiv h' (h' * h') → h ≤ h' :=
by
  sorry

end smallest_clock_equiv_to_square_greater_than_10_l234_23465


namespace math_problem_solution_l234_23469

theorem math_problem_solution : 8 / 4 - 3 - 9 + 3 * 9 = 17 := 
by 
  sorry

end math_problem_solution_l234_23469


namespace average_of_seven_consecutive_l234_23414

variable (a : ℕ) 

def average_of_consecutive_integers (x : ℕ) : ℕ :=
  (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6)) / 7

theorem average_of_seven_consecutive (a : ℕ) :
  average_of_consecutive_integers (average_of_consecutive_integers a) = a + 6 :=
by
  sorry

end average_of_seven_consecutive_l234_23414


namespace sum_of_odd_base4_digits_of_152_and_345_l234_23446

def base_4_digit_count (n : ℕ) : ℕ :=
    n.digits 4 |>.filter (λ x => x % 2 = 1) |>.length

theorem sum_of_odd_base4_digits_of_152_and_345 :
    base_4_digit_count 152 + base_4_digit_count 345 = 6 :=
by
    sorry

end sum_of_odd_base4_digits_of_152_and_345_l234_23446


namespace find_n_l234_23419

theorem find_n (n : ℕ) (k : ℕ) (x : ℝ) (h1 : k = 1) (h2 : x = 180 - 360 / n) (h3 : 1.5 * x = 180 - 360 / (n + 1)) :
    n = 3 :=
by
  -- proof steps will be provided here
  sorry

end find_n_l234_23419


namespace simplify_equation_l234_23422

theorem simplify_equation (x : ℝ) : 
  (x / 0.3 = 1 + (1.2 - 0.3 * x) / 0.2) -> 
  (10 * x / 3 = 1 + (12 - 3 * x) / 2) :=
by 
  sorry

end simplify_equation_l234_23422


namespace least_number_to_subtract_l234_23423

theorem least_number_to_subtract :
  ∃ k : ℕ, k = 45 ∧ (568219 - k) % 89 = 0 :=
by
  sorry

end least_number_to_subtract_l234_23423


namespace value_of_expression_l234_23490

theorem value_of_expression (x : ℝ) (h : x = 1 + Real.sqrt 2) : x^4 - 4 * x^3 + 4 * x^2 + 4 = 5 := by
  sorry

end value_of_expression_l234_23490


namespace water_level_height_l234_23417

/-- Problem: An inverted frustum with a bottom diameter of 12 and height of 18, filled with water, 
    is emptied into another cylindrical container with a bottom diameter of 24. Assuming the 
    cylindrical container is sufficiently tall, the height of the water level in the cylindrical container -/
theorem water_level_height
  (V_cone : ℝ := (1 / 3) * π * (12 / 2) ^ 2 * 18)
  (R_cyl : ℝ := 24 / 2)
  (H_cyl : ℝ) :
  V_cone = π * R_cyl ^ 2 * H_cyl →
  H_cyl = 1.5 :=
by 
  sorry

end water_level_height_l234_23417


namespace delaney_left_home_at_7_50_l234_23411

theorem delaney_left_home_at_7_50 :
  (bus_time = 8 * 60 ∧ travel_time = 30 ∧ miss_time = 20) →
  (delaney_leave_time = bus_time + miss_time - travel_time) →
  delaney_leave_time = 7 * 60 + 50 :=
by
  intros
  sorry

end delaney_left_home_at_7_50_l234_23411


namespace remaining_area_correct_l234_23402

-- Define the side lengths of the large rectangle
def large_rectangle_length1 (x : ℝ) := 2 * x + 5
def large_rectangle_length2 (x : ℝ) := x + 8

-- Define the side lengths of the rectangular hole
def hole_length1 (x : ℝ) := 3 * x - 2
def hole_length2 (x : ℝ) := x + 1

-- Define the area of the large rectangle
def large_rectangle_area (x : ℝ) := (large_rectangle_length1 x) * (large_rectangle_length2 x)

-- Define the area of the hole
def hole_area (x : ℝ) := (hole_length1 x) * (hole_length2 x)

-- Prove the remaining area after accounting for the hole
theorem remaining_area_correct (x : ℝ) : 
  large_rectangle_area x - hole_area x = -x^2 + 20 * x + 42 := 
  by 
    sorry

end remaining_area_correct_l234_23402


namespace sqrt_of_4_l234_23450

theorem sqrt_of_4 : ∃ y : ℝ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  sorry

end sqrt_of_4_l234_23450


namespace salary_increase_l234_23494

theorem salary_increase (S P : ℝ) (h1 : 0.70 * S + P * (0.70 * S) = 0.91 * S) : P = 0.30 :=
by
  have eq1 : 0.70 * S * (1 + P) = 0.91 * S := by sorry
  have eq2 : S * (0.70 + 0.70 * P) = 0.91 * S := by sorry
  have eq3 : 0.70 + 0.70 * P = 0.91 := by sorry
  have eq4 : 0.70 * P = 0.21 := by sorry
  have eq5 : P = 0.21 / 0.70 := by sorry
  have eq6 : P = 0.30 := by sorry
  exact eq6

end salary_increase_l234_23494


namespace total_trip_cost_l234_23437

-- Definitions for the problem
def price_per_person : ℕ := 147
def discount : ℕ := 14
def number_of_people : ℕ := 2

-- Statement to prove
theorem total_trip_cost :
  (price_per_person - discount) * number_of_people = 266 :=
by
  sorry

end total_trip_cost_l234_23437


namespace fraction_identity_l234_23466

noncomputable def simplify_fraction (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : ℝ :=
  (1 / (2 * a * b)) + (b / (4 * a))

theorem fraction_identity (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  simplify_fraction a b h₁ h₂ = (2 + b^2) / (4 * a * b) :=
by sorry

end fraction_identity_l234_23466


namespace a_share_is_6300_l234_23496

noncomputable def investment_split (x : ℝ) :  ℝ × ℝ × ℝ :=
  let a_share := x * 12
  let b_share := 2 * x * 6
  let c_share := 3 * x * 4
  (a_share, b_share, c_share)

noncomputable def total_gain : ℝ := 18900

noncomputable def a_share_calculation : ℝ :=
  let (a_share, b_share, c_share) := investment_split 1
  total_gain / (a_share + b_share + c_share) * a_share

theorem a_share_is_6300 : a_share_calculation = 6300 := by
  -- Here, you would provide the proof, but for now we skip it.
  sorry

end a_share_is_6300_l234_23496


namespace find_passing_marks_l234_23498

-- Defining the conditions as Lean statements
def condition1 (T P : ℝ) : Prop := 0.30 * T = P - 50
def condition2 (T P : ℝ) : Prop := 0.45 * T = P + 25

-- The theorem to prove
theorem find_passing_marks (T P : ℝ) (h1 : condition1 T P) (h2 : condition2 T P) : P = 200 :=
by
  -- Placeholder proof
  sorry

end find_passing_marks_l234_23498


namespace daphne_two_visits_in_365_days_l234_23421

def visits_in_days (d1 d2 : ℕ) (days : ℕ) : ℕ :=
  days / Nat.lcm d1 d2

theorem daphne_two_visits_in_365_days :
  let days := 365
  let lcm_all := Nat.lcm 4 (Nat.lcm 6 (Nat.lcm 8 10))
  (visits_in_days 4 6 lcm_all + 
   visits_in_days 4 8 lcm_all + 
   visits_in_days 4 10 lcm_all + 
   visits_in_days 6 8 lcm_all + 
   visits_in_days 6 10 lcm_all + 
   visits_in_days 8 10 lcm_all) * 
   (days / lcm_all) = 129 :=
by
  sorry

end daphne_two_visits_in_365_days_l234_23421


namespace union_complement_eq_l234_23454

open Set

def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1}
def B : Set ℕ := {1, 2}

theorem union_complement_eq : A ∪ (U \ B) = {1, 3} := by
  sorry

end union_complement_eq_l234_23454


namespace minimal_inverse_presses_l234_23407

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem minimal_inverse_presses (x : ℚ) (h : x = 50) : 
  ∃ n, n = 2 ∧ (reciprocal^[n] x = x) :=
by
  sorry

end minimal_inverse_presses_l234_23407


namespace sum_of_roots_l234_23432

open Real

theorem sum_of_roots (x1 x2 k c : ℝ) (h1 : 4 * x1^2 - k * x1 = c) (h2 : 4 * x2^2 - k * x2 = c) (h3 : x1 ≠ x2) :
  x1 + x2 = k / 4 :=
by
  sorry

end sum_of_roots_l234_23432


namespace repeating_decimal_to_fraction_l234_23479

theorem repeating_decimal_to_fraction :
  (0.512341234123412341234 : ℝ) = (51229 / 99990 : ℝ) :=
sorry

end repeating_decimal_to_fraction_l234_23479


namespace common_ratio_of_geometric_sequence_l234_23471

variable (a : ℕ → ℝ) (d : ℝ)
variable (a1 : ℝ) (h_d : d ≠ 0)
variable (h_arith : ∀ n, a (n + 1) = a n + d)

theorem common_ratio_of_geometric_sequence :
  (a 0 = a1) →
  (a 4 = a1 + 4 * d) →
  (a 16 = a1 + 16 * d) →
  (a1 + 4 * d) / a1 = (a1 + 16 * d) / (a1 + 4 * d) →
  (a1 + 16 * d) / (a1 + 4 * d) = 3 :=
by
  sorry

end common_ratio_of_geometric_sequence_l234_23471


namespace range_of_a_l234_23431

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x < 1 then (3 - a) * x - a else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3 / 2 ≤ a ∧ a < 3) :=
by
  sorry

end range_of_a_l234_23431


namespace simplify_exponent_l234_23401

variable {x : ℝ} {m n : ℕ}

theorem simplify_exponent (x : ℝ) : (3 * x ^ 5) * (4 * x ^ 3) = 12 * x ^ 8 := by
  sorry

end simplify_exponent_l234_23401


namespace decagonal_pyramid_volume_l234_23461

noncomputable def volume_of_decagonal_pyramid (m : ℝ) (apex_angle : ℝ) : ℝ :=
  let sin18 := Real.sin (18 * Real.pi / 180)
  let sin36 := Real.sin (36 * Real.pi / 180)
  let cos18 := Real.cos (18 * Real.pi / 180)
  (5 * m^3 * sin36) / (3 * (1 + 2 * cos18))

theorem decagonal_pyramid_volume : volume_of_decagonal_pyramid 39 (18 * Real.pi / 180) = 20023 :=
  sorry

end decagonal_pyramid_volume_l234_23461


namespace alice_vs_bob_payment_multiple_l234_23467

theorem alice_vs_bob_payment_multiple :
  let alice_acorns := 3600
  let price_per_acorn := 15
  let bob_payment := 6000
  let total_alice_payment := alice_acorns * price_per_acorn
  total_alice_payment / bob_payment = 9 := by
  -- define the variables as per the conditions
  let alice_acorns := 3600
  let price_per_acorn := 15
  let bob_payment := 6000
  let total_alice_payment := alice_acorns * price_per_acorn
  -- define the target statement
  show total_alice_payment / bob_payment = 9
  sorry

end alice_vs_bob_payment_multiple_l234_23467


namespace spoons_in_set_l234_23457

def number_of_spoons_in_set (total_cost_set : ℕ) (cost_five_spoons : ℕ) : ℕ :=
  let c := cost_five_spoons / 5
  let s := total_cost_set / c
  s

theorem spoons_in_set (total_cost_set : ℕ) (cost_five_spoons : ℕ) (h1 : total_cost_set = 21) (h2 : cost_five_spoons = 15) : 
  number_of_spoons_in_set total_cost_set cost_five_spoons = 7 :=
by
  sorry

end spoons_in_set_l234_23457


namespace dot_product_a_b_equals_neg5_l234_23480

-- Defining vectors and conditions
structure vector2 := (x : ℝ) (y : ℝ)

def a : vector2 := ⟨2, 1⟩
def b (x : ℝ) : vector2 := ⟨x, -1⟩

-- Collinearity condition
def parallel (v w : vector2) : Prop :=
  v.x * w.y = v.y * w.x

-- Dot product definition
def dot_product (v w : vector2) : ℝ :=
  v.x * w.x + v.y * w.y

-- Given condition
theorem dot_product_a_b_equals_neg5 (x : ℝ) (h : parallel a ⟨a.x - x, a.y - (-1)⟩) : dot_product a (b x) = -5 :=
sorry

end dot_product_a_b_equals_neg5_l234_23480


namespace figure_100_squares_l234_23443

theorem figure_100_squares : (∃ f : ℕ → ℕ, f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 ∧ f 100 = 30301) :=
  sorry

end figure_100_squares_l234_23443


namespace probability_length_error_in_interval_l234_23440

noncomputable def normal_dist_prob (μ σ : ℝ) (a b : ℝ) : ℝ :=
∫ x in a..b, (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-((x - μ) ^ 2) / (2 * σ ^ 2))

theorem probability_length_error_in_interval :
  normal_dist_prob 0 3 3 6 = 0.1359 :=
by
  sorry

end probability_length_error_in_interval_l234_23440


namespace sheela_deposit_amount_l234_23429

theorem sheela_deposit_amount (monthly_income : ℕ) (deposit_percentage : ℕ) :
  monthly_income = 25000 → deposit_percentage = 20 → (deposit_percentage / 100 * monthly_income) = 5000 :=
  by
    intros h_income h_percentage
    rw [h_income, h_percentage]
    sorry

end sheela_deposit_amount_l234_23429


namespace sail_pressure_l234_23458

def pressure (k A V : ℝ) : ℝ := k * A * V^2

theorem sail_pressure (k : ℝ)
  (h_k : k = 1 / 800) 
  (A : ℝ) 
  (V : ℝ) 
  (P : ℝ)
  (h_initial : A = 1 ∧ V = 20 ∧ P = 0.5) 
  (A2 : ℝ) 
  (V2 : ℝ) 
  (h_doubled : A2 = 2 ∧ V2 = 30) :
  pressure k A2 V2 = 2.25 :=
by
  sorry

end sail_pressure_l234_23458


namespace q_joins_after_2_days_l234_23445

-- Define the conditions
def work_rate_p := 1 / 10
def work_rate_q := 1 / 6
def total_days := 5

-- Define the proof problem
theorem q_joins_after_2_days (a b : ℝ) (t x : ℕ) : 
  a = work_rate_p → b = work_rate_q → t = total_days →
  x * a + (t - x) * (a + b) = 1 → 
  x = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end q_joins_after_2_days_l234_23445


namespace valid_tickets_percentage_l234_23478

theorem valid_tickets_percentage (cars : ℕ) (people_without_payment : ℕ) (P : ℚ) 
  (h_cars : cars = 300) (h_people_without_payment : people_without_payment = 30) 
  (h_total_valid_or_passes : (cars - people_without_payment = 270)) :
  P + (P / 5) = 90 → P = 75 :=
by
  sorry

end valid_tickets_percentage_l234_23478


namespace units_digit_of_17_pow_2025_l234_23474

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_17_pow_2025 :
  units_digit (17 ^ 2025) = 7 :=
by sorry

end units_digit_of_17_pow_2025_l234_23474


namespace complement_U_A_union_B_is_1_and_9_l234_23416

-- Define the universe set U
def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define set A according to the given condition
def is_elem_of_A (x : ℕ) : Prop := 2 < x ∧ x ≤ 6
def A : Set ℕ := {x | is_elem_of_A x}

-- Define set B explicitly
def B : Set ℕ := {0, 2, 4, 5, 7, 8}

-- Define the union A ∪ B
def A_union_B : Set ℕ := A ∪ B

-- Define the complement of A ∪ B in U
def complement_U_A_union_B : Set ℕ := {x ∈ U | x ∉ A_union_B}

-- State the theorem
theorem complement_U_A_union_B_is_1_and_9 :
  complement_U_A_union_B = {1, 9} :=
by
  sorry

end complement_U_A_union_B_is_1_and_9_l234_23416


namespace matrix_determinant_transformation_l234_23442

theorem matrix_determinant_transformation (p q r s : ℝ) (h : p * s - q * r = -3) :
  (p * (5 * r + 4 * s) - r * (5 * p + 4 * q)) = -12 :=
sorry

end matrix_determinant_transformation_l234_23442


namespace Z_real_iff_m_eq_neg3_or_5_Z_pure_imaginary_iff_m_eq_neg2_Z_in_fourth_quadrant_iff_neg2_lt_m_lt_5_l234_23485

open Complex

noncomputable def Z (m : ℝ) : ℂ :=
  (m ^ 2 + 5 * m + 6) + (m ^ 2 - 2 * m - 15) * Complex.I

namespace ComplexNumbersProofs

-- Prove that Z is a real number if and only if m = -3 or m = 5
theorem Z_real_iff_m_eq_neg3_or_5 (m : ℝ) :
  (Z m).im = 0 ↔ (m = -3 ∨ m = 5) := 
by
  sorry

-- Prove that Z is a pure imaginary number if and only if m = -2
theorem Z_pure_imaginary_iff_m_eq_neg2 (m : ℝ) :
  (Z m).re = 0 ↔ (m = -2) := 
by
  sorry

-- Prove that the point corresponding to Z lies in the fourth quadrant if and only if -2 < m < 5
theorem Z_in_fourth_quadrant_iff_neg2_lt_m_lt_5 (m : ℝ) :
  (Z m).re > 0 ∧ (Z m).im < 0 ↔ (-2 < m ∧ m < 5) :=
by
  sorry

end ComplexNumbersProofs

end Z_real_iff_m_eq_neg3_or_5_Z_pure_imaginary_iff_m_eq_neg2_Z_in_fourth_quadrant_iff_neg2_lt_m_lt_5_l234_23485


namespace negate_proposition_l234_23483

theorem negate_proposition (x y : ℝ) :
  (¬ (x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔ (x^2 + y^2 ≠ 0 → ¬ (x = 0 ∧ y = 0)) :=
by
  sorry

end negate_proposition_l234_23483


namespace range_of_x_l234_23470

theorem range_of_x (x : ℝ) : x ≠ 3 ↔ ∃ y : ℝ, y = (x + 2) / (x - 3) :=
by {
  sorry
}

end range_of_x_l234_23470


namespace R2_perfect_fit_l234_23486

variables {n : ℕ} (x y : Fin n → ℝ) (b a : ℝ)

-- Condition: Observations \( (x_i, y_i) \) such that \( y_i = bx_i + a \)
def observations (i : Fin n) : Prop :=
  y i = b * x i + a

-- Condition: \( e_i = 0 \) for all \( i \)
def no_error (i : Fin n) : Prop := (b * x i + a + 0 = y i)

theorem R2_perfect_fit (h_obs: ∀ i, observations x y b a i)
                       (h_no_error: ∀ i, no_error x y b a i) : R_squared = 1 := by
  sorry

end R2_perfect_fit_l234_23486


namespace triangular_array_sum_digits_l234_23426

theorem triangular_array_sum_digits (N : ℕ) (h : N * (N + 1) / 2 = 2080) : 
  (N.digits 10).sum = 10 :=
sorry

end triangular_array_sum_digits_l234_23426


namespace find_larger_number_l234_23430

theorem find_larger_number 
  (L S : ℕ) 
  (h1 : L - S = 2342) 
  (h2 : L = 9 * S + 23) : 
  L = 2624 := 
sorry

end find_larger_number_l234_23430


namespace zoe_recycled_correctly_l234_23441

-- Let Z be the number of pounds recycled by Zoe
def pounds_by_zoe (total_points : ℕ) (friends_pounds : ℕ) (pounds_per_point : ℕ) : ℕ :=
  total_points * pounds_per_point - friends_pounds

-- Given conditions
def total_points : ℕ := 6
def friends_pounds : ℕ := 23
def pounds_per_point : ℕ := 8

-- Lean statement for the proof problem
theorem zoe_recycled_correctly : pounds_by_zoe total_points friends_pounds pounds_per_point = 25 :=
by
  -- proof to be provided here
  sorry

end zoe_recycled_correctly_l234_23441


namespace tennis_to_soccer_ratio_l234_23455

theorem tennis_to_soccer_ratio
  (total_balls : ℕ)
  (soccer_balls : ℕ)
  (basketball_offset : ℕ)
  (baseball_offset : ℕ)
  (volleyballs : ℕ)
  (tennis_balls : ℕ)
  (total_balls_eq : total_balls = 145)
  (soccer_balls_eq : soccer_balls = 20)
  (basketball_count : soccer_balls + basketball_offset = 20 + 5)
  (baseball_count : soccer_balls + baseball_offset = 20 + 10)
  (volleyballs_eq : volleyballs = 30)
  (accounted_balls : soccer_balls + (soccer_balls + basketball_offset) + (soccer_balls + baseball_offset) + volleyballs = 105)
  (tennis_balls_eq : tennis_balls = 145 - 105) :
  tennis_balls / soccer_balls = 2 :=
sorry

end tennis_to_soccer_ratio_l234_23455


namespace child_ticket_cost_is_2_l234_23409

-- Define the conditions
def adult_ticket_cost : ℕ := 5
def total_tickets_sold : ℕ := 85
def total_revenue : ℕ := 275
def adult_tickets_sold : ℕ := 35

-- Define the function to calculate child ticket cost
noncomputable def child_ticket_cost (adult_ticket_cost : ℕ) (total_tickets_sold : ℕ) (total_revenue : ℕ) (adult_tickets_sold : ℕ) : ℕ :=
  let total_adult_revenue := adult_tickets_sold * adult_ticket_cost
  let total_child_revenue := total_revenue - total_adult_revenue
  let child_tickets_sold := total_tickets_sold - adult_tickets_sold
  total_child_revenue / child_tickets_sold

theorem child_ticket_cost_is_2 : child_ticket_cost adult_ticket_cost total_tickets_sold total_revenue adult_tickets_sold = 2 := 
by
  -- This is a placeholder for the actual proof which we can fill in separately.
  sorry

end child_ticket_cost_is_2_l234_23409


namespace second_number_l234_23481

theorem second_number (x : ℕ) (h1 : ∃ k : ℕ, 1428 = 129 * k + 9)
  (h2 : ∃ m : ℕ, x = 129 * m + 13) (h_gcd : ∀ (d : ℕ), d ∣ (1428 - 9 : ℕ) ∧ d ∣ (x - 13 : ℕ) → d ≤ 129) :
  x = 1561 :=
by
  sorry

end second_number_l234_23481


namespace common_denominator_step1_error_in_step3_simplified_expression_l234_23427

theorem common_denominator_step1 (x : ℝ) (h1: x ≠ 2) (h2: x ≠ -2):
  (3 * x / (x - 2) - x / (x + 2)) = (3 * x * (x + 2)) / ((x - 2) * (x + 2)) - (x * (x - 2)) / ((x - 2) * (x + 2)) :=
sorry

theorem error_in_step3 (x : ℝ) (h1: x ≠ 2) (h2: x ≠ -2) :
  (3 * x^2 + 6 * x - (x^2 - 2 * x)) / ((x - 2) * (x + 2)) ≠ (3 * x^2 + 6 * x * (x^2 - 2 * x)) / ((x - 2) * (x + 2)) :=
sorry

theorem simplified_expression (x : ℝ) (h1: x ≠ 0) (h2: x ≠ 2) (h3: x ≠ -2) :
  ((3 * x / (x - 2) - x / (x + 2)) * (x^2 - 4) / x) = 2 * x + 8 :=
sorry

end common_denominator_step1_error_in_step3_simplified_expression_l234_23427


namespace car_travel_distance_l234_23415

theorem car_travel_distance:
  (∃ r, r = 3 / 4 ∧ ∀ t, t = 2 → ((r * 60) * t = 90)) :=
by
  sorry

end car_travel_distance_l234_23415


namespace expected_value_eight_l234_23459

-- Define the 10-sided die roll outcomes
def outcomes := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the value function for a roll outcome
def value (x : ℕ) : ℕ :=
  if x % 2 = 0 then x  -- even value
  else 2 * x  -- odd value

-- Calculate the expected value
def expected_value : ℚ :=
  (1 / 10 : ℚ) * (2 + 2 + 6 + 4 + 10 + 6 + 14 + 8 + 18 + 10)

-- The theorem stating the expected value equals 8
theorem expected_value_eight :
  expected_value = 8 := by
  sorry

end expected_value_eight_l234_23459


namespace smallest_possible_value_of_c_l234_23460

theorem smallest_possible_value_of_c (b c : ℝ) (h1 : 1 < b) (h2 : b < c)
    (h3 : ¬∃ (u v w : ℝ), u = 1 ∧ v = b ∧ w = c ∧ u + v > w ∧ u + w > v ∧ v + w > u)
    (h4 : ¬∃ (x y z : ℝ), x = 1 ∧ y = 1/b ∧ z = 1/c ∧ x + y > z ∧ x + z > y ∧ y + z > x) :
    c = (5 + Real.sqrt 5) / 2 :=
by
  sorry

end smallest_possible_value_of_c_l234_23460


namespace common_rational_root_l234_23463

theorem common_rational_root (a b c d e f g : ℚ) (p : ℚ) :
  (48 * p^4 + a * p^3 + b * p^2 + c * p + 16 = 0) ∧
  (16 * p^5 + d * p^4 + e * p^3 + f * p^2 + g * p + 48 = 0) ∧
  (∃ m n : ℤ, p = m / n ∧ Int.gcd m n = 1 ∧ n ≠ 1 ∧ p < 0 ∧ n > 0) →
  p = -1/2 :=
by
  sorry

end common_rational_root_l234_23463


namespace evaluate_expression_l234_23464

theorem evaluate_expression : 
  101^3 + 3 * (101^2) * 2 + 3 * 101 * (2^2) + 2^3 = 1092727 := 
by 
  sorry

end evaluate_expression_l234_23464


namespace part_a_part_b_l234_23449

-- Given distinct primes p and q
variables (p q : ℕ) [hp : Fact (Nat.Prime p)] [hq : Fact (Nat.Prime q)] (h : p ≠ q)

-- Prove p^q + q^p ≡ p + q (mod pq)
theorem part_a (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] (h : p ≠ q) :
  (p^q + q^p) % (p * q) = (p + q) % (p * q) := by
  sorry

-- Given distinct primes p and q, and neither are 2
theorem part_b (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] (h : p ≠ q) (hp2 : p ≠ 2) (hq2 : q ≠ 2) :
  Even (Nat.floor ((p^q + q^p) / (p * q))) := by
  sorry

end part_a_part_b_l234_23449


namespace least_subtraction_divisible_by13_l234_23492

theorem least_subtraction_divisible_by13 (n : ℕ) (h : n = 427398) : ∃ k : ℕ, k = 2 ∧ (n - k) % 13 = 0 := by
  sorry

end least_subtraction_divisible_by13_l234_23492


namespace find_a_b_range_of_a_l234_23424

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * a * x + 2

-- Problem 1
theorem find_a_b (a b : ℝ) :
  f a 1 = 0 ∧ f a b = 0 ∧ (∀ x, f a x > 0 ↔ x < 1 ∨ x > b) → a = 1 ∧ b = 2 := sorry

-- Problem 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x > 0) → (0 ≤ a ∧ a < 8/9) := sorry

end find_a_b_range_of_a_l234_23424


namespace number_of_friends_l234_23489

-- Definitions based on conditions
def total_bill_divided_among_all (n : ℕ) : ℕ := 12 * (n + 2)
def total_bill_divided_among_friends (n : ℕ) : ℕ := 16 * n

-- The theorem to prove
theorem number_of_friends (n : ℕ) : total_bill_divided_among_all n = total_bill_divided_among_friends n → n = 6 :=
by
  sorry

end number_of_friends_l234_23489


namespace infinite_B_l234_23420

open Set Function

variable (A B : Type) 

theorem infinite_B (hA_inf : Infinite A) (f : A → B) : Infinite B :=
by
  sorry

end infinite_B_l234_23420


namespace tan_of_angle_l234_23499

theorem tan_of_angle (α : ℝ) (h₁ : α ∈ Set.Ioo (Real.pi / 2) Real.pi) (h₂ : Real.sin α = 3 / 5) : 
  Real.tan α = -3 / 4 := 
sorry

end tan_of_angle_l234_23499


namespace surface_area_of_segmented_part_l234_23436

theorem surface_area_of_segmented_part (h_prism : ∀ (base_height prism_height : ℝ), base_height = 9 ∧ prism_height = 20)
  (isosceles_triangle : ∀ (a b c : ℝ), a = 18 ∧ b = 15 ∧ c = 15 ∧ b = c)
  (midpoints : ∀ (X Y Z : ℝ), X = 9 ∧ Y = 10 ∧ Z = 9) 
  : let triangle_CZX_area := 45
    let triangle_CZY_area := 45
    let triangle_CXY_area := 9
    let triangle_XYZ_area := 9
    (triangle_CZX_area + triangle_CZY_area + triangle_CXY_area + triangle_XYZ_area = 108) :=
sorry

end surface_area_of_segmented_part_l234_23436


namespace triangle_angles_l234_23412

-- Define the problem and the conditions as Lean statements.
theorem triangle_angles (x y z : ℝ) 
  (h1 : y + 150 + 160 = 360)
  (h2 : z + 150 + 160 = 360)
  (h3 : x + y + z = 180) : 
  x = 80 ∧ y = 50 ∧ z = 50 := 
by 
  sorry

end triangle_angles_l234_23412


namespace find_m_l234_23433

theorem find_m (S : ℕ → ℝ) (a : ℕ → ℝ) (m : ℝ) (hS : ∀ n, S n = m * 2^(n-1) - 3) 
               (ha1 : a 1 = S 1) (han : ∀ n > 1, a n = S n - S (n - 1)) 
               (ratio : ∀ n > 1, a (n+1) / a n = 1/2): 
  m = 6 := 
sorry

end find_m_l234_23433


namespace carl_weight_l234_23408

variable (C R B : ℕ)

theorem carl_weight (h1 : B = R + 9) (h2 : R = C + 5) (h3 : B = 159) : C = 145 :=
by
  sorry

end carl_weight_l234_23408


namespace find_unknown_rate_of_blankets_l234_23462

theorem find_unknown_rate_of_blankets (x : ℕ) 
  (h1 : 3 * 100 = 300) 
  (h2 : 5 * 150 = 750)
  (h3 : 3 + 5 + 2 = 10) 
  (h4 : 10 * 160 = 1600) 
  (h5 : 300 + 750 + 2 * x = 1600) : 
  x = 275 := 
sorry

end find_unknown_rate_of_blankets_l234_23462


namespace max_working_groups_l234_23482

theorem max_working_groups (teachers groups : ℕ) (memberships_per_teacher group_size : ℕ) 
  (h_teachers : teachers = 36) (h_memberships_per_teacher : memberships_per_teacher = 2)
  (h_group_size : group_size = 4) 
  (h_max_memberships : teachers * memberships_per_teacher = 72) :
  groups ≤ 18 :=
by
  sorry

end max_working_groups_l234_23482


namespace vasya_result_correct_l234_23447

def num : ℕ := 10^1990 + (10^1989 * 6 - 1)
def denom : ℕ := 10 * (10^1989 * 6 - 1) + 4

theorem vasya_result_correct : (num / denom) = (1 / 4) := 
  sorry

end vasya_result_correct_l234_23447


namespace probability_of_stock_price_increase_l234_23453

namespace StockPriceProbability

variables (P_A P_B P_C P_D_given_A P_D_given_B P_D_given_C : ℝ)

def P_D : ℝ := P_A * P_D_given_A + P_B * P_D_given_B + P_C * P_D_given_C

theorem probability_of_stock_price_increase :
    P_A = 0.6 → P_B = 0.3 → P_C = 0.1 → 
    P_D_given_A = 0.7 → P_D_given_B = 0.2 → P_D_given_C = 0.1 → 
    P_D P_A P_B P_C P_D_given_A P_D_given_B P_D_given_C = 0.49 :=
by intros h₁ h₂ h₃ h₄ h₅ h₆; sorry

end StockPriceProbability

end probability_of_stock_price_increase_l234_23453


namespace fraction_subtraction_l234_23400

theorem fraction_subtraction : (1 / 6 : ℚ) - (5 / 12) = -1 / 4 := 
by sorry

end fraction_subtraction_l234_23400


namespace find_tan_α_l234_23428

variable (α : ℝ) (h1 : Real.sin (α - Real.pi / 3) = 3 / 5)
variable (h2 : Real.pi / 4 < α ∧ α < Real.pi / 2)

theorem find_tan_α (h1 : Real.sin (α - Real.pi / 3) = 3 / 5) (h2 : Real.pi / 4 < α ∧ α < Real.pi / 2) : 
  Real.tan α = - (48 + 25 * Real.sqrt 3) / 11 :=
sorry

end find_tan_α_l234_23428


namespace Ryan_dig_time_alone_l234_23472

theorem Ryan_dig_time_alone :
  ∃ R : ℝ, ∀ Castel_time together_time,
    Castel_time = 6 ∧ together_time = 30 / 11 →
    (1 / R + 1 / Castel_time = 11 / 30) →
    R = 5 :=
by 
  sorry

end Ryan_dig_time_alone_l234_23472


namespace second_smallest_five_digit_in_pascal_l234_23473

theorem second_smallest_five_digit_in_pascal :
  ∃ (x : ℕ), (x > 10000) ∧ (∀ y : ℕ, (y ≠ 10000) → (y < x) → (y < 10000)) ∧ (x = 10001) :=
sorry

end second_smallest_five_digit_in_pascal_l234_23473


namespace compare_f_values_l234_23405

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem compare_f_values (a : ℝ) (h_pos : 0 < a) :
  (a > 2 * Real.sqrt 2 → f a > f (a / 2) * f (a / 2)) ∧
  (a = 2 * Real.sqrt 2 → f a = f (a / 2) * f (a / 2)) ∧
  (0 < a ∧ a < 2 * Real.sqrt 2 → f a < f (a / 2) * f (a / 2)) :=
by
  sorry

end compare_f_values_l234_23405


namespace point_B_coordinates_l234_23403

/-
Problem Statement:
Given a point A(2, 4) which is symmetric to point B with respect to the origin,
we need to prove the coordinates of point B.
-/

structure Point where
  x : ℝ
  y : ℝ

def symmetric_wrt_origin (A B : Point) : Prop :=
  B.x = -A.x ∧ B.y = -A.y

noncomputable def point_A : Point := ⟨2, 4⟩
noncomputable def point_B : Point := ⟨-2, -4⟩

theorem point_B_coordinates : symmetric_wrt_origin point_A point_B :=
  by
    -- Proof is omitted
    sorry

end point_B_coordinates_l234_23403


namespace no_possible_seating_arrangement_l234_23413

theorem no_possible_seating_arrangement : 
  ¬(∃ (students : Fin 11 → Fin 4),
    ∀ (i : Fin 11),
    ∃ (s1 s2 s3 s4 s5 : Fin 11),
      s1 = i ∧ 
      (s2 = (i + 1) % 11) ∧ 
      (s3 = (i + 2) % 11) ∧ 
      (s4 = (i + 3) % 11) ∧ 
      (s5 = (i + 4) % 11) ∧
      ∃ (g1 g2 g3 g4 : Fin 4),
        (students s1 = g1) ∧ 
        (students s2 = g2) ∧ 
        (students s3 = g3) ∧ 
        (students s4 = g4) ∧ 
        (students s5).val ≠ (students s1).val ∧ 
        (students s5).val ≠ (students s2).val ∧ 
        (students s5).val ≠ (students s3).val ∧ 
        (students s5).val ≠ (students s4).val) :=
sorry

end no_possible_seating_arrangement_l234_23413


namespace equilateral_triangle_distances_l234_23444

-- Defining the necessary conditions
variables {h x y z : ℝ}
variables (hx : 0 < h) (hx_cond : x + y + z = h)
variables (triangle_ineqs : x + y > z ∧ y + z > x ∧ z + x > y)

-- Lean 4 statement to express the proof problem
theorem equilateral_triangle_distances (hx : 0 < h) (hx_cond : x + y + z = h) (triangle_ineqs : x + y > z ∧ y + z > x ∧ z + x > y) : 
  x < h / 2 ∧ y < h / 2 ∧ z < h / 2 :=
sorry

end equilateral_triangle_distances_l234_23444


namespace triangle_proportion_l234_23438

theorem triangle_proportion (p q r x y : ℝ)
  (h1 : x / q = y / r)
  (h2 : x + y = p) :
  y / r = p / (q + r) := sorry

end triangle_proportion_l234_23438


namespace area_of_triangle_l234_23439

theorem area_of_triangle (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (angle : ℝ) (h_side_ratio : side2 / side3 = 8 / 5)
  (h_side_opposite : side1 = 14)
  (h_angle_opposite : angle = 60) :
  (1/2 * side2 * side3 * Real.sin (angle * Real.pi / 180)) = 40 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l234_23439


namespace hotel_room_count_l234_23435

theorem hotel_room_count {total_lamps lamps_per_room : ℕ} (h_total_lamps : total_lamps = 147) (h_lamps_per_room : lamps_per_room = 7) : total_lamps / lamps_per_room = 21 := by
  -- We will insert this placeholder auto-proof, as the actual arithmetic proof isn't the focus.
  sorry

end hotel_room_count_l234_23435


namespace boundary_of_shadow_of_sphere_l234_23406

theorem boundary_of_shadow_of_sphere (x y : ℝ) :
  let O := (0, 0, 2)
  let P := (1, -2, 3)
  let r := 2
  (∃ T : ℝ × ℝ × ℝ,
    T = (0, -2, 2) ∧
    (∃ g : ℝ → ℝ,
      y = g x ∧
      g x = (x^2 - 2 * x - 11) / 6)) → 
  y = (x^2 - 2 * x - 11) / 6 :=
by
  sorry

end boundary_of_shadow_of_sphere_l234_23406


namespace smallest_solution_of_quartic_equation_l234_23491

theorem smallest_solution_of_quartic_equation :
  ∃ x : ℝ, (x^4 - 50 * x^2 + 625 = 0) ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := sorry

end smallest_solution_of_quartic_equation_l234_23491


namespace parallelogram_angle_bisector_l234_23404

theorem parallelogram_angle_bisector (a b S Q : ℝ) (α : ℝ) 
  (hS : S = a * b * Real.sin α)
  (hQ : Q = (1 / 2) * (a - b) ^ 2 * Real.sin α) :
  (2 * a * b) / (a - b) ^ 2 = (S + Q + Real.sqrt (Q ^ 2 + 2 * Q * S)) / S :=
by
  sorry

end parallelogram_angle_bisector_l234_23404


namespace fuel_consumption_l234_23452

open Real

theorem fuel_consumption (initial_fuel : ℝ) (final_fuel : ℝ) (distance_covered : ℝ) (consumption_rate : ℝ) (fuel_left : ℝ) (x : ℝ) :
  initial_fuel = 60 ∧ final_fuel = 50 ∧ distance_covered = 100 ∧ 
  consumption_rate = (initial_fuel - final_fuel) / distance_covered ∧ consumption_rate = 0.1 ∧ 
  fuel_left = initial_fuel - consumption_rate * x ∧ x = 260 →
  fuel_left = 34 :=
by
  sorry

end fuel_consumption_l234_23452


namespace transformation_invariant_l234_23477

-- Define the initial and transformed parabolas
def initial_parabola (x : ℝ) : ℝ := 2 * x^2
def transformed_parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 3

-- Define the transformation process
def move_right_1 (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - 1)
def move_up_3 (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 3

-- Concatenate transformations to form the final transformation
def combined_transformation (x : ℝ) : ℝ :=
  move_up_3 (move_right_1 initial_parabola) x

-- Statement to prove
theorem transformation_invariant :
  ∀ x : ℝ, combined_transformation x = transformed_parabola x := 
by {
  sorry
}

end transformation_invariant_l234_23477


namespace average_price_of_fruit_l234_23497

theorem average_price_of_fruit 
  (price_apple price_orange : ℝ)
  (total_fruits initial_fruits kept_oranges kept_fruits : ℕ)
  (average_price_kept average_price_initial : ℝ)
  (h1 : price_apple = 40)
  (h2 : price_orange = 60)
  (h3 : initial_fruits = 10)
  (h4 : kept_oranges = initial_fruits - 6)
  (h5 : average_price_kept = 50) :
  average_price_initial = 56 := 
sorry

end average_price_of_fruit_l234_23497


namespace candies_bought_is_18_l234_23456

-- Define the original number of candies
def original_candies : ℕ := 9

-- Define the total number of candies after buying more
def total_candies : ℕ := 27

-- Define the function to calculate the number of candies bought
def candies_bought (o t : ℕ) : ℕ := t - o

-- The main theorem stating that the number of candies bought is 18
theorem candies_bought_is_18 : candies_bought original_candies total_candies = 18 := by
  -- This is where the proof would go
  sorry

end candies_bought_is_18_l234_23456


namespace M_gt_N_l234_23448

-- Define the variables and conditions
variables (x y : ℝ)
noncomputable def M := x^2 + y^2
noncomputable def N := 2*x + 6*y - 11

-- State the theorem
theorem M_gt_N : M x y > N x y := by
  sorry -- Placeholder for the proof

end M_gt_N_l234_23448


namespace determine_ABC_l234_23495

theorem determine_ABC : 
  ∀ (A B C : ℝ), 
    A = 2 * B - 3 * C ∧ 
    B = 2 * C - 5 ∧ 
    A + B + C = 100 → 
    A = 18.75 ∧ B = 52.5 ∧ C = 28.75 :=
by
  intro A B C h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end determine_ABC_l234_23495


namespace probability_interval_l234_23488

noncomputable def Phi : ℝ → ℝ := sorry -- assuming Φ is a given function for CDF of a standard normal distribution

theorem probability_interval (h : Phi 1.98 = 0.9762) : 
  2 * Phi 1.98 - 1 = 0.9524 :=
by
  sorry

end probability_interval_l234_23488


namespace bottles_stolen_at_dance_l234_23487

-- Define the initial conditions
def initial_bottles := 10
def bottles_lost_at_school := 2
def total_stickers := 21
def stickers_per_bottle := 3

-- Calculate remaining bottles after loss at school
def remaining_bottles_after_school := initial_bottles - bottles_lost_at_school

-- Calculate the remaining bottles after the theft
def remaining_bottles_after_theft := total_stickers / stickers_per_bottle

-- Prove the number of bottles stolen
theorem bottles_stolen_at_dance : remaining_bottles_after_school - remaining_bottles_after_theft = 1 :=
by
  sorry

end bottles_stolen_at_dance_l234_23487


namespace train_average_speed_l234_23451

-- Define the variables used in the conditions
variables (D V : ℝ)
-- Condition: Distance D in 50 minutes at average speed V kmph
-- 50 minutes to hours conversion
def condition1 : D = V * (50 / 60) := sorry
-- Condition: Distance D in 40 minutes at speed 60 kmph
-- 40 minutes to hours conversion
def condition2 : D = 60 * (40 / 60) := sorry

-- Claim: Current average speed V
theorem train_average_speed : V = 48 :=
by
  -- Using the conditions to prove the claim
  sorry

end train_average_speed_l234_23451


namespace rational_sum_abs_ratios_l234_23468

theorem rational_sum_abs_ratios (a b c : ℚ) (h : |a * b * c| / (a * b * c) = 1) : (|a| / a + |b| / b + |c| / c = 3) ∨ (|a| / a + |b| / b + |c| / c = -1) := 
sorry

end rational_sum_abs_ratios_l234_23468
