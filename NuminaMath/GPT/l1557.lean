import Mathlib

namespace farmer_turkeys_l1557_155791

variable (n c : ℝ)

theorem farmer_turkeys (h1 : n * c = 60) (h2 : (c + 0.10) * (n - 15) = 54) : n = 75 :=
sorry

end farmer_turkeys_l1557_155791


namespace households_both_brands_l1557_155732

theorem households_both_brands
  (T : ℕ) (N : ℕ) (A : ℕ) (B : ℕ)
  (hT : T = 300) (hN : N = 80) (hA : A = 60) (hB : ∃ X : ℕ, B = 3 * X ∧ T = N + A + B + X) :
  ∃ X : ℕ, X = 40 :=
by
  -- Upon extracting values from conditions, solving for both brand users X = 40
  sorry

end households_both_brands_l1557_155732


namespace depth_of_channel_l1557_155706

theorem depth_of_channel (a b A : ℝ) (h : ℝ) (h_area : A = (1 / 2) * (a + b) * h)
  (ha : a = 12) (hb : b = 6) (hA : A = 630) : h = 70 :=
by
  sorry

end depth_of_channel_l1557_155706


namespace golf_tees_per_member_l1557_155739

theorem golf_tees_per_member (T : ℕ) : 
  (∃ (t : ℕ), 
     t = 4 * T ∧ 
     (∀ (g : ℕ), g ≤ 2 → g * 12 + 28 * 2 = t)
  ) → T = 20 :=
by
  intros h
  -- problem statement is enough for this example
  sorry

end golf_tees_per_member_l1557_155739


namespace park_shape_l1557_155707

def cost_of_fencing (side_count : ℕ) (side_cost : ℕ) := side_count * side_cost

theorem park_shape (total_cost : ℕ) (side_cost : ℕ) (h_total : total_cost = 224) (h_side : side_cost = 56) : 
  (∃ sides : ℕ, sides = total_cost / side_cost ∧ sides = 4) ∧ (∀ (sides : ℕ),  cost_of_fencing sides side_cost = total_cost → sides = 4 → sides = 4 ∧ (∀ (x y z w : ℕ), x = y → y = z → z = w → w = x)) :=
by
  sorry

end park_shape_l1557_155707


namespace problem_l1557_155722

variable (f g h : ℕ → ℕ)

-- Define the conditions as hypotheses
axiom h1 : ∀ (n m : ℕ), n ≠ m → h n ≠ h m
axiom h2 : ∀ y, ∃ x, g x = y
axiom h3 : ∀ n, f n = g n - h n + 1

theorem problem : ∀ n, f n = 1 := 
by 
  sorry

end problem_l1557_155722


namespace real_value_of_b_l1557_155765

open Real

theorem real_value_of_b : ∃ x : ℝ, (x^2 - 2 * x + 1 = 0) ∧ (x^2 + x - 2 = 0) :=
by
  sorry

end real_value_of_b_l1557_155765


namespace find_divisor_l1557_155728

theorem find_divisor 
  (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h_dividend : dividend = 190) (h_quotient : quotient = 9) (h_remainder : remainder = 1) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 21 := 
by
  sorry

end find_divisor_l1557_155728


namespace average_charge_proof_l1557_155744

noncomputable def averageChargePerPerson
  (chargeFirstDay : ℝ)
  (chargeSecondDay : ℝ)
  (chargeThirdDay : ℝ)
  (chargeFourthDay : ℝ)
  (ratioFirstDay : ℝ)
  (ratioSecondDay : ℝ)
  (ratioThirdDay : ℝ)
  (ratioFourthDay : ℝ)
  : ℝ :=
  let totalRevenue := ratioFirstDay * chargeFirstDay + ratioSecondDay * chargeSecondDay + ratioThirdDay * chargeThirdDay + ratioFourthDay * chargeFourthDay
  let totalVisitors := ratioFirstDay + ratioSecondDay + ratioThirdDay + ratioFourthDay
  totalRevenue / totalVisitors

theorem average_charge_proof :
  averageChargePerPerson 25 15 7.5 2.5 3 7 11 19 = 7.75 := by
  simp [averageChargePerPerson]
  sorry

end average_charge_proof_l1557_155744


namespace cost_of_items_l1557_155790

theorem cost_of_items (e t b : ℝ) 
    (h1 : 3 * e + 4 * t = 3.20)
    (h2 : 4 * e + 3 * t = 3.50)
    (h3 : 5 * e + 5 * t + 2 * b = 5.70) :
    4 * e + 4 * t + 3 * b = 5.20 :=
by
  sorry

end cost_of_items_l1557_155790


namespace solution_set_inequality_l1557_155740

open Real

theorem solution_set_inequality (f : ℝ → ℝ) (h1 : f e = 0) (h2 : ∀ x > 0, x * deriv f x < 2) :
    ∀ x, 0 < x → x ≤ e → f x + 2 ≥ 2 * log x :=
by
  sorry

end solution_set_inequality_l1557_155740


namespace range_of_k_l1557_155725

theorem range_of_k (k : ℝ) :
  (∃ (x : ℝ), 2 < x ∧ x < 3 ∧ x^2 + (1 - k) * x - 2 * (k + 1) = 0) →
  1 < k ∧ k < 2 :=
by
  sorry

end range_of_k_l1557_155725


namespace weighted_mean_is_correct_l1557_155701

-- Define the given values
def dollar_from_aunt : ℝ := 9
def euros_from_uncle : ℝ := 9
def dollar_from_sister : ℝ := 7
def dollar_from_friends_1 : ℝ := 22
def dollar_from_friends_2 : ℝ := 23
def euros_from_friends_3 : ℝ := 18
def pounds_from_friends_4 : ℝ := 15
def dollar_from_friends_5 : ℝ := 22

-- Define the exchange rates
def exchange_rate_euro_to_usd : ℝ := 1.20
def exchange_rate_pound_to_usd : ℝ := 1.38

-- Calculate the amounts in USD
def dollar_from_uncle : ℝ := euros_from_uncle * exchange_rate_euro_to_usd
def dollar_from_friends_3_converted : ℝ := euros_from_friends_3 * exchange_rate_euro_to_usd
def dollar_from_friends_4_converted : ℝ := pounds_from_friends_4 * exchange_rate_pound_to_usd

-- Define total amounts from family and friends in USD
def family_total : ℝ := dollar_from_aunt + dollar_from_uncle + dollar_from_sister
def friends_total : ℝ := dollar_from_friends_1 + dollar_from_friends_2 + dollar_from_friends_3_converted + dollar_from_friends_4_converted + dollar_from_friends_5

-- Define weights
def family_weight : ℝ := 0.40
def friends_weight : ℝ := 0.60

-- Calculate the weighted mean
def weighted_mean : ℝ := (family_total * family_weight) + (friends_total * friends_weight)

theorem weighted_mean_is_correct : weighted_mean = 76.30 := by
  sorry

end weighted_mean_is_correct_l1557_155701


namespace remainder_when_divided_by_7_l1557_155704

theorem remainder_when_divided_by_7 (n : ℕ) (h1 : n % 2 = 1) (h2 : ∃ p : ℕ, p > 0 ∧ (n + p) % 10 = 0 ∧ p = 5) : n % 7 = 5 :=
by
  sorry

end remainder_when_divided_by_7_l1557_155704


namespace enrollment_difference_l1557_155752

theorem enrollment_difference :
  let Varsity := 1680
  let Northwest := 1170
  let Central := 1840
  let Greenbriar := 1090
  let Eastside := 1450
  Central - Greenbriar = 750 := 
by
  intros Varsity Northwest Central Greenbriar Eastside
  -- calculate the difference
  have h1 : 750 = 750 := rfl
  sorry

end enrollment_difference_l1557_155752


namespace Cheryl_total_distance_l1557_155733

theorem Cheryl_total_distance :
  let speed := 2
  let duration := 3
  let distance_away := speed * duration
  let distance_home := distance_away
  let total_distance := distance_away + distance_home
  total_distance = 12 := by
  sorry

end Cheryl_total_distance_l1557_155733


namespace solution_set_unique_line_l1557_155742

theorem solution_set_unique_line (x y : ℝ) : 
  (x - 2 * y = 1 ∧ x^3 - 6 * x * y - 8 * y^3 = 1) ↔ (y = (x - 1) / 2) := 
by
  sorry

end solution_set_unique_line_l1557_155742


namespace seventh_root_binomial_expansion_l1557_155708

theorem seventh_root_binomial_expansion : 
  (∃ (n : ℕ), n = 137858491849 ∧ (∃ (k : ℕ), n = (10 + 1) ^ k)) →
  (∃ a, a = 11 ∧ 11 ^ 7 = 137858491849) := 
by {
  sorry 
}

end seventh_root_binomial_expansion_l1557_155708


namespace general_solution_of_differential_eq_l1557_155785

theorem general_solution_of_differential_eq (x y : ℝ) (C : ℝ) :
  (x^2 - y^2) * (y * (1 - C^2)) - 2 * (y * x) * (x) = 0 → (x^2 + y^2 = C * y) := by
  sorry

end general_solution_of_differential_eq_l1557_155785


namespace parabola_axis_symmetry_value_p_l1557_155794

theorem parabola_axis_symmetry_value_p (p : ℝ) (h_parabola : ∀ y x, y^2 = 2 * p * x) (h_axis_symmetry : ∀ (a: ℝ), a = -1 → a = -p / 2) : p = 2 :=
by 
  sorry

end parabola_axis_symmetry_value_p_l1557_155794


namespace max_value_xy_xz_yz_l1557_155770

theorem max_value_xy_xz_yz (x y z : ℝ) (h : x + 2 * y + z = 6) :
  xy + xz + yz ≤ 6 :=
sorry

end max_value_xy_xz_yz_l1557_155770


namespace factorize_9_minus_a_squared_l1557_155746

theorem factorize_9_minus_a_squared (a : ℤ) : 9 - a^2 = (3 + a) * (3 - a) :=
by
  sorry

end factorize_9_minus_a_squared_l1557_155746


namespace michael_matchstick_houses_l1557_155780

theorem michael_matchstick_houses :
  ∃ n : ℕ, n = (600 / 2) / 10 ∧ n = 30 := 
sorry

end michael_matchstick_houses_l1557_155780


namespace parallelogram_perimeter_eq_60_l1557_155709

-- Given conditions from the problem
variables (P Q R M N O : Type*)
variables (PQ PR QR PM MN NO PO : ℝ)
variables {PQ_eq_PR : PQ = PR}
variables {PQ_val : PQ = 30}
variables {PR_val : PR = 30}
variables {QR_val : QR = 28}
variables {MN_parallel_PR : true}  -- Parallel condition we can treat as true for simplification
variables {NO_parallel_PQ : true}  -- Another parallel condition treated as true

-- Statement of the problem to be proved
theorem parallelogram_perimeter_eq_60 :
  PM + MN + NO + PO = 60 :=
sorry

end parallelogram_perimeter_eq_60_l1557_155709


namespace mod_inverse_35_36_l1557_155750

theorem mod_inverse_35_36 : ∃ a : ℤ, 0 ≤ a ∧ a < 36 ∧ (35 * a) % 36 = 1 :=
  ⟨35, by sorry⟩

end mod_inverse_35_36_l1557_155750


namespace factor_sum_l1557_155727

theorem factor_sum :
  ∃ d e f : ℤ, (∀ x : ℤ, x^2 + 11 * x + 24 = (x + d) * (x + e)) ∧
              (∀ x : ℤ, x^2 + 9 * x - 36 = (x + e) * (x - f)) ∧
              d + e + f = 14 := by
  sorry

end factor_sum_l1557_155727


namespace range_of_3a_minus_b_l1557_155751

theorem range_of_3a_minus_b (a b : ℝ) (h1 : 2 ≤ a + b ∧ a + b ≤ 5) (h2 : -2 ≤ a - b ∧ a - b ≤ 1) : 
    -2 ≤ 3 * a - b ∧ 3 * a - b ≤ 7 := 
by 
  sorry

end range_of_3a_minus_b_l1557_155751


namespace hexagon_triangle_count_l1557_155760

-- Definitions based on problem conditions
def numPoints : ℕ := 7
def totalTriangles := Nat.choose numPoints 3
def collinearCases : ℕ := 3

-- Proof problem
theorem hexagon_triangle_count : totalTriangles - collinearCases = 32 :=
by
  -- Calculation is expected here
  sorry

end hexagon_triangle_count_l1557_155760


namespace proof_problem_l1557_155793

def p := 8 + 7 = 16
def q := Real.pi > 3

theorem proof_problem :
  (¬p ∧ q) ∧ ((p ∨ q) = true) ∧ ((p ∧ q) = false) ∧ ((¬p) = true) := sorry

end proof_problem_l1557_155793


namespace arithmetic_sequence_terms_l1557_155700

theorem arithmetic_sequence_terms (a : ℕ → ℝ) (n : ℕ) 
  (h1 : a 3 + a 4 = 10) 
  (h2 : a (n - 3) + a (n - 2) = 30) 
  (h3 : (n * (a 1 + a n)) / 2 = 100) : 
  n = 10 :=
sorry

end arithmetic_sequence_terms_l1557_155700


namespace robot_cost_max_units_A_l1557_155702

noncomputable def cost_price_A (x : ℕ) := 1600
noncomputable def cost_price_B (x : ℕ) := 2800

theorem robot_cost (x : ℕ) (y : ℕ) (a : ℕ) (b : ℕ) :
  y = 2 * x - 400 →
  a = 96000 →
  b = 168000 →
  a / x = 6000 →
  b / y = 6000 →
  (x = 1600 ∧ y = 2800) :=
by sorry

theorem max_units_A (m n total_units : ℕ) : 
  total_units = 100 →
  m + n = 100 →
  m ≤ 2 * n →
  m ≤ 66 :=
by sorry

end robot_cost_max_units_A_l1557_155702


namespace factor_polynomial_l1557_155772

theorem factor_polynomial (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) = 
  (x - y) * (y - z) * (z - x) * (x * y + x * z + y * z) := by
  sorry

end factor_polynomial_l1557_155772


namespace quiz_probability_l1557_155719

theorem quiz_probability :
  let probMCQ := 1/3
  let probTF1 := 1/2
  let probTF2 := 1/2
  probMCQ * probTF1 * probTF2 = 1/12 := by
  sorry

end quiz_probability_l1557_155719


namespace shoes_multiple_l1557_155718

-- Define the number of shoes each has
variables (J E B : ℕ)

-- Conditions
axiom h1 : B = 22
axiom h2 : J = E / 2
axiom h3 : J + E + B = 121

-- Prove the multiple of E to B is 3
theorem shoes_multiple : E / B = 3 :=
by
  -- Inject the provisional proof
  sorry

end shoes_multiple_l1557_155718


namespace price_of_cookie_cookie_price_verification_l1557_155723

theorem price_of_cookie 
  (total_spent : ℝ) 
  (cost_per_cupcake : ℝ)
  (num_cupcakes : ℕ)
  (cost_per_doughnut : ℝ)
  (num_doughnuts : ℕ)
  (cost_per_pie_slice : ℝ)
  (num_pie_slices : ℕ)
  (num_cookies : ℕ)
  (total_cookies_cost : ℝ)
  (total_cost : ℝ) :
  (num_cupcakes * cost_per_cupcake + num_doughnuts * cost_per_doughnut + num_pie_slices * cost_per_pie_slice 
  + num_cookies * total_cookies_cost = total_spent) → 
  total_cookies_cost = 0.60 :=
by
  sorry

noncomputable def sophie_cookies_price : ℝ := 
  let total_cost := 33
  let num_cupcakes := 5
  let cost_per_cupcake := 2
  let num_doughnuts := 6
  let cost_per_doughnut := 1
  let num_pie_slices := 4
  let cost_per_pie_slice := 2
  let num_cookies := 15
  let total_spent_on_other_items := 
    num_cupcakes * cost_per_cupcake + num_doughnuts * cost_per_doughnut + num_pie_slices * cost_per_pie_slice 
  let remaining_cost := total_cost - total_spent_on_other_items 
  remaining_cost / num_cookies

theorem cookie_price_verification :
  sophie_cookies_price = 0.60 :=
by
  sorry

end price_of_cookie_cookie_price_verification_l1557_155723


namespace bicycle_speed_l1557_155745

theorem bicycle_speed
  (dist : ℝ := 15) -- Distance between the school and the museum
  (bus_factor : ℝ := 1.5) -- Bus speed is 1.5 times the bicycle speed
  (time_diff : ℝ := 1 / 4) -- Bicycle students leave 1/4 hour earlier
  (x : ℝ) -- Speed of bicycles
  (h : (dist / x) - (dist / (bus_factor * x)) = time_diff) :
  x = 20 :=
sorry

end bicycle_speed_l1557_155745


namespace Dans_placed_scissors_l1557_155767

theorem Dans_placed_scissors (initial_scissors placed_scissors total_scissors : ℕ) 
  (h1 : initial_scissors = 39) 
  (h2 : total_scissors = initial_scissors + placed_scissors) 
  (h3 : total_scissors = 52) : placed_scissors = 13 := 
by 
  sorry

end Dans_placed_scissors_l1557_155767


namespace product_of_two_primes_l1557_155796

theorem product_of_two_primes (p q z : ℕ) (hp_prime : Nat.Prime p) (hq_prime : Nat.Prime q) 
    (h_p_range : 2 < p ∧ p < 6) 
    (h_q_range : 8 < q ∧ q < 24) 
    (h_z_def : z = p * q) 
    (h_z_range : 15 < z ∧ z < 36) : 
    z = 33 := 
by 
    sorry

end product_of_two_primes_l1557_155796


namespace exists_set_X_gcd_condition_l1557_155755

theorem exists_set_X_gcd_condition :
  ∃ (X : Finset ℕ), X.card = 2022 ∧
  (∀ (a b c : ℕ) (n : ℕ) (ha : a ∈ X) (hb : b ∈ X) (hc : c ∈ X) (hn_pos : 0 < n)
    (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c),
  Nat.gcd (a^n + b^n) c = 1) :=
sorry

end exists_set_X_gcd_condition_l1557_155755


namespace p_q_r_inequality_l1557_155735

theorem p_q_r_inequality (p q r : ℝ) (h₁ : ∀ x, (x < -6 ∨ (3 ≤ x ∧ x ≤ 8)) ↔ (x - p) * (x - q) ≤ 0) (h₂ : p < q) : p + 2 * q + 3 * r = 1 :=
by
  sorry

end p_q_r_inequality_l1557_155735


namespace remainder_of_n_div_7_l1557_155749

theorem remainder_of_n_div_7 (n : ℕ) (h1 : n^2 % 7 = 3) (h2 : n^3 % 7 = 6) : n % 7 = 5 :=
sorry

end remainder_of_n_div_7_l1557_155749


namespace greatest_prime_factor_341_l1557_155757

theorem greatest_prime_factor_341 : ∃ p : ℕ, Nat.Prime p ∧ p = 17 ∧ p = Nat.gcd 341 (Nat.gcd 341 (Nat.gcd 341 341)) :=
by
  sorry

end greatest_prime_factor_341_l1557_155757


namespace triangle_sides_inequality_l1557_155774

-- Define the sides of a triangle and their sum
variables {a b c : ℝ}

-- Define the condition that they are sides of a triangle.
def triangle_sides (a b c : ℝ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the condition that their sum is 1
axiom sum_of_sides (a b c : ℝ) (h : triangle_sides a b c) : a + b + c = 1

-- Define the proof theorem for the inequality
theorem triangle_sides_inequality (h : triangle_sides a b c) (h_sum : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
sorry

end triangle_sides_inequality_l1557_155774


namespace tan_product_identity_l1557_155766

-- Lean statement for the mathematical problem
theorem tan_product_identity : 
  (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 6)) = 2 := by
  sorry

end tan_product_identity_l1557_155766


namespace company_pays_per_month_l1557_155775

theorem company_pays_per_month
  (length width height : ℝ)
  (total_volume : ℝ)
  (cost_per_box : ℝ)
  (h1 : length = 15)
  (h2 : width = 12)
  (h3 : height = 10)
  (h4 : total_volume = 1.08 * 10^6)
  (h5 : cost_per_box = 0.6) :
  (total_volume / (length * width * height) * cost_per_box) = 360 :=
by
  -- sorry to skip proof
  sorry

end company_pays_per_month_l1557_155775


namespace factorize_polynomial_l1557_155756

theorem factorize_polynomial (x y : ℝ) : x * y^2 - 16 * x = x * (y + 4) * (y - 4) := 
by
  sorry

end factorize_polynomial_l1557_155756


namespace sandy_correct_sums_l1557_155743

theorem sandy_correct_sums :
  ∃ x y : ℕ, x + y = 30 ∧ 3 * x - 2 * y = 60 ∧ x = 24 :=
by
  sorry

end sandy_correct_sums_l1557_155743


namespace parabola_focus_l1557_155761

theorem parabola_focus (F : ℝ × ℝ) :
  (∀ (x y : ℝ), y^2 = 4 * x → (x + 1)^2 + y^2 = ((x - F.1)^2 + (y - F.2)^2)) → 
  F = (1, 0) :=
sorry

end parabola_focus_l1557_155761


namespace total_marbles_correct_l1557_155778

variable (r : ℝ) -- number of red marbles
variable (b : ℝ) -- number of blue marbles
variable (g : ℝ) -- number of green marbles

-- Conditions
def red_blue_ratio : Prop := r = 1.5 * b
def green_red_ratio : Prop := g = 1.8 * r

-- Total number of marbles
def total_marbles (r b g : ℝ) : ℝ := r + b + g

theorem total_marbles_correct (r b g : ℝ) (h1 : red_blue_ratio r b) (h2 : green_red_ratio r g) : 
  total_marbles r b g = 3.467 * r :=
by 
  sorry

end total_marbles_correct_l1557_155778


namespace value_of_xy_l1557_155788

noncomputable def distinct_nonzero_reals (x y : ℝ) : Prop :=
x ≠ 0 ∧ y ≠ 0 ∧ x ≠ y

theorem value_of_xy (x y : ℝ) (h : distinct_nonzero_reals x y) (h_eq : x + 4 / x = y + 4 / y) :
  x * y = 4 :=
sorry

end value_of_xy_l1557_155788


namespace snooker_tournament_total_cost_l1557_155713

def VIP_cost : ℝ := 45
def GA_cost : ℝ := 20
def total_tickets_sold : ℝ := 320
def vip_and_general_admission_relationship := 276

def total_cost_of_tickets : ℝ := 6950

theorem snooker_tournament_total_cost 
  (V G : ℝ)
  (h1 : VIP_cost * V + GA_cost * G = total_cost_of_tickets)
  (h2 : V + G = total_tickets_sold)
  (h3 : V = G - vip_and_general_admission_relationship) : 
  VIP_cost * V + GA_cost * G = total_cost_of_tickets := 
by {
  sorry
}

end snooker_tournament_total_cost_l1557_155713


namespace quadratic_vertex_property_l1557_155734

variable {a b c x0 y0 m n : ℝ}

-- Condition 1: (x0, y0) is a fixed point on the graph of the quadratic function y = ax^2 + bx + c
axiom fixed_point_on_graph : y0 = a * x0^2 + b * x0 + c

-- Condition 2: (m, n) is a moving point on the graph of the quadratic function
axiom moving_point_on_graph : n = a * m^2 + b * m + c

-- Condition 3: For any real number m, a(y0 - n) ≤ 0
axiom inequality_condition : ∀ m : ℝ, a * (y0 - (a * m^2 + b * m + c)) ≤ 0

-- Statement to prove
theorem quadratic_vertex_property : 2 * a * x0 + b = 0 := 
sorry

end quadratic_vertex_property_l1557_155734


namespace maximum_marks_l1557_155773

theorem maximum_marks (M : ℝ)
  (pass_threshold_percentage : ℝ := 33)
  (marks_obtained : ℝ := 92)
  (marks_failed_by : ℝ := 40) :
  (marks_obtained + marks_failed_by) = (pass_threshold_percentage / 100) * M → M = 400 := by
  sorry

end maximum_marks_l1557_155773


namespace polygon_sides_eq_seven_l1557_155747

-- Given conditions:
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180
def sum_exterior_angles : ℝ := 360
def difference_in_angles (n : ℕ) : ℝ := sum_interior_angles n - sum_exterior_angles

-- Proof statement:
theorem polygon_sides_eq_seven (n : ℕ) (h : difference_in_angles n = 540) : n = 7 := sorry

end polygon_sides_eq_seven_l1557_155747


namespace birdseed_mixture_l1557_155795

theorem birdseed_mixture (x : ℝ) (h1 : 0.40 * x + 0.65 * (100 - x) = 50) : x = 60 :=
by
  sorry

end birdseed_mixture_l1557_155795


namespace function_classification_l1557_155777

theorem function_classification (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 + f y) = f (f x) + f (y^2) + 2 * f (x * y)) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) := by
  sorry

end function_classification_l1557_155777


namespace maximum_value_of_n_l1557_155717

noncomputable def max_n (a b c : ℝ) (n : ℕ) :=
  a > b ∧ b > c ∧ (∀ (n : ℕ), (1 / (a - b) + 1 / (b - c) ≥ n^2 / (a - c))) → n ≤ 2

theorem maximum_value_of_n (a b c : ℝ) (n : ℕ) : 
  a > b → b > c → (∀ (n : ℕ), (1 / (a - b) + 1 / (b - c) ≥ n^2 / (a - c))) → n ≤ 2 :=
  by sorry

end maximum_value_of_n_l1557_155717


namespace not_possible_127_points_l1557_155716

theorem not_possible_127_points (n_correct n_unanswered n_incorrect : ℕ) :
  n_correct + n_unanswered + n_incorrect = 25 →
  127 ≠ 5 * n_correct + 2 * n_unanswered - n_incorrect :=
by
  intro h_total
  sorry

end not_possible_127_points_l1557_155716


namespace sqrt_factorial_sq_l1557_155781

theorem sqrt_factorial_sq : ((Real.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2) = 2880 := by
  sorry

end sqrt_factorial_sq_l1557_155781


namespace lambda_range_l1557_155787

noncomputable def lambda (S1 S2 S3 S4: ℝ) (S: ℝ) : ℝ :=
  4 * (S1 + S2 + S3 + S4) / S

theorem lambda_range (S1 S2 S3 S4: ℝ) (S: ℝ) (h_max: S = max (max S1 S2) (max S3 S4)) :
  2 < lambda S1 S2 S3 S4 S ∧ lambda S1 S2 S3 S4 S ≤ 4 :=
by
  sorry

end lambda_range_l1557_155787


namespace largest_number_of_square_plots_l1557_155731

/-- A rectangular field measures 30 meters by 60 meters with 2268 meters of internal fencing to partition into congruent, square plots. The entire field must be partitioned with sides of squares parallel to the edges. Prove the largest number of square plots is 722. -/
theorem largest_number_of_square_plots (s n : ℕ) (h_length : 60 = n * s) (h_width : 30 = s * 2 * n) (h_fence : 120 * n - 90 ≤ 2268) :
(s * 2 * n) = 722 :=
sorry

end largest_number_of_square_plots_l1557_155731


namespace hexagon_probability_same_length_l1557_155764

-- Define the problem
theorem hexagon_probability_same_length (T : Finset (Fin 15)) :
  let num_favorable_outcomes := 33
  let total_possible_outcomes := 105
  (num_favorable_outcomes / total_possible_outcomes) = (11 / 35) :=
by
  sorry

end hexagon_probability_same_length_l1557_155764


namespace cubic_eq_one_complex_solution_l1557_155714

theorem cubic_eq_one_complex_solution (k : ℂ) :
  (∃ (x : ℂ), 8 * x^3 + 12 * x^2 + k * x + 1 = 0) ∧
  (∀ (x y z : ℂ), 8 * x^3 + 12 * x^2 + k * x + 1 = 0 → 8 * y^3 + 12 * y^2 + k * y + 1 = 0
    → 8 * z^3 + 12 * z^2 + k * z + 1 = 0 → x = y ∧ y = z) →
  k = 6 :=
sorry

end cubic_eq_one_complex_solution_l1557_155714


namespace car_grid_probability_l1557_155784

theorem car_grid_probability:
  let m := 11
  let n := 48
  100 * m + n = 1148 := by
  sorry

end car_grid_probability_l1557_155784


namespace natasha_average_speed_climbing_l1557_155710

theorem natasha_average_speed_climbing :
  ∀ D : ℝ,
    (total_time = 3 + 2) →
    (total_distance = 2 * D) →
    (average_speed = total_distance / total_time) →
    (average_speed = 3) →
    (D = 7.5) →
    (climb_speed = D / 3) →
    (climb_speed = 2.5) :=
by
  intros D total_time_eq total_distance_eq average_speed_eq average_speed_is_3 D_is_7_5 climb_speed_eq
  sorry

end natasha_average_speed_climbing_l1557_155710


namespace prime_in_A_l1557_155779

def is_in_A (n : ℕ) : Prop :=
  ∃ a b : ℤ, n = a^2 + 2 * b^2 ∧ b ≠ 0

theorem prime_in_A (p : ℕ) (hp : Nat.Prime p) (h : is_in_A (p^2)) : is_in_A p :=
by
  sorry

end prime_in_A_l1557_155779


namespace smallest_solution_x4_50x2_576_eq_0_l1557_155797

theorem smallest_solution_x4_50x2_576_eq_0 :
  ∃ x : ℝ, (x^4 - 50*x^2 + 576 = 0) ∧ ∀ y : ℝ, y^4 - 50*y^2 + 576 = 0 → x ≤ y :=
sorry

end smallest_solution_x4_50x2_576_eq_0_l1557_155797


namespace problem_l1557_155782

theorem problem (a : ℤ) (n : ℕ) : (a + 1) ^ (2 * n + 1) + a ^ (n + 2) ∣ a ^ 2 + a + 1 :=
sorry

end problem_l1557_155782


namespace hcf_of_numbers_l1557_155758

theorem hcf_of_numbers (x y : ℕ) (hcf lcm : ℕ) 
    (h_sum : x + y = 45) 
    (h_lcm : lcm = 100)
    (h_reciprocal_sum : 1 / (x : ℝ) + 1 / (y : ℝ) = 0.3433333333333333) :
    hcf = 1 :=
by
  sorry

end hcf_of_numbers_l1557_155758


namespace solve_inequality_for_a_l1557_155726

theorem solve_inequality_for_a (a : ℝ) :
  (∀ x : ℝ, abs (x^2 + 3 * a * x + 4 * a) ≤ 3 → x = -3 * a / 2)
  ↔ (a = 8 + 2 * Real.sqrt 13 ∨ a = 8 - 2 * Real.sqrt 13) :=
by 
  sorry

end solve_inequality_for_a_l1557_155726


namespace temperature_decrease_l1557_155712

theorem temperature_decrease (T : ℝ) 
    (h1 : T * (3 / 4) = T - 21)
    (h2 : T > 0) : 
    T = 84 := 
  sorry

end temperature_decrease_l1557_155712


namespace inequality_abc_l1557_155759

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a * (1 + b))) + (1 / (b * (1 + c))) + (1 / (c * (1 + a))) ≥ 3 / (1 + a * b * c) :=
by 
  sorry

end inequality_abc_l1557_155759


namespace algebraic_expression_value_l1557_155738

variable (x y : ℝ)

def condition1 : Prop := y - x = -1
def condition2 : Prop := x * y = 2

def expression : ℝ := -2 * x^3 * y + 4 * x^2 * y^2 - 2 * x * y^3

theorem algebraic_expression_value (h1 : condition1 x y) (h2 : condition2 x y) : expression x y = -4 := 
by
  sorry

end algebraic_expression_value_l1557_155738


namespace cost_of_bread_l1557_155748

-- Definition of the conditions
def total_purchase_amount : ℕ := 205  -- in cents
def amount_given_to_cashier : ℕ := 700  -- in cents
def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def num_nickels_received : ℕ := 8

-- Statement of the problem
theorem cost_of_bread :
  (∃ (B C : ℕ), B + C = total_purchase_amount ∧
                  amount_given_to_cashier - total_purchase_amount = 
                  (quarter_value + dime_value + num_nickels_received * nickel_value + 420) ∧
                  B = 125) :=
by
  -- Skipping the proof
  sorry

end cost_of_bread_l1557_155748


namespace simplify_expression_l1557_155754

def operation (a b : ℚ) : ℚ := 2 * a - b

theorem simplify_expression (x y : ℚ) : 
  operation (operation (x - y) (x + y)) (-3 * y) = 2 * x - 3 * y :=
by
  sorry

end simplify_expression_l1557_155754


namespace alcohol_to_water_ratio_l1557_155737

theorem alcohol_to_water_ratio (p q r : ℝ) :
  let alcohol := (p / (p + 1) + q / (q + 1) + r / (r + 1))
  let water := (1 / (p + 1) + 1 / (q + 1) + 1 / (r + 1))
  (alcohol / water) = (p * q * r + p * q + p * r + q * r + p + q + r) / (p * q + p * r + q * r + p + q + r + 1) :=
sorry

end alcohol_to_water_ratio_l1557_155737


namespace hyperbola_distance_to_left_focus_l1557_155715

theorem hyperbola_distance_to_left_focus (P : ℝ × ℝ)
  (h1 : (P.1^2) / 9 - (P.2^2) / 16 = 1)
  (dPF2 : dist P (4, 0) = 4) : dist P (-4, 0) = 10 := 
sorry

end hyperbola_distance_to_left_focus_l1557_155715


namespace large_painting_area_l1557_155776

theorem large_painting_area :
  ∃ (large_painting : ℕ),
  (3 * (6 * 6) + 4 * (2 * 3) + large_painting = 282) → large_painting = 150 := by
  sorry

end large_painting_area_l1557_155776


namespace vertical_throw_time_l1557_155792

theorem vertical_throw_time (h v g t : ℝ)
  (h_def: h = v * t - (1/2) * g * t^2)
  (initial_v: v = 25)
  (gravity: g = 10)
  (target_h: h = 20) :
  t = 1 ∨ t = 4 := 
by
  sorry

end vertical_throw_time_l1557_155792


namespace evaluate_infinite_series_l1557_155768

noncomputable def infinite_series (n : ℕ) : ℝ := (n^2) / (3^n)

theorem evaluate_infinite_series :
  (∑' k : ℕ, infinite_series (k+1)) = 4.5 :=
by sorry

end evaluate_infinite_series_l1557_155768


namespace first_digit_base12_1025_l1557_155741

theorem first_digit_base12_1025 : (1025 : ℕ) / (12^2 : ℕ) = 7 := by
  sorry

end first_digit_base12_1025_l1557_155741


namespace cylinder_not_occupied_volume_l1557_155721

theorem cylinder_not_occupied_volume :
  let r := 10
  let h_cylinder := 30
  let h_full_cone := 10
  let volume_cylinder := π * r^2 * h_cylinder
  let volume_full_cone := (1 / 3) * π * r^2 * h_full_cone
  let volume_half_cone := (1 / 2) * volume_full_cone
  let volume_unoccupied := volume_cylinder - (volume_full_cone + volume_half_cone)
  volume_unoccupied = 2500 * π := 
by
  sorry

end cylinder_not_occupied_volume_l1557_155721


namespace total_watermelons_l1557_155771

def watermelons_grown_by_jason : ℕ := 37
def watermelons_grown_by_sandy : ℕ := 11

theorem total_watermelons : watermelons_grown_by_jason + watermelons_grown_by_sandy = 48 := by
  sorry

end total_watermelons_l1557_155771


namespace expected_value_boy_girl_adjacent_pairs_l1557_155783

/-- Considering 10 boys and 15 girls lined up in a row, we need to show that
    the expected number of adjacent positions where a boy and a girl stand next to each other is 12. -/
theorem expected_value_boy_girl_adjacent_pairs :
  let boys := 10
  let girls := 15
  let total_people := boys + girls
  let total_adjacent_pairs := total_people - 1
  let p_boy_then_girl := (boys / total_people) * (girls / (total_people - 1))
  let p_girl_then_boy := (girls / total_people) * (boys / (total_people - 1))
  let expected_T := total_adjacent_pairs * (p_boy_then_girl + p_girl_then_boy)
  expected_T = 12 :=
by
  sorry

end expected_value_boy_girl_adjacent_pairs_l1557_155783


namespace not_valid_mapping_circle_triangle_l1557_155711

inductive Point
| mk : ℝ → ℝ → Point

inductive Circle
| mk : ℝ → ℝ → ℝ → Circle

inductive Triangle
| mk : Point → Point → Point → Triangle

open Point (mk)
open Circle (mk)
open Triangle (mk)

def valid_mapping (A B : Type) (f : A → B) := ∀ a₁ a₂ : A, f a₁ = f a₂ → a₁ = a₂

def inscribed_triangle_mapping (c : Circle) : Triangle := sorry -- map a circle to one of its inscribed triangles

theorem not_valid_mapping_circle_triangle :
  ¬ valid_mapping Circle Triangle inscribed_triangle_mapping :=
sorry

end not_valid_mapping_circle_triangle_l1557_155711


namespace find_f_of_2_l1557_155786

theorem find_f_of_2 
  (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x - 1/x) = x^2 + 1/x^2) : f 2 = 6 :=
sorry

end find_f_of_2_l1557_155786


namespace determine_m_l1557_155799

open Set Real

theorem determine_m (m : ℝ) : (∀ x, x ∈ { x | x ≥ 3 } ∪ { x | x < m }) ∧ (∀ x, x ∉ { x | x ≥ 3 } ∩ { x | x < m }) → m = 3 :=
by
  intros h
  sorry

end determine_m_l1557_155799


namespace candy_division_l1557_155789

theorem candy_division (total_candy num_students : ℕ) (h1 : total_candy = 344) (h2 : num_students = 43) : total_candy / num_students = 8 := by
  sorry

end candy_division_l1557_155789


namespace slope_of_line_l1557_155729

theorem slope_of_line 
  (p l t : ℝ) (p_pos : p > 0)
  (h_parabola : (2:ℝ)*p = 4) -- Since the parabola passes through M(l,2)
  (h_incircle_center : ∃ (k m : ℝ), (k + 1 = 0) ∧ (k^2 - k - 2 = 0)) :
  ∃ (k : ℝ), k = -1 :=
by {
  sorry
}

end slope_of_line_l1557_155729


namespace exchange_silver_cards_l1557_155762

theorem exchange_silver_cards : 
  (∃ red gold silver : ℕ,
    (∀ (r g s : ℕ), ((2 * g = 5 * r) ∧ (g = r + s) ∧ (r = 3) ∧ (g = 3) → s = 7))) :=
by
  sorry

end exchange_silver_cards_l1557_155762


namespace Pythagorean_triple_example_1_Pythagorean_triple_example_2_l1557_155724

theorem Pythagorean_triple_example_1 : 3^2 + 4^2 = 5^2 := by
  sorry

theorem Pythagorean_triple_example_2 : 5^2 + 12^2 = 13^2 := by
  sorry

end Pythagorean_triple_example_1_Pythagorean_triple_example_2_l1557_155724


namespace triangle_shortest_side_l1557_155769

theorem triangle_shortest_side (x y z : ℝ) (h : x / y = 1 / 2) (h1 : x / z = 1 / 3) (hyp : x = 6) : z = 3 :=
sorry

end triangle_shortest_side_l1557_155769


namespace bookseller_loss_l1557_155753

theorem bookseller_loss (C S : ℝ) (h : 20 * C = 25 * S) : (C - S) / C * 100 = 20 := by
  sorry

end bookseller_loss_l1557_155753


namespace series_sum_l1557_155730

theorem series_sum :
  let a_1 := 2
  let d := 3
  let s := [2, -5, 8, -11, 14, -17, 20, -23, 26, -29, 32, -35, 38, -41, 44, -47, 50, -53, 56]
  -- We define the sequence in list form for clarity
  (s.sum = 29) :=
by
  let a_1 := 2
  let d := 3
  let s := [2, -5, 8, -11, 14, -17, 20, -23, 26, -29, 32, -35, 38, -41, 44, -47, 50, -53, 56]
  sorry

end series_sum_l1557_155730


namespace other_position_in_arithmetic_progression_l1557_155705

theorem other_position_in_arithmetic_progression 
  (a d : ℝ) (x : ℕ)
  (h1 : a + (4 - 1) * d + a + (x - 1) * d = 20)
  (h2 : 5 * (2 * a + 9 * d) = 100) :
  x = 7 := by
  sorry

end other_position_in_arithmetic_progression_l1557_155705


namespace initial_bees_l1557_155763

theorem initial_bees (B : ℕ) (h : B + 7 = 23) : B = 16 :=
by {
  sorry
}

end initial_bees_l1557_155763


namespace average_marks_l1557_155720

theorem average_marks (P C M : ℝ) (h1 : P = 95) (h2 : (P + M) / 2 = 90) (h3 : (P + C) / 2 = 70) :
  (P + C + M) / 3 = 75 := 
by
  sorry

end average_marks_l1557_155720


namespace paint_fence_together_time_l1557_155736

-- Define the times taken by Jamshid and Taimour
def Taimour_time := 18 -- Taimour takes 18 hours to paint the fence
def Jamshid_time := Taimour_time / 2 -- Jamshid takes half the time Taimour takes

-- Define the work rates
def Taimour_rate := 1 / Taimour_time
def Jamshid_rate := 1 / Jamshid_time

-- Define the combined work rate
def combined_rate := Taimour_rate + Jamshid_rate

-- Define the total time taken when working together
def together_time := 1 / combined_rate

-- State the main theorem
theorem paint_fence_together_time : together_time = 6 := 
sorry

end paint_fence_together_time_l1557_155736


namespace sufficient_not_necessary_l1557_155798

variables (A B : Prop)

theorem sufficient_not_necessary (h : B → A) : ¬(A → B) :=
by sorry

end sufficient_not_necessary_l1557_155798


namespace largest_difference_l1557_155703

noncomputable def A : ℕ := 3 * 2010 ^ 2011
noncomputable def B : ℕ := 2010 ^ 2011
noncomputable def C : ℕ := 2009 * 2010 ^ 2010
noncomputable def D : ℕ := 3 * 2010 ^ 2010
noncomputable def E : ℕ := 2010 ^ 2010
noncomputable def F : ℕ := 2010 ^ 2009

theorem largest_difference :
  (A - B = 2 * 2010 ^ 2011) ∧ 
  (B - C = 2010 ^ 2010) ∧ 
  (C - D = 2006 * 2010 ^ 2010) ∧ 
  (D - E = 2 * 2010 ^ 2010) ∧ 
  (E - F = 2009 * 2010 ^ 2009) ∧ 
  (2 * 2010 ^ 2011 > 2010 ^ 2010) ∧
  (2 * 2010 ^ 2011 > 2006 * 2010 ^ 2010) ∧
  (2 * 2010 ^ 2011 > 2 * 2010 ^ 2010) ∧
  (2 * 2010 ^ 2011 > 2009 * 2010 ^ 2009) :=
by
  sorry

end largest_difference_l1557_155703
