import Mathlib

namespace calculate_percentage_l329_32917

theorem calculate_percentage :
  let total_students := 40
  let A_on_both := 4
  let B_on_both := 6
  let C_on_both := 3
  let D_on_Test1_C_on_Test2 := 2
  let valid_students := A_on_both + B_on_both + C_on_both + D_on_Test1_C_on_Test2
  (valid_students / total_students) * 100 = 37.5 :=
by
  sorry

end calculate_percentage_l329_32917


namespace a_n_strictly_monotonic_increasing_l329_32939

noncomputable def a_n (n : ℕ) : ℝ := 
  2 * ((1 + 1 / (n : ℝ)) ^ (2 * n + 1)) / (((1 + 1 / (n : ℝ)) ^ n) + ((1 + 1 / (n : ℝ)) ^ (n + 1)))

theorem a_n_strictly_monotonic_increasing : ∀ n : ℕ, a_n (n + 1) > a_n n :=
sorry

end a_n_strictly_monotonic_increasing_l329_32939


namespace value_of_z_l329_32955

theorem value_of_z :
  let mean_of_4_16_20 := (4 + 16 + 20) / 3
  let mean_of_8_z := (8 + z) / 2
  ∀ z : ℚ, mean_of_4_16_20 = mean_of_8_z → z = 56 / 3 := 
by
  intro z mean_eq
  sorry

end value_of_z_l329_32955


namespace rectangle_area_l329_32981

theorem rectangle_area (L W : ℕ) (h1 : 2 * L + 2 * W = 280) (h2 : L = 5 * (W / 2)) : L * W = 4000 :=
sorry

end rectangle_area_l329_32981


namespace cube_surface_area_difference_l329_32959

theorem cube_surface_area_difference :
  let large_cube_volume := 8
  let small_cube_volume := 1
  let num_small_cubes := 8
  let large_cube_side := (large_cube_volume : ℝ) ^ (1 / 3)
  let small_cube_side := (small_cube_volume : ℝ) ^ (1 / 3)
  let large_cube_surface_area := 6 * (large_cube_side ^ 2)
  let small_cube_surface_area := 6 * (small_cube_side ^ 2)
  let total_small_cubes_surface_area := num_small_cubes * small_cube_surface_area
  total_small_cubes_surface_area - large_cube_surface_area = 24 :=
by
  sorry

end cube_surface_area_difference_l329_32959


namespace rick_division_steps_l329_32913

theorem rick_division_steps (initial_books : ℕ) (final_books : ℕ) 
  (h_initial : initial_books = 400) (h_final : final_books = 25) : 
  (∀ n : ℕ, (initial_books / (2^n) = final_books) → n = 4) :=
by
  sorry

end rick_division_steps_l329_32913


namespace sum_quotient_dividend_divisor_l329_32936

theorem sum_quotient_dividend_divisor (n : ℕ) (d : ℕ) (h : n = 45) (h1 : d = 3) : 
  (n / d) + n + d = 63 :=
by
  sorry

end sum_quotient_dividend_divisor_l329_32936


namespace min_value_of_one_over_a_and_one_over_b_l329_32968

noncomputable def minValue (a b : ℝ) : ℝ :=
  if 2 * a + 3 * b = 1 then 1 / a + 1 / b else 0

theorem min_value_of_one_over_a_and_one_over_b :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + 3 * b = 1 ∧ minValue a b = 65 / 6 :=
by
  sorry

end min_value_of_one_over_a_and_one_over_b_l329_32968


namespace sum_of_integers_l329_32937

theorem sum_of_integers (a b : ℕ) (h1 : a * a + b * b = 585) (h2 : Nat.gcd a b + Nat.lcm a b = 87) : a + b = 33 := 
sorry

end sum_of_integers_l329_32937


namespace find_solutions_l329_32919

theorem find_solutions (k : ℤ) (x y : ℤ) (h : x^2 - 2*y^2 = k) :
  ∃ t u : ℤ, t^2 - 2*u^2 = -k ∧ (t = x + 2*y ∨ t = x - 2*y) ∧ (u = x + y ∨ u = x - y) :=
sorry

end find_solutions_l329_32919


namespace cumulus_to_cumulonimbus_ratio_l329_32987

theorem cumulus_to_cumulonimbus_ratio (cirrus cumulonimbus cumulus : ℕ) (x : ℕ)
  (h1 : cirrus = 4 * cumulus)
  (h2 : cumulus = x * cumulonimbus)
  (h3 : cumulonimbus = 3)
  (h4 : cirrus = 144) :
  x = 12 := by
  sorry

end cumulus_to_cumulonimbus_ratio_l329_32987


namespace geom_seq_min_m_l329_32986

def initial_capital : ℝ := 50
def growth_rate : ℝ := 0.5
def annual_payment (t : ℝ) : Prop := t ≤ 2500
def capital_remaining (aₙ : ℕ → ℝ) (n : ℕ) (t : ℝ) : ℝ := aₙ n * (1 + growth_rate) - t

theorem geom_seq (aₙ : ℕ → ℝ) (t : ℝ) (h₁ : annual_payment t) :
  (∀ n, aₙ (n + 1) = 3 / 2 * aₙ n - t) →
  (t ≠ 2500) →
  ∃ r : ℝ, ∀ n, aₙ n - 2 * t = (aₙ 0 - 2 * t) * r ^ n :=
sorry

theorem min_m (t : ℝ) (h₁ : t = 1500) (aₙ : ℕ → ℝ) :
  (∀ n, aₙ (n + 1) = 3 / 2 * aₙ n - t) →
  (aₙ 0 = initial_capital * (1 + growth_rate) - t) →
  ∃ m : ℕ, aₙ m > 21000 ∧ ∀ k < m, aₙ k ≤ 21000 :=
sorry

end geom_seq_min_m_l329_32986


namespace stephanie_total_remaining_bills_l329_32989

-- Conditions
def electricity_bill : ℕ := 60
def electricity_paid : ℕ := electricity_bill
def gas_bill : ℕ := 40
def gas_paid : ℕ := (3 * gas_bill) / 4 + 5
def water_bill : ℕ := 40
def water_paid : ℕ := water_bill / 2
def internet_bill : ℕ := 25
def internet_payment : ℕ := 5
def internet_paid : ℕ := 4 * internet_payment

-- Define
def remaining_electricity : ℕ := electricity_bill - electricity_paid
def remaining_gas : ℕ := gas_bill - gas_paid
def remaining_water : ℕ := water_bill - water_paid
def remaining_internet : ℕ := internet_bill - internet_paid

def total_remaining : ℕ := remaining_electricity + remaining_gas + remaining_water + remaining_internet

-- Problem Statement
theorem stephanie_total_remaining_bills :
  total_remaining = 30 :=
by
  -- proof goes here (not required as per the instructions)
  sorry

end stephanie_total_remaining_bills_l329_32989


namespace bookshop_shipment_correct_l329_32991

noncomputable def bookshop_shipment : ℕ :=
  let Initial_books := 743
  let Saturday_instore_sales := 37
  let Saturday_online_sales := 128
  let Sunday_instore_sales := 2 * Saturday_instore_sales
  let Sunday_online_sales := Saturday_online_sales + 34
  let books_sold := Saturday_instore_sales + Saturday_online_sales + Sunday_instore_sales + Sunday_online_sales
  let Final_books := 502
  Final_books - (Initial_books - books_sold)

theorem bookshop_shipment_correct : bookshop_shipment = 160 := by
  sorry

end bookshop_shipment_correct_l329_32991


namespace find_chord_line_eq_l329_32900

theorem find_chord_line_eq (P : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ)
    (hP : P = (1, 1)) (hC : C = (3, 0)) (hr : r = 3)
    (circle_eq : ∀ (x y : ℝ), (x - 3)^2 + y^2 = r^2) :
    ∃ (a b c : ℝ), a = 2 ∧ b = -1 ∧ c = -1 ∧ ∀ (x y : ℝ), a * x + b * y + c = 0 := by
  sorry

end find_chord_line_eq_l329_32900


namespace infinite_series_sum_l329_32958

theorem infinite_series_sum :
  ∑' (k : ℕ), (k + 1) / 4^(k + 1) = 4 / 9 :=
sorry

end infinite_series_sum_l329_32958


namespace discriminant_of_quad_eq_l329_32957

def a : ℕ := 5
def b : ℕ := 8
def c : ℤ := -6

def discriminant (a b c : ℤ) : ℤ := b^2 - 4 * a * c

theorem discriminant_of_quad_eq : discriminant 5 8 (-6) = 184 :=
by
  -- The proof is skipped
  sorry

end discriminant_of_quad_eq_l329_32957


namespace polynomial_identity_l329_32961

theorem polynomial_identity (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a^2 + b^2 + c^2 = 15) 
  (h3 : a^3 + b^3 + c^3 = 47) : 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) = 625 := 
by 
  sorry

end polynomial_identity_l329_32961


namespace propositions_p_q_l329_32970

theorem propositions_p_q
  (p q : Prop)
  (h : ¬(p ∧ q) = False) : p ∧ q :=
by
  sorry

end propositions_p_q_l329_32970


namespace expression_equals_4096_l329_32941

noncomputable def calculate_expression : ℕ :=
  ((16^15 / 16^14)^3 * 8^3) / 2^9

theorem expression_equals_4096 : calculate_expression = 4096 :=
by {
  -- proof would go here
  sorry
}

end expression_equals_4096_l329_32941


namespace optimal_purchase_interval_discount_advantage_l329_32998

/- The functions and assumptions used here. -/
def purchase_feed_days (feed_per_day : ℕ) (price_per_kg : ℝ) 
  (storage_cost_per_kg_per_day : ℝ) (transportation_fee : ℝ) : ℕ :=
-- Implementation omitted
sorry

def should_use_discount (feed_per_day : ℕ) (price_per_kg : ℝ) 
  (storage_cost_per_kg_per_day : ℝ) (transportation_fee : ℝ) 
  (discount_threshold : ℕ) (discount_rate : ℝ) : Prop :=
-- Implementation omitted
sorry

/- Conditions -/
def conditions : Prop :=
  let feed_per_day := 200
  let price_per_kg := 1.8
  let storage_cost_per_kg_per_day := 0.03
  let transportation_fee := 300
  let discount_threshold := 5000 -- in kg, since 5 tons = 5000 kg
  let discount_rate := 0.85
  True -- We apply these values in the proofs below.

/- Main statements -/
theorem optimal_purchase_interval : conditions → 
  purchase_feed_days 200 1.8 0.03 300 = 10 :=
by
  intros
  -- Proof is omitted.
  sorry

theorem discount_advantage : conditions →
  should_use_discount 200 1.8 0.03 300 5000 0.85 :=
by
  intros
  -- Proof is omitted.
  sorry

end optimal_purchase_interval_discount_advantage_l329_32998


namespace Derek_more_than_Zoe_l329_32995

-- Define the variables for the number of books Emily, Derek, and Zoe have
variables (E : ℝ)

-- Condition: Derek has 75% more books than Emily
def Derek_books : ℝ := 1.75 * E

-- Condition: Zoe has 50% more books than Emily
def Zoe_books : ℝ := 1.5 * E

-- Statement asserting that Derek has 16.67% more books than Zoe
theorem Derek_more_than_Zoe (hD: Derek_books E = 1.75 * E) (hZ: Zoe_books E = 1.5 * E) :
  (Derek_books E - Zoe_books E) / Zoe_books E = 0.1667 :=
by
  sorry

end Derek_more_than_Zoe_l329_32995


namespace determine_teeth_l329_32973

theorem determine_teeth (x V : ℝ) (h1 : V = 63 * x / (x + 10)) (h2 : V = 28 * (x + 10)) :
  x = 20 ∧ (x + 10) = 30 :=
by
  sorry

end determine_teeth_l329_32973


namespace intersection_M_N_l329_32990

def M : Set ℝ := {x | x < 2016}

def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l329_32990


namespace scientific_notation_correct_l329_32924

theorem scientific_notation_correct : 657000 = 6.57 * 10^5 :=
by
  sorry

end scientific_notation_correct_l329_32924


namespace max_sum_m_n_l329_32978

noncomputable def ellipse_and_hyperbola_max_sum : Prop :=
  ∃ m n : ℝ, m > 0 ∧ n > 0 ∧ (∃ x y : ℝ, (x^2 / 25 + y^2 / m^2 = 1 ∧ x^2 / 7 - y^2 / n^2 = 1)) ∧
  (25 - m^2 = 7 + n^2) ∧ (m + n = 6)

theorem max_sum_m_n : ellipse_and_hyperbola_max_sum :=
  sorry

end max_sum_m_n_l329_32978


namespace other_number_remainder_l329_32933

theorem other_number_remainder (x : ℕ) (k n : ℤ) (hx : x > 0) (hk : 200 = k * x + 2) (hnk : n ≠ k) : ∃ m : ℤ, (n * ↑x + 2) = m * ↑x + 2 ∧ (n * ↑x + 2) % x = 2 := 
by
  sorry

end other_number_remainder_l329_32933


namespace sum_of_products_is_70_l329_32964

theorem sum_of_products_is_70 (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 149) (h2 : a + b + c = 17) :
  a * b + b * c + c * a = 70 :=
by
  sorry 

end sum_of_products_is_70_l329_32964


namespace speed_ratio_l329_32971

theorem speed_ratio :
  ∀ (v_A v_B : ℝ), (v_A / v_B = 3 / 2) ↔ (v_A = 3 * v_B / 2) :=
by
  intros
  sorry

end speed_ratio_l329_32971


namespace exponential_inequality_l329_32929

theorem exponential_inequality (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 1) : 
  x₂ * Real.exp x₁ > x₁ * Real.exp x₂ :=
sorry

end exponential_inequality_l329_32929


namespace jean_candy_count_l329_32923

theorem jean_candy_count : ∃ C : ℕ, 
  C - 7 = 16 ∧ 
  (C - 7 + 7 = C) ∧ 
  (C - 7 = 16) ∧ 
  (C + 0 = C) ∧
  (C - 7 = 16) :=
by 
  sorry 

end jean_candy_count_l329_32923


namespace sufficient_not_necessary_condition_l329_32975

variables (a b c : ℝ)

theorem sufficient_not_necessary_condition (h1 : c < b) (h2 : b < a) :
  (ac < 0 → ab > ac) ∧ (ab > ac → ac < 0) → false :=
sorry

end sufficient_not_necessary_condition_l329_32975


namespace retail_price_eq_120_l329_32903

noncomputable def retail_price : ℝ :=
  let W := 90
  let P := 0.20 * W
  let SP := W + P
  SP / 0.90

theorem retail_price_eq_120 : retail_price = 120 := by
  sorry

end retail_price_eq_120_l329_32903


namespace min_cos_beta_l329_32943

open Real

theorem min_cos_beta (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_eq : sin (2 * α + β) = (3 / 2) * sin β) :
  cos β = sqrt 5 / 3 := 
sorry

end min_cos_beta_l329_32943


namespace max_number_of_band_members_l329_32940

-- Conditions definitions
def num_band_members (r x : ℕ) : ℕ := r * x + 3

def num_band_members_new (r x : ℕ) : ℕ := (r - 1) * (x + 2)

-- The main statement
theorem max_number_of_band_members :
  ∃ (r x : ℕ), num_band_members r x = 231 ∧ num_band_members_new r x = 231 
  ∧ ∀ (r' x' : ℕ), (num_band_members r' x' < 120 ∧ num_band_members_new r' x' = num_band_members r' x') → (num_band_members r' x' ≤ 231) :=
sorry

end max_number_of_band_members_l329_32940


namespace mode_of_shoe_sizes_is_25_5_l329_32988

def sales_data := [(24, 2), (24.5, 5), (25, 3), (25.5, 6), (26, 4)]

theorem mode_of_shoe_sizes_is_25_5 
  (h : ∀ x ∈ sales_data, 2 ≤ x.1 ∧ 
        (∀ y ∈ sales_data, x.2 ≤ y.2 → x.1 = 25.5 ∨ x.2 < 6)) : 
  (∃ s, s ∈ sales_data ∧ s.1 = 25.5 ∧ s.2 = 6) :=
sorry

end mode_of_shoe_sizes_is_25_5_l329_32988


namespace permutation_six_two_l329_32921

-- Definition for permutation
def permutation (n k : ℕ) : ℕ := n * (n - 1)

-- Theorem stating that the permutation of 6 taken 2 at a time is 30
theorem permutation_six_two : permutation 6 2 = 30 :=
by
  -- proof will be filled here
  sorry

end permutation_six_two_l329_32921


namespace area_of_triangle_with_given_sides_l329_32927

variable (a b c : ℝ)
variable (s : ℝ := (a + b + c) / 2)
variable (area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c)))

theorem area_of_triangle_with_given_sides (ha : a = 65) (hb : b = 60) (hc : c = 25) :
  area = 750 := by
  sorry

end area_of_triangle_with_given_sides_l329_32927


namespace maximize_container_volume_l329_32945

theorem maximize_container_volume :
  ∃ x : ℝ, 0 < x ∧ x < 24 ∧ (∀ y : ℝ, 0 < y ∧ y < 24 → (90 - 2*y) * (48 - 2*y) * y ≤ (90 - 2*x) * (48 - 2*x) * x) ∧ x = 10 :=
sorry

end maximize_container_volume_l329_32945


namespace root_equation_solution_l329_32928

-- Given conditions from the problem
def is_root_of_quadratic (m : ℝ) : Prop :=
  m^2 - m - 110 = 0

-- Statement of the proof problem
theorem root_equation_solution (m : ℝ) (h : is_root_of_quadratic m) : (m - 1)^2 + m = 111 := 
sorry

end root_equation_solution_l329_32928


namespace determine_e_l329_32920

-- Define the polynomial Q(x)
def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

-- Define the problem statement
theorem determine_e (d e f : ℝ)
  (h1 : f = 9)
  (h2 : (d * (d + 9)) - 168 = 0)
  (h3 : d^2 - 6 * e = 12 + d + e)
  : e = -24 ∨ e = 20 :=
by
  sorry

end determine_e_l329_32920


namespace student_failed_by_40_marks_l329_32980

theorem student_failed_by_40_marks (total_marks : ℕ) (passing_percentage : ℝ) (marks_obtained : ℕ) (h1 : total_marks = 500) (h2 : passing_percentage = 33) (h3 : marks_obtained = 125) :
  ((passing_percentage / 100) * total_marks - marks_obtained : ℝ) = 40 :=
sorry

end student_failed_by_40_marks_l329_32980


namespace find_constants_l329_32994

noncomputable def f (x m n : ℝ) := (m * x + 1) / (x + n)

theorem find_constants (m n : ℝ) (h_symm : ∀ x y, f x m n = y → f (4 - x) m n = 8 - y) : 
  m = 4 ∧ n = -2 := 
by
  sorry

end find_constants_l329_32994


namespace correct_average_weight_is_58_6_l329_32969

noncomputable def initial_avg_weight : ℚ := 58.4
noncomputable def num_boys : ℕ := 20
noncomputable def incorrect_weight : ℚ := 56
noncomputable def correct_weight : ℚ := 60
noncomputable def correct_avg_weight := (initial_avg_weight * num_boys + (correct_weight - incorrect_weight)) / num_boys

theorem correct_average_weight_is_58_6 :
  correct_avg_weight = 58.6 :=
sorry

end correct_average_weight_is_58_6_l329_32969


namespace farmer_planning_problem_l329_32910

theorem farmer_planning_problem
  (A : ℕ) (D : ℕ)
  (h1 : A = 120 * D)
  (h2 : ∀ t : ℕ, t = 85 * (D + 5) + 40)
  (h3 : 85 * (D + 5) + 40 = 120 * D) : 
  A = 1560 ∧ D = 13 := 
by
  sorry

end farmer_planning_problem_l329_32910


namespace mean_age_Mendez_children_l329_32946

def Mendez_children_ages : List ℕ := [5, 5, 10, 12, 15]

theorem mean_age_Mendez_children : 
  (5 + 5 + 10 + 12 + 15) / 5 = 9.4 := 
by
  sorry

end mean_age_Mendez_children_l329_32946


namespace speed_conversion_l329_32912

-- Define the conversion factor
def conversion_factor := 3.6

-- Define the given speed in meters per second
def speed_mps := 16.668

-- Define the expected speed in kilometers per hour
def expected_speed_kmph := 60.0048

-- The theorem to prove that the given speed in m/s converts to the expected speed in km/h
theorem speed_conversion : speed_mps * conversion_factor = expected_speed_kmph := 
  by
    sorry

end speed_conversion_l329_32912


namespace circuit_analysis_l329_32901

/-
There are 3 conducting branches connected between points A and B.
First branch: a 2 Volt EMF and a 2 Ohm resistor connected in series.
Second branch: a 2 Volt EMF and a 1 Ohm resistor.
Third branch: a conductor with a resistance of 1 Ohm.
Prove the currents and voltage drop are as follows:
- Current in first branch: i1 = 0.4 A
- Current in second branch: i2 = 0.8 A
- Current in third branch: i3 = 1.2 A
- Voltage between A and B: E_AB = 1.2 Volts
-/
theorem circuit_analysis :
  ∃ (i1 i2 i3 : ℝ) (E_AB : ℝ),
    (i1 = 0.4) ∧
    (i2 = 0.8) ∧
    (i3 = 1.2) ∧
    (E_AB = 1.2) ∧
    (2 = 2 * i1 + i3) ∧
    (2 = i2 + i3) ∧
    (i3 = i1 + i2) ∧
    (E_AB = i3 * 1) := sorry

end circuit_analysis_l329_32901


namespace possible_birches_l329_32992

theorem possible_birches (N B L : ℕ) (hN : N = 130) (h_sum : B + L = 130)
  (h_linden_false : ∀ l, l < L → (∀ b, b < B → b + l < N → b < B → False))
  (h_birch_false : ∃ b, b < B ∧ (∀ l, l < L → l + b < N → l + b = 2 * B))
  : B = 87 :=
sorry

end possible_birches_l329_32992


namespace earl_envelope_rate_l329_32907

theorem earl_envelope_rate:
  ∀ (E L : ℝ),
  L = (2/3) * E ∧
  (E + L = 60) →
  E = 36 :=
by
  intros E L h
  sorry

end earl_envelope_rate_l329_32907


namespace range_of_a_l329_32949

def is_ellipse (a : ℝ) : Prop :=
  2 * a > 0 ∧ 3 * a - 6 > 0 ∧ 2 * a < 3 * a - 6

def discriminant_neg (a : ℝ) : Prop :=
  a^2 + 8 * a - 48 < 0

def p (a : ℝ) : Prop := is_ellipse a
def q (a : ℝ) : Prop := discriminant_neg a

theorem range_of_a (a : ℝ) : p a ∧ q a → 2 < a ∧ a < 4 :=
by
  sorry

end range_of_a_l329_32949


namespace min_value_frac_l329_32926

theorem min_value_frac (m n : ℝ) (hmn : m + n = 1) (hm : 0 < m) (hn : 0 < n) :
  ∃ (x : ℝ), x = 1/m + 4/n ∧ x ≥ 9 :=
by
  sorry

end min_value_frac_l329_32926


namespace vaishali_total_stripes_l329_32904

theorem vaishali_total_stripes
  (hats1 : ℕ) (stripes1 : ℕ)
  (hats2 : ℕ) (stripes2 : ℕ)
  (hats3 : ℕ) (stripes3 : ℕ)
  (hats4 : ℕ) (stripes4 : ℕ)
  (total_stripes : ℕ) :
  hats1 = 4 → stripes1 = 3 →
  hats2 = 3 → stripes2 = 4 →
  hats3 = 6 → stripes3 = 0 →
  hats4 = 2 → stripes4 = 5 →
  total_stripes = (hats1 * stripes1) + (hats2 * stripes2) + (hats3 * stripes3) + (hats4 * stripes4) →
  total_stripes = 34 := by
  sorry

end vaishali_total_stripes_l329_32904


namespace tetrahedron_max_volume_l329_32972

noncomputable def tetrahedron_volume (AC AB BD CD : ℝ) : ℝ :=
  let x := (2 : ℝ) * (Real.sqrt 3) / 3
  let m := Real.sqrt (1 - x^2 / 4)
  let α := Real.pi / 2 -- Maximize with sin α = 1
  x * m^2 * Real.sin α / 6

theorem tetrahedron_max_volume : ∀ (AC AB BD CD : ℝ),
  AC = 1 → AB = 1 → BD = 1 → CD = 1 →
  tetrahedron_volume AC AB BD CD = 2 * Real.sqrt 3 / 27 :=
by
  intros AC AB BD CD hAC hAB hBD hCD
  rw [hAC, hAB, hBD, hCD]
  dsimp [tetrahedron_volume]
  norm_num
  sorry

end tetrahedron_max_volume_l329_32972


namespace absolute_value_inequality_solution_l329_32909

theorem absolute_value_inequality_solution (x : ℝ) :
  |x - 2| + |x - 4| ≤ 3 ↔ (3 / 2 ≤ x ∧ x < 4) :=
by
  sorry

end absolute_value_inequality_solution_l329_32909


namespace dot_product_vec_a_vec_b_l329_32954

def vec_a : ℝ × ℝ := (-1, 2)
def vec_b : ℝ × ℝ := (1, 2)

theorem dot_product_vec_a_vec_b : vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 = 3 := by
  sorry

end dot_product_vec_a_vec_b_l329_32954


namespace cos_double_angle_l329_32911

theorem cos_double_angle (α : ℝ) (h : Real.sin α = (Real.sqrt 3) / 2) : 
  Real.cos (2 * α) = -1 / 2 :=
by
  sorry

end cos_double_angle_l329_32911


namespace length_of_overlapping_part_l329_32908

theorem length_of_overlapping_part
  (l_p : ℕ)
  (n : ℕ)
  (total_length : ℕ)
  (l_o : ℕ) :
  n = 3 →
  l_p = 217 →
  total_length = 627 →
  3 * l_p - 2 * l_o = total_length →
  l_o = 12 := by
  intros n_eq l_p_eq total_length_eq equation
  sorry

end length_of_overlapping_part_l329_32908


namespace compute_expression_l329_32905

theorem compute_expression (p q : ℝ) (h1 : p + q = 5) (h2 : p * q = 6) :
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 503 :=
by
  sorry

end compute_expression_l329_32905


namespace product_of_y_coordinates_l329_32974

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem product_of_y_coordinates : 
  let P1 := (1, 2 + 4 * Real.sqrt 2)
  let P2 := (1, 2 - 4 * Real.sqrt 2)
  distance (5, 2) P1 = 12 ∧ distance (5, 2) P2 = 12 →
  (P1.2 * P2.2 = -28) :=
by
  intros
  sorry

end product_of_y_coordinates_l329_32974


namespace ed_initial_money_l329_32902

-- Define initial conditions
def cost_per_hour_night : ℝ := 1.50
def hours_at_night : ℕ := 6
def cost_per_hour_morning : ℝ := 2
def hours_in_morning : ℕ := 4
def money_left : ℝ := 63

-- Total cost calculation
def total_cost : ℝ :=
  (cost_per_hour_night * hours_at_night) + (cost_per_hour_morning * hours_in_morning)

-- Problem statement to prove
theorem ed_initial_money : money_left + total_cost = 80 :=
by sorry

end ed_initial_money_l329_32902


namespace candy_weight_reduction_l329_32944

theorem candy_weight_reduction:
  ∀ (W P : ℝ), (33.333333333333314 / 100) * (P / W) = (P / (W - (1/4) * W)) →
  (1 - (W - (1/4) * W) / W) * 100 = 25 :=
by
  intros W P h
  sorry

end candy_weight_reduction_l329_32944


namespace condition_B_is_necessary_but_not_sufficient_l329_32999

-- Definitions of conditions A and B
def condition_A (x : ℝ) : Prop := 0 < x ∧ x < 5
def condition_B (x : ℝ) : Prop := abs (x - 2) < 3

-- The proof problem statement
theorem condition_B_is_necessary_but_not_sufficient : 
∀ x, condition_A x → condition_B x ∧ ¬(∀ x, condition_B x → condition_A x) := 
sorry

end condition_B_is_necessary_but_not_sufficient_l329_32999


namespace probability_pair_tile_l329_32950

def letters_in_word : List Char := ['P', 'R', 'O', 'B', 'A', 'B', 'I', 'L', 'I', 'T', 'Y']
def target_letters : List Char := ['P', 'A', 'I', 'R']

def num_favorable_outcomes : Nat :=
  -- Count occurrences of letters in target_letters within letters_in_word
  List.count 'P' letters_in_word +
  List.count 'A' letters_in_word +
  List.count 'I' letters_in_word + 
  List.count 'R' letters_in_word

def total_outcomes : Nat := letters_in_word.length

theorem probability_pair_tile :
  (num_favorable_outcomes : ℚ) / total_outcomes = 5 / 12 := by
  sorry

end probability_pair_tile_l329_32950


namespace total_customers_served_l329_32914

-- Definitions for the hours worked by Ann, Becky, and Julia
def hours_ann : ℕ := 8
def hours_becky : ℕ := 8
def hours_julia : ℕ := 6

-- Definition for the number of customers served per hour
def customers_per_hour : ℕ := 7

-- Total number of customers served by Ann, Becky, and Julia
def total_customers : ℕ :=
  (hours_ann * customers_per_hour) + 
  (hours_becky * customers_per_hour) + 
  (hours_julia * customers_per_hour)

theorem total_customers_served : total_customers = 154 :=
  by 
    -- This is where the proof would go, but we'll use sorry to indicate it's incomplete
    sorry

end total_customers_served_l329_32914


namespace decision_represented_by_D_l329_32942

-- Define the basic symbols in the flowchart
inductive BasicSymbol
| Start
| Process
| Decision
| End

open BasicSymbol

-- Define the meaning of each basic symbol
def meaning_of (sym : BasicSymbol) : String :=
  match sym with
  | Start => "start"
  | Process => "process"
  | Decision => "decision"
  | End => "end"

-- The theorem stating that the Decision symbol represents a decision
theorem decision_represented_by_D : meaning_of Decision = "decision" :=
by sorry

end decision_represented_by_D_l329_32942


namespace probability_square_area_l329_32993

theorem probability_square_area (AB : ℝ) (M : ℝ) (h1 : AB = 12) (h2 : 0 ≤ M) (h3 : M ≤ AB) :
  (∃ (AM : ℝ), (AM = M) ∧ (36 ≤ AM^2 ∧ AM^2 ≤ 81)) → 
  (∃ (p : ℝ), p = 1/4) :=
by
  sorry

end probability_square_area_l329_32993


namespace bus_speed_excluding_stoppages_l329_32967

theorem bus_speed_excluding_stoppages 
  (v_s : ℕ) -- Speed including stoppages in kmph
  (stop_duration_minutes : ℕ) -- Duration of stoppages in minutes per hour
  (stop_duration_fraction : ℚ := stop_duration_minutes / 60) -- Fraction of hour stopped
  (moving_fraction : ℚ := 1 - stop_duration_fraction) -- Fraction of hour moving
  (distance_per_hour : ℚ := v_s) -- Distance traveled per hour including stoppages
  (v : ℚ) -- Speed excluding stoppages
  
  (h1 : v_s = 50)
  (h2 : stop_duration_minutes = 10)
  
  -- Equation representing the total distance equals the distance traveled moving
  (h3 : v * moving_fraction = distance_per_hour)
: v = 60 := sorry

end bus_speed_excluding_stoppages_l329_32967


namespace jellybeans_needed_l329_32931

-- Define the initial conditions as constants
def jellybeans_per_large_glass := 50
def jellybeans_per_small_glass := jellybeans_per_large_glass / 2
def number_of_large_glasses := 5
def number_of_small_glasses := 3

-- Calculate the total number of jellybeans needed
def total_jellybeans : ℕ :=
  (number_of_large_glasses * jellybeans_per_large_glass) + 
  (number_of_small_glasses * jellybeans_per_small_glass)

-- Prove that the total number of jellybeans needed is 325
theorem jellybeans_needed : total_jellybeans = 325 :=
sorry

end jellybeans_needed_l329_32931


namespace locus_of_P_l329_32935

-- Definitions based on conditions
def F : ℝ × ℝ := (2, 0)
def Q (k : ℝ) : ℝ × ℝ := (0, -2 * k)
def T (k : ℝ) : ℝ × ℝ := (-2 * k^2, 0)
def P (k : ℝ) : ℝ × ℝ := (2 * k^2, -4 * k)

-- Theorem statement based on the proof problem
theorem locus_of_P (x y : ℝ) (k : ℝ) (hf : F = (2, 0)) (hq : Q k = (0, -2 * k))
  (ht : T k = (-2 * k^2, 0)) (hp : P k = (2 * k^2, -4 * k)) :
  y^2 = 8 * x :=
sorry

end locus_of_P_l329_32935


namespace intersection_M_N_l329_32963

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x * x = x}

theorem intersection_M_N :
  M ∩ N = {0, 1} :=
sorry

end intersection_M_N_l329_32963


namespace percentage_non_defective_l329_32985

theorem percentage_non_defective :
  let total_units : ℝ := 100
  let M1_percentage : ℝ := 0.20
  let M2_percentage : ℝ := 0.25
  let M3_percentage : ℝ := 0.30
  let M4_percentage : ℝ := 0.15
  let M5_percentage : ℝ := 0.10
  let M1_defective_percentage : ℝ := 0.02
  let M2_defective_percentage : ℝ := 0.04
  let M3_defective_percentage : ℝ := 0.05
  let M4_defective_percentage : ℝ := 0.07
  let M5_defective_percentage : ℝ := 0.08

  let M1_total := total_units * M1_percentage
  let M2_total := total_units * M2_percentage
  let M3_total := total_units * M3_percentage
  let M4_total := total_units * M4_percentage
  let M5_total := total_units * M5_percentage

  let M1_defective := M1_total * M1_defective_percentage
  let M2_defective := M2_total * M2_defective_percentage
  let M3_defective := M3_total * M3_defective_percentage
  let M4_defective := M4_total * M4_defective_percentage
  let M5_defective := M5_total * M5_defective_percentage

  let total_defective := M1_defective + M2_defective + M3_defective + M4_defective + M5_defective
  let total_non_defective := total_units - total_defective
  let percentage_non_defective := (total_non_defective / total_units) * 100

  percentage_non_defective = 95.25 := by
  sorry

end percentage_non_defective_l329_32985


namespace correct_operation_l329_32997

theorem correct_operation (a b : ℝ) :
  2 * a^2 * b - 3 * a^2 * b = - (a^2 * b) :=
by
  sorry

end correct_operation_l329_32997


namespace problem_statement_l329_32962

-- Define y as the sum of the given terms
def y : ℤ := 128 + 192 + 256 + 320 + 576 + 704 + 6464

-- The theorem to prove that y is a multiple of 8, 16, 32, and 64
theorem problem_statement : 
  (8 ∣ y) ∧ (16 ∣ y) ∧ (32 ∣ y) ∧ (64 ∣ y) :=
by sorry

end problem_statement_l329_32962


namespace question1_1_question1_2_question2_l329_32984

open Set

noncomputable def universal_set : Set ℝ := univ

def setA : Set ℝ := { x | x^2 - 9 * x + 18 ≥ 0 }

def setB : Set ℝ := { x | -2 < x ∧ x < 9 }

def setC (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

theorem question1_1 : ∀ x, x ∈ setA ∨ x ∈ setB :=
by sorry

theorem question1_2 : ∀ x, x ∈ (universal_set \ setA) ∩ setB ↔ (3 < x ∧ x < 6) :=
by sorry

theorem question2 (a : ℝ) (h : setC a ⊆ setB) : -2 ≤ a ∧ a ≤ 8 :=
by sorry

end question1_1_question1_2_question2_l329_32984


namespace sales_tax_percentage_l329_32922

theorem sales_tax_percentage 
  (total_spent : ℝ)
  (tip_percent : ℝ)
  (food_price : ℝ) 
  (total_with_tip : total_spent = food_price * (1 + tip_percent / 100))
  (sales_tax_percent : ℝ) 
  (total_paid : total_spent = food_price * (1 + sales_tax_percent / 100) * (1 + tip_percent / 100)) :
  sales_tax_percent = 10 :=
by sorry

end sales_tax_percentage_l329_32922


namespace investment_compound_half_yearly_l329_32925

theorem investment_compound_half_yearly
  (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) (t : ℝ) 
  (h1 : P = 6000) 
  (h2 : r = 0.10) 
  (h3 : n = 2) 
  (h4 : A = 6615) :
  t = 1 :=
by
  sorry

end investment_compound_half_yearly_l329_32925


namespace quadratic_always_positive_l329_32982

theorem quadratic_always_positive (x : ℝ) : x^2 + x + 1 > 0 :=
sorry

end quadratic_always_positive_l329_32982


namespace find_three_digit_number_l329_32951

def is_three_digit_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000

def digits_sum (n : ℕ) : ℕ :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  a + b + c

theorem find_three_digit_number : 
  ∃ n : ℕ, is_three_digit_number n ∧ n^2 = (digits_sum n)^5 ∧ n = 243 :=
sorry

end find_three_digit_number_l329_32951


namespace problem_statement_l329_32983

variable {x y : ℝ}

theorem problem_statement 
  (h1 : y > x)
  (h2 : x > 0)
  (h3 : x + y = 1) :
  x < 2 * x * y ∧ 2 * x * y < (x + y) / 2 ∧ (x + y) / 2 < y := by
  sorry

end problem_statement_l329_32983


namespace circle_properties_radius_properties_l329_32965

theorem circle_properties (m x y : ℝ) :
  (x^2 + y^2 - 2*(m + 3)*x + 2*(1 - 4*m^2)*y + 16*m^4 + 9 = 0) ↔
    (-((1 : ℝ) / (7 : ℝ)) < m ∧ m < (1 : ℝ)) :=
sorry

theorem radius_properties (m : ℝ) (h : -((1 : ℝ) / (7 : ℝ)) < m ∧ m < (1 : ℝ)) :
  ∃ r : ℝ, (0 < r ∧ r ≤ (4 / Real.sqrt 7)) :=
sorry

end circle_properties_radius_properties_l329_32965


namespace sum_six_smallest_multiples_of_eleven_l329_32953

theorem sum_six_smallest_multiples_of_eleven : 
  (11 + 22 + 33 + 44 + 55 + 66) = 231 :=
by
  sorry

end sum_six_smallest_multiples_of_eleven_l329_32953


namespace cos_neg_300_l329_32966

theorem cos_neg_300 : Real.cos (-(300 : ℝ) * Real.pi / 180) = 1 / 2 :=
by
  -- Proof goes here
  sorry

end cos_neg_300_l329_32966


namespace expression_evaluation_l329_32979

theorem expression_evaluation : 
  (1 : ℝ)^(6 * z - 3) / (7⁻¹ + 4⁻¹) = 28 / 11 :=
by
  sorry

end expression_evaluation_l329_32979


namespace vaccination_target_failure_l329_32932

noncomputable def percentage_vaccination_target_failed (original_target : ℕ) (first_year : ℕ) (second_year_increase_rate : ℚ) (third_year : ℕ) : ℚ :=
  let second_year := first_year + second_year_increase_rate * first_year
  let total_vaccinated := first_year + second_year + third_year
  let shortfall := original_target - total_vaccinated
  (shortfall / original_target) * 100

theorem vaccination_target_failure :
  percentage_vaccination_target_failed 720 60 (65/100 : ℚ) 150 = 57.11 := 
  by sorry

end vaccination_target_failure_l329_32932


namespace smallest_number_of_students_l329_32948

theorem smallest_number_of_students (n : ℕ) :
  (n % 3 = 2) ∧
  (n % 5 = 3) ∧
  (n % 8 = 5) →
  n = 53 :=
by
  intro h
  sorry

end smallest_number_of_students_l329_32948


namespace fundraiser_full_price_revenue_l329_32918

theorem fundraiser_full_price_revenue :
  ∃ (f h p : ℕ), f + h = 200 ∧ 
                f * p + h * (p / 2) = 2700 ∧ 
                f * p = 600 :=
by 
  sorry

end fundraiser_full_price_revenue_l329_32918


namespace fixed_fee_rental_l329_32930

theorem fixed_fee_rental (F C h : ℕ) (hC : C = F + 7 * h) (hC80 : C = 80) (hh9 : h = 9) : F = 17 :=
by
  sorry

end fixed_fee_rental_l329_32930


namespace Rockets_won_38_games_l329_32947

-- Definitions for each team and their respective wins
variables (Sharks Dolphins Rockets Wolves Comets : ℕ)
variables (wins : Finset ℕ)
variables (shArks_won_more_than_Dolphins : Sharks > Dolphins)
variables (rockets_won_more_than_Wolves : Rockets > Wolves)
variables (rockets_won_fewer_than_Comets : Rockets < Comets)
variables (Wolves_won_more_than_25_games : Wolves > 25)
variables (possible_wins : wins = {28, 33, 38, 43})

-- Statement that the Rockets won 38 games given the conditions
theorem Rockets_won_38_games
  (shArks_won_more_than_Dolphins : Sharks > Dolphins)
  (rockets_won_more_than_Wolves : Rockets > Wolves)
  (rockets_won_fewer_than_Comets : Rockets < Comets)
  (Wolves_won_more_than_25_games : Wolves > 25)
  (possible_wins : wins = {28, 33, 38, 43}) :
  Rockets = 38 :=
sorry

end Rockets_won_38_games_l329_32947


namespace new_weight_l329_32916

-- Conditions
def avg_weight_increase (n : ℕ) (avg_increase : ℝ) : ℝ := n * avg_increase
def weight_replacement (initial_weight : ℝ) (total_increase : ℝ) : ℝ := initial_weight + total_increase

-- Problem Statement: Proving the weight of the new person
theorem new_weight {n : ℕ} {avg_increase initial_weight W : ℝ} 
  (h_n : n = 8) (h_avg_increase : avg_increase = 2.5) (h_initial_weight : initial_weight = 65) (h_W : W = 85) :
  weight_replacement initial_weight (avg_weight_increase n avg_increase) = W :=
by 
  rw [h_n, h_avg_increase, h_initial_weight, h_W]
  sorry

end new_weight_l329_32916


namespace vitamin_d3_total_days_l329_32915

def vitamin_d3_days (capsules_per_bottle : ℕ) (daily_serving_size : ℕ) (bottles_needed : ℕ) : ℕ :=
  (capsules_per_bottle / daily_serving_size) * bottles_needed

theorem vitamin_d3_total_days :
  vitamin_d3_days 60 2 6 = 180 :=
by
  sorry

end vitamin_d3_total_days_l329_32915


namespace calc_factorial_sum_l329_32952

theorem calc_factorial_sum : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end calc_factorial_sum_l329_32952


namespace mechanic_worked_hours_l329_32956

theorem mechanic_worked_hours (total_spent : ℕ) (cost_per_part : ℕ) (labor_cost_per_minute : ℚ) (parts_needed : ℕ) :
  total_spent = 220 → cost_per_part = 20 → labor_cost_per_minute = 0.5 → parts_needed = 2 →
  (total_spent - cost_per_part * parts_needed) / labor_cost_per_minute / 60 = 6 := by
  -- Proof will be inserted here
  sorry

end mechanic_worked_hours_l329_32956


namespace calculation_A_correct_l329_32976

theorem calculation_A_correct : (-1: ℝ)^4 * (-1: ℝ)^3 = 1 := by
  sorry

end calculation_A_correct_l329_32976


namespace correct_operation_l329_32906

theorem correct_operation (a b : ℝ) : 
  (2 * a) * (3 * a) = 6 * a^2 :=
by
  -- The proof would be here; using "sorry" to skip the actual proof steps.
  sorry

end correct_operation_l329_32906


namespace paul_final_balance_l329_32996

def initial_balance : ℝ := 400
def transfer1 : ℝ := 90
def transfer2 : ℝ := 60
def service_charge_rate : ℝ := 0.02

def service_charge (x : ℝ) : ℝ := service_charge_rate * x

def total_deduction : ℝ := transfer1 + service_charge transfer1 + service_charge transfer2

def final_balance (init_balance : ℝ) (deduction : ℝ) : ℝ := init_balance - deduction

theorem paul_final_balance :
  final_balance initial_balance total_deduction = 307 :=
by
  sorry

end paul_final_balance_l329_32996


namespace minimum_guests_l329_32938

theorem minimum_guests (total_food : ℤ) (max_food_per_guest : ℤ) (food_bound : total_food = 325) (guest_bound : max_food_per_guest = 2) : (⌈total_food / max_food_per_guest⌉ : ℤ) = 163 :=
by {
  sorry 
}

end minimum_guests_l329_32938


namespace solve_quadratic_l329_32934

theorem solve_quadratic : ∀ (x : ℝ), x^2 - 5 * x + 1 = 0 →
  (x = (5 + Real.sqrt 21) / 2) ∨ (x = (5 - Real.sqrt 21) / 2) :=
by
  intro x
  intro h
  sorry

end solve_quadratic_l329_32934


namespace find_side_length_l329_32960

theorem find_side_length (a : ℝ) (b : ℝ) (A B : ℝ) (ha : a = 4) (hA : A = 45) (hB : B = 60) :
    b = 2 * Real.sqrt 6 := by
  sorry

end find_side_length_l329_32960


namespace lcm_210_297_l329_32977

theorem lcm_210_297 : Nat.lcm 210 297 = 20790 := 
by sorry

end lcm_210_297_l329_32977
