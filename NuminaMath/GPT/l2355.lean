import Mathlib

namespace find_d_plus_q_l2355_235569

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ := b₁ * q ^ (n - 1)

noncomputable def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * a₁ + d * (n * (n - 1) / 2)

noncomputable def sum_geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * b₁
  else b₁ * (q ^ n - 1) / (q - 1)

noncomputable def sum_combined_sequence (a₁ d b₁ q : ℝ) (n : ℕ) : ℝ :=
  sum_arithmetic_sequence a₁ d n + sum_geometric_sequence b₁ q n

theorem find_d_plus_q (a₁ d b₁ q : ℝ) (h_seq: ∀ n : ℕ, 0 < n → sum_combined_sequence a₁ d b₁ q n = n^2 - n + 2^n - 1) :
  d + q = 4 :=
  sorry

end find_d_plus_q_l2355_235569


namespace machine_A_time_to_produce_x_boxes_l2355_235564

-- Definitions of the conditions
def machine_A_rate (T : ℕ) (x : ℕ) : ℚ := x / T
def machine_B_rate (x : ℕ) : ℚ := 2 * x / 5
def combined_rate (T : ℕ) (x : ℕ) : ℚ := (x / 2) 

-- The theorem statement
theorem machine_A_time_to_produce_x_boxes (x : ℕ) : 
  ∀ T : ℕ, 20 * (machine_A_rate T x + machine_B_rate x) = 10 * x → T = 10 :=
by
  intros T h
  sorry

end machine_A_time_to_produce_x_boxes_l2355_235564


namespace work_rate_b_l2355_235503

theorem work_rate_b (A C B : ℝ) (hA : A = 1 / 8) (hC : C = 1 / 24) (h_combined : A + B + C = 1 / 4) : B = 1 / 12 :=
by
  -- Proof goes here
  sorry

end work_rate_b_l2355_235503


namespace find_A_plus_B_l2355_235522

def f (A B x : ℝ) : ℝ := A * x + B
def g (A B x : ℝ) : ℝ := B * x + A
def A_ne_B (A B : ℝ) : Prop := A ≠ B

theorem find_A_plus_B (A B x : ℝ) (h1 : A_ne_B A B)
  (h2 : (f A B (g A B x)) - (g A B (f A B x)) = 2 * (B - A)) : A + B = 3 :=
sorry

end find_A_plus_B_l2355_235522


namespace remove_terms_to_get_two_thirds_l2355_235549

noncomputable def sum_of_terms : ℚ := 
  (1/3) + (1/6) + (1/9) + (1/12) + (1/15) + (1/18)

noncomputable def sum_of_remaining_terms := 
  (1/3) + (1/6) + (1/9) + (1/18)

theorem remove_terms_to_get_two_thirds :
  sum_of_terms - (1/12 + 1/15) = (2/3) :=
by
  sorry

end remove_terms_to_get_two_thirds_l2355_235549


namespace finding_b_for_infinite_solutions_l2355_235538

theorem finding_b_for_infinite_solutions :
  ∀ b : ℝ, (∀ x : ℝ, 5 * (4 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 :=
by
  sorry

end finding_b_for_infinite_solutions_l2355_235538


namespace songs_can_be_stored_l2355_235582

def totalStorageGB : ℕ := 16
def usedStorageGB : ℕ := 4
def songSizeMB : ℕ := 30
def gbToMb : ℕ := 1000

def remainingStorageGB := totalStorageGB - usedStorageGB
def remainingStorageMB := remainingStorageGB * gbToMb
def numberOfSongs := remainingStorageMB / songSizeMB

theorem songs_can_be_stored : numberOfSongs = 400 :=
by
  sorry

end songs_can_be_stored_l2355_235582


namespace inscribed_circle_radius_l2355_235534

-- Conditions
variables {S A B C D O : Point} -- Points in 3D space
variables (AC : ℝ) (cos_SBD : ℝ)
variables (r : ℝ) -- Radius of inscribed circle

-- Given conditions
def AC_eq_one := AC = 1
def cos_angle_SBD := cos_SBD = 2/3

-- Assertion to be proved
theorem inscribed_circle_radius :
  AC_eq_one AC →
  cos_angle_SBD cos_SBD →
  (0 < r ∧ r ≤ 1/6) ∨ r = 1/3 :=
by
  intro hAC hcos
  -- Proof goes here
  sorry

end inscribed_circle_radius_l2355_235534


namespace flag_height_l2355_235527

-- Definitions based on conditions
def flag_width : ℝ := 5
def paint_cost_per_quart : ℝ := 2
def sqft_per_quart : ℝ := 4
def total_spent : ℝ := 20

-- The theorem to prove the height h of the flag
theorem flag_height (h : ℝ) (paint_needed : ℝ -> ℝ) :
  paint_needed h = 4 := sorry

end flag_height_l2355_235527


namespace gcd_12345_6789_l2355_235560

theorem gcd_12345_6789 : Int.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_l2355_235560


namespace price_comparison_l2355_235510

variable (x y : ℝ)
variable (h1 : 6 * x + 3 * y > 24)
variable (h2 : 4 * x + 5 * y < 22)

theorem price_comparison : 2 * x > 3 * y :=
sorry

end price_comparison_l2355_235510


namespace empty_drainpipe_rate_l2355_235528

theorem empty_drainpipe_rate :
  (∀ x : ℝ, (1/5 + 1/4 - 1/x = 1/2.5) → x = 20) :=
by 
    intro x
    intro h
    sorry -- Proof is omitted, only the statement is required

end empty_drainpipe_rate_l2355_235528


namespace carl_watermelons_left_l2355_235568

-- Define the conditions
def price_per_watermelon : ℕ := 3
def profit : ℕ := 105
def starting_watermelons : ℕ := 53

-- Define the main proof statement
theorem carl_watermelons_left :
  (starting_watermelons - (profit / price_per_watermelon) = 18) :=
sorry

end carl_watermelons_left_l2355_235568


namespace mika_stickers_l2355_235507

theorem mika_stickers 
    (initial_stickers : ℝ := 20.5)
    (bought_stickers : ℝ := 26.25)
    (birthday_stickers : ℝ := 19.75)
    (friend_stickers : ℝ := 7.5)
    (sister_stickers : ℝ := 6.3)
    (greeting_card_stickers : ℝ := 58.5)
    (yard_sale_stickers : ℝ := 3.2) :
    initial_stickers + bought_stickers + birthday_stickers + friend_stickers
    - sister_stickers - greeting_card_stickers - yard_sale_stickers = 6 := 
by
    sorry

end mika_stickers_l2355_235507


namespace sandy_spent_money_l2355_235552

theorem sandy_spent_money :
  let shorts := 13.99
  let shirt := 12.14
  let jacket := 7.43
  shorts + shirt + jacket = 33.56 :=
by
  let shorts := 13.99
  let shirt := 12.14
  let jacket := 7.43
  have total_spent : shorts + shirt + jacket = 33.56 := sorry
  exact total_spent

end sandy_spent_money_l2355_235552


namespace bus_speed_l2355_235578

def distance : ℝ := 350.028
def time : ℝ := 10
def speed_kmph : ℝ := 126.01

theorem bus_speed :
  (distance / time) * 3.6 = speed_kmph := 
sorry

end bus_speed_l2355_235578


namespace tape_for_small_box_l2355_235579

theorem tape_for_small_box (S : ℝ) :
  (2 * 4) + (8 * 2) + (5 * S) + (2 + 8 + 5) = 44 → S = 1 :=
by
  intro h
  sorry

end tape_for_small_box_l2355_235579


namespace average_reading_time_correct_l2355_235523

-- We define total_reading_time as a parameter representing the sum of reading times
noncomputable def total_reading_time : ℝ := sorry

-- We define the number of students as a constant
def number_of_students : ℕ := 50

-- We define the average reading time per student based on the provided data
noncomputable def average_reading_time : ℝ :=
  total_reading_time / number_of_students

-- The theorem we need to prove: that the average reading time per student is correctly calculated
theorem average_reading_time_correct :
  ∃ (total_reading_time : ℝ), average_reading_time = total_reading_time / number_of_students :=
by
  -- since total_reading_time and number_of_students are already defined, we prove the theorem using them
  use total_reading_time
  exact rfl

end average_reading_time_correct_l2355_235523


namespace last_three_digits_of_expression_l2355_235589

theorem last_three_digits_of_expression : 
  let prod := 301 * 402 * 503 * 604 * 646 * 547 * 448 * 349
  (prod ^ 3) % 1000 = 976 :=
by
  sorry

end last_three_digits_of_expression_l2355_235589


namespace Alice_has_3_more_dimes_than_quarters_l2355_235599

-- Definitions of the conditions given in the problem
variable (n d : ℕ) -- number of 5-cent and 10-cent coins
def q : ℕ := 10
def total_coins : ℕ := 30
def total_value : ℕ := 435
def extra_dimes : ℕ := 6

-- Conditions translated to Lean
axiom total_coin_count : n + d + q = total_coins
axiom total_value_count : 5 * n + 10 * d + 25 * q = total_value
axiom dime_difference : d = n + extra_dimes

-- The theorem that needs to be proven: Alice has 3 more 10-cent coins than 25-cent coins.
theorem Alice_has_3_more_dimes_than_quarters :
  d - q = 3 :=
sorry

end Alice_has_3_more_dimes_than_quarters_l2355_235599


namespace octagon_mass_l2355_235573

theorem octagon_mass :
  let side_length := 1 -- side length of the original square (meters)
  let thickness := 0.3 -- thickness of the sheet (cm)
  let density := 7.8 -- density of steel (g/cm^3)
  let x := 50 * (2 - Real.sqrt 2) -- side length of the triangles (cm)
  let octagon_area := 20000 * (Real.sqrt 2 - 1) -- area of the octagon (cm^2)
  let volume := octagon_area * thickness -- volume of the octagon (cm^3)
  let mass := volume * density / 1000 -- mass of the octagon (kg), converted from g to kg
  mass = 19 :=
by
  sorry

end octagon_mass_l2355_235573


namespace carter_baseball_cards_l2355_235540

theorem carter_baseball_cards (M C : ℕ) (h1 : M = 210) (h2 : M = C + 58) : C = 152 :=
by
  sorry

end carter_baseball_cards_l2355_235540


namespace factorization_correct_l2355_235548

theorem factorization_correct (x : ℝ) :
  (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 12 * x^4 + 3) = 12 * (x^6 + 4 * x^4 - 1) := by
  sorry

end factorization_correct_l2355_235548


namespace min_value_of_quadratic_l2355_235587

def quadratic_function (x : ℝ) : ℝ := x^2 + 6 * x + 13

theorem min_value_of_quadratic :
  (∃ x : ℝ, quadratic_function x = 4) ∧ (∀ y : ℝ, quadratic_function y ≥ 4) :=
sorry

end min_value_of_quadratic_l2355_235587


namespace volume_ratio_l2355_235585

theorem volume_ratio (A B C : ℝ) 
  (h1 : A = (B + C) / 4)
  (h2 : B = (C + A) / 6) : 
  C / (A + B) = 23 / 12 :=
sorry

end volume_ratio_l2355_235585


namespace range_of_m_for_log_function_domain_l2355_235561

theorem range_of_m_for_log_function_domain (m : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 8 * x + m > 0) → m > 8 :=
by
  sorry

end range_of_m_for_log_function_domain_l2355_235561


namespace angle_difference_l2355_235554

theorem angle_difference (X Y Z Z1 Z2 : ℝ) (h1 : Y = 2 * X) (h2 : X = 30) (h3 : Z1 + Z2 = Z) (h4 : Z1 = 60) (h5 : Z2 = 30) : Z1 - Z2 = 30 := 
by 
  sorry

end angle_difference_l2355_235554


namespace central_angle_nonagon_l2355_235521

theorem central_angle_nonagon : (360 / 9 = 40) :=
by
  sorry

end central_angle_nonagon_l2355_235521


namespace complement_A_in_U_l2355_235586

/-- Problem conditions -/
def is_universal_set (x : ℕ) : Prop := (x - 6) * (x + 1) ≤ 0
def A : Set ℕ := {1, 2, 4}
def U : Set ℕ := { x | is_universal_set x }

/-- Proof statement -/
theorem complement_A_in_U : (U \ A) = {3, 5, 6} :=
by
  sorry  -- replacement for the proof

end complement_A_in_U_l2355_235586


namespace determineHairColors_l2355_235563

structure Person where
  name : String
  hairColor : String

def Belokurov : Person := { name := "Belokurov", hairColor := "" }
def Chernov : Person := { name := "Chernov", hairColor := "" }
def Ryzhev : Person := { name := "Ryzhev", hairColor := "" }

-- Define the possible hair colors
def Blonde : String := "Blonde"
def Brunette : String := "Brunette"
def RedHaired : String := "Red-Haired"

-- Define the conditions based on the problem statement
axiom hairColorConditions :
  Belokurov.hairColor ≠ Blonde ∧
  Belokurov.hairColor ≠ Brunette ∧
  Chernov.hairColor ≠ Brunette ∧
  Chernov.hairColor ≠ RedHaired ∧
  Ryzhev.hairColor ≠ RedHaired ∧
  Ryzhev.hairColor ≠ Blonde ∧
  ∀ p : Person, p.hairColor = Brunette → p.name ≠ "Belokurov"

-- Define the uniqueness condition that each person has a different hair color
axiom uniqueHairColors :
  Belokurov.hairColor ≠ Chernov.hairColor ∧
  Belokurov.hairColor ≠ Ryzhev.hairColor ∧
  Chernov.hairColor ≠ Ryzhev.hairColor

-- Define the proof problem
theorem determineHairColors :
  Belokurov.hairColor = RedHaired ∧
  Chernov.hairColor = Blonde ∧
  Ryzhev.hairColor = Brunette := by
  sorry

end determineHairColors_l2355_235563


namespace expand_polynomial_l2355_235524

theorem expand_polynomial (x : ℝ) :
  (x + 4) * (x - 9) = x^2 - 5 * x - 36 := 
sorry

end expand_polynomial_l2355_235524


namespace price_decrease_necessary_l2355_235511

noncomputable def final_price_decrease (P : ℝ) (x : ℝ) : Prop :=
  let increased_price := 1.2 * P
  let final_price := increased_price * (1 - x / 100)
  final_price = 0.88 * P

theorem price_decrease_necessary (x : ℝ) : 
  final_price_decrease 100 x -> x = 26.67 :=
by 
  intros h
  unfold final_price_decrease at h
  sorry

end price_decrease_necessary_l2355_235511


namespace workouts_difference_l2355_235513

theorem workouts_difference
  (workouts_monday : ℕ := 8)
  (workouts_tuesday : ℕ := 5)
  (workouts_wednesday : ℕ := 12)
  (workouts_thursday : ℕ := 17)
  (workouts_friday : ℕ := 10) :
  workouts_thursday - workouts_tuesday = 12 := 
by
  sorry

end workouts_difference_l2355_235513


namespace solve_AlyoshaCube_l2355_235535

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end solve_AlyoshaCube_l2355_235535


namespace sum_of_mapped_elements_is_ten_l2355_235536

theorem sum_of_mapped_elements_is_ten (a b : ℝ) (h1 : a = 1) (h2 : b = 9) : a + b = 10 := by
  sorry

end sum_of_mapped_elements_is_ten_l2355_235536


namespace addition_example_l2355_235598

theorem addition_example : 248 + 64 = 312 := by
  sorry

end addition_example_l2355_235598


namespace slope_problem_l2355_235572

theorem slope_problem (m : ℝ) (h₀ : m > 0) (h₁ : (3 - m) = m * (1 - m)) : m = Real.sqrt 3 := by
  sorry

end slope_problem_l2355_235572


namespace cos_C_value_l2355_235593

theorem cos_C_value (a b c : ℝ) (A B C : ℝ) (h1 : 8 * b = 5 * c) (h2 : C = 2 * B) : 
  Real.cos C = 7 / 25 :=
  sorry

end cos_C_value_l2355_235593


namespace standard_deviation_of_applicants_ages_l2355_235576

noncomputable def average_age : ℝ := 30
noncomputable def max_different_ages : ℝ := 15

theorem standard_deviation_of_applicants_ages 
  (σ : ℝ)
  (h : max_different_ages = 2 * σ) 
  : σ = 7.5 :=
by
  sorry

end standard_deviation_of_applicants_ages_l2355_235576


namespace series_sum_eq_l2355_235505

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l2355_235505


namespace vector_sum_is_correct_l2355_235583

-- Definitions for vectors a and b
def vector_a := (1, -2)
def vector_b (m : ℝ) := (2, m)

-- Condition for parallel vectors a and b
def parallel_vectors (m : ℝ) : Prop :=
  1 * m - (-2) * 2 = 0

-- Defining the target calculation for given m
def calculate_sum (m : ℝ) : ℝ × ℝ :=
  let a := vector_a
  let b := vector_b m
  (3 * a.1 + 2 * b.1, 3 * a.2 + 2 * b.2)

-- Statement of the theorem to be proved
theorem vector_sum_is_correct (m : ℝ) (h : parallel_vectors m) : calculate_sum m = (7, -14) :=
by sorry

end vector_sum_is_correct_l2355_235583


namespace bus_stops_per_hour_l2355_235575

-- Define the speeds as constants
def speed_excluding_stoppages : ℝ := 60
def speed_including_stoppages : ℝ := 50

-- Formulate the main theorem
theorem bus_stops_per_hour :
  (1 - speed_including_stoppages / speed_excluding_stoppages) * 60 = 10 := 
by
  sorry

end bus_stops_per_hour_l2355_235575


namespace sales_decrease_percentage_l2355_235577

theorem sales_decrease_percentage 
  (P S : ℝ) 
  (P_new : ℝ := 1.30 * P) 
  (R : ℝ := P * S) 
  (R_new : ℝ := 1.04 * R) 
  (x : ℝ) 
  (S_new : ℝ := S * (1 - x/100)) 
  (h1 : 1.30 * P * S * (1 - x/100) = 1.04 * P * S) : 
  x = 20 :=
by
  sorry

end sales_decrease_percentage_l2355_235577


namespace alpha_beta_square_l2355_235504

theorem alpha_beta_square (α β : ℝ) (h₁ : α^2 = 2*α + 1) (h₂ : β^2 = 2*β + 1) (hαβ : α ≠ β) :
  (α - β)^2 = 8 := 
sorry

end alpha_beta_square_l2355_235504


namespace max_profit_achieved_l2355_235581

theorem max_profit_achieved :
  ∃ x : ℤ, 
    (x = 21) ∧ 
    (21 + 14 = 35) ∧ 
    (30 - 21 = 9) ∧ 
    (21 - 5 = 16) ∧
    (-x + 1965 = 1944) :=
by
  sorry

end max_profit_achieved_l2355_235581


namespace possible_values_quotient_l2355_235515

theorem possible_values_quotient (α : ℝ) (h_pos : α > 0) (h_rounded : ∃ (n : ℕ) (α1 : ℝ), α = n / 100 + α1 ∧ 0 ≤ α1 ∧ α1 < 1 / 100) :
  ∃ (values : List ℝ), values = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
                                  0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
                                  0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
                                  0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
                                  0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
                                  1.00] :=
  sorry

end possible_values_quotient_l2355_235515


namespace milk_left_is_correct_l2355_235567

def total_morning_milk : ℕ := 365
def total_evening_milk : ℕ := 380
def milk_sold : ℕ := 612
def leftover_milk_from_yesterday : ℕ := 15

def total_milk_left : ℕ :=
  (total_morning_milk + total_evening_milk - milk_sold) + leftover_milk_from_yesterday

theorem milk_left_is_correct : total_milk_left = 148 := by
  sorry

end milk_left_is_correct_l2355_235567


namespace gina_good_tipper_l2355_235562

noncomputable def calculate_tip_difference (bill_in_usd : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) (low_tip_rate : ℝ) (high_tip_rate : ℝ) (conversion_rate : ℝ) : ℝ :=
  let discounted_bill := bill_in_usd * (1 - discount_rate)
  let taxed_bill := discounted_bill * (1 + tax_rate)
  let low_tip := taxed_bill * low_tip_rate
  let high_tip := taxed_bill * high_tip_rate
  let difference_in_usd := high_tip - low_tip
  let difference_in_eur := difference_in_usd * conversion_rate
  difference_in_eur * 100

theorem gina_good_tipper : calculate_tip_difference 26 0.08 0.07 0.05 0.20 0.85 = 326.33 := 
by
  sorry

end gina_good_tipper_l2355_235562


namespace find_x_l2355_235545

theorem find_x (p : ℕ) (hprime : Nat.Prime p) (hgt5 : p > 5) (x : ℕ) (hx : x ≠ 0) :
    (∀ n : ℕ, 0 < n → (5 * p + x) ∣ (5 * p ^ n + x ^ n)) ↔ x = p := by
  sorry

end find_x_l2355_235545


namespace boys_and_girls_are_equal_l2355_235592

theorem boys_and_girls_are_equal (B G : ℕ) (h1 : B + G = 30)
    (h2 : ∀ b₁ b₂, b₁ ≠ b₂ → (0 ≤ b₁) ∧ (b₁ ≤ G - 1) → (0 ≤ b₂) ∧ (b₂ ≤ G - 1) → b₁ ≠ b₂)
    (h3 : ∀ g₁ g₂, g₁ ≠ g₂ → (0 ≤ g₁) ∧ (g₁ ≤ B - 1) → (0 ≤ g₂) ∧ (g₂ ≤ B - 1) → g₁ ≠ g₂) : 
    B = 15 ∧ G = 15 := by
  sorry

end boys_and_girls_are_equal_l2355_235592


namespace expr1_simplified_expr2_simplified_l2355_235574

variable (a x : ℝ)

theorem expr1_simplified : (-a^3 + (-4 * a^2) * a) = -5 * a^3 := 
by
  sorry

theorem expr2_simplified : (-x^2 * (-x)^2 * (-x^2)^3 - 2 * x^10) = -x^10 := 
by
  sorry

end expr1_simplified_expr2_simplified_l2355_235574


namespace carmela_gives_each_l2355_235542

noncomputable def money_needed_to_give_each (carmela : ℕ) (cousins : ℕ) (cousins_count : ℕ) : ℕ :=
  let total_cousins_money := cousins * cousins_count
  let total_money := carmela + total_cousins_money
  let people_count := 1 + cousins_count
  let equal_share := total_money / people_count
  let total_giveaway := carmela - equal_share
  total_giveaway / cousins_count

theorem carmela_gives_each (carmela : ℕ) (cousins : ℕ) (cousins_count : ℕ) (h_carmela : carmela = 7) (h_cousins : cousins = 2) (h_cousins_count : cousins_count = 4) :
  money_needed_to_give_each carmela cousins cousins_count = 1 :=
by
  rw [h_carmela, h_cousins, h_cousins_count]
  sorry

end carmela_gives_each_l2355_235542


namespace sequence_an_correct_l2355_235594

theorem sequence_an_correct (S_n : ℕ → ℕ) (a : ℕ → ℕ) (h1 : ∀ n, S_n n = n^2 + 1) :
  (a 1 = 2) ∧ (∀ n, n ≥ 2 → a n = 2 * n - 1) :=
by
  -- We assume S_n is defined such that S_n = n^2 + 1
  -- From this, we have to show that:
  -- for n = 1, a_1 = 2,
  -- and for n ≥ 2, a_n = 2n - 1
  sorry

end sequence_an_correct_l2355_235594


namespace sin_2alpha_value_l2355_235509

noncomputable def sin_2alpha_through_point (x y : ℝ) : ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let sin_alpha := y / r
  let cos_alpha := x / r
  2 * sin_alpha * cos_alpha

theorem sin_2alpha_value :
  sin_2alpha_through_point (-3) 4 = -24 / 25 :=
by
  sorry

end sin_2alpha_value_l2355_235509


namespace find_length_of_first_train_l2355_235557

noncomputable def length_of_first_train (speed_train1 speed_train2 : ℕ) (time_to_cross : ℕ) (length_train2 : ℚ) : ℚ :=
  let relative_speed := (speed_train1 + speed_train2) * 1000 / 3600
  let combined_length := relative_speed * time_to_cross
  combined_length - length_train2

theorem find_length_of_first_train :
  length_of_first_train 120 80 9 280.04 = 220 := sorry

end find_length_of_first_train_l2355_235557


namespace max_k_for_ineq_l2355_235591

theorem max_k_for_ineq (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : m^3 + n^3 > (m + n)^2) :
  m^3 + n^3 ≥ (m + n)^2 + 10 :=
sorry

end max_k_for_ineq_l2355_235591


namespace number_wall_top_block_value_l2355_235533

theorem number_wall_top_block_value (a b c d : ℕ) 
    (h1 : a = 8) (h2 : b = 5) (h3 : c = 3) (h4 : d = 2) : 
    (a + b + (b + c) + (c + d) = 34) :=
by
  sorry

end number_wall_top_block_value_l2355_235533


namespace max_abc_l2355_235546

def A_n (a : ℕ) (n : ℕ) : ℕ := a * (10^(3*n) - 1) / 9
def B_n (b : ℕ) (n : ℕ) : ℕ := b * (10^(2*n) - 1) / 9
def C_n (c : ℕ) (n : ℕ) : ℕ := c * (10^(2*n) - 1) / 9

theorem max_abc (a b c n : ℕ) (hpos : n > 0) (h1 : 1 ≤ a ∧ a < 10) (h2 : 1 ≤ b ∧ b < 10) (h3 : 1 ≤ c ∧ c < 10) (h_eq : C_n c n - B_n b n = A_n a n ^ 2) :  a + b + c ≤ 18 :=
by sorry

end max_abc_l2355_235546


namespace rocket_max_speed_l2355_235529

theorem rocket_max_speed (M m : ℝ) (h : 2000 * Real.log (1 + M / m) = 12000) : 
  M / m = Real.exp 6 - 1 := 
by {
  sorry
}

end rocket_max_speed_l2355_235529


namespace find_f_value_l2355_235531

noncomputable def f (a b x : ℝ) : ℝ := a * (Real.cos x)^2 - b * (Real.sin x) * (Real.cos x) - a / 2

theorem find_f_value (a b : ℝ)
  (h_max : ∀ x, f a b x ≤ 1/2)
  (h_at_pi_over_3 : f a b (Real.pi / 3) = (Real.sqrt 3) / 4) :
  f a b (-Real.pi / 3) = 0 ∨ f a b (-Real.pi / 3) = -(Real.sqrt 3) / 4 :=
sorry

end find_f_value_l2355_235531


namespace angle_turned_by_hour_hand_l2355_235512

theorem angle_turned_by_hour_hand (rotation_degrees_per_hour : ℝ) (total_degrees_per_rotation : ℝ) :
  rotation_degrees_per_hour * 1 = -30 :=
by
  have rotation_degrees_per_hour := - total_degrees_per_rotation / 12
  have total_degrees_per_rotation := 360
  sorry

end angle_turned_by_hour_hand_l2355_235512


namespace total_students_in_class_l2355_235595

theorem total_students_in_class (front_pos back_pos : ℕ) (H_front : front_pos = 23) (H_back : back_pos = 23) : front_pos + back_pos - 1 = 45 :=
by
  -- No proof required as per instructions
  sorry

end total_students_in_class_l2355_235595


namespace ratio_of_eggs_used_l2355_235551

theorem ratio_of_eggs_used (total_eggs : ℕ) (eggs_left : ℕ) (eggs_broken : ℕ) (eggs_bought : ℕ) :
  total_eggs = 72 →
  eggs_left = 21 →
  eggs_broken = 15 →
  eggs_bought = total_eggs - (eggs_left + eggs_broken) →
  (eggs_bought / total_eggs) = 1 / 2 :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_of_eggs_used_l2355_235551


namespace minimalBananasTotal_is_408_l2355_235501

noncomputable def minimalBananasTotal : ℕ :=
  let b₁ := 11 * 8
  let b₂ := 13 * 8
  let b₃ := 27 * 8
  b₁ + b₂ + b₃

theorem minimalBananasTotal_is_408 : minimalBananasTotal = 408 := by
  sorry

end minimalBananasTotal_is_408_l2355_235501


namespace number_of_benches_l2355_235518

-- Define the conditions
def bench_capacity : ℕ := 4
def people_sitting : ℕ := 80
def available_spaces : ℕ := 120
def total_capacity : ℕ := people_sitting + available_spaces -- this equals 200

-- The theorem to prove the number of benches
theorem number_of_benches (B : ℕ) : bench_capacity * B = total_capacity → B = 50 :=
by
  intro h
  exact sorry

end number_of_benches_l2355_235518


namespace average_total_goals_l2355_235550

theorem average_total_goals (carter_avg shelby_avg judah_avg total_avg : ℕ) 
    (h1: carter_avg = 4) 
    (h2: shelby_avg = carter_avg / 2)
    (h3: judah_avg = 2 * shelby_avg - 3) 
    (h4: total_avg = carter_avg + shelby_avg + judah_avg) :
  total_avg = 7 :=
by
  sorry

end average_total_goals_l2355_235550


namespace same_function_C_l2355_235539

theorem same_function_C (x : ℝ) (hx : x ≠ 0) : (x^0 = 1) ∧ ((1 / x^0) = 1) :=
by
  -- Definition for domain exclusion
  have h1 : x ^ 0 = 1 := by 
    sorry -- proof skipped
  have h2 : 1 / x ^ 0 = 1 := by 
    sorry -- proof skipped
  exact ⟨h1, h2⟩

end same_function_C_l2355_235539


namespace second_job_hourly_wage_l2355_235590

-- Definitions based on conditions
def total_wages : ℕ := 160
def first_job_wages : ℕ := 52
def second_job_hours : ℕ := 12

-- Proof statement
theorem second_job_hourly_wage : 
  (total_wages - first_job_wages) / second_job_hours = 9 :=
by
  sorry

end second_job_hourly_wage_l2355_235590


namespace ratio_of_w_to_y_l2355_235555

theorem ratio_of_w_to_y
  (w x y z : ℚ)
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 6) :
  w / y = 16 / 3 :=
by sorry

end ratio_of_w_to_y_l2355_235555


namespace positive_difference_is_127_div_8_l2355_235544

-- Defining the basic expressions
def eight_squared : ℕ := 8 ^ 2 -- 64

noncomputable def expr1 : ℝ := (eight_squared + eight_squared) / 8
noncomputable def expr2 : ℝ := (eight_squared / eight_squared) / 8

-- Problem statement
theorem positive_difference_is_127_div_8 :
  (expr1 - expr2) = 127 / 8 :=
by
  sorry

end positive_difference_is_127_div_8_l2355_235544


namespace tom_needs_44000_pounds_salt_l2355_235541

theorem tom_needs_44000_pounds_salt 
  (flour_needed : ℕ)
  (flour_bag_weight : ℕ)
  (flour_bag_cost : ℕ)
  (salt_cost_per_pound : ℝ)
  (promotion_cost : ℕ)
  (ticket_price : ℕ)
  (tickets_sold : ℕ)
  (total_revenue : ℕ) 
  (expected_salt_cost : ℝ) 
  (S : ℝ) : 
  flour_needed = 500 → 
  flour_bag_weight = 50 → 
  flour_bag_cost = 20 → 
  salt_cost_per_pound = 0.2 → 
  promotion_cost = 1000 → 
  ticket_price = 20 → 
  tickets_sold = 500 → 
  total_revenue = 8798 → 
  0.2 * S = (500 * 20) - (500 / 50) * 20 - 1000 →
  S = 44000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end tom_needs_44000_pounds_salt_l2355_235541


namespace inequality_proof_l2355_235543

theorem inequality_proof 
  (x y z : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0)
  (hxz : x * z = 1) 
  (h₁ : x * (1 + z) > 1) 
  (h₂ : y * (1 + x) > 1) 
  (h₃ : z * (1 + y) > 1) :
  2 * (x + y + z) ≥ -1/x + 1/y + 1/z + 3 :=
sorry

end inequality_proof_l2355_235543


namespace share_of_B_l2355_235553

theorem share_of_B (x : ℕ) (A B C : ℕ) (h1 : A = 3 * B) (h2 : B = C + 25)
  (h3 : A + B + C = 645) : B = 134 :=
by
  sorry

end share_of_B_l2355_235553


namespace hannah_total_savings_l2355_235556

theorem hannah_total_savings :
  let a1 := 4
  let a2 := 2 * a1
  let a3 := 2 * a2
  let a4 := 2 * a3
  let a5 := 20
  a1 + a2 + a3 + a4 + a5 = 80 :=
by
  sorry

end hannah_total_savings_l2355_235556


namespace math_expression_identity_l2355_235580

theorem math_expression_identity :
  |2 - Real.sqrt 3| - (2022 - Real.pi)^0 + Real.sqrt 12 = 1 + Real.sqrt 3 :=
by
  sorry

end math_expression_identity_l2355_235580


namespace range_a_l2355_235517

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then Real.log x / Real.log a else -2 * x + 8

theorem range_a (a : ℝ) (hf : ∀ x, f a x ≤ f a 2) :
  1 < a ∧ a ≤ Real.sqrt 3 := by
  sorry

end range_a_l2355_235517


namespace f_nested_seven_l2355_235597

-- Definitions for the given conditions
variables (f : ℝ → ℝ) (odd_f : ∀ x, f (-x) = -f x)
variables (period_f : ∀ x, f (x + 4) = f x)
variables (f_one : f 1 = 4)

theorem f_nested_seven (f : ℝ → ℝ)
  (odd_f : ∀ x, f (-x) = -f x)
  (period_f : ∀ x, f (x + 4) = f x)
  (f_one : f 1 = 4) :
  f (f 7) = 0 :=
sorry

end f_nested_seven_l2355_235597


namespace a_square_plus_one_over_a_square_l2355_235508

theorem a_square_plus_one_over_a_square (a : ℝ) (h : a + 1/a = 5) : a^2 + 1/a^2 = 23 :=
by 
  sorry

end a_square_plus_one_over_a_square_l2355_235508


namespace problem_statement_l2355_235520

noncomputable def p (k : ℝ) (x : ℝ) : ℝ := k * x
noncomputable def q (x : ℝ) : ℝ := (x + 4) * (x - 1)

theorem problem_statement (k : ℝ) (h_p_linear : ∀ x, p k x = k * x) 
    (h_q_quadratic : ∀ x, q x = (x + 4) * (x - 1)) 
    (h_pass_origin : p k 0 / q 0 = 0)
    (h_pass_point : p k 2 / q 2 = -1) :
    p k 1 / q 1 = -3 / 5 :=
sorry

end problem_statement_l2355_235520


namespace earned_points_l2355_235500

def points_per_enemy := 3
def total_enemies := 6
def enemies_undefeated := 2
def enemies_defeated := total_enemies - enemies_undefeated

theorem earned_points : enemies_defeated * points_per_enemy = 12 :=
by sorry

end earned_points_l2355_235500


namespace relationship_between_abc_l2355_235566

noncomputable def a : ℝ := Real.exp 0.9 + 1
def b : ℝ := 2.9
noncomputable def c : ℝ := Real.log (0.9 * Real.exp 3)

theorem relationship_between_abc : a > b ∧ b > c :=
by {
  sorry
}

end relationship_between_abc_l2355_235566


namespace winning_probability_correct_l2355_235530

-- Define the conditions
def numPowerBalls : ℕ := 30
def numLuckyBalls : ℕ := 49
def numChosenBalls : ℕ := 6

-- Define the probability of picking the correct PowerBall
def powerBallProb : ℚ := 1 / numPowerBalls

-- Define the combination function for choosing LuckyBalls
noncomputable def combination (n k : ℕ) : ℕ := n.choose k

-- Define the probability of picking the correct LuckyBalls
noncomputable def luckyBallProb : ℚ := 1 / (combination numLuckyBalls numChosenBalls)

-- Define the total winning probability
noncomputable def totalWinningProb : ℚ := powerBallProb * luckyBallProb

-- State the theorem to prove
theorem winning_probability_correct : totalWinningProb = 1 / 419512480 :=
by
  sorry

end winning_probability_correct_l2355_235530


namespace problem_l2355_235526

theorem problem (triangle square : ℕ) (h1 : triangle + 5 ≡ 1 [MOD 7]) (h2 : 2 + square ≡ 3 [MOD 7]) :
  triangle = 3 ∧ square = 1 := by
  sorry

end problem_l2355_235526


namespace range_of_m_l2355_235584

variable {x m : ℝ}
variable (q: ℝ → Prop) (p: ℝ → Prop)

-- Definition of q
def q_cond : Prop := (x - (1 + m)) * (x - (1 - m)) ≤ 0

-- Definition of p
def p_cond : Prop := |1 - (x - 1) / 3| ≤ 2

-- Statement of the proof problem
theorem range_of_m (h1 : ∀ x, q x → p x) (h2 : ∃ x, ¬p x → q x) 
  (h3 : m > 0) :
  0 < m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l2355_235584


namespace fourth_square_state_l2355_235547

inductive Shape
| Circle
| Triangle
| LineSegment
| Square

inductive Position
| TopLeft
| TopRight
| BottomLeft
| BottomRight

structure SquareState where
  circle : Position
  triangle : Position
  line_segment_parallel_to : Bool -- True = Top & Bottom; False = Left & Right
  square : Position

def move_counterclockwise : Position → Position
| Position.TopLeft => Position.BottomLeft
| Position.BottomLeft => Position.BottomRight
| Position.BottomRight => Position.TopRight
| Position.TopRight => Position.TopLeft

def update_square_states (s1 s2 s3 : SquareState) : Prop :=
  move_counterclockwise s1.circle = s2.circle ∧
  move_counterclockwise s2.circle = s3.circle ∧
  move_counterclockwise s1.triangle = s2.triangle ∧
  move_counterclockwise s2.triangle = s3.triangle ∧
  s1.line_segment_parallel_to = !s2.line_segment_parallel_to ∧
  s2.line_segment_parallel_to = !s3.line_segment_parallel_to ∧
  move_counterclockwise s1.square = s2.square ∧
  move_counterclockwise s2.square = s3.square

theorem fourth_square_state (s1 s2 s3 s4 : SquareState) (h : update_square_states s1 s2 s3) :
  s4.circle = move_counterclockwise s3.circle ∧
  s4.triangle = move_counterclockwise s3.triangle ∧
  s4.line_segment_parallel_to = !s3.line_segment_parallel_to ∧
  s4.square = move_counterclockwise s3.square :=
sorry

end fourth_square_state_l2355_235547


namespace division_sum_l2355_235558

theorem division_sum (quotient divisor remainder : ℕ) (hquot : quotient = 65) (hdiv : divisor = 24) (hrem : remainder = 5) : 
  (divisor * quotient + remainder) = 1565 := by 
  sorry

end division_sum_l2355_235558


namespace wedge_volume_cylinder_l2355_235502

theorem wedge_volume_cylinder (r h : ℝ) (theta : ℝ) (V : ℝ) 
  (hr : r = 6) (hh : h = 6) (htheta : theta = 60) (hV : V = 113) : 
  V = (theta / 360) * π * r^2 * h :=
by
  sorry

end wedge_volume_cylinder_l2355_235502


namespace smallest_w_l2355_235525

theorem smallest_w (w : ℕ) (w_pos : 0 < w) : 
  (∀ n : ℕ, (2^5 ∣ 936 * n) ∧ (3^3 ∣ 936 * n) ∧ (11^2 ∣ 936 * n) ↔ n = w) → w = 4356 :=
sorry

end smallest_w_l2355_235525


namespace volume_of_largest_sphere_from_cube_l2355_235596

theorem volume_of_largest_sphere_from_cube : 
  (∃ (V : ℝ), 
    (∀ (l : ℝ), l = 1 → (V = (4 / 3) * π * ((l / 2)^3)) → V = π / 6)) :=
sorry

end volume_of_largest_sphere_from_cube_l2355_235596


namespace minimum_a_l2355_235532

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (2 - x)

theorem minimum_a
  (a : ℝ)
  (h : ∀ x : ℤ, (f x)^2 - a * f x ≤ 0 → ∃! x : ℤ, (f x)^2 - a * f x = 0) :
  a = Real.exp 2 + 1 :=
sorry

end minimum_a_l2355_235532


namespace min_value_of_a_plus_b_l2355_235570

theorem min_value_of_a_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (1 / (a + 1)) + (2 / (1 + b)) = 1) : 
  a + b ≥ 2 * Real.sqrt 2 + 1 :=
sorry

end min_value_of_a_plus_b_l2355_235570


namespace maximum_value_l2355_235516

theorem maximum_value (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a^2 + b^2 + c^2 = 1) :
  3 * a * b * Real.sqrt 2 + 6 * b * c ≤ 4.5 :=
sorry

end maximum_value_l2355_235516


namespace Liza_reads_more_pages_than_Suzie_l2355_235519

def Liza_reading_speed : ℕ := 20
def Suzie_reading_speed : ℕ := 15
def hours : ℕ := 3

theorem Liza_reads_more_pages_than_Suzie :
  Liza_reading_speed * hours - Suzie_reading_speed * hours = 15 := by
  sorry

end Liza_reads_more_pages_than_Suzie_l2355_235519


namespace matrix_pow_2020_l2355_235537

-- Define the matrix type and basic multiplication rule
def M : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 0], ![3, 1]]

theorem matrix_pow_2020 :
  M ^ 2020 = ![![1, 0], ![6060, 1]] := by
  sorry

end matrix_pow_2020_l2355_235537


namespace probability_of_spinner_stopping_in_region_G_l2355_235559

theorem probability_of_spinner_stopping_in_region_G :
  let pE := (1:ℝ) / 2
  let pF := (1:ℝ) / 4
  let y  := (1:ℝ) / 6
  let z  := (1:ℝ) / 12
  pE + pF + y + z = 1 → y = 2 * z → y = (1:ℝ) / 6 := by
  intros htotal hdouble
  sorry

end probability_of_spinner_stopping_in_region_G_l2355_235559


namespace leaks_empty_time_l2355_235514

theorem leaks_empty_time (A L1 L2: ℝ) (hA: A = 1/2) (hL1_rate: A - L1 = 1/3) 
  (hL2_rate: A - L1 - L2 = 1/4) : 1 / (L1 + L2) = 4 :=
by
  sorry

end leaks_empty_time_l2355_235514


namespace simplify_expression_l2355_235588

theorem simplify_expression (x y : ℝ) : (5 - 4 * y) - (6 + 5 * y - 2 * x) = -1 - 9 * y + 2 * x := by
  sorry

end simplify_expression_l2355_235588


namespace orcs_per_squad_is_eight_l2355_235571

-- Defining the conditions
def total_weight_of_swords := 1200
def weight_each_orc_can_carry := 15
def number_of_squads := 10

-- Proof statement to demonstrate the answer
theorem orcs_per_squad_is_eight :
  (total_weight_of_swords / weight_each_orc_can_carry) / number_of_squads = 8 := by
  sorry

end orcs_per_squad_is_eight_l2355_235571


namespace exist_divisible_number_l2355_235506

theorem exist_divisible_number (d : ℕ) (hd : d > 0) :
  ∃ n : ℕ, (n % d = 0) ∧ ∃ k : ℕ, (k > 0) ∧ (k < 10) ∧ 
  ((∃ m : ℕ, m = n - k*(10^k / 10^k) ∧ m % d = 0) ∨ ∃ m : ℕ, m = n - k * (10^(k - 1)) ∧ m % d = 0) :=
sorry

end exist_divisible_number_l2355_235506


namespace min_value_m2n_mn_l2355_235565

theorem min_value_m2n_mn (m n : ℝ) 
  (h1 : (x - m)^2 + (y - n)^2 = 9)
  (h2 : x + 2 * y + 2 = 0)
  (h3 : 0 < m)
  (h4 : 0 < n)
  (h5 : m + 2 * n + 2 = 5)
  (h6 : ∃ l : ℝ, l = 4 ): (m + 2 * n) / (m * n) = 8/3 :=
by
  sorry

end min_value_m2n_mn_l2355_235565
