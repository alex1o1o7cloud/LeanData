import Mathlib

namespace asymptote_sum_l9_979

noncomputable def f (x : ℝ) : ℝ := (x^3 + 4*x^2 + 3*x) / (x^3 + x^2 - 2*x)

def holes := 0 -- a
def vertical_asymptotes := 2 -- b
def horizontal_asymptotes := 1 -- c
def oblique_asymptotes := 0 -- d

theorem asymptote_sum : holes + 2 * vertical_asymptotes + 3 * horizontal_asymptotes + 4 * oblique_asymptotes = 7 :=
by
  unfold holes vertical_asymptotes horizontal_asymptotes oblique_asymptotes
  norm_num

end asymptote_sum_l9_979


namespace ratio_area_III_IV_l9_906

theorem ratio_area_III_IV 
  (perimeter_I : ℤ)
  (perimeter_II : ℤ)
  (perimeter_IV : ℤ)
  (side_III_is_three_times_side_I : ℤ)
  (h1 : perimeter_I = 16)
  (h2 : perimeter_II = 20)
  (h3 : perimeter_IV = 32)
  (h4 : side_III_is_three_times_side_I = 3 * (perimeter_I / 4)) :
  (3 * (perimeter_I / 4))^2 / (perimeter_IV / 4)^2 = 9 / 4 :=
by
  sorry

end ratio_area_III_IV_l9_906


namespace arithmetic_mean_of_integers_from_neg3_to_6_l9_929

def integer_range := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

noncomputable def arithmetic_mean : ℚ :=
  (integer_range.sum : ℚ) / (integer_range.length : ℚ)

theorem arithmetic_mean_of_integers_from_neg3_to_6 :
  arithmetic_mean = 1.5 := by
  sorry

end arithmetic_mean_of_integers_from_neg3_to_6_l9_929


namespace value_of_abs_sum_l9_956

noncomputable def cos_squared (θ : ℝ) : ℝ := (Real.cos θ) ^ 2

theorem value_of_abs_sum (θ x : ℝ) (h : Real.log x / Real.log 2 = 3 - 2 * cos_squared θ) :
  |x - 2| + |x - 8| = 6 := by
    sorry

end value_of_abs_sum_l9_956


namespace largest_fraction_l9_997

theorem largest_fraction (p q r s : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : q < r) (h4 : r < s) :
  (∃ (x : ℝ), x = (r + s) / (p + q) ∧ 
  (x > (p + s) / (q + r)) ∧ 
  (x > (p + q) / (r + s)) ∧ 
  (x > (q + r) / (p + s)) ∧ 
  (x > (q + s) / (p + r))) :=
sorry

end largest_fraction_l9_997


namespace pyramid_height_l9_915

noncomputable def height_pyramid (perimeter_base : ℝ) (distance_apex_vertex : ℝ) : ℝ :=
  let side_length := perimeter_base / 4
  let half_diagonal := (side_length * Real.sqrt 2) / 2
  Real.sqrt (distance_apex_vertex ^ 2 - half_diagonal ^ 2)

theorem pyramid_height
  (perimeter_base: ℝ)
  (h_perimeter : perimeter_base = 32)
  (distance_apex_vertex: ℝ)
  (h_distance : distance_apex_vertex = 10) :
  height_pyramid perimeter_base distance_apex_vertex = 2 * Real.sqrt 17 :=
by
  sorry

end pyramid_height_l9_915


namespace smallest_possible_bob_number_l9_971

theorem smallest_possible_bob_number : 
  let alices_number := 60
  let bobs_smallest_number := 30
  ∃ (bob_number : ℕ), (∀ p : ℕ, Prime p → p ∣ alices_number → p ∣ bob_number) ∧ bob_number = bobs_smallest_number :=
by
  sorry

end smallest_possible_bob_number_l9_971


namespace value_of_k_l9_983

theorem value_of_k (k : ℕ) (h : 24 / k = 4) : k = 6 := by
  sorry

end value_of_k_l9_983


namespace trigonometric_identity_l9_977

theorem trigonometric_identity :
  8 * Real.cos (4 * Real.pi / 9) * Real.cos (2 * Real.pi / 9) * Real.cos (Real.pi / 9) = 1 :=
by
  sorry

end trigonometric_identity_l9_977


namespace no_first_or_fourth_quadrant_l9_912

theorem no_first_or_fourth_quadrant (a b : ℝ) (h : a * b > 0) : 
  ¬ ((∃ x, a * x + b = 0 ∧ x > 0) ∧ (∃ x, b * x + a = 0 ∧ x > 0)) 
  ∧ ¬ ((∃ x, a * x + b = 0 ∧ x < 0) ∧ (∃ x, b * x + a = 0 ∧ x < 0)) := sorry

end no_first_or_fourth_quadrant_l9_912


namespace recruits_total_l9_907

theorem recruits_total (x y z : ℕ) (total_people : ℕ) :
  (x = total_people - 51) ∧
  (y = total_people - 101) ∧
  (z = total_people - 171) ∧
  (x = 4 * y ∨ y = 4 * z ∨ x = 4 * z) ∧
  (∃ total_people, total_people = 211) :=
sorry

end recruits_total_l9_907


namespace speed_conversion_l9_921

theorem speed_conversion (speed_mps : ℝ) (conversion_factor : ℝ) (speed_kmph_expected : ℝ) :
  speed_mps = 35.0028 →
  conversion_factor = 3.6 →
  speed_kmph_expected = 126.01008 →
  speed_mps * conversion_factor = speed_kmph_expected :=
by
  intros h_mps h_cf h_kmph
  rw [h_mps, h_cf, h_kmph]
  sorry

end speed_conversion_l9_921


namespace find_parallel_line_l9_980

-- Definition of the point (0, 1)
def point : ℝ × ℝ := (0, 1)

-- Definition of the original line equation
def original_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0

-- Definition of the desired line equation
def desired_line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Theorem statement: defining the desired line based on the point and parallelism condition
theorem find_parallel_line (x y : ℝ) (hx : point.fst = 0) (hy : point.snd = 1) :
  ∃ m : ℝ, (2 * x + y + m = 0) ∧ (2 * 0 + 1 + m = 0) → desired_line x y :=
sorry

end find_parallel_line_l9_980


namespace length_of_AB_l9_986

noncomputable def isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ a = c ∨ b = c

theorem length_of_AB 
  (a b c d e : ℕ)
  (h_iso_ABC : isosceles_triangle a b c)
  (h_iso_CDE : isosceles_triangle c d e)
  (h_perimeter_CDE : c + d + e = 25)
  (h_perimeter_ABC : a + b + c = 24)
  (h_CE : c = 9)
  (h_AB_DE : a = e) : a = 7 :=
by
  sorry

end length_of_AB_l9_986


namespace walking_time_l9_902

theorem walking_time (r s : ℕ) (h₁ : r + s = 50) (h₂ : 2 * s = 30) : 2 * r = 70 :=
by
  sorry

end walking_time_l9_902


namespace average_runs_in_30_matches_l9_968

theorem average_runs_in_30_matches 
  (avg1 : ℕ) (matches1 : ℕ) (avg2 : ℕ) (matches2 : ℕ) (total_matches : ℕ)
  (h1 : avg1 = 40) (h2 : matches1 = 20) (h3 : avg2 = 13) (h4 : matches2 = 10) (h5 : total_matches = 30) :
  ((avg1 * matches1 + avg2 * matches2) / total_matches) = 31 := by
  sorry

end average_runs_in_30_matches_l9_968


namespace determine_y_l9_931

theorem determine_y : 
  ∀ y : ℝ, 
    (2 * Real.arctan (1 / 5) + Real.arctan (1 / 25) + Real.arctan (1 / y) = Real.pi / 4) -> 
    y = -121 / 60 :=
by
  sorry

end determine_y_l9_931


namespace lilly_can_buy_flowers_l9_965

-- Define variables
def days_until_birthday : ℕ := 22
def daily_savings : ℕ := 2
def flower_cost : ℕ := 4

-- Statement: Given the conditions, prove the number of flowers Lilly can buy.
theorem lilly_can_buy_flowers :
  (days_until_birthday * daily_savings) / flower_cost = 11 := 
by
  -- proof steps
  sorry

end lilly_can_buy_flowers_l9_965


namespace students_not_picked_l9_934

theorem students_not_picked (total_students groups group_size : ℕ) (h1 : total_students = 64)
(h2 : groups = 4) (h3 : group_size = 7) :
total_students - groups * group_size = 36 :=
by
  sorry

end students_not_picked_l9_934


namespace find_a12_l9_940

variable {a : ℕ → ℝ}
variable (d : ℝ)

-- Definition of the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- The Lean statement for the problem
theorem find_a12 (h_seq : arithmetic_sequence a d)
  (h_cond1 : a 7 + a 9 = 16) (h_cond2 : a 4 = 1) : 
  a 12 = 15 :=
sorry

end find_a12_l9_940


namespace cos_seven_theta_l9_919

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -953 / 1024 :=
by sorry

end cos_seven_theta_l9_919


namespace repeating_decimal_to_fraction_l9_975

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 0.3 + 56 / 9900) : x = 3969 / 11100 := 
sorry

end repeating_decimal_to_fraction_l9_975


namespace sum_remainder_l9_927

theorem sum_remainder (n : ℤ) : ((9 - n) + (n + 4)) % 9 = 4 := 
by 
  sorry

end sum_remainder_l9_927


namespace min_value_S_l9_910

theorem min_value_S (a b c : ℤ) (h1 : a + b + c = 2) (h2 : (2 * a + b * c) * (2 * b + c * a) * (2 * c + a * b) > 200) :
  ∃ a b c : ℤ, a + b + c = 2 ∧ (2 * a + b * c) * (2 * b + c * a) * (2 * c + a * b) = 256 :=
sorry

end min_value_S_l9_910


namespace kelseys_sisters_age_l9_936

theorem kelseys_sisters_age :
  ∀ (current_year : ℕ) (kelsey_birth_year : ℕ)
    (kelsey_sister_birth_year : ℕ),
    kelsey_birth_year = 1999 - 25 →
    kelsey_sister_birth_year = kelsey_birth_year - 3 →
    current_year = 2021 →
    current_year - kelsey_sister_birth_year = 50 :=
by
  intros current_year kelsey_birth_year kelsey_sister_birth_year h1 h2 h3
  sorry

end kelseys_sisters_age_l9_936


namespace profit_percentage_l9_957

theorem profit_percentage (CP SP : ℝ) (h : 18 * CP = 16 * SP) : 
  (SP - CP) / CP * 100 = 12.5 := by
sorry

end profit_percentage_l9_957


namespace angle_A_is_pi_over_3_l9_923

theorem angle_A_is_pi_over_3 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : (a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C)
  (h2 : a ^ 2 = b ^ 2 + c ^ 2 - bc * (2 * Real.cos A))
  (triangle_ABC : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ A + B + C = π) :
  A = π / 3 :=
by
  sorry

end angle_A_is_pi_over_3_l9_923


namespace remainder_eq_six_l9_933

theorem remainder_eq_six
  (Dividend : ℕ) (Divisor : ℕ) (Quotient : ℕ) (Remainder : ℕ)
  (h1 : Dividend = 139)
  (h2 : Divisor = 19)
  (h3 : Quotient = 7)
  (h4 : Dividend = (Divisor * Quotient) + Remainder) :
  Remainder = 6 :=
by
  sorry

end remainder_eq_six_l9_933


namespace inheritance_amount_l9_953

theorem inheritance_amount (x : ℝ) 
    (federal_tax : ℝ := 0.25 * x) 
    (remaining_after_federal_tax : ℝ := x - federal_tax) 
    (state_tax : ℝ := 0.15 * remaining_after_federal_tax) 
    (total_taxes : ℝ := federal_tax + state_tax) 
    (taxes_paid : total_taxes = 15000) : 
    x = 41379 :=
sorry

end inheritance_amount_l9_953


namespace shaded_area_is_20_l9_948

-- Represents the square PQRS with the necessary labeled side lengths
noncomputable def square_side_length : ℝ := 8

-- Represents the four labeled smaller squares' positions and their side lengths
noncomputable def smaller_square_side_lengths : List ℝ := [2, 2, 2, 6]

-- The coordinates or relations to describe their overlaying positions are not needed for the proof.

-- Define the calculated areas from the solution steps
noncomputable def vertical_rectangle_area : ℝ := 6 * 2
noncomputable def horizontal_rectangle_area : ℝ := 6 * 2
noncomputable def overlap_area : ℝ := 2 * 2

-- The total shaded T-shaped region area calculation
noncomputable def total_shaded_area : ℝ := vertical_rectangle_area + horizontal_rectangle_area - overlap_area

-- Theorem statement to prove the area of the T-shaped region is 20
theorem shaded_area_is_20 : total_shaded_area = 20 :=
by
  -- Proof steps are not required as per the instruction.
  sorry

end shaded_area_is_20_l9_948


namespace B_listing_method_l9_962

-- Definitions for given conditions
def A : Set ℤ := {-2, -1, 1, 2, 3, 4}
def B : Set ℤ := {x | ∃ t ∈ A, x = t*t}

-- The mathematically equivalent proof problem
theorem B_listing_method :
  B = {4, 1, 9, 16} := 
by {
  sorry
}

end B_listing_method_l9_962


namespace rain_at_least_one_day_probability_l9_922

-- Definitions based on given conditions
def P_rain_Friday : ℝ := 0.30
def P_rain_Monday : ℝ := 0.20

-- Events probabilities based on independence
def P_no_rain_Friday := 1 - P_rain_Friday
def P_no_rain_Monday := 1 - P_rain_Monday
def P_no_rain_both := P_no_rain_Friday * P_no_rain_Monday

-- The probability of raining at least one day
def P_rain_at_least_one_day := 1 - P_no_rain_both

-- Expected probability
def expected_probability : ℝ := 0.44

theorem rain_at_least_one_day_probability : 
  P_rain_at_least_one_day = expected_probability := by
  sorry

end rain_at_least_one_day_probability_l9_922


namespace solve_for_m_l9_976

open Real

theorem solve_for_m (a b m : ℝ)
  (h1 : (1/2)^a = m)
  (h2 : 3^b = m)
  (h3 : 1/a - 1/b = 2) :
  m = sqrt 6 / 6 := 
  sorry

end solve_for_m_l9_976


namespace tap_filling_time_l9_937

theorem tap_filling_time
  (T : ℝ)
  (H1 : 10 > 0) -- Second tap can empty the cistern in 10 hours
  (H2 : T > 0)  -- First tap's time must be positive
  (H3 : (1 / T) - (1 / 10) = (3 / 20))  -- Both taps together fill the cistern in 6.666... hours
  : T = 4 := sorry

end tap_filling_time_l9_937


namespace difference_of_digits_is_six_l9_909

theorem difference_of_digits_is_six (a b : ℕ) (h_sum : a + b = 10) (h_number : 10 * a + b = 82) : a - b = 6 :=
sorry

end difference_of_digits_is_six_l9_909


namespace fraction_raised_to_zero_l9_911

theorem fraction_raised_to_zero:
  (↑(-4305835) / ↑1092370457 : ℚ)^0 = 1 := 
by
  sorry

end fraction_raised_to_zero_l9_911


namespace mandy_more_than_three_friends_l9_946

noncomputable def stickers_given_to_three_friends : ℕ := 4 * 3
noncomputable def total_initial_stickers : ℕ := 72
noncomputable def stickers_left : ℕ := 42
noncomputable def total_given_away : ℕ := total_initial_stickers - stickers_left
noncomputable def mandy_justin_total : ℕ := total_given_away - stickers_given_to_three_friends
noncomputable def mandy_stickers : ℕ := 14
noncomputable def three_friends_stickers : ℕ := stickers_given_to_three_friends

theorem mandy_more_than_three_friends : 
  mandy_stickers - three_friends_stickers = 2 :=
by
  sorry

end mandy_more_than_three_friends_l9_946


namespace complex_multiplication_l9_941

-- Definition of the imaginary unit
def is_imaginary_unit (i : ℂ) : Prop := i * i = -1

theorem complex_multiplication (i : ℂ) (h : is_imaginary_unit i) : (1 + i) * (1 - i) = 2 :=
by
  -- Given that i is the imaginary unit satisfying i^2 = -1
  -- We need to show that (1 + i) * (1 - i) = 2
  sorry

end complex_multiplication_l9_941


namespace glued_cubes_surface_area_l9_964

theorem glued_cubes_surface_area (L l : ℝ) (h1 : L = 2) (h2 : l = L / 2) : 
  6 * L^2 + 4 * l^2 = 28 :=
by
  sorry

end glued_cubes_surface_area_l9_964


namespace arithmetic_sequence_problem_l9_925

variable (a_2 a_4 a_3 : ℤ)

theorem arithmetic_sequence_problem (h : a_2 + a_4 = 16) : a_3 = 8 :=
by
  -- The proof is not needed as per the instructions
  sorry

end arithmetic_sequence_problem_l9_925


namespace fraction_of_track_Scottsdale_to_Forest_Grove_l9_958

def distance_between_Scottsdale_and_Sherbourne : ℝ := 200
def round_trip_duration : ℝ := 5
def time_Harsha_to_Sherbourne : ℝ := 2

theorem fraction_of_track_Scottsdale_to_Forest_Grove :
  ∃ f : ℝ, f = 1/5 ∧
    ∀ (d : ℝ) (t : ℝ) (h : ℝ),
    d = distance_between_Scottsdale_and_Sherbourne →
    t = round_trip_duration →
    h = time_Harsha_to_Sherbourne →
    (2.5 - h) / t = f :=
sorry

end fraction_of_track_Scottsdale_to_Forest_Grove_l9_958


namespace correct_operation_l9_996

theorem correct_operation (x : ℝ) : (2 * x ^ 3) ^ 2 = 4 * x ^ 6 := 
  sorry

end correct_operation_l9_996


namespace factorize_difference_of_squares_l9_972

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l9_972


namespace calculate_earths_atmosphere_mass_l9_945

noncomputable def mass_of_earths_atmosphere (R p0 g : ℝ) : ℝ :=
  (4 * Real.pi * R^2 * p0) / g

theorem calculate_earths_atmosphere_mass (R p0 g : ℝ) (h : 0 < g) : 
  mass_of_earths_atmosphere R p0 g = 5 * 10^18 := 
sorry

end calculate_earths_atmosphere_mass_l9_945


namespace solve_for_y_l9_950

theorem solve_for_y (y : ℝ) (h : 5^(3 * y) = Real.sqrt 125) : y = 1 / 2 :=
by sorry

end solve_for_y_l9_950


namespace man_l9_970

theorem man's_age (x : ℕ) : 6 * (x + 6) - 6 * (x - 6) = x → x = 72 :=
by
  sorry

end man_l9_970


namespace cannot_sum_85_with_five_coins_l9_900

def coin_value (c : Nat) : Prop :=
  c = 1 ∨ c = 5 ∨ c = 10 ∨ c = 25 ∨ c = 50

theorem cannot_sum_85_with_five_coins : 
  ¬ ∃ (a b c d e : Nat), 
    coin_value a ∧ 
    coin_value b ∧ 
    coin_value c ∧ 
    coin_value d ∧ 
    coin_value e ∧ 
    a + b + c + d + e = 85 :=
by
  sorry

end cannot_sum_85_with_five_coins_l9_900


namespace find_number_l9_960

theorem find_number:
  ∃ x: ℕ, (∃ k: ℕ, ∃ r: ℕ, 5 * (x + 3) = 8 * k + r ∧ k = 156 ∧ r = 2) ∧ x = 247 :=
by 
  sorry

end find_number_l9_960


namespace gcd_lcm_mul_l9_938

theorem gcd_lcm_mul (a b : ℕ) : Nat.gcd a b * Nat.lcm a b = a * b := 
by
  sorry

end gcd_lcm_mul_l9_938


namespace ball_returns_to_bob_after_13_throws_l9_994

theorem ball_returns_to_bob_after_13_throws:
  ∃ n : ℕ, n = 13 ∧ (∀ k, k < 13 → (1 + 3 * k) % 13 = 0) :=
sorry

end ball_returns_to_bob_after_13_throws_l9_994


namespace total_coins_l9_966

theorem total_coins (x y : ℕ) (h : x ≠ y) (h1 : x^2 - y^2 = 81 * (x - y)) : x + y = 81 := by
  sorry

end total_coins_l9_966


namespace nathaniel_initial_tickets_l9_969

theorem nathaniel_initial_tickets (a b c : ℕ) (h1 : a = 2) (h2 : b = 4) (h3 : c = 3) :
  a * b + c = 11 :=
by
  sorry

end nathaniel_initial_tickets_l9_969


namespace external_tangent_inequality_l9_978

variable (x y z : ℝ)
variable (a b c T : ℝ)

-- Definitions based on conditions
def a_def : a = x + y := sorry
def b_def : b = y + z := sorry
def c_def : c = z + x := sorry
def T_def : T = π * x^2 + π * y^2 + π * z^2 := sorry

-- The theorem to prove
theorem external_tangent_inequality
    (a_def : a = x + y) 
    (b_def : b = y + z) 
    (c_def : c = z + x) 
    (T_def : T = π * x^2 + π * y^2 + π * z^2) : 
    π * (a + b + c) ^ 2 ≤ 12 * T := 
sorry

end external_tangent_inequality_l9_978


namespace a₁₀_greater_than_500_l9_905

variables (a : ℕ → ℕ) (b : ℕ → ℕ)

-- Conditions
def strictly_increasing (a : ℕ → ℕ) : Prop := ∀ n, a n < a (n + 1)

def largest_divisor (a : ℕ → ℕ) (b : ℕ → ℕ) : Prop :=
  ∀ n, b n < a n ∧ ∃ d > 1, d ∣ a n ∧ b n = a n / d

def greater_sequence (b : ℕ → ℕ) : Prop := ∀ n, b n > b (n + 1)

-- Statement to prove
theorem a₁₀_greater_than_500
  (h1 : strictly_increasing a)
  (h2 : largest_divisor a b)
  (h3 : greater_sequence b) :
  a 10 > 500 :=
sorry

end a₁₀_greater_than_500_l9_905


namespace sum_of_first_53_odd_numbers_l9_988

theorem sum_of_first_53_odd_numbers :
  let first_term := 1
  let last_term := first_term + (53 - 1) * 2
  let sum := 53 / 2 * (first_term + last_term)
  sum = 2809 :=
by
  let first_term := 1
  let last_term := first_term + (53 - 1) * 2
  have last_term_val : last_term = 105 := by
    sorry
  let sum := 53 / 2 * (first_term + last_term)
  have sum_val : sum = 2809 := by
    sorry
  exact sum_val

end sum_of_first_53_odd_numbers_l9_988


namespace smallest_n_for_polygon_cutting_l9_944

theorem smallest_n_for_polygon_cutting : 
  ∃ n : ℕ, (∃ k : ℕ, n - 2 = k * 31) ∧ (∃ k' : ℕ, n - 2 = k' * 65) ∧ n = 2017 :=
sorry

end smallest_n_for_polygon_cutting_l9_944


namespace ratio_of_hypotenuse_segments_l9_920

theorem ratio_of_hypotenuse_segments (a b c d : ℝ) 
  (h1 : a^2 + b^2 = c^2)
  (h2 : b = (3/4) * a)
  (h3 : d^2 = (c - d)^2 + b^2) :
  (d / (c - d)) = (4 / 3) :=
sorry

end ratio_of_hypotenuse_segments_l9_920


namespace eight_hash_four_eq_ten_l9_955

def operation (a b : ℚ) : ℚ := a + a / b

theorem eight_hash_four_eq_ten : operation 8 4 = 10 :=
by
  sorry

end eight_hash_four_eq_ten_l9_955


namespace new_ratio_is_three_half_l9_904

theorem new_ratio_is_three_half (F J : ℕ) (h1 : F * 4 = J * 5) (h2 : J = 120) :
  ((F + 30) : ℚ) / J = 3 / 2 :=
by
  sorry

end new_ratio_is_three_half_l9_904


namespace union_M_N_eq_l9_943

open Set

-- Define M according to the condition x^2 < 15 for x in ℕ
def M : Set ℕ := {x | x^2 < 15}

-- Define N according to the correct answer
def N : Set ℕ := {x | 0 < x ∧ x < 5}

-- Prove that M ∪ N = {x | 0 ≤ x ∧ x < 5}
theorem union_M_N_eq : M ∪ N = {x : ℕ | 0 ≤ x ∧ x < 5} :=
sorry

end union_M_N_eq_l9_943


namespace find_number_l9_917

theorem find_number (x : ℝ) (h : 0.6667 * x + 1 = 0.75 * x) : x = 12 := 
by
  sorry

end find_number_l9_917


namespace certain_number_of_tenths_l9_916

theorem certain_number_of_tenths (n : ℝ) (h : n = 375 * (1/10)) : n = 37.5 :=
by
  sorry

end certain_number_of_tenths_l9_916


namespace dot_product_a_b_l9_952

def a : ℝ × ℝ := (-2, 3)
def b : ℝ × ℝ := (1, 2)

theorem dot_product_a_b : a.1 * b.1 + a.2 * b.2 = 4 := by
  sorry

end dot_product_a_b_l9_952


namespace intersection_of_diagonals_l9_928

-- Define the four lines based on the given conditions
def line1 (k b x : ℝ) : ℝ := k*x + b
def line2 (k b x : ℝ) : ℝ := k*x - b
def line3 (m b x : ℝ) : ℝ := m*x + b
def line4 (m b x : ℝ) : ℝ := m*x - b

-- Define a function to represent the problem
noncomputable def point_of_intersection_of_diagonals (k m b : ℝ) : ℝ × ℝ :=
(0, 0)

-- State the theorem to be proved
theorem intersection_of_diagonals (k m b : ℝ) :
  point_of_intersection_of_diagonals k m b = (0, 0) :=
sorry

end intersection_of_diagonals_l9_928


namespace cubic_roots_expression_l9_967

theorem cubic_roots_expression (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b + a * c + b * c = -1) (h3 : a * b * c = 2) :
  2 * a * (b - c) ^ 2 + 2 * b * (c - a) ^ 2 + 2 * c * (a - b) ^ 2 = -36 :=
by
  sorry

end cubic_roots_expression_l9_967


namespace trigonometric_sign_l9_981

open Real

theorem trigonometric_sign :
  (0 < 1 ∧ 1 < π / 2) ∧ 
  (∀ x y, (0 ≤ x ∧ x ≤ y ∧ y ≤ π / 2 → sin x ≤ sin y)) ∧ 
  (∀ x y, (0 ≤ x ∧ x ≤ y ∧ y ≤ π / 2 → cos x ≥ cos y)) →
  (cos (cos 1) - cos 1) * (sin (sin 1) - sin 1) < 0 :=
by
  sorry

end trigonometric_sign_l9_981


namespace math_proof_problem_l9_985

noncomputable def f (a b : ℚ) : ℝ := sorry

axiom f_cond1 (a b c : ℚ) : f (a * b) c = f a c * f b c ∧ f c (a * b) = f c a * f c b
axiom f_cond2 (a : ℚ) : f a (1 - a) = 1

theorem math_proof_problem (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (f a a = 1) ∧ 
  (f a (-a) = 1) ∧
  (f a b * f b a = 1) := 
by 
  sorry

end math_proof_problem_l9_985


namespace proof_system_l9_954

-- Define the system of equations
def system_of_equations (x y : ℝ) : Prop :=
  6 * x - 2 * y = 1 ∧ 2 * x + y = 2

-- Define the solution to the system of equations
def solution_equations (x y : ℝ) : Prop :=
  x = 0.5 ∧ y = 1

-- Define the system of inequalities
def system_of_inequalities (x : ℝ) : Prop :=
  2 * x - 10 < 0 ∧ (x + 1) / 3 < x - 1

-- Define the solution set for the system of inequalities
def solution_inequalities (x : ℝ) : Prop :=
  2 < x ∧ x < 5

-- The final theorem to be proved
theorem proof_system :
  ∃ x y : ℝ, system_of_equations x y ∧ solution_equations x y ∧ system_of_inequalities x ∧ solution_inequalities x :=
by
  sorry

end proof_system_l9_954


namespace find_smallest_n_l9_951

theorem find_smallest_n (k : ℕ) (hk: 0 < k) :
        ∃ n : ℕ, (∀ (s : Finset ℤ), s.card = n → 
        ∃ (x y : ℤ), x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ (x + y) % (2 * k) = 0 ∨ (x - y) % (2 * k) = 0) 
        ∧ n = k + 2 :=
sorry

end find_smallest_n_l9_951


namespace new_machine_rate_l9_913

def old_machine_rate : ℕ := 100
def total_bolts : ℕ := 500
def time_hours : ℕ := 2

theorem new_machine_rate (R : ℕ) : 
  (old_machine_rate * time_hours + R * time_hours = total_bolts) → 
  R = 150 := 
by
  sorry

end new_machine_rate_l9_913


namespace v2004_eq_1_l9_987

def g (x: ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 1
  | 4 => 2
  | 5 => 4
  | _ => 0  -- assuming default value for undefined cases

def v : ℕ → ℕ
| 0     => 3
| (n+1) => g (v n + 1)

theorem v2004_eq_1 : v 2004 = 1 :=
  sorry

end v2004_eq_1_l9_987


namespace problem1_problem2_problem3_problem4_l9_901

-- Question 1
theorem problem1 (a b : ℝ) (h : 5 * a + 3 * b = -4) : 2 * (a + b) + 4 * (2 * a + b) = -8 :=
by
  sorry

-- Question 2
theorem problem2 (a : ℝ) (h : a^2 + a = 3) : 2 * a^2 + 2 * a + 2023 = 2029 :=
by
  sorry

-- Question 3
theorem problem3 (a b : ℝ) (h : a - 2 * b = -3) : 3 * (a - b) - 7 * a + 11 * b + 2 = 14 :=
by
  sorry

-- Question 4
theorem problem4 (a b : ℝ) 
  (h1 : a^2 + 2 * a * b = -5) 
  (h2 : a * b - 2 * b^2 = -3) : a^2 + a * b + 2 * b^2 = -2 :=
by
  sorry

end problem1_problem2_problem3_problem4_l9_901


namespace triangle_inequality_l9_995

theorem triangle_inequality (a : ℝ) (h₁ : a > 5) (h₂ : a < 19) : 5 < a ∧ a < 19 :=
by
  exact ⟨h₁, h₂⟩

end triangle_inequality_l9_995


namespace combinatorial_identity_l9_924

theorem combinatorial_identity :
  (Nat.factorial 15) / ((Nat.factorial 6) * (Nat.factorial 9)) = 5005 :=
sorry

end combinatorial_identity_l9_924


namespace greatest_value_of_x_l9_973

theorem greatest_value_of_x (x : ℕ) : (Nat.lcm (Nat.lcm x 12) 18 = 180) → x ≤ 180 :=
by
  sorry

end greatest_value_of_x_l9_973


namespace unique_root_range_l9_984

theorem unique_root_range (a : ℝ) :
  (x^3 + (1 - 3 * a) * x^2 + 2 * a^2 * x - 2 * a * x + x + a^2 - a = 0) 
  → (∃! x : ℝ, x^3 + (1 - 3 * a) * x^2 + 2 * a^2 * x - 2 * a * x + x + a^2 - a = 0) 
  → - (Real.sqrt 3) / 2 < a ∧ a < (Real.sqrt 3) / 2 :=
by
  sorry

end unique_root_range_l9_984


namespace surface_area_invisible_block_l9_926

-- Define the given areas of the seven blocks
def A1 := 148
def A2 := 46
def A3 := 72
def A4 := 28
def A5 := 88
def A6 := 126
def A7 := 58

-- Define total surface areas of the black and white blocks
def S_black := A1 + A2 + A3 + A4
def S_white := A5 + A6 + A7

-- Define the proof problem
theorem surface_area_invisible_block : S_black - S_white = 22 :=
by
  -- This sorry allows the Lean statement to build successfully
  sorry

end surface_area_invisible_block_l9_926


namespace possible_red_ball_draws_l9_992

/-- 
Given two balls in a bag where one is white and the other is red, 
if a ball is drawn and returned, and then another ball is drawn, 
prove that the possible number of times a red ball is drawn is 0, 1, or 2.
-/
theorem possible_red_ball_draws : 
  (∀ balls : Finset (ℕ × ℕ), 
    balls = {(0, 1), (1, 0)} →
    ∀ draw1 draw2 : ℕ × ℕ, 
    draw1 ∈ balls →
    draw2 ∈ balls →
    ∃ n : ℕ, (n = 0 ∨ n = 1 ∨ n = 2) ∧ 
    n = (if draw1 = (1, 0) then 1 else 0) + 
        (if draw2 = (1, 0) then 1 else 0)) → 
    True := sorry

end possible_red_ball_draws_l9_992


namespace chips_probability_l9_949

def total_chips : ℕ := 12
def blue_chips : ℕ := 4
def green_chips : ℕ := 3
def red_chips : ℕ := 5

def total_ways : ℕ := Nat.factorial total_chips

def blue_group_ways : ℕ := Nat.factorial blue_chips
def green_group_ways : ℕ := Nat.factorial green_chips
def red_group_ways : ℕ := Nat.factorial red_chips
def group_permutations : ℕ := Nat.factorial 3

def satisfying_arrangements : ℕ :=
  group_permutations * blue_group_ways * green_group_ways * red_group_ways

noncomputable def probability_of_event_B : ℚ :=
  (satisfying_arrangements : ℚ) / (total_ways : ℚ)

theorem chips_probability :
  probability_of_event_B = 1 / 4620 :=
by
  sorry

end chips_probability_l9_949


namespace price_of_one_table_l9_982

variable (C T : ℝ)

def cond1 := 2 * C + T = 0.6 * (C + 2 * T)
def cond2 := C + T = 60
def solution := T = 52.5

theorem price_of_one_table (h1 : cond1 C T) (h2 : cond2 C T) : solution T :=
by
  sorry

end price_of_one_table_l9_982


namespace original_faculty_members_l9_991

theorem original_faculty_members
  (x : ℝ) (h : 0.87 * x = 195) : x = 224 := sorry

end original_faculty_members_l9_991


namespace solution_set_contains_0_and_2_l9_935

theorem solution_set_contains_0_and_2 (k : ℝ) : 
  ∀ x, ((1 + k^2) * x ≤ k^4 + 4) → (x = 0 ∨ x = 2) :=
by {
  sorry -- Proof is omitted
}

end solution_set_contains_0_and_2_l9_935


namespace find_box_value_l9_918

theorem find_box_value (r x : ℕ) 
  (h1 : x + r = 75)
  (h2 : (x + r) + 2 * r = 143) : 
  x = 41 := 
by
  sorry

end find_box_value_l9_918


namespace students_not_like_any_l9_989

variables (F B P T F_cap_B F_cap_P F_cap_T B_cap_P B_cap_T P_cap_T F_cap_B_cap_P_cap_T : ℕ)

def total_students := 30

def students_like_F := 18
def students_like_B := 12
def students_like_P := 14
def students_like_T := 10

def students_like_F_and_B := 8
def students_like_F_and_P := 6
def students_like_F_and_T := 4
def students_like_B_and_P := 5
def students_like_B_and_T := 3
def students_like_P_and_T := 7

def students_like_all_four := 2

theorem students_not_like_any :
  total_students - ((students_like_F + students_like_B + students_like_P + students_like_T)
                    - (students_like_F_and_B + students_like_F_and_P + students_like_F_and_T
                      + students_like_B_and_P + students_like_B_and_T + students_like_P_and_T)
                    + students_like_all_four) = 11 :=
by sorry

end students_not_like_any_l9_989


namespace problem1_problem2_l9_947

def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x^2 - 3 * x

theorem problem1 (a : ℝ) : (∀ x : ℝ, x ≥ 1 → 3 * x^2 - 2 * a * x - 3 ≥ 0) → a ≤ 0 :=
sorry

theorem problem2 (a : ℝ) (h : a = 6) :
  x = 3 ∧ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 6 → f x 6 ≤ -6 ∧ f x 6 ≥ -18) :=
sorry

end problem1_problem2_l9_947


namespace simplify_expression_l9_974

theorem simplify_expression (x : ℝ) : 3 * x + 4 * x^2 + 2 - (5 - 3 * x - 5 * x^2 + x^3) = -x^3 + 9 * x^2 + 6 * x - 3 :=
by
  sorry -- Proof is omitted.

end simplify_expression_l9_974


namespace reaction2_follows_markovnikov_l9_993

-- Define Markovnikov's rule - applying to case with protic acid (HX) to an alkene.
def follows_markovnikov_rule (HX : String) (initial_molecule final_product : String) : Prop :=
  initial_molecule = "CH3-CH=CH2 + HBr" ∧ final_product = "CH3-CHBr-CH3"

-- Example reaction data
def reaction1_initial : String := "CH2=CH2 + Br2"
def reaction1_final : String := "CH2Br-CH2Br"

def reaction2_initial : String := "CH3-CH=CH2 + HBr"
def reaction2_final : String := "CH3-CHBr-CH3"

def reaction3_initial : String := "CH4 + Cl2"
def reaction3_final : String := "CH3Cl + HCl"

def reaction4_initial : String := "CH ≡ CH + HOH"
def reaction4_final : String := "CH3''-C-H"

-- Proof statement
theorem reaction2_follows_markovnikov : follows_markovnikov_rule "HBr" reaction2_initial reaction2_final := by
  sorry

end reaction2_follows_markovnikov_l9_993


namespace smallest_n_for_abc_factorials_l9_930

theorem smallest_n_for_abc_factorials (a b c : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a + b + c = 2006) :
  ∃ m n : ℕ, (¬ ∃ k : ℕ, m = 10 * k) ∧ a.factorial * b.factorial * c.factorial = m * 10^n ∧ n = 492 :=
sorry

end smallest_n_for_abc_factorials_l9_930


namespace fraction_of_menu_items_my_friend_can_eat_l9_942

theorem fraction_of_menu_items_my_friend_can_eat {menu_size vegan_dishes nut_free_vegan_dishes : ℕ}
    (h1 : vegan_dishes = 6)
    (h2 : vegan_dishes = menu_size / 6)
    (h3 : nut_free_vegan_dishes = vegan_dishes - 5) :
    (nut_free_vegan_dishes : ℚ) / menu_size = 1 / 36 :=
by
  sorry

end fraction_of_menu_items_my_friend_can_eat_l9_942


namespace find_weekday_rate_l9_959

-- Definitions of given conditions
def num_people : ℕ := 6
def days_weekdays : ℕ := 2
def days_weekend : ℕ := 2
def weekend_rate : ℕ := 540
def payment_per_person : ℕ := 320

-- Theorem to prove the weekday rental rate
theorem find_weekday_rate (W : ℕ) :
  (num_people * payment_per_person) = (days_weekdays * W) + (days_weekend * weekend_rate) →
  W = 420 :=
by 
  intros h
  sorry

end find_weekday_rate_l9_959


namespace red_grapes_count_l9_963

theorem red_grapes_count (G : ℕ) (total_fruit : ℕ) (red_grapes : ℕ) (raspberries : ℕ)
  (h1 : red_grapes = 3 * G + 7) 
  (h2 : raspberries = G - 5) 
  (h3 : total_fruit = G + red_grapes + raspberries) 
  (h4 : total_fruit = 102) : 
  red_grapes = 67 :=
by
  sorry

end red_grapes_count_l9_963


namespace solve_for_y_l9_999

theorem solve_for_y (y : ℝ) (h : (5 - 1 / y)^(1/3) = -3) : y = 1 / 32 :=
by
  sorry

end solve_for_y_l9_999


namespace perfect_square_form_l9_932

theorem perfect_square_form (N : ℕ) (hN : 0 < N) : 
  ∃ x : ℤ, 2^N - 2 * (N : ℤ) = x^2 ↔ N = 1 ∨ N = 2 :=
by
  sorry

end perfect_square_form_l9_932


namespace exists_triangle_with_edges_l9_990

variable {A B C D: Type}
variables (AB AC AD BC BD CD : ℝ)
variables (tetrahedron : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)

def x := AB * CD
def y := AC * BD
def z := AD * BC

theorem exists_triangle_with_edges :
  ∃ (x y z : ℝ), 
  ∃ (A B C D: Type),
  ∃ (AB AC AD BC BD CD : ℝ) (tetrahedron : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D),
  x = AB * CD ∧ y = AC * BD ∧ z = AD * BC → 
  (x + y > z ∧ y + z > x ∧ z + x > y) :=
by
  sorry

end exists_triangle_with_edges_l9_990


namespace medium_ceiling_lights_count_l9_908

theorem medium_ceiling_lights_count (S M L : ℕ) 
  (h1 : L = 2 * M) 
  (h2 : S = M + 10) 
  (h_bulbs : S + 2 * M + 3 * L = 118) : M = 12 :=
by
  -- Proof omitted
  sorry

end medium_ceiling_lights_count_l9_908


namespace donut_cubes_eaten_l9_914

def cube_dimensions := 5

def total_cubes_in_cube : ℕ := cube_dimensions ^ 3

def even_neighbors (faces_sharing_cubes : ℕ) : Prop :=
  faces_sharing_cubes % 2 = 0

/-- A corner cube in a 5x5x5 cube has 3 neighbors. --/
def corner_cube_neighbors := 3

/-- An edge cube in a 5x5x5 cube (excluding corners) has 4 neighbors. --/
def edge_cube_neighbors := 4

/-- A face center cube in a 5x5x5 cube has 5 neighbors. --/
def face_center_cube_neighbors := 5

/-- An inner cube in a 5x5x5 cube has 6 neighbors. --/
def inner_cube_neighbors := 6

/-- Count of edge cubes that share 4 neighbors in a 5x5x5 cube. --/
def edge_cubes_count := 12 * (cube_dimensions - 2)

def inner_cubes_count := (cube_dimensions - 2) ^ 3

theorem donut_cubes_eaten :
  (edge_cubes_count + inner_cubes_count) = 63 := by
  sorry

end donut_cubes_eaten_l9_914


namespace most_balls_l9_939

def soccerballs : ℕ := 50
def basketballs : ℕ := 26
def baseballs : ℕ := basketballs + 8

theorem most_balls :
  max (max soccerballs basketballs) baseballs = soccerballs := by
  sorry

end most_balls_l9_939


namespace option_A_option_C_option_D_l9_998

noncomputable def ratio_12_11 := (12 : ℝ) / 11
noncomputable def ratio_11_10 := (11 : ℝ) / 10

theorem option_A : ratio_12_11^11 > ratio_11_10^10 := sorry

theorem option_C : ratio_12_11^10 > ratio_11_10^9 := sorry

theorem option_D : ratio_11_10^12 > ratio_12_11^13 := sorry

end option_A_option_C_option_D_l9_998


namespace simplify_and_evaluate_expression_l9_961

/-
Problem: Prove ( (a + 1) / (a - 1) + 1 ) / ( 2a / (a^2 - 1) ) = 2024 given a = 2023.
-/

theorem simplify_and_evaluate_expression (a : ℕ) (h : a = 2023) :
  ( (a + 1) / (a - 1) + 1 ) / ( 2 * a / (a^2 - 1) ) = 2024 :=
by
  sorry

end simplify_and_evaluate_expression_l9_961


namespace average_hit_targets_value_average_hit_targets_ge_half_l9_903

noncomputable def average_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - 1 / n)^n)

theorem average_hit_targets_value (n : ℕ) :
  average_hit_targets n = n * (1 - (1 - 1 / n)^n) :=
by sorry

theorem average_hit_targets_ge_half (n : ℕ) :
  average_hit_targets n >= n / 2 :=
by sorry

end average_hit_targets_value_average_hit_targets_ge_half_l9_903
