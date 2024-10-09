import Mathlib

namespace find_hourly_rate_l426_42664

theorem find_hourly_rate (x : ℝ) (h1 : 40 * x + 10.75 * 16 = 622) : x = 11.25 :=
sorry

end find_hourly_rate_l426_42664


namespace triangle_A1B1C1_sides_l426_42614

theorem triangle_A1B1C1_sides
  (a b c x y z R : ℝ) 
  (h_positive_a : a > 0)
  (h_positive_b : b > 0)
  (h_positive_c : c > 0)
  (h_positive_x : x > 0)
  (h_positive_y : y > 0)
  (h_positive_z : z > 0)
  (h_positive_R : R > 0) :
  (↑a * ↑y / (2 * ↑R), ↑b * ↑z / (2 * ↑R), ↑c * ↑x / (2 * ↑R)) = (↑c * ↑x / (2 * ↑R), ↑a * ↑y / (2 * ↑R), ↑b * ↑z / (2 * ↑R)) :=
by sorry

end triangle_A1B1C1_sides_l426_42614


namespace work_equivalence_l426_42603

variable (m d r : ℕ)

theorem work_equivalence (h : d > 0) : (m * d) / (m + r^2) = d := sorry

end work_equivalence_l426_42603


namespace geometric_sequence_sum_n5_l426_42663

def geometric_sum (a₁ q : ℕ) (n : ℕ) : ℕ :=
  a₁ * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_sum_n5 (a₁ q : ℕ) (n : ℕ) (h₁ : a₁ = 3) (h₂ : q = 4) (h₃ : n = 5) : 
  geometric_sum a₁ q n = 1023 :=
by
  sorry

end geometric_sequence_sum_n5_l426_42663


namespace greatest_product_obtainable_l426_42698

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l426_42698


namespace price_of_candied_grape_l426_42697

theorem price_of_candied_grape (x : ℝ) (h : 15 * 2 + 12 * x = 48) : x = 1.5 :=
by
  sorry

end price_of_candied_grape_l426_42697


namespace train_speed_l426_42616

theorem train_speed
    (length_train : ℝ) (length_platform : ℝ) (time_seconds : ℝ)
    (h_train : length_train = 250)
    (h_platform : length_platform = 250.04)
    (h_time : time_seconds = 25) :
    (length_train + length_platform) / time_seconds * 3.6 = 72.006 :=
by sorry

end train_speed_l426_42616


namespace find_x_l426_42637

/--
Given the following conditions:
1. The sum of angles around a point is 360 degrees.
2. The angles are 7x, 6x, 3x, and (2x + y).
3. y = 2x.

Prove that x = 18 degrees.
-/
theorem find_x (x y : ℝ) (h : 18 * x + y = 360) (h_y : y = 2 * x) : x = 18 :=
by
  sorry

end find_x_l426_42637


namespace distance_interval_l426_42657

theorem distance_interval (d : ℝ) :
  (d < 8) ∧ (d > 7) ∧ (d > 5) ∧ (d ≠ 3) ↔ (7 < d ∧ d < 8) :=
by
  sorry

end distance_interval_l426_42657


namespace length_PZ_l426_42621

-- Define the given conditions
variables (CD WX : ℝ) -- segments CD and WX
variable (CW : ℝ) -- length of segment CW
variable (DP : ℝ) -- length of segment DP
variable (PX : ℝ) -- length of segment PX

-- Define the similarity condition
-- segment CD is parallel to segment WX implies that the triangles CDP and WXP are similar

-- Define what we want to prove
theorem length_PZ (hCD_WX_parallel : CD = WX)
                  (hCW : CW = 56)
                  (hDP : DP = 18)
                  (hPX : PX = 36) :
  ∃ PZ : ℝ, PZ = 4 / 3 :=
by
  -- proof steps here (omitted)
  sorry

end length_PZ_l426_42621


namespace area_excluding_hole_l426_42650

def area_large_rectangle (x : ℝ) : ℝ :=
  (2 * x + 9) * (x + 6)

def area_square_hole (x : ℝ) : ℝ :=
  (x - 1) * (x - 1)

theorem area_excluding_hole (x : ℝ) : 
  area_large_rectangle x - area_square_hole x = x^2 + 23 * x + 53 :=
by
  sorry

end area_excluding_hole_l426_42650


namespace A_n_squared_l426_42699

-- Define C(n-2)
def C_n_2 (n : ℕ) : ℕ := n * (n - 1) / 2

-- Define A_n_2
def A_n_2 (n : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - 2)

theorem A_n_squared (n : ℕ) (hC : C_n_2 n = 15) : A_n_2 n = 30 := by
  sorry

end A_n_squared_l426_42699


namespace joe_dropped_score_l426_42642

theorem joe_dropped_score (A B C D : ℕ) (h1 : (A + B + C + D) / 4 = 60) (h2 : (A + B + C) / 3 = 65) :
  min A (min B (min C D)) = D → D = 45 :=
by sorry

end joe_dropped_score_l426_42642


namespace hyperbola_equation_l426_42641

theorem hyperbola_equation
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (e : ℝ) (he : e = 2 * Real.sqrt 3 / 3)
  (dist_from_origin : ∀ A B : ℝ × ℝ, A = (0, -b) ∧ B = (a, 0) →
    abs (a * b) / Real.sqrt (a^2 + b^2) = Real.sqrt 3 / 2) :
  (a^2 = 3 ∧ b^2 = 1) → (∀ x y : ℝ, (x^2 / 3 - y^2 = 1)) := 
sorry

end hyperbola_equation_l426_42641


namespace ratio_of_female_to_male_members_l426_42607

theorem ratio_of_female_to_male_members 
  (f m : ℕ)
  (avg_age_female avg_age_male avg_age_membership : ℕ)
  (hf : avg_age_female = 35)
  (hm : avg_age_male = 30)
  (ha : avg_age_membership = 32)
  (h_avg : (35 * f + 30 * m) / (f + m) = 32) : 
  f / m = 2 / 3 :=
sorry

end ratio_of_female_to_male_members_l426_42607


namespace half_MN_correct_l426_42624

noncomputable def OM : ℝ × ℝ := (-2, 3)
noncomputable def ON : ℝ × ℝ := (-1, -5)
noncomputable def MN : ℝ × ℝ := (ON.1 - OM.1, ON.2 - OM.2)
noncomputable def half_MN : ℝ × ℝ := (MN.1 / 2, MN.2 / 2)

theorem half_MN_correct : half_MN = (1 / 2, -4) :=
by
  -- define the values of OM and ON
  let OM : ℝ × ℝ := (-2, 3)
  let ON : ℝ × ℝ := (-1, -5)
  -- calculate MN
  let MN : ℝ × ℝ := (ON.1 - OM.1, ON.2 - OM.2)
  -- calculate half of MN
  let half_MN : ℝ × ℝ := (MN.1 / 2, MN.2 / 2)
  -- assert the expected value
  exact sorry

end half_MN_correct_l426_42624


namespace hamsters_count_l426_42632

-- Define the conditions as parameters
variables (ratio_rabbit_hamster : ℕ × ℕ)
variables (rabbits : ℕ)
variables (hamsters : ℕ)

-- Given conditions
def ratio_condition : ratio_rabbit_hamster = (4, 5) := sorry
def rabbits_condition : rabbits = 20 := sorry

-- The theorem to be proven
theorem hamsters_count : ratio_rabbit_hamster = (4, 5) -> rabbits = 20 -> hamsters = 25 :=
by
  intro h1 h2
  sorry

end hamsters_count_l426_42632


namespace find_y_l426_42670

theorem find_y (x y : ℕ) (h₀ : x > 0) (h₁ : y > 0) (h₂ : ∃ q : ℕ, x = q * y + 9) (h₃ : x / y = 96 + 3 / 20) : y = 60 :=
sorry

end find_y_l426_42670


namespace total_books_l426_42617

def numberOfMysteryShelves := 6
def numberOfPictureShelves := 2
def booksPerShelf := 9

theorem total_books (hMystery : numberOfMysteryShelves = 6) 
                    (hPicture : numberOfPictureShelves = 2) 
                    (hBooksPerShelf : booksPerShelf = 9) :
  numberOfMysteryShelves * booksPerShelf + numberOfPictureShelves * booksPerShelf = 72 :=
  by 
  sorry

end total_books_l426_42617


namespace Emily_average_speed_l426_42658

noncomputable def Emily_run_distance : ℝ := 10

noncomputable def speed_first_uphill : ℝ := 4
noncomputable def distance_first_uphill : ℝ := 2

noncomputable def speed_first_downhill : ℝ := 6
noncomputable def distance_first_downhill : ℝ := 1

noncomputable def speed_flat_ground : ℝ := 5
noncomputable def distance_flat_ground : ℝ := 3

noncomputable def speed_second_uphill : ℝ := 4.5
noncomputable def distance_second_uphill : ℝ := 2

noncomputable def speed_second_downhill : ℝ := 6
noncomputable def distance_second_downhill : ℝ := 2

noncomputable def break_first : ℝ := 5 / 60
noncomputable def break_second : ℝ := 7 / 60
noncomputable def break_third : ℝ := 3 / 60

noncomputable def time_first_uphill : ℝ := distance_first_uphill / speed_first_uphill
noncomputable def time_first_downhill : ℝ := distance_first_downhill / speed_first_downhill
noncomputable def time_flat_ground : ℝ := distance_flat_ground / speed_flat_ground
noncomputable def time_second_uphill : ℝ := distance_second_uphill / speed_second_uphill
noncomputable def time_second_downhill : ℝ := distance_second_downhill / speed_second_downhill

noncomputable def total_running_time : ℝ := time_first_uphill + time_first_downhill + time_flat_ground + time_second_uphill + time_second_downhill
noncomputable def total_break_time : ℝ := break_first + break_second + break_third
noncomputable def total_time : ℝ := total_running_time + total_break_time

noncomputable def average_speed : ℝ := Emily_run_distance / total_time

theorem Emily_average_speed : abs (average_speed - 4.36) < 0.01 := by
  sorry

end Emily_average_speed_l426_42658


namespace number_of_strikers_l426_42678

theorem number_of_strikers 
  (goalies defenders midfielders strikers : ℕ) 
  (h1 : goalies = 3) 
  (h2 : defenders = 10) 
  (h3 : midfielders = 2 * defenders) 
  (h4 : goalies + defenders + midfielders + strikers = 40) : 
  strikers = 7 := 
sorry

end number_of_strikers_l426_42678


namespace bottles_have_200_mL_l426_42643

def liters_to_milliliters (liters : ℕ) : ℕ :=
  liters * 1000

def total_milliliters (liters : ℕ) : ℕ :=
  liters_to_milliliters liters

def milliliters_per_bottle (total_mL : ℕ) (num_bottles : ℕ) : ℕ :=
  total_mL / num_bottles

theorem bottles_have_200_mL (num_bottles : ℕ) (total_oil_liters : ℕ) (h1 : total_oil_liters = 4) (h2 : num_bottles = 20) :
  milliliters_per_bottle (total_milliliters total_oil_liters) num_bottles = 200 := 
by
  sorry

end bottles_have_200_mL_l426_42643


namespace tuna_per_customer_l426_42673

noncomputable def total_customers := 100
noncomputable def total_tuna := 10
noncomputable def weight_per_tuna := 200
noncomputable def customers_without_fish := 20

theorem tuna_per_customer : (total_tuna * weight_per_tuna) / (total_customers - customers_without_fish) = 25 := by
  sorry

end tuna_per_customer_l426_42673


namespace division_powers_5_half_division_powers_6_3_division_powers_formula_division_powers_combination_l426_42654

def f (n : ℕ) (a : ℚ) : ℚ := a ^ (2 - n)

theorem division_powers_5_half : f 5 (1/2) = 8 := by
  -- skip the proof
  sorry

theorem division_powers_6_3 : f 6 3 = 1/81 := by
  -- skip the proof
  sorry

theorem division_powers_formula (n : ℕ) (a : ℚ) (h : n > 0) : f n a = a^(2 - n) := by
  -- skip the proof
  sorry

theorem division_powers_combination : f 5 (1/3) * f 4 3 * f 5 (1/2) + f 5 (-1/4) / f 6 (-1/2) = 20 := by
  -- skip the proof
  sorry

end division_powers_5_half_division_powers_6_3_division_powers_formula_division_powers_combination_l426_42654


namespace distance_A_B_l426_42672

theorem distance_A_B 
  (perimeter_small_square : ℝ)
  (area_large_square : ℝ)
  (h1 : perimeter_small_square = 8)
  (h2 : area_large_square = 64) :
  let side_small_square := perimeter_small_square / 4
  let side_large_square := Real.sqrt area_large_square
  let horizontal_distance := side_small_square + side_large_square
  let vertical_distance := side_large_square - side_small_square
  let distance_AB := Real.sqrt (horizontal_distance^2 + vertical_distance^2)
  distance_AB = 11.7 :=
  by sorry

end distance_A_B_l426_42672


namespace largest_multiple_of_7_whose_negation_greater_than_neg80_l426_42685

theorem largest_multiple_of_7_whose_negation_greater_than_neg80 : ∃ (n : ℤ), n = 77 ∧ (∃ (k : ℤ), n = k * 7) ∧ (-n > -80) :=
by
  sorry

end largest_multiple_of_7_whose_negation_greater_than_neg80_l426_42685


namespace slips_with_3_l426_42667

theorem slips_with_3 (x : ℤ) 
    (h1 : 15 > 0) 
    (h2 : 3 > 0 ∧ 9 > 0) 
    (h3 : (3 * x + 9 * (15 - x)) / 15 = 5) : 
    x = 10 := 
sorry

end slips_with_3_l426_42667


namespace smaller_package_contains_correct_number_of_cupcakes_l426_42649

-- Define the conditions
def number_of_packs_large : ℕ := 4
def cupcakes_per_large_pack : ℕ := 15
def total_children : ℕ := 100
def needed_packs_small : ℕ := 4

-- Define the total cupcakes bought initially
def total_cupcakes_bought : ℕ := number_of_packs_large * cupcakes_per_large_pack

-- Define the total additional cupcakes needed
def additional_cupcakes_needed : ℕ := total_children - total_cupcakes_bought

-- Define the number of cupcakes per smaller package
def cupcakes_per_small_pack : ℕ := additional_cupcakes_needed / needed_packs_small

-- The theorem statement to prove
theorem smaller_package_contains_correct_number_of_cupcakes :
  cupcakes_per_small_pack = 10 :=
by
  -- This is where the proof would go
  sorry

end smaller_package_contains_correct_number_of_cupcakes_l426_42649


namespace paddington_more_goats_l426_42612

theorem paddington_more_goats (W P total : ℕ) (hW : W = 140) (hTotal : total = 320) (hTotalGoats : W + P = total) : P - W = 40 :=
by
  sorry

end paddington_more_goats_l426_42612


namespace correct_choice_for_games_l426_42687
  
-- Define the problem context
def games_preferred (question : String) (answer : String) :=
  question = "Which of the two computer games did you prefer?" ∧
  answer = "Actually I didn’t like either of them."

-- Define the proof that the correct choice is 'either of them'
theorem correct_choice_for_games (question : String) (answer : String) :
  games_preferred question answer → answer = "either of them" :=
by
  -- Provided statement and proof assumptions
  intro h
  cases h
  exact sorry -- Proof steps will be here
  -- Here, the conclusion should be derived from given conditions

end correct_choice_for_games_l426_42687


namespace total_rainfall_l426_42644

theorem total_rainfall
  (r₁ r₂ : ℕ)
  (T t₁ : ℕ)
  (H1 : r₁ = 30)
  (H2 : r₂ = 15)
  (H3 : T = 45)
  (H4 : t₁ = 20) :
  r₁ * t₁ + r₂ * (T - t₁) = 975 := by
  sorry

end total_rainfall_l426_42644


namespace total_missing_keys_l426_42620

theorem total_missing_keys :
  let total_vowels := 5
  let total_consonants := 21
  let missing_consonants := total_consonants / 7
  let missing_vowels := 2
  missing_consonants + missing_vowels = 5 :=
by {
  sorry
}

end total_missing_keys_l426_42620


namespace initial_investment_l426_42695

theorem initial_investment :
  ∃ x : ℝ, P = 705.03 ∧ r = 0.12 ∧ n = 5 ∧ P = x * (1 + r)^n ∧ x = 400 :=
by
  let P := 705.03
  let r := 0.12
  let n := 5
  use 400
  simp [P, r, n]
  sorry

end initial_investment_l426_42695


namespace terry_lunch_combo_l426_42666

theorem terry_lunch_combo :
  let lettuce_options : ℕ := 2
  let tomato_options : ℕ := 3
  let olive_options : ℕ := 4
  let soup_options : ℕ := 2
  (lettuce_options * tomato_options * olive_options * soup_options = 48) := 
by
  sorry

end terry_lunch_combo_l426_42666


namespace order_abc_l426_42693

noncomputable def a : ℝ := Real.log 0.8 / Real.log 0.7
noncomputable def b : ℝ := Real.log 0.9 / Real.log 1.1
noncomputable def c : ℝ := Real.exp (0.9 * Real.log 1.1)

theorem order_abc : b < a ∧ a < c := by
  sorry

end order_abc_l426_42693


namespace root_bounds_l426_42690

noncomputable def sqrt (r : ℝ) (n : ℕ) := r^(1 / n)

theorem root_bounds (a b c d : ℝ) (n p x y : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hn : 0 < n) (hp : 0 < p) (hx : 0 < x) (hy : 0 < y) :
  sqrt d y < sqrt (a * b * c * d) (n + p + x + y) ∧
  sqrt (a * b * c * d) (n + p + x + y) < sqrt a n := 
sorry

end root_bounds_l426_42690


namespace no_positive_integer_satisfies_l426_42676

theorem no_positive_integer_satisfies : ¬ ∃ n : ℕ, 0 < n ∧ (20 * n + 2) ∣ (2003 * n + 2002) :=
by sorry

end no_positive_integer_satisfies_l426_42676


namespace principal_argument_of_z_l426_42665

-- Mathematical definitions based on provided conditions
noncomputable def theta : ℝ := Real.arctan (5 / 12)

-- The complex number z defined in the problem
noncomputable def z : ℂ := (Real.cos (2 * theta) + Real.sin (2 * theta) * Complex.I) / (239 + Complex.I)

-- Lean statement to prove the argument of z
theorem principal_argument_of_z : Complex.arg z = Real.pi / 4 :=
by
  sorry

end principal_argument_of_z_l426_42665


namespace abs_diff_roots_quad_eq_l426_42688

theorem abs_diff_roots_quad_eq : 
  ∀ (r1 r2 : ℝ), 
  (r1 * r2 = 12) ∧ (r1 + r2 = 7) → |r1 - r2| = 1 :=
by
  intro r1 r2 h
  sorry

end abs_diff_roots_quad_eq_l426_42688


namespace solve_equation_l426_42622

theorem solve_equation (x : ℝ) (h : x ≠ 3) : (x + 6) / (x - 3) = 4 ↔ x = 6 :=
by
  sorry

end solve_equation_l426_42622


namespace eliza_received_12_almonds_l426_42655

theorem eliza_received_12_almonds (y : ℕ) (h1 : y - 8 = y / 3) : y = 12 :=
sorry

end eliza_received_12_almonds_l426_42655


namespace expected_score_two_free_throws_is_correct_l426_42602

noncomputable def expected_score_two_free_throws (p : ℝ) (n : ℕ) : ℝ :=
n * p

theorem expected_score_two_free_throws_is_correct : expected_score_two_free_throws 0.7 2 = 1.4 :=
by
  -- Proof will be written here.
  sorry

end expected_score_two_free_throws_is_correct_l426_42602


namespace point_outside_circle_l426_42639

theorem point_outside_circle (D E F x0 y0 : ℝ) (h : (x0 + D / 2)^2 + (y0 + E / 2)^2 > (D^2 + E^2 - 4 * F) / 4) :
  x0^2 + y0^2 + D * x0 + E * y0 + F > 0 :=
sorry

end point_outside_circle_l426_42639


namespace MathContestMeanMedianDifference_l426_42627

theorem MathContestMeanMedianDifference :
  (15 / 100 * 65 + 20 / 100 * 85 + 40 / 100 * 95 + 25 / 100 * 110) - 95 = -3 := 
by
  sorry

end MathContestMeanMedianDifference_l426_42627


namespace Daniel_had_more_than_200_marbles_at_day_6_l426_42648

noncomputable def marbles (k : ℕ) : ℕ :=
  5 * 2^k

theorem Daniel_had_more_than_200_marbles_at_day_6 :
  ∃ k : ℕ, marbles k > 200 ∧ ∀ m < k, marbles m ≤ 200 :=
by
  sorry

end Daniel_had_more_than_200_marbles_at_day_6_l426_42648


namespace area_of_triangle_ABC_sinA_value_l426_42674

noncomputable def cosC := 3 / 4
noncomputable def sinC := Real.sqrt (1 - cosC ^ 2)
noncomputable def a := 1
noncomputable def b := 2
noncomputable def c := Real.sqrt (a ^ 2 + b ^ 2 - 2 * a * b * cosC)
noncomputable def area := (1 / 2) * a * b * sinC
noncomputable def sinA := (a * sinC) / c

theorem area_of_triangle_ABC : area = Real.sqrt 7 / 4 :=
by sorry

theorem sinA_value : sinA = Real.sqrt 14 / 8 :=
by sorry

end area_of_triangle_ABC_sinA_value_l426_42674


namespace area_ratio_XYZ_PQR_l426_42605

theorem area_ratio_XYZ_PQR 
  (PR PQ QR : ℝ)
  (p q r : ℝ) 
  (hPR : PR = 15) 
  (hPQ : PQ = 20) 
  (hQR : QR = 25)
  (hPX : p * PR = PR * p)
  (hQY : q * QR = QR * q) 
  (hPZ : r * PQ = PQ * r) 
  (hpq_sum : p + q + r = 3 / 4) 
  (hpq_sq_sum : p^2 + q^2 + r^2 = 9 / 16) : 
  (area_triangle_XYZ / area_triangle_PQR = 1 / 4) :=
sorry

end area_ratio_XYZ_PQR_l426_42605


namespace total_regular_and_diet_soda_bottles_l426_42635

-- Definitions from the conditions
def regular_soda_bottles := 49
def diet_soda_bottles := 40

-- The statement to prove
theorem total_regular_and_diet_soda_bottles :
  regular_soda_bottles + diet_soda_bottles = 89 :=
by
  sorry

end total_regular_and_diet_soda_bottles_l426_42635


namespace robin_cut_hair_l426_42606

-- Definitions as per the given conditions
def initial_length := 17
def current_length := 13

-- Statement of the proof problem
theorem robin_cut_hair : initial_length - current_length = 4 := 
by 
  sorry

end robin_cut_hair_l426_42606


namespace distinct_pairs_count_l426_42625

theorem distinct_pairs_count :
  ∃ (n : ℕ), n = 2 ∧ 
  ∀ (x y : ℝ), (x = 3 * x^2 + y^2) ∧ (y = 3 * x * y) → 
    ((x = 0 ∧ y = 0) ∨ (x = 1 / 3 ∧ y = 0)) :=
by
  sorry

end distinct_pairs_count_l426_42625


namespace sum_of_four_consecutive_even_numbers_l426_42669

theorem sum_of_four_consecutive_even_numbers (n : ℤ) (h : n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 = 344) :
  n + (n + 2) + (n + 4) + (n + 6) = 36 := sorry

end sum_of_four_consecutive_even_numbers_l426_42669


namespace sufficient_condition_for_equation_l426_42646

theorem sufficient_condition_for_equation (x y z : ℤ) (h1 : x = y + 1) (h2 : z = y) :
    x * (x - y) + y * (y - z) + z * (z - x) = 1 :=
by
  -- Proof omitted
  sorry

end sufficient_condition_for_equation_l426_42646


namespace jane_rejects_percent_l426_42686

theorem jane_rejects_percent :
  -- Declare the conditions as hypotheses
  ∀ (P : ℝ) (J : ℝ) (john_frac_reject : ℝ) (total_reject_percent : ℝ) (jane_inspect_frac : ℝ),
  john_frac_reject = 0.005 →
  total_reject_percent = 0.0075 →
  jane_inspect_frac = 5 / 6 →
  -- Given the rejection equation
  (john_frac_reject * (1 / 6) * P + (J / 100) * jane_inspect_frac * P = total_reject_percent * P) →
  -- Prove that Jane rejected 0.8% of the products she inspected
  J = 0.8 :=
by {
  sorry
}

end jane_rejects_percent_l426_42686


namespace degree_of_k_l426_42629

open Polynomial

theorem degree_of_k (h k : Polynomial ℝ) 
  (h_def : h = -5 * X^5 + 4 * X^3 - 2 * X^2 + C 8)
  (deg_sum : (h + k).degree = 2) : k.degree = 5 :=
sorry

end degree_of_k_l426_42629


namespace range_of_x_l426_42615

noncomputable def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
noncomputable def specific_function (f : ℝ → ℝ) := ∀ x : ℝ, x ≥ 0 → f x = 2^x

theorem range_of_x (f : ℝ → ℝ)  
  (hf_even : even_function f) 
  (hf_specific : specific_function f) : {x : ℝ | f (1 - 2 * x) < f 3} = {x : ℝ | -1 < x ∧ x < 2} := 
by
  sorry

end range_of_x_l426_42615


namespace rectangle_perimeter_l426_42633

variables (L B P : ℝ)

theorem rectangle_perimeter (h1 : B = 0.60 * L) (h2 : L * B = 37500) : P = 800 :=
by
  sorry

end rectangle_perimeter_l426_42633


namespace root_relationship_specific_root_five_l426_42652

def f (x : ℝ) : ℝ := x^3 - 6 * x^2 - 39 * x - 10
def g (x : ℝ) : ℝ := x^3 + x^2 - 20 * x - 50

theorem root_relationship :
  ∃ (x_0 : ℝ), g x_0 = 0 ∧ f (2 * x_0) = 0 :=
sorry

theorem specific_root_five :
  g 5 = 0 ∧ f 10 = 0 :=
sorry

end root_relationship_specific_root_five_l426_42652


namespace blue_dress_difference_l426_42647

theorem blue_dress_difference 
(total_space : ℕ)
(red_dresses : ℕ)
(blue_dresses : ℕ)
(h1 : total_space = 200)
(h2 : red_dresses = 83)
(h3 : blue_dresses = total_space - red_dresses) :
blue_dresses - red_dresses = 34 :=
by
  rw [h1, h2] at h3
  sorry -- Proof details go here.

end blue_dress_difference_l426_42647


namespace maximize_cubic_quartic_l426_42619

theorem maximize_cubic_quartic (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + 2 * y = 35) : 
  (x, y) = (21, 7) ↔ x^3 * y^4 = (21:ℝ)^3 * (7:ℝ)^4 := 
by
  sorry

end maximize_cubic_quartic_l426_42619


namespace point_P_path_length_l426_42662

/-- A rectangle PQRS in the plane with points P Q R S, where PQ = RS = 2 and QR = SP = 6. 
    The rectangle is rotated 90 degrees twice: first about point R and then 
    about the new position of point S after the first rotation. 
    The goal is to prove that the length of the path P travels is (3 + sqrt 10) * pi. -/
theorem point_P_path_length :
  ∀ (P Q R S : ℝ × ℝ), 
    dist P Q = 2 ∧ dist Q R = 6 ∧ dist R S = 2 ∧ dist S P = 6 →
    ∃ path_length : ℝ, path_length = (3 + Real.sqrt 10) * Real.pi :=
by
  sorry

end point_P_path_length_l426_42662


namespace cockatiel_weekly_consumption_is_50_l426_42638

def boxes_bought : ℕ := 3
def boxes_existing : ℕ := 5
def grams_per_box : ℕ := 225
def parrot_weekly_consumption : ℕ := 100
def weeks_supply : ℕ := 12

def total_boxes : ℕ := boxes_bought + boxes_existing
def total_birdseed_grams : ℕ := total_boxes * grams_per_box
def parrot_total_consumption : ℕ := parrot_weekly_consumption * weeks_supply
def cockatiel_total_consumption : ℕ := total_birdseed_grams - parrot_total_consumption
def cockatiel_weekly_consumption : ℕ := cockatiel_total_consumption / weeks_supply

theorem cockatiel_weekly_consumption_is_50 :
  cockatiel_weekly_consumption = 50 := by
  -- Proof goes here
  sorry

end cockatiel_weekly_consumption_is_50_l426_42638


namespace set_union_eq_l426_42636

open Set

noncomputable def A : Set ℤ := {x | x^2 - x = 0}
def B : Set ℤ := {-1, 0}
def C : Set ℤ := {-1, 0, 1}

theorem set_union_eq :
  A ∪ B = C :=
by {
  sorry
}

end set_union_eq_l426_42636


namespace maria_trip_time_l426_42683

theorem maria_trip_time 
(s_highway : ℕ) (s_mountain : ℕ) (d_highway : ℕ) (d_mountain : ℕ) (t_mountain : ℕ) (t_break : ℕ) : 
  (s_highway = 4 * s_mountain) -> 
  (t_mountain = d_mountain / s_mountain) -> 
  t_mountain = 40 -> 
  t_break = 15 -> 
  d_highway = 100 -> 
  d_mountain = 20 ->
  s_mountain = d_mountain / t_mountain -> 
  s_highway = 4 * s_mountain -> 
  d_highway / s_highway = 50 ->
  40 + 50 + 15 = 105 := 
by 
  sorry

end maria_trip_time_l426_42683


namespace mass_of_man_l426_42682

def density_of_water : ℝ := 1000  -- kg/m³
def boat_length : ℝ := 4  -- meters
def boat_breadth : ℝ := 2  -- meters
def sinking_depth : ℝ := 0.01  -- meters (1 cm)

theorem mass_of_man
  (V : ℝ := boat_length * boat_breadth * sinking_depth)
  (m : ℝ := V * density_of_water) :
  m = 80 :=
by
  sorry

end mass_of_man_l426_42682


namespace find_number_l426_42604

-- Define the problem statement
theorem find_number (n : ℕ) (h : (n + 2 * n + 3 * n + 4 * n + 5 * n) / 5 = 27) : n = 9 :=
sorry

end find_number_l426_42604


namespace add_to_both_num_and_denom_l426_42610

theorem add_to_both_num_and_denom (n : ℕ) : (4 + n) / (7 + n) = 7 / 8 ↔ n = 17 := by
  sorry

end add_to_both_num_and_denom_l426_42610


namespace max_sum_factors_of_60_exists_max_sum_factors_of_60_l426_42671

theorem max_sum_factors_of_60 (d Δ : ℕ) (h : d * Δ = 60) : (d + Δ) ≤ 61 :=
sorry

theorem exists_max_sum_factors_of_60 : ∃ d Δ : ℕ, d * Δ = 60 ∧ d + Δ = 61 :=
sorry

end max_sum_factors_of_60_exists_max_sum_factors_of_60_l426_42671


namespace fill_pool_time_l426_42668

theorem fill_pool_time (R : ℝ) (T : ℝ) (hSlowerPipe : R = 1 / 9) (hFasterPipe : 1.25 * R = 1.25 / 9)
                     (hCombinedRate : 2.25 * R = 2.25 / 9) : T = 4 := by
  sorry

end fill_pool_time_l426_42668


namespace total_length_of_visible_edges_l426_42631

theorem total_length_of_visible_edges (shortest_side : ℕ) (removed_side : ℕ) (longest_side : ℕ) (new_visible_sides_sum : ℕ) 
  (h1 : shortest_side = 4) 
  (h2 : removed_side = 2 * shortest_side) 
  (h3 : removed_side = longest_side / 2) 
  (h4 : longest_side = 16) 
  (h5 : new_visible_sides_sum = shortest_side + removed_side + removed_side) : 
  new_visible_sides_sum = 20 := by 
sorry

end total_length_of_visible_edges_l426_42631


namespace total_games_friends_l426_42661

def new_friends_games : ℕ := 88
def old_friends_games : ℕ := 53

theorem total_games_friends :
  new_friends_games + old_friends_games = 141 :=
by
  sorry

end total_games_friends_l426_42661


namespace correct_option_l426_42689

-- Definitions based on conditions
def sentence_structure : String := "He’s never interested in what ______ is doing."

def option_A : String := "no one else"
def option_B : String := "anyone else"
def option_C : String := "someone else"
def option_D : String := "nobody else"

-- The proof statement
theorem correct_option : option_B = "anyone else" := by
  sorry

end correct_option_l426_42689


namespace right_triangle_midpoints_distances_l426_42675

theorem right_triangle_midpoints_distances (a b : ℝ) 
  (hXON : 19^2 = a^2 + (b/2)^2)
  (hYOM : 22^2 = b^2 + (a/2)^2) :
  a^2 + b^2 = 676 :=
by
  sorry

end right_triangle_midpoints_distances_l426_42675


namespace sum_of_coefficients_l426_42680

theorem sum_of_coefficients (a a_1 a_2 a_3 a_4 a_5 a_6 : ℤ) :
  (∀ x : ℤ, (1 + x)^6 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) →
  a = 1 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 63 :=
by
  intros h ha
  sorry

end sum_of_coefficients_l426_42680


namespace sum_of_fractions_l426_42659

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l426_42659


namespace Mina_additional_miles_l426_42656

theorem Mina_additional_miles:
  let distance1 := 20 -- distance in miles for the first part of the trip
  let speed1 := 40 -- speed in mph for the first part of the trip
  let speed2 := 60 -- speed in mph for the second part of the trip
  let avg_speed := 55 -- average speed needed for the entire trip in mph
  let distance2 := (distance1 / speed1 + (avg_speed * (distance1 / speed1)) / (speed1 - avg_speed * speed1 / speed2)) * speed2 -- formula to find the additional distance
  distance2 = 90 :=
by {
  sorry
}

end Mina_additional_miles_l426_42656


namespace randy_final_amount_l426_42609

-- Conditions as definitions
def initial_dollars : ℝ := 30
def initial_euros : ℝ := 20
def lunch_cost : ℝ := 10
def ice_cream_percentage : ℝ := 0.25
def snack_percentage : ℝ := 0.10
def conversion_rate : ℝ := 0.85

-- Main proof statement without the proof body
theorem randy_final_amount :
  let euros_in_dollars := initial_euros / conversion_rate
  let total_dollars := initial_dollars + euros_in_dollars
  let dollars_after_lunch := total_dollars - lunch_cost
  let ice_cream_cost := dollars_after_lunch * ice_cream_percentage
  let dollars_after_ice_cream := dollars_after_lunch - ice_cream_cost
  let snack_euros := initial_euros * snack_percentage
  let snack_dollars := snack_euros / conversion_rate
  let final_dollars := dollars_after_ice_cream - snack_dollars
  final_dollars = 30.30 :=
by
  sorry

end randy_final_amount_l426_42609


namespace triangle_properties_l426_42677

open Real

variables (A B C a b c : ℝ) (triangle_obtuse triangle_right triangle_acute : Prop)

-- Declaration of properties 
def sin_gt (A B : ℝ) := sin A > sin B
def tan_product_lt (A C : ℝ) := tan A * tan C < 1
def cos_squared_eq (A B C : ℝ) := cos A ^ 2 + cos B ^ 2 - cos C ^ 2 = 1

theorem triangle_properties :
  (sin_gt A B → A > B) ∧
  (triangle_obtuse → tan_product_lt A C) ∧
  (cos_squared_eq A B C → triangle_right) :=
  by sorry

end triangle_properties_l426_42677


namespace robie_initial_cards_l426_42611

def total_initial_boxes : Nat := 2 + 5
def cards_per_box : Nat := 10
def unboxed_cards : Nat := 5

theorem robie_initial_cards :
  (total_initial_boxes * cards_per_box + unboxed_cards) = 75 :=
by
  sorry

end robie_initial_cards_l426_42611


namespace factor_expression_l426_42696

theorem factor_expression (a b : ℕ) (h_factor : (x - a) * (x - b) = x^2 - 18 * x + 72) (h_nonneg : 0 ≤ a ∧ 0 ≤ b) (h_order : a > b) : 4 * b - a = 27 := by
  sorry

end factor_expression_l426_42696


namespace broken_seashells_count_l426_42645

def total_seashells : Nat := 6
def unbroken_seashells : Nat := 2
def broken_seashells : Nat := total_seashells - unbroken_seashells

theorem broken_seashells_count :
  broken_seashells = 4 :=
by
  -- The proof would go here, but for now, we use 'sorry' to denote it.
  sorry

end broken_seashells_count_l426_42645


namespace necessary_but_not_sufficient_condition_l426_42608

theorem necessary_but_not_sufficient_condition :
  (∀ x : ℝ, x = 1 → x^2 - 3 * x + 2 = 0) ∧ (∃ x : ℝ, x^2 - 3 * x + 2 = 0 ∧ x ≠ 1) :=
by
  sorry

end necessary_but_not_sufficient_condition_l426_42608


namespace time_first_tap_to_fill_cistern_l426_42613

-- Defining the conditions
axiom second_tap_empty_time : ℝ
axiom combined_tap_fill_time : ℝ
axiom second_tap_rate : ℝ
axiom combined_tap_rate : ℝ

-- Specifying the given conditions
def problem_conditions :=
  second_tap_empty_time = 8 ∧
  combined_tap_fill_time = 8 ∧
  second_tap_rate = 1 / 8 ∧
  combined_tap_rate = 1 / 8

-- Defining the problem statement
theorem time_first_tap_to_fill_cistern :
  problem_conditions →
  (∃ T : ℝ, (1 / T - 1 / 8 = 1 / 8) ∧ T = 4) :=
by
  intro h
  sorry

end time_first_tap_to_fill_cistern_l426_42613


namespace sum_of_first_six_terms_l426_42626

theorem sum_of_first_six_terms 
  (a₁ : ℝ) 
  (r : ℝ) 
  (h_ratio : r = 2) 
  (h_sum_three : a₁ + 2*a₁ + 4*a₁ = 3) 
  : a₁ * (r^6 - 1) / (r - 1) = 27 := 
by {
  sorry
}

end sum_of_first_six_terms_l426_42626


namespace smallest_positive_integer_form_3003_55555_l426_42600

theorem smallest_positive_integer_form_3003_55555 :
  ∃ (m n : ℤ), 3003 * m + 55555 * n = 57 :=
by {
  sorry
}

end smallest_positive_integer_form_3003_55555_l426_42600


namespace min_value_of_expression_l426_42618

theorem min_value_of_expression (a b : ℝ) (h1 : 1 < a) (h2 : 0 < b) (h3 : a + 2 * b = 2) : 
  4 * (1 + Real.sqrt 2) ≤ (2 / (a - 1) + a / b) :=
by
  sorry

end min_value_of_expression_l426_42618


namespace chinese_characters_digits_l426_42651

theorem chinese_characters_digits:
  ∃ (a b g s t : ℕ), -- Chinese characters represented by digits
    -- Different characters represent different digits
    a ≠ b ∧ a ≠ g ∧ a ≠ s ∧ a ≠ t ∧
    b ≠ g ∧ b ≠ s ∧ b ≠ t ∧
    g ≠ s ∧ g ≠ t ∧
    s ≠ t ∧
    -- Equation: 业步高 * 业步高 = 高升抬步高
    (a * 100 + b * 10 + g) * (a * 100 + b * 10 + g) = (g * 10000 + s * 1000 + t * 100 + b * 10 + g) :=
by {
  -- We need to prove that the number represented by "高升抬步高" is 50625.
  sorry
}

end chinese_characters_digits_l426_42651


namespace sum_x_coordinates_intersection_mod_9_l426_42630

theorem sum_x_coordinates_intersection_mod_9 :
  ∃ x y : ℤ, (y ≡ 3 * x + 4 [ZMOD 9]) ∧ (y ≡ 7 * x + 2 [ZMOD 9]) ∧ x ≡ 5 [ZMOD 9] := sorry

end sum_x_coordinates_intersection_mod_9_l426_42630


namespace sphere_volume_from_area_l426_42601

/-- Given the surface area of a sphere is 24π, prove that the volume of the sphere is 8√6π. -/ 
theorem sphere_volume_from_area :
  ∀ {R : ℝ},
    4 * Real.pi * R^2 = 24 * Real.pi →
    (4 / 3) * Real.pi * R^3 = 8 * Real.sqrt 6 * Real.pi :=
by
  intro R h
  sorry

end sphere_volume_from_area_l426_42601


namespace compute_fraction_mul_l426_42684

theorem compute_fraction_mul :
  (1 / 3) ^ 2 * (1 / 8) = 1 / 72 :=
by
  sorry

end compute_fraction_mul_l426_42684


namespace greatest_two_digit_product_is_12_l426_42634

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end greatest_two_digit_product_is_12_l426_42634


namespace correct_translation_of_tradition_l426_42660

def is_adjective (s : String) : Prop :=
  s = "传统的"

def is_correct_translation (s : String) (translation : String) : Prop :=
  s = "传统的" → translation = "traditional"

theorem correct_translation_of_tradition : 
  is_adjective "传统的" ∧ is_correct_translation "传统的" "traditional" :=
by
  sorry

end correct_translation_of_tradition_l426_42660


namespace geometric_sequence_a6_l426_42653

noncomputable def a_sequence (n : ℕ) : ℝ := 1 * 2^(n-1)

theorem geometric_sequence_a6 (S : ℕ → ℝ)
  (h1 : S 10 = 3 * S 5)
  (h2 : ∀ n, S n = (1 - 2^n) / (1 - 2))
  (h3 : a_sequence 1 = 1) :
  a_sequence 6 = 2 := by
  sorry

end geometric_sequence_a6_l426_42653


namespace entree_cost_14_l426_42640

-- Define the conditions as given in part a)
def total_cost (e d : ℕ) : Prop := e + d = 23
def entree_more (e d : ℕ) : Prop := e = d + 5

-- The theorem to be proved
theorem entree_cost_14 (e d : ℕ) (h1 : total_cost e d) (h2 : entree_more e d) : e = 14 := 
by 
  sorry

end entree_cost_14_l426_42640


namespace alex_avg_speed_l426_42691

theorem alex_avg_speed (v : ℝ) : 
  (4.5 * v + 2.5 * 12 + 1.5 * 24 + 8 = 164) → v = 20 := 
by 
  intro h
  sorry

end alex_avg_speed_l426_42691


namespace percentage_sales_tax_on_taxable_purchases_l426_42679

-- Definitions
def total_cost : ℝ := 30
def tax_free_cost : ℝ := 24.7
def tax_rate : ℝ := 0.06

-- Statement to prove
theorem percentage_sales_tax_on_taxable_purchases :
  (tax_rate * (total_cost - tax_free_cost)) / total_cost * 100 = 1 := by
  sorry

end percentage_sales_tax_on_taxable_purchases_l426_42679


namespace regular_polygon_perimeter_l426_42692

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l426_42692


namespace total_amount_earned_l426_42623

-- Definitions of the conditions.
def work_done_per_day (days : ℕ) : ℚ := 1 / days

def total_work_done_per_day : ℚ :=
  work_done_per_day 6 + work_done_per_day 8 + work_done_per_day 12

def b_share : ℚ := work_done_per_day 8

def total_amount (b_earnings : ℚ) : ℚ := b_earnings * (total_work_done_per_day / b_share)

-- Main theorem stating that the total amount earned is $1170 if b's share is $390.
theorem total_amount_earned (h_b : b_share * 390 = 390) : total_amount 390 = 1170 := by sorry

end total_amount_earned_l426_42623


namespace find_t_l426_42681

variable (a t : ℝ)

def f (x : ℝ) : ℝ := a * x + 19

theorem find_t (h1 : f a 3 = 7) (h2 : f a t = 15) : t = 1 :=
by
  sorry

end find_t_l426_42681


namespace total_matches_played_l426_42694

-- Definitions
def victories_points := 3
def draws_points := 1
def defeats_points := 0
def points_after_5_games := 8
def games_played := 5
def target_points := 40
def remaining_wins_required := 9

-- Statement to prove
theorem total_matches_played :
  ∃ M : ℕ, points_after_5_games + victories_points * remaining_wins_required < target_points -> M = games_played + remaining_wins_required + 1 :=
sorry

end total_matches_played_l426_42694


namespace cheap_feed_amount_l426_42628

theorem cheap_feed_amount (x y : ℝ) (h1 : x + y = 27) (h2 : 0.17 * x + 0.36 * y = 7.02) : 
  x = 14.21 :=
sorry

end cheap_feed_amount_l426_42628
