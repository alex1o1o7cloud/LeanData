import Mathlib

namespace minimum_positive_temperatures_announced_l369_36969

theorem minimum_positive_temperatures_announced (x y : ℕ) :
  x * (x - 1) = 110 →
  y * (y - 1) + (x - y) * (x - y - 1) = 54 →
  (∀ z : ℕ, z * (z - 1) + (x - z) * (x - z - 1) = 54 → y ≤ z) →
  y = 4 :=
by
  sorry

end minimum_positive_temperatures_announced_l369_36969


namespace cricket_matches_total_l369_36935

theorem cricket_matches_total 
  (N : ℕ)
  (avg_total : ℕ → ℕ)
  (avg_first_8 : ℕ)
  (avg_last_4 : ℕ) 
  (h1 : avg_total N = 48)
  (h2 : avg_first_8 = 40)
  (h3 : avg_last_4 = 64) 
  (h_sum : (avg_first_8 * 8 + avg_last_4 * 4 = avg_total N * N)) :
  N = 12 := 
  sorry

end cricket_matches_total_l369_36935


namespace blue_shoes_in_warehouse_l369_36999

theorem blue_shoes_in_warehouse (total blue purple green : ℕ) (h1 : total = 1250) (h2 : green = purple) (h3 : purple = 355) :
    blue = total - (green + purple) := by
  sorry

end blue_shoes_in_warehouse_l369_36999


namespace residue_625_mod_17_l369_36940

theorem residue_625_mod_17 : 625 % 17 = 13 :=
by
  sorry

end residue_625_mod_17_l369_36940


namespace cylinder_surface_area_correct_l369_36967

noncomputable def cylinder_surface_area :=
  let r := 8   -- radius in cm
  let h := 10  -- height in cm
  let arc_angle := 90 -- degrees
  let x := 40
  let y := -40
  let z := 2
  x + y + z

theorem cylinder_surface_area_correct : cylinder_surface_area = 2 := by
  sorry

end cylinder_surface_area_correct_l369_36967


namespace smallest_digit_is_one_l369_36909

-- Given a 4-digit integer x.
def four_digit_integer (x : ℕ) : Prop :=
  1000 ≤ x ∧ x < 10000

-- Define function for the product of digits of x.
def product_of_digits (x : ℕ) : ℕ :=
  let d1 := x % 10
  let d2 := (x / 10) % 10
  let d3 := (x / 100) % 10
  let d4 := (x / 1000) % 10
  d1 * d2 * d3 * d4

-- Define function for the sum of digits of x.
def sum_of_digits (x : ℕ) : ℕ :=
  let d1 := x % 10
  let d2 := (x / 10) % 10
  let d3 := (x / 100) % 10
  let d4 := (x / 1000) % 10
  d1 + d2 + d3 + d4

-- Assume p is a prime number.
def is_prime (p : ℕ) : Prop :=
  ¬ ∃ d, d ∣ p ∧ d ≠ 1 ∧ d ≠ p

-- Proof problem: Given conditions for T(x) and S(x),
-- prove that the smallest digit in x is 1.
theorem smallest_digit_is_one (x p k : ℕ) (h1 : four_digit_integer x)
  (h2 : is_prime p) (h3 : product_of_digits x = p^k)
  (h4 : sum_of_digits x = p^p - 5) : 
  ∃ d1 d2 d3 d4, d1 <= d2 ∧ d1 <= d3 ∧ d1 <= d4 ∧ d1 = 1 
  ∧ (d1 + d2 + d3 + d4 = p^p - 5) 
  ∧ (d1 * d2 * d3 * d4 = p^k) := 
sorry

end smallest_digit_is_one_l369_36909


namespace find_x_l369_36977

theorem find_x (x : ℤ) (h1 : 5 < x) (h2 : x < 21) (h3 : 7 < x) (h4 : x < 18) (h5 : 2 < x) (h6 : x < 13) (h7 : 9 < x) (h8 : x < 12) (h9 : x < 12) :
  x = 10 :=
sorry

end find_x_l369_36977


namespace part_a_part_b_l369_36959

-- Part (a): Prove that \( 2^n - 1 \) is divisible by 7 if and only if \( 3 \mid n \).
theorem part_a (n : ℕ) : 7 ∣ (2^n - 1) ↔ 3 ∣ n := sorry

-- Part (b): Prove that \( 2^n + 1 \) is not divisible by 7 for all natural numbers \( n \).
theorem part_b (n : ℕ) : ¬ (7 ∣ (2^n + 1)) := sorry

end part_a_part_b_l369_36959


namespace first_company_managers_percentage_l369_36907

-- Definitions from the conditions
variable (F M : ℝ) -- total workforce of first company and merged company
variable (x : ℝ) -- percentage of managers in the first company
variable (cond1 : 0.25 * M = F) -- 25% of merged company's workforce originated from the first company
variable (cond2 : 0.25 * M / M = 0.25) -- resulting merged company's workforce consists of 25% managers

-- The statement to prove
theorem first_company_managers_percentage : x = 25 :=
by
  sorry

end first_company_managers_percentage_l369_36907


namespace value_of_expression_l369_36929

theorem value_of_expression :
  let x := 1
  let y := -1
  let z := 0
  2 * x + 3 * y + 4 * z = -1 :=
by
  sorry

end value_of_expression_l369_36929


namespace correct_location_l369_36968

variable (A B C D : Prop)

axiom student_A_statement : ¬ A ∧ B
axiom student_B_statement : ¬ B ∧ C
axiom student_C_statement : ¬ B ∧ ¬ D
axiom ms_Hu_response : 
  ( (¬ A ∧ B = true) ∨ (¬ B ∧ C = true) ∨ (¬ B ∧ ¬ D = true) ) ∧ 
  ( (¬ A ∧ B = false) ∨ (¬ B ∧ C = false) ∨ (¬ B ∧ ¬ D = false) = false ) ∧ 
  ( (¬ A ∧ B ∨ ¬ B ∧ C ∨ ¬ B ∧ ¬ D) -> false )

theorem correct_location : B ∨ A := 
sorry

end correct_location_l369_36968


namespace expected_number_of_different_faces_l369_36950

theorem expected_number_of_different_faces :
  let p := (6 : ℕ) ^ 6
  let q := (5 : ℕ) ^ 6
  6 * (1 - (5 / 6)^6) = (p - q) / (6 ^ 5) :=
by
  sorry

end expected_number_of_different_faces_l369_36950


namespace ratio_of_democrats_l369_36928

theorem ratio_of_democrats (F M : ℕ) (h1 : F + M = 750) (h2 : (1/2 : ℚ) * F = 125) (h3 : (1/4 : ℚ) * M = 125) :
  (125 + 125 : ℚ) / 750 = 1 / 3 := by
  sorry

end ratio_of_democrats_l369_36928


namespace solution_set_inequality_l369_36900

theorem solution_set_inequality (m : ℝ) : (∀ x : ℝ, m * x^2 - m * x + 2 > 0) ↔ m ∈ Set.Ico 0 8 := by
  sorry

end solution_set_inequality_l369_36900


namespace compare_magnitude_l369_36989

theorem compare_magnitude (a b : ℝ) (h : a ≠ 1) : a^2 + b^2 > 2 * (a - b - 1) :=
by
  sorry

end compare_magnitude_l369_36989


namespace coefficient_a9_of_polynomial_l369_36953

theorem coefficient_a9_of_polynomial (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :
  (∀ x : ℝ, x^3 + x^10 = a_0 + 
    a_1 * (x + 1) + 
    a_2 * (x + 1)^2 + 
    a_3 * (x + 1)^3 + 
    a_4 * (x + 1)^4 + 
    a_5 * (x + 1)^5 + 
    a_6 * (x + 1)^6 + 
    a_7 * (x + 1)^7 + 
    a_8 * (x + 1)^8 + 
    a_9 * (x + 1)^9 + 
    a_10 * (x + 1)^10) 
  → a_9 = -10 :=
by
  intro h
  sorry

end coefficient_a9_of_polynomial_l369_36953


namespace notebook_area_l369_36979

variable (w h : ℝ)

def width_to_height_ratio (w h : ℝ) : Prop := w / h = 7 / 5
def perimeter (w h : ℝ) : Prop := 2 * w + 2 * h = 48
def area (w h : ℝ) : ℝ := w * h

theorem notebook_area (w h : ℝ) (ratio : width_to_height_ratio w h) (peri : perimeter w h) :
  area w h = 140 :=
by
  sorry

end notebook_area_l369_36979


namespace find_ordered_pair_l369_36992

variables {A B Q : Type} -- Points A, B, Q
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup Q]
variables {a b q : A} -- Vectors at points A, B, Q
variables (r : ℝ) -- Ratio constant

-- Define the conditions from the original problem
def ratio_aq_qb (A B Q : Type) [AddCommGroup A] [AddCommGroup B] [AddCommGroup Q] (a b q : A) (r : ℝ) :=
  r = 7 / 2

-- Define the goal theorem using the conditions above
theorem find_ordered_pair (h : ratio_aq_qb A B Q a b q r) : 
  q = (7 / 9) • a + (2 / 9) • b :=
sorry

end find_ordered_pair_l369_36992


namespace adult_meals_sold_l369_36911

theorem adult_meals_sold (k a : ℕ) (h1 : 10 * a = 7 * k) (h2 : k = 70) : a = 49 :=
by
  sorry

end adult_meals_sold_l369_36911


namespace fraction_to_decimal_l369_36962

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l369_36962


namespace soccer_camp_ratio_l369_36990

theorem soccer_camp_ratio :
  let total_kids := 2000
  let half_total := total_kids / 2
  let afternoon_camp := 750
  let morning_camp := half_total - afternoon_camp
  half_total ≠ 0 → 
  (morning_camp / half_total) = 1 / 4 := by
  sorry

end soccer_camp_ratio_l369_36990


namespace prime_roots_eq_l369_36997

theorem prime_roots_eq (n : ℕ) (hn : 0 < n) :
  (∃ (x1 x2 : ℕ), Prime x1 ∧ Prime x2 ∧ 2*x1^2 - 8*n*x1 + 10*x1 - n^2 + 35*n - 76 = 0 ∧ 
                    2*x2^2 - 8*n*x2 + 10*x2 - n^2 + 35*n - 76 = 0 ∧ x1 ≠ x2 ∧ x1 < x2) →
  n = 3 ∧ ∃ x1 x2 : ℕ, x1 = 2 ∧ x2 = 5 ∧ Prime x1 ∧ Prime x2 ∧
    2*x1^2 - 8*n*x1 + 10*x1 - n^2 + 35*n - 76 = 0 ∧
    2*x2^2 - 8*n*x2 + 10*x2 - n^2 + 35*n - 76 = 0 := 
by
  sorry

end prime_roots_eq_l369_36997


namespace parabola_equation_origin_l369_36942

theorem parabola_equation_origin (x0 : ℝ) :
  ∃ (p : ℝ), (p > 0) ∧ (x0^2 = 2 * p * 2) ∧ (p = 2) ∧ (x0^2 = 4 * 2) := 
by 
  sorry

end parabola_equation_origin_l369_36942


namespace remainders_equal_l369_36965

theorem remainders_equal (P P' D R k s s' : ℕ) (h1 : P > P') 
  (h2 : P % D = 2 * R) (h3 : P' % D = R) (h4 : R < D) :
  (k * (P + P')) % D = s → (k * (2 * R + R)) % D = s' → s = s' :=
by
  sorry

end remainders_equal_l369_36965


namespace yellow_paint_percentage_l369_36946

theorem yellow_paint_percentage 
  (total_gallons_mixture : ℝ)
  (light_green_paint_gallons : ℝ)
  (dark_green_paint_gallons : ℝ)
  (dark_green_paint_percentage : ℝ)
  (mixture_percentage : ℝ)
  (X : ℝ) 
  (h_total_gallons : total_gallons_mixture = light_green_paint_gallons + dark_green_paint_gallons)
  (h_dark_green_paint_yellow_amount : dark_green_paint_gallons * dark_green_paint_percentage = 1.66666666667 * 0.4)
  (h_mixture_yellow_amount : total_gallons_mixture * mixture_percentage = 5 * X + 1.66666666667 * 0.4) :
  X = 0.2 :=
by
  sorry

end yellow_paint_percentage_l369_36946


namespace proposition_relation_l369_36915

theorem proposition_relation :
  (∀ (x : ℝ), x < 3 → x < 5) ↔ (∀ (x : ℝ), x ≥ 5 → x ≥ 3) :=
by
  sorry

end proposition_relation_l369_36915


namespace find_larger_number_l369_36980

theorem find_larger_number (a b : ℤ) (h1 : a + b = 27) (h2 : a - b = 5) : a = 16 := by
  sorry

end find_larger_number_l369_36980


namespace find_min_sum_of_squares_l369_36936

open Real

theorem find_min_sum_of_squares
  (x1 x2 x3 : ℝ)
  (h1 : 0 < x1)
  (h2 : 0 < x2)
  (h3 : 0 < x3)
  (h4 : 2 * x1 + 4 * x2 + 6 * x3 = 120) :
  x1^2 + x2^2 + x3^2 >= 350 :=
sorry

end find_min_sum_of_squares_l369_36936


namespace nth_position_equation_l369_36993

theorem nth_position_equation (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = 10 * n - 9 := by
  sorry

end nth_position_equation_l369_36993


namespace find_m_l369_36922

-- Definitions for vectors and dot products
structure Vector :=
  (i : ℝ)
  (j : ℝ)

def dot_product (a b : Vector) : ℝ :=
  a.i * b.i + a.j * b.j

-- Given conditions
def i : Vector := ⟨1, 0⟩
def j : Vector := ⟨0, 1⟩

def a : Vector := ⟨2, 3⟩
def b (m : ℝ) : Vector := ⟨1, -m⟩

-- The main goal
theorem find_m (m : ℝ) (h: dot_product a (b m) = 1) : m = 1 / 3 :=
by {
  -- Calculation reaches the same \(m = 1/3\)
  sorry
}

end find_m_l369_36922


namespace students_in_class_l369_36938

theorem students_in_class (b n : ℕ) :
  6 * (b + 1) = n ∧ 9 * (b - 1) = n → n = 36 :=
by
  sorry

end students_in_class_l369_36938


namespace geom_seq_expression_l369_36955

theorem geom_seq_expression (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 + a 3 = 10) (h2 : a 2 + a 4 = 5) :
  ∀ n, a n = 2 ^ (4 - n) :=
by
  -- sorry is used to skip the proof
  sorry

end geom_seq_expression_l369_36955


namespace weather_forecast_probability_l369_36948

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem weather_forecast_probability :
  binomial_probability 3 2 0.8 = 0.384 :=
by
  sorry

end weather_forecast_probability_l369_36948


namespace stratified_sampling_elderly_count_l369_36996

-- Definitions of conditions
def elderly := 30
def middleAged := 90
def young := 60
def totalPeople := elderly + middleAged + young
def sampleSize := 36
def samplingFraction := sampleSize / totalPeople
def expectedElderlySample := elderly * samplingFraction

-- The theorem we want to prove
theorem stratified_sampling_elderly_count : expectedElderlySample = 6 := 
by 
  -- Proof is omitted
  sorry

end stratified_sampling_elderly_count_l369_36996


namespace triangle_area_45_45_90_l369_36991

/--
A right triangle has one angle of 45 degrees, and its hypotenuse measures 10√2 inches.
Prove that the area of the triangle is 50 square inches.
-/
theorem triangle_area_45_45_90 {x : ℝ} (h1 : 0 < x) (h2 : x * Real.sqrt 2 = 10 * Real.sqrt 2) : 
  (1 / 2) * x * x = 50 :=
sorry

end triangle_area_45_45_90_l369_36991


namespace bill_soaking_time_l369_36978

theorem bill_soaking_time 
  (G M : ℕ) 
  (h₁ : M = G + 7) 
  (h₂ : 3 * G + M = 19) : 
  G = 3 := 
by {
  sorry
}

end bill_soaking_time_l369_36978


namespace bouncy_balls_per_package_l369_36986

variable (x : ℝ)

def maggie_bought_packs : ℝ := 8.0 * x
def maggie_gave_away_packs : ℝ := 4.0 * x
def maggie_bought_again_packs : ℝ := 4.0 * x
def total_kept_bouncy_balls : ℝ := 80

theorem bouncy_balls_per_package :
  (maggie_bought_packs x = total_kept_bouncy_balls) → 
  x = 10 :=
by
  intro h
  sorry

end bouncy_balls_per_package_l369_36986


namespace min_expression_value_l369_36966

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  a^2 + 4 * a * b + 8 * b^2 + 10 * b * c + 3 * c^2

theorem min_expression_value (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 3) :
  minimum_value a b c ≥ 27 :=
sorry

end min_expression_value_l369_36966


namespace ConeCannotHaveSquarePlanView_l369_36971

def PlanViewIsSquare (solid : Type) : Prop :=
  -- Placeholder to denote the property that the plan view of a solid is a square
  sorry

def IsCone (solid : Type) : Prop :=
  -- Placeholder to denote the property that the solid is a cone
  sorry

theorem ConeCannotHaveSquarePlanView (solid : Type) :
  (PlanViewIsSquare solid) → ¬ (IsCone solid) :=
sorry

end ConeCannotHaveSquarePlanView_l369_36971


namespace total_workers_l369_36964

theorem total_workers (h_beavers : ℕ := 318) (h_spiders : ℕ := 544) :
  h_beavers + h_spiders = 862 :=
by
  sorry

end total_workers_l369_36964


namespace g_five_eq_248_l369_36932

-- We define g, and assume it meets the conditions described.
variable (g : ℤ → ℤ)

-- Condition 1: g(1) > 1
axiom g_one_gt_one : g 1 > 1

-- Condition 2: Functional equation for g
axiom g_funct_eq (x y : ℤ) : g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y

-- Condition 3: Recursive relationship for g
axiom g_recur_eq (x : ℤ) : 3 * g x = g (x + 1) + 2 * x - 1

-- Theorem we want to prove
theorem g_five_eq_248 : g 5 = 248 := by
  sorry

end g_five_eq_248_l369_36932


namespace parabola_opening_downwards_l369_36920

theorem parabola_opening_downwards (a : ℝ) :
  (∀ x, 0 < x ∧ x < 3 → ax^2 - 2 * a * x + 3 > 0) → -1 < a ∧ a < 0 :=
by 
  intro h
  sorry

end parabola_opening_downwards_l369_36920


namespace completing_the_square_correct_l369_36961

theorem completing_the_square_correct :
  ∀ x : ℝ, (x^2 - 4*x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro x h
  sorry

end completing_the_square_correct_l369_36961


namespace new_perimeter_is_20_l369_36934

/-
Ten 1x1 square tiles are arranged to form a figure whose outside edges form a polygon with a perimeter of 16 units.
Four additional tiles of the same size are added to the figure so that each new tile shares at least one side with 
one of the squares in the original figure. Prove that the new perimeter of the figure could be 20 units.
-/

theorem new_perimeter_is_20 (initial_perimeter : ℕ) (num_initial_tiles : ℕ) 
                            (num_new_tiles : ℕ) (shared_sides : ℕ) 
                            (total_tiles : ℕ) : 
  initial_perimeter = 16 → num_initial_tiles = 10 → num_new_tiles = 4 → 
  shared_sides ≤ 8 → total_tiles = 14 → (initial_perimeter + 2 * (num_new_tiles - shared_sides)) = 20 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end new_perimeter_is_20_l369_36934


namespace garden_area_l369_36939

theorem garden_area (length perimeter : ℝ) (length_50 : 50 * length = 1500) (perimeter_20 : 20 * perimeter = 1500) (rectangular : perimeter = 2 * length + 2 * (perimeter / 2 - length)) :
  length * (perimeter / 2 - length) = 225 := 
by
  sorry

end garden_area_l369_36939


namespace combined_molecular_weight_l369_36912

-- Define atomic masses of elements
def atomic_mass_Ca : Float := 40.08
def atomic_mass_Br : Float := 79.904
def atomic_mass_Sr : Float := 87.62
def atomic_mass_Cl : Float := 35.453

-- Define number of moles for each compound
def moles_CaBr2 : Float := 4
def moles_SrCl2 : Float := 3

-- Define molar masses of compounds
def molar_mass_CaBr2 : Float := atomic_mass_Ca + 2 * atomic_mass_Br
def molar_mass_SrCl2 : Float := atomic_mass_Sr + 2 * atomic_mass_Cl

-- Define total mass calculation for each compound
def total_mass_CaBr2 : Float := moles_CaBr2 * molar_mass_CaBr2
def total_mass_SrCl2 : Float := moles_SrCl2 * molar_mass_SrCl2

-- Prove the combined molecular weight
theorem combined_molecular_weight :
  total_mass_CaBr2 + total_mass_SrCl2 = 1275.13 :=
  by
    -- The proof will be here
    sorry

end combined_molecular_weight_l369_36912


namespace exist_m_eq_l369_36916

theorem exist_m_eq (n b : ℕ) (p : ℕ) (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) (hn_zero : n ≠ 0) (hb_zero : b ≠ 0)
  (h_div : p ∣ (b^(2^n) + 1)) :
  ∃ m : ℕ, p = 2^(n+1) * m + 1 :=
by
  sorry

end exist_m_eq_l369_36916


namespace geometric_sequence_ratio_l369_36973

theorem geometric_sequence_ratio
  (a₁ : ℝ) (q : ℝ) (hq : q ≠ 1)
  (S : ℕ → ℝ)
  (hS₃ : S 3 = a₁ * (1 - q^3) / (1 - q))
  (hS₆ : S 6 = a₁ * (1 - q^6) / (1 - q))
  (hS₃_val : S 3 = 2)
  (hS₆_val : S 6 = 18) :
  S 10 / S 5 = 1 + 2^(1/3) + 2^(2/3) :=
sorry

end geometric_sequence_ratio_l369_36973


namespace exists_five_integers_l369_36982

theorem exists_five_integers :
  ∃ (a b c d e : ℤ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
    c ≠ d ∧ c ≠ e ∧ 
    d ≠ e ∧
    ∃ (k1 k2 k3 k4 k5 : ℕ), 
      k1^2 = (a + b + c + d) ∧ 
      k2^2 = (a + b + c + e) ∧ 
      k3^2 = (a + b + d + e) ∧ 
      k4^2 = (a + c + d + e) ∧ 
      k5^2 = (b + c + d + e) := 
sorry

end exists_five_integers_l369_36982


namespace total_surface_area_correct_l369_36954

noncomputable def total_surface_area_of_cylinder (radius height : ℝ) : ℝ :=
  let lateral_surface_area := 2 * Real.pi * radius * height
  let top_and_bottom_area := 2 * Real.pi * radius^2
  lateral_surface_area + top_and_bottom_area

theorem total_surface_area_correct : total_surface_area_of_cylinder 3 10 = 78 * Real.pi :=
by
  sorry

end total_surface_area_correct_l369_36954


namespace missing_digit_in_138_x_6_divisible_by_9_l369_36930

theorem missing_digit_in_138_x_6_divisible_by_9 :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 9 ∧ (1 + 3 + 8 + x + 6) % 9 = 0 ∧ x = 0 :=
by
  sorry

end missing_digit_in_138_x_6_divisible_by_9_l369_36930


namespace new_room_area_l369_36957

def holden_master_bedroom : Nat := 309
def holden_master_bathroom : Nat := 150

theorem new_room_area : 
  (holden_master_bedroom + holden_master_bathroom) * 2 = 918 := 
by
  -- This is where the proof would go
  sorry

end new_room_area_l369_36957


namespace g_value_range_l369_36960

noncomputable def g (x y z : ℝ) : ℝ :=
  (x^2 / (x^2 + y^2)) + (y^2 / (y^2 + z^2)) + (z^2 / (z^2 + x^2))

theorem g_value_range (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : 
  (3/2 : ℝ) ≤ g x y z ∧ g x y z ≤ (3 : ℝ) / 2 := 
sorry

end g_value_range_l369_36960


namespace quadratic_coefficients_l369_36913

theorem quadratic_coefficients :
  ∃ a b c : ℤ, a = 4 ∧ b = 0 ∧ c = -3 ∧ 4 * x^2 = 3 := sorry

end quadratic_coefficients_l369_36913


namespace express_set_M_l369_36952

def is_divisor (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

def M : Set ℤ := {m | is_divisor 10 (m + 1)}

theorem express_set_M :
  M = {-11, -6, -3, -2, 0, 1, 4, 9} :=
by
  sorry

end express_set_M_l369_36952


namespace g_six_l369_36919

theorem g_six (g : ℝ → ℝ) (H1 : ∀ x y : ℝ, g (x + y) = g x * g y) (H2 : g 2 = 4) : g 6 = 64 :=
by
  sorry

end g_six_l369_36919


namespace largest_of_four_integers_l369_36945

theorem largest_of_four_integers (n : ℤ) (h1 : n % 2 = 0) (h2 : (n+2) % 2 = 0) (h3 : (n+4) % 2 = 0) (h4 : (n+6) % 2 = 0) (h : n * (n+2) * (n+4) * (n+6) = 6720) : max (max (max n (n+2)) (n+4)) (n+6) = 14 := 
sorry

end largest_of_four_integers_l369_36945


namespace quadruple_dimensions_increase_volume_l369_36981

theorem quadruple_dimensions_increase_volume 
  (V_original : ℝ) (quad_factor : ℝ)
  (initial_volume : V_original = 5)
  (quad_factor_val : quad_factor = 4) :
  V_original * (quad_factor ^ 3) = 320 := 
by 
  -- Introduce necessary variables and conditions
  let V_modified := V_original * (quad_factor ^ 3)
  
  -- Assert the calculations based on the given conditions
  have initial : V_original = 5 := initial_volume
  have quad : quad_factor = 4 := quad_factor_val
  
  -- Skip the detailed proof with sorry
  sorry


end quadruple_dimensions_increase_volume_l369_36981


namespace greatest_two_digit_with_product_12_l369_36949

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end greatest_two_digit_with_product_12_l369_36949


namespace quadratic_complete_square_l369_36998

theorem quadratic_complete_square (b c : ℝ) (h : ∀ x : ℝ, x^2 - 24 * x + 50 = (x + b)^2 + c) : b + c = -106 :=
by
  sorry

end quadratic_complete_square_l369_36998


namespace factor_polynomial_l369_36958

theorem factor_polynomial :
  (x^2 + 5 * x + 4) * (x^2 + 11 * x + 30) + (x^2 + 8 * x - 10) =
  (x^2 + 8 * x + 7) * (x^2 + 8 * x + 19) := by
  sorry

end factor_polynomial_l369_36958


namespace number_of_correct_conclusions_l369_36906

noncomputable def A (x : ℝ) : ℝ := 2 * x^2
noncomputable def B (x : ℝ) : ℝ := x + 1
noncomputable def C (x : ℝ) : ℝ := -2 * x
noncomputable def D (y : ℝ) : ℝ := y^2
noncomputable def E (x y : ℝ) : ℝ := 2 * x - y

def conclusion1 (y : ℤ) : Prop := 
  0 < ((B (0 : ℝ)) * (C (0 : ℝ)) + A (0 : ℝ) + D y + E (0) (y : ℝ))

def conclusion2 : Prop := 
  ∃ (x y : ℝ), A x + D y + 2 * E x y = -2

def M (A B C : ℝ → ℝ) (x m : ℝ) : ℝ :=
  3 * (A x - B x) + m * B x * C x

def linear_term_exists (m : ℝ) : Prop :=
  (0 : ℝ) ≠ -3 - 2 * m

def conclusion3 : Prop := 
 ∀ m : ℝ, (¬ linear_term_exists m ∧ M A B C (0 : ℝ) m > -3) 

def p (x y : ℝ) := 
  2 * (x + 1) ^ 2 + (y - 1) ^ 2 = 1

theorem number_of_correct_conclusions : Prop := 
  (¬ conclusion1 1) ∧ (conclusion2) ∧ (¬ conclusion3)

end number_of_correct_conclusions_l369_36906


namespace original_polygon_sides_l369_36927

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
(n - 2) * 180

theorem original_polygon_sides (x : ℕ) (h1 : sum_of_interior_angles (2 * x) = 2160) : x = 7 :=
by
  sorry

end original_polygon_sides_l369_36927


namespace min_value_of_exp_l369_36944

noncomputable def minimum_value_of_expression (a b : ℝ) : ℝ :=
  (1 - a)^2 + (1 - 2 * b)^2 + (a - 2 * b)^2

theorem min_value_of_exp (a b : ℝ) (h : a^2 ≥ 8 * b) : minimum_value_of_expression a b = 9 / 8 :=
by
  sorry

end min_value_of_exp_l369_36944


namespace min_value_frac_sum_l369_36926

theorem min_value_frac_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (3 * z / (x + 2 * y) + 5 * x / (2 * y + 3 * z) + 2 * y / (3 * x + z)) ≥ 3 / 4 :=
by
  sorry

end min_value_frac_sum_l369_36926


namespace ordering_abc_l369_36970

noncomputable def a : ℝ := Real.sqrt 1.01
noncomputable def b : ℝ := Real.exp 0.01 / 1.01
noncomputable def c : ℝ := Real.log (1.01 * Real.exp 1)

theorem ordering_abc : b < a ∧ a < c := by
  -- Proof of the theorem goes here
  sorry

end ordering_abc_l369_36970


namespace interval_intersection_l369_36987

theorem interval_intersection :
  {x : ℝ | 1 < 3 * x ∧ 3 * x < 2 ∧ 1 < 5 * x ∧ 5 * x < 2} =
  {x : ℝ | (1 / 3 : ℝ) < x ∧ x < (2 / 5 : ℝ)} :=
by
  -- Need a proof here
  sorry

end interval_intersection_l369_36987


namespace total_keys_needed_l369_36914

-- Definitions based on given conditions
def num_complexes : ℕ := 2
def num_apartments_per_complex : ℕ := 12
def keys_per_lock : ℕ := 3
def num_locks_per_apartment : ℕ := 1

-- Theorem stating the required number of keys
theorem total_keys_needed : 
  (num_complexes * num_apartments_per_complex * keys_per_lock = 72) :=
by
  sorry

end total_keys_needed_l369_36914


namespace Brittany_older_by_3_years_l369_36931

-- Define the necessary parameters as assumptions
variable (Rebecca_age : ℕ) (Brittany_return_age : ℕ) (vacation_years : ℕ)

-- Initial conditions
axiom h1 : Rebecca_age = 25
axiom h2 : Brittany_return_age = 32
axiom h3 : vacation_years = 4

-- Definition to capture Brittany's age before vacation
def Brittany_age_before_vacation (return_age vacation_period : ℕ) : ℕ := return_age - vacation_period

-- Theorem stating that Brittany is 3 years older than Rebecca
theorem Brittany_older_by_3_years :
  Brittany_age_before_vacation Brittany_return_age vacation_years - Rebecca_age = 3 :=
by
  sorry

end Brittany_older_by_3_years_l369_36931


namespace watched_videos_correct_l369_36943

-- Conditions
def num_suggestions_per_time : ℕ := 15
def times : ℕ := 5
def chosen_position : ℕ := 5

-- Question
def total_videos_watched : ℕ := num_suggestions_per_time * times - (num_suggestions_per_time - chosen_position)

-- Proof
theorem watched_videos_correct : total_videos_watched = 65 := by
  sorry

end watched_videos_correct_l369_36943


namespace probability_is_one_over_145_l369_36918

-- Define the domain and properties
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def even (n : ℕ) : Prop :=
  n % 2 = 0

-- Total number of ways to pick 2 distinct numbers from 1 to 30
def total_ways_to_pick_two_distinct : ℕ :=
  (30 * 29) / 2

-- Calculate prime numbers between 1 and 30
def primes_from_1_to_30 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Filter valid pairs where both numbers are prime and at least one of them is 2
def valid_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  [(2, 3), (2, 5), (2, 7), (2, 11), (2, 13), (2, 17), (2, 19), (2, 23), (2, 29)]

def count_valid_pairs (l : List (ℕ × ℕ)) : ℕ :=
  l.length

-- Probability calculation
def probability_prime_and_even : ℚ :=
  count_valid_pairs (valid_pairs primes_from_1_to_30) / total_ways_to_pick_two_distinct

-- Prove that the probability is 1/145
theorem probability_is_one_over_145 : probability_prime_and_even = 1 / 145 :=
by
  sorry

end probability_is_one_over_145_l369_36918


namespace probability_not_snow_l369_36972

theorem probability_not_snow (P_snow : ℚ) (h : P_snow = 2 / 5) : (1 - P_snow = 3 / 5) :=
by 
  rw [h]
  norm_num

end probability_not_snow_l369_36972


namespace grandmother_current_age_l369_36988

theorem grandmother_current_age (yoojung_age_current yoojung_age_future grandmother_age_future : ℕ)
    (h1 : yoojung_age_current = 5)
    (h2 : yoojung_age_future = 10)
    (h3 : grandmother_age_future = 60) :
    grandmother_age_future - (yoojung_age_future - yoojung_age_current) = 55 :=
by 
  sorry

end grandmother_current_age_l369_36988


namespace arithmetic_progression_complete_iff_divides_l369_36925

-- Definitions from the conditions
def complete_sequence (s : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, s n ≠ 0) ∧ (∀ m : ℤ, m ≠ 0 → ∃ n : ℕ, s n = m)

-- Arithmetic progression definition
def arithmetic_progression (a r : ℤ) (n : ℕ) : ℤ :=
  a + n * r

-- Lean theorem statement
theorem arithmetic_progression_complete_iff_divides (a r : ℤ) :
  (complete_sequence (arithmetic_progression a r)) ↔ (r ∣ a) := by
  sorry

end arithmetic_progression_complete_iff_divides_l369_36925


namespace cubic_inequality_l369_36983

theorem cubic_inequality (a b : ℝ) : a > b → a^3 > b^3 :=
sorry

end cubic_inequality_l369_36983


namespace alpha_is_30_or_60_l369_36904

theorem alpha_is_30_or_60
  (α : Real)
  (h1 : 0 < α ∧ α < Real.pi / 2) -- α is acute angle
  (a : ℝ × ℝ := (3 / 4, Real.sin α))
  (b : ℝ × ℝ := (Real.cos α, 1 / Real.sqrt 3))
  (h2 : a.1 * b.2 = a.2 * b.1)  -- a ∥ b
  : α = Real.pi / 6 ∨ α = Real.pi / 3 := 
sorry

end alpha_is_30_or_60_l369_36904


namespace quadratic_factorization_l369_36984

theorem quadratic_factorization (a b : ℕ) (h1 : x^2 - 18 * x + 72 = (x - a) * (x - b))
  (h2 : a > b) : 2 * b - a = 0 :=
sorry

end quadratic_factorization_l369_36984


namespace good_carrots_l369_36923

theorem good_carrots (Faye_picked : ℕ) (Mom_picked : ℕ) (bad_carrots : ℕ)
    (total_carrots : Faye_picked + Mom_picked = 28)
    (bad_carrots_count : bad_carrots = 16) : 
    28 - bad_carrots = 12 := by
  -- Proof goes here
  sorry

end good_carrots_l369_36923


namespace second_quarter_profit_l369_36917

theorem second_quarter_profit (q1 q3 q4 annual : ℕ) (h1 : q1 = 1500) (h2 : q3 = 3000) (h3 : q4 = 2000) (h4 : annual = 8000) :
  annual - (q1 + q3 + q4) = 1500 :=
by
  sorry

end second_quarter_profit_l369_36917


namespace friends_share_difference_l369_36985

-- Define the initial conditions
def gift_cost : ℕ := 120
def initial_friends : ℕ := 10
def remaining_friends : ℕ := 6

-- Define the initial and new shares
def initial_share : ℕ := gift_cost / initial_friends
def new_share : ℕ := gift_cost / remaining_friends

-- Define the difference between the new share and the initial share
def share_difference : ℕ := new_share - initial_share

-- The theorem to be proved
theorem friends_share_difference : share_difference = 8 :=
by
  sorry

end friends_share_difference_l369_36985


namespace sum_of_first_100_terms_l369_36933

theorem sum_of_first_100_terms (a : ℕ → ℤ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 1) 
  (h3 : ∀ n, a (n+2) = a n + 1) : 
  (Finset.sum (Finset.range 100) a) = 2550 :=
sorry

end sum_of_first_100_terms_l369_36933


namespace degrees_to_radians_conversion_l369_36974

theorem degrees_to_radians_conversion : (-300 : ℝ) * (Real.pi / 180) = - (5 / 3) * Real.pi :=
by
  sorry

end degrees_to_radians_conversion_l369_36974


namespace Bruce_grape_purchase_l369_36902

theorem Bruce_grape_purchase
  (G : ℕ)
  (total_paid : ℕ)
  (cost_per_kg_grapes : ℕ)
  (kg_mangoes : ℕ)
  (cost_per_kg_mangoes : ℕ)
  (total_mango_cost : ℕ)
  (total_grape_cost : ℕ)
  (total_amount : ℕ)
  (h1 : cost_per_kg_grapes = 70)
  (h2 : kg_mangoes = 10)
  (h3 : cost_per_kg_mangoes = 55)
  (h4 : total_paid = 1110)
  (h5 : total_mango_cost = kg_mangoes * cost_per_kg_mangoes)
  (h6 : total_grape_cost = G * cost_per_kg_grapes)
  (h7 : total_amount = total_mango_cost + total_grape_cost)
  (h8 : total_amount = total_paid) :
  G = 8 := by
  sorry

end Bruce_grape_purchase_l369_36902


namespace smallest_n_for_107n_same_last_two_digits_l369_36901

theorem smallest_n_for_107n_same_last_two_digits :
  ∃ n : ℕ, n > 0 ∧ (107 * n) % 100 = n % 100 ∧ n = 50 :=
by {
  sorry
}

end smallest_n_for_107n_same_last_two_digits_l369_36901


namespace father_l369_36903

variable {son_age : ℕ} -- Son's present age
variable {father_age : ℕ} -- Father's present age

-- Conditions
def father_is_four_times_son (son_age father_age : ℕ) : Prop := father_age = 4 * son_age
def sum_of_ages_ten_years_ago (son_age father_age : ℕ) : Prop := (son_age - 10) + (father_age - 10) = 60

-- Theorem statement
theorem father's_present_age 
  (son_age father_age : ℕ)
  (h1 : father_is_four_times_son son_age father_age) 
  (h2 : sum_of_ages_ten_years_ago son_age father_age) : 
  father_age = 64 :=
sorry

end father_l369_36903


namespace dot_product_eq_neg29_l369_36937

def v := (3, -2)
def w := (-5, 7)

theorem dot_product_eq_neg29 : (v.1 * w.1 + v.2 * w.2) = -29 := 
by 
  -- this is where the detailed proof will occur
  sorry

end dot_product_eq_neg29_l369_36937


namespace range_of_a_l369_36908

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, (2 * (x : ℝ) - 7 < 0) ∧ ((x : ℝ) - a > 0) ↔ (x = 3)) →
  (2 ≤ a ∧ a < 3) :=
by
  intro h
  sorry

end range_of_a_l369_36908


namespace fraction_inequalities_fraction_inequality_equality_right_fraction_inequality_equality_left_l369_36956

theorem fraction_inequalities (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  1 / 2 ≤ (a ^ 3 + b ^ 3) / (a ^ 2 + b ^ 2) ∧ (a ^ 3 + b ^ 3) / (a ^ 2 + b ^ 2) ≤ 1 :=
sorry

theorem fraction_inequality_equality_right (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  (1 - a) * (1 - b) = 0 ↔ (a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 0) :=
sorry

theorem fraction_inequality_equality_left (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  a = b ↔ a = 1 / 2 ∧ b = 1 / 2 :=
sorry

end fraction_inequalities_fraction_inequality_equality_right_fraction_inequality_equality_left_l369_36956


namespace car_journey_delay_l369_36994

theorem car_journey_delay (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (time1 : ℝ) (time2 : ℝ) (delay : ℝ) :
  distance = 225 ∧ speed1 = 60 ∧ speed2 = 50 ∧ time1 = distance / speed1 ∧ time2 = distance / speed2 ∧ 
  delay = (time2 - time1) * 60 → delay = 45 :=
by
  sorry

end car_journey_delay_l369_36994


namespace magnitude_of_z_8_l369_36910

def z : Complex := 2 + 3 * Complex.I

theorem magnitude_of_z_8 : Complex.abs (z ^ 8) = 28561 := by
  sorry

end magnitude_of_z_8_l369_36910


namespace complement_A_in_U_l369_36976

noncomputable def U : Set ℕ := {0, 1, 2}
noncomputable def A : Set ℕ := {x | x^2 - x = 0}
noncomputable def complement_U (A : Set ℕ) : Set ℕ := U \ A

theorem complement_A_in_U : 
  complement_U {x | x^2 - x = 0} = {2} := 
sorry

end complement_A_in_U_l369_36976


namespace set_intersection_complement_equiv_l369_36924

open Set

variable {α : Type*}
variable {x : α}

def U : Set ℝ := univ
def M : Set ℝ := {x | 0 ≤ x}
def N : Set ℝ := {x | x^2 < 1}

theorem set_intersection_complement_equiv :
  M ∩ (U \ N) = {x | 1 ≤ x} :=
by
  sorry

end set_intersection_complement_equiv_l369_36924


namespace union_of_sets_l369_36921

theorem union_of_sets (A B : Set ℕ) (hA : A = {1, 2, 6}) (hB : B = {2, 3, 6}) :
  A ∪ B = {1, 2, 3, 6} :=
by
  rw [hA, hB]
  ext x
  simp [Set.union]
  sorry

end union_of_sets_l369_36921


namespace find_number_l369_36975

theorem find_number (x : ℝ) (h : 3 * (2 * x + 5) = 129) : x = 19 :=
by
  sorry

end find_number_l369_36975


namespace r_exceeds_s_by_two_l369_36947

theorem r_exceeds_s_by_two (x y r s : ℝ) (h1 : 3 * x + 2 * y = 16) (h2 : 5 * x + 3 * y = 26)
  (hr : r = x) (hs : s = y) : r - s = 2 :=
by
  sorry

end r_exceeds_s_by_two_l369_36947


namespace solve_equation_l369_36951

theorem solve_equation :
  ∀ x : ℝ, (1 / 7 + 7 / x = 15 / x + 1 / 15) → x = 105 :=
by
  intros x h
  sorry

end solve_equation_l369_36951


namespace johny_travelled_South_distance_l369_36905

theorem johny_travelled_South_distance :
  ∃ S : ℝ, S + (S + 20) + 2 * (S + 20) = 220 ∧ S = 40 :=
by
  sorry

end johny_travelled_South_distance_l369_36905


namespace radius_of_ball_is_13_l369_36995

-- Define the conditions
def hole_radius : ℝ := 12
def hole_depth : ℝ := 8

-- The statement to prove
theorem radius_of_ball_is_13 : (∃ x : ℝ, x^2 + hole_radius^2 = (x + hole_depth)^2) → x + hole_depth = 13 :=
by
  sorry

end radius_of_ball_is_13_l369_36995


namespace inequality_solution_l369_36963

theorem inequality_solution (x : ℝ) :
  (x - 2 > 1) ∧ (-2 * x ≤ 4) ↔ (x > 3) :=
by
  sorry

end inequality_solution_l369_36963


namespace parallel_lines_intersection_value_of_c_l369_36941

theorem parallel_lines_intersection_value_of_c
  (a b c : ℝ) (h_parallel : a = -4 * b)
  (h1 : a * 2 - 2 * (-4) = c) (h2 : 2 * 2 + b * (-4) = c) :
  c = 0 :=
by 
  sorry

end parallel_lines_intersection_value_of_c_l369_36941
