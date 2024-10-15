import Mathlib

namespace NUMINAMATH_GPT_negation_of_universal_statement_l2327_232731

theorem negation_of_universal_statement :
  ¬ (∀ x : ℝ, x^3 - 3 * x > 0) ↔ ∃ x : ℝ, x^3 - 3 * x ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_statement_l2327_232731


namespace NUMINAMATH_GPT_calc_value_l2327_232700

theorem calc_value (a b : ℝ) (h : b = 3 * a - 2) : 2 * b - 6 * a + 2 = -2 := 
by 
  sorry

end NUMINAMATH_GPT_calc_value_l2327_232700


namespace NUMINAMATH_GPT_jenna_practice_minutes_l2327_232758

theorem jenna_practice_minutes :
  ∀ (practice_6_days practice_2_days target_total target_average: ℕ),
    practice_6_days = 6 * 80 →
    practice_2_days = 2 * 105 →
    target_average = 100 →
    target_total = 9 * target_average →
  ∃ practice_9th_day, (practice_6_days + practice_2_days + practice_9th_day = target_total) ∧ practice_9th_day = 210 :=
by sorry

end NUMINAMATH_GPT_jenna_practice_minutes_l2327_232758


namespace NUMINAMATH_GPT_clock_angle_8_15_l2327_232778

theorem clock_angle_8_15:
  ∃ angle : ℝ, time_on_clock = 8.25 → angle = 157.5 := sorry

end NUMINAMATH_GPT_clock_angle_8_15_l2327_232778


namespace NUMINAMATH_GPT_shrub_height_at_end_of_2_years_l2327_232727

theorem shrub_height_at_end_of_2_years (h₅ : ℕ) (h : ∀ n : ℕ, 0 < n → 243 = 3^5 * h₅) : ∃ h₂ : ℕ, h₂ = 9 :=
by sorry

end NUMINAMATH_GPT_shrub_height_at_end_of_2_years_l2327_232727


namespace NUMINAMATH_GPT_find_interval_for_a_l2327_232753

-- Define the system of equations as a predicate
def system_of_equations (a x y z : ℝ) : Prop := 
  x + y + z = 0 ∧ x * y + y * z + a * z * x = 0

-- Define the condition that (0, 0, 0) is the only solution
def unique_solution (a : ℝ) : Prop :=
  ∀ x y z : ℝ, system_of_equations a x y z → x = 0 ∧ y = 0 ∧ z = 0

-- Rewrite the proof problem as a Lean statement
theorem find_interval_for_a :
  ∀ a : ℝ, unique_solution a ↔ 0 < a ∧ a < 4 :=
by
  sorry

end NUMINAMATH_GPT_find_interval_for_a_l2327_232753


namespace NUMINAMATH_GPT_trigonometric_identity_application_l2327_232723

theorem trigonometric_identity_application :
  (1 / 2) * (Real.sin (Real.pi / 12)) * (Real.cos (Real.pi / 12)) = (1 / 8) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_application_l2327_232723


namespace NUMINAMATH_GPT_colored_pencils_count_l2327_232772

-- Given conditions
def bundles := 7
def pencils_per_bundle := 10
def extra_colored_pencils := 3

-- Calculations based on conditions
def total_pencils : ℕ := bundles * pencils_per_bundle
def total_colored_pencils : ℕ := total_pencils + extra_colored_pencils

-- Statement to be proved
theorem colored_pencils_count : total_colored_pencils = 73 := by
  sorry

end NUMINAMATH_GPT_colored_pencils_count_l2327_232772


namespace NUMINAMATH_GPT_least_possible_b_prime_l2327_232705

theorem least_possible_b_prime :
  ∃ b a : ℕ, Nat.Prime a ∧ Nat.Prime b ∧ 2 * a + b = 180 ∧ a > b ∧ b = 2 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_b_prime_l2327_232705


namespace NUMINAMATH_GPT_manufacturing_cost_of_shoe_l2327_232704

theorem manufacturing_cost_of_shoe
  (transportation_cost_per_shoe : ℝ)
  (selling_price_per_shoe : ℝ)
  (gain_percentage : ℝ)
  (manufacturing_cost : ℝ)
  (H1 : transportation_cost_per_shoe = 5)
  (H2 : selling_price_per_shoe = 282)
  (H3 : gain_percentage = 0.20)
  (H4 : selling_price_per_shoe = (manufacturing_cost + transportation_cost_per_shoe) * (1 + gain_percentage)) :
  manufacturing_cost = 230 :=
sorry

end NUMINAMATH_GPT_manufacturing_cost_of_shoe_l2327_232704


namespace NUMINAMATH_GPT_unique_pair_natural_numbers_l2327_232768

theorem unique_pair_natural_numbers (a b : ℕ) :
  (∀ n : ℕ, ∃ c : ℕ, a ^ n + b ^ n = c ^ (n + 1)) → (a = 2 ∧ b = 2) :=
by
  sorry

end NUMINAMATH_GPT_unique_pair_natural_numbers_l2327_232768


namespace NUMINAMATH_GPT_divides_if_not_divisible_by_4_l2327_232749

theorem divides_if_not_divisible_by_4 (n : ℕ) :
  (¬ (4 ∣ n)) → (5 ∣ (1^n + 2^n + 3^n + 4^n)) :=
by sorry

end NUMINAMATH_GPT_divides_if_not_divisible_by_4_l2327_232749


namespace NUMINAMATH_GPT_range_of_a_l2327_232792

variable {a : ℝ}

def A := Set.Ioo (-1 : ℝ) 1
def B (a : ℝ) := Set.Ioo a (a + 1)

theorem range_of_a :
  B a ⊆ A ↔ (-1 : ℝ) ≤ a ∧ a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2327_232792


namespace NUMINAMATH_GPT_gcd_1729_1314_l2327_232710

theorem gcd_1729_1314 : Nat.gcd 1729 1314 = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_1729_1314_l2327_232710


namespace NUMINAMATH_GPT_find_y_l2327_232714

theorem find_y (x y : ℝ) : x - y = 8 ∧ x + y = 14 → y = 3 := by
  sorry

end NUMINAMATH_GPT_find_y_l2327_232714


namespace NUMINAMATH_GPT_cost_of_one_pack_l2327_232730

-- Given condition
def total_cost (packs: ℕ) : ℕ := 110
def number_of_packs : ℕ := 10

-- Question: How much does one pack cost?
-- We need to prove that one pack costs 11 dollars
theorem cost_of_one_pack : (total_cost number_of_packs) / number_of_packs = 11 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_one_pack_l2327_232730


namespace NUMINAMATH_GPT_sufficient_not_necessary_l2327_232717

variable (x : ℝ)

theorem sufficient_not_necessary (h : |x| > 0) : (x > 0 ↔ true) :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l2327_232717


namespace NUMINAMATH_GPT_symmetric_points_a_minus_b_l2327_232722

theorem symmetric_points_a_minus_b (a b : ℝ) 
  (h1 : a = -5) 
  (h2 : b = -1) :
  a - b = -4 := 
sorry

end NUMINAMATH_GPT_symmetric_points_a_minus_b_l2327_232722


namespace NUMINAMATH_GPT_H2O_production_l2327_232719

theorem H2O_production (n : Nat) (m : Nat)
  (h1 : n = 3)
  (h2 : m = 3) :
  n = m → n = 3 := by
  sorry

end NUMINAMATH_GPT_H2O_production_l2327_232719


namespace NUMINAMATH_GPT_sum_first_2017_terms_l2327_232769

-- Given sequence definition
def a : ℕ → ℕ
| 0       => 0 -- a_0 (dummy term for 1-based index convenience)
| 1       => 1
| (n + 2) => 3 * 2^(n) - a (n + 1)

-- Sum of the first n terms of the sequence {a_n}
def S : ℕ → ℕ
| 0       => 0
| (n + 1) => S n + a (n + 1)

-- Theorem to prove
theorem sum_first_2017_terms : S 2017 = 2^2017 - 1 :=
sorry

end NUMINAMATH_GPT_sum_first_2017_terms_l2327_232769


namespace NUMINAMATH_GPT_winston_initial_gas_l2327_232724

theorem winston_initial_gas (max_gas : ℕ) (store_gas : ℕ) (doctor_gas : ℕ) :
  store_gas = 6 → doctor_gas = 2 → max_gas = 12 → max_gas - (store_gas + doctor_gas) = 4 → max_gas = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_winston_initial_gas_l2327_232724


namespace NUMINAMATH_GPT_multiplication_example_l2327_232796

theorem multiplication_example : 28 * (9 + 2 - 5) * 3 = 504 := by 
  sorry

end NUMINAMATH_GPT_multiplication_example_l2327_232796


namespace NUMINAMATH_GPT_leaves_problem_l2327_232736

noncomputable def leaves_dropped_last_day (L : ℕ) (n : ℕ) : ℕ :=
  L - n * (L / 10)

theorem leaves_problem (L : ℕ) (n : ℕ) (h1 : L = 340) (h2 : leaves_dropped_last_day L n = 204) :
  n = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_leaves_problem_l2327_232736


namespace NUMINAMATH_GPT_team_A_games_42_l2327_232741

noncomputable def team_games (a b : ℕ) : Prop :=
  (a * 2 / 3 + 7) = b * 5 / 8

theorem team_A_games_42 (a b : ℕ) (h1 : a * 2 / 3 = b * 5 / 8 - 7)
                                 (h2 : b = a + 14) :
  a = 42 :=
by
  sorry

end NUMINAMATH_GPT_team_A_games_42_l2327_232741


namespace NUMINAMATH_GPT_number_of_hardbacks_l2327_232718

theorem number_of_hardbacks (H P : ℕ) (books total_books selections : ℕ) (comb : ℕ → ℕ → ℕ) :
  total_books = 8 →
  P = 2 →
  comb total_books 3 - comb H 3 = 36 →
  H = 6 :=
by sorry

end NUMINAMATH_GPT_number_of_hardbacks_l2327_232718


namespace NUMINAMATH_GPT_train_crossing_time_l2327_232774
-- Part a: Identifying the questions and conditions

-- Question: How long does it take for the train to cross the platform?
-- Conditions:
-- 1. Speed of the train: 72 km/hr
-- 2. Length of the goods train: 440 m
-- 3. Length of the platform: 80 m

-- Part b: Identifying the solution steps and the correct answers

-- The solution steps involve:
-- 1. Summing the lengths of the train and the platform to get the total distance the train needs to cover.
-- 2. Converting the speed of the train from km/hr to m/s.
-- 3. Using the formula Time = Distance / Speed to find the time.

-- Correct answer: 26 seconds

-- Part c: Translating the question, conditions, and correct answer to a mathematically equivalent proof problem

-- Goal: Prove that the time it takes for the train to cross the platform is 26 seconds given the provided conditions.

-- Part d: Writing the Lean 4 statement


-- Definitions based on the given conditions
def speed_kmh : ℕ := 72
def length_train : ℕ := 440
def length_platform : ℕ := 80

-- Definition based on the conversion step in the solution
def speed_ms : ℕ := (72 * 1000) / 3600 -- Converting speed from km/hr to m/s

-- Goal: Prove that the time it takes for the train to cross the platform is 26 seconds
theorem train_crossing_time : ((length_train + length_platform) : ℕ) / speed_ms = 26 := by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l2327_232774


namespace NUMINAMATH_GPT_weekly_earnings_proof_l2327_232740

def minutes_in_hour : ℕ := 60
def hourly_rate : ℕ := 4

def monday_minutes : ℕ := 150
def tuesday_minutes : ℕ := 40
def wednesday_minutes : ℕ := 155
def thursday_minutes : ℕ := 45

def weekly_minutes : ℕ := monday_minutes + tuesday_minutes + wednesday_minutes + thursday_minutes
def weekly_hours : ℕ := weekly_minutes / minutes_in_hour

def sylvia_earnings : ℕ := weekly_hours * hourly_rate

theorem weekly_earnings_proof :
  sylvia_earnings = 26 := by
  sorry

end NUMINAMATH_GPT_weekly_earnings_proof_l2327_232740


namespace NUMINAMATH_GPT_three_digit_max_l2327_232709

theorem three_digit_max (n : ℕ) : 
  n % 9 = 1 ∧ n % 5 = 3 ∧ n % 7 = 2 ∧ 100 <= n ∧ n <= 999 → n = 793 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_max_l2327_232709


namespace NUMINAMATH_GPT_intersection_complement_eq_l2327_232729

/-- Define the sets U, A, and B -/
def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {3, 7, 9}
def B : Set ℕ := {1, 9}

/-- Define the complement of B with respect to U -/
def complement_U_B : Set ℕ := U \ B

/-- Theorem stating the intersection of A and the complement of B with respect to U -/
theorem intersection_complement_eq : A ∩ complement_U_B = {3, 7} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_l2327_232729


namespace NUMINAMATH_GPT_total_football_games_l2327_232773

theorem total_football_games (games_this_year : ℕ) (games_last_year : ℕ) (total_games : ℕ) : 
  games_this_year = 14 → games_last_year = 29 → total_games = games_this_year + games_last_year → total_games = 43 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_total_football_games_l2327_232773


namespace NUMINAMATH_GPT_find_first_day_income_l2327_232708

def income_4 (i2 i3 i4 i5 : ℕ) : ℕ := i2 + i3 + i4 + i5

def total_income_5 (average_income : ℕ) : ℕ := 5 * average_income

def income_1 (total : ℕ) (known : ℕ) : ℕ := total - known

theorem find_first_day_income (i2 i3 i4 i5 a income5 : ℕ) (h1 : income_4 i2 i3 i4 i5 = 1800)
  (h2 : a = 440)
  (h3 : total_income_5 a = income5)
  : income_1 income5 (income_4 i2 i3 i4 i5) = 400 := 
sorry

end NUMINAMATH_GPT_find_first_day_income_l2327_232708


namespace NUMINAMATH_GPT_bus_speed_excluding_stoppages_l2327_232755

theorem bus_speed_excluding_stoppages (S : ℝ) (h₀ : 0 < S) (h₁ : 36 = (2/3) * S) : S = 54 :=
by 
  sorry

end NUMINAMATH_GPT_bus_speed_excluding_stoppages_l2327_232755


namespace NUMINAMATH_GPT_find_p_l2327_232781

-- Define the coordinates of the points
structure Point where
  x : Real
  y : Real

def Q := Point.mk 0 15
def A := Point.mk 3 15
def B := Point.mk 15 0
def O := Point.mk 0 0
def C (p : Real) := Point.mk 0 p

-- Given the area of triangle ABC and the coordinates of Q, A, B, O, and C, prove that p = 12.75
theorem find_p (p : Real) (h_area_ABC : 36 = 36) (h_Q : Q = Point.mk 0 15)
                (h_A : A = Point.mk 3 15) (h_B : B = Point.mk 15 0) 
                (h_O : O = Point.mk 0 0) : p = 12.75 := 
sorry

end NUMINAMATH_GPT_find_p_l2327_232781


namespace NUMINAMATH_GPT_claudia_has_three_25_cent_coins_l2327_232742

def number_of_coins (x y z : ℕ) := x + y + z = 15
def number_of_combinations (x y : ℕ) := 4 * x + 3 * y = 51

theorem claudia_has_three_25_cent_coins (x y z : ℕ) 
  (h1: number_of_coins x y z) 
  (h2: number_of_combinations x y): 
  z = 3 := 
by 
sorry

end NUMINAMATH_GPT_claudia_has_three_25_cent_coins_l2327_232742


namespace NUMINAMATH_GPT_acquaintances_unique_l2327_232747

theorem acquaintances_unique (N : ℕ) : ∃ acquaintances : ℕ → ℕ, 
  (∀ i j k : ℕ, i < N → j < N → k < N → i ≠ j → j ≠ k → i ≠ k → 
    acquaintances i ≠ acquaintances j ∨ acquaintances j ≠ acquaintances k ∨ acquaintances i ≠ acquaintances k) :=
sorry

end NUMINAMATH_GPT_acquaintances_unique_l2327_232747


namespace NUMINAMATH_GPT_necessary_and_sufficient_l2327_232794

theorem necessary_and_sufficient (a b : ℝ) : a > b ↔ a * |a| > b * |b| := sorry

end NUMINAMATH_GPT_necessary_and_sufficient_l2327_232794


namespace NUMINAMATH_GPT_sugarCubeWeight_l2327_232782

theorem sugarCubeWeight
  (ants1 : ℕ) (sugar_cubes1 : ℕ) (weight1 : ℕ) (hours1 : ℕ)
  (ants2 : ℕ) (sugar_cubes2 : ℕ) (hours2 : ℕ) :
  ants1 = 15 →
  sugar_cubes1 = 600 →
  weight1 = 10 →
  hours1 = 5 →
  ants2 = 20 →
  sugar_cubes2 = 960 →
  hours2 = 3 →
  ∃ weight2 : ℕ, weight2 = 5 := by
  sorry

end NUMINAMATH_GPT_sugarCubeWeight_l2327_232782


namespace NUMINAMATH_GPT_point_A_is_closer_to_origin_l2327_232786

theorem point_A_is_closer_to_origin (A B : ℤ) (hA : A = -2) (hB : B = 3) : abs A < abs B := by 
sorry

end NUMINAMATH_GPT_point_A_is_closer_to_origin_l2327_232786


namespace NUMINAMATH_GPT_number_div_addition_l2327_232738

-- Define the given conditions
def original_number (q d r : ℕ) : ℕ := (q * d) + r

theorem number_div_addition (q d r a b : ℕ) (h1 : d = 6) (h2 : q = 124) (h3 : r = 4) (h4 : a = 24) (h5 : b = 8) :
  ((original_number q d r + a) / b : ℚ) = 96.5 :=
by 
  sorry

end NUMINAMATH_GPT_number_div_addition_l2327_232738


namespace NUMINAMATH_GPT_inequality_abc_l2327_232776

variable {a b c : ℝ}

-- Assume a, b, c are positive real numbers
def positive_real_numbers (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

-- Assume the sum of any two numbers is greater than the third
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Lean 4 statement for the proof problem
theorem inequality_abc (h1 : positive_real_numbers a b c) (h2 : triangle_inequality a b c) :
  abc ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
sorry

end NUMINAMATH_GPT_inequality_abc_l2327_232776


namespace NUMINAMATH_GPT_parallelogram_area_l2327_232784

theorem parallelogram_area (base height : ℝ) (h_base : base = 36) (h_height : height = 24) : 
    base * height = 864 :=
by
  rw [h_base, h_height]
  norm_num

end NUMINAMATH_GPT_parallelogram_area_l2327_232784


namespace NUMINAMATH_GPT_maximum_obtuse_dihedral_angles_l2327_232715

-- condition: define what a tetrahedron is and its properties
structure Tetrahedron :=
  (edges : Fin 6 → ℝ)   -- represents the 6 edges
  (dihedral_angles : Fin 6 → ℝ) -- represents the 6 dihedral angles

-- Define obtuse angle in degrees
def is_obtuse (angle : ℝ) : Prop := angle > 90 ∧ angle < 180

-- Theorem statement
theorem maximum_obtuse_dihedral_angles (T : Tetrahedron) : 
  (∃ count : ℕ, count = 3 ∧ (∀ i, is_obtuse (T.dihedral_angles i) → count <= 3)) := sorry

end NUMINAMATH_GPT_maximum_obtuse_dihedral_angles_l2327_232715


namespace NUMINAMATH_GPT_students_not_taking_music_nor_art_l2327_232703

theorem students_not_taking_music_nor_art (total_students music_students art_students both_students neither_students : ℕ) 
  (h_total : total_students = 500) 
  (h_music : music_students = 50) 
  (h_art : art_students = 20) 
  (h_both : both_students = 10) 
  (h_neither : neither_students = total_students - (music_students + art_students - both_students)) : 
  neither_students = 440 :=
by
  sorry

end NUMINAMATH_GPT_students_not_taking_music_nor_art_l2327_232703


namespace NUMINAMATH_GPT_sum_of_roots_l2327_232754

theorem sum_of_roots (x₁ x₂ b : ℝ) (h : ∀ x : ℝ, x^2 - 2 * x + b = 0 → x = x₁ ∨ x = x₂) :
  x₁ + x₂ = 2 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l2327_232754


namespace NUMINAMATH_GPT_pyramid_volume_in_unit_cube_l2327_232751

noncomputable def base_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

noncomputable def pyramid_volume (base_area : ℝ) (height : ℝ) : ℝ := (1 / 3) * base_area * height

theorem pyramid_volume_in_unit_cube : 
  let s := Real.sqrt 2 / 2
  let height := 1
  pyramid_volume (base_area s) height = Real.sqrt 3 / 24 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_volume_in_unit_cube_l2327_232751


namespace NUMINAMATH_GPT_gcd_gx_x_eq_one_l2327_232743

   variable (x : ℤ)
   variable (hx : ∃ k : ℤ, x = 34567 * k)

   def g (x : ℤ) : ℤ := (3 * x + 4) * (8 * x + 3) * (15 * x + 11) * (x + 15)

   theorem gcd_gx_x_eq_one : Int.gcd (g x) x = 1 :=
   by 
     sorry
   
end NUMINAMATH_GPT_gcd_gx_x_eq_one_l2327_232743


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l2327_232728

theorem speed_of_man_in_still_water
  (v_m v_s : ℝ)
  (h1 : v_m + v_s = 4)
  (h2 : v_m - v_s = 2) :
  v_m = 3 := 
by sorry

end NUMINAMATH_GPT_speed_of_man_in_still_water_l2327_232728


namespace NUMINAMATH_GPT_three_tenths_of_number_l2327_232734

theorem three_tenths_of_number (x : ℝ) (h : (1/3) * (1/4) * x = 18) : (3/10) * x = 64.8 :=
sorry

end NUMINAMATH_GPT_three_tenths_of_number_l2327_232734


namespace NUMINAMATH_GPT_appropriate_selling_price_l2327_232770

-- Define the given conditions
def cost_per_kg : ℝ := 40
def base_price : ℝ := 50
def base_sales_volume : ℝ := 500
def sales_decrease_per_yuan : ℝ := 10
def available_capital : ℝ := 10000
def desired_profit : ℝ := 8000

-- Define the sales volume function dependent on selling price x
def sales_volume (x : ℝ) : ℝ := base_sales_volume - (x - base_price) * sales_decrease_per_yuan

-- Define the profit function dependent on selling price x
def profit (x : ℝ) : ℝ := (x - cost_per_kg) * sales_volume x

-- Prove that the appropriate selling price is 80 yuan
theorem appropriate_selling_price : 
  ∃ x : ℝ, profit x = desired_profit ∧ x = 80 :=
by
  sorry

end NUMINAMATH_GPT_appropriate_selling_price_l2327_232770


namespace NUMINAMATH_GPT_function_increasing_range_l2327_232711

theorem function_increasing_range (a : ℝ) : 
    (∀ x : ℝ, x ≥ 4 → (2*x + 2*(a-1)) > 0) ↔ a ≥ -3 := 
by
  sorry

end NUMINAMATH_GPT_function_increasing_range_l2327_232711


namespace NUMINAMATH_GPT_prove_expression_value_l2327_232779

-- Define the conditions
variables {a b c d m : ℤ}
variable (h1 : a + b = 0)
variable (h2 : |m| = 2)
variable (h3 : c * d = 1)

-- State the theorem
theorem prove_expression_value : (a + b) / (4 * m) + 2 * m ^ 2 - 3 * c * d = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_prove_expression_value_l2327_232779


namespace NUMINAMATH_GPT_perfectSquareLastFourDigits_l2327_232759

noncomputable def lastThreeDigitsForm (n : ℕ) : Prop :=
  ∃ a : ℕ, a ≤ 9 ∧ n % 1000 = a * 111

noncomputable def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem perfectSquareLastFourDigits (n : ℕ) :
  lastThreeDigitsForm n →
  isPerfectSquare n →
  (n % 10000 = 0 ∨ n % 10000 = 1444) :=
by {
  sorry
}

end NUMINAMATH_GPT_perfectSquareLastFourDigits_l2327_232759


namespace NUMINAMATH_GPT_amanda_final_quiz_score_l2327_232733

theorem amanda_final_quiz_score
  (average_score_4quizzes : ℕ)
  (total_quizzes : ℕ)
  (average_a : ℕ)
  (current_score : ℕ)
  (required_total_score : ℕ)
  (required_score_final_quiz : ℕ) :
  average_score_4quizzes = 92 →
  total_quizzes = 5 →
  average_a = 93 →
  current_score = 4 * average_score_4quizzes →
  required_total_score = total_quizzes * average_a →
  required_score_final_quiz = required_total_score - current_score →
  required_score_final_quiz = 97 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_amanda_final_quiz_score_l2327_232733


namespace NUMINAMATH_GPT_smallest_w_l2327_232756

theorem smallest_w (w : ℕ) (h1 : Nat.gcd 1452 w = 1) (h2 : 2 ∣ w ∧ 3 ∣ w ∧ 13 ∣ w) :
  (∃ (w : ℕ), 2^4 ∣ 1452 * w ∧ 3^3 ∣ 1452 * w ∧ 13^3 ∣ 1452 * w ∧ w > 0) ∧
  ∀ (w' : ℕ), (2^4 ∣ 1452 * w' ∧ 3^3 ∣ 1452 * w' ∧ 13^3 ∣ 1452 * w' ∧ w' > 0) → w ≤ w' :=
  sorry

end NUMINAMATH_GPT_smallest_w_l2327_232756


namespace NUMINAMATH_GPT_animal_market_problem_l2327_232732

theorem animal_market_problem:
  ∃ (s c : ℕ), 0 < s ∧ 0 < c ∧ 28 * s + 27 * c = 1200 ∧ c > s :=
by
  sorry

end NUMINAMATH_GPT_animal_market_problem_l2327_232732


namespace NUMINAMATH_GPT_area_circle_minus_square_l2327_232785

theorem area_circle_minus_square {r : ℝ} (h : r = 1/2) : 
  (π * r^2) - (1^2) = (π / 4) - 1 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_area_circle_minus_square_l2327_232785


namespace NUMINAMATH_GPT_find_original_price_l2327_232735

-- Define the original price P
variable (P : ℝ)

-- Define the conditions as per the given problem
def revenue_equation (P : ℝ) : Prop :=
  820 = (10 * 0.60 * P) + (20 * 0.85 * P) + (18 * P)

-- Prove that the revenue equation implies P = 20
theorem find_original_price (P : ℝ) (h : revenue_equation P) : P = 20 :=
  by sorry

end NUMINAMATH_GPT_find_original_price_l2327_232735


namespace NUMINAMATH_GPT_solve_for_x_l2327_232788

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (-2, x)
def add_vectors (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def sub_vectors (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def is_parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

theorem solve_for_x : ∀ x : ℝ, is_parallel (add_vectors a (b x)) (sub_vectors a (b x)) → x = -4 :=
by
  intros x h_par
  sorry

end NUMINAMATH_GPT_solve_for_x_l2327_232788


namespace NUMINAMATH_GPT_ab_value_l2327_232750

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end NUMINAMATH_GPT_ab_value_l2327_232750


namespace NUMINAMATH_GPT_rotation_phenomena_l2327_232775

/-- 
The rotation of the hour hand fits the definition of rotation since it turns around 
the center of the clock, covering specific angles as time passes.
-/
def is_rotation_of_hour_hand : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
The rotation of the Ferris wheel fits the definition of rotation since it turns around 
its central axis, making a complete circle.
-/
def is_rotation_of_ferris_wheel : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
The annual decline of the groundwater level does not fit the definition of rotation 
since it is a vertical movement (translation).
-/
def is_not_rotation_of_groundwater_level : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
The movement of the robots on the conveyor belt does not fit the definition of rotation 
since it is a linear/translational movement.
-/
def is_not_rotation_of_robots_on_conveyor : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
Proof that the phenomena which belong to rotation are exactly the rotation of the hour hand 
and the rotation of the Ferris wheel.
-/
theorem rotation_phenomena :
  is_rotation_of_hour_hand ∧ 
  is_rotation_of_ferris_wheel ∧ 
  is_not_rotation_of_groundwater_level ∧ 
  is_not_rotation_of_robots_on_conveyor →
  "①②" = "①②" :=
by
  intro h
  sorry

end NUMINAMATH_GPT_rotation_phenomena_l2327_232775


namespace NUMINAMATH_GPT_blocks_used_l2327_232764

theorem blocks_used (initial_blocks used_blocks : ℕ) (h_initial : initial_blocks = 78) (h_left : initial_blocks - used_blocks = 59) : used_blocks = 19 := by
  sorry

end NUMINAMATH_GPT_blocks_used_l2327_232764


namespace NUMINAMATH_GPT_selling_price_calculation_l2327_232739

-- Given conditions
def cost_price : ℚ := 110
def gain_percent : ℚ := 13.636363636363626

-- Theorem Statement
theorem selling_price_calculation : 
  (cost_price * (1 + gain_percent / 100)) = 125 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_calculation_l2327_232739


namespace NUMINAMATH_GPT_total_soldiers_correct_l2327_232787

-- Definitions based on conditions
def num_generals := 8
def num_vanguards := 8^2
def num_flags := 8^3
def num_team_leaders := 8^4
def num_armored_soldiers := 8^5
def num_soldiers := 8 + 8^2 + 8^3 + 8^4 + 8^5 + 8^6

-- Prove total number of soldiers
theorem total_soldiers_correct : num_soldiers = (1 / 7 : ℝ) * (8^7 - 8) := by
  sorry

end NUMINAMATH_GPT_total_soldiers_correct_l2327_232787


namespace NUMINAMATH_GPT_number_of_classes_l2327_232761

theorem number_of_classes (total_basketballs classes_basketballs : ℕ) (h1 : total_basketballs = 54) (h2 : classes_basketballs = 7) : total_basketballs / classes_basketballs = 7 := by
  sorry

end NUMINAMATH_GPT_number_of_classes_l2327_232761


namespace NUMINAMATH_GPT_sequence_bound_l2327_232798

theorem sequence_bound (a : ℕ → ℝ) (n : ℕ) 
  (h₁ : a 0 = 0) 
  (h₂ : a (n + 1) = 0)
  (h₃ : ∀ k, 1 ≤ k → k ≤ n → a (k - 1) - 2 * (a k) + (a (k + 1)) ≤ 1) 
  : ∀ k, 0 ≤ k → k ≤ n + 1 → a k ≤ (k * (n + 1 - k)) / 2 :=
sorry

end NUMINAMATH_GPT_sequence_bound_l2327_232798


namespace NUMINAMATH_GPT_greatest_third_term_of_arithmetic_sequence_l2327_232707

theorem greatest_third_term_of_arithmetic_sequence (a d : ℕ) (h : 4 * a + 6 * d = 46) : a + 2 * d ≤ 15 :=
sorry

end NUMINAMATH_GPT_greatest_third_term_of_arithmetic_sequence_l2327_232707


namespace NUMINAMATH_GPT_recommended_cooking_time_is_5_minutes_l2327_232702

-- Define the conditions
def time_cooked := 45 -- seconds
def time_remaining := 255 -- seconds

-- Define the total cooking time in seconds
def total_time_seconds := time_cooked + time_remaining

-- Define the conversion from seconds to minutes
def to_minutes (seconds : ℕ) : ℕ := seconds / 60

-- The main theorem to prove
theorem recommended_cooking_time_is_5_minutes :
  to_minutes total_time_seconds = 5 :=
by
  sorry

end NUMINAMATH_GPT_recommended_cooking_time_is_5_minutes_l2327_232702


namespace NUMINAMATH_GPT_parallelogram_area_288_l2327_232791

/-- A statement of the area of a given parallelogram -/
theorem parallelogram_area_288 
  (AB BC : ℝ)
  (hAB : AB = 24)
  (hBC : BC = 30)
  (height_from_A_to_DC : ℝ)
  (h_height : height_from_A_to_DC = 12)
  (is_parallelogram : true) :
  AB * height_from_A_to_DC = 288 :=
by
  -- We are focusing only on stating the theorem; the proof is not required.
  sorry

end NUMINAMATH_GPT_parallelogram_area_288_l2327_232791


namespace NUMINAMATH_GPT_marble_draw_probability_l2327_232777

def probability_first_white_second_red : ℚ :=
  let total_marbles := 10
  let white_marbles := 6
  let red_marbles := 4
  let first_white_probability := white_marbles / total_marbles
  let remaining_marbles_after_first_draw := total_marbles - 1
  let second_red_probability := red_marbles / remaining_marbles_after_first_draw
  first_white_probability * second_red_probability

theorem marble_draw_probability :
  probability_first_white_second_red = 4 / 15 := by
  sorry

end NUMINAMATH_GPT_marble_draw_probability_l2327_232777


namespace NUMINAMATH_GPT_expand_product_correct_l2327_232793

noncomputable def expand_product (x : ℝ) : ℝ :=
  (3 / 7) * (7 / x^2 + 6 * x^3 - 2)

theorem expand_product_correct (x : ℝ) (h : x ≠ 0) :
  expand_product x = (3 / x^2) + (18 * x^3 / 7) - (6 / 7) := by
  unfold expand_product
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_expand_product_correct_l2327_232793


namespace NUMINAMATH_GPT_men_in_second_group_l2327_232721

theorem men_in_second_group (M : ℕ) (h1 : 36 * 18 = M * 24) : M = 27 :=
by {
  sorry
}

end NUMINAMATH_GPT_men_in_second_group_l2327_232721


namespace NUMINAMATH_GPT_find_smallest_x_l2327_232745

def smallest_x_divisible (y : ℕ) : ℕ :=
  if y = 11 then 257 else 0

theorem find_smallest_x : 
  smallest_x_divisible 11 = 257 ∧ 
  ∃ k : ℕ, 264 * k - 7 = 257 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_x_l2327_232745


namespace NUMINAMATH_GPT_cowboy_cost_problem_l2327_232797

/-- The cost of a sandwich, a cup of coffee, and a donut adds up to 0.40 dollars given the expenditure details of two cowboys. -/
theorem cowboy_cost_problem (S C D : ℝ) (h1 : 4 * S + C + 10 * D = 1.69) (h2 : 3 * S + C + 7 * D = 1.26) :
  S + C + D = 0.40 :=
by
  sorry

end NUMINAMATH_GPT_cowboy_cost_problem_l2327_232797


namespace NUMINAMATH_GPT_quadratic_congruence_solution_l2327_232760

theorem quadratic_congruence_solution (p : ℕ) (hp : Nat.Prime p) : 
  ∃ n : ℕ, 6 * n^2 + 5 * n + 1 ≡ 0 [MOD p] := 
sorry

end NUMINAMATH_GPT_quadratic_congruence_solution_l2327_232760


namespace NUMINAMATH_GPT_find_number_multiplied_l2327_232713

theorem find_number_multiplied (m : ℕ) (h : 9999 * m = 325027405) : m = 32505 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_number_multiplied_l2327_232713


namespace NUMINAMATH_GPT_fish_population_estimate_l2327_232799

theorem fish_population_estimate 
  (caught_first : ℕ) 
  (caught_first_marked : ℕ) 
  (caught_second : ℕ) 
  (caught_second_marked : ℕ) 
  (proportion_eq : (caught_second_marked : ℚ) / caught_second = (caught_first_marked : ℚ) / caught_first) 
  : caught_first * caught_second / caught_second_marked = 750 := 
by 
  sorry

-- Conditions used as definitions in Lean 4
def pond_fish_total (caught_first : ℕ) (caught_second : ℕ) (caught_second_marked : ℕ) : ℚ :=
  (caught_first : ℚ) * (caught_second : ℚ) / (caught_second_marked : ℚ)

-- Example usage of conditions
example : pond_fish_total 30 50 2 = 750 := 
by
  sorry

end NUMINAMATH_GPT_fish_population_estimate_l2327_232799


namespace NUMINAMATH_GPT_mrs_hilt_bees_l2327_232789

theorem mrs_hilt_bees (n : ℕ) (h : 3 * n = 432) : n = 144 := by
  sorry

end NUMINAMATH_GPT_mrs_hilt_bees_l2327_232789


namespace NUMINAMATH_GPT_find_y_l2327_232795

theorem find_y {x y : ℝ} (hx : (8 : ℝ) = (1/4 : ℝ) * x) (hy : (y : ℝ) = (1/4 : ℝ) * (20 : ℝ)) (hprod : x * y = 160) : y = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_y_l2327_232795


namespace NUMINAMATH_GPT_abs_diff_101st_term_l2327_232701

theorem abs_diff_101st_term 
  (C D : ℕ → ℤ)
  (hC_start : C 0 = 20)
  (hD_start : D 0 = 20)
  (hC_diff : ∀ n, C (n + 1) = C n + 12)
  (hD_diff : ∀ n, D (n + 1) = D n - 6) :
  |C 100 - D 100| = 1800 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_101st_term_l2327_232701


namespace NUMINAMATH_GPT_tic_tac_toe_winning_boards_l2327_232720

-- Define the board as a 4x4 grid
def Board := Array (Array (Option Bool))

-- Define a function that returns all possible board states after 3 moves
noncomputable def numberOfWinningBoards : Nat := 140

theorem tic_tac_toe_winning_boards:
  numberOfWinningBoards = 140 :=
by
  sorry

end NUMINAMATH_GPT_tic_tac_toe_winning_boards_l2327_232720


namespace NUMINAMATH_GPT_rubber_duck_charity_fundraiser_l2327_232762

noncomputable def charity_raised (price_small price_medium price_large : ℕ) 
(bulk_discount_threshold_small bulk_discount_threshold_medium bulk_discount_threshold_large : ℕ)
(bulk_discount_rate_small bulk_discount_rate_medium bulk_discount_rate_large : ℝ)
(tax_rate_small tax_rate_medium tax_rate_large : ℝ)
(sold_small sold_medium sold_large : ℕ) : ℝ :=
  let cost_small := price_small * sold_small
  let cost_medium := price_medium * sold_medium
  let cost_large := price_large * sold_large

  let discount_small := if sold_small >= bulk_discount_threshold_small then 
                          (bulk_discount_rate_small * cost_small) else 0
  let discount_medium := if sold_medium >= bulk_discount_threshold_medium then 
                          (bulk_discount_rate_medium * cost_medium) else 0
  let discount_large := if sold_large >= bulk_discount_threshold_large then 
                          (bulk_discount_rate_large * cost_large) else 0

  let after_discount_small := cost_small - discount_small
  let after_discount_medium := cost_medium - discount_medium
  let after_discount_large := cost_large - discount_large

  let tax_small := tax_rate_small * after_discount_small
  let tax_medium := tax_rate_medium * after_discount_medium
  let tax_large := tax_rate_large * after_discount_large

  let total_small := after_discount_small + tax_small
  let total_medium := after_discount_medium + tax_medium
  let total_large := after_discount_large + tax_large

  total_small + total_medium + total_large

theorem rubber_duck_charity_fundraiser :
  charity_raised 2 3 5 10 15 20 0.1 0.15 0.2
  0.05 0.07 0.09 150 221 185 = 1693.10 :=
by 
  -- implementation of math corresponding to problem's solution
  sorry

end NUMINAMATH_GPT_rubber_duck_charity_fundraiser_l2327_232762


namespace NUMINAMATH_GPT_total_animals_in_savanna_l2327_232716

/-- Define the number of lions, snakes, and giraffes in Safari National Park. --/
def safari_lions : ℕ := 100
def safari_snakes : ℕ := safari_lions / 2
def safari_giraffes : ℕ := safari_snakes - 10

/-- Define the number of lions, snakes, and giraffes in Savanna National Park based on conditions. --/
def savanna_lions : ℕ := 2 * safari_lions
def savanna_snakes : ℕ := 3 * safari_snakes
def savanna_giraffes : ℕ := safari_giraffes + 20

/-- Calculate the total number of animals in Savanna National Park. --/
def total_savanna_animals : ℕ := savanna_lions + savanna_snakes + savanna_giraffes

/-- Proof statement that the total number of animals in Savanna National Park is 410.
My goal is to prove that total_savanna_animals is equal to 410. --/
theorem total_animals_in_savanna : total_savanna_animals = 410 :=
by
  sorry

end NUMINAMATH_GPT_total_animals_in_savanna_l2327_232716


namespace NUMINAMATH_GPT_rationalize_denominator_l2327_232744

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l2327_232744


namespace NUMINAMATH_GPT_ratio_sum_of_square_lengths_equals_68_l2327_232780

theorem ratio_sum_of_square_lengths_equals_68 (a b c : ℕ) 
  (h1 : (∃ (r : ℝ), r = 50 / 98) → a = 5 ∧ b = 14 ∧ c = 49) :
  a + b + c = 68 :=
by
  sorry -- Proof is not required

end NUMINAMATH_GPT_ratio_sum_of_square_lengths_equals_68_l2327_232780


namespace NUMINAMATH_GPT_dawn_bananas_l2327_232712

-- Definitions of the given conditions
def total_bananas : ℕ := 200
def lydia_bananas : ℕ := 60
def donna_bananas : ℕ := 40

-- Proof that Dawn has 100 bananas
theorem dawn_bananas : (total_bananas - donna_bananas) - lydia_bananas = 100 := by
  sorry

end NUMINAMATH_GPT_dawn_bananas_l2327_232712


namespace NUMINAMATH_GPT_staplers_left_l2327_232763

-- Definitions based on conditions
def initial_staplers : ℕ := 50
def dozen : ℕ := 12
def reports_stapled : ℕ := 3 * dozen

-- Statement of the theorem
theorem staplers_left (h : initial_staplers = 50) (d : dozen = 12) (r : reports_stapled = 3 * dozen) :
  (initial_staplers - reports_stapled) = 14 :=
sorry

end NUMINAMATH_GPT_staplers_left_l2327_232763


namespace NUMINAMATH_GPT_total_square_footage_l2327_232767

-- Definitions from the problem conditions
def price_per_square_foot : ℝ := 98
def total_property_value : ℝ := 333200

-- The mathematical statement to prove
theorem total_square_footage : (total_property_value / price_per_square_foot) = 3400 :=
by
  -- Proof goes here (skipped with sorry)
  sorry

end NUMINAMATH_GPT_total_square_footage_l2327_232767


namespace NUMINAMATH_GPT_people_sharing_pizzas_l2327_232725

-- Definitions based on conditions
def number_of_pizzas : ℝ := 21.0
def slices_per_person : ℝ := 2.625
def slices_per_pizza : ℝ := 8.0

-- Theorem to prove the number of people
theorem people_sharing_pizzas : (number_of_pizzas * slices_per_pizza) / slices_per_person = 64 :=
by
  sorry

end NUMINAMATH_GPT_people_sharing_pizzas_l2327_232725


namespace NUMINAMATH_GPT_solve_quadratic_eq_solve_cubic_eq_l2327_232766

-- Statement for the first equation
theorem solve_quadratic_eq (x : ℝ) : 9 * x^2 - 25 = 0 ↔ x = 5 / 3 ∨ x = -5 / 3 :=
by sorry

-- Statement for the second equation
theorem solve_cubic_eq (x : ℝ) : (x + 1)^3 - 27 = 0 ↔ x = 2 :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_eq_solve_cubic_eq_l2327_232766


namespace NUMINAMATH_GPT_cube_root_expression_l2327_232765

theorem cube_root_expression (N : ℝ) (h : N > 1) : 
    (N^(1/3)^(1/3)^(1/3)^(1/3)) = N^(40/81) :=
sorry

end NUMINAMATH_GPT_cube_root_expression_l2327_232765


namespace NUMINAMATH_GPT_at_least_one_negative_l2327_232757

theorem at_least_one_negative (a : Fin 7 → ℤ) :
  (∀ i j : Fin 7, i ≠ j → a i ≠ a j) ∧
  (∀ l1 l2 l3 : Fin 7, 
    a l1 + a l2 + a l3 = a l1 + a l2 + a l3) ∧
  (∃ i : Fin 7, a i = 0) →
  (∃ i : Fin 7, a i < 0) :=
  by
  sorry

end NUMINAMATH_GPT_at_least_one_negative_l2327_232757


namespace NUMINAMATH_GPT_remove_max_rooks_l2327_232746

-- Defines the problem of removing the maximum number of rooks under given conditions
theorem remove_max_rooks (n : ℕ) (attacks_odd : (ℕ × ℕ) → ℕ) :
  (∀ p : ℕ × ℕ, (attacks_odd p) % 2 = 1 → true) →
  n = 8 →
  (∃ m, m = 59) :=
by
  intros _ _
  existsi 59
  sorry

end NUMINAMATH_GPT_remove_max_rooks_l2327_232746


namespace NUMINAMATH_GPT_attraction_ticket_cost_for_parents_l2327_232737

noncomputable def total_cost (children parents adults: ℕ) (entrance_cost child_attraction_cost adult_attraction_cost: ℕ) : ℕ :=
  (children + parents + adults) * entrance_cost + children * child_attraction_cost + adults * (adult_attraction_cost)

theorem attraction_ticket_cost_for_parents
  (children parents adults: ℕ) 
  (entrance_cost child_attraction_cost total_cost_of_family: ℕ) 
  (h_children: children = 4)
  (h_parents: parents = 2)
  (h_adults: adults = 1)
  (h_entrance_cost: entrance_cost = 5)
  (h_child_attraction_cost: child_attraction_cost = 2)
  (h_total_cost_of_family: total_cost_of_family = 55)
  : (total_cost children parents adults entrance_cost child_attraction_cost 4 / 3) = total_cost_of_family - (children + parents + adults) * entrance_cost - children * child_attraction_cost := 
sorry

end NUMINAMATH_GPT_attraction_ticket_cost_for_parents_l2327_232737


namespace NUMINAMATH_GPT_total_oranges_in_stack_l2327_232748

-- Definitions based on the given conditions
def base_layer_oranges : Nat := 5 * 8
def second_layer_oranges : Nat := 4 * 7
def third_layer_oranges : Nat := 3 * 6
def fourth_layer_oranges : Nat := 2 * 5
def fifth_layer_oranges : Nat := 1 * 4

-- Theorem statement equivalent to the math problem
theorem total_oranges_in_stack : base_layer_oranges + second_layer_oranges + third_layer_oranges + fourth_layer_oranges + fifth_layer_oranges = 100 :=
by
  sorry

end NUMINAMATH_GPT_total_oranges_in_stack_l2327_232748


namespace NUMINAMATH_GPT_range_of_m_l2327_232790

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 3| - |x - 1| ≤ m^2 - 3 * m) ↔ m ≤ -1 ∨ m ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2327_232790


namespace NUMINAMATH_GPT_sum_of_altitudes_l2327_232783

theorem sum_of_altitudes (x y : ℝ) (hline : 10 * x + 8 * y = 80):
  let A := 1 / 2 * 8 * 10
  let hypotenuse := Real.sqrt (8 ^ 2 + 10 ^ 2)
  let third_altitude := 80 / hypotenuse
  let sum_altitudes := 8 + 10 + third_altitude
  sum_altitudes = 18 + 40 / Real.sqrt 41 := by
  sorry

end NUMINAMATH_GPT_sum_of_altitudes_l2327_232783


namespace NUMINAMATH_GPT_direction_vectors_of_line_l2327_232726

theorem direction_vectors_of_line : 
  ∃ v : ℝ × ℝ, (3 * v.1 - 4 * v.2 = 0) ∧ (v = (1, 3/4) ∨ v = (4, 3)) :=
by
  sorry

end NUMINAMATH_GPT_direction_vectors_of_line_l2327_232726


namespace NUMINAMATH_GPT_unique_solution_condition_l2327_232771

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 4) ↔ d ≠ 4 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_condition_l2327_232771


namespace NUMINAMATH_GPT_ordered_triples_count_l2327_232706

open Real

theorem ordered_triples_count :
  ∃ (S : Finset (ℝ × ℝ × ℝ)),
    (∀ (a b c : ℝ), (a, b, c) ∈ S ↔ (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ab = c ∧ bc = a + b ∧ ca = b)) ∧
    S.card = 2 := 
sorry

end NUMINAMATH_GPT_ordered_triples_count_l2327_232706


namespace NUMINAMATH_GPT_factorize_difference_of_squares_l2327_232752

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 81 = (a + 9) * (a - 9) :=
by
  sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_l2327_232752
