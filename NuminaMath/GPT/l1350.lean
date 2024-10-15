import Mathlib

namespace NUMINAMATH_GPT_increasing_function_solution_l1350_135018

noncomputable def solution (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f (x + y) * (f x + f y) = f x * f y

theorem increasing_function_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f (x + y) * (f x + f y) = f x * f y)
  ∧ (∀ x y : ℝ, x < y → f x < f y)
  → ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = 1 / (a * x) :=
by {
  sorry
}

end NUMINAMATH_GPT_increasing_function_solution_l1350_135018


namespace NUMINAMATH_GPT_factor_x6_minus_64_l1350_135048

theorem factor_x6_minus_64 :
  ∀ x : ℝ, (x^6 - 64) = (x-2) * (x+2) * (x^4 + 4*x^2 + 16) :=
by
  sorry

end NUMINAMATH_GPT_factor_x6_minus_64_l1350_135048


namespace NUMINAMATH_GPT_range_of_f_neg2_l1350_135026

def quadratic_fn (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem range_of_f_neg2 (a b : ℝ) (h1 : 1 ≤ quadratic_fn a b (-1) ∧ quadratic_fn a b (-1) ≤ 2)
    (h2 : 2 ≤ quadratic_fn a b 1 ∧ quadratic_fn a b 1 ≤ 4) :
    3 ≤ quadratic_fn a b (-2) ∧ quadratic_fn a b (-2) ≤ 12 :=
sorry

end NUMINAMATH_GPT_range_of_f_neg2_l1350_135026


namespace NUMINAMATH_GPT_train_speed_conversion_l1350_135053

theorem train_speed_conversion (s_mps : ℝ) (h : s_mps = 30.002399999999998) : 
  s_mps * 3.6 = 108.01 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_conversion_l1350_135053


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1350_135071

-- Definitions for the side lengths
def side_a (x : ℝ) := 4 * x - 2
def side_b (x : ℝ) := x + 1
def side_c (x : ℝ) := 15 - 6 * x

-- Main theorem statement
theorem isosceles_triangle_perimeter (x : ℝ) (h1 : side_a x = side_b x ∨ side_a x = side_c x ∨ side_b x = side_c x) :
  (side_a x + side_b x + side_c x = 12.3) :=
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1350_135071


namespace NUMINAMATH_GPT_remove_denominators_l1350_135086

theorem remove_denominators (x : ℝ) : (1 / 2 - (x - 1) / 3 = 1) → (3 - 2 * (x - 1) = 6) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_remove_denominators_l1350_135086


namespace NUMINAMATH_GPT_even_sum_exactly_one_even_l1350_135045

theorem even_sum_exactly_one_even (a b c : ℕ) (h : (a + b + c) % 2 = 0) :
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_even_sum_exactly_one_even_l1350_135045


namespace NUMINAMATH_GPT_avg_writing_speed_l1350_135056

theorem avg_writing_speed 
  (words1 hours1 words2 hours2 : ℕ)
  (h_words1 : words1 = 30000)
  (h_hours1 : hours1 = 60)
  (h_words2 : words2 = 50000)
  (h_hours2 : hours2 = 100) :
  (words1 + words2) / (hours1 + hours2) = 500 :=
by {
  sorry
}

end NUMINAMATH_GPT_avg_writing_speed_l1350_135056


namespace NUMINAMATH_GPT_gold_coins_percentage_is_35_l1350_135023

-- Define the conditions: percentage of beads and percentage of silver coins
def percent_beads : ℝ := 0.30
def percent_silver_coins : ℝ := 0.50

-- Definition of the percentage of all objects that are gold coins
def percent_gold_coins (percent_beads percent_silver_coins : ℝ) : ℝ :=
  (1 - percent_beads) * (1 - percent_silver_coins)

-- The statement that we need to prove:
theorem gold_coins_percentage_is_35 :
  percent_gold_coins percent_beads percent_silver_coins = 0.35 :=
  by
    unfold percent_gold_coins percent_beads percent_silver_coins
    sorry

end NUMINAMATH_GPT_gold_coins_percentage_is_35_l1350_135023


namespace NUMINAMATH_GPT_scalene_triangle_geometric_progression_l1350_135097

theorem scalene_triangle_geometric_progression :
  ∀ (q : ℝ), q ≠ 0 → 
  (∀ b : ℝ, b > 0 → b + q * b > q^2 * b ∧ q * b + q^2 * b > b ∧ b + q^2 * b > q * b) → 
  ¬((0.5 < q ∧ q < 1.7) ∨ q = 2.0) → false :=
by
  intros q hq_ne_zero hq hq_interval
  sorry

end NUMINAMATH_GPT_scalene_triangle_geometric_progression_l1350_135097


namespace NUMINAMATH_GPT_total_plums_correct_l1350_135078

/-- Each picked number of plums. -/
def melanie_picked := 4
def dan_picked := 9
def sally_picked := 3
def ben_picked := 2 * (melanie_picked + dan_picked)
def sally_ate := 2

/-- The total number of plums picked in the end. -/
def total_plums_picked :=
  melanie_picked + dan_picked + sally_picked + ben_picked - sally_ate

theorem total_plums_correct : total_plums_picked = 40 := by
  sorry

end NUMINAMATH_GPT_total_plums_correct_l1350_135078


namespace NUMINAMATH_GPT_primes_count_l1350_135009

open Int

theorem primes_count (p : ℕ) (hp : Nat.Prime p) :
  ¬ ∃ r s : ℤ, ∀ x : ℤ, (x^3 - x + 2) % p = ((x - r)^2 * (x - s)) % p := 
  by
    sorry

end NUMINAMATH_GPT_primes_count_l1350_135009


namespace NUMINAMATH_GPT_solve_for_x_l1350_135036

-- Define the operation
def triangle (a b : ℝ) : ℝ := 2 * a - b

-- Define the necessary conditions and the goal
theorem solve_for_x :
  (∀ (a b : ℝ), triangle a b = 2 * a - b) →
  (∃ x : ℝ, triangle x (triangle 1 3) = 2) →
  ∃ x : ℝ, x = 1 / 2 :=
by 
  intros h_main h_eqn
  -- We can skip the proof part as requested.
  sorry

end NUMINAMATH_GPT_solve_for_x_l1350_135036


namespace NUMINAMATH_GPT_find_number_l1350_135013

theorem find_number (x : ℝ) (h : 0.3 * x - (1 / 3) * (0.3 * x) = 36) : x = 180 :=
sorry

end NUMINAMATH_GPT_find_number_l1350_135013


namespace NUMINAMATH_GPT_percentage_increase_is_20_l1350_135001

noncomputable def total_stocks : ℕ := 1980
noncomputable def stocks_higher : ℕ := 1080
noncomputable def stocks_lower : ℕ := total_stocks - stocks_higher

/--
Given that the total number of stocks is 1,980, and 1,080 stocks closed at a higher price today than yesterday.
Furthermore, the number of stocks that closed higher today is greater than the number that closed lower.

Prove that the percentage increase in the number of stocks that closed at a higher price today compared to the number that closed at a lower price is 20%.
-/
theorem percentage_increase_is_20 :
  (stocks_higher - stocks_lower) / stocks_lower * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_is_20_l1350_135001


namespace NUMINAMATH_GPT_math_problem_l1350_135091

theorem math_problem (x : ℝ) :
  (x^3 - 8*x^2 + 16*x > 64) ∧ (x^2 - 4*x + 5 > 0) → x > 4 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1350_135091


namespace NUMINAMATH_GPT_find_number_to_add_l1350_135093

theorem find_number_to_add : ∃ n : ℚ, (4 + n) / (7 + n) = 7 / 9 ∧ n = 13 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_number_to_add_l1350_135093


namespace NUMINAMATH_GPT_diamonds_in_G15_l1350_135030

theorem diamonds_in_G15 (G : ℕ → ℕ) 
  (h₁ : G 1 = 3)
  (h₂ : ∀ n, n ≥ 2 → G (n + 1) = 3 * (2 * (n - 1) + 3) - 3 ) :
  G 15 = 90 := sorry

end NUMINAMATH_GPT_diamonds_in_G15_l1350_135030


namespace NUMINAMATH_GPT_systematic_sampling_first_group_l1350_135079

theorem systematic_sampling_first_group 
  (total_students sample_size group_size group_number drawn_number : ℕ)
  (h1 : total_students = 160)
  (h2 : sample_size = 20)
  (h3 : total_students = sample_size * group_size)
  (h4 : group_number = 16)
  (h5 : drawn_number = 126) 
  : (drawn_lots_first_group : ℕ) 
      = ((drawn_number - ((group_number - 1) * group_size + 1)) + 1) :=
sorry


end NUMINAMATH_GPT_systematic_sampling_first_group_l1350_135079


namespace NUMINAMATH_GPT_rotary_club_eggs_needed_l1350_135050

theorem rotary_club_eggs_needed 
  (small_children_tickets : ℕ := 53)
  (older_children_tickets : ℕ := 35)
  (adult_tickets : ℕ := 75)
  (senior_tickets : ℕ := 37)
  (waste_percentage : ℝ := 0.03)
  (extra_omelets : ℕ := 25)
  (eggs_per_extra_omelet : ℝ := 2.5) :
  53 * 1 + 35 * 2 + 75 * 3 + 37 * 4 + 
  Nat.ceil (waste_percentage * (53 * 1 + 35 * 2 + 75 * 3 + 37 * 4)) + 
  Nat.ceil (extra_omelets * eggs_per_extra_omelet) = 574 := 
by 
  sorry

end NUMINAMATH_GPT_rotary_club_eggs_needed_l1350_135050


namespace NUMINAMATH_GPT_solution_to_inequality_l1350_135014

theorem solution_to_inequality (x : ℝ) (hx : 0 < x ∧ x < 1) : 1 / x > 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_to_inequality_l1350_135014


namespace NUMINAMATH_GPT_right_angle_triangle_exists_l1350_135040

theorem right_angle_triangle_exists (color : ℤ × ℤ → ℕ) (H1 : ∀ c : ℕ, ∃ p : ℤ × ℤ, color p = c) : 
  ∃ (A B C : ℤ × ℤ), A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ (color A ≠ color B ∧ color B ≠ color C ∧ color C ≠ color A) ∧
  ((A.1 = B.1 ∧ B.2 = C.2 ∧ A.1 - C.1 = A.2 - B.2) ∨ (A.2 = B.2 ∧ B.1 = C.1 ∧ A.1 - B.1 = A.2 - C.2)) :=
sorry

end NUMINAMATH_GPT_right_angle_triangle_exists_l1350_135040


namespace NUMINAMATH_GPT_total_blue_balloons_l1350_135059

theorem total_blue_balloons (Joan_balloons : ℕ) (Melanie_balloons : ℕ) (Alex_balloons : ℕ) 
  (hJoan : Joan_balloons = 60) (hMelanie : Melanie_balloons = 85) (hAlex : Alex_balloons = 37) :
  Joan_balloons + Melanie_balloons + Alex_balloons = 182 :=
by
  sorry

end NUMINAMATH_GPT_total_blue_balloons_l1350_135059


namespace NUMINAMATH_GPT_max_mass_of_grain_l1350_135005

theorem max_mass_of_grain (length width : ℝ) (angle : ℝ) (density : ℝ) 
  (h_length : length = 10) (h_width : width = 5) (h_angle : angle = 45) (h_density : density = 1200) : 
  volume * density = 175000 :=
by
  let height := width / 2
  let base_area := length * width
  let prism_volume := base_area * height
  let pyramid_volume := (1 / 3) * (width / 2 * length) * height
  let total_volume := prism_volume + 2 * pyramid_volume
  let volume := total_volume
  sorry

end NUMINAMATH_GPT_max_mass_of_grain_l1350_135005


namespace NUMINAMATH_GPT_triangle_inequality_sum_zero_l1350_135066

theorem triangle_inequality_sum_zero (a b c p q r : ℝ) (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) (hpqr : p + q + r = 0) : a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_inequality_sum_zero_l1350_135066


namespace NUMINAMATH_GPT_apples_ratio_l1350_135072

theorem apples_ratio (bonnie_apples samuel_extra_apples samuel_left_over samuel_total_pies : ℕ) 
  (h_bonnie : bonnie_apples = 8)
  (h_samuel_extra : samuel_extra_apples = 20)
  (h_samuel_left_over : samuel_left_over = 10)
  (h_pie_ratio : samuel_total_pies = (8 + 20) / 7) :
  (28 - samuel_total_pies - 10) / 28 = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_apples_ratio_l1350_135072


namespace NUMINAMATH_GPT_solve_system_of_equations_l1350_135075

theorem solve_system_of_equations :
  ∃ x y : ℤ, (2 * x + 7 * y = -6) ∧ (2 * x - 5 * y = 18) ∧ (x = 4) ∧ (y = -2) := 
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1350_135075


namespace NUMINAMATH_GPT_employed_males_percent_l1350_135039

variable (population : ℝ) (percent_employed : ℝ) (percent_employed_females : ℝ)

theorem employed_males_percent :
  percent_employed = 120 →
  percent_employed_females = 33.33333333333333 →
  2 / 3 * percent_employed = 80 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_employed_males_percent_l1350_135039


namespace NUMINAMATH_GPT_find_f_value_l1350_135096

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x^α

theorem find_f_value (α : ℝ) (h : f 3 α = Real.sqrt 3) : f (1 / 4) α = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_value_l1350_135096


namespace NUMINAMATH_GPT_min_chocolates_for_most_l1350_135000

theorem min_chocolates_for_most (a b c d : ℕ) (h : a < b ∧ b < c ∧ c < d)
  (h_sum : a + b + c + d = 50) : d ≥ 14 := sorry

end NUMINAMATH_GPT_min_chocolates_for_most_l1350_135000


namespace NUMINAMATH_GPT_arccos_half_eq_pi_over_three_l1350_135022

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_arccos_half_eq_pi_over_three_l1350_135022


namespace NUMINAMATH_GPT_fraction_product_simplified_l1350_135017

theorem fraction_product_simplified:
  (2 / 9 : ℚ) * (5 / 8 : ℚ) = 5 / 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_product_simplified_l1350_135017


namespace NUMINAMATH_GPT_real_roots_exist_l1350_135006

noncomputable def cubic_equation (x : ℝ) := x^3 - x^2 - 2*x + 1

theorem real_roots_exist : ∃ (a b : ℝ), 
  cubic_equation a = 0 ∧ cubic_equation b = 0 ∧ a - a * b = 1 := 
by
  sorry

end NUMINAMATH_GPT_real_roots_exist_l1350_135006


namespace NUMINAMATH_GPT_smallest_prime_dividing_sum_l1350_135087

theorem smallest_prime_dividing_sum (a b : ℕ) (h₁ : a = 7^15) (h₂ : b = 9^17) (h₃ : a % 2 = 1) (h₄ : b % 2 = 1) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (a + b) ∧ ∀ q : ℕ, (Nat.Prime q ∧ q ∣ (a + b)) → q ≥ p := by
  sorry

end NUMINAMATH_GPT_smallest_prime_dividing_sum_l1350_135087


namespace NUMINAMATH_GPT_max_vehicles_div_by_100_l1350_135027

noncomputable def max_vehicles_passing_sensor (n : ℕ) : ℕ :=
  2 * (20000 * n / (5 + 10 * n))

theorem max_vehicles_div_by_100 : 
  (∀ n : ℕ, (n > 0) → (∃ M : ℕ, M = max_vehicles_passing_sensor n ∧ M / 100 = 40)) :=
sorry

end NUMINAMATH_GPT_max_vehicles_div_by_100_l1350_135027


namespace NUMINAMATH_GPT_greatest_possible_x_l1350_135061

theorem greatest_possible_x (x : ℕ) (h : x^4 / x^2 < 18) : x ≤ 4 :=
sorry

end NUMINAMATH_GPT_greatest_possible_x_l1350_135061


namespace NUMINAMATH_GPT_problem1_problem2_l1350_135002

theorem problem1 : 
  (5 / 7 : ℚ) * (-14 / 3) / (5 / 3) = -2 := 
by 
  sorry

theorem problem2 : 
  (-15 / 7 : ℚ) / (-6 / 5) * (-7 / 5) = -5 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1350_135002


namespace NUMINAMATH_GPT_num_valid_combinations_l1350_135063

-- Definitions based on the conditions
def num_herbs := 4
def num_gems := 6
def num_incompatible_gems := 3
def num_incompatible_herbs := 2

-- Statement to be proved
theorem num_valid_combinations :
  (num_herbs * num_gems) - (num_incompatible_gems * num_incompatible_herbs) = 18 :=
by
  sorry

end NUMINAMATH_GPT_num_valid_combinations_l1350_135063


namespace NUMINAMATH_GPT_no_nat_numbers_satisfy_lcm_eq_l1350_135062

theorem no_nat_numbers_satisfy_lcm_eq (n m : ℕ) :
  ¬ (Nat.lcm (n^2) m + Nat.lcm n (m^2) = 2019) :=
sorry

end NUMINAMATH_GPT_no_nat_numbers_satisfy_lcm_eq_l1350_135062


namespace NUMINAMATH_GPT_alex_money_left_l1350_135052

noncomputable def alex_main_income : ℝ := 900
noncomputable def alex_side_income : ℝ := 300
noncomputable def main_job_tax_rate : ℝ := 0.15
noncomputable def side_job_tax_rate : ℝ := 0.20
noncomputable def water_bill : ℝ := 75
noncomputable def main_job_tithe_rate : ℝ := 0.10
noncomputable def side_job_tithe_rate : ℝ := 0.15
noncomputable def grocery_expense : ℝ := 150
noncomputable def transportation_expense : ℝ := 50

theorem alex_money_left :
  let main_income_after_tax := alex_main_income * (1 - main_job_tax_rate)
  let side_income_after_tax := alex_side_income * (1 - side_job_tax_rate)
  let total_income_after_tax := main_income_after_tax + side_income_after_tax
  let main_tithe := alex_main_income * main_job_tithe_rate
  let side_tithe := alex_side_income * side_job_tithe_rate
  let total_tithe := main_tithe + side_tithe
  let total_deductions := water_bill + grocery_expense + transportation_expense + total_tithe
  let money_left := total_income_after_tax - total_deductions
  money_left = 595 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_alex_money_left_l1350_135052


namespace NUMINAMATH_GPT_focus_parabola_l1350_135012

theorem focus_parabola (x : ℝ) (y : ℝ): (y = 8 * x^2) → (0, 1 / 32) = (0, 1 / 32) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_focus_parabola_l1350_135012


namespace NUMINAMATH_GPT_additional_length_of_track_l1350_135011

theorem additional_length_of_track (rise : ℝ) (grade1 grade2 : ℝ) (h_rise : rise = 800) (h_grade1 : grade1 = 0.04) (h_grade2 : grade2 = 0.02) :
  (rise / grade2) - (rise / grade1) = 20000 :=
by
  sorry

end NUMINAMATH_GPT_additional_length_of_track_l1350_135011


namespace NUMINAMATH_GPT_find_temp_M_l1350_135094

section TemperatureProof

variables (M T W Th F : ℕ)

-- Conditions
def avg_temp_MTWT := (M + T + W + Th) / 4 = 48
def avg_temp_TWThF := (T + W + Th + F) / 4 = 40
def temp_F := F = 10

-- Proof
theorem find_temp_M (h1 : avg_temp_MTWT M T W Th)
                    (h2 : avg_temp_TWThF T W Th F)
                    (h3 : temp_F F)
                    : M = 42 :=
sorry

end TemperatureProof

end NUMINAMATH_GPT_find_temp_M_l1350_135094


namespace NUMINAMATH_GPT_diff_lines_not_parallel_perpendicular_same_plane_l1350_135098

-- Variables
variables (m n : Type) (α β : Type)

-- Conditions
-- m and n are different lines, which we can assume as different types (or elements of some type).
-- α and β are different planes, which we can assume as different types (or elements of some type).
-- There exist definitions for parallel and perpendicular relationships between lines and planes.

def areParallel (x y : Type) : Prop := sorry
def arePerpendicularToSamePlane (x y : Type) : Prop := sorry

-- Theorem Statement
theorem diff_lines_not_parallel_perpendicular_same_plane
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : ¬ areParallel m n) :
  ¬ arePerpendicularToSamePlane m n :=
sorry

end NUMINAMATH_GPT_diff_lines_not_parallel_perpendicular_same_plane_l1350_135098


namespace NUMINAMATH_GPT_distance_midpoint_AD_to_BC_l1350_135088

variable (AC BC BD : ℕ)
variable (perpendicular : Prop)
variable (d : ℝ)

theorem distance_midpoint_AD_to_BC
  (h1 : AC = 6)
  (h2 : BC = 5)
  (h3 : BD = 3)
  (h4 : perpendicular) :
  d = Real.sqrt 5 + 2 := by
  sorry

end NUMINAMATH_GPT_distance_midpoint_AD_to_BC_l1350_135088


namespace NUMINAMATH_GPT_minimal_moves_for_7_disks_l1350_135070

/-- Mathematical model of the Tower of Hanoi problem with special rules --/
def tower_of_hanoi_moves (n : ℕ) : ℚ :=
  if n = 7 then 23 / 4 else sorry

/-- Proof problem for the minimal number of moves required to transfer all seven disks to rod C --/
theorem minimal_moves_for_7_disks : tower_of_hanoi_moves 7 = 23 / 4 := 
  sorry

end NUMINAMATH_GPT_minimal_moves_for_7_disks_l1350_135070


namespace NUMINAMATH_GPT_count_interesting_quadruples_l1350_135099

def interesting_quadruples (a b c d : ℤ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + 2 * d > b + 2 * c 

theorem count_interesting_quadruples : 
  (∃ n : ℤ, n = 582 ∧ ∀ a b c d : ℤ, interesting_quadruples a b c d → n = 582) :=
sorry

end NUMINAMATH_GPT_count_interesting_quadruples_l1350_135099


namespace NUMINAMATH_GPT_length_of_PW_l1350_135025

-- Given variables
variables (CD WX DP PX : ℝ) (CW : ℝ)

-- Condition 1: CD is parallel to WX
axiom h1 : true -- Parallelism is given as part of the problem

-- Condition 2: CW = 60 units
axiom h2 : CW = 60

-- Condition 3: DP = 18 units
axiom h3 : DP = 18

-- Condition 4: PX = 36 units
axiom h4 : PX = 36

-- Question/Answer: Prove that the length of PW = 40 units
theorem length_of_PW (PW CP : ℝ) (h5 : CP = PW / 2) (h6 : CW = CP + PW) : PW = 40 :=
by sorry

end NUMINAMATH_GPT_length_of_PW_l1350_135025


namespace NUMINAMATH_GPT_range_of_m_l1350_135010

noncomputable def f (x m : ℝ) : ℝ := |x^2 - 4| + x^2 + m * x

theorem range_of_m 
  (f_has_two_distinct_zeros : ∃ a b : ℝ, 0 < a ∧ a < b ∧ b < 3 ∧ f a m = 0 ∧ f b m = 0) :
  -14 / 3 < m ∧ m < -2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1350_135010


namespace NUMINAMATH_GPT_slices_ratio_l1350_135032

theorem slices_ratio (total_slices : ℕ) (hawaiian_slices : ℕ) (cheese_slices : ℕ) 
  (dean_hawaiian_eaten : ℕ) (frank_hawaiian_eaten : ℕ) (sammy_cheese_eaten : ℕ)
  (total_leftover : ℕ) (hawaiian_leftover : ℕ) (cheese_leftover : ℕ)
  (H1 : total_slices = 12)
  (H2 : hawaiian_slices = 12)
  (H3 : cheese_slices = 12)
  (H4 : dean_hawaiian_eaten = 6)
  (H5 : frank_hawaiian_eaten = 3)
  (H6 : total_leftover = 11)
  (H7 : hawaiian_leftover = hawaiian_slices - dean_hawaiian_eaten - frank_hawaiian_eaten)
  (H8 : cheese_leftover = total_leftover - hawaiian_leftover)
  (H9 : sammy_cheese_eaten = cheese_slices - cheese_leftover)
  : sammy_cheese_eaten / cheese_slices = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_slices_ratio_l1350_135032


namespace NUMINAMATH_GPT_smallest_possible_sum_l1350_135021

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_ne : x ≠ y) (h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 50 :=
sorry

end NUMINAMATH_GPT_smallest_possible_sum_l1350_135021


namespace NUMINAMATH_GPT_solve_for_a_l1350_135024

theorem solve_for_a (a x : ℝ) (h₁ : 2 * x - 3 = 5 * x - 2 * a) (h₂ : x = 1) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l1350_135024


namespace NUMINAMATH_GPT_rate_of_interest_is_12_percent_l1350_135003

variables (P r : ℝ)
variables (A5 A8 : ℝ)

-- Given conditions: 
axiom A5_condition : A5 = 9800
axiom A8_condition : A8 = 12005
axiom simple_interest_5_year : A5 = P + 5 * P * r / 100
axiom simple_interest_8_year : A8 = P + 8 * P * r / 100

-- The statement we aim to prove
theorem rate_of_interest_is_12_percent : r = 12 := 
sorry

end NUMINAMATH_GPT_rate_of_interest_is_12_percent_l1350_135003


namespace NUMINAMATH_GPT_find_x_l1350_135047

theorem find_x (n : ℕ) (hn : n % 2 = 1) (hpf : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 * p2 * p3 = 9^n - 1 ∧ [p1, p2, p3].contains 61) :
  9^n - 1 = 59048 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1350_135047


namespace NUMINAMATH_GPT_left_square_side_length_l1350_135092

theorem left_square_side_length 
  (x y z : ℝ)
  (H1 : y = x + 17)
  (H2 : z = x + 11)
  (H3 : x + y + z = 52) : 
  x = 8 := by
  sorry

end NUMINAMATH_GPT_left_square_side_length_l1350_135092


namespace NUMINAMATH_GPT_problem1_l1350_135042

def setA : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def setB (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 2 * m - 2}

theorem problem1 (m : ℝ) : 
  (∀ x, x ∈ setA → x ∈ setB m) ∧ ¬(∀ x, x ∈ setA ↔ x ∈ setB m) → 3 ≤ m :=
sorry

end NUMINAMATH_GPT_problem1_l1350_135042


namespace NUMINAMATH_GPT_negation_proposition_l1350_135046

theorem negation_proposition (x : ℝ) : ¬(∀ x, x > 0 → x^2 > 0) ↔ ∃ x, x > 0 ∧ x^2 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1350_135046


namespace NUMINAMATH_GPT_option_A_incorrect_l1350_135044

theorem option_A_incorrect {a b m : ℤ} (h : am = bm) : m = 0 ∨ a = b :=
by sorry

end NUMINAMATH_GPT_option_A_incorrect_l1350_135044


namespace NUMINAMATH_GPT_incorrect_value_of_observation_l1350_135057

theorem incorrect_value_of_observation
  (mean_initial : ℝ) (n : ℕ) (sum_initial: ℝ) (incorrect_value : ℝ) (correct_value : ℝ) (mean_corrected : ℝ)
  (h1 : mean_initial = 36) 
  (h2 : n = 50) 
  (h3 : sum_initial = n * mean_initial) 
  (h4 : correct_value = 45) 
  (h5 : mean_corrected = 36.5) 
  (sum_corrected : ℝ) 
  (h6 : sum_corrected = n * mean_corrected) : 
  incorrect_value = 20 := 
by 
  sorry

end NUMINAMATH_GPT_incorrect_value_of_observation_l1350_135057


namespace NUMINAMATH_GPT_esperanzas_tax_ratio_l1350_135037

theorem esperanzas_tax_ratio :
  let rent := 600
  let food_expenses := (3 / 5) * rent
  let mortgage_bill := 3 * food_expenses
  let savings := 2000
  let gross_salary := 4840
  let total_expenses := rent + food_expenses + mortgage_bill + savings
  let taxes := gross_salary - total_expenses
  (taxes / savings) = (2 / 5) := by
  sorry

end NUMINAMATH_GPT_esperanzas_tax_ratio_l1350_135037


namespace NUMINAMATH_GPT_total_students_l1350_135074

theorem total_students (m f : ℕ) (h_ratio : 3 * f = 7 * m) (h_males : m = 21) : m + f = 70 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l1350_135074


namespace NUMINAMATH_GPT_allan_initial_balloons_l1350_135089

theorem allan_initial_balloons (jake_balloons allan_bought_more allan_total_balloons : ℕ) 
  (h1 : jake_balloons = 4)
  (h2 : allan_bought_more = 3)
  (h3 : allan_total_balloons = 8) :
  ∃ (allan_initial_balloons : ℕ), allan_total_balloons = allan_initial_balloons + allan_bought_more ∧ allan_initial_balloons = 5 := 
by
  sorry

end NUMINAMATH_GPT_allan_initial_balloons_l1350_135089


namespace NUMINAMATH_GPT_rabbit_hid_carrots_l1350_135033

theorem rabbit_hid_carrots (h_r h_f : ℕ) (x : ℕ)
  (rabbit_holes : 5 * h_r = x) 
  (fox_holes : 7 * h_f = x)
  (holes_relation : h_r = h_f + 6) :
  x = 105 :=
by
  sorry

end NUMINAMATH_GPT_rabbit_hid_carrots_l1350_135033


namespace NUMINAMATH_GPT_abs_difference_l1350_135008

theorem abs_difference (a b : ℝ) (h1 : a * b = 6) (h2 : a + b = 8) : 
  |a - b| = 2 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_abs_difference_l1350_135008


namespace NUMINAMATH_GPT_initial_pipes_num_l1350_135080

variable {n : ℕ}

theorem initial_pipes_num (h1 : ∀ t : ℕ, (n * t = 8) → n = 3) (h2 : ∀ t : ℕ, (2 * t = 12) → n = 3) : n = 3 := 
by 
  sorry

end NUMINAMATH_GPT_initial_pipes_num_l1350_135080


namespace NUMINAMATH_GPT_intersecting_lines_k_value_l1350_135031

theorem intersecting_lines_k_value (k : ℝ) : 
  (∃ x y : ℝ, y = 7 * x + 5 ∧ y = -3 * x - 35 ∧ y = 4 * x + k) → k = -7 :=
by
  sorry

end NUMINAMATH_GPT_intersecting_lines_k_value_l1350_135031


namespace NUMINAMATH_GPT_books_left_l1350_135035

variable (initialBooks : ℕ) (soldBooks : ℕ) (remainingBooks : ℕ)

-- Conditions
def initial_conditions := initialBooks = 136 ∧ soldBooks = 109

-- Question: Proving the remaining books after the sale
theorem books_left (initial_conditions : initialBooks = 136 ∧ soldBooks = 109) : remainingBooks = 27 :=
by
  cases initial_conditions
  sorry

end NUMINAMATH_GPT_books_left_l1350_135035


namespace NUMINAMATH_GPT_find_extrema_l1350_135067

noncomputable def y (x : ℝ) := (Real.sin (3 * x))^2

theorem find_extrema : 
  ∃ (x : ℝ), (0 < x ∧ x < 0.6) ∧ (∀ ε > 0, ε < 0.6 - x → y (x + ε) ≤ y x ∧ y (x - ε) ≤ y x) ∧ x = Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_extrema_l1350_135067


namespace NUMINAMATH_GPT_max_cookies_l1350_135049

-- Definitions for the conditions
def John_money : ℕ := 2475
def cookie_cost : ℕ := 225

-- Statement of the problem
theorem max_cookies (x : ℕ) : cookie_cost * x ≤ John_money → x ≤ 11 :=
sorry

end NUMINAMATH_GPT_max_cookies_l1350_135049


namespace NUMINAMATH_GPT_prob_two_red_balls_consecutively_without_replacement_l1350_135058

def numOfRedBalls : ℕ := 3
def totalNumOfBalls : ℕ := 8

theorem prob_two_red_balls_consecutively_without_replacement :
  (numOfRedBalls / totalNumOfBalls) * ((numOfRedBalls - 1) / (totalNumOfBalls - 1)) = 3 / 28 :=
by
  sorry

end NUMINAMATH_GPT_prob_two_red_balls_consecutively_without_replacement_l1350_135058


namespace NUMINAMATH_GPT_find_theta_l1350_135081

-- Define the angles
variables (VEK KEW EVG θ : ℝ)

-- State the conditions as hypotheses
def conditions (VEK KEW EVG θ : ℝ) := 
  VEK = 70 ∧
  KEW = 40 ∧
  EVG = 110

-- State the theorem
theorem find_theta (VEK KEW EVG θ : ℝ)
  (h : conditions VEK KEW EVG θ) : 
  θ = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_theta_l1350_135081


namespace NUMINAMATH_GPT_inequality_solution_solution_set_l1350_135069

noncomputable def f (x a : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + 6

theorem inequality_solution (a : ℝ) : 
  f 1 a > 0 ↔ 3 - 2 * Real.sqrt 3 < a ∧ a < 3 + 2 * Real.sqrt 3 :=
by sorry

theorem solution_set (a b : ℝ) : 
  (∀ x, -1 < x ∧ x < 3 → f x a > b) ∧ (∃ x, -1 ≤ x ∧ x ≤ 3 ∧ f x a = b) ↔ 
  (a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ b = -3 :=
by sorry

end NUMINAMATH_GPT_inequality_solution_solution_set_l1350_135069


namespace NUMINAMATH_GPT_inequality_subtraction_l1350_135015

theorem inequality_subtraction {a b c : ℝ} (h : a > b) : a - c > b - c := 
sorry

end NUMINAMATH_GPT_inequality_subtraction_l1350_135015


namespace NUMINAMATH_GPT_coeff_sum_eq_minus_243_l1350_135084

theorem coeff_sum_eq_minus_243 (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x y : ℝ, (x - 2 * y) ^ 5 = a * (x + 2 * y) ^ 5 + a₁ * (x + 2 * y)^4 * y + a₂ * (x + 2 * y)^3 * y^2 
             + a₃ * (x + 2 * y)^2 * y^3 + a₄ * (x + 2 * y) * y^4 + a₅ * y^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = -243 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_coeff_sum_eq_minus_243_l1350_135084


namespace NUMINAMATH_GPT_standard_eq_of_tangent_circle_l1350_135034

-- Define the center and tangent condition of the circle
def center : ℝ × ℝ := (1, 2)
def tangent_to_x_axis (r : ℝ) : Prop := r = center.snd

-- The standard equation of the circle given the center and radius
def standard_eq_circle (h k r : ℝ) : Prop := ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement to prove the standard equation of the circle
theorem standard_eq_of_tangent_circle : 
  ∃ r, tangent_to_x_axis r ∧ standard_eq_circle 1 2 r := 
by 
  sorry

end NUMINAMATH_GPT_standard_eq_of_tangent_circle_l1350_135034


namespace NUMINAMATH_GPT_cost_of_27_pounds_l1350_135068

def rate_per_pound : ℝ := 1
def weight_pounds : ℝ := 27

theorem cost_of_27_pounds :
  weight_pounds * rate_per_pound = 27 := 
by 
  -- sorry placeholder indicates that the proof is not provided
  sorry

end NUMINAMATH_GPT_cost_of_27_pounds_l1350_135068


namespace NUMINAMATH_GPT_slope_of_tangent_line_at_zero_l1350_135060

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem slope_of_tangent_line_at_zero : (deriv f 0) = 1 :=
by
  sorry 

end NUMINAMATH_GPT_slope_of_tangent_line_at_zero_l1350_135060


namespace NUMINAMATH_GPT_min_b_l1350_135076

-- Definitions
def S (n : ℕ) : ℤ := 2^n - 1
def a (n : ℕ) : ℤ :=
  if n = 1 then 1 else 2^(n-1)
def b (n : ℕ) : ℤ := (a n)^2 - 7 * (a n) + 6

-- Theorem
theorem min_b : ∃ n : ℕ, (b n = -6) :=
sorry

end NUMINAMATH_GPT_min_b_l1350_135076


namespace NUMINAMATH_GPT_evaluate_expression_at_2_l1350_135090

theorem evaluate_expression_at_2 : (3^2 - 2^3) = 1 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_2_l1350_135090


namespace NUMINAMATH_GPT_range_of_a_l1350_135073

-- Define the function f
def f (a x : ℝ) : ℝ := -x^3 + a * x^2 - x - 1

-- Define the derivative of f
def f_prime (a x : ℝ) : ℝ := -3 * x^2 + 2 * a * x - 1

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f_prime a x ≤ 0) ↔ -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1350_135073


namespace NUMINAMATH_GPT_question1_question2_l1350_135041

-- Definitions:
def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a - 1) * x + (a^2 - 5) = 0}

-- Question 1 Statement:
theorem question1 (a : ℝ) (h : A ∩ B a = {2}) : a = -5 ∨ a = 1 := by
  sorry

-- Question 2 Statement:
theorem question2 (a : ℝ) (h : A ∪ B a = A) : a > 3 := by
  sorry

end NUMINAMATH_GPT_question1_question2_l1350_135041


namespace NUMINAMATH_GPT_negation_equiv_l1350_135077

def is_even (n : ℕ) : Prop := n % 2 = 0

def exactly_one_even (a b c : ℕ) : Prop := 
  (is_even a ∧ ¬is_even b ∧ ¬is_even c) ∨ 
  (¬is_even a ∧ is_even b ∧ ¬is_even c) ∨ 
  (¬is_even a ∧ ¬is_even b ∧ is_even c)

def at_least_two_even_or_all_odd (a b c : ℕ) : Prop := 
  (is_even a ∧ is_even b) ∨ 
  (is_even a ∧ is_even c) ∨ 
  (is_even b ∧ is_even c) ∨ 
  (¬is_even a ∧ ¬is_even b ∧ ¬is_even c)
  
theorem negation_equiv (a b c : ℕ) : 
  ¬(exactly_one_even a b c) ↔ at_least_two_even_or_all_odd a b c := 
sorry

end NUMINAMATH_GPT_negation_equiv_l1350_135077


namespace NUMINAMATH_GPT_value_of_y_l1350_135019

theorem value_of_y (x y : ℕ) (h1 : x % y = 6) (h2 : (x : ℝ) / y = 6.12) : y = 50 :=
sorry

end NUMINAMATH_GPT_value_of_y_l1350_135019


namespace NUMINAMATH_GPT_find_p_l1350_135064

-- Lean 4 definitions corresponding to the conditions
variables {p a b x0 y0 : ℝ} (hp : p > 0) (ha : a > 0) (hb : b > 0) (hx0 : x0 ≠ 0)
variables (hA : (y0^2 = 2 * p * x0) ∧ ((x0 / a)^2 - (y0 / b)^2 = 1))
variables (h_dist : x0 + x0 = p^2)
variables (h_ecc : (5^.half) = sqrt 5)

-- The proof problem
theorem find_p :
  p = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l1350_135064


namespace NUMINAMATH_GPT_cubic_expression_l1350_135054

theorem cubic_expression {x : ℝ} (h : x + (1/x) = 5) : x^3 + (1/x^3) = 110 := 
by
  sorry

end NUMINAMATH_GPT_cubic_expression_l1350_135054


namespace NUMINAMATH_GPT_parabola_constant_term_l1350_135082

theorem parabola_constant_term :
  ∃ b c : ℝ, (∀ x : ℝ, (x = 2 → 3 = x^2 + b * x + c) ∧ (x = 4 → 3 = x^2 + b * x + c)) → c = 11 :=
by
  sorry

end NUMINAMATH_GPT_parabola_constant_term_l1350_135082


namespace NUMINAMATH_GPT_yards_gained_l1350_135007

variable {G : ℤ}

theorem yards_gained (h : -5 + G = 3) : G = 8 :=
  by
  sorry

end NUMINAMATH_GPT_yards_gained_l1350_135007


namespace NUMINAMATH_GPT_find_notebooks_l1350_135029

theorem find_notebooks (S N : ℕ) (h1 : N = 4 * S + 3) (h2 : N + 6 = 5 * S) : N = 39 := 
by
  sorry 

end NUMINAMATH_GPT_find_notebooks_l1350_135029


namespace NUMINAMATH_GPT_inequality_am_gm_l1350_135028

theorem inequality_am_gm (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (2 * x^2) / (y + z) + (2 * y^2) / (x + z) + (2 * z^2) / (x + y) ≥ x + y + z :=
by
  sorry

end NUMINAMATH_GPT_inequality_am_gm_l1350_135028


namespace NUMINAMATH_GPT_length_BC_l1350_135083

theorem length_BC {A B C : ℝ} (r1 r2 : ℝ) (AB : ℝ) (h1 : r1 = 8) (h2 : r2 = 5) (h3 : AB = r1 + r2) :
  C = B + (65 : ℝ) / 3 :=
by
  -- Problem set-up and solving comes here if needed
  sorry

end NUMINAMATH_GPT_length_BC_l1350_135083


namespace NUMINAMATH_GPT_cafeteria_extra_fruit_l1350_135085

theorem cafeteria_extra_fruit 
    (red_apples : ℕ)
    (green_apples : ℕ)
    (students : ℕ)
    (total_apples := red_apples + green_apples)
    (apples_taken := students)
    (extra_apples := total_apples - apples_taken)
    (h1 : red_apples = 42)
    (h2 : green_apples = 7)
    (h3 : students = 9) :
    extra_apples = 40 := 
by 
  sorry

end NUMINAMATH_GPT_cafeteria_extra_fruit_l1350_135085


namespace NUMINAMATH_GPT_largest_possible_value_l1350_135004

noncomputable def largest_log_expression (a b : ℝ) (h1 : a ≥ b) (h2 : b > 2) : ℝ := 
  Real.log (a^2 / b^2) / Real.log a + Real.log (b^2 / a^2) / Real.log b

theorem largest_possible_value (a b : ℝ) (h1 : a ≥ b) (h2 : b > 2) (h3 : a = b) : 
  largest_log_expression a b h1 h2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_value_l1350_135004


namespace NUMINAMATH_GPT_inequality_not_true_l1350_135020

variable {x y : ℝ}

theorem inequality_not_true (h : x > y) : ¬(-3 * x + 6 > -3 * y + 6) :=
by
  sorry

end NUMINAMATH_GPT_inequality_not_true_l1350_135020


namespace NUMINAMATH_GPT_find_g_five_l1350_135016

theorem find_g_five 
  (g : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
  (h2 : ∀ x : ℝ, g x ≠ 0)
  (h3 : g 0 = 1) : g 5 = Real.exp 5 :=
sorry

end NUMINAMATH_GPT_find_g_five_l1350_135016


namespace NUMINAMATH_GPT_max_area_circle_eq_l1350_135051

theorem max_area_circle_eq (m : ℝ) :
  (x y : ℝ) → (x - 1) ^ 2 + (y + m) ^ 2 = -(m - 3) ^ 2 + 1 → 
  (∃ (r : ℝ), (r = (1 : ℝ)) ∧ (m = 3) ∧ ((x - 1) ^ 2 + (y + 3) ^ 2 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_max_area_circle_eq_l1350_135051


namespace NUMINAMATH_GPT_find_a7_l1350_135095

variable {a : ℕ → ℝ} (q : ℝ)

-- Define that the sequence a_n is geometric with ratio q.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The conditions given in the problem.
variables (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
          (h2 : a 9 * a 10 = -8)

theorem find_a7
  (hg : is_geometric_sequence a q) :
  a 7 = -2 :=
sorry

end NUMINAMATH_GPT_find_a7_l1350_135095


namespace NUMINAMATH_GPT_find_a_l1350_135065

theorem find_a : 
  ∃ a : ℝ, (a > 0) ∧ (1 / Real.logb 5 a + 1 / Real.logb 6 a + 1 / Real.logb 7 a = 1) ∧ a = 210 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1350_135065


namespace NUMINAMATH_GPT_connie_total_markers_l1350_135038

theorem connie_total_markers : 2315 + 1028 = 3343 :=
by
  sorry

end NUMINAMATH_GPT_connie_total_markers_l1350_135038


namespace NUMINAMATH_GPT_necessary_sufficient_condition_l1350_135043

noncomputable def f (x a : ℝ) : ℝ := x^2 + (a - 4) * x + (4 - 2 * a)

theorem necessary_sufficient_condition (a : ℝ) (h_a : -1 ≤ a ∧ a ≤ 1) : 
  (∀ (x : ℝ), f x a > 0) ↔ (x < 1 ∨ x > 3) :=
by
  sorry

end NUMINAMATH_GPT_necessary_sufficient_condition_l1350_135043


namespace NUMINAMATH_GPT_find_f_l1350_135055

variable (f : ℝ → ℝ)

open Function

theorem find_f (h : ∀ x: ℝ, f (3 * x + 2) = 9 * x + 8) : ∀ x: ℝ, f x = 3 * x + 2 := 
sorry

end NUMINAMATH_GPT_find_f_l1350_135055
