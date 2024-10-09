import Mathlib

namespace manager_salary_l655_65548

theorem manager_salary
    (average_salary_employees : ℝ)
    (num_employees : ℕ)
    (increase_in_average_due_to_manager : ℝ)
    (total_salary_20_employees : ℝ)
    (new_average_salary : ℝ)
    (total_salary_with_manager : ℝ) :
  average_salary_employees = 1300 →
  num_employees = 20 →
  increase_in_average_due_to_manager = 100 →
  total_salary_20_employees = average_salary_employees * num_employees →
  new_average_salary = average_salary_employees + increase_in_average_due_to_manager →
  total_salary_with_manager = new_average_salary * (num_employees + 1) →
  total_salary_with_manager - total_salary_20_employees = 3400 :=
by 
  sorry

end manager_salary_l655_65548


namespace isosceles_triangle_third_side_l655_65512

theorem isosceles_triangle_third_side (a b : ℝ) (h₁ : a = 4) (h₂ : b = 9) (h₃ : a = b ∨ ∃ c, c = 9 ∧ (a = c ∨ b = c) ∧ (a + b > c ∧ a + c > b ∧ b + c > a)) :
  a = 9 ∨ b = 9 :=
by
  sorry

end isosceles_triangle_third_side_l655_65512


namespace xyz_value_l655_65509

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24) 
  (h2 : x ^ 2 * (y + z) + y ^ 2 * (x + z) + z ^ 2 * (x + y) = 9) : 
  x * y * z = 5 :=
by
  sorry

end xyz_value_l655_65509


namespace josie_total_animals_is_correct_l655_65521

noncomputable def totalAnimals : Nat :=
  let antelopes := 80
  let rabbits := antelopes + 34
  let hyenas := antelopes + rabbits - 42
  let wild_dogs := hyenas + 50
  let leopards := rabbits / 2
  let giraffes := antelopes + 15
  let lions := leopards + giraffes
  let elephants := 3 * lions
  antelopes + rabbits + hyenas + wild_dogs + leopards + giraffes + lions + elephants

theorem josie_total_animals_is_correct : totalAnimals = 1308 := by
  sorry

end josie_total_animals_is_correct_l655_65521


namespace total_blood_cells_correct_l655_65564

-- Define the number of blood cells in the first and second samples.
def sample_1_blood_cells : ℕ := 4221
def sample_2_blood_cells : ℕ := 3120

-- Define the total number of blood cells.
def total_blood_cells : ℕ := sample_1_blood_cells + sample_2_blood_cells

-- Theorem stating the total number of blood cells based on the conditions.
theorem total_blood_cells_correct : total_blood_cells = 7341 :=
by
  -- Proof is omitted
  sorry

end total_blood_cells_correct_l655_65564


namespace simplify_expression_l655_65529

open Nat

theorem simplify_expression (x : ℤ) : 2 - (3 - (2 - (5 - (3 - x)))) = -1 - x :=
by
  sorry

end simplify_expression_l655_65529


namespace park_area_is_120000_l655_65500

noncomputable def area_of_park : ℕ :=
  let speed_km_hr := 12
  let speed_m_min := speed_km_hr * 1000 / 60
  let time_min := 8
  let perimeter := speed_m_min * time_min
  let ratio_l_b := (1, 3)
  let length := perimeter / (2 * (ratio_l_b.1 + ratio_l_b.2))
  let breadth := ratio_l_b.2 * length
  length * breadth

theorem park_area_is_120000 :
  area_of_park = 120000 :=
by
  sorry

end park_area_is_120000_l655_65500


namespace sam_catches_alice_in_40_minutes_l655_65551

def sam_speed := 7 -- mph
def alice_speed := 4 -- mph
def initial_distance := 2 -- miles

theorem sam_catches_alice_in_40_minutes : 
  (initial_distance / (sam_speed - alice_speed)) * 60 = 40 :=
by sorry

end sam_catches_alice_in_40_minutes_l655_65551


namespace average_diesel_rate_l655_65528

theorem average_diesel_rate (r1 r2 r3 r4 : ℝ) (H1: (r1 + r2 + r3 + r4) / 4 = 1.52) :
    ((r1 + r2 + r3 + r4) / 4 = 1.52) :=
by
  exact H1

end average_diesel_rate_l655_65528


namespace football_game_attendance_l655_65544

-- Define the initial conditions
def saturday : ℕ := 80
def monday : ℕ := saturday - 20
def wednesday : ℕ := monday + 50
def friday : ℕ := saturday + monday
def total_week_actual : ℕ := saturday + monday + wednesday + friday
def total_week_expected : ℕ := 350

-- Define the proof statement
theorem football_game_attendance : total_week_actual - total_week_expected = 40 :=
by 
  -- Proof steps would go here
  sorry

end football_game_attendance_l655_65544


namespace percentage_blue_shirts_l655_65531

theorem percentage_blue_shirts (total_students := 600) 
 (percent_red := 23)
 (percent_green := 15)
 (students_other := 102)
 : (100 - (percent_red + percent_green + (students_other / total_students) * 100)) = 45 := by
  sorry

end percentage_blue_shirts_l655_65531


namespace terminal_side_in_second_quadrant_l655_65533

theorem terminal_side_in_second_quadrant (α : ℝ) (h : (Real.tan α < 0) ∧ (Real.cos α < 0)) :
  (2 < α / (π / 2)) ∧ (α / (π / 2) < 3) :=
by
  sorry

end terminal_side_in_second_quadrant_l655_65533


namespace sum_gcd_lcm_is_159_l655_65558

-- Definitions for GCD and LCM for specific values
def gcd_45_75 := Int.gcd 45 75
def lcm_48_18 := Int.lcm 48 18

-- The proof problem statement
theorem sum_gcd_lcm_is_159 : gcd_45_75 + lcm_48_18 = 159 := by
  sorry

end sum_gcd_lcm_is_159_l655_65558


namespace find_n_in_arithmetic_sequence_l655_65540

noncomputable def arithmetic_sequence (n : ℕ) (a_n S_n d : ℕ) :=
  ∀ (a₁ : ℕ), 
    a₁ + d * (n - 1) = a_n →
    n * a₁ + d * n * (n - 1) / 2 = S_n

theorem find_n_in_arithmetic_sequence 
   (a_n S_n d n : ℕ) 
   (h_a_n : a_n = 44) 
   (h_S_n : S_n = 158) 
   (h_d : d = 3) :
   arithmetic_sequence n a_n S_n d → 
   n = 4 := 
by 
  sorry

end find_n_in_arithmetic_sequence_l655_65540


namespace find_number_l655_65585

-- Given conditions and declarations
variable (x : ℕ)
variable (h : x / 3 = x - 42)

-- Proof problem statement
theorem find_number : x = 63 := 
sorry

end find_number_l655_65585


namespace most_economical_is_small_l655_65542

noncomputable def most_economical_size (c_S q_S c_M q_M c_L q_L : ℝ) :=
  c_M = 1.3 * c_S ∧
  q_M = 0.85 * q_L ∧
  q_L = 1.5 * q_S ∧
  c_L = 1.4 * c_M →
  (c_S / q_S < c_M / q_M) ∧ (c_S / q_S < c_L / q_L)

theorem most_economical_is_small (c_S q_S c_M q_M c_L q_L : ℝ) :
  most_economical_size c_S q_S c_M q_M c_L q_L := by 
  sorry

end most_economical_is_small_l655_65542


namespace correct_division_result_l655_65537

-- Define the conditions
def incorrect_divisor : ℕ := 48
def correct_divisor : ℕ := 36
def incorrect_quotient : ℕ := 24
def dividend : ℕ := incorrect_divisor * incorrect_quotient

-- Theorem statement
theorem correct_division_result : (dividend / correct_divisor) = 32 := by
  -- proof to be filled later
  sorry

end correct_division_result_l655_65537


namespace range_of_m_l655_65556

theorem range_of_m (m : ℝ) :
  (∀ x y : ℝ, (y = (m - 2) * x + m - 1 → (x ≥ 0 ∨ y ≥ 0))) ↔ (1 ≤ m ∧ m < 2) :=
by sorry

end range_of_m_l655_65556


namespace polynomial_solution_l655_65595

noncomputable def p (x : ℝ) : ℝ := (7 / 4) * x^2 + 1

theorem polynomial_solution :
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) ∧ p 2 = 8 :=
by
  sorry

end polynomial_solution_l655_65595


namespace volume_ratio_l655_65534

noncomputable def salinity_bay (salt_bay volume_bay : ℝ) : ℝ :=
  salt_bay / volume_bay

noncomputable def salinity_sea_excluding_bay (salt_sea_excluding_bay volume_sea_excluding_bay : ℝ) : ℝ :=
  salt_sea_excluding_bay / volume_sea_excluding_bay

noncomputable def salinity_whole_sea (salt_sea volume_sea : ℝ) : ℝ :=
  salt_sea / volume_sea

theorem volume_ratio (salt_bay volume_bay salt_sea_excluding_bay volume_sea_excluding_bay : ℝ) 
  (h_bay : salinity_bay salt_bay volume_bay = 240 / 1000)
  (h_sea_excluding_bay : salinity_sea_excluding_bay salt_sea_excluding_bay volume_sea_excluding_bay = 110 / 1000)
  (h_whole_sea : salinity_whole_sea (salt_bay + salt_sea_excluding_bay) (volume_bay + volume_sea_excluding_bay) = 120 / 1000) :
  (volume_bay + volume_sea_excluding_bay) / volume_bay = 13 := 
sorry

end volume_ratio_l655_65534


namespace points_earned_l655_65501

-- Definitions of the types of enemies and their point values
def points_A := 10
def points_B := 15
def points_C := 20

-- Number of each type of enemies in the level
def num_A_total := 3
def num_B_total := 2
def num_C_total := 3

-- Number of each type of enemies defeated
def num_A_defeated := num_A_total -- 3 Type A enemies
def num_B_defeated := 1 -- Half of 2 Type B enemies
def num_C_defeated := 1 -- 1 Type C enemy

-- Calculation of total points earned
def total_points : ℕ :=
  num_A_defeated * points_A + num_B_defeated * points_B + num_C_defeated * points_C

-- Proof that the total points earned is 65
theorem points_earned : total_points = 65 := by
  -- Placeholder for the proof, which calculates the total points
  sorry

end points_earned_l655_65501


namespace solution_set_of_inequality_l655_65517

theorem solution_set_of_inequality :
  { x : ℝ | x > 0 ∧ x < 1 } = { x : ℝ | 1 / x > 1 } :=
by
  sorry

end solution_set_of_inequality_l655_65517


namespace negation_of_universal_l655_65513

variable (P : ℝ → Prop)
def pos (x : ℝ) : Prop := x > 0
def gte_zero (x : ℝ) : Prop := x^2 - x ≥ 0
def lt_zero (x : ℝ) : Prop := x^2 - x < 0

theorem negation_of_universal :
  ¬ (∀ x, pos x → gte_zero x) ↔ ∃ x, pos x ∧ lt_zero x := by
  sorry

end negation_of_universal_l655_65513


namespace probability_circle_or_square_l655_65588

theorem probability_circle_or_square (total_figures : ℕ)
    (num_circles : ℕ) (num_squares : ℕ) (num_triangles : ℕ)
    (total_figures_eq : total_figures = 10)
    (num_circles_eq : num_circles = 3)
    (num_squares_eq : num_squares = 4)
    (num_triangles_eq : num_triangles = 3) :
    (num_circles + num_squares) / total_figures = 7 / 10 :=
by sorry

end probability_circle_or_square_l655_65588


namespace cone_volume_surface_area_sector_l655_65580

theorem cone_volume_surface_area_sector (V : ℝ):
  (∃ (r l h : ℝ), (π * r * (r + l) = 15 * π) ∧ (l = 6 * r) ∧ (h = Real.sqrt (l^2 - r^2)) ∧ (V = (1/3) * π * r^2 * h)) →
  V = (25 * Real.sqrt 3 / 7) * π :=
by 
  sorry

end cone_volume_surface_area_sector_l655_65580


namespace max_board_size_l655_65577

theorem max_board_size : ∀ (n : ℕ), 
  (∃ (board : Fin n → Fin n → Prop),
    ∀ i j k l : Fin n,
      (i ≠ k ∧ j ≠ l) → board i j ≠ board k l) ↔ n ≤ 4 :=
by sorry

end max_board_size_l655_65577


namespace trapezium_area_l655_65546

variables {A B C D O : Type}
variables (P Q : ℕ)

-- Conditions
def trapezium (ABCD : Type) : Prop := true
def parallel_lines (AB DC : Type) : Prop := true
def intersection (AC BD O : Type) : Prop := true
def area_AOB (P : ℕ) : Prop := P = 16
def area_COD : ℕ := 25

theorem trapezium_area (ABCD AC BD AB DC O : Type) (P Q : ℕ)
  (h1 : trapezium ABCD)
  (h2 : parallel_lines AB DC)
  (h3 : intersection AC BD O)
  (h4 : area_AOB P) 
  (h5 : area_COD = 25) :
  Q = 81 :=
sorry

end trapezium_area_l655_65546


namespace find_p_plus_q_l655_65597

noncomputable def p (d e : ℝ) (x : ℝ) : ℝ := d * x + e
noncomputable def q (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_p_plus_q (d e a b c : ℝ)
  (h1 : p d e 0 / q a b c 0 = 4)
  (h2 : p d e (-1) = -1)
  (h3 : q a b c 1 = 3)
  (e_eq : e = 4 * c):
  (p d e x + q a b c x) = (3*x^2 + 26*x - 30) :=
by
  sorry

end find_p_plus_q_l655_65597


namespace exercise_l655_65550

noncomputable def f : ℝ → ℝ := sorry

theorem exercise
  (h_even : ∀ x : ℝ, f (x + 1) = f (-(x + 1)))
  (h_increasing : ∀ ⦃a b : ℝ⦄, 1 ≤ a → a ≤ b → f a ≤ f b)
  (x1 x2 : ℝ)
  (h_x1_neg : x1 < 0)
  (h_x2_pos : x2 > 0)
  (h_sum_neg : x1 + x2 < -2) :
  f (-x1) > f (-x2) :=
sorry

end exercise_l655_65550


namespace segment_AC_length_l655_65555

noncomputable def circle_radius := 8
noncomputable def chord_length_AB := 10
noncomputable def arc_length_AC (circumference : ℝ) := circumference / 3

theorem segment_AC_length :
  ∀ (C : ℝ) (r : ℝ) (AB : ℝ) (AC : ℝ),
    r = circle_radius →
    AB = chord_length_AB →
    C = 2 * Real.pi * r →
    AC = arc_length_AC C →
    AC = 8 * Real.sqrt 3 :=
by
  intros C r AB AC hr hAB hC hAC
  sorry

end segment_AC_length_l655_65555


namespace not_divisible_by_n_only_prime_3_l655_65591

-- Problem 1: Prove that for any natural number \( n \) greater than 1, \( 2^n - 1 \) is not divisible by \( n \)
theorem not_divisible_by_n (n : ℕ) (h1 : 1 < n) : ¬ (n ∣ (2^n - 1)) :=
sorry

-- Problem 2: Prove that the only prime number \( n \) such that \( 2^n + 1 \) is divisible by \( n^2 \) is \( n = 3 \)
theorem only_prime_3 (n : ℕ) (hn : Nat.Prime n) (hdiv : n^2 ∣ (2^n + 1)) : n = 3 :=
sorry

end not_divisible_by_n_only_prime_3_l655_65591


namespace car_speeds_l655_65583

-- Definitions and conditions
def distance_AB : ℝ := 200
def distance_meet : ℝ := 80
def car_A_speed : ℝ := sorry -- To Be Proved
def car_B_speed : ℝ := sorry -- To Be Proved

axiom car_B_faster (x : ℝ) : car_B_speed = car_A_speed + 30
axiom time_equal (x : ℝ) : (distance_meet / car_A_speed) = ((distance_AB - distance_meet) / car_B_speed)

-- Proof (only statement, without steps)
theorem car_speeds : car_A_speed = 60 ∧ car_B_speed = 90 :=
  by
  have car_A_speed := 60
  have car_B_speed := 90
  sorry

end car_speeds_l655_65583


namespace bonus_distribution_plans_l655_65592

theorem bonus_distribution_plans (x y : ℕ) (A B : ℕ) 
  (h1 : x + y = 15)
  (h2 : x = 2 * y)
  (h3 : 10 * A + 5 * B = 20000)
  (hA : A ≥ B)
  (hB : B ≥ 800)
  (hAB_mult_100 : ∃ (k m : ℕ), A = k * 100 ∧ B = m * 100) :
  (x = 10 ∧ y = 5) ∧
  ((A = 1600 ∧ B = 800) ∨
   (A = 1500 ∧ B = 1000) ∨
   (A = 1400 ∧ B = 1200)) :=
by
  -- The proof should be provided here
  sorry

end bonus_distribution_plans_l655_65592


namespace operation_equivalence_l655_65581

theorem operation_equivalence :
  (∀ (x : ℝ), (x * (4 / 5) / (2 / 7)) = x * (7 / 5)) :=
by
  sorry

end operation_equivalence_l655_65581


namespace ratio_pat_mark_l655_65578

theorem ratio_pat_mark (P K M : ℕ) (h1 : P + K + M = 180) 
  (h2 : P = 2 * K) (h3 : M = K + 100) : P / gcd P M = 1 ∧ M / gcd P M = 3 := by
  sorry

end ratio_pat_mark_l655_65578


namespace geometric_sequence_solution_l655_65527

theorem geometric_sequence_solution (x : ℝ) (h : ∃ r : ℝ, 12 * r = x ∧ x * r = 3) : x = 6 ∨ x = -6 := 
by
  sorry

end geometric_sequence_solution_l655_65527


namespace roots_quadratic_diff_by_12_l655_65573

theorem roots_quadratic_diff_by_12 (P : ℝ) : 
  (∀ α β : ℝ, (α + β = 2) ∧ (α * β = -P) ∧ ((α - β) = 12)) → P = 35 := 
by
  intro h
  sorry

end roots_quadratic_diff_by_12_l655_65573


namespace number_of_boundaries_l655_65574

theorem number_of_boundaries 
  (total_runs : ℕ) 
  (number_of_sixes : ℕ) 
  (percentage_runs_by_running : ℝ) 
  (runs_per_six : ℕ) 
  (runs_per_boundary : ℕ)
  (h_total_runs : total_runs = 125)
  (h_number_of_sixes : number_of_sixes = 5)
  (h_percentage_runs_by_running : percentage_runs_by_running = 0.60)
  (h_runs_per_six : runs_per_six = 6)
  (h_runs_per_boundary : runs_per_boundary = 4) :
  (total_runs - percentage_runs_by_running * total_runs - number_of_sixes * runs_per_six) / runs_per_boundary = 5 := by 
  sorry

end number_of_boundaries_l655_65574


namespace proper_divisors_increased_by_one_l655_65541

theorem proper_divisors_increased_by_one
  (n : ℕ)
  (hn1 : 2 ≤ n)
  (exists_m : ∃ m : ℕ, ∀ d : ℕ, d ∣ n ∧ 1 < d ∧ d < n → d + 1 ∣ m ∧ d + 1 ≠ m)
  : n = 4 ∨ n = 8 :=
  sorry

end proper_divisors_increased_by_one_l655_65541


namespace number_of_ways_to_arrange_matches_l655_65516

open Nat

theorem number_of_ways_to_arrange_matches :
  (factorial 7) * (2 ^ 3) = 40320 := by
  sorry

end number_of_ways_to_arrange_matches_l655_65516


namespace sum_of_roots_l655_65587

theorem sum_of_roots (x1 x2 : ℝ) (h1 : x1^2 + 5*x1 - 3 = 0) (h2 : x2^2 + 5*x2 - 3 = 0) (h3 : x1 ≠ x2) :
  x1 + x2 = -5 :=
sorry

end sum_of_roots_l655_65587


namespace unitD_questionnaires_l655_65522

theorem unitD_questionnaires :
  ∀ (numA numB numC numD total_drawn : ℕ),
  (2 * numB = numA + numC) →  -- arithmetic sequence condition for B
  (2 * numC = numB + numD) →  -- arithmetic sequence condition for C
  (numA + numB + numC + numD = 1000) →  -- total number condition
  (total_drawn = 150) →  -- total drawn condition
  (numB = 30) →  -- unit B condition
  (total_drawn = (30 - d) + 30 + (30 + d) + (30 + 2 * d)) →
  (d = 15) →
  30 + 2 * d = 60 :=
by
  sorry

end unitD_questionnaires_l655_65522


namespace tom_total_payment_l655_65575

def fruit_cost (lemons papayas mangos : ℕ) : ℕ :=
  2 * lemons + 1 * papayas + 4 * mangos

def discount (total_fruits : ℕ) : ℕ :=
  total_fruits / 4

def total_cost_with_discount (lemons papayas mangos : ℕ) : ℕ :=
  let total_fruits := lemons + papayas + mangos
  fruit_cost lemons papayas mangos - discount total_fruits

theorem tom_total_payment :
  total_cost_with_discount 6 4 2 = 21 :=
  by
    sorry

end tom_total_payment_l655_65575


namespace find_m_value_l655_65559

theorem find_m_value (m : ℚ) :
  (m - 10) / -10 = (5 - m) / -8 → m = 65 / 9 :=
by
  sorry

end find_m_value_l655_65559


namespace arithmetic_sequence_terms_sum_l655_65562

theorem arithmetic_sequence_terms_sum
  (a : ℕ → ℝ)
  (h₁ : ∀ n, a (n+1) = a n + d)
  (h₂ : a 2 = 1 - a 1)
  (h₃ : a 4 = 9 - a 3)
  (h₄ : ∀ n, a n > 0):
  a 4 + a 5 = 27 :=
sorry

end arithmetic_sequence_terms_sum_l655_65562


namespace function_odd_on_domain_l655_65510

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

theorem function_odd_on_domain :
  ∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = -f x :=
by
  intros x h
  sorry

end function_odd_on_domain_l655_65510


namespace find_f_neg_5pi_over_6_l655_65571

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined_on_R : ∀ x : ℝ, ∃ y : ℝ, f y = f x
axiom f_periodic : ∀ x : ℝ, f (x + (3 * Real.pi / 2)) = f x
axiom f_on_interval : ∀ x : ℝ, 0 ≤ x → x ≤ Real.pi → f x = Real.cos x

theorem find_f_neg_5pi_over_6 : f (-5 * Real.pi / 6) = -1 / 2 := 
by 
  -- use the axioms to prove the result 
  sorry

end find_f_neg_5pi_over_6_l655_65571


namespace number_of_substitution_ways_mod_1000_l655_65598

theorem number_of_substitution_ways_mod_1000 :
  let a_0 := 1
  let a_1 := 12 * 12 * a_0
  let a_2 := 12 * 11 * a_1
  let a_3 := 12 * 10 * a_2
  let a_4 := 12 * 9 * a_3
  let total_ways := a_0 + a_1 + a_2 + a_3 + a_4
  total_ways % 1000 = 573 := by
  -- Definition
  let a_0 := 1
  let a_1 := 12 * 12 * a_0
  let a_2 := 12 * 11 * a_1
  let a_3 := 12 * 10 * a_2
  let a_4 := 12 * 9 * a_3
  let total_ways := a_0 + a_1 + a_2 + a_3 + a_4
  -- Proof is omitted
  sorry

end number_of_substitution_ways_mod_1000_l655_65598


namespace bowling_ball_weight_l655_65561

theorem bowling_ball_weight (b c : ℝ) (h1 : 5 * b = 3 * c) (h2 : 2 * c = 56) : b = 16.8 := by
  sorry

end bowling_ball_weight_l655_65561


namespace ratio_proof_l655_65547

theorem ratio_proof (a b c d : ℝ) (h1 : b = 3 * a) (h2 : c = 4 * b) (h3 : d = 2 * b - a) :
  (a + b + d) / (b + c + d) = 9 / 20 :=
by sorry

end ratio_proof_l655_65547


namespace FashionDesignNotInServiceAreas_l655_65505

-- Define the service areas of Digital China
def ServiceAreas (x : String) : Prop :=
  x = "Understanding the situation of soil and water loss in the Yangtze River Basin" ∨
  x = "Understanding stock market trends" ∨
  x = "Wanted criminals"

-- Prove that "Fashion design" is not in the service areas of Digital China
theorem FashionDesignNotInServiceAreas : ¬ ServiceAreas "Fashion design" :=
sorry

end FashionDesignNotInServiceAreas_l655_65505


namespace part1_part2_l655_65508

-- Definitions for part 1
def prop_p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def prop_q (x : ℝ) : Prop := (x - 3) / (x + 2) < 0

-- Definitions for part 2
def neg_prop_q (x : ℝ) : Prop := ¬((x - 3) / (x + 2) < 0)
def neg_prop_p (a x : ℝ) : Prop := ¬(x^2 - 4*a*x + 3*a^2 < 0)

-- Proof problems
theorem part1 (a : ℝ) (x : ℝ) (h : a = 1) (hpq : prop_p a x ∧ prop_q x) : 1 < x ∧ x < 3 := 
by
  sorry

theorem part2 (a : ℝ) (h : ∀ x, neg_prop_q x → neg_prop_p a x) : 0 < a ∧ a ≤ 1 :=
by
  sorry

end part1_part2_l655_65508


namespace ratio_sum_of_arithmetic_sequences_l655_65523

-- Definitions for the arithmetic sequences
def a_num := 3
def d_num := 3
def l_num := 99

def a_den := 4
def d_den := 4
def l_den := 96

-- Number of terms in each sequence
def n_num := (l_num - a_num) / d_num + 1
def n_den := (l_den - a_den) / d_den + 1

-- Sum of the sequences using the sum formula for arithmetic series
def S_num := n_num * (a_num + l_num) / 2
def S_den := n_den * (a_den + l_den) / 2

-- The theorem statement
theorem ratio_sum_of_arithmetic_sequences : S_num / S_den = 1683 / 1200 := by sorry

end ratio_sum_of_arithmetic_sequences_l655_65523


namespace base_4_digits_l655_65579

theorem base_4_digits (b : ℕ) (h1 : b^3 ≤ 216) (h2 : 216 < b^4) : b = 5 :=
sorry

end base_4_digits_l655_65579


namespace ratio_proof_l655_65572

variable {x y : ℝ}

theorem ratio_proof (h : x / y = 2 / 3) : x / (x + y) = 2 / 5 :=
by
  sorry

end ratio_proof_l655_65572


namespace value_bounds_of_expression_l655_65593

theorem value_bounds_of_expression
  (a b c : ℝ)
  (ha : 0 ≤ a)
  (hab : a ≤ b)
  (hbc : b ≤ c)
  (triangle_ineq1 : a + b > c)
  (triangle_ineq2 : a + c > b)
  (triangle_ineq3 : b + c > a)
  : 4 ≤ (a+b+c)^2 / (b*c) ∧ (a+b+c)^2 / (b*c) ≤ 9 := sorry

end value_bounds_of_expression_l655_65593


namespace find_cost_of_apple_l655_65596

theorem find_cost_of_apple (A O : ℝ) 
  (h1 : 6 * A + 3 * O = 1.77) 
  (h2 : 2 * A + 5 * O = 1.27) : 
  A = 0.21 :=
by 
  sorry

end find_cost_of_apple_l655_65596


namespace correct_result_l655_65515

theorem correct_result (x : ℕ) (h : x + 65 = 125) : x + 95 = 155 :=
sorry

end correct_result_l655_65515


namespace find_n_l655_65530

theorem find_n (n : ℤ) (h : n * 1296 / 432 = 36) : n = 12 :=
sorry

end find_n_l655_65530


namespace parabola_point_comparison_l655_65554

theorem parabola_point_comparison :
  let y1 := (1: ℝ)^2 - 2 * (1: ℝ) - 2
  let y2 := (3: ℝ)^2 - 2 * (3: ℝ) - 2
  y1 < y2 :=
by
  let y1 := (1: ℝ)^2 - 2 * (1: ℝ) - 2
  let y2 := (3: ℝ)^2 - 2 * (3: ℝ) - 2
  have h : y1 < y2 := by sorry
  exact h

end parabola_point_comparison_l655_65554


namespace total_number_of_cantelopes_l655_65553

def number_of_cantelopes_fred : ℕ := 38
def number_of_cantelopes_tim : ℕ := 44

theorem total_number_of_cantelopes : number_of_cantelopes_fred + number_of_cantelopes_tim = 82 := by
  sorry

end total_number_of_cantelopes_l655_65553


namespace probability_neither_red_nor_purple_correct_l655_65567

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 10
def yellow_balls : ℕ := 7
def red_balls : ℕ := 15
def purple_balls : ℕ := 6

def neither_red_nor_purple_balls : ℕ := total_balls - (red_balls + purple_balls)
def probability_neither_red_nor_purple : ℚ := (neither_red_nor_purple_balls : ℚ) / (total_balls : ℚ)

theorem probability_neither_red_nor_purple_correct : 
  probability_neither_red_nor_purple = 13 / 20 := 
by sorry

end probability_neither_red_nor_purple_correct_l655_65567


namespace minimum_value_of_f_l655_65506

variable (a k : ℝ)
variable (k_gt_1 : k > 1)
variable (a_gt_0 : a > 0)

noncomputable def f (x : ℝ) : ℝ := k * Real.sqrt (a^2 + x^2) - x

theorem minimum_value_of_f : ∃ x_0, ∀ x, f a k x ≥ f a k x_0 ∧ f a k x_0 = a * Real.sqrt (k^2 - 1) :=
by
  sorry

end minimum_value_of_f_l655_65506


namespace geometric_proportion_l655_65589

theorem geometric_proportion (a b c d : ℝ) (h1 : a / b = c / d) (h2 : a / b = d / c) :
  (a = b ∧ b = c ∧ c = d) ∨ (|a| = |b| ∧ |b| = |c| ∧ |c| = |d| ∧ (a * b * c * d < 0)) :=
by
  sorry

end geometric_proportion_l655_65589


namespace portion_of_work_done_l655_65568

variable (P W : ℕ)

-- Given conditions
def work_rate_P (P W : ℕ) : ℕ := W / 16
def work_rate_2P (P W : ℕ) : ℕ := 2 * (work_rate_P P W)

-- Lean theorem
theorem portion_of_work_done (h : work_rate_2P P W * 4 = W / 2) : 
    work_rate_2P P W * 4 = W / 2 := 
by 
  sorry

end portion_of_work_done_l655_65568


namespace integral_2x_minus_1_eq_6_l655_65543

noncomputable def definite_integral_example : ℝ :=
  ∫ x in (0:ℝ)..(3:ℝ), (2 * x - 1)

theorem integral_2x_minus_1_eq_6 : definite_integral_example = 6 :=
by
  sorry

end integral_2x_minus_1_eq_6_l655_65543


namespace books_a_count_l655_65524

-- Variables representing the number of books (a) and (b)
variables (A B : ℕ)

-- Conditions given in the problem
def condition1 : Prop := A + B = 20
def condition2 : Prop := A = B + 4

-- The theorem to prove
theorem books_a_count (h1 : condition1 A B) (h2 : condition2 A B) : A = 12 :=
sorry

end books_a_count_l655_65524


namespace triangle_cosine_rule_c_triangle_tangent_C_l655_65549

-- Define a proof statement for the cosine rule-based proof of c = 4.
theorem triangle_cosine_rule_c (a b : ℝ) (angleB : ℝ) (ha : a = 2)
                              (hb : b = 2 * Real.sqrt 3) (hB : angleB = π / 3) :
  ∃ (c : ℝ), c = 4 := by
  sorry

-- Define a proof statement for the tangent identity-based proof of tan C = 3 * sqrt 3 / 5.
theorem triangle_tangent_C (tanA : ℝ) (tanB : ℝ) (htA : tanA = 2 * Real.sqrt 3)
                           (htB : tanB = Real.sqrt 3) :
  ∃ (tanC : ℝ), tanC = 3 * Real.sqrt 3 / 5 := by
  sorry

end triangle_cosine_rule_c_triangle_tangent_C_l655_65549


namespace sin_beta_acute_l655_65590

theorem sin_beta_acute (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (hcosα : Real.cos α = 4 / 5)
  (hcosαβ : Real.cos (α + β) = 5 / 13) :
  Real.sin β = 33 / 65 :=
sorry

end sin_beta_acute_l655_65590


namespace same_terminal_side_l655_65545

theorem same_terminal_side
  (k : ℤ)
  (angle1 := (π / 5))
  (angle2 := (21 * π / 5)) :
  ∃ k : ℤ, angle2 = 2 * k * π + angle1 := by
  sorry

end same_terminal_side_l655_65545


namespace square_side_length_l655_65519

theorem square_side_length (s : ℝ) (h : s^2 = 9/16) : s = 3/4 :=
sorry

end square_side_length_l655_65519


namespace total_age_of_siblings_l655_65538

def age_total (Susan Arthur Tom Bob : ℕ) : ℕ := Susan + Arthur + Tom + Bob

theorem total_age_of_siblings :
  ∀ (Susan Arthur Tom Bob : ℕ),
    (Arthur = Susan + 2) →
    (Tom = Bob - 3) →
    (Bob = 11) →
    (Susan = 15) →
    age_total Susan Arthur Tom Bob = 51 :=
by
  intros Susan Arthur Tom Bob h1 h2 h3 h4
  rw [h4, h1, h3, h2]    -- Use the conditions
  norm_num               -- Simplify numerical expressions
  sorry                  -- Placeholder for the proof

end total_age_of_siblings_l655_65538


namespace lcm_of_4_9_10_27_l655_65566

theorem lcm_of_4_9_10_27 : Nat.lcm (Nat.lcm 4 9) (Nat.lcm 10 27) = 540 :=
by
  sorry

end lcm_of_4_9_10_27_l655_65566


namespace plane_eq_passing_A_perpendicular_BC_l655_65507

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def subtract_points (P Q : Point3D) : Point3D :=
  { x := P.x - Q.x, y := P.y - Q.y, z := P.z - Q.z }

-- Points A, B, and C given in the conditions
def A : Point3D := { x := 1, y := -5, z := -2 }
def B : Point3D := { x := 6, y := -2, z := 1 }
def C : Point3D := { x := 2, y := -2, z := -2 }

-- Vector BC
def BC : Point3D := subtract_points C B

theorem plane_eq_passing_A_perpendicular_BC :
  (-4 : ℝ) * (A.x - 1) + (0 : ℝ) * (A.y + 5) + (-3 : ℝ) * (A.z + 2) = 0 :=
  sorry

end plane_eq_passing_A_perpendicular_BC_l655_65507


namespace class_size_count_l655_65511

theorem class_size_count : 
  ∃ (n : ℕ), 
  n = 6 ∧ 
  (∀ (b g : ℕ), (2 < b ∧ b < 10) → (14 < g ∧ g < 23) → b + g > 25 → 
    ∃ (sizes : Finset ℕ), sizes.card = n ∧ 
    ∀ (s : ℕ), s ∈ sizes → (∃ (b' g' : ℕ), s = b' + g' ∧ s > 25)) :=
sorry

end class_size_count_l655_65511


namespace semesters_per_year_l655_65586

-- Definitions of conditions
def cost_per_semester : ℕ := 20000
def total_cost_13_years : ℕ := 520000
def years : ℕ := 13

-- Main theorem to prove
theorem semesters_per_year (S : ℕ) (h1 : total_cost_13_years = years * (S * cost_per_semester)) : S = 2 := by
  sorry

end semesters_per_year_l655_65586


namespace cody_increases_steps_by_1000_l655_65576

theorem cody_increases_steps_by_1000 (x : ℕ) 
  (initial_steps : ℕ := 7000)
  (steps_logged_in_four_weeks : ℕ := 70000)
  (goal_steps : ℕ := 100000)
  (remaining_steps : ℕ := 30000)
  (condition : 1000 + 7 * (1 + 2 + 3) * x = 70000 → x = 1000) : x = 1000 :=
by
  sorry

end cody_increases_steps_by_1000_l655_65576


namespace price_per_glass_second_day_l655_65539

-- Given conditions
variables {O P : ℝ}
axiom condition1 : 0.82 * 2 * O = P * 3 * O

-- Problem statement
theorem price_per_glass_second_day : 
  P = 0.55 :=
by
  -- This is where the actual proof would go
  sorry

end price_per_glass_second_day_l655_65539


namespace original_number_l655_65584

-- Define the original statement and conditions
theorem original_number (x : ℝ) (h : 3 * (2 * x + 9) = 81) : x = 9 := by
  -- Sorry placeholder stands for the proof steps
  sorry

end original_number_l655_65584


namespace yura_picture_dimensions_l655_65552

theorem yura_picture_dimensions (l w : ℕ) (h_frame : (l + 2) * (w + 2) - l * w = l * w) :
    (l = 3 ∧ w = 10) ∨ (l = 4 ∧ w = 6) :=
by {
  sorry
}

end yura_picture_dimensions_l655_65552


namespace verify_equation_holds_l655_65565

noncomputable def verify_equation (m n : ℝ) : Prop :=
  1.55 * Real.sqrt (6 * m + 2 * Real.sqrt (9 * m^2 - n^2)) 
  - Real.sqrt (6 * m - 2 * Real.sqrt (9 * m^2 - n^2)) 
  = 2 * Real.sqrt (3 * m - n)

theorem verify_equation_holds (m n : ℝ) (h : 9 * m^2 - n^2 ≥ 0) : verify_equation m n :=
by
  -- Proof goes here. 
  -- Implement the proof as per the solution steps sketched in the problem statement.
  sorry

end verify_equation_holds_l655_65565


namespace pow_mod_1000_of_6_eq_296_l655_65520

theorem pow_mod_1000_of_6_eq_296 : (6 ^ 1993) % 1000 = 296 := by
  sorry

end pow_mod_1000_of_6_eq_296_l655_65520


namespace probability_losing_ticket_l655_65557

theorem probability_losing_ticket (winning : ℕ) (losing : ℕ)
  (h_odds : winning = 5 ∧ losing = 8) :
  (losing : ℚ) / (winning + losing : ℚ) = 8 / 13 := by
  sorry

end probability_losing_ticket_l655_65557


namespace g_at_5_l655_65560

def g (x : ℝ) : ℝ := 2 * x^4 - 17 * x^3 + 28 * x^2 - 20 * x - 80

theorem g_at_5 : g 5 = -5 := 
  by 
  -- Proof goes here
  sorry

end g_at_5_l655_65560


namespace find_age_of_b_l655_65599

-- Definitions for the conditions
def is_two_years_older (a b : ℕ) : Prop := a = b + 2
def is_twice_as_old (b c : ℕ) : Prop := b = 2 * c
def total_age (a b c : ℕ) : Prop := a + b + c = 12

-- Proof statement
theorem find_age_of_b (a b c : ℕ) 
  (h1 : is_two_years_older a b) 
  (h2 : is_twice_as_old b c) 
  (h3 : total_age a b c) : 
  b = 4 := 
by 
  sorry

end find_age_of_b_l655_65599


namespace equal_area_split_l655_65532

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def circle1 : Circle := { center := (10, 90), radius := 4 }
def circle2 : Circle := { center := (15, 80), radius := 4 }
def circle3 : Circle := { center := (20, 85), radius := 4 }

theorem equal_area_split :
  ∃ m : ℝ, ∀ x y : ℝ, m * (x - 15) = y - 80 ∧ m = 0 ∧   
    ∀ circle : Circle, circle ∈ [circle1, circle2, circle3] →
      ∃ k : ℝ, k * (x - circle.center.1) + y - circle.center.2 = 0 :=
sorry

end equal_area_split_l655_65532


namespace min_value_a_plus_b_plus_c_l655_65563

theorem min_value_a_plus_b_plus_c 
  (a b c : ℕ) 
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_c_pos : c > 0)
  (x1 x2 : ℝ)
  (hx1_neg : -1 < x1)
  (hx1_pos : x1 < 0)
  (hx2_neg : 0 < x2)
  (hx2_pos : x2 < 1)
  (h_distinct : x1 ≠ x2)
  (h_eqn_x1 : a * x1^2 + b * x1 + c = 0)
  (h_eqn_x2 : a * x2^2 + b * x2 + c = 0) :
  a + b + c = 11 :=
sorry

end min_value_a_plus_b_plus_c_l655_65563


namespace angle_B_eq_18_l655_65582

theorem angle_B_eq_18 
  (A B : ℝ) 
  (h1 : A = 4 * B) 
  (h2 : 90 - B = 4 * (90 - A)) : 
  B = 18 :=
by
  sorry

end angle_B_eq_18_l655_65582


namespace equation_has_real_roots_l655_65503

theorem equation_has_real_roots (a b : ℝ) (h : ¬ (a = 0 ∧ b = 0)) :
  ∃ x : ℝ, x ≠ 1 ∧ (a^2 / x + b^2 / (x - 1) = 1) :=
by
  sorry

end equation_has_real_roots_l655_65503


namespace tree_heights_l655_65502

theorem tree_heights (h : ℕ) (ratio : 5 / 7 = (h - 20) / h) : h = 70 :=
sorry

end tree_heights_l655_65502


namespace log12_eq_abc_l655_65569

theorem log12_eq_abc (a b : ℝ) (h1 : a = Real.log 7 / Real.log 6) (h2 : b = Real.log 4 / Real.log 3) : 
  Real.log 7 / Real.log 12 = (a * b + 2 * a) / (2 * b + 2) :=
by
  sorry

end log12_eq_abc_l655_65569


namespace number_of_students_at_end_of_year_l655_65514

def students_at_start_of_year : ℕ := 35
def students_left_during_year : ℕ := 10
def students_joined_during_year : ℕ := 10

theorem number_of_students_at_end_of_year : students_at_start_of_year - students_left_during_year + students_joined_during_year = 35 :=
by
  sorry -- Proof goes here

end number_of_students_at_end_of_year_l655_65514


namespace arrangement_count_l655_65525

-- Define the sets of books
def italian_books : Finset String := { "I1", "I2", "I3" }
def german_books : Finset String := { "G1", "G2", "G3" }
def french_books : Finset String := { "F1", "F2", "F3", "F4", "F5" }

-- Define the arrangement count as a noncomputable definition, because we are going to use factorial which involves an infinite structure
noncomputable def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Prove the required arrangement
theorem arrangement_count : 
  (factorial 3) * ((factorial 3) * (factorial 3) * (factorial 5)) = 25920 := 
by
  -- Provide the solution steps here (omitted for now)
  sorry

end arrangement_count_l655_65525


namespace solved_just_B_is_six_l655_65526

variables (a b c d e f g : ℕ)

-- Conditions given
axiom total_competitors : a + b + c + d + e + f + g = 25
axiom twice_as_many_solved_B : b + d = 2 * (c + d)
axiom only_A_one_more : a = 1 + (e + f + g)
axiom A_equals_B_plus_C : a = b + c

-- Prove that the number of competitors solving just problem B is 6.
theorem solved_just_B_is_six : b = 6 :=
by
  sorry

end solved_just_B_is_six_l655_65526


namespace percentage_of_seeds_germinated_l655_65518

theorem percentage_of_seeds_germinated (P1 P2 : ℕ) (GP1 GP2 : ℕ) (SP1 SP2 TotalGerminated TotalPlanted : ℕ) (PG : ℕ) 
  (h1 : P1 = 300) (h2 : P2 = 200) (h3 : GP1 = 60) (h4 : GP2 = 70) (h5 : SP1 = P1) (h6 : SP2 = P2)
  (h7 : TotalGerminated = GP1 + GP2) (h8 : TotalPlanted = SP1 + SP2) : 
  PG = (TotalGerminated * 100) / TotalPlanted :=
sorry

end percentage_of_seeds_germinated_l655_65518


namespace decipher_rebus_l655_65536

theorem decipher_rebus (a b c d : ℕ) :
  (a = 10 ∧ b = 14 ∧ c = 12 ∧ d = 13) ↔
  (∀ (x y z w: ℕ), 
    (x = 10 → 5 + 5 * 7 = 49) ∧
    (y = 14 → 2 - 4 * 3 = 9) ∧
    (z = 12 → 12 - 1 - 1 * 2 = 20) ∧
    (w = 13 → 13 - 1 + 10 - 5 = 17) ∧
    (49 + 9 + 20 + 17 = 95)) :=
by sorry

end decipher_rebus_l655_65536


namespace div_100_by_a8_3a4_minus_4_l655_65535

theorem div_100_by_a8_3a4_minus_4 (a : ℕ) (h : ¬ (5 ∣ a)) : 100 ∣ (a^8 + 3 * a^4 - 4) :=
sorry

end div_100_by_a8_3a4_minus_4_l655_65535


namespace negation_all_swans_white_l655_65504

variables {α : Type} (swan white : α → Prop)

theorem negation_all_swans_white :
  (¬ ∀ x, swan x → white x) ↔ (∃ x, swan x ∧ ¬ white x) :=
by {
  sorry
}

end negation_all_swans_white_l655_65504


namespace smallest_percent_both_l655_65594

theorem smallest_percent_both (S J : ℝ) (hS : S = 0.9) (hJ : J = 0.8) : 
  ∃ B, B = S + J - 1 ∧ B = 0.7 :=
by
  sorry

end smallest_percent_both_l655_65594


namespace josh_money_left_l655_65570

def initial_amount : ℝ := 9
def spent_on_drink : ℝ := 1.75
def spent_on_item : ℝ := 1.25

theorem josh_money_left : initial_amount - (spent_on_drink + spent_on_item) = 6 := by
  sorry

end josh_money_left_l655_65570
