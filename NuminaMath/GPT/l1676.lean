import Mathlib

namespace NUMINAMATH_GPT_trajectory_of_B_l1676_167671

-- Define the points and the line for the given conditions
def A : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (2, -3)
def D_line (x : ℝ) (y : ℝ) : Prop := 3 * x - y + 1 = 0

-- Define the statement to be proved
theorem trajectory_of_B (x y : ℝ) :
  D_line x y → ∃ Bx By, (3 * Bx - By - 20 = 0) :=
sorry

end NUMINAMATH_GPT_trajectory_of_B_l1676_167671


namespace NUMINAMATH_GPT_size_of_angle_C_max_value_of_a_add_b_l1676_167681

variable (A B C a b c : ℝ)
variable (h₀ : 0 < A ∧ A < π / 2)
variable (h₁ : 0 < B ∧ B < π / 2)
variable (h₂ : 0 < C ∧ C < π / 2)
variable (h₃ : a = 2 * c * sin A / sqrt 3)
variable (h₄ : a * a + b * b - 2 * a * b * cos (π / 3) = c * c)

theorem size_of_angle_C (h₅: a ≠ 0):
  C = π / 3 :=
by sorry

theorem max_value_of_a_add_b (h₆: c = 2):
  a + b ≤ 4 :=
by sorry

end NUMINAMATH_GPT_size_of_angle_C_max_value_of_a_add_b_l1676_167681


namespace NUMINAMATH_GPT_fraction_value_l1676_167666

variable (a b : ℚ)  -- Variables a and b are rational numbers

theorem fraction_value (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_fraction_value_l1676_167666


namespace NUMINAMATH_GPT_percentage_calculation_l1676_167621

-- Definitions based on conditions
def x : ℕ := 5200
def p1 : ℚ := 0.50
def p2 : ℚ := 0.30
def p3 : ℚ := 0.15

-- The theorem stating the desired proof
theorem percentage_calculation : p3 * (p2 * (p1 * x)) = 117 := by
  sorry

end NUMINAMATH_GPT_percentage_calculation_l1676_167621


namespace NUMINAMATH_GPT_stuffed_animal_cost_l1676_167659

variables 
  (M S A A_single C : ℝ)
  (Coupon_discount : ℝ)
  (Maximum_budget : ℝ)

noncomputable def conditions : Prop :=
  M = 6 ∧
  M = 3 * S ∧
  M = A / 4 ∧
  A_single = A / 2 ∧
  C = A_single / 2 ∧
  C = 2 * S ∧
  Coupon_discount = 0.10 ∧
  Maximum_budget = 30

theorem stuffed_animal_cost (h : conditions M S A A_single C Coupon_discount Maximum_budget) :
  A_single = 12 :=
sorry

end NUMINAMATH_GPT_stuffed_animal_cost_l1676_167659


namespace NUMINAMATH_GPT_friday_can_determine_arrival_date_l1676_167632

-- Define the conditions
def Robinson_crusoe (day : ℕ) : Prop := day % 365 = 0

-- Goal: Within 183 days, Friday can determine his arrival date.
theorem friday_can_determine_arrival_date : 
  (∀ day : ℕ, day < 183 → (Robinson_crusoe day ↔ ¬ Robinson_crusoe (day + 1)) ∨ (day % 365 = 0)) :=
sorry

end NUMINAMATH_GPT_friday_can_determine_arrival_date_l1676_167632


namespace NUMINAMATH_GPT_area_of_shaded_region_l1676_167677

theorem area_of_shaded_region 
    (large_side : ℝ) (small_side : ℝ)
    (h_large : large_side = 10) 
    (h_small : small_side = 4) : 
    (large_side^2 - small_side^2) / 4 = 21 :=
by
  -- All proof steps are to be completed and checked,
  -- and sorry is used as placeholder for the final proof.
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l1676_167677


namespace NUMINAMATH_GPT_cosine_eq_one_fifth_l1676_167648

theorem cosine_eq_one_fifth {α : ℝ} 
  (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : 
  Real.cos α = 1 / 5 := 
sorry

end NUMINAMATH_GPT_cosine_eq_one_fifth_l1676_167648


namespace NUMINAMATH_GPT_y_z_add_x_eq_160_l1676_167654

theorem y_z_add_x_eq_160 (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x * (y + z) = 132) (h5 : z * (x + y) = 180) (h6 : x * y * z = 160) :
  y * (z + x) = 160 := 
by 
  sorry

end NUMINAMATH_GPT_y_z_add_x_eq_160_l1676_167654


namespace NUMINAMATH_GPT_eunji_initial_money_l1676_167640

-- Define the conditions
def snack_cost : ℕ := 350
def allowance : ℕ := 800
def money_left_after_pencil : ℕ := 550

-- Define what needs to be proven
theorem eunji_initial_money (initial_money : ℕ) :
  initial_money - snack_cost + allowance = money_left_after_pencil * 2 →
  initial_money = 650 :=
by
  sorry

end NUMINAMATH_GPT_eunji_initial_money_l1676_167640


namespace NUMINAMATH_GPT_minimum_questions_to_determine_village_l1676_167653

-- Step 1: Define the types of villages
inductive Village
| A : Village
| B : Village
| C : Village

-- Step 2: Define the properties of residents in each village
def tells_truth (v : Village) (p : Prop) : Prop :=
  match v with
  | Village.A => p
  | Village.B => ¬p
  | Village.C => p ∨ ¬p

-- Step 3: Define the problem context in Lean
theorem minimum_questions_to_determine_village :
    ∀ (tourist_village person_village : Village), ∃ (n : ℕ), n = 4 := by
  sorry

end NUMINAMATH_GPT_minimum_questions_to_determine_village_l1676_167653


namespace NUMINAMATH_GPT_time_to_cross_bridge_l1676_167647

theorem time_to_cross_bridge (speed_km_hr : ℝ) (length_m : ℝ) (speed_conversion_factor : ℝ) (time_conversion_factor : ℝ) (expected_time : ℝ) :
  speed_km_hr = 5 →
  length_m = 1250 →
  speed_conversion_factor = 1000 →
  time_conversion_factor = 60 →
  expected_time = length_m / (speed_km_hr * (speed_conversion_factor / time_conversion_factor)) →
  expected_time = 15 :=
by
  intros
  sorry

end NUMINAMATH_GPT_time_to_cross_bridge_l1676_167647


namespace NUMINAMATH_GPT_simplest_quadratic_radicals_l1676_167618

theorem simplest_quadratic_radicals (a : ℝ) :
  (3 * a - 8 ≥ 0) ∧ (17 - 2 * a ≥ 0) → a = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_simplest_quadratic_radicals_l1676_167618


namespace NUMINAMATH_GPT_length_AD_l1676_167665

open Real

-- Define the properties of the quadrilateral
variable (A B C D: Point)
variable (angle_ABC angle_BCD: ℝ)
variable (AB BC CD: ℝ)

-- Given conditions
axiom angle_ABC_eq_135 : angle_ABC = 135 * π / 180
axiom angle_BCD_eq_120 : angle_BCD = 120 * π / 180
axiom AB_eq_sqrt_6 : AB = sqrt 6
axiom BC_eq_5_minus_sqrt_3 : BC = 5 - sqrt 3
axiom CD_eq_6 : CD = 6

-- The theorem to prove
theorem length_AD {AD : ℝ} (h : True) :
  AD = 2 * sqrt 19 :=
sorry

end NUMINAMATH_GPT_length_AD_l1676_167665


namespace NUMINAMATH_GPT_right_triangle_perimeter_l1676_167651

theorem right_triangle_perimeter (a b : ℕ) (h : a^2 + b^2 = 100) (r : ℕ := 1) :
  (a + b + 10) = 24 :=
sorry

end NUMINAMATH_GPT_right_triangle_perimeter_l1676_167651


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l1676_167612

-- Definition and proof for System (1)
theorem system1_solution (x y : ℝ) (h1 : x - y = 2) (h2 : 2 * x + y = 7) : x = 3 ∧ y = 1 := 
by 
  sorry

-- Definition and proof for System (2)
theorem system2_solution (x y : ℝ) (h1 : x - 2 * y = 3) (h2 : (1 / 2) * x + (3 / 4) * y = 13 / 4) : x = 5 ∧ y = 1 :=
by 
  sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l1676_167612


namespace NUMINAMATH_GPT_systematic_sampling_student_number_l1676_167631

theorem systematic_sampling_student_number 
  (total_students : ℕ)
  (sample_size : ℕ)
  (interval_between_numbers : ℕ)
  (student_17_in_sample : ∃ n, 17 = n ∧ n ≤ total_students ∧ n % interval_between_numbers = 5)
  : ∃ m, m = 41 ∧ m ≤ total_students ∧ m % interval_between_numbers = 5 := 
sorry

end NUMINAMATH_GPT_systematic_sampling_student_number_l1676_167631


namespace NUMINAMATH_GPT_ratio_of_parallel_vectors_l1676_167614

theorem ratio_of_parallel_vectors (m n : ℝ) 
  (h1 : ∃ k : ℝ, (m, 1, 3) = (k * 2, k * n, k)) : (m / n) = 18 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_parallel_vectors_l1676_167614


namespace NUMINAMATH_GPT_lloyd_normal_hours_l1676_167683

-- Definitions based on the conditions
def regular_rate : ℝ := 3.50
def overtime_rate : ℝ := 1.5 * regular_rate
def total_hours_worked : ℝ := 10.5
def total_earnings : ℝ := 42
def normal_hours_worked (h : ℝ) : Prop := 
  h * regular_rate + (total_hours_worked - h) * overtime_rate = total_earnings

-- The theorem to prove
theorem lloyd_normal_hours : ∃ h : ℝ, normal_hours_worked h ∧ h = 7.5 := sorry

end NUMINAMATH_GPT_lloyd_normal_hours_l1676_167683


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1676_167699

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x ≥ 1 → |x + 1| + |x - 1| = 2 * |x|)
  ∧ (∃ y : ℝ, ¬ (y ≥ 1) ∧ |y + 1| + |y - 1| = 2 * |y|) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1676_167699


namespace NUMINAMATH_GPT_sector_central_angle_in_radians_l1676_167626

/-- 
Given a sector of a circle where the perimeter is 4 cm 
and the area is 1 cm², prove that the central angle 
of the sector in radians is 2.
-/
theorem sector_central_angle_in_radians 
  (r l : ℝ) 
  (h_perimeter : 2 * r + l = 4) 
  (h_area : (1 / 2) * l * r = 1) : 
  l / r = 2 :=
by
  sorry

end NUMINAMATH_GPT_sector_central_angle_in_radians_l1676_167626


namespace NUMINAMATH_GPT_probability_red_or_white_l1676_167675

noncomputable def total_marbles : ℕ := 50
noncomputable def blue_marbles : ℕ := 5
noncomputable def red_marbles : ℕ := 9
noncomputable def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

theorem probability_red_or_white : 
  (red_marbles + white_marbles) / total_marbles = 9 / 10 :=
by sorry

end NUMINAMATH_GPT_probability_red_or_white_l1676_167675


namespace NUMINAMATH_GPT_remainder_example_l1676_167682

def P (x : ℝ) := 8 * x^3 - 20 * x^2 + 28 * x - 26
def D (x : ℝ) := 4 * x - 8

theorem remainder_example : P 2 = 14 :=
by
  sorry

end NUMINAMATH_GPT_remainder_example_l1676_167682


namespace NUMINAMATH_GPT_no_odd_total_rows_columns_l1676_167634

open Function

def array_odd_column_row_count (n : ℕ) (array : ℕ → ℕ → ℤ) : Prop :=
  n % 2 = 1 ∧
  (∀ i j, 0 ≤ array i j ∧ array i j ≤ 1 ∧ array i j = -1 ∨ array i j = 1) →
  (∃ (rows cols : Finset ℕ),
    rows.card + cols.card = n ∧
    ∀ r ∈ rows, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array r) k = -1 ∧
    ∀ c ∈ cols, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array c) k = -1
    )

theorem no_odd_total_rows_columns (n : ℕ) (array : ℕ → ℕ → ℤ) :
  n % 2 = 1 →
  (∀ i j, 0 ≤ array i j ∧ array i j ≤ 1 ∧ (array i j = -1 ∨ array i j = 1)) →
  ¬ (∃ rows cols : Finset ℕ,
       rows.card + cols.card = n ∧
       ∀ r ∈ rows, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array r k = -1) ∧
       ∀ c ∈ cols, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array c k = -1)) :=
by
  intros h_array
  sorry

end NUMINAMATH_GPT_no_odd_total_rows_columns_l1676_167634


namespace NUMINAMATH_GPT_percentage_increase_in_expenses_l1676_167617

variable (a b c : ℝ)

theorem percentage_increase_in_expenses :
  (10 / 100 * a + 30 / 100 * b + 20 / 100 * c) / (a + b + c) =
  (10 * a + 30 * b + 20 * c) / (100 * (a + b + c)) :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_in_expenses_l1676_167617


namespace NUMINAMATH_GPT_remi_water_intake_l1676_167680

def bottle_capacity := 20
def daily_refills := 3
def num_days := 7
def spill1 := 5
def spill2 := 8

def daily_intake := daily_refills * bottle_capacity
def total_intake_without_spill := daily_intake * num_days
def total_spill := spill1 + spill2
def total_intake_with_spill := total_intake_without_spill - total_spill

theorem remi_water_intake : total_intake_with_spill = 407 := 
by
  -- Provide proof here
  sorry

end NUMINAMATH_GPT_remi_water_intake_l1676_167680


namespace NUMINAMATH_GPT_find_f3_l1676_167689

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 6

theorem find_f3 (a b c : ℝ) (h : f a b c (-3) = -12) : f a b c 3 = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_f3_l1676_167689


namespace NUMINAMATH_GPT_quadratic_real_roots_l1676_167658

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 4 * x - 1 = 0) ↔ m ≥ -3 ∧ m ≠ 1 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l1676_167658


namespace NUMINAMATH_GPT_sum_of_geometric_ratios_l1676_167667

theorem sum_of_geometric_ratios (k a2 a3 b2 b3 p r : ℝ)
  (h_seq1 : a2 = k * p)
  (h_seq2 : a3 = k * p^2)
  (h_seq3 : b2 = k * r)
  (h_seq4 : b3 = k * r^2)
  (h_diff : a3 - b3 = 3 * (a2 - b2) - k) :
  p + r = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_geometric_ratios_l1676_167667


namespace NUMINAMATH_GPT_value_of_f_at_3_l1676_167663

def f (x : ℝ) := 2 * x - 1

theorem value_of_f_at_3 : f 3 = 5 := by
  sorry

end NUMINAMATH_GPT_value_of_f_at_3_l1676_167663


namespace NUMINAMATH_GPT_composite_expression_l1676_167692

theorem composite_expression (n : ℕ) (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 3^(2*n+1) - 2^(2*n+1) - 6^n = a * b :=
sorry

end NUMINAMATH_GPT_composite_expression_l1676_167692


namespace NUMINAMATH_GPT_determine_num_chickens_l1676_167635

def land_acres : ℕ := 30
def land_cost_per_acre : ℕ := 20
def house_cost : ℕ := 120000
def num_cows : ℕ := 20
def cow_cost_per_cow : ℕ := 1000
def install_hours : ℕ := 6
def install_cost_per_hour : ℕ := 100
def equipment_cost : ℕ := 6000
def total_expenses : ℕ := 147700
def chicken_cost_per_chicken : ℕ := 5

def total_cost_before_chickens : ℕ := 
  (land_acres * land_cost_per_acre) + 
  house_cost + 
  (num_cows * cow_cost_per_cow) + 
  (install_hours * install_cost_per_hour) + 
  equipment_cost

def chickens_cost : ℕ := total_expenses - total_cost_before_chickens

def num_chickens : ℕ := chickens_cost / chicken_cost_per_chicken

theorem determine_num_chickens : num_chickens = 100 := by
  sorry

end NUMINAMATH_GPT_determine_num_chickens_l1676_167635


namespace NUMINAMATH_GPT_cookies_yesterday_l1676_167603

theorem cookies_yesterday (cookies_today : ℕ) (difference : ℕ)
  (h1 : cookies_today = 140)
  (h2 : difference = 30) :
  cookies_today - difference = 110 :=
by
  sorry

end NUMINAMATH_GPT_cookies_yesterday_l1676_167603


namespace NUMINAMATH_GPT_treasure_contains_645_coins_max_leftover_coins_when_choosing_93_pirates_l1676_167656

namespace PirateTreasure

-- Given conditions
def num_pirates_excl_captain := 100
def max_coins := 1000
def remaining_coins_99_pirates := 51
def remaining_coins_77_pirates := 29

-- Problem Part (a): Prove the number of coins in treasure
theorem treasure_contains_645_coins : 
  ∃ (N : ℕ), N < max_coins ∧ (N % 99 = remaining_coins_99_pirates ∧ N % 77 = remaining_coins_77_pirates) ∧ N = 645 :=
  sorry

-- Problem Part (b): Prove the number of pirates Barbaroxa should choose
theorem max_leftover_coins_when_choosing_93_pirates :
  ∃ (n : ℕ), n ≤ num_pirates_excl_captain ∧ (∀ k, k ≤ num_pirates_excl_captain → (645 % k) ≤ (645 % k) ∧ n = 93) :=
  sorry

end PirateTreasure

end NUMINAMATH_GPT_treasure_contains_645_coins_max_leftover_coins_when_choosing_93_pirates_l1676_167656


namespace NUMINAMATH_GPT_number_of_white_balls_l1676_167691

-- Definitions based on the problem conditions
def total_balls : Nat := 120
def red_freq : ℝ := 0.15
def black_freq : ℝ := 0.45

-- Result to prove
theorem number_of_white_balls :
  let red_balls := total_balls * red_freq
  let black_balls := total_balls * black_freq
  total_balls - red_balls - black_balls = 48 :=
by
  sorry

end NUMINAMATH_GPT_number_of_white_balls_l1676_167691


namespace NUMINAMATH_GPT_linda_savings_l1676_167652

theorem linda_savings :
  ∀ (S : ℝ), (5 / 6 * S + 500 = S) → S = 3000 :=
by
  intros S h
  sorry

end NUMINAMATH_GPT_linda_savings_l1676_167652


namespace NUMINAMATH_GPT_measure_angle_PSR_is_40_l1676_167613

noncomputable def isosceles_triangle (P Q R : Point) (PQ PR : ℝ) (hPQ_PR : PQ = PR) : Triangle := sorry
noncomputable def square (D R S T : Point) : Square := sorry
noncomputable def angle (A B C : Point) (θ : ℝ) : Prop := sorry

def angle_PQR (P Q R : Point) (PQ PR : ℝ) (hPQ_PR : PQ = PR) : ℝ := sorry
def angle_PRQ (P Q R : Point) (PQ PR : ℝ) (hPQ_PR : PQ = PR) : ℝ := sorry

theorem measure_angle_PSR_is_40
  (P Q R S T D : Point)
  (PQ PR : ℝ)
  (hPQ_PR : PQ = PR)
  (hQ_eq_D : Q = D)
  (hQPS : angle P Q S 100)
  (hDRST_square : square D R S T) : angle P S R 40 :=
by
  -- Proof omitted for brevity
  sorry

end NUMINAMATH_GPT_measure_angle_PSR_is_40_l1676_167613


namespace NUMINAMATH_GPT_smallest_integer_l1676_167609

-- Define a function to calculate the LCM of a list of numbers
def lcm_list (l : List ℕ) : ℕ :=
  l.foldl Nat.lcm 1

-- List of divisors
def divisors : List ℕ := [4, 5, 6, 7, 8, 9, 10]

-- Calculating the required integer
noncomputable def required_integer : ℕ := lcm_list divisors + 1

-- The proof statement
theorem smallest_integer : required_integer = 2521 :=
  by 
  sorry

end NUMINAMATH_GPT_smallest_integer_l1676_167609


namespace NUMINAMATH_GPT_water_height_in_cylinder_l1676_167696

theorem water_height_in_cylinder :
  let r_cone := 10 -- Radius of the cone in cm
  let h_cone := 15 -- Height of the cone in cm
  let r_cylinder := 20 -- Radius of the cylinder in cm
  let volume_cone := (1 / 3) * Real.pi * r_cone^2 * h_cone
  volume_cone = 500 * Real.pi -> 
  let h_cylinder := volume_cone / (Real.pi * r_cylinder^2)
  h_cylinder = 1.25 := 
by
  intros r_cone h_cone r_cylinder volume_cone h_volume
  let h_cylinder := volume_cone / (Real.pi * r_cylinder^2)
  have : h_cylinder = 1.25 := by
    sorry
  exact this

end NUMINAMATH_GPT_water_height_in_cylinder_l1676_167696


namespace NUMINAMATH_GPT_min_value_2a_minus_ab_l1676_167637

theorem min_value_2a_minus_ab (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (ha_lt_11 : a < 11) (hb_lt_11 : b < 11) : 
  ∃ (min_val : ℤ), min_val = -80 ∧ ∀ x y : ℕ, 0 < x → 0 < y → x < 11 → y < 11 → 2 * x - x * y ≥ min_val :=
by
  use -80
  sorry

end NUMINAMATH_GPT_min_value_2a_minus_ab_l1676_167637


namespace NUMINAMATH_GPT_miniVanTankCapacity_is_65_l1676_167639

noncomputable def miniVanTankCapacity : ℝ :=
  let serviceCostPerVehicle := 2.10
  let fuelCostPerLiter := 0.60
  let numMiniVans := 3
  let numTrucks := 2
  let totalCost := 299.1
  let truckFactor := 1.2
  let V := (totalCost - serviceCostPerVehicle * (numMiniVans + numTrucks)) /
            (fuelCostPerLiter * (numMiniVans + numTrucks * (1 + truckFactor)))
  V

theorem miniVanTankCapacity_is_65 : miniVanTankCapacity = 65 :=
  sorry

end NUMINAMATH_GPT_miniVanTankCapacity_is_65_l1676_167639


namespace NUMINAMATH_GPT_inequality_neg_mul_l1676_167641

theorem inequality_neg_mul (a b : ℝ) (h : a > b) : -3 * a < -3 * b :=
sorry

end NUMINAMATH_GPT_inequality_neg_mul_l1676_167641


namespace NUMINAMATH_GPT_distance_to_origin_l1676_167601

theorem distance_to_origin (a : ℝ) (h: |a| = 5) : 3 - a = -2 ∨ 3 - a = 8 :=
sorry

end NUMINAMATH_GPT_distance_to_origin_l1676_167601


namespace NUMINAMATH_GPT_trigonometric_identity_l1676_167664

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 1) : 
  1 - 2 * Real.sin α * Real.cos α - 3 * (Real.cos α)^2 = -3 / 2 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1676_167664


namespace NUMINAMATH_GPT_total_amount_l1676_167645

variable (A B C : ℕ)
variable (h1 : C = 495)
variable (h2 : (A - 10) * 18 = (B - 20) * 11)
variable (h3 : (B - 20) * 24 = (C - 15) * 18)

theorem total_amount (A B C : ℕ) (h1 : C = 495)
  (h2 : (A - 10) * 18 = (B - 20) * 11)
  (h3 : (B - 20) * 24 = (C - 15) * 18) :
  A + B + C = 1105 :=
sorry

end NUMINAMATH_GPT_total_amount_l1676_167645


namespace NUMINAMATH_GPT_find_x_plus_inv_x_l1676_167642

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_GPT_find_x_plus_inv_x_l1676_167642


namespace NUMINAMATH_GPT_find_b_l1676_167628

variable (a b c : ℕ)

def conditions (a b c : ℕ) : Prop :=
  a = b + 2 ∧ 
  b = 2 * c ∧ 
  a + b + c = 42

theorem find_b (a b c : ℕ) (h : conditions a b c) : b = 16 := 
sorry

end NUMINAMATH_GPT_find_b_l1676_167628


namespace NUMINAMATH_GPT_inequality_solution_set_l1676_167625

theorem inequality_solution_set (x : ℝ) : (2 * x + 1 ≥ 3) ∧ (4 * x - 1 < 7) ↔ (1 ≤ x ∧ x < 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1676_167625


namespace NUMINAMATH_GPT_perfect_square_conditions_l1676_167685

theorem perfect_square_conditions (k : ℕ) : 
  (∃ m : ℤ, k^2 - 101 * k = m^2) ↔ (k = 101 ∨ k = 2601) := 
by 
  sorry

end NUMINAMATH_GPT_perfect_square_conditions_l1676_167685


namespace NUMINAMATH_GPT_equation_of_the_line_l1676_167674

theorem equation_of_the_line (a b : ℝ) :
    ((a - b = 5) ∧ (9 / a + 4 / b = 1)) → 
    ( (2 * 9 + 3 * 4 - 30 = 0) ∨ (2 * 9 - 3 * 4 - 6 = 0) ∨ (9 - 4 - 5 = 0)) :=
  by
    sorry

end NUMINAMATH_GPT_equation_of_the_line_l1676_167674


namespace NUMINAMATH_GPT_correct_calculation_l1676_167686

theorem correct_calculation :
  (∀ a : ℝ, (a^2)^3 = a^6) ∧
  ¬(∀ a : ℝ, a * a^3 = a^3) ∧
  ¬(∀ a : ℝ, a + 2 * a^2 = 3 * a^3) ∧
  ¬(∀ (a b : ℝ), (-2 * a^2 * b)^2 = -4 * a^4 * b^2) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1676_167686


namespace NUMINAMATH_GPT_smallest_b_satisfying_inequality_l1676_167619

theorem smallest_b_satisfying_inequality : ∀ b : ℝ, (b^2 - 16 * b + 55) ≥ 0 ↔ b ≤ 5 ∨ b ≥ 11 := sorry

end NUMINAMATH_GPT_smallest_b_satisfying_inequality_l1676_167619


namespace NUMINAMATH_GPT_find_number_l1676_167644

theorem find_number (number : ℝ) (h : 0.003 * number = 0.15) : number = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1676_167644


namespace NUMINAMATH_GPT_difference_SP_l1676_167616

-- Definitions for amounts
variables (P Q R S : ℕ)

-- Conditions given in the problem
def total_amount := P + Q + R + S = 1000
def P_condition := P = 2 * Q
def S_condition := S = 4 * R
def Q_R_equal := Q = R

-- Statement of the problem that needs to be proven
theorem difference_SP (P Q R S : ℕ) (h1 : total_amount P Q R S) 
  (h2 : P_condition P Q) (h3 : S_condition S R) (h4 : Q_R_equal Q R) : 
  S - P = 250 :=
by 
  sorry

end NUMINAMATH_GPT_difference_SP_l1676_167616


namespace NUMINAMATH_GPT_Bill_order_combinations_l1676_167676

def donut_combinations (num_donuts num_kinds : ℕ) : ℕ :=
  Nat.choose (num_donuts + num_kinds - 1) (num_kinds - 1)

theorem Bill_order_combinations : donut_combinations 10 5 = 126 :=
by
  -- This would be the place to insert the proof steps, but we're using sorry as the placeholder.
  sorry

end NUMINAMATH_GPT_Bill_order_combinations_l1676_167676


namespace NUMINAMATH_GPT_smallest_of_three_numbers_l1676_167610

theorem smallest_of_three_numbers : ∀ (a b c : ℕ), (a = 5) → (b = 8) → (c = 4) → min (min a b) c = 4 :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  sorry

end NUMINAMATH_GPT_smallest_of_three_numbers_l1676_167610


namespace NUMINAMATH_GPT_find_f_neg_2017_l1676_167622

-- Define f as given in the problem
def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 2

-- State the given problem condition
def condition (a b : ℝ) : Prop :=
  f a b 2017 = 10

-- The main problem statement proving the solution
theorem find_f_neg_2017 (a b : ℝ) (h : condition a b) : f a b (-2017) = -14 :=
by
  -- We state this theorem and provide a sorry to skip the proof
  sorry

end NUMINAMATH_GPT_find_f_neg_2017_l1676_167622


namespace NUMINAMATH_GPT_find_x_l1676_167670

variable (x y : ℝ)

theorem find_x (h1 : 0 < x) (h2 : 0 < y) (h3 : 5 * x^2 + 10 * x * y = x^3 + 2 * x^2 * y) : x = 5 := by
  sorry

end NUMINAMATH_GPT_find_x_l1676_167670


namespace NUMINAMATH_GPT_max_value_of_d_l1676_167668

-- Define the conditions
variable (a b c d : ℝ) (h_sum : a + b + c + d = 10) 
          (h_prod_sum : ab + ac + ad + bc + bd + cd = 20)

-- Define the theorem statement
theorem max_value_of_d : 
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_d_l1676_167668


namespace NUMINAMATH_GPT_pseudo_code_output_l1676_167606

theorem pseudo_code_output (a b c : Int)
  (h1 : a = 3)
  (h2 : b = -5)
  (h3 : c = 8)
  (ha : a = -5)
  (hb : b = 8)
  (hc : c = -5) : 
  a = -5 ∧ b = 8 ∧ c = -5 :=
by
  sorry

end NUMINAMATH_GPT_pseudo_code_output_l1676_167606


namespace NUMINAMATH_GPT_probability_of_log_ge_than_1_l1676_167615

noncomputable def probability_log_greater_than_one : ℝ := sorry

theorem probability_of_log_ge_than_1 :
  probability_log_greater_than_one = 1 / 2 :=
sorry

end NUMINAMATH_GPT_probability_of_log_ge_than_1_l1676_167615


namespace NUMINAMATH_GPT_function_relationship_l1676_167620

theorem function_relationship (f : ℝ → ℝ)
  (h₁ : ∀ x, f (x + 1) = f (-x + 1))
  (h₂ : ∀ x, x ≥ 1 → f x = (1 / 2) ^ x - 1) :
  f (2 / 3) > f (3 / 2) ∧ f (3 / 2) > f (1 / 3) :=
by sorry

end NUMINAMATH_GPT_function_relationship_l1676_167620


namespace NUMINAMATH_GPT_find_x_plus_y_l1676_167608

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 3000)
  (h2 : x + 3000 * Real.sin y = 2999) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2999 := by
  sorry

end NUMINAMATH_GPT_find_x_plus_y_l1676_167608


namespace NUMINAMATH_GPT_tulip_count_l1676_167679

theorem tulip_count (total_flowers : ℕ) (daisies : ℕ) (roses_ratio : ℚ)
  (tulip_count : ℕ) :
  total_flowers = 102 →
  daisies = 6 →
  roses_ratio = 5 / 6 →
  tulip_count = (total_flowers - daisies) * (1 - roses_ratio) →
  tulip_count = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_tulip_count_l1676_167679


namespace NUMINAMATH_GPT_daleyza_contracted_units_l1676_167660

variable (units_building1 : ℕ)
variable (units_building2 : ℕ)
variable (units_building3 : ℕ)

def total_units (units_building1 units_building2 units_building3 : ℕ) : ℕ :=
  units_building1 + units_building2 + units_building3

theorem daleyza_contracted_units :
  units_building1 = 4000 →
  units_building2 = 2 * units_building1 / 5 →
  units_building3 = 120 * units_building2 / 100 →
  total_units units_building1 units_building2 units_building3 = 7520 :=
by
  intros h1 h2 h3
  unfold total_units
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_daleyza_contracted_units_l1676_167660


namespace NUMINAMATH_GPT_factor_expression_l1676_167661

theorem factor_expression (x : ℝ) : 16 * x ^ 2 + 8 * x = 8 * x * (2 * x + 1) :=
by
  -- Problem: Completely factor the expression
  -- Given Condition
  -- Conclusion
  sorry

end NUMINAMATH_GPT_factor_expression_l1676_167661


namespace NUMINAMATH_GPT_hands_per_hoopit_l1676_167649

-- Defining conditions
def num_hoopits := 7
def num_neglarts := 8
def total_toes := 164
def toes_per_hand_hoopit := 3
def toes_per_hand_neglart := 2
def hands_per_neglart := 5

-- The statement to prove
theorem hands_per_hoopit : 
  ∃ (H : ℕ), (H * toes_per_hand_hoopit * num_hoopits + hands_per_neglart * toes_per_hand_neglart * num_neglarts = total_toes) → H = 4 :=
sorry

end NUMINAMATH_GPT_hands_per_hoopit_l1676_167649


namespace NUMINAMATH_GPT_max_principals_in_10_years_l1676_167678

theorem max_principals_in_10_years :
  ∀ (term_length : ℕ) (P : ℕ → Prop),
  (∀ n, P n → 3 ≤ n ∧ n ≤ 5) → 
  ∃ (n : ℕ), (n ≤ 10 / 3 ∧ P n) ∧ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_max_principals_in_10_years_l1676_167678


namespace NUMINAMATH_GPT_average_rainfall_feb_1983_l1676_167655

theorem average_rainfall_feb_1983 (total_rainfall : ℕ) (days_in_february : ℕ) (hours_per_day : ℕ) 
  (H1 : total_rainfall = 789) (H2 : days_in_february = 28) (H3 : hours_per_day = 24) : 
  total_rainfall / (days_in_february * hours_per_day) = 789 / 672 :=
by
  sorry

end NUMINAMATH_GPT_average_rainfall_feb_1983_l1676_167655


namespace NUMINAMATH_GPT_find_inequality_solution_set_l1676_167657

noncomputable def inequality_solution_set : Set ℝ :=
  { x | (1 / (x * (x + 1))) - (1 / ((x + 1) * (x + 2))) < (1 / 4) }

theorem find_inequality_solution_set :
  inequality_solution_set = { x : ℝ | x < -2 } ∪ { x : ℝ | -1 < x ∧ x < 0 } ∪ { x : ℝ | 1 < x } :=
by
  sorry

end NUMINAMATH_GPT_find_inequality_solution_set_l1676_167657


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l1676_167611

variable {a b c e : ℝ}
variable (h_hyperbola : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
variable (ha_pos : a > 0)
variable (hb_pos : b > 0)
variable (h_vertices : A1 = (-a, 0) ∧ A2 = (a, 0))
variable (h_imaginary_axis : B1 = (0, b) ∧ B2 = (0, -b))
variable (h_foci : F1 = (-c, 0) ∧ F2 = (c, 0))
variable (h_relation : a^2 + b^2 = c^2)
variable (h_tangent_circle : ∀ d, (d = 2*a) → (tangent (circle d) (rhombus F1 B1 F2 B2)))

theorem eccentricity_of_hyperbola : e = (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l1676_167611


namespace NUMINAMATH_GPT_max_sum_of_squares_70_l1676_167602

theorem max_sum_of_squares_70 :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  a^2 + b^2 + c^2 + d^2 = 70 ∧ a + b + c + d = 16 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_of_squares_70_l1676_167602


namespace NUMINAMATH_GPT_minimum_apples_to_guarantee_18_one_color_l1676_167600

theorem minimum_apples_to_guarantee_18_one_color :
  let red := 32
  let green := 24
  let yellow := 22
  let blue := 15
  let orange := 14
  ∀ n, (n >= 81) →
  (∃ red_picked green_picked yellow_picked blue_picked orange_picked : ℕ,
    red_picked + green_picked + yellow_picked + blue_picked + orange_picked = n
    ∧ red_picked ≤ red ∧ green_picked ≤ green ∧ yellow_picked ≤ yellow ∧ blue_picked ≤ blue ∧ orange_picked ≤ orange
    ∧ (red_picked = 18 ∨ green_picked = 18 ∨ yellow_picked = 18 ∨ blue_picked = 18 ∨ orange_picked = 18)) :=
by {
  -- The proof is omitted for now.
  sorry
}

end NUMINAMATH_GPT_minimum_apples_to_guarantee_18_one_color_l1676_167600


namespace NUMINAMATH_GPT_find_divisor_l1676_167672

theorem find_divisor (d : ℕ) (h1 : 2319 % d = 0) (h2 : 2304 % d = 0) (h3 : (2319 - 2304) % d = 0) : d = 3 :=
  sorry

end NUMINAMATH_GPT_find_divisor_l1676_167672


namespace NUMINAMATH_GPT_total_handshakes_l1676_167633

theorem total_handshakes (players_team1 players_team2 referees : ℕ) 
  (h1 : players_team1 = 11) (h2 : players_team2 = 11) (h3 : referees = 3) : 
  players_team1 * players_team2 + (players_team1 + players_team2) * referees = 187 := 
by
  sorry

end NUMINAMATH_GPT_total_handshakes_l1676_167633


namespace NUMINAMATH_GPT_cos_difference_l1676_167690

theorem cos_difference (α β : ℝ) (h_α_acute : 0 < α ∧ α < π / 2)
                      (h_β_acute : 0 < β ∧ β < π / 2)
                      (h_cos_α : Real.cos α = 1 / 3)
                      (h_cos_sum : Real.cos (α + β) = -1 / 3) :
  Real.cos (α - β) = 23 / 27 := 
sorry

end NUMINAMATH_GPT_cos_difference_l1676_167690


namespace NUMINAMATH_GPT_correct_factorization_l1676_167627

-- Definitions for the given conditions of different options
def condition_A (a : ℝ) : Prop := 2 * a^2 - 2 * a + 1 = 2 * a * (a - 1) + 1
def condition_B (x y : ℝ) : Prop := (x + y) * (x - y) = x^2 - y^2
def condition_C (x y : ℝ) : Prop := x^2 - 4 * x * y + 4 * y^2 = (x - 2 * y)^2
def condition_D (x : ℝ) : Prop := x^2 + 1 = x * (x + 1 / x)

-- The theorem to prove that option C is correct
theorem correct_factorization (x y : ℝ) : condition_C x y :=
by sorry

end NUMINAMATH_GPT_correct_factorization_l1676_167627


namespace NUMINAMATH_GPT_mrs_sheridan_initial_cats_l1676_167695

def cats_initial (cats_given_away : ℕ) (cats_left : ℕ) : ℕ :=
  cats_given_away + cats_left

theorem mrs_sheridan_initial_cats : cats_initial 14 3 = 17 :=
by
  sorry

end NUMINAMATH_GPT_mrs_sheridan_initial_cats_l1676_167695


namespace NUMINAMATH_GPT_rectangular_C₁_general_C₂_intersection_and_sum_l1676_167638

-- Definition of curve C₁ in polar coordinates
def C₁_polar (ρ θ : ℝ) : Prop := ρ * Real.cos θ ^ 2 = Real.sin θ

-- Definition of curve C₂ in parametric form
def C₂_param (k x y : ℝ) : Prop := 
  x = 8 * k / (1 + k^2) ∧ y = 2 * (1 - k^2) / (1 + k^2)

-- Rectangular coordinate equation of curve C₁ is x² = y
theorem rectangular_C₁ (ρ θ : ℝ) (x y : ℝ) (h₁ : ρ * Real.cos θ ^ 2 = Real.sin θ)
  (h₂ : x = ρ * Real.cos θ) (h₃ : y = ρ * Real.sin θ) : x^2 = y :=
sorry

-- General equation of curve C₂ is x² / 16 + y² / 4 = 1 with y ≠ -2
theorem general_C₂ (k x y : ℝ) (h₁ : x = 8 * k / (1 + k^2))
  (h₂ : y = 2 * (1 - k^2) / (1 + k^2)) : x^2 / 16 + y^2 / 4 = 1 ∧ y ≠ -2 :=
sorry

-- Given point M and parametric line l, prove the value of sum reciprocals of distances to points of intersection with curve C₁ is √7
theorem intersection_and_sum (t m₁ m₂ x y : ℝ) 
  (M : ℝ × ℝ) (hM : M = (0, 1/2))
  (hline : x = Real.sqrt 3 * t ∧ y = 1/2 + t)
  (hintersect1 : 3 * m₁^2 - 2 * m₁ - 2 = 0)
  (hintersect2 : 3 * m₂^2 - 2 * m₂ - 2 = 0)
  (hroot1_2 : m₁ + m₂ = 2/3 ∧ m₁ * m₂ = -2/3) : 
  1 / abs (M.fst - x) + 1 / abs (M.snd - y) = Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_rectangular_C₁_general_C₂_intersection_and_sum_l1676_167638


namespace NUMINAMATH_GPT_divide_plane_into_regions_l1676_167607

theorem divide_plane_into_regions (n : ℕ) (h₁ : n < 199) (h₂ : ∃ (k : ℕ), k = 99):
  n = 100 ∨ n = 198 :=
sorry

end NUMINAMATH_GPT_divide_plane_into_regions_l1676_167607


namespace NUMINAMATH_GPT_arithmetic_geom_seq_a5_l1676_167636

theorem arithmetic_geom_seq_a5 (a : ℕ → ℝ) (s : ℕ → ℝ) (q : ℝ)
  (a1 : a 1 = 1)
  (S8 : s 8 = 17 * s 4) :
  a 5 = 16 :=
sorry

end NUMINAMATH_GPT_arithmetic_geom_seq_a5_l1676_167636


namespace NUMINAMATH_GPT_camping_trip_percentage_l1676_167693

theorem camping_trip_percentage (T : ℝ)
  (h1 : 16 / 100 ≤ 1)
  (h2 : T - 16 / 100 ≤ 1)
  (h3 : T = 64 / 100) :
  T = 64 / 100 := by
  sorry

end NUMINAMATH_GPT_camping_trip_percentage_l1676_167693


namespace NUMINAMATH_GPT_gcd_1821_2993_l1676_167605

theorem gcd_1821_2993 : Nat.gcd 1821 2993 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_1821_2993_l1676_167605


namespace NUMINAMATH_GPT_geometric_sequence_sum_5_l1676_167629

theorem geometric_sequence_sum_5 
  (a : ℕ → ℝ) 
  (h_geom : ∃ q, ∀ n, a (n + 1) = a n * q) 
  (h_a2 : a 2 = 2) 
  (h_a3 : a 3 = 4) : 
  (a 1 * (1 - (2:ℝ)^5) / (1 - (2:ℝ))) = 31 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_5_l1676_167629


namespace NUMINAMATH_GPT_largest_whole_number_l1676_167650

theorem largest_whole_number (x : ℕ) (h : 6 * x + 3 < 150) : x ≤ 24 :=
sorry

end NUMINAMATH_GPT_largest_whole_number_l1676_167650


namespace NUMINAMATH_GPT_coloring_ways_l1676_167662

-- Define the vertices and edges of the graph
def vertices : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8}

def edges : Finset (ℕ × ℕ) :=
  { (0, 1), (1, 2), (2, 0),  -- First triangle
    (3, 4), (4, 5), (5, 3),  -- Middle triangle
    (6, 7), (7, 8), (8, 6),  -- Third triangle
    (2, 5),   -- Connecting top horizontal edge
    (1, 7) }  -- Connecting bottom horizontal edge

-- Define the number of colors available
def colors := 4

-- Define a function to count the valid colorings given the vertices and edges
noncomputable def countValidColorings (vertices : Finset ℕ) (edges : Finset (ℕ × ℕ)) (colors : ℕ) : ℕ := sorry

-- The theorem statement
theorem coloring_ways : countValidColorings vertices edges colors = 3456 := 
sorry

end NUMINAMATH_GPT_coloring_ways_l1676_167662


namespace NUMINAMATH_GPT_cost_of_each_new_shirt_l1676_167697

theorem cost_of_each_new_shirt (pants_cost shorts_cost shirts_cost : ℕ)
  (pants_sold shorts_sold shirts_sold : ℕ) (money_left : ℕ) (new_shirts : ℕ)
  (h₁ : pants_cost = 5) (h₂ : shorts_cost = 3) (h₃ : shirts_cost = 4)
  (h₄ : pants_sold = 3) (h₅ : shorts_sold = 5) (h₆ : shirts_sold = 5)
  (h₇ : money_left = 30) (h₈ : new_shirts = 2) :
  (pants_cost * pants_sold + shorts_cost * shorts_sold + shirts_cost * shirts_sold - money_left) / new_shirts = 10 :=
by sorry

end NUMINAMATH_GPT_cost_of_each_new_shirt_l1676_167697


namespace NUMINAMATH_GPT_least_k_square_divisible_by_240_l1676_167604

theorem least_k_square_divisible_by_240 (k : ℕ) (h : ∃ m : ℕ, k ^ 2 = 240 * m) : k ≥ 60 :=
by
  sorry

end NUMINAMATH_GPT_least_k_square_divisible_by_240_l1676_167604


namespace NUMINAMATH_GPT_shortest_distance_between_circles_l1676_167623

theorem shortest_distance_between_circles :
  let c1 := (1, -3)
  let r1 := 2 * Real.sqrt 2
  let c2 := (-3, 1)
  let r2 := 1
  let distance_centers := Real.sqrt ((1 - -3)^2 + (-3 - 1)^2)
  let shortest_distance := distance_centers - (r1 + r2)
  shortest_distance = 2 * Real.sqrt 2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_shortest_distance_between_circles_l1676_167623


namespace NUMINAMATH_GPT_g_sum_eq_neg_one_l1676_167687

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Main theorem to prove g(1) + g(-1) = -1 given the conditions
theorem g_sum_eq_neg_one
  (h1 : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y)
  (h2 : f (-2) = f 1)
  (h3 : f 1 ≠ 0) :
  g 1 + g (-1) = -1 :=
sorry

end NUMINAMATH_GPT_g_sum_eq_neg_one_l1676_167687


namespace NUMINAMATH_GPT_milk_leftover_l1676_167624

theorem milk_leftover 
  (total_milk : ℕ := 24)
  (kids_percent : ℝ := 0.80)
  (cooking_percent : ℝ := 0.60)
  (neighbor_percent : ℝ := 0.25)
  (husband_percent : ℝ := 0.06) :
  let milk_after_kids := total_milk * (1 - kids_percent)
  let milk_after_cooking := milk_after_kids * (1 - cooking_percent)
  let milk_after_neighbor := milk_after_cooking * (1 - neighbor_percent)
  let milk_after_husband := milk_after_neighbor * (1 - husband_percent)
  milk_after_husband = 1.3536 :=
by 
  -- skip the proof for simplicity
  sorry

end NUMINAMATH_GPT_milk_leftover_l1676_167624


namespace NUMINAMATH_GPT_triangle_inequality_l1676_167630

theorem triangle_inequality (a b c : ℝ) (h : a + b > c ∧ a + c > b ∧ b + c > a) : 
  a * b * c ≥ (-a + b + c) * (a - b + c) * (a + b - c) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l1676_167630


namespace NUMINAMATH_GPT_quadratic_eq_roots_quadratic_eq_positive_integer_roots_l1676_167698

theorem quadratic_eq_roots (m : ℝ) (hm : m ≠ 0 ∧ m ≤ 9 / 8) :
  ∃ x1 x2 : ℝ, (m * x1^2 + 3 * x1 + 2 = 0) ∧ (m * x2^2 + 3 * x2 + 2 = 0) :=
sorry

theorem quadratic_eq_positive_integer_roots (m : ℕ) (hm : m = 1) :
  ∃ x1 x2 : ℝ, (x1 = -1) ∧ (x2 = -2) ∧ (m * x1^2 + 3 * x1 + 2 = 0) ∧ (m * x2^2 + 3 * x2 + 2 = 0) :=
sorry

end NUMINAMATH_GPT_quadratic_eq_roots_quadratic_eq_positive_integer_roots_l1676_167698


namespace NUMINAMATH_GPT_total_snakes_seen_l1676_167673

-- Define the number of snakes in each breeding ball
def snakes_in_first_breeding_ball : Nat := 15
def snakes_in_second_breeding_ball : Nat := 20
def snakes_in_third_breeding_ball : Nat := 25
def snakes_in_fourth_breeding_ball : Nat := 30
def snakes_in_fifth_breeding_ball : Nat := 35
def snakes_in_sixth_breeding_ball : Nat := 40
def snakes_in_seventh_breeding_ball : Nat := 45

-- Define the number of pairs of extra snakes
def extra_pairs_of_snakes : Nat := 23

-- Define the total number of snakes observed
def total_snakes_observed : Nat :=
  snakes_in_first_breeding_ball +
  snakes_in_second_breeding_ball +
  snakes_in_third_breeding_ball +
  snakes_in_fourth_breeding_ball +
  snakes_in_fifth_breeding_ball +
  snakes_in_sixth_breeding_ball +
  snakes_in_seventh_breeding_ball +
  (extra_pairs_of_snakes * 2)

theorem total_snakes_seen : total_snakes_observed = 256 := by
  sorry

end NUMINAMATH_GPT_total_snakes_seen_l1676_167673


namespace NUMINAMATH_GPT_product_mod_5_l1676_167688

theorem product_mod_5 : (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_product_mod_5_l1676_167688


namespace NUMINAMATH_GPT_card_game_probability_l1676_167646

theorem card_game_probability :
  let A_wins := 4;  -- number of heads needed for A to win all cards
  let B_wins := 4;  -- number of tails needed for B to win all cards
  let total_flips := 5;  -- exactly 5 flips
  (Nat.choose total_flips 1 + Nat.choose total_flips 1) / (2^total_flips) = 5 / 16 :=
by
  sorry

end NUMINAMATH_GPT_card_game_probability_l1676_167646


namespace NUMINAMATH_GPT_remaining_slices_after_weekend_l1676_167684

theorem remaining_slices_after_weekend 
  (initial_pies : ℕ) (slices_per_pie : ℕ) (rebecca_initial_slices : ℕ) 
  (family_fraction : ℚ) (sunday_evening_slices : ℕ) : 
  initial_pies = 2 → 
  slices_per_pie = 8 → 
  rebecca_initial_slices = 2 → 
  family_fraction = 0.5 → 
  sunday_evening_slices = 2 → 
  (initial_pies * slices_per_pie 
   - rebecca_initial_slices 
   - family_fraction * (initial_pies * slices_per_pie - rebecca_initial_slices) 
   - sunday_evening_slices) = 5 :=
by 
  intros initial_pies_eq slices_per_pie_eq rebecca_initial_slices_eq family_fraction_eq sunday_evening_slices_eq
  sorry

end NUMINAMATH_GPT_remaining_slices_after_weekend_l1676_167684


namespace NUMINAMATH_GPT_monthly_income_l1676_167669

variable {I : ℝ} -- George's monthly income

def donated_to_charity (I : ℝ) := 0.60 * I -- 60% of the income left
def paid_in_taxes (I : ℝ) := 0.75 * donated_to_charity I -- 75% of the remaining income after donation
def saved_for_future (I : ℝ) := 0.80 * paid_in_taxes I -- 80% of the remaining income after taxes
def expenses (I : ℝ) := saved_for_future I - 125 -- Remaining income after groceries and transportation expenses
def remaining_for_entertainment := 150 -- $150 left for entertainment and miscellaneous expenses

theorem monthly_income : I = 763.89 := 
by
  -- Using the conditions of the problem
  sorry

end NUMINAMATH_GPT_monthly_income_l1676_167669


namespace NUMINAMATH_GPT_solve_recurrence_relation_l1676_167694

noncomputable def a_n (n : ℕ) : ℝ := 2 * 4^n - 2 * n + 2
noncomputable def b_n (n : ℕ) : ℝ := 2 * 4^n + 2 * n - 2

theorem solve_recurrence_relation :
  a_n 0 = 4 ∧ b_n 0 = 0 ∧
  (∀ n : ℕ, a_n (n + 1) = 3 * a_n n + b_n n - 4) ∧
  (∀ n : ℕ, b_n (n + 1) = 2 * a_n n + 2 * b_n n + 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_recurrence_relation_l1676_167694


namespace NUMINAMATH_GPT_douglas_votes_in_Y_is_46_l1676_167643

variable (V : ℝ)
variable (P : ℝ)

def percentage_won_in_Y :=
  let total_voters_X := 2 * V
  let total_voters_Y := V
  let votes_in_X := 0.64 * total_voters_X
  let votes_in_Y := P / 100 * total_voters_Y
  let total_votes := 1.28 * V + (P / 100 * V)
  let combined_voters := 3 * V
  let combined_votes_percentage := 0.58 * combined_voters
  P = 46

theorem douglas_votes_in_Y_is_46
  (V_pos : V > 0)
  (H : 1.28 * V + (P / 100 * V) = 0.58 * 3 * V) :
  percentage_won_in_Y V P := by
  sorry

end NUMINAMATH_GPT_douglas_votes_in_Y_is_46_l1676_167643
