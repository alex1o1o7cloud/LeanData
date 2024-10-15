import Mathlib

namespace NUMINAMATH_GPT_tan_diff_eq_rat_l1636_163668

theorem tan_diff_eq_rat (A : ℝ × ℝ) (B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (5, 1))
  (α β : ℝ)
  (hα : Real.tan α = 2) (hβ : Real.tan β = 1 / 5) :
  Real.tan (α - β) = 9 / 7 := by
  sorry

end NUMINAMATH_GPT_tan_diff_eq_rat_l1636_163668


namespace NUMINAMATH_GPT_engineering_students_pass_percentage_l1636_163644

theorem engineering_students_pass_percentage :
  let num_male_students := 120
  let num_female_students := 100
  let perc_male_eng_students := 0.25
  let perc_female_eng_students := 0.20
  let perc_male_eng_pass := 0.20
  let perc_female_eng_pass := 0.25
  
  let num_male_eng_students := num_male_students * perc_male_eng_students
  let num_female_eng_students := num_female_students * perc_female_eng_students
  
  let num_male_eng_pass := num_male_eng_students * perc_male_eng_pass
  let num_female_eng_pass := num_female_eng_students * perc_female_eng_pass
  
  let total_eng_students := num_male_eng_students + num_female_eng_students
  let total_eng_pass := num_male_eng_pass + num_female_eng_pass
  
  (total_eng_pass / total_eng_students) * 100 = 22 :=
by
  sorry

end NUMINAMATH_GPT_engineering_students_pass_percentage_l1636_163644


namespace NUMINAMATH_GPT_loss_calculation_l1636_163614

-- Given conditions: 
-- The ratio of the amount of money Cara, Janet, and Jerry have is 4:5:6
-- The total amount of money they have is $75

theorem loss_calculation :
  let cara_ratio := 4
  let janet_ratio := 5
  let jerry_ratio := 6
  let total_ratio := cara_ratio + janet_ratio + jerry_ratio
  let total_money := 75
  let part_value := total_money / total_ratio
  let cara_money := cara_ratio * part_value
  let janet_money := janet_ratio * part_value
  let combined_money := cara_money + janet_money
  let selling_price := 0.80 * combined_money
  combined_money - selling_price = 9 :=
by
  sorry

end NUMINAMATH_GPT_loss_calculation_l1636_163614


namespace NUMINAMATH_GPT_some_seniors_not_club_members_l1636_163660

variables {People : Type} (Senior ClubMember : People → Prop) (Punctual : People → Prop)

-- Conditions:
def some_seniors_not_punctual := ∃ x, Senior x ∧ ¬Punctual x
def all_club_members_punctual := ∀ x, ClubMember x → Punctual x

-- Theorem statement to be proven:
theorem some_seniors_not_club_members (h1 : some_seniors_not_punctual Senior Punctual) (h2 : all_club_members_punctual ClubMember Punctual) : 
  ∃ x, Senior x ∧ ¬ ClubMember x :=
sorry

end NUMINAMATH_GPT_some_seniors_not_club_members_l1636_163660


namespace NUMINAMATH_GPT_boat_speed_l1636_163601

theorem boat_speed (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 7) : b = 9 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_l1636_163601


namespace NUMINAMATH_GPT_sheets_per_day_l1636_163651

-- Definitions based on conditions
def total_sheets : ℕ := 60
def total_days_per_week : ℕ := 7
def days_off : ℕ := 2

-- Derived condition from the problem
def work_days_per_week : ℕ := total_days_per_week - days_off

-- The statement to prove
theorem sheets_per_day : total_sheets / work_days_per_week = 12 :=
by
  sorry

end NUMINAMATH_GPT_sheets_per_day_l1636_163651


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1636_163607

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 2 → x^2 + 2 * x - 8 > 0) ∧ (¬(x > 2) → ¬(x^2 + 2 * x - 8 > 0)) → false :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1636_163607


namespace NUMINAMATH_GPT_john_paid_percentage_l1636_163628

theorem john_paid_percentage (SRP WP : ℝ) (h1 : SRP = 1.40 * WP) (h2 : ∀ P, P = (1 / 3) * SRP) : ((1 / 3) * SRP / SRP * 100) = 33.33 :=
by
  sorry

end NUMINAMATH_GPT_john_paid_percentage_l1636_163628


namespace NUMINAMATH_GPT_number_divided_is_144_l1636_163642

theorem number_divided_is_144 (n divisor quotient remainder : ℕ) (h_divisor : divisor = 11) (h_quotient : quotient = 13) (h_remainder : remainder = 1) (h_division : n = (divisor * quotient) + remainder) : n = 144 :=
by
  sorry

end NUMINAMATH_GPT_number_divided_is_144_l1636_163642


namespace NUMINAMATH_GPT_reciprocal_neg_2023_l1636_163666

theorem reciprocal_neg_2023 : (1 / (-2023: ℤ)) = - (1 / 2023) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_reciprocal_neg_2023_l1636_163666


namespace NUMINAMATH_GPT_pile_limit_exists_l1636_163617

noncomputable def log_floor (b x : ℝ) : ℤ :=
  Int.floor (Real.log x / Real.log b)

theorem pile_limit_exists (k : ℝ) (hk : k < 2) : ∃ Nk : ℤ, 
  Nk = 2 * (log_floor (2 / k) 2 + 1) := 
  by
    sorry

end NUMINAMATH_GPT_pile_limit_exists_l1636_163617


namespace NUMINAMATH_GPT_nature_of_roots_l1636_163609

noncomputable def P (x : ℝ) : ℝ := x^6 - 5 * x^5 - 7 * x^3 - 2 * x + 9

theorem nature_of_roots : 
  (∀ x < 0, P x > 0) ∧ ∃ x > 0, P 0 * P x < 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_nature_of_roots_l1636_163609


namespace NUMINAMATH_GPT_ruby_initial_apples_l1636_163699

theorem ruby_initial_apples (apples_taken : ℕ) (apples_left : ℕ) (initial_apples : ℕ) 
  (h1 : apples_taken = 55) (h2 : apples_left = 8) (h3 : initial_apples = apples_taken + apples_left) : 
  initial_apples = 63 := 
by
  sorry

end NUMINAMATH_GPT_ruby_initial_apples_l1636_163699


namespace NUMINAMATH_GPT_max_covered_squares_by_tetromino_l1636_163692

-- Definition of the grid size
def grid_size := (5, 5)

-- Definition of S-Tetromino (Z-Tetromino) coverage covering four contiguous squares
def is_STetromino (coords: List (Nat × Nat)) : Prop := 
  coords.length = 4 ∧ ∃ (x y : Nat), coords = [(x, y), (x, y+1), (x+1, y+1), (x+1, y+2)]

-- Definition of the coverage constraint
def no_more_than_two_tiles (cover: List (Nat × Nat)) : Prop :=
  ∀ (coord: Nat × Nat), cover.count coord ≤ 2

-- Definition of the total tiled squares covered by at least one tile
def tiles_covered (cover: List (Nat × Nat)) : Nat := 
  cover.toFinset.card 

-- Definition of the problem using proof equivalence
theorem max_covered_squares_by_tetromino
  (cover: List (List (Nat × Nat)))
  (H_tiles: ∀ t, t ∈ cover → is_STetromino t)
  (H_coverage: no_more_than_two_tiles (cover.join)) :
  tiles_covered (cover.join) = 24 :=
sorry 

end NUMINAMATH_GPT_max_covered_squares_by_tetromino_l1636_163692


namespace NUMINAMATH_GPT_fraction_milk_in_mug1_is_one_fourth_l1636_163680

-- Condition definitions
def initial_tea_mug1 := 6 -- ounces
def initial_milk_mug2 := 6 -- ounces
def tea_transferred_mug1_to_mug2 := initial_tea_mug1 / 3
def tea_remaining_mug1 := initial_tea_mug1 - tea_transferred_mug1_to_mug2
def total_liquid_mug2 := initial_milk_mug2 + tea_transferred_mug1_to_mug2
def portion_transferred_back := total_liquid_mug2 / 4
def tea_ratio_mug2 := tea_transferred_mug1_to_mug2 / total_liquid_mug2
def milk_ratio_mug2 := initial_milk_mug2 / total_liquid_mug2
def tea_transferred_back := portion_transferred_back * tea_ratio_mug2
def milk_transferred_back := portion_transferred_back * milk_ratio_mug2
def final_tea_mug1 := tea_remaining_mug1 + tea_transferred_back
def final_milk_mug1 := milk_transferred_back
def final_total_liquid_mug1 := final_tea_mug1 + final_milk_mug1

-- Lean statement of the problem
theorem fraction_milk_in_mug1_is_one_fourth :
  final_milk_mug1 / final_total_liquid_mug1 = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_milk_in_mug1_is_one_fourth_l1636_163680


namespace NUMINAMATH_GPT_last_popsicle_melts_32_times_faster_l1636_163685

theorem last_popsicle_melts_32_times_faster (t : ℕ) : 
  let time_first := t
  let time_sixth := t / 2^5
  (time_first / time_sixth) = 32 :=
by
  sorry

end NUMINAMATH_GPT_last_popsicle_melts_32_times_faster_l1636_163685


namespace NUMINAMATH_GPT_find_k_if_lines_parallel_l1636_163675

theorem find_k_if_lines_parallel (k : ℝ) : (∀ x y, y = 5 * x + 3 → y = (3 * k) * x + 7) → k = 5 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_k_if_lines_parallel_l1636_163675


namespace NUMINAMATH_GPT_length_of_PQ_is_8_l1636_163622

-- Define the lengths of the sides and conditions
variables (PQ QR PS SR : ℕ) (perimeter : ℕ)

-- State the conditions
def conditions : Prop :=
  SR = 16 ∧
  perimeter = 40 ∧
  PQ = QR ∧ QR = PS

-- State the goal
theorem length_of_PQ_is_8 (h : conditions PQ QR PS SR perimeter) : PQ = 8 :=
sorry

end NUMINAMATH_GPT_length_of_PQ_is_8_l1636_163622


namespace NUMINAMATH_GPT_expression_equals_base10_l1636_163677

-- Define numbers in various bases
def base7ToDec (n : ℕ) : ℕ := 1 * (7^2) + 6 * (7^1) + 5 * (7^0)
def base2ToDec (n : ℕ) : ℕ := 1 * (2^1) + 1 * (2^0)
def base6ToDec (n : ℕ) : ℕ := 1 * (6^2) + 2 * (6^1) + 1 * (6^0)
def base3ToDec (n : ℕ) : ℕ := 2 * (3^1) + 1 * (3^0)

-- Prove the given expression equals 39 in base 10
theorem expression_equals_base10 :
  (base7ToDec 165 / base2ToDec 11) + (base6ToDec 121 / base3ToDec 21) = 39 :=
by
  -- Convert the base n numbers to base 10
  let num1 := base7ToDec 165
  let den1 := base2ToDec 11
  let num2 := base6ToDec 121
  let den2 := base3ToDec 21
  
  -- Simplify the expression (skipping actual steps for brevity, replaced by sorry)
  sorry

end NUMINAMATH_GPT_expression_equals_base10_l1636_163677


namespace NUMINAMATH_GPT_jackson_points_l1636_163696

theorem jackson_points (team_total_points : ℕ) (other_players_count : ℕ) (other_players_avg_score : ℕ) 
  (total_points_by_team : team_total_points = 72) 
  (total_points_by_others : other_players_count = 7) 
  (avg_points_by_others : other_players_avg_score = 6) :
  ∃ points_by_jackson : ℕ, points_by_jackson = 30 :=
by
  sorry

end NUMINAMATH_GPT_jackson_points_l1636_163696


namespace NUMINAMATH_GPT_angle_P_measure_l1636_163649

theorem angle_P_measure (P Q R S : ℝ) 
  (h1 : P = 3 * Q)
  (h2 : P = 4 * R)
  (h3 : P = 6 * S)
  (h_sum : P + Q + R + S = 360) : 
  P = 206 :=
by 
  sorry

end NUMINAMATH_GPT_angle_P_measure_l1636_163649


namespace NUMINAMATH_GPT_gcd_qr_l1636_163689

theorem gcd_qr (p q r : ℕ) (hpq : Nat.gcd p q = 210) (hpr : Nat.gcd p r = 770) : Nat.gcd q r = 70 := sorry

end NUMINAMATH_GPT_gcd_qr_l1636_163689


namespace NUMINAMATH_GPT_age_problem_l1636_163695

theorem age_problem (x y : ℕ) (h1 : y - 5 = 2 * (x - 5)) (h2 : x + y + 16 = 50) : x = 13 :=
by sorry

end NUMINAMATH_GPT_age_problem_l1636_163695


namespace NUMINAMATH_GPT_conor_work_times_per_week_l1636_163661

-- Definitions for the conditions
def vegetables_per_day (eggplants carrots potatoes : ℕ) : ℕ :=
  eggplants + carrots + potatoes

def total_vegetables_per_week (days vegetables_per_day : ℕ) : ℕ :=
  days * vegetables_per_day

-- Theorem statement to be proven
theorem conor_work_times_per_week :
  let eggplants := 12
  let carrots := 9
  let potatoes := 8
  let weekly_total := 116
  vegetables_per_day eggplants carrots potatoes = 29 →
  total_vegetables_per_week 4 29 = 116 →
  4 = weekly_total / 29 :=
by
  intros _ _ h1 h2
  sorry

end NUMINAMATH_GPT_conor_work_times_per_week_l1636_163661


namespace NUMINAMATH_GPT_solve_equation_l1636_163650

theorem solve_equation : ∀ x : ℝ, 4 * x - 2 * x + 1 - 3 = 0 → x = 1 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_equation_l1636_163650


namespace NUMINAMATH_GPT_CD_eq_CE_l1636_163630

theorem CD_eq_CE {Point : Type*} [MetricSpace Point]
  (A B C D E : Point) (m : Set Point)
  (hAm : A ∈ m) (hBm : B ∈ m) (hCm : C ∈ m)
  (hDm : D ∉ m) (hEm : E ∉ m) 
  (hAD_AE : dist A D = dist A E)
  (hBD_BE : dist B D = dist B E) :
  dist C D = dist C E :=
sorry

end NUMINAMATH_GPT_CD_eq_CE_l1636_163630


namespace NUMINAMATH_GPT_final_grey_cats_l1636_163654

def initially_total_cats : Nat := 16
def initial_white_cats : Nat := 2
def percent_black_cats : Nat := 25
def black_cats_left_fraction : Nat := 2
def new_white_cats : Nat := 2
def new_grey_cats : Nat := 1

/- We will calculate the number of grey cats after all specified events -/
theorem final_grey_cats :
  let total_cats := initially_total_cats
  let white_cats := initial_white_cats + new_white_cats
  let black_cats := (percent_black_cats * total_cats / 100) / black_cats_left_fraction
  let initial_grey_cats := total_cats - white_cats - black_cats
  let final_grey_cats := initial_grey_cats + new_grey_cats
  final_grey_cats = 11 := by
  sorry

end NUMINAMATH_GPT_final_grey_cats_l1636_163654


namespace NUMINAMATH_GPT_farmer_sowed_correct_amount_l1636_163655

def initial_buckets : ℝ := 8.75
def final_buckets : ℝ := 6
def buckets_sowed : ℝ := initial_buckets - final_buckets

theorem farmer_sowed_correct_amount : buckets_sowed = 2.75 :=
by {
  sorry
}

end NUMINAMATH_GPT_farmer_sowed_correct_amount_l1636_163655


namespace NUMINAMATH_GPT_cricketer_runs_l1636_163679

theorem cricketer_runs (R x : ℝ) : 
  (R / 85 = 12.4) →
  ((R + x) / 90 = 12.0) →
  x = 26 := 
by
  sorry

end NUMINAMATH_GPT_cricketer_runs_l1636_163679


namespace NUMINAMATH_GPT_frequency_interval_20_to_inf_l1636_163612

theorem frequency_interval_20_to_inf (sample_size : ℕ)
  (freq_5_10 : ℕ) (freq_10_15 : ℕ) (freq_15_20 : ℕ)
  (freq_20_25 : ℕ) (freq_25_30 : ℕ) (freq_30_35 : ℕ) :
  sample_size = 35 ∧
  freq_5_10 = 5 ∧
  freq_10_15 = 12 ∧
  freq_15_20 = 7 ∧
  freq_20_25 = 5 ∧
  freq_25_30 = 4 ∧
  freq_30_35 = 2 →
  (1 - (freq_5_10 + freq_10_15 + freq_15_20 : ℕ) / (sample_size : ℕ) : ℝ) = 11 / 35 :=
by sorry

end NUMINAMATH_GPT_frequency_interval_20_to_inf_l1636_163612


namespace NUMINAMATH_GPT_numberOfKidsInOtherClass_l1636_163600

-- Defining the conditions as given in the problem
def kidsInSwansonClass := 25
def averageZitsSwansonClass := 5
def averageZitsOtherClass := 6
def additionalZitsInOtherClass := 67

-- Total number of zits in Ms. Swanson's class
def totalZitsSwansonClass := kidsInSwansonClass * averageZitsSwansonClass

-- Total number of zits in the other class
def totalZitsOtherClass := totalZitsSwansonClass + additionalZitsInOtherClass

-- Proof that the number of kids in the other class is 32
theorem numberOfKidsInOtherClass : 
  (totalZitsOtherClass / averageZitsOtherClass = 32) :=
by
  -- Proof is left as an exercise.
  sorry

end NUMINAMATH_GPT_numberOfKidsInOtherClass_l1636_163600


namespace NUMINAMATH_GPT_same_terminal_side_angle_l1636_163676

theorem same_terminal_side_angle (k : ℤ) : 
  ∃ (θ : ℤ), θ = k * 360 + 257 ∧ (θ % 360 = (-463) % 360) :=
by
  sorry

end NUMINAMATH_GPT_same_terminal_side_angle_l1636_163676


namespace NUMINAMATH_GPT_modified_monotonous_count_l1636_163674

def is_modified_monotonous (n : ℕ) : Prop :=
  -- Definition that determines if a number is modified-monotonous
  -- Must include digit '5', and digits must form a strictly increasing or decreasing sequence
  sorry 

def count_modified_monotonous (n : ℕ) : ℕ :=
  2 * (8 * (2^8) + 2^8) + 1 -- Formula for counting modified-monotonous numbers including '5'

theorem modified_monotonous_count : count_modified_monotonous 5 = 4609 := 
  by 
    sorry

end NUMINAMATH_GPT_modified_monotonous_count_l1636_163674


namespace NUMINAMATH_GPT_min_sum_ab_l1636_163627

theorem min_sum_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (2 / b) = 2) :
  a + b ≥ (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_GPT_min_sum_ab_l1636_163627


namespace NUMINAMATH_GPT_vertex_angle_measure_l1636_163608

-- Define the isosceles triangle and its properties
def is_isosceles_triangle (A B C : ℝ) (a b c : ℝ) :=
  (A = B ∨ B = C ∨ C = A) ∧ (a + b + c = 180)

-- Define the conditions based on the problem statement
def two_angles_sum_to_100 (x y : ℝ) := x + y = 100

-- The measure of the vertex angle
theorem vertex_angle_measure (A B C : ℝ) (a b c : ℝ) 
  (h1 : is_isosceles_triangle A B C a b c) (h2 : two_angles_sum_to_100 A B) :
  C = 20 ∨ C = 80 :=
sorry

end NUMINAMATH_GPT_vertex_angle_measure_l1636_163608


namespace NUMINAMATH_GPT_extreme_value_l1636_163618

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * (x - 1)

theorem extreme_value (a : ℝ) (h : a > 0) :
  ∃ x : ℝ, f x a = a - Real.log a - 1 ∧ (∀ y : ℝ, f y a ≤ f x a) :=
sorry

end NUMINAMATH_GPT_extreme_value_l1636_163618


namespace NUMINAMATH_GPT_equal_triangle_area_l1636_163638

theorem equal_triangle_area
  (ABC_area : ℝ)
  (AP PB : ℝ)
  (AB_area : ℝ)
  (PQ_BQ_equal : Prop)
  (AP_ratio: AP / (AP + PB) = 3 / 5)
  (ABC_area_val : ABC_area = 15)
  (AP_val : AP = 3)
  (PB_val : PB = 2)
  (PQ_BQ_equal : PQ_BQ_equal = true) :
  ∃ area, area = 9 ∧ area = 9 :=
by
  sorry

end NUMINAMATH_GPT_equal_triangle_area_l1636_163638


namespace NUMINAMATH_GPT_rectangle_side_lengths_l1636_163686

variables (x y m n S : ℝ) (hx_y_ratio : x / y = m / n) (hxy_area : x * y = S)

theorem rectangle_side_lengths :
  x = Real.sqrt (m * S / n) ∧ y = Real.sqrt (n * S / m) :=
sorry

end NUMINAMATH_GPT_rectangle_side_lengths_l1636_163686


namespace NUMINAMATH_GPT_range_of_m_l1636_163615

theorem range_of_m (x m : ℝ) (h1: |x - m| < 1) (h2: x^2 - 8 * x + 12 < 0) (h3: ∀ x, (x^2 - 8 * x + 12 < 0) → ((m - 1) < x ∧ x < (m + 1))) : 
  3 ≤ m ∧ m ≤ 5 := 
sorry

end NUMINAMATH_GPT_range_of_m_l1636_163615


namespace NUMINAMATH_GPT_quadrilateral_area_is_6_l1636_163639

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 2⟩
def B : Point := ⟨1, 1⟩
def C : Point := ⟨3, 1⟩
def D : Point := ⟨5, 5⟩

def area_triangle (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y))

def quadrilateral_area (A B C D : Point) : ℝ :=
  area_triangle A B C + area_triangle A C D

theorem quadrilateral_area_is_6 : quadrilateral_area A B C D = 6 :=
  sorry

end NUMINAMATH_GPT_quadrilateral_area_is_6_l1636_163639


namespace NUMINAMATH_GPT_ratio_of_perimeters_l1636_163613

theorem ratio_of_perimeters (d : ℝ) (s1 s2 P1 P2 : ℝ) (h1 : d^2 = 2 * s1^2)
  (h2 : (3 * d)^2 = 2 * s2^2) (h3 : P1 = 4 * s1) (h4 : P2 = 4 * s2) :
  P2 / P1 = 3 := 
by sorry

end NUMINAMATH_GPT_ratio_of_perimeters_l1636_163613


namespace NUMINAMATH_GPT_solve_system_l1636_163684

theorem solve_system : ∃ x y : ℚ, 
  (2 * x + 3 * y = 7 - 2 * x + 7 - 3 * y) ∧ 
  (3 * x - 2 * y = x - 2 + y - 2) ∧ 
  x = 3 / 4 ∧ 
  y = 11 / 6 := 
by 
  sorry

end NUMINAMATH_GPT_solve_system_l1636_163684


namespace NUMINAMATH_GPT_Adam_picks_apples_days_l1636_163623

theorem Adam_picks_apples_days (total_apples remaining_apples daily_pick : ℕ) 
  (h1 : total_apples = 350) 
  (h2 : remaining_apples = 230) 
  (h3 : daily_pick = 4) : 
  (total_apples - remaining_apples) / daily_pick = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_Adam_picks_apples_days_l1636_163623


namespace NUMINAMATH_GPT_angle_in_first_quadrant_l1636_163667

theorem angle_in_first_quadrant (α : ℝ) (h : 90 < α ∧ α < 180) : 0 < 180 - α ∧ 180 - α < 90 :=
by
  sorry

end NUMINAMATH_GPT_angle_in_first_quadrant_l1636_163667


namespace NUMINAMATH_GPT_tables_chairs_legs_l1636_163643

theorem tables_chairs_legs (t : ℕ) (c : ℕ) (total_legs : ℕ) 
  (h1 : c = 8 * t) 
  (h2 : total_legs = 4 * c + 6 * t) 
  (h3 : total_legs = 798) : 
  t = 21 :=
by
  sorry

end NUMINAMATH_GPT_tables_chairs_legs_l1636_163643


namespace NUMINAMATH_GPT_num_students_in_class_l1636_163634

-- Define the conditions
variables (S : ℕ) (num_boys : ℕ) (num_boys_under_6ft : ℕ)

-- Assume the conditions given in the problem
axiom two_thirds_boys : num_boys = (2 * S) / 3
axiom three_fourths_under_6ft : num_boys_under_6ft = (3 * num_boys) / 4
axiom nineteen_boys_under_6ft : num_boys_under_6ft = 19

-- The statement we want to prove
theorem num_students_in_class : S = 38 :=
by
  -- Proof omitted (insert proof here)
  sorry

end NUMINAMATH_GPT_num_students_in_class_l1636_163634


namespace NUMINAMATH_GPT_george_max_pencils_l1636_163646

-- Define the conditions for the problem
def total_money : ℝ := 9.30
def pencil_cost : ℝ := 1.05
def discount_rate : ℝ := 0.10

-- Define the final statement to prove
theorem george_max_pencils (n : ℕ) :
  (n ≤ 8 ∧ pencil_cost * n ≤ total_money) ∨ 
  (n > 8 ∧ pencil_cost * (1 - discount_rate) * n ≤ total_money) →
  n ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_george_max_pencils_l1636_163646


namespace NUMINAMATH_GPT_Mia_biking_speed_l1636_163683

theorem Mia_biking_speed
    (Eugene_speed : ℝ)
    (Carlos_ratio : ℝ)
    (Mia_ratio : ℝ)
    (Mia_speed : ℝ)
    (h1 : Eugene_speed = 5)
    (h2 : Carlos_ratio = 3 / 4)
    (h3 : Mia_ratio = 4 / 3)
    (h4 : Mia_speed = Mia_ratio * (Carlos_ratio * Eugene_speed)) :
    Mia_speed = 5 :=
by
  sorry

end NUMINAMATH_GPT_Mia_biking_speed_l1636_163683


namespace NUMINAMATH_GPT_problem_l1636_163693

def p : Prop := 0 % 2 = 0
def q : Prop := ¬(3 % 2 = 0)

theorem problem : p ∨ q :=
by
  sorry

end NUMINAMATH_GPT_problem_l1636_163693


namespace NUMINAMATH_GPT_two_a_minus_five_d_eq_zero_l1636_163624

variables {α : Type*} [Field α]

def f (a b c d x : α) : α :=
  (2*a*x + 3*b) / (4*c*x - 5*d)

theorem two_a_minus_five_d_eq_zero
  (a b c d : α) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (hf : ∀ x, f a b c d (f a b c d x) = x) :
  2*a - 5*d = 0 :=
sorry

end NUMINAMATH_GPT_two_a_minus_five_d_eq_zero_l1636_163624


namespace NUMINAMATH_GPT_fourth_arithmetic_sequence_equation_l1636_163691

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables (a : ℕ → ℝ) (h : is_arithmetic_sequence a)
variable (h1 : a 1 - 2 * a 2 + a 3 = 0)
variable (h2 : a 1 - 3 * a 2 + 3 * a 3 - a 4 = 0)
variable (h3 : a 1 - 4 * a 2 + 6 * a 3 - 4 * a 4 + a 5 = 0)

-- Theorem statement to be proven
theorem fourth_arithmetic_sequence_equation : a 1 - 5 * a 2 + 10 * a 3 - 10 * a 4 + 5 * a 5 - a 6 = 0 :=
by
  sorry

end NUMINAMATH_GPT_fourth_arithmetic_sequence_equation_l1636_163691


namespace NUMINAMATH_GPT_abs_conditions_iff_l1636_163672

theorem abs_conditions_iff (x y : ℝ) :
  (|x| < 1 ∧ |y| < 1) ↔ (|x + y| + |x - y| < 2) :=
by
  sorry

end NUMINAMATH_GPT_abs_conditions_iff_l1636_163672


namespace NUMINAMATH_GPT_second_group_students_l1636_163605

theorem second_group_students 
  (total_students : ℕ) 
  (first_group_students : ℕ) 
  (h1 : total_students = 71) 
  (h2 : first_group_students = 34) : 
  total_students - first_group_students = 37 :=
by 
  sorry

end NUMINAMATH_GPT_second_group_students_l1636_163605


namespace NUMINAMATH_GPT_least_positive_integer_divisibility_l1636_163645

theorem least_positive_integer_divisibility :
  ∃ n > 1, (∀ k ∈ [2, 3, 4, 5, 6, 7, 8, 9], n % k = 1) ∧ n = 2521 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_divisibility_l1636_163645


namespace NUMINAMATH_GPT_addition_subtraction_questions_l1636_163625

theorem addition_subtraction_questions (total_questions word_problems answered_questions add_sub_questions : ℕ)
  (h1 : total_questions = 45)
  (h2 : word_problems = 17)
  (h3 : answered_questions = total_questions - 7)
  (h4 : add_sub_questions = answered_questions - word_problems) : 
  add_sub_questions = 21 := 
by 
  -- the proof steps are skipped
  sorry

end NUMINAMATH_GPT_addition_subtraction_questions_l1636_163625


namespace NUMINAMATH_GPT_leftmost_rectangle_is_B_l1636_163681

def isLeftmostRectangle (wA wB wC wD wE : ℕ) : Prop := 
  wB < wD ∧ wB < wE

theorem leftmost_rectangle_is_B :
  let wA := 5
  let wB := 2
  let wC := 4
  let wD := 9
  let wE := 10
  let xA := 2
  let xB := 1
  let xC := 7
  let xD := 6
  let xE := 4
  let yA := 8
  let yB := 6
  let yC := 3
  let yD := 5
  let yE := 7
  let zA := 10
  let zB := 9
  let zC := 0
  let zD := 11
  let zE := 2
  isLeftmostRectangle wA wB wC wD wE :=
by
  simp only
  sorry

end NUMINAMATH_GPT_leftmost_rectangle_is_B_l1636_163681


namespace NUMINAMATH_GPT_avg_of_6_10_N_is_10_if_even_l1636_163636

theorem avg_of_6_10_N_is_10_if_even (N : ℕ) (h1 : 9 ≤ N) (h2 : N ≤ 17) (h3 : (6 + 10 + N) % 2 = 0) : (6 + 10 + N) / 3 = 10 :=
by
-- sorry is placed here since we are not including the actual proof
sorry

end NUMINAMATH_GPT_avg_of_6_10_N_is_10_if_even_l1636_163636


namespace NUMINAMATH_GPT_lizas_final_balance_l1636_163626

-- Define the initial condition and subsequent changes
def initial_balance : ℕ := 800
def rent_payment : ℕ := 450
def paycheck_deposit : ℕ := 1500
def electricity_bill : ℕ := 117
def internet_bill : ℕ := 100
def phone_bill : ℕ := 70

-- Calculate the final balance
def final_balance : ℕ :=
  let balance_after_rent := initial_balance - rent_payment
  let balance_after_paycheck := balance_after_rent + paycheck_deposit
  let balance_after_bills := balance_after_paycheck - (electricity_bill + internet_bill)
  balance_after_bills - phone_bill

-- Theorem to prove that the final balance is 1563
theorem lizas_final_balance : final_balance = 1563 :=
by
  sorry

end NUMINAMATH_GPT_lizas_final_balance_l1636_163626


namespace NUMINAMATH_GPT_tan_alpha_tan_beta_l1636_163671

theorem tan_alpha_tan_beta (α β : ℝ) (h1 : Real.cos (α + β) = 3 / 5) (h2 : Real.cos (α - β) = 4 / 5) :
  Real.tan α * Real.tan β = 1 / 7 := by
  sorry

end NUMINAMATH_GPT_tan_alpha_tan_beta_l1636_163671


namespace NUMINAMATH_GPT_fixed_point_l1636_163673

noncomputable def function (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) (x : ℝ) : ℝ :=
  a ^ (x - 1) + 1

theorem fixed_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  function a h_pos h_ne_one 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_l1636_163673


namespace NUMINAMATH_GPT_kilometers_to_meters_kilograms_to_grams_l1636_163669

def km_to_meters (km: ℕ) : ℕ := km * 1000
def kg_to_grams (kg: ℕ) : ℕ := kg * 1000

theorem kilometers_to_meters (h: 3 = 3): km_to_meters 3 = 3000 := by {
 sorry
}

theorem kilograms_to_grams (h: 4 = 4): kg_to_grams 4 = 4000 := by {
 sorry
}

end NUMINAMATH_GPT_kilometers_to_meters_kilograms_to_grams_l1636_163669


namespace NUMINAMATH_GPT_obtain_2001_from_22_l1636_163656

theorem obtain_2001_from_22 :
  ∃ (f : ℕ → ℕ), (∀ n, f (n + 1) = n ∨ f (n) = n + 1) ∧ (f 22 = 2001) := 
sorry

end NUMINAMATH_GPT_obtain_2001_from_22_l1636_163656


namespace NUMINAMATH_GPT_petrol_expenses_l1636_163640

-- Definitions based on the conditions stated in the problem
def salary_saved (salary : ℝ) : ℝ := 0.10 * salary
def total_known_expenses : ℝ := 5000 + 1500 + 4500 + 2500 + 3940

-- Main theorem statement that needs to be proved
theorem petrol_expenses (salary : ℝ) (petrol : ℝ) :
  salary_saved salary = 2160 ∧ salary - 2160 = 19440 ∧ 
  5000 + 1500 + 4500 + 2500 + 3940 = total_known_expenses  →
  petrol = 2000 :=
sorry

end NUMINAMATH_GPT_petrol_expenses_l1636_163640


namespace NUMINAMATH_GPT_sequence_closed_form_l1636_163610

theorem sequence_closed_form (a : ℕ → ℤ) (h1 : a 1 = 1) (h_rec : ∀ n : ℕ, a (n + 1) = 2 * a n + 3) :
  ∀ n : ℕ, a n = 2^(n + 1) - 3 :=
by 
sorry

end NUMINAMATH_GPT_sequence_closed_form_l1636_163610


namespace NUMINAMATH_GPT_total_bugs_eaten_l1636_163653

-- Define the conditions
def gecko_eats : ℕ := 12
def lizard_eats : ℕ := gecko_eats / 2
def frog_eats : ℕ := lizard_eats * 3
def toad_eats : ℕ := frog_eats + (frog_eats / 2)

-- Define the proof
theorem total_bugs_eaten : gecko_eats + lizard_eats + frog_eats + toad_eats = 63 :=
by
  sorry

end NUMINAMATH_GPT_total_bugs_eaten_l1636_163653


namespace NUMINAMATH_GPT_hostel_cost_l1636_163621

def first_week_rate : ℝ := 18
def additional_week_rate : ℝ := 12
def first_week_days : ℕ := 7
def total_days : ℕ := 23

theorem hostel_cost :
  (first_week_days * first_week_rate + 
  (total_days - first_week_days) / first_week_days * first_week_days * additional_week_rate + 
  (total_days - first_week_days) % first_week_days * additional_week_rate) = 318 := 
by
  sorry

end NUMINAMATH_GPT_hostel_cost_l1636_163621


namespace NUMINAMATH_GPT_competition_end_time_l1636_163659

-- Definitions for the problem conditions
def start_time : ℕ := 15 * 60  -- 3:00 p.m. in minutes from midnight
def duration : ℕ := 1300       -- competition duration in minutes
def end_time : ℕ := start_time + duration

-- The expected end time in minutes from midnight, where 12:40 p.m. is (12*60 + 40) = 760 + 40 = 800 minutes from midnight.
def expected_end_time : ℕ := 12 * 60 + 40 

-- The theorem to prove
theorem competition_end_time : end_time = expected_end_time := by
  sorry

end NUMINAMATH_GPT_competition_end_time_l1636_163659


namespace NUMINAMATH_GPT_solve_for_y_l1636_163606

theorem solve_for_y (x y : ℝ) (h1 : 2 * x - y = 10) (h2 : x + 3 * y = 2) : y = -6 / 7 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1636_163606


namespace NUMINAMATH_GPT_T_30_is_13515_l1636_163697

def sequence_first_element (n : ℕ) : ℕ := 1 + n * (n - 1) / 2

def sequence_last_element (n : ℕ) : ℕ := sequence_first_element n + n - 1

def sum_sequence_set (n : ℕ) : ℕ :=
  n * (sequence_first_element n + sequence_last_element n) / 2

theorem T_30_is_13515 : sum_sequence_set 30 = 13515 := by
  sorry

end NUMINAMATH_GPT_T_30_is_13515_l1636_163697


namespace NUMINAMATH_GPT_total_volume_needed_l1636_163688

def box_length : ℕ := 20
def box_width : ℕ := 20
def box_height : ℕ := 12
def box_cost : ℕ := 50 -- in cents to avoid using floats
def total_spent : ℕ := 20000 -- $200 in cents

def volume_of_box : ℕ := box_length * box_width * box_height
def number_of_boxes : ℕ := total_spent / box_cost

theorem total_volume_needed : number_of_boxes * volume_of_box = 1920000 := by
  sorry

end NUMINAMATH_GPT_total_volume_needed_l1636_163688


namespace NUMINAMATH_GPT_mangoes_ratio_l1636_163602

theorem mangoes_ratio (a d_a : ℕ)
  (h1 : a = 60)
  (h2 : a + d_a = 75) : a / (75 - a) = 4 := by
  sorry

end NUMINAMATH_GPT_mangoes_ratio_l1636_163602


namespace NUMINAMATH_GPT_range_of_a_min_value_of_a_l1636_163698

variable (f : ℝ → ℝ) (a x : ℝ)

-- Part 1
theorem range_of_a (f_def : ∀ x, f x = abs (x - a)) 
  (h₁ : ∀ x, 1 ≤ x ∧ x ≤ 3 → f x ≤ 3) : 0 ≤ a ∧ a ≤ 4 :=
sorry

-- Part 2
theorem min_value_of_a (f_def : ∀ x, f x = abs (x - a)) 
  (h₂ : ∀ x, f (x - a) + f (x + a) ≥ 1 - a) : a ≥ 1/3 :=
sorry

end NUMINAMATH_GPT_range_of_a_min_value_of_a_l1636_163698


namespace NUMINAMATH_GPT_inequality_proof_l1636_163648

variable {a1 a2 a3 a4 a5 : ℝ}

theorem inequality_proof (h1 : 1 < a1) (h2 : 1 < a2) (h3 : 1 < a3) (h4 : 1 < a4) (h5 : 1 < a5) :
  16 * (a1 * a2 * a3 * a4 * a5 + 1) > (1 + a1) * (1 + a2) * (1 + a3) * (1 + a4) * (1 + a5) := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1636_163648


namespace NUMINAMATH_GPT_fill_40x41_table_l1636_163629

-- Define the condition on integers in the table
def valid_integer_filling (m n : ℕ) (table : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, i < m → j < n →
    table i j =
    ((if i > 0 then if table i j = table (i - 1) j then 1 else 0 else 0) +
    (if j > 0 then if table i j = table i (j - 1) then 1 else 0 else 0) +
    (if i < m - 1 then if table i j = table (i + 1) j then 1 else 0 else 0) +
    (if j < n - 1 then if table i j = table i (j + 1) then 1 else 0 else 0))

-- Define the specific problem for a 40 × 41 table.
theorem fill_40x41_table :
  ∃ (table : ℕ → ℕ → ℕ), valid_integer_filling 40 41 table :=
by
  sorry

end NUMINAMATH_GPT_fill_40x41_table_l1636_163629


namespace NUMINAMATH_GPT_stones_on_perimeter_of_square_l1636_163687

theorem stones_on_perimeter_of_square (n : ℕ) (h : n = 5) : 
  4 * n - 4 = 16 :=
by
  sorry

end NUMINAMATH_GPT_stones_on_perimeter_of_square_l1636_163687


namespace NUMINAMATH_GPT_inheritance_amount_l1636_163637

-- Define the conditions
def federal_tax_rate : ℝ := 0.2
def state_tax_rate : ℝ := 0.1
def total_taxes_paid : ℝ := 10500

-- Lean statement for the proof
theorem inheritance_amount (I : ℝ)
  (h1 : federal_tax_rate = 0.2)
  (h2 : state_tax_rate = 0.1)
  (h3 : total_taxes_paid = 10500)
  (taxes_eq : total_taxes_paid = (federal_tax_rate * I) + (state_tax_rate * (I - (federal_tax_rate * I))))
  : I = 37500 :=
sorry

end NUMINAMATH_GPT_inheritance_amount_l1636_163637


namespace NUMINAMATH_GPT_milk_jars_good_for_sale_l1636_163632

noncomputable def good_whole_milk_jars : ℕ := 
  let initial_jars := 60 * 30
  let short_deliveries := 20 * 30 * 2
  let damaged_jars_1 := 3 * 5
  let damaged_jars_2 := 4 * 6
  let totally_damaged_cartons := 2 * 30
  let received_jars := initial_jars - short_deliveries - damaged_jars_1 - damaged_jars_2 - totally_damaged_cartons
  let spoilage := (5 * received_jars) / 100
  received_jars - spoilage

noncomputable def good_skim_milk_jars : ℕ := 
  let initial_jars := 40 * 40
  let short_delivery := 10 * 40
  let damaged_jars := 5 * 4
  let totally_damaged_carton := 1 * 40
  let received_jars := initial_jars - short_delivery - damaged_jars - totally_damaged_carton
  let spoilage := (3 * received_jars) / 100
  received_jars - spoilage

noncomputable def good_almond_milk_jars : ℕ := 
  let initial_jars := 30 * 20
  let short_delivery := 5 * 20
  let damaged_jars := 2 * 3
  let received_jars := initial_jars - short_delivery - damaged_jars
  let spoilage := (1 * received_jars) / 100
  received_jars - spoilage

theorem milk_jars_good_for_sale : 
  good_whole_milk_jars = 476 ∧
  good_skim_milk_jars = 1106 ∧
  good_almond_milk_jars = 489 :=
by
  sorry

end NUMINAMATH_GPT_milk_jars_good_for_sale_l1636_163632


namespace NUMINAMATH_GPT_more_girls_than_boys_l1636_163662

theorem more_girls_than_boys (total students : ℕ) (girls boys : ℕ) (h1 : total = 41) (h2 : girls = 22) (h3 : girls + boys = total) : (girls - boys) = 3 :=
by
  sorry

end NUMINAMATH_GPT_more_girls_than_boys_l1636_163662


namespace NUMINAMATH_GPT_sum_of_all_two_digit_numbers_l1636_163620

theorem sum_of_all_two_digit_numbers : 
  let digits := [0, 1, 2, 3, 4, 5]
  let tens_digits := [1, 2, 3, 4, 5]
  let num_ones_digits := digits.length
  let num_tens_digits := tens_digits.length
  let sum_tens_place := 10 * (tens_digits.sum) * num_ones_digits
  let sum_ones_place := (digits.sum) * num_tens_digits
  sum_tens_place + sum_ones_place = 975 :=
by 
  let digits := [0, 1, 2, 3, 4, 5]
  let tens_digits := [1, 2, 3, 4, 5]
  let num_ones_digits := digits.length
  let num_tens_digits := tens_digits.length
  let sum_tens_place := 10 * (tens_digits.sum) * num_ones_digits
  let sum_ones_place := (digits.sum) * num_tens_digits
  show sum_tens_place + sum_ones_place = 975
  sorry

end NUMINAMATH_GPT_sum_of_all_two_digit_numbers_l1636_163620


namespace NUMINAMATH_GPT_stereos_production_fraction_l1636_163670

/-
Company S produces three kinds of stereos: basic, deluxe, and premium.
Of the stereos produced by Company S last month, 2/5 were basic, 3/10 were deluxe, and the rest were premium.
It takes 1.6 as many hours to produce a deluxe stereo as it does to produce a basic stereo, and 2.5 as many hours to produce a premium stereo as it does to produce a basic stereo.
Prove that the number of hours it took to produce the deluxe and premium stereos last month was 123/163 of the total number of hours it took to produce all the stereos.
-/

def stereos_production (total_stereos : ℕ) (basic_ratio deluxe_ratio : ℚ)
  (deluxe_time_multiplier premium_time_multiplier : ℚ) : ℚ :=
  let basic_stereos := total_stereos * basic_ratio
  let deluxe_stereos := total_stereos * deluxe_ratio
  let premium_stereos := total_stereos - basic_stereos - deluxe_stereos
  let basic_time := basic_stereos
  let deluxe_time := deluxe_stereos * deluxe_time_multiplier
  let premium_time := premium_stereos * premium_time_multiplier
  let total_time := basic_time + deluxe_time + premium_time
  (deluxe_time + premium_time) / total_time

-- Given values
def total_stereos : ℕ := 100
def basic_ratio : ℚ := 2 / 5
def deluxe_ratio : ℚ := 3 / 10
def deluxe_time_multiplier : ℚ := 1.6
def premium_time_multiplier : ℚ := 2.5

theorem stereos_production_fraction : stereos_production total_stereos basic_ratio deluxe_ratio deluxe_time_multiplier premium_time_multiplier = 123 / 163 := by
  sorry

end NUMINAMATH_GPT_stereos_production_fraction_l1636_163670


namespace NUMINAMATH_GPT_rosie_pie_count_l1636_163631

def total_apples : ℕ := 40
def initial_apples_required : ℕ := 3
def apples_per_pie : ℕ := 5

theorem rosie_pie_count : (total_apples - initial_apples_required) / apples_per_pie = 7 :=
by
  sorry

end NUMINAMATH_GPT_rosie_pie_count_l1636_163631


namespace NUMINAMATH_GPT_james_distance_ridden_l1636_163694

theorem james_distance_ridden : 
  let speed := 16 
  let time := 5 
  speed * time = 80 := 
by
  sorry

end NUMINAMATH_GPT_james_distance_ridden_l1636_163694


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l1636_163664

theorem sum_of_squares_of_roots :
  ∀ (x₁ x₂ : ℝ), (∀ a b c : ℝ, (a ≠ 0) →
  6 * x₁ ^ 2 + 5 * x₁ - 4 = 0 ∧ 6 * x₂ ^ 2 + 5 * x₂ - 4 = 0 →
  x₁ ^ 2 + x₂ ^ 2 = 73 / 36) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l1636_163664


namespace NUMINAMATH_GPT_number_of_sandwiches_l1636_163616

-- Definitions based on the conditions in the problem
def sandwich_cost : Nat := 3
def water_cost : Nat := 2
def total_cost : Nat := 11

-- Lean statement to prove the number of sandwiches bought is 3
theorem number_of_sandwiches (S : Nat) (h : sandwich_cost * S + water_cost = total_cost) : S = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sandwiches_l1636_163616


namespace NUMINAMATH_GPT_cost_of_fencing_l1636_163663

/-- The sides of a rectangular field are in the ratio 3:4.
If the area of the field is 10092 sq. m and the cost of fencing the field is 25 paise per meter,
then the cost of fencing the field is 101.5 rupees. --/
theorem cost_of_fencing (area : ℕ) (fencing_cost : ℝ) (ratio1 ratio2 perimeter : ℝ)
  (h_area : area = 10092)
  (h_ratio : ratio1 = 3 ∧ ratio2 = 4)
  (h_fencing_cost : fencing_cost = 0.25)
  (h_perimeter : perimeter = 406) :
  perimeter * fencing_cost = 101.5 := by
  sorry

end NUMINAMATH_GPT_cost_of_fencing_l1636_163663


namespace NUMINAMATH_GPT_weight_of_new_person_l1636_163633

/-- The average weight of 10 persons increases by 7.2 kg when a new person
replaces one who weighs 65 kg. Prove that the weight of the new person is 137 kg. -/
theorem weight_of_new_person (W_new : ℝ) (W_old : ℝ) (n : ℝ) (increase : ℝ) 
  (h1 : W_old = 65) (h2 : n = 10) (h3 : increase = 7.2) 
  (h4 : W_new = W_old + n * increase) : W_new = 137 := 
by
  -- proof to be done later
  sorry

end NUMINAMATH_GPT_weight_of_new_person_l1636_163633


namespace NUMINAMATH_GPT_f_at_2_l1636_163657

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  x ^ 2017 + a * x ^ 3 - b / x - 8

theorem f_at_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by sorry

end NUMINAMATH_GPT_f_at_2_l1636_163657


namespace NUMINAMATH_GPT_university_math_students_l1636_163678

theorem university_math_students
  (total_students : ℕ)
  (math_only : ℕ)
  (stats_only : ℕ)
  (both_courses : ℕ)
  (H1 : total_students = 75)
  (H2 : math_only + stats_only + both_courses = total_students)
  (H3 : math_only = 2 * (stats_only + both_courses))
  (H4 : both_courses = 9) :
  math_only + both_courses = 53 :=
by
  sorry

end NUMINAMATH_GPT_university_math_students_l1636_163678


namespace NUMINAMATH_GPT_bird_family_problem_l1636_163690

def initial_bird_families (f s i : Nat) : Prop :=
  i = f + s

theorem bird_family_problem : initial_bird_families 32 35 67 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_bird_family_problem_l1636_163690


namespace NUMINAMATH_GPT_problem1_range_of_f_problem2_range_of_m_l1636_163619

noncomputable def f (x : ℝ) : ℝ :=
  (Real.log x / Real.log 2 - 2) * (Real.log x / Real.log 4 - 1/2)

theorem problem1_range_of_f :
  Set.range (fun x => f x) ∩ Set.Icc 1 4 = Set.Icc (-1/8 : ℝ) 1 :=
sorry

theorem problem2_range_of_m :
  ∀ x, x ∈ Set.Icc 4 16 → f x > (m : ℝ) * (Real.log x / Real.log 4) ↔ m < 0 :=
sorry

end NUMINAMATH_GPT_problem1_range_of_f_problem2_range_of_m_l1636_163619


namespace NUMINAMATH_GPT_remaining_miles_l1636_163635

theorem remaining_miles (total_miles : ℕ) (driven_miles : ℕ) (h1: total_miles = 1200) (h2: driven_miles = 642) :
  total_miles - driven_miles = 558 :=
by
  sorry

end NUMINAMATH_GPT_remaining_miles_l1636_163635


namespace NUMINAMATH_GPT_stadium_revenue_difference_l1636_163658

theorem stadium_revenue_difference :
  let total_capacity := 2000
  let vip_capacity := 200
  let standard_capacity := 1000
  let general_capacity := 800
  let vip_price := 50
  let standard_price := 30
  let general_price := 20
  let three_quarters (n : ℕ) := (3 * n) / 4
  let three_quarter_full := three_quarters total_capacity
  let vip_three_quarter := three_quarters vip_capacity
  let standard_three_quarter := three_quarters standard_capacity
  let general_three_quarter := three_quarters general_capacity
  let revenue_three_quarter := vip_three_quarter * vip_price + standard_three_quarter * standard_price + general_three_quarter * general_price
  let revenue_full := vip_capacity * vip_price + standard_capacity * standard_price + general_capacity * general_price
  revenue_three_quarter = 42000 ∧ (revenue_full - revenue_three_quarter) = 14000 :=
by
  sorry

end NUMINAMATH_GPT_stadium_revenue_difference_l1636_163658


namespace NUMINAMATH_GPT_red_lucky_stars_l1636_163647

theorem red_lucky_stars (x : ℕ) : (20 + x + 15 > 0) → (x / (20 + x + 15) : ℚ) = 0.5 → x = 35 := by
  sorry

end NUMINAMATH_GPT_red_lucky_stars_l1636_163647


namespace NUMINAMATH_GPT_range_of_m_l1636_163611

-- Definitions according to the problem conditions
def p (x : ℝ) : Prop := (-2 ≤ x ∧ x ≤ 10)
def q (x : ℝ) (m : ℝ) : Prop := (1 - m ≤ x ∧ x ≤ 1 + m) ∧ m > 0

-- Rephrasing the problem statement in Lean
theorem range_of_m (x : ℝ) (m : ℝ) :
  (∀ x, p x → q x m) → m ≥ 9 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1636_163611


namespace NUMINAMATH_GPT_length_of_chord_l1636_163665

theorem length_of_chord (x y : ℝ) 
  (h1 : (x - 1)^2 + y^2 = 4) 
  (h2 : x + y + 1 = 0) 
  : ∃ (l : ℝ), l = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_length_of_chord_l1636_163665


namespace NUMINAMATH_GPT_speed_of_journey_l1636_163604

-- Define the conditions
def journey_time : ℕ := 10
def journey_distance : ℕ := 200
def half_journey_distance : ℕ := journey_distance / 2

-- Define the hypothesis that the journey is split into two equal parts, each traveled at the same speed
def equal_speed (v : ℕ) : Prop :=
  (half_journey_distance / v) + (half_journey_distance / v) = journey_time

-- Prove the speed v is 20 km/hr given the conditions
theorem speed_of_journey : ∃ v : ℕ, equal_speed v ∧ v = 20 :=
by
  have h : equal_speed 20 := sorry
  exact ⟨20, h, rfl⟩

end NUMINAMATH_GPT_speed_of_journey_l1636_163604


namespace NUMINAMATH_GPT_base_six_product_correct_l1636_163652

namespace BaseSixProduct

-- Definitions of the numbers in base six
def num1_base6 : ℕ := 1 * 6^2 + 3 * 6^1 + 2 * 6^0
def num2_base6 : ℕ := 1 * 6^1 + 4 * 6^0

-- Their product in base ten
def product_base10 : ℕ := num1_base6 * num2_base6

-- Convert the base ten product back to base six
def product_base6 : ℕ := 2 * 6^3 + 3 * 6^2 + 3 * 6^1 + 2 * 6^0

-- Theorem statement
theorem base_six_product_correct : product_base10 = 560 ∧ product_base6 = 2332 := by
  sorry

end BaseSixProduct

end NUMINAMATH_GPT_base_six_product_correct_l1636_163652


namespace NUMINAMATH_GPT_fraction_of_AD_eq_BC_l1636_163641

theorem fraction_of_AD_eq_BC (x y : ℝ) (B C D A : ℝ) 
  (h1 : B < C) 
  (h2 : C < D)
  (h3 : D < A) 
  (hBD : B < D)
  (hCD : C < D)
  (hAD : A = D)
  (hAB : A - B = 3 * (D - B)) 
  (hAC : A - C = 7 * (D - C))
  (hx_eq : x = 2 * y) 
  (hADx : A - D = 4 * x)
  (hADy : A - D = 8 * y)
  : (C - B) = 1/8 * (A - D) := 
sorry

end NUMINAMATH_GPT_fraction_of_AD_eq_BC_l1636_163641


namespace NUMINAMATH_GPT_determine_max_weight_l1636_163682

theorem determine_max_weight {a b : ℕ} (n : ℕ) (x : ℕ) (ha : a > 0) (hb : b > 0) (hx : 1 ≤ x ∧ x ≤ n) :
  n = 9 :=
sorry

end NUMINAMATH_GPT_determine_max_weight_l1636_163682


namespace NUMINAMATH_GPT_loss_percentage_l1636_163603

theorem loss_percentage (CP SP SP_new : ℝ) (L : ℝ) 
  (h1 : CP = 1428.57)
  (h2 : SP = CP - (L / 100 * CP))
  (h3 : SP_new = CP + 0.04 * CP)
  (h4 : SP_new = SP + 200) :
  L = 10 := by
    sorry

end NUMINAMATH_GPT_loss_percentage_l1636_163603
