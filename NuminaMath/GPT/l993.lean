import Mathlib

namespace NUMINAMATH_GPT_ben_and_sue_answer_l993_99355

theorem ben_and_sue_answer :
  let x := 8
  let y := 3 * (x + 2)
  let z := 3 * (y - 2)
  z = 84
:= by
  let x := 8
  let y := 3 * (x + 2)
  let z := 3 * (y - 2)
  show z = 84
  sorry

end NUMINAMATH_GPT_ben_and_sue_answer_l993_99355


namespace NUMINAMATH_GPT_bananas_left_correct_l993_99334

def initial_bananas : ℕ := 12
def eaten_bananas : ℕ := 1
def bananas_left (initial eaten : ℕ) := initial - eaten

theorem bananas_left_correct : bananas_left initial_bananas eaten_bananas = 11 :=
by
  sorry

end NUMINAMATH_GPT_bananas_left_correct_l993_99334


namespace NUMINAMATH_GPT_length_of_bridge_l993_99360

-- Define the problem conditions
def length_train : ℝ := 110 -- Length of the train in meters
def speed_kmph : ℝ := 60 -- Speed of the train in kmph

-- Convert speed from kmph to m/s
noncomputable def speed_mps : ℝ := speed_kmph * 1000 / 3600

-- Define the time taken to cross the bridge
def time_seconds : ℝ := 16.7986561075114

-- Define the total distance covered
noncomputable def total_distance : ℝ := speed_mps * time_seconds

-- Prove the length of the bridge
theorem length_of_bridge : total_distance - length_train = 170 := 
by
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_length_of_bridge_l993_99360


namespace NUMINAMATH_GPT_female_democrats_l993_99374

theorem female_democrats (F M : ℕ) 
    (h₁ : F + M = 990)
    (h₂ : F / 2 + M / 4 = 330) : F / 2 = 275 := 
by sorry

end NUMINAMATH_GPT_female_democrats_l993_99374


namespace NUMINAMATH_GPT_original_square_perimeter_l993_99345

theorem original_square_perimeter (P : ℝ) (x : ℝ) (h1 : 4 * x * 2 + 4 * x = 56) : P = 32 :=
by
  sorry

end NUMINAMATH_GPT_original_square_perimeter_l993_99345


namespace NUMINAMATH_GPT_cube_inequality_of_greater_l993_99309

theorem cube_inequality_of_greater {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end NUMINAMATH_GPT_cube_inequality_of_greater_l993_99309


namespace NUMINAMATH_GPT_michael_bought_crates_on_thursday_l993_99390

theorem michael_bought_crates_on_thursday :
  ∀ (eggs_per_crate crates_tuesday crates_given current_eggs bought_on_thursday : ℕ),
    crates_tuesday = 6 →
    crates_given = 2 →
    eggs_per_crate = 30 →
    current_eggs = 270 →
    bought_on_thursday = (current_eggs - (crates_tuesday * eggs_per_crate - crates_given * eggs_per_crate)) / eggs_per_crate →
    bought_on_thursday = 5 :=
by
  intros _ _ _ _ _
  sorry

end NUMINAMATH_GPT_michael_bought_crates_on_thursday_l993_99390


namespace NUMINAMATH_GPT_v_function_expression_f_max_value_l993_99365

noncomputable def v (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 4 then 2
else if 4 < x ∧ x ≤ 20 then - (1/8) * x + (5/2)
else 0

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 4 then 2 * x
else if 4 < x ∧ x ≤ 20 then - (1/8) * x^2 + (5/2) * x
else 0

theorem v_function_expression :
  ∀ x, 0 < x ∧ x ≤ 20 → 
  v x = (if 0 < x ∧ x ≤ 4 then 2 else if 4 < x ∧ x ≤ 20 then - (1/8) * x + (5/2) else 0) :=
by sorry

theorem f_max_value :
  ∃ x, 0 < x ∧ x ≤ 20 ∧ f x = 12.5 :=
by sorry

end NUMINAMATH_GPT_v_function_expression_f_max_value_l993_99365


namespace NUMINAMATH_GPT_sled_dog_race_l993_99317

theorem sled_dog_race (d t : ℕ) (h1 : d + t = 315) (h2 : (1.2 : ℚ) * d + t = (1 / 2 : ℚ) * (2 * d + 3 * t)) :
  d = 225 ∧ t = 90 :=
sorry

end NUMINAMATH_GPT_sled_dog_race_l993_99317


namespace NUMINAMATH_GPT_total_right_handed_players_l993_99321

theorem total_right_handed_players
  (total_players throwers mp_players non_throwers L R : ℕ)
  (ratio_L_R : 2 * R = 3 * L)
  (total_eq : total_players = 120)
  (throwers_eq : throwers = 60)
  (mp_eq : mp_players = 20)
  (non_throwers_eq : non_throwers = total_players - throwers - mp_players)
  (non_thrower_sum_eq : L + R = non_throwers) :
  (throwers + mp_players + R = 104) :=
by
  sorry

end NUMINAMATH_GPT_total_right_handed_players_l993_99321


namespace NUMINAMATH_GPT_snow_first_day_eq_six_l993_99308

variable (snow_first_day snow_second_day snow_fourth_day snow_fifth_day : ℤ)

theorem snow_first_day_eq_six
  (h1 : snow_second_day = snow_first_day + 8)
  (h2 : snow_fourth_day = snow_second_day - 2)
  (h3 : snow_fifth_day = snow_fourth_day + 2 * snow_first_day)
  (h4 : snow_fifth_day = 24) :
  snow_first_day = 6 := by
  sorry

end NUMINAMATH_GPT_snow_first_day_eq_six_l993_99308


namespace NUMINAMATH_GPT_milk_left_in_storage_l993_99361

-- Define initial and rate conditions
def initialMilk : ℕ := 30000
def pumpedRate : ℕ := 2880
def pumpedHours : ℕ := 4
def addedRate : ℕ := 1500
def addedHours : ℕ := 7

-- The proof problem: Prove the final amount in storage tank == 28980 gallons
theorem milk_left_in_storage : 
  initialMilk - (pumpedRate * pumpedHours) + (addedRate * addedHours) = 28980 := 
sorry

end NUMINAMATH_GPT_milk_left_in_storage_l993_99361


namespace NUMINAMATH_GPT_circles_intersect_l993_99349

variable (r1 r2 d : ℝ)
variable (h1 : r1 = 4)
variable (h2 : r2 = 5)
variable (h3 : d = 7)

theorem circles_intersect : 1 < d ∧ d < r1 + r2 :=
by sorry

end NUMINAMATH_GPT_circles_intersect_l993_99349


namespace NUMINAMATH_GPT_proof_problem_l993_99354

-- Define the problem space
variables (x y : ℝ)

-- Define the conditions
def satisfies_condition (x y : ℝ) : Prop :=
  (0 < x) ∧ (0 < y) ∧ (4 * Real.log x + 2 * Real.log (2 * y) ≥ x^2 + 8 * y - 4)

-- The theorem statement
theorem proof_problem (hx : 0 < x) (hy : 0 < y) (hcond : satisfies_condition x y) :
  x + 2 * y = 1/2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_proof_problem_l993_99354


namespace NUMINAMATH_GPT_domain_of_function_l993_99326

theorem domain_of_function :
  ∀ x : ℝ, (1 / (1 - x) ≥ 0 ∧ 1 - x ≠ 0) ↔ (x < 1) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l993_99326


namespace NUMINAMATH_GPT_field_length_to_width_ratio_l993_99328
-- Import the math library

-- Define the problem conditions and proof goal statement
theorem field_length_to_width_ratio (w : ℝ) (l : ℝ) (area_pond : ℝ) (area_field : ℝ) 
    (h_length : l = 16) (h_area_pond : area_pond = 64) 
    (h_area_relation : area_pond = (1/2) * area_field)
    (h_field_area : area_field = l * w) : l / w = 2 :=
by 
  -- Leaving the proof as an exercise
  sorry

end NUMINAMATH_GPT_field_length_to_width_ratio_l993_99328


namespace NUMINAMATH_GPT_range_of_f_l993_99305

/-- Define the piecewise function f(x) -/
noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 1 else Real.cos x

/-- Prove that the range of f(x) is [-1, ∞) -/
theorem range_of_f : Set.range f = Set.Ici (-1) :=
by sorry

end NUMINAMATH_GPT_range_of_f_l993_99305


namespace NUMINAMATH_GPT_smallest_third_term_geometric_l993_99339

theorem smallest_third_term_geometric (d : ℝ) : 
  (∃ d, (7 + d) ^ 2 = 4 * (26 + 2 * d)) → ∃ g3, (g3 = 10 ∨ g3 = 36) ∧ g3 = min (10) (36) :=
by
  sorry

end NUMINAMATH_GPT_smallest_third_term_geometric_l993_99339


namespace NUMINAMATH_GPT_weight_of_b_l993_99379

theorem weight_of_b (a b c : ℝ) (h1 : a + b + c = 126) (h2 : a + b = 80) (h3 : b + c = 86) : b = 40 :=
sorry

end NUMINAMATH_GPT_weight_of_b_l993_99379


namespace NUMINAMATH_GPT_largest_number_of_HCF_LCM_l993_99311

theorem largest_number_of_HCF_LCM (HCF : ℕ) (k1 k2 : ℕ) (n1 n2 : ℕ) 
  (hHCF : HCF = 50)
  (hk1 : k1 = 11) 
  (hk2 : k2 = 12) 
  (hn1 : n1 = HCF * k1) 
  (hn2 : n2 = HCF * k2) :
  max n1 n2 = 600 := by
  sorry

end NUMINAMATH_GPT_largest_number_of_HCF_LCM_l993_99311


namespace NUMINAMATH_GPT_rhombus_diagonal_length_l993_99376

theorem rhombus_diagonal_length (d1 d2 : ℝ) (Area : ℝ) 
  (h1 : d1 = 12) (h2 : Area = 60) 
  (h3 : Area = (d1 * d2) / 2) : d2 = 10 := 
by
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_length_l993_99376


namespace NUMINAMATH_GPT_geometric_sequence_b_eq_neg3_l993_99386

theorem geometric_sequence_b_eq_neg3 (a b c : ℝ) : 
  (∃ r : ℝ, -1 = r * a ∧ a = r * b ∧ b = r * c ∧ c = r * (-9)) → b = -3 :=
by
  intro h
  obtain ⟨r, h1, h2, h3, h4⟩ := h
  -- Proof to be filled in later.
  sorry

end NUMINAMATH_GPT_geometric_sequence_b_eq_neg3_l993_99386


namespace NUMINAMATH_GPT_ken_height_l993_99387

theorem ken_height 
  (height_ivan : ℝ) (height_jackie : ℝ) (height_ken : ℝ)
  (h1 : height_ivan = 175) (h2 : height_jackie = 175)
  (h_avg : (height_ivan + height_jackie + height_ken) / 3 = (height_ivan + height_jackie) / 2 * 1.04) :
  height_ken = 196 := 
sorry

end NUMINAMATH_GPT_ken_height_l993_99387


namespace NUMINAMATH_GPT_set_difference_M_N_l993_99325

def setM : Set ℝ := { x | -1 < x ∧ x < 1 }
def setN : Set ℝ := { x | x / (x - 1) ≤ 0 }

theorem set_difference_M_N :
  setM \ setN = { x | -1 < x ∧ x < 0 } := sorry

end NUMINAMATH_GPT_set_difference_M_N_l993_99325


namespace NUMINAMATH_GPT_eccentricity_ratio_l993_99375

noncomputable def ellipse_eccentricity (m n : ℝ) : ℝ := (1 - (1 / n) / (1 / m))^(1/2)

theorem eccentricity_ratio (m n : ℝ) (h : ellipse_eccentricity m n = 1 / 2) :
  m / n = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_ratio_l993_99375


namespace NUMINAMATH_GPT_total_tickets_sold_l993_99331

-- Define the parameters and conditions
def VIP_ticket_price : ℝ := 45.00
def general_ticket_price : ℝ := 20.00
def total_revenue : ℝ := 7500.00
def tickets_difference : ℕ := 276

-- Define the total number of tickets sold
def total_number_of_tickets (V G : ℕ) : ℕ := V + G

-- The theorem to be proved
theorem total_tickets_sold (V G : ℕ) 
  (h1 : VIP_ticket_price * V + general_ticket_price * G = total_revenue)
  (h2 : V = G - tickets_difference) : 
  total_number_of_tickets V G = 336 :=
by
  sorry

end NUMINAMATH_GPT_total_tickets_sold_l993_99331


namespace NUMINAMATH_GPT_sum_of_legs_le_sqrt2_hypotenuse_l993_99346

theorem sum_of_legs_le_sqrt2_hypotenuse
  (a b c : ℝ)
  (h : a^2 + b^2 = c^2) :
  a + b ≤ Real.sqrt 2 * c :=
sorry

end NUMINAMATH_GPT_sum_of_legs_le_sqrt2_hypotenuse_l993_99346


namespace NUMINAMATH_GPT_measure_of_angle_C_l993_99338

variable (A B C : Real)

theorem measure_of_angle_C (h1 : 4 * Real.sin A + 2 * Real.cos B = 4) 
                           (h2 : (1/2) * Real.sin B + Real.cos A = Real.sqrt 3 / 2) :
                           C = Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_C_l993_99338


namespace NUMINAMATH_GPT_find_second_number_l993_99363

def average (nums : List ℕ) : ℕ :=
  nums.sum / nums.length

theorem find_second_number (nums : List ℕ) (a b : ℕ) (avg : ℕ) :
  average [10, 70, 28] = 36 ∧ average (10 :: 70 :: 28 :: []) + 4 = avg ∧ average (a :: b :: nums) = avg ∧ a = 20 ∧ b = 60 → b = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_second_number_l993_99363


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_45_deg_4_inradius_l993_99357

theorem right_triangle_hypotenuse_45_deg_4_inradius : 
  ∀ (R : ℝ) (hypotenuse_length : ℝ), R = 4 ∧ 
  (∀ (A B C : ℝ), A = 45 ∧ B = 45 ∧ C = 90) →
  hypotenuse_length = 8 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_45_deg_4_inradius_l993_99357


namespace NUMINAMATH_GPT_units_digit_27_3_sub_17_3_l993_99351

theorem units_digit_27_3_sub_17_3 : 
  (27 ^ 3 - 17 ^ 3) % 10 = 0 :=
sorry

end NUMINAMATH_GPT_units_digit_27_3_sub_17_3_l993_99351


namespace NUMINAMATH_GPT_complement_intersection_l993_99310

open Set

variable (U M N : Set ℕ)
variable (H₁ : U = {1, 2, 3, 4, 5, 6})
variable (H₂ : M = {1, 2, 3, 5})
variable (H₃ : N = {1, 3, 4, 6})

theorem complement_intersection :
  (U \ (M ∩ N)) = {2, 4, 5, 6} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l993_99310


namespace NUMINAMATH_GPT_simplify_and_evaluate_l993_99342

theorem simplify_and_evaluate (a b : ℝ) (h_eqn : a^2 + b^2 - 2 * a + 4 * b = -5) :
  (a - 2 * b) * (a^2 + 2 * a * b + 4 * b^2) - a * (a - 5 * b) * (a + 3 * b) = 120 :=
sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l993_99342


namespace NUMINAMATH_GPT_coloring_count_l993_99391

theorem coloring_count : 
  ∀ (n : ℕ), n = 2021 → 
  ∃ (ways : ℕ), ways = 3 * 2 ^ 2020 :=
by
  intros n hn
  existsi 3 * 2 ^ 2020
  sorry

end NUMINAMATH_GPT_coloring_count_l993_99391


namespace NUMINAMATH_GPT_quadratic_root_proof_l993_99333

noncomputable def root_condition (p q m n : ℝ) :=
  ∃ x : ℝ, x^2 + p * x + q = 0 ∧ x ≠ 0 ∧ (1/x)^2 + m * (1/x) + n = 0

theorem quadratic_root_proof (p q m n : ℝ) (h : root_condition p q m n) :
  (pn - m) * (qm - p) = (qn - 1)^2 :=
sorry

end NUMINAMATH_GPT_quadratic_root_proof_l993_99333


namespace NUMINAMATH_GPT_tom_candies_left_is_ten_l993_99314

-- Define initial conditions
def initial_candies: ℕ := 2
def friend_gave_candies: ℕ := 7
def bought_candies: ℕ := 10

-- Define total candies before sharing
def total_candies := initial_candies + friend_gave_candies + bought_candies

-- Define the number of candies Tom gives to his sister
def candies_given := total_candies / 2

-- Define the number of candies Tom has left
def candies_left := total_candies - candies_given

-- Prove the final number of candies left
theorem tom_candies_left_is_ten : candies_left = 10 :=
by
  -- The proof is left as an exercise
  sorry

end NUMINAMATH_GPT_tom_candies_left_is_ten_l993_99314


namespace NUMINAMATH_GPT_range_of_a_l993_99329

theorem range_of_a (a x y : ℝ) (h1: x + 3 * y = 2 + a) (h2: 3 * x + y = -4 * a) (h3: x + y > 2) : a < -2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l993_99329


namespace NUMINAMATH_GPT_students_needed_to_fill_buses_l993_99388

theorem students_needed_to_fill_buses (n : ℕ) (c : ℕ) (h_n : n = 254) (h_c : c = 30) : 
  (c * ((n + c - 1) / c) - n) = 16 :=
by
  sorry

end NUMINAMATH_GPT_students_needed_to_fill_buses_l993_99388


namespace NUMINAMATH_GPT_landscape_length_l993_99348

theorem landscape_length (b : ℝ) 
  (h1 : ∀ (l : ℝ), l = 8 * b) 
  (A : ℝ)
  (h2 : A = 8 * b^2)
  (Playground_area : ℝ)
  (h3 : Playground_area = 1200)
  (h4 : Playground_area = (1 / 6) * A) :
  ∃ (l : ℝ), l = 240 :=
by 
  sorry

end NUMINAMATH_GPT_landscape_length_l993_99348


namespace NUMINAMATH_GPT_travel_speed_l993_99352

theorem travel_speed (distance : ℝ) (time : ℝ) (h_distance : distance = 195) (h_time : time = 3) : 
  distance / time = 65 :=
by 
  rw [h_distance, h_time]
  norm_num

end NUMINAMATH_GPT_travel_speed_l993_99352


namespace NUMINAMATH_GPT_employees_6_or_more_percentage_is_18_l993_99335

-- Defining the employee counts for different year ranges
def count_less_than_1 (y : ℕ) : ℕ := 4 * y
def count_1_to_2 (y : ℕ) : ℕ := 6 * y
def count_2_to_3 (y : ℕ) : ℕ := 7 * y
def count_3_to_4 (y : ℕ) : ℕ := 4 * y
def count_4_to_5 (y : ℕ) : ℕ := 3 * y
def count_5_to_6 (y : ℕ) : ℕ := 3 * y
def count_6_to_7 (y : ℕ) : ℕ := 2 * y
def count_7_to_8 (y : ℕ) : ℕ := 2 * y
def count_8_to_9 (y : ℕ) : ℕ := y
def count_9_to_10 (y : ℕ) : ℕ := y

-- Sum of all employees T
def total_employees (y : ℕ) : ℕ := count_less_than_1 y + count_1_to_2 y + count_2_to_3 y +
                                    count_3_to_4 y + count_4_to_5 y + count_5_to_6 y +
                                    count_6_to_7 y + count_7_to_8 y + count_8_to_9 y +
                                    count_9_to_10 y

-- Employees with 6 years or more E
def employees_6_or_more (y : ℕ) : ℕ := count_6_to_7 y + count_7_to_8 y + count_8_to_9 y + count_9_to_10 y

-- Calculate percentage
def percentage (y : ℕ) : ℚ := (employees_6_or_more y : ℚ) / (total_employees y : ℚ) * 100

-- Proving the final statement
theorem employees_6_or_more_percentage_is_18 (y : ℕ) (hy : y ≠ 0) : percentage y = 18 :=
by
  sorry

end NUMINAMATH_GPT_employees_6_or_more_percentage_is_18_l993_99335


namespace NUMINAMATH_GPT_DavidCrunchesLessThanZachary_l993_99370

-- Definitions based on conditions
def ZacharyPushUps : ℕ := 44
def ZacharyCrunches : ℕ := 17
def DavidPushUps : ℕ := ZacharyPushUps + 29
def DavidCrunches : ℕ := 4

-- Problem statement we need to prove:
theorem DavidCrunchesLessThanZachary : DavidCrunches = ZacharyCrunches - 13 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_DavidCrunchesLessThanZachary_l993_99370


namespace NUMINAMATH_GPT_opposite_of_one_fourth_l993_99372

/-- The opposite of the fraction 1/4 is -1/4 --/
theorem opposite_of_one_fourth : - (1 / 4) = -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_one_fourth_l993_99372


namespace NUMINAMATH_GPT_measure_of_angle_C_sin_A_plus_sin_B_l993_99302

-- Problem 1
theorem measure_of_angle_C (a b c : ℝ) (h1 : a^2 + b^2 - c^2 = 8) (h2 : (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) : C = Real.pi / 3 := 
sorry

-- Problem 2
theorem sin_A_plus_sin_B (a b c A B C : ℝ) (h1 : a^2 + b^2 - c^2 = 8) (h2 : (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) (h3 : c = 2 * Real.sqrt 3) : Real.sin A + Real.sin B = 3 / 2 := 
sorry

end NUMINAMATH_GPT_measure_of_angle_C_sin_A_plus_sin_B_l993_99302


namespace NUMINAMATH_GPT_flour_needed_for_two_loaves_l993_99371

-- Define the amount of flour needed for one loaf.
def flour_per_loaf : ℝ := 2.5

-- Define the number of loaves.
def number_of_loaves : ℕ := 2

-- Define the total amount of flour needed for the given number of loaves.
def total_flour_needed : ℝ := flour_per_loaf * number_of_loaves

-- The theorem statement: Prove that the total amount of flour needed is 5 cups.
theorem flour_needed_for_two_loaves : total_flour_needed = 5 := by
  sorry

end NUMINAMATH_GPT_flour_needed_for_two_loaves_l993_99371


namespace NUMINAMATH_GPT_number_is_40_l993_99340

theorem number_is_40 (N : ℝ) (h : N = (3/8) * N + (1/4) * N + 15) : N = 40 :=
by
  sorry

end NUMINAMATH_GPT_number_is_40_l993_99340


namespace NUMINAMATH_GPT_difference_between_numbers_l993_99394

theorem difference_between_numbers 
  (L S : ℕ) 
  (hL : L = 1584) 
  (hDiv : L = 6 * S + 15) : 
  L - S = 1323 := 
by
  sorry

end NUMINAMATH_GPT_difference_between_numbers_l993_99394


namespace NUMINAMATH_GPT_grocery_packs_l993_99392

theorem grocery_packs (cookie_packs cake_packs : ℕ)
  (h1 : cookie_packs = 23)
  (h2 : cake_packs = 4) :
  cookie_packs + cake_packs = 27 :=
by
  sorry

end NUMINAMATH_GPT_grocery_packs_l993_99392


namespace NUMINAMATH_GPT_arrangeable_sequence_l993_99382

theorem arrangeable_sequence (n : Fin 2017 → ℤ) :
  (∀ i : Fin 2017, ∃ (perm : Fin 5 → Fin 5),
    let a := n ((i + perm 0) % 2017)
    let b := n ((i + perm 1) % 2017)
    let c := n ((i + perm 2) % 2017)
    let d := n ((i + perm 3) % 2017)
    let e := n ((i + perm 4) % 2017)
    a - b + c - d + e = 29) →
  (∀ i : Fin 2017, n i = 29) :=
by
  sorry

end NUMINAMATH_GPT_arrangeable_sequence_l993_99382


namespace NUMINAMATH_GPT_lindy_distance_l993_99380

theorem lindy_distance
  (d : ℝ) (v_j : ℝ) (v_c : ℝ) (v_l : ℝ) (t : ℝ)
  (h1 : d = 270)
  (h2 : v_j = 4)
  (h3 : v_c = 5)
  (h4 : v_l = 8)
  (h_time : t = d / (v_j + v_c)) :
  v_l * t = 240 := by
  sorry

end NUMINAMATH_GPT_lindy_distance_l993_99380


namespace NUMINAMATH_GPT_problem1_problem2_l993_99393

open Set

noncomputable def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3 * a) < 0}

theorem problem1 (a : ℝ) (h1 : A ⊆ (A ∩ B a)) : (4 / 3 : ℝ) ≤ a ∧ a ≤ 2 :=
sorry

theorem problem2 (a : ℝ) (h2 : A ∩ B a = ∅) : a ≤ (2 / 3 : ℝ) ∨ a ≥ 4 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l993_99393


namespace NUMINAMATH_GPT_kylie_stamps_l993_99367

theorem kylie_stamps (K N : ℕ) (h1 : N = K + 44) (h2 : K + N = 112) : K = 34 :=
by
  sorry

end NUMINAMATH_GPT_kylie_stamps_l993_99367


namespace NUMINAMATH_GPT_valid_set_example_l993_99358

def is_valid_set (S : Set ℝ) : Prop :=
  ∀ x ∈ S, ∃ y ∈ S, x ≠ y

theorem valid_set_example : is_valid_set { x : ℝ | x > Real.sqrt 2 } :=
sorry

end NUMINAMATH_GPT_valid_set_example_l993_99358


namespace NUMINAMATH_GPT_percentage_expression_l993_99318

variable {A B : ℝ} (hA : A > 0) (hB : B > 0)

theorem percentage_expression (h : A = (x / 100) * B) : x = 100 * (A / B) :=
sorry

end NUMINAMATH_GPT_percentage_expression_l993_99318


namespace NUMINAMATH_GPT_combined_mpg_19_l993_99347

theorem combined_mpg_19 (m: ℕ) (h: m = 100) :
  let ray_car_mpg := 50
  let tom_car_mpg := 25
  let jerry_car_mpg := 10
  let ray_gas_used := m / ray_car_mpg
  let tom_gas_used := m / tom_car_mpg
  let jerry_gas_used := m / jerry_car_mpg
  let total_gas_used := ray_gas_used + tom_gas_used + jerry_gas_used
  let total_miles := 3 * m
  let combined_mpg := total_miles * 25 / (4 * m)
  combined_mpg = 19 := 
by {
  sorry
}

end NUMINAMATH_GPT_combined_mpg_19_l993_99347


namespace NUMINAMATH_GPT_find_B_coords_l993_99327

-- Define point A and vector a
def A : (ℝ × ℝ) := (1, -3)
def a : (ℝ × ℝ) := (3, 4)

-- Assume B is at coordinates (m, n) and AB = 2a
def B : (ℝ × ℝ) := (7, 5)
def AB : (ℝ × ℝ) := (B.1 - A.1, B.2 - A.2)

-- Prove point B has the correct coordinates
theorem find_B_coords : AB = (2 * a.1, 2 * a.2) → B = (7, 5) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_B_coords_l993_99327


namespace NUMINAMATH_GPT_ticket_price_profit_condition_maximize_profit_at_7_point_5_l993_99366

-- Define the ticket price increase and the total profit function
def ticket_price (x : ℝ) := (10 + x) * (500 - 20 * x)

-- Prove that the function equals 6000 at x = 10 and x = 25
theorem ticket_price_profit_condition (x : ℝ) :
  ticket_price x = 6000 ↔ (x = 10 ∨ x = 25) :=
by sorry

-- Prove that m = 7.5 maximizes the profit
def profit (m : ℝ) := -20 * m^2 + 300 * m + 5000

theorem maximize_profit_at_7_point_5 (m : ℝ) :
  m = 7.5 ↔ (∀ m, profit 7.5 ≥ profit m) :=
by sorry

end NUMINAMATH_GPT_ticket_price_profit_condition_maximize_profit_at_7_point_5_l993_99366


namespace NUMINAMATH_GPT_length_of_first_train_l993_99397

noncomputable def length_first_train
  (speed_train1_kmh : ℝ)
  (speed_train2_kmh : ℝ)
  (time_sec : ℝ)
  (length_train2_m : ℝ) : ℝ :=
  let relative_speed_mps := (speed_train1_kmh + speed_train2_kmh) * (1000 / 3600)
  let total_distance_m := relative_speed_mps * time_sec
  total_distance_m - length_train2_m

theorem length_of_first_train :
  length_first_train 80 65 7.82006405004841 165 = 150.106201 :=
  by
  -- Proof steps would go here.
  sorry

end NUMINAMATH_GPT_length_of_first_train_l993_99397


namespace NUMINAMATH_GPT_total_carriages_proof_l993_99343

noncomputable def total_carriages (E N' F N : ℕ) : ℕ :=
  E + N + N' + F

theorem total_carriages_proof
  (E N N' F : ℕ)
  (h1 : E = 130)
  (h2 : E = N + 20)
  (h3 : N' = 100)
  (h4 : F = N' + 20) :
  total_carriages E N' F N = 460 := by
  sorry

end NUMINAMATH_GPT_total_carriages_proof_l993_99343


namespace NUMINAMATH_GPT_is_possible_to_finish_7th_l993_99323

theorem is_possible_to_finish_7th 
  (num_teams : ℕ)
  (wins_ASTC : ℕ)
  (losses_ASTC : ℕ)
  (points_per_win : ℕ)
  (points_per_draw : ℕ) 
  (total_points : ℕ)
  (rank_ASTC : ℕ)
  (points_ASTC : ℕ)
  (points_needed_by_top_6 : ℕ → ℕ)
  (points_8th_and_9th : ℕ) :
  num_teams = 9 ∧ wins_ASTC = 5 ∧ losses_ASTC = 3 ∧ points_per_win = 3 ∧ points_per_draw = 1 ∧ 
  total_points = 108 ∧ rank_ASTC = 7 ∧ points_ASTC = 15 ∧ points_needed_by_top_6 7 = 105 ∧ points_8th_and_9th ≤ 3 →
  ∃ (top_7_points : ℕ), 
  top_7_points = 105 ∧ (top_7_points + points_8th_and_9th) = total_points := 
sorry

end NUMINAMATH_GPT_is_possible_to_finish_7th_l993_99323


namespace NUMINAMATH_GPT_functional_eq_solution_l993_99322

open Real

theorem functional_eq_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = f (x - y) + 4 * x * y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x^2 + c := 
sorry

end NUMINAMATH_GPT_functional_eq_solution_l993_99322


namespace NUMINAMATH_GPT_simplify_polynomials_l993_99307

-- Define the polynomials
def poly1 (q : ℝ) : ℝ := 5 * q^4 + 3 * q^3 - 7 * q + 8
def poly2 (q : ℝ) : ℝ := 6 - 9 * q^3 + 4 * q - 3 * q^4

-- The goal is to prove that the sum of poly1 and poly2 simplifies correctly
theorem simplify_polynomials (q : ℝ) : 
  poly1 q + poly2 q = 2 * q^4 - 6 * q^3 - 3 * q + 14 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_polynomials_l993_99307


namespace NUMINAMATH_GPT_junghyeon_stickers_l993_99300

def total_stickers : ℕ := 25
def junghyeon_sticker_count (yejin_stickers : ℕ) : ℕ := 2 * yejin_stickers + 1

theorem junghyeon_stickers (yejin_stickers : ℕ) (h : yejin_stickers + junghyeon_sticker_count yejin_stickers = total_stickers) : 
  junghyeon_sticker_count yejin_stickers = 17 :=
  by
  sorry

end NUMINAMATH_GPT_junghyeon_stickers_l993_99300


namespace NUMINAMATH_GPT_cube_sphere_volume_relation_l993_99316

theorem cube_sphere_volume_relation (n : ℕ) (h : 2 < n)
  (h_volume : n^3 - (n^3 * pi / 6) = (n^3 * pi / 3)) : n = 8 :=
sorry

end NUMINAMATH_GPT_cube_sphere_volume_relation_l993_99316


namespace NUMINAMATH_GPT_machine_working_time_l993_99378

def shirts_per_minute : ℕ := 3
def total_shirts_made : ℕ := 6

theorem machine_working_time : 
  (total_shirts_made / shirts_per_minute) = 2 :=
by
  sorry

end NUMINAMATH_GPT_machine_working_time_l993_99378


namespace NUMINAMATH_GPT_smallest_number_of_ones_l993_99315

-- Definitions inferred from the problem conditions
def N := (10^100 - 1) / 3
def M_k (k : ℕ) := (10^k - 1) / 9

theorem smallest_number_of_ones (k : ℕ) : M_k k % N = 0 → k = 300 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_number_of_ones_l993_99315


namespace NUMINAMATH_GPT_negative_number_unique_l993_99373

theorem negative_number_unique (a b c d : ℚ) (h₁ : a = 1) (h₂ : b = 0) (h₃ : c = 1/2) (h₄ : d = -2) :
  ∃! x : ℚ, x < 0 ∧ (x = a ∨ x = b ∨ x = c ∨ x = d) :=
by 
  sorry

end NUMINAMATH_GPT_negative_number_unique_l993_99373


namespace NUMINAMATH_GPT_passengers_initial_count_l993_99306

-- Let's define the initial number of passengers
variable (P : ℕ)

-- Given conditions:
def final_passengers (initial additional left : ℕ) : ℕ := initial + additional - left

-- The theorem statement to prove P = 28 given the conditions
theorem passengers_initial_count
  (final_count : ℕ)
  (h1 : final_count = 26)
  (h2 : final_passengers P 7 9 = final_count) 
  : P = 28 :=
by
  sorry

end NUMINAMATH_GPT_passengers_initial_count_l993_99306


namespace NUMINAMATH_GPT_cricket_team_rh_players_l993_99341

theorem cricket_team_rh_players (total_players throwers non_throwers lh_non_throwers rh_non_throwers rh_players : ℕ)
    (h1 : total_players = 58)
    (h2 : throwers = 37)
    (h3 : non_throwers = total_players - throwers)
    (h4 : lh_non_throwers = non_throwers / 3)
    (h5 : rh_non_throwers = non_throwers - lh_non_throwers)
    (h6 : rh_players = throwers + rh_non_throwers) :
  rh_players = 51 := by
  sorry

end NUMINAMATH_GPT_cricket_team_rh_players_l993_99341


namespace NUMINAMATH_GPT_valid_k_values_l993_99336

theorem valid_k_values
  (k : ℝ)
  (h : k = -7 ∨ k = -5 ∨ k = 1 ∨ k = 4) :
  (∀ x, -4 < x ∧ x < 1 → (x < k ∨ x > k + 2)) → (k = -7 ∨ k = 1 ∨ k = 4) :=
by sorry

end NUMINAMATH_GPT_valid_k_values_l993_99336


namespace NUMINAMATH_GPT_total_distance_is_10_miles_l993_99385

noncomputable def total_distance_back_to_town : ℕ :=
  let distance1 := 3
  let distance2 := 3
  let distance3 := 4
  distance1 + distance2 + distance3

theorem total_distance_is_10_miles :
  total_distance_back_to_town = 10 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_is_10_miles_l993_99385


namespace NUMINAMATH_GPT_negation_exists_gt_one_l993_99312

theorem negation_exists_gt_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) :=
sorry

end NUMINAMATH_GPT_negation_exists_gt_one_l993_99312


namespace NUMINAMATH_GPT_right_angled_triangle_exists_l993_99396

def is_triangle (a b c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_right_angled_triangle (a b c : ℕ) : Prop :=
  (a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2) ∨ (b^2 + c^2 = a^2)

theorem right_angled_triangle_exists :
  is_triangle 3 4 5 ∧ is_right_angled_triangle 3 4 5 :=
by
  sorry

end NUMINAMATH_GPT_right_angled_triangle_exists_l993_99396


namespace NUMINAMATH_GPT_distance_between_consecutive_trees_l993_99364

theorem distance_between_consecutive_trees 
  (yard_length : ℕ) (num_trees : ℕ) (tree_at_each_end : yard_length > 0 ∧ num_trees ≥ 2) 
  (equal_distances : ∀ k, k < num_trees - 1 → (yard_length / (num_trees - 1) : ℝ) = 12) :
  yard_length = 360 → num_trees = 31 → (yard_length / (num_trees - 1) : ℝ) = 12 := 
by
  sorry

end NUMINAMATH_GPT_distance_between_consecutive_trees_l993_99364


namespace NUMINAMATH_GPT_evaluate_expression_l993_99304

noncomputable def greatest_integer (x : Real) : Int := ⌊x⌋

theorem evaluate_expression (y : Real) (h : y = 7.2) :
  greatest_integer 6.5 * greatest_integer (2 / 3)
  + greatest_integer 2 * y
  + greatest_integer 8.4 - 6.0 = 16.4 := by
  simp [greatest_integer, h]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l993_99304


namespace NUMINAMATH_GPT_difference_of_squares_divisible_by_9_l993_99337

theorem difference_of_squares_divisible_by_9 (a b : ℤ) : 
  9 ∣ ((3 * a + 2)^2 - (3 * b + 2)^2) :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_divisible_by_9_l993_99337


namespace NUMINAMATH_GPT_round_robin_total_points_l993_99319

theorem round_robin_total_points :
  let points_per_match := 2
  let total_matches := 3
  (total_matches * points_per_match) = 6 :=
by
  sorry

end NUMINAMATH_GPT_round_robin_total_points_l993_99319


namespace NUMINAMATH_GPT_students_all_three_classes_l993_99356

variables (H M E HM HE ME HME : ℕ)

-- Conditions from the problem
def student_distribution : Prop :=
  H = 12 ∧
  M = 17 ∧
  E = 36 ∧
  HM + HE + ME = 3 ∧
  86 = H + M + E - (HM + HE + ME) + HME

-- Prove the number of students registered for all three classes
theorem students_all_three_classes (h : student_distribution H M E HM HE ME HME) : HME = 24 :=
  by sorry

end NUMINAMATH_GPT_students_all_three_classes_l993_99356


namespace NUMINAMATH_GPT_train_length_is_correct_l993_99369

noncomputable def length_of_train (speed_train_kmh : ℝ) (speed_man_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let relative_speed_kmh := speed_train_kmh + speed_man_kmh
  let relative_speed_ms := relative_speed_kmh * (5/18)
  let length := relative_speed_ms * time_s
  length

theorem train_length_is_correct (h1 : 84 = 84) (h2 : 6 = 6) (h3 : 4.399648028157747 = 4.399648028157747) :
  length_of_train 84 6 4.399648028157747 = 110.991201 := by
  dsimp [length_of_train]
  norm_num
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l993_99369


namespace NUMINAMATH_GPT_horizontal_distance_l993_99320

def curve (x : ℝ) := x^3 - x^2 - x - 6

def P_condition (x : ℝ) := curve x = 10
def Q_condition1 (x : ℝ) := curve x = 2
def Q_condition2 (x : ℝ) := curve x = -2

theorem horizontal_distance (x_P x_Q: ℝ) (hP: P_condition x_P) (hQ1: Q_condition1 x_Q ∨ Q_condition2 x_Q) :
  |x_P - x_Q| = 3 := sorry

end NUMINAMATH_GPT_horizontal_distance_l993_99320


namespace NUMINAMATH_GPT_lcm_of_36_and_100_l993_99324

theorem lcm_of_36_and_100 : Nat.lcm 36 100 = 900 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_lcm_of_36_and_100_l993_99324


namespace NUMINAMATH_GPT_two_digit_number_tens_place_l993_99344

theorem two_digit_number_tens_place (x y : Nat) (hx1 : 0 ≤ x) (hx2 : x ≤ 9) (hy1 : 0 ≤ y) (hy2 : y ≤ 9)
    (h : (x + y) * 3 = 10 * x + y - 2) : x = 2 := 
sorry

end NUMINAMATH_GPT_two_digit_number_tens_place_l993_99344


namespace NUMINAMATH_GPT_relationship_a_b_l993_99384

noncomputable def e : ℝ := Real.exp 1

theorem relationship_a_b
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : e^a + 2 * a = e^b + 3 * b) :
  a > b :=
sorry

end NUMINAMATH_GPT_relationship_a_b_l993_99384


namespace NUMINAMATH_GPT_hallie_hours_worked_on_tuesday_l993_99398

theorem hallie_hours_worked_on_tuesday
    (hourly_wage : ℝ := 10)
    (hours_monday : ℝ := 7)
    (tips_monday : ℝ := 18)
    (hours_wednesday : ℝ := 7)
    (tips_wednesday : ℝ := 20)
    (tips_tuesday : ℝ := 12)
    (total_earnings : ℝ := 240)
    (tuesday_hours : ℝ) :
    (hourly_wage * hours_monday + tips_monday) +
    (hourly_wage * hours_wednesday + tips_wednesday) +
    (hourly_wage * tuesday_hours + tips_tuesday) = total_earnings →
    tuesday_hours = 5 :=
by
  sorry

end NUMINAMATH_GPT_hallie_hours_worked_on_tuesday_l993_99398


namespace NUMINAMATH_GPT_correct_scientific_notation_representation_l993_99303

-- Defining the given number of visitors in millions
def visitors_in_millions : Float := 8.0327
-- Converting this number to an integer and expressing in scientific notation
def rounded_scientific_notation (num : Float) : String :=
  if num == 8.0327 then "8.0 × 10^6" else "incorrect"

-- The mathematical proof statement
theorem correct_scientific_notation_representation :
  rounded_scientific_notation visitors_in_millions = "8.0 × 10^6" :=
by
  sorry

end NUMINAMATH_GPT_correct_scientific_notation_representation_l993_99303


namespace NUMINAMATH_GPT_transformed_cubic_polynomial_l993_99389

theorem transformed_cubic_polynomial (x z : ℂ) 
    (h1 : z = x + x⁻¹) (h2 : x^3 - 3 * x^2 + x + 2 = 0) : 
    x^2 * (z^2 - z - 1) + 3 = 0 :=
sorry

end NUMINAMATH_GPT_transformed_cubic_polynomial_l993_99389


namespace NUMINAMATH_GPT_smaller_of_two_digit_product_l993_99362

theorem smaller_of_two_digit_product (a b : ℕ) (ha : 10 ≤ a) (hb : 10 ≤ b) (ha' : a < 100) (hb' : b < 100) 
  (hprod : a * b = 4680) : min a b = 52 :=
by
  sorry

end NUMINAMATH_GPT_smaller_of_two_digit_product_l993_99362


namespace NUMINAMATH_GPT_prime_cannot_be_sum_of_three_squares_l993_99301

theorem prime_cannot_be_sum_of_three_squares (p : ℕ) (hp : Nat.Prime p) (hmod : p % 8 = 7) :
  ¬∃ a b c : ℤ, p = a^2 + b^2 + c^2 :=
by
  sorry

end NUMINAMATH_GPT_prime_cannot_be_sum_of_three_squares_l993_99301


namespace NUMINAMATH_GPT_log_product_l993_99383

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_product (x y : ℝ) (hx : 0 < x) (hy : 1 < y) :
  log_base (y^3) x * log_base (x^4) (y^3) * log_base (y^5) (x^2) * log_base (x^2) (y^5) * log_base (y^3) (x^4) =
  (1/3) * log_base y x :=
by
  sorry

end NUMINAMATH_GPT_log_product_l993_99383


namespace NUMINAMATH_GPT_perpendicular_lines_slope_condition_l993_99332

theorem perpendicular_lines_slope_condition (k : ℝ) :
  (∀ x y : ℝ, y = k * x - 1 ↔ x + 2 * y + 3 = 0) → k = 2 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_slope_condition_l993_99332


namespace NUMINAMATH_GPT_claire_balloon_count_l993_99399

variable (start_balloons lost_balloons initial_give_away more_give_away final_balloons grabbed_from_coworker : ℕ)

theorem claire_balloon_count (h1 : start_balloons = 50)
                           (h2 : lost_balloons = 12)
                           (h3 : initial_give_away = 1)
                           (h4 : more_give_away = 9)
                           (h5 : final_balloons = 39)
                           (h6 : start_balloons - initial_give_away - lost_balloons - more_give_away + grabbed_from_coworker = final_balloons) :
                           grabbed_from_coworker = 11 :=
by
  sorry

end NUMINAMATH_GPT_claire_balloon_count_l993_99399


namespace NUMINAMATH_GPT_greatest_whole_number_solution_l993_99353

theorem greatest_whole_number_solution (x : ℤ) (h : 6 * x - 5 < 7 - 3 * x) : x ≤ 1 :=
sorry

end NUMINAMATH_GPT_greatest_whole_number_solution_l993_99353


namespace NUMINAMATH_GPT_sum_of_first_two_digits_of_repeating_decimal_l993_99350

theorem sum_of_first_two_digits_of_repeating_decimal (c d : ℕ) (h : (c, d) = (3, 5)) : c + d = 8 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_first_two_digits_of_repeating_decimal_l993_99350


namespace NUMINAMATH_GPT_ratio_of_height_to_radius_min_surface_area_l993_99368

theorem ratio_of_height_to_radius_min_surface_area 
  (r h : ℝ)
  (V : ℝ := 500)
  (volume_cond : π * r^2 * h = V)
  (surface_area : ℝ := 2 * π * r^2 + 2 * π * r * h) : 
  h / r = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_height_to_radius_min_surface_area_l993_99368


namespace NUMINAMATH_GPT_count_even_numbers_l993_99330

theorem count_even_numbers : 
  ∃ n : ℕ, n = 199 ∧ ∀ m : ℕ, (302 ≤ m ∧ m < 700 ∧ m % 2 = 0) → 
    151 ≤ ((m - 300) / 2) ∧ ((m - 300) / 2) ≤ 349 :=
sorry

end NUMINAMATH_GPT_count_even_numbers_l993_99330


namespace NUMINAMATH_GPT_part1_part2_l993_99313

open Set Real

def M (x : ℝ) : Prop := x^2 - 3 * x - 18 ≤ 0
def N (x : ℝ) (a : ℝ) : Prop := 1 - a ≤ x ∧ x ≤ 2 * a + 1

theorem part1 (a : ℝ) (h : a = 3) : (Icc (-2 : ℝ) 6 = {x | M x ∧ N x a}) ∧ (compl {x | N x a} = Iic (-2) ∪ Ioi 7) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x, M x ∧ N x a ↔ N x a) → a ≤ 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l993_99313


namespace NUMINAMATH_GPT_cos_B_arithmetic_sequence_sin_A_sin_C_geometric_sequence_l993_99377

theorem cos_B_arithmetic_sequence (A B C : ℝ) (h1 : 2 * B = A + C) (h2 : A + B + C = 180) :
  Real.cos B = 1 / 2 :=
by
  sorry

theorem sin_A_sin_C_geometric_sequence (A B C a b c : ℝ) (h1 : 2 * B = A + C) (h2 : A + B + C = 180)
  (h3 : b^2 = a * c) (h4 : b^2 = a^2 + c^2 - 2 * a * c * Real.cos B) :
  Real.sin A * Real.sin C = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_cos_B_arithmetic_sequence_sin_A_sin_C_geometric_sequence_l993_99377


namespace NUMINAMATH_GPT_molecular_weight_calculation_l993_99395

-- Define the condition given in the problem
def molecular_weight_of_4_moles := 488 -- molecular weight of 4 moles in g/mol

-- Define the number of moles
def number_of_moles := 4

-- Define the expected molecular weight of 1 mole
def expected_molecular_weight_of_1_mole := 122 -- molecular weight of 1 mole in g/mol

-- Theorem statement
theorem molecular_weight_calculation : 
  molecular_weight_of_4_moles / number_of_moles = expected_molecular_weight_of_1_mole := 
by
  sorry

end NUMINAMATH_GPT_molecular_weight_calculation_l993_99395


namespace NUMINAMATH_GPT_simplify_fraction_l993_99381

variable (x y : ℝ)

theorem simplify_fraction :
  (2 * x + y) / 4 + (5 * y - 4 * x) / 6 - y / 12 = (-x + 6 * y) / 6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l993_99381


namespace NUMINAMATH_GPT_absolute_value_inequality_l993_99359

theorem absolute_value_inequality (x : ℝ) : ¬ (|x - 3| + |x + 4| < 6) :=
sorry

end NUMINAMATH_GPT_absolute_value_inequality_l993_99359
