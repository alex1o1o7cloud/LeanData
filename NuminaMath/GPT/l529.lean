import Mathlib

namespace f_cos_eq_l529_52960

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Given condition
axiom f_sin_eq : f (Real.sin x) = 3 - Real.cos (2 * x)

-- The statement we want to prove
theorem f_cos_eq : f (Real.cos x) = 3 + Real.cos (2 * x) := 
by
  sorry

end f_cos_eq_l529_52960


namespace max_value_of_expression_l529_52949

theorem max_value_of_expression (x y : ℝ) (h1 : |x - y| ≤ 2) (h2 : |3 * x + y| ≤ 6) : x^2 + y^2 ≤ 10 :=
sorry

end max_value_of_expression_l529_52949


namespace number_of_rolls_in_case_l529_52950

-- Definitions: Cost of a case, cost per roll individually, percent savings per roll
def cost_of_case : ℝ := 9
def cost_per_roll_individual : ℝ := 1
def percent_savings_per_roll : ℝ := 0.25

-- Theorem: Proving the number of rolls in the case is 12
theorem number_of_rolls_in_case (n : ℕ) (h1 : cost_of_case = 9)
    (h2 : cost_per_roll_individual = 1)
    (h3 : percent_savings_per_roll = 0.25) : n = 12 := 
  sorry

end number_of_rolls_in_case_l529_52950


namespace time_after_4350_minutes_is_march_6_00_30_l529_52974

-- Define the start time as a date
def startDate := (2015, 3, 3, 0, 0) -- March 3, 2015 at midnight (00:00)

-- Define the total minutes to add
def totalMinutes := 4350

-- Function to convert minutes to a date and time given a start date
def addMinutes (date : (Nat × Nat × Nat × Nat × Nat)) (minutes : Nat) : (Nat × Nat × Nat × Nat × Nat) :=
  let hours := minutes / 60
  let remainMinutes := minutes % 60
  let days := hours / 24
  let remainHours := hours % 24
  let (year, month, day, hour, min) := date
  (year, month, day + days, remainHours, remainMinutes)

-- Expected result date and time
def expectedDate := (2015, 3, 6, 0, 30) -- March 6, 2015 at 00:30 AM

theorem time_after_4350_minutes_is_march_6_00_30 :
  addMinutes startDate totalMinutes = expectedDate :=
by
  sorry

end time_after_4350_minutes_is_march_6_00_30_l529_52974


namespace vector_dot_product_proof_l529_52957

def vector_dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem vector_dot_product_proof : 
  let a := (-1, 2)
  let b := (2, 3)
  vector_dot_product a (a.1 - b.1, a.2 - b.2) = 1 :=
by {
  sorry
}

end vector_dot_product_proof_l529_52957


namespace teal_sold_pumpkin_pies_l529_52969

def pies_sold 
  (pumpkin_pie_slices : ℕ) (pumpkin_pie_price : ℕ) 
  (custard_pie_slices : ℕ) (custard_pie_price : ℕ) 
  (custard_pies_sold : ℕ) (total_revenue : ℕ) : ℕ :=
  total_revenue / (pumpkin_pie_slices * pumpkin_pie_price)

theorem teal_sold_pumpkin_pies : 
  pies_sold 8 5 6 6 5 340 = 4 := 
by 
  sorry

end teal_sold_pumpkin_pies_l529_52969


namespace lena_candy_bars_l529_52997

/-- Lena has some candy bars. She needs 5 more candy bars to have 3 times as many as Kevin,
and Kevin has 4 candy bars less than Nicole. Lena has 5 more candy bars than Nicole.
How many candy bars does Lena have? -/
theorem lena_candy_bars (L K N : ℕ) 
  (h1 : L + 5 = 3 * K)
  (h2 : K = N - 4)
  (h3 : L = N + 5) : 
  L = 16 :=
sorry

end lena_candy_bars_l529_52997


namespace max_value_of_f_l529_52947

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x + Real.sin x

theorem max_value_of_f : ∃ M, ∀ x, f x ≤ M ∧ (∃ y, f y = M) := by
  use Real.sqrt 5
  sorry

end max_value_of_f_l529_52947


namespace jerry_remaining_money_l529_52915

def cost_of_mustard_oil (price_per_liter : ℕ) (liters : ℕ) : ℕ := price_per_liter * liters
def cost_of_pasta (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds
def cost_of_sauce (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds

def total_cost (price_mustard_oil : ℕ) (liters_mustard : ℕ) (price_pasta : ℕ) (pounds_pasta : ℕ) (price_sauce : ℕ) (pounds_sauce : ℕ) : ℕ := 
  cost_of_mustard_oil price_mustard_oil liters_mustard + cost_of_pasta price_pasta pounds_pasta + cost_of_sauce price_sauce pounds_sauce

def remaining_money (initial_money : ℕ) (total_cost : ℕ) : ℕ := initial_money - total_cost

theorem jerry_remaining_money : 
  remaining_money 50 (total_cost 13 2 4 3 5 1) = 7 := by 
  sorry

end jerry_remaining_money_l529_52915


namespace frac_1_7_correct_l529_52952

-- Define the fraction 1/7
def frac_1_7 : ℚ := 1 / 7

-- Define the decimal approximation 0.142857142857 as a rational number
def dec_approx : ℚ := 142857142857 / 10^12

-- Define the small fractional difference
def small_diff : ℚ := 1 / (7 * 10^12)

-- The theorem to be proven
theorem frac_1_7_correct :
  frac_1_7 = dec_approx + small_diff := 
sorry

end frac_1_7_correct_l529_52952


namespace sector_area_l529_52934

theorem sector_area (R : ℝ) (hR_pos : R > 0) (h_circumference : 4 * R = 2 * R + arc_length) :
  (1 / 2) * arc_length * R = R^2 :=
by sorry

end sector_area_l529_52934


namespace average_sale_six_months_l529_52977

-- Define the sales for the first five months
def sale_month1 : ℕ := 6335
def sale_month2 : ℕ := 6927
def sale_month3 : ℕ := 6855
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6562

-- Define the required sale for the sixth month
def sale_month6 : ℕ := 5091

-- Proof that the desired average sale for the six months is 6500
theorem average_sale_six_months : 
  (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) / 6 = 6500 :=
by
  sorry

end average_sale_six_months_l529_52977


namespace last_three_digits_7_pow_80_l529_52911

theorem last_three_digits_7_pow_80 : (7^80) % 1000 = 961 := by
  sorry

end last_three_digits_7_pow_80_l529_52911


namespace instantaneous_velocity_at_2_l529_52978

noncomputable def S (t : ℝ) : ℝ := 3 * t^2 - 2 * t + 1

theorem instantaneous_velocity_at_2 :
  (deriv S 2) = 10 :=
by 
  sorry

end instantaneous_velocity_at_2_l529_52978


namespace second_term_of_geo_series_l529_52916

theorem second_term_of_geo_series
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h_r : r = -1 / 3)
  (h_S : S = 25)
  (h_sum : S = a / (1 - r)) :
  (a * r) = -100 / 9 :=
by
  -- Definitions and conditions here are provided
  have hr : r = -1 / 3 := by exact h_r
  have hS : S = 25 := by exact h_S
  have hsum : S = a / (1 - r) := by exact h_sum
  -- The proof of (a * r) = -100 / 9 goes here
  sorry

end second_term_of_geo_series_l529_52916


namespace mean_of_five_numbers_l529_52904

theorem mean_of_five_numbers (S : ℚ) (n : ℕ) (h1 : S = 3/4) (h2 : n = 5) :
  (S / n) = 3/20 :=
by
  rw [h1, h2]
  sorry

end mean_of_five_numbers_l529_52904


namespace annual_income_increase_l529_52959

variable (x y : ℝ)

-- Definitions of the conditions
def regression_line (x : ℝ) : ℝ := 0.254 * x + 0.321

-- The statement we want to prove
theorem annual_income_increase (x : ℝ) : regression_line (x + 1) - regression_line x = 0.254 := 
sorry

end annual_income_increase_l529_52959


namespace existence_of_function_implies_a_le_1_l529_52994

open Real

noncomputable def positive_reals := { x : ℝ // 0 < x }

theorem existence_of_function_implies_a_le_1 (a : ℝ) :
  (∃ f : positive_reals → positive_reals, ∀ x : positive_reals, 3 * (f x).val^2 = 2 * (f (f x)).val + a * x.val^4) → a ≤ 1 :=
by
  sorry

end existence_of_function_implies_a_le_1_l529_52994


namespace dice_roll_probability_l529_52963

theorem dice_roll_probability : 
  ∃ (m n : ℕ), (1 ≤ m ∧ m ≤ 6) ∧ (1 ≤ n ∧ n ≤ 6) ∧ (m - n > 0) ∧ 
  ( (15 : ℚ) / 36 = (5 : ℚ) / 12 ) :=
by {
  sorry
}

end dice_roll_probability_l529_52963


namespace min_value_2x_minus_y_l529_52931

theorem min_value_2x_minus_y :
  ∃ (x y : ℝ), (y = abs (x - 1) ∨ y = 2) ∧ (y ≤ 2) ∧ (2 * x - y = -4) :=
by
  sorry

end min_value_2x_minus_y_l529_52931


namespace vets_recommend_yummy_dog_kibble_l529_52972

theorem vets_recommend_yummy_dog_kibble :
  (let total_vets := 1000
   let percentage_puppy_kibble := 20
   let vets_puppy_kibble := (percentage_puppy_kibble * total_vets) / 100
   let diff_yummy_puppy := 100
   let vets_yummy_kibble := vets_puppy_kibble + diff_yummy_puppy
   let percentage_yummy_kibble := (vets_yummy_kibble * 100) / total_vets
   percentage_yummy_kibble = 30) :=
by
  sorry

end vets_recommend_yummy_dog_kibble_l529_52972


namespace amount_of_water_formed_l529_52976

-- Define chemical compounds and reactions
def NaOH : Type := Unit
def HClO4 : Type := Unit
def NaClO4 : Type := Unit
def H2O : Type := Unit

-- Define the balanced chemical equation
def balanced_reaction (n_NaOH n_HClO4 : Int) : (n_NaOH = n_HClO4) → (n_NaOH = 1 → n_HClO4 = 1 → Int × Int × Int × Int) :=
  λ h_ratio h_NaOH h_HClO4 => 
    (n_NaOH, n_HClO4, 1, 1)  -- 1 mole of NaOH reacts with 1 mole of HClO4 to form 1 mole of NaClO4 and 1 mole of H2O

noncomputable def molar_mass_H2O : Float := 18.015 -- g/mol

theorem amount_of_water_formed :
  ∀ (n_NaOH n_HClO4 : Int), 
  (n_NaOH = 1 ∧ n_HClO4 = 1) →
  ((n_NaOH = n_HClO4) → molar_mass_H2O = 18.015) :=
by
  intros n_NaOH n_HClO4 h_condition h_ratio
  sorry

end amount_of_water_formed_l529_52976


namespace parallel_lines_condition_suff_not_nec_l529_52933

theorem parallel_lines_condition_suff_not_nec 
  (a : ℝ) : (a = -2) → 
  (∀ x y : ℝ, ax + 2 * y - 1 = 0) → 
  (∀ x y : ℝ, x + (a + 1) * y + 4 = 0) → 
  (∀ x1 y1 x2 y2 : ℝ, ((a = -2) → (2 * y1 - 2 * x1 = 1) → (y2 - x2 = -4) → (x1 = x2 → y1 = y2))) ∧ 
  (∃ b : ℝ, ¬ (b = -2) ∧ ((2 * y1 - b * x1 = 1) → (x2 - (b + 1) * y2 = -4) → ¬(x1 = x2 → y1 = y2)))
   :=
by
  sorry

end parallel_lines_condition_suff_not_nec_l529_52933


namespace increasing_function_range_l529_52946

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + 4*a*x else (2*a + 3)*x - 4*a + 5

theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (1/2 ≤ a ∧ a ≤ 3/2) :=
sorry

end increasing_function_range_l529_52946


namespace parabola_ordinate_l529_52982

theorem parabola_ordinate (x y : ℝ) (h : y = 2 * x^2) (d : dist (x, y) (0, 1 / 8) = 9 / 8) : y = 1 := 
sorry

end parabola_ordinate_l529_52982


namespace find_quotient_l529_52944

theorem find_quotient :
  ∃ q : ℕ, ∀ L S : ℕ, L = 1584 ∧ S = 249 ∧ (L - S = 1335) ∧ (L = S * q + 15) → q = 6 :=
by
  sorry

end find_quotient_l529_52944


namespace solve_for_y_l529_52988

variable (k y : ℝ)

-- Define the first equation for x
def eq1 (x : ℝ) : Prop := (1 / 2023) * x - 2 = 3 * x + k

-- Define the condition that x = -5 satisfies eq1
def condition1 : Prop := eq1 k (-5)

-- Define the second equation for y
def eq2 : Prop := (1 / 2023) * (2 * y + 1) - 5 = 6 * y + k

-- Prove that given condition1, y = -3 satisfies eq2
theorem solve_for_y : condition1 k → eq2 k (-3) :=
sorry

end solve_for_y_l529_52988


namespace simple_interest_rate_l529_52948

/-- 
  Given conditions:
  1. Time period T is 10 years.
  2. Simple interest SI is 7/5 of the principal amount P.
  Prove that the rate percent per annum R for which the simple interest is 7/5 of the principal amount in 10 years is 14%.
-/
theorem simple_interest_rate (P : ℝ) (T : ℝ) (SI : ℝ) (R : ℝ) (hT : T = 10) (hSI : SI = (7 / 5) * P) : 
  (SI = (P * R * T) / 100) → R = 14 := 
by 
  sorry

end simple_interest_rate_l529_52948


namespace percentage_increase_of_sides_l529_52923

noncomputable def percentage_increase_in_area (L W : ℝ) (p : ℝ) : ℝ :=
  let A : ℝ := L * W
  let L' : ℝ := L * (1 + p / 100)
  let W' : ℝ := W * (1 + p / 100)
  let A' : ℝ := L' * W'
  ((A' - A) / A) * 100

theorem percentage_increase_of_sides (L W : ℝ) (hL : L > 0) (hW : W > 0) :
    percentage_increase_in_area L W 20 = 44 :=
by
  sorry

end percentage_increase_of_sides_l529_52923


namespace mean_of_remaining_students_l529_52913

noncomputable def mean_remaining_students (k : ℕ) (h : k > 18) (mean_class : ℚ) (mean_18_students : ℚ) : ℚ :=
  (12 * k - 360) / (k - 18)

theorem mean_of_remaining_students (k : ℕ) (h : k > 18) (mean_class_eq : mean_class = 12) (mean_18_eq : mean_18_students = 20) :
  mean_remaining_students k h mean_class mean_18_students = (12 * k - 360) / (k - 18) :=
by sorry

end mean_of_remaining_students_l529_52913


namespace milk_production_l529_52995

theorem milk_production (a b c x y z w : ℕ) : 
  ((b:ℝ) / c) * w + ((y:ℝ) / z) * w = (bw / c) + (yw / z) := sorry

end milk_production_l529_52995


namespace total_marbles_left_is_correct_l529_52984

def marbles_left_after_removal : ℕ :=
  let red_initial := 80
  let blue_initial := 120
  let green_initial := 75
  let yellow_initial := 50
  let red_removed := red_initial / 4
  let blue_removed := 3 * (green_initial / 5)
  let green_removed := (green_initial * 3) / 10
  let yellow_removed := 25
  let red_left := red_initial - red_removed
  let blue_left := blue_initial - blue_removed
  let green_left := green_initial - green_removed
  let yellow_left := yellow_initial - yellow_removed
  red_left + blue_left + green_left + yellow_left

theorem total_marbles_left_is_correct :
  marbles_left_after_removal = 213 :=
  by
    sorry

end total_marbles_left_is_correct_l529_52984


namespace songs_per_album_correct_l529_52962

-- Define the number of albums and total number of songs as conditions
def number_of_albums : ℕ := 8
def total_songs : ℕ := 16

-- Define the number of songs per album
def songs_per_album (albums : ℕ) (songs : ℕ) : ℕ := songs / albums

-- The main theorem stating that the number of songs per album is 2
theorem songs_per_album_correct :
  songs_per_album number_of_albums total_songs = 2 :=
by
  unfold songs_per_album
  sorry

end songs_per_album_correct_l529_52962


namespace isabel_remaining_pages_l529_52909

def total_problems : ℕ := 72
def finished_problems : ℕ := 32
def problems_per_page : ℕ := 8

theorem isabel_remaining_pages :
  (total_problems - finished_problems) / problems_per_page = 5 := 
sorry

end isabel_remaining_pages_l529_52909


namespace xy_system_solution_l529_52970

theorem xy_system_solution (x y : ℝ) (h₁ : x + 5 * y = 6) (h₂ : 3 * x - y = 2) : x + y = 2 := 
by 
  sorry

end xy_system_solution_l529_52970


namespace min_travel_time_l529_52920

/-- Two people, who have one bicycle, need to travel from point A to point B, which is 40 km away from point A. 
The first person walks at a speed of 4 km/h and rides the bicycle at 30 km/h, 
while the second person walks at a speed of 6 km/h and rides the bicycle at 20 km/h. 
Prove that the minimum time in which they can both get to point B is 25/9 hours. -/
theorem min_travel_time (d : ℕ) (v_w1 v_c1 v_w2 v_c2 : ℕ) (min_time : ℚ) 
  (h_d : d = 40)
  (h_v1_w : v_w1 = 4)
  (h_v1_c : v_c1 = 30)
  (h_v2_w : v_w2 = 6)
  (h_v2_c : v_c2 = 20)
  (h_min_time : min_time = 25 / 9) :
  ∃ y x : ℚ, 4*y + (2/3)*y*30 = 40 ∧ min_time = y + (2/3)*y :=
sorry

end min_travel_time_l529_52920


namespace abs_neg_three_halves_l529_52980

theorem abs_neg_three_halves : |(-3 : ℚ) / 2| = 3 / 2 := 
sorry

end abs_neg_three_halves_l529_52980


namespace total_minutes_ironing_over_4_weeks_l529_52925

/-- Define the time spent ironing each day -/
def minutes_ironing_per_day : Nat := 5 + 3

/-- Define the number of days Hayden irons per week -/
def days_ironing_per_week : Nat := 5

/-- Define the number of weeks considered -/
def number_of_weeks : Nat := 4

/-- The main theorem we're proving is that Hayden spends 160 minutes ironing over 4 weeks -/
theorem total_minutes_ironing_over_4_weeks :
  (minutes_ironing_per_day * days_ironing_per_week * number_of_weeks) = 160 := by
  sorry

end total_minutes_ironing_over_4_weeks_l529_52925


namespace break_room_capacity_l529_52966

theorem break_room_capacity :
  let people_per_table := 8
  let number_of_tables := 4
  people_per_table * number_of_tables = 32 :=
by
  let people_per_table := 8
  let number_of_tables := 4
  have h : people_per_table * number_of_tables = 32 := by sorry
  exact h

end break_room_capacity_l529_52966


namespace speed_of_man_in_still_water_l529_52953

-- Definition of the conditions
def effective_downstream_speed (v_m v_c : ℝ) : Prop := (v_m + v_c) = 10
def effective_upstream_speed (v_m v_c : ℝ) : Prop := (v_m - v_c) = 11.25

-- The proof problem statement
theorem speed_of_man_in_still_water (v_m v_c : ℝ) 
  (h1 : effective_downstream_speed v_m v_c)
  (h2 : effective_upstream_speed v_m v_c)
  : v_m = 10.625 :=
sorry

end speed_of_man_in_still_water_l529_52953


namespace no_positive_integer_solution_l529_52996

theorem no_positive_integer_solution (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  ¬ (∃ (k : ℕ), (xy + 1) * (xy + x + 2) = k^2) :=
by {
  sorry
}

end no_positive_integer_solution_l529_52996


namespace min_pictures_needed_l529_52941

theorem min_pictures_needed (n m : ℕ) (participants : Fin n → Fin m → Prop)
  (h1 : n = 60) (h2 : m ≤ 30)
  (h3 : ∀ (i j : Fin n), ∃ (k : Fin m), participants i k ∧ participants j k) :
  m = 6 :=
sorry

end min_pictures_needed_l529_52941


namespace cost_of_55_lilies_l529_52956

-- Define the problem conditions
def price_per_dozen_lilies (p : ℝ) : Prop :=
  p * 24 = 30

def directly_proportional_price (p : ℝ) (n : ℕ) : ℝ :=
  p * n

-- State the problem to prove the cost of a 55 lily bouquet
theorem cost_of_55_lilies (p : ℝ) (c : ℝ) :
  price_per_dozen_lilies p →
  c = directly_proportional_price p 55 →
  c = 68.75 :=
by
  sorry

end cost_of_55_lilies_l529_52956


namespace quadratic_eq_mn_sum_l529_52907

theorem quadratic_eq_mn_sum (m n : ℤ) 
  (h1 : m - 1 = 2) 
  (h2 : 16 + 4 * n = 0) 
  : m + n = -1 :=
by
  sorry

end quadratic_eq_mn_sum_l529_52907


namespace degree_sum_interior_angles_of_star_l529_52999

-- Definitions based on conditions provided.
def extended_polygon_star (n : Nat) (h : n ≥ 6) : Nat := 
  180 * (n - 2)

-- Theorem to prove the degree-sum of the interior angles.
theorem degree_sum_interior_angles_of_star (n : Nat) (h : n ≥ 6) : 
  extended_polygon_star n h = 180 * (n - 2) :=
by
  sorry

end degree_sum_interior_angles_of_star_l529_52999


namespace heating_time_correct_l529_52902

structure HeatingProblem where
  initial_temp : ℕ
  final_temp : ℕ
  heating_rate : ℕ

def time_to_heat (hp : HeatingProblem) : ℕ :=
  (hp.final_temp - hp.initial_temp) / hp.heating_rate

theorem heating_time_correct (hp : HeatingProblem) (h1 : hp.initial_temp = 20) (h2 : hp.final_temp = 100) (h3 : hp.heating_rate = 5) :
  time_to_heat hp = 16 :=
by
  sorry

end heating_time_correct_l529_52902


namespace balloon_arrangements_correct_l529_52912

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

-- Define the number of ways to arrange "BALLOON"
noncomputable def arrangements_balloon : ℕ := factorial 7 / (factorial 2 * factorial 2)

-- State the theorem
theorem balloon_arrangements_correct : arrangements_balloon = 1260 := by sorry

end balloon_arrangements_correct_l529_52912


namespace units_digit_base8_of_sum_34_8_47_8_l529_52983

def is_units_digit (n m : ℕ) (u : ℕ) := (n + m) % 8 = u

theorem units_digit_base8_of_sum_34_8_47_8 :
  ∀ (n m : ℕ), n = 34 ∧ m = 47 → (is_units_digit (n % 8) (m % 8) 3) :=
by
  intros n m h
  rw [h.1, h.2]
  sorry

end units_digit_base8_of_sum_34_8_47_8_l529_52983


namespace length_of_escalator_l529_52918

-- Define the conditions
def escalator_speed : ℝ := 15 -- ft/sec
def person_speed : ℝ := 5 -- ft/sec
def time_taken : ℝ := 10 -- sec

-- Define the length of the escalator
def escalator_length (escalator_speed : ℝ) (person_speed : ℝ) (time : ℝ) : ℝ := 
  (escalator_speed + person_speed) * time

-- Theorem to prove
theorem length_of_escalator : escalator_length escalator_speed person_speed time_taken = 200 := by
  sorry

end length_of_escalator_l529_52918


namespace intersection_eq_l529_52958

variable (A : Set ℤ) (B : Set ℤ)

def A_def := A = {-1, 0, 1, 2}
def B_def := B = {x | -1 < x ∧ x < 2}

theorem intersection_eq : A ∩ B = {0, 1} :=
by
  have A_def : A = {-1, 0, 1, 2} := sorry
  have B_def : B = {x | -1 < x ∧ x < 2} := sorry
  sorry

end intersection_eq_l529_52958


namespace find_value_of_expression_l529_52914

theorem find_value_of_expression (a b : ℝ) (h : a + 2 * b - 1 = 0) : 3 * a + 6 * b = 3 :=
by
  sorry

end find_value_of_expression_l529_52914


namespace range_of_a_l529_52929

noncomputable def problem_statement : Prop :=
  ∃ x : ℝ, (1 ≤ x) ∧ (∀ a : ℝ, (1 + 1 / x) ^ (x + a) ≥ Real.exp 1 → a ≥ 1 / Real.log 2 - 1)

theorem range_of_a : problem_statement :=
sorry

end range_of_a_l529_52929


namespace units_digit_of_n_l529_52937

theorem units_digit_of_n (n : ℕ) (h : n = 56^78 + 87^65) : (n % 10) = 3 :=
by
  sorry

end units_digit_of_n_l529_52937


namespace car_speed_l529_52992

theorem car_speed (time : ℕ) (distance : ℕ) (h1 : time = 5) (h2 : distance = 300) : distance / time = 60 := by
  sorry

end car_speed_l529_52992


namespace salt_solution_mixture_l529_52924

theorem salt_solution_mixture (x : ℝ) :  
  (0.80 * x + 0.35 * 150 = 0.55 * (150 + x)) → x = 120 :=
by 
  sorry

end salt_solution_mixture_l529_52924


namespace length_segment_midpoints_diagonals_trapezoid_l529_52932

theorem length_segment_midpoints_diagonals_trapezoid
  (a b c d : ℝ)
  (h_side_lengths : (2 = a ∨ 2 = b ∨ 2 = c ∨ 2 = d) ∧ 
                    (10 = a ∨ 10 = b ∨ 10 = c ∨ 10 = d) ∧ 
                    (10 = a ∨ 10 = b ∨ 10 = c ∨ 10 = d) ∧ 
                    (20 = a ∨ 20 = b ∨ 20 = c ∨ 20 = d))
  (h_parallel_sides : (a = 20 ∧ b = 2) ∨ (a = 2 ∧ b = 20)) :
  (1/2) * |a - b| = 9 :=
by
  sorry

end length_segment_midpoints_diagonals_trapezoid_l529_52932


namespace cargo_arrival_in_days_l529_52951

-- Definitions for conditions
def days_navigate : ℕ := 21
def days_customs : ℕ := 4
def days_transport : ℕ := 7
def days_departed : ℕ := 30

-- Calculate the days since arrival in Vancouver
def days_arrival_vancouver : ℕ := days_departed - days_navigate

-- Calculate the days since customs processes finished
def days_since_customs_done : ℕ := days_arrival_vancouver - days_customs

-- Calculate the days for cargo to arrive at the warehouse from today
def days_until_arrival : ℕ := days_transport - days_since_customs_done

-- Expected number of days from today for the cargo to arrive at the warehouse
theorem cargo_arrival_in_days : days_until_arrival = 2 := by
  -- Insert the proof steps here
  sorry

end cargo_arrival_in_days_l529_52951


namespace cornelia_european_countries_l529_52908

def total_countries : Nat := 42
def south_american_countries : Nat := 10
def asian_countries : Nat := 6

def non_european_countries : Nat :=
  south_american_countries + 2 * asian_countries

def european_countries : Nat :=
  total_countries - non_european_countries

theorem cornelia_european_countries :
  european_countries = 20 := by
  sorry

end cornelia_european_countries_l529_52908


namespace father_has_4_chocolate_bars_left_l529_52928

noncomputable def chocolate_bars_given_to_father (initial_bars : ℕ) (num_people : ℕ) : ℕ :=
  let bars_per_person := initial_bars / num_people
  let bars_given := num_people * (bars_per_person / 2)
  bars_given

noncomputable def chocolate_bars_left_with_father (bars_given : ℕ) (bars_given_away : ℕ) : ℕ :=
  bars_given - bars_given_away

theorem father_has_4_chocolate_bars_left :
  ∀ (initial_bars num_people bars_given_away : ℕ), 
  initial_bars = 40 →
  num_people = 7 →
  bars_given_away = 10 →
  chocolate_bars_left_with_father (chocolate_bars_given_to_father initial_bars num_people) bars_given_away = 4 :=
by
  intros initial_bars num_people bars_given_away h_initial h_num h_given_away
  unfold chocolate_bars_given_to_father chocolate_bars_left_with_father
  rw [h_initial, h_num, h_given_away]
  exact sorry

end father_has_4_chocolate_bars_left_l529_52928


namespace sum_of_thetas_l529_52901

noncomputable def theta (k : ℕ) : ℝ := (54 + 72 * k) % 360

theorem sum_of_thetas : (theta 0 + theta 1 + theta 2 + theta 3 + theta 4) = 990 :=
by
  -- proof goes here
  sorry

end sum_of_thetas_l529_52901


namespace find_divisor_l529_52998

theorem find_divisor (D N : ℕ) (k l : ℤ)
  (h1 : N % D = 255)
  (h2 : (2 * N) % D = 112) :
  D = 398 := by
  -- Proof here
  sorry

end find_divisor_l529_52998


namespace inequality_a_inequality_b_l529_52973

theorem inequality_a (R_A R_B R_C R_D d_A d_B d_C d_D : ℝ) :
  (R_A + R_B + R_C + R_D) * (1 / d_A + 1 / d_B + 1 / d_C + 1 / d_D) ≥ 48 :=
sorry

theorem inequality_b (R_A R_B R_C R_D d_A d_B d_C d_D : ℝ) :
  (R_A^2 + R_B^2 + R_C^2 + R_D^2) * (1 / d_A^2 + 1 / d_B^2 + 1 / d_C^2 + 1 / d_D^2) ≥ 144 :=
sorry

end inequality_a_inequality_b_l529_52973


namespace certain_number_l529_52986

theorem certain_number (x : ℝ) (h : x - 4 = 2) : x^2 - 3 * x = 18 :=
by
  -- Proof yet to be completed
  sorry

end certain_number_l529_52986


namespace number_of_pairs_l529_52927

theorem number_of_pairs : 
  (∀ n m : ℕ, (1 ≤ m ∧ m ≤ 2012) → (5^n < 2^m ∧ 2^m < 2^(m+2) ∧ 2^(m+2) < 5^(n+1))) → 
  (∃ c, c = 279) :=
by
  sorry

end number_of_pairs_l529_52927


namespace max_length_OB_l529_52961

-- Define the problem conditions
def angle_AOB : ℝ := 45
def length_AB : ℝ := 2
def max_sin_angle_OAB : ℝ := 1

-- Claim to be proven
theorem max_length_OB : ∃ OB_max, OB_max = 2 * Real.sqrt 2 :=
by
  sorry

end max_length_OB_l529_52961


namespace double_acute_angle_l529_52987

theorem double_acute_angle (θ : ℝ) (h : 0 < θ ∧ θ < 90) : 0 < 2 * θ ∧ 2 * θ < 180 :=
by
  sorry

end double_acute_angle_l529_52987


namespace count_terms_expansion_l529_52926

/-
This function verifies that the number of distinct terms in the expansion
of (a + b + c)(a + d + e + f + g) is equal to 15.
-/

theorem count_terms_expansion : 
    (a b c d e f g : ℕ) → 
    3 * 5 = 15 :=
by 
    intros a b c d e f g
    sorry

end count_terms_expansion_l529_52926


namespace value_of_a_l529_52938

theorem value_of_a (a x : ℝ) (h1 : x = 2) (h2 : a * x = 4) : a = 2 :=
by
  sorry

end value_of_a_l529_52938


namespace probability_third_winning_l529_52903

-- Definitions based on the conditions provided
def num_tickets : ℕ := 10
def num_winning_tickets : ℕ := 3
def num_non_winning_tickets : ℕ := num_tickets - num_winning_tickets

-- Define the probability function
def probability_of_third_draw_winning : ℚ :=
  (num_non_winning_tickets / num_tickets) * 
  ((num_non_winning_tickets - 1) / (num_tickets - 1)) * 
  (num_winning_tickets / (num_tickets - 2))

-- The theorem to prove
theorem probability_third_winning : probability_of_third_draw_winning = 7 / 40 :=
  by sorry

end probability_third_winning_l529_52903


namespace jimmys_speed_l529_52989

theorem jimmys_speed 
(Mary_speed : ℕ) (total_distance : ℕ) (t : ℕ)
(h1 : Mary_speed = 5)
(h2 : total_distance = 9)
(h3 : t = 1)
: ∃ (Jimmy_speed : ℕ), Jimmy_speed = 4 :=
by
  -- calculation steps skipped here
  sorry

end jimmys_speed_l529_52989


namespace min_total_bananas_l529_52971

noncomputable def total_bananas_condition (b1 b2 b3 : ℕ) : Prop :=
  let m1 := (5/8 : ℚ) * b1 + (5/16 : ℚ) * b2 + (23/48 : ℚ) * b3
  let m2 := (3/16 : ℚ) * b1 + (3/8 : ℚ) * b2 + (23/48 : ℚ) * b3
  let m3 := (3/16 : ℚ) * b1 + (5/16 : ℚ) * b2 + (1/24 : ℚ) * b3
  (((m1 : ℚ) * 4) = ((m2 : ℚ) * 3)) ∧ (((m1 : ℚ) * 4) = ((m3 : ℚ) * 2))

theorem min_total_bananas : ∃ (b1 b2 b3 : ℕ), b1 + b2 + b3 = 192 ∧ total_bananas_condition b1 b2 b3 :=
sorry

end min_total_bananas_l529_52971


namespace greatest_three_digit_multiple_of_23_l529_52900

def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def is_multiple_of_23 (n : ℕ) : Prop :=
  n % 23 = 0

theorem greatest_three_digit_multiple_of_23 :
  ∀ n, is_three_digit n ∧ is_multiple_of_23 n → n ≤ 989 :=
by
  sorry

end greatest_three_digit_multiple_of_23_l529_52900


namespace find_b_l529_52954

theorem find_b (b : ℝ) :
  (∃ x1 x2 : ℝ, (x1 + x2 = -2) ∧
    ((x1 + 1)^3 + x1 / (x1 + 1) = -x1 + b) ∧
    ((x2 + 1)^3 + x2 / (x2 + 1) = -x2 + b)) →
  b = 0 :=
by
  sorry

end find_b_l529_52954


namespace solution_set_of_f_gt_7_minimum_value_of_m_n_l529_52991

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

theorem solution_set_of_f_gt_7 :
  { x : ℝ | f x > 7 } = { x | x > 4 ∨ x < -3 } :=
by
  ext x
  sorry

theorem minimum_value_of_m_n (m n : ℝ) (h : 0 < m ∧ 0 < n) (hfmin : ∀ x : ℝ, f x ≥ m + n) :
  m = n ∧ m = 3 / 2 ∧ m^2 + n^2 = 9 / 2 :=
by
  sorry

end solution_set_of_f_gt_7_minimum_value_of_m_n_l529_52991


namespace pens_per_student_l529_52945

theorem pens_per_student (n : ℕ) (h1 : 0 < n) (h2 : n ≤ 50) (h3 : 100 % n = 0) (h4 : 50 % n = 0) : 100 / n = 2 :=
by
  -- proof goes here
  sorry

end pens_per_student_l529_52945


namespace smallest_four_digit_multiple_of_18_l529_52936

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ (∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0) → n ≤ m) ∧ n = 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l529_52936


namespace weight_of_replaced_person_l529_52968

-- Define the conditions in Lean 4
variables {w_replaced : ℝ}   -- Weight of the person who was replaced
variables {w_new : ℝ}        -- Weight of the new person
variables {n : ℕ}            -- Number of persons
variables {avg_increase : ℝ} -- Increase in average weight

-- Set up the given conditions
axiom h1 : n = 8
axiom h2 : avg_increase = 2.5
axiom h3 : w_new = 40

-- Theorem that states the weight of the replaced person
theorem weight_of_replaced_person : w_replaced = 20 :=
by
  sorry

end weight_of_replaced_person_l529_52968


namespace total_cats_l529_52943

variable (initialCats : ℝ)
variable (boughtCats : ℝ)

theorem total_cats (h1 : initialCats = 11.0) (h2 : boughtCats = 43.0) :
    initialCats + boughtCats = 54.0 :=
by
  sorry

end total_cats_l529_52943


namespace theater_ticket_sales_l529_52905

theorem theater_ticket_sales (O B : ℕ) 
  (h1 : O + B = 370) 
  (h2 : 12 * O + 8 * B = 3320) : 
  B - O = 190 := 
sorry

end theater_ticket_sales_l529_52905


namespace num_of_consecutive_sets_sum_18_eq_2_l529_52939

theorem num_of_consecutive_sets_sum_18_eq_2 : 
  ∃ (sets : Finset (Finset ℕ)), 
    (∀ s ∈ sets, (∃ n a, n ≥ 3 ∧ (s = Finset.range (a + n - 1) \ Finset.range (a - 1)) ∧ 
    s.sum id = 18)) ∧ 
    sets.card = 2 := 
sorry

end num_of_consecutive_sets_sum_18_eq_2_l529_52939


namespace find_v4_l529_52921

noncomputable def horner_method (x : ℤ) : ℤ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  let v3 := v2 * x + 20
  let v4 := v3 * x - 8
  v4

theorem find_v4 : horner_method (-2) = -16 :=
  by {
    -- Proof goes here, but we are only required to write the statement.
    sorry
  }

end find_v4_l529_52921


namespace octahedron_parallel_edge_pairs_count_l529_52919

-- defining a regular octahedron structure
structure RegularOctahedron where
  vertices : Fin 8
  edges : Fin 12
  faces : Fin 8

noncomputable def numberOfStrictlyParallelEdgePairs (O : RegularOctahedron) : Nat :=
  12 -- Given the symmetry and structure.

theorem octahedron_parallel_edge_pairs_count (O : RegularOctahedron) : 
  numberOfStrictlyParallelEdgePairs O = 12 :=
by
  sorry

end octahedron_parallel_edge_pairs_count_l529_52919


namespace three_hundred_percent_of_x_equals_seventy_five_percent_of_y_l529_52940

theorem three_hundred_percent_of_x_equals_seventy_five_percent_of_y
  (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 20) : y = 80 := by
  sorry

end three_hundred_percent_of_x_equals_seventy_five_percent_of_y_l529_52940


namespace smallest_int_x_l529_52922

theorem smallest_int_x (x : ℤ) (h : 2 * x + 5 < 3 * x - 10) : x = 16 :=
sorry

end smallest_int_x_l529_52922


namespace rectangle_area_l529_52985

theorem rectangle_area (P : ℕ) (a : ℕ) (b : ℕ) (h₁ : P = 2 * (a + b)) (h₂ : P = 40) (h₃ : a = 5) : a * b = 75 :=
by
  sorry

end rectangle_area_l529_52985


namespace solve_equation_l529_52955

theorem solve_equation (x : ℝ) (h₀ : x = 46) :
  ( (8 / (Real.sqrt (x - 10) - 10)) + 
    (2 / (Real.sqrt (x - 10) - 5)) + 
    (9 / (Real.sqrt (x - 10) + 5)) + 
    (15 / (Real.sqrt (x - 10) + 10)) = 0) := 
by 
  sorry

end solve_equation_l529_52955


namespace bianca_total_drawing_time_l529_52935

def total_drawing_time (a b : ℕ) : ℕ := a + b

theorem bianca_total_drawing_time :
  let a := 22
  let b := 19
  total_drawing_time a b = 41 :=
by
  sorry

end bianca_total_drawing_time_l529_52935


namespace liquor_and_beer_cost_l529_52990

-- Define the variables and conditions
variables (p_liquor p_beer : ℕ)

-- Main theorem to prove
theorem liquor_and_beer_cost (h1 : 2 * p_liquor + 12 * p_beer = 56)
                             (h2 : p_liquor = 8 * p_beer) :
  p_liquor + p_beer = 18 :=
sorry

end liquor_and_beer_cost_l529_52990


namespace sin_sum_leq_3div2_sqrt3_l529_52930

theorem sin_sum_leq_3div2_sqrt3 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π) :
  Real.sin A + Real.sin B + Real.sin C ≤ (3 / 2) * Real.sqrt 3 :=
by
  sorry

end sin_sum_leq_3div2_sqrt3_l529_52930


namespace no_integer_solutions_system_l529_52975

theorem no_integer_solutions_system :
  ¬(∃ x y z : ℤ, 
    x^6 + x^3 + x^3 * y + y = 147^157 ∧ 
    x^3 + x^3 * y + y^2 + y + z^9 = 157^147) :=
  sorry

end no_integer_solutions_system_l529_52975


namespace CitadelSchoolEarnings_l529_52967

theorem CitadelSchoolEarnings :
  let apex_students : Nat := 9
  let apex_days : Nat := 5
  let beacon_students : Nat := 3
  let beacon_days : Nat := 4
  let citadel_students : Nat := 6
  let citadel_days : Nat := 7
  let total_payment : ℕ := 864
  let total_student_days : ℕ := (apex_students * apex_days) + (beacon_students * beacon_days) + (citadel_students * citadel_days)
  let daily_wage_per_student : ℚ := total_payment / total_student_days
  let citadel_student_days : ℕ := citadel_students * citadel_days
  let citadel_earnings : ℚ := daily_wage_per_student * citadel_student_days
  citadel_earnings = 366.55 := by
  sorry

end CitadelSchoolEarnings_l529_52967


namespace solve_equation_l529_52965

theorem solve_equation (x : ℝ) :
  (4 * x + 1) * (3 * x + 1) * (2 * x + 1) * (x + 1) = 3 * x ^ 4  →
  x = (-5 + Real.sqrt 13) / 6 ∨ x = (-5 - Real.sqrt 13) / 6 :=
by
  sorry

end solve_equation_l529_52965


namespace solve_for_x_l529_52917

noncomputable def valid_x (x : ℝ) : Prop :=
  let l := 4 * x
  let w := 2 * x + 6
  l * w = 2 * (l + w)

theorem solve_for_x : 
  ∃ (x : ℝ), valid_x x ↔ x = (-3 + Real.sqrt 33) / 4 :=
by
  sorry

end solve_for_x_l529_52917


namespace average_score_l529_52910

-- Definitions from conditions
def June_score := 97
def Patty_score := 85
def Josh_score := 100
def Henry_score := 94
def total_children := 4
def total_score := June_score + Patty_score + Josh_score + Henry_score

-- Prove the average score
theorem average_score : (total_score / total_children) = 94 :=
by
  sorry

end average_score_l529_52910


namespace solve_for_x_l529_52906

theorem solve_for_x : 
  (∀ (x y : ℝ), y = 1 / (4 * x + 2) → y = 2 → x = -3 / 8) :=
by
  intro x y
  intro h₁ h₂
  rw [h₂] at h₁
  sorry

end solve_for_x_l529_52906


namespace ratio_of_green_to_blue_l529_52993

def balls (total blue red green yellow : ℕ) : Prop :=
  total = 36 ∧ blue = 6 ∧ red = 4 ∧ yellow = 2 * red ∧ green = total - (blue + red + yellow)

theorem ratio_of_green_to_blue (total blue red green yellow : ℕ) (h : balls total blue red green yellow) :
  (green / blue = 3) :=
by
  -- Unpack the conditions
  obtain ⟨total_eq, blue_eq, red_eq, yellow_eq, green_eq⟩ := h
  -- Simplify values based on the given conditions
  have blue_val := blue_eq
  have green_val := green_eq
  rw [blue_val, green_val]
  sorry

end ratio_of_green_to_blue_l529_52993


namespace second_company_managers_percent_l529_52942

/-- A company's workforce consists of 10 percent managers and 90 percent software engineers.
    Another company's workforce consists of some percent managers, 10 percent software engineers, 
    and 60 percent support staff. The two companies merge, and the resulting company's 
    workforce consists of 25 percent managers. If 25 percent of the workforce originated from the 
    first company, what percent of the second company's workforce were managers? -/
theorem second_company_managers_percent
  (F S : ℝ)
  (h1 : 0.10 * F + m * S = 0.25 * (F + S))
  (h2 : F = 0.25 * (F + S)) :
  m = 0.225 :=
sorry

end second_company_managers_percent_l529_52942


namespace smallest_possible_n_l529_52981

theorem smallest_possible_n : ∃ (n : ℕ), (∀ (r g b : ℕ), 24 * n = 18 * r ∧ 24 * n = 16 * g ∧ 24 * n = 20 * b) ∧ n = 30 :=
by
  -- Sorry, we're skipping the proof, as specified.
  sorry

end smallest_possible_n_l529_52981


namespace tom_paid_450_l529_52964

-- Define the conditions
def hours_per_day : ℕ := 2
def number_of_days : ℕ := 3
def cost_per_hour : ℕ := 75

-- Calculated total number of hours Tom rented the helicopter
def total_hours_rented : ℕ := hours_per_day * number_of_days

-- Calculated total cost for renting the helicopter
def total_cost_rented : ℕ := total_hours_rented * cost_per_hour

-- Theorem stating that Tom paid $450 to rent the helicopter
theorem tom_paid_450 : total_cost_rented = 450 := by
  sorry

end tom_paid_450_l529_52964


namespace find_number_l529_52979

theorem find_number (x : ℕ) (h : x - 18 = 3 * (86 - x)) : x = 69 :=
by
  sorry

end find_number_l529_52979
