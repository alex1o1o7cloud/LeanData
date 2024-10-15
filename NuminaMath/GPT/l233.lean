import Mathlib

namespace NUMINAMATH_GPT_tom_paid_450_l233_23353

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

end NUMINAMATH_GPT_tom_paid_450_l233_23353


namespace NUMINAMATH_GPT_abs_neg_three_halves_l233_23393

theorem abs_neg_three_halves : |(-3 : ℚ) / 2| = 3 / 2 := 
sorry

end NUMINAMATH_GPT_abs_neg_three_halves_l233_23393


namespace NUMINAMATH_GPT_vector_dot_product_proof_l233_23359

def vector_dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem vector_dot_product_proof : 
  let a := (-1, 2)
  let b := (2, 3)
  vector_dot_product a (a.1 - b.1, a.2 - b.2) = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_vector_dot_product_proof_l233_23359


namespace NUMINAMATH_GPT_length_of_escalator_l233_23339

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

end NUMINAMATH_GPT_length_of_escalator_l233_23339


namespace NUMINAMATH_GPT_unit_digit_product_l233_23329

theorem unit_digit_product (n1 n2 n3 : ℕ) (a : ℕ) (b : ℕ) (c : ℕ) :
  (n1 = 68) ∧ (n2 = 59) ∧ (n3 = 71) ∧ (a = 3) ∧ (b = 6) ∧ (c = 7) →
  (a ^ n1 * b ^ n2 * c ^ n3) % 10 = 8 := by
  sorry

end NUMINAMATH_GPT_unit_digit_product_l233_23329


namespace NUMINAMATH_GPT_min_travel_time_l233_23349

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

end NUMINAMATH_GPT_min_travel_time_l233_23349


namespace NUMINAMATH_GPT_number_of_rolls_in_case_l233_23381

-- Definitions: Cost of a case, cost per roll individually, percent savings per roll
def cost_of_case : ℝ := 9
def cost_per_roll_individual : ℝ := 1
def percent_savings_per_roll : ℝ := 0.25

-- Theorem: Proving the number of rolls in the case is 12
theorem number_of_rolls_in_case (n : ℕ) (h1 : cost_of_case = 9)
    (h2 : cost_per_roll_individual = 1)
    (h3 : percent_savings_per_roll = 0.25) : n = 12 := 
  sorry

end NUMINAMATH_GPT_number_of_rolls_in_case_l233_23381


namespace NUMINAMATH_GPT_bianca_total_drawing_time_l233_23332

def total_drawing_time (a b : ℕ) : ℕ := a + b

theorem bianca_total_drawing_time :
  let a := 22
  let b := 19
  total_drawing_time a b = 41 :=
by
  sorry

end NUMINAMATH_GPT_bianca_total_drawing_time_l233_23332


namespace NUMINAMATH_GPT_farmer_total_acres_l233_23322

-- Define the problem conditions and the total acres
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) : 5 * x + 2 * x + 4 * x = 1034 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_farmer_total_acres_l233_23322


namespace NUMINAMATH_GPT_sphere_in_cone_volume_l233_23326

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem sphere_in_cone_volume :
  let d := 12
  let θ := 45
  let r := 3 * Real.sqrt 2
  let V := volume_of_sphere r
  d = 12 → θ = 45 → V = 72 * Real.sqrt 2 * Real.pi := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sphere_in_cone_volume_l233_23326


namespace NUMINAMATH_GPT_amount_of_water_formed_l233_23395

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

end NUMINAMATH_GPT_amount_of_water_formed_l233_23395


namespace NUMINAMATH_GPT_find_value_of_expression_l233_23308

theorem find_value_of_expression :
  3 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2400 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l233_23308


namespace NUMINAMATH_GPT_faster_train_cross_time_l233_23300

/-- Statement of the problem in Lean 4 -/
theorem faster_train_cross_time :
  let speed_faster_train_kmph := 72
  let speed_slower_train_kmph := 36
  let length_faster_train_m := 180
  let relative_speed_kmph := speed_faster_train_kmph - speed_slower_train_kmph
  let relative_speed_mps := relative_speed_kmph * (5 / 18 : ℝ)
  let time_taken := length_faster_train_m / relative_speed_mps
  time_taken = 18 :=
by
  sorry

end NUMINAMATH_GPT_faster_train_cross_time_l233_23300


namespace NUMINAMATH_GPT_max_value_of_f_l233_23346

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x + Real.sin x

theorem max_value_of_f : ∃ M, ∀ x, f x ≤ M ∧ (∃ y, f y = M) := by
  use Real.sqrt 5
  sorry

end NUMINAMATH_GPT_max_value_of_f_l233_23346


namespace NUMINAMATH_GPT_consumer_installment_credit_l233_23310

theorem consumer_installment_credit : 
  ∃ C : ℝ, 
    (0.43 * C = 200) ∧ 
    (C = 465.116) :=
by
  sorry

end NUMINAMATH_GPT_consumer_installment_credit_l233_23310


namespace NUMINAMATH_GPT_songs_per_album_correct_l233_23387

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

end NUMINAMATH_GPT_songs_per_album_correct_l233_23387


namespace NUMINAMATH_GPT_dice_roll_probability_l233_23336

theorem dice_roll_probability : 
  ∃ (m n : ℕ), (1 ≤ m ∧ m ≤ 6) ∧ (1 ≤ n ∧ n ≤ 6) ∧ (m - n > 0) ∧ 
  ( (15 : ℚ) / 36 = (5 : ℚ) / 12 ) :=
by {
  sorry
}

end NUMINAMATH_GPT_dice_roll_probability_l233_23336


namespace NUMINAMATH_GPT_percentage_increase_of_sides_l233_23344

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

end NUMINAMATH_GPT_percentage_increase_of_sides_l233_23344


namespace NUMINAMATH_GPT_tax_free_amount_is_600_l233_23307

variable (X : ℝ) -- X is the tax-free amount

-- Given conditions
variable (total_value : ℝ := 1720)
variable (tax_paid : ℝ := 89.6)
variable (tax_rate : ℝ := 0.08)

-- Proof problem
theorem tax_free_amount_is_600
  (h1 : 0.08 * (total_value - X) = tax_paid) :
  X = 600 :=
by
  sorry

end NUMINAMATH_GPT_tax_free_amount_is_600_l233_23307


namespace NUMINAMATH_GPT_soccer_game_goals_l233_23302

theorem soccer_game_goals (A1_first_half A2_first_half B1_first_half B2_first_half : ℕ) 
  (h1 : A1_first_half = 8)
  (h2 : B1_first_half = A1_first_half / 2)
  (h3 : B2_first_half = A1_first_half)
  (h4 : A2_first_half = B2_first_half - 2) : 
  A1_first_half + A2_first_half + B1_first_half + B2_first_half = 26 :=
by
  -- The proof is not needed, so we use sorry to skip it.
  sorry

end NUMINAMATH_GPT_soccer_game_goals_l233_23302


namespace NUMINAMATH_GPT_rowing_time_75_minutes_l233_23320

-- Definition of time duration Ethan rowed.
def EthanRowingTime : ℕ := 25  -- minutes

-- Definition of the time duration Frank rowed.
def FrankRowingTime : ℕ := 2 * EthanRowingTime  -- twice as long as Ethan.

-- Definition of the total rowing time.
def TotalRowingTime : ℕ := EthanRowingTime + FrankRowingTime

-- Theorem statement proving the total rowing time is 75 minutes.
theorem rowing_time_75_minutes : TotalRowingTime = 75 := by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_rowing_time_75_minutes_l233_23320


namespace NUMINAMATH_GPT_parallel_lines_condition_suff_not_nec_l233_23377

theorem parallel_lines_condition_suff_not_nec 
  (a : ℝ) : (a = -2) → 
  (∀ x y : ℝ, ax + 2 * y - 1 = 0) → 
  (∀ x y : ℝ, x + (a + 1) * y + 4 = 0) → 
  (∀ x1 y1 x2 y2 : ℝ, ((a = -2) → (2 * y1 - 2 * x1 = 1) → (y2 - x2 = -4) → (x1 = x2 → y1 = y2))) ∧ 
  (∃ b : ℝ, ¬ (b = -2) ∧ ((2 * y1 - b * x1 = 1) → (x2 - (b + 1) * y2 = -4) → ¬(x1 = x2 → y1 = y2)))
   :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_condition_suff_not_nec_l233_23377


namespace NUMINAMATH_GPT_increasing_function_range_l233_23335

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + 4*a*x else (2*a + 3)*x - 4*a + 5

theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (1/2 ≤ a ∧ a ≤ 3/2) :=
sorry

end NUMINAMATH_GPT_increasing_function_range_l233_23335


namespace NUMINAMATH_GPT_find_starting_number_l233_23315

theorem find_starting_number (S : ℤ) (n : ℤ) (sum_eq : 10 = S) (consec_eq : S = (20 / 2) * (n + (n + 19))) : 
  n = -9 := 
by
  sorry

end NUMINAMATH_GPT_find_starting_number_l233_23315


namespace NUMINAMATH_GPT_mean_of_remaining_students_l233_23371

noncomputable def mean_remaining_students (k : ℕ) (h : k > 18) (mean_class : ℚ) (mean_18_students : ℚ) : ℚ :=
  (12 * k - 360) / (k - 18)

theorem mean_of_remaining_students (k : ℕ) (h : k > 18) (mean_class_eq : mean_class = 12) (mean_18_eq : mean_18_students = 20) :
  mean_remaining_students k h mean_class mean_18_students = (12 * k - 360) / (k - 18) :=
by sorry

end NUMINAMATH_GPT_mean_of_remaining_students_l233_23371


namespace NUMINAMATH_GPT_number_of_pairs_l233_23376

theorem number_of_pairs : 
  (∀ n m : ℕ, (1 ≤ m ∧ m ≤ 2012) → (5^n < 2^m ∧ 2^m < 2^(m+2) ∧ 2^(m+2) < 5^(n+1))) → 
  (∃ c, c = 279) :=
by
  sorry

end NUMINAMATH_GPT_number_of_pairs_l233_23376


namespace NUMINAMATH_GPT_coin_flip_probability_difference_l233_23331

theorem coin_flip_probability_difference :
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  abs (p3 - p4) = (7/16:ℝ) :=
by
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  sorry

end NUMINAMATH_GPT_coin_flip_probability_difference_l233_23331


namespace NUMINAMATH_GPT_volleyball_team_lineup_l233_23397

theorem volleyball_team_lineup : 
  let team_members := 10
  let lineup_positions := 6
  10 * 9 * 8 * 7 * 6 * 5 = 151200 := by sorry

end NUMINAMATH_GPT_volleyball_team_lineup_l233_23397


namespace NUMINAMATH_GPT_possible_slopes_of_line_intersecting_ellipse_l233_23319

theorem possible_slopes_of_line_intersecting_ellipse :
  ∃ m : ℝ, 
    (∀ (x y : ℝ), (y = m * x + 3) → (4 * x^2 + 25 * y^2 = 100)) →
    (m ∈ Set.Iio (-Real.sqrt (16 / 405)) ∪ Set.Ici (Real.sqrt (16 / 405))) :=
sorry

end NUMINAMATH_GPT_possible_slopes_of_line_intersecting_ellipse_l233_23319


namespace NUMINAMATH_GPT_polar_coordinates_full_circle_l233_23306

theorem polar_coordinates_full_circle :
  ∀ (r : ℝ) (θ : ℝ), (r = 3 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) → (r = 3 ∧ ∀ (θ : ℝ), 0 ≤ θ ∧ θ < 2 * Real.pi ↔ r = 3) :=
by
  intros r θ h
  sorry

end NUMINAMATH_GPT_polar_coordinates_full_circle_l233_23306


namespace NUMINAMATH_GPT_intersection_eq_l233_23373

variable (A : Set ℤ) (B : Set ℤ)

def A_def := A = {-1, 0, 1, 2}
def B_def := B = {x | -1 < x ∧ x < 2}

theorem intersection_eq : A ∩ B = {0, 1} :=
by
  have A_def : A = {-1, 0, 1, 2} := sorry
  have B_def : B = {x | -1 < x ∧ x < 2} := sorry
  sorry

end NUMINAMATH_GPT_intersection_eq_l233_23373


namespace NUMINAMATH_GPT_total_minutes_ironing_over_4_weeks_l233_23380

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

end NUMINAMATH_GPT_total_minutes_ironing_over_4_weeks_l233_23380


namespace NUMINAMATH_GPT_range_of_a_l233_23333

noncomputable def problem_statement : Prop :=
  ∃ x : ℝ, (1 ≤ x) ∧ (∀ a : ℝ, (1 + 1 / x) ^ (x + a) ≥ Real.exp 1 → a ≥ 1 / Real.log 2 - 1)

theorem range_of_a : problem_statement :=
sorry

end NUMINAMATH_GPT_range_of_a_l233_23333


namespace NUMINAMATH_GPT_three_hundred_percent_of_x_equals_seventy_five_percent_of_y_l233_23370

theorem three_hundred_percent_of_x_equals_seventy_five_percent_of_y
  (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 20) : y = 80 := by
  sorry

end NUMINAMATH_GPT_three_hundred_percent_of_x_equals_seventy_five_percent_of_y_l233_23370


namespace NUMINAMATH_GPT_solve_for_x_l233_23358

noncomputable def valid_x (x : ℝ) : Prop :=
  let l := 4 * x
  let w := 2 * x + 6
  l * w = 2 * (l + w)

theorem solve_for_x : 
  ∃ (x : ℝ), valid_x x ↔ x = (-3 + Real.sqrt 33) / 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l233_23358


namespace NUMINAMATH_GPT_length_segment_midpoints_diagonals_trapezoid_l233_23384

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

end NUMINAMATH_GPT_length_segment_midpoints_diagonals_trapezoid_l233_23384


namespace NUMINAMATH_GPT_cargo_arrival_in_days_l233_23382

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

end NUMINAMATH_GPT_cargo_arrival_in_days_l233_23382


namespace NUMINAMATH_GPT_mangoes_rate_l233_23390

theorem mangoes_rate (grapes_weight mangoes_weight total_amount grapes_rate mango_rate : ℕ)
  (h1 : grapes_weight = 7)
  (h2 : grapes_rate = 68)
  (h3 : total_amount = 908)
  (h4 : mangoes_weight = 9)
  (h5 : total_amount - grapes_weight * grapes_rate = mangoes_weight * mango_rate) :
  mango_rate = 48 :=
by
  sorry

end NUMINAMATH_GPT_mangoes_rate_l233_23390


namespace NUMINAMATH_GPT_find_value_of_expression_l233_23372

theorem find_value_of_expression (a b : ℝ) (h : a + 2 * b - 1 = 0) : 3 * a + 6 * b = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l233_23372


namespace NUMINAMATH_GPT_average_sale_six_months_l233_23398

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

end NUMINAMATH_GPT_average_sale_six_months_l233_23398


namespace NUMINAMATH_GPT_line_slope_intercept_sum_l233_23312

theorem line_slope_intercept_sum (m b : ℝ)
    (h1 : m = 4)
    (h2 : ∃ b, ∀ x y : ℝ, y = mx + b → y = 5 ∧ x = -2)
    : m + b = 17 := by
  sorry

end NUMINAMATH_GPT_line_slope_intercept_sum_l233_23312


namespace NUMINAMATH_GPT_value_of_a_l233_23356

theorem value_of_a (a x : ℝ) (h1 : x = 2) (h2 : a * x = 4) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l233_23356


namespace NUMINAMATH_GPT_f_cos_eq_l233_23350

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Given condition
axiom f_sin_eq : f (Real.sin x) = 3 - Real.cos (2 * x)

-- The statement we want to prove
theorem f_cos_eq : f (Real.cos x) = 3 + Real.cos (2 * x) := 
by
  sorry

end NUMINAMATH_GPT_f_cos_eq_l233_23350


namespace NUMINAMATH_GPT_simple_interest_rate_l233_23385

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

end NUMINAMATH_GPT_simple_interest_rate_l233_23385


namespace NUMINAMATH_GPT_weight_of_replaced_person_l233_23389

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

end NUMINAMATH_GPT_weight_of_replaced_person_l233_23389


namespace NUMINAMATH_GPT_no_integer_solutions_system_l233_23394

theorem no_integer_solutions_system :
  ¬(∃ x y z : ℤ, 
    x^6 + x^3 + x^3 * y + y = 147^157 ∧ 
    x^3 + x^3 * y + y^2 + y + z^9 = 157^147) :=
  sorry

end NUMINAMATH_GPT_no_integer_solutions_system_l233_23394


namespace NUMINAMATH_GPT_smallest_four_digit_multiple_of_18_l233_23366

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ (∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0) → n ≤ m) ∧ n = 1008 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_multiple_of_18_l233_23366


namespace NUMINAMATH_GPT_salt_solution_mixture_l233_23363

theorem salt_solution_mixture (x : ℝ) :  
  (0.80 * x + 0.35 * 150 = 0.55 * (150 + x)) → x = 120 :=
by 
  sorry

end NUMINAMATH_GPT_salt_solution_mixture_l233_23363


namespace NUMINAMATH_GPT_num_of_consecutive_sets_sum_18_eq_2_l233_23369

theorem num_of_consecutive_sets_sum_18_eq_2 : 
  ∃ (sets : Finset (Finset ℕ)), 
    (∀ s ∈ sets, (∃ n a, n ≥ 3 ∧ (s = Finset.range (a + n - 1) \ Finset.range (a - 1)) ∧ 
    s.sum id = 18)) ∧ 
    sets.card = 2 := 
sorry

end NUMINAMATH_GPT_num_of_consecutive_sets_sum_18_eq_2_l233_23369


namespace NUMINAMATH_GPT_find_2a_plus_b_l233_23330

open Function

noncomputable def f (a b : ℝ) (x : ℝ) := 2 * a * x - 3 * b
noncomputable def g (x : ℝ) := 5 * x + 4
noncomputable def h (a b : ℝ) (x : ℝ) := g (f a b x)
noncomputable def h_inv (x : ℝ) := 2 * x - 9

theorem find_2a_plus_b (a b : ℝ) (h_comp_inv_eq_id : ∀ x, h a b (h_inv x) = x) :
  2 * a + b = 1 / 15 := 
sorry

end NUMINAMATH_GPT_find_2a_plus_b_l233_23330


namespace NUMINAMATH_GPT_find_unknown_number_l233_23303

theorem find_unknown_number (x : ℝ) (h : (8 / 100) * x = 96) : x = 1200 :=
by
  sorry

end NUMINAMATH_GPT_find_unknown_number_l233_23303


namespace NUMINAMATH_GPT_count_distinct_m_values_l233_23305

theorem count_distinct_m_values : 
  ∃ m_values : Finset ℤ, 
  (∀ x1 x2 : ℤ, x1 * x2 = 30 → (m_values : Set ℤ) = { x1 + x2 }) ∧ 
  m_values.card = 8 :=
by
  sorry

end NUMINAMATH_GPT_count_distinct_m_values_l233_23305


namespace NUMINAMATH_GPT_find_b_l233_23343

theorem find_b (b : ℝ) :
  (∃ x1 x2 : ℝ, (x1 + x2 = -2) ∧
    ((x1 + 1)^3 + x1 / (x1 + 1) = -x1 + b) ∧
    ((x2 + 1)^3 + x2 / (x2 + 1) = -x2 + b)) →
  b = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l233_23343


namespace NUMINAMATH_GPT_find_quotient_l233_23348

theorem find_quotient :
  ∃ q : ℕ, ∀ L S : ℕ, L = 1584 ∧ S = 249 ∧ (L - S = 1335) ∧ (L = S * q + 15) → q = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_quotient_l233_23348


namespace NUMINAMATH_GPT_CitadelSchoolEarnings_l233_23352

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

end NUMINAMATH_GPT_CitadelSchoolEarnings_l233_23352


namespace NUMINAMATH_GPT_min_pictures_needed_l233_23360

theorem min_pictures_needed (n m : ℕ) (participants : Fin n → Fin m → Prop)
  (h1 : n = 60) (h2 : m ≤ 30)
  (h3 : ∀ (i j : Fin n), ∃ (k : Fin m), participants i k ∧ participants j k) :
  m = 6 :=
sorry

end NUMINAMATH_GPT_min_pictures_needed_l233_23360


namespace NUMINAMATH_GPT_sin_sum_leq_3div2_sqrt3_l233_23379

theorem sin_sum_leq_3div2_sqrt3 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π) :
  Real.sin A + Real.sin B + Real.sin C ≤ (3 / 2) * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_sin_sum_leq_3div2_sqrt3_l233_23379


namespace NUMINAMATH_GPT_jerry_remaining_money_l233_23367

def cost_of_mustard_oil (price_per_liter : ℕ) (liters : ℕ) : ℕ := price_per_liter * liters
def cost_of_pasta (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds
def cost_of_sauce (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds

def total_cost (price_mustard_oil : ℕ) (liters_mustard : ℕ) (price_pasta : ℕ) (pounds_pasta : ℕ) (price_sauce : ℕ) (pounds_sauce : ℕ) : ℕ := 
  cost_of_mustard_oil price_mustard_oil liters_mustard + cost_of_pasta price_pasta pounds_pasta + cost_of_sauce price_sauce pounds_sauce

def remaining_money (initial_money : ℕ) (total_cost : ℕ) : ℕ := initial_money - total_cost

theorem jerry_remaining_money : 
  remaining_money 50 (total_cost 13 2 4 3 5 1) = 7 := by 
  sorry

end NUMINAMATH_GPT_jerry_remaining_money_l233_23367


namespace NUMINAMATH_GPT_line_through_point_perpendicular_l233_23399

theorem line_through_point_perpendicular :
  ∃ (a b : ℝ), ∀ (x : ℝ), y = - (3 / 2) * x + 8 ∧ y - 2 = - (3 / 2) * (x - 4) ∧ 2*x - 3*y = 6 → y = - (3 / 2) * x + 8 :=
by 
  sorry

end NUMINAMATH_GPT_line_through_point_perpendicular_l233_23399


namespace NUMINAMATH_GPT_domain_of_f_l233_23317

open Real

noncomputable def f (x : ℝ) : ℝ := log ((2 - x) / (2 + x))

theorem domain_of_f : ∀ x : ℝ, (2 - x) / (2 + x) > 0 ∧ 2 + x ≠ 0 ↔ -2 < x ∧ x < 2 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_domain_of_f_l233_23317


namespace NUMINAMATH_GPT_second_term_of_geo_series_l233_23347

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

end NUMINAMATH_GPT_second_term_of_geo_series_l233_23347


namespace NUMINAMATH_GPT_pens_per_student_l233_23388

theorem pens_per_student (n : ℕ) (h1 : 0 < n) (h2 : n ≤ 50) (h3 : 100 % n = 0) (h4 : 50 % n = 0) : 100 / n = 2 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_pens_per_student_l233_23388


namespace NUMINAMATH_GPT_range_of_a_l233_23324

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (2 * x - 1 < 3) ∧ (x - a < 0) → (x < a)) → (a ≤ 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l233_23324


namespace NUMINAMATH_GPT_alcohol_percentage_proof_l233_23328

noncomputable def percentage_alcohol_new_mixture 
  (original_solution_volume : ℕ)
  (percent_A : ℚ)
  (concentration_A : ℚ)
  (percent_B : ℚ)
  (concentration_B : ℚ)
  (percent_C : ℚ)
  (concentration_C : ℚ)
  (water_added_volume : ℕ) : ℚ :=
((original_solution_volume * percent_A * concentration_A) +
 (original_solution_volume * percent_B * concentration_B) +
 (original_solution_volume * percent_C * concentration_C)) /
 (original_solution_volume + water_added_volume) * 100

theorem alcohol_percentage_proof : 
  percentage_alcohol_new_mixture 24 0.30 0.80 0.40 0.90 0.30 0.95 16 = 53.1 := 
by 
  sorry

end NUMINAMATH_GPT_alcohol_percentage_proof_l233_23328


namespace NUMINAMATH_GPT_sector_area_l233_23378

theorem sector_area (R : ℝ) (hR_pos : R > 0) (h_circumference : 4 * R = 2 * R + arc_length) :
  (1 / 2) * arc_length * R = R^2 :=
by sorry

end NUMINAMATH_GPT_sector_area_l233_23378


namespace NUMINAMATH_GPT_calculate_f_at_2x_l233_23313

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem using the given condition and the desired result
theorem calculate_f_at_2x (x : ℝ) : f (2 * x) = 4 * x^2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_calculate_f_at_2x_l233_23313


namespace NUMINAMATH_GPT_jimmys_speed_l233_23396

theorem jimmys_speed 
(Mary_speed : ℕ) (total_distance : ℕ) (t : ℕ)
(h1 : Mary_speed = 5)
(h2 : total_distance = 9)
(h3 : t = 1)
: ∃ (Jimmy_speed : ℕ), Jimmy_speed = 4 :=
by
  -- calculation steps skipped here
  sorry

end NUMINAMATH_GPT_jimmys_speed_l233_23396


namespace NUMINAMATH_GPT_solve_system_of_equations_l233_23314

theorem solve_system_of_equations (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x^4 + y^4 - x^2 * y^2 = 13)
  (h2 : x^2 - y^2 + 2 * x * y = 1) :
  x = 1 ∧ y = 2 :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l233_23314


namespace NUMINAMATH_GPT_break_room_capacity_l233_23345

theorem break_room_capacity :
  let people_per_table := 8
  let number_of_tables := 4
  people_per_table * number_of_tables = 32 :=
by
  let people_per_table := 8
  let number_of_tables := 4
  have h : people_per_table * number_of_tables = 32 := by sorry
  exact h

end NUMINAMATH_GPT_break_room_capacity_l233_23345


namespace NUMINAMATH_GPT_count_terms_expansion_l233_23375

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

end NUMINAMATH_GPT_count_terms_expansion_l233_23375


namespace NUMINAMATH_GPT_smallest_possible_n_l233_23392

theorem smallest_possible_n : ∃ (n : ℕ), (∀ (r g b : ℕ), 24 * n = 18 * r ∧ 24 * n = 16 * g ∧ 24 * n = 20 * b) ∧ n = 30 :=
by
  -- Sorry, we're skipping the proof, as specified.
  sorry

end NUMINAMATH_GPT_smallest_possible_n_l233_23392


namespace NUMINAMATH_GPT_cost_of_55_lilies_l233_23362

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

end NUMINAMATH_GPT_cost_of_55_lilies_l233_23362


namespace NUMINAMATH_GPT_otimes_self_twice_l233_23301

def otimes (x y : ℝ) := x^2 - y^2

theorem otimes_self_twice (a : ℝ) : (otimes (otimes a a) (otimes a a)) = 0 :=
  sorry

end NUMINAMATH_GPT_otimes_self_twice_l233_23301


namespace NUMINAMATH_GPT_father_has_4_chocolate_bars_left_l233_23365

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

end NUMINAMATH_GPT_father_has_4_chocolate_bars_left_l233_23365


namespace NUMINAMATH_GPT_annual_income_increase_l233_23374

variable (x y : ℝ)

-- Definitions of the conditions
def regression_line (x : ℝ) : ℝ := 0.254 * x + 0.321

-- The statement we want to prove
theorem annual_income_increase (x : ℝ) : regression_line (x + 1) - regression_line x = 0.254 := 
sorry

end NUMINAMATH_GPT_annual_income_increase_l233_23374


namespace NUMINAMATH_GPT_second_company_managers_percent_l233_23334

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

end NUMINAMATH_GPT_second_company_managers_percent_l233_23334


namespace NUMINAMATH_GPT_last_three_digits_7_pow_80_l233_23338

theorem last_three_digits_7_pow_80 : (7^80) % 1000 = 961 := by
  sorry

end NUMINAMATH_GPT_last_three_digits_7_pow_80_l233_23338


namespace NUMINAMATH_GPT_solve_equation_l233_23361

theorem solve_equation (x : ℝ) (h₀ : x = 46) :
  ( (8 / (Real.sqrt (x - 10) - 10)) + 
    (2 / (Real.sqrt (x - 10) - 5)) + 
    (9 / (Real.sqrt (x - 10) + 5)) + 
    (15 / (Real.sqrt (x - 10) + 10)) = 0) := 
by 
  sorry

end NUMINAMATH_GPT_solve_equation_l233_23361


namespace NUMINAMATH_GPT_ana_bonita_age_difference_l233_23309

theorem ana_bonita_age_difference (A B n : ℕ) 
  (h1 : A = B + n)
  (h2 : A - 1 = 7 * (B - 1))
  (h3 : A = B^3) : 
  n = 6 :=
sorry

end NUMINAMATH_GPT_ana_bonita_age_difference_l233_23309


namespace NUMINAMATH_GPT_frac_1_7_correct_l233_23383

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

end NUMINAMATH_GPT_frac_1_7_correct_l233_23383


namespace NUMINAMATH_GPT_solve_equation_l233_23357

theorem solve_equation (x : ℝ) :
  (4 * x + 1) * (3 * x + 1) * (2 * x + 1) * (x + 1) = 3 * x ^ 4  →
  x = (-5 + Real.sqrt 13) / 6 ∨ x = (-5 - Real.sqrt 13) / 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l233_23357


namespace NUMINAMATH_GPT_unique_real_solution_l233_23323

theorem unique_real_solution (x y : ℝ) (h1 : x^3 = 2 - y) (h2 : y^3 = 2 - x) : x = 1 ∧ y = 1 :=
sorry

end NUMINAMATH_GPT_unique_real_solution_l233_23323


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l233_23342

-- Definition of the conditions
def effective_downstream_speed (v_m v_c : ℝ) : Prop := (v_m + v_c) = 10
def effective_upstream_speed (v_m v_c : ℝ) : Prop := (v_m - v_c) = 11.25

-- The proof problem statement
theorem speed_of_man_in_still_water (v_m v_c : ℝ) 
  (h1 : effective_downstream_speed v_m v_c)
  (h2 : effective_upstream_speed v_m v_c)
  : v_m = 10.625 :=
sorry

end NUMINAMATH_GPT_speed_of_man_in_still_water_l233_23342


namespace NUMINAMATH_GPT_units_digit_of_n_l233_23368

theorem units_digit_of_n (n : ℕ) (h : n = 56^78 + 87^65) : (n % 10) = 3 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_n_l233_23368


namespace NUMINAMATH_GPT_max_value_of_expression_l233_23386

theorem max_value_of_expression (x y : ℝ) (h1 : |x - y| ≤ 2) (h2 : |3 * x + y| ≤ 6) : x^2 + y^2 ≤ 10 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l233_23386


namespace NUMINAMATH_GPT_negation_of_exists_l233_23316

-- Lean definition of the proposition P
def P (a : ℝ) : Prop :=
  ∃ x0 : ℝ, x0 > 0 ∧ 2^x0 * (x0 - a) > 1

-- The negation of the proposition P
def neg_P (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → 2^x * (x - a) ≤ 1

-- Theorem stating that the negation of P is neg_P
theorem negation_of_exists (a : ℝ) : ¬ P a ↔ neg_P a :=
by
  -- (Proof to be provided)
  sorry

end NUMINAMATH_GPT_negation_of_exists_l233_23316


namespace NUMINAMATH_GPT_circle_B_area_l233_23311

theorem circle_B_area
  (r R : ℝ)
  (h1 : ∀ (x : ℝ), x = 5)  -- derived from r = 5
  (h2 : R = 2 * r)
  (h3 : 25 * Real.pi = Real.pi * r^2)
  (h4 : R = 10)  -- derived from diameter relation
  : ∃ A_B : ℝ, A_B = 100 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circle_B_area_l233_23311


namespace NUMINAMATH_GPT_original_price_of_shirt_l233_23327

variables (S C P : ℝ)

def shirt_condition := S = C / 3
def pants_condition := S = P / 2
def total_paid := 0.90 * S + 0.95 * C + P = 900

theorem original_price_of_shirt :
  shirt_condition S C →
  pants_condition S P →
  total_paid S C P →
  S = 900 / 5.75 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_shirt_l233_23327


namespace NUMINAMATH_GPT_balloon_arrangements_correct_l233_23337

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

-- Define the number of ways to arrange "BALLOON"
noncomputable def arrangements_balloon : ℕ := factorial 7 / (factorial 2 * factorial 2)

-- State the theorem
theorem balloon_arrangements_correct : arrangements_balloon = 1260 := by sorry

end NUMINAMATH_GPT_balloon_arrangements_correct_l233_23337


namespace NUMINAMATH_GPT_total_cats_l233_23364

variable (initialCats : ℝ)
variable (boughtCats : ℝ)

theorem total_cats (h1 : initialCats = 11.0) (h2 : boughtCats = 43.0) :
    initialCats + boughtCats = 54.0 :=
by
  sorry

end NUMINAMATH_GPT_total_cats_l233_23364


namespace NUMINAMATH_GPT_initial_population_l233_23391

theorem initial_population (P : ℝ) (h : P * 1.21 = 12000) : P = 12000 / 1.21 :=
by sorry

end NUMINAMATH_GPT_initial_population_l233_23391


namespace NUMINAMATH_GPT_find_v4_l233_23354

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

end NUMINAMATH_GPT_find_v4_l233_23354


namespace NUMINAMATH_GPT_smallest_int_x_l233_23355

theorem smallest_int_x (x : ℤ) (h : 2 * x + 5 < 3 * x - 10) : x = 16 :=
sorry

end NUMINAMATH_GPT_smallest_int_x_l233_23355


namespace NUMINAMATH_GPT_min_value_2x_minus_y_l233_23341

theorem min_value_2x_minus_y :
  ∃ (x y : ℝ), (y = abs (x - 1) ∨ y = 2) ∧ (y ≤ 2) ∧ (2 * x - y = -4) :=
by
  sorry

end NUMINAMATH_GPT_min_value_2x_minus_y_l233_23341


namespace NUMINAMATH_GPT_value_of_M_l233_23318

theorem value_of_M (M : ℝ) (h : (20 / 100) * M = (60 / 100) * 1500) : M = 4500 :=
by {
    sorry
}

end NUMINAMATH_GPT_value_of_M_l233_23318


namespace NUMINAMATH_GPT_part1_part2_l233_23304

-- Part 1: Prove that x < -12 given the inequality 2(-3 + x) > 3(x + 2)
theorem part1 (x : ℝ) : 2 * (-3 + x) > 3 * (x + 2) → x < -12 := 
  by
  intro h
  sorry

-- Part 2: Prove that 0 ≤ x < 3 given the system of inequalities
theorem part2 (x : ℝ) : 
    (1 / 2) * (x + 1) < 2 ∧ (x + 2) / 2 ≥ (x + 3) / 3 → 0 ≤ x ∧ x < 3 :=
  by
  intro h
  sorry

end NUMINAMATH_GPT_part1_part2_l233_23304


namespace NUMINAMATH_GPT_combined_share_a_c_l233_23321

-- Define the conditions
def total_money : ℕ := 15800
def ratio_a : ℕ := 5
def ratio_b : ℕ := 9
def ratio_c : ℕ := 6
def ratio_d : ℕ := 5

-- The total parts in the ratio
def total_parts : ℕ := ratio_a + ratio_b + ratio_c + ratio_d

-- The value of each part
def value_per_part : ℕ := total_money / total_parts

-- The shares of a and c
def share_a : ℕ := ratio_a * value_per_part
def share_c : ℕ := ratio_c * value_per_part

-- Prove that the combined share of a + c equals 6952
theorem combined_share_a_c : share_a + share_c = 6952 :=
by
  -- This is the proof placeholder
  sorry

end NUMINAMATH_GPT_combined_share_a_c_l233_23321


namespace NUMINAMATH_GPT_octahedron_parallel_edge_pairs_count_l233_23340

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

end NUMINAMATH_GPT_octahedron_parallel_edge_pairs_count_l233_23340


namespace NUMINAMATH_GPT_divisible_by_6_of_cubed_sum_div_by_18_l233_23325

theorem divisible_by_6_of_cubed_sum_div_by_18 (a b c : ℤ) 
  (h : a^3 + b^3 + c^3 ≡ 0 [ZMOD 18]) : (a * b * c) ≡ 0 [ZMOD 6] :=
sorry

end NUMINAMATH_GPT_divisible_by_6_of_cubed_sum_div_by_18_l233_23325


namespace NUMINAMATH_GPT_max_length_OB_l233_23351

-- Define the problem conditions
def angle_AOB : ℝ := 45
def length_AB : ℝ := 2
def max_sin_angle_OAB : ℝ := 1

-- Claim to be proven
theorem max_length_OB : ∃ OB_max, OB_max = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_max_length_OB_l233_23351
