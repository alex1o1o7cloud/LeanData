import Mathlib

namespace NUMINAMATH_GPT_x_intercept_is_34_l1956_195625

-- Definitions of the initial line, rotation, and point.
def line_l (x y : ℝ) : Prop := 4 * x - 3 * y + 50 = 0

def rotation_angle : ℝ := 30
def rotation_center : ℝ × ℝ := (10, 10)

-- Define the slope of the line l
noncomputable def slope_of_l : ℝ := 4 / 3

-- Define the slope of the line m after rotating line l by 30 degrees counterclockwise
noncomputable def tan_30 : ℝ := 1 / Real.sqrt 3
noncomputable def slope_of_m : ℝ := (slope_of_l + tan_30) / (1 - slope_of_l * tan_30)

-- Assume line m goes through the point (rotation_center.x, rotation_center.y)
-- This defines line m
def line_m (x y : ℝ) : Prop := y - rotation_center.2 = slope_of_m * (x - rotation_center.1)

-- To find the x-intercept of line m, we set y = 0 and solve for x
noncomputable def x_intercept_of_m : ℝ := rotation_center.1 - rotation_center.2 / slope_of_m

-- Proof statement that the x-intercept of line m is 34
theorem x_intercept_is_34 : x_intercept_of_m = 34 :=
by
  -- This would be the proof, but for now we leave it as sorry
  sorry

end NUMINAMATH_GPT_x_intercept_is_34_l1956_195625


namespace NUMINAMATH_GPT_time_to_fill_pool_l1956_195669

-- Define constants based on the conditions
def pool_capacity : ℕ := 30000
def hose_count : ℕ := 5
def flow_rate_per_hose : ℕ := 25 / 10  -- 2.5 gallons per minute
def conversion_minutes_to_hours : ℕ := 60

-- Define the total flow rate per minute
def total_flow_rate_per_minute : ℕ := hose_count * flow_rate_per_hose

-- Define the total flow rate per hour
def total_flow_rate_per_hour : ℕ := total_flow_rate_per_minute * conversion_minutes_to_hours

-- Theorem stating the number of hours required to fill the pool
theorem time_to_fill_pool : pool_capacity / total_flow_rate_per_hour = 40 := by
  sorry -- Proof will be provided here

end NUMINAMATH_GPT_time_to_fill_pool_l1956_195669


namespace NUMINAMATH_GPT_ratio_of_books_l1956_195666

theorem ratio_of_books (longest_pages : ℕ) (middle_pages : ℕ) (shortest_pages : ℕ) :
  longest_pages = 396 ∧ middle_pages = 297 ∧ shortest_pages = longest_pages / 4 →
  (middle_pages / shortest_pages = 3) :=
by
  intros h
  obtain ⟨h_longest, h_middle, h_shortest⟩ := h
  sorry

end NUMINAMATH_GPT_ratio_of_books_l1956_195666


namespace NUMINAMATH_GPT_john_investment_in_bankA_l1956_195635

-- Definitions to set up the conditions
def total_investment : ℝ := 1500
def bankA_rate : ℝ := 0.04
def bankB_rate : ℝ := 0.06
def final_amount : ℝ := 1575

-- Definition of the question to be proved
theorem john_investment_in_bankA (x : ℝ) (h : 0 ≤ x ∧ x ≤ total_investment) :
  (x * (1 + bankA_rate) + (total_investment - x) * (1 + bankB_rate) = final_amount) -> x = 750 := sorry


end NUMINAMATH_GPT_john_investment_in_bankA_l1956_195635


namespace NUMINAMATH_GPT_ratio_a_to_c_l1956_195621

theorem ratio_a_to_c (a b c d : ℕ) 
  (h1 : a / b = 5 / 4) 
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 5) : 
  a / c = 75 / 16 := 
  by 
    sorry

end NUMINAMATH_GPT_ratio_a_to_c_l1956_195621


namespace NUMINAMATH_GPT_cost_of_notebook_l1956_195634

theorem cost_of_notebook (num_students : ℕ) (more_than_half_bought : ℕ) (num_notebooks : ℕ) 
                         (cost_per_notebook : ℕ) (total_cost : ℕ) 
                         (half_students : more_than_half_bought > 18) 
                         (more_than_one_notebook : num_notebooks > 1) 
                         (cost_gt_notebooks : cost_per_notebook > num_notebooks) 
                         (calc_total_cost : more_than_half_bought * cost_per_notebook * num_notebooks = 2310) :
  cost_per_notebook = 11 := 
sorry

end NUMINAMATH_GPT_cost_of_notebook_l1956_195634


namespace NUMINAMATH_GPT_length_of_FD_l1956_195638

theorem length_of_FD (a b c d f e : ℝ) (x : ℝ) :
  a = 0 ∧ b = 8 ∧ c = 8 ∧ d = 0 ∧ 
  e = 8 * (2 / 3) ∧ 
  (8 - x)^2 = x^2 + (8 / 3)^2 ∧ 
  a = d → c = b → 
  d = 8 → 
  x = 32 / 9 :=
by
  sorry

end NUMINAMATH_GPT_length_of_FD_l1956_195638


namespace NUMINAMATH_GPT_exists_rational_non_integer_xy_no_rational_non_integer_xy_l1956_195675

-- Part (a)
theorem exists_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  (∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
   ∃ z1 z2 : ℤ, 19 * x + 8 * y = ↑z1 ∧ 8 * x + 3 * y = ↑z2) :=
sorry

-- Part (b)
theorem no_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  ¬ ∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
  ∃ z1 z2 : ℤ, 19 * x^2 + 8 * y^2 = ↑z1 ∧ 8 * x^2 + 3 * y^2 = ↑z2 :=
sorry

end NUMINAMATH_GPT_exists_rational_non_integer_xy_no_rational_non_integer_xy_l1956_195675


namespace NUMINAMATH_GPT_sandy_spent_on_shorts_l1956_195637

variable (amount_on_shirt amount_on_jacket total_amount amount_on_shorts : ℝ)

theorem sandy_spent_on_shorts :
  amount_on_shirt = 12.14 →
  amount_on_jacket = 7.43 →
  total_amount = 33.56 →
  amount_on_shorts = total_amount - amount_on_shirt - amount_on_jacket →
  amount_on_shorts = 13.99 :=
by
  intros h_shirt h_jacket h_total h_computation
  sorry

end NUMINAMATH_GPT_sandy_spent_on_shorts_l1956_195637


namespace NUMINAMATH_GPT_reciprocal_of_minus_one_half_l1956_195629

theorem reciprocal_of_minus_one_half : (1 / (-1 / 2)) = -2 := 
by sorry

end NUMINAMATH_GPT_reciprocal_of_minus_one_half_l1956_195629


namespace NUMINAMATH_GPT_eval_five_over_two_l1956_195683

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x - 2 else Real.log (x - 1) / Real.log 2

theorem eval_five_over_two : f (5 / 2) = -1 := by
  sorry

end NUMINAMATH_GPT_eval_five_over_two_l1956_195683


namespace NUMINAMATH_GPT_pear_sales_l1956_195697

theorem pear_sales (sale_afternoon : ℕ) (h1 : sale_afternoon = 260)
  (h2 : ∃ sale_morning : ℕ, sale_afternoon = 2 * sale_morning) :
  sale_afternoon / 2 + sale_afternoon = 390 :=
by
  sorry

end NUMINAMATH_GPT_pear_sales_l1956_195697


namespace NUMINAMATH_GPT_distance_from_axis_gt_l1956_195685

theorem distance_from_axis_gt 
  (a b x1 x2 y1 y2 : ℝ) (h₁ : a > 0) 
  (h₂ : y1 = a * x1^2 - 2 * a * x1 + b) 
  (h₃ : y2 = a * x2^2 - 2 * a * x2 + b) 
  (h₄ : y1 > y2) : 
  |x1 - 1| > |x2 - 1| := 
sorry

end NUMINAMATH_GPT_distance_from_axis_gt_l1956_195685


namespace NUMINAMATH_GPT_max_product_is_negative_one_l1956_195663

def f (x : ℝ) : ℝ := sorry    -- Assume some function f
def g (x : ℝ) : ℝ := sorry    -- Assume some function g

theorem max_product_is_negative_one (h_f_range : ∀ y, 1 ≤ y ∧ y ≤ 6 → ∃ x, f x = y) 
    (h_g_range : ∀ y, -4 ≤ y ∧ y ≤ -1 → ∃ x, g x = y) : 
    ∃ b, b = -1 ∧ ∀ x, f x * g x ≤ b :=
sorry

end NUMINAMATH_GPT_max_product_is_negative_one_l1956_195663


namespace NUMINAMATH_GPT_solve_cubed_root_equation_l1956_195616

theorem solve_cubed_root_equation :
  (∃ x : ℚ, (5 - 2 / x) ^ (1 / 3) = -3) ↔ x = 1 / 16 := 
by
  sorry

end NUMINAMATH_GPT_solve_cubed_root_equation_l1956_195616


namespace NUMINAMATH_GPT_instantaneous_velocity_at_2_l1956_195636

def displacement (t : ℝ) : ℝ := 2 * t^2 + 3

theorem instantaneous_velocity_at_2 : (deriv displacement 2) = 8 :=
by 
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_2_l1956_195636


namespace NUMINAMATH_GPT_ln_n_lt_8m_l1956_195642

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := 
  Real.log x - m * x^2 + 2 * n * x

theorem ln_n_lt_8m (m : ℝ) (n : ℝ) (h₀ : 0 < n) (h₁ : ∀ x > 0, f x m n ≤ f 1 m n) : 
  Real.log n < 8 * m := 
sorry

end NUMINAMATH_GPT_ln_n_lt_8m_l1956_195642


namespace NUMINAMATH_GPT_packaging_combinations_l1956_195649

-- Conditions
def wrapping_paper_choices : ℕ := 10
def ribbon_colors : ℕ := 5
def gift_tag_styles : ℕ := 6

-- Question and proof
theorem packaging_combinations : wrapping_paper_choices * ribbon_colors * gift_tag_styles = 300 := by
  sorry

end NUMINAMATH_GPT_packaging_combinations_l1956_195649


namespace NUMINAMATH_GPT_number_of_BMWs_sold_l1956_195661

-- Defining the percentages of Mercedes, Toyota, and Acura cars sold
def percentageMercedes : ℕ := 18
def percentageToyota  : ℕ := 25
def percentageAcura   : ℕ := 15

-- Defining the total number of cars sold
def totalCars : ℕ := 250

-- The theorem to be proved
theorem number_of_BMWs_sold : (totalCars * (100 - (percentageMercedes + percentageToyota + percentageAcura)) / 100) = 105 := by
  sorry -- Proof to be filled in later

end NUMINAMATH_GPT_number_of_BMWs_sold_l1956_195661


namespace NUMINAMATH_GPT_goats_count_l1956_195665

variable (h d c t g : Nat)
variable (l : Nat)

theorem goats_count 
  (h_eq : h = 2)
  (d_eq : d = 5)
  (c_eq : c = 7)
  (t_eq : t = 3)
  (l_eq : l = 72)
  (legs_eq : 4 * h + 4 * d + 4 * c + 4 * t + 4 * g = l) : 
  g = 1 := by
  sorry

end NUMINAMATH_GPT_goats_count_l1956_195665


namespace NUMINAMATH_GPT_completing_the_square_l1956_195656

theorem completing_the_square (x : ℝ) : x^2 - 8 * x + 1 = 0 → (x - 4)^2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_completing_the_square_l1956_195656


namespace NUMINAMATH_GPT_additional_flowers_grew_l1956_195681

-- Define the initial conditions
def initial_flowers : ℕ := 10  -- Dane’s two daughters planted 5 flowers each (5 + 5).
def flowers_died : ℕ := 10     -- 10 flowers died.
def baskets : ℕ := 5
def flowers_per_basket : ℕ := 4

-- Total flowers harvested (from the baskets)
def total_harvested : ℕ := baskets * flowers_per_basket  -- 5 * 4 = 20

-- The proof to show additional flowers grown
theorem additional_flowers_grew : (total_harvested - initial_flowers + flowers_died) = 10 :=
by
  -- The final number of flowers and the initial number of flowers are known
  have final_flowers : ℕ := total_harvested
  have initial_plus_grown : ℕ := initial_flowers + (total_harvested - initial_flowers)
  -- Show the equality that defines the additional flowers grown
  show (total_harvested - initial_flowers + flowers_died) = 10
  sorry

end NUMINAMATH_GPT_additional_flowers_grew_l1956_195681


namespace NUMINAMATH_GPT_total_doll_count_l1956_195610

noncomputable def sister_dolls : ℕ := 8
noncomputable def hannah_dolls : ℕ := 5 * sister_dolls
noncomputable def total_dolls : ℕ := hannah_dolls + sister_dolls

theorem total_doll_count : total_dolls = 48 := 
by 
  sorry

end NUMINAMATH_GPT_total_doll_count_l1956_195610


namespace NUMINAMATH_GPT_circulation_ratio_l1956_195692

theorem circulation_ratio (A C_1971 C_total : ℕ) 
(hC1971 : C_1971 = 4 * A) 
(hCtotal : C_total = C_1971 + 9 * A) : 
(C_1971 : ℚ) / (C_total : ℚ) = 4 / 13 := 
sorry

end NUMINAMATH_GPT_circulation_ratio_l1956_195692


namespace NUMINAMATH_GPT_find_sum_of_integers_l1956_195646

theorem find_sum_of_integers (w x y z : ℤ)
  (h1 : w - x + y = 7)
  (h2 : x - y + z = 8)
  (h3 : y - z + w = 4)
  (h4 : z - w + x = 3) : w + x + y + z = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_of_integers_l1956_195646


namespace NUMINAMATH_GPT_first_car_distance_l1956_195660

-- Definitions for conditions
variable (x : ℝ) -- distance the first car ran before taking the right turn
def distance_apart_initial := 150 -- initial distance between the cars
def distance_first_car_main_road := 2 * x -- total distance first car ran on the main road
def distance_second_car := 62 -- distance the second car ran due to breakdown
def distance_between_cars := 38 -- distance between the cars after running 

-- Proof (statement only, no solution steps)
theorem first_car_distance (hx : distance_apart_initial = distance_first_car_main_road + distance_second_car + distance_between_cars) : 
  x = 25 :=
by
  unfold distance_apart_initial distance_first_car_main_road distance_second_car distance_between_cars at hx
  -- Implementation placeholder
  sorry

end NUMINAMATH_GPT_first_car_distance_l1956_195660


namespace NUMINAMATH_GPT_part_1_part_2_l1956_195614

noncomputable def prob_pass_no_fee : ℚ :=
  (3 / 4) * (2 / 3) +
  (1 / 4) * (3 / 4) * (2 / 3) +
  (3 / 4) * (1 / 3) * (2 / 3) +
  (1 / 4) * (3 / 4) * (1 / 3) * (2 / 3)

noncomputable def prob_pass_200_fee : ℚ :=
  (1 / 4) * (1 / 4) * (3 / 4) * ((2 / 3) + (1 / 3) * (2 / 3)) +
  (1 / 3) * (1 / 3) * (2 / 3) * ((3 / 4) + (1 / 4) * (3 / 4))

theorem part_1 : prob_pass_no_fee = 5 / 6 := by
  sorry

theorem part_2 : prob_pass_200_fee = 1 / 9 := by
  sorry

end NUMINAMATH_GPT_part_1_part_2_l1956_195614


namespace NUMINAMATH_GPT_not_true_diamond_self_zero_l1956_195682

-- Define the operator ⋄
def diamond (x y : ℝ) := |x - 2*y|

-- The problem statement in Lean4
theorem not_true_diamond_self_zero : ¬ (∀ x : ℝ, diamond x x = 0) := by
  sorry

end NUMINAMATH_GPT_not_true_diamond_self_zero_l1956_195682


namespace NUMINAMATH_GPT_points_per_member_correct_l1956_195620

noncomputable def points_per_member (total_members: ℝ) (absent_members: ℝ) (total_points: ℝ) :=
  (total_points / (total_members - absent_members))

theorem points_per_member_correct:
  points_per_member 5.0 2.0 6.0 = 2.0 :=
by 
  sorry

end NUMINAMATH_GPT_points_per_member_correct_l1956_195620


namespace NUMINAMATH_GPT_fraction_of_shaded_area_l1956_195699

theorem fraction_of_shaded_area
  (total_smaller_rectangles : ℕ)
  (shaded_smaller_rectangles : ℕ)
  (h1 : total_smaller_rectangles = 18)
  (h2 : shaded_smaller_rectangles = 4) :
  (shaded_smaller_rectangles : ℚ) / total_smaller_rectangles = 1 / 4 := 
sorry

end NUMINAMATH_GPT_fraction_of_shaded_area_l1956_195699


namespace NUMINAMATH_GPT_silver_tokens_at_end_l1956_195651

theorem silver_tokens_at_end {R B S : ℕ} (x y : ℕ) 
  (hR_init : R = 60) (hB_init : B = 90) 
  (hR_final : R = 60 - 3 * x + y) 
  (hB_final : B = 90 + 2 * x - 4 * y) 
  (h_end_conditions : 0 ≤ R ∧ R < 3 ∧ 0 ≤ B ∧ B < 4) : 
  S = x + y → 
  S = 23 :=
sorry

end NUMINAMATH_GPT_silver_tokens_at_end_l1956_195651


namespace NUMINAMATH_GPT_number_of_diagonals_in_octagon_l1956_195695

theorem number_of_diagonals_in_octagon :
  let n : ℕ := 8
  let num_diagonals := n * (n - 3) / 2
  num_diagonals = 20 := by
  sorry

end NUMINAMATH_GPT_number_of_diagonals_in_octagon_l1956_195695


namespace NUMINAMATH_GPT_sum_a4_a5_a6_l1956_195632

variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)
variable (h_a5 : a 5 = 21)

theorem sum_a4_a5_a6 : a 4 + a 5 + a 6 = 63 := by
  sorry

end NUMINAMATH_GPT_sum_a4_a5_a6_l1956_195632


namespace NUMINAMATH_GPT_gcf_90_108_l1956_195698

-- Given two integers 90 and 108
def a : ℕ := 90
def b : ℕ := 108

-- Question: What is the greatest common factor (GCF) of 90 and 108?
theorem gcf_90_108 : Nat.gcd a b = 18 :=
by {
  sorry
}

end NUMINAMATH_GPT_gcf_90_108_l1956_195698


namespace NUMINAMATH_GPT_total_infections_second_wave_l1956_195680

theorem total_infections_second_wave (cases_per_day_first_wave : ℕ)
                                     (factor_increase : ℕ)
                                     (duration_weeks : ℕ)
                                     (days_per_week : ℕ) :
                                     cases_per_day_first_wave = 300 →
                                     factor_increase = 4 →
                                     duration_weeks = 2 →
                                     days_per_week = 7 →
                                     (duration_weeks * days_per_week) * (cases_per_day_first_wave + factor_increase * cases_per_day_first_wave) = 21000 :=
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_total_infections_second_wave_l1956_195680


namespace NUMINAMATH_GPT_knitting_time_is_correct_l1956_195609

-- Definitions of the conditions
def time_per_hat : ℕ := 2
def time_per_scarf : ℕ := 3
def time_per_mitten : ℕ := 1
def time_per_sock : ℕ := 3 / 2 -- fractional time in hours
def time_per_sweater : ℕ := 6
def number_of_grandchildren : ℕ := 3

-- Compute total time for one complete outfit
def time_per_outfit : ℕ := time_per_hat + time_per_scarf + (time_per_mitten * 2) + (time_per_sock * 2) + time_per_sweater

-- Compute total time for all outfits
def total_knitting_time : ℕ := number_of_grandchildren * time_per_outfit

-- Prove that total knitting time is 48 hours
theorem knitting_time_is_correct : total_knitting_time = 48 := by
  unfold total_knitting_time time_per_outfit
  norm_num
  sorry

end NUMINAMATH_GPT_knitting_time_is_correct_l1956_195609


namespace NUMINAMATH_GPT_calculate_principal_amount_l1956_195696

theorem calculate_principal_amount (P : ℝ) (h1 : P * 0.1025 - P * 0.1 = 25) : 
  P = 10000 :=
by
  sorry

end NUMINAMATH_GPT_calculate_principal_amount_l1956_195696


namespace NUMINAMATH_GPT_cannot_form_triangle_l1956_195633

theorem cannot_form_triangle {a b c : ℝ} (h1 : a = 2) (h2 : b = 3) (h3 : c = 6) : 
  ¬ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
by
  sorry

end NUMINAMATH_GPT_cannot_form_triangle_l1956_195633


namespace NUMINAMATH_GPT_john_money_left_l1956_195644

-- Definitions for initial conditions
def initial_amount : ℤ := 100
def cost_roast : ℤ := 17
def cost_vegetables : ℤ := 11

-- Total spent calculation
def total_spent : ℤ := cost_roast + cost_vegetables

-- Remaining money calculation
def remaining_money : ℤ := initial_amount - total_spent

-- Theorem stating that John has €72 left
theorem john_money_left : remaining_money = 72 := by
  sorry

end NUMINAMATH_GPT_john_money_left_l1956_195644


namespace NUMINAMATH_GPT_number_of_perfect_cubes_l1956_195668

theorem number_of_perfect_cubes (n : ℤ) : 
  (∃ (count : ℤ), (∀ (x : ℤ), (100 < x^3 ∧ x^3 < 400) ↔ x = 5 ∨ x = 6 ∨ x = 7) ∧ (count = 3)) := 
sorry

end NUMINAMATH_GPT_number_of_perfect_cubes_l1956_195668


namespace NUMINAMATH_GPT_donovan_lap_time_l1956_195624

-- Definitions based on problem conditions
def lap_time_michael := 40  -- Michael's lap time in seconds
def laps_michael := 9       -- Laps completed by Michael to pass Donovan
def laps_donovan := 8       -- Laps completed by Donovan in the same time

-- Condition based on the solution
def race_duration := laps_michael * lap_time_michael

-- define the conjecture
theorem donovan_lap_time : 
  (race_duration = laps_donovan * 45) := 
sorry

end NUMINAMATH_GPT_donovan_lap_time_l1956_195624


namespace NUMINAMATH_GPT_simplify_polynomial_l1956_195690

theorem simplify_polynomial : 
  ∀ (x : ℝ), 
    (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1 
    = 32 * x ^ 5 := 
by sorry

end NUMINAMATH_GPT_simplify_polynomial_l1956_195690


namespace NUMINAMATH_GPT_ellipse_condition_l1956_195672

theorem ellipse_condition (m : ℝ) : 
  (∃ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) → m > 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_ellipse_condition_l1956_195672


namespace NUMINAMATH_GPT_fraction_inequality_solution_l1956_195670

theorem fraction_inequality_solution (x : ℝ) :
  (x < -5 ∨ x ≥ 2) ↔ (x-2) / (x+5) ≥ 0 :=
sorry

end NUMINAMATH_GPT_fraction_inequality_solution_l1956_195670


namespace NUMINAMATH_GPT_faith_weekly_earnings_l1956_195606

theorem faith_weekly_earnings :
  let hourly_pay := 13.50
  let regular_hours_per_day := 8
  let workdays_per_week := 5
  let overtime_hours_per_day := 2
  let regular_pay_per_day := hourly_pay * regular_hours_per_day
  let regular_pay_per_week := regular_pay_per_day * workdays_per_week
  let overtime_pay_per_day := hourly_pay * overtime_hours_per_day
  let overtime_pay_per_week := overtime_pay_per_day * workdays_per_week
  let total_weekly_earnings := regular_pay_per_week + overtime_pay_per_week
  total_weekly_earnings = 675 := 
  by
    sorry

end NUMINAMATH_GPT_faith_weekly_earnings_l1956_195606


namespace NUMINAMATH_GPT_part1_part2_l1956_195689

variable {A B C a b c : ℝ}

-- Part (1): Prove that 2a^2 = b^2 + c^2 given the condition
theorem part1 (h : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) : 2 * a^2 = b^2 + c^2 := 
sorry

-- Part (2): Prove the perimeter of triangle ABC
theorem part2 (a : ℝ) (h_a : a = 5) (cosA : ℝ) (h_cosA : cosA = 25 / 31) : 5 + b + c = 14 := 
sorry

end NUMINAMATH_GPT_part1_part2_l1956_195689


namespace NUMINAMATH_GPT_candy_distribution_l1956_195631

theorem candy_distribution (A B : ℕ) (h1 : 7 * A = B + 12) (h2 : 3 * A = B - 20) : A + B = 52 :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_candy_distribution_l1956_195631


namespace NUMINAMATH_GPT_larger_number_l1956_195673

theorem larger_number (L S : ℕ) (h1 : L - S = 1345) (h2 : L = 6 * S + 15) : L = 1611 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_l1956_195673


namespace NUMINAMATH_GPT_inequality_solution_l1956_195662

theorem inequality_solution (x y : ℝ) : y - x < abs x ↔ y < 0 ∨ y < 2 * x :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l1956_195662


namespace NUMINAMATH_GPT_set_diff_example_l1956_195654

-- Definitions of sets A and B
def A : Set ℕ := {2, 3}
def B : Set ℕ := {1, 3, 4}

-- Definition of set difference
def set_diff (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- The mathematically equivalent proof problem statement
theorem set_diff_example :
  set_diff A B = {2} :=
sorry

end NUMINAMATH_GPT_set_diff_example_l1956_195654


namespace NUMINAMATH_GPT_range_of_m_l1956_195613

noncomputable def f (x : ℝ) := Real.exp x * (x - 1)
noncomputable def g (m x : ℝ) := m * x

theorem range_of_m :
  (∀ x₁ ∈ Set.Icc (-2 : ℝ) 2, ∃ x₂ ∈ Set.Icc (1 : ℝ) 2, f x₁ > g m x₂) ↔ m ∈ Set.Iio (-1/2 : ℝ) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1956_195613


namespace NUMINAMATH_GPT_five_digit_number_count_l1956_195645

theorem five_digit_number_count : ∃ n, n = 1134 ∧ ∀ (a b c d e : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧ 
  (a < b ∧ b < c ∧ c > d ∧ d > e) → n = 1134 :=
by 
  sorry

end NUMINAMATH_GPT_five_digit_number_count_l1956_195645


namespace NUMINAMATH_GPT_circle_and_tangent_lines_l1956_195671

open Real

noncomputable def equation_of_circle_center_on_line (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), (x - a)^2 + (y - (a + 1))^2 = 2 ∧ (a = 4) ∧ (b = 5)

noncomputable def tangent_line_through_point (x y : ℝ) : Prop :=
  y = x - 1 ∨ y = (23 / 7) * x - (23 / 7)

theorem circle_and_tangent_lines :
  (∃ (a b : ℝ), (a = 4) ∧ (b = 5) ∧ (∀ x y : ℝ, equation_of_circle_center_on_line x y)) ∧
  (∀ x y : ℝ, tangent_line_through_point x y) := 
  by
  sorry

end NUMINAMATH_GPT_circle_and_tangent_lines_l1956_195671


namespace NUMINAMATH_GPT_reduced_population_l1956_195655

theorem reduced_population (initial_population : ℕ)
  (percentage_died : ℝ)
  (percentage_left : ℝ)
  (h_initial : initial_population = 8515)
  (h_died : percentage_died = 0.10)
  (h_left : percentage_left = 0.15) :
  ((initial_population - (⌊percentage_died * initial_population⌋₊ : ℕ)) - 
   (⌊percentage_left * (initial_population - (⌊percentage_died * initial_population⌋₊ : ℕ))⌋₊ : ℕ)) = 6515 :=
by
  sorry

end NUMINAMATH_GPT_reduced_population_l1956_195655


namespace NUMINAMATH_GPT_effective_price_l1956_195603

-- Definitions based on conditions
def upfront_payment (C : ℝ) := 0.20 * C = 240
def cashback (C : ℝ) := 0.10 * C

-- Problem statement
theorem effective_price (C : ℝ) (h₁ : upfront_payment C) : C - cashback C = 1080 :=
by
  sorry

end NUMINAMATH_GPT_effective_price_l1956_195603


namespace NUMINAMATH_GPT_complement_intersection_l1956_195626

open Set

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}
noncomputable def A : Set ℕ := {1, 2, 3}
noncomputable def B : Set ℕ := {3, 4, 5}

theorem complement_intersection (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {3, 4, 5}) :
  U \ (A ∩ B) = {1, 2, 4, 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1956_195626


namespace NUMINAMATH_GPT_max_daily_sales_revenue_l1956_195618

noncomputable def p (t : ℕ) : ℝ :=
if 0 < t ∧ t < 25 then t + 20
else if 25 ≤ t ∧ t ≤ 30 then -t + 70
else 0

noncomputable def Q (t : ℕ) : ℝ :=
if 0 < t ∧ t ≤ 30 then -t + 40 else 0

theorem max_daily_sales_revenue :
  ∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ (p t) * (Q t) = 1125 ∧
  ∀ t' : ℕ, 0 < t' ∧ t' ≤ 30 → (p t') * (Q t') ≤ 1125 :=
sorry

end NUMINAMATH_GPT_max_daily_sales_revenue_l1956_195618


namespace NUMINAMATH_GPT_total_buttons_l1956_195664

-- Define the conditions
def shirts_per_kid : Nat := 3
def number_of_kids : Nat := 3
def buttons_per_shirt : Nat := 7

-- Define the statement to prove
theorem total_buttons : shirts_per_kid * number_of_kids * buttons_per_shirt = 63 := by
  sorry

end NUMINAMATH_GPT_total_buttons_l1956_195664


namespace NUMINAMATH_GPT_simplify_expression_l1956_195687

variable (b : ℝ)

theorem simplify_expression (b : ℝ) : 
  (3 * b + 7 - 5 * b) / 3 = (-2 / 3) * b + (7 / 3) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1956_195687


namespace NUMINAMATH_GPT_sin_cos_sixth_power_sum_l1956_195653

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = Real.sqrt 2 / 2) : 
  (Real.sin θ)^6 + (Real.cos θ)^6 = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_sixth_power_sum_l1956_195653


namespace NUMINAMATH_GPT_factor_theorem_solution_l1956_195617

theorem factor_theorem_solution (t : ℝ) :
  (6 * t ^ 2 - 17 * t - 7 = 0) ↔ 
  (t = (17 + Real.sqrt 457) / 12 ∨ t = (17 - Real.sqrt 457) / 12) :=
by sorry

end NUMINAMATH_GPT_factor_theorem_solution_l1956_195617


namespace NUMINAMATH_GPT_total_candies_is_829_l1956_195639

-- Conditions as definitions
def Adam : ℕ := 6
def James : ℕ := 3 * Adam
def Rubert : ℕ := 4 * James
def Lisa : ℕ := 2 * Rubert
def Chris : ℕ := Lisa + 5
def Emily : ℕ := 3 * Chris - 7

-- Total candies
def total_candies : ℕ := Adam + James + Rubert + Lisa + Chris + Emily

-- Theorem to prove
theorem total_candies_is_829 : total_candies = 829 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_total_candies_is_829_l1956_195639


namespace NUMINAMATH_GPT_no_such_number_exists_l1956_195679

theorem no_such_number_exists : ¬ ∃ n : ℕ, 10^(n+1) + 35 ≡ 0 [MOD 63] :=
by {
  sorry 
}

end NUMINAMATH_GPT_no_such_number_exists_l1956_195679


namespace NUMINAMATH_GPT_contingency_fund_allocation_l1956_195604

theorem contingency_fund_allocation :
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  contingency_fund = 30 :=
by
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  show contingency_fund = 30
  sorry

end NUMINAMATH_GPT_contingency_fund_allocation_l1956_195604


namespace NUMINAMATH_GPT_find_g_3_8_l1956_195676

variable (g : ℝ → ℝ)
variable (x : ℝ)

-- Conditions
axiom g0 : g 0 = 0
axiom monotonicity (x y : ℝ) : 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry (x : ℝ) : 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling (x : ℝ) : 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- Statement to prove
theorem find_g_3_8 : g (3 / 8) = 2 / 9 := 
sorry

end NUMINAMATH_GPT_find_g_3_8_l1956_195676


namespace NUMINAMATH_GPT_inverse_proportion_graph_l1956_195623

theorem inverse_proportion_graph (k : ℝ) (x : ℝ) (y : ℝ) (h1 : y = k / x) (h2 : (3, -4) ∈ {p : ℝ × ℝ | p.snd = k / p.fst}) :
  k < 0 → ∀ x1 x2 : ℝ, x1 < x2 → y1 = k / x1 → y2 = k / x2 → y1 < y2 := by
  sorry

end NUMINAMATH_GPT_inverse_proportion_graph_l1956_195623


namespace NUMINAMATH_GPT_nth_equation_l1956_195611

theorem nth_equation (n : ℕ) : 2 * n * (2 * n + 2) + 1 = (2 * n + 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_nth_equation_l1956_195611


namespace NUMINAMATH_GPT_min_vertical_segment_length_l1956_195686

noncomputable def minVerticalSegLength : ℤ → ℝ 
| x => abs (2 * abs x + x^2 + 4 * x + 1)

theorem min_vertical_segment_length :
  ∀ x : ℤ, minVerticalSegLength x = 1 ↔  x = 0 := 
by
  intros x
  sorry

end NUMINAMATH_GPT_min_vertical_segment_length_l1956_195686


namespace NUMINAMATH_GPT_tom_tickets_left_l1956_195694

-- Define the conditions
def tickets_whack_a_mole : ℕ := 32
def tickets_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

-- Define what we need to prove
theorem tom_tickets_left : tickets_whack_a_mole + tickets_skee_ball - tickets_spent_on_hat = 50 :=
by sorry

end NUMINAMATH_GPT_tom_tickets_left_l1956_195694


namespace NUMINAMATH_GPT_range_of_2alpha_minus_beta_l1956_195693

def condition_range_alpha_beta (α β : ℝ) : Prop := 
  - (Real.pi / 2) < α ∧ α < β ∧ β < (Real.pi / 2)

theorem range_of_2alpha_minus_beta (α β : ℝ) (h : condition_range_alpha_beta α β) : 
  - Real.pi < 2 * α - β ∧ 2 * α - β < Real.pi / 2 :=
sorry

end NUMINAMATH_GPT_range_of_2alpha_minus_beta_l1956_195693


namespace NUMINAMATH_GPT_num_valid_values_n_l1956_195667

theorem num_valid_values_n :
  ∃ n : ℕ, (∃ a b c : ℕ,
    8 * a + 88 * b + 888 * c = 8880 ∧
    n = a + 2 * b + 3  * c) ∧
  (∃! k : ℕ, k = 119) :=
by sorry

end NUMINAMATH_GPT_num_valid_values_n_l1956_195667


namespace NUMINAMATH_GPT_find_four_consecutive_odd_numbers_l1956_195622

noncomputable def four_consecutive_odd_numbers (a b c d : ℤ) : Prop :=
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧
  (a = b + 2 ∨ a = b - 2) ∧ (b = c + 2 ∨ b = c - 2) ∧ (c = d + 2 ∨ c = d - 2)

def numbers_sum_to_26879 (a b c d : ℤ) : Prop :=
  1 + (a + b + c + d) +
  (a * b + a * c + a * d + b * c + b * d + c * d) +
  (a * b * c + a * b * d + a * c * d + b * c * d) +
  (a * b * c * d) = 26879

theorem find_four_consecutive_odd_numbers (a b c d : ℤ) :
  four_consecutive_odd_numbers a b c d ∧ numbers_sum_to_26879 a b c d →
  ((a, b, c, d) = (9, 11, 13, 15) ∨ (a, b, c, d) = (-17, -15, -13, -11)) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_four_consecutive_odd_numbers_l1956_195622


namespace NUMINAMATH_GPT_largest_4_digit_number_divisible_by_1615_l1956_195619

theorem largest_4_digit_number_divisible_by_1615 (X : ℕ) (hX: 8640 = 1615 * X) (h1: 1000 ≤ 1615 * X ∧ 1615 * X ≤ 9999) : X = 5 :=
by
  sorry

end NUMINAMATH_GPT_largest_4_digit_number_divisible_by_1615_l1956_195619


namespace NUMINAMATH_GPT_voting_problem_l1956_195691

theorem voting_problem (x y x' y' : ℕ) (m : ℕ) (h1 : x + y = 500) (h2 : y > x)
    (h3 : y - x = m) (h4 : x' = (10 * y) / 9) (h5 : x' + y' = 500)
    (h6 : x' - y' = 3 * m) :
    x' - x = 59 := 
sorry

end NUMINAMATH_GPT_voting_problem_l1956_195691


namespace NUMINAMATH_GPT_OneEmptyBox_NoBoxEmptyNoCompleteMatch_AtLeastTwoMatches_l1956_195643

def combination (n k : ℕ) : ℕ := Nat.choose n k
def arrangement (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem OneEmptyBox (n : ℕ) (hn : n = 5) : (combination 5 2) * (arrangement 5 5) = 1200 := by
  sorry

theorem NoBoxEmptyNoCompleteMatch (n : ℕ) (hn : n = 5) : (arrangement 5 5) - 1 = 119 := by
  sorry

theorem AtLeastTwoMatches (n : ℕ) (hn : n = 5) : (arrangement 5 5) - (combination 5 1 * 9 + 44) = 31 := by
  sorry

end NUMINAMATH_GPT_OneEmptyBox_NoBoxEmptyNoCompleteMatch_AtLeastTwoMatches_l1956_195643


namespace NUMINAMATH_GPT_final_coordinates_of_F_l1956_195647

-- Define the points D, E, F
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the initial points D, E, F
def D : Point := ⟨3, -4⟩
def E : Point := ⟨5, -1⟩
def F : Point := ⟨-2, -3⟩

-- Define the reflection over the y-axis
def reflect_over_y (p : Point) : Point := ⟨-p.x, p.y⟩

-- Define the reflection over the x-axis
def reflect_over_x (p : Point) : Point := ⟨p.x, -p.y⟩

-- First reflection over the y-axis
def F' : Point := reflect_over_y F

-- Second reflection over the x-axis
def F'' : Point := reflect_over_x F'

-- The proof problem
theorem final_coordinates_of_F'' :
  F'' = ⟨2, 3⟩ := 
sorry

end NUMINAMATH_GPT_final_coordinates_of_F_l1956_195647


namespace NUMINAMATH_GPT_sqrt_range_l1956_195640

theorem sqrt_range (x : ℝ) (h : 5 - x ≥ 0) : x ≤ 5 :=
sorry

end NUMINAMATH_GPT_sqrt_range_l1956_195640


namespace NUMINAMATH_GPT_union_of_sets_l1956_195652

variable (A : Set ℤ) (B : Set ℤ)

theorem union_of_sets (hA : A = {0, 1, 2}) (hB : B = {-1, 0}) : A ∪ B = {-1, 0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l1956_195652


namespace NUMINAMATH_GPT_alissa_presents_l1956_195600

def ethan_presents : ℝ := 31.0
def difference : ℝ := 22.0

theorem alissa_presents : ethan_presents - difference = 9.0 := by sorry

end NUMINAMATH_GPT_alissa_presents_l1956_195600


namespace NUMINAMATH_GPT_find_m_l1956_195658

theorem find_m (x y m : ℤ) (h1 : x = 1) (h2 : y = -1) (h3 : 2 * x + m + y = 0) : m = -1 := by
  -- Proof can be completed here
  sorry

end NUMINAMATH_GPT_find_m_l1956_195658


namespace NUMINAMATH_GPT_expression_divisible_by_11_l1956_195684

theorem expression_divisible_by_11 (n : ℕ) : (6^(2*n) + 3^(n+2) + 3^n) % 11 = 0 := 
sorry

end NUMINAMATH_GPT_expression_divisible_by_11_l1956_195684


namespace NUMINAMATH_GPT_a_2017_value_l1956_195601

theorem a_2017_value (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n : ℕ, S (n + 1) = 2 * (n + 1) - 1) :
  a 2017 = 2 :=
by
  sorry

end NUMINAMATH_GPT_a_2017_value_l1956_195601


namespace NUMINAMATH_GPT_boat_travel_distance_l1956_195678

theorem boat_travel_distance
  (D : ℝ) -- Distance traveled in both directions
  (t : ℝ) -- Time in hours it takes to travel upstream
  (speed_boat : ℝ) -- Speed of the boat in still water
  (speed_stream : ℝ) -- Speed of the stream
  (time_diff : ℝ) -- Difference in time between downstream and upstream travel
  (h1 : speed_boat = 10)
  (h2 : speed_stream = 2)
  (h3 : time_diff = 1.5)
  (h4 : D = 8 * t)
  (h5 : D = 12 * (t - time_diff)) :
  D = 36 := by
  sorry

end NUMINAMATH_GPT_boat_travel_distance_l1956_195678


namespace NUMINAMATH_GPT_tom_can_go_on_three_rides_l1956_195641

def rides_possible (total_tickets : ℕ) (spent_tickets : ℕ) (tickets_per_ride : ℕ) : ℕ :=
  (total_tickets - spent_tickets) / tickets_per_ride

theorem tom_can_go_on_three_rides :
  rides_possible 40 28 4 = 3 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_tom_can_go_on_three_rides_l1956_195641


namespace NUMINAMATH_GPT_Mary_bought_stickers_initially_l1956_195657

variable (S A M : ℕ) -- Define S, A, and M as natural numbers

-- Given conditions in the problem
def condition1 : Prop := S = A
def condition2 : Prop := M = 3 * A
def condition3 : Prop := A + (2 / 3) * M = 900

-- The theorem we need to prove
theorem Mary_bought_stickers_initially
  (h1 : condition1 S A)
  (h2 : condition2 A M)
  (h3 : condition3 A M)
  : S + A + M = 1500 :=
sorry -- Proof

end NUMINAMATH_GPT_Mary_bought_stickers_initially_l1956_195657


namespace NUMINAMATH_GPT_rotated_D_coords_l1956_195628

-- Definitions of the points used in the problem
def point (x y : ℤ) : ℤ × ℤ := (x, y)

-- Definitions of the vertices of the triangle DEF
def D : ℤ × ℤ := point 2 (-3)
def E : ℤ × ℤ := point 2 0
def F : ℤ × ℤ := point 5 (-3)

-- Definition of the rotation center
def center : ℤ × ℤ := point 3 (-2)

-- Function to rotate a point (x, y) by 180 degrees around (h, k)
def rotate_180 (p c : ℤ × ℤ) : ℤ × ℤ := 
  let (x, y) := p
  let (h, k) := c
  (2 * h - x, 2 * k - y)

-- Statement to prove the required coordinates after rotation
theorem rotated_D_coords : rotate_180 D center = point 4 (-1) :=
  sorry

end NUMINAMATH_GPT_rotated_D_coords_l1956_195628


namespace NUMINAMATH_GPT_fraction_zero_iff_x_is_four_l1956_195648

theorem fraction_zero_iff_x_is_four (x : ℝ) (h_ne_zero: x + 4 ≠ 0) :
  (16 - x^2) / (x + 4) = 0 ↔ x = 4 :=
sorry

end NUMINAMATH_GPT_fraction_zero_iff_x_is_four_l1956_195648


namespace NUMINAMATH_GPT_cos_sum_arithmetic_seq_l1956_195650

theorem cos_sum_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 + a 5 + a 9 = 5 * Real.pi) : 
  Real.cos (a 2 + a 8) = -1 / 2 :=
  sorry

end NUMINAMATH_GPT_cos_sum_arithmetic_seq_l1956_195650


namespace NUMINAMATH_GPT_index_card_area_reduction_index_card_area_when_other_side_shortened_l1956_195630

-- Conditions
def original_length := 4
def original_width := 6
def shortened_length := 2
def target_area := 12
def shortened_other_width := 5

-- Theorems to prove
theorem index_card_area_reduction :
  (original_length - 2) * original_width = target_area := by
  sorry

theorem index_card_area_when_other_side_shortened :
  (original_length) * (original_width - 1) = 20 := by
  sorry

end NUMINAMATH_GPT_index_card_area_reduction_index_card_area_when_other_side_shortened_l1956_195630


namespace NUMINAMATH_GPT_more_bottles_of_regular_soda_l1956_195607

theorem more_bottles_of_regular_soda (reg_soda diet_soda : ℕ) (h1 : reg_soda = 79) (h2 : diet_soda = 53) :
  reg_soda - diet_soda = 26 :=
by
  sorry

end NUMINAMATH_GPT_more_bottles_of_regular_soda_l1956_195607


namespace NUMINAMATH_GPT_jessica_watermelons_l1956_195677

theorem jessica_watermelons (original : ℕ) (eaten : ℕ) (remaining : ℕ) 
    (h1 : original = 35) 
    (h2 : eaten = 27) 
    (h3 : remaining = original - eaten) : 
  remaining = 8 := 
by {
    -- This is where the proof would go
    sorry
}

end NUMINAMATH_GPT_jessica_watermelons_l1956_195677


namespace NUMINAMATH_GPT_greatest_integer_value_l1956_195602

theorem greatest_integer_value (x : ℤ) (h : ∃ x : ℤ, x = 29 ∧ ∀ x : ℤ, (x ≠ 3 → ∃ k : ℤ, (x^2 + 3*x + 8) = (x-3)*(x+6) + 26)) :
  (∀ x : ℤ, (x ≠ 3 → ∃ k : ℤ, (x^2 + 3*x + 8) = (x-3)*k + 26) → x = 29) :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_value_l1956_195602


namespace NUMINAMATH_GPT_Sam_has_38_dollars_l1956_195615

theorem Sam_has_38_dollars (total_money erica_money sam_money : ℕ) 
  (h1 : total_money = 91)
  (h2 : erica_money = 53) 
  (h3 : total_money = erica_money + sam_money) : 
  sam_money = 38 := 
by 
  sorry

end NUMINAMATH_GPT_Sam_has_38_dollars_l1956_195615


namespace NUMINAMATH_GPT_initial_number_of_numbers_is_five_l1956_195627

-- Define the conditions and the given problem
theorem initial_number_of_numbers_is_five
  (n : ℕ) (S : ℕ)
  (h1 : S / n = 27)
  (h2 : (S - 35) / (n - 1) = 25) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_numbers_is_five_l1956_195627


namespace NUMINAMATH_GPT_red_users_count_l1956_195674

noncomputable def total_students : ℕ := 70
noncomputable def green_users : ℕ := 52
noncomputable def both_colors_users : ℕ := 38

theorem red_users_count : 
  ∀ (R : ℕ), total_students = green_users + R - both_colors_users → R = 56 :=
by
  sorry

end NUMINAMATH_GPT_red_users_count_l1956_195674


namespace NUMINAMATH_GPT_max_chord_length_l1956_195659

theorem max_chord_length (x1 y1 x2 y2 : ℝ) (h_parabola1 : x1^2 = 8 * y1) (h_parabola2 : x2^2 = 8 * y2)
  (h_midpoint_ordinate : (y1 + y2) / 2 = 4) :
  abs ((y1 + y2) + 4) = 12 :=
by
  sorry

end NUMINAMATH_GPT_max_chord_length_l1956_195659


namespace NUMINAMATH_GPT_max_food_cost_l1956_195612

theorem max_food_cost (total_cost : ℝ) (food_cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (max_allowable : ℝ)
  (h1 : tax_rate = 0.07) (h2 : tip_rate = 0.15) (h3 : max_allowable = 75) (h4 : total_cost = food_cost * (1 + tax_rate + tip_rate)) :
  food_cost ≤ 61.48 :=
sorry

end NUMINAMATH_GPT_max_food_cost_l1956_195612


namespace NUMINAMATH_GPT_allen_total_blocks_l1956_195608

/-- 
  If there are 7 blocks for every color of paint used and Shiela used 7 colors, 
  then the total number of blocks Allen has is 49.
-/
theorem allen_total_blocks
  (blocks_per_color : ℕ) 
  (number_of_colors : ℕ)
  (h1 : blocks_per_color = 7) 
  (h2 : number_of_colors = 7) : 
  blocks_per_color * number_of_colors = 49 := 
by 
  sorry

end NUMINAMATH_GPT_allen_total_blocks_l1956_195608


namespace NUMINAMATH_GPT_total_balloons_correct_l1956_195688

-- Define the number of balloons each person has
def dan_balloons : ℕ := 29
def tim_balloons : ℕ := 7 * dan_balloons
def molly_balloons : ℕ := 5 * dan_balloons

-- Define the total number of balloons
def total_balloons : ℕ := dan_balloons + tim_balloons + molly_balloons

-- The theorem to prove
theorem total_balloons_correct : total_balloons = 377 :=
by
  -- This part is where the proof will go
  sorry

end NUMINAMATH_GPT_total_balloons_correct_l1956_195688


namespace NUMINAMATH_GPT_verify_statements_l1956_195605

theorem verify_statements (S : Set ℝ) (m l : ℝ) (hS : ∀ x, x ∈ S → x^2 ∈ S) :
  (m = 1 → S = {1}) ∧
  (m = -1/2 → (1/4 ≤ l ∧ l ≤ 1)) ∧
  (l = 1/2 → -Real.sqrt 2 / 2 ≤ m ∧ m ≤ 0) ∧
  (l = 1 → -1 ≤ m ∧ m ≤ 1) :=
  sorry

end NUMINAMATH_GPT_verify_statements_l1956_195605
