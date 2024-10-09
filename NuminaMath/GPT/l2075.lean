import Mathlib

namespace digit_in_tens_place_l2075_207504

theorem digit_in_tens_place (n : ℕ) (cycle : List ℕ) (h_cycle : cycle = [16, 96, 76, 56]) (hk : n % 4 = 3) :
  (6 ^ n % 100) / 10 % 10 = 7 := by
  sorry

end digit_in_tens_place_l2075_207504


namespace min_days_is_9_l2075_207569

theorem min_days_is_9 (n : ℕ) (rain_morning rain_afternoon sunny_morning sunny_afternoon : ℕ)
  (h1 : rain_morning + rain_afternoon = 7)
  (h2 : rain_afternoon ≤ sunny_morning)
  (h3 : sunny_afternoon = 5)
  (h4 : sunny_morning = 6) :
  n ≥ 9 :=
sorry

end min_days_is_9_l2075_207569


namespace percentage_of_dogs_l2075_207552

theorem percentage_of_dogs (total_pets : ℕ) (percent_cats : ℕ) (bunnies : ℕ) 
  (h1 : total_pets = 36) (h2 : percent_cats = 50) (h3 : bunnies = 9) : 
  ((total_pets - ((percent_cats * total_pets) / 100) - bunnies) / total_pets * 100) = 25 := by
  sorry

end percentage_of_dogs_l2075_207552


namespace quadratic_function_equal_values_l2075_207549

theorem quadratic_function_equal_values (a m n : ℝ) (h : a ≠ 0) (hmn : a * m^2 - 4 * a * m - 3 = a * n^2 - 4 * a * n - 3) : m + n = 4 :=
by
  sorry

end quadratic_function_equal_values_l2075_207549


namespace total_water_intake_l2075_207513

def morning_water : ℝ := 1.5
def afternoon_water : ℝ := 3 * morning_water
def evening_water : ℝ := 0.5 * afternoon_water

theorem total_water_intake : 
  (morning_water + afternoon_water + evening_water) = 8.25 :=
by
  sorry

end total_water_intake_l2075_207513


namespace percentage_with_diploma_l2075_207574

-- Define the percentages as variables for clarity
def low_income_perc := 0.25
def lower_middle_income_perc := 0.35
def upper_middle_income_perc := 0.25
def high_income_perc := 0.15

def low_income_diploma := 0.05
def lower_middle_income_diploma := 0.35
def upper_middle_income_diploma := 0.60
def high_income_diploma := 0.80

theorem percentage_with_diploma :
  (low_income_perc * low_income_diploma +
   lower_middle_income_perc * lower_middle_income_diploma +
   upper_middle_income_perc * upper_middle_income_diploma +
   high_income_perc * high_income_diploma) = 0.405 :=
by sorry

end percentage_with_diploma_l2075_207574


namespace kenneth_money_left_l2075_207570

noncomputable def baguettes : ℝ := 2 * 2
noncomputable def water : ℝ := 2 * 1

noncomputable def chocolate_bars_cost_before_discount : ℝ := 2 * 1.5
noncomputable def chocolate_bars_cost_after_discount : ℝ := chocolate_bars_cost_before_discount * (1 - 0.20)
noncomputable def chocolate_bars_final_cost : ℝ := chocolate_bars_cost_after_discount * 1.08

noncomputable def milk_cost_after_discount : ℝ := 3.5 * (1 - 0.10)

noncomputable def chips_cost_before_tax : ℝ := 2.5 + (2.5 * 0.50)
noncomputable def chips_final_cost : ℝ := chips_cost_before_tax * 1.08

noncomputable def total_cost : ℝ :=
  baguettes + water + chocolate_bars_final_cost + milk_cost_after_discount + chips_final_cost

noncomputable def initial_amount : ℝ := 50
noncomputable def amount_left : ℝ := initial_amount - total_cost

theorem kenneth_money_left : amount_left = 50 - 15.792 := by
  sorry

end kenneth_money_left_l2075_207570


namespace triangle_overlap_angle_is_30_l2075_207501

noncomputable def triangle_rotation_angle (hypotenuse : ℝ) (overlap_ratio : ℝ) :=
  if hypotenuse = 10 ∧ overlap_ratio = 0.5 then 30 else sorry

theorem triangle_overlap_angle_is_30 :
  triangle_rotation_angle 10 0.5 = 30 :=
sorry

end triangle_overlap_angle_is_30_l2075_207501


namespace necessary_condition_l2075_207562

theorem necessary_condition :
  (∀ x : ℝ, (1 / x < 3) → (x > 1 / 3)) → (∀ x : ℝ, (1 / x < 3) ↔ (x > 1 / 3)) → False :=
by
  sorry

end necessary_condition_l2075_207562


namespace find_x_average_is_60_l2075_207524

theorem find_x_average_is_60 : 
  ∃ x : ℕ, (54 + 55 + 57 + 58 + 59 + 62 + 62 + 63 + x) / 9 = 60 ∧ x = 70 :=
by
  existsi 70
  sorry

end find_x_average_is_60_l2075_207524


namespace line_graph_displays_trend_l2075_207595

-- Define the types of statistical graphs
inductive StatisticalGraph : Type
| barGraph : StatisticalGraph
| lineGraph : StatisticalGraph
| pieChart : StatisticalGraph
| histogram : StatisticalGraph

-- Define the property of displaying trends over time
def displaysTrend (g : StatisticalGraph) : Prop := 
  g = StatisticalGraph.lineGraph

-- Theorem to prove that the type of statistical graph that displays the trend of data is the line graph
theorem line_graph_displays_trend : displaysTrend StatisticalGraph.lineGraph :=
sorry

end line_graph_displays_trend_l2075_207595


namespace compute_fg_l2075_207558

def f (x : ℤ) : ℤ := x * x
def g (x : ℤ) : ℤ := 3 * x + 4

theorem compute_fg : f (g (-3)) = 25 := by
  sorry

end compute_fg_l2075_207558


namespace trajectory_of_circle_center_l2075_207565

open Real

noncomputable def circle_trajectory_equation (x y : ℝ) : Prop :=
  (y ^ 2 = 8 * x - 16)

theorem trajectory_of_circle_center (x y : ℝ) :
  (∃ C : ℝ × ℝ, (C.1 = 4 ∧ C.2 = 0) ∧
    (∃ MN : ℝ × ℝ, (MN.1 = 0 ∧ MN.2 ^ 2 = 64) ∧
    (x = C.1 ∧ y = C.2)) ∧
    circle_trajectory_equation x y) :=
sorry

end trajectory_of_circle_center_l2075_207565


namespace trader_profit_l2075_207539

theorem trader_profit (P : ℝ) (hP : 0 < P) : 
  let purchase_price := 0.80 * P
  let selling_price := 1.36 * P
  let profit := selling_price - P
  (profit / P) * 100 = 36 :=
by
  -- The proof will go here
  sorry

end trader_profit_l2075_207539


namespace linear_regression_change_l2075_207576

theorem linear_regression_change : ∀ (x : ℝ), ∀ (y : ℝ), 
  y = 2 - 3.5 * x → (y - (2 - 3.5 * (x + 1))) = 3.5 :=
by
  intros x y h
  sorry

end linear_regression_change_l2075_207576


namespace sale_in_fifth_month_l2075_207525

theorem sale_in_fifth_month (sale1 sale2 sale3 sale4 sale6 : ℕ) (avg : ℕ) (months : ℕ) (total_sales : ℕ)
    (known_sales : sale1 = 6335 ∧ sale2 = 6927 ∧ sale3 = 6855 ∧ sale4 = 7230 ∧ sale6 = 5091)
    (avg_condition : avg = 6500)
    (months_condition : months = 6)
    (total_sales_condition : total_sales = avg * months) :
    total_sales - (sale1 + sale2 + sale3 + sale4 + sale6) = 6562 :=
by
  sorry

end sale_in_fifth_month_l2075_207525


namespace power_of_three_l2075_207553

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l2075_207553


namespace exists_colored_right_triangle_l2075_207596

theorem exists_colored_right_triangle (color : ℝ × ℝ → ℕ) 
  (h_nonempty_blue  : ∃ p, color p = 0)
  (h_nonempty_green : ∃ p, color p = 1)
  (h_nonempty_red   : ∃ p, color p = 2) :
  ∃ p1 p2 p3 : ℝ × ℝ, 
    (p1 ≠ p2) ∧ (p2 ≠ p3) ∧ (p1 ≠ p3) ∧ 
    ((color p1 = 0) ∧ (color p2 = 1) ∧ (color p3 = 2) ∨ 
     (color p1 = 0) ∧ (color p2 = 2) ∧ (color p3 = 1) ∨ 
     (color p1 = 1) ∧ (color p2 = 0) ∧ (color p3 = 2) ∨ 
     (color p1 = 1) ∧ (color p2 = 2) ∧ (color p3 = 0) ∨ 
     (color p1 = 2) ∧ (color p2 = 0) ∧ (color p3 = 1) ∨ 
     (color p1 = 2) ∧ (color p2 = 1) ∧ (color p3 = 0))
  ∧ ((p1.1 = p2.1 ∧ p2.2 = p3.2) ∨ (p1.2 = p2.2 ∧ p2.1 = p3.1)) :=
sorry

end exists_colored_right_triangle_l2075_207596


namespace problem1_problem2_problem3_l2075_207500

variables (a b c : ℝ)

-- First proof problem
theorem problem1 (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) : a * b * c ≠ 0 :=
sorry

-- Second proof problem
theorem problem2 (h : a = 0 ∨ b = 0 ∨ c = 0) : a * b * c = 0 :=
sorry

-- Third proof problem
theorem problem3 (h : a * b < 0 ∨ a = 0 ∨ b = 0) : a * b ≤ 0 :=
sorry

end problem1_problem2_problem3_l2075_207500


namespace work_days_of_A_and_B_l2075_207526

theorem work_days_of_A_and_B (B : ℝ) (A : ℝ) (h1 : A = 2 * B) (h2 : B = 1 / 27) :
  1 / (A + B) = 9 :=
by
  sorry

end work_days_of_A_and_B_l2075_207526


namespace solve_eqn_in_integers_l2075_207507

theorem solve_eqn_in_integers :
  ∃ (x y : ℤ), xy + 3*x - 5*y = -3 ∧ 
  ((x, y) = (6, 9) ∨ (x, y) = (7, 3) ∨ (x, y) = (8, 1) ∨ 
  (x, y) = (9, 0) ∨ (x, y) = (11, -1) ∨ (x, y) = (17, -2) ∨ 
  (x, y) = (4, -15) ∨ (x, y) = (3, -9) ∨ (x, y) = (2, -7) ∨ 
  (x, y) = (1, -6) ∨ (x, y) = (-1, -5) ∨  (x, y) = (-7, -4)) :=
sorry

end solve_eqn_in_integers_l2075_207507


namespace solve_for_y_l2075_207543

theorem solve_for_y (y : ℕ) (h : 9^y = 3^12) : y = 6 :=
by {
  sorry
}

end solve_for_y_l2075_207543


namespace tan_expression_val_l2075_207581

theorem tan_expression_val (A B : ℝ) (hA : A = 30) (hB : B = 15) :
  (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2 :=
by
  sorry

end tan_expression_val_l2075_207581


namespace image_of_neg2_3_preimages_2_neg3_l2075_207546

variables {A B : Type}
def f (x y : ℤ) : ℤ × ℤ := (x + y, x * y)

-- Prove that the image of (-2, 3) under f is (1, -6)
theorem image_of_neg2_3 : f (-2) 3 = (1, -6) := sorry

-- Find the preimages of (2, -3) under f
def preimages_of_2_neg3 (p : ℤ × ℤ) : Prop := f p.1 p.2 = (2, -3)

theorem preimages_2_neg3 : preimages_of_2_neg3 (-1, 3) ∧ preimages_of_2_neg3 (3, -1) := sorry

end image_of_neg2_3_preimages_2_neg3_l2075_207546


namespace division_expression_evaluation_l2075_207584

theorem division_expression_evaluation : 120 / (6 / 2) = 40 := by
  sorry

end division_expression_evaluation_l2075_207584


namespace intersection_A_B_l2075_207554

def setA (x : ℝ) : Prop := 3 * x + 2 > 0
def setB (x : ℝ) : Prop := (x + 1) * (x - 3) > 0
def A : Set ℝ := { x | setA x }
def B : Set ℝ := { x | setB x }

theorem intersection_A_B : A ∩ B = { x | 3 < x } := by
  sorry

end intersection_A_B_l2075_207554


namespace polygon_sides_l2075_207575

theorem polygon_sides (n : ℕ) (h₁ : ∀ (m : ℕ), m = n → n > 2) (h₂ : 180 * (n - 2) = 156 * n) : n = 15 :=
by
  sorry

end polygon_sides_l2075_207575


namespace doug_initial_marbles_l2075_207594

theorem doug_initial_marbles (E D : ℕ) (H1 : E = D + 5) (H2 : E = 27) : D = 22 :=
by
  -- proof provided here would infer the correct answer from the given conditions
  sorry

end doug_initial_marbles_l2075_207594


namespace game_show_prizes_l2075_207547

theorem game_show_prizes :
  let digits := [1, 1, 2, 2, 3, 3, 3, 3]
  let permutations := Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 2 * Nat.factorial 2)
  let partitions := Nat.choose 7 3
  permutations * partitions = 14700 :=
by
  let digits := [1, 1, 2, 2, 3, 3, 3, 3]
  let permutations := Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 2 * Nat.factorial 2)
  let partitions := Nat.choose 7 3
  exact sorry

end game_show_prizes_l2075_207547


namespace not_divisible_by_n_plus_4_l2075_207566

theorem not_divisible_by_n_plus_4 (n : ℕ) (h : n > 0) : ¬ ∃ k : ℕ, n^2 + 8 * n + 15 = k * (n + 4) := by
  sorry

end not_divisible_by_n_plus_4_l2075_207566


namespace net_income_after_tax_l2075_207579

theorem net_income_after_tax (gross_income : ℝ) (tax_rate : ℝ) : 
  (gross_income = 45000) → (tax_rate = 0.13) → 
  (gross_income - gross_income * tax_rate = 39150) :=
by
  intro h1 h2
  rw [h1, h2]
  sorry

end net_income_after_tax_l2075_207579


namespace triangular_prism_distance_sum_l2075_207586

theorem triangular_prism_distance_sum (V K H1 H2 H3 H4 S1 S2 S3 S4 : ℝ)
  (h1 : S1 = K)
  (h2 : S2 = 2 * K)
  (h3 : S3 = 3 * K)
  (h4 : S4 = 4 * K)
  (hV : (S1 * H1 + S2 * H2 + S3 * H3 + S4 * H4) / 3 = V) :
  H1 + 2 * H2 + 3 * H3 + 4 * H4 = 3 * V / K :=
by sorry

end triangular_prism_distance_sum_l2075_207586


namespace tan_15pi_over_4_is_neg1_l2075_207590

noncomputable def tan_15pi_over_4 : ℝ :=
  Real.tan (15 * Real.pi / 4)

theorem tan_15pi_over_4_is_neg1 :
  tan_15pi_over_4 = -1 :=
sorry

end tan_15pi_over_4_is_neg1_l2075_207590


namespace common_difference_is_4_l2075_207542

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Defining the arithmetic sequence {a_n}
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
variable (d : ℤ) (a4_a5_sum : a 4 + a 5 = 24) (S6_val : S 6 = 48)

-- Statement to prove: given the conditions, d = 4
theorem common_difference_is_4 (h_seq : is_arithmetic_sequence a d) :
  d = 4 := sorry

end common_difference_is_4_l2075_207542


namespace MarkBenchPressAmount_l2075_207537

def DaveWeight : ℝ := 175
def DaveBenchPressMultiplier : ℝ := 3
def CraigBenchPressFraction : ℝ := 0.20
def MarkDeficitFromCraig : ℝ := 50

theorem MarkBenchPressAmount : 
  let DaveBenchPress := DaveWeight * DaveBenchPressMultiplier
  let CraigBenchPress := DaveBenchPress * CraigBenchPressFraction
  let MarkBenchPress := CraigBenchPress - MarkDeficitFromCraig
  MarkBenchPress = 55 := by
  let DaveBenchPress := DaveWeight * DaveBenchPressMultiplier
  let CraigBenchPress := DaveBenchPress * CraigBenchPressFraction
  let MarkBenchPress := CraigBenchPress - MarkDeficitFromCraig
  sorry

end MarkBenchPressAmount_l2075_207537


namespace bakery_regular_price_l2075_207556

theorem bakery_regular_price (y : ℝ) (h₁ : y / 4 * 0.4 = 2) : y = 20 :=
by {
  sorry
}

end bakery_regular_price_l2075_207556


namespace parallel_lines_sufficient_condition_l2075_207583

theorem parallel_lines_sufficient_condition :
  ∀ a : ℝ, (a^2 - a) = 2 → (a = 2 ∨ a = -1) :=
by
  intro a h
  sorry

end parallel_lines_sufficient_condition_l2075_207583


namespace more_regular_than_diet_l2075_207502

-- Define the conditions
def num_regular_soda : Nat := 67
def num_diet_soda : Nat := 9

-- State the theorem
theorem more_regular_than_diet :
  num_regular_soda - num_diet_soda = 58 :=
by
  sorry

end more_regular_than_diet_l2075_207502


namespace square_101_l2075_207515

theorem square_101:
  (101 : ℕ)^2 = 10201 :=
by
  sorry

end square_101_l2075_207515


namespace largest_expression_l2075_207518

theorem largest_expression :
  let A := 0.9387
  let B := 0.9381
  let C := 9385 / 10000
  let D := 0.9379
  let E := 0.9389
  E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  let A := 0.9387
  let B := 0.9381
  let C := 9385 / 10000
  let D := 0.9379
  let E := 0.9389
  sorry

end largest_expression_l2075_207518


namespace parabola_y_intercepts_zero_l2075_207555

-- Define the quadratic equation
def quadratic (a b c y: ℝ) : ℝ := a * y^2 + b * y + c

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Condition: equation of the parabola and discriminant calculation
def parabola_equation : Prop := 
  let a := 3
  let b := -4
  let c := 5
  discriminant a b c < 0

-- Statement to prove
theorem parabola_y_intercepts_zero : 
  (parabola_equation) → (∀ y : ℝ, quadratic 3 (-4) 5 y ≠ 0) :=
by
  intro h
  sorry

end parabola_y_intercepts_zero_l2075_207555


namespace imaginary_part_of_z_l2075_207589

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + I) = 2 - 2 * I) : z.im = -2 :=
sorry

end imaginary_part_of_z_l2075_207589


namespace jesse_mia_total_miles_per_week_l2075_207573

noncomputable def jesse_miles_per_day_first_three := 2 / 3
noncomputable def jesse_miles_day_four := 10
noncomputable def mia_miles_per_day_first_four := 3
noncomputable def average_final_three_days := 6

theorem jesse_mia_total_miles_per_week :
  let jesse_total_first_four_days := 3 * jesse_miles_per_day_first_three + jesse_miles_day_four
  let mia_total_first_four_days := 4 * mia_miles_per_day_first_four
  let total_miles_needed_final_three_days := 3 * average_final_three_days * 2
  jesse_total_first_four_days + total_miles_needed_final_three_days = 48 ∧
  mia_total_first_four_days + total_miles_needed_final_three_days = 48 :=
by
  sorry

end jesse_mia_total_miles_per_week_l2075_207573


namespace num_children_in_family_l2075_207593

def regular_ticket_cost := 15
def elderly_ticket_cost := 10
def adult_ticket_cost := 12
def child_ticket_cost := adult_ticket_cost - 5
def total_money_handled := 3 * 50
def change_received := 3
def num_adults := 4
def num_elderly := 2
def total_cost_for_adults := num_adults * adult_ticket_cost
def total_cost_for_elderly := num_elderly * elderly_ticket_cost
def total_cost_of_tickets := total_money_handled - change_received

theorem num_children_in_family : ∃ (num_children : ℕ), 
  total_cost_of_tickets = total_cost_for_adults + total_cost_for_elderly + num_children * child_ticket_cost ∧ 
  num_children = 11 := 
by
  sorry

end num_children_in_family_l2075_207593


namespace joes_speed_second_part_l2075_207527

theorem joes_speed_second_part
  (d1 d2 t1 t_total: ℝ)
  (s1 s_avg: ℝ)
  (h_d1: d1 = 420)
  (h_d2: d2 = 120)
  (h_s1: s1 = 60)
  (h_s_avg: s_avg = 54) :
  (d1 / s1 + d2 / (d2 / 40) = t_total ∧ t_total = (d1 + d2) / s_avg) →
  d2 / (t_total - d1 / s1) = 40 :=
by
  sorry

end joes_speed_second_part_l2075_207527


namespace expected_sixes_correct_l2075_207536

-- Define probabilities for rolling individual numbers on a die
def P (n : ℕ) (k : ℕ) : ℚ := if k = n then 1 / 6 else 0

-- Expected value calculation for two dice
noncomputable def expected_sixes_two_dice_with_resets : ℚ :=
(0 * (13/18)) + (1 * (2/9)) + (2 * (1/36))

-- Main theorem to prove
theorem expected_sixes_correct :
  expected_sixes_two_dice_with_resets = 5 / 18 :=
by
  -- The actual proof steps go here; added sorry to skip the proof.
  sorry

end expected_sixes_correct_l2075_207536


namespace max_m_x_range_l2075_207517

variables {a b x : ℝ}

theorem max_m (h1 : a * b > 0) (h2 : a^2 * b = 4) : 
  a + b ≥ 3 :=
sorry

theorem x_range (h : 2 * |x - 1| + |x| ≤ 3) : 
  -1/3 ≤ x ∧ x ≤ 5/3 :=
sorry

end max_m_x_range_l2075_207517


namespace number_of_pencils_broken_l2075_207503

theorem number_of_pencils_broken
  (initial_pencils : ℕ)
  (misplaced_pencils : ℕ)
  (found_pencils : ℕ)
  (bought_pencils : ℕ)
  (final_pencils : ℕ)
  (h_initial : initial_pencils = 20)
  (h_misplaced : misplaced_pencils = 7)
  (h_found : found_pencils = 4)
  (h_bought : bought_pencils = 2)
  (h_final : final_pencils = 16) :
  (initial_pencils - misplaced_pencils + found_pencils + bought_pencils - final_pencils) = 3 := 
by
  sorry

end number_of_pencils_broken_l2075_207503


namespace largest_even_integer_product_l2075_207545

theorem largest_even_integer_product (n : ℕ) (h : 2 * n * (2 * n + 2) * (2 * n + 4) * (2 * n + 6) = 5040) :
  2 * n + 6 = 20 :=
by
  sorry

end largest_even_integer_product_l2075_207545


namespace count_multiples_less_than_300_l2075_207599

theorem count_multiples_less_than_300 : ∀ n : ℕ, n < 300 → (2 * 3 * 5 * 7 ∣ n) ↔ n = 210 :=
by
  sorry

end count_multiples_less_than_300_l2075_207599


namespace problem_statement_l2075_207522

def sequence_arithmetic (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (a (n+1) / 2^(n+1) - a n / 2^n = 1)

theorem problem_statement : 
  ∃ a : ℕ → ℝ, a 1 = 2 ∧ a 2 = 8 ∧ (∀ n : ℕ, n ≥ 1 → a (n+1) - 2 * a n = 2^(n+1)) → sequence_arithmetic a :=
by
  sorry

end problem_statement_l2075_207522


namespace pentagon_area_pq_sum_l2075_207531

theorem pentagon_area_pq_sum 
  (p q : ℤ) 
  (hp : 0 < q ∧ q < p) 
  (harea : 5 * p * q - q * q = 700) : 
  ∃ sum : ℤ, sum = p + q :=
by
  sorry

end pentagon_area_pq_sum_l2075_207531


namespace sum_of_distinct_roots_eq_zero_l2075_207529

theorem sum_of_distinct_roots_eq_zero
  (a b m n p : ℝ)
  (h1 : m ≠ n)
  (h2 : m ≠ p)
  (h3 : n ≠ p)
  (h_m : m^3 + a * m + b = 0)
  (h_n : n^3 + a * n + b = 0)
  (h_p : p^3 + a * p + b = 0) : 
  m + n + p = 0 :=
sorry

end sum_of_distinct_roots_eq_zero_l2075_207529


namespace common_tangent_theorem_l2075_207551

-- Define the first circle with given equation (x+2)^2 + (y-2)^2 = 1
def circle1 (x y : ℝ) : Prop := (x + 2)^2 + (y - 2)^2 = 1

-- Define the second circle with given equation (x-2)^2 + (y-5)^2 = 16
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 16

-- Define a predicate that expresses the concept of common tangents between two circles
def common_tangents_count (circle1 circle2 : ℝ → ℝ → Prop) : ℕ := sorry

-- The statement to prove that the number of common tangents is 3
theorem common_tangent_theorem : common_tangents_count circle1 circle2 = 3 :=
by
  -- We would proceed with the proof if required, but we end with sorry as requested.
  sorry

end common_tangent_theorem_l2075_207551


namespace first_installment_amount_l2075_207578

-- Define the conditions stated in the problem
def original_price : ℝ := 480
def discount_rate : ℝ := 0.05
def monthly_installment : ℝ := 102
def number_of_installments : ℕ := 3

-- The final price after discount
def final_price : ℝ := original_price * (1 - discount_rate)

-- The total amount of the 3 monthly installments
def total_of_3_installments : ℝ := monthly_installment * number_of_installments

-- The first installment paid
def first_installment : ℝ := final_price - total_of_3_installments

-- The main theorem to prove the first installment amount
theorem first_installment_amount : first_installment = 150 := by
  unfold first_installment
  unfold final_price
  unfold total_of_3_installments
  unfold original_price
  unfold discount_rate
  unfold monthly_installment
  unfold number_of_installments
  sorry

end first_installment_amount_l2075_207578


namespace nylon_cord_length_l2075_207598

-- Let the length of cord be w
-- Dog runs 30 feet forming a semicircle, that is pi * w = 30
-- Prove that w is approximately 9.55

theorem nylon_cord_length (pi_approx : Real := 3.14) : Real :=
  let w := 30 / pi_approx
  w

end nylon_cord_length_l2075_207598


namespace intersection_P_Q_l2075_207534

noncomputable def P : Set ℝ := { x | x < 1 }
noncomputable def Q : Set ℝ := { x | x^2 < 4 }

theorem intersection_P_Q :
  P ∩ Q = { x | -2 < x ∧ x < 1 } :=
by 
  sorry

end intersection_P_Q_l2075_207534


namespace gcd_of_75_and_360_l2075_207530

theorem gcd_of_75_and_360 : Nat.gcd 75 360 = 15 := by
  sorry

end gcd_of_75_and_360_l2075_207530


namespace external_tangent_b_value_l2075_207597

theorem external_tangent_b_value:
  ∀ {C1 C2 : ℝ × ℝ} (r1 r2 : ℝ) (m b : ℝ),
  C1 = (3, -2) ∧ r1 = 3 ∧ 
  C2 = (15, 8) ∧ r2 = 8 ∧
  m = (60 / 11) →
  (∃ b, y = m * x + b ∧ b = 720 / 11) :=
by 
  sorry

end external_tangent_b_value_l2075_207597


namespace rotated_clockwise_120_correct_l2075_207588

-- Problem setup definitions
structure ShapePosition :=
  (triangle : Point)
  (smaller_circle : Point)
  (square : Point)

-- Conditions for the initial positions of the shapes
variable (initial : ShapePosition)

def rotated_positions (initial: ShapePosition) : ShapePosition :=
  { 
    triangle := initial.smaller_circle,
    smaller_circle := initial.square,
    square := initial.triangle 
  }

-- Problem statement: show that after a 120° clockwise rotation, 
-- the shapes move to the specified new positions.
theorem rotated_clockwise_120_correct (initial : ShapePosition) 
  (after_rotation : ShapePosition) :
  after_rotation = rotated_positions initial := 
sorry

end rotated_clockwise_120_correct_l2075_207588


namespace connie_total_markers_l2075_207521

/-
Connie has 4 different types of markers: red, blue, green, and yellow.
She has twice as many red markers as green markers.
She has three times as many blue markers as red markers.
She has four times as many yellow markers as green markers.
She has 36 green markers.
Prove that the total number of markers she has is 468.
-/

theorem connie_total_markers
 (g r b y : ℕ) 
 (hg : g = 36) 
 (hr : r = 2 * g)
 (hb : b = 3 * r)
 (hy : y = 4 * g) :
 g + r + b + y = 468 := 
 by
  sorry

end connie_total_markers_l2075_207521


namespace NaCl_moles_formed_l2075_207544

-- Definitions for the conditions
def NaOH_moles : ℕ := 2
def Cl2_moles : ℕ := 1

-- Chemical reaction of NaOH and Cl2 resulting in NaCl and H2O
def reaction (n_NaOH n_Cl2 : ℕ) : ℕ :=
  if n_NaOH = 2 ∧ n_Cl2 = 1 then 2 else 0

-- Statement to be proved
theorem NaCl_moles_formed : reaction NaOH_moles Cl2_moles = 2 :=
by
  sorry

end NaCl_moles_formed_l2075_207544


namespace crocus_bulb_cost_l2075_207510

theorem crocus_bulb_cost 
  (space_bulbs : ℕ)
  (crocus_bulbs : ℕ)
  (cost_daffodil_bulb : ℝ)
  (budget : ℝ)
  (purchased_crocus_bulbs : ℕ)
  (total_cost : ℝ)
  (c : ℝ)
  (h_space : space_bulbs = 55)
  (h_cost_daffodil : cost_daffodil_bulb = 0.65)
  (h_budget : budget = 29.15)
  (h_purchased_crocus : purchased_crocus_bulbs = 22)
  (h_total_cost_eq : total_cost = (33:ℕ) * cost_daffodil_bulb)
  (h_eqn : (purchased_crocus_bulbs : ℝ) * c + total_cost = budget) :
  c = 0.35 :=
by 
  sorry

end crocus_bulb_cost_l2075_207510


namespace jane_ends_with_crayons_l2075_207511

-- Definitions for the conditions in the problem
def initial_crayons : Nat := 87
def crayons_eaten : Nat := 7
def packs_bought : Nat := 5
def crayons_per_pack : Nat := 10
def crayons_break : Nat := 3

-- Statement to prove: Jane ends with 127 crayons
theorem jane_ends_with_crayons :
  initial_crayons - crayons_eaten + (packs_bought * crayons_per_pack) - crayons_break = 127 :=
by
  sorry

end jane_ends_with_crayons_l2075_207511


namespace original_price_per_lesson_l2075_207533

theorem original_price_per_lesson (piano_cost lessons_cost : ℤ) (number_of_lessons discount_percent : ℚ) (total_cost : ℤ) (original_price : ℚ) :
  piano_cost = 500 ∧
  number_of_lessons = 20 ∧
  discount_percent = 0.25 ∧
  total_cost = 1100 →
  lessons_cost = total_cost - piano_cost →
  0.75 * (number_of_lessons * original_price) = lessons_cost →
  original_price = 40 :=
by
  intros h h1 h2
  sorry

end original_price_per_lesson_l2075_207533


namespace sales_volume_maximum_profit_l2075_207571

noncomputable def profit (x : ℝ) : ℝ := (x - 34) * (-2 * x + 296)

theorem sales_volume (x : ℝ) : 200 - 2 * (x - 48) = -2 * x + 296 := by
  sorry

theorem maximum_profit :
  (∀ x : ℝ, profit x ≤ profit 91) ∧ profit 91 = 6498 := by
  sorry

end sales_volume_maximum_profit_l2075_207571


namespace cos_three_theta_l2075_207557

theorem cos_three_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 := by
  sorry

end cos_three_theta_l2075_207557


namespace smallest_value_of_3a_plus_2_l2075_207580

theorem smallest_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 10 * a + 6 = 2) : 
  ∃ (x : ℝ), x = 3 * a + 2 ∧ x = -1 :=
by
  sorry

end smallest_value_of_3a_plus_2_l2075_207580


namespace range_of_a_l2075_207506

def p (a : ℝ) : Prop := 0 < a ∧ a < 1
def q (a : ℝ) : Prop := a > 1 / 4

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) : a ∈ Set.Ioc 0 (1 / 4) ∨ a ∈ Set.Ioi 1 :=
by
  sorry

end range_of_a_l2075_207506


namespace min_value_f_range_m_l2075_207567

-- Part I: Prove that the minimum value of f(a) = a^2 + 2/a for a > 0 is 3
theorem min_value_f (a : ℝ) (h : a > 0) : a^2 + 2 / a ≥ 3 :=
sorry

-- Part II: Prove the range of m given the inequality for any positive real number a
theorem range_m (m : ℝ) : (∀ (a : ℝ), a > 0 → a^3 + 2 ≥ 3 * a * (|m - 1| - |2 * m + 3|)) → (m ≤ -3 ∨ m ≥ -1) :=
sorry

end min_value_f_range_m_l2075_207567


namespace joy_pencils_count_l2075_207591

theorem joy_pencils_count :
  ∃ J, J = 30 ∧ (∃ (pencils_cost_J pencils_cost_C : ℕ), 
  pencils_cost_C = 50 * 4 ∧ pencils_cost_J = pencils_cost_C - 80 ∧ J = pencils_cost_J / 4) := sorry

end joy_pencils_count_l2075_207591


namespace value_of_t_eq_3_over_4_l2075_207538

-- Define the values x and y as per the conditions
def x (t : ℝ) : ℝ := 1 - 2 * t
def y (t : ℝ) : ℝ := 2 * t - 2

-- Statement only, proof is omitted using sorry
theorem value_of_t_eq_3_over_4 (t : ℝ) (h : x t = y t) : t = 3 / 4 :=
by
  sorry

end value_of_t_eq_3_over_4_l2075_207538


namespace compute_expression_l2075_207519

theorem compute_expression : 45 * 28 + 72 * 45 = 4500 :=
by
  sorry

end compute_expression_l2075_207519


namespace baker_cakes_l2075_207514

theorem baker_cakes (initial_cakes sold_cakes remaining_cakes final_cakes new_cakes : ℕ)
  (h1 : initial_cakes = 110)
  (h2 : sold_cakes = 75)
  (h3 : final_cakes = 111)
  (h4 : new_cakes = final_cakes - (initial_cakes - sold_cakes)) :
  new_cakes = 76 :=
by {
  sorry
}

end baker_cakes_l2075_207514


namespace polynomial_solution_l2075_207532

open Polynomial
open Real

theorem polynomial_solution (P : Polynomial ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → P.eval (x * sqrt 2) = P.eval (x + sqrt (1 - x^2))) :
  ∃ U : Polynomial ℝ, P = (U.comp (Polynomial.C (1/4) - 2 * X^2 + 5 * X^4 - 4 * X^6 + X^8)) :=
sorry

end polynomial_solution_l2075_207532


namespace region_area_l2075_207508

theorem region_area {x y : ℝ} (h : x^2 + y^2 - 4*x + 2*y = -1) : 
  ∃ (r : ℝ), r = 4*pi := 
sorry

end region_area_l2075_207508


namespace regular_polygon_sides_l2075_207550

theorem regular_polygon_sides (n : ℕ) (h : ∀ (x : ℕ), x = 180 * (n - 2) / n → x = 144) :
  n = 10 :=
sorry

end regular_polygon_sides_l2075_207550


namespace sum_of_coords_of_four_points_l2075_207559

noncomputable def four_points_sum_coords : ℤ :=
  let y1 := 13 + 5
  let y2 := 13 - 5
  let x1 := 7 + 12
  let x2 := 7 - 12
  ((x2 + y2) + (x2 + y1) + (x1 + y2) + (x1 + y1))

theorem sum_of_coords_of_four_points : four_points_sum_coords = 80 :=
  by
    sorry

end sum_of_coords_of_four_points_l2075_207559


namespace pages_per_side_is_4_l2075_207587

-- Define the conditions
def num_books := 2
def pages_per_book := 600
def sheets_used := 150
def sides_per_sheet := 2

-- Define the total number of pages and sides
def total_pages := num_books * pages_per_book
def total_sides := sheets_used * sides_per_sheet

-- Prove the number of pages per side is 4
theorem pages_per_side_is_4 : total_pages / total_sides = 4 := by
  sorry

end pages_per_side_is_4_l2075_207587


namespace no_divisor_neighbors_l2075_207540

def is_divisor (a b : ℕ) : Prop := b % a = 0

def circle_arrangement (arr : Fin 8 → ℕ) : Prop :=
  arr 0 = 7 ∧ arr 1 = 9 ∧ arr 2 = 4 ∧ arr 3 = 5 ∧ arr 4 = 3 ∧ arr 5 = 6 ∧ arr 6 = 8 ∧ arr 7 = 2

def valid_neighbors (arr : Fin 8 → ℕ) : Prop :=
  ¬ is_divisor (arr 0) (arr 1) ∧ ¬ is_divisor (arr 0) (arr 3) ∧
  ¬ is_divisor (arr 1) (arr 2) ∧ ¬ is_divisor (arr 1) (arr 3) ∧ ¬ is_divisor (arr 1) (arr 5) ∧
  ¬ is_divisor (arr 2) (arr 1) ∧ ¬ is_divisor (arr 2) (arr 6) ∧ ¬ is_divisor (arr 2) (arr 3) ∧
  ¬ is_divisor (arr 3) (arr 1) ∧ ¬ is_divisor (arr 3) (arr 4) ∧ ¬ is_divisor (arr 3) (arr 2) ∧ ¬ is_divisor (arr 3) (arr 0) ∧
  ¬ is_divisor (arr 4) (arr 3) ∧ ¬ is_divisor (arr 4) (arr 5) ∧
  ¬ is_divisor (arr 5) (arr 1) ∧ ¬ is_divisor (arr 5) (arr 4) ∧ ¬ is_divisor (arr 5) (arr 6) ∧
  ¬ is_divisor (arr 6) (arr 2) ∧ ¬ is_divisor (arr 6) (arr 5) ∧ ¬ is_divisor (arr 6) (arr 7) ∧
  ¬ is_divisor (arr 7) (arr 6)

theorem no_divisor_neighbors :
  ∀ (arr : Fin 8 → ℕ), circle_arrangement arr → valid_neighbors arr :=
by
  intros arr h
  sorry

end no_divisor_neighbors_l2075_207540


namespace line_segment_value_of_x_l2075_207568

theorem line_segment_value_of_x (x : ℝ) (h1 : (1 - 4)^2 + (3 - x)^2 = 25) (h2 : x > 0) : x = 7 :=
sorry

end line_segment_value_of_x_l2075_207568


namespace find_number_l2075_207585

def number_condition (N : ℝ) : Prop := 
  0.20 * 0.15 * 0.40 * 0.30 * 0.50 * N = 180

theorem find_number (N : ℝ) (h : number_condition N) : N = 1000000 :=
sorry

end find_number_l2075_207585


namespace common_ratio_common_difference_l2075_207582

noncomputable def common_ratio_q {a b : ℕ → ℝ} (d : ℝ) (q : ℝ) :=
  (∀ n, b (n+1) = q * b n) ∧ (a 2 = -1) ∧ (a 1 < a 2) ∧ 
  (b 1 = (a 1)^2) ∧ (b 2 = (a 2)^2) ∧ (b 3 = (a 3)^2) ∧ 
  (∀ n, a (n+1) = a n + d)

theorem common_ratio
  {a b : ℕ → ℝ} {d : ℝ}
  (h_arith : ∀ n, a (n + 1) = a n + d) (h_nonzero : d ≠ 0)
  (h_geom : ∀ n, b (n + 1) = (b 1^(1/2)) ^ (2 ^ n))
  (h_b1 : b 1 = (a 1) ^ 2) (h_b2 : b 2 = (a 2) ^ 2)
  (h_b3 : b 3 = (a 3) ^ 2) (h_a2 : a 2 = -1) (h_a1a2 : a 1 < a 2) :
  q = 3 - 2 * (2:ℝ).sqrt :=
sorry

theorem common_difference
  {a b : ℕ → ℝ} {d : ℝ}
  (h_arith : ∀ n, a (n + 1) = a n + d) (h_nonzero : d ≠ 0)
  (h_geom : ∀ n, b (n + 1) = (b 1^(1/2)) ^ (2 ^ n))
  (h_b1 : b 1 = (a 1) ^ 2) (h_b2 : b 2 = (a 2) ^ 2)
  (h_b3 : b 3 = (a 3) ^ 2) (h_a2 : a 2 = -1) (h_a1a2 : a 1 < a 2) :
  d = (2 : ℝ).sqrt :=
sorry

end common_ratio_common_difference_l2075_207582


namespace no_real_y_for_two_equations_l2075_207548

theorem no_real_y_for_two_equations:
  ¬ ∃ (x y : ℝ), x^2 + y^2 = 16 ∧ x^2 + 3 * y + 30 = 0 :=
by
  sorry

end no_real_y_for_two_equations_l2075_207548


namespace quadratic_roots_proof_l2075_207572

noncomputable def quadratic_roots_statement : Prop :=
  ∃ (x1 x2 : ℝ), 
    (x1 ≠ x2 ∨ x1 = x2) ∧ 
    (x1 = -20 ∧ x2 = -20) ∧ 
    (x1^2 + 40 * x1 + 300 = -100) ∧ 
    (x1 - x2 = 0 ∧ x1 * x2 = 400)  

theorem quadratic_roots_proof : quadratic_roots_statement :=
sorry

end quadratic_roots_proof_l2075_207572


namespace radius_of_inscribed_circle_l2075_207560

variable (p q r : ℝ)

theorem radius_of_inscribed_circle (hp : p > 0) (hq : q > 0) (area_eq : q^2 = r * p) : r = q^2 / p :=
by
  sorry

end radius_of_inscribed_circle_l2075_207560


namespace range_of_function_l2075_207563

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_function : 
  (∀ x : ℝ, x ≠ -2 → f x ≠ 1) ∧
  (∀ y : ℝ, y ≠ 1 → ∃ x : ℝ, f x = y) :=
sorry

end range_of_function_l2075_207563


namespace tunnel_build_equation_l2075_207516

theorem tunnel_build_equation (x : ℝ) (h1 : 1280 > 0) (h2 : x > 0) : 
  (1280 - x) / x = (1280 - x) / (1.4 * x) + 2 := 
by
  sorry

end tunnel_build_equation_l2075_207516


namespace fred_green_balloons_l2075_207577

theorem fred_green_balloons (initial : ℕ) (given : ℕ) (final : ℕ) (h1 : initial = 709) (h2 : given = 221) (h3 : final = initial - given) : final = 488 :=
by
  sorry

end fred_green_balloons_l2075_207577


namespace sufficient_condition_for_inequality_l2075_207528

theorem sufficient_condition_for_inequality (a x : ℝ) (h1 : -2 < x) (h2 : x < -1) :
  (a + x) * (1 + x) < 0 → a > 2 :=
sorry

end sufficient_condition_for_inequality_l2075_207528


namespace min_value_expression_l2075_207505

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1/a + (a/b^2) + b) ≥ 2 * Real.sqrt 2 :=
sorry

end min_value_expression_l2075_207505


namespace equivalent_lengthEF_l2075_207523

namespace GeometryProof

noncomputable def lengthEF 
  (AB CD EF : ℝ) 
  (h_AB_parallel_CD : true) 
  (h_lengthAB : AB = 200) 
  (h_lengthCD : CD = 50) 
  (h_angleEF : true) 
  : ℝ := 
  50

theorem equivalent_lengthEF
  (AB CD EF : ℝ) 
  (h_AB_parallel_CD : true) 
  (h_lengthAB : AB = 200) 
  (h_lengthCD : CD = 50) 
  (h_angleEF : true) 
  : lengthEF AB CD EF h_AB_parallel_CD h_lengthAB h_lengthCD h_angleEF = 50 :=
by
  sorry

end GeometryProof

end equivalent_lengthEF_l2075_207523


namespace compute_fg_difference_l2075_207541

def f (x : ℕ) : ℕ := x^2 + 3
def g (x : ℕ) : ℕ := 2 * x + 5

theorem compute_fg_difference : f (g 5) - g (f 5) = 167 := by
  sorry

end compute_fg_difference_l2075_207541


namespace close_to_one_below_l2075_207509

theorem close_to_one_below (k l m n : ℕ) (h1 : k > l) (h2 : l > m) (h3 : m > n) (hk : k = 43) (hl : l = 7) (hm : m = 3) (hn : n = 2) :
  (1 : ℚ) / k + 1 / l + 1 / m + 1 / n < 1 := by
  sorry

end close_to_one_below_l2075_207509


namespace games_required_for_champion_l2075_207561

-- Define the number of players in the tournament
def players : ℕ := 512

-- Define the tournament conditions
def single_elimination_tournament (n : ℕ) : Prop :=
  ∀ (g : ℕ), g = n - 1

-- State the theorem that needs to be proven
theorem games_required_for_champion : single_elimination_tournament players :=
by
  sorry

end games_required_for_champion_l2075_207561


namespace intimate_interval_proof_l2075_207592

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
def g (x : ℝ) : ℝ := 2 * x - 3

-- Define the concept of intimate functions over an interval
def are_intimate_functions (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- Prove that the interval [2, 3] is a subset of [a, b]
theorem intimate_interval_proof (a b : ℝ) (h : are_intimate_functions a b) :
  2 ≤ b ∧ a ≤ 3 :=
sorry

end intimate_interval_proof_l2075_207592


namespace commute_time_difference_l2075_207535

-- Define the conditions as constants
def distance_to_work : ℝ := 1.5
def walking_speed : ℝ := 3
def train_speed : ℝ := 20
def additional_train_time_minutes : ℝ := 10.5

-- The main proof problem
theorem commute_time_difference : 
  (distance_to_work / walking_speed * 60) - 
  ((distance_to_work / train_speed * 60) + additional_train_time_minutes) = 15 :=
by
  sorry

end commute_time_difference_l2075_207535


namespace ratio_first_term_common_difference_l2075_207512

theorem ratio_first_term_common_difference
  (a d : ℚ)
  (h : (15 / 2) * (2 * a + 14 * d) = 4 * (8 / 2) * (2 * a + 7 * d)) :
  a / d = -7 / 17 := 
by {
  sorry
}

end ratio_first_term_common_difference_l2075_207512


namespace total_chairs_in_canteen_l2075_207564

theorem total_chairs_in_canteen (numRoundTables : ℕ) (numRectangularTables : ℕ) 
                                (chairsPerRoundTable : ℕ) (chairsPerRectangularTable : ℕ)
                                (h1 : numRoundTables = 2)
                                (h2 : numRectangularTables = 2)
                                (h3 : chairsPerRoundTable = 6)
                                (h4 : chairsPerRectangularTable = 7) : 
                                (numRoundTables * chairsPerRoundTable + numRectangularTables * chairsPerRectangularTable = 26) :=
by
  sorry

end total_chairs_in_canteen_l2075_207564


namespace max_correct_answers_l2075_207520

variable (x y z : ℕ)

theorem max_correct_answers
  (h1 : x + y + z = 100)
  (h2 : x - 3 * y - 2 * z = 50) :
  x ≤ 87 := by
    sorry

end max_correct_answers_l2075_207520
