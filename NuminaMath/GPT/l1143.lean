import Mathlib

namespace NUMINAMATH_GPT_determine_percentage_of_second_mixture_l1143_114310

-- Define the given conditions and question
def mixture_problem (P : ℝ) : Prop :=
  ∃ (V1 V2 : ℝ) (A1 A2 A_final : ℝ),
  V1 = 2.5 ∧ A1 = 0.30 ∧
  V2 = 7.5 ∧ A2 = P / 100 ∧
  A_final = 0.45 ∧
  (V1 * A1 + V2 * A2) / (V1 + V2) = A_final

-- State the theorem
theorem determine_percentage_of_second_mixture : mixture_problem 50 := sorry

end NUMINAMATH_GPT_determine_percentage_of_second_mixture_l1143_114310


namespace NUMINAMATH_GPT_carrots_thrown_out_l1143_114380

variable (x : ℕ)

theorem carrots_thrown_out :
  let initial_carrots := 23
  let picked_later := 47
  let total_carrots := 60
  initial_carrots - x + picked_later = total_carrots → x = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_carrots_thrown_out_l1143_114380


namespace NUMINAMATH_GPT_sequence_periodicity_l1143_114315

theorem sequence_periodicity (a : ℕ → ℤ) 
  (h1 : a 1 = 3) 
  (h2 : a 2 = 6) 
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n): 
  a 2015 = -6 := 
sorry

end NUMINAMATH_GPT_sequence_periodicity_l1143_114315


namespace NUMINAMATH_GPT_alex_new_salary_in_may_l1143_114372

def initial_salary : ℝ := 50000
def february_increase (s : ℝ) : ℝ := s * 1.10
def april_bonus (s : ℝ) : ℝ := s + 2000
def may_pay_cut (s : ℝ) : ℝ := s * 0.95

theorem alex_new_salary_in_may : may_pay_cut (april_bonus (february_increase initial_salary)) = 54150 :=
by
  sorry

end NUMINAMATH_GPT_alex_new_salary_in_may_l1143_114372


namespace NUMINAMATH_GPT_problem_statement_l1143_114317

noncomputable def c := 3 + Real.sqrt 21
noncomputable def d := 3 - Real.sqrt 21

theorem problem_statement : 
  (c + 2 * d) = 9 - Real.sqrt 21 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1143_114317


namespace NUMINAMATH_GPT_ratio_of_ages_l1143_114346

theorem ratio_of_ages (Sandy_age : ℕ) (Molly_age : ℕ)
  (h1 : Sandy_age = 56)
  (h2 : Molly_age = Sandy_age + 16) :
  (Sandy_age : ℚ) / Molly_age = 7 / 9 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l1143_114346


namespace NUMINAMATH_GPT_horses_eat_oats_twice_a_day_l1143_114399

-- Define the main constants and assumptions
def number_of_horses : ℕ := 4
def oats_per_meal : ℕ := 4
def grain_per_day : ℕ := 3
def total_food : ℕ := 132
def duration_in_days : ℕ := 3

-- Main theorem statement
theorem horses_eat_oats_twice_a_day (x : ℕ) (h : duration_in_days * number_of_horses * (oats_per_meal * x + grain_per_day) = total_food) : x = 2 := 
sorry

end NUMINAMATH_GPT_horses_eat_oats_twice_a_day_l1143_114399


namespace NUMINAMATH_GPT_sin_330_deg_l1143_114383

theorem sin_330_deg : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_330_deg_l1143_114383


namespace NUMINAMATH_GPT_tan_theta_cos_double_angle_minus_pi_over_3_l1143_114332

open Real

-- Given conditions
variable (θ : ℝ)
axiom sin_theta : sin θ = 3 / 5
axiom theta_in_second_quadrant : π / 2 < θ ∧ θ < π

-- Questions and answers to prove:
theorem tan_theta : tan θ = - 3 / 4 :=
sorry

theorem cos_double_angle_minus_pi_over_3 : cos (2 * θ - π / 3) = (7 - 24 * Real.sqrt 3) / 50 :=
sorry

end NUMINAMATH_GPT_tan_theta_cos_double_angle_minus_pi_over_3_l1143_114332


namespace NUMINAMATH_GPT_abs_neg_number_l1143_114344

theorem abs_neg_number : abs (-2023) = 2023 := sorry

end NUMINAMATH_GPT_abs_neg_number_l1143_114344


namespace NUMINAMATH_GPT_randolph_age_l1143_114387

theorem randolph_age (R Sy S : ℕ) 
  (h1 : R = Sy + 5) 
  (h2 : Sy = 2 * S) 
  (h3 : S = 25) : 
  R = 55 :=
by 
  sorry

end NUMINAMATH_GPT_randolph_age_l1143_114387


namespace NUMINAMATH_GPT_compute_expression_l1143_114328

theorem compute_expression : (3 + 9)^3 + (3^3 + 9^3) = 2484 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l1143_114328


namespace NUMINAMATH_GPT_value_of_expression_l1143_114373

theorem value_of_expression (a b : ℝ) (h : 2 * a + 4 * b = 3) : 4 * a + 8 * b - 2 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_expression_l1143_114373


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1143_114370

theorem solution_set_of_inequality : {x : ℝ | x^2 - 2 * x ≤ 0} = {x | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1143_114370


namespace NUMINAMATH_GPT_number_of_boys_l1143_114334

theorem number_of_boys 
  (B G : ℕ) 
  (h1 : B + G = 650) 
  (h2 : G = B + 106) :
  B = 272 :=
sorry

end NUMINAMATH_GPT_number_of_boys_l1143_114334


namespace NUMINAMATH_GPT_y_coord_of_equidistant_point_on_y_axis_l1143_114304

/-!
  # Goal
  Prove that the $y$-coordinate of the point P on the $y$-axis that is equidistant from points $A(5, 0)$ and $B(3, 6)$ is \( \frac{5}{3} \).
  Conditions:
  - Point A has coordinates (5, 0).
  - Point B has coordinates (3, 6).
-/

theorem y_coord_of_equidistant_point_on_y_axis :
  ∃ y : ℝ, y = 5 / 3 ∧ (dist (⟨0, y⟩ : ℝ × ℝ) (⟨5, 0⟩ : ℝ × ℝ) = dist (⟨0, y⟩ : ℝ × ℝ) (⟨3, 6⟩ : ℝ × ℝ)) :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_y_coord_of_equidistant_point_on_y_axis_l1143_114304


namespace NUMINAMATH_GPT_number_of_red_balls_l1143_114349

-- Conditions
variables (w r : ℕ)
variable (ratio_condition : 4 * r = 3 * w)
variable (white_balls : w = 8)

-- Prove the number of red balls
theorem number_of_red_balls : r = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_red_balls_l1143_114349


namespace NUMINAMATH_GPT_projection_areas_are_correct_l1143_114398

noncomputable def S1 := 1/2 * 2 * 2
noncomputable def S2 := 1/2 * 2 * Real.sqrt 2
noncomputable def S3 := 1/2 * 2 * Real.sqrt 2

theorem projection_areas_are_correct :
  S3 = S2 ∧ S3 ≠ S1 :=
by
  sorry

end NUMINAMATH_GPT_projection_areas_are_correct_l1143_114398


namespace NUMINAMATH_GPT_selling_price_30_items_sales_volume_functional_relationship_selling_price_for_1200_profit_l1143_114393

-- Problem conditions
def cost_price : ℕ := 70
def max_price : ℕ := 99
def initial_price : ℕ := 110
def initial_sales : ℕ := 20
def price_drop_rate : ℕ := 1
def sales_increase_rate : ℕ := 2
def sales_increase_per_yuan : ℕ := 2
def profit_target : ℕ := 1200

-- Selling price for given sales volume
def selling_price_for_sales_volume (sales_volume : ℕ) : ℕ :=
  initial_price - (sales_volume - initial_sales) / sales_increase_per_yuan

-- Functional relationship between sales volume (y) and price (x)
def sales_volume_function (x : ℕ) : ℕ :=
  initial_sales + sales_increase_rate * (initial_price - x)

-- Profit for given price and resulting sales volume
def daily_profit (x : ℕ) : ℤ :=
  (x - cost_price) * (sales_volume_function x)

-- Part 1: Selling price for 30 items sold
theorem selling_price_30_items : selling_price_for_sales_volume 30 = 105 :=
by
  sorry

-- Part 2: Functional relationship between sales volume and selling price
theorem sales_volume_functional_relationship (x : ℕ) (hx : 70 ≤ x ∧ x ≤ 99) :
  sales_volume_function x = 240 - 2 * x :=
by
  sorry

-- Part 3: Selling price for a daily profit of 1200 yuan
theorem selling_price_for_1200_profit {x : ℕ} (hx : 70 ≤ x ∧ x ≤ 99) :
  daily_profit x = 1200 → x = 90 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_30_items_sales_volume_functional_relationship_selling_price_for_1200_profit_l1143_114393


namespace NUMINAMATH_GPT_red_more_than_yellow_l1143_114369

-- Define the total number of marbles
def total_marbles : ℕ := 19

-- Define the number of yellow marbles
def yellow_marbles : ℕ := 5

-- Calculate the number of remaining marbles
def remaining_marbles : ℕ := total_marbles - yellow_marbles

-- Define the ratio of blue to red marbles
def blue_ratio : ℕ := 3
def red_ratio : ℕ := 4

-- Calculate the sum of ratio parts
def sum_ratio : ℕ := blue_ratio + red_ratio

-- Calculate the number of shares per ratio part
def share_per_part : ℕ := remaining_marbles / sum_ratio

-- Calculate the number of red marbles
def red_marbles : ℕ := red_ratio * share_per_part

-- Theorem to prove: the difference between red marbles and yellow marbles is 3
theorem red_more_than_yellow : red_marbles - yellow_marbles = 3 :=
by
  sorry

end NUMINAMATH_GPT_red_more_than_yellow_l1143_114369


namespace NUMINAMATH_GPT_rainfall_hydroville_2012_l1143_114392

-- Define the average monthly rainfall for each year
def avg_rainfall_2010 : ℝ := 37.2
def avg_rainfall_2011 : ℝ := avg_rainfall_2010 + 3.5
def avg_rainfall_2012 : ℝ := avg_rainfall_2011 - 1.2

-- Define the total rainfall for 2012
def total_rainfall_2012 : ℝ := 12 * avg_rainfall_2012

-- The theorem to be proved
theorem rainfall_hydroville_2012 : total_rainfall_2012 = 474 := by
  sorry

end NUMINAMATH_GPT_rainfall_hydroville_2012_l1143_114392


namespace NUMINAMATH_GPT_gumballs_per_box_l1143_114305

-- Given conditions
def total_gumballs : ℕ := 20
def total_boxes : ℕ := 4

-- Mathematically equivalent proof problem
theorem gumballs_per_box:
  total_gumballs / total_boxes = 5 := by
  sorry

end NUMINAMATH_GPT_gumballs_per_box_l1143_114305


namespace NUMINAMATH_GPT_simplest_fraction_sum_l1143_114378

theorem simplest_fraction_sum (c d : ℕ) (h1 : 0.325 = (c:ℚ)/d) (h2 : Int.gcd c d = 1) : c + d = 53 :=
by sorry

end NUMINAMATH_GPT_simplest_fraction_sum_l1143_114378


namespace NUMINAMATH_GPT_sasha_quarters_l1143_114319

theorem sasha_quarters (h₁ : 2.10 = 0.35 * q) : q = 6 := 
sorry

end NUMINAMATH_GPT_sasha_quarters_l1143_114319


namespace NUMINAMATH_GPT_alpha_value_l1143_114382

noncomputable def alpha (x : ℝ) := Real.arccos x

theorem alpha_value (h1 : Real.cos α = -1/6) (h2 : 0 < α ∧ α < Real.pi) : 
  α = Real.pi - alpha (1/6) :=
by
  sorry

end NUMINAMATH_GPT_alpha_value_l1143_114382


namespace NUMINAMATH_GPT_find_original_number_l1143_114312

theorem find_original_number (n : ℝ) (h : n / 2 = 9) : n = 18 :=
sorry

end NUMINAMATH_GPT_find_original_number_l1143_114312


namespace NUMINAMATH_GPT_correct_value_l1143_114301

theorem correct_value : ∀ (x : ℕ),  (x / 6 = 12) → (x * 7 = 504) :=
  sorry

end NUMINAMATH_GPT_correct_value_l1143_114301


namespace NUMINAMATH_GPT_largest_number_among_options_l1143_114395

theorem largest_number_among_options :
  let A := 8.12366
  let B := 8.1236666666666 -- Repeating decimal 8.123\overline{6}
  let C := 8.1236363636363 -- Repeating decimal 8.12\overline{36}
  let D := 8.1236236236236 -- Repeating decimal 8.1\overline{236}
  let E := 8.1236123612361 -- Repeating decimal 8.\overline{1236}
  B > A ∧ B > C ∧ B > D ∧ B > E :=
by
  let A := 8.12366
  let B := 8.12366666666666
  let C := 8.12363636363636
  let D := 8.12362362362362
  let E := 8.12361236123612
  sorry

end NUMINAMATH_GPT_largest_number_among_options_l1143_114395


namespace NUMINAMATH_GPT_translate_right_one_unit_l1143_114397

theorem translate_right_one_unit (x y : ℤ) (hx : x = 4) (hy : y = -3) : (x + 1, y) = (5, -3) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_translate_right_one_unit_l1143_114397


namespace NUMINAMATH_GPT_richmond_population_l1143_114333

theorem richmond_population (R V B : ℕ) (h0 : R = V + 1000) (h1 : V = 4 * B) (h2 : B = 500) : R = 3000 :=
by
  -- skipping proof
  sorry

end NUMINAMATH_GPT_richmond_population_l1143_114333


namespace NUMINAMATH_GPT_most_likely_sitting_people_l1143_114361

theorem most_likely_sitting_people :
  let num_people := 100
  let seats := 100
  let favorite_seats : Fin num_people → Fin seats := sorry
  -- Conditions related to people sitting behavior
  let sits_in_row (i : Fin num_people) : Prop :=
    ∀ j : Fin num_people, j < i → favorite_seats j ≠ favorite_seats i
  let num_sitting_in_row := Finset.card (Finset.filter sits_in_row (Finset.univ : Finset (Fin num_people)))
  -- Prove
  num_sitting_in_row = 10 := 
sorry

end NUMINAMATH_GPT_most_likely_sitting_people_l1143_114361


namespace NUMINAMATH_GPT_perpendicular_lines_sin_2alpha_l1143_114323

theorem perpendicular_lines_sin_2alpha (α : ℝ) 
  (l1 : ∀ (x y : ℝ), x * (Real.sin α) + y - 1 = 0) 
  (l2 : ∀ (x y : ℝ), x - 3 * y * Real.cos α + 1 = 0) 
  (perp : ∀ (x1 y1 x2 y2 : ℝ), 
        (x1 * (Real.sin α) + y1 - 1 = 0) ∧ 
        (x2 - 3 * y2 * Real.cos α + 1 = 0) → 
        ((-Real.sin α) * (1 / (3 * Real.cos α)) = -1)) :
  Real.sin (2 * α) = (3/5) :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_sin_2alpha_l1143_114323


namespace NUMINAMATH_GPT_largest_number_l1143_114321

def HCF (a b c d : ℕ) : Prop := d ∣ a ∧ d ∣ b ∧ d ∣ c ∧ 
                                ∀ e, (e ∣ a ∧ e ∣ b ∧ e ∣ c) → e ≤ d
def LCM (a b c m : ℕ) : Prop := m % a = 0 ∧ m % b = 0 ∧ m % c = 0 ∧ 
                                ∀ n, (n % a = 0 ∧ n % b = 0 ∧ n % c = 0) → m ≤ n

theorem largest_number (a b c : ℕ)
  (hcf: HCF a b c 210)
  (lcm_has_factors: ∃ k1 k2 k3, k1 = 11 ∧ k2 = 17 ∧ k3 = 23 ∧
                                LCM a b c (210 * k1 * k2 * k3)) :
  max a (max b c) = 4830 := 
by
  sorry

end NUMINAMATH_GPT_largest_number_l1143_114321


namespace NUMINAMATH_GPT_part1_part2_l1143_114362

noncomputable def determinant (a b c d : ℤ) : ℤ :=
  a * d - b * c

-- Lean statement for Question (1)
theorem part1 :
  determinant 2022 2023 2021 2022 = 1 :=
by sorry

-- Lean statement for Question (2)
theorem part2 (m : ℤ) :
  determinant (m + 2) (m - 2) (m - 2) (m + 2) = 32 → m = 4 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1143_114362


namespace NUMINAMATH_GPT_paint_cost_is_correct_l1143_114394

-- Definition of known conditions
def costPerKg : ℕ := 50
def coveragePerKg : ℕ := 20
def sideOfCube : ℕ := 20

-- Definition of correct answer
def totalCost : ℕ := 6000

-- Theorem statement
theorem paint_cost_is_correct : (6 * (sideOfCube * sideOfCube) / coveragePerKg) * costPerKg = totalCost :=
by
  sorry

end NUMINAMATH_GPT_paint_cost_is_correct_l1143_114394


namespace NUMINAMATH_GPT_robert_has_2_more_years_l1143_114318

theorem robert_has_2_more_years (R P T Rb M : ℕ) 
                                 (h1 : R = P + T + Rb + M)
                                 (h2 : R = 42)
                                 (h3 : P = 12)
                                 (h4 : T = 2 * Rb)
                                 (h5 : Rb = P - 4) : Rb - M = 2 := 
by 
-- skipped proof
  sorry

end NUMINAMATH_GPT_robert_has_2_more_years_l1143_114318


namespace NUMINAMATH_GPT_rectangle_ratio_l1143_114348

theorem rectangle_ratio (s x y : ℝ) (h1 : 4 * (x * y) + s * s = 9 * s * s) (h2 : s + 2 * y = 3 * s) (h3 : x + y = 3 * s): x / y = 2 :=
by sorry

end NUMINAMATH_GPT_rectangle_ratio_l1143_114348


namespace NUMINAMATH_GPT_second_monkey_took_20_peaches_l1143_114386

theorem second_monkey_took_20_peaches (total_peaches : ℕ) 
  (h1 : total_peaches > 0)
  (eldest_share : ℕ)
  (middle_share : ℕ)
  (youngest_share : ℕ)
  (h3 : total_peaches = eldest_share + middle_share + youngest_share)
  (h4 : eldest_share = (total_peaches * 5) / 9)
  (second_total : ℕ := total_peaches - eldest_share)
  (h5 : middle_share = (second_total * 5) / 9)
  (h6 : youngest_share = second_total - middle_share)
  (h7 : eldest_share - youngest_share = 29) :
  middle_share = 20 :=
by
  sorry

end NUMINAMATH_GPT_second_monkey_took_20_peaches_l1143_114386


namespace NUMINAMATH_GPT_remainder_15_plus_3y_l1143_114374

theorem remainder_15_plus_3y (y : ℕ) (hy : 7 * y ≡ 1 [MOD 31]) : (15 + 3 * y) % 31 = 11 :=
by
  sorry

end NUMINAMATH_GPT_remainder_15_plus_3y_l1143_114374


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l1143_114337

noncomputable def eq1 (x : ℝ) : Prop := x - 2 = 4 * (x - 2)^2
noncomputable def eq2 (x : ℝ) : Prop := x * (2 * x + 1) = 8 * x - 3

theorem solve_eq1 (x : ℝ) : eq1 x ↔ x = 2 ∨ x = 9 / 4 :=
by
  sorry

theorem solve_eq2 (x : ℝ) : eq2 x ↔ x = 1 / 2 ∨ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l1143_114337


namespace NUMINAMATH_GPT_probability_tile_C_less_than_20_and_tile_D_odd_or_greater_than_40_l1143_114347

/-- 
There are 30 tiles in box C numbered from 1 to 30 and 30 tiles in box D numbered from 21 to 50. 
We want to prove that the probability of drawing a tile less than 20 from box C and a tile that 
is either odd or greater than 40 from box D is 19/45. 
-/
theorem probability_tile_C_less_than_20_and_tile_D_odd_or_greater_than_40 :
  (19 / 30) * (2 / 3) = (19 / 45) :=
by sorry

end NUMINAMATH_GPT_probability_tile_C_less_than_20_and_tile_D_odd_or_greater_than_40_l1143_114347


namespace NUMINAMATH_GPT_range_of_x_l1143_114316

theorem range_of_x :
  (∀ t : ℝ, |t - 3| + |2 * t + 1| ≥ |2 * x - 1| + |x + 2|) →
  (-1/2 ≤ x ∧ x ≤ 5/6) :=
by
  intro h 
  sorry

end NUMINAMATH_GPT_range_of_x_l1143_114316


namespace NUMINAMATH_GPT_sum_of_digits_ABCED_l1143_114375

theorem sum_of_digits_ABCED {A B C D E : ℕ} (hABCED : 3 * (10000 * A + 1000 * B + 100 * C + 10 * D + E) = 111111) :
  A + B + C + D + E = 20 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_ABCED_l1143_114375


namespace NUMINAMATH_GPT_total_selling_price_l1143_114363

theorem total_selling_price (total_commissions : ℝ) (number_of_appliances : ℕ) (fixed_commission_rate_per_appliance : ℝ) (percentage_commission_rate : ℝ) :
  total_commissions = number_of_appliances * fixed_commission_rate_per_appliance + percentage_commission_rate * S →
  total_commissions = 662 →
  number_of_appliances = 6 →
  fixed_commission_rate_per_appliance = 50 →
  percentage_commission_rate = 0.10 →
  S = 3620 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_total_selling_price_l1143_114363


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1143_114390

theorem simplify_and_evaluate_expression (x : ℝ) (hx : x = 6) :
  (1 + (2 / (x + 1))) * ((x^2 + x) / (x^2 - 9)) = 2 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1143_114390


namespace NUMINAMATH_GPT_A_in_terms_of_B_l1143_114368

-- Definitions based on conditions
def f (A B x : ℝ) : ℝ := A * x^2 - 3 * B^3
def g (B x : ℝ) : ℝ := B * x^2

-- Theorem statement
theorem A_in_terms_of_B (A B : ℝ) (hB : B ≠ 0) (h : f A B (g B 2) = 0) : A = 3 * B / 16 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_A_in_terms_of_B_l1143_114368


namespace NUMINAMATH_GPT_radius_of_circle_l1143_114364

theorem radius_of_circle
  (r : ℝ) (r_pos : r > 0)
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1^2 + y1^2 = r^2)
  (h2 : x2^2 + y2^2 = r^2)
  (h3 : x1 + y1 = 3)
  (h4 : x2 + y2 = 3)
  (h5 : x1 * x2 + y1 * y2 = -0.5 * r^2) : 
  r = 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_l1143_114364


namespace NUMINAMATH_GPT_common_fraction_equiv_l1143_114336

noncomputable def decimal_equivalent_frac : Prop :=
  ∃ (x : ℚ), x = 413 / 990 ∧ x = 0.4 + (7/10^2 + 1/10^3) / (1 - 1/10^2)

theorem common_fraction_equiv : decimal_equivalent_frac :=
by
  sorry

end NUMINAMATH_GPT_common_fraction_equiv_l1143_114336


namespace NUMINAMATH_GPT_simplify_expression_l1143_114381

theorem simplify_expression (n : ℕ) (hn : 0 < n) :
  (3^(n+5) - 3 * 3^n) / (3 * 3^(n+4) - 6) = 80 / 81 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1143_114381


namespace NUMINAMATH_GPT_polynomial_exists_int_coeff_l1143_114307

theorem polynomial_exists_int_coeff (n : ℕ) (hn : n > 1) : 
  ∃ P : Polynomial ℤ × Polynomial ℤ × Polynomial ℤ → Polynomial ℤ, 
  ∀ x : Polynomial ℤ, P ⟨x^n, x^(n+1), x + x^(n+2)⟩ = x :=
by sorry

end NUMINAMATH_GPT_polynomial_exists_int_coeff_l1143_114307


namespace NUMINAMATH_GPT_find_integers_l1143_114308

theorem find_integers (A B C : ℤ) (hA : A = 500) (hB : B = -1) (hC : C = -500) : 
  (A : ℚ) / 999 + (B : ℚ) / 1000 + (C : ℚ) / 1001 = 1 / (999 * 1000 * 1001) :=
by 
  rw [hA, hB, hC]
  sorry

end NUMINAMATH_GPT_find_integers_l1143_114308


namespace NUMINAMATH_GPT_find_A_l1143_114314

theorem find_A (d q r A : ℕ) (h1 : d = 7) (h2 : q = 5) (h3 : r = 3) (h4 : A = d * q + r) : A = 38 := 
by 
  { sorry }

end NUMINAMATH_GPT_find_A_l1143_114314


namespace NUMINAMATH_GPT_combination_15_5_l1143_114385

theorem combination_15_5 : 
  ∀ (n r : ℕ), n = 15 → r = 5 → n.choose r = 3003 :=
by
  intro n r h1 h2
  rw [h1, h2]
  exact Nat.choose_eq_factorial_div_factorial (by norm_num)

end NUMINAMATH_GPT_combination_15_5_l1143_114385


namespace NUMINAMATH_GPT_smallest_clock_equivalent_number_l1143_114311

theorem smallest_clock_equivalent_number :
  ∃ h : ℕ, h > 4 ∧ h^2 % 24 = h % 24 ∧ h = 12 := by
  sorry

end NUMINAMATH_GPT_smallest_clock_equivalent_number_l1143_114311


namespace NUMINAMATH_GPT_exists_integers_for_prime_l1143_114353

theorem exists_integers_for_prime (p : ℕ) (hp : Nat.Prime p) : 
  ∃ x y z w : ℤ, x^2 + y^2 + z^2 = w * p ∧ 0 < w ∧ w < p :=
by 
  sorry

end NUMINAMATH_GPT_exists_integers_for_prime_l1143_114353


namespace NUMINAMATH_GPT_eccentricities_proof_l1143_114355

variable (e1 e2 m n c : ℝ)
variable (h1 : e1 = 2 * c / (m + n))
variable (h2 : e2 = 2 * c / (m - n))
variable (h3 : m ^ 2 + n ^ 2 = 4 * c ^ 2)

theorem eccentricities_proof :
  (e1 * e2) / (Real.sqrt (e1 ^ 2 + e2 ^ 2)) = (Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_GPT_eccentricities_proof_l1143_114355


namespace NUMINAMATH_GPT_length_BC_l1143_114343

theorem length_BC (AB AC AM : ℝ)
  (hAB : AB = 5)
  (hAC : AC = 7)
  (hAM : AM = 4)
  (M_midpoint_of_BC : ∃ (BM MC : ℝ), BM = MC ∧ ∀ (BC: ℝ), BC = BM + MC) :
  ∃ (BC : ℝ), BC = 2 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_GPT_length_BC_l1143_114343


namespace NUMINAMATH_GPT_tangent_line_condition_l1143_114329

-- statement only, no proof required
theorem tangent_line_condition {m n u v x y : ℝ}
  (hm : m > 1)
  (curve_eq : x^m + y^m = 1)
  (line_eq : u * x + v * y = 1)
  (u_v_condition : u^n + v^n = 1)
  (mn_condition : 1/m + 1/n = 1)
  : (u * x + v * y = 1) ↔ (u^n + v^n = 1 ∧ 1/m + 1/n = 1) :=
sorry

end NUMINAMATH_GPT_tangent_line_condition_l1143_114329


namespace NUMINAMATH_GPT_no_negatives_l1143_114302

theorem no_negatives (x y : ℝ) (h : |x^2 + y^2 - 4*x - 4*y + 5| = |2*x + 2*y - 4|) : 
  ¬ (x < 0) ∧ ¬ (y < 0) :=
by
  sorry

end NUMINAMATH_GPT_no_negatives_l1143_114302


namespace NUMINAMATH_GPT_factorization_of_polynomial_l1143_114365

theorem factorization_of_polynomial (x : ℝ) : 2 * x^2 - 12 * x + 18 = 2 * (x - 3)^2 := by
  sorry

end NUMINAMATH_GPT_factorization_of_polynomial_l1143_114365


namespace NUMINAMATH_GPT_net_profit_is_correct_l1143_114330

-- Define the purchase price, markup, and overhead percentage
def purchase_price : ℝ := 48
def markup : ℝ := 55
def overhead_percentage : ℝ := 0.30

-- Define the overhead cost calculation
def overhead_cost : ℝ := overhead_percentage * purchase_price

-- Define the net profit calculation
def net_profit : ℝ := markup - overhead_cost

-- State the theorem
theorem net_profit_is_correct : net_profit = 40.60 :=
by
  sorry

end NUMINAMATH_GPT_net_profit_is_correct_l1143_114330


namespace NUMINAMATH_GPT_speed_of_current_l1143_114367

-- Define the conditions in Lean
theorem speed_of_current (c : ℝ) (r : ℝ) 
  (hu : c - r = 12 / 6) -- upstream speed equation
  (hd : c + r = 12 / 0.75) -- downstream speed equation
  : r = 7 := 
sorry

end NUMINAMATH_GPT_speed_of_current_l1143_114367


namespace NUMINAMATH_GPT_cartesian_equation_of_circle_c2_positional_relationship_between_circles_l1143_114384
noncomputable def circle_c1 := {p : ℝ × ℝ | (p.1)^2 - 2*p.1 + (p.2)^2 = 0}
noncomputable def circle_c2_polar (theta : ℝ) : ℝ × ℝ := (2 * Real.sin theta * Real.cos theta, 2 * Real.sin theta * Real.sin theta)
noncomputable def circle_c2_cartesian := {p : ℝ × ℝ | (p.1)^2 + (p.2 - 1)^2 = 1}

theorem cartesian_equation_of_circle_c2 :
  ∀ p : ℝ × ℝ, (∃ θ : ℝ, p = circle_c2_polar θ) ↔ p ∈ circle_c2_cartesian :=
by
  sorry

theorem positional_relationship_between_circles :
  ∃ p : ℝ × ℝ, p ∈ circle_c1 ∧ p ∈ circle_c2_cartesian :=
by
  sorry

end NUMINAMATH_GPT_cartesian_equation_of_circle_c2_positional_relationship_between_circles_l1143_114384


namespace NUMINAMATH_GPT_sum_of_factors_of_30_l1143_114335

/--
Given the positive integer factors of 30, prove that their sum is 72.
-/
theorem sum_of_factors_of_30 : 
  (1 + 2 + 3 + 5 + 6 + 10 + 15 + 30) = 72 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_factors_of_30_l1143_114335


namespace NUMINAMATH_GPT_reflected_ray_equation_l1143_114357

-- Define the initial point
def point_of_emanation : (ℝ × ℝ) := (-1, 3)

-- Define the point after reflection which the ray passes through
def point_after_reflection : (ℝ × ℝ) := (4, 6)

-- Define the expected equation of the line in general form
def expected_line_equation (x y : ℝ) : Prop := 9 * x - 5 * y - 6 = 0

-- The theorem we need to prove
theorem reflected_ray_equation :
  ∃ (m b : ℝ), ∀ x y : ℝ, (y = m * x + b) → expected_line_equation x y :=
sorry

end NUMINAMATH_GPT_reflected_ray_equation_l1143_114357


namespace NUMINAMATH_GPT_circle_tangent_to_xaxis_at_origin_l1143_114356

theorem circle_tangent_to_xaxis_at_origin (G E F : ℝ)
  (h : ∀ x y: ℝ, x^2 + y^2 + G*x + E*y + F = 0 → y = 0 ∧ x = 0 ∧ 0 < E) :
  G = 0 ∧ F = 0 ∧ E ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_circle_tangent_to_xaxis_at_origin_l1143_114356


namespace NUMINAMATH_GPT_max_colors_for_valid_coloring_l1143_114350

-- Define the 4x4 grid as a type synonym for a set of cells
def Grid4x4 := Fin 4 × Fin 4

-- Condition: Define a valid coloring function for a 4x4 grid
def valid_coloring (colors : ℕ) (f : Grid4x4 → Fin colors) : Prop :=
  ∀ i j : Fin 3, ∃ c : Fin colors, (f (i, j) = c ∨ f (i+1, j) = c) ∧ (f (i+1, j) = c ∨ f (i, j+1) = c)

-- The main theorem to prove
theorem max_colors_for_valid_coloring : 
  ∃ (colors : ℕ), colors = 11 ∧ ∀ f : Grid4x4 → Fin colors, valid_coloring colors f :=
sorry

end NUMINAMATH_GPT_max_colors_for_valid_coloring_l1143_114350


namespace NUMINAMATH_GPT_proof_l1143_114331

-- Define proposition p as negated form: ∀ x < 1, log_3 x ≤ 0
def p : Prop := ∀ x : ℝ, x < 1 → Real.log x / Real.log 3 ≤ 0

-- Define proposition q: ∃ x_0 ∈ ℝ, x_0^2 ≥ 2^x_0
def q : Prop := ∃ x_0 : ℝ, x_0^2 ≥ Real.exp (x_0 * Real.log 2)

-- State we need to prove: p ∨ q
theorem proof : p ∨ q := sorry

end NUMINAMATH_GPT_proof_l1143_114331


namespace NUMINAMATH_GPT_vote_proportion_inequality_l1143_114327

theorem vote_proportion_inequality
  (a b k : ℕ)
  (hb_odd : b % 2 = 1)
  (hb_min : 3 ≤ b)
  (vote_same : ∀ (i j : ℕ) (hi hj : i ≠ j) (votes : ℕ → ℕ), ∃ (k_max : ℕ), ∀ (cont : ℕ), votes cont ≤ k_max) :
  (k : ℚ) / a ≥ (b - 1) / (2 * b) := sorry

end NUMINAMATH_GPT_vote_proportion_inequality_l1143_114327


namespace NUMINAMATH_GPT_solve_equation_l1143_114366

noncomputable def f (x : ℝ) := (1 / (x^2 + 17 * x + 20)) + (1 / (x^2 + 12 * x + 20)) + (1 / (x^2 - 15 * x + 20))

theorem solve_equation :
  {x : ℝ | f x = 0} = {-1, -4, -5, -20} :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1143_114366


namespace NUMINAMATH_GPT_contrapositive_of_square_root_l1143_114351

theorem contrapositive_of_square_root (a b : ℝ) :
  (a^2 < b → -Real.sqrt b < a ∧ a < Real.sqrt b) ↔ (a ≥ Real.sqrt b ∨ a ≤ -Real.sqrt b → a^2 ≥ b) := 
sorry

end NUMINAMATH_GPT_contrapositive_of_square_root_l1143_114351


namespace NUMINAMATH_GPT_basil_plants_count_l1143_114341

-- Define the number of basil plants and the number of oregano plants
variables (B O : ℕ)

-- Define the conditions
def condition1 : Prop := O = 2 * B + 2
def condition2 : Prop := B + O = 17

-- The proof statement
theorem basil_plants_count (h1 : condition1 B O) (h2 : condition2 B O) : B = 5 := by
  sorry

end NUMINAMATH_GPT_basil_plants_count_l1143_114341


namespace NUMINAMATH_GPT_find_original_price_l1143_114313

-- Definitions based on Conditions
def original_price (P : ℝ) : Prop :=
  let increased_price := 1.25 * P
  let final_price := increased_price * 0.75
  final_price = 187.5

theorem find_original_price (P : ℝ) (h : original_price P) : P = 200 :=
  by sorry

end NUMINAMATH_GPT_find_original_price_l1143_114313


namespace NUMINAMATH_GPT_calculate_selling_price_l1143_114396

noncomputable def purchase_price : ℝ := 225
noncomputable def overhead_expenses : ℝ := 20
noncomputable def profit_percent : ℝ := 22.448979591836732

noncomputable def total_cost : ℝ := purchase_price + overhead_expenses
noncomputable def profit : ℝ := (profit_percent / 100) * total_cost
noncomputable def selling_price : ℝ := total_cost + profit

theorem calculate_selling_price : selling_price = 300 := by
  sorry

end NUMINAMATH_GPT_calculate_selling_price_l1143_114396


namespace NUMINAMATH_GPT_least_number_added_1054_l1143_114322

theorem least_number_added_1054 (x d: ℕ) (h_cond: 1054 + x = 1058) (h_div: d = 2) : 1058 % d = 0 :=
by
  sorry

end NUMINAMATH_GPT_least_number_added_1054_l1143_114322


namespace NUMINAMATH_GPT_min_value_f_l1143_114303

def f (x y : ℝ) : ℝ := x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y

theorem min_value_f : ∃ x y : ℝ, f x y = -9 / 5 :=
sorry

end NUMINAMATH_GPT_min_value_f_l1143_114303


namespace NUMINAMATH_GPT_exponentiation_correct_l1143_114339

theorem exponentiation_correct (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 :=
sorry

end NUMINAMATH_GPT_exponentiation_correct_l1143_114339


namespace NUMINAMATH_GPT_probability_of_choosing_gulongzhong_l1143_114377

def num_attractions : Nat := 4
def num_ways_gulongzhong : Nat := 1
def probability_gulongzhong : ℚ := num_ways_gulongzhong / num_attractions

theorem probability_of_choosing_gulongzhong : probability_gulongzhong = 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_choosing_gulongzhong_l1143_114377


namespace NUMINAMATH_GPT_find_sum_l1143_114306

theorem find_sum (x y : ℝ) (h₁ : 3 * |x| + 2 * x + y = 20) (h₂ : 2 * x + 3 * |y| - y = 30) : x + y = 15 :=
sorry

end NUMINAMATH_GPT_find_sum_l1143_114306


namespace NUMINAMATH_GPT_largest_composite_in_five_consecutive_ints_l1143_114345

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_of_five_composite_ints : ℕ :=
  36

theorem largest_composite_in_five_consecutive_ints (a b c d e : ℕ) :
  a < 40 ∧ b < 40 ∧ c < 40 ∧ d < 40 ∧ e < 40 ∧ 
  ¬is_prime a ∧ ¬is_prime b ∧ ¬is_prime c ∧ ¬is_prime d ∧ ¬is_prime e ∧ 
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a = 32 ∧ b = 33 ∧ c = 34 ∧ d = 35 ∧ e = 36 →
  e = largest_of_five_composite_ints :=
by 
  sorry

end NUMINAMATH_GPT_largest_composite_in_five_consecutive_ints_l1143_114345


namespace NUMINAMATH_GPT_number_of_workers_in_each_block_is_200_l1143_114354

-- Conditions
def total_amount : ℕ := 6000
def worth_of_each_gift : ℕ := 2
def number_of_blocks : ℕ := 15

-- Question and answer to be proven
def number_of_workers_in_each_block : ℕ := total_amount / worth_of_each_gift / number_of_blocks

theorem number_of_workers_in_each_block_is_200 :
  number_of_workers_in_each_block = 200 :=
by
  -- Skip the proof with sorry
  sorry

end NUMINAMATH_GPT_number_of_workers_in_each_block_is_200_l1143_114354


namespace NUMINAMATH_GPT_find_AB_l1143_114300

-- Definitions based on conditions
variables (AB CD : ℝ)

-- Given conditions
def area_ratio_condition : Prop :=
  AB / CD = 5 / 3

def sum_condition : Prop :=
  AB + CD = 160

-- The main statement to be proven
theorem find_AB (h_ratio : area_ratio_condition AB CD) (h_sum : sum_condition AB CD) :
  AB = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_AB_l1143_114300


namespace NUMINAMATH_GPT_intersection_complement_l1143_114309

open Set

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

-- Theorem
theorem intersection_complement :
  A ∩ (U \ B) = {1} :=
sorry

end NUMINAMATH_GPT_intersection_complement_l1143_114309


namespace NUMINAMATH_GPT_vasya_no_purchase_days_l1143_114326

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end NUMINAMATH_GPT_vasya_no_purchase_days_l1143_114326


namespace NUMINAMATH_GPT_length_of_bridge_l1143_114376

theorem length_of_bridge
  (walking_speed_km_hr : ℝ) (time_minutes : ℝ) (length_bridge : ℝ) 
  (h1 : walking_speed_km_hr = 5) 
  (h2 : time_minutes = 15) 
  (h3 : length_bridge = 1250) : 
  length_bridge = (walking_speed_km_hr * 1000 / 60) * time_minutes := 
by 
  sorry

end NUMINAMATH_GPT_length_of_bridge_l1143_114376


namespace NUMINAMATH_GPT_algebraic_identity_l1143_114391

theorem algebraic_identity (a b : ℝ) : a^2 - b^2 = (a + b) * (a - b) :=
by
  sorry

example : (2011 : ℝ)^2 - (2010 : ℝ)^2 = 4021 := 
by
  have h := algebraic_identity 2011 2010
  rw [h]
  norm_num

end NUMINAMATH_GPT_algebraic_identity_l1143_114391


namespace NUMINAMATH_GPT_total_weight_of_2_meters_l1143_114352

def tape_measure_length : ℚ := 5
def tape_measure_weight : ℚ := 29 / 8
def computer_length : ℚ := 4
def computer_weight : ℚ := 2.8

noncomputable def weight_per_meter_tape_measure : ℚ := tape_measure_weight / tape_measure_length
noncomputable def weight_per_meter_computer : ℚ := computer_weight / computer_length

noncomputable def total_weight : ℚ :=
  2 * weight_per_meter_tape_measure + 2 * weight_per_meter_computer

theorem total_weight_of_2_meters (h1 : tape_measure_length = 5)
    (h2 : tape_measure_weight = 29 / 8) 
    (h3 : computer_length = 4) 
    (h4 : computer_weight = 2.8): 
    total_weight = 57 / 20 := by 
  unfold total_weight
  sorry

end NUMINAMATH_GPT_total_weight_of_2_meters_l1143_114352


namespace NUMINAMATH_GPT_max_m_value_l1143_114379

variables {x y m : ℝ}

theorem max_m_value (h1 : 4 * x + 3 * y = 4 * m + 5)
                     (h2 : 3 * x - y = m - 1)
                     (h3 : x + 4 * y ≤ 3) :
                     m ≤ -1 :=
sorry

end NUMINAMATH_GPT_max_m_value_l1143_114379


namespace NUMINAMATH_GPT_matilda_father_chocolates_left_l1143_114324

-- definitions for each condition
def initial_chocolates : ℕ := 20
def persons : ℕ := 5
def chocolates_per_person := initial_chocolates / persons
def half_chocolates_per_person := chocolates_per_person / 2
def total_given_to_father := half_chocolates_per_person * persons
def chocolates_given_to_mother := 3
def chocolates_eaten_by_father := 2

-- statement to prove
theorem matilda_father_chocolates_left :
  total_given_to_father - chocolates_given_to_mother - chocolates_eaten_by_father = 5 :=
by
  sorry

end NUMINAMATH_GPT_matilda_father_chocolates_left_l1143_114324


namespace NUMINAMATH_GPT_number_without_daughters_l1143_114358

-- Given conditions
def Marilyn_daughters : Nat := 10
def total_women : Nat := 40
def daughters_with_daughters_women_have_each : Nat := 5

-- Helper definition representing the computation of granddaughters
def Marilyn_granddaughters : Nat := total_women - Marilyn_daughters

-- Proving the main statement
theorem number_without_daughters : 
  (Marilyn_daughters - (Marilyn_granddaughters / daughters_with_daughters_women_have_each)) + Marilyn_granddaughters = 34 := by
  sorry

end NUMINAMATH_GPT_number_without_daughters_l1143_114358


namespace NUMINAMATH_GPT_min_value_of_expression_l1143_114371

/-- 
Given α and β are the two real roots of the quadratic equation x^2 - 2a * x + a + 6 = 0,
prove that the minimum value of (α - 1)^2 + (β - 1)^2 is 8.
-/
theorem min_value_of_expression (a α β : ℝ) (h1 : α ^ 2 - 2 * a * α + a + 6 = 0) (h2 : β ^ 2 - 2 * a * β + a + 6 = 0) :
  (α - 1)^2 + (β - 1)^2 ≥ 8 := 
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1143_114371


namespace NUMINAMATH_GPT_negation_of_proposition_l1143_114325

open Nat 

theorem negation_of_proposition : 
  (¬ ∃ n : ℕ, n > 0 ∧ n^2 > 2^n) ↔ ∀ n : ℕ, n > 0 → n^2 ≤ 2^n :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1143_114325


namespace NUMINAMATH_GPT_amy_points_per_treasure_l1143_114340

theorem amy_points_per_treasure (treasures_first_level treasures_second_level total_score : ℕ) (h1 : treasures_first_level = 6) (h2 : treasures_second_level = 2) (h3 : total_score = 32) :
  total_score / (treasures_first_level + treasures_second_level) = 4 := by
  sorry

end NUMINAMATH_GPT_amy_points_per_treasure_l1143_114340


namespace NUMINAMATH_GPT_triangle_angle_ratio_arbitrary_convex_quadrilateral_angle_ratio_not_arbitrary_convex_pentagon_angle_ratio_not_arbitrary_l1143_114359

theorem triangle_angle_ratio_arbitrary (k1 k2 k3 : ℕ) :
  ∃ (A B C : ℝ), A + B + C = 180 ∧ (A / B = k1 / k2) ∧ (A / C = k1 / k3) :=
  sorry

theorem convex_quadrilateral_angle_ratio_not_arbitrary (k1 k2 k3 k4 : ℕ) :
  ¬(∃ (A B C D : ℝ), A + B + C + D = 360 ∧
  A < B + C + D ∧
  B < A + C + D ∧
  C < A + B + D ∧
  D < A + B + C) :=
  sorry

theorem convex_pentagon_angle_ratio_not_arbitrary (k1 k2 k3 k4 k5 : ℕ) :
  ¬(∃ (A B C D E : ℝ), A + B + C + D + E = 540 ∧
  A < (B + C + D + E) / 2 ∧
  B < (A + C + D + E) / 2 ∧
  C < (A + B + D + E) / 2 ∧
  D < (A + B + C + E) / 2 ∧
  E < (A + B + C + D) / 2) :=
  sorry

end NUMINAMATH_GPT_triangle_angle_ratio_arbitrary_convex_quadrilateral_angle_ratio_not_arbitrary_convex_pentagon_angle_ratio_not_arbitrary_l1143_114359


namespace NUMINAMATH_GPT_solve_for_x_l1143_114338

theorem solve_for_x (x : ℝ) (h : 3 * x - 7 = 2 * x + 5) : x = 12 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1143_114338


namespace NUMINAMATH_GPT_factor_theorem_solution_l1143_114342

theorem factor_theorem_solution (t : ℝ) :
  (∃ p q : ℝ, 10 * p * q = 10 * t * t + 21 * t - 10 ∧ (x - q) = (x - t)) →
  t = 2 / 5 ∨ t = -5 / 2 := by
  sorry

end NUMINAMATH_GPT_factor_theorem_solution_l1143_114342


namespace NUMINAMATH_GPT_fraction_equality_l1143_114388

theorem fraction_equality (a b c : ℝ) (h1 : b + c + a ≠ 0) (h2 : b + c ≠ a) : 
  (b^2 + a^2 - c^2 + 2*b*c) / (b^2 + c^2 - a^2 + 2*b*c) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_equality_l1143_114388


namespace NUMINAMATH_GPT_greatest_possible_percentage_of_airlines_both_services_l1143_114320

noncomputable def maxPercentageOfAirlinesWithBothServices (percentageInternet percentageSnacks : ℝ) : ℝ :=
  if percentageInternet <= percentageSnacks then percentageInternet else percentageSnacks

theorem greatest_possible_percentage_of_airlines_both_services:
  let p_internet := 0.35
  let p_snacks := 0.70
  maxPercentageOfAirlinesWithBothServices p_internet p_snacks = 0.35 :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_percentage_of_airlines_both_services_l1143_114320


namespace NUMINAMATH_GPT_corrected_mean_l1143_114360

theorem corrected_mean (n : ℕ) (obs_mean : ℝ) (obs_count : ℕ) (wrong_val correct_val : ℝ) :
  obs_count = 40 →
  obs_mean = 100 →
  wrong_val = 75 →
  correct_val = 50 →
  (obs_count * obs_mean - (wrong_val - correct_val)) / obs_count = 3975 / 40 :=
by
  sorry

end NUMINAMATH_GPT_corrected_mean_l1143_114360


namespace NUMINAMATH_GPT_partI_inequality_solution_partII_minimum_value_l1143_114389

-- Part (I)
theorem partI_inequality_solution (x : ℝ) : 
  (abs (x + 1) + abs (2 * x - 1) ≤ 3) ↔ (-1 ≤ x ∧ x ≤ 1) :=
sorry

-- Part (II)
theorem partII_minimum_value (a b c : ℝ) (h1 : a + b + c = 2) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) :
  (∀ a b c : ℝ, a + b + c = 2 ->  a > 0 -> b > 0 -> c > 0 -> 
    (1 / a + 1 / b + 1 / c) = (9 / 2)) :=
sorry

end NUMINAMATH_GPT_partI_inequality_solution_partII_minimum_value_l1143_114389
