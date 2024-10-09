import Mathlib

namespace D_180_equals_43_l1581_158174

-- Define D(n) as the number of ways to express the positive integer n
-- as a product of integers strictly greater than 1, where the order of factors matters.
def D (n : Nat) : Nat := sorry  -- The actual implementation is not provided, as per instructions.

theorem D_180_equals_43 : D 180 = 43 :=
by
  sorry  -- The proof is omitted as the task specifies.

end D_180_equals_43_l1581_158174


namespace sum_of_solutions_l1581_158150

-- Define the initial condition
def initial_equation (x : ℝ) : Prop := (x - 8) ^ 2 = 49

-- Define the conclusion we want to reach
def sum_of_solutions_is_16 : Prop :=
  ∃ x1 x2 : ℝ, initial_equation x1 ∧ initial_equation x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 16

theorem sum_of_solutions :
  sum_of_solutions_is_16 :=
by
  sorry

end sum_of_solutions_l1581_158150


namespace factor_quadratic_l1581_158160

theorem factor_quadratic (x : ℝ) : 
  (x^2 + 6 * x + 9 - 16 * x^4) = (-4 * x^2 + 2 * x + 3) * (4 * x^2 + 2 * x + 3) := 
by 
  sorry

end factor_quadratic_l1581_158160


namespace sum_of_first_four_terms_of_geometric_sequence_l1581_158141

noncomputable def geometric_sum_first_four (a : ℕ → ℝ) (q : ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3

theorem sum_of_first_four_terms_of_geometric_sequence 
  (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 : q > 0) 
  (h3 : a 2 = 1) 
  (h4 : ∀ n, a (n + 2) + a (n + 1) = 6 * a n) :
  geometric_sum_first_four a q = 15 / 2 :=
sorry

end sum_of_first_four_terms_of_geometric_sequence_l1581_158141


namespace find_N_l1581_158182

theorem find_N : 
  ∀ (a b c N : ℝ), 
  a + b + c = 80 → 
  2 * a = N → 
  b - 10 = N → 
  3 * c = N → 
  N = 38 := 
by sorry

end find_N_l1581_158182


namespace sum_ai_le_sum_bi_l1581_158187

open BigOperators

variable {α : Type*} [LinearOrderedField α]

theorem sum_ai_le_sum_bi {n : ℕ} {a b : Fin n → α}
  (h1 : ∀ i, 0 < a i)
  (h2 : ∀ i, 0 < b i)
  (h3 : ∑ i, (a i)^2 / b i ≤ ∑ i, b i) :
  ∑ i, a i ≤ ∑ i, b i :=
sorry

end sum_ai_le_sum_bi_l1581_158187


namespace no_roots_in_disk_l1581_158139

noncomputable def homogeneous_polynomial_deg2 (a b c : ℝ) (x y : ℝ) := a * x^2 + b * x * y + c * y^2
noncomputable def homogeneous_polynomial_deg3 (q : ℝ → ℝ → ℝ) (x y : ℝ) := q x y

theorem no_roots_in_disk 
  (a b c : ℝ) (h_poly_deg2 : ∀ x y, homogeneous_polynomial_deg2 a b c x y = a * x^2 + b * x * y + c * y^2)
  (q : ℝ → ℝ → ℝ) (h_poly_deg3 : ∀ x y, homogeneous_polynomial_deg3 q x y = q x y)
  (h_cond : b^2 < 4 * a * c) :
  ∃ k > 0, ∀ x y, x^2 + y^2 < k → homogeneous_polynomial_deg2 a b c x y ≠ homogeneous_polynomial_deg3 q x y ∨ (x = 0 ∧ y = 0) :=
sorry

end no_roots_in_disk_l1581_158139


namespace negation_of_proposition_l1581_158173

theorem negation_of_proposition : 
  ¬ (∀ x : ℝ, x > 0 → x^2 ≤ 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 > 0 := by
  sorry

end negation_of_proposition_l1581_158173


namespace trader_bags_correct_l1581_158119

-- Definitions according to given conditions
def initial_bags := 55
def sold_bags := 23
def restocked_bags := 132

-- Theorem that encapsulates the problem's question and the proven answer
theorem trader_bags_correct :
  (initial_bags - sold_bags + restocked_bags) = 164 :=
by
  sorry

end trader_bags_correct_l1581_158119


namespace Julie_work_hours_per_week_l1581_158123

theorem Julie_work_hours_per_week 
  (hours_summer_per_week : ℕ)
  (weeks_summer : ℕ)
  (total_earnings_summer : ℕ)
  (planned_weeks_school_year : ℕ)
  (needed_income_school_year : ℕ)
  (hourly_wage : ℝ := total_earnings_summer / (hours_summer_per_week * weeks_summer))
  (total_hours_needed_school_year : ℝ := needed_income_school_year / hourly_wage)
  (hours_per_week_needed : ℝ := total_hours_needed_school_year / planned_weeks_school_year) :
  hours_summer_per_week = 60 →
  weeks_summer = 8 →
  total_earnings_summer = 6000 →
  planned_weeks_school_year = 40 →
  needed_income_school_year = 10000 →
  hours_per_week_needed = 20 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end Julie_work_hours_per_week_l1581_158123


namespace proof_A_proof_C_l1581_158199

theorem proof_A (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a * b ≤ ( (a + b) / 2) ^ 2 := 
sorry

theorem proof_C (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) : 
  ∃ y, y = x * (4 - x^2).sqrt ∧ y ≤ 2 := 
sorry

end proof_A_proof_C_l1581_158199


namespace f_difference_l1581_158104

noncomputable def f (n : ℕ) : ℝ :=
  (6 + 4 * Real.sqrt 3) / 12 * ((1 + Real.sqrt 3) / 2)^n + 
  (6 - 4 * Real.sqrt 3) / 12 * ((1 - Real.sqrt 3) / 2)^n

theorem f_difference (n : ℕ) : f (n + 1) - f n = (Real.sqrt 3 - 3) / 4 * f n :=
  sorry

end f_difference_l1581_158104


namespace solution_to_problem_l1581_158131

def problem_statement : Prop :=
  (2.017 * 2016 - 10.16 * 201.7 = 2017)

theorem solution_to_problem : problem_statement :=
by
  sorry

end solution_to_problem_l1581_158131


namespace circles_intersect_if_and_only_if_l1581_158142

noncomputable def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 10 * y + 1 = 0

noncomputable def circle2 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 2 * y - m = 0

theorem circles_intersect_if_and_only_if (m : ℝ) :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) ↔ -1 < m ∧ m < 79 :=
by {
  sorry
}

end circles_intersect_if_and_only_if_l1581_158142


namespace pencil_count_l1581_158135

theorem pencil_count (a : ℕ) (h1 : 200 ≤ a) (h2 : a ≤ 300)
    (h3 : a % 10 = 7) (h4 : a % 12 = 9) : a = 237 ∨ a = 297 :=
by {
  sorry
}

end pencil_count_l1581_158135


namespace ratio_dark_blue_to_total_l1581_158176

-- Definitions based on the conditions
def total_marbles := 63
def red_marbles := 38
def green_marbles := 4
def dark_blue_marbles := total_marbles - red_marbles - green_marbles

-- The statement to be proven
theorem ratio_dark_blue_to_total : (dark_blue_marbles : ℚ) / total_marbles = 1 / 3 := by
  sorry

end ratio_dark_blue_to_total_l1581_158176


namespace suff_but_not_necc_condition_l1581_158133

def x_sq_minus_1_pos (x : ℝ) : Prop := x^2 - 1 > 0
def x_minus_1_pos (x : ℝ) : Prop := x - 1 > 0

theorem suff_but_not_necc_condition : 
  (∀ x : ℝ, x_minus_1_pos x → x_sq_minus_1_pos x) ∧
  (∃ x : ℝ, x_sq_minus_1_pos x ∧ ¬ x_minus_1_pos x) :=
by 
  sorry

end suff_but_not_necc_condition_l1581_158133


namespace number_added_to_x_is_2_l1581_158154

/-- Prove that in a set of integers {x, x + y, x + 4, x + 7, x + 22}, 
    where the mean is 3 greater than the median, the number added to x 
    to get the second integer is 2. --/

theorem number_added_to_x_is_2 (x y : ℤ) (h_pos : 0 < x ∧ 0 < y) 
  (h_median : (x + 4) = ((x + y) + (x + (x + y) + (x + 4) + (x + 7) + (x + 22)) / 5 - 3)) : 
  y = 2 := by
  sorry

end number_added_to_x_is_2_l1581_158154


namespace difference_between_shares_l1581_158185

def investment_months (amount : ℕ) (months : ℕ) : ℕ :=
  amount * months

def ratio (investment_months : ℕ) (total_investment_months : ℕ) : ℚ :=
  investment_months / total_investment_months

def profit_share (ratio : ℚ) (total_profit : ℝ) : ℝ :=
  ratio * total_profit

theorem difference_between_shares :
  let suresh_investment := 18000
  let rohan_investment := 12000
  let sudhir_investment := 9000
  let suresh_months := 12
  let rohan_months := 9
  let sudhir_months := 8
  let total_profit := 3795
  let suresh_investment_months := investment_months suresh_investment suresh_months
  let rohan_investment_months := investment_months rohan_investment rohan_months
  let sudhir_investment_months := investment_months sudhir_investment sudhir_months
  let total_investment_months := suresh_investment_months + rohan_investment_months + sudhir_investment_months
  let suresh_ratio := ratio suresh_investment_months total_investment_months
  let rohan_ratio := ratio rohan_investment_months total_investment_months
  let sudhir_ratio := ratio sudhir_investment_months total_investment_months
  let rohan_share := profit_share rohan_ratio total_profit
  let sudhir_share := profit_share sudhir_ratio total_profit
  rohan_share - sudhir_share = 345 :=
by
  sorry

end difference_between_shares_l1581_158185


namespace enclosed_region_area_l1581_158116

theorem enclosed_region_area :
  (∃ x y : ℝ, x ^ 2 + y ^ 2 - 6 * x + 8 * y = -9) →
  ∃ (r : ℝ), r ^ 2 = 16 ∧ ∀ (area : ℝ), area = π * 4 ^ 2 :=
by
  sorry

end enclosed_region_area_l1581_158116


namespace total_points_l1581_158178

theorem total_points (gwen_points_per_4 : ℕ) (lisa_points_per_5 : ℕ) (jack_points_per_7 : ℕ) 
                     (gwen_recycled : ℕ) (lisa_recycled : ℕ) (jack_recycled : ℕ)
                     (gwen_ratio : gwen_points_per_4 = 2) (lisa_ratio : lisa_points_per_5 = 3) 
                     (jack_ratio : jack_points_per_7 = 1) (gwen_pounds : gwen_recycled = 12) 
                     (lisa_pounds : lisa_recycled = 25) (jack_pounds : jack_recycled = 21) 
                     : gwen_points_per_4 * (gwen_recycled / 4) + 
                       lisa_points_per_5 * (lisa_recycled / 5) + 
                       jack_points_per_7 * (jack_recycled / 7) = 24 := by
  sorry

end total_points_l1581_158178


namespace platform_length_l1581_158102

/-- Given:
1. The speed of the train is 72 kmph.
2. The train crosses a platform in 32 seconds.
3. The train crosses a man standing on the platform in 18 seconds.

Prove:
The length of the platform is 280 meters.
-/
theorem platform_length
  (train_speed_kmph : ℕ)
  (cross_platform_time_sec cross_man_time_sec : ℕ)
  (h1 : train_speed_kmph = 72)
  (h2 : cross_platform_time_sec = 32)
  (h3 : cross_man_time_sec = 18) :
  ∃ (L_platform : ℕ), L_platform = 280 :=
by
  sorry

end platform_length_l1581_158102


namespace octahedron_volume_l1581_158194

theorem octahedron_volume (a : ℝ) (h1 : a > 0) :
  (∃ V : ℝ, V = (a^3 * Real.sqrt 2) / 3) :=
sorry

end octahedron_volume_l1581_158194


namespace original_profit_percentage_is_10_l1581_158171

-- Define the conditions and the theorem
theorem original_profit_percentage_is_10
  (original_selling_price : ℝ)
  (price_reduction: ℝ)
  (additional_profit: ℝ)
  (profit_percentage: ℝ)
  (new_profit_percentage: ℝ)
  (new_selling_price: ℝ) :
  original_selling_price = 659.9999999999994 →
  price_reduction = 0.10 →
  additional_profit = 42 →
  profit_percentage = 30 →
  new_profit_percentage = 1.30 →
  new_selling_price = original_selling_price + additional_profit →
  ((original_selling_price / (original_selling_price / (new_profit_percentage * (1 - price_reduction)))) - 1) * 100 = 10 :=
by
  sorry

end original_profit_percentage_is_10_l1581_158171


namespace Kim_drink_amount_l1581_158110

namespace MathProof

-- Define the conditions
variable (milk_initial t_drinks k_drinks : ℚ)
variable (H1 : milk_initial = 3/4)
variable (H2 : t_drinks = 1/3 * milk_initial)
variable (H3 : k_drinks = 1/2 * (milk_initial - t_drinks))

-- Theorem statement
theorem Kim_drink_amount : k_drinks = 1/4 :=
by
  sorry -- Proof steps would go here, but we're just setting up the statement

end MathProof

end Kim_drink_amount_l1581_158110


namespace geometric_sequence_a6_l1581_158188

theorem geometric_sequence_a6 (a : ℕ → ℝ) (r : ℝ)
  (h₁ : a 4 = 7)
  (h₂ : a 8 = 63)
  (h_geom : ∀ n, a n = a 1 * r^(n - 1)) :
  a 6 = 21 :=
sorry

end geometric_sequence_a6_l1581_158188


namespace cone_height_of_semicircular_sheet_l1581_158170

theorem cone_height_of_semicircular_sheet (R h : ℝ) (h_cond: h = R) : h = R :=
by
  exact h_cond

end cone_height_of_semicircular_sheet_l1581_158170


namespace amy_minimum_disks_l1581_158161

theorem amy_minimum_disks :
  ∃ (d : ℕ), (d = 19) ∧ ( ∀ (f : ℕ), 
  (f = 40) ∧ ( ∀ (n m k : ℕ), 
  (n + m + k = f) ∧ ( ∀ (a b c : ℕ),
  (a = 8) ∧ (b = 15) ∧ (c = (f - a - b))
  ∧ ( ∀ (size_a size_b size_c : ℚ),
  (size_a = 0.6) ∧ (size_b = 0.55) ∧ (size_c = 0.45)
  ∧ ( ∀ (disk_space : ℚ),
  (disk_space = 1.44)
  ∧ ( ∀ (x y z : ℕ),
  (x = n * ⌈size_a / disk_space⌉) 
  ∧ (y = m * ⌈size_b / disk_space⌉) 
  ∧ (z = k * ⌈size_c / disk_space⌉)
  ∧ (x + y + z = d)) ∧ (size_a * a + size_b * b + size_c * c ≤ disk_space * d)))))) := sorry

end amy_minimum_disks_l1581_158161


namespace desert_area_2020_correct_desert_area_less_8_10_5_by_2023_l1581_158122

-- Define initial desert area
def initial_desert_area : ℝ := 9 * 10^5

-- Define increase in desert area each year as observed
def yearly_increase (n : ℕ) : ℝ :=
  match n with
  | 1998 => 2000
  | 1999 => 4000
  | 2000 => 6001
  | 2001 => 7999
  | 2002 => 10001
  | _    => 0

-- Define arithmetic progression of increases
def common_difference : ℝ := 2000

-- Define desert area in 2020
def desert_area_2020 : ℝ :=
  initial_desert_area + 10001 + 18 * common_difference

-- Statement: Desert area by the end of 2020 is approximately 9.46 * 10^5 hm^2
theorem desert_area_2020_correct :
  desert_area_2020 = 9.46 * 10^5 :=
sorry

-- Define yearly transformation and desert increment with afforestation from 2003
def desert_area_with_afforestation (n : ℕ) : ℝ :=
  if n < 2003 then
    initial_desert_area + yearly_increase n
  else
    initial_desert_area + 10001 + (n - 2002) * (common_difference - 8000)

-- Statement: Desert area will be less than 8 * 10^5 hm^2 by the end of 2023
theorem desert_area_less_8_10_5_by_2023 :
  desert_area_with_afforestation 2023 < 8 * 10^5 :=
sorry

end desert_area_2020_correct_desert_area_less_8_10_5_by_2023_l1581_158122


namespace sum_of_tesseract_elements_l1581_158195

noncomputable def tesseract_edges : ℕ := 32
noncomputable def tesseract_vertices : ℕ := 16
noncomputable def tesseract_faces : ℕ := 24

theorem sum_of_tesseract_elements : tesseract_edges + tesseract_vertices + tesseract_faces = 72 := by
  -- proof here
  sorry

end sum_of_tesseract_elements_l1581_158195


namespace total_travel_cost_is_47100_l1581_158151

-- Define the dimensions of the lawn
def lawn_length : ℝ := 200
def lawn_breadth : ℝ := 150

-- Define the roads' widths and their respective travel costs per sq m
def road1_width : ℝ := 12
def road1_travel_cost : ℝ := 4
def road2_width : ℝ := 15
def road2_travel_cost : ℝ := 5
def road3_width : ℝ := 10
def road3_travel_cost : ℝ := 3
def road4_width : ℝ := 20
def road4_travel_cost : ℝ := 6

-- Define the areas of the roads
def road1_area : ℝ := lawn_length * road1_width
def road2_area : ℝ := lawn_length * road2_width
def road3_area : ℝ := lawn_breadth * road3_width
def road4_area : ℝ := lawn_breadth * road4_width

-- Define the costs for the roads
def road1_cost : ℝ := road1_area * road1_travel_cost
def road2_cost : ℝ := road2_area * road2_travel_cost
def road3_cost : ℝ := road3_area * road3_travel_cost
def road4_cost : ℝ := road4_area * road4_travel_cost

-- Define the total cost
def total_cost : ℝ := road1_cost + road2_cost + road3_cost + road4_cost

-- The theorem statement
theorem total_travel_cost_is_47100 : total_cost = 47100 := by
  sorry

end total_travel_cost_is_47100_l1581_158151


namespace find_g_four_l1581_158156

theorem find_g_four (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x ^ 2) : g 4 = 11 / 2 := 
by
  sorry

end find_g_four_l1581_158156


namespace quadratic_factored_b_l1581_158128

theorem quadratic_factored_b (b : ℤ) : 
  (∃ (m n p q : ℤ), 15 * x^2 + b * x + 30 = (m * x + n) * (p * x + q) ∧ m * p = 15 ∧ n * q = 30 ∧ m * q + n * p = b) ↔ b = 43 :=
by {
  sorry
}

end quadratic_factored_b_l1581_158128


namespace total_cats_and_kittens_received_l1581_158121

theorem total_cats_and_kittens_received (total_adult_cats : ℕ) (percentage_female : ℕ) (fraction_with_kittens : ℚ) (kittens_per_litter : ℕ) 
  (h1 : total_adult_cats = 100) (h2 : percentage_female = 40) (h3 : fraction_with_kittens = 2 / 3) (h4 : kittens_per_litter = 3) :
  total_adult_cats + ((percentage_female * total_adult_cats / 100) * (fraction_with_kittens * total_adult_cats * kittens_per_litter) / 100) = 181 := by
  sorry

end total_cats_and_kittens_received_l1581_158121


namespace david_moore_total_time_l1581_158111

-- Given conditions
def david_work_rate := 1 / 12
def days_david_worked_alone := 6
def remaining_work_days_together := 3
def total_work := 1

-- Definition of total time taken for both to complete the job
def combined_total_time := 6

-- Proof problem statement in Lean
theorem david_moore_total_time :
  let d_work_done_alone := days_david_worked_alone * david_work_rate
  let remaining_work := total_work - d_work_done_alone
  let combined_work_rate := remaining_work / remaining_work_days_together
  let moore_work_rate := combined_work_rate - david_work_rate
  let new_combined_work_rate := david_work_rate + moore_work_rate
  total_work / new_combined_work_rate = combined_total_time := by
    sorry

end david_moore_total_time_l1581_158111


namespace base_conversion_l1581_158189

theorem base_conversion (C D : ℕ) (hC : 0 ≤ C) (hC_lt : C < 8) (hD : 0 ≤ D) (hD_lt : D < 5) :
  (8 * C + D = 5 * D + C) → (8 * C + D = 0) :=
by 
  intro h
  sorry

end base_conversion_l1581_158189


namespace true_root_30_40_l1581_158101

noncomputable def u (x : ℝ) : ℝ := Real.sqrt (x + 15)
noncomputable def original_eqn (x : ℝ) : Prop := u x - 3 / (u x) = 4

theorem true_root_30_40 : ∃ (x : ℝ), 30 < x ∧ x < 40 ∧ original_eqn x :=
by
  sorry

end true_root_30_40_l1581_158101


namespace total_spent_l1581_158134

theorem total_spent (puppy_cost dog_food_cost treats_cost_per_bag toys_cost crate_cost bed_cost collar_leash_cost bags_of_treats discount_rate : ℝ) :
  puppy_cost = 20 →
  dog_food_cost = 20 →
  treats_cost_per_bag = 2.5 →
  toys_cost = 15 →
  crate_cost = 20 →
  bed_cost = 20 →
  collar_leash_cost = 15 →
  bags_of_treats = 2 →
  discount_rate = 0.2 →
  (dog_food_cost + treats_cost_per_bag * bags_of_treats + toys_cost + crate_cost + bed_cost + collar_leash_cost) * (1 - discount_rate) + puppy_cost = 96 :=
by sorry

end total_spent_l1581_158134


namespace perimeter_of_garden_l1581_158147

-- Define the area of the square garden
def area_square_garden : ℕ := 49

-- Define the relationship between q and p
def q_equals_p_plus_21 (q p : ℕ) : Prop := q = p + 21

-- Define the length of the side of the square garden
def side_length (area : ℕ) : ℕ := Nat.sqrt area

-- Define the perimeter of the square garden
def perimeter (side_length : ℕ) : ℕ := 4 * side_length

-- Define the perimeter of the square garden as a specific perimeter
def specific_perimeter (side_length : ℕ) : ℕ := perimeter side_length

-- Statement of the theorem
theorem perimeter_of_garden (q p : ℕ) (h1 : q = 49) (h2 : q_equals_p_plus_21 q p) : 
  specific_perimeter (side_length 49) = 28 := by
  sorry

end perimeter_of_garden_l1581_158147


namespace ticket_number_l1581_158115

-- Define the conditions and the problem
theorem ticket_number (x y z N : ℕ) (hx : 0 ≤ x ∧ x ≤ 9) (hy: 0 ≤ y ∧ y ≤ 9) (hz: 0 ≤ z ∧ z ≤ 9) 
(hN1: N = 100 * x + 10 * y + z) (hN2: N = 11 * (x + y + z)) : 
N = 198 :=
sorry

end ticket_number_l1581_158115


namespace john_marbles_l1581_158165

theorem john_marbles : ∃ m : ℕ, (m ≡ 3 [MOD 7]) ∧ (m ≡ 2 [MOD 4]) ∧ m = 10 := by
  sorry

end john_marbles_l1581_158165


namespace more_stable_performance_l1581_158140

theorem more_stable_performance (S_A2 S_B2 : ℝ) (hA : S_A2 = 0.2) (hB : S_B2 = 0.09) (h : S_A2 > S_B2) : 
  "B" = "B" :=
by
  sorry

end more_stable_performance_l1581_158140


namespace boys_sitting_10_boys_sitting_11_l1581_158113

def exists_two_boys_with_4_between (n : ℕ) : Prop :=
  ∃ (b : Finset ℕ), b.card = n ∧ ∀ (i j : ℕ) (h₁ : i ≠ j) (h₂ : i < 25) (h₃ : j < 25),
    (i + 5) % 25 = j

theorem boys_sitting_10 :
  ¬exists_two_boys_with_4_between 10 :=
sorry

theorem boys_sitting_11 :
  exists_two_boys_with_4_between 11 :=
sorry

end boys_sitting_10_boys_sitting_11_l1581_158113


namespace simplified_expression_value_l1581_158129

noncomputable def expression (a b : ℝ) : ℝ :=
  3 * a ^ 2 - b ^ 2 - (a ^ 2 - 6 * a) - 2 * (-b ^ 2 + 3 * a)

theorem simplified_expression_value :
  expression (-1/2) 3 = 19 / 2 :=
by
  sorry

end simplified_expression_value_l1581_158129


namespace next_in_sequence_is_65_by_19_l1581_158167

section
  open Int

  -- Definitions for numerators
  def numerator_sequence : ℕ → ℤ
  | 0 => -3
  | 1 => 5
  | 2 => -9
  | 3 => 17
  | 4 => -33
  | (n + 5) => numerator_sequence n * (-2) + 1

  -- Definitions for denominators
  def denominator_sequence : ℕ → ℕ
  | 0 => 4
  | 1 => 7
  | 2 => 10
  | 3 => 13
  | 4 => 16
  | (n + 5) => denominator_sequence n + 3

  -- Next term in the sequence
  def next_term (n : ℕ) : ℚ :=
    (numerator_sequence (n + 5) : ℚ) / (denominator_sequence (n + 5) : ℚ)

  -- Theorem stating the next number in the sequence
  theorem next_in_sequence_is_65_by_19 :
    next_term 0 = 65 / 19 :=
  by
    unfold next_term
    simp [numerator_sequence, denominator_sequence]
    sorry
end

end next_in_sequence_is_65_by_19_l1581_158167


namespace lines_perpendicular_l1581_158191

theorem lines_perpendicular 
  (x y : ℝ)
  (first_angle : ℝ)
  (second_angle : ℝ)
  (h1 : first_angle = 50 + x - y)
  (h2 : second_angle = first_angle - (10 + 2 * x - 2 * y)) :
  first_angle + second_angle = 90 :=
by 
  sorry

end lines_perpendicular_l1581_158191


namespace inequality_am_gm_l1581_158145

variable (a b c d : ℝ)

theorem inequality_am_gm (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h_sum : a + b + c + d = 1) :
  (b * c * d) / (1 - a)^2 + (a * c * d) / (1 - b)^2 + (a * b * d) / (1 - c)^2 + (a * b * c) / (1 - d)^2 ≤ 1 / 9  :=
by    
  sorry

end inequality_am_gm_l1581_158145


namespace sum_of_extrema_l1581_158106

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3

-- Main statement to prove
theorem sum_of_extrema :
  let a := -1
  let b := 1
  let f_min := f a
  let f_max := f b
  f_min + f_max = Real.exp 1 + Real.exp (-1) :=
by
  sorry

end sum_of_extrema_l1581_158106


namespace area_of_PQ_square_l1581_158130

theorem area_of_PQ_square (a b c : ℕ)
  (h1 : a^2 = 144)
  (h2 : b^2 = 169)
  (h3 : a^2 + c^2 = b^2) :
  c^2 = 25 :=
by
  sorry

end area_of_PQ_square_l1581_158130


namespace angle_C_ne_5pi_over_6_l1581_158100

-- Define the triangle ∆ABC
variables (A B C : ℝ)

-- Assume the conditions provided
axiom condition_1 : 3 * Real.sin A + 4 * Real.cos B = 6
axiom condition_2 : 3 * Real.cos A + 4 * Real.sin B = 1

-- State that the size of angle C cannot be 5π/6
theorem angle_C_ne_5pi_over_6 : C ≠ 5 * Real.pi / 6 :=
sorry

end angle_C_ne_5pi_over_6_l1581_158100


namespace tan_alpha_problem_l1581_158126

theorem tan_alpha_problem (α : ℝ) (h : Real.tan α = 3) : (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := by
  sorry

end tan_alpha_problem_l1581_158126


namespace function_passes_through_point_l1581_158137

theorem function_passes_through_point :
  (∃ (a : ℝ), a = 1 ∧ (∀ (x y : ℝ), y = a * x + a → y = x + 1)) →
  ∃ x y : ℝ, x = -2 ∧ y = -1 ∧ y = x + 1 :=
by
  sorry

end function_passes_through_point_l1581_158137


namespace intersection_A_B_l1581_158107

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := { x | ∃ k : ℤ, x = 3 * k - 1 }

theorem intersection_A_B :
  A ∩ B = {-1, 2} :=
by
  sorry

end intersection_A_B_l1581_158107


namespace bales_in_barn_l1581_158190

theorem bales_in_barn (stacked today total original : ℕ) (h1 : stacked = 67) (h2 : total = 89) (h3 : total = stacked + original) : original = 22 :=
by
  sorry

end bales_in_barn_l1581_158190


namespace find_sum_of_angles_l1581_158153

open Real

namespace math_problem

theorem find_sum_of_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : cos (α - β / 2) = sqrt 3 / 2)
  (h2 : sin (α / 2 - β) = -1 / 2) : α + β = 2 * π / 3 :=
sorry

end math_problem

end find_sum_of_angles_l1581_158153


namespace range_of_k_l1581_158109

noncomputable section

open Classical

variables {A B C k : ℝ}

def is_acute_triangle (A B C : ℝ) := A < 90 ∧ B < 90 ∧ C < 90

theorem range_of_k (hA : A = 60) (hBC : BC = 6) (h_acute : is_acute_triangle A B C) : 
  2 * Real.sqrt 3 < k ∧ k < 4 * Real.sqrt 3 :=
sorry

end range_of_k_l1581_158109


namespace find_wrong_observation_value_l1581_158143

theorem find_wrong_observation_value :
  ∃ (wrong_value : ℝ),
    let n := 50
    let mean_initial := 36
    let mean_corrected := 36.54
    let observation_incorrect := 48
    let sum_initial := n * mean_initial
    let sum_corrected := n * mean_corrected
    let difference := sum_corrected - sum_initial
    wrong_value = observation_incorrect - difference := sorry

end find_wrong_observation_value_l1581_158143


namespace circle_equation_and_lines_l1581_158197

noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def A : ℝ × ℝ := (6, 2)
noncomputable def B : ℝ × ℝ := (4, 4)
noncomputable def C_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 10

structure Line (κ β: ℝ) where
  passes_through : ℝ × ℝ → Prop
  definition : Prop

def line_passes_through_point (κ β : ℝ) (p : ℝ × ℝ) : Prop := p.2 = κ * p.1 + β

theorem circle_equation_and_lines : 
  (∀ p : ℝ × ℝ, p = O ∨ p = A ∨ p = B → C_eq p.1 p.2) ∧
  ((∀ p : ℝ × ℝ, line_passes_through_point 0 2 p → C_eq 2 6 ∧ (∃ x1 x2 y : ℝ, C_eq x1 y ∧ C_eq x2 y ∧ ((x1 - x2)^2 + (y - y)^2) = 4)) ∧
   (∀ p : ℝ × ℝ, line_passes_through_point (-7 / 3) (32 / 3) p → C_eq 2 6 ∧ (∃ x1 x2 y : ℝ, C_eq x1 y ∧ C_eq x2 y ∧ ((x1 - x2)^2 + (y - y)^2) = 4))) :=
by 
  sorry

end circle_equation_and_lines_l1581_158197


namespace cos_identity_l1581_158172

theorem cos_identity (α : ℝ) (h : Real.cos (π / 4 - α) = -1 / 3) :
  Real.cos (3 * π / 4 + α) = 1 / 3 :=
sorry

end cos_identity_l1581_158172


namespace determine_phi_l1581_158158

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem determine_phi 
  (φ : ℝ)
  (H1 : ∀ x : ℝ, f x φ ≤ |f (π / 6) φ|)
  (H2 : f (π / 3) φ > f (π / 2) φ) :
  φ = π / 6 :=
sorry

end determine_phi_l1581_158158


namespace find_square_sum_l1581_158127

theorem find_square_sum (x y z : ℝ)
  (h1 : x^2 - 6 * y = 10)
  (h2 : y^2 - 8 * z = -18)
  (h3 : z^2 - 10 * x = -40) :
  x^2 + y^2 + z^2 = 50 :=
sorry

end find_square_sum_l1581_158127


namespace cadence_total_earnings_l1581_158117

noncomputable def total_earnings (old_years : ℕ) (old_monthly : ℕ) (new_increment : ℤ) (extra_months : ℕ) : ℤ :=
  let old_months := old_years * 12
  let old_earnings := old_monthly * old_months
  let new_monthly := old_monthly + ((old_monthly * new_increment) / 100)
  let new_months := old_months + extra_months
  let new_earnings := new_monthly * new_months
  old_earnings + new_earnings

theorem cadence_total_earnings :
  total_earnings 3 5000 20 5 = 426000 :=
by
  sorry

end cadence_total_earnings_l1581_158117


namespace smallest_even_n_l1581_158157

theorem smallest_even_n (n : ℕ) :
  (∃ n, 0 < n ∧ n % 2 = 0 ∧ (∀ k, 1 ≤ k → k ≤ n / 2 → k = 2213 ∨ k = 3323 ∨ k = 6121) ∧ (2^k * (k!)) % (2213 * 3323 * 6121) = 0) → n = 12242 :=
sorry

end smallest_even_n_l1581_158157


namespace cat_ratio_l1581_158112

theorem cat_ratio (jacob_cats annie_cats melanie_cats : ℕ)
  (H1 : jacob_cats = 90)
  (H2 : annie_cats = jacob_cats / 3)
  (H3 : melanie_cats = 60) :
  melanie_cats / annie_cats = 2 := 
  by
  sorry

end cat_ratio_l1581_158112


namespace part_1_part_2_l1581_158114

-- Definitions based on given conditions
def a : ℕ → ℝ := λ n => 2 * n + 1
noncomputable def b : ℕ → ℝ := λ n => 1 / ((2 * n + 1)^2 - 1)
noncomputable def S : ℕ → ℝ := λ n => n ^ 2 + 2 * n
noncomputable def T : ℕ → ℝ := λ n => n / (4 * (n + 1))

-- Lean statement for proving the problem
theorem part_1 (n : ℕ) :
  ∀ a_3 a_5 a_7 : ℝ, 
  a 3 = a_3 → 
  a_3 = 7 →
  a_5 = a 5 →
  a_7 = a 7 →
  a_5 + a_7 = 26 →
  ∃ a_1 d : ℝ,
    (a 1 = a_1 + 0 * d) ∧
    (a 2 = a_1 + 1 * d) ∧
    (a 3 = a_1 + 2 * d) ∧
    (a 4 = a_1 + 3 * d) ∧
    (a 5 = a_1 + 4 * d) ∧
    (a 7 = a_1 + 6 * d) ∧
    (a n = a_1 + (n - 1) * d) ∧
    (S n = n^2 + 2*n) := sorry

theorem part_2 (n : ℕ) :
  ∀ a_n b_n : ℝ,
  b n = b_n →
  a n = a_n →
  1 / b n = a_n^2 - 1 →
  T n = τ →
  (T n = n / (4 * (n + 1))) := sorry

end part_1_part_2_l1581_158114


namespace prob1_prob2_prob3_l1581_158184

-- Problem 1
theorem prob1 (k : ℝ) (h₀ : k > 0) 
  (h₁ : ∀ x : ℝ, 2 < x ∧ x < 3 → (k * x^2 - 2 * x + 6 * k) < 0) :
  k = 2/5 := 
sorry

-- Problem 2
theorem prob2 (k : ℝ) (h₀ : k > 0) 
  (h₁ : ∀ x : ℝ, 2 < x ∧ x < 3 → (k * x^2 - 2 * x + 6 * k) < 0) :
  0 < k ∧ k ≤ 2/5 := 
sorry

-- Problem 3
theorem prob3 (k : ℝ) (h₀ : k > 0)
  (h₁ : ∀ x : ℝ, 2 < x ∧ x < 3 → (k * x^2 - 2 * x + 6 * k) < 0) :
  k ≥ 2/5 := 
sorry

end prob1_prob2_prob3_l1581_158184


namespace rented_room_percentage_l1581_158125

theorem rented_room_percentage (total_rooms : ℕ) (h1 : 3 * total_rooms / 4 = 3 * total_rooms / 4) 
                               (h2 : 3 * total_rooms / 5 = 3 * total_rooms / 5) 
                               (h3 : 2 * (3 * total_rooms / 5) / 3 = 2 * (3 * total_rooms / 5) / 3) :
  (1 * (3 * total_rooms / 5) / 5) / (1 * total_rooms / 4) * 100 = 80 := by
  sorry

end rented_room_percentage_l1581_158125


namespace overall_support_percentage_l1581_158159

def men_support_percentage : ℝ := 0.75
def women_support_percentage : ℝ := 0.70
def number_of_men : ℕ := 200
def number_of_women : ℕ := 800

theorem overall_support_percentage :
  ((men_support_percentage * ↑number_of_men + women_support_percentage * ↑number_of_women) / (↑number_of_men + ↑number_of_women) * 100) = 71 := 
by 
sorry

end overall_support_percentage_l1581_158159


namespace quadratic_decreasing_on_nonneg_real_l1581_158166

theorem quadratic_decreasing_on_nonneg_real (a b c : ℝ) (h_a : a < 0) (h_b : b < 0) : 
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → (a * x^2 + b * x + c) ≥ (a * y^2 + b * y + c) :=
by
  sorry

end quadratic_decreasing_on_nonneg_real_l1581_158166


namespace apples_per_basket_l1581_158196

theorem apples_per_basket (total_apples : ℕ) (baskets : ℕ) (h1 : total_apples = 629) (h2 : baskets = 37) :
  total_apples / baskets = 17 :=
by
  sorry

end apples_per_basket_l1581_158196


namespace arithmetic_sequence_general_term_l1581_158193

theorem arithmetic_sequence_general_term (a₁ : ℕ) (d : ℕ) (n : ℕ) (h₁ : a₁ = 2) (h₂ : d = 3) :
  ∃ a_n, a_n = a₁ + (n - 1) * d ∧ a_n = 3 * n - 1 :=
by
  sorry

end arithmetic_sequence_general_term_l1581_158193


namespace ratio_Jake_sister_l1581_158169

theorem ratio_Jake_sister (Jake_weight : ℕ) (total_weight : ℕ) (weight_loss : ℕ) (expected_ratio : ℕ) :
  Jake_weight = 113 →
  total_weight = 153 →
  weight_loss = 33 →
  expected_ratio = 2 →
  (Jake_weight - weight_loss) / (total_weight - Jake_weight) = expected_ratio :=
by
  intros hJake hTotal hLoss hRatio
  sorry

end ratio_Jake_sister_l1581_158169


namespace m_divides_n_l1581_158132

theorem m_divides_n 
  (m n : ℕ) 
  (hm_pos : 0 < m) 
  (hn_pos : 0 < n) 
  (h : 5 * m + n ∣ 5 * n + m) 
  : m ∣ n :=
sorry

end m_divides_n_l1581_158132


namespace sum_of_largest_and_smallest_l1581_158103

theorem sum_of_largest_and_smallest (d1 d2 d3 d4 : ℕ) (h1 : d1 = 1) (h2 : d2 = 6) (h3 : d3 = 3) (h4 : d4 = 9) :
  let largest := 9631
  let smallest := 1369
  largest + smallest = 11000 :=
by
  let largest := 9631
  let smallest := 1369
  sorry

end sum_of_largest_and_smallest_l1581_158103


namespace intersection_A_complement_B_l1581_158198

-- Definition of the universal set U
def U : Set ℝ := Set.univ

-- Definition of the set A
def A : Set ℝ := {x | x^2 - 2 * x < 0}

-- Definition of the set B
def B : Set ℝ := {x | x > 1}

-- Definition of the complement of B in U
def complement_B : Set ℝ := {x | x ≤ 1}

-- The intersection A ∩ complement_B
def intersection : Set ℝ := {x | 0 < x ∧ x ≤ 1}

-- The theorem to prove
theorem intersection_A_complement_B : A ∩ complement_B = intersection :=
by
  -- Proof goes here
  sorry

end intersection_A_complement_B_l1581_158198


namespace find_boys_l1581_158186

-- Variable declarations
variables (B G : ℕ)

-- Conditions
def total_students (B G : ℕ) : Prop := B + G = 466
def more_girls_than_boys (B G : ℕ) : Prop := G = B + 212

-- Proof statement: Prove B = 127 given both conditions
theorem find_boys (h1 : total_students B G) (h2 : more_girls_than_boys B G) : B = 127 :=
sorry

end find_boys_l1581_158186


namespace rotate_right_triangle_along_right_angle_produces_cone_l1581_158105

-- Define a right triangle and the conditions for its rotation
structure RightTriangle (α β γ : ℝ) :=
  (zero_angle : α = 0)
  (ninety_angle_1 : β = 90)
  (ninety_angle_2 : γ = 90)
  (sum_180 : α + β + γ = 180)

-- Define the theorem for the resulting shape when rotating the right triangle
theorem rotate_right_triangle_along_right_angle_produces_cone
  (T : RightTriangle α β γ) (line_of_rotation_contains_right_angle : α = 90 ∨ β = 90 ∨ γ = 90) :
  ∃ shape, shape = "cone" :=
sorry

end rotate_right_triangle_along_right_angle_produces_cone_l1581_158105


namespace odd_function_and_monotonic_decreasing_l1581_158120

variable (f : ℝ → ℝ)

-- Given conditions:
axiom condition_1 : ∀ x y : ℝ, f (x + y) = f x + f y
axiom condition_2 : ∀ x : ℝ, x > 0 → f x < 0

-- Statement to prove:
theorem odd_function_and_monotonic_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x1 x2 : ℝ, x1 > x2 → f x1 < f x2) := by
  sorry

end odd_function_and_monotonic_decreasing_l1581_158120


namespace small_paintings_completed_l1581_158152

variable (S : ℕ)

def uses_paint : Prop :=
  3 * 3 + 2 * S = 17

theorem small_paintings_completed : uses_paint S → S = 4 := by
  intro h
  sorry

end small_paintings_completed_l1581_158152


namespace num_regular_soda_l1581_158136

theorem num_regular_soda (t d r : ℕ) (h₁ : t = 17) (h₂ : d = 8) (h₃ : r = t - d) : r = 9 :=
by
  rw [h₁, h₂] at h₃
  exact h₃

end num_regular_soda_l1581_158136


namespace sum_primes_between_20_and_40_l1581_158177

open Nat

def primesBetween20And40 : List Nat := [23, 29, 31, 37]

theorem sum_primes_between_20_and_40 :
  (primesBetween20And40.sum = 120) :=
by
  sorry

end sum_primes_between_20_and_40_l1581_158177


namespace amount_of_money_C_l1581_158175

variable (A B C : ℝ)

theorem amount_of_money_C (h1 : A + B + C = 500)
                         (h2 : A + C = 200)
                         (h3 : B + C = 360) :
    C = 60 :=
sorry

end amount_of_money_C_l1581_158175


namespace time_in_vancouver_l1581_158148

theorem time_in_vancouver (toronto_time vancouver_time : ℕ) (h : toronto_time = 18 + 30 / 60) (h_diff : vancouver_time = toronto_time - 3) :
  vancouver_time = 15 + 30 / 60 :=
by
  sorry

end time_in_vancouver_l1581_158148


namespace percent_of_x_is_y_l1581_158180

theorem percent_of_x_is_y (x y : ℝ) (h : 0.25 * (x - y) = 0.15 * (x + y)) : y = 0.25 * x := by
  sorry

end percent_of_x_is_y_l1581_158180


namespace expected_value_is_correct_l1581_158144

noncomputable def expected_winnings : ℝ :=
  (1/12 : ℝ) * (9 + 8 + 7 + 6 + 5 + 1 + 2 + 3 + 4 + 5 + 6 + 7)

theorem expected_value_is_correct : expected_winnings = 5.25 := by
  sorry

end expected_value_is_correct_l1581_158144


namespace find_m_value_l1581_158181

theorem find_m_value (x y m : ℤ) (h₁ : x = 2) (h₂ : y = -3) (h₃ : 5 * x + m * y + 2 = 0) : m = 4 := 
by 
  sorry

end find_m_value_l1581_158181


namespace x_value_not_unique_l1581_158183

theorem x_value_not_unique (x y : ℝ) (h1 : y = x) (h2 : y = (|x + y - 2|) / (Real.sqrt 2)) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
(∃ y1 y2 : ℝ, (y1 = x1 ∧ y2 = x2 ∧ y1 = (|x1 + y1 - 2|) / Real.sqrt 2 ∧ y2 = (|x2 + y2 - 2|) / Real.sqrt 2)) :=
by
  sorry

end x_value_not_unique_l1581_158183


namespace cone_surface_area_ratio_l1581_158155

noncomputable def sector_angle := 135
noncomputable def sector_area (B : ℝ) := B
noncomputable def cone (A : ℝ) (B : ℝ) := A

theorem cone_surface_area_ratio (A B : ℝ) (h_sector_angle: sector_angle = 135) (h_sector_area: sector_area B = B) (h_cone_formed: cone A B = A) :
  A / B = 11 / 8 :=
by
  sorry

end cone_surface_area_ratio_l1581_158155


namespace bounded_sequence_exists_l1581_158179

noncomputable def positive_sequence := ℕ → ℝ

variables {a : positive_sequence}

axiom positive_sequence_pos (n : ℕ) : 0 < a n

axiom sequence_condition (k n m l : ℕ) (h : k + n = m + l) : 
  (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)

theorem bounded_sequence_exists 
  (a : positive_sequence) 
  (h_pos : ∀ n, 0 < a n)
  (h_cond : ∀ (k n m l : ℕ), k + n = m + l → 
              (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)) :
  ∃ (b c : ℝ), (0 < b) ∧ (0 < c) ∧ (∀ n, b ≤ a n ∧ a n ≤ c) :=
sorry

end bounded_sequence_exists_l1581_158179


namespace range_of_m_l1581_158164

theorem range_of_m (x : ℝ) (h₁ : 1/2 ≤ x) (h₂ : x ≤ 2) :
  2 - Real.log 2 ≤ -Real.log x + 3*x - x^2 ∧ -Real.log x + 3*x - x^2 ≤ 2 :=
sorry

end range_of_m_l1581_158164


namespace average_output_assembly_line_l1581_158192

theorem average_output_assembly_line (initial_cogs second_batch_cogs rate1 rate2 : ℕ) (time1 time2 : ℚ)
  (h1 : initial_cogs = 60)
  (h2 : second_batch_cogs = 60)
  (h3 : rate1 = 90)
  (h4 : rate2 = 60)
  (h5 : time1 = 60 / 90)
  (h6 : time2 = 60 / 60)
  (h7 : (120 : ℚ) / (time1 + time2) = (72 : ℚ)) :
  (120 : ℚ) / (time1 + time2) = 72 := by
  sorry

end average_output_assembly_line_l1581_158192


namespace tan_11pi_over_6_l1581_158168

theorem tan_11pi_over_6 :
  Real.tan (11 * Real.pi / 6) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_11pi_over_6_l1581_158168


namespace absolute_value_positive_l1581_158146

theorem absolute_value_positive (a : ℝ) (h : a ≠ 0) : |a| > 0 := by
  sorry

end absolute_value_positive_l1581_158146


namespace trapezoid_height_l1581_158163

theorem trapezoid_height (AD BC : ℝ) (AB CD : ℝ) (h₁ : AD = 25) (h₂ : BC = 4) (h₃ : AB = 20) (h₄ : CD = 13) : ∃ h : ℝ, h = 12 :=
by
  -- Definitions
  let AD := 25
  let BC := 4
  let AB := 20
  let CD := 13
  
  sorry

end trapezoid_height_l1581_158163


namespace non_neg_reals_inequality_l1581_158149

theorem non_neg_reals_inequality (a b c : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c)
  (h₃ : a + b + c ≤ 3) :
  (a / (1 + a^2) + b / (1 + b^2) + c / (1 + c^2) ≤ 3/2) ∧
  (3/2 ≤ (1 / (1 + a) + 1 / (1 + b) + 1 / (1 + c))) :=
by
  sorry

end non_neg_reals_inequality_l1581_158149


namespace number_of_masters_students_l1581_158162

theorem number_of_masters_students (total_sample : ℕ) (ratio_assoc : ℕ) (ratio_undergrad : ℕ) (ratio_masters : ℕ) (ratio_doctoral : ℕ) 
(h1 : ratio_assoc = 5) (h2 : ratio_undergrad = 15) (h3 : ratio_masters = 9) (h4 : ratio_doctoral = 1) (h_total_sample : total_sample = 120) :
  (ratio_masters * total_sample) / (ratio_assoc + ratio_undergrad + ratio_masters + ratio_doctoral) = 36 :=
by
  sorry

end number_of_masters_students_l1581_158162


namespace sum_of_distinct_integers_l1581_158108

noncomputable def a : ℤ := 11
noncomputable def b : ℤ := 9
noncomputable def c : ℤ := 4
noncomputable def d : ℤ := 2
noncomputable def e : ℤ := 1

def condition : Prop := (6 - a) * (6 - b) * (6 - c) * (6 - d) * (6 - e) = 120
def distinct_integers : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

theorem sum_of_distinct_integers (h1 : condition) (h2 : distinct_integers) : a + b + c + d + e = 27 :=
by
  sorry

end sum_of_distinct_integers_l1581_158108


namespace new_circle_radius_shaded_region_l1581_158118

theorem new_circle_radius_shaded_region {r1 r2 : ℝ} 
    (h1 : r1 = 35) 
    (h2 : r2 = 24) : 
    ∃ r : ℝ, π * r^2 = π * (r1^2 - r2^2) ∧ r = Real.sqrt 649 := 
by
  sorry

end new_circle_radius_shaded_region_l1581_158118


namespace two_triangles_not_separable_by_plane_l1581_158124

/-- Definition of a point in three-dimensional space -/
structure Point (α : Type) :=
(x : α)
(y : α)
(z : α)

/-- Definition of a segment joining two points -/
structure Segment (α : Type) :=
(p1 : Point α)
(p2 : Point α)

/-- Definition of a triangle formed by three points -/
structure Triangle (α : Type) :=
(a : Point α)
(b : Point α)
(c : Point α)

/-- Definition of a plane given by a normal vector and a point on the plane -/
structure Plane (α : Type) :=
(n : Point α)
(p : Point α)

/-- Definition of separation of two triangles by a plane -/
def separates (plane : Plane ℝ) (t1 t2 : Triangle ℝ) : Prop :=
  -- Placeholder for the actual separation condition
  sorry

/-- The theorem to be proved -/
theorem two_triangles_not_separable_by_plane (points : Fin 6 → Point ℝ) :
  ∃ t1 t2 : Triangle ℝ, ¬∃ plane : Plane ℝ, separates plane t1 t2 :=
sorry

end two_triangles_not_separable_by_plane_l1581_158124


namespace value_of_a_plus_c_l1581_158138

-- Define the polynomials
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a * x + b
def g (c d : ℝ) (x : ℝ) : ℝ := x^2 + c * x + d

-- Define the condition for the vertex of polynomial f being a root of g
def vertex_of_f_is_root_of_g (a b c d : ℝ) : Prop :=
  g c d (-a / 2) = 0

-- Define the condition for the vertex of polynomial g being a root of f
def vertex_of_g_is_root_of_f (a b c d : ℝ) : Prop :=
  f a b (-c / 2) = 0

-- Define the condition that both polynomials have the same minimum value
def same_minimum_value (a b c d : ℝ) : Prop :=
  f a b (-a / 2) = g c d (-c / 2)

-- Define the condition that the polynomials intersect at (100, -100)
def polynomials_intersect (a b c d : ℝ) : Prop :=
  f a b 100 = -100 ∧ g c d 100 = -100

-- Lean theorem statement for the problem
theorem value_of_a_plus_c (a b c d : ℝ) 
  (h1 : vertex_of_f_is_root_of_g a b c d)
  (h2 : vertex_of_g_is_root_of_f a b c d)
  (h3 : same_minimum_value a b c d)
  (h4 : polynomials_intersect a b c d) :
  a + c = -400 := 
sorry

end value_of_a_plus_c_l1581_158138
