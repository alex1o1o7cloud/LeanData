import Mathlib

namespace employees_age_distribution_l600_60015

-- Define the total number of employees
def totalEmployees : ℕ := 15000

-- Define the percentages
def malePercentage : ℝ := 0.58
def femalePercentage : ℝ := 0.42

-- Define the age distribution percentages for male employees
def maleBelow30Percentage : ℝ := 0.25
def male30To50Percentage : ℝ := 0.40
def maleAbove50Percentage : ℝ := 0.35

-- Define the percentage of female employees below 30
def femaleBelow30Percentage : ℝ := 0.30

-- Define the number of male employees
def numMaleEmployees : ℝ := malePercentage * totalEmployees

-- Calculate the number of male employees in each age group
def numMaleBelow30 : ℝ := maleBelow30Percentage * numMaleEmployees
def numMale30To50 : ℝ := male30To50Percentage * numMaleEmployees
def numMaleAbove50 : ℝ := maleAbove50Percentage * numMaleEmployees

-- Define the number of female employees
def numFemaleEmployees : ℝ := femalePercentage * totalEmployees

-- Calculate the number of female employees below 30
def numFemaleBelow30 : ℝ := femaleBelow30Percentage * numFemaleEmployees

-- Calculate the total number of employees below 30
def totalBelow30 : ℝ := numMaleBelow30 + numFemaleBelow30

-- We now state our theorem to prove
theorem employees_age_distribution :
  numMaleBelow30 = 2175 ∧
  numMale30To50 = 3480 ∧
  numMaleAbove50 = 3045 ∧
  totalBelow30 = 4065 := by
    sorry

end employees_age_distribution_l600_60015


namespace no_integer_solutions_3a2_eq_b2_plus_1_l600_60004

theorem no_integer_solutions_3a2_eq_b2_plus_1 :
  ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 :=
by
  sorry

end no_integer_solutions_3a2_eq_b2_plus_1_l600_60004


namespace find_f_2002_l600_60062

-- Definitions based on conditions
variable {R : Type} [CommRing R] [NoZeroDivisors R]

-- Condition 1: f is an even function.
def even_function (f : R → R) : Prop :=
  ∀ x : R, f (-x) = f x

-- Condition 2: f(2) = 0
def f_value_at_two (f : R → R) : Prop :=
  f 2 = 0

-- Condition 3: g is an odd function.
def odd_function (g : R → R) : Prop :=
  ∀ x : R, g (-x) = -g x

-- Condition 4: g(x) = f(x-1)
def g_equals_f_shifted (f g : R → R) : Prop :=
  ∀ x : R, g x = f (x - 1)

-- The main proof problem
theorem find_f_2002 (f g : R → R)
  (hf : even_function f)
  (hf2 : f_value_at_two f)
  (hg : odd_function g)
  (hgf : g_equals_f_shifted f g) :
  f 2002 = 0 :=
sorry

end find_f_2002_l600_60062


namespace senior_ticket_cost_l600_60023

variable (tickets_total : ℕ)
variable (adult_ticket_price senior_ticket_price : ℕ)
variable (total_receipts : ℕ)
variable (senior_tickets_sold : ℕ)

theorem senior_ticket_cost (h1 : tickets_total = 529) 
                           (h2 : adult_ticket_price = 25)
                           (h3 : total_receipts = 9745)
                           (h4 : senior_tickets_sold = 348) 
                           (h5 : senior_ticket_price * 348 + 25 * (529 - 348) = 9745) : 
                           senior_ticket_price = 15 := by
  sorry

end senior_ticket_cost_l600_60023


namespace find_total_buffaloes_l600_60032

-- Define the problem parameters.
def number_of_ducks : ℕ := sorry
def number_of_cows : ℕ := 8

-- Define the conditions.
def duck_legs : ℕ := 2 * number_of_ducks
def cow_legs : ℕ := 4 * number_of_cows
def total_heads : ℕ := number_of_ducks + number_of_cows

-- The given equation as a condition.
def total_legs : ℕ := duck_legs + cow_legs

-- Translate condition from the problem:
def condition : Prop := total_legs = 2 * total_heads + 16

-- The proof statement.
theorem find_total_buffaloes : number_of_cows = 8 :=
by
  -- Place the placeholder proof here.
  sorry

end find_total_buffaloes_l600_60032


namespace calc_hash_80_l600_60005

def hash (N : ℝ) : ℝ := 0.4 * N * 1.5

theorem calc_hash_80 : hash (hash (hash 80)) = 17.28 :=
by 
  sorry

end calc_hash_80_l600_60005


namespace greatest_integer_leq_fraction_l600_60084

theorem greatest_integer_leq_fraction (N D : ℝ) (hN : N = 4^103 + 3^103 + 2^103) (hD : D = 4^100 + 3^100 + 2^100) :
  ⌊N / D⌋ = 64 :=
by
  sorry

end greatest_integer_leq_fraction_l600_60084


namespace find_b_c_l600_60033

-- Definitions and the problem statement
theorem find_b_c (b c : ℝ) (x1 x2 : ℝ) (h1 : x1 = 1) (h2 : x2 = -2) 
  (h_eq : ∀ x, x^2 - b * x + c = (x - x1) * (x - x2)) :
  b = -1 ∧ c = -2 :=
by
  sorry

end find_b_c_l600_60033


namespace solution_set_l600_60076

variable (x : ℝ)

noncomputable def expr := (x - 1)^2 / (x - 5)^2

theorem solution_set :
  { x : ℝ | expr x ≥ 0 } = { x | x < 5 } ∪ { x | x > 5 } :=
by
  sorry

end solution_set_l600_60076


namespace sin_330_value_l600_60071

noncomputable def sin_330 : ℝ := Real.sin (330 * Real.pi / 180)

theorem sin_330_value : sin_330 = -1/2 :=
by {
  sorry
}

end sin_330_value_l600_60071


namespace electric_sharpens_more_l600_60019

noncomputable def number_of_pencils_hand_crank : ℕ := 360 / 45
noncomputable def number_of_pencils_electric : ℕ := 360 / 20

theorem electric_sharpens_more : number_of_pencils_electric - number_of_pencils_hand_crank = 10 := by
  sorry

end electric_sharpens_more_l600_60019


namespace kite_area_is_28_l600_60094

noncomputable def area_of_kite : ℝ :=
  let base_upper := 8
  let height_upper := 2
  let base_lower := 8
  let height_lower := 5
  let area_upper := (1 / 2 : ℝ) * base_upper * height_upper
  let area_lower := (1 / 2 : ℝ) * base_lower * height_lower
  area_upper + area_lower

theorem kite_area_is_28 :
  area_of_kite = 28 :=
by
  simp [area_of_kite]
  sorry

end kite_area_is_28_l600_60094


namespace debby_weekly_jog_distance_l600_60048

theorem debby_weekly_jog_distance :
  let monday_distance := 3.0
  let tuesday_distance := 5.5
  let wednesday_distance := 9.7
  let thursday_distance := 10.8
  let friday_distance_miles := 2.0
  let miles_to_km := 1.60934
  let friday_distance := friday_distance_miles * miles_to_km
  let total_distance := monday_distance + tuesday_distance + wednesday_distance + thursday_distance + friday_distance
  total_distance = 32.21868 :=
by
  sorry

end debby_weekly_jog_distance_l600_60048


namespace gardener_works_days_l600_60073

theorem gardener_works_days :
  let rose_bushes := 20
  let cost_per_rose_bush := 150
  let gardener_hourly_wage := 30
  let gardener_hours_per_day := 5
  let soil_volume := 100
  let cost_per_soil := 5
  let total_project_cost := 4100
  let total_gardening_days := 4
  (rose_bushes * cost_per_rose_bush + soil_volume * cost_per_soil + total_gardening_days * gardener_hours_per_day * gardener_hourly_wage = total_project_cost) →
  total_gardening_days = 4 :=
by
  intros
  sorry

end gardener_works_days_l600_60073


namespace sum_is_two_l600_60044

-- Define the numbers based on conditions
def a : Int := 9
def b : Int := -9 + 2

-- Theorem stating that the sum of the two numbers is 2
theorem sum_is_two : a + b = 2 :=
by
  -- proof goes here
  sorry

end sum_is_two_l600_60044


namespace possibleValuesOfSum_l600_60060

noncomputable def symmetricMatrixNonInvertible (x y z : ℝ) : Prop := 
  -(x + y + z) * ( x^2 + y^2 + z^2 - x * y - x * z - y * z ) = 0

theorem possibleValuesOfSum (x y z : ℝ) (h : symmetricMatrixNonInvertible x y z) :
  ∃ v : ℝ, v = -3 ∨ v = 3 / 2 := 
sorry

end possibleValuesOfSum_l600_60060


namespace arcsin_cos_eq_l600_60063

theorem arcsin_cos_eq :
  Real.arcsin (Real.cos (2 * Real.pi / 3)) = -Real.pi / 6 :=
by
  have h1 : Real.cos (2 * Real.pi / 3) = -1 / 2 := sorry
  have h2 : Real.arcsin (-1 / 2) = -Real.pi / 6 := sorry
  rw [h1, h2]

end arcsin_cos_eq_l600_60063


namespace pizza_slices_per_pizza_l600_60000

theorem pizza_slices_per_pizza (num_coworkers slices_per_person num_pizzas : ℕ) (h1 : num_coworkers = 12) (h2 : slices_per_person = 2) (h3 : num_pizzas = 3) :
  (num_coworkers * slices_per_person) / num_pizzas = 8 :=
by
  sorry

end pizza_slices_per_pizza_l600_60000


namespace co_complementary_angles_equal_l600_60064

def co_complementary (A : ℝ) : ℝ := 90 - A

theorem co_complementary_angles_equal (A B : ℝ) (h : co_complementary A = co_complementary B) : A = B :=
sorry

end co_complementary_angles_equal_l600_60064


namespace total_earning_l600_60035

theorem total_earning (days_a days_b days_c : ℕ) (wage_ratio_a wage_ratio_b wage_ratio_c daily_wage_c total : ℕ)
  (h_ratio : wage_ratio_a = 3 ∧ wage_ratio_b = 4 ∧ wage_ratio_c = 5)
  (h_days : days_a = 6 ∧ days_b = 9 ∧ days_c = 4)
  (h_daily_wage_c : daily_wage_c = 125)
  (h_total : total = ((wage_ratio_a * (daily_wage_c / wage_ratio_c) * days_a) +
                     (wage_ratio_b * (daily_wage_c / wage_ratio_c) * days_b) +
                     (daily_wage_c * days_c))) : total = 1850 := by
  sorry

end total_earning_l600_60035


namespace HeatherIsHeavier_l600_60079

-- Definitions
def HeatherWeight : ℕ := 87
def EmilyWeight : ℕ := 9

-- Theorem statement
theorem HeatherIsHeavier : HeatherWeight - EmilyWeight = 78 := by
  sorry

end HeatherIsHeavier_l600_60079


namespace greatest_c_value_l600_60068

theorem greatest_c_value (c : ℤ) : 
  (∀ (x : ℝ), x^2 + (c : ℝ) * x + 20 ≠ -7) → c = 10 :=
by
  sorry

end greatest_c_value_l600_60068


namespace problem1_problem2_l600_60016

-- Definitions of the sets A, B, and C based on conditions given
def setA : Set ℝ := {x | x^2 + 2*x - 8 ≥ 0}
def setB : Set ℝ := {x | Real.sqrt (9 - 3*x) ≤ Real.sqrt (2*x + 19)}
def setC (a : ℝ) : Set ℝ := {x | x^2 + 2*a*x + 2 ≤ 0}

-- Problem (1): Prove values of b and c
theorem problem1 (b c : ℝ) :
  (∀ x, x ∈ (setA ∩ setB) ↔ b*x^2 + 10*x + c ≥ 0) → b = -2 ∧ c = -12 := sorry

-- Universal set definition and its complement
def universalSet : Set ℝ := {x | True}
def complementA : Set ℝ := {x | (x ∉ setA)}

-- Problem (2): Range of a
theorem problem2 (a : ℝ) :
  (setC a ⊆ setB ∪ complementA) → a ∈ Set.Icc (-11/6) (9/4) := sorry

end problem1_problem2_l600_60016


namespace first_car_gas_consumed_l600_60009

theorem first_car_gas_consumed 
    (sum_avg_mpg : ℝ) (g2_gallons : ℝ) (total_miles : ℝ) 
    (avg_mpg_car1 : ℝ) (avg_mpg_car2 : ℝ) (g1_gallons : ℝ) :
    sum_avg_mpg = avg_mpg_car1 + avg_mpg_car2 →
    g2_gallons = 35 →
    total_miles = 2275 →
    avg_mpg_car1 = 40 →
    avg_mpg_car2 = 35 →
    g1_gallons = (total_miles - (avg_mpg_car2 * g2_gallons)) / avg_mpg_car1 →
    g1_gallons = 26.25 :=
by
  intros h_sum_avg_mpg h_g2_gallons h_total_miles h_avg_mpg_car1 h_avg_mpg_car2 h_g1_gallons
  sorry

end first_car_gas_consumed_l600_60009


namespace find_number_l600_60020

theorem find_number (x : ℝ) (h : -200 * x = 1600) : x = -8 :=
by sorry

end find_number_l600_60020


namespace rectangle_dimension_area_l600_60061

theorem rectangle_dimension_area (x : Real) 
  (h_dim1 : x + 3 > 0) 
  (h_dim2 : 3 * x - 2 > 0) :
  ((x + 3) * (3 * x - 2) = 9 * x + 1) ↔ x = (11 + Real.sqrt 205) / 6 := 
sorry

end rectangle_dimension_area_l600_60061


namespace minimum_score_for_fourth_term_l600_60053

variable (score1 score2 score3 score4 : ℕ)
variable (avg_required : ℕ)

theorem minimum_score_for_fourth_term :
  score1 = 80 →
  score2 = 78 →
  score3 = 76 →
  avg_required = 85 →
  4 * avg_required - (score1 + score2 + score3) ≤ score4 :=
by
  sorry

end minimum_score_for_fourth_term_l600_60053


namespace solve_system_of_inequalities_l600_60099

theorem solve_system_of_inequalities (x : ℝ) : 
  (3 * x > x - 4) ∧ ((4 + x) / 3 > x + 2) → -2 < x ∧ x < -1 :=
by {
  sorry
}

end solve_system_of_inequalities_l600_60099


namespace find_f_neg_2_l600_60022

def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 4

variable (a b : ℝ)

theorem find_f_neg_2 (h1 : f a b 2 = 6) : f a b (-2) = -14 :=
by
  sorry

end find_f_neg_2_l600_60022


namespace intersection_M_N_l600_60025

def M := {p : ℝ × ℝ | p.snd = 2 - p.fst}
def N := {p : ℝ × ℝ | p.fst - p.snd = 4}
def intersection := {p : ℝ × ℝ | p = (3, -1)}

theorem intersection_M_N : M ∩ N = intersection := 
by sorry

end intersection_M_N_l600_60025


namespace amanda_car_round_trip_time_l600_60028

theorem amanda_car_round_trip_time :
  let bus_time := 40
  let bus_distance := 120
  let detour := 15
  let reduced_time := 5
  let amanda_trip_one_way_time := bus_time - reduced_time
  let amanda_round_trip_distance := (bus_distance * 2) + (detour * 2)
  let required_time := amanda_round_trip_distance * amanda_trip_one_way_time / bus_distance
  required_time = 79 :=
by
  sorry

end amanda_car_round_trip_time_l600_60028


namespace cos_75_eq_l600_60021

theorem cos_75_eq : Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_75_eq_l600_60021


namespace jerry_water_usage_l600_60059

noncomputable def total_water_usage 
  (drinking_cooking : ℕ) 
  (shower_per_gallon : ℕ) 
  (length width height : ℕ) 
  (gallon_per_cubic_ft : ℕ) 
  (number_of_showers : ℕ) 
  : ℕ := 
   drinking_cooking + 
   (number_of_showers * shower_per_gallon) + 
   (length * width * height / gallon_per_cubic_ft)

theorem jerry_water_usage 
  (drinking_cooking : ℕ := 100)
  (shower_per_gallon : ℕ := 20)
  (length : ℕ := 10)
  (width : ℕ := 10)
  (height : ℕ := 6)
  (gallon_per_cubic_ft : ℕ := 1)
  (number_of_showers : ℕ := 15)
  : total_water_usage drinking_cooking shower_per_gallon length width height gallon_per_cubic_ft number_of_showers = 1400 := 
by
  sorry

end jerry_water_usage_l600_60059


namespace problem_a_problem_b_problem_c_l600_60038

noncomputable def inequality_a (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : Prop :=
  (1 / (x * (0 * y + 1)) + 1 / (y * (0 * z + 1)) + 1 / (z * (0 * x + 1))) ≥ 3

noncomputable def inequality_b (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : Prop :=
  (1 / (x * (1 * y + 0)) + 1 / (y * (1 * z + 0)) + 1 / (z * (1 * x + 0))) ≥ 3

noncomputable def inequality_c (x y z : ℝ) (a b : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) (h5 : a + b = 1) (h6 : a > 0) (h7 : b > 0) : Prop :=
  (1 / (x * (a * y + b)) + 1 / (y * (a * z + b)) + 1 / (z * (a * x + b))) ≥ 3

theorem problem_a (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : inequality_a x y z h1 h2 h3 h4 := 
  by sorry

theorem problem_b (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : inequality_b x y z h1 h2 h3 h4 := 
  by sorry

theorem problem_c (x y z a b : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) (h5 : a + b = 1) (h6 : a > 0) (h7 : b > 0) : inequality_c x y z a b h1 h2 h3 h4 h5 h6 h7 :=
  by sorry

end problem_a_problem_b_problem_c_l600_60038


namespace ex3_solutions_abs_eq_l600_60006

theorem ex3_solutions_abs_eq (a : ℝ) : (∃ x1 x2 x3 x4 : ℝ, 
        2 * abs (abs (x1 - 1) - 3) = a ∧ 
        2 * abs (abs (x2 - 1) - 3) = a ∧ 
        2 * abs (abs (x3 - 1) - 3) = a ∧ 
        2 * abs (abs (x4 - 1) - 3) = a ∧ 
        x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ (x1 = x4 ∨ x2 = x4 ∨ x3 = x4)) ↔ a = 6 :=
by
    sorry

end ex3_solutions_abs_eq_l600_60006


namespace find_m_l600_60049

variables (m x y : ℤ)

-- Conditions
def cond1 := x = 3 * m + 1
def cond2 := y = 2 * m - 2
def cond3 := 4 * x - 3 * y = 10

theorem find_m (h1 : cond1 m x) (h2 : cond2 m y) (h3 : cond3 x y) : m = 0 :=
by sorry

end find_m_l600_60049


namespace apples_distribution_l600_60008

variable (x : ℕ)

theorem apples_distribution :
  0 ≤ 5 * x + 12 - 8 * (x - 1) ∧ 5 * x + 12 - 8 * (x - 1) < 8 :=
sorry

end apples_distribution_l600_60008


namespace composite_A_l600_60027

def A : ℕ := 10^1962 + 1

theorem composite_A : ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ A = p * q :=
  sorry

end composite_A_l600_60027


namespace geraldine_banana_count_l600_60037

variable (b : ℕ) -- the number of bananas Geraldine ate on June 1

theorem geraldine_banana_count 
    (h1 : (5 * b + 80 = 150)) 
    : (b + 32 = 46) :=
by
  sorry

end geraldine_banana_count_l600_60037


namespace pool_one_quarter_capacity_in_six_hours_l600_60007

theorem pool_one_quarter_capacity_in_six_hours (d : ℕ → ℕ) :
  (∀ n : ℕ, d (n + 1) = 2 * d n) → d 8 = 2^8 →
  d 6 = 2^6 :=
by
  intros h1 h2
  sorry

end pool_one_quarter_capacity_in_six_hours_l600_60007


namespace rectangle_area_inscribed_circle_l600_60067

theorem rectangle_area_inscribed_circle (r l w : ℝ) (h_r : r = 7)
(h_ratio : l / w = 2) (h_w : w = 2 * r) :
  l * w = 392 :=
by sorry

end rectangle_area_inscribed_circle_l600_60067


namespace max_quotient_l600_60010

theorem max_quotient (a b : ℕ) 
  (h1 : 400 ≤ a) (h2 : a ≤ 800) 
  (h3 : 400 ≤ b) (h4 : b ≤ 1600) 
  (h5 : a + b ≤ 2000) 
  : b / a ≤ 4 := 
sorry

end max_quotient_l600_60010


namespace percent_psychology_majors_l600_60054

theorem percent_psychology_majors
  (total_students : ℝ)
  (pct_freshmen : ℝ)
  (pct_freshmen_liberal_arts : ℝ)
  (pct_freshmen_psychology_majors : ℝ)
  (h1 : pct_freshmen = 0.6)
  (h2 : pct_freshmen_liberal_arts = 0.4)
  (h3 : pct_freshmen_psychology_majors = 0.048)
  :
  (pct_freshmen_psychology_majors / (pct_freshmen * pct_freshmen_liberal_arts)) * 100 = 20 := 
by
  sorry

end percent_psychology_majors_l600_60054


namespace simplify_expression_l600_60085

theorem simplify_expression (b : ℝ) :
  (1 * (2 * b) * (3 * b^2) * (4 * b^3) * (5 * b^4) * (6 * b^5)) = 720 * b^15 :=
by
  sorry

end simplify_expression_l600_60085


namespace solve_xyz_integers_l600_60041

theorem solve_xyz_integers (x y z : ℤ) : x^2 + y^2 + z^2 = 2 * x * y * z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end solve_xyz_integers_l600_60041


namespace intersection_A_B_l600_60013

def setA (x : ℝ) : Prop := x^2 - 2 * x > 0
def setB (x : ℝ) : Prop := abs (x + 1) < 2

theorem intersection_A_B :
  {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -3 < x ∧ x < 0} :=
by
  sorry

end intersection_A_B_l600_60013


namespace optimal_production_distribution_l600_60034

noncomputable def min_production_time (unitsI_A unitsI_B unitsII_B : ℕ) : ℕ :=
let rateI_A := 30
let rateII_B := 40
let rateI_B := 50
let initial_days_B := 20
let remaining_units_I := 1500 - (rateI_A * initial_days_B)
let combined_rateI_AB := rateI_A + rateI_B
let days_remaining_I := remaining_units_I / combined_rateI_AB
initial_days_B + days_remaining_I

theorem optimal_production_distribution :
  ∃ (unitsI_A unitsI_B unitsII_B : ℕ),
    unitsI_A + unitsI_B = 1500 ∧ unitsII_B = 800 ∧
    min_production_time unitsI_A unitsI_B unitsII_B = 31 := sorry

end optimal_production_distribution_l600_60034


namespace root_of_polynomial_l600_60066

theorem root_of_polynomial (k : ℝ) (h : (3 : ℝ) ^ 4 + k * (3 : ℝ) ^ 2 + 27 = 0) : k = -12 :=
by
  sorry

end root_of_polynomial_l600_60066


namespace geometric_sequence_sum_l600_60086

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) 
  (h1 : ∀ n, a (n + 1) = r * a n)
  (h2 : 0 < r)
  (h3 : a 1 = 3)
  (h4 : a 1 + a 2 + a 3 = 21) :
  a 3 + a 4 + a 5 = 84 :=
sorry

end geometric_sequence_sum_l600_60086


namespace candle_ratio_l600_60003

theorem candle_ratio (r b : ℕ) (h1: r = 45) (h2: b = 27) : r / Nat.gcd r b = 5 ∧ b / Nat.gcd r b = 3 := 
by
  sorry

end candle_ratio_l600_60003


namespace sqrt_factorial_mul_factorial_eq_l600_60058

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l600_60058


namespace circle_second_x_intercept_l600_60030

theorem circle_second_x_intercept :
  ∀ (circle : ℝ × ℝ → Prop), (∀ (x y : ℝ), circle (x, y) ↔ (x - 5) ^ 2 + y ^ 2 = 25) →
    ∃ x : ℝ, (x ≠ 0 ∧ circle (x, 0) ∧ x = 10) :=
by {
  sorry
}

end circle_second_x_intercept_l600_60030


namespace expected_left_handed_l600_60014

theorem expected_left_handed (p : ℚ) (n : ℕ) (h : p = 1/6) (hs : n = 300) : n * p = 50 :=
by 
  -- Proof goes here
  sorry

end expected_left_handed_l600_60014


namespace temperature_reaches_100_at_5_hours_past_noon_l600_60018

theorem temperature_reaches_100_at_5_hours_past_noon :
  ∃ t : ℝ, (-2 * t^2 + 16 * t + 40 = 100) ∧ ∀ t' : ℝ, (-2 * t'^2 + 16 * t' + 40 = 100) → 5 ≤ t' :=
by
  -- We skip the proof and assume the theorem is true.
  sorry

end temperature_reaches_100_at_5_hours_past_noon_l600_60018


namespace minimize_sum_of_legs_l600_60017

noncomputable def area_of_right_angle_triangle (a b : ℝ) : Prop :=
  1/2 * a * b = 50

theorem minimize_sum_of_legs (a b : ℝ) (h : area_of_right_angle_triangle a b) :
  a + b = 20 ↔ a = 10 ∧ b = 10 :=
by
  sorry

end minimize_sum_of_legs_l600_60017


namespace shaded_fraction_is_four_fifteenths_l600_60089

noncomputable def shaded_fraction : ℚ :=
  let a := (1/4 : ℚ)
  let r := (1/16 : ℚ)
  a / (1 - r)

theorem shaded_fraction_is_four_fifteenths :
  shaded_fraction = (4 / 15 : ℚ) := sorry

end shaded_fraction_is_four_fifteenths_l600_60089


namespace range_of_a_for_domain_of_f_l600_60077

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := sqrt (-5 / (a * x^2 + a * x - 3))

theorem range_of_a_for_domain_of_f :
  {a : ℝ | ∀ x : ℝ, a * x^2 + a * x - 3 < 0} = {a : ℝ | -12 < a ∧ a ≤ 0} :=
by
  sorry

end range_of_a_for_domain_of_f_l600_60077


namespace probability_after_5_rounds_l600_60095

def initial_coins : ℕ := 5
def rounds : ℕ := 5
def final_probability : ℚ := 1 / 2430000

structure Player :=
  (name : String)
  (initial_coins : ℕ)
  (final_coins : ℕ)

def Abby : Player := ⟨"Abby", 5, 5⟩
def Bernardo : Player := ⟨"Bernardo", 4, 3⟩
def Carl : Player := ⟨"Carl", 3, 3⟩
def Debra : Player := ⟨"Debra", 4, 5⟩

def check_final_state (players : List Player) : Prop :=
  ∀ (p : Player), p ∈ players →
  (p.name = "Abby" ∧ p.final_coins = 5 ∨
   p.name = "Bernardo" ∧ p.final_coins = 3 ∨
   p.name = "Carl" ∧ p.final_coins = 3 ∨
   p.name = "Debra" ∧ p.final_coins = 5)

theorem probability_after_5_rounds :
  ∃ prob : ℚ, prob = final_probability ∧ check_final_state [Abby, Bernardo, Carl, Debra] :=
sorry

end probability_after_5_rounds_l600_60095


namespace arithmetic_seq_proof_l600_60031

noncomputable def arithmetic_sequence : Type := ℕ → ℝ

variables (a : ℕ → ℝ) (d : ℝ)

def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

variables (a₁ a₂ a₃ a₄ : ℝ)
variables (h1 : a 1 + a 2 = 10)
variables (h2 : a 4 = a 3 + 2)
variables (h3 : is_arithmetic_seq a d)

theorem arithmetic_seq_proof :
  a 3 + a 4 = 18 :=
sorry

end arithmetic_seq_proof_l600_60031


namespace monopoly_favor_durable_machine_competitive_market_prefer_durable_l600_60002

-- Define the conditions
def consumer_valuation : ℕ := 10
def durable_cost : ℕ := 6

-- Define the monopoly decision problem: prove C > 3
theorem monopoly_favor_durable_machine (C : ℕ) : 
  consumer_valuation * 2 - durable_cost > 2 * (consumer_valuation - C) → C > 3 := 
by 
  sorry

-- Define the competitive market decision problem: prove C > 3
theorem competitive_market_prefer_durable (C : ℕ) :
  2 * consumer_valuation - durable_cost > 2 * (consumer_valuation - C) → C > 3 := 
by 
  sorry

end monopoly_favor_durable_machine_competitive_market_prefer_durable_l600_60002


namespace negation_proof_l600_60047

theorem negation_proof :
  (¬ ∀ x : ℝ, x > 0 → x + 1/x ≥ 2) ↔ (∃ x : ℝ, x > 0 ∧ x + 1/x < 2) :=
by
  sorry

end negation_proof_l600_60047


namespace max_area_of_triangle_MAN_l600_60087

noncomputable def maximum_area_triangle_MAN (e : ℝ) (F : ℝ × ℝ) (A : ℝ × ℝ) : ℝ :=
  if h : e = Real.sqrt 3 / 2 ∧ F = (Real.sqrt 3, 0) ∧ A = (1, 1 / 2) then
    Real.sqrt 2
  else
    0

theorem max_area_of_triangle_MAN :
  maximum_area_triangle_MAN (Real.sqrt 3 / 2) (Real.sqrt 3, 0) (1, 1 / 2) = Real.sqrt 2 :=
by
  sorry

end max_area_of_triangle_MAN_l600_60087


namespace total_age_is_47_l600_60052

-- Define the ages of B and conditions
def B : ℕ := 18
def A : ℕ := B + 2
def C : ℕ := B / 2

-- Prove the total age of A, B, and C
theorem total_age_is_47 : A + B + C = 47 :=
by
  sorry

end total_age_is_47_l600_60052


namespace quadratic_inequality_solution_l600_60090

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 2 * x + 1 > 0) ↔ (a > 1) :=
by
  sorry

end quadratic_inequality_solution_l600_60090


namespace sqrt_expression_eq_seven_div_two_l600_60065

theorem sqrt_expression_eq_seven_div_two :
  (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 12 / Real.sqrt 24) = 7 / 2 :=
by
  sorry

end sqrt_expression_eq_seven_div_two_l600_60065


namespace final_position_3000_l600_60050

def initial_position : ℤ × ℤ := (0, 0)
def moves_up_first_minute (pos : ℤ × ℤ) : ℤ × ℤ := (pos.1, pos.2 + 1)

def next_position (n : ℕ) (pos : ℤ × ℤ) : ℤ × ℤ :=
  if n % 4 = 0 then (pos.1 + n, pos.2)
  else if n % 4 = 1 then (pos.1, pos.2 + n)
  else if n % 4 = 2 then (pos.1 - n, pos.2)
  else (pos.1, pos.2 - n)

def final_position (minutes : ℕ) : ℤ × ℤ := sorry

theorem final_position_3000 : final_position 3000 = (0, 27) :=
by {
  -- logic to compute final_position
  sorry -- proof exists here
}

end final_position_3000_l600_60050


namespace parallelepiped_diagonal_relationship_l600_60012

theorem parallelepiped_diagonal_relationship {a b c d e f g : ℝ} 
  (h1 : c = d) 
  (h2 : e = e) 
  (h3 : f = f) 
  (h4 : g = g) 
  : a^2 + b^2 + c^2 + g^2 = d^2 + e^2 + f^2 :=
by
  sorry

end parallelepiped_diagonal_relationship_l600_60012


namespace m_eq_n_is_necessary_but_not_sufficient_l600_60091

noncomputable def circle_condition (m n : ℝ) : Prop :=
  m = n ∧ m > 0

theorem m_eq_n_is_necessary_but_not_sufficient 
  (m n : ℝ) :
  (circle_condition m n → mx^2 + ny^2 = 3 → False) ∧
  (mx^2 + ny^2 = 3 → circle_condition m n) :=
by 
  sorry

end m_eq_n_is_necessary_but_not_sufficient_l600_60091


namespace domain_fraction_function_l600_60026

theorem domain_fraction_function (f : ℝ → ℝ):
  (∀ x : ℝ, -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 0) →
  (∀ x : ℝ, x ≠ 0 → -2 ≤ x ∧ x < 0) →
  (∀ x, (2^x - 1) ≠ 0) →
  true := sorry

end domain_fraction_function_l600_60026


namespace value_of_expression_at_3_l600_60046

theorem value_of_expression_at_3 :
  ∀ (x : ℕ), x = 3 → (x^4 - 6 * x) = 63 :=
by
  intros x h
  sorry

end value_of_expression_at_3_l600_60046


namespace range_of_z_l600_60081

theorem range_of_z (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
(h₁ : x + y = x * y) (h₂ : x + y + z = x * y * z) : 1 < z ∧ z ≤ 4 / 3 :=
sorry

end range_of_z_l600_60081


namespace red_to_green_speed_ratio_l600_60042

-- Conditions
def blue_car_speed : Nat := 80 -- The blue car's speed is 80 miles per hour
def green_car_speed : Nat := 8 * blue_car_speed -- The green car's speed is 8 times the blue car's speed
def red_car_speed : Nat := 1280 -- The red car's speed is 1280 miles per hour

-- Theorem stating the ratio of red car's speed to green car's speed
theorem red_to_green_speed_ratio : red_car_speed / green_car_speed = 2 := by
  sorry -- proof goes here

end red_to_green_speed_ratio_l600_60042


namespace simplify_fraction_l600_60078

theorem simplify_fraction (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) : 
  (15 * x^2 * y^4 * z^2) / (9 * x * y^3 * z) = 10 := 
by
  sorry

end simplify_fraction_l600_60078


namespace ratio_of_perimeters_l600_60024

theorem ratio_of_perimeters (s : ℝ) (hs : s > 0) :
  let small_triangle_perimeter := s + (s / 2) + (s / 2)
  let large_rectangle_perimeter := 2 * (s + (s / 2))
  small_triangle_perimeter / large_rectangle_perimeter = 2 / 3 :=
by
  sorry

end ratio_of_perimeters_l600_60024


namespace trigonometric_expression_evaluation_l600_60043

theorem trigonometric_expression_evaluation :
  let tan30 := (Real.sqrt 3) / 3
  let sin60 := (Real.sqrt 3) / 2
  let cot60 := 1 / (Real.sqrt 3)
  let tan60 := Real.sqrt 3
  let cos45 := (Real.sqrt 2) / 2
  (3 * tan30) / (1 - sin60) + (cot60 + Real.cos (Real.pi * 70 / 180))^0 - tan60 / (cos45^4) = 7 :=
by
  -- This is where the proof would go
  sorry

end trigonometric_expression_evaluation_l600_60043


namespace simplify_expression_l600_60082

theorem simplify_expression :
  (1024 ^ (1/5) * 125 ^ (1/3)) = 20 :=
by
  have h1 : 1024 = 2 ^ 10 := by norm_num
  have h2 : 125 = 5 ^ 3 := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end simplify_expression_l600_60082


namespace fabric_ratio_wednesday_tuesday_l600_60056

theorem fabric_ratio_wednesday_tuesday :
  let fabric_monday := 20
  let fabric_tuesday := 2 * fabric_monday
  let cost_per_yard := 2
  let total_earnings := 140
  let earnings_monday := fabric_monday * cost_per_yard
  let earnings_tuesday := fabric_tuesday * cost_per_yard
  let earnings_wednesday := total_earnings - (earnings_monday + earnings_tuesday)
  let fabric_wednesday := earnings_wednesday / cost_per_yard
  (fabric_wednesday / fabric_tuesday = 1 / 4) :=
by
  sorry

end fabric_ratio_wednesday_tuesday_l600_60056


namespace katharina_order_is_correct_l600_60039

-- Define the mixed up order around a circle starting with L
def mixedUpOrder : List Char := ['L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']

-- Define the positions and process of Jaxon's list generation
def jaxonList : List Nat := [1, 4, 7, 3, 8, 5, 2, 6]

-- Define the resulting order from Jaxon's process
def resultingOrder (initialList : List Char) (positions : List Nat) : List Char :=
  positions.map (λ i => initialList.get! (i - 1))

-- Define the function to prove Katharina's order
theorem katharina_order_is_correct :
  resultingOrder mixedUpOrder jaxonList = ['L', 'R', 'O', 'M', 'S', 'Q', 'N', 'P'] :=
by
  -- Proof omitted
  sorry

end katharina_order_is_correct_l600_60039


namespace students_with_uncool_parents_but_cool_siblings_l600_60040

-- The total number of students in the classroom
def total_students : ℕ := 40

-- The number of students with cool dads
def students_with_cool_dads : ℕ := 18

-- The number of students with cool moms
def students_with_cool_moms : ℕ := 22

-- The number of students with both cool dads and cool moms
def students_with_both_cool_parents : ℕ := 10

-- The number of students with cool siblings
def students_with_cool_siblings : ℕ := 8

-- The theorem we want to prove
theorem students_with_uncool_parents_but_cool_siblings
  (h1 : total_students = 40)
  (h2 : students_with_cool_dads = 18)
  (h3 : students_with_cool_moms = 22)
  (h4 : students_with_both_cool_parents = 10)
  (h5 : students_with_cool_siblings = 8) :
  8 = (students_with_cool_siblings) :=
sorry

end students_with_uncool_parents_but_cool_siblings_l600_60040


namespace evaluate_fraction_l600_60083

theorem evaluate_fraction : ∃ p q : ℤ, gcd p q = 1 ∧ (2023 : ℤ) / (2022 : ℤ) - 2 * (2022 : ℤ) / (2023 : ℤ) = (p : ℚ) / (q : ℚ) ∧ p = -(2022^2 : ℤ) + 4045 :=
by
  sorry

end evaluate_fraction_l600_60083


namespace arc_length_of_pentagon_side_l600_60096

theorem arc_length_of_pentagon_side 
  (r : ℝ) (h : r = 4) :
  (2 * r * Real.pi * (72 / 360)) = (8 * Real.pi / 5) :=
by
  sorry

end arc_length_of_pentagon_side_l600_60096


namespace pencils_given_out_l600_60072
-- Define the problem conditions
def students : ℕ := 96
def dozens_per_student : ℕ := 7
def pencils_per_dozen : ℕ := 12

-- Define the expected total pencils
def expected_pencils : ℕ := 8064

-- Define the statement to be proven
theorem pencils_given_out : (students * (dozens_per_student * pencils_per_dozen)) = expected_pencils := 
  by
  sorry

end pencils_given_out_l600_60072


namespace sin_neg_pi_l600_60097

theorem sin_neg_pi : Real.sin (-Real.pi) = 0 := by
  sorry

end sin_neg_pi_l600_60097


namespace y_share_is_correct_l600_60093

noncomputable def share_of_y (a : ℝ) := 0.45 * a

theorem y_share_is_correct :
  ∃ a : ℝ, (1 * a + 0.45 * a + 0.30 * a = 245) ∧ (share_of_y a = 63) :=
by
  sorry

end y_share_is_correct_l600_60093


namespace first_tree_height_l600_60070

theorem first_tree_height
  (branches_first : ℕ)
  (branches_second : ℕ)
  (height_second : ℕ)
  (branches_third : ℕ)
  (height_third : ℕ)
  (branches_fourth : ℕ)
  (height_fourth : ℕ)
  (average_branches_per_foot : ℕ) :
  branches_first = 200 →
  height_second = 40 →
  branches_second = 180 →
  height_third = 60 →
  branches_third = 180 →
  height_fourth = 34 →
  branches_fourth = 153 →
  average_branches_per_foot = 4 →
  branches_first / average_branches_per_foot = 50 :=
by
  sorry

end first_tree_height_l600_60070


namespace infinitely_many_solutions_implies_b_eq_neg6_l600_60011

theorem infinitely_many_solutions_implies_b_eq_neg6 (b : ℤ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 8)) → b = -6 :=
  sorry

end infinitely_many_solutions_implies_b_eq_neg6_l600_60011


namespace sum_squares_of_roots_l600_60092

def a := 8
def b := 12
def c := -14

theorem sum_squares_of_roots : (b^2 - 2 * a * c)/(a^2) = 23/4 := by
  sorry

end sum_squares_of_roots_l600_60092


namespace sqrt_sum_gt_l600_60098

theorem sqrt_sum_gt (a b : ℝ) (ha : a = 2) (hb : b = 3) : 
  Real.sqrt a + Real.sqrt b > Real.sqrt (a + b) := by 
  sorry

end sqrt_sum_gt_l600_60098


namespace one_div_i_plus_i_pow_2015_eq_neg_two_i_l600_60055

def is_imaginary_unit (x : ℂ) : Prop := x * x = -1

theorem one_div_i_plus_i_pow_2015_eq_neg_two_i (i : ℂ) (h : is_imaginary_unit i) : 
  (1 / i + i ^ 2015) = -2 * i :=
sorry

end one_div_i_plus_i_pow_2015_eq_neg_two_i_l600_60055


namespace cone_curved_surface_area_l600_60001

def radius (r : ℝ) := r = 3
def slantHeight (l : ℝ) := l = 15
def curvedSurfaceArea (csa : ℝ) := csa = 45 * Real.pi

theorem cone_curved_surface_area 
  (r l csa : ℝ) 
  (hr : radius r) 
  (hl : slantHeight l) 
  : curvedSurfaceArea (Real.pi * r * l) 
  := by
  unfold radius at hr
  unfold slantHeight at hl
  unfold curvedSurfaceArea
  rw [hr, hl]
  norm_num
  sorry

end cone_curved_surface_area_l600_60001


namespace cricket_run_rate_l600_60080

theorem cricket_run_rate (run_rate_first_10_overs : ℝ) (total_target : ℝ) (overs_first_period : ℕ) (overs_remaining_period : ℕ)
  (h1 : run_rate_first_10_overs = 3.2)
  (h2 : total_target = 252)
  (h3 : overs_first_period = 10)
  (h4 : overs_remaining_period = 40) :
  (total_target - (run_rate_first_10_overs * overs_first_period)) / overs_remaining_period = 5.5 := 
by
  sorry

end cricket_run_rate_l600_60080


namespace num_valid_10_digit_sequences_l600_60045

theorem num_valid_10_digit_sequences : 
  ∃ (n : ℕ), n = 64 ∧ 
  (∀ (seq : Fin 10 → Fin 3), 
    (∀ i : Fin 9, abs (seq i.succ - seq i) = 1) → 
    (∀ i : Fin 10, seq i < 3) →
    ∃ k : Nat, k = 10 ∧ seq 0 < 10 ∧ seq 1 < 10 ∧ seq 2 < 10 ∧ seq 3 < 10 ∧ 
      seq 4 < 10 ∧ seq 5 < 10 ∧ seq 6 < 10 ∧ seq 7 < 10 ∧ 
      seq 8 < 10 ∧ seq 9 < 10 ∧ k = 10 → n = 64) :=
sorry

end num_valid_10_digit_sequences_l600_60045


namespace toby_total_time_l600_60036

def speed_unloaded := 20 -- Speed of Toby pulling unloaded sled in mph
def speed_loaded := 10   -- Speed of Toby pulling loaded sled in mph

def distance_part1 := 180 -- Distance for the first part (loaded sled) in miles
def distance_part2 := 120 -- Distance for the second part (unloaded sled) in miles
def distance_part3 := 80  -- Distance for the third part (loaded sled) in miles
def distance_part4 := 140 -- Distance for the fourth part (unloaded sled) in miles

def time_part1 := distance_part1 / speed_loaded -- Time for the first part in hours
def time_part2 := distance_part2 / speed_unloaded -- Time for the second part in hours
def time_part3 := distance_part3 / speed_loaded -- Time for the third part in hours
def time_part4 := distance_part4 / speed_unloaded -- Time for the fourth part in hours

def total_time := time_part1 + time_part2 + time_part3 + time_part4 -- Total time in hours

theorem toby_total_time : total_time = 39 :=
by 
  sorry

end toby_total_time_l600_60036


namespace dorchester_daily_pay_l600_60051

theorem dorchester_daily_pay (D : ℝ) (P : ℝ) (total_earnings : ℝ) (num_puppies : ℕ) (earn_per_puppy : ℝ) 
  (h1 : total_earnings = 76) (h2 : num_puppies = 16) (h3 : earn_per_puppy = 2.25) 
  (h4 : total_earnings = D + num_puppies * earn_per_puppy) : D = 40 :=
by
  sorry

end dorchester_daily_pay_l600_60051


namespace average_speed_is_70_kmh_l600_60057

-- Define the given conditions
def distance1 : ℕ := 90
def distance2 : ℕ := 50
def time1 : ℕ := 1
def time2 : ℕ := 1

-- We need to prove that the average speed of the car is 70 km/h
theorem average_speed_is_70_kmh :
    ((distance1 + distance2) / (time1 + time2)) = 70 := 
by 
    -- This is the proof placeholder
    sorry

end average_speed_is_70_kmh_l600_60057


namespace robinson_crusoe_sees_multiple_colors_l600_60029

def chameleons_multiple_colors (r b v : ℕ) : Prop :=
  let d1 := (r - b) % 3
  let d2 := (b - v) % 3
  let d3 := (r - v) % 3
  -- Given initial counts and rules.
  (r = 155) ∧ (b = 49) ∧ (v = 96) ∧
  -- Translate specific steps and conditions into properties
  (d1 = 1 % 3) ∧ (d2 = 1 % 3) ∧ (d3 = 2 % 3)

noncomputable def will_see_multiple_colors : Prop :=
  chameleons_multiple_colors 155 49 96 →
  ∃ (r b v : ℕ), r + b + v = 300 ∧
  ((r % 3 = 0 ∧ b % 3 ≠ 0 ∧ v % 3 ≠ 0) ∨
   (r % 3 ≠ 0 ∧ b % 3 = 0 ∧ v % 3 ≠ 0) ∨
   (r % 3 ≠ 0 ∧ b % 3 ≠ 0 ∧ v % 3 = 0))

theorem robinson_crusoe_sees_multiple_colors : will_see_multiple_colors :=
sorry

end robinson_crusoe_sees_multiple_colors_l600_60029


namespace problem_l600_60088

def polynomial (x : ℝ) : ℝ := 9 * x ^ 3 - 27 * x + 54

theorem problem (a b c : ℝ) 
  (h_roots : polynomial a = 0 ∧ polynomial b = 0 ∧ polynomial c = 0) :
  (a + b) ^ 3 + (b + c) ^ 3 + (c + a) ^ 3 = 18 :=
by
  sorry

end problem_l600_60088


namespace class_president_is_yi_l600_60075

variable (Students : Type)
variable (Jia Yi Bing StudyCommittee SportsCommittee ClassPresident : Students)
variable (age : Students → ℕ)

-- Conditions
axiom bing_older_than_study_committee : age Bing > age StudyCommittee
axiom jia_age_different_from_sports_committee : age Jia ≠ age SportsCommittee
axiom sports_committee_younger_than_yi : age SportsCommittee < age Yi

-- Prove that Yi is the class president
theorem class_president_is_yi : ClassPresident = Yi :=
sorry

end class_president_is_yi_l600_60075


namespace number_of_boys_in_school_l600_60074

variable (x : ℕ) (y : ℕ)

theorem number_of_boys_in_school 
    (h1 : 1200 = x + (1200 - x))
    (h2 : 200 = y + (y + 10))
    (h3 : 105 / 200 = (x : ℝ) / 1200) 
    : x = 630 := 
  by 
  sorry

end number_of_boys_in_school_l600_60074


namespace vasya_new_scoring_system_l600_60069

theorem vasya_new_scoring_system (a b c : ℕ) 
  (h1 : a + b + c = 52) 
  (h2 : a + b / 2 = 35) : a - c = 18 :=
by
  sorry

end vasya_new_scoring_system_l600_60069
