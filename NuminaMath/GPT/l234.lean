import Mathlib

namespace cubic_inches_in_two_cubic_feet_l234_234909

theorem cubic_inches_in_two_cubic_feet (conv : 1 = 12) : 2 * (12 * 12 * 12) = 3456 :=
by
  sorry

end cubic_inches_in_two_cubic_feet_l234_234909


namespace find_x_l234_234715

theorem find_x (x : ℝ) (h : ⌈x⌉ * x = 210) : x = 14 := by
  sorry

end find_x_l234_234715


namespace quadratic_no_real_solutions_l234_234882

theorem quadratic_no_real_solutions (k : ℝ) :
  k < -9 / 4 ↔ ∀ x : ℝ, ¬ (x^2 - 3 * x - k = 0) :=
by
  sorry

end quadratic_no_real_solutions_l234_234882


namespace Felix_can_lift_150_pounds_l234_234875

theorem Felix_can_lift_150_pounds : ∀ (weightFelix weightBrother : ℝ),
  (weightBrother = 2 * weightFelix) →
  (3 * weightBrother = 600) →
  (Felix_can_lift = 1.5 * weightFelix) →
  Felix_can_lift = 150 :=
by
  intros weightFelix weightBrother h1 h2 h3
  sorry

end Felix_can_lift_150_pounds_l234_234875


namespace inscribed_circle_radius_l234_234080

-- Conditions
variables {S A B C D O : Point} -- Points in 3D space
variables (AC : ℝ) (cos_SBD : ℝ)
variables (r : ℝ) -- Radius of inscribed circle

-- Given conditions
def AC_eq_one := AC = 1
def cos_angle_SBD := cos_SBD = 2/3

-- Assertion to be proved
theorem inscribed_circle_radius :
  AC_eq_one AC →
  cos_angle_SBD cos_SBD →
  (0 < r ∧ r ≤ 1/6) ∨ r = 1/3 :=
by
  intro hAC hcos
  -- Proof goes here
  sorry

end inscribed_circle_radius_l234_234080


namespace inequality_solution_ab_l234_234919

theorem inequality_solution_ab (a b : ℝ) (h : ∀ x : ℝ, 2 < x ∧ x < 4 ↔ |x + a| < b) : a * b = -3 := 
by
  sorry

end inequality_solution_ab_l234_234919


namespace units_digit_of_24_pow_4_add_42_pow_4_l234_234034

theorem units_digit_of_24_pow_4_add_42_pow_4 : 
  (24^4 + 42^4) % 10 = 2 := 
by sorry

end units_digit_of_24_pow_4_add_42_pow_4_l234_234034


namespace remaining_shirt_cost_l234_234727

theorem remaining_shirt_cost (total_shirts : ℕ) (cost_3_shirts : ℕ) (total_cost : ℕ) 
  (h1 : total_shirts = 5) 
  (h2 : cost_3_shirts = 3 * 15) 
  (h3 : total_cost = 85) :
  (total_cost - cost_3_shirts) / (total_shirts - 3) = 20 :=
by
  sorry

end remaining_shirt_cost_l234_234727


namespace square_diagonal_l234_234842

theorem square_diagonal (p : ℤ) (h : p = 28) : ∃ d : ℝ, d = 7 * Real.sqrt 2 :=
by
  sorry

end square_diagonal_l234_234842


namespace danielles_rooms_l234_234454

variable (rooms_heidi rooms_danielle : ℕ)

theorem danielles_rooms 
  (h1 : rooms_heidi = 3 * rooms_danielle)
  (h2 : 2 = 1 / 9 * rooms_heidi) :
  rooms_danielle = 6 := by
  -- Proof omitted
  sorry

end danielles_rooms_l234_234454


namespace lindsey_final_money_l234_234159

-- Define the savings in each month
def save_sep := 50
def save_oct := 37
def save_nov := 11

-- Total savings over the three months
def total_savings := save_sep + save_oct + save_nov

-- Condition for Mom's contribution
def mom_contribution := if total_savings > 75 then 25 else 0

-- Total savings including mom's contribution
def total_with_mom := total_savings + mom_contribution

-- Amount spent on the video game
def spent := 87

-- Final amount left
def final_amount := total_with_mom - spent

-- Proof statement
theorem lindsey_final_money : final_amount = 36 := by
  sorry

end lindsey_final_money_l234_234159


namespace cos_squared_alpha_plus_pi_over_4_correct_l234_234894

variable (α : ℝ)
axiom sin_two_alpha : Real.sin (2 * α) = 2 / 3

theorem cos_squared_alpha_plus_pi_over_4_correct :
  Real.cos (α + Real.pi / 4) ^ 2 = 1 / 6 :=
by
  sorry

end cos_squared_alpha_plus_pi_over_4_correct_l234_234894


namespace trolley_length_l234_234792

theorem trolley_length (L F : ℝ) (h1 : 4 * L + 3 * F = 108) (h2 : 10 * L + 9 * F = 168) : L = 78 := 
by
  sorry

end trolley_length_l234_234792


namespace discount_percentage_is_20_l234_234481

theorem discount_percentage_is_20
  (regular_price_per_shirt : ℝ) (number_of_shirts : ℝ) (total_sale_price : ℝ)
  (h₁ : regular_price_per_shirt = 50) (h₂ : number_of_shirts = 6) (h₃ : total_sale_price = 240) :
  ( ( (regular_price_per_shirt * number_of_shirts - total_sale_price) / (regular_price_per_shirt * number_of_shirts) ) * 100 ) = 20 :=
by
  sorry

end discount_percentage_is_20_l234_234481


namespace total_sides_tom_tim_l234_234965

def sides_per_die : Nat := 6

def tom_dice_count : Nat := 4
def tim_dice_count : Nat := 4

theorem total_sides_tom_tim : tom_dice_count * sides_per_die + tim_dice_count * sides_per_die = 48 := by
  sorry

end total_sides_tom_tim_l234_234965


namespace trigonometric_inequality_l234_234639

theorem trigonometric_inequality (x : Real) (h1 : 0 < x) (h2 : x < (3 * Real.pi) / 8) :
  (1 / Real.sin (x / 3) + 1 / Real.sin (8 * x / 3) > (Real.sin (3 * x / 2)) / (Real.sin (x / 2) * Real.sin (2 * x))) :=
  by
  sorry

end trigonometric_inequality_l234_234639


namespace scooter_travel_time_l234_234543

variable (x : ℝ)
variable (h_speed : x > 0)
variable (h_travel_time : (50 / (x - 1/2)) - (50 / x) = 3/4)

theorem scooter_travel_time : 50 / x = 50 / x := 
  sorry

end scooter_travel_time_l234_234543


namespace inequality_abc_l234_234489

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (a^2 + a * b + b^2)) + (b^3 / (b^2 + b * c + c^2)) + (c^3 / (c^2 + c * a + a^2)) >= (a + b + c) / 3 := by
  sorry

end inequality_abc_l234_234489


namespace intersection_points_l234_234970

noncomputable def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2
noncomputable def parabola2 (x : ℝ) : ℝ := -x^2 + 2 * x + 3

theorem intersection_points :
  {p : ℝ × ℝ |
    (∃ x : ℝ, p = (x, parabola1 x) ∧ parabola1 x = parabola2 x)} =
  { 
    ( (3 + Real.sqrt 13) / 4, (74 + 14 * Real.sqrt 13) / 16 ),
    ( (3 - Real.sqrt 13) / 4, (74 - 14 * Real.sqrt 13) / 16 )
  } := sorry

end intersection_points_l234_234970


namespace total_sides_is_48_l234_234967

-- Definitions based on the conditions
def num_dice_tom : Nat := 4
def num_dice_tim : Nat := 4
def sides_per_die : Nat := 6

-- The proof problem statement
theorem total_sides_is_48 : (num_dice_tom + num_dice_tim) * sides_per_die = 48 := by
  sorry

end total_sides_is_48_l234_234967


namespace barbara_spent_on_other_goods_l234_234389

theorem barbara_spent_on_other_goods
  (cost_tuna : ℝ := 5 * 2)
  (cost_water : ℝ := 4 * 1.5)
  (total_paid : ℝ := 56) :
  total_paid - (cost_tuna + cost_water) = 40 := by
  sorry

end barbara_spent_on_other_goods_l234_234389


namespace butter_remaining_correct_l234_234060

-- Definitions of the conditions
def cupsOfBakingMix : ℕ := 6
def butterPerCup : ℕ := 2
def substituteRatio : ℕ := 1
def coconutOilUsed : ℕ := 8

-- Calculation based on the conditions
def butterNeeded : ℕ := butterPerCup * cupsOfBakingMix
def butterReplaced : ℕ := coconutOilUsed * substituteRatio
def butterRemaining : ℕ := butterNeeded - butterReplaced

-- The theorem to prove the chef has 4 ounces of butter remaining
theorem butter_remaining_correct : butterRemaining = 4 := 
by
  -- Note: We insert 'sorry' since the proof itself is not required.
  sorry

end butter_remaining_correct_l234_234060


namespace neg_pi_lt_neg_314_l234_234583

theorem neg_pi_lt_neg_314 (h : Real.pi > 3.14) : -Real.pi < -3.14 :=
sorry

end neg_pi_lt_neg_314_l234_234583


namespace smallest_four_digit_number_divisible_by_9_with_conditions_l234_234810

def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_sum_of_digits_divisible_by_9 (n : ℕ) : Prop := 
  let digits := (List.ofDigits 10 (Nat.digits 10 n)).sum
  digits % 9 = 0

def has_three_even_digits_and_one_odd_digit (n : ℕ) : Prop := 
  let digits := Nat.digits 10 n
  (digits.filter (λ d => d % 2 = 0)).length = 3 ∧
  (digits.filter (λ d => d % 2 = 1)).length = 1

theorem smallest_four_digit_number_divisible_by_9_with_conditions : 
  ∃ n : ℕ, is_four_digit_number n ∧ 
            is_sum_of_digits_divisible_by_9 n ∧ 
            has_three_even_digits_and_one_odd_digit n ∧ 
            n = 2043 :=
sorry

end smallest_four_digit_number_divisible_by_9_with_conditions_l234_234810


namespace largest_integer_m_l234_234935

theorem largest_integer_m (m n : ℕ) (h1 : ∀ n ≤ m, (2 * n + 1) / (3 * n + 8) < (Real.sqrt 5 - 1) / 2) 
(h2 : ∀ n ≤ m, (Real.sqrt 5 - 1) / 2 < (n + 7) / (2 * n + 1)) : 
  m = 27 :=
sorry

end largest_integer_m_l234_234935


namespace candies_eaten_l234_234574

-- Definitions

def Andrey_rate_eq_Boris_rate (candies_eaten_by_Andrey candies_eaten_by_Boris : ℕ) : Prop :=
  candies_eaten_by_Andrey / 4 = candies_eaten_by_Boris / 3

def Denis_rate_eq_Andrey_rate (candies_eaten_by_Denis candies_eaten_by_Andrey : ℕ) : Prop :=
  candies_eaten_by_Denis / 7 = candies_eaten_by_Andrey / 6

def total_candies (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) : Prop :=
  candies_eaten_by_Andrey + candies_eaten_by_Boris + candies_eaten_by_Denis = 70

-- Theorem to prove the candies eaten by Andrey, Boris, and Denis
theorem candies_eaten (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) :
  Andrey_rate_eq_Boris_rate candies_eaten_by_Andrey candies_eaten_by_Boris →
  Denis_rate_eq_Andrey_rate candies_eaten_by_Denis candies_eaten_by_Andrey →
  total_candies candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis →
  candies_eaten_by_Andrey = 24 ∧ candies_eaten_by_Boris = 18 ∧ candies_eaten_by_Denis = 28 :=
  by sorry

end candies_eaten_l234_234574


namespace candy_eating_l234_234563

-- Definitions based on the conditions
def candies (andrey_rate boris_rate denis_rate : ℕ) (andrey_candies boris_candies denis_candies total_candies : ℕ) : Prop :=
  andrey_rate = 4 ∧ boris_rate = 3 ∧ denis_rate = 7 ∧ andrey_candies = 24 ∧ boris_candies = 18 ∧ denis_candies = 28 ∧
  total_candies = andrey_candies + boris_candies + denis_candies

-- Problem statement
theorem candy_eating :
  ∃ (a b d : ℕ), 
    candies 4 3 7 a b d 70 :=
sorry

end candy_eating_l234_234563


namespace pencils_added_by_mike_l234_234962

-- Definitions and assumptions based on conditions
def initial_pencils : ℕ := 41
def final_pencils : ℕ := 71

-- Statement of the problem
theorem pencils_added_by_mike : final_pencils - initial_pencils = 30 := 
by 
  sorry

end pencils_added_by_mike_l234_234962


namespace fred_earnings_l234_234298

-- Conditions as definitions
def initial_amount : ℕ := 23
def final_amount : ℕ := 86

-- Theorem to prove
theorem fred_earnings : final_amount - initial_amount = 63 := by
  sorry

end fred_earnings_l234_234298


namespace math_problem_l234_234516

theorem math_problem
  (x_1 y_1 x_2 y_2 x_3 y_3 : ℝ)
  (h1 : x_1^3 - 3 * x_1 * y_1^2 = 2006)
  (h2 : y_1^3 - 3 * x_1^2 * y_1 = 2007)
  (h3 : x_2^3 - 3 * x_2 * y_2^2 = 2006)
  (h4 : y_2^3 - 3 * x_2^2 * y_2 = 2007)
  (h5 : x_3^3 - 3 * x_3 * y_3^2 = 2006)
  (h6 : y_3^3 - 3 * x_3^2 * y_3 = 2007)
  : (1 - x_1 / y_1) * (1 - x_2 / y_2) * (1 - x_3 / y_3) = -1 / 2006 := by
  sorry

end math_problem_l234_234516


namespace vector_a_properties_l234_234602

-- Definitions of the points in space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of vector subtraction to find the vector between two points
def vector_sub (p1 p2 : Point3D) : Point3D :=
  { x := p2.x - p1.x, y := p2.y - p1.y, z := p2.z - p1.z }

-- Definition of dot product for vectors
def dot_product (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Definition of vector magnitude squared for vectors
def magnitude_squared (v : Point3D) : ℝ :=
  v.x * v.x + v.y * v.y + v.z * v.z

-- Main theorem statement
theorem vector_a_properties :
  let A := {x := 0, y := 2, z := 3}
  let B := {x := -2, y := 1, z := 6}
  let C := {x := 1, y := -1, z := 5}
  let AB := vector_sub A B
  let AC := vector_sub A C
  ∀ (a : Point3D), 
    (magnitude_squared a = 3) → 
    (dot_product a AB = 0) → 
    (dot_product a AC = 0) → 
    (a = {x := 1, y := 1, z := 1} ∨ a = {x := -1, y := -1, z := -1}) := 
by
  intros A B C AB AC a ha_magnitude ha_perpendicular_AB ha_perpendicular_AC
  sorry

end vector_a_properties_l234_234602


namespace speed_conversion_l234_234067

def speed_mps : ℝ := 10.0008
def conversion_factor : ℝ := 3.6

theorem speed_conversion : speed_mps * conversion_factor = 36.003 :=
by
  sorry

end speed_conversion_l234_234067


namespace work_together_days_l234_234981

theorem work_together_days
  (a_days : ℝ) (ha : a_days = 18)
  (b_days : ℝ) (hb : b_days = 30)
  (c_days : ℝ) (hc : c_days = 45)
  (combined_days : ℝ) :
  (combined_days = 1 / ((1 / a_days) + (1 / b_days) + (1 / c_days))) → combined_days = 9 := 
by
  sorry

end work_together_days_l234_234981


namespace sin_product_identity_l234_234704

theorem sin_product_identity :
  (Real.sin (12 * Real.pi / 180)) * (Real.sin (48 * Real.pi / 180)) * 
  (Real.sin (54 * Real.pi / 180)) * (Real.sin (72 * Real.pi / 180)) = 1 / 16 := 
by 
  sorry

end sin_product_identity_l234_234704


namespace bus_trip_distance_l234_234822

theorem bus_trip_distance 
  (T : ℝ)  -- Time in hours
  (D : ℝ)  -- Distance in miles
  (h : D = 30 * T)  -- condition 1: the trip with 30 mph
  (h' : D = 35 * (T - 1))  -- condition 2: the trip with 35 mph
  : D = 210 := 
by
  sorry

end bus_trip_distance_l234_234822


namespace total_cost_of_horse_and_saddle_l234_234986

noncomputable def saddle_cost : ℝ := 1000
noncomputable def horse_cost : ℝ := 4 * saddle_cost
noncomputable def total_cost : ℝ := saddle_cost + horse_cost

theorem total_cost_of_horse_and_saddle :
    total_cost = 5000 := by
  sorry

end total_cost_of_horse_and_saddle_l234_234986


namespace shift_left_by_pi_over_six_l234_234958

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin x + Real.cos x
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin x

theorem shift_left_by_pi_over_six : f = λ x => g (x + Real.pi / 6) := by
  sorry

end shift_left_by_pi_over_six_l234_234958


namespace total_fish_l234_234927

theorem total_fish (x y : ℕ) : (19 - 2 * x) + (27 - 4 * y) = 46 - 2 * x - 4 * y :=
  by
    sorry

end total_fish_l234_234927


namespace probability_x_gt_2y_is_1_over_3_l234_234694

noncomputable def probability_x_gt_2y_in_rectangle : ℝ :=
  let A_rect := 6 * 1
  let A_triangle := (1/2) * 4 * 1
  A_triangle / A_rect

theorem probability_x_gt_2y_is_1_over_3 :
  probability_x_gt_2y_in_rectangle = 1 / 3 :=
sorry

end probability_x_gt_2y_is_1_over_3_l234_234694


namespace john_has_leftover_correct_l234_234474

-- Define the initial conditions
def initial_gallons : ℚ := 5
def given_away : ℚ := 18 / 7

-- Define the target result after subtraction
def remaining_gallons : ℚ := 17 / 7

-- The theorem statement
theorem john_has_leftover_correct :
  initial_gallons - given_away = remaining_gallons :=
by
  sorry

end john_has_leftover_correct_l234_234474


namespace true_q_if_not_p_and_p_or_q_l234_234460

variables {p q : Prop}

theorem true_q_if_not_p_and_p_or_q (h1 : ¬p) (h2 : p ∨ q) : q :=
by 
  sorry

end true_q_if_not_p_and_p_or_q_l234_234460


namespace work_completion_time_l234_234376

/-
Conditions:
1. A man alone can do the work in 6 days.
2. A woman alone can do the work in 18 days.
3. A boy alone can do the work in 9 days.

Question:
How long will they take to complete the work together?

Correct Answer:
3 days
-/

theorem work_completion_time (M W B : ℕ) (hM : M = 6) (hW : W = 18) (hB : B = 9) : 1 / (1/M + 1/W + 1/B) = 3 := 
by
  sorry

end work_completion_time_l234_234376


namespace find_x_l234_234717

theorem find_x (x : ℝ) (h : ⌈x⌉ * x = 210) : x = 14 := by
  sorry

end find_x_l234_234717


namespace tangent_line_eq_extreme_values_interval_l234_234427

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^3 + b * x + 2

theorem tangent_line_eq (a b : ℝ) (h1 : 3 * a * 2^2 + b = 0) (h2 : a * 2^3 + b * 2 + 2 = -14) :
  9 * 1 + (f 1 a b) = 0 :=
sorry

theorem extreme_values_interval (a b : ℝ) (h1 : 3 * a * 2^2 + b = 0) (h2 : a * 2^3 + b * 2 + 2 = -14) :
  ∃ (min_val max_val : ℝ), 
    min_val = -14 ∧ f 2 a b = min_val ∧
    max_val = 18 ∧ f (-2) a b = max_val ∧
    ∀ x, (x ∈ Set.Icc (-3 : ℝ) 3 → f x a b ≥ min_val ∧ f x a b ≤ max_val) :=
sorry

end tangent_line_eq_extreme_values_interval_l234_234427


namespace leah_birds_duration_l234_234764

-- Define the conditions
def boxes_bought : ℕ := 3
def boxes_existing : ℕ := 5
def parrot_weekly_consumption : ℕ := 100
def cockatiel_weekly_consumption : ℕ := 50
def grams_per_box : ℕ := 225

-- Define the question as a theorem
theorem leah_birds_duration : 
  (boxes_bought + boxes_existing) * grams_per_box / 
  (parrot_weekly_consumption + cockatiel_weekly_consumption) = 12 :=
by
  -- Proof would go here
  sorry

end leah_birds_duration_l234_234764


namespace veranda_width_l234_234183

theorem veranda_width (l w : ℝ) (room_area veranda_area : ℝ) (h1 : l = 20) (h2 : w = 12) (h3 : veranda_area = 144) : 
  ∃ w_v : ℝ, (l + 2 * w_v) * (w + 2 * w_v) - l * w = veranda_area ∧ w_v = 2 := 
by
  sorry

end veranda_width_l234_234183


namespace Ali_money_left_l234_234851

theorem Ali_money_left (initial_money : ℕ) 
  (spent_on_food_ratio : ℚ) 
  (spent_on_glasses_ratio : ℚ) 
  (spent_on_food : ℕ) 
  (left_after_food : ℕ) 
  (spent_on_glasses : ℕ) 
  (final_left : ℕ) :
    initial_money = 480 →
    spent_on_food_ratio = 1 / 2 →
    spent_on_food = initial_money * spent_on_food_ratio →
    left_after_food = initial_money - spent_on_food →
    spent_on_glasses_ratio = 1 / 3 →
    spent_on_glasses = left_after_food * spent_on_glasses_ratio →
    final_left = left_after_food - spent_on_glasses →
    final_left = 160 :=
by
  sorry

end Ali_money_left_l234_234851


namespace f_2020_minus_f_2018_l234_234278

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 5) = f x
axiom f_seven : f 7 = 9

theorem f_2020_minus_f_2018 : f 2020 - f 2018 = 9 := by
  sorry

end f_2020_minus_f_2018_l234_234278


namespace store_total_profit_l234_234843

theorem store_total_profit
  (purchase_price : ℕ)
  (selling_price_total : ℕ)
  (max_selling_price : ℕ)
  (profit : ℕ)
  (N : ℕ)
  (selling_price_per_card : ℕ)
  (h1 : purchase_price = 21)
  (h2 : selling_price_total = 1457)
  (h3 : max_selling_price = 2 * purchase_price)
  (h4 : selling_price_per_card * N = selling_price_total)
  (h5 : selling_price_per_card ≤ max_selling_price)
  (h_profit : profit = (selling_price_per_card - purchase_price) * N)
  : profit = 470 :=
sorry

end store_total_profit_l234_234843


namespace minimum_tetrahedra_partition_l234_234031

-- Definitions for the problem conditions
def cube_faces : ℕ := 6
def tetrahedron_faces : ℕ := 4

def face_constraint (cube_faces : ℕ) (tetrahedral_faces : ℕ) : Prop :=
  cube_faces * 2 = 12

def volume_constraint (cube_volume : ℝ) (tetrahedron_volume : ℝ) : Prop :=
  tetrahedron_volume < cube_volume / 6

-- Main proof statement
theorem minimum_tetrahedra_partition (cube_faces tetrahedron_faces : ℕ) (cube_volume tetrahedron_volume : ℝ) :
  face_constraint cube_faces tetrahedron_faces →
  volume_constraint cube_volume tetrahedron_volume →
  5 ≤ cube_faces * 2 / 3 :=
  sorry

end minimum_tetrahedra_partition_l234_234031


namespace eval_nabla_l234_234867

namespace MathProblem

-- Definition of the operation
def nabla (a b : ℕ) : ℕ :=
  3 + b ^ a

-- Theorem statement
theorem eval_nabla : nabla (nabla 2 3) 4 = 16777219 :=
by
  sorry

end MathProblem

end eval_nabla_l234_234867


namespace surveyDSuitableForComprehensiveSurvey_l234_234531

inductive Survey where
| A : Survey
| B : Survey
| C : Survey
| D : Survey

def isComprehensiveSurvey (s : Survey) : Prop :=
  match s with
  | Survey.A => False
  | Survey.B => False
  | Survey.C => False
  | Survey.D => True

theorem surveyDSuitableForComprehensiveSurvey : isComprehensiveSurvey Survey.D :=
by
  sorry

end surveyDSuitableForComprehensiveSurvey_l234_234531


namespace tangent_function_property_l234_234897

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.tan (ϕ - x)

theorem tangent_function_property 
  (ϕ a : ℝ) 
  (h1 : π / 2 < ϕ) 
  (h2 : ϕ < 3 * π / 2) 
  (h3 : f 0 ϕ = 0) 
  (h4 : f (-a) ϕ = 1/2) : 
  f (a + π / 4) ϕ = -3 := by
  sorry

end tangent_function_property_l234_234897


namespace polynomial_sum_of_squares_l234_234477

theorem polynomial_sum_of_squares (P : Polynomial ℝ) (hP : ∀ x : ℝ, 0 ≤ P.eval x) :
  ∃ (Q R : Polynomial ℝ), P = Q^2 + R^2 :=
sorry

end polynomial_sum_of_squares_l234_234477


namespace equal_rental_costs_l234_234497

variable {x : ℝ}

def SunshineCarRentalsCost (x : ℝ) : ℝ := 17.99 + 0.18 * x
def CityRentalsCost (x : ℝ) : ℝ := 18.95 + 0.16 * x

theorem equal_rental_costs (x : ℝ) : SunshineCarRentalsCost x = CityRentalsCost x ↔ x = 48 :=
by
  sorry

end equal_rental_costs_l234_234497


namespace encyclopedia_pages_count_l234_234484

theorem encyclopedia_pages_count (digits_used : ℕ) (h : digits_used = 6869) : ∃ pages : ℕ, pages = 1994 :=
by 
  sorry

end encyclopedia_pages_count_l234_234484


namespace polynomial_divisibility_l234_234590

theorem polynomial_divisibility (r s : ℝ) :
  (∀ x, 10 * x^4 - 15 * x^3 - 55 * x^2 + 85 * x - 51 = 10 * (x - r)^2 * (x - s)) →
  r = 3 / 2 ∧ s = -5 / 2 :=
by
  intros h
  sorry

end polynomial_divisibility_l234_234590


namespace second_alloy_amount_l234_234467

theorem second_alloy_amount (x : ℝ) :
  let chromium_first_alloy := 0.12 * 15
  let chromium_second_alloy := 0.08 * x
  let total_weight := 15 + x
  let chromium_percentage_new_alloy := (0.12 * 15 + 0.08 * x) / (15 + x)
  chromium_percentage_new_alloy = (28 / 300) →
  x = 30 := sorry

end second_alloy_amount_l234_234467


namespace total_unique_items_l234_234553

-- Define the conditions
def shared_albums : ℕ := 12
def total_andrew_albums : ℕ := 23
def exclusive_andrew_memorabilia : ℕ := 5
def exclusive_john_albums : ℕ := 8

-- Define the number of unique items in Andrew's and John's collection 
def unique_andrew_albums : ℕ := total_andrew_albums - shared_albums
def unique_total_items : ℕ := unique_andrew_albums + exclusive_john_albums + exclusive_andrew_memorabilia

-- The proof goal
theorem total_unique_items : unique_total_items = 24 := by
  -- Proof steps would go here
  sorry

end total_unique_items_l234_234553


namespace cos_double_angle_l234_234104

theorem cos_double_angle (x : ℝ) (h : 2 * Real.sin (Real.pi - x) + 1 = 0) : Real.cos (2 * x) = 1 / 2 := 
sorry

end cos_double_angle_l234_234104


namespace units_digit_24_pow_4_plus_42_pow_4_l234_234037

theorem units_digit_24_pow_4_plus_42_pow_4 : 
    (24^4 + 42^4) % 10 = 2 :=
by
  sorry

end units_digit_24_pow_4_plus_42_pow_4_l234_234037


namespace shaded_area_eight_l234_234145

-- Definitions based on given conditions
def arcAQB (r : ℝ) : Prop := r = 2
def arcBRC (r : ℝ) : Prop := r = 2
def midpointQ (r : ℝ) : Prop := arcAQB r
def midpointR (r : ℝ) : Prop := arcBRC r
def midpointS (r : ℝ) : Prop := arcAQB r ∧ arcBRC r ∧ (arcAQB r ∨ arcBRC r)
def arcQRS (r : ℝ) : Prop := r = 2 ∧ midpointS r

-- The theorem to prove
theorem shaded_area_eight (r : ℝ) : arcAQB r ∧ arcBRC r ∧ arcQRS r → area_shaded_region = 8 := by
  sorry

end shaded_area_eight_l234_234145


namespace roots_of_quadratic_eq_l234_234593

theorem roots_of_quadratic_eq:
  (8 * γ^3 + 15 * δ^2 = 179) ↔ (γ^2 - 3 * γ + 1 = 0 ∧ δ^2 - 3 * δ + 1 = 0) :=
sorry

end roots_of_quadratic_eq_l234_234593


namespace solution_intervals_l234_234400

noncomputable def cubic_inequality (x : ℝ) : Prop :=
  x^3 - 3 * x^2 - 4 * x - 12 ≤ 0

noncomputable def linear_inequality (x : ℝ) : Prop :=
  2 * x + 6 > 0

theorem solution_intervals :
  { x : ℝ | cubic_inequality x ∧ linear_inequality x } = { x | -2 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end solution_intervals_l234_234400


namespace even_number_of_odd_degrees_l234_234381

-- Define the village as a set of vertices
def vertex_set : FinSet := {v : Fin 101}

-- Define the friendship graph as undirected edges between vertices
def friendship_graph : SimpleGraph vertex_set := {
  adj := λ (a b : vertex_set), (--- Insert conditions for a and b being friends ---)
  symm := λ (a b : vertex_set) (hab : adj a b), adj b a, -- Friendship is mutual
  loopless := λ (a : vertex_set), ¬adj a a -- No self-loops
}

-- Define the degree function
def degree (v : vertex_set) : Nat := (friendship_graph.degree v)

-- Statement to be proven:
theorem even_number_of_odd_degrees (V : vertex_set) (E : Nat) :
  2 * E = (V.toFin∑ λ v, degree v) →
  ∃ S : Finset V, (∀ v ∈ S, degree v % 2 = 1) ∧ S.card % 2 = 0 :=
sorry

end even_number_of_odd_degrees_l234_234381


namespace solution_set_inequality_l234_234409

theorem solution_set_inequality (a : ℝ) :
  ∀ x : ℝ,
    (12 * x^2 - a * x > a^2) →
    ((a > 0 ∧ (x < -a / 4 ∨ x > a / 3)) ∨
     (a = 0 ∧ x ≠ 0) ∨
     (a < 0 ∧ (x > -a / 4 ∨ x < a / 3))) :=
by
  sorry

end solution_set_inequality_l234_234409


namespace evaluate_expression_l234_234108

theorem evaluate_expression (x y z : ℝ) (hx : x ≠ 3) (hy : y ≠ 5) (hz : z ≠ 7) :
  (x - 3) / (7 - z) * (y - 5) / (3 - x) * (z - 7) / (5 - y) = -1 :=
by 
  sorry

end evaluate_expression_l234_234108


namespace a1_lt_a3_iff_an_lt_an1_l234_234891

-- Define arithmetic sequence and required properties
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variables (a : ℕ → ℝ)

-- Define the necessary and sufficient condition theorem
theorem a1_lt_a3_iff_an_lt_an1 (h_arith : is_arithmetic_sequence a) :
  (a 1 < a 3) ↔ (∀ n : ℕ, a n < a (n + 1)) :=
sorry

end a1_lt_a3_iff_an_lt_an1_l234_234891


namespace distance_is_correct_l234_234534

noncomputable def distance_from_home_to_forest_park : ℝ := 11  -- distance in kilometers

structure ProblemData where
  v : ℝ                  -- Xiao Wu's bicycling speed (in meters per minute)
  t_catch_up : ℝ          -- time it takes for father to catch up (in minutes)
  d_forest : ℝ            -- distance from catch-up point to forest park (in kilometers)
  t_remaining : ℝ        -- time remaining for Wu to reach park after wallet delivered (in minutes)
  bike_speed_factor : ℝ   -- speed factor of father's car compared to Wu's bike
  
open ProblemData

def problem_conditions : ProblemData :=
  { v := 350,
    t_catch_up := 7.5,
    d_forest := 3.5,
    t_remaining := 10,
    bike_speed_factor := 5 }

theorem distance_is_correct (data : ProblemData) :
  data.v = 350 →
  data.t_catch_up = 7.5 →
  data.d_forest = 3.5 →
  data.t_remaining = 10 →
  data.bike_speed_factor = 5 →
  distance_from_home_to_forest_park = 11 := 
by
  intros
  sorry

end distance_is_correct_l234_234534


namespace complex_eq_l234_234611

theorem complex_eq : ∀ (z : ℂ), (i * z = i + z) → (z = (1 - i) / 2) :=
by
  intros z h
  sorry

end complex_eq_l234_234611


namespace find_number_l234_234407

theorem find_number (x : ℝ) (h : x - (3/5) * x = 56) : x = 140 :=
sorry

end find_number_l234_234407


namespace painting_time_l234_234178

theorem painting_time (rate_taylor rate_jennifer rate_alex : ℚ) 
  (h_taylor : rate_taylor = 1 / 12) 
  (h_jennifer : rate_jennifer = 1 / 10) 
  (h_alex : rate_alex = 1 / 15) : 
  ∃ t : ℚ, t = 4 ∧ (1 / t) = rate_taylor + rate_jennifer + rate_alex :=
by
  sorry

end painting_time_l234_234178


namespace angle_y_value_l234_234870

theorem angle_y_value (ABC ABD ABE BAE y : ℝ) (h1 : ABC = 180) (h2 : ABD = 66) 
  (h3 : ABE = 114) (h4 : BAE = 31) (h5 : 31 + 114 + y = 180) : y = 35 :=
  sorry

end angle_y_value_l234_234870


namespace angle_C_ne_5pi_over_6_l234_234472

-- Define the triangle ∆ABC
variables (A B C : ℝ)

-- Assume the conditions provided
axiom condition_1 : 3 * Real.sin A + 4 * Real.cos B = 6
axiom condition_2 : 3 * Real.cos A + 4 * Real.sin B = 1

-- State that the size of angle C cannot be 5π/6
theorem angle_C_ne_5pi_over_6 : C ≠ 5 * Real.pi / 6 :=
sorry

end angle_C_ne_5pi_over_6_l234_234472


namespace part1_part2_l234_234466

variables {A B C : ℝ} {a b c : ℝ} -- Angles and sides of the triangle
variable (h1 : (a - b + c) * (a - b - c) + a * b = 0)
variable (h2 : b * c * Real.sin C = 3 * c * Real.cos A + 3 * a * Real.cos C)

theorem part1 : c = 2 * Real.sqrt 3 :=
by
  sorry

theorem part2 : 6 < a + b ∧ a + b <= 4 * Real.sqrt 3 :=
by
  sorry

end part1_part2_l234_234466


namespace gerald_added_crayons_l234_234797

namespace Proof

variable (original_crayons : ℕ) (total_crayons : ℕ)

theorem gerald_added_crayons (h1 : original_crayons = 7) (h2 : total_crayons = 13) : 
  total_crayons - original_crayons = 6 := by
  sorry

end Proof

end gerald_added_crayons_l234_234797


namespace point_on_circle_l234_234869

theorem point_on_circle 
    (P : ℝ × ℝ) 
    (h_l1 : 2 * P.1 - 3 * P.2 + 4 = 0)
    (h_l2 : 3 * P.1 - 2 * P.2 + 1 = 0) 
    (h_circle : (P.1 - 2) ^ 2 + (P.2 - 4) ^ 2 = 5) : 
    (P.1 - 2) ^ 2 + (P.2 - 4) ^ 2 = 5 :=
by
  sorry

end point_on_circle_l234_234869


namespace solve_for_2023_minus_a_minus_2b_l234_234273

theorem solve_for_2023_minus_a_minus_2b (a b : ℝ) (h : 1^2 + a*1 + 2*b = 0) : 2023 - a - 2*b = 2024 := 
by sorry

end solve_for_2023_minus_a_minus_2b_l234_234273


namespace no_such_a_exists_l234_234155

def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2*a^2 - a + 7}
def B (a : ℝ) : Set ℝ := {1, 5*a - 5, -1/2*a^2 + 3/2*a + 4, a^3 + a^2 + 3*a + 7}

theorem no_such_a_exists (a : ℝ) : ¬(A a ∩ B a = {2, 5}) :=
by
  sorry

end no_such_a_exists_l234_234155


namespace quadratic_points_relationship_l234_234280

theorem quadratic_points_relationship (c y1 y2 y3 : ℝ) 
  (hA : y1 = (-3)^2 + 2*(-3) + c)
  (hB : y2 = (1/2)^2 + 2*(1/2) + c)
  (hC : y3 = 2^2 + 2*2 + c) : y2 < y1 ∧ y1 < y3 := 
sorry

end quadratic_points_relationship_l234_234280


namespace range_of_a_l234_234284

theorem range_of_a (x y a : ℝ) (h1 : 3 * x + y = a + 1) (h2 : x + 3 * y = 3) (h3 : x + y > 5) : a > 16 := 
sorry 

end range_of_a_l234_234284


namespace solve_for_m_l234_234259

theorem solve_for_m (m : ℝ) (x1 x2 : ℝ)
    (h1 : x1^2 - (2 * m - 1) * x1 + m^2 = 0)
    (h2 : x2^2 - (2 * m - 1) * x2 + m^2 = 0)
    (h3 : (x1 + 1) * (x2 + 1) = 3)
    (h_reality : (2 * m - 1)^2 - 4 * m^2 ≥ 0) :
    m = -3 := by
  sorry

end solve_for_m_l234_234259


namespace inequality_reversal_l234_234268

theorem inequality_reversal (a b : ℝ) (h : a > b) : -2 * a < -2 * b :=
by
  sorry

end inequality_reversal_l234_234268


namespace candy_eating_l234_234561

-- Definitions based on the conditions
def candies (andrey_rate boris_rate denis_rate : ℕ) (andrey_candies boris_candies denis_candies total_candies : ℕ) : Prop :=
  andrey_rate = 4 ∧ boris_rate = 3 ∧ denis_rate = 7 ∧ andrey_candies = 24 ∧ boris_candies = 18 ∧ denis_candies = 28 ∧
  total_candies = andrey_candies + boris_candies + denis_candies

-- Problem statement
theorem candy_eating :
  ∃ (a b d : ℕ), 
    candies 4 3 7 a b d 70 :=
sorry

end candy_eating_l234_234561


namespace tangent_line_equations_l234_234059

theorem tangent_line_equations :
  ∃ l : ℝ → ℝ,
  (∀ x, l x = 0 ∨ l x = -3/2 * x + 9/125 ∨ l x = 5 * x - 3) ∧
  l (3/5) = 0 ∧
  (∃ t : ℝ, (l t) = t^2 * (t + 1) ∧ ∀ x, x ≠ t → l x ≠ (x^2 * (x + 1)))
  :=
sorry

end tangent_line_equations_l234_234059


namespace pens_purchased_is_30_l234_234545

def num_pens_purchased (cost_total: ℕ) 
                       (num_pencils: ℕ) 
                       (price_per_pencil: ℚ) 
                       (price_per_pen: ℚ)
                       (expected_pens: ℕ): Prop :=
   let cost_pencils := num_pencils * price_per_pencil
   let cost_pens := cost_total - cost_pencils
   let num_pens := cost_pens / price_per_pen
   num_pens = expected_pens

theorem pens_purchased_is_30 : num_pens_purchased 630 75 2.00 16 30 :=
by
  -- Unfold the definition manually if needed
  sorry

end pens_purchased_is_30_l234_234545


namespace greatest_divisor_420_smaller_than_50_and_factor_of_90_l234_234522

theorem greatest_divisor_420_smaller_than_50_and_factor_of_90 : 
  ∃ d, d ∣ 420 ∧ d ∣ 90 ∧ d < 50 ∧ ∀ k, k ∣ 420 ∧ k ∣ 90 ∧ k < 50 → k ≤ d := 
begin
  use 30,
  split,
  { exact dvd_refl 420 },     -- 30 is a divisor of 420
  split,
  { exact dvd_refl 90 },      -- 30 is a divisor of 90
  split,
  { linarith },               -- 30 < 50
  intros k hk,
  cases hk with hk1 hk_rest,
  cases hk_rest with hk2 hk_lt,
  sorry                     -- missing the internal proof steps
end

end greatest_divisor_420_smaller_than_50_and_factor_of_90_l234_234522


namespace part1_part2_l234_234117

variable (x α β : ℝ)

noncomputable def f (x : ℝ) : ℝ := 
  2 * Real.sqrt 3 * (Real.cos x) ^ 2 + 2 * (Real.sin x) * (Real.cos x) - Real.sqrt 3

theorem part1 (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) : 
  -Real.sqrt 3 ≤ f x ∧ f x ≤ 2 := 
sorry

theorem part2 (hα : 0 < α ∧ α < Real.pi / 2) (hβ : 0 < β ∧ β < Real.pi / 2) 
(h1 : f (α / 2 - Real.pi / 6) = 8 / 5) 
(h2 : Real.cos (α + β) = -12 / 13) : 
  Real.sin β = 63 / 65 := 
sorry

end part1_part2_l234_234117


namespace sandy_correct_sums_l234_234491

variable (c i : ℕ)

theorem sandy_correct_sums (h1 : c + i = 30) (h2 : 3 * c - 2 * i = 55) : c = 23 :=
by {
  -- Proof goes here
  sorry
}

end sandy_correct_sums_l234_234491


namespace sum_exterior_angles_const_l234_234385

theorem sum_exterior_angles_const (n : ℕ) (h : n ≥ 3) : 
  ∃ s : ℝ, s = 360 :=
by
  sorry

end sum_exterior_angles_const_l234_234385


namespace equal_rental_costs_l234_234496

variable {x : ℝ}

def SunshineCarRentalsCost (x : ℝ) : ℝ := 17.99 + 0.18 * x
def CityRentalsCost (x : ℝ) : ℝ := 18.95 + 0.16 * x

theorem equal_rental_costs (x : ℝ) : SunshineCarRentalsCost x = CityRentalsCost x ↔ x = 48 :=
by
  sorry

end equal_rental_costs_l234_234496


namespace counterexample_to_conjecture_l234_234708

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, 1 < m ∧ m < n → ¬(m ∣ n)

def is_power_of_two (k : ℕ) : Prop := ∃ m : ℕ, m > 0 ∧ k = 2 ^ m

theorem counterexample_to_conjecture :
  ∃ n : ℤ, n > 5 ∧ ¬ (3 ∣ n) ∧ ¬ (∃ p k : ℕ, is_prime p ∧ is_power_of_two k ∧ n = p + k) :=
sorry

end counterexample_to_conjecture_l234_234708


namespace alex_cakes_l234_234849

theorem alex_cakes :
  let slices_first_cake := 8
  let slices_second_cake := 12
  let given_away_friends_first := slices_first_cake / 4
  let remaining_after_friends_first := slices_first_cake - given_away_friends_first
  let given_away_family_first := remaining_after_friends_first / 2
  let remaining_after_family_first := remaining_after_friends_first - given_away_family_first
  let stored_in_freezer_first := remaining_after_family_first / 4
  let remaining_after_freezer_first := remaining_after_family_first - stored_in_freezer_first
  let remaining_after_eating_first := remaining_after_freezer_first - 2
  
  let given_away_friends_second := slices_second_cake / 3
  let remaining_after_friends_second := slices_second_cake - given_away_friends_second
  let given_away_family_second := remaining_after_friends_second / 6
  let remaining_after_family_second := remaining_after_friends_second - given_away_family_second
  let stored_in_freezer_second := remaining_after_family_second / 4
  let remaining_after_freezer_second := remaining_after_family_second - stored_in_freezer_second
  let remaining_after_eating_second := remaining_after_freezer_second - 1

  remaining_after_eating_first + stored_in_freezer_first + remaining_after_eating_second + stored_in_freezer_second = 7 :=
by
  -- Proof goes here
  sorry

end alex_cakes_l234_234849


namespace solve_real_equation_l234_234878

theorem solve_real_equation (x : ℝ) (h : (x + 2)^4 + x^4 = 82) : x = 1 ∨ x = -3 :=
  sorry

end solve_real_equation_l234_234878


namespace division_of_fractions_l234_234456

theorem division_of_fractions : (1 / 6) / (1 / 3) = 1 / 2 :=
by
  sorry

end division_of_fractions_l234_234456


namespace num_diamonds_in_G6_l234_234234

noncomputable def triangular_number (k : ℕ) : ℕ :=
  (k * (k + 1)) / 2

noncomputable def total_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 1 + 4 * (Finset.sum (Finset.range (n - 1)) (λ k => triangular_number (k + 1)))

theorem num_diamonds_in_G6 :
  total_diamonds 6 = 141 := by
  -- This will be proven
  sorry

end num_diamonds_in_G6_l234_234234


namespace ratio_of_oranges_to_limes_l234_234070

-- Constants and Definitions
def initial_fruits : ℕ := 150
def half_fruits : ℕ := 75
def oranges : ℕ := 50
def limes : ℕ := half_fruits - oranges
def ratio_oranges_limes : ℕ × ℕ := (oranges / Nat.gcd oranges limes, limes / Nat.gcd oranges limes)

-- Theorem Statement
theorem ratio_of_oranges_to_limes : ratio_oranges_limes = (2, 1) := by
  sorry

end ratio_of_oranges_to_limes_l234_234070


namespace derivative_at_1_l234_234311

def f (x : ℝ) : ℝ := (1 - 2 * x^3) ^ 10

theorem derivative_at_1 : deriv f 1 = 60 :=
by
  sorry

end derivative_at_1_l234_234311


namespace sodium_chloride_formed_l234_234725

section 

-- Definitions based on the conditions
def hydrochloric_acid_moles : ℕ := 2
def sodium_bicarbonate_moles : ℕ := 2

-- Balanced chemical equation represented as a function (1:1 reaction ratio)
def reaction (hcl_moles naHCO3_moles : ℕ) : ℕ := min hcl_moles naHCO3_moles

-- Theorem stating the reaction outcome
theorem sodium_chloride_formed : reaction hydrochloric_acid_moles sodium_bicarbonate_moles = 2 :=
by
  -- Proof is omitted
  sorry

end

end sodium_chloride_formed_l234_234725


namespace dhoni_remaining_earnings_l234_234087

theorem dhoni_remaining_earnings (rent_percent dishwasher_percent : ℝ) 
  (h1 : rent_percent = 20) (h2 : dishwasher_percent = 15) : 
  100 - (rent_percent + dishwasher_percent) = 65 := 
by 
  sorry

end dhoni_remaining_earnings_l234_234087


namespace employees_excluding_manager_l234_234788

theorem employees_excluding_manager (average_salary average_increase manager_salary n : ℕ)
  (h_avg_salary : average_salary = 2400)
  (h_avg_increase : average_increase = 100)
  (h_manager_salary : manager_salary = 4900)
  (h_new_avg_salary : average_salary + average_increase = 2500)
  (h_total_salary : (n + 1) * (average_salary + average_increase) = n * average_salary + manager_salary) :
  n = 24 :=
by
  sorry

end employees_excluding_manager_l234_234788


namespace sam_correct_percent_l234_234464

variable (y : ℝ)
variable (h_pos : 0 < y)

theorem sam_correct_percent :
  ((8 * y - 3 * y) / (8 * y) * 100) = 62.5 := by
sorry

end sam_correct_percent_l234_234464


namespace max_mark_cells_l234_234841

theorem max_mark_cells (n : Nat) (grid : Fin n → Fin n → Bool) :
  (∀ i : Fin n, ∃ j : Fin n, grid i j = true) ∧ 
  (∀ j : Fin n, ∃ i : Fin n, grid i j = true) ∧ 
  (∀ (x1 x2 y1 y2 : Fin n), (x1 ≤ x2 ∧ y1 ≤ y2 ∧ (x2.1 - x1.1 + 1) * (y2.1 - y1.1 + 1) ≥ n) → 
   ∃ i : Fin n, ∃ j : Fin n, grid i j = true ∧ x1 ≤ i ∧ i ≤ x2 ∧ y1 ≤ j ∧ j ≤ y2) → 
  (n ≤ 7) := sorry

end max_mark_cells_l234_234841


namespace tiles_needed_l234_234149

theorem tiles_needed (tile_area : ℝ) (kitchen_width kitchen_height tile_size: ℝ)
  (h1 : tile_size^2 = tile_area)
  (h2 : kitchen_width = 48)
  (h3 : kitchen_height = 72)
  (h4 : tile_size = 6) : kitchen_width / tile_size * kitchen_height / tile_size = 96 :=
by
  have width_tiles := kitchen_width / tile_size
  have height_tiles := kitchen_height / tile_size
  calc
    width_tiles * height_tiles = (kitchen_width / tile_size) * (kitchen_height / tile_size) : by rw [width_tiles, height_tiles]
                        ... = 48 / 6 * 72 / 6                       : by rw [←h2, ←h3, ←h4]
                        ... = 8 * 12                                 : by simp
                        ... = 96                                     : by norm_num

end tiles_needed_l234_234149


namespace sum_of_first_15_terms_l234_234137

-- Given conditions: Sum of 4th and 12th term is 24
variable (a d : ℤ) (a_4 a_12 : ℤ)
variable (S : ℕ → ℤ)
variable (arithmetic_series_4_12_sum : 2 * a + 14 * d = 24)
variable (nth_term_def : ∀ n, a + (n - 1) * d = a_n)

-- Question: Sum of the first 15 terms of the progression
theorem sum_of_first_15_terms : S 15 = 180 := by
  sorry

end sum_of_first_15_terms_l234_234137


namespace function_passes_through_vertex_l234_234659

theorem function_passes_through_vertex (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : a^(2 - 2) + 1 = 2 :=
by
  sorry

end function_passes_through_vertex_l234_234659


namespace regular_polygon_sides_l234_234833

theorem regular_polygon_sides (perimeter side_length : ℕ) (h_perim : perimeter = 180) (h_side : side_length = 15) : 
  let n := perimeter / side_length in n = 12 :=
by
  have h_n : n = perimeter / side_length := rfl
  rw [h_perim, h_side] at h_n
  have h_res : 180 / 15 = 12 := rfl
  rw h_res at h_n
  exact h_n

end regular_polygon_sides_l234_234833


namespace find_a_l234_234744

-- Defining the curve y in terms of x and a
def curve (x : ℝ) (a : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Defining the derivative of the curve
def derivative (x : ℝ) (a : ℝ) : ℝ := 4*x^3 + 2*a*x

-- The proof statement asserting the value of a
theorem find_a (a : ℝ) (h1 : derivative (-1) a = 8): a = -6 :=
by
  -- we assume here the necessary calculations and logical steps to prove the theorem
  sorry

end find_a_l234_234744


namespace find_wall_width_l234_234208

-- Define the dimensions of the brick in meters
def brick_length : ℝ := 0.20
def brick_width : ℝ := 0.1325
def brick_height : ℝ := 0.08

-- Define the dimensions of the wall in meters
def wall_length : ℝ := 7
def wall_height : ℝ := 15.5
def number_of_bricks : ℝ := 4094.3396226415093

-- Volume of one brick
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- Total volume of bricks used
def total_brick_volume : ℝ := number_of_bricks * brick_volume

-- Wall volume in terms of width W
def wall_volume (W : ℝ) : ℝ := wall_length * W * wall_height

-- The theorem we want to prove
theorem find_wall_width (W : ℝ) (h : wall_volume W = total_brick_volume) : W = 0.08 := by
  sorry

end find_wall_width_l234_234208


namespace original_number_of_matchsticks_l234_234001

-- Define the conditions
def matchsticks_per_house : ℕ := 10
def houses_created : ℕ := 30
def total_matchsticks_used := houses_created * matchsticks_per_house

-- Define the question and the proof goal
theorem original_number_of_matchsticks (h : total_matchsticks_used = (Michael's_original_matchsticks / 2)) :
  (Michael's_original_matchsticks = 600) :=
by
  sorry

end original_number_of_matchsticks_l234_234001


namespace last_fish_in_swamp_l234_234316

noncomputable def final_fish (perches pikes sudaks : ℕ) : String :=
  let p := perches
  let pi := pikes
  let s := sudaks
  if p = 6 ∧ pi = 7 ∧ s = 8 then "Sudak" else "Unknown"

theorem last_fish_in_swamp : final_fish 6 7 8 = "Sudak" := by
  sorry

end last_fish_in_swamp_l234_234316


namespace consecutive_numbers_average_l234_234192

theorem consecutive_numbers_average (a b c d e f g : ℕ)
  (h1 : (a + b + c + d + e + f + g) / 7 = 9)
  (h2 : 2 * a = g) : 
  7 = 7 :=
by sorry

end consecutive_numbers_average_l234_234192


namespace smallest_odd_n_3_product_gt_5000_l234_234357

theorem smallest_odd_n_3_product_gt_5000 :
  ∃ n : ℕ, (∃ k : ℤ, n = 2 * k + 1 ∧ n > 0) ∧ (3 ^ ((n + 1)^2 / 8)) > 5000 ∧ n = 8 :=
by
  sorry

end smallest_odd_n_3_product_gt_5000_l234_234357


namespace distance_per_interval_l234_234604

-- Definitions for the conditions
def total_distance : ℕ := 3  -- miles
def total_time : ℕ := 45  -- minutes
def interval_time : ℕ := 15  -- minutes per interval

-- Mathematical problem statement
theorem distance_per_interval :
  (total_distance / (total_time / interval_time) = 1) :=
by 
  sorry

end distance_per_interval_l234_234604


namespace line_intersects_circle_l234_234344

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, (x^2 + y^2 - 2*y = 0) ∧ (y - 1 = k * (x - 1)) :=
sorry

end line_intersects_circle_l234_234344


namespace regular_polygon_sides_l234_234832

theorem regular_polygon_sides (perimeter side_length : ℕ) (h_perim : perimeter = 180) (h_side : side_length = 15) : 
  let n := perimeter / side_length in n = 12 :=
by
  have h_n : n = perimeter / side_length := rfl
  rw [h_perim, h_side] at h_n
  have h_res : 180 / 15 = 12 := rfl
  rw h_res at h_n
  exact h_n

end regular_polygon_sides_l234_234832


namespace total_revenue_is_correct_l234_234515

def category_a_price : ℝ := 65
def category_b_price : ℝ := 45
def category_c_price : ℝ := 25

def category_a_discounted_price : ℝ := category_a_price - 0.55 * category_a_price
def category_b_discounted_price : ℝ := category_b_price - 0.35 * category_b_price
def category_c_discounted_price : ℝ := category_c_price - 0.20 * category_c_price

def category_a_full_price_quantity : ℕ := 100
def category_b_full_price_quantity : ℕ := 50
def category_c_full_price_quantity : ℕ := 60

def category_a_discounted_quantity : ℕ := 20
def category_b_discounted_quantity : ℕ := 30
def category_c_discounted_quantity : ℕ := 40

def revenue_from_category_a : ℝ :=
  category_a_discounted_quantity * category_a_discounted_price +
  category_a_full_price_quantity * category_a_price

def revenue_from_category_b : ℝ :=
  category_b_discounted_quantity * category_b_discounted_price +
  category_b_full_price_quantity * category_b_price

def revenue_from_category_c : ℝ :=
  category_c_discounted_quantity * category_c_discounted_price +
  category_c_full_price_quantity * category_c_price

def total_revenue : ℝ :=
  revenue_from_category_a + revenue_from_category_b + revenue_from_category_c

theorem total_revenue_is_correct :
  total_revenue = 12512.50 :=
by
  unfold total_revenue
  unfold revenue_from_category_a
  unfold revenue_from_category_b
  unfold revenue_from_category_c
  unfold category_a_discounted_price
  unfold category_b_discounted_price
  unfold category_c_discounted_price
  sorry

end total_revenue_is_correct_l234_234515


namespace horner_evaluation_of_f_at_5_l234_234413

def f (x : ℝ) : ℝ := x^5 - 2*x^4 + x^3 + x^2 - x - 5

theorem horner_evaluation_of_f_at_5 : f 5 = 2015 :=
by sorry

end horner_evaluation_of_f_at_5_l234_234413


namespace solve_quadratic_and_linear_equations_l234_234175

theorem solve_quadratic_and_linear_equations :
  (∀ x : ℝ, x^2 - 4*x - 1 = 0 → x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) ∧
  (∀ x : ℝ, (x + 3) * (x - 3) = 3 * (x + 3) → x = -3 ∨ x = 6) :=
by
  sorry

end solve_quadratic_and_linear_equations_l234_234175


namespace line_MN_eq_l234_234900

-- Definitions for the function f and the line y = kx - 2k + 3
def f (a : ℝ) (x : ℝ) := a^(x - 1)
def line (k : ℝ) (x : ℝ) : ℝ := k * x - 2 * k + 3
-- M and N coordinates
def M := (1, 1)
def N := (2, 3)

-- The theorem specific to proving the line equation
theorem line_MN_eq (a : ℝ) (k : ℝ) (h : a > 0 ∧ a ≠ 1) : 
  ∀ (x y : ℝ), (x, y) = M ∨ (x, y) = N -> 2 * x - y - 1 = 0 :=
by
  -- Using sorry to indicate the proof is not provided here
  sorry

end line_MN_eq_l234_234900


namespace minimum_point_coordinates_l234_234505

open Real

noncomputable def original_function (x : ℝ) : ℝ :=
  abs x ^ 2 - 3

noncomputable def translated_function (x : ℝ) : ℝ :=
  original_function (x - 1) - 4

theorem minimum_point_coordinates :
  (∃ x y : ℝ, translated_function x = y ∧ ∀ z : ℝ, translated_function z ≥ y ∧ (x, y) = (1, -7)) :=
by
  sorry

end minimum_point_coordinates_l234_234505


namespace problem_1_problem_2_l234_234600

-- Definitions and conditions for the problems
def A : Set ℝ := { x | abs (x - 2) < 3 }
def B (m : ℝ) : Set ℝ := { x | x^2 - 2 * x - m < 0 }

-- Problem (I)
theorem problem_1 : (A ∩ (Set.univ \ B 3)) = { x | 3 ≤ x ∧ x < 5 } :=
sorry

-- Problem (II)
theorem problem_2 (m : ℝ) : (A ∩ B m = { x | -1 < x ∧ x < 4 }) → m = 8 :=
sorry

end problem_1_problem_2_l234_234600


namespace dragos_wins_l234_234402

variable (S : Set ℕ) [Infinite S]
variable (x : ℕ → ℕ)
variable (M N : ℕ)
variable (p : ℕ)

theorem dragos_wins (h_prime_p : Nat.Prime p) (h_subset_S : p ∈ S) 
  (h_xn_distinct : ∀ i j, i ≠ j → x i ≠ x j) 
  (h_pM_div_xn : ∀ n, n ≥ N → p^M ∣ x n): 
  ∃ N, ∀ n, n ≥ N → p^M ∣ x n :=
sorry

end dragos_wins_l234_234402


namespace find_common_ratio_l234_234470

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_common_ratio (a1 a4 q : ℝ) (hq : q ^ 3 = 8) (ha1 : a1 = 8) (ha4 : a4 = 64)
  (a_def : is_geometric_sequence (fun n => a1 * q ^ n) q) :
  q = 2 :=
by
  sorry

end find_common_ratio_l234_234470


namespace downstream_distance_l234_234653

-- Define the speeds and distances as constants or variables
def speed_boat := 30 -- speed in kmph
def speed_stream := 10 -- speed in kmph
def distance_upstream := 40 -- distance in km
def time_upstream := distance_upstream / (speed_boat - speed_stream) -- time in hours

-- Define the variable for the downstream distance
variable {D : ℝ}

-- The Lean 4 statement to prove that the downstream distance is the specified value
theorem downstream_distance : 
  (time_upstream = D / (speed_boat + speed_stream)) → D = 80 :=
by
  sorry

end downstream_distance_l234_234653


namespace solve_quadratic_equation1_solve_quadratic_equation2_l234_234174

theorem solve_quadratic_equation1 (x : ℝ) : x^2 - 4 * x - 1 = 0 ↔ x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 := 
by sorry

theorem solve_quadratic_equation2 (x : ℝ) : (x + 3) * (x - 3) = 3 * (x + 3) ↔ x = -3 ∨ x = 6 :=
by sorry

end solve_quadratic_equation1_solve_quadratic_equation2_l234_234174


namespace sum_of_first_3m_terms_l234_234890

variable {a : ℕ → ℝ}   -- The arithmetic sequence
variable {S : ℕ → ℝ}   -- The sum of the first n terms of the sequence

def arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) : Prop :=
  S m = 30 ∧ S (2 * m) = 100 ∧ S (3 * m) = 170

theorem sum_of_first_3m_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) :
  arithmetic_sequence_sum a S m :=
by
  sorry

end sum_of_first_3m_terms_l234_234890


namespace fabric_amount_for_each_dress_l234_234768

def number_of_dresses (total_hours : ℕ) (hours_per_dress : ℕ) : ℕ :=
  total_hours / hours_per_dress 

def fabric_per_dress (total_fabric : ℕ) (number_of_dresses : ℕ) : ℕ :=
  total_fabric / number_of_dresses

theorem fabric_amount_for_each_dress (total_fabric : ℕ) (hours_per_dress : ℕ) (total_hours : ℕ) :
  total_fabric = 56 ∧ hours_per_dress = 3 ∧ total_hours = 42 →
  fabric_per_dress total_fabric (number_of_dresses total_hours hours_per_dress) = 4 :=
by
  sorry

end fabric_amount_for_each_dress_l234_234768


namespace unique_two_digit_integer_solution_l234_234022

variable {s : ℕ}

-- Conditions
def is_two_digit_positive_integer (s : ℕ) : Prop :=
  10 ≤ s ∧ s < 100

def last_two_digits_of_13s_are_52 (s : ℕ) : Prop :=
  13 * s % 100 = 52

-- Theorem statement
theorem unique_two_digit_integer_solution (h1 : is_two_digit_positive_integer s)
                                          (h2 : last_two_digits_of_13s_are_52 s) :
  s = 4 :=
sorry

end unique_two_digit_integer_solution_l234_234022


namespace semesters_needed_l234_234936

def total_credits : ℕ := 120
def credits_per_class : ℕ := 3
def classes_per_semester : ℕ := 5

theorem semesters_needed (h1 : total_credits = 120)
                         (h2 : credits_per_class = 3)
                         (h3 : classes_per_semester = 5) :
  total_credits / (credits_per_class * classes_per_semester) = 8 := 
by {
  sorry
}

end semesters_needed_l234_234936


namespace initial_distance_l234_234027

def relative_speed (v1 v2 : ℝ) : ℝ := v1 + v2

def total_distance (rel_speed time : ℝ) : ℝ := rel_speed * time

theorem initial_distance (v1 v2 time : ℝ) : (v1 = 1.6) → (v2 = 1.9) → 
                                            (time = 100) →
                                            total_distance (relative_speed v1 v2) time = 350 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp [relative_speed, total_distance]
  sorry

end initial_distance_l234_234027


namespace bennett_brothers_count_l234_234848

theorem bennett_brothers_count :
  ∃ B, B = 2 * 4 - 2 ∧ B = 6 :=
by
  sorry

end bennett_brothers_count_l234_234848


namespace find_b_of_triangle_ABC_l234_234634

theorem find_b_of_triangle_ABC (a b c : ℝ) (cos_A : ℝ) 
  (h1 : a = 2) 
  (h2 : c = 2 * Real.sqrt 3) 
  (h3 : cos_A = Real.sqrt 3 / 2) 
  (h4 : b < c) : 
  b = 2 := 
by
  sorry

end find_b_of_triangle_ABC_l234_234634


namespace candies_eaten_l234_234568

theorem candies_eaten (A B D : ℕ) 
                      (h1 : 4 * B = 3 * A) 
                      (h2 : 7 * A = 6 * D) 
                      (h3 : A + B + D = 70) :
  A = 24 ∧ B = 18 ∧ D = 28 := 
by
  sorry

end candies_eaten_l234_234568


namespace jake_fewer_peaches_l234_234644

theorem jake_fewer_peaches (steven_peaches : ℕ) (jake_peaches : ℕ) (h1 : steven_peaches = 19) (h2 : jake_peaches = 7) : steven_peaches - jake_peaches = 12 :=
sorry

end jake_fewer_peaches_l234_234644


namespace Mary_forgot_pigs_l234_234482

theorem Mary_forgot_pigs (Mary_thinks : ℕ) (actual_animals : ℕ) (double_counted_sheep : ℕ)
  (H_thinks : Mary_thinks = 60) (H_actual : actual_animals = 56)
  (H_double_counted : double_counted_sheep = 7) :
  ∃ pigs_forgot : ℕ, pigs_forgot = 3 :=
by
  let counted_animals := Mary_thinks - double_counted_sheep
  have H_counted_correct : counted_animals = 53 := by sorry -- 60 - 7 = 53
  have pigs_forgot := actual_animals - counted_animals
  have H_pigs_forgot : pigs_forgot = 3 := by sorry -- 56 - 53 = 3
  exact ⟨pigs_forgot, H_pigs_forgot⟩

end Mary_forgot_pigs_l234_234482


namespace range_of_a_l234_234746

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 3 ∧ -x^2 + 4*x + a = 0) ↔ (-3 ≤ a ∧ a ≤ 21) :=
by
  sorry

end range_of_a_l234_234746


namespace tens_digit_of_3_pow_2023_l234_234670

theorem tens_digit_of_3_pow_2023 : (3 ^ 2023 % 100) / 10 = 2 := 
sorry

end tens_digit_of_3_pow_2023_l234_234670


namespace arithmetic_mean_is_ten_l234_234109

theorem arithmetic_mean_is_ten (a b x : ℝ) (h₁ : a = 4) (h₂ : b = 16) (h₃ : x = (a + b) / 2) : x = 10 :=
by
  sorry

end arithmetic_mean_is_ten_l234_234109


namespace solve_for_question_mark_l234_234683

def cube_root (x : ℝ) := x^(1/3)
def square_root (x : ℝ) := x^(1/2)

theorem solve_for_question_mark : 
  cube_root (5568 / 87) + square_root (72 * 2) = square_root 256 := by
  sorry

end solve_for_question_mark_l234_234683


namespace part1_solution_set_part2_range_of_a_l234_234444

/-- Proof that the solution set of the inequality f(x) ≥ 6 when a = 1 is (-∞, -4] ∪ [2, ∞) -/
theorem part1_solution_set (x : ℝ) (h1 : f1 x ≥ 6) : 
  (x ≤ -4 ∨ x ≥ 2) :=
begin
  sorry
end

/-- Proof that the range of values for a if f(x) > -a is (-3/2, ∞) -/
theorem part2_range_of_a (a : ℝ) (h2 : ∀ x, f a x > -a) : 
  (-3/2 < a ∨ 0 ≤ a) :=
begin
  sorry
end

/-- Defining function f(x) = |x - a| + |x + 3| with specific 'a' value for part1 -/
def f1 (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

/-- Defining function f(x) = |x - a| + |x + 3| for part2 -/
def f (a : ℝ) (x : ℝ) : ℝ := abs (x - a) + abs (x + 3)

end part1_solution_set_part2_range_of_a_l234_234444


namespace lattice_point_count_l234_234205

noncomputable def countLatticePoints (N : ℤ) : ℤ :=
  2 * N * (N + 1) + 1

theorem lattice_point_count (N : ℤ) (hN : 71 * N > 0) :
    ∃ P, P = countLatticePoints N := sorry

end lattice_point_count_l234_234205


namespace right_triangle_perimeter_l234_234839

theorem right_triangle_perimeter (a b : ℝ) (c : ℝ) (h1 : a * b = 72) 
  (h2 : c ^ 2 = a ^ 2 + b ^ 2) (h3 : a = 12) :
  a + b + c = 18 + 6 * Real.sqrt 5 := 
by
  sorry

end right_triangle_perimeter_l234_234839


namespace fraction_product_sum_l234_234671

theorem fraction_product_sum :
  (1/3) * (5/6) * (3/7) + (1/4) * (1/8) = 101/672 :=
by
  sorry

end fraction_product_sum_l234_234671


namespace range_of_k_l234_234449

theorem range_of_k (k : ℝ) : 
  (∀ x : ℝ, k * x ^ 2 + 2 * k * x + 3 ≠ 0) ↔ (0 ≤ k ∧ k < 3) :=
by sorry

end range_of_k_l234_234449


namespace alpha_sufficient_not_necessary_l234_234917

def A := {x : ℝ | 2 < x ∧ x < 3}

def B (α : ℝ) := {x : ℝ | (x + 2) * (x - α) < 0}

theorem alpha_sufficient_not_necessary (α : ℝ) : 
  (α = 1 → A ∩ B α = ∅) ∧ (∃ β : ℝ, β ≠ 1 ∧ A ∩ B β = ∅) :=
by
  sorry

end alpha_sufficient_not_necessary_l234_234917


namespace max_blocks_fit_l234_234028

-- Define the dimensions of the block
def block_length := 2
def block_width := 3
def block_height := 1

-- Define the dimensions of the container box
def box_length := 4
def box_width := 3
def box_height := 3

-- Define the volume calculations
def volume (length width height : ℕ) : ℕ := length * width * height

def block_volume := volume block_length block_width block_height
def box_volume := volume box_length box_width box_height

-- The theorem to prove
theorem max_blocks_fit : (box_volume / block_volume) = 6 :=
by
  sorry

end max_blocks_fit_l234_234028


namespace graph_passes_through_point_l234_234334

theorem graph_passes_through_point :
  ∀ (a : ℝ), 0 < a ∧ a < 1 → (∃ (x y : ℝ), (x = 2) ∧ (y = -1) ∧ (y = 2 * a * x - 1)) :=
by
  sorry

end graph_passes_through_point_l234_234334


namespace Shekar_science_marks_l234_234946

theorem Shekar_science_marks (S : ℕ) : 
  let math_marks := 76
  let social_studies_marks := 82
  let english_marks := 67
  let biology_marks := 75
  let average_marks := 73
  let num_subjects := 5
  ((math_marks + S + social_studies_marks + english_marks + biology_marks) / num_subjects = average_marks) → S = 65 :=
by
  sorry

end Shekar_science_marks_l234_234946


namespace max_coefficient_terms_l234_234616

theorem max_coefficient_terms (x : ℝ) :
  let n := 8
  let T_3 := 7 * x^2
  let T_4 := 7 * x
  true := by
  sorry

end max_coefficient_terms_l234_234616


namespace find_missing_number_l234_234039

theorem find_missing_number (x : ℕ) :
  (6 + 16 + 8 + x) / 4 = 13 → x = 22 :=
by
  sorry

end find_missing_number_l234_234039


namespace train_speed_l234_234371

-- Definition of the problem
def train_length : ℝ := 350
def time_to_cross_man : ℝ := 4.5
def expected_speed : ℝ := 77.78

-- Theorem statement
theorem train_speed :
  train_length / time_to_cross_man = expected_speed :=
sorry

end train_speed_l234_234371


namespace waiters_dropped_out_l234_234791

theorem waiters_dropped_out (initial_chefs initial_waiters chefs_dropped remaining_staff : ℕ)
  (h1 : initial_chefs = 16) 
  (h2 : initial_waiters = 16) 
  (h3 : chefs_dropped = 6) 
  (h4 : remaining_staff = 23) : 
  initial_waiters - (remaining_staff - (initial_chefs - chefs_dropped)) = 3 := 
by 
  sorry

end waiters_dropped_out_l234_234791


namespace part1_tangent_line_at_x2_part2_inequality_l234_234426

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x + Real.exp 2 - 7

theorem part1_tangent_line_at_x2 (a : ℝ) (h_a : a = 2) :
  ∃ m b : ℝ, (∀ x : ℝ, f x a = m * x + b) ∧ m = Real.exp 2 - 2 ∧ b = -(2 * Real.exp 2 - 7) := by
  sorry

theorem part2_inequality (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x a ≥ (7 / 4) * x^2) → a ≤ Real.exp 2 - 7 := by
  sorry

end part1_tangent_line_at_x2_part2_inequality_l234_234426


namespace red_or_black_prob_red_black_or_white_prob_l234_234980

-- Defining the probabilities
def prob_red : ℚ := 5 / 12
def prob_black : ℚ := 4 / 12
def prob_white : ℚ := 2 / 12
def prob_green : ℚ := 1 / 12

-- Question 1: Probability of drawing a red or black ball
theorem red_or_black_prob : prob_red + prob_black = 3 / 4 :=
by sorry

-- Question 2: Probability of drawing a red, black, or white ball
theorem red_black_or_white_prob : prob_red + prob_black + prob_white = 11 / 12 :=
by sorry

end red_or_black_prob_red_black_or_white_prob_l234_234980


namespace top_triangle_is_multiple_of_5_l234_234383

-- Definitions of the conditions given in the problem

def lower_left_triangle := 12
def lower_right_triangle := 3

-- Let a, b, c, d be the four remaining numbers in the bottom row
variables (a b c d : ℤ)

-- Conditions that the sums of triangles must be congruent to multiples of 5
def second_lowest_row : Prop :=
  (3 - a) % 5 = 0 ∧
  (-a - b) % 5 = 0 ∧
  (-b - c) % 5 = 0 ∧
  (-c - d) % 5 = 0 ∧
  (2 - d) % 5 = 0

def third_lowest_row : Prop :=
  (2 + 2*a + b) % 5 = 0 ∧
  (a + 2*b + c) % 5 = 0 ∧
  (b + 2*c + d) % 5 = 0 ∧
  (3 + c + 2*d) % 5 = 0

def fourth_lowest_row : Prop :=
  (3 + 2*a + 2*b - c) % 5 = 0 ∧
  (-a + 2*b + 2*c - d) % 5 = 0 ∧
  (2 - b + 2*c + 2*d) % 5 = 0

def second_highest_row : Prop :=
  (2 - a + b - c + d) % 5 = 0 ∧
  (3 + a - b + c - d) % 5 = 0

def top_triangle : Prop :=
  (2 - a + b - c + d + 3 + a - b + c - d) % 5 = 0

theorem top_triangle_is_multiple_of_5 (a b c d : ℤ) :
  second_lowest_row a b c d →
  third_lowest_row a b c d →
  fourth_lowest_row a b c d →
  second_highest_row a b c d →
  top_triangle a b c d →
  ∃ k : ℤ, (2 - a + b - c + d + 3 + a - b + c - d) = 5 * k :=
by sorry

end top_triangle_is_multiple_of_5_l234_234383


namespace find_difference_l234_234452

theorem find_difference (a b : ℕ) (h1 : a < b) (h2 : a + b = 78) (h3 : Nat.lcm a b = 252) : b - a = 6 :=
by
  sorry

end find_difference_l234_234452


namespace common_root_equation_l234_234501

theorem common_root_equation (a b r : ℝ) (h₁ : a ≠ b)
  (h₂ : r^2 + 2019 * a * r + b = 0)
  (h₃ : r^2 + 2019 * b * r + a = 0) :
  r = 1 / 2019 :=
by
  sorry

end common_root_equation_l234_234501


namespace problem1_problem2_l234_234394

-- Statement for Problem 1
theorem problem1 (x y : ℝ) : (x - y) ^ 2 + x * (x + 2 * y) = 2 * x ^ 2 + y ^ 2 :=
by sorry

-- Statement for Problem 2
theorem problem2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  ((-3 * x + 4) / (x - 1) + x) / ((x - 2) / (x ^ 2 - x)) = x ^ 2 - 2 * x :=
by sorry

end problem1_problem2_l234_234394


namespace sum_of_two_numbers_l234_234879

theorem sum_of_two_numbers (S : ℝ) (L : ℝ) (h1 : S = 3.5) (h2 : L = 3 * S) : S + L = 14 :=
by
  sorry

end sum_of_two_numbers_l234_234879


namespace find_x_l234_234095

theorem find_x (x : ℝ) (h1 : x > 0) (h2 : ⌊x⌋ * x = 72) : x = 9 := by
  sorry

end find_x_l234_234095


namespace in_proportion_d_value_l234_234739

noncomputable def d_length (a b c : ℝ) : ℝ := (b * c) / a

theorem in_proportion_d_value :
  let a := 2
  let b := 3
  let c := 6
  d_length a b c = 9 := 
by
  sorry

end in_proportion_d_value_l234_234739


namespace ab_eq_neg_two_l234_234133

theorem ab_eq_neg_two (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : a * b^a = -2 :=
by
  sorry

end ab_eq_neg_two_l234_234133


namespace central_angle_eighth_grade_is_correct_l234_234921

-- Define the total number of students in each grade
def seventh_grade_students := 374
def eighth_grade_students := 420
def ninth_grade_students := 406

-- Define the total number of students
def total_students := seventh_grade_students + eighth_grade_students + ninth_grade_students

-- Define the fraction of 8th grade students
def fraction_eighth_grade := (eighth_grade_students : ℚ) / total_students

-- Define the central angle for 8th grade students in the pie chart
def central_angle_eighth_grade := fraction_eighth_grade * 360

theorem central_angle_eighth_grade_is_correct : central_angle_eighth_grade = 126 := by
  -- calculations steps are skipped as per the instruction to add proof with sorry
  sorry

end central_angle_eighth_grade_is_correct_l234_234921


namespace record_expenditure_20_l234_234753

-- Define the concept of recording financial transactions
def record_income (amount : ℤ) : ℤ := amount

def record_expenditure (amount : ℤ) : ℤ := -amount

-- Given conditions
variable (income : ℤ) (expenditure : ℤ)

-- Condition: the income of 30 yuan is recorded as +30 yuan
axiom income_record : record_income 30 = 30

-- Prove an expenditure of 20 yuan is recorded as -20 yuan
theorem record_expenditure_20 : record_expenditure 20 = -20 := 
  by sorry

end record_expenditure_20_l234_234753


namespace Freddie_ratio_l234_234195

noncomputable def Veronica_distance : ℕ := 1000

noncomputable def Freddie_distance (F : ℕ) : Prop :=
  1000 + 12000 = 5 * F - 2000

theorem Freddie_ratio (F : ℕ) (h : Freddie_distance F) :
  F / Veronica_distance = 3 := by
  sorry

end Freddie_ratio_l234_234195


namespace gcf_7fact_8fact_l234_234246

theorem gcf_7fact_8fact : 
  let f_7 := Nat.factorial 7
  let f_8 := Nat.factorial 8
  Nat.gcd f_7 f_8 = f_7 := 
by
  sorry

end gcf_7fact_8fact_l234_234246


namespace speed_of_water_l234_234212

theorem speed_of_water (v : ℝ) (swim_speed_still_water : ℝ)
  (distance : ℝ) (time : ℝ)
  (h1 : swim_speed_still_water = 4) 
  (h2 : distance = 14) 
  (h3 : time = 7) 
  (h4 : 4 - v = distance / time) : 
  v = 2 := 
sorry

end speed_of_water_l234_234212


namespace candies_eaten_l234_234569

variables (A B D : ℕ)

-- Conditions:
def condition1 : Prop := ∃ k1 k2 k3 : ℕ, k1 * 4 + k2 * 3 + k3 * 7 = 70
def condition2 : Prop := (B * 3 = A * 4) ∧ (D * 7 = A * 6)
def condition3 : Prop := A + B + D = 70

-- Theorem statement:
theorem candies_eaten (h1 : condition1) (h2 : condition2) (h3 : condition3) :
    A = 24 ∧ B = 18 ∧ D = 28 := sorry

end candies_eaten_l234_234569


namespace solution_l234_234915

-- Conditions
def x : ℚ := 3/5
def y : ℚ := 5/3

-- Proof problem
theorem solution : (1/3) * x^8 * y^9 = 5/9 := sorry

end solution_l234_234915


namespace division_problem_l234_234857

theorem division_problem :
  0.045 / 0.0075 = 6 :=
sorry

end division_problem_l234_234857


namespace shape_is_cylinder_l234_234728

def is_cylinder (c : ℝ) (r θ z : ℝ) : Prop :=
  c > 0 ∧ r = c ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ True

theorem shape_is_cylinder (c : ℝ) (r θ z : ℝ) (h : c > 0) :
  is_cylinder c r θ z :=
by
  -- Proof is omitted
  sorry

end shape_is_cylinder_l234_234728


namespace chameleons_all_white_l234_234485

theorem chameleons_all_white :
  ∀ (a b c : ℕ), a = 800 → b = 1000 → c = 1220 → 
  (a + b + c = 3020) → (a % 3 = 2) → (b % 3 = 1) → (c % 3 = 2) →
    ∃ k : ℕ, (k = 3020 ∧ (k % 3 = 1)) ∧ 
    (if k = b then a = 0 ∧ c = 0 else false) :=
by
  sorry

end chameleons_all_white_l234_234485


namespace subtraction_is_addition_of_negatives_l234_234227

theorem subtraction_is_addition_of_negatives : (-1) - 3 = -4 := by
  sorry

end subtraction_is_addition_of_negatives_l234_234227


namespace expression_evaluation_l234_234038

theorem expression_evaluation (a b : ℤ) (h1 : a = 4) (h2 : b = -2) : -a - b^4 + a * b = -28 := 
by 
  sorry

end expression_evaluation_l234_234038


namespace problem_l234_234862

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem problem : (nabla (nabla 2 3) 4) = 16777219 :=
by
  unfold nabla
  -- First compute 2 ∇ 3
  have h1 : nabla 2 3 = 12 := by norm_num
  rw [h1]
  -- Now compute 12 ∇ 4
  unfold nabla
  norm_num
  sorry

end problem_l234_234862


namespace solution_set_quadratic_inequality_l234_234652

def quadratic_inequality_solution (x : ℝ) : Prop := x^2 + x - 2 > 0

theorem solution_set_quadratic_inequality :
  {x : ℝ | quadratic_inequality_solution x} = {x : ℝ | x < -2 ∨ x > 1} :=
by
  sorry

end solution_set_quadratic_inequality_l234_234652


namespace mom_younger_than_grandmom_l234_234230

def cara_age : ℕ := 40
def cara_younger_mom : ℕ := 20
def grandmom_age : ℕ := 75

def mom_age : ℕ := cara_age + cara_younger_mom
def age_difference : ℕ := grandmom_age - mom_age

theorem mom_younger_than_grandmom : age_difference = 15 := by
  sorry

end mom_younger_than_grandmom_l234_234230


namespace exists_disjoint_A_B_l234_234881

def S (C : Finset ℕ) := C.sum id

theorem exists_disjoint_A_B : 
  ∃ (A B : Finset ℕ), 
  A ≠ ∅ ∧ B ≠ ∅ ∧ 
  A ∩ B = ∅ ∧ 
  A ∪ B = (Finset.range (2021 + 1)).erase 0 ∧ 
  ∃ k : ℕ, S A * S B = k^2 :=
by 
  sorry

end exists_disjoint_A_B_l234_234881


namespace chicago_bulls_wins_l234_234140

theorem chicago_bulls_wins (B H : ℕ) (h1 : B + H = 145) (h2 : H = B + 5) : B = 70 :=
by
  sorry

end chicago_bulls_wins_l234_234140


namespace cube_root_product_l234_234948

theorem cube_root_product : (343 : ℝ)^(1/3) * (125 : ℝ)^(1/3) = 35 := 
by
  sorry

end cube_root_product_l234_234948


namespace stone_radius_l234_234214

theorem stone_radius (hole_diameter hole_depth : ℝ) (r : ℝ) :
  hole_diameter = 30 → hole_depth = 10 → (r - 10)^2 + 15^2 = r^2 → r = 16.25 :=
by
  intros h_diam h_depth hyp_eq
  sorry

end stone_radius_l234_234214


namespace largest_value_of_number_l234_234655

theorem largest_value_of_number 
  (v w x y z : ℝ)
  (h1 : v + w + x + y + z = 8)
  (h2 : v^2 + w^2 + x^2 + y^2 + z^2 = 16) :
  ∃ (m : ℝ), m = 2.4 ∧ (m = v ∨ m = w ∨ m = x ∨ m = y ∨ m = z) :=
sorry

end largest_value_of_number_l234_234655


namespace tangent_line_through_P_l234_234731

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem tangent_line_through_P (x : ℝ) :
  let P := (-2 : ℝ, -2 : ℝ) in
  ∀ (l : ℝ → ℝ), (∃ a b : ℝ, l = λ x, a*x + b) →
  (∀ x, (l x = f x) → a = -9 ∧ b = 16) ∨ (l = λ x, -2) :=
by
  sorry

end tangent_line_through_P_l234_234731


namespace compute_expression_l234_234079

theorem compute_expression : 1 + 6 * 2 - 3 + 5 * 4 / 2 = 20 :=
by sorry

end compute_expression_l234_234079


namespace complete_square_eq_l234_234077

theorem complete_square_eq (b c : ℤ) (h : ∃ b c : ℤ, (∀ x : ℝ, (x - 5)^2 = b * x + c) ∧ b + c = 5) :
  b + c = 5 :=
sorry

end complete_square_eq_l234_234077


namespace time_to_travel_A_to_C_is_6_l234_234989

-- Assume the existence of a real number t representing the time taken
-- Assume constant speed r for the river current and p for the power boat relative to the river.
variables (t r p : ℝ)

-- Conditions
axiom condition1 : p > 0
axiom condition2 : r > 0
axiom condition3 : t * (1.5 * (p + r)) + (p - r) * (12 - t) = 12 * r

-- Define the time taken for the power boat to travel from A to C
def time_from_A_to_C : ℝ := t

-- The proof problem: Prove time_from_A_to_C = 6 under the given conditions
theorem time_to_travel_A_to_C_is_6 : time_from_A_to_C = 6 := by
  sorry

end time_to_travel_A_to_C_is_6_l234_234989


namespace andrey_boris_denis_eat_candies_l234_234558

def andrey_boris_condition (a b : ℕ) : Prop :=
  a = 4 ∧ b = 3

def andrey_denis_condition (a d : ℕ) : Prop :=
  a = 6 ∧ d = 7

def total_candies_condition (total : ℕ) : Prop :=
  total = 70

theorem andrey_boris_denis_eat_candies :
  ∃ (a b d : ℕ), andrey_boris_condition a b ∧ andrey_denis_condition a d ∧ 
                  (total_candies_condition (2 * (12 + 9 + 14)) ∧ 
                   2 * 12 = 24 ∧ 2 * 9 = 18 ∧ 2 * 14 = 28) →
                  (a = 24 ∧ b = 18 ∧ d = 28) :=
by
  sorry

end andrey_boris_denis_eat_candies_l234_234558


namespace cooler1_water_left_l234_234532

noncomputable def waterLeftInFirstCooler (gallons1 gallons2 : ℝ) (chairs rows : ℕ) (ozSmall ozLarge ozPerGallon : ℝ) : ℝ :=
  let totalChairs := chairs * rows
  let totalSmallOunces := totalChairs * ozSmall
  let initialOunces1 := gallons1 * ozPerGallon
  initialOunces1 - totalSmallOunces

theorem cooler1_water_left :
  waterLeftInFirstCooler 4.5 3.25 12 7 4 8 128 = 240 :=
by
  sorry

end cooler1_water_left_l234_234532


namespace min_S_value_l234_234199

theorem min_S_value (n : ℕ) (h₁ : n ≥ 375) :
    let R := 3000
    let S := 9 * n - R
    let dice_sum (s : ℕ) := ∃ L : List ℕ, (∀ x ∈ L, 1 ≤ x ∧ x ≤ 8) ∧ L.sum = s
    dice_sum R ∧ S = 375 := 
by
  sorry

end min_S_value_l234_234199


namespace black_balls_count_l234_234050

theorem black_balls_count
  (P_red P_white : ℝ)
  (Red_balls_count : ℕ)
  (h1 : P_red = 0.42)
  (h2 : P_white = 0.28)
  (h3 : Red_balls_count = 21) :
  ∃ B, B = 15 :=
by
  sorry

end black_balls_count_l234_234050


namespace functional_eq_solution_l234_234585

theorem functional_eq_solution (f : ℝ → ℝ) 
  (H : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x := 
by 
  sorry

end functional_eq_solution_l234_234585


namespace sum_max_min_fourth_row_from_bottom_l234_234770

def grid_size : Nat := 16
def num_elements : Nat := grid_size * grid_size
def start_position : Nat × Nat := (8, 8)

noncomputable def fill_grid : List (Nat × Nat × Nat) := sorry
-- This would be a function generating the counterclockwise spiral grid filling, but 
-- it's non-trivial and not essential to write out for this proof problem.

theorem sum_max_min_fourth_row_from_bottom :
  let row := 4
  let bottom_offset := row - 1
  let grid : List (Nat × Nat × Nat) := fill_grid
  let bottom_row_fourth := List.filter (λ (triplet : Nat × Nat × Nat), triplet.snd.snd = grid_size - bottom_offset) grid
  let max_in_row := List.maximum bottom_row_fourth.map (λ triplet => triplet.fst)
  let min_in_row := List.minimum bottom_row_fourth.map (λ triplet => triplet.fst)
  max_in_row + min_in_row = 497 :=
by
  sorry

end sum_max_min_fourth_row_from_bottom_l234_234770


namespace fourth_grade_students_l234_234977

theorem fourth_grade_students (initial_students : ℕ) (students_left : ℕ) (new_students : ℕ) 
  (h_initial : initial_students = 35) (h_left : students_left = 10) (h_new : new_students = 10) :
  initial_students - students_left + new_students = 35 :=
by
  -- The proof goes here
  sorry

end fourth_grade_students_l234_234977


namespace max_value_frac_sqrt_eq_sqrt_35_l234_234877

theorem max_value_frac_sqrt_eq_sqrt_35 :
  ∀ x y : ℝ, 
  (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 35 
  ∧ (∃ x y : ℝ, x = 2 / 5 ∧ y = 6 / 5 ∧ (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4) = Real.sqrt 35) :=
by {
  sorry
}

end max_value_frac_sqrt_eq_sqrt_35_l234_234877


namespace candies_eaten_l234_234571

variables (A B D : ℕ)

-- Conditions:
def condition1 : Prop := ∃ k1 k2 k3 : ℕ, k1 * 4 + k2 * 3 + k3 * 7 = 70
def condition2 : Prop := (B * 3 = A * 4) ∧ (D * 7 = A * 6)
def condition3 : Prop := A + B + D = 70

-- Theorem statement:
theorem candies_eaten (h1 : condition1) (h2 : condition2) (h3 : condition3) :
    A = 24 ∧ B = 18 ∧ D = 28 := sorry

end candies_eaten_l234_234571


namespace store_profit_l234_234546

theorem store_profit (m n : ℝ) (hmn : m > n) : 
  let selling_price := (m + n) / 2
  let profit_a := 40 * (selling_price - m)
  let profit_b := 60 * (selling_price - n)
  let total_profit := profit_a + profit_b
  total_profit > 0 :=
by sorry

end store_profit_l234_234546


namespace all_sets_form_right_angled_triangle_l234_234217

theorem all_sets_form_right_angled_triangle :
    (6 * 6 + 8 * 8 = 10 * 10) ∧
    (7 * 7 + 24 * 24 = 25 * 25) ∧
    (3 * 3 + 4 * 4 = 5 * 5) ∧
    (Real.sqrt 2 * Real.sqrt 2 + Real.sqrt 3 * Real.sqrt 3 = Real.sqrt 5 * Real.sqrt 5) :=
by {
  sorry
}

end all_sets_form_right_angled_triangle_l234_234217


namespace eval_nabla_l234_234866

namespace MathProblem

-- Definition of the operation
def nabla (a b : ℕ) : ℕ :=
  3 + b ^ a

-- Theorem statement
theorem eval_nabla : nabla (nabla 2 3) 4 = 16777219 :=
by
  sorry

end MathProblem

end eval_nabla_l234_234866


namespace num_ways_for_volunteers_l234_234663

theorem num_ways_for_volunteers:
  let pavilions := 4
  let volunteers := 5
  let ways_to_choose_A := 4
  let ways_to_choose_B_after_A := 3
  let total_distributions := 
    let case_1 := 2
    let case_2 := (2^3) - 2
    case_1 + case_2
  ways_to_choose_A * ways_to_choose_B_after_A * total_distributions = 72 := 
by
  sorry

end num_ways_for_volunteers_l234_234663


namespace Bella_average_speed_l234_234074

theorem Bella_average_speed :
  ∀ (distance time : ℝ), 
  distance = 790 → 
  time = 15.8 → 
  (distance / time) = 50 :=
by intros distance time h_dist h_time
   -- According to the provided distances and time,
   -- we need to prove that the calculated speed is 50.
   sorry

end Bella_average_speed_l234_234074


namespace find_quadratic_function_l234_234586

def quadratic_function (c d : ℝ) (x : ℝ) : ℝ :=
  x^2 + c * x + d

theorem find_quadratic_function :
  ∃ c d, (∀ x, 
    (quadratic_function c d (quadratic_function c d x + 2 * x)) / (quadratic_function c d x) = 2 * x^2 + 1984 * x + 2024) ∧ 
    quadratic_function c d x = x^2 + 1982 * x + 21 :=
by
  sorry

end find_quadratic_function_l234_234586


namespace joan_apples_l234_234621

def initial_apples : ℕ := 43
def additional_apples : ℕ := 27
def total_apples (initial additional: ℕ) := initial + additional

theorem joan_apples : total_apples initial_apples additional_apples = 70 := by
  sorry

end joan_apples_l234_234621


namespace consecutive_sum_is_10_l234_234513

theorem consecutive_sum_is_10 (a : ℕ) (h : a + (a + 1) + (a + 2) + (a + 3) + (a + 4) = 50) : a + 2 = 10 :=
sorry

end consecutive_sum_is_10_l234_234513


namespace contrapositive_inequality_l234_234606

theorem contrapositive_inequality {x y : ℝ} (h : x^2 ≤ y^2) : x ≤ y :=
  sorry

end contrapositive_inequality_l234_234606


namespace ratio_a_to_c_l234_234018

-- Declaring the variables a, b, c, and d as real numbers.
variables (a b c d : ℝ)

-- Define the conditions given in the problem.
def ratio_conditions : Prop :=
  (a / b = 5 / 4) ∧ (c / d = 4 / 3) ∧ (d / b = 1 / 5)

-- State the theorem we need to prove based on the conditions.
theorem ratio_a_to_c (h : ratio_conditions a b c d) : a / c = 75 / 16 :=
by
  sorry

end ratio_a_to_c_l234_234018


namespace largest_three_digit_int_l234_234197

theorem largest_three_digit_int (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 999) (h2 : 75 * n ≡ 225 [MOD 300]) : n = 999 :=
sorry

end largest_three_digit_int_l234_234197


namespace equivalent_single_discount_calculation_l234_234226

-- Definitions for the successive discounts
def discount10 (x : ℝ) : ℝ := 0.90 * x
def discount15 (x : ℝ) : ℝ := 0.85 * x
def discount25 (x : ℝ) : ℝ := 0.75 * x

-- Final price after applying all discounts
def final_price (x : ℝ) : ℝ := discount25 (discount15 (discount10 x))

-- Equivalent single discount fraction
def equivalent_discount (x : ℝ) : ℝ := 0.57375 * x

theorem equivalent_single_discount_calculation (x : ℝ) : 
  final_price x = equivalent_discount x :=
sorry

end equivalent_single_discount_calculation_l234_234226


namespace monotonic_decreasing_interval_l234_234794

noncomputable def y (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem monotonic_decreasing_interval :
  {x : ℝ | (∃ y', y' = 3 * x^2 - 3 ∧ y' < 0)} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  sorry

end monotonic_decreasing_interval_l234_234794


namespace gcf_fact7_fact8_l234_234245

-- Definition of factorial
def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the values 7! and 8!
def fact_7 : ℕ := factorial 7
def fact_8 : ℕ := factorial 8

-- Prove that the greatest common factor of 7! and 8! is 7!
theorem gcf_fact7_fact8 : Nat.gcd fact_7 fact_8 = fact_7 :=
by
  sorry

end gcf_fact7_fact8_l234_234245


namespace prob_two_girls_is_one_fourth_l234_234699

-- Define the probability of giving birth to a girl
def prob_girl : ℚ := 1 / 2

-- Define the probability of having two girls
def prob_two_girls : ℚ := prob_girl * prob_girl

-- Theorem statement: The probability of having two girls is 1/4
theorem prob_two_girls_is_one_fourth : prob_two_girls = 1 / 4 :=
by sorry

end prob_two_girls_is_one_fourth_l234_234699


namespace a_4_value_l234_234119

def seq (n : ℕ) : ℚ :=
  if n = 0 then 0 -- To handle ℕ index starting from 0.
  else if n = 1 then 1
  else seq (n - 1) + 1 / ((n:ℚ) * (n-1))

noncomputable def a_4 : ℚ := seq 4

theorem a_4_value : a_4 = 7 / 4 := 
  by sorry

end a_4_value_l234_234119


namespace find_a11_l234_234049

-- Defining the sequence a_n and its properties
def seq (a : ℕ → ℝ) : Prop :=
  (a 3 = 2) ∧ 
  (a 5 = 1) ∧ 
  (∃ d, ∀ n, (1 / (1 + a n)) = (1 / (1 + a 1)) + (n - 1) * d)

-- The goal is to prove that the value of a_{11} is 0
theorem find_a11 (a : ℕ → ℝ) (h : seq a) : a 11 = 0 :=
sorry

end find_a11_l234_234049


namespace total_pawns_left_is_10_l234_234012

noncomputable def total_pawns_left_in_game 
    (initial_pawns : ℕ)
    (sophia_lost : ℕ)
    (chloe_lost : ℕ) : ℕ :=
  initial_pawns - sophia_lost + (initial_pawns - chloe_lost)

theorem total_pawns_left_is_10 :
  total_pawns_left_in_game 8 5 1 = 10 := by
  sorry

end total_pawns_left_is_10_l234_234012


namespace eval_nabla_l234_234868

namespace MathProblem

-- Definition of the operation
def nabla (a b : ℕ) : ℕ :=
  3 + b ^ a

-- Theorem statement
theorem eval_nabla : nabla (nabla 2 3) 4 = 16777219 :=
by
  sorry

end MathProblem

end eval_nabla_l234_234868


namespace solution_to_system_l234_234722

theorem solution_to_system (x y z : ℝ) (h1 : x^2 + y^2 = 6 * z) (h2 : y^2 + z^2 = 6 * x) (h3 : z^2 + x^2 = 6 * y) :
  (x = 3) ∧ (y = 3) ∧ (z = 3) :=
sorry

end solution_to_system_l234_234722


namespace min_value_m2n_mn_l234_234592

theorem min_value_m2n_mn (m n : ℝ) 
  (h1 : (x - m)^2 + (y - n)^2 = 9)
  (h2 : x + 2 * y + 2 = 0)
  (h3 : 0 < m)
  (h4 : 0 < n)
  (h5 : m + 2 * n + 2 = 5)
  (h6 : ∃ l : ℝ, l = 4 ): (m + 2 * n) / (m * n) = 8/3 :=
by
  sorry

end min_value_m2n_mn_l234_234592


namespace compute_B_93_l234_234929

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 0, 0], ![0, 0, -1], ![0, 1, 0]]

theorem compute_B_93 : B^93 = B := by
  sorry

end compute_B_93_l234_234929


namespace elena_marco_sum_ratio_l234_234873

noncomputable def sum_odds (n : Nat) : Nat := (n / 2 + 1) * n

noncomputable def sum_integers (n : Nat) : Nat := n * (n + 1) / 2

theorem elena_marco_sum_ratio :
  (sum_odds 499) / (sum_integers 250) = 2 :=
by
  sorry

end elena_marco_sum_ratio_l234_234873


namespace value_of_f_2011_l234_234740

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_2011 (h_even : ∀ x : ℝ, f x = f (-x))
                       (h_sym : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → f (2 + x) = f (2 - x))
                       (h_def : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → f x = 2^x) : 
  f 2011 = 1 / 2 := 
sorry

end value_of_f_2011_l234_234740


namespace solve_fraction_l234_234815

theorem solve_fraction : (3.242 * 10) / 100 = 0.3242 := by
  sorry

end solve_fraction_l234_234815


namespace sum_c_d_l234_234270

theorem sum_c_d (c d : ℝ) (h : ∀ x, (x - 2) * (x + 3) = x^2 + c * x + d) :
  c + d = -5 :=
sorry

end sum_c_d_l234_234270


namespace phase_shift_right_by_pi_div_3_l234_234353

noncomputable def graph_shift_right_by_pi_div_3 
  (A : ℝ := 1) 
  (ω : ℝ := 1) 
  (φ : ℝ := - (Real.pi / 3)) 
  (y : ℝ → ℝ := fun x => Real.sin (x - Real.pi / 3)) : 
  Prop :=
  y = fun x => Real.sin (x - (Real.pi / 3))

theorem phase_shift_right_by_pi_div_3 (A : ℝ := 1) (ω : ℝ := 1) (φ : ℝ := - (Real.pi / 3)) :
  graph_shift_right_by_pi_div_3 A ω φ (fun x => Real.sin (x - Real.pi / 3)) :=
sorry

end phase_shift_right_by_pi_div_3_l234_234353


namespace geometric_sequence_S6_l234_234765

noncomputable def sum_of_first_n_terms (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_S6 (a r : ℝ) (h1 : sum_of_first_n_terms a r 2 = 6) (h2 : sum_of_first_n_terms a r 4 = 30) : 
  sum_of_first_n_terms a r 6 = 126 :=
sorry

end geometric_sequence_S6_l234_234765


namespace rental_cost_equal_mileage_l234_234494

theorem rental_cost_equal_mileage:
  ∃ x : ℝ, (17.99 + 0.18 * x = 18.95 + 0.16 * x) ∧ x = 48 := 
by
  sorry

end rental_cost_equal_mileage_l234_234494


namespace second_quadrant_condition_l234_234820

def is_obtuse (α : ℝ) : Prop := 90 < α ∧ α < 180
def is_in_second_quadrant (α : ℝ) : Prop := 90 < α ∧ α < 180 ∨ -270 < α ∧ α < -180

theorem second_quadrant_condition (α : ℝ) : 
  (is_obtuse α → is_in_second_quadrant α) ∧ ¬(is_in_second_quadrant α → is_obtuse α) := 
by
  sorry

end second_quadrant_condition_l234_234820


namespace unique_positive_integer_solution_l234_234362

theorem unique_positive_integer_solution :
  ∃! (x : ℕ), (4 * x)^2 - 2 * x = 2652 := sorry

end unique_positive_integer_solution_l234_234362


namespace find_sum_of_p_q_r_s_l234_234306

theorem find_sum_of_p_q_r_s 
    (p q r s : ℝ)
    (h1 : r + s = 12 * p)
    (h2 : r * s = -13 * q)
    (h3 : p + q = 12 * r)
    (h4 : p * q = -13 * s)
    (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
    p + q + r + s = 2028 := 
sorry

end find_sum_of_p_q_r_s_l234_234306


namespace polygon_number_of_sides_l234_234834

theorem polygon_number_of_sides (P : ℝ) (L : ℝ) (n : ℕ) : P = 180 ∧ L = 15 ∧ n = P / L → n = 12 := by
  sorry

end polygon_number_of_sides_l234_234834


namespace b_a_range_l234_234614
open Real

-- Definitions of angles A, B, and sides a, b in an acute triangle ABC we assume that these are given.
variables {A B C a b c : ℝ}
variable {ABC_acute : A + B + C = π}
variable {angle_condition : B = 2 * A}
variable {sides : a = b * (sin A / sin B)}

theorem b_a_range (h₁ : 0 < A) (h₂ : A < π/2) (h₃ : 0 < C) (h₄ : C < π/2) :
  (∃ A, 30 * (π/180) < A ∧ A < 45 * (π/180)) → 
  (∃ b a, b / a = 2 * cos A) → 
  (∃ x : ℝ, x = b / a ∧ sqrt 2 < x ∧ x < sqrt 3) :=
sorry

end b_a_range_l234_234614


namespace arithmetic_sequence_sum_l234_234767

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ)
  (h1 : ∀ n, S n = n * ((a 1 + a n) / 2))
  (h2 : S 9 = 27) :
  a 4 + a 6 = 6 := 
sorry

end arithmetic_sequence_sum_l234_234767


namespace total_pools_l234_234939

def patsPools (numAStores numPStores poolsA ratio : ℕ) : ℕ :=
  numAStores * poolsA + numPStores * (ratio * poolsA)

theorem total_pools : 
  patsPools 6 4 200 3 = 3600 := 
by 
  sorry

end total_pools_l234_234939


namespace sum_gcd_lcm_l234_234673

theorem sum_gcd_lcm (a b c d : ℕ) (ha : a = 15) (hb : b = 45) (hc : c = 30) :
  Int.gcd a b + Nat.lcm a c = 45 := 
by
  sorry

end sum_gcd_lcm_l234_234673


namespace total_people_counted_l234_234999

-- Definitions based on conditions
def people_second_day : ℕ := 500
def people_first_day : ℕ := 2 * people_second_day

-- Theorem statement
theorem total_people_counted : people_first_day + people_second_day = 1500 := 
by 
  sorry

end total_people_counted_l234_234999


namespace arithmetic_mean_x_is_16_point_4_l234_234952

theorem arithmetic_mean_x_is_16_point_4 {x : ℝ}
  (h : (x + 10 + 17 + 2 * x + 15 + 2 * x + 6) / 5 = 26):
  x = 16.4 := 
sorry

end arithmetic_mean_x_is_16_point_4_l234_234952


namespace father_children_age_l234_234786

theorem father_children_age (F C n : Nat) (h1 : F = C) (h2 : F = 75) (h3 : C + 5 * n = 2 * (F + n)) : 
  n = 25 :=
by
  sorry

end father_children_age_l234_234786


namespace arithmetic_sequence_a6_l234_234293

theorem arithmetic_sequence_a6 (a : ℕ → ℕ)
  (h_arith_seq : ∀ n, ∃ d, a (n+1) = a n + d)
  (h_sum : a 4 + a 8 = 16) : a 6 = 8 :=
sorry

end arithmetic_sequence_a6_l234_234293


namespace range_of_a_l234_234116

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = Real.exp x) :
  (∀ x : ℝ, f x ≥ Real.exp x + a) ↔ a ≤ 0 :=
by
  sorry

end range_of_a_l234_234116


namespace farm_field_proof_l234_234688

section FarmField

variables 
  (planned_rate daily_rate : ℕ) -- planned_rate is 260 hectares/day, daily_rate is 85 hectares/day 
  (extra_days remaining_hectares : ℕ) -- extra_days is 2, remaining_hectares is 40
  (max_hours_per_day : ℕ) -- max_hours_per_day is 12

-- Definitions for soils
variables
  (A_percent B_percent C_percent : ℚ) (A_hours B_hours C_hours : ℕ)
  -- A_percent is 0.4, B_percent is 0.3, C_percent is 0.3
  -- A_hours is 4, B_hours is 6, C_hours is 3

-- Given conditions
axiom planned_rate_eq : planned_rate = 260
axiom daily_rate_eq : daily_rate = 85
axiom extra_days_eq : extra_days = 2
axiom remaining_hectares_eq : remaining_hectares = 40
axiom max_hours_per_day_eq : max_hours_per_day = 12

axiom A_percent_eq : A_percent = 0.4
axiom B_percent_eq : B_percent = 0.3
axiom C_percent_eq : C_percent = 0.3

axiom A_hours_eq : A_hours = 4
axiom B_hours_eq : B_hours = 6
axiom C_hours_eq : C_hours = 3

-- Theorem stating the problem
theorem farm_field_proof :
  ∃ (total_area initial_days : ℕ),
    total_area = 340 ∧ initial_days = 2 :=
by
  sorry

end FarmField

end farm_field_proof_l234_234688


namespace opposite_of_two_l234_234509

def opposite (n : ℤ) : ℤ := -n

theorem opposite_of_two : opposite 2 = -2 :=
by
  -- proof skipped
  sorry

end opposite_of_two_l234_234509


namespace andrey_boris_denis_eat_candies_l234_234557

def andrey_boris_condition (a b : ℕ) : Prop :=
  a = 4 ∧ b = 3

def andrey_denis_condition (a d : ℕ) : Prop :=
  a = 6 ∧ d = 7

def total_candies_condition (total : ℕ) : Prop :=
  total = 70

theorem andrey_boris_denis_eat_candies :
  ∃ (a b d : ℕ), andrey_boris_condition a b ∧ andrey_denis_condition a d ∧ 
                  (total_candies_condition (2 * (12 + 9 + 14)) ∧ 
                   2 * 12 = 24 ∧ 2 * 9 = 18 ∧ 2 * 14 = 28) →
                  (a = 24 ∧ b = 18 ∧ d = 28) :=
by
  sorry

end andrey_boris_denis_eat_candies_l234_234557


namespace shapeB_is_symmetric_to_original_l234_234676

-- Assume a simple type to represent our shapes
inductive Shape
| shapeA
| shapeB
| shapeC
| shapeD
| shapeE
| originalShape

-- Define the symmetry condition
def is_symmetric (s1 s2 : Shape) : Prop := sorry  -- this would be the condition to check symmetry

-- The theorem to prove that shapeB is symmetric to the original shape
theorem shapeB_is_symmetric_to_original :
  is_symmetric Shape.shapeB Shape.originalShape :=
sorry

end shapeB_is_symmetric_to_original_l234_234676


namespace at_least_one_is_one_l234_234930

theorem at_least_one_is_one (a b c : ℝ) 
  (h1 : a + b + c = (1 / a) + (1 / b) + (1 / c)) 
  (h2 : a * b * c = 1) : a = 1 ∨ b = 1 ∨ c = 1 := 
by 
  sorry

end at_least_one_is_one_l234_234930


namespace david_started_with_15_samsung_phones_l234_234082

-- Definitions
def SamsungPhonesAtEnd : ℕ := 10 -- S_e
def IPhonesAtEnd : ℕ := 5 -- I_e
def SamsungPhonesThrownOut : ℕ := 2 -- S_d
def IPhonesThrownOut : ℕ := 1 -- I_d
def TotalPhonesSold : ℕ := 4 -- C

-- Number of iPhones sold
def IPhonesSold : ℕ := IPhonesThrownOut

-- Assume: The remaining phones sold are Samsung phones
def SamsungPhonesSold : ℕ := TotalPhonesSold - IPhonesSold

-- Calculate the number of Samsung phones David started the day with
def SamsungPhonesAtStart : ℕ := SamsungPhonesAtEnd + SamsungPhonesThrownOut + SamsungPhonesSold

-- Statement
theorem david_started_with_15_samsung_phones : SamsungPhonesAtStart = 15 := by
  sorry

end david_started_with_15_samsung_phones_l234_234082


namespace calculate_expression_l234_234669

theorem calculate_expression : (50 - (5020 - 520) + (5020 - (520 - 50))) = 100 := 
by
  sorry

end calculate_expression_l234_234669


namespace length_of_EF_l234_234468

theorem length_of_EF (AB BC : ℝ) (DE DF : ℝ) (Area_ABC : ℝ) (Area_DEF : ℝ) (EF : ℝ) 
  (h₁ : AB = 10) (h₂ : BC = 15) (h₃ : DE = DF) (h₄ : Area_DEF = (1/3) * Area_ABC) 
  (h₅ : Area_ABC = AB * BC) (h₆ : Area_DEF = (1/2) * (DE * DF)) : 
  EF = 10 * Real.sqrt 2 := 
by 
  sorry

end length_of_EF_l234_234468


namespace cube_root_of_nine_irrational_l234_234216

theorem cube_root_of_nine_irrational : ¬ ∃ (r : ℚ), r^3 = 9 :=
by sorry

end cube_root_of_nine_irrational_l234_234216


namespace part1_inequality_part2_range_of_a_l234_234441

-- Part (1)
theorem part1_inequality (x : ℝ) : 
  let a := 1
  let f := λ x, |x - 1| + |x + 3|
  in (f x ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) : 
  let f := λ x, |x - a| + |x + 3|
  in (∀ x, f x > -a ↔ a > -3/2) :=
sorry

end part1_inequality_part2_range_of_a_l234_234441


namespace john_catch_train_probability_l234_234926

noncomputable def train_wait_probability : ℝ := 5 / 18

theorem john_catch_train_probability :
  let t_train := uniform 0 60 in
  let t_john := uniform 0 60 in
  ∃ p : ℝ,
    p = P(t_john ∈ interval t_train (t_train + 20)) ∧ p = train_wait_probability :=
sorry

end john_catch_train_probability_l234_234926


namespace comparison_of_a_and_c_l234_234508

variable {α : Type _} [LinearOrderedField α]

theorem comparison_of_a_and_c (a b c : α) (h1 : a > b) (h2 : (a - b) * (b - c) * (c - a) > 0) : a > c :=
sorry

end comparison_of_a_and_c_l234_234508


namespace calculate_expression_l234_234228

theorem calculate_expression :
  (-2)^(4^2) + 2^(3^2) = 66048 := by sorry

end calculate_expression_l234_234228


namespace sum_series_eq_seven_twelve_l234_234395

noncomputable def sum_series : ℝ :=
  ∑' n : ℕ, if n > 0 then (3 * (n:ℝ)^2 + 2 * (n:ℝ) + 1) / ((n:ℝ) * (n + 1) * (n + 2) * (n + 3)) else 0

theorem sum_series_eq_seven_twelve : sum_series = 7 / 12 :=
by
  sorry

end sum_series_eq_seven_twelve_l234_234395


namespace complete_square_expression_l234_234461

theorem complete_square_expression :
  ∃ (a h k : ℝ), (∀ x : ℝ, 2 * x^2 + 8 * x + 6 = a * (x - h)^2 + k) ∧ (a + h + k = -2) :=
by
  sorry

end complete_square_expression_l234_234461


namespace total_games_played_l234_234188

noncomputable def win_ratio : ℝ := 5.5
noncomputable def lose_ratio : ℝ := 4.5
noncomputable def tie_ratio : ℝ := 2.5
noncomputable def rained_out_ratio : ℝ := 1
noncomputable def higher_league_ratio : ℝ := 3.5
noncomputable def lost_games : ℝ := 13.5

theorem total_games_played :
  let total_parts := win_ratio + lose_ratio + tie_ratio + rained_out_ratio + higher_league_ratio
  let games_per_part := lost_games / lose_ratio
  total_parts * games_per_part = 51 :=
by
  let total_parts := win_ratio + lose_ratio + tie_ratio + rained_out_ratio + higher_league_ratio
  let games_per_part := lost_games / lose_ratio
  have : total_parts * games_per_part = 51 := sorry
  exact this

end total_games_played_l234_234188


namespace units_digit_of_24_pow_4_add_42_pow_4_l234_234035

theorem units_digit_of_24_pow_4_add_42_pow_4 : 
  (24^4 + 42^4) % 10 = 2 := 
by sorry

end units_digit_of_24_pow_4_add_42_pow_4_l234_234035


namespace problem1_problem2_l234_234048

-- Problem 1: Prove the expression
theorem problem1 (a b : ℝ) : 
  2 * a * (a - 2 * b) - (2 * a - b) ^ 2 = -2 * a ^ 2 - b ^ 2 := 
sorry

-- Problem 2: Prove the solution to the equation
theorem problem2 (x : ℝ) (h : (x - 1) ^ 3 - 3 = 3 / 8) : 
  x = 5 / 2 := 
sorry

end problem1_problem2_l234_234048


namespace total_cost_other_goods_l234_234390

/-- The total cost of the goods other than tuna and water -/
theorem total_cost_other_goods :
  let cost_tuna := 5 * 2
  let cost_water := 4 * 1.5
  let total_paid := 56
  let cost_other := total_paid - cost_tuna - cost_water
  cost_other = 40 :=
by
  let cost_tuna := 5 * 2
  let cost_water := 4 * 1.5
  let total_paid := 56
  let cost_other := total_paid - cost_tuna - cost_water
  show cost_other = 40
  sorry

end total_cost_other_goods_l234_234390


namespace james_found_bills_l234_234295

def initial_money : ℝ := 75
def final_money : ℝ := 135
def bill_value : ℝ := 20

theorem james_found_bills :
  (final_money - initial_money) / bill_value = 3 :=
by
  sorry

end james_found_bills_l234_234295


namespace find_d_l234_234100

variable (x y d : ℤ)

-- Condition from the problem
axiom condition1 : (7 * x + 4 * y) / (x - 2 * y) = 13

-- The main proof goal
theorem find_d : x = 5 * y → x / (2 * y) = d / 2 → d = 5 :=
by
  intro h1 h2
  -- proof goes here
  sorry

end find_d_l234_234100


namespace calculate_expression_l234_234701

theorem calculate_expression : 8 / 2 - 3 - 12 + 3 * (5^2 - 4) = 52 := 
by
  sorry

end calculate_expression_l234_234701


namespace intersection_of_A_and_B_l234_234916

section intersection_proof

-- Definitions of the sets A and B
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x + 1 > 0}

-- Statement of the theorem
theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := 
by {
  sorry
}

end intersection_proof

end intersection_of_A_and_B_l234_234916


namespace shopkeeper_gain_percent_l234_234991

theorem shopkeeper_gain_percent
    (SP₁ SP₂ CP : ℝ)
    (h₁ : SP₁ = 187)
    (h₂ : SP₂ = 264)
    (h₃ : SP₁ = 0.85 * CP) :
    ((SP₂ - CP) / CP) * 100 = 20 := by 
  sorry

end shopkeeper_gain_percent_l234_234991


namespace bus_length_is_200_l234_234535

def length_of_bus (distance_km distance_secs passing_secs : ℕ) : ℕ :=
  let speed_kms := distance_km / distance_secs
  let speed_ms := speed_kms * 1000
  speed_ms * passing_secs

theorem bus_length_is_200 
  (distance_km : ℕ) (distance_secs : ℕ) (passing_secs : ℕ)
  (h1 : distance_km = 12) (h2 : distance_secs = 300) (h3 : passing_secs = 5) : 
  length_of_bus distance_km distance_secs passing_secs = 200 := 
  by
    sorry

end bus_length_is_200_l234_234535


namespace candy_eating_l234_234560

-- Definitions based on the conditions
def candies (andrey_rate boris_rate denis_rate : ℕ) (andrey_candies boris_candies denis_candies total_candies : ℕ) : Prop :=
  andrey_rate = 4 ∧ boris_rate = 3 ∧ denis_rate = 7 ∧ andrey_candies = 24 ∧ boris_candies = 18 ∧ denis_candies = 28 ∧
  total_candies = andrey_candies + boris_candies + denis_candies

-- Problem statement
theorem candy_eating :
  ∃ (a b d : ℕ), 
    candies 4 3 7 a b d 70 :=
sorry

end candy_eating_l234_234560


namespace sum_of_integers_between_neg20_5_and_10_5_l234_234358

theorem sum_of_integers_between_neg20_5_and_10_5 :
  let a := -20
  let l := 10
  let n := (l - a) / 1 + 1
  let S := n / 2 * (a + l)
  S = -155 := by
{
  sorry
}

end sum_of_integers_between_neg20_5_and_10_5_l234_234358


namespace inequality_solution_l234_234328

theorem inequality_solution (x : ℝ) : 2 * (3 * x - 2) > x + 1 ↔ x > 1 := by
  sorry

end inequality_solution_l234_234328


namespace factor_problem_l234_234649

theorem factor_problem (C D : ℤ) (h1 : 16 * x^2 - 88 * x + 63 = (C * x - 21) * (D * x - 3)) (h2 : C * D + C = 21) : C = 7 ∧ D = 2 :=
by 
  sorry

end factor_problem_l234_234649


namespace product_of_roots_quadratic_l234_234248

noncomputable def product_of_roots (a b c : ℝ) : ℝ :=
  let x1 := (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let x2 := (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  x1 * x2

theorem product_of_roots_quadratic :
  (product_of_roots 1 3 (-5)) = -5 :=
by
  sorry

end product_of_roots_quadratic_l234_234248


namespace part_a_part_b_part_c_part_d_l234_234396

-- (a)
theorem part_a : ∃ x y : ℤ, x > 0 ∧ y > 0 ∧ x ≤ 5 ∧ x^2 - 2 * y^2 = 1 :=
by
  -- proof here
  sorry

-- (b)
theorem part_b : ∃ u v : ℤ, (3 + 2 * Real.sqrt 2)^2 = u + v * Real.sqrt 2 ∧ u^2 - 2 * v^2 = 1 :=
by
  -- proof here
  sorry

-- (c)
theorem part_c : ∀ a b c d : ℤ, a^2 - 2 * b^2 = 1 → (a + b * Real.sqrt 2) * (3 + 2 * Real.sqrt 2) = c + d * Real.sqrt 2
                  → c^2 - 2 * d^2 = 1 :=
by
  -- proof here
  sorry

-- (d)
theorem part_d : ∃ x y : ℤ, y > 100 ∧ x^2 - 2 * y^2 = 1 :=
by
  -- proof here
  sorry

end part_a_part_b_part_c_part_d_l234_234396


namespace candy_eating_l234_234559

-- Definitions based on the conditions
def candies (andrey_rate boris_rate denis_rate : ℕ) (andrey_candies boris_candies denis_candies total_candies : ℕ) : Prop :=
  andrey_rate = 4 ∧ boris_rate = 3 ∧ denis_rate = 7 ∧ andrey_candies = 24 ∧ boris_candies = 18 ∧ denis_candies = 28 ∧
  total_candies = andrey_candies + boris_candies + denis_candies

-- Problem statement
theorem candy_eating :
  ∃ (a b d : ℕ), 
    candies 4 3 7 a b d 70 :=
sorry

end candy_eating_l234_234559


namespace b1f_hex_to_dec_l234_234235

/-- 
  Convert the given hexadecimal digit to its corresponding decimal value.
  -/
def hex_to_dec (c : Char) : Nat :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | '0' => 0
  | '1' => 1
  | '2' => 2
  | '3' => 3
  | '4' => 4
  | '5' => 5
  | '6' => 6
  | '7' => 7
  | '8' => 8
  | '9' => 9
  | _ => 0

/-- 
  Convert a hexadecimal string to a decimal number.
  -/
def hex_string_to_dec (s : String) : Nat :=
  s.foldl (λ acc c => acc * 16 + hex_to_dec c) 0

theorem b1f_hex_to_dec : hex_string_to_dec "B1F" = 2847 :=
by
  sorry

end b1f_hex_to_dec_l234_234235


namespace sum_of_first_two_digits_of_repeating_decimal_l234_234512

theorem sum_of_first_two_digits_of_repeating_decimal (c d : ℕ) (h : (c, d) = (3, 5)) : c + d = 8 :=
by 
  sorry

end sum_of_first_two_digits_of_repeating_decimal_l234_234512


namespace profit_correct_l234_234322

-- Conditions
def initial_outlay : ℕ := 10000
def cost_per_set : ℕ := 20
def selling_price_per_set : ℕ := 50
def sets : ℕ := 500

-- Definitions used in the problem
def manufacturing_cost : ℕ := initial_outlay + (sets * cost_per_set)
def revenue : ℕ := sets * selling_price_per_set
def profit : ℕ := revenue - manufacturing_cost

-- The theorem statement
theorem profit_correct : profit = 5000 := by
  sorry

end profit_correct_l234_234322


namespace total_amount_received_l234_234769

theorem total_amount_received (h1 : 12 = 12)
                              (h2 : 10 = 10)
                              (h3 : 8 = 8)
                              (h4 : 14 = 14)
                              (rate : 15 = 15) :
  (3 * (12 + 10 + 8 + 14) * 15) = 1980 :=
by sorry

end total_amount_received_l234_234769


namespace parallelogram_and_triangle_area_eq_l234_234211

noncomputable def parallelogram_area (AB AD : ℝ) : ℝ :=
  AB * AD

noncomputable def right_triangle_area (DG FG : ℝ) : ℝ :=
  (DG * FG) / 2

variables (AB AD DG FG : ℝ)
variables (angleDFG : ℝ)

def parallelogram_ABCD (AB : ℝ) (AD : ℝ) (angleDFG : ℝ) (DG : ℝ) : Prop :=
  parallelogram_area AB AD = 24 ∧ angleDFG = 90 ∧ DG = 6

theorem parallelogram_and_triangle_area_eq (h1 : parallelogram_ABCD AB AD angleDFG DG)
    (h2 : parallelogram_area AB AD = right_triangle_area DG FG) : FG = 8 :=
by
  sorry

end parallelogram_and_triangle_area_eq_l234_234211


namespace problem_statement_l234_234605

theorem problem_statement (x y : ℤ) (k : ℤ) (h : 4 * x - y = 3 * k) : 9 ∣ 4 * x^2 + 7 * x * y - 2 * y^2 :=
by
  sorry

end problem_statement_l234_234605


namespace kamal_chemistry_marks_l234_234296

variables (english math physics biology average total numSubjects : ℕ)

theorem kamal_chemistry_marks 
  (marks_in_english : english = 66)
  (marks_in_math : math = 65)
  (marks_in_physics : physics = 77)
  (marks_in_biology : biology = 75)
  (avg_marks : average = 69)
  (number_of_subjects : numSubjects = 5)
  (total_marks_known : total = 283) :
  ∃ chemistry : ℕ, chemistry = 62 := 
by 
  sorry

end kamal_chemistry_marks_l234_234296


namespace meal_center_adults_l234_234987

theorem meal_center_adults (cans : ℕ) (children_served : ℕ) (adults_served : ℕ) (total_children : ℕ) 
  (initial_cans : cans = 10) 
  (children_per_can : children_served = 7) 
  (adults_per_can : adults_served = 4) 
  (children_to_feed : total_children = 21) : 
  (cans - (total_children / children_served)) * adults_served = 28 := by
  have h1: 3 = total_children / children_served := by
    sorry
  have h2: 7 = cans - 3 := by
    sorry
  have h3: 28 = 7 * adults_served := by
    sorry
  have h4: adults_served = 4 := by
    sorry
  sorry

end meal_center_adults_l234_234987


namespace fraction_of_shaded_area_l234_234333

theorem fraction_of_shaded_area
  (total_smaller_rectangles : ℕ)
  (shaded_smaller_rectangles : ℕ)
  (h1 : total_smaller_rectangles = 18)
  (h2 : shaded_smaller_rectangles = 4) :
  (shaded_smaller_rectangles : ℚ) / total_smaller_rectangles = 1 / 4 := 
sorry

end fraction_of_shaded_area_l234_234333


namespace cost_comparison_cost_effectiveness_47_l234_234068

section
variable (x : ℕ)

-- Conditions
def price_teapot : ℕ := 25
def price_teacup : ℕ := 5
def quantity_teapots : ℕ := 4
def discount_scheme_2 : ℝ := 0.94

-- Total cost for Scheme 1
def cost_scheme_1 (x : ℕ) : ℕ :=
  (quantity_teapots * price_teapot) + (price_teacup * (x - quantity_teapots))

-- Total cost for Scheme 2
def cost_scheme_2 (x : ℕ) : ℝ :=
  (quantity_teapots * price_teapot + price_teacup * x : ℝ) * discount_scheme_2

-- The proof problem
theorem cost_comparison (x : ℕ) (h : x ≥ 4) :
  cost_scheme_1 x = 5 * x + 80 ∧ cost_scheme_2 x = 4.7 * x + 94 :=
sorry

-- When x = 47
theorem cost_effectiveness_47 : cost_scheme_2 47 < cost_scheme_1 47 :=
sorry

end

end cost_comparison_cost_effectiveness_47_l234_234068


namespace part1_solution_part2_solution_l234_234437

-- Proof problem for Part (1)
theorem part1_solution (x : ℝ) (a : ℝ) (h : a = 1) : 
  (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

-- Proof problem for Part (2)
theorem part2_solution (a : ℝ) : 
  (∀ x : ℝ, (|x - a| + |x + 3|) > -a) ↔ a > -3 / 2 :=
by
  sorry

end part1_solution_part2_solution_l234_234437


namespace gcd_power_sub_one_l234_234806

theorem gcd_power_sub_one (a b : ℕ) (h1 : b = a + 30) : 
  Nat.gcd (2^a - 1) (2^b - 1) = 2^30 - 1 := 
by 
  sorry

end gcd_power_sub_one_l234_234806


namespace classify_abc_l234_234004

theorem classify_abc (a b c : ℝ) 
  (h1 : (a > 0 ∨ a < 0 ∨ a = 0) ∧ (b > 0 ∨ b < 0 ∨ b = 0) ∧ (c > 0 ∨ c < 0 ∨ c = 0))
  (h2 : (a > 0 ∧ b < 0 ∧ c = 0) ∨ (a > 0 ∧ b = 0 ∧ c < 0) ∨ (a < 0 ∧ b > 0 ∧ c = 0) ∨
        (a < 0 ∧ b = 0 ∧ c > 0) ∨ (a = 0 ∧ b > 0 ∧ c < 0) ∨ (a = 0 ∧ b < 0 ∧ c > 0))
  (h3 : |a| = b^2 * (b - c)) : 
  a < 0 ∧ b > 0 ∧ c = 0 :=
by 
  sorry

end classify_abc_l234_234004


namespace ab_greater_than_a_plus_b_l234_234127

theorem ab_greater_than_a_plus_b (a b : ℝ) (ha : a ≥ 2) (hb : b > 2) : a * b > a + b :=
by
  sorry

end ab_greater_than_a_plus_b_l234_234127


namespace parabola_equation_l234_234083

noncomputable def parabola_vertex_form (x y a : ℝ) : Prop := y = a * (x - 3)^2 + 5

noncomputable def parabola_standard_form (x y : ℝ) : Prop := y = -3 * x^2 + 18 * x - 22

theorem parabola_equation (a : ℝ) (h_vertex : parabola_vertex_form 3 5 a) (h_point : parabola_vertex_form 2 2 a) :
  ∃ x y, parabola_standard_form x y :=
by
  sorry

end parabola_equation_l234_234083


namespace fraction_sum_ratio_l234_234795

theorem fraction_sum_ratio
    (a b c : ℝ) (m n : ℝ)
    (h1 : a = (b + c) / m)
    (h2 : b = (c + a) / n) :
    (m * n ≠ 1 → (a + b) / c = (m + n + 2) / (m * n - 1)) ∧ 
    (m = -1 ∧ n = -1 → (a + b) / c = -1) :=
by
    sorry

end fraction_sum_ratio_l234_234795


namespace isosceles_right_triangle_ratio_l234_234384

theorem isosceles_right_triangle_ratio {a : ℝ} (h_pos : 0 < a) :
  (a + 2 * a) / Real.sqrt (a^2 + a^2) = 3 * Real.sqrt 2 / 2 :=
sorry

end isosceles_right_triangle_ratio_l234_234384


namespace friendP_walks_23_km_l234_234519

noncomputable def friendP_distance (v : ℝ) : ℝ :=
  let trail_length := 43
  let speedP := 1.15 * v
  let speedQ := v
  let dQ := trail_length - 23
  let timeP := 23 / speedP
  let timeQ := dQ / speedQ
  if timeP = timeQ then 23 else 0  -- Ensuring that both reach at the same time.

theorem friendP_walks_23_km (v : ℝ) : 
  friendP_distance v = 23 :=
by
  sorry

end friendP_walks_23_km_l234_234519


namespace find_two_digit_numbers_l234_234351

def first_two_digit_number (x y : ℕ) : ℕ := 10 * x + y
def second_two_digit_number (x y : ℕ) : ℕ := 10 * (x + 5) + y

theorem find_two_digit_numbers :
  ∃ (x_2 y : ℕ), 
  (first_two_digit_number x_2 y = x_2^2 + x_2 * y + y^2) ∧ 
  (second_two_digit_number x_2 y = (x_2 + 5)^2 + (x_2 + 5) * y + y^2) ∧ 
  (second_two_digit_number x_2 y - first_two_digit_number x_2 y = 50) ∧ 
  (y = 1 ∨ y = 3) := 
sorry

end find_two_digit_numbers_l234_234351


namespace blocks_combination_count_l234_234840

-- Definition statements reflecting all conditions in the problem
def select_4_blocks_combinations : ℕ :=
  let choose (n k : ℕ) := Nat.choose n k
  let factorial (n : ℕ) := Nat.factorial n
  choose 6 4 * choose 6 4 * factorial 4

-- Theorem stating the result we want to prove
theorem blocks_combination_count : select_4_blocks_combinations = 5400 :=
by
  -- We will provide the proof steps here
  sorry

end blocks_combination_count_l234_234840


namespace remainder_when_divided_by_100_l234_234685

/-- A basketball team has 15 available players. A fixed set of 5 players starts the game, while the other 
10 are available as substitutes. During the game, the coach may make up to 4 substitutions. No player 
removed from the game may reenter, and no two substitutions can happen simultaneously. The players 
involved and the order of substitutions matter. -/
def num_substitution_sequences : ℕ :=
  let a_0 := 1
  let a_1 := 5 * 10
  let a_2 := a_1 * 4 * 9
  let a_3 := a_2 * 3 * 8
  let a_4 := a_3 * 2 * 7
  a_0 + a_1 + a_2 + a_3 + a_4

theorem remainder_when_divided_by_100 : num_substitution_sequences % 100 = 51 :=
by
  -- proof to be written
  sorry

end remainder_when_divided_by_100_l234_234685


namespace max_value_of_quadratic_expression_l234_234029

theorem max_value_of_quadratic_expression (s : ℝ) : ∃ x : ℝ, -3 * s^2 + 24 * s - 8 ≤ x ∧ x = 40 :=
sorry

end max_value_of_quadratic_expression_l234_234029


namespace lindsey_final_money_l234_234160

-- Define the savings in each month
def save_sep := 50
def save_oct := 37
def save_nov := 11

-- Total savings over the three months
def total_savings := save_sep + save_oct + save_nov

-- Condition for Mom's contribution
def mom_contribution := if total_savings > 75 then 25 else 0

-- Total savings including mom's contribution
def total_with_mom := total_savings + mom_contribution

-- Amount spent on the video game
def spent := 87

-- Final amount left
def final_amount := total_with_mom - spent

-- Proof statement
theorem lindsey_final_money : final_amount = 36 := by
  sorry

end lindsey_final_money_l234_234160


namespace max_value_90_l234_234339

noncomputable def max_value_expression (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_90 (a b c d : ℝ) (h₁ : -4.5 ≤ a) (h₂ : a ≤ 4.5)
                                   (h₃ : -4.5 ≤ b) (h₄ : b ≤ 4.5)
                                   (h₅ : -4.5 ≤ c) (h₆ : c ≤ 4.5)
                                   (h₇ : -4.5 ≤ d) (h₈ : d ≤ 4.5) :
  max_value_expression a b c d ≤ 90 :=
sorry

end max_value_90_l234_234339


namespace average_salary_for_company_l234_234984

theorem average_salary_for_company
    (number_of_managers : Nat)
    (number_of_associates : Nat)
    (average_salary_managers : Nat)
    (average_salary_associates : Nat)
    (hnum_managers : number_of_managers = 15)
    (hnum_associates : number_of_associates = 75)
    (has_managers : average_salary_managers = 90000)
    (has_associates : average_salary_associates = 30000) : 
    (number_of_managers * average_salary_managers + number_of_associates * average_salary_associates) / 
    (number_of_managers + number_of_associates) = 40000 := 
    by
    sorry

end average_salary_for_company_l234_234984


namespace problem_relation_l234_234853

-- Definitions indicating relationships.
def related₁ : Prop := ∀ (s : ℝ), (s ≥ 0) → (∃ a p : ℝ, a = s^2 ∧ p = 4 * s)
def related₂ : Prop := ∀ (d t : ℝ), (t > 0) → (∃ v : ℝ, d = v * t)
def related₃ : Prop := ∃ (h w : ℝ) (f : ℝ → ℝ), w = f h
def related₄ : Prop := ∀ (h : ℝ) (v : ℝ), False

-- The theorem stating that A, B, and C are related.
theorem problem_relation : 
  related₁ ∧ related₂ ∧ related₃ ∧ ¬ related₄ :=
by sorry

end problem_relation_l234_234853


namespace functions_are_equal_l234_234041

-- Define the functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := (x^4)^(1/4)

-- Statement to be proven
theorem functions_are_equal : ∀ x : ℝ, f x = g x := by
  sorry

end functions_are_equal_l234_234041


namespace product_pos_implies_pos_or_neg_pos_pair_implies_product_pos_product_pos_necessary_for_pos_product_pos_not_sufficient_for_pos_l234_234251

variable {x y : ℝ}

-- The formal statement in Lean
theorem product_pos_implies_pos_or_neg (h : x * y > 0) : (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) :=
sorry

theorem pos_pair_implies_product_pos (hx : x > 0) (hy : y > 0) : x * y > 0 :=
sorry

theorem product_pos_necessary_for_pos (h : x > 0 ∧ y > 0) : x * y > 0 :=
pos_pair_implies_product_pos h.1 h.2

theorem product_pos_not_sufficient_for_pos (h : x * y > 0) : ¬ (x > 0 ∧ y > 0) :=
sorry

end product_pos_implies_pos_or_neg_pos_pair_implies_product_pos_product_pos_necessary_for_pos_product_pos_not_sufficient_for_pos_l234_234251


namespace max_ways_to_ascend_descend_l234_234661

theorem max_ways_to_ascend_descend :
  let east_paths := 2
  let west_paths := 1
  let south_paths := 3
  let north_paths := 4

  let descend_from_east := west_paths + south_paths + north_paths
  let descend_from_west := east_paths + south_paths + north_paths
  let descend_from_south := east_paths + west_paths + north_paths
  let descend_from_north := east_paths + west_paths + south_paths

  let ways_from_east := east_paths * descend_from_east
  let ways_from_west := west_paths * descend_from_west
  let ways_from_south := south_paths * descend_from_south
  let ways_from_north := north_paths * descend_from_north

  max ways_from_east (max ways_from_west (max ways_from_south ways_from_north)) = 24 := 
by
  -- Insert the proof here
  sorry

end max_ways_to_ascend_descend_l234_234661


namespace rotation_150_positions_l234_234996

/-
Define the initial positions and the shapes involved.
-/
noncomputable def initial_positions := ["A", "B", "C", "D"]
noncomputable def initial_order := ["triangle", "smaller_circle", "square", "pentagon"]

def rotate_clockwise_150 (pos : List String) : List String :=
  -- 1 full position and two-thirds into the next position
  [pos.get! 1, pos.get! 2, pos.get! 3, pos.get! 0]

theorem rotation_150_positions :
  rotate_clockwise_150 initial_positions = ["Triangle between B and C", 
                                            "Smaller circle between C and D", 
                                            "Square between D and A", 
                                            "Pentagon between A and B"] :=
by sorry

end rotation_150_positions_l234_234996


namespace four_cards_probability_l234_234275

theorem four_cards_probability :
  let deck_size := 52
  let suits_size := 13
  ∀ (C D H S : ℕ), 
  C = 1 ∧ D = 13 ∧ H = 13 ∧ S = 13 →
  (C / deck_size) *
  (D / (deck_size - 1)) *
  (H / (deck_size - 2)) *
  (S / (deck_size - 3)) = (2197 / 499800) :=
by
  intros deck_size suits_size C D H S h
  sorry

end four_cards_probability_l234_234275


namespace compute_difference_l234_234933

def distinct_solutions (p q : ℝ) : Prop :=
  (p ≠ q) ∧ (∃ (x : ℝ), (x = p ∨ x = q) ∧ (x-3)*(x+3) = 21*x - 63) ∧
  (p > q)

theorem compute_difference (p q : ℝ) (h : distinct_solutions p q) : p - q = 15 :=
by
  sorry

end compute_difference_l234_234933


namespace range_of_a_l234_234889

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 - a * x - 2 * a^2 > -9) : -2 < a ∧ a < 2 := 
sorry

end range_of_a_l234_234889


namespace compute_xy_l234_234520

variable (x y : ℝ)
variable (h1 : x - y = 6)
variable (h2 : x^3 - y^3 = 108)

theorem compute_xy : x * y = 0 := by
  sorry

end compute_xy_l234_234520


namespace profitability_when_x_gt_94_daily_profit_when_x_le_94_max_profit_occurs_at_84_l234_234825

theorem profitability_when_x_gt_94 (A : ℕ) (x : ℕ) (hx : x > 94) : 
  1/3 * x * A - (2/3 * x * (A / 2)) = 0 := 
sorry

theorem daily_profit_when_x_le_94 (A : ℕ) (x : ℕ) (hx : 1 ≤ x ∧ x ≤ 94) : 
  ∃ T : ℕ, T = (x - 3 * x / (2 * (96 - x))) * A := 
sorry

theorem max_profit_occurs_at_84 (A : ℕ) : 
  ∃ x : ℕ, 1 ≤ x ∧ x ≤ 94 ∧ 
  (∀ y : ℕ, 1 ≤ y ∧ y ≤ 94 → 
    (y - 3 * y / (2 * (96 - y))) * A ≤ (84 - 3 * 84 / (2 * (96 - 84))) * A) := 
sorry

end profitability_when_x_gt_94_daily_profit_when_x_le_94_max_profit_occurs_at_84_l234_234825


namespace maximum_vertices_no_rectangle_l234_234526

theorem maximum_vertices_no_rectangle (n : ℕ) (h : n = 2016) :
  ∃ m : ℕ, m = 1009 ∧
  ∀ (V : Finset (Fin n)), V.card = m →
  ∀ (v1 v2 v3 v4 : Fin n), {v1, v2, v3, v4}.subset V →
  ¬ (v1.val + v3.val = v2.val + v4.val ∧ v1.val ≠ v2.val ∧ v1.val ≠ v3.val ∧ v1.val ≠ v4.val ∧ v2.val ≠ v3.val ∧ v2.val ≠ v4.val ∧ v3.val ≠ v4.val) :=
sorry

end maximum_vertices_no_rectangle_l234_234526


namespace total_sides_tom_tim_l234_234966

def sides_per_die : Nat := 6

def tom_dice_count : Nat := 4
def tim_dice_count : Nat := 4

theorem total_sides_tom_tim : tom_dice_count * sides_per_die + tim_dice_count * sides_per_die = 48 := by
  sorry

end total_sides_tom_tim_l234_234966


namespace initial_fish_count_l234_234963

theorem initial_fish_count (F T : ℕ) 
  (h1 : T = 3 * F)
  (h2 : T / 2 = (F - 7) + 32) : F = 50 :=
by
  sorry

end initial_fish_count_l234_234963


namespace garden_square_char_l234_234787

theorem garden_square_char (s q p x : ℕ) (h1 : p = 28) (h2 : q = p + x) (h3 : q = s^2) (h4 : p = 4 * s) : x = 21 :=
by
  sorry

end garden_square_char_l234_234787


namespace union_of_A_B_l234_234121

def A : Set ℝ := { x | x^2 - x - 2 ≤ 0 }

def B : Set ℝ := { x | 1 < x ∧ x ≤ 3 }

theorem union_of_A_B : A ∪ B = { x | -1 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end union_of_A_B_l234_234121


namespace felix_lift_calculation_l234_234876

variables (F B : ℝ)

-- Felix can lift off the ground 1.5 times more than he weighs
def felixLift := 1.5 * F

-- Felix's brother weighs twice as much as Felix
def brotherWeight := 2 * F

-- Felix's brother can lift three times his weight off the ground
def brotherLift := 3 * B

-- Felix's brother can lift 600 pounds
def brotherLiftCondition := brotherLift B = 600

theorem felix_lift_calculation (h1 : brotherLiftCondition) (h2 : brotherWeight B = 2 * F) : felixLift F = 150 :=
by
  sorry

end felix_lift_calculation_l234_234876


namespace candle_height_relation_l234_234518

variables (t : ℝ)

def height_candle_A (t : ℝ) := 12 - 2 * t
def height_candle_B (t : ℝ) := 9 - 2 * t

theorem candle_height_relation : 
  12 - 2 * (15 / 4) = 3 * (9 - 2 * (15 / 4)) :=
by
  sorry

end candle_height_relation_l234_234518


namespace eccentricity_of_ellipse_l234_234706

open Real

theorem eccentricity_of_ellipse (a b c : ℝ) 
  (h1 : a > b ∧ b > 0)
  (h2 : c^2 = a^2 - b^2)
  (x : ℝ)
  (h3 : 3 * x = 2 * a)
  (h4 : sqrt 3 * x = 2 * c) :
  c / a = sqrt 3 / 3 :=
by
  sorry

end eccentricity_of_ellipse_l234_234706


namespace percentage_chromium_first_alloy_l234_234291

theorem percentage_chromium_first_alloy
  (x : ℝ) (h : (x / 100) * 15 + (8 / 100) * 35 = (9.2 / 100) * 50) : x = 12 :=
sorry

end percentage_chromium_first_alloy_l234_234291


namespace quadratic_roots_two_l234_234944

theorem quadratic_roots_two (m : ℝ) :
  let a := 1
  let b := -m
  let c := m - 2
  let Δ := b^2 - 4 * a * c
  Δ > 0 :=
by
  let a := 1
  let b := -m
  let c := m - 2
  let Δ := b^2 - 4 * a * c
  sorry

end quadratic_roots_two_l234_234944


namespace ellipse_foci_y_axis_l234_234754

theorem ellipse_foci_y_axis (k : ℝ) (h_eq : ∀ x y : ℝ, x^2 + k * y^2 = 2)
  (h_foci : ∀ x y : ℝ, x^2 ≤ 2 ∧ k * y^2 ≤ 2) :
  0 < k ∧ k < 1 :=
  sorry

end ellipse_foci_y_axis_l234_234754


namespace triangle_third_side_one_third_perimeter_l234_234539

theorem triangle_third_side_one_third_perimeter
  (a b x y p c : ℝ)
  (h1 : x^2 - y^2 = a^2 - b^2)
  (h2 : p = (a + b + c) / 2)
  (h3 : x - y = 2 * (a - b)) :
  c = (a + b + c) / 3 := by
  sorry

end triangle_third_side_one_third_perimeter_l234_234539


namespace solve_inequality_l234_234784

theorem solve_inequality (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < 6) 
  (h3 : x ≠ 1) 
  (h4 : x ≠ 2) :
  (x ∈ (Set.Ioo 0 1 ∪ Set.Ioo 1 2 ∪ Set.Ioo 2 6)) → 
  ((x ∈ (Set.Ioo 0 1 ∪ Set.Ioo 1 2 ∪ Set.Icc 3 5))) :=
by 
  introv h
  sorry

end solve_inequality_l234_234784


namespace smallest_debt_exists_l234_234354

theorem smallest_debt_exists :
  ∃ (p g : ℤ), 50 = 200 * p + 150 * g := by
  sorry

end smallest_debt_exists_l234_234354


namespace no_solution_ineq_system_l234_234363

def inequality_system (x : ℝ) : Prop :=
  (x / 6 + 7 / 2 > (3 * x + 29) / 5) ∧
  (x + 9 / 2 > x / 8) ∧
  (11 / 3 - x / 6 < (34 - 3 * x) / 5)

theorem no_solution_ineq_system : ¬ ∃ x : ℝ, inequality_system x :=
  sorry

end no_solution_ineq_system_l234_234363


namespace solution_to_system_of_inequalities_l234_234817

variable {x y : ℝ}

theorem solution_to_system_of_inequalities :
  11 * (-1/3 : ℝ)^2 + 8 * (-1/3 : ℝ) * (2/3 : ℝ) + 8 * (2/3 : ℝ)^2 ≤ 3 ∧
  (-1/3 : ℝ) - 4 * (2/3 : ℝ) ≤ -3 :=
by
  sorry

end solution_to_system_of_inequalities_l234_234817


namespace sugar_amount_l234_234762

variables (S F B : ℝ)

-- Conditions
def condition1 : Prop := S / F = 5 / 2
def condition2 : Prop := F / B = 10 / 1
def condition3 : Prop := F / (B + 60) = 8 / 1

-- Theorem to prove
theorem sugar_amount (h1 : condition1 S F) (h2 : condition2 F B) (h3 : condition3 F B) : S = 6000 :=
sorry

end sugar_amount_l234_234762


namespace find_a_l234_234418

theorem find_a (a : ℝ) : 
  (a + 3)^2 = (a + 1)^2 + (a + 2)^2 → a = 2 := 
by
  intro h
  -- Proof should go here
  sorry

end find_a_l234_234418


namespace value_of_x_squared_plus_9y_squared_l234_234457

theorem value_of_x_squared_plus_9y_squared (x y : ℝ) (h1 : x - 3 * y = 3) (h2 : x * y = -9) : x^2 + 9 * y^2 = -45 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l234_234457


namespace sum_arithmetic_sequence_l234_234656

variable {α : Type*} [LinearOrderedField α]

def sum_first_n (a d : α) (n : ℕ) : α := 
  n * (2 * a + (n - 1) * d) / 2

theorem sum_arithmetic_sequence
    (a d : α) 
    (S_5 S_10 S_15 : α)
    (h1 : S_5 = 5 * (2 * a + 4 * d) / 2)
    (h2 : S_10 = 10 * (2 * a + 9 * d) / 2)
    (h3 : S_5 = 10)
    (h4 : S_10 = 50) : 
  S_15 = 15 * (2 * a + 14 * d) / 2 := 
sorry

end sum_arithmetic_sequence_l234_234656


namespace sum_of_first_15_terms_of_geometric_sequence_l234_234285

theorem sum_of_first_15_terms_of_geometric_sequence (a r : ℝ) 
  (h₁ : (a * (1 - r^5)) / (1 - r) = 10) 
  (h₂ : (a * (1 - r^10)) / (1 - r) = 50) : 
  (a * (1 - r^15)) / (1 - r) = 210 := 
by 
  sorry

end sum_of_first_15_terms_of_geometric_sequence_l234_234285


namespace find_sr_division_l234_234490

theorem find_sr_division (k : ℚ) (c r s : ℚ)
  (h_c : c = 10)
  (h_r : r = -3 / 10)
  (h_s : s = 191 / 10)
  (h_expr : 10 * k^2 - 6 * k + 20 = c * (k + r)^2 + s) :
  s / r = -191 / 3 :=
by
  sorry

end find_sr_division_l234_234490


namespace symmetric_line_eq_l234_234122

/-- 
Given two circles O: x^2 + y^2 = 4 and C: x^2 + y^2 + 4x - 4y + 4 = 0, 
prove the equation of the line l such that the two circles are symmetric 
with respect to line l is x - y + 2 = 0.
-/
theorem symmetric_line_eq {x y : ℝ} :
  (∀ x y : ℝ, (x^2 + y^2 = 4) → (x^2 + y^2 + 4*x - 4*y + 4 = 0)) → (∀ x y : ℝ, (x - y + 2 = 0)) :=
  sorry

end symmetric_line_eq_l234_234122


namespace solution_exists_l234_234132

theorem solution_exists (x : ℝ) : (x - 1)^2 = 4 → (x = 3 ∨ x = -1) :=
by
  sorry

end solution_exists_l234_234132


namespace no_common_complex_roots_l234_234702

theorem no_common_complex_roots (a b : ℚ) :
  ¬ ∃ α : ℂ, (α^5 - α - 1 = 0) ∧ (α^2 + a * α + b = 0) :=
sorry

end no_common_complex_roots_l234_234702


namespace small_seats_capacity_l234_234179

-- Definitions
def num_small_seats : ℕ := 2
def people_per_small_seat : ℕ := 14

-- Statement to prove
theorem small_seats_capacity :
  num_small_seats * people_per_small_seat = 28 :=
by
  -- Proof goes here
  sorry

end small_seats_capacity_l234_234179


namespace probability_of_triangle_or_circle_l234_234486

/-- The total number of figures -/
def total_figures : ℕ := 10

/-- The number of triangles -/
def triangles : ℕ := 3

/-- The number of circles -/
def circles : ℕ := 3

/-- The number of figures that are either triangles or circles -/
def favorable_figures : ℕ := triangles + circles

/-- The probability that the chosen figure is either a triangle or a circle -/
theorem probability_of_triangle_or_circle : (favorable_figures : ℚ) / (total_figures : ℚ) = 3 / 5 := 
by
  sorry

end probability_of_triangle_or_circle_l234_234486


namespace generate_1_generate_2_generate_3_generate_4_generate_5_generate_6_generate_7_generate_8_generate_9_generate_10_generate_11_generate_12_generate_13_generate_14_generate_15_generate_16_generate_17_generate_18_generate_19_generate_20_generate_21_generate_22_l234_234803

-- Define the number five as 4, as we are using five 4s
def four := 4

-- Now prove that each number from 1 to 22 can be generated using the conditions
theorem generate_1 : 1 = (4 / 4) * (4 / 4) := sorry
theorem generate_2 : 2 = (4 / 4) + (4 / 4) := sorry
theorem generate_3 : 3 = ((4 + 4 + 4) / 4) - (4 / 4) := sorry
theorem generate_4 : 4 = 4 * (4 - 4) + 4 := sorry
theorem generate_5 : 5 = 4 + (4 / 4) := sorry
theorem generate_6 : 6 = 4 + 4 - (4 / 4) := sorry
theorem generate_7 : 7 = 4 + 4 - (4 / 4) := sorry
theorem generate_8 : 8 = 4 + 4 := sorry
theorem generate_9 : 9 = 4 + 4 + (4 / 4) := sorry
theorem generate_10 : 10 = 4 * (2 + 4 / 4) := sorry
theorem generate_11 : 11 = 4 * (3 - 1 / 4) := sorry
theorem generate_12 : 12 = 4 + 4 + 4 := sorry
theorem generate_13 : 13 = (4 * 4) - (4 / 4) - 4 := sorry
theorem generate_14 : 14 = 4 * (4 - 1 / 4) := sorry
theorem generate_15 : 15 = 4 * 4 - (4 / 4) - 1 := sorry
theorem generate_16 : 16 = 4 * (4 - (4 - 4) / 4) := sorry
theorem generate_17 : 17 = 4 * (4 + 4 / 4) := sorry
theorem generate_18 : 18 = 4 * 4 + 4 - 4 / 4 := sorry
theorem generate_19 : 19 = 4 + 4 + 4 + 4 + 3 := sorry
theorem generate_20 : 20 = 4 + 4 + 4 + 4 + 4 := sorry
theorem generate_21 : 21 = 4 * 4 + (4 - 1) / 4 := sorry
theorem generate_22 : 22 = (4 * 4 + 4) / 4 := sorry

end generate_1_generate_2_generate_3_generate_4_generate_5_generate_6_generate_7_generate_8_generate_9_generate_10_generate_11_generate_12_generate_13_generate_14_generate_15_generate_16_generate_17_generate_18_generate_19_generate_20_generate_21_generate_22_l234_234803


namespace multiplier_of_product_l234_234458

variable {a b : ℝ}

theorem multiplier_of_product (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : a + b = k * (a * b))
  (h4 : (1 / a) + (1 / b) = 6) : k = 6 := by
  sorry

end multiplier_of_product_l234_234458


namespace positive_difference_of_two_numbers_l234_234347

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_of_two_numbers_l234_234347


namespace volleyball_team_points_l234_234286

theorem volleyball_team_points (lizzie_points nathalie_points aimee_points teammate_points total_points : ℕ)
  (h1 : lizzie_points = 4)
  (h2 : nathalie_points = lizzie_points + 3)
  (h3 : aimee_points = 2 * (lizzie_points + nathalie_points))
  (h4 : total_points = 50)
  (h5 : total_points = lizzie_points + nathalie_points + aimee_points + teammate_points) :
  teammate_points = 17 :=
begin
  sorry
end

end volleyball_team_points_l234_234286


namespace new_average_of_subtracted_elements_l234_234646

theorem new_average_of_subtracted_elements (a b c d e : ℝ) 
  (h_average : (a + b + c + d + e) / 5 = 5) 
  (new_a : ℝ := a - 2) 
  (new_b : ℝ := b - 2) 
  (new_c : ℝ := c - 2) 
  (new_d : ℝ := d - 2) :
  (new_a + new_b + new_c + new_d + e) / 5 = 3.4 := 
by 
  sorry

end new_average_of_subtracted_elements_l234_234646


namespace max_value_xyz_l234_234310

theorem max_value_xyz 
  (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_sum : x + y + z = 3) : 
  ∃ M, M = 243 ∧ (x + y^4 + z^5) ≤ M := 
  by sorry

end max_value_xyz_l234_234310


namespace percentage_donated_to_orphan_house_l234_234061

-- Given conditions as definitions in Lean 4
def income : ℝ := 400000
def children_percentage : ℝ := 0.2
def children_count : ℕ := 3
def wife_percentage : ℝ := 0.25
def remaining_after_donation : ℝ := 40000

-- Define the problem as a theorem
theorem percentage_donated_to_orphan_house :
  (children_count * children_percentage + wife_percentage) * income = 0.85 * income →
  (income - 0.85 * income = 60000) →
  remaining_after_donation = 40000 →
  (100 * (60000 - remaining_after_donation) / 60000) = 33.33 := 
by
  intros h1 h2 h3 
  sorry

end percentage_donated_to_orphan_house_l234_234061


namespace nancy_hours_to_work_l234_234314

def tuition := 22000
def scholarship := 3000
def hourly_wage := 10
def parents_contribution := tuition / 2
def student_loan := 2 * scholarship
def total_financial_aid := scholarship + student_loan
def remaining_tuition := tuition - parents_contribution - total_financial_aid
def hours_to_work := remaining_tuition / hourly_wage

theorem nancy_hours_to_work : hours_to_work = 200 := by
  -- This by block demonstrates that a proof would go here
  sorry

end nancy_hours_to_work_l234_234314


namespace sweeties_remainder_l234_234089

theorem sweeties_remainder (m : ℕ) (h : m % 6 = 4) : (2 * m) % 6 = 2 :=
by {
  sorry
}

end sweeties_remainder_l234_234089


namespace min_value_of_expression_l234_234106

theorem min_value_of_expression (x y : ℝ) (h : 2 * x - y = 4) : ∃ z : ℝ, (z = 4^x + (1/2)^y) ∧ z = 8 :=
by 
  sorry

end min_value_of_expression_l234_234106


namespace family_member_bites_count_l234_234678

-- Definitions based on the given conditions
def cyrus_bites_arms_and_legs : Nat := 14
def cyrus_bites_body : Nat := 10
def family_size : Nat := 6
def total_bites_cyrus : Nat := cyrus_bites_arms_and_legs + cyrus_bites_body
def total_bites_family : Nat := total_bites_cyrus / 2

-- Translation of the question to a theorem statement
theorem family_member_bites_count : (total_bites_family / family_size) = 2 := by
  -- use sorry to indicate the proof is skipped
  sorry

end family_member_bites_count_l234_234678


namespace no_cube_sum_of_three_consecutive_squares_l234_234405

theorem no_cube_sum_of_three_consecutive_squares :
  ¬∃ x y : ℤ, x^3 = (y-1)^2 + y^2 + (y+1)^2 :=
by
  sorry

end no_cube_sum_of_three_consecutive_squares_l234_234405


namespace negation_universal_proposition_l234_234507

theorem negation_universal_proposition :
  ¬ (∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 2 < 0 :=
by
  sorry

end negation_universal_proposition_l234_234507


namespace problem1_problem2_l234_234433

noncomputable def f (x a : ℝ) : ℝ :=
  abs (x - a) + abs (x + 3)

theorem problem1 : (∀ x : ℝ, f x 1 ≥ 6 → x ∈ Iic (-4) ∨ x ∈ Ici 2) :=
sorry

theorem problem2 : (∀ a x : ℝ, f x a > -a → a > -3/2) :=
sorry

end problem1_problem2_l234_234433


namespace product_check_l234_234161

theorem product_check : 
  (1200 < 31 * 53 ∧ 31 * 53 < 2400) ∧ 
  ¬ (1200 < 32 * 84 ∧ 32 * 84 < 2400) ∧ 
  ¬ (1200 < 63 * 54 ∧ 63 * 54 < 2400) ∧ 
  (1200 < 72 * 24 ∧ 72 * 24 < 2400) :=
by 
  sorry

end product_check_l234_234161


namespace original_number_is_80_l234_234955

theorem original_number_is_80 (t : ℝ) (h : t * 1.125 - t * 0.75 = 30) : t = 80 := by
  sorry

end original_number_is_80_l234_234955


namespace median_price_l234_234345

-- Definitions from conditions
def price1 : ℝ := 10
def price2 : ℝ := 12
def price3 : ℝ := 15

def sales1 : ℝ := 0.50
def sales2 : ℝ := 0.30
def sales3 : ℝ := 0.20

-- Statement of the problem
theorem median_price : (price1 * sales1 + price2 * sales2 + price3 * sales3) / 2 = 11 := by
  sorry

end median_price_l234_234345


namespace first_term_of_geometric_sequence_l234_234880

theorem first_term_of_geometric_sequence :
  ∀ (a b c : ℝ), 
    (∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ 16 = b * r ∧ c = 16 * r ∧ 128 = c * r) →
    a = 1 / 4 :=
by
  intros a b c
  rintro ⟨r, hr0, hbr, h16r, hcr, h128r⟩
  sorry

end first_term_of_geometric_sequence_l234_234880


namespace find_a3_l234_234146

open Nat

def seq (a : ℕ → ℕ) : Prop := 
  (a 1 = 1) ∧ (∀ n : ℕ, n > 0 → a (n + 1) - a n = n)

theorem find_a3 (a : ℕ → ℕ) (h : seq a) : a 3 = 4 := by
  sorry

end find_a3_l234_234146


namespace common_ratio_geometric_sequence_l234_234105

variable (a_n : ℕ → ℝ) (a1 : ℝ) (d : ℝ)

noncomputable def is_arithmetic_sequence (a_n : ℕ → ℝ) (a1 : ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a_n n = a1 + n * d

noncomputable def forms_geometric_sequence (a_n : ℕ → ℝ) : Prop :=
(a_n 4) / (a_n 0) = (a_n 16) / (a_n 4)

theorem common_ratio_geometric_sequence :
  d ≠ 0 → 
  forms_geometric_sequence (a_n : ℕ → ℝ) →
  is_arithmetic_sequence a_n a1 d →
  ((a_n 4) / (a1) = 9) :=
by
  sorry

end common_ratio_geometric_sequence_l234_234105


namespace distance_between_chords_l234_234023

-- Definitions based on the conditions
structure CircleGeometry where
  radius: ℝ
  d1: ℝ -- distance from the center to the closest chord (34 units)
  d2: ℝ -- distance from the center to the second chord (38 units)
  d3: ℝ -- distance from the center to the outermost chord (38 units)

-- The problem itself
theorem distance_between_chords (circle: CircleGeometry) (h1: circle.d2 = 3) (h2: circle.d1 = 3 * circle.d2) (h3: circle.d3 = circle.d2) :
  2 * circle.d2 = 6 :=
by
  sorry

end distance_between_chords_l234_234023


namespace intersection_A_B_l234_234299

-- Define set A
def A : Set ℤ := {-1, 1, 2, 3, 4}

-- Define set B with the given condition
def B : Set ℤ := {x : ℤ | 1 ≤ x ∧ x < 3}

-- The main theorem statement showing the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 2} :=
    sorry -- Placeholder for the proof

end intersection_A_B_l234_234299


namespace eggplant_weight_l234_234336

-- Define the conditions
def number_of_cucumbers : ℕ := 25
def weight_per_cucumber_basket : ℕ := 30
def number_of_eggplants : ℕ := 32
def total_weight : ℕ := 1870

-- Define the statement to be proved
theorem eggplant_weight :
  (total_weight - (number_of_cucumbers * weight_per_cucumber_basket)) / number_of_eggplants =
  (1870 - (25 * 30)) / 32 := 
by sorry

end eggplant_weight_l234_234336


namespace inverse_function_point_l234_234733

theorem inverse_function_point (f : ℝ → ℝ) (h_inv : Function.LeftInverse f f⁻¹) (h_point : f 2 = -1) : f⁻¹ (-1) = 2 :=
by
  sorry

end inverse_function_point_l234_234733


namespace add_decimals_l234_234995

theorem add_decimals :
  5.623 + 4.76 = 10.383 :=
by sorry

end add_decimals_l234_234995


namespace am_gm_inequality_l234_234741

theorem am_gm_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) : (1 + x) * (1 + y) * (1 + z) ≥ 8 :=
sorry

end am_gm_inequality_l234_234741


namespace setB_is_correct_l234_234421

def setA : Set ℤ := {-1, 0, 1, 2}
def f (x : ℤ) : ℤ := x^2 - 2*x
def setB : Set ℤ := {y | ∃ x ∈ setA, f x = y}

theorem setB_is_correct : setB = {-1, 0, 3} := by
  sorry

end setB_is_correct_l234_234421


namespace joann_third_day_lollipops_l234_234148

theorem joann_third_day_lollipops
  (a b c d e : ℕ)
  (h1 : b = a + 6)
  (h2 : c = b + 6)
  (h3 : d = c + 6)
  (h4 : e = d + 6)
  (h5 : a + b + c + d + e = 100) :
  c = 20 :=
by
  sorry

end joann_third_day_lollipops_l234_234148


namespace find_number_l234_234610

theorem find_number (x a_3 a_4 : ℕ) (h1 : x + a_4 = 5574) (h2 : x + a_3 = 557) : x = 5567 :=
  sorry

end find_number_l234_234610


namespace angle_P_of_extended_sides_l234_234323

noncomputable def regular_pentagon_angle_sum : ℕ := 540

noncomputable def internal_angle_regular_pentagon (n : ℕ) (h : 5 = n) : ℕ :=
  regular_pentagon_angle_sum / n

def interior_angle_pentagon : ℕ := 108

theorem angle_P_of_extended_sides (ABCDE : Prop) (h1 : interior_angle_pentagon = 108)
  (P : Prop) (h3 : 72 + 72 = 144) : 180 - 144 = 36 := by 
  sorry

end angle_P_of_extended_sides_l234_234323


namespace candies_eaten_l234_234564

theorem candies_eaten (A B D : ℕ) 
                      (h1 : 4 * B = 3 * A) 
                      (h2 : 7 * A = 6 * D) 
                      (h3 : A + B + D = 70) :
  A = 24 ∧ B = 18 ∧ D = 28 := 
by
  sorry

end candies_eaten_l234_234564


namespace sum_of_arith_geo_progression_l234_234666

noncomputable def sum_two_numbers (a b : ℝ) : ℝ :=
  a + b

theorem sum_of_arith_geo_progression : 
  ∃ (a b : ℝ), (∃ d : ℝ, a = 4 + d ∧ b = 4 + 2 * d) ∧ 
  (∃ r : ℝ, a * r = b ∧ b * r = 16) ∧ 
  sum_two_numbers a b = 8 + 6 * Real.sqrt 3 :=
by
  sorry

end sum_of_arith_geo_progression_l234_234666


namespace each_squirrel_needs_more_acorns_l234_234542

noncomputable def acorns_needed : ℕ := 300
noncomputable def total_acorns_collected : ℕ := 4500
noncomputable def number_of_squirrels : ℕ := 20

theorem each_squirrel_needs_more_acorns : 
  (acorns_needed - total_acorns_collected / number_of_squirrels) = 75 :=
by
  sorry

end each_squirrel_needs_more_acorns_l234_234542


namespace range_of_a_l234_234934

noncomputable def A : Set ℝ := {x | -2 ≤ x ∧ x < 4 }

noncomputable def B (a : ℝ) : Set ℝ := { x | x^2 - a * x - 4 ≤ 0 }

theorem range_of_a (a : ℝ) : (B a ⊆ A) ↔ (0 ≤ a ∧ a < 3) := sorry

end range_of_a_l234_234934


namespace monotonic_intervals_a_eq_1_range_of_a_no_zero_points_in_interval_l234_234115

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

theorem monotonic_intervals_a_eq_1 :
  ∀ x : ℝ, (0 < x ∧ x ≤ 2 → (f x 1) < (f 2 1)) ∧ 
           (2 ≤ x → (f x 1) > (f 2 1)) :=
by
  sorry

theorem range_of_a_no_zero_points_in_interval :
  ∀ a : ℝ, (∀ x : ℝ, (0 < x ∧ x < 1/3) → ((2 - a) * (x - 1) - 2 * Real.log x) > 0) ↔ 2 - 3 * Real.log 3 ≤ a :=
by
  sorry

end monotonic_intervals_a_eq_1_range_of_a_no_zero_points_in_interval_l234_234115


namespace alex_score_l234_234164

theorem alex_score (initial_students : ℕ) (initial_average : ℕ) (total_students : ℕ) (new_average : ℕ) (initial_total : ℕ) (new_total : ℕ) :
  initial_students = 19 →
  initial_average = 76 →
  total_students = 20 →
  new_average = 78 →
  initial_total = initial_students * initial_average →
  new_total = total_students * new_average →
  new_total - initial_total = 116 :=
by
  sorry

end alex_score_l234_234164


namespace total_cost_other_goods_l234_234391

/-- The total cost of the goods other than tuna and water -/
theorem total_cost_other_goods :
  let cost_tuna := 5 * 2
  let cost_water := 4 * 1.5
  let total_paid := 56
  let cost_other := total_paid - cost_tuna - cost_water
  cost_other = 40 :=
by
  let cost_tuna := 5 * 2
  let cost_water := 4 * 1.5
  let total_paid := 56
  let cost_other := total_paid - cost_tuna - cost_water
  show cost_other = 40
  sorry

end total_cost_other_goods_l234_234391


namespace salary_increase_l234_234277

variable (S : ℝ) (P : ℝ)

theorem salary_increase (h1 : 1.16 * S = 406) (h2 : 350 + 350 * P = 420) : P * 100 = 20 := 
by
  sorry

end salary_increase_l234_234277


namespace decrease_in_average_salary_l234_234759

-- Define the conditions
variable (I : ℕ := 20)
variable (L : ℕ := 10)
variable (initial_wage_illiterate : ℕ := 25)
variable (new_wage_illiterate : ℕ := 10)

-- Define the theorem statement
theorem decrease_in_average_salary :
  (I * (initial_wage_illiterate - new_wage_illiterate)) / (I + L) = 10 := by
  sorry

end decrease_in_average_salary_l234_234759


namespace tens_digit_19_2021_l234_234712

theorem tens_digit_19_2021 : (19^2021 % 100) / 10 % 10 = 1 :=
by sorry

end tens_digit_19_2021_l234_234712


namespace part1_solution_set_part2_range_a_l234_234435

-- Define the function f
noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

-- Part 1: Proving the solution set of the inequality
theorem part1_solution_set (x : ℝ) :
  (∀ x, f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2: Finding the range of values for a
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a > -a ↔ a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_a_l234_234435


namespace probability_2_1_to_2_5_l234_234046

noncomputable def F (x : ℝ) : ℝ :=
if x ≤ 2 then 0
else if x ≤ 3 then (x - 2)^2
else 1

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 2 then 0
else if x ≤ 3 then 2 * (x - 2)
else 0

theorem probability_2_1_to_2_5 : 
  (F 2.5 - F 2.1 = 0.24) := 
by
  -- calculations and proof go here, but we skip it with sorry
  sorry

end probability_2_1_to_2_5_l234_234046


namespace lines_are_parallel_l234_234266

def line1 (x : ℝ) : ℝ := 2 * x + 1
def line2 (x : ℝ) : ℝ := 2 * x + 5

theorem lines_are_parallel : ∀ x y : ℝ, line1 x = y → line2 x = y → false :=
by
  sorry

end lines_are_parallel_l234_234266


namespace total_delegates_l234_234997

theorem total_delegates 
  (D: ℕ) 
  (h1: 16 ≤ D)
  (h2: (D - 16) % 2 = 0)
  (h3: 10 ≤ D - 16) : D = 36 := 
sorry

end total_delegates_l234_234997


namespace sector_arc_length_l234_234331

theorem sector_arc_length (n : ℝ) (r : ℝ) (l : ℝ) (h1 : n = 90) (h2 : r = 3) (h3 : l = (n * Real.pi * r) / 180) :
  l = (3 / 2) * Real.pi := by
  rw [h1, h2] at h3
  sorry

end sector_arc_length_l234_234331


namespace max_value_of_a_plus_b_l234_234932

def max_possible_sum (a b : ℝ) (h1 : 4 * a + 3 * b ≤ 10) (h2 : a + 2 * b ≤ 4) : ℝ :=
  a + b

theorem max_value_of_a_plus_b :
  ∃a b : ℝ, (4 * a + 3 * b ≤ 10) ∧ (a + 2 * b ≤ 4) ∧ (a + b = 14 / 5) :=
by {
  sorry
}

end max_value_of_a_plus_b_l234_234932


namespace smallest_four_digit_number_divisible_by_9_with_conditions_l234_234809

theorem smallest_four_digit_number_divisible_by_9_with_conditions :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n % 9 = 0) ∧ (odd (digit_1 n) + even (digit_2 n) + even (digit_3 n) + even (digit_4 n) = 1 + 3) ∧ n = 2008 :=
sorry

end smallest_four_digit_number_divisible_by_9_with_conditions_l234_234809


namespace sin_expression_l234_234898

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x

theorem sin_expression (a b x₀ : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0)
  (h₂ : ∀ x, f a b x = f a b (π / 6 - x)) 
  (h₃ : f a b x₀ = (8 / 5) * a) 
  (h₄ : b = Real.sqrt 3 * a) :
  Real.sin (2 * x₀ + π / 6) = 7 / 25 :=
by
  sorry

end sin_expression_l234_234898


namespace farmer_goats_l234_234057

theorem farmer_goats (G C P : ℕ) (h1 : P = 2 * C) (h2 : C = G + 4) (h3 : G + C + P = 56) : G = 11 :=
by
  sorry

end farmer_goats_l234_234057


namespace cubic_inches_in_two_cubic_feet_l234_234907

-- Define the conversion factor between feet and inches
def foot_to_inch : ℕ := 12
-- Define the conversion factor between cubic feet and cubic inches
def cubic_foot_to_cubic_inch : ℕ := foot_to_inch ^ 3

-- State the theorem to be proved
theorem cubic_inches_in_two_cubic_feet : 2 * cubic_foot_to_cubic_inch = 3456 :=
by
  -- Proof steps go here
  sorry

end cubic_inches_in_two_cubic_feet_l234_234907


namespace find_n_in_arithmetic_sequence_l234_234292

noncomputable def arithmetic_sequence_n : ℕ :=
  sorry

theorem find_n_in_arithmetic_sequence (a : ℕ → ℕ) (d n : ℕ) :
  (a 3) + (a 4) = 10 → (a (n-3) + a (n-2)) = 30 → n * (a 1 + a n) / 2 = 100 → n = 10 :=
  sorry

end find_n_in_arithmetic_sequence_l234_234292


namespace evaluate_expression_l234_234865

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem evaluate_expression : (nabla (nabla 2 3) 4) = 16777219 :=
by sorry

end evaluate_expression_l234_234865


namespace find_real_number_l234_234097

theorem find_real_number (x : ℝ) (h1 : 0 < x) (h2 : ⌊x⌋ * x = 72) : x = 9 :=
sorry

end find_real_number_l234_234097


namespace find_base_tax_rate_l234_234923

noncomputable def income : ℝ := 10550
noncomputable def tax_paid : ℝ := 950
noncomputable def base_income : ℝ := 5000
noncomputable def excess_income : ℝ := income - base_income
noncomputable def excess_tax_rate : ℝ := 0.10

theorem find_base_tax_rate (base_tax_rate: ℝ) :
  base_tax_rate * base_income + excess_tax_rate * excess_income = tax_paid -> 
  base_tax_rate = 7.9 / 100 :=
by sorry

end find_base_tax_rate_l234_234923


namespace keith_remaining_cards_l234_234297

-- Definitions and conditions
def initial_cards := 0
def new_cards := 8
def total_cards_after_purchase := initial_cards + new_cards
def remaining_cards := total_cards_after_purchase / 2

-- Proof statement (in Lean, the following would be a theorem)
theorem keith_remaining_cards : remaining_cards = 4 := sorry

end keith_remaining_cards_l234_234297


namespace profit_without_discount_l234_234549

noncomputable def cost_price : ℝ := 100
noncomputable def profit_percentage_with_discount : ℝ := 44
noncomputable def discount : ℝ := 4

theorem profit_without_discount (CP MP SP : ℝ) (h_CP : CP = cost_price) (h_pwpd : profit_percentage_with_discount = 44) (h_discount : discount = 4) (h_SP : SP = CP * (1 + profit_percentage_with_discount / 100)) (h_MP : SP = MP * (1 - discount / 100)) :
  ((MP - CP) / CP * 100) = 50 :=
by
  sorry

end profit_without_discount_l234_234549


namespace find_m_l234_234156

def sum_of_first_n_terms_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem find_m (a1 d : ℝ) (m : ℕ) :
  sum_of_first_n_terms_arithmetic_sequence a1 d (m - 1) = -2 →
  sum_of_first_n_terms_arithmetic_sequence a1 d m = 0 →
  sum_of_first_n_terms_arithmetic_sequence a1 d (m + 1) = 3 →
  m = 5 := by
  sorry

end find_m_l234_234156


namespace am_gm_inequality_for_x_l234_234478

theorem am_gm_inequality_for_x (x : ℝ) : 1 + x^2 + x^6 + x^8 ≥ 4 * x^4 := by 
  sorry

end am_gm_inequality_for_x_l234_234478


namespace inequality_solution_set_l234_234101

theorem inequality_solution_set 
  (a b : ℝ) 
  (h : ∀ x, x ∈ Set.Icc (-3) 1 → ax^2 + (a + b)*x + 2 > 0) : 
  a + b = -4/3 := 
sorry

end inequality_solution_set_l234_234101


namespace depth_of_water_in_smaller_container_l234_234698

theorem depth_of_water_in_smaller_container 
  (H_big : ℝ) (R_big : ℝ) (h_water : ℝ) 
  (H_small : ℝ) (R_small : ℝ) (expected_depth : ℝ) 
  (v_water_small : ℝ) 
  (v_water_big : ℝ) 
  (h_total_water : ℝ)
  (above_brim : ℝ) 
  (v_water_final : ℝ) : 

  H_big = 20 ∧ R_big = 6 ∧ h_water = 17 ∧ H_small = 18 ∧ R_small = 5 ∧ expected_depth = 2.88 ∧
  v_water_big = π * R_big^2 * H_big ∧ v_water_small = π * R_small^2 * H_small ∧ 
  h_total_water = π * R_big^2 * h_water ∧ above_brim = π * R_big^2 * (H_big - H_small) ∧ 
  v_water_final = above_brim →

  expected_depth = v_water_final / (π * R_small^2) :=
by
  intro h
  sorry

end depth_of_water_in_smaller_container_l234_234698


namespace prove_R36_div_R6_minus_R3_l234_234398

noncomputable def R (k : ℕ) : ℤ := (10^k - 1) / 9

theorem prove_R36_div_R6_minus_R3 :
  (R 36 / R 6) - R 3 = 100000100000100000100000100000099989 := sorry

end prove_R36_div_R6_minus_R3_l234_234398


namespace units_digit_17_pow_17_l234_234359

theorem units_digit_17_pow_17 : (17^17 % 10) = 7 := by
  sorry

end units_digit_17_pow_17_l234_234359


namespace payroll_amount_l234_234213

theorem payroll_amount (P : ℝ) 
  (h1 : P > 500000) 
  (h2 : 0.004 * (P - 500000) - 1000 = 600) :
  P = 900000 :=
by
  sorry

end payroll_amount_l234_234213


namespace find_positive_x_l234_234905

theorem find_positive_x (x y z : ℝ) (h1 : x * y = 10 - 3 * x - 2 * y)
  (h2 : y * z = 8 - 3 * y - 2 * z) (h3 : x * z = 40 - 5 * x - 3 * z) :
  x = 3 :=
by sorry

end find_positive_x_l234_234905


namespace polar_to_rectangular_conversion_l234_234236

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_conversion :
  polar_to_rectangular 5 (5 * Real.pi / 4) = (-5 * Real.sqrt 2 / 2, -5 * Real.sqrt 2 / 2) := by
  sorry

end polar_to_rectangular_conversion_l234_234236


namespace x_intercept_of_line_l234_234723

theorem x_intercept_of_line (x y : ℝ) : (4 * x + 7 * y = 28) ∧ (y = 0) → x = 7 :=
by
  sorry

end x_intercept_of_line_l234_234723


namespace sequence_formula_l234_234737

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = -2) (h2 : a 2 = -1.2) :
  ∀ n, a n = 0.8 * n - 2.8 :=
by
  sorry

end sequence_formula_l234_234737


namespace bennett_brother_count_l234_234847

def arora_brothers := 4

def twice_brothers_of_arora := 2 * arora_brothers

def bennett_brothers := twice_brothers_of_arora - 2

theorem bennett_brother_count : bennett_brothers = 6 :=
by
  unfold arora_brothers twice_brothers_of_arora bennett_brothers
  sorry

end bennett_brother_count_l234_234847


namespace magnitude_BC_eq_sqrt29_l234_234416

noncomputable def A : (ℝ × ℝ) := (2, -1)
noncomputable def C : (ℝ × ℝ) := (0, 2)
noncomputable def AB : (ℝ × ℝ) := (3, 5)

theorem magnitude_BC_eq_sqrt29
    (A : ℝ × ℝ := (2, -1))
    (C : ℝ × ℝ := (0, 2))
    (AB : ℝ × ℝ := (3, 5)) :
    ∃ B : ℝ × ℝ, (B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2 = 29 := 
by
  sorry

end magnitude_BC_eq_sqrt29_l234_234416


namespace range_a_exp_sub_one_gt_log_diff_l234_234901

section

variable {x m n a : ℝ}

/--
  Given the function f(x) = x - ln(x + 1) + (a - 1) / a, 
  prove that 0 < a ≤ 1 such that ∀ x > -1, f(x) ≤ 0.
-/
theorem range_a (h : ∀ x > -1, x - Real.log(x + 1) + (a - 1) / a ≤ 0) : 0 < a ∧ a ≤ 1 :=
sorry

/--
  Prove that e^(m - n) - 1 > ln(m + 1) - ln(n + 1) when m > n > 0.
-/
theorem exp_sub_one_gt_log_diff (h : m > n ∧ n > 0) : Real.exp(m - n) - 1 > Real.log(m + 1) - Real.log(n + 1) :=
sorry

end

end range_a_exp_sub_one_gt_log_diff_l234_234901


namespace exactly_one_female_student_l234_234021

-- Definitions directly from the conditions
def groupA_males : ℕ := 5
def groupA_females : ℕ := 3
def groupB_males : ℕ := 6
def groupB_females : ℕ := 2

-- The number of ways to choose 1 female student and the remaining students accordingly
def scenario1 : ℕ := Nat.choose 3 1 * Nat.choose 5 1 * Nat.choose 6 2
def scenario2 : ℕ := Nat.choose 2 1 * Nat.choose 5 2 * Nat.choose 6 1

-- The total number of ways
def total_ways : ℕ := scenario1 + scenario2

-- Lean statement for the proof problem
theorem exactly_one_female_student : total_ways = 345 := by
  sorry

end exactly_one_female_student_l234_234021


namespace determine_a_l234_234504

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem determine_a :
  {a : ℝ | 0 < a ∧ (f (a + 1) ≤ f (2 * a^2))} = {a : ℝ | 1 ≤ a ∧ a ≤ Real.sqrt 6 / 2 } :=
by
  sorry

end determine_a_l234_234504


namespace number_of_exclusive_students_l234_234172

-- Definitions from the conditions
def S_both : ℕ := 16
def S_alg : ℕ := 36
def S_geo_only : ℕ := 15

-- Theorem to prove the number of students taking algebra or geometry but not both
theorem number_of_exclusive_students : (S_alg - S_both) + S_geo_only = 35 :=
by
  sorry

end number_of_exclusive_students_l234_234172


namespace sector_perimeter_l234_234419

-- Conditions:
def theta : ℝ := 54  -- central angle in degrees
def r : ℝ := 20      -- radius in cm

-- Translation of given conditions and expected result:
theorem sector_perimeter (theta_eq : theta = 54) (r_eq : r = 20) :
  let l := (θ * r) / 180 * Real.pi 
  let perim := l + 2 * r 
  perim = 6 * Real.pi + 40 := sorry

end sector_perimeter_l234_234419


namespace find_A_l234_234541

theorem find_A (A B C D: ℕ) (h1: A ≠ B) (h2: A ≠ C) (h3: A ≠ D) (h4: B ≠ C) (h5: B ≠ D) (h6: C ≠ D)
  (hAB: A * B = 72) (hCD: C * D = 72) (hDiff: A - B = C + D + 2) : A = 6 :=
sorry

end find_A_l234_234541


namespace zookeeper_feeding_problem_l234_234552

noncomputable def feeding_ways : ℕ :=
  sorry

theorem zookeeper_feeding_problem :
  feeding_ways = 2880 := 
sorry

end zookeeper_feeding_problem_l234_234552


namespace calculate_value_l234_234821

theorem calculate_value :
  let number := 1.375
  let coef := 0.6667
  let increment := 0.75
  coef * number + increment = 1.666675 :=
by
  sorry

end calculate_value_l234_234821


namespace solution_set_l234_234261

noncomputable def f (x : ℝ) : ℝ := sorry -- Assume f is some real function (The existence of f is granted by the problem statement)

-- Given conditions
axiom f_at_1 : f 1 = 1
axiom f'_lt_half : ∀ x : ℝ, deriv f x < 1 / 2

-- Problem statement
theorem solution_set (x : ℝ) : f (x ^ 2) < 1 / 2 * x ^ 2 + 1 / 2 ↔ x < -1 ∨ 1 < x :=
by sorry

end solution_set_l234_234261


namespace probability_3a_3b_event_l234_234257

open MeasureTheory

noncomputable def probability_event (μ : Measure (ℝ × ℝ)) : ℝ :=
  μ {p : ℝ × ℝ | p.1 > 1/3 ∧ p.2 > 1/3} / μ univ

theorem probability_3a_3b_event (a b : ℝ) (h_a : 0 ≤ a ∧ a ≤ 1) (h_b : 0 ≤ b ∧ b ≤ 1) :
  probability_event (volume.restrict (Ioo (0 : ℝ) (1 : ℝ) ×ˢ Ioo (0 : ℝ) (1 : ℝ))) = 4 / 9 :=
  sorry

end probability_3a_3b_event_l234_234257


namespace valid_inequalities_l234_234319

theorem valid_inequalities (a b c : ℝ) (h : 0 < c) 
  (h1 : b > c - b)
  (h2 : c > a)
  (h3 : c > b - a) :
  a < c / 2 ∧ b < a + c / 2 :=
by
  sorry

end valid_inequalities_l234_234319


namespace total_sides_is_48_l234_234968

-- Definitions based on the conditions
def num_dice_tom : Nat := 4
def num_dice_tim : Nat := 4
def sides_per_die : Nat := 6

-- The proof problem statement
theorem total_sides_is_48 : (num_dice_tom + num_dice_tim) * sides_per_die = 48 := by
  sorry

end total_sides_is_48_l234_234968


namespace probability_y_eq_x_probability_x_plus_y_ge_10_l234_234760

/-- An experiment of throwing 2 dice, 
  where the coordinate of point P is represented by (x, y), 
  x is the number shown on the first die, 
  and y is the number shown on the second die. -/
def dice_outcomes := { (x, y) | x ∈ finset.range 1 7 ∧ y ∈ finset.range 1 7 }

/-- Number of possible outcomes for the experiment -/
noncomputable def total_outcomes := (6:ℕ) * (6:ℕ)

/-- Number of successful outcomes when P lies on the line y = x -/
noncomputable def successful_y_eq_x := finset.card { (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6) }

/-- Number of successful outcomes when P satisfies x + y ≥ 10 -/
noncomputable def successful_x_plus_y_ge_10 := 
  finset.card { (4, 6), (5, 5), (5, 6), (6, 4), (6, 5), (6, 6) }

/-- 
  The probability that point P lies on the line y = x 
  is 1/6, given the conditions. 
-/
theorem probability_y_eq_x : successful_y_eq_x / total_outcomes = 1 / 6 := 
sorry

/-- 
  The probability that point P satisfies x + y ≥ 10 
  is 1/6, given the conditions. 
-/
theorem probability_x_plus_y_ge_10 : successful_x_plus_y_ge_10 / total_outcomes = 1 / 6 := 
sorry

end probability_y_eq_x_probability_x_plus_y_ge_10_l234_234760


namespace weekly_profit_function_maximize_weekly_profit_weekly_sales_quantity_l234_234846

noncomputable def cost_price : ℝ := 10
noncomputable def y (x : ℝ) : ℝ := -10 * x + 400
noncomputable def w (x : ℝ) : ℝ := -10 * x ^ 2 + 500 * x - 4000

-- Proof Step 1: Show the functional relationship between w and x
theorem weekly_profit_function : ∀ x : ℝ, w x = -10 * x ^ 2 + 500 * x - 4000 := by
  intro x
  -- This is the function definition provided, proof omitted
  sorry

-- Proof Step 2: Find the selling price x that maximizes weekly profit
theorem maximize_weekly_profit : ∃ x : ℝ, x = 25 ∧ (∀ y : ℝ, y ≠ x → w y ≤ w x) := by
  use 25
  -- The details of solving the optimization are omitted
  sorry

-- Proof Step 3: Given weekly profit w = 2000 and constraints on y, find the weekly sales quantity
theorem weekly_sales_quantity (x : ℝ) (H : w x = 2000 ∧ y x ≥ 180) : y x = 200 := by
  have Hy : y x = -10 * x + 400 := by rfl
  have Hconstraint : y x ≥ 180 := H.2
  have Hprofit : w x = 2000 := H.1
  -- The details of solving for x and ensuring constraints are omitted
  sorry

end weekly_profit_function_maximize_weekly_profit_weekly_sales_quantity_l234_234846


namespace inequality_solution_l234_234327

theorem inequality_solution (x : ℝ) :
  (x^2 - 9) / (x + 3) < 0 ↔ x ∈ Set.Ioo (-∞) (-3) ∪ Set.Ioo (-3) 3 :=
by
  sorry

end inequality_solution_l234_234327


namespace suff_but_not_necessary_condition_l234_234204

theorem suff_but_not_necessary_condition (x y : ℝ) :
  (xy ≠ 6 → x ≠ 2 ∨ y ≠ 3) ∧ ¬ (x ≠ 2 ∨ y ≠ 3 → xy ≠ 6) :=
by
  sorry

end suff_but_not_necessary_condition_l234_234204


namespace linear_correlation_l234_234612

variable (r : ℝ) (r_critical : ℝ)

theorem linear_correlation (h1 : r = -0.9362) (h2 : r_critical = 0.8013) :
  |r| > r_critical :=
by
  sorry

end linear_correlation_l234_234612


namespace moles_of_C2H6_formed_l234_234406

-- Define the initial conditions
def initial_moles_H2 : ℕ := 3
def initial_moles_C2H4 : ℕ := 3
def reaction_ratio_C2H4_H2_C2H6 (C2H4 H2 C2H6 : ℕ) : Prop :=
  C2H4 = H2 ∧ C2H4 = C2H6

-- State the theorem to prove
theorem moles_of_C2H6_formed : reaction_ratio_C2H4_H2_C2H6 initial_moles_C2H4 initial_moles_H2 3 :=
by {
  sorry
}

end moles_of_C2H6_formed_l234_234406


namespace rationalize_denominator_l234_234776

theorem rationalize_denominator :
  (2 / (Real.cbrt 3 + Real.cbrt 27)) = (Real.cbrt 9 / 6) :=
by
  have h1 : Real.cbrt 27 = 3 * Real.cbrt 3 := sorry
  sorry

end rationalize_denominator_l234_234776


namespace regular_polygon_sides_l234_234828

theorem regular_polygon_sides (P s : ℕ) (hP : P = 180) (hs : s = 15) : P / s = 12 := by
  -- Given
  -- P = 180  -- the perimeter in cm
  -- s = 15   -- the side length in cm
  sorry

end regular_polygon_sides_l234_234828


namespace female_democrats_l234_234682

theorem female_democrats (F M : ℕ) 
    (h₁ : F + M = 990)
    (h₂ : F / 2 + M / 4 = 330) : F / 2 = 275 := 
by sorry

end female_democrats_l234_234682


namespace power_addition_l234_234414

variable {R : Type*} [CommRing R]

theorem power_addition (x : R) (m n : ℕ) (h1 : x^m = 6) (h2 : x^n = 2) : x^(m + n) = 12 :=
by
  sorry

end power_addition_l234_234414


namespace sum_of_cube_edges_l234_234823

/-- A cube has 12 edges. Each edge of a cube is of equal length. Given the length of one
edge as 15 cm, the sum of the lengths of all the edges of the cube is 180 cm. -/
theorem sum_of_cube_edges (edge_length : ℝ) (num_edges : ℕ) (h1 : edge_length = 15) (h2 : num_edges = 12) :
  num_edges * edge_length = 180 :=
by
  sorry

end sum_of_cube_edges_l234_234823


namespace smallest_t_in_colored_grid_l234_234099

theorem smallest_t_in_colored_grid :
  ∃ (t : ℕ), (t > 0) ∧
  (∀ (coloring : Fin (100*100) → ℕ),
      (∀ (n : ℕ), (∃ (squares : Finset (Fin (100*100))), squares.card ≤ 104 ∧ ∀ x ∈ squares, coloring x = n)) →
      (∃ (rectangle : Finset (Fin (100*100))),
        (rectangle.card = t ∧ (t = 1 ∨ (t = 2 ∨ ∃ (l : ℕ), (l = 12 ∧ rectangle.card = l) ∧ (∃ (c : ℕ), (c = 3 ∧ ∃ (a b c : ℕ), (a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ∃(s1 s2 s3 : Fin (100*100)), (s1 ∈ rectangle ∧ coloring s1 = a) ∧ (s2 ∈ rectangle ∧ coloring s2 = b) ∧ (s3 ∈ rectangle ∧ coloring s3 = c))))))))) :=
sorry

end smallest_t_in_colored_grid_l234_234099


namespace k_range_l234_234596

noncomputable def valid_k (k : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → x / Real.exp x < 1 / (k + 2 * x - x^2)

theorem k_range : {k : ℝ | valid_k k} = {k : ℝ | 0 ≤ k ∧ k < Real.exp 1 - 1} :=
by sorry

end k_range_l234_234596


namespace segment_length_l234_234902

theorem segment_length (x y : ℝ) (A B : ℝ × ℝ) 
  (h1 : A.2^2 = 4 * A.1) 
  (h2 : B.2^2 = 4 * B.1) 
  (h3 : A.2 = 2 * A.1 - 2)
  (h4 : B.2 = 2 * B.1 - 2)
  (h5 : A ≠ B) :
  dist A B = 5 :=
sorry

end segment_length_l234_234902


namespace find_a_45_l234_234154

theorem find_a_45 (a : ℕ → ℝ) 
  (h0 : a 0 = 11) 
  (h1 : a 1 = 11) 
  (h_rec : ∀ m n : ℕ, a (m + n) = (1 / 2) * (a (2 * m) + a (2 * n)) - (m - n) ^ 2) 
  : a 45 = 1991 :=
sorry

end find_a_45_l234_234154


namespace percentage_calculation_l234_234675

theorem percentage_calculation (Part Whole : ℕ) (h1 : Part = 90) (h2 : Whole = 270) : 
  ((Part : ℝ) / (Whole : ℝ) * 100) = 33.33 :=
by
  sorry

end percentage_calculation_l234_234675


namespace lcm_of_two_numbers_l234_234975

-- Define the given conditions: Two numbers a and b, their HCF, and their product.
variables (a b : ℕ)
def hcf : ℕ := 55
def product := 82500

-- Define the concept of HCF and LCM, using the provided relationship in the problem
def gcd_ab := hcf
def lcm_ab := (product / gcd_ab)

-- State the main theorem to prove: The LCM of the two numbers is 1500
theorem lcm_of_two_numbers : lcm_ab = 1500 := by
  -- This is the place where the actual proof steps would go
  sorry

end lcm_of_two_numbers_l234_234975


namespace find_p_l234_234425

-- Lean 4 definitions corresponding to the conditions
variables {p a b x0 y0 : ℝ} (hp : p > 0) (ha : a > 0) (hb : b > 0) (hx0 : x0 ≠ 0)
variables (hA : (y0^2 = 2 * p * x0) ∧ ((x0 / a)^2 - (y0 / b)^2 = 1))
variables (h_dist : x0 + x0 = p^2)
variables (h_ecc : (5^.half) = sqrt 5)

-- The proof problem
theorem find_p :
  p = 1 :=
by
  sorry

end find_p_l234_234425


namespace smallest_positive_four_digit_number_divisible_by_9_with_even_and_odd_l234_234808

theorem smallest_positive_four_digit_number_divisible_by_9_with_even_and_odd :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ 
             (n % 9 = 0) ∧ 
             (∃ d1 d2 d3 d4 : ℕ, 
               d1 * 1000 + d2 * 100 + d3 * 10 + d4 = n ∧ 
               d1 % 2 = 1 ∧ 
               d2 % 2 = 0 ∧ 
               d3 % 2 = 0 ∧ 
               d4 % 2 = 0) ∧ 
             (∀ m : ℕ, (1000 ≤ m ∧ m < 10000 ∧ m % 9 = 0 ∧ 
               ∃ e1 e2 e3 e4 : ℕ, 
                 e1 * 1000 + e2 * 100 + e3 * 10 + e4 = m ∧ 
                 e1 % 2 = 1 ∧ 
                 e2 % 2 = 0 ∧ 
                 e3 % 2 = 0 ∧ 
                 e4 % 2 = 0) → n ≤ m) ∧ 
             n = 1026 :=
sorry

end smallest_positive_four_digit_number_divisible_by_9_with_even_and_odd_l234_234808


namespace vanessa_made_16_l234_234047

/-
Each chocolate bar in a box costs $4.
There are 11 bars in total in the box.
Vanessa sold all but 7 bars.
Prove that Vanessa made $16.
-/

def cost_per_bar : ℕ := 4
def total_bars : ℕ := 11
def bars_unsold : ℕ := 7
def bars_sold : ℕ := total_bars - bars_unsold
def money_made : ℕ := bars_sold * cost_per_bar

theorem vanessa_made_16 : money_made = 16 :=
by
  sorry

end vanessa_made_16_l234_234047


namespace shaded_area_fraction_l234_234772

/-- The fraction of the larger square's area that is inside the shaded rectangle 
    formed by the points (2,2), (3,2), (3,5), and (2,5) on a 6 by 6 grid 
    is 1/12. -/
theorem shaded_area_fraction : 
  let grid_size := 6
  let rectangle_points := [(2, 2), (3, 2), (3, 5), (2, 5)]
  let rectangle_length := 1
  let rectangle_height := 3
  let rectangle_area := rectangle_length * rectangle_height
  let square_area := grid_size^2
  rectangle_area / square_area = 1 / 12 := 
by 
  sorry

end shaded_area_fraction_l234_234772


namespace blood_drug_concentration_at_13_hours_l234_234581

theorem blood_drug_concentration_at_13_hours :
  let peak_time := 3
  let test_interval := 2
  let decrease_rate := 0.4
  let target_rate := 0.01024
  let time_to_reach_target := (fun n => (2 * n + 1))
  peak_time + test_interval * 5 = 13 :=
sorry

end blood_drug_concentration_at_13_hours_l234_234581


namespace tangent_line_extreme_values_l234_234430

-- Define the function f and its conditions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 2

-- Given conditions
def cond1 (a b : ℝ) : Prop := 3 * a * 2^2 + b = 0
def cond2 (a b : ℝ) : Prop := a * 2^3 + b * 2 + 2 = -14

-- Part 1: Tangent line equation at (1, f(1))
theorem tangent_line (a b : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) : 
  ∃ m c : ℝ, m = -9 ∧ c = 9 ∧ (∀ y : ℝ, y = f a b 1 → 9 * 1 + y = 0) :=
sorry

-- Part 2: Extreme values on [-3, 3]
-- Define critical points and endpoints
def f_value_at (a b x : ℝ) := f a b x

theorem extreme_values (a b : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) :
  ∃ min max : ℝ, min = -14 ∧ max = 18 ∧ 
  f_value_at a b 2 = min ∧ f_value_at a b (-2) = max :=
sorry

end tangent_line_extreme_values_l234_234430


namespace candies_eaten_l234_234565

theorem candies_eaten (A B D : ℕ) 
                      (h1 : 4 * B = 3 * A) 
                      (h2 : 7 * A = 6 * D) 
                      (h3 : A + B + D = 70) :
  A = 24 ∧ B = 18 ∧ D = 28 := 
by
  sorry

end candies_eaten_l234_234565


namespace not_divisible_by_5_count_l234_234365

-- Define the total number of four-digit numbers using the digits 0, 1, 2, 3, 4, 5 without repetition
def total_four_digit_numbers : ℕ := 300

-- Define the number of four-digit numbers ending with 0
def numbers_ending_with_0 : ℕ := 60

-- Define the number of four-digit numbers ending with 5
def numbers_ending_with_5 : ℕ := 48

-- Theorem stating the number of four-digit numbers that cannot be divided by 5
theorem not_divisible_by_5_count : total_four_digit_numbers - numbers_ending_with_0 - numbers_ending_with_5 = 192 :=
by
  -- Proof skipped
  sorry

end not_divisible_by_5_count_l234_234365


namespace assignment_statement_correct_l234_234142

def meaning_of_assignment_statement (N : ℕ) := N + 1

theorem assignment_statement_correct :
  meaning_of_assignment_statement N = N + 1 :=
sorry

end assignment_statement_correct_l234_234142


namespace alcohol_solution_mixing_l234_234372

theorem alcohol_solution_mixing :
  ∀ (V_i C_i C_f C_a x : ℝ),
    V_i = 6 →
    C_i = 0.40 →
    C_f = 0.50 →
    C_a = 0.90 →
    x = 1.5 →
  0.50 * (V_i + x) = (C_i * V_i) + C_a * x →
  C_f * (V_i + x) = (C_i * V_i) + (C_a * x) := 
by
  intros V_i C_i C_f C_a x Vi_eq Ci_eq Cf_eq Ca_eq x_eq h
  sorry

end alcohol_solution_mixing_l234_234372


namespace polar_to_rectangular_conversion_l234_234237

theorem polar_to_rectangular_conversion:
  ∀ (r θ : ℝ), r = 5 → θ = (5 * π) / 4 →
    let x := r * cos θ, y := r * sin θ in
    (x, y) = (- (5 * real.sqrt 2) / 2, - (5 * real.sqrt 2) / 2) :=
by
  intros r θ hr hθ
  rw [hr, hθ]
  simp [(5 : ℝ), real.cos, real.sin, real.sqrt]
  sorry

end polar_to_rectangular_conversion_l234_234237


namespace area_enclosed_by_circle_l234_234196

theorem area_enclosed_by_circle : Π (x y : ℝ), x^2 + y^2 + 8 * x - 6 * y = -9 → 
  ∃ A, A = 7 * Real.pi :=
by
  sorry

end area_enclosed_by_circle_l234_234196


namespace no_four_distinct_real_roots_l234_234239

theorem no_four_distinct_real_roots (a b : ℝ) : ¬ (∃ (x1 x2 x3 x4 : ℝ), 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
  (x1^4 - 4*x1^3 + 6*x1^2 + a*x1 + b = 0) ∧ 
  (x2^4 - 4*x2^3 + 6*x2^2 + a*x2 + b = 0) ∧ 
  (x3^4 - 4*x3^3 + 6*x3^2 + a*x3 + b = 0) ∧ 
  (x4^4 - 4*x4^3 + 6*x4^2 + a*x4 + b = 0)) :=
by
  sorry

end no_four_distinct_real_roots_l234_234239


namespace average_weight_of_whole_class_l234_234976

/-- Section A has 30 students -/
def num_students_A : ℕ := 30

/-- Section B has 20 students -/
def num_students_B : ℕ := 20

/-- The average weight of Section A is 40 kg -/
def avg_weight_A : ℕ := 40

/-- The average weight of Section B is 35 kg -/
def avg_weight_B : ℕ := 35

/-- The average weight of the whole class is 38 kg -/
def avg_weight_whole_class : ℕ := 38

-- Proof that the average weight of the whole class is equal to 38 kg

theorem average_weight_of_whole_class : 
  ((num_students_A * avg_weight_A) + (num_students_B * avg_weight_B)) / (num_students_A + num_students_B) = avg_weight_whole_class :=
by
  -- Sorry indicates that the proof is omitted.
  sorry

end average_weight_of_whole_class_l234_234976


namespace solve_for_x_l234_234011

theorem solve_for_x (x : ℝ) : 64 = 4 * (16:ℝ)^(x - 2) → x = 3 :=
by 
  intro h
  sorry

end solve_for_x_l234_234011


namespace product_of_two_equal_numbers_l234_234015

-- Definitions and conditions
def arithmetic_mean (xs : List ℚ) : ℚ :=
  xs.sum / xs.length

-- Theorem stating the product of the two equal numbers
theorem product_of_two_equal_numbers (a b c : ℚ) (x : ℚ) :
  arithmetic_mean [a, b, c, x, x] = 20 → a = 22 → b = 18 → c = 32 → x * x = 196 :=
by
  intros h_mean h_a h_b h_c
  sorry

end product_of_two_equal_numbers_l234_234015


namespace round_24_6375_to_nearest_tenth_l234_234778

def round_to_nearest_tenth (n : ℚ) : ℚ :=
  let tenths := (n * 10).floor / 10
  let hundredths := (n * 100).floor % 10
  if hundredths < 5 then tenths else (tenths + 0.1)

theorem round_24_6375_to_nearest_tenth :
  round_to_nearest_tenth 24.6375 = 24.6 :=
by
  sorry

end round_24_6375_to_nearest_tenth_l234_234778


namespace ratio_proof_l234_234135

-- Define x and y as real numbers
variables (x y : ℝ)
-- Define the given condition
def given_condition : Prop := (3 * x - 2 * y) / (2 * x + y) = 3 / 4
-- Define the result to prove
def result : Prop := x / y = 11 / 6

-- State the theorem
theorem ratio_proof (h : given_condition x y) : result x y :=
by 
  sorry

end ratio_proof_l234_234135


namespace circle_radius_center_l234_234500

theorem circle_radius_center (x y : ℝ) (h : x^2 + y^2 - 2*x - 2*y - 2 = 0) :
  (∃ a b r, (x - a)^2 + (y - b)^2 = r^2 ∧ a = 1 ∧ b = 1 ∧ r = 2) := 
sorry

end circle_radius_center_l234_234500


namespace tangent_line_eq_extreme_values_interval_l234_234428

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^3 + b * x + 2

theorem tangent_line_eq (a b : ℝ) (h1 : 3 * a * 2^2 + b = 0) (h2 : a * 2^3 + b * 2 + 2 = -14) :
  9 * 1 + (f 1 a b) = 0 :=
sorry

theorem extreme_values_interval (a b : ℝ) (h1 : 3 * a * 2^2 + b = 0) (h2 : a * 2^3 + b * 2 + 2 = -14) :
  ∃ (min_val max_val : ℝ), 
    min_val = -14 ∧ f 2 a b = min_val ∧
    max_val = 18 ∧ f (-2) a b = max_val ∧
    ∀ x, (x ∈ Set.Icc (-3 : ℝ) 3 → f x a b ≥ min_val ∧ f x a b ≤ max_val) :=
sorry

end tangent_line_eq_extreme_values_interval_l234_234428


namespace tangent_line_extreme_values_l234_234429

-- Define the function f and its conditions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 2

-- Given conditions
def cond1 (a b : ℝ) : Prop := 3 * a * 2^2 + b = 0
def cond2 (a b : ℝ) : Prop := a * 2^3 + b * 2 + 2 = -14

-- Part 1: Tangent line equation at (1, f(1))
theorem tangent_line (a b : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) : 
  ∃ m c : ℝ, m = -9 ∧ c = 9 ∧ (∀ y : ℝ, y = f a b 1 → 9 * 1 + y = 0) :=
sorry

-- Part 2: Extreme values on [-3, 3]
-- Define critical points and endpoints
def f_value_at (a b x : ℝ) := f a b x

theorem extreme_values (a b : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) :
  ∃ min max : ℝ, min = -14 ∧ max = 18 ∧ 
  f_value_at a b 2 = min ∧ f_value_at a b (-2) = max :=
sorry

end tangent_line_extreme_values_l234_234429


namespace complement_A_in_U_l234_234601

noncomputable def U : Set ℝ := {x | x > -Real.sqrt 3}
noncomputable def A : Set ℝ := {x | 1 < 4 - x^2 ∧ 4 - x^2 ≤ 2}

theorem complement_A_in_U :
  (U \ A) = {x | -Real.sqrt 3 < x ∧ x ≤ -Real.sqrt 2} ∪ {x | Real.sqrt 2 ≤ x ∧ x < (Real.sqrt 3) ∨ Real.sqrt 3 ≤ x} :=
by
  sorry

end complement_A_in_U_l234_234601


namespace trigonometric_expression_proof_l234_234399

theorem trigonometric_expression_proof :
  (Real.cos (76 * Real.pi / 180) * Real.cos (16 * Real.pi / 180) +
   Real.cos (14 * Real.pi / 180) * Real.cos (74 * Real.pi / 180) -
   2 * Real.cos (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)) = 0 :=
by
  sorry

end trigonometric_expression_proof_l234_234399


namespace least_possible_c_l234_234684

theorem least_possible_c 
  (a b c : ℕ) 
  (h_avg : (a + b + c) / 3 = 20)
  (h_median : b = a + 13)
  (h_ord : a ≤ b ∧ b ≤ c)
  : c = 45 :=
sorry

end least_possible_c_l234_234684


namespace neg_p_equivalent_to_forall_x2_ge_1_l234_234337

open Classical

variable {x : ℝ}

-- Definition of the original proposition p
def p : Prop := ∃ (x : ℝ), x^2 < 1

-- The negation of the proposition p
def not_p : Prop := ∀ (x : ℝ), x^2 ≥ 1

-- The theorem stating the equivalence
theorem neg_p_equivalent_to_forall_x2_ge_1 : ¬ p ↔ not_p := by
  sorry

end neg_p_equivalent_to_forall_x2_ge_1_l234_234337


namespace regular_polygon_sides_l234_234829

theorem regular_polygon_sides (P s : ℕ) (hP : P = 180) (hs : s = 15) : P / s = 12 := by
  -- Given
  -- P = 180  -- the perimeter in cm
  -- s = 15   -- the side length in cm
  sorry

end regular_polygon_sides_l234_234829


namespace non_negative_real_sum_expressions_l234_234111

theorem non_negative_real_sum_expressions (x y z : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) (h_sum : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by
  sorry

end non_negative_real_sum_expressions_l234_234111


namespace observations_number_l234_234800

theorem observations_number 
  (mean : ℚ)
  (wrong_obs corrected_obs : ℚ)
  (new_mean : ℚ)
  (n : ℚ)
  (initial_mean : mean = 36)
  (wrong_obs_taken : wrong_obs = 23)
  (corrected_obs_value : corrected_obs = 34)
  (corrected_mean : new_mean = 36.5) :
  (n * mean + (corrected_obs - wrong_obs) = n * new_mean) → 
  n = 22 :=
by
  sorry

end observations_number_l234_234800


namespace select_people_with_at_least_one_boy_l234_234779

-- Define the problem conditions
def num_boys := 8
def num_girls := 6
def total_people := num_boys + num_girls
def select_people := 3

-- Prove the main statement
theorem select_people_with_at_least_one_boy :
  (Nat.choose total_people select_people) - (Nat.choose num_girls select_people) = 344 :=
by
  sorry

end select_people_with_at_least_one_boy_l234_234779


namespace cubic_inches_in_two_cubic_feet_l234_234908

-- Define the conversion factor between feet and inches
def foot_to_inch : ℕ := 12
-- Define the conversion factor between cubic feet and cubic inches
def cubic_foot_to_cubic_inch : ℕ := foot_to_inch ^ 3

-- State the theorem to be proved
theorem cubic_inches_in_two_cubic_feet : 2 * cubic_foot_to_cubic_inch = 3456 :=
by
  -- Proof steps go here
  sorry

end cubic_inches_in_two_cubic_feet_l234_234908


namespace exists_triangle_cut_into_2005_congruent_l234_234006

theorem exists_triangle_cut_into_2005_congruent (n : ℕ) (hn : n = 2005) : 
  ∃ (Δ : Type) [triangle Δ], ∃ (cut : Δ → list Δ), list.all (congruent Δ) (cut Δ) ∧ list.length (cut Δ) = n := 
sorry

end exists_triangle_cut_into_2005_congruent_l234_234006


namespace relatively_prime_days_in_february_l234_234085

-- Define the number of days in February based on leap year status
def days_in_february (is_leap_year : Bool) : Nat :=
  if is_leap_year then 29 else 28

-- Define a function to count how many days are relatively prime to 2 (February)
def count_relatively_prime_days (days : Nat) : Nat :=
  ((List.range days).filter (λ d => Nat.gcd (d + 1) 2 = 1)).length

-- The main theorem to prove the number of relatively prime days in February
theorem relatively_prime_days_in_february (is_leap_year : Bool) :
  count_relatively_prime_days (days_in_february is_leap_year) = if is_leap_year then 15 else 14 := by
  sorry

end relatively_prime_days_in_february_l234_234085


namespace arithmetic_square_root_of_16_l234_234953

theorem arithmetic_square_root_of_16 : ∃ x : ℝ, x^2 = 16 ∧ x > 0 ∧ x = 4 :=
by
  sorry

end arithmetic_square_root_of_16_l234_234953


namespace work_completion_days_l234_234544

theorem work_completion_days (A B C : ℕ) (A_rate B_rate C_rate : ℚ) :
  A_rate = 1 / 30 → B_rate = 1 / 55 → C_rate = 1 / 45 →
  1 / (A_rate + B_rate + C_rate) = 55 / 4 :=
by
  intro hA hB hC
  rw [hA, hB, hC]
  sorry

end work_completion_days_l234_234544


namespace one_minus_repeating_six_l234_234393

noncomputable def repeating_six : Real := 2 / 3

theorem one_minus_repeating_six : 1 - repeating_six = 1 / 3 :=
by
  sorry

end one_minus_repeating_six_l234_234393


namespace minimal_S_n_l234_234185

theorem minimal_S_n (a_n : ℕ → ℤ) 
  (h : ∀ n, a_n n = 3 * (n : ℤ) - 23) :
  ∃ n, (∀ m < n, (∀ k ≥ n, a_n k ≤ 0)) → n = 7 :=
by
  sorry

end minimal_S_n_l234_234185


namespace gcd_fac_7_and_8_equals_5040_l234_234244

theorem gcd_fac_7_and_8_equals_5040 : Nat.gcd 7! 8! = 5040 := 
by 
  sorry

end gcd_fac_7_and_8_equals_5040_l234_234244


namespace gcd_7_8_fact_l234_234247

-- Define factorial function in lean
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- Define the GCD function
noncomputable def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Define specific factorial values
def f7 := fact 7
def f8 := fact 8

-- Theorem stating the gcd of 7! and 8!
theorem gcd_7_8_fact : gcd f7 f8 = 5040 := by
  sorry

end gcd_7_8_fact_l234_234247


namespace manager_salary_is_correct_l234_234954

noncomputable def manager_salary (avg_salary_50_employees : ℝ) (increase_in_avg : ℝ) : ℝ :=
  let total_salary_50_employees := 50 * avg_salary_50_employees
  let new_avg_salary := avg_salary_50_employees + increase_in_avg
  let total_salary_51_people := 51 * new_avg_salary
  let manager_salary := total_salary_51_people - total_salary_50_employees
  manager_salary

theorem manager_salary_is_correct :
  manager_salary 2500 1500 = 79000 :=
by
  sorry

end manager_salary_is_correct_l234_234954


namespace chess_tournament_l234_234141

theorem chess_tournament (n games : ℕ) 
  (h_games : games = 81)
  (h_equation : (n - 2) * (n - 3) = 156) :
  n = 15 :=
sorry

end chess_tournament_l234_234141


namespace complex_number_solution_l234_234732

open Complex

theorem complex_number_solution (z : ℂ) (h : (z - 2 * I) / z = 2 + I) :
  im z = -1 ∧ abs z = Real.sqrt 2 ∧ z ^ 6 = -8 * I :=
by
  sorry

end complex_number_solution_l234_234732


namespace calculation_correct_l234_234677

theorem calculation_correct (a b : ℝ) : 4 * a^2 * b - 3 * b * a^2 = a^2 * b :=
by sorry

end calculation_correct_l234_234677


namespace dot_product_of_a_and_b_l234_234750

noncomputable def vector_a (a b : ℝ × ℝ) (h1 : a + b = (1, -3)) (h2 : a - b = (3, 7)) : ℝ × ℝ := 
a

noncomputable def vector_b (a b : ℝ × ℝ) (h1 : a + b = (1, -3)) (h2 : a - b = (3, 7)) : ℝ × ℝ := 
b

theorem dot_product_of_a_and_b {a b : ℝ × ℝ} 
  (h1 : a + b = (1, -3)) 
  (h2 : a - b = (3, 7)) : 
  (a.1 * b.1 + a.2 * b.2) = -12 := 
sorry

end dot_product_of_a_and_b_l234_234750


namespace max_cards_mod3_l234_234088

theorem max_cards_mod3 (s : Finset ℕ) (h : s = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) : 
  ∃ t ⊆ s, t.card = 6 ∧ (t.prod id) % 3 = 1 := sorry

end max_cards_mod3_l234_234088


namespace repeating_decimal_fraction_form_l234_234084

noncomputable def repeating_decimal_rational := 2.71717171

theorem repeating_decimal_fraction_form : 
  repeating_decimal_rational = 269 / 99 ∧ (269 + 99 = 368) := 
by 
  sorry

end repeating_decimal_fraction_form_l234_234084


namespace arithmetic_sequence_5_7_9_l234_234114

variable {a : ℕ → ℕ}

theorem arithmetic_sequence_5_7_9 (h : 13 * (a 7) = 39) : a 5 + a 7 + a 9 = 9 := 
sorry

end arithmetic_sequence_5_7_9_l234_234114


namespace Mabel_marble_count_l234_234852

variable (K A M : ℕ)

axiom Amanda_condition : A + 12 = 2 * K
axiom Mabel_K_condition : M = 5 * K
axiom Mabel_A_condition : M = A + 63

theorem Mabel_marble_count : M = 85 := by
  sorry

end Mabel_marble_count_l234_234852


namespace sasha_train_problem_l234_234945

def wagon_number (W : ℕ) (S : ℕ) : Prop :=
  -- Conditions
  (1 ≤ W ∧ W ≤ 9) ∧          -- Wagon number is a single-digit number
  (S < W) ∧                  -- Seat number is less than the wagon number
  ( (W = 1 ∧ S ≠ 1) ∨ 
    (W = 2 ∧ S = 1)
  ) -- Monday is the 1st or 2nd day of the month and corresponding seat constraints

theorem sasha_train_problem :
  ∃ (W S : ℕ), wagon_number W S ∧ W = 2 ∧ S = 1 :=
by
  sorry

end sasha_train_problem_l234_234945


namespace painter_total_rooms_l234_234827

theorem painter_total_rooms (hours_per_room : ℕ) (rooms_already_painted : ℕ) (additional_painting_hours : ℕ) 
  (h1 : hours_per_room = 8) (h2 : rooms_already_painted = 8) (h3 : additional_painting_hours = 16) : 
  rooms_already_painted + (additional_painting_hours / hours_per_room) = 10 := by
  sorry

end painter_total_rooms_l234_234827


namespace circle_area_l234_234013

theorem circle_area (r : ℝ) (h : 5 * (1 / (2 * π * r)) = r / 2) : π * r^2 = 5 := 
by
  sorry -- Proof is not required, placeholder for the actual proof

end circle_area_l234_234013


namespace find_initial_population_l234_234510

-- Define the conditions that the population increases annually by 20%
-- and that the population after 2 years is 14400.
def initial_population (P : ℝ) : Prop :=
  1.44 * P = 14400

-- The theorem states that given the conditions, the initial population is 10000.
theorem find_initial_population (P : ℝ) (h : initial_population P) : P = 10000 :=
  sorry

end find_initial_population_l234_234510


namespace evaluate_expression_l234_234863

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem evaluate_expression : (nabla (nabla 2 3) 4) = 16777219 :=
by sorry

end evaluate_expression_l234_234863


namespace find_square_number_divisible_by_five_l234_234404

noncomputable def is_square (n : ℕ) : Prop :=
∃ k : ℕ, k * k = n

theorem find_square_number_divisible_by_five :
  ∃ x : ℕ, x ≥ 50 ∧ x ≤ 120 ∧ is_square x ∧ x % 5 = 0 ↔ x = 100 := by
sorry

end find_square_number_divisible_by_five_l234_234404


namespace remainder_when_divided_by_7_l234_234361

theorem remainder_when_divided_by_7 (n : ℕ) (h1 : n % 2 = 1) (h2 : ∃ p : ℕ, p > 0 ∧ (n + p) % 10 = 0 ∧ p = 5) : n % 7 = 5 :=
by
  sorry

end remainder_when_divided_by_7_l234_234361


namespace add_numerator_denominator_add_numerator_denominator_gt_one_l234_234488

variable {a b n : ℕ}

/-- Adding the same natural number to both the numerator and the denominator of a fraction 
    increases the fraction if it is less than one, and decreases the fraction if it is greater than one. -/
theorem add_numerator_denominator (h1: a < b) : (a + n) / (b + n) > a / b := sorry

theorem add_numerator_denominator_gt_one (h2: a > b) : (a + n) / (b + n) < a / b := sorry

end add_numerator_denominator_add_numerator_denominator_gt_one_l234_234488


namespace condition_for_positive_expression_l234_234804

theorem condition_for_positive_expression (a b c : ℝ) :
  (∀ x y : ℝ, x^2 + x * y + y^2 + a * x + b * y + c > 0) ↔ a^2 - a * b + b^2 < 3 * c :=
by
  -- Proof should be provided here
  sorry

end condition_for_positive_expression_l234_234804


namespace find_principal_l234_234408

variable (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)

theorem find_principal (h₁ : SI = 8625) (h₂ : R = 50 / 3) (h₃ : T = 3 / 4) :
  SI = (P * R * T) / 100 → P = 69000 := sorry

end find_principal_l234_234408


namespace transformations_result_l234_234191

theorem transformations_result :
  ∃ (r g : ℕ), r + g = 15 ∧ 
  21 + r - 5 * g = 0 ∧ 
  30 - 2 * r + 2 * g = 24 :=
by
  sorry

end transformations_result_l234_234191


namespace compute_expression_l234_234584

-- Define the conditions as specific values and operations within the theorem itself
theorem compute_expression : 5 + 7 * (2 - 9)^2 = 348 := 
  by
  sorry

end compute_expression_l234_234584


namespace repeating_decimal_denominators_l234_234329

theorem repeating_decimal_denominators (a b c : ℕ) (ha : 0 ≤ a ∧ a < 10) (hb : 0 ≤ b ∧ b < 10) (hc : 0 ≤ c ∧ c < 10) (h_not_all_nine : ¬(a = 9 ∧ b = 9 ∧ c = 9)) : 
  ∃ denominators : Finset ℕ, denominators.card = 7 ∧ (∀ d ∈ denominators, d ∣ 999) ∧ ¬ 1 ∈ denominators :=
sorry

end repeating_decimal_denominators_l234_234329


namespace penultimate_digit_odd_of_square_last_digit_six_l234_234591

theorem penultimate_digit_odd_of_square_last_digit_six 
  (n : ℕ) 
  (h : (n * n) % 10 = 6) : 
  ((n * n) / 10) % 2 = 1 :=
sorry

end penultimate_digit_odd_of_square_last_digit_six_l234_234591


namespace apply_f_2019_times_l234_234599

noncomputable def f (x : ℝ) : ℝ := (1 - x^3) ^ (-1/3 : ℝ)

theorem apply_f_2019_times (x : ℝ) (n : ℕ) (h : n = 2019) (hx : x = 2018) : 
  (f^[n]) x = 2018 :=
by
  sorry

end apply_f_2019_times_l234_234599


namespace inequality_solution_correct_l234_234417

variable (f : ℝ → ℝ)

def f_one : Prop := f 1 = 1

def f_prime_half : Prop := ∀ x : ℝ, (deriv f x) > (1 / 2)

def inequality_solution_set : Prop := ∀ x : ℝ, f (x^2) < (x^2 / 2 + 1 / 2) ↔ -1 < x ∧ x < 1

theorem inequality_solution_correct (h1 : f_one f) (h2 : f_prime_half f) : inequality_solution_set f := sorry

end inequality_solution_correct_l234_234417


namespace solve_part_a_solve_part_b_l234_234783

-- Part (a)
theorem solve_part_a (x : ℝ) (h1 : 36 * x^2 - 1 = (6 * x + 1) * (6 * x - 1)) :
  (3 / (1 - 6 * x) = 2 / (6 * x + 1) - (8 + 9 * x) / (36 * x^2 - 1)) ↔ x = 1 / 3 :=
sorry

-- Part (b)
theorem solve_part_b (z : ℝ) (h2 : 1 - z^2 = (1 + z) * (1 - z)) :
  (3 / (1 - z^2) = 2 / (1 + z)^2 - 5 / (1 - z)^2) ↔ z = -3 / 7 :=
sorry

end solve_part_a_solve_part_b_l234_234783


namespace problem1_problem2_l234_234432

noncomputable def f (x a : ℝ) : ℝ :=
  abs (x - a) + abs (x + 3)

theorem problem1 : (∀ x : ℝ, f x 1 ≥ 6 → x ∈ Iic (-4) ∨ x ∈ Ici 2) :=
sorry

theorem problem2 : (∀ a x : ℝ, f x a > -a → a > -3/2) :=
sorry

end problem1_problem2_l234_234432


namespace g_zero_value_l234_234629

variables {R : Type*} [Ring R]

def polynomial_h (f g h : Polynomial R) : Prop :=
  h = f * g

def constant_term (p : Polynomial R) : R :=
  p.coeff 0

variables {f g h : Polynomial R}

theorem g_zero_value
  (Hf : constant_term f = 6)
  (Hh : constant_term h = -18)
  (H : polynomial_h f g h) :
  g.coeff 0 = -3 :=
by
  sorry

end g_zero_value_l234_234629


namespace solve_for_x_l234_234755

theorem solve_for_x (x : ℝ) (h : 3 - 1 / (1 - x) = 2 * (1 / (1 - x))) : x = 0 :=
by
  sorry

end solve_for_x_l234_234755


namespace cookies_count_l234_234410

theorem cookies_count :
  ∀ (Tom Lucy Millie Mike Frank : ℕ), 
  (Tom = 16) →
  (Lucy = Nat.sqrt Tom) →
  (Millie = 2 * Lucy) →
  (Mike = 3 * Millie) →
  (Frank = Mike / 2 - 3) →
  Frank = 9 :=
by
  intros Tom Lucy Millie Mike Frank hTom hLucy hMillie hMike hFrank
  have h1 : Tom = 16 := hTom
  have h2 : Lucy = Nat.sqrt Tom := hLucy
  have h3 : Millie = 2 * Lucy := hMillie
  have h4 : Mike = 3 * Millie := hMike
  have h5 : Frank = Mike / 2 - 3 := hFrank
  sorry

end cookies_count_l234_234410


namespace dans_car_mpg_l234_234081

noncomputable def milesPerGallon (distance money gas_price : ℝ) : ℝ :=
  distance / (money / gas_price)

theorem dans_car_mpg :
  let gas_price := 4
  let distance := 432
  let money := 54
  milesPerGallon distance money gas_price = 32 :=
by
  simp [milesPerGallon]
  sorry

end dans_car_mpg_l234_234081


namespace work_completion_days_l234_234368

theorem work_completion_days (Dx : ℕ) (Dy : ℕ) (days_y_worked : ℕ) (days_x_finished_remaining : ℕ)
  (work_rate_y : ℝ) (work_rate_x : ℝ) 
  (h1 : Dy = 24)
  (h2 : days_y_worked = 12)
  (h3 : days_x_finished_remaining = 18)
  (h4 : work_rate_y = 1 / Dy)
  (h5 : 12 * work_rate_y = 1 / 2)
  (h6 : work_rate_x = 1 / (2 * days_x_finished_remaining))
  (h7 : Dx * work_rate_x = 1) : Dx = 36 := sorry

end work_completion_days_l234_234368


namespace inverse_function_condition_l234_234475

noncomputable def f (m x : ℝ) := (3 * x + 4) / (m * x - 5)

theorem inverse_function_condition (m : ℝ) :
  (∀ x : ℝ, f m (f m x) = x) ↔ m = -4 / 5 :=
by
  sorry

end inverse_function_condition_l234_234475


namespace least_common_multiple_of_marble_sharing_l234_234696

theorem least_common_multiple_of_marble_sharing : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 2 4) 5) 7) 8) 10 = 280 :=
sorry

end least_common_multiple_of_marble_sharing_l234_234696


namespace alex_guarantees_victory_with_52_bullseyes_l234_234124

variable (m : ℕ) -- total score of Alex after the first half
variable (opponent_score : ℕ) -- total score of opponent after the first half
variable (remaining_shots : ℕ := 60) -- shots remaining for both players

-- Assume Alex always scores at least 3 points per shot and a bullseye earns 10 points
def min_bullseyes_to_guarantee_victory (m opponent_score : ℕ) : Prop :=
  ∃ n : ℕ, n ≥ 52 ∧
  (m + 7 * n + 180) > (opponent_score + 540)

-- Statement: Prove that if Alex leads by 60 points halfway through, then the minimum number of bullseyes he needs to guarantee a win is 52.
theorem alex_guarantees_victory_with_52_bullseyes (m opponent_score : ℕ) :
  m >= opponent_score + 60 → min_bullseyes_to_guarantee_victory m opponent_score :=
  sorry

end alex_guarantees_victory_with_52_bullseyes_l234_234124


namespace john_needs_60_bags_l234_234924

theorem john_needs_60_bags
  (horses : ℕ)
  (feeding_per_day : ℕ)
  (food_per_feeding : ℕ)
  (bag_weight : ℕ)
  (days : ℕ)
  (tons_in_pounds : ℕ)
  (half : ℕ)
  (h1 : horses = 25)
  (h2 : feeding_per_day = 2)
  (h3 : food_per_feeding = 20)
  (h4 : bag_weight = 1000)
  (h5 : days = 60)
  (h6 : tons_in_pounds = 2000)
  (h7 : half = 1 / 2) :
  ((horses * feeding_per_day * food_per_feeding * days) / (tons_in_pounds * half)) = 60 := by
  sorry

end john_needs_60_bags_l234_234924


namespace kiril_age_problem_l234_234624

theorem kiril_age_problem (x : ℕ) (h1 : x % 5 = 0) (h2 : (x - 1) % 7 = 0) : 26 - x = 11 :=
by
  sorry

end kiril_age_problem_l234_234624


namespace solution_set_of_inequality_l234_234126

theorem solution_set_of_inequality (a x : ℝ) (h1 : a < 2) (h2 : a * x > 2 * x + a - 2) : x < 1 :=
sorry

end solution_set_of_inequality_l234_234126


namespace setB_is_correct_l234_234420

def setA : Set ℤ := {-1, 0, 1, 2}
def f (x : ℤ) : ℤ := x^2 - 2*x
def setB : Set ℤ := {y | ∃ x ∈ setA, f x = y}

theorem setB_is_correct : setB = {-1, 0, 3} := by
  sorry

end setB_is_correct_l234_234420


namespace set_B_correct_l234_234423

-- Define the set A
def A : Set ℤ := {-1, 0, 1, 2}

-- Define the set B using the given formula
def B : Set ℤ := {y | ∃ x ∈ A, y = x^2 - 2 * x}

-- State the theorem that B is equal to the given set {-1, 0, 3}
theorem set_B_correct : B = {-1, 0, 3} := 
by 
  sorry

end set_B_correct_l234_234423


namespace find_intersection_sums_l234_234186

noncomputable def cubic_expression (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 2
noncomputable def linear_expression (x : ℝ) : ℝ := -x / 2 + 1

theorem find_intersection_sums :
  (∃ x1 x2 x3 y1 y2 y3,
    cubic_expression x1 = linear_expression x1 ∧
    cubic_expression x2 = linear_expression x2 ∧
    cubic_expression x3 = linear_expression x3 ∧
    (x1 + x2 + x3 = 4) ∧ (y1 + y2 + y3 = 1)) :=
sorry

end find_intersection_sums_l234_234186


namespace nat_divisibility_l234_234005

theorem nat_divisibility (n : ℕ) : 27 ∣ (10^n + 18 * n - 1) :=
  sorry

end nat_divisibility_l234_234005


namespace distance_between_x_intercepts_l234_234690

noncomputable def slope_intercept_form (m : ℝ) (x1 y1 x : ℝ) : ℝ :=
  m * (x - x1) + y1

def x_intercept (m : ℝ) (x1 y1 : ℝ) : ℝ :=
  (y1 - m * x1) / m

theorem distance_between_x_intercepts : 
  ∀ (m1 m2 : ℝ) (x1 y1 : ℝ), 
  m1 = 4 → m2 = -2 → x1 = 8 → y1 = 20 →
  abs (x_intercept m1 x1 y1 - x_intercept m2 x1 y1) = 15 :=
by
  intros m1 m2 x1 y1 h_m1 h_m2 h_x1 h_y1
  rw [h_m1, h_m2, h_x1, h_y1]
  sorry

end distance_between_x_intercepts_l234_234690


namespace profit_amount_l234_234054

theorem profit_amount (SP : ℝ) (P : ℝ) (profit : ℝ) : 
  SP = 850 → P = 36 → profit = SP - SP / (1 + P / 100) → profit = 225 :=
by
  intros hSP hP hProfit
  rw [hSP, hP] at *
  simp at *
  sorry

end profit_amount_l234_234054


namespace cube_painting_equiv_1260_l234_234209

def num_distinguishable_paintings_of_cube : Nat :=
  1260

theorem cube_painting_equiv_1260 :
  ∀ (colors : Fin 8 → Color), -- assuming we have a type Color representing colors
    (∀ i j : Fin 6, i ≠ j → colors i ≠ colors j) →  -- each face has a different color
    ∃ f : Cube × Fin 8 → Cube × Fin 8, -- considering symmetry transformations (rotations)
      num_distinguishable_paintings_of_cube = 1260 :=
by
  -- Proof would go here
  sorry

end cube_painting_equiv_1260_l234_234209


namespace find_two_digit_numbers_l234_234092

theorem find_two_digit_numbers (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h : 2 * (a + b) = a * b) : 
  10 * a + b = 63 ∨ 10 * a + b = 44 ∨ 10 * a + b = 36 :=
by sorry

end find_two_digit_numbers_l234_234092


namespace max_perfect_squares_eq_60_l234_234369

open Finset

noncomputable def max_perfect_squares (n : ℕ) (a : Fin n → ℕ) : ℕ :=
if h : n = 100 ∧ (∀ i, a i ∈ range 1 101) ∧ (injective (a : Fin n → ℕ)) then
  let S : Fin n → ℕ := λ k, (range (k.1 + 1)).sum (a ∘ Fin.mk) in
  let perfectSquares := filter (λ x, ∃ m, x = m * m) (image S univ) in
  perfectSquares.card
else 0

theorem max_perfect_squares_eq_60 :
  ∃ (a : Fin 100 → ℕ), max_perfect_squares 100 a = 60 := sorry

end max_perfect_squares_eq_60_l234_234369


namespace problem_l234_234861

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem problem : (nabla (nabla 2 3) 4) = 16777219 :=
by
  unfold nabla
  -- First compute 2 ∇ 3
  have h1 : nabla 2 3 = 12 := by norm_num
  rw [h1]
  -- Now compute 12 ∇ 4
  unfold nabla
  norm_num
  sorry

end problem_l234_234861


namespace find_number_l234_234370

theorem find_number (x : ℝ) (h : 160 = 3.2 * x) : x = 50 :=
by 
  sorry

end find_number_l234_234370


namespace find_real_number_l234_234098

theorem find_real_number (x : ℝ) (h1 : 0 < x) (h2 : ⌊x⌋ * x = 72) : x = 9 :=
sorry

end find_real_number_l234_234098


namespace solve_for_x_l234_234645

theorem solve_for_x : 
  ∀ (x : ℝ), (∀ (a b : ℝ), a * b = 4 * a - 2 * b) → (3 * (6 * x) = -2) → (x = 17 / 2) :=
by
  sorry

end solve_for_x_l234_234645


namespace lindsey_savings_l234_234158

theorem lindsey_savings
  (september_savings : Nat := 50)
  (october_savings : Nat := 37)
  (november_savings : Nat := 11)
  (additional_savings : Nat := 25)
  (video_game_cost : Nat := 87)
  (total_savings := september_savings + october_savings + november_savings)
  (mom_bonus : Nat := if total_savings > 75 then additional_savings else 0)
  (final_amount := total_savings + mom_bonus - video_game_cost) :
  final_amount = 36 := by
  sorry

end lindsey_savings_l234_234158


namespace library_width_l234_234799

theorem library_width 
  (num_libraries : ℕ) 
  (length_per_library : ℕ) 
  (total_area_km2 : ℝ) 
  (conversion_factor : ℝ) 
  (total_area : ℝ) 
  (area_of_one_library : ℝ) 
  (width_of_library : ℝ) :

  num_libraries = 8 →
  length_per_library = 300 →
  total_area_km2 = 0.6 →
  conversion_factor = 1000000 →
  total_area = total_area_km2 * conversion_factor →
  area_of_one_library = total_area / num_libraries →
  width_of_library = area_of_one_library / length_per_library →
  width_of_library = 250 :=
by
  intros;
  sorry

end library_width_l234_234799


namespace total_people_counted_l234_234998

-- Definitions based on conditions
def people_second_day : ℕ := 500
def people_first_day : ℕ := 2 * people_second_day

-- Theorem statement
theorem total_people_counted : people_first_day + people_second_day = 1500 := 
by 
  sorry

end total_people_counted_l234_234998


namespace right_triangle_perimeter_l234_234066

noncomputable def perimeter_right_triangle (a b : ℝ) (hypotenuse : ℝ) : ℝ :=
  a + b + hypotenuse

theorem right_triangle_perimeter (a b : ℝ) (ha : a^2 + b^2 = 25) (hab : a * b = 10) (hhypotenuse : hypotenuse = 5) :
  perimeter_right_triangle a b hypotenuse = 5 + 3 * Real.sqrt 5 :=
by
  sorry

end right_triangle_perimeter_l234_234066


namespace Dalton_saved_amount_l234_234709

theorem Dalton_saved_amount (total_cost uncle_contribution additional_needed saved_from_allowance : ℕ) 
  (h_total_cost : total_cost = 7 + 12 + 4)
  (h_uncle_contribution : uncle_contribution = 13)
  (h_additional_needed : additional_needed = 4)
  (h_current_amount : total_cost - additional_needed = 19)
  (h_saved_amount : 19 - uncle_contribution = saved_from_allowance) :
  saved_from_allowance = 6 :=
sorry

end Dalton_saved_amount_l234_234709


namespace barbara_spent_on_other_goods_l234_234388

theorem barbara_spent_on_other_goods
  (cost_tuna : ℝ := 5 * 2)
  (cost_water : ℝ := 4 * 1.5)
  (total_paid : ℝ := 56) :
  total_paid - (cost_tuna + cost_water) = 40 := by
  sorry

end barbara_spent_on_other_goods_l234_234388


namespace friend_pays_correct_percentage_l234_234620

theorem friend_pays_correct_percentage (adoption_fee : ℝ) (james_payment : ℝ) (friend_payment : ℝ) 
  (h1 : adoption_fee = 200) 
  (h2 : james_payment = 150)
  (h3 : friend_payment = adoption_fee - james_payment) : 
  (friend_payment / adoption_fee) * 100 = 25 :=
by
  sorry

end friend_pays_correct_percentage_l234_234620


namespace volume_in_cubic_yards_l234_234548

-- Definition: A box with a specific volume in cubic feet.
def volume_in_cubic_feet (v : ℝ) : Prop :=
  v = 200

-- Definition: Conversion factor from cubic feet to cubic yards.
def cubic_feet_per_cubic_yard : ℝ := 27

-- Theorem: The volume of the box in cubic yards given the volume in cubic feet.
theorem volume_in_cubic_yards (v_cubic_feet : ℝ) 
    (h : volume_in_cubic_feet v_cubic_feet) : 
    v_cubic_feet / cubic_feet_per_cubic_yard = 200 / 27 :=
  by
    rw [h]
    sorry

end volume_in_cubic_yards_l234_234548


namespace Geli_pushups_total_l234_234884

variable (x : ℕ)
variable (total_pushups : ℕ)

theorem Geli_pushups_total (h : 10 + (10 + x) + (10 + 2 * x) = 45) : x = 5 :=
by
  sorry

end Geli_pushups_total_l234_234884


namespace shots_and_hits_l234_234697

theorem shots_and_hits (n k : ℕ) (h₀ : 10 < n) (h₁ : n < 20) (h₂ : 5 * k = 3 * (n - k)) : (n = 16) ∧ (k = 6) :=
by {
  -- We state the result that we wish to prove
  sorry
}

end shots_and_hits_l234_234697


namespace xy_sum_l234_234009

theorem xy_sum (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 44) : x + y = 2 :=
sorry

end xy_sum_l234_234009


namespace square_side_length_l234_234672

-- Definition of the problem (statements)
theorem square_side_length (A : ℝ) (s : ℝ) (h : A = s * s) (hA : A = 49) : s = 7 := 
by 
  sorry

end square_side_length_l234_234672


namespace infinite_coprime_pairs_with_divisibility_l234_234943

theorem infinite_coprime_pairs_with_divisibility :
  ∃ (A : ℕ → ℕ) (B : ℕ → ℕ), (∀ n, gcd (A n) (B n) = 1) ∧
    ∀ n, (A n ∣ (B n)^2 - 5) ∧ (B n ∣ (A n)^2 - 5) :=
sorry

end infinite_coprime_pairs_with_divisibility_l234_234943


namespace negation_of_proposition_l234_234450

open Real

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > sin x) ↔ (∃ x : ℝ, x ≤ sin x) :=
by
  sorry

end negation_of_proposition_l234_234450


namespace smallest_positive_integer_adding_to_725_is_5_l234_234356

theorem smallest_positive_integer_adding_to_725_is_5 :
  ∃ n : ℕ, n > 0 ∧ (725 + n) % 5 = 0 ∧ (∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → n ≤ m) :=
begin
  use 5,
  split,
  { exact nat.succ_pos' 4 },
  split,
  { exact nat.mod_eq_of_lt (by norm_num) },
  {
    intros m hm_mod hm_le,
    by_contra h,
    have : m < 5 := lt_of_not_ge h,
    have hm_lt_5 : (725 + m) % 5 < 5 := (nat.mod_lt (725 + m) (nat.zero_lt_succ 4)),
    linarith,
  }
end

end smallest_positive_integer_adding_to_725_is_5_l234_234356


namespace gcd_m_n_l234_234632

noncomputable def m := 55555555
noncomputable def n := 111111111

theorem gcd_m_n : Int.gcd m n = 1 := by
  sorry

end gcd_m_n_l234_234632


namespace relatively_prime_27x_plus_4_18x_plus_3_l234_234942

theorem relatively_prime_27x_plus_4_18x_plus_3 (x : ℕ) :
  Nat.gcd (27 * x + 4) (18 * x + 3) = 1 :=
sorry

end relatively_prime_27x_plus_4_18x_plus_3_l234_234942


namespace candies_eaten_l234_234575

-- Definitions

def Andrey_rate_eq_Boris_rate (candies_eaten_by_Andrey candies_eaten_by_Boris : ℕ) : Prop :=
  candies_eaten_by_Andrey / 4 = candies_eaten_by_Boris / 3

def Denis_rate_eq_Andrey_rate (candies_eaten_by_Denis candies_eaten_by_Andrey : ℕ) : Prop :=
  candies_eaten_by_Denis / 7 = candies_eaten_by_Andrey / 6

def total_candies (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) : Prop :=
  candies_eaten_by_Andrey + candies_eaten_by_Boris + candies_eaten_by_Denis = 70

-- Theorem to prove the candies eaten by Andrey, Boris, and Denis
theorem candies_eaten (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) :
  Andrey_rate_eq_Boris_rate candies_eaten_by_Andrey candies_eaten_by_Boris →
  Denis_rate_eq_Andrey_rate candies_eaten_by_Denis candies_eaten_by_Andrey →
  total_candies candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis →
  candies_eaten_by_Andrey = 24 ∧ candies_eaten_by_Boris = 18 ∧ candies_eaten_by_Denis = 28 :=
  by sorry

end candies_eaten_l234_234575


namespace max_marked_vertices_no_rectangle_l234_234525

-- Definitions for the conditions
def regular_polygon (n : ℕ) := n ≥ 3

def no_four_marked_vertices_form_rectangle (n : ℕ) (marked_vertices : Finset ℕ) : Prop :=
  ∀ (v1 v2 v3 v4 : ℕ), 
  v1 ∈ marked_vertices ∧ 
  v2 ∈ marked_vertices ∧ 
  v3 ∈ marked_vertices ∧ 
  v4 ∈ marked_vertices → 
  (v1, v2, v3, v4) do not form the vertices of a rectangle in a regular n-gon

-- The theorem we need to prove
theorem max_marked_vertices_no_rectangle (marked_vertices : Finset ℕ) :
  marked_vertices.card ≤ 1009 :=
begin
  assume h1: regular_polygon 2016,
  assume h2: no_four_marked_vertices_form_rectangle 2016 marked_vertices,
  sorry
end

end max_marked_vertices_no_rectangle_l234_234525


namespace jerseys_sold_l234_234014

theorem jerseys_sold (unit_price_jersey : ℕ) (total_revenue_jersey : ℕ) (n : ℕ) 
  (h_unit_price : unit_price_jersey = 165) 
  (h_total_revenue : total_revenue_jersey = 25740) 
  (h_eq : n * unit_price_jersey = total_revenue_jersey) : 
  n = 156 :=
by
  rw [h_unit_price, h_total_revenue] at h_eq
  sorry

end jerseys_sold_l234_234014


namespace eval_expr_l234_234949

def a := -1
def b := 1 / 7
def expr := (3 * a^3 - 2 * a * b + b^2) - 2 * (-a^3 - a * b + 4 * b^2)

theorem eval_expr : expr = -36 / 7 := by
  -- Inserting the proof using the original mathematical solution steps is not required here.
  sorry

end eval_expr_l234_234949


namespace longer_diagonal_length_l234_234837

-- Conditions
def rhombus_side_length := 65
def shorter_diagonal_length := 72

-- Prove that the length of the longer diagonal is 108
theorem longer_diagonal_length : 
  (2 * (Real.sqrt ((rhombus_side_length: ℝ)^2 - (shorter_diagonal_length / 2)^2))) = 108 := 
by 
  sorry

end longer_diagonal_length_l234_234837


namespace range_of_a_l234_234913

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (1 < x ∧ x < 2) → ((x - a) ^ 2 < 1)) ↔ (1 ≤ a ∧ a ≤ 2) :=
by 
  sorry

end range_of_a_l234_234913


namespace randi_peter_ratio_l234_234008

-- Given conditions
def ray_cents := 175
def cents_per_nickel := 5
def peter_cents := 30
def randi_extra_nickels := 6

-- Define the nickels Ray has
def ray_nickels := ray_cents / cents_per_nickel
-- Define the nickels Peter receives
def peter_nickels := peter_cents / cents_per_nickel
-- Define the nickels Randi receives
def randi_nickels := peter_nickels + randi_extra_nickels
-- Define the cents Randi receives
def randi_cents := randi_nickels * cents_per_nickel

-- The goal is to prove the ratio of the cents given to Randi to the cents given to Peter is 2.
theorem randi_peter_ratio : randi_cents / peter_cents = 2 := by
  sorry

end randi_peter_ratio_l234_234008


namespace candies_eaten_l234_234577

-- Definitions

def Andrey_rate_eq_Boris_rate (candies_eaten_by_Andrey candies_eaten_by_Boris : ℕ) : Prop :=
  candies_eaten_by_Andrey / 4 = candies_eaten_by_Boris / 3

def Denis_rate_eq_Andrey_rate (candies_eaten_by_Denis candies_eaten_by_Andrey : ℕ) : Prop :=
  candies_eaten_by_Denis / 7 = candies_eaten_by_Andrey / 6

def total_candies (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) : Prop :=
  candies_eaten_by_Andrey + candies_eaten_by_Boris + candies_eaten_by_Denis = 70

-- Theorem to prove the candies eaten by Andrey, Boris, and Denis
theorem candies_eaten (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) :
  Andrey_rate_eq_Boris_rate candies_eaten_by_Andrey candies_eaten_by_Boris →
  Denis_rate_eq_Andrey_rate candies_eaten_by_Denis candies_eaten_by_Andrey →
  total_candies candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis →
  candies_eaten_by_Andrey = 24 ∧ candies_eaten_by_Boris = 18 ∧ candies_eaten_by_Denis = 28 :=
  by sorry

end candies_eaten_l234_234577


namespace proportional_segments_l234_234072

theorem proportional_segments (a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4 d1 d2 d3 d4 : ℕ)
  (hA : a1 = 1 ∧ a2 = 2 ∧ a3 = 3 ∧ a4 = 4)
  (hB : b1 = 1 ∧ b2 = 2 ∧ b3 = 2 ∧ b4 = 4)
  (hC : c1 = 3 ∧ c2 = 5 ∧ c3 = 9 ∧ c4 = 13)
  (hD : d1 = 1 ∧ d2 = 2 ∧ d3 = 2 ∧ d4 = 3) :
  (b1 * b4 = b2 * b3) :=
by
  sorry

end proportional_segments_l234_234072


namespace solve_quadratic_and_linear_equations_l234_234176

theorem solve_quadratic_and_linear_equations :
  (∀ x : ℝ, x^2 - 4*x - 1 = 0 → x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) ∧
  (∀ x : ℝ, (x + 3) * (x - 3) = 3 * (x + 3) → x = -3 ∨ x = 6) :=
by
  sorry

end solve_quadratic_and_linear_equations_l234_234176


namespace teammates_score_is_correct_l234_234287

-- Definitions based on the given conditions
def Lizzie_score : ℕ := 4
def Nathalie_score : ℕ := Lizzie_score + 3
def Combined_score : ℕ := Lizzie_score + Nathalie_score
def Aimee_score : ℕ := 2 * Combined_score
def Total_score : ℕ := Lizzie_score + Nathalie_score + Aimee_score
def Whole_team_score : ℕ := 50
def Teammates_score : ℕ := Whole_team_score - Total_score

-- Proof statement
theorem teammates_score_is_correct : Teammates_score = 17 := by
  sorry

end teammates_score_is_correct_l234_234287


namespace percent_non_sugar_l234_234378

-- Definitions based on the conditions in the problem.
def pie_weight : ℕ := 200
def sugar_weight : ℕ := 50

-- Statement of the proof problem.
theorem percent_non_sugar : ((pie_weight - sugar_weight) * 100) / pie_weight = 75 :=
by
  sorry

end percent_non_sugar_l234_234378


namespace totalMountainNumbers_l234_234582

-- Define a 4-digit mountain number based on the given conditions.
def isMountainNumber (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d ∧
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    1 ≤ c ∧ c ≤ 9 ∧
    1 ≤ d ∧ d ≤ 9 ∧
    b > a ∧ b > d ∧ c > a ∧ c > d ∧
    a ≠ d

-- Define the main theorem stating that the total number of 4-digit mountain numbers is 1512.
theorem totalMountainNumbers : 
  ∃ n, (∀ m, isMountainNumber m → ∃ l, l = 1 ∧ 4 ≤ m ∧ m ≤ 9999) ∧ n = 1512 := sorry

end totalMountainNumbers_l234_234582


namespace eight_n_plus_nine_is_perfect_square_l234_234338

theorem eight_n_plus_nine_is_perfect_square 
  (n : ℕ) (N : ℤ) 
  (hN : N = 2 ^ (4 * n + 1) - 4 ^ n - 1)
  (hdiv : 9 ∣ N) :
  ∃ k : ℤ, 8 * N + 9 = k ^ 2 :=
by
  sorry

end eight_n_plus_nine_is_perfect_square_l234_234338


namespace selection_of_hexagonal_shape_l234_234641

-- Lean 4 Statement: Prove that there are 78 distinct ways to select diagram b from the hexagonal grid of diagram a, considering rotations.

theorem selection_of_hexagonal_shape :
  let center_positions := 1
  let first_ring_positions := 6
  let second_ring_positions := 12
  let third_ring_positions := 6
  let fourth_ring_positions := 1
  let total_positions := center_positions + first_ring_positions + second_ring_positions + third_ring_positions + fourth_ring_positions
  let rotations := 3
  total_positions * rotations = 78 := by
  -- You can skip the explicit proof body here, replace with sorry
  sorry

end selection_of_hexagonal_shape_l234_234641


namespace x_intercept_of_perpendicular_line_l234_234805

theorem x_intercept_of_perpendicular_line (x y : ℝ) (h1 : 5 * x - 3 * y = 9) (y_intercept : ℝ) 
  (h2 : y_intercept = 4) : x = 20 / 3 :=
sorry

end x_intercept_of_perpendicular_line_l234_234805


namespace number_of_total_flowers_l234_234662

theorem number_of_total_flowers :
  let n_pots := 141
  let flowers_per_pot := 71
  n_pots * flowers_per_pot = 10011 :=
by
  sorry

end number_of_total_flowers_l234_234662


namespace calculate_side_a_l234_234412

noncomputable def side_a (b c : ℝ) (A : ℝ) : ℝ :=
  let B := Real.arccos (1 / 7)
  b * Real.sin A / Real.sin B

theorem calculate_side_a :
  side_a 8 3 (Real.pi / 3) ≈ 7.47 :=
by
  sorry

end calculate_side_a_l234_234412


namespace cubic_representation_l234_234202

variable (a b : ℝ) (x : ℝ)
variable (v u w : ℝ)

axiom h1 : 6.266 * x^3 - 3 * a * x^2 + (3 * a^2 - b) * x - (a^3 - a * b) = 0
axiom h2 : b ≥ 0

theorem cubic_representation : v = a ∧ u = a ∧ w^2 = b → 
  6.266 * x^3 - 3 * v * x^2 + (3 * u^2 - w^2) * x - (u^3 - u * w^2) = 0 :=
by
  sorry

end cubic_representation_l234_234202


namespace find_sum_of_distinct_numbers_l234_234301

variable {R : Type} [LinearOrderedField R]

theorem find_sum_of_distinct_numbers (p q r s : R) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p ∧ r * s = -13 * q)
  (h2 : p + q = 12 * r ∧ p * q = -13 * s) :
  p + q + r + s = 2028 := 
by 
  sorry

end find_sum_of_distinct_numbers_l234_234301


namespace tanvi_rank_among_girls_correct_l234_234922

def Vikas_rank : ℕ := 9
def Tanvi_rank : ℕ := 17
def girls_between : ℕ := 2
def Tanvi_rank_among_girls : ℕ := 8

theorem tanvi_rank_among_girls_correct (Vikas_rank Tanvi_rank girls_between Tanvi_rank_among_girls : ℕ) 
  (h1 : Vikas_rank = 9) 
  (h2 : Tanvi_rank = 17) 
  (h3 : girls_between = 2)
  (h4 : Tanvi_rank_among_girls = 8): 
  Tanvi_rank_among_girls = 8 := by
  sorry

end tanvi_rank_among_girls_correct_l234_234922


namespace surface_area_implies_side_length_diagonal_l234_234528

noncomputable def cube_side_length_diagonal (A : ℝ) := 
  A = 864 → ∃ s d : ℝ, s = 12 ∧ d = 12 * Real.sqrt 3

theorem surface_area_implies_side_length_diagonal : 
  cube_side_length_diagonal 864 := by
  sorry

end surface_area_implies_side_length_diagonal_l234_234528


namespace proposition_four_l234_234892

variables (a b c : Type)

noncomputable def perpend_lines (a b : Type) : Prop := sorry
noncomputable def parallel_lines (a b : Type) : Prop := sorry

theorem proposition_four (a b c : Type) 
  (h1 : perpend_lines a b) (h2 : parallel_lines b c) :
  perpend_lines a c :=
sorry

end proposition_four_l234_234892


namespace time_to_write_numbers_in_minutes_l234_234125

theorem time_to_write_numbers_in_minutes : 
  (1 * 5 + 2 * (99 - 10 + 1) + 3 * (105 - 100 + 1)) / 60 = 4 := 
  by
  -- Calculation steps would go here
  sorry

end time_to_write_numbers_in_minutes_l234_234125


namespace function_behavior_l234_234742

noncomputable def f (x a : ℝ) : ℝ := (1 / 6) * x^2 - (1 / 2) * a * x^2 + x

theorem function_behavior (a : ℝ) (h1 : a ≤ 2) :
  convex_on ℝ (set.Ioo (-1 : ℝ) 2) (λ x, f x a) → 
  ∃ x ∈ set.Ioo (-1 : ℝ) 2, is_max_on (λ x, f x a) (set.Ioo (-1 : ℝ) 2) x ∧ 
  ¬ ∃ x ∈ set.Ioo (-1 : ℝ) 2, is_min_on (λ x, f x a) (set.Ioo (-1 : ℝ) 2) x :=
sorry

end function_behavior_l234_234742


namespace triangle_area_l234_234253

theorem triangle_area (a b C : ℝ) (h₁ : a = 45) (h₂ : b = 60) (h₃ : C = 37 * real.pi / 180) :
  1 / 2 * a * b * real.sin(C) ≈ 812.45 :=
by
  have area := 1 / 2 * a * b * real.sin(C)
  rw [h₁, h₂, h₃] at area
  sorry

end triangle_area_l234_234253


namespace difference_of_fractions_l234_234707

theorem difference_of_fractions (x y : ℝ) (h1 : x = 497) (h2 : y = 325) :
  (2/5) * (3 * x + 7 * y) - (3/5) * (x * y) = -95408.6 := by
  rw [h1, h2]
  sorry

end difference_of_fractions_l234_234707


namespace anna_grams_l234_234103

-- Definitions based on conditions
def gary_grams : ℕ := 30
def gary_cost_per_gram : ℝ := 15
def anna_cost_per_gram : ℝ := 20
def combined_cost : ℝ := 1450

-- Statement to prove
theorem anna_grams : (combined_cost - (gary_grams * gary_cost_per_gram)) / anna_cost_per_gram = 50 :=
by 
  sorry

end anna_grams_l234_234103


namespace novels_per_month_l234_234276

theorem novels_per_month (pages_per_novel : ℕ) (total_pages_per_year : ℕ) (months_in_year : ℕ) 
  (h1 : pages_per_novel = 200) (h2 : total_pages_per_year = 9600) (h3 : months_in_year = 12) : 
  (total_pages_per_year / pages_per_novel) / months_in_year = 4 :=
by
  have novels_per_year := total_pages_per_year / pages_per_novel
  have novels_per_month := novels_per_year / months_in_year
  sorry

end novels_per_month_l234_234276


namespace regular_polygon_sides_l234_234136

theorem regular_polygon_sides (ratio : ℕ) (interior exterior : ℕ) (sum_angles : ℕ) 
  (h1 : ratio = 5)
  (h2 : interior = 5 * exterior)
  (h3 : interior + exterior = sum_angles)
  (h4 : sum_angles = 180) : 

∃ (n : ℕ), n = 12 := 
by 
  sorry

end regular_polygon_sides_l234_234136


namespace part1_part2_l234_234447

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := sorry

theorem part2 (a : ℝ) : 
  (∀ x, f x a > -a) ↔ (a > -3/2) := sorry

end part1_part2_l234_234447


namespace domain_sqrt_log_l234_234711

def domain_condition1 (x : ℝ) : Prop := x + 1 ≥ 0
def domain_condition2 (x : ℝ) : Prop := 6 - 3 * x > 0

theorem domain_sqrt_log (x : ℝ) : domain_condition1 x ∧ domain_condition2 x ↔ -1 ≤ x ∧ x < 2 :=
  sorry

end domain_sqrt_log_l234_234711


namespace black_squares_covered_by_trominoes_l234_234254

theorem black_squares_covered_by_trominoes (n : ℕ) (h_odd : n % 2 = 1) :
  (∃ (k : ℕ), k * k = (n + 1) / 2 ∧ n ≥ 7) ↔ n ≥ 7 :=
by
  sorry

end black_squares_covered_by_trominoes_l234_234254


namespace compare_squares_l234_234078

theorem compare_squares : -6 * Real.sqrt 5 < -5 * Real.sqrt 6 := sorry

end compare_squares_l234_234078


namespace work_completion_l234_234689

theorem work_completion (original_men planned_days absent_men remaining_men completion_days : ℕ) :
  original_men = 180 → 
  planned_days = 55 →
  absent_men = 15 →
  remaining_men = original_men - absent_men →
  remaining_men * completion_days = original_men * planned_days →
  completion_days = 60 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end work_completion_l234_234689


namespace x_pow_4_minus_inv_x_pow_4_eq_727_l234_234110

theorem x_pow_4_minus_inv_x_pow_4_eq_727 (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 :=
by
  sorry

end x_pow_4_minus_inv_x_pow_4_eq_727_l234_234110


namespace triangle_inequality_l234_234912

-- Define the side lengths of a triangle
variables {a b c : ℝ}

-- State the main theorem
theorem triangle_inequality :
  (a + b) * (b + c) * (c + a) ≥ 8 * (a + b - c) * (b + c - a) * (c + a - b) :=
sorry

end triangle_inequality_l234_234912


namespace share_of_A_correct_l234_234215

theorem share_of_A_correct :
  let investment_A1 := 20000
  let investment_A2 := 15000
  let investment_B1 := 20000
  let investment_B2 := 16000
  let investment_C1 := 20000
  let investment_C2 := 26000
  let total_months1 := 5
  let total_months2 := 7
  let total_profit := 69900

  let total_investment_A := (investment_A1 * total_months1) + (investment_A2 * total_months2)
  let total_investment_B := (investment_B1 * total_months1) + (investment_B2 * total_months2)
  let total_investment_C := (investment_C1 * total_months1) + (investment_C2 * total_months2)
  let total_investment := total_investment_A + total_investment_B + total_investment_C

  let share_A := (total_investment_A : ℝ) / (total_investment : ℝ)
  let profit_A := share_A * (total_profit : ℝ)

  profit_A = 20500.99 :=
by
  sorry

end share_of_A_correct_l234_234215


namespace train_stoppage_time_l234_234587

theorem train_stoppage_time
    (speed_without_stoppages : ℕ)
    (speed_with_stoppages : ℕ)
    (time_unit : ℕ)
    (h1 : speed_without_stoppages = 50)
    (h2 : speed_with_stoppages = 30)
    (h3 : time_unit = 60) :
    (time_unit * (speed_without_stoppages - speed_with_stoppages) / speed_without_stoppages) = 24 :=
by
  sorry

end train_stoppage_time_l234_234587


namespace cone_radius_from_melted_cylinder_l234_234824

theorem cone_radius_from_melted_cylinder :
  ∀ (r_cylinder h_cylinder r_cone h_cone : ℝ),
  r_cylinder = 8 ∧ h_cylinder = 2 ∧ h_cone = 6 ∧
  (π * r_cylinder^2 * h_cylinder = (1 / 3) * π * r_cone^2 * h_cone) →
  r_cone = 8 :=
by
  sorry

end cone_radius_from_melted_cylinder_l234_234824


namespace range_of_t_l234_234315

def ellipse (x y t : ℝ) : Prop := (x^2) / 4 + (y^2) / t = 1

def distance_greater_than_one (x y t : ℝ) : Prop := 
  let a := if t > 4 then Real.sqrt t else 2
  let b := if t > 4 then 2 else Real.sqrt t
  let c := if t > 4 then Real.sqrt (t - 4) else Real.sqrt (4 - t)
  a - c > 1

theorem range_of_t (t : ℝ) : 
  (∀ x y, ellipse x y t → distance_greater_than_one x y t) ↔ 
  (3 < t ∧ t < 4) ∨ (4 < t ∧ t < 25 / 4) := 
sorry

end range_of_t_l234_234315


namespace min_value_reciprocal_l234_234107

theorem min_value_reciprocal (m n : ℝ) (hmn_gt : 0 < m * n) (hmn_add : m + n = 2) :
  (∃ x : ℝ, x = (1/m + 1/n) ∧ x = 2) :=
by sorry

end min_value_reciprocal_l234_234107


namespace andrey_boris_denis_eat_candies_l234_234556

def andrey_boris_condition (a b : ℕ) : Prop :=
  a = 4 ∧ b = 3

def andrey_denis_condition (a d : ℕ) : Prop :=
  a = 6 ∧ d = 7

def total_candies_condition (total : ℕ) : Prop :=
  total = 70

theorem andrey_boris_denis_eat_candies :
  ∃ (a b d : ℕ), andrey_boris_condition a b ∧ andrey_denis_condition a d ∧ 
                  (total_candies_condition (2 * (12 + 9 + 14)) ∧ 
                   2 * 12 = 24 ∧ 2 * 9 = 18 ∧ 2 * 14 = 28) →
                  (a = 24 ∧ b = 18 ∧ d = 28) :=
by
  sorry

end andrey_boris_denis_eat_candies_l234_234556


namespace campers_went_rowing_and_hiking_in_all_l234_234979

def C_rm : Nat := 41
def C_hm : Nat := 4
def C_ra : Nat := 26

theorem campers_went_rowing_and_hiking_in_all : (C_rm + C_ra) + C_hm = 71 :=
by
  sorry

end campers_went_rowing_and_hiking_in_all_l234_234979


namespace selection_plans_l234_234250

-- Definitions for the students
inductive Student
| A | B | C | D | E | F

open Student

-- Definitions for the subjects
inductive Subject
| Mathematics | Physics | Chemistry | Biology

open Subject

-- A function to count the number of valid selections such that A and B do not participate in Biology.
def countValidSelections : Nat :=
  let totalWays := Nat.factorial 6 / Nat.factorial 2 / Nat.factorial (6 - 4)
  let forbiddenWays := 2 * (Nat.factorial 5 / Nat.factorial 2 / Nat.factorial (5 - 3))
  totalWays - forbiddenWays

theorem selection_plans :
  countValidSelections = 240 :=
by
  sorry

end selection_plans_l234_234250


namespace sum_of_three_numbers_l234_234019

theorem sum_of_three_numbers
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 252)
  (h2 : ab + bc + ca = 116) :
  a + b + c = 22 :=
by
  sorry

end sum_of_three_numbers_l234_234019


namespace certain_number_is_10000_l234_234918

theorem certain_number_is_10000 (n : ℕ) (h1 : n - 999 = 9001) : n = 10000 :=
by
  sorry

end certain_number_is_10000_l234_234918


namespace log_inequality_solution_l234_234594

noncomputable def log_a (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (log_a a (3 / 5) < 1) ↔ (a ∈ Set.Ioo 0 (3 / 5) ∪ Set.Ioi 1) := 
by
  sorry

end log_inequality_solution_l234_234594


namespace find_number_l234_234609

theorem find_number : ∃ x : ℝ, 0.35 * x = 0.15 * 40 ∧ x = 120 / 7 :=
by
  sorry

end find_number_l234_234609


namespace swimmers_speed_in_still_water_l234_234994

theorem swimmers_speed_in_still_water
  (v : ℝ) -- swimmer's speed in still water
  (current_speed : ℝ) -- speed of the water current
  (time : ℝ) -- time taken to swim against the current
  (distance : ℝ) -- distance swum against the current
  (h_current_speed : current_speed = 2)
  (h_time : time = 3.5)
  (h_distance : distance = 7)
  (h_eqn : time = distance / (v - current_speed)) :
  v = 4 :=
by
  sorry

end swimmers_speed_in_still_water_l234_234994


namespace find_x_l234_234716

theorem find_x (x : ℝ) (h : ⌈x⌉ * x = 210) : x = 14 := by
  sorry

end find_x_l234_234716


namespace convex_polygon_sides_ne_14_l234_234382

noncomputable def side_length : ℝ := 1

def is_triangle (s : ℝ) : Prop :=
  s = side_length

def is_dodecagon (s : ℝ) : Prop :=
  s = side_length

def side_coincide (t : ℝ) (d : ℝ) : Prop :=
  is_triangle t ∧ is_dodecagon d ∧ t = d

def valid_resulting_sides (s : ℤ) : Prop :=
  s = 11 ∨ s = 12 ∨ s = 13

theorem convex_polygon_sides_ne_14 : ∀ t d, side_coincide t d → ¬ valid_resulting_sides 14 := 
by
  intro t d h
  sorry

end convex_polygon_sides_ne_14_l234_234382


namespace positive_solution_of_system_l234_234748

theorem positive_solution_of_system (x y z : ℝ) (h1 : x * y = 5 - 3 * x - 2 * y)
                                    (h2 : y * z = 8 - 5 * y - 3 * z)
                                    (h3 : x * z = 18 - 2 * x - 5 * z)
                                    (hx_pos : 0 < x) : x = 6 := 
sorry

end positive_solution_of_system_l234_234748


namespace cos_ratio_l234_234139

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (angle_A angle_B angle_C : ℝ)
variable (bc_coeff : 2 * c = 3 * b)
variable (sin_coeff : Real.sin angle_A = 2 * Real.sin angle_B)

theorem cos_ratio :
  (2 * c = 3 * b) →
  (Real.sin angle_A = 2 * Real.sin angle_B) →
  let cos_A := (b^2 + c^2 - a^2) / (2 * b * c)
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  (Real.cos angle_A / Real.cos angle_B = -2 / 7) :=
by
  intros bc_coeff sin_coeff
  sorry

end cos_ratio_l234_234139


namespace variance_of_ξ_l234_234747

noncomputable def probability_distribution (ξ : ℕ) : ℚ :=
  if ξ = 2 ∨ ξ = 4 ∨ ξ = 6 ∨ ξ = 8 ∨ ξ = 10 then 1/5 else 0

def expected_value (ξ_values : List ℕ) (prob : ℕ → ℚ) : ℚ :=
  ξ_values.map (λ ξ => ξ * prob ξ) |>.sum

def variance (ξ_values : List ℕ) (prob : ℕ → ℚ) (Eξ : ℚ) : ℚ :=
  ξ_values.map (λ ξ => prob ξ * (ξ - Eξ) ^ 2) |>.sum

theorem variance_of_ξ :
  let ξ_values := [2, 4, 6, 8, 10]
  let prob := probability_distribution
  let Eξ := expected_value ξ_values prob
  variance ξ_values prob Eξ = 8 :=
by
  -- Proof goes here
  sorry

end variance_of_ξ_l234_234747


namespace find_positive_x_l234_234093

theorem find_positive_x (x : ℝ) (hx : 0 < x) (h : ⌊x⌋ * x = 72) : x = 9 := sorry

end find_positive_x_l234_234093


namespace average_difference_l234_234053

def differences : List ℤ := [15, -5, 25, 35, -15, 10, 20]
def days : ℤ := 7

theorem average_difference (diff : List ℤ) (n : ℤ) 
  (h : diff = [15, -5, 25, 35, -15, 10, 20]) (h_days : n = 7) : 
  (diff.sum / n : ℚ) = 12 := 
by 
  rw [h, h_days]
  norm_num
  sorry

end average_difference_l234_234053


namespace pqrs_sum_l234_234307

noncomputable def distinct_real_numbers (p q r s : ℝ) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

theorem pqrs_sum (p q r s : ℝ) 
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s)
  (distinct : distinct_real_numbers p q r s) :
  p + q + r + s = -13 :=
  sorry

end pqrs_sum_l234_234307


namespace reciprocal_of_repeating_decimal_6_l234_234527

-- Define a repeating decimal .\overline{6}
noncomputable def repeating_decimal_6 : ℚ := 2 / 3

-- The theorem statement: reciprocal of .\overline{6} is 3/2
theorem reciprocal_of_repeating_decimal_6 :
  repeating_decimal_6⁻¹ = (3 / 2) :=
sorry

end reciprocal_of_repeating_decimal_6_l234_234527


namespace green_function_solution_l234_234858

noncomputable def G (x ξ : ℝ) (α : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ ξ then α + Real.log ξ else if ξ ≤ x ∧ x ≤ 1 then α + Real.log x else 0

theorem green_function_solution (x ξ α : ℝ) (hα : α ≠ 0) (hx_bound : 0 < x ∧ x ≤ 1) :
  ( G x ξ α = if 0 < x ∧ x ≤ ξ then α + Real.log ξ else if ξ ≤ x ∧ x ≤ 1 then α + Real.log x else 0 ) :=
sorry

end green_function_solution_l234_234858


namespace percent_of_x_l234_234680

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 50 + x / 25 - x / 10 + x / 5) = (16 / 100) * x := by
  sorry

end percent_of_x_l234_234680


namespace sin_alpha_plus_pi_over_2_l234_234738

theorem sin_alpha_plus_pi_over_2 
  (h1 : Real.pi / 2 < α) (h2 : α < Real.pi) (h3 : Real.tan α = -4 / 3) :
  Real.sin (α + Real.pi / 2) = -3 / 5 :=
by
  sorry

end sin_alpha_plus_pi_over_2_l234_234738


namespace smallest_number_is_33_l234_234169

theorem smallest_number_is_33 (x : ℝ) 
  (h1 : 2 * x = third)
  (h2 : 4 * x = second)
  (h3 : (x + 2 * x + 4 * x) / 3 = 77) : 
  x = 33 := 
by 
  sorry

end smallest_number_is_33_l234_234169


namespace total_cars_in_group_l234_234756

theorem total_cars_in_group (C : ℕ)
  (h1 : 37 ≤ C)
  (h2 : ∃ n ≥ 51, n ≤ C)
  (h3 : ∃ n ≤ 49, n + 51 = C - 37) :
  C = 137 :=
by
  sorry

end total_cars_in_group_l234_234756


namespace smallest_root_of_polynomial_l234_234032

theorem smallest_root_of_polynomial :
  ∃ x : ℝ, (24 * x^3 - 106 * x^2 + 116 * x - 70 = 0) ∧ x = 0.67 :=
by
  sorry

end smallest_root_of_polynomial_l234_234032


namespace problem1_problem2_l234_234431

noncomputable def f (x a : ℝ) : ℝ :=
  abs (x - a) + abs (x + 3)

theorem problem1 : (∀ x : ℝ, f x 1 ≥ 6 → x ∈ Iic (-4) ∨ x ∈ Ici 2) :=
sorry

theorem problem2 : (∀ a x : ℝ, f x a > -a → a > -3/2) :=
sorry

end problem1_problem2_l234_234431


namespace line_intersection_and_conditions_l234_234265

theorem line_intersection_and_conditions :
  let l1 := (3 * x + 4 * y - 2 = 0) 
  let l2 := (2 * x + y + 2 = 0) 
  let P := (-2, 2)
  let d := (| 4 * -2 - 3 * 2 - 6 | / sqrt (4^2 + (-3)^2) = 4)
  let line_parallel := (3 * x - y + 8 = 0)
  let line_perpendicular := (x + 3 * y - 4 = 0)
  P ∈ l1 ∧ P ∈ l2 ∧ d ∧ line_parallel ∧ line_perpendicular :=
  by sorry

end line_intersection_and_conditions_l234_234265


namespace value_of_x_in_equation_l234_234812

theorem value_of_x_in_equation : 
  (∀ x : ℕ, 8 ^ 17 + 8 ^ 17 + 8 ^ 17 + 8 ^ 17 = 2 ^ x → x = 53) := 
by 
  sorry

end value_of_x_in_equation_l234_234812


namespace hundred_thousandth_permutation_l234_234940

open List

-- Define the main structure for the digits
def digits := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Define a function that generates all permutations of a list of unique elements
def permutations {α : Type*} [DecidableEq α] (l : List α) : List (List α) :=
  l.permutations

-- Define a function that brings the permutations in lexicographic order
def lexicographic_order {α : Type*} [LinearOrder α] (l : List (List α)) : List (List α) :=
  l.qsort (fun a b => (a < b))

-- Define the desired index (100000th permutation)
def target_index := 100000

-- Prove that the 100000th permutation of the digits 1-9 is 358926471
theorem hundred_thousandth_permutation : 
  nth (lexicographic_order (permutations digits)) (target_index - 1) = some [3, 5, 8, 9, 2, 6, 4, 7, 1] :=
by {
  sorry
}

end hundred_thousandth_permutation_l234_234940


namespace independence_test_categorical_l234_234763

-- Define what an independence test entails
def independence_test (X Y : Type) : Prop :=  
  ∃ (P : X → Y → Prop), ∀ x y1 y2, P x y1 → P x y2 → y1 = y2

-- Define the type of variables (categorical)
def is_categorical (V : Type) : Prop :=
  ∃ (f : V → ℕ), true

-- State the proposition that an independence test checks the relationship between categorical variables
theorem independence_test_categorical (X Y : Type) (hx : is_categorical X) (hy : is_categorical Y) :
  independence_test X Y := 
sorry

end independence_test_categorical_l234_234763


namespace find_sum_of_distinct_numbers_l234_234302

variable {R : Type} [LinearOrderedField R]

theorem find_sum_of_distinct_numbers (p q r s : R) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p ∧ r * s = -13 * q)
  (h2 : p + q = 12 * r ∧ p * q = -13 * s) :
  p + q + r + s = 2028 := 
by 
  sorry

end find_sum_of_distinct_numbers_l234_234302


namespace find_nat_number_l234_234243

theorem find_nat_number (N : ℕ) (d : ℕ) (hd : d < 10) (h : N = 5 * d + d) : N = 25 :=
by
  sorry

end find_nat_number_l234_234243


namespace smallest_period_sum_l234_234538

noncomputable def smallest_positive_period (f : ℝ → ℝ) (g : ℝ → ℝ): ℝ → ℝ :=
λ x => f x + g x

theorem smallest_period_sum
  (f g : ℝ → ℝ)
  (m n : ℕ)
  (hf : ∀ x, f (x + m) = f x)
  (hg : ∀ x, g (x + n) = g x)
  (hm : m > 1)
  (hn : n > 1)
  (hgcd : Nat.gcd m n = 1)
  : ∃ T, T > 0 ∧ (∀ x, smallest_positive_period f g (x + T) = smallest_positive_period f g x) ∧ T = m * n := by
  sorry

end smallest_period_sum_l234_234538


namespace solution_to_system_of_inequalities_l234_234816

variable {x y : ℝ}

theorem solution_to_system_of_inequalities :
  11 * (-1/3 : ℝ)^2 + 8 * (-1/3 : ℝ) * (2/3 : ℝ) + 8 * (2/3 : ℝ)^2 ≤ 3 ∧
  (-1/3 : ℝ) - 4 * (2/3 : ℝ) ≤ -3 :=
by
  sorry

end solution_to_system_of_inequalities_l234_234816


namespace pqrs_sum_l234_234309

noncomputable def distinct_real_numbers (p q r s : ℝ) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

theorem pqrs_sum (p q r s : ℝ) 
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s)
  (distinct : distinct_real_numbers p q r s) :
  p + q + r + s = -13 :=
  sorry

end pqrs_sum_l234_234309


namespace candies_eaten_l234_234567

theorem candies_eaten (A B D : ℕ) 
                      (h1 : 4 * B = 3 * A) 
                      (h2 : 7 * A = 6 * D) 
                      (h3 : A + B + D = 70) :
  A = 24 ∧ B = 18 ∧ D = 28 := 
by
  sorry

end candies_eaten_l234_234567


namespace number_of_teams_l234_234144

-- Given the conditions and the required proof problem
theorem number_of_teams (n : ℕ) (h : (n * (n - 1)) / 2 = 28) : n = 8 := by
  sorry

end number_of_teams_l234_234144


namespace tom_paths_avoiding_construction_l234_234969

def tom_home : (ℕ × ℕ) := (0, 0)
def friend_home : (ℕ × ℕ) := (4, 3)
def construction_site : (ℕ × ℕ) := (2, 2)

def total_paths_without_restriction : ℕ := Nat.choose 7 4
def paths_via_construction_site : ℕ := (Nat.choose 4 2) * (Nat.choose 3 1)
def valid_paths : ℕ := total_paths_without_restriction - paths_via_construction_site

theorem tom_paths_avoiding_construction : valid_paths = 17 := by
  sorry

end tom_paths_avoiding_construction_l234_234969


namespace range_of_a_l234_234120

noncomputable def A (a : ℝ) : Set ℝ := {-1, a^2 + 1, a^2 - 3}
noncomputable def B (a : ℝ) : Set ℝ := {a - 3, a - 1, a + 1}

theorem range_of_a (a : ℝ) : (A a ∩ B a = {-2}) ↔ (a = -1) :=
by {
  sorry
}

end range_of_a_l234_234120


namespace sufficient_but_not_necessary_l234_234888

theorem sufficient_but_not_necessary (a : ℝ) : (a > 6 → a^2 > 36) ∧ ¬(a^2 > 36 → a > 6) := 
by
  sorry

end sufficient_but_not_necessary_l234_234888


namespace candies_eaten_l234_234573

variables (A B D : ℕ)

-- Conditions:
def condition1 : Prop := ∃ k1 k2 k3 : ℕ, k1 * 4 + k2 * 3 + k3 * 7 = 70
def condition2 : Prop := (B * 3 = A * 4) ∧ (D * 7 = A * 6)
def condition3 : Prop := A + B + D = 70

-- Theorem statement:
theorem candies_eaten (h1 : condition1) (h2 : condition2) (h3 : condition3) :
    A = 24 ∧ B = 18 ∧ D = 28 := sorry

end candies_eaten_l234_234573


namespace Tom_total_yearly_intake_l234_234090

def soda_weekday := 5 * 12
def water_weekday := 64
def juice_weekday := 3 * 8
def sports_drink_weekday := 2 * 16

def total_weekday_intake := soda_weekday + water_weekday + juice_weekday + sports_drink_weekday

def soda_weekend_holiday := 5 * 12
def water_weekend_holiday := 64
def juice_weekend_holiday := 3 * 8
def sports_drink_weekend_holiday := 1 * 16
def fruit_smoothie_weekend_holiday := 32

def total_weekend_holiday_intake := soda_weekend_holiday + water_weekend_holiday + juice_weekend_holiday + sports_drink_weekend_holiday + fruit_smoothie_weekend_holiday

def weekdays := 260
def weekend_days := 104
def holidays := 1

def total_yearly_intake := (weekdays * total_weekday_intake) + (weekend_days * total_weekend_holiday_intake) + (holidays * total_weekend_holiday_intake)

theorem Tom_total_yearly_intake :
  total_yearly_intake = 67380 := by
  sorry

end Tom_total_yearly_intake_l234_234090


namespace distance_between_x_intercepts_l234_234691

theorem distance_between_x_intercepts :
  let slope1 := 4
  let slope2 := -2
  let point := (8, 20)
  let line1 (x : ℝ) := slope1 * (x - point.1) + point.2
  let line2 (x : ℝ) := slope2 * (x - point.1) + point.2
  let x_intercept1 := (0 - point.2) / slope1 + point.1
  let x_intercept2 := (0 - point.2) / slope2 + point.1
  abs (x_intercept1 - x_intercept2) = 15 := sorry

end distance_between_x_intercepts_l234_234691


namespace image_of_center_l234_234233

-- Define the initial coordinates
def initial_coordinate : ℝ × ℝ := (-3, 4)

-- Function to reflect a point across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Function to translate a point up
def translate_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

-- Definition of the final coordinate
noncomputable def final_coordinate : ℝ × ℝ :=
  translate_up (reflect_x initial_coordinate) 5

-- Theorem stating the final coordinate after transformations
theorem image_of_center : final_coordinate = (-3, 1) := by
  -- Proof is omitted
  sorry

end image_of_center_l234_234233


namespace students_after_joining_l234_234647

theorem students_after_joining (N : ℕ) (T : ℕ)
  (h1 : T = 48 * N)
  (h2 : 120 * 32 / (N + 120) + (T / (N + 120)) = 44)
  : N + 120 = 480 :=
by
  sorry

end students_after_joining_l234_234647


namespace andrey_boris_denis_eat_candies_l234_234555

def andrey_boris_condition (a b : ℕ) : Prop :=
  a = 4 ∧ b = 3

def andrey_denis_condition (a d : ℕ) : Prop :=
  a = 6 ∧ d = 7

def total_candies_condition (total : ℕ) : Prop :=
  total = 70

theorem andrey_boris_denis_eat_candies :
  ∃ (a b d : ℕ), andrey_boris_condition a b ∧ andrey_denis_condition a d ∧ 
                  (total_candies_condition (2 * (12 + 9 + 14)) ∧ 
                   2 * 12 = 24 ∧ 2 * 9 = 18 ∧ 2 * 14 = 28) →
                  (a = 24 ∧ b = 18 ∧ d = 28) :=
by
  sorry

end andrey_boris_denis_eat_candies_l234_234555


namespace harvest_duration_l234_234000

theorem harvest_duration (total_earnings earnings_per_week : ℕ) (h1 : total_earnings = 1216) (h2 : earnings_per_week = 16) :
  total_earnings / earnings_per_week = 76 :=
by
  sorry

end harvest_duration_l234_234000


namespace int_valued_fractions_l234_234721

theorem int_valued_fractions (a : ℤ) :
  ∃ k : ℤ, (a^2 - 21 * a + 17) = k * a ↔ a = 1 ∨ a = -1 ∨ a = 17 ∨ a = -17 :=
by {
  sorry
}

end int_valued_fractions_l234_234721


namespace part1_solution_part2_solution_l234_234438

-- Proof problem for Part (1)
theorem part1_solution (x : ℝ) (a : ℝ) (h : a = 1) : 
  (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

-- Proof problem for Part (2)
theorem part2_solution (a : ℝ) : 
  (∀ x : ℝ, (|x - a| + |x + 3|) > -a) ↔ a > -3 / 2 :=
by
  sorry

end part1_solution_part2_solution_l234_234438


namespace andrey_boris_denis_eat_candies_l234_234554

def andrey_boris_condition (a b : ℕ) : Prop :=
  a = 4 ∧ b = 3

def andrey_denis_condition (a d : ℕ) : Prop :=
  a = 6 ∧ d = 7

def total_candies_condition (total : ℕ) : Prop :=
  total = 70

theorem andrey_boris_denis_eat_candies :
  ∃ (a b d : ℕ), andrey_boris_condition a b ∧ andrey_denis_condition a d ∧ 
                  (total_candies_condition (2 * (12 + 9 + 14)) ∧ 
                   2 * 12 = 24 ∧ 2 * 9 = 18 ∧ 2 * 14 = 28) →
                  (a = 24 ∧ b = 18 ∧ d = 28) :=
by
  sorry

end andrey_boris_denis_eat_candies_l234_234554


namespace parabola_tangent_line_l234_234113

noncomputable def verify_a_value (a : ℝ) : Prop :=
  ∃ x₀ y₀ : ℝ, (y₀ = a * x₀^2) ∧ (x₀ - y₀ - 1 = 0) ∧ (2 * a * x₀ = 1)

theorem parabola_tangent_line :
  verify_a_value (1 / 4) :=
by
  sorry

end parabola_tangent_line_l234_234113


namespace correctness_of_option_C_l234_234736

noncomputable def vec_a : ℝ × ℝ := (-1/2, Real.sqrt 3 / 2)
noncomputable def vec_b : ℝ × ℝ := (Real.sqrt 3 / 2, -1/2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def is_orthogonal (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

theorem correctness_of_option_C :
  is_orthogonal (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2) (vec_a.1 - vec_b.1, vec_a.2 - vec_b.2) :=
by
  sorry

end correctness_of_option_C_l234_234736


namespace number_of_solutions_l234_234267

theorem number_of_solutions :
  ∃ sols: Finset (ℕ × ℕ), (∀ (x y : ℕ), (x, y) ∈ sols ↔ x^2 + y^2 + 2*x*y - 1988*x - 1988*y = 1989 ∧ x > 0 ∧ y > 0)
  ∧ sols.card = 1988 :=
by
  sorry

end number_of_solutions_l234_234267


namespace custom_op_example_l234_234130

-- Definition of the custom operation
def custom_op (a b : ℕ) : ℕ := 4 * a + 5 * b - a * b

-- The proof statement
theorem custom_op_example : custom_op 7 3 = 22 := by
  sorry

end custom_op_example_l234_234130


namespace perimeter_of_similar_triangle_l234_234971

def is_isosceles (a b c : ℕ) : Prop :=
  (a = b) ∨ (b = c) ∨ (a = c)

theorem perimeter_of_similar_triangle (a b c : ℕ) (d e f : ℕ) 
  (h1 : is_isosceles a b c) (h2 : min a b = 15) (h3 : min (min a b) c = 15)
  (h4 : d = 75) (h5 : (d / 15) = e / b) (h6 : f = e) :
  d + e + f = 375 :=
by
  sorry

end perimeter_of_similar_triangle_l234_234971


namespace find_sum_of_p_q_r_s_l234_234305

theorem find_sum_of_p_q_r_s 
    (p q r s : ℝ)
    (h1 : r + s = 12 * p)
    (h2 : r * s = -13 * q)
    (h3 : p + q = 12 * r)
    (h4 : p * q = -13 * s)
    (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
    p + q + r + s = 2028 := 
sorry

end find_sum_of_p_q_r_s_l234_234305


namespace part1_solution_set_part2_range_of_a_l234_234443

/-- Proof that the solution set of the inequality f(x) ≥ 6 when a = 1 is (-∞, -4] ∪ [2, ∞) -/
theorem part1_solution_set (x : ℝ) (h1 : f1 x ≥ 6) : 
  (x ≤ -4 ∨ x ≥ 2) :=
begin
  sorry
end

/-- Proof that the range of values for a if f(x) > -a is (-3/2, ∞) -/
theorem part2_range_of_a (a : ℝ) (h2 : ∀ x, f a x > -a) : 
  (-3/2 < a ∨ 0 ≤ a) :=
begin
  sorry
end

/-- Defining function f(x) = |x - a| + |x + 3| with specific 'a' value for part1 -/
def f1 (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

/-- Defining function f(x) = |x - a| + |x + 3| for part2 -/
def f (a : ℝ) (x : ℝ) : ℝ := abs (x - a) + abs (x + 3)

end part1_solution_set_part2_range_of_a_l234_234443


namespace fixed_cost_to_break_even_l234_234002

def cost_per_handle : ℝ := 0.6
def selling_price_per_handle : ℝ := 4.6
def num_handles_to_break_even : ℕ := 1910

theorem fixed_cost_to_break_even (F : ℝ) (h : F = num_handles_to_break_even * (selling_price_per_handle - cost_per_handle)) :
  F = 7640 := by
  sorry

end fixed_cost_to_break_even_l234_234002


namespace hardcover_books_count_l234_234714

theorem hardcover_books_count
  (h p : ℕ)
  (h_plus_p_eq_10 : h + p = 10)
  (total_cost_eq_250 : 30 * h + 20 * p = 250) :
  h = 5 :=
by
  sorry

end hardcover_books_count_l234_234714


namespace geometric_sequence_decreasing_iff_l234_234914

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

noncomputable def is_decreasing_sequence (a : ℕ → ℝ) : Prop := ∀ n : ℕ, a n > a (n + 1)

theorem geometric_sequence_decreasing_iff (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  (a 0 > a 1 ∧ a 1 > a 2) ↔ is_decreasing_sequence a :=
by
  sorry

end geometric_sequence_decreasing_iff_l234_234914


namespace distribute_tickets_among_people_l234_234401

noncomputable def distribution_ways : ℕ := 84

theorem distribute_tickets_among_people (tickets : Fin 5 → ℕ) (persons : Fin 4 → ℕ)
  (h1 : ∀ p : Fin 4, ∃ t : Fin 5, tickets t = persons p)
  (h2 : ∀ p : Fin 4, ∀ t1 t2 : Fin 5, tickets t1 = persons p ∧ tickets t2 = persons p → (t1.val + 1 = t2.val ∨ t2.val + 1 = t1.val)) :
  ∃ n : ℕ, n = distribution_ways := by
  use 84
  trivial

end distribute_tickets_among_people_l234_234401


namespace num_two_digit_numbers_with_digit_sum_10_l234_234603

theorem num_two_digit_numbers_with_digit_sum_10 : 
  ∃ n, n = 9 ∧ ∀ a b, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a + b = 10 → ∃ m, 10 * a + b = m :=
sorry

end num_two_digit_numbers_with_digit_sum_10_l234_234603


namespace max_integer_k_l234_234262

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then Real.log (1 - x) else 2 / (x - 1)

noncomputable def g (x : ℝ) (k : ℝ) : ℝ := k / (x * x)

theorem max_integer_k :
  ∃ k : ℤ, k = 7 ∧ ∀ p : ℝ, (1 < p) →
    ∃ m n : ℝ, m < 0 ∧ 0 < n ∧ n < p ∧ f p = f m ∧ f m = g n 7 :=
begin
  sorry
end

end max_integer_k_l234_234262


namespace dogwood_trees_tomorrow_l234_234798

def initial_dogwood_trees : Nat := 7
def trees_planted_today : Nat := 3
def final_total_dogwood_trees : Nat := 12

def trees_after_today : Nat := initial_dogwood_trees + trees_planted_today
def trees_planted_tomorrow : Nat := final_total_dogwood_trees - trees_after_today

theorem dogwood_trees_tomorrow :
  trees_planted_tomorrow = 2 :=
by
  sorry

end dogwood_trees_tomorrow_l234_234798


namespace custom_op_example_l234_234131

-- Definition of the custom operation
def custom_op (a b : ℕ) : ℕ := 4 * a + 5 * b - a * b

-- The proof statement
theorem custom_op_example : custom_op 7 3 = 22 := by
  sorry

end custom_op_example_l234_234131


namespace find_point_B_l234_234476

noncomputable def point_A : ℝ × ℝ := (2, 4)

def parabola (x : ℝ) : ℝ := x^2

def tangent_slope (x : ℝ) : ℝ := 2 * x

def normal_slope (x : ℝ) : ℝ := -1 / (tangent_slope x)

def normal_line (x : ℝ) : ℝ × ℝ → ℝ := 
  λ (p : ℝ × ℝ), 
  p.2 + normal_slope p.1 * (x - p.1)

theorem find_point_B :
  let A := (2 : ℝ, 4 : ℝ),
      B := (-9/4 : ℝ, 81/16 : ℝ) in
      normal_line (-9/4) A = parabola (-9/4) → 
      B = (-9/4 : ℝ, 81/16 : ℝ) := 
by
  intros A B h
  sorry

end find_point_B_l234_234476


namespace solve_for_2023_minus_a_minus_2b_l234_234274

theorem solve_for_2023_minus_a_minus_2b (a b : ℝ) (h : 1^2 + a*1 + 2*b = 0) : 2023 - a - 2*b = 2024 := 
by sorry

end solve_for_2023_minus_a_minus_2b_l234_234274


namespace triangle_is_right_angle_l234_234595

theorem triangle_is_right_angle (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 - 12*a - 16*b - 20*c + 200 = 0) : 
  a^2 + b^2 = c^2 :=
by 
  sorry

end triangle_is_right_angle_l234_234595


namespace fraction_percent_of_y_l234_234138

theorem fraction_percent_of_y (y : ℝ) (h : y > 0) : ((2 * y) / 10 + (3 * y) / 10) = 0.5 * y := by
  sorry

end fraction_percent_of_y_l234_234138


namespace set_B_correct_l234_234422

-- Define the set A
def A : Set ℤ := {-1, 0, 1, 2}

-- Define the set B using the given formula
def B : Set ℤ := {y | ∃ x ∈ A, y = x^2 - 2 * x}

-- State the theorem that B is equal to the given set {-1, 0, 3}
theorem set_B_correct : B = {-1, 0, 3} := 
by 
  sorry

end set_B_correct_l234_234422


namespace sailboat_speed_max_power_correct_l234_234503

noncomputable def sailboat_speed_max_power
  (B S ρ v_0 v : ℝ)
  (hS : S = 7)
  (hv0 : v_0 = 6.3)
  (F : ℝ → ℝ := λ v, (B * S * ρ * (v_0 - v) ^ 2) / 2)
  (N : ℝ → ℝ := λ v, F v * v) : Prop :=
  N v = (N (6.3 / 3)) ∧ v = 6.3 / 3

theorem sailboat_speed_max_power_correct :
  sailboat_speed_max_power B S ρ 6.3 2.1 :=
by
  -- Following the above methodology and conditions, we can verify
  -- that the speed v = 6.3 / 3 satisfies the condition when the power
  -- N(v') reaches its maximum value.
  sorry

end sailboat_speed_max_power_correct_l234_234503


namespace tony_lift_ratio_l234_234801

noncomputable def curl_weight := 90
noncomputable def military_press_weight := 2 * curl_weight
noncomputable def squat_weight := 900

theorem tony_lift_ratio : 
  squat_weight / military_press_weight = 5 :=
by
  sorry

end tony_lift_ratio_l234_234801


namespace nancy_kept_tortilla_chips_l234_234166

theorem nancy_kept_tortilla_chips (initial_chips : ℕ) (chips_to_brother : ℕ) (chips_to_sister : ℕ) (remaining_chips : ℕ) 
  (h1 : initial_chips = 22) 
  (h2 : chips_to_brother = 7) 
  (h3 : chips_to_sister = 5) 
  (h_total_given : initial_chips - (chips_to_brother + chips_to_sister) = remaining_chips) :
  remaining_chips = 10 :=
sorry

end nancy_kept_tortilla_chips_l234_234166


namespace largest_sum_faces_l234_234377

theorem largest_sum_faces (a b c d e f : ℕ)
  (h_ab : a + b ≤ 7) (h_ac : a + c ≤ 7) (h_ad : a + d ≤ 7) (h_ae : a + e ≤ 7) (h_af : a + f ≤ 7)
  (h_bc : b + c ≤ 7) (h_bd : b + d ≤ 7) (h_be : b + e ≤ 7) (h_bf : b + f ≤ 7)
  (h_cd : c + d ≤ 7) (h_ce : c + e ≤ 7) (h_cf : c + f ≤ 7)
  (h_de : d + e ≤ 7) (h_df : d + f ≤ 7)
  (h_ef : e + f ≤ 7) :
  ∃ x y z, 
  ((x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e ∨ x = f) ∧ 
   (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e ∨ y = f) ∧ 
   (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e ∨ z = f)) ∧ 
  (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
  (x + y ≤ 7) ∧ (y + z ≤ 7) ∧ (x + z ≤ 7) ∧
  (x + y + z = 9) :=
sorry

end largest_sum_faces_l234_234377


namespace range_of_m_l234_234263

theorem range_of_m (m : Real) :
  (∀ x y : Real, 0 < x ∧ x < y ∧ y < (π / 2) → 
    (m - 2 * Real.sin x) / Real.cos x > (m - 2 * Real.sin y) / Real.cos y) →
  m ≤ 2 := 
sorry

end range_of_m_l234_234263


namespace union_M_N_l234_234451

def M := {x : ℝ | x^2 - 4*x + 3 ≤ 0}
def N := {x : ℝ | Real.log x / Real.log 2 ≤ 1}

theorem union_M_N :
  M ∪ N = {x : ℝ | 0 < x ∧ x ≤ 3} := by
  sorry

end union_M_N_l234_234451


namespace solve_quadratic_equation1_solve_quadratic_equation2_l234_234173

theorem solve_quadratic_equation1 (x : ℝ) : x^2 - 4 * x - 1 = 0 ↔ x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 := 
by sorry

theorem solve_quadratic_equation2 (x : ℝ) : (x + 3) * (x - 3) = 3 * (x + 3) ↔ x = -3 ∨ x = 6 :=
by sorry

end solve_quadratic_equation1_solve_quadratic_equation2_l234_234173


namespace trig_matrix_det_zero_l234_234703

noncomputable def trig_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    [Real.sin (30 * Real.pi / 180), Real.sin (60 * Real.pi / 180), Real.sin (90 * Real.pi / 180)],
    [Real.sin (150 * Real.pi / 180), Real.sin (180 * Real.pi / 180), Real.sin (210 * Real.pi / 180)],
    [Real.sin (270 * Real.pi / 180), Real.sin (300 * Real.pi / 180), Real.sin (330 * Real.pi / 180)]
  ]

theorem trig_matrix_det_zero : trig_matrix.det = 0 := by
  sorry

end trig_matrix_det_zero_l234_234703


namespace rachel_homework_l234_234640

theorem rachel_homework : 5 + 2 = 7 := by
  sorry

end rachel_homework_l234_234640


namespace part1_solution_part2_solution_l234_234439

-- Proof problem for Part (1)
theorem part1_solution (x : ℝ) (a : ℝ) (h : a = 1) : 
  (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

-- Proof problem for Part (2)
theorem part2_solution (a : ℝ) : 
  (∀ x : ℝ, (|x - a| + |x + 3|) > -a) ↔ a > -3 / 2 :=
by
  sorry

end part1_solution_part2_solution_l234_234439


namespace collinear_values_k_l234_234190

/-- Define the vectors OA, OB, and OC using the given conditions. -/
def vectorOA (k : ℝ) : ℝ × ℝ := (k, 12)
def vectorOB : ℝ × ℝ := (4, 5)
def vectorOC (k : ℝ) : ℝ × ℝ := (10, k)

/-- Define vectors AB and BC using vector subtraction. -/
def vectorAB (k : ℝ) : ℝ × ℝ := (4 - k, -7)
def vectorBC (k : ℝ) : ℝ × ℝ := (6, k - 5)

/-- Collinearity condition for vectors AB and BC. -/
def collinear (k : ℝ) : Prop :=
  (4 - k) * (k - 5) + 42 = 0

/-- Prove that the value of k is 11 or -2 given the collinearity condition. -/
theorem collinear_values_k : ∀ k : ℝ, collinear k → (k = 11 ∨ k = -2) :=
by
  intros k h
  sorry

end collinear_values_k_l234_234190


namespace tetrahedron_a_exists_tetrahedron_b_not_exists_l234_234619

/-- Part (a): There exists a tetrahedron with two edges shorter than 1 cm,
    and the other four edges longer than 1 km. -/
theorem tetrahedron_a_exists : 
  ∃ (a b c d : ℝ), 
    ((a < 1 ∧ b < 1 ∧ 1000 < c ∧ 1000 < d ∧ 1000 < (a + c) ∧ 1000 < (b + d)) ∧ 
     a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) := 
sorry

/-- Part (b): There does not exist a tetrahedron with four edges shorter than 1 cm,
    and the other two edges longer than 1 km. -/
theorem tetrahedron_b_not_exists : 
  ¬ ∃ (a b c d : ℝ), 
    ((a < 1 ∧ b < 1 ∧ c < 1 ∧ d < 1 ∧ 1000 < (a + c) ∧ 1000 < (b + d)) ∧ 
     a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ) := 
sorry

end tetrahedron_a_exists_tetrahedron_b_not_exists_l234_234619


namespace barbara_wins_iff_multiple_of_6_l234_234150

-- Define the conditions and the statement to be proved
theorem barbara_wins_iff_multiple_of_6 (n : ℕ) (h : n > 1) :
  (∃ a b : ℕ, a > 0 ∧ b > 1 ∧ (b ∣ a ∨ a ∣ b) ∧ ∀ k ≤ 50, (b + k = n ∨ b - k = n)) ↔ 6 ∣ n :=
sorry

end barbara_wins_iff_multiple_of_6_l234_234150


namespace not_collinear_C_vector_decomposition_l234_234264

namespace VectorProof

open Function

structure Vector2 where
  x : ℝ
  y : ℝ

def add (v1 v2 : Vector2) : Vector2 := ⟨v1.x + v2.x, v1.y + v2.y⟩
def scale (c : ℝ) (v : Vector2) : Vector2 := ⟨c * v.x, c * v.y⟩

def collinear (v1 v2 : Vector2) : Prop :=
  ∃ k : ℝ, v2 = scale k v1

def vector_a : Vector2 := ⟨3, 4⟩
def e₁_C : Vector2 := ⟨-1, 2⟩
def e₂_C : Vector2 := ⟨3, -1⟩

theorem not_collinear_C :
  ¬ collinear e₁_C e₂_C :=
sorry

theorem vector_decomposition :
  ∃ (x y : ℝ), vector_a = add (scale x e₁_C) (scale y e₂_C) :=
sorry

end VectorProof

end not_collinear_C_vector_decomposition_l234_234264


namespace parabola_vertex_coordinate_l234_234726

theorem parabola_vertex_coordinate :
  ∀ x_P : ℝ, 
  (P : ℝ × ℝ) → 
  (P = (x_P, 1/2 * x_P^2)) → 
  (dist P (0, 1/2) = 3) →
  P.2 = 5 / 2 :=
by sorry

end parabola_vertex_coordinate_l234_234726


namespace trapezoid_circle_center_l234_234026

theorem trapezoid_circle_center 
  (EF GH : ℝ)
  (FG HE : ℝ)
  (p q : ℕ) 
  (rel_prime : Nat.gcd p q = 1)
  (EQ GH : ℝ)
  (h1 : EF = 105)
  (h2 : FG = 57)
  (h3 : GH = 22)
  (h4 : HE = 80)
  (h5 : EQ = p / q)
  (h6 : p = 10)
  (h7 : q = 1) :
  p + q = 11 :=
by
  sorry

end trapezoid_circle_center_l234_234026


namespace sailboat_speed_max_power_l234_234502

-- Define constants for the problem.
def B : ℝ := sorry -- Aerodynamic force coefficient (to be provided)
def ρ : ℝ := sorry -- Air density (to be provided)
def S : ℝ := 7 -- sail area in m²
def v0 : ℝ := 6.3 -- wind speed in m/s

-- Define the force formula
def F (v : ℝ) : ℝ := (B * S * ρ * (v0 - v)^2) / 2

-- Define the power formula
def N (v : ℝ) : ℝ := F v * v

-- Define the condition that the power reaches its maximum value at some speed
def N_max : ℝ := sorry -- maximum instantaneous power (to be provided)

-- The proof statement that the speed of the sailboat when the power is maximized is v0 / 3
theorem sailboat_speed_max_power : ∃ v : ℝ, (N v = N_max ∧ v = v0 / 3) := 
  sorry

end sailboat_speed_max_power_l234_234502


namespace batsman_boundaries_l234_234686

theorem batsman_boundaries
  (total_runs : ℕ)
  (boundaries : ℕ)
  (sixes : ℕ)
  (runs_by_running : ℕ)
  (runs_by_sixes : ℕ)
  (runs_by_boundaries : ℕ)
  (half_runs : ℕ)
  (sixes_runs : ℕ)
  (boundaries_runs : ℕ)
  (total_runs_eq : total_runs = 120)
  (sixes_eq : sixes = 8)
  (half_total_eq : half_runs = total_runs / 2)
  (runs_by_running_eq : runs_by_running = half_runs)
  (sixes_runs_eq : runs_by_sixes = sixes * 6)
  (boundaries_runs_eq : runs_by_boundaries = total_runs - runs_by_running - runs_by_sixes)
  (boundaries_eq : boundaries_runs = boundaries * 4) :
  boundaries = 3 :=
by
  sorry

end batsman_boundaries_l234_234686


namespace distance_between_intersections_l234_234793

-- Given conditions
def line_eq (x : ℝ) : ℝ := 5
def quad_eq (x : ℝ) : ℝ := 5 * x^2 + 2 * x - 2

-- The proof statement
theorem distance_between_intersections : 
  ∃ (C D : ℝ), line_eq C = quad_eq C ∧ line_eq D = quad_eq D ∧ abs (C - D) = 2.4 :=
by
  -- We will later fill in the proof here
  sorry

end distance_between_intersections_l234_234793


namespace solution_point_satisfies_inequalities_l234_234818

theorem solution_point_satisfies_inequalities:
  let x := -1/3
  let y := 2/3
  11 * x^2 + 8 * x * y + 8 * y^2 ≤ 3 ∧ x - 4 * y ≤ -3 :=
by
  let x := -1/3
  let y := 2/3
  sorry

end solution_point_satisfies_inequalities_l234_234818


namespace barbara_shopping_l234_234387

theorem barbara_shopping :
  let total_paid := 56
  let tuna_cost := 5 * 2
  let water_cost := 4 * 1.5
  let other_goods_cost := total_paid - tuna_cost - water_cost
  other_goods_cost = 40 :=
by
  sorry

end barbara_shopping_l234_234387


namespace value_of_expression_l234_234271

theorem value_of_expression (a b : ℤ) (h : 1^2 + a * 1 + 2 * b = 0) : 2023 - a - 2 * b = 2024 :=
by
  sorry

end value_of_expression_l234_234271


namespace gcd_55555555_111111111_l234_234631

/-- Let \( m = 55555555 \) and \( n = 111111111 \).
We want to prove that the greatest common divisor (gcd) of \( m \) and \( n \) is 1. -/
theorem gcd_55555555_111111111 :
  let m := 55555555
  let n := 111111111
  Nat.gcd m n = 1 :=
by
  sorry

end gcd_55555555_111111111_l234_234631


namespace rental_cost_equal_mileage_l234_234495

theorem rental_cost_equal_mileage:
  ∃ x : ℝ, (17.99 + 0.18 * x = 18.95 + 0.16 * x) ∧ x = 48 := 
by
  sorry

end rental_cost_equal_mileage_l234_234495


namespace perimeter_smallest_square_l234_234193

theorem perimeter_smallest_square 
  (d : ℝ) (side_largest : ℝ)
  (h1 : d = 3) 
  (h2 : side_largest = 22) : 
  4 * (side_largest - 2 * d - 2 * d) = 40 := by
  sorry

end perimeter_smallest_square_l234_234193


namespace angle_in_parallelogram_l234_234143

theorem angle_in_parallelogram (EFGH : Parallelogram) (angle_EFG angle_FGH : ℝ)
  (h1 : angle_EFG = angle_FGH + 90) : angle_EHG = 45 :=
by sorry

end angle_in_parallelogram_l234_234143


namespace gcd_m_n_l234_234633

noncomputable def m := 55555555
noncomputable def n := 111111111

theorem gcd_m_n : Int.gcd m n = 1 := by
  sorry

end gcd_m_n_l234_234633


namespace min_coins_for_any_amount_below_dollar_l234_234521

-- Definitions of coin values
def penny := 1
def nickel := 5
def dime := 10
def half_dollar := 50

-- Statement: The minimum number of coins required to pay any amount less than a dollar
theorem min_coins_for_any_amount_below_dollar :
  ∃ (n : ℕ), n = 11 ∧
  (∀ (amount : ℕ), 1 ≤ amount ∧ amount < 100 →
   ∃ (a b c d : ℕ), amount = a * penny + b * nickel + c * dime + d * half_dollar ∧ 
   a + b + c + d ≤ n) :=
sorry

end min_coins_for_any_amount_below_dollar_l234_234521


namespace original_square_area_l234_234654

theorem original_square_area {x y : ℕ} (h1 : y ≠ 1)
  (h2 : x^2 = 24 + y^2) : x^2 = 49 :=
sorry

end original_square_area_l234_234654


namespace sale_in_fifth_month_l234_234985

theorem sale_in_fifth_month (sale1 sale2 sale3 sale4 sale6 : ℕ) (avg : ℕ) (months : ℕ) (total_sales : ℕ)
    (known_sales : sale1 = 6335 ∧ sale2 = 6927 ∧ sale3 = 6855 ∧ sale4 = 7230 ∧ sale6 = 5091)
    (avg_condition : avg = 6500)
    (months_condition : months = 6)
    (total_sales_condition : total_sales = avg * months) :
    total_sales - (sale1 + sale2 + sale3 + sale4 + sale6) = 6562 :=
by
  sorry

end sale_in_fifth_month_l234_234985


namespace laser_total_distance_l234_234826

noncomputable def laser_path_distance : ℝ :=
  let A := (2, 4)
  let B := (2, -4)
  let C := (-2, -4)
  let D := (8, 4)
  let distance (p q : ℝ × ℝ) : ℝ :=
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance A B + distance B C + distance C D

theorem laser_total_distance :
  laser_path_distance = 12 + 2 * Real.sqrt 41 :=
by sorry

end laser_total_distance_l234_234826


namespace original_proposition_false_converse_false_inverse_false_contrapositive_false_l234_234533

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop := 
  (a * b ≤ 0) → (a ≤ 0 ∨ b ≤ 0)

-- Define the converse
def converse (a b : ℝ) : Prop := 
  (a ≤ 0 ∨ b ≤ 0) → (a * b ≤ 0)

-- Define the inverse
def inverse (a b : ℝ) : Prop := 
  (a * b > 0) → (a > 0 ∧ b > 0)

-- Define the contrapositive
def contrapositive (a b : ℝ) : Prop := 
  (a > 0 ∧ b > 0) → (a * b > 0)

-- Prove that the original proposition is false
theorem original_proposition_false : ∀ (a b : ℝ), ¬ original_proposition a b :=
by sorry

-- Prove that the converse is false
theorem converse_false : ∀ (a b : ℝ), ¬ converse a b :=
by sorry

-- Prove that the inverse is false
theorem inverse_false : ∀ (a b : ℝ), ¬ inverse a b :=
by sorry

-- Prove that the contrapositive is false
theorem contrapositive_false : ∀ (a b : ℝ), ¬ contrapositive a b :=
by sorry

end original_proposition_false_converse_false_inverse_false_contrapositive_false_l234_234533


namespace cubic_inches_in_two_cubic_feet_l234_234910

theorem cubic_inches_in_two_cubic_feet (conv : 1 = 12) : 2 * (12 * 12 * 12) = 3456 :=
by
  sorry

end cubic_inches_in_two_cubic_feet_l234_234910


namespace range_of_m_l234_234397

-- Definitions of the propositions
def p (m : ℝ) : Prop := ∀ x : ℝ, |x| + |x + 1| > m
def q (m : ℝ) : Prop := ∀ x > 2, 2 * x - 2 * m > 0

-- The main theorem statement
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l234_234397


namespace part1_part2_l234_234446

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := sorry

theorem part2 (a : ℝ) : 
  (∀ x, f x a > -a) ↔ (a > -3/2) := sorry

end part1_part2_l234_234446


namespace tim_total_points_l234_234992

-- Definitions based on the conditions
def points_single : ℕ := 1000
def points_tetris : ℕ := 8 * points_single
def singles_scored : ℕ := 6
def tetrises_scored : ℕ := 4

-- Theorem stating the total points scored by Tim
theorem tim_total_points : singles_scored * points_single + tetrises_scored * points_tetris = 38000 := by
  sorry

end tim_total_points_l234_234992


namespace union_of_sets_l234_234893

def A : Set Int := {-1, 2, 3, 5}
def B : Set Int := {2, 4, 5}

theorem union_of_sets :
  A ∪ B = {-1, 2, 3, 4, 5} := by
  sorry

end union_of_sets_l234_234893


namespace polygon_number_of_sides_l234_234836

theorem polygon_number_of_sides (P : ℝ) (L : ℝ) (n : ℕ) : P = 180 ∧ L = 15 ∧ n = P / L → n = 12 := by
  sorry

end polygon_number_of_sides_l234_234836


namespace can_construct_prism_with_fewer_than_20_shapes_l234_234667

/-
  We have 5 congruent unit cubes glued together to form complex shapes.
  4 of these cubes form a 4-unit high prism, and the fifth is attached to one of the inner cubes with a full face.
  Prove that we can construct a solid rectangular prism using fewer than 20 of these shapes.
-/

theorem can_construct_prism_with_fewer_than_20_shapes :
  ∃ (n : ℕ), n < 20 ∧ (∃ (length width height : ℕ), length * width * height = 5 * n) :=
sorry

end can_construct_prism_with_fewer_than_20_shapes_l234_234667


namespace find_s_when_t_is_64_l234_234597

theorem find_s_when_t_is_64 (s : ℝ) (t : ℝ) (h1 : t = 8 * s^3) (h2 : t = 64) : s = 2 :=
by
  -- Proof will be written here
  sorry

end find_s_when_t_is_64_l234_234597


namespace line_AB_bisects_segment_DE_l234_234499

variables {A B C D E : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
  {trapezoid : A × B × C × D} (AC CD : Prop) (BD_sym : Prop) (intersect_E : Prop)
  (line_AB : Prop) (bisects_DE : Prop)

-- Given a trapezoid ABCD
def is_trapezoid (A B C D : Type) : Prop := sorry

-- Given the diagonal AC is equal to the side CD
def diagonal_eq_leg (AC CD : Prop) : Prop := sorry

-- Given line BD is symmetric with respect to AD intersects AC at point E
def symmetric_line_intersect (BD_sym AD AC E : Prop) : Prop := sorry

-- Prove that line AB bisects segment DE
theorem line_AB_bisects_segment_DE
  (h_trapezoid : is_trapezoid A B C D)
  (h_diagonal_eq_leg : diagonal_eq_leg AC CD)
  (h_symmetric_line_intersect : symmetric_line_intersect BD_sym (sorry : Prop) AC intersect_E)
  (h_line_AB : line_AB) :
  bisects_DE := sorry

end line_AB_bisects_segment_DE_l234_234499


namespace ana_multiplied_numbers_l234_234218

theorem ana_multiplied_numbers (x : ℕ) (y : ℕ) 
    (h_diff : y = x + 202) 
    (h_mistake : x * y - 1000 = 288 * x + 67) :
    x = 97 ∧ y = 299 :=
sorry

end ana_multiplied_numbers_l234_234218


namespace gcd_g_y_l234_234112

noncomputable def g (y : ℕ) : ℕ := (3 * y + 5) * (6 * y + 7) * (10 * y + 3) * (5 * y + 11) * (y + 7)

theorem gcd_g_y (y : ℕ) (h : ∃ k : ℕ, y = 18090 * k) : Nat.gcd (g y) y = 8085 := 
sorry

end gcd_g_y_l234_234112


namespace nancy_kept_tortilla_chips_l234_234165

theorem nancy_kept_tortilla_chips (initial_chips : ℕ) (chips_to_brother : ℕ) (chips_to_sister : ℕ) (remaining_chips : ℕ) 
  (h1 : initial_chips = 22) 
  (h2 : chips_to_brother = 7) 
  (h3 : chips_to_sister = 5) 
  (h_total_given : initial_chips - (chips_to_brother + chips_to_sister) = remaining_chips) :
  remaining_chips = 10 :=
sorry

end nancy_kept_tortilla_chips_l234_234165


namespace part1_l234_234978

theorem part1 (m n p : ℝ) (h1 : m > n) (h2 : n > 0) (h3 : p > 0) : 
  (n / m) < (n + p) / (m + p) := 
sorry

end part1_l234_234978


namespace calculate_bags_l234_234925

theorem calculate_bags (num_horses : ℕ) (feedings_per_day : ℕ) (food_per_feeding : ℕ) (days : ℕ) (bag_weight : ℕ):
  num_horses = 25 → 
  feedings_per_day = 2 → 
  food_per_feeding = 20 → 
  days = 60 → 
  bag_weight = 1000 → 
  (num_horses * feedings_per_day * food_per_feeding * days) / bag_weight = 60 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  exact (60 : ℕ)
  sorry

end calculate_bags_l234_234925


namespace exists_triangle_cut_into_2005_congruent_l234_234007

theorem exists_triangle_cut_into_2005_congruent :
  ∃ (Δ : Type) (a b c : Δ → ℝ )
  (h₁ : a^2 + b^2 = c^2) (h₂ : a * b / 2 = 2005 / 2),
  true :=
sorry

end exists_triangle_cut_into_2005_congruent_l234_234007


namespace inequality_addition_l234_234885

theorem inequality_addition (a b : ℝ) (h : a > b) : a + 3 > b + 3 := by
  sorry

end inequality_addition_l234_234885


namespace tom_has_7_blue_tickets_l234_234024

def number_of_blue_tickets_needed_for_bible := 10 * 10 * 10
def toms_current_yellow_tickets := 8
def toms_current_red_tickets := 3
def toms_needed_blue_tickets := 163

theorem tom_has_7_blue_tickets : 
  (number_of_blue_tickets_needed_for_bible - 
    (toms_current_yellow_tickets * 10 * 10 + 
     toms_current_red_tickets * 10 + 
     toms_needed_blue_tickets)) = 7 :=
by
  -- Proof can be provided here
  sorry

end tom_has_7_blue_tickets_l234_234024


namespace alcohol_solution_problem_l234_234325

theorem alcohol_solution_problem (x_vol y_vol : ℚ) (x_alcohol y_alcohol target_alcohol : ℚ) (target_vol : ℚ) :
  x_vol = 250 ∧ x_alcohol = 10/100 ∧ y_alcohol = 30/100 ∧ target_alcohol = 25/100 ∧ target_vol = 250 + y_vol →
  (x_alcohol * x_vol + y_alcohol * y_vol = target_alcohol * target_vol) →
  y_vol = 750 :=
by
  sorry

end alcohol_solution_problem_l234_234325


namespace find_general_term_arithmetic_sequence_l234_234743

-- Definitions needed
variable {a_n : ℕ → ℚ}
variable {S_n : ℕ → ℚ}

-- The main theorem to prove
theorem find_general_term_arithmetic_sequence 
  (h1 : a_n 4 - a_n 2 = 4)
  (h2 : S_n 3 = 9)
  (h3 : ∀ n : ℕ, S_n n = n / 2 * (2 * (a_n 1) + (n - 1) * (a_n 2 - a_n 1))) :
  (∀ n : ℕ, a_n n = 2 * n - 1) :=
by
  sorry

end find_general_term_arithmetic_sequence_l234_234743


namespace sphere_radius_eq_3_l234_234660

theorem sphere_radius_eq_3 (r : ℝ) (h : (4/3) * Real.pi * r^3 = 4 * Real.pi * r^2) : r = 3 :=
sorry

end sphere_radius_eq_3_l234_234660


namespace best_of_five_advantageous_l234_234380

theorem best_of_five_advantageous (p : ℝ) (h : p > 0.5) :
    let p1 := p^2 + 2 * p^2 * (1 - p)
    let p2 := p^3 + 3 * p^3 * (1 - p) + 6 * p^3 * (1 - p)^2
    p2 > p1 :=
by 
    let p1 := p^2 + 2 * p^2 * (1 - p)
    let p2 := p^3 + 3 * p^3 * (1 - p) + 6 * p^3 * (1 - p)^2
    sorry -- an actual proof would go here

end best_of_five_advantageous_l234_234380


namespace trigonometric_inequality_l234_234638

theorem trigonometric_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  0 < (1 / (Real.sin x)^2) - (1 / x^2) ∧ (1 / (Real.sin x)^2) - (1 / x^2) < 1 := 
sorry

end trigonometric_inequality_l234_234638


namespace grape_juice_percentage_after_addition_l234_234608

def initial_mixture_volume : ℝ := 40
def initial_grape_juice_percentage : ℝ := 0.10
def added_grape_juice_volume : ℝ := 10

theorem grape_juice_percentage_after_addition :
  ((initial_mixture_volume * initial_grape_juice_percentage + added_grape_juice_volume) /
  (initial_mixture_volume + added_grape_juice_volume)) * 100 = 28 :=
by 
  sorry

end grape_juice_percentage_after_addition_l234_234608


namespace longer_diagonal_of_rhombus_l234_234838

theorem longer_diagonal_of_rhombus (s d1 d2 : ℝ) 
  (hs : s = 65) (hd1 : d1 = 72) :
  d2 = 108 :=
by 
  -- Definitions
  have a : ℝ := 36                                 -- Half of shorter diagonal
  have b : ℝ := Math.sqrt(2929)                    -- Half of longer diagonal calculated
  calc 
    d2 = 2 * b : by simp [b]
    ... = 108 : by norm_num -- Final calculation to get 108

end sorry

end longer_diagonal_of_rhombus_l234_234838


namespace min_m_plus_n_l234_234040

theorem min_m_plus_n (m n : ℕ) (h₁ : m > n) (h₂ : 4^m + 4^n % 100 = 0) : m + n = 7 :=
sorry

end min_m_plus_n_l234_234040


namespace multiplication_of_positive_and_negative_l234_234076

theorem multiplication_of_positive_and_negative :
  9 * (-3) = -27 := by
  sorry

end multiplication_of_positive_and_negative_l234_234076


namespace decimal_equiv_half_squared_l234_234045

theorem decimal_equiv_half_squared :
  ((1 / 2 : ℝ) ^ 2) = 0.25 := by
  sorry

end decimal_equiv_half_squared_l234_234045


namespace prism_aligns_l234_234972

theorem prism_aligns (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ prism_dimensions = (a * 5, b * 10, c * 20) :=
by
  sorry

end prism_aligns_l234_234972


namespace value_of_x_l234_234973

theorem value_of_x : ∃ (x : ℚ), (10 - 2 * x) ^ 2 = 4 * x ^ 2 + 20 * x ∧ x = 5 / 3 :=
by
  sorry

end value_of_x_l234_234973


namespace problem_solution_l234_234906

-- Define the variables and the conditions
variable (a b c : ℝ)
axiom h1 : a^2 + 2 * b = 7
axiom h2 : b^2 - 2 * c = -1
axiom h3 : c^2 - 6 * a = -17

-- State the theorem to be proven
theorem problem_solution : a + b + c = 3 := 
by sorry

end problem_solution_l234_234906


namespace coin_sum_even_odd_l234_234859

theorem coin_sum_even_odd (S : ℕ) (h : S > 1) : 
  (∃ even_count, (even_count : ℕ) ∈ [0, 2, S]) ∧ (∃ odd_count, ((odd_count : ℕ) - 1) ∈ [0, 2, S]) :=
  sorry

end coin_sum_even_odd_l234_234859


namespace shift_down_two_units_l234_234651

theorem shift_down_two_units (x : ℝ) : 
  (y = 2 * x) → (y - 2 = 2 * x - 2) := by
sorry

end shift_down_two_units_l234_234651


namespace part1_part2_l234_234448

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := sorry

theorem part2 (a : ℝ) : 
  (∀ x, f x a > -a) ↔ (a > -3/2) := sorry

end part1_part2_l234_234448


namespace find_circle_equation_l234_234258

-- Define the conditions on the circle
def passes_through_points (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ c : ℝ × ℝ, ∃ r : ℝ, (c = center ∧ r = radius) ∧ 
  dist (0, 2) c = r ∧ dist (0, 4) c = r

def lies_on_line (center : ℝ × ℝ) : Prop :=
  2 * center.1 - center.2 - 1 = 0

-- Define the problem
theorem find_circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
  passes_through_points center radius ∧ lies_on_line center ∧ 
  (∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = radius^2 
  ↔ (x - 2)^2 + (y - 3)^2 = 5) :=
sorry

end find_circle_equation_l234_234258


namespace mixture_concentration_l234_234664

-- Definitions reflecting the given conditions
def sol1_concentration : ℝ := 0.30
def sol1_volume : ℝ := 8

def sol2_concentration : ℝ := 0.50
def sol2_volume : ℝ := 5

def sol3_concentration : ℝ := 0.70
def sol3_volume : ℝ := 7

-- The proof problem stating that the resulting concentration is 49%
theorem mixture_concentration :
  (sol1_concentration * sol1_volume + sol2_concentration * sol2_volume + sol3_concentration * sol3_volume) /
  (sol1_volume + sol2_volume + sol3_volume) * 100 = 49 :=
by
  sorry

end mixture_concentration_l234_234664


namespace largest_divisor_of_Pn_for_even_n_l234_234625

def P (n : ℕ) : ℕ := 
  (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9)

theorem largest_divisor_of_Pn_for_even_n : 
  ∀ (n : ℕ), (0 < n ∧ n % 2 = 0) → ∃ d, d = 15 ∧ d ∣ P n :=
by
  intro n h
  sorry

end largest_divisor_of_Pn_for_even_n_l234_234625


namespace solve_eq1_solve_eq2_l234_234492

theorem solve_eq1 (x : ℝ) :
  3 * x^2 - 11 * x + 9 = 0 ↔ x = (11 + Real.sqrt 13) / 6 ∨ x = (11 - Real.sqrt 13) / 6 :=
by
  sorry

theorem solve_eq2 (x : ℝ) :
  5 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 / 2 :=
by
  sorry

end solve_eq1_solve_eq2_l234_234492


namespace sum_of_distinct_prime_factors_315_l234_234529

theorem sum_of_distinct_prime_factors_315 : 
  ∃ factors : List ℕ, factors = [3, 5, 7] ∧ 315 = 3 * 3 * 5 * 7 ∧ factors.sum = 15 :=
by
  sorry

end sum_of_distinct_prime_factors_315_l234_234529


namespace gum_pack_size_is_5_l234_234453
noncomputable def find_gum_pack_size (x : ℕ) : Prop :=
  let cherry_initial := 25
  let grape_initial := 40
  let cherry_lost := cherry_initial - 2 * x
  let grape_found := grape_initial + 4 * x
  (cherry_lost * grape_found) = (cherry_initial * grape_initial)

theorem gum_pack_size_is_5 : find_gum_pack_size 5 :=
by
  sorry

end gum_pack_size_is_5_l234_234453


namespace right_handed_players_total_l234_234003

-- Definitions of the given quantities
def total_players : ℕ := 70
def throwers : ℕ := 49
def non_throwers : ℕ := total_players - throwers
def one_third_non_throwers : ℕ := non_throwers / 3
def left_handed_non_throwers : ℕ := one_third_non_throwers
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers
def right_handed_throwers : ℕ := throwers
def total_right_handed : ℕ := right_handed_throwers + right_handed_non_throwers

-- The theorem stating the main proof goal
theorem right_handed_players_total (h1 : total_players = 70)
                                   (h2 : throwers = 49)
                                   (h3 : total_players - throwers = non_throwers)
                                   (h4 : non_throwers = 21) -- derived from the above
                                   (h5 : non_throwers / 3 = left_handed_non_throwers)
                                   (h6 : non_throwers - left_handed_non_throwers = right_handed_non_throwers)
                                   (h7 : right_handed_throwers = throwers)
                                   (h8 : total_right_handed = right_handed_throwers + right_handed_non_throwers) :
  total_right_handed = 63 := sorry

end right_handed_players_total_l234_234003


namespace tabitha_color_start_l234_234874

def add_color_each_year (n : ℕ) : ℕ := n + 1

theorem tabitha_color_start 
  (age_start age_now future_colors years_future current_colors : ℕ)
  (h1 : age_start = 15)
  (h2 : age_now = 18)
  (h3 : years_future = 3)
  (h4 : age_now + years_future = 21)
  (h5 : future_colors = 8)
  (h6 : future_colors - years_future = current_colors + 3)
  (h7 : current_colors = 5)
  : age_start + (current_colors - (age_now - age_start)) = 3 := 
by
  sorry

end tabitha_color_start_l234_234874


namespace probability_three_red_before_two_green_l234_234462

noncomputable def probability_red_chips_drawn_before_green (red_chips green_chips : ℕ) (total_chips : ℕ) : ℚ := sorry

theorem probability_three_red_before_two_green 
    (red_chips green_chips : ℕ) (total_chips : ℕ)
    (h_red : red_chips = 3) (h_green : green_chips = 2) 
    (h_total: total_chips = red_chips + green_chips) :
  probability_red_chips_drawn_before_green red_chips green_chips total_chips = 3 / 10 :=
  sorry

end probability_three_red_before_two_green_l234_234462


namespace typing_difference_l234_234073

theorem typing_difference (initial_speed after_speed : ℕ) (time_interval : ℕ) (h_initial : initial_speed = 10) 
  (h_after : after_speed = 8) (h_time : time_interval = 5) : 
  (initial_speed * time_interval) - (after_speed * time_interval) = 10 := 
by 
  sorry

end typing_difference_l234_234073


namespace molecular_weight_calculated_l234_234807

def atomic_weight_Ba : ℚ := 137.33
def atomic_weight_O  : ℚ := 16.00
def atomic_weight_H  : ℚ := 1.01

def molecular_weight_compound : ℚ :=
  (1 * atomic_weight_Ba) + (2 * atomic_weight_O) + (2 * atomic_weight_H)

theorem molecular_weight_calculated :
  molecular_weight_compound = 171.35 :=
by {
  sorry
}

end molecular_weight_calculated_l234_234807


namespace pages_remaining_total_l234_234231

-- Define the conditions
def total_pages_book1 : ℕ := 563
def read_pages_book1 : ℕ := 147

def total_pages_book2 : ℕ := 849
def read_pages_book2 : ℕ := 389

def total_pages_book3 : ℕ := 700
def read_pages_book3 : ℕ := 134

-- The theorem to be proved
theorem pages_remaining_total :
  (total_pages_book1 - read_pages_book1) + 
  (total_pages_book2 - read_pages_book2) + 
  (total_pages_book3 - read_pages_book3) = 1442 := 
by
  sorry

end pages_remaining_total_l234_234231


namespace angle_measure_of_P_l234_234324

noncomputable def measure_angle_P (ABCDE : EuclideanGeometry.ConvexPolygon 5) (AB DE : EuclideanGeometry.Line) (P : EuclideanGeometry.Point) 
  (h1 : ABCDE.IsRegular) (h2 : AB ∈ ABCDE.Sides) (h3 : DE ∈ ABCDE.Sides) (h4 : EuclideanGeometry.IsExtension AB P) (h5 : EuclideanGeometry.IsExtension DE P)
  : Real :=
  36

theorem angle_measure_of_P (ABCDE : EuclideanGeometry.ConvexPolygon 5) (AB DE : EuclideanGeometry.Line) (P : EuclideanGeometry.Point) 
  (h1 : ABCDE.IsRegular) (h2 : AB ∈ ABCDE.Sides) (h3 : DE ∈ ABCDE.Sides) (h4 : EuclideanGeometry.IsExtension AB P) (h5 : EuclideanGeometry.IsExtension DE P) 
  : measure_angle_P ABCDE AB DE P h1 h2 h3 h4 h5 = 36 :=
sorry

end angle_measure_of_P_l234_234324


namespace inverse_implies_negation_l234_234134

-- Let's define p as a proposition
variable (p : Prop)

-- The inverse of a proposition p, typically the implication of not p implies not q
def inverse (p q : Prop) := ¬p → ¬q

-- The negation of a proposition p is just ¬p
def negation (p : Prop) := ¬p

-- The math problem statement. Prove that if the inverse of p is true, the negation of p is true.
theorem inverse_implies_negation (q : Prop) (h : inverse p q) : negation q := by
  sorry

end inverse_implies_negation_l234_234134


namespace statement_two_statement_three_l234_234729

section
variables {R : Type*} [Field R]
variables (a b c p q : R)
noncomputable def f (x : R) := a * x^2 + b * x + c

-- Statement ②
theorem statement_two (hpq : f a b c p = f a b c q) (hpq_neq : p ≠ q) : 
  f a b c (p + q) = c :=
sorry

-- Statement ③
theorem statement_three (hf : f a b c (p + q) = c) (hpq_neq : p ≠ q) : 
  p + q = 0 ∨ f a b c p = f a b c q :=
sorry

end

end statement_two_statement_three_l234_234729


namespace min_degree_g_l234_234668

theorem min_degree_g (f g h : Polynomial ℝ) (hf : f.degree = 8) (hh : h.degree = 9) (h_eq : 3 * f + 4 * g = h) : g.degree ≥ 9 :=
sorry

end min_degree_g_l234_234668


namespace arithmetic_geom_sequences_l234_234735

theorem arithmetic_geom_sequences
  (a : ℕ → ℤ)
  (b : ℕ → ℤ)
  (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_geom : ∃ q, ∀ n, b (n + 1) = b n * q)
  (h1 : a 2 + a 3 = 14)
  (h2 : a 4 - a 1 = 6)
  (h3 : b 2 = a 1)
  (h4 : b 3 = a 3) :
  (∀ n, a n = 2 * n + 2) ∧ (∃ m, b 6 = a m ∧ m = 31) := sorry

end arithmetic_geom_sequences_l234_234735


namespace cos_of_F_in_def_l234_234471

theorem cos_of_F_in_def (E F : ℝ) (h₁ : E + F = π / 2) (h₂ : Real.sin E = 3 / 5) : Real.cos F = 3 / 5 :=
sorry

end cos_of_F_in_def_l234_234471


namespace next_thursday_Aug_15_2012_l234_234312

/-- Define a function that returns the day of the week for August 15 of a given year. -/
def day_of_week_Aug_15 (year : ℕ) : Nat :=
  let days_in_year (y : ℕ) := if leap_year y then 366 else 365
  let starting_day_2010 := 0  -- Monday is 0
  let days_passed : Nat := (List.range' 2010 (year - 2010)).foldl (λ acc y => acc + days_in_year y) 0
  (starting_day_2010 + days_passed) % 7

/-- Prove that the next year when August 15 is a Thursday is 2012. -/
theorem next_thursday_Aug_15_2012 : (day_of_week_Aug_15 2012) = 4 := by
  sorry

end next_thursday_Aug_15_2012_l234_234312


namespace candies_eaten_l234_234572

variables (A B D : ℕ)

-- Conditions:
def condition1 : Prop := ∃ k1 k2 k3 : ℕ, k1 * 4 + k2 * 3 + k3 * 7 = 70
def condition2 : Prop := (B * 3 = A * 4) ∧ (D * 7 = A * 6)
def condition3 : Prop := A + B + D = 70

-- Theorem statement:
theorem candies_eaten (h1 : condition1) (h2 : condition2) (h3 : condition3) :
    A = 24 ∧ B = 18 ∧ D = 28 := sorry

end candies_eaten_l234_234572


namespace farmer_has_11_goats_l234_234056

theorem farmer_has_11_goats
  (pigs cows goats : ℕ)
  (h1 : pigs = 2 * cows)
  (h2 : cows = goats + 4)
  (h3 : goats + cows + pigs = 56) :
  goats = 11 := by
  sorry

end farmer_has_11_goats_l234_234056


namespace max_value_expression_l234_234342

theorem max_value_expression : ∀ (a b c d : ℝ), 
  a ∈ Set.Icc (-4.5) 4.5 → 
  b ∈ Set.Icc (-4.5) 4.5 → 
  c ∈ Set.Icc (-4.5) 4.5 → 
  d ∈ Set.Icc (-4.5) 4.5 → 
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 90 :=
by sorry

end max_value_expression_l234_234342


namespace find_sum_of_p_q_r_s_l234_234304

theorem find_sum_of_p_q_r_s 
    (p q r s : ℝ)
    (h1 : r + s = 12 * p)
    (h2 : r * s = -13 * q)
    (h3 : p + q = 12 * r)
    (h4 : p * q = -13 * s)
    (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
    p + q + r + s = 2028 := 
sorry

end find_sum_of_p_q_r_s_l234_234304


namespace lambda_n_inequality_l234_234766

theorem lambda_n_inequality (n : ℕ) (hn : n > 1) 
(z : Fin n → ℂ) (hz : ∀ i, z i ≠ 0): 
  (∑ k in Finset.range n, Complex.normSq (z k)) ≥ 
  (π^2 / n) * 
  (min (Finset.range n) (λ k, Complex.normSq (z (⟨(k+1) % n, by simp⟩) - z ⟨k, by simp⟩))) :=
sorry

end lambda_n_inequality_l234_234766


namespace angle_C_modified_l234_234705

theorem angle_C_modified (A B C : ℝ) (h_eq_triangle: A = B) (h_C_modified: C = A + 40) (h_sum_angles: A + B + C = 180) : 
  C = 86.67 := 
by 
  sorry

end angle_C_modified_l234_234705


namespace problem_l234_234860

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem problem : (nabla (nabla 2 3) 4) = 16777219 :=
by
  unfold nabla
  -- First compute 2 ∇ 3
  have h1 : nabla 2 3 = 12 := by norm_num
  rw [h1]
  -- Now compute 12 ∇ 4
  unfold nabla
  norm_num
  sorry

end problem_l234_234860


namespace original_profit_percentage_l234_234375

theorem original_profit_percentage (C : ℝ) (C' : ℝ) (S' : ℝ) (H1 : C = 40) (H2 : C' = 32) (H3 : S' = 41.60) 
  (H4 : S' = (1.30 * C')) : (S' + 8.40 - C) / C * 100 = 25 := 
by 
  sorry

end original_profit_percentage_l234_234375


namespace weight_of_b_l234_234790

theorem weight_of_b (A B C : ℝ) 
  (h1 : A + B + C = 129)
  (h2 : A + B = 96)
  (h3 : B + C = 84) : 
  B = 51 :=
sorry

end weight_of_b_l234_234790


namespace max_value_90_l234_234340

noncomputable def max_value_expression (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_90 (a b c d : ℝ) (h₁ : -4.5 ≤ a) (h₂ : a ≤ 4.5)
                                   (h₃ : -4.5 ≤ b) (h₄ : b ≤ 4.5)
                                   (h₅ : -4.5 ≤ c) (h₆ : c ≤ 4.5)
                                   (h₇ : -4.5 ≤ d) (h₈ : d ≤ 4.5) :
  max_value_expression a b c d ≤ 90 :=
sorry

end max_value_90_l234_234340


namespace eval_7_star_3_l234_234128

def operation (a b : ℕ) : ℕ := (4 * a + 5 * b - a * b)

theorem eval_7_star_3 : operation 7 3 = 22 :=
  by {
    -- substitution and calculation steps
    sorry
  }

end eval_7_star_3_l234_234128


namespace abs_neg_sqrt_six_l234_234951

noncomputable def abs_val (x : ℝ) : ℝ :=
  if x < 0 then -x else x

theorem abs_neg_sqrt_six : abs_val (- Real.sqrt 6) = Real.sqrt 6 := by
  -- Proof goes here
  sorry

end abs_neg_sqrt_six_l234_234951


namespace quadratic_equation_with_distinct_roots_l234_234459

theorem quadratic_equation_with_distinct_roots 
  (a p q b α : ℝ) 
  (hα1 : α ≠ 0) 
  (h_quad1 : α^2 + a * α + b = 0) 
  (h_quad2 : α^2 + p * α + q = 0) : 
  ∃ x : ℝ, x^2 - (b + q) * (a - p) / (q - b) * x + b * q * (a - p)^2 / (q - b)^2 = 0 :=
by
  sorry

end quadratic_equation_with_distinct_roots_l234_234459


namespace cos_half_angle_quadrant_l234_234424

theorem cos_half_angle_quadrant 
  (α : ℝ) 
  (h1 : 25 * Real.sin α ^ 2 + Real.sin α - 24 = 0) 
  (h2 : π / 2 < α ∧ α < π) 
  : Real.cos (α / 2) = 3 / 5 ∨ Real.cos (α / 2) = -3 / 5 :=
by
  sorry

end cos_half_angle_quadrant_l234_234424


namespace intersecting_sets_a_eq_1_l234_234123

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := { x | a * x^2 - 1 = 0 }
def N : Set ℝ := { -1/2, 1/2, 1 }

-- Define the intersection condition
def sets_intersect (M N : Set ℝ) : Prop :=
  ∃ x, x ∈ M ∧ x ∈ N

-- Statement of the problem
theorem intersecting_sets_a_eq_1 (a : ℝ) (h_intersect : sets_intersect (M a) N) : a = 1 :=
  sorry

end intersecting_sets_a_eq_1_l234_234123


namespace max_value_of_M_l234_234153

noncomputable def M (x y z : ℝ) := min (min x y) z

theorem max_value_of_M
  (a b c : ℝ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_zero : b^2 - 4 * a * c ≥ 0) :
  M ((b + c) / a) ((c + a) / b) ((a + b) / c) ≤ 5 / 4 :=
sorry

end max_value_of_M_l234_234153


namespace cos_60_eq_half_l234_234657

theorem cos_60_eq_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_60_eq_half_l234_234657


namespace two_digit_remainder_one_when_divided_by_4_and_17_l234_234679

-- Given the conditions
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def yields_remainder (n d r : ℕ) : Prop := n % d = r

-- Define the main problem that checks if there is only one such number
theorem two_digit_remainder_one_when_divided_by_4_and_17 :
  ∃! n : ℕ, is_two_digit n ∧ yields_remainder n 4 1 ∧ yields_remainder n 17 1 :=
sorry

end two_digit_remainder_one_when_divided_by_4_and_17_l234_234679


namespace infinitely_many_c_exist_l234_234947

theorem infinitely_many_c_exist :
  ∃ c: ℕ, ∃ x y z: ℕ, (x^2 - c) * (y^2 - c) = z^2 - c ∧ (x^2 + c) * (y^2 - c) = z^2 - c :=
by
  sorry

end infinitely_many_c_exist_l234_234947


namespace employee_payment_correct_l234_234065

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the retail price increase percentage
def retail_increase_percentage : ℝ := 0.20

-- Define the employee discount percentage
def employee_discount_percentage : ℝ := 0.30

-- Define the retail price as wholesale cost increased by the retail increase percentage
def retail_price : ℝ := wholesale_cost * (1 + retail_increase_percentage)

-- Define the discount amount as the retail price multiplied by the discount percentage
def discount_amount : ℝ := retail_price * employee_discount_percentage

-- Define the final employee payment as retail price minus the discount amount
def employee_final_payment : ℝ := retail_price - discount_amount

-- Theorem statement: Prove that the employee final payment equals $168
theorem employee_payment_correct : employee_final_payment = 168 := by
  sorry

end employee_payment_correct_l234_234065


namespace factorable_iff_some_even_b_l234_234017

open Int

theorem factorable_iff_some_even_b (b : ℤ) :
  (∃ m n p q : ℤ,
    (35 : ℤ) = m * p ∧
    (35 : ℤ) = n * q ∧
    b = m * q + n * p) →
  (∃ k : ℤ, b = 2 * k) :=
by
  sorry

end factorable_iff_some_even_b_l234_234017


namespace worker_idle_days_l234_234974

variable (x y : ℤ)

theorem worker_idle_days :
  (30 * x - 5 * y = 500) ∧ (x + y = 60) → y = 38 :=
by
  intros h
  have h1 : 30 * x - 5 * y = 500 := h.left
  have h2 : x + y = 60 := h.right
  sorry

end worker_idle_days_l234_234974


namespace probability_penny_nickel_dime_heads_l234_234177

noncomputable def probability_heads (n : ℕ) : ℚ := (1 : ℚ) / (2 ^ n)

theorem probability_penny_nickel_dime_heads :
  probability_heads 3 = 1 / 8 := 
by
  sorry

end probability_penny_nickel_dime_heads_l234_234177


namespace blocks_tower_l234_234321

theorem blocks_tower (T H Total : ℕ) (h1 : H = 53) (h2 : Total = 80) (h3 : T + H = Total) : T = 27 :=
by
  -- proof goes here
  sorry

end blocks_tower_l234_234321


namespace may_make_total_scarves_l234_234162

theorem may_make_total_scarves (red_yarns blue_yarns yellow_yarns : ℕ) (scarves_per_yarn : ℕ)
    (h_red: red_yarns = 2) (h_blue: blue_yarns = 6) (h_yellow: yellow_yarns = 4) (h_scarves : scarves_per_yarn = 3) :
    (red_yarns * scarves_per_yarn + blue_yarns * scarves_per_yarn + yellow_yarns * scarves_per_yarn) = 36 := 
by
    rw [h_red, h_blue, h_yellow, h_scarves]
    norm_num
    sorry

end may_make_total_scarves_l234_234162


namespace center_of_circle_l234_234086

theorem center_of_circle (x y : ℝ) (h : x^2 + y^2 = 4 * x - 6 * y + 9) : x + y = -1 := 
by 
  sorry

end center_of_circle_l234_234086


namespace initial_performers_count_l234_234713

theorem initial_performers_count (n : ℕ)
    (h1 : ∃ rows, 8 * rows = n)
    (h2 : ∃ (m : ℕ), n + 16 = m ∧ ∃ s, s * s = m)
    (h3 : ∃ (k : ℕ), n + 1 = k ∧ ∃ t, t * t = k) : 
    n = 48 := 
sorry

end initial_performers_count_l234_234713


namespace bottles_remaining_after_2_days_l234_234687

def total_bottles := 48 

def first_day_father_consumption := total_bottles / 4
def first_day_mother_consumption := total_bottles / 6
def first_day_son_consumption := total_bottles / 8

def total_first_day_consumption := first_day_father_consumption + first_day_mother_consumption + first_day_son_consumption 
def remaining_after_first_day := total_bottles - total_first_day_consumption

def second_day_father_consumption := remaining_after_first_day / 5
def remaining_after_father := remaining_after_first_day - second_day_father_consumption
def second_day_mother_consumption := remaining_after_father / 7
def remaining_after_mother := remaining_after_father - second_day_mother_consumption
def second_day_son_consumption := remaining_after_mother / 9
def remaining_after_son := remaining_after_mother - second_day_son_consumption
def second_day_daughter_consumption := remaining_after_son / 9
def remaining_after_daughter := remaining_after_son - second_day_daughter_consumption

theorem bottles_remaining_after_2_days : ∀ (total_bottles : ℕ), remaining_after_daughter = 14 := 
by
  sorry

end bottles_remaining_after_2_days_l234_234687


namespace chord_cos_theta_condition_l234_234147

open Real

-- Translation of the given conditions and proof problem
theorem chord_cos_theta_condition
  (a b x y θ : ℝ)
  (h1 : a^2 = b^2 + 2) :
  x * y = cos θ := 
sorry

end chord_cos_theta_condition_l234_234147


namespace dave_total_time_l234_234710

variable (W J : ℕ)

-- Given conditions
def time_walked := W = 9
def ratio := J / W = 4 / 3

-- Statement to prove
theorem dave_total_time (time_walked : time_walked W) (ratio : ratio J W) : W + J = 21 := 
by
  sorry

end dave_total_time_l234_234710


namespace lindsey_savings_l234_234157

theorem lindsey_savings
  (september_savings : Nat := 50)
  (october_savings : Nat := 37)
  (november_savings : Nat := 11)
  (additional_savings : Nat := 25)
  (video_game_cost : Nat := 87)
  (total_savings := september_savings + october_savings + november_savings)
  (mom_bonus : Nat := if total_savings > 75 then additional_savings else 0)
  (final_amount := total_savings + mom_bonus - video_game_cost) :
  final_amount = 36 := by
  sorry

end lindsey_savings_l234_234157


namespace inequality_addition_l234_234886

theorem inequality_addition (a b : ℝ) (h : a > b) : a + 3 > b + 3 := by
  sorry

end inequality_addition_l234_234886


namespace angle_between_generatrix_and_base_of_cone_l234_234547

theorem angle_between_generatrix_and_base_of_cone (r R H : ℝ) (α : ℝ)
  (h_cylinder_height : H = 2 * R)
  (h_total_surface_area : 2 * Real.pi * r * H + 2 * Real.pi * r^2 = Real.pi * R^2) :
  α = Real.arctan (2 * (4 + Real.sqrt 6) / 5) :=
sorry

end angle_between_generatrix_and_base_of_cone_l234_234547


namespace square_diff_theorem_l234_234637

theorem square_diff_theorem
  (a b c p x : ℝ)
  (h1 : a + b + c = 2 * p)
  (h2 : x = (b^2 + c^2 - a^2) / (2 * c))
  (h3 : c ≠ 0) :
  b^2 - x^2 = 4 / c^2 * (p * (p - a) * (p - b) * (p - c)) := by
  sorry

end square_diff_theorem_l234_234637


namespace min_tetrahedrons_to_partition_cube_l234_234030

theorem min_tetrahedrons_to_partition_cube : ∃ n : ℕ, n = 5 ∧ (∀ m : ℕ, m < 5 → ¬partitions_cube_into_tetrahedra m) :=
by
  sorry

end min_tetrahedrons_to_partition_cube_l234_234030


namespace find_x_l234_234720

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 210) (h2 : ⌈x⌉ - 1 < x ∧ x ≤ ⌈x⌉) (h3 : 0 < x) : x = 14 :=
sorry

end find_x_l234_234720


namespace positive_difference_l234_234348

theorem positive_difference
  (x y : ℝ)
  (h1 : x + y = 10)
  (h2 : x^2 - y^2 = 40) : abs (x - y) = 4 :=
sorry

end positive_difference_l234_234348


namespace solve_log_eq_l234_234950

theorem solve_log_eq (x : ℝ) (h : 0 < x) :
  (1 / (Real.sqrt (Real.logb 5 (5 * x)) + Real.sqrt (Real.logb 5 x)) + Real.sqrt (Real.logb 5 x) = 2) ↔ x = 125 := 
  sorry

end solve_log_eq_l234_234950


namespace length_AE_l234_234290

structure Point where
  x : ℕ
  y : ℕ

def A : Point := ⟨0, 4⟩
def B : Point := ⟨7, 0⟩
def C : Point := ⟨5, 3⟩
def D : Point := ⟨3, 0⟩

noncomputable def dist (P Q : Point) : ℝ :=
  Real.sqrt (((Q.x - P.x : ℝ) ^ 2) + ((Q.y - P.y : ℝ) ^ 2))

noncomputable def AE_length : ℝ :=
  (5 * (dist A B)) / 9

theorem length_AE :
  ∃ E : Point, AE_length = (5 * Real.sqrt 65) / 9 := by
  sorry

end length_AE_l234_234290


namespace max_marked_vertices_no_rectangle_l234_234523

theorem max_marked_vertices_no_rectangle (n : ℕ) (hn : n = 2016) : 
  ∃ m ≤ n, m = 1009 ∧ 
  ∀ A B C D : Fin n, 
    (A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D) ∧ 
    (marked A → marked B → marked C → marked D → 
     ¬is_rectangle A B C D) → 
      (∃ f : Fin n → Bool, marked f ∧ 
      (count_marked f ≤ 1009)) := sorry

end max_marked_vertices_no_rectangle_l234_234523


namespace sequence_term_formula_l234_234479

def sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = 1/2 - 1/2 * a n

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n ≥ 2, a n = r * a (n - 1)

theorem sequence_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n ≥ 1, S n = 1/2 - 1/2 * a n) →
  (S 1 = 1/2 - 1/2 * a 1) →
  a 1 = 1/3 →
  (∀ n ≥ 2, S n = 1/2 - 1/2 * (a n) → S (n - 1) = 1/2 - 1/2 * (a (n - 1)) → a n = 1/3 * a (n-1)) →
  ∀ n, a n = (1/3)^n :=
by
  intro h1 h2 h3 h4
  sorry

end sequence_term_formula_l234_234479


namespace notebook_cost_l234_234692

theorem notebook_cost (s n c : ℕ) (h1 : s > 25)
                                 (h2 : n % 2 = 1)
                                 (h3 : n > 1)
                                 (h4 : c > n)
                                 (h5 : s * n * c = 2739) :
  c = 7 :=
sorry

end notebook_cost_l234_234692


namespace attendance_second_concert_l234_234771

-- Define the given conditions
def attendance_first_concert : ℕ := 65899
def additional_people : ℕ := 119

-- Prove the number of people at the second concert
theorem attendance_second_concert : 
  attendance_first_concert + additional_people = 66018 := 
by
  -- Placeholder for the proof
  sorry

end attendance_second_concert_l234_234771


namespace trigonometric_inequality_l234_234300

theorem trigonometric_inequality (a b : ℝ) (ha : 0 < a ∧ a < Real.pi / 2) (hb : 0 < b ∧ b < Real.pi / 2) :
  5 / Real.cos a ^ 2 + 5 / (Real.sin a ^ 2 * Real.sin b ^ 2 * Real.cos b ^ 2) ≥ 27 * Real.cos a + 36 * Real.sin a :=
sorry

end trigonometric_inequality_l234_234300


namespace part1_part2_l234_234317

def partsProcessedA : ℕ → ℕ
| 0 => 10
| (n + 1) => if n = 0 then 8 else partsProcessedA n - 2

def partsProcessedB : ℕ → ℕ
| 0 => 8
| (n + 1) => if n = 0 then 7 else partsProcessedB n - 1

def partsProcessedLineB_A (n : ℕ) := 7 * n
def partsProcessedLineB_B (n : ℕ) := 8 * n

def maxSetsIn14Days : ℕ := 
  let aLineA := 2 * (10 + 8 + 6) + (10 + 8)
  let aLineB := 2 * (8 + 7 + 6) + (8 + 8)
  min aLineA aLineB

theorem part1 :
  partsProcessedA 0 + partsProcessedA 1 + partsProcessedA 2 = 24 := 
by sorry

theorem part2 :
  maxSetsIn14Days = 106 :=
by sorry

end part1_part2_l234_234317


namespace ab_value_l234_234745

theorem ab_value 
  (a b c : ℝ)
  (h1 : a - b = 5)
  (h2 : a^2 + b^2 = 34)
  (h3 : a^3 - b^3 = 30)
  (h4 : a^2 + b^2 - c^2 = 50)
  (h5 : c = 2 * a - b) : 
  a * b = 17 := 
by 
  sorry

end ab_value_l234_234745


namespace group_C_forms_triangle_l234_234335

theorem group_C_forms_triangle :
  ∀ (a b c : ℕ), (a + b > c ∧ a + c > b ∧ b + c > a) ↔ ((a, b, c) = (2, 3, 4)) :=
by
  -- we'll prove the forward and backward directions separately
  sorry

end group_C_forms_triangle_l234_234335


namespace candidate_lost_by_l234_234207

noncomputable def candidate_votes (total_votes : ℝ) := 0.35 * total_votes
noncomputable def rival_votes (total_votes : ℝ) := 0.65 * total_votes

theorem candidate_lost_by (total_votes : ℝ) (h : total_votes = 7899.999999999999) :
  rival_votes total_votes - candidate_votes total_votes = 2370 :=
by
  sorry

end candidate_lost_by_l234_234207


namespace rationalize_denominator_l234_234774

theorem rationalize_denominator (a b c : Real) (h : b*c*c = a) :
  2 / (b + c) = (c*c) / (3 * 2) :=
by
  sorry

end rationalize_denominator_l234_234774


namespace possible_values_of_ab_plus_ac_plus_bc_l234_234627

theorem possible_values_of_ab_plus_ac_plus_bc (a b c : ℝ) (h : a + b + 2 * c = 0) :
  ∃ x ∈ Set.Iic 0, ab + ac + bc = x :=
sorry

end possible_values_of_ab_plus_ac_plus_bc_l234_234627


namespace farmer_goats_l234_234058

theorem farmer_goats (G C P : ℕ) (h1 : P = 2 * C) (h2 : C = G + 4) (h3 : G + C + P = 56) : G = 11 :=
by
  sorry

end farmer_goats_l234_234058


namespace olivia_paper_count_l234_234636

-- State the problem conditions and the final proof statement.
theorem olivia_paper_count :
  let math_initial := 220
  let science_initial := 150
  let math_used := 95
  let science_used := 68
  let math_received := 30
  let science_given := 15
  let math_remaining := math_initial - math_used + math_received
  let science_remaining := science_initial - science_used - science_given
  let total_pieces := math_remaining + science_remaining
  total_pieces = 222 :=
by
  -- Placeholder for the proof
  sorry

end olivia_paper_count_l234_234636


namespace greatest_value_of_n_l234_234044

theorem greatest_value_of_n (n : ℤ) (h : 101 * n ^ 2 ≤ 3600) : n ≤ 5 :=
by
  sorry

end greatest_value_of_n_l234_234044


namespace inequality_solution_l234_234326

noncomputable def solve_inequality (x : ℝ) : Prop :=
  x ∈ Set.Ioo (-3 : ℝ) 3

theorem inequality_solution (x : ℝ) (h : x ≠ -3) :
  (x^2 - 9) / (x + 3) < 0 ↔ solve_inequality x :=
by
  sorry

end inequality_solution_l234_234326


namespace three_digit_solutions_l234_234814

def three_digit_number (n a x y z : ℕ) : Prop :=
  n = 100 * x + 10 * y + z ∧
  1 ≤ x ∧ x < 10 ∧ 
  0 ≤ y ∧ y < 10 ∧ 
  0 ≤ z ∧ z < 10 ∧ 
  n + (x + y + z) = 111 * a

theorem three_digit_solutions (n : ℕ) (a x y z : ℕ) :
  three_digit_number n a x y z ↔ 
  n = 105 ∨ n = 324 ∨ n = 429 ∨ n = 543 ∨ 
  n = 648 ∨ n = 762 ∨ n = 867 ∨ n = 981 :=
sorry

end three_digit_solutions_l234_234814


namespace a_plus_b_eq_neg7_l234_234151

theorem a_plus_b_eq_neg7 (a b : ℝ) :
  (∀ x : ℝ, (x^2 - 2 * x - 3 > 0) ∨ (x^2 + a * x + b ≤ 0)) ∧
  (∀ x : ℝ, (3 < x ∧ x ≤ 4) → ((x^2 - 2 * x - 3 > 0) ∧ (x^2 + a * x + b ≤ 0))) →
  a + b = -7 :=
by
  sorry

end a_plus_b_eq_neg7_l234_234151


namespace max_S_n_value_l234_234626

theorem max_S_n_value (a : ℕ → ℤ) (d : ℤ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = d)
  (h_d_neg : d < 0)
  (h_S8_S12 : (∑ i in Finset.range 8, a i) = (∑ i in Finset.range 12, a i)) :
  ∃ n : ℕ, n = 10 :=
by
  sorry

end max_S_n_value_l234_234626


namespace nancy_kept_chips_l234_234168

def nancy_initial_chips : ℕ := 22
def chips_given_to_brother : ℕ := 7
def chips_given_to_sister : ℕ := 5

theorem nancy_kept_chips : nancy_initial_chips - (chips_given_to_brother + chips_given_to_sister) = 10 :=
by
  sorry

end nancy_kept_chips_l234_234168


namespace theo_needs_84_eggs_l234_234219

def customers_hour1 := 5
def customers_hour2 := 7
def customers_hour3 := 3
def customers_hour4 := 8

def eggs_per_omelette_3 := 3
def eggs_per_omelette_4 := 4

def total_eggs_needed : Nat :=
  (customers_hour1 * eggs_per_omelette_3) +
  (customers_hour2 * eggs_per_omelette_4) +
  (customers_hour3 * eggs_per_omelette_3) +
  (customers_hour4 * eggs_per_omelette_4)

theorem theo_needs_84_eggs : total_eggs_needed = 84 :=
by
  sorry

end theo_needs_84_eggs_l234_234219


namespace shop_owner_profitable_l234_234990

noncomputable def shop_owner_profit (CP_SP_difference_percentage: ℚ) (CP: ℚ) (buy_cheat_percentage: ℚ) (sell_cheat_percentage: ℚ) (buy_discount_percentage: ℚ) (sell_markup_percentage: ℚ) : ℚ := 
  CP_SP_difference_percentage * 100

theorem shop_owner_profitable :
  shop_owner_profit ((114 * (110 / 80 / 100) - 90) / 90) 1 0.14 0.20 0.10 0.10 = 74.17 := 
by
  sorry

end shop_owner_profitable_l234_234990


namespace area_of_sandbox_is_correct_l234_234379

-- Define the length and width of the sandbox
def length_sandbox : ℕ := 312
def width_sandbox : ℕ := 146

-- Define the area calculation
def area_sandbox (length width : ℕ) : ℕ := length * width

-- The theorem stating that the area of the sandbox is 45552 cm²
theorem area_of_sandbox_is_correct : area_sandbox length_sandbox width_sandbox = 45552 := sorry

end area_of_sandbox_is_correct_l234_234379


namespace smith_trip_times_same_l234_234781

theorem smith_trip_times_same (v : ℝ) (hv : v > 0) : 
  let t1 := 80 / v 
  let t2 := 160 / (2 * v) 
  t1 = t2 :=
by
  sorry

end smith_trip_times_same_l234_234781


namespace beach_weather_condition_l234_234579

theorem beach_weather_condition
  (T : ℝ) -- Temperature in degrees Fahrenheit
  (sunny : Prop) -- Whether it is sunny
  (crowded : Prop) -- Whether the beach is crowded
  (H1 : ∀ (T : ℝ) (sunny : Prop), (T ≥ 80) ∧ sunny → crowded) -- Condition 1
  (H2 : ¬ crowded) -- Condition 2
  : T < 80 ∨ ¬ sunny := sorry

end beach_weather_condition_l234_234579


namespace problem_proof_l234_234540

theorem problem_proof (n : ℕ) 
  (h : ∃ k, 2 * k = n) :
  4 ∣ n :=
sorry

end problem_proof_l234_234540


namespace vector_condition_l234_234749

def vec_a : ℝ × ℝ := (5, 2)
def vec_b : ℝ × ℝ := (-4, -3)
def vec_c : ℝ × ℝ := (-23, -12)

theorem vector_condition : 3 • (vec_a.1, vec_a.2) - 2 • (vec_b.1, vec_b.2) + vec_c = (0, 0) :=
by
  sorry

end vector_condition_l234_234749


namespace correct_choice_D_l234_234887

variable (a b : Line) (α : Plane)

-- Definitions for the conditions
def is_perpendicular (l : Line) (p : Plane) : Prop := sorry  -- Definition of perpendicular
def is_parallel_line (l1 l2 : Line) : Prop := sorry  -- Definition of parallel lines
def is_parallel_plane (l : Line) (p : Plane) : Prop := sorry  -- Definition of line parallel to plane
def is_subset (l : Line) (p : Plane) : Prop := sorry  -- Definition of line being in a plane

-- The statement of the problem
theorem correct_choice_D :
  (is_parallel_plane a α) ∧ (is_subset b α) → (is_parallel_plane a α) := 
by 
  sorry

end correct_choice_D_l234_234887


namespace roots_ratio_quadratic_l234_234249

theorem roots_ratio_quadratic (p : ℤ) (h : (∃ x1 x2 : ℤ, x1*x2 = -16 ∧ x1 + x2 = -p ∧ x2 = -4 * x1)) :
  p = 6 ∨ p = -6 :=
sorry

end roots_ratio_quadratic_l234_234249


namespace non_adjacent_girls_arrangements_l234_234589

-- Definitions corresponding to conditions in the problem

def boys : ℕ := 3
def girls : ℕ := 2
def total_students : ℕ := boys + girls

theorem non_adjacent_girls_arrangements : 
  ∀ (boys girls : ℕ), 
  boys = 3 → 
  girls = 2 → 
  ∑ i in finset.range (boys!), 
  ∑ j in finset.range ((boys + 1) - girls)! = 72 :=
begin
  -- sorry is used to indicate that the proof is omitted
  sorry
end

end non_adjacent_girls_arrangements_l234_234589


namespace value_of_sin_l234_234899

variable {a b x π : ℝ}
variable {x0 : ℝ}

-- Assuming all conditions given in the problem
def f (x : ℝ) : ℝ := a * Math.sin x + b * Math.cos x

axiom h_ab : a ≠ 0 ∧ b ≠ 0
axiom h_symmetry : ∀ x, f (x + π / 6) = f (π / 6 - x)
axiom h_f_x0 : f x0 = 8/5 * a

-- The goal is to prove this statement
theorem value_of_sin (h_ab : a ≠ 0 ∧ b ≠ 0)
                      (h_symmetry : ∀ x, f (x + π / 6) = f (π / 6 - x))
                      (h_f_x0 : f x0 = 8 / 5 * a) :
  Math.sin (2 * x0 + π / 6) = 7 / 25 :=
sorry

end value_of_sin_l234_234899


namespace tangent_line_hyperbola_eq_l234_234957

noncomputable def tangent_line_ellipse (a b x0 y0 x y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a > 0) 
  (h_ell : x0 ^ 2 / a ^ 2 + y0 ^ 2 / b ^ 2 = 1) : Prop :=
  (x0 * x) / (a ^ 2) + (y0 * y) / (b ^ 2) = 1

noncomputable def tangent_line_hyperbola (a b x0 y0 x y : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h_hyp : x0 ^ 2 / a ^ 2 - y0 ^ 2 / b ^ 2 = 1) : Prop :=
  (x0 * x) / (a ^ 2) - (y0 * y) / (b ^ 2) = 1

theorem tangent_line_hyperbola_eq (a b x0 y0 x y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a > 0)
  (h_ellipse_tangent : tangent_line_ellipse a b x0 y0 x y h1 h2 h3 (by sorry))
  (h_hyperbola : x0 ^ 2 / a ^ 2 - y0 ^ 2 / b ^ 2 = 1) : 
  tangent_line_hyperbola a b x0 y0 x y h3 h2 h_hyperbola :=
by sorry

end tangent_line_hyperbola_eq_l234_234957


namespace max_value_a_l234_234281

def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → |a - 2| ≤ |x + 1 / x|

theorem max_value_a : ∃ (a : ℝ), condition a ∧ (∀ b : ℝ, condition b → b ≤ 4) :=
  sorry

end max_value_a_l234_234281


namespace shaded_area_percentage_is_correct_l234_234674

noncomputable def total_area_of_square : ℕ := 49

noncomputable def area_of_first_shaded_region : ℕ := 2^2

noncomputable def area_of_second_shaded_region : ℕ := 25 - 9

noncomputable def area_of_third_shaded_region : ℕ := 49 - 36

noncomputable def total_shaded_area : ℕ :=
  area_of_first_shaded_region + area_of_second_shaded_region + area_of_third_shaded_region

noncomputable def percent_shaded_area : ℚ :=
  (total_shaded_area : ℚ) / total_area_of_square * 100

theorem shaded_area_percentage_is_correct :
  percent_shaded_area = 67.35 := by
sorry

end shaded_area_percentage_is_correct_l234_234674


namespace candies_eaten_l234_234570

variables (A B D : ℕ)

-- Conditions:
def condition1 : Prop := ∃ k1 k2 k3 : ℕ, k1 * 4 + k2 * 3 + k3 * 7 = 70
def condition2 : Prop := (B * 3 = A * 4) ∧ (D * 7 = A * 6)
def condition3 : Prop := A + B + D = 70

-- Theorem statement:
theorem candies_eaten (h1 : condition1) (h2 : condition2) (h3 : condition3) :
    A = 24 ∧ B = 18 ∧ D = 28 := sorry

end candies_eaten_l234_234570


namespace find_angle_C_find_area_l234_234615

open Real

def is_acute_triangle (A B C a b c : ℝ) : Prop :=
  0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π

theorem find_angle_C (A B C a b c : ℝ) 
    (h_acute : is_acute_triangle A B C a b c)
    (h_sides : 2 * c * sin A = sqrt 3 * a) :
  C = π / 3 :=
sorry

theorem find_area (A B C a b c : ℝ) 
    (h_acute : is_acute_triangle A B C a b c)
    (h_C : C = π / 3) (h_b : b = 2) (h_c : c = sqrt 7) (h_a : a = 3) :
  let S := 1 / 2 * a * b * sin C in
  S = (3 * sqrt 3) / 2 :=
sorry

end find_angle_C_find_area_l234_234615


namespace probability_of_observing_color_change_l234_234551

def cycle_duration := 100
def observation_interval := 4
def change_times := [45, 50, 100]

def probability_of_change : ℚ :=
  (observation_interval * change_times.length : ℚ) / cycle_duration

theorem probability_of_observing_color_change :
  probability_of_change = 0.12 := by
  -- Proof goes here
  sorry

end probability_of_observing_color_change_l234_234551


namespace diagonals_in_nine_sided_polygon_l234_234988

-- Define the conditions
def sides : ℕ := 9
def right_angles : ℕ := 2

-- The function to calculate the number of diagonals for a polygon
def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- The theorem to prove
theorem diagonals_in_nine_sided_polygon : number_of_diagonals sides = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l234_234988


namespace measure_of_angle_x_in_triangle_l234_234198

theorem measure_of_angle_x_in_triangle
  (x : ℝ)
  (h1 : x + 2 * x + 45 = 180) :
  x = 45 :=
sorry

end measure_of_angle_x_in_triangle_l234_234198


namespace number_of_glass_bottles_l234_234180

theorem number_of_glass_bottles (total_litter : ℕ) (aluminum_cans : ℕ) (glass_bottles : ℕ) : 
  total_litter = 18 → aluminum_cans = 8 → glass_bottles = total_litter - aluminum_cans → glass_bottles = 10 :=
by
  intros h_total h_aluminum h_glass
  rw [h_total, h_aluminum] at h_glass
  exact h_glass.trans rfl


end number_of_glass_bottles_l234_234180


namespace tangent_line_at_point_P_fx_gt_2x_minus_ln_x_l234_234598

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x
noncomputable def g (x : ℝ) := f x - 2 * (x - Real.log x)

theorem tangent_line_at_point_P :
  let P := (2 : ℝ, Real.exp 2 / 2)
  let tangent_eq := (fun (x y : ℝ) => Real.exp 2 * x - 4 * y = 0)
  tangent_eq (P.1) (P.2) :=
sorry

theorem fx_gt_2x_minus_ln_x (x : ℝ) (hx : 0 < x) : 
  f x > 2 * (x - Real.log x) :=
sorry

end tangent_line_at_point_P_fx_gt_2x_minus_ln_x_l234_234598


namespace boxes_needed_l234_234455

theorem boxes_needed (total_oranges boxes_capacity : ℕ) (h1 : total_oranges = 94) (h2 : boxes_capacity = 8) : 
  (total_oranges + boxes_capacity - 1) / boxes_capacity = 12 := 
by
  sorry

end boxes_needed_l234_234455


namespace common_difference_is_4_l234_234895

variable (a : ℕ → ℤ) (d : ℤ)

-- Conditions of the problem
def arithmetic_sequence := ∀ n m : ℕ, a n = a m + (n - m) * d

axiom a7_eq_25 : a 7 = 25
axiom a4_eq_13 : a 4 = 13

-- The theorem to prove
theorem common_difference_is_4 : d = 4 :=
by
  sorry

end common_difference_is_4_l234_234895


namespace determine_m_l234_234903

theorem determine_m 
  (f : ℝ → ℝ) 
  (m : ℕ) 
  (h_nat: 0 < m) 
  (h_f: ∀ x, f x = x ^ (m^2 - 2 * m - 3)) 
  (h_no_intersection: ∀ x, f x ≠ 0) 
  (h_symmetric_origin : ∀ x, f (-x) = -f x) : 
  m = 2 :=
by
  sorry

end determine_m_l234_234903


namespace polynomial_value_l234_234415

theorem polynomial_value (a : ℝ) (h : a^2 + 2 * a = 1) : 
  2 * a^5 + 7 * a^4 + 5 * a^3 + 2 * a^2 + 5 * a + 1 = 4 :=
by
  sorry

end polynomial_value_l234_234415


namespace range_of_a_l234_234118

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x^2

theorem range_of_a (a : ℝ) :
  (∀ (b : ℝ), (b ≤ 0) → ∀ (x : ℝ), (x > Real.exp 1 ∧ x ≤ Real.exp 2) → f a b x ≥ x) →
  a ≥ Real.exp 2 / 2 :=
by
  sorry

end range_of_a_l234_234118


namespace no_perf_square_of_prime_three_digit_l234_234959

theorem no_perf_square_of_prime_three_digit {A B C : ℕ} (h_prime: Prime (100 * A + 10 * B + C)) : ¬ ∃ n : ℕ, B^2 - 4 * A * C = n^2 :=
by
  sorry

end no_perf_square_of_prime_three_digit_l234_234959


namespace wire_ratio_l234_234069

theorem wire_ratio (a b : ℝ) (h : (a / 4) ^ 2 = (b / (2 * Real.pi)) ^ 2 * Real.pi) : a / b = 2 / Real.sqrt Real.pi := by
  sorry

end wire_ratio_l234_234069


namespace albert_needs_more_money_l234_234700

def cost_paintbrush : Real := 1.50
def cost_paints : Real := 4.35
def cost_easel : Real := 12.65
def cost_canvas : Real := 7.95
def cost_palette : Real := 3.75
def money_albert_has : Real := 10.60
def total_cost : Real := cost_paintbrush + cost_paints + cost_easel + cost_canvas + cost_palette
def money_needed : Real := total_cost - money_albert_has

theorem albert_needs_more_money : money_needed = 19.60 := by
  sorry

end albert_needs_more_money_l234_234700


namespace find_cartesian_equations_and_min_distance_l234_234761

def parametric_curve (t : ℝ) := (1 - t^2) / (1 + t^2), (4 * t) / (1 + t^2)

def polar_to_cartesian_line (θ ρ : ℝ) := 
  2 * ρ * Real.cos θ + Real.sqrt 3 * ρ * Real.sin θ + 11 = 0

theorem find_cartesian_equations_and_min_distance :
  (∀ t : ℝ, (let (x, y) := parametric_curve t in x^2 + (y^2 / 4) = 1 ∧ x ≠ -1)) ∧
  (∀ θ ρ : ℝ, polar_to_cartesian_line θ ρ → 2 * (ρ * Real.cos θ) + Real.sqrt 3 * (ρ * Real.sin θ) + 11 = 0) ∧
  (∀ t : ℝ, (let (x, y) := parametric_curve t in 
    let d := |11 - 4| / Real.sqrt(2^2 + 3) in d = Real.sqrt 7)) :=
by
  sorry

end find_cartesian_equations_and_min_distance_l234_234761


namespace dispatch_plans_count_l234_234493

theorem dispatch_plans_count:
  -- conditions
  let total_athletes := 9
  let basketball_players := 5
  let soccer_players := 6
  let both_players := 2
  let only_basketball := 3
  let only_soccer := 4
  -- proof
  (both_players.choose 2 + both_players * only_basketball + both_players * only_soccer + only_basketball * only_soccer) = 28 :=
by
  sorry

end dispatch_plans_count_l234_234493


namespace exam_max_incorrect_answers_l234_234693

theorem exam_max_incorrect_answers :
  ∀ (c w b : ℕ),
  (c + w + b = 30) →
  (4 * c - w ≥ 85) → 
  (c ≥ 22) →
  (w ≤ 3) :=
by
  intros c w b h1 h2 h3
  sorry

end exam_max_incorrect_answers_l234_234693


namespace find_x_l234_234719

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 210) (h2 : ⌈x⌉ - 1 < x ∧ x ≤ ⌈x⌉) (h3 : 0 < x) : x = 14 :=
sorry

end find_x_l234_234719


namespace simplify_fractions_l234_234642

theorem simplify_fractions :
  (36 / 51) * (35 / 24) * (68 / 49) = (20 / 7) :=
by
  have h1 : 36 = 2^2 * 3^2 := by norm_num
  have h2 : 51 = 3 * 17 := by norm_num
  have h3 : 35 = 5 * 7 := by norm_num
  have h4 : 24 = 2^3 * 3 := by norm_num
  have h5 : 68 = 2^2 * 17 := by norm_num
  have h6 : 49 = 7^2 := by norm_num
  sorry

end simplify_fractions_l234_234642


namespace maximum_people_shaked_hands_l234_234288

-- Given conditions
variables (N : ℕ) (hN : N > 4)
def has_not_shaken_hands_with (a b : ℕ) : Prop := sorry -- This should define the shaking hand condition

-- Main statement
theorem maximum_people_shaked_hands (h : ∃ i, has_not_shaken_hands_with i 2) :
  ∃ k, k = N - 3 := 
sorry

end maximum_people_shaked_hands_l234_234288


namespace triangle_A1B1C1_has_angles_α_β_γ_l234_234773

open Real Geometry

-- Lean definition for the problem conditions
variables {A B C A1 B1 C1 : Point}
variables {α β γ : Real}
variables h1 : isosceles_triangle A C1 B (2*α)
variables h2 : isosceles_triangle B A1 C (2*β)
variables h3 : isosceles_triangle C B1 A (2*γ)
variables hsum : α + β + γ = 180

-- Lean theorem statement for the problem
theorem triangle_A1B1C1_has_angles_α_β_γ :
  ∠ A1 B1 C1 = α ∧ ∠ B1 C1 A1 = β ∧ ∠ C1 A1 B1 = γ :=
sorry

end triangle_A1B1C1_has_angles_α_β_γ_l234_234773


namespace janes_stick_shorter_than_sarahs_l234_234487

theorem janes_stick_shorter_than_sarahs :
  ∀ (pat_length jane_length pat_dirt sarah_factor : ℕ),
    pat_length = 30 →
    jane_length = 22 →
    pat_dirt = 7 →
    sarah_factor = 2 →
    (sarah_factor * (pat_length - pat_dirt)) - jane_length = 24 :=
by
  intros pat_length jane_length pat_dirt sarah_factor h1 h2 h3 h4
  -- sorry skips the proof
  sorry

end janes_stick_shorter_than_sarahs_l234_234487


namespace find_ending_number_divisible_by_eleven_l234_234796

theorem find_ending_number_divisible_by_eleven (start n end_num : ℕ) (h1 : start = 29) (h2 : n = 5) (h3 : ∀ k : ℕ, ∃ m : ℕ, m = start + k * 11) : end_num = 77 :=
sorry

end find_ending_number_divisible_by_eleven_l234_234796


namespace candies_eaten_l234_234566

theorem candies_eaten (A B D : ℕ) 
                      (h1 : 4 * B = 3 * A) 
                      (h2 : 7 * A = 6 * D) 
                      (h3 : A + B + D = 70) :
  A = 24 ∧ B = 18 ∧ D = 28 := 
by
  sorry

end candies_eaten_l234_234566


namespace angle_CBD_is_4_l234_234931

theorem angle_CBD_is_4 (angle_ABC : ℝ) (angle_ABD : ℝ) (h₁ : angle_ABC = 24) (h₂ : angle_ABD = 20) : angle_ABC - angle_ABD = 4 :=
by 
  sorry

end angle_CBD_is_4_l234_234931


namespace g_func_eq_l234_234650

theorem g_func_eq (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → g (x / y) = y * g x)
  (h2 : g 50 = 10) :
  g 25 = 20 :=
sorry

end g_func_eq_l234_234650


namespace bakery_batches_per_day_l234_234648

-- Definitions for the given problem's conditions
def baguettes_per_batch := 48
def baguettes_sold_batch1 := 37
def baguettes_sold_batch2 := 52
def baguettes_sold_batch3 := 49
def baguettes_left := 6

-- Theorem stating the number of batches made
theorem bakery_batches_per_day : 
  (baguettes_sold_batch1 + baguettes_sold_batch2 + baguettes_sold_batch3 + baguettes_left) / baguettes_per_batch = 3 :=
by 
  sorry

end bakery_batches_per_day_l234_234648


namespace abs_floor_value_l234_234240

theorem abs_floor_value : (Int.floor (|(-56.3: Real)|)) = 56 := 
by
  sorry

end abs_floor_value_l234_234240


namespace tangent_line_equation_monotonicity_and_maximum_l234_234896

open Function

noncomputable def f (x : ℝ) := Real.exp x * (x + 1) - (x^2 + 4 * x)

-- Define the first proof problem: the equation of the tangent line
theorem tangent_line_equation : 
  let f' := deriv f in
  f' 0 = -2 ∧ 2 * (0:ℝ) + (1:ℝ) - 1 = 0 := 
by 
  sorry

-- Define the second proof problem: monotonicity and maximum value
theorem monotonicity_and_maximum : 
  let f' := deriv f in
  (∀ x, x < -2 → 0 < f' x) ∧
  (∀ x, x > Real.ln 2 → 0 < f' x) ∧
  (∀ x, -2 < x ∧ x < Real.ln 2 → f' x < 0) ∧
  (∀ x, x = -2 → f x = 4 - Real.exp 2) :=
by 
  sorry

end tangent_line_equation_monotonicity_and_maximum_l234_234896


namespace max_largest_integer_of_five_l234_234623

theorem max_largest_integer_of_five (a b c d e : ℕ) (h1 : (a + b + c + d + e) = 500)
    (h2 : e > c ∧ c > d ∧ d > b ∧ b > a)
    (h3 : (a + b + d + e) / 4 = 105)
    (h4 : b + e = 150) : d ≤ 269 := 
sorry

end max_largest_integer_of_five_l234_234623


namespace alphanumeric_puzzle_l234_234206

/-- Alphanumeric puzzle proof problem -/
theorem alphanumeric_puzzle
  (A B C D E F H J K L : Nat)
  (h1 : A * B = B)
  (h2 : B * C = 10 * A + C)
  (h3 : C * D = 10 * B + C)
  (h4 : D * E = 100 * C + H)
  (h5 : E * F = 10 * D + K)
  (h6 : F * H = 100 * C + J)
  (h7 : H * J = 10 * K + J)
  (h8 : J * K = E)
  (h9 : K * L = L)
  (h10 : A * L = L) :
  A = 1 ∧ B = 3 ∧ C = 5 ∧ D = 7 ∧ E = 8 ∧ F = 9 ∧ H = 6 ∧ J = 4 ∧ K = 2 ∧ L = 0 :=
sorry

end alphanumeric_puzzle_l234_234206


namespace gcd_55555555_111111111_l234_234630

/-- Let \( m = 55555555 \) and \( n = 111111111 \).
We want to prove that the greatest common divisor (gcd) of \( m \) and \( n \) is 1. -/
theorem gcd_55555555_111111111 :
  let m := 55555555
  let n := 111111111
  Nat.gcd m n = 1 :=
by
  sorry

end gcd_55555555_111111111_l234_234630


namespace fraction_of_q_age_l234_234221

theorem fraction_of_q_age (P Q : ℕ) (h1 : P / Q = 3 / 4) (h2 : P + Q = 28) : (P - 0) / (Q - 0) = 3 / 4 :=
by
  sorry

end fraction_of_q_age_l234_234221


namespace tax_free_amount_l234_234550

theorem tax_free_amount (X : ℝ) (total_value : ℝ) (tax_paid : ℝ) 
    (tax_rate : ℝ) (exceeds_value : ℝ) :
    total_value = 1720 → 
    tax_rate = 0.11 → 
    tax_paid = 123.2 → 
    total_value - X = exceeds_value → 
    tax_paid = tax_rate * exceeds_value → 
    X = 600 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end tax_free_amount_l234_234550


namespace candy_eating_l234_234562

-- Definitions based on the conditions
def candies (andrey_rate boris_rate denis_rate : ℕ) (andrey_candies boris_candies denis_candies total_candies : ℕ) : Prop :=
  andrey_rate = 4 ∧ boris_rate = 3 ∧ denis_rate = 7 ∧ andrey_candies = 24 ∧ boris_candies = 18 ∧ denis_candies = 28 ∧
  total_candies = andrey_candies + boris_candies + denis_candies

-- Problem statement
theorem candy_eating :
  ∃ (a b d : ℕ), 
    candies 4 3 7 a b d 70 :=
sorry

end candy_eating_l234_234562


namespace correct_calculation_l234_234201

variable {a b : ℝ}

theorem correct_calculation : 
  (2 * a^3 + 2 * a ≠ 2 * a^4) ∧
  ((a - 2 * b)^2 ≠ a^2 - 4 * b^2) ∧
  (-5 * (2 * a - b) ≠ -10 * a - 5 * b) ∧
  ((-2 * a^2 * b)^3 = -8 * a^6 * b^3) :=
by
  sorry

end correct_calculation_l234_234201


namespace michael_age_multiple_l234_234938

theorem michael_age_multiple (M Y O k : ℤ) (hY : Y = 5) (hO : O = 3 * Y) (h_combined : M + O + Y = 28) (h_relation : O = k * (M - 1) + 1) : k = 2 :=
by
  -- Definitions and given conditions are provided:
  have hY : Y = 5 := hY
  have hO : O = 3 * Y := hO
  have h_combined : M + O + Y = 28 := h_combined
  have h_relation : O = k * (M - 1) + 1 := h_relation
  
  -- Begin the proof by using the provided conditions
  sorry

end michael_age_multiple_l234_234938


namespace benny_gave_sandy_books_l234_234392

theorem benny_gave_sandy_books :
  ∀ (Benny_initial Tim_books total_books Benny_after_giving : ℕ), 
    Benny_initial = 24 → 
    Tim_books = 33 →
    total_books = 47 → 
    total_books - Tim_books = Benny_after_giving →
    Benny_initial - Benny_after_giving = 10 :=
by
  intros Benny_initial Tim_books total_books Benny_after_giving
  intros hBenny_initial hTim_books htotal_books hBooks_after
  simp [hBenny_initial, hTim_books, htotal_books, hBooks_after]
  sorry


end benny_gave_sandy_books_l234_234392


namespace find_sum_of_distinct_numbers_l234_234303

variable {R : Type} [LinearOrderedField R]

theorem find_sum_of_distinct_numbers (p q r s : R) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p ∧ r * s = -13 * q)
  (h2 : p + q = 12 * r ∧ p * q = -13 * s) :
  p + q + r + s = 2028 := 
by 
  sorry

end find_sum_of_distinct_numbers_l234_234303


namespace farmer_has_11_goats_l234_234055

theorem farmer_has_11_goats
  (pigs cows goats : ℕ)
  (h1 : pigs = 2 * cows)
  (h2 : cows = goats + 4)
  (h3 : goats + cows + pigs = 56) :
  goats = 11 := by
  sorry

end farmer_has_11_goats_l234_234055


namespace solution_point_satisfies_inequalities_l234_234819

theorem solution_point_satisfies_inequalities:
  let x := -1/3
  let y := 2/3
  11 * x^2 + 8 * x * y + 8 * y^2 ≤ 3 ∧ x - 4 * y ≤ -3 :=
by
  let x := -1/3
  let y := 2/3
  sorry

end solution_point_satisfies_inequalities_l234_234819


namespace find_x_l234_234096

theorem find_x (x : ℝ) (h1 : x > 0) (h2 : ⌊x⌋ * x = 72) : x = 9 := by
  sorry

end find_x_l234_234096


namespace batsman_average_l234_234051

theorem batsman_average 
  (inns : ℕ)
  (highest : ℕ)
  (diff : ℕ)
  (avg_excl : ℕ)
  (total_in_44 : ℕ)
  (total_in_46 : ℕ)
  (average_in_46 : ℕ)
  (H1 : inns = 46)
  (H2 : highest = 202)
  (H3 : diff = 150)
  (H4 : avg_excl = 58)
  (H5 : total_in_44 = avg_excl * (inns - 2))
  (H6 : total_in_46 = total_in_44 + highest + (highest - diff))
  (H7 : average_in_46 = total_in_46 / inns) :
  average_in_46 = 61 := 
sorry

end batsman_average_l234_234051


namespace melanie_dimes_l234_234937

variable (initial_dimes : ℕ) -- initial dimes Melanie had
variable (dimes_from_dad : ℕ) -- dimes given by dad
variable (dimes_to_mother : ℕ) -- dimes given to mother

def final_dimes (initial_dimes dimes_from_dad dimes_to_mother : ℕ) : ℕ :=
  initial_dimes + dimes_from_dad - dimes_to_mother

theorem melanie_dimes :
  initial_dimes = 7 →
  dimes_from_dad = 8 →
  dimes_to_mother = 4 →
  final_dimes initial_dimes dimes_from_dad dimes_to_mother = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end melanie_dimes_l234_234937


namespace bill_spots_l234_234224

theorem bill_spots (b p : ℕ) (h1 : b + p = 59) (h2 : b = 2 * p - 1) : b = 39 := by
  sorry

end bill_spots_l234_234224


namespace div_by_squares_l234_234780

variables {R : Type*} [CommRing R] (a b c x y z : R)

theorem div_by_squares (a b c x y z : R) :
  (a * y - b * x) ^ 2 + (b * z - c * y) ^ 2 + (c * x - a * z) ^ 2 + (a * x + b * y + c * z) ^ 2 =
    (a ^ 2 + b ^ 2 + c ^ 2) * (x ^ 2 + y ^ 2 + z ^ 2) := sorry

end div_by_squares_l234_234780


namespace eval_7_star_3_l234_234129

def operation (a b : ℕ) : ℕ := (4 * a + 5 * b - a * b)

theorem eval_7_star_3 : operation 7 3 = 22 :=
  by {
    -- substitution and calculation steps
    sorry
  }

end eval_7_star_3_l234_234129


namespace max_bananas_l234_234071

theorem max_bananas (a o b : ℕ) (h_a : a ≥ 1) (h_o : o ≥ 1) (h_b : b ≥ 1) (h_eq : 3 * a + 5 * o + 8 * b = 100) : b ≤ 11 :=
by {
  sorry
}

end max_bananas_l234_234071


namespace max_vertices_no_rectangle_l234_234524

/-- In a regular 2016-sided polygon (2016-gon), prove that the maximum number of vertices 
    that can be marked such that no four marked vertices form the vertices of a rectangle is 1009. -/
theorem max_vertices_no_rectangle (n : ℕ) (h : n = 2016) : 
  ∃ (m : ℕ), m = 1009 ∧ 
    ∀ (marked : finset (fin n)), 
      marked.card ≤ m → 
      (¬ ∃ (a b c d : fin n), a ∈ marked ∧ b ∈ marked ∧ c ∈ marked ∧ d ∈ marked ∧ 
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
        (is_rectangle a b c d)) := 
begin 
  sorry 
end

/-- Predicate to determine if four vertices form a rectangle in a regular n-gon. -/
def is_rectangle (a b c d : fin 2016) : Prop := 
  ∃ (k : ℕ), k ∈ finset.range 1008 ∧ 
    ((a = fin.of_nat k) ∧ (b = fin.of_nat (k + 1008)) ∧ 
     (c = fin.of_nat (k + 1008 + 1)) ∧ (d = fin.of_nat (k + 1)) ∨ 
     (a = fin.of_nat (k + 1008)) ∧ (b = fin.of_nat k) ∧ 
     (c = fin.of_nat (k + 1)) ∧ (d = fin.of_nat (k + 1008 + 1)))

end max_vertices_no_rectangle_l234_234524


namespace ratio_of_average_speeds_l234_234366

theorem ratio_of_average_speeds
    (time_eddy : ℝ) (distance_eddy : ℝ)
    (time_freddy : ℝ) (distance_freddy : ℝ) :
  time_eddy = 3 ∧ distance_eddy = 600 ∧ time_freddy = 4 ∧ distance_freddy = 460 →
  (distance_eddy / time_eddy) / (distance_freddy / time_freddy) = 200 / 115 :=
by
  sorry

end ratio_of_average_speeds_l234_234366


namespace sum_of_three_numbers_l234_234506

theorem sum_of_three_numbers :
  ∃ (a b c : ℕ), 
    (a ≤ b ∧ b ≤ c) ∧ 
    (b = 8) ∧ 
    ((a + b + c) / 3 = a + 8) ∧ 
    ((a + b + c) / 3 = c - 20) ∧ 
    (a + b + c = 60) :=
by
  sorry

end sum_of_three_numbers_l234_234506


namespace ratio_of_areas_l234_234643

-- Define the side lengths of Squared B and Square C
variables (y : ℝ)

-- Define the areas of Square B and C
def area_B := (2 * y) * (2 * y)
def area_C := (8 * y) * (8 * y)

-- The theorem statement proving the ratio of the areas
theorem ratio_of_areas : area_B y / area_C y = 1 / 16 := 
by sorry

end ratio_of_areas_l234_234643


namespace simplify_polynomial_l234_234010

variable (r : ℝ)

theorem simplify_polynomial : (2 * r^2 + 5 * r - 7) - (r^2 + 9 * r - 3) = r^2 - 4 * r - 4 := by
  sorry

end simplify_polynomial_l234_234010


namespace members_playing_both_l234_234757

variable (N B T Neither BT : ℕ)

theorem members_playing_both (hN : N = 30) (hB : B = 17) (hT : T = 17) (hNeither : Neither = 2) 
  (hBT : BT = B + T - (N - Neither)) : BT = 6 := 
by 
  rw [hN, hB, hT, hNeither] at hBT
  exact hBT

end members_playing_both_l234_234757


namespace remaining_money_after_expenditures_l234_234681

def initial_amount : ℝ := 200.50
def spent_on_sweets : ℝ := 35.25
def given_to_each_friend : ℝ := 25.20

theorem remaining_money_after_expenditures :
  ((initial_amount - spent_on_sweets) - 2 * given_to_each_friend) = 114.85 :=
by
  sorry

end remaining_money_after_expenditures_l234_234681


namespace prob_first_red_light_at_C_is_correct_expected_waiting_time_is_correct_l234_234993

noncomputable def prob_red_light_first_time_at_C : ℚ :=
  let prob_A := 1 / 3;
  let prob_B := 1 / 4;
  let prob_C := 3 / 4;
  (1 - prob_A) * (1 - prob_B) * prob_C

theorem prob_first_red_light_at_C_is_correct :
  prob_red_light_first_time_at_C = 3 / 8 :=
by 
  -- Proof needs to be provided here
  sorry

noncomputable def expected_waiting_time : ℚ :=
  let prob_A := 1 / 3;
  let prob_B := 1 / 4;
  let prob_C := 3 / 4;
  let times := [(0, (1 - prob_A) * (1 - prob_B) * (1 - prob_C)),
                (40, prob_A * (1 - prob_B) * (1 - prob_C)),
                (20, (1 - prob_A) * prob_B * (1 - prob_C)),
                (80, (1 - prob_A) * (1 - prob_B) * prob_C),
                (60, prob_A * prob_B * (1 - prob_C)),
                (100, (1 - prob_A) * prob_B * prob_C),
                (120, prob_A * (1 - prob_B) * prob_C),
                (140, prob_A * prob_B * prob_C)];
  times.foldl (λ acc (t, p), acc + t * p) 0

theorem expected_waiting_time_is_correct : 
  expected_waiting_time = 235 / 3 :=
by 
  -- Proof needs to be provided here
  sorry

end prob_first_red_light_at_C_is_correct_expected_waiting_time_is_correct_l234_234993


namespace solve_for_m_l234_234883

theorem solve_for_m (x y m : ℤ) (h1 : x - 2 * y = -3) (h2 : 2 * x + 3 * y = m - 1) (h3 : x = -y) : m = 2 :=
by
  sorry

end solve_for_m_l234_234883


namespace player_match_count_l234_234063

open Real

theorem player_match_count (n : ℕ) : 
  (∃ T, T = 32 * n ∧ (T + 98) / (n + 1) = 38) → n = 10 :=
by
  sorry

end player_match_count_l234_234063


namespace trapezoid_height_proof_l234_234465

-- Given lengths of the diagonals and the midline of the trapezoid
def diagonal1Length : ℝ := 6
def diagonal2Length : ℝ := 8
def midlineLength : ℝ := 5

-- Target to prove: Height of the trapezoid
def trapezoidHeight : ℝ := 4.8

theorem trapezoid_height_proof :
  ∀ (d1 d2 m : ℝ), d1 = diagonal1Length → d2 = diagonal2Length → m = midlineLength → trapezoidHeight = 4.8 :=
by intros d1 d2 m hd1 hd2 hm; sorry

end trapezoid_height_proof_l234_234465


namespace mean_transformation_l234_234252

theorem mean_transformation (x1 x2 x3 x4 : ℝ)
                            (h_pos : 0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4)
                            (s2 : ℝ)
                            (h_var : s2 = (1 / 4) * (x1^2 + x2^2 + x3^2 + x4^2 - 16)) :
                            (x1 + 2 + x2 + 2 + x3 + 2 + x4 + 2) / 4 = 4 :=
by
  sorry

end mean_transformation_l234_234252


namespace find_x_l234_234718

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 210) (h2 : ⌈x⌉ - 1 < x ∧ x ≤ ⌈x⌉) (h3 : 0 < x) : x = 14 :=
sorry

end find_x_l234_234718


namespace cone_volume_l234_234282

noncomputable def volume_of_cone_from_lateral_surface (radius_semicircle : ℝ) 
  (circumference_base : ℝ := 2 * radius_semicircle * Real.pi) 
  (radius_base : ℝ := circumference_base / (2 * Real.pi)) 
  (height_cone : ℝ := Real.sqrt ((radius_semicircle:ℝ) ^ 2 - (radius_base:ℝ) ^ 2)) : ℝ := 
  (1 / 3) * Real.pi * (radius_base ^ 2) * height_cone

theorem cone_volume (h_semicircle : 2 = 2) : volume_of_cone_from_lateral_surface 2 = (Real.sqrt 3) / 3 * Real.pi := 
by
  -- Importing Real.sqrt and Real.pi to bring them into scope
  sorry

end cone_volume_l234_234282


namespace expenditure_ratio_l234_234187

theorem expenditure_ratio 
  (I1 : ℝ) (I2 : ℝ) (E1 : ℝ) (E2 : ℝ) (S1 : ℝ) (S2 : ℝ)
  (h1 : I1 = 3500)
  (h2 : I2 = (4 / 5) * I1)
  (h3 : S1 = I1 - E1)
  (h4 : S2 = I2 - E2)
  (h5 : S1 = 1400)
  (h6 : S2 = 1400) : 
  E1 / E2 = 3 / 2 :=
by
  -- Steps of the proof will go here
  sorry

end expenditure_ratio_l234_234187


namespace ferris_wheel_capacity_l234_234785

theorem ferris_wheel_capacity 
  (num_seats : ℕ)
  (people_per_seat : ℕ)
  (h1 : num_seats = 4)
  (h2 : people_per_seat = 5) :
  num_seats * people_per_seat = 20 := by
  sorry

end ferris_wheel_capacity_l234_234785


namespace polygon_number_of_sides_l234_234835

theorem polygon_number_of_sides (P : ℝ) (L : ℝ) (n : ℕ) : P = 180 ∧ L = 15 ∧ n = P / L → n = 12 := by
  sorry

end polygon_number_of_sides_l234_234835


namespace bill_spots_l234_234225

theorem bill_spots (b p : ℕ) (h1 : b + p = 59) (h2 : b = 2 * p - 1) : b = 39 := by
  sorry

end bill_spots_l234_234225


namespace sum_of_three_numbers_from_1_to_100_is_odd_l234_234043

open Probability

noncomputable def probability_sum_odd : ℚ :=
  let numbers := finset.range 101
  let balls := numbers.filter (λ n, n > 0)
  let odd_count := (balls.filter (λ n, n % 2 = 1)).card
  let even_count := (balls.filter (λ n, n % 2 = 0)).card
  let odd_prob := (odd_count : ℚ) / balls.card
  let even_prob := (even_count : ℚ) / balls.card
  let odd_sum_prob := (even_prob * even_prob * odd_prob) + (odd_prob * odd_prob * odd_prob)
  odd_sum_prob

theorem sum_of_three_numbers_from_1_to_100_is_odd : probability_sum_odd = 1 / 2 :=
sorry

end sum_of_three_numbers_from_1_to_100_is_odd_l234_234043


namespace friends_team_division_l234_234751

theorem friends_team_division :
  let num_friends : ℕ := 8
  let num_teams : ℕ := 4
  let ways_to_divide := num_teams ^ num_friends
  ways_to_divide = 65536 :=
by
  sorry

end friends_team_division_l234_234751


namespace find_alpha_l234_234255

noncomputable def angle_in_interval (α : ℝ) : Prop :=
  370 < α ∧ α < 520 

theorem find_alpha (α : ℝ) (h_cos : Real.cos α = 1 / 2) (h_interval: angle_in_interval α) : α = 420 :=
sorry

end find_alpha_l234_234255


namespace problem_a_problem_b_l234_234536

-- Define necessary elements for the problem
def is_divisible_by_seven (n : ℕ) : Prop := n % 7 = 0

-- Define the method to check divisibility by seven
noncomputable def check_divisibility_by_seven (n : ℕ) : ℕ :=
  let last_digit := n % 10
  let remaining_digits := n / 10
  remaining_digits - 2 * last_digit

-- Problem a: Prove that 4578 is divisible by 7
theorem problem_a : is_divisible_by_seven 4578 :=
  sorry

-- Problem b: Prove that there are 13 three-digit numbers of the form AB5 divisible by 7
theorem problem_b : ∃ (count : ℕ), count = 13 ∧ (∀ a b : ℕ, a ≠ 0 ∧ 1 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 → is_divisible_by_seven (100 * a + 10 * b + 5) → count = count + 1) :=
  sorry

end problem_a_problem_b_l234_234536


namespace maximize_GDP_investment_l234_234665

def invest_A_B_max_GDP : Prop :=
  ∃ (A B : ℝ), 
  A + B ≤ 30 ∧
  20000 * A + 40000 * B ≤ 1000000 ∧
  24 * A + 32 * B ≥ 800 ∧
  A = 20 ∧ B = 10

theorem maximize_GDP_investment : invest_A_B_max_GDP :=
by
  sorry

end maximize_GDP_investment_l234_234665


namespace total_treats_is_237_l234_234964

def num_children : ℕ := 3
def hours_out : ℕ := 4
def houses_visited (hour : ℕ) : ℕ :=
  match hour with
  | 1 => 4
  | 2 => 6
  | 3 => 5
  | 4 => 7
  | _ => 0

def treats_per_kid_per_house (hour : ℕ) : ℕ :=
  match hour with
  | 1 => 3
  | 3 => 3
  | 2 => 4
  | 4 => 4
  | _ => 0

def total_treats : ℕ :=
  (houses_visited 1 * treats_per_kid_per_house 1 * num_children) + 
  (houses_visited 2 * treats_per_kid_per_house 2 * num_children) +
  (houses_visited 3 * treats_per_kid_per_house 3 * num_children) +
  (houses_visited 4 * treats_per_kid_per_house 4 * num_children)

theorem total_treats_is_237 : total_treats = 237 :=
by
  -- Placeholder for the proof
  sorry

end total_treats_is_237_l234_234964


namespace units_digit_24_pow_4_plus_42_pow_4_l234_234036

theorem units_digit_24_pow_4_plus_42_pow_4 : 
    (24^4 + 42^4) % 10 = 2 :=
by
  sorry

end units_digit_24_pow_4_plus_42_pow_4_l234_234036


namespace candies_eaten_l234_234576

-- Definitions

def Andrey_rate_eq_Boris_rate (candies_eaten_by_Andrey candies_eaten_by_Boris : ℕ) : Prop :=
  candies_eaten_by_Andrey / 4 = candies_eaten_by_Boris / 3

def Denis_rate_eq_Andrey_rate (candies_eaten_by_Denis candies_eaten_by_Andrey : ℕ) : Prop :=
  candies_eaten_by_Denis / 7 = candies_eaten_by_Andrey / 6

def total_candies (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) : Prop :=
  candies_eaten_by_Andrey + candies_eaten_by_Boris + candies_eaten_by_Denis = 70

-- Theorem to prove the candies eaten by Andrey, Boris, and Denis
theorem candies_eaten (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) :
  Andrey_rate_eq_Boris_rate candies_eaten_by_Andrey candies_eaten_by_Boris →
  Denis_rate_eq_Andrey_rate candies_eaten_by_Denis candies_eaten_by_Andrey →
  total_candies candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis →
  candies_eaten_by_Andrey = 24 ∧ candies_eaten_by_Boris = 18 ∧ candies_eaten_by_Denis = 28 :=
  by sorry

end candies_eaten_l234_234576


namespace value_of_expression_l234_234272

theorem value_of_expression (a b : ℤ) (h : 1^2 + a * 1 + 2 * b = 0) : 2023 - a - 2 * b = 2024 :=
by
  sorry

end value_of_expression_l234_234272


namespace paul_eats_sandwiches_l234_234171

theorem paul_eats_sandwiches (S : ℕ) (h : (S + 2 * S + 4 * S) * 2 = 28) : S = 2 :=
by
  sorry

end paul_eats_sandwiches_l234_234171


namespace probability_at_least_one_humanities_l234_234871

theorem probability_at_least_one_humanities :
  let morning_classes := ["mathematics", "Chinese", "politics", "geography"]
  let afternoon_classes := ["English", "history", "physical_education"]
  let humanities := ["politics", "history", "geography"]
  let total_choices := List.length morning_classes * List.length afternoon_classes
  let favorable_morning := List.length (List.filter (fun x => x ∈ humanities) morning_classes)
  let favorable_afternoon := List.length (List.filter (fun x => x ∈ humanities) afternoon_classes)
  let favorable_choices := favorable_morning * List.length afternoon_classes + favorable_afternoon * (List.length morning_classes - favorable_morning)
  (favorable_choices / total_choices) = (2 / 3) := by sorry

end probability_at_least_one_humanities_l234_234871


namespace average_age_of_women_is_37_33_l234_234332

noncomputable def women_average_age (A : ℝ) : ℝ :=
  let total_age_men := 12 * A
  let removed_men_age := (25 : ℝ) + 15 + 30
  let new_average := A + 3.5
  let total_age_with_women := 12 * new_average
  let total_age_women := total_age_with_women -  (total_age_men - removed_men_age)
  total_age_women / 3

theorem average_age_of_women_is_37_33 (A : ℝ) (h_avg : women_average_age A = 37.33) :
  true :=
by
  sorry

end average_age_of_women_is_37_33_l234_234332


namespace John_lost_3_ebook_readers_l234_234622

-- Definitions based on the conditions
def A : Nat := 50  -- Anna bought 50 eBook readers
def J : Nat := A - 15  -- John bought 15 less than Anna
def total : Nat := 82  -- Total eBook readers now

-- The number of eBook readers John has after the loss:
def J_after_loss : Nat := total - A

-- The number of eBook readers John lost:
def John_loss : Nat := J - J_after_loss

theorem John_lost_3_ebook_readers : John_loss = 3 :=
by
  sorry

end John_lost_3_ebook_readers_l234_234622


namespace horizontal_length_of_rectangle_l234_234042

theorem horizontal_length_of_rectangle
  (P : ℕ)
  (h v : ℕ)
  (hP : P = 54)
  (hv : v = h - 3) :
  2*h + 2*v = 54 → h = 15 :=
by sorry

end horizontal_length_of_rectangle_l234_234042


namespace find_m_value_l234_234260

-- Define the conditions
def quadratic_has_real_roots (m : ℝ) : Prop :=
  let Δ := (2 * m - 1)^2 - 4 * m^2 in Δ ≥ 0

def correct_m_value (m : ℝ) : Prop :=
  let quadratic_solution_product := (x1 + 1) * (x2 + 1) in
  quadratic_solution_product = 3 → m = -3

theorem find_m_value (m : ℝ) :
  quadratic_has_real_roots m →
  correct_m_value m :=
  sorry

end find_m_value_l234_234260


namespace sum_of_squares_eq_power_l234_234941

theorem sum_of_squares_eq_power (n : ℕ) : ∃ x y z : ℕ, x^2 + y^2 = z^n :=
sorry

end sum_of_squares_eq_power_l234_234941


namespace part1_solution_set_part2_range_a_l234_234436

-- Define the function f
noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

-- Part 1: Proving the solution set of the inequality
theorem part1_solution_set (x : ℝ) :
  (∀ x, f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2: Finding the range of values for a
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a > -a ↔ a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_a_l234_234436


namespace candies_eaten_l234_234578

-- Definitions

def Andrey_rate_eq_Boris_rate (candies_eaten_by_Andrey candies_eaten_by_Boris : ℕ) : Prop :=
  candies_eaten_by_Andrey / 4 = candies_eaten_by_Boris / 3

def Denis_rate_eq_Andrey_rate (candies_eaten_by_Denis candies_eaten_by_Andrey : ℕ) : Prop :=
  candies_eaten_by_Denis / 7 = candies_eaten_by_Andrey / 6

def total_candies (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) : Prop :=
  candies_eaten_by_Andrey + candies_eaten_by_Boris + candies_eaten_by_Denis = 70

-- Theorem to prove the candies eaten by Andrey, Boris, and Denis
theorem candies_eaten (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) :
  Andrey_rate_eq_Boris_rate candies_eaten_by_Andrey candies_eaten_by_Boris →
  Denis_rate_eq_Andrey_rate candies_eaten_by_Denis candies_eaten_by_Andrey →
  total_candies candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis →
  candies_eaten_by_Andrey = 24 ∧ candies_eaten_by_Boris = 18 ∧ candies_eaten_by_Denis = 28 :=
  by sorry

end candies_eaten_l234_234578


namespace intersection_has_one_element_l234_234904

noncomputable def A (a : ℝ) : Set ℝ := {1, a, 5}
noncomputable def B (a : ℝ) : Set ℝ := {2, a^2 + 1}

theorem intersection_has_one_element (a : ℝ) (h : ∃ x, A a ∩ B a = {x}) : a = 0 ∨ a = -2 :=
by {
  sorry
}

end intersection_has_one_element_l234_234904


namespace find_y_l234_234752

theorem find_y (y : ℕ) (hy1 : y % 9 = 0) (hy2 : y^2 > 200) (hy3 : y < 30) : y = 18 :=
sorry

end find_y_l234_234752


namespace chord_length_intercepted_l234_234498

theorem chord_length_intercepted 
  (line_eq : ∀ x y : ℝ, 3 * x - 4 * y = 0)
  (circle_eq : ∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 2) : 
  ∃ l : ℝ, l = 2 :=
by 
  sorry

end chord_length_intercepted_l234_234498


namespace thief_speed_l234_234845

theorem thief_speed
  (distance_initial : ℝ := 100 / 1000) -- distance (100 meters converted to kilometers)
  (policeman_speed : ℝ := 10) -- speed of the policeman in km/hr
  (thief_distance : ℝ := 400 / 1000) -- distance thief runs in kilometers (400 meters converted)
  : ∃ V_t : ℝ, V_t = 8 :=
by
  sorry

end thief_speed_l234_234845


namespace regular_polygon_sides_l234_234831

theorem regular_polygon_sides (perimeter side_length : ℕ) (h_perim : perimeter = 180) (h_side : side_length = 15) : 
  let n := perimeter / side_length in n = 12 :=
by
  have h_n : n = perimeter / side_length := rfl
  rw [h_perim, h_side] at h_n
  have h_res : 180 / 15 = 12 := rfl
  rw h_res at h_n
  exact h_n

end regular_polygon_sides_l234_234831


namespace volunteer_arrangement_l234_234064

theorem volunteer_arrangement (volunteers : Fin 5) (elderly : Fin 2) 
  (h1 : elderly.1 ≠ 0 ∧ elderly.1 ≠ 6) : 
  ∃ arrangements : ℕ, arrangements = 960 := 
sorry

end volunteer_arrangement_l234_234064


namespace find_x_when_water_added_l234_234352

variable (m x : ℝ)

theorem find_x_when_water_added 
  (h1 : m > 25)
  (h2 : (m * m / 100) = ((m - 15) / 100) * (m + x)) :
  x = 15 * m / (m - 15) :=
sorry

end find_x_when_water_added_l234_234352


namespace total_scarves_l234_234163

def total_yarns_red : ℕ := 2
def total_yarns_blue : ℕ := 6
def total_yarns_yellow : ℕ := 4
def scarves_per_yarn : ℕ := 3

theorem total_scarves : 
  (total_yarns_red * scarves_per_yarn) + 
  (total_yarns_blue * scarves_per_yarn) + 
  (total_yarns_yellow * scarves_per_yarn) = 36 := 
by
  sorry

end total_scarves_l234_234163


namespace instructors_teach_together_in_360_days_l234_234091

def Felicia_teaches_every := 5
def Greg_teaches_every := 3
def Hannah_teaches_every := 9
def Ian_teaches_every := 2
def Joy_teaches_every := 8

def lcm_multiple (a b c d e : ℕ) : ℕ := Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e)))

theorem instructors_teach_together_in_360_days :
  lcm_multiple Felicia_teaches_every
               Greg_teaches_every
               Hannah_teaches_every
               Ian_teaches_every
               Joy_teaches_every = 360 :=
by
  -- Since the real proof is omitted, we close with sorry
  sorry

end instructors_teach_together_in_360_days_l234_234091


namespace distance_ratio_gt_9_l234_234469

theorem distance_ratio_gt_9 (points : Fin 1997 → ℝ × ℝ × ℝ) (M m : ℝ) :
  (∀ i j, i ≠ j → dist (points i) (points j) ≤ M) →
  (∀ i j, i ≠ j → dist (points i) (points j) ≥ m) →
  m ≠ 0 →
  M / m > 9 :=
by
  sorry

end distance_ratio_gt_9_l234_234469


namespace positive_difference_of_two_numbers_l234_234346

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_of_two_numbers_l234_234346


namespace sum_of_squares_l234_234872

theorem sum_of_squares (r b s : ℕ) 
  (h1 : 2 * r + 3 * b + s = 80) 
  (h2 : 4 * r + 2 * b + 3 * s = 98) : 
  r^2 + b^2 + s^2 = 485 := 
by {
  sorry
}

end sum_of_squares_l234_234872


namespace tax_diminished_by_20_percent_l234_234189

theorem tax_diminished_by_20_percent
(T C : ℝ) 
(hT : T > 0) 
(hC : C > 0) 
(X : ℝ) 
(h_increased_consumption : ∀ (T C : ℝ), (C * 1.15) = C + 0.15 * C)
(h_decrease_revenue : T * (1 - X / 100) * C * 1.15 = T * C * 0.92) :
X = 20 := 
sorry

end tax_diminished_by_20_percent_l234_234189


namespace balls_in_each_package_l234_234928

theorem balls_in_each_package (x : ℕ) (h : 21 * x = 399) : x = 19 :=
by
  sorry

end balls_in_each_package_l234_234928


namespace nancy_kept_chips_l234_234167

def nancy_initial_chips : ℕ := 22
def chips_given_to_brother : ℕ := 7
def chips_given_to_sister : ℕ := 5

theorem nancy_kept_chips : nancy_initial_chips - (chips_given_to_brother + chips_given_to_sister) = 10 :=
by
  sorry

end nancy_kept_chips_l234_234167


namespace area_of_trapezoid_l234_234152

variable (a d : ℝ)
variable (h b1 b2 : ℝ)

def is_arithmetic_progression (a d : ℝ) (h b1 b2 : ℝ) : Prop :=
  h = a ∧ b1 = a + d ∧ b2 = a - d

theorem area_of_trapezoid (a d : ℝ) (h b1 b2 : ℝ) (hAP : is_arithmetic_progression a d h b1 b2) :
  ∃ J : ℝ, J = a^2 ∧ ∀ x : ℝ, 0 ≤ x → (J = x → x ≥ 0) :=
by
  sorry

end area_of_trapezoid_l234_234152


namespace cubic_function_not_monotonically_increasing_l234_234960

theorem cubic_function_not_monotonically_increasing (b : ℝ) :
  ¬(∀ x y : ℝ, x ≤ y → (1/3)*x^3 + b*x^2 + (b+2)*x + 3 ≤ (1/3)*y^3 + b*y^2 + (b+2)*y + 3) ↔ b ∈ (Set.Iio (-1) ∪ Set.Ioi 2) :=
by sorry

end cubic_function_not_monotonically_increasing_l234_234960


namespace middle_integer_of_sum_is_120_l234_234514

-- Define the condition that three consecutive integers sum to 360
def consecutive_integers_sum_to (n : ℤ) (sum : ℤ) : Prop :=
  (n - 1) + n + (n + 1) = sum

-- The statement to prove
theorem middle_integer_of_sum_is_120 (n : ℤ) :
  consecutive_integers_sum_to n 360 → n = 120 :=
by
  sorry

end middle_integer_of_sum_is_120_l234_234514


namespace miles_driven_each_day_l234_234330

-- Definition of the given conditions
def total_miles : ℝ := 1250
def number_of_days : ℝ := 5.0

-- The statement to be proved
theorem miles_driven_each_day :
  total_miles / number_of_days = 250 :=
by
  sorry

end miles_driven_each_day_l234_234330


namespace unoccupied_seats_l234_234102

theorem unoccupied_seats (rows chairs_per_row seats_taken : Nat) (h1 : rows = 40)
  (h2 : chairs_per_row = 20) (h3 : seats_taken = 790) :
  rows * chairs_per_row - seats_taken = 10 :=
by
  sorry

end unoccupied_seats_l234_234102


namespace school_committee_count_l234_234411

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binom (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

def valid_committees_count (total_students total_teachers committee_size : ℕ) : ℕ :=
  let total_people := total_students + total_teachers
  let total_combinations := binom total_people committee_size
  let student_only_combinations := binom total_students committee_size
  total_combinations - student_only_combinations

theorem school_committee_count :
  valid_committees_count 12 3 9 = 4785 :=
by {
  -- Translate the calculation described in the problem to a Lean statement.
  let total_combinations := binom 15 9,
  let student_only_combinations := binom 12 9,
  let valid_com := total_combinations - student_only_combinations,

  -- General binomial coefficient computation simplification is omitted.
  -- Simplify the exact computation here using known binomial identities as required.
  have h1 : binom 15 9 = 5005 := sorry,
  have h2 : binom 12 9 = 220 := sorry,
  
  -- Valid committee count check
  have h3: valid_com = 5005 - 220 := sorry,
  have h4: valid_com = 4785 := by norm_num,
  exact h4,
}

end school_committee_count_l234_234411


namespace zoey_holidays_in_a_year_l234_234364

-- Definitions based on the conditions
def holidays_per_month := 2
def months_in_year := 12

-- Lean statement representing the proof problem
theorem zoey_holidays_in_a_year : (holidays_per_month * months_in_year) = 24 :=
by sorry

end zoey_holidays_in_a_year_l234_234364


namespace pqrs_sum_l234_234308

noncomputable def distinct_real_numbers (p q r s : ℝ) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

theorem pqrs_sum (p q r s : ℝ) 
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s)
  (distinct : distinct_real_numbers p q r s) :
  p + q + r + s = -13 :=
  sorry

end pqrs_sum_l234_234308


namespace part1_problem_part2_problem_l234_234982

/-- Given initial conditions and price adjustment, prove the expected number of helmets sold and the monthly profit. -/
theorem part1_problem (initial_price : ℕ) (initial_sales : ℕ) 
(price_reduction : ℕ) (sales_per_reduction : ℕ) (cost_price : ℕ) : 
  initial_price = 80 → initial_sales = 200 → price_reduction = 10 → 
  sales_per_reduction = 20 → cost_price = 50 → 
  (initial_sales + price_reduction * sales_per_reduction = 400) ∧ 
  ((initial_price - price_reduction - cost_price) * 
  (initial_sales + price_reduction * sales_per_reduction) = 8000) :=
by
  intros
  sorry

/-- Given initial conditions and profit target, prove the expected selling price of helmets. -/
theorem part2_problem (initial_price : ℕ) (initial_sales : ℕ) 
(cost_price : ℕ) (profit_target : ℕ) (x : ℕ) :
  initial_price = 80 → initial_sales = 200 → cost_price = 50 → 
  profit_target = 7500 → (x = 15) → 
  (initial_price - x = 65) :=
by
  intros
  sorry

end part1_problem_part2_problem_l234_234982


namespace tire_circumference_l234_234367

theorem tire_circumference 
  (rev_per_min : ℝ) -- revolutions per minute
  (car_speed_kmh : ℝ) -- car speed in km/h
  (conversion_factor : ℝ) -- conversion factor for speed from km/h to m/min
  (min_to_meter : ℝ) -- multiplier to convert minutes to meters
  (C : ℝ) -- circumference of the tire in meters
  : rev_per_min = 400 ∧ car_speed_kmh = 120 ∧ conversion_factor = 1000 / 60 ∧ min_to_meter = 1000 / 60 ∧ (C * rev_per_min = car_speed_kmh * min_to_meter) → C = 5 :=
by
  sorry

end tire_circumference_l234_234367


namespace sum_of_coordinates_l234_234283

def g : ℝ → ℝ := sorry
def h (x : ℝ) : ℝ := (g x)^3

theorem sum_of_coordinates (hg : g 4 = 8) : 4 + h 4 = 516 :=
by
  sorry

end sum_of_coordinates_l234_234283


namespace oranges_for_juice_l234_234181

theorem oranges_for_juice (total_oranges : ℝ) (exported_percentage : ℝ) (juice_percentage : ℝ) :
  total_oranges = 7 →
  exported_percentage = 0.30 →
  juice_percentage = 0.60 →
  (total_oranges * (1 - exported_percentage) * juice_percentage) = 2.9 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end oranges_for_juice_l234_234181


namespace average_salary_for_company_l234_234983

theorem average_salary_for_company
    (number_of_managers : Nat)
    (number_of_associates : Nat)
    (average_salary_managers : Nat)
    (average_salary_associates : Nat)
    (hnum_managers : number_of_managers = 15)
    (hnum_associates : number_of_associates = 75)
    (has_managers : average_salary_managers = 90000)
    (has_associates : average_salary_associates = 30000) : 
    (number_of_managers * average_salary_managers + number_of_associates * average_salary_associates) / 
    (number_of_managers + number_of_associates) = 40000 := 
    by
    sorry

end average_salary_for_company_l234_234983


namespace function_intersection_le_one_l234_234480

theorem function_intersection_le_one (f : ℝ → ℝ)
  (h : ∀ x t : ℝ, t ≠ 0 → t * (f (x + t) - f x) > 0) :
  ∀ a : ℝ, ∃! x : ℝ, f x = a :=
by 
sorry

end function_intersection_le_one_l234_234480


namespace right_triangle_legs_sum_l234_234695

theorem right_triangle_legs_sum (x : ℕ) (hx1 : x * x + (x + 1) * (x + 1) = 41 * 41) : x + (x + 1) = 59 :=
by sorry

end right_triangle_legs_sum_l234_234695


namespace new_person_weight_l234_234182

/-- 
The average weight of 10 persons increases by 6.3 kg when a new person comes 
in place of one of them weighing 65 kg. Prove that the weight of the new person 
is 128 kg.
-/
theorem new_person_weight (w_old : ℝ) (n : ℝ) (delta_w : ℝ) (w_new : ℝ) 
  (h1 : w_old = 65) 
  (h2 : n = 10) 
  (h3 : delta_w = 6.3) 
  (h4 : w_new = w_old + n * delta_w) : 
  w_new = 128 :=
by 
  rw [h1, h2, h3] at h4 
  rw [h4]
  norm_num

end new_person_weight_l234_234182


namespace number_of_roses_per_set_l234_234025

-- Define the given conditions
def total_days : ℕ := 7
def sets_per_day : ℕ := 2
def total_roses : ℕ := 168

-- Define the statement to be proven
theorem number_of_roses_per_set : 
  (sets_per_day * total_days * (total_roses / (sets_per_day * total_days)) = total_roses) ∧ 
  (total_roses / (sets_per_day * total_days) = 12) :=
by 
  sorry

end number_of_roses_per_set_l234_234025


namespace ellipse_eccentricity_l234_234844

theorem ellipse_eccentricity (a b : ℝ) (h_ab : a > b) (h_b : b > 0) (c : ℝ)
  (h_ellipse : (b^2 / c^2) = 3)
  (eccentricity_eq : ∀ (e : ℝ), e = c / a ↔ e = 1 / 2) : 
  ∃ e, e = (c / a) :=
by {
  sorry
}

end ellipse_eccentricity_l234_234844


namespace units_digit_x4_invx4_l234_234607

theorem units_digit_x4_invx4 (x : ℝ) (h : x^2 - 12 * x + 1 = 0) : 
  (x^4 + (1 / x)^4) % 10 = 2 := 
by
  sorry

end units_digit_x4_invx4_l234_234607


namespace sandwich_cost_l234_234033

theorem sandwich_cost (soda_cost sandwich_cost total_cost : ℝ) (h1 : soda_cost = 0.87) (h2 : total_cost = 10.46) (h3 : 4 * soda_cost + 2 * sandwich_cost = total_cost) :
  sandwich_cost = 3.49 :=
by
  sorry

end sandwich_cost_l234_234033


namespace trig_inequality_l234_234730

open Real

theorem trig_inequality (a b c : ℝ) (h₁ : a = sin (2 * π / 7))
  (h₂ : b = cos (2 * π / 7)) (h₃ : c = tan (2 * π / 7)) :
  c > a ∧ a > b :=
by 
  sorry

end trig_inequality_l234_234730


namespace simplify_fraction_l234_234360

theorem simplify_fraction : 
  1 + (1 / (1 + (1 / (2 + 1)))) = 7 / 4 :=
by
  sorry

end simplify_fraction_l234_234360


namespace ellipse_area_quadrants_eq_zero_l234_234020

theorem ellipse_area_quadrants_eq_zero 
(E : Type)
(x y : E → ℝ) 
(h_ellipse : ∀ (x y : ℝ), (x - 19)^2 / (19 * 1998) + (y - 98)^2 / (98 * 1998) = 1998) 
(R1 R2 R3 R4 : ℝ)
(H1 : ∀ (R1 R2 R3 R4 : ℝ), R1 = R_ellipse / 4 ∧ R2 = R_ellipse / 4 ∧ R3 = R_ellipse / 4 ∧ R4 = R_ellipse / 4)
: R1 - R2 + R3 - R4 = 0 := 
by 
sorry

end ellipse_area_quadrants_eq_zero_l234_234020


namespace hyperbola_eccentricity_range_l234_234210

theorem hyperbola_eccentricity_range
  (a b t : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_condition : a > b) :
  ∃ e : ℝ, e = Real.sqrt (1 + (b / a)^2) ∧ 1 < e ∧ e < Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_range_l234_234210


namespace ratio_addition_l234_234813

theorem ratio_addition (x : ℝ) : 
  (2 + x) / (3 + x) = 4 / 5 → x = 2 :=
by
  sorry

end ratio_addition_l234_234813


namespace lucas_siblings_product_is_35_l234_234463

-- Definitions based on the given conditions
def total_girls (lauren_sisters : ℕ) : ℕ := lauren_sisters + 1
def total_boys (lauren_brothers : ℕ) : ℕ := lauren_brothers + 1

-- Given conditions
def lauren_sisters : ℕ := 4
def lauren_brothers : ℕ := 7

-- Compute number of sisters (S) and brothers (B) Lucas has
def lucas_sisters : ℕ := total_girls lauren_sisters
def lucas_brothers : ℕ := lauren_brothers

theorem lucas_siblings_product_is_35 : 
  (lucas_sisters * lucas_brothers = 35) := by
  -- Asserting the correctness based on given family structure conditions
  sorry

end lucas_siblings_product_is_35_l234_234463


namespace sammy_mistakes_l234_234855

def bryan_score : ℕ := 20
def jen_score : ℕ := bryan_score + 10
def sammy_score : ℕ := jen_score - 2
def total_points : ℕ := 35
def mistakes : ℕ := total_points - sammy_score

theorem sammy_mistakes : mistakes = 7 := by
  sorry

end sammy_mistakes_l234_234855


namespace angle_A_eq_pi_over_3_perimeter_eq_24_l234_234618

namespace TriangleProof

-- We introduce the basic setup for the triangle
variables {A B C : ℝ} {a b c : ℝ}

-- Given condition
axiom condition : 2 * b = 2 * a * Real.cos C + c

-- Part 1: Prove angle A is π/3
theorem angle_A_eq_pi_over_3 (h : 2 * b = 2 * a * Real.cos C + c) :
  A = Real.pi / 3 :=
sorry

-- Part 2: Given a = 10 and the area is 8√3, prove perimeter is 24
theorem perimeter_eq_24 (a_eq_10 : a = 10) (area_eq_8sqrt3 : 8 * Real.sqrt 3 = (1 / 2) * b * c * Real.sin A) :
  a + b + c = 24 :=
sorry

end TriangleProof

end angle_A_eq_pi_over_3_perimeter_eq_24_l234_234618


namespace geom_seq_q_eq_l234_234617

theorem geom_seq_q_eq (a1 : ℕ := 2) (S3 : ℕ := 26) 
  (h1 : a1 = 2) 
  (h2 : S3 = 26) : 
  ∃ q : ℝ, (q = 3 ∨ q = -4) := by
  sorry

end geom_seq_q_eq_l234_234617


namespace maria_strawberries_l234_234313

theorem maria_strawberries (S : ℕ) :
  (21 = 8 + 9 + S) → (S = 4) :=
by
  intro h
  sorry

end maria_strawberries_l234_234313


namespace sum_of_y_for_f_l234_234628

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x + 5

theorem sum_of_y_for_f (y1 y2 y3 : ℝ) :
  (∀ y, 64 * y^3 - 8 * y + 5 = 7) →
  y1 + y2 + y3 = 0 :=
by
  -- placeholder for actual proof
  sorry

end sum_of_y_for_f_l234_234628


namespace reading_days_l234_234052

theorem reading_days (total_pages pages_per_day_1 pages_per_day_2 : ℕ ) :
  total_pages = 525 →
  pages_per_day_1 = 25 →
  pages_per_day_2 = 21 →
  (total_pages / pages_per_day_1 = 21) ∧ (total_pages / pages_per_day_2 = 25) :=
by
  sorry

end reading_days_l234_234052


namespace johns_bakery_fraction_l234_234580

theorem johns_bakery_fraction :
  ∀ (M : ℝ), 
  (M / 4 + M / 3 + 6 + (24 - (M / 4 + M / 3 + 6)) = 24) →
  (24 : ℝ) = M →
  (4 + 8 + 6 = 18) →
  (24 - 18 = 6) →
  (6 / 24 = (1 / 6 : ℝ)) :=
by
  intros M h1 h2 h3 h4
  sorry

end johns_bakery_fraction_l234_234580


namespace ryan_spends_7_hours_on_english_l234_234241

variable (C : ℕ)
variable (E : ℕ)

def hours_spent_on_english (C : ℕ) : ℕ := C + 2

theorem ryan_spends_7_hours_on_english :
  C = 5 → E = hours_spent_on_english C → E = 7 :=
by
  intro hC hE
  rw [hC] at hE
  exact hE

end ryan_spends_7_hours_on_english_l234_234241


namespace log_sum_identity_l234_234658

-- Prove that: lg 8 + 3 * lg 5 = 3

noncomputable def common_logarithm (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_sum_identity : 
    common_logarithm 8 + 3 * common_logarithm 5 = 3 := 
by
  sorry

end log_sum_identity_l234_234658


namespace crackers_eaten_l234_234483

-- Define the number of packs and their respective number of crackers
def num_packs_8 : ℕ := 5
def num_packs_10 : ℕ := 10
def num_packs_12 : ℕ := 7
def num_packs_15 : ℕ := 3

def crackers_per_pack_8 : ℕ := 8
def crackers_per_pack_10 : ℕ := 10
def crackers_per_pack_12 : ℕ := 12
def crackers_per_pack_15 : ℕ := 15

-- Calculate the total number of animal crackers
def total_crackers : ℕ :=
  (num_packs_8 * crackers_per_pack_8) +
  (num_packs_10 * crackers_per_pack_10) +
  (num_packs_12 * crackers_per_pack_12) +
  (num_packs_15 * crackers_per_pack_15)

-- Define the number of students who didn't eat their crackers and the respective number of crackers per pack
def num_students_not_eaten : ℕ := 4
def different_crackers_not_eaten : List ℕ := [8, 10, 12, 15]

-- Calculate the total number of crackers not eaten by adding those packs.
def total_crackers_not_eaten : ℕ := different_crackers_not_eaten.sum

-- Theorem to prove the total number of crackers eaten.
theorem crackers_eaten : total_crackers - total_crackers_not_eaten = 224 :=
by
  -- Total crackers: 269
  -- Subtract crackers not eaten: 8 + 10 + 12 + 15 = 45
  -- Therefore: 269 - 45 = 224
  sorry

end crackers_eaten_l234_234483


namespace trader_loss_percentage_l234_234203

def profit_loss_percentage (SP1 SP2 CP1 CP2 : ℚ) : ℚ :=
  ((SP1 + SP2) - (CP1 + CP2)) / (CP1 + CP2) * 100

theorem trader_loss_percentage :
  let SP1 := 325475
  let SP2 := 325475
  let CP1 := SP1 / (1 + 0.10)
  let CP2 := SP2 / (1 - 0.10)
  profit_loss_percentage SP1 SP2 CP1 CP2 = -1 := by
  sorry

end trader_loss_percentage_l234_234203


namespace simplify_fraction_l234_234811

theorem simplify_fraction :
  (2023^3 - 3 * 2023^2 * 2024 + 4 * 2023 * 2024^2 - 2024^3 + 2) / (2023 * 2024) = 2023 := by
sorry

end simplify_fraction_l234_234811


namespace part1_inequality_part2_range_of_a_l234_234442

-- Part (1)
theorem part1_inequality (x : ℝ) : 
  let a := 1
  let f := λ x, |x - 1| + |x + 3|
  in (f x ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) : 
  let f := λ x, |x - a| + |x + 3|
  in (∀ x, f x > -a ↔ a > -3/2) :=
sorry

end part1_inequality_part2_range_of_a_l234_234442


namespace part1_solution_set_part2_range_of_a_l234_234445

/-- Proof that the solution set of the inequality f(x) ≥ 6 when a = 1 is (-∞, -4] ∪ [2, ∞) -/
theorem part1_solution_set (x : ℝ) (h1 : f1 x ≥ 6) : 
  (x ≤ -4 ∨ x ≥ 2) :=
begin
  sorry
end

/-- Proof that the range of values for a if f(x) > -a is (-3/2, ∞) -/
theorem part2_range_of_a (a : ℝ) (h2 : ∀ x, f a x > -a) : 
  (-3/2 < a ∨ 0 ≤ a) :=
begin
  sorry
end

/-- Defining function f(x) = |x - a| + |x + 3| with specific 'a' value for part1 -/
def f1 (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

/-- Defining function f(x) = |x - a| + |x + 3| for part2 -/
def f (a : ℝ) (x : ℝ) : ℝ := abs (x - a) + abs (x + 3)

end part1_solution_set_part2_range_of_a_l234_234445


namespace theo_total_eggs_needed_l234_234220

theorem theo_total_eggs_needed:
  (num_3egg_hour1: ℕ) (num_4egg_hour2: ℕ) (num_3egg_hour3: ℕ) (num_4egg_hour4: ℕ)
  (eggs_per_3egg: ℕ) (eggs_per_4egg: ℕ) 
  (num_3egg_customers: ℕ := num_3egg_hour1 + num_3egg_hour3)
  (num_4egg_customers: ℕ := num_4egg_hour2 + num_4egg_hour4)
  (total_eggs_needed: ℕ := eggs_per_3egg * num_3egg_customers + eggs_per_4egg * num_4egg_customers):
  (num_3egg_hour1 = 5) → (num_4egg_hour2 = 7) → (num_3egg_hour3 = 3) → (num_4egg_hour4 = 8) →
  (eggs_per_3egg = 3) → (eggs_per_4egg = 4) → 
  total_eggs_needed = 84 :=
by
  intros
  sorry

end theo_total_eggs_needed_l234_234220


namespace find_y_l234_234200

theorem find_y
  (x y : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y)
  (hr : x % y = 8)
  (hq : x / y = 96) 
  (hr_decimal : (x:ℚ) / (y:ℚ) = 96.16) :
  y = 50 := 
sorry

end find_y_l234_234200


namespace distinct_shading_patterns_l234_234911

/-- How many distinct patterns can be made by shading exactly three of the sixteen squares 
    in a 4x4 grid, considering that patterns which can be matched by flips and/or turns are 
    not considered different? The answer is 8. -/
theorem distinct_shading_patterns : 
  (number_of_distinct_patterns : ℕ) = 8 :=
by
  /- Define the 4x4 Grid and the condition of shading exactly three squares, considering 
     flips and turns -/
  sorry

end distinct_shading_patterns_l234_234911


namespace smallest_integer_to_make_multiple_of_five_l234_234355

/-- The smallest positive integer that can be added to 725 to make it a multiple of 5 is 5. -/
theorem smallest_integer_to_make_multiple_of_five : 
  ∃ k : ℕ, k > 0 ∧ (725 + k) % 5 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → k ≤ m :=
sorry

end smallest_integer_to_make_multiple_of_five_l234_234355


namespace exists_positive_integers_x_y_l234_234294

theorem exists_positive_integers_x_y (x y : ℕ) : 0 < x ∧ 0 < y ∧ x^2 = y^2 + 2023 :=
  sorry

end exists_positive_integers_x_y_l234_234294


namespace vertices_of_equilateral_triangle_l234_234343

noncomputable def a : ℝ := 52 / 3
noncomputable def b : ℝ := -13 / 3 - 15 * Real.sqrt 3 / 2

theorem vertices_of_equilateral_triangle (a b : ℝ)
  (h₀ : (0, 0) = (0, 0))
  (h₁ : (a, 15) = (52 / 3, 15))
  (h₂ : (b, 41) = (-13 / 3 - 15 * Real.sqrt 3 / 2, 41)) :
  a * b = -676 / 9 := 
by
  sorry

end vertices_of_equilateral_triangle_l234_234343


namespace find_num_of_boys_l234_234350

-- Define the constants for number of girls and total number of kids
def num_of_girls : ℕ := 3
def total_kids : ℕ := 9

-- The theorem stating the number of boys based on the given conditions
theorem find_num_of_boys (g t : ℕ) (h1 : g = num_of_girls) (h2 : t = total_kids) :
  t - g = 6 :=
by
  sorry

end find_num_of_boys_l234_234350


namespace sum_of_x_and_y_l234_234854

theorem sum_of_x_and_y (x y : ℕ) (hxpos : 0 < x) (hypos : 1 < y) (hxy : x^y < 500) (hmax : ∀ (a b : ℕ), 0 < a → 1 < b → a^b < 500 → a^b ≤ x^y) : x + y = 24 := 
sorry

end sum_of_x_and_y_l234_234854


namespace solve_quadratic1_solve_quadratic2_l234_234782

theorem solve_quadratic1 (x : ℝ) :
  x^2 - 4 * x - 7 = 0 →
  (x = 2 - Real.sqrt 11) ∨ (x = 2 + Real.sqrt 11) :=
by
  sorry

theorem solve_quadratic2 (x : ℝ) :
  (x - 3)^2 + 2 * (x - 3) = 0 →
  (x = 3) ∨ (x = 1) :=
by
  sorry

end solve_quadratic1_solve_quadratic2_l234_234782


namespace probability_at_least_one_l234_234403

theorem probability_at_least_one (
    pA pB pC : ℝ
) (hA : pA = 0.9) (hB : pB = 0.8) (hC : pC = 0.7) (independent : true) : 
    (1 - (1 - pA) * (1 - pB) * (1 - pC)) = 0.994 := 
by
  rw [hA, hB, hC]
  sorry

end probability_at_least_one_l234_234403


namespace percentage_of_total_is_sixty_l234_234758

def num_boys := 600
def diff_boys_girls := 400
def num_girls := num_boys + diff_boys_girls
def total_people := num_boys + num_girls
def target_number := 960
def target_percentage := (target_number / total_people) * 100

theorem percentage_of_total_is_sixty :
  target_percentage = 60 := by
  sorry

end percentage_of_total_is_sixty_l234_234758


namespace triangle_area_l234_234289

structure Point where
  x : ℝ
  y : ℝ

def area_triangle (A B C : Point) : ℝ := 
  0.5 * (B.x - A.x) * (C.y - A.y)

theorem triangle_area :
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨8, 15⟩
  let C : Point := ⟨8, 0⟩
  area_triangle A B C = 60 :=
by
  sorry

end triangle_area_l234_234289


namespace part1_solution_set_part2_range_a_l234_234434

-- Define the function f
noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

-- Part 1: Proving the solution set of the inequality
theorem part1_solution_set (x : ℝ) :
  (∀ x, f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2: Finding the range of values for a
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a > -a ↔ a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_a_l234_234434


namespace favorite_food_sandwiches_l234_234016

theorem favorite_food_sandwiches (total_students : ℕ) (cookies_percent pizza_percent pasta_percent : ℝ)
  (h_total : total_students = 200)
  (h_cookies : cookies_percent = 0.25)
  (h_pizza : pizza_percent = 0.30)
  (h_pasta : pasta_percent = 0.35) :
  let sandwiches_percent := 1 - (cookies_percent + pizza_percent + pasta_percent)
  sandwiches_percent * total_students = 20 :=
by
  sorry

end favorite_food_sandwiches_l234_234016


namespace total_seashells_found_intact_seashells_found_l234_234517

-- Define the constants for seashells found
def tom_seashells : ℕ := 15
def fred_seashells : ℕ := 43

-- Define total_intercept
def total_intercept : ℕ := 29

-- Statement that the total seashells found by Tom and Fred is 58
theorem total_seashells_found : tom_seashells + fred_seashells = 58 := by
  sorry

-- Statement that the intact seashells are obtained by subtracting cracked ones
theorem intact_seashells_found : tom_seashells + fred_seashells - total_intercept = 29 := by
  sorry

end total_seashells_found_intact_seashells_found_l234_234517


namespace bill_spots_39_l234_234222

theorem bill_spots_39 (P : ℕ) (h1 : P + (2 * P - 1) = 59) : 2 * P - 1 = 39 :=
by sorry

end bill_spots_39_l234_234222


namespace value_of_5_S_3_l234_234238

def S (a b : ℕ) : ℕ := 4 * a + 6 * b + 1

theorem value_of_5_S_3 : S 5 3 = 39 := by
  sorry

end value_of_5_S_3_l234_234238


namespace max_g_value_l234_234184

def g (n : ℕ) : ℕ :=
if h : n < 10 then 2 * n + 3 else g (n - 7)

theorem max_g_value : ∃ n, g n = 21 ∧ ∀ m, g m ≤ 21 :=
sorry

end max_g_value_l234_234184


namespace tailwind_speed_rate_of_change_of_ground_speed_l234_234062

-- Define constants and variables
variables (Vp Vw : ℝ) (altitude Vg1 Vg2 : ℝ)

-- Define conditions
def conditions := Vg1 = Vp + Vw ∧ altitude = 10000 ∧ Vg1 = 460 ∧
                  Vg2 = Vp - Vw ∧ altitude = 5000 ∧ Vg2 = 310

-- Define theorems to prove
theorem tailwind_speed (Vp Vw : ℝ) (altitude Vg1 Vg2 : ℝ) :
  conditions Vp Vw altitude Vg1 Vg2 → Vw = 75 :=
by
  sorry

theorem rate_of_change_of_ground_speed (altitude1 altitude2 Vg1 Vg2 : ℝ) :
  altitude1 = 10000 → altitude2 = 5000 → Vg1 = 460 → Vg2 = 310 →
  (Vg2 - Vg1) / (altitude2 - altitude1) = 0.03 :=
by
  sorry

end tailwind_speed_rate_of_change_of_ground_speed_l234_234062


namespace instantaneous_velocity_at_t2_l234_234956

def displacement (t : ℝ) : ℝ := 14 * t - t ^ 2

theorem instantaneous_velocity_at_t2 : (deriv displacement 2) = 10 := by
  sorry

end instantaneous_velocity_at_t2_l234_234956


namespace three_segments_form_triangle_l234_234961

theorem three_segments_form_triangle
    (lengths : Fin 10 → ℕ)
    (h1 : lengths 0 = 1)
    (h2 : lengths 1 = 1)
    (h3 : lengths 9 = 50) :
    ∃ i j k : Fin 10, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    lengths i + lengths j > lengths k ∧ 
    lengths i + lengths k > lengths j ∧ 
    lengths j + lengths k > lengths i := 
sorry

end three_segments_form_triangle_l234_234961


namespace min_height_bounces_l234_234373

noncomputable def geometric_sequence (a r: ℝ) (n: ℕ) : ℝ := 
  a * r^n

theorem min_height_bounces (k : ℕ) : 
  ∀ k, 20 * (2 / 3 : ℝ) ^ k < 3 → k ≥ 7 := 
by
  sorry

end min_height_bounces_l234_234373


namespace skipping_rates_l234_234194

theorem skipping_rates (x y : ℕ) (h₀ : 300 / (x + 19) = 270 / x) (h₁ : y = x + 19) :
  x = 171 ∧ y = 190 := by
  sorry

end skipping_rates_l234_234194


namespace positive_difference_l234_234349

theorem positive_difference
  (x y : ℝ)
  (h1 : x + y = 10)
  (h2 : x^2 - y^2 = 40) : abs (x - y) = 4 :=
sorry

end positive_difference_l234_234349


namespace quadratic_algebraic_expression_l234_234269

theorem quadratic_algebraic_expression (a b : ℝ) (h₁ : a^2 - 3 * a + 1 = 0) (h₂ : b^2 - 3 * b + 1 = 0) :
    a + b - a * b = 2 := by
  sorry

end quadratic_algebraic_expression_l234_234269


namespace calc_3a2b_times_neg_a_squared_l234_234229

variables {a b : ℝ}

theorem calc_3a2b_times_neg_a_squared : 3 * a^2 * b * (-a)^2 = 3 * a^4 * b :=
by
  sorry

end calc_3a2b_times_neg_a_squared_l234_234229


namespace Ali_money_left_l234_234850

theorem Ali_money_left (initial_money : ℕ) 
  (spent_on_food_ratio : ℚ) 
  (spent_on_glasses_ratio : ℚ) 
  (spent_on_food : ℕ) 
  (left_after_food : ℕ) 
  (spent_on_glasses : ℕ) 
  (final_left : ℕ) :
    initial_money = 480 →
    spent_on_food_ratio = 1 / 2 →
    spent_on_food = initial_money * spent_on_food_ratio →
    left_after_food = initial_money - spent_on_food →
    spent_on_glasses_ratio = 1 / 3 →
    spent_on_glasses = left_after_food * spent_on_glasses_ratio →
    final_left = left_after_food - spent_on_glasses →
    final_left = 160 :=
by
  sorry

end Ali_money_left_l234_234850


namespace circle_area_greater_than_hexagon_area_l234_234320

theorem circle_area_greater_than_hexagon_area (h : ℝ) (r : ℝ) (π : ℝ) (sqrt3 : ℝ) (ratio : ℝ) : 
  (h = 1) →
  (r = sqrt3 / 2) →
  (π > 3) →
  (sqrt3 > 1.7) →
  (ratio = (π * sqrt3) / 6) →
  ratio > 0.9 :=
by
  intros h_eq r_eq pi_gt sqrt3_gt ratio_eq
  -- Proof omitted
  sorry

end circle_area_greater_than_hexagon_area_l234_234320


namespace rationalize_denominator_l234_234777

theorem rationalize_denominator :
  (2 / (Real.cbrt 3 + Real.cbrt 27)) = (Real.cbrt 9 / 6) :=
by
  have h1 : Real.cbrt 27 = 3 * Real.cbrt 3 := sorry
  sorry

end rationalize_denominator_l234_234777


namespace parallelogram_area_288_l234_234170

/-- A statement of the area of a given parallelogram -/
theorem parallelogram_area_288 
  (AB BC : ℝ)
  (hAB : AB = 24)
  (hBC : BC = 30)
  (height_from_A_to_DC : ℝ)
  (h_height : height_from_A_to_DC = 12)
  (is_parallelogram : true) :
  AB * height_from_A_to_DC = 288 :=
by
  -- We are focusing only on stating the theorem; the proof is not required.
  sorry

end parallelogram_area_288_l234_234170


namespace math_problem_l234_234256

noncomputable def proof_problem (a b c : ℝ) (h₀ : a < 0) (h₁ : b < 0) (h₂ : c < 0) : Prop :=
  let n1 := a + 1/b
  let n2 := b + 1/c
  let n3 := c + 1/a
  (n1 ≤ -2) ∨ (n2 ≤ -2) ∨ (n3 ≤ -2)

theorem math_problem (a b c : ℝ) (h₀ : a < 0) (h₁ : b < 0) (h₂ : c < 0) : proof_problem a b c h₀ h₁ h₂ :=
sorry

end math_problem_l234_234256


namespace rope_length_third_post_l234_234856

theorem rope_length_third_post (total first second fourth : ℕ) (h_total : total = 70) 
    (h_first : first = 24) (h_second : second = 20) (h_fourth : fourth = 12) : 
    (total - first - second - fourth) = 14 :=
by
  -- Proof is skipped, but we can state that the theorem should follow from the given conditions.
  sorry

end rope_length_third_post_l234_234856


namespace binomials_product_evaluation_l234_234588

-- Define the binomials and the resulting polynomial
def binomial_one (x : ℝ) := 4 * x + 3
def binomial_two (x : ℝ) := 2 * x - 6
def resulting_polynomial (x : ℝ) := 8 * x^2 - 18 * x - 18

-- Define the proof problem
theorem binomials_product_evaluation :
  ∀ (x : ℝ), (binomial_one x) * (binomial_two x) = resulting_polynomial x ∧ 
  resulting_polynomial (-1) = 8 := 
by 
  intro x
  have h1 : (4 * x + 3) * (2 * x - 6) = 8 * x^2 - 18 * x - 18 := sorry
  have h2 : resulting_polynomial (-1) = 8 := sorry
  exact ⟨h1, h2⟩

end binomials_product_evaluation_l234_234588


namespace speed_of_stream_l234_234537

theorem speed_of_stream
  (D : ℝ) (v : ℝ)
  (h : D / (72 - v) = 2 * D / (72 + v)) :
  v = 24 := by
  sorry

end speed_of_stream_l234_234537


namespace range_of_x_l234_234511

def valid_domain (x : ℝ) : Prop :=
  (3 - x ≥ 0) ∧ (x ≠ 4)

theorem range_of_x : ∀ x : ℝ, valid_domain x ↔ (x ≤ 3) :=
by sorry

end range_of_x_l234_234511


namespace division_problem_l234_234075

theorem division_problem :
  250 / (5 + 12 * 3^2) = 250 / 113 :=
by sorry

end division_problem_l234_234075


namespace right_triangle_hypotenuse_l234_234279

def is_nat (n : ℕ) : Prop := n > 0

theorem right_triangle_hypotenuse (x : ℕ) (x_pos : is_nat x) (consec : x + 1 > x) (h : 11^2 + x^2 = (x + 1)^2) : x + 1 = 61 :=
by
  sorry

end right_triangle_hypotenuse_l234_234279


namespace rationalize_denominator_l234_234775

theorem rationalize_denominator (a b c : Real) (h : b*c*c = a) :
  2 / (b + c) = (c*c) / (3 * 2) :=
by
  sorry

end rationalize_denominator_l234_234775


namespace max_value_expression_l234_234341

theorem max_value_expression : ∀ (a b c d : ℝ), 
  a ∈ Set.Icc (-4.5) 4.5 → 
  b ∈ Set.Icc (-4.5) 4.5 → 
  c ∈ Set.Icc (-4.5) 4.5 → 
  d ∈ Set.Icc (-4.5) 4.5 → 
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 90 :=
by sorry

end max_value_expression_l234_234341


namespace initial_quarters_l234_234473

variable (q : ℕ)

theorem initial_quarters (h : q + 3 = 11) : q = 8 :=
by
  sorry

end initial_quarters_l234_234473


namespace factorize_expr1_factorize_expr2_factorize_expr3_factorize_expr4_l234_234242

variable {R : Type} [CommRing R]
variables {m n x a b : R}

-- 1. Factorization of 3mn - 6m^2n^2
theorem factorize_expr1 : 3 * m * n - 6 * m^2 * n^2 = 3 * m * n * (1 - 2 * m * n) :=
by
  sorry

-- 2. Factorization of m^2 - 4mn + 4n^2
theorem factorize_expr2 : m^2 - 4 * m * n + 4 * n^2 = (m - 2 * n)^2 :=
by
  sorry

-- 3. Factorization of x^3 - 9x
theorem factorize_expr3 : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
by
  sorry

-- 4. Factorization of 6ab^2 - 9a^2b - b^3
theorem factorize_expr4 : 6 * a * b^2 - 9 * a^2 * b - b^3 = -b * (b - 3 * a)^2 :=
by
  sorry

end factorize_expr1_factorize_expr2_factorize_expr3_factorize_expr4_l234_234242


namespace triangle_equilateral_l234_234920

noncomputable def is_equilateral (a b c : ℝ) : Prop :=
a = b ∧ b = c

theorem triangle_equilateral (A B C a b c : ℝ) (hB : B = 60) (hb : b^2 = a * c) (hcos : b^2 = a^2 + c^2 - a * c):
  is_equilateral a b c :=
by
  sorry

end triangle_equilateral_l234_234920


namespace sequence_bounds_l234_234734

theorem sequence_bounds (c : ℝ) (a : ℕ+ → ℝ) (h : ∀ n : ℕ+, a n = ↑n + c / ↑n) 
  (h2 : ∀ n : ℕ+, a n ≥ a 3) : 6 ≤ c ∧ c ≤ 12 :=
by 
  -- We will prove that 6 ≤ c and c ≤ 12 given the conditions stated
  sorry

end sequence_bounds_l234_234734


namespace evaluate_expression_l234_234864

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem evaluate_expression : (nabla (nabla 2 3) 4) = 16777219 :=
by sorry

end evaluate_expression_l234_234864


namespace shorter_piece_length_l234_234374

theorem shorter_piece_length (x : ℝ) :
  (120 - (2 * x + 15) = x) → x = 35 := 
by
  intro h
  sorry

end shorter_piece_length_l234_234374


namespace Frank_time_correct_l234_234232

def Dave_time := 10
def Chuck_time := 5 * Dave_time
def Erica_time := 13 * Chuck_time / 10
def Frank_time := 12 * Erica_time / 10

theorem Frank_time_correct : Frank_time = 78 :=
by
  sorry

end Frank_time_correct_l234_234232


namespace foci_equality_ellipse_hyperbola_l234_234613

theorem foci_equality_ellipse_hyperbola (m : ℝ) (h : m > 0) 
  (hl: ∀ x y : ℝ, x^2 / 4 + y^2 / m^2 = 1 → 
     ∃ c : ℝ, c = Real.sqrt (4 - m^2)) 
  (hh: ∀ x y : ℝ, x^2 / m^2 - y^2 / 2 = 1 → 
     ∃ c : ℝ, c = Real.sqrt (m^2 + 2)) : 
  m = 1 :=
by {
  sorry
}

end foci_equality_ellipse_hyperbola_l234_234613


namespace cheenu_time_difference_l234_234530

theorem cheenu_time_difference :
  let boy_distance : ℝ := 18
  let boy_time_hours : ℝ := 4
  let old_man_distance : ℝ := 12
  let old_man_time_hours : ℝ := 5
  let hour_to_minute : ℝ := 60
  
  let boy_time_minutes := boy_time_hours * hour_to_minute
  let old_man_time_minutes := old_man_time_hours * hour_to_minute

  let boy_time_per_mile := boy_time_minutes / boy_distance
  let old_man_time_per_mile := old_man_time_minutes / old_man_distance
  
  old_man_time_per_mile - boy_time_per_mile = 12 :=
by sorry

end cheenu_time_difference_l234_234530


namespace find_period_for_interest_l234_234724

noncomputable def period_for_compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) : ℝ :=
  (Real.log A - Real.log P) / (n * Real.log (1 + r / n))

theorem find_period_for_interest :
  period_for_compound_interest 8000 0.15 1 11109 = 2 := 
sorry

end find_period_for_interest_l234_234724


namespace Mrs_Lara_Late_l234_234635

noncomputable def required_speed (d t : ℝ) : ℝ := d / t

theorem Mrs_Lara_Late (d t : ℝ) (h1 : d = 50 * (t + 7 / 60)) (h2 : d = 70 * (t - 5 / 60)) :
  required_speed d t = 70 := by
  sorry

end Mrs_Lara_Late_l234_234635


namespace cost_of_each_scoop_l234_234318

theorem cost_of_each_scoop (x : ℝ) 
  (pierre_scoops : ℝ := 3)
  (mom_scoops : ℝ := 4)
  (total_bill : ℝ := 14) 
  (h : 7 * x = total_bill) :
  x = 2 :=
by 
  sorry

end cost_of_each_scoop_l234_234318


namespace part1_inequality_part2_range_of_a_l234_234440

-- Part (1)
theorem part1_inequality (x : ℝ) : 
  let a := 1
  let f := λ x, |x - 1| + |x + 3|
  in (f x ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) : 
  let f := λ x, |x - a| + |x + 3|
  in (∀ x, f x > -a ↔ a > -3/2) :=
sorry

end part1_inequality_part2_range_of_a_l234_234440


namespace regular_polygon_sides_l234_234830

theorem regular_polygon_sides (P s : ℕ) (hP : P = 180) (hs : s = 15) : P / s = 12 := by
  -- Given
  -- P = 180  -- the perimeter in cm
  -- s = 15   -- the side length in cm
  sorry

end regular_polygon_sides_l234_234830


namespace barbara_shopping_l234_234386

theorem barbara_shopping :
  let total_paid := 56
  let tuna_cost := 5 * 2
  let water_cost := 4 * 1.5
  let other_goods_cost := total_paid - tuna_cost - water_cost
  other_goods_cost = 40 :=
by
  sorry

end barbara_shopping_l234_234386


namespace initial_group_size_l234_234789

theorem initial_group_size (n : ℕ) (W : ℝ) 
  (h1 : (W + 20) / n = W / n + 4) : 
  n = 5 := 
by 
  sorry

end initial_group_size_l234_234789


namespace quadratic_eq_has_real_root_l234_234802

theorem quadratic_eq_has_real_root (a b : ℝ) :
  ¬(∀ x : ℝ, x^2 + a * x + b ≠ 0) → ∃ x : ℝ, x^2 + a * x + b = 0 :=
by
  sorry

end quadratic_eq_has_real_root_l234_234802


namespace bill_spots_39_l234_234223

theorem bill_spots_39 (P : ℕ) (h1 : P + (2 * P - 1) = 59) : 2 * P - 1 = 39 :=
by sorry

end bill_spots_39_l234_234223


namespace find_positive_x_l234_234094

theorem find_positive_x (x : ℝ) (hx : 0 < x) (h : ⌊x⌋ * x = 72) : x = 9 := sorry

end find_positive_x_l234_234094
