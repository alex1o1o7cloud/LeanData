import Mathlib

namespace breadth_of_rectangular_plot_l1381_138124

theorem breadth_of_rectangular_plot (b : ℝ) (h1 : 3 * b * b = 972) : b = 18 :=
sorry

end breadth_of_rectangular_plot_l1381_138124


namespace find_x_in_interval_l1381_138192

noncomputable def a : ℝ := Real.sqrt 2014 - Real.sqrt 2013

theorem find_x_in_interval :
  ∀ x : ℝ, (0 < x) → (x < Real.pi) →
  (a^(Real.tan x ^ 2) + (Real.sqrt 2014 + Real.sqrt 2013)^(-Real.tan x ^ 2) = 2 * a^3) →
  (x = Real.pi / 3 ∨ x = 2 * Real.pi / 3) := by
  -- add proof here
  sorry

end find_x_in_interval_l1381_138192


namespace division_of_decimals_l1381_138173

theorem division_of_decimals : (0.5 : ℝ) / (0.025 : ℝ) = 20 := 
sorry

end division_of_decimals_l1381_138173


namespace discount_percentage_is_20_l1381_138105

theorem discount_percentage_is_20
  (regular_price_per_shirt : ℝ) (number_of_shirts : ℝ) (total_sale_price : ℝ)
  (h₁ : regular_price_per_shirt = 50) (h₂ : number_of_shirts = 6) (h₃ : total_sale_price = 240) :
  ( ( (regular_price_per_shirt * number_of_shirts - total_sale_price) / (regular_price_per_shirt * number_of_shirts) ) * 100 ) = 20 :=
by
  sorry

end discount_percentage_is_20_l1381_138105


namespace joe_cars_after_getting_more_l1381_138163

-- Defining the initial conditions as Lean variables
def initial_cars : ℕ := 50
def additional_cars : ℕ := 12

-- Stating the proof problem
theorem joe_cars_after_getting_more : initial_cars + additional_cars = 62 := by
  sorry

end joe_cars_after_getting_more_l1381_138163


namespace total_cards_traded_l1381_138131

-- Define the total number of cards traded in both trades
def total_traded (p1_t: ℕ) (r1_t: ℕ) (p2_t: ℕ) (r2_t: ℕ): ℕ :=
  (p1_t + r1_t) + (p2_t + r2_t)

-- Given conditions as definitions
def padma_trade1 := 2   -- Cards Padma traded in the first trade
def robert_trade1 := 10  -- Cards Robert traded in the first trade
def padma_trade2 := 15  -- Cards Padma traded in the second trade
def robert_trade2 := 8   -- Cards Robert traded in the second trade

-- Theorem stating the total number of cards traded is 35
theorem total_cards_traded : 
  total_traded padma_trade1 robert_trade1 padma_trade2 robert_trade2 = 35 :=
by
  sorry

end total_cards_traded_l1381_138131


namespace domain_of_function_l1381_138190

theorem domain_of_function (x : ℝ) :
  (x^2 - 5*x + 6 ≥ 0) → (x ≠ 2) → (x < 2 ∨ x ≥ 3) :=
by
  intros h1 h2
  sorry

end domain_of_function_l1381_138190


namespace smallest_product_is_298150_l1381_138119

def digits : List ℕ := [5, 6, 7, 8, 9, 0]

theorem smallest_product_is_298150 :
  ∃ (a b c : ℕ), 
    a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (a * b * c = 298150) :=
sorry

end smallest_product_is_298150_l1381_138119


namespace total_wet_surface_area_eq_l1381_138176

-- Definitions based on given conditions
def length_cistern : ℝ := 10
def width_cistern : ℝ := 6
def height_water : ℝ := 1.35

-- Problem statement: Prove the total wet surface area is as calculated
theorem total_wet_surface_area_eq :
  let area_bottom : ℝ := length_cistern * width_cistern
  let area_longer_sides : ℝ := 2 * (length_cistern * height_water)
  let area_shorter_sides : ℝ := 2 * (width_cistern * height_water)
  let total_wet_surface_area : ℝ := area_bottom + area_longer_sides + area_shorter_sides
  total_wet_surface_area = 103.2 :=
by
  -- Since we do not need the proof, we use sorry here
  sorry

end total_wet_surface_area_eq_l1381_138176


namespace min_x_plus_y_of_positive_l1381_138100

open Real

theorem min_x_plus_y_of_positive (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 1 / x + 4 / y = 1) : x + y ≥ 9 :=
sorry

end min_x_plus_y_of_positive_l1381_138100


namespace effective_weight_lowered_l1381_138101

theorem effective_weight_lowered 
    (num_weight_plates : ℕ) 
    (weight_per_plate : ℝ) 
    (increase_percentage : ℝ) 
    (total_weight_without_technology : ℝ) 
    (additional_weight : ℝ) 
    (effective_weight_lowering : ℝ) 
    (h1 : num_weight_plates = 10)
    (h2 : weight_per_plate = 30)
    (h3 : increase_percentage = 0.20)
    (h4 : total_weight_without_technology = num_weight_plates * weight_per_plate)
    (h5 : additional_weight = increase_percentage * total_weight_without_technology)
    (h6 : effective_weight_lowering = total_weight_without_technology + additional_weight) :
    effective_weight_lowering = 360 := 
by
  sorry

end effective_weight_lowered_l1381_138101


namespace second_group_children_is_16_l1381_138126

def cases_purchased : ℕ := 13
def bottles_per_case : ℕ := 24
def camp_days : ℕ := 3
def first_group_children : ℕ := 14
def third_group_children : ℕ := 12
def bottles_per_child_per_day : ℕ := 3
def additional_bottles_needed : ℕ := 255

def fourth_group_children (x : ℕ) : ℕ := (14 + x + 12) / 2
def total_initial_bottles : ℕ := cases_purchased * bottles_per_case
def total_children (x : ℕ) : ℕ := 14 + x + 12 + fourth_group_children x 

def total_consumption (x : ℕ) : ℕ := (total_children x) * bottles_per_child_per_day * camp_days
def total_bottles_needed : ℕ := total_initial_bottles + additional_bottles_needed

theorem second_group_children_is_16 :
  ∃ x : ℕ, total_consumption x = total_bottles_needed ∧ x = 16 :=
by
  sorry

end second_group_children_is_16_l1381_138126


namespace eggs_in_larger_omelette_l1381_138130

theorem eggs_in_larger_omelette :
  ∀ (total_eggs : ℕ) (orders_3_eggs_first_hour orders_3_eggs_third_hour orders_large_eggs_second_hour orders_large_eggs_last_hour num_eggs_per_3_omelette : ℕ),
    total_eggs = 84 →
    orders_3_eggs_first_hour = 5 →
    orders_3_eggs_third_hour = 3 →
    orders_large_eggs_second_hour = 7 →
    orders_large_eggs_last_hour = 8 →
    num_eggs_per_3_omelette = 3 →
    (total_eggs - (orders_3_eggs_first_hour * num_eggs_per_3_omelette + orders_3_eggs_third_hour * num_eggs_per_3_omelette)) / (orders_large_eggs_second_hour + orders_large_eggs_last_hour) = 4 :=
by
  intros total_eggs orders_3_eggs_first_hour orders_3_eggs_third_hour orders_large_eggs_second_hour orders_large_eggs_last_hour num_eggs_per_3_omelette
  sorry

end eggs_in_larger_omelette_l1381_138130


namespace isosceles_right_triangle_square_ratio_l1381_138174

noncomputable def x : ℝ := 1 / 2
noncomputable def y : ℝ := Real.sqrt 2 / 2

theorem isosceles_right_triangle_square_ratio :
  x / y = Real.sqrt 2 := by
  sorry

end isosceles_right_triangle_square_ratio_l1381_138174


namespace interest_rate_l1381_138114

theorem interest_rate (part1_amount part2_amount total_amount total_income : ℝ) (interest_rate1 interest_rate2 : ℝ) :
  part1_amount = 2000 →
  part2_amount = total_amount - part1_amount →
  interest_rate2 = 6 →
  total_income = (part1_amount * interest_rate1 / 100) + (part2_amount * interest_rate2 / 100) →
  total_amount = 2500 →
  total_income = 130 →
  interest_rate1 = 5 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end interest_rate_l1381_138114


namespace number_of_intersection_points_l1381_138122

-- Definitions of the given lines
def line1 (x y : ℝ) : Prop := 6 * y - 4 * x = 2
def line2 (x y : ℝ) : Prop := x + 2 * y = 2
def line3 (x y : ℝ) : Prop := -4 * x + 6 * y = 3

-- Definitions of the intersection points
def intersection1 (x y : ℝ) : Prop := line1 x y ∧ line2 x y
def intersection2 (x y : ℝ) : Prop := line2 x y ∧ line3 x y

-- Definition of the problem
theorem number_of_intersection_points : 
  (∃ x y : ℝ, intersection1 x y) ∧
  (∃ x y : ℝ, intersection2 x y) ∧
  (¬ ∃ x y : ℝ, line1 x y ∧ line3 x y) →
  (∃ z : ℕ, z = 2) :=
sorry

end number_of_intersection_points_l1381_138122


namespace second_offset_length_l1381_138121

theorem second_offset_length (d h1 area : ℝ) (h_diagonal : d = 28) (h_offset1 : h1 = 8) (h_area : area = 140) :
  ∃ x : ℝ, area = (1/2) * d * (h1 + x) ∧ x = 2 :=
by
  sorry

end second_offset_length_l1381_138121


namespace abcd_product_l1381_138136

noncomputable def A := (Real.sqrt 3000 + Real.sqrt 3001)
noncomputable def B := (-Real.sqrt 3000 - Real.sqrt 3001)
noncomputable def C := (Real.sqrt 3000 - Real.sqrt 3001)
noncomputable def D := (Real.sqrt 3001 - Real.sqrt 3000)

theorem abcd_product :
  A * B * C * D = -1 :=
by
  sorry

end abcd_product_l1381_138136


namespace mitch_earns_correctly_l1381_138184

noncomputable def mitch_weekly_earnings : ℝ :=
  let earnings_mw := 3 * (3 * 5 : ℝ) -- Monday to Wednesday
  let earnings_tf := 2 * (6 * 4 : ℝ) -- Thursday and Friday
  let earnings_sat := 4 * 6         -- Saturday
  let earnings_sun := 5 * 8         -- Sunday
  let total_earnings := earnings_mw + earnings_tf + earnings_sat + earnings_sun
  let after_expenses := total_earnings - 25
  let after_tax := after_expenses - 0.10 * after_expenses
  after_tax

theorem mitch_earns_correctly : mitch_weekly_earnings = 118.80 := by
  sorry

end mitch_earns_correctly_l1381_138184


namespace cut_problem_l1381_138175

theorem cut_problem (n : ℕ) : (1 / 2 : ℝ) ^ n = 1 / 64 ↔ n = 6 :=
by
  sorry

end cut_problem_l1381_138175


namespace product_of_midpoint_l1381_138171

-- Define the coordinates of the endpoints
def x1 := 5
def y1 := -4
def x2 := 1
def y2 := 14

-- Define the formulas for the midpoint coordinates
def xm := (x1 + x2) / 2
def ym := (y1 + y2) / 2

-- Define the product of the midpoint coordinates
def product := xm * ym

-- Now state the theorem
theorem product_of_midpoint :
  product = 15 := 
by
  -- Optional: detailed steps can go here if necessary
  sorry

end product_of_midpoint_l1381_138171


namespace truck_sand_amount_l1381_138180

theorem truck_sand_amount (initial_sand loss_sand final_sand : ℝ) (h1 : initial_sand = 4.1) (h2 : loss_sand = 2.4) :
  initial_sand - loss_sand = final_sand ↔ final_sand = 1.7 := 
by
  sorry

end truck_sand_amount_l1381_138180


namespace find_fraction_identity_l1381_138179

variable (x y z : ℝ)

theorem find_fraction_identity
 (h1 : 16 * y^2 = 15 * x * z)
 (h2 : y = 2 * x * z / (x + z)) :
 x / z + z / x = 34 / 15 := by
-- proof skipped
sorry

end find_fraction_identity_l1381_138179


namespace gravel_cost_correct_l1381_138185

-- Definitions from the conditions
def lawn_length : ℕ := 80
def lawn_breadth : ℕ := 60
def road_width : ℕ := 15
def gravel_cost_per_sq_m : ℕ := 3

-- Calculate areas of the roads
def area_road_length : ℕ := lawn_length * road_width
def area_road_breadth : ℕ := (lawn_breadth - road_width) * road_width

-- Total area to be graveled
def total_area : ℕ := area_road_length + area_road_breadth

-- Total cost
def total_cost : ℕ := total_area * gravel_cost_per_sq_m

-- Prove the total cost is 5625 Rs
theorem gravel_cost_correct : total_cost = 5625 := by
  sorry

end gravel_cost_correct_l1381_138185


namespace valid_root_l1381_138160

theorem valid_root:
  ∃ x : ℚ, 
    (3 * x^2 + 5) / (x - 2) - (3 * x + 10) / 4 + (5 - 9 * x) / (x - 2) + 2 = 0 ∧ x = 2 / 3 := 
by
  sorry

end valid_root_l1381_138160


namespace least_multiple_of_17_gt_450_l1381_138152

def least_multiple_gt (n x : ℕ) (k : ℕ) : Prop :=
  k * n > x ∧ ∀ m : ℕ, m * n > x → m ≥ k

theorem least_multiple_of_17_gt_450 : ∃ k : ℕ, least_multiple_gt 17 450 k :=
by
  use 27
  sorry

end least_multiple_of_17_gt_450_l1381_138152


namespace math_problem_l1381_138193

variable {x a b : ℝ}

theorem math_problem (h1 : x < a) (h2 : a < 0) (h3 : b = -a) : x^2 > b^2 ∧ b^2 > 0 :=
by {
  sorry
}

end math_problem_l1381_138193


namespace n_mul_n_plus_one_even_l1381_138116

theorem n_mul_n_plus_one_even (n : ℤ) : Even (n * (n + 1)) := 
sorry

end n_mul_n_plus_one_even_l1381_138116


namespace simplify_fraction_l1381_138187

theorem simplify_fraction (k : ℤ) : 
  (∃ (a b : ℤ), a = 1 ∧ b = 2 ∧ (6 * k + 12) / 6 = a * k + b) → (1 / 2 : ℚ) = (1 / 2 : ℚ) := 
by
  intro h
  sorry

end simplify_fraction_l1381_138187


namespace regular_triangular_pyramid_volume_l1381_138199

theorem regular_triangular_pyramid_volume (a γ : ℝ) : 
  ∃ V, V = (a^3 * Real.sin (γ / 2)^2) / (12 * Real.sqrt (1 - (Real.sin (γ / 2))^2)) := 
sorry

end regular_triangular_pyramid_volume_l1381_138199


namespace required_speed_is_85_l1381_138150

-- Definitions based on conditions
def speed1 := 60
def time1 := 3
def total_time := 5
def average_speed := 70

-- Derived conditions
def distance1 := speed1 * time1
def total_distance := average_speed * total_time
def remaining_distance := total_distance - distance1
def remaining_time := total_time - time1
def required_speed := remaining_distance / remaining_time

-- Theorem statement
theorem required_speed_is_85 : required_speed = 85 := by
    sorry

end required_speed_is_85_l1381_138150


namespace number_of_kids_at_circus_l1381_138159

theorem number_of_kids_at_circus (K A : ℕ) 
(h1 : ∀ x, 5 * x = 1 / 2 * 10 * x)
(h2 : 5 * K + 10 * A = 50) : K = 2 :=
sorry

end number_of_kids_at_circus_l1381_138159


namespace height_of_C_l1381_138195

noncomputable def height_A_B_C (h_A h_B h_C : ℝ) : Prop := 
  (h_A + h_B + h_C) / 3 = 143 ∧ 
  h_A + 4.5 = (h_B + h_C) / 2 ∧ 
  h_B = h_C + 3

theorem height_of_C (h_A h_B h_C : ℝ) (h : height_A_B_C h_A h_B h_C) : h_C = 143 :=
  sorry

end height_of_C_l1381_138195


namespace cakes_served_dinner_l1381_138102

def total_cakes_today : Nat := 15
def cakes_served_lunch : Nat := 6

theorem cakes_served_dinner : total_cakes_today - cakes_served_lunch = 9 :=
by
  -- Define what we need to prove
  sorry -- to skip the proof

end cakes_served_dinner_l1381_138102


namespace total_earnings_correct_l1381_138161

-- Define the conditions as initial parameters

def ticket_price : ℕ := 3
def weekday_visitors_per_day : ℕ := 100
def saturday_visitors : ℕ := 200
def sunday_visitors : ℕ := 300

def total_weekday_visitors : ℕ := 5 * weekday_visitors_per_day
def total_weekend_visitors : ℕ := saturday_visitors + sunday_visitors
def total_visitors : ℕ := total_weekday_visitors + total_weekend_visitors

def total_earnings := total_visitors * ticket_price

-- Prove that the total earnings of the amusement park in a week is $3000
theorem total_earnings_correct : total_earnings = 3000 :=
by
  sorry

end total_earnings_correct_l1381_138161


namespace inequality_addition_l1381_138172

-- Definitions and Conditions
variables (a b c d : ℝ)
variable (h1 : a > b)
variable (h2 : c > d)

-- Theorem statement: Prove that a + c > b + d
theorem inequality_addition (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a + c > b + d := 
sorry

end inequality_addition_l1381_138172


namespace min_value_expression_l1381_138132

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c + 1) * (1 / (a + b + 1) + 1 / (b + c + 1) + 1 / (c + a + 1)) ≥ 9 / 2 :=
sorry

end min_value_expression_l1381_138132


namespace fraction_of_paint_used_l1381_138140

theorem fraction_of_paint_used 
  (total_paint : ℕ)
  (paint_used_first_week : ℚ)
  (total_paint_used : ℕ)
  (paint_fraction_first_week : ℚ)
  (remaining_paint : ℚ)
  (paint_used_second_week : ℚ)
  (paint_fraction_second_week : ℚ)
  (h1 : total_paint = 360)
  (h2 : paint_fraction_first_week = 2/3)
  (h3 : paint_used_first_week = paint_fraction_first_week * total_paint)
  (h4 : remaining_paint = total_paint - paint_used_first_week)
  (h5 : remaining_paint = 120)
  (h6 : total_paint_used = 264)
  (h7 : paint_used_second_week = total_paint_used - paint_used_first_week)
  (h8 : paint_fraction_second_week = paint_used_second_week / remaining_paint):
  paint_fraction_second_week = 1/5 := 
by 
  sorry

end fraction_of_paint_used_l1381_138140


namespace average_temps_l1381_138194

-- Define the temperature lists
def temps_C : List ℚ := [
  37.3, 37.2, 36.9, -- Sunday
  36.6, 36.9, 37.1, -- Monday
  37.1, 37.3, 37.2, -- Tuesday
  36.8, 37.3, 37.5, -- Wednesday
  37.1, 37.7, 37.3, -- Thursday
  37.5, 37.4, 36.9, -- Friday
  36.9, 37.0, 37.1  -- Saturday
]

def temps_K : List ℚ := [
  310.4, 310.3, 310.0, -- Sunday
  309.8, 310.0, 310.2, -- Monday
  310.2, 310.4, 310.3, -- Tuesday
  309.9, 310.4, 310.6, -- Wednesday
  310.2, 310.8, 310.4, -- Thursday
  310.6, 310.5, 310.0, -- Friday
  310.0, 310.1, 310.2  -- Saturday
]

def temps_R : List ℚ := [
  558.7, 558.6, 558.1, -- Sunday
  557.7, 558.1, 558.3, -- Monday
  558.3, 558.7, 558.6, -- Tuesday
  558.0, 558.7, 559.1, -- Wednesday
  558.3, 559.4, 558.7, -- Thursday
  559.1, 558.9, 558.1, -- Friday
  558.1, 558.2, 558.3  -- Saturday
]

-- Calculate the average of a list of temperatures
def average (temps : List ℚ) : ℚ :=
  temps.sum / temps.length

-- Define the average temperatures
def avg_C := average temps_C
def avg_K := average temps_K
def avg_R := average temps_R

-- State that the computed averages are equal to the provided values
theorem average_temps :
  avg_C = 37.1143 ∧
  avg_K = 310.1619 ∧
  avg_R = 558.2524 :=
by
  -- Proof can be completed here
  sorry

end average_temps_l1381_138194


namespace box_tape_length_l1381_138137

variable (L S : ℕ)
variable (tape_total : ℕ)
variable (num_boxes : ℕ)
variable (square_side : ℕ)

theorem box_tape_length (h1 : num_boxes = 5) (h2 : square_side = 40) (h3 : tape_total = 540) :
  tape_total = 5 * (L + 2 * S) + 2 * 3 * square_side → L = 60 - 2 * S := 
by
  sorry

end box_tape_length_l1381_138137


namespace problem_k_value_l1381_138129

theorem problem_k_value (k x1 x2 : ℝ) 
  (h_eq : 8 * x1^2 + 2 * k * x1 + k - 1 = 0) 
  (h_eq2 : 8 * x2^2 + 2 * k * x2 + k - 1 = 0) 
  (h_sum_sq : x1^2 + x2^2 = 1) : 
  k = -2 :=
sorry

end problem_k_value_l1381_138129


namespace sum_of_squares_l1381_138123

theorem sum_of_squares : 
  let nums := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
  let squares := nums.map (λ x => x * x)
  (squares.sum = 195) := 
by
  let nums := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
  let squares := nums.map (λ x => x * x)
  have h : squares.sum = 195 := sorry
  exact h

end sum_of_squares_l1381_138123


namespace pyramid_height_l1381_138142

theorem pyramid_height (lateral_edge : ℝ) (h : ℝ) (equilateral_angles : ℝ × ℝ × ℝ) (lateral_edge_length : lateral_edge = 3)
  (lateral_faces_are_equilateral : equilateral_angles = (60, 60, 60)) :
  h = 3 / 4 := by
  sorry

end pyramid_height_l1381_138142


namespace multiple_of_six_as_four_cubes_integer_as_five_cubes_l1381_138109

-- Part (a)
theorem multiple_of_six_as_four_cubes (n : ℤ) : ∃ a b c d : ℤ, 6 * n = a ^ 3 + b ^ 3 + c ^ 3 + d ^ 3 :=
by
  sorry

-- Part (b)
theorem integer_as_five_cubes (k : ℤ) : ∃ a b c d e : ℤ, k = a ^ 3 + b ^ 3 + c ^ 3 + d ^ 3 + e ^ 3 :=
by
  have h := multiple_of_six_as_four_cubes
  sorry

end multiple_of_six_as_four_cubes_integer_as_five_cubes_l1381_138109


namespace triangle_angle_C_right_l1381_138118

theorem triangle_angle_C_right {a b c A B C : ℝ}
  (h1 : a / Real.sin B + b / Real.sin A = 2 * c) 
  (h2 : a / Real.sin A = b / Real.sin B) 
  (h3 : b / Real.sin B = c / Real.sin C) : 
  C = Real.pi / 2 :=
by sorry

end triangle_angle_C_right_l1381_138118


namespace roots_equation_l1381_138177

theorem roots_equation (α β : ℝ) (h1 : α^2 - 4 * α - 1 = 0) (h2 : β^2 - 4 * β - 1 = 0) :
  3 * α^3 + 4 * β^2 = 80 + 35 * α :=
by
  sorry

end roots_equation_l1381_138177


namespace simplify_abs_expression_l1381_138197

theorem simplify_abs_expression
  (a b : ℝ)
  (h1 : a < 0)
  (h2 : a * b < 0)
  : |a - b - 3| - |4 + b - a| = -1 := by
  sorry

end simplify_abs_expression_l1381_138197


namespace arithmetic_sequence_sum_l1381_138144

/-
In an arithmetic sequence, if the sum of terms \( a_2 + a_3 + a_4 + a_5 + a_6 = 90 \), 
prove that \( a_1 + a_7 = 36 \).
-/

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_sum : a 2 + a 3 + a 4 + a 5 + a 6 = 90) :
  a 1 + a 7 = 36 := by
  sorry

end arithmetic_sequence_sum_l1381_138144


namespace point_M_coordinates_l1381_138166

open Real

theorem point_M_coordinates (θ : ℝ) (h_tan : tan θ = -4 / 3) (h_theta : π / 2 < θ ∧ θ < π) :
  let x := 5 * cos θ
  let y := 5 * sin θ
  (x, y) = (-3, 4) := 
by 
  sorry

end point_M_coordinates_l1381_138166


namespace g_composition_evaluation_l1381_138149

def g (x : ℤ) : ℤ :=
  if x < 5 then x^3 + x^2 - 6 else 2 * x - 18

theorem g_composition_evaluation : g (g (g 16)) = 2 := by
  sorry

end g_composition_evaluation_l1381_138149


namespace pascal_fifth_number_in_row_15_l1381_138110

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l1381_138110


namespace negative_implies_neg_reciprocal_positive_l1381_138146

theorem negative_implies_neg_reciprocal_positive {x : ℝ} (h : x < 0) : -x⁻¹ > 0 :=
sorry

end negative_implies_neg_reciprocal_positive_l1381_138146


namespace stickers_distribution_l1381_138106

-- Define the mathematical problem: distributing 10 stickers among 5 sheets with each sheet getting at least one sticker.

def partitions_count (n k : ℕ) : ℕ := sorry

theorem stickers_distribution (n : ℕ) (k : ℕ) (h₁ : n = 10) (h₂ : k = 5) :
  partitions_count (n - k) k = 7 := by
  sorry

end stickers_distribution_l1381_138106


namespace total_sides_of_cookie_cutters_l1381_138120

theorem total_sides_of_cookie_cutters :
  let top_layer := 6 * 3
  let middle_layer := 4 * 4 + 2 * 6
  let bottom_layer := 3 * 8 + 5 * 0 + 1 * 5
  let total_sides := top_layer + middle_layer + bottom_layer
  total_sides = 75 :=
by
  let top_layer := 6 * 3
  let middle_layer := 4 * 4 + 2 * 6
  let bottom_layer := 3 * 8 + 5 * 0 + 1 * 5
  let total_sides := top_layer + middle_layer + bottom_layer
  show total_sides = 75
  sorry

end total_sides_of_cookie_cutters_l1381_138120


namespace find_an_find_n_l1381_138138

noncomputable def a_n (n : ℕ) : ℤ := 12 + (n - 1) * 2

noncomputable def S_n (n : ℕ) : ℤ := n * 12 + (n * (n - 1) / 2) * 2

theorem find_an (n : ℕ) : a_n n = 2 * n + 10 :=
by sorry

theorem find_n (n : ℕ) (S_n : ℤ) : S_n = 242 → n = 11 :=
by sorry

end find_an_find_n_l1381_138138


namespace mass_of_cork_l1381_138107

theorem mass_of_cork (ρ_p ρ_w ρ_s : ℝ) (m_p x : ℝ) :
  ρ_p = 2.15 * 10^4 → 
  ρ_w = 2.4 * 10^2 →
  ρ_s = 4.8 * 10^2 →
  m_p = 86.94 →
  x = 2.4 * 10^2 * (m_p / ρ_p) →
  x = 85 :=
by
  intros
  sorry

end mass_of_cork_l1381_138107


namespace sin_pi_over_six_l1381_138115

theorem sin_pi_over_six : Real.sin (π / 6) = 1 / 2 :=
sorry

end sin_pi_over_six_l1381_138115


namespace floor_of_ten_times_expected_value_of_fourth_largest_l1381_138108

theorem floor_of_ten_times_expected_value_of_fourth_largest : 
  let n := 90
  let m := 5
  let k := 4
  let E := (k * (n + 1)) / (m + 1)
  ∀ (X : Fin m → Fin n) (h : ∀ i j : Fin m, i ≠ j → X i ≠ X j), 
  Nat.floor (10 * E) = 606 := 
by
  sorry

end floor_of_ten_times_expected_value_of_fourth_largest_l1381_138108


namespace UPOMB_position_l1381_138157

-- Define the set of letters B, M, O, P, and U
def letters : List Char := ['B', 'M', 'O', 'P', 'U']

-- Define the word UPOMB
def word := "UPOMB"

-- Define a function that calculates the factorial of a number
def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to calculate the position of a word in the alphabetical permutations of a list of characters
def word_position (w : String) (chars : List Char) : Nat :=
  let rec aux (w : List Char) (remaining : List Char) : Nat :=
    match w with
    | [] => 1
    | c :: cs =>
      let before_count := remaining.filter (· < c) |>.length
      let rest_count := factorial (remaining.length - 1)
      before_count * rest_count + aux cs (remaining.erase c)
  aux w.data chars

-- The desired theorem statement
theorem UPOMB_position : word_position word letters = 119 := by
  sorry

end UPOMB_position_l1381_138157


namespace smallest_digit_divisible_by_11_l1381_138188

theorem smallest_digit_divisible_by_11 : ∃ d : ℕ, (0 ≤ d ∧ d ≤ 9) ∧ d = 6 ∧ (d + 7 - (4 + 3 + 6)) % 11 = 0 := by
  sorry

end smallest_digit_divisible_by_11_l1381_138188


namespace total_toys_given_l1381_138139

theorem total_toys_given (toys_for_boys : ℕ) (toys_for_girls : ℕ) (h1 : toys_for_boys = 134) (h2 : toys_for_girls = 269) : 
  toys_for_boys + toys_for_girls = 403 := 
by 
  sorry

end total_toys_given_l1381_138139


namespace exists_nat_m_inequality_for_large_n_l1381_138181

section sequence_problem

-- Define the sequence
noncomputable def a (n : ℕ) : ℚ :=
if n = 7 then 16 / 3 else
if n < 7 then 0 else -- hands off values before a7 that are not needed
3 * a (n - 1) / (7 - a (n - 1) + 4)

-- Define the properties to be proven
theorem exists_nat_m {m : ℕ} :
  (∀ n, n > m → a n < 2) ∧ (∀ n, n ≤ m → a n > 2) :=
sorry

theorem inequality_for_large_n (n : ℕ) (hn : n ≥ 10) :
  (a (n - 1) + a n + 1) / 2 < a n :=
sorry

end sequence_problem

end exists_nat_m_inequality_for_large_n_l1381_138181


namespace problem_statement_l1381_138168

theorem problem_statement (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) :
(a + b = 2) ∧ ¬( (a^2 + a > 2) ∧ (b^2 + b > 2) ) := by
  sorry

end problem_statement_l1381_138168


namespace solve_prime_equation_l1381_138154

theorem solve_prime_equation (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
    p ^ 2 - 6 * p * q + q ^ 2 + 3 * q - 1 = 0 ↔ (p = 17 ∧ q = 3) :=
by
  sorry

end solve_prime_equation_l1381_138154


namespace product_equals_eight_l1381_138112

theorem product_equals_eight : (1 + 1 / 2) * (1 + 1 / 3) * (1 + 1 / 4) * (1 + 1 / 5) * (1 + 1 / 6) * (1 + 1 / 7) = 8 := 
sorry

end product_equals_eight_l1381_138112


namespace fg_of_5_eq_163_l1381_138164

def g (x : ℤ) : ℤ := 4 * x + 9
def f (x : ℤ) : ℤ := 6 * x - 11

theorem fg_of_5_eq_163 : f (g 5) = 163 :=
by
  sorry

end fg_of_5_eq_163_l1381_138164


namespace basis_group1_basis_group2_basis_group3_basis_l1381_138135

def vector (α : Type*) := α × α

def is_collinear (v1 v2: vector ℝ) : Prop :=
  v1.1 * v2.2 - v2.1 * v1.2 = 0

def group1_v1 : vector ℝ := (-1, 2)
def group1_v2 : vector ℝ := (5, 7)

def group2_v1 : vector ℝ := (3, 5)
def group2_v2 : vector ℝ := (6, 10)

def group3_v1 : vector ℝ := (2, -3)
def group3_v2 : vector ℝ := (0.5, 0.75)

theorem basis_group1 : ¬ is_collinear group1_v1 group1_v2 :=
by sorry

theorem basis_group2 : is_collinear group2_v1 group2_v2 :=
by sorry

theorem basis_group3 : ¬ is_collinear group3_v1 group3_v2 :=
by sorry

theorem basis : (¬ is_collinear group1_v1 group1_v2) ∧ (is_collinear group2_v1 group2_v2) ∧ (¬ is_collinear group3_v1 group3_v2) :=
by sorry

end basis_group1_basis_group2_basis_group3_basis_l1381_138135


namespace inequality_solution_sets_l1381_138117

theorem inequality_solution_sets (a b m : ℝ) (h_sol_set : ∀ x, x^2 - a * x - 2 > 0 ↔ x < -1 ∨ x > b) (hb : b > -1) (hm : m > -1 / 2) :
  a = 1 ∧ b = 2 ∧ 
  (if m > 0 then ∀ x, (x < -1/m ∨ x > 2) ↔ (mx + 1) * (x - 2) > 0 
   else if m = 0 then ∀ x, x > 2 ↔ (mx + 1) * (x - 2) > 0 
   else ∀ x, (2 < x ∧ x < -1/m) ↔ (mx + 1) * (x - 2) > 0) :=
by
  sorry

end inequality_solution_sets_l1381_138117


namespace rancher_cows_l1381_138151

theorem rancher_cows : ∃ (C H : ℕ), (C = 5 * H) ∧ (C + H = 168) ∧ (C = 140) := by
  sorry

end rancher_cows_l1381_138151


namespace find_a_from_conditions_l1381_138158

theorem find_a_from_conditions (a b c : ℤ) 
  (h1 : a + b = c) 
  (h2 : b + c = 9) 
  (h3 : c = 4) : 
  a = -1 := 
by 
  sorry

end find_a_from_conditions_l1381_138158


namespace average_price_mixed_sugar_l1381_138103

def average_selling_price_per_kg (weightA weightB weightC costA costB costC : ℕ) := 
  (costA * weightA + costB * weightB + costC * weightC) / (weightA + weightB + weightC : ℚ)

theorem average_price_mixed_sugar : 
  average_selling_price_per_kg 3 2 5 28 20 12 = 18.4 := 
by
  sorry

end average_price_mixed_sugar_l1381_138103


namespace cost_per_mile_proof_l1381_138198

noncomputable def daily_rental_cost : ℝ := 50
noncomputable def daily_budget : ℝ := 88
noncomputable def max_miles : ℝ := 190.0

theorem cost_per_mile_proof : 
  (daily_budget - daily_rental_cost) / max_miles = 0.20 := 
by
  sorry

end cost_per_mile_proof_l1381_138198


namespace q_evaluation_l1381_138182

def q (x y : ℤ) : ℤ :=
if x >= 0 ∧ y >= 0 then x - y
else if x < 0 ∧ y < 0 then x + 3 * y
else 2 * x + 2 * y

theorem q_evaluation : q (q 1 (-1)) (q (-2) (-3)) = -22 := by
sorry

end q_evaluation_l1381_138182


namespace hyperbola_eccentricity_l1381_138186

theorem hyperbola_eccentricity (x y : ℝ) :
  (∃ a b : ℝ, a^2 = 4 ∧ b^2 = 12 ∧ (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 4) ∧
  ∃ e : ℝ, e = c / a ∧ e = 2) :=
sorry

end hyperbola_eccentricity_l1381_138186


namespace min_value_y_l1381_138153

theorem min_value_y (x : ℝ) : ∃ x : ℝ, (y = x^2 + 16 * x + 20) ∧ ∀ z : ℝ, (y = z^2 + 16 * z + 20) → y ≥ -44 := 
sorry

end min_value_y_l1381_138153


namespace find_largest_number_among_three_l1381_138156

noncomputable def A (B : ℝ) := 2 * B - 43
noncomputable def C (A : ℝ) := 0.5 * A + 5

-- The main statement to be proven
theorem find_largest_number_among_three : 
  ∃ (A B C : ℝ), 
  A + B + C = 50 ∧ 
  A = 2 * B - 43 ∧ 
  C = 0.5 * A + 5 ∧ 
  max A (max B C) = 27.375 :=
by
  sorry

end find_largest_number_among_three_l1381_138156


namespace product_xy_min_value_x_plus_y_min_value_attained_l1381_138196

theorem product_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) : x * y = 64 := 
sorry

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) : 
  x + y = 18 := 
sorry

-- Additional theorem to prove that the minimum value is attained when x = 6 and y = 12
theorem min_value_attained (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) :
  x = 6 ∧ y = 12 := 
sorry

end product_xy_min_value_x_plus_y_min_value_attained_l1381_138196


namespace pure_imaginary_iff_real_part_zero_l1381_138183

theorem pure_imaginary_iff_real_part_zero (a b : ℝ) : (∃ z : ℂ, z = a + bi ∧ z.im ≠ 0) ↔ (a = 0 ∧ b ≠ 0) :=
sorry

end pure_imaginary_iff_real_part_zero_l1381_138183


namespace middle_number_l1381_138143

theorem middle_number {a b c : ℕ} (h1 : a + b = 12) (h2 : a + c = 17) (h3 : b + c = 19) (h4 : a < b) (h5 : b < c) : b = 7 :=
sorry

end middle_number_l1381_138143


namespace nth_equation_proof_l1381_138165

theorem nth_equation_proof (n : ℕ) (h : n ≥ 1) :
  1 / (n + 1 : ℚ) + 1 / (n * (n + 1)) = 1 / n := 
sorry

end nth_equation_proof_l1381_138165


namespace cards_with_1_count_l1381_138189

theorem cards_with_1_count (m k : ℕ) 
  (h1 : k = m + 100) 
  (sum_of_products : (m * (m - 1) / 2) + (k * (k - 1) / 2) - m * k = 1000) : 
  m = 3950 :=
by
  sorry

end cards_with_1_count_l1381_138189


namespace quadratic_roots_inequality_solution_set_l1381_138104

-- Problem 1 statement
theorem quadratic_roots : 
  (∀ x : ℝ, x^2 - 4 * x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) := 
by
  sorry

-- Problem 2 statement
theorem inequality_solution_set :
  (∀ x : ℝ, (x - 2 * (x - 1) ≤ 1 ∧ (1 + x) / 3 > x - 1) ↔ -1 ≤ x ∧ x < 2) :=
by
  sorry

end quadratic_roots_inequality_solution_set_l1381_138104


namespace solution_l1381_138170

-- Definition of the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + 3 * m * x + 1

-- Statement of the problem
theorem solution (m : ℝ) (x : ℝ) (h : quadratic_equation m x = (m - 2) * x^2 + 3 * m * x + 1) : m ≠ 2 :=
by
  sorry

end solution_l1381_138170


namespace possible_slopes_of_line_intersecting_ellipse_l1381_138191

theorem possible_slopes_of_line_intersecting_ellipse (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ m ∈ (Set.Iic (-2/5) ∪ Set.Ici (2/5)) :=
by
  sorry

end possible_slopes_of_line_intersecting_ellipse_l1381_138191


namespace watermelon_price_in_units_of_1000_l1381_138147

theorem watermelon_price_in_units_of_1000
  (initial_price discounted_price: ℝ)
  (h_price: initial_price = 5000)
  (h_discount: discounted_price = initial_price - 200) :
  discounted_price / 1000 = 4.8 :=
by
  sorry

end watermelon_price_in_units_of_1000_l1381_138147


namespace inequality_holds_for_all_x_l1381_138134

theorem inequality_holds_for_all_x (m : ℝ) (h : ∀ x : ℝ, |x + 5| ≥ m + 2) : m ≤ -2 :=
sorry

end inequality_holds_for_all_x_l1381_138134


namespace probability_of_disease_given_positive_test_l1381_138128

-- Define the probabilities given in the problem
noncomputable def pr_D : ℝ := 1 / 1000
noncomputable def pr_Dc : ℝ := 1 - pr_D
noncomputable def pr_T_given_D : ℝ := 1
noncomputable def pr_T_given_Dc : ℝ := 0.05

-- Define the total probability of a positive test using the law of total probability
noncomputable def pr_T := 
  pr_T_given_D * pr_D + pr_T_given_Dc * pr_Dc

-- Using Bayes' theorem
noncomputable def pr_D_given_T := 
  pr_T_given_D * pr_D / pr_T

-- Theorem to prove the desired probability
theorem probability_of_disease_given_positive_test : 
  pr_D_given_T = 1 / 10 :=
by
  sorry

end probability_of_disease_given_positive_test_l1381_138128


namespace ages_sum_13_and_product_72_l1381_138145

theorem ages_sum_13_and_product_72 (g b s : ℕ) (h1 : b < g) (h2 : g < s) (h3 : b * g * s = 72) : b + g + s = 13 :=
sorry

end ages_sum_13_and_product_72_l1381_138145


namespace simplify_fraction_l1381_138162

theorem simplify_fraction (a b c : ℕ) (h1 : a = 2^2 * 3^2 * 5) 
  (h2 : b = 2^1 * 3^3 * 5) (h3 : c = (2^1 * 3^2 * 5)) :
  (a / c) / (b / c) = 2 / 3 := 
by {
  sorry
}

end simplify_fraction_l1381_138162


namespace total_percentage_of_samplers_l1381_138125

theorem total_percentage_of_samplers :
  let pA := 12
  let pB := 5
  let pC := 9
  let pD := 4
  let pA_not_caught := 7
  let pB_not_caught := 6
  let pC_not_caught := 3
  let pD_not_caught := 8
  (pA + pA_not_caught + pB + pB_not_caught + pC + pC_not_caught + pD + pD_not_caught) = 54 :=
by
  let pA := 12
  let pB := 5
  let pC := 9
  let pD := 4
  let pA_not_caught := 7
  let pB_not_caught := 6
  let pC_not_caught := 3
  let pD_not_caught := 8
  sorry

end total_percentage_of_samplers_l1381_138125


namespace work_done_by_student_l1381_138155

theorem work_done_by_student
  (M : ℝ)  -- mass of the student
  (m : ℝ)  -- mass of the stone
  (h : ℝ)  -- height from which the stone is thrown
  (L : ℝ)  -- distance on the ice where the stone lands
  (g : ℝ)  -- acceleration due to gravity
  (t : ℝ := Real.sqrt (2 * h / g))  -- time it takes for the stone to hit the ice derived from free fall equation
  (Vk : ℝ := L / t)  -- initial speed of the stone derived from horizontal motion
  (Vu : ℝ := m / M * Vk)  -- initial speed of the student derived from conservation of momentum
  : (1/2 * m * Vk^2 + (1/2) * M * Vu^2) = 126.74 :=
by
  sorry

end work_done_by_student_l1381_138155


namespace tims_seashells_now_l1381_138133

def initial_seashells : ℕ := 679
def seashells_given_away : ℕ := 172

theorem tims_seashells_now : (initial_seashells - seashells_given_away) = 507 :=
by
  sorry

end tims_seashells_now_l1381_138133


namespace intersection_A_B_l1381_138111

-- Defining set A condition
def A : Set ℝ := {x | x - 1 < 2}

-- Defining set B condition
def B : Set ℝ := {y | ∃ x ∈ A, y = 2^x}

-- The goal to prove
theorem intersection_A_B : {x | x > 0 ∧ x < 3} = (A ∩ { x | 0 < x ∧ x < 8 }) :=
by
  sorry

end intersection_A_B_l1381_138111


namespace question1_question2_l1381_138178

-- Condition: p
def p (x : ℝ) : Prop := 4 * x^2 + 12 * x - 7 ≤ 0

-- Condition: q
def q (a : ℝ) (x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

-- Question 1 statement: Given p is true and q is false when a = 0, find range of x
theorem question1 (x : ℝ) (h : p x ∧ ¬q 0 x) : -7/2 ≤ x ∧ x < -3 :=
sorry

-- Question 2 statement: If p is a sufficient condition for q, find range of a
theorem question2 (a : ℝ) (h : ∀ x, p x → q a x) : -5/2 ≤ a ∧ a ≤ 1/2 :=
sorry

end question1_question2_l1381_138178


namespace cuboid_dimensions_exist_l1381_138169

theorem cuboid_dimensions_exist (l w h : ℝ) 
  (h1 : l * w = 5) 
  (h2 : l * h = 8) 
  (h3 : w * h = 10) 
  (h4 : l * w * h = 200) : 
  ∃ (l w h : ℝ), l = 4 ∧ w = 2.5 ∧ h = 2 := 
sorry

end cuboid_dimensions_exist_l1381_138169


namespace largest_possible_package_l1381_138148

/-- Alice, Bob, and Carol bought certain numbers of markers and the goal is to find the greatest number of markers per package. -/
def alice_markers : Nat := 60
def bob_markers : Nat := 36
def carol_markers : Nat := 48

theorem largest_possible_package :
  Nat.gcd (Nat.gcd alice_markers bob_markers) carol_markers = 12 :=
sorry

end largest_possible_package_l1381_138148


namespace find_range_of_function_l1381_138141

variable (a : ℝ) (x : ℝ)

def func (a x : ℝ) : ℝ := x^2 - 2*a*x - 1

theorem find_range_of_function (a : ℝ) :
  if a < 0 then
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ -1 ≤ y ∧ y ≤ 3 - 4*a
  else if 0 ≤ a ∧ a ≤ 1 then
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ -(a^2 + 1) ≤ y ∧ y ≤ 3 - 4*a
  else if 1 < a ∧ a ≤ 2 then
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ -(a^2 + 1) ≤ y ∧ y ≤ -1
  else
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ 3 - 4*a ≤ y ∧ y ≤ -1
:= sorry

end find_range_of_function_l1381_138141


namespace find_a3_plus_a9_l1381_138167

noncomputable def arithmetic_sequence (a : ℕ → ℕ) : Prop := 
∀ n m : ℕ, a (n + m) = a n + a m

theorem find_a3_plus_a9 (a : ℕ → ℕ) 
  (is_arithmetic : arithmetic_sequence a)
  (h : a 1 + a 6 + a 11 = 3) : 
  a 3 + a 9 = 2 :=
sorry

end find_a3_plus_a9_l1381_138167


namespace polynomial_identity_l1381_138127

noncomputable def p (x : ℝ) : ℝ := x 

theorem polynomial_identity (p : ℝ → ℝ) (h : ∀ q : ℝ → ℝ, ∀ x : ℝ, p (q x) = q (p x)) : 
  (∀ x : ℝ, p x = x) :=
by
  sorry

end polynomial_identity_l1381_138127


namespace simplify_expression_l1381_138113

theorem simplify_expression (x y z : ℝ) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) 
  (hx2 : x ≠ 2) (hy3 : y ≠ 3) (hz5 : z ≠ 5) :
  ( ( (x - 2) / (3 - z) * ( (y - 3) / (5 - x) ) * ( (z - 5) / (2 - y) ) ) ^ 2 ) = 1 :=
by
  sorry

end simplify_expression_l1381_138113
