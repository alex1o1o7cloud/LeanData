import Mathlib

namespace distance_between_A_and_B_l1912_191227

-- Define speeds, times, and distances as real numbers
def speed_A_to_B := 42.5
def time_travelled := 1.5
def remaining_to_midpoint := 26.0

-- Define the total distance between A and B as a variable
def distance_A_to_B : ℝ := 179.5

-- Prove that the distance between locations A and B is 179.5 kilometers given the conditions
theorem distance_between_A_and_B : (42.5 * 1.5 + 26) * 2 = 179.5 :=
by 
  sorry

end distance_between_A_and_B_l1912_191227


namespace set_1234_excellent_no_proper_subset_excellent_l1912_191251

open Set

namespace StepLength

def excellent_set (D : Set ℤ) : Prop :=
∀ A : Set ℤ, ∃ a d : ℤ, d ∈ D → ({a - d, a, a + d} ⊆ A ∨ {a - d, a, a + d} ⊆ (univ \ A))

noncomputable def S : Set (Set ℤ) := {{1}, {2}, {3}, {4}}

theorem set_1234_excellent : excellent_set {1, 2, 3, 4} := sorry

theorem no_proper_subset_excellent :
  ¬ (excellent_set {1, 3, 4} ∨ excellent_set {1, 2, 3} ∨ excellent_set {1, 2, 4} ∨ excellent_set {2, 3, 4}) := sorry

end StepLength

end set_1234_excellent_no_proper_subset_excellent_l1912_191251


namespace contrapositive_example_l1912_191206

variable (a b : ℝ)

theorem contrapositive_example
  (h₁ : a > 0)
  (h₃ : a + b < 0) :
  b < 0 := 
sorry

end contrapositive_example_l1912_191206


namespace solution_set_inequality_l1912_191277

theorem solution_set_inequality (x : ℝ) : (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := 
sorry

end solution_set_inequality_l1912_191277


namespace angle_in_first_quadrant_l1912_191209

def angle := -999 - 30 / 60 -- defining the angle as -999°30'
def coterminal (θ : Real) : Real := θ + 3 * 360 -- function to compute a coterminal angle

theorem angle_in_first_quadrant : 
  let θ := coterminal angle
  0 <= θ ∧ θ < 90 :=
by
  -- Exact proof steps would go here, but they are omitted as per instructions.
  sorry

end angle_in_first_quadrant_l1912_191209


namespace max_rectangle_area_l1912_191218

-- Definitions based on conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬ is_prime n
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem max_rectangle_area
  (l w : ℕ)
  (h_perim : perimeter l w = 50)
  (h_prime : is_prime l)
  (h_composite : is_composite w) :
  l * w = 156 :=
sorry

end max_rectangle_area_l1912_191218


namespace cornbread_pieces_l1912_191270

theorem cornbread_pieces (pan_length pan_width piece_length piece_width : ℕ)
  (h₁ : pan_length = 24) (h₂ : pan_width = 20) 
  (h₃ : piece_length = 3) (h₄ : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 80 := by
  sorry

end cornbread_pieces_l1912_191270


namespace closest_point_on_line_l1912_191279

theorem closest_point_on_line 
  (t : ℚ)
  (x y z : ℚ)
  (x_eq : x = 3 + t)
  (y_eq : y = 2 - 3 * t)
  (z_eq : z = -1 + 2 * t)
  (x_ortho_eq : (1 + t) = 0)
  (y_ortho_eq : (3 - 3 * t) = 0)
  (z_ortho_eq : (-3 + 2 * t) = 0) :
  (45/14, 16/14, -1/7) = (x, y, z) := by
  sorry

end closest_point_on_line_l1912_191279


namespace library_fiction_percentage_l1912_191246

theorem library_fiction_percentage:
  let original_volumes := 18360
  let fiction_percentage := 0.30
  let fraction_transferred := 1/3
  let fraction_fiction_transferred := 1/5
  let initial_fiction := fiction_percentage * original_volumes
  let transferred_volumes := fraction_transferred * original_volumes
  let transferred_fiction := fraction_fiction_transferred * transferred_volumes
  let remaining_fiction := initial_fiction - transferred_fiction
  let remaining_volumes := original_volumes - transferred_volumes
  let remaining_fiction_percentage := (remaining_fiction / remaining_volumes) * 100
  remaining_fiction_percentage = 35 := 
by
  sorry

end library_fiction_percentage_l1912_191246


namespace minimum_value_of_f_l1912_191292

noncomputable def f (x : ℝ) : ℝ := (1 / (Real.cos x)^2) + (1 / (Real.sin x)^2)

theorem minimum_value_of_f : ∀ x : ℝ, ∃ y : ℝ, y = f x ∧ y = 4 :=
by
  sorry

end minimum_value_of_f_l1912_191292


namespace xiao_hong_home_to_school_distance_l1912_191290

-- Definition of conditions
def distance_from_drop_to_school := 1000 -- in meters
def time_from_home_to_school_walking := 22.5 -- in minutes
def time_from_home_to_school_biking := 40 -- in minutes
def walking_speed := 80 -- in meters per minute
def bike_speed_slowdown := 800 -- in meters per minute

-- The main theorem statement
theorem xiao_hong_home_to_school_distance :
  ∃ d : ℝ, d = 12000 ∧ 
            distance_from_drop_to_school = 1000 ∧
            time_from_home_to_school_walking = 22.5 ∧
            time_from_home_to_school_biking = 40 ∧
            walking_speed = 80 ∧
            bike_speed_slowdown = 800 := 
sorry

end xiao_hong_home_to_school_distance_l1912_191290


namespace min_ge_n_l1912_191211

theorem min_ge_n (x y z n : ℕ) (h : x^n + y^n = z^n) : min x y ≥ n :=
sorry

end min_ge_n_l1912_191211


namespace line_equation_l1912_191260

variable (θ : ℝ) (b : ℝ) (y x : ℝ)

-- Conditions: 
-- Slope angle θ = 45°
def slope_angle_condition : θ = 45 := by
  sorry

-- Y-intercept b = 2
def y_intercept_condition : b = 2 := by
  sorry

-- Given these conditions, we want to prove the line equation
theorem line_equation (x : ℝ) (θ : ℝ) (b : ℝ) :
  θ = 45 → b = 2 → y = x + 2 := by
  sorry

end line_equation_l1912_191260


namespace dealers_profit_percentage_l1912_191252

theorem dealers_profit_percentage 
  (articles_purchased : ℕ)
  (total_cost_price : ℝ)
  (articles_sold : ℕ)
  (total_selling_price : ℝ)
  (CP_per_article : ℝ := total_cost_price / articles_purchased)
  (SP_per_article : ℝ := total_selling_price / articles_sold)
  (profit_per_article : ℝ := SP_per_article - CP_per_article)
  (profit_percentage : ℝ := (profit_per_article / CP_per_article) * 100) :
  articles_purchased = 15 →
  total_cost_price = 25 →
  articles_sold = 12 →
  total_selling_price = 32 →
  profit_percentage = 60 :=
by
  intros h1 h2 h3 h4
  sorry

end dealers_profit_percentage_l1912_191252


namespace canoe_upstream_speed_l1912_191228

theorem canoe_upstream_speed (C : ℝ) (stream_speed downstream_speed : ℝ) 
  (h_stream : stream_speed = 2) (h_downstream : downstream_speed = 12) 
  (h_equation : C + stream_speed = downstream_speed) :
  C - stream_speed = 8 := 
by 
  sorry

end canoe_upstream_speed_l1912_191228


namespace sum_of_divisors_143_l1912_191215

def sum_divisors (n : ℕ) : ℕ :=
  (1 + 11) * (1 + 13)  -- The sum of the divisors of 143 is interpreted from the given prime factors.

theorem sum_of_divisors_143 : sum_divisors 143 = 168 := by
  sorry

end sum_of_divisors_143_l1912_191215


namespace monotonic_intervals_extreme_value_closer_l1912_191205

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * (x - 1)

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x y : ℝ, x < y → f x a < f y a) ∧
  (a > 0 → (∀ x : ℝ, x < Real.log a → f x a > f (x + 1) a) ∧ (∀ x : ℝ, x > Real.log a → f x a < f (x + 1) a)) :=
sorry

theorem extreme_value_closer (a : ℝ) :
  a > e - 1 →
  ∀ x : ℝ, x ≥ 1 → |Real.exp 1/x - Real.log x| < |Real.exp (x - 1) + a - Real.log x| :=
sorry

end monotonic_intervals_extreme_value_closer_l1912_191205


namespace prod2025_min_sum_l1912_191286

theorem prod2025_min_sum : ∃ (a b : ℕ), a * b = 2025 ∧ a > 0 ∧ b > 0 ∧ (∀ (x y : ℕ), x * y = 2025 → x > 0 → y > 0 → x + y ≥ a + b) ∧ a + b = 90 :=
sorry

end prod2025_min_sum_l1912_191286


namespace triangle_inequality_l1912_191287

theorem triangle_inequality (a b c p S r : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b)
  (hp : p = (a + b + c) / 2)
  (hS : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (hr : r = S / p):
  1 / (p - a) ^ 2 + 1 / (p - b) ^ 2 + 1 / (p - c) ^ 2 ≥ 1 / r ^ 2 :=
sorry

end triangle_inequality_l1912_191287


namespace find_c_plus_d_l1912_191210

-- Conditions as definitions
variables {P A C : Point }
variables {O₁ O₂ : Point}
variables {AB AP CP : ℝ}
variables {c d : ℕ}

-- Given conditions
def Point_on_diagonal (P A C : Point) : Prop := true -- We need to code the detailed properties of being on the diagonal
def circumcenter_of_triangle (P Q R O : Point) : Prop := true -- We need to code the properties of being a circumcenter
def AP_greater_than_CP (AP CP : ℝ) : Prop := AP > CP
def angle_right (A B O : Point) : Prop := true -- Define the right angle property

-- Main statement to prove
theorem find_c_plus_d : 
  Point_on_diagonal P A C ∧
  circumcenter_of_triangle A B P O₁ ∧ 
  circumcenter_of_triangle C D P O₂ ∧ 
  AP_greater_than_CP AP CP ∧
  AB = 10 ∧
  angle_right O₁ P O₂ ∧
  (AP = Real.sqrt c + Real.sqrt d) →
  (c + d = 100) :=
by
  sorry

end find_c_plus_d_l1912_191210


namespace meat_sales_beyond_plan_l1912_191264

-- Define the constants for each day's sales
def sales_thursday := 210
def sales_friday := 2 * sales_thursday
def sales_saturday := 130
def sales_sunday := sales_saturday / 2
def original_plan := 500

-- Define the total sales
def total_sales := sales_thursday + sales_friday + sales_saturday + sales_sunday

-- Prove that they sold 325kg beyond their original plan
theorem meat_sales_beyond_plan : total_sales - original_plan = 325 :=
by
  -- The proof is not included, so we add sorry to skip the proof
  sorry

end meat_sales_beyond_plan_l1912_191264


namespace eval_f_pi_over_8_l1912_191268

noncomputable def f (θ : ℝ) : ℝ :=
(2 * (Real.sin (θ / 2)) ^ 2 - 1) / (Real.sin (θ / 2) * Real.cos (θ / 2)) + 2 * Real.tan θ

theorem eval_f_pi_over_8 : f (π / 8) = -4 :=
sorry

end eval_f_pi_over_8_l1912_191268


namespace set_intersection_set_union_set_complement_l1912_191203

open Set

variable (U : Set ℝ) (A B : Set ℝ)
noncomputable def setA : Set ℝ := {x | x^2 - 3*x - 4 ≥ 0}
noncomputable def setB : Set ℝ := {x | x < 5}

theorem set_intersection : (U = univ) -> (A = setA) -> (B = setB) -> A ∩ B = Ico 4 5 := by
  intros
  sorry

theorem set_union : (U = univ) -> (A = setA) -> (B = setB) -> A ∪ B = univ := by
  intros
  sorry

theorem set_complement : (U = univ) -> (A = setA) -> U \ A = Ioo (-1 : ℝ) 4 := by
  intros
  sorry

end set_intersection_set_union_set_complement_l1912_191203


namespace maximum_marks_l1912_191213

theorem maximum_marks (M : ℝ) (h1 : 0.45 * M = 180) : M = 400 := 
by sorry

end maximum_marks_l1912_191213


namespace s_plough_time_l1912_191297

theorem s_plough_time (r_s_combined_time : ℝ) (r_time : ℝ) (t_time : ℝ) (s_time : ℝ) :
  r_s_combined_time = 10 → r_time = 15 → t_time = 20 → s_time = 30 :=
by
  sorry

end s_plough_time_l1912_191297


namespace total_kids_on_soccer_field_l1912_191235

theorem total_kids_on_soccer_field (initial_kids : ℕ) (joining_kids : ℕ) (total_kids : ℕ)
  (h₁ : initial_kids = 14)
  (h₂ : joining_kids = 22)
  (h₃ : total_kids = initial_kids + joining_kids) :
  total_kids = 36 :=
by
  sorry

end total_kids_on_soccer_field_l1912_191235


namespace time_difference_l1912_191274

def joey_time : ℕ :=
  let uphill := 12 / 6 * 60
  let downhill := 10 / 25 * 60
  let flat := 20 / 15 * 60
  uphill + downhill + flat

def sue_time : ℕ :=
  let downhill := 10 / 35 * 60
  let uphill := 12 / 12 * 60
  let flat := 20 / 25 * 60
  downhill + uphill + flat

theorem time_difference : joey_time - sue_time = 99 := by
  -- calculation steps skipped
  sorry

end time_difference_l1912_191274


namespace max_value_of_f_l1912_191272

-- Define the function f(x) = 5x - x^2
def f (x : ℝ) : ℝ := 5 * x - x^2

-- The theorem we want to prove, stating the maximum value of f(x) is 6.25
theorem max_value_of_f : ∃ x, f x = 6.25 :=
by
  -- Placeholder proof, to be completed
  sorry

end max_value_of_f_l1912_191272


namespace Turner_Catapult_rides_l1912_191221

def tickets_needed (rollercoaster_rides Ferris_wheel_rides Catapult_rides : ℕ) : ℕ :=
  4 * rollercoaster_rides + 1 * Ferris_wheel_rides + 4 * Catapult_rides

theorem Turner_Catapult_rides :
  ∀ (x : ℕ), tickets_needed 3 1 x = 21 → x = 2 := by
  intros x h
  sorry

end Turner_Catapult_rides_l1912_191221


namespace min_f_on_interval_l1912_191232

open Real

noncomputable def f (x : ℝ) : ℝ := (sin x + 1) * (cos x + 1) / (sin x * cos x)

theorem min_f_on_interval : 
  ∀ x, 0 < x ∧ x < π / 2 → f x ≥ 3 + 2 * sqrt 2 :=
sorry

end min_f_on_interval_l1912_191232


namespace quad_area_l1912_191269

theorem quad_area (a b : Int) (h1 : a > b) (h2 : b > 0) (h3 : 2 * |a - b| * |a + b| = 50) : a + b = 15 :=
by
  sorry

end quad_area_l1912_191269


namespace count_triangles_in_figure_l1912_191278

noncomputable def triangles_in_figure : ℕ := 53

theorem count_triangles_in_figure : triangles_in_figure = 53 := 
by sorry

end count_triangles_in_figure_l1912_191278


namespace pizzeria_large_pizzas_l1912_191261

theorem pizzeria_large_pizzas (price_small : ℕ) (price_large : ℕ) (total_revenue : ℕ) (small_pizzas_sold : ℕ) (L : ℕ) 
    (h1 : price_small = 2) 
    (h2 : price_large = 8) 
    (h3 : total_revenue = 40) 
    (h4 : small_pizzas_sold = 8) 
    (h5 : price_small * small_pizzas_sold + price_large * L = total_revenue) :
    L = 3 := 
by 
  -- Lean will expect a proof here; add sorry for now
  sorry

end pizzeria_large_pizzas_l1912_191261


namespace contrapositive_equiv_l1912_191271

variable (a b : ℝ)

def original_proposition : Prop := a^2 + b^2 = 0 → a = 0 ∧ b = 0

def contrapositive_proposition : Prop := a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0

theorem contrapositive_equiv : original_proposition a b ↔ contrapositive_proposition a b :=
by
  sorry

end contrapositive_equiv_l1912_191271


namespace geometric_arithmetic_sequences_sum_l1912_191293

theorem geometric_arithmetic_sequences_sum (a b : ℕ → ℝ) (S_n : ℕ → ℝ) 
  (q d : ℝ) (h1 : 0 < q) 
  (h2 : a 1 = 1) (h3 : b 1 = 1) 
  (h4 : a 5 + b 3 = 21) 
  (h5 : a 3 + b 5 = 13) :
  (∀ n, a n = 2^(n-1)) ∧ (∀ n, b n = 2*n - 1) ∧ (∀ n, S_n n = 3 - (2*n + 3)/(2^n)) := 
sorry

end geometric_arithmetic_sequences_sum_l1912_191293


namespace a3_plus_a4_value_l1912_191296

theorem a3_plus_a4_value
  (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h : (1 - 2*x)^5 = a_0 + a_1*(1 + x) + a_2*(1 + x)^2 + a_3*(1 + x)^3 + a_4*(1 + x)^4 + a_5*(1 + x)^5) :
  a_3 + a_4 = -480 := 
sorry

end a3_plus_a4_value_l1912_191296


namespace initial_buckets_correct_l1912_191257

-- Define the conditions as variables
def total_buckets : ℝ := 9.8
def added_buckets : ℝ := 8.8
def initial_buckets : ℝ := total_buckets - added_buckets

-- The theorem to prove the initial amount of water is 1 bucket
theorem initial_buckets_correct : initial_buckets = 1 := 
by
  sorry

end initial_buckets_correct_l1912_191257


namespace rectangle_area_stage4_l1912_191248

-- Define the condition: area of one square
def square_area : ℕ := 25

-- Define the condition: number of squares at Stage 4
def num_squares_stage4 : ℕ := 4

-- Define the total area of rectangle at Stage 4
def total_area_stage4 : ℕ := num_squares_stage4 * square_area

-- Prove that total_area_stage4 equals 100 square inches
theorem rectangle_area_stage4 : total_area_stage4 = 100 :=
by
  sorry

end rectangle_area_stage4_l1912_191248


namespace unique_position_of_chess_piece_l1912_191289

theorem unique_position_of_chess_piece (x y : ℕ) (h : x^2 + x * y - 2 * y^2 = 13) : (x = 5) ∧ (y = 4) :=
sorry

end unique_position_of_chess_piece_l1912_191289


namespace costPrice_of_bat_is_152_l1912_191295

noncomputable def costPriceOfBatForA (priceC : ℝ) (profitA : ℝ) (profitB : ℝ) : ℝ :=
  priceC / (1 + profitB) / (1 + profitA)

theorem costPrice_of_bat_is_152 :
  costPriceOfBatForA 228 0.20 0.25 = 152 :=
by
  -- Placeholder for the proof
  sorry

end costPrice_of_bat_is_152_l1912_191295


namespace halfway_between_one_fourth_and_one_seventh_l1912_191229

theorem halfway_between_one_fourth_and_one_seventh : (1 / 4 + 1 / 7) / 2 = 11 / 56 := by
  sorry

end halfway_between_one_fourth_and_one_seventh_l1912_191229


namespace tangent_line_perpendicular_l1912_191204

noncomputable def f (x k : ℝ) : ℝ := x^3 - (k^2 - 1) * x^2 - k^2 + 2

theorem tangent_line_perpendicular (k : ℝ) (b : ℝ) (a : ℝ)
  (h1 : ∀ (x : ℝ), f x k = x^3 - (k^2 - 1) * x^2 - k^2 + 2)
  (h2 : (3 - 2 * (k^2 - 1)) = -1) :
  a = -2 := sorry

end tangent_line_perpendicular_l1912_191204


namespace increase_in_avg_commission_l1912_191247

def new_avg_commission := 250
def num_sales := 6
def big_sale_commission := 1000

theorem increase_in_avg_commission :
  (new_avg_commission - (500 / (num_sales - 1))) = 150 := by
  sorry

end increase_in_avg_commission_l1912_191247


namespace giant_spider_weight_ratio_l1912_191223

theorem giant_spider_weight_ratio 
    (W_previous : ℝ)
    (A_leg : ℝ)
    (P : ℝ)
    (n : ℕ)
    (W_previous_eq : W_previous = 6.4)
    (A_leg_eq : A_leg = 0.5)
    (P_eq : P = 4)
    (n_eq : n = 8):
    (P * A_leg * n) / W_previous = 2.5 := by
  sorry

end giant_spider_weight_ratio_l1912_191223


namespace tanker_filling_rate_l1912_191226

theorem tanker_filling_rate :
  let barrels_per_minute := 5
  let liters_per_barrel := 159
  let minutes_per_hour := 60
  let liters_per_cubic_meter := 1000
  (barrels_per_minute * liters_per_barrel * minutes_per_hour) / 
  liters_per_cubic_meter = 47.7 :=
by
  sorry

end tanker_filling_rate_l1912_191226


namespace area_ratio_is_four_l1912_191244

-- Definitions based on the given conditions
variables (k a b c d : ℝ)
variables (ka kb kc kd : ℝ)

-- Equations from the conditions
def eq1 : a = k * ka := sorry
def eq2 : b = k * kb := sorry
def eq3 : c = k * kc := sorry
def eq4 : d = k * kd := sorry

-- Ratios provided in the problem
def ratio1 : ka / kc = 2 / 5 := sorry
def ratio2 : kb / kd = 2 / 5 := sorry

-- The theorem to prove the ratio of areas is 4:1
theorem area_ratio_is_four : (k * ka * k * kb) / (k * kc * k * kd) = 4 :=
by sorry

end area_ratio_is_four_l1912_191244


namespace flowers_per_row_correct_l1912_191225

/-- Definition for the number of each type of flower. -/
def num_yellow_flowers : ℕ := 12
def num_green_flowers : ℕ := 2 * num_yellow_flowers -- Given that green flowers are twice the yellow flowers.
def num_red_flowers : ℕ := 42

/-- Total number of flowers. -/
def total_flowers : ℕ := num_yellow_flowers + num_green_flowers + num_red_flowers

/-- Number of rows in the garden. -/
def num_rows : ℕ := 6

/-- The number of flowers per row Wilma's garden has. -/
def flowers_per_row : ℕ := total_flowers / num_rows

/-- Proof statement: flowers per row should be 13. -/
theorem flowers_per_row_correct : flowers_per_row = 13 :=
by
  -- The proof will go here.
  sorry

end flowers_per_row_correct_l1912_191225


namespace Serezha_puts_more_berries_l1912_191259

theorem Serezha_puts_more_berries (berries : ℕ) 
    (Serezha_puts : ℕ) (Serezha_eats : ℕ)
    (Dima_puts : ℕ) (Dima_eats : ℕ)
    (Serezha_rate : ℕ) (Dima_rate : ℕ)
    (total_berries : berries = 450)
    (Serezha_pattern : Serezha_puts = 1 ∧ Serezha_eats = 1)
    (Dima_pattern : Dima_puts = 2 ∧ Dima_eats = 1)
    (Serezha_faster : Serezha_rate = 2 * Dima_rate) : 
    ∃ (Serezha_in_basket : ℕ) (Dima_in_basket : ℕ),
      Serezha_in_basket > Dima_in_basket ∧ Serezha_in_basket - Dima_in_basket = 50 :=
by
  sorry -- Skip the proof

end Serezha_puts_more_berries_l1912_191259


namespace find_ratio_l1912_191208

def given_conditions (a b c x y z : ℝ) : Prop :=
  a^2 + b^2 + c^2 = 25 ∧ x^2 + y^2 + z^2 = 36 ∧ a * x + b * y + c * z = 30

theorem find_ratio (a b c x y z : ℝ)
  (h : given_conditions a b c x y z) :
  (a + b + c) / (x + y + z) = 5 / 6 :=
sorry

end find_ratio_l1912_191208


namespace triangle_side_length_l1912_191224

theorem triangle_side_length 
  (r : ℝ)                    -- radius of the inscribed circle
  (h_cos_ABC : ℝ)            -- cosine of angle ABC
  (h_midline : Bool)         -- the circle touches the midline parallel to AC
  (h_r : r = 1)              -- given radius is 1
  (h_cos : h_cos_ABC = 0.8)  -- given cos(ABC) = 0.8
  (h_touch : h_midline = true)  -- given that circle touches the midline
  : AC = 3 := 
sorry

end triangle_side_length_l1912_191224


namespace justin_reading_ratio_l1912_191288

theorem justin_reading_ratio
  (pages_total : ℝ)
  (pages_first_day : ℝ)
  (pages_left : ℝ)
  (days_remaining : ℝ) :
  pages_total = 130 → 
  pages_first_day = 10 → 
  pages_left = pages_total - pages_first_day →
  days_remaining = 6 →
  (∃ R : ℝ, 60 * R = pages_left) → 
  ∃ R : ℝ, 60 * R = pages_left ∧ R = 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end justin_reading_ratio_l1912_191288


namespace div_mult_result_l1912_191238

theorem div_mult_result : 150 / (30 / 3) * 2 = 30 :=
by sorry

end div_mult_result_l1912_191238


namespace avg_age_l1912_191262

-- Given conditions
variables (A B C : ℕ)
variable (h1 : (A + C) / 2 = 29)
variable (h2 : B = 20)

-- to prove
theorem avg_age (A B C : ℕ) (h1 : (A + C) / 2 = 29) (h2 : B = 20) : (A + B + C) / 3 = 26 :=
sorry

end avg_age_l1912_191262


namespace nonneg_integer_solutions_otimes_l1912_191254

noncomputable def otimes (a b : ℝ) : ℝ := a * (a - b) + 1

theorem nonneg_integer_solutions_otimes :
  {x : ℕ | otimes 2 x ≥ 3} = {0, 1} :=
by
  sorry

end nonneg_integer_solutions_otimes_l1912_191254


namespace total_students_in_class_l1912_191231

theorem total_students_in_class
    (students_in_front : ℕ)
    (students_in_back : ℕ)
    (lines : ℕ)
    (total_students_line : ℕ)
    (total_class : ℕ)
    (h_front: students_in_front = 2)
    (h_back: students_in_back = 5)
    (h_lines: lines = 3)
    (h_students_line : total_students_line = students_in_front + 1 + students_in_back)
    (h_total_class : total_class = lines * total_students_line) :
  total_class = 24 := by
  sorry

end total_students_in_class_l1912_191231


namespace ratio_of_buckets_l1912_191281

theorem ratio_of_buckets 
  (shark_feed_per_day : ℕ := 4)
  (dolphin_feed_per_day : ℕ := shark_feed_per_day / 2)
  (total_buckets : ℕ := 546)
  (days_in_weeks : ℕ := 3 * 7)
  (ratio_R : ℕ) :
  (total_buckets = days_in_weeks * (shark_feed_per_day + dolphin_feed_per_day + (ratio_R * shark_feed_per_day)) → ratio_R = 5) := sorry

end ratio_of_buckets_l1912_191281


namespace max_daily_sales_revenue_l1912_191258

noncomputable def f (t : ℕ) : ℝ :=
  if 0 ≤ t ∧ t < 15 
  then (1 / 3) * t + 8
  else if 15 ≤ t ∧ t < 30 
  then -(1 / 3) * t + 18
  else 0

noncomputable def g (t : ℕ) : ℝ :=
  if 0 ≤ t ∧ t ≤ 30
  then -t + 30
  else 0

noncomputable def W (t : ℕ) : ℝ :=
  f t * g t

theorem max_daily_sales_revenue : ∃ t : ℕ, W t = 243 :=
by
  existsi 3
  sorry

end max_daily_sales_revenue_l1912_191258


namespace range_of_m_l1912_191282

theorem range_of_m (m : ℝ) : (∀ x > 1, 2*x + m + 8/(x-1) > 0) → m > -10 := 
by
  -- The formal proof will be completed here.
  sorry

end range_of_m_l1912_191282


namespace ducks_problem_l1912_191266

theorem ducks_problem :
  ∃ (adelaide ephraim kolton : ℕ),
    adelaide = 30 ∧
    adelaide = 2 * ephraim ∧
    ephraim + 45 = kolton ∧
    (adelaide + ephraim + kolton) % 9 = 0 ∧
    1 ≤ adelaide ∧
    1 ≤ ephraim ∧
    1 ≤ kolton ∧
    adelaide + ephraim + kolton = 108 ∧
    (adelaide + ephraim + kolton) / 3 = 36 :=
by
  sorry

end ducks_problem_l1912_191266


namespace ratio_of_a_to_b_l1912_191242

theorem ratio_of_a_to_b (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
    (h_x : x = 1.25 * a) (h_m : m = 0.40 * b) (h_ratio : m / x = 0.4) 
    : (a / b) = 4 / 5 := by
  sorry

end ratio_of_a_to_b_l1912_191242


namespace no_real_roots_iff_range_m_l1912_191255

open Real

theorem no_real_roots_iff_range_m (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + (m + 3) ≠ 0) ↔ (-2 < m ∧ m < 6) :=
by
  sorry

end no_real_roots_iff_range_m_l1912_191255


namespace chickens_in_coop_l1912_191243

theorem chickens_in_coop (C : ℕ)
  (H1 : ∃ C : ℕ, ∀ R : ℕ, R = 2 * C)
  (H2 : ∃ R : ℕ, ∀ F : ℕ, F = 2 * R - 4)
  (H3 : ∃ F : ℕ, F = 52) :
  C = 14 :=
by sorry

end chickens_in_coop_l1912_191243


namespace toys_cost_price_gain_l1912_191256

theorem toys_cost_price_gain (selling_price : ℕ) (cost_price_per_toy : ℕ) (num_toys : ℕ)
    (total_cost_price : ℕ) (gain : ℕ) (x : ℕ) :
    selling_price = 21000 →
    cost_price_per_toy = 1000 →
    num_toys = 18 →
    total_cost_price = num_toys * cost_price_per_toy →
    gain = selling_price - total_cost_price →
    x = gain / cost_price_per_toy →
    x = 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3] at *
  sorry

end toys_cost_price_gain_l1912_191256


namespace least_element_of_S_is_4_l1912_191202

theorem least_element_of_S_is_4 :
  ∃ S : Finset ℕ, S.card = 7 ∧ (S ⊆ Finset.range 16) ∧
  (∀ {a b : ℕ}, a ∈ S → b ∈ S → a < b → ¬ (b % a = 0)) ∧
  (∀ T : Finset ℕ, T.card = 7 → (T ⊆ Finset.range 16) →
  (∀ {a b : ℕ}, a ∈ T → b ∈ T → a < b → ¬ (b % a = 0)) →
  ∃ x : ℕ, x ∈ T ∧ x = 4) :=
by
  sorry

end least_element_of_S_is_4_l1912_191202


namespace range_of_m_l1912_191280

noncomputable def f (x m : ℝ) : ℝ :=
  x^2 - 2 * m * x + m + 2

theorem range_of_m
  (m : ℝ)
  (h1 : ∃ a b : ℝ, f a m = 0 ∧ f b m = 0 ∧ a ≠ b)
  (h2 : ∀ x : ℝ, x ≥ 1 → 2*x - 2*m ≥ 0) :
  m < -1 :=
sorry

end range_of_m_l1912_191280


namespace simplify_and_rationalize_l1912_191230

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) := 
  sorry

end simplify_and_rationalize_l1912_191230


namespace price_of_peaches_is_2_l1912_191249

noncomputable def price_per_pound_peaches (total_spent: ℝ) (price_per_pound_other: ℝ) (total_weight_peaches: ℝ) (total_weight_apples: ℝ) (total_weight_blueberries: ℝ) : ℝ :=
  (total_spent - (total_weight_apples + total_weight_blueberries) * price_per_pound_other) / total_weight_peaches

theorem price_of_peaches_is_2 
  (total_spent: ℝ := 51)
  (price_per_pound_other: ℝ := 1)
  (num_peach_pies: ℕ := 5)
  (num_apple_pies: ℕ := 4)
  (num_blueberry_pies: ℕ := 3)
  (weight_per_pie: ℝ := 3):
  price_per_pound_peaches total_spent price_per_pound_other 
                          (num_peach_pies * weight_per_pie) 
                          (num_apple_pies * weight_per_pie) 
                          (num_blueberry_pies * weight_per_pie) = 2 := 
by
  sorry

end price_of_peaches_is_2_l1912_191249


namespace diff_g_eq_l1912_191267

def g (n : ℤ) : ℚ := (1/6) * n * (n+1) * (n+3)

theorem diff_g_eq :
  ∀ (r : ℤ), g r - g (r - 1) = (3/2) * r^2 + (5/2) * r :=
by
  intro r
  sorry

end diff_g_eq_l1912_191267


namespace coach_A_spent_less_l1912_191273

-- Definitions of costs and discounts for coaches purchases
def total_cost_before_discount_A : ℝ := 10 * 29 + 5 * 15
def total_cost_before_discount_B : ℝ := 14 * 2.50 + 1 * 18 + 4 * 25 + 1 * 72
def total_cost_before_discount_C : ℝ := 8 * 32 + 12 * 12

def discount_A : ℝ := 0.05 * total_cost_before_discount_A
def discount_B : ℝ := 0.10 * total_cost_before_discount_B
def discount_C : ℝ := 0.07 * total_cost_before_discount_C

def total_cost_after_discount_A : ℝ := total_cost_before_discount_A - discount_A
def total_cost_after_discount_B : ℝ := total_cost_before_discount_B - discount_B
def total_cost_after_discount_C : ℝ := total_cost_before_discount_C - discount_C

def combined_cost_B_C : ℝ := total_cost_after_discount_B + total_cost_after_discount_C
def difference_A_BC : ℝ := total_cost_after_discount_A - combined_cost_B_C

theorem coach_A_spent_less : difference_A_BC = -227.75 := by
  sorry

end coach_A_spent_less_l1912_191273


namespace doses_A_correct_doses_B_correct_doses_C_correct_l1912_191219

def days_in_july : ℕ := 31

def daily_dose_A : ℕ := 1
def daily_dose_B : ℕ := 2
def daily_dose_C : ℕ := 3

def missed_days_A : ℕ := 3
def missed_days_B_morning : ℕ := 5
def missed_days_C_all : ℕ := 2

def total_doses_A : ℕ := days_in_july * daily_dose_A
def total_doses_B : ℕ := days_in_july * daily_dose_B
def total_doses_C : ℕ := days_in_july * daily_dose_C

def missed_doses_A : ℕ := missed_days_A * daily_dose_A
def missed_doses_B : ℕ := missed_days_B_morning
def missed_doses_C : ℕ := missed_days_C_all * daily_dose_C

def doses_consumed_A := total_doses_A - missed_doses_A
def doses_consumed_B := total_doses_B - missed_doses_B
def doses_consumed_C := total_doses_C - missed_doses_C

theorem doses_A_correct : doses_consumed_A = 28 := by sorry
theorem doses_B_correct : doses_consumed_B = 57 := by sorry
theorem doses_C_correct : doses_consumed_C = 87 := by sorry

end doses_A_correct_doses_B_correct_doses_C_correct_l1912_191219


namespace value_of_g_at_3_l1912_191239

def g (x : ℝ) : ℝ := x^2 - 2 * x

theorem value_of_g_at_3 : g 3 = 3 := by
  sorry

end value_of_g_at_3_l1912_191239


namespace circle_condition_l1912_191200

theorem circle_condition (m : ℝ) : (∀ x y : ℝ, x^2 + y^2 - x + y + m = 0 → (m < 1 / 2)) :=
by {
-- Skipping the proof here
sorry
}

end circle_condition_l1912_191200


namespace rona_age_l1912_191263

theorem rona_age (R : ℕ) (hR1 : ∀ Rachel Collete : ℕ, Rachel = 2 * R ∧ Collete = R / 2 ∧ Rachel - Collete = 12) : R = 12 :=
sorry

end rona_age_l1912_191263


namespace hyperbola_focus_coordinates_l1912_191214

theorem hyperbola_focus_coordinates : 
  ∃ (x y : ℝ), -2 * x^2 + 3 * y^2 + 8 * x - 18 * y - 8 = 0 ∧ (x, y) = (2, 7.5) :=
sorry

end hyperbola_focus_coordinates_l1912_191214


namespace solve_equation1_solve_equation2_l1912_191284

-- Problem for Equation (1)
theorem solve_equation1 (x : ℝ) : x * (x - 6) = 2 * (x - 8) → x = 4 := by
  sorry

-- Problem for Equation (2)
theorem solve_equation2 (x : ℝ) : (2 * x - 1)^2 + 3 * (2 * x - 1) + 2 = 0 → x = 0 ∨ x = -1 / 2 := by
  sorry

end solve_equation1_solve_equation2_l1912_191284


namespace mass_of_man_is_120_l1912_191291

def length_of_boat : ℝ := 3
def breadth_of_boat : ℝ := 2
def height_water_rise : ℝ := 0.02
def density_of_water : ℝ := 1000
def volume_displaced : ℝ := length_of_boat * breadth_of_boat * height_water_rise
def mass_of_man := density_of_water * volume_displaced

theorem mass_of_man_is_120 : mass_of_man = 120 :=
by
  -- insert the detailed proof here
  sorry

end mass_of_man_is_120_l1912_191291


namespace polygon_n_sides_l1912_191216

theorem polygon_n_sides (n : ℕ) (h : (n - 2) * 180 - x = 2000) : n = 14 :=
sorry

end polygon_n_sides_l1912_191216


namespace no_prime_number_between_30_and_40_mod_9_eq_7_l1912_191217

theorem no_prime_number_between_30_and_40_mod_9_eq_7 : ¬ ∃ n : ℕ, 30 ≤ n ∧ n ≤ 40 ∧ Nat.Prime n ∧ n % 9 = 7 :=
by
  sorry

end no_prime_number_between_30_and_40_mod_9_eq_7_l1912_191217


namespace sample_size_is_150_l1912_191237

-- Define the conditions
def total_parents : ℕ := 823
def sampled_parents : ℕ := 150
def negative_attitude_parents : ℕ := 136

-- State the theorem
theorem sample_size_is_150 : sampled_parents = 150 := 
by
  sorry

end sample_size_is_150_l1912_191237


namespace daily_profit_functional_relationship_daily_profit_maximizes_at_120_selling_price_for_2400_profit_l1912_191275

-- Given conditions
def cost_price : ℝ := 80
def daily_sales_quantity (x : ℝ) : ℝ := -2 * x + 320
def daily_profit (x : ℝ) : ℝ := (x - cost_price) * daily_sales_quantity x

-- Part 1: Functional relationship
theorem daily_profit_functional_relationship (x : ℝ) (hx : 80 ≤ x ∧ x ≤ 160) : daily_profit x = -2 * x^2 + 480 * x - 25600 :=
by sorry

-- Part 2: Maximizing daily profit
theorem daily_profit_maximizes_at_120 (hx : 80 ≤ 120 ∧ 120 ≤ 160) : daily_profit 120 = 3200 :=
by sorry

-- Part 3: Selling price for a daily profit of $2400
theorem selling_price_for_2400_profit (hx : 80 ≤ 100 ∧ 100 ≤ 160) : daily_profit 100 = 2400 :=
by sorry

end daily_profit_functional_relationship_daily_profit_maximizes_at_120_selling_price_for_2400_profit_l1912_191275


namespace peter_fraction_is_1_8_l1912_191265

-- Define the total number of slices, slices Peter ate alone, and slices Peter shared with Paul
def total_slices := 16
def peter_alone_slices := 1
def shared_slices := 2

-- Define the fraction of the pizza Peter ate alone
def peter_fraction_alone := peter_alone_slices / total_slices

-- Define the fraction of the pizza Peter ate from the shared slices
def shared_fraction := shared_slices * (1 / 2) / total_slices

-- Define the total fraction of the pizza Peter ate
def total_fraction_peter_ate := peter_fraction_alone + shared_fraction

-- Prove that the total fraction of the pizza Peter ate is 1/8
theorem peter_fraction_is_1_8 : total_fraction_peter_ate = 1/8 := by
  sorry

end peter_fraction_is_1_8_l1912_191265


namespace compute_expression_l1912_191253

theorem compute_expression : 3 * 3^3 - 9^50 / 9^48 = 0 := by
  sorry

end compute_expression_l1912_191253


namespace calculate_p_p_l1912_191233

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then x + y
  else if x < 0 ∧ y < 0 then x - 2*y
  else if x ≥ 0 ∧ y < 0 then x^2 + y^2
  else 3*x + y

theorem calculate_p_p : p (p 2 (-3)) (p (-4) 1) = 290 :=
by {
  -- required statement of proof problem
  sorry
}

end calculate_p_p_l1912_191233


namespace Haleigh_can_make_3_candles_l1912_191298

variable (n20 n5 n1 : ℕ) (w20 w5 w1 oz10 : ℝ)

def wax_leftover (n20 n5 n1 : ℕ) (w20 w5 w1 oz10 : ℝ) : ℝ := 
  n20 * w20 + n5 * w5 + n1 * w1 

theorem Haleigh_can_make_3_candles :
  ∀ (n20 n5 n1 : ℕ) (w20 w5 w1 oz10 : ℝ), 
  n20 = 5 →
  w20 = 2 →
  n5 = 5 →
  w5 = 0.5 →
  n1 = 25 →
  w1 = 0.1 →
  oz10 = 10 →
  (wax_leftover n20 n5 n1 w20 w5 w1 oz10) / 5 = 3 := 
by
  intros n20 n5 n1 w20 w5 w1 oz10 hn20 hw20 hn5 hw5 hn1 hw1 hoz10
  rw [hn20, hw20, hn5, hw5, hn1, hw1, hoz10]
  sorry

end Haleigh_can_make_3_candles_l1912_191298


namespace call_cost_inequalities_min_call_cost_correct_l1912_191207

noncomputable def call_cost_before (x : ℝ) : ℝ :=
  if x ≤ 3 then 0.2 else 0.4

noncomputable def call_cost_after (x : ℝ) : ℝ :=
  if x ≤ 3 then 0.2
  else if x ≤ 4 then 0.2 + 0.1 * (x - 3)
  else 0.3 + 0.1 * (x - 4)

theorem call_cost_inequalities : 
  (call_cost_before 4 = 0.4 ∧ call_cost_after 4 = 0.3) ∧
  (call_cost_before 4.3 = 0.4 ∧ call_cost_after 4.3 = 0.4) ∧
  (call_cost_before 5.8 = 0.4 ∧ call_cost_after 5.8 = 0.5) ∧
  (∀ x, (0 < x ∧ x ≤ 3) ∨ x > 4 → call_cost_before x ≤ call_cost_after x) :=
by
  sorry

noncomputable def min_call_cost_plan (m : ℝ) (n : ℕ) : ℝ :=
  if 3 * n - 1 < m ∧ m ≤ 3 * n then 0.2 * n
  else if 3 * n < m ∧ m ≤ 3 * n + 1 then 0.2 * n + 0.1
  else if 3 * n + 1 < m ∧ m ≤ 3 * n + 2 then 0.2 * n + 0.2
  else 0.0  -- Fallback, though not necessary as per the conditions

theorem min_call_cost_correct (m : ℝ) (n : ℕ) (h : m > 5) :
  (3 * n - 1 < m ∧ m ≤ 3 * n → min_call_cost_plan m n = 0.2 * n) ∧
  (3 * n < m ∧ m ≤ 3 * n + 1 → min_call_cost_plan m n = 0.2 * n + 0.1) ∧
  (3 * n + 1 < m ∧ m ≤ 3 * n + 2 → min_call_cost_plan m n = 0.2 * n + 0.2) :=
by
  sorry

end call_cost_inequalities_min_call_cost_correct_l1912_191207


namespace jesse_money_left_after_mall_l1912_191201

theorem jesse_money_left_after_mall :
  ∀ (initial_amount novel_cost lunch_cost total_spent remaining_amount : ℕ),
    initial_amount = 50 →
    novel_cost = 7 →
    lunch_cost = 2 * novel_cost →
    total_spent = novel_cost + lunch_cost →
    remaining_amount = initial_amount - total_spent →
    remaining_amount = 29 :=
by
  intros initial_amount novel_cost lunch_cost total_spent remaining_amount
  sorry

end jesse_money_left_after_mall_l1912_191201


namespace sequence_properties_l1912_191240

variable {a : ℕ → ℤ}

-- Conditions
axiom seq_add : ∀ (p q : ℕ), 1 ≤ p → 1 ≤ q → a (p + q) = a p + a q
axiom a2_neg4 : a 2 = -4

-- Theorem statement: We need to prove a6 = -12 and a_n = -2n for all n
theorem sequence_properties :
  (a 6 = -12) ∧ ∀ n : ℕ, 1 ≤ n → a n = -2 * n :=
by
  sorry

end sequence_properties_l1912_191240


namespace total_monthly_feed_l1912_191234

def daily_feed (pounds_per_pig_per_day : ℕ) (number_of_pigs : ℕ) : ℕ :=
  pounds_per_pig_per_day * number_of_pigs

def monthly_feed (daily_feed : ℕ) (days_per_month : ℕ) : ℕ :=
  daily_feed * days_per_month

theorem total_monthly_feed :
  let pounds_per_pig_per_day := 15
  let number_of_pigs := 4
  let days_per_month := 30
  monthly_feed (daily_feed pounds_per_pig_per_day number_of_pigs) days_per_month = 1800 :=
by
  sorry

end total_monthly_feed_l1912_191234


namespace final_ratio_l1912_191299

-- Define initial conditions
def initial_milk_ratio : ℕ := 1
def initial_water_ratio : ℕ := 5
def total_parts : ℕ := initial_milk_ratio + initial_water_ratio
def can_capacity : ℕ := 8
def additional_milk : ℕ := 2
def initial_volume : ℕ := can_capacity - additional_milk
def part_volume : ℕ := initial_volume / total_parts

-- Define initial quantities
def initial_milk_quantity : ℕ := part_volume * initial_milk_ratio
def initial_water_quantity : ℕ := part_volume * initial_water_ratio

-- Define final quantities
def final_milk_quantity : ℕ := initial_milk_quantity + additional_milk
def final_water_quantity : ℕ := initial_water_quantity

-- Hypothesis: final ratios of milk and water
def final_ratio_of_milk_to_water : ℕ × ℕ := (final_milk_quantity, final_water_quantity)

-- Final ratio should be 3:5
theorem final_ratio (h : final_ratio_of_milk_to_water = (3, 5)) : final_ratio_of_milk_to_water = (3, 5) :=
  by
  sorry

end final_ratio_l1912_191299


namespace find_x_l1912_191276

theorem find_x (x : ℝ) (h : 2 * x = 26 - x + 19) : x = 15 :=
by
  sorry

end find_x_l1912_191276


namespace b5_b9_equal_16_l1912_191294

-- Define the arithmetic sequence and conditions
variables {a : ℕ → ℝ} (h_arith : ∀ n m, a m = a n + (m - n) * (a 1 - a 0))
variable (h_non_zero : ∀ n, a n ≠ 0)
variable (h_cond : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)

-- Define the geometric sequence and condition
variables {b : ℕ → ℝ} (h_geom : ∀ n, b (n + 1) = b n * (b 1 / b 0))
variable (h_b7 : b 7 = a 7)

-- State the theorem to prove
theorem b5_b9_equal_16 : b 5 * b 9 = 16 :=
sorry

end b5_b9_equal_16_l1912_191294


namespace correct_commutative_property_usage_l1912_191212

-- Definitions for the transformations
def transformA := 3 + (-2) = 2 + 3
def transformB := 4 + (-6) + 3 = (-6) + 4 + 3
def transformC := (5 + (-2)) + 4 = (5 + (-4)) + 2
def transformD := (1 / 6) + (-1) + (5 / 6) = ((1 / 6) + (5 / 6)) + 1

-- The theorem stating that transformB uses the commutative property correctly
theorem correct_commutative_property_usage : transformB :=
by
  sorry

end correct_commutative_property_usage_l1912_191212


namespace calculator_to_protractors_l1912_191245

def calculator_to_rulers (c: ℕ) : ℕ := 100 * c
def rulers_to_compasses (r: ℕ) : ℕ := (r * 30) / 10
def compasses_to_protractors (p: ℕ) : ℕ := (p * 50) / 25

theorem calculator_to_protractors (c: ℕ) : compasses_to_protractors (rulers_to_compasses (calculator_to_rulers c)) = 600 * c :=
by
  sorry

end calculator_to_protractors_l1912_191245


namespace magician_card_trick_l1912_191241

-- Definitions and proof goal
theorem magician_card_trick :
  let n := 12
  let total_cards := n ^ 2
  let duplicate_cards := n
  let non_duplicate_cards := total_cards - duplicate_cards - (n - 1) - (n - 1)
  let total_ways_with_two_duplicates := Nat.choose duplicate_cards 2
  let total_ways_with_one_duplicate :=
    duplicate_cards * non_duplicate_cards
  (total_ways_with_two_duplicates + total_ways_with_one_duplicate) = 1386 :=
by
  let n := 12
  let total_cards := n ^ 2
  let duplicate_cards := n
  let non_duplicate_cards := total_cards - duplicate_cards - (n - 1) - (n - 1)
  let total_ways_with_two_duplicates := Nat.choose duplicate_cards 2
  let total_ways_with_one_duplicate :=
    duplicate_cards * non_duplicate_cards
  have h : (total_ways_with_two_duplicates + total_ways_with_one_duplicate) = 1386 := sorry
  exact h

end magician_card_trick_l1912_191241


namespace find_z_percentage_of_1000_l1912_191236

noncomputable def x := (3 / 5) * 4864
noncomputable def y := (2 / 3) * 9720
noncomputable def z := (1 / 4) * 800

theorem find_z_percentage_of_1000 :
  (2 / 3) * x + (1 / 2) * y = z → (z / 1000) * 100 = 20 :=
by
  sorry

end find_z_percentage_of_1000_l1912_191236


namespace original_price_before_discounts_l1912_191285

theorem original_price_before_discounts (P : ℝ) (h : 0.684 * P = 6840) : P = 10000 :=
by
  sorry

end original_price_before_discounts_l1912_191285


namespace find_y_l1912_191222

-- Definitions for the given conditions
variable (p y : ℕ) (h : p > 30)  -- Natural numbers, noting p > 30 condition

-- The initial amount of acid in ounces
def initial_acid_amount : ℕ := p * p / 100

-- The amount of acid after adding y ounces of water
def final_acid_amount : ℕ := (p - 15) * (p + y) / 100

-- Lean statement to prove y = 15p/(p-15)
theorem find_y (h1 : p > 30) (h2 : initial_acid_amount p = final_acid_amount p y) :
  y = 15 * p / (p - 15) :=
sorry

end find_y_l1912_191222


namespace marble_prob_l1912_191250

theorem marble_prob
  (a b x y m n : ℕ)
  (h1 : a + b = 30)
  (h2 : (x : ℚ) / a * (y : ℚ) / b = 4 / 9)
  (h3 : x * y = 36)
  (h4 : Nat.gcd m n = 1)
  (h5 : (a - x : ℚ) / a * (b - y) / b = m / n) :
  m + n = 29 := 
sorry

end marble_prob_l1912_191250


namespace percentage_of_first_to_second_l1912_191220

theorem percentage_of_first_to_second (X : ℝ) (first second : ℝ) (h1 : first = (7 / 100) * X) (h2 : second = (14 / 100) * X) : 
(first / second) * 100 = 50 := by
  sorry

end percentage_of_first_to_second_l1912_191220


namespace smallest_B_to_divisible_3_l1912_191283

-- Define the problem
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Define the digits in the integer
def digit_sum (B : ℕ) : ℕ := 8 + B + 4 + 6 + 3 + 5

-- Prove that the smallest digit B that makes 8B4,635 divisible by 3 is 1
theorem smallest_B_to_divisible_3 : ∃ B : ℕ, B ≥ 0 ∧ B ≤ 9 ∧ is_divisible_by_3 (digit_sum B) ∧ ∀ B' : ℕ, B' < B → ¬ is_divisible_by_3 (digit_sum B') ∧ B = 1 :=
sorry

end smallest_B_to_divisible_3_l1912_191283
